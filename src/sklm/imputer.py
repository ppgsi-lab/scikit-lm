"""Language-model imputer.

Fits on the data as-is (missing cells are simply dropped from each row's
serialization), then fills every NaN by conditioning on that row's observed
cells and generating the missing ones. A row whose missing cells stay malformed
raises rather than falling back to a baseline, so a model that cannot generate
valid values never masquerades as a working imputer.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Self, Unpack, override

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils import Tags
from sklearn.utils.validation import check_is_fitted

from .base import (
    align_features,
    forget,
    make_tabular_lm,
    records,
    reduce_estimate,
    resolve_discretization,
    select_candidates,
    to_frame,
)
from .callbacks import predict_batches
from .config import DiscretizationConfig, GenerationConfig
from .core import _ScoreSpec
from .params import AnnotatedDefault, ImputerArgs, _FlatParams
from .serialize import is_missing

__all__ = ["LanguageModelImputer"]


class LanguageModelImputer(_FlatParams, OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Fill missing values by conditioning a language model on observed cells.

    Parameters
    ----------
    model : str, optional
        Hugging Face model id. Default ``"distilgpt2"``.
    backend : LanguageModelBackend or str, optional
        ``"huggingface"`` (default) builds a fresh :class:`~sklm.HFBackend` per
        fit, or pass an :class:`~sklm.LanguageModelBackend` instance.
    training : TrainingConfig, optional
        Fine-tuning hyperparameters.
    generation : GenerationConfig, optional
        Sampling hyperparameters (temperature, token budget) plus ``n_samples``:
        each missing cell is generated ``n_samples`` times and the draws are
        aggregated (mean for numeric columns, mode otherwise). Default ``1`` draws
        once per cell.
    discretization : DiscretizationConfig or Mapping[str, DiscretizationConfig], optional
        Fill missing numeric cells by scoring candidate observed values by
        likelihood (then reducing the distribution) instead of generating them.
        A single :class:`~sklm.DiscretizationConfig` applies to every numeric
        column; a mapping is per-column (columns absent from it are generated).
        Categorical cells always generate. Default off (a
        :class:`~sklm.DiscretizationConfig` with ``bins=0``), keeping the
        all-generative path. When on, each row is filled cell by cell -- scored
        cells are deterministic, generated cells draw and aggregate
        ``generation.n_samples`` per cell.
    serializer : str or Serializer, optional
        ``"json"`` or a custom :class:`~sklm.Serializer`. Default ``"json"``.
    max_decimals : int or None, optional
        Round numeric cells to at most this many decimal places when
        serializing. Applies only to the string ``serializer`` selectors; a
        :class:`~sklm.Serializer` instance keeps its own number format.
        Default ``3``.
    random_state : int or None, optional
        Seed forwarded to the backend and serializer.
    callback : Callback, list of Callback or None, optional
        Feedback hooks for fitting and inference. A list is wrapped in a
        :class:`~sklm.CompositeCallback`. Default ``None`` auto-selects a
        dashboard for the runtime environment (Jupyter, rich, or
        logging).
    lora : LoRAConfig or None, optional
        Fine-tune with LoRA adapters when set; full-weight otherwise (default).
    quantization : {"2bit", "3bit", "4bit", "6bit", "8bit"}, QuantizationConfig or None, optional
        Quantize the base weights. A ``"<n>bit"`` string is shorthand for
        ``QuantizationConfig(bits=n)``; pass a ``QuantizationConfig`` to also pick
        the library (``method``) or ``group_size``. ``None`` (default) loads at
        ``precision``. MLX does 2/3/4/6/8-bit; HF does 4-/8-bit (bitsandbytes) and
        2-/3-bit (HQQ).
    precision : {"fp32", "bf16", "fp16"}, optional
        Compute dtype for the weights and the train/generate autocast. Default
        ``"fp32"``.
    tokenizer : str or None, optional
        Tokenizer id/path; ``None`` (default) derives it from the model.
    trust_remote_code : bool, optional
        Allow custom model/tokenizer code from the hub. Default ``False``.
    device : str, optional
        Target device (``"cuda"``/``"mps"``/``"cpu"``) or ``"auto"`` (default).
    attn_implementation : str or None, optional
        Attention kernel passed to ``from_pretrained``. ``None`` (default) keeps
        the model default.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    feature_names_in_ : numpy.ndarray
        Feature names, set only when ``X`` is a DataFrame.
    """

    _args = ImputerArgs

    generation: GenerationConfig
    discretization: DiscretizationConfig | Mapping[str, DiscretizationConfig]

    def __init__(self, model: str = "distilgpt2", **kwargs: Unpack[ImputerArgs]) -> None:
        self.model = model
        for key, value in AnnotatedDefault.create_with_defaults(ImputerArgs, **kwargs).items():
            setattr(self, key, value)

    def fit(self, X: object, y: object = None) -> Self:
        """Fine-tune the backend on ``X``.

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape (n_samples, n_features)
            Feature table, possibly containing NaNs.
        y : ignored
            Present for API compatibility.

        Returns
        -------
        Self
            The fitted estimator.

        Raises
        ------
        ValueError
            If ``X`` is not 2-dimensional, if ``backend``/``serializer`` is an
            unknown string selector, or if a per-column ``discretization``
            mapping references an unknown or non-numeric column.
        """
        X_df = to_frame(X)
        self.n_features_in_ = X_df.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        else:
            forget(self, "feature_names_in_")
        X_df = X_df.set_axis([str(c) for c in X_df.columns], axis=1)
        self.feature_cols_ = [str(c) for c in X_df.columns]
        numeric_cols = [str(c) for c in X_df.select_dtypes(include="number").columns]
        self._validate_discretization(numeric_cols)
        self.column_values_ = {c: X_df[c].to_numpy() for c in numeric_cols}
        target_cols = frozenset(c for c in self.feature_cols_ if bool(X_df[c].isna().any()))
        self.lm_ = make_tabular_lm(self).fit(X_df, target_cols=target_cols)
        return self

    def _validate_discretization(self, numeric_cols: list[str]) -> None:
        """Reject per-column ``discretization`` keys that cannot be scored."""
        if isinstance(self.discretization, DiscretizationConfig):
            return
        known, numeric = set(self.feature_cols_), set(numeric_cols)
        for col in self.discretization:
            if col not in known:
                raise ValueError(f"discretization references unknown column {col!r}")
            if col not in numeric:
                raise ValueError(
                    f"discretization can only score numeric columns; {col!r} is not numeric"
                )

    def transform(self, X: object) -> pd.DataFrame | np.ndarray:
        """Return ``X`` with every NaN filled.

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape (n_samples, n_features)
            Feature table to impute.

        Returns
        -------
        pandas.DataFrame or numpy.ndarray
            Same shape and row order as ``X`` (DataFrame in, DataFrame out).

        Raises
        ------
        ValueError
            If ``X``'s feature count does not match what was seen at fit.
        RuntimeError
            If a row's *generated* missing cells stay malformed after retries
            (the model produced no valid value). Scored cells always yield a
            value, so they never trigger this.
        """
        check_is_fitted(self)
        original_columns = to_frame(X).columns
        work = align_features(self, X)
        out = work.copy()
        cb = self.lm_.callback
        rows = records(work)
        knowns = [{c: row[c] for c in self.feature_cols_ if not is_missing(row[c])} for row in rows]
        targets = [[c for c in self.feature_cols_ if is_missing(row[c])] for row in rows]
        batch_size = self.generation.inference_batch_size or self.training.batch_size
        scored = resolve_discretization(self.discretization, self.lm_.numeric_cols_)
        score = {
            col: _ScoreSpec(
                select_candidates(self.column_values_[col], cfg),
                partial(reduce_estimate, config=cfg),
            )
            for col, cfg in scored.items()
        }
        for start, stop in predict_batches(cb, len(rows), batch_size):
            if score:
                filled = self.lm_.impute_many(
                    knowns[start:stop], targets[start:stop], self.generation, score=score
                )
            else:
                filled = self.lm_.sample_aggregate_many(
                    knowns[start:stop], targets[start:stop], self.generation
                )
            for j, i in enumerate(range(start, stop)):
                row = filled[j]
                if row is None:
                    raise RuntimeError(
                        f"generation failed for column(s) {targets[i]} at row {i} after "
                        f"{self.lm_.max_retries} retries; the model is not producing valid values"
                    )
                for col in targets[i]:
                    out.iloc[i, out.columns.get_loc(col)] = row[col]
        if not isinstance(X, pd.DataFrame):
            return out.to_numpy()
        # align_features may have reordered columns into the training order; relabel
        # from the serialization names back to the fit names, then restore the
        # caller's own column order so DataFrame-in yields the same layout out.
        names = getattr(self, "feature_names_in_", None)
        if names is not None:
            return out.set_axis(names, axis=1).reindex(columns=original_columns)
        return out.set_axis(original_columns, axis=1)

    @override
    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.string = True
        tags.input_tags.categorical = True
        tags.input_tags.allow_nan = True
        return tags
