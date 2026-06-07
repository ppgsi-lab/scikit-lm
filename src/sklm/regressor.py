"""Language-model regressor.

Conditions on all features and generates the numeric target. Because greedy
decoding returns the mode rather than the mean, :meth:`predict` draws
``generation.n_samples`` completions per row and averages them (a Monte-Carlo
estimate of the conditional mean); the default ``n_samples == 1`` draws a single
value, so raise it to average. A row whose every draw is malformed raises rather
than falling back to a baseline, so a model that cannot generate valid values
never masquerades as a working regressor.
"""

from __future__ import annotations

from typing import Self, Unpack, cast, override

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import Tags
from sklearn.utils.validation import check_is_fitted, column_or_1d

from .base import (
    align_features,
    forget,
    make_tabular_lm,
    records,
    reduce_estimate,
    select_candidates,
    to_frame,
    unique_name,
)
from .callbacks import predict_batches
from .config import DiscretizationConfig, GenerationConfig
from .params import AnnotatedDefault, RegressorArgs, _FlatParams
from .serialize import is_missing

__all__ = ["LanguageModelRegressor"]


class LanguageModelRegressor(_FlatParams, RegressorMixin, BaseEstimator):
    """Predict a continuous target by averaging language-model samples.

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
        Sampling hyperparameters (temperature, token budget) plus ``n_samples``,
        the number of completions drawn and averaged per row at predict time
        (default ``1``).
    discretization : DiscretizationConfig, optional
        When enabled (``bins`` non-zero), predict by scoring a candidate set of
        observed target values by likelihood and reducing the distribution,
        instead of generating and averaging. Default off (``bins=0``), keeping
        the generative path.
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

    _args = RegressorArgs

    generation: GenerationConfig
    discretization: DiscretizationConfig

    def __init__(self, model: str = "distilgpt2", **kwargs: Unpack[RegressorArgs]) -> None:
        self.model = model
        for key, value in AnnotatedDefault.create_with_defaults(RegressorArgs, **kwargs).items():
            setattr(self, key, value)

    def fit(self, X: object, y: object) -> Self:
        """Fine-tune the backend on ``X`` with the numeric target appended.

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape (n_samples, n_features)
            Feature table.
        y : array-like of shape (n_samples,)
            Continuous target.

        Returns
        -------
        Self
            The fitted estimator.

        Raises
        ------
        ValueError
            If ``X`` is not 2-dimensional, if ``backend``/``serializer`` is an
            unknown string selector, or if ``y`` contains NaN or infinite values
            (the target supervises training and cannot be missing).
        """
        X_df = to_frame(X)
        y_arr = np.asarray(column_or_1d(y, warn=True), dtype=float)
        if not np.isfinite(y_arr).all():
            raise ValueError("Input y contains NaN or infinite values; y must be finite")
        self.n_features_in_ = X_df.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        else:
            forget(self, "feature_names_in_")
        X_df = X_df.set_axis([str(c) for c in X_df.columns], axis=1)
        self.feature_cols_ = [str(c) for c in X_df.columns]
        self.target_col_ = unique_name("target", self.feature_cols_)
        self.target_values_ = y_arr
        frame = X_df.copy()
        frame[self.target_col_] = y_arr
        self.lm_ = make_tabular_lm(self).fit(frame, target_cols=frozenset({self.target_col_}))
        return self

    def predict(
        self,
        X: object,
        *,
        discretization: DiscretizationConfig | None = None,
        generation: GenerationConfig | None = None,
    ) -> np.ndarray:
        """Predict the continuous target for each row.

        With ``discretization`` off (default) the target is the mean of
        ``generation.n_samples`` generated values; with it on, candidate observed
        values are scored by likelihood and the distribution is reduced
        (:func:`~sklm.base.reduce_estimate`).

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape (n_samples, n_features)
            Feature table.
        discretization : DiscretizationConfig, optional
            Per-call override of the estimator's ``discretization``. ``None``
            (default) uses the fitted estimator's attribute; passing a config
            switches decoders for this call only, without mutating the estimator.
        generation : GenerationConfig, optional
            Per-call override of the estimator's ``generation``, with the same
            fall-back-to-attribute semantics as ``discretization``.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            One finite prediction per row.

        Raises
        ------
        ValueError
            If ``X``'s feature count does not match what was seen at fit.
        RuntimeError
            Generative path only: if every one of a row's ``generation.n_samples``
            draws is malformed (the model produced no valid value). The scoring
            path always yields a distribution, so it never raises this.
        """
        check_is_fitted(self)
        discretization = self.discretization if discretization is None else discretization
        generation = self.generation if generation is None else generation
        rows = records(align_features(self, X))
        cb = self.lm_.callback
        knowns = [{c: v for c, v in row.items() if not is_missing(v)} for row in rows]
        batch_size = generation.inference_batch_size or self.training.batch_size
        preds = np.empty(len(rows))
        if discretization.bins:
            candidates = select_candidates(self.target_values_, discretization)
            for start, stop in predict_batches(cb, len(rows), batch_size):
                proba = self.lm_.predict_proba_many(
                    knowns[start:stop], self.target_col_, candidates, generation
                )
                for j in range(proba.shape[0]):
                    preds[start + j] = reduce_estimate(proba[j], candidates, discretization)
            return preds
        for start, stop in predict_batches(cb, len(rows), batch_size):
            chunk = knowns[start:stop]
            outs = self.lm_.sample_aggregate_many(
                chunk, [[self.target_col_]] * len(chunk), generation
            )
            for j, out in enumerate(outs):
                if out is None:
                    raise RuntimeError(
                        f"all {generation.n_samples} generated samples for row "
                        f"{start + j} were malformed after {self.lm_.max_retries} retries "
                        "each; the model is not producing valid numeric values"
                    )
                # aggregate_default returns the mean as a float for numeric targets
                preds[start + j] = cast(float, out[self.target_col_])
        return preds

    @override
    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.string = True
        tags.input_tags.categorical = True
        tags.input_tags.allow_nan = True
        return tags
