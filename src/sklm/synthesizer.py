"""Language-model tabular synthesizer.

The four supervised estimators each fix which columns of the shared fine-tune go
in the prompt. Drop that constraint and the same :class:`~sklm.TabularLanguageModel`
becomes a generator of whole rows: fit on a table with no fixed target, then draw
new rows from the learned joint distribution, optionally conditioning on columns
held fixed. This estimator is a thin facade over the core -- it only exposes the
flat constructor knobs the other estimators share and forwards ``fit`` / ``sample``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Self, Unpack, override

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils import Tags
from sklearn.utils.validation import check_is_fitted

from .base import forget, make_tabular_lm, to_frame
from .bridge import Model
from .config import GenerationConfig
from .params import AnnotatedDefault, EstimatorArgs, _FlatParams

__all__ = ["LanguageModelSynthesizer"]


class LanguageModelSynthesizer(_FlatParams, BaseEstimator):
    """Synthesize tabular rows by sampling a language model's learned joint.

    Fit on a whole table (features and any target alike are just columns); column
    order is permuted throughout training, so the model learns
    ``p(any column | any subset of the others)``. :meth:`sample` then draws rows
    one column at a time -- each cell conditioning on ``condition`` plus the cells
    already produced -- the autoregressive factorization the fine-tune was trained
    for. Categorical columns are drawn from their scored level distribution;
    numeric columns are generated.

    Parameters
    ----------
    model : Model, optional
        A model id string (default ``"distilgpt2"``), an already-loaded HF/MLX
        model, or a zero-argument factory returning one. A factory reloads on
        each fit (refit-safe); a bare object is fine-tuned in place.
    backend : LanguageModelBackend or str, optional
        ``"huggingface"`` (default) builds a fresh :class:`~sklm.HFBackend` per
        fit, or pass an :class:`~sklm.LanguageModelBackend` instance.
    training : TrainingConfig, optional
        Fine-tuning hyperparameters.
    generation : GenerationConfig, optional
        Sampling hyperparameters (temperature, token budget). ``temperature > 0``
        is what gives the rows their diversity -- greedy decoding collapses every
        row to the same mode.
    serializer : str or Serializer, optional
        ``"json"``, ``"key-value"``, ``"bracket"``, or a custom
        :class:`~sklm.Serializer`. Default ``"json"``.
    max_decimals : int or None, optional
        Round numeric cells to at most this many decimal places when
        serializing. Applies only to the string ``serializer`` selectors; a
        :class:`~sklm.Serializer` instance keeps its own number format.
        Default ``3``.
    random_state : int or None, optional
        Seed forwarded to the backend and serializer.
    callback : Callback, list of Callback, "auto" or None, optional
        Feedback hooks for fitting and inference. A list is wrapped in a
        :class:`~sklm.CompositeCallback`. ``"auto"`` (default) selects a
        dashboard for the runtime environment (Jupyter, rich, or logging);
        ``None`` disables feedback entirely.
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
    lm_ : TabularLanguageModel
        The fitted core model rows are drawn from.
    n_features_in_ : int
        Number of columns seen during :meth:`fit`.
    feature_names_in_ : numpy.ndarray
        Column names, set only when ``X`` is a DataFrame.
    """

    _args = EstimatorArgs

    generation: GenerationConfig

    def __init__(self, model: Model = "distilgpt2", **kwargs: Unpack[EstimatorArgs]) -> None:
        self.model = model
        defaults = AnnotatedDefault.create_with_defaults(
            EstimatorArgs, valid_params=self._get_param_names(), **kwargs
        )
        for key, value in defaults.items():
            setattr(self, key, value)

    def fit(self, X: object, y: object = None) -> Self:
        """Fine-tune the backend on the whole table's joint distribution.

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape (n_samples, n_features)
            Training table. Every column is modeled jointly; a supervised target,
            if any, is just another column of ``X``.
        y : ignored
            Present for the scikit-learn API. The model is joint, so there is no
            separate target.

        Returns
        -------
        Self
            The fitted estimator.

        Raises
        ------
        ValueError
            If ``X`` is not 2-dimensional, or if ``backend``/``serializer`` is an
            unknown string selector.
        """
        X_df = to_frame(X)
        self.n_features_in_ = X_df.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        else:
            forget(self, "feature_names_in_")
        X_df = X_df.set_axis([str(c) for c in X_df.columns], axis=1)
        self.lm_ = make_tabular_lm(self).fit(X_df)
        return self

    def sample(
        self,
        n_samples: int = 1,
        *,
        condition: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
        generation: GenerationConfig | None = None,
    ) -> pd.DataFrame:
        """Draw rows from the learned joint distribution.

        Parameters
        ----------
        n_samples : int, optional
            Number of rows to draw. Default ``1``.
        condition : mapping, sequence of mappings, or None, optional
            Columns to hold fixed. A single mapping is broadcast to all
            ``n_samples`` rows; a sequence gives one mapping per row and overrides
            ``n_samples``. ``None`` (default) samples unconditionally.
        generation : GenerationConfig, optional
            Per-call override of the estimator's ``generation``; ``None`` (default)
            uses the fitted attribute, without mutating the estimator.

        Returns
        -------
        pandas.DataFrame
            One row per draw, columns in the training order; numeric columns are
            cast to float.

        Raises
        ------
        ValueError
            If ``condition`` references a column not seen at fit.
        RuntimeError
            If any row stays malformed after the core's retries -- never a silent
            partial table.
        """
        check_is_fitted(self)
        generation = self.generation if generation is None else generation
        return self.lm_.sample(n_samples, condition=condition, generation=generation)

    @override
    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.string = True
        tags.input_tags.categorical = True
        tags.input_tags.allow_nan = True
        return tags
