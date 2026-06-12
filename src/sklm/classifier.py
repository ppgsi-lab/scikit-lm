"""Language-model classifier.

Conditions on all features and ranks the (fixed) candidate labels by their
likelihood under the model, so ``predict_proba`` is well defined and predicted
labels are always valid members of ``classes_``.
"""

from __future__ import annotations

from typing import Self, Unpack, override

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import Tags
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_consistent_length, check_is_fitted, column_or_1d

from .base import align_features, forget, make_tabular_lm, records, to_frame, unique_name
from .callbacks import predict_batches
from .config import GenerationConfig
from .params import AnnotatedDefault, EstimatorArgs, _FlatParams
from .serialize import is_missing

__all__ = ["LanguageModelClassifier"]


class LanguageModelClassifier(_FlatParams, ClassifierMixin, BaseEstimator):
    """Classify tabular rows by scoring candidate labels with a language model.

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
        Drives candidate ranking. ``permute_order`` / ``n_samples`` /
        ``score_pool`` marginalize each candidate's likelihood over feature
        order, and ``inference_batch_size`` sets the scoring batch. Candidate
        scoring is deterministic, so the stochastic sampling fields
        (``temperature``, ``top_p``, ``top_k``, ``repetition_penalty``,
        ``max_new_tokens``) are inert.
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
    classes_ : numpy.ndarray
        Sorted unique labels; the column order of ``predict_proba``.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    feature_names_in_ : numpy.ndarray
        Feature names, set only when ``X`` is a DataFrame.
    """

    _args = EstimatorArgs

    def __init__(self, model: str = "distilgpt2", **kwargs: Unpack[EstimatorArgs]) -> None:
        self.model = model
        defaults = AnnotatedDefault.create_with_defaults(
            EstimatorArgs, valid_params=self._get_param_names(), **kwargs
        )
        for key, value in defaults.items():
            setattr(self, key, value)

    def fit(self, X: object, y: object) -> Self:
        """Fine-tune the backend on ``X`` with ``y`` appended as a column.

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape (n_samples, n_features)
            Feature table.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        Self
            The fitted estimator.

        Raises
        ------
        ValueError
            If ``X`` is not 2-dimensional, or if ``backend``/``serializer`` is
            an unknown string selector.
        """
        X_df = to_frame(X)
        y_arr = column_or_1d(y, warn=True)
        check_consistent_length(X_df, y_arr)
        check_classification_targets(y_arr)
        self.classes_ = np.unique(y_arr)
        self.n_features_in_ = X_df.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        else:
            forget(self, "feature_names_in_")
        X_df = X_df.set_axis([str(c) for c in X_df.columns], axis=1)
        self.feature_cols_ = [str(c) for c in X_df.columns]
        self.target_col_ = unique_name("target", self.feature_cols_)
        frame = X_df.copy()
        frame[self.target_col_] = y_arr
        self.lm_ = make_tabular_lm(self).fit(frame, target_cols=frozenset({self.target_col_}))
        return self

    def _known_rows(self, X: object) -> list[dict[str, object]]:
        return records(align_features(self, X))

    def predict_proba(self, X: object, *, generation: GenerationConfig | None = None) -> np.ndarray:
        """Return class probabilities, with columns ordered as ``classes_``.

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape (n_samples, n_features)
            Feature table.
        generation : GenerationConfig, optional
            Per-call override of the estimator's ``generation``. ``None``
            (default) uses the fitted estimator's attribute; passing a config
            (e.g. to change order marginalization or the scoring batch) applies
            to this call only, without mutating the estimator.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_classes)
            Per-row distribution over ``classes_`` (each row sums to 1).

        Raises
        ------
        ValueError
            If ``X``'s feature count does not match what was seen at fit.
        """
        check_is_fitted(self)
        generation = self.generation if generation is None else generation
        rows = self._known_rows(X)
        candidates = list(self.classes_)
        cb = self.lm_.callback
        knowns = [{c: v for c, v in row.items() if not is_missing(v)} for row in rows]
        batch_size = generation.inference_batch_size or self.training.batch_size
        proba = np.empty((len(rows), len(candidates)))
        for start, stop in predict_batches(cb, len(rows), batch_size):
            proba[start:stop] = self.lm_.predict_proba_many(
                knowns[start:stop],
                self.target_col_,
                candidates,
                generation,
                row_ids=range(start, stop),
            )
        return proba

    def predict(self, X: object, *, generation: GenerationConfig | None = None) -> np.ndarray:
        """Return the most likely label per row.

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape (n_samples, n_features)
            Feature table.
        generation : GenerationConfig, optional
            Per-call override of the estimator's ``generation``, forwarded to
            :meth:`predict_proba`; ``None`` (default) uses the fitted attribute.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Predicted labels drawn from ``classes_``.
        """
        proba = self.predict_proba(X, generation=generation)
        return self.classes_[np.argmax(proba, axis=1)]

    @override
    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.string = True
        tags.input_tags.categorical = True
        tags.input_tags.allow_nan = True
        return tags
