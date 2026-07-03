"""Language-model oversampler (imbalanced-learn compatible).

Fits on the labeled table, then balances classes by conditioning generation on
each minority label and synthesizing the remaining features. Numeric features
are generated; categorical features are drawn from their conditional
distribution over observed levels (scored, then sampled to keep the synthetic
rows diverse). Unlike SMOTE it operates on text, so categorical columns and
feature correlations need no numeric encoding.
"""

from __future__ import annotations

import warnings
from typing import Unpack, cast, override

import numpy as np
import pandas as pd
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import check_target_type

# imbalanced-learn redefines Tags (adding sampler_tags) and offers no public re-export;
# sklearn's Tags is not assignable to the BaseSampler.__sklearn_tags__ return type.
from imblearn.utils._tags import Tags
from sklearn.utils.validation import check_consistent_length

from .base import forget, make_tabular_lm, to_frame, unique_name
from .bridge import Model
from .config import GenerationConfig
from .params import AnnotatedDefault, OversamplerArgs, _FlatParams

__all__ = ["LanguageModelOverSampler"]


class LanguageModelOverSampler(_FlatParams, BaseOverSampler):
    """Balance classes by generating synthetic rows with a language model.

    Implements the imbalanced-learn sampler API: :meth:`fit_resample` returns
    ``(X_resampled, y_resampled)``. For each class that needs more samples, rows
    are generated conditioned on that class label and appended to the data.

    Parameters
    ----------
    sampling_strategy : str, float, dict or callable, optional
        Forwarded to imbalanced-learn; controls how many samples each class
        gets. Default ``"auto"``.
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
        Sampling hyperparameters (temperature, token budget).
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
    sampling_strategy_ : dict
        Maps each class label to the number of samples to synthesize.
    """

    _args = OversamplerArgs

    generation: GenerationConfig
    sampling_strategy_: dict[object, int]

    def __init__(
        self,
        *,
        sampling_strategy: object = "auto",
        model: Model = "distilgpt2",
        **kwargs: Unpack[OversamplerArgs],
    ) -> None:
        # imbalanced-learn accepts str | float | dict | callable but types the
        # parameter as str in its stub, so the real (often dict/float) value is
        # forwarded unchanged and the stub mismatch is suppressed.
        super().__init__(sampling_strategy=sampling_strategy)  # pyright: ignore[reportArgumentType]
        self.model = model
        defaults = AnnotatedDefault.create_with_defaults(
            OversamplerArgs, valid_params=self._get_param_names(), **kwargs
        )
        for key, value in defaults.items():
            setattr(self, key, value)

    @override
    def _check_X_y(
        self, X: object, y: object, accept_sparse: object | None = None
    ) -> tuple[pd.DataFrame, np.ndarray, object]:
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X_df = to_frame(X)
        self.n_features_in_ = X_df.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        else:
            forget(self, "feature_names_in_")
        return X_df, np.asarray(y), binarize_y

    # imbalanced-learn's stub types the abstract _fit_resample as returning None.
    @override
    def _fit_resample(self, X: object, y: object) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[reportIncompatibleMethodOverride]
        """Append synthetic minority rows so each class meets its target count.

        Parameters
        ----------
        X : pandas.DataFrame
            Validated feature table from :meth:`_check_X_y`.
        y : numpy.ndarray
            Validated labels.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Resampled ``(X, y)``; imbalanced-learn restores the original
            container type around this return.
        """
        X_df = to_frame(X, getattr(self, "feature_names_in_", None))
        X_df = X_df.set_axis([str(c) for c in X_df.columns], axis=1)
        y_arr = np.asarray(y)
        check_consistent_length(X_df, y_arr)
        feature_cols = [str(c) for c in X_df.columns]
        int_cols = {c for c in feature_cols if pd.api.types.is_integer_dtype(X_df[c])}
        target_col = unique_name("target", feature_cols)
        frame = X_df.copy()
        frame[target_col] = y_arr
        ignored = [
            flag
            for flag, on in (
                ("loss_on_target_only", self.training.loss_on_target_only),
                ("target_at_end", self.training.target_at_end),
            )
            if on
        ]
        if ignored:
            warnings.warn(
                f"{' and '.join(ignored)} {'have' if len(ignored) > 1 else 'has'} no effect on "
                "the oversampler: it conditions generation on the label and produces the "
                "features, so there is no fixed target column. Ignoring.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.lm_ = make_tabular_lm(self).fit(frame)

        cb = self.lm_.callback
        total = sum(self.sampling_strategy_.values())
        cb.on_predict_start(total)
        batch_size = self.generation.inference_batch_size or self.training.batch_size
        synth_rows: list[list[object]] = []
        synth_y: list[object] = []
        done = 0
        for label, n_to_generate in self.sampling_strategy_.items():
            made = 0
            attempts = 0
            budget = n_to_generate * 5 + 10
            while made < n_to_generate and attempts < budget:
                # Generate up to one inference batch of candidates per round
                # (capped by the leftover need and attempt budget), so on_row_end
                # fires as each round lands rather than all at the end.
                want = min(n_to_generate - made, budget - attempts, batch_size)
                attempts += want
                outs = self.lm_.complete_many(
                    [{target_col: label}] * want,
                    [feature_cols] * want,
                    self.generation,
                    sample_categorical=frozenset(self.lm_.categories_),
                    row_ids=range(attempts - want, attempts),
                )
                for out in outs:
                    if out is None:
                        continue
                    # Round integer-column values: imbalanced-learn restores the
                    # original int dtype afterwards, which would otherwise truncate.
                    # complete_many types each cell as object; numeric columns decode
                    # to a number, so the cast to float for round() is safe.
                    synth_rows.append(
                        [
                            round(cast(float, out[c])) if c in int_cols else out[c]
                            for c in feature_cols
                        ]
                    )
                    synth_y.append(label)
                    made += 1
                    cb.on_row_end(done, total)
                    done += 1
                    if made >= n_to_generate:
                        break
            if made < n_to_generate:
                raise RuntimeError(
                    f"only generated {made} of {n_to_generate} synthetic rows for class "
                    f"{label!r} within {budget} attempts; the model is not producing valid rows"
                )
        cb.on_predict_end()

        synth_df = pd.DataFrame(synth_rows, columns=pd.Index(feature_cols))
        X_res = pd.concat([X_df, synth_df], ignore_index=True)
        y_res = np.concatenate([y_arr, np.asarray(synth_y, dtype=y_arr.dtype)])
        return X_res.to_numpy(), y_res

    @override
    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.string = True
        tags.input_tags.categorical = True
        tags.input_tags.allow_nan = True
        tags.input_tags.sparse = False
        return tags
