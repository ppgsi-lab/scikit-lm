"""Hyperparameter configuration objects.

``TrainingConfig``, ``GenerationConfig`` and ``LoRAConfig`` are held by the
estimators as nested parameters and handed to the backend at call time;
``ModelConfig`` is reassembled internally from the estimators' flat
model-loading fields. The first three subclass
:class:`~sklearn.base.BaseEstimator` so scikit-learn recurses into them with
``__`` addressing -- e.g. ``GridSearchCV(param_grid={"training__epochs": [2, 4]})``
-- which a plain (frozen) dataclass, being an opaque leaf, cannot support.
``ModelConfig`` stays a frozen dataclass because it is never a tunable parameter.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from sklearn.base import BaseEstimator

__all__ = [
    "Device",
    "DiscretizationConfig",
    "GenerationConfig",
    "LRScheduler",
    "LoRAConfig",
    "ModelConfig",
    "Optimizer",
    "Precision",
    "QuantMethod",
    "Quantization",
    "QuantizationConfig",
    "TrainingConfig",
    "aggregate_default",
]

type Quantization = Literal["2bit", "3bit", "4bit", "6bit", "8bit"]
type QuantMethod = Literal["auto", "bitsandbytes", "hqq", "mlx"]
type Precision = Literal["fp32", "bf16", "fp16"]
type Optimizer = Literal["adamw", "adamw_8bit", "paged_adamw_8bit", "adafactor", "lion"]
type LRScheduler = Literal["constant", "linear", "cosine"]
type Device = Literal["auto", "cuda", "mps", "cpu"] | str


@dataclass
class LoRAConfig(BaseEstimator):
    """LoRA adapter configuration.

    Parameters
    ----------
    rank : int
        Low-rank decomposition size. Default ``16``.
    alpha : int
        Scaling factor; the effective scale is ``alpha / rank``. Default ``32``.
    dropout : float
        Dropout applied inside the LoRA layers. Default ``0.0``.
    target_modules : str, list of str or None
        Modules to adapt: a list of names/regex patterns, or the sentinel
        ``"all-linear"`` (every linear layer). ``None`` lets the backend decide.
    rank_pattern : dict[str, int] or None
        Per-module ``rank`` overrides keyed by module-name suffix.
    alpha_pattern : dict[str, int] or None
        Per-module ``alpha`` overrides keyed by module-name suffix.
    rslora : bool
        Rank-stabilized LoRA: effective scale becomes ``alpha / sqrt(rank)``.
    dora : bool
        Weight-decomposed low-rank adaptation (DoRA).
    """

    rank: int = 16
    # Independent of ``rank`` (no auto-tracking); set both explicitly.
    alpha: int = 32
    dropout: float = 0.0
    target_modules: str | list[str] | None = None
    rank_pattern: dict[str, int] | None = None
    alpha_pattern: dict[str, int] | None = None
    rslora: bool = False
    dora: bool = False


@dataclass
class QuantizationConfig(BaseEstimator):
    """How the base weights are quantized.

    The bit width is portable across backends; the library that performs the
    quantization (``method``) is backend-specific and resolved against each
    backend's capability matrix at load time -- an explicit method the resolved
    backend cannot provide raises there.

    Parameters
    ----------
    bits : int
        Weight bit width. Supported widths depend on the backend and ``method``:
        the MLX backend does 2/3/4/6/8-bit natively; the HF backend does 4/8-bit
        via bitsandbytes and 2/3-bit via HQQ.
    method : {"auto", "bitsandbytes", "hqq", "mlx"}
        Quantization library. ``"auto"`` (default) lets the backend choose: the
        HF backend uses bitsandbytes for 4-/8-bit and HQQ otherwise; the MLX
        backend always uses its native quantizer.
    group_size : int or None
        Quantization group size for methods that expose it (HQQ, MLX); ignored by
        bitsandbytes. ``None`` (default) uses the library default.
    """

    bits: int
    method: QuantMethod = "auto"
    group_size: int | None = None


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """How the base model is loaded.

    Parameters
    ----------
    model : str
        Model id/path handed to the backend at fit time. Default ``"gpt2"``.
    lora : LoRAConfig or None
        Fine-tune with LoRA adapters when set; full-weight fine-tuning when
        ``None`` (default).
    quantization : QuantizationConfig or None
        How to quantize the base weights, resolved from the estimator's
        ``quantization`` selector (a ``"<n>bit"`` string or a
        :class:`QuantizationConfig`). ``None`` (default) loads at ``precision``.
    precision : {"fp32", "bf16", "fp16"}
        Compute dtype for the (unquantized) weights and the train/generate
        autocast. Default ``"fp32"``.
    tokenizer : str or None
        Tokenizer id/path; ``None`` (default) derives it from the model.
    trust_remote_code : bool
        Allow custom model/tokenizer code from the hub. Default ``False``.
    device : str
        Target device (``"cuda"``/``"mps"``/``"cpu"``) or ``"auto"`` (default).
    attn_implementation : str or None
        Attention kernel passed to ``from_pretrained`` (e.g.
        ``"flash_attention_2"``). ``None`` (default) keeps the model default.
    """

    model: str = "gpt2"
    lora: LoRAConfig | None = None
    quantization: QuantizationConfig | None = None
    precision: Precision = "fp32"
    tokenizer: str | None = None
    trust_remote_code: bool = False
    device: Device = "auto"
    attn_implementation: str | None = None


@dataclass
class TrainingConfig(BaseEstimator):
    """Fine-tuning hyperparameters.

    Parameters
    ----------
    epochs : int
        Number of passes over the training rows. Default ``50``.
    batch_size : int
        Per-device batch size. Default ``16``.
    learning_rate : float or "auto"
        Optimizer learning rate. ``"auto"`` (default) picks ``2e-5`` for
        full-weight fine-tuning and ``2e-4`` when LoRA is enabled (see
        ``ModelConfig.lora``); pass a float to override.
    max_steps : int or None
        Optimizer-step ceiling. When set, training runs for
        ``min(epochs * steps_per_epoch, max_steps)``. ``None`` (default) ties
        duration to ``epochs`` alone.
    weight_decay : float
        L2 regularization. Default ``0.0``.
    grad_accumulation_steps : int
        Micro-batches accumulated before each optimizer step. Default ``1``.
    lr_scheduler : {"constant", "linear", "cosine"}
        Learning-rate schedule. Default ``"cosine"``.
    warmup_ratio : float
        Fraction of total steps spent linearly warming the LR up from 0.
        Default ``0.0``.
    max_grad_norm : float or None
        Global gradient-norm clip threshold; ``None`` disables clipping.
        Default ``1.0``.
    optimizer : {"adamw", "adamw_8bit", "paged_adamw_8bit", "adafactor", "lion"}
        Optimizer. The 8-bit variants require bitsandbytes + CUDA and fall back
        to plain AdamW elsewhere. Default ``"adamw"``.
    label_smoothing : float
        Cross-entropy label-smoothing factor in ``[0, 1)``. Default ``0.0``.
    neftune_noise_alpha : float or None
        NEFTune embedding-noise magnitude; ``None`` (default) disables it.
    gradient_checkpointing : bool
        Recompute activations in the backward pass to save memory. Default
        ``False``.
    max_seq_length : int or None
        Token cap per serialized row (also bounds inference prompts). ``None``
        (default) resolves it at ``fit`` to the longest serialized training row,
        rounded up to a multiple of 8, so no row is truncated. Set an int to cap
        explicitly -- rows beyond it are truncated from the right at training,
        which drops the trailing target columns.
    augmentation_factor : int
        Maximum number of distinct column orders emitted per row each epoch.
        Each variant is a permutation of the row's present (non-missing)
        columns; a row with ``m`` present columns has at most ``m!`` distinct
        orders, so the effective count is ``min(augmentation_factor, m!)``.
        ``1`` (default) keeps a single permutation per row.
    loss_on_target_only : bool
        When ``True``, the loss is computed only on the target columns of each
        row (the appended label for the classifier/regressor, the NaN columns
        for the imputer). Those columns are serialized last and the preceding
        context tokens are masked out with ``-100``, so the model is supervised
        only on what it must predict at inference. Rows with no observed target
        column are supervised in full. Default ``False`` (loss on every token).
        Inert for the oversampler.
    validation_split : float
        Fraction of training rows held out for validation each ``fit``, in
        ``[0.0, 1.0)``. ``0.0`` (default) trains on every row and reports no eval
        metric. When positive, the held-out rows are serialized once (a single
        fixed column order) and the backend reports validation loss through
        :meth:`~sklm.Callback.on_eval_report`.
    stratify : bool
        When splitting (``validation_split > 0``), stratify the hold-out on the
        sole target column -- binning it into quantiles when numeric, so
        classification labels and regression targets are both balanced across the
        split. Falls back to a random split when stratification is infeasible
        (e.g. a class too rare to appear on both sides) or when the estimator has
        no single fully-observed target (imputer, oversampler). Default ``True``.
    checkpoint_steps : int or None
        Save a model checkpoint every this many optimizer steps. ``None``
        (default) saves only what early stopping needs. Combined with
        ``validation_split``, the checkpoint with the lowest validation loss is
        the one early stopping restores.
    checkpoint_dir : str or None
        Directory to persist checkpoints in. ``None`` (default) writes them to a
        temporary directory removed when ``fit`` returns (so checkpoints serve
        only early stopping's best-model restore); set a path to keep them.
    early_stopping_patience : int or None
        Stop training after this many consecutive validations without improvement
        in validation loss, restoring the best checkpoint. Requires
        ``validation_split > 0``. ``None`` (default) disables early stopping.
    """

    epochs: int = 50
    batch_size: int = 16
    learning_rate: float | Literal["auto"] = "auto"
    max_steps: int | None = None
    weight_decay: float = 0.0
    grad_accumulation_steps: int = 1
    lr_scheduler: LRScheduler = "cosine"
    warmup_ratio: float = 0.0
    max_grad_norm: float | None = 1.0
    optimizer: Optimizer = "adamw"
    label_smoothing: float = 0.0
    neftune_noise_alpha: float | None = None
    gradient_checkpointing: bool = False
    max_seq_length: int | None = None
    augmentation_factor: int = 1
    loss_on_target_only: bool = False
    validation_split: float = 0.0
    stratify: bool = True
    checkpoint_steps: int | None = None
    checkpoint_dir: str | None = None
    early_stopping_patience: int | None = None

    def __post_init__(self) -> None:
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError(f"max_steps must be a positive integer or None, got {self.max_steps}")
        if self.max_seq_length is not None and self.max_seq_length <= 0:
            raise ValueError(
                f"max_seq_length must be a positive integer or None, got {self.max_seq_length}"
            )
        if self.augmentation_factor < 1:
            raise ValueError(
                f"augmentation_factor must be a positive integer, got {self.augmentation_factor}"
            )
        if not 0.0 <= self.validation_split < 1.0:
            raise ValueError(f"validation_split must be in [0.0, 1.0), got {self.validation_split}")
        if self.checkpoint_steps is not None and self.checkpoint_steps <= 0:
            raise ValueError(
                f"checkpoint_steps must be a positive integer or None, got {self.checkpoint_steps}"
            )
        if self.early_stopping_patience is not None:
            if self.early_stopping_patience <= 0:
                raise ValueError(
                    "early_stopping_patience must be a positive integer or None, got "
                    f"{self.early_stopping_patience}"
                )
            if self.validation_split <= 0.0:
                raise ValueError(
                    "early_stopping_patience requires validation_split > 0 "
                    "(no validation metric to monitor otherwise)"
                )

    def resolved_learning_rate(self, model_config: ModelConfig) -> float:
        """Concrete learning rate, resolving the ``"auto"`` sentinel.

        ``"auto"`` selects ``2e-4`` when ``model_config`` enables LoRA and
        ``2e-5`` for full-weight fine-tuning; an explicit float passes through.

        Returns
        -------
        float
            The resolved learning rate.
        """
        if self.learning_rate != "auto":
            return self.learning_rate
        return 2e-4 if model_config.lora is not None else 2e-5


def aggregate_default(draws: list[object], numeric: bool) -> object:
    """Collapse ``n_samples`` draws of a single cell into one value.

    The default :attr:`GenerationConfig.aggregate`: the mean of numeric draws, or
    the most common value otherwise (ties broken by first occurrence). ``draws``
    is always non-empty -- the caller drops malformed draws and never aggregates
    an empty group.

    Parameters
    ----------
    draws : list of object
        The per-sample values produced for one cell.
    numeric : bool
        Whether the cell's column is numeric.

    Returns
    -------
    object
        The mean of the numeric draws, or the most common value otherwise.
    """
    if numeric:
        values = [float(d) for d in draws if isinstance(d, int | float)]
        return sum(values) / len(values)
    return Counter(draws).most_common(1)[0][0]


@dataclass
class GenerationConfig(BaseEstimator):
    """Generation hyperparameters for the generative estimators.

    Parameters
    ----------
    temperature : float
        Sampling temperature; ``<= 0`` selects greedy decoding. Default ``0.7``.
    top_p : float
        Nucleus-sampling threshold; ``1.0`` (default) disables it.
    top_k : int
        Top-k sampling; ``0`` (default) disables it.
    max_new_tokens : int or None
        Token budget per generated value. ``None`` (default) resolves at generate
        time to the fitted ``max_seq_length`` (the longest serialized training
        row), an upper bound that covers any single value without manual tuning.
    repetition_penalty : float or None
        Penalty for repeated tokens; ``None`` (default) disables it.
    inference_batch_size : int or None
        Number of prompts per backend inference call (generation and scoring).
        ``None`` (default) resolves to the training ``batch_size`` at inference
        time, keeping the memory footprint in line with what fine-tuning already
        handled. Inference work is chunked into batches of this size.
    n_samples : int
        For the generative estimators (regressor, imputer): draws generated per
        cell and collapsed via :attr:`aggregate`. For the classifier (which
        scores rather than generates): how many distinct column orders to
        marginalize the candidate likelihood over, but only when
        ``permute_order`` is enabled. ``1`` (default) draws/scores once (no
        ensembling). Inert for the oversampler, which keeps each draw as a
        distinct synthetic row.
    permute_order : bool
        Re-permute the conditioning columns' order per draw, so the ``n_samples``
        draws marginalize over feature order rather than fixing one arbitrary
        order. For the classifier this scores each candidate under ``n_samples``
        distinct column orders and pools the results via :attr:`score_pool`.
        ``False`` (default) reuses the same order. Has no effect when
        ``n_samples == 1`` or a row conditions on fewer than two columns.
    aggregate : callable
        Generative estimators only: ``(draws, numeric) -> value`` collapsing the
        per-cell draws into one value. Default :func:`aggregate_default` (mean if
        numeric, else mode).
    score_pool : callable or None
        Classifier only: ``(logprob_rows) -> distribution`` pooling the per-order
        candidate log-likelihood vectors (one raw, unnormalized
        ``Sequence[float]`` per column order) into one probability row. ``None``
        (default) averages the per-order softmax distributions (proper order
        marginalization). Inert unless ``permute_order`` is enabled.
    """

    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    max_new_tokens: int | None = None
    repetition_penalty: float | None = None
    inference_batch_size: int | None = None
    n_samples: int = 1
    permute_order: bool = True
    aggregate: Callable[[list[object], bool], object] = aggregate_default
    score_pool: Callable[[list[Sequence[float]]], Sequence[float]] | None = None


@dataclass
class DiscretizationConfig(BaseEstimator):
    """Discretize a numeric target so it is predicted by candidate scoring.

    When enabled, the regressor stops generating-and-averaging and instead ranks
    a fixed set of candidate values by conditional likelihood (the classifier's
    mechanism), then reduces the resulting distribution to a single number. The
    candidates are real observed values of the target, kept in-distribution so the
    model scores tokens it saw during fine-tuning.

    Parameters
    ----------
    bins : int or float
        Discretization amount, and the on/off switch. ``0`` or ``0.0`` (default)
        keeps the generative path. An ``int`` ``K`` scores ``K`` candidates
        (capped at the number of distinct observed values). A ``float`` in
        ``(0, 1]`` is a fraction of that distinct support -- ``0.5`` keeps half
        (rounded up), ``1.0`` keeps every distinct value.
    strategy : {"quantile", "uniform"}, optional
        How the observed support is partitioned before a representative is drawn
        per partition: ``"quantile"`` (default) builds equal-mass strata (more
        resolution where the data concentrates); ``"uniform"`` builds equal-width
        bins. Inert when ``bins`` selects the full support.
    representative : {"median", "mode", "mean"}, optional
        The candidate value taken from each partition. ``"median"`` (default, the
        lower median) and ``"mode"`` are real observed values; ``"mean"`` is
        synthetic and may serialize to tokens the model never emitted.
    estimate : {"mean", "mode"}, optional
        How the scored distribution collapses to a prediction. ``"mean"``
        (default) is the probability-weighted expectation over the candidates;
        ``"mode"`` is the single most likely candidate (argmax).
    """

    bins: int | float = 0
    strategy: Literal["quantile", "uniform"] = "quantile"
    representative: Literal["median", "mode", "mean"] = "median"
    estimate: Literal["mean", "mode"] = "mean"

    def __post_init__(self) -> None:
        if isinstance(self.bins, bool):
            raise ValueError(f"bins must be an int or float, not bool, got {self.bins!r}")
        if isinstance(self.bins, float):
            if not 0.0 <= self.bins <= 1.0:
                raise ValueError(f"a float bins must be a fraction in [0.0, 1.0], got {self.bins}")
        elif self.bins < 0:
            raise ValueError(f"an int bins must be non-negative, got {self.bins}")

    def resolve_k(self, n_unique: int) -> int:
        """Number of candidates to draw from ``n_unique`` distinct observed values.

        An ``int`` ``bins`` is the count itself, capped at ``n_unique``; a
        ``float`` is a fraction of ``n_unique`` rounded up (never below 1), with
        ``1.0`` meaning the full support.

        Returns
        -------
        int
            The number of candidates to draw.
        """
        if isinstance(self.bins, float):
            return n_unique if self.bins == 1.0 else max(1, math.ceil(self.bins * n_unique))
        return min(self.bins, n_unique)
