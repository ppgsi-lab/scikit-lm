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
from abc import ABC
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from sklearn.base import BaseEstimator

from .bridge import Model, Tokenizer

__all__ = [
    "CheckpointConfig",
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
    model : Model
        What the backend fine-tunes: a model id/path string (loaded at fit time
        with the fields below), an already-loaded HF/MLX model, or a
        zero-argument factory returning one of those. A factory is re-invoked on
        every ``fit`` (so refits start clean); a bare loaded object is used in
        place. Default ``"gpt2"``.
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
    tokenizer : Tokenizer or None
        Tokenizer id/path, an already-loaded tokenizer, or a factory returning
        one. ``None`` (default) derives it from the model id, and is required to
        be set when ``model`` is a pre-loaded object (there is no id to derive
        from).
    trust_remote_code : bool
        Allow custom model/tokenizer code from the hub. Default ``False``.
    device : str
        Target device (``"cuda"``/``"mps"``/``"cpu"``) or ``"auto"`` (default).
    attn_implementation : str or None
        Attention kernel passed to ``from_pretrained`` (e.g.
        ``"flash_attention_2"``). ``None`` (default) keeps the model default.
    """

    model: Model = "gpt2"
    lora: LoRAConfig | None = None
    quantization: QuantizationConfig | None = None
    precision: Precision = "fp32"
    tokenizer: Tokenizer | None = None
    trust_remote_code: bool = False
    device: Device = "auto"
    attn_implementation: str | None = None


@dataclass
class CheckpointConfig(BaseEstimator):
    """When and where to persist model checkpoints during fine-tuning.

    Both backends write numbered ``checkpoint-<step>`` snapshots into ``dir`` on
    the cadence set by ``each``/``on``, keeping the ``keep`` most recent. When a
    validation set is held out (``TrainingConfig.validation_split > 0``) the
    lowest-validation-loss checkpoint is additionally tracked and protected from
    pruning, so it is always available to restore. ``dir`` is read as well as
    written: if it already holds ``checkpoint-*`` snapshots, ``fit`` resumes from
    the most recent one instead of starting from the base model.

    Parameters
    ----------
    each : int
        Cadence count, in the unit set by ``on``. Default ``1``.
    on : {"step", "epoch"}
        Cadence unit: every ``each`` optimizer steps, or every ``each`` epochs.
        Default ``"step"``.
    dir : str or None
        Directory checkpoints are written to and resumed from. ``None`` (default)
        writes them to a temporary directory removed when ``fit`` returns, so
        checkpoints serve only the best-model restore.
    keep : int or None
        Number of most-recent snapshots to retain (the best is kept on top of
        these when a validation set exists). ``None`` keeps every snapshot (the
        full training trajectory). Default ``1``.
    """

    each: int = 1
    on: Literal["step", "epoch"] = "step"
    dir: str | None = None
    keep: int | None = 1

    def __post_init__(self) -> None:
        if self.each <= 0:
            raise ValueError(f"each must be a positive integer, got {self.each}")
        if self.keep is not None and self.keep <= 0:
            raise ValueError(f"keep must be a positive integer or None, got {self.keep}")


@dataclass
class LRScheduler(BaseEstimator, ABC):
    """Learning-rate schedule for fine-tuning.

    Build a schedule through one of the named constructors rather than
    instantiating this base directly:

    - :meth:`constant` -- flat learning rate.
    - :meth:`linear` -- linear decay to zero.
    - :meth:`cosine` -- half-cosine decay to zero.
    - :meth:`plateau` -- lower the rate when validation loss stops improving.

    Every schedule carries a peak ``learning_rate`` (the initial rate it
    operates on; ``"auto"`` picks a LoRA-aware default -- see
    :meth:`resolved_learning_rate`). The step-based schedules (``constant``,
    ``linear``, ``cosine``) also take a ``warmup_ratio``; ``plateau`` reacts to
    the validation metric instead and requires
    ``TrainingConfig.validation_split > 0``.

    Examples
    --------
    >>> TrainingConfig(lr_scheduler=LRScheduler.cosine(learning_rate=2e-5, warmup_ratio=0.1))
    >>> TrainingConfig(
    ...     lr_scheduler=LRScheduler.plateau(patience=5, factor=0.5),
    ...     validation_split=0.2,
    ... )
    """

    learning_rate: float | Literal["auto"] = "auto"

    @staticmethod
    def constant(
        *, learning_rate: float | Literal["auto"] = "auto", warmup_ratio: float = 0.0
    ) -> LRScheduler:
        """Build a constant (flat) learning-rate schedule.

        Parameters
        ----------
        learning_rate : float or "auto", optional
            Peak learning rate. ``"auto"`` (default) selects ``2e-4`` with LoRA
            and ``2e-5`` for full-weight fine-tuning; a float overrides.
        warmup_ratio : float, optional
            Fraction of total optimizer steps spent linearly warming up from
            zero before the rate goes flat. Must be in ``[0.0, 1.0)``.
            Default ``0.0``.

        Returns
        -------
        LRScheduler
            The configured constant schedule.
        """
        return ConstantLR(learning_rate=learning_rate, warmup_ratio=warmup_ratio)

    @staticmethod
    def linear(
        *,
        learning_rate: float | Literal["auto"] = "auto",
        warmup_ratio: float = 0.0,
        floor: float = 0.0,
    ) -> LRScheduler:
        """Build a linear-decay schedule.

        The learning rate decays linearly from ``learning_rate`` to ``floor``
        over the planned training steps, after an optional linear warmup.

        Parameters
        ----------
        learning_rate : float or "auto", optional
            Peak learning rate. ``"auto"`` (default) selects ``2e-4`` with LoRA
            and ``2e-5`` for full-weight fine-tuning; a float overrides.
        warmup_ratio : float, optional
            Fraction of total optimizer steps spent linearly warming up from
            zero before the decay. Must be in ``[0.0, 1.0)``. Default ``0.0``.
        floor : float, optional
            Lower bound the decay ends at, instead of zero. Must be ``>= 0`` and
            below ``learning_rate``. Default ``0.0`` (decay to zero).

        Returns
        -------
        LRScheduler
            The configured linear schedule.
        """
        return LinearLR(learning_rate=learning_rate, warmup_ratio=warmup_ratio, floor=floor)

    @staticmethod
    def cosine(
        *,
        learning_rate: float | Literal["auto"] = "auto",
        warmup_ratio: float = 0.0,
        floor: float = 0.0,
    ) -> LRScheduler:
        """Build a half-cosine decay schedule.

        The learning rate decays from ``learning_rate`` to ``floor`` following a
        half cosine over the planned training steps, after an optional linear
        warmup.

        Parameters
        ----------
        learning_rate : float or "auto", optional
            Peak learning rate. ``"auto"`` (default) selects ``2e-4`` with LoRA
            and ``2e-5`` for full-weight fine-tuning; a float overrides.
        warmup_ratio : float, optional
            Fraction of total optimizer steps spent linearly warming up from
            zero before the decay. Must be in ``[0.0, 1.0)``. Default ``0.0``.
        floor : float, optional
            Lower bound the decay ends at, instead of zero. Must be ``>= 0`` and
            below ``learning_rate``. Default ``0.0`` (decay to zero).

        Returns
        -------
        LRScheduler
            The configured cosine schedule.
        """
        return CosineLR(learning_rate=learning_rate, warmup_ratio=warmup_ratio, floor=floor)

    @staticmethod
    def plateau(
        *,
        learning_rate: float | Literal["auto"] = "auto",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        floor: float = 0.0,
        cooldown: int = 0,
    ) -> LRScheduler:
        """Build a schedule that lowers the rate on a validation-loss plateau.

        The learning rate starts at ``learning_rate`` and is multiplied by
        ``factor`` whenever the validation loss fails to improve for
        ``patience`` consecutive evaluations, down to a floor of ``floor``.
        Mirrors ``torch.optim.lr_scheduler.ReduceLROnPlateau`` (minimum mode,
        relative threshold) on both backends. Requires
        ``TrainingConfig.validation_split > 0``.

        Parameters
        ----------
        learning_rate : float or "auto", optional
            Peak (starting) learning rate the reductions apply to. ``"auto"``
            (default) selects ``2e-4`` with LoRA and ``2e-5`` for full-weight
            fine-tuning; a float overrides.
        factor : float, optional
            Multiplier applied on each reduction, in ``(0.0, 1.0)``.
            Default ``0.1``.
        patience : int, optional
            Consecutive evaluations with no improvement tolerated before the
            rate is reduced. Default ``10``.
        threshold : float, optional
            Minimum relative improvement that counts as progress
            (``new < best * (1 - threshold)``). Default ``1e-4``.
        floor : float, optional
            Lower bound the reductions stop at. Default ``0.0``.
        cooldown : int, optional
            Evaluations to wait after a reduction before resuming the
            no-improvement count. Default ``0``.

        Returns
        -------
        LRScheduler
            The configured plateau schedule.

        Notes
        -----
        ``patience`` is counted in evaluations, whose cadence differs slightly
        between backends (per epoch on MLX; per save cadence on HF) -- the same
        convention as ``TrainingConfig.early_stopping_patience``.
        """
        return PlateauLR(
            learning_rate=learning_rate,
            factor=factor,
            patience=patience,
            threshold=threshold,
            floor=floor,
            cooldown=cooldown,
        )

    def resolved_learning_rate(self, model_config: ModelConfig) -> float:
        """Concrete learning rate, resolving the ``"auto"`` sentinel.

        ``"auto"`` selects ``2e-4`` when ``model_config`` enables LoRA and
        ``2e-5`` for full-weight fine-tuning; an explicit float passes through.

        Parameters
        ----------
        model_config : ModelConfig
            Model configuration; its ``lora`` field selects the auto rate.

        Returns
        -------
        float
            The resolved learning rate.
        """
        if self.learning_rate != "auto":
            return self.learning_rate
        return 2e-4 if model_config.lora is not None else 2e-5


@dataclass
class _StepLR(LRScheduler):
    """Base for step-based schedules; carries the linear-warmup fraction.

    Parameters
    ----------
    warmup_ratio : float
        Fraction of total optimizer steps spent linearly warming up from zero.
        Must be in ``[0.0, 1.0)``.
    """

    warmup_ratio: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError(f"warmup_ratio must be in [0.0, 1.0), got {self.warmup_ratio}")


@dataclass
class _DecayLR(_StepLR):
    """Base for decaying step-based schedules; adds the learning-rate floor.

    Parameters
    ----------
    floor : float
        Lower bound the decay ends at, instead of zero. Must be ``>= 0``.
    """

    floor: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.floor < 0.0:
            raise ValueError(f"floor must be non-negative, got {self.floor}")


@dataclass
class ConstantLR(_StepLR):
    """Constant (flat) schedule. Build via :meth:`LRScheduler.constant`."""


@dataclass
class LinearLR(_DecayLR):
    """Linear-decay schedule. Build via :meth:`LRScheduler.linear`."""


@dataclass
class CosineLR(_DecayLR):
    """Half-cosine decay schedule. Build via :meth:`LRScheduler.cosine`."""


@dataclass
class PlateauLR(LRScheduler):
    """Reduce-on-plateau schedule. Build via :meth:`LRScheduler.plateau`."""

    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
    floor: float = 0.0
    cooldown: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.factor < 1.0:
            raise ValueError(f"factor must be in (0.0, 1.0), got {self.factor}")
        if self.patience <= 0:
            raise ValueError(f"patience must be a positive integer, got {self.patience}")
        if self.threshold < 0.0:
            raise ValueError(f"threshold must be non-negative, got {self.threshold}")
        if self.floor < 0.0:
            raise ValueError(f"floor must be non-negative, got {self.floor}")
        if self.cooldown < 0:
            raise ValueError(f"cooldown must be non-negative, got {self.cooldown}")


@dataclass
class TrainingConfig(BaseEstimator):
    """Fine-tuning hyperparameters.

    Parameters
    ----------
    epochs : int
        Number of passes over the training rows. Default ``50``.
    batch_size : int
        Per-device batch size. Default ``16``.
    max_steps : int or None
        Optimizer-step ceiling. When set, training runs for
        ``min(epochs * steps_per_epoch, max_steps)``. ``None`` (default) ties
        duration to ``epochs`` alone.
    weight_decay : float
        L2 regularization. Default ``0.0``.
    grad_accumulation_steps : int
        Micro-batches accumulated before each optimizer step. Default ``1``.
    lr_scheduler : LRScheduler
        Learning-rate schedule object, built via the :class:`LRScheduler`
        factories (:meth:`~LRScheduler.constant`, :meth:`~LRScheduler.linear`,
        :meth:`~LRScheduler.cosine`, :meth:`~LRScheduler.plateau`). Carries the
        peak ``learning_rate`` and, for the step-based shapes, ``warmup_ratio``.
        Default :meth:`LRScheduler.cosine`.
    max_grad_norm : float or None
        Global gradient-norm clip threshold; ``None`` disables clipping.
        Default ``1.0``.
    optimizer : {"adamw", "adamw_8bit", "paged_adamw_8bit", "adafactor", "lion"}
        Optimizer. The 8-bit variants require bitsandbytes + CUDA and fall back
        to plain AdamW elsewhere. ``"lion"`` diverges per backend: the HF
        backend runs it 8-bit (bitsandbytes ``lion_8bit`` on CUDA,
        mps-bitsandbytes ``Lion8bit`` on MPS, the plain-AdamW fallback on CPU),
        while the MLX backend runs full-precision ``Lion``. Default ``"adamw"``.
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
        Implies ``target_at_end`` (masking requires the target to be last).
        Inert for the oversampler.
    target_at_end : bool
        When ``True``, the target columns are serialized last in every row while
        the context columns keep getting permuted among themselves, but the loss
        stays on every token. Decouples target *position* from loss *scope*:
        ``loss_on_target_only`` already fixes the target last (and additionally
        masks the context), so it implies this; set ``target_at_end`` alone to
        pin the target's position while still supervising the whole row. No
        effect on rows without an observed target column, or on the oversampler.
        Default ``False`` (the target is permuted freely with the rest).
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
    checkpoint : CheckpointConfig or None
        When and where to persist checkpoints (cadence, directory, retention) and
        the directory to resume from. ``None`` (default) saves only what
        early stopping needs, to a temporary directory removed when ``fit``
        returns. See :class:`CheckpointConfig`.
    early_stopping_patience : int or None
        Stop training after this many consecutive validations without improvement
        in validation loss, restoring the best checkpoint. Requires
        ``validation_split > 0``. ``None`` (default) disables early stopping.
    """

    epochs: int = 50
    batch_size: int = 16
    max_steps: int | None = None
    weight_decay: float = 0.0
    grad_accumulation_steps: int = 1
    lr_scheduler: LRScheduler = field(default_factory=CosineLR)
    max_grad_norm: float | None = 1.0
    optimizer: Optimizer = "adamw"
    label_smoothing: float = 0.0
    neftune_noise_alpha: float | None = None
    gradient_checkpointing: bool = False
    max_seq_length: int | None = None
    augmentation_factor: int = 1
    loss_on_target_only: bool = False
    target_at_end: bool = False
    validation_split: float = 0.0
    stratify: bool = True
    checkpoint: CheckpointConfig | None = None
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
        if isinstance(self.lr_scheduler, PlateauLR) and self.validation_split <= 0.0:
            raise ValueError(
                "PlateauLR requires validation_split > 0 "
                "(no validation metric to monitor otherwise)"
            )


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
        ``True`` (default) always permutes the order. Has no effect when
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
    candidate_scoring : {"mean", "sum"}
        How the candidate-ranking path (classifier, imputer/oversampler
        categorical cells, numeric cells scored under discretization) reduces a
        candidate's per-token log-likelihoods before the softmax. ``"mean"``
        (default) is length-normalized: it favours levels that BPE-split into
        more, individually-predictable tokens. ``"sum"`` is the total
        log-likelihood -- the proper probability of the candidate string over a
        closed set -- which removes that bias but penalises longer candidates.
        Inert for the generative path (regressor; generated imputer/oversampler
        cells), which samples rather than scores.
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
    candidate_scoring: Literal["mean", "sum"] = "mean"


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
    estimate : {"mean", "mode", "median"}, optional
        How the scored distribution collapses to a prediction. ``"mean"``
        (default) is the probability-weighted expectation over the candidates;
        ``"median"`` is the probability-weighted median (the smallest candidate
        whose cumulative probability reaches half) -- a robust central estimate
        that pools nearby mass and, unlike ``"mean"``, is not dragged off by a
        distant low-probability candidate; ``"mode"`` is the single most likely
        candidate (argmax). ``"median"`` and ``"mode"`` return a real observed
        candidate; ``"mean"`` interpolates.
    sharpness : float, optional
        Temperature applied to the scored distribution before ``estimate``:
        probabilities are raised to this power and renormalized. ``1.0``
        (default) keeps the distribution as scored; larger values concentrate
        mass on the top candidates, so ``"mean"`` and ``"median"`` interpolate
        continuously toward ``"mode"`` as sharpness grows. ``"mode"`` (argmax)
        is invariant to it. Useful when the scored distribution is
        underconfident.
    """

    bins: int | float = 0
    strategy: Literal["quantile", "uniform"] = "quantile"
    representative: Literal["median", "mode", "mean"] = "median"
    estimate: Literal["mean", "mode", "median"] = "mean"
    sharpness: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.bins, bool):
            raise ValueError(f"bins must be an int or float, not bool, got {self.bins!r}")
        if isinstance(self.bins, float):
            if not 0.0 <= self.bins <= 1.0:
                raise ValueError(f"a float bins must be a fraction in [0.0, 1.0], got {self.bins}")
        elif self.bins < 0:
            raise ValueError(f"an int bins must be non-negative, got {self.bins}")
        if self.sharpness <= 0.0:
            raise ValueError(f"sharpness must be a positive float, got {self.sharpness}")

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
