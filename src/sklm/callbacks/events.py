from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ..config import TrainingConfig
    from ..serialize import TrainingExample


# --------------------------------------------------------------------------- state


@dataclass
class TrainingState:
    """Accumulated snapshot of the current fit/predict run.

    Built incrementally by :class:`Callback` as events arrive and handed to
    :meth:`Callback.on_event` on every change. The fit fields cover fine-tuning
    (loss series, derived epoch, memory); the predict fields cover inference
    progress. Per-value payloads (a single generation, score or retry) ride on
    the :class:`Event` instead -- a subclass that wants a rolling window keeps
    its own.
    """

    phase: Literal["idle", "fitting", "predicting"] = "idle"
    model: str = ""
    training: TrainingConfig | None = None
    n_rows: int = 0
    n_epochs: int = 0
    steps: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    eval_steps: list[int] = field(default_factory=list)
    eval_losses: list[float] = field(default_factory=list)
    epoch_losses: list[float] = field(default_factory=list)
    step: int | None = None
    loss: float | None = None
    eval_loss: float | None = None
    epoch: float | None = None
    learning_rate: float | None = None
    grad_norm: float | None = None
    mem: int | None = None
    peak_mem: int = 0
    examples: list[TrainingExample] = field(default_factory=list)
    example_epoch: int = 0
    predict_total: int = 0

    def recent_losses(self, window: int) -> list[float]:
        """The last ``window`` training losses (all of them if fewer)."""
        return self.losses[-window:]


# --------------------------------------------------------------------------- events


@dataclass(frozen=True)
class FitInfo:
    """Static fit configuration, emitted once before :class:`FitStart`."""

    model: str
    training: TrainingConfig


@dataclass(frozen=True)
class FitStart:
    """Fine-tuning is about to start over ``n_rows`` rows for ``n_epochs``."""

    n_rows: int
    n_epochs: int


@dataclass(frozen=True)
class TrainExamples:
    """A per-epoch sample of serialized training rows (the exact fine-tune text)."""

    examples: Sequence[TrainingExample]


@dataclass(frozen=True)
class TrainReport:
    """A training-loss report at ``step`` (``total_steps``/``epoch`` may be ``None``)."""

    step: int
    total_steps: int | None
    loss: float
    epoch: float | None
    learning_rate: float | None
    grad_norm: float | None


@dataclass(frozen=True)
class EvalReport:
    """A validation-loss report (fires only when ``validation_split > 0``)."""

    step: int
    loss: float
    epoch: float | None


@dataclass(frozen=True)
class Memory:
    """Accelerator memory in use, in bytes (``None`` when there is no accelerator)."""

    device_bytes: int | None


@dataclass(frozen=True)
class FitEnd:
    """Fine-tuning has finished."""


@dataclass(frozen=True)
class PredictStart:
    """Inference is about to process ``n`` rows."""

    n: int


@dataclass(frozen=True)
class RowEnd:
    """Row ``index`` of ``total`` is done."""

    index: int
    total: int


@dataclass(frozen=True)
class PredictEnd:
    """Inference has finished."""


@dataclass(frozen=True)
class Generation:
    """The backend generated ``completion`` for ``target`` from ``prompt``.

    Fires once per generated value, including each retry, with the raw prompt and
    raw generated text (before decoding), so malformed outputs are visible too.
    ``value`` is that text decoded into the target's value, or ``None`` when the
    decoding was malformed.
    """

    prompt: str
    completion: str
    target: str
    value: object | None


@dataclass(frozen=True)
class Score:
    """The classifier scored ``candidates`` for one row across column orders.

    ``prompts`` lists one conditioning prompt per order (a single element unless
    ``permute_order`` marginalizes over several); ``probs`` and ``logprobs`` are
    aligned to it, so ``probs[k]`` is order ``k``'s normalized distribution over
    ``candidates`` and ``logprobs[k]`` its raw mean per-token log-likelihoods.
    """

    prompts: Sequence[str]
    candidates: Sequence[object]
    probs: Sequence[Sequence[float]]
    logprobs: Sequence[Sequence[float]]


@dataclass(frozen=True)
class Retry:
    """Generation for ``target`` produced a malformed value on ``attempt`` of ``max_attempts``."""

    target: str
    attempt: int
    max_attempts: int


type Event = (
    FitInfo
    | FitStart
    | TrainExamples
    | TrainReport
    | EvalReport
    | Memory
    | FitEnd
    | PredictStart
    | RowEnd
    | PredictEnd
    | Generation
    | Score
    | Retry
)
