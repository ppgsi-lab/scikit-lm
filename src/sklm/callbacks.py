"""Observability for fitting and inference.

The base :class:`Callback` is both the wire protocol the backends/``core`` call
(13 granular ``on_*`` events) **and** the aggregator: each event folds into a
:class:`TrainingState` and is then dispatched, together with that state, through
a single :meth:`Callback.on_event` hook. Subclass and override ``on_event`` --
the granular methods are ``@final``; overriding one would silently lose the
aggregation, so react in ``on_event`` instead.

The shipped dashboards (:class:`LoggingCallback`, :class:`TqdmCallback`,
:class:`RichCallback`) are themselves just such subclasses. The library never
configures the root logger -- handlers and levels are the caller's choice.
"""

from __future__ import annotations

import logging
import math
import os
import statistics
import sys
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from time import monotonic
from typing import TYPE_CHECKING, Any, Literal, final, override

if TYPE_CHECKING:
    from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
    from rich.live import Live
    from rich.measure import Measurement
    from rich.progress import Progress, Task, TaskID

    from .config import TrainingConfig
    from .serialize import TrainingExample

__all__ = [
    "Callback",
    "EvalReport",
    "Event",
    "FitEnd",
    "FitInfo",
    "FitStart",
    "Generation",
    "LoggingCallback",
    "Memory",
    "PredictEnd",
    "PredictStart",
    "Retry",
    "RichCallback",
    "RowEnd",
    "Score",
    "TqdmCallback",
    "TrainExamples",
    "TrainReport",
    "TrainingState",
    "predict_batches",
    "resolve_callbacks",
]


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
    """

    prompt: str
    completion: str
    target: str


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


# --------------------------------------------------------------------------- callback


class Callback:
    """Wire protocol + state aggregator; subclass and override :meth:`on_event`.

    Backends and ``core`` call the granular ``on_*`` methods. The base folds each
    into ``self.state`` and dispatches a single :meth:`on_event` carrying that
    state and the :class:`Event` that just occurred. The granular methods are
    ``@final``: overriding one would skip the aggregation, so subclasses react in
    ``on_event``. A bare ``Callback()`` is the no-op used when no callback is set.

    Attributes
    ----------
    state : TrainingState
        The running snapshot, reset at each :meth:`on_fit_start`.
    """

    def __init__(self) -> None:
        self.state = TrainingState()
        self._epoch_bucket: int | None = None

    def on_event(self, state: TrainingState, event: Event) -> None:
        """A new ``event`` updated ``state``. No-op by default; override to react."""

    @final
    def on_fit_info(self, model: str, training: TrainingConfig) -> None:
        self.state.model = model
        self.state.training = training
        self.on_event(self.state, FitInfo(model, training))

    @final
    def on_fit_start(self, n_rows: int, n_epochs: int) -> None:
        self.state = TrainingState(
            phase="fitting",
            model=self.state.model,
            training=self.state.training,
            n_rows=n_rows,
            n_epochs=n_epochs,
        )
        self._epoch_bucket = None
        self.on_event(self.state, FitStart(n_rows, n_epochs))

    @final
    def on_train_examples(self, examples: Sequence[TrainingExample]) -> None:
        self.state.example_epoch += 1
        self.state.examples = list(examples)
        self.on_event(self.state, TrainExamples(self.state.examples))

    @final
    def on_train_report(
        self,
        step: int,
        total_steps: int | None,
        loss: float,
        epoch: float | None,
        learning_rate: float | None,
    ) -> None:
        s = self.state
        # MLX does not report epoch; derive it from step progress instead.
        if epoch is None and total_steps:
            epoch = step / total_steps * s.n_epochs
        s.step, s.loss, s.epoch, s.learning_rate = step, loss, epoch, learning_rate
        s.steps.append(step)
        s.losses.append(loss)
        if epoch is not None:
            bucket = math.ceil(epoch)
            if bucket != self._epoch_bucket:
                self._epoch_bucket = bucket
                s.epoch_losses = []
            s.epoch_losses.append(loss)
        self.on_event(s, TrainReport(step, total_steps, loss, epoch, learning_rate))

    @final
    def on_eval_report(self, step: int, loss: float, epoch: float | None) -> None:
        s = self.state
        s.eval_loss = loss
        s.eval_steps.append(step)
        s.eval_losses.append(loss)
        self.on_event(s, EvalReport(step, loss, epoch))

    @final
    def on_memory(self, device_bytes: int | None) -> None:
        if device_bytes is not None:
            self.state.mem = device_bytes
            self.state.peak_mem = max(self.state.peak_mem, device_bytes)
        self.on_event(self.state, Memory(device_bytes))

    @final
    def on_fit_end(self) -> None:
        self.state.phase = "idle"
        self.on_event(self.state, FitEnd())

    @final
    def on_predict_start(self, n: int) -> None:
        self.state.phase = "predicting"
        self.state.predict_total = n
        self.on_event(self.state, PredictStart(n))

    @final
    def on_row_end(self, index: int, total: int) -> None:
        self.on_event(self.state, RowEnd(index, total))

    @final
    def on_predict_end(self) -> None:
        self.state.phase = "idle"
        self.on_event(self.state, PredictEnd())

    @final
    def on_generation(self, prompt: str, completion: str, target: str) -> None:
        self.on_event(self.state, Generation(prompt, completion, target))

    @final
    def on_score(
        self,
        prompts: Sequence[str],
        candidates: Sequence[object],
        probs: Sequence[Sequence[float]],
        logprobs: Sequence[Sequence[float]],
    ) -> None:
        self.on_event(self.state, Score(prompts, candidates, probs, logprobs))

    @final
    def on_retry(self, target: str, attempt: int, max_attempts: int) -> None:
        self.on_event(self.state, Retry(target, attempt, max_attempts))


def resolve_callbacks(callbacks: Callback | None) -> Callback:
    """Return ``callbacks`` or a no-op :class:`Callback` when ``None``."""
    return callbacks if callbacks is not None else Callback()


def predict_batches(callbacks: Callback, n_rows: int, batch_size: int) -> Iterator[tuple[int, int]]:
    """Yield ``(start, stop)`` row-batch bounds while driving the inference lifecycle.

    Emits ``on_predict_start`` once up front, then ``on_row_end`` for each row of
    a batch right after the caller finishes processing it (the body following the
    ``yield``) -- so progress advances batch by batch as inference runs rather
    than all at once at the end -- and ``on_predict_end`` when the rows are
    exhausted.
    """
    callbacks.on_predict_start(n_rows)
    for start in range(0, n_rows, max(batch_size, 1)):
        stop = min(start + batch_size, n_rows)
        yield start, stop
        for index in range(start, stop):
            callbacks.on_row_end(index, n_rows)
    callbacks.on_predict_end()


# --------------------------------------------------------------------------- helpers


def _format_bytes(n: int) -> str:
    """Render a byte count as a human-readable MiB/GiB string."""
    mib = n / 1024**2
    return f"{mib / 1024:.2f} GiB" if mib >= 1024 else f"{mib:.1f} MiB"


def _format_rate(steps_per_second: float) -> str:
    """Render step throughput tqdm-style: ``step/s`` when >= 1, else ``s/step``."""
    if steps_per_second >= 1:
        return f"{steps_per_second:.2f} step/s"
    return f"{1 / steps_per_second:.2f} s/step"


def _mean_std(values: Sequence[float]) -> str:
    """Render ``mean±std`` (population std) over ``values``."""
    return f"{statistics.fmean(values):.4f}±{statistics.pstdev(values):.4f}"


def _format_duration(seconds: float) -> str:
    """Render a duration as whole seconds (no minute/hour rollover)."""
    return f"{round(seconds)} sec"


def _format_params(training: TrainingConfig) -> str:
    """Render the headline training hyperparameters as a ``·``-joined string."""
    lr = training.learning_rate
    parts = [
        f"{training.epochs} epochs",
        f"bs={training.batch_size}",
        f"lr={lr}" if isinstance(lr, str) else f"lr={lr:.1e}",
        f"seq={training.max_seq_length or 'auto'}",
    ]
    if training.augmentation_factor > 1:
        parts.append(f"aug={training.augmentation_factor}")
    if training.validation_split > 0:
        parts.append(f"val={training.validation_split:g}")
    if training.loss_on_target_only:
        parts.append("target-only")
    return " · ".join(parts)


def _ranking(candidates: Sequence[object], probs: Sequence[float]) -> str:
    """Render ``candidate=prob`` pairs sorted by probability descending."""
    return ", ".join(
        f"{c}={p:.3f}"
        for c, p in sorted(zip(candidates, probs, strict=True), key=lambda cp: cp[1], reverse=True)
    )


def _spread_summary(candidates: Sequence[object], probs: Sequence[Sequence[float]]) -> str:
    """Per-candidate ``mean±std [min,max]`` across the per-order distributions.

    Summarizes how much each candidate's probability moved with the conditioning
    column order (the spread :class:`Score` exposes under ``permute_order``),
    sorted by mean descending.
    """
    parts: list[tuple[float, str]] = []
    for c, col in zip(candidates, zip(*probs, strict=True), strict=True):
        vals = list(col)
        mean = statistics.fmean(vals)
        summary = f"{c}={mean:.3f}±{statistics.pstdev(vals):.3f} [{min(vals):.3f},{max(vals):.3f}]"
        parts.append((mean, summary))
    return ", ".join(s for _, s in sorted(parts, key=lambda ms: ms[0], reverse=True))


def _score_detail(score: Score) -> str:
    """One-line summary of a :class:`Score`: ranking, or per-order spread when marginalizing."""
    if len(score.prompts) > 1:
        return f"({len(score.prompts)} orders) {_spread_summary(score.candidates, score.probs)}"
    return _ranking(score.candidates, score.probs[0])


_RESET = "\x1b[0m"
_DIM = "\x1b[2m"
_BOLD = "\x1b[1m"
_CYAN = "\x1b[36m"
_GREEN = "\x1b[32m"
_YELLOW = "\x1b[33m"

_START, _STEP, _DONE, _WARN = "╶", "·", "✓", "⚠"


def _supports_color(stream: object) -> bool:
    """Whether ANSI is safe on ``stream``: a real TTY, unless ``NO_COLOR`` is set."""
    if os.environ.get("NO_COLOR"):
        return False
    isatty = getattr(stream, "isatty", None)
    return bool(callable(isatty) and isatty())


# --------------------------------------------------------------------------- logging


class LoggingCallback(Callback):
    """Render every event through the standard :mod:`logging` module.

    A compact, colorized line per event: fine-tuning opens with a ``╶`` header
    (model + headline hyperparameters), each surviving training report is a ``·``
    line carrying the raw loss, the running epoch loss (``mean±std`` over the
    epoch so far), epoch, learning rate, progress, ETA and memory, and the run
    closes with a ``✓`` summary (steps and the ``last N`` loss ``mean±std``).
    Inference mirrors it (``╶`` start, the per-value generations / scored rankings,
    ``✓`` end). ANSI color is applied only when this instance auto-configures a
    console handler whose stream is a TTY (and ``NO_COLOR`` is unset); routed to a
    file or piped, the output stays plain.

    Parameters
    ----------
    level : int, optional
        Level for the fit/predict lifecycle, training reports and generation
        previews. Default ``logging.INFO``. Per-row and per-retry events log at
        ``DEBUG`` regardless of this value. When this instance auto-configures the
        logger (see Notes), this is also the logger's own threshold.
    logger : logging.Logger or None, optional
        Logger to emit through; ``None`` (default) uses ``getLogger("sklm")``.
    n_train_examples : int, optional
        How many serialized training examples to log at each epoch start. ``0``
        (default) logs none.
    n_predict_examples : int, optional
        How many per-value generations / per-row scores to log during a single
        ``predict`` / ``transform``. ``0`` (default) logs none; ``N`` logs only
        the first ``N``. The counter resets at each ``on_predict_start``.
    log_every : {"step", "epoch"}, optional
        The unit the training report is throttled in. Default ``"epoch"``.
    log_each : int or float, optional
        How often to emit a training report, in ``log_every`` units. An ``int``
        is an absolute count (every ``N`` steps / epochs); a ``float`` in
        ``(0, 1]`` is a fraction of the total (e.g. ``0.1`` with
        ``log_every="step"`` logs every 10 % of ``total_steps``; with
        ``log_every="epoch"`` every 10 % of the configured epochs). Default
        ``1`` (every reported step). A fractional step cadence needs a known
        ``total_steps``; when the run reports none, every report is logged.
        Memory rides on the surviving line; eval and lifecycle events are
        unaffected.
    loss_window : int, optional
        How many of the most recent training steps the ``last N`` loss
        ``mean±std`` in the ``✓`` fit-end summary averages over. Default ``20``.

    Notes
    -----
    Accelerator memory is not logged on its own line: it only updates the running
    current/peak in the state, which then rides along on the next ``step …`` line
    (``mem … ↑ peak``). So memory is throttled together with the training report
    rather than emitted per sample.

    Construction attaches a :class:`logging.StreamHandler` to the logger and sets
    its level, so a bare ``LoggingCallback()`` prints with no separate
    :mod:`logging` setup -- unless the logger already has a handler of its own, in
    which case it is left untouched.
    """

    def __init__(
        self,
        level: int = logging.INFO,
        logger: logging.Logger | None = None,
        n_train_examples: int = 0,
        n_predict_examples: int = 0,
        log_every: Literal["step", "epoch"] = "epoch",
        log_each: int | float = 1,
        loss_window: int = 20,
    ) -> None:
        super().__init__()
        if log_every not in ("step", "epoch"):
            raise ValueError(f"log_every must be 'step' or 'epoch', got {log_every!r}")
        if isinstance(log_each, float):
            if not 0.0 < log_each <= 1.0:
                raise ValueError(f"log_each as a fraction must be in (0, 1], got {log_each}")
        elif log_each < 1:
            raise ValueError(f"log_each as a count must be >= 1, got {log_each}")
        self._level = level
        self._logger = logger if logger is not None else logging.getLogger("sklm")
        self._n_train_examples = n_train_examples
        self._n_predict_examples = n_predict_examples
        self._log_every = log_every
        self._log_each = log_each
        self._loss_window = loss_window
        self._predict_logged = 0
        self._last_bucket = 0
        self._fit_t0: float | None = None
        self._color = self._configure_logger()

    def _configure_logger(self) -> bool:
        """Add a console handler unless the logger already has a non-null one.

        Returns whether that handler was added to a color-capable TTY -- the cue
        for whether events may carry ANSI. ``False`` when the caller owns the
        handlers (we stay out of their stream) or the stream is not a terminal.
        """
        if any(not isinstance(h, logging.NullHandler) for h in self._logger.handlers):
            return False
        handler = logging.StreamHandler()
        color = _supports_color(handler.stream)
        time = f"{_DIM}%(asctime)s{_RESET}" if color else "%(asctime)s"
        handler.setFormatter(logging.Formatter(f"{time} %(message)s", datefmt="%H:%M:%S"))
        self._logger.addHandler(handler)
        self._logger.setLevel(self._level)
        return color

    def _c(self, text: str, *codes: str) -> str:
        """Wrap ``text`` in the ANSI ``codes`` when color is on, else return it bare."""
        return f"{''.join(codes)}{text}{_RESET}" if self._color and codes else text

    def _kv(self, label: str, value: str, *codes: str) -> str:
        """A ``dim-label value`` metric cell; ``codes`` color the value."""
        return f"{self._c(label, _DIM)} {self._c(value, *codes)}"

    @override
    def on_event(self, state: TrainingState, event: Event) -> None:
        match event:
            case FitStart(n_rows=n_rows, n_epochs=n_epochs):
                self._last_bucket = 0
                self._fit_t0 = monotonic()
                params = (
                    _format_params(state.training)
                    if state.training is not None
                    else f"{n_epochs} epochs"
                )
                model = f"  {state.model}" if state.model else ""
                self._logger.log(
                    self._level,
                    "%s %s%s%s",
                    self._c(_START, _CYAN),
                    self._c("fine-tuning", _BOLD, _CYAN),
                    model,
                    self._c(f"  {n_rows} rows · {params}", _DIM),
                )
            case TrainExamples(examples=examples):
                for i, ex in enumerate(examples[: self._n_train_examples]):
                    self._logger.log(
                        self._level, "train example %d: %s", i + 1, self._c(ex.text, _DIM)
                    )
            case TrainReport(step=step, total_steps=total, epoch=epoch):
                self._log_train(state, step, total, epoch)
            case EvalReport(step=step, loss=loss):
                cells = [self._kv("eval step", str(step)), self._kv("loss", f"{loss:.4f}", _YELLOW)]
                self._logger.log(self._level, "%s %s", self._c(_STEP, _DIM), "  ".join(cells))
            case FitEnd():
                self._log_fit_end(state)
            case PredictStart(n=n):
                self._predict_logged = 0
                self._logger.log(
                    self._level,
                    "%s %s%s",
                    self._c(_START, _CYAN),
                    self._c("predicting", _BOLD, _CYAN),
                    self._c(f"  {n} rows", _DIM),
                )
            case RowEnd(index=index, total=total):
                self._logger.debug("%s row %d/%d", _STEP, index + 1, total)
            case PredictEnd():
                self._logger.log(
                    self._level,
                    "%s %s%s",
                    self._c(_DONE, _GREEN),
                    self._c("done", _GREEN),
                    self._c(f"  {state.predict_total} rows", _DIM),
                )
            case Generation(prompt=prompt, completion=completion):
                if self._predict_logged < self._n_predict_examples:
                    self._predict_logged += 1
                    self._logger.log(
                        self._level,
                        "%s prompt: %s\n  generated: %s",
                        self._c(_STEP, _DIM),
                        prompt,
                        self._c(completion, _GREEN),
                    )
            case Score():
                self._log_score(event)
            case Retry(target=target, attempt=attempt, max_attempts=max_attempts):
                self._logger.debug(
                    "%s retry %s: attempt %d/%d", _WARN, target, attempt, max_attempts
                )
            case FitInfo() | Memory():
                pass  # memory rides on the next train line via state

    def _fit_eta(self, step: int, total_steps: int) -> float | None:
        """Seconds left, extrapolated from the steps done so far (``None`` until timed)."""
        if self._fit_t0 is None or step <= 0:
            return None
        elapsed = monotonic() - self._fit_t0
        if elapsed <= 0:
            return None
        return max((total_steps - step) * elapsed / step, 0.0)

    def _log_train(
        self, state: TrainingState, step: int, total_steps: int | None, epoch: float | None
    ) -> None:
        bucket = self._cadence_bucket(step, total_steps, epoch, state.n_epochs)
        if bucket is not None:
            if bucket <= self._last_bucket:
                return
            self._last_bucket = bucket
        of = f"/{total_steps}" if total_steps is not None else ""
        cells = [self._kv("step", f"{step}{of}")]
        if state.loss is not None:
            cells.append(self._kv("loss", f"{state.loss:.4f}", _YELLOW))
        if state.epoch_losses:
            cells.append(self._kv("epoch loss", _mean_std(state.epoch_losses), _CYAN))
        if state.losses:
            window = _mean_std(state.recent_losses(self._loss_window))
            cells.append(self._kv(f"last{self._loss_window}", window, _CYAN))
        if state.epoch is not None:
            cells.append(self._kv("epoch", f"{state.epoch:.2f}/{state.n_epochs}"))
        if state.learning_rate is not None:
            cells.append(self._kv("lr", f"{state.learning_rate:.2e}"))
        if total_steps:
            cells.append(self._c(f"{step / total_steps * 100:.0f}%", _DIM))
            eta = self._fit_eta(step, total_steps)
            if eta is not None:
                cells.append(self._kv("eta", _format_duration(eta), _DIM))
        if state.mem is not None:
            mem = f"{_format_bytes(state.mem)}↑{_format_bytes(state.peak_mem)}"
            cells.append(self._kv("mem", mem, _DIM))
        self._logger.log(self._level, "%s %s", self._c(_STEP, _DIM), "  ".join(cells))

    def _log_fit_end(self, state: TrainingState) -> None:
        cells: list[str] = []
        if state.step is not None:
            cells.append(self._kv("steps", str(state.step)))
        if state.losses:
            window = state.recent_losses(self._loss_window)
            cells.append(self._kv(f"last{self._loss_window}", _mean_std(window)))
        if self._fit_t0 is not None:
            cells.append(self._c(_format_duration(monotonic() - self._fit_t0), _DIM))
        summary = f"  {'  '.join(cells)}" if cells else ""
        self._logger.log(
            self._level, "%s %s%s", self._c(_DONE, _GREEN), self._c("done", _GREEN), summary
        )

    def _log_score(self, score: Score) -> None:
        if self._predict_logged >= self._n_predict_examples:
            return
        self._predict_logged += 1
        sym = self._c(_STEP, _DIM)
        if len(score.prompts) > 1:
            self._logger.log(
                self._level,
                "%s prompt: %s\n  spread: (%d orders) %s",
                sym,
                score.prompts[0],
                len(score.prompts),
                _spread_summary(score.candidates, score.probs),
            )
            return
        self._logger.log(
            self._level,
            "%s prompt: %s\n  scored: %s",
            sym,
            score.prompts[0],
            self._c(_ranking(score.candidates, score.probs[0]), _GREEN),
        )

    def _cadence_bucket(
        self, step: int, total_steps: int | None, epoch: float | None, n_epochs: int
    ) -> int | None:
        """Index of the ``log_each``-sized interval this report falls in.

        A new bucket since the last one means "time to log". Returns ``None``
        when the cadence can't be resolved (a step fraction with no
        ``total_steps``, or an epoch cadence on a backend that reports neither
        ``epoch`` nor a step total) -- the caller then logs every report.
        """
        if self._log_every == "epoch":
            position = epoch
            if position is None and total_steps and n_epochs:
                position = step / total_steps * n_epochs
            if position is None:
                return None
            interval = (
                self._log_each * n_epochs
                if isinstance(self._log_each, float)
                else float(self._log_each)
            )
        else:
            position = float(step)
            if isinstance(self._log_each, float):
                if not total_steps:
                    return None
                interval = self._log_each * total_steps
            else:
                interval = float(self._log_each)
        if interval <= 0:
            return None
        return int((position + 1e-9) // interval)


# ----------------------------------------------------------------------------- rich


class _LossPlot:
    """A plotext loss curve sized to the Rich region it is drawn in.

    Reads the owner's live :class:`TrainingState` series on every render, so the
    curve grows as reports arrive. The eval series is drawn only when the backend
    reports one, so the plot stays agnostic to whether validation is enabled.
    """

    def __init__(self, owner: RichCallback) -> None:
        self._owner = owner

    def __rich_measure__(self, console: Console, options: ConsoleOptions) -> Measurement:
        from rich.measure import Measurement

        # Claim the whole available width so a grid cell expands the plot to fill.
        return Measurement(8, options.max_width)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        from rich.ansi import AnsiDecoder
        from rich.console import Group
        from rich.text import Text

        s = self._owner.state
        if not s.losses:
            yield Text("waiting for the first loss report…", style="dim")
            return
        width = options.max_width
        height = options.height or self._owner._plot_height
        if width < 8 or height < 3:
            yield Text(f"loss={s.losses[-1]:.4f}")
            return
        import plotext as plt  # plotext ships no type stubs (untyped third-party)

        plt.clf()
        plt.theme("clear")
        plt.plotsize(width, height)
        plt.plot(
            s.steps,
            s.losses,
            marker=self._owner._marker,
            color=self._owner._train_color,
            label="train",
        )
        if s.eval_losses:
            plt.scatter(
                s.eval_steps,
                s.eval_losses,
                marker=self._owner._marker,
                color=self._owner._eval_color,
                label="eval",
            )
        yield Group(*AnsiDecoder().decode(plt.build()))


class _Dynamic:
    """Renderable that re-evaluates ``render`` on every Live refresh."""

    def __init__(self, render: Callable[[], RenderableType]) -> None:
        self._render = render

    def __rich__(self) -> RenderableType:
        return self._render()


class RichCallback(Callback):
    """Render fitting and inference as a live :mod:`rich` dashboard.

    Requires the ``rich`` extra (``pip install scikit-lm[rich]``), which also
    pulls in ``plotext`` for the in-terminal loss curve. The layout is flat (no
    nested boxes): fine-tuning shows a header line (model name and headline
    hyperparameters), a progress bar, a live train-loss plot, a one-line metrics
    summary (loss, epoch, learning rate, memory) and -- when ``n_train_examples``
    is set -- the latest serialized training rows. Inference shows the header, a
    row progress bar and the last ``n_generations`` generations (or scored
    rankings, with the per-permutation spread when the classifier marginalizes
    over column orders). A single :class:`rich.live.Live` redraws everything at
    ``refresh_per_second`` straight from :attr:`Callback.state`.

    Parameters
    ----------
    n_train_examples : int, optional
        How many serialized training rows to show, sampled at each epoch start.
        ``0`` hides them. Default ``5``.
    n_generations : int, optional
        How many of the most recent generations/scores to keep in the inference
        section. Default ``8``.
    plot_height : int, optional
        Height in rows of the loss plot. Default ``12``.
    refresh_per_second : float, optional
        How often the live dashboard redraws. Default ``4``.
    marker : str, optional
        The plotext marker for the loss curve. A density mode (``"braille"``,
        ``"hd"``, ``"fhd"``, ``"dot"``, ``"sd"``) or any single character used as
        the point glyph. Default ``"hd"``.
    train_color, eval_color : str, optional
        plotext colors for the train and eval curves. Default ``"cyan"`` /
        ``"magenta"``.
    loss_window : int, optional
        How many of the most recent training steps the ``last N`` loss
        ``mean±std`` in the metrics summary averages over. Default ``20``.

    Notes
    -----
    Like :class:`TqdmCallback`, the live display captures the real ``sys.stdout``
    when fitting/predicting starts, so it keeps animating even while a backend
    redirects ``sys.stdout`` during training (as the MLX backend does).
    """

    def __init__(
        self,
        n_train_examples: int = 5,
        n_generations: int = 8,
        plot_height: int = 12,
        refresh_per_second: float = 4.0,
        marker: str = "hd",
        train_color: str = "cyan",
        eval_color: str = "magenta",
        loss_window: int = 20,
    ) -> None:
        super().__init__()
        # plotext and rich are provided by the optional 'rich' extra.
        try:
            import plotext  # noqa: F401
            import rich  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "RichCallback requires the 'rich' extra: pip install scikit-lm[rich]"
            ) from exc
        self._n_train_examples = n_train_examples
        self._n_generations = n_generations
        self._plot_height = plot_height
        self._rps = refresh_per_second
        self._marker = marker
        self._train_color = train_color
        self._eval_color = eval_color
        self._loss_window = loss_window
        self._generations: deque[tuple[str, str]] = deque(maxlen=n_generations)
        self._loss_plot = _LossPlot(self)
        self._console: Console | None = None
        self._live: Live | None = None
        self._progress: Progress | None = None
        self._task: TaskID | None = None

    @override
    def on_event(self, state: TrainingState, event: Event) -> None:
        match event:
            case FitStart():
                self._generations.clear()
                self._start_live(self._render_fit, total=None)
            case PredictStart(n=n):
                self._generations = deque(maxlen=self._n_generations)
                self._start_live(self._render_predict, total=n)
            case FitEnd() | PredictEnd():
                self._stop_live()
            case TrainReport(step=step, total_steps=total):
                if self._progress is not None and self._task is not None:
                    self._progress.update(self._task, completed=step, total=total)
            case RowEnd(index=index, total=total):
                if self._progress is not None and self._task is not None:
                    self._progress.update(self._task, completed=index + 1, total=total)
            case Generation(prompt=prompt, completion=completion, target=target):
                self._generations.append((prompt, f"{target}={completion}"))
            case Score():
                self._generations.append((event.prompts[0], _score_detail(event)))
            case _:
                pass

    def _start_live(self, render: Callable[[], RenderableType], total: int | None) -> None:
        from rich.console import Console
        from rich.live import Live

        self._console = Console(file=sys.stdout)
        self._make_progress(total=total)
        self._live = Live(_Dynamic(render), console=self._console, refresh_per_second=self._rps)
        self._live.start()

    def _make_progress(self, total: int | None) -> None:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TaskProgressColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        self._progress = Progress(
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        self._task = self._progress.add_task("", total=total)

    def _stop_live(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
        self._progress = None
        self._task = None

    def _current_task(self) -> Task | None:
        """The active :class:`rich.progress.Task`, source of progress/elapsed/eta."""
        if self._progress is None:
            return None
        return next((t for t in self._progress.tasks if t.id == self._task), None)

    def _current_speed(self) -> float | None:
        """Steps per second for the active task, or ``None`` until rich can estimate it."""
        task = self._current_task()
        return task.speed if task is not None else None

    def _header(self, phase: str, extra: str) -> RenderableType:
        from rich.text import Text

        line = Text()
        line.append(phase, style="bold cyan")
        if self.state.model:
            line.append("  ")
            line.append(self.state.model)
        if extra:
            line.append("  ")
            line.append(extra, style="dim")
        return line

    def _metrics_grid(self) -> RenderableType:
        from rich.table import Table

        s = self.state
        grid = Table.grid(padding=(0, 1))
        grid.add_column(justify="right", style="dim")
        grid.add_column()
        grid.add_row("rows", str(s.n_rows))
        task = self._current_task()
        if task is not None:
            if task.total is not None:
                grid.add_row("progress", f"{task.percentage:.0f}%")
            if task.elapsed is not None:
                grid.add_row("elapsed", _format_duration(task.elapsed))
            if task.time_remaining is not None:
                grid.add_row("eta", _format_duration(task.time_remaining))
        if s.epoch_losses:
            grid.add_row("epoch loss", _mean_std(s.epoch_losses))
        if s.losses:
            grid.add_row(f"last {self._loss_window}", _mean_std(s.recent_losses(self._loss_window)))
        if s.loss is not None:
            grid.add_row("loss", f"{s.loss:.4f}")
        if s.eval_loss is not None:
            grid.add_row("eval", f"{s.eval_loss:.4f}")
        if s.epoch is not None:
            grid.add_row("epoch", f"{s.epoch:.2f}/{s.n_epochs}")
        if speed := self._current_speed():
            grid.add_row("speed", _format_rate(speed))
        if s.learning_rate is not None:
            grid.add_row("lr", f"{s.learning_rate:.2e}")
        if s.mem is not None:
            grid.add_row("mem", _format_bytes(s.mem))
            grid.add_row("peak", _format_bytes(s.peak_mem))
        return grid

    def _section(self, title: str, lines: Sequence[RenderableType]) -> RenderableType:
        from rich.console import Group
        from rich.text import Text

        return Group(Text(title, style="dim italic"), *lines)

    def _render_fit(self) -> RenderableType:
        from rich.console import Group
        from rich.table import Table
        from rich.text import Text

        s = self.state
        if self._progress is None:
            return Text("")
        body = Table.grid(expand=True, padding=(0, 2))
        body.add_column(ratio=3)
        body.add_column(ratio=1, justify="left", vertical="middle")
        body.add_row(self._loss_plot, self._metrics_grid())
        params = _format_params(s.training) if s.training is not None else ""
        blocks: list[RenderableType] = [
            self._header("fine-tuning", params),
            Text(""),
            body,
        ]
        if self._n_train_examples and s.examples:
            rows = [
                Text(f"  {ex.text}", style="dim", no_wrap=True, overflow="ellipsis")
                for ex in s.examples[: self._n_train_examples]
            ]
            blocks += [
                Text(""),
                self._section(f"training examples · epoch {s.example_epoch}", rows),
            ]
        return Group(*blocks)

    def _render_predict(self) -> RenderableType:
        from rich.console import Group
        from rich.text import Text

        if self._progress is None:
            return Text("")
        rows: list[RenderableType] = []
        if self._generations:
            for prompt, detail in self._generations:
                rows.append(Text(f"  {prompt}", no_wrap=True, overflow="ellipsis"))
                rows.append(Text(f"    {detail}", style="dim"))
        else:
            rows.append(Text("  waiting…", style="dim"))
        return Group(
            self._header("predicting", f"{self.state.predict_total} rows"),
            self._progress,
            Text(""),
            self._section("recent predictions", rows),
        )


# ----------------------------------------------------------------------------- tqdm


class TqdmCallback(Callback):
    """Render fitting and inference as live ``tqdm`` progress bars.

    Requires the ``tqdm`` extra (``pip install scikit-lm[tqdm]``). One bar per
    phase: fine-tuning advances by training step, inference by row. The bar
    itself carries the step count, ETA and steps-per-second; the live loss,
    epoch and learning rate ride along in the postfix.

    Parameters
    ----------
    n_train_examples : int, optional
        How many serialized training examples to print above the bar at each
        epoch start. ``0`` (default) prints none.

    Notes
    -----
    Each bar captures the real ``sys.stdout`` when it starts, so it keeps
    animating even while a backend redirects ``sys.stdout`` during training (as
    the MLX backend does).
    """

    def __init__(self, n_train_examples: int = 0) -> None:
        super().__init__()
        try:
            from tqdm import tqdm
        except ImportError as exc:
            raise ImportError(
                "TqdmCallback requires the 'tqdm' extra: pip install scikit-lm[tqdm]"
            ) from exc
        self._tqdm = tqdm
        self._bar: Any = None
        self._out: Any = None
        self._n_train_examples = n_train_examples

    @override
    def on_event(self, state: TrainingState, event: Event) -> None:
        match event:
            case FitStart():
                self._start("fine-tuning", total=None, unit="step")
            case TrainExamples(examples=examples):
                self._tqdm.write("[training examples]")
                for ex in examples[: self._n_train_examples]:
                    self._tqdm.write(f"  > {ex.text}", file=self._out)
            case TrainReport(step=step, total_steps=total):
                self._update_fit(state, step, total)
            case FitEnd() | PredictEnd():
                self._stop()
            case PredictStart(n=n):
                self._start("predicting", total=n, unit="row")
            case RowEnd(index=index, total=total):
                if self._bar is not None:
                    self._advance(index + 1, total)
            case Generation(prompt=prompt, completion=completion, target=target):
                self._tqdm.write(
                    f"[generated]\n  > prompt={prompt}\n  > completion={completion}"
                    f"\n  > target={target}",
                    file=self._out,
                )
            case Score():
                self._write_score(event)
            case _:
                pass

    def _start(self, description: str, total: int | None, unit: str) -> None:
        self._out = sys.stdout
        self._bar = self._tqdm(
            total=total, desc=description, unit=unit, file=self._out, dynamic_ncols=True
        )

    def _advance(self, target: int, total: int | None) -> None:
        if total is not None and self._bar.total != total:
            self._bar.total = total
        delta = target - self._bar.n
        if delta > 0:
            self._bar.update(delta)

    def _stop(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def _update_fit(self, state: TrainingState, step: int, total_steps: int | None) -> None:
        if self._bar is None:
            return
        postfix: dict[str, str] = {"loss": f"{state.loss:.4f}"}
        if state.eval_loss is not None:
            postfix["eval"] = f"{state.eval_loss:.4f}"
        if state.epoch is not None:
            postfix["epoch"] = f"{state.epoch:.2f}/{state.n_epochs}"
        if state.learning_rate is not None:
            postfix["lr"] = f"{state.learning_rate:.2e}"
        if state.mem is not None:
            postfix["mem"] = _format_bytes(state.mem)
            postfix["peak"] = _format_bytes(state.peak_mem)
        self._bar.set_postfix(postfix, refresh=False)
        self._advance(step, total_steps)

    def _write_score(self, score: Score) -> None:
        if len(score.prompts) > 1:
            block = (
                f"[scored] ({len(score.prompts)} orders)\n  > prompt={score.prompts[0]}"
                f"\n  > spread={_spread_summary(score.candidates, score.probs)}"
            )
        else:
            block = (
                f"[scored]\n  > prompt={score.prompts[0]}"
                f"\n  > ranking={_ranking(score.candidates, score.probs[0])}"
            )
        self._tqdm.write(block, file=self._out)
