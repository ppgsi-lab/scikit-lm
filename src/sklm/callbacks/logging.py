from __future__ import annotations

import logging
import os
from time import monotonic
from typing import Literal, override

from .base import Callback
from .events import (
    EvalReport,
    Event,
    FitEnd,
    FitInfo,
    FitStart,
    Generation,
    Memory,
    PredictEnd,
    PredictStart,
    Retry,
    RowEnd,
    Score,
    TrainExamples,
    TrainingState,
    TrainReport,
)
from .format import (
    _format_bytes,
    _format_duration,
    _format_params,
    _mean_std,
    _ranking,
    _spread_summary,
)

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
    epoch so far), epoch, learning rate, progress, ETA and memory, each validation
    report is a ``·`` line with the eval loss, the best so far (and the epoch it
    was reached at) and the patience counter when ``EvalConfig.patience`` is
    set, and the run closes with a ``✓`` summary (steps and the ``last N`` loss
    ``mean±std``).
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
        # Our handler renders the full line; propagating to the root logger would
        # print every event a second time under any ``logging.basicConfig``.
        self._logger.propagate = False
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
                if state.best_eval is not None:
                    best = f"{state.best_eval:.4f}"
                    if state.best_eval_epoch is not None:
                        best += f" @ep {state.best_eval_epoch:.0f}"
                    cells.append(self._kv("best", best, _GREEN))
                evaluation = state.training.evaluation if state.training is not None else None
                patience = evaluation.patience if evaluation is not None else None
                if patience is not None:
                    cells.append(self._kv("patience", f"{state.evals_since_best}/{patience}"))
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
