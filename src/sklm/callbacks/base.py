from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, final, override

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

if TYPE_CHECKING:
    from ..config import TrainingConfig
    from ..serialize import TrainingExample

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
        self._announced_epoch: int | None = None

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
        self._announced_epoch = None
        self.on_event(self.state, FitStart(n_rows, n_epochs))

    @final
    def on_train_examples(self, examples: Sequence[TrainingExample], epoch: int) -> None:
        # ``epoch`` is 0-indexed and may repeat: the backend requests epoch 0 once
        # to measure sequence length and once to seed the dataset. The generator is
        # idempotent per epoch, so drop the duplicate instead of re-announcing it.
        if epoch == self._announced_epoch:
            return
        self._announced_epoch = epoch
        self.state.example_epoch = epoch + 1
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
        grad_norm: float | None = None,
    ) -> None:
        s = self.state
        # MLX does not report epoch; derive it from step progress instead.
        if epoch is None and total_steps:
            epoch = step / total_steps * s.n_epochs
        s.step, s.loss, s.epoch, s.learning_rate = step, loss, epoch, learning_rate
        s.grad_norm = grad_norm
        s.steps.append(step)
        s.losses.append(loss)
        if epoch is not None:
            bucket = math.ceil(epoch)
            if bucket != self._epoch_bucket:
                self._epoch_bucket = bucket
                s.epoch_losses = []
            s.epoch_losses.append(loss)
        self.on_event(s, TrainReport(step, total_steps, loss, epoch, learning_rate, grad_norm))

    @final
    def on_eval_report(self, step: int, loss: float, epoch: float | None) -> None:
        s = self.state
        s.eval_loss = loss
        s.eval_steps.append(step)
        s.eval_losses.append(loss)
        if s.best_eval is None or loss < s.best_eval:
            s.best_eval = loss
            # MLX reports no epoch on eval; fall back to the one derived from steps.
            s.best_eval_epoch = epoch if epoch is not None else s.epoch
            s.evals_since_best = 0
        else:
            s.evals_since_best += 1
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
    def on_generation(
        self, prompt: str, completion: str, target: str, value: object | None = None
    ) -> None:
        self.on_event(self.state, Generation(prompt, completion, target, value))

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


class CompositeCallback(Callback):
    """Fan one event stream out to several callbacks.

    Aggregates the :class:`TrainingState` once -- through the inherited
    ``@final`` granular methods -- then dispatches every event to each wrapped
    callback, sharing that aggregated snapshot so they all render from the same
    state. Estimators build one automatically when handed a list of callbacks.

    Parameters
    ----------
    callbacks : sequence of Callback
        The callbacks to drive, in order.
    """

    def __init__(self, callbacks: Sequence[Callback]) -> None:
        super().__init__()
        self.callbacks = list(callbacks)

    @override
    def on_event(self, state: TrainingState, event: Event) -> None:
        for callback in self.callbacks:
            callback.state = state
            callback.on_event(state, event)


def predict_batches(callback: Callback, n_rows: int, batch_size: int) -> Iterator[tuple[int, int]]:
    """Yield ``(start, stop)`` row-batch bounds while driving the inference lifecycle.

    Emits ``on_predict_start`` once up front, then ``on_row_end`` for each row of
    a batch right after the caller finishes processing it (the body following the
    ``yield``) -- so progress advances batch by batch as inference runs rather
    than all at once at the end -- and ``on_predict_end`` when the rows are
    exhausted.
    """
    callback.on_predict_start(n_rows)
    for start in range(0, n_rows, max(batch_size, 1)):
        stop = min(start + batch_size, n_rows)
        yield start, stop
        for index in range(start, stop):
            callback.on_row_end(index, n_rows)
    callback.on_predict_end()
