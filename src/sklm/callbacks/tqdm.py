from __future__ import annotations

import sys
from typing import Any, override

from .base import Callback
from .events import (
    Event,
    FitEnd,
    FitStart,
    Generation,
    PredictEnd,
    PredictStart,
    RowEnd,
    Score,
    TrainExamples,
    TrainingState,
    TrainReport,
)
from .format import _format_bytes, _ranking, _spread_summary

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
