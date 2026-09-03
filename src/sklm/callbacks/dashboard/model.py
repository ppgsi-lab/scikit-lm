from __future__ import annotations

import math
import statistics
from collections import deque
from time import monotonic
from typing import Any, Literal

from ..events import (
    EvalReport,
    Event,
    FitEnd,
    FitStart,
    Generation,
    PredictStart,
    Retry,
    RowEnd,
    Score,
    TrainingState,
    TrainReport,
)
from ..format import _ranking_pairs

# Categorical hues for candidate classes / bins in the verdict view, assigned on
# first sighting. The renderer maps a candidate's color through this same cursor,
# so the palette itself lives here (renderer-agnostic) rather than in the markup.
_PALETTE = (
    "#3b6cf6",
    "#1f9e84",
    "#c6841b",
    "#9b5de5",
    "#e26d5c",
    "#2a9d8f",
    "#e8a33d",
    "#577590",
    "#bc4749",
    "#4d908e",
)


class DashboardState:
    """The shared accumulator folding events into the dashboards' auxiliary state.

    The :class:`TrainingState` carried on every event holds the loss series and the
    headline metrics; this holds everything the dashboards derive on top of it: the
    structured generation/score records (kept structured, not pre-formatted, so a
    Rich one-line detail and a Jupyter widget can both be built from them), the
    aggregated training-log rows plus the open interval, the eval-by-step map (the
    early-stopping trackers -- best validation loss, its epoch and the patience
    counter -- live on the :class:`TrainingState` itself), the verdict aggregates (count,
    confidence sum, uncertain count, bin counts and the candidate colors), the
    malformed-generation retry count and the progress cursor (done / total / start
    time) with the log-bucket cursor.

    Both live callbacks (:class:`JupyterCallback` and :class:`RichCallback`) fold
    events through this same accumulator; :class:`build.build_fit_dashboard` /
    :func:`build.build_predict_dashboard` then fold it together with the
    :class:`TrainingState` into the renderer-agnostic :class:`widgets.Dashboard`
    each renderer walks.

    Parameters
    ----------
    n_generations : int
        Window of most-recent predictions kept for the inference table.
    uncertain_threshold : float
        Top-1 probability below which a scored row counts as uncertain.
    log_every : int or float or {"epoch"}
        Cadence at which an interval is closed into a training-log row.
    """

    def __init__(
        self,
        *,
        n_generations: int,
        uncertain_threshold: float,
        log_every: int | float | Literal["epoch"],
    ) -> None:
        self._n_generations = n_generations
        self.uncertain_threshold = uncertain_threshold
        self._log_every = log_every
        self.generations: deque[dict[str, Any]] = deque(maxlen=n_generations)
        self.score_count = 0
        self.conf_sum = 0.0
        self.uncertain = 0
        self.bin_counts: dict[object, int] = {}
        self.cand_colors: dict[object, str] = {}
        self.retries = 0
        self.log: list[dict[str, Any]] = []
        self.pending: list[dict[str, Any]] = []
        self.eval_by_step: dict[int, float] = {}
        self.cur_bucket: int | None = None
        self.done = 0
        self.total: int | None = None
        self.t0: float | None = None

    def color_for(self, candidate: object) -> str:
        """Stable categorical color for a candidate, assigned on first sighting."""
        if candidate not in self.cand_colors:
            self.cand_colors[candidate] = _PALETTE[len(self.cand_colors) % len(_PALETTE)]
        return self.cand_colors[candidate]

    def fold(self, state: TrainingState, event: Event) -> None:
        """Fold one ``event`` into the accumulator.

        Mutates the live containers in place; the owning callback decides when to
        flush the open interval (at :class:`FitEnd`) and when to rebuild the markup.
        """
        match event:
            case FitStart():
                self.generations.clear()
                self.log = []
                self.pending = []
                self.eval_by_step = {}
                self.cur_bucket = None
                self.done, self.total, self.t0 = 0, None, monotonic()
            case PredictStart(n=n):
                self.generations = deque(maxlen=self._n_generations)
                self.score_count, self.conf_sum, self.uncertain = 0, 0.0, 0
                self.bin_counts = {}
                self.cand_colors = {}
                self.retries = 0
                self.done, self.total, self.t0 = 0, n, monotonic()
            case FitEnd():
                self.flush_log()
            case TrainReport(step=step, total_steps=total, grad_norm=grad_norm):
                self.done, self.total = step, total
                bucket = self._log_bucket(step, total, state.epoch, state.n_epochs)
                if self.pending and bucket != self.cur_bucket:
                    self.flush_log()
                self.cur_bucket = bucket
                self.pending.append({
                    "step": step,
                    "epoch": state.epoch,
                    "loss": state.loss,
                    "lr": state.learning_rate,
                    "grad_norm": grad_norm,
                    "t": monotonic(),
                })
            case EvalReport(step=step, loss=loss):
                self.eval_by_step[step] = loss
            case RowEnd(index=index, total=total):
                self.done, self.total = index + 1, total
            case Generation(prompt=prompt, target=target, value=value):
                self.generations.append({
                    "kind": "gen",
                    "prompt": prompt,
                    "target": target,
                    "value": value,
                })
            case Score(prompts=prompts, candidates=candidates, probs=probs):
                ranking = _ranking_pairs(candidates, probs)
                _, top_p = ranking[0]
                self.score_count += 1
                self.conf_sum += top_p
                if top_p < self.uncertain_threshold:
                    self.uncertain += 1
                top_label = ranking[0][0]
                self.bin_counts[top_label] = self.bin_counts.get(top_label, 0) + 1
                for candidate in candidates:
                    self.color_for(candidate)
                self.generations.append({"kind": "score", "prompt": prompts[0], "ranking": ranking})
            case Retry():
                self.retries += 1
            case _:
                pass

    def _log_bucket(
        self, step: int, total_steps: int | None, epoch: float | None, n_epochs: int
    ) -> int | None:
        """The ``log_every`` interval index this report falls in.

        A change from the previous report's bucket closes the current row. Returns
        ``None`` when the cadence cannot be resolved (a fractional or epoch cadence
        with no known total) -- a ``None`` always differs from the stored bucket,
        so every report becomes its own row.
        """
        le = self._log_every
        if le == "epoch":
            position = epoch
            if position is None and total_steps and n_epochs:
                position = step / total_steps * n_epochs
            return math.ceil(position - 1e-9) if position is not None else None
        if isinstance(le, float):
            interval = le * total_steps if total_steps else None
        else:
            interval = float(le)
        return int((step - 1 + 1e-9) // interval) if interval and interval > 0 else None

    def flush_log(self) -> None:
        """Summarize the open interval's reports into one training-log row."""
        if not self.pending:
            return
        losses = [r["loss"] for r in self.pending if r["loss"] is not None]
        grads = [r["grad_norm"] for r in self.pending if r["grad_norm"] is not None]
        last = self.pending[-1]
        self.log.append({
            "step": last["step"],
            "epoch": last["epoch"],
            "loss": statistics.fmean(losses) if losses else None,
            "std": statistics.pstdev(losses) if len(losses) > 1 else None,
            "lr": last["lr"],
            "grad_norm": statistics.fmean(grads) if grads else None,
            "t": last["t"],
        })
        self.pending = []

    def eta(self) -> float | None:
        """Estimated seconds remaining from the rate so far, or ``None`` if unknown."""
        if self.t0 is None or not self.total or self.done <= 0:
            return None
        elapsed = monotonic() - self.t0
        if elapsed <= 0:
            return None
        return max((self.total - self.done) * elapsed / self.done, 0.0)
