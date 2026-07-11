from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from time import monotonic
from typing import Literal

from ..events import TrainingState
from ..format import _format_bytes, _mean_std
from .model import DashboardState
from .widgets import (
    Badge,
    BinShare,
    Dashboard,
    ExamplesPanel,
    GenerationRow,
    Header,
    KeyValue,
    KeyValues,
    LogRow,
    LogTable,
    LossCurve,
    MetricRow,
    Metrics,
    PredictionRow,
    Predictions,
    ScoreRow,
    Stat,
    StatCards,
    Widget,
)


@dataclass(frozen=True)
class RenderConfig:
    """Display knobs the builder reads to shape the dashboard tree.

    These are the data-shaping choices (how many examples, which view, the log
    window and cadence); the visual ones (smoothing, colors) live on the renderer.

    Parameters
    ----------
    n_train_examples : int
        How many serialized training rows the examples panel shows.
    examples_view : {"table", "raw"}
        The requested examples view; ``"table"`` falls back to ``"raw"`` for
        non-JSON serializers.
    n_log_rows : int
        How many of the most recent log rows the training-log table shows.
    log_every : int or float or {"epoch"}
        The configured log cadence, rendered into the table's cadence label.
    uid : str
        Per-instance id scoping the examples-panel CSS tabs.
    loss_window : int
        How many recent steps the :class:`Metrics` ``last N`` row averages over.
        Read only by renderers that show the compact summary (the Rich one).
    """

    n_train_examples: int
    examples_view: Literal["table", "raw"]
    n_log_rows: int
    log_every: int | float | Literal["epoch"]
    uid: str
    loss_window: int = 20


# --------------------------------------------------------------------------- format


def _sci(value: float) -> str:
    """Format a learning rate compactly: ``0.0002 -> '2e-4'``, ``0.00018 -> '1.8e-4'``."""
    mantissa, exponent = f"{value:.1e}".split("e")
    if mantissa.endswith(".0"):
        mantissa = mantissa[:-2]
    return f"{mantissa}e{int(exponent)}"


def _metric(value: float | None) -> str:
    """A four-decimal metric, or an em dash when the value is not yet known."""
    return f"{value:.4f}" if value is not None else "—"


def _clock(seconds: float) -> str:
    """Render an elapsed duration as ``HH:MM:SS``."""
    whole = int(seconds)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _eta_label(seconds: float) -> str:
    """A coarse 'time left' label, e.g. ``~12m left`` or ``~1h 05m left``."""
    whole = int(seconds)
    if whole < 60:
        return f"~{whole}s left"
    minutes, _ = divmod(whole, 60)
    if minutes < 60:
        return f"~{minutes}m left"
    hours, minutes = divmod(minutes, 60)
    return f"~{hours}h {minutes:02d}m left"


def _json_rows(texts: Sequence[str]) -> tuple[list[str], list[dict[str, object]]] | None:
    """Parse serialized JSON-object rows into ``(columns, rows)``.

    Columns are the union of keys in first-seen order. Returns ``None`` when any
    text is not a JSON object, so the caller can fall back to the raw view for
    non-JSON serializers.
    """
    rows: list[dict[str, object]] = []
    columns: list[str] = []
    for text in texts:
        try:
            obj = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(obj, dict):
            return None
        rows.append(obj)
        for key in obj:
            if key not in columns:
                columns.append(key)
    return columns, rows


# --------------------------------------------------------------------------- pieces


def _step_stat(label: str, dash: DashboardState) -> Stat:
    suffix = f" / {dash.total:,}" if dash.total else None
    return Stat(label, f"{dash.done:,}", suffix=suffix)


def _badge(phase: str) -> Badge:
    if phase in ("predicting", "fitting"):
        return Badge("predicting" if phase == "predicting" else "fine-tuning", "active")
    return Badge("done", "done")


def _header(state: TrainingState, dash: DashboardState) -> Header:
    if state.phase == "predicting":
        title, sub = state.model or "predicting", f"{state.predict_total} rows"
    else:
        title, sub = state.model or "fine-tuning", f"{state.n_rows} rows"
    return Header(title, sub, _badge(state.phase))


def _fit_cards(state: TrainingState, dash: DashboardState) -> StatCards:
    lr = _sci(state.learning_rate) if state.learning_rate is not None else "—"
    return StatCards([
        Stat("train loss", _metric(state.loss)),
        Stat("val loss", _metric(state.eval_loss)),
        Stat("learning rate", lr, mono=True),
        _step_stat("step", dash),
    ])


def _loss_curve(state: TrainingState) -> LossCurve:
    parts = ["train", *(["val"] if state.eval_losses else [])]
    return LossCurve(
        steps=list(state.steps),
        train=list(state.losses),
        eval_steps=list(state.eval_steps),
        evals=list(state.eval_losses),
        subtitle=" · ".join(parts),
    )


def _fit_metrics(state: TrainingState, cfg: RenderConfig) -> Metrics:
    rows: list[MetricRow] = [MetricRow("rows", str(state.n_rows))]
    if state.epoch_losses:
        rows.append(MetricRow("epoch loss", _mean_std(state.epoch_losses)))
    if state.losses:
        rows.append(
            MetricRow(f"last {cfg.loss_window}", _mean_std(state.recent_losses(cfg.loss_window)))
        )
    if state.loss is not None:
        rows.append(MetricRow("loss", f"{state.loss:.4f}"))
    if state.eval_loss is not None:
        rows.append(MetricRow("eval", f"{state.eval_loss:.4f}"))
    if state.epoch is not None:
        rows.append(MetricRow("epoch", f"{state.epoch:.2f}/{state.n_epochs}"))
    if state.learning_rate is not None:
        rows.append(MetricRow("lr", f"{state.learning_rate:.2e}"))
    if state.mem is not None:
        rows.append(MetricRow("mem", _format_bytes(state.mem)))
        rows.append(MetricRow("peak", _format_bytes(state.peak_mem)))
    return Metrics(rows)


def _hyperparameters(state: TrainingState, dash: DashboardState) -> KeyValues:
    rows: list[KeyValue] = []
    if state.model:
        rows.append(KeyValue("base model", state.model, mono=True))
    t = state.training
    if t is not None:
        lr = t.lr_scheduler.learning_rate
        rows.append(KeyValue("peak lr", lr if isinstance(lr, str) else _sci(lr), mono=True))
        rows.append(KeyValue("batch size", str(t.batch_size)))
        rows.append(KeyValue("epochs", str(t.epochs)))
        rows.append(KeyValue("seq length", str(t.max_seq_length or "auto")))
        if t.augmentation_factor > 1:
            rows.append(KeyValue("augmentation", f"×{t.augmentation_factor}"))
        if t.validation_split > 0:
            rows.append(KeyValue("validation", f"{t.validation_split:g}"))
    elapsed = eta = None
    if dash.t0 is not None:
        elapsed = _clock(monotonic() - dash.t0)
        seconds = dash.eta()
        eta = _eta_label(seconds) if seconds is not None else None
    return KeyValues("hyperparameters", rows, elapsed=elapsed, eta=eta)


def _examples_panel(state: TrainingState, cfg: RenderConfig) -> ExamplesPanel:
    texts = [ex.text for ex in state.examples[: cfg.n_train_examples]]
    parsed = _json_rows(texts)
    return ExamplesPanel(
        epoch=state.example_epoch, texts=texts, parsed=parsed, view=cfg.examples_view, uid=cfg.uid
    )


def _cadence(log_every: int | float | Literal["epoch"]) -> str:
    if log_every == "epoch":
        return "every epoch"
    if isinstance(log_every, float):
        return f"every {log_every:g} of training"
    return "every step" if log_every == 1 else f"every {log_every} steps"


def _log_table(dash: DashboardState, cfg: RenderConfig) -> LogTable:
    start = max(len(dash.log) - cfg.n_log_rows, 0)
    eval_steps = sorted(dash.eval_by_step)
    rows: list[LogRow] = []
    for i in reversed(range(start, len(dash.log))):
        r = dash.log[i]
        prev = dash.log[i - 1] if i > 0 else None
        sps = None
        if prev is not None:
            dt = r["t"] - prev["t"]
            dstep = r["step"] - prev["step"]
            if dt > 0 and dstep > 0:
                sps = dstep / dt
        lo = prev["step"] if prev is not None else -1
        within = [s for s in eval_steps if lo < s <= r["step"]]
        eval_loss = dash.eval_by_step[within[-1]] if within else None
        rows.append(
            LogRow(
                step=r["step"],
                epoch=r["epoch"],
                loss=r["loss"],
                std=r["std"],
                lr=r["lr"],
                grad_norm=r["grad_norm"],
                eval_loss=eval_loss,
                steps_per_second=sps,
            )
        )
    return LogTable(rows, _cadence(cfg.log_every))


def _prediction_rows(dash: DashboardState) -> list[PredictionRow]:
    # Dispatch per record, not on a global flag: a discretized estimator scores
    # some columns and generates others within one predict phase, so the window
    # can hold both kinds at once.
    rows: list[PredictionRow] = []
    for g in dash.generations:
        if g["kind"] == "score":
            ranking: list[tuple[object, float]] = g["ranking"]
            _, top_p = ranking[0]
            rows.append(ScoreRow(g["prompt"], ranking, uncertain=top_p < dash.uncertain_threshold))
        else:
            rows.append(GenerationRow(g["prompt"], g["value"]))
    return rows


def _bins(dash: DashboardState) -> tuple[list[BinShare], str]:
    mix = sorted(dash.bin_counts.items(), key=lambda kv: kv[1], reverse=True)
    label = "predicted bins"
    if len(mix) > 5:
        head = mix[:5]
        others = sum(n for _, n in mix[5:])
        mix = [*head, ("others", others)]
        label += " · top 5"
    return [BinShare(c, n) for c, n in mix], label


def _predictions(dash: DashboardState) -> Predictions:
    scoring = dash.score_count > 0
    bins, bins_label = _bins(dash) if scoring else ([], "predicted bins")
    return Predictions(
        scoring=scoring,
        rows=_prediction_rows(dash),
        bins=bins,
        bins_label=bins_label,
        retries=dash.retries,
    )


# --------------------------------------------------------------------------- public


def build_fit_dashboard(state: TrainingState, dash: DashboardState, cfg: RenderConfig) -> Dashboard:
    """Assemble the fine-tuning dashboard tree.

    Header, stat cards, the loss curve, the compact metrics summary and the
    hyperparameters come first; the examples panel and the training-log table
    follow when configured and populated. Pure: no HTML and no I/O.

    Parameters
    ----------
    state : TrainingState
        The running fit snapshot (loss series, headline metrics, examples).
    dash : DashboardState
        The auxiliary accumulator (log rows, eval-by-step, progress).
    cfg : RenderConfig
        The data-shaping knobs.

    Returns
    -------
    Dashboard
        The fit tree.
    """
    children: list[Widget] = [
        _header(state, dash),
        _fit_cards(state, dash),
        _loss_curve(state),
        _fit_metrics(state, cfg),
        _hyperparameters(state, dash),
    ]
    if cfg.n_train_examples and state.examples:
        children.append(_examples_panel(state, cfg))
    if cfg.n_log_rows and dash.log:
        children.append(_log_table(dash, cfg))
    return Dashboard(children)


def build_predict_dashboard(
    state: TrainingState, dash: DashboardState, cfg: RenderConfig
) -> Dashboard:
    """Assemble the inference dashboard tree.

    A header, the verdict/step stat cards and the predictions table. Pure: no HTML
    and no I/O.

    Parameters
    ----------
    state : TrainingState
        The running predict snapshot.
    dash : DashboardState
        The auxiliary accumulator (generations, verdict aggregates, progress).
    cfg : RenderConfig
        The data-shaping knobs (unused for predict beyond parity with the fit
        signature).

    Returns
    -------
    Dashboard
        The predict tree.
    """
    predictions = _predictions(dash)
    if dash.score_count:
        mean_conf = dash.conf_sum / dash.score_count if dash.score_count else 0.0
        cards = StatCards([
            _step_stat("predictions", dash),
            Stat("mean top-1 conf", f"{mean_conf:.2f}", mono=True),
            Stat("uncertain rows", f"{dash.uncertain} / {dash.score_count}"),
        ])
    else:
        cards = StatCards([_step_stat("rows", dash)])
    children: list[Widget] = [_header(state, dash), cards, predictions]
    return Dashboard(children)
