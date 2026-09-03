from __future__ import annotations

from time import monotonic
from typing import Any, Literal, override

from .base import Callback
from .dashboard import (
    DashboardState,
    HtmlRenderer,
    RenderConfig,
    build_fit_dashboard,
    build_predict_dashboard,
)
from .dashboard.html_renderer import _JUP_INFO, _JUP_TK_KEY
from .events import Event, FitEnd, FitStart, PredictEnd, PredictStart, TrainExamples
from .events import TrainingState as _TrainingState

__all__ = ["JupyterCallback", "_JUP_TK_KEY"]


class JupyterCallback(Callback):
    """Render fitting and inference as a live notebook dashboard.

    Requires the ``jupyter`` extra (``pip install scikit-lm[jupyter]``), which
    pulls in ``ipywidgets`` for the live container; the loss curve is drawn as an
    inline SVG, so it needs no plotting library. Fine-tuning shows a header with a
    status badge, a row of stat cards (train loss, val loss, learning rate, step,
    and -- once validation is on and the first eval has landed -- how many evals
    have passed since the best validation loss, over the
    ``EvalConfig.patience`` budget, noting that best and the epoch it was
    reached at),
    a bordered panel pairing
    the train-loss curve with the run's hyperparameters, and -- when
    ``n_train_examples`` is set -- the latest serialized training rows.

    Inference renders by event kind. Candidate-ranking predictions (the
    classifier, or any discretized target -- :class:`Score` events) get a
    "verdict" view: summary cards (predictions, mean top-1 confidence, uncertain
    rows, predicted-bin mix) above a table with one row per prediction -- the
    conditioning fields (``given``) and a ``distribution`` cell of per-candidate
    colored probability badges, the top-1 first and flagged ``⚠`` when it falls
    below ``uncertain_threshold``. Generative
    predictions (synthesis -- :class:`Generation` events) use the same table: each
    row's conditioning fields (``given``) and the generated value, highlighted
    (``next``). A single :class:`ipywidgets.HTML` widget is displayed once per
    phase and its markup rebuilt in place from :attr:`Callback.state`, throttled to
    ``refresh_per_second``.

    Parameters
    ----------
    n_train_examples : int, optional
        How many serialized training rows to show, sampled at each epoch start.
        ``0`` hides them. Default ``5``.
    examples_view : {"table", "raw"}, optional
        How the training examples are shown: ``"raw"`` (default) shows the
        serialized text with JSON syntax highlighting; ``"table"`` lays the rows
        out in a fixed-column table. ``"table"`` falls back to ``"raw"`` for
        serializers whose output is not a JSON object.
    n_generations : int, optional
        How many of the most recent predictions to keep in the inference section.
        Default ``8``.
    uncertain_threshold : float, optional
        Top-1 probability below which a candidate-ranking prediction is flagged
        ``uncertain`` (and counted in the "uncertain rows" card). Default ``0.6``.
    top_k : int, optional
        How many candidates each probability bar shows as its own segment before
        grouping the remainder into an ``others`` segment. Default ``5``.
    loss_smoothing : float or None, optional
        Exponential-moving-average factor for the train-loss curve. ``None``
        (default) picks a factor from the run length (heavier smoothing for longer
        runs); a float in ``(0, 1]`` fixes it (smaller = smoother); ``0`` disables
        smoothing and plots the raw per-step loss.
    n_log_rows : int, optional
        How many of the most recent log entries to show in the training-log
        table, newest first. ``0`` hides the table. Default ``12``.
    log_every : int or float or {"epoch"}, optional
        How often a row is appended to the training-log table. ``"epoch"`` logs
        once per epoch; an ``int`` logs every ``N`` steps; a ``float`` in
        ``(0, 1]`` logs every that fraction of the run (e.g. ``0.1`` every 10 %
        of the total steps). A fractional or epoch cadence needs a known total;
        when the backend reports none, every report is logged. Default
        ``"epoch"``. Each row aggregates the reports in its interval: the loss is
        their mean and the ``±std`` their spread (shown only when an interval
        spans more than one report).
    refresh_per_second : float, optional
        Upper bound on how often the loss curve and metrics redraw. The progress
        bar advances on every report regardless. Default ``4``.
    train_color, eval_color : str, optional
        Stroke colors for the train and eval loss curves. Default a blue / magenta
        pair.
    """

    def __init__(
        self,
        n_train_examples: int = 5,
        examples_view: Literal["table", "raw"] = "raw",
        n_generations: int = 8,
        uncertain_threshold: float = 0.6,
        top_k: int = 5,
        loss_smoothing: float | None = None,
        n_log_rows: int = 12,
        log_every: int | float | Literal["epoch"] = "epoch",
        refresh_per_second: float = 4.0,
        train_color: str = _JUP_INFO,
        eval_color: str = "#d946ef",
    ) -> None:
        super().__init__()
        # ipywidgets is provided by the optional 'jupyter' extra.
        try:
            import ipywidgets  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "JupyterCallback requires the 'jupyter' extra: pip install scikit-lm[jupyter]"
            ) from exc
        if log_every != "epoch":
            if isinstance(log_every, float):
                if not 0.0 < log_every <= 1.0:
                    raise ValueError(f"log_every as a fraction must be in (0, 1], got {log_every}")
            elif log_every < 1:
                raise ValueError(f"log_every as a step count must be >= 1, got {log_every}")
        if loss_smoothing is not None and not 0.0 <= loss_smoothing <= 1.0:
            raise ValueError(f"loss_smoothing must be None or in [0, 1], got {loss_smoothing}")
        if examples_view not in ("table", "raw"):
            raise ValueError(f"examples_view must be 'table' or 'raw', got {examples_view!r}")
        self._n_train_examples = n_train_examples
        self._n_generations = n_generations
        self._n_log_rows = n_log_rows
        self._log_every = log_every
        self._min_interval = 1.0 / refresh_per_second if refresh_per_second > 0 else 0.0
        # Per-instance id scoping the examples-panel CSS tabs (avoids cross-cell clashes).
        self._ex_uid = f"sklmex{id(self) & 0xFFFFFFFF:x}"
        self._dash_state = DashboardState(
            n_generations=n_generations,
            uncertain_threshold=uncertain_threshold,
            log_every=log_every,
        )
        self._render_cfg = RenderConfig(
            n_train_examples=n_train_examples,
            examples_view=examples_view,
            n_log_rows=n_log_rows,
            log_every=log_every,
            uid=self._ex_uid,
        )
        self._renderer = HtmlRenderer(
            top_k=top_k,
            uncertain_threshold=uncertain_threshold,
            loss_smoothing=loss_smoothing,
            train_color=train_color,
            eval_color=eval_color,
            color_for=self._dash_state.color_for,
        )
        self._last_refresh = 0.0
        self._mode: Literal["fit", "predict"] | None = None
        # ipywidgets is untyped (no stubs); the widget handles stay Any.
        self._dash: Any = None
        self._ex_body: Any = None
        self._log_html: Any = None

    @override
    def on_event(self, state: _TrainingState, event: Event) -> None:
        self._dash_state.fold(state, event)
        match event:
            case FitStart():
                self._mode = "fit"
                self._build()
                self._refresh(force=True)
            case PredictStart():
                self._mode = "predict"
                self._build()
                self._refresh(force=True)
            case TrainExamples():
                # Re-render the examples body only here (per epoch), not on every
                # refresh, so the CSS tab selection persists between epoch boundaries.
                if self._ex_body is not None:
                    self._ex_body.value = self._examples_panel_html()
                self._refresh(force=True)
            case FitEnd() | PredictEnd():
                self._refresh(force=True)
            case _:
                self._refresh()

    def _build(self) -> None:
        import ipywidgets as w
        from IPython.display import display

        self._dash = w.HTML()
        # When training examples are shown, split the examples panel and the log into
        # their own HTML widgets. The examples body carries CSS-only tabs (no kernel
        # round-trip, so they switch live even while fit() blocks the kernel) and is
        # rewritten only per epoch, so the tab selection sticks between epochs.
        if self._mode == "fit" and self._n_train_examples:
            self._ex_body = w.HTML()
            self._log_html = w.HTML()
            display(w.VBox([self._dash, self._ex_body, self._log_html]))
        else:
            self._ex_body = self._log_html = None
            display(self._dash)

    def _refresh(self, *, force: bool = False) -> None:
        if self._dash is None:
            return
        now = monotonic()
        if not force and now - self._last_refresh < self._min_interval:
            return
        self._last_refresh = now
        if self._mode == "predict":
            dashboard = build_predict_dashboard(self.state, self._dash_state, self._render_cfg)
            self._dash.value = self._renderer.predict_html(dashboard)
        elif self._ex_body is not None:
            dashboard = build_fit_dashboard(self.state, self._dash_state, self._render_cfg)
            # The examples body is updated separately, per epoch (see on TrainExamples).
            self._dash.value = self._renderer.fit_top_html(dashboard)
            self._log_html.value = self._renderer.log_table_html(dashboard)
        else:
            dashboard = build_fit_dashboard(self.state, self._dash_state, self._render_cfg)
            self._dash.value = self._renderer.fit_html(dashboard)

    def _examples_panel_html(self) -> str:
        dashboard = build_fit_dashboard(self.state, self._dash_state, self._render_cfg)
        return self._renderer.examples_panel_html(dashboard)
