from __future__ import annotations

import io
import sys
from collections.abc import Callable
from time import monotonic
from typing import TYPE_CHECKING, Any, override

from .base import Callback
from .dashboard import (
    DashboardState,
    RenderConfig,
    build_fit_dashboard,
    build_predict_dashboard,
)
from .dashboard.rich_renderer import RichRenderer
from .events import (
    Event,
    FitEnd,
    FitStart,
    PredictEnd,
    PredictStart,
    RowEnd,
    TrainingState,
    TrainReport,
)

if TYPE_CHECKING:
    from rich.console import Console, RenderableType
    from rich.live import Live
    from rich.progress import Progress, Task, TaskID


# A single styled <pre> fragment for the Jupyter HTML widget; inline styles carry
# the rich/plotext colors so it renders the same in any notebook frontend.
_HTML_FORMAT = (
    '<pre style="white-space:pre;line-height:1.2;font-size:13px;'
    "color:{foreground};background-color:{background};"
    'font-family:ui-monospace,SFMono-Regular,Menlo,monospace">{code}</pre>'
)


class _Dynamic:
    """Renderable that re-evaluates ``render`` on every Live refresh."""

    def __init__(self, render: Callable[[], RenderableType]) -> None:
        self._render = render

    def __rich__(self) -> RenderableType:
        return self._render()


class RichCallback(Callback):
    """Render fitting and inference as a live :mod:`rich` dashboard.

    Requires the ``rich`` extra (``pip install scikit-lm[rich]``), which also
    pulls in ``plotext`` for the loss curve; used inside a Jupyter kernel it
    additionally needs the ``jupyter`` extra (``ipywidgets``). The layout is flat (no
    nested boxes): fine-tuning shows a header line (status, model name and row
    count), a live train-loss plot beside a metrics column (loss, epoch, learning
    rate, memory, plus progress/elapsed/ETA/throughput from the live timer) and --
    when ``n_train_examples`` is set -- the latest serialized training rows.
    Inference shows the header, a row progress bar and the last ``n_generations``
    generations (or scored rankings). A single :class:`rich.live.Live` redraws
    everything at ``refresh_per_second``.

    Folds events through a shared :class:`DashboardState` and renders the same
    renderer-agnostic widget tree :class:`JupyterCallback` does (built by
    :func:`build_fit_dashboard` / :func:`build_predict_dashboard`), differing only
    in the renderer (:class:`RichRenderer` here, an HTML one there).

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
    Works in a terminal and in a Jupyter kernel, with the same dashboard. In a
    terminal it drives a :class:`rich.live.Live` that captures the real
    ``sys.stdout`` when fitting/predicting starts, so it keeps animating even while
    a backend redirects ``sys.stdout`` during training (as the MLX backend does).
    In a Jupyter kernel it does **not** use rich's notebook ``Live`` path (which
    can stack frames); instead it renders the dashboard into one
    ``ipywidgets.HTML`` widget per phase and replaces its ``value`` in place, so a
    kernel needs the ``jupyter`` extra.
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
        self._rps = refresh_per_second
        self._dash_state = DashboardState(
            n_generations=n_generations,
            uncertain_threshold=0.6,
            log_every="epoch",
        )
        # n_log_rows=0 / uid="" disable the HTML-only widgets (the training-log
        # table and the CSS-tab examples panel) the Rich renderer never draws.
        self._render_cfg = RenderConfig(
            n_train_examples=n_train_examples,
            examples_view="raw",
            n_log_rows=0,
            log_every="epoch",
            uid="",
            loss_window=loss_window,
        )
        self._renderer = RichRenderer(
            marker=marker,
            train_color=train_color,
            eval_color=eval_color,
            plot_height=plot_height,
        )
        self._console: Console | None = None
        self._live: Live | None = None
        self._progress: Progress | None = None
        self._task: TaskID | None = None
        # Jupyter path: an HTML widget the dashboard is rendered into in place.
        self._jupyter = self._detect_jupyter()
        if self._jupyter:
            # ipywidgets is provided by the optional 'jupyter' extra; failing here
            # keeps _auto_callback's fall-through to LoggingCallback working.
            try:
                import ipywidgets  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "RichCallback in a Jupyter kernel needs the 'jupyter' extra: "
                    "pip install scikit-lm[jupyter]"
                ) from exc
        self._min_interval = 1.0 / refresh_per_second if refresh_per_second > 0 else 0.0
        self._last_refresh = 0.0
        self._render: Callable[[], RenderableType] | None = None
        # ipywidgets is untyped (no stubs); the widget handle stays Any.
        self._output: Any = None

    @staticmethod
    def _detect_jupyter() -> bool:
        """Whether rich considers the current environment a Jupyter kernel."""
        from rich.console import Console

        return Console().is_jupyter

    @override
    def on_event(self, state: TrainingState, event: Event) -> None:
        self._dash_state.fold(state, event)
        if self._jupyter:
            self._on_event_jupyter(event)
        else:
            self._on_event_terminal(event)

    # --------------------------------------------------------------- terminal (Live)

    def _on_event_terminal(self, event: Event) -> None:
        match event:
            case FitStart():
                self._start_live(self._render_fit, total=None)
            case PredictStart(n=n):
                self._start_live(self._render_predict, total=n)
            case FitEnd() | PredictEnd():
                self._stop_live()
            case TrainReport(step=step, total_steps=total):
                self._update_progress(step, total)
            case RowEnd(index=index, total=total):
                self._update_progress(index + 1, total)
            case _:
                pass

    def _start_live(self, render: Callable[[], RenderableType], total: int | None) -> None:
        from rich.console import Console
        from rich.live import Live

        self._console = Console(file=sys.stdout)
        self._make_progress(total=total)
        self._live = Live(_Dynamic(render), console=self._console, refresh_per_second=self._rps)
        self._live.start()

    def _stop_live(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
        self._progress = None
        self._task = None

    # --------------------------------------------------------------- jupyter (HTML)

    def _on_event_jupyter(self, event: Event) -> None:
        match event:
            case FitStart():
                self._start_output(self._render_fit, total=None)
            case PredictStart(n=n):
                self._start_output(self._render_predict, total=n)
            case FitEnd() | PredictEnd():
                self._refresh_output(force=True)
                self._stop_output()
            case TrainReport(step=step, total_steps=total):
                self._update_progress(step, total)
                self._refresh_output()
            case RowEnd(index=index, total=total):
                self._update_progress(index + 1, total)
                self._refresh_output()
            case _:
                self._refresh_output()

    def _start_output(self, render: Callable[[], RenderableType], total: int | None) -> None:
        import ipywidgets as w
        from IPython.display import display

        self._console = self._jupyter_console()
        self._make_progress(total=total)
        self._render = render
        self._output = w.HTML()
        display(self._output)
        self._last_refresh = 0.0
        self._refresh_output(force=True)

    def _jupyter_console(self) -> Console:
        """A recording console whose contents are exported to HTML each frame.

        ``force_terminal`` keeps the full color so ``export_html`` carries it; it
        is deliberately not ``is_jupyter`` -- we manage the widget ourselves
        rather than relying on rich's notebook ``Live`` path.
        """
        from rich.console import Console

        return Console(
            record=True,
            file=io.StringIO(),
            force_terminal=True,
            force_jupyter=False,
            color_system="truecolor",
            width=100,
        )

    def _refresh_output(self, *, force: bool = False) -> None:
        if self._output is None or self._render is None or self._console is None:
            return
        now = monotonic()
        if not force and now - self._last_refresh < self._min_interval:
            return
        self._last_refresh = now
        self._console.print(self._render())
        self._output.value = self._console.export_html(inline_styles=True, code_format=_HTML_FORMAT)

    def _stop_output(self) -> None:
        self._output = None
        self._render = None
        self._progress = None
        self._task = None

    # ----------------------------------------------------------------------- shared

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

    def _update_progress(self, completed: int, total: int | None) -> None:
        if self._progress is not None and self._task is not None:
            self._progress.update(self._task, completed=completed, total=total)

    def _current_task(self) -> Task | None:
        """The active :class:`rich.progress.Task`, source of progress/elapsed/eta."""
        if self._progress is None:
            return None
        return next((t for t in self._progress.tasks if t.id == self._task), None)

    def _current_speed(self) -> float | None:
        """Steps per second for the active task, or ``None`` until rich can estimate it."""
        task = self._current_task()
        return task.speed if task is not None else None

    def _render_fit(self) -> RenderableType:
        from rich.text import Text

        if self._progress is None:
            return Text("")
        dashboard = build_fit_dashboard(self.state, self._dash_state, self._render_cfg)
        return self._renderer.fit(dashboard, task=self._current_task(), speed=self._current_speed())

    def _render_predict(self) -> RenderableType:
        from rich.text import Text

        if self._progress is None:
            return Text("")
        dashboard = build_predict_dashboard(self.state, self._dash_state, self._render_cfg)
        return self._renderer.predict(dashboard, progress=self._progress)
