from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..format import _format_duration, _format_rate
from .widgets import (
    Dashboard,
    ExamplesPanel,
    GenerationRow,
    Header,
    LossCurve,
    Metrics,
    Predictions,
    ScoreRow,
    Widget,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
    from rich.measure import Measurement
    from rich.progress import Progress, Task


class _LossPlot:
    """A plotext loss curve sized to the Rich region it is drawn in.

    Reads a :class:`LossCurve` widget and draws its raw train/eval series, growing
    as reports arrive. The eval series is drawn only when the backend reports one,
    so the plot stays agnostic to whether validation is enabled. The renderer owns
    the marker, colors and fallback height.

    Parameters
    ----------
    curve : LossCurve
        The raw train/eval loss series to plot.
    marker : str
        The plotext marker glyph or density mode for both curves.
    train_color, eval_color : str
        plotext colors for the train and eval curves.
    plot_height : int
        Height in rows used when the host region reports none.
    """

    def __init__(
        self,
        curve: LossCurve,
        *,
        marker: str,
        train_color: str,
        eval_color: str,
        plot_height: int,
    ) -> None:
        self._curve = curve
        self._marker = marker
        self._train_color = train_color
        self._eval_color = eval_color
        self._plot_height = plot_height

    def __rich_measure__(self, console: Console, options: ConsoleOptions) -> Measurement:
        from rich.measure import Measurement

        # Claim the whole available width so a grid cell expands the plot to fill.
        return Measurement(8, options.max_width)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        from rich.ansi import AnsiDecoder
        from rich.console import Group
        from rich.text import Text

        c = self._curve
        if not c.train:
            yield Text("waiting for the first loss report…", style="dim")
            return
        width = options.max_width
        height = options.height or self._plot_height
        if width < 8 or height < 3:
            yield Text(f"loss={c.train[-1]:.4f}")
            return
        import plotext as plt  # plotext ships no type stubs (untyped third-party)

        plt.clf()
        plt.theme("clear")
        plt.plotsize(width, height)
        plt.plot(
            list(c.steps),
            list(c.train),
            marker=self._marker,
            color=self._train_color,
            label="train",
        )
        if c.evals:
            plt.scatter(
                list(c.eval_steps),
                list(c.evals),
                marker=self._marker,
                color=self._eval_color,
                label="eval",
            )
        yield Group(*AnsiDecoder().decode(plt.build()))


class RichRenderer:
    """Walk a :class:`Dashboard` and emit :mod:`rich` renderables.

    The counterpart to :class:`html_renderer.HtmlRenderer`: both consume the same
    widget tree built by :mod:`build`, differing only in the output. This one owns
    the terminal visuals -- the plotext loss curve's marker and colors, the metrics
    column layout, the dim/ellipsized example and prediction lines -- and folds in
    the live progress the callback owns (the :class:`rich.progress.Task` and its
    throughput, which come from the live timer rather than from the widget tree).

    Parameters
    ----------
    marker : str
        The plotext marker for the loss curve (a density mode or a single glyph).
    train_color, eval_color : str
        plotext colors for the train and eval loss curves.
    plot_height : int
        Fallback height in rows for the loss curve when the region reports none.
    """

    def __init__(self, *, marker: str, train_color: str, eval_color: str, plot_height: int) -> None:
        self._marker = marker
        self._train_color = train_color
        self._eval_color = eval_color
        self._plot_height = plot_height

    # ----------------------------------------------------------------- top level

    def fit(
        self, dashboard: Dashboard, *, task: Task | None, speed: float | None
    ) -> RenderableType:
        """The fine-tuning dashboard: header, loss plot beside metrics, examples.

        Parameters
        ----------
        dashboard : Dashboard
            The fit tree from :func:`build.build_fit_dashboard`.
        task : rich.progress.Task or None
            The live progress task, source of the progress/elapsed/ETA rows the
            metrics column shows; ``None`` before the task exists.
        speed : float or None
            Steps per second for ``task``, or ``None`` until rich can estimate it.

        Returns
        -------
        rich.console.RenderableType
            The composed fit renderable.
        """
        from rich.console import Group
        from rich.table import Table
        from rich.text import Text

        header = _pick(dashboard, Header)
        curve = _pick(dashboard, LossCurve)
        metrics = _pick(dashboard, Metrics)
        plot = _LossPlot(
            curve,
            marker=self._marker,
            train_color=self._train_color,
            eval_color=self._eval_color,
            plot_height=self._plot_height,
        )
        body = Table.grid(expand=True, padding=(0, 2))
        body.add_column(ratio=3)
        body.add_column(ratio=1, justify="left", vertical="middle")
        body.add_row(plot, self._metrics_grid(metrics, task, speed))
        blocks: list[RenderableType] = [
            self._header(header),
            Text(""),
            body,
        ]
        examples = _find(dashboard, ExamplesPanel)
        if examples is not None and examples.texts:
            blocks += [Text(""), self._examples(examples)]
        return Group(*blocks)

    def predict(self, dashboard: Dashboard, *, progress: Progress) -> RenderableType:
        """The inference dashboard: header, progress bar and recent predictions.

        A ``retries N`` line appears between the bar and the table once a
        malformed generation forces a retry, so a stalling predict is visible.

        Parameters
        ----------
        dashboard : Dashboard
            The predict tree from :func:`build.build_predict_dashboard`.
        progress : rich.progress.Progress
            The live row-progress bar, rendered between the header and the table.

        Returns
        -------
        rich.console.RenderableType
            The composed predict renderable.
        """
        from rich.console import Group
        from rich.text import Text

        header = _pick(dashboard, Header)
        predictions = _pick(dashboard, Predictions)
        blocks: list[RenderableType] = [self._header(header), progress, Text("")]
        if predictions.retries:
            blocks.append(Text(f"retries {predictions.retries}", style="yellow"))
        blocks.append(self._section("recent predictions", self._prediction_lines(predictions)))
        return Group(*blocks)

    # --------------------------------------------------------------------- header

    def _header(self, header: Header) -> RenderableType:
        from rich.text import Text

        line = Text()
        line.append(header.badge.label, style="bold cyan")
        if header.title and header.title != header.badge.label:
            line.append("  ")
            line.append(header.title)
        if header.subtitle:
            line.append("  ")
            line.append(header.subtitle, style="dim")
        return line

    # -------------------------------------------------------------------- metrics

    def _metrics_grid(
        self, metrics: Metrics, task: Task | None, speed: float | None
    ) -> RenderableType:
        from rich.table import Table

        grid = Table.grid(padding=(0, 1))
        grid.add_column(justify="right", style="dim")
        grid.add_column()
        for row in metrics.rows:
            grid.add_row(row.label, row.value)
            if row.label == "rows":
                self._task_rows(grid, task)
            elif row.label == "epoch" and speed:
                grid.add_row("speed", _format_rate(speed))
        return grid

    def _task_rows(self, grid: Any, task: Task | None) -> None:
        # rich.table.Table.grid is untyped at the third-party boundary.
        if task is None:
            return
        if task.total is not None:
            grid.add_row("progress", f"{task.percentage:.0f}%")
        if task.elapsed is not None:
            grid.add_row("elapsed", _format_duration(task.elapsed))
        if task.time_remaining is not None:
            grid.add_row("eta", _format_duration(task.time_remaining))

    # ------------------------------------------------------------------- sections

    def _section(self, title: str, lines: Sequence[RenderableType]) -> RenderableType:
        from rich.console import Group
        from rich.text import Text

        return Group(Text(title, style="dim italic"), *lines)

    def _examples(self, panel: ExamplesPanel) -> RenderableType:
        from rich.text import Text

        rows = [
            Text(f"  {text}", style="dim", no_wrap=True, overflow="ellipsis")
            for text in panel.texts
        ]
        return self._section(f"training examples · epoch {panel.epoch}", rows)

    # ---------------------------------------------------------------- predictions

    def _prediction_lines(self, predictions: Predictions) -> list[RenderableType]:
        from rich.text import Text

        if not predictions.rows:
            return [Text("  waiting…", style="dim")]
        lines: list[RenderableType] = []
        for row in predictions.rows:
            lines.append(Text(f"  {row.prompt}", no_wrap=True, overflow="ellipsis"))
            lines.append(Text(f"    {self._detail(row)}", style="dim"))
        return lines

    def _detail(self, row: GenerationRow | ScoreRow) -> str:
        if isinstance(row, ScoreRow):
            return ", ".join(f"{c}={p:.3f}" for c, p in row.ranking)
        return "?" if row.value is None else str(row.value)


def _find[W: Widget](dashboard: Dashboard, kind: type[W]) -> W | None:
    return next((c for c in dashboard.children if isinstance(c, kind)), None)


def _pick[W: Widget](dashboard: Dashboard, kind: type[W]) -> W:
    return next(c for c in dashboard.children if isinstance(c, kind))
