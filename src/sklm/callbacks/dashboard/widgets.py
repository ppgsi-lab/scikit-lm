from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

# --------------------------------------------------------------------------- leaves


@dataclass(frozen=True)
class Badge:
    """A status pill: a label tinted by ``kind`` (active vs. done).

    Parameters
    ----------
    label : str
        Text shown inside the pill (e.g. ``"fine-tuning"``).
    kind : {"active", "done"}
        Which palette the renderer colors the pill with.
    """

    label: str
    kind: Literal["active", "done"]


@dataclass(frozen=True)
class Header:
    """The dashboard title block: an icon, a title, a subtitle and a status badge.

    Parameters
    ----------
    title : str
        Headline (the model name, or a phase placeholder).
    subtitle : str
        Secondary line under the title (e.g. ``"120 rows"``).
    badge : Badge
        The status pill rendered on the trailing edge.
    """

    title: str
    subtitle: str
    badge: Badge


@dataclass(frozen=True)
class Stat:
    """One headline metric: a labeled value, optionally with a secondary suffix.

    Parameters
    ----------
    label : str
        Card caption (e.g. ``"train loss"``).
    value : str
        Primary value text, pre-formatted by the builder.
    suffix : str or None
        Dimmed trailing text appended after ``value`` (e.g. ``" / 1,000"`` for a
        step counter). ``None`` when there is no suffix.
    mono : bool
        Whether the renderer sets the value in a monospace face.
    """

    label: str
    value: str
    suffix: str | None = None
    mono: bool = False


@dataclass(frozen=True)
class StatCards:
    """A row of :class:`Stat` cards.

    Rendered only by the HTML renderer; the Rich one shows :class:`Metrics` instead.

    Parameters
    ----------
    cards : sequence of Stat
        The cards, in display order.
    """

    cards: Sequence[Stat]


@dataclass(frozen=True)
class KeyValue:
    """One ``label / value`` pair of a :class:`KeyValues` table.

    Parameters
    ----------
    label : str
        Left-hand caption.
    value : str
        Right-hand value, pre-formatted (HTML-escaped) by the builder.
    mono : bool
        Whether the renderer sets the value in a monospace face.
    """

    label: str
    value: str
    mono: bool = False


@dataclass(frozen=True)
class KeyValues:
    """A label/value table, plus an optional elapsed/ETA footer.

    Rendered only by the HTML renderer (the hyperparameters panel); the Rich one
    ignores it.

    Parameters
    ----------
    title : str
        Panel heading (e.g. ``"hyperparameters"``).
    rows : sequence of KeyValue
        The table rows, in order.
    elapsed : str or None
        Pre-formatted elapsed-time string for the footer, or ``None`` to hide it.
    eta : str or None
        Pre-formatted "time left" string appended after ``elapsed``; ``None`` when
        no estimate is available yet.
    """

    title: str
    rows: Sequence[KeyValue]
    elapsed: str | None = None
    eta: str | None = None


@dataclass(frozen=True)
class MetricRow:
    """One ``label / value`` line of the :class:`Metrics` summary.

    Parameters
    ----------
    label : str
        Left-hand caption (e.g. ``"epoch loss"``).
    value : str
        Right-hand value, pre-formatted by the builder.
    """

    label: str
    value: str


@dataclass(frozen=True)
class Metrics:
    """The compact fit-metrics summary the Rich dashboard shows beside the curve.

    Carries the full ordered list of :class:`TrainingState`-derived metric rows
    (rows, epoch loss, last-N spread, loss, eval, epoch, learning rate, memory),
    so the renderer never reaches back into the raw state. Progress/elapsed/ETA and
    throughput stay on the renderer because they come from its own live timer, not
    from the state. Renderers that have no compact-summary slot (the HTML one)
    simply ignore this widget.

    Parameters
    ----------
    rows : sequence of MetricRow
        The metric lines, in display order; only the populated ones are present.
    """

    rows: Sequence[MetricRow]


@dataclass(frozen=True)
class LossCurve:
    """The raw loss series for the train/eval curves.

    The renderer owns smoothing, axis choice (linear vs. log) and colors; this
    carries only the data plus the subtitle the builder composed.

    Parameters
    ----------
    steps : sequence of int
        X positions of the train series.
    train : sequence of float
        Train losses, aligned to ``steps``.
    eval_steps : sequence of int
        X positions of the eval series (empty when validation is off).
    evals : sequence of float
        Eval losses, aligned to ``eval_steps``.
    subtitle : str
        The ``train · val · ema`` style caption (the builder drops absent parts).
    """

    steps: Sequence[int]
    train: Sequence[float]
    eval_steps: Sequence[int]
    evals: Sequence[float]
    subtitle: str


@dataclass(frozen=True)
class ExamplesPanel:
    """The per-epoch sample of serialized training rows.

    Parameters
    ----------
    epoch : int
        1-indexed epoch the sample was taken at.
    texts : sequence of str
        The serialized rows (``prompt + completion``), already truncated to the
        configured count.
    parsed : tuple of (sequence of str, sequence of mapping) or None
        ``(columns, rows)`` when every text is a JSON object (columns in first-seen
        order), else ``None`` so the renderer falls back to the raw view.
    view : {"table", "raw"}
        The view requested by the user; the renderer decides the default-checked
        tab from it.
    uid : str
        Per-instance id scoping the examples-panel CSS tabs.
    """

    epoch: int
    texts: Sequence[str]
    parsed: tuple[Sequence[str], Sequence[dict[str, object]]] | None
    view: Literal["table", "raw"]
    uid: str


@dataclass(frozen=True)
class LogRow:
    """One aggregated training-log row.

    Parameters
    ----------
    step : int
        Last optimizer step of the interval.
    epoch : float or None
        Fractional epoch at that step, or ``None`` when unknown.
    loss : float or None
        Mean training loss over the interval, or ``None`` when none was reported.
    std : float or None
        Population std of the interval's losses; ``None`` for a single-report
        interval.
    lr : float or None
        Learning rate at the last report of the interval.
    grad_norm : float or None
        Mean gradient norm over the interval, or ``None`` when unreported.
    eval_loss : float or None
        Eval loss recorded at this exact ``step``, or ``None`` when none matched.
    steps_per_second : float or None
        Throughput since the previous row, or ``None`` for the first row.
    """

    step: int
    epoch: float | None
    loss: float | None
    std: float | None
    lr: float | None
    grad_norm: float | None
    eval_loss: float | None
    steps_per_second: float | None


@dataclass(frozen=True)
class LogTable:
    """The recent training-log rows, newest first.

    Rendered only by the HTML renderer; the Rich one ignores it (the
    :class:`RichCallback` disables it with ``n_log_rows=0``).

    Parameters
    ----------
    rows : sequence of LogRow
        The rows to show, already windowed and ordered newest-first.
    cadence : str
        Human-readable cadence label (e.g. ``"every epoch"``).
    """

    rows: Sequence[LogRow]
    cadence: str


@dataclass(frozen=True)
class GenerationRow:
    """One generative prediction: conditioning fields and the generated value.

    Parameters
    ----------
    prompt : str
        The conditioning prompt (a possibly dangling JSON fragment).
    value : object or None
        The decoded generated value, or ``None`` when decoding was malformed.
    """

    prompt: str
    value: object | None


@dataclass(frozen=True)
class ScoreRow:
    """One candidate-ranking prediction.

    Parameters
    ----------
    prompt : str
        The conditioning prompt (a possibly dangling JSON fragment).
    ranking : sequence of (object, float)
        ``(candidate, probability)`` pairs, sorted by probability descending.
    uncertain : bool
        Whether the top-1 probability fell below the uncertain threshold.
    """

    prompt: str
    ranking: Sequence[tuple[object, float]]
    uncertain: bool


type PredictionRow = GenerationRow | ScoreRow


@dataclass(frozen=True)
class BinShare:
    """One predicted-bin entry of the verdict mix.

    Parameters
    ----------
    label : object
        The candidate label (or ``"others"`` for the grouped remainder).
    count : int
        How many predictions landed in this bin.
    """

    label: object
    count: int


@dataclass(frozen=True)
class Predictions:
    """The inference table plus its verdict aggregates.

    Parameters
    ----------
    scoring : bool
        Whether the rows are candidate-ranking (:class:`ScoreRow`) or generative
        (:class:`GenerationRow`); selects the header tag and the columns.
    rows : sequence of PredictionRow
        The recent predictions, oldest-first.
    bins : sequence of BinShare
        The predicted-bin mix (empty for the generative view); already topped and
        grouped into an ``others`` entry when needed.
    bins_label : str
        Caption for the bins card (``"predicted bins"`` or its ``· top 5`` form).
    retries : int
        How many malformed-generation retries the phase has seen so far; renderers
        surface it only when nonzero.
    """

    scoring: bool
    rows: Sequence[PredictionRow]
    bins: Sequence[BinShare]
    bins_label: str
    retries: int = 0


type Widget = (
    Header | StatCards | KeyValues | LossCurve | Metrics | ExamplesPanel | LogTable | Predictions
)


@dataclass(frozen=True)
class Dashboard:
    """The whole dashboard as an ordered tree of widgets.

    Parameters
    ----------
    children : sequence of Widget
        Top-level widgets in display order. The renderer may also pick named
        sub-sections out of this list (the Jupyter layout splits the examples panel
        and the log into their own widgets).
    """

    children: Sequence[Widget]
