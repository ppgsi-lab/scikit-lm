from __future__ import annotations

import html
import json
import math
import re
from collections.abc import Callable, Sequence

from .build import _sci
from .widgets import (
    BinShare,
    Dashboard,
    ExamplesPanel,
    GenerationRow,
    Header,
    KeyValues,
    LogRow,
    LogTable,
    LossCurve,
    Predictions,
    ScoreRow,
    StatCards,
)

# -------------------------------------------------------------------------- palette

_JUP_ACCENT = "#06b6d4"
_JUP_DONE = "#16a34a"
_JUP_DIM = "#9ca3af"
_JUP_INFO = "#378add"
_JUP_INFO_BG = "rgba(55,138,221,0.14)"
_JUP_DONE_BG = "rgba(22,163,74,0.14)"
_JUP_CARD = "rgba(128,128,128,0.09)"
_JUP_BORDER = "rgba(128,128,128,0.22)"
_JUP_MONO = "ui-monospace,SFMono-Regular,Menlo,monospace"
# JSON token colors drawn from JupyterLab's CodeMirror theme variables, so the
# highlighted training examples follow the active light/dark theme.
_JUP_TK_KEY = "var(--jp-mirror-editor-property-color,#0b7285)"
_JUP_TK_STR = "var(--jp-mirror-editor-string-color,#c2185b)"
_JUP_TK_NUM = "var(--jp-mirror-editor-number-color,#1565c0)"
_JUP_TK_KW = "var(--jp-mirror-editor-keyword-color,#9c27b0)"
_JUP_TK_PUNCT = _JUP_DIM
_JUP_OTHERS = "#b8b8b0"
_JUP_ICON = (
    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/>'
    '<line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/>'
    '<line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/>'
    '<line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/>'
    '<line x1="17" y1="16" x2="23" y2="16"/></svg>'
)


def _jtoken(value: object) -> str:
    """A single JSON scalar wrapped in its theme-colored span."""
    if isinstance(value, str):
        return f'<span style="color:{_JUP_TK_STR}">{html.escape(json.dumps(value))}</span>'
    if isinstance(value, bool) or value is None:
        return f'<span style="color:{_JUP_TK_KW}">{json.dumps(value)}</span>'
    if isinstance(value, (int, float)):
        return f'<span style="color:{_JUP_TK_NUM}">{json.dumps(value)}</span>'
    return html.escape(json.dumps(value))


def _highlight_json(text: str) -> str | None:
    """Render a JSON object string as a theme-colored, one-line HTML fragment.

    Returns ``None`` when ``text`` is not a JSON object, so the caller can fall
    back to the plain serialized line for non-JSON serializers.
    """
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None

    def punct(text: str) -> str:
        return f'<span style="color:{_JUP_TK_PUNCT}">{text}</span>'

    pairs = [
        f'<span style="color:{_JUP_TK_KEY}">{html.escape(json.dumps(k))}</span>'
        f"{punct(': ')}{_jtoken(v)}"
        for k, v in obj.items()
    ]
    return punct("{") + punct(", ").join(pairs) + punct("}")


_JSON_TOKEN = re.compile(
    r'"(?:[^"\\]|\\.)*"'  # string
    r"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"  # number
    r"|true|false|null"  # keyword
    r"|[{}\[\],:]"  # structural punctuation
)


def _highlight_json_fragment(text: str) -> str:
    """Colorize a JSON(-ish) string, tolerating a partial (dangling) object.

    Unlike :func:`_highlight_json`, this tokenizes with a regex instead of
    parsing, so a conditioning prompt that ends mid-object -- right before the
    target value, e.g. ``{"x": 1, "species": `` -- is still highlighted. A quoted
    token is a key when its next non-space character is a colon; anything the
    regex does not recognize passes through escaped.
    """
    out: list[str] = []
    pos = 0
    for m in _JSON_TOKEN.finditer(text):
        if m.start() > pos:
            out.append(html.escape(text[pos : m.start()]))
        tok = m.group()
        if tok[0] == '"':
            color = _JUP_TK_KEY if text[m.end() :].lstrip().startswith(":") else _JUP_TK_STR
            out.append(f'<span style="color:{color}">{html.escape(tok)}</span>')
        elif tok in ("true", "false", "null"):
            out.append(f'<span style="color:{_JUP_TK_KW}">{tok}</span>')
        elif tok[0] == "-" or tok[0].isdigit():
            out.append(f'<span style="color:{_JUP_TK_NUM}">{tok}</span>')
        else:
            out.append(f'<span style="color:{_JUP_TK_PUNCT}">{html.escape(tok)}</span>')
        pos = m.end()
    if pos < len(text):
        out.append(html.escape(text[pos:]))
    return "".join(out)


def _jline(text: str, *, color: str | None = None) -> str:
    """An ellipsis-truncated single line, mirroring the Rich ``no_wrap`` rows."""
    style = "white-space:nowrap;overflow:hidden;text-overflow:ellipsis"
    if color is not None:
        style += f";color:{color}"
    return f'<div style="{style}">{html.escape(text)}</div>'


def _jpredval(value: object) -> str:
    """A generated value as it reads in a prediction line: strings quoted, ``?`` if malformed."""
    if value is None:
        return "?"
    if isinstance(value, str):
        return json.dumps(value)
    return str(value)


def _jcand(label: object) -> str:
    """A candidate label as it reads in a chip/legend: floats trimmed, others stringified."""
    if isinstance(label, float):
        return f"{label:g}"
    return str(label)


def _nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    """About ``count`` round tick values spanning ``[lo, hi]`` (1/2/5 x 10^k steps)."""
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / count
    mag = 10.0 ** math.floor(math.log10(raw))
    norm = raw / mag
    step = (1 if norm < 1.5 else 2 if norm < 3 else 5 if norm < 7 else 10) * mag
    ticks: list[float] = []
    v = math.ceil(lo / step) * step
    while v <= hi + step * 1e-6:
        ticks.append(round(v, 10))
        v += step
    return ticks


def _log_ticks(lo: float, hi: float, limit: int = 7) -> list[float]:
    """Round ticks on a base-10 log axis, thinned to at most ``limit`` labels.

    Candidates are 1/2/3/5 within each decade of ``[lo, hi]``; over a wide span
    that is far too many to label without overlap, so the list is strided down to
    ``limit`` evenly-spaced ticks.
    """
    cands: list[float] = []
    k = math.floor(math.log10(lo))
    while 10.0**k <= hi * (1 + 1e-9):
        cands.extend(s * 10.0**k for s in (1, 2, 3, 5) if lo <= s * 10.0**k <= hi)
        k += 1
    if len(cands) <= limit:
        return cands
    stride = math.ceil(len(cands) / limit)
    return cands[::stride]


def _ema(values: Sequence[float], alpha: float) -> list[float]:
    """Exponential moving average of ``values`` (smoothing factor ``alpha``)."""
    out: list[float] = []
    acc = values[0]
    for v in values:
        acc = alpha * v + (1 - alpha) * acc
        out.append(acc)
    return out


def _downsample(points: list[tuple[int, float]], limit: int) -> list[tuple[int, float]]:
    """At most ``limit`` evenly-spaced points (plus the last), bounding the SVG path size."""
    if len(points) <= limit:
        return points
    stride = len(points) / limit
    sampled = [points[int(i * stride)] for i in range(limit)]
    sampled.append(points[-1])
    return sampled


class HtmlRenderer:
    """Walk a :class:`Dashboard` and emit the Jupyter HTML/SVG markup.

    Holds the styling and config the markup needs (smoothing, curve colors, the
    candidate-color cursor, the top-k cutoff, the uncertain threshold). The widget
    tree carries the data; this owns every visual decision -- the loss-curve
    smoothing and axis, the palette and the exact HTML -- so the same tree can feed
    a different renderer.

    Parameters
    ----------
    top_k : int
        How many candidates each probability bar shows before the ``others`` group.
    uncertain_threshold : float
        Top-1 probability below which a scored row is flagged ``⚠``.
    loss_smoothing : float or None
        EMA factor for the train-loss curve; ``None`` derives it from the run
        length, ``0`` plots the raw loss.
    train_color, eval_color : str
        Stroke colors for the train and eval curves.
    color_for : callable
        Maps a candidate to its stable categorical color (the shared cursor on
        :class:`DashboardState`).
    """

    def __init__(
        self,
        *,
        top_k: int,
        uncertain_threshold: float,
        loss_smoothing: float | None,
        train_color: str,
        eval_color: str,
        color_for: Callable[[object], str],
    ) -> None:
        self._top_k = top_k
        self._uncertain_threshold = uncertain_threshold
        self._loss_smoothing = loss_smoothing
        self._train_color = train_color
        self._eval_color = eval_color
        self._color_for = color_for

    # ------------------------------------------------------------------ top level

    def fit_html(self, dashboard: Dashboard) -> str:
        """The whole fit dashboard as one HTML string (single-widget layout)."""
        return self._wrap(self.fit_top_html(dashboard) + self._fit_extras(dashboard))

    def fit_top_html(self, dashboard: Dashboard) -> str:
        """Header, stat cards and the chart/hyperparameter grid (no examples/log)."""
        header = next(c for c in dashboard.children if isinstance(c, Header))
        cards = next(c for c in dashboard.children if isinstance(c, StatCards))
        curve = next(c for c in dashboard.children if isinstance(c, LossCurve))
        hyper = next(c for c in dashboard.children if isinstance(c, KeyValues))
        grid = (
            '<div style="display:grid;grid-template-columns:1.1fr 1fr;gap:16px;'
            f'align-items:stretch">{self._chart_panel(curve)}{self._hyper_panel(hyper)}</div>'
        )
        return self._header_html(header) + self._cards_html(cards) + grid

    def _fit_extras(self, dashboard: Dashboard) -> str:
        out = ""
        examples = next((c for c in dashboard.children if isinstance(c, ExamplesPanel)), None)
        log = next((c for c in dashboard.children if isinstance(c, LogTable)), None)
        if examples is not None:
            out += self._examples_html(examples)
        if log is not None:
            out += self._log_table_html(log)
        return out

    def examples_panel_html(self, dashboard: Dashboard) -> str:
        """The standalone examples panel (CSS-tab layout), or ``""`` when absent."""
        examples = next((c for c in dashboard.children if isinstance(c, ExamplesPanel)), None)
        if examples is None:
            return ""
        return self._examples_panel_html(examples)

    def log_table_html(self, dashboard: Dashboard) -> str:
        """The standalone training-log table, or ``""`` when absent/empty."""
        log = next((c for c in dashboard.children if isinstance(c, LogTable)), None)
        if log is None or not log.rows:
            return ""
        return self._log_table_html(log)

    def predict_html(self, dashboard: Dashboard) -> str:
        """The whole inference dashboard as one HTML string."""
        header = next(c for c in dashboard.children if isinstance(c, Header))
        cards = next(c for c in dashboard.children if isinstance(c, StatCards))
        predictions = next(c for c in dashboard.children if isinstance(c, Predictions))
        if predictions.scoring:
            cards_html = self._score_cards_html(cards, predictions)
        else:
            cards_html = (
                '<div style="display:grid;'
                "grid-template-columns:repeat(auto-fit,minmax(120px,1fr));"
                f'gap:12px;margin-bottom:20px">{self._card_from_stat(cards)}</div>'
            )
        body = (
            self._header_html(header)
            + cards_html
            + self._panel(self._generations_html(predictions))
        )
        return self._wrap(body)

    # ---------------------------------------------------------------------- frame

    def _wrap(self, inner: str) -> str:
        return (
            '<div style="font-family:var(--jp-ui-font-family,system-ui,sans-serif);'
            f'max-width:760px;padding:4px 0;line-height:1.4">{inner}</div>'
        )

    def _panel(self, inner: str) -> str:
        return (
            f'<div style="border:0.5px solid {_JUP_BORDER};border-radius:14px;'
            f'padding:14px 16px">{inner}</div>'
        )

    # --------------------------------------------------------------------- header

    def _header_html(self, header: Header) -> str:
        icon = (
            f'<div style="width:40px;height:40px;border-radius:9px;background:{_JUP_INFO_BG};'
            f"color:{_JUP_INFO};display:flex;align-items:center;justify-content:center;"
            f'flex:none">{_JUP_ICON}</div>'
        )
        text = (
            f'<div><div style="font-weight:600;font-size:15px">{html.escape(header.title)}</div>'
            f'<div style="font-size:13px;color:{_JUP_DIM}">{html.escape(header.subtitle)}</div>'
            "</div>"
        )
        left = f'<div style="display:flex;align-items:center;gap:12px">{icon}{text}</div>'
        return (
            '<div style="display:flex;align-items:center;justify-content:space-between;'
            f'gap:8px;flex-wrap:wrap;margin-bottom:18px">{left}'
            f"{self._badge_html(header)}</div>"
        )

    def _badge_html(self, header: Header) -> str:
        if header.badge.kind == "done":
            color, bg = _JUP_DONE, _JUP_DONE_BG
        else:
            color, bg = _JUP_INFO, _JUP_INFO_BG
        label = header.badge.label
        dot = f'<span style="width:7px;height:7px;border-radius:50%;background:{color}"></span>'
        return (
            f'<span style="display:inline-flex;align-items:center;gap:6px;background:{bg};'
            f'color:{color};font-size:12px;font-weight:500;padding:5px 12px;border-radius:8px">'
            f"{dot}{label}</span>"
        )

    # ---------------------------------------------------------------------- cards

    def _cards_html(self, cards: StatCards) -> str:
        body = "".join(
            self._card_html(
                s.label,
                f"{s.value}{self._suffix(s.suffix)}",
                mono=s.mono,
            )
            for s in cards.cards
        )
        return (
            '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));'
            f'gap:12px;margin-bottom:20px">{body}</div>'
        )

    def _suffix(self, suffix: str | None) -> str:
        if suffix is None:
            return ""
        return f'<span style="font-size:14px;color:{_JUP_DIM}">{suffix}</span>'

    def _card_html(self, label: str, value: str, *, mono: bool = False) -> str:
        font = f";font-family:{_JUP_MONO}" if mono else ""
        return (
            f'<div style="background:{_JUP_CARD};border-radius:10px;padding:14px 16px">'
            f'<div style="font-size:13px;color:{_JUP_DIM};margin-bottom:4px">{label}</div>'
            f'<div style="font-size:24px;font-weight:500{font}">{value}</div></div>'
        )

    def _card_from_stat(self, cards: StatCards) -> str:
        s = cards.cards[0]
        return self._card_html(s.label, f"{s.value}{self._suffix(s.suffix)}", mono=s.mono)

    def _score_cards_html(self, cards: StatCards, predictions: Predictions) -> str:
        body = "".join(
            self._card_html(s.label, f"{s.value}{self._suffix(s.suffix)}", mono=s.mono)
            for s in cards.cards
        )
        body += self._bins_card_html(predictions)
        return (
            '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));'
            f'gap:12px;margin-bottom:20px">{body}</div>'
        )

    def _bins_card_html(self, predictions: Predictions) -> str:
        mix = predictions.bins
        peak = max((b.count for b in mix), default=1)
        rows = "".join(self._bin_row_html(b, peak) for b in mix)
        body = (
            f'<div style="display:flex;flex-direction:column;gap:6px;margin-top:2px">{rows}</div>'
        )
        return (
            f'<div style="background:{_JUP_CARD};border-radius:10px;padding:14px 16px">'
            f'<div style="font-size:13px;color:{_JUP_DIM};margin-bottom:8px">'
            f"{predictions.bins_label}</div>{body}</div>"
        )

    def _bin_row_html(self, b: BinShare, peak: int) -> str:
        color = _JUP_OTHERS if b.label == "others" else self._color_for(b.label)
        width = b.count / peak * 100
        bar = (
            f'<span style="flex:1;height:7px;border-radius:4px;background:{_JUP_CARD};'
            'overflow:hidden">'
            f'<span style="display:block;height:100%;border-radius:4px;width:{width:.1f}%;'
            f'background:{color}"></span></span>'
        )
        return (
            f'<div style="display:flex;align-items:center;gap:8px;font-family:{_JUP_MONO};'
            'font-size:11.5px">'
            f'<span style="width:64px;color:{_JUP_DIM};overflow:hidden;text-overflow:ellipsis;'
            f'white-space:nowrap">{html.escape(_jcand(b.label))}</span>'
            f'{bar}<span style="width:16px;text-align:right;font-weight:600">{b.count}</span></div>'
        )

    # ---------------------------------------------------------------------- chart

    def _chart_panel(self, curve: LossCurve) -> str:
        if curve.train:
            ema = self._loss_smoothing != 0
            sub = curve.subtitle + (" · ema" if ema else "")
            plot = (
                '<div style="position:relative;flex:1;min-height:150px">'
                f"{self._chart_svg(curve)}</div>"
            )
        else:
            sub = ""
            plot = (
                f'<div style="flex:1;min-height:150px;display:flex;align-items:center;'
                f'justify-content:center;color:{_JUP_DIM};font-size:13px">'
                "waiting for the first loss report…</div>"
            )
        head = (
            '<div style="display:flex;align-items:center;justify-content:space-between;'
            'margin-bottom:8px"><span style="font-weight:500;font-size:14px">training loss'
            f'</span><span style="font-size:12px;color:{_JUP_DIM}">{sub}</span></div>'
        )
        return (
            f'<div style="border:0.5px solid {_JUP_BORDER};border-radius:14px;'
            f'padding:14px 16px;display:flex;flex-direction:column">{head}{plot}</div>'
        )

    def _chart_svg(self, curve: LossCurve) -> str:
        """The loss curve as an inline, stretchable SVG with HTML axis labels.

        The ``<svg>`` uses ``preserveAspectRatio="none"`` so the curve fills the
        panel at any size, with ``vector-effect="non-scaling-stroke"`` keeping the
        line crisp; the tick labels live in absolutely-positioned HTML so they
        never distort. A sharp early drop flattens the tail on a linear axis, so
        the y axis goes log once ``max / min`` exceeds 4.

        The per-step loss is usually too noisy to plot raw, so the train line is an
        exponential moving average (see ``loss_smoothing``) and the y range is
        taken from that smoothed series -- raw single-step dips toward zero would
        otherwise blow the log axis out to many empty decades. Smoothing off
        (``0``) plots the raw loss and ranges over it.
        """
        losses = curve.train
        steps = curve.steps
        if self._loss_smoothing is None:
            alpha = max(0.03, min(0.3, 30 / len(losses)))
        else:
            alpha = self._loss_smoothing
        train = _ema(losses, alpha) if alpha > 0 else list(losses)
        span = [*train, *curve.evals]
        lo, hi = min(span), max(span)
        use_log = lo > 0 and hi / lo > 4
        tf: Callable[[float], float] = math.log10 if use_log else (lambda v: v)
        tlo, thi = tf(lo), tf(hi)
        pad = (thi - tlo) * 0.06 or 0.1
        ymin_t, ymax_t = tlo - pad, thi + pad
        smin, smax = steps[0], steps[-1]
        sden = (smax - smin) or 1

        def xf(step: float) -> float:
            return (step - smin) / sden

        def yf(y: float) -> float:
            return (ymax_t - tf(y)) / (ymax_t - ymin_t)

        yticks = _log_ticks(lo, hi) if use_log else _nice_ticks(lo, hi)
        xticks = [v for v in _nice_ticks(float(smin), float(smax)) if smin <= v <= smax]

        gline = ' vector-effect="non-scaling-stroke"'
        grid = "".join(
            f'<line x1="0" y1="{yf(v) * 1000:.1f}" x2="1000" y2="{yf(v) * 1000:.1f}"{gline}/>'
            for v in yticks
        ) + "".join(
            f'<line x1="{xf(v) * 1000:.1f}" y1="0" x2="{xf(v) * 1000:.1f}" y2="1000"{gline}/>'
            for v in xticks
        )

        def polyline(pts: list[tuple[int, float]], color: str, width: float) -> str:
            coords = " ".join(f"{xf(st) * 1000:.1f},{yf(y) * 1000:.1f}" for st, y in pts)
            return (
                f'<polyline points="{coords}" fill="none" stroke="{color}" '
                f'stroke-width="{width}" vector-effect="non-scaling-stroke" '
                'stroke-linejoin="round" stroke-linecap="round"/>'
            )

        curves = polyline(
            _downsample(list(zip(steps, train, strict=True)), 600), self._train_color, 1.6
        )
        if curve.evals:
            ev = _downsample(list(zip(curve.eval_steps, curve.evals, strict=True)), 600)
            curves += polyline(ev, self._eval_color, 1.4)

        svg = (
            '<svg viewBox="0 0 1000 1000" preserveAspectRatio="none" '
            'style="position:absolute;inset:0;width:100%;height:100%">'
            f'<g stroke="{_JUP_DIM}" stroke-opacity="0.2" stroke-width="1">{grid}</g>'
            f"{curves}</svg>"
        )
        ylabels = "".join(
            f'<div style="position:absolute;left:0;width:24px;text-align:right;'
            f"top:calc(6px + {yf(v):.4f} * (100% - 30px));transform:translateY(-50%);"
            f'font-size:11px;color:{_JUP_DIM}">{v:g}</div>'
            for v in yticks
        )
        xlabels = "".join(
            f'<div style="position:absolute;bottom:9px;'
            f"left:calc(28px + {xf(v):.4f} * (100% - 36px));transform:translateX(-50%);"
            f'font-size:11px;color:{_JUP_DIM}">{int(v):,}</div>'
            for v in xticks
        )
        axis = (
            f'<div style="position:absolute;bottom:0;left:calc(28px + 0.5 * (100% - 36px));'
            f'transform:translateX(-50%);font-size:11px;color:{_JUP_DIM}">step</div>'
        )
        plot = f'<div style="position:absolute;top:6px;left:28px;right:8px;bottom:24px">{svg}</div>'
        return plot + ylabels + xlabels + axis

    # ----------------------------------------------------------------- hyperparams

    def _hyper_panel(self, hyper: KeyValues) -> str:
        body = "".join(
            self._hyper_row(kv.label, kv.value, mono=kv.mono, first=i == 0)
            for i, kv in enumerate(hyper.rows)
        )
        head = (
            '<div style="font-weight:500;font-size:14px;margin-bottom:12px">hyperparameters</div>'
        )
        table = f'<table style="width:100%;font-size:13px;border-collapse:collapse">{body}</table>'
        return (
            f'<div style="border:0.5px solid {_JUP_BORDER};border-radius:14px;'
            f'padding:14px 16px;display:flex;flex-direction:column">'
            f"{head}{table}{self._elapsed_html(hyper)}</div>"
        )

    def _hyper_row(self, label: str, value: str, *, mono: bool, first: bool) -> str:
        top = "" if first else f";border-top:0.5px solid {_JUP_BORDER}"
        font = f";font-family:{_JUP_MONO}" if mono else ""
        return (
            f'<tr><td style="color:{_JUP_DIM};padding:6px 0{top}">{label}</td>'
            f'<td style="text-align:right;padding:6px 0{top}{font}">{value}</td></tr>'
        )

    def _elapsed_html(self, hyper: KeyValues) -> str:
        if hyper.elapsed is None:
            return ""
        value = hyper.elapsed if hyper.eta is None else f"{hyper.elapsed} · {hyper.eta}"
        return (
            f'<div style="margin-top:auto;padding-top:12px;border-top:0.5px solid {_JUP_BORDER};'
            'display:flex;align-items:center;justify-content:space-between">'
            f'<span style="font-size:13px;color:{_JUP_DIM}">elapsed</span>'
            f'<span style="font-size:13px;font-family:{_JUP_MONO}">{value}</span></div>'
        )

    # ------------------------------------------------------------------- examples

    def _examples_html(self, panel: ExamplesPanel) -> str:
        if panel.parsed is not None and panel.view == "table":
            view, body = "table", self._examples_table_html(*panel.parsed)
        else:
            view, body = "raw", self._examples_raw_html(panel.texts)
        head = (
            '<div style="display:flex;align-items:center;justify-content:space-between;'
            'margin-bottom:10px"><span style="font-weight:500;font-size:14px">training examples'
            f'</span><span style="font-size:12px;color:{_JUP_DIM}">'
            f"epoch {panel.epoch} · {view}</span></div>"
        )
        return f'<div style="margin-top:18px">{head}{self._panel(body)}</div>'

    def _examples_panel_html(self, panel: ExamplesPanel) -> str:
        """The examples panel with CSS-only ``table``/``raw`` tabs.

        Both views are rendered and toggled by a scoped ``:checked`` radio, so the
        switch is pure CSS -- it works while ``fit()`` blocks the kernel. Rewritten
        only per epoch (not per refresh), so the selection survives within an epoch.
        """
        texts = panel.texts
        if not texts:
            return ""
        title = (
            '<span style="font-weight:500;font-size:14px">training examples</span>'
            f'<span style="font-size:12px;color:{_JUP_DIM};margin-left:10px">'
            f"epoch {panel.epoch}</span>"
        )
        if panel.parsed is None:
            head = (
                '<div style="display:flex;justify-content:space-between;align-items:center;'
                f'margin-bottom:10px">{title}</div>'
            )
            inner = head + self._examples_raw_html(texts)
            return f'<div style="max-width:760px;margin-top:18px">{self._panel(inner)}</div>'
        uid = panel.uid
        raw = panel.view == "raw"
        style = (
            f"<style>#{uid} input{{position:absolute;opacity:0;pointer-events:none}}"
            f"#{uid} .v{{display:none}}"
            f"#{uid} #{uid}-t:checked~.b .v-t{{display:block}}"
            f"#{uid} #{uid}-r:checked~.b .v-r{{display:block}}"
            f"#{uid} label{{cursor:pointer;font-size:12px;font-weight:600;color:{_JUP_DIM};"
            "padding:3px 12px;border-radius:7px}"
            f'#{uid} #{uid}-t:checked~.h label[for="{uid}-t"],'
            f'#{uid} #{uid}-r:checked~.h label[for="{uid}-r"]'
            f"{{color:{_JUP_INFO};background:{_JUP_INFO_BG}}}</style>"
        )
        inputs = (
            f'<input type="radio" name="{uid}" id="{uid}-t"{"" if raw else " checked"}>'
            f'<input type="radio" name="{uid}" id="{uid}-r"{" checked" if raw else ""}>'
        )
        tabs = (
            f'<span style="display:inline-flex;gap:2px;background:{_JUP_CARD};border-radius:9px;'
            f'padding:3px"><label for="{uid}-t">table</label>'
            f'<label for="{uid}-r">raw</label></span>'
        )
        head = (
            '<div class="h" style="display:flex;justify-content:space-between;align-items:center;'
            f'margin-bottom:10px">{title}{tabs}</div>'
        )
        body = (
            f'<div class="b"><div class="v v-t">{self._examples_table_html(*panel.parsed)}</div>'
            f'<div class="v v-r">{self._examples_raw_html(texts)}</div></div>'
        )
        wrapped = self._panel(style + inputs + head + body)
        return f'<div id="{uid}" style="max-width:760px;margin-top:18px">{wrapped}</div>'

    def _examples_raw_html(self, texts: Sequence[str]) -> str:
        rows = "".join(self._example_row(i, text) for i, text in enumerate(texts))
        return (
            f'<div style="font-family:{_JUP_MONO};font-size:12.5px;line-height:1.85;'
            f'overflow-x:auto">{rows}</div>'
        )

    def _examples_table_html(
        self, columns: Sequence[str], rows: Sequence[dict[str, object]]
    ) -> str:
        th = (
            f'<th style="text-align:left;font-weight:600;font-size:10px;letter-spacing:.04em;'
            f"text-transform:uppercase;color:{_JUP_DIM};padding:0 12px 8px 0;"
            f'border-bottom:0.5px solid {_JUP_BORDER}">#</th>'
        )
        for c in columns:
            th += (
                f'<th style="text-align:right;font-weight:600;font-size:10px;letter-spacing:.04em;'
                f"text-transform:uppercase;color:{_JUP_DIM};padding:0 0 8px 12px;"
                f'border-bottom:0.5px solid {_JUP_BORDER}">{html.escape(c)}</th>'
            )
        body = "".join(self._examples_table_row(i, columns, r) for i, r in enumerate(rows))
        return (
            '<div style="overflow-x:auto">'
            f'<table style="width:100%;border-collapse:collapse;font-family:{_JUP_MONO};'
            f'font-size:12.5px;font-variant-numeric:tabular-nums">'
            f"<thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>"
        )

    def _examples_table_row(
        self, index: int, columns: Sequence[str], row: dict[str, object]
    ) -> str:
        cells = (
            f'<td style="text-align:left;color:{_JUP_DIM};padding:7px 12px 7px 0;'
            f'border-bottom:0.5px solid {_JUP_BORDER}">{index + 1:02d}</td>'
        )
        for c in columns:
            if c not in row:
                value, color = "—", _JUP_DIM
            elif isinstance(row[c], str):
                value, color = html.escape(str(row[c])), _JUP_TK_STR
            else:
                value, color = html.escape(_jpredval(row[c])), None
            tint = f";color:{color}" if color else ""
            cells += (
                f'<td style="text-align:right;padding:7px 0 7px 12px;'
                f'border-bottom:0.5px solid {_JUP_BORDER}{tint}">{value}</td>'
            )
        return f"<tr>{cells}</tr>"

    def _example_row(self, index: int, text: str) -> str:
        code = _highlight_json(text)
        rendered = code if code is not None else html.escape(text)
        num = (
            f'<span style="display:inline-block;width:2em;text-align:right;'
            f'margin-right:1.2em;color:{_JUP_DIM};user-select:none">{index + 1:02d}</span>'
        )
        return f'<div style="white-space:nowrap">{num}{rendered}</div>'

    # ------------------------------------------------------------------ log table

    def _log_table_html(self, log: LogTable) -> str:
        head = (
            '<div style="display:flex;align-items:center;justify-content:space-between;'
            'margin-bottom:10px"><span style="font-weight:500;font-size:14px">training log'
            f'</span><span style="font-size:12px;color:{_JUP_DIM}">{log.cadence}'
            "</span></div>"
        )
        cols = ("step", "epoch", "train loss", "eval loss", "lr", "grad norm", "steps/s")
        header = "".join(
            f'<th style="text-align:{"left" if i < 2 else "right"};font-weight:600;'
            f"font-size:10px;letter-spacing:.04em;text-transform:uppercase;color:{_JUP_DIM};"
            f'padding:0 0 8px;border-bottom:0.5px solid {_JUP_BORDER}">{c}</th>'
            for i, c in enumerate(cols)
        )
        rows = "".join(self._log_row_html(r) for r in log.rows)
        table = (
            f'<table style="width:100%;font-family:{_JUP_MONO};font-size:12px;'
            f'border-collapse:collapse;font-variant-numeric:tabular-nums">'
            f"<thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>"
        )
        return f'<div style="max-width:760px;margin-top:18px">{head}{self._panel(table)}</div>'

    def _log_row_html(self, r: LogRow) -> str:
        std = (
            f' <span style="color:{_JUP_DIM};font-size:10.5px">± {r.std:.3f}</span>'
            if r.std is not None
            else ""
        )
        sps = f"{r.steps_per_second:.1f}" if r.steps_per_second is not None else ""
        cells = [
            (f"{r.step:,}", "left", _JUP_DIM),
            (f"{r.epoch:.1f}" if r.epoch is not None else "—", "left", None),
            (f"{r.loss:.4f}{std}" if r.loss is not None else "—", "right", None),
            (
                f"{r.eval_loss:.4f}" if r.eval_loss is not None else "—",
                "right",
                _JUP_DONE if r.eval_loss is not None else _JUP_DIM,
            ),
            (_sci(r.lr) if r.lr is not None else "—", "right", _JUP_DIM),
            (f"{r.grad_norm:.2f}" if r.grad_norm is not None else "—", "right", None),
            (sps or "—", "right", _JUP_DIM),
        ]
        tds = "".join(
            f'<td style="text-align:{align};padding:7px 0;border-bottom:0.5px solid '
            f'{_JUP_BORDER}{f";color:{color}" if color else ""}">{value}</td>'
            for value, align, color in cells
        )
        return f"<tr>{tds}</tr>"

    # ----------------------------------------------------------------- predictions

    def _generations_html(self, predictions: Predictions) -> str:
        scoring = predictions.scoring
        tag = (
            "given fields → predicted distribution"
            if scoring
            else "given fields → most likely next field"
        )
        head = (
            '<div style="display:flex;align-items:baseline;justify-content:space-between;'
            'gap:8px;margin-bottom:12px"><span style="font-weight:500;font-size:14px">'
            f'recent predictions</span><span style="font-size:12px;color:{_JUP_DIM}">'
            f"{tag}</span></div>"
        )
        if not predictions.rows:
            return head + _jline("waiting…", color=_JUP_DIM)
        return head + self._predictions_table(predictions)

    def _predictions_table(self, predictions: Predictions) -> str:
        if predictions.scoring:
            columns: tuple[tuple[str, str | None], ...] = (
                ("#", "36px"),
                ("given", None),
                ("distribution", None),
            )
            body = "".join(
                self._score_row(i, r)
                for i, r in enumerate(predictions.rows)
                if isinstance(r, ScoreRow)
            )
        else:
            columns = (("#", "36px"), ("given", None), ("next", "38%"))
            body = "".join(
                self._gen_row(i, r)
                for i, r in enumerate(predictions.rows)
                if isinstance(r, GenerationRow)
            )
        cols = "".join(
            f'<th style="text-align:left;font-weight:500;color:{_JUP_DIM};font-size:10.5px;'
            "letter-spacing:.05em;text-transform:uppercase;padding:0 12px 8px 0;"
            f'border-bottom:0.5px solid {_JUP_BORDER}{f";width:{w}" if w else ""}">{c}</th>'
            for c, w in columns
        )
        return (
            '<table style="width:100%;border-collapse:collapse;table-layout:fixed;'
            f'font-family:{_JUP_MONO};font-size:12.5px"><thead><tr>{cols}</tr></thead>'
            f"<tbody>{body}</tbody></table>"
        )

    def _td(self, inner: str, extra: str = "") -> str:
        # text-align:left is explicit because Jupyter's rendered-HTML CSS defaults
        # table cells to right, which would split them from the left-aligned headers.
        return (
            f'<td style="padding:9px 12px 9px 0;border-bottom:0.5px solid {_JUP_BORDER};'
            f'text-align:left;vertical-align:middle;{extra}">{inner}</td>'
        )

    def _index_td(self, index: int) -> str:
        return self._td(str(index + 1), f"color:{_JUP_DIM};font-size:11.5px;vertical-align:top")

    def _given_td(self, prompt: str) -> str:
        wrap = "word-break:break-word;line-height:1.5;vertical-align:top"
        return self._td(_highlight_json_fragment(prompt), wrap)

    def _score_row(self, index: int, r: ScoreRow) -> str:
        warn = (
            '<span title="below uncertain threshold" style="color:#b4671a">⚠</span>'
            if r.uncertain
            else ""
        )
        return (
            "<tr>"
            + self._index_td(index)
            + self._given_td(r.prompt)
            + self._td(self._dist_badges(list(r.ranking), warn), "min-width:0;vertical-align:top")
            + "</tr>"
        )

    def _gen_row(self, index: int, r: GenerationRow) -> str:
        return (
            "<tr>"
            + self._index_td(index)
            + self._given_td(r.prompt)
            + self._td(self._next_span(_jpredval(r.value)))
            + "</tr>"
        )

    def _next_span(self, text: str) -> str:
        return (
            f'<span style="color:{_JUP_INFO};font-weight:600;background:{_JUP_INFO_BG};'
            'padding:1px 6px;border-radius:6px">'
            '<span style="opacity:.6;font-size:11px;margin-right:3px">▸</span>'
            f"{html.escape(text)}</span>"
        )

    def _dist_badges(self, ranking: list[tuple[object, float]], trailing: str = "") -> str:
        head = ranking[: self._top_k]
        badges = [self._prob_badge(c, p, self._color_for(c)) for c, p in head]
        others = sum(p for _, p in ranking[self._top_k :])
        if others > 1e-4:
            badges.append(self._prob_badge("others", others, _JUP_OTHERS))
        if trailing:
            badges.append(trailing)
        return (
            '<div style="display:flex;flex-wrap:wrap;align-items:center;gap:6px;min-width:0">'
            + "".join(badges)
            + "</div>"
        )

    def _prob_badge(self, label: object, p: float, color: str) -> str:
        return (
            f'<span style="color:{color};background:{color}22;padding:2px 9px;border-radius:6px;'
            f'white-space:nowrap;font-weight:600;font-size:11.5px">'
            f"{html.escape(_jcand(label))} → {p:.2f}</span>"
        )
