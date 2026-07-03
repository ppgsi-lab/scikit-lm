"""Observability for fitting and inference.

The base :class:`Callback` is both the wire protocol the backends/``core`` call
(13 granular ``on_*`` events) **and** the aggregator: each event folds into a
:class:`TrainingState` and is then dispatched, together with that state, through
a single :meth:`Callback.on_event` hook. Subclass and override ``on_event`` --
the granular methods are ``@final``; overriding one would silently lose the
aggregation, so react in ``on_event`` instead.

The shipped dashboards (:class:`LoggingCallback`, :class:`TqdmCallback`,
:class:`RichCallback`) are themselves just such subclasses. The library never
configures the root logger -- handlers and levels are the caller's choice.
"""

from __future__ import annotations

import importlib.util
from typing import Literal

from .base import Callback, CompositeCallback, predict_batches
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
from .format import _spread_summary as _spread_summary
from .jupyter import _JUP_TK_KEY as _JUP_TK_KEY
from .jupyter import JupyterCallback
from .logging import LoggingCallback
from .rich import RichCallback
from .tqdm import TqdmCallback

__all__ = [
    "Callback",
    "CompositeCallback",
    "EvalReport",
    "Event",
    "FitEnd",
    "FitInfo",
    "FitStart",
    "Generation",
    "JupyterCallback",
    "LoggingCallback",
    "Memory",
    "PredictEnd",
    "PredictStart",
    "Retry",
    "RichCallback",
    "RowEnd",
    "Score",
    "TqdmCallback",
    "TrainExamples",
    "TrainReport",
    "TrainingState",
    "predict_batches",
    "resolve_callback",
]


def _in_jupyter() -> bool:
    """Whether we are executing inside a Jupyter kernel (not a plain IPython shell)."""
    if importlib.util.find_spec("IPython") is None:
        return False
    from IPython.core.getipython import get_ipython

    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


def _has_rich() -> bool:
    """Whether the ``rich`` extra (rich + plotext) is importable."""
    return (
        importlib.util.find_spec("rich") is not None
        and importlib.util.find_spec("plotext") is not None
    )


def _auto_callback() -> Callback:
    """Pick a live dashboard from the runtime environment.

    A Jupyter kernel gets :class:`JupyterCallback`, the ``rich`` extra
    :class:`RichCallback`, and everything else :class:`LoggingCallback`. The
    environment-specific choices fall through to the next option if their
    optional deps are missing, so :class:`LoggingCallback` (no extras) is always
    reachable.
    """
    candidates: list[type[Callback]] = []
    if _in_jupyter():
        candidates.append(JupyterCallback)
    if _has_rich():
        candidates.append(RichCallback)
    for build in candidates:
        try:
            return build()
        except ImportError:
            continue
    return LoggingCallback()


def resolve_callback(callback: Callback | list[Callback] | Literal["auto"] | None) -> Callback:
    """Resolve the estimator ``callback`` argument to a single :class:`Callback`.

    ``"auto"`` selects a dashboard for the runtime environment
    (:func:`_auto_callback`); ``None`` means no feedback (the no-op base
    :class:`Callback`); a list is wrapped in a :class:`CompositeCallback`;
    a single :class:`Callback` is returned unchanged.
    """
    if callback == "auto":
        return _auto_callback()
    if callback is None:
        return Callback()
    if isinstance(callback, Callback):
        return callback
    return CompositeCallback(callback)
