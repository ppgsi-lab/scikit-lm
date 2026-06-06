"""Renderer-agnostic dashboard model shared across the live callbacks.

:mod:`widgets` is the pure-data widget tree, :class:`model.DashboardState` the
shared event accumulator, :mod:`build` the (TrainingState + DashboardState) ->
:class:`widgets.Dashboard` mapping, and the renderers walk that tree:
:class:`html_renderer.HtmlRenderer` emits Jupyter HTML/SVG,
:class:`rich_renderer.RichRenderer` emits terminal renderables. Both callbacks
fold events through the same :class:`model.DashboardState` and build the same
widget :class:`widgets.Dashboard`, differing only in the renderer.
"""

from __future__ import annotations

from .build import RenderConfig, build_fit_dashboard, build_predict_dashboard
from .html_renderer import HtmlRenderer
from .model import DashboardState

__all__ = [
    "DashboardState",
    "HtmlRenderer",
    "RenderConfig",
    "build_fit_dashboard",
    "build_predict_dashboard",
]
