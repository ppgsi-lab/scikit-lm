"""The :class:`Callback` contract, tested generically against the base class.

A single :class:`RecordingCallback` observes the event stream through the public
estimator API; concrete dashboards (logging / tqdm / rich / jupyter) are only
smoke-checked to run end-to-end, never asserted on their rendered output. What
they derive on top of the event stream is asserted one level down, on the
renderer-agnostic widget tree.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import override

import pandas as pd
import pytest
from sklearn.base import clone

from sklm import (
    Callback,
    CompositeCallback,
    EvalConfig,
    Event,
    FitEnd,
    FitStart,
    Generation,
    JupyterCallback,
    LanguageModelClassifier,
    LanguageModelImputer,
    LoggingCallback,
    PredictEnd,
    PredictStart,
    Retry,
    RichCallback,
    RowEnd,
    Score,
    TqdmCallback,
    TrainExamples,
    TrainingConfig,
    TrainingState,
)
from sklm.callbacks import resolve_callback
from sklm.callbacks.dashboard import DashboardState, RenderConfig, build_fit_dashboard
from sklm.callbacks.dashboard.widgets import Stat, StatCards

from .conftest import FakeBackend


class RecordingCallback(Callback):
    """Record the event stream and per-event payloads for contract assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, tuple[object, ...]]] = []
        self.retries: list[tuple[str, int, int]] = []
        self.generations: list[tuple[str, str, str, object]] = []
        self.scorings: list[tuple[list[str], list[object], list[list[float]]]] = []

    @override
    def on_event(self, state: TrainingState, event: Event) -> None:
        match event:
            case FitStart(n_rows=n_rows, n_epochs=n_epochs):
                self.events.append(("fit_start", (n_rows, n_epochs)))
            case TrainExamples():
                self.events.append(("train_examples", ()))
            case FitEnd():
                self.events.append(("fit_end", ()))
            case PredictStart(n=n):
                self.events.append(("predict_start", (n,)))
            case RowEnd(index=index, total=total):
                self.events.append(("row_end", (index, total)))
            case PredictEnd():
                self.events.append(("predict_end", ()))
            case Generation(prompt=prompt, completion=completion, target=target, value=value):
                self.events.append(("generation", ()))
                self.generations.append((prompt, completion, target, value))
            case Score(prompts=prompts, candidates=candidates, probs=probs):
                self.events.append(("score", ()))
                self.scorings.append((list(prompts), list(candidates), [list(p) for p in probs]))
            case Retry(target=target, attempt=attempt, max_attempts=max_attempts):
                self.events.append(("retry", ()))
                self.retries.append((target, attempt, max_attempts))
            case _:
                pass

    def names(self) -> list[str]:
        return [name for name, _ in self.events]


def test_lifecycle_event_order(nan_data: pd.DataFrame) -> None:
    rec = RecordingCallback()
    imp = LanguageModelImputer(
        backend=FakeBackend(value="0"),
        callback=rec,
        # no hold-out, so fit_start reports the whole frame
        training=TrainingConfig(evaluation=None),
    )
    imp.fit(nan_data)
    assert rec.names() == ["fit_start", "train_examples", "fit_end"]
    assert rec.events[0] == ("fit_start", (len(nan_data), imp.training.epochs))

    rec.events.clear()
    imp.transform(nan_data)
    names = rec.names()
    assert names[0] == "predict_start"
    assert names[-1] == "predict_end"
    assert rec.events[0] == ("predict_start", (len(nan_data),))
    assert names.count("row_end") == len(nan_data)


def test_state_accumulates_across_train_reports() -> None:
    rec = RecordingCallback()
    rec.on_fit_start(n_rows=10, n_epochs=2)
    for step in range(1, 5):
        rec.on_train_report(
            step=step, total_steps=4, loss=1.0 / step, epoch=None, learning_rate=1e-4
        )
    # the base folds the granular reports into one running state
    assert rec.state.steps == [1, 2, 3, 4]
    assert rec.state.loss == pytest.approx(0.25)
    assert rec.state.epoch == pytest.approx(2.0)  # derived from step / total * n_epochs


def test_retry_then_raise_on_malformed_generation(nan_data: pd.DataFrame) -> None:
    rec = RecordingCallback()
    imp = LanguageModelImputer(backend=FakeBackend(value="nope"), callback=rec).fit(nan_data)
    with pytest.raises(RuntimeError, match="not producing valid values"):
        imp.transform(nan_data)
    assert "retry" in rec.names()
    assert all(1 <= attempt <= mx for _, attempt, mx in rec.retries)


def test_generation_event_exposes_prompt_and_output(nan_data: pd.DataFrame) -> None:
    rec = RecordingCallback()
    imp = LanguageModelImputer(backend=FakeBackend(value="7"), callback=rec).fit(nan_data)
    rec.events.clear()
    imp.transform(nan_data)
    # nan_data misses two numeric `age` cells (generated) and one `city` cell (scored,
    # which rides a score event, not a generation one)
    assert len(rec.generations) == 2
    assert "retry" not in rec.names()
    for prompt, completion, target, value in rec.generations:
        assert completion == "7"
        assert target == "age"
        assert '"age": ' in prompt
        assert value is not None  # the decoded value rides on the event


def test_score_event_exposes_prompt_and_distribution(clf_data) -> None:
    X, y = clf_data
    rec = RecordingCallback()
    clf = LanguageModelClassifier(backend=FakeBackend(), callback=rec).fit(X, y)
    rec.events.clear()
    clf.predict(X)
    assert "generation" not in rec.names()  # the classifier ranks, never generates
    assert len(rec.scorings) == len(X)  # one call per row
    classes = list(clf.classes_)
    for prompts, candidates, probs in rec.scorings:
        assert list(candidates) == classes
        assert len(prompts) == len(probs) == 1  # one order without permute
        assert abs(sum(probs[0]) - 1.0) < 1e-9
        assert f'"{clf.target_col_}": ' in prompts[0]


def test_composite_fans_events_out_to_children(nan_data: pd.DataFrame) -> None:
    first, second = RecordingCallback(), RecordingCallback()
    imp = LanguageModelImputer(backend=FakeBackend(value="0"), callback=[first, second]).fit(
        nan_data
    )
    imp.transform(nan_data)
    assert first.names() == second.names()
    assert "predict_start" in first.names()
    assert first.names().count("row_end") == len(nan_data)


def test_explicit_noop_callback_imputes(nan_data: pd.DataFrame) -> None:
    imp = LanguageModelImputer(backend=FakeBackend(value="0"), callback=Callback())
    out = imp.fit_transform(nan_data)
    assert not out.isna().any().any()


def test_resolve_callback_passes_single_through() -> None:
    cb = LoggingCallback()
    assert resolve_callback(cb) is cb


def test_resolve_callback_wraps_list_in_composite() -> None:
    a, b = LoggingCallback(), LoggingCallback()
    resolved = resolve_callback([a, b])
    assert isinstance(resolved, CompositeCallback)
    assert resolved.callbacks == [a, b]


def test_resolve_callback_auto_selects_a_dashboard() -> None:
    resolved = resolve_callback("auto")
    assert isinstance(resolved, Callback)
    assert type(resolved) is not Callback  # a real dashboard, not the no-op base


def test_resolve_callback_none_is_noop() -> None:
    assert type(resolve_callback(None)) is Callback


def test_callback_defaults_to_auto() -> None:
    assert LanguageModelImputer(backend=FakeBackend()).callback == "auto"


def test_clone_preserves_callback() -> None:
    imp = LanguageModelImputer(backend=FakeBackend(), callback=LoggingCallback())
    cloned = clone(imp)
    assert isinstance(cloned, LanguageModelImputer)
    assert isinstance(cloned.callback, LoggingCallback)


@pytest.mark.parametrize(
    ("build", "deps"),
    [
        (LoggingCallback, ()),
        (TqdmCallback, ("tqdm",)),
        (RichCallback, ("rich", "plotext")),
        (JupyterCallback, ("ipywidgets",)),
    ],
)
def test_dashboard_runs_end_to_end(
    build: type[Callback], deps: Sequence[str], nan_data: pd.DataFrame, clf_data
) -> None:
    """Every shipped dashboard drives a full fit + inference without raising,
    across the generation (imputer) and score (classifier) paths."""
    for dep in deps:
        pytest.importorskip(dep)
    out = LanguageModelImputer(backend=FakeBackend(value="0"), callback=build()).fit_transform(
        nan_data
    )
    assert not out.isna().any().any()

    X, y = clf_data
    clf = LanguageModelClassifier(backend=FakeBackend(), callback=build()).fit(X, y)
    assert set(clf.predict(X)).issubset(set(clf.classes_))


class _FoldingCallback(Callback):
    """Fold the event stream into a :class:`DashboardState`, as the live dashboards do."""

    def __init__(self, dash: DashboardState) -> None:
        super().__init__()
        self.dash = dash

    @override
    def on_event(self, state: TrainingState, event: Event) -> None:
        self.dash.fold(state, event)


def _fit_cards(training: TrainingConfig, evals: Sequence[float]) -> Sequence[Stat]:
    """The fit stat cards after one eval report per epoch, at losses ``evals``."""
    dash = DashboardState(n_generations=0, uncertain_threshold=0.6, log_every="epoch")
    cb = _FoldingCallback(dash)
    cb.on_fit_info("model", training)
    cb.on_fit_start(n_rows=10, n_epochs=len(evals))
    for epoch, loss in enumerate(evals, start=1):
        cb.on_train_report(
            step=epoch, total_steps=len(evals), loss=loss, epoch=float(epoch), learning_rate=1e-4
        )
        cb.on_eval_report(step=epoch, loss=loss, epoch=float(epoch))
    cfg = RenderConfig(
        n_train_examples=0, examples_view="raw", n_log_rows=0, log_every="epoch", uid="test"
    )
    dashboard = build_fit_dashboard(cb.state, dash, cfg)
    return next(c for c in dashboard.children if isinstance(c, StatCards)).cards


def test_patience_card_counts_evals_since_best() -> None:
    cards = _fit_cards(
        TrainingConfig(evaluation=EvalConfig(split=0.2, patience=3)), evals=[0.9, 0.5, 0.6, 0.7]
    )
    assert cards[-1].label == "patience"
    # best at epoch 2; the two worse evals after it spent 2 of the 3 budget
    assert cards[-1].value == "2 / 3"
    assert cards[-1].note == "best 0.5000 · ep 2"


def test_patience_card_resets_on_a_new_best() -> None:
    cards = _fit_cards(
        TrainingConfig(evaluation=EvalConfig(split=0.2, patience=3)), evals=[0.9, 0.6, 0.4]
    )
    assert cards[-1].value == "0 / 3"
    assert cards[-1].note == "best 0.4000 · ep 3"


def test_best_card_replaces_patience_without_a_budget() -> None:
    cards = _fit_cards(
        TrainingConfig(evaluation=EvalConfig(split=0.2, patience=None)), evals=[0.9, 0.5, 0.6]
    )
    assert cards[-1].label == "best val"
    assert (cards[-1].value, cards[-1].note) == ("0.5000", "ep 2")


def test_no_early_stopping_card_without_validation() -> None:
    cards = _fit_cards(TrainingConfig(evaluation=None), evals=[])
    assert [c.label for c in cards] == ["train loss", "val loss", "learning rate", "step"]


def test_eval_reports_fold_best_and_patience_into_state() -> None:
    cb = Callback()
    cb.on_fit_start(n_rows=10, n_epochs=4)
    for epoch, loss in enumerate([0.9, 0.5, 0.6, 0.7], start=1):
        cb.on_eval_report(step=epoch, loss=loss, epoch=float(epoch))
    assert cb.state.best_eval == 0.5
    assert cb.state.best_eval_epoch == 2.0
    assert cb.state.evals_since_best == 2


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    @override
    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


def test_logging_eval_line_carries_best_and_patience() -> None:
    logger = logging.getLogger("sklm-test-eval-line")
    logger.setLevel(logging.INFO)
    capture = _CaptureHandler()
    logger.addHandler(capture)
    cb = LoggingCallback(logger=logger)
    cb.on_fit_info("model", TrainingConfig(evaluation=EvalConfig(split=0.2, patience=3)))
    cb.on_fit_start(n_rows=10, n_epochs=3)
    for epoch, loss in enumerate([0.9, 0.5, 0.6], start=1):
        cb.on_eval_report(step=epoch, loss=loss, epoch=float(epoch))
    line = capture.lines[-1]
    assert "loss 0.6000" in line
    assert "best 0.5000 @ep 2" in line
    assert "patience 1/3" in line


def test_logging_autoconfig_disables_propagation() -> None:
    auto = logging.getLogger("sklm-test-propagate-auto")
    LoggingCallback(logger=auto)
    assert auto.propagate is False  # a root basicConfig would double every line

    owned = logging.getLogger("sklm-test-propagate-owned")
    owned.addHandler(logging.StreamHandler())
    LoggingCallback(logger=owned)
    assert owned.propagate is True  # the caller's setup is left untouched
