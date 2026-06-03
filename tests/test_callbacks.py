"""Public ``Callback`` API: state aggregation, lifecycle order, per-value events,
and that progress is reported *during* inference rather than bunched at the end."""

from __future__ import annotations

import logging
from typing import override

import pandas as pd
import pytest
from sklearn.base import clone

from sklm import (
    Callback,
    Event,
    FitEnd,
    FitStart,
    Generation,
    GenerationConfig,
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
    TrainingState,
)
from sklm.callbacks import _spread_summary
from sklm.serialize import TrainingExample

from .conftest import FakeBackend


class RecordingCallback(Callback):
    """Record the ``(state, event)`` stream for white-box assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, tuple[object, ...]]] = []
        self.retries: list[tuple[str, int, int]] = []
        self.generations: list[tuple[str, str, str]] = []
        self.scorings: list[
            tuple[list[str], list[object], list[list[float]], list[list[float]]]
        ] = []
        self.train_examples: list[TrainingExample] = []

    @override
    def on_event(self, state: TrainingState, event: Event) -> None:
        match event:
            case FitStart(n_rows=n_rows, n_epochs=n_epochs):
                self.events.append(("fit_start", (n_rows, n_epochs)))
            case TrainExamples(examples=examples):
                self.events.append(("train_examples", (len(examples),)))
                self.train_examples = list(examples)
            case FitEnd():
                self.events.append(("fit_end", ()))
            case PredictStart(n=n):
                self.events.append(("predict_start", (n,)))
            case RowEnd(index=index, total=total):
                self.events.append(("row_end", (index, total)))
            case PredictEnd():
                self.events.append(("predict_end", ()))
            case Generation(prompt=prompt, completion=completion, target=target):
                self.events.append(("generation", (prompt, completion, target)))
                self.generations.append((prompt, completion, target))
            case Score(prompts=prompts, candidates=candidates, probs=probs, logprobs=logprobs):
                self.events.append(("score", (len(prompts), len(candidates))))
                self.scorings.append(
                    (
                        list(prompts),
                        list(candidates),
                        [list(p) for p in probs],
                        [list(lp) for lp in logprobs],
                    )
                )
            case Retry(target=target, attempt=attempt, max_attempts=max_attempts):
                self.events.append(("retry", (target, attempt, max_attempts)))
                self.retries.append((target, attempt, max_attempts))
            case _:
                pass

    def names(self) -> list[str]:
        return [name for name, _ in self.events]


def test_lifecycle_event_order(nan_data: pd.DataFrame) -> None:
    rec = RecordingCallback()
    imp = LanguageModelImputer(backend=FakeBackend(value="0"), callbacks=rec)
    imp.fit(nan_data)
    # the per-epoch examples sample is always surfaced now (the listener slices it)
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


def test_state_tracks_peak_memory() -> None:
    rec = RecordingCallback()
    rec.on_fit_start(n_rows=4, n_epochs=1)
    rec.on_memory(300)
    rec.on_memory(100)  # current drops, peak holds
    assert rec.state.mem == 100
    assert rec.state.peak_mem == 300
    rec.on_fit_start(n_rows=4, n_epochs=1)  # peak resets per fit
    assert rec.state.peak_mem == 0


def test_progress_is_reported_during_inference(nan_data: pd.DataFrame) -> None:
    rec = RecordingCallback()
    imp = LanguageModelImputer(
        backend=FakeBackend(value="0"),
        callbacks=rec,
        generation=GenerationConfig(inference_batch_size=1),
    ).fit(nan_data)
    rec.events.clear()
    imp.transform(nan_data)
    names = rec.names()
    first_row = names.index("row_end")
    # some generation happens after a row was already reported done -> interleaved
    assert "generation" in names[first_row + 1 :]


def test_retry_then_raise_on_malformed_generation(nan_data: pd.DataFrame) -> None:
    rec = RecordingCallback()
    imp = LanguageModelImputer(backend=FakeBackend(value="nope"), callbacks=rec)
    imp.fit(nan_data)
    with pytest.raises(RuntimeError, match="not producing valid values"):
        imp.transform(nan_data)
    assert "retry" in rec.names()
    assert all(1 <= attempt <= mx for _, attempt, mx in rec.retries)


def test_generation_event_exposes_prompt_and_output(nan_data: pd.DataFrame) -> None:
    rec = RecordingCallback()
    imp = LanguageModelImputer(backend=FakeBackend(value="7"), callbacks=rec)
    imp.fit(nan_data)
    rec.events.clear()
    imp.transform(nan_data)
    assert len(rec.generations) == 3  # nan_data has 3 missing cells; "7" parses, no retries
    assert "retry" not in rec.names()
    prompts, completions, targets = zip(*rec.generations, strict=True)
    assert all(c == "7" for c in completions)
    assert all(t in ("age", "city") for t in targets)
    assert all(f'"{t}": ' in p for p, t in zip(prompts, targets, strict=True))


def test_score_event_exposes_prompt_and_distribution(clf_data) -> None:
    X, y = clf_data
    rec = RecordingCallback()
    clf = LanguageModelClassifier(backend=FakeBackend(), callbacks=rec)
    clf.fit(X, y)
    rec.events.clear()
    clf.predict(X)
    assert "score" in rec.names()
    assert "generation" not in rec.names()  # the classifier ranks, never generates
    assert len(rec.scorings) == len(X)  # one call per row
    classes = list(clf.classes_)
    for prompts, candidates, probs, logprobs in rec.scorings:
        assert list(candidates) == classes
        assert len(prompts) == len(probs) == len(logprobs) == 1  # one order without permute
        assert len(probs[0]) == len(classes) == len(logprobs[0])
        assert abs(sum(probs[0]) - 1.0) < 1e-9
        assert f'"{clf.target_col_}": ' in prompts[0]


def test_score_event_carries_every_order_under_permutation(clf_data) -> None:
    X, y = clf_data
    rec = RecordingCallback()
    clf = LanguageModelClassifier(
        backend=FakeBackend(),
        callbacks=rec,
        generation=GenerationConfig(permute_order=True, n_samples=4),
    )
    clf.fit(X, y)
    rec.events.clear()
    clf.predict(X.head(1))
    assert len(rec.scorings) == 1  # a single on_score for the row
    prompts, candidates, probs, logprobs = rec.scorings[0]
    # one row, 3 features -> min(4, 3!) = 4 distinct column orders, all inside the one call
    assert len(prompts) == len(probs) == len(logprobs) == 4
    assert len(set(prompts)) == 4
    for order_probs in probs:
        assert abs(sum(order_probs) - 1.0) < 1e-9


def test_spread_summary_reports_mean_std_and_range() -> None:
    # two column orders, two candidates
    summary = _spread_summary(["a", "b"], [[0.8, 0.2], [0.6, 0.4]])
    assert "a=0.700±0.100 [0.600,0.800]" in summary
    assert "b=0.300±0.100 [0.200,0.400]" in summary
    assert summary.index("a=") < summary.index("b=")  # sorted by mean descending


def test_train_examples_always_emitted_and_capped(nan_data: pd.DataFrame) -> None:
    rec = RecordingCallback()
    LanguageModelImputer(backend=FakeBackend(value="0"), callbacks=rec).fit(nan_data)
    assert "train_examples" in rec.names()
    # the wire always surfaces a bounded sample; listeners slice it for display
    assert 0 < len(rec.train_examples) <= 16
    assert all(isinstance(ex, TrainingExample) and ex.text for ex in rec.train_examples)


def test_default_callbacks_is_noop(nan_data: pd.DataFrame) -> None:
    out = LanguageModelImputer(backend=FakeBackend(value="0")).fit_transform(nan_data)
    assert not out.isna().any().any()


def test_clone_preserves_callbacks() -> None:
    imp = LanguageModelImputer(backend=FakeBackend(), callbacks=LoggingCallback())
    cloned = clone(imp)
    assert isinstance(cloned, LanguageModelImputer)
    assert isinstance(cloned.callbacks, LoggingCallback)


def test_logging_callback_logs_generation(nan_data: pd.DataFrame, caplog) -> None:
    imp = LanguageModelImputer(
        backend=FakeBackend(value="7"), callbacks=LoggingCallback(n_predict_examples=10)
    )
    imp.fit(nan_data)
    with caplog.at_level(logging.INFO, logger="sklm"):
        imp.transform(nan_data)
    messages = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "prompt:" in messages
    assert "generated: 7" in messages


def test_logging_callback_train_examples_preview(nan_data: pd.DataFrame, caplog) -> None:
    imp = LanguageModelImputer(
        backend=FakeBackend(value="0"), callbacks=LoggingCallback(n_train_examples=2)
    )
    with caplog.at_level(logging.INFO, logger="sklm"):
        imp.fit(nan_data)
    previews = [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("train example")
    ]
    assert len(previews) == 2  # listener slices the surfaced sample down to its display count


def test_logging_callback_predict_examples_default_logs_none(
    nan_data: pd.DataFrame, caplog
) -> None:
    imp = LanguageModelImputer(backend=FakeBackend(value="7"), callbacks=LoggingCallback())
    imp.fit(nan_data)
    with caplog.at_level(logging.INFO, logger="sklm"):
        imp.transform(nan_data)
    assert not [r for r in caplog.records if "generated:" in r.getMessage()]


def test_logging_callback_caps_generations_and_resets_each_predict(
    nan_data: pd.DataFrame, caplog
) -> None:
    # nan_data has 3 missing cells -> 3 generations per transform; cap to 2
    cb = LoggingCallback(n_predict_examples=2)
    imp = LanguageModelImputer(backend=FakeBackend(value="7"), callbacks=cb).fit(nan_data)
    with caplog.at_level(logging.INFO, logger="sklm"):
        imp.transform(nan_data)
        imp.transform(nan_data)
    gens = [r.getMessage() for r in caplog.records if "generated:" in r.getMessage()]
    assert len(gens) == 4  # 2 per predict, counter resets at on_predict_start


def test_logging_callback_caps_scored_rows(clf_data, caplog) -> None:
    X, y = clf_data
    clf = LanguageModelClassifier(
        backend=FakeBackend(), callbacks=LoggingCallback(n_predict_examples=3)
    ).fit(X, y)
    with caplog.at_level(logging.INFO, logger="sklm"):
        clf.predict(X)  # clf_data has >3 rows
    scored = [r.getMessage() for r in caplog.records if "scored:" in r.getMessage()]
    assert len(scored) == 3  # one score line per row, capped at 3


def test_logging_callback_levels(caplog) -> None:
    cb = LoggingCallback()
    with caplog.at_level(logging.DEBUG, logger="sklm"):
        cb.on_train_report(step=10, total_steps=100, loss=1.5, epoch=1.0, learning_rate=5e-5)
        cb.on_retry("age", 1, 3)
    levels = {rec.levelno for rec in caplog.records}
    assert logging.INFO in levels  # training report
    assert logging.DEBUG in levels  # retry
    retry = next(rec for rec in caplog.records if rec.levelno == logging.DEBUG)
    assert "age" in retry.getMessage()


def test_logging_callback_throttles_train_reports(caplog) -> None:
    cb = LoggingCallback(log_every="step", log_each=5)
    with caplog.at_level(logging.INFO, logger="sklm"):
        cb.on_fit_start(n_rows=10, n_epochs=1)
        for step in range(1, 11):
            cb.on_train_report(step=step, total_steps=10, loss=1.0, epoch=None, learning_rate=None)
    steps = [r.getMessage() for r in caplog.records if "step " in r.getMessage()]
    assert len(steps) == 2  # interval=5 -> steps 5 and 10
    assert "step 5/10" in steps[0]
    assert "step 10/10" in steps[1]


def test_logging_callback_logs_every_step_with_step_cadence(caplog) -> None:
    cb = LoggingCallback(log_every="step", log_each=1)
    with caplog.at_level(logging.INFO, logger="sklm"):
        cb.on_fit_start(n_rows=3, n_epochs=1)
        for step in range(1, 4):
            cb.on_train_report(step=step, total_steps=3, loss=1.0, epoch=None, learning_rate=None)
    steps = [r.getMessage() for r in caplog.records if "step " in r.getMessage()]
    assert len(steps) == 3


def test_logging_callback_step_line_shows_epoch_loss(caplog) -> None:
    cb = LoggingCallback(log_every="step")
    reports = [(0.4, 0.5), (0.6, 1.0), (0.2, 1.5), (0.3, 2.0)]  # two losses per epoch
    with caplog.at_level(logging.INFO, logger="sklm"):
        cb.on_fit_start(n_rows=10, n_epochs=2)
        for step, (loss, epoch) in enumerate(reports, start=1):
            cb.on_train_report(step=step, total_steps=4, loss=loss, epoch=epoch, learning_rate=None)
    steps = [r.getMessage() for r in caplog.records if "step " in r.getMessage()]
    assert "epoch loss 0.4000±0.0000" in steps[0]  # logged from the first step in the epoch
    assert "last20 0.4000±0.0000" in steps[0]  # rolling window logged from the start too
    assert "loss 0.6000" in steps[1]  # the raw point of the report
    assert "epoch loss 0.5000±0.1000" in steps[1]  # epoch 1: mean±std over (0.4, 0.6)
    assert "epoch loss 0.2000±0.0000" in steps[2]  # window reset at the epoch-2 boundary
    assert "epoch loss 0.2500±0.0500" in steps[3]  # epoch 2: mean±std over (0.2, 0.3)


def test_logging_callback_step_fraction_cadence(caplog) -> None:
    cb = LoggingCallback(log_every="step", log_each=0.25)  # every 25% of total_steps
    with caplog.at_level(logging.INFO, logger="sklm"):
        cb.on_fit_start(n_rows=10, n_epochs=1)
        for step in range(1, 21):
            cb.on_train_report(step=step, total_steps=20, loss=1.0, epoch=None, learning_rate=None)
    steps = [r.getMessage() for r in caplog.records if "step " in r.getMessage()]
    assert len(steps) == 4  # interval = 0.25*20 = 5 -> steps 5,10,15,20
    assert "step 5/20" in steps[0]
    assert "step 20/20" in steps[3]


def test_logging_callback_step_fraction_without_total_logs_every_report(caplog) -> None:
    cb = LoggingCallback(log_every="step", log_each=0.5)
    with caplog.at_level(logging.INFO, logger="sklm"):
        cb.on_fit_start(n_rows=10, n_epochs=1)
        for step in range(1, 4):  # total_steps=None -> fraction unresolvable
            cb.on_train_report(step, None, loss=1.0, epoch=None, learning_rate=None)
    steps = [r.getMessage() for r in caplog.records if "step " in r.getMessage()]
    assert len(steps) == 3  # falls back to logging every report


def test_logging_callback_epoch_cadence(caplog) -> None:
    cb = LoggingCallback(log_every="epoch", log_each=1)
    reports = [(1, 0.5), (2, 1.0), (3, 1.5), (4, 2.0), (5, 2.5), (6, 3.0)]
    with caplog.at_level(logging.INFO, logger="sklm"):
        cb.on_fit_start(n_rows=10, n_epochs=3)
        for step, epoch in reports:
            cb.on_train_report(step=step, total_steps=6, loss=1.0, epoch=epoch, learning_rate=None)
    steps = [r.getMessage() for r in caplog.records if "step " in r.getMessage()]
    assert len(steps) == 3  # one per epoch boundary: epoch 1.0, 2.0, 3.0
    assert "step 2/6" in steps[0]
    assert "step 4/6" in steps[1]
    assert "step 6/6" in steps[2]


def test_logging_callback_epoch_fraction_cadence(caplog) -> None:
    cb = LoggingCallback(log_every="epoch", log_each=0.5)  # every 50% of the epochs
    reports = [(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0)]
    with caplog.at_level(logging.INFO, logger="sklm"):
        cb.on_fit_start(n_rows=10, n_epochs=4)  # interval = 0.5*4 = 2 epochs
        for step, epoch in reports:
            cb.on_train_report(step=step, total_steps=4, loss=1.0, epoch=epoch, learning_rate=None)
    steps = [r.getMessage() for r in caplog.records if "step " in r.getMessage()]
    assert len(steps) == 2  # epoch 2.0 and 4.0


def test_logging_callback_epoch_cadence_derives_epoch_from_steps(caplog) -> None:
    # MLX reports epoch=None; epoch is derived from step / total_steps * n_epochs
    cb = LoggingCallback(log_every="epoch", log_each=1)
    with caplog.at_level(logging.INFO, logger="sklm"):
        cb.on_fit_start(n_rows=10, n_epochs=2)  # 10 steps, 2 epochs -> boundaries at step 5, 10
        for step in range(1, 11):
            cb.on_train_report(step=step, total_steps=10, loss=1.0, epoch=None, learning_rate=None)
    steps = [r.getMessage() for r in caplog.records if "step " in r.getMessage()]
    assert len(steps) == 2
    assert "step 5/10" in steps[0]
    assert "step 10/10" in steps[1]


def test_logging_callback_rejects_bad_cadence() -> None:
    with pytest.raises(ValueError, match="count must be >= 1"):
        LoggingCallback(log_each=0)
    with pytest.raises(ValueError, match=r"fraction must be in \(0, 1\]"):
        LoggingCallback(log_each=1.5)


def _report(cb: LoggingCallback, step: int, *, total: int = 3) -> None:
    cb.on_train_report(step=step, total_steps=total, loss=1.0, epoch=None, learning_rate=None)


def test_logging_callback_memory_rides_on_step_report(caplog) -> None:
    mib = 1024**2
    cb = LoggingCallback(log_every="step")
    with caplog.at_level(logging.INFO, logger="sklm"):
        cb.on_fit_start(n_rows=10, n_epochs=1)
        cb.on_memory(100 * mib)
        _report(cb, 1)
        cb.on_memory(300 * mib)  # peak rises
        _report(cb, 2)
        cb.on_memory(150 * mib)  # current drops, peak holds at 300
        _report(cb, 3)
    assert not [r for r in caplog.records if r.getMessage().startswith("memory:")]  # no lone line
    steps = [r.getMessage() for r in caplog.records if "step " in r.getMessage()]
    assert len(steps) == 3
    assert "mem 100.0 MiB↑100.0 MiB" in steps[0]
    assert "mem 300.0 MiB↑300.0 MiB" in steps[1]
    assert "mem 150.0 MiB↑300.0 MiB" in steps[2]  # current, with peak held


def test_logging_callback_memory_omitted_without_accelerator(caplog) -> None:
    cb = LoggingCallback(log_every="step")
    with caplog.at_level(logging.INFO, logger="sklm"):
        cb.on_fit_start(n_rows=10, n_epochs=1)
        cb.on_memory(None)  # no accelerator
        _report(cb, 1)
    step = next(r.getMessage() for r in caplog.records if "step " in r.getMessage())
    assert "mem" not in step


def test_logging_callback_memory_peak_resets_per_fit(caplog) -> None:
    mib = 1024**2
    cb = LoggingCallback(log_every="step")
    cb.on_fit_start(n_rows=10, n_epochs=1)
    cb.on_memory(300 * mib)
    with caplog.at_level(logging.INFO, logger="sklm"):
        cb.on_fit_start(n_rows=10, n_epochs=1)  # peak resets per fit
        cb.on_memory(50 * mib)
        _report(cb, 1)
    step = next(r.getMessage() for r in caplog.records if "step " in r.getMessage())
    assert "mem 50.0 MiB↑50.0 MiB" in step


def test_logging_callback_memory_peak_accumulates_across_throttled_steps(caplog) -> None:
    mib = 1024**2
    cb = LoggingCallback(log_every="step", log_each=2)
    with caplog.at_level(logging.INFO, logger="sklm"):
        cb.on_fit_start(n_rows=10, n_epochs=1)
        cb.on_memory(100 * mib)
        _report(cb, 1, total=4)  # skipped, but peak still tracked
        cb.on_memory(300 * mib)
        _report(cb, 2, total=4)  # logged
        cb.on_memory(150 * mib)
        _report(cb, 3, total=4)  # skipped
        cb.on_memory(120 * mib)
        _report(cb, 4, total=4)  # logged
    steps = [r.getMessage() for r in caplog.records if "step " in r.getMessage()]
    assert len(steps) == 2
    assert "step 2/4" in steps[0] and "loss 1.0000" in steps[0]
    assert "mem 300.0 MiB↑300.0 MiB" in steps[0]
    assert "step 4/4" in steps[1] and "loss 1.0000" in steps[1]
    assert "mem 120.0 MiB↑300.0 MiB" in steps[1]


def test_tqdm_callback_lifecycle(nan_data: pd.DataFrame) -> None:
    pytest.importorskip("tqdm")
    imp = LanguageModelImputer(backend=FakeBackend(value="0"), callbacks=TqdmCallback())
    out = imp.fit_transform(nan_data)
    assert not out.isna().any().any()


def test_rich_callback_lifecycle(nan_data: pd.DataFrame) -> None:
    pytest.importorskip("rich")
    pytest.importorskip("plotext")
    imp = LanguageModelImputer(backend=FakeBackend(value="0"), callbacks=RichCallback())
    out = imp.fit_transform(nan_data)
    assert not out.isna().any().any()
