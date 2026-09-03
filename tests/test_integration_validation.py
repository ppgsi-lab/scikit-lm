# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Validation hold-out on the real backends: eval-loss reporting, early stopping
and checkpoint persistence. Each test fine-tunes a real model and self-skips
without its backend dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, override

import numpy as np
import pandas as pd
import pytest

from sklm import (
    Callback,
    CheckpointConfig,
    EvalConfig,
    EvalReport,
    Event,
    LanguageModelClassifier,
    LRScheduler,
    TrainingConfig,
    TrainingState,
    TrainReport,
)

from .conftest import _has_hf, _has_mlx

pytestmark = pytest.mark.slow

_MLX_MODEL = "mlx-community/distilgpt2"


class _EvalRecorder(Callback):
    """Capture the validation- and training-loss reports the backend emits."""

    def __init__(self) -> None:
        super().__init__()
        self.eval_reports: list[tuple[int, float]] = []
        self.train_steps: list[int] = []

    @override
    def on_event(self, state: TrainingState, event: Event) -> None:
        match event:
            case EvalReport(step=step, loss=loss):
                self.eval_reports.append((step, loss))
            case TrainReport(step=step):
                self.train_steps.append(step)
            case _:
                pass


class _LRRecorder(Callback):
    """Capture the learning rate carried by each training-loss report."""

    def __init__(self) -> None:
        super().__init__()
        self.lrs: list[float] = []

    @override
    def on_event(self, state: TrainingState, event: Event) -> None:
        if isinstance(event, TrainReport) and event.learning_rate is not None:
            self.lrs.append(event.learning_rate)


def _clf_xy(n: int = 40) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.choice(["x", "y"], size=n)})
    y = rng.choice(["pos", "neg"], size=n)
    return X, y


def _eval_reports_fire(model: str, backend: str) -> None:
    X, y = _clf_xy()
    rec = _EvalRecorder()
    clf = LanguageModelClassifier(
        model=model,
        backend=backend,
        training=TrainingConfig(epochs=2, batch_size=8, evaluation=EvalConfig(split=0.25)),
        callback=rec,
        random_state=0,
    )
    clf.fit(X, y)
    assert rec.eval_reports, "expected at least one on_eval_report"
    assert all(np.isfinite(loss) for _, loss in rec.eval_reports)
    # the model still works after a held-out fit
    assert set(clf.predict(X.head(4))).issubset(set(clf.classes_))


def _early_stopping_stops_short(model: str, backend: str) -> None:
    X, y = _clf_xy()
    rec = _EvalRecorder()
    epochs = 15
    clf = LanguageModelClassifier(
        model=model,
        backend=backend,
        training=TrainingConfig(
            epochs=epochs,
            batch_size=8,
            # overfit fast so validation loss rises and early stop fires
            lr_scheduler=LRScheduler.cosine(learning_rate=2e-3),
            evaluation=EvalConfig(split=0.25, stratify=False, patience=2),
        ),
        callback=rec,
        random_state=0,
    )
    clf.fit(X, y)
    # a high LR memorizes the tiny train set within a few epochs; validation loss
    # then stops improving, so far fewer than ``epochs`` evaluations run.
    assert 0 < len(rec.eval_reports) < epochs
    assert set(clf.predict(X.head(4))).issubset(set(clf.classes_))


def _plateau_reduces_lr(model: str, backend: str) -> None:
    X, y = _clf_xy()
    rec = _LRRecorder()
    clf = LanguageModelClassifier(
        model=model,
        backend=backend,
        training=TrainingConfig(
            epochs=15,
            batch_size=8,
            # a high LR makes validation loss stop improving; plateau then halves it
            lr_scheduler=LRScheduler.plateau(learning_rate=2e-3, patience=1, factor=0.5),
            evaluation=EvalConfig(split=0.25, stratify=False),
        ),
        callback=rec,
        random_state=0,
    )
    clf.fit(X, y)
    steps = rec.lrs
    assert steps, "expected training-loss reports carrying the learning rate"
    pairs = list(zip(steps, steps[1:], strict=False))
    assert all(b <= a + 1e-12 for a, b in pairs), "LR must be non-increasing"
    assert min(steps) < max(steps), "plateau should have reduced the learning rate"
    # Piecewise-constant: the LR holds between evaluations and only drops at a
    # reduction -- unlike a smooth schedule, which would change at every step.
    changes = sum(1 for a, b in pairs if abs(a - b) > 1e-12)
    assert changes * 2 < len(steps), "plateau holds the LR between reductions"
    assert set(clf.predict(X.head(4))).issubset(set(clf.classes_))


def _checkpoint_dir_persists(model: str, backend: str, tmp_path: Path) -> None:
    X, y = _clf_xy()
    ckpt = tmp_path / "ckpts"
    clf = LanguageModelClassifier(
        model=model,
        backend=backend,
        training=TrainingConfig(
            epochs=2,
            batch_size=8,
            checkpoint=CheckpointConfig(each=2, on="step", dir=str(ckpt), keep=None),
        ),
        random_state=0,
    )
    clf.fit(X, y)
    assert ckpt.is_dir()
    snapshots = list(ckpt.glob("checkpoint-*"))
    assert len(snapshots) >= 2, "expected numbered checkpoint snapshots to persist"


def _checkpoint_resumes_from_dir(model: str, backend: str, tmp_path: Path) -> None:
    X, y = _clf_xy()
    ckpt = tmp_path / "ckpts"
    cfg = TrainingConfig(
        epochs=2, batch_size=8, checkpoint=CheckpointConfig(each=2, on="step", dir=str(ckpt))
    )
    LanguageModelClassifier(model=model, backend=backend, training=cfg, random_state=0).fit(X, y)
    before = {p.name for p in ckpt.glob("checkpoint-*")}
    # A second fit pointed at the same dir resumes and keeps writing there.
    LanguageModelClassifier(model=model, backend=backend, training=cfg, random_state=0).fit(X, y)
    assert ckpt.is_dir()
    assert any(ckpt.glob("checkpoint-*")), "expected checkpoints after resume"
    assert before, "first fit should have produced checkpoints"


def _resume_continues_to_budget(model: str, backend: str, tmp_path: Path) -> None:
    """A resume picks up at the saved step and trains only what the budget has
    left, instead of restarting at step 1 -- the MLX backend matching the HF
    Trainer's ``resume_from_checkpoint`` (restored optimizer state, step and LR
    schedule position). A first 2-epoch fit checkpoints per epoch; a second fit
    over the same dir under a larger 4-epoch budget must report only the steps
    above the first run's last and carry on to the extended total.
    """
    X, y = _clf_xy()
    ckpt = tmp_path / "ckpts"

    def fit(epochs: int) -> tuple[LanguageModelClassifier, _EvalRecorder]:
        rec = _EvalRecorder()
        clf = LanguageModelClassifier(
            model=model,
            backend=backend,
            training=TrainingConfig(
                epochs=epochs,
                batch_size=8,
                checkpoint=CheckpointConfig(each=1, on="epoch", dir=str(ckpt), keep=None),
            ),
            callback=rec,
            random_state=0,
        )
        clf.fit(X, y)
        return clf, rec

    _, first = fit(2)
    clf, second = fit(4)

    assert first.train_steps, "first fit should report training steps"
    assert second.train_steps, "resume should still have steps left to train"
    # The resume neither restarts at step 1 nor redoes the first run's steps,
    assert min(second.train_steps) > max(first.train_steps)
    # and it carries on to the extended budget's final step (twice the epochs).
    assert max(second.train_steps) == 2 * max(first.train_steps)
    assert set(clf.predict(X.head(4))).issubset(set(clf.classes_))


def _eval_on_step_cadence(model: str, backend: str) -> None:
    """``EvalConfig(each=2, on="step")`` evaluates every second optimizer step on
    both backends, not once per epoch: 30 training rows x 2 variants at batch 8
    is 8 steps per epoch, so 2 epochs report evaluations at every even step."""
    X, y = _clf_xy()
    rec = _EvalRecorder()
    clf = LanguageModelClassifier(
        model=model,
        backend=backend,
        training=TrainingConfig(
            epochs=2,
            batch_size=8,
            evaluation=EvalConfig(split=0.25, stratify=False, each=2, on="step", patience=None),
        ),
        callback=rec,
        random_state=0,
    )
    clf.fit(X, y)
    steps = [step for step, _ in rec.eval_reports]
    assert steps == list(range(2, max(rec.train_steps) + 1, 2))
    assert set(clf.predict(X.head(4))).issubset(set(clf.classes_))


# --- HuggingFace ----------------------------------------------------------


@pytest.mark.skipif(not _has_hf(), reason="requires the 'hf' extra")
def test_hf_eval_reports_fire() -> None:
    _eval_reports_fire("distilgpt2", "huggingface")


@pytest.mark.skipif(not _has_hf(), reason="requires the 'hf' extra")
def test_hf_early_stopping_stops_short() -> None:
    _early_stopping_stops_short("distilgpt2", "huggingface")


@pytest.mark.skipif(not _has_hf(), reason="requires the 'hf' extra")
def test_hf_plateau_reduces_lr() -> None:
    _plateau_reduces_lr("distilgpt2", "huggingface")


@pytest.mark.skipif(not _has_hf(), reason="requires the 'hf' extra")
def test_hf_checkpoint_dir_persists(tmp_path: Path) -> None:
    _checkpoint_dir_persists("distilgpt2", "huggingface", tmp_path)


@pytest.mark.skipif(not _has_hf(), reason="requires the 'hf' extra")
def test_hf_checkpoint_resumes_from_dir(tmp_path: Path) -> None:
    _checkpoint_resumes_from_dir("distilgpt2", "huggingface", tmp_path)


@pytest.mark.skipif(not _has_hf(), reason="requires the 'hf' extra")
def test_hf_resume_continues_to_budget(tmp_path: Path) -> None:
    _resume_continues_to_budget("distilgpt2", "huggingface", tmp_path)


@pytest.mark.skipif(not _has_hf(), reason="requires the 'hf' extra")
def test_hf_eval_on_step_cadence() -> None:
    _eval_on_step_cadence("distilgpt2", "huggingface")


# --- MLX ------------------------------------------------------------------


@pytest.mark.skipif(not _has_mlx(), reason="requires the 'mlx' extra on Apple Silicon")
def test_mlx_eval_reports_fire() -> None:
    _eval_reports_fire(_MLX_MODEL, "mlx")


@pytest.mark.skipif(not _has_mlx(), reason="requires the 'mlx' extra on Apple Silicon")
def test_mlx_early_stopping_stops_short() -> None:
    _early_stopping_stops_short(_MLX_MODEL, "mlx")


@pytest.mark.skipif(not _has_mlx(), reason="requires the 'mlx' extra on Apple Silicon")
def test_mlx_plateau_reduces_lr() -> None:
    _plateau_reduces_lr(_MLX_MODEL, "mlx")


@pytest.mark.skipif(not _has_mlx(), reason="requires the 'mlx' extra on Apple Silicon")
def test_mlx_checkpoint_dir_persists(tmp_path: Path) -> None:
    _checkpoint_dir_persists(_MLX_MODEL, "mlx", tmp_path)


@pytest.mark.skipif(not _has_mlx(), reason="requires the 'mlx' extra on Apple Silicon")
def test_mlx_checkpoint_resumes_from_dir(tmp_path: Path) -> None:
    _checkpoint_resumes_from_dir(_MLX_MODEL, "mlx", tmp_path)


@pytest.mark.skipif(not _has_mlx(), reason="requires the 'mlx' extra on Apple Silicon")
def test_mlx_resume_continues_to_budget(tmp_path: Path) -> None:
    _resume_continues_to_budget(_MLX_MODEL, "mlx", tmp_path)


@pytest.mark.skipif(not _has_mlx(), reason="requires the 'mlx' extra on Apple Silicon")
def test_mlx_eval_on_step_cadence() -> None:
    _eval_on_step_cadence(_MLX_MODEL, "mlx")


@pytest.mark.skipif(not _has_mlx(), reason="requires the 'mlx' extra on Apple Silicon")
def test_mlx_checkpoint_round_trips_optimizer_state(tmp_path: Path) -> None:
    """White-box: a ``state-<step>`` sidecar round-trips the optimizer state, RNG
    and plateau/early-stop/best trackers, so a resumed optimizer reproduces the
    saved step, Adam moments and LR-schedule position (no model download)."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as opt
    from mlx.utils import tree_flatten, tree_unflatten

    from sklm.mlx_backend import _resume_state, _save_best, _save_checkpoint

    def build() -> tuple[Any, Any]:
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 4))
        optimizer = opt.AdamW(learning_rate=opt.cosine_decay(1e-2, 20), weight_decay=0.01)
        return model, optimizer

    def loss(model: Any, xb: Any, yb: Any) -> Any:
        return nn.losses.mse_loss(model(xb), yb)

    mx.random.seed(0)
    model, optimizer = build()
    x, target = mx.random.normal((4, 4)), mx.random.normal((4, 4))
    for _ in range(7):
        _, grad = nn.value_and_grad(model, loss)(model, x, target)
        optimizer.update(model, grad)
        mx.eval(model.parameters(), optimizer.state)

    weights = tree_flatten(model.trainable_parameters())
    opt_state = tree_flatten(optimizer.state)
    rng_saved: list[Any] = list(mx.random.state)  # type: ignore[arg-type]  # no stub for RNG global
    sidecar: dict[str, Any] = {f"opt.{k}": v for k, v in opt_state}
    sidecar |= {f"rng.{i}": a for i, a in enumerate(rng_saved)}
    sidecar |= {
        "meta.no_improve": mx.array(2),
        "meta.plateau_best": mx.array(0.5),
        "meta.plateau_num_bad": mx.array(1),
        "meta.plateau_cooldown": mx.array(0),
    }
    mx.eval([v for _, v in weights], [v for _, v in opt_state])
    _save_checkpoint(str(tmp_path), 7, weights, sidecar, keep=None)
    _save_best(str(tmp_path), weights, 0.5)

    fresh, restored = build()
    resume = _resume_state(fresh, str(tmp_path))
    assert resume is not None
    assert (resume.resumed_it, resume.no_improve, resume.best_loss) == (7, 2, 0.5)
    assert resume.plateau == (0.5, 1, 0)
    assert resume.opt_state is not None

    restored.init(fresh.trainable_parameters())
    restored.state = tree_unflatten(resume.opt_state)
    restored._initialized = True
    mx.eval(restored.state)
    saved_leaves = dict(tree_flatten(optimizer.state))
    restored_leaves = dict(tree_flatten(restored.state))
    assert int(restored_leaves["step"].item()) == int(saved_leaves["step"].item()) == 7
    moment = next(k for k in saved_leaves if k.endswith(".m"))
    assert np.allclose(np.array(saved_leaves[moment]), np.array(restored_leaves[moment]))

    # One further identical update advances both schedules off the resumed step
    # in lockstep -- proving the step counter (not just the moments) was restored.
    for mdl, optr in ((model, optimizer), (fresh, restored)):
        _, grad = nn.value_and_grad(mdl, loss)(mdl, x, target)
        optr.update(mdl, grad)
        mx.eval(mdl.parameters(), optr.state)
    assert int(restored.state["step"].item()) == 8
    assert abs(optimizer.learning_rate.item() - restored.learning_rate.item()) < 1e-9
