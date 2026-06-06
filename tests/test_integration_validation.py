# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Validation hold-out on the real backends: eval-loss reporting, early stopping
and checkpoint persistence. Each test fine-tunes a real model and self-skips
without its backend dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import override

import numpy as np
import pandas as pd
import pytest

from sklm import (
    Callback,
    EvalReport,
    Event,
    LanguageModelClassifier,
    TrainingConfig,
    TrainingState,
    TrainReport,
)

from .conftest import _has_hf, _has_mlx

pytestmark = pytest.mark.slow

_MLX_MODEL = "gabfssilva/distilgpt2"


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
        training=TrainingConfig(epochs=2, batch_size=8, validation_split=0.25),
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
            learning_rate=2e-3,  # overfit fast so validation loss rises and early stop fires
            validation_split=0.25,
            stratify=False,
            early_stopping_patience=2,
        ),
        callback=rec,
        random_state=0,
    )
    clf.fit(X, y)
    # a high LR memorizes the tiny train set within a few epochs; validation loss
    # then stops improving, so far fewer than ``epochs`` evaluations run.
    assert 0 < len(rec.eval_reports) < epochs
    assert set(clf.predict(X.head(4))).issubset(set(clf.classes_))


def _checkpoint_dir_persists(model: str, backend: str, tmp_path: Path) -> None:
    X, y = _clf_xy()
    ckpt = tmp_path / "ckpts"
    clf = LanguageModelClassifier(
        model=model,
        backend=backend,
        training=TrainingConfig(
            epochs=2, batch_size=8, checkpoint_steps=2, checkpoint_dir=str(ckpt)
        ),
        random_state=0,
    )
    clf.fit(X, y)
    assert ckpt.is_dir()
    assert any(ckpt.iterdir()), "expected checkpoint artifacts to be persisted"


# --- HuggingFace ----------------------------------------------------------


@pytest.mark.skipif(not _has_hf(), reason="requires the 'hf' extra")
def test_hf_eval_reports_fire() -> None:
    _eval_reports_fire("distilgpt2", "huggingface")


@pytest.mark.skipif(not _has_hf(), reason="requires the 'hf' extra")
def test_hf_early_stopping_stops_short() -> None:
    _early_stopping_stops_short("distilgpt2", "huggingface")


@pytest.mark.skipif(not _has_hf(), reason="requires the 'hf' extra")
def test_hf_checkpoint_dir_persists(tmp_path: Path) -> None:
    _checkpoint_dir_persists("distilgpt2", "huggingface", tmp_path)


# --- MLX ------------------------------------------------------------------


@pytest.mark.skipif(not _has_mlx(), reason="requires the 'mlx' extra on Apple Silicon")
def test_mlx_eval_reports_fire() -> None:
    _eval_reports_fire(_MLX_MODEL, "mlx")


@pytest.mark.skipif(not _has_mlx(), reason="requires the 'mlx' extra on Apple Silicon")
def test_mlx_early_stopping_stops_short() -> None:
    _early_stopping_stops_short(_MLX_MODEL, "mlx")


@pytest.mark.skipif(not _has_mlx(), reason="requires the 'mlx' extra on Apple Silicon")
def test_mlx_checkpoint_dir_persists(tmp_path: Path) -> None:
    _checkpoint_dir_persists(_MLX_MODEL, "mlx", tmp_path)
