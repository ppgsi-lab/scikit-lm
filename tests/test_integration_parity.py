# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Cross-backend parity: given the same data ``X`` and the same params ``P``,
fine-tuning on each installed backend must reach a comparable training loss and a
comparable reconstruction accuracy.

This is the net for bugs the per-backend unit tests miss. It runs the
``loss_on_target_only`` path on purpose: that mask is plumbed differently on each
backend (HF emits ``labels`` with the prompt set to -100; MLX masks by token
offset), and a backend that drops it silently supervises the whole row instead of
the target -- which shows up here as a backend whose masked training loss never
collapses on a trivially reconstructible target. Speed is out of scope (MLX is
typically faster); only quality has to match.

The test self-skips unless at least two of the requested backends are importable,
so in practice it runs on a dev box rather than CI. Adding a backend to compare is
a one-line entry in the parametrized ``backends`` set.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import override

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from sklm import (
    Callback,
    Event,
    LanguageModelClassifier,
    TrainingConfig,
    TrainingState,
    TrainReport,
)

from .conftest import _has_hf, _has_mlx

pytestmark = pytest.mark.slow

# The target is trivially reconstructible and the loss is masked to it alone, so a
# backend whose mask works drives the training loss to ~0. The ceiling catches a
# backend that fails to mask (its loss stalls high); the bands catch one backend
# landing far from the rest.
_MASKED_LOSS_CEIL = 0.20  # each backend's final masked train loss must clear this
_LOSS_BAND = 0.20  # max spread across the backends' final losses
_FLOOR_MARGIN = 0.15  # each backend must beat the majority floor by this much
_ACC_BAND = 0.15  # max spread across the backends' accuracies


@dataclass(frozen=True)
class _Backend:
    """One real backend to fine-tune in the parity comparison."""

    name: str
    model: str
    backend: str
    available: Callable[[], bool]


# distilgpt2 and its mlx-loadable mirror share weights, so the only difference is
# the runtime. Append an entry to a ``backends`` set below to compare more.
_HF = _Backend("hf", "distilgpt2", "huggingface", _has_hf)
_MLX = _Backend("mlx", "mlx-community/distilgpt2", "mlx", _has_mlx)


def _xy(n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """A strongly reconstructible binary task: a well-separated numeric column and
    a mostly-informative categorical one, so a fitted model should land near the
    ceiling and any backend that fails to learn (or to mask) stands out."""
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, size=n)
    a = rng.normal(loc=label * 4.0, scale=0.6, size=n)
    b = np.where(label == 1, "hi", "lo")
    flip = rng.random(n) < 0.1
    b = np.where(flip, np.where(b == "hi", "lo", "hi"), b)
    X = pd.DataFrame({"a": a, "b": b})
    y = np.where(label == 1, "pos", "neg")
    return X, y


def _params() -> TrainingConfig:
    """The shared ``P``. ``loss_on_target_only`` focuses the loss on the target,
    which every backend must honor; ``label_smoothing`` is pinned to 0 so the
    reported losses are plain per-token cross-entropy and thus comparable."""
    return TrainingConfig(epochs=12, batch_size=8, label_smoothing=0.0, loss_on_target_only=True)


class _LossRecorder(Callback):
    """Collect the per-step training losses the backend reports."""

    def __init__(self) -> None:
        super().__init__()
        self.losses: list[float] = []

    @override
    def on_event(self, state: TrainingState, event: Event) -> None:
        match event:
            case TrainReport(loss=loss):
                self.losses.append(loss)
            case _:
                pass

    def final_loss(self) -> float:
        assert self.losses, "no TrainReport fired during fine-tuning"
        return float(np.mean(self.losses[-3:]))


def _train_and_score(
    spec: _Backend,
    train: tuple[pd.DataFrame, np.ndarray],
    test: tuple[pd.DataFrame, np.ndarray],
) -> tuple[float, float]:
    """Fine-tune one backend on ``train`` and return ``(final_loss, test_accuracy)``."""
    rec = _LossRecorder()
    clf = LanguageModelClassifier(
        model=spec.model,
        backend=spec.backend,
        training=_params(),
        callback=rec,
        random_state=0,
    )
    clf.fit(*train)
    acc = float(accuracy_score(test[1], clf.predict(test[0])))
    return rec.final_loss(), acc


@pytest.mark.parametrize("backends", [pytest.param((_HF, _MLX), id="hf-mlx")])
def test_backend_loss_and_accuracy_parity(backends: tuple[_Backend, ...]) -> None:
    available = [b for b in backends if b.available()]
    if len(available) < 2:
        pytest.skip("parity needs at least two of the requested backends installed")

    train = _xy(48, seed=0)
    test = _xy(32, seed=1)
    floor = float(pd.Series(test[1]).value_counts(normalize=True).max())

    results = {b.name: _train_and_score(b, train, test) for b in available}
    parts = [f"{name}: loss={loss:.3f} acc={acc:.3f}" for name, (loss, acc) in results.items()]
    summary = " | ".join(parts) + f" | floor={floor:.3f}"
    print(summary)

    losses = [loss for loss, _ in results.values()]
    accs = [acc for _, acc in results.values()]
    # 1) the target-only mask works on each backend: a reconstructible target masked
    #    to itself drives the training loss to ~0, and each backend learns the signal
    #    (a backend that drops the mask supervises the whole row and stalls high)
    for loss, acc in results.values():
        assert loss <= _MASKED_LOSS_CEIL, summary
        assert acc >= floor + _FLOOR_MARGIN, summary
    # 2) the backends agree with each other on loss and accuracy
    assert max(losses) - min(losses) <= _LOSS_BAND, summary
    assert max(accs) - min(accs) <= _ACC_BAND, summary
