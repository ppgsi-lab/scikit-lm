from __future__ import annotations

import statistics
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import TrainingConfig


# --------------------------------------------------------------------------- helpers


def _format_bytes(n: int) -> str:
    """Render a byte count as a human-readable MiB/GiB string."""
    mib = n / 1024**2
    return f"{mib / 1024:.2f} GiB" if mib >= 1024 else f"{mib:.1f} MiB"


def _format_rate(steps_per_second: float) -> str:
    """Render step throughput tqdm-style: ``step/s`` when >= 1, else ``s/step``."""
    if steps_per_second >= 1:
        return f"{steps_per_second:.2f} step/s"
    return f"{1 / steps_per_second:.2f} s/step"


def _mean_std(values: Sequence[float]) -> str:
    """Render ``mean±std`` (population std) over ``values``."""
    return f"{statistics.fmean(values):.4f}±{statistics.pstdev(values):.4f}"


def _format_duration(seconds: float) -> str:
    """Render a duration as whole seconds (no minute/hour rollover)."""
    return f"{round(seconds)} sec"


def _format_params(training: TrainingConfig) -> str:
    """Render the headline training hyperparameters as a ``·``-joined string."""
    lr = training.learning_rate
    parts = [
        f"{training.epochs} epochs",
        f"bs={training.batch_size}",
        f"lr={lr}" if isinstance(lr, str) else f"lr={lr:.1e}",
        f"seq={training.max_seq_length or 'auto'}",
    ]
    if training.augmentation_factor > 1:
        parts.append(f"aug={training.augmentation_factor}")
    if training.validation_split > 0:
        parts.append(f"val={training.validation_split:g}")
    if training.loss_on_target_only:
        parts.append("target-only")
    return " · ".join(parts)


def _ranking(candidates: Sequence[object], probs: Sequence[float]) -> str:
    """Render ``candidate=prob`` pairs sorted by probability descending."""
    return ", ".join(
        f"{c}={p:.3f}"
        for c, p in sorted(zip(candidates, probs, strict=True), key=lambda cp: cp[1], reverse=True)
    )


def _ranking_pairs(
    candidates: Sequence[object], probs: Sequence[Sequence[float]]
) -> list[tuple[object, float]]:
    """``(candidate, prob)`` pairs sorted by probability descending.

    Probabilities are averaged over the conditioning orders ``probs`` carries (a
    single order unless the classifier marginalizes over column permutations)."""
    means = [statistics.fmean(col) for col in zip(*probs, strict=True)]
    return sorted(zip(candidates, means, strict=True), key=lambda cp: cp[1], reverse=True)


def _spread_summary(candidates: Sequence[object], probs: Sequence[Sequence[float]]) -> str:
    """Per-candidate ``mean±std [min,max]`` across the per-order distributions.

    Summarizes how much each candidate's probability moved with the conditioning
    column order (the spread :class:`Score` exposes under ``permute_order``),
    sorted by mean descending.
    """
    parts: list[tuple[float, str]] = []
    for c, col in zip(candidates, zip(*probs, strict=True), strict=True):
        vals = list(col)
        mean = statistics.fmean(vals)
        summary = f"{c}={mean:.3f}±{statistics.pstdev(vals):.3f} [{min(vals):.3f},{max(vals):.3f}]"
        parts.append((mean, summary))
    return ", ".join(s for _, s in sorted(parts, key=lambda ms: ms[0], reverse=True))
