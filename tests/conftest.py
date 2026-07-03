"""Shared fixtures: a torch-free fake backend, seeded data, and a backend matrix.

``FakeBackend`` implements the :class:`sklm.LanguageModelBackend` protocol
deterministically so estimator behaviour can be exercised fast, without
fine-tuning a real model. The ``make`` fixture builds an estimator against a
parametrized backend (``fake`` always, ``hf`` / ``mlx`` under ``-m slow``), so a
single contract test runs unchanged across every backend.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
import pytest

from sklm import GenerationConfig, TrainingConfig
from sklm.backend import LanguageModelBackend


def _stable(text: str) -> float:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(text)) / (len(text) + 1)


class FakeBackend:
    """Deterministic backend: ``value`` fixes ``generate``; ``scores`` (or a stable
    hash) fixes ``score``. When ``scores`` is given, a continuation missing from it
    raises (a silent fallback would hide tests scoring the wrong encoding). Records
    what it was asked to do for white-box assertions."""

    def __init__(
        self,
        *,
        value: str = "0",
        scores: dict[str, float] | None = None,
        token_counts: dict[str, int] | None = None,
    ) -> None:
        self.value = value
        self.scores = scores
        self.token_counts = token_counts or {}
        self.fit_calls = 0
        self.epoch_texts: Callable[[int], list] | None = None
        self.last_epoch_texts: list[str] | None = None
        self.last_examples: list | None = None
        self.last_eval_examples: list | None = None
        self.generate_batches: list[int] = []
        self.score_batches: list[int] = []

    def fit(
        self,
        epoch_texts: Callable[[int], list],
        training: object,
        model_config: object,
        *,
        random_state: int | None,
        callback: object = None,
        eval_examples: list | None = None,
    ) -> None:
        self.fit_calls += 1
        self.epoch_texts = epoch_texts
        examples = epoch_texts(0)
        self.last_examples = examples
        self.last_epoch_texts = [ex.text for ex in examples]
        self.last_eval_examples = eval_examples

    def generate(self, prompts: Sequence[str], generation: object) -> list[str]:
        self.generate_batches.append(len(prompts))
        return [self.value for _ in prompts]

    def score(
        self,
        prompts: Sequence[str],
        continuations: Sequence[str],
        *,
        reduce: str = "mean",
    ) -> list[float]:
        self.score_batches.append(len(prompts))
        if self.scores is not None:
            missing = sorted({c for c in continuations if c not in self.scores})
            if missing:
                raise KeyError(f"continuations not covered by FakeBackend.scores: {missing}")
            means = [self.scores[c] for c in continuations]
        else:
            means = [-_stable(c) for c in continuations]
        # `scores` values are per-token means; sum scales by the stubbed token count
        # (default 1), so length effects only surface when a test sets token_counts.
        if reduce == "sum":
            return [
                m * self.token_counts.get(c, 1) for m, c in zip(means, continuations, strict=True)
            ]
        return means


@pytest.fixture
def clf_data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "age": rng.integers(18, 80, size=20),
            "city": rng.choice(["SP", "RJ", "BH"], size=20),
            "score": rng.normal(size=20),
        }
    )
    y = rng.choice(["yes", "no"], size=20)
    return X, y


@pytest.fixture
def reg_data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(1)
    age = rng.integers(18, 80, size=20).astype(float)
    city = rng.choice(["SP", "RJ"], size=20)
    X = pd.DataFrame({"age": age, "city": city})
    y = age / 10 + rng.normal(scale=0.1, size=20)
    return X, y


@pytest.fixture
def nan_data() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    X = pd.DataFrame(
        {
            "age": rng.integers(18, 80, size=15).astype(float),
            "city": rng.choice(["SP", "RJ"], size=15).astype(object),
        }
    )
    X.loc[0, "age"] = np.nan
    X.loc[1, "city"] = np.nan
    X.loc[2, "age"] = np.nan
    return X


@pytest.fixture
def imbalanced_data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(20, 3)), columns=pd.Index(["a", "b", "c"]))
    y = np.array([0] * 16 + [1] * 4)
    return X, y


@dataclass(frozen=True)
class BackendSpec:
    """How to build an estimator for one backend under test.

    ``deterministic`` marks the fake backend, whose fixed outputs let a test
    assert exact values; real backends only support shape/contract assertions.
    """

    name: str
    deterministic: bool
    backend: Callable[[], LanguageModelBackend | str]
    model: str
    training: TrainingConfig
    generation: GenerationConfig | None = None


def _has_hf() -> bool:
    return bool(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"))


def _has_mlx() -> bool:
    # Any installed MLX build -- CPU, CUDA or Metal; the backend runs on all three.
    return bool(importlib.util.find_spec("mlx") and importlib.util.find_spec("mlx_lm"))


# distilgpt2's own repo is not mlx-loadable; this mirror is (same GPT-2 arch).
_MLX_MODEL = "mlx-community/distilgpt2"

_BACKENDS = [
    pytest.param(
        BackendSpec(
            name="fake",
            deterministic=True,
            backend=FakeBackend,
            model="distilgpt2",
            training=TrainingConfig(epochs=1),
        ),
        id="fake",
    ),
    pytest.param(
        BackendSpec(
            name="hf",
            deterministic=False,
            backend=lambda: "huggingface",
            model="distilgpt2",
            training=TrainingConfig(epochs=6, batch_size=4),
            generation=GenerationConfig(),
        ),
        id="hf",
        marks=[
            pytest.mark.slow,
            pytest.mark.skipif(not _has_hf(), reason="requires the 'hf' extra"),
        ],
    ),
    pytest.param(
        BackendSpec(
            name="mlx",
            deterministic=False,
            backend=lambda: "mlx",
            model=_MLX_MODEL,
            training=TrainingConfig(epochs=4, batch_size=4),
            generation=GenerationConfig(),
        ),
        id="mlx",
        marks=[
            pytest.mark.slow,
            pytest.mark.skipif(not _has_mlx(), reason="requires the 'mlx' extra"),
        ],
    ),
]


@pytest.fixture(params=_BACKENDS)
def spec(request: pytest.FixtureRequest) -> BackendSpec:
    """One entry of the backend matrix (``fake`` | ``hf`` | ``mlx``)."""
    return request.param


class MakeEstimator(Protocol):
    """Builds an estimator of the requested class wired to the active backend."""

    def __call__[T](self, cls: type[T], **overrides: object) -> T: ...


@pytest.fixture
def make(spec: BackendSpec) -> MakeEstimator:
    """Build an estimator wired to the active backend; ``overrides`` win over defaults."""

    def _make[T](cls: type[T], **overrides: object) -> T:
        kwargs: dict[str, object] = {
            "model": spec.model,
            "backend": spec.backend(),
            "training": deepcopy(spec.training),
            "random_state": 0,
        }
        if spec.generation is not None:
            kwargs["generation"] = deepcopy(spec.generation)
        kwargs.update(overrides)
        return cls(**kwargs)

    return _make
