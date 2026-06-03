# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Backend-specific paths that the parametrized contract cannot cover: LoRA
adapters and quantization differ by backend. Each test is slow (it fine-tunes a
real model) and self-skips without its dependency.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from sklm import (
    GenerationConfig,
    HFBackend,
    LanguageModelClassifier,
    LanguageModelImputer,
    LoRAConfig,
    MLXBackend,
    ModelConfig,
    TrainingConfig,
)

from .conftest import _has_hf, _has_mlx

pytestmark = pytest.mark.slow

_MLX_MODEL = "gabfssilva/distilgpt2"


def _has_mps_bnb() -> bool:
    if not (_has_hf() and importlib.util.find_spec("mps_bitsandbytes")):
        return False
    import torch

    return bool(torch.backends.mps.is_available())


def _clf_xy() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=24), "b": rng.choice(["x", "y"], size=24)})
    y = rng.choice(["pos", "neg"], size=24)
    return X, y


# --- HuggingFace ----------------------------------------------------------


@pytest.mark.skipif(not _has_hf(), reason="requires the 'hf' extra")
def test_hf_lora_and_training_knobs() -> None:
    """LoRA adapters (GPT-2 ``c_attn``) plus a non-default optimizer schedule."""
    X, y = _clf_xy()
    clf = LanguageModelClassifier(
        training=TrainingConfig(
            epochs=2,
            batch_size=8,
            grad_accumulation_steps=2,
            lr_scheduler="cosine",
            warmup_ratio=0.1,
            weight_decay=0.01,
        ),
        lora=LoRAConfig(rank=4, alpha=8, target_modules=["c_attn"]),
        random_state=0,
    ).fit(X, y)
    assert set(clf.predict(X.head(3))).issubset(set(clf.classes_))


@pytest.mark.skipif(not _has_hf(), reason="requires the 'hf' extra")
def test_hf_generation_knobs() -> None:
    """top_p / top_k / repetition_penalty reach the sampler without error."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=20), "b": rng.choice(["x", "y"], size=20)})
    X.loc[0, "a"] = np.nan
    out = LanguageModelImputer(
        training=TrainingConfig(epochs=2, batch_size=8),
        generation=GenerationConfig(
            temperature=0.8, top_p=0.9, top_k=40, repetition_penalty=1.2
        ),
        random_state=0,
    ).fit_transform(X)
    assert out.isna().sum().sum() == 0


@pytest.mark.skipif(not _has_mps_bnb(), reason="requires MPS + mps-bitsandbytes")
def test_hf_qlora_mps() -> None:
    """QLoRA on Apple Silicon: 4-bit quantization + LoRA + an 8-bit optimizer."""
    X, y = _clf_xy()
    clf = LanguageModelClassifier(
        training=TrainingConfig(epochs=2, batch_size=8, optimizer="adamw_8bit"),
        quantization="4bit",
        lora=LoRAConfig(rank=4, alpha=8, target_modules=["c_attn"]),
        device="mps",
        random_state=0,
    ).fit(X, y)
    assert np.isfinite(clf.predict_proba(X.head(3))).all()


# --- MLX ------------------------------------------------------------------


@pytest.mark.skipif(not _has_mlx(), reason="requires the 'mlx' extra on Apple Silicon")
def test_mlx_lora_all_linear() -> None:
    """ "all-linear" auto-discovers LoRA targets (portable across backends)."""
    X, y = _clf_xy()
    clf = LanguageModelClassifier(
        model=_MLX_MODEL,
        backend="mlx",
        training=TrainingConfig(epochs=2, batch_size=8),
        lora=LoRAConfig(rank=4, alpha=8, target_modules="all-linear"),
        random_state=0,
    ).fit(X, y)
    assert set(clf.predict(X.head(3))).issubset(set(clf.classes_))


@pytest.mark.skipif(not _has_mlx(), reason="requires the 'mlx' extra on Apple Silicon")
@pytest.mark.parametrize("quantization", ["2bit", "3bit", "4bit", "6bit", "8bit"])
def test_mlx_quantization(quantization: str) -> None:
    """MLX quantization + LoRA at every supported bit width: convert, load
    quantized, fine-tune, predict."""
    X, y = _clf_xy()
    clf = LanguageModelClassifier(
        model=_MLX_MODEL,
        backend="mlx",
        training=TrainingConfig(epochs=2, batch_size=8),
        quantization=quantization,  # type: ignore[arg-type]  # parametrized Quantization literal
        # MLX matches the relative module path inside a block, not a name suffix.
        lora=LoRAConfig(rank=4, alpha=8, target_modules=["attn.c_attn"]),
        random_state=0,
    ).fit(X, y)
    assert np.isfinite(clf.predict_proba(X.head(3))).all()


# --- score over-length truncation parity (H2) -----------------------------


def _score_truncates_right(backend: HFBackend | MLXBackend, model: str) -> None:
    """Both backends right-truncate over-length pairs, keeping the prompt prefix
    so the longest-common-prefix boundary stays valid. When the prompt alone fills
    the window no continuation tokens survive, so the score is -inf identically on
    both backends -- the property that left-truncation (the old HF path) violated.
    """
    backend._load(model, ModelConfig())
    backend._max_seq_length = 16
    (fitting,) = backend.score(["color: "], ["red"])
    assert fitting != float("-inf")
    long_prompt = "the quick brown fox jumps over the lazy dog " * 4
    (overflow,) = backend.score([long_prompt], ["red"])
    assert overflow == float("-inf")


@pytest.mark.skipif(not _has_hf(), reason="requires the 'hf' extra")
def test_hf_score_truncates_right() -> None:
    _score_truncates_right(HFBackend(), "distilgpt2")


@pytest.mark.skipif(not _has_mlx(), reason="requires the 'mlx' extra on Apple Silicon")
def test_mlx_score_truncates_right() -> None:
    _score_truncates_right(MLXBackend(), _MLX_MODEL)
