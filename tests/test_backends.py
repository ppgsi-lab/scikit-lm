# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Backend-specific paths that the parametrized contract cannot cover: LoRA
adapters and quantization differ by backend. Tests marked ``slow`` fine-tune or
load a real model and self-skip without their dependency; the
``resolve_backend`` tests are fast and run everywhere.
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
    LRScheduler,
    MLXBackend,
    ModelConfig,
    TrainingConfig,
)
from sklm import base as sklm_base
from sklm.backend import LanguageModelBackend
from sklm.base import resolve_backend

from .conftest import FakeBackend, _has_hf, _has_mlx

_MLX_MODEL = "mlx-community/distilgpt2"


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


def test_zero_base_dropout_silences_only_dropout_fields() -> None:
    """``_zero_base_dropout`` zeroes float dropout rates and leaves the rest."""
    from types import SimpleNamespace

    from sklm.hf_backend import _zero_base_dropout

    config = SimpleNamespace(
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        attention_dropout=0.2,
        n_layer=6,
        scale_attn_weights=True,
        model_type="gpt2",
    )
    _zero_base_dropout(config)
    assert config.resid_pdrop == config.embd_pdrop == config.attn_pdrop == 0.0
    assert config.attention_dropout == 0.0
    assert config.n_layer == 6
    assert config.scale_attn_weights is True
    assert config.model_type == "gpt2"


@pytest.mark.skipif(not _has_hf(), reason="requires the 'hf' extra (torch dataset)")
def test_hf_dataset_masks_prompt_with_minus_100() -> None:
    """The dataset emits ``labels`` with -100 over the prompt span (and supervises
    everything when the prompt is empty). Building the mask as a first-class
    ``labels`` column keeps it from being dropped by the Trainer's
    ``remove_unused_columns``, which silently disabled ``loss_on_target_only``."""
    from sklm.hf_backend import _text_dataset
    from sklm.serialize import TrainingExample

    class _CharTok:
        eos_token = ""

        def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
            ids = [ord(c) for c in text]
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    masked = _text_dataset(lambda _: [TrainingExample("ab", "cd")], _CharTok(), 100)[0]
    assert masked["labels"] == [-100, -100, ord("c"), ord("d")]

    unmasked = _text_dataset(lambda _: [TrainingExample("", "cd")], _CharTok(), 100)[0]
    assert unmasked["labels"] == [ord("c"), ord("d")]


@pytest.mark.slow
@pytest.mark.skipif(not _has_hf(), reason="requires the 'hf' extra")
def test_hf_zeroes_base_dropout_on_real_config() -> None:
    """A real GPT-2 config has every dropout zeroed (parity with the MLX backend)."""
    from transformers import AutoConfig

    from sklm.hf_backend import _zero_base_dropout

    config = AutoConfig.from_pretrained(_MLX_MODEL)
    assert config.resid_pdrop > 0  # the checkpoint ships nonzero dropout
    _zero_base_dropout(config)
    assert config.resid_pdrop == config.embd_pdrop == config.attn_pdrop == 0.0


@pytest.mark.slow
@pytest.mark.skipif(not _has_hf(), reason="requires the 'hf' extra")
def test_hf_lora_and_training_knobs() -> None:
    """LoRA adapters (GPT-2 ``c_attn``) plus a non-default optimizer schedule."""
    X, y = _clf_xy()
    clf = LanguageModelClassifier(
        training=TrainingConfig(
            epochs=2,
            batch_size=8,
            grad_accumulation_steps=2,
            lr_scheduler=LRScheduler.cosine(warmup_ratio=0.1),
            weight_decay=0.01,
        ),
        lora=LoRAConfig(rank=4, alpha=8, target_modules=["c_attn"]),
        random_state=0,
    ).fit(X, y)
    assert set(clf.predict(X.head(3))).issubset(set(clf.classes_))


@pytest.mark.slow
@pytest.mark.skipif(not _has_hf(), reason="requires the 'hf' extra")
def test_hf_generation_knobs() -> None:
    """top_p / top_k / repetition_penalty reach the sampler without error."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=20), "b": rng.choice(["x", "y"], size=20)})
    X.loc[0, "a"] = np.nan
    out = LanguageModelImputer(
        training=TrainingConfig(epochs=2, batch_size=8),
        generation=GenerationConfig(temperature=0.8, top_p=0.9, top_k=40, repetition_penalty=1.2),
        random_state=0,
    ).fit_transform(X)
    assert out.isna().sum().sum() == 0


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
@pytest.mark.skipif(not _has_hf(), reason="requires the 'hf' extra")
def test_hf_score_truncates_right() -> None:
    _score_truncates_right(HFBackend(), "distilgpt2")


@pytest.mark.slow
@pytest.mark.skipif(not _has_mlx(), reason="requires the 'mlx' extra on Apple Silicon")
def test_mlx_score_truncates_right() -> None:
    _score_truncates_right(MLXBackend(), _MLX_MODEL)


# --- HF/MLX score rank parity on shared weights ----------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not (_has_hf() and _has_mlx()),
    reason="requires both the 'hf' and 'mlx' extras on Apple Silicon",
)
def test_score_ranks_identically_across_backends() -> None:
    """Convention: ``HFBackend.score`` and ``MLXBackend.score`` are behaviorally
    identical. On the unmodified base model (``mlx-community/distilgpt2`` mirrors
    distilgpt2's GPT-2 weights in an mlx-loadable layout) both backends must rank
    a candidate set the same way -- including a prompt ending in a space, where
    BPE merges the space into the continuation's first token and the
    longest-common-token-prefix boundary diverges from the character boundary.
    """
    hf = HFBackend()
    hf._load("distilgpt2", ModelConfig())
    mlx = MLXBackend()
    mlx._load(_MLX_MODEL, ModelConfig())

    candidates = ["Paris", "London", "banana", "the the the"]
    for prompt in ["The capital of France is ", "The capital of France is"]:
        prompts = [prompt] * len(candidates)
        hf_scores = hf.score(prompts, candidates)
        mlx_scores = mlx.score(prompts, candidates)
        assert np.argsort(hf_scores).tolist() == np.argsort(mlx_scores).tolist(), prompt
        # observed cross-stack drift is ~0.02 in mean per-token log-likelihood
        np.testing.assert_allclose(hf_scores, mlx_scores, atol=0.1)


# --- resolve_backend (fast, no torch/mlx needed) ----------------------------


def test_resolve_backend_unknown_selector_raises() -> None:
    with pytest.raises(ValueError, match="unknown backend 'nope'"):
        resolve_backend("nope")


def test_resolve_backend_passes_instances_through() -> None:
    backend = FakeBackend()
    assert resolve_backend(backend) is backend


class _SentinelMLXBackend(FakeBackend):
    """Stands in for the lazily imported MLX backend in resolution tests."""


def _resolve_auto_with(
    monkeypatch: pytest.MonkeyPatch,
    *,
    system: str,
    hf: bool = False,
    mlx: bool = False,
    hf_gpu: bool = False,
    mlx_gpu: bool = False,
) -> LanguageModelBackend:
    monkeypatch.setattr(sklm_base.platform, "system", lambda: system)
    monkeypatch.setattr(sklm_base, "_has_hf", lambda: hf)
    monkeypatch.setattr(sklm_base, "_has_mlx", lambda: mlx)
    monkeypatch.setattr(sklm_base, "_hf_gpu_available", lambda: hf_gpu)
    monkeypatch.setattr(sklm_base, "_mlx_gpu_available", lambda: mlx_gpu)
    monkeypatch.setattr(sklm_base, "_mlx_backend", _SentinelMLXBackend)
    return resolve_backend("auto")


def test_resolve_auto_darwin_prefers_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    resolved = _resolve_auto_with(monkeypatch, system="Darwin", hf=True, mlx=True, hf_gpu=True)
    assert isinstance(resolved, _SentinelMLXBackend)


def test_resolve_auto_darwin_without_mlx_falls_back_to_hf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = _resolve_auto_with(monkeypatch, system="Darwin", hf=True)
    assert isinstance(resolved, HFBackend)


def test_resolve_auto_linux_prefers_hf_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    resolved = _resolve_auto_with(
        monkeypatch, system="Linux", hf=True, mlx=True, hf_gpu=True, mlx_gpu=True
    )
    assert isinstance(resolved, HFBackend)


def test_resolve_auto_linux_mlx_gpu_beats_hf_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    resolved = _resolve_auto_with(monkeypatch, system="Linux", hf=True, mlx=True, mlx_gpu=True)
    assert isinstance(resolved, _SentinelMLXBackend)


def test_resolve_auto_linux_hf_cpu_beats_mlx_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    resolved = _resolve_auto_with(monkeypatch, system="Linux", hf=True, mlx=True)
    assert isinstance(resolved, HFBackend)


def test_resolve_auto_linux_mlx_cpu_last(monkeypatch: pytest.MonkeyPatch) -> None:
    resolved = _resolve_auto_with(monkeypatch, system="Linux", mlx=True)
    assert isinstance(resolved, _SentinelMLXBackend)


def test_resolve_auto_nothing_installed_falls_back_to_hf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = _resolve_auto_with(monkeypatch, system="Linux")
    assert isinstance(resolved, HFBackend)
