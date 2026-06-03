from __future__ import annotations

import importlib.util

import pytest

from sklm import (
    DiscretizationConfig,
    LoRAConfig,
    ModelConfig,
    QuantizationConfig,
    TrainingConfig,
    aggregate_default,
)
from sklm.backend import resolve_max_seq_length
from sklm.base import resolve_quantization
from sklm.hf_backend import _hqq_config, _resolve_hf_method
from sklm.mlx_backend import _resolve_mlx_method
from sklm.serialize import TrainingExample


def test_resolved_learning_rate_auto_full() -> None:
    assert TrainingConfig().resolved_learning_rate(ModelConfig()) == 2e-5


def test_resolved_learning_rate_auto_lora() -> None:
    cfg = ModelConfig(lora=LoRAConfig())
    assert TrainingConfig().resolved_learning_rate(cfg) == 2e-4


def test_resolved_learning_rate_explicit_overrides() -> None:
    cfg = ModelConfig(lora=LoRAConfig())
    assert TrainingConfig(learning_rate=1e-3).resolved_learning_rate(cfg) == 1e-3


def test_aggregate_default_numeric_is_mean() -> None:
    assert aggregate_default([1.0, 2.0, 3.0], numeric=True) == 2.0


def test_aggregate_default_categorical_is_mode() -> None:
    assert aggregate_default(["x", "y", "x"], numeric=False) == "x"


def test_aggregate_default_mode_breaks_ties_by_first_occurrence() -> None:
    assert aggregate_default(["b", "a", "a", "b"], numeric=False) == "b"


def test_aggregate_default_singleton_returns_the_value() -> None:
    assert aggregate_default([4.5], numeric=True) == 4.5
    assert aggregate_default(["only"], numeric=False) == "only"


def test_max_seq_length_defaults_to_none() -> None:
    assert TrainingConfig().max_seq_length is None


@pytest.mark.parametrize("value", [0, -1])
def test_max_seq_length_rejects_non_positive(value: int) -> None:
    with pytest.raises(ValueError):
        TrainingConfig(max_seq_length=value)


def test_resolve_max_seq_length_rounds_longest_row_up() -> None:
    examples = [TrainingExample("", "x" * 10), TrainingExample("", "x" * 25)]
    assert resolve_max_seq_length(examples, len) == 32


def test_resolve_max_seq_length_keeps_exact_multiple() -> None:
    assert resolve_max_seq_length([TrainingExample("", "x" * 16)], len) == 16


@pytest.mark.parametrize("bins", [0, 0.0, 3, 0.5, 1.0])
def test_discretization_valid_bins_construct(bins: int | float) -> None:
    assert DiscretizationConfig(bins=bins).bins == bins


@pytest.mark.parametrize("bins", [1.5, -1, -0.3, True])
def test_discretization_invalid_bins_raise(bins: object) -> None:
    with pytest.raises(ValueError):
        DiscretizationConfig(bins=bins)  # type: ignore[arg-type]  # invalid by design


def test_discretization_resolve_k_int_caps_at_n_unique() -> None:
    assert DiscretizationConfig(bins=3).resolve_k(10) == 3
    assert DiscretizationConfig(bins=100).resolve_k(7) == 7


def test_discretization_resolve_k_float_ceils_with_floor_of_one() -> None:
    assert DiscretizationConfig(bins=0.5).resolve_k(5) == 3  # ceil(2.5), not round -> 2
    assert DiscretizationConfig(bins=0.1).resolve_k(3) == 1  # ceil(0.3) floored to 1
    assert DiscretizationConfig(bins=1.0).resolve_k(7) == 7  # full support


def test_resolve_quantization_literal_to_config() -> None:
    assert resolve_quantization("3bit") == QuantizationConfig(bits=3, method="auto")


def test_resolve_quantization_passes_config_through() -> None:
    cfg = QuantizationConfig(bits=2, method="hqq", group_size=32)
    assert resolve_quantization(cfg) is cfg


def test_resolve_quantization_none_stays_none() -> None:
    assert resolve_quantization(None) is None


@pytest.mark.parametrize(
    ("bits", "method"),
    [(4, "bitsandbytes"), (8, "bitsandbytes"), (2, "hqq"), (3, "hqq")],
)
def test_hf_method_auto_routes_by_bits(bits: int, method: str) -> None:
    assert _resolve_hf_method(QuantizationConfig(bits=bits)) == method


def test_hf_method_six_bit_has_no_provider() -> None:
    # 6-bit is in neither bitsandbytes {4,8} nor HQQ {1,2,3,4,8}: MLX-only in practice.
    with pytest.raises(ValueError, match="bits"):
        _resolve_hf_method(QuantizationConfig(bits=6))


def test_hf_method_explicit_bitsandbytes_rejects_low_bits() -> None:
    with pytest.raises(ValueError, match="bitsandbytes"):
        _resolve_hf_method(QuantizationConfig(bits=2, method="bitsandbytes"))


def test_hf_method_rejects_mlx_method() -> None:
    with pytest.raises(ValueError, match="not available"):
        _resolve_hf_method(QuantizationConfig(bits=4, method="mlx"))


@pytest.mark.parametrize("bits", [2, 3, 4, 6, 8])
def test_mlx_method_accepts_native_widths(bits: int) -> None:
    assert _resolve_mlx_method(QuantizationConfig(bits=bits)) == "mlx"


def test_mlx_method_rejects_unsupported_bits() -> None:
    with pytest.raises(ValueError, match="bits"):
        _resolve_mlx_method(QuantizationConfig(bits=5))


def test_mlx_method_rejects_non_mlx_method() -> None:
    with pytest.raises(ValueError, match="mlx"):
        _resolve_mlx_method(QuantizationConfig(bits=4, method="bitsandbytes"))


@pytest.mark.skipif(
    importlib.util.find_spec("hqq") is not None, reason="exercises the missing-'hqq'-extra message"
)
def test_hqq_config_requires_extra_when_missing() -> None:
    with pytest.raises(ImportError, match="hqq"):
        _hqq_config(QuantizationConfig(bits=3, method="hqq"))
