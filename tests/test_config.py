from __future__ import annotations

import importlib.util
from collections.abc import Callable

import pytest

from sklm import (
    DiscretizationConfig,
    EvalConfig,
    LoRAConfig,
    LRScheduler,
    ModelConfig,
    QuantizationConfig,
    TrainingConfig,
    aggregate_default,
)
from sklm.backend import resolve_max_seq_length
from sklm.base import resolve_quantization
from sklm.config import CosineLR, PlateauLR
from sklm.hf_backend import _hqq_config, _resolve_hf_method, _training_kwargs
from sklm.mlx_backend import _PlateauReducer, _resolve_mlx_method
from sklm.serialize import TrainingExample


def test_resolved_learning_rate_auto_full() -> None:
    assert LRScheduler.cosine().resolved_learning_rate(ModelConfig()) == 2e-5


def test_resolved_learning_rate_auto_lora() -> None:
    cfg = ModelConfig(lora=LoRAConfig())
    assert LRScheduler.cosine().resolved_learning_rate(cfg) == 2e-4


def test_resolved_learning_rate_explicit_overrides() -> None:
    cfg = ModelConfig(lora=LoRAConfig())
    assert LRScheduler.cosine(learning_rate=1e-3).resolved_learning_rate(cfg) == 1e-3


def test_default_scheduler_is_cosine_auto() -> None:
    sched = TrainingConfig().lr_scheduler
    assert isinstance(sched, CosineLR)
    assert sched.learning_rate == "auto"
    assert sched.warmup_ratio == 0.1
    assert sched.floor == 1e-7


def test_default_schedulers_are_independent_per_config() -> None:
    # field(default_factory=...) hands each TrainingConfig its own scheduler, so
    # a set_params on one never leaks into another (convention #1).
    a, b = TrainingConfig(), TrainingConfig()
    assert a.lr_scheduler is not b.lr_scheduler


def test_plateau_requires_evaluation() -> None:
    with pytest.raises(ValueError, match="evaluation"):
        TrainingConfig(lr_scheduler=LRScheduler.plateau(), evaluation=None)


def test_plateau_with_evaluation_constructs() -> None:
    cfg = TrainingConfig(
        lr_scheduler=LRScheduler.plateau(patience=3), evaluation=EvalConfig(split=0.2)
    )
    assert isinstance(cfg.lr_scheduler, PlateauLR)


@pytest.mark.parametrize(
    "make",
    [
        lambda: LRScheduler.plateau(factor=0.0),
        lambda: LRScheduler.plateau(factor=1.0),
        lambda: LRScheduler.plateau(patience=0),
        lambda: LRScheduler.plateau(threshold=-0.1),
        lambda: LRScheduler.plateau(floor=-1.0),
        lambda: LRScheduler.plateau(cooldown=-1),
    ],
)
def test_plateau_invalid_params_raise(make: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        make()


@pytest.mark.parametrize("warmup", [-0.1, 1.0, 1.5])
def test_step_scheduler_invalid_warmup_raises(warmup: float) -> None:
    with pytest.raises(ValueError, match="warmup_ratio"):
        LRScheduler.cosine(warmup_ratio=warmup)


def test_plateau_reducer_reduces_after_patience() -> None:
    # patience=2 -> a reduction fires on the 3rd consecutive non-improving eval.
    reducer = _PlateauReducer(PlateauLR(learning_rate=1.0, factor=0.5, patience=2))
    lr, trajectory = 1.0, []
    for loss in [1.0, 1.0, 1.0, 1.0, 1.0]:
        lr = reducer.step(loss, lr)
        trajectory.append(lr)
    assert trajectory == [1.0, 1.0, 1.0, 0.5, 0.5]


def test_plateau_reducer_improvement_resets_patience() -> None:
    reducer = _PlateauReducer(PlateauLR(learning_rate=1.0, factor=0.5, patience=1))
    lr, trajectory = 1.0, []
    for loss in [1.0, 1.0, 0.5, 1.0, 1.0, 1.0]:
        lr = reducer.step(loss, lr)
        trajectory.append(lr)
    # the improvement at index 2 resets the bad-eval count, delaying the cut.
    assert trajectory == [1.0, 1.0, 1.0, 1.0, 0.5, 0.5]


def test_plateau_reducer_respects_floor() -> None:
    reducer = _PlateauReducer(PlateauLR(learning_rate=1.0, factor=0.1, patience=1, floor=0.5))
    lr = 1.0
    for loss in [1.0] * 10:
        lr = reducer.step(loss, lr)
    assert lr == 0.5


def test_training_kwargs_step_scheduler_maps_type_and_warmup() -> None:
    cfg = TrainingConfig(
        lr_scheduler=LRScheduler.cosine(learning_rate=1e-4, warmup_ratio=0.1, floor=0.0)
    )
    kwargs = _training_kwargs(cfg, "cpu", "fp32", "/tmp/out", 16, 0, has_eval=False)
    assert kwargs["lr_scheduler_type"] == "cosine"
    assert kwargs["warmup_ratio"] == 0.1
    assert kwargs["learning_rate"] == 1e-4
    assert "lr_scheduler_kwargs" not in kwargs


def test_training_kwargs_plateau_maps_reduce_on_plateau() -> None:
    cfg = TrainingConfig(
        lr_scheduler=LRScheduler.plateau(learning_rate=2e-5, factor=0.5, patience=3),
        evaluation=EvalConfig(split=0.2),
    )
    kwargs = _training_kwargs(cfg, "cpu", "fp32", "/tmp/out", 16, 0, has_eval=True)
    assert kwargs["lr_scheduler_type"] == "reduce_lr_on_plateau"
    assert kwargs["lr_scheduler_kwargs"] == {
        "mode": "min",
        "factor": 0.5,
        "patience": 3,
        "threshold": 1e-4,
        "min_lr": 0.0,
        "cooldown": 0,
    }
    assert "warmup_ratio" not in kwargs
    assert kwargs["metric_for_best_model"] == "eval_loss"


def test_training_kwargs_cosine_floor_maps_cosine_with_min_lr() -> None:
    cfg = TrainingConfig(lr_scheduler=LRScheduler.cosine(learning_rate=1e-3, floor=1e-5))
    kwargs = _training_kwargs(cfg, "cpu", "fp32", "/tmp/out", 16, 0, has_eval=False)
    assert kwargs["lr_scheduler_type"] == "cosine_with_min_lr"
    assert kwargs["lr_scheduler_kwargs"] == {"min_lr": 1e-5}


def test_training_kwargs_linear_floor_maps_polynomial() -> None:
    cfg = TrainingConfig(lr_scheduler=LRScheduler.linear(learning_rate=1e-3, floor=1e-5))
    kwargs = _training_kwargs(cfg, "cpu", "fp32", "/tmp/out", 16, 0, has_eval=False)
    assert kwargs["lr_scheduler_type"] == "polynomial"
    assert kwargs["lr_scheduler_kwargs"] == {"lr_end": 1e-5, "power": 1.0}


def test_training_kwargs_cosine_without_floor_stays_cosine() -> None:
    cfg = TrainingConfig(lr_scheduler=LRScheduler.cosine(learning_rate=1e-3, floor=0.0))
    kwargs = _training_kwargs(cfg, "cpu", "fp32", "/tmp/out", 16, 0, has_eval=False)
    assert kwargs["lr_scheduler_type"] == "cosine"
    assert "lr_scheduler_kwargs" not in kwargs


@pytest.mark.parametrize(
    "make", [lambda: LRScheduler.cosine(floor=-1e-6), lambda: LRScheduler.linear(floor=-0.5)]
)
def test_decay_scheduler_invalid_floor_raises(make: Callable[[], object]) -> None:
    with pytest.raises(ValueError, match="floor"):
        make()


def test_constant_has_no_floor() -> None:
    with pytest.raises(TypeError):
        LRScheduler.constant(floor=1e-5)  # type: ignore[call-arg]  # constant does not decay


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
    ("bits", "method"), [(4, "bitsandbytes"), (8, "bitsandbytes"), (2, "hqq"), (3, "hqq")]
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
