"""The shared mechanism, observable only through the (fake) backend.

Two guarantees the public output never exposes: per-epoch column-order
re-permutation (``augmentation_factor``) and ``target_loss_weight`` masking.
Plus inference batch-size invariance: chunking must never change a result.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from itertools import permutations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from sklm import (
    EvalConfig,
    GenerationConfig,
    JSONSerializer,
    LanguageModelClassifier,
    LanguageModelImputer,
    LanguageModelRegressor,
    ModelConfig,
    Serializer,
    SpacedDigits,
    TabularLanguageModel,
    TrainingConfig,
)
from sklm.core import _ScoreSpec
from sklm.order import positional_order

from .conftest import FakeBackend, _stable


def _fit(
    frame: pd.DataFrame,
    *,
    target_cols: set[str] | None = None,
    augmentation_factor: int = 1,
    target_at_end: bool = False,
    permute_order: bool = True,
    target_loss_weight: float | None = None,
    numeric_loss_weight: float = 0.0,
    numeric_noise: float = 0.0,
    column_dropout: float = 0.0,
    serializer: Serializer | None = None,
    random_state: int = 0,
) -> list:
    fake = FakeBackend()
    lm = TabularLanguageModel(
        backend=fake,
        serializer=serializer if serializer is not None else JSONSerializer(),
        training=TrainingConfig(
            augmentation_factor=augmentation_factor,
            epochs=1,
            target_at_end=target_at_end,
            permute_order=permute_order,
            target_loss_weight=target_loss_weight,
            numeric_loss_weight=numeric_loss_weight,
            numeric_noise=numeric_noise,
            column_dropout=column_dropout,
            # every example must reach the backend: these frames are 2-4 rows
            evaluation=None,
        ),
        model=ModelConfig(model="m"),
        random_state=random_state,
    )
    lm.fit(frame, target_cols=frozenset(target_cols) if target_cols else frozenset())
    assert fake.last_examples is not None
    return fake.last_examples


# --- column-order re-permutation -----------------------------------------


def test_default_factor_emits_one_variant_per_row() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [4.0, 5.0, 6.0]})
    assert len(_fit(frame)) == len(frame)


def test_factor_is_capped_by_column_factorial() -> None:
    frame = pd.DataFrame({"a": [1], "b": ["x"], "c": [2.0]})
    examples = _fit(frame, augmentation_factor=10)  # min(10, 3!) = 6
    orders = [tuple(json.loads(ex.text).keys()) for ex in examples]
    assert len(orders) == math.factorial(3)
    assert len(set(orders)) == math.factorial(3)


def test_missing_cells_reduce_per_row_variants() -> None:
    frame = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", "y", np.nan], "c": [4.0, 5.0, 6.0]})
    # present per row: 3 -> min(4,6)=4, 2 -> min(4,2)=2, 2 -> min(4,2)=2
    assert len(_fit(frame, augmentation_factor=4)) == 4 + 2 + 2


def test_permutation_is_reproducible_across_fits() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [4.0, 5.0, 6.0]})
    first = _fit(frame, augmentation_factor=3, random_state=42)
    second = _fit(frame, augmentation_factor=3, random_state=42)
    assert [e.text for e in first] == [e.text for e in second]


def test_invalid_factor_raises() -> None:
    with pytest.raises(ValueError, match="augmentation_factor"):
        TrainingConfig(augmentation_factor=0)


def test_column_dropout_defaults_to_zero() -> None:
    assert TrainingConfig().column_dropout == 0.0


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.1])
def test_column_dropout_domain(bad: float) -> None:
    with pytest.raises(ValueError, match="column_dropout"):
        TrainingConfig(column_dropout=bad)


# --- epoch_texts contract --------------------------------------------------


def _fitted_fake(frame: pd.DataFrame) -> FakeBackend:
    fake = FakeBackend()
    TabularLanguageModel(
        backend=fake,
        serializer=JSONSerializer(),
        training=TrainingConfig(epochs=2),
        model=ModelConfig(model="m"),
        random_state=0,
    ).fit(frame)
    return fake


def test_epoch_texts_is_pure_within_an_epoch() -> None:
    # both real backends call epoch_texts(0) twice (max_seq_length pre-pass,
    # then the dataset) and rely on getting identical data back
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [4.0, 5.0, 6.0]})
    fake = _fitted_fake(frame)
    assert fake.epoch_texts is not None
    assert [ex.text for ex in fake.epoch_texts(0)] == [ex.text for ex in fake.epoch_texts(0)]


def test_epoch_texts_repermutes_column_order_across_epochs() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [4.0, 5.0, 6.0]})
    fake = _fitted_fake(frame)
    assert fake.epoch_texts is not None
    # sorting removes the example shuffle, so any difference is a column-order change
    epoch0 = sorted(ex.text for ex in fake.epoch_texts(0))
    epoch1 = sorted(ex.text for ex in fake.epoch_texts(1))
    assert epoch0 != epoch1


# --- numeric_noise ----------------------------------------------------------


_NOISE_FRAME = pd.DataFrame({
    "a": [1, 2, 3, 4],
    "b": ["x", "y", "z", "w"],
    "c": [4.5, 5.1, 6.3, 7.9],
    "d": [7.0, 7.0, 7.0, 7.0],
})


def test_noise_perturbs_numeric_cells_preserving_format() -> None:
    rows = [json.loads(ex.text) for ex in _fit(_NOISE_FRAME, numeric_noise=1.0)]
    originals = {"x": (1, 4.5), "y": (2, 5.1), "z": (3, 6.3), "w": (4, 7.9)}
    assert any(tuple([row["a"], row["c"]]) != originals[row["b"]] for row in rows)
    for row in rows:
        assert isinstance(row["a"], int)  # an integer cell never grows a decimal point
        assert isinstance(row["c"], float)
        assert round(row["c"], 1) == row["c"]  # the cell's decimal count is preserved
        assert row["d"] == 7.0  # a constant column has no sigma and stays untouched


def test_noise_is_pure_within_an_epoch() -> None:
    fake = FakeBackend()
    TabularLanguageModel(
        backend=fake,
        serializer=JSONSerializer(),
        training=TrainingConfig(epochs=2, numeric_noise=1.0),
        model=ModelConfig(model="m"),
        random_state=0,
    ).fit(_NOISE_FRAME)
    assert fake.epoch_texts is not None
    assert [ex.text for ex in fake.epoch_texts(0)] == [ex.text for ex in fake.epoch_texts(0)]


def test_noise_redraws_across_epochs() -> None:
    fake = FakeBackend()
    TabularLanguageModel(
        backend=fake,
        serializer=JSONSerializer(),
        # canonical column order isolates the epoch difference to the noise draws
        training=TrainingConfig(epochs=2, numeric_noise=1.0, permute_order=False),
        model=ModelConfig(model="m"),
        random_state=0,
    ).fit(_NOISE_FRAME)
    assert fake.epoch_texts is not None
    epoch0 = sorted(ex.text for ex in fake.epoch_texts(0))
    epoch1 = sorted(ex.text for ex in fake.epoch_texts(1))
    assert epoch0 != epoch1


def test_noise_is_reproducible_across_fits() -> None:
    first = _fit(_NOISE_FRAME, numeric_noise=1.0, random_state=42)
    second = _fit(_NOISE_FRAME, numeric_noise=1.0, random_state=42)
    assert [e.text for e in first] == [e.text for e in second]


def test_noise_leaves_validation_rows_clean() -> None:
    fake = FakeBackend()
    TabularLanguageModel(
        backend=fake,
        serializer=JSONSerializer(),
        training=TrainingConfig(
            epochs=1, numeric_noise=1.0, evaluation=EvalConfig(split=0.5, stratify=False)
        ),
        model=ModelConfig(model="m"),
        random_state=0,
    ).fit(_NOISE_FRAME)
    assert fake.last_eval_examples
    originals = {"x": (1, 4.5), "y": (2, 5.1), "z": (3, 6.3), "w": (4, 7.9)}
    for ex in fake.last_eval_examples:
        row = json.loads(ex.text)
        assert tuple([row["a"], row["c"]]) == originals[row["b"]]


def test_negative_noise_raises() -> None:
    with pytest.raises(ValueError, match="numeric_noise"):
        TrainingConfig(numeric_noise=-0.1)


def test_noise_allows_augmentation_without_permutation() -> None:
    examples = _fit(_NOISE_FRAME, augmentation_factor=3, permute_order=False, numeric_noise=1.0)
    assert len(examples) == 3 * len(_NOISE_FRAME)
    rows = [json.loads(ex.text) for ex in examples]
    assert all(tuple(row.keys()) == ("a", "b", "c", "d") for row in rows)  # canonical order
    by_key = {key: {json.dumps(r) for r in rows if r["b"] == key} for key in ("x", "y", "z", "w")}
    assert all(len(copies) > 1 for copies in by_key.values())  # noise differentiates the copies


def test_augmentation_without_permutation_still_raises_without_noise() -> None:
    with pytest.raises(ValueError, match="identical duplicates"):
        TrainingConfig(augmentation_factor=2, permute_order=False, numeric_noise=0.0)


def test_augmentation_without_permutation_collapses_when_nothing_to_noise() -> None:
    frame = pd.DataFrame({"b": ["x", "y", "z"], "d": [7.0, 7.0, 7.0]})  # constant numeric
    examples = _fit(frame, augmentation_factor=3, permute_order=False, numeric_noise=1.0)
    assert len(examples) == len(frame)


# --- column_dropout ----------------------------------------------------------


def test_column_dropout_removes_context_only_when_target_is_observed() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "t": [0.0, 1.0, np.nan]})
    examples = _fit(
        frame,
        target_cols={"t"},
        target_loss_weight=1.0,
        column_dropout=0.999999,
        permute_order=False,
    )
    keys = sorted(tuple(json.loads(ex.text)) for ex in examples)
    assert keys == [("a", "b"), ("t",), ("t",)]


def test_column_dropout_preserves_targets_without_target_last() -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "t": [0.0, 1.0]})
    examples = _fit(frame, target_cols={"t"}, column_dropout=0.999999, permute_order=False)
    assert all(tuple(json.loads(ex.text)) == ("t",) for ex in examples)


def test_column_dropout_allows_augmentation_without_permutation() -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6], "d": [7, 8], "t": [0, 1]})
    examples = _fit(
        frame,
        target_cols={"t"},
        target_loss_weight=1.0,
        column_dropout=0.5,
        augmentation_factor=4,
        permute_order=False,
    )
    rows = [json.loads(ex.text) for ex in examples]
    assert len(rows) == 4 * len(frame)
    assert all(len({tuple(row) for row in rows if row["t"] == target}) > 1 for target in (0, 1))


def test_column_dropout_collapses_augmentation_without_context() -> None:
    frame = pd.DataFrame({"t1": [0, 1], "t2": ["x", "y"]})
    examples = _fit(
        frame,
        target_cols={"t1", "t2"},
        target_loss_weight=1.0,
        column_dropout=0.5,
        augmentation_factor=3,
        permute_order=False,
    )
    assert len(examples) == len(frame)


def test_column_dropout_leaves_validation_context_complete() -> None:
    frame = pd.DataFrame({"a": range(8), "b": [f"v{i}" for i in range(8)], "t": range(8)})
    fake = FakeBackend()
    TabularLanguageModel(
        backend=fake,
        serializer=JSONSerializer(),
        training=TrainingConfig(
            epochs=1,
            column_dropout=0.999999,
            target_loss_weight=1.0,
            evaluation=EvalConfig(split=0.5, stratify=False),
            permute_order=False,
        ),
        model=ModelConfig(model="m"),
        random_state=0,
    ).fit(frame, target_cols=frozenset({"t"}))
    assert fake.last_eval_examples
    assert all(tuple(json.loads(ex.text)) == ("a", "b", "t") for ex in fake.last_eval_examples)


# --- target_loss_weight masking ---------------------------------------------


def test_no_weight_leaves_every_prompt_empty() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "t": [0, 1, 0]})
    assert all(ex.prompt == "" for ex in _fit(frame, target_cols={"t"}))


def test_masking_splits_context_and_target_last() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "t": [0, 1, 0]})
    examples = _fit(frame, target_cols={"t"}, target_loss_weight=1.0)
    assert len(examples) == len(frame)
    for ex in examples:
        assert ex.prompt != ""
        assert ex.prompt + ex.completion == ex.text
        assert list(json.loads(ex.text).keys())[-1] == "t"  # target serialized last


def test_target_at_end_fixes_position_without_masking() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "t": [0, 1, 0]})
    examples = _fit(frame, target_cols={"t"}, target_at_end=True)
    assert len(examples) == len(frame)
    for ex in examples:
        assert ex.prompt == ""  # loss stays on the whole row
        assert list(json.loads(ex.text).keys())[-1] == "t"  # but the target is fixed last


def test_masking_skips_rows_with_missing_target() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "t": [0.0, np.nan, 4.0]})
    prompts = [ex.prompt for ex in _fit(frame, target_cols={"t"}, target_loss_weight=1.0)]
    # rows 0 and 2 observe t -> masked; row 1 has t=NaN -> context-only
    assert prompts.count("") == 1
    assert sum(p != "" for p in prompts) == 2


def test_masking_all_columns_target_yields_brace_prompt() -> None:
    frame = pd.DataFrame({"t1": [0, 1], "t2": ["x", "y"]})
    examples = _fit(frame, target_cols={"t1", "t2"}, target_loss_weight=1.0)
    assert all(ex.prompt == "{" for ex in examples)


# --- training permute_order -------------------------------------------------


def test_permute_order_off_keeps_canonical_column_order() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [4.0, 5.0, 6.0]})
    examples = _fit(frame, permute_order=False)
    assert all(list(json.loads(ex.text).keys()) == ["a", "b", "c"] for ex in examples)


def test_permute_order_off_keeps_canonical_blocks_with_target_at_end() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "t": [0, 1, 0]})
    examples = _fit(frame, target_cols={"t"}, target_at_end=True, permute_order=False)
    assert all(list(json.loads(ex.text).keys()) == ["a", "b", "t"] for ex in examples)


def test_permute_order_off_with_augmentation_and_no_masking_raises() -> None:
    with pytest.raises(ValueError, match="permute_order"):
        TrainingConfig(permute_order=False, augmentation_factor=2, numeric_noise=0.0)


def test_generation_permute_order_raises_on_canonical_trained_model(reg_data) -> None:
    X, y = reg_data
    reg = LanguageModelRegressor(
        backend=FakeBackend(value="3.5"), training=TrainingConfig(epochs=1, permute_order=False)
    ).fit(X, y)
    reg.generation = GenerationConfig(n_samples=4)  # permute_order defaults to True
    with pytest.raises(ValueError, match="training.permute_order"):
        reg.predict(X)
    reg.generation = GenerationConfig(n_samples=4, permute_order=False)
    assert len(reg.predict(X)) == len(X)


def test_generation_permute_order_is_inert_on_canonical_model_when_single_sample(clf_data) -> None:
    X, y = clf_data
    clf = LanguageModelClassifier(
        backend=FakeBackend(), training=TrainingConfig(epochs=1, permute_order=False)
    ).fit(X, y)
    assert clf.predict_proba(X).shape == (len(X), len(clf.classes_))  # n_samples=1: no permutation


# --- target_loss_weight -----------------------------------------------------


def test_partial_weight_keeps_prompt_boundary() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "t": [0, 1, 0]})
    examples = _fit(frame, target_cols={"t"}, target_loss_weight=0.65)
    assert len(examples) == len(frame)
    for ex in examples:
        assert ex.prompt != ""  # boundary preserved for the backend's weighting
        assert ex.prompt + ex.completion == ex.text
        assert list(json.loads(ex.text).keys())[-1] == "t"  # target serialized last


def test_target_loss_weight_without_target_cols_leaves_prompts_empty() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    examples = _fit(frame, target_loss_weight=0.65)
    assert all(ex.prompt == "" for ex in examples)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_target_loss_weight_domain(bad: float) -> None:
    with pytest.raises(ValueError, match="target_loss_weight"):
        TrainingConfig(target_loss_weight=bad)


def test_target_loss_weight_boundary_one_is_allowed() -> None:
    assert TrainingConfig(target_loss_weight=1.0).target_loss_weight == 1.0


def test_negative_numeric_loss_weight_raises() -> None:
    with pytest.raises(ValueError, match="numeric_loss_weight"):
        TrainingConfig(numeric_loss_weight=-0.1)


# --- numeric_loss_weight spans ----------------------------------------------


def test_numeric_spans_are_empty_when_the_term_is_off() -> None:
    frame = pd.DataFrame({"a": [1.5, 2.5], "b": ["x", "y"]})
    assert all(ex.numeric_spans == () for ex in _fit(frame))


def test_numeric_spans_slice_the_encoded_numbers() -> None:
    frame = pd.DataFrame({"a": [1.5, 2.5], "b": ["x", "y"], "c": [10.0, -3.25]})
    examples = _fit(
        frame, numeric_loss_weight=0.5, serializer=JSONSerializer(number=SpacedDigits())
    )
    for ex in examples:
        assert len(ex.numeric_spans) == 2  # a and c; b is categorical
        for span in ex.numeric_spans:
            assert float(ex.text[span.start : span.end].replace(" ", "")) == span.value


def test_numeric_spans_cover_only_the_completion_under_masking() -> None:
    frame = pd.DataFrame({"a": [1.5, 2.5], "b": ["x", "y"], "t": [3.0, 4.0]})
    examples = _fit(
        frame,
        target_cols={"t"},
        target_loss_weight=1.0,
        numeric_loss_weight=0.5,
        serializer=JSONSerializer(number=SpacedDigits()),
    )
    for ex in examples:
        assert ex.prompt != ""
        assert len(ex.numeric_spans) == 1  # the context number 'a' is masked out
        span = ex.numeric_spans[0]
        assert span.start >= len(ex.prompt)
        assert float(ex.text[span.start : span.end].replace(" ", "")) == span.value


def test_numeric_spans_carry_the_inverse_column_scale() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, 6.0], "b": [1000.0, 2000.0, 6000.0]})
    examples = _fit(
        frame,
        numeric_loss_weight=0.5,
        permute_order=False,
        serializer=JSONSerializer(number=SpacedDigits()),
    )
    expected = [1.0 / float(frame[c].std(ddof=0)) for c in ("a", "b")]
    for ex in examples:
        assert [s.inv_scale for s in ex.numeric_spans] == pytest.approx(expected)


def test_the_scaled_numeric_error_is_invariant_to_the_column_unit() -> None:
    """The point of ``inv_scale``: a column rescaled by k scales its values by k
    and its ``inv_scale`` by 1/k, so the auxiliary error in sigma units --
    ``(expected - value) * inv_scale`` -- costs the same in either unit. Without
    it a gram-scale column would dominate the term over a millimetre-scale one.
    """
    spaced = JSONSerializer(number=SpacedDigits())
    grams = pd.DataFrame({"a": [1.5, 2.5, 4.0]})
    milligrams = pd.DataFrame({"a": grams["a"] * 1000})
    base = _fit(grams, numeric_loss_weight=0.5, permute_order=False, serializer=spaced)
    scaled = _fit(milligrams, numeric_loss_weight=0.5, permute_order=False, serializer=spaced)
    for small, big in zip(base, scaled, strict=True):
        for x, y in zip(small.numeric_spans, big.numeric_spans, strict=True):
            assert x.value * x.inv_scale == pytest.approx(y.value * y.inv_scale)


def test_a_constant_column_keeps_a_unit_scale() -> None:
    frame = pd.DataFrame({"a": [2.0, 2.0], "b": [1.0, 3.0]})
    examples = _fit(
        frame,
        numeric_loss_weight=0.5,
        permute_order=False,
        serializer=JSONSerializer(number=SpacedDigits()),
    )
    for ex in examples:
        assert ex.numeric_spans[0].inv_scale == 1.0  # std == 0: nothing to normalize by


def test_numeric_loss_weight_requires_spaced_digits() -> None:
    frame = pd.DataFrame({"a": [1.5, 2.5]})
    with pytest.raises(ValueError, match="SpacedDigits"):
        _fit(frame, numeric_loss_weight=0.5)


# --- inference batch-size invariance --------------------------------------


def _gen(batch_size: int) -> GenerationConfig:
    return GenerationConfig(inference_batch_size=batch_size)


def test_classifier_result_is_batch_size_invariant(clf_data) -> None:
    X, y = clf_data
    clf = LanguageModelClassifier(backend=FakeBackend()).fit(X, y)
    clf.generation = _gen(1)
    one = clf.predict_proba(X)
    clf.generation = _gen(1000)
    np.testing.assert_array_equal(one, clf.predict_proba(X))


def test_regressor_result_is_batch_size_invariant(reg_data) -> None:
    X, y = reg_data
    reg = LanguageModelRegressor(backend=FakeBackend(value="3.5")).fit(X, y)
    reg.generation = GenerationConfig(inference_batch_size=1, n_samples=4)
    one = reg.predict(X)
    reg.generation = GenerationConfig(inference_batch_size=1000, n_samples=4)
    np.testing.assert_array_equal(one, reg.predict(X))


def test_imputer_result_is_batch_size_invariant(nan_data) -> None:
    imp = LanguageModelImputer(backend=FakeBackend(value="0")).fit(nan_data)
    imp.generation = _gen(1)
    one = imp.transform(nan_data)
    imp.generation = _gen(1000)
    np.testing.assert_array_equal(np.asarray(one), np.asarray(imp.transform(nan_data)))


# --- multi-sample generation (sample_aggregate_many) ----------------------


def _lm(backend: FakeBackend) -> TabularLanguageModel:
    return TabularLanguageModel(
        backend=backend,
        serializer=JSONSerializer(),
        training=TrainingConfig(epochs=1, evaluation=None),
        model=ModelConfig(model="m"),
        random_state=0,
    )


class _SeqBackend(FakeBackend):
    """Cycle through ``outputs`` across generate calls, so the draws of one cell differ."""

    def __init__(self, outputs: list[str]) -> None:
        super().__init__()
        self._outputs = outputs
        self._cursor = 0

    def generate(
        self,
        prompts: Sequence[str],
        generation: object,
        *,
        constraint: object = None,
        random_state: int | None = None,
    ) -> list[str]:
        self.generate_batches.append(len(prompts))
        out: list[str] = []
        for _ in prompts:
            out.append(self._outputs[self._cursor % len(self._outputs)])
            self._cursor += 1
        return out


class _RecordingBackend(FakeBackend):
    """Record every prompt passed to ``generate`` (to observe column-order permutation)."""

    def __init__(self) -> None:
        super().__init__(value="1.0")
        self.prompts: list[str] = []

    def generate(
        self,
        prompts: Sequence[str],
        generation: object,
        *,
        constraint: object = None,
        random_state: int | None = None,
    ) -> list[str]:
        self.prompts.extend(prompts)
        return super().generate(prompts, generation, random_state=random_state)


def test_sample_aggregate_averages_numeric_draws() -> None:
    lm = _lm(_SeqBackend(["1.0", "2.0", "3.0"]))
    lm.fit(pd.DataFrame({"age": [10.0, 20.0]}))
    out = lm.sample_aggregate_many([{}], [["age"]], GenerationConfig(n_samples=3))
    assert out[0] is not None
    assert out[0]["age"] == 2.0


def test_generation_batches_are_seeded_per_row_draw_and_attempt() -> None:
    # regression: generate() drew from the backend's global stream, so a fitted model
    # with random_state set produced different texts on every call
    fake = FakeBackend(value="1.0")
    lm = _lm(fake)
    lm.fit(pd.DataFrame({"age": [10.0, 20.0]}))
    gen = GenerationConfig(n_samples=3, inference_batch_size=1)
    lm.sample_aggregate_many([{}, {}], [["age"], ["age"]], gen, row_ids=[5, 6])
    first = list(fake.generate_random_states)
    assert len(first) == 6 and len(set(first)) == 6  # one distinct seed per (row, draw)
    fake.generate_random_states.clear()
    lm.sample_aggregate_many([{}, {}], [["age"], ["age"]], gen, row_ids=[5, 6])
    assert fake.generate_random_states == first  # the same rows redraw the same seeds
    fake.generate_random_states.clear()
    lm.sample_aggregate_many([{}], [["age"]], gen, row_ids=[6])
    assert fake.generate_random_states == first[3:]  # keyed on absolute identity, not position
    fake.generate_random_states.clear()
    lm.random_state = None
    lm.sample_aggregate_many([{}], [["age"]], gen, row_ids=[6])
    assert fake.generate_random_states == [None, None, None]


def test_sample_aggregate_takes_mode_for_categorical_draws() -> None:
    lm = _lm(_SeqBackend(['"a"', '"a"', '"b"']))
    lm.fit(pd.DataFrame({"city": ["x", "y"]}))
    out = lm.sample_aggregate_many([{}], [["city"]], GenerationConfig(n_samples=3))
    assert out[0] is not None
    assert out[0]["city"] == "a"


def test_sample_aggregate_returns_none_when_all_draws_malformed() -> None:
    lm = _lm(FakeBackend(value="garbage"))
    lm.fit(pd.DataFrame({"age": [10.0, 20.0]}))
    out = lm.sample_aggregate_many([{}], [["age"]], GenerationConfig(n_samples=3))
    assert out == [None]


def test_sample_aggregate_permutes_conditioning_order_per_draw() -> None:
    fake = _RecordingBackend()
    lm = _lm(fake)
    lm.fit(pd.DataFrame({"a": [1.0], "b": [2.0], "t": [3.0]}))
    lm.sample_aggregate_many(
        [{"a": 1.0, "b": 2.0}], [["t"]], GenerationConfig(n_samples=6, permute_order=True)
    )
    assert len(set(fake.prompts)) > 1


# --- order marginalization vs the target-last training layout ---------------


def _tail_respected(prompt: str) -> bool:
    """``t`` (a target cell) serialized after both context cells ``a`` and ``b``."""
    return prompt.index('"t"') > max(prompt.index('"a"'), prompt.index('"b"'))


def test_scored_marginalization_keeps_target_cells_in_tail_block() -> None:
    # A target-last fit never showed a target cell mid-row, so the drawn orders
    # must keep the observed target cell `t` after the context block.
    fake = _ScoreRecordingBackend()
    lm = _lm(fake)
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "t": [5.0, 6.0], "u": ["x", "y"]})
    lm.fit(frame, target_cols=frozenset({"t", "u"}))
    assert lm.target_last_ and lm.target_cols_ == {"t", "u"}
    lm.predict_proba_many(
        [{"a": 1.0, "b": 3.0, "t": 5.0}],
        "u",
        ["x", "y"],
        GenerationConfig(n_samples=6, permute_order=True),
    )
    assert fake.scored_prompts and all(_tail_respected(p) for p in fake.scored_prompts)


def test_generative_marginalization_keeps_target_cells_in_tail_block() -> None:
    fake = _RecordingBackend()
    lm = _lm(fake)
    frame = pd.DataFrame({"a": [1.0], "b": [2.0], "t": [3.0], "u": [4.0]})
    lm.fit(frame, target_cols=frozenset({"t", "u"}))
    lm.sample_aggregate_many(
        [{"a": 1.0, "b": 2.0, "t": 3.0}], [["u"]], GenerationConfig(n_samples=6, permute_order=True)
    )
    assert fake.prompts and all(_tail_respected(p) for p in fake.prompts)


def test_marginalization_without_target_last_mixes_all_columns() -> None:
    # No target_cols at fit -> no block layout was trained, so the permutation
    # keeps drawing over the whole context, exactly as before.
    fake = _RecordingBackend()
    lm = _lm(fake)
    lm.fit(pd.DataFrame({"a": [1.0], "b": [2.0], "t": [3.0], "u": [4.0]}))
    assert not lm.target_last_
    lm.sample_aggregate_many(
        [{"a": 1.0, "b": 2.0, "t": 3.0}], [["u"]], GenerationConfig(n_samples=6, permute_order=True)
    )
    assert any(not _tail_respected(p) for p in fake.prompts)


# --- numeric constrained decoding ------------------------------------------


def test_constrain_numeric_sends_constraint_only_for_numeric_cells() -> None:
    fake = FakeBackend(value="7")
    lm = _lm(fake)
    lm.fit(pd.DataFrame({"age": [10.0, 20.0], "city": ["SP", "RJ"]}))
    lm.complete_many([{}], [["city", "age"]], GenerationConfig())  # on by default
    assert fake.last_constraints == [None, JSONSerializer().numeric_constraint()]


def test_constrain_numeric_off_never_sends_a_constraint() -> None:
    fake = FakeBackend(value="7")
    lm = _lm(fake)
    lm.fit(pd.DataFrame({"age": [10.0, 20.0], "city": ["SP", "RJ"]}))
    lm.complete_many([{}], [["city", "age"]], GenerationConfig(constrain_numeric=False))
    assert fake.last_constraints == [None, None]


def test_constrain_numeric_splits_mixed_batches_by_target_dtype() -> None:
    fake = FakeBackend(value="7")
    lm = _lm(fake)
    lm.fit(pd.DataFrame({"age": [10.0, 20.0], "city": ["SP", "RJ"]}))
    lm.complete_many(
        [{"age": 1.0}, {"city": "SP"}],
        [["city"], ["age"]],
        GenerationConfig(constrain_numeric=True),
    )
    assert fake.generate_batches == [1, 1]
    assert fake.last_constraints == [JSONSerializer().numeric_constraint(), None]


# --- whole-row sampling (sample) -------------------------------------------


def test_fit_records_sklearn_ceremony() -> None:
    lm = _lm(FakeBackend())
    lm.fit(pd.DataFrame({"age": [1.0, 2.0], "city": ["SP", "RJ"]}))
    assert lm.n_features_in_ == 2
    assert list(lm.feature_names_in_) == ["age", "city"]
    lm.fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert lm.n_features_in_ == 2
    assert not hasattr(lm, "feature_names_in_")
    assert lm.columns_ == ["x0", "x1"]


def test_sample_returns_n_rows_in_training_column_order() -> None:
    lm = _lm(FakeBackend(value="1.5"))
    lm.fit(pd.DataFrame({"age": [10.0, 20.0], "score": [0.1, 0.2]}))
    out = lm.sample(3)
    assert list(out.columns) == ["age", "score"]
    assert out.shape == (3, 2)
    assert out.dtypes.map(pd.api.types.is_float_dtype).all()
    assert (out == 1.5).all().all()


def test_sample_broadcasts_a_single_condition_mapping() -> None:
    lm = _lm(FakeBackend(value="1.5"))
    lm.fit(pd.DataFrame({"age": [10.0, 20.0], "city": ["SP", "RJ"]}))
    out = lm.sample(4, condition={"city": "SP"})
    assert list(out.columns) == ["age", "city"]
    assert (out["city"] == "SP").all()
    assert (out["age"] == 1.5).all()


def test_sample_condition_sequence_gives_one_row_each() -> None:
    lm = _lm(FakeBackend(value="1.5"))
    lm.fit(pd.DataFrame({"age": [10.0, 20.0], "city": ["SP", "RJ"]}))
    out = lm.sample(condition=[{"city": "SP"}, {"city": "RJ"}, {"city": "SP"}])
    assert out["city"].tolist() == ["SP", "RJ", "SP"]
    assert len(out) == 3


def test_sample_rejects_condition_on_unseen_column() -> None:
    lm = _lm(FakeBackend(value="1.5"))
    lm.fit(pd.DataFrame({"age": [10.0, 20.0]}))
    with pytest.raises(ValueError, match="not seen at fit"):
        lm.sample(2, condition={"bogus": 1})


def test_sample_raises_when_rows_stay_malformed() -> None:
    lm = _lm(FakeBackend(value="garbage"))
    lm.max_retries = 2
    lm.fit(pd.DataFrame({"age": [10.0, 20.0]}))
    with pytest.raises(RuntimeError, match="malformed"):
        lm.sample(2)


def test_sample_requires_fit() -> None:
    with pytest.raises(NotFittedError):
        _lm(FakeBackend()).sample(1)


# --- classifier order marginalization (permute_order on the score path) ----


class _ScoreRecordingBackend(FakeBackend):
    """Record every prompt passed to ``score`` (to observe column-order permutation)."""

    def __init__(self) -> None:
        super().__init__()
        self.scored_prompts: list[str] = []

    def score(
        self, prompts: Sequence[str], continuations: Sequence[str], *, reduce: str = "mean"
    ) -> list[float]:
        self.scored_prompts.extend(prompts)
        return super().score(prompts, continuations, reduce=reduce)


def test_classifier_permute_order_is_noop_when_scores_ignore_order(clf_data) -> None:
    X, y = clf_data
    clf = LanguageModelClassifier(backend=FakeBackend()).fit(X, y)
    clf.generation = GenerationConfig()
    expected = clf.predict_proba(X)
    clf.generation = GenerationConfig(permute_order=True, n_samples=4)
    np.testing.assert_allclose(clf.predict_proba(X), expected)


def test_classifier_score_pool_receives_raw_logprobs(clf_data) -> None:
    X, y = clf_data
    clf = LanguageModelClassifier(backend=FakeBackend()).fit(X, y)
    n_classes = len(clf.classes_)
    seen: list[int] = []

    def pool(logprob_rows: list[Sequence[float]]) -> list[float]:
        seen.append(len(logprob_rows))
        assert all(len(r) == n_classes for r in logprob_rows)
        return [1.0] + [0.0] * (n_classes - 1)

    clf.generation = GenerationConfig(permute_order=True, n_samples=3, score_pool=pool)
    proba = clf.predict_proba(X.head(2))
    assert seen and all(k >= 1 for k in seen)
    np.testing.assert_allclose(proba[:, 0], 1.0)


def test_classifier_permute_order_scores_multiple_column_orders(clf_data) -> None:
    X, y = clf_data
    fake = _ScoreRecordingBackend()
    clf = LanguageModelClassifier(backend=fake).fit(X, y)
    clf.generation = GenerationConfig(permute_order=True, n_samples=4)
    clf.predict_proba(X.head(1))
    assert len(set(fake.scored_prompts)) > 1


class _PromptSensitiveBackend(FakeBackend):
    """Hash the prompt into every result, so outputs expose which orders were drawn."""

    def score(
        self, prompts: Sequence[str], continuations: Sequence[str], *, reduce: str = "mean"
    ) -> list[float]:
        self.score_batches.append(len(prompts))
        return [-_stable(p + c) for p, c in zip(prompts, continuations, strict=True)]

    def generate(
        self,
        prompts: Sequence[str],
        generation: object,
        *,
        constraint: object = None,
        random_state: int | None = None,
    ) -> list[str]:
        self.generate_batches.append(len(prompts))
        return [repr(round(_stable(p) % 100, 3)) for p in prompts]


def test_classifier_order_marginalization_is_batch_size_invariant(clf_data) -> None:
    # regression: order draws are seeded per row identity, not per stream position
    X, y = clf_data
    clf = LanguageModelClassifier(backend=_PromptSensitiveBackend(), random_state=0).fit(X, y)
    clf.generation = GenerationConfig(permute_order=True, n_samples=4, inference_batch_size=1)
    one = clf.predict_proba(X)
    clf.generation = GenerationConfig(permute_order=True, n_samples=4, inference_batch_size=1000)
    np.testing.assert_array_equal(one, clf.predict_proba(X))


def test_regressor_order_marginalization_is_batch_size_invariant(reg_data) -> None:
    X, y = reg_data
    reg = LanguageModelRegressor(backend=_PromptSensitiveBackend(), random_state=0).fit(X, y)
    reg.generation = GenerationConfig(permute_order=True, n_samples=4, inference_batch_size=1)
    one = reg.predict(X)
    reg.generation = GenerationConfig(permute_order=True, n_samples=4, inference_batch_size=1000)
    np.testing.assert_array_equal(one, reg.predict(X))


def test_order_marginalized_predictions_repeat_for_fixed_seed(clf_data) -> None:
    X, y = clf_data
    clf = LanguageModelClassifier(
        backend=_PromptSensitiveBackend(),
        generation=GenerationConfig(permute_order=True, n_samples=4),
        random_state=0,
    ).fit(X, y)
    np.testing.assert_array_equal(clf.predict_proba(X), clf.predict_proba(X))


# --- numpy scalar inputs on the inference paths ----------------------------


def test_numpy_scalar_knowns_and_candidates_match_native_floats() -> None:
    # regression: repr(np.float64(1.5)) is "np.float64(1.5)" on numpy >= 2, so
    # boxed inputs silently corrupted prompts and flattened the distribution
    lm = _lm(_PromptSensitiveBackend())
    lm.fit(pd.DataFrame({"age": [10.0, 20.0], "size": [1.0, 2.0]}))
    native = lm.predict_proba({"age": 10.5}, "size", [1.0, 2.0])
    boxed = lm.predict_proba({"age": np.float64(10.5)}, "size", [np.float64(1.0), np.float64(2.0)])
    np.testing.assert_array_equal(native, boxed)


def test_complete_serializes_numpy_scalar_knowns_as_native() -> None:
    fake = _RecordingBackend()
    lm = _lm(fake)
    lm.fit(pd.DataFrame({"age": [10.0, 20.0], "size": [1.0, 2.0]}))
    lm.complete({"age": np.float64(10.5)}, ["size"], GenerationConfig())
    assert fake.prompts == ['{"age": 10.5, "size": ']


# --- permute_order semantics ------------------------------------------------


def test_regressor_single_sample_permute_order_is_inert() -> None:
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    y = np.array([1.0, 2.0])
    fake = _RecordingBackend()
    reg = LanguageModelRegressor(
        backend=fake, generation=GenerationConfig(permute_order=True, n_samples=1)
    ).fit(X, y)
    reg.predict(X)
    first = list(fake.prompts)
    reg.predict(X)
    assert fake.prompts == first * 2  # no order draw -> deterministic with random_state=None
    assert all(p.index('"a"') < p.index('"b"') for p in first)  # canonical feature order


def test_imputer_single_sample_permute_order_is_inert() -> None:
    X = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [3.0, 4.0, 5.0], "c": ["x", "y", "z"]})
    fake = _RecordingBackend()
    imp = LanguageModelImputer(
        backend=fake, generation=GenerationConfig(permute_order=True, n_samples=1)
    ).fit(X)
    imp.transform(X)
    first = list(fake.prompts)
    imp.transform(X)
    assert fake.prompts == first * 2  # no order draw -> deterministic with random_state=None
    assert all(p.index('"b"') < p.index('"c"') for p in first)  # canonical feature order


def test_classifier_never_exceeds_configured_batch(clf_data) -> None:
    X, y = clf_data
    fake = FakeBackend()
    clf = LanguageModelClassifier(backend=fake, generation=_gen(8)).fit(X, y)
    clf.predict_proba(X)
    n_classes = len(clf.classes_)
    assert sum(fake.score_batches) == len(X) * n_classes
    # A batch holds up to 8 rows, each with its whole candidate set: the
    # backend forwards a row's prompt once for all of its classes.
    assert max(fake.score_batches) <= 8 * n_classes
    assert all(b % n_classes == 0 for b in fake.score_batches)


# --- confidence-ordered imputation (impute_many) ----------------------------


class _ContextSensitiveBackend(FakeBackend):
    """``sure`` is peaked at "y" regardless of context; ``unsure`` prefers "p" only
    once the prompt carries the sibling's "y" fill, and leans "q" (barely) when
    blind."""

    def score(self, prompts, continuations, *, reduce="mean"):
        self.score_batches.append(len(prompts))
        out = []
        for prompt, cont in zip(prompts, continuations, strict=True):
            if '"x"' in cont or '"y"' in cont:
                out.append(5.0 if '"y"' in cont else -5.0)
            elif '"y"' in prompt:
                out.append(5.0 if '"p"' in cont else -5.0)
            else:
                out.append(0.1 if '"q"' in cont else -0.1)
        return out


def _argmax_candidate(proba: np.ndarray, candidates: Sequence[object]) -> object:
    return candidates[int(np.argmax(proba))]


def test_impute_many_confidence_order_fills_the_most_confident_cell_first() -> None:
    """cell_order="confidence" commits the low-entropy `sure` cell first even though
    `unsure` comes first in column order, so `unsure` is then scored with the fill in
    its prompt and flips from "q" to "p"."""
    lm = _lm(_ContextSensitiveBackend())
    lm.fit(pd.DataFrame({"unsure": ["p", "q"], "sure": ["x", "y"], "anchor": [1.0, 2.0]}))
    score = {
        "unsure": _ScoreSpec(["p", "q"], _argmax_candidate),
        "sure": _ScoreSpec(["x", "y"], _argmax_candidate),
    }

    def fill(generation: GenerationConfig) -> dict[str, object]:
        out = lm.impute_many([{"anchor": 1.0}], [["unsure", "sure"]], generation, score=score)
        assert out[0] is not None
        return out[0]

    column = fill(GenerationConfig())
    confident = fill(GenerationConfig(cell_order="confidence"))
    assert column["sure"] == confident["sure"] == "y"
    assert column["unsure"] == "q"
    assert confident["unsure"] == "p"


# --- prior correction (PMI) -------------------------------------------------


class _PriorSkewedBackend(FakeBackend):
    """The empty-context prior prefers "a" overwhelmingly; conditioned on a row,
    "a" still wins but mildly. Ranking by belief shift must flip to "b"."""

    def score(self, prompts, continuations, *, reduce="mean"):
        self.score_batches.append(len(prompts))
        out = []
        for prompt, cont in zip(prompts, continuations, strict=True):
            conditioned = "anchor" in prompt
            if '"a"' in cont:
                out.append(2.0 if conditioned else 5.0)
            else:
                out.append(1.0 if conditioned else 0.0)
        return out


def test_prior_correction_ranks_by_belief_shift_and_caches_the_prior() -> None:
    fake = _PriorSkewedBackend()
    lm = _lm(fake)
    lm.fit(pd.DataFrame({"cat": ["a", "b"], "anchor": [1.0, 2.0]}))

    raw = lm.predict_proba_many([{"anchor": 1.0}], "cat", ["a", "b"], GenerationConfig())
    assert raw[0].argmax() == 0  # conditional likelihood alone still prefers "a"

    pmi = GenerationConfig(prior_correction=1.0)
    corrected = lm.predict_proba_many([{"anchor": 1.0}], "cat", ["a", "b"], pmi)
    assert corrected[0].argmax() == 1  # the row shifts belief toward "b"
    np.testing.assert_allclose(corrected[0].sum(), 1.0)

    calls = len(fake.score_batches)
    lm.predict_proba_many([{"anchor": 1.0}], "cat", ["a", "b"], pmi)
    assert len(fake.score_batches) == calls + 1  # conditional only: the prior was cached


# --- learned column order (GenerationConfig.column_order / infer_optimal_order) ----


def _prompt_keys(prompt: str) -> list[str]:
    return [part.split('"')[1] for part in prompt.split(", ") if part.lstrip("{").startswith('"')]


def test_column_order_serializes_context_in_the_learned_order() -> None:
    fake = _ScoreRecordingBackend()
    lm = _lm(fake)
    lm.fit(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0], "t": ["p", "q"]}))
    learned = GenerationConfig(column_order=["t", "c", "a", "b"])

    lm.predict_proba_many([{"a": 1.0, "b": 3.0, "c": 5.0}], "t", ["p", "q"], learned)
    assert _prompt_keys(fake.scored_prompts[-1]) == ["c", "a", "b", "t"]  # the target skipped

    fake.scored_prompts.clear()
    permuted = GenerationConfig(column_order=["t", "c", "a", "b"], permute_order=True, n_samples=4)
    lm.predict_proba_many([{"a": 1.0, "b": 3.0, "c": 5.0}], "t", ["p", "q"], permuted)
    assert len(set(fake.scored_prompts)) == 1  # the learned order replaces the permutation

    partial = GenerationConfig(column_order=["c", "a"])
    with pytest.raises(ValueError, match="permutation of the training columns"):
        lm.predict_proba_many([{"a": 1.0, "b": 3.0, "c": 5.0}], "t", ["p", "q"], partial)


class _BLastBackend(FakeBackend):
    """Predict ``t`` from the sign of ``b`` -- but only when ``b`` is written right before ``t``."""

    def score(
        self, prompts: Sequence[str], continuations: Sequence[str], *, reduce: str = "mean"
    ) -> list[float]:
        out: list[float] = []
        for prompt, cont in zip(prompts, continuations, strict=True):
            keys = _prompt_keys(prompt)
            if keys[-2:] != ["b", "t"]:
                out.append(-1.0)
                continue
            b = float(prompt.split('"b": ')[1].split(",")[0])
            truth = '"p"' if b > 0 else '"q"'
            out.append(0.0 if cont == truth else -5.0)
        return out


def test_infer_optimal_order_puts_the_informative_column_last() -> None:
    rng = np.random.default_rng(0)
    b = rng.normal(size=40)
    frame = pd.DataFrame({
        "a": rng.normal(size=40),
        "b": b,
        "c": rng.normal(size=40),
        "d": rng.normal(size=40),
        "t": np.where(b > 0, "p", "q"),
    })
    lm = _lm(_BLastBackend()).fit(frame)
    order = lm.infer_optimal_order(frame, targets=["t"], n_rows=16, n_orders=12)
    assert sorted(order) == ["a", "b", "c", "d", "t"]
    assert order[-1] == "t"  # never conditioned on: last, after the context
    assert order[-2] == "b"  # the informative column ends the context, adjacent to the target


def test_infer_optimal_order_halves_the_orders_between_row_batches() -> None:
    class Counting(_BLastBackend):
        def __init__(self) -> None:
            super().__init__()
            self.scored = 0

        def score(
            self, prompts: Sequence[str], continuations: Sequence[str], *, reduce: str = "mean"
        ) -> list[float]:
            self.scored += len(prompts)
            return super().score(prompts, continuations, reduce=reduce)

    rng = np.random.default_rng(0)
    b = rng.normal(size=40)
    frame = pd.DataFrame({
        "a": rng.normal(size=40),
        "b": b,
        "c": rng.normal(size=40),
        "d": rng.normal(size=40),
        "t": np.where(b > 0, "p", "q"),
    })
    backend = Counting()
    order = (
        _lm(backend).fit(frame).infer_optimal_order(frame, targets=["t"], n_rows=16, n_orders=12)
    )
    # 16 rows x 12 orders x 2 candidates = 384 pairs at full cost; successive halving
    # (4 rows x 12, 4 x 6, 8 x 3 orders) caps it at 192, and the answer is unchanged
    assert backend.scored <= 192
    assert order[-2] == "b"


def test_positional_order_discounts_the_rows_each_order_saw() -> None:
    """An order cut early saw only the first rows; when those happen to be the
    easy ones its raw mean beats a survivor that went on to the hard rows. The
    row effects in the position model remove that, leaving the positional signal."""
    orders = [list(p) for p in permutations(["x", "y", "t"])]
    easy, hard = [0.0] * 4, [-2.0] * 12
    # t's position is worth +0.1 per slot; the two t-last orders survived the cut
    # and went on to the hard rows, the rest only ever saw the easy ones
    hits = [
        [v + 0.1 * o.index("t") for v in (easy + hard if o[-1] == "t" else easy)] for o in orders
    ]
    assert positional_order(orders, hits)[-1] == "t"


def test_infer_optimal_order_defaults_to_missing_columns_and_is_deterministic() -> None:
    frame = pd.DataFrame({
        "a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
        "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "c": ["x", "y", "x", "y", "x", "y"],
    })
    lm = _lm(_PromptSensitiveBackend()).fit(frame)
    first = lm.infer_optimal_order(frame, n_rows=4, n_orders=2)
    assert sorted(first) == ["a", "b", "c"]
    assert lm.infer_optimal_order(frame, n_rows=4, n_orders=2) == first

    imputer = LanguageModelImputer(backend=FakeBackend()).fit(frame)
    filled = imputer.transform(frame, generation=GenerationConfig(column_order=first))
    assert not pd.DataFrame(filled).isna().to_numpy().any()


def test_infer_optimal_order_hides_the_columns_missing_alongside_the_target() -> None:
    # every row missing `t` is also missing `b`: calibration must never condition on `b`
    frame = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan, np.nan],
        "c": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "t": ["p", "q", "p", "q", "p", "q", None, None],
    })
    fake = _ScoreRecordingBackend()
    lm = _lm(fake).fit(frame)
    order = lm.infer_optimal_order(frame, targets=["t"], n_rows=4, n_orders=6)
    assert sorted(order) == ["a", "b", "c", "t"]
    # never conditioned on -> last, mirroring training's ctx-then-targets layout;
    # the calibrated target ahead of the column it would fill from
    assert order[-2:] == ["t", "b"]
    assert fake.scored_prompts and all("b" not in _prompt_keys(p) for p in fake.scored_prompts)
