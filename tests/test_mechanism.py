"""The shared mechanism, observable only through the (fake) backend.

Two guarantees the public output never exposes: per-epoch column-order
re-permutation (``augmentation_factor``) and ``loss_on_target_only`` masking.
Plus inference batch-size invariance: chunking must never change a result.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from sklm import (
    GenerationConfig,
    JSONSerializer,
    LanguageModelClassifier,
    LanguageModelImputer,
    LanguageModelRegressor,
    ModelConfig,
    TabularLanguageModel,
    TrainingConfig,
)

from .conftest import FakeBackend, _stable


def _fit(
    frame: pd.DataFrame,
    *,
    target_cols: set[str] | None = None,
    augmentation_factor: int = 1,
    loss_on_target_only: bool = True,
    target_at_end: bool = False,
    random_state: int = 0,
) -> list:
    fake = FakeBackend()
    lm = TabularLanguageModel(
        backend=fake,
        serializer=JSONSerializer(),
        training=TrainingConfig(
            augmentation_factor=augmentation_factor,
            epochs=1,
            loss_on_target_only=loss_on_target_only,
            target_at_end=target_at_end,
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
    assert len(_fit(frame, augmentation_factor=4, loss_on_target_only=False)) == 4 + 2 + 2


def test_permutation_is_reproducible_across_fits() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [4.0, 5.0, 6.0]})
    first = _fit(frame, augmentation_factor=3, random_state=42, loss_on_target_only=False)
    second = _fit(frame, augmentation_factor=3, random_state=42, loss_on_target_only=False)
    assert [e.text for e in first] == [e.text for e in second]


def test_invalid_factor_raises() -> None:
    with pytest.raises(ValueError, match="augmentation_factor"):
        TrainingConfig(augmentation_factor=0)


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


# --- loss_on_target_only masking -----------------------------------------


def test_flag_off_leaves_every_prompt_empty() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "t": [0, 1, 0]})
    assert all(ex.prompt == "" for ex in _fit(frame, target_cols={"t"}, loss_on_target_only=False))


def test_masking_splits_context_and_target_last() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "t": [0, 1, 0]})
    examples = _fit(frame, target_cols={"t"})
    assert len(examples) == len(frame)
    for ex in examples:
        assert ex.prompt != ""
        assert ex.prompt + ex.completion == ex.text
        assert list(json.loads(ex.text).keys())[-1] == "t"  # target serialized last


def test_target_at_end_fixes_position_without_masking() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "t": [0, 1, 0]})
    examples = _fit(frame, target_cols={"t"}, loss_on_target_only=False, target_at_end=True)
    assert len(examples) == len(frame)
    for ex in examples:
        assert ex.prompt == ""  # loss stays on the whole row
        assert list(json.loads(ex.text).keys())[-1] == "t"  # but the target is fixed last


def test_masking_skips_rows_with_missing_target() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "t": [0.0, np.nan, 4.0]})
    prompts = [ex.prompt for ex in _fit(frame, target_cols={"t"})]
    # rows 0 and 2 observe t -> masked; row 1 has t=NaN -> context-only
    assert prompts.count("") == 1
    assert sum(p != "" for p in prompts) == 2


def test_masking_all_columns_target_yields_brace_prompt() -> None:
    frame = pd.DataFrame({"t1": [0, 1], "t2": ["x", "y"]})
    assert all(ex.prompt == "{" for ex in _fit(frame, target_cols={"t1", "t2"}))


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
        training=TrainingConfig(epochs=1),
        model=ModelConfig(model="m"),
        random_state=0,
    )


class _SeqBackend(FakeBackend):
    """Cycle through ``outputs`` across generate calls, so the draws of one cell differ."""

    def __init__(self, outputs: list[str]) -> None:
        super().__init__()
        self._outputs = outputs
        self._cursor = 0

    def generate(self, prompts: Sequence[str], generation: object) -> list[str]:
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

    def generate(self, prompts: Sequence[str], generation: object) -> list[str]:
        self.prompts.extend(prompts)
        return super().generate(prompts, generation)


def test_sample_aggregate_averages_numeric_draws() -> None:
    lm = _lm(_SeqBackend(["1.0", "2.0", "3.0"]))
    lm.fit(pd.DataFrame({"age": [10.0, 20.0]}))
    out = lm.sample_aggregate_many([{}], [["age"]], GenerationConfig(n_samples=3))
    assert out[0] is not None
    assert out[0]["age"] == 2.0


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

    def generate(self, prompts: Sequence[str], generation: object) -> list[str]:
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
    assert sum(fake.score_batches) == len(X) * len(clf.classes_)
    assert max(fake.score_batches) <= 8
