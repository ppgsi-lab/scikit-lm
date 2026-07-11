"""Discretized numeric prediction: candidate selection, reduction, regressor branch.

``select_candidates`` / ``reduce_estimate`` are pure helpers checked in isolation;
the regressor's scoring branch and its imputer counterpart (deferred) are checked
against the deterministic fake backend, whose ``scores`` dict fixes candidate
log-likelihoods so exact predictions can be asserted with no real model.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest
from sklearn import clone

from sklm import (
    DiscretizationConfig,
    GenerationConfig,
    LanguageModelImputer,
    LanguageModelOverSampler,
    LanguageModelRegressor,
    TabularLanguageModel,
)
from sklm.base import reduce_estimate, select_candidates

from .conftest import FakeBackend

# --- select_candidates ----------------------------------------------------


def test_select_candidates_fraction_rounds_up() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cands = select_candidates(values, DiscretizationConfig(bins=0.5))  # ceil(2.5) == 3
    assert len(cands) == 3
    assert cands == sorted(cands)
    assert set(cands) <= set(values)  # representatives are observed values


def test_select_candidates_all_via_float_one_returns_sorted_unique() -> None:
    values = np.array([3.0, 1.0, 2.0, 1.0, 3.0])
    # representative is inert on the "all" shortcut: a synthetic one must not change the result
    cands = select_candidates(values, DiscretizationConfig(bins=1.0, representative="mean"))
    assert cands == [1.0, 2.0, 3.0]


def test_select_candidates_int_at_or_above_n_unique_returns_unique() -> None:
    values = np.array([10.0, 20.0, 30.0, 10.0])
    assert select_candidates(values, DiscretizationConfig(bins=99)) == [10.0, 20.0, 30.0]


def test_select_candidates_single_unique() -> None:
    assert select_candidates(np.array([7.0, 7.0, 7.0]), DiscretizationConfig(bins=3)) == [7.0]


def test_select_candidates_drops_nan() -> None:
    clean = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dirty = np.array([1.0, 2.0, np.nan, 3.0, 4.0, np.nan, 5.0])
    cfg = DiscretizationConfig(bins=3)
    assert select_candidates(dirty, cfg) == select_candidates(clean, cfg)


def test_select_candidates_empty_after_nan_drop_raises() -> None:
    with pytest.raises(ValueError, match="no observed"):
        select_candidates(np.array([np.nan, np.nan]), DiscretizationConfig(bins=2))


def test_select_candidates_quantile_and_uniform_differ_on_skew() -> None:
    values = np.array([1.0, 2.0, 3.0, 100.0, 200.0])
    quant = select_candidates(values, DiscretizationConfig(bins=3, strategy="quantile"))
    unif = select_candidates(values, DiscretizationConfig(bins=3, strategy="uniform"))
    assert quant != unif


def test_select_candidates_mode_breaks_ties_by_smaller_value() -> None:
    # bins=1 -> single stratum over all values; mode of {1,1,2,2} ties -> smaller wins
    cands = select_candidates(
        np.array([1.0, 1.0, 2.0, 2.0]), DiscretizationConfig(bins=1, representative="mode")
    )
    assert cands == [1.0]


def test_select_candidates_median_is_lower_median() -> None:
    # bins=1 -> single stratum {2,4}; lower median is 2.0 (an observed value), not 3.0
    cands = select_candidates(
        np.array([2.0, 4.0]), DiscretizationConfig(bins=1, representative="median")
    )
    assert cands == [2.0]


# --- reduce_estimate ------------------------------------------------------


def test_reduce_estimate_mean_is_weighted_expectation() -> None:
    proba = np.array([0.5, 0.5])
    assert reduce_estimate(proba, [10.0, 20.0], DiscretizationConfig(estimate="mean")) == 15.0


def test_reduce_estimate_mode_is_argmax_candidate() -> None:
    proba = np.array([0.2, 0.8])
    assert reduce_estimate(proba, [10.0, 20.0], DiscretizationConfig(estimate="mode")) == 20.0


def test_reduce_estimate_sharpness_tempers_the_mean() -> None:
    proba = np.array([0.25, 0.75])
    sharpened = reduce_estimate(
        proba, [10.0, 20.0], DiscretizationConfig(estimate="mean", sharpness=2.0)
    )
    # p**2 renormalized = [0.1, 0.9] -> 10*0.1 + 20*0.9
    assert sharpened == pytest.approx(19.0)


def test_reduce_estimate_sharpness_one_keeps_the_plain_mean() -> None:
    proba = np.array([0.25, 0.75])
    plain = reduce_estimate(proba, [10.0, 20.0], DiscretizationConfig(estimate="mean"))
    assert plain == pytest.approx(17.5)


def test_reduce_estimate_high_sharpness_converges_to_mode() -> None:
    proba = np.array([0.4, 0.6])
    cfg = DiscretizationConfig(estimate="mean", sharpness=200.0)
    assert reduce_estimate(proba, [10.0, 20.0], cfg) == pytest.approx(20.0, abs=1e-6)


def test_reduce_estimate_mode_is_invariant_to_sharpness() -> None:
    proba = np.array([0.2, 0.8])
    cfg = DiscretizationConfig(estimate="mode", sharpness=8.0)
    assert reduce_estimate(proba, [10.0, 20.0], cfg) == 20.0


def test_reduce_estimate_median_is_probability_weighted_median() -> None:
    # mass: 0.4 at 10, then crosses half at 20; argmax (mode) would pick 10 instead
    proba = np.array([0.4, 0.35, 0.25])
    cfg = DiscretizationConfig(estimate="median")
    assert reduce_estimate(proba, [10.0, 20.0, 30.0], cfg) == 20.0
    assert reduce_estimate(proba, [10.0, 20.0, 30.0], DiscretizationConfig(estimate="mode")) == 10.0


def test_reduce_estimate_median_is_lower_median_on_even_split() -> None:
    # cumulative mass reaches exactly half at the first candidate -> the smaller value
    proba = np.array([0.5, 0.5])
    cfg = DiscretizationConfig(estimate="median")
    assert reduce_estimate(proba, [10.0, 20.0], cfg) == 10.0


def test_reduce_estimate_median_sorts_unsorted_candidates() -> None:
    # candidates need not arrive sorted; the median orders by value internally
    proba = np.array([0.3, 0.2, 0.5])
    cfg = DiscretizationConfig(estimate="median")
    assert reduce_estimate(proba, [30.0, 10.0, 20.0], cfg) == 20.0


def test_reduce_estimate_median_ignores_distant_outlier() -> None:
    # three near-equal candidates hold most of the mass; a lone distant candidate
    # drags the mean far off but cannot move the median off the cluster
    candidates = [0.3, 7.392, 7.4, 7.403, 8.0]
    proba = np.array([0.2, 0.3, 0.3, 0.3, 0.3]) / 1.4
    median = reduce_estimate(proba, candidates, DiscretizationConfig(estimate="median"))
    mean = reduce_estimate(proba, candidates, DiscretizationConfig(estimate="mean"))
    assert median == 7.4
    assert mean == pytest.approx(6.513, abs=1e-3)


def test_reduce_estimate_median_high_sharpness_converges_to_mode() -> None:
    proba = np.array([0.3, 0.4, 0.3])
    cfg = DiscretizationConfig(estimate="median", sharpness=200.0)
    # mass collapses onto the argmax candidate -> the median crosses there too
    assert reduce_estimate(proba, [10.0, 20.0, 30.0], cfg) == pytest.approx(20.0)


def test_discretization_rejects_non_positive_sharpness() -> None:
    with pytest.raises(ValueError, match="sharpness"):
        DiscretizationConfig(sharpness=0.0)


# --- regressor scoring branch ---------------------------------------------


def _two_class_target() -> tuple[pd.DataFrame, np.ndarray]:
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": ["x", "y", "x", "y"]})
    y = np.array([10.0, 20.0, 10.0, 20.0])  # candidates encode to "10.0" / "20.0"
    return X, y


def test_regressor_discretized_mean_weights_candidates() -> None:
    X, y = _two_class_target()
    reg = LanguageModelRegressor(
        backend=FakeBackend(scores={"10.0": 0.0, "20.0": 0.0}),  # equal logprobs -> 0.5/0.5
        discretization=DiscretizationConfig(bins=1.0, estimate="mean"),
    ).fit(X, y)
    np.testing.assert_allclose(reg.predict(X), 15.0)


def test_regressor_discretized_mode_picks_most_likely_candidate() -> None:
    X, y = _two_class_target()
    reg = LanguageModelRegressor(
        backend=FakeBackend(scores={"10.0": -5.0, "20.0": 0.0}),  # mass on 20
        discretization=DiscretizationConfig(bins=1.0, estimate="mode"),
    ).fit(X, y)
    np.testing.assert_allclose(reg.predict(X), 20.0)


def test_regressor_bins_routes_scoring_vs_generation() -> None:
    X, y = _two_class_target()

    off = FakeBackend(value="3.5")
    LanguageModelRegressor(backend=off).fit(X, y).predict(X)  # bins=0 default
    assert off.generate_batches and not off.score_batches

    on = FakeBackend(value="3.5")
    LanguageModelRegressor(backend=on, discretization=DiscretizationConfig(bins=2)).fit(
        X, y
    ).predict(X)
    assert on.score_batches and not on.generate_batches


def test_regressor_discretized_is_batch_size_invariant(reg_data) -> None:
    X, y = reg_data
    reg = LanguageModelRegressor(
        backend=FakeBackend(), discretization=DiscretizationConfig(bins=4)
    ).fit(X, y)
    reg.generation = GenerationConfig(inference_batch_size=1)
    one = reg.predict(X)
    reg.generation = GenerationConfig(inference_batch_size=1000)
    np.testing.assert_array_equal(one, reg.predict(X))


def test_regressor_discretized_scores_every_candidate_within_batch_cap(reg_data) -> None:
    X, y = reg_data
    fake = FakeBackend()
    reg = LanguageModelRegressor(
        backend=fake,
        discretization=DiscretizationConfig(bins=4),
        generation=GenerationConfig(inference_batch_size=8),
    ).fit(X, y)
    reg.predict(X)
    candidates = select_candidates(reg.target_values_, reg.discretization)
    assert sum(fake.score_batches) == len(X) * len(candidates)
    assert max(fake.score_batches) <= 8


def test_regressor_set_params_addresses_nested_discretization() -> None:
    reg = LanguageModelRegressor(backend=FakeBackend())
    reg.set_params(discretization__bins=5)
    assert reg.discretization.bins == 5


# --- imputer: cell-by-cell score/generate ---------------------------------


def _mixed_frame() -> pd.DataFrame:
    """Numeric + categorical columns, last row missing both."""
    return pd.DataFrame({"n": [10.0, 20.0, 10.0, 20.0, np.nan], "c": ["x", "y", "x", "y", np.nan]})


def test_imputer_scores_numeric_and_categorical() -> None:
    # numeric scored to its mean; categorical scored to the argmax over its levels
    out = LanguageModelImputer(
        backend=FakeBackend(scores={"10.0": 0.0, "20.0": 0.0, '"x"': 0.0, '"y"': -1.0}),
        discretization=DiscretizationConfig(bins=1.0, estimate="mean"),
    ).fit_transform(_mixed_frame())
    assert isinstance(out, pd.DataFrame)
    assert out.loc[4, "n"] == 15.0  # numeric scored: mean(10, 20)
    assert out.loc[4, "c"] == "x"  # categorical scored: argmax over {x, y}


def test_imputer_routes_numeric_and_categorical() -> None:
    X = _mixed_frame()

    # discretization on: numeric `n` and categorical `c` both scored, nothing generated
    on = FakeBackend()
    LanguageModelImputer(backend=on, discretization=DiscretizationConfig(bins=2)).fit_transform(X)
    assert on.score_batches and not on.generate_batches

    # default: numeric `n` generated, categorical `c` still scored over its levels
    off = FakeBackend(value="0")
    LanguageModelImputer(backend=off).fit_transform(X)
    assert off.generate_batches and off.score_batches


def test_imputer_discretized_single_config_scores_all_numeric() -> None:
    X = pd.DataFrame({"a": [1.0, 2.0, 1.0, 2.0, np.nan], "b": [10.0, 20.0, 10.0, 20.0, np.nan]})
    fake = FakeBackend(scores={"1.0": 0.0, "2.0": 0.0, "10.0": 0.0, "20.0": 0.0})
    out = LanguageModelImputer(
        backend=fake, discretization=DiscretizationConfig(bins=1.0, estimate="mean")
    ).fit_transform(X)
    assert isinstance(out, pd.DataFrame)
    assert out.loc[4, "a"] == 1.5
    assert out.loc[4, "b"] == 15.0
    assert not fake.generate_batches  # no categorical target -> nothing generated


def test_imputer_discretized_map_scores_only_listed_column() -> None:
    X = pd.DataFrame({"a": [1.0, 2.0, 1.0, 2.0, np.nan], "b": [10.0, 20.0, 10.0, 20.0, np.nan]})
    fake = FakeBackend(value="0", scores={"1.0": 0.0, "2.0": 0.0})
    LanguageModelImputer(
        backend=fake, discretization={"a": DiscretizationConfig(bins=1.0)}
    ).fit_transform(X)
    # only "a" is scored; "b" (absent from the map) is generated
    assert fake.score_batches and fake.generate_batches


def test_imputer_discretization_map_unknown_column_raises() -> None:
    X = pd.DataFrame({"a": [1.0, 2.0, np.nan], "c": ["x", "y", "x"]})
    imp = LanguageModelImputer(
        backend=FakeBackend(), discretization={"missing": DiscretizationConfig(bins=2)}
    )
    with pytest.raises(ValueError, match="unknown column"):
        imp.fit(X)


def test_imputer_discretization_map_disables_categorical_scoring() -> None:
    # a categorical column is scored by default; a bins=0 entry routes it back to
    # generation (the only way `c` reaches the generate path is if scoring is off)
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "c": ["x", "y", np.nan]})
    fake = FakeBackend(value='"z"')
    out = LanguageModelImputer(
        backend=fake, discretization={"c": DiscretizationConfig(bins=0)}
    ).fit_transform(X)
    assert isinstance(out, pd.DataFrame)
    assert out.loc[2, "c"] == "z"  # generated, not scored
    assert fake.generate_batches and not fake.score_batches


def test_imputer_discretized_is_batch_size_invariant() -> None:
    X = pd.DataFrame({
        "n": [10.0, 20.0, 30.0, 40.0, np.nan, np.nan, np.nan, np.nan],
        "c": ["x", "y", "x", "y", np.nan, np.nan, np.nan, np.nan],
    })
    fake = FakeBackend()  # stable-hash scores: both `n` and categorical `c` are deterministic
    imp = LanguageModelImputer(backend=fake, discretization=DiscretizationConfig(bins=4)).fit(X)
    imp.generation = GenerationConfig(inference_batch_size=1)
    one = imp.transform(X)
    imp.generation = GenerationConfig(inference_batch_size=1000)
    big = imp.transform(X)
    assert isinstance(one, pd.DataFrame) and isinstance(big, pd.DataFrame)
    pd.testing.assert_frame_equal(one, big)


def test_imputer_discretized_scores_within_batch_cap() -> None:
    X = pd.DataFrame({
        "n": [10.0, 20.0, 30.0, 40.0, np.nan, np.nan, np.nan, np.nan],
        "c": ["x", "y", "x", "y", "x", "y", "x", "y"],  # fully observed -> only "n" imputed
    })
    fake = FakeBackend(scores={"10.0": 0.0, "20.0": 0.0, "30.0": 0.0, "40.0": 0.0})
    LanguageModelImputer(
        backend=fake,
        discretization=DiscretizationConfig(bins=4),
        generation=GenerationConfig(inference_batch_size=8),
    ).fit_transform(X)
    assert sum(fake.score_batches) == 4 * 4  # 4 missing rows x 4 candidates
    assert max(fake.score_batches) <= 8
    assert not fake.generate_batches  # "c" fully observed


def test_imputer_generation_failure_raises_runtime_error() -> None:
    X = _mixed_frame()  # `n` numeric (generated by default), `c` categorical (scored)
    imp = LanguageModelImputer(backend=FakeBackend(value="garbage")).fit(X)
    with pytest.raises(RuntimeError, match="generation failed"):
        imp.transform(X)  # the generated numeric cell never decodes -> loud failure


def test_imputer_discretization_is_a_tunable_param() -> None:
    # convention 1: clone / set_params / GridSearch must see `discretization`
    cfg_imp = LanguageModelImputer(
        backend=FakeBackend(), discretization=DiscretizationConfig(bins=3)
    )
    clone(cfg_imp)  # round-trips without raising
    assert isinstance(cfg_imp.get_params()["discretization"], DiscretizationConfig)

    map_imp = LanguageModelImputer(
        backend=FakeBackend(), discretization={"x": DiscretizationConfig(bins=2)}
    )
    clone(map_imp)
    carried = map_imp.get_params()["discretization"]
    assert isinstance(carried, Mapping) and carried["x"].bins == 2


def test_imputer_set_params_addresses_nested_discretization() -> None:
    imp = LanguageModelImputer(backend=FakeBackend())  # default DiscretizationConfig()
    imp.set_params(discretization__bins=5)
    assert isinstance(imp.discretization, DiscretizationConfig)
    assert imp.discretization.bins == 5


# --- categorical synthesis: scored-and-sampled, not generated ----------------


def test_core_sample_draws_categorical_from_levels() -> None:
    # whole-row synthesis samples a categorical from its scored levels (never the
    # generated fake value "0"); the numeric column still generates.
    frame = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0], "cat": ["a", "b", "a", "b"]})
    lm = TabularLanguageModel(backend=FakeBackend(value="0"), random_state=0).fit(frame)
    rows = lm.sample(8)
    assert len(rows) == 8
    assert set(rows["cat"]) <= {"a", "b"}


def test_oversampler_samples_categorical_from_levels() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"num": rng.normal(size=20), "cat": ["a"] * 10 + ["b"] * 10})
    y = np.array([0] * 16 + [1] * 4)
    result = LanguageModelOverSampler(backend=FakeBackend(value="0"), random_state=0).fit_resample(
        X, y
    )
    # 12 synthetic minority rows are appended after the 20 originals
    synth_cat = result[0].iloc[len(X) :]["cat"]
    assert len(synth_cat) == 12
    assert set(synth_cat) <= {"a", "b"}  # drawn from levels, never the generated "0"
