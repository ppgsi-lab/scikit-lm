"""Per-call decoder overrides on ``predict`` / ``predict_proba`` / ``transform``.

Each estimator with a standalone inference method accepts keyword-only config
overrides (``discretization`` / ``generation``) that fall back to the fitted
attribute when ``None`` and never mutate the estimator. Checked against the
deterministic fake backend, whose ``generate_batches`` / ``score_batches`` reveal
which decoder ran.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklm import (
    DiscretizationConfig,
    GenerationConfig,
    LanguageModelClassifier,
    LanguageModelImputer,
    LanguageModelRegressor,
)

from .conftest import FakeBackend


def _two_class_target() -> tuple[pd.DataFrame, np.ndarray]:
    X = pd.DataFrame({"f": [1.0, 2.0, 1.0, 2.0]})
    y = np.array([10.0, 20.0, 10.0, 20.0])  # candidates encode to "10.0" / "20.0"
    return X, y


def _numeric_frame() -> pd.DataFrame:
    return pd.DataFrame({"a": [1.0, 2.0, 1.0, 2.0, np.nan], "b": [10.0, 20.0, 10.0, 20.0, np.nan]})


def _mixed_frame() -> pd.DataFrame:
    return pd.DataFrame({"n": [10.0, 20.0, 10.0, 20.0, np.nan], "c": ["x", "y", "x", "y", np.nan]})


# --- regressor -------------------------------------------------------------


def test_regressor_predict_discretization_override_routes_scoring() -> None:
    X, y = _two_class_target()
    fake = FakeBackend(value="3.5", scores={"10.0": 0.0, "20.0": 0.0})
    reg = LanguageModelRegressor(backend=fake).fit(X, y)  # default: generative

    reg.predict(X, discretization=DiscretizationConfig(bins=2))
    assert fake.score_batches and not fake.generate_batches  # override routed to scoring
    assert reg.discretization.bins == 0  # estimator attribute untouched


def test_regressor_predict_generation_override_controls_n_samples() -> None:
    X, y = _two_class_target()
    fake = FakeBackend(value="3.5")
    reg = LanguageModelRegressor(backend=fake).fit(X, y)

    reg.predict(X, generation=GenerationConfig(n_samples=3))
    assert sum(fake.generate_batches) == len(X) * 3  # three draws per row
    assert reg.generation.n_samples == 1  # estimator attribute untouched


def test_regressor_predict_override_equals_set_params() -> None:
    X, y = _two_class_target()
    reg = LanguageModelRegressor(backend=FakeBackend(scores={"10.0": -1.0, "20.0": 0.0})).fit(X, y)

    via_override = reg.predict(X, discretization=DiscretizationConfig(bins=1.0))
    assert reg.discretization.bins == 0  # not mutated by the override
    reg.set_params(discretization=DiscretizationConfig(bins=1.0))
    np.testing.assert_array_equal(via_override, reg.predict(X))


# --- classifier ------------------------------------------------------------


def test_classifier_predict_proba_generation_override_used_and_preserved(clf_data) -> None:
    X, y = clf_data
    fake = FakeBackend()
    clf = LanguageModelClassifier(backend=fake).fit(X, y)

    clf.predict_proba(X, generation=GenerationConfig(inference_batch_size=1))
    assert max(fake.score_batches) == 1  # the override's batch size was used
    assert clf.generation.inference_batch_size is None  # estimator attribute untouched


def test_classifier_predict_forwards_generation_override(clf_data) -> None:
    X, y = clf_data
    fake = FakeBackend()
    clf = LanguageModelClassifier(backend=fake).fit(X, y)

    clf.predict(X, generation=GenerationConfig(inference_batch_size=1))
    assert max(fake.score_batches) == 1  # predict forwarded the override to predict_proba


def test_classifier_predict_proba_override_is_batch_invariant(clf_data) -> None:
    X, y = clf_data
    clf = LanguageModelClassifier(backend=FakeBackend()).fit(X, y)
    base = clf.predict_proba(X)
    over = clf.predict_proba(X, generation=GenerationConfig(inference_batch_size=1))
    np.testing.assert_array_equal(base, over)


# --- imputer ---------------------------------------------------------------


def test_imputer_transform_discretization_override_routes_scoring() -> None:
    frame = _mixed_frame()
    fake = FakeBackend()  # stable-hash scores cover both the numeric and categorical candidates
    imp = LanguageModelImputer(backend=fake).fit(frame)  # default: numeric generated, `c` scored

    # the override scores the numeric column too, so nothing is generated this call
    imp.transform(frame, discretization=DiscretizationConfig(bins=2))
    assert fake.score_batches and not fake.generate_batches
    assert isinstance(imp.discretization, DiscretizationConfig)
    assert imp.discretization.bins == 0  # estimator attribute untouched


def test_imputer_transform_generation_override_controls_n_samples() -> None:
    frame = _numeric_frame()
    fake = FakeBackend(value="3.5")
    imp = LanguageModelImputer(backend=fake).fit(frame)  # all-generative (numeric)

    imp.transform(frame, generation=GenerationConfig(n_samples=4))
    assert sum(fake.generate_batches) == 2 * 4  # two missing cells, four draws each
    assert imp.generation.n_samples == 1  # estimator attribute untouched
