"""scikit-learn conformance.

``parametrize_with_checks`` runs the official estimator-check suite against the
classifier, regressor, and imputer (wired to the fake backend). Checks that
assume a backend which actually *learns*, or numeric-only input, are listed in
``_EXPECTED_FAILED`` with a reason so they xfail instead of masking real
regressions. The oversampler is an imbalanced-learn sampler (outside that suite),
so it gets the shared clone / set_params contract directly.
"""

from __future__ import annotations

from typing import cast

import pytest
from sklearn.base import BaseEstimator, clone
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklm import (
    LanguageModelClassifier,
    LanguageModelImputer,
    LanguageModelOverSampler,
    LanguageModelRegressor,
)

from .conftest import FakeBackend

_CHECKED: list[BaseEstimator] = [
    LanguageModelClassifier(backend=FakeBackend(), model="m"),
    LanguageModelRegressor(backend=FakeBackend(value="0"), model="m"),
    LanguageModelImputer(backend=FakeBackend(value="0"), model="m"),
]

# Root causes for the checks that cannot hold for this family of estimators.
_STRUCT = "PEP 692 Unpack init: sklearn introspects the raw signature (only `model` + **kwargs)"
_FAKE_MUT = "fake backend records each call, so it mutates during fit (test-double artifact)"
_FAKE_LEARN = "fake backend produces fixed outputs; cannot satisfy a learning check"
_MSG = "estimator raises its own validation message, not sklearn's expected pattern"
_SPARSE = "text/LM estimator does not accept sparse input"
_NUMERIC = "text/LM estimator does not enforce numeric-only dtype/NaN assumptions"

_SHARED = {
    "check_no_attributes_set_in_init": _STRUCT,
    "check_do_not_raise_errors_in_init_or_set_params": _STRUCT,
    "check_estimators_overwrite_params": _FAKE_MUT,
    "check_n_features_in_after_fitting": _MSG,
    "check_estimators_empty_data_messages": _MSG,
    "check_fit2d_predict1d": _MSG,
    "check_estimator_sparse_tag": _SPARSE,
    "check_estimator_sparse_array": _SPARSE,
    "check_estimator_sparse_matrix": _SPARSE,
}

_EXPECTED_FAILED: dict[str, dict[str, str]] = {
    "LanguageModelClassifier": {
        **_SHARED,
        "check_classifiers_train": _FAKE_LEARN,
        "check_supervised_y_2d": _NUMERIC,
        "check_requires_y_none": _MSG,
    },
    "LanguageModelRegressor": {
        **_SHARED,
        "check_regressors_train": _FAKE_LEARN,
        "check_complex_data": _NUMERIC,
        "check_supervised_y_2d": _NUMERIC,
        "check_requires_y_none": _MSG,
    },
    "LanguageModelImputer": {
        **_SHARED,
        "check_complex_data": _NUMERIC,
    },
}


def _expected_failed_checks(estimator: BaseEstimator) -> dict[str, str]:
    return _EXPECTED_FAILED.get(type(estimator).__name__, {})


@parametrize_with_checks(_CHECKED, expected_failed_checks=_expected_failed_checks)
def test_sklearn_compatible(estimator: BaseEstimator, check) -> None:
    check(estimator)


# --- shared estimator contract (all four, including the sampler) ----------


def _estimators() -> list[BaseEstimator]:
    return [
        LanguageModelClassifier(backend=FakeBackend()),
        LanguageModelRegressor(backend=FakeBackend()),
        LanguageModelImputer(backend=FakeBackend()),
        LanguageModelOverSampler(backend=FakeBackend()),
    ]


@pytest.mark.parametrize("est", _estimators())
def test_clone_preserves_params(est: BaseEstimator) -> None:
    cloned = cast(BaseEstimator, clone(est))
    assert cloned.get_params()["model"] == est.get_params()["model"]


@pytest.mark.parametrize("est", _estimators())
def test_set_params_roundtrip(est: BaseEstimator) -> None:
    est.set_params(random_state=7)
    assert est.get_params()["random_state"] == 7


@pytest.mark.parametrize(
    "cls",
    [
        LanguageModelClassifier,
        LanguageModelRegressor,
        LanguageModelImputer,
        LanguageModelOverSampler,
    ],
)
def test_unknown_constructor_kwarg_raises(cls: type[BaseEstimator]) -> None:
    kwargs: dict[str, object] = {"backend": FakeBackend(), "bogus_param": 1}
    with pytest.raises(TypeError, match="bogus_param"):
        cls(**kwargs)


@pytest.mark.parametrize("est", _estimators())
def test_no_fitted_attributes_before_fit(est: BaseEstimator) -> None:
    assert not any(attr.endswith("_") and not attr.startswith("__") for attr in vars(est))
