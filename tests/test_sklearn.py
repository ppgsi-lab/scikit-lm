"""scikit-learn conformance, run against a real (MLX) backend.

``parametrize_with_checks`` runs the official estimator-check suite against the
classifier, regressor, and imputer wired to a real MLX backend -- no test double,
so the genuine serialize / fine-tune / infer path is exercised. The bulk fits a
single epoch (a real but shallow fit): only the two learning checks need a
trained model, and those are split into ``test_sklearn_learns_mlx`` at full
epochs. Inference is kept deterministic -- the classifier scores candidates, the
regressor discretizes, the imputer decodes greedily -- so the invariance checks
hold.

``_EXPECTED_FAILED`` lists only what genuinely cannot pass on this backend:

- the PEP 692 ``Unpack`` constructor, which scikit-learn introspects from the raw
  ``(model, **kwargs)`` signature;
- ``check_fit_idempotent`` / ``check_regressor_data_not_an_array``, which compare
  two fits -- MLX training is not bitwise-deterministic (GPU float-reduction
  order), so the refit lands on slightly different weights;
- the two learning checks, which a one-epoch fit cannot clear (run for real in
  ``test_sklearn_learns_mlx``).

The conformance suite is ``slow`` + MLX-gated. The oversampler is an
imbalanced-learn sampler (outside that suite), so it gets the shared clone /
set_params contract directly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest
from sklearn.base import BaseEstimator, clone
from sklearn.utils.estimator_checks import (
    check_classifiers_train,
    check_regressors_train,
    parametrize_with_checks,
)

from sklm import (
    DiscretizationConfig,
    GenerationConfig,
    LanguageModelClassifier,
    LanguageModelImputer,
    LanguageModelOverSampler,
    LanguageModelRegressor,
    TrainingConfig,
)

from .conftest import _has_mlx

_MLX_MODEL = "gabfssilva/distilgpt2"
requires_mlx = pytest.mark.skipif(
    not _has_mlx(), reason="requires the 'mlx' extra on Apple Silicon"
)


# --- bulk conformance: real MLX backend, one epoch, deterministic inference ---


def _bulk_clf() -> LanguageModelClassifier:
    return LanguageModelClassifier(
        model=_MLX_MODEL, backend="mlx", random_state=0, callback=[],
        training=TrainingConfig(epochs=1, batch_size=8),
    )


def _bulk_reg() -> LanguageModelRegressor:
    return LanguageModelRegressor(
        model=_MLX_MODEL, backend="mlx", random_state=0, callback=[],
        training=TrainingConfig(epochs=1, batch_size=8),
        discretization=DiscretizationConfig(bins=64),
    )


def _bulk_imp() -> LanguageModelImputer:
    return LanguageModelImputer(
        model=_MLX_MODEL, backend="mlx", random_state=0, callback=[],
        training=TrainingConfig(epochs=1, batch_size=8),
        generation=GenerationConfig(temperature=0.0),
    )


_CHECKED: list[BaseEstimator] = [_bulk_clf(), _bulk_reg(), _bulk_imp()]

_KWARGS = "PEP 692 Unpack init: sklearn introspects the raw signature (only `model` + **kwargs)"
_NONDET = "compares two fits; MLX training is not bitwise-deterministic (GPU reduction order)"
_LEARN = "a one-epoch bulk fit can't clear the threshold; run for real in test_sklearn_learns_mlx"

_KWARGS_CHECKS = {
    "check_no_attributes_set_in_init": _KWARGS,
    "check_do_not_raise_errors_in_init_or_set_params": _KWARGS,
}

_EXPECTED_FAILED: dict[str, dict[str, str]] = {
    "LanguageModelClassifier": {
        **_KWARGS_CHECKS,
        "check_fit_idempotent": _NONDET,
        "check_classifiers_train": _LEARN,
    },
    "LanguageModelRegressor": {
        **_KWARGS_CHECKS,
        "check_fit_idempotent": _NONDET,
        "check_regressor_data_not_an_array": _NONDET,
        "check_regressors_train": _LEARN,
    },
    "LanguageModelImputer": dict(_KWARGS_CHECKS),
}


def _expected_failed_checks(estimator: BaseEstimator) -> dict[str, str]:
    return _EXPECTED_FAILED.get(type(estimator).__name__, {})


@pytest.mark.slow
@requires_mlx
@parametrize_with_checks(_CHECKED, expected_failed_checks=_expected_failed_checks)
def test_sklearn_compatible(estimator: BaseEstimator, check) -> None:
    check(estimator)


# --- the two learning checks at full epochs (slow) ------------------------
#
# The bulk fits one epoch -- enough for every structural check but not to clear
# an accuracy / R^2 threshold -- so these re-run the learning checks on a real
# fine-tune. The classifier ranks candidates deterministically, so a few epochs
# suffice. The regressor's target is harder (the check quantizes X to ~8 levels
# and only 1 of 10 features is informative), so it leans on loss-on-target,
# discretized scoring, and heavy column-order augmentation.


def _learn_clf() -> LanguageModelClassifier:
    return LanguageModelClassifier(
        model=_MLX_MODEL, backend="mlx", random_state=0, callback=[],
        training=TrainingConfig(epochs=25, batch_size=8),
    )


def _learn_reg() -> LanguageModelRegressor:
    return LanguageModelRegressor(
        model=_MLX_MODEL, backend="mlx", random_state=0, callback=[],
        training=TrainingConfig(
            epochs=10, batch_size=8, loss_on_target_only=True, augmentation_factor=16
        ),
        discretization=DiscretizationConfig(bins=64),
    )


@pytest.mark.slow
@requires_mlx
@pytest.mark.parametrize(
    ("name", "estimator", "check"),
    [
        pytest.param(
            "LanguageModelClassifier", _learn_clf(), check_classifiers_train, id="clf-train"
        ),
        pytest.param(
            "LanguageModelRegressor", _learn_reg(), check_regressors_train, id="reg-train"
        ),
    ],
)
def test_sklearn_learns_mlx(
    name: str, estimator: BaseEstimator, check: Callable[..., None]
) -> None:
    """The learning checks the one-epoch bulk can't clear, on a real fine-tune."""
    check(name, estimator)


# --- shared estimator contract (all four, including the sampler) ----------
#
# Constructor / clone / set_params plumbing only -- no fit, so a string backend
# selector is enough and these stay fast and backend-agnostic.


def _estimators() -> list[BaseEstimator]:
    return [
        LanguageModelClassifier(backend="mlx"),
        LanguageModelRegressor(backend="mlx"),
        LanguageModelImputer(backend="mlx"),
        LanguageModelOverSampler(backend="mlx"),
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
    kwargs: dict[str, object] = {"backend": "mlx", "bogus_param": 1}
    with pytest.raises(TypeError, match="bogus_param"):
        cls(**kwargs)


@pytest.mark.parametrize("est", _estimators())
def test_no_fitted_attributes_before_fit(est: BaseEstimator) -> None:
    assert not any(attr.endswith("_") and not attr.startswith("__") for attr in vars(est))
