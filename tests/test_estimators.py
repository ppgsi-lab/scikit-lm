"""Public API of the four estimators.

Happy-path contracts (``fit`` then a valid prediction) are parametrized over the
backend matrix via the ``make`` fixture, so they run on the fake backend by
default and on HF / MLX under ``-m slow``. Backend-agnostic plumbing (scikit-learn
validation, input coercion) and exact-value semantics are checked once against the
deterministic fake backend.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from sklm import (
    BracketSerializer,
    Field,
    GenerationConfig,
    KeyValueSerializer,
    LanguageModelClassifier,
    LanguageModelImputer,
    LanguageModelOverSampler,
    LanguageModelRegressor,
    Serializer,
    TrainingConfig,
)

from .conftest import FakeBackend, MakeEstimator

# --- classifier -----------------------------------------------------------


def test_classifier_fit_sets_sklearn_attrs(make: MakeEstimator, clf_data) -> None:
    X, y = clf_data
    clf = make(LanguageModelClassifier)
    assert clf.fit(X, y) is clf
    assert list(clf.classes_) == sorted(set(y))
    assert clf.n_features_in_ == X.shape[1]
    np.testing.assert_array_equal(clf.feature_names_in_, np.asarray(X.columns))


def test_classifier_predict_returns_valid_labels(make: MakeEstimator, clf_data) -> None:
    X, y = clf_data
    clf = make(LanguageModelClassifier).fit(X, y)
    pred = clf.predict(X.head(4))
    assert pred.shape == (4,)
    assert set(pred).issubset(set(clf.classes_))


def test_classifier_predict_proba_is_a_distribution(make: MakeEstimator, clf_data) -> None:
    X, y = clf_data
    clf = make(LanguageModelClassifier).fit(X, y)
    proba = clf.predict_proba(X.head(4))
    assert proba.shape == (4, len(clf.classes_))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert np.all(proba >= 0)


def test_classifier_predict_before_fit_raises(clf_data) -> None:
    X, _ = clf_data
    with pytest.raises(NotFittedError):
        LanguageModelClassifier(backend=FakeBackend()).predict(X)


def test_classifier_feature_count_mismatch_raises(clf_data) -> None:
    X, y = clf_data
    clf = LanguageModelClassifier(backend=FakeBackend()).fit(X, y)
    with pytest.raises(ValueError, match="features"):
        clf.predict(X.iloc[:, :2])


def test_classifier_renamed_columns_raise(clf_data) -> None:
    X, y = clf_data
    clf = LanguageModelClassifier(backend=FakeBackend()).fit(X, y)
    with pytest.raises(ValueError, match="missing features"):
        clf.predict(X.rename(columns={"age": "AGE"}))


def test_classifier_accepts_array_input() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 3))
    y = rng.choice(["a", "b"], size=12)
    clf = LanguageModelClassifier(backend=FakeBackend()).fit(X, y)
    assert clf.predict(X).shape == (12,)


def test_classifier_accepts_integer_column_dataframe() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(12, 3)))  # integer RangeIndex columns
    y = rng.choice(["a", "b"], size=12)
    clf = LanguageModelClassifier(backend=FakeBackend()).fit(X, y)
    assert clf.predict(X).shape == (12,)


def test_classifier_handles_nan_feature(clf_data) -> None:
    X, y = clf_data
    clf = LanguageModelClassifier(backend=FakeBackend()).fit(X, y)
    X_nan = X.copy()
    X_nan.loc[0, "score"] = np.nan
    np.testing.assert_allclose(clf.predict_proba(X_nan).sum(axis=1), 1.0)


def test_classifier_feature_names_cleared_on_array_refit(clf_data) -> None:
    X, y = clf_data
    clf = LanguageModelClassifier(backend=FakeBackend()).fit(X, y)
    assert hasattr(clf, "feature_names_in_")
    clf.fit(X.to_numpy(), y)
    assert not hasattr(clf, "feature_names_in_")


def test_classifier_impossible_candidate_gets_zero_mass(clf_data) -> None:
    X, y = clf_data
    backend = FakeBackend(scores={'"yes"': float("-inf"), '"no"': -1.0})
    clf = LanguageModelClassifier(backend=backend).fit(X, y)
    proba = clf.predict_proba(X)
    yes = list(clf.classes_).index("yes")
    np.testing.assert_allclose(proba[:, yes], 0.0)
    assert set(clf.predict(X)) == {"no"}


def test_classifier_duplicate_columns_raise() -> None:
    X = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["age", "age", "score"])
    y = np.array(["yes", "no"])
    clf = LanguageModelClassifier(backend=FakeBackend())
    with pytest.raises(ValueError, match="duplicate column names"):
        clf.fit(X, y)


def test_classifier_custom_score_pool_rows_sum_to_one(clf_data) -> None:
    X, y = clf_data

    def unnormalized_pool(rows: Sequence[Sequence[float]]) -> list[float]:
        # one raw log-likelihood row (n_samples == 1); exponentiate without normalizing
        return list(np.exp(np.asarray(rows[0])))

    gen = GenerationConfig(score_pool=unnormalized_pool)
    clf = LanguageModelClassifier(backend=FakeBackend(), generation=gen).fit(X, y)
    proba = clf.predict_proba(X.head(4))
    assert proba.shape == (4, len(clf.classes_))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert np.all(proba >= 0)


# --- regressor ------------------------------------------------------------


def test_regressor_predict_is_a_finite_vector(make: MakeEstimator, reg_data) -> None:
    X, y = reg_data
    reg = make(LanguageModelRegressor)
    assert reg.fit(X, y) is reg
    pred = reg.predict(X.head(4))
    assert pred.shape == (4,)
    assert np.all(np.isfinite(pred))


def test_regressor_averages_samples(reg_data) -> None:
    X, y = reg_data
    reg = LanguageModelRegressor(
        backend=FakeBackend(value="3.5"), generation=GenerationConfig(n_samples=4)
    ).fit(X, y)
    pred = reg.predict(X)
    assert pred.shape == (len(X),)
    np.testing.assert_allclose(pred, 3.5)


def test_regressor_raises_when_every_draw_is_malformed(reg_data) -> None:
    X, y = reg_data
    reg = LanguageModelRegressor(
        backend=FakeBackend(value="garbage"), generation=GenerationConfig(n_samples=2)
    ).fit(X, y)
    with pytest.raises(RuntimeError, match="not producing valid numeric values"):
        reg.predict(X)


def test_regressor_accepts_integer_column_dataframe() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(12, 2)))  # integer RangeIndex columns
    y = rng.normal(size=12)
    reg = LanguageModelRegressor(
        backend=FakeBackend(value="1.0"), generation=GenerationConfig(n_samples=2)
    ).fit(X, y)
    assert reg.predict(X).shape == (12,)


# --- imputer --------------------------------------------------------------


def test_imputer_fills_all_nans_and_preserves_observed(make: MakeEstimator, nan_data) -> None:
    out = make(LanguageModelImputer).fit_transform(nan_data)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == nan_data.shape
    assert out.isna().sum().sum() == 0
    observed = nan_data["age"].notna()
    np.testing.assert_array_equal(
        out.loc[observed, "age"].to_numpy(), nan_data.loc[observed, "age"].to_numpy()
    )


def test_imputer_get_feature_names_out(nan_data) -> None:
    imp = LanguageModelImputer(backend=FakeBackend()).fit(nan_data)
    np.testing.assert_array_equal(imp.get_feature_names_out(), np.asarray(nan_data.columns))


def test_imputer_integer_column_dataframe_restores_labels() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(10, 3)))  # integer RangeIndex columns
    X.iloc[0, 0] = np.nan
    out = LanguageModelImputer(backend=FakeBackend(value="0")).fit_transform(X)
    assert out.isna().sum().sum() == 0
    assert list(out.columns) == list(X.columns)


class _RecordingImputeBackend(FakeBackend):
    """Record the prompts ``generate`` sees, to observe which cells condition each
    imputation."""

    def __init__(self) -> None:
        super().__init__(value='"z"')  # JSON-quoted: decodable as the categorical "z"
        self.prompts: list[str] = []

    def generate(self, prompts: Sequence[str], generation: object) -> list[str]:
        self.prompts.extend(prompts)
        return super().generate(prompts, generation)


def test_imputer_transform_aligns_reordered_columns_by_name() -> None:
    """transform aligns columns by name, not position: a reordered DataFrame
    conditions each imputation on the same cells (identical prompts) and returns in
    the caller's own column order. Positional aliasing would condition on the wrong
    neighbor and so produce different prompts."""
    # all-categorical so imputation takes the generate (not score) path
    fit_df = pd.DataFrame({"a": ["p", "q", "p", "q"], "b": ["x", "y", "x", "y"]})
    in_order = pd.DataFrame({"a": ["p", "q"], "b": [None, "y"]})

    rec1 = _RecordingImputeBackend()
    LanguageModelImputer(backend=rec1, random_state=0).fit(fit_df).transform(in_order)

    rec2 = _RecordingImputeBackend()
    imp = LanguageModelImputer(backend=rec2, random_state=0).fit(fit_df)
    out = imp.transform(in_order[["b", "a"]])

    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["b", "a"]
    assert rec2.prompts == rec1.prompts


def test_imputer_accepts_custom_serializer(nan_data) -> None:
    """A serializer that only implements the protocol must work end-to-end."""

    class KVSerializer:
        def encode_value(self, value: object, *, numeric: bool) -> str:
            return str(value)

        def serialize(self, fields: Sequence[Field]) -> str:
            return "|".join(
                f"{f.name}={self.encode_value(f.value, numeric=f.numeric)}" for f in fields
            )

        def prefix(self, known: Sequence[Field], target: object) -> str:
            head = "|".join(
                f"{f.name}={self.encode_value(f.value, numeric=f.numeric)}" for f in known
            )
            return (head + "|" if known else "") + f"{target}="

        def split(self, context: Sequence[Field], target: Sequence[Field]) -> tuple[str, str]:
            full = self.serialize([*context, *target])
            prompt = self.prefix(context, target[0].name)
            return prompt, full[len(prompt) :]

        def decode_value(self, text: str, *, numeric: bool) -> object | None:
            token = text.strip().split("|")[0]
            if numeric:
                try:
                    return float(token)
                except ValueError:
                    return None
            return token

    out = LanguageModelImputer(
        backend=FakeBackend(value="0"), serializer=KVSerializer()
    ).fit_transform(nan_data)
    assert out.isna().sum().sum() == 0


@pytest.mark.parametrize("serializer", [BracketSerializer(), KeyValueSerializer()])
def test_imputer_accepts_builtin_non_json_serializers(serializer: Serializer, nan_data) -> None:
    """The non-JSON built-in serializers must work end-to-end: ``"0"`` decodes to
    a number for the numeric cell and verbatim text for the categorical one."""
    out = LanguageModelImputer(backend=FakeBackend(value="0"), serializer=serializer).fit_transform(
        nan_data
    )
    assert isinstance(out, pd.DataFrame)
    assert out.isna().sum().sum() == 0


# --- oversampler ----------------------------------------------------------


def test_oversampler_balances_classes(make: MakeEstimator, imbalanced_data) -> None:
    X, y = imbalanced_data
    result = make(LanguageModelOverSampler).fit_resample(X, y)
    X_res, y_res = result[0], result[1]
    counts = pd.Series(y_res).value_counts()
    assert counts.min() == counts.max()
    assert len(X_res) == len(y_res) > len(X)
    np.testing.assert_array_equal(np.asarray(y_res)[: len(y)], np.asarray(y))  # originals kept


def test_oversampler_rounds_integer_columns() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"age": rng.integers(20, 60, size=20)})  # int64 column
    y = np.array([0] * 16 + [1] * 4)
    result = LanguageModelOverSampler(backend=FakeBackend(value="2.9")).fit_resample(X, y)
    X_res = result[0]
    synthetic = X_res.iloc[len(X) :]
    assert (synthetic["age"] == 3).all()  # round(2.9), not truncate to 2


def test_oversampler_preserves_float_columns() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"weight": rng.normal(size=20)})  # float64 column
    y = np.array([0] * 16 + [1] * 4)
    result = LanguageModelOverSampler(backend=FakeBackend(value="2.5")).fit_resample(X, y)
    X_res = result[0]
    synthetic = X_res.iloc[len(X) :]
    assert (synthetic["weight"] == 2.5).all()  # float round-trips, not rounded to 2 or 3


def test_oversampler_warns_on_loss_on_target_only(imbalanced_data) -> None:
    X, y = imbalanced_data
    over = LanguageModelOverSampler(
        backend=FakeBackend(value="0"), training=TrainingConfig(loss_on_target_only=True, epochs=1)
    )
    with pytest.warns(RuntimeWarning, match="oversampler"):
        over.fit_resample(X, y)
