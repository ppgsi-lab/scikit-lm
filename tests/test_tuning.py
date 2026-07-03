"""Tests for :mod:`sklm.tuning` -- the Optuna search with fold pruning + variance penalty."""

from __future__ import annotations

import datetime
from typing import cast

import numpy as np
import optuna
import pandas as pd
import pytest
from optuna.distributions import BaseDistribution, CategoricalChoiceType, CategoricalDistribution
from optuna.pruners import BasePruner, NopPruner
from optuna.samplers import GridSampler
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState, create_trial
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.linear_model import LogisticRegression

from sklm.config import CosineLR, LRScheduler
from sklm.tuning import (
    LiveLeaderboard,
    PruningSearchCV,
    _reconstruction_error,
    best_trial,
    reconstruction_scorer,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _object_categorical(*choices: object) -> CategoricalDistribution:
    """Categorical over arbitrary objects. Optuna's stub admits only primitives
    (it stores trial params by value), but the constructor accepts any object at
    runtime -- ``sklm.tuning`` records such choices by ``repr``, which is the
    behaviour under test."""
    return CategoricalDistribution(cast("list[CategoricalChoiceType]", list(choices)))


class _ScriptedClassifier(ClassifierMixin, BaseEstimator):
    """Classifier whose per-fold score is replayed from ``scripts[q]``.

    Each successive ``score`` call returns the next entry of the selected script, so a trial's
    four folds yield ``scripts[q][0..3]`` -- letting a test fix a config's exact mean and
    fold-to-fold std while keeping the searched parameter (``q``) a plain primitive.

    Parameters
    ----------
    q : int
        Key selecting which script to replay.
    scripts : dict[int, tuple[float, ...]] | None
        Per-fold score sequences keyed by ``q``.
    """

    def __init__(self, q: int = 0, scripts: dict[int, tuple[float, ...]] | None = None) -> None:
        self.q = q
        self.scripts = scripts

    def fit(self, X: object, y: object) -> _ScriptedClassifier:  # noqa: N803
        self.classes_ = np.unique(np.asarray(y))
        return self

    def predict(self, X: object) -> np.ndarray:  # noqa: N803
        return np.full(len(X), self.classes_[0])  # type: ignore[arg-type]

    def score(self, X: object, y: object, sample_weight: object = None) -> float:  # noqa: N803
        assert self.scripts is not None
        seq = self.scripts[self.q]
        i = getattr(self, "_i", 0)
        self._i = i + 1
        return float(seq[i % len(seq)])


class _SchedClassifier(ClassifierMixin, BaseEstimator):
    """Classifier whose score is a deterministic function of its scheduler config.

    Cosine shape earns a bonus over constant and the learning rate adds directly, so the
    search over (shape, rate) has a single known winner.
    """

    def __init__(self, sched: LRScheduler | None = None) -> None:
        self.sched = sched

    def fit(self, X: object, y: object) -> _SchedClassifier:  # noqa: N803
        self.classes_ = np.unique(np.asarray(y))
        return self

    def predict(self, X: object) -> np.ndarray:  # noqa: N803
        return np.full(len(X), self.classes_[0])  # type: ignore[arg-type]

    def score(self, X: object, y: object, sample_weight: object = None) -> float:  # noqa: N803
        assert isinstance(self.sched, LRScheduler)
        assert self.sched.learning_rate != "auto"
        bonus = 0.5 if isinstance(self.sched, CosineLR) else 0.0
        return bonus + float(self.sched.learning_rate)


class _PruneAfterFirstComplete(BasePruner):
    """Prune every trial except the first to complete, at its first reported step."""

    def prune(self, study: Study, trial: FrozenTrial) -> bool:
        completed = sum(
            1 for t in study.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE
        )
        return completed >= 1 and trial.last_step is not None


class _AlwaysPrune(BasePruner):
    """Prune every trial as soon as it reports its first step."""

    def prune(self, study: Study, trial: FrozenTrial) -> bool:
        return trial.last_step is not None


def _binary_frame(n_per_class: int = 8) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = n_per_class * 2
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series([0] * n_per_class + [1] * n_per_class, name="t")
    return X, y


def _frozen(number: int, mean: float, std: float, seconds: float) -> FrozenTrial:
    start = datetime.datetime(2024, 1, 1)
    return FrozenTrial(
        number=number,
        state=TrialState.COMPLETE,
        value=mean,
        datetime_start=start,
        datetime_complete=start + datetime.timedelta(seconds=seconds),
        params={"q": number},
        distributions={"q": CategoricalDistribution([0, 1, 2, 3])},
        user_attrs={"mean_test_score": mean, "std_test_score": std},
        system_attrs={},
        intermediate_values={},
        trial_id=number,
    )


def test_fit_selects_best_and_predicts() -> None:
    X, y = _binary_frame()
    search = PruningSearchCV(
        LogisticRegression(max_iter=200),
        {"C": CategoricalDistribution([0.1, 1.0, 10.0])},
        cv=4,
        scoring="accuracy",
        n_trials=3,
        sampler=GridSampler({"C": [0.1, 1.0, 10.0]}),
        random_state=0,
    )
    search.fit(X, y)

    assert len(search.study_.trials) == 3
    assert search.best_params_["C"] in (0.1, 1.0, 10.0)
    assert search.predict(X).shape == (len(X),)


def test_std_penalty_flips_selection() -> None:
    scripts = {0: (1.0, 1.0, 0.88, 1.0), 1: (0.95, 0.95, 0.95, 0.95)}  # 0: high mean/std, 1: steady
    space = {"q": CategoricalDistribution([0, 1])}
    X, y = _binary_frame()

    def run(std_penalty: float) -> object:
        search = PruningSearchCV(
            _ScriptedClassifier(scripts=scripts),
            space,
            cv=4,
            n_trials=2,
            std_penalty=std_penalty,
            sampler=GridSampler({"q": [0, 1]}),
            pruner=NopPruner(),
        )
        return search.fit(X, y).best_params_["q"]

    assert run(0.0) == 0  # no penalty -> higher mean wins
    assert run(1.0) == 1  # penalty -> the consistent config wins


def test_object_choices_stored_by_repr_and_resolved_fresh() -> None:
    cosine, constant = LRScheduler.cosine(), LRScheduler.constant()
    X, y = _binary_frame()
    search = PruningSearchCV(
        _SchedClassifier(),
        {
            "sched": _object_categorical(cosine, constant),
            "sched__learning_rate": CategoricalDistribution([0.1, 0.2]),
        },
        cv=4,
        n_trials=4,
        sampler=GridSampler(
            {"sched": [repr(cosine), repr(constant)], "sched__learning_rate": [0.1, 0.2]}
        ),
        pruner=NopPruner(),
    )
    search.fit(X, y)

    assert all(isinstance(t.params["sched"], str) for t in search.study_.trials)
    assert cosine.learning_rate == "auto"  # the shared choices were never mutated
    assert constant.learning_rate == "auto"
    assert search.best_params_["sched"] == repr(cosine)
    best_sched = search.best_estimator_.sched  # type: ignore[attr-defined]
    assert isinstance(best_sched, CosineLR)
    assert best_sched is not cosine
    assert best_sched.learning_rate == 0.2


def test_object_choices_require_unique_reprs() -> None:
    X, y = _binary_frame()
    search = PruningSearchCV(
        _SchedClassifier(),
        {"sched": _object_categorical(LRScheduler.cosine(), LRScheduler.cosine())},
        cv=4,
        n_trials=1,
        pruner=NopPruner(),
    )

    with pytest.raises(ValueError, match="unique reprs"):
        search.fit(X, y)


def test_best_trial_breaks_adj_ties_by_duration() -> None:
    study = optuna.create_study(direction="maximize")
    study.add_trial(_frozen(0, mean=0.9, std=0.02, seconds=100))
    study.add_trial(_frozen(1, mean=0.9, std=0.02, seconds=10))

    assert best_trial(study, std_penalty=0.5).number == 1


def test_best_trial_raises_when_all_pruned() -> None:
    study = optuna.create_study(direction="maximize")
    study.add_trial(create_trial(state=TrialState.PRUNED, intermediate_values={0: 0.3}))

    with pytest.raises(RuntimeError, match="no trial completed"):
        best_trial(study)


def test_pruning_skips_remaining_folds() -> None:
    X, y = _binary_frame()
    search = PruningSearchCV(
        _ScriptedClassifier(scripts={0: (0.5, 0.5, 0.5, 0.5), 1: (0.6, 0.6, 0.6, 0.6)}),
        {"q": CategoricalDistribution([0, 1])},
        cv=4,
        n_trials=2,
        sampler=GridSampler({"q": [0, 1]}),
        pruner=_PruneAfterFirstComplete(),
    )
    search.fit(X, y)

    states = [t.state for t in search.study_.trials]
    assert states.count(TrialState.COMPLETE) == 1
    pruned = [t for t in search.study_.trials if t.state == TrialState.PRUNED]
    assert pruned and all(len(t.intermediate_values) < 4 for t in pruned)
    assert search.best_trial_.state == TrialState.COMPLETE


def test_fit_raises_when_every_trial_pruned() -> None:
    X, y = _binary_frame()
    search = PruningSearchCV(
        _ScriptedClassifier(scripts={0: (0.5, 0.5, 0.5, 0.5)}),
        {"q": CategoricalDistribution([0])},
        cv=4,
        n_trials=1,
        sampler=GridSampler({"q": [0]}),
        pruner=_AlwaysPrune(),
    )
    with pytest.raises(RuntimeError, match="no trial completed"):
        search.fit(X, y)


class _ExplodingClassifier(ClassifierMixin, BaseEstimator):
    """Classifier whose broken config (``q == 0``) raises ``RuntimeError`` at fit time."""

    def __init__(self, q: int = 0) -> None:
        self.q = q

    def fit(self, X: object, y: object) -> _ExplodingClassifier:  # noqa: N803
        if self.q == 0:
            raise RuntimeError("generation failed")
        self.classes_ = np.unique(np.asarray(y))
        return self

    def predict(self, X: object) -> np.ndarray:  # noqa: N803
        return np.full(len(X), self.classes_[0])  # type: ignore[arg-type]

    def score(self, X: object, y: object, sample_weight: object = None) -> float:  # noqa: N803
        return 0.75


def test_catch_marks_trial_failed_and_continues() -> None:
    X, y = _binary_frame()
    search = PruningSearchCV(
        _ExplodingClassifier(),
        {"q": CategoricalDistribution([0, 1])},
        cv=4,
        n_trials=2,
        sampler=GridSampler({"q": [0, 1]}),
        pruner=NopPruner(),
        refit=False,
        catch=(RuntimeError,),
    )
    search.fit(X, y)

    states = {t.params["q"]: t.state for t in search.study_.trials}
    assert states[0] == TrialState.FAIL
    assert states[1] == TrialState.COMPLETE
    assert search.best_params_["q"] == 1

    failed = next(t for t in search.study_.trials if t.state == TrialState.FAIL)
    assert "generation failed" in failed.user_attrs["error"]

    means = np.asarray(search.cv_results_["mean_test_score"], dtype=float)
    assert len(means) == 2
    failed_row = next(i for i, t in enumerate(search.study_.trials) if t.state == TrialState.FAIL)
    assert np.isnan(means[failed_row])
    assert search.cv_results_["rank_test_score"][failed_row] == 2


def test_errors_propagate_without_catch() -> None:
    X, y = _binary_frame()
    search = PruningSearchCV(
        _ExplodingClassifier(),
        {"q": CategoricalDistribution([0])},
        cv=4,
        n_trials=1,
        sampler=GridSampler({"q": [0]}),
        pruner=NopPruner(),
    )
    with pytest.raises(RuntimeError, match="generation failed"):
        search.fit(X, y)


def test_refit_false_has_no_best_estimator() -> None:
    X, y = _binary_frame()
    search = PruningSearchCV(
        LogisticRegression(max_iter=200),
        {"C": CategoricalDistribution([1.0])},
        cv=4,
        scoring="accuracy",
        n_trials=1,
        sampler=GridSampler({"C": [1.0]}),
        refit=False,
    )
    search.fit(X, y)

    assert search.best_params_ == {"C": 1.0}
    assert not hasattr(search, "best_estimator_")


class _ScriptedImputer(TransformerMixin, BaseEstimator):
    """Transformer that 'imputes' column ``a`` with the constant ``fill``.

    Reconstruction error against a ground truth whose ``a`` is constant is therefore controlled by
    ``fill`` alone -- letting a test fix which config reconstructs best while keeping the searched
    parameter a plain primitive.

    Parameters
    ----------
    fill : float
        Value written into column ``a`` by :meth:`transform`.
    """

    def __init__(self, fill: float = 0.0) -> None:
        self.fill = fill

    def fit(self, X: object, y: object = None) -> _ScriptedImputer:  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        out = X.copy()
        out["a"] = self.fill
        return out


def _impute_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = pd.DataFrame({"a": [1.0] * 8, "b": np.arange(8.0)})
    corrupted = truth.copy()
    corrupted["a"] = np.nan
    return corrupted, truth


def test_reconstruction_error_mixes_numeric_and_categorical() -> None:
    truth = pd.DataFrame({"n": [0.0, 10.0], "c": ["x", "y"]})
    full = np.ones((2, 2), dtype=bool)
    assert _reconstruction_error(truth.copy(), truth, full) == 0.0

    # n: std([0, 10]) = 5; ((5-0)/5)**2 = 1 and ((5-10)/5)**2 = 1 -> 1.0
    # c: mismatch [x==x, z!=y] -> 0.5 ;  mean(1.0, 0.5) = 0.75
    bad = pd.DataFrame({"n": [5.0, 5.0], "c": ["x", "z"]})
    assert _reconstruction_error(bad, truth, full) == pytest.approx(0.75)


def test_reconstruction_error_scores_only_masked_cells() -> None:
    truth = pd.DataFrame({"a": [1.0, 1.0, 1.0, 1.0], "b": [0.0, 1.0, 2.0, 3.0]})
    pred = pd.DataFrame({"a": [2.0, 2.0, 1.0, 1.0], "b": [0.0, 1.0, 2.0, 3.0]})
    mask = np.array([[True, False], [True, False], [False, False], [False, False]])
    # only column 'a' has imputed cells (rows 0-1); 'b' is fully observed and is skipped, not
    # averaged in as a zero. std('a') = 0 -> guarded to 1; ((2-1)/1)**2 = 1 on both masked cells
    # -> 1.0, undiluted by the observed 'b' column.
    assert _reconstruction_error(pred, truth, mask) == 1.0


def test_transformer_mode_selects_best_reconstruction() -> None:
    X_na, truth = _impute_data()
    search = PruningSearchCV(
        _ScriptedImputer(),
        {"fill": CategoricalDistribution([0.0, 1.0, 2.0])},
        cv=4,
        n_trials=3,
        scoring=reconstruction_scorer,
        sampler=GridSampler({"fill": [0.0, 1.0, 2.0]}),
        pruner=NopPruner(),
        random_state=0,
    )
    search.fit(X_na, truth)

    assert search.best_params_["fill"] == 1.0
    imputed = search.transform(X_na)
    assert isinstance(imputed, pd.DataFrame)
    assert imputed["a"].tolist() == [1.0] * 8


def test_transformer_mode_prunes_weak_folds() -> None:
    X_na, truth = _impute_data()
    search = PruningSearchCV(
        _ScriptedImputer(),
        {"fill": CategoricalDistribution([0.0, 1.0])},
        cv=4,
        n_trials=2,
        scoring=reconstruction_scorer,
        sampler=GridSampler({"fill": [0.0, 1.0]}),
        pruner=_PruneAfterFirstComplete(),
        random_state=0,
    )
    search.fit(X_na, truth)

    states = [t.state for t in search.study_.trials]
    assert states.count(TrialState.COMPLETE) == 1
    pruned = [t for t in search.study_.trials if t.state == TrialState.PRUNED]
    assert pruned and all(len(t.intermediate_values) < 4 for t in pruned)


def test_available_if_exposes_only_the_supported_delegate() -> None:
    X, y = _binary_frame()
    clf = PruningSearchCV(
        LogisticRegression(max_iter=200),
        {"C": CategoricalDistribution([1.0])},
        cv=4,
        scoring="accuracy",
        n_trials=1,
        sampler=GridSampler({"C": [1.0]}),
    )
    clf.fit(X, y)
    assert hasattr(clf, "predict")
    assert not hasattr(clf, "transform")  # a classifier search never exposes transform

    X_na, truth = _impute_data()
    imp = PruningSearchCV(
        _ScriptedImputer(),
        {"fill": CategoricalDistribution([1.0])},
        cv=4,
        n_trials=1,
        scoring=reconstruction_scorer,
        sampler=GridSampler({"fill": [1.0]}),
        pruner=NopPruner(),
    )
    imp.fit(X_na, truth)
    assert hasattr(imp, "transform")
    assert not hasattr(imp, "predict")  # a transformer search never exposes predict


def test_score_delegates_to_the_search_scorer() -> None:
    X, y = _binary_frame()
    search = PruningSearchCV(
        LogisticRegression(max_iter=200),
        {"C": CategoricalDistribution([1.0])},
        cv=4,
        scoring="accuracy",
        n_trials=1,
        sampler=GridSampler({"C": [1.0]}),
    )
    search.fit(X, y)

    expected = float(np.mean(search.predict(X) == np.asarray(y)))
    assert search.score(X, y) == pytest.approx(expected)


def test_predict_proba_and_classes_delegate_for_classifiers() -> None:
    X, y = _binary_frame()
    search = PruningSearchCV(
        LogisticRegression(max_iter=200),
        {"C": CategoricalDistribution([1.0])},
        cv=4,
        scoring="accuracy",
        n_trials=1,
        sampler=GridSampler({"C": [1.0]}),
    )
    search.fit(X, y)

    proba = search.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert list(search.classes_) == [0, 1]


def test_predict_proba_and_classes_absent_for_transformers() -> None:
    X_na, truth = _impute_data()
    search = PruningSearchCV(
        _ScriptedImputer(),
        {"fill": CategoricalDistribution([1.0])},
        cv=4,
        n_trials=1,
        scoring=reconstruction_scorer,
        sampler=GridSampler({"fill": [1.0]}),
        pruner=NopPruner(),
    )
    search.fit(X_na, truth)

    assert not hasattr(search, "predict_proba")
    assert not hasattr(search, "classes_")


def test_cv_results_table_and_best_index() -> None:
    X, y = _binary_frame()
    search = PruningSearchCV(
        LogisticRegression(max_iter=200),
        {"C": CategoricalDistribution([0.1, 1.0, 10.0])},
        cv=4,
        scoring="accuracy",
        n_trials=3,
        sampler=GridSampler({"C": [0.1, 1.0, 10.0]}),
        random_state=0,
    )
    search.fit(X, y)

    results = search.cv_results_
    assert set(results) == {
        "params",
        "param_C",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
    }
    assert len(results["params"]) == 3
    assert results["params"][search.best_index_] == search.best_params_
    assert results["rank_test_score"][search.best_index_] == 1  # std_penalty=0 -> best mean
    assert list(results["param_C"]) == [t.params["C"] for t in search.study_.trials]


def test_cv_results_marks_pruned_trials_nan_and_ranks_them_last() -> None:
    X, y = _binary_frame()
    search = PruningSearchCV(
        _ScriptedClassifier(scripts={0: (0.5,) * 4, 1: (0.6,) * 4}),
        {"q": CategoricalDistribution([0, 1])},
        cv=4,
        n_trials=2,
        sampler=GridSampler({"q": [0, 1]}),
        pruner=_PruneAfterFirstComplete(),
    )
    search.fit(X, y)

    means = np.asarray(search.cv_results_["mean_test_score"], dtype=float)
    ranks = np.asarray(search.cv_results_["rank_test_score"], dtype=int)
    pruned_rows = [i for i, t in enumerate(search.study_.trials) if t.state == TrialState.PRUNED]
    assert pruned_rows
    for i in pruned_rows:
        assert np.isnan(means[i])
        assert ranks[i] == 2  # after the single completed trial
    assert ranks[search.best_index_] == 1


def test_live_leaderboard_renders_complete_and_pruned(capsys: pytest.CaptureFixture[str]) -> None:
    study = optuna.create_study(direction="maximize")
    dist: dict[str, BaseDistribution] = {"q": CategoricalDistribution([0, 1])}
    study.add_trial(
        create_trial(
            state=TrialState.COMPLETE,
            value=0.9,
            params={"q": 0},
            distributions=dist,
            user_attrs={"mean_test_score": 0.9, "std_test_score": 0.05},
        )
    )
    study.add_trial(
        create_trial(
            state=TrialState.PRUNED,
            params={"q": 1},
            distributions=dist,
            intermediate_values={0: 0.3},
        )
    )

    study.add_trial(
        create_trial(
            state=TrialState.FAIL,
            params={"q": 0},
            distributions=dist,
            user_attrs={"error": "RuntimeError: generation failed"},
        )
    )

    LiveLeaderboard(std_penalty=0.5)(study, study.trials[-1])

    out = capsys.readouterr().out
    assert "0.9" in out
    assert "pruned" in out.lower()
    assert "failed" in out.lower()
    assert "generation" in out  # the failed trial's error surfaces in the table
    assert "params" in out  # one params column, not one column per key
    assert "q=0" in out  # params rendered as leaf=value (completed trial)
    assert "q=1" in out  # and the pruned trial's params too
