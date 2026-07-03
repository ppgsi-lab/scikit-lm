"""Optuna-backed hyperparameter search with fold-level pruning and a variance penalty.

This module is the home for third-party tuning integrations. Its single offering,
:class:`PruningSearchCV`, is a scikit-learn-style search that hand-rolls the cross-validation
loop so it can do two things ``optuna.integration.OptunaSearchCV`` cannot:

- **Fold-level pruning** -- after each fold it reports the running mean to the trial and consults
  the study's pruner, so a clearly-weak config stops without fine-tuning the remaining folds.
  (``OptunaSearchCV`` only prunes along a ``partial_fit`` learning curve, which the LM estimators
  do not expose.)
- **A variance penalty** -- selection ranks by ``adj = mean - std_penalty * std`` over the folds,
  preferring configs that are consistent across folds, with wall-clock as a defensive tie-break.

``optuna`` is an optional dependency; importing this module without it raises a helpful error.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Protocol, Self, cast

try:
    import optuna
    from optuna.distributions import (
        BaseDistribution,
        CategoricalDistribution,
        FloatDistribution,
        IntDistribution,
    )
    from optuna.pruners import BasePruner, MedianPruner
    from optuna.samplers import BaseSampler, TPESampler
    from optuna.study import Study
    from optuna.trial import FrozenTrial, Trial, TrialState
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "sklm.tuning requires the 'optuna' extra: pip install scikit-lm[optuna]"
    ) from exc

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone, is_classifier
from sklearn.metrics import check_scoring
from sklearn.model_selection import BaseCrossValidator, KFold, StratifiedKFold
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

__all__ = ["LiveLeaderboard", "PruningSearchCV", "best_trial", "reconstruction_scorer"]

type ParamValue = bool | int | float | str | None
type StudyCallback = Callable[[Study, FrozenTrial], None]


class _Estimator(Protocol):
    """The estimator surface the search uses; sklearn's ``BaseEstimator`` stub omits these.

    A fitted estimator implements ``predict``/``predict_proba``/``classes_`` (predictors) or
    ``transform`` (transformers); the search calls the one matching its mode, so none is a
    runtime guarantee on every estimator.
    """

    classes_: np.ndarray

    def set_params(self, **params: object) -> Self: ...
    def fit(self, X: object, y: object = ..., **kwargs: object) -> object: ...  # noqa: N803
    def predict(self, X: object) -> np.ndarray: ...  # noqa: N803
    def predict_proba(self, X: object) -> np.ndarray: ...  # noqa: N803
    def transform(self, X: object) -> object: ...  # noqa: N803


type Scorer = Callable[[_Estimator, object, object], float]


def _configured(estimator: BaseEstimator, params: Mapping[str, object]) -> _Estimator:
    """Return a fresh clone of ``estimator`` with ``params`` applied.

    The ``cast`` bridges sklearn's loosely typed ``clone`` (broad union return) to the concrete
    estimator surface the search relies on.
    """
    fresh = cast("_Estimator", clone(estimator))
    fresh.set_params(**params)
    return fresh


def _resolved(
    params: Mapping[str, ParamValue], distributions: Mapping[str, BaseDistribution]
) -> dict[str, object]:
    """Trial params as ``set_params`` values: repr-stored object choices become fresh copies.

    The deep copy keeps trials (and the refit) from sharing a mutable config instance -- a
    nested ``set_params`` on a shared choice would silently rewrite it for every other trial.
    """
    out: dict[str, object] = {}
    for name, value in params.items():
        objects = _object_choices(distributions[name])
        if objects is not None and isinstance(value, str):
            out[name] = deepcopy(objects[value])
        else:
            out[name] = value
    return out


def _object_choices(distribution: BaseDistribution) -> dict[str, object] | None:
    """``repr -> choice`` for a categorical whose choices are not all Optuna primitives.

    Optuna stores only ``None | bool | int | float | str`` in a trial, so a categorical over
    arbitrary objects (e.g. ``LRScheduler`` shapes) is suggested and recorded by ``repr`` and
    mapped back to the object at ``set_params`` time. ``None`` means the distribution needs no
    mapping.
    """
    if not isinstance(distribution, CategoricalDistribution):
        return None
    if all(c is None or isinstance(c, bool | int | float | str) for c in distribution.choices):
        return None
    mapping: dict[str, object] = {repr(c): c for c in distribution.choices}
    if len(mapping) != len(distribution.choices):
        raise ValueError(f"object choices must have unique reprs, got {distribution.choices}")
    return mapping


def _suggest(trial: Trial, name: str, distribution: BaseDistribution) -> ParamValue:
    """Sample ``name`` from ``distribution`` via the matching ``trial.suggest_*`` call."""
    if isinstance(distribution, CategoricalDistribution):
        objects = _object_choices(distribution)
        if objects is not None:
            return trial.suggest_categorical(name, list(objects))
        return trial.suggest_categorical(name, list(distribution.choices))
    if isinstance(distribution, IntDistribution):
        return trial.suggest_int(
            name, distribution.low, distribution.high, step=distribution.step, log=distribution.log
        )
    if isinstance(distribution, FloatDistribution):
        return trial.suggest_float(
            name, distribution.low, distribution.high, step=distribution.step, log=distribution.log
        )
    raise TypeError(f"unsupported distribution for {name!r}: {type(distribution).__name__}")


def _adj(trial: FrozenTrial, std_penalty: float) -> float:
    """Risk-adjusted score: mean CV score minus ``std_penalty`` times its fold-to-fold std."""
    attrs = trial.user_attrs
    return float(attrs["mean_test_score"]) - std_penalty * float(attrs["std_test_score"])


def _duration(trial: FrozenTrial) -> float:
    return trial.duration.total_seconds() if trial.duration is not None else 0.0


def best_trial(study: Study, *, std_penalty: float = 0.0) -> FrozenTrial:
    """Return the best completed trial, ranked by risk-adjusted score then by speed.

    Score is the primary key (via :func:`_adj`), so a better-but-slower or better-but-noisier
    trial is never displaced by the penalty -- the std only separates configs with a smaller
    adjusted score, and the wall-clock only breaks adjusted-score ties.

    Parameters
    ----------
    study : optuna.study.Study
        A finished (or in-progress) study.
    std_penalty : float, optional
        Weight ``lambda`` in ``adj = mean - lambda * std``. Default is ``0.0`` (plain mean).

    Returns
    -------
    optuna.trial.FrozenTrial
        The selected trial.

    Raises
    ------
    RuntimeError
        If no trial completed (every trial was pruned or failed).
    """
    completed = [
        t
        for t in study.get_trials(deepcopy=False)
        if t.state == TrialState.COMPLETE and t.value is not None
    ]
    if not completed:
        raise RuntimeError("no trial completed; every trial was pruned or failed")
    return min(completed, key=lambda t: (-_adj(t, std_penalty), _duration(t)))


def _cv_results(
    trials: list[FrozenTrial], param_names: list[str]
) -> dict[str, np.ndarray | list[dict[str, ParamValue]]]:
    """Assemble the trials into a ``GridSearchCV``-style ``cv_results_`` table.

    One row per trial (completed, pruned, and failed alike, in trial order). Pruned and failed
    trials never finished their folds, so their ``mean_test_score`` / ``std_test_score`` are
    ``NaN`` and they rank after every completed trial. Only the score and param keys are
    provided -- no per-fold ``splitN_test_score`` or timing columns; ``study_`` remains the
    richer record.
    """
    means = np.array([t.user_attrs.get("mean_test_score", np.nan) for t in trials], dtype=float)
    stds = np.array([t.user_attrs.get("std_test_score", np.nan) for t in trials], dtype=float)
    ranks = rankdata(-np.where(np.isnan(means), -np.inf, means), method="min")
    results: dict[str, np.ndarray | list[dict[str, ParamValue]]] = {
        "params": [dict(t.params) for t in trials],
        "mean_test_score": means,
        "std_test_score": stds,
        "rank_test_score": ranks.astype(np.int32),
    }
    for name in param_names:
        results[f"param_{name}"] = np.array([t.params.get(name) for t in trials], dtype=object)
    return results


def _rows(data: object, indices: np.ndarray) -> object:
    """Select ``indices`` rows from a DataFrame/Series (``iloc``) or a NumPy array."""
    iloc = getattr(data, "iloc", None)
    if iloc is not None:
        return iloc[indices]
    return np.asarray(data)[indices]


def _aligned(pred: object, truth: object) -> tuple[np.ndarray, np.ndarray, list[bool]]:
    """Return ``pred`` and ``truth`` as 2D arrays plus a per-column ``is_numeric`` mask.

    The mask is read from ``truth``: each column is numeric (scored by standardized squared error)
    or not (scored by mismatch rate). DataFrames carry per-column dtypes; a plain array is numeric
    iff its dtype is.
    """
    if isinstance(truth, pd.DataFrame):
        numeric = [bool(is_numeric_dtype(truth[col])) for col in truth.columns]
        truth_arr = truth.to_numpy()
    else:
        truth_arr = np.asarray(truth)
        numeric = [bool(np.issubdtype(truth_arr.dtype, np.number))] * truth_arr.shape[1]
    pred_arr = pred.to_numpy() if isinstance(pred, pd.DataFrame) else np.asarray(pred)
    return pred_arr, truth_arr, numeric


def _missing_mask(X: object) -> np.ndarray:  # noqa: N803
    """Boolean mask of the imputed cells -- where the input ``X`` was missing (``NaN``/``None``)."""
    values = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    return np.asarray(pd.isna(values))


def _reconstruction_error(pred: object, truth: object, mask: np.ndarray) -> float:
    """Mean per-column reconstruction error over the imputed cells (where ``mask`` is True).

    Only columns with at least one imputed cell count, and only their masked cells -- so a fully
    observed column riding along in the frame is skipped, not averaged in as a zero that dilutes the
    score. Numeric columns contribute the mean standardized squared error (dividing by the column's
    std so scales are comparable); non-numeric columns contribute the mismatch rate.
    """
    pred_arr, truth_arr, numeric = _aligned(pred, truth)
    errors: list[float] = []
    for col, is_numeric in enumerate(numeric):
        col_mask = mask[:, col]
        if not col_mask.any():
            continue
        p, t = pred_arr[col_mask, col], truth_arr[col_mask, col]
        if is_numeric:
            std = float(np.std(truth_arr[:, col].astype(float))) or 1.0
            errors.append(float(np.mean(((p.astype(float) - t.astype(float)) / std) ** 2)))
        else:
            errors.append(float(np.mean(p != t)))
    return float(np.mean(errors)) if errors else 0.0


def reconstruction_scorer(estimator: _Estimator, X: object, y: object) -> float:  # noqa: N803
    """Score an imputer by how well it reconstructs known values (higher is better).

    A scorer with the ``(estimator, X, y) -> float`` signature :class:`PruningSearchCV` accepts as
    ``scoring``: it imputes ``X``'s missing cells with ``estimator.transform`` and scores them
    against the ground truth ``y`` -- numeric columns by standardized squared error, non-numeric by
    mismatch rate, averaged over the columns that had an imputed cell (see
    :func:`_reconstruction_error`). Pass it as ``scoring=reconstruction_scorer`` and call the search
    as ``fit(X_corrupted, X_truth)``.

    Parameters
    ----------
    estimator : object
        A fitted transformer (e.g. an imputer) exposing ``transform``.
    X : array-like or pandas.DataFrame
        Rows with missing cells to impute; the imputed mask is read from here (``NaN``/``None``).
    y : array-like or pandas.DataFrame
        Ground truth aligned with ``X``.

    Returns
    -------
    float
        Negative dtype-aware reconstruction error over the imputed cells.
    """
    return -_reconstruction_error(estimator.transform(X), y, _missing_mask(X))


def _delegates(attr: str) -> Callable[[PruningSearchCV], bool]:
    """Return an ``available_if`` check: does the estimator to delegate to expose ``attr``?

    Consults the refit ``best_estimator_`` once present, else the base ``estimator`` -- so the
    method's availability tracks the estimator's capability before and after ``fit``. Used to expose
    ``predict`` only for predictors and ``transform`` only for transformers.
    """

    def check(search: PruningSearchCV) -> bool:
        return hasattr(getattr(search, "best_estimator_", search.estimator), attr)

    return check


class PruningSearchCV(MetaEstimatorMixin, BaseEstimator):
    """Optuna search that prunes weak trials per fold and selects by a variance penalty.

    Each trial fine-tunes ``estimator`` over the CV folds one at a time; after every fold the
    running mean is reported to the trial and the study's pruner may stop it early. The objective
    Optuna maximizes is the plain mean CV score, but *selection* (``best_trial_`` / ``best_params_``
    and the optional refit) ranks completed trials by ``mean - std_penalty * std``, breaking ties
    toward the faster trial.

    To tune a transformer (e.g. an imputer) by reconstruction, pass ``scoring=``
    :func:`reconstruction_scorer` and call ``fit(X_corrupted, X_truth)``: each fold fits on the
    corrupted train rows, transforms the corrupted test rows, and scores the imputed cells (those
    missing in ``X``) against the ground truth passed as ``y`` (higher is better). The search owns
    its CV loop, so this needs no pipeline or passthrough wrapper. ``predict`` and ``transform`` are
    exposed only when the refit estimator supports them.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        Base estimator, transformer, or pipeline cloned for every trial.
    param_distributions : dict[str, optuna.distributions.BaseDistribution]
        Search space, keyed by ``set_params`` path (e.g. ``"lm__training__epochs"``).
        Categorical choices may be arbitrary objects (e.g. ``LRScheduler`` shapes): they are
        recorded in the study by ``repr`` and resolved back to a fresh deep copy of the choice
        at each ``set_params``, so choices must have unique reprs.
    cv : int or sklearn.model_selection.BaseCrossValidator, optional
        Number of folds (an ``int`` becomes a shuffled ``StratifiedKFold`` for classifiers,
        ``KFold`` otherwise -- shuffling matters because fold order drives pruning) or a splitter.
        Default is ``5``.
    scoring : str, callable or None, optional
        A ``(estimator, X, y) -> float`` callable (higher is better) is used directly -- the seam
        for scoring a transformer by reconstruction, e.g. :func:`reconstruction_scorer`. A string or
        ``None`` is passed to :func:`sklearn.metrics.check_scoring` (``None`` uses
        ``estimator.score``).
    n_trials : int, optional
        Number of trials. Default is ``20``.
    std_penalty : float, optional
        Weight ``lambda`` in ``adj = mean - lambda * std`` used for selection. Default ``0.0``.
    sampler : optuna.samplers.BaseSampler or None, optional
        Sampler. Default is ``TPESampler(seed=random_state)``.
    pruner : optuna.pruners.BasePruner or None, optional
        Pruner. Default is ``MedianPruner()``.
    callbacks : list[StudyCallback] or None, optional
        Optuna study callbacks invoked after each finished trial (e.g. :class:`LiveLeaderboard`).
    refit : bool, optional
        If ``True`` (default), refit the selected config on the full data into ``best_estimator_``.
    catch : tuple[type[Exception], ...], optional
        Exception types that mark the raising trial ``FAILED`` instead of aborting the search
        (forwarded to ``study.optimize``). The failed trial records the message under
        ``user_attrs["error"]``, carries ``NaN`` scores in ``cv_results_``, and is never
        selected. Default is ``()`` -- any error propagates.
    random_state : int or None, optional
        Seed for the default sampler and the shuffled splitter.

    Attributes
    ----------
    study_ : optuna.study.Study
        The optimized study (every trial, including pruned ones).
    scorer_ : Scorer
        The resolved ``(estimator, X, y) -> float`` scorer the search optimized.
    cv_results_ : dict[str, numpy.ndarray or list]
        ``GridSearchCV``-style table, one row per trial (pruned and failed trials carry ``NaN``
        scores and rank last). Keys: ``params``, ``param_<name>``, ``mean_test_score``,
        ``std_test_score``, ``rank_test_score``. ``rank_test_score`` ranks by the plain mean,
        so with ``std_penalty > 0`` the selected ``best_index_`` need not be rank 1.
    best_index_ : int
        Row of ``cv_results_`` holding the selected trial.
    best_trial_ : optuna.trial.FrozenTrial
        The selected trial.
    best_params_ : dict[str, ParamValue]
        ``best_trial_.params``.
    best_score_ : float
        The selected trial's risk-adjusted score.
    best_estimator_ : sklearn.base.BaseEstimator
        Refit estimator on the full data. Present only when ``refit`` is ``True``.
    """

    study_: Study
    scorer_: Scorer
    cv_results_: dict[str, np.ndarray | list[dict[str, ParamValue]]]
    best_index_: int
    best_trial_: FrozenTrial
    best_params_: dict[str, ParamValue]
    best_score_: float
    best_estimator_: _Estimator

    def __init__(
        self,
        estimator: BaseEstimator,
        param_distributions: Mapping[str, BaseDistribution],
        *,
        cv: int | BaseCrossValidator = 5,
        scoring: str | Scorer | None = None,
        n_trials: int = 20,
        std_penalty: float = 0.0,
        sampler: BaseSampler | None = None,
        pruner: BasePruner | None = None,
        callbacks: list[StudyCallback] | None = None,
        refit: bool = True,
        catch: tuple[type[Exception], ...] = (),
        random_state: int | None = None,
    ) -> None:
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.cv = cv
        self.scoring = scoring
        self.n_trials = n_trials
        self.std_penalty = std_penalty
        self.sampler = sampler
        self.pruner = pruner
        self.callbacks = callbacks
        self.refit = refit
        self.catch = catch
        self.random_state = random_state

    def _make_splitter(self) -> BaseCrossValidator:
        if isinstance(self.cv, int):
            factory = StratifiedKFold if is_classifier(self.estimator) else KFold
            return factory(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        return self.cv

    def fit(self, X: object, y: object = None, **fit_params: object) -> Self:
        """Run the search and select (and optionally refit) the best configuration.

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target values.
        **fit_params : object
            Extra keyword arguments forwarded to each ``estimator.fit`` call.

        Returns
        -------
        Self
            The fitted search.

        Raises
        ------
        RuntimeError
            If every trial was pruned or failed, so no configuration could be selected.
        """
        # A string or ``None`` goes through ``check_scoring`` (the cast pins its broadly-typed
        # return to the concrete (estimator, X, y) -> float call the loop makes); any other value is
        # a scorer callable used as-is -- the seam for scoring a transformer by reconstruction
        # (e.g. ``reconstruction_scorer``), with no special mode in the search.
        if isinstance(self.scoring, str) or self.scoring is None:
            scorer: Scorer = cast("Scorer", check_scoring(self.estimator, scoring=self.scoring))
        else:
            scorer = self.scoring
        splitter = self._make_splitter()
        sampler = self.sampler if self.sampler is not None else TPESampler(seed=self.random_state)
        pruner = self.pruner if self.pruner is not None else MedianPruner()
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

        def objective(trial: Trial) -> float:
            params = {
                name: _suggest(trial, name, dist) for name, dist in self.param_distributions.items()
            }
            estimator = _configured(self.estimator, _resolved(params, self.param_distributions))
            scores: list[float] = []
            try:
                for step, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
                    estimator.fit(_rows(X, train_idx), _rows(y, train_idx), **fit_params)
                    scores.append(float(scorer(estimator, _rows(X, test_idx), _rows(y, test_idx))))
                    trial.report(float(np.mean(scores)), step)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            except optuna.TrialPruned:
                raise
            except self.catch as exc:
                trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
                raise
            trial.set_user_attr("mean_test_score", float(np.mean(scores)))
            trial.set_user_attr("std_test_score", float(np.std(scores)))
            return float(np.mean(scores))

        study.optimize(
            objective, n_trials=self.n_trials, callbacks=self.callbacks, catch=self.catch
        )
        self.study_ = study
        self.scorer_ = scorer
        self.best_trial_ = best_trial(study, std_penalty=self.std_penalty)
        self.best_params_ = self.best_trial_.params
        self.best_score_ = _adj(self.best_trial_, self.std_penalty)
        rows = [
            t
            for t in study.get_trials(deepcopy=False)
            if t.state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
        ]
        self.cv_results_ = _cv_results(rows, list(self.param_distributions))
        self.best_index_ = next(
            i for i, t in enumerate(rows) if t.number == self.best_trial_.number
        )
        if self.refit:
            best = _configured(
                self.estimator, _resolved(self.best_params_, self.param_distributions)
            )
            best.fit(X, y, **fit_params)
            self.best_estimator_ = best
        return self

    def score(self, X: object, y: object = None) -> float:
        """Score ``X`` (and ``y``) on the refit ``best_estimator_`` with the search's scorer.

        Uses the same ``scoring`` the search optimized, so nesting the search inside
        ``cross_val_score`` / ``cross_validate`` without an explicit outer ``scoring`` works.

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape (n_samples, n_features)
            Samples to score.
        y : array-like of shape (n_samples,), optional
            Target values, forwarded to the scorer.

        Returns
        -------
        float
            The scorer's value (higher is better).

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the search was not fitted with ``refit=True``.
        """
        check_is_fitted(self, "best_estimator_")
        return float(self.scorer_(self.best_estimator_, X, y))

    @property
    def classes_(self) -> np.ndarray:
        """Class labels of the refit ``best_estimator_``. Available only for classifiers.

        Raises ``AttributeError`` before the refit (``NotFittedError`` subclasses it) or when the
        refit estimator has no ``classes_``, so ``hasattr`` stays honest for meta-tools probing
        the search.
        """
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.classes_

    @available_if(_delegates("predict"))
    def predict(self, X: object) -> np.ndarray:
        """Predict with the refit ``best_estimator_``.

        Available only when the estimator is a predictor.

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        numpy.ndarray
            Predictions from ``best_estimator_``.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the search was not fitted with ``refit=True``.
        """
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.predict(X)

    @available_if(_delegates("predict_proba"))
    def predict_proba(self, X: object) -> np.ndarray:
        """Predict class probabilities with the refit ``best_estimator_``.

        Available only when the estimator exposes ``predict_proba``.

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_classes)
            Per-row probabilities over ``classes_``.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the search was not fitted with ``refit=True``.
        """
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.predict_proba(X)

    @available_if(_delegates("transform"))
    def transform(self, X: object) -> object:
        """Transform ``X`` with the refit ``best_estimator_``.

        Available only when the estimator is a transformer (e.g. an imputer).

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape (n_samples, n_features)
            Samples to transform (e.g. rows with missing cells to impute).

        Returns
        -------
        object
            The transformer's output (e.g. the imputed frame).

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the search was not fitted with ``refit=True``.
        """
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.transform(X)


def _leaf(name: str) -> str:
    """Return the trailing field of a nested ``__`` parameter path, for a column header."""
    return name.rsplit("__", 1)[-1]


def _fmt(value: object) -> str:
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def _params_cell(trial: FrozenTrial, keys: list[str]) -> str:
    """A trial's params as one cell: space-joined ``leaf=value`` pairs that wrap across lines."""
    return "  ".join(f"{_leaf(k)}={_fmt(trial.params.get(k))}" for k in keys)


def _maybe_clear() -> None:
    """Clear the cell's previous output when running under an IPython kernel, else do nothing."""
    try:
        from IPython.core.getipython import get_ipython
    except ImportError:
        return
    if get_ipython() is None:
        return
    from IPython.display import clear_output

    clear_output(wait=True)


class LiveLeaderboard:
    """Optuna study callback that redraws one leaderboard table in place after each trial.

    Completed trials are ranked by ``adj = mean - std_penalty * std`` (then by wall-clock) with the
    top row highlighted; pruned trials are listed dim, showing the running mean and the fold they
    stopped at; failed trials (a search with ``catch``) are listed dim red with their error.
    Under an IPython kernel the table is redrawn in place (``clear_output``); elsewhere
    each call simply prints. Requires the ``rich`` extra.

    Parameters
    ----------
    std_penalty : float, optional
        Weight ``lambda`` used for the ``adj`` column and the ranking. Default is ``0.0``.
    """

    def __init__(self, *, std_penalty: float = 0.0) -> None:
        try:
            from rich.console import Console
        except ImportError as exc:
            raise ImportError(
                "LiveLeaderboard requires the 'rich' extra: pip install scikit-lm[rich]"
            ) from exc
        self.std_penalty = std_penalty
        self._console = Console()

    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        from rich.table import Table

        trials = study.get_trials(deepcopy=False)
        done = [t for t in trials if t.state == TrialState.COMPLETE and t.value is not None]
        pruned = [t for t in trials if t.state == TrialState.PRUNED]
        failed = [t for t in trials if t.state == TrialState.FAIL]
        if not done and not pruned and not failed:
            return
        done.sort(key=lambda t: (-_adj(t, self.std_penalty), _duration(t)))
        keys = list((done or pruned or failed)[0].params)

        table = Table(title=f"Trials - adj = mean - {self.std_penalty:g}*std (then speed)")
        for column in ("rank", "trial", "score+/-std", "adj", "time"):
            table.add_column(column, justify="right")
        table.add_column("params", overflow="fold", max_width=48)

        for rank, t in enumerate(done, 1):
            attrs = t.user_attrs
            table.add_row(
                str(rank),
                str(t.number),
                f"{attrs['mean_test_score']:.4f}+/-{attrs['std_test_score']:.3f}",
                f"{_adj(t, self.std_penalty):.4f}",
                f"{_duration(t):.0f}s",
                _params_cell(t, keys),
                style="green" if rank == 1 else None,
            )
        for t in sorted(pruned, key=lambda t: t.number):
            step = max(t.intermediate_values, default=-1)
            last = t.intermediate_values.get(step, float("nan"))
            table.add_row(
                "-",
                str(t.number),
                f"{last:.4f}",
                f"pruned@f{step}",
                f"{_duration(t):.0f}s",
                _params_cell(t, keys),
                style="dim",
            )
        for t in sorted(failed, key=lambda t: t.number):
            error = str(t.user_attrs.get("error", "-"))
            table.add_row(
                "-",
                str(t.number),
                error if len(error) <= 48 else error[:47] + "…",
                "failed",
                f"{_duration(t):.0f}s",
                _params_cell(t, keys),
                style="red dim",
            )

        _maybe_clear()
        self._console.print(table)
