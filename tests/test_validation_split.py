"""Validation hold-out: config validation, the row split, and its plumbing.

The split is backend-agnostic, so it is exercised once against the deterministic
:class:`FakeBackend`: the held-out rows reach the backend as ``eval_examples``
while training sees the rest. Stratification and its fallback are checked on the
free helpers directly.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from sklm import (
    CheckpointConfig,
    EvalConfig,
    LanguageModelClassifier,
    LanguageModelRegressor,
    TrainingConfig,
)
from sklm.backend import EarlyStopping
from sklm.core import _split_indices, _strata

from .conftest import FakeBackend

# --- config validation ----------------------------------------------------


def test_eval_config_defaults() -> None:
    ev = EvalConfig()
    assert (ev.split, ev.stratify, ev.each, ev.on, ev.patience) == (0.1, True, 1, "epoch", 5)


@pytest.mark.parametrize("split", [0.0, 1.0, -0.1])
def test_eval_split_must_be_a_proper_fraction(split: float) -> None:
    with pytest.raises(ValueError, match="split"):
        EvalConfig(split=split)


def test_eval_each_must_be_positive() -> None:
    with pytest.raises(ValueError, match="each"):
        EvalConfig(each=0)


def test_eval_patience_must_be_positive_or_none() -> None:
    with pytest.raises(ValueError, match="patience"):
        EvalConfig(patience=0)
    assert EvalConfig(patience=None).patience is None


def test_eval_nested_set_params_recurses() -> None:
    clf = LanguageModelClassifier(training=TrainingConfig(evaluation=EvalConfig(each=2, on="step")))
    clf.set_params(training__evaluation__patience=9)
    ev = clf.training.evaluation
    assert ev is not None and (ev.patience, ev.each, ev.on) == (9, 2, "step")
    cloned = clone(clf)
    assert isinstance(cloned, LanguageModelClassifier)
    cloned_ev = cloned.training.evaluation
    assert cloned_ev is not None and cloned_ev.on == "step"


def test_checkpoint_config_defaults() -> None:
    ckpt = CheckpointConfig()
    assert ckpt.each == 1
    assert ckpt.on == "step"
    assert ckpt.dir is None
    assert ckpt.keep == 1


def test_checkpoint_each_must_be_positive() -> None:
    with pytest.raises(ValueError, match="each"):
        CheckpointConfig(each=0)


def test_checkpoint_keep_must_be_positive_or_none() -> None:
    with pytest.raises(ValueError, match="keep"):
        CheckpointConfig(keep=0)
    assert CheckpointConfig(keep=None).keep is None


def test_checkpoint_nested_set_params_recurses() -> None:
    clf = LanguageModelClassifier(
        training=TrainingConfig(checkpoint=CheckpointConfig(each=2, on="epoch"))
    )
    clf.set_params(training__checkpoint__each=4)
    ckpt = clf.training.checkpoint
    assert ckpt is not None and ckpt.each == 4
    cloned = clone(clf)
    assert isinstance(cloned, LanguageModelClassifier)
    cloned_ckpt = cloned.training.checkpoint
    assert cloned_ckpt is not None and cloned_ckpt.on == "epoch"


# --- early stopping -------------------------------------------------------


def test_early_stopping_exhausts_after_patience_misses() -> None:
    stopping = EarlyStopping(patience=2)
    verdicts = [stopping.observe(loss) for loss in [0.9, 0.5, 0.6, 0.7]]
    assert verdicts == ["improved", "improved", "no_improvement", "exhausted"]
    assert (stopping.best, stopping.streak) == (0.5, 2)


def test_early_stopping_improvement_resets_streak() -> None:
    stopping = EarlyStopping(patience=2)
    verdicts = [stopping.observe(loss) for loss in [0.9, 0.9, 0.4, 0.5]]
    assert verdicts == ["improved", "no_improvement", "improved", "no_improvement"]


def test_early_stopping_without_patience_never_exhausts() -> None:
    stopping = EarlyStopping(patience=None)
    assert [stopping.observe(1.0) for _ in range(5)][1:] == ["no_improvement"] * 4


def test_early_stopping_resumes_from_seeded_trackers() -> None:
    stopping = EarlyStopping(patience=3, best=0.5, streak=2)
    assert stopping.observe(0.6) == "exhausted"


# --- stratification key ----------------------------------------------------


def test_strata_bins_numeric_with_many_uniques() -> None:
    key = _strata(pd.Series(np.arange(40, dtype=float)), n_bins=10)
    assert len(key) == 40
    assert len(set(key.tolist())) == 10


def test_strata_passes_categorical_through() -> None:
    s = pd.Series(["a", "b", "a", "c"])
    np.testing.assert_array_equal(_strata(s), np.array(["a", "b", "a", "c"], dtype=object))


def test_strata_keeps_low_cardinality_numeric_verbatim() -> None:
    s = pd.Series([0, 1, 0, 1, 0], dtype=int)
    np.testing.assert_array_equal(_strata(s, n_bins=10), s.to_numpy())


# --- the split ------------------------------------------------------------


def _balanced_frame() -> pd.DataFrame:
    return pd.DataFrame({"x": range(20), "t": ["a", "b"] * 10})


def test_split_partitions_all_rows_disjointly() -> None:
    train, ev = _split_indices(_balanced_frame(), frozenset({"t"}), 0.25, True, 0)
    assert len(ev) == math.ceil(0.25 * 20)
    assert set(train).isdisjoint(ev)
    assert sorted(train + ev) == list(range(20))


def test_split_is_deterministic_under_seed() -> None:
    a = _split_indices(_balanced_frame(), frozenset({"t"}), 0.25, True, 0)
    b = _split_indices(_balanced_frame(), frozenset({"t"}), 0.25, True, 0)
    assert a == b


def test_split_stratifies_on_single_target() -> None:
    frame = _balanced_frame()
    _, ev = _split_indices(frame, frozenset({"t"}), 0.25, True, 0)
    labels = frame.loc[ev, "t"]
    assert set(labels) == {"a", "b"}


def test_split_falls_back_to_random_when_class_too_rare() -> None:
    frame = pd.DataFrame({"x": range(10), "t": ["a"] * 9 + ["rare"]})
    with pytest.warns(RuntimeWarning, match="stratified"):
        train, ev = _split_indices(frame, frozenset({"t"}), 0.25, True, 0)
    assert sorted(train + ev) == list(range(10))


def test_split_does_not_stratify_with_multiple_targets() -> None:
    frame = _balanced_frame()
    train, ev = _split_indices(frame, frozenset({"x", "t"}), 0.25, True, 0)
    assert sorted(train + ev) == list(range(20))


def test_split_does_not_stratify_target_with_missing() -> None:
    frame = pd.DataFrame({"t": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    train, ev = _split_indices(frame, frozenset({"t"}), 0.25, True, 0)
    assert sorted(train + ev) == list(range(8))


# --- core.fit plumbing (through the classifier + fake backend) -------------


def test_fit_holds_out_eval_examples(clf_data) -> None:
    X, y = clf_data  # 20 rows
    fake = FakeBackend()
    clf = LanguageModelClassifier(
        model="m",
        backend=fake,
        training=TrainingConfig(epochs=1, evaluation=EvalConfig(split=0.25), augmentation_factor=1),
        random_state=0,
    )
    clf.fit(X, y)
    assert fake.last_eval_examples is not None and fake.last_examples is not None
    assert len(fake.last_eval_examples) == 5
    assert len(fake.last_examples) == 15


def test_fit_without_split_passes_no_eval(clf_data) -> None:
    X, y = clf_data
    fake = FakeBackend()
    clf = LanguageModelClassifier(
        model="m",
        backend=fake,
        training=TrainingConfig(epochs=1, evaluation=None, augmentation_factor=1),
        random_state=0,
    )
    clf.fit(X, y)
    assert fake.last_eval_examples is None
    assert fake.last_examples is not None
    assert len(fake.last_examples) == 20


def test_fit_eval_examples_single_permutation_per_row(reg_data) -> None:
    X, y = reg_data  # 20 rows, regressor target binned for stratification
    fake = FakeBackend()
    reg = LanguageModelRegressor(
        model="m",
        backend=fake,
        training=TrainingConfig(
            epochs=1,
            evaluation=EvalConfig(split=0.25, stratify=False),
            augmentation_factor=4,
            # the target must permute with the rest for 3 columns to reach 4 orders
            target_loss_weight=None,
        ),
        random_state=0,
    )
    reg.fit(X, y)
    assert fake.last_eval_examples is not None and fake.last_examples is not None
    # augmentation_factor lifts the train set, but the eval set stays one row.
    assert len(fake.last_eval_examples) == 5
    assert len(fake.last_examples) == 15 * 4
