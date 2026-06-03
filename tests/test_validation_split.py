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

from sklm import LanguageModelClassifier, LanguageModelRegressor, TrainingConfig
from sklm.core import _split_indices, _strata

from .conftest import FakeBackend

# --- config validation ----------------------------------------------------


def test_validation_split_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="validation_split"):
        TrainingConfig(validation_split=1.0)
    with pytest.raises(ValueError, match="validation_split"):
        TrainingConfig(validation_split=-0.1)


def test_checkpoint_steps_must_be_positive() -> None:
    with pytest.raises(ValueError, match="checkpoint_steps"):
        TrainingConfig(checkpoint_steps=0)


def test_early_stopping_requires_validation_split() -> None:
    with pytest.raises(ValueError, match="validation_split > 0"):
        TrainingConfig(early_stopping_patience=3)


def test_early_stopping_with_split_is_valid() -> None:
    cfg = TrainingConfig(validation_split=0.2, early_stopping_patience=3)
    assert cfg.early_stopping_patience == 3


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
        training=TrainingConfig(epochs=1, validation_split=0.25),
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
        model="m", backend=fake, training=TrainingConfig(epochs=1), random_state=0
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
            epochs=1, validation_split=0.25, augmentation_factor=4, stratify=False
        ),
        random_state=0,
    )
    reg.fit(X, y)
    assert fake.last_eval_examples is not None and fake.last_examples is not None
    # augmentation_factor lifts the train set, but the eval set stays one row.
    assert len(fake.last_eval_examples) == 5
    assert len(fake.last_examples) == 15 * 4
