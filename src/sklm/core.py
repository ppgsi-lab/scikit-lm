"""The shared fitted model behind every estimator.

``TabularLanguageModel`` fine-tunes one backend on serialized rows with dynamic
feature-order permutation, then exposes the conditional primitives:

- :meth:`complete` -- open-ended generation of target columns given known ones
  (imputer, oversampler, regressor).
- :meth:`predict_proba` -- rank a fixed candidate set by likelihood (classifier).
- :meth:`sample` -- draw whole rows from the learned joint distribution,
  optionally holding columns fixed (tabular synthesis).
"""

from __future__ import annotations

import contextlib
import math
import os
import warnings
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from itertools import pairwise, permutations
from typing import Literal, Self

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.utils import Tags
from sklearn.utils.validation import check_array, check_is_fitted

from .backend import LanguageModelBackend, prompt_groups
from .callbacks import Callback
from .config import DiscretizationConfig, GenerationConfig, ModelConfig, TrainingConfig
from .hf_backend import HFBackend
from .serialize import (
    Field,
    JSONSerializer,
    NumericSpan,
    Serializer,
    SpacedDigits,
    TrainingExample,
    is_missing,
)

__all__ = ["TabularLanguageModel", "forget", "select_candidates", "to_frame"]

_ENUMERATE_CAP = 5040  # 7!

# Per-epoch serialized rows surfaced to the callback for preview; the full epoch
# list is already in memory, so this only bounds how many are handed over.
_TRAIN_PREVIEW = 16


def forget(obj: object, name: str) -> None:
    """Delete attribute ``name`` from ``obj`` if present (no-op otherwise)."""
    with contextlib.suppress(AttributeError):
        delattr(obj, name)


def to_frame(X: object, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Coerce array-like input to a DataFrame, naming columns ``x0..`` (or with
    ``columns``) when the input is not already a DataFrame.

    Non-DataFrame input is validated with scikit-learn's ``check_array`` in a mode
    that still admits string/categorical cells and NaN (``dtype=None``,
    ``ensure_all_finite=False``), so sparse, complex, empty, and 1-D inputs are
    rejected with scikit-learn's standard messages while text tables pass through.
    A DataFrame skips ``check_array`` but is still rejected when it has 0 rows.
    """
    if isinstance(X, pd.DataFrame):
        if len(X) == 0:
            raise ValueError(
                f"Found array with 0 sample(s) (shape={X.shape}) while a minimum of 1 is required."
            )
        return X
    # dtype=None keeps string/object cells & NaN; sklearn infers check_array's dtype as str
    check_array(X, dtype=None, ensure_all_finite=False)  # pyright: ignore[reportArgumentType]
    arr = np.asarray(X)
    names = list(columns) if columns is not None else [f"x{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=pd.Index(names))


def _distinct_orders(columns: list[str], k: int, rng: np.random.Generator) -> list[list[str]]:
    """Up to ``k`` distinct orderings of ``columns`` (capped at ``len(columns)!``)."""
    n = len(columns)
    if k <= 1 or n <= 1:
        order = list(columns)
        rng.shuffle(order)
        return [order]
    if n <= 7:  # n! <= 5040 -- enumerate and sample without replacement
        perms = [list(p) for p in permutations(columns)]
        idx = rng.permutation(len(perms))[: min(k, len(perms))]
        return [perms[i] for i in idx]
    seen: set[tuple[str, ...]] = set()
    out: list[list[str]] = []
    while len(out) < k:  # n >= 8 => k << n!, rejection collisions negligible
        p = tuple(rng.permutation(columns).tolist())
        if p not in seen:
            seen.add(p)
            out.append(list(p))
    return out


def _distinct_block_orders(
    ctx: list[str], tgt: list[str], k: int, rng: np.random.Generator
) -> list[tuple[list[str], list[str]]]:
    """Up to ``k`` distinct ``(context_order, target_order)`` pairs.

    Each block is permuted independently, so the distinct count caps at
    ``|ctx|! * |tgt|!`` (empty blocks contribute a single empty order).
    """
    max_unique = math.factorial(len(ctx)) * math.factorial(len(tgt))
    k = min(k, max_unique)
    if k <= 1:
        c, t = list(ctx), list(tgt)
        rng.shuffle(c)
        rng.shuffle(t)
        return [(c, t)]
    if max_unique <= _ENUMERATE_CAP:  # enumerate and sample without replacement
        pairs = [(list(c), list(t)) for c in permutations(ctx) for t in permutations(tgt)]
        idx = rng.permutation(len(pairs))[:k]
        return [pairs[i] for i in idx]
    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    out: list[tuple[list[str], list[str]]] = []
    while len(out) < k:
        c = tuple(rng.permutation(ctx).tolist()) if ctx else ()
        t = tuple(rng.permutation(tgt).tolist()) if tgt else ()
        if (c, t) not in seen:
            seen.add((c, t))
            out.append((list(c), list(t)))
    return out


def _strata(values: pd.Series, n_bins: int = 10) -> np.ndarray:
    """Discrete stratification key for ``values``.

    Numeric columns with more than ``n_bins`` distinct values are binned into
    quantiles (so a regression target stratifies like a set of class labels);
    everything else is used verbatim. This makes the hold-out's stratification
    target-type agnostic -- one rule covers classification and regression.
    """
    if pd.api.types.is_numeric_dtype(values) and values.nunique() > n_bins:
        return np.asarray(pd.qcut(values, q=n_bins, duplicates="drop", labels=False))
    return values.to_numpy()


def _split_indices(
    frame: pd.DataFrame,
    target_cols: frozenset[str],
    split: float,
    stratify: bool,
    random_state: int | None,
) -> tuple[list[int], list[int]]:
    """Partition ``frame``'s row indices into ``(train, eval)`` for the hold-out.

    Stratifies on the sole target column when ``stratify`` is set and that column
    is fully observed -- binning it if numeric -- so classifier labels and
    regressor targets are both balanced across the split; otherwise splits at
    random. Falls back to a random split (with a warning) when a stratified split
    is infeasible, e.g. a class with too few members for both sides.
    """
    indices = np.arange(len(frame))
    strata: np.ndarray | None = None
    if stratify and len(target_cols) == 1:
        (col,) = target_cols
        column = frame[col]
        if isinstance(column, pd.Series) and bool(column.notna().all()):
            strata = _strata(column)
    try:
        train_idx, eval_idx = train_test_split(
            indices, test_size=split, random_state=random_state, stratify=strata
        )
    except ValueError:
        if strata is None:
            raise
        warnings.warn(
            "stratified validation split was infeasible (a class has too few "
            "members); falling back to a random split.",
            RuntimeWarning,
            stacklevel=2,
        )
        train_idx, eval_idx = train_test_split(
            indices, test_size=split, random_state=random_state, stratify=None
        )
    return np.asarray(train_idx).tolist(), np.asarray(eval_idx).tolist()


def _native(value: object) -> object:
    """Unbox a NumPy scalar into its native Python equivalent.

    Inference inputs (knowns, candidates, sample conditions) arrive from user
    code and often carry NumPy scalars; serializers must see native types
    (``repr(np.float64(1.5))`` is ``"np.float64(1.5)"`` on NumPy >= 2)."""
    return value.item() if isinstance(value, np.generic) else value


def _noisy(value: object, sigma: float, rng: np.random.Generator) -> object:
    """Perturb one numeric cell with Gaussian noise at the cell's own precision.

    The perturbed value must serialize exactly like the original would: an
    integer cell stays an integer and a float cell keeps its decimal count, so
    ``numeric_noise`` augments the value without changing the number format the
    model sees. A float whose repr is scientific notation has no decimal count
    to preserve and is left unperturbed.
    """
    v = value.item() if isinstance(value, np.generic) else value
    if not isinstance(v, int | float):
        return value
    draw = float(rng.normal(0.0, sigma))
    if isinstance(v, int):
        return round(v + draw)
    text = repr(v)
    if "e" in text or "E" in text:
        return value
    decimals = len(text.split(".")[1]) if "." in text else 0
    return round(v + draw, decimals)


def _decimal_places(value: object) -> int:
    """Decimal places the value's serialized text carries (0 for integers).

    ``Decimal(repr(v))`` rather than splitting on ``"."``: a scientific-notation
    repr (``repr(1e-05) == "1e-05"``) has no point to split on but still encodes
    five decimal places.
    """
    v = value.item() if isinstance(value, np.generic) else value
    if isinstance(v, int):
        return 0
    exponent = Decimal(repr(v)).as_tuple().exponent
    return max(0, -exponent) if isinstance(exponent, int) else 0


@dataclass(frozen=True, slots=True)
class _ScoreSpec:
    """How to fill one scored column: candidates to rank and a reducer.

    Lets :meth:`TabularLanguageModel.impute_many` stay agnostic of what makes a
    column discrete -- the imputer injects, per column, the candidate set (a
    numeric grid or a categorical's observed levels) and a
    ``(proba_row, candidates) -> value`` reduction closure (the numeric estimate
    or a categorical argmax).
    """

    candidates: Sequence[object]
    reduce: Callable[[np.ndarray, Sequence[object]], object]


def _normalized_entropy(proba: np.ndarray) -> float:
    """Entropy of a probability row, scaled to [0, 1] by its support size."""
    if len(proba) < 2:
        return 0.0
    positive = proba[proba > 0]
    return float(-(positive * np.log(positive)).sum() / math.log(len(proba)))


def _representative(group: np.ndarray, kind: Literal["median", "mode", "mean"]) -> float:
    """One value standing in for a partition of observed target values.

    ``"median"`` is the lower median and ``"mode"`` the most frequent value
    (ties broken by the smaller value) -- both real observed values; ``"mean"``
    is the synthetic arithmetic mean.
    """
    if kind == "mean":
        return float(np.mean(group))
    if kind == "mode":
        values, counts = np.unique(group, return_counts=True)
        return float(values[counts == counts.max()].min())
    return float(np.percentile(group, 50, method="lower"))


def select_candidates(values: np.ndarray, config: DiscretizationConfig) -> list[float]:
    """Candidate values to score, drawn from a target column's observed values.

    Non-NaN values are partitioned into :meth:`DiscretizationConfig.resolve_k`
    strata (equal-mass quantiles or equal-width bins, per ``config.strategy``)
    and one representative is taken from each (``config.representative``). When
    the requested count reaches the number of distinct values the full sorted
    support is returned and partitioning is skipped.

    Parameters
    ----------
    values : numpy.ndarray
        Observed values of the target column (may contain NaN, dropped here).
    config : DiscretizationConfig
        The active discretization settings; assumed enabled (``bins`` truthy).

    Returns
    -------
    list of float
        Sorted, de-duplicated candidate values.

    Raises
    ------
    ValueError
        If no observed (non-NaN) values remain.
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        raise ValueError("discretization target has no observed (non-NaN) values")
    unique = np.unique(v)
    k = config.resolve_k(unique.size)
    if k >= unique.size:
        return [float(x) for x in unique]
    if config.strategy == "quantile":
        labels = np.asarray(pd.qcut(v, q=k, duplicates="drop", labels=False))
    else:
        edges = np.linspace(v.min(), v.max(), k + 1)
        labels = np.clip(np.digitize(v, edges[1:-1]), 0, k - 1)
    reps = {_representative(v[labels == lbl], config.representative) for lbl in np.unique(labels)}
    return sorted(reps)


def _ordered(columns: Sequence[str], order: Sequence[str]) -> list[str]:
    """``columns`` in the order ``order`` lists them."""
    present = set(columns)
    return [c for c in order if c in present]


def _column_order(generation: GenerationConfig, columns: Sequence[str]) -> list[str] | None:
    """``generation.column_order``, checked to be a permutation of the training ``columns``."""
    order = generation.column_order
    if order is None:
        return None
    if sorted(order) != sorted(columns):
        raise ValueError(
            f"column_order must be a permutation of the training columns {list(columns)}, "
            f"got {list(order)}"
        )
    return list(order)


def _positional_order(
    orders: Sequence[Sequence[str]], hits: Sequence[Sequence[float]]
) -> list[str]:
    """Best column order under an additive position model fit to scored orders.

    ``hits[k]`` holds order ``k``'s log p(true) on the first ``len(hits[k])``
    calibration rows. Each observation contributes
    ``hit ~ sum_i w[column at i, i] + row effect``; the row effects absorb
    per-row difficulty, so orders scored on different numbers of rows stay
    comparable (an order that only saw the easy rows is not credited for it).
    ``w`` is fit by ridge regression on the indicator design and the order
    maximizing the additive total is the linear assignment of columns to
    positions -- it need not be one of the scored orders.
    """
    cols = sorted(orders[0])
    n = len(cols)
    index = {c: i for i, c in enumerate(cols)}
    positional = np.zeros((len(orders), n * n))
    for k, order in enumerate(orders):
        for pos, c in enumerate(order):
            positional[k, index[c] * n + pos] = 1.0
    who = np.concatenate([np.full(len(h), k) for k, h in enumerate(hits)])
    row = np.concatenate([np.arange(len(h)) for h in hits])
    design = np.hstack([positional[who], np.eye(max(len(h) for h in hits))[row]])
    y = np.concatenate([np.asarray(h, dtype=float) for h in hits])
    y = y - y.mean()
    ridge = 1e-3 * np.eye(design.shape[1])
    w = np.linalg.solve(design.T @ design + ridge, design.T @ y)[: n * n].reshape(n, n)
    rows, positions = linear_sum_assignment(-w)
    out = [""] * n
    for r, pos in zip(rows, positions, strict=True):
        out[pos] = cols[r]
    return out


@dataclass(frozen=True)
class _ImputeRow:
    """One row mid-imputation: its observed cells, the cells filled so far, the pending ones."""

    known: Mapping[str, object]
    filled: Mapping[str, object]
    pending: tuple[str, ...]
    alive: bool = True


def _fill_scored(
    row: _ImputeRow,
    cells: Mapping[str, np.ndarray],
    score: Mapping[str, _ScoreSpec],
    quantize: Callable[[str, object], object],
) -> _ImputeRow:
    """Fill the probed cell whose distribution is most confident (lowest entropy)."""
    col, proba = min(cells.items(), key=lambda item: _normalized_entropy(item[1]))
    return replace(
        row,
        filled={**row.filled, col: quantize(col, score[col].reduce(proba, score[col].candidates))},
        pending=tuple(c for c in row.pending if c != col),
    )


def _fill_generated(
    row: _ImputeRow, out: Mapping[str, object] | None, quantize: Callable[[str, object], object]
) -> _ImputeRow:
    """Fill the row's next pending cell with a generated value, or kill the row."""
    if out is None:
        return replace(row, alive=False)
    col = row.pending[0]
    filled = {**row.filled, col: quantize(col, out[col])}
    return replace(row, filled=filled, pending=row.pending[1:])


@dataclass(kw_only=True)
class TabularLanguageModel(BaseEstimator):
    """The fitted model shared by every estimator.

    A scikit-learn :class:`~sklearn.base.BaseEstimator` in its own right
    (clonable, ``get_params``/``set_params`` with the nested-config ``__``
    convention), in the mold of the library's generators such as
    :class:`~sklearn.neighbors.KernelDensity`: ``fit`` learns the joint
    distribution and :meth:`sample` draws rows from it.

    Parameters
    ----------
    backend : LanguageModelBackend
        Fine-tunes, generates, and scores text.
    serializer : Serializer
        Converts rows to and from text.
    training : TrainingConfig
        Fine-tuning hyperparameters handed to the backend.
    model : ModelConfig
        Model-loading configuration (including the model id) handed to the backend.
    random_state : int or None, optional
        Seed for the per-epoch column-permutation RNG, and for inference draws:
        column orders, categorical sampling and the backend's sampler (seeded
        per generation batch), so a fitted model redraws the same outputs call
        after call. ``None`` leaves all of them non-deterministic.
    max_retries : int, optional
        Generation attempts per target value before giving up. Default ``15``.
    callback : Callback, optional
        Feedback hooks; defaults to a no-op instance.

    Attributes
    ----------
    columns_ : list[str]
        Training column order, recorded at :meth:`fit`.
    numeric_cols_ : frozenset[str]
        Names of the numeric columns, recorded at :meth:`fit`.
    decimals_ : dict[str, int]
        Max decimal places among each numeric column's observed values,
        recorded at :meth:`fit` (0 for an integer column). Computed fills are
        rounded to this precision before being stored -- and so before they
        re-enter a prompt under ``cell_context="chained"``.
    categories_ : dict[str, list]
        Observed levels of each non-numeric column, recorded at :meth:`fit`;
        the candidate set used to score (rather than generate) that column.
    n_features_in_ : int
        Number of columns seen at :meth:`fit`.
    feature_names_in_ : numpy.ndarray
        Column names seen at :meth:`fit`; only set for DataFrame input.
    target_cols_ : frozenset[str]
        The ``target_cols`` given to :meth:`fit`.
    target_last_ : bool
        Whether training serialized the target cells as a tail block (implied
        by ``target_loss_weight`` or ``target_at_end`` with non-empty
        ``target_cols``). When set, inference-side order marginalization keeps
        target-column cells in a tail block, mirroring the training layout.
    """

    backend: LanguageModelBackend = field(default_factory=HFBackend)
    serializer: Serializer = field(default_factory=JSONSerializer)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    random_state: int | None = None
    max_retries: int = 15
    callback: Callback = field(default_factory=Callback)

    def __getstate__(self) -> dict[str, object]:
        # The callback is live observability (a dashboard may hold an open stream
        # or logging handler) and not part of the model state; drop it so a fitted
        # estimator stays picklable, restoring a no-op on load.
        return {**self.__dict__, "callback": Callback()}

    def fit(
        self, X: object, y: object = None, *, target_cols: frozenset[str] = frozenset()
    ) -> Self:
        """Fine-tune the backend on serialized rows.

        Each epoch re-permutes the column order per row (NaN cells are dropped),
        so the model learns to condition on any subset of columns; disable
        ``training.permute_order`` to fine-tune on the canonical column order
        alone. When ``training.numeric_noise`` is set, each example's numeric
        cells are additionally perturbed with per-epoch Gaussian noise
        (validation rows stay clean). ``training.column_dropout`` independently
        omits observed context columns while preserving every observed target.

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape (n_samples, n_features)
            Training table; numeric columns are detected via dtype. Array-like
            input gets ``x0..`` column names.
        y : ignored
            Present for scikit-learn API compatibility. The model learns the
            joint distribution over every column, so a supervised target enters
            as a column of ``X`` (the estimators append it before calling this).
        target_cols : frozenset of str, optional
            Columns to treat as targets when ``training.target_loss_weight`` or
            ``training.target_at_end`` is enabled: each row's observed target
            cells are serialized last. ``target_loss_weight`` additionally
            down-weights the preceding context in the loss (masking it out
            entirely at ``1.0``); ``target_at_end`` keeps loss on the whole
            row. Empty (default) leaves the column order fully permuted.

        Returns
        -------
        Self
            The fitted model.

        Raises
        ------
        ValueError
            If ``X`` has duplicate column names (which would make
            row serialization last-wins and break per-column indexing), or if
            ``training.numeric_loss_weight`` is active without a serializer
            rendering numbers as :class:`~sklm.SpacedDigits` (the auxiliary
            numeric loss needs one token per digit).
        """
        frame = to_frame(X)
        duplicates = frame.columns[frame.columns.duplicated()].unique().tolist()
        if duplicates:
            raise ValueError(f"duplicate column names are not supported: {duplicates}")
        if self.training.numeric_loss_weight > 0 and not isinstance(
            getattr(self.serializer, "number", None), SpacedDigits
        ):
            raise ValueError(
                "numeric_loss_weight requires a serializer rendering numbers with "
                "SpacedDigits (one token per digit), e.g. "
                "JSONSerializer(number=SpacedDigits())"
            )
        self.columns_ = list(frame.columns)
        self.numeric_cols_ = frozenset(frame.select_dtypes(include="number").columns)
        # A computed fill (a scored mean, a Monte-Carlo aggregate) re-enters the
        # prompt under chained cell context; rounding it to the column's observed
        # precision keeps that text in the distribution the model trained on
        # (``25.8``, never ``25.78333333333333``).
        self.decimals_ = {
            c: max((_decimal_places(v) for v in frame[c].dropna()), default=0)
            for c in self.numeric_cols_
        }
        # The discrete space of every non-numeric column: its observed levels,
        # scored like the classifier ranks ``classes_`` rather than generated.
        self.categories_ = {
            c: sorted((_native(v) for v in pd.unique(frame[c].dropna().to_numpy())), key=str)
            for c in self.columns_
            if c not in self.numeric_cols_
        }
        self.n_features_in_ = frame.shape[1]
        self._prior_cache_: dict[tuple[str, tuple[object, ...], str], np.ndarray] = {}
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        else:
            forget(self, "feature_names_in_")
        # sklearn convention: random_state=None is non-deterministic. Draw one base
        # seed per fit so each epoch's permutation is seeded on (base_seed, epoch) --
        # idempotent within the fit, fresh across fits when None.
        base_seed = (
            self.random_state if self.random_state is not None else int.from_bytes(os.urandom(8))
        )
        # A filled prompt marks the context/target boundary for the backend:
        # the context tokens weigh 1 - target_loss_weight in the loss (0 at 1.0).
        masking = self.training.target_loss_weight is not None and bool(target_cols)
        target_last = masking or (self.training.target_at_end and bool(target_cols))
        # The block layout is a property of the fitted model: inference-side order
        # marginalization mirrors it (target cells in a tail block) when set.
        self.target_cols_ = frozenset(target_cols)
        self.target_last_ = target_last
        permute = self.training.permute_order
        spans_on = self.training.numeric_loss_weight > 0
        # The auxiliary numeric error is squared in the column's own unit, so a
        # large-scale column would otherwise monopolize the term; 1/sigma puts
        # every column's error in standard deviations. A constant column has
        # nothing to normalize by.
        inv_scales = (
            {
                c: 1.0 / sigma if (sigma := float(frame[c].std(ddof=0))) > 0 else 1.0
                for c in self.numeric_cols_
            }
            if spans_on
            else {}
        )
        # numeric_noise is a fraction of each column's own sigma, so one knob
        # covers heterogeneous units; a constant column has no scale to noise by.
        noise_scales = (
            {
                c: self.training.numeric_noise * sigma
                for c in self.numeric_cols_
                if (sigma := float(frame[c].std(ddof=0))) > 0
            }
            if self.training.numeric_noise > 0
            else {}
        )

        def _noisy_row(row: Mapping[str, object], rng: np.random.Generator) -> Mapping[str, object]:
            if not noise_scales:
                return row
            return {
                c: _noisy(v, noise_scales[c], rng) if c in noise_scales and not is_missing(v) else v
                for c, v in row.items()
            }

        def _ordered_fields(row: Mapping[str, object], order: Sequence[str]) -> list[Field]:
            return [Field(c, row[c], c in self.numeric_cols_) for c in order]

        def _numeric_spans(
            fields: list[Field], text: str, prompt_len: int
        ) -> tuple[NumericSpan, ...]:
            if not spans_on:
                return ()
            # The true value is what the digits encode (max_decimals applied),
            # not the raw cell -- the auxiliary loss compares against the text.
            return tuple(
                NumericSpan(start, end, float(text[start:end].replace(" ", "")), inv_scales[f.name])
                for f, (start, end) in zip(fields, self.serializer.value_spans(fields), strict=True)
                if f.numeric and start >= prompt_len
            )

        def row_examples(
            row: Mapping[str, object], rng: np.random.Generator, aug: int, *, noisy: bool = True
        ) -> list[TrainingExample]:
            present = [c for c in self.columns_ if not is_missing(row[c])]
            observed_targets = [c for c in present if c in target_cols]
            observed_context = [c for c in present if c not in target_cols]
            tgt_present = observed_targets if target_last else []
            dropout = self.training.column_dropout if noisy and observed_targets else 0.0
            # Without permutation the copies share one order and differ only by
            # their noise draw; nothing to noise collapses back to a single copy.
            copies = aug if noise_scales or (observed_context and dropout > 0) else 1
            if not tgt_present:
                orders = _distinct_orders(present, aug, rng) if permute else [present] * copies
                examples: list[TrainingExample] = []
                for order in orders:
                    if dropout > 0:
                        order = [c for c in order if c in target_cols or rng.random() >= dropout]
                    fields = _ordered_fields(_noisy_row(row, rng) if noisy else row, order)
                    text = self.serializer.serialize(fields)
                    examples.append(TrainingExample("", text, _numeric_spans(fields, text, 0)))
                return examples
            ctx_cols = observed_context
            pairs = (
                _distinct_block_orders(ctx_cols, tgt_present, aug, rng)
                if permute
                else [(ctx_cols, tgt_present)] * copies
            )
            examples = []
            for ctx, tgt in pairs:
                if dropout > 0:
                    ctx = [c for c in ctx if rng.random() >= dropout]
                src = _noisy_row(row, rng) if noisy else row
                ctx_fields = _ordered_fields(src, ctx)
                tgt_fields = _ordered_fields(src, tgt)
                prompt, completion = self.serializer.split(ctx_fields, tgt_fields)
                # The prompt boundary is the loss mask: masking supervises only the
                # target (context lives in the prompt); otherwise the loss stays on
                # the whole row (empty prompt) but the target is still fixed last.
                spans = _numeric_spans(
                    ctx_fields + tgt_fields, prompt + completion, len(prompt) if masking else 0
                )
                examples.append(
                    TrainingExample(prompt, completion, spans)
                    if masking
                    else TrainingExample("", prompt + completion, spans)
                )
            return examples

        all_rows = frame.to_dict("records")
        if (evaluation := self.training.evaluation) is not None:
            train_idx, eval_idx = _split_indices(
                frame, target_cols, evaluation.split, evaluation.stratify, self.random_state
            )
        else:
            train_idx, eval_idx = list(range(len(all_rows))), []
        train_rows = [all_rows[i] for i in train_idx]
        eval_rows = [all_rows[i] for i in eval_idx]

        aug = self.training.augmentation_factor

        def epoch_texts(epoch: int) -> list[TrainingExample]:
            # Seed per epoch so the call is pure in ``epoch``: the backend requests
            # epoch 0 twice (sequence-length measurement, then the dataset seed) and
            # must get identical data, and the permutation stream must not depend on
            # whether ``max_seq_length`` was auto-measured.
            rng = np.random.default_rng([base_seed, epoch])
            examples = [ex for row in train_rows for ex in row_examples(row, rng, aug)]
            rng.shuffle(examples)
            self.callback.on_train_examples(examples[:_TRAIN_PREVIEW], epoch)
            return examples

        eval_examples: list[TrainingExample] | None = None
        if eval_rows:
            eval_rng = np.random.default_rng(self.random_state)
            eval_examples = [
                ex for row in eval_rows for ex in row_examples(row, eval_rng, 1, noisy=False)
            ]

        configured = self.model.model
        label = configured if isinstance(configured, str) else type(configured).__name__
        self.callback.on_fit_info(label, self.training)
        self.callback.on_fit_start(len(train_rows), self.training.epochs)
        self.backend.fit(
            epoch_texts,
            self.training,
            self.model,
            random_state=self.random_state,
            callback=self.callback,
            eval_examples=eval_examples,
        )
        self.callback.on_fit_end()
        return self

    def _fields(self, known: Mapping[str, object]) -> list[Field]:
        return [Field(c, _native(v), c in self.numeric_cols_) for c, v in known.items()]

    def _ensure_permutable(self) -> None:
        """Reject order marginalization on a model that never saw permuted orders."""
        if not self.training.permute_order:
            raise ValueError(
                "generation.permute_order with n_samples > 1 requires a model trained "
                "with training.permute_order=True; this fit only saw the canonical "
                "column order"
            )

    def _order_rng(self, row_id: int) -> np.random.Generator:
        """RNG for one row's column-order draws, seeded on ``(random_state, row_id)``.

        Seeding on the row's absolute identity rather than its position within
        one call keeps order draws invariant to how callers chunk rows across
        calls -- the batch-size invariance the estimators promise.
        """
        base = self.random_state if self.random_state is not None else int.from_bytes(os.urandom(8))
        return np.random.default_rng([base, row_id])

    def _sample_rng(self, row_id: int, step: int) -> np.random.Generator:
        """RNG for drawing one categorical cell, seeded on ``(random_state, row_id, step)``.

        Keyed on the row's absolute identity and the target step so the draw is
        reproducible and invariant to how callers chunk rows -- the batch-size
        invariance the estimators promise -- while differing across cells.
        """
        base = self.random_state if self.random_state is not None else int.from_bytes(os.urandom(8))
        return np.random.default_rng([base, row_id, step])

    def _generate_random_state(self, row_id: int, step: int, attempt: int) -> int | None:
        """Sampler seed for one generation batch.

        Keyed on ``(random_state, row_id, step, attempt)``: ``row_id`` is the
        absolute identity of the batch's first row, so the same rows chunked
        the same way redraw the same texts across calls, and
        each retry re-seeds differently. ``None`` under ``random_state=None``
        (the backend's stream is left alone: non-deterministic, sklearn
        convention).
        """
        if self.random_state is None:
            return None
        rng = np.random.default_rng([self.random_state, row_id, step, attempt])
        return int(rng.integers(2**32))

    def _resolve_batch_size(self, generation: GenerationConfig) -> int:
        """Inference batch size: ``generation.inference_batch_size`` or training ``batch_size``."""
        return generation.inference_batch_size or self.training.batch_size

    def _generate_values(
        self,
        requests: Sequence[tuple[str, str]],
        generation: GenerationConfig,
        *,
        row_ids: Sequence[int],
        step: int,
    ) -> list[object | None]:
        """Decode one value per ``(prompt, target)`` request, batched with retries.

        Requests are generated in chunks of ``inference_batch_size``; malformed
        decodings are re-batched and retried (firing ``on_retry`` each round)
        until they parse or ``max_retries`` is exhausted, where they stay
        ``None``. Per-request attempt budget matches the unbatched path. With
        greedy decoding (``generation.temperature <= 0``) every retry would
        reproduce the same text byte for byte, so a single attempt is made.
        With ``generation.constrain_numeric``, numeric requests are grouped
        apart from the rest so a whole batch shares one logits constraint.
        Each backend call is seeded by :meth:`_generate_random_state` on its
        first request's ``row_ids`` entry, ``step`` and the attempt, so a fitted
        model with ``random_state`` set redraws the same texts call after call.
        """
        results: list[object | None] = [None] * len(requests)
        pending = list(range(len(requests)))
        batch_size = self._resolve_batch_size(generation)
        max_attempts = 1 if generation.temperature <= 0 else self.max_retries
        constraint = self.serializer.numeric_constraint() if generation.constrain_numeric else None
        for attempt in range(1, max_attempts + 1):
            if not pending:
                break
            if constraint is None:
                groups = [(pending, None)]
            else:
                numeric = [i for i in pending if requests[i][1] in self.numeric_cols_]
                free = [i for i in pending if requests[i][1] not in self.numeric_cols_]
                groups = [(numeric, constraint), (free, None)]
            still: list[int] = []
            for group, group_constraint in groups:
                for start in range(0, len(group), batch_size):
                    chunk = group[start : start + batch_size]
                    prompts = [requests[i][0] for i in chunk]
                    continuations = self.backend.generate(
                        prompts,
                        generation,
                        constraint=group_constraint,
                        random_state=self._generate_random_state(row_ids[chunk[0]], step, attempt),
                    )
                    for i, continuation in zip(chunk, continuations, strict=True):
                        numeric_cell = requests[i][1] in self.numeric_cols_
                        value = self.serializer.decode_value(continuation, numeric=numeric_cell)
                        self.callback.on_generation(
                            requests[i][0], continuation, requests[i][1], value
                        )
                        if value is None:
                            still.append(i)
                        else:
                            results[i] = value
            for i in still:
                self.callback.on_retry(requests[i][1], attempt, max_attempts)
            pending = still
        return results

    def complete_many(
        self,
        knowns: Sequence[Mapping[str, object]],
        targets: Sequence[Sequence[str]],
        generation: GenerationConfig = GenerationConfig(),
        *,
        sample_categorical: Collection[str] = frozenset(),
        row_ids: Sequence[int] | None = None,
    ) -> list[dict[str, object] | None]:
        """Complete many rows at once, batching across rows at each target step.

        Targets stay sequential *within* a row (each conditions on the prior
        outputs of that row), but the same step across all rows is processed in
        one batched pass, so the backend sees ``inference_batch_size`` prompts
        per call instead of one.

        A target column named in ``sample_categorical`` is filled by **scoring**
        its observed levels (:attr:`categories_`) and drawing one from the
        resulting conditional distribution, instead of generating free text --
        the synthesis counterpart to the imputer's argmax over the same scores.
        Every other column is generated.

        Parameters
        ----------
        knowns : Sequence[Mapping[str, object]]
            Observed columns to condition on, one mapping per row.
        targets : Sequence[Sequence[str]]
            Columns to produce per row, in order.
        generation : GenerationConfig
            Sampling hyperparameters (and ``inference_batch_size``).
        sample_categorical : Collection[str], optional
            Categorical columns to fill by scoring-and-sampling their levels
            rather than generating. Default empty (every column is generated, the
            historical behavior); the synthesis paths pass :attr:`categories_`.
        row_ids : Sequence[int] or None, optional
            Absolute row identities used to seed each categorical draw and each
            generation batch, so chunked calls produce the same samples as a
            single call. ``None`` (default) numbers the rows ``0..len(knowns)-1``.

        Returns
        -------
        list[dict[str, object] or None]
            One result per row: ``known`` merged with its produced targets, or
            ``None`` if any of that row's *generated* targets stayed malformed
            after ``max_retries`` (scored categorical cells always yield a value).
        """
        ids = row_ids if row_ids is not None else range(len(knowns))
        filled = [dict(k) for k in knowns]
        alive = [True] * len(knowns)
        max_steps = max((len(t) for t in targets), default=0)
        for step in range(max_steps):
            active = [i for i in range(len(knowns)) if alive[i] and step < len(targets[i])]
            cat_by_col: dict[str, list[int]] = {}
            gen_rows: list[int] = []
            for i in active:
                col = targets[i][step]
                if col in sample_categorical and self.categories_.get(col):
                    cat_by_col.setdefault(col, []).append(i)
                else:
                    gen_rows.append(i)
            for col, rows in cat_by_col.items():
                candidates = self.categories_[col]
                proba = self.predict_proba_many(
                    [filled[i] for i in rows],
                    col,
                    candidates,
                    generation,
                    row_ids=[ids[i] for i in rows],
                )
                for j, i in enumerate(rows):
                    p = proba[j] / proba[j].sum()
                    drawn = self._sample_rng(ids[i], step).choice(len(candidates), p=p)
                    filled[i][col] = candidates[int(drawn)]
            if gen_rows:
                gen_cols = {i: targets[i][step] for i in gen_rows}
                requests = [
                    (self.serializer.prefix(self._fields(filled[i]), gen_cols[i]), gen_cols[i])
                    for i in gen_rows
                ]
                values = self._generate_values(
                    requests, generation, row_ids=[ids[i] for i in gen_rows], step=step
                )
                for i, value in zip(gen_rows, values, strict=True):
                    if value is None:
                        alive[i] = False
                    else:
                        filled[i][gen_cols[i]] = value
        return [filled[i] if alive[i] else None for i in range(len(knowns))]

    def complete(
        self, known: Mapping[str, object], targets: Sequence[str], generation: GenerationConfig
    ) -> dict[str, object] | None:
        """Generate each target column in turn, conditioning on prior outputs.

        Single-row convenience wrapper over :meth:`complete_many`.

        Parameters
        ----------
        known : Mapping[str, object]
            Observed columns to condition on.
        targets : Sequence[str]
            Columns to generate, in order; each is conditioned on ``known`` plus
            the targets already produced.
        generation : GenerationConfig
            Sampling hyperparameters.

        Returns
        -------
        dict[str, object] or None
            ``known`` merged with the generated targets, or ``None`` if any
            target stays malformed after ``max_retries``.
        """
        return self.complete_many([known], [targets], generation)[0]

    def sample(
        self,
        n_samples: int = 1,
        *,
        condition: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
        generation: GenerationConfig = GenerationConfig(),
    ) -> pd.DataFrame:
        """Draw rows from the learned joint distribution.

        Each row is generated one column at a time in the training column
        order, every cell conditioning on ``condition`` plus the cells already
        produced -- the autoregressive factorization the permuted fine-tune was
        trained for.

        Parameters
        ----------
        n_samples : int, optional
            Number of rows to draw. Default ``1``.
        condition : mapping, sequence of mappings, or None, optional
            Columns to hold fixed. A single mapping is broadcast to all
            ``n_samples`` rows; a sequence gives one mapping per row and
            overrides ``n_samples``. ``None`` (default) samples
            unconditionally.
        generation : GenerationConfig, optional
            Sampling hyperparameters (and ``inference_batch_size``).

        Returns
        -------
        pandas.DataFrame
            One row per draw, columns in the training order; numeric columns
            are cast to float.

        Raises
        ------
        ValueError
            If ``condition`` references a column not seen at fit.
        RuntimeError
            If any row keeps a malformed cell after ``max_retries`` -- never a
            silent partial table.
        """
        check_is_fitted(self, "columns_")
        if condition is None:
            knowns: list[Mapping[str, object]] = [{}] * n_samples
        elif isinstance(condition, Mapping):
            knowns = [condition] * n_samples
        else:
            knowns = list(condition)
        seen = set(self.columns_)
        unknown = sorted({c for known in knowns for c in known if c not in seen})
        if unknown:
            raise ValueError(f"condition references columns not seen at fit: {unknown}")
        targets = [[c for c in self.columns_ if c not in known] for known in knowns]
        rows = self.complete_many(
            knowns,
            targets,
            generation,
            sample_categorical=frozenset(self.categories_),
            row_ids=range(len(knowns)),
        )
        completed = [r for r in rows if r is not None]
        if len(completed) != len(rows):
            raise RuntimeError(
                f"{len(rows) - len(completed)} of {len(rows)} sampled rows stayed "
                f"malformed after {self.max_retries} attempts"
            )
        out = pd.DataFrame(completed, columns=pd.Index(self.columns_))
        numeric = [c for c in self.columns_ if c in self.numeric_cols_]
        out[numeric] = out[numeric].astype(float)
        return out

    def sample_aggregate_many(
        self,
        knowns: Sequence[Mapping[str, object]],
        targets: Sequence[Sequence[str]],
        generation: GenerationConfig,
        *,
        row_ids: Sequence[int] | None = None,
    ) -> list[dict[str, object] | None]:
        """Draw ``generation.n_samples`` completions per row and aggregate them.

        Each row is expanded into ``n_samples`` independent completion requests
        (optionally re-permuting the conditioning columns' order per draw when
        ``generation.permute_order`` and ``n_samples > 1``), generated through
        :meth:`complete_many`, then collapsed cell by cell with
        ``generation.aggregate`` -- numeric
        columns and the rest routed by :attr:`numeric_cols_`. This is the
        ensemble-and-aggregate counterpart to :meth:`complete_many`, used by the
        estimators that average over draws (regressor, imputer).

        Parameters
        ----------
        knowns : Sequence[Mapping[str, object]]
            Observed columns to condition on, one mapping per row.
        targets : Sequence[Sequence[str]]
            Columns to generate per row, in order.
        generation : GenerationConfig
            Carries ``n_samples``, ``permute_order``, ``aggregate`` and the
            sampling hyperparameters.
        row_ids : Sequence[int] or None, optional
            Absolute row identities (within the caller's full prediction set)
            used to seed the per-row order draws and, expanded to one identity
            per draw, the categorical sampling and generation batches of
            :meth:`complete_many` -- so chunked calls produce the same samples
            as a single call. ``None`` (default) numbers the rows
            ``0..len(knowns)-1``.

        Returns
        -------
        list[dict[str, object] or None]
            One result per row: ``known`` merged with its aggregated targets, or
            ``None`` when every one of that row's ``n_samples`` draws stayed
            malformed after ``max_retries``.

        Raises
        ------
        ValueError
            If ``generation.permute_order`` would re-permute (``n_samples > 1``)
            but the model was fitted with ``training.permute_order=False``.
        """
        n = generation.n_samples
        if generation.permute_order and n > 1:
            self._ensure_permutable()
        ids = row_ids if row_ids is not None else range(len(knowns))
        flat_knowns: list[Mapping[str, object]] = []
        flat_targets: list[Sequence[str]] = []
        flat_ids: list[int] = []
        order = _column_order(generation, self.columns_)
        for known, target, row_id in zip(knowns, targets, ids, strict=True):
            cols = list(known)
            if order is not None:
                orders = [_ordered(cols, order)] * n
            elif generation.permute_order and n > 1 and len(cols) > 1:
                # Target columns stay in a tail block under the per-draw permutation,
                # mirroring the target-last training layout; see predict_proba_many.
                tail = [c for c in cols if c in self.target_cols_] if self.target_last_ else []
                if tail:
                    ctx = [c for c in cols if c not in self.target_cols_]
                    pairs = _distinct_block_orders(ctx, tail, n, self._order_rng(row_id))
                    distinct = [c_order + t_order for c_order, t_order in pairs]
                else:
                    distinct = _distinct_orders(cols, n, self._order_rng(row_id))
                orders = [distinct[j % len(distinct)] for j in range(n)]
            else:
                orders = [cols] * n
            flat_knowns.extend({c: known[c] for c in order} for order in orders)
            flat_targets.extend([list(target)] * n)
            # One absolute identity per (row, draw): draws stay distinct within a
            # row and invariant to how callers chunk rows across calls.
            flat_ids.extend(row_id * n + j for j in range(n))
        flat = self.complete_many(flat_knowns, flat_targets, generation, row_ids=flat_ids)
        out: list[dict[str, object] | None] = []
        for i, (known, target) in enumerate(zip(knowns, targets, strict=True)):
            group = [g for g in flat[i * n : (i + 1) * n] if g is not None]
            if not group:
                out.append(None)
                continue
            merged: dict[str, object] = dict(known)
            for col in target:
                merged[col] = generation.aggregate(
                    [g[col] for g in group], col in self.numeric_cols_
                )
            out.append(merged)
        return out

    def impute_many(
        self,
        knowns: Sequence[Mapping[str, object]],
        targets: Sequence[Sequence[str]],
        generation: GenerationConfig,
        *,
        score: Mapping[str, _ScoreSpec],
        row_ids: Sequence[int] | None = None,
    ) -> list[dict[str, object] | None]:
        """Fill each row's targets cell by cell, scoring some and generating the rest.

        Under ``generation.cell_context="chained"`` cells condition on the prior
        fills; under ``"observed"`` every cell conditions on the row's observed
        cells alone. Numeric fills are rounded to the column's observed decimal
        precision (:attr:`decimals_`) as they are stored, so a chained prompt
        never carries a full-precision mean the training text could not contain.
        On a ``target-last`` fitted model, target-column cells in
        the context (prior fills, or observed target cells) serialize in a tail
        block under order marginalization -- the two blocks permute
        independently, mirroring the training layout. The same column is
        batched across rows (grouped by column, since candidates are
        per-column).
        ``generation.cell_order`` picks the within-row order: ``"column"``
        (default) follows ``targets`` as given; ``"confidence"`` re-scores every
        still-pending scored cell each round and fills the one whose distribution
        has the lowest normalized entropy, so later fills condition on the values
        the model was most certain about. Columns in ``score`` are ranked with
        :meth:`predict_proba_many` and reduced to a value; a row whose turn falls
        on a generated column (under ``"confidence"``, one with only generated
        columns pending) draws it with :meth:`sample_aggregate_many` on a
        single-column target (so ``generation.n_samples`` aggregates per cell). A
        scored cell always yields a value, so only generated cells can fail and
        drop a row to ``None``.

        Parameters
        ----------
        knowns : Sequence[Mapping[str, object]]
            Observed columns to condition on, one mapping per row.
        targets : Sequence[Sequence[str]]
            Columns to fill per row, in order.
        generation : GenerationConfig
            Sampling/scoring hyperparameters (and ``inference_batch_size``).
        score : Mapping[str, _ScoreSpec]
            Columns to fill by candidate scoring, each carrying its candidates and
            a reduction closure. Columns absent from this mapping are generated.
        row_ids : Sequence[int] or None, optional
            Absolute row identities (within the caller's full prediction set)
            used to seed the per-row order draws, so chunked calls produce the
            same orders as a single call. ``None`` (default) numbers the rows
            ``0..len(knowns)-1``.

        Returns
        -------
        list[dict[str, object] or None]
            One result per row: ``known`` merged with its filled targets, or
            ``None`` if one of that row's generated cells stayed malformed after
            ``max_retries``.
        """
        ids = list(row_ids) if row_ids is not None else list(range(len(knowns)))
        rows = [
            _ImputeRow(dict(known), dict(known), tuple(target))
            for known, target in zip(knowns, targets, strict=True)
        ]
        confidence = generation.cell_order == "confidence"
        chained = generation.cell_context == "chained"

        def context(row: _ImputeRow) -> Mapping[str, object]:
            return row.filled if chained else row.known

        while any(row.alive and row.pending for row in rows):
            probe: dict[str, list[int]] = {}
            for i, row in enumerate(rows):
                if not (row.alive and row.pending):
                    continue
                for col in row.pending if confidence else row.pending[:1]:
                    if col in score:
                        probe.setdefault(col, []).append(i)
            cells: dict[int, dict[str, np.ndarray]] = {}
            for col, members in probe.items():
                proba = self.predict_proba_many(
                    [context(rows[i]) for i in members],
                    col,
                    score[col].candidates,
                    generation,
                    row_ids=[ids[i] for i in members],
                )
                for j, i in enumerate(members):
                    cells.setdefault(i, {})[col] = proba[j]
            generate = [
                i for i, row in enumerate(rows) if row.alive and row.pending and i not in cells
            ]
            rows = [
                _fill_scored(row, cells[i], score, self._quantized) if i in cells else row
                for i, row in enumerate(rows)
            ]
            if generate:
                outs = self.sample_aggregate_many(
                    [context(rows[i]) for i in generate],
                    [[rows[i].pending[0]] for i in generate],
                    generation,
                    row_ids=[ids[i] for i in generate],
                )
                drawn = dict(zip(generate, outs, strict=True))
                rows = [
                    _fill_generated(row, drawn[i], self._quantized) if i in drawn else row
                    for i, row in enumerate(rows)
                ]
        return [dict(row.filled) if row.alive else None for row in rows]

    def _quantized(self, col: str, value: object) -> object:
        """Round a computed numeric fill to the column's observed precision.

        A scored estimate or a generated aggregate is an arithmetic mean, so its
        repr carries far more decimals than any observed value; serialized into a
        chained prompt it would be text the model never trained on. Non-numeric
        columns and non-numeric values pass through untouched; an integer
        column's fill becomes an ``int`` so it serializes without a decimal
        point, exactly like its observed cells.
        """
        decimals = self.decimals_.get(col)
        v = _native(value)
        if decimals is None or not isinstance(v, int | float):
            return value
        return int(round(v)) if decimals == 0 else round(float(v), decimals)

    def _distribution(self, logprobs: np.ndarray) -> np.ndarray:
        """Softmax a candidate log-likelihood vector into a probability row.

        Falls back to a uniform distribution when no candidate is finite.
        """
        n = len(logprobs)
        uniform = np.full(n, 1.0 / n)
        finite = np.isfinite(logprobs)
        if not finite.any():
            return uniform
        weights = np.exp(logprobs - logprobs[finite].max())
        weights[~finite] = 0.0  # -inf candidates are impossible -> zero mass
        total = weights.sum()
        return weights / total if total > 0 else uniform

    def predict_proba_many(
        self,
        knowns: Sequence[Mapping[str, object]],
        target: str,
        candidates: Sequence[object],
        generation: GenerationConfig,
        *,
        row_ids: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Rank ``candidates`` for ``target`` across many rows at once.

        Every ``(row, candidate)`` prompt/continuation pair is scored; each
        backend call carries up to ``inference_batch_size`` distinct prompts
        with all of their candidates, so a prompt's shared prefix is forwarded
        once for its whole candidate set and the backend batches across rows.

        Parameters
        ----------
        knowns : Sequence[Mapping[str, object]]
            Observed columns to condition on, one mapping per row.
        target : str
            Column being predicted.
        candidates : Sequence[object]
            Values to score (shared across rows).
        generation : GenerationConfig
            Carries ``inference_batch_size`` and, for order marginalization,
            ``permute_order`` / ``n_samples`` / ``score_pool``. The temperature
            sampling fields are unused (scoring is deterministic).
        row_ids : Sequence[int] or None, optional
            Absolute row identities (within the caller's full prediction set)
            used to seed the per-row order draws, so chunked calls produce the
            same orders as a single call. ``None`` (default) numbers the rows
            ``0..len(knowns)-1``.

        Returns
        -------
        numpy.ndarray
            Shape ``(len(knowns), len(candidates))``; each row a probability
            distribution over ``candidates``.

        Notes
        -----
        When ``permute_order`` is enabled, each candidate is scored under up to
        ``n_samples`` distinct conditioning-column orders and the per-order
        results are pooled (``score_pool``, defaulting to the mean of the
        per-order softmax distributions) -- marginalizing the likelihood over
        feature order. Scoring is deterministic, so repeated orders would be
        identical; only distinct orders are used (no cycling).

        When ``generation.prior_correction`` is non-zero the pooled distribution
        is divided by the model's empty-context distribution over the same
        candidates raised to that exponent, then renormalized -- ranking
        candidates by how much the row shifts the model's belief (pointwise
        mutual information at ``1.0``). The prior is scored once per
        ``(target, candidates)`` pair and cached until the next fit.

        Raises
        ------
        ValueError
            If ``candidates`` is empty (there is nothing to rank), or if
            ``generation.permute_order`` would score under multiple orders
            (``n_samples > 1``) but the model was fitted with
            ``training.permute_order=False``.
        """
        if not candidates:
            raise ValueError("candidates must be non-empty to rank")
        numeric = target in self.numeric_cols_
        encoded = [self.serializer.encode_value(_native(c), numeric=numeric) for c in candidates]
        ids = row_ids if row_ids is not None else range(len(knowns))
        order = _column_order(generation, self.columns_)
        multi = order is None and generation.permute_order and generation.n_samples > 1
        if multi:
            self._ensure_permutable()
        n_cand = len(candidates)
        row_prompts: list[list[str]] = []
        for k, row_id in zip(knowns, ids, strict=True):
            cols = list(k) if order is None else _ordered(list(k), order)
            if not (multi and len(cols) > 1):
                orders = [cols]
            else:
                # The target-last fine-tune only ever showed target columns (observed
                # or chained fills) in a tail block, so the marginalization permutes
                # the two blocks independently (the training layout), never mixing them.
                tail = [c for c in cols if c in self.target_cols_] if self.target_last_ else []
                if tail:
                    ctx = [c for c in cols if c not in self.target_cols_]
                    pairs = _distinct_block_orders(
                        ctx, tail, generation.n_samples, self._order_rng(row_id)
                    )
                    orders = [c_order + t_order for c_order, t_order in pairs]
                else:
                    orders = _distinct_orders(cols, generation.n_samples, self._order_rng(row_id))
            row_prompts.append([
                self.serializer.prefix(
                    [Field(c, _native(k[c]), c in self.numeric_cols_) for c in order], target
                )
                for order in orders
            ])
        pairs_prompt = [p for prompts in row_prompts for p in prompts for _ in encoded]
        pairs_cont = [c for prompts in row_prompts for _ in prompts for c in encoded]
        flat = self._score_pairs(pairs_prompt, pairs_cont, generation)
        proba = np.empty((len(knowns), n_cand))
        cursor = 0
        for i, prompts in enumerate(row_prompts):
            width = len(prompts) * n_cand
            block = np.asarray(flat[cursor : cursor + width], dtype=float)
            block = block.reshape(len(prompts), n_cand)
            cursor += width
            dists = [self._distribution(row) for row in block]
            raw = [row.tolist() for row in block]
            if generation.score_pool is not None:
                pooled = np.asarray(generation.score_pool(raw), dtype=float)
                total = pooled.sum()
                proba[i] = pooled / total if total > 0 else np.full(n_cand, 1.0 / n_cand)
            else:
                proba[i] = np.mean(dists, axis=0)
            self.callback.on_score(prompts, candidates, [d.tolist() for d in dists], raw)
        if generation.prior_correction:
            prior = self._prior_distribution(target, candidates, generation)
            proba = proba / np.clip(prior, 1e-12, None) ** generation.prior_correction
            proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def _prior_distribution(
        self, target: str, candidates: Sequence[object], generation: GenerationConfig
    ) -> np.ndarray:
        """The model's empty-context distribution over ``candidates``, cached per fit."""
        key = (target, tuple(candidates), generation.candidate_scoring)
        cached = self._prior_cache_.get(key)
        if cached is None:
            neutral = replace(generation, prior_correction=0.0)
            cached = self.predict_proba_many([{}], target, candidates, neutral)[0]
            self._prior_cache_[key] = cached
        return cached

    def predict_proba(
        self, known: Mapping[str, object], target: str, candidates: Sequence[object]
    ) -> np.ndarray:
        """Rank candidate values for ``target`` by conditional likelihood.

        Single-row convenience wrapper over :meth:`predict_proba_many`.

        Parameters
        ----------
        known : Mapping[str, object]
            Observed columns to condition on.
        target : str
            Column being predicted.
        candidates : Sequence[object]
            Values to score.

        Returns
        -------
        numpy.ndarray
            Shape ``(len(candidates),)``; a probability distribution summing to
            1, obtained as a softmax over the per-candidate mean log-likelihood.

        Notes
        -----
        Falls back to a uniform distribution when no candidate is finite.
        """
        return self.predict_proba_many([known], target, candidates, GenerationConfig())[0]

    def infer_optimal_order(
        self,
        X: object,
        *,
        targets: Collection[str] | None = None,
        generation: GenerationConfig | None = None,
        n_rows: int = 48,
        n_orders: int = 20,
        bins: int = 24,
    ) -> list[str]:
        """Learn the column order the model predicts ``targets`` best under.

        The model was fine-tuned on permuted column orders, so any order is
        admissible at inference -- but it is not equally accurate under all of
        them. One order is learned for the whole table: it lays out every
        target's conditioning cells (:attr:`GenerationConfig.column_order`) and
        is the order the imputer fills a row's missing cells in, so under
        ``cell_context="chained"`` a cell conditions on the fills before it and
        never on the ones after. Each
        target is calibrated on rows of ``X`` where it is *observed*: the cell
        is hidden, the other columns are serialized in each of ``n_orders``
        sampled orders, and an order is scored by the log-probability the model
        assigns to the true value among the candidates (the levels of a
        categorical target; for a numeric one, ``bins`` representatives of its
        observed values). The calibration context mirrors inference: each
        calibration row also hides a pattern of columns drawn from those missing
        *alongside* the target in the rows to be predicted, so the learned
        order never leans on a column that will be absent when the target is
        scored. An additive position model (``score ~ sum of per-(column,
        position) effects``) is fit to the scored orders of every target at
        once and the assignment of columns to positions that maximizes it is
        returned -- typically an order that was never scored directly.

        Columns never conditioned on during calibration -- missing alongside
        every target -- carry no positional evidence and come last, right
        before the target being predicted, the best-predicted among them (by
        calibration log-likelihood over chance) first. That is the layout
        training produced them in: under
        :attr:`TrainingConfig.target_loss_weight` the target columns are
        serialized as a block at the row's end, never inside the context -- so
        a prompt that holds one (an observed target cell, or a chained fill)
        matches training only with that cell adjacent to the predicted target.
        The skill ordering also makes a hard target's fill condition on an
        easier target's fill and not the other way round.

        Cost is at most ``n_rows * n_orders * n_candidates`` scored pairs per
        target, independent of the number of permutations. The rows are scored
        in three batches (a quarter, a quarter, a half) and only the better
        half of the orders survives each batch (successive halving); orders cut
        early still enter the position model with the rows they saw, a per-row
        effect keeping them comparable to orders that saw more. Scoring stops
        as soon as the assignment stops changing between batches.

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape (n_samples, n_features)
            Rows in the training columns; may contain NaN. Calibration uses the
            rows where each target is observed, conditioning on their observed
            cells.
        targets : collection of str or None, optional
            Columns to calibrate on. ``None`` (default) takes the columns with
            at least one missing cell in ``X``, or every column when ``X`` is
            complete.
        generation : GenerationConfig or None, optional
            Scoring settings (``candidate_scoring``, ``prior_correction``,
            ``inference_batch_size``); pass the one used at prediction so the
            calibration scores what inference will. ``n_samples`` and
            ``column_order`` are overridden (single, fixed order per probe).
        n_rows : int, optional
            Calibration rows sampled per target. Default ``48``.
        n_orders : int, optional
            Distinct column orders sampled (capped at the number of
            permutations). Default ``20``.
        bins : int, optional
            Candidate count for a numeric target (see
            :class:`DiscretizationConfig`). Default ``24``.

        Returns
        -------
        list[str]
            The training columns in learned order. Targets that cannot be
            calibrated -- fewer than two candidates, no observed rows, or fewer
            than two other columns -- contribute nothing; with no calibratable
            target the training order is returned.
        """
        check_is_fitted(self)
        frame = to_frame(X, self.columns_)
        base = GenerationConfig() if generation is None else generation
        single = replace(base, n_samples=1, column_order=None)
        if targets is None:
            missing = [c for c in self.columns_ if bool(frame[c].isna().any())]
            targets = missing or list(self.columns_)
        rng = np.random.default_rng(self.random_state)
        columns = list(self.columns_)

        @dataclass(frozen=True)
        class Probe:
            """One target's calibration set: observed rows, what each hides, the truth."""

            target: str
            rows: list[dict[str, object]]
            hidden: list[frozenset[str]]
            candidates: list[object]
            true_idx: np.ndarray

            def knowns(self, order: Sequence[str], lo: int, hi: int) -> list[dict[str, object]]:
                return [
                    {
                        c: row[c]
                        for c in order
                        if c != self.target and c not in hide and not is_missing(row[c])
                    }
                    for row, hide in zip(self.rows[lo:hi], self.hidden[lo:hi], strict=True)
                ]

        def probe(target: str) -> Probe | None:
            numeric = target in self.numeric_cols_
            observed = frame.loc[frame[target].notna()]
            if numeric:
                candidates: list[object] = list(
                    select_candidates(observed[target].to_numpy(), DiscretizationConfig(bins=bins))
                )
            else:
                candidates = list(self.categories_[target])
                observed = observed.loc[observed[target].isin(candidates)]
            if len(candidates) < 2 or observed.empty or len(columns) < 3:
                return None
            picks = rng.choice(len(observed), size=min(n_rows, len(observed)), replace=False)
            rows = observed.iloc[picks].to_dict("records")
            # Calibrate in the context inference will have: hide, in each calibration
            # row, the pattern of other columns missing alongside the target in the
            # rows to be predicted (sampled from their empirical patterns).
            context = [c for c in columns if c != target]
            to_fill = frame.loc[frame[target].isna(), context]
            patterns = [frozenset(to_fill.columns[r]) for r in to_fill.isna().to_numpy()] or [
                frozenset()
            ]
            hidden = [patterns[i] for i in rng.integers(len(patterns), size=len(rows))]
            truth = [_native(row[target]) for row in rows]
            if numeric:
                grid = np.asarray(candidates, dtype=float)
                true_idx = np.abs(np.asarray(truth, dtype=float)[:, None] - grid).argmin(axis=1)
            else:
                lookup = {c: i for i, c in enumerate(candidates)}
                true_idx = np.asarray([lookup[v] for v in truth])
            return Probe(target, rows, hidden, candidates, true_idx)

        probes = [p for p in (probe(t) for t in targets) if p is not None]
        if not probes:
            return columns
        orders = _distinct_orders(columns, n_orders, rng)
        seen = {
            c
            for p in probes
            for row, hide in zip(p.rows, p.hidden, strict=True)
            for c in columns
            if c != p.target and c not in hide and not is_missing(row[c])
        }
        # per target, log p(true) over chance (log n_candidates) of every scored pair:
        # how well it is predicted, comparable across candidate counts
        gain: dict[str, list[float]] = {p.target: [] for p in probes}

        @dataclass(frozen=True)
        class Race:
            """Per order, the log p(true) of every (target, row) scored so far; who still runs."""

            hits: tuple[tuple[float, ...], ...]
            alive: frozenset[int]
            best: tuple[str, ...] | None = None
            converged: bool = False

        def score_batch(race: Race, lo: int, hi: int) -> dict[int, list[float]]:
            """The effectful step: score every live order on rows[lo:hi], one call per target."""
            alive = sorted(race.alive)
            logs: dict[int, list[float]] = {k: [] for k in alive}
            for p in probes:
                knowns = [known for k in alive for known in p.knowns(orders[k], lo, hi)]
                if not knowns:
                    continue
                proba = self.predict_proba_many(knowns, p.target, p.candidates, single)
                truth_idx = np.tile(p.true_idx[lo:hi], len(alive))
                hit = np.log(np.clip(proba[np.arange(len(knowns)), truth_idx], 1e-12, None))
                per_order = len(knowns) // len(alive)
                for i, k in enumerate(alive):
                    logs[k].extend(hit[i * per_order : (i + 1) * per_order].tolist())
                gain[p.target].extend((hit + np.log(len(p.candidates))).tolist())
            return logs

        def advance(race: Race, scored: Mapping[int, Sequence[float]]) -> Race:
            """Fold one batch in: extend the hits, keep the better half, refit."""
            hits = tuple(h + tuple(scored.get(k, ())) for k, h in enumerate(race.hits))
            means = [float(np.mean(h)) for h in hits]
            ranked = sorted(race.alive, key=means.__getitem__, reverse=True)
            best = tuple(_positional_order(orders, hits))
            return Race(
                hits=hits,
                alive=frozenset(ranked[: max(2, len(ranked) // 2)]),
                best=best,
                converged=best == race.best,
            )

        def batches(n: int) -> list[tuple[int, int]]:
            """Row slices doubling in size (n/4, n/4, n/2): a successive-halving schedule."""
            edges = [0, n // 4, n // 2, n]
            return [(lo, hi) for lo, hi in pairwise(edges) if hi > lo]

        race = Race(hits=tuple(() for _ in orders), alive=frozenset(range(len(orders))))
        for lo, hi in batches(max(len(p.rows) for p in probes)):
            race = advance(race, score_batch(race, lo, hi))
            if race.converged:
                break
        best = list(race.best or orders[0])
        skill = {target: float(np.mean(g)) for target, g in gain.items() if g}
        unseen = sorted((c for c in best if c not in seen), key=lambda c: -skill.get(c, -np.inf))
        return [c for c in best if c in seen] + unseen

    def _score_pairs(
        self, prompts: Sequence[str], continuations: Sequence[str], generation: GenerationConfig
    ) -> list[float]:
        """Log-likelihood of each ``(prompt, continuation)`` pair, batched.

        Chunks hold ``inference_batch_size`` runs of identical prompts (a row's
        candidate set, as ``predict_proba_many`` emits them), never splitting a
        run: the backend forwards a run's shared prefix once, so a run split
        across calls would re-forward it per chunk. Reduced per
        ``generation.candidate_scoring`` (mean or sum log-likelihood).
        """
        batch_size = self._resolve_batch_size(generation)
        groups = prompt_groups(prompts)
        scores: list[float] = []
        for g in range(0, len(groups), batch_size):
            start, stop = groups[g][0], groups[min(g + batch_size, len(groups)) - 1][1]
            scores.extend(
                self.backend.score(
                    prompts[start:stop],
                    continuations[start:stop],
                    reduce=generation.candidate_scoring,
                )
            )
        return scores

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.string = True
        tags.input_tags.categorical = True
        tags.input_tags.allow_nan = True
        return tags
