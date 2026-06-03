"""The shared fitted model behind every estimator.

``TabularLanguageModel`` fine-tunes one backend on serialized rows with dynamic
feature-order permutation, then exposes two conditional primitives:

- :meth:`complete` -- open-ended generation of target columns given known ones
  (imputer, oversampler, regressor).
- :meth:`predict_proba` -- rank a fixed candidate set by likelihood (classifier).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import permutations
from typing import Self

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .backend import LanguageModelBackend
from .callbacks import Callback
from .config import GenerationConfig, ModelConfig, TrainingConfig
from .hf_backend import HFBackend
from .serialize import Field, JSONSerializer, Serializer, TrainingExample, is_missing

__all__ = ["TabularLanguageModel"]

_ENUMERATE_CAP = 5040  # 7!

# Per-epoch serialized rows surfaced to the callback for preview; the full epoch
# list is already in memory, so this only bounds how many are handed over.
_TRAIN_PREVIEW = 16


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


@dataclass(frozen=True, slots=True)
class _ScoreSpec:
    """How to fill one scored numeric column: candidates to rank and a reducer.

    Lets :meth:`TabularLanguageModel.impute_many` stay agnostic of the
    discretization configuration -- the imputer injects, per column, the candidate
    set and a ``(proba_row, candidates) -> value`` reduction closure.
    """

    candidates: Sequence[float]
    reduce: Callable[[np.ndarray, Sequence[float]], object]


@dataclass(kw_only=True)
class TabularLanguageModel:
    """The fitted model shared by every estimator.

    Parameters
    ----------
    backend : LanguageModelBackend
        Fine-tunes, generates, and scores text.
    serializer : Serializer
        Converts rows to and from text.
    training : TrainingConfig
        Fine-tuning hyperparameters handed to the backend.
    model_config : ModelConfig
        Model-loading configuration (including the model id) handed to the backend.
    random_state : int or None, optional
        Seed for the per-epoch column-permutation RNG.
    max_retries : int, optional
        Generation attempts per target value before giving up. Default ``15``.
    callbacks : Callback, optional
        Feedback hooks; defaults to a no-op instance.

    Attributes
    ----------
    columns_ : list[str]
        Training column order, recorded at :meth:`fit`.
    numeric_cols_ : frozenset[str]
        Names of the numeric columns, recorded at :meth:`fit`.
    """

    backend: LanguageModelBackend = field(default_factory=HFBackend)
    serializer: Serializer = field(default_factory=JSONSerializer)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    random_state: int | None = None
    max_retries: int = 15
    callbacks: Callback = field(default_factory=Callback)

    def fit(self, frame: pd.DataFrame, *, target_cols: frozenset[str] = frozenset()) -> Self:
        """Fine-tune the backend on serialized rows.

        Each epoch re-permutes the column order per row (NaN cells are dropped),
        so the model learns to condition on any subset of columns.

        Parameters
        ----------
        frame : pandas.DataFrame
            Training table; numeric columns are detected via dtype.
        target_cols : frozenset of str, optional
            Columns to treat as targets when ``training.loss_on_target_only`` is
            enabled: each row's observed target cells are serialized last and the
            preceding context tokens are masked out of the loss. Empty (default)
            keeps loss on every token regardless of the flag.

        Returns
        -------
        Self
            The fitted model.

        Raises
        ------
        ValueError
            If ``frame`` has duplicate column names (which would make
            row serialization last-wins and break per-column indexing).
        """
        duplicates = frame.columns[frame.columns.duplicated()].unique().tolist()
        if duplicates:
            raise ValueError(f"duplicate column names are not supported: {duplicates}")
        self.columns_ = list(frame.columns)
        self.numeric_cols_ = frozenset(frame.select_dtypes(include="number").columns)
        rng = np.random.default_rng(self.random_state)
        masking = self.training.loss_on_target_only and bool(target_cols)

        def _ordered_fields(row: Mapping[str, object], order: Sequence[str]) -> list[Field]:
            return [Field(c, row[c], c in self.numeric_cols_) for c in order]

        def row_examples(
            row: Mapping[str, object], rng: np.random.Generator, aug: int
        ) -> list[TrainingExample]:
            present = [c for c in self.columns_ if not is_missing(row[c])]
            tgt_present = [c for c in present if c in target_cols] if masking else []
            if not tgt_present:
                return [
                    TrainingExample("", self.serializer.serialize(_ordered_fields(row, order)))
                    for order in _distinct_orders(present, aug, rng)
                ]
            ctx_cols = [c for c in present if c not in target_cols]
            return [
                TrainingExample(
                    *self.serializer.split(_ordered_fields(row, ctx), _ordered_fields(row, tgt))
                )
                for ctx, tgt in _distinct_block_orders(ctx_cols, tgt_present, aug, rng)
            ]

        all_rows = frame.to_dict("records")
        if self.training.validation_split > 0:
            train_idx, eval_idx = _split_indices(
                frame,
                target_cols,
                self.training.validation_split,
                self.training.stratify,
                self.random_state,
            )
        else:
            train_idx, eval_idx = list(range(len(all_rows))), []
        train_rows = [all_rows[i] for i in train_idx]
        eval_rows = [all_rows[i] for i in eval_idx]

        aug = self.training.augmentation_factor

        def epoch_texts() -> list[TrainingExample]:
            examples = [ex for row in train_rows for ex in row_examples(row, rng, aug)]
            rng.shuffle(examples)
            self.callbacks.on_train_examples(examples[:_TRAIN_PREVIEW])
            return examples

        eval_examples: list[TrainingExample] | None = None
        if eval_rows:
            eval_rng = np.random.default_rng(self.random_state)
            eval_examples = [ex for row in eval_rows for ex in row_examples(row, eval_rng, 1)]

        self.callbacks.on_fit_info(self.model_config.model, self.training)
        self.callbacks.on_fit_start(len(train_rows), self.training.epochs)
        self.backend.fit(
            epoch_texts,
            self.training,
            self.model_config,
            random_state=self.random_state,
            callbacks=self.callbacks,
            eval_examples=eval_examples,
        )
        self.callbacks.on_fit_end()
        return self

    def _fields(self, known: Mapping[str, object]) -> list[Field]:
        return [Field(c, v, c in self.numeric_cols_) for c, v in known.items()]

    def _resolve_batch_size(self, generation: GenerationConfig) -> int:
        """Inference batch size: ``generation.inference_batch_size`` or training ``batch_size``."""
        return generation.inference_batch_size or self.training.batch_size

    def _generate_values(
        self,
        requests: Sequence[tuple[str, str]],
        generation: GenerationConfig,
    ) -> list[object | None]:
        """Decode one value per ``(prompt, target)`` request, batched with retries.

        Requests are generated in chunks of ``inference_batch_size``; malformed
        decodings are re-batched and retried (firing ``on_retry`` each round)
        until they parse or ``max_retries`` is exhausted, where they stay
        ``None``. Per-request attempt budget matches the unbatched path.
        """
        results: list[object | None] = [None] * len(requests)
        pending = list(range(len(requests)))
        batch_size = self._resolve_batch_size(generation)
        for attempt in range(1, self.max_retries + 1):
            if not pending:
                break
            still: list[int] = []
            for start in range(0, len(pending), batch_size):
                chunk = pending[start : start + batch_size]
                prompts = [requests[i][0] for i in chunk]
                continuations = self.backend.generate(prompts, generation)
                for i, continuation in zip(chunk, continuations, strict=True):
                    self.callbacks.on_generation(requests[i][0], continuation, requests[i][1])
                    numeric = requests[i][1] in self.numeric_cols_
                    value = self.serializer.decode_value(continuation, numeric=numeric)
                    if value is None:
                        still.append(i)
                    else:
                        results[i] = value
            for i in still:
                self.callbacks.on_retry(requests[i][1], attempt, self.max_retries)
            pending = still
        return results

    def complete_many(
        self,
        knowns: Sequence[Mapping[str, object]],
        targets: Sequence[Sequence[str]],
        generation: GenerationConfig = GenerationConfig(),
    ) -> list[dict[str, object] | None]:
        """Complete many rows at once, batching across rows at each target step.

        Targets stay sequential *within* a row (each conditions on the prior
        outputs of that row), but the same step across all rows is generated in
        one batched pass, so the backend sees ``inference_batch_size`` prompts
        per call instead of one.

        Parameters
        ----------
        knowns : Sequence[Mapping[str, object]]
            Observed columns to condition on, one mapping per row.
        targets : Sequence[Sequence[str]]
            Columns to generate per row, in order.
        generation : GenerationConfig
            Sampling hyperparameters (and ``inference_batch_size``).

        Returns
        -------
        list[dict[str, object] or None]
            One result per row: ``known`` merged with its generated targets, or
            ``None`` if any of that row's targets stayed malformed after
            ``max_retries``.
        """
        filled = [dict(k) for k in knowns]
        alive = [True] * len(knowns)
        max_steps = max((len(t) for t in targets), default=0)
        for step in range(max_steps):
            active = [i for i in range(len(knowns)) if alive[i] and step < len(targets[i])]
            if not active:
                continue
            tgt = {i: targets[i][step] for i in active}
            requests = [
                (self.serializer.prefix(self._fields(filled[i]), tgt[i]), tgt[i]) for i in active
            ]
            values = self._generate_values(requests, generation)
            for i, value in zip(active, values, strict=True):
                if value is None:
                    alive[i] = False
                else:
                    filled[i][tgt[i]] = value
        return [filled[i] if alive[i] else None for i in range(len(knowns))]

    def complete(
        self,
        known: Mapping[str, object],
        targets: Sequence[str],
        generation: GenerationConfig,
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

    def sample_aggregate_many(
        self,
        knowns: Sequence[Mapping[str, object]],
        targets: Sequence[Sequence[str]],
        generation: GenerationConfig,
    ) -> list[dict[str, object] | None]:
        """Draw ``generation.n_samples`` completions per row and aggregate them.

        Each row is expanded into ``n_samples`` independent completion requests
        (optionally re-permuting the conditioning columns' order per draw when
        ``generation.permute_order``), generated through :meth:`complete_many`,
        then collapsed cell by cell with ``generation.aggregate`` -- numeric
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

        Returns
        -------
        list[dict[str, object] or None]
            One result per row: ``known`` merged with its aggregated targets, or
            ``None`` when every one of that row's ``n_samples`` draws stayed
            malformed after ``max_retries``.
        """
        n = generation.n_samples
        rng = np.random.default_rng(self.random_state)
        flat_knowns: list[Mapping[str, object]] = []
        flat_targets: list[Sequence[str]] = []
        for known, target in zip(knowns, targets, strict=True):
            cols = list(known)
            if generation.permute_order and len(cols) > 1:
                distinct = _distinct_orders(cols, n, rng)
                orders = [distinct[j % len(distinct)] for j in range(n)]
            else:
                orders = [cols] * n
            flat_knowns.extend({c: known[c] for c in order} for order in orders)
            flat_targets.extend([list(target)] * n)
        flat = self.complete_many(flat_knowns, flat_targets, generation)
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
    ) -> list[dict[str, object] | None]:
        """Fill each row's targets cell by cell, scoring some and generating the rest.

        Like :meth:`complete_many`, targets stay sequential within a row (each
        conditions on the prior fills) and the same step is batched across rows.
        At each step the active rows split by their step column: columns in
        ``score`` are ranked with :meth:`predict_proba_many` (grouped by column,
        since candidates are per-column) and reduced to a value; every other column
        is generated with :meth:`sample_aggregate_many` on a single-column target
        (so ``generation.n_samples`` aggregates per cell). A scored cell always
        yields a value, so only generated cells can fail and drop a row to ``None``.

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

        Returns
        -------
        list[dict[str, object] or None]
            One result per row: ``known`` merged with its filled targets, or
            ``None`` if one of that row's generated cells stayed malformed after
            ``max_retries``.
        """
        filled = [dict(k) for k in knowns]
        alive = [True] * len(knowns)
        max_steps = max((len(t) for t in targets), default=0)
        for step in range(max_steps):
            active = [i for i in range(len(knowns)) if alive[i] and step < len(targets[i])]
            score_by_col: dict[str, list[int]] = {}
            gen_rows: list[int] = []
            for i in active:
                col = targets[i][step]
                if col in score:
                    score_by_col.setdefault(col, []).append(i)
                else:
                    gen_rows.append(i)
            for col, rows in score_by_col.items():
                spec = score[col]
                proba = self.predict_proba_many(
                    [filled[i] for i in rows], col, spec.candidates, generation
                )
                for j, i in enumerate(rows):
                    filled[i][col] = spec.reduce(proba[j], spec.candidates)
            if gen_rows:
                gen_cols = {i: targets[i][step] for i in gen_rows}
                outs = self.sample_aggregate_many(
                    [filled[i] for i in gen_rows],
                    [[gen_cols[i]] for i in gen_rows],
                    generation,
                )
                for j, i in enumerate(gen_rows):
                    out = outs[j]
                    if out is None:
                        alive[i] = False
                    else:
                        filled[i][gen_cols[i]] = out[gen_cols[i]]
        return [filled[i] if alive[i] else None for i in range(len(knowns))]

    def _distribution(self, logprobs: np.ndarray) -> np.ndarray:
        """Softmax a candidate log-likelihood vector into a probability row.

        Falls back to a uniform distribution when no candidate is finite, and
        puts all mass on the ``+inf`` candidates when any has infinite
        log-likelihood.
        """
        n = len(logprobs)
        uniform = np.full(n, 1.0 / n)
        if np.isposinf(logprobs).any():
            mass = np.where(np.isposinf(logprobs), 1.0, 0.0)
            return mass / mass.sum()
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
    ) -> np.ndarray:
        """Rank ``candidates`` for ``target`` across many rows at once.

        All ``(row, candidate)`` prompt/continuation pairs are scored in chunks
        of ``inference_batch_size``, so the backend batches across both rows and
        candidates.

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

        Raises
        ------
        ValueError
            If ``candidates`` is empty (there is nothing to rank).
        """
        if not candidates:
            raise ValueError("candidates must be non-empty to rank")
        numeric = target in self.numeric_cols_
        encoded = [self.serializer.encode_value(c, numeric=numeric) for c in candidates]
        rng = np.random.default_rng(self.random_state)
        multi = generation.permute_order and generation.n_samples > 1
        n_cand = len(candidates)
        row_prompts: list[list[str]] = []
        for k in knowns:
            cols = list(k)
            orders = (
                _distinct_orders(cols, generation.n_samples, rng)
                if multi and len(cols) > 1
                else [cols]
            )
            row_prompts.append(
                [
                    self.serializer.prefix(
                        [Field(c, k[c], c in self.numeric_cols_) for c in order], target
                    )
                    for order in orders
                ]
            )
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
            self.callbacks.on_score(
                prompts,
                candidates,
                [d.tolist() for d in dists],
                raw,
            )
        return proba

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
        Falls back to a uniform distribution when no candidate is finite, and
        puts all mass on the ``+inf`` candidates when any has infinite
        log-likelihood.
        """
        return self.predict_proba_many([known], target, candidates, GenerationConfig())[0]

    def _score_pairs(
        self, prompts: Sequence[str], continuations: Sequence[str], generation: GenerationConfig
    ) -> list[float]:
        """Mean log-likelihood of each ``(prompt, continuation)`` pair, batched."""
        batch_size = self._resolve_batch_size(generation)
        scores: list[float] = []
        for start in range(0, len(prompts), batch_size):
            scores.extend(
                self.backend.score(
                    prompts[start : start + batch_size], continuations[start : start + batch_size]
                )
            )
        return scores
