"""Column orders: drawing distinct ones for training, learning the best one for inference.

The permuted fine-tune makes any column order admissible at inference, but the
model is not equally accurate under all of them. :func:`infer_order` learns one
order for a whole table from calibration :class:`Probe` sets, racing sampled
orders by successive halving and fitting an additive position model
(:func:`positional_order`) to what they scored.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise, permutations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .serialize import is_missing

__all__ = [
    "Probe",
    "Race",
    "Scorer",
    "build_probe",
    "distinct_block_orders",
    "distinct_orders",
    "infer_order",
    "positional_order",
]

type Scorer = Callable[[Sequence[Mapping[str, object]], str, Sequence[object]], np.ndarray]
"""``(knowns, target, candidates) -> proba``: the fitted model's candidate ranking."""

_ENUMERATE_CAP = 5040  # 7!


def distinct_orders(columns: list[str], k: int, rng: np.random.Generator) -> list[list[str]]:
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


def distinct_block_orders(
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


def positional_order(orders: Sequence[Sequence[str]], hits: Sequence[Sequence[float]]) -> list[str]:
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
class Probe:
    """One target's calibration set: observed rows, what each hides, the truth.

    Parameters
    ----------
    target : str
        The column being calibrated; hidden from every row's context.
    rows : list of dict
        Calibration rows, sampled from those where ``target`` is observed.
    hidden : list of frozenset of str
        Per row, further columns hidden from the context -- a missingness
        pattern drawn from the rows the target will be predicted on, so the
        calibration conditions on what inference will actually have.
    candidates : list
        The candidate set the target is scored over.
    true_idx : numpy.ndarray
        Per row, the index in ``candidates`` of the true value (nearest grid
        point for a numeric target).
    """

    target: str
    rows: list[dict[str, object]]
    hidden: list[frozenset[str]]
    candidates: list[object]
    true_idx: np.ndarray

    def knowns(self, order: Sequence[str], lo: int, hi: int) -> list[dict[str, object]]:
        """Context of rows ``lo:hi`` laid out in ``order``."""
        return [
            {
                c: row[c]
                for c in order
                if c != self.target and c not in hide and not is_missing(row[c])
            }
            for row, hide in zip(self.rows[lo:hi], self.hidden[lo:hi], strict=True)
        ]

    @property
    def seen(self) -> set[str]:
        """Columns conditioned on in at least one calibration row."""
        return {
            c
            for row, hide in zip(self.rows, self.hidden, strict=True)
            for c in row
            if c != self.target and c not in hide and not is_missing(row[c])
        }


@dataclass(frozen=True)
class Race:
    """Successive halving over sampled orders.

    ``hits[k]`` is order ``k``'s log p(true) on every (target, row) scored so
    far; ``alive`` are the orders still being scored; ``best`` is the
    positional-model assignment after the last batch and ``converged`` whether
    it matched the batch before.
    """

    hits: tuple[tuple[float, ...], ...]
    alive: frozenset[int]
    best: tuple[str, ...] | None = None
    converged: bool = False

    def advance(
        self, orders: Sequence[Sequence[str]], scored: Mapping[int, Sequence[float]]
    ) -> Race:
        """Fold one batch in: extend the hits, keep the better half, refit."""
        hits = tuple(h + tuple(scored.get(k, ())) for k, h in enumerate(self.hits))
        means = [float(np.mean(h)) for h in hits]
        ranked = sorted(self.alive, key=means.__getitem__, reverse=True)
        best = tuple(positional_order(orders, hits))
        return replace(
            self,
            hits=hits,
            alive=frozenset(ranked[: max(2, len(ranked) // 2)]),
            best=best,
            converged=best == self.best,
        )


def build_probe(
    frame: pd.DataFrame,
    target: str,
    candidates: Sequence[object],
    *,
    columns: Sequence[str],
    numeric: bool,
    n_rows: int,
    rng: np.random.Generator,
) -> Probe | None:
    """Sample ``target``'s calibration set from the rows of ``frame`` where it is observed.

    Returns ``None`` when the target cannot be calibrated: fewer than two
    candidates, no observed rows, or fewer than two other columns to order.
    """
    observed = frame.loc[frame[target].notna()]
    if not numeric:
        observed = observed.loc[observed[target].isin(list(candidates))]
    if len(candidates) < 2 or observed.empty or len(columns) < 3:
        return None
    picks = rng.choice(len(observed), size=min(n_rows, len(observed)), replace=False)
    rows = observed.iloc[picks].to_dict("records")
    context = [c for c in columns if c != target]
    to_fill = frame.loc[frame[target].isna(), context]
    patterns = [frozenset(to_fill.columns[r]) for r in to_fill.isna().to_numpy()] or [frozenset()]
    hidden = [patterns[i] for i in rng.integers(len(patterns), size=len(rows))]
    truth = [row[target] for row in rows]
    if numeric:
        grid = np.asarray(candidates, dtype=float)
        true_idx = np.abs(np.asarray(truth, dtype=float)[:, None] - grid).argmin(axis=1)
    else:
        lookup = {c: i for i, c in enumerate(candidates)}
        true_idx = np.asarray([lookup[v] for v in truth])
    return Probe(target, rows, hidden, list(candidates), true_idx)


def infer_order(
    scorer: Scorer,
    probes: Sequence[Probe],
    columns: Sequence[str],
    *,
    n_orders: int,
    rng: np.random.Generator,
) -> list[str]:
    """Learn the order of ``columns`` the model predicts the probed targets best under.

    ``n_orders`` distinct orders are sampled and raced over the calibration
    rows in three batches (a quarter, a quarter, a half): every live order
    serializes each row's context in its layout, ``scorer`` ranks the
    candidates, and the order is credited with log p(true). Only the better
    half survives each batch; orders cut early still enter the position model
    with the rows they saw, whose per-row effect keeps them comparable. The
    race stops once :func:`positional_order`'s assignment stops changing.

    Columns never conditioned on -- missing alongside every target -- carry no
    positional evidence and go last, right before the predicted target, the
    best-predicted among them (calibration log-likelihood over chance) first.
    That mirrors the target-last training layout, where a target cell only ever
    appears adjacent to the predicted one, and makes a hard target's fill
    condition on an easier target's fill rather than the reverse.

    Cost is at most ``n_rows * n_orders * n_candidates`` scored pairs per
    target, independent of the number of permutations. With no probe the
    training order is returned unchanged.
    """
    if not probes:
        return list(columns)
    orders = distinct_orders(list(columns), n_orders, rng)
    race = Race(hits=tuple(() for _ in orders), alive=frozenset(range(len(orders))))
    gain: dict[str, list[float]] = {p.target: [] for p in probes}
    for lo, hi in _halving_batches(max(len(p.rows) for p in probes)):
        scored, gains = _score_batch(scorer, probes, orders, sorted(race.alive), lo, hi)
        for target, g in gains.items():
            gain[target].extend(g)
        race = race.advance(orders, scored)
        if race.converged:
            break
    best = list(race.best or orders[0])
    seen = set().union(*(p.seen for p in probes))
    skill = {target: float(np.mean(g)) for target, g in gain.items() if g}
    unseen = sorted((c for c in best if c not in seen), key=lambda c: -skill.get(c, -np.inf))
    return [c for c in best if c in seen] + unseen


def _score_batch(
    scorer: Scorer,
    probes: Sequence[Probe],
    orders: Sequence[Sequence[str]],
    alive: Sequence[int],
    lo: int,
    hi: int,
) -> tuple[dict[int, list[float]], dict[str, list[float]]]:
    """Score every live order on rows ``lo:hi`` of every probe, one call per target.

    Returns each order's log p(true) per scored row, and each target's log
    p(true) over chance (``log n_candidates``) -- how well it is predicted,
    comparable across candidate counts.
    """
    logs: dict[int, list[float]] = {k: [] for k in alive}
    gains: dict[str, list[float]] = {}
    for p in probes:
        knowns = [known for k in alive for known in p.knowns(orders[k], lo, hi)]
        if not knowns:
            continue
        proba = scorer(knowns, p.target, p.candidates)
        truth_idx = np.tile(p.true_idx[lo:hi], len(alive))
        hit = np.log(np.clip(proba[np.arange(len(knowns)), truth_idx], 1e-12, None))
        per_order = len(knowns) // len(alive)
        for i, k in enumerate(alive):
            logs[k].extend(hit[i * per_order : (i + 1) * per_order].tolist())
        gains[p.target] = (hit + np.log(len(p.candidates))).tolist()
    return logs, gains


def _halving_batches(n: int) -> list[tuple[int, int]]:
    """Row slices doubling in size (n/4, n/4, n/2): a successive-halving schedule."""
    edges = [0, n // 4, n // 2, n]
    return [(lo, hi) for lo, hi in pairwise(edges) if hi > lo]
