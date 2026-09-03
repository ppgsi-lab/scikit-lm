"""The language-model backend protocol.

A backend is a pure execution engine: it receives a :class:`~sklm.ModelConfig`
(carrying the model id) and a :class:`~sklm.TrainingConfig` at fit time and a
:class:`~sklm.GenerationConfig` at generate time, and stores no user-facing
hyperparameters of its own. ``LanguageModelBackend`` is the only abstraction the
rest of the library depends on, so torch/transformers (:class:`~sklm.HFBackend`)
and mlx (:class:`~sklm.MLXBackend`) stay optional extras, and tests inject a
lightweight fake backend instead.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal, NamedTuple, Protocol, runtime_checkable

from .callbacks import Callback
from .config import GenerationConfig, ModelConfig, TrainingConfig
from .serialize import TrainingExample, ValueConstraint

__all__ = [
    "DigitTokens",
    "EarlyStopping",
    "LanguageModelBackend",
    "NumericTokenArrays",
    "Verdict",
    "common_token_prefix",
    "digit_scales",
    "numeric_token_arrays",
    "prompt_groups",
    "resolve_digit_tokens",
    "resolve_max_new_tokens",
    "resolve_max_seq_length",
]


def common_token_prefix(prompt_ids: Sequence[int], ids: Sequence[int]) -> int:
    """Length of the longest token prefix ``prompt_ids`` shares with ``ids``.

    Both backends locate the prompt/continuation boundary this way -- in
    ``score`` and in the loss-on-target-only prompt mask -- because tokenizing
    ``prompt + continuation`` may BPE-merge tokens at the prompt's trailing
    edge, so ``len(prompt_ids)`` alone over- or under-shoots. Returns the raw
    prefix length; each call site applies its own clamp (``score`` keeps at
    least one continuation token, the datasets supervise at least one token).

    Parameters
    ----------
    prompt_ids : Sequence[int]
        Token ids of the prompt alone.
    ids : Sequence[int]
        Token ids of the full ``prompt + continuation`` text.

    Returns
    -------
    int
        Number of leading tokens identical in both sequences.
    """
    n = 0
    for a, b in zip(prompt_ids, ids, strict=False):
        if a != b:
            break
        n += 1
    return n


def prompt_groups(prompts: Sequence[str]) -> list[tuple[int, int]]:
    """Consecutive ``[start, stop)`` runs of identical prompts.

    Both backends' ``score`` forward each run's shared token prefix once (primed
    into a KV cache) and then score only the per-pair remainders. Callers such
    as ``predict_proba_many`` emit a row's candidates contiguously, so one run
    is typically one row's whole candidate set.

    Parameters
    ----------
    prompts : Sequence[str]
        The score call's prompts, in order.

    Returns
    -------
    list[tuple[int, int]]
        Half-open index ranges covering ``prompts``, one per run.
    """
    groups: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(prompts) + 1):
        if i == len(prompts) or prompts[i] != prompts[start]:
            groups.append((start, i))
            start = i
    return groups


def digit_scales(encoded: str) -> list[float]:
    """Scale ``sign * 10**place`` of each digit character in an encoded number.

    The auxiliary numeric loss (``TrainingConfig.numeric_loss_weight``)
    reconstructs the expected number as ``sum(E[digit_i] * scales[i])``, so the
    scales carry both the decimal place and the sign of the true value (the
    ``-`` and ``.`` tokens themselves stay supervised only by the
    cross-entropy). Returns ``[]`` for scientific notation, whose exponent
    digits are not positional -- such values are skipped by the term.

    Parameters
    ----------
    encoded : str
        The number exactly as serialized (spaces allowed, e.g. ``"- 2 5 . 7"``).

    Returns
    -------
    list of float
        One scale per digit character, in text order.
    """
    compact = encoded.replace(" ", "")
    if "e" in compact or "E" in compact:
        return []
    sign = -1.0 if compact.startswith("-") else 1.0
    integer, _, fraction = compact.lstrip("+-").partition(".")
    return [sign * 10.0 ** (len(integer) - 1 - i) for i in range(len(integer))] + [
        sign * 10.0 ** -(i + 1) for i in range(len(fraction))
    ]


class DigitTokens(NamedTuple):
    """Token ids of the digit surface forms usable by the numeric loss.

    Parameters
    ----------
    variant_of : dict of int to int
        Token id -> variant index (0 = bare ``"5"``, 1 = leading-space
        ``" 5"``) for every single-token digit form.
    candidates : tuple of list of int
        Per variant, the ten token ids of digits ``0..9`` (the restricted
        softmax support), or ``[]`` when that variant is not single-token.
    """

    variant_of: dict[int, int]
    candidates: tuple[list[int], list[int]]


def resolve_digit_tokens(encode: Callable[[str], Sequence[int]]) -> DigitTokens:
    """Resolve the digit-token tables the numeric loss scores over.

    A variant (bare or leading-space) is usable only when all ten of its digit
    forms encode to a single token; positions whose token falls outside every
    usable form are skipped by :func:`numeric_token_arrays`.

    Raises
    ------
    ValueError
        If neither variant is usable -- the tokenizer cannot support
        ``TrainingConfig.numeric_loss_weight``.
    """
    variant_of: dict[int, int] = {}
    candidates: list[list[int]] = []
    for variant, prefix in enumerate(("", " ")):
        ids = [encode(f"{prefix}{d}") for d in range(10)]
        if all(len(i) == 1 for i in ids):
            vec = [i[0] for i in ids]
            for tid in vec:
                variant_of[tid] = variant
            candidates.append(vec)
        else:
            candidates.append([])
    if not variant_of:
        raise ValueError(
            "numeric_loss_weight requires every digit (bare or with a leading "
            "space) to be a single token in the model's tokenizer"
        )
    return DigitTokens(variant_of, (candidates[0], candidates[1]))


class NumericTokenArrays(NamedTuple):
    """Per-token numeric-loss annotations for one training example.

    ``scale``/``variant``/``number_id`` align 1:1 with the example's token ids
    (``0.0``/``-1``/``-1`` outside digit positions); ``targets[j]`` and
    ``weights[j]`` are the true value and the reciprocal column scale
    (:attr:`~sklm.NumericSpan.inv_scale`) of the example's ``j``-th numeric
    span, so the loss can measure each slot's error in standard deviations. A
    span whose digits cannot all be mapped to single digit tokens (truncation,
    a BPE merge at its edge, scientific notation) keeps its target slot but
    marks no token, so the loss masks it out by its zero token count.
    """

    scale: list[float]
    variant: list[int]
    number_id: list[int]
    targets: list[float]
    weights: list[float]


def numeric_token_arrays(
    example: TrainingExample,
    ids: Sequence[int],
    encode: Callable[[str], Sequence[int]],
    variant_of: Mapping[int, int],
) -> NumericTokenArrays:
    """Map an example's ``numeric_spans`` from character to token positions.

    Each span's token range is located with the :func:`common_token_prefix`
    discipline (tokenize ``text[:offset]`` and measure the shared prefix), then
    its digit tokens are matched in order against the digit characters of the
    encoded value, whose place scales come from :func:`digit_scales`.
    """
    scale = [0.0] * len(ids)
    variant = [-1] * len(ids)
    number_id = [-1] * len(ids)
    targets: list[float] = []
    weights: list[float] = []
    for span in example.numeric_spans:
        slot = len(targets)
        targets.append(span.value)
        weights.append(span.inv_scale)
        scales = digit_scales(example.text[span.start : span.end])
        if not scales:
            continue
        t0 = common_token_prefix(encode(example.text[: span.start]), ids)
        t1 = min(common_token_prefix(encode(example.text[: span.end]), ids), len(ids))
        assigned: list[tuple[int, int]] = []
        for t in range(t0, t1):
            v = variant_of.get(ids[t])
            if v is not None:
                assigned.append((t, v))
        if len(assigned) != len(scales):
            continue
        for (t, v), s in zip(assigned, scales, strict=True):
            scale[t] = s
            variant[t] = v
            number_id[t] = slot
    return NumericTokenArrays(scale, variant, number_id, targets, weights)


def resolve_max_new_tokens(generation: GenerationConfig, max_seq_length: int) -> int:
    """Generation token budget, defaulting to the longest training row.

    ``GenerationConfig.max_new_tokens`` is ``None`` by default because the config
    is constructed before any data is seen; at generate time each backend
    substitutes its fitted ``max_seq_length`` (the longest serialized row), an
    upper bound that covers any single value the model could emit without manual
    tuning. The generated tokens are trimmed to the target value's delimiter by
    the serializer afterwards, so an over-budget is harmless.
    """
    if generation.max_new_tokens is not None:
        return generation.max_new_tokens
    return max_seq_length


def resolve_max_seq_length(
    examples: Sequence[TrainingExample], token_len: Callable[[str], int], *, multiple: int = 8
) -> int:
    """Smallest ``multiple`` of tokens holding the longest serialized row.

    Fills ``TrainingConfig.max_seq_length`` when left ``None``: each backend
    passes a ``token_len`` matching how it tokenizes a row (including EOS), and
    the cap is the max over ``examples`` rounded up so no row is truncated.
    Column-order permutation reorders a row's tokens without changing their
    count, so measuring one serialization per row is representative.
    """
    longest = max((token_len(ex.text) for ex in examples), default=multiple)
    return ((longest + multiple - 1) // multiple) * multiple


def checkpoint_workdir(prefix: str) -> tempfile.TemporaryDirectory[str]:
    """A fit's temporary checkpoint directory, named after its owner process.

    The directory is removed when the ``with`` block exits -- also on an
    exception -- but a process killed outright (OOM killer, node reclaim) never
    runs the cleanup and leaks a multi-GB directory that starves later fits on
    the same disk. Encoding the PID in the name lets the next fit reap exactly
    the directories whose owner is gone, so concurrent fits on one machine
    never touch each other's live checkpoints.
    """
    for stale in Path(tempfile.gettempdir()).glob(f"{prefix}*"):
        pid = stale.name.removeprefix(prefix).split("_", 1)[0]
        if not pid.isdigit():
            continue
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            shutil.rmtree(stale, ignore_errors=True)
        except PermissionError:
            pass  # alive, owned by another user
    return tempfile.TemporaryDirectory(prefix=f"{prefix}{os.getpid()}_")


type Verdict = Literal["improved", "no_improvement", "exhausted"]


class EarlyStopping:
    """Best-validation-loss tracker with patience, shared by both backends.

    :meth:`observe` is fed every validation loss and says whether it improved on
    the best so far, failed to (extending the no-improvement streak), or
    exhausted ``patience`` -- at which point the backend stops training. A loss
    counts as an improvement when it is strictly below the best (beyond a 1e-9
    tolerance); ``patience=None`` never exhausts. ``best`` and ``streak`` are
    plain attributes so a backend can persist them into a checkpoint and seed
    a resumed run through the constructor.
    """

    def __init__(self, patience: int | None, *, best: float = math.inf, streak: int = 0) -> None:
        self.patience = patience
        self.best = best
        self.streak = streak

    def observe(self, loss: float) -> Verdict:
        if loss < self.best - 1e-9:
            self.best, self.streak = loss, 0
            return "improved"
        self.streak += 1
        if self.patience is not None and self.streak >= self.patience:
            return "exhausted"
        return "no_improvement"


@runtime_checkable
class LanguageModelBackend(Protocol):
    """Execution engine that fine-tunes, generates, and scores text.

    The only abstraction the rest of the library depends on: estimators hand it
    the configs (the model id rides on :class:`~sklm.ModelConfig`) and consume
    its three primitives, so torch and transformers stay an optional extra.

    Notes
    -----
    Shipped implementations are :class:`~sklm.HFBackend` (transformers) and
    :class:`~sklm.MLXBackend` (mlx-lm).
    """

    def fit(
        self,
        epoch_texts: Callable[[int], list[TrainingExample]],
        training: TrainingConfig,
        model_config: ModelConfig,
        *,
        random_state: int | None,
        callback: Callback,
        eval_examples: list[TrainingExample] | None = None,
    ) -> None:
        """Fine-tune ``model_config.model`` on serialized rows. ``epoch_texts(epoch)``
        returns one freshly (re-)permuted :class:`~sklm.TrainingExample` per training
        row for the 0-indexed ``epoch``, so feature-order permutation stays dynamic
        across epochs. The permutation is seeded on ``(seed, epoch)``: calling it
        again with the same epoch yields identical examples, so the measurement
        pre-pass and the epoch-0 dataset seed can both request epoch 0 without
        diverging. Each example's ``prompt`` (empty unless
        ``training.target_loss_weight`` marks that row's context) is
        down-weighted in the loss. ``callback`` receives training-loss reports.

        ``eval_examples`` is the held-out validation set (``None`` when
        ``training.evaluation`` is ``None``): a fixed list serialized once, so
        validation loss is comparable across evaluations. When provided, the
        backend evaluates on it on the ``training.evaluation`` cadence
        (``each`` steps or epochs), reports the loss through
        :meth:`~sklm.Callback.on_eval_report`, tracks the best weights and stops
        early through :class:`EarlyStopping`, and checkpoints per
        ``training.checkpoint``."""
        ...

    def generate(
        self,
        prompts: Sequence[str],
        generation: GenerationConfig,
        *,
        constraint: ValueConstraint | None = None,
        random_state: int | None = None,
    ) -> list[str]:
        """Sample a continuation for each prompt (the generated text only).

        When ``constraint`` is set, every decoding step masks out the tokens it
        does not :meth:`~sklm.ValueConstraint.allows` (the end-of-sequence token
        stays allowed, so generation can stop) -- the numeric constrained
        decoding behind ``GenerationConfig.constrain_numeric``. ``None``
        (default) decodes unconstrained. ``random_state`` seeds the backend's
        sampler before this batch is drawn, so the same call reproduces the same
        texts; ``None`` leaves the sampler's stream untouched (non-deterministic)."""
        ...

    def score(
        self,
        prompts: Sequence[str],
        continuations: Sequence[str],
        *,
        reduce: Literal["mean", "sum"] = "mean",
    ) -> list[float]:
        """Per-token log-likelihood of ``continuations[i]`` given ``prompts[i]``.

        The two sequences are paired element-wise (equal length) and scored as a
        single batch, so callers chunk by ``inference_batch_size`` and score many
        prompts at once. ``reduce`` selects how each pair's per-token
        log-likelihoods collapse to one number: ``"mean"`` (default,
        length-normalized) or ``"sum"`` (total log-likelihood). The two reductions
        must be implemented identically across backends so candidate ranking
        matches (invariant #3)."""
        ...
