"""Row <-> text serialization.

A ``Serializer`` turns a (partial) tabular row into the text the language model
is trained and prompted on, and parses generated text back into values. Three
structural formats ship built in -- :class:`JSONSerializer` (``{"age": 39}``),
:class:`KeyValueSerializer` (``age: 39, city: SP``) and
:class:`BracketSerializer` (``age[39] city[SP]``). How a *numeric* cell is
rendered is an orthogonal concern handled by a :class:`NumberFormat`
(:class:`PlainNumber` or :class:`SpacedDigits`), which any serializer composes.
Custom formats only need to implement the :class:`Serializer` protocol.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np
import pandas as pd

__all__ = [
    "BracketSerializer",
    "Field",
    "IfThenSerializer",
    "JSONSerializer",
    "KeyValueSerializer",
    "NumberFormat",
    "NumericSpan",
    "PlainNumber",
    "Serializer",
    "SpacedDigits",
    "TrainingExample",
    "ValueConstraint",
    "is_missing",
]


class NumericSpan(NamedTuple):
    """One numeric value's character span inside a serialized row.

    Parameters
    ----------
    start, end : int
        Character offsets of the value inside :attr:`TrainingExample.text`.
    value : float
        The true value, as the digits encode it (``max_decimals`` applied).
    inv_scale : float
        Reciprocal of the column's standard deviation, so the auxiliary numeric
        loss measures ``(expected - value) * inv_scale`` -- the error in
        standard deviations rather than in the column's own unit. Without it a
        column of grams would outweigh one of millimetres by the square of
        their scale ratio. ``1.0`` for a constant column (nothing to normalize
        by).
    """

    start: int
    end: int
    value: float
    inv_scale: float


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """One serialized training row, split into a masked and a supervised span.

    Parameters
    ----------
    prompt : str
        The context span, weighted ``1 - target_loss_weight`` in the loss
        (masked out entirely at ``1.0``). ``""`` means "supervise everything
        evenly" -- the full row sits in ``completion`` (the default whenever
        ``target_loss_weight`` is ``None``).
    completion : str
        The supervised span; ``prompt + completion`` is the full serialized row.
    numeric_spans : tuple of NumericSpan
        Each numeric value of the row, in row order. Populated only when
        ``TrainingConfig.numeric_loss_weight`` is active; backends map the
        character offsets to token positions to compute the auxiliary numeric
        loss. ``()`` (default) otherwise.
    """

    prompt: str
    completion: str
    numeric_spans: tuple[NumericSpan, ...] = ()

    @property
    def text(self) -> str:
        """The full serialized row: ``prompt + completion``."""
        return self.prompt + self.completion


@dataclass(frozen=True, slots=True)
class Field:
    """Describe one cell to serialize.

    Parameters
    ----------
    name : object
        Column name. Coerced to ``str`` at serialization, so a non-string
        column label (e.g. an integer) is serialized by its string form.
    value : object
        Cell value.
    numeric : bool
        Whether the column is numeric (controls number vs. quoted-string
        encoding).
    """

    name: object
    value: object
    numeric: bool


@dataclass(frozen=True, slots=True)
class ValueConstraint:
    """Character-level constraint on the decoding of one numeric value.

    Produced by :meth:`Serializer.numeric_constraint` and consumed by backends
    when ``GenerationConfig.constrain_numeric`` is enabled: at each generation
    step, tokens whose text is not :meth:`allows`-ed are masked out of the
    logits, so the model still chooses *which* number to generate but cannot
    emit non-numbers.

    Parameters
    ----------
    alphabet : frozenset of str
        Characters the value's text may contain.
    terminators : tuple of str
        Delimiter texts that end the value (the serializer's pair/row
        delimiters); ``decode_value`` trims the continuation at the first one.
    """

    alphabet: frozenset[str]
    terminators: tuple[str, ...]

    def allows(self, text: str) -> bool:
        """Whether a token with this text may appear while decoding the value.

        Character-level and deliberately conservative (no positional automaton):
        a token passes when every one of its characters could occur in the value
        or in a terminator. The end-of-sequence token is allowed separately by
        each backend.
        """
        chars = self.alphabet.union(*self.terminators)
        return bool(text) and all(ch in chars for ch in text)


@runtime_checkable
class Serializer(Protocol):
    """Convert tabular rows to and from the text the model trains and samples on.

    ``number`` is the :class:`NumberFormat` numeric cells are rendered with; the
    estimators' ``max_decimals`` rebuilds it through
    :meth:`NumberFormat.with_max_decimals`, and ``numeric_loss_weight`` checks it
    renders one digit per token.
    """

    number: NumberFormat

    def serialize(self, fields: Sequence[Field]) -> str:
        """Serialize a full (ordered) row into one training/sampling string."""
        ...

    def prefix(self, known: Sequence[Field], target: object) -> str:
        """Prompt that serializes ``known`` and stops right before ``target``'s
        value, so the model generates (or is scored on) that value next.

        ``target`` is the target column name; like :class:`Field` names it is
        coerced to ``str``, so non-string column labels are supported."""
        ...

    def split(self, context: Sequence[Field], target: Sequence[Field]) -> tuple[str, str]:
        """Split a row into ``(prompt, completion)`` for loss-on-target-only.

        ``context`` is serialized first (masked out of the loss) and ``target``
        last (supervised). The invariant ``prompt + completion ==
        serialize(context + target)`` must hold so backends can locate the
        boundary by token prefix."""
        ...

    def value_spans(self, fields: Sequence[Field]) -> tuple[tuple[int, int], ...]:
        """Character span ``(start, end)`` of each field's value inside
        ``serialize(fields)``, aligned 1:1 with ``fields``.

        The invariant ``serialize(fields)[start:end] ==
        encode_value(field.value, numeric=field.numeric)`` must hold for every
        field -- backends rely on it to locate numeric values for the
        auxiliary numeric loss (``TrainingConfig.numeric_loss_weight``)."""
        ...

    def encode_value(self, value: object, *, numeric: bool) -> str:
        """Encode one value exactly as it would appear after ``prefix``."""
        ...

    def decode_value(self, text: str, *, numeric: bool) -> object | None:
        """Parse the first value out of a generated continuation, or ``None``
        if it is malformed."""
        ...

    def numeric_constraint(self) -> ValueConstraint:
        """Constraint for decoding one numeric value: the characters its text
        may contain plus the delimiters that end it.

        The alphabet includes the space -- backends strip a prompt's trailing
        space (the tokenizer boundary), so the model re-emits it as the
        continuation's first character."""
        ...


@runtime_checkable
class NumberFormat(Protocol):
    """Encode and decode the textual form of a numeric cell.

    Orthogonal to row structure: any :class:`Serializer` delegates numeric
    cells here, so the same format (``json``, ``key-value``, ``bracket``) can
    render numbers plainly or with one token per digit. ``alphabet`` is the set
    of characters :meth:`encode` can emit, used to build the serializer's
    :class:`ValueConstraint`.
    """

    @property
    def alphabet(self) -> frozenset[str]: ...

    def with_max_decimals(self, max_decimals: int | None) -> NumberFormat:
        """This format rounding to ``max_decimals`` places (``None``: no rounding)."""
        ...

    def encode(self, value: object) -> str:
        """Render a numeric value as text."""
        ...

    def decode(self, text: str) -> float | None:
        """Parse a numeric value back from text, or ``None`` if malformed."""
        ...


def is_missing(value: object) -> bool:
    """Return whether ``value`` is missing or non-finite (``None``/NaN/NaT/inf).

    Such values are never serialized: training drops them and inference
    conditions only on the remaining observed cells. The scalar missing check is
    pandas' ``isna`` (covering ``None``, NaN and ``NaT``), extended so that
    non-finite floats (``inf``/``-inf``, not missing under ``isna``) also count
    as missing.
    """
    if isinstance(value, float):
        return not math.isfinite(value)
    return bool(pd.isna(value))


# "e" and "+" cover repr's scientific notation (repr(1e16) == "1e+16").
_PLAIN_ALPHABET = frozenset("0123456789+-.e")


def _round(value: float, max_decimals: int | None) -> float:
    """Round ``value`` to at most ``max_decimals`` places, never adding digits.

    Rounds the numeric value directly rather than parsing decimals out of
    ``repr(value)``: a scientific-notation repr (``repr(1e-05) == "1e-05"``) has
    no ``"."`` to count, which would otherwise round every such value to 0 places.
    """
    if max_decimals is None:
        return value
    return round(value, max_decimals)


def _digits(value: object, max_decimals: int | None) -> str:
    """Text for a numeric cell: a float-typed value keeps a decimal point even
    when whole (``7.0``); an integer-typed value renders without one (``7``).

    NumPy scalars are unboxed first: ``np.float64`` is a ``float`` subclass
    whose ``repr`` is ``"np.float64(1.5)"`` on NumPy >= 2, which would corrupt
    the serialized text silently. Anything that is not a number after unboxing
    raises rather than degrading into garbage prompt text.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return repr(_round(value, max_decimals))
    if isinstance(value, int):
        return str(value)
    raise TypeError(
        f"cannot encode {value!r} ({type(value).__name__}) as a numeric cell; "
        "expected a Python or NumPy number"
    )


@dataclass(frozen=True, slots=True)
class PlainNumber:
    """Plain number text: ``25.7`` -> ``"25.7"``, ``100.0`` -> ``"100.0"``, ``100`` -> ``"100"``.

    Parameters
    ----------
    max_decimals : int or None, optional
        If set, truncate to this many decimal places (without introducing
        spurious trailing digits).
    """

    max_decimals: int | None = None

    @property
    def alphabet(self) -> frozenset[str]:
        return _PLAIN_ALPHABET

    def with_max_decimals(self, max_decimals: int | None) -> NumberFormat:
        return replace(self, max_decimals=max_decimals)

    def encode(self, value: object) -> str:
        return _digits(value, self.max_decimals)

    def decode(self, text: str) -> float | None:
        try:
            return float(text)
        except (TypeError, ValueError):
            return None


@dataclass(frozen=True, slots=True)
class SpacedDigits:
    """One token per digit: ``25.7`` -> ``"2 5 . 7"``, ``-3`` -> ``"- 3"``.

    Spacing every character keeps each digit on its own token, which helps the
    model treat numbers positionally (TapTap's "split" mode).

    Parameters
    ----------
    max_decimals : int or None, optional
        If set, truncate to this many decimal places before spacing.
    """

    max_decimals: int | None = 3

    @property
    def alphabet(self) -> frozenset[str]:
        return _PLAIN_ALPHABET | {" "}

    def with_max_decimals(self, max_decimals: int | None) -> NumberFormat:
        return replace(self, max_decimals=max_decimals)

    def encode(self, value: object) -> str:
        return " ".join(_digits(value, self.max_decimals))

    def decode(self, text: str) -> float | None:
        try:
            return float(text.replace(" ", ""))
        except (TypeError, ValueError):
            return None


class JSONSerializer:
    """Serialize rows as JSON objects, e.g. ``{"age": 39, "city": "SP"}``.

    Parameters
    ----------
    number : NumberFormat, optional
        How numeric cells are rendered. Defaults to :class:`PlainNumber`.
        :class:`SpacedDigits` produces JSON that is no longer valid (e.g.
        ``{"age": 2 5}``); the model still trains on it because decoding reads
        each value up to its delimiter rather than parsing the whole object.
    spaced_delimiters : bool, optional
        Pad every structural delimiter with spaces:
        ``{ "age" : 39 , "city" : "SP" }``. Compact JSON lets BPE merge a
        value's closing quote with the delimiter that follows (``",`` /
        ``"}``), so a scored candidate ending in a bare ``"`` is a token
        sequence the model never saw in training; padding keeps every
        delimiter its own token in both training and scoring. Default
        ``False`` (compact).
    """

    def __init__(
        self, *, number: NumberFormat = PlainNumber(), spaced_delimiters: bool = False
    ) -> None:
        self.number = number
        self.spaced_delimiters = spaced_delimiters
        pad = " " if spaced_delimiters else ""
        self._open = "{" + pad
        self._close = pad + "}"
        self._pair_sep = pad + ", "
        self._kv_sep = pad + ": "

    def encode_value(self, value: object, *, numeric: bool) -> str:
        if numeric:
            return self.number.encode(value)
        return json.dumps(str(value), ensure_ascii=False)

    def _pair(self, field: Field) -> str:
        return (
            f"{json.dumps(str(field.name), ensure_ascii=False)}{self._kv_sep}"
            f"{self.encode_value(field.value, numeric=field.numeric)}"
        )

    def serialize(self, fields: Sequence[Field]) -> str:
        return self._open + self._pair_sep.join(self._pair(f) for f in fields) + self._close

    def prefix(self, known: Sequence[Field], target: object) -> str:
        head = self._open + self._pair_sep.join(self._pair(f) for f in known)
        if known:
            head += self._pair_sep
        return head + json.dumps(str(target), ensure_ascii=False) + self._kv_sep

    def split(self, context: Sequence[Field], target: Sequence[Field]) -> tuple[str, str]:
        tgt = self._pair_sep.join(self._pair(f) for f in target) + self._close
        if context:
            return (
                self._open + self._pair_sep.join(self._pair(f) for f in context) + self._pair_sep,
                tgt,
            )
        return self._open, tgt

    def value_spans(self, fields: Sequence[Field]) -> tuple[tuple[int, int], ...]:
        spans: list[tuple[int, int]] = []
        pos = len(self._open)
        for f in fields:
            key = json.dumps(str(f.name), ensure_ascii=False) + self._kv_sep
            start = pos + len(key)
            end = start + len(self.encode_value(f.value, numeric=f.numeric))
            spans.append((start, end))
            pos = end + len(self._pair_sep)
        return tuple(spans)

    def decode_value(self, text: str, *, numeric: bool) -> object | None:
        if numeric:
            span = text.lstrip().split(",", 1)[0].split("}", 1)[0].strip()
            return self.number.decode(span)
        stripped = text.lstrip()
        try:
            value, end = json.JSONDecoder().raw_decode(stripped)
        except json.JSONDecodeError:
            return None
        if isinstance(value, str):
            return value
        return stripped[:end]

    def numeric_constraint(self) -> ValueConstraint:
        return ValueConstraint(self.number.alphabet | {" "}, (",", "}"))


class KeyValueSerializer:
    """Serialize rows as ``key<sep>value`` pairs, e.g. ``age:39|city:SP``.

    Categorical values are written verbatim (unquoted), so values containing
    the pair separator cannot be recovered -- use :class:`JSONSerializer` when
    that matters.

    Parameters
    ----------
    key_value_separator : str, optional
        Text between a column name and its value, padding included
        (e.g. ``": "``, ``"="``, ``" is "``). Default ``":"``.
    pair_separator : str, optional
        Text between consecutive pairs (e.g. ``", "``, ``";"``). Default ``"|"``.
    number : NumberFormat, optional
        How numeric cells are rendered. Defaults to :class:`PlainNumber`.
    """

    def __init__(
        self,
        *,
        key_value_separator: str = ":",
        pair_separator: str = "|",
        number: NumberFormat = PlainNumber(),
    ) -> None:
        self.key_value_separator = key_value_separator
        self.pair_separator = pair_separator
        self.number = number

    def encode_value(self, value: object, *, numeric: bool) -> str:
        return self.number.encode(value) if numeric else str(value)

    def _pair(self, field: Field) -> str:
        value = self.encode_value(field.value, numeric=field.numeric)
        return f"{field.name!s}{self.key_value_separator}{value}"

    def serialize(self, fields: Sequence[Field]) -> str:
        return self.pair_separator.join(self._pair(f) for f in fields)

    def prefix(self, known: Sequence[Field], target: object) -> str:
        head = self.pair_separator.join(self._pair(f) for f in known)
        if known:
            head += self.pair_separator
        return head + str(target) + self.key_value_separator

    def split(self, context: Sequence[Field], target: Sequence[Field]) -> tuple[str, str]:
        tgt = self.pair_separator.join(self._pair(f) for f in target)
        if context:
            return self.pair_separator.join(
                self._pair(f) for f in context
            ) + self.pair_separator, tgt
        return "", tgt

    def value_spans(self, fields: Sequence[Field]) -> tuple[tuple[int, int], ...]:
        spans: list[tuple[int, int]] = []
        pos = 0
        for f in fields:
            start = pos + len(f"{f.name!s}{self.key_value_separator}")
            end = start + len(self.encode_value(f.value, numeric=f.numeric))
            spans.append((start, end))
            pos = end + len(self.pair_separator)
        return tuple(spans)

    def decode_value(self, text: str, *, numeric: bool) -> object | None:
        terminator = self.pair_separator.strip() or self.pair_separator
        span = text.split(terminator, 1)[0].strip()
        if numeric:
            return self.number.decode(span)
        return span or None

    def numeric_constraint(self) -> ValueConstraint:
        # The stripped separator mirrors decode_value's terminator.
        return ValueConstraint(
            self.number.alphabet | {" "}, (self.pair_separator.strip() or self.pair_separator,)
        )


class BracketSerializer:
    """Serialize rows as space-separated ``col[value]`` pairs, e.g. ``age[39] city[SP]``.

    Categorical values are written verbatim (unbracketed), so values containing
    the closing bracket ``]`` cannot be recovered -- use :class:`JSONSerializer`
    when that matters.

    Parameters
    ----------
    number : NumberFormat, optional
        How numeric cells are rendered. Defaults to :class:`PlainNumber`.
    """

    def __init__(self, *, number: NumberFormat = PlainNumber()) -> None:
        self.number = number

    def encode_value(self, value: object, *, numeric: bool) -> str:
        return self.number.encode(value) if numeric else str(value)

    def _pair(self, field: Field) -> str:
        return f"{field.name!s}[{self.encode_value(field.value, numeric=field.numeric)}]"

    def serialize(self, fields: Sequence[Field]) -> str:
        return " ".join(self._pair(f) for f in fields)

    def prefix(self, known: Sequence[Field], target: object) -> str:
        head = " ".join(self._pair(f) for f in known)
        if known:
            head += " "
        return head + f"{target!s}["

    def split(self, context: Sequence[Field], target: Sequence[Field]) -> tuple[str, str]:
        tgt = " ".join(self._pair(f) for f in target)
        if context:
            return " ".join(self._pair(f) for f in context) + " ", tgt
        return "", tgt

    def value_spans(self, fields: Sequence[Field]) -> tuple[tuple[int, int], ...]:
        spans: list[tuple[int, int]] = []
        pos = 0
        for f in fields:
            start = pos + len(f"{f.name!s}[")
            end = start + len(self.encode_value(f.value, numeric=f.numeric))
            spans.append((start, end))
            pos = end + 2  # "] " between pairs
        return tuple(spans)

    def decode_value(self, text: str, *, numeric: bool) -> object | None:
        span = text.split("]", 1)[0].strip()
        if numeric:
            return self.number.decode(span)
        return span or None

    def numeric_constraint(self) -> ValueConstraint:
        return ValueConstraint(self.number.alphabet | {" "}, ("]",))


class IfThenSerializer:
    """Serialize rows as a natural-language rule, e.g.
    ``if x is 12, y is 24 and a is 6, then target is 32``.

    The fields are read in order: every field but the last becomes a condition
    in the ``if`` clause (comma-separated, the last two joined by ``and``), and
    the final field becomes the ``then`` consequent -- which is exactly the
    target the model is trained to produce last. A single-field row drops the
    ``if`` clause (``then x is 12``).

    Categorical values are written verbatim (unquoted) and decoding reads up to
    the first comma, so values containing a comma cannot be recovered -- use
    :class:`JSONSerializer` when that matters.

    Parameters
    ----------
    number : NumberFormat, optional
        How numeric cells are rendered. Defaults to :class:`PlainNumber`.
    """

    def __init__(self, *, number: NumberFormat = PlainNumber()) -> None:
        self.number = number

    def encode_value(self, value: object, *, numeric: bool) -> str:
        return self.number.encode(value) if numeric else str(value)

    def _pair(self, field: Field) -> str:
        return f"{field.name!s} is {self.encode_value(field.value, numeric=field.numeric)}"

    @staticmethod
    def _connector(index: int, n: int) -> str:
        """Text that precedes field ``index`` in an ``n``-field rule."""
        if index == 0:
            return "then " if n == 1 else "if "
        if index == n - 1:
            return ", then "
        return " and " if index == n - 2 else ", "

    def serialize(self, fields: Sequence[Field]) -> str:
        n = len(fields)
        return "".join(self._connector(i, n) + self._pair(f) for i, f in enumerate(fields))

    def prefix(self, known: Sequence[Field], target: object) -> str:
        n = len(known) + 1
        head = "".join(self._connector(i, n) + self._pair(f) for i, f in enumerate(known))
        return head + self._connector(n - 1, n) + f"{target!s} is "

    def split(self, context: Sequence[Field], target: Sequence[Field]) -> tuple[str, str]:
        full = [*context, *target]
        n = len(full)
        c = len(context)
        prompt = "".join(self._connector(i, n) + self._pair(full[i]) for i in range(c))
        prompt += self._connector(c, n)
        completion = self._pair(full[c]) + "".join(
            self._connector(i, n) + self._pair(full[i]) for i in range(c + 1, n)
        )
        return prompt, completion

    def value_spans(self, fields: Sequence[Field]) -> tuple[tuple[int, int], ...]:
        spans: list[tuple[int, int]] = []
        n = len(fields)
        pos = 0
        for i, f in enumerate(fields):
            start = pos + len(self._connector(i, n)) + len(f"{f.name!s} is ")
            end = start + len(self.encode_value(f.value, numeric=f.numeric))
            spans.append((start, end))
            pos = end
        return tuple(spans)

    def decode_value(self, text: str, *, numeric: bool) -> object | None:
        span = text.split(",", 1)[0].strip()
        if numeric:
            return self.number.decode(span)
        return span or None

    def numeric_constraint(self) -> ValueConstraint:
        return ValueConstraint(self.number.alphabet | {" "}, (",",))
