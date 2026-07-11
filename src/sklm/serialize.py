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
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

__all__ = [
    "BracketSerializer",
    "Field",
    "IfThenSerializer",
    "JSONSerializer",
    "KeyValueSerializer",
    "NumberFormat",
    "PlainNumber",
    "Serializer",
    "SpacedDigits",
    "TrainingExample",
    "is_missing",
]


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """One serialized training row, split into a masked and a supervised span.

    Parameters
    ----------
    prompt : str
        The context span, whose tokens are masked out of the loss. ``""`` means
        "supervise everything" -- the full row sits in ``completion`` and no
        token is masked (the default whenever ``loss_on_target_only`` is off).
    completion : str
        The supervised span; ``prompt + completion`` is the full serialized row.
    """

    prompt: str
    completion: str

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


@runtime_checkable
class Serializer(Protocol):
    """Convert tabular rows to and from the text the model trains and samples on."""

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

    def encode_value(self, value: object, *, numeric: bool) -> str:
        """Encode one value exactly as it would appear after ``prefix``."""
        ...

    def decode_value(self, text: str, *, numeric: bool) -> object | None:
        """Parse the first value out of a generated continuation, or ``None``
        if it is malformed."""
        ...


@runtime_checkable
class NumberFormat(Protocol):
    """Encode and decode the textual form of a numeric cell.

    Orthogonal to row structure: any :class:`Serializer` delegates numeric
    cells here, so the same format (``json``, ``key-value``, ``bracket``) can
    render numbers plainly or with one token per digit.
    """

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
    """

    def __init__(self, *, number: NumberFormat = PlainNumber()) -> None:
        self.number = number

    def encode_value(self, value: object, *, numeric: bool) -> str:
        if numeric:
            return self.number.encode(value)
        return json.dumps(str(value), ensure_ascii=False)

    def _pair(self, field: Field) -> str:
        return (
            f"{json.dumps(str(field.name), ensure_ascii=False)}: "
            f"{self.encode_value(field.value, numeric=field.numeric)}"
        )

    def serialize(self, fields: Sequence[Field]) -> str:
        return "{" + ", ".join(self._pair(f) for f in fields) + "}"

    def prefix(self, known: Sequence[Field], target: object) -> str:
        head = "{" + ", ".join(self._pair(f) for f in known)
        if known:
            head += ", "
        return head + json.dumps(str(target), ensure_ascii=False) + ": "

    def split(self, context: Sequence[Field], target: Sequence[Field]) -> tuple[str, str]:
        tgt = ", ".join(self._pair(f) for f in target) + "}"
        if context:
            return "{" + ", ".join(self._pair(f) for f in context) + ", ", tgt
        return "{", tgt

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

    def decode_value(self, text: str, *, numeric: bool) -> object | None:
        terminator = self.pair_separator.strip() or self.pair_separator
        span = text.split(terminator, 1)[0].strip()
        if numeric:
            return self.number.decode(span)
        return span or None


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

    def decode_value(self, text: str, *, numeric: bool) -> object | None:
        span = text.split("]", 1)[0].strip()
        if numeric:
            return self.number.decode(span)
        return span or None


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

    def decode_value(self, text: str, *, numeric: bool) -> object | None:
        span = text.split(",", 1)[0].strip()
        if numeric:
            return self.number.decode(span)
        return span or None
