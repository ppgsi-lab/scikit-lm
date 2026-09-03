from __future__ import annotations

import numpy as np
import pytest

from sklm import (
    BracketSerializer,
    Field,
    IfThenSerializer,
    JSONSerializer,
    KeyValueSerializer,
    PlainNumber,
    Serializer,
    SpacedDigits,
)
from sklm.base import resolve_serializer


def test_serialize_full_row() -> None:
    s = JSONSerializer()
    fields = [Field("age", 39, True), Field("city", "São Paulo", False)]
    assert s.serialize(fields) == '{"age": 39, "city": "São Paulo"}'


def test_prefix_stops_before_target_value() -> None:
    s = JSONSerializer()
    assert s.prefix([Field("age", 39, True)], "city") == '{"age": 39, "city": '
    assert s.prefix([], "age") == '{"age": '


def test_spaced_delimiters_layout() -> None:
    s = JSONSerializer(spaced_delimiters=True)
    fields = [Field("age", 39, True), Field("city", "São Paulo", False)]
    assert s.serialize(fields) == '{ "age" : 39 , "city" : "São Paulo" }'
    assert s.prefix([Field("age", 39, True)], "city") == '{ "age" : 39 , "city" : '
    assert s.prefix([], "age") == '{ "age" : '


def test_spaced_delimiters_split_empty_context_keeps_padding() -> None:
    s = JSONSerializer(spaced_delimiters=True)
    prompt, completion = s.split([], [Field("label", "yes", False)])
    assert prompt == "{ "
    assert completion == '"label" : "yes" }'


def test_encode_value() -> None:
    s = JSONSerializer()
    assert s.encode_value(39.0, numeric=True) == "39.0"
    assert s.encode_value(3.5, numeric=True) == "3.5"
    assert s.encode_value("a, b", numeric=False) == '"a, b"'


def test_numeric_encode_unboxes_numpy_scalars() -> None:
    # repr(np.float64(1.5)) is "np.float64(1.5)" on numpy >= 2; encoding it
    # verbatim would silently corrupt the serialized text
    s = JSONSerializer()
    assert s.encode_value(np.float64(1.5), numeric=True) == "1.5"
    assert s.encode_value(np.float32(1.5), numeric=True) == "1.5"
    assert s.encode_value(np.int64(7), numeric=True) == "7"


def test_numeric_encode_rejects_non_numbers() -> None:
    with pytest.raises(TypeError, match="numeric cell"):
        JSONSerializer().encode_value("39", numeric=True)


def test_decode_value_tolerates_trailing_text() -> None:
    s = JSONSerializer()
    assert s.decode_value('"São Paulo", "x": 1', numeric=False) == "São Paulo"
    assert s.decode_value("39.5, foo", numeric=True) == 39.5
    assert s.decode_value("0", numeric=False) == "0"


def test_decode_value_rejects_malformed() -> None:
    s = JSONSerializer()
    assert s.decode_value("garbage", numeric=True) is None
    assert s.decode_value("", numeric=False) is None


def test_decode_value_non_string_literal_returns_raw_token() -> None:
    s = JSONSerializer()
    assert s.decode_value("true, foo", numeric=False) == "true"
    assert s.decode_value("null}", numeric=False) == "null"
    assert s.decode_value("false", numeric=False) == "false"


@pytest.mark.parametrize(
    "context",
    [[Field("age", 39, True), Field("city", "São Paulo", False)], [Field("age", 39, True)], []],
)
def test_split_concatenates_to_full_serialization(context: list[Field]) -> None:
    s = JSONSerializer()
    target = [Field("score", 3.5, True), Field("label", "yes", False)]
    prompt, completion = s.split(context, target)
    assert prompt + completion == s.serialize([*context, *target])


def test_split_empty_context_starts_at_brace() -> None:
    s = JSONSerializer()
    prompt, completion = s.split([], [Field("label", "yes", False)])
    assert prompt == "{"
    assert completion == '"label": "yes"}'


def test_split_boundary_space_belongs_to_prompt() -> None:
    s = JSONSerializer()
    prompt, _ = s.split([Field("age", 39, True)], [Field("label", "yes", False)])
    assert prompt == '{"age": 39, '


# --- NumberFormat ---------------------------------------------------------


def test_plain_number_keeps_decimal_for_floats() -> None:
    assert PlainNumber().encode(39.0) == "39.0"
    assert PlainNumber().encode(3.5) == "3.5"
    assert PlainNumber().encode(-3) == "-3"
    assert PlainNumber().encode(39) == "39"


def test_plain_number_max_decimals_truncates_without_padding() -> None:
    assert PlainNumber(max_decimals=2).encode(3.14159) == "3.14"
    assert PlainNumber(max_decimals=5).encode(3.5) == "3.5"


def test_plain_number_decode_round_trips() -> None:
    assert PlainNumber().decode("3.5") == 3.5
    assert PlainNumber().decode("39") == 39.0
    assert PlainNumber().decode("nope") is None


def test_spaced_digits_puts_one_token_per_character() -> None:
    assert SpacedDigits().encode(39.0) == "3 9 . 0"
    assert SpacedDigits().encode(25.7) == "2 5 . 7"
    assert SpacedDigits().encode(-3) == "- 3"


def test_spaced_digits_decode_strips_spaces() -> None:
    assert SpacedDigits().decode("2 5 . 7") == 25.7
    assert SpacedDigits().decode("3 9") == 39.0
    assert SpacedDigits().decode("") is None


def test_spaced_digits_compose_with_json() -> None:
    s = JSONSerializer(number=SpacedDigits())
    assert s.encode_value(1250, numeric=True) == "1 2 5 0"
    assert s.decode_value("1 2 5 0, foo", numeric=True) == 1250.0


# --- KeyValueSerializer ---------------------------------------------------


def test_key_value_serialize_and_prefix() -> None:
    s = KeyValueSerializer()
    fields = [Field("age", 39, True), Field("city", "SP", False)]
    assert s.serialize(fields) == "age:39|city:SP"
    assert s.prefix([Field("age", 39, True)], "city") == "age:39|city:"
    assert s.prefix([], "age") == "age:"


def test_key_value_decode_value() -> None:
    s = KeyValueSerializer()
    assert s.decode_value("39|city:SP", numeric=True) == 39.0
    assert s.decode_value("SP|x:1", numeric=False) == "SP"


def test_key_value_custom_delimiters() -> None:
    s = KeyValueSerializer(key_value_separator="=", pair_separator=";")
    fields = [Field("name", "gabriel", False), Field("age", 33, True)]
    assert s.serialize(fields) == "name=gabriel;age=33"
    assert s.decode_value("33;next=1", numeric=True) == 33.0


# --- BracketSerializer ----------------------------------------------------


def test_bracket_serialize_and_prefix() -> None:
    s = BracketSerializer()
    fields = [Field("age", 39, True), Field("city", "SP", False)]
    assert s.serialize(fields) == "age[39] city[SP]"
    assert s.prefix([Field("age", 39, True)], "city") == "age[39] city["
    assert s.prefix([], "age") == "age["


def test_bracket_decode_value() -> None:
    s = BracketSerializer()
    assert s.decode_value("39] city[SP]", numeric=True) == 39.0
    assert s.decode_value("SP] x[1]", numeric=False) == "SP"


def test_bracket_spaced_digits_decode() -> None:
    s = BracketSerializer(number=SpacedDigits())
    assert s.encode_value(1250, numeric=True) == "1 2 5 0"
    assert s.decode_value("1 2 5 0] x[1]", numeric=True) == 1250.0


# --- IfThenSerializer -----------------------------------------------------


def test_if_then_serialize_and_prefix() -> None:
    s = IfThenSerializer()
    fields = [Field("x", 12, True), Field("y", 24, True), Field("a", 6, True), Field("t", 32, True)]
    assert s.serialize(fields) == "if x is 12, y is 24 and a is 6, then t is 32"
    assert s.prefix(fields[:-1], "t") == "if x is 12, y is 24 and a is 6, then t is "
    assert s.serialize([Field("t", 32, True)]) == "then t is 32"
    assert s.prefix([], "t") == "then t is "
    assert s.serialize(fields[:2]) == "if x is 12, then y is 24"


def test_if_then_decode_value() -> None:
    s = IfThenSerializer()
    assert s.decode_value("32", numeric=True) == 32.0
    assert s.decode_value("yes, then z is 1", numeric=False) == "yes"


# --- non-string field names -----------------------------------------------


@pytest.mark.parametrize("s", [KeyValueSerializer(), BracketSerializer()])
def test_non_string_field_name_is_coerced(s: Serializer) -> None:
    known = [Field(0, "x", False)]
    full = s.serialize([*known, Field(1, 3.5, True)])
    prefix = s.prefix(known, 1)
    assert full.startswith(prefix)
    assert s.decode_value(full[len(prefix) :], numeric=True) == 3.5


# --- cross-format invariants ----------------------------------------------


def _serializers() -> list[Serializer]:
    return [
        JSONSerializer(),
        JSONSerializer(number=SpacedDigits()),
        JSONSerializer(spaced_delimiters=True),
        JSONSerializer(spaced_delimiters=True, number=SpacedDigits()),
        KeyValueSerializer(),
        KeyValueSerializer(key_value_separator="=", pair_separator=";"),
        BracketSerializer(),
        BracketSerializer(number=SpacedDigits()),
        IfThenSerializer(),
        IfThenSerializer(number=SpacedDigits()),
    ]


@pytest.mark.parametrize("s", _serializers())
@pytest.mark.parametrize(
    "context", [[Field("age", 39, True), Field("city", "SP", False)], [Field("age", 39, True)], []]
)
def test_split_concatenates_to_full_serialization_all_formats(
    s: Serializer, context: list[Field]
) -> None:
    target = [Field("score", 3.5, True), Field("label", "yes", False)]
    prompt, completion = s.split(context, target)
    assert prompt + completion == s.serialize([*context, *target])


@pytest.mark.parametrize("s", _serializers())
def test_prefix_then_value_recovers_through_decode(s: Serializer) -> None:
    known = [Field("city", "SP", False)]
    full = s.serialize([*known, Field("score", 3.5, True)])
    prefix = s.prefix(known, "score")
    assert full.startswith(prefix)
    assert s.decode_value(full[len(prefix) :], numeric=True) == 3.5


# --- value_spans -------------------------------------------------------------


@pytest.mark.parametrize("s", _serializers())
@pytest.mark.parametrize(
    "fields",
    [
        [Field("age", 39, True), Field("city", "SP", False), Field("score", -3.5, True)],
        [Field("score", 3.5, True)],
        [Field("city", "SP", False)],
        [],
    ],
)
def test_value_spans_slice_the_encoded_values(s: Serializer, fields: list[Field]) -> None:
    text = s.serialize(fields)
    spans = s.value_spans(fields)
    assert len(spans) == len(fields)
    for f, (start, end) in zip(fields, spans, strict=True):
        assert text[start:end] == s.encode_value(f.value, numeric=f.numeric)


# --- numeric_constraint -----------------------------------------------------


@pytest.mark.parametrize("s", _serializers())
def test_numeric_constraint_covers_encoded_numbers(s: Serializer) -> None:
    c = s.numeric_constraint()
    for value in (39, -3, 1250, 3.14159, 0.001, 1e16):
        assert set(s.encode_value(value, numeric=True)) <= c.alphabet


@pytest.mark.parametrize("s", _serializers())
def test_numeric_constraint_terminators_end_the_value(s: Serializer) -> None:
    c = s.numeric_constraint()
    encoded = s.encode_value(3.5, numeric=True)
    for terminator in c.terminators:
        assert s.decode_value(f"{encoded}{terminator}junk", numeric=True) == 3.5


def test_value_constraint_allows_values_terminators_and_nothing_else() -> None:
    c = JSONSerializer().numeric_constraint()
    assert c.allows(" 39")
    assert c.allows("3.5")
    assert c.allows(",")
    assert not c.allows("abc")
    assert not c.allows("3a")
    assert not c.allows("")


# --- resolve_serializer ---------------------------------------------------


@pytest.mark.parametrize(
    ("selector", "cls"),
    [
        ("json", JSONSerializer),
        ("key-value", KeyValueSerializer),
        ("bracket", BracketSerializer),
        ("if-then", IfThenSerializer),
    ],
)
def test_resolve_serializer_selectors(selector: str, cls: type) -> None:
    assert isinstance(resolve_serializer(selector), cls)


def test_resolve_serializer_passes_instances_through() -> None:
    s = BracketSerializer(number=SpacedDigits())
    assert resolve_serializer(s) is s


def test_resolve_serializer_applies_max_decimals_to_instances() -> None:
    # An explicit max_decimals reaches an instance's number format; the caller's
    # serializer object itself stays untouched.
    s = JSONSerializer(spaced_delimiters=True)
    resolved = resolve_serializer(s, 2)
    assert isinstance(resolved, JSONSerializer)
    number = resolved.number
    assert isinstance(number, PlainNumber)
    assert number.max_decimals == 2
    assert resolved.encode_value(4.2525, numeric=True) == "4.25"
    original = s.number
    assert isinstance(original, PlainNumber)
    assert original.max_decimals is None
    assert resolved.spaced_delimiters


def test_resolve_serializer_rejects_max_decimals_without_number_format() -> None:
    class Custom:
        def serialize(self, fields: object) -> str:
            return ""

    with pytest.raises(ValueError, match="max_decimals=2 cannot be applied"):
        resolve_serializer(Custom(), 2)  # pyright: ignore[reportArgumentType]


def test_resolve_serializer_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown serializer"):
        resolve_serializer("toml")


def test_resolve_serializer_number_format() -> None:
    plain = resolve_serializer("json", 2)
    spaced = resolve_serializer("json", 2, "spaced")
    assert isinstance(plain, JSONSerializer)
    assert isinstance(plain.number, PlainNumber)
    assert isinstance(spaced, JSONSerializer)
    assert isinstance(spaced.number, SpacedDigits)
    assert spaced.number.max_decimals == 2


def test_resolve_serializer_rejects_unknown_number_format() -> None:
    with pytest.raises(ValueError, match="unknown number_format"):
        # "roman" is not in the Literal -- the rejection under test
        resolve_serializer("json", None, "roman")  # pyright: ignore[reportArgumentType]
