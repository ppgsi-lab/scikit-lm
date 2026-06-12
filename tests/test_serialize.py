from __future__ import annotations

import numpy as np
import pytest

from sklm import (
    BracketSerializer,
    Field,
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
    [
        [Field("age", 39, True), Field("city", "São Paulo", False)],
        [Field("age", 39, True)],
        [],
    ],
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
        KeyValueSerializer(),
        KeyValueSerializer(key_value_separator="=", pair_separator=";"),
        BracketSerializer(),
        BracketSerializer(number=SpacedDigits()),
    ]


@pytest.mark.parametrize("s", _serializers())
@pytest.mark.parametrize(
    "context",
    [
        [Field("age", 39, True), Field("city", "SP", False)],
        [Field("age", 39, True)],
        [],
    ],
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


# --- resolve_serializer ---------------------------------------------------


@pytest.mark.parametrize(
    ("selector", "cls"),
    [
        ("json", JSONSerializer),
        ("key-value", KeyValueSerializer),
        ("bracket", BracketSerializer),
    ],
)
def test_resolve_serializer_selectors(selector: str, cls: type) -> None:
    assert isinstance(resolve_serializer(selector), cls)


def test_resolve_serializer_passes_instances_through() -> None:
    s = BracketSerializer(number=SpacedDigits())
    assert resolve_serializer(s) is s


def test_resolve_serializer_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown serializer"):
        resolve_serializer("toml")
