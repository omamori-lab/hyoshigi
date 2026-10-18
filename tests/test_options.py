"""The checks on constructor options: counts, integers and thresholds."""

import math
import numbers
from decimal import Decimal
from fractions import Fraction

import pytest

from hyoshigi.errors import ConfigurationError
from hyoshigi.options import (
    check_instance,
    describe_option_value,
    read_count,
    read_integer,
    read_threshold,
)

HUGE = 10**5000


class ArrayInteger:
    """Stands in for numpy's integers: registered as integral, but not a subclass of int."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __index__(self) -> int:
        return self.value


numbers.Integral.register(ArrayInteger)


@pytest.mark.parametrize(
    ("value", "described"),
    [
        (None, "None"),
        ("text", "'text'"),
        (2.5, "2.5"),
        (Decimal("sNaN"), "Decimal('sNaN')"),
        ([0.6], "an instance of list"),
        (int, "the class int; pass an instance"),
        (ConfigurationError, "the class ConfigurationError; pass an instance"),
        (ConfigurationError("x"), "an instance of ConfigurationError"),
        (type("bool", (), {"__module__": "numpy"})(), "an instance of numpy.bool"),
        (HUGE, "an integer too long to write out"),
        (Fraction(HUGE, 3), "a fractions.Fraction too long to write out"),
    ],
    ids=[
        "none",
        "string",
        "float",
        "decimal",
        "list",
        "builtin-class",
        "own-class",
        "own-instance",
        "foreign-type-sharing-a-builtin-name",
        "integer-too-long",
        "fraction-too-long",
    ],
)
def test_a_refused_value_is_described_without_writing_out_a_long_one(
    value: object, described: str
) -> None:
    description = describe_option_value(value)

    assert description == described


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1, 1), (3, 3), (ArrayInteger(3), 3)],
    ids=["the-minimum", "above-the-minimum", "array-integer"],
)
def test_a_count_is_a_whole_number_of_at_least_the_minimum(value: object, expected: int) -> None:
    count = read_count(value, name="ensemble_size", minimum=1)

    assert (count, type(count)) == (expected, int)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (2.5, "ensemble_size must be a whole number of at least 1, got 2.5"),
        (2.0, "ensemble_size must be a whole number of at least 1, got 2.0"),
        ("3", "ensemble_size must be a whole number of at least 1, got '3'"),
        (True, "ensemble_size must be a whole number of at least 1, got True"),
        (None, "ensemble_size must be a whole number of at least 1, got None"),
        (0, "ensemble_size must be at least 1, got 0"),
        (-3, "ensemble_size must be at least 1, got -3"),
        (ArrayInteger(0), "ensemble_size must be at least 1, got 0"),
    ],
    ids=["float", "whole-float", "string", "bool", "none", "zero", "negative", "array-integer"],
)
def test_a_count_that_is_not_a_whole_number_of_at_least_the_minimum_raises(
    value: object, message: str
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        read_count(value, name="ensemble_size", minimum=1)

    assert str(raised.value) == message


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-5, -5), (0, 0), (9, 9), (ArrayInteger(9), 9)],
    ids=["negative", "zero", "positive", "array-integer"],
)
def test_an_integer_is_any_whole_number(value: object, expected: int) -> None:
    integer = read_integer(value, name="lowest_score")

    assert (integer, type(integer)) == (expected, int)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (2.5, "lowest_score must be an integer, got 2.5"),
        (2.0, "lowest_score must be an integer, got 2.0"),
        ("3", "lowest_score must be an integer, got '3'"),
        (True, "lowest_score must be an integer, got True"),
        (math.nan, "lowest_score must be an integer, got nan"),
    ],
    ids=["float", "whole-float", "string", "bool", "nan"],
)
def test_an_integer_that_is_not_a_whole_number_raises(value: object, message: str) -> None:
    with pytest.raises(ConfigurationError) as raised:
        read_integer(value, name="lowest_score")

    assert str(raised.value) == message


# Any real number counts, and so does a Decimal; the range is checked on the exact value.
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0.0),
        (1, 1.0),
        (0.6, 0.6),
        (Fraction(3, 5), 0.6),
        (Decimal("0.6"), 0.6),
        (Decimal("0.3"), 0.3),
    ],
    ids=["zero", "one", "float", "fraction", "decimal", "decimal-below-its-float"],
)
def test_a_threshold_is_any_real_number_from_zero_to_one_kept_as_a_float(
    value: object, expected: float
) -> None:
    threshold = read_threshold(value, name="escalation_threshold")

    assert (threshold, type(threshold)) == (expected, float)


def test_a_negative_zero_threshold_is_read_as_zero() -> None:
    threshold = read_threshold(-0.0, name="escalation_threshold")

    assert repr(threshold) == "0.0"


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("0.6", "escalation_threshold must be a number between 0 and 1, got '0.6'"),
        (True, "escalation_threshold must be a number between 0 and 1, got True"),
        (None, "escalation_threshold must be a number between 0 and 1, got None"),
        ([0.6], "escalation_threshold must be a number between 0 and 1, got an instance of list"),
        (math.nan, "escalation_threshold must be between 0 and 1, got nan"),
        (1.5, "escalation_threshold must be between 0 and 1, got 1.5"),
        (-0.1, "escalation_threshold must be between 0 and 1, got -0.1"),
        (Decimal("sNaN"), "escalation_threshold must be between 0 and 1, got Decimal('sNaN')"),
    ],
    ids=["string", "bool", "none", "list", "nan", "above-one", "below-zero", "signalling-nan"],
)
def test_a_threshold_that_is_not_a_number_from_zero_to_one_raises(
    value: object, message: str
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        read_threshold(value, name="escalation_threshold")

    assert str(raised.value) == message


# A float would round the first four into [0, 1].
@pytest.mark.parametrize(
    "value",
    [
        Decimal("1.0000000000000000001"),
        Decimal("-1E-400"),
        Fraction(10**20 + 1, 10**20),
        Fraction(-1, 10**400),
        Decimal("NaN"),
        Decimal("Infinity"),
        10**400,
        Fraction(10**400, 1),
        math.inf,
    ],
    ids=[
        "decimal-just-above-one",
        "decimal-just-below-zero",
        "fraction-just-above-one",
        "fraction-just-below-zero",
        "decimal-nan",
        "decimal-infinity",
        "huge-integer",
        "huge-fraction",
        "infinity",
    ],
)
def test_a_threshold_just_outside_the_range_raises_before_it_is_rounded(value: object) -> None:
    with pytest.raises(ConfigurationError, match="escalation_threshold must be between 0 and 1"):
        read_threshold(value, name="escalation_threshold")


# Classes outside hyoshigi are named with their module, here test_options.
class Ordinary:
    """A class whose instances an option may require."""


class AnotherOrdinary:
    """A class whose instances are not Ordinary's."""


def test_an_instance_of_the_expected_class_is_accepted() -> None:
    accepted = check_instance(Ordinary(), expected=Ordinary, name="model")

    assert accepted is None


@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize(
    ("value", "expected", "hint", "message"),
    [
        (
            AnotherOrdinary(),
            Ordinary,
            "",
            "model must be an Ordinary, got an instance of test_options.AnotherOrdinary",
        ),
        (
            Ordinary,
            Ordinary,
            "",
            "model must be an Ordinary, got the class test_options.Ordinary; pass an instance",
        ),
        (
            AnotherOrdinary(),
            Ordinary,
            "Use the async class for an async model.",
            "model must be an Ordinary, got an instance of test_options.AnotherOrdinary. Use the "
            "async class for an async model.",
        ),
        (3, Decimal, "", "model must be a Decimal, got 3"),
    ],
    ids=["another-class", "the-class-itself", "with-a-hint", "a-consonant"],
)
def test_a_value_that_is_not_an_instance_of_the_expected_class_raises_naming_both(
    value: object, expected: type, hint: str, message: str
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        check_instance(value, expected=expected, name="model", hint=hint)

    assert str(raised.value) == message
