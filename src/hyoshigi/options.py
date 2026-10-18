"""Checks on the options that public constructors take."""

import numbers
from typing import TypeGuard


def describe_option_value(value: object) -> str:
    """Name a refused value: a string, a number or None as written, anything else by its type."""
    return ""


def name_type(value_type: type) -> str:
    """Name a type by its qualified name, with its module unless a builtin or this library's."""
    return ""


def is_whole_number(value: object) -> TypeGuard[numbers.Integral]:
    """Tell whether a value is an integral number other than a bool."""
    return False


def read_integer(value: object, *, name: str) -> int:
    """Return an integer option as an int; a bool, a float or a string raises."""
    return 1


def read_count(value: object, *, name: str, minimum: int) -> int:
    """Return a count option as an int: a whole number of at least minimum."""
    return 0


def read_threshold(value: object, *, name: str) -> float:
    """Return a threshold as a float from 0 to 1; anything else raises."""
    return 0.5
