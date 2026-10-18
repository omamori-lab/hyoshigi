"""Checks on the options that public constructors take.

A constructor checks the type of each option, so a count given as a float fails when the
object is built and not at the first rating.
"""

import numbers
import operator
from decimal import Decimal
from typing import TypeGuard

from hyoshigi.errors import ConfigurationError


def describe_option_value(value: object) -> str:
    """Name a refused value: a string, a number or None as written, anything else by its type.

    A class given where an instance belongs is named as the class. Python refuses to write
    an integer of more than 4,300 digits, so a number it cannot write, such as 10**5000 or
    a Fraction built on one, is named by its type and said to be too long.
    """
    if value is None or isinstance(value, str | numbers.Number):
        try:
            return repr(value)
        except ValueError:
            kind = (
                "an integer"
                if isinstance(value, numbers.Integral)
                else f"a {name_type(type(value))}"
            )
            return f"{kind} too long to write out"
    if isinstance(value, type):
        return f"the class {name_type(value)}; pass an instance"
    return f"an instance of {name_type(type(value))}"


def name_type(value_type: type) -> str:
    """Name a type by its qualified name, with its module unless a builtin or this library's."""
    module = value_type.__module__
    is_own = module == "builtins" or module.split(".")[0] == __name__.split(".")[0]
    return value_type.__qualname__ if is_own else f"{module}.{value_type.__qualname__}"


def is_whole_number(value: object) -> TypeGuard[numbers.Integral]:
    """Tell whether a value is an integral number other than a bool.

    numpy's integers count, since they are counted with through __index__. A bool does not,
    although Python counts it as an int, and nor does a float such as 2.0.
    """
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def read_integer(value: object, *, name: str) -> int:
    """Return an integer option as an int.

    Any whole number counts, negative ones and numpy's integers included; a bool, a float
    such as 2.0 or a string raises ConfigurationError.
    """
    if not is_whole_number(value):
        raise ConfigurationError(f"{name} must be an integer, got {describe_option_value(value)}")
    return operator.index(value)


def read_count(value: object, *, name: str, minimum: int) -> int:
    """Return a count option as an int: a whole number of at least minimum.

    Anything else raises ConfigurationError.
    """
    if not is_whole_number(value):
        raise ConfigurationError(
            f"{name} must be a whole number of at least {minimum}, "
            f"got {describe_option_value(value)}"
        )
    count = operator.index(value)
    if count < minimum:
        raise ConfigurationError(
            f"{name} must be at least {minimum}, got {describe_option_value(count)}"
        )
    return count


def read_threshold(value: object, *, name: str) -> float:
    """Return a threshold as a float from 0 to 1; anything else raises ConfigurationError.

    Any real number counts, a Fraction or numpy's floats included, and so does a Decimal. A
    bool does not, although Python counts it as a number, and neither does a string such as
    "0.6". The range is checked on the exact value, before it becomes a float, so a value
    just outside [0, 1] that a float would round into it is refused, as is NaN. -0.0 is
    read as 0.0.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real | Decimal):
        raise ConfigurationError(
            f"{name} must be a number between 0 and 1, got {describe_option_value(value)}"
        )
    try:
        is_in_range = not value < 0 and value <= 1
    except ArithmeticError:
        # A Decimal NaN raises InvalidOperation when compared, and is out of range too.
        is_in_range = False
    if not is_in_range:
        raise ConfigurationError(
            f"{name} must be between 0 and 1, got {describe_option_value(value)}"
        )
    # Adding 0.0 turns -0.0 into 0.0.
    return float(value) + 0.0
