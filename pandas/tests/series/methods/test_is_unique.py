# [MODULE DOC]
# Tests for the Series is_unique property, including missing values and an
# exception-raising inequality operator. No import-time side effects.

import numpy as np
import pytest

from pandas import Series


@pytest.mark.parametrize(
    "data, expected",
    [
        (np.random.default_rng(2).integers(0, 10, size=1000), False),
        (np.arange(1000), True),
        ([], True),
        ([np.nan], True),
        (["foo", "bar", np.nan], True),
        (["foo", "foo", np.nan], False),
        (["foo", "bar", np.nan, np.nan], False),
    ],
)
def test_is_unique(data, expected):
    # [FUNCTION DOC]
    # Regression test (GH#11946, GH#25180), parametrized over data and the expected
    # flag: random integers with repeats are not unique, a range, an empty list and a
    # single NaN are unique; missing values count as one value (two NaN are not
    # unique). The property must be exactly a boolean.
    # GH#11946 / GH#25180
    ser = Series(data)
    assert ser.is_unique is expected


def test_is_unique_class_ne(capsys):
    # [FUNCTION DOC]
    # Regression test (GH#20661): checking uniqueness of a Series of objects whose
    # inequality comparison raises an exception must not write anything to standard
    # error. Uses the capsys fixture to capture output.
    # GH#20661
    class Foo:
        # [CLASS DOC]
        # HELPER: local test class holding a value; its inequality operator always raises
        # an exception to prove that the uniqueness check does not rely on it.
        def __init__(self, val) -> None:
            # [FUNCTION DOC]
            # HELPER: constructor storing the given value in an attribute.
            self._value = val

        def __ne__(self, other):
            # [FUNCTION DOC]
            # HELPER: inequality operator that always raises a generic exception with the
            # message "NEQ not supported" (takes the other operand but ignores it).
            raise Exception("NEQ not supported")

    with capsys.disabled():
        li = [Foo(i) for i in range(5)]
        ser = Series(li, index=list(range(5)))

    ser.is_unique
    captured = capsys.readouterr()
    assert len(captured.err) == 0
