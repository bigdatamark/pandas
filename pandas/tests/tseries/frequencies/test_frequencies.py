# [MODULE DOC]
# Tests for the frequency-relationship helpers that decide whether one frequency is a
# sub-period or super-period of another. Contains a single parametrized test.
# No import-time side effects.
import pytest

from pandas._libs.tslibs import offsets

from pandas.tseries.frequencies import (
    is_subperiod,
    is_superperiod,
)


@pytest.mark.parametrize(
    "p1,p2,expected",
    [
        # Input validation.
        (offsets.MonthEnd(), None, False),
        (offsets.YearEnd(), None, False),
        (None, offsets.YearEnd(), False),
        (None, offsets.MonthEnd(), False),
        (None, None, False),
        (offsets.YearEnd(), offsets.MonthEnd(), True),
        (offsets.Hour(), offsets.Minute(), True),
        (offsets.Second(), offsets.Milli(), True),
        (offsets.Milli(), offsets.Micro(), True),
        (offsets.Micro(), offsets.Nano(), True),
    ],
)
def test_super_sub_symmetry(p1, p2, expected):
    # [FUNCTION DOC]
    # Parametrized over pairs of offsets (including missing values) and the expected boolean.
    # Verifies the super-period check on (first, second) and the sub-period check on the
    # swapped pair (second, first) both return exactly the expected boolean (identity
    # comparison). Missing inputs yield False; year-end/month-end and finer
    # time units (hour/minute, second/milli, milli/micro, micro/nano) yield True.
    assert is_superperiod(p1, p2) is expected
    assert is_subperiod(p2, p1) is expected
