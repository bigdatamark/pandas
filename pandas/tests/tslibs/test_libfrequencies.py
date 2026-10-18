# [MODULE DOC]
# Tests for the helper that derives the anchor month name of a frequency string (used for period and quarter frequencies).

import pytest

from pandas._libs.tslibs.parsing import get_rule_month

from pandas.tseries import offsets


@pytest.mark.parametrize(
    "obj,expected",
    [
        ("W", "DEC"),
        (offsets.Week().freqstr, "DEC"),
        ("D", "DEC"),
        (offsets.Day().freqstr, "DEC"),
        ("Q", "DEC"),
        (offsets.QuarterEnd(startingMonth=12).freqstr, "DEC"),
        ("Q-JAN", "JAN"),
        (offsets.QuarterEnd(startingMonth=1).freqstr, "JAN"),
        ("Y-DEC", "DEC"),
        (offsets.YearEnd().freqstr, "DEC"),
        ("Y-MAY", "MAY"),
        (offsets.YearEnd(month=5).freqstr, "MAY"),
    ],
)
def test_get_rule_month(obj, expected):
    # [FUNCTION DOC]
    # Parametrized over frequency strings, given directly or taken from offset instances, with expected month abbreviations: weekly, daily and December-anchored quarterly or yearly strings give DEC, while January-anchored quarterly or May-anchored yearly strings give the corresponding month. Asserts the helper returns the expected month name.
    result = get_rule_month(obj)
    assert result == expected
