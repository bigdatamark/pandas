# [MODULE DOC]
# Test of DatetimeIndex.snap (round each stamp to the nearest point of a frequency).
import pytest

from pandas.errors import Pandas4Warning

from pandas import (
    DatetimeIndex,
    date_range,
)
import pandas._testing as tm


@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
@pytest.mark.filterwarnings("ignore:Period with BDay freq:FutureWarning")
@pytest.mark.parametrize("tz", [None, "Asia/Shanghai", "Europe/Berlin"])
@pytest.mark.parametrize("name", [None, "my_dti"])
def test_dti_snap(name, tz, unit):
    # [FUNCTION DOC]
    # Parametrized over timezone (none, Shanghai, Berlin), name (none or a string), plus the unit
    # fixture; ignores deprecation warnings about business-day periods. Verifies snapping a week of
    # daily dates to the weekly Monday frequency repeats the nearest Monday (3 then 4 entries), and
    # snapping to business days moves weekend dates to the nearest business day (counts 1,1,1,2,2),
    # each time with the same timezone and name, and freq None on both result and expected. Building
    # the expected ranges with the lowercase deprecated frequency aliases is asserted to emit the
    # Pandas4Warning with a message that the alias is deprecated.
    dti = DatetimeIndex(
        [
            "1/1/2002",
            "1/2/2002",
            "1/3/2002",
            "1/4/2002",
            "1/5/2002",
            "1/6/2002",
            "1/7/2002",
        ],
        name=name,
        tz=tz,
        freq="D",
    )
    dti = dti.as_unit(unit)

    result = dti.snap(freq="W-MON")
    msg = "'w-mon' is deprecated and will be removed in a future version."
    with tm.assert_produces_warning(Pandas4Warning, match=msg):
        expected = date_range("12/31/2001", "1/7/2002", name=name, tz=tz, freq="w-mon")
    expected = expected.repeat([3, 4])
    expected = expected.as_unit(unit)
    tm.assert_index_equal(result, expected)
    assert result.tz == expected.tz
    assert result.freq is None
    assert expected.freq is None

    result = dti.snap(freq="B")

    msg = "'b' is deprecated and will be removed in a future version."
    with tm.assert_produces_warning(Pandas4Warning, match=msg):
        expected = date_range("1/1/2002", "1/7/2002", name=name, tz=tz, freq="b")
    expected = expected.repeat([1, 1, 1, 2, 2])
    expected = expected.as_unit(unit)
    tm.assert_index_equal(result, expected)
    assert result.tz == expected.tz
    assert result.freq is None
    assert expected.freq is None
