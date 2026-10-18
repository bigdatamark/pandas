# [MODULE DOC]
# Tests that the date and timedelta field extraction routines accept read-only integer arrays. No import-time side effects.

import numpy as np
import pytest

from pandas._libs.tslibs import fields

import pandas._testing as tm


@pytest.fixture
def dtindex():
    # [FUNCTION DOC]
    # HELPER: fixture returning a read-only int64 array of five values spaced 32 days apart in nanoseconds from the epoch (nominally the start of five consecutive months from January 1970).
    dtindex = np.arange(5, dtype=np.int64) * 10**9 * 3600 * 24 * 32
    dtindex.flags.writeable = False
    return dtindex


def test_get_date_name_field_readonly(dtindex):
    # [FUNCTION DOC]
    # Regression test (vaex issue 357): extracting the month name from read-only data does not raise and gives the five month names January to May as an object array.
    # https://github.com/vaexio/vaex/issues/357
    #  fields functions shouldn't raise when we pass read-only data
    result = fields.get_date_name_field(dtindex, "month_name")
    expected = np.array(["January", "February", "March", "April", "May"], dtype=object)
    tm.assert_numpy_array_equal(result, expected)


def test_get_date_field_readonly(dtindex):
    # [FUNCTION DOC]
    # Extracting the year field from read-only data gives five 1970 values as 32-bit integers.
    result = fields.get_date_field(dtindex, "Y")
    expected = np.array([1970, 1970, 1970, 1970, 1970], dtype=np.int32)
    tm.assert_numpy_array_equal(result, expected)


def test_get_start_end_field_readonly(dtindex):
    # [FUNCTION DOC]
    # Extracting the is-month-start flag (no frequency given) from read-only data gives a boolean array with only the first element true.
    result = fields.get_start_end_field(dtindex, "is_month_start", None)
    expected = np.array([True, False, False, False, False], dtype=np.bool_)
    tm.assert_numpy_array_equal(result, expected)


def test_get_timedelta_field_readonly(dtindex):
    # [FUNCTION DOC]
    # Treating the same values as timedeltas, extracting the seconds field from read-only data gives five zeros as 32-bit integers.
    # treat dtindex as timedeltas for this next one
    result = fields.get_timedelta_field(dtindex, "seconds")
    expected = np.array([0] * 5, dtype=np.int32)
    tm.assert_numpy_array_equal(result, expected)
