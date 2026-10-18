# [MODULE DOC]
# Test for Series.head and Series.tail. Uses the string_series fixture. No
# import-time side effects.

import pandas._testing as tm


def test_head_tail(string_series):
    # [FUNCTION DOC]
    # head and tail default to five rows (first/last five), and a count of zero gives an
    # empty Series, each compared to the equivalent slice.
    tm.assert_series_equal(string_series.head(), string_series[:5])
    tm.assert_series_equal(string_series.head(0), string_series[0:0])
    tm.assert_series_equal(string_series.tail(), string_series[-5:])
    tm.assert_series_equal(string_series.tail(0), string_series[0:0])
