# [MODULE DOC]
# Tests for the time-parsing conversion function that turns strings, time objects
# and array-likes of those into time-of-day values.
# Defines a locale-dependent xfail marker (non-strict, applied when the machine locale
# is Chinese or Italian, since am/pm parsing fails there) and one test class.
# Relies on the testing helpers module for series comparison. No other import-time effects.
from datetime import time
import locale

import numpy as np
import pytest

from pandas import Series
import pandas._testing as tm
from pandas.core.tools.times import to_time

# The tests marked with this are locale-dependent.
# They pass, except when the machine locale is zh_CN or it_IT.
fails_on_non_english = pytest.mark.xfail(
    locale.getlocale()[0] in ("zh_CN", "it_IT"),
    reason="fail on a CI build with LC_ALL=zh_CN.utf8/it_IT.utf8",
    strict=False,
)


class TestToTime:
    # [CLASS DOC]
    # Groups tests of the time conversion function: single-value parsing, custom format
    # strings, and array-like inputs including error modes.
    @pytest.mark.parametrize(
        "time_string",
        [
            "14:15",
            "1415",
            pytest.param("2:15pm", marks=fails_on_non_english),
            pytest.param("0215pm", marks=fails_on_non_english),
            "14:15:00",
            "141500",
            pytest.param("2:15:00pm", marks=fails_on_non_english),
            pytest.param("021500pm", marks=fails_on_non_english),
            time(14, 15),
        ],
    )
    def test_parsers_time(self, time_string):
        # [FUNCTION DOC]
        # Parametrized over many spellings of 2:15 pm (24-hour with and without colons, 12-hour
        # with am/pm marker, with seconds, and an existing time object); the 12-hour variants
        # carry the locale xfail marker. Verifies each converts to the time 14:15.
        # GH#11818
        assert to_time(time_string) == time(14, 15)

    def test_odd_format(self):
        # [FUNCTION DOC]
        # Verifies that an explicit format string with a dot separator parses the string
        # "14.15" into the time 14:15.
        new_string = "14.15"
        assert to_time(new_string, format="%H.%M") == time(14, 15)

    def test_arraylike(self):
        # [FUNCTION DOC]
        # Verifies array-like handling using a list of two time strings. Checks that the result is
        # a list of time values with default settings, with an explicit format, and with time
        # format inference on. With a non-matching 12-hour format and coerce mode, expects a list of
        # missing values. The ignore error mode is expected to raise a value error (message says
        # errors must be one of the allowed options), the raise mode raises a value error saying
        # it cannot convert to a time with the given format. A named Series input returns an
        # equal Series with the same name; a NumPy array input returns a plain list.
        arg = ["14:15", "20:20"]
        expected_arr = [time(14, 15), time(20, 20)]
        assert to_time(arg) == expected_arr
        assert to_time(arg, format="%H:%M") == expected_arr
        assert to_time(arg, infer_time_format=True) == expected_arr
        assert to_time(arg, format="%I:%M%p", errors="coerce") == [None, None]

        with pytest.raises(ValueError, match="errors must be"):
            to_time(arg, format="%I:%M%p", errors="ignore")

        msg = "Cannot convert.+to a time with given format"
        with pytest.raises(ValueError, match=msg):
            to_time(arg, format="%I:%M%p", errors="raise")

        tm.assert_series_equal(
            to_time(Series(arg, name="test")), Series(expected_arr, name="test")
        )

        res = to_time(np.array(arg))
        assert isinstance(res, list)
        assert res == expected_arr
