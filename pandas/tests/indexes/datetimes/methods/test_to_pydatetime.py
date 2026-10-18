# [MODULE DOC]
# Tests for DatetimeIndex.to_pydatetime, including local and fixed-offset timezones. At import
# time it builds a module-level fixed-offset timezone (seven hours behind UTC) from the timezone
# test helper module.
from datetime import (
    datetime,
    timezone,
)

import dateutil.parser
import dateutil.tz
from dateutil.tz import tzlocal
import numpy as np

from pandas import (
    DatetimeIndex,
    date_range,
    to_datetime,
)
import pandas._testing as tm
from pandas.tests.indexes.datetimes.test_timezones import FixedOffset

fixed_off = FixedOffset(-420, "-07:00")


class TestToPyDatetime:
    # [CLASS DOC]
    # Groups to_pydatetime tests; no state.
    def test_dti_to_pydatetime(self):
        # [FUNCTION DOC]
        # Regression test: a datetime in the local timezone converted by to_datetime with utc gives a UTC
        # index; and python datetimes obtained from a local-timezone range also convert to UTC.
        dt = dateutil.parser.parse("2012-06-13T01:39:00Z")
        dt = dt.replace(tzinfo=tzlocal())

        arr = np.array([dt], dtype=object)

        result = to_datetime(arr, utc=True)
        assert result.tz is timezone.utc

        rng = date_range("2012-11-03 03:00", "2012-11-05 03:00", tz=tzlocal())
        arr = rng.to_pydatetime()
        result = to_datetime(arr, utc=True)
        assert result.tz is timezone.utc

    def test_dti_to_pydatetime_fizedtz(self):
        # [FUNCTION DOC]
        # Verifies that the object array of datetimes with a fixed offset timezone round-trips through a
        # DatetimeIndex and both to_pydatetime and the private matplotlib representation return an equal
        # array.
        dates = np.array(
            [
                datetime(2000, 1, 1, tzinfo=fixed_off),
                datetime(2000, 1, 2, tzinfo=fixed_off),
                datetime(2000, 1, 3, tzinfo=fixed_off),
            ]
        )
        dti = DatetimeIndex(dates)

        result = dti.to_pydatetime()
        tm.assert_numpy_array_equal(dates, result)

        result = dti._mpl_repr()
        tm.assert_numpy_array_equal(dates, result)
