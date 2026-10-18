# [MODULE DOC]
# Tests for DatetimeIndex.asof (last label at or before a key).
from datetime import timedelta

from pandas import (
    DatetimeIndex,
    Index,
    Timestamp,
    date_range,
    isna,
)


class TestAsOf:
    # [CLASS DOC]
    # Groups asof tests for DatetimeIndex; no state.
    def test_asof_partial(self):
        # [FUNCTION DOC]
        # Verifies asof with a partial-date string on a month-end index returns the Timestamp of the
        # last month end within that month, as a scalar rather than an Index.
        index = date_range("2010-01-01", periods=2, freq="ME")
        expected = Timestamp("2010-02-28")
        result = index.asof("2010-02")
        assert result == expected
        assert not isinstance(result, Index)

    def test_asof(self):
        # [FUNCTION DOC]
        # Verifies asof on a daily index: an exact label returns itself, a key before the first label
        # returns NA, a key after the last label returns the last, and a python datetime key gives a
        # Timestamp.
        index = date_range("2020-01-01", periods=10)

        dt = index[0]
        assert index.asof(dt) == dt
        assert isna(index.asof(dt - timedelta(1)))

        dt = index[-1]
        assert index.asof(dt + timedelta(1)) == dt

        dt = index[0].to_pydatetime()
        assert isinstance(index.asof(dt), Timestamp)

    def test_asof_datetime_string(self):
        # [FUNCTION DOC]
        # Regression test: asof with a date string on a daily index returns the matching label, also
        # after appending a non-midnight timestamp past the key.
        # GH#50946

        dti = date_range("2021-08-05", "2021-08-10", freq="1D")

        key = "2021-08-09"
        res = dti.asof(key)
        exp = dti[4]
        assert res == exp

        # add a non-midnight time caused a bug
        dti2 = DatetimeIndex([*list(dti), "2021-08-11 00:00:01"])
        res = dti2.asof(key)
        assert res == exp
