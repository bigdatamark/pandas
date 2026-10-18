# [MODULE DOC]
# Tests for joining DatetimeIndex objects with other indexes: self joins,
# mixed dtypes, timezone handling, frequency preservation and errors on
# naive versus aware mixes.
# Uses the join_type, sort and using_infer_string fixtures from the shared conftest.

from datetime import (
    datetime,
    timezone,
)

import numpy as np
import pytest

from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    Timestamp,
    date_range,
    period_range,
    to_datetime,
)
import pandas._testing as tm

from pandas.tseries.offsets import (
    BDay,
    BMonthEnd,
)


class TestJoin:
    # [CLASS DOC]
    # Groups tests of the join method on DatetimeIndex.
    def test_does_not_convert_mixed_integer(self):
        # [FUNCTION DOC]
        # Verifies that an outer join of datetime columns with an integer row index gives
        # an object-dtype result, and that joining it back with the columns keeps the
        # same object dtype and values (no conversion to datetime).
        df = DataFrame(np.ones((3, 2)), columns=date_range("2020-01-01", periods=2))
        cols = df.columns.join(df.index, how="outer")
        joined = cols.join(df.columns)
        assert cols.dtype == np.dtype("O")
        assert cols.dtype == joined.dtype
        tm.assert_numpy_array_equal(cols.values, joined.values)

    def test_join_self(self, join_type):
        # [FUNCTION DOC]
        # Parametrized over join_type. Verifies joining an index with itself returns the
        # very same object.
        index = date_range("1/1/2000", periods=10)
        joined = index.join(index, how=join_type)
        assert index is joined

    def test_join_with_period_index(self, join_type):
        # [FUNCTION DOC]
        # Parametrized over join_type. Verifies joining datetime-index columns of a
        # frame: joining a period-index with a datetime-index gives the same result as
        # first casting the period-index to object dtype.
        df = DataFrame(
            np.ones((10, 2)),
            index=date_range("2020-01-01", periods=10),
            columns=period_range("2020-01-01", periods=2),
        )
        s = df.iloc[:5, 0]

        expected = df.columns.astype("O").join(s.index, how=join_type)
        result = df.columns.join(s.index, how=join_type)
        tm.assert_index_equal(expected, result)

    def test_join_object_index(self):
        # [FUNCTION DOC]
        # Verifies an outer join of a datetime index with an object index of strings
        # yields an index whose elements are Timestamp objects at the front.
        rng = date_range("1/1/2000", periods=10)
        idx = Index(["a", "b", "c", "d"])

        result = rng.join(idx, how="outer")
        assert isinstance(result[0], Timestamp)

    def test_join_utc_convert(self, join_type):
        # [FUNCTION DOC]
        # Parametrized over join_type. Joins two differently-converted views of a UTC
        # index. Verifies that joining with a same-zone slice keeps that zone, and that
        # joining with an index in another zone gives a DatetimeIndex in UTC.
        rng = date_range("1/1/2011", periods=100, freq="h", tz="utc")

        left = rng.tz_convert("US/Eastern")
        right = rng.tz_convert("Europe/Berlin")

        result = left.join(left[:-5], how=join_type)
        assert isinstance(result, DatetimeIndex)
        assert result.tz == left.tz

        result = left.join(right[:-5], how=join_type)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is timezone.utc

    def test_datetimeindex_union_join_empty(self, sort, using_infer_string):
        # [FUNCTION DOC]
        # Parametrized over sort and infer-string mode. Verifies a union of a datetime
        # index with an empty object index gives an object-dtype conversion of the
        # datetimes (or the datetime index unchanged when string inference is on), and
        # that joining with the empty index returns a datetime index equal to the
        # original.
        dti = date_range(start="1/1/2001", end="2/1/2001", freq="D")
        empty = Index([])

        result = dti.union(empty, sort=sort)
        if using_infer_string:
            assert isinstance(result, DatetimeIndex)
            tm.assert_index_equal(result, dti)
        else:
            expected = dti.astype("O")
            tm.assert_index_equal(result, expected)

        result = dti.join(empty)
        assert isinstance(result, DatetimeIndex)
        tm.assert_index_equal(result, dti)

    def test_join_nonunique(self):
        # [FUNCTION DOC]
        # Verifies an outer join of two non-unique datetime indexes (duplicates in each)
        # is monotonic increasing.
        idx1 = to_datetime(["2012-11-06 16:00:11.477563", "2012-11-06 16:00:11.477563"])
        idx2 = to_datetime(["2012-11-06 15:11:09.006507", "2012-11-06 15:11:09.006507"])
        rs = idx1.join(idx2, how="outer")
        assert rs.is_monotonic_increasing

    @pytest.mark.parametrize("freq", ["B", "C"])
    def test_outer_join(self, freq):
        # [FUNCTION DOC]
        # Parametrized over business frequencies. Verifies an outer join behaves like a
        # union: results are DatetimeIndex for overlapping, gapped and adjacent slices;
        # the frequency is none when there is a gap or when the offsets differ.
        # should just behave as union
        start, end = datetime(2009, 1, 1), datetime(2010, 1, 1)
        rng = date_range(start=start, end=end, freq=freq)

        # overlapping
        left = rng[:10]
        right = rng[5:10]

        the_join = left.join(right, how="outer")
        assert isinstance(the_join, DatetimeIndex)

        # non-overlapping, gap in middle
        left = rng[:5]
        right = rng[10:]

        the_join = left.join(right, how="outer")
        assert isinstance(the_join, DatetimeIndex)
        assert the_join.freq is None

        # non-overlapping, no gap
        left = rng[:5]
        right = rng[5:10]

        the_join = left.join(right, how="outer")
        assert isinstance(the_join, DatetimeIndex)

        # overlapping, but different offset
        other = date_range(start, end, freq=BMonthEnd())

        the_join = rng.join(other, how="outer")
        assert isinstance(the_join, DatetimeIndex)
        assert the_join.freq is None

    def test_naive_aware_conflicts(self):
        # [FUNCTION DOC]
        # Verifies joining a tz-naive and a tz-aware index raises a type error
        # mentioning naive and aware, in both argument orders.
        start, end = datetime(2009, 1, 1), datetime(2010, 1, 1)
        naive = date_range(start, end, freq=BDay(), tz=None)
        aware = date_range(start, end, freq=BDay(), tz="Asia/Hong_Kong")

        msg = "tz-naive.*tz-aware"
        with pytest.raises(TypeError, match=msg):
            naive.join(aware)

        with pytest.raises(TypeError, match=msg):
            aware.join(naive)

    @pytest.mark.parametrize("tz", [None, "US/Pacific"])
    def test_join_preserves_freq(self, tz):
        # [FUNCTION DOC]
        # Parametrized over timezone (none or US/Pacific). Verifies (GH#32157) an outer
        # join of two adjacent slices restores the full index and its frequency, and
        # that leaving a gap yields no frequency and equals the index with the missing
        # element deleted.
        # GH#32157
        dti = date_range("2016-01-01", periods=10, tz=tz)
        result = dti[:5].join(dti[5:], how="outer")
        assert result.freq == dti.freq
        tm.assert_index_equal(result, dti)

        result = dti[:5].join(dti[6:], how="outer")
        assert result.freq is None
        expected = dti.delete(5)
        tm.assert_index_equal(result, expected)
