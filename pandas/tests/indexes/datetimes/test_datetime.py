# [MODULE DOC]
# Miscellaneous general tests for DatetimeIndex: identity, overflow with platform integers, NaT,
# append/union behavior, grouping, nanosecond frequencies, array conversion, and invalid
# frequency strings in range creation.
import datetime as dt
from datetime import date

import numpy as np
import pytest

from pandas.compat.numpy import np_long

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    Timestamp,
    date_range,
    offsets,
)
import pandas._testing as tm


class TestDatetimeIndex:
    # [CLASS DOC]
    # Groups general DatetimeIndex tests; defines one helper check method (not collected).
    def test_is_(self):
        # [FUNCTION DOC]
        # Verifies the identity check: an index is itself and its view but not its copy.
        dti = date_range(start="1/1/2005", end="12/1/2005", freq="ME")
        assert dti.is_(dti)
        assert dti.is_(dti.view())
        assert not dti.is_(dti.copy())

    def test_time_overflow_for_32bit_machines(self):
        # [FUNCTION DOC]
        # Regression test: building ranges from a platform-long number of periods with a second frequency
        # (both start-based and end-based) yields the requested length (guards against integer overflow
        # on 32-bit platforms).
        # GH8943.  On some machines NumPy defaults to np.int32 (for example,
        # 32-bit Linux machines).  In the function _generate_regular_range
        # found in tseries/index.py, `periods` gets multiplied by `strides`
        # (which has value 1e9) and since the max value for np.int32 is ~2e9,
        # and since those machines won't promote np.int32 to np.int64, we get
        # overflow.
        periods = np_long(1000)

        idx1 = date_range(start="2000", periods=periods, freq="s")
        assert len(idx1) == periods

        idx2 = date_range(end="2000", periods=periods, freq="s")
        assert len(idx2) == periods

    def test_nat(self):
        # [FUNCTION DOC]
        # Verifies that an index built from NaN has NaT as its first element.
        assert DatetimeIndex([np.nan])[0] is pd.NaT

    def test_week_of_month_frequency(self):
        # [FUNCTION DOC]
        # Regression test: appending and taking the union of indexes of python dates works (append
        # keeps order, union sorts).
        # GH 5348: "ValueError: Could not evaluate WOM-1SUN" shouldn't raise
        d1 = date(2002, 9, 1)
        d2 = date(2013, 10, 27)
        d3 = date(2012, 9, 30)
        idx1 = DatetimeIndex([d1, d2])
        idx2 = DatetimeIndex([d3])
        result_append = idx1.append(idx2)
        expected = DatetimeIndex([d1, d2, d3])
        tm.assert_index_equal(result_append, expected)
        result_union = idx1.union(idx2)
        expected = DatetimeIndex([d1, d3, d2])
        tm.assert_index_equal(result_union, expected)

    def test_append_nondatetimeindex(self):
        # [FUNCTION DOC]
        # Verifies appending a string Index to a DatetimeIndex yields an index whose first element is
        # still a Timestamp.
        rng = date_range("1/1/2000", periods=10)
        idx = Index(["a", "b", "c", "d"])

        result = rng.append(idx)
        assert isinstance(result[0], Timestamp)

    def test_misc_coverage(self):
        # [FUNCTION DOC]
        # Verifies grouping an index by its day attribute returns groups whose members are Timestamps.
        rng = date_range("1/1/2000", periods=5)
        result = rng.groupby(rng.day)
        assert isinstance(next(iter(result.values()))[0], Timestamp)

    # TODO: belongs in frame groupby tests?
    def test_groupby_function_tuple_1677(self):
        # [FUNCTION DOC]
        # Regression test: grouping a frame by a function returning (year, month) tuples and taking the
        # mean gives an index of tuples. (A TODO suggests it belongs with frame groupby tests.)
        df = DataFrame(
            np.random.default_rng(2).random(100),
            index=date_range("1/1/2000", periods=100),
        )
        monthly_group = df.groupby(lambda x: (x.year, x.month))

        result = monthly_group.mean()
        assert isinstance(result.index[0], tuple)

    def assert_index_parameters(self, index):
        # [FUNCTION DOC]
        # HELPER: used by test_ns_index. Asserts the index freq and inferred freq both equal the
        # 40960-nanosecond frequency string.
        assert index.freq == "40960ns"
        assert index.inferred_freq == "40960ns"

    def test_ns_index(self):
        # [FUNCTION DOC]
        # Verifies an index of 400 samples spaced by about 41 microseconds has a nanosecond freq that is
        # retained and inferred, and that a date_range using the same start, end and freq has the same
        # parameters.
        nsamples = 400
        ns = int(1e9 / 24414)
        dtstart = np.datetime64("2012-09-20T00:00:00")

        dt = dtstart + np.arange(nsamples) * np.timedelta64(ns, "ns")
        freq = ns * offsets.Nano()
        index = DatetimeIndex(dt, freq=freq, name="time")
        self.assert_index_parameters(index)

        new_index = date_range(start=index[0], end=index[-1], freq=index.freq)
        self.assert_index_parameters(new_index)

    def test_asarray_tz_naive(self):
        # [FUNCTION DOC]
        # Verifies converting a naive index to a numpy array gives nanosecond datetime64 values with no
        # warning, and as object gives Timestamps.
        # This shouldn't produce a warning.
        idx = date_range("2000", periods=2, unit="ns")
        # M8[ns] by default
        result = np.asarray(idx)

        expected = np.array(["2000-01-01", "2000-01-02"], dtype="M8[ns]")
        tm.assert_numpy_array_equal(result, expected)

        # optionally, object
        result = np.asarray(idx, dtype=object)

        expected = np.array([Timestamp("2000-01-01"), Timestamp("2000-01-02")])
        tm.assert_numpy_array_equal(result, expected)

    def test_asarray_tz_aware(self):
        # [FUNCTION DOC]
        # Verifies converting a US/Central index to a datetime64 array gives UTC values, as it did
        # before, with no warning, and converting to object gives timezone-aware Timestamps.
        tz = "US/Central"
        idx = date_range("2000", periods=2, tz=tz, unit="ns")
        expected = np.array(["2000-01-01T06", "2000-01-02T06"], dtype="M8[ns]")
        result = np.asarray(idx, dtype="datetime64[ns]")

        tm.assert_numpy_array_equal(result, expected)

        # Old behavior with no warning
        result = np.asarray(idx, dtype="M8[ns]")

        tm.assert_numpy_array_equal(result, expected)

        # Future behavior with no warning
        expected = np.array(
            [Timestamp("2000-01-01", tz=tz), Timestamp("2000-01-02", tz=tz)]
        )
        result = np.asarray(idx, dtype=object)

        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("freq", ["2H", "2BH", "2S"])
    def test_CBH_raises(self, freq):
        # [FUNCTION DOC]
        # Parametrized over frequency strings (uppercase hour, business hour, and second aliases that
        # are no longer valid). Verifies date_range raises a value error saying the frequency is invalid.
        msg = f"Invalid frequency: {freq}"

        with pytest.raises(ValueError, match=msg):
            date_range(dt.datetime(2022, 12, 11), dt.datetime(2022, 12, 13), freq=freq)

    @pytest.mark.parametrize("freq", ["2BM", "1bm", "2BQ", "1BQ-MAR", "2BY-JUN", "1by"])
    def test_BM_BQ_BY_raises(self, freq):
        # [FUNCTION DOC]
        # Parametrized over invalid business month/quarter/year frequency strings. Verifies date_range
        # raises a value error that the frequency is invalid.
        msg = f"Invalid frequency: {freq}"

        with pytest.raises(ValueError, match=msg):
            date_range(start="2016-02-21", end="2016-08-21", freq=freq)

    @pytest.mark.parametrize("freq", ["2BA-MAR", "1BAS-MAY", "2AS-AUG"])
    def test_BA_BAS_raises(self, freq):
        # [FUNCTION DOC]
        # Parametrized over invalid business-annual style frequency strings. Verifies date_range raises a
        # value error that the frequency is invalid.
        msg = f"Invalid frequency: {freq}"

        with pytest.raises(ValueError, match=msg):
            date_range(start="2016-02-21", end="2016-08-21", freq=freq)
