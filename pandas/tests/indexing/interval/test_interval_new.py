# [MODULE DOC]
# Tests for loc-style indexing of Series labelled by an interval index under the current (new) semantics: exact interval matches, scalar containment, slices, overlap errors, non-unique intervals, error messages for missing keys, and a regression test for repeated intervals with infinities. Most tests take the indexer_sl fixture.

import re

import numpy as np
import pytest

from pandas.compat import WASM

from pandas import (
    Index,
    Interval,
    IntervalIndex,
    Series,
)
import pandas._testing as tm


class TestIntervalIndex:
    # [CLASS DOC]
    # Test class for interval-labelled loc behavior; provides a series fixture.
    @pytest.fixture
    def series_with_interval_index(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a series of integers zero to four labelled by a right-closed interval index with breaks zero through five.
        """
        Fixture providing a Series with an IntervalIndex.
        """
        return Series(np.arange(5), IntervalIndex.from_breaks(np.arange(6)))

    def test_loc_with_interval(self, series_with_interval_index, indexer_sl):
        # [FUNCTION DOC]
        # Interval keys match only exactly: an existing interval returns its value; a list of intervals returns the matching rows; non-matching intervals (different closedness, partially covering, below range, beyond range) raise a key error naming the interval.
        # loc with single label / list of labels:
        #   - Intervals: only exact matches
        #   - scalars: those that contain it

        ser = series_with_interval_index.copy()

        expected = 0
        result = indexer_sl(ser)[Interval(0, 1)]
        assert result == expected

        expected = ser.iloc[3:5]
        result = indexer_sl(ser)[[Interval(3, 4), Interval(4, 5)]]
        tm.assert_series_equal(expected, result)

        # missing or not exact
        with pytest.raises(KeyError, match=re.escape("Interval(3, 5, closed='left')")):
            indexer_sl(ser)[Interval(3, 5, closed="left")]

        with pytest.raises(KeyError, match=re.escape("Interval(3, 5, closed='right')")):
            indexer_sl(ser)[Interval(3, 5)]

        with pytest.raises(
            KeyError, match=re.escape("Interval(-2, 0, closed='right')")
        ):
            indexer_sl(ser)[Interval(-2, 0)]

        with pytest.raises(KeyError, match=re.escape("Interval(5, 6, closed='right')")):
            indexer_sl(ser)[Interval(5, 6)]

    def test_loc_with_scalar(self, series_with_interval_index, indexer_sl):
        # [FUNCTION DOC]
        # Scalar keys select the interval containing them (including a boundary belonging to the right-closed interval); lists of scalars give the corresponding rows (repeated keys give repeated rows); a boolean mask works.
        # loc with single label / list of labels:
        #   - Intervals: only exact matches
        #   - scalars: those that contain it

        ser = series_with_interval_index.copy()

        assert indexer_sl(ser)[1] == 0
        assert indexer_sl(ser)[1.5] == 1
        assert indexer_sl(ser)[2] == 1

        expected = ser.iloc[1:4]
        tm.assert_series_equal(expected, indexer_sl(ser)[[1.5, 2.5, 3.5]])
        tm.assert_series_equal(expected, indexer_sl(ser)[[2, 3, 4]])
        tm.assert_series_equal(expected, indexer_sl(ser)[[1.5, 3, 4]])

        expected = ser.iloc[[1, 1, 2, 1]]
        tm.assert_series_equal(expected, indexer_sl(ser)[[1.5, 2, 2.5, 1.5]])

        expected = ser.iloc[2:5]
        tm.assert_series_equal(expected, indexer_sl(ser)[ser >= 2])

    def test_loc_with_slices(self, series_with_interval_index, indexer_sl):
        # [FUNCTION DOC]
        # Slices of interval objects select between the exactly matching endpoints; an open-ended slice from an interval works; start intervals not exactly matching (wrong bounds or closed side) raise a not-implemented error saying interval objects are not currently supported.
        # loc with slices:
        #   - Interval objects: only works with exact matches
        #   - scalars: only works for non-overlapping, monotonic intervals,
        #     and start/stop select location based on the interval that
        #     contains them:
        #    (slice_loc(start, stop) == (idx.get_loc(start), idx.get_loc(stop))

        ser = series_with_interval_index.copy()

        # slice of interval

        expected = ser.iloc[:3]
        result = indexer_sl(ser)[Interval(0, 1) : Interval(2, 3)]
        tm.assert_series_equal(expected, result)

        expected = ser.iloc[3:]
        result = indexer_sl(ser)[Interval(3, 4) :]
        tm.assert_series_equal(expected, result)

        msg = "Interval objects are not currently supported"
        with pytest.raises(NotImplementedError, match=msg):
            indexer_sl(ser)[Interval(3, 6) :]

        with pytest.raises(NotImplementedError, match=msg):
            indexer_sl(ser)[Interval(3, 4, closed="left") :]

    def test_slice_step_ne1(self, series_with_interval_index):
        # [FUNCTION DOC]
        # Integer slicing with a step other than one is positional and matches taking the slice then stepping.
        # GH#31658 slice of scalar with step != 1
        ser = series_with_interval_index.copy()
        expected = ser.iloc[0:4:2]

        result = ser[0:4:2]
        tm.assert_series_equal(result, expected)

        result2 = ser[0:4][::2]
        tm.assert_series_equal(result2, expected)

    def test_slice_float_start_stop(self, series_with_interval_index):
        # [FUNCTION DOC]
        # Float slice bounds with a step of two raise a value error saying label-based slicing with step other than one is not supported for interval indexes.
        # GH#31658 slicing with integers is positional, with floats is not
        #  supported
        ser = series_with_interval_index.copy()

        msg = "label-based slicing with step!=1 is not supported for IntervalIndex"
        with pytest.raises(ValueError, match=msg):
            ser[1.5:9.5:2]

    def test_slice_interval_step(self, series_with_interval_index):
        # [FUNCTION DOC]
        # An interval used as a slice step raises the same value error.
        # GH#31658 allows for integer step!=1, not Interval step
        ser = series_with_interval_index.copy()
        msg = "label-based slicing with step!=1 is not supported for IntervalIndex"
        with pytest.raises(ValueError, match=msg):
            ser[0 : 4 : Interval(0, 1)]

    def test_loc_with_overlap(self, indexer_sl):
        # [FUNCTION DOC]
        # For an index of overlapping intervals: a scalar in both intervals returns both rows; a list of that scalar likewise; exact interval keys return the single row (a list of both exact intervals returns all); a non-exact interval raises a key error naming it; a list of only non-exact raises key error "None of ... are in the [index]"; slicing between exact intervals works; slicing by non-matching intervals, and (for loc) by scalar bounds on overlapping intervals, raises a key error stating slices need non-overlapping monotonic bounds. (A source TODO questions whether key error is the right exception.)
        idx = IntervalIndex.from_tuples([(1, 5), (3, 7)])
        ser = Series(range(len(idx)), index=idx)

        # scalar
        expected = ser
        result = indexer_sl(ser)[4]
        tm.assert_series_equal(expected, result)

        result = indexer_sl(ser)[[4]]
        tm.assert_series_equal(expected, result)

        # interval
        expected = 0
        result = indexer_sl(ser)[Interval(1, 5)]
        assert expected == result

        expected = ser
        result = indexer_sl(ser)[[Interval(1, 5), Interval(3, 7)]]
        tm.assert_series_equal(expected, result)

        with pytest.raises(KeyError, match=re.escape("Interval(3, 5, closed='right')")):
            indexer_sl(ser)[Interval(3, 5)]

        msg = (
            r"None of \[IntervalIndex\(\[\(3, 5\]\], "
            r"dtype='interval\[int64, right\]'\)\] are in the \[index\]"
        )
        with pytest.raises(KeyError, match=msg):
            indexer_sl(ser)[[Interval(3, 5)]]

        # slices with interval (only exact matches)
        expected = ser
        result = indexer_sl(ser)[Interval(1, 5) : Interval(3, 7)]
        tm.assert_series_equal(expected, result)

        msg = (
            "'can only get slices from an IntervalIndex if bounds are "
            "non-overlapping and all monotonic increasing or decreasing'"
        )
        with pytest.raises(KeyError, match=msg):
            indexer_sl(ser)[Interval(1, 6) : Interval(3, 8)]

        if indexer_sl is tm.loc:
            # slices with scalar raise for overlapping intervals
            # TODO KeyError is the appropriate error?
            with pytest.raises(KeyError, match=msg):
                ser.loc[1:4]

    def test_non_unique(self, indexer_sl):
        # [FUNCTION DOC]
        # For adjacent intervals sharing an endpoint, an exact interval key returns the single scalar and a list key returns a one-row series.
        idx = IntervalIndex.from_tuples([(1, 3), (3, 7)])
        ser = Series(range(len(idx)), index=idx)

        result = indexer_sl(ser)[Interval(1, 3)]
        assert result == 0

        result = indexer_sl(ser)[[Interval(1, 3)]]
        expected = ser.iloc[0:1]
        tm.assert_series_equal(expected, result)

    def test_non_unique_moar(self, indexer_sl):
        # [FUNCTION DOC]
        # With a duplicated interval: an exact key returns both rows; an open-ended interval slice returns everything; a list key returns the two matching rows.
        idx = IntervalIndex.from_tuples([(1, 3), (1, 3), (3, 7)])
        ser = Series(range(len(idx)), index=idx)

        expected = ser.iloc[[0, 1]]
        result = indexer_sl(ser)[Interval(1, 3)]
        tm.assert_series_equal(expected, result)

        expected = ser
        result = indexer_sl(ser)[Interval(1, 3) :]
        tm.assert_series_equal(expected, result)

        expected = ser.iloc[[0, 1]]
        result = indexer_sl(ser)[[Interval(1, 3)]]
        tm.assert_series_equal(expected, result)

    def test_loc_getitem_missing_key_error_message(
        self, frame_or_series, series_with_interval_index
    ):
        # [FUNCTION DOC]
        # Uses frame_or_series. Label-based selection with a list that includes a value outside every interval raises a key error mentioning that value.
        # GH#27365
        ser = series_with_interval_index.copy()
        obj = frame_or_series(ser)
        with pytest.raises(KeyError, match=r"\[6\]"):
            obj.loc[[4, 5, 6]]


@pytest.mark.xfail(WASM, reason="GH 23440")
@pytest.mark.parametrize(
    "intervals",
    [
        ([Interval(-np.inf, 0.0), Interval(0.0, 1.0)]),
        ([Interval(-np.inf, -2.0), Interval(-2.0, -1.0)]),
        ([Interval(-1.0, 0.0), Interval(0.0, np.inf)]),
        ([Interval(1.0, 2.0), Interval(2.0, np.inf)]),
    ],
)
def test_repeating_interval_index_with_infs(intervals):
    # [FUNCTION DOC]
    # Parametrized over four pairs of adjacent intervals with infinite ends; expected to fail on WebAssembly. An index made by repeating the pair 51 times, asked for the indexer of the second interval, returns the odd positions one to 101 (all 51 occurrences) as platform integers.
    # GH 46658

    interval_index = Index(intervals * 51)

    expected = np.arange(1, 102, 2, dtype=np.intp)
    result = interval_index.get_indexer_for([intervals[1]])

    tm.assert_equal(result, expected)
