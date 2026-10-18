# [MODULE DOC]
# Tests for indexing Series and DataFrame objects labelled by an interval index: getitem and loc with scalars, intervals, lists, slices (including large series that exceed an engine size cutoff), missing keys, NaN-valued intervals, setting with slices, and interval levels inside a multi-level index. Uses the indexer_sl fixture so each test runs with both label-based and plain getitem access where applicable.

import numpy as np
import pytest

from pandas._libs import index as libindex
from pandas.compat import WASM

import pandas as pd
from pandas import (
    DataFrame,
    IntervalIndex,
    Series,
)
import pandas._testing as tm


class TestIntervalIndex:
    # [CLASS DOC]
    # Test class for interval-labelled indexing; provides a series fixture.
    @pytest.fixture
    def series_with_interval_index(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a series of the integers zero to four labelled by an interval index with breaks zero through five (right-closed intervals).
        return Series(np.arange(5), IntervalIndex.from_breaks(np.arange(6)))

    def test_getitem_with_scalar(self, series_with_interval_index, indexer_sl):
        # [FUNCTION DOC]
        # Uses the fixture and indexer_sl. Slicing by scalars (integer or float stop, start and stop, and for label-based access a start below the first interval) selects the first three rows; lists of scalars in the intervals one to three or four select the middle three rows; a boolean mask selects rows two to four.
        ser = series_with_interval_index.copy()

        expected = ser.iloc[:3]
        tm.assert_series_equal(expected, indexer_sl(ser)[:3])
        tm.assert_series_equal(expected, indexer_sl(ser)[:2.5])
        tm.assert_series_equal(expected, indexer_sl(ser)[0.1:2.5])
        if indexer_sl is tm.loc:
            tm.assert_series_equal(expected, ser.loc[-1:3])

        expected = ser.iloc[1:4]
        tm.assert_series_equal(expected, indexer_sl(ser)[[1.5, 2.5, 3.5]])
        tm.assert_series_equal(expected, indexer_sl(ser)[[2, 3, 4]])
        tm.assert_series_equal(expected, indexer_sl(ser)[[1.5, 3, 4]])

        expected = ser.iloc[2:5]
        tm.assert_series_equal(expected, indexer_sl(ser)[ser >= 2])

    @pytest.mark.parametrize("direction", ["increasing", "decreasing"])
    def test_getitem_nonoverlapping_monotonic(self, direction, closed, indexer_sl):
        # [FUNCTION DOC]
        # Parametrized over increasing and decreasing orderings, with the closed fixture (which side(s) are closed) and indexer_sl. For each interval: lookup by the left endpoint works only when the left side is closed (else a key error); by the right endpoint only when the right side is closed; by the midpoint always works.
        tpls = [(0, 1), (2, 3), (4, 5)]
        if direction == "decreasing":
            tpls = tpls[::-1]

        idx = IntervalIndex.from_tuples(tpls, closed=closed)
        ser = Series(list("abc"), idx)

        for key, expected in zip(idx.left, ser):
            if idx.closed_left:
                assert indexer_sl(ser)[key] == expected
            else:
                with pytest.raises(KeyError, match=str(key)):
                    indexer_sl(ser)[key]

        for key, expected in zip(idx.right, ser):
            if idx.closed_right:
                assert indexer_sl(ser)[key] == expected
            else:
                with pytest.raises(KeyError, match=str(key)):
                    indexer_sl(ser)[key]

        for key, expected in zip(idx.mid, ser):
            assert indexer_sl(ser)[key] == expected

    def test_getitem_non_matching(self, series_with_interval_index, indexer_sl):
        # [FUNCTION DOC]
        # A list containing one value outside every interval raises a key error saying that value is not in the index (even if other values match).
        ser = series_with_interval_index.copy()

        # this is a departure from our current
        # indexing scheme, but simpler
        with pytest.raises(KeyError, match=r"\[-1\] not in index"):
            indexer_sl(ser)[[-1, 3, 4, 5]]

        with pytest.raises(KeyError, match=r"\[-1\] not in index"):
            indexer_sl(ser)[[-1, 3]]

    def test_loc_getitem_large_series(self, monkeypatch):
        # [FUNCTION DOC]
        # Patches the module-level engine size cutoff down to 20 so the large-index path is taken. For a series of that length, label slices with stop 8, start 0 stop 8, and with step 1 give equal results.
        size_cutoff = 20
        with monkeypatch.context():
            monkeypatch.setattr(libindex, "_SIZE_CUTOFF", size_cutoff)
            ser = Series(
                np.arange(size_cutoff),
                index=IntervalIndex.from_breaks(np.arange(size_cutoff + 1)),
            )

            result1 = ser.loc[:8]
            result2 = ser.loc[0:8]
            result3 = ser.loc[0:8:1]
        tm.assert_series_equal(result1, result2)
        tm.assert_series_equal(result1, result3)

    def test_loc_getitem_frame(self):
        # [FUNCTION DOC]
        # A frame indexed by the binned values of a column (categorical with interval categories): a scalar lookup returns the rows of the matching bin; a missing scalar raises key error; a one-element list works; a list of two scalars returns duplicated rows by position; an all-missing list raises a key error "None of [...] are in the [index]"; a partially missing list raises a key error naming the missing element.
        # CategoricalIndex with IntervalIndex categories
        df = DataFrame({"A": range(10)})
        ser = pd.cut(df.A, 5)
        df["B"] = ser
        df = df.set_index("B")

        result = df.loc[4]
        expected = df.iloc[4:6]
        tm.assert_frame_equal(result, expected)

        with pytest.raises(KeyError, match="10"):
            df.loc[10]

        # single list-like
        result = df.loc[[4]]
        expected = df.iloc[4:6]
        tm.assert_frame_equal(result, expected)

        # non-unique
        result = df.loc[[4, 5]]
        expected = df.take([4, 5, 4, 5])
        tm.assert_frame_equal(result, expected)

        msg = (
            r"None of \[Index\(\[10\], dtype='object', name='B'\)\] "
            r"are in the \[index\]"
        )
        with pytest.raises(KeyError, match=msg):
            df.loc[[10]]

        # partial missing
        with pytest.raises(KeyError, match=r"\[10\] not in index"):
            df.loc[[10, 4]]

    def test_getitem_interval_with_nans(self, frame_or_series, indexer_sl):
        # [FUNCTION DOC]
        # Uses frame_or_series and indexer_sl. For an interval index of two missing entries, selecting with the first-entry sub-index as key returns the whole object (frames transposed for item-setting access).
        # GH#41831

        index = IntervalIndex([np.nan, np.nan])
        key = index[:-1]

        obj = frame_or_series(range(2), index=index)
        if frame_or_series is DataFrame and indexer_sl is tm.setitem:
            obj = obj.T

        result = indexer_sl(obj)[key]
        expected = obj

        tm.assert_equal(result, expected)

    def test_setitem_interval_with_slice(self):
        # [FUNCTION DOC]
        # A label slice that does not overlap any interval is a no-op rather than an error; a slice matching intervals sets those positions; the same for a series on a stepped sub-index of the intervals. Tests both with the sets expected by position.
        # GH#54722
        ii = IntervalIndex.from_breaks(range(4, 15))
        ser = Series(range(10), index=ii)

        orig = ser.copy()

        # This should be a no-op (used to raise)
        ser.loc[1:3] = 20
        tm.assert_series_equal(ser, orig)

        ser.loc[6:8] = 19
        orig.iloc[1:4] = 19
        tm.assert_series_equal(ser, orig)

        ser2 = Series(range(5), index=ii[::2])
        orig2 = ser2.copy()

        # this used to raise
        ser2.loc[6:8] = 22  # <- raises on main, sets on branch
        orig2.iloc[1] = 22
        tm.assert_series_equal(ser2, orig2)

        ser2.loc[5:7] = 21
        orig2.iloc[:2] = 21
        tm.assert_series_equal(ser2, orig2)


class TestIntervalIndexInsideMultiIndex:
    # [CLASS DOC]
    # Tests with an interval index as one level of a multi-level index.
    def test_mi_intervalindex_slicing_with_scalar(self):
        # [FUNCTION DOC]
        # Builds a frame with a three-level index (item, id, interval) and a query frame whose float values in the third level fall within the intervals. Selecting by the query's multi-level index returns the matching values with the interval level preserved as interval index (the matching intervals taken at expected positions).
        # GH#27456
        ii = IntervalIndex.from_arrays(
            [0, 1, 10, 11, 0, 1, 10, 11], [1, 2, 11, 12, 1, 2, 11, 12], name="MP"
        )
        idx = pd.MultiIndex.from_arrays(
            [
                pd.Index(["FC", "FC", "FC", "FC", "OWNER", "OWNER", "OWNER", "OWNER"]),
                pd.Index(
                    ["RID1", "RID1", "RID2", "RID2", "RID1", "RID1", "RID2", "RID2"]
                ),
                ii,
            ]
        )

        idx.names = ["Item", "RID", "MP"]
        df = DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 8]})
        df.index = idx

        query_df = DataFrame(
            {
                "Item": ["FC", "OWNER", "FC", "OWNER", "OWNER"],
                "RID": ["RID1", "RID1", "RID1", "RID2", "RID2"],
                "MP": [0.2, 1.5, 1.6, 11.1, 10.9],
            }
        )

        query_df = query_df.sort_index()

        idx = pd.MultiIndex.from_arrays([query_df.Item, query_df.RID, query_df.MP])
        query_df.index = idx
        result = df.value.loc[query_df.index]

        # the IntervalIndex level is indexed with floats, which map to
        #  the intervals containing them.  Matching the behavior we would get
        #  with _only_ an IntervalIndex, we get an IntervalIndex level back.
        sliced_level = ii.take([0, 1, 1, 3, 2])
        expected_index = pd.MultiIndex.from_arrays(
            [idx.get_level_values(0), idx.get_level_values(1), sliced_level]
        )
        expected = Series([1, 6, 2, 8, 7], index=expected_index, name="value")
        tm.assert_series_equal(result, expected)

    @pytest.mark.xfail(WASM, reason="GH 23440")
    @pytest.mark.parametrize("base", [101, 1010])
    def test_reindex_behavior_with_interval_index(self, base):
        # [FUNCTION DOC]
        # Parametrized over base sizes 101 and 1010; expected to fail on WebAssembly. Reindexing a series with interval index to NaN and one gives NaN for NaN and the value for the interval containing one.
        # GH 51826

        ser = Series(
            range(base),
            index=IntervalIndex.from_arrays(range(base), range(1, base + 1)),
        )
        expected_result = Series([np.nan, 0], index=[np.nan, 1.0], dtype=float)
        result = ser.reindex(index=[np.nan, 1.0])
        tm.assert_series_equal(result, expected_result)

    def test_multiindex_with_interval_index(self):
        # [FUNCTION DOC]
        # A frame with a string level and an interval level: selecting with a tuple of a string and a number inside an interval returns the row as a series named with the tuple using the matching interval.
        # for GH#25298
        intIndex = IntervalIndex.from_arrays([1, 5, 8, 13, 16], [4, 9, 12, 17, 20])
        multiIndex = pd.MultiIndex.from_arrays([["a", "a", "b", "b", "c"], intIndex])
        data = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]
        df = DataFrame(data, index=multiIndex)
        result = df.loc[("b", 16)]
        expected = Series([7, 8], name=("b", pd.Interval(13, 17, closed="right")))
        tm.assert_series_equal(result, expected)
