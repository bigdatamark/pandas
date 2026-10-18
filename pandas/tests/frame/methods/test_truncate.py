# [MODULE DOC]
# Tests for DataFrame.truncate and Series.truncate (drop rows before/after labels on a sorted axis).

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    Series,
    date_range,
)
import pandas._testing as tm


class TestDataFrameTruncate:
    # [CLASS DOC]
    # Test container for truncate.
    def test_truncate(self, datetime_frame, frame_or_series):
        # [FUNCTION DOC]
        # Takes every third row of the datetime fixture frame (as frame or series) and checks: no bounds returns the
        # object unchanged; both bounds present or missing (between existing labels) give the same slice of the middle rows; only
        # before or only after bounds each give the matching head/tail slice; bounds just outside the data give an empty
        # result; and a before bound later than the after bound raises a value error saying the truncate bound must be after the other.
        ts = datetime_frame[::3]
        ts = tm.get_obj(ts, frame_or_series)

        start, end = datetime_frame.index[3], datetime_frame.index[6]

        start_missing = datetime_frame.index[2]
        end_missing = datetime_frame.index[7]

        # neither specified
        truncated = ts.truncate()
        tm.assert_equal(truncated, ts)

        # both specified
        expected = ts[1:3]

        truncated = ts.truncate(start, end)
        tm.assert_equal(truncated, expected)

        truncated = ts.truncate(start_missing, end_missing)
        tm.assert_equal(truncated, expected)

        # start specified
        expected = ts[1:]

        truncated = ts.truncate(before=start)
        tm.assert_equal(truncated, expected)

        truncated = ts.truncate(before=start_missing)
        tm.assert_equal(truncated, expected)

        # end specified
        expected = ts[:3]

        truncated = ts.truncate(after=end)
        tm.assert_equal(truncated, expected)

        truncated = ts.truncate(after=end_missing)
        tm.assert_equal(truncated, expected)

        # corner case, empty series/frame returned
        truncated = ts.truncate(after=ts.index[0] - ts.index.freq)
        assert len(truncated) == 0

        truncated = ts.truncate(before=ts.index[-1] + ts.index.freq)
        assert len(truncated) == 0

        msg = "Truncate: 2000-01-06 00:00:00 must be after 2000-01-11 00:00:00"
        with pytest.raises(ValueError, match=msg):
            ts.truncate(
                before=ts.index[-1] - ts.index.freq, after=ts.index[0] + ts.index.freq
            )

    def test_truncate_nonsortedindex(self, frame_or_series):
        # [FUNCTION DOC]
        # Regression test (GH 17935). Truncating an object with an unsorted integer index must raise a value error saying
        # truncate requires a sorted index.
        # GH#17935

        obj = DataFrame({"A": ["a", "b", "c", "d", "e"]}, index=[5, 3, 2, 9, 0])
        obj = tm.get_obj(obj, frame_or_series)

        msg = "truncate requires a sorted index"
        with pytest.raises(ValueError, match=msg):
            obj.truncate(before=3, after=9)

    def test_sort_values_nonsortedindex(self):
        # [FUNCTION DOC]
        # Frame with a weekly datetime index and random columns is sorted by one column in descending order, which
        # leaves the index unsorted. Truncating it with string bounds must raise a value error that truncate requires a sorted index.
        # KNOWN ISSUE: the test name mentions sort_values but it verifies truncate on a non-sorted index; it is effectively another non-sorted-index truncate test.
        rng = date_range("2011-01-01", "2012-01-01", freq="W")
        ts = DataFrame(
            {
                "A": np.random.default_rng(2).standard_normal(len(rng)),
                "B": np.random.default_rng(2).standard_normal(len(rng)),
            },
            index=rng,
        )

        decreasing = ts.sort_values("A", ascending=False)

        msg = "truncate requires a sorted index"
        with pytest.raises(ValueError, match=msg):
            decreasing.truncate(before="2011-11", after="2011-12")

    def test_truncate_nonsortedindex_axis1(self):
        # [FUNCTION DOC]
        # Regression test (GH 17935). Frame with unsorted integer columns truncated along axis 1 must raise a value error that truncate requires a sorted index.
        # GH#17935

        df = DataFrame(
            {
                3: np.random.default_rng(2).standard_normal(5),
                20: np.random.default_rng(2).standard_normal(5),
                2: np.random.default_rng(2).standard_normal(5),
                0: np.random.default_rng(2).standard_normal(5),
            },
            columns=[3, 20, 2, 0],
        )
        msg = "truncate requires a sorted index"
        with pytest.raises(ValueError, match=msg):
            df.truncate(before=2, after=20, axis=1)

    @pytest.mark.parametrize(
        "before, after, indices",
        [(1, 2, [2, 1]), (None, 2, [2, 1, 0]), (1, None, [3, 2, 1])],
    )
    @pytest.mark.parametrize("dtyp", [*tm.ALL_REAL_NUMPY_DTYPES, "datetime64[ns]"])
    def test_truncate_decreasing_index(
        self, before, after, indices, dtyp, frame_or_series
    ):
        # [FUNCTION DOC]
        # Regression test (GH 33756). Parametrized over (before, after, expected labels) cases and numeric or datetime dtypes
        # for a strictly decreasing index of 3,2,1,0. For datetime dtype the bounds and expected labels are converted to timestamps.
        # Truncation must work on a monotonic decreasing index and equal selecting the expected labels from the object.
        # https://github.com/pandas-dev/pandas/issues/33756
        idx = Index([3, 2, 1, 0], dtype=dtyp)
        if isinstance(idx, DatetimeIndex):
            before = pd.Timestamp(before) if before is not None else None
            after = pd.Timestamp(after) if after is not None else None
            indices = [pd.Timestamp(i) for i in indices]
        values = frame_or_series(range(len(idx)), index=idx)
        result = values.truncate(before=before, after=after)
        expected = values.loc[indices]
        tm.assert_equal(result, expected)

    def test_truncate_multiindex(self, frame_or_series):
        # [FUNCTION DOC]
        # Regression test (GH 34564). Truncating a two-level index (1 to 4 by A, B) with before 2 and after 3 keeps the
        # rows whose first level is 2 or 3, equal to the matching frame with the index set from those levels.
        # GH 34564
        mi = pd.MultiIndex.from_product([[1, 2, 3, 4], ["A", "B"]], names=["L1", "L2"])
        s1 = DataFrame(range(mi.shape[0]), index=mi, columns=["col"])
        s1 = tm.get_obj(s1, frame_or_series)

        result = s1.truncate(before=2, after=3)

        df = DataFrame.from_dict(
            {"L1": [2, 2, 3, 3], "L2": ["A", "B", "A", "B"], "col": [2, 3, 4, 5]}
        )
        expected = df.set_index(["L1", "L2"])
        expected = tm.get_obj(expected, frame_or_series)

        tm.assert_equal(result, expected)

    def test_truncate_index_only_one_unique_value(self, frame_or_series):
        # [FUNCTION DOC]
        # Regression test (GH 42365). A date index with one repeated date (5 rows); truncating with a wider
        # date range returns the whole object unchanged.
        # GH 42365
        obj = Series(0, index=date_range("2021-06-30", "2021-06-30")).repeat(5)
        if frame_or_series is DataFrame:
            obj = obj.to_frame(name="a")

        truncated = obj.truncate("2021-06-28", "2021-07-01")

        tm.assert_equal(truncated, obj)
