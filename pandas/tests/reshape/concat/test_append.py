# [MODULE DOC]
# Tests for appending-style behavior of the concatenate function on frames and
# series (plus the internal single-row append helper on frames). Covers empty
# frames, overlapping index verification, new columns, record arrays, column
# sorting, differing column index types, dtype coercion/upcasting, timezone and
# timedelta missing-value handling. Uses shared fixtures: sort and float_frame.
import datetime as dt
from itertools import combinations

import dateutil
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    Timestamp,
    concat,
    isna,
)
import pandas._testing as tm


class TestAppend:
    # [CLASS DOC]
    # Groups the append-oriented concatenation tests. Stateless. Also holds a class
    # attribute listing index objects of many kinds (range, integer, float, string,
    # categorical, interval, datetime, multi-level) that can be used as column
    # labels in the parametrized append tests.
    def test_append(self, sort, float_frame):
        # [FUNCTION DOC]
        # Verifies that concatenating row slices of a float frame reproduces it, that
        # dropping a column from one part still yields that column in either order, and
        # that mixed-type (string column added) frames concatenate correctly; frames
        # differing only in the extra string column agree on the shared columns.
        # Parametrized by the sort fixture.
        mixed_frame = float_frame.copy()
        mixed_frame["foo"] = "bar"

        begin_index = float_frame.index[:5]
        end_index = float_frame.index[5:]

        begin_frame = float_frame.reindex(begin_index)
        end_frame = float_frame.reindex(end_index)

        appended = concat([begin_frame, end_frame])
        tm.assert_almost_equal(appended["A"], float_frame["A"])

        del end_frame["A"]
        partial_appended = concat([begin_frame, end_frame], sort=sort)
        assert "A" in partial_appended

        partial_appended = concat([end_frame, begin_frame], sort=sort)
        assert "A" in partial_appended

        # mixed type handling
        appended = concat([mixed_frame[:5], mixed_frame[5:]])
        tm.assert_frame_equal(appended, mixed_frame)

        # what to test here
        mixed_appended = concat([mixed_frame[:5], float_frame[5:]], sort=sort)
        mixed_appended2 = concat([float_frame[:5], mixed_frame[5:]], sort=sort)

        # all equal except 'foo' column
        tm.assert_frame_equal(
            mixed_appended.reindex(columns=["A", "B", "C", "D"]),
            mixed_appended2.reindex(columns=["A", "B", "C", "D"]),
        )

    def test_append_empty(self, float_frame):
        # [FUNCTION DOC]
        # Verifies that concatenating a frame with an empty frame (either order)
        # returns an equal frame and a new object, not the original.
        empty = DataFrame()

        appended = concat([float_frame, empty])
        tm.assert_frame_equal(float_frame, appended)
        assert appended is not float_frame

        appended = concat([empty, float_frame])
        tm.assert_frame_equal(float_frame, appended)
        assert appended is not float_frame

    def test_append_overlap_raises(self, float_frame):
        # [FUNCTION DOC]
        # Verifies that concatenating a frame with itself with integrity verification
        # enabled raises a value error stating the indexes have overlapping values.
        msg = "Indexes have overlapping values"
        with pytest.raises(ValueError, match=msg):
            concat([float_frame, float_frame], verify_integrity=True)

    def test_append_new_columns(self):
        # [FUNCTION DOC]
        # Verifies the internal single-row append helper: appending a series whose index
        # adds a new label yields a new column holding the value only for the new row
        # (GH 6129).
        # see gh-6129: new columns
        df = DataFrame({"a": {"x": 1, "y": 2}, "b": {"x": 3, "y": 4}})
        row = Series([5, 6, 7], index=["a", "b", "c"], name="z")
        expected = DataFrame(
            {
                "a": {"x": 1, "y": 2, "z": 5},
                "b": {"x": 3, "y": 4, "z": 6},
                "c": {"z": 7},
            }
        )
        result = df._append_internal(row)
        tm.assert_frame_equal(result, expected)

    def test_append_length0_frame(self, sort):
        # [FUNCTION DOC]
        # Verifies that concatenating a zero-row frame having three columns with a
        # two-row frame lacking one column gives a two-row frame with all three columns
        # (all missing), for each sort setting.
        df = DataFrame(columns=["A", "B", "C"])
        df3 = DataFrame(index=[0, 1], columns=["A", "B"])
        df5 = concat([df, df3], sort=sort)

        expected = DataFrame(index=[0, 1], columns=["A", "B", "C"])
        tm.assert_frame_equal(df5, expected)

    def test_append_records(self):
        # [FUNCTION DOC]
        # Verifies that frames built from structured numpy arrays concatenate with an
        # ignored index to the same frame as built from the concatenated array.
        arr1 = np.zeros((2,), dtype=("i4,f4,S10"))
        arr1[:] = [(1, 2.0, "Hello"), (2, 3.0, "World")]

        arr2 = np.zeros((3,), dtype=("i4,f4,S10"))
        arr2[:] = [(3, 4.0, "foo"), (5, 6.0, "bar"), (7.0, 8.0, "baz")]

        df1 = DataFrame(arr1)
        df2 = DataFrame(arr2)

        result = concat([df1, df2], ignore_index=True)
        expected = DataFrame(np.concatenate((arr1, arr2)))
        tm.assert_frame_equal(result, expected)

    # rewrite sort fixture, since we also want to test default of None
    def test_append_sorts(self, sort):
        # [FUNCTION DOC]
        # Takes the sort fixture (the comment notes it is meant to also cover the
        # default). Verifies columns are sorted alphabetically in the result when sort is
        # true or unset, and keep first-appearance order when sort is false; values are
        # filled with missing where absent.
        df1 = DataFrame({"a": [1, 2], "b": [1, 2]}, columns=["b", "a"])
        df2 = DataFrame({"a": [1, 2], "c": [3, 4]}, index=[2, 3])

        result = concat([df1, df2], sort=sort)

        # for None / True
        expected = DataFrame(
            {"b": [1, 2, None, None], "a": [1, 2, 1, 2], "c": [None, None, 3, 4]},
            columns=["a", "b", "c"],
        )
        if sort is False:
            expected = expected[["b", "a", "c"]]
        tm.assert_frame_equal(result, expected)

    def test_append_different_columns(self, sort):
        # [FUNCTION DOC]
        # Verifies that concatenating two frames with only partially shared columns
        # yields missing values in the columns absent from each part (strings missing in
        # the first rows, bools missing in the later rows).
        df = DataFrame(
            {
                "bools": np.random.default_rng(2).standard_normal(10) > 0,
                "ints": np.random.default_rng(2).integers(0, 10, 10),
                "floats": np.random.default_rng(2).standard_normal(10),
                "strings": ["foo", "bar"] * 5,
            }
        )

        a = df[:5].loc[:, ["bools", "ints", "floats"]]
        b = df[5:].loc[:, ["strings", "ints", "floats"]]

        appended = concat([a, b], sort=sort)
        assert isna(appended["strings"][0:4]).all()
        assert isna(appended["bools"][5:]).all()

    def test_append_preserve_index_name(self):
        # [FUNCTION DOC]
        # Verifies that concatenating an empty frame and a frame that both use a column
        # set as the index keeps the index name (GH 980).
        # #980
        df1 = DataFrame(columns=["A", "B", "C"])
        df1 = df1.set_index(["A"])
        df2 = DataFrame(data=[[1, 4, 7], [2, 5, 8], [3, 6, 9]], columns=["A", "B", "C"])
        df2 = df2.set_index(["A"])

        result = concat([df1, df2])
        assert result.index.name == "A"

    indexes_can_append = [
        pd.RangeIndex(3),
        Index([4, 5, 6]),
        Index([4.5, 5.5, 6.5]),
        Index(list("abc")),
        pd.CategoricalIndex("A B C".split()),
        pd.CategoricalIndex("D E F".split(), ordered=True),
        pd.IntervalIndex.from_breaks([7, 8, 9, 10]),
        pd.DatetimeIndex(
            [
                dt.datetime(2013, 1, 3, 0, 0),
                dt.datetime(2013, 1, 3, 6, 10),
                dt.datetime(2013, 1, 3, 7, 12),
            ]
        ),
        pd.MultiIndex.from_arrays(["A B C".split(), "D E F".split()]),
    ]

    @pytest.mark.parametrize(
        "index", indexes_can_append, ids=lambda x: type(x).__name__
    )
    def test_append_same_columns_type(self, index):
        # [FUNCTION DOC]
        # Parametrized over the index kinds in the class list. Verifies the internal
        # append helper in two cases: a series shorter than the frame columns (result
        # has missing value for absent labels, integer dtype kept for present columns)
        # and a series longer than the frame columns (frame columns are extended).
        # Result column labels must equal the original index type (GH 18359).
        # GH18359

        # df wider than ser
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=index)
        ser_index = index[:2]
        ser = Series([7, 8], index=ser_index, name=2)
        result = df._append_internal(ser)
        expected = DataFrame(
            [[1, 2, 3.0], [4, 5, 6], [7, 8, np.nan]], index=[0, 1, 2], columns=index
        )
        # integer dtype is preserved for columns present in ser.index
        assert expected.dtypes.iloc[0].kind == "i"
        assert expected.dtypes.iloc[1].kind == "i"

        tm.assert_frame_equal(result, expected)

        # ser wider than df
        ser_index = index
        index = index[:2]
        df = DataFrame([[1, 2], [4, 5]], columns=index)
        ser = Series([7, 8, 9], index=ser_index, name=2)
        result = df._append_internal(ser)
        expected = DataFrame(
            [[1, 2, np.nan], [4, 5, np.nan], [7, 8, 9]],
            index=[0, 1, 2],
            columns=ser_index,
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "df_columns, series_index",
        combinations(indexes_can_append, r=2),
        ids=lambda x: type(x).__name__,
    )
    def test_append_different_columns_types(self, df_columns, series_index):
        # [FUNCTION DOC]
        # Parametrized over all pairs of the index kinds used as frame columns and
        # series index. Verifies the internal append helper unions the labels:
        # the result columns equal the frame labels followed by the series-only labels,
        # with missing values filled in the off-diagonal blocks (GH 18359).
        # GH18359
        # See also test 'test_append_different_columns_types_raises' below
        # for errors raised when appending

        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=df_columns)
        ser = Series([7, 8, 9], index=series_index, name=2)

        result = df._append_internal(ser)
        idx_diff = ser.index.difference(df_columns)
        combined_columns = Index(df_columns.tolist()).append(idx_diff)
        expected = DataFrame(
            [
                [1.0, 2.0, 3.0, np.nan, np.nan, np.nan],
                [4, 5, 6, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, 7, 8, 9],
            ],
            index=[0, 1, 2],
            columns=combined_columns,
        )
        tm.assert_frame_equal(result, expected)

    def test_append_dtype_coerce(self, sort):
        # [FUNCTION DOC]
        # Verifies that concatenating datetime-valued frames with different columns keeps
        # datetime dtype and fills missing end times with NaT, with column order sorted
        # or not depending on the sort fixture (GH 4993).
        # GH 4993
        # appending with datetime will incorrectly convert datetime64

        df1 = DataFrame(
            index=[1, 2],
            data=[dt.datetime(2013, 1, 1, 0, 0), dt.datetime(2013, 1, 2, 0, 0)],
            columns=["start_time"],
        )
        df2 = DataFrame(
            index=[4, 5],
            data=[
                [dt.datetime(2013, 1, 3, 0, 0), dt.datetime(2013, 1, 3, 6, 10)],
                [dt.datetime(2013, 1, 4, 0, 0), dt.datetime(2013, 1, 4, 7, 10)],
            ],
            columns=["start_time", "end_time"],
        )

        expected = concat(
            [
                Series(
                    [
                        pd.NaT,
                        pd.NaT,
                        dt.datetime(2013, 1, 3, 6, 10),
                        dt.datetime(2013, 1, 4, 7, 10),
                    ],
                    name="end_time",
                ),
                Series(
                    [
                        dt.datetime(2013, 1, 1, 0, 0),
                        dt.datetime(2013, 1, 2, 0, 0),
                        dt.datetime(2013, 1, 3, 0, 0),
                        dt.datetime(2013, 1, 4, 0, 0),
                    ],
                    name="start_time",
                ),
            ],
            axis=1,
            sort=sort,
        )
        result = concat([df1, df2], ignore_index=True, sort=sort)
        if sort:
            expected = expected[["end_time", "start_time"]]
        else:
            expected = expected[["start_time", "end_time"]]

        tm.assert_frame_equal(result, expected)

    def test_append_missing_column_proper_upcast(self, sort):
        # [FUNCTION DOC]
        # Verifies that concatenating frames with disjoint columns upcasts the integer
        # column to float and the boolean column to object.
        df1 = DataFrame({"A": np.array([1, 2, 3, 4], dtype="i8")})
        df2 = DataFrame({"B": np.array([True, False, True, False], dtype=bool)})

        appended = concat([df1, df2], sort=sort)
        assert appended["A"].dtype == "f8"
        assert appended["B"].dtype == "O"

    def test_append_empty_frame_to_series_with_dateutil_tz(self):
        # [FUNCTION DOC]
        # Verifies that appending a series containing a tz-aware (dateutil UTC) timestamp
        # to an empty frame with ignored index yields the expected row, with the empty
        # frame's columns cast to object, and that appending a second time adds another
        # row (GH 23682).
        # GH 23682
        date = Timestamp("2018-10-24 07:30:00", tz=dateutil.tz.tzutc())
        ser = Series({"a": 1.0, "b": 2.0, "date": date})
        df = DataFrame(columns=["c", "d"])
        result_a = df._append_internal(ser, ignore_index=True)
        expected = DataFrame(
            [[np.nan, np.nan, 1.0, 2.0, date]], columns=["c", "d", "a", "b", "date"]
        )
        # These columns get cast to object after append
        expected["c"] = expected["c"].astype(object)
        expected["d"] = expected["d"].astype(object)
        tm.assert_frame_equal(result_a, expected)

        expected = DataFrame(
            [[np.nan, np.nan, 1.0, 2.0, date]] * 2, columns=["c", "d", "a", "b", "date"]
        )
        expected["c"] = expected["c"].astype(object)
        expected["d"] = expected["d"].astype(object)
        result_b = result_a._append_internal(ser, ignore_index=True)
        tm.assert_frame_equal(result_b, expected)

    def test_append_empty_tz_frame_with_datetime64ns(self):
        # [FUNCTION DOC]
        # Verifies that concatenating an empty tz-aware datetime frame with a frame of
        # missing values (naive or a different timezone) yields an object column with
        # NaT.
        # KNOWN ISSUE: the frame is constructed twice in a row with identical
        # arguments, so the first construction is dead code and the comment about a typed
        # value to append is misplaced.
        # https://github.com/pandas-dev/pandas/issues/35460
        df = DataFrame(columns=["a"]).astype("datetime64[ns, UTC]")

        # also test with typed value to append
        df = DataFrame(columns=["a"]).astype("datetime64[ns, UTC]")
        other = Series({"a": pd.NaT}, dtype="datetime64[ns]").to_frame().T
        result = concat([df, other], ignore_index=True)
        expected = DataFrame({"a": [pd.NaT]}, dtype=object)
        tm.assert_frame_equal(result, expected)

        # mismatched tz
        other = Series({"a": pd.NaT}, dtype="datetime64[ns, US/Pacific]").to_frame().T
        result = concat([df, other], ignore_index=True)
        expected = DataFrame({"a": [pd.NaT]}).astype(object)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "dtype_str", ["datetime64[ns, UTC]", "datetime64[ns]", "Int64", "int64"]
    )
    @pytest.mark.parametrize("val", [1, "NaT"])
    def test_append_empty_frame_with_timedelta64ns_nat(self, dtype_str, val):
        # [FUNCTION DOC]
        # Parametrized over dtype strings and a value (an integer or the NaT string).
        # Verifies that concatenating an empty frame of the given dtype with a timedelta
        # frame gives the timedelta frame cast to object.
        # https://github.com/pandas-dev/pandas/issues/35460
        df = DataFrame(columns=["a"]).astype(dtype_str)

        other = DataFrame({"a": [np.timedelta64(val, "ns")]})
        result = concat([df, other])

        expected = other.astype(object)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "dtype_str", ["datetime64[ns, UTC]", "datetime64[ns]", "Int64", "int64"]
    )
    @pytest.mark.parametrize("val", [1, "NaT"])
    def test_append_frame_with_timedelta64ns_nat(self, dtype_str, val):
        # [FUNCTION DOC]
        # Same parametrization as above but the first frame is non-empty. Verifies the
        # result with ignored index is an object column holding the first frame value
        # followed by the timedelta value.
        # https://github.com/pandas-dev/pandas/issues/35460
        df = DataFrame({"a": pd.array([1], dtype=dtype_str)})

        other = DataFrame({"a": [np.timedelta64(val, "ns")]})
        result = concat([df, other], ignore_index=True)

        expected = DataFrame({"a": [df.iloc[0, 0], other.iloc[0, 0]]}, dtype=object)
        tm.assert_frame_equal(result, expected)
