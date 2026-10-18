"""test label based indexing with loc"""

# [MODULE DOC]
# Tests for label-based indexing with the label indexer on series and frames: getting, setting and enlarging, with many index types (integer, float, string, datetime, period, categorical, interval, multi-level, sparse, timezone-aware), boolean and callable keys, slices, duplicated labels, missing keys, dtype preservation, and error messages.
# Main classes: general tests of the label indexer, tests independent of the base index test class, tests for slicing and dtype handling, and a group of datetime-specific behaviors. Uses the shared fixtures for frame-or-series, the infer-string option and others, and the smoke-test helper from the common indexing test module.

from collections import namedtuple
import contextlib
from datetime import (
    date,
    datetime,
    time,
    timedelta,
)
import re

from dateutil.tz import gettz
import numpy as np
import pytest

from pandas._libs import index as libindex
from pandas.errors import IndexingError
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    Categorical,
    CategoricalDtype,
    CategoricalIndex,
    DataFrame,
    DatetimeIndex,
    Index,
    IndexSlice,
    MultiIndex,
    Period,
    PeriodIndex,
    Series,
    SparseDtype,
    Timedelta,
    Timestamp,
    date_range,
    timedelta_range,
    to_datetime,
    to_timedelta,
)
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.core.indexing import _one_ellipsis_message
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises


@pytest.mark.parametrize(
    "series, new_series, expected_ser",
    [
        [[np.nan, np.nan, "b"], ["a", np.nan, np.nan], [False, True, True]],
        [[np.nan, "b"], ["a", np.nan], [False, True]],
    ],
)
def test_not_change_nan_loc(series, new_series, expected_ser):
    # [FUNCTION DOC]
    # Verifies that label-assigning a whole column with values replaces the column and the missing-value mask follows the new values (issue 28403).
    # Parametrized over (original column, new column, expected missing mask): the missing mask of the result equals the expected, and the not-missing mask is its inverse.
    # GH 28403
    df = DataFrame({"A": series})
    df.loc[:, "A"] = new_series
    expected = DataFrame({"A": expected_ser})
    tm.assert_frame_equal(df.isna(), expected)
    tm.assert_frame_equal(df.notna(), ~expected)


def test_loc_dtype():
    # [FUNCTION DOC]
    # Verifies that selecting a row and a list of numeric columns from a frame whose first column is text returns a float series named by the row label with the selected column labels (issue 60600).
    # GH 60600
    df = DataFrame([["a", 1.0, 2.0], ["b", 3.0, 4.0]])
    result = df.loc[0, [1, 2]]
    expected = Series([1.0, 2.0], index=[1, 2], dtype=float, name=0)
    tm.assert_series_equal(result, expected)


class TestLoc:
    # [CLASS DOC]
    # General tests of the label indexer: smoke tests that expected errors are raised for missing or invalid keys on various indexes, and several assorted regressions.
    def test_none_values_on_string_columns(self, using_infer_string):
        # [FUNCTION DOC]
        # Verifies that a missing value in an object column is returned as none, and in a text-dtype column as none or as nan depending on the infer-string option (issue 32218).
        # Issue #32218
        df = DataFrame(["1", "2", None], columns=["a"], dtype=object)
        assert df.loc[2, "a"] is None

        df = DataFrame(["1", "2", None], columns=["a"], dtype="str")
        if using_infer_string:
            assert np.isnan(df.loc[2, "a"])
        else:
            assert df.loc[2, "a"] is None

    def test_loc_getitem_int(self, frame_or_series):
        # [FUNCTION DOC]
        # Verifies that an integer key on a string-labelled object raises a key error (checked via the smoke-test helper).
        # int label
        obj = frame_or_series(range(3), index=Index(list("abc"), dtype=object))
        check_indexing_smoketest_or_raises(obj, "loc", 2, fails=KeyError)

    def test_loc_getitem_label(self, frame_or_series):
        # [FUNCTION DOC]
        # Verifies that a missing string key on an empty object raises a key error.
        # label
        obj = frame_or_series()
        check_indexing_smoketest_or_raises(obj, "loc", "c", fails=KeyError)

    @pytest.mark.parametrize("key", ["f", 20])
    @pytest.mark.parametrize(
        "index",
        [
            Index(list("abcd"), dtype=object),
            Index([2, 4, "null", 8], dtype=object),
            date_range("20130101", periods=4),
            Index(range(0, 8, 2), dtype=np.float64),
            Index([]),
        ],
    )
    def test_loc_getitem_label_out_of_range(self, key, index, frame_or_series):
        # [FUNCTION DOC]
        # Verifies that an out-of-range label (a string or integer not present) raises a key error for strings, mixed, datetime, float and empty indexes. Parametrized over the key and index.
        obj = frame_or_series(range(len(index)), index=index)
        # out of range label
        check_indexing_smoketest_or_raises(obj, "loc", key, fails=KeyError)

    @pytest.mark.parametrize("key", [[0, 1, 2], [1, 3.0, "A"]])
    @pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float64])
    def test_loc_getitem_label_list(self, key, dtype, frame_or_series):
        # [FUNCTION DOC]
        # Verifies that lists of labels not all present (for integer, unsigned and float indexes) raise a key error. Parametrized over two key lists and three index dtypes.
        obj = frame_or_series(range(3), index=Index([0, 1, 2], dtype=dtype))
        # list of labels
        check_indexing_smoketest_or_raises(obj, "loc", key, fails=KeyError)

    @pytest.mark.parametrize(
        "index",
        [
            None,
            Index([0, 1, 2], dtype=np.int64),
            Index([0, 1, 2], dtype=np.uint64),
            Index([0, 1, 2], dtype=np.float64),
            MultiIndex.from_arrays([range(3), range(3)]),
        ],
    )
    @pytest.mark.parametrize(
        "key", [[0, 1, 2], [0, 2, 10], [3, 6, 7], [(1, 3), (1, 4), (2, 5)]]
    )
    def test_loc_getitem_label_list_with_missing(self, key, index, frame_or_series):
        # [FUNCTION DOC]
        # Verifies key errors for lists of labels with missing entries, over none (empty object), integer, unsigned, float and two-level indexes and several key lists including tuples.
        if index is None:
            obj = frame_or_series()
        else:
            obj = frame_or_series(range(len(index)), index=index)
        check_indexing_smoketest_or_raises(obj, "loc", key, fails=KeyError)

    @pytest.mark.parametrize("dtype", [np.int64, np.uint64])
    def test_loc_getitem_label_list_fails(self, dtype, frame_or_series):
        # [FUNCTION DOC]
        # Verifies a list of labels entirely absent from an integer or unsigned index raises a key error on the column axis (axis one). Parametrized over index dtype.
        # fails
        obj = frame_or_series(range(3), Index([0, 1, 2], dtype=dtype))
        check_indexing_smoketest_or_raises(
            obj, "loc", [20, 30, 40], axes=1, fails=KeyError
        )

    def test_loc_getitem_bool(self, frame_or_series):
        # [FUNCTION DOC]
        # Verifies that a boolean list key on an empty object raises an index error.
        obj = frame_or_series()
        # boolean indexers
        b = [True, False, True, False]

        check_indexing_smoketest_or_raises(obj, "loc", b, fails=IndexError)

    @pytest.mark.parametrize(
        "slc, indexes, axes, fails",
        [
            [
                slice(1, 3),
                [
                    Index(list("abcd"), dtype=object),
                    Index([2, 4, "null", 8], dtype=object),
                    None,
                    date_range("20130101", periods=4),
                    Index(range(0, 12, 3), dtype=np.float64),
                ],
                None,
                TypeError,
            ],
            [
                slice("20130102", "20130104"),
                [date_range("20130101", periods=4)],
                1,
                TypeError,
            ],
            [slice(2, 8), [Index([2, 4, "null", 8], dtype=object)], 0, TypeError],
            [slice(2, 8), [Index([2, 4, "null", 8], dtype=object)], 1, KeyError],
            [slice(2, 4, 2), [Index([2, 4, "null", 8], dtype=object)], 0, TypeError],
        ],
    )
    def test_loc_getitem_label_slice(self, slc, indexes, axes, fails, frame_or_series):
        # [FUNCTION DOC]
        # Verifies that label slices of the wrong kind for an index raise the expected error (issue 14316).
        # Parametrized over (slice, indexes, axis, expected error): integer slices on string, mixed, empty, datetime and float indexes, string-date slices on the column axis of a datetime index, and slices on a mixed object index. Each combination is exercised through the smoke-test helper.
        # label slices (with ints)

        # real label slices

        # GH 14316
        for index in indexes:
            if index is None:
                obj = frame_or_series()
            else:
                obj = frame_or_series(range(len(index)), index=index)
            check_indexing_smoketest_or_raises(
                obj,
                "loc",
                slc,
                axes=axes,
                fails=fails,
            )

    def test_setitem_from_duplicate_axis(self):
        # [FUNCTION DOC]
        # Verifies setting a column for the duplicated row label with an array of two values assigns them in order to the duplicated rows (issue 34034).
        # GH#34034
        df = DataFrame(
            [[20, "a"], [200, "a"], [200, "a"]],
            columns=["col1", "col2"],
            index=[10, 1, 1],
        )
        df.loc[1, "col1"] = np.arange(2)
        expected = DataFrame(
            [[20, "a"], [0, "a"], [1, "a"]], columns=["col1", "col2"], index=[10, 1, 1]
        )
        tm.assert_frame_equal(df, expected)

    def test_column_types_consistent(self):
        # [FUNCTION DOC]
        # Verifies that setting a text column and a datetime column for rows selected by a missing-value mask from the values of another frame keeps both column types consistent (issue 26779): the missing text and datetime are replaced by the new values.
        # GH 26779
        df = DataFrame(
            data={
                "channel": [1, 2, 3],
                "A": ["String 1", np.nan, "String 2"],
                "B": [
                    Timestamp("2019-06-11 11:00:00"),
                    pd.NaT,
                    Timestamp("2019-06-11 12:00:00"),
                ],
            }
        )
        df2 = DataFrame(
            data={"A": ["String 3"], "B": [Timestamp("2019-06-11 12:00:00")]}
        )
        # Change Columns A and B to df2.values wherever Column A is NaN
        df.loc[df["A"].isna(), ["A", "B"]] = df2.values
        expected = DataFrame(
            data={
                "channel": [1, 2, 3],
                "A": ["String 1", "String 3", "String 2"],
                "B": [
                    Timestamp("2019-06-11 11:00:00"),
                    Timestamp("2019-06-11 12:00:00"),
                    Timestamp("2019-06-11 12:00:00"),
                ],
            }
        )
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "obj, key, exp",
        [
            (
                DataFrame([[1]], columns=Index([False])),
                IndexSlice[:, False],
                Series([1], name=False),
            ),
            (Series([1], index=Index([False])), False, [1]),
            (DataFrame([[1]], index=Index([False])), False, Series([1], name=False)),
        ],
    )
    def test_loc_getitem_single_boolean_arg(self, obj, key, exp):
        # [FUNCTION DOC]
        # Verifies that a single boolean key works when the index (or columns) are boolean (issue 44322).
        # Parametrized over (object, key, expected): a frame with a boolean column and a column selection gives a series named false; a series with boolean index and a scalar key returns the value; a frame with boolean index returns the row as a series named false.
        # GH 44322
        res = obj.loc[key]
        if isinstance(exp, (DataFrame, Series)):
            tm.assert_equal(res, exp)
        else:
            assert res == exp

    @pytest.mark.parametrize(
        "data",
        [
            [datetime(2025, 10, 30)],
            [Timestamp(2025, 10, 30)] * 2,
            [Timedelta(1)],
            [Timedelta(1), Timedelta(2)],
        ],
    )
    def test_loc_empty_slice_assignment_with_datetime(self, data):
        # [FUNCTION DOC]
        # Verifies that assigning a frame with an all-false boolean mask on datetime or timedelta data does not raise and leaves the frame unchanged (issue 50942). Parametrized over datetime, timestamp and timedelta data.
        # issue #50942
        # empty slice assignment with datetime or timedelta should not raise exceptions
        mask = [False] * len(data)

        df = DataFrame(data=data, columns=["A"])
        expected = df.copy()
        df.loc[mask] = df
        tm.assert_frame_equal(df, expected)


class TestLocBaseIndependent:
    # [CLASS DOC]
    # Tests of the label indexer that do not depend on the shared base index test class.
    # Tests for loc that do not depend on subclassing Base
    def test_loc_npstr(self):
        # [FUNCTION DOC]
        # Verifies that slicing a daily datetime index with an open-ended slice whose start is a numpy string element works (issue 45580): result equals the positional tail from position 151.
        # GH#45580
        df = DataFrame(index=date_range("2021", "2022"))
        result = df.loc[np.array(["2021/6/1"])[0] :]
        expected = df.iloc[151:]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "msg, key",
        [
            (r"Period\('2019', 'Y-DEC'\), 'foo', 'bar'", (Period(2019), "foo", "bar")),
            (r"Period\('2019', 'Y-DEC'\), 'y1', 'bar'", (Period(2019), "y1", "bar")),
            (r"Period\('2019', 'Y-DEC'\), 'foo', 'z1'", (Period(2019), "foo", "z1")),
            (
                r"Period\('2018', 'Y-DEC'\), Period\('2016', 'Y-DEC'\), 'bar'",
                (Period(2018), Period(2016), "bar"),
            ),
            (r"Period\('2018', 'Y-DEC'\), 'foo', 'y1'", (Period(2018), "foo", "y1")),
            (
                r"Period\('2017', 'Y-DEC'\), 'foo', Period\('2015', 'Y-DEC'\)",
                (Period(2017), "foo", Period(2015)),
            ),
            (r"Period\('2017', 'Y-DEC'\), 'z1', 'bar'", (Period(2017), "z1", "bar")),
        ],
    )
    def test_contains_raise_error_if_period_index_is_in_multi_index(self, msg, key):
        # [FUNCTION DOC]
        # Verifies that looking up a three-part key whose parts do not exist in a multi-level index containing periods raises a key error whose message shows the key (issue 20684). Parametrized over (expected message pattern, key). The docstring-like text in the function describes the parse helper that returns a tuple for mismatched types.
        # GH#20684
        """
        parse_datetime_string_with_reso return parameter if type not matched.
        PeriodIndex.get_loc takes returned value from parse_datetime_string_with_reso
        as a tuple.
        If first argument is Period and a tuple has 3 items,
        process go on not raise exception
        """
        df = DataFrame(
            {
                "A": [Period(2019), "x1", "x2"],
                "B": [Period(2018), Period(2016), "y1"],
                "C": [Period(2017), "z1", Period(2015)],
                "V1": [1, 2, 3],
                "V2": [10, 20, 30],
            }
        ).set_index(["A", "B", "C"])
        with pytest.raises(KeyError, match=msg):
            df.loc[key]

    def test_loc_getitem_missing_unicode_key(self):
        # [FUNCTION DOC]
        # Verifies that a missing non-ASCII column key raises a key error with that key, not an encoding error.
        df = DataFrame({"a": [1]})
        with pytest.raises(KeyError, match="\u05d0"):
            df.loc[:, "\u05d0"]  # should not raise UnicodeEncodeError

    def test_loc_getitem_dups(self):
        # [FUNCTION DOC]
        # Verifies repeated lookups on a duplicated index returning arrays are consistent (issue 5678): selecting a label then a column equals selecting a column then the label.
        # GH 5678
        # repeated getitems on a dup index returning an ndarray
        df = DataFrame(
            np.random.default_rng(2).random((20, 5)),
            index=["ABCDE"[x % 5] for x in range(20)],
        )
        expected = df.loc["A", 0]
        result = df.loc[:, 0].loc["A"]
        tm.assert_series_equal(result, expected)

    def test_loc_getitem_dups2(self):
        # [FUNCTION DOC]
        # Verifies that a one-row frame with five identically named columns returns the row as the same series by position and by label (issue 4726), including mixed value types.
        # GH4726
        # dup indexing with iloc/loc
        df = DataFrame(
            [[1, 2, "foo", "bar", Timestamp("20130101")]],
            columns=["a", "a", "a", "a", "a"],
            index=[1],
        )
        expected = Series(
            [1, 2, "foo", "bar", Timestamp("20130101")],
            index=["a", "a", "a", "a", "a"],
            name=1,
        )

        result = df.iloc[0]
        tm.assert_series_equal(result, expected)

        result = df.loc[1]
        tm.assert_series_equal(result, expected)

    def test_loc_setitem_dups(self):
        # [FUNCTION DOC]
        # Verifies in-place multiplication through label indexers on a frame with a duplicated string index (issue 6541), for a label with a list of columns, a label with one column (scalar result) and a duplicated label with a list of columns.
        # GH 6541
        df_orig = DataFrame(
            {
                "me": list("rttti"),
                "foo": list("aaade"),
                "bar": np.arange(5, dtype="float64") * 1.34 + 2,
                "bar2": np.arange(5, dtype="float64") * -0.34 + 2,
            }
        ).set_index("me")

        indexer = (
            "r",
            ["bar", "bar2"],
        )
        df = df_orig.copy()
        df.loc[indexer] *= 2.0
        tm.assert_series_equal(df.loc[indexer], 2.0 * df_orig.loc[indexer])

        indexer = (
            "r",
            "bar",
        )
        df = df_orig.copy()
        df.loc[indexer] *= 2.0
        assert df.loc[indexer] == 2.0 * df_orig.loc[indexer]

        indexer = (
            "t",
            ["bar", "bar2"],
        )
        df = df_orig.copy()
        df.loc[indexer] *= 2.0
        tm.assert_frame_equal(df.loc[indexer], 2.0 * df_orig.loc[indexer])

    def test_loc_setitem_slice(self):
        # [FUNCTION DOC]
        # Verifies that assigning to a column through a boolean mask keeps the dtype when the type matches and raises when it does not (issue 10503).
        # Assigning back values of the same unsigned 32-bit type keeps the dtype; assigning a column of unsigned 64-bit values into the 32-bit column raises a type error about an invalid value.
        # GH10503

        # assigning the same type should not change the type
        df1 = DataFrame({"a": [0, 1, 1], "b": Series([100, 200, 300], dtype="uint32")})
        ix = df1["a"] == 1
        newb1 = df1.loc[ix, "b"] + 1
        df1.loc[ix, "b"] = newb1
        expected = DataFrame(
            {"a": [0, 1, 1], "b": Series([100, 201, 301], dtype="uint32")}
        )
        tm.assert_frame_equal(df1, expected)

        # assigning a new type should get the inferred type
        df2 = DataFrame({"a": [0, 1, 1], "b": [100, 200, 300]}, dtype="uint64")
        ix = df1["a"] == 1
        newb2 = df2.loc[ix, "b"]
        with pytest.raises(TypeError, match="Invalid value"):
            df1.loc[ix, "b"] = newb2

    def test_loc_setitem_dtype(self):
        # [FUNCTION DOC]
        # Verifies that assigning float32 values to float64 columns via the label indexer over all rows and several columns writes in place and keeps float64 (issue 31340); the text column stays object.
        # GH31340
        df = DataFrame({"id": ["A"], "a": [1.2], "b": [0.0], "c": [-2.5]})
        cols = ["a", "b", "c"]
        df.loc[:, cols] = df.loc[:, cols].astype("float32")

        # pre-2.0 this setting would swap in new arrays, in 2.0 it is correctly
        #  in-place, consistent with non-split-path
        expected = DataFrame(
            {
                "id": ["A"],
                "a": np.array([1.2], dtype="float64"),
                "b": np.array([0.0], dtype="float64"),
                "c": np.array([-2.5], dtype="float64"),
            }
        )  # id is inferred as object

        tm.assert_frame_equal(df, expected)

    def test_getitem_label_list_with_missing(self):
        # [FUNCTION DOC]
        # Verifies that direct list-of-labels selection on a series with a missing label raises a key error saying labels are not in the index, for string and default integer indexes.
        s = Series(range(3), index=["a", "b", "c"])

        # consistency
        with pytest.raises(KeyError, match="not in index"):
            s[["a", "d"]]

        s = Series(range(3))
        with pytest.raises(KeyError, match="not in index"):
            s[[0, 3]]

    @pytest.mark.parametrize("index", [[True, False], [True, False, True, False]])
    def test_loc_getitem_bool_diff_len(self, index):
        # [FUNCTION DOC]
        # Verifies that a boolean list key whose length differs from the series raises an index error that gives both lengths (issue 26658). Parametrized over a shorter and a longer list.
        # GH26658
        s = Series([1, 2, 3])
        msg = f"Boolean index has wrong length: {len(index)} instead of {len(s)}"
        with pytest.raises(IndexError, match=msg):
            s.loc[index]

    def test_loc_getitem_int_slice(self):
        # [FUNCTION DOC]
        # Placeholder test with an empty body; there is nothing to verify.
        # KNOWN ISSUE: the test only holds a note asking whether something should be tested and performs no check.
        # TODO: test something here?
        pass

    def test_loc_to_fail(self):
        # [FUNCTION DOC]
        # Verifies that integer label lists on both axes of a string-labelled frame raise a key error saying none of the labels are in the columns (issue 3449).
        # GH3449
        df = DataFrame(
            np.random.default_rng(2).random((3, 3)),
            index=["a", "b", "c"],
            columns=["e", "f", "g"],
        )

        msg = (
            rf"\"None of \[Index\(\[1, 2\], dtype='{np.dtype(int)}'\)\] are "
            r"in the \[columns\]\""
        )
        with pytest.raises(KeyError, match=msg):
            df.loc[[1, 2], [1, 2]]

    def test_loc_to_fail2(self):
        # [FUNCTION DOC]
        # Verifies that the label indexer never falls back to positions (issue 7496).
        # On an object series with labels one and "a": a missing negative scalar raises a key error with only that key; a list of missing integers or an object index of text raises a key error saying none are in the index; after adding the label minus one, a list with a missing element raises a key error "not in index"; a list with a single missing label gives the none-found error; the same holds for list assignment after deleting a label.
        # GH  7496
        # loc should not fallback

        s = Series(dtype=object)
        s.loc[1] = 1
        s.loc["a"] = 2

        with pytest.raises(KeyError, match=r"^-1$"):
            s.loc[-1]

        msg = (
            rf"\"None of \[Index\(\[-1, -2\], dtype='{np.dtype(int)}'\)\] are "
            r"in the \[index\]\""
        )
        with pytest.raises(KeyError, match=msg):
            s.loc[[-1, -2]]

        msg = r"\"None of \[Index\(\['4'\], dtype='object'\)\] are in the \[index\]\""
        with pytest.raises(KeyError, match=msg):
            s.loc[Index(["4"], dtype=object)]

        s.loc[-1] = 3
        with pytest.raises(KeyError, match="not in index"):
            s.loc[[-1, -2]]

        s["a"] = 2
        msg = (
            rf"\"None of \[Index\(\[-2\], dtype='{np.dtype(int)}'\)\] are "
            r"in the \[index\]\""
        )
        with pytest.raises(KeyError, match=msg):
            s.loc[[-2]]

        del s["a"]

        with pytest.raises(KeyError, match=msg):
            s.loc[[-2]] = 0

    def test_loc_to_fail3(self):
        # [FUNCTION DOC]
        # Verifies that selecting a missing integer list with and without a trailing full slice both raise the same key error saying none of the labels are in the index (issue 7999).
        # inconsistency between .loc[values] and .loc[values,:]
        # GH 7999
        df = DataFrame([["a"], ["b"]], index=[1, 2], columns=["value"])

        msg = (
            rf"\"None of \[Index\(\[3\], dtype='{np.dtype(int)}'\)\] are "
            r"in the \[index\]\""
        )
        with pytest.raises(KeyError, match=msg):
            df.loc[[3], :]

        with pytest.raises(KeyError, match=msg):
            df.loc[[3]]

    def test_loc_getitem_list_with_fail(self):
        # [FUNCTION DOC]
        # Verifies that a list of labels raises a key error if any label is missing (issue 15747): a fully-present list works; a fully-missing list raises the none-found error showing the range index; a list with a present and a missing label raises a key error "not in index".
        # 15747
        # should KeyError if *any* missing labels

        s = Series([1, 2, 3])

        s.loc[[2]]

        msg = "None of [RangeIndex(start=3, stop=4, step=1)] are in the [index]"
        with pytest.raises(KeyError, match=re.escape(msg)):
            s.loc[[3]]

        # a non-match and a match
        with pytest.raises(KeyError, match="not in index"):
            s.loc[[2, 3]]

    def test_loc_index(self):
        # [FUNCTION DOC]
        # Verifies a boolean index (from mapping row labels) indexes rows like a boolean numpy array (issue 17131): the index object, its values and a nullable boolean array all give the same rows as the numpy boolean array.
        # gh-17131
        # a boolean index should index like a boolean numpy array

        df = DataFrame(
            np.random.default_rng(2).random(size=(5, 10)),
            index=["alpha_0", "alpha_1", "alpha_2", "beta_0", "beta_1"],
        )

        mask = df.index.map(lambda x: "alpha" in x)
        expected = df.loc[np.array(mask)]

        result = df.loc[mask]
        tm.assert_frame_equal(result, expected)

        result = df.loc[mask.values]
        tm.assert_frame_equal(result, expected)

        result = df.loc[pd.array(mask, dtype="boolean")]
        tm.assert_frame_equal(result, expected)

    def test_loc_general(self):
        # [FUNCTION DOC]
        # Verifies a label slice of columns followed by a position slice of rows gives the first two labels on both axes, and that a one-row frame with a timestamp and an integer returns the row as an object-dtype series.
        df = DataFrame(
            np.random.default_rng(2).random((4, 4)),
            columns=["A", "B", "C", "D"],
            index=["A", "B", "C", "D"],
        )

        # want this to work
        result = df.loc[:, "A":"B"].iloc[0:2, :]
        assert (result.columns == ["A", "B"]).all()
        assert (result.index == ["A", "B"]).all()

        # mixed type
        result = DataFrame({"a": [Timestamp("20130101")], "b": [1]}).iloc[0]
        expected = Series([Timestamp("20130101"), 1], index=["a", "b"], name=0)
        tm.assert_series_equal(result, expected)
        assert result.dtype == object

    @pytest.fixture
    def frame_for_consistency(self):
        # [FUNCTION DOC]
        # Fixture returning a five-row frame with a daily datetime column named date and an int64 column named val.
        return DataFrame(
            {
                "date": date_range("2000-01-01", "2000-01-5"),
                "val": Series(range(5), dtype=np.int64),
            }
        )

    @pytest.mark.parametrize(
        "val",
        [0, np.array(0, dtype=np.int64), np.array([0, 0, 0, 0, 0], dtype=np.int64)],
    )
    def test_loc_setitem_consistency(self, frame_for_consistency, val):
        # [FUNCTION DOC]
        # Verifies that assigning an integer scalar, a zero-dimensional integer array or an integer array to a whole datetime column via the label indexer raises a type error about an invalid value (issue 6149), consistent with plain setting. Parametrized over the three values.
        # GH 6149
        # coerce similarly for setitem and loc when rows have a null-slice
        df = frame_for_consistency.copy()
        with pytest.raises(TypeError, match="Invalid value"):
            df.loc[:, "date"] = val

    def test_loc_setitem_consistency_dt64_to_str(self, frame_for_consistency):
        # [FUNCTION DOC]
        # Same as above for assigning a string to the whole datetime column: raises a type error about an invalid value (issue 6149).
        # GH 6149
        # coerce similarly for setitem and loc when rows have a null-slice

        df = frame_for_consistency.copy()
        with pytest.raises(TypeError, match="Invalid value"):
            df.loc[:, "date"] = "foo"

    def test_loc_setitem_consistency_dt64_to_float(self, frame_for_consistency):
        # [FUNCTION DOC]
        # Same as above for assigning a float: raises a type error about an invalid value (issue 6149).
        # GH 6149
        # coerce similarly for setitem and loc when rows have a null-slice
        df = frame_for_consistency.copy()
        with pytest.raises(TypeError, match="Invalid value"):
            df.loc[:, "date"] = 1.0

    def test_loc_setitem_consistency_single_row(self):
        # [FUNCTION DOC]
        # Verifies assigning a string over a one-row datetime column through the label indexer raises a type error about an invalid value (issue 15494).
        # GH 15494
        # setting on frame with single row
        df = DataFrame({"date": Series([Timestamp("20180101")])})
        with pytest.raises(TypeError, match="Invalid value"):
            df.loc[:, "date"] = "string"

    def test_loc_setitem_consistency_empty(self):
        # [FUNCTION DOC]
        # Verifies assigning a scalar to a column of an empty frame via the label indexer is a no-op that emits no warning and does not change the dtype, whereas plain column assignment swaps in a new integer column.
        # empty (essentially noops)
        # before the enforcement of #45333 in 2.0, the loc.setitem here would
        #  change the dtype of df.x to int64
        expected = DataFrame(columns=["x", "y"])
        df = DataFrame(columns=["x", "y"])
        with tm.assert_produces_warning(None):
            df.loc[:, "x"] = 1
        tm.assert_frame_equal(df, expected)

        # setting with setitem swaps in a new array, so changes the dtype
        df = DataFrame(columns=["x", "y"])
        df["x"] = 1
        expected["x"] = expected["x"].astype(np.int64)
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_consistency_slice_column_len(self, using_infer_string):
        # [FUNCTION DOC]
        # Verifies setting columns of a frame with multi-level rows and columns using converted datetimes (issue 10408), parametrized by the infer-string fixture.
        # With string inference on, assigning datetimes into text columns raises a type error and the test returns early. Otherwise: both date columns are converted in place, the object types are inferred, a new duration column is added, and assigning the duration converted to float (not possible in place) raises a type error about an invalid value.
        # .loc[:,column] setting with slice == len of the column
        # GH10408
        levels = [
            ["Region_1"] * 4,
            ["Site_1", "Site_1", "Site_2", "Site_2"],
            [3987227376, 3980680971, 3977723249, 3977723089],
        ]
        mi = MultiIndex.from_arrays(levels, names=["Region", "Site", "RespondentID"])

        clevels = [
            ["Respondent", "Respondent", "Respondent", "OtherCat", "OtherCat"],
            ["Something", "StartDate", "EndDate", "Yes/No", "SomethingElse"],
        ]
        cols = MultiIndex.from_arrays(clevels, names=["Level_0", "Level_1"])

        values = [
            ["A", "5/25/2015 10:59", "5/25/2015 11:22", "Yes", np.nan],
            ["A", "5/21/2015 9:40", "5/21/2015 9:52", "Yes", "Yes"],
            ["A", "5/20/2015 8:27", "5/20/2015 8:41", "Yes", np.nan],
            ["A", "5/20/2015 8:33", "5/20/2015 9:09", "Yes", "No"],
        ]
        df = DataFrame(values, index=mi, columns=cols)

        ctx = contextlib.nullcontext()
        if using_infer_string:
            ctx = pytest.raises(TypeError, match="Invalid value")

        with ctx:
            df.loc[:, ("Respondent", "StartDate")] = to_datetime(
                df.loc[:, ("Respondent", "StartDate")]
            )
        with ctx:
            df.loc[:, ("Respondent", "EndDate")] = to_datetime(
                df.loc[:, ("Respondent", "EndDate")]
            )

        if using_infer_string:
            # infer-objects won't infer stuff anymore
            return

        df = df.infer_objects()

        # Adding a new key
        df.loc[:, ("Respondent", "Duration")] = (
            df.loc[:, ("Respondent", "EndDate")]
            - df.loc[:, ("Respondent", "StartDate")]
        )

        # timedelta64[m] -> float, so this cannot be done inplace, so
        #  no warning
        with pytest.raises(TypeError, match="Invalid value"):
            df.loc[:, ("Respondent", "Duration")] = df.loc[
                :, ("Respondent", "Duration")
            ] / Timedelta(60_000_000_000)

    @pytest.mark.parametrize("unit", ["Y", "M", "D", "h", "m", "s", "ms", "us"])
    def test_loc_assign_non_ns_datetime(self, unit):
        # [FUNCTION DOC]
        # Verifies assigning datetime64 arrays of non-nanosecond units (year to microsecond) through the label indexer gives the same result as plain assignment (issue 27395). Parametrized over eight units.
        # GH 27395, non-ns dtype assignment via .loc should work
        # and return the same result when using simple assignment
        df = DataFrame(
            {
                "timestamp": [
                    np.datetime64("2017-02-11 12:41:29"),
                    np.datetime64("1991-11-07 04:22:37"),
                ]
            }
        )

        df.loc[:, unit] = df.loc[:, "timestamp"].values.astype(f"datetime64[{unit}]")
        df["expected"] = df.loc[:, "timestamp"].values.astype(f"datetime64[{unit}]")
        expected = Series(df.loc[:, "expected"], name=unit)
        tm.assert_series_equal(df.loc[:, unit], expected)

    def test_loc_modify_datetime(self):
        # [FUNCTION DOC]
        # Verifies copying a datetime column into a new column through the label indexer, whole column and then a list of rows (issue 28837), gives datetimes equal to the original with microsecond unit.
        # see gh-28837
        df = DataFrame.from_dict(
            {"date": [1485264372711, 1485265925110, 1540215845888, 1540282121025]}
        )

        df["date_dt"] = to_datetime(df["date"], unit="ms", cache=True).dt.as_unit("us")

        df.loc[:, "date_dt_cp"] = df.loc[:, "date_dt"]
        df.loc[[2, 3], "date_dt_cp"] = df.loc[[2, 3], "date_dt"]

        expected = DataFrame(
            [
                [1485264372711, "2017-01-24 13:26:12.711", "2017-01-24 13:26:12.711"],
                [1485265925110, "2017-01-24 13:52:05.110", "2017-01-24 13:52:05.110"],
                [1540215845888, "2018-10-22 13:44:05.888", "2018-10-22 13:44:05.888"],
                [1540282121025, "2018-10-23 08:08:41.025", "2018-10-23 08:08:41.025"],
            ],
            columns=["date", "date_dt", "date_dt_cp"],
        )

        columns = ["date_dt", "date_dt_cp"]
        expected[columns] = expected[columns].apply(to_datetime)

        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("has_ref", [True, False])
    def test_loc_setitem_frame_with_reindex(self, has_ref):
        # [FUNCTION DOC]
        # Verifies that setting integers into a float column with an unordered row list aligns by label and is in place so the dtype stays float (issue 6254). Parametrized over whether another object holds a reference to the frame.
        # GH#6254 setting issue
        df = DataFrame(index=[3, 5, 4], columns=["A"], dtype=float)
        if has_ref:
            view = df[:]  # noqa: F841
        df.loc[[4, 3, 5], "A"] = np.array([1, 2, 3], dtype="int64")

        # setting integer values into a float dataframe with loc is inplace,
        #  so we retain float dtype
        ser = Series([2, 3, 1], index=[3, 5, 4], dtype=float)
        expected = DataFrame({"A": ser})
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_with_reindex_mixed(self):
        # [FUNCTION DOC]
        # Verifies the same for a frame with an additional text column (issue 40480): values aligned by row label, column stays float, text column preserved.
        # GH#40480
        df = DataFrame(index=[3, 5, 4], columns=["A", "B"], dtype=float)
        df["B"] = "string"
        df.loc[[4, 3, 5], "A"] = np.array([1, 2, 3], dtype="int64")
        ser = Series([2, 3, 1], index=[3, 5, 4], dtype="int64")
        # pre-2.0 this setting swapped in a new array, now it is inplace
        #  consistent with non-split-path
        expected = DataFrame({"A": ser.astype(float)})
        expected["B"] = "string"
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_with_inverted_slice(self):
        # [FUNCTION DOC]
        # Verifies setting through a reversed label slice (issue 40480) assigns in reversed order, in place, preserving float dtype and the text column.
        # GH#40480
        df = DataFrame(index=[1, 2, 3], columns=["A", "B"], dtype=float)
        df["B"] = "string"
        df.loc[slice(3, 0, -1), "A"] = np.array([1, 2, 3], dtype="int64")
        # pre-2.0 this setting swapped in a new array, now it is inplace
        #  consistent with non-split-path
        expected = DataFrame({"A": [3.0, 2.0, 1.0], "B": "string"}, index=[1, 2, 3])
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("has_ref", [True, False])
    def test_loc_setitem_empty_frame(self, has_ref):
        # [FUNCTION DOC]
        # Verifies setting values into columns of a frame created with only an index (issue 6252), parametrized over whether a reference exists. Because the columns were initialized as float, assignment is in place and keeps float dtype for the first column; the second column holds the values for the keys present, with missing for others.
        # GH#6252 setting with an empty frame
        keys1 = ["@" + str(i) for i in range(5)]
        val1 = np.arange(5, dtype="int64")

        keys2 = ["@" + str(i) for i in range(4)]
        val2 = np.arange(4, dtype="int64")

        index = list(set(keys1).union(keys2))
        df = DataFrame(index=index)
        df["A"] = np.nan
        if has_ref:
            view = df[:]  # noqa: F841
        df.loc[keys1, "A"] = val1

        df["B"] = np.nan
        df.loc[keys2, "B"] = val2

        # Because df["A"] was initialized as float64, setting values into it
        #  is inplace, so that dtype is retained
        sera = Series(val1, index=keys1, dtype=np.float64)
        serb = Series(val2, index=keys2)
        expected = DataFrame({"A": sera, "B": serb}, columns=Index(["A", "B"])).reindex(
            index=index
        )
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("has_ref", [True, False])
    def test_loc_setitem_frame(self, has_ref):
        # [FUNCTION DOC]
        # Verifies scalar and block setting through the label indexer is visible through the position indexer: setting one cell, then setting columns B to D to zero (issue-less). Parametrized over whether a reference exists.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=list("abcd"),
            columns=list("ABCD"),
        )
        if has_ref:
            view = df[:]  # noqa: F841

        result = df.iloc[0, 0]

        df.loc["a", "A"] = 1
        result = df.loc["a", "A"]
        assert result == 1

        result = df.iloc[0, 0]
        assert result == 1

        df.loc[:, "B":"D"] = 0
        expected = df.loc[:, "B":"D"]
        result = df.iloc[:, 1:]
        tm.assert_frame_equal(result, expected)

    def test_loc_setitem_frame_nan_int_coercion_invalid(self):
        # [FUNCTION DOC]
        # Verifies that a boolean-mask assignment that selects no rows does not coerce an all-missing float column (issue 8669).
        # GH 8669
        # invalid coercion of nan -> int
        df = DataFrame({"A": [1, 2, 3], "B": np.nan})
        df.loc[df.B > df.A, "B"] = df.A
        expected = DataFrame({"A": [1, 2, 3], "B": np.nan})
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_mixed_labels(self):
        # [FUNCTION DOC]
        # Verifies a frame with integer and text column labels: selecting a row and integer-labelled columns gives an integer series with an object index, and assigning a list there sets those cells (issue 6546).
        # GH 6546
        # setting with mixed labels
        df = DataFrame({1: [1, 2], 2: [3, 4], "a": ["a", "b"]})

        result = df.loc[0, [1, 2]]
        expected = Series(
            [1, 3], index=Index([1, 2], dtype=object), dtype="int64", name=0
        )
        tm.assert_series_equal(result, expected)

        expected = DataFrame({1: [5, 2], 2: [6, 4], "a": ["a", "b"]})
        df.loc[0, [1, 2]] = [5, 6]
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_multiples(self):
        # [FUNCTION DOC]
        # Verifies assigning a frame to a slice of rows aligns on the (modified) index and does not align on the original labels (multiple setting): first with text and integer columns; then with a datetime column (nanosecond unit) and integer column, where the assigned rows come from earlier rows.
        # multiple setting
        df = DataFrame(
            {"A": ["foo", "bar", "baz"], "B": Series(range(3), dtype=np.int64)}
        )
        rhs = df.loc[1:2]
        rhs.index = df.index[0:2]
        df.loc[0:1] = rhs
        expected = DataFrame(
            {"A": ["bar", "baz", "baz"], "B": Series([1, 2, 2], dtype=np.int64)}
        )
        tm.assert_frame_equal(df, expected)

        # multiple setting with frame on rhs (with M8)
        df = DataFrame(
            {
                "date": date_range("2000-01-01", "2000-01-5", unit="ns"),
                "val": Series(range(5), dtype=np.int64),
            }
        )
        expected = DataFrame(
            {
                "date": [
                    Timestamp("20000101"),
                    Timestamp("20000102"),
                    Timestamp("20000101"),
                    Timestamp("20000102"),
                    Timestamp("20000103"),
                ],
                "val": Series([0, 1, 0, 1, 2], dtype=np.int64),
            }
        )
        expected["date"] = expected["date"].astype("M8[ns]")
        rhs = df.loc[0:2]
        rhs.index = df.index[2:5]
        df.loc[2:4] = rhs
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "indexer", [["A"], slice(None, "A", None), np.array(["A"])]
    )
    @pytest.mark.parametrize("value", [["Z"], np.array(["Z"])])
    def test_loc_setitem_with_scalar_index(self, indexer, value):
        # [FUNCTION DOC]
        # Verifies that setting a single row with a list-like of one value over a column indexer given as list, slice or array is applied element by element, not by passing the whole list to a setter (issue 19474). Parametrized over three column indexers and two value containers. The first column is object to avoid casting; the cell result must be the scalar text.
        # GH #19474
        # assigning like "df.loc[0, ['A']] = ['Z']" should be evaluated
        # elementwisely, not using "setter('A', ['Z'])".

        # Set object dtype to avoid upcast when setting 'Z'
        df = DataFrame([[1, 2], [3, 4]], columns=["A", "B"]).astype({"A": object})
        df.loc[0, indexer] = value
        result = df.loc[0, "A"]

        assert is_scalar(result) and result == "Z"

    @pytest.mark.parametrize(
        "index,box,expected",
        [
            (
                ([0, 2], ["A", "B", "C", "D"]),
                7,
                DataFrame(
                    [[7, 7, 7, 7], [3, 4, np.nan, np.nan], [7, 7, 7, 7]],
                    columns=["A", "B", "C", "D"],
                ),
            ),
            (
                (1, ["C", "D"]),
                [7, 8],
                DataFrame(
                    [[1, 2, np.nan, np.nan], [3, 4, 7, 8], [5, 6, np.nan, np.nan]],
                    columns=["A", "B", "C", "D"],
                ),
            ),
            (
                (1, ["A", "B", "C"]),
                np.array([7, 8, 9], dtype=np.int64),
                DataFrame(
                    [[1, 2, np.nan], [7, 8, 9], [5, 6, np.nan]], columns=["A", "B", "C"]
                ),
            ),
            (
                (slice(1, 3, None), ["B", "C", "D"]),
                [[7, 8, 9], [10, 11, 12]],
                DataFrame(
                    [[1, 2, np.nan, np.nan], [3, 7, 8, 9], [5, 10, 11, 12]],
                    columns=["A", "B", "C", "D"],
                ),
            ),
            (
                (slice(1, 3, None), ["C", "A", "D"]),
                np.array([[7, 8, 9], [10, 11, 12]], dtype=np.int64),
                DataFrame(
                    [[1, 2, np.nan, np.nan], [8, 4, 7, 9], [11, 6, 10, 12]],
                    columns=["A", "B", "C", "D"],
                ),
            ),
            (
                (slice(None, None, None), ["A", "C"]),
                DataFrame([[7, 8], [9, 10], [11, 12]], columns=["A", "C"]),
                DataFrame(
                    [[7, 2, 8], [9, 4, 10], [11, 6, 12]], columns=["A", "B", "C"]
                ),
            ),
        ],
    )
    def test_loc_setitem_missing_columns(self, index, box, expected):
        # [FUNCTION DOC]
        # Verifies enlargement with missing columns when setting (issue 29334).
        # Parametrized over (indexer, assigned value, expected frame): setting rows and new columns with a scalar, setting one row for new columns with a list or array, setting a row range with nested lists or a 2D array including reordered new columns, and setting with a frame aligned on columns. New columns are appended, unmatched entries are missing.
        # GH 29334
        df = DataFrame([[1, 2], [3, 4], [5, 6]], columns=["A", "B"])

        df.loc[index] = box
        tm.assert_frame_equal(df, expected)

    def test_loc_coercion(self):
        # [FUNCTION DOC]
        # Verifies that selecting rows by position lists from a frame with a timezone-aware timestamp and a missing value keeps the dtype (issue 12411).
        # KNOWN ISSUE: the test is named for the label indexer, but all selections use the position indexer.
        # GH#12411
        df = DataFrame({"date": [Timestamp("20130101").tz_localize("UTC"), pd.NaT]})
        expected = df.dtypes

        result = df.iloc[[0]]
        tm.assert_series_equal(result.dtypes, expected)

        result = df.iloc[[1]]
        tm.assert_series_equal(result.dtypes, expected)

    def test_loc_coercion2(self):
        # [FUNCTION DOC]
        # Verifies that position-list selection of datetimes including one outside nanosecond range keeps the dtype (issue 12045).
        # KNOWN ISSUE: named for the label indexer, but only the position indexer is used.
        # GH#12045
        df = DataFrame({"date": [datetime(2012, 1, 1), datetime(1012, 1, 2)]})
        expected = df.dtypes

        result = df.iloc[[0]]
        tm.assert_series_equal(result.dtypes, expected)

        result = df.iloc[[1]]
        tm.assert_series_equal(result.dtypes, expected)

    def test_loc_coercion3(self):
        # [FUNCTION DOC]
        # Verifies that position slices of a column of text with missing values keep the dtype (issue 11594).
        # KNOWN ISSUE: named for the label indexer, but only the position indexer is used.
        # GH#11594
        df = DataFrame({"text": ["some words"] + [None] * 9})
        expected = df.dtypes

        result = df.iloc[0:2]
        tm.assert_series_equal(result.dtypes, expected)

        result = df.iloc[3:]
        tm.assert_series_equal(result.dtypes, expected)

    def test_setitem_new_key_tz(self, indexer_sl):
        # [FUNCTION DOC]
        # Verifies that setting two new labels with timezone-aware timestamps on an empty object series through the label and bracket setter wrappers does not raise and gives a series of two UTC timestamps (issue 12862). Parametrized by the indexer fixture.
        # GH#12862 should not raise on assigning the second value
        vals = [
            to_datetime(42).tz_localize("UTC"),
            to_datetime(666).tz_localize("UTC"),
        ]
        expected = Series(vals, index=Index(["foo", "bar"]))

        ser = Series(dtype=object)
        indexer_sl(ser)["foo"] = vals[0]
        indexer_sl(ser)["bar"] = vals[1]

        tm.assert_series_equal(ser, expected)

    def test_loc_non_unique(self):
        # [FUNCTION DOC]
        # Verifies slicing with duplicate labels (issue 3659).
        # On a non-monotonic duplicated index, slices starting at a duplicated label raise a key error with the message "cannot get left slice bound for non-unique label". After sorting the index, open-ended and bounded slices work and return the rows covering the duplicate labels.
        # GH3659
        # non-unique indexer with loc slice
        # https://groups.google.com/forum/?fromgroups#!topic/pydata/zTm2No0crYs

        # these are going to raise because the we are non monotonic
        df = DataFrame(
            {"A": [1, 2, 3, 4, 5, 6], "B": [3, 4, 5, 6, 7, 8]}, index=[0, 1, 0, 1, 2, 3]
        )
        msg = "'Cannot get left slice bound for non-unique label: 1'"
        with pytest.raises(KeyError, match=msg):
            df.loc[1:]
        msg = "'Cannot get left slice bound for non-unique label: 0'"
        with pytest.raises(KeyError, match=msg):
            df.loc[0:]
        msg = "'Cannot get left slice bound for non-unique label: 1'"
        with pytest.raises(KeyError, match=msg):
            df.loc[1:2]

        # monotonic are ok
        df = DataFrame(
            {"A": [1, 2, 3, 4, 5, 6], "B": [3, 4, 5, 6, 7, 8]}, index=[0, 1, 0, 1, 2, 3]
        ).sort_index(axis=0)
        result = df.loc[1:]
        expected = DataFrame({"A": [2, 4, 5, 6], "B": [4, 6, 7, 8]}, index=[1, 1, 2, 3])
        tm.assert_frame_equal(result, expected)

        result = df.loc[0:]
        tm.assert_frame_equal(result, df)

        result = df.loc[1:2]
        expected = DataFrame({"A": [2, 4, 5], "B": [4, 6, 7]}, index=[1, 1, 2])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.arm_slow
    @pytest.mark.slow
    @pytest.mark.parametrize("length, l2", [[900, 100], [900000, 100000]])
    def test_loc_non_unique_memory_error(self, length, l2):
        # [FUNCTION DOC]
        # Verifies that a large selection on a non-unique index does not run out of memory and returns the expected rows (issue 4280).
        # Marked slow and slow on arm. Parametrized over (length, number of duplicated rows): (900, 100) and (900000, 100000). Builds a frame of random rows followed by rows of ones with the label zero repeated; the index is not unique. Selecting a range of labels returns the first row, the repeated rows and the following rows.
        # GH 4280
        # non_unique index with a large selection triggers a memory error

        columns = list("ABCDEFG")

        df = pd.concat(
            [
                DataFrame(
                    np.random.default_rng(2).standard_normal((length, len(columns))),
                    index=np.arange(length),
                    columns=columns,
                ),
                DataFrame(np.ones((l2, len(columns))), index=[0] * l2, columns=columns),
            ]
        )

        assert df.index.is_unique is False

        mask = np.arange(l2)
        result = df.loc[mask]
        expected = pd.concat(
            [
                df.take([0]),
                DataFrame(
                    np.ones((len(mask), len(columns))),
                    index=[0] * len(mask),
                    columns=columns,
                ),
                df.take(mask[1:]),
            ]
        )
        tm.assert_frame_equal(result, expected)

    def test_loc_name(self):
        # [FUNCTION DOC]
        # Verifies the index name is preserved when selecting with a list of labels, for position and label selectors (issue 3880).
        # GH 3880
        df = DataFrame([[1, 1], [1, 1]])
        df.index.name = "index_name"
        result = df.iloc[[0, 1]].index.name
        assert result == "index_name"

        result = df.loc[[0, 1]].index.name
        assert result == "index_name"

    def test_loc_empty_list_indexer_is_ok(self):
        # [FUNCTION DOC]
        # Verifies that empty list indexers on columns, rows, or alone return empty frames equal to the positional empty selections, keeping the index and column types and names.
        df = DataFrame(
            np.ones((5, 2)),
            index=Index([f"i-{i}" for i in range(5)], name="a"),
            columns=Index([f"i-{i}" for i in range(2)], name="a"),
        )
        # vertical empty
        tm.assert_frame_equal(
            df.loc[:, []], df.iloc[:, :0], check_index_type=True, check_column_type=True
        )
        # horizontal empty
        tm.assert_frame_equal(
            df.loc[[], :], df.iloc[:0, :], check_index_type=True, check_column_type=True
        )
        # horizontal empty
        tm.assert_frame_equal(
            df.loc[[]], df.iloc[:0, :], check_index_type=True, check_column_type=True
        )

    def test_identity_slice_returns_new_object(self):
        # [FUNCTION DOC]
        # Verifies that a full slice returns a new object that shares data (a shallow copy) (issue 13873).
        # For a frame: the label full slice, the plain full slice and the two-axis full slice are not the same object as the original; the data memory is shared; assigning a whole column afterwards in place through the label setter does not alter the earlier result (depends on copy-on-write); selecting a single column is a new object. Same idea for a series with position slices.
        # GH13873

        original_df = DataFrame({"a": [1, 2, 3]})
        sliced_df = original_df.loc[:]
        assert sliced_df is not original_df
        assert original_df[:] is not original_df
        assert original_df.loc[:, :] is not original_df

        # should be a shallow copy
        assert np.shares_memory(original_df["a"]._values, sliced_df["a"]._values)

        # Setting using .loc[:, "a"] sets inplace so alters both sliced and orig
        # depending on CoW
        original_df.loc[:, "a"] = [4, 4, 4]
        assert (sliced_df["a"] == [1, 2, 3]).all()

        # These should not return copies
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        assert df[0] is not df.loc[:, 0]

        # Same tests for Series
        original_series = Series([1, 2, 3, 4, 5, 6])
        sliced_series = original_series.loc[:]
        assert sliced_series is not original_series
        assert original_series[:] is not original_series

        original_series[:3] = [7, 8, 9]
        assert all(sliced_series[:3] == [1, 2, 3])

    def test_loc_copy_vs_view(self, request):
        # [FUNCTION DOC]
        # Verifies that modifying an object selected by a column label or by the index and column does not modify the original frame (issue 15631). The request fixture is accepted but not used.
        # GH 15631
        x = DataFrame(zip(range(3), range(3)), columns=["a", "b"])

        y = x.copy()
        q = y.loc[:, "a"]
        q += 2

        tm.assert_frame_equal(x, y)

        z = x.copy()
        q = z.loc[x.index, "a"]
        q += 2

        tm.assert_frame_equal(x, z)

    def test_loc_uint64(self):
        # [FUNCTION DOC]
        # Verifies the label indexer accepts the largest unsigned 64-bit value as an index label as scalar, in a list and in a full list (issue 20722).
        # GH20722
        # Test whether loc accept uint64 max value as index.
        umax = np.iinfo("uint64").max
        ser = Series([1, 2], index=[umax - 1, umax])

        result = ser.loc[umax - 1]
        expected = ser.iloc[0]
        assert result == expected

        result = ser.loc[[umax - 1]]
        expected = ser.iloc[[0]]
        tm.assert_series_equal(result, expected)

        result = ser.loc[[umax - 1, umax]]
        tm.assert_series_equal(result, ser)

    def test_loc_uint64_disallow_negative(self):
        # [FUNCTION DOC]
        # Verifies a negative scalar or list key on an unsigned 64-bit index raises a key error and does not wrap around (issue 41775).
        # GH#41775
        umax = np.iinfo("uint64").max
        ser = Series([1, 2], index=[umax - 1, umax])

        with pytest.raises(KeyError, match="-1"):
            # don't wrap around
            ser.loc[-1]

        with pytest.raises(KeyError, match="-1"):
            # don't wrap around
            ser.loc[[-1]]

    def test_loc_setitem_empty_append_expands_rows(self):
        # [FUNCTION DOC]
        # Verifies that label-assigning a list of three values to a column of a frame with columns but no rows appends rows to fit the data, the other column gets missing object values (issue 6173).
        # GH6173, various appends to an empty dataframe

        data = [1, 2, 3]
        expected = DataFrame(
            {"x": data, "y": np.array([np.nan] * len(data), dtype=object)}
        )

        # appends to fit length of data
        df = DataFrame(columns=["x", "y"])
        df.loc[:, "x"] = data
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_append_expands_rows_mixed_dtype(self):
        # [FUNCTION DOC]
        # Same as above but with the target column first cast to integer so the split path is used (issue 37932); same expected result.
        # GH#37932 same as test_loc_setitem_empty_append_expands_rows
        #  but with mixed dtype so we go through take_split_path
        data = [1, 2, 3]
        expected = DataFrame(
            {"x": data, "y": np.array([np.nan] * len(data), dtype=object)}
        )

        df = DataFrame(columns=["x", "y"])
        df["x"] = df["x"].astype(np.int64)
        df.loc[:, "x"] = data
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_append_single_value(self):
        # [FUNCTION DOC]
        # Verifies label-assigning one scalar into a new row of an empty float frame appends exactly one row, leaving the other column missing.
        # only appends one value
        expected = DataFrame({"x": [1.0], "y": [np.nan]})
        df = DataFrame(columns=["x", "y"], dtype=float)
        df.loc[0, "x"] = expected.loc[0, "x"]
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_append_raises(self):
        # [FUNCTION DOC]
        # Verifies errors when assigning to an empty frame (issue 6173): assigning a list to a column for absent row labels raises a key error saying none of the labels are in the index; assigning a list of two values to a three-row-wide slice raises a value error about setting an array element with a sequence.
        # GH6173, various appends to an empty dataframe

        data = [1, 2]
        df = DataFrame(columns=["x", "y"])
        df.index = df.index.astype(np.int64)
        msg = r"None of .*Index.* are in the \[index\]"
        with pytest.raises(KeyError, match=msg):
            df.loc[[0, 1], "x"] = data

        msg = "setting an array element with a sequence."
        with pytest.raises(ValueError, match=msg):
            df.loc[0:2, "x"] = data

    def test_indexing_zerodim_np_array(self):
        # [FUNCTION DOC]
        # Verifies a zero-dimensional integer array as the key selects the row as a series for frames (issue 24924).
        # GH24924
        df = DataFrame([[1, 2], [3, 4]])
        result = df.loc[np.array(0)]
        s = Series([1, 2], name=0)
        tm.assert_series_equal(result, s)

    def test_series_indexing_zerodim_np_array(self):
        # [FUNCTION DOC]
        # Verifies a zero-dimensional integer array key on a series returns the scalar element (issue 24924).
        # GH24924
        s = Series([1, 2])
        result = s.loc[np.array(0)]
        assert result == 1

    def test_loc_reverse_assignment(self):
        # [FUNCTION DOC]
        # Verifies assigning a list through a reversed label slice on a float series with a range index fills labels in descending order, leaving the rest missing (issue 26939).
        # GH26939
        data = [1, 2, 3, 4, 5, 6] + [None] * 4
        expected = Series(data, index=range(2010, 2020))

        result = Series(index=range(2010, 2020), dtype=np.float64)
        result.loc[2015:2010:-1] = [6, 5, 4, 3, 2, 1]

        tm.assert_series_equal(result, expected)

    def test_loc_setitem_str_to_small_float_conversion_type(self, using_infer_string):
        # [FUNCTION DOC]
        # Verifies assigning floats to a text column of tiny numeric strings (issue 20388). Parametrized by the infer-string fixture.
        # Label assignment over all rows tries to set in place: with string inference it raises a type error about an invalid value; otherwise the column stays object holding floats. Plain column assignment replaces the column with a float column.
        # KNOWN ISSUE: a comment in the code marks an issue number as unknown and there is a first comparison of two identically built frames that checks nothing.
        # GH#20388

        col_data = [str(np.random.default_rng(2).random() * 1e-12) for _ in range(5)]
        result = DataFrame(col_data, columns=["A"])
        expected = DataFrame(col_data, columns=["A"])
        tm.assert_frame_equal(result, expected)

        # assigning with loc/iloc attempts to set the values inplace, which
        #  in this case is successful
        if using_infer_string:
            with pytest.raises(TypeError, match="Invalid value"):
                result.loc[result.index, "A"] = [float(x) for x in col_data]
        else:
            result.loc[result.index, "A"] = [float(x) for x in col_data]
            expected = DataFrame(col_data, columns=["A"], dtype=float).astype(object)
            tm.assert_frame_equal(result, expected)

        # assigning the entire column using __setitem__ swaps in the new array
        # GH#???
        result["A"] = [float(x) for x in col_data]
        expected = DataFrame(col_data, columns=["A"], dtype=float)
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_time_object(self, frame_or_series):
        # [FUNCTION DOC]
        # Verifies selection with a time-of-day key on a five-minute datetime index (series or frame, via the fixture): the result equals the boolean mask on hour and minute; the same after a date-string slice of the last day, with frequencies reset before comparing.
        rng = date_range("1/1/2000", "1/5/2000", freq="5min")
        mask = (rng.hour == 9) & (rng.minute == 30)

        obj = DataFrame(
            np.random.default_rng(2).standard_normal((len(rng), 3)), index=rng
        )
        obj = tm.get_obj(obj, frame_or_series)

        result = obj.loc[time(9, 30)]
        exp = obj.loc[mask]
        tm.assert_equal(result, exp)

        chunk = obj.loc["1/4/2000":]
        result = chunk.loc[time(9, 30)]
        expected = result[-1:]

        # Without resetting the freqs, these are 5 min and 1440 min, respectively
        result.index = result.index._with_freq(None)
        expected.index = expected.index._with_freq(None)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("spmatrix_t", ["coo_matrix", "csc_matrix", "csr_matrix"])
    @pytest.mark.parametrize("dtype", [np.complex128, np.float64, np.int64, bool])
    def test_loc_getitem_range_from_spmatrix(self, spmatrix_t, dtype):
        # [FUNCTION DOC]
        # Verifies selecting a range of rows from a frame built from a sparse matrix keeps the values and sparse dtypes (issues 34526 and 34540). Parametrized over sparse matrix formats (coo, csc, csr) and dtypes (complex, float, int, bool). Skipped if scipy is missing. Uses a rectangular identity matrix so the last two columns are purely sparse; dense values must match the matrix rows and each column must have the sparse dtype.
        sp_sparse = pytest.importorskip("scipy.sparse")

        spmatrix_t = getattr(sp_sparse, spmatrix_t)

        # The bug is triggered by a sparse matrix with purely sparse columns.  So the
        # recipe below generates a rectangular matrix of dimension (5, 7) where all the
        # diagonal cells are ones, meaning the last two columns are purely sparse.
        rows, cols = 5, 7
        spmatrix = spmatrix_t(np.eye(rows, cols, dtype=dtype), dtype=dtype)
        df = DataFrame.sparse.from_spmatrix(spmatrix)

        # regression test for GH#34526
        itr_idx = range(2, rows)
        result = np.nan_to_num(df.loc[itr_idx].values)
        expected = spmatrix.toarray()[itr_idx]
        tm.assert_numpy_array_equal(result, expected)

        # regression test for GH#34540
        result = df.loc[itr_idx].dtypes.values
        expected = np.full(cols, SparseDtype(dtype))
        tm.assert_numpy_array_equal(result, expected)

    def test_loc_getitem_listlike_all_retains_sparse(self):
        # [FUNCTION DOC]
        # Verifies that selecting all rows via a list of labels keeps the sparse dtype.
        df = DataFrame({"A": pd.array([0, 0], dtype=SparseDtype("int64"))})
        result = df.loc[[0, 1]]
        tm.assert_frame_equal(result, df)

    def test_loc_getitem_sparse_frame(self):
        # [FUNCTION DOC]
        # Verifies row selection by range from a sparse frame, and a second chained selection, keeps the sparse integer dtype (issue 34687). Skipped without scipy.
        # GH34687
        sp_sparse = pytest.importorskip("scipy.sparse")

        df = DataFrame.sparse.from_spmatrix(sp_sparse.eye(5, dtype=np.int64))
        result = df.loc[range(2)]
        expected = DataFrame(
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
            dtype=SparseDtype(np.int64),
        )
        tm.assert_frame_equal(result, expected)

        result = df.loc[range(2)].loc[range(1)]
        expected = DataFrame([[1, 0, 0, 0, 0]], dtype=SparseDtype(np.int64))
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_sparse_series(self):
        # [FUNCTION DOC]
        # Verifies row selection by range from a sparse series, and chained selection, keeps the sparse dtype and the fill value (issue 34687).
        # GH34687
        s = Series([1.0, 0.0, 0.0, 0.0, 0.0], dtype=SparseDtype("float64", 0.0))

        result = s.loc[range(2)]
        expected = Series([1.0, 0.0], dtype=SparseDtype("float64", 0.0))
        tm.assert_series_equal(result, expected)

        result = s.loc[range(3)].loc[range(2)]
        expected = Series([1.0, 0.0], dtype=SparseDtype("float64", 0.0))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("indexer", ["loc", "iloc"])
    def test_getitem_single_row_sparse_df(self, indexer):
        # [FUNCTION DOC]
        # Verifies selecting a single row from a sparse-dtype frame with the label and position indexers gives a sparse series named by the row (issue 46406). Parametrized over the two indexer names.
        # GH#46406
        df = DataFrame([[1.0, 0.0, 1.5], [0.0, 2.0, 0.0]], dtype=SparseDtype(float))
        result = getattr(df, indexer)[0]
        expected = Series([1.0, 0.0, 1.5], dtype=SparseDtype(float), name=0)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("key_type", [iter, np.array, Series, Index])
    def test_loc_getitem_iterable(self, float_frame, key_type):
        # [FUNCTION DOC]
        # Verifies a column key given as an iterator, array, series or index of labels gives the same result as a list (uses the float frame fixture). Parametrized over four key types.
        idx = key_type(["A", "B", "C"])
        result = float_frame.loc[:, idx]
        expected = float_frame.loc[:, ["A", "B", "C"]]
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_timedelta_0seconds(self):
        # [FUNCTION DOC]
        # Verifies a string slice starting at zero seconds on a timedelta index equals slicing from the matching time delta (issue 10583).
        # GH#10583
        df = DataFrame(np.random.default_rng(2).normal(size=(10, 4)))
        df.index = timedelta_range(start="0s", periods=10, freq="s")
        expected = df.loc[Timedelta("0s") :, :]
        result = df.loc["0s":, :]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("val,expected", [(2**63 - 1, 1), (2**63, 2)])
    def test_loc_getitem_uint64_scalar(self, val, expected):
        # [FUNCTION DOC]
        # Verifies lookup of the largest signed 64-bit value and the next value (an unsigned label) in a two-row frame returns the proper row as a series named by the label (issue 19399). Parametrized over (label, expected value).
        # see GH#19399
        df = DataFrame([1, 2], index=[2**63 - 1, 2**63])
        result = df.loc[val]
        expected = Series([expected])
        expected.name = val
        tm.assert_series_equal(result, expected)

    def test_loc_setitem_int_label_with_float_index(self, float_numpy_dtype):
        # [FUNCTION DOC]
        # Verifies that setting with an integer label on a float index treats it as the label one (not a position), so the third element is replaced. Parametrized by float dtype fixture.
        # note labels are floats
        dtype = float_numpy_dtype
        ser = Series(["a", "b", "c"], index=Index([0, 0.5, 1], dtype=dtype))
        expected = ser.copy()

        ser.loc[1] = "zoo"
        expected.iloc[2] = "zoo"

        tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize(
        "indexer, expected",
        [
            # The test name is a misnomer in the 0 case as df.index[indexer]
            #  is a scalar.
            (0, [20, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            (slice(4, 8), [0, 1, 2, 3, 20, 20, 20, 20, 8, 9]),
            ([3, 5], [0, 1, 2, 20, 4, 20, 6, 7, 8, 9]),
        ],
    )
    def test_loc_setitem_listlike_with_timedelta64index(self, indexer, expected):
        # [FUNCTION DOC]
        # Verifies setting a column value through labels taken from a time delta index by a position scalar, slice or list (issue 16637). Parametrized over (indexer, expected data). The scalar case is a scalar label, not list-like, despite the test name (a comment states this).
        # GH#16637
        tdi = to_timedelta(range(10), unit="s")
        df = DataFrame({"x": range(10)}, dtype="int64", index=tdi)

        df.loc[df.index[indexer], "x"] = 20

        expected = DataFrame(
            expected,
            index=tdi,
            columns=["x"],
            dtype="int64",
        )

        tm.assert_frame_equal(expected, df)

    def test_loc_setitem_categorical_values_partial_column_slice(self):
        # [FUNCTION DOC]
        # Verifies that assigning categorical values to part of an integer column raises a type error about an invalid value.
        # KNOWN ISSUE: the raise-context contains two assignments but the first one raises, so the second assignment (to the text column) never runs and is untested.
        # Assigning a Category to parts of an int/... column uses the values of
        # the Categorical
        df = DataFrame({"a": [1, 1, 1, 1, 1], "b": list("aaaaa")})
        with pytest.raises(TypeError, match="Invalid value"):
            df.loc[1:2, "a"] = Categorical(["b", "b"], categories=["a", "b"])
            df.loc[2:3, "b"] = Categorical(["b", "b"], categories=["a", "b"])

    def test_loc_setitem_single_row_categorical(self, using_infer_string):
        # [FUNCTION DOC]
        # Verifies label-assigning a categorical column to a single-row frame (issue 25495), parametrized by the infer-string fixture. The label assignment operates in place so the column keeps its old type (object or string) with the categorical's values; plain column assignment keeps the categorical dtype.
        # GH#25495
        df = DataFrame({"Alpha": ["a"], "Numeric": [0]})
        categories = Categorical(df["Alpha"], categories=["a", "b", "c"])

        # pre-2.0 this swapped in a new array, in 2.0 it operates inplace,
        #  consistent with non-split-path
        df.loc[:, "Alpha"] = categories

        result = df["Alpha"]
        expected = Series(categories, index=df.index, name="Alpha").astype(
            object if not using_infer_string else "str"
        )
        tm.assert_series_equal(result, expected)

        # double-check that the non-loc setting retains categoricalness
        df["Alpha"] = categories
        tm.assert_series_equal(df["Alpha"], Series(categories, name="Alpha"))

    def test_loc_setitem_datetime_coercion(self):
        # [FUNCTION DOC]
        # Verifies setting a datetime64 scalar into part of a datetime column works (issue 1048), while setting a plain date object raises a type error about an invalid value.
        # GH#1048
        df = DataFrame({"c": [Timestamp("2010-10-01")] * 3})
        df.loc[0:1, "c"] = np.datetime64("2008-08-08")
        assert Timestamp("2008-08-08") == df.loc[0, "c"]
        assert Timestamp("2008-08-08") == df.loc[1, "c"]
        with pytest.raises(TypeError, match="Invalid value"):
            df.loc[2, "c"] = date(2005, 5, 5)

    @pytest.mark.parametrize("idxer", ["var", ["var"]])
    def test_loc_setitem_datetimeindex_tz(self, idxer, tz_naive_fixture):
        # [FUNCTION DOC]
        # Verifies assigning a frame of floats to a column of a timezone-aware datetime-indexed frame (issue 11365). Parametrized over a scalar column key (raises a type error about an invalid value because the target is a series-like assignment) and a list of one column key (works and equals the expected frame, see issue 56223). Uses the timezone fixture.
        # GH#11365
        tz = tz_naive_fixture
        idx = date_range(start="2015-07-12", periods=3, freq="h", tz=tz)
        expected = DataFrame(1.2, index=idx, columns=["var"])
        # if result started off with object dtype, then the .loc.__setitem__
        #  below would retain object dtype
        result = DataFrame(index=idx, columns=["var"], dtype=np.float64)
        if idxer == "var":
            with pytest.raises(TypeError, match="Invalid value"):
                result.loc[:, idxer] = expected
        else:
            # See https://github.com/pandas-dev/pandas/issues/56223
            result.loc[:, idxer] = expected
            tm.assert_frame_equal(result, expected)

    def test_loc_setitem_time_key(self):
        # [FUNCTION DOC]
        # Verifies setting and reading rows selected by exact time-of-day and by time-of-day slices on a half-hourly datetime index.
        index = date_range("2012-01-01", "2012-01-05", freq="30min")
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(index), 5)), index=index
        )
        akey = time(12, 0, 0)
        bkey = slice(time(13, 0, 0), time(14, 0, 0))
        ainds = [24, 72, 120, 168]
        binds = [26, 27, 28, 74, 75, 76, 122, 123, 124, 170, 171, 172]

        result = df.copy()
        result.loc[akey] = 0
        result = result.loc[akey]
        expected = df.loc[akey].copy()
        expected.loc[:] = 0
        tm.assert_frame_equal(result, expected)

        result = df.copy()
        result.loc[akey] = 0
        result.loc[akey] = df.iloc[ainds]
        tm.assert_frame_equal(result, df)

        result = df.copy()
        result.loc[bkey] = 0
        result = result.loc[bkey]
        expected = df.loc[bkey].copy()
        expected.loc[:] = 0
        tm.assert_frame_equal(result, expected)

        result = df.copy()
        result.loc[bkey] = 0
        result.loc[bkey] = df.iloc[binds]
        tm.assert_frame_equal(result, df)

    @pytest.mark.parametrize("key", ["A", ["A"], ("A", slice(None))])
    def test_loc_setitem_unsorted_multiindex_columns(self, key):
        # [FUNCTION DOC]
        # Verifies assigning a block of zeros to columns selected by the first level of an unsorted two-level column index (issue 38601). Parametrized over key forms: a single label, a list with one label, and a tuple of label and full slice. The columns with that first label are zeroed; the same holds after sorting the columns.
        # GH#38601
        mi = MultiIndex.from_tuples([("A", 4), ("B", "3"), ("A", "2")])
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=mi)
        obj = df.copy()
        obj.loc[:, key] = np.zeros((2, 2), dtype="int64")
        expected = DataFrame([[0, 2, 0], [0, 5, 0]], columns=mi)
        tm.assert_frame_equal(obj, expected)

        df = df.sort_index(axis=1)
        df.loc[:, key] = np.zeros((2, 2), dtype="int64")
        expected = expected.sort_index(axis=1)
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_uint_drop(self, any_int_numpy_dtype):
        # [FUNCTION DOC]
        # Verifies assigning an in-range integer to an existing label of an integer series keeps the original integer dtype (issue 18311). Parametrized over all integer numpy dtypes via the fixture.
        # see GH#18311
        # assigning series.loc[0] = 4 changed series.dtype to int
        series = Series([1, 2, 3], dtype=any_int_numpy_dtype)
        series.loc[0] = 4
        expected = Series([4, 2, 3], dtype=any_int_numpy_dtype)
        tm.assert_series_equal(series, expected)

    def test_loc_setitem_td64_non_nano(self):
        # [FUNCTION DOC]
        # Verifies setting a non-nanosecond time delta value into several elements of a minutes-unit series gives the same result as setting a time delta object (issue 14155).
        # GH#14155
        ser = Series(10 * [np.timedelta64(10, "m")])
        ser.loc[[1, 2, 3]] = np.timedelta64(20, "m")
        expected = Series(10 * [np.timedelta64(10, "m")])
        expected.loc[[1, 2, 3]] = Timedelta(np.timedelta64(20, "m"))
        tm.assert_series_equal(ser, expected)

    def test_loc_setitem_2d_to_1d_raises(self):
        # [FUNCTION DOC]
        # Verifies assigning a 2D array to a float series through a list of labels or the full slice raises a value error about setting an array element with a sequence.
        data = np.random.default_rng(2).standard_normal((2, 2))
        # float64 dtype to avoid upcast when trying to set float data
        ser = Series(range(2), dtype="float64")

        msg = "setting an array element with a sequence."
        with pytest.raises(ValueError, match=msg):
            ser.loc[range(2)] = data

        with pytest.raises(ValueError, match=msg):
            ser.loc[:] = data

    def test_loc_getitem_interval_index(self):
        # [FUNCTION DOC]
        # Verifies a scalar inside an interval selects the row containing it from a right-closed interval index (issue 19977).
        # GH#19977
        index = pd.interval_range(start=0, periods=3)
        df = DataFrame(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=index, columns=["A", "B", "C"]
        )

        expected = 1
        result = df.loc[0.5, "A"]
        tm.assert_almost_equal(result, expected)

    def test_loc_getitem_interval_index2(self):
        # [FUNCTION DOC]
        # Verifies a point on a boundary of a both-closed interval index selects two rows (issue 19977): the result is a series over the two intervals.
        # GH#19977
        index = pd.interval_range(start=0, periods=3, closed="both")
        df = DataFrame(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=index, columns=["A", "B", "C"]
        )

        index_exp = pd.interval_range(start=0, periods=2, freq=1, closed="both")
        expected = Series([1, 4], index=index_exp, name="A")
        result = df.loc[1, "A"]
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("tpl", [(1,), (1, 2)])
    def test_loc_getitem_index_single_double_tuples(self, tpl):
        # [FUNCTION DOC]
        # Verifies that tuples of length one or two stored as plain labels (without splitting into levels) can be looked up in a list (issue 20991). Parametrized over the two tuples.
        # GH#20991
        idx = Index(
            [(1,), (1, 2)],
            name="A",
            tupleize_cols=False,
        )
        df = DataFrame(index=idx)

        result = df.loc[[tpl]]
        idx = Index([tpl], name="A", tupleize_cols=False)
        expected = DataFrame(index=idx)
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_index_namedtuple(self):
        # [FUNCTION DOC]
        # Verifies lookup by a named tuple on an index of named tuples returns the matching row.
        IndexType = namedtuple("IndexType", ["a", "b"])
        idx1 = IndexType("foo", "bar")
        idx2 = IndexType("baz", "bof")
        index = Index([idx1, idx2], name="composite_index", tupleize_cols=False)
        df = DataFrame([(1, 2), (3, 4)], index=index, columns=["A", "B"])

        result = df.loc[IndexType("foo", "bar")]["A"]
        assert result == 1

    def test_loc_setitem_single_column_mixed(self, using_infer_string):
        # [FUNCTION DOC]
        # Verifies setting missing values via label on every other row of a text column in a mixed frame gives the expected values; the dtype is object or text depending on the infer-string fixture.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)),
            index=["a", "b", "c", "d", "e"],
            columns=["foo", "bar", "baz"],
        )
        df["str"] = "qux"
        df.loc[df.index[::2], "str"] = np.nan
        expected = Series(
            [np.nan, "qux", np.nan, "qux", np.nan],
            dtype=object if not using_infer_string else "str",
        ).values
        tm.assert_almost_equal(df["str"].values, expected)

    def test_loc_setitem_cast2(self):
        # [FUNCTION DOC]
        # Verifies that assigning text into an all-missing float column cell raises a type error about an invalid value (issue 7704).
        # GH#7704
        # dtype conversion on setting
        df = DataFrame(np.random.default_rng(2).random((30, 3)), columns=tuple("ABC"))
        df["event"] = np.nan
        with pytest.raises(TypeError, match="Invalid value"):
            df.loc[10, "event"] = "foo"

    def test_loc_setitem_cast3(self):
        # [FUNCTION DOC]
        # Verifies that setting an in-range value keeps the 8-bit integer dtype, also with a plain attribute assignment of a numpy 8-bit integer (issue 5782).
        # Test that data type is preserved . GH#5782
        df = DataFrame({"one": np.arange(6, dtype=np.int8)})
        df.loc[1, "one"] = 6
        assert df.dtypes.one == np.dtype(np.int8)
        df.one = np.int8(7)
        assert df.dtypes.one == np.dtype(np.int8)

    def test_loc_setitem_range_key(self, frame_or_series):
        # [FUNCTION DOC]
        # Verifies that a range object key is treated as labels, not positions (issue 45479), for both frame and series via the fixture: on an unordered integer index, the rows labelled zero to two are replaced by the given values.
        # GH#45479 don't treat range key as positional
        obj = frame_or_series(range(5), index=[3, 4, 1, 0, 2])

        values = [9, 10, 11]
        if obj.ndim == 2:
            values = [[9], [10], [11]]

        obj.loc[range(3)] = values

        expected = frame_or_series([0, 1, 10, 9, 11], index=obj.index)
        tm.assert_equal(obj, expected)

    def test_loc_setitem_numpy_frame_categorical_value(self):
        # [FUNCTION DOC]
        # Verifies assigning a categorical of integers to a slice of an integer column sets the values (issue 52927).
        # GH#52927
        df = DataFrame({"a": [1, 1, 1, 1, 1], "b": ["a", "a", "a", "a", "a"]})
        df.loc[1:2, "a"] = Categorical([2, 2], categories=[1, 2])

        expected = DataFrame({"a": [1, 2, 2, 1, 1], "b": ["a", "a", "a", "a", "a"]})
        tm.assert_frame_equal(df, expected)

    def test_loc_with_nat_in_tzaware_index(self):
        # [FUNCTION DOC]
        # Verifies lookups on a timezone-aware index that contains not-a-time (issue 54409): a mixed lookup returns the values for each key, and a lookup of only not-a-time returns the value of the missing entry instead of raising.
        # GH#54409
        timestamp = to_datetime("2023-01-01", utc=True)
        df = DataFrame(
            {
                "index": Series([pd.NaT, timestamp]),
                "value": Series([0, 1]),
            }
        ).set_index("index")

        # Works fine when mixing NaT and valid values
        result = df.loc[
            Series([pd.NaT, timestamp, timestamp], dtype=df.index.dtype),
            "value",
        ]
        expected = [0, 1, 1]
        assert result.tolist() == expected

        # Regression check: all-NaT lookup should return [0], not raise
        result = df.loc[
            Series([pd.NaT], dtype=df.index.dtype),
            "value",
        ]
        assert result.tolist() == [0]


class TestLocWithEllipsis:
    # [CLASS DOC]
    # Tests for ellipsis keys with the label and position indexers.
    @pytest.fixture
    def indexer(self, indexer_li):
        # [FUNCTION DOC]
        # Fixture returning the indexer wrapper provided by the label-and-position indexer fixture (so the position indexer is tested too).
        # Test iloc while we're here
        return indexer_li

    @pytest.fixture
    def obj(self, series_with_simple_index, frame_or_series):
        # [FUNCTION DOC]
        # Fixture returning the series with a simple index from the series fixture, converted to a frame when the frame-or-series fixture asks for a frame.
        obj = series_with_simple_index
        if frame_or_series is not Series:
            obj = obj.to_frame()
        return obj

    def test_loc_iloc_getitem_ellipsis(self, obj, indexer):
        # [FUNCTION DOC]
        # Verifies a lone ellipsis returns an object equal to the original.
        result = indexer(obj)[...]
        tm.assert_equal(result, obj)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_loc_iloc_getitem_leading_ellipses(self, series_with_simple_index, indexer):
        # [FUNCTION DOC]
        # Verifies an ellipsis followed by a list key behaves like the list key alone, for series and for frames (ellipsis, list equals full slice, list). Ignores a deprecation warning for the business-day period dtype.
        # Special cases: boolean-index series are skipped for the label indexer because the key would be taken as a mask; multi-level indexes with the label indexer raise a not-implemented error saying ellipsis is not supported; empty objects skip the series check.
        obj = series_with_simple_index
        key = 0 if (indexer is tm.iloc or len(obj) == 0) else obj.index[0]

        if indexer is tm.loc and obj.index.inferred_type == "boolean":
            # passing [False] will get interpreted as a boolean mask
            # TODO: should it?  unambiguous when lengths dont match?
            return
        if indexer is tm.loc and isinstance(obj.index, MultiIndex):
            msg = "MultiIndex does not support indexing with Ellipsis"
            with pytest.raises(NotImplementedError, match=msg):
                result = indexer(obj)[..., [key]]

        elif len(obj) != 0:
            result = indexer(obj)[..., [key]]
            expected = indexer(obj)[[key]]
            tm.assert_series_equal(result, expected)

        key2 = 0 if indexer is tm.iloc else obj.name
        df = obj.to_frame()
        result = indexer(df)[..., [key2]]
        expected = indexer(df)[:, [key2]]
        tm.assert_frame_equal(result, expected)

    def test_loc_iloc_getitem_ellipses_only_one_ellipsis(self, obj, indexer):
        # [FUNCTION DOC]
        # Verifies that more than one ellipsis raises an indexing error with the single-ellipsis message (issue 37750), whenever the first key is an ellipsis; if the first key is not, the "too many indexers" error comes first.
        # GH37750
        key = 0 if (indexer is tm.iloc or len(obj) == 0) else obj.index[0]

        with pytest.raises(IndexingError, match=_one_ellipsis_message):
            indexer(obj)[..., ...]

        with pytest.raises(IndexingError, match=_one_ellipsis_message):
            indexer(obj)[..., [key], ...]

        with pytest.raises(IndexingError, match=_one_ellipsis_message):
            indexer(obj)[..., ..., key]

        # one_ellipsis_message takes precedence over "Too many indexers"
        #  only when the first key is Ellipsis
        with pytest.raises(IndexingError, match="Too many indexers"):
            indexer(obj)[key, ..., ...]


class TestLocWithMultiIndex:
    # [CLASS DOC]
    # Tests for the label indexer with multi-level indexes.
    @pytest.mark.parametrize(
        "keys, expected",
        [
            (["b", "a"], [["b", "b", "a", "a"], [1, 2, 1, 2]]),
            (["a", "b"], [["a", "a", "b", "b"], [1, 2, 1, 2]]),
            ((["a", "b"], [1, 2]), [["a", "a", "b", "b"], [1, 2, 1, 2]]),
            ((["a", "b"], [2, 1]), [["a", "a", "b", "b"], [2, 1, 2, 1]]),
            ((["b", "a"], [2, 1]), [["b", "b", "a", "a"], [2, 1, 2, 1]]),
            ((["b", "a"], [1, 2]), [["b", "b", "a", "a"], [1, 2, 1, 2]]),
            ((["c", "a"], [2, 1]), [["c", "a", "a"], [1, 2, 1]]),
        ],
    )
    @pytest.mark.parametrize("dim", ["index", "columns"])
    def test_loc_getitem_multilevel_index_order(self, dim, keys, expected):
        # [FUNCTION DOC]
        # Verifies the order of keys is respected for multi-level index selection (issue 22797). Parametrized over key lists/tuples with expected resulting index arrays (missing keys dropped) and over whether the index or the columns are multi-level.
        # GH#22797
        # Try to respect order of keys given for MultiIndex.loc
        kwargs = {dim: [["c", "a", "a", "b", "b"], [1, 1, 2, 1, 2]]}
        df = DataFrame(np.arange(25).reshape(5, 5), **kwargs)
        exp_index = MultiIndex.from_arrays(expected)
        if dim == "index":
            res = df.loc[keys, :]
            tm.assert_index_equal(res.index, exp_index)
        elif dim == "columns":
            res = df.loc[:, keys]
            tm.assert_index_equal(res.columns, exp_index)

    def test_loc_preserve_names(self, multiindex_year_month_day_dataframe_random_data):
        # [FUNCTION DOC]
        # Verifies that selecting by the first level (and by two levels) from a year/month/day indexed frame and a column of it keeps the remaining level names (uses the multi-index fixture).
        ymd = multiindex_year_month_day_dataframe_random_data

        result = ymd.loc[2000]
        result2 = ymd["A"].loc[2000]
        assert result.index.names == ymd.index.names[1:]
        assert result2.index.names == ymd.index.names[1:]

        result = ymd.loc[2000, 2]
        result2 = ymd["A"].loc[2000, 2]
        assert result.index.name == ymd.index.names[2]
        assert result2.index.name == ymd.index.names[2]

    def test_loc_getitem_multiindex_nonunique_len_zero(self):
        # [FUNCTION DOC]
        # Verifies an empty list or an empty series key on a series with a non-unique two-level index returns an empty series equal to the empty slice (issue 13691).
        # GH#13691
        mi = MultiIndex.from_product([[0], [1, 1]])
        ser = Series(0, index=mi)

        res = ser.loc[[]]

        expected = ser[:0]
        tm.assert_series_equal(res, expected)

        res2 = ser.loc[ser.iloc[0:0]]
        tm.assert_series_equal(res2, expected)

    def test_loc_getitem_access_none_value_in_multiindex(self):
        # [FUNCTION DOC]
        # Verifies a none value (or an integer in an object series) can be read through a full tuple key on a multi-level index and is returned as is (issue 34318).
        # GH#34318: test that you can access a None value using .loc
        #  through a Multiindex

        ser = Series([None], MultiIndex.from_arrays([["Level1"], ["Level2"]]))
        result = ser.loc[("Level1", "Level2")]
        assert result is None

        midx = MultiIndex.from_product([["Level1"], ["Level2_a", "Level2_b"]])
        ser = Series([None] * len(midx), dtype=object, index=midx)
        result = ser.loc[("Level1", "Level2_a")]
        assert result is None

        ser = Series([1] * len(midx), dtype=object, index=midx)
        result = ser.loc[("Level1", "Level2_a")]
        assert result == 1

    def test_loc_setitem_multiindex_slice(self):
        # [FUNCTION DOC]
        # Verifies setting a value through a tuple-bounded slice of a multi-level index sets the covered entries (issue 34870).
        # GH 34870

        index = MultiIndex.from_tuples(
            zip(
                ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
                ["one", "two", "one", "two", "one", "two", "one", "two"],
            ),
            names=["first", "second"],
        )

        result = Series([1, 1, 1, 1, 1, 1, 1, 1], index=index)
        result.loc[("baz", "one") : ("foo", "two")] = 100

        expected = Series([1, 1, 100, 100, 100, 100, 1, 1], index=index)

        tm.assert_series_equal(result, expected)

    def test_loc_getitem_slice_datetime_objs_with_datetimeindex(self):
        # [FUNCTION DOC]
        # Verifies a slice with Python datetime bounds far outside the range returns the whole 100000-element series.
        times = date_range("2000-01-01", freq="10min", periods=100000)
        ser = Series(range(100000), times)
        result = ser.loc[datetime(1900, 1, 1) : datetime(2100, 1, 1)]
        tm.assert_series_equal(result, ser)

    def test_loc_getitem_datetime_string_with_datetimeindex(self):
        # [FUNCTION DOC]
        # Verifies label selection with a list of date strings and column names returns rows with a nanosecond datetime index (issue 16710).
        # GH 16710
        df = DataFrame(
            {"a": range(10), "b": range(10)},
            index=date_range("2010-01-01", "2010-01-10", unit="ns"),
        )
        result = df.loc[["2010-01-01", "2010-01-05"], ["a", "b"]]
        expected = DataFrame(
            {"a": [0, 4], "b": [0, 4]},
            index=DatetimeIndex(["2010-01-01", "2010-01-05"]).as_unit("ns"),
        )
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_sorted_index_level_with_duplicates(self):
        # [FUNCTION DOC]
        # Verifies selecting a duplicated key tuple from a sorted multi-level index with duplicates returns all matching rows (issue 4516).
        # GH#4516 sorting a MultiIndex with duplicates and multiple dtypes
        mi = MultiIndex.from_tuples(
            [
                ("foo", "bar"),
                ("foo", "bar"),
                ("bah", "bam"),
                ("bah", "bam"),
                ("foo", "bar"),
                ("bah", "bam"),
            ],
            names=["A", "B"],
        )
        df = DataFrame(
            [
                [1.0, 1],
                [2.0, 2],
                [3.0, 3],
                [4.0, 4],
                [5.0, 5],
                [6.0, 6],
            ],
            index=mi,
            columns=["C", "D"],
        )
        df = df.sort_index(level=0)

        expected = DataFrame(
            [[1.0, 1], [2.0, 2], [5.0, 5]], columns=["C", "D"], index=mi.take([0, 1, 4])
        )

        result = df.loc[("foo", "bar")]
        tm.assert_frame_equal(result, expected)

    def test_additional_element_to_categorical_series_loc(self):
        # [FUNCTION DOC]
        # Verifies enlarging a categorical series with a value not among the categories converts it to object (issue 47677).
        # GH#47677
        result = Series(["a", "b", "c"], dtype="category")
        result.loc[3] = 0
        expected = Series(["a", "b", "c", 0], dtype="object")
        tm.assert_series_equal(result, expected)

    def test_additional_categorical_element_loc(self):
        # [FUNCTION DOC]
        # Verifies enlarging a categorical series with an existing category keeps the categorical dtype (issue 47677).
        # GH#47677
        result = Series(["a", "b", "c"], dtype="category")
        result.loc[3] = "a"
        expected = Series(["a", "b", "c", "a"], dtype="category")
        tm.assert_series_equal(result, expected)

    def test_loc_set_nan_in_categorical_series(self, any_numeric_ea_dtype):
        # [FUNCTION DOC]
        # Verifies that enlarging a categorical series with a missing value and then setting a missing value into an existing element both keep the categorical dtype (issue 47677). Parametrized over numeric extension array dtypes for the categories.
        # GH#47677
        srs = Series(
            [1, 2, 3],
            dtype=CategoricalDtype(Index([1, 2, 3], dtype=any_numeric_ea_dtype)),
        )
        # enlarge
        srs.loc[3] = np.nan
        expected = Series(
            [1, 2, 3, np.nan],
            dtype=CategoricalDtype(Index([1, 2, 3], dtype=any_numeric_ea_dtype)),
        )
        tm.assert_series_equal(srs, expected)
        # set into
        srs.loc[1] = np.nan
        expected = Series(
            [1, np.nan, 3, np.nan],
            dtype=CategoricalDtype(Index([1, 2, 3], dtype=any_numeric_ea_dtype)),
        )
        tm.assert_series_equal(srs, expected)

    @pytest.mark.parametrize("na", (np.nan, pd.NA, None, pd.NaT))
    def test_loc_consistency_series_enlarge_set_into(self, na):
        # [FUNCTION DOC]
        # Verifies enlarging a categorical series with a missing marker and setting the marker into an existing element give the same result, with a missing value in the categorical (issue 47677). Parametrized over four missing markers (nan, missing-array marker, none and not-a-time).
        # GH#47677
        srs_enlarge = Series(["a", "b", "c"], dtype="category")
        srs_enlarge.loc[3] = na

        srs_setinto = Series(["a", "b", "c", "a"], dtype="category")
        srs_setinto.loc[3] = na

        tm.assert_series_equal(srs_enlarge, srs_setinto)
        expected = Series(["a", "b", "c", na], dtype="category")
        tm.assert_series_equal(srs_enlarge, expected)

    def test_loc_getitem_preserves_index_level_category_dtype(self):
        # [FUNCTION DOC]
        # Verifies that selecting rows by a list of labels keeps the categorical dtype of the first level of the index including all categories (issue 15166).
        # GH#15166
        df = DataFrame(
            data=np.arange(2, 22, 2),
            index=MultiIndex(
                levels=[CategoricalIndex(["a", "b"]), range(10)],
                codes=[[0] * 5 + [1] * 5, range(10)],
                names=["Index1", "Index2"],
            ),
        )

        expected = CategoricalIndex(
            ["a", "b"],
            categories=["a", "b"],
            ordered=False,
            name="Index1",
            dtype="category",
        )

        result = df.index.levels[0]
        tm.assert_index_equal(result, expected)

        result = df.loc[["a"]].index.levels[0]
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("lt_value", [30, 10])
    def test_loc_multiindex_levels_contain_values_not_in_index_anymore(self, lt_value):
        # [FUNCTION DOC]
        # Verifies selecting a label that was filtered out of the index but still lives in the levels raises a key error naming the label as not in the index (issue 41170). Parametrized over two threshold values.
        # GH#41170
        df = DataFrame({"a": [12, 23, 34, 45]}, index=[list("aabb"), [0, 1, 2, 3]])
        with pytest.raises(KeyError, match=r"\['b'\] not in index"):
            df.loc[df["a"] < lt_value, :].loc[["b"], :]

    def test_loc_multiindex_null_slice_na_level(self):
        # [FUNCTION DOC]
        # Verifies a null-slice-and-label key on a multi-level index with a missing first level returns the element with a missing-value index (issue 42055); a note asks whether the result should carry a name.
        # GH#42055
        lev1 = np.array([np.nan, np.nan])
        lev2 = ["bar", "baz"]
        mi = MultiIndex.from_arrays([lev1, lev2])
        ser = Series([0, 1], index=mi)
        result = ser.loc[:, "bar"]

        # TODO: should we have name="bar"?
        expected = Series([0], index=[np.nan])
        tm.assert_series_equal(result, expected)

    def test_loc_drops_level(self):
        # [FUNCTION DOC]
        # Verifies selecting by a first-level label and full slices drops the first level of the index (the result index equals the original with the first level dropped for the first four entries).
        # Based on test_series_varied_multiindex_alignment, where
        #  this used to fail to drop the first level
        mi = MultiIndex.from_product(
            [list("ab"), list("xy"), [1, 2]], names=["ab", "xy", "num"]
        )
        ser = Series(range(8), index=mi)

        loc_result = ser.loc["a", :, :]
        expected = ser.index.droplevel(0)[:4]
        tm.assert_index_equal(loc_result.index, expected)


class TestLocSetitemWithExpansion:
    # [CLASS DOC]
    # Tests for enlarging objects by label assignment, including extension index dtypes.
    @td.skip_if_no("pyarrow")
    def test_loc_setitem_with_expansion_preserves_ea_dtype(self):
        # [FUNCTION DOC]
        # Verifies enlargement keeps the extension dtype of the index (issue 41626). Skipped if pyarrow is not installed. Setup: a one-row frame with an Arrow date32 index. Label-assigning a row with a new date appends it and the index equals the old plus the new date with the same extension dtype; the same holds when enlarging a series made from the column without its last row.
        # GH#41626 retain index.dtype in setitem-with-expansion
        idx = Index([Timestamp(0).date()], dtype="date32[pyarrow]")
        df = DataFrame({"A": range(1)}, index=idx)
        item = Timestamp("1970-01-02").date()

        df.loc[item] = 1

        exp_index = Index([idx[0], item], dtype=idx.dtype)
        tm.assert_index_equal(df.index, exp_index)

        ser = df["A"].iloc[:-1]
        ser.loc[item] = 1
        tm.assert_index_equal(ser.index, exp_index)

    def test_loc_setitem_with_expansion_large_dataframe(self, monkeypatch):
        # [FUNCTION DOC]
        # Verifies enlarging a large frame by label works even when the index engine switches strategy at a size cutoff (issue 10692). Uses the monkeypatch fixture to lower the engine's size cutoff to fifty, builds a fifty-row integer frame and sets a row at label fifty; expected equal to a fifty-one row frame.
        # GH#10692
        size_cutoff = 50
        with monkeypatch.context():
            monkeypatch.setattr(libindex, "_SIZE_CUTOFF", size_cutoff)
            result = DataFrame({"x": range(size_cutoff)}, dtype="int64")
            result.loc[size_cutoff] = size_cutoff
        expected = DataFrame({"x": range(size_cutoff + 1)}, dtype="int64")
        tm.assert_frame_equal(result, expected)

    def test_loc_setitem_empty_series(self):
        # [FUNCTION DOC]
        # Verifies enlarging an empty object series with integer labels twice gives an integer series with those labels (issue 5226).
        # GH#5226

        # partially set with an empty object series
        ser = Series(dtype=object)
        ser.loc[1] = 1
        tm.assert_series_equal(ser, Series([1], index=range(1, 2)))
        ser.loc[3] = 3
        tm.assert_series_equal(ser, Series([1, 3], index=[1, 3]))

    def test_loc_setitem_empty_series_float(self):
        # [FUNCTION DOC]
        # Same as above with float values: the result keeps float values and the labels one and three (issue 5226).
        # GH#5226

        # partially set with an empty object series
        ser = Series(dtype=object)
        ser.loc[1] = 1.0
        tm.assert_series_equal(ser, Series([1.0], index=range(1, 2)))
        ser.loc[3] = 3.0
        tm.assert_series_equal(ser, Series([1.0, 3.0], index=[1, 3]))

    def test_loc_setitem_empty_series_str_idx(self):
        # [FUNCTION DOC]
        # Verifies enlarging an empty series with string labels, then an integer label, gives a mixed object index (issue 5226).
        # GH#5226

        # partially set with an empty object series
        ser = Series(dtype=object)
        ser.loc["foo"] = 1
        tm.assert_series_equal(ser, Series([1], index=Index(["foo"])))
        ser.loc["bar"] = 3
        tm.assert_series_equal(ser, Series([1, 3], index=Index(["foo", "bar"])))
        ser.loc[3] = 4
        tm.assert_series_equal(ser, Series([1, 3, 4], index=Index(["foo", "bar", 3])))

    def test_loc_setitem_incremental_with_dst(self):
        # [FUNCTION DOC]
        # Verifies enlarging a series with consecutive timestamps across a daylight saving boundary in the US Pacific zone, one every fifteen minutes, setting each label (issue 20724); the result equals a series of ones with those timestamps.
        # GH#20724
        base = datetime(2015, 11, 1, tzinfo=gettz("US/Pacific"))
        idxs = [base + timedelta(seconds=i * 900) for i in range(16)]
        result = Series([0], index=[idxs[0]])
        for ts in idxs:
            result.loc[ts] = 1
        expected = Series(1, index=idxs)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "conv",
        [
            lambda x: x,
            lambda x: x.to_datetime64(),
            lambda x: x.to_pydatetime(),
            lambda x: np.datetime64(x),
        ],
        ids=["self", "to_datetime64", "to_pydatetime", "np.datetime64"],
    )
    def test_loc_setitem_datetime_keys_cast(self, conv, using_infer_string):
        # [FUNCTION DOC]
        # Verifies enlarging an empty frame with datetime keys in four forms (timestamp, numpy datetime64 value, Python datetime, numpy datetime) (issues 9516 and 51363). Expected: a float column with an index that is not cast to a datetime index (object dtype, or the inferred dtype when string inference is on). Parametrized over the conversion and the infer-string fixture.
        # GH#9516, GH#51363 changed in 3.0 to not cast on Index.insert
        dt1 = Timestamp("20130101 09:00:00")
        dt2 = Timestamp("20130101 10:00:00")
        df = DataFrame()
        df.loc[conv(dt1), "one"] = 100
        df.loc[conv(dt2), "one"] = 200

        expected = DataFrame(
            {"one": [100.0, 200.0]},
            index=Index(
                [conv(dt1), conv(dt2)], dtype=None if using_infer_string else object
            ),
            columns=Index(["one"]),
        )
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_categorical_column_retains_dtype(self, ordered):
        # [FUNCTION DOC]
        # Verifies assigning a categorical (ordered or not, via the ordered fixture) to a new column by label gives a column with that categorical dtype (issue 16360).
        # GH16360
        result = DataFrame({"A": [1]})
        result.loc[:, "B"] = Categorical(["b"], ordered=ordered)
        expected = DataFrame({"A": [1], "B": Categorical(["b"], ordered=ordered)})
        tm.assert_frame_equal(result, expected)

    def test_loc_setitem_with_expansion_and_existing_dst(self):
        # [FUNCTION DOC]
        # Verifies enlarging a timezone-aware datetime-indexed frame with a timestamp outside the index, where the index covers a daylight saving transition (issue 18308). The new entry is appended with an object-dtype value column; the key is checked to be absent from the index.
        # GH#18308
        start = Timestamp("2017-10-29 00:00:00+0200", tz="Europe/Madrid")
        end = Timestamp("2017-10-29 03:00:00+0100", tz="Europe/Madrid")
        ts = Timestamp("2016-10-10 03:00:00", tz="Europe/Madrid")
        idx = date_range(start, end, inclusive="left", freq="h")
        assert ts not in idx  # i.e. result.loc setitem is with-expansion

        result = DataFrame(index=idx, columns=["value"])
        result.loc[ts, "value"] = 12
        expected = DataFrame(
            [np.nan] * len(idx) + [12],
            index=idx.append(DatetimeIndex([ts])),
            columns=["value"],
            dtype=object,
        )
        tm.assert_frame_equal(result, expected)

    def test_setitem_with_expansion(self):
        # [FUNCTION DOC]
        # Verifies setting an element or part of a column with timezone-aware times of a different timezone (and adding a time delta) retains the dtype and values.
        # Steps: a frame with a time column localized to UTC; a converted selection in the US Pacific zone assigned back to the matching rows keeps UTC values; then adding one second to the matching rows via the label setter updates them (before 2.0 mixing time zones gave object).
        # indexing - setting an element
        df = DataFrame(
            data=to_datetime(["2015-03-30 20:12:32", "2015-03-12 00:11:11"]),
            columns=["time"],
        )
        df["new_col"] = ["new", "old"]
        df.time = df.set_index("time").index.tz_localize("UTC")
        v = df[df.new_col == "new"].set_index("time").index.tz_convert("US/Pacific")

        # pre-2.0  trying to set a single element on a part of a different
        #  timezone converted to object; in 2.0 it retains dtype
        df2 = df.copy()
        df2.loc[df2.new_col == "new", "time"] = v

        expected = Series([v[0].tz_convert("UTC"), df.loc[1, "time"]], name="time")
        tm.assert_series_equal(df2.time, expected)

        v = df.loc[df.new_col == "new", "time"] + Timedelta("1s").as_unit("s")
        df.loc[df.new_col == "new", "time"] = v
        tm.assert_series_equal(df.loc[df.new_col == "new", "time"], v)

    def test_loc_setitem_with_expansion_inf_upcast_empty(self):
        # [FUNCTION DOC]
        # Verifies that adding a column labelled infinity to an empty frame via label assignment upcasts the column index to float, containing zero, one and infinity.
        # Test with np.inf in columns
        df = DataFrame()
        df.loc[0, 0] = 1
        df.loc[1, 1] = 2
        df.loc[0, np.inf] = 3

        result = df.columns
        expected = Index([0, 1, np.inf], dtype=np.float64)
        tm.assert_index_equal(result, expected)

    @pytest.mark.filterwarnings("ignore:indexing past lexsort depth")
    @pytest.mark.parametrize("has_ref", [True, False])
    def test_loc_setitem_with_expansion_nonunique_index(self, index, has_ref):
        # [FUNCTION DOC]
        # Verifies enlargement on non-unique indexes of many types (issue 40096). Parametrized by the index fixture and by whether another object references the data; skipped for the empty index; ignores a warning about indexing past lexsort depth.
        # Setup: each index is repeated to be non-unique; the key is a new text label. A new row with an existing column (frame, results become float), a new row on a series (integer dtype retained), and a new row with a new column are checked against expected frames with the extended index.
        # GH#40096
        if not len(index):
            pytest.skip("Not relevant for empty Index")

        index = index.repeat(2)  # ensure non-unique
        N = len(index)
        arr = np.arange(N).astype(np.int64)

        orig = DataFrame(arr, index=index)

        # key that will requiring object-dtype casting in the index
        key = "kapow"
        assert key not in index  # otherwise test is invalid
        # TODO: using a tuple key breaks here in many cases

        exp_index = index.insert(len(index), key)
        if isinstance(index, MultiIndex):
            assert exp_index[-1][0] == key
        else:
            assert exp_index[-1] == key
        exp_data = np.arange(N + 1).astype(np.float64)
        expected = DataFrame(exp_data, index=exp_index)

        # Add new row, but no new columns
        df = orig.copy()
        if has_ref:
            view = df[:]
        df.loc[key, 0] = N
        tm.assert_frame_equal(df, expected)

        # add new row on a Series
        ser = orig.copy()[0]
        if has_ref:
            view = ser[:]
        ser.loc[key] = N
        # the series machinery lets us preserve int dtype instead of float
        expected = expected[0].astype(np.int64)
        tm.assert_series_equal(ser, expected)

        # add new row and new column
        df = orig.copy()
        if has_ref:
            view = df[:]  # noqa: F841
        df.loc[key, 1] = N
        expected = DataFrame(
            {0: [*list(arr), np.nan], 1: [np.nan] * N + [float(N)]},
            index=exp_index,
        )
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_with_expansion_preserves_nullable_int(
        self, any_numeric_ea_dtype
    ):
        # [FUNCTION DOC]
        # Verifies enlargement into an empty frame with a nullable numeric series, given as series and as raw values, keeps the extension dtype (issue 42099). Parametrized by numeric extension dtype fixture; column index type not compared.
        # GH#42099
        ser = Series([0, 1, 2, 3], dtype=any_numeric_ea_dtype)
        df = DataFrame({"data": ser})

        result = DataFrame(index=df.index)
        result.loc[df.index, "data"] = ser

        tm.assert_frame_equal(result, df, check_column_type=False)

        result = DataFrame(index=df.index)
        result.loc[df.index, "data"] = ser._values
        tm.assert_frame_equal(result, df, check_column_type=False)

    def test_loc_setitem_ea_not_full_column(self):
        # [FUNCTION DOC]
        # Verifies assigning timezone-aware datetime values to a subset of rows of a new column keeps the datetime dtype with missing values for the other rows (issue 39163).
        # GH#39163
        df = DataFrame({"A": range(5)})

        val = date_range("2016-01-01", periods=3, tz="US/Pacific")

        df.loc[[0, 1, 2], "B"] = val

        bex = val.append(DatetimeIndex([pd.NaT, pd.NaT], dtype=val.dtype))
        expected = DataFrame({"A": range(5), "B": bex})
        assert expected.dtypes["B"] == val.dtype
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_with_expansion_new_row_and_new_columns(self):
        # [FUNCTION DOC]
        # Verifies setting a scalar for a new row label and two new column labels in one assignment adds both columns and the row, with float dtype and missing values elsewhere (issue 58316).
        # GH#58316
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        df.loc["x", ["C", "D"]] = 91
        expected = DataFrame(
            {
                "A": [1.0, 2.0, 3.0, np.nan],
                "B": [4.0, 5.0, 6.0, np.nan],
                "C": [np.nan, np.nan, np.nan, 91.0],
                "D": [np.nan, np.nan, np.nan, 91.0],
            },
            index=Index([0, 1, 2, "x"]),
        )
        tm.assert_frame_equal(df, expected)


class TestLocCallable:
    # [CLASS DOC]
    # Tests for callables as label indexers (issue 11485): the callable receives the object and returns a mask, label, list of labels or scalar.
    def test_frame_loc_getitem_callable(self):
        # [FUNCTION DOC]
        # Verifies callable keys on rows and columns return the same as the equivalent direct keys: boolean row mask callables, boolean column masks, a callable returning a single column label or a list, and a pair of scalar callables.
        # GH#11485
        df = DataFrame({"A": [1, 2, 3, 4], "B": list("aabb"), "C": [1, 2, 3, 4]})
        # iloc cannot use boolean Series (see GH3635)

        # return bool indexer
        res = df.loc[lambda x: x.A > 2]
        tm.assert_frame_equal(res, df.loc[df.A > 2])

        res = df.loc[lambda x: x.B == "b", :]
        tm.assert_frame_equal(res, df.loc[df.B == "b", :])

        res = df.loc[lambda x: x.A > 2, lambda x: x.columns == "B"]
        tm.assert_frame_equal(res, df.loc[df.A > 2, [False, True, False]])

        res = df.loc[lambda x: x.A > 2, lambda x: "B"]
        tm.assert_series_equal(res, df.loc[df.A > 2, "B"])

        res = df.loc[lambda x: x.A > 2, lambda x: ["A", "B"]]
        tm.assert_frame_equal(res, df.loc[df.A > 2, ["A", "B"]])

        res = df.loc[lambda x: x.A == 2, lambda x: ["A", "B"]]
        tm.assert_frame_equal(res, df.loc[df.A == 2, ["A", "B"]])

        # scalar
        res = df.loc[lambda x: 1, lambda x: "A"]
        assert res == df.loc[1, "A"]

    def test_frame_loc_getitem_callable_mixture(self):
        # [FUNCTION DOC]
        # Verifies mixing a callable with a non-callable key in either position, including a callable returning a list of column labels for a scalar row label.
        # GH#11485
        df = DataFrame({"A": [1, 2, 3, 4], "B": list("aabb"), "C": [1, 2, 3, 4]})

        res = df.loc[lambda x: x.A > 2, ["A", "B"]]
        tm.assert_frame_equal(res, df.loc[df.A > 2, ["A", "B"]])

        res = df.loc[[2, 3], lambda x: ["A", "B"]]
        tm.assert_frame_equal(res, df.loc[[2, 3], ["A", "B"]])

        res = df.loc[3, lambda x: ["A", "B"]]
        tm.assert_series_equal(res, df.loc[3, ["A", "B"]])

    def test_frame_loc_getitem_callable_labels(self):
        # [FUNCTION DOC]
        # Verifies callables returning labels (lists or scalar) on rows, columns and both, and mixed with direct labels, equal the direct equivalents.
        # GH#11485
        df = DataFrame({"X": [1, 2, 3, 4], "Y": list("aabb")}, index=list("ABCD"))

        # return label
        res = df.loc[lambda x: ["A", "C"]]
        tm.assert_frame_equal(res, df.loc[["A", "C"]])

        res = df.loc[lambda x: ["A", "C"], :]
        tm.assert_frame_equal(res, df.loc[["A", "C"], :])

        res = df.loc[lambda x: ["A", "C"], lambda x: "X"]
        tm.assert_series_equal(res, df.loc[["A", "C"], "X"])

        res = df.loc[lambda x: ["A", "C"], lambda x: ["X"]]
        tm.assert_frame_equal(res, df.loc[["A", "C"], ["X"]])

        # mixture
        res = df.loc[["A", "C"], lambda x: "X"]
        tm.assert_series_equal(res, df.loc[["A", "C"], "X"])

        res = df.loc[["A", "C"], lambda x: ["X"]]
        tm.assert_frame_equal(res, df.loc[["A", "C"], ["X"]])

        res = df.loc[lambda x: ["A", "C"], "X"]
        tm.assert_series_equal(res, df.loc[["A", "C"], "X"])

        res = df.loc[lambda x: ["A", "C"], ["X"]]
        tm.assert_frame_equal(res, df.loc[["A", "C"], ["X"]])

    def test_frame_loc_setitem_callable(self):
        # [FUNCTION DOC]
        # Verifies assignment with callable keys equals assignment with the direct keys for rows only, rows with full slice, both axes, mixes with arrays and scalars. Compares each case on a copy.
        # GH#11485
        df = DataFrame(
            {"X": [1, 2, 3, 4], "Y": Series(list("aabb"), dtype=object)},
            index=list("ABCD"),
        )

        # return label
        res = df.copy()
        res.loc[lambda x: ["A", "C"]] = -20
        exp = df.copy()
        exp.loc[["A", "C"]] = -20
        tm.assert_frame_equal(res, exp)

        res = df.copy()
        res.loc[lambda x: ["A", "C"], :] = 20
        exp = df.copy()
        exp.loc[["A", "C"], :] = 20
        tm.assert_frame_equal(res, exp)

        res = df.copy()
        res.loc[lambda x: ["A", "C"], lambda x: "X"] = -1
        exp = df.copy()
        exp.loc[["A", "C"], "X"] = -1
        tm.assert_frame_equal(res, exp)

        res = df.copy()
        res.loc[lambda x: ["A", "C"], lambda x: ["X"]] = [5, 10]
        exp = df.copy()
        exp.loc[["A", "C"], ["X"]] = [5, 10]
        tm.assert_frame_equal(res, exp)

        # mixture
        res = df.copy()
        res.loc[["A", "C"], lambda x: "X"] = np.array([-1, -2])
        exp = df.copy()
        exp.loc[["A", "C"], "X"] = np.array([-1, -2])
        tm.assert_frame_equal(res, exp)

        res = df.copy()
        res.loc[["A", "C"], lambda x: ["X"]] = 10
        exp = df.copy()
        exp.loc[["A", "C"], ["X"]] = 10
        tm.assert_frame_equal(res, exp)

        res = df.copy()
        res.loc[lambda x: ["A", "C"], "X"] = -2
        exp = df.copy()
        exp.loc[["A", "C"], "X"] = -2
        tm.assert_frame_equal(res, exp)

        res = df.copy()
        res.loc[lambda x: ["A", "C"], ["X"]] = -4
        exp = df.copy()
        exp.loc[["A", "C"], ["X"]] = -4
        tm.assert_frame_equal(res, exp)


class TestPartialStringSlicing:
    # [CLASS DOC]
    # Tests for selection by partial date strings.
    def test_loc_getitem_partial_string_slicing_datetimeindex(self):
        # [FUNCTION DOC]
        # Verifies a month string selects all rows in that month from a non-monotonic datetime index (issue 35509).
        # GH#35509
        df = DataFrame(
            {"col1": ["a", "b", "c"], "col2": [1, 2, 3]},
            index=to_datetime(["2020-08-01", "2020-07-02", "2020-08-05"]),
        )
        expected = DataFrame(
            {"col1": ["a", "c"], "col2": [1, 3]},
            index=to_datetime(["2020-08-01", "2020-08-05"]),
        )
        result = df.loc["2020-08"]
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_partial_string_slicing_with_periodindex(self):
        # [FUNCTION DOC]
        # Verifies a slice ending with a month string on a monthly period index selects up to and including that month.
        pi = pd.period_range(start="2017-01-01", end="2018-01-01", freq="M")
        ser = pi.to_series()
        result = ser.loc[:"2017-12"]
        expected = ser.iloc[:-1]

        tm.assert_series_equal(result, expected)

    def test_loc_getitem_partial_string_slicing_with_timedeltaindex(self):
        # [FUNCTION DOC]
        # Verifies an open-start slice with the end given as a text day count on an hourly time delta index excludes the following day's first value only (all before day two).
        ix = timedelta_range(start="1 day", end="2 days", freq="1h")
        ser = ix.to_series()
        result = ser.loc[:"1 days"]
        expected = ser.iloc[:-1]

        tm.assert_series_equal(result, expected)

    def test_loc_getitem_str_timedeltaindex(self):
        # [FUNCTION DOC]
        # Verifies a time-delta-like string selects the first row of a daily time delta index as a series (issue 16896).
        # GH#16896
        df = DataFrame({"x": range(3)}, index=to_timedelta(range(3), unit="days"))
        expected = df.iloc[0]
        sliced = df.loc["0 days"]
        tm.assert_series_equal(sliced, expected)

    @pytest.mark.parametrize("indexer_end", [None, "2020-01-02 23:59:59.999999999"])
    def test_loc_getitem_partial_slice_non_monotonicity(
        self, tz_aware_fixture, indexer_end, frame_or_series
    ):
        # [FUNCTION DOC]
        # Verifies partial string slices on a non-monotonic timezone-aware datetime index with and without an explicit end (issue 33146), for series and frame via the fixture and time zone fixture: the result includes the dates from the start date onward that fall within the specified range, in original order, both through bracket and label selection.
        # GH#33146
        obj = frame_or_series(
            [1] * 5,
            index=DatetimeIndex(
                [
                    Timestamp("2019-12-30"),
                    Timestamp("2020-01-01"),
                    Timestamp("2019-12-25"),
                    Timestamp("2020-01-02 23:59:59.999999999"),
                    Timestamp("2019-12-19"),
                ],
                tz=tz_aware_fixture,
            ),
        )
        expected = frame_or_series(
            [1] * 2,
            index=DatetimeIndex(
                [
                    Timestamp("2020-01-01"),
                    Timestamp("2020-01-02 23:59:59.999999999"),
                ],
                tz=tz_aware_fixture,
            ),
        )
        indexer = slice("2020-01-01", indexer_end)

        result = obj[indexer]
        tm.assert_equal(result, expected)

        result = obj.loc[indexer]
        tm.assert_equal(result, expected)


class TestLabelSlicing:
    # [CLASS DOC]
    # Tests for label slicing semantics across index types.
    def test_loc_getitem_slicing_datetimes_frame(self):
        # [FUNCTION DOC]
        # Verifies slicing a frame whose index consists of datetimes, with and without duplicates (issue 7523), using open and closed datetime bounds and a partial string bound with an hour; a later start excludes the first row.
        # GH#7523

        # unique
        df_unique = DataFrame(
            np.arange(4.0, dtype="float64"),
            index=[datetime(2001, 1, i, 10, 00) for i in [1, 2, 3, 4]],
        )

        # duplicates
        df_dups = DataFrame(
            np.arange(5.0, dtype="float64"),
            index=[datetime(2001, 1, i, 10, 00) for i in [1, 2, 2, 3, 4]],
        )

        for df in [df_unique, df_dups]:
            result = df.loc[datetime(2001, 1, 1, 10) :]
            tm.assert_frame_equal(result, df)
            result = df.loc[: datetime(2001, 1, 4, 10)]
            tm.assert_frame_equal(result, df)
            result = df.loc[datetime(2001, 1, 1, 10) : datetime(2001, 1, 4, 10)]
            tm.assert_frame_equal(result, df)

            result = df.loc[datetime(2001, 1, 1, 11) :]
            expected = df.iloc[1:]
            tm.assert_frame_equal(result, expected)
            result = df.loc["20010101 11":]
            tm.assert_frame_equal(result, expected)

    def test_loc_getitem_label_slice_across_dst(self):
        # [FUNCTION DOC]
        # Verifies slicing and scalar lookup across the end of daylight saving time in a Berlin-zoned index (issue 21846): the slice from a later timestamp to an earlier-offset equal-looking timestamp returns two entries and the scalar lookup returns the third element.
        # GH#21846
        idx = date_range(
            "2017-10-29 01:30:00", tz="Europe/Berlin", periods=5, freq="30 min"
        )
        series2 = Series([0, 1, 2, 3, 4], index=idx)

        t_1 = Timestamp("2017-10-29 02:30:00+02:00", tz="Europe/Berlin")
        t_2 = Timestamp("2017-10-29 02:00:00+01:00", tz="Europe/Berlin")
        result = series2.loc[t_1:t_2]
        expected = Series([2, 3], index=idx[2:4])
        tm.assert_series_equal(result, expected)

        result = series2[t_1]
        expected = 2
        assert result == expected

    @pytest.mark.parametrize(
        "index",
        [
            pd.period_range(start="2017-01-01", end="2018-01-01", freq="M"),
            timedelta_range(start="1 day", end="2 days", freq="1h"),
        ],
    )
    def test_loc_getitem_label_slice_period_timedelta(self, index):
        # [FUNCTION DOC]
        # Verifies a slice ending at the second-to-last label of a period or time delta index returns all but the last element. Parametrized over a monthly period index and an hourly time delta index.
        ser = index.to_series()
        result = ser.loc[: index[-2]]
        expected = ser.iloc[:-1]

        tm.assert_series_equal(result, expected)

    def test_loc_getitem_slice_floats_inexact(self):
        # [FUNCTION DOC]
        # Verifies slicing a float-indexed frame with bounds that fall between labels: includes only labels inside the bounds (lengths two, two and three for three bound pairs).
        index = [52195.504153, 52196.303147, 52198.369883]
        df = DataFrame(np.random.default_rng(2).random((3, 2)), index=index)

        s1 = df.loc[52195.1:52196.5]
        assert len(s1) == 2

        s1 = df.loc[52195.1:52196.6]
        assert len(s1) == 2

        s1 = df.loc[52195.1:52198.9]
        assert len(s1) == 3

    def test_loc_getitem_float_slice_floatindex(self, float_numpy_dtype):
        # [FUNCTION DOC]
        # Verifies slicing a series with a float index of each float numpy dtype from a start label that exists and from one that falls between labels returns eight and seven elements; also after changing one label to a non-integer float, still sorted. Parametrized by the float dtype fixture.
        dtype = float_numpy_dtype
        ser = Series(
            np.random.default_rng(2).random(10), index=np.arange(10, 20, dtype=dtype)
        )

        assert len(ser.loc[12.0:]) == 8
        assert len(ser.loc[12.5:]) == 7

        idx = np.arange(10, 20, dtype=dtype)
        idx[2] = 12.2
        ser.index = idx
        assert len(ser.loc[12.0:]) == 8
        assert len(ser.loc[12.5:]) == 7

    @pytest.mark.parametrize(
        "start,stop, expected_slice",
        [
            [np.timedelta64(0, "ns"), None, slice(0, 11)],
            [np.timedelta64(1, "D"), np.timedelta64(6, "D"), slice(1, 7)],
            [None, np.timedelta64(4, "D"), slice(0, 5)],
        ],
    )
    def test_loc_getitem_slice_label_td64obj(self, start, stop, expected_slice):
        # [FUNCTION DOC]
        # Verifies slicing by numpy time delta bounds (or an open bound) on a daily time delta index (issue 20393). Parametrized over (start, stop, expected position slice).
        # GH#20393
        ser = Series(range(11), timedelta_range("0 days", "10 days"))
        result = ser.loc[slice(start, stop)]
        expected = ser.iloc[expected_slice]
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("start", ["2018", "2020"])
    def test_loc_getitem_slice_unordered_dt_index(self, frame_or_series, start):
        # [FUNCTION DOC]
        # Verifies a partial string slice on an unsorted datetime index raises a key error saying value-based partial slicing on non-monotonic indexes with a missing bound is not allowed. Parametrized over start strings 2018 and 2020 and frame-or-series.
        obj = frame_or_series(
            [1, 2, 3],
            index=[Timestamp("2016"), Timestamp("2019"), Timestamp("2017")],
        )
        with pytest.raises(
            KeyError, match="Value based partial slicing on non-monotonic"
        ):
            obj.loc[start:"2022"]

    @pytest.mark.parametrize("value", [1, 1.5])
    def test_loc_getitem_slice_labels_int_in_object_index(self, frame_or_series, value):
        # [FUNCTION DOC]
        # Verifies a slice between an integer or float label and a text label on an object index returns all four elements (issue 26491). Parametrized over the two start values and frame-or-series.
        # GH: 26491
        obj = frame_or_series(range(4), index=[value, "first", 2, "third"])
        result = obj.loc[value:"third"]
        expected = frame_or_series(range(4), index=[value, "first", 2, "third"])
        tm.assert_equal(result, expected)

    def test_loc_getitem_slice_columns_mixed_dtype(self):
        # [FUNCTION DOC]
        # Verifies slicing columns from an integer label on a frame with mixed text and integer column labels returns the later integer-labelled columns with an object column index (issue 20975).
        # GH: 20975
        df = DataFrame({"test": 1, 1: 2, 2: 3}, index=[0])
        expected = DataFrame(
            data=[[2, 3]], index=[0], columns=Index([1, 2], dtype=object)
        )
        tm.assert_frame_equal(df.loc[:, 1:], expected)


class TestLocBooleanLabelsAndSlices:
    # [CLASS DOC]
    # Tests for boolean labels and boolean slices with the label indexer.
    @pytest.mark.parametrize("bool_value", [True, False])
    def test_loc_bool_incompatible_index_raises(
        self, index, frame_or_series, bool_value
    ):
        # [FUNCTION DOC]
        # Verifies that a boolean scalar key on a non-boolean index raises a key error saying a boolean label cannot be used without a boolean index (issue 20432). Parametrized over true and false, and over the index fixture, frame-or-series. For boolean-typed indexes the test performs no check.
        # GH20432
        message = f"{bool_value}: boolean label can not be used without a boolean index"
        if index.inferred_type != "boolean":
            obj = frame_or_series(index=index, dtype="object")
            with pytest.raises(KeyError, match=message):
                obj.loc[bool_value]

    @pytest.mark.parametrize("bool_value", [True, False])
    def test_loc_bool_should_not_raise(self, frame_or_series, bool_value):
        # [FUNCTION DOC]
        # Verifies that a boolean scalar key on an object frame or series with a nullable-boolean index does not raise. There is no assertion beyond the absence of an error.
        obj = frame_or_series(
            index=Index([True, False], dtype="boolean"), dtype="object"
        )
        obj.loc[bool_value]

    def test_loc_bool_slice_raises(self, index, frame_or_series):
        # [FUNCTION DOC]
        # Verifies that a slice with boolean bounds raises a type error saying boolean values cannot be used in a slice (issue 20432), for every index fixture and frame-or-series.
        # GH20432
        message = (
            r"slice\(True, False, None\): boolean values can not be used in a slice"
        )
        obj = frame_or_series(index=index, dtype="object")
        with pytest.raises(TypeError, match=message):
            obj.loc[True:False]


class TestLocBooleanMask:
    # [CLASS DOC]
    # Tests for boolean masks with the label indexer.
    def test_loc_setitem_bool_mask_timedeltaindex(self):
        # [FUNCTION DOC]
        # Verifies setting a column to ten through different boolean conditions (greater, equal, less than three) on a frame with a time delta index (issue 14946); each condition gives the expected data.
        # GH#14946
        df = DataFrame({"x": range(10)})
        df.index = to_timedelta(range(10), unit="s")
        conditions = [df["x"] > 3, df["x"] == 3, df["x"] < 3]
        expected_data = [
            [0, 1, 2, 3, 10, 10, 10, 10, 10, 10],
            [0, 1, 2, 10, 4, 5, 6, 7, 8, 9],
            [10, 10, 10, 3, 4, 5, 6, 7, 8, 9],
        ]
        for cond, data in zip(conditions, expected_data):
            result = df.copy()
            result.loc[cond, "x"] = 10

            expected = DataFrame(
                data,
                index=to_timedelta(range(10), unit="s"),
                columns=["x"],
                dtype="int64",
            )
            tm.assert_frame_equal(expected, result)

    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_loc_setitem_mask_with_datetimeindex_tz(self, tz):
        # [FUNCTION DOC]
        # Verifies setting rows by boolean array from the same selected rows of a datetime-indexed frame (naive or UTC) leaves the frame unchanged, with and without the column selector (issue 16889). Parametrized over time zone none and UTC.
        # GH#16889
        # support .loc with alignment and tz-aware DatetimeIndex
        mask = np.array([True, False, True, False])

        idx = date_range("20010101", periods=4, tz=tz)
        df = DataFrame({"a": np.arange(4)}, index=idx).astype("float64")

        result = df.copy()
        result.loc[mask, :] = df.loc[mask, :]
        tm.assert_frame_equal(result, df)

        result = df.copy()
        result.loc[mask] = df.loc[mask]
        tm.assert_frame_equal(result, df)

    def test_loc_setitem_mask_and_label_with_datetimeindex(self):
        # [FUNCTION DOC]
        # Verifies a mask combined with a new column label on a datetime-indexed frame sets only the masked row to the index value, leaving not-a-time elsewhere (issue 9478).
        # GH#9478
        # a datetimeindex alignment issue with partial setting
        df = DataFrame(
            np.arange(6.0).reshape(3, 2),
            columns=list("AB"),
            index=date_range("1/1/2000", periods=3, freq="1h"),
        )
        expected = df.copy()
        expected["C"] = [expected.index[0], pd.NaT, pd.NaT]

        mask = df.A < 1
        df.loc[mask, "C"] = df.loc[mask].index
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_mask_td64_series_value(self):
        # [FUNCTION DOC]
        # Verifies setting rows with a boolean list and a one-element time delta series leaves the frame unchanged since the single-row mask has no matching alignment (issue 23462) and the second element of the column, compared by internal integer value, stays the same.
        # GH#23462 key list of bools, value is a Series
        td1 = Timedelta(0)
        td2 = Timedelta(28767471428571405)
        df = DataFrame({"col": Series([td1, td2])})
        df_copy = df.copy()
        ser = Series([td1])

        expected = df["col"].iloc[1]._value
        df.loc[[True, False]] = ser
        result = df["col"].iloc[1]._value

        assert expected == result
        tm.assert_frame_equal(df, df_copy)

    def test_loc_setitem_boolean_and_column(self, float_frame):
        # [FUNCTION DOC]
        # Verifies setting zero in a column for rows where another column is positive (uses the float frame fixture) equals the same operation done on the raw array.
        expected = float_frame.copy()
        mask = float_frame["A"] > 0

        float_frame.loc[mask, "B"] = 0

        values = expected.values.copy()
        values[mask.values, 1] = 0
        expected = DataFrame(values, index=expected.index, columns=expected.columns)
        tm.assert_frame_equal(float_frame, expected)

    def test_loc_setitem_ndframe_values_alignment(self):
        # [FUNCTION DOC]
        # Verifies a frame or series assigned by label with a boolean mask aligns on the index of the right side (issue 45501): a frame with reversed index, a series, and setting a scalar column; and assignment on a series extracted from a frame does not change the frame (copy-on-write).
        # GH#45501
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.loc[[False, False, True], ["a"]] = DataFrame(
            {"a": [10, 20, 30]}, index=[2, 1, 0]
        )

        expected = DataFrame({"a": [1, 2, 10], "b": [4, 5, 6]})
        tm.assert_frame_equal(df, expected)

        # same thing with Series RHS
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.loc[[False, False, True], ["a"]] = Series([10, 11, 12], index=[2, 1, 0])
        tm.assert_frame_equal(df, expected)

        # same thing but setting "a" instead of ["a"]
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.loc[[False, False, True], "a"] = Series([10, 11, 12], index=[2, 1, 0])
        tm.assert_frame_equal(df, expected)

        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df_orig = df.copy()
        ser = df["a"]
        ser.loc[[False, False, True]] = Series([10, 11, 12], index=[2, 1, 0])
        tm.assert_frame_equal(df, df_orig)

    def test_loc_indexer_empty_broadcast(self):
        # [FUNCTION DOC]
        # Verifies an empty boolean array mask with a column list and a right side of the same column is a no-op on an empty object frame (issue 51450).
        # GH#51450
        df = DataFrame({"a": [], "b": []}, dtype=object)
        expected = df.copy()
        df.loc[np.array([], dtype=np.bool_), ["a"]] = df["a"].copy()
        tm.assert_frame_equal(df, expected)

    def test_loc_indexer_all_false_broadcast(self):
        # [FUNCTION DOC]
        # Verifies an all-false mask with a right side from another column does not change the one-row object frame (issue 51450).
        # GH#51450
        df = DataFrame({"a": ["x"], "b": ["y"]}, dtype=object)
        expected = df.copy()
        df.loc[np.array([False], dtype=np.bool_), ["a"]] = df["b"].copy()
        tm.assert_frame_equal(df, expected)

    def test_loc_indexer_length_one(self):
        # [FUNCTION DOC]
        # Verifies a mask of one true value with a right side from another column copies the value into the column for the one row (issue 51435).
        # GH#51435
        df = DataFrame({"a": ["x"], "b": ["y"]}, dtype=object)
        expected = DataFrame({"a": ["y"], "b": ["y"]}, dtype=object)
        df.loc[np.array([True], dtype=np.bool_), ["a"]] = df["b"].copy()
        tm.assert_frame_equal(df, expected)


class TestLocListlike:
    # [CLASS DOC]
    # Tests for list-like keys with the label indexer.
    @pytest.mark.parametrize("box", [lambda x: x, np.asarray, list])
    def test_loc_getitem_list_of_labels_categoricalindex_with_na(self, box):
        # [FUNCTION DOC]
        # Verifies list-of-labels keys on a categorical index that contains a missing value (an index, array or list of the same labels) return the whole series by label, bracket, and via a frame; if the key has no missing value entry while the key contains none-matching, a key error saying not in the index is raised for these three paths. Parametrized over three key containers.
        # passing a list can include valid categories _or_ NA values
        ci = CategoricalIndex(["A", "B", np.nan])
        ser = Series(range(3), index=ci)

        result = ser.loc[box(ci)]
        tm.assert_series_equal(result, ser)

        result = ser[box(ci)]
        tm.assert_series_equal(result, ser)

        result = ser.to_frame().loc[box(ci)]
        tm.assert_frame_equal(result, ser.to_frame())

        ser2 = ser[:-1]
        ci2 = ci[1:]
        # but if there are no NAs present, this should raise KeyError
        msg = "not in index"
        with pytest.raises(KeyError, match=msg):
            ser2.loc[box(ci2)]

        with pytest.raises(KeyError, match=msg):
            ser2[box(ci2)]

        with pytest.raises(KeyError, match=msg):
            ser2.to_frame().loc[box(ci2)]

    def test_loc_getitem_series_label_list_missing_values(self):
        # [FUNCTION DOC]
        # Verifies a datetime64 array key containing a missing date raises a key error saying not in the index (issue 11428).
        # gh-11428
        key = np.array(
            ["2001-01-04", "2001-01-02", "2001-01-04", "2001-01-14"], dtype="datetime64"
        )
        ser = Series([2, 5, 8, 11], date_range("2001-01-01", freq="D", periods=4))
        with pytest.raises(KeyError, match="not in index"):
            ser.loc[key]

    def test_loc_getitem_series_label_list_missing_integer_values(self):
        # [FUNCTION DOC]
        # Verifies a key array with a large integer not in the index raises a key error "not in index" even with 64-bit values (issue 25927).
        # GH: 25927
        ser = Series(
            index=np.array([9730701000001104, 10049011000001109]),
            data=np.array([999000011000001104, 999000011000001104]),
        )
        with pytest.raises(KeyError, match="not in index"):
            ser.loc[np.array([9730701000001104, 10047311000001102])]

    @pytest.mark.parametrize("to_period", [True, False])
    def test_loc_getitem_listlike_of_datetimelike_keys(self, to_period):
        # [FUNCTION DOC]
        # Verifies list-like keys of timestamps (or periods) on a named datetime or period index (issue 11497).
        # Present keys in order return a series with the same names (the frequency is dropped for datetimes); repeated keys keep order and duplicates; a list including an absent key raises a key error. Parametrized over whether the index is a period index.
        # GH#11497

        idx = date_range("2011-01-01", "2011-01-02", freq="D", name="idx")
        if to_period:
            idx = idx.to_period("D")
        ser = Series([0.1, 0.2], index=idx, name="s")

        keys = [Timestamp("2011-01-01"), Timestamp("2011-01-02")]
        if to_period:
            keys = [x.to_period("D") for x in keys]
        result = ser.loc[keys]
        exp = Series([0.1, 0.2], index=idx, name="s")
        if not to_period:
            exp.index = exp.index._with_freq(None)
        tm.assert_series_equal(result, exp, check_index_type=True)

        keys = [
            Timestamp("2011-01-02"),
            Timestamp("2011-01-02"),
            Timestamp("2011-01-01"),
        ]
        if to_period:
            keys = [x.to_period("D") for x in keys]
        exp = Series(
            [0.2, 0.2, 0.1], index=Index(keys, name="idx", dtype=idx.dtype), name="s"
        )
        result = ser.loc[keys]
        tm.assert_series_equal(result, exp, check_index_type=True)

        keys = [
            Timestamp("2011-01-03"),
            Timestamp("2011-01-02"),
            Timestamp("2011-01-03"),
        ]
        if to_period:
            keys = [x.to_period("D") for x in keys]

        with pytest.raises(KeyError, match="not in index"):
            ser.loc[keys]

    def test_loc_named_index(self):
        # [FUNCTION DOC]
        # Verifies that selecting with a named index key returns rows with that name (issue 42790).
        # GH 42790
        df = DataFrame(
            [[1, 2], [4, 5], [7, 8]],
            index=["cobra", "viper", "sidewinder"],
            columns=["max_speed", "shield"],
        )
        expected = df.iloc[:2]
        expected.index.name = "foo"
        result = df.loc[Index(["cobra", "viper"], name="foo")]
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "columns, column_key, expected_columns",
    [
        ([2011, 2012, 2013], [2011, 2012], [0, 1]),
        ([2011, 2012, "All"], [2011, 2012], [0, 1]),
        ([2011, 2012, "All"], [2011, "All"], [0, 2]),
    ],
)
def test_loc_getitem_label_list_integer_labels(columns, column_key, expected_columns):
    # [FUNCTION DOC]
    # Verifies selecting rows by labels and columns by a list of integer or mixed labels returns the columns at the expected positions with the proper column type (issue 14836). Parametrized over (columns, column key, expected positions).
    # gh-14836
    df = DataFrame(
        np.random.default_rng(2).random((3, 3)), columns=columns, index=list("ABC")
    )
    expected = df.iloc[:, expected_columns]
    result = df.loc[["A", "B", "C"], column_key]

    tm.assert_frame_equal(result, expected, check_column_type=True)


def test_loc_setitem_float_intindex():
    # [FUNCTION DOC]
    # Verifies that setting a column labelled with a float on a frame with integer columns adds a float-labelled column of missing values (issue 8720).
    # KNOWN ISSUE: the second half of the test repeats the same construction, assignment and assertion as the first half.
    # GH 8720
    rand_data = np.random.default_rng(2).standard_normal((8, 4))
    result = DataFrame(rand_data)
    result.loc[:, 0.5] = np.nan
    expected_data = np.hstack((rand_data, np.array([np.nan] * 8).reshape(8, 1)))
    expected = DataFrame(expected_data, columns=[0.0, 1.0, 2.0, 3.0, 0.5])
    tm.assert_frame_equal(result, expected)

    result = DataFrame(rand_data)
    result.loc[:, 0.5] = np.nan
    tm.assert_frame_equal(result, expected)


def test_loc_axis_1_slice():
    # [FUNCTION DOC]
    # Verifies column slicing with tuple bounds on a multi-level column index through the axis-bound label indexer (issue 10586): selects the columns from the first bound to the second in index order.
    # GH 10586
    cols = [(yr, m) for yr in [2014, 2015] for m in [7, 8, 9, 10]]
    df = DataFrame(
        np.ones((10, 8)),
        index=tuple("ABCDEFGHIJ"),
        columns=MultiIndex.from_tuples(cols),
    )
    result = df.loc(axis=1)[(2014, 9) : (2015, 8)]
    expected = DataFrame(
        np.ones((10, 4)),
        index=tuple("ABCDEFGHIJ"),
        columns=MultiIndex.from_tuples([(2014, 9), (2014, 10), (2015, 7), (2015, 8)]),
    )
    tm.assert_frame_equal(result, expected)


def test_loc_set_dataframe_multiindex():
    # [FUNCTION DOC]
    # Verifies assigning a selection back to itself with a one-tuple list column key on multi-level columns keeps the frame unchanged (issue 14592).
    # GH 14592
    expected = DataFrame(
        "a", index=range(2), columns=MultiIndex.from_product([range(2), range(2)])
    )
    result = expected.copy()
    result.loc[0, [(0, 1)]] = result.loc[0, [(0, 1)]]
    tm.assert_frame_equal(result, expected)


def test_loc_mixed_int_float():
    # [FUNCTION DOC]
    # Verifies an integer key finds the equal float label in an object index with mixed integer and float labels (issue 19456).
    # GH#19456
    ser = Series(range(2), Index([1, 2.0], dtype=object))

    result = ser.loc[1]
    assert result == 0


def test_loc_with_positional_slice_raises():
    # [FUNCTION DOC]
    # Verifies that assigning with an integer positional slice via the label indexer on a string-indexed series raises a type error saying slicing a positional slice with the label indexer is not allowed (issue 31840).
    # GH#31840
    ser = Series(range(4), index=["A", "B", "C", "D"])

    with pytest.raises(TypeError, match="Slicing a positional slice with .loc"):
        ser.loc[:3] = 2


def test_loc_slice_disallows_positional():
    # [FUNCTION DOC]
    # Verifies integer slices on a datetime index with the label indexer raise a type error about slice indexing with integer indexers (issues 16121, 24612, 31810), for frame and series, in getting; setting raises a type error about slicing a positional slice (issue 31840), also with a column key.
    # GH#16121, GH#24612, GH#31810
    dti = date_range("2016-01-01", periods=3)
    df = DataFrame(np.random.default_rng(2).random((3, 2)), index=dti)

    ser = df[0]

    msg = (
        "cannot do slice indexing on DatetimeIndex with these "
        r"indexers \[1\] of type int"
    )

    for obj in [df, ser]:
        with pytest.raises(TypeError, match=msg):
            obj.loc[1:3]

        with pytest.raises(TypeError, match="Slicing a positional slice with .loc"):
            # GH#31840 enforce incorrect behavior
            obj.loc[1:3] = 1

    with pytest.raises(TypeError, match=msg):
        df.loc[1:3, 1]

    with pytest.raises(TypeError, match="Slicing a positional slice with .loc"):
        # GH#31840 enforce incorrect behavior
        df.loc[1:3, 1] = 2


def test_loc_datetimelike_mismatched_dtypes():
    # [FUNCTION DOC]
    # Verifies that a time delta index with the same integer values as a datetime index does not match it (issue 32650): selection raises a key error saying none of the labels are in the index, for frame and series.
    # GH#32650 dont mix and match datetime/timedelta/period dtypes

    df = DataFrame(
        np.random.default_rng(2).standard_normal((5, 3)),
        columns=["a", "b", "c"],
        index=date_range("2012", freq="h", periods=5),
    )
    # create dataframe with non-unique DatetimeIndex
    df = df.iloc[[0, 2, 2, 3]].copy()

    dti = df.index
    tdi = pd.TimedeltaIndex(dti.asi8)  # matching i8 values

    msg = r"None of \[TimedeltaIndex.* are in the \[index\]"
    with pytest.raises(KeyError, match=msg):
        df.loc[tdi]

    with pytest.raises(KeyError, match=msg):
        df["a"].loc[tdi]


def test_loc_with_period_index_indexer():
    # [FUNCTION DOC]
    # Verifies a period index, a list of periods and slices of the index select the corresponding rows (issue 4125).
    # KNOWN ISSUE: the list-of-periods comparison is written three times with identical content.
    # GH#4125
    idx = pd.period_range("2002-01", "2003-12", freq="M")
    df = DataFrame(np.random.default_rng(2).standard_normal((24, 10)), index=idx)
    tm.assert_frame_equal(df, df.loc[idx])
    tm.assert_frame_equal(df, df.loc[list(idx)])
    tm.assert_frame_equal(df, df.loc[list(idx)])
    tm.assert_frame_equal(df.iloc[0:5], df.loc[idx[0:5]])
    tm.assert_frame_equal(df, df.loc[list(idx)])


def test_loc_setitem_multiindex_timestamp():
    # [FUNCTION DOC]
    # Verifies setting a missing value at a row label and a tuple of column labels on a frame with a datetime index and plain columns (issue 13831), comparing with the same change on the raw array.
    # GH#13831
    vals = np.random.default_rng(2).standard_normal((8, 6))
    idx = date_range("1/1/2000", periods=8)
    cols = ["A", "B", "C", "D", "E", "F"]
    exp = DataFrame(vals, index=idx, columns=cols)
    exp.loc[exp.index[1], ("A", "B")] = np.nan
    vals[1][0:2] = np.nan
    res = DataFrame(vals, index=idx, columns=cols)
    tm.assert_frame_equal(res, exp)


def test_loc_getitem_multiindex_tuple_level():
    # [FUNCTION DOC]
    # Verifies a tuple that is a single label inside one level of a three-level column index is treated as one label (issue 27591).
    # Setup: columns from the product of three lists where the middle level holds tuples. Selecting columns by a full key tuple returns one column (a note says it arguably should drop levels); the cross-section method gives the same; the same on a series row; and the full key on the series returns the scalar six.
    # GH#27591
    lev1 = ["a", "b", "c"]
    lev2 = [(0, 1), (1, 0)]
    lev3 = [0, 1]
    cols = MultiIndex.from_product([lev1, lev2, lev3], names=["x", "y", "z"])
    df = DataFrame(6, index=range(5), columns=cols)

    # the lev2[0] here should be treated as a single label, not as a sequence
    #  of labels
    result = df.loc[:, (lev1[0], lev2[0], lev3[0])]

    # TODO: i think this actually should drop levels
    expected = df.iloc[:, :1]
    tm.assert_frame_equal(result, expected)

    alt = df.xs((lev1[0], lev2[0], lev3[0]), level=[0, 1, 2], axis=1)
    tm.assert_frame_equal(alt, expected)

    # same thing on a Series
    ser = df.iloc[0]
    expected2 = ser.iloc[:1]

    alt2 = ser.xs((lev1[0], lev2[0], lev3[0]), level=[0, 1, 2], axis=0)
    tm.assert_series_equal(alt2, expected2)

    result2 = ser.loc[lev1[0], lev2[0], lev3[0]]
    assert result2 == 6


def test_loc_getitem_nullable_index_with_duplicates():
    # [FUNCTION DOC]
    # Verifies lookup on a nullable-integer index containing missing values and duplicates (issue 34497): a label present twice returns a series of the matching rows; the result is identical after converting the index to object dtype.
    # GH#34497
    df = DataFrame(
        data=np.array([[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, pd.NA, pd.NA]]).T,
        columns=["a", "b", "c"],
        dtype="Int64",
    )
    df2 = df.set_index("c")
    assert df2.index.dtype == "Int64"

    res = df2.loc[1]
    expected = Series([1, 5], index=df2.columns, dtype="Int64", name=1)
    tm.assert_series_equal(res, expected)

    # pd.NA and duplicates in an object-dtype Index
    df2.index = df2.index.astype(object)
    res = df2.loc[1]
    tm.assert_series_equal(res, expected)


@pytest.mark.parametrize("value", [300, np.uint16(300), np.int16(300)])
def test_loc_setitem_uint8_upcast(value):
    # [FUNCTION DOC]
    # Verifies assigning a value that does not fit an unsigned 8-bit column (Python int, unsigned 16-bit or signed 16-bit scalar) raises a type error about an invalid value (issue 26049).
    # GH#26049

    df = DataFrame([1, 2, 3, 4], columns=["col1"], dtype="uint8")
    with pytest.raises(TypeError, match="Invalid value"):
        df.loc[2, "col1"] = value  # value that can't be held in uint8


@pytest.mark.parametrize(
    "fill_val,exp_dtype",
    [
        (Timestamp("2022-01-06"), "datetime64[ns]"),
        (Timestamp("2022-01-07", tz="US/Eastern"), "datetime64[ns, US/Eastern]"),
    ],
)
def test_loc_setitem_using_datetimelike_str_as_index(fill_val, exp_dtype):
    # [FUNCTION DOC]
    # Verifies enlarging a datetime-indexed frame (naive or US Eastern) by a date string that is not in the index appends the label and keeps the datetime index dtype (and unit) exactly. Parametrized over (fill timestamp, expected dtype).
    data = ["2022-01-02", "2022-01-03", "2022-01-04", fill_val.date()]
    index = DatetimeIndex(data, tz=fill_val.tz, dtype=exp_dtype)
    df = DataFrame([10, 11, 12, 14], columns=["a"], index=index)
    # adding new row using an unexisting datetime-like str index
    df.loc["2022-01-08", "a"] = 13

    data.append("2022-01-08")
    expected_index = DatetimeIndex(data, dtype=exp_dtype)
    tm.assert_index_equal(df.index, expected_index, exact=True)


def test_loc_set_int_dtype():
    # [FUNCTION DOC]
    # Verifies adding a new column labelled with text by label over all rows of a frame with text columns appends an integer column (issue 23326).
    # GH#23326
    df = DataFrame([list("abc")])
    df.loc[:, "col1"] = 5

    expected = DataFrame({0: ["a"], 1: ["b"], 2: ["c"], "col1": [5]})
    tm.assert_frame_equal(df, expected)


@pytest.mark.filterwarnings(r"ignore:Period with BDay freq is deprecated:FutureWarning")
@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
def test_loc_periodindex_3_levels():
    # [FUNCTION DOC]
    # Verifies a three-part key with a period level (business-day frequency) reads the scalar from a three-level index series (issue 24091). Ignores future warnings about the business-day period.
    # GH#24091
    p_index = PeriodIndex(
        ["20181101 1100", "20181101 1200", "20181102 1300", "20181102 1400"],
        name="datetime",
        freq="B",
    )
    mi_series = DataFrame(
        [["A", "B", 1.0], ["A", "C", 2.0], ["Z", "Q", 3.0], ["W", "F", 4.0]],
        index=p_index,
        columns=["ONE", "TWO", "VALUES"],
    )
    mi_series = mi_series.set_index(["ONE", "TWO"], append=True)["VALUES"]
    assert mi_series.loc[(p_index[0], "A", "B")] == 1.0


def test_loc_setitem_pyarrow_strings():
    # [FUNCTION DOC]
    # Verifies assigning a series (shorter than the column) through a boolean mask to an Arrow string column keeps the Arrow string dtype and places the values in masked order (issue 52319). Skipped without pyarrow.
    # GH#52319
    pytest.importorskip("pyarrow")
    df = DataFrame(
        {
            "strings": Series(["A", "B", "C"], dtype="string[pyarrow]"),
            "ids": Series([True, True, False]),
        }
    )
    new_value = Series(["X", "Y"])
    df.loc[df.ids, "strings"] = new_value

    expected_df = DataFrame(
        {
            "strings": Series(["X", "Y", "C"], dtype="string[pyarrow]"),
            "ids": Series([True, True, False]),
        }
    )

    tm.assert_frame_equal(df, expected_df)


class TestLocSeries:
    # [CLASS DOC]
    # Tests of the label indexer on series, using the shared string-indexed and datetime-indexed series fixtures.
    @pytest.mark.parametrize("val,expected", [(2**63 - 1, 3), (2**63, 4)])
    def test_loc_uint64(self, val, expected):
        # [FUNCTION DOC]
        # Verifies lookup of labels at the signed 64-bit maximum and the next value (unsigned) on a series built from a mapping (issue 19399). Parametrized over (label, expected value).
        # see GH#19399
        ser = Series({2**63 - 1: 3, 2**63: 4})
        assert ser.loc[val] == expected

    def test_loc_getitem(self, string_series, datetime_series):
        # [FUNCTION DOC]
        # Verifies the label indexer on a series: a label list equals the reindexed series, position slice equals direct slice, a datetime label slice equals the truncated series, a boolean mask equals direct masking, and scalar lookups equal direct lookups.
        inds = string_series.index[[3, 4, 7]]
        tm.assert_series_equal(string_series.loc[inds], string_series.reindex(inds))
        tm.assert_series_equal(string_series.iloc[5::2], string_series[5::2])

        # slice with indices
        d1, d2 = datetime_series.index[[5, 15]]
        result = datetime_series.loc[d1:d2]
        expected = datetime_series.truncate(d1, d2)
        tm.assert_series_equal(result, expected)

        # boolean
        mask = string_series > string_series.median()
        tm.assert_series_equal(string_series.loc[mask], string_series[mask])

        # ask for index value
        assert datetime_series.loc[d1] == datetime_series[d1]
        assert datetime_series.loc[d2] == datetime_series[d2]

    def test_loc_getitem_not_monotonic(self, datetime_series):
        # [FUNCTION DOC]
        # Verifies that slicing and assigning with a label slice on a non-monotonic datetime-indexed series with missing bound raise a key error naming the missing timestamp.
        d1, d2 = datetime_series.index[[5, 15]]

        ts2 = datetime_series[::2].iloc[[1, 2, 0]]

        msg = r"Timestamp\('2000-01-10 00:00:00'\)"
        with pytest.raises(KeyError, match=msg):
            ts2.loc[d1:d2]
        with pytest.raises(KeyError, match=msg):
            ts2.loc[d1:d2] = 0

    def test_loc_getitem_setitem_integer_slice_keyerrors(self):
        # [FUNCTION DOC]
        # Verifies position and label slices on an even-labelled integer series: position slices assign fine; the label slice from three to eleven selects the labels between (four to ten); on a non-monotonic series the same slice get and set raise a key error about a non-monotonic index with a missing label three.
        ser = Series(
            np.random.default_rng(2).standard_normal(10), index=list(range(0, 20, 2))
        )

        # this is OK
        cp = ser.copy()
        cp.iloc[4:10] = 0
        assert (cp.iloc[4:10] == 0).all()

        # so is this
        cp = ser.copy()
        cp.iloc[3:11] = 0
        assert (cp.iloc[3:11] == 0).values.all()

        result = ser.iloc[2:6]
        result2 = ser.loc[3:11]
        expected = ser.reindex([4, 6, 8, 10])

        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(result2, expected)

        # non-monotonic, raise KeyError
        s2 = ser.iloc[list(range(5)) + list(range(9, 4, -1))]
        msg = "non-monotonic index with a missing label 3"
        with pytest.raises(KeyError, match=msg):
            s2.loc[3:11]
        with pytest.raises(KeyError, match=msg):
            s2.loc[3:11] = 0

    def test_loc_getitem_iterator(self, string_series):
        # [FUNCTION DOC]
        # Verifies an iterator of labels as the key selects those labels in order.
        idx = iter(string_series.index[:10])
        result = string_series.loc[idx]
        tm.assert_series_equal(result, string_series[:10])

    def test_loc_setitem_boolean(self, string_series):
        # [FUNCTION DOC]
        # Verifies setting zero by a boolean mask via the label indexer equals the direct bracket assignment with the same mask.
        mask = string_series > string_series.median()

        result = string_series.copy()
        result.loc[mask] = 0
        expected = string_series
        expected[mask] = 0
        tm.assert_series_equal(result, expected)

    def test_loc_setitem_corner(self, string_series):
        # [FUNCTION DOC]
        # Verifies that setting a list of existing labels works, and a list with an extra missing label raises a key error naming the missing label as not in the index. Modifies the shared series fixture.
        inds = list(string_series.index[[5, 8, 12]])
        string_series.loc[inds] = 5
        msg = r"\['foo'\] not in index"
        with pytest.raises(KeyError, match=msg):
            string_series.loc[[*inds, "foo"]] = 5

    def test_basic_setitem_with_labels(self, datetime_series):
        # [FUNCTION DOC]
        # Verifies bracket assignment with a list of labels or a label slice equals the same with the label indexer on a datetime series.
        indices = datetime_series.index[[5, 10, 15]]

        cp = datetime_series.copy()
        exp = datetime_series.copy()
        cp[indices] = 0
        exp.loc[indices] = 0
        tm.assert_series_equal(cp, exp)

        cp = datetime_series.copy()
        exp = datetime_series.copy()
        cp[indices[0] : indices[2]] = 0
        exp.loc[indices[0] : indices[2]] = 0
        tm.assert_series_equal(cp, exp)

    def test_loc_setitem_listlike_of_ints(self):
        # [FUNCTION DOC]
        # Verifies list and array of integer labels for bracket and label assignment on an integer-indexed series, and that a list containing a missing label raises a key error naming it.
        # KNOWN ISSUE: the comparison before the not-found checks compares two unmodified copies of the series with each other; the assignments are applied to the original series, never to the compared objects, so the first two comparisons can never fail.
        # integer indexes, be careful
        ser = Series(
            np.random.default_rng(2).standard_normal(10), index=list(range(0, 20, 2))
        )
        inds = [0, 4, 6]
        arr_inds = np.array([0, 4, 6])

        cp = ser.copy()
        exp = ser.copy()
        ser[inds] = 0
        ser.loc[inds] = 0
        tm.assert_series_equal(cp, exp)

        cp = ser.copy()
        exp = ser.copy()
        ser[arr_inds] = 0
        ser.loc[arr_inds] = 0
        tm.assert_series_equal(cp, exp)

        inds_notfound = [0, 4, 5, 6]
        arr_inds_notfound = np.array([0, 4, 5, 6])
        msg = r"\[5\] not in index"
        with pytest.raises(KeyError, match=msg):
            ser[inds_notfound] = 0
        with pytest.raises(Exception, match=msg):
            ser[arr_inds_notfound] = 0

    def test_loc_setitem_dt64tz_values(self):
        # [FUNCTION DOC]
        # Verifies setting and reading a timezone-aware datetime through the label indexer, the position indexer and bracket access on a series with string labels (issue 12089).
        # GH#12089
        ser = Series(
            date_range("2011-01-01", periods=3, tz="US/Eastern"),
            index=["a", "b", "c"],
        )
        s2 = ser.copy()
        expected = Timestamp("2011-01-03", tz="US/Eastern")
        s2.loc["a"] = expected
        result = s2.loc["a"]
        assert result == expected

        s2 = ser.copy()
        s2.iloc[0] = expected
        result = s2.iloc[0]
        assert result == expected

        s2 = ser.copy()
        s2["a"] = expected
        result = s2["a"]
        assert result == expected

    @pytest.mark.parametrize("array_fn", [np.array, pd.array, list, tuple])
    @pytest.mark.parametrize("size", [0, 4, 5, 6])
    def test_loc_iloc_setitem_with_listlike(self, size, array_fn):
        # [FUNCTION DOC]
        # Verifies that assigning a list-like (list, tuple, numpy array or extension array) of size zero, four, five or six into one element of an object series stores the whole list-like as the element, for label and position setters (issue 37748). Parametrized over container and size.
        # GH37748
        # testing insertion, in a Series of size N (here 5), of a listlike object
        # of size  0, N-1, N, N+1

        arr = array_fn([0] * size)
        expected = Series([arr, 0, 0, 0, 0], index=list("abcde"), dtype=object)

        ser = Series(0, index=list("abcde"), dtype=object)
        ser.loc["a"] = arr
        tm.assert_series_equal(ser, expected)

        ser = Series(0, index=list("abcde"), dtype=object)
        ser.iloc[0] = arr
        tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize("indexer", [IndexSlice["A", :], ("A", slice(None))])
    def test_loc_series_getitem_too_many_dimensions(self, indexer):
        # [FUNCTION DOC]
        # Verifies that passing a two-dimensional key on a series with a multi-level index raises an indexing error "Too many indexers", for getting and setting (issue 35349). Parametrized over two key forms (slice helper and plain tuple).
        # GH#35349
        ser = Series(
            index=MultiIndex.from_tuples([("A", "0"), ("A", "1"), ("B", "0")]),
            data=[21, 22, 23],
        )
        msg = "Too many indexers"
        with pytest.raises(IndexingError, match=msg):
            ser.loc[indexer, :]

        with pytest.raises(IndexingError, match=msg):
            ser.loc[indexer, :] = 1

    def test_loc_setitem(self, string_series):
        # [FUNCTION DOC]
        # Verifies label assignment on the string-indexed series: list of labels equals the position assignment; position slice and bracket slice agree; label slice assignment is inclusive of the end; scalar assignments take effect.
        inds = string_series.index[[3, 4, 7]]

        result = string_series.copy()
        result.loc[inds] = 5

        expected = string_series.copy()
        expected.iloc[[3, 4, 7]] = 5
        tm.assert_series_equal(result, expected)

        result.iloc[5:10] = 10
        expected[5:10] = 10
        tm.assert_series_equal(result, expected)

        # set slice with indices
        d1, d2 = string_series.index[[5, 15]]
        result.loc[d1:d2] = 6
        expected[5:16] = 6  # because it's inclusive
        tm.assert_series_equal(result, expected)

        # set index value
        string_series.loc[d1] = 4
        string_series.loc[d2] = 6
        assert string_series[d1] == 4
        assert string_series[d2] == 6

    @pytest.mark.parametrize("dtype", ["object", "string"])
    def test_loc_assign_dict_to_row(self, dtype):
        # [FUNCTION DOC]
        # Verifies assigning a dict to a whole row sets values by column name for object and string dtypes (issue 41044). Parametrized over the dtype.
        # GH41044
        df = DataFrame({"A": ["abc", "def"], "B": ["ghi", "jkl"]}, dtype=dtype)
        df.loc[0, :] = {"A": "newA", "B": "newB"}

        expected = DataFrame({"A": ["newA", "def"], "B": ["newB", "jkl"]}, dtype=dtype)

        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_dict_timedelta_multiple_set(self):
        # [FUNCTION DOC]
        # Verifies assigning a dict containing a time delta twice to the same row of an empty two-column frame gives a single row (issue 16309).
        # GH 16309
        result = DataFrame(columns=["time", "value"])
        result.loc[1] = {"time": Timedelta(6, unit="s"), "value": "foo"}
        result.loc[1] = {"time": Timedelta(6, unit="s"), "value": "foo"}
        expected = DataFrame(
            [[Timedelta(6, unit="s"), "foo"]], columns=["time", "value"], index=[1]
        )
        tm.assert_frame_equal(result, expected)

    def test_loc_set_multiple_items_in_multiple_new_columns(self):
        # [FUNCTION DOC]
        # Verifies setting two values for two new columns at one row creates the columns with missing values for the other row (issue 25594).
        # GH 25594
        df = DataFrame(index=[1, 2], columns=["a"])
        df.loc[1, ["b", "c"]] = [6, 7]

        expected = DataFrame(
            {
                "a": Series([np.nan, np.nan], dtype="object"),
                "b": [6, np.nan],
                "c": [7, np.nan],
            },
            index=[1, 2],
        )

        tm.assert_frame_equal(df, expected)

    def test_getitem_loc_str_periodindex(self):
        # [FUNCTION DOC]
        # Verifies a date string selects the element from a business-day period index (issue 33964), within a block asserting that a future warning about the business-day frequency is emitted.
        # GH#33964
        msg = "Period with BDay freq is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            index = pd.period_range(start="2000", periods=20, freq="B")
            series = Series(range(20), index=index)
            assert series.loc["2000-01-14"] == 9

    def test_loc_nonunique_masked_index(self):
        # [FUNCTION DOC]
        # Verifies lookup of a list of labels in a non-unique nullable-integer index gives all rows in the stable original order grouped by label (issue 57027).
        # GH 57027
        ids = list(range(11))
        index = Index(ids * 1000, dtype="Int64")
        df = DataFrame({"val": np.arange(len(index), dtype=np.intp)}, index=index)
        result = df.loc[ids]
        expected = DataFrame(
            {"val": index.argsort(kind="stable").astype(np.intp)},
            index=Index(np.array(ids).repeat(1000), dtype="Int64"),
        )
        tm.assert_frame_equal(result, expected)

    def test_loc_index_alignment_for_series(self):
        # [FUNCTION DOC]
        # Verifies assigning a series with a different order index to a column aligns on the index (issue 56024).
        # GH #56024
        df = DataFrame({"a": [1, 2], "b": [3, 4]})
        other = Series([200, 999], index=[1, 0])
        df.loc[:, "a"] = other
        expected = DataFrame({"a": [999, 200], "b": [3, 4]})
        tm.assert_frame_equal(expected, df)

    @pytest.mark.parametrize("dtype", ["str", object])
    def test_loc_reindexing_of_empty_index(self, dtype):
        # [FUNCTION DOC]
        # Verifies setting through a boolean series mask that is all false on a duplicated-index frame leaves it unchanged (issue 57735). Parametrized over string and object dtype.
        # GH 57735
        df = DataFrame(index=[1, 1, 2, 2], data=["1", "1", "2", "2"], dtype=dtype)
        df.loc[Series([False] * 4, index=df.index, name=0), 0] = df[0]
        expected = DataFrame(index=[1, 1, 2, 2], data=["1", "1", "2", "2"], dtype=dtype)
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "df, row_index, col_index, expected_df",
        [
            [
                DataFrame([[1, 2, 3], [4, 5, 6]], columns=list("ABC")),
                slice(0, 3),
                ["A", "B", "C"],
                DataFrame([[10, 10, 10], [20, 20, 20]], columns=list("ABC")),
            ],
            [
                DataFrame([[1, 2, 3], [4, 5, 6]], columns=list("ABC")),
                slice(None),
                ["A", "B", "C"],
                DataFrame([[10, 10, 10], [20, 20, 20]], columns=list("ABC")),
            ],
            [
                DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=list("ABC")),
                [True, True, True],
                ["A", "B", "C"],
                DataFrame(
                    [[10, 10, 10], [20, 20, 20], [30, 30, 30]], columns=list("ABC")
                ),
            ],
            [
                DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=list("ABC")),
                slice(0, 4),
                ["A", "B", "C"],
                DataFrame(
                    [[10, 10, 10], [20, 20, 20], [30, 30, 30]], columns=list("ABC")
                ),
            ],
            [
                DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=list("ABC")),
                slice(None),
                slice("A", "C"),
                DataFrame(
                    [[10, 10, 10], [20, 20, 20], [30, 30, 30]], columns=list("ABC")
                ),
            ],
            [
                DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=list("ABC")),
                slice(None),
                Series(
                    {
                        "A": True,
                        "C": False,
                        "B": True,
                    }
                ),
                DataFrame([[10, 10, 3], [20, 20, 6], [30, 30, 9]], columns=list("ABC")),
            ],
        ],
    )
    def test_loc_set_series_to_multiple_columns(
        self, df, row_index, col_index, expected_df
    ):
        # [FUNCTION DOC]
        # Verifies assigning a series of three values to several columns at once (issue 59933).
        # Parametrized over (frame, row indexer, column indexer, expected): slices (bounded, open and beyond the end), a boolean list for rows, label slices and a boolean series for columns. The series is aligned on the row positions and written into every selected column; columns not selected stay.
        # GH 59933
        df.loc[row_index, col_index] = Series([10, 20, 30])
        tm.assert_frame_equal(df, expected_df)

    def test_loc_setitem_matching_index(self):
        # [FUNCTION DOC]
        # Verifies assigning a series to a list of labels aligns by index (issue 25548): a series with matching labels sets the values, a series with unmatched labels sets missing values.
        # GH 25548
        s = Series(0.0, index=list("abcd"))
        s1 = Series(1.0, index=list("ab"))
        s2 = Series(2.0, index=list("xy"))

        # Test matching indices
        s.loc[["a", "b"]] = s1

        result = s[["a", "b"]]
        expected = s1
        tm.assert_series_equal(result, expected)

        # Test unmatched indices
        s.loc[["a", "b"]] = s2

        result = s[["a", "b"]]
        expected = Series([np.nan, np.nan], index=["a", "b"])
        tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "dtype",
    [
        "Int64",
        # string dtypes
        ("python", np.nan),
        pytest.param(("pyarrow", np.nan), marks=td.skip_if_no("pyarrow")),
    ],
)
@pytest.mark.parametrize(
    "box",
    [list, np.array, lambda x: Series(x, index=["a"]), Index, pd.array],
    ids=["list", "ndarray", "Series", "Index", "pd.array"],
)
def test_loc_setitem_single_column_frame_ea_dtype(dtype, box):
    # [FUNCTION DOC]
    # Verifies setting through a boolean column mask of length one on a one-column frame of an extension dtype keeps the dtype, for frame and scalar right sides (gh issue 66527).
    # Parametrized over dtype (nullable integer and the python-backed and pyarrow-backed string variants; the arrow one skipped when missing) and the container used for the mask (list, array, series, index, extension array). Sets all rows with a doubled frame, a subset of rows with doubled values, then a scalar to all rows and to some rows; each compared with an expected frame built by array arithmetic.
    # https://github.com/pandas-dev/pandas/issues/66527
    # column boolean mask that sets into the single column of a 1-column df
    if isinstance(dtype, tuple):
        dtype = pd.StringDtype(*dtype)
    df = DataFrame({"a": pd.array([1, 2, 3], dtype=dtype)})

    # setting with a 2d dataframe
    df.loc[:, box([True])] = df * 2

    expected = DataFrame({"a": pd.array([1, 2, 3], dtype=dtype) * 2})
    tm.assert_frame_equal(df, expected)

    df.loc[[0, 1], box([True])] = df.loc[[0, 1], :] * 2

    arr = pd.array([1, 2, 3], dtype=dtype) * 2
    arr[[0, 1]] = arr[[0, 1]] * 2
    expected = DataFrame({"a": arr})
    tm.assert_frame_equal(df, expected)

    # setting with a scalar
    df = DataFrame({"a": pd.array([1, 2, 3], dtype=dtype)})
    scalar = df.iloc[1, 0]

    df.loc[:, box([True])] = scalar

    expected = DataFrame({"a": pd.array([scalar] * 3, dtype=dtype)})
    tm.assert_frame_equal(df, expected)

    df.loc[[0, 1], box([True])] = scalar * 2

    expected = DataFrame({"a": pd.array([scalar * 2, scalar * 2, scalar], dtype=dtype)})
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    "dtype",
    ["float64", "Float64"],
)
@pytest.mark.parametrize(
    "box",
    [list, np.array, lambda x: Series(x, index=["a"]), Index, pd.array],
    ids=["list", "ndarray", "Series", "Index", "pd.array"],
)
def test_loc_setitem_empty_boolean_column_mask(dtype, box):
    # [FUNCTION DOC]
    # Verifies that a scalar assignment with a column boolean mask that selects no column leaves the frame unchanged (gh issue 66255). Parametrized over float and nullable float dtypes and five mask containers.
    # https://github.com/pandas-dev/pandas/issues/66255
    df = DataFrame({"a": [1, 2, 3, np.nan]}, dtype=dtype)
    df_orig = df.copy()

    # setting scalar
    df.loc[:, box([False])] = 100

    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize(
    "dtype",
    [
        "float64",
        pytest.param(
            "Float64",
            marks=pytest.mark.xfail(reason="Setting frame causes AssertionError"),
        ),
    ],
)
@pytest.mark.parametrize(
    "box",
    # Series boolean key needs to be aligned with the indexed axis
    [list, np.array, lambda x: Series(x, index=["a"]), Index, pd.array],
    ids=["list", "ndarray", "Series", "Index", "pd.array"],
)
def test_loc_setitem_empty_boolean_column_mask_frame_value(dtype, box):
    # [FUNCTION DOC]
    # Same as the previous test but the right side is a frame (gh issue 66255). The nullable-float case is marked as an expected failure because setting a frame causes an assertion error. A comment notes that a series mask must align with the indexed axis.
    # https://github.com/pandas-dev/pandas/issues/66255
    df = DataFrame({"a": [1, 2, 3, np.nan]}, dtype=dtype)
    df_orig = df.copy()

    # setting frame
    df.loc[:, box([False])] = df * 2

    tm.assert_frame_equal(df, df_orig)
