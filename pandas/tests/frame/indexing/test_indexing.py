# [MODULE DOC]
# Large test module for DataFrame indexing and assignment: getitem, setitem, boolean masks,
# label (loc), position (iloc), scalar (at, iat) and related accessors, plus many regression
# tests keyed by issue number. The main class covers general behavior; further classes cover
# specific topics (for example loc and iloc corner cases, mixed dtypes, slicing, setting with
# enlargement, boolean-mask assignment). Defines a module-level private message constant used for
# slice error matching: it is chosen at import time depending on whether the running interpreter
# is at least version 3.15, because the message text raised from numpy differs
# (the newer one omits the none option). Uses fixtures float_frame, mixed_float_frame,
# mixed_int_frame, datetime_frame, using_infer_string and others.
from collections import namedtuple
from datetime import (
    datetime,
    timedelta,
)
from decimal import Decimal
import re

import numpy as np
import pytest

from pandas._libs import iNaT
from pandas.compat import PY315
from pandas.errors import InvalidIndexError

from pandas.core.dtypes.common import is_integer

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
    isna,
    to_datetime,
)
import pandas._testing as tm

# We pass through a TypeError raised by numpy
if PY315:
    _slice_msg = "slice indices must be integers or have an __index__ method"
else:
    _slice_msg = "slice indices must be integers or None or have an __index__ method"


class TestDataFrameIndexing:
    # [CLASS DOC]
    # Main test class collecting general DataFrame indexing and setitem tests.
    def test_getitem(self, float_frame):
        # [FUNCTION DOC]
        # Verifies slicing the float frame to twenty rows keeps each column with that many rows and
        # the same index; every stored column is retrievable; and a missing label is reported as not
        # contained and raises a key error.
        # Slicing
        sl = float_frame[:20]
        assert len(sl.index) == 20

        # Column access
        for _, series in sl.items():
            assert len(series.index) == 20
            tm.assert_index_equal(series.index, sl.index)

        for key, _ in float_frame._series.items():
            assert float_frame[key] is not None

        assert "random" not in float_frame
        with pytest.raises(KeyError, match="random"):
            float_frame["random"]

    def test_getitem_numeric_should_not_fallback_to_positional(self, any_numeric_dtype):
        # [FUNCTION DOC]
        # Regression test for GH51053: with a column index of every numeric dtype (parametrized) holding
        # duplicated label one and zero, selecting label one returns the two matching columns by label,
        # not by position, and the result compares exactly.
        # GH51053
        dtype = any_numeric_dtype
        idx = Index([1, 0, 1], dtype=dtype)
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=idx)
        result = df[1]
        expected = DataFrame([[1, 3], [4, 6]], columns=Index([1, 1], dtype=dtype))
        tm.assert_frame_equal(result, expected, check_exact=True)

    def test_getitem2(self, float_frame):
        # [FUNCTION DOC]
        # Verifies columns whose names look like code or domain strings can be created and read back:
        # a column named with a dollar sign and a column named with an at sign. A lookup of an invented
        # string key raises a key error showing the string.
        df = float_frame.copy()
        df["$10"] = np.random.default_rng(2).standard_normal(len(df))

        ad = np.random.default_rng(2).standard_normal(len(df))
        df["@awesome_domain"] = ad

        with pytest.raises(KeyError, match=re.escape("'df[\"$10\"]'")):
            df.__getitem__('df["$10"]')

        res = df["@awesome_domain"]
        tm.assert_numpy_array_equal(ad, res.values)

    def test_setitem_numeric_should_not_fallback_to_positional(self, any_numeric_dtype):
        # [FUNCTION DOC]
        # Companion to the getitem case for GH51053: assigning a scalar to label one on a numeric index
        # with duplicates sets all matching columns by label, not by position; exact comparison.
        # GH51053
        dtype = any_numeric_dtype
        idx = Index([1, 0, 1], dtype=dtype)
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=idx)
        df[1] = 10
        expected = DataFrame([[10, 2, 10], [10, 5, 10]], columns=idx)
        tm.assert_frame_equal(df, expected, check_exact=True)

    def test_setitem_list(self, float_frame):
        # [FUNCTION DOC]
        # Verifies assigning a list of two labels from a two-column frame swaps data; assigning a
        # single-column key from a two-column frame raises a value error saying columns must be the same
        # length as the key; assigning a too-short range to one column raises a value error that the
        # length of values does not match the length of the index.
        float_frame["E"] = "foo"
        data = float_frame[["A", "B"]]
        float_frame[["B", "A"]] = data

        tm.assert_series_equal(float_frame["B"], data["A"], check_names=False)
        tm.assert_series_equal(float_frame["A"], data["B"], check_names=False)

        msg = "Columns must be same length as key"
        with pytest.raises(ValueError, match=msg):
            data[["A"]] = float_frame[["A", "B"]]
        newcolumndata = range(len(data.index) - 1)
        msg = (
            rf"Length of values \({len(newcolumndata)}\) "
            rf"does not match length of index \({len(data)}\)"
        )
        with pytest.raises(ValueError, match=msg):
            data["A"] = newcolumndata

    def test_setitem_list2(self):
        # [FUNCTION DOC]
        # Verifies label-based setting of a row for a list of columns with integer then string
        # values, reading back a Series named for the row with matching dtype (integer then object).
        df = DataFrame(0, index=range(3), columns=["tt1", "tt2"], dtype=int)
        df.loc[1, ["tt1", "tt2"]] = [1, 2]

        result = df.loc[df.index[1], ["tt1", "tt2"]]
        expected = Series([1, 2], df.columns, dtype=int, name=1)
        tm.assert_series_equal(result, expected)

        df["tt1"] = df["tt2"] = "0"
        df.loc[df.index[1], ["tt1", "tt2"]] = ["1", "2"]
        result = df.loc[df.index[1], ["tt1", "tt2"]]
        expected = Series(["1", "2"], df.columns, name=1)
        tm.assert_series_equal(result, expected)

    def test_getitem_boolean(self, mixed_float_frame, mixed_int_frame, datetime_frame):
        # [FUNCTION DOC]
        # Boolean indexing tests using the datetime, mixed float and mixed int frame fixtures.
        # Checks selection by boolean index; a too short boolean array raises a value error about
        # wrong item length; an object-dtype boolean array and a boolean Series give the same
        # result; a frame as the key raises a type error about a boolean array being expected; a Series
        # key with a reversed index is reindexed with a user warning about reindexing; and for each of
        # the three frames, masking with the positive condition equals a manual where-like frame
        # with missing values (dtype of unchanged columns is preserved).
        # boolean indexing
        d = datetime_frame.index[len(datetime_frame) // 2]
        indexer = datetime_frame.index > d
        indexer_obj = indexer.astype(object)

        subindex = datetime_frame.index[indexer]
        subframe = datetime_frame[indexer]

        tm.assert_index_equal(subindex, subframe.index)
        with pytest.raises(ValueError, match="Item wrong length"):
            datetime_frame[indexer[:-1]]

        subframe_obj = datetime_frame[indexer_obj]
        tm.assert_frame_equal(subframe_obj, subframe)

        with pytest.raises(TypeError, match="Boolean array expected"):
            datetime_frame[datetime_frame]

        # test that Series work
        indexer_obj = Series(indexer_obj, datetime_frame.index)

        subframe_obj = datetime_frame[indexer_obj]
        tm.assert_frame_equal(subframe_obj, subframe)

        # test that Series indexers reindex
        # we are producing a warning that since the passed boolean
        # key is not the same as the given index, we will reindex
        # not sure this is really necessary
        with tm.assert_produces_warning(UserWarning, match="will be reindexed"):
            indexer_obj = indexer_obj.reindex(datetime_frame.index[::-1])
            subframe_obj = datetime_frame[indexer_obj]
            tm.assert_frame_equal(subframe_obj, subframe)

        # test df[df > 0]
        for df in [
            datetime_frame,
            mixed_float_frame,
            mixed_int_frame,
        ]:
            data = df._get_numeric_data()
            bif = df[df > 0]
            bifw = DataFrame(
                {c: np.where(data[c] > 0, data[c], np.nan) for c in data.columns},
                index=data.index,
                columns=data.columns,
            )

            # add back other columns to compare
            for c in df.columns:
                if c not in bifw:
                    bifw[c] = df[c]
            bifw = bifw.reindex(columns=df.columns)

            tm.assert_frame_equal(bif, bifw, check_dtype=False)
            for c in df.columns:
                if bif[c].dtype != bifw[c].dtype:
                    assert bif[c].dtype == df[c].dtype

    def test_getitem_boolean_casting(self, datetime_frame):
        # [FUNCTION DOC]
        # Verifies masking with a positive condition does not upcast integer columns that have no
        # missing result: 32-bit and 64-bit integer columns stay integer, while after setting some
        # rows of a copy of each to zero, those columns are split and become float (integer block
        # splitting).
        # don't upcast if we don't need to
        df = datetime_frame.copy()
        df["E"] = 1
        df["E"] = df["E"].astype("int32")
        df["E1"] = df["E"].copy()
        df["F"] = 1
        df["F"] = df["F"].astype("int64")
        df["F1"] = df["F"].copy()

        casted = df[df > 0]
        result = casted.dtypes
        expected = Series(
            [np.dtype("float64")] * 4
            + [np.dtype("int32")] * 2
            + [np.dtype("int64")] * 2,
            index=["A", "B", "C", "D", "E", "E1", "F", "F1"],
        )
        tm.assert_series_equal(result, expected)

        # int block splitting
        df.loc[df.index[1:3], ["E1", "F1"]] = 0
        casted = df[df > 0]
        result = casted.dtypes
        expected = Series(
            [np.dtype("float64")] * 4
            + [np.dtype("int32")]
            + [np.dtype("float64")]
            + [np.dtype("int64")]
            + [np.dtype("float64")],
            index=["A", "B", "C", "D", "E", "E1", "F", "F1"],
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "lst", [[True, False, True], [True, True, True], [False, False, False]]
    )
    def test_getitem_boolean_list(self, lst):
        # [FUNCTION DOC]
        # Parametrized over three boolean lists (mixed, all true, all false). Verifies row selection by a
        # boolean list equals label-based selection of the chosen positions.
        df = DataFrame(np.arange(12).reshape(3, 4))
        result = df[lst]
        expected = df.loc[df.index[lst]]
        tm.assert_frame_equal(result, expected)

    def test_getitem_boolean_iadd(self):
        # [FUNCTION DOC]
        # Verifies in-place addition through a boolean frame mask adds one only where entries are
        # negative, matching the same operation on the raw array.
        arr = np.random.default_rng(2).standard_normal((5, 5))

        df = DataFrame(arr.copy(), columns=["A", "B", "C", "D", "E"])

        df[df < 0] += 1
        arr[arr < 0] += 1

        tm.assert_almost_equal(df.values, arr)

    def test_boolean_index_empty_corner(self):
        # [FUNCTION DOC]
        # Regression test for issue 2096: an empty float frame with an empty datetime index accepts
        # an empty boolean numpy array as both a getitem key and a setitem key without raising.
        # #2096
        blah = DataFrame(np.empty([0, 1]), columns=["A"], index=DatetimeIndex([]))

        # both of these should succeed trivially
        k = np.array([], bool)

        blah[k]
        blah[k] = 0

    def test_getitem_ix_mixed_integer(self):
        # [FUNCTION DOC]
        # Verifies on a frame with a mixed integer and string row index that positional slicing and
        # label selection with a list of integer labels agree with label-based selection.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 3)),
            index=[1, 10, "C", "E"],
            columns=[1, 2, 3],
        )

        result = df.iloc[:-1]
        expected = df.loc[df.index[:-1]]
        tm.assert_frame_equal(result, expected)

        result = df.loc[[1, 10]]
        expected = df.loc[Index([1, 10])]
        tm.assert_frame_equal(result, expected)

    def test_getitem_ix_mixed_integer2(self):
        # [FUNCTION DOC]
        # Regression test for issue 11320: on a frame with columns of mixed type labels (a string
        # and integers including negative and large values), selecting a list of integer labels returns
        # columns by label, not by position.
        # 11320
        df = DataFrame(
            {
                "rna": (1.5, 2.2, 3.2, 4.5),
                -1000: [11, 21, 36, 40],
                0: [10, 22, 43, 34],
                1000: [0, 10, 20, 30],
            },
            columns=["rna", -1000, 0, 1000],
        )
        result = df[[1000]]
        expected = df.iloc[:, [3]]
        tm.assert_frame_equal(result, expected)
        result = df[[-1000]]
        expected = df.iloc[:, [1]]
        tm.assert_frame_equal(result, expected)

    def test_getattr(self, float_frame):
        # [FUNCTION DOC]
        # Verifies attribute access returns the column Series; a nonexistent attribute raises an
        # attribute error with the standard message.
        tm.assert_series_equal(float_frame.A, float_frame["A"])
        msg = "'DataFrame' object has no attribute 'NONEXISTENT_NAME'"
        with pytest.raises(AttributeError, match=msg):
            float_frame.NONEXISTENT_NAME

    def test_setattr_column(self):
        # [FUNCTION DOC]
        # Verifies assigning through attribute access to an existing column replaces all of its values.
        df = DataFrame({"foobar": 1}, index=range(10))

        df.foobar = 5
        assert (df.foobar == 5).all()

    def test_setitem(self, float_frame, using_infer_string):
        # [FUNCTION DOC]
        # Verifies setitem with a stepped Series (alignment fills missing with NaN), an aligned
        # Series, an array, scalars (integer, float, string) and, on a sliced partial view, a list of
        # strings creating an object (or string, if inferring) column. Row count of the frame stays
        # unchanged.
        # not sure what else to do here
        series = float_frame["A"][::2]
        float_frame["col5"] = series
        assert "col5" in float_frame

        assert len(series) == 15
        assert len(float_frame) == 30

        exp = np.ravel(np.column_stack((series.values, [np.nan] * 15)))
        exp = Series(exp, index=float_frame.index, name="col5")
        tm.assert_series_equal(float_frame["col5"], exp)

        series = float_frame["A"]
        float_frame["col6"] = series
        tm.assert_series_equal(series, float_frame["col6"], check_names=False)

        # set ndarray
        arr = np.random.default_rng(2).standard_normal(len(float_frame))
        float_frame["col9"] = arr
        assert (float_frame["col9"] == arr).all()

        float_frame["col7"] = 5
        assert (float_frame["col7"] == 5).all()

        float_frame["col0"] = 3.14
        assert (float_frame["col0"] == 3.14).all()

        float_frame["col8"] = "foo"
        assert (float_frame["col8"] == "foo").all()

        # this is partially a view (e.g. some blocks are view)
        # so raise/warn
        smaller = float_frame[:2]

        smaller["col10"] = ["1", "2"]

        if using_infer_string:
            assert smaller["col10"].dtype == "str"
        else:
            assert smaller["col10"].dtype == np.object_
        assert (smaller["col10"] == ["1", "2"]).all()

    def test_setitem2(self):
        # [FUNCTION DOC]
        # Regression test for GH4204: assigning not-a-number to the first row by position and by label
        # on an integer frame converts the values to float NaN.
        # dtype changing GH4204
        df = DataFrame([[0, 0]])
        df.iloc[0] = np.nan
        expected = DataFrame([[np.nan, np.nan]])
        tm.assert_frame_equal(df, expected)

        df = DataFrame([[0, 0]])
        df.loc[0] = np.nan
        tm.assert_frame_equal(df, expected)

    def test_setitem_boolean(self, float_frame):
        # [FUNCTION DOC]
        # Verifies boolean setitem forms against a numpy array model: assigning by a boolean Series
        # (rows), a reindexed Series, boolean frames (with greater-than, equal), a frame needing alignment,
        # and a row-reversed frame. A non-boolean frame key raises a type error about needing a frame or
        # a two-dimensional array with boolean values only. Also verifies setting masked positions to
        # NaN and from another frame.
        df = float_frame.copy()
        values = float_frame.values.copy()

        df[df["A"] > 0] = 4
        values[values[:, 0] > 0] = 4
        tm.assert_almost_equal(df.values, values)

        # test that column reindexing works
        series = df["A"] == 4
        series = series.reindex(df.index[::-1])
        df[series] = 1
        values[values[:, 0] == 4] = 1
        tm.assert_almost_equal(df.values, values)

        df[df > 0] = 5
        values[values > 0] = 5
        tm.assert_almost_equal(df.values, values)

        df[df == 5] = 0
        values[values == 5] = 0
        tm.assert_almost_equal(df.values, values)

        # a df that needs alignment first
        df[df[:-1] < 0] = 2
        np.putmask(values[:-1], values[:-1] < 0, 2)
        tm.assert_almost_equal(df.values, values)

        # indexed with same shape but rows-reversed df
        df[df[::-1] == 2] = 3
        values[values == 2] = 3
        tm.assert_almost_equal(df.values, values)

        msg = "Must pass DataFrame or 2-d ndarray with boolean values only"
        with pytest.raises(TypeError, match=msg):
            df[df * 0] = 2

        # index with DataFrame
        df_orig = df.copy()
        mask = df > np.abs(df)
        df[df > np.abs(df)] = np.nan
        values = df_orig.values.copy()
        values[mask.values] = np.nan
        expected = DataFrame(values, index=df_orig.index, columns=df_orig.columns)
        tm.assert_frame_equal(df, expected)

        # set from DataFrame
        df[df > np.abs(df)] = df * 2
        np.putmask(values, mask.values, df.values * 2)
        expected = DataFrame(values, index=df_orig.index, columns=df_orig.columns)
        tm.assert_frame_equal(df, expected)

    def test_setitem_cast(self, float_frame):
        # [FUNCTION DOC]
        # Verifies column dtype follows the assigned value type: casting to 64-bit integer, assigning an
        # integer scalar produces integer (replaces the column rather than casting it), arrays cast,
        # and replacing a string column with an integer or a float scalar gives integer or float dtype.
        float_frame["D"] = float_frame["D"].astype("i8")
        assert float_frame["D"].dtype == np.int64

        # #669, should not cast?
        # this is now set to int64, which means a replacement of the column to
        # the value dtype (and nothing to do with the existing dtype)
        float_frame["B"] = 0
        assert float_frame["B"].dtype == np.int64

        # cast if pass array of course
        float_frame["B"] = np.arange(len(float_frame))
        assert issubclass(float_frame["B"].dtype.type, np.integer)

        float_frame["foo"] = "bar"
        float_frame["foo"] = 0
        assert float_frame["foo"].dtype == np.int64

        float_frame["foo"] = "bar"
        float_frame["foo"] = 2.5
        assert float_frame["foo"].dtype == np.float64

        float_frame["something"] = 0
        assert float_frame["something"].dtype == np.int64
        float_frame["something"] = 2
        assert float_frame["something"].dtype == np.int64
        float_frame["something"] = 2.5
        assert float_frame["something"].dtype == np.float64

    def test_setitem_corner(self, float_frame, using_infer_string):
        # [FUNCTION DOC]
        # Corner cases of setitem on new frames: delete and re-add a column; adding columns with
        # datetime keys; an empty frame with an index gaining string columns (object values) then an
        # integer and a float column (upcast to their dtypes); replacing an existing column; and string
        # columns being object dtype, or the string dtype when inferring strings.
        # corner case
        df = DataFrame({"B": [1.0, 2.0, 3.0], "C": ["a", "b", "c"]}, index=np.arange(3))
        del df["B"]
        df["B"] = [1.0, 2.0, 3.0]
        assert "B" in df
        assert len(df.columns) == 2

        df["A"] = "beginning"
        df["E"] = "foo"
        df["D"] = "bar"
        df[datetime.now()] = "date"
        df[datetime.now()] = 5.0

        # what to do when empty frame with index
        dm = DataFrame(index=float_frame.index)
        dm["A"] = "foo"
        dm["B"] = "bar"
        assert len(dm.columns) == 2
        assert dm.values.dtype == np.object_

        # upcast
        dm["C"] = 1
        assert dm["C"].dtype == np.int64

        dm["E"] = 1.0
        assert dm["E"].dtype == np.float64

        # set existing column
        dm["A"] = "bar"
        assert "bar" == dm["A"].iloc[0]

        dm = DataFrame(index=np.arange(3))
        dm["A"] = 1
        dm["foo"] = "bar"
        del dm["foo"]
        dm["foo"] = "bar"
        if using_infer_string:
            assert dm["foo"].dtype == "str"
        else:
            assert dm["foo"].dtype == np.object_

        dm["coercible"] = ["1", "2", "3"]
        if using_infer_string:
            assert dm["coercible"].dtype == "str"
        else:
            assert dm["coercible"].dtype == np.object_

    def test_setitem_corner2(self):
        # [FUNCTION DOC]
        # Verifies setting values through a label list in a frame with duplicate title text and a numeric
        # column: a row selected by a condition has its title and numeric entries replaced.
        data = {
            "title": ["foobar", "bar", "foobar"] + ["foobar"] * 17,
            "cruft": np.random.default_rng(2).random(20),
        }

        df = DataFrame(data)
        ix = df[df["title"] == "bar"].index

        df.loc[ix, ["title"]] = "foobar"
        df.loc[ix, ["cruft"]] = 0

        assert df.loc[1, "title"] == "foobar"
        assert df.loc[1, "cruft"] == 0

    def test_setitem_ambig(self, using_infer_string):
        # [FUNCTION DOC]
        # Verifies setting columns of an empty-valued frame (object columns) from a float array, a
        # decimal Series and a string Series keeps three columns; the string one is object (or string).
        # Difficulties with mixed-type data
        # Created as float type
        dm = DataFrame(index=range(3), columns=range(3))

        coercable_series = Series([Decimal(1) for _ in range(3)], index=range(3))
        uncoercable_series = Series(["foo", "bzr", "baz"], index=range(3))

        dm[0] = np.ones(3)
        assert len(dm.columns) == 3

        dm[1] = coercable_series
        assert len(dm.columns) == 3

        dm[2] = uncoercable_series
        assert len(dm.columns) == 3
        if using_infer_string:
            assert dm[2].dtype == "str"
        else:
            assert dm[2].dtype == np.object_

    def test_setitem_None(self, float_frame):
        # [FUNCTION DOC]
        # Regression test for GH 766: assigning a column using None as the label works, and the new
        # column can be retrieved by position, by label and by item with the same data.
        # GH #766
        float_frame[None] = float_frame["A"]
        tm.assert_series_equal(
            float_frame.iloc[:, -1], float_frame["A"], check_names=False
        )
        tm.assert_series_equal(
            float_frame.loc[:, None], float_frame["A"], check_names=False
        )
        tm.assert_series_equal(float_frame[None], float_frame["A"], check_names=False)

    def test_loc_setitem_boolean_mask_allfalse(self):
        # [FUNCTION DOC]
        # Regression test for GH 9596: assigning a column onto itself by label with an all-false
        # boolean mask (derived from a missing-value check) on a frame of string columns leaves the frame unchanged.
        # GH 9596
        df = DataFrame(
            {"a": ["1", "2", "3"], "b": ["11", "22", "33"], "c": ["111", "222", "333"]}
        )

        result = df.copy()
        result.loc[result.b.isna(), "a"] = result.a.copy()
        tm.assert_frame_equal(result, df)

    def test_getitem_slice_empty(self):
        # [FUNCTION DOC]
        # Verifies full slicing of a frame with multi-index columns returns an equal frame that is a
        # different object than the original.
        df = DataFrame([[1]], columns=MultiIndex.from_product([["A"], ["a"]]))
        result = df[:]

        expected = DataFrame([[1]], columns=MultiIndex.from_product([["A"], ["a"]]))

        tm.assert_frame_equal(result, expected)
        # Ensure df[:] returns a view of df, not the same object
        assert result is not df

    def test_getitem_fancy_slice_integers_step(self):
        # [FUNCTION DOC]
        # Verifies a stepped positional slice can be read and assigned missing values, and the assigned
        # rows are all missing.
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))

        # this is OK
        df.iloc[:8:2]
        df.iloc[:8:2] = np.nan
        assert isna(df.iloc[:8:2]).values.all()

    def test_getitem_setitem_integer_slice_keyerrors(self):
        # [FUNCTION DOC]
        # Verifies positional slice assignment and reading with out-of-range bounds on a frame indexed
        # by even integers, and that label slicing between absent labels selects intermediate present
        # labels. On a non-monotonic index, label slicing or slice assignment with a bound missing from
        # the index raises a key error mentioning a non-monotonic index with a missing label.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 5)), index=range(0, 20, 2)
        )

        # this is OK
        cp = df.copy()
        cp.iloc[4:10] = 0
        assert (cp.iloc[4:10] == 0).values.all()

        # so is this
        cp = df.copy()
        cp.iloc[3:11] = 0
        assert (cp.iloc[3:11] == 0).values.all()

        result = df.iloc[2:6]
        result2 = df.loc[3:11]
        expected = df.reindex([4, 6, 8, 10])

        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

        # non-monotonic, raise KeyError
        df2 = df.iloc[list(range(5)) + list(range(5, 10))[::-1]]
        msg = "non-monotonic index with a missing label 3"
        with pytest.raises(KeyError, match=msg):
            df2.loc[3:11]
        with pytest.raises(KeyError, match=msg):
            df2.loc[3:11] = 0

    def test_fancy_getitem_slice_mixed(self, float_frame, float_string_frame):
        # [FUNCTION DOC]
        # Verifies a column slice of a mixed float and string frame keeps float dtype for a float
        # column, and that a column slice of the single-block float frame shares memory with it, so that
        # assigning through the slice (which triggers copy on write) leaves the original frame unchanged.
        sliced = float_string_frame.iloc[:, -3:]
        assert sliced["D"].dtype == np.float64

        # get view with single block
        # setting it triggers setting with copy
        original = float_frame.copy()
        sliced = float_frame.iloc[:, -3:]

        assert np.shares_memory(sliced["C"]._values, float_frame["C"]._values)

        sliced.loc[:, "C"] = 4.0
        tm.assert_frame_equal(float_frame, original)

    def test_getitem_setitem_non_ix_labels(self):
        # [FUNCTION DOC]
        # Verifies on a date-indexed frame that label slicing via label location and via plain getitem
        # (date bounds, inclusive) equals positional slicing, and the same for slice assignment.
        df = DataFrame(range(20), index=date_range("2020-01-01", periods=20))

        start, end = df.index[[5, 10]]

        result = df.loc[start:end]
        result2 = df[start:end]
        expected = df[5:11]
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

        result = df.copy()
        result.loc[start:end] = 0
        result2 = df.copy()
        result2[start:end] = 0
        expected = df.copy()
        expected[5:11] = 0
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

    def test_ix_multi_take(self):
        # [FUNCTION DOC]
        # Verifies selecting with a boolean array for rows and a boolean array for columns via label
        # location equals reindexing; second part regression for GH#1321 combining a row and column condition.
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 2)))
        rs = df.loc[df.index == 0, :]
        xp = df.reindex([0])
        tm.assert_frame_equal(rs, xp)

        # GH#1321
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 2)))
        rs = df.loc[df.index == 0, df.columns == 1]
        xp = df.reindex(index=[0], columns=[1])
        tm.assert_frame_equal(rs, xp)

    def test_getitem_fancy_scalar(self, float_frame):
        # [FUNCTION DOC]
        # Verifies label location of scalar row and column pairs equals Series lookup, over every fifth row.
        f = float_frame
        ix = f.loc

        # individual value
        for col in f.columns:
            ts = f[col]
            for idx in f.index[::5]:
                assert ix[idx, col] == ts[idx]

    def test_setitem_fancy_scalar(self, float_frame):
        # [FUNCTION DOC]
        # Verifies scalar assignment through label location, at every fifth row of each column, equals
        # positional assignment on a copy of the frame.
        f = float_frame
        expected = float_frame.copy()
        ix = f.loc

        # individual value
        for j, col in enumerate(f.columns):
            f[col]
            for idx in f.index[::5]:
                i = f.index.get_loc(idx)
                val = np.random.default_rng(2).standard_normal()
                expected.iloc[i, j] = val

                ix[idx, col] = val
                tm.assert_frame_equal(f, expected)

    def test_getitem_fancy_boolean(self, float_frame):
        # [FUNCTION DOC]
        # Verifies label location with boolean lists and boolean arrays for columns, rows, and combined
        # with a column label slice, each equal to reindexing results.
        f = float_frame
        ix = f.loc

        expected = f.reindex(columns=["B", "D"])
        result = ix[:, [False, True, False, True]]
        tm.assert_frame_equal(result, expected)

        expected = f.reindex(index=f.index[5:10], columns=["B", "D"])
        result = ix[f.index[5:10], [False, True, False, True]]
        tm.assert_frame_equal(result, expected)

        boolvec = f.index > f.index[7]
        expected = f.reindex(index=f.index[boolvec])
        result = ix[boolvec]
        tm.assert_frame_equal(result, expected)
        result = ix[boolvec, :]
        tm.assert_frame_equal(result, expected)

        result = ix[boolvec, f.columns[2:]]
        expected = f.reindex(index=f.index[boolvec], columns=["C", "D"])
        tm.assert_frame_equal(result, expected)

    def test_setitem_fancy_boolean(self, float_frame):
        # [FUNCTION DOC]
        # Verifies assigning zero through label location with a boolean row mask, over all columns and
        # over two columns, equals the same change on a numpy array copy.
        # from 2d, set with booleans
        frame = float_frame.copy()
        expected = float_frame.copy()
        values = expected.values.copy()

        mask = frame["A"] > 0
        frame.loc[mask] = 0.0
        values[mask.values] = 0.0
        expected = DataFrame(values, index=expected.index, columns=expected.columns)
        tm.assert_frame_equal(frame, expected)

        frame = float_frame.copy()
        expected = float_frame.copy()
        values = expected.values.copy()
        frame.loc[mask, ["A", "B"]] = 0.0
        values[mask.values, :2] = 0.0
        expected = DataFrame(values, index=expected.index, columns=expected.columns)
        tm.assert_frame_equal(frame, expected)

    def test_getitem_fancy_ints(self, float_frame):
        # [FUNCTION DOC]
        # Verifies positional selection by integer list for rows or columns equals label-based selection
        # of the same positions.
        result = float_frame.iloc[[1, 4, 7]]
        expected = float_frame.loc[float_frame.index[[1, 4, 7]]]
        tm.assert_frame_equal(result, expected)

        result = float_frame.iloc[:, [2, 0, 1]]
        expected = float_frame.loc[:, float_frame.columns[[2, 0, 1]]]
        tm.assert_frame_equal(result, expected)

    def test_getitem_setitem_boolean_misaligned(self, float_frame):
        # [FUNCTION DOC]
        # Verifies a boolean Series with a reversed index used with label location is aligned to the
        # frame for reading and assigning; the result equals using the reordered mask.
        # boolean index misaligned labels
        mask = float_frame["A"][::-1] > 1

        result = float_frame.loc[mask]
        expected = float_frame.loc[mask[::-1]]
        tm.assert_frame_equal(result, expected)

        cp = float_frame.copy()
        expected = float_frame.copy()
        cp.loc[mask] = 0
        expected.loc[mask] = 0
        tm.assert_frame_equal(cp, expected)

    def test_getitem_setitem_boolean_multi(self):
        # [FUNCTION DOC]
        # Verifies label location with two boolean numpy arrays (rows and columns) selects and assigns
        # the intersection, equal to selecting by the corresponding lists of labels.
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 2)))

        # get
        k1 = np.array([True, False, True])
        k2 = np.array([False, True])
        result = df.loc[k1, k2]
        expected = df.loc[[0, 2], [1]]
        tm.assert_frame_equal(result, expected)

        expected = df.copy()
        df.loc[np.array([True, False, True]), np.array([False, True])] = 5
        expected.loc[[0, 2], [1]] = 5
        tm.assert_frame_equal(df, expected)

    def test_getitem_float_label_positional(self):
        # [FUNCTION DOC]
        # Regression test for GH 53338: slicing with integers on a frame with a float index behaves
        # positionally (slice from one to two selects the second row) and keeps the float index.
        # GH 53338
        index = Index([1.5, 2])
        df = DataFrame(range(2), index=index)
        result = df[1:2]
        expected = DataFrame([1], index=[2.0])
        tm.assert_frame_equal(result, expected)

    def test_getitem_setitem_float_labels(self):
        # [FUNCTION DOC]
        # Verifies label and positional slicing with float labels on a float index: label slices include
        # bounds, may use integer bounds, and compare to reindex results. For an index of non-integral
        # floats, positional slicing with a float bound raises a type error about positional indexing
        # with float indexers (message changes with interpreter version via the module constant for
        # assignment), positional integer slices read and assign correctly, and label float bounds select
        # the expected ranges (issue 2727).
        # KNOWN ISSUE: the later check of the final result is outside the context that raises: the
        # comparison inside the error-raising block never assigns the name, so the assertion uses a
        # stale value from an earlier unrelated slice.
        index = Index([1.5, 2, 3, 4, 5])
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)), index=index)

        result = df.loc[1.5:4]
        expected = df.reindex([1.5, 2, 3, 4])
        tm.assert_frame_equal(result, expected)
        assert len(result) == 4

        result = df.loc[4:5]
        expected = df.reindex([4, 5])  # reindex with int
        tm.assert_frame_equal(result, expected, check_index_type=False)
        assert len(result) == 2

        result = df.loc[4:5]
        expected = df.reindex([4.0, 5.0])  # reindex with float
        tm.assert_frame_equal(result, expected)
        assert len(result) == 2

        # loc_float changes this to work properly
        result = df.loc[1:2]
        expected = df.iloc[0:2]
        tm.assert_frame_equal(result, expected)

        # #2727
        index = Index([1.0, 2.5, 3.5, 4.5, 5.0])
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)), index=index)

        # positional slicing only via iloc!
        msg = (
            "cannot do positional indexing on Index with "
            r"these indexers \[1.0\] of type float"
        )
        with pytest.raises(TypeError, match=msg):
            df.iloc[1.0:5]

        result = df.iloc[4:5]
        expected = df.reindex([5.0])
        tm.assert_frame_equal(result, expected)
        assert len(result) == 1

        cp = df.copy()

        with pytest.raises(TypeError, match=_slice_msg):
            cp.iloc[1.0:5] = 0

        with pytest.raises(TypeError, match=msg):
            result = cp.iloc[1.0:5] == 0

        assert result.values.all()
        assert (cp.iloc[0:1] == df.iloc[0:1]).values.all()

        cp = df.copy()
        cp.iloc[4:5] = 0
        assert (cp.iloc[4:5] == 0).values.all()
        assert (cp.iloc[0:4] == df.iloc[0:4]).values.all()

        # float slicing
        result = df.loc[1.0:5]
        expected = df
        tm.assert_frame_equal(result, expected)
        assert len(result) == 5

        result = df.loc[1.1:5]
        expected = df.reindex([2.5, 3.5, 4.5, 5.0])
        tm.assert_frame_equal(result, expected)
        assert len(result) == 4

        result = df.loc[4.51:5]
        expected = df.reindex([5.0])
        tm.assert_frame_equal(result, expected)
        assert len(result) == 1

        result = df.loc[1.0:5.0]
        expected = df.reindex([1.0, 2.5, 3.5, 4.5, 5.0])
        tm.assert_frame_equal(result, expected)
        assert len(result) == 5

        cp = df.copy()
        cp.loc[1.0:5.0] = 0
        result = cp.loc[1.0:5.0]
        assert (result == 0).values.all()

    def test_setitem_single_column_mixed_datetime(self):
        # [FUNCTION DOC]
        # Verifies adding a column of seconds-unit timestamps to a float frame gives a datetime64
        # second-unit dtype; assigning the integer sentinel for not-a-time raises a type error about an
        # invalid value (GH#16674); assigning NaN to one cell or a whole row is allowed (GH#3216).
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)),
            index=["a", "b", "c", "d", "e"],
            columns=["foo", "bar", "baz"],
        )

        df["timestamp"] = Timestamp("20010102").as_unit("s")

        # check our dtypes
        result = df.dtypes
        expected = Series(
            [np.dtype("float64")] * 3 + [np.dtype("datetime64[s]")],
            index=["foo", "bar", "baz", "timestamp"],
        )
        tm.assert_series_equal(result, expected)

        # GH#16674 iNaT is treated as an integer when given by the user
        with pytest.raises(TypeError, match="Invalid value"):
            df.loc["b", "timestamp"] = iNaT

        # allow this syntax (as of GH#3216)
        df.loc["c", "timestamp"] = np.nan
        assert isna(df.loc["c", "timestamp"])

        # allow this syntax
        df.loc["d", :] = np.nan
        assert not isna(df.loc["c", :]).all()

    def test_setitem_mixed_datetime(self):
        # [FUNCTION DOC]
        # Regression test for GH 9336: a column set to NaT accepts a datetime for one cell, and
        # assigning an integer raises a type error about an invalid value.
        # GH 9336
        df = DataFrame(0, columns=list("ab"), index=range(6))
        df["b"] = pd.NaT
        df.loc[0, "b"] = datetime(2012, 1, 1)
        with pytest.raises(TypeError, match="Invalid value"):
            df.loc[1, "b"] = 1

    def test_setitem_frame_float(self, float_frame):
        # [FUNCTION DOC]
        # Verifies assigning values from the first two rows of two columns into the last two rows
        # through label location (values only, no alignment) gives matching values.
        piece = float_frame.loc[float_frame.index[:2], ["A", "B"]]
        float_frame.loc[float_frame.index[-2] :, ["A", "B"]] = piece.values
        result = float_frame.loc[float_frame.index[-2:], ["A", "B"]].values
        expected = piece.values
        tm.assert_almost_equal(result, expected)

    def test_setitem_frame_mixed(self, float_string_frame):
        # [FUNCTION DOC]
        # Regression test for GH 3216: assigning a two-by-two frame aligned on both axes into a mixed
        # float and string frame via a tuple key of labels sets the values.
        # GH 3216

        # already aligned
        f = float_string_frame.copy()
        piece = DataFrame(
            [[1.0, 2.0], [3.0, 4.0]], index=f.index[0:2], columns=["A", "B"]
        )
        key = (f.index[slice(None, 2)], ["A", "B"])
        f.loc[key] = piece
        tm.assert_almost_equal(f.loc[f.index[0:2], ["A", "B"]].values, piece.values)

    def test_setitem_frame_mixed_rows_unaligned(self, float_string_frame):
        # [FUNCTION DOC]
        # Regression test for GH#3216: assigning a four-row frame with only two rows matching the target
        # rows (the other labels are foo and bar) sets matching rows; extra rows ignored.
        # GH#3216 rows unaligned
        f = float_string_frame.copy()
        piece = DataFrame(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            index=[*list(f.index[0:2]), "foo", "bar"],
            columns=["A", "B"],
        )
        key = (f.index[slice(None, 2)], ["A", "B"])
        f.loc[key] = piece
        tm.assert_almost_equal(
            f.loc[f.index[0:2:], ["A", "B"]].values, piece.values[0:2]
        )

    def test_setitem_frame_mixed_key_unaligned(self, float_string_frame):
        # [FUNCTION DOC]
        # Regression test for GH#3216: when the key rows differ from the value frame's index, assignment
        # aligns on labels, so the missing second column becomes missing values in the target.
        # GH#3216 key is unaligned with values
        f = float_string_frame.copy()
        piece = f.loc[f.index[:2], ["A"]]
        piece.index = f.index[-2:]
        key = (f.index[slice(-2, None)], ["A", "B"])
        f.loc[key] = piece
        piece["B"] = np.nan
        tm.assert_almost_equal(f.loc[f.index[-2:], ["A", "B"]].values, piece.values)

    def test_setitem_frame_mixed_ndarray(self, float_string_frame):
        # [FUNCTION DOC]
        # Regression test for GH#3216: assigning a plain numpy array into the last two rows of two
        # columns bypasses alignment and sets values positionally.
        # GH#3216 ndarray
        f = float_string_frame.copy()
        piece = float_string_frame.loc[f.index[:2], ["A", "B"]]
        key = (f.index[slice(-2, None)], ["A", "B"])
        f.loc[key] = piece.values
        tm.assert_almost_equal(f.loc[f.index[-2:], ["A", "B"]].values, piece.values)

    def test_setitem_frame_upcast(self):
        # [FUNCTION DOC]
        # Verifies assigning a float frame into integer columns via label location raises a type error
        # about an invalid value; after explicitly casting both frames' columns to float it succeeds and
        # equals the expected frame.
        # needs upcasting
        df = DataFrame([[1, 2, "foo"], [3, 4, "bar"]], columns=["A", "B", "C"])
        df2 = df.copy()
        with pytest.raises(TypeError, match="Invalid value"):
            df2.loc[:, ["A", "B"]] = df.loc[:, ["A", "B"]] + 0.5
        # Manually upcast so we can add .5
        df = df.astype({"A": "float64", "B": "float64"})
        df2 = df2.astype({"A": "float64", "B": "float64"})
        df2.loc[:, ["A", "B"]] = df.loc[:, ["A", "B"]] + 0.5
        expected = df.reindex(columns=["A", "B"])
        expected += 0.5
        expected["C"] = df["C"]
        tm.assert_frame_equal(df2, expected)

    def test_setitem_frame_align(self, float_frame):
        # [FUNCTION DOC]
        # Verifies assigning a frame whose index is relabeled to the target rows aligns and sets values.
        piece = float_frame.loc[float_frame.index[:2], ["A", "B"]]
        piece.index = float_frame.index[-2:]
        piece.columns = ["A", "B"]
        float_frame.loc[float_frame.index[-2:], ["A", "B"]] = piece
        result = float_frame.loc[float_frame.index[-2:], ["A", "B"]].values
        expected = piece.values
        tm.assert_almost_equal(result, expected)

    def test_getitem_setitem_ix_duplicates(self):
        # [FUNCTION DOC]
        # Regression test for issue 1201: label location on an index with duplicates returns a frame
        # for repeated labels and a Series for a unique label.
        # #1201
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)),
            index=["foo", "foo", "bar", "baz", "bar"],
        )

        result = df.loc["foo"]
        expected = df[:2]
        tm.assert_frame_equal(result, expected)

        result = df.loc["bar"]
        expected = df.iloc[[2, 4]]
        tm.assert_frame_equal(result, expected)

        result = df.loc["baz"]
        expected = df.iloc[3]
        tm.assert_series_equal(result, expected)

    def test_getitem_ix_boolean_duplicates_multiple(self):
        # [FUNCTION DOC]
        # Regression test for issue 1201: label location with a list of duplicate labels and with
        # boolean Series conditions on a frame with duplicate index labels equals the matching row selections.
        # #1201
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)),
            index=["foo", "foo", "bar", "baz", "bar"],
        )

        result = df.loc[["bar"]]
        exp = df.iloc[[2, 4]]
        tm.assert_frame_equal(result, exp)

        result = df.loc[df[1] > 0]
        exp = df[df[1] > 0]
        tm.assert_frame_equal(result, exp)

        result = df.loc[df[0] > 0]
        exp = df[df[0] > 0]
        tm.assert_frame_equal(result, exp)

    @pytest.mark.parametrize("bool_value", [True, False])
    def test_getitem_setitem_ix_bool_keyerror(self, bool_value):
        # [FUNCTION DOC]
        # Regression test for issue 2199, parametrized over true and false. A single boolean used as a
        # label raises a key error saying a boolean label cannot be used without a boolean index, and
        # assigning to it raises a key error saying a single bool cannot index setitem.
        # #2199
        df = DataFrame({"a": [1, 2, 3]})
        message = f"{bool_value}: boolean label can not be used without a boolean index"
        with pytest.raises(KeyError, match=message):
            df.loc[bool_value]

        msg = "cannot use a single bool to index into setitem"
        with pytest.raises(KeyError, match=msg):
            df.loc[bool_value] = 0

    # TODO: rename?  remove?
    def test_single_element_ix_dont_upcast(self, float_frame):
        # [FUNCTION DOC]
        # Verifies a column set to an integer stays integer and a scalar read from it is an integer, and
        # GH 11617: a list lookup of that single element returns an integer Series (no upcast to float).
        # A source comment questions whether the test should be renamed or removed.
        float_frame["E"] = 1
        assert issubclass(float_frame["E"].dtype.type, (int, np.integer))

        result = float_frame.loc[float_frame.index[5], "E"]
        assert is_integer(result)

        # GH 11617
        df = DataFrame({"a": [1.23]})
        df["b"] = 666

        result = df.loc[0, "b"]
        assert is_integer(result)

        expected = Series([666], index=range(1), name="b")
        result = df.loc[[0], "b"]
        tm.assert_series_equal(result, expected)

    def test_iloc_callable_tuple_return_value_raises(self):
        # [FUNCTION DOC]
        # Enforced in version 3.0 (GH53769): positional indexing with a callable returning a tuple
        # raises a value error for both reading and assigning, with a message starting with the
        # returning-a-tuple text.
        # GH53769: Enforced pandas 3.0
        df = DataFrame(np.arange(40).reshape(10, 4), index=range(0, 20, 2))
        msg = "Returning a tuple from"
        with pytest.raises(ValueError, match=msg):
            df.iloc[lambda _: (0,)]
        with pytest.raises(ValueError, match=msg):
            df.iloc[lambda _: (0,)] = 1

    def test_iloc_row(self):
        # [FUNCTION DOC]
        # Verifies positional row selection by integer, by slice and by integer list equals the label
        # selection on a frame indexed by even integers.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)), index=range(0, 20, 2)
        )

        result = df.iloc[1]
        exp = df.loc[2]
        tm.assert_series_equal(result, exp)

        result = df.iloc[2]
        exp = df.loc[4]
        tm.assert_series_equal(result, exp)

        # slice
        result = df.iloc[slice(4, 8)]
        expected = df.loc[8:14]
        tm.assert_frame_equal(result, expected)

        # list of integers
        result = df.iloc[[1, 2, 4, 6]]
        expected = df.reindex(df.index[[1, 2, 4, 6]])
        tm.assert_frame_equal(result, expected)

    def test_iloc_row_slice_view(self):
        # [FUNCTION DOC]
        # Verifies a positional row slice of a frame indexed by even integers shares memory with the
        # parent column; assigning to the subset through label location (copy on write) leaves the parent unchanged.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)), index=range(0, 20, 2)
        )
        original = df.copy()

        # verify slice is view
        # setting it makes it raise/warn
        subset = df.iloc[slice(4, 8)]

        assert np.shares_memory(df[2], subset[2])

        exp_col = original[2].copy()
        subset.loc[:, 2] = 0.0
        tm.assert_series_equal(df[2], exp_col)

    def test_iloc_col(self):
        # [FUNCTION DOC]
        # Verifies positional column selection by integer, slice and list on a frame with even integer
        # column labels equals the label-based selection.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 10)), columns=range(0, 20, 2)
        )

        result = df.iloc[:, 1]
        exp = df.loc[:, 2]
        tm.assert_series_equal(result, exp)

        result = df.iloc[:, 2]
        exp = df.loc[:, 4]
        tm.assert_series_equal(result, exp)

        # slice
        result = df.iloc[:, slice(4, 8)]
        expected = df.loc[:, 8:14]
        tm.assert_frame_equal(result, expected)

        # list of integers
        result = df.iloc[:, [1, 2, 4, 6]]
        expected = df.reindex(columns=df.columns[[1, 2, 4, 6]])
        tm.assert_frame_equal(result, expected)

    def test_iloc_col_slice_view(self):
        # [FUNCTION DOC]
        # Verifies a positional column slice is initially a view (shares memory with the parent), but
        # assigning a new column in the subset replaces the whole column in the subset only: the subset
        # changes while the parent equals its original copy.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 10)), columns=range(0, 20, 2)
        )
        original = df.copy()
        subset = df.iloc[:, slice(4, 8)]

        # verify slice is view
        assert np.shares_memory(df[8]._values, subset[8]._values)
        subset[8] = 0.0
        # subset changed
        assert (subset[8] == 0).all()
        # but df itself did not change (setitem replaces full column)
        tm.assert_frame_equal(df, original)

    def test_loc_duplicates(self):
        # [FUNCTION DOC]
        # Regression test for gh-17105: a datetime index with an inserted duplicate final date; assigning
        # with a boolean mask selecting the duplicated label sets both matching rows, and in-place
        # addition produces the same expected frame.
        # gh-17105

        # insert a duplicate element to the index
        trange = date_range(
            start=Timestamp(year=2017, month=1, day=1),
            end=Timestamp(year=2017, month=1, day=5),
        )

        trange = trange.insert(loc=5, item=Timestamp(year=2017, month=1, day=5))

        df = DataFrame(0, index=trange, columns=["A", "B"])
        bool_idx = np.array([False, False, False, False, False, True])

        # assignment
        df.loc[trange[bool_idx], "A"] = 6

        expected = DataFrame(
            {"A": [0, 0, 0, 0, 6, 6], "B": [0, 0, 0, 0, 0, 0]}, index=trange
        )
        tm.assert_frame_equal(df, expected)

        # in-place
        df = DataFrame(0, index=trange, columns=["A", "B"])
        df.loc[trange[bool_idx], "A"] += 6
        tm.assert_frame_equal(df, expected)

    def test_setitem_with_unaligned_tz_aware_datetime_column(self):
        # [FUNCTION DOC]
        # Regression test for GH 12981: assigning a timezone-aware UTC datetime Series with shuffled index
        # (via setitem or label location) aligns on the index and keeps the time zone.
        # GH 12981
        # Assignment of unaligned offset-aware datetime series.
        # Make sure timezone isn't lost
        column = Series(date_range("2015-01-01", periods=3, tz="utc"), name="dates")
        df = DataFrame({"dates": column})
        df["dates"] = column[[1, 0, 2]]
        tm.assert_series_equal(df["dates"], column)

        df = DataFrame({"dates": column})
        df.loc[[0, 1, 2], "dates"] = column[[1, 0, 2]]
        tm.assert_series_equal(df["dates"], column)

    def test_loc_setitem_datetimelike_with_inference(self):
        # [FUNCTION DOC]
        # Regression test for GH 7592: building timedelta and datetime columns incrementally with various
        # setters (full column, partial rows creating missing values, NaT scalars) infers six timedelta
        # nanosecond columns and two datetime nanosecond columns.
        # GH 7592
        # assignment of timedeltas with NaT

        one_hour = timedelta(hours=1)
        df = DataFrame(index=date_range("20130101", periods=4, unit="ns"))
        df["A"] = np.array([1 * one_hour] * 4, dtype="m8[ns]")
        df.loc[:, "B"] = np.array([2 * one_hour] * 4, dtype="m8[ns]")
        df.loc[df.index[:3], "C"] = np.array([3 * one_hour] * 3, dtype="m8[ns]")
        df.loc[:, "D"] = np.array([4 * one_hour] * 4, dtype="m8[ns]")
        df.loc[df.index[:3], "E"] = np.array([5 * one_hour] * 3, dtype="m8[ns]")
        df["F"] = np.timedelta64("NaT", "ns")
        df.loc[df.index[:-1], "F"] = np.array([6 * one_hour] * 3, dtype="m8[ns]")
        df.loc[df.index[-3] :, "G"] = date_range("20130101", periods=3, unit="ns")
        df["H"] = np.datetime64("NaT", "ns")
        result = df.dtypes
        expected = Series(
            [np.dtype("timedelta64[ns]")] * 6 + [np.dtype("datetime64[ns]")] * 2,
            index=list("ABCDEFGH"),
        )
        tm.assert_series_equal(result, expected)

    def test_getitem_boolean_indexing_mixed(self):
        # [FUNCTION DOC]
        # Verifies boolean-frame setitem on a mixed int/float frame with many missing values: entries
        # above a threshold are set to one, equal to explicit expected edits. After adding a string column,
        # the same assignment raises a type error about unsupported or invalid comparison.
        df = DataFrame(
            {
                0: {35: np.nan, 40: np.nan, 43: np.nan, 49: np.nan, 50: np.nan},
                1: {
                    35: np.nan,
                    40: 0.32632316859446198,
                    43: np.nan,
                    49: 0.32632316859446198,
                    50: 0.39114724480578139,
                },
                2: {
                    35: np.nan,
                    40: np.nan,
                    43: 0.29012581014105987,
                    49: np.nan,
                    50: np.nan,
                },
                3: {35: np.nan, 40: np.nan, 43: np.nan, 49: np.nan, 50: np.nan},
                4: {
                    35: 0.34215328467153283,
                    40: np.nan,
                    43: np.nan,
                    49: np.nan,
                    50: np.nan,
                },
                "y": {35: 0, 40: 0, 43: 0, 49: 0, 50: 1},
            }
        )

        # mixed int/float ok
        df2 = df.copy()
        df2[df2 > 0.3] = 1
        expected = df.copy()
        expected.loc[40, 1] = 1
        expected.loc[49, 1] = 1
        expected.loc[50, 1] = 1
        expected.loc[35, 4] = 1
        tm.assert_frame_equal(df2, expected)

        df["foo"] = "test"
        msg = "not supported between instances|unorderable types|Invalid comparison"

        with pytest.raises(TypeError, match=msg):
            df[df > 0.3] = 1

    def test_type_error_multiindex(self):
        # [FUNCTION DOC]
        # Regression test for gh-12218: on a frame with two-level columns, getitem with a slice and
        # scalar tuple raises an invalid-index error mentioning slice; label location with a slice and
        # level value selects matching columns keeping level names; a tuple key of two scalars gives a Series named by the tuple.
        # See gh-12218
        mi = MultiIndex.from_product([["x", "y"], [0, 1]], names=[None, "c"])
        dg = DataFrame(
            [[1, 1, 2, 2], [3, 3, 4, 4]], columns=mi, index=Index(range(2), name="i")
        )
        with pytest.raises(InvalidIndexError, match="slice"):
            dg[:, 0]

        index = Index(range(2), name="i")
        columns = MultiIndex(
            levels=[["x", "y"], [0, 1]], codes=[[0, 1], [0, 0]], names=[None, "c"]
        )
        expected = DataFrame([[1, 2], [3, 4]], columns=columns, index=index)

        result = dg.loc[:, (slice(None), 0)]
        tm.assert_frame_equal(result, expected)

        name = ("x", 0)
        index = Index(range(2), name="i")
        expected = Series([1, 3], index=index, name=name)

        result = dg["x", 0]
        tm.assert_series_equal(result, expected)

    def test_getitem_interval_index_partial_indexing(self):
        # [FUNCTION DOC]
        # Regression test for GH#36490: a frame with an interval index for columns selected with a
        # scalar inside an interval returns that column, both by getitem and label location.
        # GH#36490
        df = DataFrame(
            np.ones((3, 4)), columns=pd.IntervalIndex.from_breaks(np.arange(5))
        )

        expected = df.iloc[:, 0]

        res = df[0.5]
        tm.assert_series_equal(res, expected)

        res = df.loc[:, 0.5]
        tm.assert_series_equal(res, expected)

    def test_setitem_array_as_cell_value(self):
        # [FUNCTION DOC]
        # Regression test for GH#43422: assigning a row dictionary whose values are numpy arrays of
        # different shapes into an empty object frame stores each array as a single cell value.
        # GH#43422
        df = DataFrame(columns=["a", "b"], dtype=object)
        df.loc[0] = {"a": np.zeros((2,)), "b": np.zeros((2, 2))}
        expected = DataFrame({"a": [np.zeros((2,))], "b": [np.zeros((2, 2))]})
        tm.assert_frame_equal(df, expected)

    def test_iloc_setitem_nullable_2d_values(self):
        # [FUNCTION DOC]
        # Verifies assigning the reversed values, a numpy extension array, or a copy of the frame into
        # all rows of a masked integer frame leaves it unchanged (single column so reversal is a no-op).
        df = DataFrame({"A": [1, 2, 3]}, dtype="Int64")
        orig = df.copy()

        df.loc[:] = df.values[:, ::-1]
        tm.assert_frame_equal(df, orig)

        df.loc[:] = pd.core.arrays.NumpyExtensionArray(df.values[:, ::-1])
        tm.assert_frame_equal(df, orig)

        df.iloc[:] = df.iloc[:, :].copy()
        tm.assert_frame_equal(df, orig)

    def test_getitem_segfault_with_empty_like_object(self):
        # [FUNCTION DOC]
        # Regression test for GH#46848 (segfault): assigning an empty-like object array to a column and
        # then selecting it with a list must not crash.
        # GH#46848
        df = DataFrame(np.empty((1, 1), dtype=object))
        df[0] = np.empty_like(df[0])
        # this produces the segfault
        df[[0]]

    @pytest.mark.filterwarnings("ignore:Setting a value on a view:FutureWarning")
    @pytest.mark.parametrize(
        "null", [pd.NaT, pd.NaT.to_numpy("M8[ns]"), pd.NaT.to_numpy("m8[ns]")]
    )
    def test_setting_mismatched_na_into_nullable_fails(
        self, null, any_numeric_ea_dtype
    ):
        # [FUNCTION DOC]
        # Regression test for GH#44514, parametrized over three null kinds (NaT, a datetime64 NaT and a
        # timedelta64 NaT) and every numeric extension dtype: setting such a mismatched null into a
        # masked numeric array, Series (by item or position, scalar or slice) or frame (single or multiple
        # blocks) raises a type error with one of several alternative messages (cannot convert to
        # Floating/Integer dtype, non-numeric NA, or invalid value). Warnings about setting on a view are ignored.
        # GH#44514 don't cast mismatched nulls to pd.NA
        df = DataFrame({"A": [1, 2, 3]}, dtype=any_numeric_ea_dtype)
        ser = df["A"].copy()
        arr = ser._values

        msg = "|".join(
            [
                r"timedelta64\[ns\] cannot be converted to (Floating|Integer)Dtype",
                r"datetime64\[ns\] cannot be converted to (Floating|Integer)Dtype",
                "'values' contains non-numeric NA",
                r"Invalid value '.*' for dtype '(U?Int|Float)\d{1,2}'",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            arr[0] = null

        with pytest.raises(TypeError, match=msg):
            arr[:2] = [null, null]

        with pytest.raises(TypeError, match=msg):
            ser[0] = null

        with pytest.raises(TypeError, match=msg):
            ser[:2] = [null, null]

        with pytest.raises(TypeError, match=msg):
            ser.iloc[0] = null

        with pytest.raises(TypeError, match=msg):
            ser.iloc[:2] = [null, null]

        with pytest.raises(TypeError, match=msg):
            df.iloc[0, 0] = null

        with pytest.raises(TypeError, match=msg):
            df.iloc[:2, 0] = [null, null]

        # Multi-Block
        df2 = df.copy()
        df2["B"] = ser.copy()
        with pytest.raises(TypeError, match=msg):
            df2.iloc[0, 0] = null

        with pytest.raises(TypeError, match=msg):
            df2.iloc[:2, 0] = [null, null]

    def test_loc_expand_empty_frame_keep_index_name(self):
        # [FUNCTION DOC]
        # Regression test for GH#45621: enlarging an empty frame with a named index by label location
        # keeps the index name.
        # GH#45621
        df = DataFrame(columns=["b"], index=Index([], name="a"))
        df.loc[0] = 1
        expected = DataFrame({"b": [1]}, index=Index([0], name="a"))
        tm.assert_frame_equal(df, expected)

    def test_loc_expand_empty_frame_keep_midx_names(self):
        # [FUNCTION DOC]
        # Regression test for GH#46317: enlarging an empty frame with a named three-level multi-index
        # via a tuple label keeps all level names.
        # GH#46317
        df = DataFrame(
            columns=["d"], index=MultiIndex.from_tuples([], names=["a", "b", "c"])
        )
        df.loc[(1, 2, 3)] = "foo"
        expected = DataFrame(
            {"d": ["foo"]},
            index=MultiIndex.from_tuples([(1, 2, 3)], names=["a", "b", "c"]),
        )
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "val, idxr",
        [
            ("x", "a"),
            ("x", ["a"]),
            (1, "a"),
            (1, ["a"]),
        ],
    )
    def test_loc_setitem_rhs_frame(self, idxr, val):
        # [FUNCTION DOC]
        # Regression test for GH#47578, parametrized over (string or integer value, scalar or list column
        # key): assigning a frame whose values include a mismatching type and index labels not equal to the
        # target raises a type error about an invalid value.
        # GH#47578
        df = DataFrame({"a": [1, 2]})

        with pytest.raises(TypeError, match="Invalid value"):
            df.loc[:, idxr] = DataFrame({"a": [val, 11]}, index=[1, 2])

    def test_iloc_setitem_enlarge_no_warning(self):
        # [FUNCTION DOC]
        # Regression test for GH#47381: assigning a float array to the first column positionally on an
        # empty two-column frame (enlarging) does not emit a warning and does not change a full-slice view taken earlier.
        # GH#47381
        df = DataFrame(columns=["a", "b"])
        expected = df.copy()
        view = df[:]
        df.iloc[:, 0] = np.array([1, 2], dtype=np.float64)
        tm.assert_frame_equal(view, expected)

    def test_loc_internals_not_updated_correctly(self):
        # [FUNCTION DOC]
        # Regression test for GH#47867 (all steps needed to reproduce the original bug): after
        # adding a column and repeatedly setting and reading a multi-index entry by label list, the final value read
        # equals the expected Series.
        # KNOWN ISSUE: the expected variable is chained-assigned to the name of the frame, silently rebinding
        # the frame name to the Series; harmless but unintended.
        # GH#47867 all steps are necessary to reproduce the initial bug
        df = DataFrame(
            {"bool_col": True, "a": 1, "b": 2.5},
            index=MultiIndex.from_arrays([[1, 2], [1, 2]], names=["idx1", "idx2"]),
        )
        idx = [(1, 1)]

        df["c"] = 3
        df.loc[idx, "c"] = 0

        df.loc[idx, "c"]
        df.loc[idx, ["a", "b"]]

        df.loc[idx, "c"] = 15
        result = df.loc[idx, "c"]
        expected = df = Series(
            15,
            index=MultiIndex.from_arrays([[1], [1]], names=["idx1", "idx2"]),
            name="c",
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("val", [None, [None], pd.NA, [pd.NA]])
    def test_iloc_setitem_string_list_na(self, val):
        # [FUNCTION DOC]
        # Regression test for GH#45469, parametrized over four missing-value forms (none, list with none, NA,
        # list with NA): positional row assignment on a string-dtype column sets the first element to NA.
        # GH#45469
        df = DataFrame({"a": ["a", "b", "c"]}, dtype="string")
        df.iloc[[0], :] = val
        expected = DataFrame({"a": [pd.NA, "b", "c"]}, dtype="string")
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("val", [None, pd.NA])
    def test_iloc_setitem_string_na(self, val):
        # [FUNCTION DOC]
        # Regression test for GH#45469, parametrized over none and NA: assigning the missing value to row
        # zero over all columns (positional) of a string-dtype frame sets NA.
        # GH#45469
        df = DataFrame({"a": ["a", "b", "c"]}, dtype="string")
        df.iloc[0, :] = val
        expected = DataFrame({"a": [pd.NA, "b", "c"]}, dtype="string")
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("func", [list, Series, np.array])
    def test_iloc_setitem_ea_null_slice_length_one_list(self, func):
        # [FUNCTION DOC]
        # Regression test for GH#48016, parametrized over list, Series and array constructors: assigning five
        # via a positional column indexer that is a one-element container on a masked integer frame fills the column.
        # GH#48016
        df = DataFrame({"a": [1, 2, 3]}, dtype="Int64")
        df.iloc[:, func([0])] = 5
        expected = DataFrame({"a": [5, 5, 5]}, dtype="Int64")
        tm.assert_frame_equal(df, expected)

    def test_loc_named_tuple_for_midx(self):
        # [FUNCTION DOC]
        # Regression test for GH#48124: label location with a named tuple whose fields match multi-index
        # level names selects the rows (one value, one list) and gives the expected two-row index.
        # GH#48124
        df = DataFrame(
            index=MultiIndex.from_product(
                [["A", "B"], ["a", "b", "c"]], names=["first", "second"]
            )
        )
        indexer_tuple = namedtuple("Indexer", df.index.names)
        idxr = indexer_tuple(first="A", second=["a", "b"])
        result = df.loc[idxr, :]
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                [("A", "a"), ("A", "b")], names=["first", "second"]
            )
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("indexer", [["a"], "a"])
    @pytest.mark.parametrize("col", [{}, {"b": 1}])
    def test_set_2d_casting_date_to_int(self, col, indexer):
        # [FUNCTION DOC]
        # Regression test for GH#49159, parametrized over single or list key and whether an additional
        # column exists: assigning a shifted date Series into one row via label location keeps date values
        # and does not cast dates to integers.
        # GH#49159
        df = DataFrame(
            {"a": [Timestamp("2022-12-29"), Timestamp("2022-12-30")], **col},
        )
        df.loc[[1], indexer] = df["a"] + pd.Timedelta(days=1)
        expected = DataFrame(
            {"a": [Timestamp("2022-12-29"), Timestamp("2022-12-31")], **col},
        )
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("has_ref", [True, False])
    @pytest.mark.parametrize("col", [{}, {"name": "a"}])
    def test_loc_setitem_reordering_with_all_true_indexer(self, col, has_ref):
        # [FUNCTION DOC]
        # Regression test for GH#48701, parametrized over an extra column and whether another reference
        # (view) exists: assigning the same values with an all-true row indexer and a reordered column list
        # leaves the frame unchanged.
        # GH#48701
        n = 17
        df = DataFrame({**col, "x": range(n), "y": range(n)})
        value = df[["x", "y"]].copy()
        expected = df.copy()
        if has_ref:
            view = df[:]  # noqa: F841
        df.loc[n * [True], ["x", "y"]] = value
        tm.assert_frame_equal(df, expected)

    def test_loc_rhs_empty_warning(self):
        # [FUNCTION DOC]
        # Regression test for GH48480: assigning an empty frame to a column of an empty two-column frame
        # via label location produces no warning and leaves the frame unchanged.
        # GH48480
        df = DataFrame(columns=["a", "b"])
        expected = df.copy()
        rhs = DataFrame(columns=["a"])
        with tm.assert_produces_warning(None):
            df.loc[:, "a"] = rhs
        tm.assert_frame_equal(df, expected)

    def test_iloc_ea_series_indexer(self):
        # [FUNCTION DOC]
        # Regression test for GH#49521: positional selection of rows and columns with masked integer Series
        # (or their underlying values) selects the expected block.
        # GH#49521
        df = DataFrame([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        indexer = Series([0, 1], dtype="Int64")
        row_indexer = Series([1], dtype="Int64")
        result = df.iloc[row_indexer, indexer]
        expected = DataFrame([[5, 6]], index=range(1, 2))
        tm.assert_frame_equal(result, expected)

        result = df.iloc[row_indexer.values, indexer.values]
        tm.assert_frame_equal(result, expected)

    def test_iloc_ea_series_indexer_with_na(self):
        # [FUNCTION DOC]
        # Regression test for GH#49521: positional column selection with an indexer containing NA raises a
        # value error with message about being unable to convert, for both the Series and its values.
        # GH#49521
        df = DataFrame([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        indexer = Series([0, pd.NA], dtype="Int64")
        msg = "cannot convert"
        with pytest.raises(ValueError, match=msg):
            df.iloc[:, indexer]
        with pytest.raises(ValueError, match=msg):
            df.iloc[:, indexer.values]

    @pytest.mark.parametrize("indexer", [True, (True,)])
    @pytest.mark.parametrize("dtype", [bool, "boolean"])
    def test_loc_bool_multiindex(self, performance_warning, dtype, indexer):
        # [FUNCTION DOC]
        # Regression test for GH#47687, parametrized over scalar or tuple indexers and bool or masked boolean
        # dtype: label location on a two-level boolean multi-index selects the matching rows and drops the first
        # level; a tuple key may emit a performance warning (uses the performance_warning fixture).
        # GH#47687
        midx = MultiIndex.from_arrays(
            [
                Series([True, True, False, False], dtype=dtype),
                Series([True, False, True, False], dtype=dtype),
            ],
            names=["a", "b"],
        )
        df = DataFrame({"c": [1, 2, 3, 4]}, index=midx)
        with tm.maybe_produces_warning(performance_warning, isinstance(indexer, tuple)):
            result = df.loc[indexer]
        expected = DataFrame(
            {"c": [1, 2]}, index=Index([True, False], name="b", dtype=dtype)
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("utc", [False, True])
    @pytest.mark.parametrize("indexer", ["date", ["date"]])
    def test_loc_datetime_assignment_dtype_does_not_change(self, utc, indexer):
        # [FUNCTION DOC]
        # Regression test for GH#49837, parametrized over naive or UTC and scalar or list column key:
        # assigning a selected datetime column onto itself through a boolean row mask keeps values and dtype unchanged.
        # GH#49837
        df = DataFrame(
            {
                "date": to_datetime(
                    [datetime(2022, 1, 20), datetime(2022, 1, 22)], utc=utc
                ),
                "update": [True, False],
            }
        )
        expected = df.copy(deep=True)

        update_df = df[df["update"]]

        df.loc[df["update"], indexer] = update_df["date"]

        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("indexer, idx", [(tm.loc, 1), (tm.iloc, 2)])
    def test_setitem_value_coercing_dtypes(self, indexer, idx):
        # [FUNCTION DOC]
        # Regression test for GH#50467, parametrized over label vs positional accessor (with the row
        # extent): assigning an integer frame into the leading rows of an object frame coerces those cells
        # while the remaining row keeps its string.
        # GH#50467
        df = DataFrame([["1", np.nan], ["2", np.nan], ["3", np.nan]], dtype=object)
        rhs = DataFrame([[1, np.nan], [2, np.nan]])
        indexer(df)[:idx, :] = rhs
        expected = DataFrame([[1, np.nan], [2, np.nan], ["3", np.nan]], dtype=object)
        tm.assert_frame_equal(df, expected)

    def test_big_endian_support_selecting_columns(self):
        # [FUNCTION DOC]
        # Regression test for GH#57457: selecting all columns of a frame with big-endian float columns
        # works and equals the result on an equivalent frame built from the dtype.
        # GH#57457
        columns = ["a"]
        data = [np.array([1, 2], dtype=">f8")]
        df = DataFrame(dict(zip(columns, data)))
        result = df[df.columns]
        dfexp = DataFrame({"a": [1, 2]}, dtype=">f8")
        expected = dfexp[dfexp.columns]
        tm.assert_frame_equal(result, expected)


class TestDataFrameIndexingUInt64:
    # [CLASS DOC]
    # Test class for indexing of frames with unsigned 64-bit integers.
    def test_setitem(self):
        # [FUNCTION DOC]
        # Verifies adding a column by assigning a renamed Series to an unsigned frame keeps the values and
        # takes the new name, including replacing an existing string column; and that assigning NaT into an
        # unsigned integer cell raises a type error about an invalid value.
        # KNOWN ISSUE: the comment says the column should be cast to object because unsigned integers have
        # no NaN, but the test asserts a type error instead; the second assignment in that block is never run.
        df = DataFrame(
            {"A": np.arange(3), "B": [2**63, 2**63 + 5, 2**63 + 10]},
            dtype=np.uint64,
        )
        idx = df["A"].rename("foo")

        # setitem
        assert "C" not in df.columns
        df["C"] = idx
        tm.assert_series_equal(df["C"], Series(idx, name="C"))

        assert "D" not in df.columns
        df["D"] = "foo"
        df["D"] = idx
        tm.assert_series_equal(df["D"], Series(idx, name="D"))
        del df["D"]

        # With NaN: because uint64 has no NaN element,
        # the column should be cast to object.
        df2 = df.copy()
        with pytest.raises(TypeError, match="Invalid value"):
            df2.iloc[1, 1] = pd.NaT
            df2.iloc[1, 2] = pd.NaT


def test_object_casting_indexing_wraps_datetimelike():
    # [FUNCTION DOC]
    # Regression test for GH#31649: rows retrieved from a mixed integer, datetime and timedelta frame
    # via label location, positional location, cross-section and internal fast cross-section
    # contain Timestamp and Timedelta objects rather than raw numbers; the datetime block is
    # nanosecond datetime dtype and the timedelta block is microsecond timedelta dtype, and each
    # block returns the wrapped scalar for its first cell.
    # GH#31649, check the indexing methods all the way down the stack
    df = DataFrame(
        {
            "A": [1, 2],
            "B": date_range("2000", periods=2, unit="ns"),
            "C": pd.timedelta_range("1 Day", periods=2),
        }
    )

    ser = df.loc[0]
    assert isinstance(ser.values[1], Timestamp)
    assert isinstance(ser.values[2], pd.Timedelta)

    ser = df.iloc[0]
    assert isinstance(ser.values[1], Timestamp)
    assert isinstance(ser.values[2], pd.Timedelta)

    ser = df.xs(0, axis=0)
    assert isinstance(ser.values[1], Timestamp)
    assert isinstance(ser.values[2], pd.Timedelta)

    mgr = df._mgr
    mgr._rebuild_blknos_and_blklocs()
    arr = mgr.fast_xs(0).array
    assert isinstance(arr[1], Timestamp)
    assert isinstance(arr[2], pd.Timedelta)

    blk = mgr.blocks[mgr.blknos[1]]
    assert blk.dtype == "M8[ns]"  # we got the right block
    val = blk.iget((0, 0))
    assert isinstance(val, Timestamp)

    blk = mgr.blocks[mgr.blknos[2]]
    assert blk.dtype == "m8[us]"  # we got the right block
    val = blk.iget((0, 0))
    assert isinstance(val, pd.Timedelta)


msg1 = r"Cannot setitem on a Categorical with a new category( \(.*\))?, set the"
msg2 = "Cannot set a Categorical with another, without identical categories"


class TestLocILocDataFrameCategorical:
    # [CLASS DOC]
    # Test class for assignment into categorical columns of a two-column frame (categorical column
    # cats with categories a and b, and an integer values column) through label and positional accessors.
    # Uses fixtures giving the original frame and two expected frames. The module-level patterns
    # msg1 (new category cannot be set) and msg2 (categoricals must have identical categories to set)
    # are used to match errors.
    @pytest.fixture
    def orig(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning the original frame: seven rows labeled h through n, categorical
        # column of all a with categories a and b, and an integer column of ones.
        cats = Categorical(["a", "a", "a", "a", "a", "a", "a"], categories=["a", "b"])
        idx = Index(["h", "i", "j", "k", "l", "m", "n"])
        values = [1, 1, 1, 1, 1, 1, 1]
        orig = DataFrame({"cats": cats, "values": values}, index=idx)
        return orig

    @pytest.fixture
    def exp_parts_cats_col(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning the expected frame after changing the third and fourth rows of
        # the categorical column to b (values column unchanged).
        # changed part of the cats column
        cats3 = Categorical(["a", "a", "b", "b", "a", "a", "a"], categories=["a", "b"])
        idx3 = Index(["h", "i", "j", "k", "l", "m", "n"])
        values3 = [1, 1, 1, 1, 1, 1, 1]
        exp_parts_cats_col = DataFrame({"cats": cats3, "values": values3}, index=idx3)
        return exp_parts_cats_col

    @pytest.fixture
    def exp_single_cats_value(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning the expected frame after changing only the third row of the
        # categorical column to b (values column unchanged).
        # changed single value in cats col
        cats4 = Categorical(["a", "a", "b", "a", "a", "a", "a"], categories=["a", "b"])
        idx4 = Index(["h", "i", "j", "k", "l", "m", "n"])
        values4 = [1, 1, 1, 1, 1, 1, 1]
        exp_single_cats_value = DataFrame(
            {"cats": cats4, "values": values4}, index=idx4
        )
        return exp_single_cats_value

    def test_loc_iloc_setitem_list_of_lists(self, orig, indexer_li):
        # [FUNCTION DOC]
        # Parametrized over the label and positional accessors (indexer_li fixture). Assigning a list of
        # two rows with a category that exists (b) and a number sets the two rows (categorical and integer
        # change); a row list with a category not in the categories (c) raises a type error with the msg1 pattern.
        #   - assign multiple rows (mixed values) -> exp_multi_row
        df = orig.copy()

        key = slice(2, 4)
        if indexer_li is tm.loc:
            key = slice("j", "k")

        indexer_li(df)[key, :] = [["b", 2], ["b", 2]]

        cats2 = Categorical(["a", "a", "b", "b", "a", "a", "a"], categories=["a", "b"])
        idx2 = Index(["h", "i", "j", "k", "l", "m", "n"])
        values2 = [1, 1, 2, 2, 1, 1, 1]
        exp_multi_row = DataFrame({"cats": cats2, "values": values2}, index=idx2)
        tm.assert_frame_equal(df, exp_multi_row)

        df = orig.copy()
        with pytest.raises(TypeError, match=msg1):
            indexer_li(df)[key, :] = [["c", 2], ["c", 2]]

    @pytest.mark.parametrize("indexer", [tm.loc, tm.iloc, tm.at, tm.iat])
    def test_loc_iloc_at_iat_setitem_single_value_in_categories(
        self, orig, exp_single_cats_value, indexer
    ):
        # [FUNCTION DOC]
        # Parametrized over label, positional, scalar-label and scalar-positional accessors. Assigning an
        # existing category to a single cell sets it as expected; a category not among the categories
        # raises a type error with the msg1 pattern.
        #   - assign a single value -> exp_single_cats_value
        df = orig.copy()

        key = (2, 0)
        if indexer in [tm.loc, tm.at]:
            key = (df.index[2], df.columns[0])

        # "b" is among the categories for df["cat"}]
        indexer(df)[key] = "b"
        tm.assert_frame_equal(df, exp_single_cats_value)

        # "c" is not among the categories for df["cat"]
        with pytest.raises(TypeError, match=msg1):
            indexer(df)[key] = "c"

    def test_loc_iloc_setitem_mask_single_value_in_categories(
        self, orig, exp_single_cats_value, indexer_li
    ):
        # [FUNCTION DOC]
        # Parametrized over label and positional accessors. A boolean mask selecting one row combined with
        # the categorical column, assigned an existing category, changes just that cell as expected.
        # mask with single True
        df = orig.copy()

        mask = df.index == "j"
        key = 0
        if indexer_li is tm.loc:
            key = df.columns[key]

        indexer_li(df)[mask, key] = "b"
        tm.assert_frame_equal(df, exp_single_cats_value)

    def test_loc_iloc_setitem_full_row_non_categorical_rhs(self, orig, indexer_li):
        # [FUNCTION DOC]
        # Parametrized over label and positional accessors. Assigning a plain (non categorical) full row
        # with an existing category and a number sets that row (categorical and integer); a category not
        # in the categories raises a type error with the msg1 pattern.
        #   - assign a complete row (mixed values) -> exp_single_row
        df = orig.copy()

        key = 2
        if indexer_li is tm.loc:
            key = df.index[2]

        # not categorical dtype, but "b" _is_ among the categories for df["cat"]
        indexer_li(df)[key, :] = ["b", 2]
        cats1 = Categorical(["a", "a", "b", "a", "a", "a", "a"], categories=["a", "b"])
        idx1 = Index(["h", "i", "j", "k", "l", "m", "n"])
        values1 = [1, 1, 2, 1, 1, 1, 1]
        exp_single_row = DataFrame({"cats": cats1, "values": values1}, index=idx1)
        tm.assert_frame_equal(df, exp_single_row)

        # "c" is not among the categories for df["cat"]
        with pytest.raises(TypeError, match=msg1):
            indexer_li(df)[key, :] = ["c", 2]

    def test_loc_iloc_setitem_partial_col_categorical_rhs(
        self, orig, exp_parts_cats_col, indexer_li
    ):
        # [FUNCTION DOC]
        # Parametrized over label and positional accessors. Assigning a categorical with the same
        # categories to a slice of the categorical column succeeds. Assigning a categorical with different
        # categories raises a type error with the msg2 pattern, even if values are holdable (the comment
        # is unsure if this should fail), and also if values are not holdable.
        # assign a part of a column with dtype == categorical ->
        # exp_parts_cats_col
        df = orig.copy()

        key = (slice(2, 4), 0)
        if indexer_li is tm.loc:
            key = (slice("j", "k"), df.columns[0])

        # same categories as we currently have in df["cats"]
        compat = Categorical(["b", "b"], categories=["a", "b"])
        indexer_li(df)[key] = compat
        tm.assert_frame_equal(df, exp_parts_cats_col)

        # categories do not match df["cat"]'s, but "b" is among them
        semi_compat = Categorical(list("bb"), categories=list("abc"))
        with pytest.raises(TypeError, match=msg2):
            # different categories but holdable values
            #  -> not sure if this should fail or pass
            indexer_li(df)[key] = semi_compat

        # categories do not match df["cat"]'s, and "c" is not among them
        incompat = Categorical(list("cc"), categories=list("abc"))
        with pytest.raises(TypeError, match=msg2):
            # different values
            indexer_li(df)[key] = incompat

    def test_loc_iloc_setitem_non_categorical_rhs(
        self, orig, exp_parts_cats_col, indexer_li
    ):
        # [FUNCTION DOC]
        # Parametrized over label and positional accessors. Assigning a plain list of existing category
        # values to a slice of the categorical column succeeds; a list with a new category raises a type
        # error with the msg1 pattern.
        # assign a part of a column with dtype != categorical -> exp_parts_cats_col
        df = orig.copy()

        key = (slice(2, 4), 0)
        if indexer_li is tm.loc:
            key = (slice("j", "k"), df.columns[0])

        # "b" is among the categories for df["cat"]
        indexer_li(df)[key] = ["b", "b"]
        tm.assert_frame_equal(df, exp_parts_cats_col)

        # "c" not part of the categories
        with pytest.raises(TypeError, match=msg1):
            indexer_li(df)[key] = ["c", "c"]

    @pytest.mark.parametrize("indexer", [tm.getitem, tm.loc, tm.iloc])
    def test_getitem_preserve_object_index_with_dates(self, indexer):
        # [FUNCTION DOC]
        # Parametrized over item, label and positional getters. Per PR 42950: selecting a column from
        # a frame with an object-dtype index of datetimes yields a Series whose index stays object
        # dtype (no inference).
        # https://github.com/pandas-dev/pandas/pull/42950 - when selecting a column
        # from dataframe, don't try to infer object dtype index on Series construction
        idx = date_range("2012", periods=3).astype(object)
        df = DataFrame({0: [1, 2, 3]}, index=idx)
        assert df.index.dtype == object

        if indexer is tm.getitem:
            ser = indexer(df)[0]
        else:
            ser = indexer(df)[:, 0]

        assert ser.index.dtype == object

    def test_loc_on_multiindex_one_level(self):
        # [FUNCTION DOC]
        # Regression test for GH#45779: label location with a scalar on a one-level multi-index returns a
        # frame with a one-level multi-index containing only the matching key and level name.
        # GH#45779
        df = DataFrame(
            data=[[0], [1]],
            index=MultiIndex.from_tuples([("a",), ("b",)], names=["first"]),
        )
        expected = DataFrame(
            data=[[0]], index=MultiIndex.from_tuples([("a",)], names=["first"])
        )
        result = df.loc["a"]
        tm.assert_frame_equal(result, expected)


class TestDeprecatedIndexers:
    # [CLASS DOC]
    # Test class for indexers (dictionaries and sets) that are rejected in current versions.
    @pytest.mark.parametrize(
        "key", [{1}, {1: 1}, ({1}, "a"), ({1: 1}, "a"), (1, {"a"}), (1, {"a": "a"})]
    )
    def test_getitem_dict_and_set_deprecated(self, key):
        # [FUNCTION DOC]
        # KNOWN ISSUE: the name says deprecated, but the comment says the deprecation was enforced in 2.0,
        # so the test checks an error.
        # Parametrized over sets, dictionaries, tuples containing them. Verifies label location getitem
        # with such a key raises a type error saying it is not supported as an indexer.
        # GH#42825 enforced in 2.0
        df = DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
        with pytest.raises(TypeError, match="as an indexer is not supported"):
            df.loc[key]

    @pytest.mark.parametrize(
        "key",
        [
            {1},
            {1: 1},
            (({1}, 2), "a"),
            (({1: 1}, 2), "a"),
            ((1, 2), {"a"}),
            ((1, 2), {"a": "a"}),
        ],
    )
    def test_getitem_dict_and_set_deprecated_multiindex(self, key):
        # [FUNCTION DOC]
        # KNOWN ISSUE: the name says deprecated, but the comment says the deprecation was enforced in 2.0,
        # so the test checks an error.
        # Same as the previous test for a frame with a two-level row index and nested tuple keys: raises a
        # type error saying it is not supported as an indexer.
        # GH#42825 enforced in 2.0
        df = DataFrame(
            [[1, 2], [3, 4]],
            columns=["a", "b"],
            index=MultiIndex.from_tuples([(1, 2), (3, 4)]),
        )
        with pytest.raises(TypeError, match="as an indexer is not supported"):
            df.loc[key]

    @pytest.mark.parametrize(
        "key", [{1}, {1: 1}, ({1}, "a"), ({1: 1}, "a"), (1, {"a"}), (1, {"a": "a"})]
    )
    def test_setitem_dict_and_set_disallowed(self, key):
        # [FUNCTION DOC]
        # Parametrized over sets, dictionaries and tuples containing them: label location assignment with
        # such a key raises a type error saying it is not supported as an indexer (enforced in 2.0).
        # GH#42825 enforced in 2.0
        df = DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
        with pytest.raises(TypeError, match="as an indexer is not supported"):
            df.loc[key] = 1

    @pytest.mark.parametrize(
        "key",
        [
            {1},
            {1: 1},
            (({1}, 2), "a"),
            (({1: 1}, 2), "a"),
            ((1, 2), {"a"}),
            ((1, 2), {"a": "a"}),
        ],
    )
    def test_setitem_dict_and_set_disallowed_multiindex(self, key):
        # [FUNCTION DOC]
        # Same as the previous test for a frame with a two-level row index and nested tuple keys.
        # GH#42825 enforced in 2.0
        df = DataFrame(
            [[1, 2], [3, 4]],
            columns=["a", "b"],
            index=MultiIndex.from_tuples([(1, 2), (3, 4)]),
        )
        with pytest.raises(TypeError, match="as an indexer is not supported"):
            df.loc[key] = 1


def test_adding_new_conditional_column() -> None:
    # [FUNCTION DOC]
    # Regression test for issue 55025: assigning a string through a boolean row mask to a new column
    # creates it with the string and missing values elsewhere; assigning a function object creates an
    # object column holding the function.
    # https://github.com/pandas-dev/pandas/issues/55025
    df = DataFrame({"x": [1]})
    df.loc[df["x"] == 1, "y"] = "1"
    expected = DataFrame({"x": [1], "y": ["1"]})
    tm.assert_frame_equal(df, expected)

    df = DataFrame({"x": [1]})
    # try inserting something which numpy would store as 'object'
    value = lambda x: x
    df.loc[df["x"] == 1, "y"] = value
    expected = DataFrame({"x": [1], "y": [value]})
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    ("dtype", "infer_string"),
    [
        (object, False),
        (pd.StringDtype(na_value=np.nan), True),
    ],
)
def test_adding_new_conditional_column_with_string(dtype, infer_string) -> None:
    # [FUNCTION DOC]
    # Regression test for issue 56204, parametrized over (object, infer disabled) and (NaN-valued
    # string dtype, infer enabled): assigning a string to a new column by a boolean row mask (within an option
    # context) gives a column of the string and NaN with the matching dtype, others integer.
    # https://github.com/pandas-dev/pandas/issues/56204
    df = DataFrame({"a": [1, 2], "b": [3, 4]})
    with pd.option_context("future.infer_string", infer_string):
        df.loc[df["a"] == 1, "c"] = "1"
    expected = DataFrame({"a": [1, 2], "b": [3, 4], "c": ["1", float("nan")]}).astype(
        {"a": "int64", "b": "int64", "c": dtype}
    )
    tm.assert_frame_equal(df, expected)


def test_add_new_column_infer_string():
    # [FUNCTION DOC]
    # Regression test for GH#55366: with string inference enabled, creating a new column through a
    # boolean mask assignment of a string gives a string-dtype column and a string-dtype column index.
    # GH#55366
    df = DataFrame({"x": [1]})
    with pd.option_context("future.infer_string", True):
        df.loc[df["x"] == 1, "y"] = "1"
    expected = DataFrame(
        {"x": [1], "y": Series(["1"], dtype=pd.StringDtype(na_value=np.nan))},
        columns=Index(["x", "y"], dtype="str"),
    )
    tm.assert_frame_equal(df, expected)


class TestSetitemValidation:
    # [CLASS DOC]
    # Test class validating that scalar assignment rejects invalid values for numpy bool, integer and float
    # dtypes through label and positional accessors and many indexer forms. Adapted from the masked-array
    # indexing tests. Class attributes list invalid scalars (complex, strings, NaT variants) and the
    # indexers (scalar, list, slice, boolean list, full slice).
    # This is adapted from pandas/tests/arrays/masked/test_indexing.py
    def _check_setitem_invalid(self, df, invalid, indexer):
        # [FUNCTION DOC]
        # HELPER: asserts that assigning the given invalid value to the first column using the given indexer
        # raises a type error with an invalid value message, for positional and for label access. Parameters:
        # df frame with column a, invalid value, indexer.
        # KNOWN ISSUE: the statement restoring the frame from the original copy is inside the block
        # that is expected to raise, after the raising line, so it never runs (the second check uses the
        # possibly modified frame).
        orig_df = df.copy()

        # iloc
        with pytest.raises(TypeError, match="Invalid value"):
            df.iloc[indexer, 0] = invalid
            df = orig_df.copy()

        # loc
        with pytest.raises(TypeError, match="Invalid value"):
            df.loc[indexer, "a"] = invalid
            df = orig_df.copy()

    def _check_setitem_valid(self, df, value, indexer):
        # [FUNCTION DOC]
        # HELPER: assigns the given valid value to the first column with the indexer, using positional then
        # label access. Passes if no error is raised.
        # KNOWN ISSUE: the frame is rebound to a copy after each assignment but never used afterwards, so
        # those lines have no effect.
        orig_df = df.copy()

        # iloc
        df.iloc[indexer, 0] = value
        df = orig_df.copy()

        # loc
        df.loc[indexer, "a"] = value
        df = orig_df.copy()

    _invalid_scalars = [
        1 + 2j,
        "True",
        "1",
        "1.0",
        pd.NaT,
        np.datetime64("NaT", "ns"),
        np.timedelta64("NaT", "ns"),
    ]
    _indexers = [0, [0], slice(0, 1), [True, False, False], slice(None, None, None)]

    @pytest.mark.parametrize(
        "invalid", [*_invalid_scalars, 1, 1.0, np.int64(1), np.float64(1)]
    )
    @pytest.mark.parametrize("indexer", _indexers)
    def test_setitem_validation_scalar_bool(self, invalid, indexer):
        # [FUNCTION DOC]
        # Parametrized over invalid scalars (plus integers and floats, which are invalid for booleans) and
        # indexers. Assigning to a boolean frame must raise a type error about an invalid value.
        df = DataFrame({"a": [True, False, False]}, dtype="bool")
        self._check_setitem_invalid(df, invalid, indexer)

    @pytest.mark.parametrize("invalid", [*_invalid_scalars, True, 1.5, np.float64(1.5)])
    @pytest.mark.parametrize("indexer", _indexers)
    def test_setitem_validation_scalar_int(self, invalid, any_int_numpy_dtype, indexer):
        # [FUNCTION DOC]
        # Parametrized over invalid scalars, True and fractional floats, any signed or unsigned integer dtype
        # and indexers. Assigning must raise a type error about an invalid value.
        # KNOWN ISSUE: the branch that expects a valid assignment is only taken for missing values that are not
        # NaT forms, but none of the listed invalid scalars qualifies, so it is never reached.
        df = DataFrame({"a": [1, 2, 3]}, dtype=any_int_numpy_dtype)
        if isna(invalid) and invalid is not pd.NaT and not np.isnat(invalid):
            self._check_setitem_valid(df, invalid, indexer)
        else:
            self._check_setitem_invalid(df, invalid, indexer)

    @pytest.mark.parametrize("invalid", [*_invalid_scalars, True])
    @pytest.mark.parametrize("indexer", _indexers)
    def test_setitem_validation_scalar_float(self, invalid, float_numpy_dtype, indexer):
        # [FUNCTION DOC]
        # Parametrized over invalid scalars plus True, each float dtype and indexers. Assigning must raise a
        # type error about an invalid value on a float frame with a missing value.
        df = DataFrame({"a": [1, 2, None]}, dtype=float_numpy_dtype)
        self._check_setitem_invalid(df, invalid, indexer)
