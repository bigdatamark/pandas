"""test positional based indexing with iloc"""
# [MODULE DOC]
# Tests for position-based indexing (the iloc accessor) on series and frames: getting and
# setting with integers, lists, slices, boolean masks, callables and arrays; bounds
# behavior; duplicates; categorical, extension-array and object data; copy-on-write
# interaction; and error cases. Defines a module-level regular expression message for numpy
# index errors and shared helpers imported from the indexing test utilities.

from datetime import datetime
import re

import numpy as np
import pytest

from pandas.compat import pa_version_under16p0
from pandas.errors import IndexingError
import pandas.util._test_decorators as td

from pandas import (
    NA,
    Categorical,
    CategoricalDtype,
    DataFrame,
    Index,
    Interval,
    NaT,
    Series,
    StringDtype,
    Timestamp,
    array,
    concat,
    date_range,
    interval_range,
    isna,
    to_datetime,
)
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises

# We pass through the error message from numpy
_slice_iloc_msg = re.escape(
    "only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) "
    "and integer or boolean arrays are valid indices"
)


class TestiLoc:
    @pytest.mark.parametrize("key", [2, -1, [0, 1, 2]])
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
    # [CLASS DOC]
    # Small test collection for the shared integer and list-of-integer smoke test.
    def test_iloc_getitem_int_and_list_int(self, key, frame_or_series, index, request):
        # [FUNCTION DOC]
        # Parametrized over key (integer, negative integer, list of integers) and five index kinds
        # (strings, mixed object, dates, floats, empty), plus frame-or-series. Delegates to the shared
        # smoke-test helper, which verifies positional getitem works or raises an index error (the
        # declared failure kind) for every combination, for example for the empty index.
        # KNOWN ISSUE: the request fixture is unused, and the trailing comment about arrays of integers
        # (GH 5006) is not followed by any code.
        obj = frame_or_series(range(len(index)), index=index)
        check_indexing_smoketest_or_raises(
            obj,
            "iloc",
            key,
            fails=IndexError,
        )

        # array of ints (GH5006), make sure that a single indexer is returning
        # the correct type


class TestiLocBaseIndependent:
    # [CLASS DOC]
    # Test collection for iloc behavior that does not depend on a shared base class.
    # KNOWN ISSUE: the docstring refers to independence of a base class, but there is no base class
    # in this module.
    """Tests Independent Of Base Class"""

    @pytest.mark.parametrize(
        "key",
        [
            slice(None),
            slice(3),
            range(3),
            [0, 1, 2],
            Index(range(3)),
            np.asarray([0, 1, 2]),
        ],
    )
    def test_iloc_setitem_fullcol_categorical(self, indexer_li, key):
        # [FUNCTION DOC]
        # Parametrized over key forms selecting all three rows (slice, range, list, index, array) and
        # the setter fixture (iloc or loc style). Setting a categorical to a whole object column: the
        # column can hold the categorical, writes go into the existing memory, the frame equals the
        # object-cast categorical, and the categorical is not a view (later changes to the frame do not
        # change it). Also for a mixed-dtype frame the column stays object.
        frame = DataFrame({0: range(3)}, dtype=object)

        cat = Categorical(["alpha", "beta", "gamma"])

        assert frame._mgr.blocks[0]._can_hold_element(cat)

        df = frame.copy()
        orig_vals = df.values

        indexer_li(df)[key, 0] = cat

        expected = DataFrame({0: cat}).astype(object)
        assert np.shares_memory(df[0].values, orig_vals)

        tm.assert_frame_equal(df, expected)

        # check we dont have a view on cat (may be undesired GH#39986)
        df.iloc[0, 0] = "gamma"
        assert cat[0] != "gamma"

        # pre-2.0 with mixed dataframe ("split" path) we always overwrote the
        #  column.  as of 2.0 we correctly write "into" the column, so
        #  we retain the object dtype.
        frame = DataFrame({0: np.array([0, 1, 2], dtype=object), 1: range(3)})
        df = frame.copy()
        indexer_li(df)[key, 0] = cat
        expected = DataFrame({0: Series(cat.astype(object), dtype=object), 1: range(3)})
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("has_ref", [True, False])
    def test_iloc_setitem_ea_inplace(
        self, frame_or_series, index_or_series_or_array, has_ref
    ):
        # GH#38952 Case with not setting a full column
        #  IntegerArray without NAs
        # [FUNCTION DOC]
        # GH 38952. Parametrized over whether a reference exists, frame-or-series, and a box fixture
        # (series, index or array). Assigning the tail half of a nullable integer array into the first
        # half of an integer-valued object writes the expected values; without a reference the write is
        # done in place (shared memory with the original values).
        arr = array([1, 2, 3, 4])
        obj = frame_or_series(arr.to_numpy("i8"))
        if has_ref:
            view = obj[:]  # noqa: F841

        if frame_or_series is Series:
            values = obj.values
        else:
            values = obj._mgr.blocks[0].values

        if frame_or_series is Series:
            obj.iloc[:2] = index_or_series_or_array(arr[2:])
        else:
            obj.iloc[:2, 0] = index_or_series_or_array(arr[2:])

        expected = frame_or_series(np.array([3, 4, 3, 4], dtype="i8"))
        tm.assert_equal(obj, expected)

        # Check that we are actually in-place
        if not has_ref:
            if frame_or_series is Series:
                assert obj.values is not values
                assert np.shares_memory(obj.values, values)
            else:
                assert np.shares_memory(obj[0].values, values)

    def test_is_scalar_access(self):
        # GH#32085 index with duplicates doesn't matter for _is_scalar_access
        # [FUNCTION DOC]
        # GH 32085. The internal scalar-access predicate returns true for a one-element key on a series
        # with duplicate labels and a two-element key on a frame.
        index = Index([1, 2, 1])
        ser = Series(range(3), index=index)

        assert ser.iloc._is_scalar_access((1,))

        df = ser.to_frame()
        assert df.iloc._is_scalar_access((1, 0))

    @pytest.mark.skipif(
        pa_version_under16p0, reason="https://github.com/apache/arrow/issues/40642"
    )
    def test_iloc_exceeds_bounds(self):
        # GH6296
        # iloc should allow indexers that exceed the bounds
        # [FUNCTION DOC]
        # GH 6296, GH 10779. Skipped for older pyarrow. Lists of positions out of range raise index errors
        # (multi-position message) for frames and series; single positions out of range raise a
        # single-position out-of-bounds message; slices with bounds beyond the length are clipped (all
        # orderings of start/stop relative to 0 and length, forward and backward) and give the expected
        # slices; documentation examples including an empty column slice and clipped row slices.
        df = DataFrame(np.random.default_rng(2).random((20, 5)), columns=list("ABCDE"))

        # lists of positions should raise IndexError!
        msg = "positional indexers are out-of-bounds"
        with pytest.raises(IndexError, match=msg):
            df.iloc[:, [0, 1, 2, 3, 4, 5]]
        with pytest.raises(IndexError, match=msg):
            df.iloc[[1, 30]]
        with pytest.raises(IndexError, match=msg):
            df.iloc[[1, -30]]
        with pytest.raises(IndexError, match=msg):
            df.iloc[[100]]

        s = df["A"]
        with pytest.raises(IndexError, match=msg):
            s.iloc[[100]]
        with pytest.raises(IndexError, match=msg):
            s.iloc[[-100]]

        # still raise on a single indexer
        msg = "single positional indexer is out-of-bounds"
        with pytest.raises(IndexError, match=msg):
            df.iloc[30]
        with pytest.raises(IndexError, match=msg):
            df.iloc[-30]

        # GH10779
        # single positive/negative indexer exceeding Series bounds should raise
        # an IndexError
        with pytest.raises(IndexError, match=msg):
            s.iloc[30]
        with pytest.raises(IndexError, match=msg):
            s.iloc[-30]

        # slices are ok
        result = df.iloc[:, 4:10]  # 0 < start < len < stop
        expected = df.iloc[:, 4:]
        tm.assert_frame_equal(result, expected)

        result = df.iloc[:, -4:-10]  # stop < 0 < start < len
        expected = df.iloc[:, :0]
        tm.assert_frame_equal(result, expected)

        result = df.iloc[:, 10:4:-1]  # 0 < stop < len < start (down)
        expected = df.iloc[:, :4:-1]
        tm.assert_frame_equal(result, expected)

        result = df.iloc[:, 4:-10:-1]  # stop < 0 < start < len (down)
        expected = df.iloc[:, 4::-1]
        tm.assert_frame_equal(result, expected)

        result = df.iloc[:, -10:4]  # start < 0 < stop < len
        expected = df.iloc[:, :4]
        tm.assert_frame_equal(result, expected)

        result = df.iloc[:, 10:4]  # 0 < stop < len < start
        expected = df.iloc[:, :0]
        tm.assert_frame_equal(result, expected)

        result = df.iloc[:, -10:-11:-1]  # stop < start < 0 < len (down)
        expected = df.iloc[:, :0]
        tm.assert_frame_equal(result, expected)

        result = df.iloc[:, 10:11]  # 0 < len < start < stop
        expected = df.iloc[:, :0]
        tm.assert_frame_equal(result, expected)

        # slice bounds exceeding is ok
        result = s.iloc[18:30]
        expected = s.iloc[18:]
        tm.assert_series_equal(result, expected)

        result = s.iloc[30:]
        expected = s.iloc[:0]
        tm.assert_series_equal(result, expected)

        result = s.iloc[30::-1]
        expected = s.iloc[::-1]
        tm.assert_series_equal(result, expected)

        # doc example
        dfl = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), columns=list("AB")
        )
        tm.assert_frame_equal(
            dfl.iloc[:, 2:3],
            DataFrame(index=dfl.index, columns=Index([], dtype=dfl.columns.dtype)),
        )
        tm.assert_frame_equal(dfl.iloc[:, 1:3], dfl.iloc[:, [1]])
        tm.assert_frame_equal(dfl.iloc[4:6], dfl.iloc[[4]])

        msg = "positional indexers are out-of-bounds"
        with pytest.raises(IndexError, match=msg):
            dfl.iloc[[4, 5, 6]]
        msg = "single positional indexer is out-of-bounds"
        with pytest.raises(IndexError, match=msg):
            dfl.iloc[:, 4]

    @pytest.mark.parametrize("index,columns", [(np.arange(20), list("ABCDE"))])
    @pytest.mark.parametrize(
        "index_vals,column_vals",
        [
            ([slice(None), ["A", "D"]]),
            (["1", "2"], slice(None)),
            ([datetime(2019, 1, 1)], slice(None)),
        ],
    )
    def test_iloc_non_integer_raises(self, index, columns, index_vals, column_vals):
        # GH 25753
        # [FUNCTION DOC]
        # GH 25753. Parametrized over row and column indexers that are label-like (column names, strings,
        # datetimes). Raises an index error that iloc requires numeric indexers.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(index), len(columns))),
            index=index,
            columns=columns,
        )
        msg = ".iloc requires numeric indexers, got"
        with pytest.raises(IndexError, match=msg):
            df.iloc[index_vals, column_vals]

    def test_iloc_getitem_invalid_scalar(self, frame_or_series):
        # GH 21982

        # [FUNCTION DOC]
        # GH 21982. Parametrized over frame / series. A string scalar raises a type error about
        # indexing by location.
        obj = DataFrame(np.arange(100).reshape(10, 10))
        obj = tm.get_obj(obj, frame_or_series)

        with pytest.raises(TypeError, match="Cannot index by location index"):
            obj.iloc["a"]

    def test_iloc_array_not_mutating_negative_indices(self):
        # GH 21867
        # [FUNCTION DOC]
        # GH 21867. Indexing rows and columns with an array containing negative values does not
        # modify the input array.
        array_with_neg_numbers = np.array([1, 2, -1])
        array_copy = array_with_neg_numbers.copy()
        df = DataFrame(
            {"A": [100, 101, 102], "B": [103, 104, 105], "C": [106, 107, 108]},
            index=[1, 2, 3],
        )
        df.iloc[array_with_neg_numbers]
        tm.assert_numpy_array_equal(array_with_neg_numbers, array_copy)
        df.iloc[:, array_with_neg_numbers]
        tm.assert_numpy_array_equal(array_with_neg_numbers, array_copy)

    def test_iloc_getitem_neg_int_can_reach_first_index(self):
        # GH10547 and GH10779
        # negative integers should be able to reach index 0
        # [FUNCTION DOC]
        # GH 10547, GH 10779. Negative positions equal to minus the length reach the first element, for
        # scalar and list indexers on frame and series, and for a length-one series.
        df = DataFrame({"A": [2, 3, 5], "B": [7, 11, 13]})
        s = df["A"]

        expected = df.iloc[0]
        result = df.iloc[-3]
        tm.assert_series_equal(result, expected)

        expected = df.iloc[[0]]
        result = df.iloc[[-3]]
        tm.assert_frame_equal(result, expected)

        expected = s.iloc[0]
        result = s.iloc[-3]
        assert result == expected

        expected = s.iloc[[0]]
        result = s.iloc[[-3]]
        tm.assert_series_equal(result, expected)

        # check the length 1 Series case highlighted in GH10547
        expected = Series(["a"], index=["A"])
        result = expected.iloc[[-1]]
        tm.assert_series_equal(result, expected)

    def test_iloc_getitem_dups(self):
        # GH 6766
        # [FUNCTION DOC]
        # GH 6766. With duplicate column labels created by concatenation, scalar positional access
        # returns missing for first cell and the row slice returns a series with duplicated labels.
        df1 = DataFrame([{"A": None, "B": 1}, {"A": 2, "B": 2}])
        df2 = DataFrame([{"A": 3, "B": 3}, {"A": 4, "B": 4}])
        df = concat([df1, df2], axis=1)

        # cross-sectional indexing
        result = df.iloc[0, 0]
        assert isna(result)

        result = df.iloc[0, :]
        expected = Series([np.nan, 1, 3, 3], index=["A", "B", "A", "B"], name=0)
        tm.assert_series_equal(result, expected)

    def test_iloc_getitem_array(self):
        # [FUNCTION DOC]
        # List of positions on rows, and rows with columns, selects the expected sub-frames.
        df = DataFrame(
            [
                {"A": 1, "B": 2, "C": 3},
                {"A": 100, "B": 200, "C": 300},
                {"A": 1000, "B": 2000, "C": 3000},
            ]
        )

        expected = DataFrame([{"A": 1, "B": 2, "C": 3}])
        tm.assert_frame_equal(df.iloc[[0]], expected)

        expected = DataFrame([{"A": 1, "B": 2, "C": 3}, {"A": 100, "B": 200, "C": 300}])
        tm.assert_frame_equal(df.iloc[[0, 1]], expected)

        expected = DataFrame([{"B": 2, "C": 3}, {"B": 2000, "C": 3000}], index=[0, 2])
        result = df.iloc[[0, 2], [1, 2]]
        tm.assert_frame_equal(result, expected)

    def test_iloc_getitem_bool(self):
        # [FUNCTION DOC]
        # A boolean list and a callable returning a boolean array select the matching rows.
        df = DataFrame(
            [
                {"A": 1, "B": 2, "C": 3},
                {"A": 100, "B": 200, "C": 300},
                {"A": 1000, "B": 2000, "C": 3000},
            ]
        )

        expected = DataFrame([{"A": 1, "B": 2, "C": 3}, {"A": 100, "B": 200, "C": 300}])
        result = df.iloc[[True, True, False]]
        tm.assert_frame_equal(result, expected)

        expected = DataFrame(
            [{"A": 1, "B": 2, "C": 3}, {"A": 1000, "B": 2000, "C": 3000}], index=[0, 2]
        )
        result = df.iloc[lambda x: x.index % 2 == 0]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("index", [[True, False], [True, False, True, False]])
    def test_iloc_getitem_bool_diff_len(self, index):
        # GH26658
        # [FUNCTION DOC]
        # GH 26658. Parametrized over a shorter and a longer boolean list than the series. Raises an
        # index error stating the boolean index length and the expected length.
        s = Series([1, 2, 3])
        msg = f"Boolean index has wrong length: {len(index)} instead of {len(s)}"
        with pytest.raises(IndexError, match=msg):
            s.iloc[index]

    def test_iloc_getitem_slice(self):
        # [FUNCTION DOC]
        # Row slice, combined row/column slices, and a callable column indexer give the expected
        # sub-frames.
        df = DataFrame(
            [
                {"A": 1, "B": 2, "C": 3},
                {"A": 100, "B": 200, "C": 300},
                {"A": 1000, "B": 2000, "C": 3000},
            ]
        )

        expected = DataFrame([{"A": 1, "B": 2, "C": 3}, {"A": 100, "B": 200, "C": 300}])
        result = df.iloc[:2]
        tm.assert_frame_equal(result, expected)

        expected = DataFrame([{"A": 100, "B": 200}], index=[1])
        result = df.iloc[1:2, 0:2]
        tm.assert_frame_equal(result, expected)

        expected = DataFrame(
            [{"A": 1, "C": 3}, {"A": 100, "C": 300}, {"A": 1000, "C": 3000}]
        )
        result = df.iloc[:, lambda df: [0, 2]]
        tm.assert_frame_equal(result, expected)

    def test_iloc_getitem_slice_dups(self):
        # [FUNCTION DOC]
        # Column and row slicing on concatenated frames with duplicate column labels returns the
        # original frames in both orders and along both axes.
        df1 = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=["A", "A", "B", "B"],
        )
        df2 = DataFrame(
            np.random.default_rng(2).integers(0, 10, size=20).reshape(10, 2),
            columns=["A", "C"],
        )

        # axis=1
        df = concat([df1, df2], axis=1)
        tm.assert_frame_equal(df.iloc[:, :4], df1)
        tm.assert_frame_equal(df.iloc[:, 4:], df2)

        df = concat([df2, df1], axis=1)
        tm.assert_frame_equal(df.iloc[:, :2], df2)
        tm.assert_frame_equal(df.iloc[:, 2:], df1)

        exp = concat([df2, df1.iloc[:, [0]]], axis=1)
        tm.assert_frame_equal(df.iloc[:, 0:3], exp)

        # axis=0
        df = concat([df, df], axis=0)
        tm.assert_frame_equal(df.iloc[0:10, :2], df2)
        tm.assert_frame_equal(df.iloc[0:10, 2:], df1)
        tm.assert_frame_equal(df.iloc[10:, :2], df2)
        tm.assert_frame_equal(df.iloc[10:, 2:], df1)

    @pytest.mark.parametrize("has_ref", [True, False])
    def test_iloc_setitem(sel, has_ref):
        # [FUNCTION DOC]
        # Parametrized over whether a reference exists. Scalar assignment at a position, column-slice
        # assignment of zero, and augmented assignment on a series slice (GH 5771) give the expected
        # values.
        # KNOWN ISSUE: the first parameter is named sel rather than self, so the method works only
        # because the instance binds to that name; in addition, for the column assignment the expected
        # and result values are both read from the frame after assignment, so that comparison is
        # tautological.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=np.arange(0, 8, 2),
            columns=np.arange(0, 12, 3),
        )
        if has_ref:
            view = df[:]  # noqa: F841

        df.iloc[1, 1] = 1
        result = df.iloc[1, 1]
        assert result == 1

        df.iloc[:, 2:3] = 0
        expected = df.iloc[:, 2:3]
        result = df.iloc[:, 2:3]
        tm.assert_frame_equal(result, expected)

        # GH5771
        s = Series(0, index=[4, 5, 6])
        s.iloc[1:2] += 1
        expected = Series([0, 1, 0], index=[4, 5, 6])
        tm.assert_series_equal(s, expected)

    @pytest.mark.parametrize("has_ref", [True, False])
    def test_iloc_setitem_axis_argument(self, has_ref):
        # GH45032
        # [FUNCTION DOC]
        # GH 45032. Parametrized over reference existence. Using the axis argument of the accessor,
        # assigning a scalar by single position sets a row for axis 0 and a column for axis 1; an object
        # column stays object.
        df = DataFrame([[6, "c", 10], [7, "d", 11], [8, "e", 12]])
        df[1] = df[1].astype(object)
        if has_ref:
            view = df[:]
        expected = DataFrame([[6, "c", 10], [7, "d", 11], [5, 5, 5]])
        expected[1] = expected[1].astype(object)
        df.iloc(axis=0)[2] = 5
        tm.assert_frame_equal(df, expected)

        df = DataFrame([[6, "c", 10], [7, "d", 11], [8, "e", 12]])
        df[1] = df[1].astype(object)
        if has_ref:
            view = df[:]  # noqa: F841
        expected = DataFrame([[6, "c", 5], [7, "d", 5], [8, "e", 5]])
        expected[1] = expected[1].astype(object)
        df.iloc(axis=1)[2] = 5
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("has_ref", [True, False])
    def test_iloc_setitem_list(self, has_ref):
        # setitem with an iloc list
        # [FUNCTION DOC]
        # Parametrized over reference existence. Augmented assignment by lists of row and column
        # positions adds 100 to the intersection cells.
        # KNOWN ISSUE: a bare getitem statement with the same indexers appears before the assignment and
        # has no effect on the result.
        df = DataFrame(
            np.arange(9).reshape((3, 3)), index=["A", "B", "C"], columns=["A", "B", "C"]
        )
        if has_ref:
            view = df[:]  # noqa: F841
        df.iloc[[0, 1], [1, 2]]
        df.iloc[[0, 1], [1, 2]] += 100

        expected = DataFrame(
            np.array([0, 101, 102, 3, 104, 105, 6, 7, 8]).reshape((3, 3)),
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        tm.assert_frame_equal(df, expected)

    def test_iloc_setitem_pandas_object(self):
        # GH 17193
        # [FUNCTION DOC]
        # GH 17193. A series or an index of positions can be used as the positional key for assignment.
        s_orig = Series([0, 1, 2, 3])
        expected = Series([0, -1, -2, 3])

        s = s_orig.copy()
        s.iloc[Series([1, 2])] = [-1, -2]
        tm.assert_series_equal(s, expected)

        s = s_orig.copy()
        s.iloc[Index([1, 2])] = [-1, -2]
        tm.assert_series_equal(s, expected)

    def test_iloc_setitem_dups(self):
        # GH 6766
        # iloc with a mask aligning from another iloc
        # [FUNCTION DOC]
        # GH 6766. A mask-derived index of positions aligns assignment from another position selection
        # filling missing; deleting a duplicate column across blocks; assigning back to self; and a
        # reversed assignment applied twice leave the expected frame.
        df1 = DataFrame([{"A": None, "B": 1}, {"A": 2, "B": 2}])
        df2 = DataFrame([{"A": 3, "B": 3}, {"A": 4, "B": 4}])
        df = concat([df1, df2], axis=1)

        expected = df.fillna(3)
        inds = np.isnan(df.iloc[:, 0])
        mask = inds[inds].index
        df.iloc[mask, 0] = df.iloc[mask, 2]
        tm.assert_frame_equal(df, expected)

        # del a dup column across blocks
        expected = DataFrame({0: [1, 2], 1: [3, 4]})
        expected.columns = ["B", "B"]
        del df["A"]
        tm.assert_frame_equal(df, expected)

        # assign back to self
        df.iloc[[0, 1], [0, 1]] = df.iloc[[0, 1], [0, 1]]
        tm.assert_frame_equal(df, expected)

        # reversed x 2
        df.iloc[[1, 0], [0, 1]] = df.iloc[[1, 0], [0, 1]].reset_index(drop=True)
        df.iloc[[1, 0], [0, 1]] = df.iloc[[1, 0], [0, 1]].reset_index(drop=True)
        tm.assert_frame_equal(df, expected)

    def test_iloc_setitem_frame_duplicate_columns_multiple_blocks(self):
        # Same as the "assign back to self" check in test_iloc_setitem_dups
        #  but on a DataFrame with multiple blocks
        # [FUNCTION DOC]
        # Frame with duplicate columns: assigning a float version of integer values stays in place (a single
        # block); assigning values that cannot be held (adds 0.5) raises a type error about an invalid
        # value; assigning back to self leaves the frame unchanged.
        # KNOWN ISSUE: the name says multiple blocks, but the test asserts that there is exactly one
        # block.
        df = DataFrame([[0, 1], [2, 3]], columns=["B", "B"])

        # setting float values that can be held by existing integer arrays
        #  is inplace
        df.iloc[:, 0] = df.iloc[:, 0].astype("f8")
        assert len(df._mgr.blocks) == 1

        # if the assigned values cannot be held by existing integer arrays,
        #  we raise
        with pytest.raises(TypeError, match="Invalid value"):
            df.iloc[:, 0] = df.iloc[:, 0] + 0.5

        expected = df.copy()

        # assign back to self
        df.iloc[[0, 1], [0, 1]] = df.iloc[[0, 1], [0, 1]]

        tm.assert_frame_equal(df, expected)

    def test_iloc_setitem_unordered_column_indexer_referenced_block(self):
        # GH#65446 assignment was silently lost when the block was referenced
        #  by another object and the column indexer was not sorted
        # [FUNCTION DOC]
        # GH 65446. With the block referenced by another object and an unsorted column position list,
        # assigning a frame writes the values by position (swapped relative to column labels); the
        # referencing object is not changed; the internal manager integrity check is invoked.
        df = DataFrame({"A": [1, 2], "B": [3, 4]})
        df_orig = df.copy()
        ref = df[["A", "B"]]
        df.iloc[:, [1, 0]] = DataFrame({"A": [10, 20], "B": [30, 40]})
        df._mgr._verify_integrity()
        expected = DataFrame({"A": [30, 40], "B": [10, 20]})
        tm.assert_frame_equal(df, expected)
        tm.assert_frame_equal(ref, df_orig)

    def test_iloc_setitem_duplicate_column_indexer_referenced_block(self):
        # GH#65446
        # [FUNCTION DOC]
        # GH 65446. Same setup with a repeated column position: the last written value wins for the
        # repeated column; the reference is unchanged; integrity check invoked.
        df = DataFrame({"A": [1, 2], "B": [3, 4]})
        df_orig = df.copy()
        ref = df[["A", "B"]]
        df.iloc[:, [0, 0]] = np.array([[10, 30], [20, 40]])
        df._mgr._verify_integrity()
        expected = DataFrame({"A": [30, 40], "B": [3, 4]})
        tm.assert_frame_equal(df, expected)
        tm.assert_frame_equal(ref, df_orig)

    def test_iloc_setitem_frame_swap_columns(self):
        # GH#65446 positional swap when the RHS shares data with the target
        # [FUNCTION DOC]
        # GH 65446. Assigning a frame that shares data with the target into swapped positions swaps
        # the column values.
        df = DataFrame({"A": [1, 2], "B": [3, 4]})
        df.iloc[:, [1, 0]] = df[["A", "B"]]
        expected = DataFrame({"A": [3, 4], "B": [1, 2]})
        tm.assert_frame_equal(df, expected)

    def test_iloc_setitem_unordered_row_and_column_indexer_referenced_block(self):
        # GH#65446 a list row indexer combined with an unsorted column indexer
        #  must set the full cross product
        # [FUNCTION DOC]
        # GH 65446. List row positions combined with unsorted column positions set the full cross product
        # from an array; the reference is unchanged; integrity check invoked.
        df = DataFrame({"A": [1, 2], "B": [3, 4]})
        df_orig = df.copy()
        ref = df[["A", "B"]]
        df.iloc[[1, 0], [1, 0]] = np.array([[10, 20], [30, 40]])
        df._mgr._verify_integrity()
        expected = DataFrame({"A": [40, 20], "B": [30, 10]})
        tm.assert_frame_equal(df, expected)
        tm.assert_frame_equal(ref, df_orig)

    # TODO: GH#27620 this test used to compare iloc against ix; check if this
    #  is redundant with another test comparing iloc against loc
    def test_iloc_getitem_frame(self):
        # [FUNCTION DOC]
        # Positional scalar, slice, list, negative, duplicated and index-like selections equal the
        # corresponding label selections on an even-numbered index.
        # KNOWN ISSUE: a TODO notes the test formerly compared against a removed accessor and may be
        # redundant with other loc comparisons.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            index=range(0, 20, 2),
            columns=range(0, 8, 2),
        )

        result = df.iloc[2]
        exp = df.loc[4]
        tm.assert_series_equal(result, exp)

        result = df.iloc[2, 2]
        exp = df.loc[4, 4]
        assert result == exp

        # slice
        result = df.iloc[4:8]
        expected = df.loc[8:14]
        tm.assert_frame_equal(result, expected)

        result = df.iloc[:, 2:3]
        expected = df.loc[:, 4:5]
        tm.assert_frame_equal(result, expected)

        # list of integers
        result = df.iloc[[0, 1, 3]]
        expected = df.loc[[0, 2, 6]]
        tm.assert_frame_equal(result, expected)

        result = df.iloc[[0, 1, 3], [0, 1]]
        expected = df.loc[[0, 2, 6], [0, 2]]
        tm.assert_frame_equal(result, expected)

        # neg indices
        result = df.iloc[[-1, 1, 3], [-1, 1]]
        expected = df.loc[[18, 2, 6], [6, 2]]
        tm.assert_frame_equal(result, expected)

        # dups indices
        result = df.iloc[[-1, -1, 1, 3], [-1, 1]]
        expected = df.loc[[18, 18, 2, 6], [6, 2]]
        tm.assert_frame_equal(result, expected)

        # with index-like
        s = Series(index=range(1, 5), dtype=object)
        result = df.iloc[s.index]
        expected = df.loc[[2, 4, 6, 8]]
        tm.assert_frame_equal(result, expected)

    def test_iloc_getitem_labelled_frame(self):
        # try with labelled frame
        # [FUNCTION DOC]
        # On a string-labelled frame, positional scalar, slice and negative access match label access; an
        # out-of-bounds pair raises an index error; label strings raise a value error about the allowed
        # location indexer types.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            index=list("abcdefghij"),
            columns=list("ABCD"),
        )

        result = df.iloc[1, 1]
        exp = df.loc["b", "B"]
        assert result == exp

        result = df.iloc[:, 2:3]
        expected = df.loc[:, ["C"]]
        tm.assert_frame_equal(result, expected)

        # negative indexing
        result = df.iloc[-1, -1]
        exp = df.loc["j", "D"]
        assert result == exp

        # out-of-bounds exception
        msg = "index 5 is out of bounds for axis 0 with size 4|index out of bounds"
        with pytest.raises(IndexError, match=msg):
            df.iloc[10, 5]

        # trying to use a label
        msg = (
            r"Location based indexing can only have \[integer, integer "
            r"slice \(START point is INCLUDED, END point is EXCLUDED\), "
            r"listlike of integers, boolean array\] types"
        )
        with pytest.raises(ValueError, match=msg):
            df.iloc["j", "D"]

    def test_iloc_getitem_doc_issue(self):
        # multi axis slicing issue with single block
        # surfaced in GH 6059

        # [FUNCTION DOC]
        # GH 6059. Multi-axis slicing on a single-block frame returns the expected sub-frame for date
        # index, for duplicate column labels, and for an integer-labelled frame after touching the
        # internal block locations.
        # KNOWN ISSUE: the line touching the internal block locations is a bare attribute access that
        # has no effect on assertions.
        arr = np.random.default_rng(2).standard_normal((6, 4))
        index = date_range("20130101", periods=6)
        columns = list("ABCD")
        df = DataFrame(arr, index=index, columns=columns)

        # defines ref_locs
        df.describe()

        result = df.iloc[3:5, 0:2]

        expected = DataFrame(arr[3:5, 0:2], index=index[3:5], columns=columns[0:2])
        tm.assert_frame_equal(result, expected)

        # for dups
        df.columns = list("aaaa")
        result = df.iloc[3:5, 0:2]

        expected = DataFrame(arr[3:5, 0:2], index=index[3:5], columns=list("aa"))
        tm.assert_frame_equal(result, expected)

        # related
        arr = np.random.default_rng(2).standard_normal((6, 4))
        index = list(range(0, 12, 2))
        columns = list(range(0, 8, 2))
        df = DataFrame(arr, index=index, columns=columns)

        df._mgr.blocks[0].mgr_locs
        result = df.iloc[1:5, 2:4]
        expected = DataFrame(arr[1:5, 2:4], index=index[1:5], columns=columns[2:4])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("has_ref", [True, False])
    def test_iloc_setitem_series(self, has_ref):
        # [FUNCTION DOC]
        # Parametrized over reference existence. Scalar and slice assignment on frame and series read
        # back; strided slice assignment with lists fills a series in order.
        # KNOWN ISSUE: for the slice assignments, expected and result are both read from the object after
        # assignment, so those comparisons are tautological.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            index=list("abcdefghij"),
            columns=list("ABCD"),
        )
        if has_ref:
            view = df[:]  # noqa: F841

        df.iloc[1, 1] = 1
        result = df.iloc[1, 1]
        assert result == 1

        df.iloc[:, 2:3] = 0
        expected = df.iloc[:, 2:3]
        result = df.iloc[:, 2:3]
        tm.assert_frame_equal(result, expected)

        s = Series(np.random.default_rng(2).standard_normal(10), index=range(0, 20, 2))

        s.iloc[1] = 1
        result = s.iloc[1]
        assert result == 1

        s.iloc[:4] = 0
        expected = s.iloc[:4]
        result = s.iloc[:4]
        tm.assert_series_equal(result, expected)

        s = Series([-1] * 6)
        s.iloc[0::2] = [0, 2, 4]
        s.iloc[1::2] = [1, 3, 5]
        result = s
        expected = Series([0, 1, 2, 3, 4, 5])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("has_ref", [True, False])
    def test_iloc_setitem_list_of_lists(self, has_ref):
        # GH 7551
        # list-of-list is set incorrectly in mixed vs. single dtyped frames
        # [FUNCTION DOC]
        # GH 7551. Parametrized over reference existence. A list of lists assigned to a row slice sets
        # rows correctly for both an all-integer and a mixed-dtype frame.
        df = DataFrame(
            {"A": np.arange(5, dtype="int64"), "B": np.arange(5, 10, dtype="int64")}
        )
        if has_ref:
            view = df[:]
        df.iloc[2:4] = [[10, 11], [12, 13]]
        expected = DataFrame({"A": [0, 1, 10, 12, 4], "B": [5, 6, 11, 13, 9]})
        tm.assert_frame_equal(df, expected)

        df = DataFrame(
            {"A": ["a", "b", "c", "d", "e"], "B": np.arange(5, 10, dtype="int64")}
        )
        if has_ref:
            view = df[:]  # noqa: F841
        df.iloc[2:4] = [["x", 11], ["y", 13]]
        expected = DataFrame({"A": ["a", "b", "x", "y", "e"], "B": [5, 6, 11, 13, 9]})
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("has_ref", [True, False])
    @pytest.mark.parametrize("indexer", [[0], slice(None, 1, None), np.array([0])])
    @pytest.mark.parametrize("value", [["Z"], np.array(["Z"])])
    def test_iloc_setitem_with_scalar_index(self, has_ref, indexer, value):
        # GH #19474
        # assigning like "df.iloc[0, [0]] = ['Z']" should be evaluated
        # elementwisely, not using "setter('A', ['Z'])".

        # Set object type to avoid upcast when setting "Z"
        # [FUNCTION DOC]
        # GH 19474. Parametrized over reference existence, indexer (list, slice, array) and value (list,
        # array). Assigning a one-element list to a position and a one-element column indexer is done
        # elementwise: the cell becomes the scalar string.
        df = DataFrame([[1, 2], [3, 4]], columns=["A", "B"]).astype({"A": object})
        if has_ref:
            view = df[:]  # noqa: F841
        df.iloc[0, indexer] = value
        result = df.iloc[0, 0]

        assert is_scalar(result) and result == "Z"

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_iloc_mask(self):
        # GH 60994, iloc with a mask (of a series) should return accordingly
        # [FUNCTION DOC]
        # GH 60994. User warnings are ignored. A boolean series with labels cannot be used as a mask
        # (value error), a series with an unalignable index raises the indexing error, and a numpy
        # boolean array is fine. Then a table of expected outcomes for the nine combinations of mask
        # index kind (none, index, or integer positions) and accessor (plain, label, positional) is
        # checked: successful cases must equal the exact sum string; error cases must contain the message.
        # KNOWN ISSUE: the comment preceding the assertion ("Fix the assertion to check for substring
        # match") is a leftover review note, not a description.
        df = DataFrame(list(range(5)), index=list("ABCDE"), columns=["a"])
        mask = df.a % 2 == 0
        msg = "iLocation based boolean indexing cannot use an indexable as a mask"
        with pytest.raises(ValueError, match=msg):
            df.iloc[mask]

        mask.index = range(len(mask))
        msg = "Unalignable boolean Series provided as indexer"
        with pytest.raises(IndexingError, match=msg):
            df.iloc[mask]

        # ndarray ok
        result = df.iloc[np.array([True] * len(mask), dtype=bool)]
        tm.assert_frame_equal(result, df)

        # the possibilities
        locs = np.arange(4)
        nums = 2**locs
        reps = [bin(num) for num in nums]
        df = DataFrame({"locs": locs, "nums": nums}, reps)

        expected = {
            (None, ""): "0b1100",
            (None, ".loc"): "0b1100",
            (None, ".iloc"): "0b1100",
            ("index", ""): "0b11",
            ("index", ".loc"): "0b11",
            (
                "index",
                ".iloc",
            ): "iLocation based boolean indexing cannot use an indexable as a mask",
            ("locs", ""): "Unalignable boolean Series provided as indexer",
            ("locs", ".loc"): "Unalignable boolean Series provided as indexer",
            ("locs", ".iloc"): "Unalignable boolean Series provided as indexer",
        }

        # UserWarnings from reindex of a boolean mask
        for idx in [None, "index", "locs"]:
            mask = (df.nums > 2).values
            if idx:
                mask_index = getattr(df, idx)[::-1]
                mask = Series(mask, list(mask_index))
            for method in ["", ".loc", ".iloc"]:
                try:
                    if method:
                        accessor = getattr(df, method[1:])
                    else:
                        accessor = df
                    answer = str(bin(accessor[mask]["nums"].sum()))
                except (ValueError, IndexingError) as err:
                    answer = str(err)

                key = (
                    idx,
                    method,
                )
                expected_result = expected.get(key)

                # Fix the assertion to check for substring match
                if (
                    idx is None or (idx == "index" and method != ".iloc")
                ) and "0b" in expected_result:
                    # For successful numeric results, exact match is needed
                    assert expected_result == answer, (
                        f"[{key}] does not match [{answer}]"
                    )
                else:
                    # For error messages, substring match is sufficient
                    assert expected_result in answer, f"[{key}] not found in [{answer}]"

    def test_iloc_with_numpy_bool_array(self):
        # [FUNCTION DOC]
        # A numpy boolean array selects matching rows.
        df = DataFrame(list(range(5)), index=list("ABCDE"), columns=["a"])
        result = df.iloc[np.array([True, False, True, False, True], dtype=bool)]
        expected = DataFrame({"a": [0, 2, 4]}, index=["A", "C", "E"])
        tm.assert_frame_equal(result, expected)

    def test_iloc_series_mask_with_index_mismatch_raises(self):
        # [FUNCTION DOC]
        # A boolean series whose index doesn't match the frame raises the unalignable-indexer error.
        df = DataFrame(list(range(5)), index=list("ABCDE"), columns=["a"])
        mask = df.a % 2 == 0
        msg = "Unalignable boolean Series provided as indexer"
        with pytest.raises(IndexingError, match=msg):
            df.iloc[Series([True] * len(mask), dtype=bool)]

    def test_iloc_series_mask_all_true(self):
        # [FUNCTION DOC]
        # A boolean series with the same default index, all true, returns the whole frame.
        df = DataFrame(list(range(5)), columns=["a"])
        mask = Series([True] * len(df), dtype=bool)
        result = df.iloc[mask]
        tm.assert_frame_equal(result, df)

    def test_iloc_series_mask_alternate_true(self):
        # [FUNCTION DOC]
        # A boolean series with the same index and alternating values selects the true rows.
        df = DataFrame(list(range(5)), columns=["a"])
        mask = Series([True, False, True, False, True], dtype=bool)
        result = df.iloc[mask]
        expected = DataFrame({"a": [0, 2, 4]}, index=[0, 2, 4])
        tm.assert_frame_equal(result, expected)

    def test_iloc_non_unique_indexing(self):
        # GH 4017, non-unique indexing (on the axis)
        # [FUNCTION DOC]
        # GH 4017. Positional selection on a frame concatenated three times with a large index works
        # and equals the same positions on the original; label selection with the same positions on a
        # shorter frame raises a key error.
        df = DataFrame({"A": [0.1] * 3000, "B": [1] * 3000})
        idx = np.arange(30) * 99
        expected = df.iloc[idx]

        df3 = concat([df, 2 * df, 3 * df])
        result = df3.iloc[idx]

        tm.assert_frame_equal(result, expected)

        df2 = DataFrame({"A": [0.1] * 1000, "B": [1] * 1000})
        df2 = concat([df2, 2 * df2, 3 * df2])

        with pytest.raises(KeyError, match="not in index"):
            df2.loc[idx]

    def test_iloc_empty_list_indexer_is_ok(self):
        # [FUNCTION DOC]
        # An empty list on either axis returns an empty frame equal to the zero-length slice, with the
        # index and column types preserved.
        df = DataFrame(
            np.ones((5, 2)),
            index=Index([f"i-{i}" for i in range(5)], name="a"),
            columns=Index([f"i-{i}" for i in range(2)], name="a"),
        )
        # vertical empty
        tm.assert_frame_equal(
            df.iloc[:, []],
            df.iloc[:, :0],
            check_index_type=True,
            check_column_type=True,
        )
        # horizontal empty
        tm.assert_frame_equal(
            df.iloc[[], :],
            df.iloc[:0, :],
            check_index_type=True,
            check_column_type=True,
        )
        # horizontal empty
        tm.assert_frame_equal(
            df.iloc[[]], df.iloc[:0, :], check_index_type=True, check_column_type=True
        )

    def test_identity_slice_returns_new_object(self):
        # GH13873
        # [FUNCTION DOC]
        # GH 13873. A full slice returns a new object that initially shares memory. After writing to the
        # original column, the sliced frame is unchanged (copy-on-write), and likewise for a series.
        # KNOWN ISSUE: a comment says the assignment alters both depending on CoW, but the assertion
        # shows only the original is changed.
        original_df = DataFrame({"a": [1, 2, 3]})
        sliced_df = original_df.iloc[:]
        assert sliced_df is not original_df

        # should be a shallow copy
        assert np.shares_memory(original_df["a"], sliced_df["a"])

        # Setting using .loc[:, "a"] sets inplace so alters both sliced and orig
        # depending on CoW
        original_df.loc[:, "a"] = [4, 4, 4]
        assert (sliced_df["a"] == [1, 2, 3]).all()

        original_series = Series([1, 2, 3, 4, 5, 6])
        sliced_series = original_series.iloc[:]
        assert sliced_series is not original_series

        # should also be a shallow copy
        original_series[:3] = [7, 8, 9]
        # shallow copy not updated (CoW)
        assert all(sliced_series[:3] == [1, 2, 3])

    def test_indexing_zerodim_np_array(self):
        # GH24919
        # [FUNCTION DOC]
        # GH 24919. A zero-dimensional array as position on a frame returns the row series.
        df = DataFrame([[1, 2], [3, 4]])
        result = df.iloc[np.array(0)]
        s = Series([1, 2], name=0)
        tm.assert_series_equal(result, s)

    def test_series_indexing_zerodim_np_array(self):
        # GH24919
        # [FUNCTION DOC]
        # GH 24919. A zero-dimensional array position on a series returns the scalar.
        s = Series([1, 2])
        result = s.iloc[np.array(0)]
        assert result == 1

    def test_iloc_setitem_categorical_updates_inplace(self):
        # Mixed dtype ensures we go through take_split_path in setitem_with_indexer
        # [FUNCTION DOC]
        # Without copying, setting a column of mixed-dtype frame from a reversed categorical modifies the
        # original categorical in place (shared memory), giving reversed categories.
        cat = Categorical(["A", "B", "C"])
        df = DataFrame({1: cat, 2: [1, 2, 3]}, copy=False)

        assert tm.shares_memory(df[1], cat)

        # With the enforcement of GH#45333 in 2.0, this modifies original
        #  values inplace
        df.iloc[:, 0] = cat[::-1]

        assert tm.shares_memory(df[1], cat)
        expected = Categorical(["C", "B", "A"], categories=["A", "B", "C"])
        tm.assert_categorical_equal(cat, expected)

    def test_iloc_with_boolean_operation(self):
        # GH 20627
        # [FUNCTION DOC]
        # GH 20627. Augmented multiplication and division through boolean masks (arrays and lists) apply
        # only on the selected rows, preserving NaN.
        result = DataFrame([[0, 1], [2, 3], [4, 5], [6, np.nan]])
        result.iloc[result.index <= 2] *= 2
        expected = DataFrame([[0, 2], [4, 6], [8, 10], [6, np.nan]])
        tm.assert_frame_equal(result, expected)

        result.iloc[result.index > 2] *= 2
        expected = DataFrame([[0, 2], [4, 6], [8, 10], [12, np.nan]])
        tm.assert_frame_equal(result, expected)

        result.iloc[[True, True, False, False]] *= 2
        expected = DataFrame([[0, 4], [8, 12], [8, 10], [12, np.nan]])
        tm.assert_frame_equal(result, expected)

        result.iloc[[False, False, True, True]] /= 2
        expected = DataFrame([[0, 4.0], [8, 12.0], [4, 5.0], [6, np.nan]])
        tm.assert_frame_equal(result, expected)

    def test_iloc_getitem_singlerow_slice_categoricaldtype_gives_series(self):
        # GH#29521
        # [FUNCTION DOC]
        # GH 29521. A positional row of a one-column categorical frame is a categorical series.
        df = DataFrame({"x": Categorical("a b c d e".split())})
        result = df.iloc[0]
        raw_cat = Categorical(["a"], categories=["a", "b", "c", "d", "e"])
        expected = Series(raw_cat, index=["x"], name=0, dtype="category")

        tm.assert_series_equal(result, expected)

    def test_iloc_getitem_categorical_values(self):
        # GH#14580
        # test iloc() on Series with Categorical data

        # [FUNCTION DOC]
        # GH 14580. Slice, list and boolean list on a categorical series return categorical values
        # keeping all categories.
        ser = Series([1, 2, 3]).astype("category")

        # get slice
        result = ser.iloc[0:2]
        expected = Series([1, 2]).astype(CategoricalDtype([1, 2, 3]))
        tm.assert_series_equal(result, expected)

        # get list of indexes
        result = ser.iloc[[0, 1]]
        expected = Series([1, 2]).astype(CategoricalDtype([1, 2, 3]))
        tm.assert_series_equal(result, expected)

        # get boolean array
        result = ser.iloc[[True, False, False]]
        expected = Series([1]).astype(CategoricalDtype([1, 2, 3]))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("value", [None, NaT, np.nan])
    def test_iloc_setitem_td64_values_cast_na(self, value):
        # GH#18586
        # [FUNCTION DOC]
        # GH 18586. Parametrized over None, NaT, NaN. Assigning any of them to a timedelta series
        # position gives NaT.
        series = Series([0, 1, 2], dtype="timedelta64[ns]")
        series.iloc[0] = value
        expected = Series([NaT, 1, 2], dtype="timedelta64[ns]")
        tm.assert_series_equal(series, expected)

    @pytest.mark.parametrize("not_na", [Interval(0, 1), "a", 1.0])
    def test_setitem_mix_of_nan_and_interval(self, not_na, nulls_fixture):
        # GH#27937
        # [FUNCTION DOC]
        # GH 27937. Parametrized over a valid category (interval, string, float) and the missing-value
        # fixtures. Assigning a list mixing missing values and the category into a categorical
        # series of missing values works and keeps the dtype.
        dtype = CategoricalDtype(categories=[not_na])
        ser = Series(
            [nulls_fixture, nulls_fixture, nulls_fixture, nulls_fixture], dtype=dtype
        )
        ser.iloc[:3] = [nulls_fixture, not_na, nulls_fixture]
        exp = Series([nulls_fixture, not_na, nulls_fixture, nulls_fixture], dtype=dtype)
        tm.assert_series_equal(ser, exp)

    def test_iloc_setitem_empty_frame_raises_with_3d_ndarray(self):
        # [FUNCTION DOC]
        # Assigning through a three-dimensional array indexer on an empty frame raises a value error
        # about values having more dimensions than the object.
        idx = Index([])
        obj = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), len(idx))),
            index=idx,
            columns=idx,
        )
        nd3 = np.random.default_rng(2).integers(5, size=(2, 2, 2))

        msg = f"Cannot set values with ndim > {obj.ndim}"
        with pytest.raises(ValueError, match=msg):
            obj.iloc[nd3] = 0

    def test_iloc_getitem_read_only_values(self, indexer_li):
        # GH#10043 this is fundamentally a test for iloc, but test loc while
        #  we're here
        # [FUNCTION DOC]
        # GH 10043. Parametrized over the getter fixture. Frames built on writable and read-only arrays
        # return equal results for list, single-list, scalar and slice positional selection.
        rw_array = np.eye(10)
        rw_df = DataFrame(rw_array)

        ro_array = np.eye(10)
        ro_array.setflags(write=False)
        ro_df = DataFrame(ro_array)

        tm.assert_frame_equal(
            indexer_li(rw_df)[[1, 2, 3]], indexer_li(ro_df)[[1, 2, 3]]
        )
        tm.assert_frame_equal(indexer_li(rw_df)[[1]], indexer_li(ro_df)[[1]])
        tm.assert_series_equal(indexer_li(rw_df)[1], indexer_li(ro_df)[1])
        tm.assert_frame_equal(indexer_li(rw_df)[1:3], indexer_li(ro_df)[1:3])

    def test_iloc_getitem_readonly_key(self):
        # GH#17192 iloc with read-only array raising TypeError
        # [FUNCTION DOC]
        # GH 17192. A read-only integer array as key works for frame and series and equals label
        # selection.
        df = DataFrame({"data": np.ones(100, dtype="float64")})
        indices = np.array([1, 3, 6])
        indices.flags.writeable = False

        result = df.iloc[indices]
        expected = df.loc[[1, 3, 6]]
        tm.assert_frame_equal(result, expected)

        result = df["data"].iloc[indices]
        expected = df["data"].loc[[1, 3, 6]]
        tm.assert_series_equal(result, expected)

    def test_iloc_assign_series_to_df_cell(self):
        # GH 37593
        # [FUNCTION DOC]
        # GH 37593. Assigning a series into one cell of an empty frame stores the series as an element.
        df = DataFrame(columns=["a"], index=[0])
        df.iloc[0, 0] = Series([1, 2, 3])
        expected = DataFrame({"a": [Series([1, 2, 3])]}, columns=["a"], index=[0])
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("klass", [list, np.array])
    def test_iloc_setitem_bool_indexer(self, klass):
        # GH#36741
        # [FUNCTION DOC]
        # GH 36741. Parametrized over list and array boolean indexers. Assignment of doubled values to
        # the selected rows of a column works.
        df = DataFrame({"flag": ["x", "y", "z"], "value": [1, 3, 4]})
        indexer = klass([True, False, False])
        df.iloc[indexer, 1] = df.iloc[indexer, 1] * 2
        expected = DataFrame({"flag": ["x", "y", "z"], "value": [2, 3, 4]})
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("has_ref", [True, False])
    @pytest.mark.parametrize("indexer", [[1], slice(1, 2)])
    def test_iloc_setitem_pure_position_based(self, indexer, has_ref):
        # GH#22046
        # [FUNCTION DOC]
        # GH 22046. Parametrized over reference existence and a list or slice column position.
        # Assigning one column from another frame is positional, not label-aligned.
        df1 = DataFrame({"a2": [11, 12, 13], "b2": [14, 15, 16]})
        df2 = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        if has_ref:
            view = df2[:]  # noqa: F841
        df2.iloc[:, indexer] = df1.iloc[:, [0]]
        expected = DataFrame({"a": [1, 2, 3], "b": [11, 12, 13], "c": [7, 8, 9]})
        tm.assert_frame_equal(df2, expected)

    @pytest.mark.parametrize("has_ref", [True, False])
    def test_iloc_setitem_dictionary_value(self, has_ref):
        # GH#37728
        # [FUNCTION DOC]
        # GH 37728, GH 38335. Parametrized over reference existence. Assigning a mapping to a row sets
        # values by matching column labels, for integer and mixed-dtype frames.
        df = DataFrame({"x": [1, 2], "y": [2, 2]})
        if has_ref:
            view = df[:]
        rhs = {"x": 9, "y": 99}
        df.iloc[1] = rhs
        expected = DataFrame({"x": [1, 9], "y": [2, 99]})
        tm.assert_frame_equal(df, expected)

        # GH#38335 same thing, mixed dtypes
        df = DataFrame({"x": [1, 2], "y": [2.0, 2.0]})
        if has_ref:
            view = df[:]  # noqa: F841
        df.iloc[1] = rhs
        expected = DataFrame({"x": [1, 9], "y": [2.0, 99.0]})
        tm.assert_frame_equal(df, expected)

    def test_iloc_getitem_float_duplicates(self):
        # [FUNCTION DOC]
        # Label lookups of a duplicated float label on indexes of various orders return the matching
        # positional rows, both for frame and for a column.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            index=[0.1, 0.2, 0.2],
            columns=list("abc"),
        )
        expect = df.iloc[1:]
        tm.assert_frame_equal(df.loc[0.2], expect)

        expect = df.iloc[1:, 0]
        tm.assert_series_equal(df.loc[0.2, "a"], expect)

        df.index = [1, 0.2, 0.2]
        expect = df.iloc[1:]
        tm.assert_frame_equal(df.loc[0.2], expect)

        expect = df.iloc[1:, 0]
        tm.assert_series_equal(df.loc[0.2, "a"], expect)

        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 3)),
            index=[1, 0.2, 0.2, 1],
            columns=list("abc"),
        )
        expect = df.iloc[1:-1]
        tm.assert_frame_equal(df.loc[0.2], expect)

        expect = df.iloc[1:-1, 0]
        tm.assert_series_equal(df.loc[0.2, "a"], expect)

        df.index = [0.1, 0.2, 2, 0.2]
        expect = df.iloc[[1, -1]]
        tm.assert_frame_equal(df.loc[0.2], expect)

        expect = df.iloc[[1, -1], 0]
        tm.assert_series_equal(df.loc[0.2, "a"], expect)

    def test_iloc_setitem_custom_object(self):
        # iloc with an object
        # [FUNCTION DOC]
        # A frame cell can hold an instance of a custom class; assigning twice leaves the last value;
        # assigning missing after an object keeps the object dtype with a missing value.
        # KNOWN ISSUE: in the second scenario the expected frame is built but never assigned the missing
        # value, so the final comparison is between the frame after the sequence and an untouched empty
        # frame; the comment says dtype remains object.
        class TO:
            # [CLASS DOC]
            # HELPER: small value class for this test. Stores one wrapped value, prints as that value in
            # square brackets, compares equal by wrapped value, and provides a method returning itself.
            def __init__(self, value) -> None:
                # [FUNCTION DOC]
                # HELPER: stores the argument value as the attribute value.
                self.value = value

            def __str__(self) -> str:
                # [FUNCTION DOC]
                # HELPER: returns the text of the wrapped value in square brackets. The class also aliases its
                # representation to this method.
                return f"[{self.value}]"

            __repr__ = __str__

            def __eq__(self, other) -> bool:
                # [FUNCTION DOC]
                # HELPER: returns whether the wrapped value equals the other object's wrapped value; fails with
                # an attribute error if the other object lacks a value attribute.
                return self.value == other.value

            def view(self):
                # [FUNCTION DOC]
                # HELPER: returns the object itself; present so the object resembles an array view when pandas
                # calls that method.
                return self

        df = DataFrame(index=[0, 1], columns=[0])
        df.iloc[1, 0] = TO(1)
        df.iloc[1, 0] = TO(2)

        result = DataFrame(index=[0, 1], columns=[0])
        result.iloc[1, 0] = TO(2)

        tm.assert_frame_equal(result, df)

        # remains object dtype even after setting it back
        df = DataFrame(index=[0, 1], columns=[0])
        df.iloc[1, 0] = TO(1)
        df.iloc[1, 0] = np.nan
        result = DataFrame(index=[0, 1], columns=[0])

        tm.assert_frame_equal(result, df)

    def test_iloc_getitem_with_duplicates(self):
        # [FUNCTION DOC]
        # On a frame with duplicate row labels, a positional row and a transposed positional column are
        # series with the same values.
        df = DataFrame(
            np.random.default_rng(2).random((3, 3)),
            columns=list("ABC"),
            index=list("aab"),
        )

        result = df.iloc[0]
        assert isinstance(result, Series)
        tm.assert_almost_equal(result.values, df.values[0])

        result = df.T.iloc[:, 0]
        assert isinstance(result, Series)
        tm.assert_almost_equal(result.values, df.values[0])

    def test_iloc_getitem_with_duplicates2(self):
        # GH#2259
        # [FUNCTION DOC]
        # GH 2259. Positional list column selection on duplicate column labels equals take on that
        # position.
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=[1, 1, 2])
        result = df.iloc[:, [0]]
        expected = df.take([0], axis=1)
        tm.assert_frame_equal(result, expected)

    def test_iloc_interval(self):
        # GH#17130
        # [FUNCTION DOC]
        # GH 17130. Interval-valued column labels: positional row and column selections give the right
        # series, and augmented assignment on the column works.
        df = DataFrame({Interval(1, 2): [1, 2]})

        result = df.iloc[0]
        expected = Series({Interval(1, 2): 1}, name=0)
        tm.assert_series_equal(result, expected)

        result = df.iloc[:, 0]
        expected = Series([1, 2], name=Interval(1, 2))
        tm.assert_series_equal(result, expected)

        result = df.copy()
        result.iloc[:, 0] += 1
        expected = DataFrame({Interval(1, 2): [2, 3]})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("indexing_func", [list, np.array])
    @pytest.mark.parametrize("rhs_func", [list, np.array])
    def test_loc_setitem_boolean_list(self, rhs_func, indexing_func):
        # GH#20438 testing specifically list key, not arraylike
        # [FUNCTION DOC]
        # GH 20438. Parametrized over indexer container (list, array) and right-hand container (list,
        # array). Boolean positional indexer assignment on a series and on a one-column frame sets the
        # selected elements.
        # KNOWN ISSUE: the name says loc, but the test exercises the positional accessor.
        ser = Series([0, 1, 2])
        ser.iloc[indexing_func([True, False, True])] = rhs_func([5, 10])
        expected = Series([5, 1, 10])
        tm.assert_series_equal(ser, expected)

        df = DataFrame({"a": [0, 1, 2]})
        df.iloc[indexing_func([True, False, True])] = rhs_func([[5], [10]])
        expected = DataFrame({"a": [5, 1, 10]})
        tm.assert_frame_equal(df, expected)

    def test_iloc_getitem_slice_negative_step_ea_block(self):
        # GH#44551
        # [FUNCTION DOC]
        # GH 44551. Reversed column slice on an extension-array frame returns the same frame; with an
        # extra object column the column order is reversed.
        df = DataFrame({"A": [1, 2, 3]}, dtype="Int64")

        res = df.iloc[:, ::-1]
        tm.assert_frame_equal(res, df)

        df["B"] = "foo"
        res = df.iloc[:, ::-1]
        expected = DataFrame({"B": df["B"], "A": df["A"]})
        tm.assert_frame_equal(res, expected)

    def test_iloc_setitem_2d_ndarray_into_ea_block(self):
        # GH#44703
        # [FUNCTION DOC]
        # GH 44703. Assigning a two-dimensional string array to categorical column positions sets the
        # values and keeps the dtype.
        df = DataFrame({"status": ["a", "b", "c"]}, dtype="category")
        df.iloc[np.array([0, 1]), np.array([0])] = np.array([["a"], ["a"]])

        expected = DataFrame({"status": ["a", "a", "c"]}, dtype=df["status"].dtype)
        tm.assert_frame_equal(df, expected)

    def test_iloc_getitem_int_single_ea_block_view(self):
        # GH#45241
        # TODO: make an extension interface test for this?
        # [FUNCTION DOC]
        # GH 45241. A row taken from a single extension-array-block frame is a view: modifying the
        # original array element changes the series.
        # KNOWN ISSUE: a TODO suggests this belongs in the extension interface tests.
        arr = interval_range(1, 10.0)._values
        df = DataFrame(arr, copy=False)

        # ser should be a *view* on the DataFrame data
        ser = df.iloc[2]

        # if we have a view, then changing arr[2] should also change ser[0]
        assert arr[2] != arr[-1]  # otherwise the rest isn't meaningful
        arr[2] = arr[-1]
        assert ser[0] == arr[-1]

    def test_iloc_setitem_multicolumn_to_datetime(self, using_infer_string):
        # GH#20511
        # [FUNCTION DOC]
        # GH 20511. Uses the infer-string fixture. Assigning a datetime frame to a string column via
        # column list: if string inference is on, raises a type error (invalid value); otherwise the column
        # becomes timestamps and the other column is kept (dtype not checked).
        df = DataFrame({"A": ["2022-01-01", "2022-01-02"], "B": ["2021", "2022"]})

        if using_infer_string:
            with pytest.raises(TypeError, match="Invalid value"):
                df.iloc[:, [0]] = DataFrame({"A": to_datetime(["2021", "2022"])})
        else:
            df.iloc[:, [0]] = DataFrame({"A": to_datetime(["2021", "2022"])})
            expected = DataFrame(
                {
                    "A": [
                        Timestamp("2021-01-01 00:00:00"),
                        Timestamp("2022-01-01 00:00:00"),
                    ],
                    "B": ["2021", "2022"],
                }
            )
            tm.assert_frame_equal(df, expected, check_dtype=False)


class TestILocErrors:
    # NB: this test should work for _any_ Series we can pass as
    #  series_with_simple_index
    # [CLASS DOC]
    # Test collection for iloc error behavior.
    def test_iloc_float_raises(self, series_with_simple_index, frame_or_series):
        # GH#4892
        # float_indexers should raise exceptions
        # on appropriate Index types & accessors
        # this duplicates the code below
        # but is specifically testing for the error
        # message

        # [FUNCTION DOC]
        # GH 4892. Uses the fixture for a series with a simple index and frame-or-series. A float key
        # raises a type error about non-integer location for getting and an index error with numpy's
        # message for setting.
        # KNOWN ISSUE: a comment says the test duplicates code below it, but no such code follows in this
        # class.
        obj = series_with_simple_index
        if frame_or_series is DataFrame:
            obj = obj.to_frame()

        msg = "Cannot index by location index with a non-integer key"
        with pytest.raises(TypeError, match=msg):
            obj.iloc[3.0]

        with pytest.raises(IndexError, match=_slice_iloc_msg):
            obj.iloc[3.0] = 0

    @pytest.mark.parametrize("has_ref", [True, False])
    def test_iloc_getitem_setitem_fancy_exceptions(self, float_frame, has_ref):
        # [FUNCTION DOC]
        # Parametrized over reference existence. Too many indexers raises the indexing error for
        # getting; for setting, numpy's index error about too many indices is surfaced (GH 32257).
        with pytest.raises(IndexingError, match="Too many indexers"):
            float_frame.iloc[:, :, :]

        if has_ref:
            view = float_frame[:]  # noqa: F841
        with pytest.raises(IndexError, match="too many indices for array"):
            # GH#32257 we let numpy do validation, get their exception
            float_frame.iloc[:, :, :] = 1

    def test_iloc_frame_indexer(self):
        # GH#39004
        # [FUNCTION DOC]
        # GH 39004. A frame used as indexer raises a type error for setting and an index error for
        # getting, each recommending label-based indexing.
        df = DataFrame({"a": [1, 2, 3]})
        indexer = DataFrame({"a": [True, False, True]})
        msg = "DataFrame indexer for .iloc is not supported. Consider using .loc"
        with pytest.raises(TypeError, match=msg):
            df.iloc[indexer] = 1

        msg = (
            "DataFrame indexer is not allowed for .iloc\n"
            "Consider using .loc for automatic alignment."
        )
        with pytest.raises(IndexError, match=msg):
            df.iloc[indexer]


class TestILocSetItemDuplicateColumns:
    # [CLASS DOC]
    # Test collection for iloc setitem on frames with duplicate column labels.
    def test_iloc_setitem_scalar_duplicate_columns(self):
        # GH#15686, duplicate columns and mixed dtype
        # [FUNCTION DOC]
        # GH 15686. Scalar assignment into mixed-dtype duplicate columns changes only that cell and
        # leaves the other column's dtype intact.
        df1 = DataFrame([{"A": None, "B": 1}, {"A": 2, "B": 2}])
        df2 = DataFrame([{"A": 3, "B": 3}, {"A": 4, "B": 4}])
        df = concat([df1, df2], axis=1)
        df.iloc[0, 0] = -1

        assert df.iloc[0, 0] == -1
        assert df.iloc[0, 2] == 3
        assert df.dtypes.iloc[2] == np.int64

    def test_iloc_setitem_list_duplicate_columns(self):
        # GH#22036 setting with same-sized list
        # [FUNCTION DOC]
        # GH 22036. A same-length list assigned to one of duplicate columns sets only that column.
        df = DataFrame([[0, "str", "str2"]], columns=["a", "b", "b"])

        df.iloc[:, 2] = ["str3"]

        expected = DataFrame([[0, "str", "str3"]], columns=["a", "b", "b"])
        tm.assert_frame_equal(df, expected)

    def test_iloc_setitem_series_duplicate_columns(self):
        # [FUNCTION DOC]
        # Assigning float values to the first column does not change the dtype of the other column
        # with the same label.
        df = DataFrame(
            np.arange(8, dtype=np.int64).reshape(2, 4), columns=["A", "B", "A", "B"]
        )
        df.iloc[:, 0] = df.iloc[:, 0].astype(np.float64)
        assert df.dtypes.iloc[2] == np.int64

    @pytest.mark.parametrize(
        ["dtypes", "init_value", "expected_value"],
        [("int64", "0", 0), ("float", "1.2", 1.2)],
    )
    def test_iloc_setitem_dtypes_duplicate_columns(
        self, dtypes, init_value, expected_value
    ):
        # GH#22035
        # [FUNCTION DOC]
        # GH 22035. Parametrized over (dtype name, initial string, expected value). Casting an
        # object column via assignment sets the values in place so the frame stays object dtype.
        df = DataFrame(
            [[init_value, "str", "str2"]], columns=["a", "b", "b"], dtype=object
        )

        # with the enforcement of GH#45333 in 2.0, this sets values inplace,
        #  so we retain object dtype
        df.iloc[:, 0] = df.iloc[:, 0].astype(dtypes)

        expected_df = DataFrame(
            [[expected_value, "str", "str2"]],
            columns=["a", "b", "b"],
            dtype=object,
        )
        tm.assert_frame_equal(df, expected_df)


class TestILocCallable:
    # [CLASS DOC]
    # Test collection for callables as iloc keys.
    def test_frame_iloc_getitem_callable(self):
        # GH#11485
        # [FUNCTION DOC]
        # GH 11485. Callable row and/or column keys (returning list or scalar) alone or mixed with
        # literal keys give the same result as the equivalent literal positions.
        df = DataFrame({"X": [1, 2, 3, 4], "Y": list("aabb")}, index=list("ABCD"))

        # return location
        res = df.iloc[lambda x: [1, 3]]
        tm.assert_frame_equal(res, df.iloc[[1, 3]])

        res = df.iloc[lambda x: [1, 3], :]
        tm.assert_frame_equal(res, df.iloc[[1, 3], :])

        res = df.iloc[lambda x: [1, 3], lambda x: 0]
        tm.assert_series_equal(res, df.iloc[[1, 3], 0])

        res = df.iloc[lambda x: [1, 3], lambda x: [0]]
        tm.assert_frame_equal(res, df.iloc[[1, 3], [0]])

        # mixture
        res = df.iloc[[1, 3], lambda x: 0]
        tm.assert_series_equal(res, df.iloc[[1, 3], 0])

        res = df.iloc[[1, 3], lambda x: [0]]
        tm.assert_frame_equal(res, df.iloc[[1, 3], [0]])

        res = df.iloc[lambda x: [1, 3], 0]
        tm.assert_series_equal(res, df.iloc[[1, 3], 0])

        res = df.iloc[lambda x: [1, 3], [0]]
        tm.assert_frame_equal(res, df.iloc[[1, 3], [0]])

    def test_frame_iloc_setitem_callable(self):
        # GH#11485
        # [FUNCTION DOC]
        # GH 11485. Setting through callable row and/or column keys (mixed with literals) gives the same
        # frame as the equivalent literal positions.
        df = DataFrame(
            {"X": [1, 2, 3, 4], "Y": Series(list("aabb"), dtype=object)},
            index=list("ABCD"),
        )

        # return location
        res = df.copy()
        res.iloc[lambda x: [1, 3]] = 0
        exp = df.copy()
        exp.iloc[[1, 3]] = 0
        tm.assert_frame_equal(res, exp)

        res = df.copy()
        res.iloc[lambda x: [1, 3], :] = -1
        exp = df.copy()
        exp.iloc[[1, 3], :] = -1
        tm.assert_frame_equal(res, exp)

        res = df.copy()
        res.iloc[lambda x: [1, 3], lambda x: 0] = 5
        exp = df.copy()
        exp.iloc[[1, 3], 0] = 5
        tm.assert_frame_equal(res, exp)

        res = df.copy()
        res.iloc[lambda x: [1, 3], lambda x: [0]] = 25
        exp = df.copy()
        exp.iloc[[1, 3], [0]] = 25
        tm.assert_frame_equal(res, exp)

        # mixture
        res = df.copy()
        res.iloc[[1, 3], lambda x: 0] = -3
        exp = df.copy()
        exp.iloc[[1, 3], 0] = -3
        tm.assert_frame_equal(res, exp)

        res = df.copy()
        res.iloc[[1, 3], lambda x: [0]] = -5
        exp = df.copy()
        exp.iloc[[1, 3], [0]] = -5
        tm.assert_frame_equal(res, exp)

        res = df.copy()
        res.iloc[lambda x: [1, 3], 0] = 10
        exp = df.copy()
        exp.iloc[[1, 3], 0] = 10
        tm.assert_frame_equal(res, exp)

        res = df.copy()
        res.iloc[lambda x: [1, 3], [0]] = [-5, -5]
        exp = df.copy()
        exp.iloc[[1, 3], [0]] = [-5, -5]
        tm.assert_frame_equal(res, exp)


class TestILocSeries:
    # [CLASS DOC]
    # Test collection for iloc on series.
    def test_iloc(self):
        # [FUNCTION DOC]
        # Each positional scalar equals the label lookup; a slice equals the label slice; modifying a slice
        # result does not warn and does not alter the original; a list of positions equals reindexing.
        ser = Series(
            np.random.default_rng(2).standard_normal(10), index=list(range(0, 20, 2))
        )
        ser_original = ser.copy()

        for i in range(len(ser)):
            result = ser.iloc[i]
            exp = ser[ser.index[i]]
            tm.assert_almost_equal(result, exp)

        # pass a slice
        result = ser.iloc[slice(1, 3)]
        expected = ser.loc[2:4]
        tm.assert_series_equal(result, expected)

        # test slice is a view
        with tm.assert_produces_warning(None):
            # GH#45324 make sure we aren't giving a spurious FutureWarning
            result[:] = 0
        tm.assert_series_equal(ser, ser_original)

        # list of integers
        result = ser.iloc[[0, 2, 3, 4, 5]]
        expected = ser.reindex(ser.index[[0, 2, 3, 4, 5]])
        tm.assert_series_equal(result, expected)

    def test_iloc_getitem_nonunique(self):
        # [FUNCTION DOC]
        # Positional access on a series with a duplicated index returns the third element.
        ser = Series([0, 1, 2], index=[0, 1, 0])
        assert ser.iloc[2] == 2

    def test_iloc_setitem_pure_position_based(self):
        # GH#22046
        # [FUNCTION DOC]
        # GH 22046. Assigning part of a series to another uses positions, not labels.
        ser1 = Series([1, 2, 3])
        ser2 = Series([4, 5, 6], index=[1, 0, 2])
        ser1.iloc[1:3] = ser2.iloc[1:3]
        expected = Series([1, 5, 6])
        tm.assert_series_equal(ser1, expected)

    def test_iloc_nullable_int64_size_1_nan(self):
        # GH 31861
        # [FUNCTION DOC]
        # GH 31861. Assigning a nullable integer series of missing to a string column through label
        # assignment raises a type error about an invalid value.
        # KNOWN ISSUE: the test lives in the series class and is named for iloc, but exercises label
        # based column assignment on a frame.
        result = DataFrame({"a": ["test"], "b": [np.nan]})

        ser = Series([NA], name="b", dtype="Int64")
        with pytest.raises(TypeError, match="Invalid value"):
            result.loc[:, "b"] = ser

    def test_iloc_arrow_extension_array(self):
        # GH#61311
        # [FUNCTION DOC]
        # GH 61311. Skipped without pyarrow. Selecting columns with an arrow-backed integer column as
        # positional key gives the same result as with numpy-backed data (dtype not checked).
        pytest.importorskip("pyarrow")
        df = DataFrame({"a": [1, 2], "c": [0, 2], "d": ["c", "a"]})
        df_arrow = DataFrame(
            {"a": [1, 2], "c": [0, 2], "d": ["c", "a"]}
        ).convert_dtypes(dtype_backend="pyarrow")
        expected = df.iloc[:, df["c"]]
        result = df_arrow.iloc[:, df_arrow["c"]]
        tm.assert_frame_equal(result, expected, check_dtype=False)

    @td.skip_if_no("pyarrow")
    def test_setitem_pyarrow_int_series(self):
        # GH#62462
        # [FUNCTION DOC]
        # GH 62462. Skipped without pyarrow. Assigning an arrow integer series through an index of
        # positions sets those entries.
        ser = Series([1, 2, 3], dtype="int64[pyarrow]")
        idx = Index([0, 1])
        vals = Series([7, 8], dtype="int64[pyarrow]")

        ser.iloc[idx] = vals

        expected = Series([7, 8, 3], dtype="int64[pyarrow]")
        tm.assert_series_equal(ser, expected)


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
    [
        list,
        np.array,
        pytest.param(
            lambda x: Series(x, index=["a"]),
            marks=pytest.mark.xfail(reason="Series indexer fails"),
        ),
        Index,
        array,
    ],
    # [np.array],
    ids=["list", "ndarray", "Series", "Index", "array"],
)
def test_iloc_setitem_single_column_frame_ea_dtype(dtype, box):
    # https://github.com/pandas-dev/pandas/issues/66527
    # column boolean mask that sets into the single column of a 1-column df
    # [FUNCTION DOC]
    # Issue 66527. Parametrized over dtype (nullable integer, python-backed and arrow-backed
    # string; a tuple dtype is converted to the string dtype) and a box for the one-element boolean
    # column mask (list, array, series expected to fail, index, array extension). On a one-column frame
    # assignment of a frame (all rows and a row subset) and a scalar (all rows and a subset) through
    # the boolean column selector gives the expected values.
    # KNOWN ISSUE: a stale commented-out parameter list sits above the ids.
    if isinstance(dtype, tuple):
        dtype = StringDtype(*dtype)
    df = DataFrame({"a": array([1, 2, 3], dtype=dtype)})

    # setting with a 2d dataframe
    df.iloc[:, box([True])] = df * 2

    expected = DataFrame({"a": array([1, 2, 3], dtype=dtype) * 2})
    tm.assert_frame_equal(df, expected)

    df.iloc[[0, 1], box([True])] = df.iloc[[0, 1], :] * 2

    arr = array([1, 2, 3], dtype=dtype) * 2
    arr[[0, 1]] = arr[[0, 1]] * 2
    expected = DataFrame({"a": arr})
    tm.assert_frame_equal(df, expected)

    # setting with a scalar
    df = DataFrame({"a": array([1, 2, 3], dtype=dtype)})
    scalar = df.iloc[1, 0]

    df.iloc[:, box([True])] = scalar

    expected = DataFrame({"a": array([scalar] * 3, dtype=dtype)})
    tm.assert_frame_equal(df, expected)

    df.iloc[[0, 1], box([True])] = scalar * 2

    expected = DataFrame({"a": array([scalar * 2, scalar * 2, scalar], dtype=dtype)})
    tm.assert_frame_equal(df, expected)
