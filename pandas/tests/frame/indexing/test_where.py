# [MODULE DOC]
# Tests for the conditional replacement methods of the frame class (where and its
# inverse mask), including in-place operation, alignment of the condition and other
# operand, axis handling, dtype preservation and upcasting rules, extension,
# categorical, datetime-like, interval, period and string dtypes, and invalid-input
# errors. Also defines a fixture of frames of several dtype mixes and a few private
# check helpers. Uses frame fixtures from conftest, the testing helpers module and a
# hypothesis strategy. No import-time side effects.

from datetime import datetime

from hypothesis import given
import numpy as np
import pytest

from pandas.core.dtypes.common import is_scalar

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    Series,
    StringDtype,
    Timestamp,
    date_range,
    isna,
)
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL


@pytest.fixture(params=["default", "float_string", "mixed_float", "mixed_int"])
def where_frame(request, float_string_frame, mixed_float_frame, mixed_int_frame):
    # [FUNCTION DOC]
    # HELPER: parametrized fixture (default, float_string, mixed_float, mixed_int) used
    # by several tests. Returns a random 5 by 3 float frame for the default case,
    # otherwise the corresponding fixture frame.
    if request.param == "default":
        return DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)), columns=["A", "B", "C"]
        )
    if request.param == "float_string":
        return float_string_frame
    if request.param == "mixed_float":
        return mixed_float_frame
    if request.param == "mixed_int":
        return mixed_int_frame


def _safe_add(df):
    # [FUNCTION DOC]
    # HELPER: returns a new frame in which every numeric column (integer or floating,
    # except unsigned 8-bit) has one added and all other columns are copied unchanged.
    # Used to build the other operand for where tests.
    # only add to the numeric items
    def is_ok(s):
        # [FUNCTION DOC]
        # HELPER: nested predicate for the safe-add helper; true when the series dtype
        # is an integer or floating numpy type other than unsigned 8-bit.
        return (
            issubclass(s.dtype.type, (np.integer, np.floating)) and s.dtype != "uint8"
        )

    return DataFrame(dict((c, s + 1) if is_ok(s) else (c, s) for c, s in df.items()))


class TestDataFrameIndexingWhere:
    # [CLASS DOC]
    # Main group of tests for where and mask on frames; stateless pytest class.
    def test_where_get(self, where_frame, float_string_frame):
        # [FUNCTION DOC]
        # Takes the where_frame fixture. For the string frame the comparison with zero
        # is expected to raise a type error and the test ends. Otherwise checks, via a
        # nested helper, that where with a frame condition and with the same condition
        # as raw values give equal results, that each column equals the numpy selection
        # between original and other, and that dtypes are preserved.
        def _check_get(df, cond, check_dtypes=True):
            # [FUNCTION DOC]
            # HELPER: nested checker for the get test. Parameters: frame, condition,
            # flag whether to compare dtypes (default true). Builds the other frame by
            # safe add, verifies per-column results against numpy selection, equality
            # between frame and array conditions, and dtype equality.
            other1 = _safe_add(df)
            rs = df.where(cond, other1)
            rs2 = df.where(cond.values, other1)
            for k, v in rs.items():
                exp = Series(np.where(cond[k], df[k], other1[k]), index=v.index)
                tm.assert_series_equal(v, exp, check_names=False)
            tm.assert_frame_equal(rs, rs2)

            # dtypes
            if check_dtypes:
                assert (rs.dtypes == df.dtypes).all()

        # check getting
        df = where_frame
        if df is float_string_frame:
            msg = (
                "'>' not supported between instances of 'str' and 'int'"
                "|Invalid comparison"
            )
            with pytest.raises(TypeError, match=msg):
                df > 0
            return
        cond = df > 0
        _check_get(df, cond)

    def test_where_upcasting(self):
        # [FUNCTION DOC]
        # Despite the name, this test does not call where: it sets a whole row to zero
        # through positional assignment on a frame of four numeric dtypes and asserts
        # that every column keeps its original dtype (no upcasting).
        # upcasting case (GH # 2794)
        df = DataFrame(
            {
                c: Series([1] * 3, dtype=c)
                for c in ["float32", "float64", "int32", "int64"]
            }
        )
        df.iloc[1, :] = 0
        result = df.dtypes
        expected = Series(
            [
                np.dtype("float32"),
                np.dtype("float64"),
                np.dtype("int32"),
                np.dtype("int64"),
            ],
            index=["float32", "float64", "int32", "int64"],
        )

        # when we don't preserve boolean casts
        #
        # expected = Series({ 'float32' : 1, 'float64' : 3 })

        tm.assert_series_equal(result, expected)

    def test_where_alignment(self, where_frame, float_string_frame):
        # [FUNCTION DOC]
        # Takes where_frame. Skips (after checking the expected type error) for the
        # string frame. For other frames checks where with a condition missing its first
        # row (aligned, with the frame other), with an array other, and with a scalar
        # missing value; integer frames skip the dtype check since they are upcast.
        # aligning
        def _check_align(df, cond, other, check_dtypes=True):
            # [FUNCTION DOC]
            # HELPER: nested checker for alignment test. Parameters: frame, condition,
            # other (frame, array or scalar), flag whether to check dtypes (default
            # true). Recomputes each column using numpy selection with the condition
            # reindexed to the column and missing values treated as false, then compares
            # ignoring dtype; compares dtypes to the original only when other is not an
            # array.
            rs = df.where(cond, other)
            for i, k in enumerate(rs.columns):
                result = rs[k]
                d = df[k].values
                c = cond[k].reindex(df[k].index).fillna(False).values

                if is_scalar(other):
                    o = other
                elif isinstance(other, np.ndarray):
                    o = Series(other[:, i], index=result.index).values
                else:
                    o = other[k].values

                new_values = d if c.all() else np.where(c, d, o)
                expected = Series(new_values, index=result.index, name=k)

                # since we can't always have the correct numpy dtype
                # as numpy doesn't know how to downcast, don't check
                tm.assert_series_equal(result, expected, check_dtype=False)

            # dtypes
            # can't check dtype when other is an ndarray

            if check_dtypes and not isinstance(other, np.ndarray):
                assert (rs.dtypes == df.dtypes).all()

        df = where_frame
        if df is float_string_frame:
            msg = (
                "'>' not supported between instances of 'str' and 'int'"
                "|Invalid comparison"
            )
            with pytest.raises(TypeError, match=msg):
                df > 0
            return

        # other is a frame
        cond = (df > 0)[1:]
        _check_align(df, cond, _safe_add(df))

        # check other is ndarray
        cond = df > 0
        _check_align(df, cond, (_safe_add(df).values))

        # integers are upcast, so don't check the dtypes
        cond = df > 0
        check_dtypes = all(not issubclass(s.type, np.integer) for s in df.dtypes)
        _check_align(df, cond, np.nan, check_dtypes=check_dtypes)

    # Ignore deprecation warning in Python 3.12 for inverting a bool
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_where_invalid(self):
        # [FUNCTION DOC]
        # Ignores a python 3.12 deprecation warning. Expects value errors when other is
        # an array of the wrong shape, when an array condition has the wrong shape, and
        # when mask is given a scalar condition (true or zero).
        # invalid conditions
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)), columns=["A", "B", "C"]
        )
        cond = df > 0

        err1 = (df + 1).values[0:2, :]
        msg = "other must be the same shape as self when an ndarray"
        with pytest.raises(ValueError, match=msg):
            df.where(cond, err1)

        err2 = cond.iloc[:2, :].values
        other1 = _safe_add(df)
        msg = "Array conditional must be same shape as self"
        with pytest.raises(ValueError, match=msg):
            df.where(err2, other1)

        with pytest.raises(ValueError, match=msg):
            df.mask(True)
        with pytest.raises(ValueError, match=msg):
            df.mask(0)

    def test_where_set(self, where_frame, float_string_frame, mixed_int_frame):
        # [FUNCTION DOC]
        # Takes where_frame (and the string and mixed-int fixtures for comparison).
        # Skips for string frame; converts mixed-int to float. Uses a nested helper to
        # verify in-place where returns the same object and equals the result of mask
        # with the inverted, aligned condition. Checks strict, non-strict and partially
        # aligned conditions.
        # where inplace

        def _check_set(df, cond, check_dtypes=True):
            # [FUNCTION DOC]
            # HELPER: nested checker for in-place where. Parameters: frame, condition,
            # check_dtypes flag (default true). Computes expected via mask on an
            # inverted, reindexed and null-filled condition, applies in-place where with
            # missing value, asserts identity of the returned object, equality and dtype
            # (integer columns with any false in condition become float64).
            dfi = df.copy()
            econd = cond.reindex_like(df).fillna(True).infer_objects()
            expected = dfi.mask(~econd)

            result = dfi.where(cond, np.nan, inplace=True)
            assert result is dfi
            tm.assert_frame_equal(dfi, expected)

            # dtypes (and confirm upcasts)x
            if check_dtypes:
                for k, v in df.dtypes.items():
                    if issubclass(v.type, np.integer) and not cond[k].all():
                        v = np.dtype("float64")
                    assert dfi[k].dtype == v

        df = where_frame
        if df is float_string_frame:
            msg = (
                "'>' not supported between instances of 'str' and 'int'"
                "|Invalid comparison"
            )
            with pytest.raises(TypeError, match=msg):
                df > 0
            return
        if df is mixed_int_frame:
            df = df.astype("float64")

        cond = df > 0
        _check_set(df, cond)

        cond = df >= 0
        _check_set(df, cond)

        # aligning
        cond = (df >= 0)[1:]
        _check_set(df, cond)

    def test_where_series_slicing(self):
        # [FUNCTION DOC]
        # Where with a boolean series as condition equals selecting rows and reindexing
        # to the original index.
        # GH 10218
        # test DataFrame.where with Series slicing
        df = DataFrame({"a": range(3), "b": range(4, 7)})
        result = df.where(df["a"] == 1)
        expected = df[df["a"] == 1].reindex(df.index)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("klass", [list, tuple, np.array])
    def test_where_array_like(self, klass):
        # [FUNCTION DOC]
        # Parametrized over list, tuple and array containers. A nested boolean sequence
        # of the same shape is accepted as condition; one and two column cases check the
        # missing-value result.
        # see gh-15414
        df = DataFrame({"a": [1, 2, 3]})
        cond = [[False], [True], [True]]
        expected = DataFrame({"a": [np.nan, 2, 3]})

        result = df.where(klass(cond))
        tm.assert_frame_equal(result, expected)

        df["b"] = 2
        expected["b"] = [2, np.nan, 2]
        cond = [[False, True], [True, False], [True, True]]

        result = df.where(klass(cond))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "cond",
        [
            [[1], [0], [1]],
            Series([[2], [5], [7]]),
            DataFrame({"a": [2, 5, 7]}),
            [["True"], ["False"], ["True"]],
            [[Timestamp("2017-01-01")], [pd.NaT], [Timestamp("2017-01-02")]],
        ],
    )
    def test_where_invalid_input_single(self, cond):
        # [FUNCTION DOC]
        # Parametrized with several non-boolean conditions (integers, series of lists,
        # frame of ints, strings, timestamps) for a one-column frame; each must raise a
        # type error stating a boolean array is expected.
        # see gh-15414: only boolean arrays accepted
        df = DataFrame({"a": [1, 2, 3]})
        msg = "Boolean array expected for the condition"

        with pytest.raises(TypeError, match=msg):
            df.where(cond)

    @pytest.mark.parametrize(
        "cond",
        [
            [[0, 1], [1, 0], [1, 1]],
            Series([[0, 2], [5, 0], [4, 7]]),
            [["False", "True"], ["True", "False"], ["True", "True"]],
            DataFrame({"a": [2, 5, 7], "b": [4, 8, 9]}),
            [
                [pd.NaT, Timestamp("2017-01-01")],
                [Timestamp("2017-01-02"), pd.NaT],
                [Timestamp("2017-01-03"), Timestamp("2017-01-03")],
            ],
        ],
    )
    def test_where_invalid_input_multiple(self, cond):
        # [FUNCTION DOC]
        # Same as the single-column version but with a two-column frame and two-column
        # conditions.
        # see gh-15414: only boolean arrays accepted
        df = DataFrame({"a": [1, 2, 3], "b": [2, 2, 2]})
        msg = "Boolean array expected for the condition"

        with pytest.raises(TypeError, match=msg):
            df.where(cond)

    def test_where_dataframe_col_match(self):
        # [FUNCTION DOC]
        # Frame condition with matching columns masks as expected (result float); a
        # frame condition with different column labels aligns to nothing, so everything
        # becomes missing.
        df = DataFrame([[1, 2, 3], [4, 5, 6]])
        cond = DataFrame([[True, False, True], [False, False, True]])

        result = df.where(cond)
        expected = DataFrame([[1.0, np.nan, 3], [np.nan, np.nan, 6]])
        tm.assert_frame_equal(result, expected)

        # this *does* align, though has no matching columns
        cond.columns = ["a", "b", "c"]
        result = df.where(cond)
        expected = DataFrame(np.nan, index=df.index, columns=df.columns)
        tm.assert_frame_equal(result, expected)

    def test_where_ndframe_align(self):
        # [FUNCTION DOC]
        # A too-short list or too-long array as condition raises a value error about
        # condition shape; the same data wrapped in a series aligns on the index and
        # masks the missing rows.
        msg = "Array conditional must be same shape as self"
        df = DataFrame([[1, 2, 3], [4, 5, 6]])

        cond = [True]
        with pytest.raises(ValueError, match=msg):
            df.where(cond)

        expected = DataFrame([[1, 2, 3], [np.nan, np.nan, np.nan]])

        out = df.where(Series(cond))
        tm.assert_frame_equal(out, expected)

        cond = np.array([False, True, False, True])
        with pytest.raises(ValueError, match=msg):
            df.where(cond)

        expected = DataFrame([[np.nan, np.nan, np.nan], [4, 5, 6]])

        out = df.where(Series(cond))
        tm.assert_frame_equal(out, expected)

    def test_where_bug(self):
        # [FUNCTION DOC]
        # Where with missing value as other, non in-place and in-place: original frame
        # unchanged in the first case; in the second the returned object is the frame
        # itself, with the expected values.
        # see gh-2793
        df_orig = DataFrame(
            {"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]}, dtype="float64"
        )
        expected = DataFrame(
            {"a": [np.nan, np.nan, 3.0, 4.0], "b": [4.0, 3.0, np.nan, np.nan]},
            dtype="float64",
        )

        df = df_orig.copy()
        result = df.where(df > 2, np.nan)
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(df, df_orig)

        df = df_orig.copy()
        result = df.where(df > 2, np.nan, inplace=True)
        assert result is df
        tm.assert_frame_equal(df, expected)

    def test_where_bug_mixed(self, any_signed_int_numpy_dtype):
        # [FUNCTION DOC]
        # Parametrized over signed integer numpy dtypes. Replaces with -1 where
        # condition false in a mixed int/float frame, in both modes; integer dtype is
        # preserved and in-place returns the same object.
        # see gh-2793
        df_orig = DataFrame(
            {
                "a": np.array([1, 2, 3, 4], dtype=any_signed_int_numpy_dtype),
                "b": np.array([4.0, 3.0, 2.0, 1.0], dtype="float64"),
            }
        )

        expected = DataFrame(
            {"a": [-1, -1, 3, 4], "b": [4.0, 3.0, -1, -1]},
        ).astype({"a": any_signed_int_numpy_dtype, "b": "float64"})

        df = df_orig.copy()
        result = df.where(df > 2, -1)
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(df, df_orig)

        df = df_orig.copy()
        result = df.where(df > 2, -1, inplace=True)
        assert result is df
        tm.assert_frame_equal(df, expected)

    def test_where_bug_transposition(self):
        # [FUNCTION DOC]
        # Regression for transposition bug: where with a frame other and a combined
        # condition equals manual assignment of the frame through an inverted boolean
        # frame, with specific float casts, for two input pairs.
        # see gh-7506
        a = DataFrame({0: [1, 2], 1: [3, 4], 2: [5, 6]})
        b = DataFrame({0: [np.nan, 8], 1: [9, np.nan], 2: [np.nan, np.nan]})
        do_not_replace = b.isna() | (a > b)

        expected = a.copy()
        expected[~do_not_replace] = b
        expected[[0, 1]] = expected[[0, 1]].astype("float64")

        result = a.where(do_not_replace, b)
        tm.assert_frame_equal(result, expected)

        a = DataFrame({0: [4, 6], 1: [1, 0]})
        b = DataFrame({0: [np.nan, 3], 1: [3, np.nan]})
        do_not_replace = b.isna() | (a > b)

        expected = a.copy()
        expected[~do_not_replace] = b
        expected[1] = expected[1].astype("float64")

        result = a.where(do_not_replace, b)
        tm.assert_frame_equal(result, expected)

    def test_where_datetime(self):
        # [FUNCTION DOC]
        # Frame with two datetime and one float column: comparing with a datetime raises
        # a type error for the float column; selecting with the condition on the
        # datetime columns only gives missing values for the first two rows of the first
        # column and for the entire float column.
        # GH 3311
        df = DataFrame(
            {
                "A": date_range("20130102", periods=5),
                "B": date_range("20130104", periods=5),
                "C": np.random.default_rng(2).standard_normal(5),
            }
        )

        stamp = datetime(2013, 1, 3)
        msg = "'>' not supported between instances of 'float' and 'datetime.datetime'"
        with pytest.raises(TypeError, match=msg):
            df > stamp

        result = df[df.iloc[:, :-1] > stamp]

        expected = df.copy()
        expected.loc[[0, 1], "A"] = np.nan

        expected.loc[:, "C"] = np.nan
        tm.assert_frame_equal(result, expected)

    def test_where_none(self):
        # [FUNCTION DOC]
        # Setting values above a threshold to None via item setting on float columns
        # yields missing values; in-place where with None on a mixed frame yields object
        # columns containing None; the mask-based item setter gives the same.
        # GH 4667
        # setting with None changes dtype
        df = DataFrame({"series": Series(range(10))}).astype(float)
        df[df > 7] = None
        expected = DataFrame(
            {"series": Series([0, 1, 2, 3, 4, 5, 6, 7, np.nan, np.nan])}
        )
        tm.assert_frame_equal(df, expected)

        # GH 7656
        df = DataFrame(
            [
                {"A": 1, "B": np.nan, "C": "Test"},
                {"A": np.nan, "B": "Test", "C": np.nan},
            ]
        )

        orig = df.copy()

        mask = ~isna(df)
        df.where(mask, None, inplace=True)
        expected = DataFrame(
            {
                "A": [1.0, np.nan],
                "B": [None, "Test"],
                "C": ["Test", None],
            }
        )
        tm.assert_frame_equal(df, expected)

        df = orig.copy()
        df[~mask] = None
        tm.assert_frame_equal(df, expected)

    def test_where_empty_df_and_empty_cond_having_non_bool_dtypes(self):
        # [FUNCTION DOC]
        # An empty frame with object columns used as its own condition is accepted and
        # returns an equal frame.
        # see gh-21947
        df = DataFrame(columns=["a"])
        cond = df
        assert (cond.dtypes == object).all()

        result = df.where(cond)
        tm.assert_frame_equal(result, df)

    def test_where_align(self):
        # [FUNCTION DOC]
        # Several scenarios with a random frame holding missing values: fill-by-column
        # means via where with axis columns (also in place), a series other aligned on
        # the index with the axis given as index or rows, and a frame other.
        def create():
            # [FUNCTION DOC]
            # HELPER: builds a 10 by 3 random frame with missing values in specific row
            # ranges of each column.
            df = DataFrame(np.random.default_rng(2).standard_normal((10, 3)))
            df.iloc[3:5, 0] = np.nan
            df.iloc[4:6, 1] = np.nan
            df.iloc[5:8, 2] = np.nan
            return df

        # series
        df = create()
        expected = df.fillna(df.mean())
        result = df.where(pd.notna(df), df.mean(), axis="columns")
        tm.assert_frame_equal(result, expected)

        result = df.where(pd.notna(df), df.mean(), inplace=True, axis="columns")
        assert result is df
        tm.assert_frame_equal(df, expected)

        df = create().fillna(0)
        expected = df.apply(lambda x, y: x.where(x > 0, y), y=df[0])
        result = df.where(df > 0, df[0], axis="index")
        tm.assert_frame_equal(result, expected)
        result = df.where(df > 0, df[0], axis="rows")
        tm.assert_frame_equal(result, expected)

        # frame
        df = create()
        expected = df.fillna(1)
        result = df.where(
            pd.notna(df), DataFrame(1, index=df.index, columns=df.columns)
        )
        tm.assert_frame_equal(result, expected)

    def test_where_complex(self):
        # [FUNCTION DOC]
        # Setting entries to missing where the absolute value of a complex frame is at
        # least five keeps complex dtype with a missing entry.
        # GH 6345
        expected = DataFrame([[1 + 1j, 2], [np.nan, 4 + 1j]], columns=["a", "b"])
        df = DataFrame([[1 + 1j, 2], [5 + 1j, 4 + 1j]], columns=["a", "b"])
        df[df.abs() >= 5] = np.nan
        tm.assert_frame_equal(df, expected)

    def test_where_axis(self):
        # [FUNCTION DOC]
        # Frame with all-false mask and a series other, axis index or columns, not in
        # place and in place (identity returned); results are broadcast along the chosen
        # axis as float.
        # GH 9736
        df_orig = DataFrame(np.random.default_rng(2).standard_normal((2, 2)))
        mask = DataFrame([[False, False], [False, False]])
        ser = Series([0, 1])

        df = df_orig.copy()
        expected = DataFrame([[0, 0], [1, 1]], dtype="float64")
        result = df.where(mask, ser, axis="index")
        tm.assert_frame_equal(result, expected)

        df = df_orig.copy()
        result = df.where(mask, ser, axis="index", inplace=True)
        assert result is df
        tm.assert_frame_equal(result, expected)

        df = df_orig.copy()
        expected = DataFrame([[0, 1], [0, 1]], dtype="float64")
        result = df.where(mask, ser, axis="columns")
        tm.assert_frame_equal(result, expected)

        df = df_orig.copy()
        result = df.copy()
        result = df.where(mask, ser, axis="columns", inplace=True)
        assert result is df
        tm.assert_frame_equal(df, expected)

    def test_where_axis_with_upcast(self):
        # [FUNCTION DOC]
        # Integer frame, all-false mask, series other with a missing value: non in-place
        # where upcasts to float; in-place raises a type error about invalid value.
        # Checked for both axes.
        # Upcast needed
        df = DataFrame([[1, 2], [3, 4]], dtype="int64")
        mask = DataFrame([[False, False], [False, False]])
        ser = Series([0, np.nan])

        expected = DataFrame([[0, 0], [np.nan, np.nan]], dtype="float64")
        result = df.where(mask, ser, axis="index")
        tm.assert_frame_equal(result, expected)

        result = df.copy()
        with pytest.raises(TypeError, match="Invalid value"):
            result.where(mask, ser, axis="index", inplace=True)

        expected = DataFrame([[0, np.nan], [0, np.nan]])
        result = df.where(mask, ser, axis="columns")
        tm.assert_frame_equal(result, expected)

        with pytest.raises(TypeError, match="Invalid value"):
            df.where(mask, ser, axis="columns", inplace=True)

    def test_where_axis_multiple_dtypes(self):
        # [FUNCTION DOC]
        # Frame of two float and two int columns (multiple blocks). Tests scalar-like
        # series other on both axes (in and out of place), then frame other with dropped
        # row or dropped column, where in-place with the dropped row raises a type error
        # about invalid value while the dropped-column variant works.
        # Multiple dtypes (=> multiple Blocks)
        df_orig = pd.concat(
            [
                DataFrame(np.random.default_rng(2).standard_normal((10, 2))),
                DataFrame(
                    np.random.default_rng(2).integers(0, 10, size=(10, 2)),
                    dtype="int64",
                ),
            ],
            ignore_index=True,
            axis=1,
        )
        mask = DataFrame(False, columns=df_orig.columns, index=df_orig.index)
        s1 = Series(1, index=df_orig.columns)
        s2 = Series(2, index=df_orig.index)

        df = df_orig.copy()
        result = df.where(mask, s1, axis="columns")
        expected = DataFrame(1.0, columns=df_orig.columns, index=df_orig.index)
        expected[2] = expected[2].astype("int64")
        expected[3] = expected[3].astype("int64")
        tm.assert_frame_equal(result, expected)

        df = df_orig.copy()
        result = df.where(mask, s1, axis="columns", inplace=True)
        assert result is df
        tm.assert_frame_equal(df, expected)

        df = df_orig.copy()
        result = df.where(mask, s2, axis="index")
        expected = DataFrame(2.0, columns=df_orig.columns, index=df_orig.index)
        expected[2] = expected[2].astype("int64")
        expected[3] = expected[3].astype("int64")
        tm.assert_frame_equal(result, expected)

        df = df_orig.copy()
        result = df.where(mask, s2, axis="index", inplace=True)
        assert result is df
        tm.assert_frame_equal(df, expected)

        # DataFrame vs DataFrame
        d1 = df_orig.copy().drop(1, axis=0)
        # Explicit cast to avoid implicit cast when setting value to np.nan
        expected = df_orig.copy().astype("float")
        expected.loc[1, :] = np.nan

        df = df_orig.copy()
        result = df.where(mask, d1)
        tm.assert_frame_equal(result, expected)
        result = df.where(mask, d1, axis="index")
        tm.assert_frame_equal(result, expected)
        df = df_orig.copy()
        with pytest.raises(TypeError, match="Invalid value"):
            df.where(mask, d1, inplace=True)
        with pytest.raises(TypeError, match="Invalid value"):
            df.where(mask, d1, inplace=True, axis="index")

        d2 = df_orig.copy().drop(1, axis=1)
        expected = df_orig.copy()
        expected.loc[:, 1] = np.nan

        df = df_orig.copy()
        result = df.where(mask, d2)
        tm.assert_frame_equal(result, expected)
        result = df.where(mask, d2, axis="columns")
        tm.assert_frame_equal(result, expected)
        df = df_orig.copy()
        result = df.where(mask, d2, inplace=True)
        assert result is df
        tm.assert_frame_equal(df, expected)
        df = df_orig.copy()
        result = df.where(mask, d2, inplace=True, axis="columns")
        assert result is df
        tm.assert_frame_equal(df, expected)

    def test_where_callable(self):
        # [FUNCTION DOC]
        # Callables as condition and as other are called with the frame; results equal
        # the non-callable versions, including callable returning an array or a scalar,
        # and a chained expression.
        # GH 12533
        df = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = df.where(lambda x: x > 4, lambda x: x + 1)
        exp = DataFrame([[2, 3, 4], [5, 5, 6], [7, 8, 9]])
        tm.assert_frame_equal(result, exp)
        tm.assert_frame_equal(result, df.where(df > 4, df + 1))

        # return ndarray and scalar
        result = df.where(lambda x: (x % 2 == 0).values, lambda x: 99)
        exp = DataFrame([[99, 2, 99], [4, 99, 6], [99, 8, 99]])
        tm.assert_frame_equal(result, exp)
        tm.assert_frame_equal(result, df.where(df % 2 == 0, 99))

        # chain
        result = (df + 2).where(lambda x: x > 8, lambda x: x + 10)
        exp = DataFrame([[13, 14, 15], [16, 17, 18], [9, 10, 11]])
        tm.assert_frame_equal(result, exp)
        tm.assert_frame_equal(result, (df + 2).where((df + 2) > 8, (df + 2) + 10))

    def test_where_tz_values(self, tz_naive_fixture, frame_or_series):
        # [FUNCTION DOC]
        # Takes tz fixture and frame-or-series fixture. Where with tz-aware datetime
        # frames/series and a boolean mask picks values from other where the mask is
        # false and retains the dtype.
        obj1 = DataFrame(
            DatetimeIndex(["20150101", "20150102", "20150103"], tz=tz_naive_fixture),
            columns=["date"],
        )
        obj2 = DataFrame(
            DatetimeIndex(["20150103", "20150104", "20150105"], tz=tz_naive_fixture),
            columns=["date"],
        )
        mask = DataFrame([True, True, False], columns=["date"])
        exp = DataFrame(
            DatetimeIndex(["20150101", "20150102", "20150105"], tz=tz_naive_fixture),
            columns=["date"],
        )
        if frame_or_series is Series:
            obj1 = obj1["date"]
            obj2 = obj2["date"]
            mask = mask["date"]
            exp = exp["date"]

        result = obj1.where(mask, obj2)
        tm.assert_equal(exp, result)

    def test_df_where_change_dtype(self):
        # [FUNCTION DOC]
        # Where with a boolean array on an integer frame upcasts to float with missing
        # values.
        # GH#16979
        df = DataFrame(np.arange(2 * 3).reshape(2, 3), columns=list("ABC"))
        mask = np.array([[True, False, False], [False, False, True]])

        result = df.where(mask)
        expected = DataFrame(
            [[0, np.nan, np.nan], [np.nan, np.nan, 5]], columns=list("ABC")
        )

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("kwargs", [{}, {"other": None}])
    def test_df_where_with_category(self, kwargs):
        # [FUNCTION DOC]
        # Parametrized over empty kwargs and other equal to None. Where on categorical
        # columns keeps categories (restricted to what remains) for frame and for a
        # series column.
        # GH#16979
        data = np.arange(2 * 3, dtype=np.int64).reshape(2, 3)
        df = DataFrame(data, columns=list("ABC"))
        mask = np.array([[True, False, False], [False, False, True]])

        # change type to category
        df.A = df.A.astype("category")
        df.B = df.B.astype("category")
        df.C = df.C.astype("category")

        result = df.where(mask, **kwargs)
        A = pd.Categorical([0, np.nan], categories=[0, 3])
        B = pd.Categorical([np.nan, np.nan], categories=[1, 4])
        C = pd.Categorical([np.nan, 5], categories=[2, 5])
        expected = DataFrame({"A": A, "B": B, "C": C})

        tm.assert_frame_equal(result, expected)

        # Check Series.where while we're here
        result = df.A.where(mask[:, 0], **kwargs)
        expected = Series(A, name="A")

        tm.assert_series_equal(result, expected)

    def test_where_categorical_filtering(self):
        # [FUNCTION DOC]
        # Where on a frame with a categorical column and a series condition: the first
        # row becomes all missing; numeric column is float.
        # GH#22609 Verify filtering operations on DataFrames with categorical Series
        df = DataFrame(data=[[0, 0], [1, 1]], columns=["a", "b"])
        df["b"] = df["b"].astype("category")

        result = df.where(df["a"] > 0)
        # Explicitly cast to 'float' to avoid implicit cast when setting np.nan
        expected = df.copy().astype({"a": "float"})
        expected.loc[0, :] = np.nan

        tm.assert_equal(result, expected)

    def test_where_ea_other(self):
        # [FUNCTION DOC]
        # Where and mask (including in-place) with a nullable integer series as other
        # along either axis, giving nullable results where needed.
        # GH#38729/GH#38742, GH#62038
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        arr = pd.array([7, pd.NA, 9])
        ser = Series(arr)
        mask = np.ones(df.shape, dtype=bool)
        mask[1, :] = False

        result1 = df.where(mask, ser, axis=0)
        expected1 = DataFrame({"A": [1, pd.NA, 3], "B": [4, pd.NA, 6]}, dtype="Int64")
        tm.assert_frame_equal(result1, expected1)

        ser2 = Series(arr[:2], index=["A", "B"])
        expected2 = DataFrame({"A": [1, 7, 3], "B": [4, pd.NA, 6]})
        expected2["B"] = expected2["B"].astype("Int64")
        result2 = df.where(mask, ser2, axis=1)
        tm.assert_frame_equal(result2, expected2)

        result3 = df.copy()
        result3.mask(mask, ser, axis=0, inplace=True)
        tm.assert_frame_equal(result3, expected1)

        result4 = df.copy()
        result4.mask(mask, ser2, axis=1, inplace=True)
        tm.assert_frame_equal(result4, expected2)

    def test_where_interval_noop(self):
        # [FUNCTION DOC]
        # A where with a condition that is all true on interval data returns equal data
        # (no error), for frame and series.
        # GH#44181
        df = DataFrame([pd.Interval(0, 0)])
        res = df.where(df.notna())
        tm.assert_frame_equal(res, df)

        ser = df[0]
        res = ser.where(ser.notna())
        tm.assert_series_equal(res, ser)

    def test_where_interval_fullop_downcast(self, frame_or_series):
        # [FUNCTION DOC]
        # Takes frame-or-series fixture. Replacing every interval value with floats
        # (object dtype) via where returns the other; in-place mask raises a type error
        # about invalid value.
        # GH#45768
        obj = frame_or_series([pd.Interval(0, 0)] * 2)
        other = frame_or_series([1.0, 2.0], dtype=object)
        res = obj.where(~obj.notna(), other)
        tm.assert_equal(res, other)

        with pytest.raises(TypeError, match="Invalid value"):
            obj.mask(obj.notna(), other, inplace=True)

    @pytest.mark.parametrize(
        "dtype",
        [
            "timedelta64[ns]",
            "datetime64[ns]",
            "datetime64[ns, Asia/Tokyo]",
        ],
    )
    def test_where_datetimelike_noop(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over timedelta, datetime and tz-aware datetime dtypes. A
        # where/mask with an all-false (no-op) condition and an incompatible string
        # other returns equal objects without upcasting; in-place mask with a real
        # replacement of an incompatible int raises a type error.
        # GH#45135, analogue to GH#44181 for Period don't raise on no-op
        # For td64/dt64/dt64tz we already don't raise, but also are
        #  checking that we don't unnecessarily upcast to object.
        ser = Series(np.arange(3) * 10**9, dtype=np.int64).astype(dtype)
        df = ser.to_frame()
        mask = np.array([False, False, False])

        res = ser.where(~mask, "foo")
        tm.assert_series_equal(res, ser)

        mask2 = mask.reshape(-1, 1)
        res2 = df.where(~mask2, "foo")
        tm.assert_frame_equal(res2, df)

        res3 = ser.mask(mask, "foo")
        tm.assert_series_equal(res3, ser)

        res4 = df.mask(mask2, "foo")
        tm.assert_frame_equal(res4, df)

        # unlike where, Block.putmask does not downcast
        with pytest.raises(TypeError, match="Invalid value"):
            df.mask(~mask2, 4, inplace=True)


def test_where_int_downcasting_deprecated():
    # [FUNCTION DOC]
    # Where on a 16-bit integer frame with a value too large for 16 bits upcasts only
    # the replaced column to 32-bit. (Name mentions deprecated downcasting, but no
    # warning is asserted.)
    # GH#44597
    arr = np.arange(6).astype(np.int16).reshape(3, 2)
    df = DataFrame(arr)

    mask = np.zeros(arr.shape, dtype=bool)
    mask[:, 0] = True

    res = df.where(mask, 2**17)

    expected = DataFrame({0: arr[:, 0], 1: np.array([2**17] * 3, dtype=np.int32)})
    tm.assert_frame_equal(res, expected)


def test_where_copies_with_noop(frame_or_series):
    # [FUNCTION DOC]
    # Takes frame-or-series fixture. A where that changes nothing must still return a
    # copy: in-place multiplication of the result must not change the original, for both
    # scalar-free and array other.
    # GH-39595
    result = frame_or_series([1, 2, 3, 4])
    expected = result.copy()
    col = result[0] if frame_or_series is DataFrame else result

    where_res = result.where(col < 5)
    where_res *= 2

    tm.assert_equal(result, expected)

    where_res = result.where(col > 5, [1, 2, 3, 4])
    where_res *= 2

    tm.assert_equal(result, expected)


def test_where_string_dtype(frame_or_series):
    # [FUNCTION DOC]
    # Takes frame-or-series fixture. Where and mask with string-dtype data and a string
    # other aligned on the index give missing values in unmatched positions.
    # KNOWN ISSUE: the last assertion, after the in-place mask call, checks the
    # previously computed result object against expected instead of the mutated input
    # object, so the in-place result is never verified.
    # GH40824
    obj = frame_or_series(
        ["a", "b", "c", "d"], index=["id1", "id2", "id3", "id4"], dtype=StringDtype()
    )
    filtered_obj = frame_or_series(
        ["b", "c"], index=["id2", "id3"], dtype=StringDtype()
    )
    filter_ser = Series([False, True, True, False])

    result = obj.where(filter_ser, filtered_obj)
    expected = frame_or_series(
        [pd.NA, "b", "c", pd.NA],
        index=["id1", "id2", "id3", "id4"],
        dtype=StringDtype(),
    )
    tm.assert_equal(result, expected)

    result = obj.mask(~filter_ser, filtered_obj)
    tm.assert_equal(result, expected)

    obj.mask(~filter_ser, filtered_obj, inplace=True)
    tm.assert_equal(result, expected)


def test_where_bool_comparison():
    # [FUNCTION DOC]
    # Where with a mask equal to the false comparison on a boolean frame yields object
    # column of missing values for the all-true column and keeps bool for others.
    # GH 10336
    df_mask = DataFrame(
        {"AAA": [True] * 4, "BBB": [False] * 4, "CCC": [True, False, True, False]}
    )
    result = df_mask.where(df_mask == False)  # noqa: E712
    expected = DataFrame(
        {
            "AAA": np.array([np.nan] * 4, dtype=object),
            "BBB": [False] * 4,
            "CCC": [np.nan, False, np.nan, False],
        }
    )
    tm.assert_frame_equal(result, expected)


def test_where_none_nan_coerce():
    # [FUNCTION DOC]
    # Where with None as other on a frame with datetime and numeric columns having
    # missing values leaves it unchanged (None coerced to the appropriate missing
    # marker).
    # GH 15613
    expected = DataFrame(
        {
            "A": [Timestamp("20130101"), pd.NaT, Timestamp("20130103")],
            "B": [1, 2, np.nan],
        }
    )
    result = expected.where(expected.notnull(), None)
    tm.assert_frame_equal(result, expected)


def test_where_duplicate_axes_mixed_dtypes():
    # [FUNCTION DOC]
    # With duplicated column labels, where with a frame mask gives equal values whether
    # the frame is object, float, transposed, or original (previously failed on
    # duplicate axis reindex).
    # GH 25399, verify manually masking is not affected anymore by dtype of column for
    # duplicate axes.
    result = DataFrame(data=[[0, np.nan]], columns=Index(["A", "A"]))
    index, columns = result.axes
    mask = DataFrame(data=[[True, True]], columns=columns, index=index)
    a = result.astype(object).where(mask)
    b = result.astype("f8").where(mask)
    c = result.T.where(mask.T).T
    d = result.where(mask)  # used to fail with "cannot reindex from a duplicate axis"
    tm.assert_frame_equal(a.astype("f8"), b.astype("f8"))
    tm.assert_frame_equal(b.astype("f8"), c.astype("f8"))
    tm.assert_frame_equal(c.astype("f8"), d.astype("f8"))


def test_where_columns_casting():
    # [FUNCTION DOC]
    # Where with a not-null mask and None as other leaves dtypes and values unchanged.
    # GH 42295

    df = DataFrame({"a": [1.0, 2.0], "b": [3, np.nan]})
    expected = df.copy()
    result = df.where(pd.notnull(df), None)
    # make sure dtypes don't change
    tm.assert_frame_equal(expected, result)


@pytest.mark.parametrize("as_cat", [True, False])
def test_where_period_invalid_na(frame_or_series, as_cat, request):
    # [FUNCTION DOC]
    # Parametrized over categorical vs plain period data, uses frame-or-series and
    # request fixtures (request unused). A timedelta missing value is not cast to
    # period: for categorical data a type error about a new category is expected from
    # where, mask and in-place mask; for plain period data result equals the object-cast
    # computation and in-place mask raises a type error.
    # GH#44697
    idx = pd.period_range("2016-01-01", periods=3, freq="D")
    if as_cat:
        idx = idx.astype("category")
    obj = frame_or_series(idx)

    # NA value that we should *not* cast to Period dtype
    tdnat = pd.NaT.to_numpy("m8[ns]")

    mask = np.array([True, True, False], ndmin=obj.ndim).T

    if as_cat:
        msg = (
            r"Cannot setitem on a Categorical with a new category \(NaT\), "
            "set the categories first"
        )
    else:
        msg = "value should be a 'Period'"

    if as_cat:
        with pytest.raises(TypeError, match=msg):
            obj.where(mask, tdnat)

        with pytest.raises(TypeError, match=msg):
            obj.mask(mask, tdnat)

        with pytest.raises(TypeError, match=msg):
            obj.mask(mask, tdnat, inplace=True)

    else:
        # With PeriodDtype, ser[i] = tdnat coerces instead of raising,
        #  so for consistency, ser[mask] = tdnat must as well
        expected = obj.astype(object).where(mask, tdnat)
        result = obj.where(mask, tdnat)
        tm.assert_equal(result, expected)

        expected = obj.astype(object).mask(mask, tdnat)
        result = obj.mask(mask, tdnat)
        tm.assert_equal(result, expected)

        with pytest.raises(TypeError, match="Invalid value"):
            obj.mask(mask, tdnat, inplace=True)


def test_where_nullable_invalid_na(frame_or_series, any_numeric_ea_dtype):
    # [FUNCTION DOC]
    # Takes frame-or-series and numeric extension dtype fixtures. For each
    # numpy-NaT-like object and NaT as other, where and mask raise a type error about an
    # invalid value for the nullable dtype.
    # GH#44697
    arr = pd.array([1, 2, 3], dtype=any_numeric_ea_dtype)
    obj = frame_or_series(arr)

    mask = np.array([True, True, False], ndmin=obj.ndim).T

    msg = r"Invalid value '.*' for dtype '(U?Int|Float)\d{1,2}'"

    for null in [*tm.NP_NAT_OBJECTS, pd.NaT]:
        # NaT is an NA value that we should *not* cast to pd.NA dtype
        with pytest.raises(TypeError, match=msg):
            obj.where(mask, null)

        with pytest.raises(TypeError, match=msg):
            obj.mask(mask, null)


@pytest.mark.slow
@given(data=OPTIONAL_ONE_OF_ALL)
def test_where_inplace_casting(data):
    # [FUNCTION DOC]
    # Marked slow; property-based test over optional values of all kinds. In-place where
    # with None must give the same result as the non in-place call.
    # GH 22051
    df = DataFrame({"a": data})
    df_copy = df.where(pd.notnull(df), None).copy()
    df.where(pd.notnull(df), None, inplace=True)
    tm.assert_equal(df, df_copy)


def test_where_downcast_to_td64():
    # [FUNCTION DOC]
    # A series of integers with an all-false mask and a timedelta other becomes an
    # object series of the timedeltas (no downcast to timedelta dtype).
    ser = Series([1, 2, 3])

    mask = np.array([False, False, False])

    td = pd.Timedelta(days=1)
    expected = Series([td, td, td], dtype="m8[ns]")

    res2 = ser.where(mask, td)
    expected2 = expected.astype(object)
    tm.assert_series_equal(res2, expected2)


def _check_where_equivalences(df, mask, other, expected):
    # [FUNCTION DOC]
    # HELPER: checks that where with the mask, mask with the inverted mask, and in-place
    # mask all agree with the expected frame. Parameters: frame, boolean mask array,
    # other, expected frame. In the in-place case, if the mask is not all true, the
    # expected column A is cast to object because in-place put-mask does not downcast.
    # similar to tests.series.indexing.test_setitem.SetitemCastingEquivalences
    #  but with DataFrame in mind and less fleshed-out
    res = df.where(mask, other)
    tm.assert_frame_equal(res, expected)

    res = df.mask(~mask, other)
    tm.assert_frame_equal(res, expected)

    # Note: frame.mask(~mask, other, inplace=True) takes some more work bc
    #  Block.putmask does *not* downcast.  The change to 'expected' here
    #  is specific to the cases in test_where_dt64_2d.
    df = df.copy()
    df.mask(~mask, other, inplace=True)
    if not mask.all():
        # with mask.all(), Block.putmask is a no-op, so does not downcast
        expected = expected.copy()
        expected["A"] = expected["A"].astype(object)
    tm.assert_frame_equal(df, expected)


def test_where_dt64_2d():
    # [FUNCTION DOC]
    # Frame from a 2-D datetime array and other as differences; partial replacement in
    # column A raises a type error from the equivalence checker (in-place mask cannot
    # hold the values), and a fully true mask leaves the frame unchanged.
    dti = date_range("2016-01-01", periods=6)
    dta = dti._data.reshape(3, 2)
    other = dta - dta[0, 0]

    df = DataFrame(dta, columns=["A", "B"])

    mask = np.asarray(df.isna()).copy()
    mask[:, 1] = True

    # setting part of one column, none of the other
    mask[1, 0] = True
    expected = DataFrame(
        {
            "A": np.array([other[0, 0], dta[1, 0], other[2, 0]], dtype=object),
            "B": dta[:, 1],
        }
    )
    with pytest.raises(TypeError, match="Invalid value"):
        _check_where_equivalences(df, mask, other, expected)

    # setting nothing in either column
    mask[:] = True
    expected = df
    _check_where_equivalences(df, mask, other, expected)


def test_where_producing_ea_cond_for_np_dtype():
    # [FUNCTION DOC]
    # A callable condition that returns a nullable boolean frame, for a frame with mixed
    # nullable and numpy columns, yields missing values where false or NA and keeps
    # dtypes.
    # GH#44014
    df = DataFrame({"a": Series([1, pd.NA, 2], dtype="Int64"), "b": [1, 2, 3]})
    result = df.where(lambda x: x.apply(lambda y: y > 1, axis=1))
    expected = DataFrame(
        {"a": Series([pd.NA, pd.NA, 2], dtype="Int64"), "b": [np.nan, 2, 3]}
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "replacement", [0.001, True, "snake", None, datetime(2022, 5, 4)]
)
def test_where_int_overflow(replacement):
    # [FUNCTION DOC]
    # Parametrized over replacement values of several types. Replaces missing entries of
    # a mixed frame with the value; result keeps large float and strings.
    # GH 31687
    df = DataFrame([[1.0, 2e25, "nine"], [np.nan, 0.1, None]])
    result = df.where(pd.notnull(df), replacement)
    expected = DataFrame([[1.0, 2e25, "nine"], [replacement, 0.1, replacement]])

    tm.assert_frame_equal(result, expected)


def test_where_inplace_no_other():
    # [FUNCTION DOC]
    # In-place where without other returns the same frame and fills the false positions
    # with missing values.
    # GH#51685
    df = DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    cond = DataFrame({"a": [True, False], "b": [False, True]})
    result = df.where(cond, inplace=True)
    assert result is df
    expected = DataFrame({"a": [1, np.nan], "b": [np.nan, "y"]})
    tm.assert_frame_equal(df, expected)


def test_where_other_nullable_dtype():
    # [FUNCTION DOC]
    # Where with a nullable-integer series other and axis rows on an int64 frame returns
    # a nullable Int64 frame, matching series behavior.
    # GH#49052 DataFrame.where should return nullable dtype when
    # other is a Series with nullable dtype, matching Series.where behavior
    df = DataFrame([1, 2, 3], dtype="int64")
    other = Series([pd.NA, pd.NA, pd.NA], dtype="Int64")
    result = df.where(df > 1, other, axis=0)
    expected = DataFrame({0: Series([pd.NA, 2, 3], dtype="Int64")})
    tm.assert_frame_equal(result, expected)


def test_where_inplace_string_array_consistency():
    # [FUNCTION DOC]
    # In-place where and non in-place where with missing value on string dtype produce
    # equal results.
    # GH#46512
    df = DataFrame({"A": ["1", "", "3"]}, dtype="string")
    df_inplace = df.copy()

    result = df.where(df != "", np.nan)
    df_inplace.where(df_inplace != "", np.nan, inplace=True)

    tm.assert_frame_equal(result, df_inplace)
