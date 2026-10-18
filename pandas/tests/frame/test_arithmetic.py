# [MODULE DOC]
# Arithmetic and comparison tests for DataFrame: operators and flexible methods (add, sub, eq, gt, ...) against
# frames, series, scalars, arrays and other array-likes; alignment and broadcasting rules (including multi-level
# indexes); fill values; dtype preservation for mixed frames; in-place operators; and misc regression cases. An
# autouse fixture runs every test twice, with the numexpr evaluation threshold at zero and at 100 elements, and
# the file is organized into comparison, flex arithmetic, plain arithmetic and an unsorted legacy section followed
# by module-level tests. Uses shared frame fixtures from the test configuration and two mixed-dtype checking helpers from the sibling common module.

from collections import deque
from datetime import (
    datetime,
    timezone,
)
from enum import Enum
import functools
import operator
import re

import numpy as np
import pytest

from pandas.compat._optional import import_optional_dependency

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
)
import pandas._testing as tm
from pandas.core.computation import expressions as expr
from pandas.tests.frame.common import (
    _check_mixed_float,
    _check_mixed_int,
)
from pandas.util.version import Version


@pytest.fixture
def simple_frame():
    # [FUNCTION DOC]
    # Fixture: returns a 3x3 float frame with columns one, two, three, index a, b, c and values 1.0 to 9.0 in row-major order.
    """
    Fixture for simple 3x3 DataFrame

    Columns are ['one', 'two', 'three'], index is ['a', 'b', 'c'].

       one  two  three
    a  1.0  2.0    3.0
    b  4.0  5.0    6.0
    c  7.0  8.0    9.0
    """
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    return DataFrame(arr, columns=["one", "two", "three"], index=["a", "b", "c"])


@pytest.fixture(autouse=True, params=[0, 100], ids=["numexpr", "python"])
def switch_numexpr_min_elements(request, monkeypatch):
    # [FUNCTION DOC]
    # HELPER: autouse fixture parametrized over 0 (labelled numexpr) and 100 (labelled python). Within a monkeypatch context sets the expression module's minimum element threshold for numexpr use to the parameter, yields the parameter value, and restores the old value afterwards. Makes each test run both with and without the accelerated evaluation path for small frames.
    with monkeypatch.context() as m:
        m.setattr(expr, "_MIN_ELEMENTS", request.param)
        yield request.param


class DummyElement:
    # [CLASS DOC]
    # HELPER: minimal array-like scalar stand-in used by the binary-operation test. Stores a value and a numpy dtype and exposes just enough of the array protocol (array conversion, cast, view, truthiness) for pandas to treat it like a 0-dimensional array.
    def __init__(self, value, dtype) -> None:
        # [FUNCTION DOC]
        # HELPER: stores the given value and the dtype converted to a numpy dtype.
        self.value = value
        self.dtype = np.dtype(dtype)

    def __array__(self, dtype=None, copy=None):
        # [FUNCTION DOC]
        # HELPER: array conversion protocol; takes optional dtype and copy arguments (both ignored) and returns a numpy array of the stored value with the stored dtype.
        return np.array(self.value, dtype=self.dtype)

    def __str__(self) -> str:
        # [FUNCTION DOC]
        # HELPER: returns a readable text with the class name, value and dtype.
        return f"DummyElement({self.value}, {self.dtype})"

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # HELPER: same text as the string form.
        return str(self)

    def astype(self, dtype, copy=False):
        # [FUNCTION DOC]
        # HELPER: mimics the array cast method. Takes a dtype and an unused copy flag; replaces the stored dtype with the given one (mutating, not converting the value) and returns the same object.
        # KNOWN ISSUE: unlike the constructor, the given dtype is stored as passed, without converting to a numpy dtype, and the value is not converted at all.
        self.dtype = dtype
        return self

    def view(self, dtype):
        # [FUNCTION DOC]
        # HELPER: mimics the array view method. Takes a dtype and returns a new object of the same class with the stored value viewed as that dtype.
        return type(self)(self.value.view(dtype), dtype)

    def any(self, axis=None):
        # [FUNCTION DOC]
        # HELPER: mimics the array reduction. Takes an optional axis (ignored) and returns the truth value of the stored value as a boolean.
        return bool(self.value)


# -------------------------------------------------------------------
# Comparisons


class TestFrameComparisons:
    # [CLASS DOC]
    # Test container for the comparison operators (not the flexible comparison methods).
    # Specifically _not_ flex-comparisons

    def test_comparison_with_categorical_dtype(self):
        # [FUNCTION DOC]
        # Regression test (GH 12564). Comparing a string column to a value gives the same result before and after casting the column to categorical.
        # GH#12564

        df = DataFrame({"A": ["foo", "bar", "baz"]})
        exp = DataFrame({"A": [True, False, False]})

        res = df == "foo"
        tm.assert_frame_equal(res, exp)

        # casting to categorical shouldn't affect the result
        df["A"] = df["A"].astype("category")

        res = df == "foo"
        tm.assert_frame_equal(res, exp)

    def test_frame_in_list(self):
        # [FUNCTION DOC]
        # Regression test (GH 12689). Testing a frame for membership in a list raises a value error that the truth value of a frame is ambiguous (raised by the frame, not by internal blocks).
        # GH#12689 this should raise at the DataFrame level, not blocks
        df = DataFrame(
            np.random.default_rng(2).standard_normal((6, 4)), columns=list("ABCD")
        )
        msg = "The truth value of a DataFrame is ambiguous"
        with pytest.raises(ValueError, match=msg):
            df in [None]

    @pytest.mark.parametrize(
        "arg, arg2",
        [
            [
                {
                    "a": np.random.default_rng(2).integers(10, size=10),
                    "b": pd.date_range("20010101", periods=10, unit="ns"),
                },
                {
                    "a": np.random.default_rng(2).integers(10, size=10),
                    "b": np.random.default_rng(2).integers(10, size=10),
                },
            ],
            [
                {
                    "a": np.random.default_rng(2).integers(10, size=10),
                    "b": np.random.default_rng(2).integers(10, size=10),
                },
                {
                    "a": np.random.default_rng(2).integers(10, size=10),
                    "b": pd.date_range("20010101", periods=10, unit="ns"),
                },
            ],
            [
                {
                    "a": pd.date_range("20010101", periods=10, unit="ns"),
                    "b": pd.date_range("20010101", periods=10, unit="ns"),
                },
                {
                    "a": np.random.default_rng(2).integers(10, size=10),
                    "b": np.random.default_rng(2).integers(10, size=10),
                },
            ],
            [
                {
                    "a": np.random.default_rng(2).integers(10, size=10),
                    "b": pd.date_range("20010101", periods=10, unit="ns"),
                },
                {
                    "a": pd.date_range("20010101", periods=10, unit="ns"),
                    "b": pd.date_range("20010101", periods=10, unit="ns"),
                },
            ],
        ],
    )
    def test_comparison_invalid(self, arg, arg2):
        # [FUNCTION DOC]
        # Regression test (GH 4968). Parametrized over four pairs of frames mixing datetime and integer columns. Equality and inequality give the same column-wise results as series comparisons; ordering comparisons (greater/less, with or without equality) raise a type error with one of several accepted numpy/pandas messages about invalid comparison or no common type.
        # GH4968
        # invalid date/int comparisons
        x = DataFrame(arg)
        y = DataFrame(arg2)
        # we expect the result to match Series comparisons for
        # == and !=, inequalities should raise
        result = x == y
        expected = DataFrame(
            {col: x[col] == y[col] for col in x.columns},
            index=x.index,
            columns=x.columns,
        )
        tm.assert_frame_equal(result, expected)

        result = x != y
        expected = DataFrame(
            {col: x[col] != y[col] for col in x.columns},
            index=x.index,
            columns=x.columns,
        )
        tm.assert_frame_equal(result, expected)

        msgs = [
            r"Invalid comparison between dtype=datetime64\[ns\] and ndarray",
            "invalid type promotion",
            (
                # npdev 1.20.0
                r"The DTypes <class 'numpy.dtype\[.*\]'> and "
                r"<class 'numpy.dtype\[.*\]'> do not have a common DType."
            ),
        ]
        msg = "|".join(msgs)
        with pytest.raises(TypeError, match=msg):
            x >= y
        with pytest.raises(TypeError, match=msg):
            x > y
        with pytest.raises(TypeError, match=msg):
            x < y
        with pytest.raises(TypeError, match=msg):
            x <= y

    @pytest.mark.parametrize(
        "left, right",
        [
            ("gt", "lt"),
            ("lt", "gt"),
            ("ge", "le"),
            ("le", "ge"),
            ("eq", "eq"),
            ("ne", "ne"),
        ],
    )
    def test_timestamp_compare(self, left, right):
        # [FUNCTION DOC]
        # Regression test (GH 4982). Parametrized over operator name pairs (an operator and its mirror). Comparing a mixed frame with a timestamp on either side: equality/inequality give the same frame for both sides; ordering comparisons raise a type error about unsupported comparison between an array and a timestamp. Repeated with NaT in place of the timestamp (error mentions the NaT type). Some entries of the frame hold missing dates.
        # make sure we can compare Timestamps on the right AND left hand side
        # GH#4982
        df = DataFrame(
            {
                "dates1": pd.date_range("20010101", periods=10),
                "dates2": pd.date_range("20010102", periods=10),
                "intcol": np.random.default_rng(2).integers(1000000000, size=10),
                "floatcol": np.random.default_rng(2).standard_normal(10),
                "stringcol": [chr(100 + i) for i in range(10)],
            }
        )
        df.loc[np.random.default_rng(2).random(len(df)) > 0.5, "dates2"] = pd.NaT
        left_f = getattr(operator, left)
        right_f = getattr(operator, right)

        # no nats
        if left in ["eq", "ne"]:
            expected = left_f(df, pd.Timestamp("20010109"))
            result = right_f(pd.Timestamp("20010109"), df)
            tm.assert_frame_equal(result, expected)
        else:
            msg = (
                "'(<|>)=?' not supported between "
                "instances of 'numpy.ndarray' and 'Timestamp'"
            )
            with pytest.raises(TypeError, match=msg):
                left_f(df, pd.Timestamp("20010109"))
            with pytest.raises(TypeError, match=msg):
                right_f(pd.Timestamp("20010109"), df)
        # nats
        if left in ["eq", "ne"]:
            expected = left_f(df, pd.Timestamp("nat"))
            result = right_f(pd.Timestamp("nat"), df)
            tm.assert_frame_equal(result, expected)
        else:
            msg = (
                "'(<|>)=?' not supported between "
                "instances of 'numpy.ndarray' and 'NaTType'"
            )
            with pytest.raises(TypeError, match=msg):
                left_f(df, pd.Timestamp("nat"))
            with pytest.raises(TypeError, match=msg):
                right_f(pd.Timestamp("nat"), df)

    def test_mixed_comparison(self):
        # [FUNCTION DOC]
        # Regression test (GH 13128, GH 22163). Comparing a frame of date strings and integers with a frame of letters: equality gives all false and inequality all true, no error.
        # GH#13128, GH#22163 != datetime64 vs non-dt64 should be False,
        # not raise TypeError
        # (this appears to be fixed before GH#22163, not sure when)
        df = DataFrame([["1989-08-01", 1], ["1989-08-01", 2]])
        other = DataFrame([["a", "b"], ["c", "d"]])

        result = df == other
        assert not result.any().any()

        result = df != other
        assert result.all().all()

    def test_df_boolean_comparison_error(self):
        # [FUNCTION DOC]
        # Regression test (GH 4576, GH 22880, GH 22800). Comparing a three-by-two frame for equality with a tuple or list whose length equals the column count compares per column and gives the expected boolean frame.
        # GH#4576, GH#22880
        # comparing DataFrame against list/tuple with len(obj) matching
        #  len(df.columns) is supported as of GH#22800
        df = DataFrame(np.arange(6).reshape((3, 2)))

        expected = DataFrame([[False, False], [True, False], [False, False]])

        result = df == (2, 2)
        tm.assert_frame_equal(result, expected)

        result = df == [2, 2]
        tm.assert_frame_equal(result, expected)

    def test_df_float_none_comparison(self):
        # [FUNCTION DOC]
        # Calling the equality method with none on a float frame gives an all-false result.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((8, 3)),
            index=range(8),
            columns=["A", "B", "C"],
        )

        result = df.__eq__(None)
        assert not result.any().any()

    def test_df_string_comparison(self):
        # [FUNCTION DOC]
        # Boolean masks from comparing a numeric column and a string column select the right rows (using unary minus as the negation of the mask).
        df = DataFrame([{"a": 1, "b": "foo"}, {"a": 2, "b": "bar"}])
        mask_a = df.a > 1
        tm.assert_frame_equal(df[mask_a], df.loc[1:1, :])
        tm.assert_frame_equal(df[-mask_a], df.loc[0:0, :])

        mask_b = df.b == "foo"
        tm.assert_frame_equal(df[mask_b], df.loc[0:0, :])
        tm.assert_frame_equal(df[-mask_b], df.loc[1:1, :])


class TestFrameFlexComparisons:
    # [CLASS DOC]
    # Test container for the flexible comparison methods (eq, ne, lt, gt, le, ge).
    # TODO: test_bool_flex_frame needs a better name
    def test_bool_flex_frame(self, comparison_op):
        # [FUNCTION DOC]
        # Uses the comparison operator fixture. Flexible method and operator agree for aligned frames, unaligned frames (other reindexed), numpy arrays, a scalar and a NaN scalar; a five-dimensional array raises a value error that it cannot be coerced to a series or frame. Also eq of a frame with itself is all true and ne all false.
        # KNOWN ISSUE: the code carries a TODO that the test needs a better name; also a message variable is defined above the NaN comment instead of next to its only use.
        data = np.random.default_rng(2).standard_normal((5, 3))
        other_data = np.random.default_rng(2).standard_normal((5, 3))
        df = DataFrame(data)
        other = DataFrame(other_data)
        ndim_5 = np.ones((*df.shape, 1, 3))

        # DataFrame
        assert df.eq(df).values.all()
        assert not df.ne(df).values.any()
        f = getattr(df, comparison_op.__name__)
        o = comparison_op
        # No NAs
        tm.assert_frame_equal(f(other), o(df, other))
        # Unaligned
        part_o = other.loc[3:, 1:].copy()
        rs = f(part_o)
        xp = o(df, part_o.reindex(index=df.index, columns=df.columns))
        tm.assert_frame_equal(rs, xp)
        # ndarray
        tm.assert_frame_equal(f(other.values), o(df, other.values))
        # scalar
        tm.assert_frame_equal(f(0), o(df, 0))
        # NAs
        msg = "Unable to coerce to Series/DataFrame"
        tm.assert_frame_equal(f(np.nan), o(df, np.nan))
        with pytest.raises(ValueError, match=msg):
            f(ndim_5)

    @pytest.mark.parametrize("box", [np.array, Series])
    def test_bool_flex_series(self, box):
        # [FUNCTION DOC]
        # Parametrized over numpy array or series as the other operand. With a row-length and column-length operand, equal/not-equal, greater/less-equal and greater-equal/less comparisons along both axes match operator-based comparison, are the negations of their complements, match the transposed computation, and give identical results for list and series forms.
        # KNOWN ISSUE: the last two assignments create new random series that are never used (dead code).
        # Series
        # list/tuple
        data = np.random.default_rng(2).standard_normal((5, 3))
        df = DataFrame(data)
        idx_ser = box(np.random.default_rng(2).standard_normal(5))
        col_ser = box(np.random.default_rng(2).standard_normal(3))

        idx_eq = df.eq(idx_ser, axis=0)
        col_eq = df.eq(col_ser)
        idx_ne = df.ne(idx_ser, axis=0)
        col_ne = df.ne(col_ser)
        tm.assert_frame_equal(col_eq, df == Series(col_ser))
        tm.assert_frame_equal(col_eq, -col_ne)
        tm.assert_frame_equal(idx_eq, -idx_ne)
        tm.assert_frame_equal(idx_eq, df.T.eq(idx_ser).T)
        tm.assert_frame_equal(col_eq, df.eq(list(col_ser)))
        tm.assert_frame_equal(idx_eq, df.eq(Series(idx_ser), axis=0))
        tm.assert_frame_equal(idx_eq, df.eq(list(idx_ser), axis=0))

        idx_gt = df.gt(idx_ser, axis=0)
        col_gt = df.gt(col_ser)
        idx_le = df.le(idx_ser, axis=0)
        col_le = df.le(col_ser)

        tm.assert_frame_equal(col_gt, df > Series(col_ser))
        tm.assert_frame_equal(col_gt, -col_le)
        tm.assert_frame_equal(idx_gt, -idx_le)
        tm.assert_frame_equal(idx_gt, df.T.gt(idx_ser).T)

        idx_ge = df.ge(idx_ser, axis=0)
        col_ge = df.ge(col_ser)
        idx_lt = df.lt(idx_ser, axis=0)
        col_lt = df.lt(col_ser)
        tm.assert_frame_equal(col_ge, df >= Series(col_ser))
        tm.assert_frame_equal(col_ge, -col_lt)
        tm.assert_frame_equal(idx_ge, -idx_lt)
        tm.assert_frame_equal(idx_ge, df.T.ge(idx_ser).T)

        idx_ser = Series(np.random.default_rng(2).standard_normal(5))
        col_ser = Series(np.random.default_rng(2).standard_normal(3))

    def test_bool_flex_frame_na(self):
        # [FUNCTION DOC]
        # A missing value compares false for equality, greater, less, greater-equal and less-equal and true for inequality.
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        # NA
        df.loc[0, 0] = np.nan
        rs = df.eq(df)
        assert not rs.loc[0, 0]
        rs = df.ne(df)
        assert rs.loc[0, 0]
        rs = df.gt(df)
        assert not rs.loc[0, 0]
        rs = df.lt(df)
        assert not rs.loc[0, 0]
        rs = df.ge(df)
        assert not rs.loc[0, 0]
        rs = df.le(df)
        assert not rs.loc[0, 0]

    def test_bool_flex_frame_complex_dtype(self):
        # [FUNCTION DOC]
        # Ordering comparisons involving complex numbers raise a type error (frame method, series method, and raw numpy arrays all behave the same), including against a complex scalar; inequality gives all true for the mixed complex/missing example.
        # complex
        arr = np.array([np.nan, 1, 6, np.nan])
        arr2 = np.array([2j, np.nan, 7, None])
        df = DataFrame({"a": arr})
        df2 = DataFrame({"a": arr2})

        msg = "|".join(
            [
                "'>' not supported between instances of '.*' and 'complex'",
                r"unorderable types: .*complex\(\)",  # PY35
            ]
        )
        with pytest.raises(TypeError, match=msg):
            # inequalities are not well-defined for complex numbers
            df.gt(df2)
        with pytest.raises(TypeError, match=msg):
            # regression test that we get the same behavior for Series
            df["a"].gt(df2["a"])
        with pytest.raises(TypeError, match=msg):
            # Check that we match numpy behavior here
            df.values > df2.values

        rs = df.ne(df2)
        assert rs.values.all()

        arr3 = np.array([2j, np.nan, None])
        df3 = DataFrame({"a": arr3})

        with pytest.raises(TypeError, match=msg):
            # inequalities are not well-defined for complex numbers
            df3.gt(2j)
        with pytest.raises(TypeError, match=msg):
            # regression test that we get the same behavior for Series
            df3["a"].gt(2j)
        with pytest.raises(TypeError, match=msg):
            # Check that we match numpy behavior here
            df3.values > 2j

    def test_bool_flex_frame_object_dtype(self):
        # [FUNCTION DOC]
        # Object-dtype columns holding a string, a missing value versus a datetime: the inequality method gives false, true, false.
        # corner, dtype=object
        df1 = DataFrame({"col": ["foo", np.nan, "bar"]}, dtype=object)
        df2 = DataFrame({"col": ["foo", datetime.now(), "bar"]}, dtype=object)
        result = df1.ne(df2)
        exp = DataFrame({"col": [False, True, False]})
        tm.assert_frame_equal(result, exp)

    def test_flex_comparison_nat(self):
        # [FUNCTION DOC]
        # Regression test (GH 15697, GH 22163). Comparing a frame holding NaT with NaT: equality (operator and method) is false and inequality true, as native numpy booleans.
        # GH 15697, GH 22163 df.eq(pd.NaT) should behave like df == pd.NaT,
        # and _definitely_ not be NaN
        df = DataFrame([pd.NaT])

        result = df == pd.NaT
        # result.iloc[0, 0] is an np.bool_ object
        assert result.iloc[0, 0].item() is False

        result = df.eq(pd.NaT)
        assert result.iloc[0, 0].item() is False

        result = df != pd.NaT
        assert result.iloc[0, 0].item() is True

        result = df.ne(pd.NaT)
        assert result.iloc[0, 0].item() is True

    def test_df_flex_cmp_constant_return_types(self, comparison_op):
        # [FUNCTION DOC]
        # Regression test (GH 15077). Each flexible comparison against a scalar on a non-empty mixed frame gives two boolean columns.
        # GH 15077, non-empty DataFrame
        df = DataFrame({"x": [1, 2, 3], "y": [1.0, 2.0, 3.0]})
        const = 2

        result = getattr(df, comparison_op.__name__)(const).dtypes.value_counts()
        tm.assert_series_equal(
            result, Series([2], index=[np.dtype(bool)], name="count")
        )

    def test_df_flex_cmp_constant_return_types_empty(self, comparison_op):
        # [FUNCTION DOC]
        # Regression test (GH 15077). Same as above for a frame with zero rows: result dtypes are still boolean.
        # GH 15077 empty DataFrame
        df = DataFrame({"x": [1, 2, 3], "y": [1.0, 2.0, 3.0]})
        const = 2

        empty = df.iloc[:0]
        result = getattr(empty, comparison_op.__name__)(const).dtypes.value_counts()
        tm.assert_series_equal(
            result, Series([2], index=[np.dtype(bool)], name="count")
        )

    def test_df_flex_cmp_ea_dtype_with_ndarray_series(self):
        # [FUNCTION DOC]
        # A frame of interval extension columns compared with equality against a numeric series along rows and a labeled series along columns gives all false.
        ii = pd.IntervalIndex.from_breaks([1, 2, 3])
        df = DataFrame({"A": ii, "B": ii})

        ser = Series([0, 0])
        res = df.eq(ser, axis=0)

        expected = DataFrame({"A": [False, False], "B": [False, False]})
        tm.assert_frame_equal(res, expected)

        ser2 = Series([1, 2], index=["A", "B"])
        res2 = df.eq(ser2, axis=1)
        tm.assert_frame_equal(res2, expected)


# -------------------------------------------------------------------
# Arithmetic


class TestFrameFlexArithmetic:
    # [CLASS DOC]
    # Test container for the flexible arithmetic methods (add, sub, mul, div, floordiv, ...), including fill values, alignment, level and axis arguments.
    def test_floordiv_axis0(self):
        # [FUNCTION DOC]
        # Floor division along rows by a series or by its raw array equals the column-wise series result.
        # make sure we df.floordiv(ser, axis=0) matches column-wise result
        arr = np.arange(3)
        ser = Series(arr)
        df = DataFrame({"A": ser, "B": ser})

        result = df.floordiv(ser, axis=0)

        expected = DataFrame({col: df[col] // ser for col in df.columns})

        tm.assert_frame_equal(result, expected)

        result2 = df.floordiv(ser.values, axis=0)
        tm.assert_frame_equal(result2, expected)

    def test_df_add_td64_columnwise(self):
        # [FUNCTION DOC]
        # Regression test (GH 22534). Adding a timedelta series along rows to a frame of a datetime and a timedelta column broadcasts column-wise correctly.
        # GH 22534 Check that column-wise addition broadcasts correctly
        dti = pd.date_range("2016-01-01", periods=10)
        tdi = pd.timedelta_range("1", periods=10)
        tser = Series(tdi)
        df = DataFrame({0: dti, 1: tdi})

        result = df.add(tser, axis=0)
        expected = DataFrame({0: dti + tdi, 1: tdi + tdi})
        tm.assert_frame_equal(result, expected)

    def test_df_add_flex_filled_mixed_dtypes(self):
        # [FUNCTION DOC]
        # Regression test (GH 19611). Adding frames of datetime and timedelta data with a timedelta fill value fills missing entries before adding, giving shifted dates and doubled timedeltas.
        # GH 19611
        dti = pd.date_range("2016-01-01", periods=3)
        ser = Series(["1 Day", "NaT", "2 Days"], dtype="timedelta64[ns]")
        df = DataFrame({"A": dti, "B": ser})
        other = DataFrame({"A": ser, "B": ser})
        fill = pd.Timedelta(days=1).to_timedelta64()
        result = df.add(other, fill_value=fill)

        expected = DataFrame(
            {
                "A": Series(
                    ["2016-01-02", "2016-01-03", "2016-01-05"], dtype="datetime64[ns]"
                ),
                "B": ser * 2,
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_arith_flex_frame(
        self, all_arithmetic_operators, float_frame, mixed_float_frame
    ):
        # [FUNCTION DOC]
        # Uses the fixture iterating all arithmetic dunder names (including reflected ones). The method called by name on the float frame and mixed-float frame with twice itself equals the operator result computed with the standard library operator (swapping arguments for reflected names); the mixed frame's dtypes are checked afterwards (column C allowed to differ).
        # one instance of parametrized fixture
        op = all_arithmetic_operators

        def f(x, y):
            # [FUNCTION DOC]
            # HELPER: applies the standard operator corresponding to the closed-over dunder name to two operands. For reflected names (starting with the reflected prefix) it uses the non-reflected operator with swapped operands.
            # r-versions not in operator-stdlib; get op without "r" and invert
            if op.startswith("__r"):
                return getattr(operator, op.replace("__r", "__"))(y, x)
            return getattr(operator, op)(x, y)

        result = getattr(float_frame, op)(2 * float_frame)
        expected = f(float_frame, 2 * float_frame)
        tm.assert_frame_equal(result, expected)

        # vs mix float
        result = getattr(mixed_float_frame, op)(2 * mixed_float_frame)
        expected = f(mixed_float_frame, 2 * mixed_float_frame)
        tm.assert_frame_equal(result, expected)
        _check_mixed_float(result, dtype={"C": None})

    @pytest.mark.parametrize("op", ["__add__", "__sub__", "__mul__"])
    def test_arith_flex_frame_mixed(
        self,
        op,
        int_frame,
        mixed_int_frame,
        mixed_float_frame,
        switch_numexpr_min_elements,
    ):
        # [FUNCTION DOC]
        # Parametrized over add, sub, mul. Uses the integer, mixed-integer and mixed-float frame fixtures and the numexpr threshold fixture. Method results equal operator results; mixed-integer dtype expectations account for unsigned overflow avoidance and, with numexpr active, for different casting of the integer 32-bit column; the mixed float frame and plain int frame are also checked.
        f = getattr(operator, op)

        # vs mix int
        result = getattr(mixed_int_frame, op)(2 + mixed_int_frame)
        expected = f(mixed_int_frame, 2 + mixed_int_frame)

        # no overflow in the uint
        dtype = None
        if op in ["__sub__"]:
            dtype = {"B": "uint64", "C": None}
        elif op in ["__add__", "__mul__"]:
            dtype = {"C": None}
        if expr.USE_NUMEXPR and switch_numexpr_min_elements == 0:
            # when using numexpr, the casting rules are slightly different:
            # in the `2 + mixed_int_frame` operation, int32 column becomes
            # and int64 column (not preserving dtype in operation with Python
            # scalar), and then the int32/int64 combo results in int64 result
            dtype["A"] = (2 + mixed_int_frame)["A"].dtype
        tm.assert_frame_equal(result, expected)
        _check_mixed_int(result, dtype=dtype)

        # vs mix float
        result = getattr(mixed_float_frame, op)(2 * mixed_float_frame)
        expected = f(mixed_float_frame, 2 * mixed_float_frame)
        tm.assert_frame_equal(result, expected)
        _check_mixed_float(result, dtype={"C": None})

        # vs plain int
        result = getattr(int_frame, op)(2 * int_frame)
        expected = f(int_frame, 2 * int_frame)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dim", range(3, 6))
    def test_arith_flex_frame_raise(self, all_arithmetic_operators, float_frame, dim):
        # [FUNCTION DOC]
        # Parametrized over array dimensions 3 to 5 and all arithmetic names. A higher-dimensional array operand raises a value error that it cannot be coerced to a series or frame.
        # one instance of parametrized fixture
        op = all_arithmetic_operators

        # Check that arrays with dim >= 3 raise
        arr = np.ones((1,) * dim)
        msg = "Unable to coerce to Series/DataFrame"
        with pytest.raises(ValueError, match=msg):
            getattr(float_frame, op)(arr)

    def test_arith_flex_frame_corner(self, float_frame):
        # [FUNCTION DOC]
        # Adding a scalar equals the plus operator; adding an empty slice in either order gives an all-NaN frame with sorted labels; supplying a fill value when the other operand is a series raises a not-implemented error mentioning the fill value (with or without an explicit axis).
        const_add = float_frame.add(1)
        tm.assert_frame_equal(const_add, float_frame + 1)

        # corner cases
        result = float_frame.add(float_frame[:0])
        expected = float_frame.sort_index() * np.nan
        tm.assert_frame_equal(result, expected)

        result = float_frame[:0].add(float_frame)
        expected = float_frame.sort_index() * np.nan
        tm.assert_frame_equal(result, expected)

        with pytest.raises(NotImplementedError, match="fill_value"):
            float_frame.add(float_frame.iloc[0], fill_value=3)

        with pytest.raises(NotImplementedError, match="fill_value"):
            float_frame.add(float_frame.iloc[0], axis="index", fill_value=3)

    @pytest.mark.parametrize("op", ["add", "sub", "mul", "mod"])
    def test_arith_flex_series_ops(self, simple_frame, op):
        # [FUNCTION DOC]
        # Parametrized over add, sub, mul, mod. Method with a row series matches the operator, and with a column series along rows matches the transposed operation.
        # KNOWN ISSUE: the comment says to add true division here after an arithmetic refactor; it is stale.
        # after arithmetic refactor, add truediv here
        df = simple_frame

        row = df.xs("a")
        col = df["two"]
        f = getattr(df, op)
        op = getattr(operator, op)
        tm.assert_frame_equal(f(row), op(df, row))
        tm.assert_frame_equal(f(col, axis=0), op(df.T, col).T)

    def test_arith_flex_series(self, simple_frame):
        # [FUNCTION DOC]
        # Adding a row with no axis (explicit none) equals the plus operator; division with a row and with a column along rows equals the divide operator forms.
        df = simple_frame

        row = df.xs("a")
        col = df["two"]
        # special case for some reason
        tm.assert_frame_equal(df.add(row, axis=None), df + row)

        # cases which will be refactored after big arithmetic refactor
        tm.assert_frame_equal(df.div(row), df / row)
        tm.assert_frame_equal(df.div(col, axis=0), (df.T / col).T)

    def test_arith_flex_series_broadcasting(self, any_real_numpy_dtype):
        # [FUNCTION DOC]
        # Regression test (GH 7325). Parametrized over real numpy dtypes. Dividing a frame by its first column along the index gives the expected quotients (infinity from division by zero, NaN from zero over zero), float32 kept when the input is float32.
        # broadcasting issue in GH 7325
        df = DataFrame(np.arange(3 * 2).reshape((3, 2)), dtype=any_real_numpy_dtype)
        expected = DataFrame([[np.nan, np.inf], [1.0, 1.5], [1.0, 1.25]])
        if any_real_numpy_dtype == "float32":
            expected = expected.astype(any_real_numpy_dtype)
        result = df.div(df[0], axis="index")
        tm.assert_frame_equal(result, expected)

    def test_arith_flex_zero_len_raises(self):
        # [FUNCTION DOC]
        # Regression test (GH 19522). A fill value passed with an empty series or empty frame operand still raises a not-implemented error.
        # GH 19522 passing fill_value to frame flex arith methods should
        # raise even in the zero-length special cases
        ser_len0 = Series([], dtype=object)
        df_len0 = DataFrame(columns=["A", "B"])
        df = DataFrame([[1, 2], [3, 4]], columns=["A", "B"])

        with pytest.raises(NotImplementedError, match="fill_value"):
            df.add(ser_len0, fill_value="E")

        with pytest.raises(NotImplementedError, match="fill_value"):
            df_len0.sub(df["A"], axis=None, fill_value=3)

    def test_flex_add_scalar_fill_value(self):
        # [FUNCTION DOC]
        # Regression test (GH 12723). Adding a scalar with fill value zero equals filling missing values with zero then adding.
        # GH#12723
        dat = np.array([0, 1, np.nan, 3, 4, 5], dtype="float")
        df = DataFrame({"foo": dat}, index=range(6))

        exp = df.fillna(0).add(2)
        res = df.add(2, fill_value=0)
        tm.assert_frame_equal(res, exp)

    def test_sub_alignment_with_duplicate_index(self):
        # [FUNCTION DOC]
        # Regression test (GH 5185). Subtracting frames where one index has duplicates aligns and gives the expected result with duplicated labels.
        # GH#5185 dup aligning operations should work
        df1 = DataFrame([1, 2, 3, 4, 5], index=[1, 2, 1, 2, 3])
        df2 = DataFrame([1, 2, 3], index=[1, 2, 3])
        expected = DataFrame([0, 2, 0, 2, 2], index=[1, 1, 2, 2, 3])
        result = df1.sub(df2)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("op", ["__add__", "__mul__", "__sub__", "__truediv__"])
    def test_arithmetic_with_duplicate_columns(self, op):
        # [FUNCTION DOC]
        # Parametrized over add, mul, sub, truediv. Operations on a frame with duplicate column labels give the same values as the unique-label version.
        # operations
        df = DataFrame({"A": np.arange(10), "B": np.random.default_rng(2).random(10)})
        expected = getattr(df, op)(df)
        expected.columns = ["A", "A"]
        df.columns = ["A", "A"]
        result = getattr(df, op)(df)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("level", [0, None])
    def test_broadcast_multiindex(self, level):
        # [FUNCTION DOC]
        # Regression test (GH 34388). Parametrized over level 0 and none. Adding a frame with a two-level column index to one with a one-level named index, joining on the shared level, broadcasts and returns the two-level columns.
        # GH34388
        df1 = DataFrame({"A": [0, 1, 2], "B": [1, 2, 3]})
        df1.columns = df1.columns.set_names("L1")

        df2 = DataFrame({("A", "C"): [0, 0, 0], ("A", "D"): [0, 0, 0]})
        df2.columns = df2.columns.set_names(["L1", "L2"])

        result = df1.add(df2, level=level)
        expected = DataFrame({("A", "C"): [0, 1, 2], ("A", "D"): [0, 1, 2]})
        expected.columns = expected.columns.set_names(["L1", "L2"])

        tm.assert_frame_equal(result, expected)

    def test_frame_multiindex_operations(self):
        # [FUNCTION DOC]
        # Regression test (GH 43321). Adding a series with a two-level index (a subset of the frame's row index levels in different order) along rows broadcasts over the remaining level, giving float sums.
        # GH 43321
        df = DataFrame(
            {2010: [1, 2, 3], 2020: [3, 4, 5]},
            index=MultiIndex.from_product(
                [["a"], ["b"], [0, 1, 2]], names=["scen", "mod", "id"]
            ),
        )

        series = Series(
            [0.4],
            index=MultiIndex.from_product([["b"], ["a"]], names=["mod", "scen"]),
        )

        expected = DataFrame(
            {2010: [1.4, 2.4, 3.4], 2020: [3.4, 4.4, 5.4]},
            index=MultiIndex.from_product(
                [["a"], ["b"], [0, 1, 2]], names=["scen", "mod", "id"]
            ),
        )
        result = df.add(series, axis=0)

        tm.assert_frame_equal(result, expected)

    def test_frame_multiindex_operations_series_index_to_frame_index(self):
        # [FUNCTION DOC]
        # Regression test (GH 43321). A series whose index has more levels than the frame's row index adds along rows and yields the longer index with broadcast frame values.
        # GH 43321
        df = DataFrame(
            {2010: [1], 2020: [3]},
            index=MultiIndex.from_product([["a"], ["b"]], names=["scen", "mod"]),
        )

        series = Series(
            [10.0, 20.0, 30.0],
            index=MultiIndex.from_product(
                [["a"], ["b"], [0, 1, 2]], names=["scen", "mod", "id"]
            ),
        )

        expected = DataFrame(
            {2010: [11.0, 21, 31.0], 2020: [13.0, 23.0, 33.0]},
            index=MultiIndex.from_product(
                [["a"], ["b"], [0, 1, 2]], names=["scen", "mod", "id"]
            ),
        )
        result = df.add(series, axis=0)

        tm.assert_frame_equal(result, expected)

    def test_frame_multiindex_operations_no_align(self):
        # [FUNCTION DOC]
        # A series whose index labels do not match any row label gives a result of all NaN with an extra row for the unmatched label.
        df = DataFrame(
            {2010: [1, 2, 3], 2020: [3, 4, 5]},
            index=MultiIndex.from_product(
                [["a"], ["b"], [0, 1, 2]], names=["scen", "mod", "id"]
            ),
        )

        series = Series(
            [0.4],
            index=MultiIndex.from_product([["c"], ["a"]], names=["mod", "scen"]),
        )

        expected = DataFrame(
            {2010: np.nan, 2020: np.nan},
            index=MultiIndex.from_tuples(
                [
                    ("a", "b", 0),
                    ("a", "b", 1),
                    ("a", "b", 2),
                    ("a", "c", np.nan),
                ],
                names=["scen", "mod", "id"],
            ),
        )
        result = df.add(series, axis=0)

        tm.assert_frame_equal(result, expected)

    def test_frame_multiindex_operations_part_align(self):
        # [FUNCTION DOC]
        # A series matching only some rows gives sums for matching rows and NaN for the others, keeping the frame's index.
        df = DataFrame(
            {2010: [1, 2, 3], 2020: [3, 4, 5]},
            index=MultiIndex.from_tuples(
                [
                    ("a", "b", 0),
                    ("a", "b", 1),
                    ("a", "c", 2),
                ],
                names=["scen", "mod", "id"],
            ),
        )

        series = Series(
            [0.4],
            index=MultiIndex.from_product([["b"], ["a"]], names=["mod", "scen"]),
        )

        expected = DataFrame(
            {2010: [1.4, 2.4, np.nan], 2020: [3.4, 4.4, np.nan]},
            index=MultiIndex.from_tuples(
                [
                    ("a", "b", 0),
                    ("a", "b", 1),
                    ("a", "c", 2),
                ],
                names=["scen", "mod", "id"],
            ),
        )
        result = df.add(series, axis=0)

        tm.assert_frame_equal(result, expected)

    def test_frame_multiindex_operations_part_align_axis1(self):
        # [FUNCTION DOC]
        # Regression test (GH 61009). Same partial-alignment behavior when the multi-index is on the columns and the series is added along axis 1.
        # GH#61009 Test DataFrame-Series arithmetic operation
        # with partly aligned MultiIndex and axis = 1
        df = DataFrame(
            [[1, 2, 3], [3, 4, 5]],
            index=[2010, 2020],
            columns=MultiIndex.from_tuples(
                [
                    ("a", "b", 0),
                    ("a", "b", 1),
                    ("a", "c", 2),
                ],
                names=["scen", "mod", "id"],
            ),
        )

        series = Series(
            [0.4],
            index=MultiIndex.from_product([["b"], ["a"]], names=["mod", "scen"]),
        )

        expected = DataFrame(
            [[1.4, 2.4, np.nan], [3.4, 4.4, np.nan]],
            index=[2010, 2020],
            columns=MultiIndex.from_tuples(
                [
                    ("a", "b", 0),
                    ("a", "b", 1),
                    ("a", "c", 2),
                ],
                names=["scen", "mod", "id"],
            ),
        )
        result = df.add(series, axis=1)

        tm.assert_frame_equal(result, expected)


class TestFrameArithmetic:
    # [CLASS DOC]
    # Test container for operator-based frame arithmetic.
    def test_td64_op_nat_casting(self):
        # [FUNCTION DOC]
        # A timedelta series of all NaT multiplied with an integer frame gives timedelta columns, not datetimes.
        # Make sure we don't accidentally treat timedelta64(NaT) as datetime64
        #  when calling dispatch_to_series in DataFrame arithmetic
        ser = Series(["NaT", "NaT"], dtype="timedelta64[ns]")
        df = DataFrame([[1, 2], [3, 4]])

        result = df * ser
        expected = DataFrame({0: ser, 1: ser})
        tm.assert_frame_equal(result, expected)

    def test_df_add_2d_array_rowlike_broadcasts(self):
        # [FUNCTION DOC]
        # Regression test (GH 23000). Adding a (1, ncols) array to a frame, on either side, broadcasts down the rows, keeping integer dtype.
        # GH#23000
        arr = np.arange(6).reshape(3, 2)
        df = DataFrame(arr, columns=[True, False], index=["A", "B", "C"])

        rowlike = arr[[1], :]  # shape --> (1, ncols)
        assert rowlike.shape == (1, df.shape[1])

        expected = DataFrame(
            [[2, 4], [4, 6], [6, 8]],
            columns=df.columns,
            index=df.index,
            # specify dtype explicitly to avoid failing
            # on 32bit builds
            dtype=arr.dtype,
        )
        result = df + rowlike
        tm.assert_frame_equal(result, expected)
        result = rowlike + df
        tm.assert_frame_equal(result, expected)

    def test_df_add_2d_array_collike_broadcasts(self):
        # [FUNCTION DOC]
        # Regression test (GH 23000). Adding an (nrows, 1) array broadcasts across columns, on either side, keeping integer dtype.
        # GH#23000
        arr = np.arange(6).reshape(3, 2)
        df = DataFrame(arr, columns=[True, False], index=["A", "B", "C"])

        collike = arr[:, [1]]  # shape --> (nrows, 1)
        assert collike.shape == (df.shape[0], 1)

        expected = DataFrame(
            [[1, 2], [5, 6], [9, 10]],
            columns=df.columns,
            index=df.index,
            # specify dtype explicitly to avoid failing
            # on 32bit builds
            dtype=arr.dtype,
        )
        result = df + collike
        tm.assert_frame_equal(result, expected)
        result = collike + df
        tm.assert_frame_equal(result, expected)

    def test_df_arith_2d_array_rowlike_broadcasts(
        self, request, all_arithmetic_operators
    ):
        # [FUNCTION DOC]
        # Regression test (GH 23000). For every arithmetic dunder name, applying it with a (1, ncols) array equals applying it between each row and the flattened array, assembled into a frame.
        # KNOWN ISSUE: the request fixture parameter is accepted but not used.
        # GH#23000
        opname = all_arithmetic_operators
        arr = np.arange(6).reshape(3, 2)
        df = DataFrame(arr, columns=[True, False], index=["A", "B", "C"])

        rowlike = arr[[1], :]  # shape --> (1, ncols)
        assert rowlike.shape == (1, df.shape[1])

        exvals = [
            getattr(df.loc["A"], opname)(rowlike.squeeze()),
            getattr(df.loc["B"], opname)(rowlike.squeeze()),
            getattr(df.loc["C"], opname)(rowlike.squeeze()),
        ]

        expected = DataFrame(exvals, columns=df.columns, index=df.index)

        result = getattr(df, opname)(rowlike)
        tm.assert_frame_equal(result, expected)

    def test_df_arith_2d_array_collike_broadcasts(
        self, request, all_arithmetic_operators
    ):
        # [FUNCTION DOC]
        # Regression test (GH 23000). For every arithmetic dunder name, applying it with an (nrows, 1) array equals applying it column by column; for reflected modulo and reflected floor division the expected frame is upcast to a common float type because the series results can be mixed int/float.
        # KNOWN ISSUE: the request fixture parameter is accepted but not used.
        # GH#23000
        opname = all_arithmetic_operators
        arr = np.arange(6).reshape(3, 2)
        df = DataFrame(arr, columns=[True, False], index=["A", "B", "C"])

        collike = arr[:, [1]]  # shape --> (nrows, 1)
        assert collike.shape == (df.shape[0], 1)

        exvals = {
            True: getattr(df[True], opname)(collike.squeeze()),
            False: getattr(df[False], opname)(collike.squeeze()),
        }

        dtype = None
        if opname in ["__rmod__", "__rfloordiv__"]:
            # Series ops may return mixed int/float dtypes in cases where
            #   DataFrame op will return all-float.  So we upcast `expected`
            dtype = np.common_type(*(x.values for x in exvals.values()))

        expected = DataFrame(exvals, columns=df.columns, index=df.index, dtype=dtype)

        result = getattr(df, opname)(collike)
        tm.assert_frame_equal(result, expected)

    def test_df_bool_mul_int(self):
        # [FUNCTION DOC]
        # Regression test (GH 22047, GH 22163). Multiplying a boolean frame by integer one (either order) gives integer columns (checked by dtype kind), not object.
        # GH 22047, GH 22163 multiplication by 1 should result in int dtype,
        # not object dtype
        df = DataFrame([[False, True], [False, False]])
        result = df * 1

        # On appveyor this comes back as np.int32 instead of np.int64,
        # so we check dtype.kind instead of just dtype
        kinds = result.dtypes.apply(lambda x: x.kind)
        assert (kinds == "i").all()

        result = 1 * df
        kinds = result.dtypes.apply(lambda x: x.kind)
        assert (kinds == "i").all()

    def test_arith_mixed(self):
        # [FUNCTION DOC]
        # Adding a frame of strings and integers to itself concatenates the strings and doubles the integers.
        left = DataFrame({"A": ["a", "b", "c"], "B": [1, 2, 3]})

        result = left + left
        expected = DataFrame({"A": ["aa", "bb", "cc"], "B": [2, 4, 6]})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("col", ["A", "B"])
    def test_arith_getitem_commute(self, all_arithmetic_functions, col):
        # [FUNCTION DOC]
        # Parametrized over column A and B and all arithmetic functions (including reflected). Applying the function with scalar one to the frame and then selecting a column equals applying it to that column.
        df = DataFrame({"A": [1.1, 3.3], "B": [2.5, -3.9]})
        result = all_arithmetic_functions(df, 1)[col]
        expected = all_arithmetic_functions(df[col], 1)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "values", [[1, 2], (1, 2), np.array([1, 2]), range(1, 3), deque([1, 2])]
    )
    def test_arith_alignment_non_pandas_object(self, values):
        # [FUNCTION DOC]
        # Regression test (GH 17901). Parametrized over list, tuple, array, range and double-ended queue operands of length equal to the column count: added to each row, giving the expected sums.
        # GH#17901
        df = DataFrame({"A": [1, 1], "B": [1, 1]})
        expected = DataFrame({"A": [2, 2], "B": [3, 3]})
        result = df + values
        tm.assert_frame_equal(result, expected)

    def test_arith_non_pandas_object(self):
        # [FUNCTION DOC]
        # A frame plus a one-dimensional array of row values, plus a list of column values with row axis, and plus a two-dimensional random array of the frame's shape equals the manually computed frame.
        df = DataFrame(
            np.arange(1, 10, dtype="f8").reshape(3, 3),
            columns=["one", "two", "three"],
            index=["a", "b", "c"],
        )

        val1 = df.xs("a").values
        added = DataFrame(df.values + val1, index=df.index, columns=df.columns)
        tm.assert_frame_equal(df + val1, added)

        added = DataFrame((df.values.T + val1).T, index=df.index, columns=df.columns)
        tm.assert_frame_equal(df.add(val1, axis=0), added)

        val2 = list(df["two"])

        added = DataFrame(df.values + val2, index=df.index, columns=df.columns)
        tm.assert_frame_equal(df + val2, added)

        added = DataFrame((df.values.T + val2).T, index=df.index, columns=df.columns)
        tm.assert_frame_equal(df.add(val2, axis="index"), added)

        val3 = np.random.default_rng(2).random(df.shape)
        added = DataFrame(df.values + val3, index=df.index, columns=df.columns)
        tm.assert_frame_equal(df.add(val3), added)

    def test_operations_with_interval_categories_index(self, all_arithmetic_operators):
        # [FUNCTION DOC]
        # Regression test (GH 27415). With columns being a categorical index of intervals, applying every arithmetic dunder with a scalar equals the elementwise integer operation results.
        # GH#27415
        op = all_arithmetic_operators
        ind = pd.CategoricalIndex(pd.interval_range(start=0.0, end=2.0))
        data = [1, 2]
        df = DataFrame([data], columns=ind)
        num = 10
        result = getattr(df, op)(num)
        expected = DataFrame([[getattr(n, op)(num) for n in data]], columns=ind)
        tm.assert_frame_equal(result, expected)

    def test_frame_with_frame_reindex(self):
        # [FUNCTION DOC]
        # Regression test (GH 31623). Subtracting a one-column datetime frame from a two-column one gives timedelta zeros in the shared column and NaN in the other, columns ordered bar then foo.
        # GH#31623
        df = DataFrame(
            {
                "foo": [pd.Timestamp("2019"), pd.Timestamp("2020")],
                "bar": [pd.Timestamp("2018"), pd.Timestamp("2021")],
            },
            columns=["foo", "bar"],
            dtype="M8[ns]",
        )
        df2 = df[["foo"]]

        result = df - df2

        expected = DataFrame(
            {"foo": [pd.Timedelta(0), pd.Timedelta(0)], "bar": [np.nan, np.nan]},
            columns=["bar", "foo"],
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "value, dtype",
        [
            (1, "i8"),
            (1.0, "f8"),
            (2**63, "f8"),
            (1j, "complex128"),
            (2**63, "complex128"),
            (True, "bool"),
            (np.timedelta64(20, "ns"), "<m8[ns]"),
            (np.datetime64(20, "ns"), "<M8[ns]"),
        ],
    )
    @pytest.mark.parametrize(
        "op",
        [
            operator.add,
            operator.sub,
            operator.mul,
            operator.truediv,
            operator.mod,
            operator.pow,
        ],
        ids=lambda x: x.__name__,
    )
    def test_binop_other(self, op, value, dtype, switch_numexpr_min_elements):
        # [FUNCTION DOC]
        # Parametrized over scalar value and dtype pairs (integer, float, large integer, complex, boolean, timedelta, datetime), operators (add, sub, mul, truediv, mod, pow) and the numexpr threshold fixture. A one-column frame is combined with the stand-in element: for operator and dtype combinations listed as invalid, a type error is expected (message depends on the combination: none for a few allowed-but-raising ones, remainder not supported for complex, numpy boolean subtract message, otherwise cannot perform with this index type), with a Python-evaluation fallback warning for boolean subtraction when numexpr is active at threshold zero; for the skipped boolean combinations a not-implemented error (or a possible user warning for add/multiply when numexpr is older than a specific version); otherwise the result dtypes equal those from using the raw value, with no warning.
        skip = {
            (operator.truediv, "bool"),
            (operator.pow, "bool"),
            (operator.add, "bool"),
            (operator.mul, "bool"),
        }

        elem = DummyElement(value, dtype)
        df = DataFrame({"A": [elem.value, elem.value]}, dtype=elem.dtype)

        invalid = {
            (operator.pow, "<M8[ns]"),
            (operator.mod, "<M8[ns]"),
            (operator.truediv, "<M8[ns]"),
            (operator.mul, "<M8[ns]"),
            (operator.add, "<M8[ns]"),
            (operator.pow, "<m8[ns]"),
            (operator.mul, "<m8[ns]"),
            (operator.sub, "bool"),
            (operator.mod, "complex128"),
        }

        ne = import_optional_dependency("numexpr", errors="ignore")
        ne_warns_on_op = ne is not None and Version(ne.__version__) < Version("2.13.1")
        if (op, dtype) in invalid:
            warn = None
            if (dtype == "<M8[ns]" and op == operator.add) or (
                dtype == "<m8[ns]" and op == operator.mul
            ):
                msg = None
            elif dtype == "complex128":
                msg = "ufunc 'remainder' not supported for the input types"
            elif op is operator.sub:
                msg = "numpy boolean subtract, the `-` operator, is "
                if (
                    dtype == "bool"
                    and expr.USE_NUMEXPR
                    and switch_numexpr_min_elements == 0
                ):
                    warn = UserWarning
            else:
                msg = (
                    f"cannot perform __{op.__name__}__ with this "
                    "index type: (DatetimeArray|TimedeltaArray)"
                )

            with pytest.raises(TypeError, match=msg):
                with tm.assert_produces_warning(warn, match="evaluating in Python"):
                    op(df, elem.value)

        elif (op, dtype) in skip:
            if op in [operator.add, operator.mul]:
                if (
                    expr.USE_NUMEXPR
                    and switch_numexpr_min_elements == 0
                    and ne_warns_on_op
                ):
                    warn = UserWarning
                else:
                    warn = None
                with tm.assert_produces_warning(warn, match="evaluating in Python"):
                    op(df, elem.value)

            else:
                msg = "operator '.*' not implemented for .* dtypes"
                with pytest.raises(NotImplementedError, match=msg):
                    op(df, elem.value)

        else:
            with tm.assert_produces_warning(None):
                result = op(df, elem.value).dtypes
                expected = op(df, value).dtypes
            tm.assert_series_equal(result, expected)

    def test_arithmetic_midx_cols_different_dtypes(self):
        # [FUNCTION DOC]
        # Regression test (GH 49769). Subtracting frames whose multi-index columns have levels with different dtypes (integer versus nullable integer) aligns and gives zeros with the left columns.
        # GH#49769
        midx = MultiIndex.from_arrays([Series([1, 2]), Series([3, 4])])
        midx2 = MultiIndex.from_arrays([Series([1, 2], dtype="Int8"), Series([3, 4])])
        left = DataFrame([[1, 2], [3, 4]], columns=midx)
        right = DataFrame([[1, 2], [3, 4]], columns=midx2)
        result = left - right
        expected = DataFrame([[0, 0], [0, 0]], columns=midx)
        tm.assert_frame_equal(result, expected)

    def test_arithmetic_midx_cols_different_dtypes_different_order(self):
        # [FUNCTION DOC]
        # Regression test (GH 49769). Same, but the right columns are in a different order: values align by label, giving -1 and 1.
        # GH#49769
        midx = MultiIndex.from_arrays([Series([1, 2]), Series([3, 4])])
        midx2 = MultiIndex.from_arrays([Series([2, 1], dtype="Int8"), Series([4, 3])])
        left = DataFrame([[1, 2], [3, 4]], columns=midx)
        right = DataFrame([[1, 2], [3, 4]], columns=midx2)
        result = left - right
        expected = DataFrame([[-1, 1], [-1, 1]], columns=midx)
        tm.assert_frame_equal(result, expected)

    def test_div_axis_level_multiindex_columns(self):
        # [FUNCTION DOC]
        # Regression test (GH 64428). Dividing a frame with two-level columns by a flat-column frame along axis 1 joining on level 1 divides each column group by the matching column, per row.
        # GH#64428
        x = DataFrame(
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            columns=MultiIndex.from_product((["X", "Y"], ["A", "B"])),
        )
        y = DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        result = x.div(y, axis=1, level=1)
        expected = DataFrame(
            [[1 / 1, 2 / 2, 3 / 1, 4 / 2], [5 / 3, 6 / 4, 7 / 3, 8 / 4]],
            columns=MultiIndex.from_product((["X", "Y"], ["A", "B"])),
        )
        tm.assert_frame_equal(result, expected)


def test_frame_with_zero_len_series_corner_cases():
    # [FUNCTION DOC]
    # Regression test (GH 28600). Adding an empty float series to a frame gives all NaN with the frame's labels. Comparing a frame (float or datetime-typed) to an empty series raises a value error that they are not aligned (automatic alignment for comparisons was removed).
    # GH#28600
    # easy all-float case
    df = DataFrame(
        np.random.default_rng(2).standard_normal(6).reshape(3, 2), columns=["A", "B"]
    )
    ser = Series(dtype=np.float64)

    result = df + ser
    expected = DataFrame(df.values * np.nan, columns=df.columns)
    tm.assert_frame_equal(result, expected)

    with pytest.raises(ValueError, match="not aligned"):
        # Automatic alignment for comparisons deprecated GH#36795, enforced 2.0
        df == ser

    # non-float case should not raise TypeError on comparison
    df2 = DataFrame(df.values.view("M8[ns]"), columns=df.columns)
    with pytest.raises(ValueError, match="not aligned"):
        # Automatic alignment for comparisons deprecated
        df2 == ser


def test_zero_len_frame_with_series_corner_cases():
    # [FUNCTION DOC]
    # Regression test (GH 28600). Adding a series to a frame with no rows returns an equal frame.
    # GH#28600
    df = DataFrame(columns=["A", "B"], dtype=np.float64)
    ser = Series([1, 2], index=["A", "B"])

    result = df + ser
    expected = df
    tm.assert_frame_equal(result, expected)


def test_frame_single_columns_object_sum_axis_1():
    # [FUNCTION DOC]
    # Regression test (GH 13758). Summing a one-column object frame along columns keeps the string and float and treats the missing value as zero.
    # GH 13758
    data = {
        "One": Series(["A", 1.2, np.nan]),
    }
    df = DataFrame(data)
    result = df.sum(axis=1)
    expected = Series(["A", 1.2, 0])
    tm.assert_series_equal(result, expected)


# -------------------------------------------------------------------
# Unsorted
#  These arithmetic tests were previously in other files, eventually
#  should be parametrized and put into tests.arithmetic


class TestFrameArithmeticUnsorted:
    # [CLASS DOC]
    # Test container for arithmetic tests moved from other files that are intended to be parametrized and moved into the arithmetic test package eventually.
    def test_frame_add_tz_mismatch_converts_to_utc(self):
        # [FUNCTION DOC]
        # Adding two frames with indexes in different timezones aligns on the instants and the result's index is in UTC (the standard library singleton), in either operand order.
        rng = pd.date_range("1/1/2011", periods=10, freq="h", tz="US/Eastern")
        df = DataFrame(
            np.random.default_rng(2).standard_normal(len(rng)), index=rng, columns=["a"]
        )

        df_moscow = df.tz_convert("Europe/Moscow")
        result = df + df_moscow
        assert result.index.tz is timezone.utc

        result = df_moscow + df
        assert result.index.tz is timezone.utc

    def test_align_frame(self):
        # [FUNCTION DOC]
        # Adding a yearly-period-indexed frame to its every-other-row slice gives the sum for matching rows and NaN for others, also when the slice is shuffled.
        rng = pd.period_range("1/1/2000", "1/1/2010", freq="Y")
        ts = DataFrame(
            np.random.default_rng(2).standard_normal((len(rng), 3)), index=rng
        )

        result = ts + ts[::2]
        expected = ts + ts
        expected.iloc[1::2] = np.nan
        tm.assert_frame_equal(result, expected)

        half = ts[::2]
        result = ts + half.take(np.random.default_rng(2).permutation(len(half)))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "op", [operator.add, operator.sub, operator.mul, operator.truediv]
    )
    def test_operators_none_as_na(self, op):
        # [FUNCTION DOC]
        # Parametrized over add, sub, mul, truediv. An object frame with None values: operations with a scalar, with itself and with a filled version treat None as NaN and give object-dtype results equal to the filled computation with missing entries set to NaN.
        df = DataFrame(
            {"col1": [2, 5.0, 123, None], "col2": [1, 2, 3, 4]}, dtype=object
        )

        # since filling converts dtypes from object, changed expected to be
        # object

        filled = df.fillna(np.nan)
        result = op(df, 3)
        expected = op(filled, 3).astype(object)
        expected[pd.isna(expected)] = np.nan
        tm.assert_frame_equal(result, expected)

        result = op(df, df)
        expected = op(filled, filled).astype(object)
        expected[pd.isna(expected)] = np.nan
        tm.assert_frame_equal(result, expected)

        result = op(df, df.fillna(7))
        tm.assert_frame_equal(result, expected)

        result = op(df.fillna(7), df)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("op,res", [("__eq__", False), ("__ne__", True)])
    # TODO: not sure what's correct here.
    @pytest.mark.filterwarnings("ignore:elementwise:FutureWarning")
    def test_logical_typeerror_with_non_valid(self, op, res, float_frame):
        # [FUNCTION DOC]
        # Parametrized over equality (all false) and inequality (all true). Comparing a float frame to a string gives a uniform result; element-wise future warnings are ignored.
        # KNOWN ISSUE: the code carries a TODO saying the correct behaviour here is not certain.
        # we are comparing floats vs a string
        result = getattr(float_frame, op)("foo")
        assert bool(result.all().all()) is res

    @pytest.mark.parametrize("op", ["add", "sub", "mul", "div", "truediv"])
    def test_binary_ops_align(self, op):
        # [FUNCTION DOC]
        # Regression test (GH 6681). Parametrized over add, sub, mul, div, truediv. A 27-row three-level frame combined with a series along rows joining on level third, and then on level second, equals the concatenation of per-label operations with the standard operator, sorted.
        # KNOWN ISSUE: the lookup for the standard operator by the name div returns nothing in Python 3 and the test then returns early, so that parameter case verifies nothing.
        # test aligning binary ops

        # GH 6681
        index = MultiIndex.from_product(
            [list("abc"), ["one", "two", "three"], [1, 2, 3]],
            names=["first", "second", "third"],
        )

        df = DataFrame(
            np.arange(27 * 3).reshape(27, 3),
            index=index,
            columns=["value1", "value2", "value3"],
        ).sort_index()

        idx = pd.IndexSlice
        opa = getattr(operator, op, None)
        if opa is None:
            return

        x = Series([1.0, 10.0, 100.0], [1, 2, 3])
        result = getattr(df, op)(x, level="third", axis=0)

        expected = pd.concat(
            [opa(df.loc[idx[:, :, i], :], v) for i, v in x.items()]
        ).sort_index()
        tm.assert_frame_equal(result, expected)

        x = Series([1.0, 10.0], ["two", "three"])
        result = getattr(df, op)(x, level="second", axis=0)

        expected = (
            pd.concat([opa(df.loc[idx[:, i], :], v) for i, v in x.items()])
            .reindex_like(df)
            .sort_index()
        )
        tm.assert_frame_equal(result, expected)

    def test_binary_ops_align_series_dataframe(self):
        # [FUNCTION DOC]
        # Regression test (GH 9463). Multiplying a frame with two-level columns by a series along axis 1 joining on level number or level name (with and without names on the frame and series) gives the series values broadcast across the matching level; frame's level names are kept when present.
        # GH9463 (alignment level of dataframe with series)

        midx = MultiIndex.from_product([["A", "B"], ["a", "b"]])
        df = DataFrame(np.ones((2, 4), dtype="int64"), columns=midx)
        s = Series({"a": 1, "b": 2})

        df2 = df.copy()
        df2.columns.names = ["lvl0", "lvl1"]
        s2 = s.copy()
        s2.index.name = "lvl1"

        # different cases of integer/string level names:
        res1 = df.mul(s, axis=1, level=1)
        res2 = df.mul(s2, axis=1, level=1)
        res3 = df2.mul(s, axis=1, level=1)
        res4 = df2.mul(s2, axis=1, level=1)
        res5 = df2.mul(s, axis=1, level="lvl1")
        res6 = df2.mul(s2, axis=1, level="lvl1")

        exp = DataFrame(
            np.array([[1, 2, 1, 2], [1, 2, 1, 2]], dtype="int64"), columns=midx
        )

        for res in [res1, res2]:
            tm.assert_frame_equal(res, exp)

        exp.columns.names = ["lvl0", "lvl1"]
        for res in [res3, res4, res5, res6]:
            tm.assert_frame_equal(res, exp)

    def test_add_with_dti_mismatched_tzs(self):
        # [FUNCTION DOC]
        # Adding frames with datetime indexes in different zones aligns on instants over the union (UTC result) with NaN for non-overlapping rows.
        base = pd.DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03"], tz="UTC")
        idx1 = base.tz_convert("Asia/Tokyo")[:2]
        idx2 = base.tz_convert("US/Eastern")[1:]

        df1 = DataFrame({"A": [1, 2]}, index=idx1)
        df2 = DataFrame({"A": [1, 1]}, index=idx2)
        exp = DataFrame({"A": [np.nan, 3, np.nan]}, index=base)
        tm.assert_frame_equal(df1 + df2, exp)

    def test_combineFrame(self, float_frame, mixed_float_frame, mixed_int_frame):
        # [FUNCTION DOC]
        # Frame-with-frame addition: overlapping rows doubled, missing rows and removed or NaN-filled columns give NaN; adding an empty frame in either order gives all NaN and two empty frames stay empty; reversed columns align; mixed float and integer frames upcast as expected (checked with the dtype helper).
        # KNOWN ISSUE: there is a commented-out assertion left in the middle, and the assertion on the all-NaN D column is repeated.
        frame_copy = float_frame.reindex(float_frame.index[::2])

        del frame_copy["D"]
        # adding NAs to first 5 values of column "C"
        frame_copy.loc[: frame_copy.index[4], "C"] = np.nan

        added = float_frame + frame_copy

        indexer = added["A"].dropna().index
        exp = (float_frame["A"] * 2).copy()

        tm.assert_series_equal(added["A"].dropna(), exp.loc[indexer])

        exp.loc[~exp.index.isin(indexer)] = np.nan
        tm.assert_series_equal(added["A"], exp.loc[added["A"].index])

        assert np.isnan(added["C"].reindex(frame_copy.index)[:5]).all()

        # assert(False)

        assert np.isnan(added["D"]).all()

        self_added = float_frame + float_frame
        tm.assert_index_equal(self_added.index, float_frame.index)

        added_rev = frame_copy + float_frame
        assert np.isnan(added["D"]).all()
        assert np.isnan(added_rev["D"]).all()

        # corner cases

        # empty
        plus_empty = float_frame + DataFrame()
        assert np.isnan(plus_empty.values).all()

        empty_plus = DataFrame() + float_frame
        assert np.isnan(empty_plus.values).all()

        empty_empty = DataFrame() + DataFrame()
        assert empty_empty.empty

        # out of order
        reverse = float_frame.reindex(columns=float_frame.columns[::-1])

        tm.assert_frame_equal(reverse + float_frame, float_frame * 2)

        # mix vs float64, upcast
        added = float_frame + mixed_float_frame
        _check_mixed_float(added, dtype="float64")
        added = mixed_float_frame + float_frame
        _check_mixed_float(added, dtype="float64")

        # mix vs mix
        added = mixed_float_frame + mixed_float_frame
        _check_mixed_float(added, dtype={"C": None})

        # with int
        added = float_frame + mixed_int_frame
        _check_mixed_float(added, dtype="float64")

    def test_combine_series(self, float_frame, mixed_float_frame, mixed_int_frame):
        # [FUNCTION DOC]
        # Frame plus a row series: each column equals column plus the matching value; extra series labels add an all-NaN column; mixed float frame plus a float series keeps the series dtype; adding downcast float series to the mixed float frame, and scaled integer series to the mixed integer frame, give the expected per-column dtypes.
        # Series
        series = float_frame.xs(float_frame.index[0])

        added = float_frame + series

        for key, s in added.items():
            tm.assert_series_equal(s, float_frame[key] + series[key])

        larger_series = series.to_dict()
        larger_series["E"] = 1
        larger_series = Series(larger_series)
        larger_added = float_frame + larger_series

        for key, s in float_frame.items():
            tm.assert_series_equal(larger_added[key], s + series[key])
        assert "E" in larger_added
        assert np.isnan(larger_added["E"]).all()

        # no upcast needed
        added = mixed_float_frame + series
        assert np.all(added.dtypes == series.dtype)

        # vs mix (upcast) as needed
        added = mixed_float_frame + series.astype("float32")
        _check_mixed_float(added, dtype={"C": None})
        added = mixed_float_frame + series.astype("float16")
        _check_mixed_float(added, dtype={"C": None})

        # these used to raise with numexpr as we are adding an int64 to an
        #  uint64....weird vs int
        added = mixed_int_frame + (100 * series).astype("int64")
        _check_mixed_int(
            added, dtype={"A": "int64", "B": "float64", "C": "int64", "D": "int64"}
        )
        added = mixed_int_frame + (100 * series).astype("int32")
        _check_mixed_int(
            added, dtype={"A": "int32", "B": "float64", "C": "int32", "D": "int64"}
        )

    def test_combine_timeseries(self, datetime_frame):
        # [FUNCTION DOC]
        # Adding a datetime-indexed series along the index: per column sums, names (column name kept in the frame, result name none unless it matches), and alignment with shorter frame/series, zero-length series and frames giving all NaN, and an empty-column frame keeping its rows.
        # TimeSeries
        ts = datetime_frame["A"]

        # 10890
        # we no longer allow auto timeseries broadcasting
        # and require explicit broadcasting
        added = datetime_frame.add(ts, axis="index")

        for key, col in datetime_frame.items():
            result = col + ts
            tm.assert_series_equal(added[key], result, check_names=False)
            assert added[key].name == key
            if col.name == ts.name:
                assert result.name == "A"
            else:
                assert result.name is None

        smaller_frame = datetime_frame[:-5]
        smaller_added = smaller_frame.add(ts, axis="index")

        tm.assert_index_equal(smaller_added.index, datetime_frame.index)

        smaller_ts = ts[:-5]
        smaller_added2 = datetime_frame.add(smaller_ts, axis="index")
        tm.assert_frame_equal(smaller_added, smaller_added2)

        # length 0, result is all-nan
        result = datetime_frame.add(ts[:0], axis="index")
        expected = DataFrame(
            np.nan, index=datetime_frame.index, columns=datetime_frame.columns
        )
        tm.assert_frame_equal(result, expected)

        # Frame is all-nan
        result = datetime_frame[:0].add(ts, axis="index")
        expected = DataFrame(
            np.nan, index=datetime_frame.index, columns=datetime_frame.columns
        )
        tm.assert_frame_equal(result, expected)

        # empty but with non-empty index
        frame = datetime_frame[:1].reindex(columns=[])
        result = frame.mul(ts, axis="index")
        assert len(result) == len(ts)

    def test_combineFunc(self, float_frame, mixed_float_frame):
        # [FUNCTION DOC]
        # Multiplying a frame by two multiplies all values (also for the mixed float frame, dtypes checked); an empty frame times two stays empty.
        result = float_frame * 2
        tm.assert_numpy_array_equal(result.values, float_frame.values * 2)

        # vs mix
        result = mixed_float_frame * 2
        for c, s in result.items():
            tm.assert_numpy_array_equal(s.values, mixed_float_frame[c].values * 2)
        _check_mixed_float(result, dtype={"C": None})

        result = DataFrame() * 2
        assert result.index.equals(DataFrame().index)
        assert len(result.columns) == 0

    @pytest.mark.parametrize(
        "func",
        [operator.eq, operator.ne, operator.lt, operator.gt, operator.ge, operator.le],
    )
    def test_comparisons(self, simple_frame, float_frame, func):
        # [FUNCTION DOC]
        # Parametrized over six comparison operators. Comparing two equal-labeled frames, a frame with a row series, and a frame with a scalar match the raw numpy comparison; a five-dimensional operand raises a value error naming the dimension limit; comparing frames with different labels raises a value error that only identically-labeled frames can be compared.
        df1 = DataFrame(
            np.random.default_rng(2).standard_normal((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=pd.date_range("2000-01-01", periods=30, freq="B"),
        )
        df2 = df1.copy()

        row = simple_frame.xs("a")
        ndim_5 = np.ones((*df1.shape, 1, 1, 1))

        result = func(df1, df2)
        tm.assert_numpy_array_equal(result.values, func(df1.values, df2.values))

        msg = (
            "Unable to coerce to Series/DataFrame, "
            "dimension must be <= 2: (30, 4, 1, 1, 1)"
        )
        with pytest.raises(ValueError, match=re.escape(msg)):
            func(df1, ndim_5)

        result2 = func(simple_frame, row)
        tm.assert_numpy_array_equal(
            result2.values, func(simple_frame.values, row.values)
        )

        result3 = func(float_frame, 0)
        tm.assert_numpy_array_equal(result3.values, func(float_frame.values, 0))

        msg = (
            r"Can only compare identically-labeled \(both index and columns\) "
            "DataFrame objects"
        )
        with pytest.raises(ValueError, match=msg):
            func(simple_frame, simple_frame[:2])

    def test_strings_to_numbers_comparisons_raises(self, compare_operators_no_eq_ne):
        # [FUNCTION DOC]
        # Regression test (GH 11565). Parametrized over the ordering operators (no equality). Comparing a string frame to the integer zero raises a type error.
        # GH 11565
        df = DataFrame(
            {x: {"x": "foo", "y": "bar", "z": "baz"} for x in ["a", "b", "c"]}
        )

        f = getattr(operator, compare_operators_no_eq_ne)
        msg = "|".join(
            [
                "'[<>]=?' not supported between instances of 'str' and 'int'",
                "Invalid comparison between dtype=str and int",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            f(df, 0)

    def test_comparison_protected_from_errstate(self):
        # [FUNCTION DOC]
        # A comparison of a float frame with a missing value to zero is unaffected by numpy's error state: raising on invalid operations does not make the comparison fail.
        missing_df = DataFrame(
            np.ones((10, 4), dtype=np.float64),
            columns=Index(list("ABCD"), dtype=object),
        )
        missing_df.loc[missing_df.index[0], "A"] = np.nan
        with np.errstate(invalid="ignore"):
            expected = missing_df.values < 0
        with np.errstate(invalid="raise"):
            result = (missing_df < 0).values
        tm.assert_numpy_array_equal(result, expected)

    def test_boolean_comparison(self):
        # [FUNCTION DOC]
        # Regression test (GH 4576, GH 23000). Compares a three-by-two frame with a one-dimensional array, a row-shaped 2-D array (broadcasts like numpy), a column-shaped 2-D array (value error), a list and a tuple of the wrong length (value error) for greater-than and equality, also against raw numpy results.
        # KNOWN ISSUE: in the final alignment section, the expected frame's index and columns are updated but the expected frame is never compared; only the errors for list and tuple are checked.
        # GH 4576
        # boolean comparisons with a tuple/list give unexpected results
        df = DataFrame(np.arange(6).reshape((3, 2)))
        b = np.array([2, 2])
        b_r = np.atleast_2d([2, 2])
        b_c = b_r.T
        lst = [2, 2, 2]
        tup = tuple(lst)

        # gt
        expected = DataFrame([[False, False], [False, True], [True, True]])
        result = df > b
        tm.assert_frame_equal(result, expected)

        result = df.values > b
        tm.assert_numpy_array_equal(result, expected.values)

        msg1d = "Unable to coerce to Series, length must be 2: given 3"
        msg2d = "Unable to coerce to DataFrame, shape must be"
        msg2db = "operands could not be broadcast together with shapes"
        with pytest.raises(ValueError, match=msg1d):
            # wrong shape
            df > lst

        with pytest.raises(ValueError, match=msg1d):
            # wrong shape
            df > tup

        # broadcasts like ndarray (GH#23000)
        result = df > b_r
        tm.assert_frame_equal(result, expected)

        result = df.values > b_r
        tm.assert_numpy_array_equal(result, expected.values)

        with pytest.raises(ValueError, match=msg2d):
            df > b_c

        with pytest.raises(ValueError, match=msg2db):
            df.values > b_c

        # ==
        expected = DataFrame([[False, False], [True, False], [False, False]])
        result = df == b
        tm.assert_frame_equal(result, expected)

        with pytest.raises(ValueError, match=msg1d):
            df == lst

        with pytest.raises(ValueError, match=msg1d):
            df == tup

        # broadcasts like ndarray (GH#23000)
        result = df == b_r
        tm.assert_frame_equal(result, expected)

        result = df.values == b_r
        tm.assert_numpy_array_equal(result, expected.values)

        with pytest.raises(ValueError, match=msg2d):
            df == b_c

        assert df.values.shape != b_c.shape

        # with alignment
        df = DataFrame(
            np.arange(6).reshape((3, 2)), columns=list("AB"), index=list("abc")
        )
        expected.index = df.index
        expected.columns = df.columns

        with pytest.raises(ValueError, match=msg1d):
            df == lst

        with pytest.raises(ValueError, match=msg1d):
            df == tup

    def test_inplace_ops_alignment(self):
        # [FUNCTION DOC]
        # Regression test (GH 8511). In-place add and subtract with a frame covering a subset of columns (given in a different order) give the same result as the non-in-place version and as using a pre-aligned operand.
        # inplace ops / ops alignment
        # GH 8511

        columns = list("abcdefg")
        X_orig = DataFrame(
            np.arange(10 * len(columns)).reshape(-1, len(columns)),
            columns=columns,
            index=range(10),
        )
        Z = 100 * X_orig.iloc[:, 1:-1].copy()
        block1 = list("bedcf")
        subs = list("bcdef")

        # add
        X = X_orig.copy()
        result1 = (X[block1] + Z).reindex(columns=subs)

        X[block1] += Z
        result2 = X.reindex(columns=subs)

        X = X_orig.copy()
        result3 = (X[block1] + Z[block1]).reindex(columns=subs)

        X[block1] += Z[block1]
        result4 = X.reindex(columns=subs)

        tm.assert_frame_equal(result1, result2)
        tm.assert_frame_equal(result1, result3)
        tm.assert_frame_equal(result1, result4)

        # sub
        X = X_orig.copy()
        result1 = (X[block1] - Z).reindex(columns=subs)

        X[block1] -= Z
        result2 = X.reindex(columns=subs)

        X = X_orig.copy()
        result3 = (X[block1] - Z[block1]).reindex(columns=subs)

        X[block1] -= Z[block1]
        result4 = X.reindex(columns=subs)

        tm.assert_frame_equal(result1, result2)
        tm.assert_frame_equal(result1, result3)
        tm.assert_frame_equal(result1, result4)

    def test_inplace_ops_identity(self):
        # [FUNCTION DOC]
        # Regression test (GH 5104). In-place add on series and frames modifies the object (identity and internal manager retained) with integer and float scalars; for a mixed-dtype frame adding to a single column via item assignment updates the frame seen through another reference.
        # GH 5104
        # make sure that we are actually changing the object
        s_orig = Series([1, 2, 3])
        df_orig = DataFrame(
            np.random.default_rng(2).integers(0, 5, size=10).reshape(-1, 5)
        )

        # no dtype change
        s = s_orig.copy()
        s2 = s
        s += 1
        tm.assert_series_equal(s, s2)
        tm.assert_series_equal(s_orig + 1, s)
        assert s is s2
        assert s._mgr is s2._mgr

        df = df_orig.copy()
        df2 = df
        df += 1
        tm.assert_frame_equal(df, df2)
        tm.assert_frame_equal(df_orig + 1, df)
        assert df is df2
        assert df._mgr is df2._mgr

        # dtype change
        s = s_orig.copy()
        s2 = s
        s += 1.5
        tm.assert_series_equal(s, s2)
        tm.assert_series_equal(s_orig + 1.5, s)

        df = df_orig.copy()
        df2 = df
        df += 1.5
        tm.assert_frame_equal(df, df2)
        tm.assert_frame_equal(df_orig + 1.5, df)
        assert df is df2
        assert df._mgr is df2._mgr

        # mixed dtype
        arr = np.random.default_rng(2).integers(0, 10, size=5)
        df_orig = DataFrame({"A": arr.copy(), "B": "foo"})
        df = df_orig.copy()
        df2 = df
        df["A"] += 1
        expected = DataFrame({"A": arr.copy() + 1, "B": "foo"})
        tm.assert_frame_equal(df, expected)
        tm.assert_frame_equal(df2, expected)
        assert df._mgr is df2._mgr

        df = df_orig.copy()
        df2 = df
        df["A"] += 1.5
        expected = DataFrame({"A": arr.copy() + 1.5, "B": "foo"})
        tm.assert_frame_equal(df, expected)
        tm.assert_frame_equal(df2, expected)
        assert df._mgr is df2._mgr

    @pytest.mark.parametrize(
        "op",
        [
            "add",
            "and",
            pytest.param(
                "div",
                marks=pytest.mark.xfail(
                    raises=AttributeError, reason="__idiv__ not implemented"
                ),
            ),
            "floordiv",
            "mod",
            "mul",
            "or",
            "pow",
            "sub",
            "truediv",
            "xor",
        ],
    )
    def test_inplace_ops_identity2(self, op):
        # [FUNCTION DOC]
        # Parametrized over add, and, div (expected failure with attribute error as in-place true division by that name is not implemented), floordiv, mod, mul, or, pow, sub, truediv, xor. Calling the in-place dunder returns the same result as the non-in-place dunder on a copy; for boolean ops the float column is replaced by booleans first.
        # KNOWN ISSUE: the identity check at the end compares the identity of the frame with a value taken from the same frame just before, which is always true, so object identity is not actually verified.
        df = DataFrame({"a": [1.0, 2.0, 3.0], "b": [1, 2, 3]})

        operand = 2
        if op in ("and", "or", "xor"):
            # cannot use floats for boolean ops
            df["a"] = [True, False, True]

        df_copy = df.copy()
        iop = f"__i{op}__"
        op = f"__{op}__"

        # no id change and value is correct
        getattr(df, iop)(operand)
        expected = getattr(df_copy, op)(operand)
        tm.assert_frame_equal(df, expected)
        expected = id(df)
        assert id(df) == expected

    @pytest.mark.parametrize(
        "val",
        [
            [1, 2, 3],
            (1, 2, 3),
            np.array([1, 2, 3], dtype=np.int64),
            range(1, 4),
        ],
    )
    def test_alignment_non_pandas(self, val):
        # [FUNCTION DOC]
        # Parametrized over list, tuple, integer array and range. The private alignment-for-operation helper broadcasts a one-dimensional operand either down the index (axis 0) or across the columns (axis 1), producing the expected frame as the second returned element.
        index = ["A", "B", "C"]
        columns = ["X", "Y", "Z"]
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            index=index,
            columns=columns,
        )

        align = DataFrame._align_for_op

        expected = DataFrame({"X": val, "Y": val, "Z": val}, index=df.index)
        tm.assert_frame_equal(align(df, val, axis=0)[1], expected)

        expected = DataFrame(
            {"X": [1, 1, 1], "Y": [2, 2, 2], "Z": [3, 3, 3]}, index=df.index
        )
        tm.assert_frame_equal(align(df, val, axis=1)[1], expected)

    @pytest.mark.parametrize("val", [[1, 2], (1, 2), np.array([1, 2]), range(1, 3)])
    def test_alignment_non_pandas_length_mismatch(self, val):
        # [FUNCTION DOC]
        # Parametrized over four length-two operands. The private alignment helper raises a value error that the length must be 3 along either axis.
        index = ["A", "B", "C"]
        columns = ["X", "Y", "Z"]
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            index=index,
            columns=columns,
        )

        align = DataFrame._align_for_op
        # length mismatch
        msg = "Unable to coerce to Series, length must be 3: given 2"
        with pytest.raises(ValueError, match=msg):
            align(df, val, axis=0)

        with pytest.raises(ValueError, match=msg):
            align(df, val, axis=1)

    def test_alignment_non_pandas_index_columns(self):
        # [FUNCTION DOC]
        # The private alignment helper with a same-shaped two-dimensional array returns a frame with the frame's labels; wrong two-dimensional shape raises a value error about shape; a three-dimensional array raises a value error stating the dimension limit.
        index = ["A", "B", "C"]
        columns = ["X", "Y", "Z"]
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            index=index,
            columns=columns,
        )

        align = DataFrame._align_for_op
        val = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        tm.assert_frame_equal(
            align(df, val, axis=0)[1],
            DataFrame(val, index=df.index, columns=df.columns),
        )
        tm.assert_frame_equal(
            align(df, val, axis=1)[1],
            DataFrame(val, index=df.index, columns=df.columns),
        )

        # shape mismatch
        msg = "Unable to coerce to DataFrame, shape must be"
        val = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError, match=msg):
            align(df, val, axis=0)

        with pytest.raises(ValueError, match=msg):
            align(df, val, axis=1)

        val = np.zeros((3, 3, 3))
        msg = re.escape(
            "Unable to coerce to Series/DataFrame, dimension must be <= 2: (3, 3, 3)"
        )
        with pytest.raises(ValueError, match=msg):
            align(df, val, axis=0)
        with pytest.raises(ValueError, match=msg):
            align(df, val, axis=1)

    def test_no_warning(self, all_arithmetic_operators):
        # [FUNCTION DOC]
        # Every arithmetic dunder with a series containing a missing value emits no warning.
        df = DataFrame({"A": [0.0, 0.0], "B": [0.0, None]})
        b = df["B"]
        with tm.assert_produces_warning(None):
            getattr(df, all_arithmetic_operators)(b)

    def test_dunder_methods_binary(self, all_arithmetic_operators):
        # [FUNCTION DOC]
        # Calling any arithmetic dunder with two arguments raises a type error that it takes two positional arguments.
        # KNOWN ISSUE: the comment refers to the issue as unknown placeholder question marks, so the originating issue is not recorded.
        # GH#??? frame.__foo__ should only accept one argument
        df = DataFrame({"A": [0.0, 0.0], "B": [0.0, None]})
        b = df["B"]
        with pytest.raises(TypeError, match="takes 2 positional arguments"):
            getattr(df, all_arithmetic_operators)(b, 0)

    def test_align_int_fill_bug(self):
        # [FUNCTION DOC]
        # Regression test (GH 910). Subtracting the column means from a float frame that has an extra integer column cast to float gives the same result as for the all-float frame.
        # GH#910
        X = np.arange(10 * 10, dtype="float64").reshape(10, 10)
        Y = np.ones((10, 1), dtype=int)

        df1 = DataFrame(X)
        df1["0.X"] = Y.squeeze()

        df2 = df1.astype(float)

        result = df1 - df1.mean()
        expected = df2 - df2.mean()
        tm.assert_frame_equal(result, expected)


def test_pow_with_realignment():
    # [FUNCTION DOC]
    # Regression test (GH 32685). Raising a frame to the power of an empty-column frame with matching index gives NaN for base values other than one and one for base one (power has special semantics for missing).
    # GH#32685 pow has special semantics for operating with null values
    left = DataFrame({"A": [0, 1, 2]})
    right = DataFrame(index=[0, 1, 2])

    result = left**right
    expected = DataFrame({"A": [np.nan, 1.0, np.nan]})
    tm.assert_frame_equal(result, expected)


def test_dataframe_series_extension_dtypes():
    # [FUNCTION DOC]
    # Regression test for issue 34311. Adding a nullable integer series along columns to a nullable integer frame gives the same result as adding the raw integer arrays and casting to nullable integer.
    # https://github.com/pandas-dev/pandas/issues/34311
    df = DataFrame(
        np.random.default_rng(2).integers(0, 100, (10, 3)), columns=["a", "b", "c"]
    )
    ser = Series([1, 2, 3], index=["a", "b", "c"])

    expected = df.to_numpy("int64") + ser.to_numpy("int64").reshape(-1, 3)
    expected = DataFrame(expected, columns=df.columns, dtype="Int64")

    df_ea = df.astype("Int64")
    result = df_ea + ser
    tm.assert_frame_equal(result, expected)
    result = df_ea + ser.astype("Int64")
    tm.assert_frame_equal(result, expected)


def test_dataframe_blockwise_slicelike():
    # [FUNCTION DOC]
    # Regression test (GH 34367). Adds pairs of frames whose NaN values lie in different column blocks (some columns cast to float so that NaN can be set); the result equals column-by-column addition.
    # GH#34367
    arr = np.random.default_rng(2).integers(0, 1000, (100, 10))
    df1 = DataFrame(arr)
    # Explicit cast to float to avoid implicit cast when setting nan
    df2 = df1.copy().astype({1: "float", 3: "float", 7: "float"})
    df2.iloc[0, [1, 3, 7]] = np.nan

    # Explicit cast to float to avoid implicit cast when setting nan
    df3 = df1.copy().astype({5: "float"})
    df3.iloc[0, [5]] = np.nan

    # Explicit cast to float to avoid implicit cast when setting nan
    df4 = df1.copy().astype({2: "float", 3: "float", 4: "float"})
    df4.iloc[0, np.arange(2, 5)] = np.nan
    # Explicit cast to float to avoid implicit cast when setting nan
    df5 = df1.copy().astype({4: "float", 5: "float", 6: "float"})
    df5.iloc[0, np.arange(4, 7)] = np.nan

    for left, right in [(df1, df2), (df2, df3), (df4, df5)]:
        res = left + right

        expected = DataFrame({i: left[i] + right[i] for i in left.columns})
        tm.assert_frame_equal(res, expected)


@pytest.mark.parametrize(
    "df, col_dtype",
    [
        (DataFrame([[1.0, 2.0], [4.0, 5.0]], columns=list("ab")), "float64"),
        (
            DataFrame([[1.0, "b"], [4.0, "b"]], columns=list("ab")).astype(
                {"b": object}
            ),
            "object",
        ),
    ],
)
def test_dataframe_operation_with_non_numeric_types(df, col_dtype):
    # [FUNCTION DOC]
    # Regression test (GH 22663). Parametrized over a float frame and a frame whose second column is object. Adding a one-element series labeled a gives a for column a and NaN in column b, preserving the column dtype.
    # GH #22663
    expected = DataFrame([[0.0, np.nan], [3.0, np.nan]], columns=list("ab"))
    expected = expected.astype({"b": col_dtype})
    result = df + Series([-1.0], index=list("a"))
    tm.assert_frame_equal(result, expected)


def test_arith_reindex_with_duplicates():
    # [FUNCTION DOC]
    # Regression test for issue 35194. Adding a frame with a unique column to one with a duplicated column label gives NaN for the new column and the sum for the duplicated labels.
    # https://github.com/pandas-dev/pandas/issues/35194
    df1 = DataFrame(data=[[0]], columns=["second"])
    df2 = DataFrame(data=[[0, 0, 0]], columns=["first", "second", "second"])
    result = df1 + df2
    expected = DataFrame([[np.nan, 0, 0]], columns=["first", "second", "second"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("to_add", [[Series([1, 1])], [Series([1, 1]), Series([1, 1])]])
def test_arith_list_of_arraylike_raise(to_add):
    # [FUNCTION DOC]
    # Regression test (GH 36702). Parametrized over a list of one and a list of two series. Adding such a list to a frame (either order) raises a value error saying it cannot coerce a list of that type to series or frame.
    # GH 36702. Raise when trying to add list of array-like to DataFrame
    df = DataFrame({"x": [1, 2], "y": [1, 2]})

    msg = f"Unable to coerce list of {type(to_add[0])} to Series/DataFrame"
    with pytest.raises(ValueError, match=msg):
        df + to_add
    with pytest.raises(ValueError, match=msg):
        to_add + df


def test_inplace_arithmetic_series_update():
    # [FUNCTION DOC]
    # Regression test for issue 36373. In-place addition on a series taken from a frame replaces the series values without modifying the frame.
    # https://github.com/pandas-dev/pandas/issues/36373
    df = DataFrame({"A": [1, 2, 3]})
    df_orig = df.copy()
    series = df["A"]
    vals = series._values

    series += 1
    assert series._values is not vals
    tm.assert_frame_equal(df, df_orig)


def test_arithmetic_multiindex_align():
    # [FUNCTION DOC]
    # Regression test for issue 33765. Subtracting a frame with a one-level named column index from one with a two-level index aligns on the shared level and gives zero.
    """
    Regression test for: https://github.com/pandas-dev/pandas/issues/33765
    """
    df1 = DataFrame(
        [[1]],
        index=["a"],
        columns=MultiIndex.from_product([[0], [1]], names=["a", "b"]),
    )
    df2 = DataFrame([[1]], index=["a"], columns=Index([0], name="a"))
    expected = DataFrame(
        [[0]],
        index=["a"],
        columns=MultiIndex.from_product([[0], [1]], names=["a", "b"]),
    )
    result = df1 - df2
    tm.assert_frame_equal(result, expected)


def test_arithmetic_multiindex_column_align():
    # [FUNCTION DOC]
    # Regression test (GH 60498). Multiplying a frame with two-level columns by a frame with matching first-level columns broadcasts over the second level.
    # GH#60498
    df1 = DataFrame(
        data=100,
        columns=MultiIndex.from_product(
            [["1A", "1B"], ["2A", "2B"]], names=["Lev1", "Lev2"]
        ),
        index=["C1", "C2"],
    )
    df2 = DataFrame(
        data=np.array([[0.1, 0.25], [0.2, 0.45]]),
        columns=MultiIndex.from_product([["1A", "1B"]], names=["Lev1"]),
        index=["C1", "C2"],
    )
    expected = DataFrame(
        data=np.array([[10.0, 10.0, 25.0, 25.0], [20.0, 20.0, 45.0, 45.0]]),
        columns=MultiIndex.from_product(
            [["1A", "1B"], ["2A", "2B"]], names=["Lev1", "Lev2"]
        ),
        index=["C1", "C2"],
    )
    result = df1 * df2
    tm.assert_frame_equal(result, expected)


def test_arithmetic_multiindex_column_align_with_fillvalue():
    # [FUNCTION DOC]
    # Regression test (GH 60903). The add method with fill value zero on frames with disjoint two-level columns returns the union of columns, with each value unchanged.
    # GH#60903
    df1 = DataFrame(
        data=[[1.0, 2.0]],
        columns=MultiIndex.from_tuples([("A", "one"), ("A", "two")]),
    )
    df2 = DataFrame(
        data=[[3.0, 4.0]],
        columns=MultiIndex.from_tuples([("B", "one"), ("B", "two")]),
    )
    expected = DataFrame(
        data=[[1.0, 2.0, 3.0, 4.0]],
        columns=MultiIndex.from_tuples(
            [("A", "one"), ("A", "two"), ("B", "one"), ("B", "two")]
        ),
    )
    result = df1.add(df2, fill_value=0)
    tm.assert_frame_equal(result, expected)


def test_bool_frame_mult_float():
    # [FUNCTION DOC]
    # Regression test (GH 18549). Multiplying a boolean frame by float one gives a float frame of ones.
    # GH 18549
    df = DataFrame(True, list("ab"), list("cd"))
    result = df * 1.0
    expected = DataFrame(np.ones((2, 2)), list("ab"), list("cd"))
    tm.assert_frame_equal(result, expected)


def test_frame_sub_nullable_int(any_int_ea_dtype):
    # [FUNCTION DOC]
    # Regression test (GH 32822). Parametrized by any nullable integer dtype. Subtracting one-column frames keeps the dtype with the missing value remaining missing.
    # GH 32822
    series1 = Series([1, 2, None], dtype=any_int_ea_dtype)
    series2 = Series([1, 2, 3], dtype=any_int_ea_dtype)
    expected = DataFrame([0, 0, None], dtype=any_int_ea_dtype)
    result = series1.to_frame() - series2.to_frame()
    tm.assert_frame_equal(result, expected)


@pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager|Passing a SingleBlockManager:DeprecationWarning"
)
def test_frame_op_subclass_nonclass_constructor():
    # [FUNCTION DOC]
    # Regression test (GH 43201). Defines a frame subclass whose constructor property returns a partial function (not the class) carrying extra data, and a series subclass. Multiplication by a scalar and addition of two such frames return subclass instances with the extra data kept. Some deprecation warnings about passing a block manager are ignored.
    # GH#43201 subclass._constructor is a function, not the subclass itself

    class SubclassedSeries(Series):
        # [CLASS DOC]
        # HELPER: series subclass for the subclass-constructor test; its constructor properties return itself for same-dimension results and the frame subclass for expansion.
        @property
        def _constructor(self):
            # [FUNCTION DOC]
            # HELPER: property returning the series subclass itself.
            return SubclassedSeries

        @property
        def _constructor_expanddim(self):
            # [FUNCTION DOC]
            # HELPER: property returning the frame subclass defined later in the enclosing test.
            return SubclassedDataFrame

    class SubclassedDataFrame(DataFrame):
        # [CLASS DOC]
        # HELPER: frame subclass that carries an extra attribute, listed in its metadata so it propagates; used to check arithmetic keeps subclass state when the constructor is a function returning a partially bound class.
        _metadata = ["my_extra_data"]

        def __init__(self, my_extra_data, *args, **kwargs) -> None:
            # [FUNCTION DOC]
            # HELPER: takes the extra data value first, stores it, then forwards the remaining positional and keyword arguments to the parent frame constructor.
            self.my_extra_data = my_extra_data
            super().__init__(*args, **kwargs)

        @property
        def _constructor(self):
            # [FUNCTION DOC]
            # HELPER: property returning a partially applied constructor with the stored extra data pre-filled (a function-like object rather than a class).
            return functools.partial(type(self), self.my_extra_data)

        @property
        def _constructor_sliced(self):
            # [FUNCTION DOC]
            # HELPER: property returning the series subclass for one-dimensional results.
            return SubclassedSeries

    sdf = SubclassedDataFrame("some_data", {"A": [1, 2, 3], "B": [4, 5, 6]})
    result = sdf * 2
    expected = SubclassedDataFrame("some_data", {"A": [2, 4, 6], "B": [8, 10, 12]})
    tm.assert_frame_equal(result, expected)

    result = sdf + sdf
    tm.assert_frame_equal(result, expected)


def test_enum_column_equality():
    # [FUNCTION DOC]
    # Columns labeled by an enumeration member: comparing two such columns gives an all-true boolean series named by the member.
    Cols = Enum("Cols", "col1 col2")

    q1 = DataFrame({Cols.col1: [1, 2, 3]})
    q2 = DataFrame({Cols.col1: [1, 2, 3]})

    result = q1[Cols.col1] == q2[Cols.col1]
    expected = Series([True, True, True], name=Cols.col1)

    tm.assert_series_equal(result, expected)


def test_mixed_col_index_dtype(string_dtype_no_object):
    # [FUNCTION DOC]
    # Regression test (GH 47382). Parametrized over string dtypes. Adding frames where one has string-typed column labels aligns correctly, yielding the string-typed labels.
    # GH 47382
    df1 = DataFrame(columns=list("abc"), data=1.0, index=[0])
    df2 = DataFrame(columns=list("abc"), data=0.0, index=[0])
    df1.columns = df2.columns.astype(string_dtype_no_object)
    result = df1 + df2
    expected = DataFrame(columns=list("abc"), data=1.0, index=[0])

    expected.columns = expected.columns.astype(string_dtype_no_object)

    tm.assert_frame_equal(result, expected)


def test_sum_mixed_empty(any_string_dtype):
    # [FUNCTION DOC]
    # Regression test (GH 64657). Parametrized over string dtypes. Summing an empty frame with a string-typed column and an integer column gives an empty string for the true string dtype and zero otherwise (object dtype), and zero for the integer column; result is object dtype.
    # GH 64657
    # for actual string dtype, sum gives "", for object dtype we get 0
    expected = Series(
        {"col1": "" if any_string_dtype == "string" else 0, "col2": 0}, dtype=object
    )
    empty_df = DataFrame(columns=["col1", "col2"]).astype(
        {"col1": any_string_dtype, "col2": int}
    )
    result = empty_df.sum()
    tm.assert_series_equal(result, expected)
