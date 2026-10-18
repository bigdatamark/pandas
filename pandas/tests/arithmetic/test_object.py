# [MODULE DOC]
# Arithmetic and comparison tests for object-dtype data across series, index, and boxed containers that should behave identically.
# Contains a comparison test class, an arithmetic test class (string concatenation, invalid operations, Decimal subtraction, timedelta addition, mixed-timezone timestamps, in-place index operators), a small index subclass that counts operator calls, and a test that index operators defer to unknown subclasses. Uses the shared fixtures for comparison operator and box type.
# Arithmetic tests for DataFrame/Series/Index/Array classes that should
# behave identically.
# Specifically for object dtype
import datetime
from decimal import Decimal
import operator

import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    Series,
    Timestamp,
    option_context,
)
import pandas._testing as tm
from pandas.core import ops

# ------------------------------------------------------------------
# Comparisons


class TestObjectComparisons:
    # [CLASS DOC]
    # Test class for comparison operators on object-dtype series.
    def test_comparison_object_numeric_nas(self, comparison_op):
        # [FUNCTION DOC]
        # Uses the comparison-operator fixture. Verifies that comparing an object-dtype series of random floats (with NaNs introduced by a shift of two) to its shifted self gives the same result as comparing the float-cast versions.
        ser = Series(np.random.default_rng(2).standard_normal(10), dtype=object)
        shifted = ser.shift(2)

        func = comparison_op

        result = func(ser, shifted)
        expected = func(ser.astype(float), shifted.astype(float))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "infer_string", [False, pytest.param(True, marks=td.skip_if_no("pyarrow"))]
    )
    def test_object_comparisons(self, infer_string):
        # [FUNCTION DOC]
        # Parametrized over the future infer-string option being off or on (on requires the pyarrow optional dependency). Within that option context, verifies for a series of strings and NaN that equality with "a" is true only for matches, less-than with "a" is all false, and inequality equals the negation of equality.
        with option_context("future.infer_string", infer_string):
            ser = Series(["a", "b", np.nan, "c", "a"])

            result = ser == "a"
            expected = Series([True, False, False, False, True])
            tm.assert_series_equal(result, expected)

            result = ser < "a"
            expected = Series([False, False, False, False, False])
            tm.assert_series_equal(result, expected)

            result = ser != "a"
            expected = -(ser == "a")
            tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("dtype", [None, object])
    def test_more_na_comparisons(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over dtype none (inferred) and object. Verifies equality and inequality between two string series with NaN in the same position (NaN never equal), and equality or inequality with a NaN scalar gives all false or all true respectively.
        left = Series(["a", np.nan, "c"], dtype=dtype)
        right = Series(["a", np.nan, "d"], dtype=dtype)

        result = left == right
        expected = Series([True, False, False])
        tm.assert_series_equal(result, expected)

        result = left != right
        expected = Series([False, True, True])
        tm.assert_series_equal(result, expected)

        result = left == np.nan
        expected = Series([False, False, False])
        tm.assert_series_equal(result, expected)

        result = left != np.nan
        expected = Series([True, True, True])
        tm.assert_series_equal(result, expected)


# ------------------------------------------------------------------
# Arithmetic


class TestArithmetic:
    # [CLASS DOC]
    # Test class for arithmetic operators on object-dtype data.
    def test_add_period_to_array_of_offset(self):
        # [FUNCTION DOC]
        # Verifies adding a period scalar to an index of offsets (created by subtracting a period range from a period) in either order yields an object index of sums computed elementwise.
        # GH#50162
        per = pd.Period("2012-1-1", freq="D")
        pi = pd.period_range("2012-1-1", periods=10, freq="D")
        idx = per - pi

        expected = pd.Index([x + per for x in idx], dtype=object)
        result = idx + per
        tm.assert_index_equal(result, expected)

        result = per + idx
        tm.assert_index_equal(result, expected)

    # TODO: parametrize
    def test_pow_ops_object(self):
        # [FUNCTION DOC]
        # Regression test for a linked issue about power with missing values. Verifies power of two object-dtype series containing ones and NaN in both operand orders equals applying power to the underlying object arrays, kept as object dtype.
        # GH#22922
        # pow is weird with masking & 1, so testing here
        a = Series([1, np.nan, 1, np.nan], dtype=object)
        b = Series([1, np.nan, np.nan, 1], dtype=object)
        result = a**b
        expected = Series(a.values**b.values, dtype=object)
        tm.assert_series_equal(result, expected)

        result = b**a
        expected = Series(b.values**a.values, dtype=object)

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("op", [operator.add, ops.radd])
    @pytest.mark.parametrize("other", ["category", "Int64"])
    def test_add_extension_scalar(self, other, box_with_array, op):
        # [FUNCTION DOC]
        # Parametrized over addition and reflected addition (from the ops module), and over scalar strings "category" and "Int64" which are valid dtype names. Verifies that these string scalars are not mistaken for extension dtypes and dispatched to an extension array operation: the result of the operator on a boxed object series of strings equals the elementwise Python operator result, boxed likewise.
        # GH#22378
        # Check that scalars satisfying is_extension_array_dtype(obj)
        # do not incorrectly try to dispatch to an ExtensionArray operation

        arr = Series(["a", "b", "c"])
        expected = Series([op(x, other) for x in arr])

        arr = tm.box_expected(arr, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = op(arr, other)
        tm.assert_equal(result, expected)

    def test_objarr_add_str(self, box_with_array):
        # [FUNCTION DOC]
        # Uses the box fixture. Verifies adding a string to an object series of strings with a NaN concatenates the string on the right and keeps NaN.
        ser = Series(["x", np.nan, "x"])
        expected = Series(["xa", np.nan, "xa"])

        ser = tm.box_expected(ser, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = ser + "a"
        tm.assert_equal(result, expected)

    def test_objarr_radd_str(self, box_with_array):
        # [FUNCTION DOC]
        # Uses the box fixture. Verifies adding an object series to a string on the left (reflected add) prepends the string and keeps NaN.
        ser = Series(["x", np.nan, "x"])
        expected = Series(["ax", np.nan, "ax"])

        ser = tm.box_expected(ser, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = "a" + ser
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "data",
        [
            [1, 2, 3],
            [1.1, 2.2, 3.3],
            [Timestamp("2011-01-01"), Timestamp("2011-01-02"), pd.NaT],
            ["x", "y", 1],
        ],
    )
    @pytest.mark.parametrize("dtype", [None, object])
    def test_objarr_radd_str_invalid(self, dtype, data, box_with_array):
        # [FUNCTION DOC]
        # Parametrized over data lists (integers, floats, timestamps with NaT, and a mixed list) and dtype none or object, plus the box fixture. Verifies a string plus such a boxed series raises a type error matching any of several accepted messages.
        ser = Series(data, dtype=dtype)

        ser = tm.box_expected(ser, box_with_array)
        msg = "|".join(
            [
                "can only concatenate str",
                "did not contain a loop with signature matching types",
                "unsupported operand type",
                "must be str",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            "foo_" + ser

    @pytest.mark.parametrize("op", [operator.add, ops.radd, operator.sub, ops.rsub])
    def test_objarr_add_invalid(self, op, box_with_array):
        # [FUNCTION DOC]
        # Parametrized over addition, reflected addition, subtraction, and reflected subtraction, and the box fixture. Verifies each operator with an integer or a NumPy integer scalar against an object series of letters raises an exception matching any of several accepted messages.
        # invalid ops
        box = box_with_array

        obj_ser = Series(list("abc"), dtype=object, name="objects")

        obj_ser = tm.box_expected(obj_ser, box)
        msg = "|".join(
            [
                "can only concatenate str",
                "unsupported operand type",
                "must be str",
                "has no kernel",
            ]
        )
        with pytest.raises(Exception, match=msg):
            op(obj_ser, 1)
        with pytest.raises(Exception, match=msg):
            op(obj_ser, np.array(1, dtype=np.int64))

    # TODO: Moved from tests.series.test_operators; needs cleanup
    def test_operators_na_handling(self):
        # [FUNCTION DOC]
        # Moved from the series operator tests (a TODO says it needs cleanup). Verifies adding a string prefix or suffix to a string series with a NaN keeps NaN and concatenates for others.
        ser = Series(["foo", "bar", "baz", np.nan])
        result = "prefix_" + ser
        expected = Series(["prefix_foo", "prefix_bar", "prefix_baz", np.nan])
        tm.assert_series_equal(result, expected)

        result = ser + "_suffix"
        expected = Series(["foo_suffix", "bar_suffix", "baz_suffix", np.nan])
        tm.assert_series_equal(result, expected)

    # TODO: parametrize over box
    @pytest.mark.parametrize("dtype", [None, object])
    def test_series_with_dtype_radd_timedelta(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over dtype none (inferred timedelta) and object. Verifies adding a timedelta scalar to a series of timedeltas on either side gives a series of the shifted timedeltas with the same dtype (object is preserved when the input is object).
        # note this test is _not_ aimed at timedelta64-dtyped Series
        # as of 2.0 we retain object dtype when ser.dtype == object
        ser = Series(
            [pd.Timedelta("1 days"), pd.Timedelta("2 days"), pd.Timedelta("3 days")],
            dtype=dtype,
        )
        expected = Series(
            [pd.Timedelta("4 days"), pd.Timedelta("5 days"), pd.Timedelta("6 days")],
            dtype=dtype,
        )

        result = pd.Timedelta("3 days") + ser
        tm.assert_series_equal(result, expected)

        result = ser + pd.Timedelta("3 days")
        tm.assert_series_equal(result, expected)

    # TODO: cleanup & parametrize over box
    def test_mixed_timezone_series_ops_object(self):
        # [FUNCTION DOC]
        # Verifies arithmetic on object series that hold timestamps with different time zones (so they cannot form a single datetime dtype): adding a timedelta in either order shifts each element keeping its zone and name; subtracting two such series gives an object series of timedeltas and its negation for the reverse order; and an object series of timedeltas supports adding a timedelta in either order.
        # GH#13043
        ser = Series(
            [
                Timestamp("2015-01-01", tz="US/Eastern"),
                Timestamp("2015-01-01", tz="Asia/Tokyo"),
            ],
            name="xxx",
        )
        assert ser.dtype == object

        exp = Series(
            [
                Timestamp("2015-01-02", tz="US/Eastern"),
                Timestamp("2015-01-02", tz="Asia/Tokyo"),
            ],
            name="xxx",
        )
        tm.assert_series_equal(ser + pd.Timedelta("1 days"), exp)
        tm.assert_series_equal(pd.Timedelta("1 days") + ser, exp)

        # object series & object series
        ser2 = Series(
            [
                Timestamp("2015-01-03", tz="US/Eastern"),
                Timestamp("2015-01-05", tz="Asia/Tokyo"),
            ],
            name="xxx",
        )
        assert ser2.dtype == object
        exp = Series(
            [pd.Timedelta("2 days"), pd.Timedelta("4 days")], name="xxx", dtype=object
        )
        tm.assert_series_equal(ser2 - ser, exp)
        tm.assert_series_equal(ser - ser2, -exp)

        ser = Series(
            [pd.Timedelta("01:00:00"), pd.Timedelta("02:00:00")],
            name="xxx",
            dtype=object,
        )
        assert ser.dtype == object

        exp = Series(
            [pd.Timedelta("01:30:00"), pd.Timedelta("02:30:00")],
            name="xxx",
            dtype=object,
        )
        tm.assert_series_equal(ser + pd.Timedelta("00:30:00"), exp)
        tm.assert_series_equal(pd.Timedelta("00:30:00") + ser, exp)

    # TODO: cleanup & parametrize over box
    def test_iadd_preserves_name(self):
        # [FUNCTION DOC]
        # Verifies that in-place addition and subtraction on a series index preserve the index name.
        # GH#17067, GH#19723 __iadd__ and __isub__ should preserve index name
        ser = Series([1, 2, 3])
        ser.index.name = "foo"

        ser.index += 1
        assert ser.index.name == "foo"

        ser.index -= 1
        assert ser.index.name == "foo"

    def test_add_string(self):
        # [FUNCTION DOC]
        # Verifies that adding a string to an index of strings concatenates it to each element (checked by membership of the old and new values).
        # from bug report
        index = pd.Index(["a", "b", "c"])
        index2 = index + "foo"

        assert "a" not in index2
        assert "afoo" in index2

    def test_iadd_string(self):
        # [FUNCTION DOC]
        # Verifies in-place addition of a string to an index of strings rebinds it to the concatenated values (membership of the new value is true).
        index = pd.Index(["a", "b", "c"])
        # doesn't fail test unless there is a check before `+=`
        assert "a" in index

        index += "_x"
        assert "a_x" in index

    def test_add(self):
        # [FUNCTION DOC]
        # Verifies adding an index of strings to itself or to a list (either order) concatenates elementwise, and that adding a string on the right or left appends or prepends it.
        index = pd.Index([str(i) for i in range(10)])
        expected = pd.Index(index.values * 2)
        tm.assert_index_equal(index + index, expected)
        tm.assert_index_equal(index + index.tolist(), expected)
        tm.assert_index_equal(index.tolist() + index, expected)

        # test add and radd
        index = pd.Index(list("abc"))
        expected = pd.Index(["a1", "b1", "c1"])
        tm.assert_index_equal(index + "1", expected)
        expected = pd.Index(["1a", "1b", "1c"])
        tm.assert_index_equal("1" + index, expected)

    def test_sub_fail(self):
        # [FUNCTION DOC]
        # Verifies that subtracting from an index of strings (a string, itself, or a list in either order) raises a type error matching an unsupported-operand, broadcast, or sub-not-supported message.
        index = pd.Index([str(i) for i in range(10)])

        msg = "unsupported operand type|Cannot broadcast|sub' not supported"
        with pytest.raises(TypeError, match=msg):
            index - "a"
        with pytest.raises(TypeError, match=msg):
            index - index
        with pytest.raises(TypeError, match=msg):
            index - index.tolist()
        with pytest.raises(TypeError, match=msg):
            index.tolist() - index

    def test_sub_object(self):
        # [FUNCTION DOC]
        # Verifies subtraction on an index of Decimal objects: subtracting a Decimal or an index of Decimals gives the elementwise difference; subtracting a string or an object array with a string raises a type error about unsupported operand type.
        # GH#19369
        index = pd.Index([Decimal(1), Decimal(2)])
        expected = pd.Index([Decimal(0), Decimal(1)])

        result = index - Decimal(1)
        tm.assert_index_equal(result, expected)

        result = index - pd.Index([Decimal(1), Decimal(1)])
        tm.assert_index_equal(result, expected)

        msg = "unsupported operand type"
        with pytest.raises(TypeError, match=msg):
            index - "foo"

        with pytest.raises(TypeError, match=msg):
            index - np.array([2, "foo"], dtype=object)

    def test_rsub_object(self, fixed_now_ts):
        # [FUNCTION DOC]
        # Uses the fixed-now timestamp fixture. Verifies reflected subtraction with an index of Decimals: a Decimal minus the index or a NumPy array of Decimals minus the index gives the elementwise difference; a string or an array containing a boolean and a timestamp minus the index raises a type error about unsupported operand type.
        # GH#19369
        index = pd.Index([Decimal(1), Decimal(2)])
        expected = pd.Index([Decimal(1), Decimal(0)])

        result = Decimal(2) - index
        tm.assert_index_equal(result, expected)

        result = np.array([Decimal(2), Decimal(2)]) - index
        tm.assert_index_equal(result, expected)

        msg = "unsupported operand type"
        with pytest.raises(TypeError, match=msg):
            "foo" - index

        with pytest.raises(TypeError, match=msg):
            np.array([True, fixed_now_ts]) - index


class MyIndex(pd.Index):
    # [CLASS DOC]
    # Simple index subclass used to test that binary operators defer to subclasses. Class attribute: a call counter annotation. It overrides construction through the internal simple-new classmethod, and addition and reflected addition, which increment the counter.
    # Simple index subclass that tracks ops calls.

    _calls: int

    @classmethod
    def _simple_new(cls, values, name=None, dtype=None):
        # [FUNCTION DOC]
        # HELPER: internal constructor for the test subclass. Takes class, values, optional name and dtype (dtype is ignored); creates an instance bypassing normal initialization, stores the values as its data and the name, sets its call counter to zero, and resets its identity. Returns the new instance.
        result = object.__new__(cls)
        result._data = values
        result._name = name
        result._calls = 0
        result._reset_identity()

        return result

    def __add__(self, other):
        # [FUNCTION DOC]
        # HELPER: addition override for the test subclass; increments the call counter and returns a new instance with the same data (the operand is ignored).
        self._calls += 1
        return self._simple_new(self._data)

    def __radd__(self, other):
        # [FUNCTION DOC]
        # HELPER: reflected addition override; delegates to the addition override.
        return self.__add__(other)


@pytest.mark.parametrize(
    "other",
    [
        [datetime.timedelta(1), datetime.timedelta(2)],
        [datetime.datetime(2000, 1, 1), datetime.datetime(2000, 1, 2)],
        [pd.Period("2000"), pd.Period("2001")],
        ["a", "b"],
    ],
    ids=["timedelta", "datetime", "period", "object"],
)
def test_index_ops_defer_to_unknown_subclasses(other):
    # [FUNCTION DOC]
    # Regression test for a linked issue. Parametrized over four lists (timedeltas, datetimes, periods, strings). Verifies that adding a regular index of those values to an instance of the test subclass (on the right) lets the subclass handle the operation: the result is an instance of the subclass and its call counter is exactly one.
    # https://github.com/pandas-dev/pandas/issues/31109
    values = np.array(
        [datetime.date(2000, 1, 1), datetime.date(2000, 1, 2)], dtype=object
    )
    a = MyIndex._simple_new(values)
    other = pd.Index(other)
    result = other + a
    assert isinstance(result, MyIndex)
    assert a._calls == 1
