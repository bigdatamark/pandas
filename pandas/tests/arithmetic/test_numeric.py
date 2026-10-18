# [MODULE DOC]
# Arithmetic and comparison tests for numeric dtypes (integer, unsigned, float, range index, Decimal, nullable integer) across arrays, indexes, series, and frames that should behave identically.
# Covers comparisons, arithmetic with datetime-like scalars and arrays (multiplication and division by timedeltas, invalid additions), division/modulo/divmod by zero, multiplication and division, addition and subtraction, NumPy ufunc compatibility, object-dtype equivalence, and range index arithmetic.
# Defines module-level helpers and a list of left operands used for parametrization, and an autouse fixture that runs every test with both the numexpr and the plain Python evaluation paths. Importing the module builds the list of operand series and indexes (no other import-time side effects).
# Arithmetic tests for DataFrame/Series/Index/Array classes that should
# behave identically.
# Specifically for numeric dtypes
from __future__ import annotations

from collections import abc
from datetime import timedelta
from decimal import Decimal
import operator

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Index,
    RangeIndex,
    Series,
    Timedelta,
    TimedeltaIndex,
    array,
    date_range,
)
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.tests.arithmetic.common import (
    assert_invalid_addsub_type,
    assert_invalid_comparison,
)


@pytest.fixture(autouse=True, params=[0, 1000000], ids=["numexpr", "python"])
def switch_numexpr_min_elements(request, monkeypatch):
    # [FUNCTION DOC]
    # HELPER: autouse fixture parametrized over zero and one million (ids numexpr and python). Uses the monkeypatch fixture to set the module-level minimum-element threshold of the computation expressions module for the duration of each test, so that arithmetic goes through the numexpr path (threshold zero) or plain Python/NumPy (threshold huge). Yields the chosen threshold; the setting is restored afterwards.
    with monkeypatch.context() as m:
        m.setattr(expr, "_MIN_ELEMENTS", request.param)
        yield request.param


@pytest.fixture(
    params=[
        # TODO: add more  dtypes here
        Index(np.arange(5, dtype="float64")),
        Index(np.arange(5, dtype="int64")),
        Index(np.arange(5, dtype="uint64")),
        RangeIndex(5),
    ],
    ids=lambda x: type(x).__name__,
)
def numeric_idx(request):
    # [FUNCTION DOC]
    # Fixture parametrized over four numeric index objects of length five (float64, int64, uint64 indexes, and a range index); ids are the index class names. Returns the chosen index.
    """
    Several types of numeric-dtypes Index objects
    """
    return request.param


def adjust_negative_zero(zero, expected):
    # [FUNCTION DOC]
    # HELPER: adjusts an expected division result when the divisor is negative zero. Takes the zero fixture value(s) and an expected result. If the divisor contains negative zeros (sign bit set), asserts that all entries are negative zeros (the fixture is all-negative or none-negative) and then multiplies the expected result in place by minus one. Returns the expected result (mutated in place for the negative case).
    """
    Helper to adjust the expected result if we are dividing by -0.0
    as opposed to 0.0
    """
    if np.signbit(np.array(zero)).any():
        # All entries in the `zero` fixture should be either
        #  all-negative or no-negative.
        assert np.signbit(np.array(zero)).all()

        expected *= -1

    return expected


def compare_op(series, other, op):
    # [FUNCTION DOC]
    # HELPER: compares an operator applied to a series with the equivalent elementwise combination. Takes a left series, an other operand, and an operator. For power and reflected power, uses absolute values of both operands to avoid invalid values. Computes the operator directly (the vectorized path) and via the series combine method with the operator elementwise (pure Python); if the other is a series with a different index, clears the frequency of the python result's index; asserts the two series are equal.
    left = np.abs(series) if op in (ops.rpow, operator.pow) else series
    right = np.abs(other) if op in (ops.rpow, operator.pow) else other

    cython_or_numpy = op(left, right)
    python = left.combine(right, op)
    if isinstance(other, Series) and not other.index.equals(series.index):
        python.index = python.index._with_freq(None)
    tm.assert_series_equal(cython_or_numpy, python)


# TODO: remove this kludge once mypy stops giving false positives here
# List comprehension has incompatible type List[PandasObject]; expected List[RangeIndex]
#  See GH#29725
_ldtypes = ["i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f2", "f4", "f8"]
lefts: list[Index | Series] = [RangeIndex(10, 40, 10)]
lefts.extend([Series([10, 20, 30], dtype=dtype) for dtype in _ldtypes])
lefts.extend([Index([10, 20, 30], dtype=dtype) for dtype in _ldtypes if dtype != "f2"])

# ------------------------------------------------------------------
# Comparisons


class TestNumericComparisons:
    # [CLASS DOC]
    # Test class for comparison operators involving numeric data.
    def test_operator_series_comparison_zerorank(self):
        # [FUNCTION DOC]
        # Regression test for a linked issue about zero-rank scalars. Verifies a NumPy float scalar compared (greater-than) to a series, and a series compared (less-than) to a NumPy float scalar, equals the comparison with a Python float; also a NumPy array element compared against a series.
        # KNOWN ISSUE: in the last case the expected value compares against a different series (values one to three) than the one used for the result (zero to two), though both happen to give all false.
        # GH#13006
        result = np.float64(0) > Series([1, 2, 3])
        expected = 0.0 > Series([1, 2, 3])
        tm.assert_series_equal(result, expected)
        result = Series([1, 2, 3]) < np.float64(0)
        expected = Series([1, 2, 3]) < 0.0
        tm.assert_series_equal(result, expected)
        result = np.array([0, 1, 2])[0] > Series([0, 1, 2])
        expected = 0.0 > Series([1, 2, 3])
        tm.assert_series_equal(result, expected)

    def test_df_numeric_cmp_dt64_raises(self, box_with_array, fixed_now_ts):
        # [FUNCTION DOC]
        # Uses the box fixture and fixed-now timestamp fixture. Verifies comparing a boxed numeric array to a timestamp is invalid in the way the shared invalid-comparison helper checks (equality all false, inequality all true, ordering raises a type error).
        # GH#8932, GH#22163
        ts = fixed_now_ts
        obj = np.array(range(5))
        obj = tm.box_expected(obj, box_with_array)

        assert_invalid_comparison(obj, ts, box_with_array)

    def test_compare_invalid(self):
        # [FUNCTION DOC]
        # Linked to an issue.
        # KNOWN ISSUE: the name suggests an invalid comparison test, but the body verifies that dividing two random float series with different names (one a timestamp) equals the reciprocal of the reverse division, i.e. that mismatched names do not break arithmetic.
        # Verifies the equality of the two results.
        # GH#8058
        # ops testing
        a = Series(np.random.default_rng(2).standard_normal(5), name=0)
        b = Series(np.random.default_rng(2).standard_normal(5))
        b.name = pd.Timestamp("2000-01-01")
        tm.assert_series_equal(a / b, 1 / (b / a))

    def test_numeric_cmp_string_numexpr_path(self, box_with_array, monkeypatch):
        # [FUNCTION DOC]
        # Linked to issues. Uses the box fixture and monkeypatch. Sets the numexpr threshold to fifty and verifies that on a boxed 51-element float series, equality with a string gives all false and inequality all true (boxed as an array when the box is an index), and an ordering comparison with a string raises a type error "Invalid comparison between dtype=float64 and str".
        # GH#36377, GH#35700
        box = box_with_array
        xbox = box if box is not Index else np.ndarray

        obj = Series(np.random.default_rng(2).standard_normal(51))
        obj = tm.box_expected(obj, box, transpose=False)
        with monkeypatch.context() as m:
            m.setattr(expr, "_MIN_ELEMENTS", 50)
            result = obj == "a"

        expected = Series(np.zeros(51, dtype=bool))
        expected = tm.box_expected(expected, xbox, transpose=False)
        tm.assert_equal(result, expected)

        with monkeypatch.context() as m:
            m.setattr(expr, "_MIN_ELEMENTS", 50)
            result = obj != "a"
        tm.assert_equal(result, ~expected)

        msg = "Invalid comparison between dtype=float64 and str"
        with pytest.raises(TypeError, match=msg):
            obj < "a"


# ------------------------------------------------------------------
# Numeric dtypes Arithmetic with Datetime/Timedelta Scalar


class TestNumericArraylikeArithmeticWithDatetimeLike:
    # [CLASS DOC]
    # Test class for arithmetic between numeric array-likes and timedelta-like or datetime-like scalars and arrays.
    def test_mul_timedelta_list(self, box_with_array):
        # [FUNCTION DOC]
        # Linked to an issue. Uses the box fixture. Verifies multiplying a boxed integer array by a list of timedeltas, in either operand order, gives a boxed timedelta index of the elementwise products.
        # GH#62524
        box = box_with_array
        left = np.array([3, 4])
        left = tm.box_expected(left, box)

        right = [Timedelta(days=1), Timedelta(days=2)]

        result = left * right

        expected = TimedeltaIndex([Timedelta(days=3), Timedelta(days=8)])
        expected = tm.box_expected(expected, box)
        tm.assert_equal(result, expected)

        result2 = right * left
        tm.assert_equal(result2, expected)

    @pytest.mark.parametrize("box_cls", [np.array, Index, Series])
    @pytest.mark.parametrize(
        "left", lefts, ids=lambda x: type(x).__name__ + str(x.dtype)
    )
    def test_mul_td64arr(self, left, box_cls):
        # [FUNCTION DOC]
        # Linked to an issue. Parametrized over the container for the right side (NumPy array, index, series) and over a module-level list of left operands (a range index, series and indexes of many integer, unsigned, and float dtypes). Verifies multiplying the left (values ten, twenty, thirty) by a second-resolution timedelta array of one, two, three in either order gives seconds-resolution timedeltas ten, forty, ninety seconds, as a series if either operand is a series, else a timedelta index; the result keeps the right operand's timedelta dtype.
        # GH#22390
        right = np.array([1, 2, 3], dtype="m8[s]")
        right = box_cls(right)

        expected = TimedeltaIndex(["10s", "40s", "90s"], dtype=right.dtype)

        if isinstance(left, Series) or box_cls is Series:
            expected = Series(expected)
        assert expected.dtype == right.dtype

        result = left * right
        tm.assert_equal(result, expected)

        result = right * left
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("box_cls", [np.array, Index, Series])
    @pytest.mark.parametrize(
        "left", lefts, ids=lambda x: type(x).__name__ + str(x.dtype)
    )
    def test_div_td64arr(self, left, box_cls):
        # [FUNCTION DOC]
        # Linked to an issue. Same parametrization as the multiplication test. Verifies dividing and floor-dividing a timedelta array (ten, forty, ninety seconds) by the numeric left gives timedeltas of one, two, three seconds keeping dtype; and that dividing or floor-dividing the numeric by the timedelta raises a type error saying the ufunc cannot use operands with these types.
        # GH#22390
        right = np.array([10, 40, 90], dtype="m8[s]")
        right = box_cls(right)

        expected = TimedeltaIndex(["1s", "2s", "3s"], dtype=right.dtype)
        if isinstance(left, Series) or box_cls is Series:
            expected = Series(expected)
        assert expected.dtype == right.dtype

        result = right / left
        tm.assert_equal(result, expected)

        result = right // left
        tm.assert_equal(result, expected)

        # (true_) needed for min-versions build 2022-12-26
        msg = "ufunc '(true_)?divide' cannot use operands with types"
        with pytest.raises(TypeError, match=msg):
            left / right

        msg = "ufunc 'floor_divide' cannot use operands with types"
        with pytest.raises(TypeError, match=msg):
            left // right

    # TODO: also test Tick objects;
    #  see test_numeric_arr_rdiv_tdscalar for note on these failing
    @pytest.mark.parametrize(
        "scalar_td",
        [
            Timedelta(days=1).as_unit("ns"),
            Timedelta(days=1).as_unit("ns").to_timedelta64(),
            Timedelta(days=1).to_pytimedelta(),
            Timedelta(days=1).to_timedelta64().astype("timedelta64[s]"),
            Timedelta(days=1).to_timedelta64().astype("timedelta64[ms]"),
        ],
        ids=lambda x: type(x).__name__,
    )
    def test_numeric_arr_mul_tdscalar(self, scalar_td, numeric_idx, box_with_array):
        # [FUNCTION DOC]
        # Parametrized over scalar timedeltas (timedelta objects of one day in nanosecond unit, as NumPy value, as Python timedelta, and NumPy seconds and milliseconds units) and the numeric index fixture and box fixture. Verifies multiplying the index by the scalar, in either order, gives a boxed timedelta index of zero to four days, whose unit follows the scalar (NumPy scalars keep their unit; a Python timedelta gives microseconds; otherwise nanoseconds).
        # GH#19333
        box = box_with_array
        index = numeric_idx
        expected = TimedeltaIndex(
            [Timedelta(days=n) for n in range(len(index))], dtype="m8[ns]"
        )
        if isinstance(scalar_td, np.timedelta64):
            dtype = scalar_td.dtype
            expected = expected.astype(dtype)
        elif type(scalar_td) is timedelta:
            expected = expected.astype("m8[us]")

        index = tm.box_expected(index, box)
        expected = tm.box_expected(expected, box)

        result = index * scalar_td
        tm.assert_equal(result, expected)

        commute = scalar_td * index
        tm.assert_equal(commute, expected)

    @pytest.mark.parametrize(
        "scalar_td",
        [
            Timedelta(days=1).as_unit("ns"),
            Timedelta(days=1).as_unit("ns").to_timedelta64(),
            Timedelta(days=1).as_unit("ns").to_pytimedelta(),
        ],
        ids=lambda x: type(x).__name__,
    )
    @pytest.mark.parametrize("dtype", [np.int64, np.float64])
    def test_numeric_arr_mul_tdscalar_numexpr_path(
        self, dtype, scalar_td, box_with_array
    ):
        # [FUNCTION DOC]
        # Linked to an issue for the float case. Parametrized over scalar timedeltas, integer or float dtype, and the box fixture. Builds 20000 values as an array of the dtype and verifies multiplying by a one-day timedelta in either order gives the same values interpreted as days in nanoseconds (microseconds for a Python timedelta); intended to exercise the numexpr path for large arrays.
        # GH#44772 for the float64 case
        box = box_with_array

        arr_i8 = np.arange(2 * 10**4).astype(np.int64, copy=False)
        arr = arr_i8.astype(dtype, copy=False)
        obj = tm.box_expected(arr, box, transpose=False)

        expected = arr_i8.view("timedelta64[D]").astype("timedelta64[ns]")
        if type(scalar_td) is timedelta:
            expected = expected.astype("timedelta64[us]")

        expected = tm.box_expected(expected, box, transpose=False)

        result = obj * scalar_td
        tm.assert_equal(result, expected)

        result = scalar_td * obj
        tm.assert_equal(result, expected)

    def test_numeric_arr_rdiv_tdscalar(self, three_days, numeric_idx, box_with_array):
        # [FUNCTION DOC]
        # Uses the three-days fixture (various three-day timedelta-like scalars), the numeric index fixture, and the box fixture. Takes elements one and two of the index. Verifies dividing the scalar by the index gives timedeltas of three days and thirty-six hours, whose unit is chosen from the scalar (NumPy: its unit but at least seconds; Python timedelta or microsecond Timedelta: microseconds; Day/Hour/Minute/Second offsets: seconds). For a Day offset, scalar divided by index raises a type error about an unsupported operand type (Day no longer timedelta-like). Dividing the index by the scalar always raises a type error.
        box = box_with_array

        index = numeric_idx[1:3]

        expected = TimedeltaIndex(["3 Days", "36 Hours"])
        if isinstance(three_days, np.timedelta64):
            dtype = three_days.dtype
            if dtype < np.dtype("m8[s]"):
                # i.e. resolution is lower -> use lowest supported resolution
                dtype = np.dtype("m8[s]")
            expected = expected.astype(dtype)
        elif type(three_days) is timedelta or (
            isinstance(three_days, Timedelta) and three_days.unit == "us"
        ):
            expected = expected.astype("m8[us]")
        elif isinstance(
            three_days,
            (pd.offsets.Day, pd.offsets.Hour, pd.offsets.Minute, pd.offsets.Second),
        ):
            # closest reso is Second
            expected = expected.astype("m8[s]")

        index = tm.box_expected(index, box)
        expected = tm.box_expected(expected, box)

        if isinstance(three_days, pd.offsets.Day):
            # GH#41943 Day is no longer timedelta-like
            msg = "unsupported operand type"
            with pytest.raises(TypeError, match=msg):
                three_days / index
        else:
            result = three_days / index
            tm.assert_equal(result, expected)
            msg = "cannot use operands with types dtype"

        with pytest.raises(TypeError, match=msg):
            index / three_days

    @pytest.mark.parametrize(
        "other",
        [
            Timedelta(hours=31),
            Timedelta(hours=31).to_pytimedelta(),
            Timedelta(hours=31).to_timedelta64(),
            Timedelta(hours=31).to_timedelta64().astype("m8[h]"),
            np.timedelta64("NaT", "ns"),
            np.timedelta64("NaT", "D"),
            pd.offsets.Minute(3),
            pd.offsets.Second(0),
            # GH#28080 numeric+datetimelike should raise; Timestamp used
            #  to raise NullFrequencyError but that behavior was removed in 1.0
            pd.Timestamp("2021-01-01", tz="Asia/Tokyo"),
            pd.Timestamp("2021-01-01"),
            pd.Timestamp("2021-01-01").to_pydatetime(),
            pd.Timestamp("2021-01-01", tz="UTC").to_pydatetime(),
            pd.Timestamp("2021-01-01").to_datetime64(),
            np.datetime64("NaT", "ns"),
            pd.NaT,
        ],
        ids=repr,
    )
    @pytest.mark.filterwarnings(
        "ignore:.*'generic' unit for NumPy timedelta:DeprecationWarning"
    )
    def test_add_sub_datetimedeltalike_invalid(
        self, numeric_idx, other, box_with_array
    ):
        # [FUNCTION DOC]
        # Parametrized (ids are the reprs) over many invalid operands: timedeltas (various forms and units), timedelta NaT variants, minute and zero-second offsets, timestamps (naive and aware, various forms), datetime NaT and NaT. Uses the numeric index and box fixtures; ignores a NumPy deprecation warning about generic unit. Verifies that addition and subtraction (both operand orders) with the boxed numeric index raises a type error matching any of a set of accepted messages, via the shared invalid-add-sub helper.
        box = box_with_array

        left = tm.box_expected(numeric_idx, box)
        msg = "|".join(
            [
                "unsupported operand type",
                "Addition/subtraction of integers and integer-arrays",
                "Instead of adding/subtracting",
                "cannot use operands with types dtype",
                "Concatenation operation is not implemented for NumPy arrays",
                "Cannot (add|subtract) NaT (to|from) ndarray",
                # pd.array vs np.datetime64 case
                r"operand type\(s\) all returned NotImplemented from __array_ufunc__",
                "can only perform ops with numeric values",
                "cannot subtract DatetimeArray from ndarray",
                # pd.Timedelta(1) + Index([0, 1, 2])
                "Cannot add or subtract Timedelta from integers",
            ]
        )
        assert_invalid_addsub_type(left, other, msg)


# ------------------------------------------------------------------
# Arithmetic


class TestDivisionByZero:
    # [CLASS DOC]
    # Test class for division, floor division, modulo, and divmod by zero on numeric data.
    def test_div_zero(self, zero, numeric_idx):
        # [FUNCTION DOC]
        # Uses the zero fixture and numeric index. Verifies dividing an index of zero to four by zero gives NaN for zero and infinity for the others (sign flipped for negative zero); and that the equivalent int64 series division by zero gives the unadjusted expected values.
        idx = numeric_idx

        expected = Index([np.nan, np.inf, np.inf, np.inf, np.inf], dtype=np.float64)
        # We only adjust for Index, because Series does not yet apply
        #  the adjustment correctly.
        expected2 = adjust_negative_zero(zero, expected)

        result = idx / zero
        tm.assert_index_equal(result, expected2)
        ser_compat = Series(idx).astype("i8") / np.array(zero).astype("i8")
        tm.assert_series_equal(ser_compat, Series(expected))

    def test_floordiv_zero(self, zero, numeric_idx):
        # [FUNCTION DOC]
        # Same as the division test but for floor division: zero over zero is NaN, positive over zero is infinity (adjusted for negative zero); also checks an int64 series floor-divided by zero.
        idx = numeric_idx

        expected = Index([np.nan, np.inf, np.inf, np.inf, np.inf], dtype=np.float64)
        # We only adjust for Index, because Series does not yet apply
        #  the adjustment correctly.
        expected2 = adjust_negative_zero(zero, expected)

        result = idx // zero
        tm.assert_index_equal(result, expected2)
        ser_compat = Series(idx).astype("i8") // np.array(zero).astype("i8")
        tm.assert_series_equal(ser_compat, Series(expected))

    def test_mod_zero(self, zero, numeric_idx):
        # [FUNCTION DOC]
        # Verifies modulo of the numeric index by zero is all NaN float64, and the int64 series modulo by zero equals the series of that result.
        idx = numeric_idx

        expected = Index([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
        result = idx % zero
        tm.assert_index_equal(result, expected)
        ser_compat = Series(idx).astype("i8") % np.array(zero).astype("i8")
        tm.assert_series_equal(ser_compat, Series(result))

    def test_divmod_zero(self, zero, numeric_idx):
        # [FUNCTION DOC]
        # Verifies divmod of the numeric index by zero gives a quotient index of NaN then infinities (adjusted for negative zero) and a remainder index of all NaN.
        idx = numeric_idx

        exleft = Index([np.nan, np.inf, np.inf, np.inf, np.inf], dtype=np.float64)
        exright = Index([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
        exleft = adjust_negative_zero(zero, exleft)

        result = divmod(idx, zero)
        tm.assert_index_equal(result[0], exleft)
        tm.assert_index_equal(result[1], exright)

    @pytest.mark.parametrize("op", [operator.truediv, operator.floordiv])
    def test_div_negative_zero(self, zero, numeric_idx, op):
        # [FUNCTION DOC]
        # Parametrized over true division and floor division. Skipped for unsigned 64-bit indexes. For the index shifted down by three divided by zero, expects negative infinity three times, NaN, then infinity, adjusted for negative zero (so minus one divided by negative zero gives positive infinity).
        # Check that -1 / -0.0 returns np.inf, not -np.inf
        if numeric_idx.dtype == np.uint64:
            pytest.skip(f"Div by negative 0 not relevant for {numeric_idx.dtype}")
        idx = numeric_idx - 3

        expected = Index([-np.inf, -np.inf, -np.inf, np.nan, np.inf], dtype=np.float64)
        expected = adjust_negative_zero(zero, expected)

        result = op(idx, zero)
        tm.assert_index_equal(result, expected)

    # ------------------------------------------------------------------

    @pytest.mark.parametrize("dtype1", [np.int64, np.float64, np.uint64])
    def test_ser_div_ser(
        self,
        switch_numexpr_min_elements,
        dtype1,
        any_real_numpy_dtype,
    ):
        # [FUNCTION DOC]
        # Parametrized over the first series dtype (int64, float64, uint64), the second from the any-real-dtype fixture, and uses the numexpr-threshold fixture. Divides two series (second with zeros) and verifies the result equals the float64 division with infinity for the first three entries (division by zero), that dividing in the opposite order differs, and that for int64 by float32 under numexpr the expected dtype is float32.
        # no longer do integer div for any ops, but deal with the 0's
        dtype2 = any_real_numpy_dtype

        first = Series([3, 4, 5, 8], name="first").astype(dtype1)
        second = Series([0, 0, 0, 3], name="second").astype(dtype2)

        with np.errstate(all="ignore"):
            expected = Series(
                first.values.astype(np.float64) / second.values,
                dtype="float64",
                name=None,
            )
        expected.iloc[0:3] = np.inf
        if first.dtype == "int64" and second.dtype == "float32":
            # when using numexpr, the casting rules are slightly different
            # and int64/float32 combo results in float32 instead of float64
            if expr.USE_NUMEXPR and switch_numexpr_min_elements == 0:
                expected = expected.astype("float32")

        result = first / second
        tm.assert_series_equal(result, expected)
        assert not result.equals(second / first)

    @pytest.mark.parametrize("dtype1", [np.int64, np.float64, np.uint64])
    def test_ser_divmod_zero(self, dtype1, any_real_numpy_dtype):
        # [FUNCTION DOC]
        # Linked to an issue. Parametrized over dtype of the left side and the any-real-dtype fixture for the right. Verifies divmod of two series, including a zero divisor, matches floor division and modulo with the pandas convention of infinity for the quotient at the zero divisor (rather than NaN as in NumPy); and also for a NumPy array divided by a series (reflected case).
        # GH#26987
        dtype2 = any_real_numpy_dtype
        left = Series([1, 1]).astype(dtype1)
        right = Series([0, 2]).astype(dtype2)

        # GH#27321 pandas convention is to set 1 // 0 to np.inf, as opposed
        #  to numpy which sets to np.nan; patch `expected[0]` below
        expected = left // right, left % right
        expected = list(expected)
        expected[0] = expected[0].astype(np.float64)
        expected[0][0] = np.inf
        result = divmod(left, right)

        tm.assert_series_equal(result[0], expected[0])
        tm.assert_series_equal(result[1], expected[1])

        # rdivmod case
        result = divmod(left.values, right)
        tm.assert_series_equal(result[0], expected[0])
        tm.assert_series_equal(result[1], expected[1])

    def test_ser_divmod_inf(self):
        # [FUNCTION DOC]
        # Verifies divmod of series containing infinities equals floor division and modulo of them, with a series or a NumPy array on the left.
        left = Series([np.inf, 1.0])
        right = Series([np.inf, 2.0])

        expected = left // right, left % right
        result = divmod(left, right)

        tm.assert_series_equal(result[0], expected[0])
        tm.assert_series_equal(result[1], expected[1])

        # rdivmod case
        result = divmod(left.values, right)
        tm.assert_series_equal(result[0], expected[0])
        tm.assert_series_equal(result[1], expected[1])

    def test_rdiv_zero_compat(self):
        # [FUNCTION DOC]
        # Linked to an issue. Verifies a zero integer array divided by a random float series, a zero series divided by a float array, and series by series all give zero floats.
        # GH#8674
        zero_array = np.array([0] * 5)
        data = np.random.default_rng(2).standard_normal(5)
        expected = Series([0.0] * 5)

        result = zero_array / Series(data)
        tm.assert_series_equal(result, expected)

        result = Series(zero_array) / data
        tm.assert_series_equal(result, expected)

        result = Series(zero_array) / Series(data)
        tm.assert_series_equal(result, expected)

    def test_div_zero_inf_signs(self):
        # [FUNCTION DOC]
        # Linked to an issue. Verifies dividing an integer series (minus one, zero, one) by integer zero gives negative infinity, NaN, positive infinity, keeping the name.
        # GH#9144, inf signing
        ser = Series([-1, 0, 1], name="first")
        expected = Series([-np.inf, np.nan, np.inf], name="first")

        result = ser / 0
        tm.assert_series_equal(result, expected)

    def test_rdiv_zero(self):
        # [FUNCTION DOC]
        # Linked to an issue. Verifies integer zero divided by an integer series (minus one, zero, one) gives zero, NaN, zero as floats.
        # GH#9144
        ser = Series([-1, 0, 1], name="first")
        expected = Series([0.0, np.nan, 0.0], name="first")

        result = 0 / ser
        tm.assert_series_equal(result, expected)

    def test_floordiv_div(self):
        # [FUNCTION DOC]
        # Linked to an issue.
        # GH#9144
        ser = Series([-1, 0, 1], name="first")

        result = ser // 0
        expected = Series([-np.inf, np.nan, np.inf], name="first")
        tm.assert_series_equal(result, expected)

    def test_df_div_zero_df(self):
        # [FUNCTION DOC]
        # Linked to an issue. Verifies a frame of integer columns divided by itself gives ones, except NaN where zero is divided by zero.
        # integer div, but deal with the 0's (GH#9144)
        df = pd.DataFrame({"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]})
        result = df / df

        first = Series([1.0, 1.0, 1.0, 1.0])
        second = Series([np.nan, np.nan, np.nan, 1])
        expected = pd.DataFrame({"first": first, "second": second})
        tm.assert_frame_equal(result, expected)

    def test_df_div_zero_array(self):
        # [FUNCTION DOC]
        # Linked to an issue. Verifies that dividing a float copy of an integer frame's values by the integer values (NumPy, suppressing warnings) and wrapping in a frame gives ones except NaN for zero over zero.
        # integer div, but deal with the 0's (GH#9144)
        df = pd.DataFrame({"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]})

        first = Series([1.0, 1.0, 1.0, 1.0])
        second = Series([np.nan, np.nan, np.nan, 1])
        expected = pd.DataFrame({"first": first, "second": second})

        with np.errstate(all="ignore"):
            arr = df.values.astype("float") / df.values
        result = pd.DataFrame(arr, index=df.index, columns=df.columns)
        tm.assert_frame_equal(result, expected)

    def test_df_div_zero_int(self):
        # [FUNCTION DOC]
        # Linked to an issue. Verifies a frame of integers divided by integer zero gives infinity everywhere except NaN where the numerator is zero, and that the NumPy float division by zero rebuilt as a frame gives the same result.
        # integer div, but deal with the 0's (GH#9144)
        df = pd.DataFrame({"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]})

        result = df / 0
        expected = pd.DataFrame(np.inf, index=df.index, columns=df.columns)
        expected.iloc[0:3, 1] = np.nan
        tm.assert_frame_equal(result, expected)

        # numpy has a slightly different (wrong) treatment
        with np.errstate(all="ignore"):
            arr = df.values.astype("float64") / 0
        result2 = pd.DataFrame(arr, index=df.index, columns=df.columns)
        tm.assert_frame_equal(result2, expected)

    def test_df_div_zero_series_does_not_commute(self):
        # [FUNCTION DOC]
        # Verifies that dividing a column of a random frame by the frame differs from dividing the frame by the column (after filling NaN with zero), i.e. frame-series division is not commutative.
        # integer div, but deal with the 0's (GH#9144)
        df = pd.DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
        ser = df[0]
        res = ser / df
        res2 = df / ser
        assert not res.fillna(0).equals(res2.fillna(0))

    # ------------------------------------------------------------------
    # Mod By Zero

    def test_df_mod_zero_df(self):
        # [FUNCTION DOC]
        # Linked to an issue. Verifies a frame of integers modulo itself gives zeros as floats (NaN where zero modulo zero for the second column), and that when the frame is built without consolidating blocks the first column stays int64 in the result.
        # GH#3590, modulo as ints
        df = pd.DataFrame({"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]})
        # this is technically wrong, as the integer portion is coerced to float
        first = Series([0, 0, 0, 0])
        first = first.astype("float64")
        second = Series([np.nan, np.nan, np.nan, 0])
        expected = pd.DataFrame({"first": first, "second": second})
        result = df % df
        tm.assert_frame_equal(result, expected)

        # GH#38939 If we dont pass copy=False, df is consolidated and
        #  result["first"] is float64 instead of int64
        df = pd.DataFrame({"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]}, copy=False)
        first = Series([0, 0, 0, 0], dtype="int64")
        second = Series([np.nan, np.nan, np.nan, 0])
        expected = pd.DataFrame({"first": first, "second": second})
        result = df % df
        tm.assert_frame_equal(result, expected)

    def test_df_mod_zero_array(self):
        # [FUNCTION DOC]
        # Linked to an issue. Verifies the NumPy modulo of the frame's values by themselves, rebuilt as a float frame with the zero-divisor entries in the second column set to NaN, equals the expected frame of zeros and NaN.
        # GH#3590, modulo as ints
        df = pd.DataFrame({"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]})

        # this is technically wrong, as the integer portion is coerced to float
        # ###
        first = Series([0, 0, 0, 0], dtype="float64")
        second = Series([np.nan, np.nan, np.nan, 0])
        expected = pd.DataFrame({"first": first, "second": second})

        # numpy has a slightly different (wrong) treatment
        with np.errstate(all="ignore"):
            arr = df.values % df.values
        result2 = pd.DataFrame(arr, index=df.index, columns=df.columns, dtype="float64")
        result2.iloc[0:3, 1] = np.nan
        tm.assert_frame_equal(result2, expected)

    def test_df_mod_zero_int(self):
        # [FUNCTION DOC]
        # Linked to an issue. Verifies a frame of integers modulo integer zero is all NaN, and that the NumPy float modulo by zero rebuilt as a frame matches.
        # GH#3590, modulo as ints
        df = pd.DataFrame({"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]})

        result = df % 0
        expected = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
        tm.assert_frame_equal(result, expected)

        # numpy has a slightly different (wrong) treatment
        with np.errstate(all="ignore"):
            arr = df.values.astype("float64") % 0
        result2 = pd.DataFrame(arr, index=df.index, columns=df.columns)
        tm.assert_frame_equal(result2, expected)

    def test_df_mod_zero_series_does_not_commute(self):
        # [FUNCTION DOC]
        # Verifies series modulo frame differs from frame modulo series for a random frame (not commutative).
        # GH#3590, modulo as ints
        # not commutative with series
        df = pd.DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
        ser = df[0]
        res = ser % df
        res2 = df % ser
        assert not res.fillna(0).equals(res2.fillna(0))


class TestMultiplicationDivision:
    # [CLASS DOC]
    # Test class for multiplication, division, floor division, modulo, power, and divmod on numeric (non-datetime-like) dtypes.
    # __mul__, __rmul__, __div__, __rdiv__, __floordiv__, __rfloordiv__
    # for non-timestamp/timedelta/period dtypes

    def test_divide_decimal(self, box_with_array):
        # [FUNCTION DOC]
        # Linked to an issue. Uses the box fixture. Verifies that true division and floor division of a boxed object series holding a Decimal by a Decimal scalar give the exact Decimal quotient.
        # resolves issue GH#9787
        box = box_with_array
        ser = Series([Decimal(10)])
        expected = Series([Decimal(5)])

        ser = tm.box_expected(ser, box)
        expected = tm.box_expected(expected, box)

        result = ser / Decimal(2)

        tm.assert_equal(result, expected)

        result = ser // Decimal(2)
        tm.assert_equal(result, expected)

    def test_div_equiv_binop(self):
        # [FUNCTION DOC]
        # Linked to an issue. Verifies the div method and the division operator on integer-by-float series give the float quotient with infinity for division by zero; the method result is compared ignoring names.
        # Test Series.div as well as Series.__div__
        # float/integer issue
        # GH#7785
        first = Series([1, 0], name="first")
        second = Series([-0.01, -0.02], name="second")
        expected = Series([-0.01, -np.inf])

        result = second.div(first)
        tm.assert_series_equal(result, expected, check_names=False)

        result = second / first
        tm.assert_series_equal(result, expected)

    def test_div_int(self, numeric_idx):
        # [FUNCTION DOC]
        # Uses the numeric index fixture. Verifies dividing by one gives the float64 cast of the index and dividing by two gives the index of half values.
        idx = numeric_idx
        result = idx / 1
        expected = idx.astype("float64")
        tm.assert_index_equal(result, expected)

        result = idx / 2
        expected = Index(idx.values / 2)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("op", [operator.mul, ops.rmul, operator.floordiv])
    def test_mul_int_identity(self, op, numeric_idx, box_with_array):
        # [FUNCTION DOC]
        # Parametrized over multiplication, reflected multiplication, and floor division; uses the numeric index and box fixtures. Verifies that operating with the integer one returns the original boxed object unchanged.
        idx = numeric_idx
        idx = tm.box_expected(idx, box_with_array)

        result = op(idx, 1)
        tm.assert_equal(result, idx)

    def test_mul_int_array(self, numeric_idx):
        # [FUNCTION DOC]
        # Uses the numeric index fixture. Verifies multiplying the index by a zero-dimensional int64 array equals multiplication by five, and multiplying by an integer array of zero to four (unsigned for the uint64 index) equals the index squared.
        idx = numeric_idx
        didx = idx * idx

        result = idx * np.array(5, dtype="int64")
        tm.assert_index_equal(result, idx * 5)

        arr_dtype = "uint64" if idx.dtype == np.uint64 else "int64"
        result = idx * np.arange(5, dtype=arr_dtype)
        tm.assert_index_equal(result, didx)

    def test_mul_int_series(self, numeric_idx):
        # [FUNCTION DOC]
        # Verifies multiplying the index by a series of integers zero to four gives a series equal to the index squared (unsigned dtype for the uint64 index).
        idx = numeric_idx
        didx = idx * idx

        arr_dtype = "uint64" if idx.dtype == np.uint64 else "int64"
        result = idx * Series(np.arange(5, dtype=arr_dtype))
        tm.assert_series_equal(result, Series(didx))

    def test_mul_float_series(self, numeric_idx):
        # [FUNCTION DOC]
        # Verifies multiplying the index by a float series offset by one tenth gives a series of the elementwise products.
        idx = numeric_idx
        rng5 = np.arange(5, dtype="float64")

        result = idx * Series(rng5 + 0.1)
        expected = Series(rng5 * (rng5 + 0.1))
        tm.assert_series_equal(result, expected)

    def test_mul_index(self, numeric_idx):
        # [FUNCTION DOC]
        # Verifies multiplying the index by itself equals raising it to the power two.
        idx = numeric_idx

        result = idx * idx
        tm.assert_index_equal(result, idx**2)

    def test_mul_datelike_raises(self, numeric_idx):
        # [FUNCTION DOC]
        # Verifies multiplying a numeric index by a datetime index raises a type error with message "cannot perform __rmul__ with this index type".
        idx = numeric_idx
        msg = "cannot perform __rmul__ with this index type"
        with pytest.raises(TypeError, match=msg):
            idx * date_range("20130101", periods=5)

    def test_mul_size_mismatch_raises(self, numeric_idx):
        # [FUNCTION DOC]
        # Verifies multiplying the index by an index of different length, or by a two-element array, raises a value error "operands could not be broadcast together".
        idx = numeric_idx
        msg = "operands could not be broadcast together"
        with pytest.raises(ValueError, match=msg):
            idx * idx[0:3]
        with pytest.raises(ValueError, match=msg):
            idx * np.array([1, 2])

    @pytest.mark.parametrize("op", [operator.pow, ops.rpow])
    def test_pow_float(self, op, numeric_idx, box_with_array):
        # [FUNCTION DOC]
        # Parametrized over power and reflected power (via the operator functions); uses the numeric index and box fixtures. Verifies that raising a boxed index to the float 2.0 (or 2.0 to the index) equals the same operation on the raw values, boxed.
        # test power calculations both ways, GH#14973
        box = box_with_array
        idx = numeric_idx
        expected = Index(op(idx.values, 2.0))

        idx = tm.box_expected(idx, box)
        expected = tm.box_expected(expected, box)

        result = op(idx, 2.0)
        tm.assert_equal(result, expected)

    def test_modulo(self, numeric_idx, box_with_array):
        # [FUNCTION DOC]
        # Linked to an issue. Uses the numeric index and box fixtures. Verifies modulo two of the boxed index equals the modulo of the raw values, boxed.
        # GH#9244
        box = box_with_array
        idx = numeric_idx
        expected = Index(idx.values % 2)

        idx = tm.box_expected(idx, box)
        expected = tm.box_expected(expected, box)

        result = idx % 2
        tm.assert_equal(result, expected)

    def test_divmod_scalar(self, numeric_idx):
        # [FUNCTION DOC]
        # Verifies divmod of the numeric index by the integer two returns an index pair equal to the divmod of the raw values (errors suppressed).
        idx = numeric_idx

        result = divmod(idx, 2)
        with np.errstate(all="ignore"):
            div, mod = divmod(idx.values, 2)

        expected = Index(div), Index(mod)
        for r, e in zip(result, expected, strict=True):
            tm.assert_index_equal(r, e)

    def test_divmod_ndarray(self, numeric_idx):
        # [FUNCTION DOC]
        # Verifies divmod of the index by an array of twos of the same dtype returns an index pair equal to the NumPy divmod.
        idx = numeric_idx
        other = np.ones(idx.values.shape, dtype=idx.values.dtype) * 2

        result = divmod(idx, other)
        with np.errstate(all="ignore"):
            div, mod = divmod(idx.values, other)

        expected = Index(div), Index(mod)
        for r, e in zip(result, expected, strict=True):
            tm.assert_index_equal(r, e)

    def test_divmod_series(self, numeric_idx):
        # [FUNCTION DOC]
        # Verifies divmod of the index by a series of twos returns a pair of series equal to the NumPy divmod of the raw values.
        idx = numeric_idx
        other = np.ones(idx.values.shape, dtype=idx.values.dtype) * 2

        result = divmod(idx, Series(other))
        with np.errstate(all="ignore"):
            div, mod = divmod(idx.values, other)

        expected = Series(div), Series(mod)
        for r, e in zip(result, expected, strict=True):
            tm.assert_series_equal(r, e)

    @pytest.mark.parametrize("other", [np.nan, 7, -23, 2.718, -3.14, np.inf])
    def test_ops_np_scalar(self, other):
        # [FUNCTION DOC]
        # Parametrized over scalars NaN, 7, -23, 2.718, -3.14, and infinity. Builds a random five-by-three float frame with labels via a local lambda and verifies division, reflected multiplication, addition, and reflected subtraction with a zero-dimensional NumPy array equal the same operations with the plain scalar applied to the raw values.
        vals = np.random.default_rng(2).standard_normal((5, 3))
        f = lambda x: pd.DataFrame(
            x, index=list("ABCDE"), columns=["jim", "joe", "jolie"]
        )

        df = f(vals)

        tm.assert_frame_equal(df / np.array(other), f(vals / other))
        tm.assert_frame_equal(np.array(other) * df, f(vals * other))
        tm.assert_frame_equal(df + np.array(other), f(vals + other))
        tm.assert_frame_equal(np.array(other) - df, f(other - vals))

    # TODO: This came from series.test.test_operators, needs cleanup
    def test_operators_frame(self):
        # [FUNCTION DOC]
        # Moved from the series operator tests (a TODO says it needs cleanup). Verifies that for a float time series and a one-column frame built from it, addition, power, less-than, and division of the series with itself equal those with the frame's column (ignoring names).
        # rpow does not work with DataFrame
        ts = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )
        ts.name = "ts"

        df = pd.DataFrame({"A": ts})

        tm.assert_series_equal(ts + ts, ts + df["A"], check_names=False)
        tm.assert_series_equal(ts**ts, ts ** df["A"], check_names=False)
        tm.assert_series_equal(ts < ts, ts < df["A"], check_names=False)
        tm.assert_series_equal(ts / ts, ts / df["A"], check_names=False)

    # TODO: this came from tests.series.test_analytics, needs cleanup and
    #  de-duplication with test_modulo above
    def test_modulo2(self):
        # [FUNCTION DOC]
        # Moved from the series analytics tests (a TODO says to clean up and de-duplicate with the modulo test above). With warnings suppressed, verifies modulo of integer columns gives float64 with NaN where the divisor is zero, modulo by zero gives all NaN (named), the same after casting to float, and that modulo is not commutative.
        # KNOWN ISSUE: the float cast statement appears twice before the final non-commutativity check; the second cast is redundant.
        with np.errstate(all="ignore"):
            # GH#3590, modulo as ints
            p = pd.DataFrame({"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]})
            result = p["first"] % p["second"]
            expected = Series(p["first"].values % p["second"].values, dtype="float64")
            expected.iloc[0:3] = np.nan
            tm.assert_series_equal(result, expected)

            result = p["first"] % 0
            expected = Series(np.nan, index=p.index, name="first")
            tm.assert_series_equal(result, expected)

            p = p.astype("float64")
            result = p["first"] % p["second"]
            expected = Series(p["first"].values % p["second"].values)
            tm.assert_series_equal(result, expected)

            p = p.astype("float64")
            result = p["first"] % p["second"]
            result2 = p["second"] % p["first"]
            assert not result.equals(result2)

    def test_modulo_zero_int(self):
        # [FUNCTION DOC]
        # Linked to an issue. With warnings suppressed, verifies an integer series (zero, one) modulo zero is all NaN, and zero modulo the series gives NaN then zero as floats.
        # GH#9144
        with np.errstate(all="ignore"):
            s = Series([0, 1])

            result = s % 0
            expected = Series([np.nan, np.nan])
            tm.assert_series_equal(result, expected)

            result = 0 % s
            expected = Series([np.nan, 0.0])
            tm.assert_series_equal(result, expected)

    def test_non_1d_ea_raises_notimplementederror(self):
        # [FUNCTION DOC]
        # Linked to an issue. Verifies that multiplying a two-dimensional nullable Int64 array by a two-dimensional NumPy array, in either order, raises a not-implemented error saying ops are only supported for one-dimensional structures.
        # GH#61866
        ea_array = array([1, 2, 3, 4, 5], dtype="Int64").reshape(5, 1)
        np_array = np.array([1, 2, 3, 4, 5], dtype=np.int64).reshape(5, 1)

        msg = "can only perform ops with 1-d structures"

        with pytest.raises(NotImplementedError, match=msg):
            ea_array * np_array

        with pytest.raises(NotImplementedError, match=msg):
            np_array * ea_array


class TestAdditionSubtraction:
    # [CLASS DOC]
    # Test class for addition and subtraction (including reflected and in-place forms) on numeric dtypes.
    # __add__, __sub__, __radd__, __rsub__, __iadd__, __isub__
    # for non-timestamp/timedelta/period dtypes

    @pytest.mark.parametrize(
        "first, second, expected",
        [
            (
                Series([1, 2, 3], index=list("ABC"), name="x"),
                Series([2, 2, 2], index=list("ABD"), name="x"),
                Series([3.0, 4.0, np.nan, np.nan], index=list("ABCD"), name="x"),
            ),
            (
                Series([1, 2, 3], index=list("ABC"), name="x"),
                Series([2, 2, 2, 2], index=list("ABCD"), name="x"),
                Series([3, 4, 5, np.nan], index=list("ABCD"), name="x"),
            ),
        ],
    )
    def test_add_series(self, first, second, expected):
        # [FUNCTION DOC]
        # Linked to an issue. Parametrized over two pairs of named series with partly different indexes. Verifies addition aligns on the index union in either operand order and yields NaN for non-overlapping labels (float result if any NaN).
        # GH#1134
        tm.assert_series_equal(first + second, expected)
        tm.assert_series_equal(second + first, expected)

    @pytest.mark.parametrize(
        "first, second, expected",
        [
            (
                pd.DataFrame({"x": [1, 2, 3]}, index=list("ABC")),
                pd.DataFrame({"x": [2, 2, 2]}, index=list("ABD")),
                pd.DataFrame({"x": [3.0, 4.0, np.nan, np.nan]}, index=list("ABCD")),
            ),
            (
                pd.DataFrame({"x": [1, 2, 3]}, index=list("ABC")),
                pd.DataFrame({"x": [2, 2, 2, 2]}, index=list("ABCD")),
                pd.DataFrame({"x": [3, 4, 5, np.nan]}, index=list("ABCD")),
            ),
        ],
    )
    def test_add_frames(self, first, second, expected):
        # [FUNCTION DOC]
        # Linked to an issue. Same as the series test but with single-column frames with different indexes: addition aligns on the union of indexes in either order.
        # GH#1134
        tm.assert_frame_equal(first + second, expected)
        tm.assert_frame_equal(second + first, expected)

    # TODO: This came from series.test.test_operators, needs cleanup
    def test_series_frame_radd_bug(self, fixed_now_ts):
        # [FUNCTION DOC]
        # Moved from the series operator tests (a TODO says it needs cleanup). Uses the fixed-now timestamp fixture. Verifies a string prefix plus a string series or a one-column frame of strings prepends the prefix, and that adding a Python datetime to a float time series in either order raises a type error with an unsupported-operand (or, due to a NumPy message bug, concatenation) message.
        # GH#353
        vals = Series([str(i) for i in range(5)])
        result = "foo_" + vals
        expected = vals.map(lambda x: "foo_" + x)
        tm.assert_series_equal(result, expected)

        frame = pd.DataFrame({"vals": vals})
        result = "foo_" + frame
        expected = pd.DataFrame({"vals": vals.map(lambda x: "foo_" + x)})
        tm.assert_frame_equal(result, expected)

        ts = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )

        # really raise this time
        fix_now = fixed_now_ts.to_pydatetime()
        msg = "|".join(
            [
                "unsupported operand type",
                # wrong error message, see https://github.com/numpy/numpy/issues/18832
                "Concatenation operation",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            fix_now + ts

        with pytest.raises(TypeError, match=msg):
            ts + fix_now

    # TODO: This came from series.test.test_operators, needs cleanup
    def test_datetime64_with_index(self):
        # [FUNCTION DOC]
        # Moved from the series operator tests (a TODO says it needs cleanup). Verifies subtracting an index from a series equals subtracting the index converted to a series, for random floats and for datetimes; that subtracting a period index from a datetime series raises a type error "cannot subtract PeriodArray from DatetimeArray"; and that a frame column of a constant timestamp minus the datetime index equals the minus of the index as series (ignoring names).
        # arithmetic integer ops with an index
        ser = Series(np.random.default_rng(2).standard_normal(5))
        expected = ser - ser.index.to_series()
        result = ser - ser.index
        tm.assert_series_equal(result, expected)

        # GH#4629
        # arithmetic datetime64 ops with an index
        ser = Series(
            date_range("20130101", periods=5),
            index=date_range("20130101", periods=5),
        )
        expected = ser - ser.index.to_series()
        result = ser - ser.index
        tm.assert_series_equal(result, expected)

        msg = "cannot subtract PeriodArray from DatetimeArray"
        with pytest.raises(TypeError, match=msg):
            # GH#18850
            result = ser - ser.index.to_period()

        df = pd.DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)),
            index=date_range("20130101", periods=5),
        )
        df["date"] = pd.Timestamp("20130102")
        df["expected"] = df["date"] - df.index.to_series()
        df["result"] = df["date"] - df.index
        tm.assert_series_equal(df["result"], df["expected"], check_names=False)

    # TODO: taken from tests.frame.test_operators, needs cleanup
    def test_frame_operators(self, float_frame):
        # [FUNCTION DOC]
        # Moved from the frame operator tests (a TODO says it needs cleanup). Uses the float-frame fixture. Verifies elementwise that adding the frame to itself doubles each value (NaN where original is NaN), and adding a series indexed by the column names adds the column's series value to each element (NaN consistent).
        frame = float_frame

        garbage = np.random.default_rng(2).random(4)
        colSeries = Series(garbage, index=np.array(frame.columns))

        idSum = frame + frame
        seriesSum = frame + colSeries

        for col, series in idSum.items():
            for idx, val in series.items():
                origVal = frame[col][idx] * 2
                if not np.isnan(val):
                    assert val == origVal
                else:
                    assert np.isnan(origVal)

        for col, series in seriesSum.items():
            for idx, val in series.items():
                origVal = frame[col][idx] + colSeries[col]
                if not np.isnan(val):
                    assert val == origVal
                else:
                    assert np.isnan(origVal)

    def test_frame_operators_col_align(self, float_frame):
        # [FUNCTION DOC]
        # Uses the float-frame fixture. Builds a frame with reordered columns and verifies adding it to itself equals doubling it.
        frame2 = pd.DataFrame(float_frame, columns=["D", "C", "B", "A"])
        added = frame2 + frame2
        expected = frame2 * 2
        tm.assert_frame_equal(added, expected)

    def test_frame_operators_none_to_nan(self):
        # [FUNCTION DOC]
        # Verifies that adding a frame of strings with a None to itself concatenates strings and gives NaN for the missing entry.
        df = pd.DataFrame({"a": ["a", None, "b"]})
        tm.assert_frame_equal(df + df, pd.DataFrame({"a": ["aa", np.nan, "bb"]}))

    @pytest.mark.parametrize("dtype", ("float", "int64"))
    def test_frame_operators_empty_like(self, dtype):
        # [FUNCTION DOC]
        # Linked to an issue. Parametrized over float and int64 dtypes. Verifies that a frame added to itself equals itself (and the equals method is true) for an empty frame, a frame with a column but no rows, and a frame with a row but no columns.
        # Test for issue #10181
        frames = [
            pd.DataFrame(dtype=dtype),
            pd.DataFrame(columns=["A"], dtype=dtype),
            pd.DataFrame(index=[0], dtype=dtype),
        ]
        for df in frames:
            assert (df + df).equals(df)
            tm.assert_frame_equal(df + df, df)

    @pytest.mark.parametrize(
        "func",
        [lambda x: x * 2, lambda x: x[::2], lambda x: 5],
        ids=["multiply", "slice", "constant"],
    )
    def test_series_operators_arithmetic(self, all_arithmetic_functions, func):
        # [FUNCTION DOC]
        # Parametrized over three functions deriving the other operand from the series (doubling it, taking every second element, and a constant five) and the arithmetic-function fixture. Verifies the operator on a ten-element float time series with that operand equals the elementwise Python combination, via the compare helper.
        op = all_arithmetic_functions
        series = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )
        other = func(series)
        compare_op(series, other, op)

    @pytest.mark.parametrize(
        "func", [lambda x: x + 1, lambda x: 5], ids=["add", "constant"]
    )
    def test_series_operators_compare(self, comparison_op, func):
        # [FUNCTION DOC]
        # Parametrized over two functions deriving the other operand (adding one, constant five) and the comparison-operator fixture. Verifies the comparison matches the elementwise Python combination, via the compare helper.
        op = comparison_op
        series = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )
        other = func(series)
        compare_op(series, other, op)

    @pytest.mark.parametrize(
        "func",
        [lambda x: x * 2, lambda x: x[::2], lambda x: 5],
        ids=["multiply", "slice", "constant"],
    )
    def test_divmod(self, func):
        # [FUNCTION DOC]
        # Parametrized over three functions deriving the other operand (doubling, every-second-element slice, a constant). Computes divmod of a ten-element float time series with the operand. When the operand is a shorter iterable (the slice case), the NumPy expectation is built by interleaving the operand values with NaN to length-align; otherwise the operand itself. Verifies each of the quotient and remainder results is almost equal to the NumPy divmod values, keeps the series name, and has an index equal to the original index with its frequency cleared.
        series = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )
        other = func(series)
        results = divmod(series, other)
        if isinstance(other, abc.Iterable) and len(series) != len(other):
            # if the lengths don't match, this is the test where we use
            # `tser[::2]`. Pad every other value in `other_np` with nan.
            other_np = []
            for n in other:
                other_np.append(n)
                other_np.append(np.nan)
        else:
            other_np = other
        other_np = np.asarray(other_np)
        with np.errstate(all="ignore"):
            expecteds = divmod(series.values, np.asarray(other_np))

        for result, expected in zip(results, expecteds, strict=True):
            # check the values, name, and index separately
            tm.assert_almost_equal(np.asarray(result), expected)

            assert result.name == series.name
            tm.assert_index_equal(result.index, series.index._with_freq(None))

    def test_series_divmod_zero(self):
        # [FUNCTION DOC]
        # Verifies divmod of a float series by zeros follows the library convention for division by zero (positive over zero is infinity, rather than NumPy's NaN): quotient all infinity and remainder all NaN, keeping name and index.
        # Check that divmod uses pandas convention for division by zero,
        #  which does not match numpy.
        # pandas convention has
        #  1/0 == np.inf
        #  -1/0 == -np.inf
        #  1/-0.0 == -np.inf
        #  -1/-0.0 == np.inf
        tser = Series(
            np.arange(1, 11, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )
        other = tser * 0

        result = divmod(tser, other)
        exp1 = Series([np.inf] * len(tser), index=tser.index, name="ts")
        exp2 = Series([np.nan] * len(tser), index=tser.index, name="ts")
        tm.assert_series_equal(result[0], exp1)
        tm.assert_series_equal(result[1], exp2)


class TestUFuncCompat:
    # [CLASS DOC]
    # Test class for compatibility of NumPy universal functions with indexes and series.
    # TODO: add more dtypes
    @pytest.mark.parametrize("holder", [Index, RangeIndex, Series])
    @pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float64])
    def test_ufunc_compat(self, holder, dtype):
        # [FUNCTION DOC]
        # Parametrized over container type (index, range index, series) and dtype (int64, uint64, float64). Applies the NumPy sine to a container named foo (range index only for int64, otherwise skipped) and verifies the result equals an equivalent container of sines of the raw values with the name kept.
        box = Series if holder is Series else Index

        if holder is RangeIndex:
            if dtype != np.int64:
                pytest.skip(f"dtype {dtype} not relevant for RangeIndex")
            idx = RangeIndex(0, 5, name="foo")
        else:
            idx = holder(np.arange(5, dtype=dtype), name="foo")
        result = np.sin(idx)
        expected = box(np.sin(np.arange(5, dtype=dtype)), name="foo")
        tm.assert_equal(result, expected)

    # TODO: add more dtypes
    @pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float64])
    def test_ufunc_coercions(self, index_or_series, dtype):
        # [FUNCTION DOC]
        # Parametrized over dtype and the index-or-series fixture. For a container of one to five named x: verifies square root, divide by 2.0, plus 2.0, minus 2.0, times 1.0, and true divide by 2.0 each return float64 results of the same container class with the name kept and the expected float values.
        idx = index_or_series([1, 2, 3, 4, 5], dtype=dtype, name="x")
        box = index_or_series

        result = np.sqrt(idx)
        assert result.dtype == "f8" and isinstance(result, box)
        exp = Index(np.sqrt(np.array([1, 2, 3, 4, 5], dtype=np.float64)), name="x")
        exp = tm.box_expected(exp, box)
        tm.assert_equal(result, exp)

        result = np.divide(idx, 2.0)
        assert result.dtype == "f8" and isinstance(result, box)
        exp = Index([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64, name="x")
        exp = tm.box_expected(exp, box)
        tm.assert_equal(result, exp)

        # _evaluate_numeric_binop
        result = idx + 2.0
        assert result.dtype == "f8" and isinstance(result, box)
        exp = Index([3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float64, name="x")
        exp = tm.box_expected(exp, box)
        tm.assert_equal(result, exp)

        result = idx - 2.0
        assert result.dtype == "f8" and isinstance(result, box)
        exp = Index([-1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float64, name="x")
        exp = tm.box_expected(exp, box)
        tm.assert_equal(result, exp)

        result = idx * 1.0
        assert result.dtype == "f8" and isinstance(result, box)
        exp = Index([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64, name="x")
        exp = tm.box_expected(exp, box)
        tm.assert_equal(result, exp)

        result = idx / 2.0
        assert result.dtype == "f8" and isinstance(result, box)
        exp = Index([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64, name="x")
        exp = tm.box_expected(exp, box)
        tm.assert_equal(result, exp)

    # TODO: add more dtypes
    @pytest.mark.parametrize("holder", [Index, Series])
    @pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float64])
    def test_ufunc_multiple_return_values(self, holder, dtype):
        # [FUNCTION DOC]
        # Parametrized over container (index or series) and dtype. Verifies a ufunc with two outputs (the fractional/integral split) applied to a container of one, two, three returns a tuple of two containers: fractional parts all zero and integral parts one, two, three, as float64, boxed like the input, with name kept.
        obj = holder([1, 2, 3], dtype=dtype, name="x")
        box = Series if holder is Series else Index

        result = np.modf(obj)
        assert isinstance(result, tuple)
        exp1 = Index([0.0, 0.0, 0.0], dtype=np.float64, name="x")
        exp2 = Index([1.0, 2.0, 3.0], dtype=np.float64, name="x")
        tm.assert_equal(result[0], tm.box_expected(exp1, box))
        tm.assert_equal(result[1], tm.box_expected(exp2, box))

    def test_ufunc_at(self):
        # [FUNCTION DOC]
        # Verifies the in-place add-at ufunc method on a series, adding ten at positions zero and two, modifies the series in place producing ten, one, twelve with the index and name unchanged.
        s = Series([0, 1, 2], index=[1, 2, 3], name="x")
        np.add.at(s, [0, 2], 10)
        expected = Series([10, 1, 12], index=[1, 2, 3], name="x")
        tm.assert_series_equal(s, expected)


class TestObjectDtypeEquivalence:
    # [CLASS DOC]
    # Test class asserting that arithmetic operations on numeric data match behavior of the same operations executed elementwise on object dtype.
    # Tests that arithmetic operations match operations executed elementwise

    @pytest.mark.parametrize("dtype", [None, object])
    def test_numarr_with_dtype_add_nan(self, dtype, box_with_array):
        # [FUNCTION DOC]
        # Parametrized over dtype none (integer) and object; uses the box fixture. Verifies adding NaN on either side to a boxed integer series gives all NaN of the same dtype (float for inferred integers).
        box = box_with_array
        ser = Series([1, 2, 3], dtype=dtype)
        expected = Series([np.nan, np.nan, np.nan], dtype=dtype)

        ser = tm.box_expected(ser, box)
        expected = tm.box_expected(expected, box)

        result = np.nan + ser
        tm.assert_equal(result, expected)

        result = ser + np.nan
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("dtype", [None, object])
    def test_numarr_with_dtype_add_int(self, dtype, box_with_array):
        # [FUNCTION DOC]
        # Parametrized over dtype none and object; uses the box fixture. Verifies adding the integer one on either side to a boxed series of one, two, three gives two, three, four with the same dtype.
        box = box_with_array
        ser = Series([1, 2, 3], dtype=dtype)
        expected = Series([2, 3, 4], dtype=dtype)

        ser = tm.box_expected(ser, box)
        expected = tm.box_expected(expected, box)

        result = 1 + ser
        tm.assert_equal(result, expected)

        result = ser + 1
        tm.assert_equal(result, expected)

    # TODO: moved from tests.series.test_operators; needs cleanup
    @pytest.mark.parametrize(
        "op",
        [operator.add, operator.sub, operator.mul, operator.truediv, operator.floordiv],
    )
    def test_operators_reverse_object(self, op):
        # [FUNCTION DOC]
        # Moved from the series operator tests (a TODO says it needs cleanup). Parametrized over add, subtract, multiply, true divide, and floor divide. Verifies the reflected operation of a float scalar with an object-dtype series of random floats equals the operation with the series cast to float (after casting the result to float).
        # GH#56
        arr = Series(
            np.random.default_rng(2).standard_normal(10),
            index=np.arange(10),
            dtype=object,
        )

        result = op(1.0, arr)
        expected = op(1.0, arr.astype(float))
        tm.assert_series_equal(result.astype(float), expected)


class TestNumericArithmeticUnsorted:
    # [CLASS DOC]
    # Test class holding assorted arithmetic tests moved from type-specific modules and not yet sorted, parametrized, or de-duplicated; covers range-index operations, divmod fallbacks and miscellaneous cases.
    # Tests in this class have been moved from type-specific test modules
    #  but not yet sorted, parametrized, and de-duplicated
    @pytest.mark.parametrize(
        "op",
        [
            operator.add,
            operator.sub,
            operator.mul,
            operator.floordiv,
            operator.truediv,
        ],
    )
    @pytest.mark.parametrize(
        "idx1",
        [
            RangeIndex(0, 10, 1),
            RangeIndex(0, 20, 2),
            RangeIndex(-10, 10, 2),
            RangeIndex(5, -5, -1),
        ],
    )
    @pytest.mark.parametrize(
        "idx2",
        [
            RangeIndex(0, 10, 1),
            RangeIndex(0, 20, 2),
            RangeIndex(-10, 10, 2),
            RangeIndex(5, -5, -1),
        ],
    )
    def test_binops_index(self, op, idx1, idx2):
        # [FUNCTION DOC]
        # Parametrized over add, subtract, multiply, floor divide, true divide and over pairs from four range indexes (various steps, negative ranges). After renaming the operands to different names, verifies the operator on two range indexes equals the operator on indexes built from their raw values, with an equivalent (range or integer) index type accepted.
        idx1 = idx1._rename("foo")
        idx2 = idx2._rename("bar")
        result = op(idx1, idx2)
        expected = op(Index(idx1.to_numpy()), Index(idx2.to_numpy()))
        tm.assert_index_equal(result, expected, exact="equiv")

    @pytest.mark.parametrize(
        "op",
        [
            operator.add,
            operator.sub,
            operator.mul,
            operator.floordiv,
            operator.truediv,
        ],
    )
    @pytest.mark.parametrize(
        "idx",
        [
            RangeIndex(0, 10, 1),
            RangeIndex(0, 20, 2),
            RangeIndex(-10, 10, 2),
            RangeIndex(5, -5, -1),
        ],
    )
    @pytest.mark.parametrize("scalar", [-1, 1, 2])
    def test_binops_index_scalar(self, op, idx, scalar):
        # [FUNCTION DOC]
        # Parametrized over the five operators, four range indexes, and scalars minus one, one, two. Verifies the operation of a range index with a scalar equals the operation on the index of its raw values (equivalent index type).
        result = op(idx, scalar)
        expected = op(Index(idx.to_numpy()), scalar)
        tm.assert_index_equal(result, expected, exact="equiv")

    @pytest.mark.parametrize("idx1", [RangeIndex(0, 10, 1), RangeIndex(0, 20, 2)])
    @pytest.mark.parametrize("idx2", [RangeIndex(0, 10, 1), RangeIndex(0, 20, 2)])
    def test_binops_index_pow(self, idx1, idx2):
        # [FUNCTION DOC]
        # Parametrized over two non-negative range indexes for each operand. Verifies power between two range indexes (named differently) equals power between plain indexes of their values; tested separately because NumPy disallows powers of negative integers.
        # numpy does not allow powers of negative integers so test separately
        # https://github.com/numpy/numpy/pull/8127
        idx1 = idx1._rename("foo")
        idx2 = idx2._rename("bar")
        result = pow(idx1, idx2)
        expected = pow(Index(idx1.to_numpy()), Index(idx2.to_numpy()))
        tm.assert_index_equal(result, expected, exact="equiv")

    @pytest.mark.parametrize("idx", [RangeIndex(0, 10, 1), RangeIndex(0, 20, 2)])
    @pytest.mark.parametrize("scalar", [1, 2])
    def test_binops_index_scalar_pow(self, idx, scalar):
        # [FUNCTION DOC]
        # Parametrized over two non-negative range indexes and scalars one and two. Verifies power of a range index by a scalar equals power on the plain index of its values.
        # numpy does not allow powers of negative integers so test separately
        # https://github.com/numpy/numpy/pull/8127
        result = pow(idx, scalar)
        expected = pow(Index(idx.to_numpy()), scalar)
        tm.assert_index_equal(result, expected, exact="equiv")

    # TODO: divmod?
    @pytest.mark.parametrize(
        "op",
        [
            operator.add,
            operator.sub,
            operator.mul,
            operator.floordiv,
            operator.truediv,
            operator.pow,
            operator.mod,
        ],
    )
    def test_arithmetic_with_frame_or_series(self, op):
        # [FUNCTION DOC]
        # Parametrized over add, subtract, multiply, floor divide, true divide, power, and modulo. Verifies that a range index operated with a series returns the same as the operator on a series of the index, and with a two-row frame equals the operator on a frame whose rows are two copies of the index, i.e. the index defers to series and frame (returns not-implemented). A TODO notes divmod is untested.
        # check that we return NotImplemented when operating with Series
        # or DataFrame
        index = RangeIndex(5)
        other = Series(np.random.default_rng(2).standard_normal(5))

        expected = op(Series(index), other)
        result = op(index, other)
        tm.assert_series_equal(result, expected)

        other = pd.DataFrame(np.random.default_rng(2).standard_normal((2, 5)))
        expected = op(pd.DataFrame([index, index]), other)
        result = op(index, other)
        tm.assert_frame_equal(result, expected)

    def test_numeric_compat2(self):
        # [FUNCTION DOC]
        # Verifies range index overrides of numeric operators and exact result types: multiplying by two, adding, subtracting two give range indexes with expected start, stop and step; true division by two gives a float64 index of the corresponding values; division by four equals the same division of an identical range; floor division by one returns the same range; multiplying a range index by itself gives an integer index of products; and a large range index raised to the power two matches the plain index power (compared by values).
        # validate that we are handling the RangeIndex overrides to numeric ops
        # and returning RangeIndex where possible

        idx = RangeIndex(0, 10, 2)

        result = idx * 2
        expected = RangeIndex(0, 20, 4)
        tm.assert_index_equal(result, expected, exact=True)

        result = idx + 2
        expected = RangeIndex(2, 12, 2)
        tm.assert_index_equal(result, expected, exact=True)

        result = idx - 2
        expected = RangeIndex(-2, 8, 2)
        tm.assert_index_equal(result, expected, exact=True)

        result = idx / 2
        expected = RangeIndex(0, 5, 1).astype("float64")
        tm.assert_index_equal(result, expected, exact=True)

        result = idx / 4
        expected = RangeIndex(0, 10, 2) / 4
        tm.assert_index_equal(result, expected, exact=True)

        result = idx // 1
        expected = idx
        tm.assert_index_equal(result, expected, exact=True)

        # __mul__
        result = idx * idx
        expected = Index(idx.values * idx.values)
        tm.assert_index_equal(result, expected, exact=True)

        # __pow__
        idx = RangeIndex(0, 1000, 2)
        result = idx**2
        expected = Index(idx._values) ** 2
        tm.assert_index_equal(Index(result.values), expected, exact=True)

    @pytest.mark.parametrize(
        "idx, div, expected",
        [
            # TODO: add more dtypes
            (RangeIndex(0, 1000, 2), 2, RangeIndex(0, 500, 1)),
            (RangeIndex(-99, -201, -3), -3, RangeIndex(33, 67, 1)),
            (
                RangeIndex(0, 1000, 1),
                2,
                Index(RangeIndex(0, 1000, 1)._values) // 2,
            ),
            (
                RangeIndex(0, 100, 1),
                2.0,
                Index(RangeIndex(0, 100, 1)._values) // 2.0,
            ),
            (RangeIndex(0), 50, RangeIndex(0)),
            (RangeIndex(2, 4, 2), 3, RangeIndex(0, 1, 1)),
            (RangeIndex(-5, -10, -6), 4, RangeIndex(-2, -1, 1)),
            (RangeIndex(-100, -200, 3), 2, RangeIndex(0)),
        ],
    )
    def test_numeric_compat2_floordiv(self, idx, div, expected):
        # [FUNCTION DOC]
        # Parametrized over (range index, divisor, expected) triples including negative ranges, an empty range, a float divisor, and cases where the result is not a range. Verifies floor division of the range index by the divisor equals the expected index exactly (including type).
        # __floordiv__
        tm.assert_index_equal(idx // div, expected, exact=True)

    @pytest.mark.parametrize("dtype", [np.int64, np.float64])
    @pytest.mark.parametrize("delta", [1, 0, -1])
    def test_addsub_arithmetic(self, dtype, delta):
        # [FUNCTION DOC]
        # Linked to an issue. Parametrized over dtype (int64 or float64) and delta of one, zero, minus one. For an index of ten, eleven, twelve, verifies addition and subtraction of the delta equal the raw-value results with the same dtype, that adding the index to itself equals doubling it, that subtracting it from itself equals zero times it, and that the latter is not empty.
        # GH#8142
        delta = dtype(delta)
        index = Index([10, 11, 12], dtype=dtype)
        result = index + delta
        expected = Index(index.values + delta, dtype=dtype)
        tm.assert_index_equal(result, expected)

        # this subtraction used to fail
        result = index - delta
        expected = Index(index.values - delta, dtype=dtype)
        tm.assert_index_equal(result, expected)

        tm.assert_index_equal(index + index, 2 * index)
        tm.assert_index_equal(index - index, 0 * index)
        assert not (index - index).empty

    def test_pow_nan_with_zero(self, box_with_array):
        # [FUNCTION DOC]
        # Uses the box fixture. Verifies that raising an index of NaN to the power of an index of zeros gives ones as floats in any box.
        left = Index([np.nan, np.nan, np.nan])
        right = Index([0, 0, 0])
        expected = Index([1.0, 1.0, 1.0])

        left = tm.box_expected(left, box_with_array)
        right = tm.box_expected(right, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = left**right
        tm.assert_equal(result, expected)


def test_fill_value_inf_masking():
    # [FUNCTION DOC]
    # Linked to an issue. Verifies reflected floor division of a frame (columns with zero and a missing value) by another frame with a different index, with fill value one, produces infinity where the numerator is zero rather than NaN, and the expected union-indexed frame (index type check disabled).
    # GH #27464 make sure we mask 0/1 with Inf and not NaN
    df = pd.DataFrame({"A": [0, 1, 2], "B": [1.1, None, 1.1]})

    other = pd.DataFrame({"A": [1.1, 1.2, 1.3]}, index=[0, 2, 3])

    result = df.rfloordiv(other, fill_value=1)

    expected = pd.DataFrame(
        {"A": [np.inf, 1.0, 0.0, 1.0], "B": [0.0, np.nan, 0.0, np.nan]}
    )
    tm.assert_frame_equal(result, expected, check_index_type=False)


def test_dataframe_div_silenced():
    # [FUNCTION DOC]
    # Linked to an issue. Verifies that dividing two frames of different shape and labels with a fill value of zero, including all-NaN columns, emits no warnings.
    # GH#26793
    pdf1 = pd.DataFrame(
        {
            "A": np.arange(10),
            "B": [np.nan, 1, 2, 3, 4] * 2,
            "C": [np.nan] * 10,
            "D": np.arange(10),
        },
        index=list("abcdefghij"),
        columns=list("ABCD"),
    )
    pdf2 = pd.DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        index=list("abcdefghjk"),
        columns=list("ABCX"),
    )
    with tm.assert_produces_warning(None):
        pdf1.div(pdf2, fill_value=0)


@pytest.mark.parametrize(
    "data, expected_data",
    [([0, 1, 2], [0, 2, 4])],
)
@pytest.mark.parametrize("box_pandas_1d_array", [Index, Series, tm.to_array])
@pytest.mark.parametrize("box_1d_array", [Index, Series, tm.to_array, np.array, list])
def test_integer_array_add_list_like(
    box_pandas_1d_array, box_1d_array, data, expected_data
):
    # [FUNCTION DOC]
    # Linked to an issue. Parametrized over the container used for the nullable-Int64 side (index, series, array helper) and over the list-like on the other side (index, series, array helper, NumPy array, list), with data zero, one, two and expected zero, two, four. Verifies adding the Int64 container to the list-like in either order gives expected values as Int64, boxed as a series if either is a series, else an index if either is an index, else a nullable array.
    # GH22606 Verify operators with IntegerArray and list-likes
    arr = array(data, dtype="Int64")
    container = box_pandas_1d_array(arr)
    left = container + box_1d_array(data)
    right = box_1d_array(data) + container

    if Series in [box_1d_array, box_pandas_1d_array]:
        cls = Series
    elif Index in [box_1d_array, box_pandas_1d_array]:
        cls = Index
    else:
        cls = array

    expected = cls(expected_data, dtype="Int64")

    tm.assert_equal(left, expected)
    tm.assert_equal(right, expected)


def test_sub_multiindex_swapped_levels():
    # [FUNCTION DOC]
    # Linked to an issue. Verifies subtracting a frame from a copy of itself whose multi-index levels are swapped aligns on level names and gives all zeros in the original index.
    # GH 9952
    df = pd.DataFrame(
        {"a": np.random.default_rng(2).standard_normal(6)},
        index=pd.MultiIndex.from_product(
            [["a", "b"], [0, 1, 2]], names=["levA", "levB"]
        ),
    )
    df2 = df.copy()
    df2.index = df2.index.swaplevel(0, 1)
    result = df - df2
    expected = pd.DataFrame([0.0] * 6, columns=["a"], index=df.index)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("power", [1, 2, 5])
@pytest.mark.parametrize("string_size", [0, 1, 2, 5])
def test_empty_str_comparison(power, string_size):
    # [FUNCTION DOC]
    # Linked to an issue. Parametrized over sizes ten, one hundred, one thousand and strings of 0, 1, 2, 5 spaces. Verifies comparing an int64 frame with a string for equality gives an all-false frame of the same shape.
    # GH 37348
    a = np.array(range(10**power))
    right = pd.DataFrame(a, dtype=np.int64)
    left = " " * string_size

    result = right == left
    expected = pd.DataFrame(np.zeros(right.shape, dtype=bool))
    tm.assert_frame_equal(result, expected)


def test_series_add_sub_with_UInt64():
    # [FUNCTION DOC]
    # Linked to an issue. Verifies adding and subtracting an integer series and an unsigned nullable UInt64 series of the same length gives a nullable Float64 series (three, three, six and minus one, one, zero).
    # GH 22023
    series1 = Series([1, 2, 3])
    series2 = Series([2, 1, 3], dtype="UInt64")

    result = series1 + series2
    expected = Series([3, 3, 6], dtype="Float64")
    tm.assert_series_equal(result, expected)

    result = series1 - series2
    expected = Series([-1, 1, 0], dtype="Float64")
    tm.assert_series_equal(result, expected)
