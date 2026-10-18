# [MODULE DOC]
# Tests for Series arithmetic and comparison: flex methods versus operators, fill values, alignment (period, datetime, timezone, categorical, duplicate and multi-level indexes), nullable and masked dtypes, list operands, comparison semantics (different lengths, categorical, tuples, comparisons with None), name retention, in-place operators, and datetime unit mismatches.
# Defines an automatically applied fixture running every test twice, once forcing the numexpr-accelerated evaluation path (minimum element threshold zero) and once the plain Python path (very large threshold), and a helper that shuffles an object. Some tests belong in the arithmetic test package but are not yet sorted. No other import-time side effects.

from datetime import (
    date,
    timedelta,
    timezone,
)
from decimal import Decimal
from enum import (
    Enum,
    auto,
)
import operator

import numpy as np
import pytest

from pandas._libs import lib

import pandas as pd
from pandas import (
    Categorical,
    DatetimeTZDtype,
    Index,
    Series,
    Timedelta,
    bdate_range,
    date_range,
    isna,
)
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr


@pytest.fixture(autouse=True, params=[0, 1000000], ids=["numexpr", "python"])
def switch_numexpr_min_elements(request, monkeypatch):
    # [FUNCTION DOC]
    # HELPER: automatically applied fixture parametrized with minimum-element thresholds zero (ids numexpr) and one million (ids python). For each test it temporarily overrides the threshold in the expression evaluation module via the monkeypatch fixture, yields, and restores it afterwards.
    with monkeypatch.context() as m:
        m.setattr(expr, "_MIN_ELEMENTS", request.param)
        yield


def _permute(obj):
    # [FUNCTION DOC]
    # HELPER: returns the given object reordered by a random permutation of its positions (seeded generator, fixed seed), using positional take.
    return obj.take(np.random.default_rng(2).permutation(len(obj)))


class TestSeriesFlexArithmetic:
    # [CLASS DOC]
    # Groups tests of flex arithmetic methods (named methods such as add and reverse variants) of Series. Includes class-level code that builds the list of (method, equivalent operator, fill value) triples used by the combine test.
    @pytest.mark.parametrize(
        "ts",
        [
            (lambda x: x, lambda x: x * 2, False),
            (lambda x: x, lambda x: x[::2], False),
            (lambda x: x, lambda x: 5, True),
            (
                lambda x: Series(range(10), dtype=np.float64),
                lambda x: Series(range(10), dtype=np.float64),
                True,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "opname", ["add", "sub", "mul", "floordiv", "truediv", "pow"]
    )
    def test_flex_method_equivalence(self, opname, ts):
        # [FUNCTION DOC]
        # Verifies that each flex method (add, sub, mul, floordiv, truediv, pow) gives results almost equal to the corresponding operator for four pairings of operands: the same Series and its double, the Series and a strided slice, the Series and a scalar (also checking the reversed method against the operator with swapped operands), and two identical float range Series (also checking reversed). Parametrized over method name and pairing.
        # check that Series.{opname} behaves like Series.__{opname}__,
        tser = Series(
            np.arange(20, dtype=np.float64),
            index=date_range("2020-01-01", periods=20),
            name="ts",
        )

        series = ts[0](tser)
        other = ts[1](tser)
        check_reverse = ts[2]

        op = getattr(Series, opname)
        alt = getattr(operator, opname)

        result = op(series, other)
        expected = alt(series, other)
        tm.assert_almost_equal(result, expected)
        if check_reverse:
            rop = getattr(Series, "r" + opname)
            result = rop(series, other)
            expected = alt(other, series)
            tm.assert_almost_equal(result, expected)

    def test_flex_method_subclass_metadata_preservation(self, all_arithmetic_operators):
        # [FUNCTION DOC]
        # Verifies (issue 13208) that every arithmetic operator (fixture of operator names) called on a Series subclass with a metadata attribute preserves the metadata (value 42) in the result, since the subclass constructor hook returns the subclass.
        # GH 13208
        class MySeries(Series):
            # [CLASS DOC]
            # HELPER: local Series subclass declaring one extra metadata attribute x and a constructor hook returning itself.
            _metadata = ["x"]

            @property
            def _constructor(self):
                # [FUNCTION DOC]
                # HELPER: property returning the local subclass as the constructor for derived objects.
                return MySeries

        opname = all_arithmetic_operators
        op = getattr(Series, opname)
        m = MySeries([1, 2, 3], name="test")
        m.x = 42
        result = op(m, 1)
        assert result.x == 42

    def test_flex_add_scalar_fill_value(self):
        # [FUNCTION DOC]
        # Verifies (issue 12723) that the add method with a scalar and fill value zero equals filling missing values with zero first and then adding.
        # GH12723
        ser = Series([0, 1, np.nan, 3, 4, 5])

        exp = ser.fillna(0).add(2)
        res = ser.add(2, fill_value=0)
        tm.assert_series_equal(res, exp)

    pairings = [(Series.div, operator.truediv, 1), (Series.rdiv, ops.rtruediv, 1)]
    for op in ["add", "sub", "mul", "pow", "truediv", "floordiv"]:
        fv = 0
        lop = getattr(Series, op)
        lequiv = getattr(operator, op)
        rop = getattr(Series, "r" + op)
        # bind op at definition time...
        requiv = lambda x, y, op=op: getattr(operator, op)(y, x)
        pairings.append((lop, lequiv, fv))
        pairings.append((rop, requiv, fv))

    @pytest.mark.parametrize("op, equiv_op, fv", pairings)
    def test_operators_combine(self, op, equiv_op, fv):
        # [FUNCTION DOC]
        # Verifies that each flex method (divide variants plus add, sub, mul, pow, truediv, floordiv and their reverse versions, parametrized together with the equivalent operator function and a fill value) equals the equivalent operator on two partially overlapping float Series with NaN, and equals a manual elementwise computation when a fill value is given. Also checks the method accepts an axis argument of zero.
        def _check_fill(meth, op, a, b, fill_value=0):
            # [FUNCTION DOC]
            # HELPER: nested checker taking a method, operator function, two Series and a fill value (default zero). Reindexes both inputs to their index union, builds the expected values elementwise (NaN where both missing; the operator applied with the fill value substituted for a single missing side; plain operator otherwise), calls the method with the fill value and compares the result with the expected Series.
            # KNOWN ISSUE: in the branch handling a missing right operand, the inner check on the left operand being missing can never be true because the left-missing case was already handled by the previous branch, so the nested NaN append is dead code.
            exp_index = a.index.union(b.index)
            a = a.reindex(exp_index)
            b = b.reindex(exp_index)

            amask = isna(a)
            bmask = isna(b)

            exp_values = []
            for i in range(len(exp_index)):
                with np.errstate(all="ignore"):
                    if amask[i]:
                        if bmask[i]:
                            exp_values.append(np.nan)
                            continue
                        exp_values.append(op(fill_value, b[i]))
                    elif bmask[i]:
                        if amask[i]:
                            exp_values.append(np.nan)
                            continue
                        exp_values.append(op(a[i], fill_value))
                    else:
                        exp_values.append(op(a[i], b[i]))

            result = meth(a, b, fill_value=fill_value)
            expected = Series(exp_values, exp_index)
            tm.assert_series_equal(result, expected)

        a = Series([np.nan, 1.0, 2.0, 3.0, np.nan], index=np.arange(5))
        b = Series([np.nan, 1, np.nan, 3, np.nan, 4.0], index=np.arange(6))

        result = op(a, b)
        exp = equiv_op(a, b)
        tm.assert_series_equal(result, exp)
        _check_fill(op, equiv_op, a, b, fill_value=fv)
        # should accept axis=0 or axis='rows'
        op(a, b, axis=0)

    @pytest.mark.parametrize("kind", ["datetime", "timedelta"])
    def test_rhs_extension_array_sub_with_fill_value(self, kind):
        # [FUNCTION DOC]
        # Verifies (issue 62467) that subtraction with a fill value works when the right operand is a datetime or timedelta extension array (obtained from the left Series' values): subtracting the array from the Series, filling with the first element, gives an all-zero timedelta Series. Parametrized over datetime and timedelta kinds.
        # GH:62467
        if kind == "datetime":
            left = Series(
                [pd.Timestamp("2025-08-20"), pd.Timestamp("2025-08-21")],
                dtype=np.dtype("datetime64[ns]"),
            )
        else:
            left = Series(
                [Timedelta(days=1), Timedelta(days=2)],
                dtype=np.dtype("timedelta64[ns]"),
            )

        right = (
            left._values
        )  # DatetimeArray or TimedeltaArray which is an ExtensionArray

        result = left.sub(right, fill_value=left.iloc[0])
        expected = Series(np.zeros(len(left), dtype=np.dtype("timedelta64[ns]")))
        tm.assert_series_equal(result, expected)

    def test_flex_disallows_dataframe(self):
        # [FUNCTION DOC]
        # Verifies (issue 46179) that the flex add method rejects a DataFrame operand with a type error saying a DataFrame is not supported, using a Series and frame with multi-level indexes.
        # GH#46179
        df = pd.DataFrame(
            {2010: [1], 2020: [3]},
            index=pd.MultiIndex.from_product([["a"], ["b"]], names=["scen", "mod"]),
        )

        ser = Series(
            [10.0, 20.0, 30.0],
            index=pd.MultiIndex.from_product(
                [["a"], ["b"], [0, 1, 2]], names=["scen", "mod", "id"]
            ),
        )

        msg = "Series.add does not support a DataFrame `other`"
        with pytest.raises(TypeError, match=msg):
            ser.add(df, axis=0)


class TestSeriesArithmetic:
    # [CLASS DOC]
    # Groups tests of Series arithmetic via operators and assorted corner cases that may later move to the arithmetic test package. Stateless.
    # Some of these may end up in tests/arithmetic, but are not yet sorted

    def test_add_series_with_period_index(self):
        # [FUNCTION DOC]
        # Verifies adding Series on a yearly period index: adding to a strided half gives the full-sum with NaN at the unmatched positions, also when the strided half is randomly shuffled first.
        rng = pd.period_range("1/1/2000", "1/1/2010", freq="Y")
        ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

        result = ts + ts[::2]
        expected = ts + ts
        expected.iloc[1::2] = np.nan
        tm.assert_series_equal(result, expected)

        result = ts + _permute(ts[::2])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "target_add,input_value,expected_value",
        [
            ("!", ["hello", "world"], ["hello!", "world!"]),
            ("m", ["hello", "world"], ["hellom", "worldm"]),
        ],
    )
    def test_string_addition(self, target_add, input_value, expected_value):
        # [FUNCTION DOC]
        # Verifies (issue 28658) adding a string scalar to a string Series appends it elementwise ("!" and "m" cases), without error. Parametrized over added text, input and expected output.
        # GH28658 - ensure adding 'm' does not raise an error
        a = Series(input_value)

        result = a + target_add
        expected = Series(expected_value)
        tm.assert_series_equal(result, expected)

    def test_divmod(self):
        # [FUNCTION DOC]
        # Verifies (issue 25557) the divmod method and reflected divmod method on partially overlapping float Series equal the built-in divmod on the operands (swapped for the reflected one), comparing quotient and remainder.
        # GH#25557
        a = Series([1, 1, 1, np.nan], index=["a", "b", "c", "d"])
        b = Series([2, np.nan, 1, np.nan], index=["a", "b", "d", "e"])

        result = a.divmod(b)
        expected = divmod(a, b)
        tm.assert_series_equal(result[0], expected[0])
        tm.assert_series_equal(result[1], expected[1])

        result = a.rdivmod(b)
        expected = divmod(b, a)
        tm.assert_series_equal(result[0], expected[0])
        tm.assert_series_equal(result[1], expected[1])

    @pytest.mark.parametrize("index", [None, range(9)])
    def test_series_integer_mod(self, index):
        # [FUNCTION DOC]
        # Verifies (issue 24396) that a string Series modulo an integer Series raises a type error (string formatting message or unsupported mod), with default and explicit range indexes.
        # GH#24396
        s1 = Series(range(1, 10))
        s2 = Series("foo", index=index)

        msg = "not all arguments converted during string formatting|'mod' not supported"

        with pytest.raises(TypeError, match=msg):
            s2 % s1

    def test_add_with_duplicate_index(self):
        # [FUNCTION DOC]
        # Verifies (issue 14227) adding two Series where the left has a duplicated index label: matches by label, the duplicated entries each pair with the one right value and unmatched labels give NaN.
        # GH14227
        s1 = Series([1, 2], index=[1, 1])
        s2 = Series([10, 10], index=[1, 2])
        result = s1 + s2
        expected = Series([11, 12, np.nan], index=[1, 1, 2])
        tm.assert_series_equal(result, expected)

    def test_add_na_handling(self):
        # [FUNCTION DOC]
        # Verifies that adding a decimal-valued Series to its shifted self, in either order, gives a missing first element.
        ser = Series(
            [Decimal("1.3"), Decimal("2.3")], index=[date(2012, 1, 1), date(2012, 1, 2)]
        )

        result = ser + ser.shift(1)
        result2 = ser.shift(1) + ser
        assert isna(result.iloc[0])
        assert isna(result2.iloc[0])

    def test_add_corner_cases(self, datetime_series):
        # [FUNCTION DOC]
        # Verifies that adding an empty float Series to the datetime fixture gives all NaN, and adding two empty Series gives an empty result.
        empty = Series([], index=Index([]), dtype=np.float64)

        result = datetime_series + empty
        assert np.isnan(result).all()

        result = empty + empty
        assert len(result) == 0

    def test_add_float_plus_int(self, datetime_series):
        # [FUNCTION DOC]
        # Verifies adding the float datetime fixture to a copy cast to integer with its last five entries dropped: over the shared leading index the sum equals the elementwise sum of the float values and the integer values, with the series name kept.
        # float + int
        int_ts = datetime_series.astype(int)[:-5]
        added = datetime_series + int_ts
        expected = Series(
            datetime_series.values[:-5] + int_ts.values,
            index=datetime_series.index[:-5],
            name="ts",
        )
        tm.assert_series_equal(added[:-5], expected)

    def test_mul_empty_int_corner_case(self):
        # [FUNCTION DOC]
        # Verifies that multiplying an empty int32 Series by a one-element float Series gives a single NaN labeled x.
        s1 = Series([], [], dtype=np.int32)
        s2 = Series({"x": 0.0})
        tm.assert_series_equal(s1 * s2, Series([np.nan], index=["x"]))

    def test_sub_datetimelike_align(self):
        # [FUNCTION DOC]
        # Verifies (issue 7500) that subtracting datetime Series aligns on index: the reversed copy minus the original gives timedeltas zero, zero and NaT, with the name reset. Doing the same through one-column frames and taking column zero gives an equal result with name zero.
        # GH#7500
        # datetimelike ops need to align
        dt = Series(date_range("2012-1-1", periods=3, freq="D", unit="ns"))
        dt.iloc[2] = np.nan
        dt2 = dt[::-1]

        expected = Series([timedelta(0), timedelta(0), pd.NaT], dtype="m8[ns]")
        # name is reset
        result = dt2 - dt
        tm.assert_series_equal(result, expected)

        expected = Series(expected, name=0)
        result = (dt2.to_frame() - dt.to_frame())[0]
        tm.assert_series_equal(result, expected)

    def test_alignment_doesnt_change_tz(self):
        # [FUNCTION DOC]
        # Verifies (issue 33671) that multiplying Series with the same instants in different time zones does not modify the operand index objects (identity retained).
        # GH#33671
        dti = date_range("2016-01-01", periods=10, tz="CET")
        dti_utc = dti.tz_convert("UTC")
        ser = Series(10, index=dti)
        ser_utc = Series(10, index=dti_utc)

        # we don't care about the result, just that original indexes are unchanged
        ser * ser_utc

        assert ser.index is dti
        assert ser_utc.index is dti_utc

    def test_alignment_categorical(self):
        # [FUNCTION DOC]
        # Verifies (issue 13365) that multiplying Series indexed by categorical values with duplicates, where the right is shorter by one, aligns the duplicates as a cross product (4, 9 and 1 entries) with NaN for the unmatched category, and the result has a categorical index.
        # GH13365
        cat = Categorical(["3z53", "3z53", "LoJG", "LoJG", "LoJG", "N503"])
        ser1 = Series(2, index=cat)
        ser2 = Series(2, index=cat[:-1])
        result = ser1 * ser2

        exp_index = ["3z53"] * 4 + ["LoJG"] * 9 + ["N503"]
        exp_index = pd.CategoricalIndex(exp_index, categories=cat.categories)
        exp_values = [4.0] * 13 + [np.nan]
        expected = Series(exp_values, exp_index)

        tm.assert_series_equal(result, expected)

    def test_arithmetic_with_duplicate_index(self):
        # [FUNCTION DOC]
        # Verifies (issue 8363) subtraction of Series with a non-unique index for integers (result one everywhere) and for datetimes (result a constant 9-hour timedelta), preserving the duplicate index.
        # GH#8363
        # integer ops with a non-unique index
        index = [2, 2, 3, 3, 4]
        ser = Series(np.arange(1, 6, dtype="int64"), index=index)
        other = Series(np.arange(5, dtype="int64"), index=index)
        result = ser - other
        expected = Series(1, index=[2, 2, 3, 3, 4])
        tm.assert_series_equal(result, expected)

        # GH#8363
        # datetime ops with a non-unique index
        ser = Series(date_range("20130101 09:00:00", periods=5, unit="ns"), index=index)
        other = Series(date_range("20130101", periods=5, unit="ns"), index=index)
        result = ser - other
        expected = Series(Timedelta("9 hours"), index=[2, 2, 3, 3, 4], dtype="m8[ns]")
        tm.assert_series_equal(result, expected)

    def test_masked_and_non_masked_propagate_na(self):
        # [FUNCTION DOC]
        # Verifies (issue 45810) that multiplying a float Series with NaN by a nullable Int64 Series gives a nullable Float64 result in which NaN becomes the missing value.
        # GH#45810
        ser1 = Series([0, np.nan], dtype="float")
        ser2 = Series([0, 1], dtype="Int64")
        result = ser1 * ser2
        expected = Series([0, pd.NA], dtype="Float64")
        tm.assert_series_equal(result, expected)

    def test_mask_div_propagate_na_for_non_na_dtype(self):
        # [FUNCTION DOC]
        # Verifies (issue 42630) true division between nullable Int64 and a float Series with NaN yields nullable Float64 with missing values where either side is missing, in both operand orders.
        # GH#42630
        ser1 = Series([15, pd.NA, 5, 4], dtype="Int64")
        ser2 = Series([15, 5, np.nan, 4])
        result = ser1 / ser2
        expected = Series([1.0, pd.NA, pd.NA, 1.0], dtype="Float64")
        tm.assert_series_equal(result, expected)

        result = ser2 / ser1
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("val", [3, 3.5])
    def test_add_list_to_masked_array(self, val):
        # [FUNCTION DOC]
        # Verifies (issues 22962, 62552) that adding a list with None and a number (integer or float, parametrized) to a nullable Int64 Series gives nullable Float64 with a missing value, in both operand orders.
        # GH#22962, behavior changed by GH#62552
        ser = Series([1, None, 3], dtype="Int64")
        result = ser + [1, None, val]  # noqa: RUF005
        expected = Series([2, pd.NA, 3 + val], dtype="Float64")
        tm.assert_series_equal(result, expected)

        result = [1, None, val] + ser  # noqa: RUF005
        tm.assert_series_equal(result, expected)

    def test_add_list_to_masked_array_boolean(self):
        # [FUNCTION DOC]
        # Verifies (issue 22962) that adding a list of booleans with None to a nullable boolean Series gives an object Series of counts and a missing marker, in both operand orders.
        # GH#22962
        ser = Series([True, None, False], dtype="boolean")
        result = ser + [True, None, True]  # noqa: RUF005
        expected = Series([2, pd.NA, 1], dtype=object)
        tm.assert_series_equal(result, expected)

        result = [True, None, True] + ser  # noqa: RUF005
        tm.assert_series_equal(result, expected)


# ------------------------------------------------------------------
# Comparisons


class TestSeriesFlexComparison:
    # [CLASS DOC]
    # Groups tests of flex comparison methods (eq, ne, lt, le, gt, ge) on Series, covering axis handling, alignment with and without fill values, extension-array operands and objects. Stateless.
    @pytest.mark.parametrize("axis", [0, None, "index"])
    def test_comparison_flex_basic(self, axis, comparison_op):
        # [FUNCTION DOC]
        # Verifies that each comparison flex method (from the comparison-operator fixture), with axis zero, none or the label index, equals the operator on two random Series.
        left = Series(np.random.default_rng(2).standard_normal(10))
        right = Series(np.random.default_rng(2).standard_normal(10))
        result = getattr(left, comparison_op.__name__)(right, axis=axis)
        expected = comparison_op(left, right)
        tm.assert_series_equal(result, expected)

    def test_comparison_bad_axis(self, comparison_op):
        # [FUNCTION DOC]
        # Verifies that a flex comparison with axis one raises a value error that no such axis exists for the object type.
        left = Series(np.random.default_rng(2).standard_normal(10))
        right = Series(np.random.default_rng(2).standard_normal(10))

        msg = "No axis named 1 for object type"
        with pytest.raises(ValueError, match=msg):
            getattr(left, comparison_op.__name__)(right, axis=1)

    @pytest.mark.parametrize(
        "values, op",
        [
            ([False, False, True, False], "eq"),
            ([True, True, False, True], "ne"),
            ([False, False, True, False], "le"),
            ([False, False, False, False], "lt"),
            ([False, True, True, False], "ge"),
            ([False, True, False, False], "gt"),
        ],
    )
    def test_comparison_flex_alignment(self, values, op):
        # [FUNCTION DOC]
        # Verifies that flex comparisons between Series with different indexes align on the union of labels with unmatched positions False for the six comparison names (the expectations are given per operator); parametrized over expected values and operator name.
        left = Series([1, 3, 2], index=list("abc"))
        right = Series([2, 2, 2], index=list("bcd"))
        result = getattr(left, op)(right)
        expected = Series(values, index=list("abcd"))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "left",
        [
            Series(Categorical(["a", "b", "a"])),
            Series(pd.period_range("2020Q1", periods=3, freq="Q")),
        ],
        ids=["categorical", "period"],
    )
    def test_rhs_extension_array_eq_with_fill_value(self, left):
        # [FUNCTION DOC]
        # Verifies (issue 62467) that the equality method with a fill value works when the right operand is an extension array (categorical or period values) taken from the left Series: all True. Parametrized over the two Series.
        # GH:#62467
        right = left._values  # this is an ExtensionArray

        result = left.eq(right, fill_value=left.iloc[0])
        expected = Series([True, True, True])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "values, op, fill_value",
        [
            ([False, False, True, True], "eq", 2),
            ([True, True, False, False], "ne", 2),
            ([False, False, True, True], "le", 0),
            ([False, False, False, True], "lt", 0),
            ([True, True, True, False], "ge", 0),
            ([True, True, False, False], "gt", 0),
        ],
    )
    def test_comparison_flex_alignment_fill(self, values, op, fill_value):
        # [FUNCTION DOC]
        # Verifies flex comparison with a fill value on misaligned indexes: missing positions are filled with the given value before comparison, for the six operators with the fill value and expected values given per case.
        left = Series([1, 3, 2], index=list("abc"))
        right = Series([2, 2, 2], index=list("bcd"))
        result = getattr(left, op)(right, fill_value=fill_value)
        expected = Series(values, index=list("abcd"))
        tm.assert_series_equal(result, expected)

    def test_eq_objects(self) -> None:
        # [FUNCTION DOC]
        # Verifies (issue 62191) equality on object Series of enum members with an enum member, a list, an array and a Series (all positional unless Series), and that for a Series of lists compared to a list, each element is compared as a whole, giving all False.
        # KNOWN ISSUE: the end of the test first asserts that the result differs from a wrongly expected value by requiring an assertion failure, which documents a surprising behavior rather than directly asserting the intended one.
        # GH#62191 Test eq with Enum and List elements

        class Thing(Enum):
            # [CLASS DOC]
            # HELPER: local enumeration with two automatically numbered members, used as object data.
            FIRST = auto()
            SECOND = auto()

        left = Series([Thing.FIRST, Thing.SECOND])
        py_l = [Thing.FIRST, Thing.SECOND]

        result = left.eq(Thing.FIRST)
        expected = Series([True, False])
        tm.assert_series_equal(result, expected)

        result = left.eq(py_l)
        expected = Series([True, True])
        tm.assert_series_equal(result, expected)

        result = left.eq(np.asarray(py_l))
        expected = Series([True, True])
        tm.assert_series_equal(result, expected)

        result = left.eq(Series(py_l))
        expected = Series([True, True])
        tm.assert_series_equal(result, expected)

        result = Series([[1, 2], [3, 4]]).eq([1, 2])
        expected = Series([True, False])
        with pytest.raises(AssertionError):
            tm.assert_series_equal(result, expected)
        expected = Series([False, False])
        tm.assert_series_equal(result, expected)

    def test_eq_with_index(self) -> None:
        # [FUNCTION DOC]
        # Verifies (issue 62191) equality with a non-default index: a list or array operand is treated as positional and the result keeps the Series' index; a Series operand aligns by label, so a Series of the same values with default labels gives all False with a default index, while a Series with the values in reversed order (matching labels) gives all True.
        # GH#62191 Test eq with non-trivial indices
        left = Series([1, 2], index=[1, 0])
        py_l = [1, 2]

        # assuming Python list has the same index as the Series
        result = left.eq(py_l)
        expected = Series([True, True], index=[1, 0])
        tm.assert_series_equal(result, expected)

        # assuming np.ndarray has the same index as the Series
        result = left.eq(np.asarray(py_l))
        expected = Series([True, True], index=[1, 0])
        tm.assert_series_equal(result, expected)

        result = left.eq(Series(py_l))
        expected = Series([False, False])
        tm.assert_series_equal(result, expected)

        result = left.eq(Series([2, 1]))
        expected = Series([True, True])
        tm.assert_series_equal(result, expected)


class TestSeriesComparison:
    # [CLASS DOC]
    # Groups tests of comparison operators on Series (identical labeling requirements, categoricals, tuples, None, frozensets, names). Stateless.
    def test_comparison_different_length(self):
        # [FUNCTION DOC]
        # Verifies that comparing Series of different length or labels with an operator raises a value error saying only identically-labeled Series can be compared (less-than and equality cases).
        a = Series(["a", "b", "c"])
        b = Series(["b", "a"])
        msg = "only compare identically-labeled Series"
        with pytest.raises(ValueError, match=msg):
            a < b

        a = Series([1, 2])
        b = Series([2, 3, 4])
        with pytest.raises(ValueError, match=msg):
            a == b

    @pytest.mark.parametrize("opname", ["eq", "ne", "gt", "lt", "ge", "le"])
    def test_ser_flex_cmp_return_dtypes(self, opname):
        # [FUNCTION DOC]
        # Verifies (issue 15115) that each flex comparison method with a scalar yields a boolean dtype.
        # GH#15115
        ser = Series([1, 3, 2], index=range(3))
        const = 2
        result = getattr(ser, opname)(const).dtypes
        expected = np.dtype("bool")
        assert result == expected

    @pytest.mark.parametrize("opname", ["eq", "ne", "gt", "lt", "ge", "le"])
    def test_ser_flex_cmp_return_dtypes_empty(self, opname):
        # [FUNCTION DOC]
        # Verifies (issue 15115) the same boolean result dtype for an empty Series.
        # GH#15115 empty Series case
        ser = Series([1, 3, 2], index=range(3))
        empty = ser.iloc[:0]
        const = 2
        result = getattr(empty, opname)(const).dtypes
        expected = np.dtype("bool")
        assert result == expected

    @pytest.mark.parametrize(
        "names", [(None, None, None), ("foo", "bar", None), ("baz", "baz", "baz")]
    )
    def test_ser_cmp_result_names(self, names, comparison_op):
        # [FUNCTION DOC]
        # Verifies the result name of comparing a Series with an index: it is the common name if the Series and index names agree, otherwise none (parametrized over three name triples and the comparison operator fixture). Covers datetime, tz-aware datetime, timedelta, and, for equality and inequality only, interval and categorical dtypes.
        # datetime64 dtype
        op = comparison_op
        dti = date_range("1949-06-07 03:00:00", freq="h", periods=5, name=names[0])
        ser = Series(dti).rename(names[1])
        result = op(ser, dti)
        assert result.name == names[2]

        # datetime64tz dtype
        dti = dti.tz_localize("US/Central")
        dti = pd.DatetimeIndex(dti, freq="infer")  # freq not preserved by tz_localize
        ser = Series(dti).rename(names[1])
        result = op(ser, dti)
        assert result.name == names[2]

        # timedelta64 dtype
        tdi = dti - dti.shift(1)
        ser = Series(tdi).rename(names[1])
        result = op(ser, tdi)
        assert result.name == names[2]

        # interval dtype
        if op in [operator.eq, operator.ne]:
            # interval dtype comparisons not yet implemented
            ii = pd.interval_range(start=0, periods=5, name=names[0])
            ser = Series(ii).rename(names[1])
            result = op(ser, ii)
            assert result.name == names[2]

        # categorical
        if op in [operator.eq, operator.ne]:
            # categorical dtype comparisons raise for inequalities
            cidx = tdi.astype("category")
            ser = Series(cidx).rename(names[1])
            result = op(ser, cidx)
            assert result.name == names[2]

    def test_comparisons(self):
        # [FUNCTION DOC]
        # Verifies that comparing a string Series with a boolean Series by equality, in both orders, gives all False without raising.
        s = Series(["a", "b", "c"])
        s2 = Series([False, True, False])

        # it works!
        exp = Series([False, False, False])
        tm.assert_series_equal(s == s2, exp)
        tm.assert_series_equal(s2 == s, exp)

    # -----------------------------------------------------------------
    # Categorical Dtype Comparisons

    def test_categorical_comparisons(self):
        # [FUNCTION DOC]
        # Verifies (issue 8938) categorical comparisons: equality and inequality work against scalars, lists, object Series and categoricals (with positional matching for equal content, differing results for different content); and ordering comparisons between a categorical Series and an object Series raise a type error that only equality can be compared.
        # GH#8938
        # allow equality comparisons
        a = Series(list("abc"), dtype="category")
        b = Series(list("abc"), dtype="object")
        c = Series(["a", "b", "cc"], dtype="object")
        d = Series(list("acb"), dtype="object")
        e = Categorical(list("abc"))
        f = Categorical(list("acb"))

        # vs scalar
        assert not (a == "a").all()
        assert ((a != "a") == ~(a == "a")).all()

        assert not ("a" == a).all()
        assert (a == "a")[0]
        assert ("a" == a)[0]
        assert not ("a" != a)[0]

        # vs list-like
        assert (a == a).all()
        assert not (a != a).all()

        assert (a == list(a)).all()
        assert (a == b).all()
        assert (b == a).all()
        assert ((~(a == b)) == (a != b)).all()
        assert ((~(b == a)) == (b != a)).all()

        assert not (a == c).all()
        assert not (c == a).all()
        assert not (a == d).all()
        assert not (d == a).all()

        # vs a cat-like
        assert (a == e).all()
        assert (e == a).all()
        assert not (a == f).all()
        assert not (f == a).all()

        assert (~(a == e) == (a != e)).all()
        assert (~(e == a) == (e != a)).all()
        assert (~(a == f) == (a != f)).all()
        assert (~(f == a) == (f != a)).all()

        # non-equality is not comparable
        msg = "can only compare equality or not"
        with pytest.raises(TypeError, match=msg):
            a < b
        with pytest.raises(TypeError, match=msg):
            b < a
        with pytest.raises(TypeError, match=msg):
            a > b
        with pytest.raises(TypeError, match=msg):
            b > a

    def test_unequal_categorical_comparison_raises_type_error(self):
        # [FUNCTION DOC]
        # Verifies that ordering comparison of an unordered categorical Series with a scalar raises a type error (only equality allowed); for an ordered one, ordering comparison with a scalar not among categories raises an invalid comparison type error for both operand orders, while equality and inequality simply give all False and all True.
        # unequal comparison should raise for unordered cats
        cat = Series(Categorical(list("abc")))
        msg = "can only compare equality or not"
        with pytest.raises(TypeError, match=msg):
            cat > "b"

        cat = Series(Categorical(list("abc"), ordered=False))
        with pytest.raises(TypeError, match=msg):
            cat > "b"

        # https://github.com/pandas-dev/pandas/issues/9836#issuecomment-92123057
        # and following comparisons with scalars not in categories should raise
        # for unequal comps, but not for equal/not equal
        cat = Series(Categorical(list("abc"), ordered=True))

        msg = "Invalid comparison between dtype=category and str"
        with pytest.raises(TypeError, match=msg):
            cat < "d"
        with pytest.raises(TypeError, match=msg):
            cat > "d"
        with pytest.raises(TypeError, match=msg):
            "d" < cat
        with pytest.raises(TypeError, match=msg):
            "d" > cat

        tm.assert_series_equal(cat == "d", Series([False, False, False]))
        tm.assert_series_equal(cat != "d", Series([True, True, True]))

    # -----------------------------------------------------------------

    def test_comparison_tuples(self):
        # [FUNCTION DOC]
        # Verifies (issue 11339) elementwise equality and inequality of a Series of tuples with a tuple scalar, treating the tuple as a single value, for several tuple values.
        # GH#11339
        # comparisons vs tuple
        s = Series([(1, 1), (1, 2)])

        result = s == (1, 2)
        expected = Series([False, True])
        tm.assert_series_equal(result, expected)

        result = s != (1, 2)
        expected = Series([True, False])
        tm.assert_series_equal(result, expected)

        result = s == (0, 0)
        expected = Series([False, False])
        tm.assert_series_equal(result, expected)

        result = s != (0, 0)
        expected = Series([True, True])
        tm.assert_series_equal(result, expected)

        s = Series([(1, 1), (1, 1)])

        result = s == (1, 1)
        expected = Series([True, True])
        tm.assert_series_equal(result, expected)

        result = s != (1, 1)
        expected = Series([False, False])
        tm.assert_series_equal(result, expected)

    def test_comparison_frozenset(self):
        # [FUNCTION DOC]
        # Verifies equality of a Series of frozensets with a frozenset scalar elementwise.
        ser = Series([frozenset([1]), frozenset([1, 2])])

        result = ser == frozenset([1])
        expected = Series([True, False])
        tm.assert_series_equal(result, expected)

    def test_comparison_operators_with_nas(self, comparison_op):
        # [FUNCTION DOC]
        # Verifies each comparison operator (fixture) on an object Series of dates with NaN at every other position against one date: the result matches the comparison of the non-missing part reindexed, with missing positions False (True for not-equal), as boolean dtype.
        ser = Series(bdate_range("1/1/2000", periods=10), dtype=object)
        ser[::2] = np.nan

        # test that comparisons work
        val = ser[5]

        result = comparison_op(ser, val)
        expected = comparison_op(ser.dropna(), val).reindex(ser.index)

        if comparison_op is operator.ne:
            expected = expected.fillna(True).astype(bool)
        else:
            expected = expected.fillna(False).astype(bool)

        tm.assert_series_equal(result, expected)

    def test_ne(self):
        # [FUNCTION DOC]
        # Verifies that not-equal on an Index of floats against 5, and the negation of equality, both give the expected boolean array.
        # KNOWN ISSUE: the test is named for Series inequality but exercises only the index comparison.
        ts = Series([3, 4, 5, 6, 7], [3, 4, 5, 6, 7], dtype=float)
        expected = np.array([True, True, False, True, True])
        tm.assert_numpy_array_equal(ts.index != 5, expected)
        tm.assert_numpy_array_equal(~(ts.index == 5), expected)

    @pytest.mark.parametrize("right_data", [[2, 2, 2], [2, 2, 2, 2]])
    def test_comp_ops_df_compat(self, right_data, frame_or_series):
        # [FUNCTION DOC]
        # Verifies (issues 1134, 50083) that comparison operators (equal, not-equal, less-than, in both operand orders) between Series, and between one-column frames (parametrized by the frame-or-Series fixture), whose labels differ raise a value error with a message about identically-labeled objects (mentioning index and columns for frames). Parametrized over the right operand data (same or different length).
        # GH 1134
        # GH 50083 to clarify that index and columns must be identically labeled
        left = Series([1, 2, 3], index=list("ABC"), name="x")
        right = Series(right_data, index=list("ABDC")[: len(right_data)], name="x")
        if frame_or_series is not Series:
            msg = (
                rf"Can only compare identically-labeled \(both index and columns\) "
                f"{frame_or_series.__name__} objects"
            )
            left = left.to_frame()
            right = right.to_frame()
        else:
            msg = (
                f"Can only compare identically-labeled {frame_or_series.__name__} "
                f"objects"
            )

        with pytest.raises(ValueError, match=msg):
            left == right
        with pytest.raises(ValueError, match=msg):
            right == left

        with pytest.raises(ValueError, match=msg):
            left != right
        with pytest.raises(ValueError, match=msg):
            right != left

        with pytest.raises(ValueError, match=msg):
            left < right
        with pytest.raises(ValueError, match=msg):
            right < left

    def test_compare_series_interval_keyword(self):
        # [FUNCTION DOC]
        # Verifies (issue 25338) that equality of a string Series with a string that looks like an interval name is a plain elementwise string comparison.
        # GH#25338
        ser = Series(["IntervalA", "IntervalB", "IntervalC"])
        result = ser == "IntervalA"
        expected = Series([True, False, False])
        tm.assert_series_equal(result, expected)


# ------------------------------------------------------------------
# Unsorted
#  These arithmetic tests were previously in other files, eventually
#  should be parametrized and put into tests.arithmetic


class TestTimeSeriesArithmetic:
    # [CLASS DOC]
    # Groups unsorted tests of time-series arithmetic that were moved from other files and should eventually be parametrized into the arithmetic package. Stateless.
    def test_series_add_tz_mismatch_converts_to_utc(self):
        # [FUNCTION DOC]
        # Verifies that adding Series with indexes in different time zones converts both to UTC, giving a UTC-indexed result equal to the sum of UTC-converted Series, sorted by index.
        rng = date_range("1/1/2011", periods=100, freq="h", tz="utc")

        perm = np.random.default_rng(2).permutation(100)[:90]
        ser1 = Series(
            np.random.default_rng(2).standard_normal(90),
            index=rng.take(perm).tz_convert("US/Eastern"),
        )

        perm = np.random.default_rng(2).permutation(100)[:90]
        ser2 = Series(
            np.random.default_rng(2).standard_normal(90),
            index=rng.take(perm).tz_convert("Europe/Berlin"),
        )

        result = ser1 + ser2

        uts1 = ser1.tz_convert("utc")
        uts2 = ser2.tz_convert("utc")
        expected = uts1 + uts2

        # sort since input indexes are not equal
        expected = expected.sort_index()

        assert result.index.tz is timezone.utc
        tm.assert_series_equal(result, expected)

    def test_series_add_aware_naive_raises(self):
        # [FUNCTION DOC]
        # Verifies that adding a tz-aware and a tz-naive indexed Series, in either order, raises an exception stating that tz-naive and tz-aware datetime indexes cannot be joined.
        rng = date_range("1/1/2011", periods=10, freq="h")
        ser = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

        ser_utc = ser.tz_localize("utc")

        msg = "Cannot join tz-naive with tz-aware DatetimeIndex"
        with pytest.raises(Exception, match=msg):
            ser + ser_utc

        with pytest.raises(Exception, match=msg):
            ser_utc + ser

    # TODO: belongs in tests/arithmetic?
    def test_datetime_understood(self, unit):
        # [FUNCTION DOC]
        # Verifies (issue 16726) subtracting a day offset (six days) from a Series of three dates gives dates six days earlier, retaining the unit fixture; carries a note that it belongs in the arithmetic package.
        # Ensures it doesn't fail to create the right series
        # reported in issue#16726
        series = Series(date_range("2012-01-01", periods=3, unit=unit))
        offset = pd.offsets.DateOffset(days=6)
        result = series - offset
        exp_dti = pd.to_datetime(["2011-12-26", "2011-12-27", "2011-12-28"]).as_unit(
            unit
        )
        expected = Series(exp_dti)
        tm.assert_series_equal(result, expected)

    def test_align_date_objects_with_datetimeindex(self):
        # [FUNCTION DOC]
        # Verifies that adding a Series indexed by datetimes to one indexed by plain date objects aligns correctly in either order, with the same result as adding the overlapping slice (with the index frequency cleared).
        rng = date_range("1/1/2000", periods=20)
        ts = Series(np.random.default_rng(2).standard_normal(20), index=rng)

        ts_slice = ts[5:]
        ts2 = ts_slice.copy()
        ts2.index = [x.date() for x in ts2.index]

        result = ts + ts2
        result2 = ts2 + ts
        expected = ts + ts[5:]
        expected.index = expected.index._with_freq(None)
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(result2, expected)


class TestNamePreservation:
    # [CLASS DOC]
    # Groups tests that arithmetic results keep or drop the Series name correctly. Stateless.
    @pytest.mark.parametrize("box", [list, tuple, np.array, Index, Series, pd.array])
    @pytest.mark.parametrize("flex", [True, False])
    def test_series_ops_name_retention(self, flex, box, names, all_binary_operators):
        # [FUNCTION DOC]
        # Verifies (issue 33930) name retention for every binary operator (fixture) with the right operand wrapped as list, tuple, array, Index, Series or extension array, both via flex methods and via operators, using name triples from a fixture. Logical operators have no flex methods so those cases return early; logical operators with list or tuple raise a type error about dtype-less sequences (issue 52264). The result is a Series named by the right operand's name (matching rule) for Index or Series operands, otherwise by the left operand's name.
        # GH#33930 consistent name-retention
        op = all_binary_operators

        left = Series(range(10), name=names[0])
        right = Series(range(10), name=names[1])

        name = op.__name__.strip("_")
        is_logical = name in ["and", "rand", "xor", "rxor", "or", "ror"]

        msg = (
            r"Logical ops \(and, or, xor\) between Pandas objects and "
            "dtype-less sequences"
        )

        right = box(right)
        if flex:
            if is_logical:
                # Series doesn't have these as flex methods
                return
            result = getattr(left, name)(right)
        else:
            if is_logical and box in [list, tuple]:
                with pytest.raises(TypeError, match=msg):
                    # GH#52264 logical ops with dtype-less sequences deprecated
                    op(left, right)
                return
            result = op(left, right)

        assert isinstance(result, Series)
        if box in [Index, Series]:
            assert result.name is names[2] or result.name == names[2]
        else:
            assert result.name is names[0] or result.name == names[0]

    def test_binop_maybe_preserve_name(self, datetime_series):
        # [FUNCTION DOC]
        # Verifies name handling in binary operations on the datetime fixture: equal names keep the name, different names give none, and the same holds for the flex methods (including reverse variants of add, subtract, multiply, divide, true divide, floor divide, modulo and power).
        # names match, preserve
        result = datetime_series * datetime_series
        assert result.name == datetime_series.name
        result = datetime_series.mul(datetime_series)
        assert result.name == datetime_series.name

        result = datetime_series * datetime_series[:-2]
        assert result.name == datetime_series.name

        # names don't match, don't preserve
        cp = datetime_series.copy()
        cp.name = "something else"
        result = datetime_series + cp
        assert result.name is None
        result = datetime_series.add(cp)
        assert result.name is None

        ops = ["add", "sub", "mul", "div", "truediv", "floordiv", "mod", "pow"]
        ops = ops + ["r" + op for op in ops]
        for op in ops:
            # names match, preserve
            ser = datetime_series.copy()
            result = getattr(ser, op)(ser)
            assert result.name == datetime_series.name

            # names don't match, don't preserve
            cp = datetime_series.copy()
            cp.name = "changed"
            result = getattr(ser, op)(cp)
            assert result.name is None

    def test_scalarop_preserve_name(self, datetime_series):
        # [FUNCTION DOC]
        # Verifies that multiplying a Series by a scalar keeps its name.
        result = datetime_series * 2
        assert result.name == datetime_series.name


class TestInplaceOperations:
    # [CLASS DOC]
    # Groups the test of in-place arithmetic operators on Series. Stateless.
    @pytest.mark.parametrize(
        "dtype1, dtype2, dtype_expected, dtype_mul",
        (
            ("Int64", "Int64", "Int64", "Int64"),
            ("float", "float", "float", "float"),
            ("Int64", "float", "Float64", "Float64"),
            ("Int64", "Float64", "Float64", "Float64"),
        ),
    )
    def test_series_inplace_ops(self, dtype1, dtype2, dtype_expected, dtype_mul):
        # [FUNCTION DOC]
        # Verifies (issue 37910) augmented assignment add, subtract and multiply on one-element Series of nullable integer and float combinations: results have the expected dtype after each step (parametrized by the two input dtypes, the dtype after add and subtract, and the dtype after multiply).
        # GH 37910

        ser1 = Series([1], dtype=dtype1)
        ser2 = Series([2], dtype=dtype2)
        ser1 += ser2
        expected = Series([3], dtype=dtype_expected)
        tm.assert_series_equal(ser1, expected)

        ser1 -= ser2
        expected = Series([1], dtype=dtype_expected)
        tm.assert_series_equal(ser1, expected)

        ser1 *= ser2
        expected = Series([2], dtype=dtype_mul)
        tm.assert_series_equal(ser1, expected)


def test_none_comparison(request, series_with_simple_index):
    # [FUNCTION DOC]
    # Verifies comparisons with None on a series with a simple index (fixture): equality gives False and inequality gives True for the first two entries, even after the first is set to NaN, in both operand orders. Ordering comparisons with None raise an invalid comparison type error for datetime-like dtypes and give False otherwise. Marked expected-to-fail when the series is empty.
    series = series_with_simple_index

    if len(series) < 1:
        request.applymarker(
            pytest.mark.xfail(reason="Test doesn't make sense on empty data")
        )

    # bug brought up by #1079
    # changed from TypeError in 0.17.0
    series.iloc[0] = np.nan

    # noinspection PyComparisonWithNone
    result = series == None  # noqa: E711
    assert not result.iat[0]
    assert not result.iat[1]

    # noinspection PyComparisonWithNone
    result = series != None  # noqa: E711
    assert result.iat[0]
    assert result.iat[1]

    result = None == series  # noqa: E711
    assert not result.iat[0]
    assert not result.iat[1]

    result = None != series  # noqa: E711
    assert result.iat[0]
    assert result.iat[1]

    if lib.is_np_dtype(series.dtype, "M") or isinstance(series.dtype, DatetimeTZDtype):
        # Following DatetimeIndex (and Timestamp) convention,
        # inequality comparisons with Series[datetime64] raise
        msg = "Invalid comparison"
        with pytest.raises(TypeError, match=msg):
            None > series
        with pytest.raises(TypeError, match=msg):
            series > None
    else:
        result = None > series
        assert not result.iat[0]
        assert not result.iat[1]

        result = series < None
        assert not result.iat[0]
        assert not result.iat[1]


def test_series_varied_multiindex_alignment():
    # [FUNCTION DOC]
    # Verifies (issue 20414) adding a sub-selection of a three-level Series to a two-level Series aligns on the shared levels, giving a three-level result.
    # GH 20414
    s1 = Series(
        range(8),
        index=pd.MultiIndex.from_product(
            [list("ab"), list("xy"), [1, 2]], names=["ab", "xy", "num"]
        ),
    )
    s2 = Series(
        [1000 * i for i in range(1, 5)],
        index=pd.MultiIndex.from_product([list("xy"), [1, 2]], names=["xy", "num"]),
    )
    result = s1.loc[pd.IndexSlice[["a"], :, :]] + s2
    expected = Series(
        [1000, 2001, 3002, 4003],
        index=pd.MultiIndex.from_tuples(
            [("a", "x", 1), ("a", "x", 2), ("a", "y", 1), ("a", "y", 2)],
            names=["ab", "xy", "num"],
        ),
    )
    tm.assert_series_equal(result, expected)


def test_rmod_consistent_large_series():
    # [FUNCTION DOC]
    # Verifies (issue 29602) the reflected modulo of minus one on a 10001-element Series of two gives one, consistent for large sizes.
    # GH 29602
    result = Series([2] * 10001).rmod(-1)
    expected = Series([1] * 10001)

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "index",
    [
        date_range("2016-01-01", periods=3),
        date_range("2016-01-01", tz="US/Pacific", periods=3),
        pd.timedelta_range("1 Day", periods=3),
    ],
)
def test_comparison_mismatched_datetime_units(index):
    # [FUNCTION DOC]
    # Verifies (issue 63459) that equality and less-than between Series whose indexes differ in datetime unit (default versus nanosecond) work for naive datetimes, tz-aware datetimes and timedeltas, treating equal instants as equal. Parametrized over three indexes.
    # GH#63459

    ser = Series(1, index=index)
    ser2 = Series(1, index=index.as_unit("ns"))

    result = ser == ser2
    expected = Series([True, True, True], index=ser.index)
    tm.assert_series_equal(result, expected)

    result2 = ser2 < ser
    expected2 = Series([False, False, False], index=ser2.index)
    tm.assert_series_equal(result2, expected2)
