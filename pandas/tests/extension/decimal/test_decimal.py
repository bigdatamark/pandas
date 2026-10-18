# [MODULE DOC]
# Test module applying the shared extension-array test suite to the decimal test array, plus
# decimal-specific tests (constructors, astype dispatch, scalar op error handling, divmod, numpy
# ufunc interoperability, groupby aggregation, lazy indexing, to-numpy keyword and copy-on-write).
# Defines the fixtures the base suite requires (dtype, data, data_for_twos, data_missing,
# data_for_sorting, data_missing_for_sorting, na_cmp, data_for_grouping), the main test class
# TestDecimalArray (which overrides several base tests to account for decimal semantics and a
# fillna override without a copy keyword), two helper subclasses of the array, and standalone
# test functions. Depends on the decimal array module in the same package.
# Import-time side effect: attaches arithmetic operators to the test-only array subclass without
# coercion.
from __future__ import annotations

import decimal

import numpy as np
import pytest

from pandas.errors import Pandas4Warning

import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.decimal.array import (
    DecimalArray,
    DecimalDtype,
    make_data,
    to_decimal,
)


@pytest.fixture
def dtype():
    # [FUNCTION DOC]
    # HELPER: fixture returning a decimal dtype instance with the default context.
    return DecimalDtype()


@pytest.fixture
def data():
    # [FUNCTION DOC]
    # HELPER: fixture returning a decimal array of ten random decimals (deterministic, from the
    # data generator).
    return DecimalArray(make_data(10))


@pytest.fixture
def data_for_twos():
    # [FUNCTION DOC]
    # HELPER: fixture returning a ten-element decimal array where every element is two.
    return DecimalArray([decimal.Decimal(2) for _ in range(10)])


@pytest.fixture
def data_missing():
    # [FUNCTION DOC]
    # HELPER: fixture returning a two-element array: decimal not-a-number then one.
    return DecimalArray([decimal.Decimal("NaN"), decimal.Decimal(1)])


@pytest.fixture
def data_for_sorting():
    # [FUNCTION DOC]
    # HELPER: fixture returning the decimal array 1, 2, 0 (matching the pattern B, C, A).
    return DecimalArray(
        [decimal.Decimal("1"), decimal.Decimal("2"), decimal.Decimal("0")]
    )


@pytest.fixture
def data_missing_for_sorting():
    # [FUNCTION DOC]
    # HELPER: fixture returning the decimal array 1, not-a-number, 0 (matching B, NA, A).
    return DecimalArray(
        [decimal.Decimal("1"), decimal.Decimal("NaN"), decimal.Decimal("0")]
    )


@pytest.fixture
def na_cmp():
    # [FUNCTION DOC]
    # HELPER: fixture returning a comparison function that is true when both arguments are
    # not-a-number decimals (calls the decimal not-a-number test on each).
    return lambda x, y: x.is_nan() and y.is_nan()


@pytest.fixture
def data_for_grouping():
    # [FUNCTION DOC]
    # HELPER: fixture returning the eight-element array B, B, NA, NA, A, A, B, C with A=0.0, B=1.0,
    # C=2.0 and NA decimal not-a-number.
    b = decimal.Decimal("1.0")
    a = decimal.Decimal("0.0")
    c = decimal.Decimal("2.0")
    na = decimal.Decimal("NaN")
    return DecimalArray([b, b, na, na, a, a, b, c])


class TestDecimalArray(base.ExtensionTests):
    # [CLASS DOC]
    # Runs the complete extension test suite for the decimal array via the combined test class,
    # with overrides below for decimal-specific expectations (all operations are expected to
    # succeed, reductions are checked against plain numpy, fillna emits the missing-copy-keyword
    # warning, etc.).
    def _get_expected_exception(
        self, op_name: str, obj, other
    ) -> type[Exception] | tuple[type[Exception], ...] | None:
        # [FUNCTION DOC]
        # HELPER: override of the expected-exception hook. Inputs: op_name, obj, other. Always returns
        # none, meaning arithmetic and comparison operations are expected to succeed for every
        # operator.
        return None

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        # [FUNCTION DOC]
        # HELPER: override of the reduction support hook. Returns false for kurtosis and standard error
        # of the mean; true for all other reductions.
        if op_name in ["kurt", "sem"]:
            return False
        return True

    def check_reduce(self, ser: pd.Series, op_name: str, skipna: bool):
        # [FUNCTION DOC]
        # HELPER: override of the reduction check. For the count reduction defers to the base check.
        # Otherwise computes the series reduction with the skipna flag and compares it (approximately)
        # to the same reduction on the plain numpy conversion of the series (without any skipna handling).
        if op_name == "count":
            return super().check_reduce(ser, op_name, skipna)
        else:
            result = getattr(ser, op_name)(skipna=skipna)
            expected = getattr(np.asarray(ser), op_name)()
            tm.assert_almost_equal(result, expected)

    def test_reduce_series_numeric(self, data, all_numeric_reductions, skipna, request):
        # [FUNCTION DOC]
        # Override. Takes extra skipna and request fixtures. Marks the test as an expected failure
        # (not-implemented error) for kurtosis, skewness, standard error and median, then runs the base
        # test.
        if all_numeric_reductions in ["kurt", "skew", "sem", "median"]:
            mark = pytest.mark.xfail(raises=NotImplementedError)
            request.applymarker(mark)
        super().test_reduce_series_numeric(data, all_numeric_reductions, skipna)

    def test_reduce_frame(self, data, all_numeric_reductions, skipna, request):
        # [FUNCTION DOC]
        # Override. Marks the test as an expected failure (not-implemented error) for skewness and
        # median, then returns the result of the base test.
        op_name = all_numeric_reductions
        if op_name in ["skew", "median"]:
            mark = pytest.mark.xfail(raises=NotImplementedError)
            request.applymarker(mark)

        return super().test_reduce_frame(data, all_numeric_reductions, skipna)

    def test_compare_scalar(self, data, comparison_op):
        # [FUNCTION DOC]
        # Override: compares a series of the data with the float 0.5 using the base comparison helper.
        ser = pd.Series(data)
        self._compare_other(ser, data, comparison_op, 0.5)

    def test_compare_array(self, data, comparison_op):
        # [FUNCTION DOC]
        # Override: multiplies the series by decimals that randomly double, halve or keep each value
        # (random choice with a fixed seed among -1, 0, 1 as exponents of two), and compares the series
        # with that result using the base comparison helper.
        ser = pd.Series(data)

        alter = np.random.default_rng(2).choice([-1, 0, 1], len(data))
        # Randomly double, halve or keep same value
        other = pd.Series(data) * [decimal.Decimal(pow(2.0, i)) for i in alter]
        self._compare_other(ser, data, comparison_op, other)

    def test_arith_series_with_array(self, data, all_arithmetic_operators):
        # [FUNCTION DOC]
        # Override of the series-with-array arithmetic test. Temporarily disables the decimal context
        # traps for division by zero and invalid operation, then checks the operator against a series of
        # integer-converted hundred-fold values, against the doubled series (except for modulo
        # operators), and against the scalars zero and five. Restores the traps at the end.
        # KNOWN ISSUE: the global decimal context traps are restored only on normal completion; if any
        # check fails the exception leaves the traps disabled for subsequent tests (no cleanup block).
        op_name = all_arithmetic_operators
        ser = pd.Series(data)

        context = decimal.getcontext()
        divbyzerotrap = context.traps[decimal.DivisionByZero]
        invalidoptrap = context.traps[decimal.InvalidOperation]
        context.traps[decimal.DivisionByZero] = 0
        context.traps[decimal.InvalidOperation] = 0

        # Decimal supports ops with int, but not float
        other = pd.Series([int(d * 100) for d in data])
        self.check_opname(ser, op_name, other)

        if "mod" not in op_name:
            self.check_opname(ser, op_name, ser * 2)

        self.check_opname(ser, op_name, 0)
        self.check_opname(ser, op_name, 5)
        context.traps[decimal.DivisionByZero] = divbyzerotrap
        context.traps[decimal.InvalidOperation] = invalidoptrap

    def test_fillna_frame(self, data_missing):
        # [FUNCTION DOC]
        # Override: runs the base frame fillna test inside a check that the future-deprecation warning
        # for the array fillna lacking a copy keyword is emitted (message match, stack level not
        # checked).
        msg = "ExtensionArray.fillna added a 'copy' keyword"
        with tm.assert_produces_warning(
            Pandas4Warning, match=msg, check_stacklevel=False
        ):
            super().test_fillna_frame(data_missing)

    def test_fillna_series(self, data_missing):
        # [FUNCTION DOC]
        # Override: runs the base series fillna test expecting a deprecation warning about the missing
        # copy keyword in the array's fillna.
        msg = "ExtensionArray.fillna added a 'copy' keyword"
        with tm.assert_produces_warning(
            DeprecationWarning, match=msg, check_stacklevel=False
        ):
            super().test_fillna_series(data_missing)

    def test_fillna_with_none(self, data_missing):
        # [FUNCTION DOC]
        # Override. Unlike pandas, this array does not treat none as the missing value, so the base
        # test is expected to raise a type error whose message says none cannot be converted to a
        # decimal (two accepted wordings, one specific to a newer interpreter version).
        # GH#57723
        # EAs that don't have special logic for None will raise, unlike pandas'
        # which interpret None as the NA value for the dtype.
        msg = "|".join(
            [
                "Cannot convert None to Decimal",  # PY315 - maybe linux specific
                "conversion from NoneType to Decimal is not supported",
            ]
        )

        with pytest.raises(TypeError, match=msg):
            super().test_fillna_with_none(data_missing)

    def test_fillna_limit_frame(self, data_missing):
        # [FUNCTION DOC]
        # Override: base limit test for frames, expecting the missing-copy-keyword deprecation warning.
        # GH#58001
        msg = "ExtensionArray.fillna added a 'copy' keyword"
        with tm.assert_produces_warning(
            DeprecationWarning, match=msg, check_stacklevel=False
        ):
            super().test_fillna_limit_frame(data_missing)

    def test_fillna_limit_series(self, data_missing):
        # [FUNCTION DOC]
        # Override: base limit test for series, expecting the missing-copy-keyword deprecation warning.
        # GH#58001
        msg = "ExtensionArray.fillna added a 'copy' keyword"
        with tm.assert_produces_warning(
            DeprecationWarning, match=msg, check_stacklevel=False
        ):
            super().test_fillna_limit_series(data_missing)

    @pytest.mark.xfail(reason="copy keyword is missing")
    def test_fillna_readonly(self, data_missing):
        # [FUNCTION DOC]
        # Override marked as an expected failure because this array's fillna has no copy keyword;
        # otherwise runs the base test.
        super().test_fillna_readonly(data_missing)

    def test_series_repr(self, data):
        # [FUNCTION DOC]
        # Override of the base printing test: asserts the series repr contains the dtype name and the
        # text Decimal followed by a colon and a space, which proves the custom formatter is used.
        # Overriding this base test to explicitly test that
        # the custom _formatter is used
        ser = pd.Series(data)
        assert data.dtype.name in repr(ser)
        assert "Decimal: " in repr(ser)

    @pytest.mark.xfail(reason="Inconsistent array-vs-scalar behavior")
    @pytest.mark.parametrize("ufunc", [np.positive, np.negative, np.abs])
    def test_unary_ufunc_dunder_equivalence(self, data, ufunc):
        # [FUNCTION DOC]
        # Override marked as an expected failure (array and scalar behavior is inconsistent); parametrized
        # over numpy positive, negative and absolute value; otherwise runs the base test.
        super().test_unary_ufunc_dunder_equivalence(data, ufunc)


def test_take_na_value_other_decimal():
    # [FUNCTION DOC]
    # Taking positions 0 and -1 with fill allowed and a decimal fill value of -1.0 from a two-element
    # array gives the first element and the fill value (not the last element).
    arr = DecimalArray([decimal.Decimal("1.0"), decimal.Decimal("2.0")])
    result = arr.take([0, -1], allow_fill=True, fill_value=decimal.Decimal("-1.0"))
    expected = DecimalArray([decimal.Decimal("1.0"), decimal.Decimal("-1.0")])
    tm.assert_extension_array_equal(result, expected)


def test_series_constructor_coerce_data_to_extension_dtype():
    # [FUNCTION DOC]
    # A series built from integers 0, 1, 2 with the decimal dtype equals a series from an array of
    # the same values as decimals.
    dtype = DecimalDtype()
    ser = pd.Series([0, 1, 2], dtype=dtype)

    arr = DecimalArray(
        [decimal.Decimal(0), decimal.Decimal(1), decimal.Decimal(2)],
        dtype=dtype,
    )
    exp = pd.Series(arr)
    tm.assert_series_equal(ser, exp)


def test_series_constructor_with_dtype():
    # [FUNCTION DOC]
    # A series from a decimal array with the same dtype equals the one without; with an int64 dtype
    # requested it equals an integer series with the value ten.
    arr = DecimalArray([decimal.Decimal("10.0")])
    result = pd.Series(arr, dtype=DecimalDtype())
    expected = pd.Series(arr)
    tm.assert_series_equal(result, expected)

    result = pd.Series(arr, dtype="int64")
    expected = pd.Series([10])
    tm.assert_series_equal(result, expected)


def test_dataframe_constructor_with_dtype():
    # [FUNCTION DOC]
    # Same as the series constructor test but for frames: decimal dtype is a no-op and int64
    # converts to an integer column.
    arr = DecimalArray([decimal.Decimal("10.0")])

    result = pd.DataFrame({"A": arr}, dtype=DecimalDtype())
    expected = pd.DataFrame({"A": arr})
    tm.assert_frame_equal(result, expected)

    arr = DecimalArray([decimal.Decimal("10.0")])
    result = pd.DataFrame({"A": arr}, dtype="int64")
    expected = pd.DataFrame({"A": [10]})
    tm.assert_frame_equal(result, expected)


def test_astype_dispatches(frame_or_series):
    # [FUNCTION DOC]
    # Uses frame_or_series fixture. Builds a decimal series (wrapped in a frame if requested), casts
    # it to a decimal dtype with a custom context of precision five, and asserts the result's dtype
    # context precision is five. Verifies that the cast reaches the array's own conversion method.
    # This is a dtype-specific test that ensures Series[decimal].astype
    # gets all the way through to ExtensionArray.astype
    # Designing a reliable smoke test that works for arbitrary data types
    # is difficult.
    data = pd.Series(DecimalArray([decimal.Decimal(2)]), name="a")
    ctx = decimal.Context()
    ctx.prec = 5

    data = frame_or_series(data)

    result = data.astype(DecimalDtype(ctx))

    if frame_or_series is pd.DataFrame:
        result = result["a"]

    assert result.dtype.context.prec == ctx.prec


class DecimalArrayWithoutFromSequence(DecimalArray):
    # [CLASS DOC]
    # HELPER: decimal array subclass for testing error handling in the sequence constructor: its
    # alternate constructor always raises a key error with a fixed message.
    """Helper class for testing error handling in _from_sequence."""

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        # [FUNCTION DOC]
        # HELPER: always raises a key error (message: for the test); ignores its arguments.
        raise KeyError("For the test")


class DecimalArrayWithoutCoercion(DecimalArrayWithoutFromSequence):
    # [CLASS DOC]
    # HELPER: subclass of the failing-constructor array whose arithmetic method factory disables
    # coercion of results back to the array type, so binary operations return plain numpy arrays.
    @classmethod
    def _create_arithmetic_method(cls, op):
        # [FUNCTION DOC]
        # HELPER: class method used when attaching arithmetic operators; builds the operator via the
        # scalar-ops mixin factory with result coercion to the dtype turned off. Input op (binary
        # function); returns the method.
        return cls._create_method(op, coerce_to_dtype=False)


DecimalArrayWithoutCoercion._add_arithmetic_ops()


def test_scalar_ops_from_sequence_raises():
    # [FUNCTION DOC]
    # Adding two non-coercing decimal arrays must return a plain object numpy array of the sums
    # (2.0, 4.0), since results cannot be coerced back (the sequence constructor raises).
    # op(EA, EA) should return an EA, or an ndarray if it's not possible
    # to return an EA with the return values.
    arr = DecimalArrayWithoutCoercion([decimal.Decimal("1.0"), decimal.Decimal("2.0")])
    result = arr + arr
    expected = np.array(
        [decimal.Decimal("2.0"), decimal.Decimal("4.0")], dtype="object"
    )
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "reverse, expected_div, expected_mod",
    [(False, [0, 1, 1, 2], [1, 0, 1, 0]), (True, [2, 1, 0, 0], [0, 0, 2, 2])],
)
def test_divmod_array(reverse, expected_div, expected_mod):
    # [FUNCTION DOC]
    # Parametrized over reverse false/true with expected quotient and remainder lists. Applies the
    # builtin divmod to an array of 1 to 4 and the integer two (in either order) and asserts the
    # resulting quotient and remainder arrays equal arrays built from the expected lists (regression
    # for an upstream issue).
    # https://github.com/pandas-dev/pandas/issues/22930
    arr = to_decimal([1, 2, 3, 4])
    if reverse:
        div, mod = divmod(2, arr)
    else:
        div, mod = divmod(arr, 2)
    expected_div = to_decimal(expected_div)
    expected_mod = to_decimal(expected_mod)

    tm.assert_extension_array_equal(div, expected_div)
    tm.assert_extension_array_equal(mod, expected_mod)


def test_ufunc_fallback(data):
    # [FUNCTION DOC]
    # Applying numpy's absolute value to a series of a decimal slice with a custom index gives a
    # series equal to the absolute value of the array with that index.
    a = data[:5]
    s = pd.Series(a, index=range(3, 8))
    result = np.abs(s)
    expected = pd.Series(np.abs(a), index=range(3, 8))
    tm.assert_series_equal(result, expected)


def test_array_ufunc():
    # [FUNCTION DOC]
    # Applying numpy's exponential to a decimal array returns a decimal array equal to the exp of
    # the underlying data.
    a = to_decimal([1, 2, 3])
    result = np.exp(a)
    expected = to_decimal(np.exp(a._data))
    tm.assert_extension_array_equal(result, expected)


def test_array_ufunc_series():
    # [FUNCTION DOC]
    # Applying the exponential to a series of decimals returns a series of decimals with the same
    # values as the array version.
    a = to_decimal([1, 2, 3])
    s = pd.Series(a)
    result = np.exp(s)
    expected = pd.Series(to_decimal(np.exp(a._data)))
    tm.assert_series_equal(result, expected)


def test_array_ufunc_series_scalar_other():
    # [FUNCTION DOC]
    # Checks the handled-types logic: adding a decimal scalar to a series via numpy's add equals the
    # series built from adding the scalar to the array.
    # check _HANDLED_TYPES
    a = to_decimal([1, 2, 3])
    s = pd.Series(a)
    result = np.add(s, decimal.Decimal(1))
    expected = pd.Series(np.add(a, decimal.Decimal(1)))
    tm.assert_series_equal(result, expected)


def test_array_ufunc_series_defer():
    # [FUNCTION DOC]
    # Adding a series and an array with numpy's add, in both orders, gives the same series of
    # doubled values (the array defers to the series).
    a = to_decimal([1, 2, 3])
    s = pd.Series(a)

    expected = pd.Series(to_decimal([2, 4, 6]))
    r1 = np.add(s, a)
    r2 = np.add(a, s)

    tm.assert_series_equal(r1, expected)
    tm.assert_series_equal(r2, expected)


def test_groupby_agg():
    # [FUNCTION DOC]
    # Regression for an upstream issue: aggregation results are inferred as decimal dtype. For a
    # frame with two keys and a decimals column checks first-element aggregation via a function: by a
    # single key on the selected column (both groupby spellings), by two keys on the selected column
    # (both spellings, with a multi-index result), and across all columns of the frame grouped by
    # one key (frame result with integer column and decimal column).
    # Ensure that the result of agg is inferred to be decimal dtype
    # https://github.com/pandas-dev/pandas/issues/29141

    data = make_data(5)
    df = pd.DataFrame(
        {"id1": [0, 0, 0, 1, 1], "id2": [0, 1, 0, 1, 1], "decimals": DecimalArray(data)}
    )

    # single key, selected column
    expected = pd.Series(to_decimal([data[0], data[3]]))
    result = df.groupby("id1")["decimals"].agg(lambda x: x.iloc[0])
    tm.assert_series_equal(result, expected, check_names=False)
    result = df["decimals"].groupby(df["id1"]).agg(lambda x: x.iloc[0])
    tm.assert_series_equal(result, expected, check_names=False)

    # multiple keys, selected column
    expected = pd.Series(
        to_decimal([data[0], data[1], data[3]]),
        index=pd.MultiIndex.from_tuples([(0, 0), (0, 1), (1, 1)]),
    )
    result = df.groupby(["id1", "id2"])["decimals"].agg(lambda x: x.iloc[0])
    tm.assert_series_equal(result, expected, check_names=False)
    result = df["decimals"].groupby([df["id1"], df["id2"]]).agg(lambda x: x.iloc[0])
    tm.assert_series_equal(result, expected, check_names=False)

    # multiple columns
    expected = pd.DataFrame({"id2": [0, 1], "decimals": to_decimal([data[0], data[3]])})
    result = df.groupby("id1").agg(lambda x: x.iloc[0])
    tm.assert_frame_equal(result, expected, check_names=False)


def test_groupby_agg_ea_method(monkeypatch):
    # [FUNCTION DOC]
    # Regression for an upstream issue. Temporarily adds a summing method to the array class and
    # aggregates per group by calling it on the group's backing array, for both a frame column group
    # and a series grouped by a numpy array; results must equal series of decimal group sums.
    # Ensure that the result of agg is inferred to be decimal dtype
    # https://github.com/pandas-dev/pandas/issues/29141

    def DecimalArray__my_sum(self):
        # [FUNCTION DOC]
        # HELPER: patched-in method returning the numpy sum of the array's elements.
        return np.sum(np.array(self))

    monkeypatch.setattr(DecimalArray, "my_sum", DecimalArray__my_sum, raising=False)

    data = make_data(5)
    df = pd.DataFrame({"id": [0, 0, 0, 1, 1], "decimals": DecimalArray(data)})
    expected = pd.Series(to_decimal([data[0] + data[1] + data[2], data[3] + data[4]]))

    result = df.groupby("id")["decimals"].agg(lambda x: x.values.my_sum())
    tm.assert_series_equal(result, expected, check_names=False)
    s = pd.Series(DecimalArray(data))
    grouper = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    result = s.groupby(grouper).agg(lambda x: x.values.my_sum())
    tm.assert_series_equal(result, expected, check_names=False)


def test_indexing_no_materialize(monkeypatch):
    # [FUNCTION DOC]
    # Regression for an upstream issue. Temporarily makes numpy conversion of the array raise an
    # error, then asserts boolean-mask selection on a series and frame, and scalar lookups by label
    # do not trigger such a conversion (do not raise).
    # See https://github.com/pandas-dev/pandas/issues/29708
    # Ensure that indexing operations do not materialize (convert to a numpy
    # array) the ExtensionArray unnecessary

    def DecimalArray__array__(self, dtype=None):
        # [FUNCTION DOC]
        # HELPER: patched-in numpy conversion method that always raises a generic exception, used to detect
        # unwanted materialization.
        raise Exception("tried to convert a DecimalArray to a numpy array")

    monkeypatch.setattr(DecimalArray, "__array__", DecimalArray__array__, raising=False)

    data = make_data(10)
    s = pd.Series(DecimalArray(data))
    df = pd.DataFrame({"a": s, "b": range(len(s))})

    # ensure the following operations do not raise an error
    s[s > 0.5]
    df[s > 0.5]
    s.at[0]
    df.at[0, "a"]


def test_to_numpy_keyword():
    # [FUNCTION DOC]
    # The decimals keyword of the array's numpy conversion rounds to two digits (1.11, 2.22) for
    # the array and for a series wrapping it, giving an object numpy array of decimals.
    # test the extra keyword
    values = [decimal.Decimal("1.1111"), decimal.Decimal("2.2222")]
    expected = np.array(
        [decimal.Decimal("1.11"), decimal.Decimal("2.22")], dtype="object"
    )
    a = pd.array(values, dtype="decimal")
    result = a.to_numpy(decimals=2)
    tm.assert_numpy_array_equal(result, expected)

    result = pd.Series(a).to_numpy(decimals=2)
    tm.assert_numpy_array_equal(result, expected)


def test_array_copy_on_write():
    # [FUNCTION DOC]
    # Casting an object-dtype frame of decimals to the decimal dtype and then mutating the source
    # frame must not change the cast frame: its values equal a frame built directly with the original
    # values.
    df = pd.DataFrame({"a": [decimal.Decimal(2), decimal.Decimal(3)]}, dtype="object")
    df2 = df.astype(DecimalDtype())
    df.iloc[0, 0] = 0
    expected = pd.DataFrame(
        {"a": [decimal.Decimal(2), decimal.Decimal(3)]}, dtype=DecimalDtype()
    )
    tm.assert_equal(df2.values, expected.values)
