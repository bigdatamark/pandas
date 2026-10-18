# [MODULE DOC]
# Base tests for arithmetic, comparison and unary operators on extension arrays. Public
# contents: BaseOpsUtil (shared helpers and per-case expected-exception class attributes),
# BaseArithmeticOpsTests, BaseComparisonOpsTests and BaseUnaryOpsTests. The general technique
# is to compare the vectorized operator result with a pointwise computation obtained by
# combining series element by element. Uses data, data_for_twos and the operator-name fixtures.
from __future__ import annotations

from typing import final

import numpy as np
import pytest

from pandas.core.dtypes.common import is_string_dtype

import pandas as pd
import pandas._testing as tm
from pandas.core import ops


class BaseOpsUtil:
    # [CLASS DOC]
    # Shared helper base class. Class attributes name the exception type expected (or none for
    # success) for four situations: series with scalar, frame with scalar, series with array-like
    # series, and divmod. Subclasses set them to indicate unsupported operations. Helper methods
    # decide the expected exception, adjust the pointwise result's dtype, resolve operator names,
    # and run the comparison checks. Several helpers are final and not intended to be overridden.
    series_scalar_exc: type[Exception] | None = TypeError
    frame_scalar_exc: type[Exception] | None = TypeError
    series_array_exc: type[Exception] | None = TypeError
    divmod_exc: type[Exception] | None = TypeError

    def _get_expected_exception(
        self, op_name: str, obj, other
    ) -> type[Exception] | tuple[type[Exception], ...] | None:
        # [FUNCTION DOC]
        # HELPER: decides which exception, if any, the operation is expected to raise.
        # Inputs: op_name (dunder name), obj (series or frame), other (operand). Returns an exception
        # class, a tuple of them, or none for success. divmod operators use the divmod attribute;
        # series-with-series uses the series-array attribute; series with other operands uses the
        # series-scalar attribute; otherwise the frame-scalar attribute.
        # Find the Exception, if any we expect to raise calling
        #  obj.__op_name__(other)

        # The self.obj_bar_exc pattern isn't great in part because it can depend
        #  on op_name or dtypes, but we use it here for backward-compatibility.
        if op_name in ["__divmod__", "__rdivmod__"]:
            result = self.divmod_exc
        elif isinstance(obj, pd.Series) and isinstance(other, pd.Series):
            result = self.series_array_exc
        elif isinstance(obj, pd.Series):
            result = self.series_scalar_exc
        else:
            result = self.frame_scalar_exc

        return result

    def _cast_pointwise_result(self, op_name: str, obj, other, pointwise_result):
        # [FUNCTION DOC]
        # HELPER: hook to convert the pointwise-combined result so its dtype matches the vectorized
        # result, for cases where inference on the pointwise scalars gives a different but acceptable
        # dtype. Inputs: op_name, obj, other, pointwise_result. Default returns the input unchanged.
        # In _check_op we check that the result of a pointwise operation
        #  (found via _combine) matches the result of the vectorized
        #  operation obj.__op_name__(other).
        #  In some cases pandas dtype inference on the scalar result may not
        #  give a matching dtype even if both operations are behaving "correctly".
        #  In these cases, do extra required casting here.
        return pointwise_result

    def get_op_from_name(self, op_name: str):
        # [FUNCTION DOC]
        # HELPER: maps an operator name to the callable by delegating to the shared testing helper.
        # Subclasses may override for custom operators.
        return tm.get_op_from_name(op_name)

    # Subclasses are not expected to need to override check_opname, _check_op,
    #  _check_divmod_op, or _combine.
    #  Ideally any relevant overriding can be done in _cast_pointwise_result,
    #  get_op_from_name, and the specification of `exc`. If you find a use
    #  case that still requires overriding _check_op or _combine, please let
    #  us know at github.com/pandas-dev/pandas/issues
    @final
    def check_opname(self, ser: pd.Series, op_name: str, other):
        # [FUNCTION DOC]
        # HELPER (final): top-level check for one named operator. Inputs: ser (series), op_name,
        # other. Looks up the expected exception and the operator, then runs the single-operation check.
        exc = self._get_expected_exception(op_name, ser, other)
        op = self.get_op_from_name(op_name)

        self._check_op(ser, op, other, op_name, exc)

    # see comment on check_opname
    @final
    def _combine(self, obj, other, op):
        # [FUNCTION DOC]
        # HELPER (final): computes the pointwise expected result. Inputs: obj (series or frame),
        # other, op. For a frame, requires exactly one column (raises a not-implemented error otherwise),
        # combines that column with other and converts back to a frame; for a series, combines directly.
        if isinstance(obj, pd.DataFrame):
            if len(obj.columns) != 1:
                raise NotImplementedError
            expected = obj.iloc[:, 0].combine(other, op).to_frame()
        else:
            expected = obj.combine(other, op)
        return expected

    # see comment on check_opname
    @final
    def _check_op(
        self, ser: pd.Series, op, other, op_name: str, exc=NotImplementedError
    ):
        # [FUNCTION DOC]
        # HELPER (final): verifies one operation. Inputs: ser, op (callable), other, op_name, exc
        # (expected exception, default not-implemented error). If exc is none: applies the operator,
        # computes the pointwise expected result, casts it with the hook, asserts the result has the
        # same container type as the input and equals the expected. Otherwise asserts the operator
        # raises exc.
        # Check that the Series/DataFrame arithmetic/comparison method matches
        #  the pointwise result from _combine.

        if exc is None:
            result = op(ser, other)
            expected = self._combine(ser, other, op)
            expected = self._cast_pointwise_result(op_name, ser, other, expected)
            assert isinstance(result, type(ser))
            tm.assert_equal(result, expected)
        else:
            with pytest.raises(exc):
                op(ser, other)

    # see comment on check_opname
    @final
    def _check_divmod_op(self, ser: pd.Series, op, other):
        # [FUNCTION DOC]
        # HELPER (final): verifies divmod or reflected divmod. Inputs: ser, op (the forward builtin or
        # the reflected function), other. Determines the expected exception for the matching dunder name.
        # If none: the quotient and remainder from the operation must equal floor division and modulo
        # computed separately (swapping operand order for the reflected form). Otherwise asserts that an
        # exception is raised.
        # KNOWN ISSUE: in the exception branch the check always calls the builtin divmod on the series and
        # other instead of the supplied op, so the reflected operation is not what is tested there.
        # check that divmod behavior matches behavior of floordiv+mod
        if op is divmod:
            exc = self._get_expected_exception("__divmod__", ser, other)
        else:
            exc = self._get_expected_exception("__rdivmod__", ser, other)
        if exc is None:
            result_div, result_mod = op(ser, other)
            if op is divmod:
                expected_div, expected_mod = ser // other, ser % other
            else:
                expected_div, expected_mod = other // ser, other % ser
            tm.assert_series_equal(result_div, expected_div)
            tm.assert_series_equal(result_mod, expected_mod)
        else:
            with pytest.raises(exc):
                divmod(ser, other)


class BaseArithmeticOpsTests(BaseOpsUtil):
    # [CLASS DOC]
    # Arithmetic operator tests. Documents (in its docstring) that subclasses set the four
    # exception class attributes to declare support; this class redeclares them with a type-error
    # default.
    """
    Various Series and DataFrame arithmetic ops methods.

    Subclasses supporting various ops should set the class variables
    to indicate that they support ops of that kind

    * series_scalar_exc = TypeError
    * frame_scalar_exc = TypeError
    * series_array_exc = TypeError
    * divmod_exc = TypeError
    """

    series_scalar_exc: type[Exception] | None = TypeError
    frame_scalar_exc: type[Exception] | None = TypeError
    series_array_exc: type[Exception] | None = TypeError
    divmod_exc: type[Exception] | None = TypeError

    def test_arith_series_with_scalar(self, data, all_arithmetic_operators):
        # [FUNCTION DOC]
        # Uses the all_arithmetic_operators fixture. Applies each operator between a series and its
        # first element via the check helper. Skips the reflected modulo for string dtypes (string
        # formatting semantics).
        # series & scalar
        if all_arithmetic_operators == "__rmod__" and is_string_dtype(data.dtype):
            pytest.skip("Skip testing Python string formatting")

        op_name = all_arithmetic_operators
        ser = pd.Series(data)
        self.check_opname(ser, op_name, ser.iloc[0])

    def test_arith_frame_with_scalar(self, data, all_arithmetic_operators):
        # [FUNCTION DOC]
        # Same as the series test but with a one-column frame and the first element of the array as
        # the scalar; skips reflected modulo for string dtypes.
        # frame & scalar
        if all_arithmetic_operators == "__rmod__" and is_string_dtype(data.dtype):
            pytest.skip("Skip testing Python string formatting")

        op_name = all_arithmetic_operators
        df = pd.DataFrame({"A": data})
        self.check_opname(df, op_name, data[0])

    def test_arith_series_with_array(self, data, all_arithmetic_operators):
        # [FUNCTION DOC]
        # Applies each arithmetic operator between a series and a same-length series of the first
        # element repeated, via the check helper.
        # ndarray & other series
        op_name = all_arithmetic_operators
        ser = pd.Series(data)
        self.check_opname(ser, op_name, pd.Series([ser.iloc[0]] * len(ser)))

    def test_divmod(self, data):
        # [FUNCTION DOC]
        # Checks divmod of a series with the integer one and reflected divmod of one with the series,
        # via the divmod check helper.
        ser = pd.Series(data)
        self._check_divmod_op(ser, divmod, 1)
        self._check_divmod_op(1, ops.rdivmod, ser)

    def test_divmod_series_array(self, data, data_for_twos):
        # [FUNCTION DOC]
        # Checks divmod of a series with the array; reflected divmod of the data_for_twos array with
        # the series; and the reflected divmod where the left operand is a series made from that array.
        ser = pd.Series(data)
        self._check_divmod_op(ser, divmod, data)

        other = data_for_twos
        self._check_divmod_op(other, ops.rdivmod, ser)

        other = pd.Series(other)
        self._check_divmod_op(other, ops.rdivmod, ser)

    def test_add_series_with_extension_array(self, data):
        # [FUNCTION DOC]
        # Adding an extension array to a series of the same dtype should equal the series built from
        # adding the arrays directly. If the expected exception for addition is set, asserts the
        # addition raises it instead.
        # Check adding an ExtensionArray to a Series of the same dtype matches
        # the behavior of adding the arrays directly and then wrapping in a
        # Series.

        ser = pd.Series(data)

        exc = self._get_expected_exception("__add__", ser, data)
        if exc is not None:
            with pytest.raises(exc):
                ser + data
            return

        result = ser + data
        expected = pd.Series(data + data)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("box", [pd.Series, pd.DataFrame, pd.Index])
    @pytest.mark.parametrize(
        "op_name",
        [
            x
            for x in tm.arithmetic_dunder_methods + tm.comparison_dunder_methods
            if not x.startswith("__r")
        ],
    )
    def test_direct_arith_with_ndframe_returns_not_implemented(
        self, data, box, op_name
    ):
        # [FUNCTION DOC]
        # Parametrized over the box type (series, frame, index) and every forward (non-reflected)
        # arithmetic and comparison dunder name. If the array defines that method, calling it directly
        # with the boxed array must return the not-implemented sentinel so that pandas can unbox
        # and dispatch.
        # EAs should return NotImplemented for ops with Series/DataFrame/Index
        # Pandas takes care of unboxing the series and calling the EA's op.
        other = box(data)

        if hasattr(data, op_name):
            result = getattr(data, op_name)(other)
            assert result is NotImplemented


class BaseComparisonOpsTests(BaseOpsUtil):
    # [CLASS DOC]
    # Comparison operator tests.
    """Various Series and DataFrame comparison ops methods."""

    def _compare_other(self, ser: pd.Series, data, op, other):
        # [FUNCTION DOC]
        # HELPER: compares a comparison operator result with the pointwise result. Inputs: ser, data
        # (unused), op, other. For equality and inequality the vectorized result must equal the
        # pointwise combine (after the dtype hook). For other operators: if the vectorized operation
        # succeeds it must match the pointwise result; if it raises, the pointwise combine must raise
        # the same exception type.
        if op.__name__ in ["eq", "ne"]:
            # comparison should match point-wise comparisons
            result = op(ser, other)
            expected = ser.combine(other, op)
            expected = self._cast_pointwise_result(op.__name__, ser, other, expected)
            tm.assert_series_equal(result, expected)

        else:
            exc = None
            try:
                result = op(ser, other)
            except Exception as err:
                exc = err

            if exc is None:
                # Didn't error, then should match pointwise behavior
                expected = ser.combine(other, op)
                expected = self._cast_pointwise_result(
                    op.__name__, ser, other, expected
                )
                tm.assert_series_equal(result, expected)
            else:
                with pytest.raises(type(exc)):
                    ser.combine(other, op)

    def test_compare_scalar(self, data, comparison_op):
        # [FUNCTION DOC]
        # Uses the comparison_op fixture; compares a series of the data with the integer zero.
        ser = pd.Series(data)
        self._compare_other(ser, data, comparison_op, 0)

    def test_compare_array(self, data, comparison_op):
        # [FUNCTION DOC]
        # Uses the comparison_op fixture; compares a series of the data with a same-length series of
        # the first element repeated.
        ser = pd.Series(data)
        other = pd.Series([data[0]] * len(data), dtype=data.dtype)
        self._compare_other(ser, data, comparison_op, other)


class BaseUnaryOpsTests(BaseOpsUtil):
    # [CLASS DOC]
    # Unary operator tests.
    def test_invert(self, data):
        # [FUNCTION DOC]
        # Checks the bitwise invert operator. If inverting the individual scalars raises a type error,
        # inverting the series and the array must also raise a type error. Otherwise the series
        # inversion must equal a series built from the array's inversion (not from pointwise Python
        # results, because of the semantics of negating booleans) and keep its name.
        ser = pd.Series(data, name="name")
        try:
            [~x for x in data]
        except TypeError:
            # scalars don't support invert -> we don't expect the vectorized
            #  operation to succeed
            with pytest.raises(TypeError):
                ~ser
            with pytest.raises(TypeError):
                ~data
        else:
            # Note we do not reuse the pointwise result to construct expected
            #  because python semantics for negating bools are weird see GH#54569
            result = ~ser
            expected = pd.Series(~data, name="name")
            tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("ufunc", [np.positive, np.negative, np.abs])
    def test_unary_ufunc_dunder_equivalence(self, data, ufunc):
        # [FUNCTION DOC]
        # Parametrized over the numpy positive, negative and absolute-value functions. Asserts the
        # corresponding dunder method (positive, negative, absolute) on the array works if and only if
        # the numpy function works: if the dunder raises, the numpy function must raise the same type or
        # a type error; otherwise both results are equal arrays.
        # the dunder __pos__ works if and only if np.positive works,
        #  same for __neg__/np.negative and __abs__/np.abs
        attr = {np.positive: "__pos__", np.negative: "__neg__", np.abs: "__abs__"}[
            ufunc
        ]

        exc = None
        try:
            result = getattr(data, attr)()
        except Exception as err:
            exc = err

            # if __pos__ raised, then so should the ufunc
            with pytest.raises((type(exc), TypeError)):
                ufunc(data)
        else:
            alt = ufunc(data)
            tm.assert_extension_array_equal(result, alt)
