# [MODULE DOC]
# Base tests for reductions (sum, min, any, ...) on series and frames backed by extension
# arrays. Public contents: BaseReduceTests mixin with overridable hooks: the support check,
# the series comparison helper, and the expected-dtype hook for frame reductions. Uses data and
# the all_boolean_reductions and all_numeric_reductions fixtures.
from typing import final

import pytest

import pandas as pd
import pandas._testing as tm


class BaseReduceTests:
    # [CLASS DOC]
    # Mixin for reduction tests; meaningful mostly for numeric and boolean data. Subclasses
    # override the support hook (and optionally the dtype hook) to opt in.
    """
    Reduction specific tests. Generally these only
    make sense for numeric/boolean operations.
    """

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        # [FUNCTION DOC]
        # HELPER: hook saying whether the named reduction is expected to succeed for the series.
        # Inputs: ser, op_name. Returns a boolean; default always false.
        # Specify if we expect this reduction to succeed.
        return False

    def check_reduce(self, ser: pd.Series, op_name: str, skipna: bool):
        # [FUNCTION DOC]
        # HELPER: compares a series reduction with the same reduction on a float64 copy.
        # Inputs: ser, op_name, skipna. The reference is the series cast to float64, falling back to
        # object dtype when the cast raises a type or value error. For the count reduction the skipna
        # argument is not passed. Asserts the two results are approximately equal. Subclasses may
        # override when a cast other than float64 is needed.
        # We perform the same operation on the np.float64 data and check
        #  that the results match. Override if you need to cast to something
        #  other than float64.
        res_op = getattr(ser, op_name)

        try:
            alt = ser.astype("float64")
        except (TypeError, ValueError):
            # e.g. Interval can't cast (TypeError), StringArray can't cast
            #  (ValueError), so let's cast to object and do
            #  the reduction pointwise
            alt = ser.astype(object)

        exp_op = getattr(alt, op_name)
        if op_name == "count":
            result = res_op()
            expected = exp_op()
        else:
            result = res_op(skipna=skipna)
            expected = exp_op(skipna=skipna)
        tm.assert_almost_equal(result, expected)

    def _get_expected_reduction_dtype(self, arr, op_name: str, skipna: bool):
        # [FUNCTION DOC]
        # HELPER: hook giving the dtype expected when a reduction runs on a frame column with this
        # array. Inputs: arr, op_name, skipna. Default retains the array's dtype (float-like
        # behavior).
        # Find the expected dtype when the given reduction is done on a DataFrame
        # column with this array.  The default assumes float64-like behavior,
        # i.e. retains the dtype.
        return arr.dtype

    # We anticipate that authors should not need to override check_reduce_frame,
    #  but should be able to do any necessary overriding in
    #  _get_expected_reduction_dtype. If you have a use case where this
    #  does not hold, please let us know at github.com/pandas-dev/pandas/issues.
    @final
    def check_reduce_frame(self, ser: pd.Series, op_name: str, skipna: bool):
        # [FUNCTION DOC]
        # HELPER (final, not meant to be overridden; customize via the dtype hook): checks that a frame
        # reduction looks like a wrapped version of the series reduction.
        # Inputs: ser, op_name, skipna. Builds a one-column frame from the series' array; for variance,
        # standard deviation and standard error uses a delta degrees of freedom of one. First
        # asserts the private array reduce with keepdims equals the frame reduction's backing array
        # (described in the code as perfunctory). Then asserts that result equals a one-element array
        # of the expected dtype holding either missing (when skipna is false and the series has missing
        # values) or the series reduction over the non-missing values.
        # Check that the 2D reduction done in a DataFrame reduction "looks like"
        # a wrapped version of the 1D reduction done by Series.
        arr = ser.array
        df = pd.DataFrame({"a": arr})

        kwargs = {"ddof": 1} if op_name in ["var", "std", "sem"] else {}

        cmp_dtype = self._get_expected_reduction_dtype(arr, op_name, skipna)

        # The DataFrame method just calls arr._reduce with keepdims=True,
        #  so this first check is perfunctory.
        result1 = arr._reduce(op_name, skipna=skipna, keepdims=True, **kwargs)
        result2 = getattr(df, op_name)(skipna=skipna, **kwargs).array
        tm.assert_extension_array_equal(result1, result2)

        # Check that the 2D reduction looks like a wrapped version of the
        #  1D reduction
        if not skipna and ser.isna().any():
            expected = pd.array([pd.NA], dtype=cmp_dtype)
        else:
            exp_value = getattr(ser.dropna(), op_name)()
            expected = pd.array([exp_value], dtype=cmp_dtype)

        tm.assert_extension_array_equal(result1, expected)

    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series_boolean(self, data, all_boolean_reductions, skipna):
        # [FUNCTION DOC]
        # Parametrized over skipna and the boolean reductions. If the reduction is not supported it must
        # raise a type error matching a message pattern; otherwise results are checked via the series
        # comparison helper.
        # KNOWN ISSUE: the message pattern ends with an alternation separator, producing an empty
        # alternative that matches any message, so the message is not actually verified (a TODO in the
        # code says the same).
        op_name = all_boolean_reductions
        ser = pd.Series(data)

        if not self._supports_reduction(ser, op_name):
            # TODO: the message being checked here isn't actually checking anything
            msg = (
                "[Cc]annot perform|Categorical is not ordered for operation|"
                "does not support operation|"
            )

            with pytest.raises(TypeError, match=msg):
                getattr(ser, op_name)(skipna=skipna)

        else:
            self.check_reduce(ser, op_name, skipna)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series_numeric(self, data, all_numeric_reductions, skipna):
        # [FUNCTION DOC]
        # Parametrized over skipna and the numeric reductions; runtime warnings are ignored since
        # min/max on empty input produce numpy warnings. Same structure as the boolean version: expects
        # a type error if unsupported, otherwise compares with the float64 reference.
        # KNOWN ISSUE: the message pattern ends with an alternation separator, producing an empty
        # alternative that matches any message, so the message is not actually verified (a TODO in the
        # code says the same).
        op_name = all_numeric_reductions
        ser = pd.Series(data)

        if not self._supports_reduction(ser, op_name):
            # TODO: the message being checked here isn't actually checking anything
            msg = (
                "[Cc]annot perform|Categorical is not ordered for operation|"
                "does not support operation|"
            )

            with pytest.raises(TypeError, match=msg):
                getattr(ser, op_name)(skipna=skipna)

        else:
            # min/max with empty produce numpy warnings
            self.check_reduce(ser, op_name, skipna)

    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_frame(self, data, all_numeric_reductions, skipna):
        # [FUNCTION DOC]
        # Parametrized over skipna and numeric reductions. Skips the count reduction (not an array
        # method) and any reduction the support hook rejects; otherwise runs the frame check helper.
        op_name = all_numeric_reductions
        ser = pd.Series(data)

        if op_name == "count":
            pytest.skip(f"{op_name} not an array method")

        if not self._supports_reduction(ser, op_name):
            pytest.skip(f"Reduction {op_name} not supported for this dtype")

        self.check_reduce_frame(ser, op_name, skipna)
