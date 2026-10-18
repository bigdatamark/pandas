"""
This file contains a minimal set of tests for compliance with the extension
array interface test suite, and should contain no other tests.
The test suite for the full functionality of the array is located in
`pandas/tests/arrays/`.
The tests in this file are inherited from the BaseExtensionTests, and only
minimal tweaks should be applied to get the tests passing (by overwriting a
parent method).
Additional tests should either be added to one of the BaseExtensionTests
classes (if they are relevant for the extension interface for all dtypes), or
be added to the array-specific tests in `pandas/tests/arrays/`.
"""
# [MODULE DOC]
# Compliance tests for the arrow-backed extension array and its dtype, run through the shared
# extension-array test suite across every pyarrow type in the shared type list, plus a large number of
# additional arrow-specific tests: arithmetic and reductions for each type, Kleene logic for booleans, bitwise operations,
# string methods (the string accessor), datetime and timedelta accessors, time zone handling, unit changes, setitem aliasing
# and copy-on-write checks, casting, factorizing, interpolation, quantile, describe, pickle compatibility and dtype
# metadata (item size, type names). The data fixtures build ten-element arrays with nulls per arrow type. The module
# skips entirely when pyarrow is not installed. Import-time side effects: imports the arrow array class after the
# import-or-skip check (so the imports follow executable code on purpose) and defines a legacy array subclass used for pickle
# compatibility. Platform and version gates use pyarrow and Python version flags and a helper that marks tests expected to fail on
# Windows CI without a time zone database.

from __future__ import annotations

from datetime import (
    date,
    datetime,
    time,
    timedelta,
)
from decimal import Decimal
from io import (
    BytesIO,
    StringIO,
)
import operator
import pickle
import re
import sys

import numpy as np
import pytest

from pandas._libs import lib
from pandas._libs.tslibs import timezones
from pandas.compat import (
    PY312,
    is_ci_environment,
    is_platform_windows,
    pa_version_under14p0,
    pa_version_under19p0,
    pa_version_under20p0,
    pa_version_under21p0,
)
from pandas.compat.pyarrow import pa_version_under22p0
from pandas.errors import Pandas4Warning

from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    CategoricalDtypeType,
)

import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import no_default
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_signed_integer_dtype,
    is_string_dtype,
    is_unsigned_integer_dtype,
)
from pandas.tests.extension import base

pa = pytest.importorskip("pyarrow")

from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.arrays.arrow.extension_types import ArrowPeriodType


def _require_timezone_database(request):
    # [FUNCTION DOC]
    # HELPER: applies an expected-failure marker when the test runs on Windows in continuous integration with a pyarrow version
    # older than 22 (a time zone database environment variable is not set on CI, producing an arrow invalid-data error). Parameter: the
    # pytest request object; no return value; used by tests with time-zone aware timestamps.
    if is_platform_windows() and is_ci_environment() and pa_version_under22p0:
        mark = pytest.mark.xfail(
            raises=pa.ArrowInvalid,
            reason=(
                "TODO: Set ARROW_TIMEZONE_DATABASE environment variable "
                "on CI to path to the tzdata for pyarrow."
            ),
        )
        request.applymarker(mark)


@pytest.fixture(params=tm.ALL_PYARROW_DTYPES, ids=str)
def dtype(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over every pyarrow type in the shared testing list (test ids are the type strings);
    # returns the arrow-backed dtype for that type.
    return ArrowDtype(pyarrow_dtype=request.param)


@pytest.fixture
def data(dtype):
    # [FUNCTION DOC]
    # HELPER: fixture returning an arrow-backed array of ten values for the dtype, with two nulls (positions four and seven) and values
    # chosen per arrow type category: boolean, floating, signed, unsigned, decimal, date, timestamp, duration, time, string and binary.
    # Raises not-implemented for any other type.
    pa_dtype = dtype.pyarrow_dtype
    if pa.types.is_boolean(pa_dtype):
        data = [True, False] * 2 + [None] + [True, False] + [None] + [True, False]
    elif pa.types.is_floating(pa_dtype):
        data = [1.0, 0.0] * 2 + [None] + [-2.0, -1.0] + [None] + [0.5, 99.5]
    elif pa.types.is_signed_integer(pa_dtype):
        data = [1, 0] * 2 + [None] + [-2, -1] + [None] + [1, 99]
    elif pa.types.is_unsigned_integer(pa_dtype):
        data = [1, 0] * 2 + [None] + [2, 1] + [None] + [1, 99]
    elif pa.types.is_decimal(pa_dtype):
        data = (
            [Decimal("1"), Decimal("0.0")] * 2
            + [None]
            + [Decimal("-2.0"), Decimal("-1.0")]
            + [None]
            + [Decimal("0.5"), Decimal("33.123")]
        )
    elif pa.types.is_date(pa_dtype):
        data = (
            [date(2022, 1, 1), date(1999, 12, 31)] * 2
            + [None]
            + [date(2022, 1, 1), date(2022, 1, 1)]
            + [None]
            + [date(1999, 12, 31), date(1999, 12, 31)]
        )
    elif pa.types.is_timestamp(pa_dtype):
        data = (
            [datetime(2020, 1, 1, 1, 1, 1, 1), datetime(1999, 1, 1, 1, 1, 1, 1)] * 2
            + [None]
            + [datetime(2020, 1, 1, 1), datetime(1999, 1, 1, 1)]
            + [None]
            + [datetime(2020, 1, 1), datetime(1999, 1, 1)]
        )
    elif pa.types.is_duration(pa_dtype):
        data = (
            [timedelta(1), timedelta(1, 1)] * 2
            + [None]
            + [timedelta(-1), timedelta(0)]
            + [None]
            + [timedelta(-10), timedelta(10)]
        )
    elif pa.types.is_time(pa_dtype):
        data = (
            [time(12, 0), time(0, 12)] * 2
            + [None]
            + [time(0, 0), time(1, 1)]
            + [None]
            + [time(0, 5), time(5, 0)]
        )
    elif pa.types.is_string(pa_dtype):
        data = ["a", "b"] * 2 + [None] + ["1", "2"] + [None] + ["!", ">"]
    elif pa.types.is_binary(pa_dtype):
        data = [b"a", b"b"] * 2 + [None] + [b"1", b"2"] + [None] + [b"!", b">"]
    else:
        raise NotImplementedError
    return pd.array(data, dtype=dtype)


@pytest.fixture
def data_missing(data):
    # [FUNCTION DOC]
    # HELPER: fixture returning a two-element array: a null followed by the first element of the data.
    """Length-2 array with [NA, Valid]"""
    return type(data)._from_sequence([None, data[0]], dtype=data.dtype)


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over the names data and data_missing; returns the corresponding fixture value, so tests run with and without missing values.
    """Parametrized fixture returning 'data' or 'data_missing' integer arrays.

    Used to test dtype conversion with and without missing values.
    """
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing


@pytest.fixture
def data_for_grouping(dtype):
    # [FUNCTION DOC]
    # HELPER: fixture returning an eight-element array [B, B, null, null, A, A, B, C] where A < B < C. The three values per arrow type category are
    # fixed (boolean uses false, true, true so B equals C). Raises not-implemented for unsupported types.
    """
    Data for factorization, grouping, and unique tests.

    Expected to be like [B, B, NA, NA, A, A, B, C]

    Where A < B < C and NA is missing
    """
    pa_dtype = dtype.pyarrow_dtype
    if pa.types.is_boolean(pa_dtype):
        A = False
        B = True
        C = True
    elif pa.types.is_floating(pa_dtype):
        A = -1.1
        B = 0.0
        C = 1.1
    elif pa.types.is_signed_integer(pa_dtype):
        A = -1
        B = 0
        C = 1
    elif pa.types.is_unsigned_integer(pa_dtype):
        A = 0
        B = 1
        C = 10
    elif pa.types.is_date(pa_dtype):
        A = date(1999, 12, 31)
        B = date(2010, 1, 1)
        C = date(2022, 1, 1)
    elif pa.types.is_timestamp(pa_dtype):
        A = datetime(1999, 1, 1, 1, 1, 1, 1)
        B = datetime(2020, 1, 1)
        C = datetime(2020, 1, 1, 1)
    elif pa.types.is_duration(pa_dtype):
        A = timedelta(-1)
        B = timedelta(0)
        C = timedelta(1, 4)
    elif pa.types.is_time(pa_dtype):
        A = time(0, 0)
        B = time(0, 12)
        C = time(12, 12)
    elif pa.types.is_string(pa_dtype):
        A = "a"
        B = "b"
        C = "c"
    elif pa.types.is_binary(pa_dtype):
        A = b"a"
        B = b"b"
        C = b"c"
    elif pa.types.is_decimal(pa_dtype):
        A = Decimal("-1.1")
        B = Decimal("0.0")
        C = Decimal("1.1")
    else:
        raise NotImplementedError
    return pd.array([B, B, None, None, A, A, B, C], dtype=dtype)


@pytest.fixture
def data_for_sorting(data_for_grouping):
    # [FUNCTION DOC]
    # HELPER: fixture returning a three-element array [B, C, A] taken from positions zero, seven and four of the grouping data.
    """
    Length-3 array with a known sort order.

    This should be three items [B, C, A] with
    A < B < C
    """
    return type(data_for_grouping)._from_sequence(
        [data_for_grouping[0], data_for_grouping[7], data_for_grouping[4]],
        dtype=data_for_grouping.dtype,
    )


@pytest.fixture
def data_missing_for_sorting(data_for_grouping):
    # [FUNCTION DOC]
    # HELPER: fixture returning [B, null, A] taken from positions zero, two and four of the grouping data.
    """
    Length-3 array with a known sort order.

    This should be three items [B, NA, A] with
    A < B and NA missing.
    """
    return type(data_for_grouping)._from_sequence(
        [data_for_grouping[0], data_for_grouping[2], data_for_grouping[4]],
        dtype=data_for_grouping.dtype,
    )


@pytest.fixture
def data_for_twos(data):
    # [FUNCTION DOC]
    # HELPER: fixture returning ten twos in the data's dtype for integer, floating, decimal and duration arrow types; otherwise returns the data itself (tests are expected to fail where two is not a valid scalar). A source TODO asks whether to skip otherwise.
    # KNOWN ISSUE: the docstring says length 100, but the array built has ten elements.
    """Length-100 array in which all the elements are two."""
    pa_dtype = data.dtype.pyarrow_dtype
    if (
        pa.types.is_integer(pa_dtype)
        or pa.types.is_floating(pa_dtype)
        or pa.types.is_decimal(pa_dtype)
        or pa.types.is_duration(pa_dtype)
    ):
        return pd.array([2] * 10, dtype=data.dtype)
    # tests will be xfailed where 2 is not a valid scalar for pa_dtype
    return data
    # TODO: skip otherwise?


class TestArrowArray(base.ExtensionTests):
    # [CLASS DOC]
    # Extension-suite test class for the arrow-backed extension array across all pyarrow types. Overrides combine, accumulation, reduction, casting, constructors, setitem,
    # sorting and many other inherited tests with per-type expected failures; contains numerous additional arrow-specific tests (many hundreds of lines).
    def _construct_for_combine_add(self, left, right):
        # [FUNCTION DOC]
        # HELPER: builds the expected result of element-wise combine-add. Parameters: left array and right (array of the same class or a scalar). Uses the left's dtype, except that
        # boolean and (when the right is also an array) 8-bit integer arrow types are widened to 64-bit integer. Returns a new array of the same class from the pairwise sums (for an array right) or the left elements plus the scalar.
        dtype = left.dtype

        # in a couple cases, addition is not dtype-preserving
        if dtype == "bool[pyarrow]":
            dtype = pandas_dtype("int64[pyarrow]")
        elif dtype == "int8[pyarrow]" and isinstance(right, type(left)):
            dtype = pandas_dtype("int64[pyarrow]")

        if isinstance(right, type(left)):
            return left._from_sequence(
                [a + b for (a, b) in zip(list(left), list(right), strict=True)],
                dtype=dtype,
            )
        else:
            return left._from_sequence(
                [a + right for a in list(left)],
                dtype=dtype,
            )

    def test_compare_scalar(self, data, comparison_op):
        # [FUNCTION DOC]
        # Compares the Series from data with the first element of the data as a scalar, using the comparison operator fixture and the comparison helper.
        ser = pd.Series(data)
        self._compare_other(ser, data, comparison_op, data[0])

    def test_compare_range_len(self, data, comparison_op):
        # [FUNCTION DOC]
        # GH#63429. Compares the Series from data with a range of the same length (a list-like that is not an array) using the comparison helper.
        # GH#63429
        ser = pd.Series(data)
        range_test = range(len(ser))
        self._compare_other(ser, range_test, comparison_op, range_test)

    @pytest.mark.parametrize("na_action", [None, "ignore"])
    def test_map(self, data_missing, na_action, using_nan_is_na):
        # [FUNCTION DOC]
        # Parametrized over missing-value action (none, ignore). Mapping the identity function: for date/time kinds (timedelta and datetime kinds) the result is the object-dtype numpy array of the data;
        # otherwise the numpy array of the data (for 32-bit arrow floats with NaN as missing the 64-bit float array with NaN, since mapping round-trips through objects).
        if data_missing.dtype.kind in "mM":
            result = data_missing.map(lambda x: x, na_action=na_action)
            expected = data_missing.to_numpy(dtype=object)
            tm.assert_numpy_array_equal(result, expected)
        else:
            result = data_missing.map(lambda x: x, na_action=na_action)
            if data_missing.dtype == "float32[pyarrow]" and using_nan_is_na:
                # map roundtrips through objects, which converts to float64
                expected = data_missing.to_numpy(dtype="float64", na_value=np.nan)
            else:
                expected = data_missing.to_numpy()
            tm.assert_numpy_array_equal(result, expected)

    def test_astype_str(self, data, request, using_infer_string):
        # [FUNCTION DOC]
        # Expected-to-fail for binary types (casting to string decodes) and, when string inference is off, for timezone-naive timestamps and durations (the repr differs from numpy's). Then delegates to the parent test.
        pa_dtype = data.dtype.pyarrow_dtype
        if pa.types.is_binary(pa_dtype):
            request.applymarker(
                pytest.mark.xfail(
                    reason=f"For {pa_dtype} .astype(str) decodes.",
                )
            )
        elif not using_infer_string and (
            (pa.types.is_timestamp(pa_dtype) and pa_dtype.tz is None)
            or pa.types.is_duration(pa_dtype)
        ):
            request.applymarker(
                pytest.mark.xfail(
                    reason="pd.Timestamp/pd.Timedelta repr different from numpy repr",
                )
            )
        super().test_astype_str(data)

    def test_from_dtype(self, data, request):
        # [FUNCTION DOC]
        # Expected-to-fail for string (arrow string dtype is not the pyarrow string dtype) and decimal (type for alias cannot infer decimals) types; delegates to the parent test.
        pa_dtype = data.dtype.pyarrow_dtype
        if pa.types.is_string(pa_dtype) or pa.types.is_decimal(pa_dtype):
            if pa.types.is_string(pa_dtype):
                reason = "ArrowDtype(pa.string()) != StringDtype('pyarrow')"
            else:
                reason = f"pyarrow.type_for_alias cannot infer {pa_dtype}"

            request.applymarker(
                pytest.mark.xfail(
                    reason=reason,
                )
            )
        super().test_from_dtype(data)

    def test_from_sequence_pa_array(self, data):
        # [FUNCTION DOC]
        # Verifies building from the data's chunked arrow array, and from its combined single chunk, returns the original data and the result always holds a chunked array.
        # https://github.com/pandas-dev/pandas/pull/47034#discussion_r955500784
        # data._pa_array = pa.ChunkedArray
        result = type(data)._from_sequence(data._pa_array, dtype=data.dtype)
        tm.assert_extension_array_equal(result, data)
        assert isinstance(result._pa_array, pa.ChunkedArray)

        result = type(data)._from_sequence(
            data._pa_array.combine_chunks(), dtype=data.dtype
        )
        tm.assert_extension_array_equal(result, data)
        assert isinstance(result._pa_array, pa.ChunkedArray)

    def test_from_sequence_pa_array_notimplemented(self, request):
        # [FUNCTION DOC]
        # Verifies the from-strings constructor on the array class with a month-day-nano interval dtype raises not-implemented with message converting strings.
        dtype = ArrowDtype(pa.month_day_nano_interval())
        with pytest.raises(NotImplementedError, match="Converting strings to"):
            ArrowExtensionArray._from_sequence_of_strings(["12-1"], dtype=dtype)

    def test_from_sequence_of_strings_pa_array(self, data, request):
        # [FUNCTION DOC]
        # Casts the data's arrow array to string, builds the array from strings with the dtype (also from the combined chunk) and checks the round trip equals the data. For time-zone aware timestamps, applies the time zone database expected-failure helper.
        pa_dtype = data.dtype.pyarrow_dtype
        if pa.types.is_timestamp(pa_dtype) and pa_dtype.tz is not None:
            _require_timezone_database(request)

        pa_array = data._pa_array.cast(pa.string())
        result = type(data)._from_sequence_of_strings(pa_array, dtype=data.dtype)
        tm.assert_extension_array_equal(result, data)

        pa_array = pa_array.combine_chunks()
        result = type(data)._from_sequence_of_strings(pa_array, dtype=data.dtype)
        tm.assert_extension_array_equal(result, data)

    def check_accumulate(self, ser, op_name, skipna):
        # [FUNCTION DOC]
        # HELPER: overridden accumulation check. For temporal types it compares via the integer representation (32-bit integer for 32-bit wide types, otherwise 64-bit) after casting both input and result; otherwise
        # casts both to the nullable 64-bit float and compares the series accumulation result with the float accumulation, ignoring dtype.
        result = getattr(ser, op_name)(skipna=skipna)

        pa_type = ser.dtype.pyarrow_dtype
        if pa.types.is_temporal(pa_type):
            # Just check that we match the integer behavior.
            if pa_type.bit_width == 32:
                int_type = "int32[pyarrow]"
            else:
                int_type = "int64[pyarrow]"
            ser = ser.astype(int_type)
            result = result.astype(int_type)

        result = result.astype("Float64")
        expected = getattr(ser.astype("Float64"), op_name)(skipna=skipna)
        tm.assert_series_equal(result, expected, check_dtype=False)

    def _supports_accumulation(self, ser: pd.Series, op_name: str) -> bool:
        # [FUNCTION DOC]
        # HELPER: reports whether the accumulation is supported for the series' arrow type. Binary and decimal: none of cumulative sum, product, max, min. String: not cumulative product. Boolean: not product, max or min.
        # Temporal: not cumulative sum (except for durations) and not product. Everything else supported.
        # error: Item "dtype[Any]" of "dtype[Any] | ExtensionDtype" has no
        # attribute "pyarrow_dtype"
        pa_type = ser.dtype.pyarrow_dtype  # type: ignore[union-attr]

        if pa.types.is_binary(pa_type) or pa.types.is_decimal(pa_type):
            if op_name in ["cumsum", "cumprod", "cummax", "cummin"]:
                return False
        elif pa.types.is_string(pa_type):
            if op_name == "cumprod":
                return False
        elif pa.types.is_boolean(pa_type):
            if op_name in ["cumprod", "cummax", "cummin"]:
                return False
        elif pa.types.is_temporal(pa_type):
            if op_name == "cumsum" and not pa.types.is_duration(pa_type):
                return False
            elif op_name == "cumprod":
                return False
        return True

    @pytest.mark.parametrize("skipna", [True, False])
    def test_accumulate_series(self, data, all_numeric_accumulations, skipna, request):
        # [FUNCTION DOC]
        # Parametrized over skipna and the numeric accumulation names. Returns early for string types with cumulative sum, min, max (tested in the series cumulative tests, GH 60633). If the accumulation is unsupported, delegates
        # to the parent test (which checks that an error is raised). Cumulative sum on boolean or decimal is marked expected-to-fail with a type error. Otherwise runs the accumulation check helper.
        pa_type = data.dtype.pyarrow_dtype
        op_name = all_numeric_accumulations

        if pa.types.is_string(pa_type) and op_name in ["cumsum", "cummin", "cummax"]:
            # https://github.com/pandas-dev/pandas/pull/60633
            # Doesn't fit test structure, tested in series/test_cumulative.py instead.
            return

        ser = pd.Series(data)

        if not self._supports_accumulation(ser, op_name):
            # The base class test will check that we raise
            return super().test_accumulate_series(
                data, all_numeric_accumulations, skipna
            )

        if all_numeric_accumulations == "cumsum" and (
            pa.types.is_boolean(pa_type) or pa.types.is_decimal(pa_type)
        ):
            request.applymarker(
                pytest.mark.xfail(
                    reason=f"{all_numeric_accumulations} not implemented for {pa_type}",
                    raises=TypeError,
                )
            )

        self.check_accumulate(ser, op_name, skipna)

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        # [FUNCTION DOC]
        # HELPER: reports whether a reduction is supported for the series' arrow type. Kurtosis is never supported; skew is unsupported for old pyarrow (before 20). Temporal types do not support sum, variance, product, skew except sum on durations;
        # binary lacks sum and skew; string and binary lack mean, median, product, std, sem, variance, skew; non-duration temporal types do not support any and all (kept as pyarrow behavior, GH#34479); booleans lack median, std, var, skew, kurt, sem. Otherwise supported.
        if op_name == "kurt" or (pa_version_under20p0 and op_name == "skew"):
            return False

        dtype = ser.dtype
        # error: Item "dtype[Any]" of "dtype[Any] | ExtensionDtype" has
        # no attribute "pyarrow_dtype"
        pa_dtype = dtype.pyarrow_dtype  # type: ignore[union-attr]
        if pa.types.is_temporal(pa_dtype) and op_name in ["sum", "var", "prod", "skew"]:
            if pa.types.is_duration(pa_dtype) and op_name in ["sum"]:
                # summing timedeltas is one case that *is* well-defined
                pass
            else:
                return False
        elif pa.types.is_binary(pa_dtype) and op_name in ["sum", "skew"]:
            return False
        elif (
            pa.types.is_string(pa_dtype) or pa.types.is_binary(pa_dtype)
        ) and op_name in ["mean", "median", "prod", "std", "sem", "var", "skew"]:
            return False

        if (
            pa.types.is_temporal(pa_dtype)
            and not pa.types.is_duration(pa_dtype)
            and op_name in ["any", "all"]
        ):
            # xref GH#34479 we support this in our non-pyarrow datetime64 dtypes,
            #  but it isn't obvious we _should_.  For now, we keep the pyarrow
            #  behavior which does not support this.
            return False

        if pa.types.is_boolean(pa_dtype) and op_name in [
            "median",
            "std",
            "var",
            "skew",
            "kurt",
            "sem",
        ]:
            return False

        return True

    def check_reduce(self, ser: pd.Series, op_name: str, skipna: bool):
        # [FUNCTION DOC]
        # HELPER: overridden reduction check. For integer and floating arrow types, the expected value comes from the series cast to nullable 64-bit float; otherwise the series itself is the alternative. Count takes no skipna. Compares the results with almost-equal.
        # A source TODO notes the non-numeric case compares the series with itself, so tests almost nothing.
        # error: Item "dtype[Any]" of "dtype[Any] | ExtensionDtype" has no
        # attribute "pyarrow_dtype"
        pa_dtype = ser.dtype.pyarrow_dtype  # type: ignore[union-attr]
        if pa.types.is_integer(pa_dtype) or pa.types.is_floating(pa_dtype):
            alt = ser.astype("Float64")
        else:
            # TODO: in the opposite case, aren't we testing... nothing? For
            # e.g. date/time dtypes trying to calculate 'expected' by converting
            # to object will raise for mean, std etc
            alt = ser

        # TODO: in the opposite case, aren't we testing... nothing?
        if op_name == "count":
            result = getattr(ser, op_name)()
            expected = getattr(alt, op_name)()
        else:
            result = getattr(ser, op_name)(skipna=skipna)
            expected = getattr(alt, op_name)(skipna=skipna)
        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series_boolean(
        self, data, all_boolean_reductions, skipna, na_value, request
    ):
        # [FUNCTION DOC]
        # Parametrized over skipna. Marks expected-to-fail with a type error for string and binary arrow types (boolean reductions not implemented in pyarrow; undecided whether to behave like non-pyarrow), then delegates to the parent test and returns its result.
        pa_dtype = data.dtype.pyarrow_dtype
        xfail_mark = pytest.mark.xfail(
            raises=TypeError,
            reason=(
                f"{all_boolean_reductions} is not implemented in "
                f"pyarrow={pa.__version__} for {pa_dtype}"
            ),
        )
        if pa.types.is_string(pa_dtype) or pa.types.is_binary(pa_dtype):
            # We *might* want to make this behave like the non-pyarrow cases,
            #  but have not yet decided.
            request.applymarker(xfail_mark)

        return super().test_reduce_series_boolean(data, all_boolean_reductions, skipna)

    def _get_expected_reduction_dtype(self, arr, op_name: str, skipna: bool):
        # [FUNCTION DOC]
        # HELPER: expected result dtype of a reduction. Min/max keep the dtype. Temporal: standard deviation and sem give a duration (same for duration; seconds for date; the type's unit for time and timestamp), other reductions keep the dtype. For the decimal type with precision 7 and scale 3: sum gives decimal with precision 38 for new pyarrow (21+),
        # other than median/var/std/sem/skew keep the dtype, else 64-bit float. Median, variance, std, mean, skew, sem give 64-bit float; sum/product of booleans give unsigned 64-bit; sum on strings keeps the dtype; otherwise by kind: signed to 64-bit integer, unsigned to 64-bit unsigned, floating to 64-bit float.
        pa_type = arr._pa_array.type

        if op_name in ["max", "min"]:
            cmp_dtype = arr.dtype
        elif pa.types.is_temporal(pa_type):
            if op_name in ["std", "sem"]:
                if pa.types.is_duration(pa_type):
                    cmp_dtype = arr.dtype
                elif pa.types.is_date(pa_type):
                    cmp_dtype = ArrowDtype(pa.duration("s"))
                elif pa.types.is_time(pa_type):
                    cmp_dtype = ArrowDtype(pa.duration(pa_type.unit))
                else:
                    cmp_dtype = ArrowDtype(pa.duration(pa_type.unit))
            else:
                cmp_dtype = arr.dtype
        elif arr.dtype.name == "decimal128(7, 3)[pyarrow]":
            if op_name == "sum" and not pa_version_under21p0:
                # https://github.com/apache/arrow/pull/44184
                cmp_dtype = ArrowDtype(pa.decimal128(38, 3))
            elif op_name not in ["median", "var", "std", "sem", "skew"]:
                cmp_dtype = arr.dtype
            else:
                cmp_dtype = "float64[pyarrow]"
        elif op_name in ["median", "var", "std", "mean", "skew", "sem"]:
            cmp_dtype = "float64[pyarrow]"
        elif op_name in ["sum", "prod"] and pa.types.is_boolean(pa_type):
            cmp_dtype = "uint64[pyarrow]"
        elif op_name == "sum" and pa.types.is_string(pa_type):
            cmp_dtype = arr.dtype
        else:
            cmp_dtype = {
                "i": "int64[pyarrow]",
                "u": "uint64[pyarrow]",
                "f": "float64[pyarrow]",
            }[arr.dtype.kind]
        return cmp_dtype

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series_numeric(self, data, all_numeric_reductions, skipna, request):
        # [FUNCTION DOC]
        # Parametrized over skipna. Ignores runtime warnings. For new pyarrow (20+), skew with skipna on integer or floating types is expected to fail (arrow issue 45733). Delegates to the parent test and returns its result.
        if (
            not pa_version_under20p0
            and skipna
            and all_numeric_reductions == "skew"
            and (
                pa.types.is_integer(data.dtype.pyarrow_dtype)
                or pa.types.is_floating(data.dtype.pyarrow_dtype)
            )
        ):
            request.applymarker(
                pytest.mark.xfail(
                    reason="https://github.com/apache/arrow/issues/45733",
                )
            )
        return super().test_reduce_series_numeric(data, all_numeric_reductions, skipna)

    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_frame(self, data, all_numeric_reductions, skipna, request):
        # [FUNCTION DOC]
        # Parametrized over skipna. For skew with old pyarrow (before 20) and numeric dtypes marks expected failure (skew not implemented); delegates to the parent frame reduction test.
        op_name = all_numeric_reductions
        if op_name == "skew" and pa_version_under20p0:
            if data.dtype._is_numeric:
                mark = pytest.mark.xfail(reason="skew not implemented")
                request.applymarker(mark)
        return super().test_reduce_frame(data, all_numeric_reductions, skipna)

    @pytest.mark.parametrize("typ", ["int64", "uint64", "float64"])
    def test_median_not_approximate(self, typ):
        # [FUNCTION DOC]
        # GH 52679. Parametrized over int64, uint64, float64 arrow types: the median of a series of one and two is exactly 1.5, not approximated.
        # GH 52679
        result = pd.Series([1, 2], dtype=f"{typ}[pyarrow]").median()
        assert result == 1.5

    def test_construct_from_string_own_name(self, dtype, request):
        # [FUNCTION DOC]
        # Marks decimal types expected-to-fail with not-implemented (alias cannot infer). For string arrow types, constructing from the dtype's own name raises a type error saying it should be constructed by the string dtype; then returns. Otherwise delegates to the parent test.
        pa_dtype = dtype.pyarrow_dtype
        if pa.types.is_decimal(pa_dtype):
            request.applymarker(
                pytest.mark.xfail(
                    raises=NotImplementedError,
                    reason=f"pyarrow.type_for_alias cannot infer {pa_dtype}",
                )
            )

        if pa.types.is_string(pa_dtype):
            # We still support StringDtype('pyarrow') over ArrowDtype(pa.string())
            msg = r"string\[pyarrow\] should be constructed by StringDtype"
            with pytest.raises(TypeError, match=msg):
                dtype.construct_from_string(dtype.name)

            return

        super().test_construct_from_string_own_name(dtype)

    def test_is_dtype_from_name(self, dtype, request):
        # [FUNCTION DOC]
        # For arrow string types, asserts the dtype name is not recognized as this dtype (the string dtype is preferred); for decimals marks expected failure with not-implemented; otherwise delegates to the parent test.
        pa_dtype = dtype.pyarrow_dtype
        if pa.types.is_string(pa_dtype):
            # We still support StringDtype('pyarrow') over ArrowDtype(pa.string())
            assert not type(dtype).is_dtype(dtype.name)
        else:
            if pa.types.is_decimal(pa_dtype):
                request.applymarker(
                    pytest.mark.xfail(
                        raises=NotImplementedError,
                        reason=f"pyarrow.type_for_alias cannot infer {pa_dtype}",
                    )
                )
            super().test_is_dtype_from_name(dtype)

    def test_construct_from_string_another_type_raises(self, dtype):
        # [FUNCTION DOC]
        # Verifies constructing from an unrelated string raises a type error saying the name must end with the pyarrow suffix.
        msg = r"'another_type' must end with '\[pyarrow\]'"
        with pytest.raises(TypeError, match=msg):
            type(dtype).construct_from_string("another_type")

    def test_get_common_dtype(self, dtype, request):
        # [FUNCTION DOC]
        # Marks expected-to-fail for date, time, tz-aware timestamp, binary and decimal types (no associated numpy dtype findable by the common-type lookup); delegates to the parent test.
        pa_dtype = dtype.pyarrow_dtype
        if (
            pa.types.is_date(pa_dtype)
            or pa.types.is_time(pa_dtype)
            or (pa.types.is_timestamp(pa_dtype) and pa_dtype.tz is not None)
            or pa.types.is_binary(pa_dtype)
            or pa.types.is_decimal(pa_dtype)
        ):
            request.applymarker(
                pytest.mark.xfail(
                    reason=(
                        f"{pa_dtype} does not have associated numpy "
                        f"dtype findable by find_common_type"
                    )
                )
            )
        super().test_get_common_dtype(dtype)

    def test_is_not_string_type(self, dtype):
        # [FUNCTION DOC]
        # For string arrow types asserts the dtype is a string dtype; otherwise delegates to the parent test.
        pa_dtype = dtype.pyarrow_dtype
        if pa.types.is_string(pa_dtype):
            assert is_string_dtype(dtype)
        else:
            super().test_is_not_string_type(dtype)

    @pytest.mark.xfail(
        reason="GH 45419: pyarrow.ChunkedArray does not support views.", run=False
    )
    def test_view(self, data):
        # [FUNCTION DOC]
        # Expected-to-fail and not run (GH 45419: pyarrow chunked arrays do not support views). Delegates to the parent test.
        super().test_view(data)

    def test_fillna_no_op_returns_copy(self, data):
        # [FUNCTION DOC]
        # Verifies filling a missing-free copy of the data with its first valid value returns a different object equal to the data.
        data = data[~data.isna()]

        valid = data[0]
        result = data.fillna(valid)
        assert result is not data
        tm.assert_extension_array_equal(result, data)

    def test_fillna_readonly(self, data_missing):
        # [FUNCTION DOC]
        # Makes a read-only copy of the missing-data array. Filling with the valid value returns a result with the valid first element and leaves the original unchanged. Then (as the comment says, no-copy is not honored by arrow arrays but new data is returned) repeats the same fill and checks the same.
        # KNOWN ISSUE: the comment says it tests the no-copy option but the second call does not pass any no-copy option, so it duplicates the first check.
        data = data_missing.copy()
        data._readonly = True

        # by default fillna(copy=True), then this works fine
        result = data.fillna(data_missing[1])
        assert result[0] == data_missing[1]
        tm.assert_extension_array_equal(data, data_missing)

        # fillna(copy=False) is generally not honored by Arrow-backed array,
        # but always returns new data -> same result as above
        result = data.fillna(data_missing[1])
        assert result[0] == data_missing[1]
        tm.assert_extension_array_equal(data, data_missing)

    @pytest.mark.xfail(
        reason="GH 45419: pyarrow.ChunkedArray does not support views", run=False
    )
    def test_transpose(self, data):
        # [FUNCTION DOC]
        # Expected-to-fail and not run (GH 45419: views not supported). Delegates to the parent test.
        super().test_transpose(data)

    @pytest.mark.xfail(
        reason="GH 45419: pyarrow.ChunkedArray does not support views", run=False
    )
    def test_setitem_preserves_views(self, data):
        # [FUNCTION DOC]
        # Expected-to-fail and not run (GH 45419: views not supported). Delegates to the parent test.
        super().test_setitem_preserves_views(data)

    @pytest.mark.parametrize("dtype_backend", ["pyarrow", no_default])
    @pytest.mark.parametrize("engine", ["c", "python"])
    def test_EA_types(self, engine, data, dtype_backend, request, using_nan_is_na):
        # [FUNCTION DOC]
        # Parametrized over parser engine (c, python) and the dtype backend (pyarrow, default). Marks expected failure for decimal (parameterized types not supported), timestamps with microsecond or nanosecond unit (issue 49767, value error), and binary
        # (CSV parsers mishandle binary). Writes a frame of the data to CSV text (or bytes for binary), with NA or NaN as the missing marker depending on the NaN-is-NA mode, reads it back with the dtype name, engine and backend and compares with the original frame.
        pa_dtype = data.dtype.pyarrow_dtype
        if pa.types.is_decimal(pa_dtype):
            request.applymarker(
                pytest.mark.xfail(
                    raises=NotImplementedError,
                    reason=f"Parameterized types {pa_dtype} not supported.",
                )
            )
        elif pa.types.is_timestamp(pa_dtype) and pa_dtype.unit in ("us", "ns"):
            request.applymarker(
                pytest.mark.xfail(
                    raises=ValueError,
                    reason="https://github.com/pandas-dev/pandas/issues/49767",
                )
            )
        elif pa.types.is_binary(pa_dtype):
            request.applymarker(
                pytest.mark.xfail(reason="CSV parsers don't correctly handle binary")
            )
        df = pd.DataFrame({"with_dtype": pd.Series(data, dtype=str(data.dtype))})
        if not using_nan_is_na:
            csv_output = df.to_csv(index=False, na_rep="NA")
        else:
            csv_output = df.to_csv(index=False, na_rep=np.nan)
        if pa.types.is_binary(pa_dtype):
            csv_output = BytesIO(csv_output)
        else:
            csv_output = StringIO(csv_output)
        result = pd.read_csv(
            csv_output,
            dtype={"with_dtype": str(data.dtype)},
            engine=engine,
            dtype_backend=dtype_backend,
        )
        expected = df
        tm.assert_frame_equal(result, expected)

    def test_invert(self, data, request):
        # [FUNCTION DOC]
        # Marks expected-to-fail (arrow not-implemented) for types other than boolean, integer and string. For booleans on Python 3.12+ expects a deprecation warning about bitwise inversion (stack level not checked); delegates to the parent invert test.
        pa_dtype = data.dtype.pyarrow_dtype
        if not (
            pa.types.is_boolean(pa_dtype)
            or pa.types.is_integer(pa_dtype)
            or pa.types.is_string(pa_dtype)
        ):
            request.applymarker(
                pytest.mark.xfail(
                    raises=pa.ArrowNotImplementedError,
                    reason=f"pyarrow.compute.invert does support {pa_dtype}",
                )
            )
        if PY312 and pa.types.is_boolean(pa_dtype):
            with tm.assert_produces_warning(
                DeprecationWarning, match="Bitwise inversion", check_stacklevel=False
            ):
                super().test_invert(data)
        else:
            super().test_invert(data)

    @pytest.mark.parametrize("periods", [1, -2])
    def test_diff(self, data, periods, request):
        # [FUNCTION DOC]
        # Parametrized over periods one and negative two. Marks expected-to-fail (arrow invalid-data error) for unsigned integer types with period one (overflow); delegates to the parent diff test.
        pa_dtype = data.dtype.pyarrow_dtype
        if pa.types.is_unsigned_integer(pa_dtype) and periods == 1:
            request.applymarker(
                pytest.mark.xfail(
                    raises=pa.ArrowInvalid,
                    reason=(
                        f"diff with {pa_dtype} and periods={periods} will overflow"
                    ),
                )
            )
        super().test_diff(data, periods)

    def test_value_counts_returns_pyarrow_int64(self, data):
        # [FUNCTION DOC]
        # GH 51462. Verifies value counts of the first ten elements have the arrow 64-bit integer dtype.
        # GH 51462
        data = data[:10]
        result = data.value_counts()
        assert result.dtype == ArrowDtype(pa.int64())

    _combine_le_expected_dtype = "bool[pyarrow]"

    def get_op_from_name(self, op_name):
        # [FUNCTION DOC]
        # HELPER: maps an operator name to a function. After stripping underscores, reflected true division and reflected floor division map to numpy functions applied with swapped arguments (the numpy versions do not raise on division by zero); all others use the testing module's mapping.
        short_opname = op_name.strip("_")
        if short_opname == "rtruediv":
            # use the numpy version that won't raise on division by zero

            def rtruediv(x, y):
                # [FUNCTION DOC]
                # HELPER: swapped-argument numpy divide used for reflected true division.
                return np.divide(y, x)

            return rtruediv
        elif short_opname == "rfloordiv":
            return lambda x, y: np.floor_divide(y, x)

        return tm.get_op_from_name(op_name)

    # TODO: use EA._cast_pointwise_result, same with other test files that
    #  override this
    def _cast_pointwise_result(self, op_name: str, obj, other, pointwise_result):
        # [FUNCTION DOC]
        # HELPER: adjusts the expected result of the pointwise (combine) path. A source TODO says to use the array's own cast method instead. Parameters: op_name, obj, other, pointwise_result. Comparison operators: returns the result cast to
        # the pyarrow boolean. Otherwise, for plain integer others (except floor division) returns the result cast to 64-bit float (when the original is floating) or 64-bit integer; for plain float others returns 64-bit float. For other cases, results are
        # returned unchanged (the base class upcasts to 64-bit while the arrow array keeps its type) unless the arrow type is a case where the result type should be kept: for a Series other (not frame): floating, integer (except true division and its reflection), duration, timestamp, date and
        # decimal types; otherwise floor division of integers, durations, timestamps, dates and decimals. For these it rebuilds the expected from the arrow array: decimals cast the expected to the dtype that the real operation produces (a 64-bit float for power with a decimal scalar or a same-dtype Series), else the arrow array is cast to the original arrow type, wrapped in the original array class and returned as a Series or DataFrame (with index and columns kept).
        # BaseOpsUtil._combine can upcast expected dtype
        # (because it generates expected on python scalars)
        # while ArrowExtensionArray maintains original type
        expected = pointwise_result

        if op_name in ["eq", "ne", "lt", "le", "gt", "ge"]:
            return pointwise_result.astype("boolean[pyarrow]")

        original_dtype = tm.get_dtype(expected)

        was_frame = False
        if isinstance(expected, pd.DataFrame):
            was_frame = True
            expected_data = expected.iloc[:, 0]
        else:
            expected_data = expected

        # the pointwise method will have retained our original dtype, while
        #  the op(ser, other) version will have cast to 64bit
        if type(other) is int and op_name not in ["__floordiv__"]:
            if original_dtype.kind == "f":
                return expected.astype("float64[pyarrow]")
            else:
                return expected.astype("int64[pyarrow]")
        elif type(other) is float:
            return expected.astype("float64[pyarrow]")

        # error: Item "ExtensionDtype" of "dtype[Any] | ExtensionDtype" has
        #  no attribute "pyarrow_dtype"
        orig_pa_type = original_dtype.pyarrow_dtype  # type: ignore[union-attr]
        if not was_frame and isinstance(other, pd.Series):
            # i.e. test_arith_series_with_array
            if not (
                pa.types.is_floating(orig_pa_type)
                or (
                    pa.types.is_integer(orig_pa_type)
                    and op_name not in ["__truediv__", "__rtruediv__"]
                )
                or pa.types.is_duration(orig_pa_type)
                or pa.types.is_timestamp(orig_pa_type)
                or pa.types.is_date(orig_pa_type)
                or pa.types.is_decimal(orig_pa_type)
            ):
                # base class _combine always returns int64, while
                #  ArrowExtensionArray does not upcast
                return expected
        elif not (
            (op_name == "__floordiv__" and pa.types.is_integer(orig_pa_type))
            or pa.types.is_duration(orig_pa_type)
            or pa.types.is_timestamp(orig_pa_type)
            or pa.types.is_date(orig_pa_type)
            or pa.types.is_decimal(orig_pa_type)
        ):
            # base class _combine always returns int64, while
            #  ArrowExtensionArray does not upcast
            return expected

        pa_expected = pa.array(expected_data._values)

        if pa.types.is_decimal(pa_expected.type) and pa.types.is_decimal(orig_pa_type):
            # decimal precision can resize in the result type depending on data
            # just compare the float values
            alt = getattr(obj, op_name)(other)
            alt_dtype = tm.get_dtype(alt)
            assert isinstance(alt_dtype, ArrowDtype)
            if op_name == "__pow__" and isinstance(other, Decimal):
                # TODO: would it make more sense to retain Decimal here?
                alt_dtype = ArrowDtype(pa.float64())
            elif (
                op_name == "__pow__"
                and isinstance(other, pd.Series)
                and other.dtype == original_dtype
            ):
                # TODO: would it make more sense to retain Decimal here?
                alt_dtype = ArrowDtype(pa.float64())
            else:
                assert pa.types.is_decimal(alt_dtype.pyarrow_dtype)
            return expected.astype(alt_dtype)

        else:
            pa_expected = pa_expected.cast(orig_pa_type)

        pd_expected = type(expected_data._values)(pa_expected)
        if was_frame:
            expected = pd.DataFrame(
                pd_expected, index=expected.index, columns=expected.columns
            )
        else:
            expected = pd.Series(pd_expected)
        return expected

    def _is_temporal_supported(self, opname, pa_dtype):
        # [FUNCTION DOC]
        # HELPER: reports whether the operation is supported for temporal arrow types: addition and reflected addition on durations; true and floor division (and reflected) on durations with pyarrow 14 or newer; subtraction and reflected subtraction on any temporal type.
        return (
            (
                opname in ("__add__", "__radd__")
                or (
                    opname
                    in ("__truediv__", "__rtruediv__", "__floordiv__", "__rfloordiv__")
                    and not pa_version_under14p0
                )
            )
            and pa.types.is_duration(pa_dtype)
        ) or (opname in ("__sub__", "__rsub__") and pa.types.is_temporal(pa_dtype))

    def _get_expected_exception(
        self, op_name: str, obj, other
    ) -> type[Exception] | tuple[type[Exception], ...] | None:
        # [FUNCTION DOC]
        # HELPER: the expected exception for an operation on arrow data. Divmod forms expect not-implemented or type error; modulus forms the same; temporal-supported operations expect none; addition on string and binary expects none; any type other than floating, integer and decimal expects a type error; the rest none.
        if op_name in ("__divmod__", "__rdivmod__"):
            return (NotImplementedError, TypeError)

        exc: type[Exception] | tuple[type[Exception], ...] | None
        dtype = tm.get_dtype(obj)
        # error: Item "dtype[Any]" of "dtype[Any] | ExtensionDtype" has no
        # attribute "pyarrow_dtype"
        pa_dtype = dtype.pyarrow_dtype  # type: ignore[union-attr]

        arrow_temporal_supported = self._is_temporal_supported(op_name, pa_dtype)
        if op_name in {
            "__mod__",
            "__rmod__",
        }:
            exc = (NotImplementedError, TypeError)
        elif arrow_temporal_supported:
            exc = None
        elif op_name in ["__add__", "__radd__"] and (
            pa.types.is_string(pa_dtype) or pa.types.is_binary(pa_dtype)
        ):
            exc = None
        elif not (
            pa.types.is_floating(pa_dtype)
            or pa.types.is_integer(pa_dtype)
            or pa.types.is_decimal(pa_dtype)
        ):
            exc = TypeError
        else:
            exc = None
        return exc

    def _get_arith_xfail_marker(self, opname, pa_dtype):
        # [FUNCTION DOC]
        # HELPER: returns an expected-failure marker (or none) for an arithmetic operator and arrow type. Reflected power with float, integer or decimal: fails because of NA power semantics (GH#29997). Temporal-supported operations on time types, or division operators on durations: type error. Reflected floor division with integer or decimal and reflected true division with decimals: arrow invalid error (divide by zero).
        # KNOWN ISSUE: the type error reason message lacks a space (between and pd.NA run together).
        mark = None

        arrow_temporal_supported = self._is_temporal_supported(opname, pa_dtype)

        if opname == "__rpow__" and (
            pa.types.is_floating(pa_dtype)
            or pa.types.is_integer(pa_dtype)
            or pa.types.is_decimal(pa_dtype)
        ):
            mark = pytest.mark.xfail(
                reason=(
                    f"GH#29997: 1**pandas.NA == 1 while 1**pyarrow.NA == NULL "
                    f"for {pa_dtype}"
                )
            )
        elif arrow_temporal_supported and (
            pa.types.is_time(pa_dtype)
            or (
                opname
                in ("__truediv__", "__rtruediv__", "__floordiv__", "__rfloordiv__")
                and pa.types.is_duration(pa_dtype)
            )
        ):
            mark = pytest.mark.xfail(
                raises=TypeError,
                reason=(
                    f"{opname} not supported betweenpd.NA and {pa_dtype} Python scalar"
                ),
            )
        elif opname == "__rfloordiv__" and (
            pa.types.is_integer(pa_dtype) or pa.types.is_decimal(pa_dtype)
        ):
            mark = pytest.mark.xfail(
                raises=pa.ArrowInvalid,
                reason="divide by 0",
            )
        elif opname == "__rtruediv__" and pa.types.is_decimal(pa_dtype):
            mark = pytest.mark.xfail(
                raises=pa.ArrowInvalid,
                reason="divide by 0",
            )

        return mark

    def test_arith_series_with_scalar(self, data, all_arithmetic_operators, request):
        # [FUNCTION DOC]
        # Skips reflected modulus for binary types (Python string formatting); applies the marker from the helper if any; delegates to the parent series-scalar arithmetic test.
        pa_dtype = data.dtype.pyarrow_dtype

        if all_arithmetic_operators == "__rmod__" and pa.types.is_binary(pa_dtype):
            pytest.skip("Skip testing Python string formatting")

        mark = self._get_arith_xfail_marker(all_arithmetic_operators, pa_dtype)
        if mark is not None:
            request.applymarker(mark)

        super().test_arith_series_with_scalar(data, all_arithmetic_operators)

    def test_arith_frame_with_scalar(self, data, all_arithmetic_operators, request):
        # [FUNCTION DOC]
        # Skips reflected modulus for string and binary types; applies the helper's expected-failure marker; delegates to the parent frame-scalar arithmetic test.
        pa_dtype = data.dtype.pyarrow_dtype

        if all_arithmetic_operators == "__rmod__" and (
            pa.types.is_string(pa_dtype) or pa.types.is_binary(pa_dtype)
        ):
            pytest.skip("Skip testing Python string formatting")

        mark = self._get_arith_xfail_marker(all_arithmetic_operators, pa_dtype)
        if mark is not None:
            request.applymarker(mark)

        super().test_arith_frame_with_scalar(data, all_arithmetic_operators)

    def test_arith_series_with_array(self, data, all_arithmetic_operators, request):
        # [FUNCTION DOC]
        # Marks expected-to-fail (arrow invalid error: the checked subtraction raises on overflow) for subtraction and reflected subtraction with unsigned integers; applies the helper's marker. Builds a Series of the data and another Series of the first element repeated with the same dtype (using a Series-constructed array because a scalar may be a Python scalar), then runs the parent's opname check.
        pa_dtype = data.dtype.pyarrow_dtype

        if all_arithmetic_operators in (
            "__sub__",
            "__rsub__",
        ) and pa.types.is_unsigned_integer(pa_dtype):
            request.applymarker(
                pytest.mark.xfail(
                    raises=pa.ArrowInvalid,
                    reason=(
                        f"Implemented pyarrow.compute.subtract_checked "
                        f"which raises on overflow for {pa_dtype}"
                    ),
                )
            )

        mark = self._get_arith_xfail_marker(all_arithmetic_operators, pa_dtype)
        if mark is not None:
            request.applymarker(mark)

        op_name = all_arithmetic_operators
        ser = pd.Series(data)
        # pd.Series([ser.iloc[0]] * len(ser)) may not return ArrowExtensionArray
        # since ser.iloc[0] is a python scalar
        other = pd.Series(pd.array([ser.iloc[0]] * len(ser), dtype=data.dtype))

        self.check_opname(ser, op_name, other)

    def test_add_series_with_extension_array(self, data, request):
        # [FUNCTION DOC]
        # Marks expected-to-fail (arrow invalid error, overflow) for 8-bit integers; delegates to the parent test.
        pa_dtype = data.dtype.pyarrow_dtype

        if pa_dtype.equals("int8"):
            request.applymarker(
                pytest.mark.xfail(
                    raises=pa.ArrowInvalid,
                    reason=f"raises on overflow for {pa_dtype}",
                )
            )
        super().test_add_series_with_extension_array(data)

    def test_invalid_other_comp(self, data, comparison_op):
        # [FUNCTION DOC]
        # GH 48833. Comparing the data with a plain object using the comparison operator fixture raises not-implemented with a message that the comparison is not implemented for the object class.
        # GH 48833
        with pytest.raises(
            NotImplementedError, match=".* not implemented for <class 'object'>"
        ):
            comparison_op(data, object())

    @pytest.mark.parametrize("masked_dtype", ["boolean", "Int64", "Float64"])
    def test_comp_masked_numpy(self, masked_dtype, comparison_op):
        # [FUNCTION DOC]
        # GH 52625. Parametrized over masked dtypes boolean, Int64, Float64: compares a pyarrow-backed Series [1, 0, missing] with a masked Series using the comparison fixture. The result is an arrow boolean Series: [False, False, missing] for less-than, greater-than, not-equal and [True, True, missing] for others.
        # GH 52625
        data = [1, 0, None]
        ser_masked = pd.Series(data, dtype=masked_dtype)
        ser_pa = pd.Series(data, dtype=f"{masked_dtype.lower()}[pyarrow]")
        result = comparison_op(ser_pa, ser_masked)
        if comparison_op in [operator.lt, operator.gt, operator.ne]:
            exp = [False, False, None]
        else:
            exp = [True, True, None]
        expected = pd.Series(exp, dtype=ArrowDtype(pa.bool_()))
        tm.assert_series_equal(result, expected)

    def test_loc_setitem_with_expansion_preserves_ea_index_dtype(self, data, request):
        # [FUNCTION DOC]
        # Marks expected-to-fail for date types (GH#62343 incorrectly casts to millisecond timestamp), then delegates to the parent test.
        pa_dtype = data.dtype.pyarrow_dtype
        if pa.types.is_date(pa_dtype):
            mark = pytest.mark.xfail(
                reason="GH#62343 incorrectly casts to timestamp[ms][pyarrow]"
            )
            request.applymarker(mark)
        super().test_loc_setitem_with_expansion_preserves_ea_index_dtype(data)


class TestLogicalOps:
    # [CLASS DOC]
    # Test class for Kleene (three-valued) logical operators on arrow boolean Series.
    """Various Series and DataFrame logical ops methods."""

    def test_kleene_or(self):
        # [FUNCTION DOC]
        # Verifies the or of two boolean arrow Series covering all nine combinations of true, false, missing gives the Kleene result, is commutative, and does not mutate either input.
        a = pd.Series([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean[pyarrow]")
        b = pd.Series([True, False, None] * 3, dtype="boolean[pyarrow]")
        result = a | b
        expected = pd.Series(
            [True, True, True, True, False, None, True, None, None],
            dtype="boolean[pyarrow]",
        )
        tm.assert_series_equal(result, expected)

        result = b | a
        tm.assert_series_equal(result, expected)

        # ensure we haven't mutated anything inplace
        tm.assert_series_equal(
            a,
            pd.Series([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean[pyarrow]"),
        )
        tm.assert_series_equal(
            b, pd.Series([True, False, None] * 3, dtype="boolean[pyarrow]")
        )

    @pytest.mark.parametrize(
        "other, expected",
        [
            (None, [True, None, None]),
            (pd.NA, [True, None, None]),
            (True, [True, True, True]),
            (np.bool_(True), [True, True, True]),
            (False, [True, False, None]),
            (np.bool_(False), [True, False, None]),
        ],
    )
    def test_kleene_or_scalar(self, other, expected):
        # [FUNCTION DOC]
        # Parametrized over scalar other (None, NA, True, false and numpy bools) and expected list. Verifies or of a [true, false, missing] arrow Series with the scalar in both operand orders gives the expected Kleene result and does not mutate the Series.
        a = pd.Series([True, False, None], dtype="boolean[pyarrow]")
        result = a | other
        expected = pd.Series(expected, dtype="boolean[pyarrow]")
        tm.assert_series_equal(result, expected)

        result = other | a
        tm.assert_series_equal(result, expected)

        # ensure we haven't mutated anything inplace
        tm.assert_series_equal(
            a, pd.Series([True, False, None], dtype="boolean[pyarrow]")
        )

    def test_kleene_and(self):
        # [FUNCTION DOC]
        # Verifies the and of two boolean arrow Series covering all nine combinations gives the Kleene result, is commutative, and does not mutate inputs.
        a = pd.Series([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean[pyarrow]")
        b = pd.Series([True, False, None] * 3, dtype="boolean[pyarrow]")
        result = a & b
        expected = pd.Series(
            [True, False, None, False, False, False, None, False, None],
            dtype="boolean[pyarrow]",
        )
        tm.assert_series_equal(result, expected)

        result = b & a
        tm.assert_series_equal(result, expected)

        # ensure we haven't mutated anything inplace
        tm.assert_series_equal(
            a,
            pd.Series([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean[pyarrow]"),
        )
        tm.assert_series_equal(
            b, pd.Series([True, False, None] * 3, dtype="boolean[pyarrow]")
        )

    @pytest.mark.parametrize(
        "other, expected",
        [
            (None, [None, False, None]),
            (pd.NA, [None, False, None]),
            (True, [True, False, None]),
            (False, [False, False, False]),
            (np.bool_(True), [True, False, None]),
            (np.bool_(False), [False, False, False]),
        ],
    )
    def test_kleene_and_scalar(self, other, expected):
        # [FUNCTION DOC]
        # Parametrized over scalar other (None, NA, True, false and numpy bools). Verifies and with the scalar in both orders gives the expected Kleene result and does not mutate the Series.
        a = pd.Series([True, False, None], dtype="boolean[pyarrow]")
        result = a & other
        expected = pd.Series(expected, dtype="boolean[pyarrow]")
        tm.assert_series_equal(result, expected)

        result = other & a
        tm.assert_series_equal(result, expected)

        # ensure we haven't mutated anything inplace
        tm.assert_series_equal(
            a, pd.Series([True, False, None], dtype="boolean[pyarrow]")
        )

    def test_kleene_xor(self):
        # [FUNCTION DOC]
        # Verifies the exclusive or of two boolean arrow Series covering all nine combinations: missing whenever either side is missing.
        a = pd.Series([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean[pyarrow]")
        b = pd.Series([True, False, None] * 3, dtype="boolean[pyarrow]")
        result = a ^ b
        expected = pd.Series(
            [False, True, None, True, False, None, None, None, None],
            dtype="boolean[pyarrow]",
        )
        tm.assert_series_equal(result, expected)

        result = b ^ a
        tm.assert_series_equal(result, expected)

        # ensure we haven't mutated anything inplace
        tm.assert_series_equal(
            a,
            pd.Series([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean[pyarrow]"),
        )
        tm.assert_series_equal(
            b, pd.Series([True, False, None] * 3, dtype="boolean[pyarrow]")
        )

    @pytest.mark.parametrize(
        "other, expected",
        [
            (None, [None, None, None]),
            (pd.NA, [None, None, None]),
            (True, [False, True, None]),
            (np.bool_(True), [False, True, None]),
            (np.bool_(False), [True, False, None]),
        ],
    )
    def test_kleene_xor_scalar(self, other, expected):
        # [FUNCTION DOC]
        # Parametrized over scalar others (None, NA, True and numpy true/false). Verifies exclusive or of a [true, false, missing] arrow Series with the scalar in both operand orders gives the expected results (all missing for a missing scalar, negation for true, identity for false) and does not mutate the Series.
        a = pd.Series([True, False, None], dtype="boolean[pyarrow]")
        result = a ^ other
        expected = pd.Series(expected, dtype="boolean[pyarrow]")
        tm.assert_series_equal(result, expected)

        result = other ^ a
        tm.assert_series_equal(result, expected)

        # ensure we haven't mutated anything inplace
        tm.assert_series_equal(
            a, pd.Series([True, False, None], dtype="boolean[pyarrow]")
        )

    @pytest.mark.parametrize(
        "op, exp",
        [
            ["__and__", True],
            ["__or__", True],
            ["__xor__", False],
        ],
    )
    def test_logical_masked_numpy(self, op, exp):
        # [FUNCTION DOC]
        # GH 52625. Parametrized over the and, or, exclusive-or operator method names with expected first element. Applies the operator of an arrow boolean Series [true, false, missing] with a masked boolean Series; the result is an arrow boolean Series with the expected first value, false, and missing.
        # GH 52625
        data = [True, False, None]
        ser_masked = pd.Series(data, dtype="boolean")
        ser_pa = pd.Series(data, dtype="boolean[pyarrow]")
        result = getattr(ser_pa, op)(ser_masked)
        expected = pd.Series([exp, False, None], dtype=ArrowDtype(pa.bool_()))
        tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("pa_type", tm.ALL_INT_PYARROW_DTYPES)
def test_bitwise(pa_type):
    # [FUNCTION DOC]
    # GH 54495. Parametrized over every integer arrow type. Verifies bitwise or, and, exclusive or of two Series with nulls give nulls where either is null and the plain integer result elsewhere, and bitwise inversion equals the inversion of the zero-filled values with nulls restored by a mask.
    # GH 54495
    dtype = ArrowDtype(pa_type)
    left = pd.Series([1, None, 3, 4], dtype=dtype)
    right = pd.Series([None, 3, 5, 4], dtype=dtype)

    result = left | right
    expected = pd.Series([None, None, 3 | 5, 4 | 4], dtype=dtype)
    tm.assert_series_equal(result, expected)

    result = left & right
    expected = pd.Series([None, None, 3 & 5, 4 & 4], dtype=dtype)
    tm.assert_series_equal(result, expected)

    result = left ^ right
    expected = pd.Series([None, None, 3 ^ 5, 4 ^ 4], dtype=dtype)
    tm.assert_series_equal(result, expected)

    result = ~left
    expected = ~(left.fillna(0).to_numpy())
    expected = pd.Series(expected, dtype=dtype).mask(left.isnull())
    tm.assert_series_equal(result, expected)


def test_arrowdtype_construct_from_string_type_with_unsupported_parameters():
    # [FUNCTION DOC]
    # Verifies constructing the arrow dtype from a string that carries unsupported type parameters (an unknown type name with unit and time zone, or a decimal with precision and scale) raises not-implemented with a message about passing pyarrow type parameters.
    with pytest.raises(NotImplementedError, match="Passing pyarrow type"):
        ArrowDtype.construct_from_string("not_a_real_dype[s, tz=UTC][pyarrow]")

    with pytest.raises(NotImplementedError, match="Passing pyarrow type"):
        ArrowDtype.construct_from_string("decimal(7, 2)[pyarrow]")


def test_arrowdtype_construct_from_string_supports_dt64tz():
    # [FUNCTION DOC]
    # GH#50689: verifies constructing the dtype from a second-resolution UTC timestamp type string equals the dtype built from the corresponding arrow timestamp type.
    # as of GH#50689, timestamptz is supported
    dtype = ArrowDtype.construct_from_string("timestamp[s, tz=UTC][pyarrow]")
    expected = ArrowDtype(pa.timestamp("s", "UTC"))
    assert dtype == expected


def test_arrowdtype_construct_from_string_type_only_one_pyarrow():
    # [FUNCTION DOC]
    # GH#51225. Constructing a Series with a dtype string that has a second pyarrow suffix raises not-implemented with the message that pyarrow type specific parameters in the string are not supported.
    # GH#51225
    invalid = "int64[pyarrow]foobar[pyarrow]"
    msg = (
        r"Passing pyarrow type specific parameters \(\[pyarrow\]\) in the "
        r"string is not supported\."
    )
    with pytest.raises(NotImplementedError, match=msg):
        pd.Series(range(3), dtype=invalid)


def test_arrow_string_multiplication():
    # [FUNCTION DOC]
    # GH 56537. Multiplying a string arrow Series by an arrow integer Series repeats strings (negative counts give the empty string) and is commutative.
    # GH 56537
    binary = pd.Series(["abc", "defg"], dtype=ArrowDtype(pa.string()))
    repeat = pd.Series([2, -2], dtype="int64[pyarrow]")
    result = binary * repeat
    expected = pd.Series(["abcabc", ""], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)
    reflected_result = repeat * binary
    tm.assert_series_equal(result, reflected_result)


def test_arrow_string_multiplication_scalar_repeat():
    # [FUNCTION DOC]
    # Multiplying a string arrow Series by an integer scalar repeats each string, in both operand orders.
    binary = pd.Series(["abc", "defg"], dtype=ArrowDtype(pa.string()))
    result = binary * 2
    expected = pd.Series(["abcabc", "defgdefg"], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)
    reflected_result = 2 * binary
    tm.assert_series_equal(reflected_result, expected)


def test_arrow_string_addition_mixed_string_types():
    # [FUNCTION DOC]
    # GH issue 65220. Adding a regular-string arrow Series and a large-string arrow Series gives the large-string dtype, concatenated in operand order, with missing propagating.
    # https://github.com/pandas-dev/pandas/issues/65220
    left = pd.Series(["a", None], dtype=ArrowDtype(pa.string()))
    right = pd.Series(["b", "c"], dtype=ArrowDtype(pa.large_string()))

    result = left + right
    expected = pd.Series(["ab", None], dtype=ArrowDtype(pa.large_string()))
    tm.assert_series_equal(result, expected)

    reflected_result = right + left
    expected_reflected = pd.Series(["ba", None], dtype=ArrowDtype(pa.large_string()))
    tm.assert_series_equal(reflected_result, expected_reflected)


@pytest.mark.parametrize("string_type", [pa.string(), pa.large_string()])
def test_arrow_string_addition_mixed_with_binary_raises(string_type):
    # [FUNCTION DOC]
    # Parametrized over the string and large-string arrow types. Adding a string Series and a binary arrow Series raises a type error with the message that the add operation is not supported for the two dtypes, in both operand orders (messages in matching order).
    left = pd.Series(["a", None], dtype=ArrowDtype(string_type))
    right = pd.Series([b"b", b"c"], dtype=ArrowDtype(pa.binary()))

    msg = (
        f"operation 'add' not supported for dtype '{left.dtype}' "
        f"with dtype '{right.dtype}'"
    )
    with pytest.raises(TypeError, match=re.escape(msg)):
        left + right

    reflected_msg = (
        f"operation 'add' not supported for dtype '{right.dtype}' "
        f"with dtype '{left.dtype}'"
    )
    with pytest.raises(TypeError, match=re.escape(reflected_msg)):
        right + left


@pytest.mark.parametrize(
    "interpolation", ["linear", "lower", "higher", "nearest", "midpoint"]
)
@pytest.mark.parametrize("quantile", [0.5, [0.5, 0.5]])
def test_quantile(data, interpolation, quantile, request):
    # [FUNCTION DOC]
    # Parametrized over interpolation methods (linear, lower, higher, nearest, midpoint) and scalar or list quantile. On three copies of the first element. For string, binary and boolean types expects an arrow not-implemented error about no kernel matching input types, then returns. For types other than numeric and temporal marks expected failure. Computes the quantile:
    # for timestamp types with interpolation other than lower/higher the result is compared to the numpy datetime analogue (to avoid rounding error; microsecond handling differs for scalar and list); otherwise a scalar quantile equals the first element and a list quantile equals a Series of the element repeated indexed by the quantiles (numeric types cast to 64-bit float).
    pa_dtype = data.dtype.pyarrow_dtype

    data = data.take([0, 0, 0])
    ser = pd.Series(data)

    if (
        pa.types.is_string(pa_dtype)
        or pa.types.is_binary(pa_dtype)
        or pa.types.is_boolean(pa_dtype)
    ):
        # For string, bytes, and bool, we don't *expect* to have quantile work
        # Note this matches the non-pyarrow behavior
        msg = r"Function 'quantile' has no kernel matching input types \(.*\)"
        with pytest.raises(pa.ArrowNotImplementedError, match=msg):
            ser.quantile(q=quantile, interpolation=interpolation)
        return

    if (
        pa.types.is_integer(pa_dtype)
        or pa.types.is_floating(pa_dtype)
        or pa.types.is_decimal(pa_dtype)
    ):
        pass
    elif pa.types.is_temporal(data._pa_array.type):
        pass
    else:
        request.applymarker(
            pytest.mark.xfail(
                raises=pa.ArrowNotImplementedError,
                reason=f"quantile not supported by pyarrow for {pa_dtype}",
            )
        )
    data = data.take([0, 0, 0])
    ser = pd.Series(data)
    result = ser.quantile(q=quantile, interpolation=interpolation)

    if pa.types.is_timestamp(pa_dtype) and interpolation not in ["lower", "higher"]:
        # rounding error will make the check below fail
        #  (e.g. '2020-01-01 01:01:01.000001' vs '2020-01-01 01:01:01.000001024'),
        #  so we'll check for now that we match the numpy analogue
        if pa_dtype.tz:
            pd_dtype = f"M8[{pa_dtype.unit}, {pa_dtype.tz}]"
        else:
            pd_dtype = f"M8[{pa_dtype.unit}]"
        ser_np = ser.astype(pd_dtype)

        expected = ser_np.quantile(q=quantile, interpolation=interpolation)
        if quantile == 0.5:
            if pa_dtype.unit == "us":
                expected = expected.to_pydatetime(warn=False)
            assert result == expected
        else:
            if pa_dtype.unit == "us":
                expected = expected.dt.floor("us")
            tm.assert_series_equal(result, expected.astype(data.dtype))
        return

    if quantile == 0.5:
        assert result == data[0]
    else:
        # Just check the values
        expected = pd.Series(data.take([0, 0]), index=[0.5, 0.5])
        if (
            pa.types.is_integer(pa_dtype)
            or pa.types.is_floating(pa_dtype)
            or pa.types.is_decimal(pa_dtype)
        ):
            expected = expected.astype("float64[pyarrow]")
            result = result.astype("float64[pyarrow]")
        tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "take_idx, exp_idx",
    [[[0, 0, 2, 2, 4, 4], [4, 0]], [[0, 0, 0, 2, 4, 4], [0]]],
    ids=["multi_mode", "single_mode"],
)
def test_mode_dropna_true(data_for_grouping, take_idx, exp_idx):
    # [FUNCTION DOC]
    # Parametrized over (take indices, expected indices) with ids multi_mode and single_mode. Takes elements of the grouping data, computes the mode dropping missing values, and compares with the Series of the expected elements.
    data = data_for_grouping.take(take_idx)
    ser = pd.Series(data)
    result = ser.mode(dropna=True)
    expected = pd.Series(data_for_grouping.take(exp_idx))
    tm.assert_series_equal(result, expected)


def test_mode_dropna_false_mode_na(data):
    # [FUNCTION DOC]
    # GH 50982. A Series with two nulls and one value has mode null when missing values are not dropped; a Series of one value and null has both as modes.
    # GH 50982
    more_nans = pd.Series([None, None, data[0]], dtype=data.dtype)
    result = more_nans.mode(dropna=False)
    expected = pd.Series([None], dtype=data.dtype)
    tm.assert_series_equal(result, expected)

    expected = pd.Series([data[0], None], dtype=data.dtype)
    result = expected.mode(dropna=False)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "arrow_dtype, expected_type",
    [
        [pa.binary(), bytes],
        [pa.binary(16), bytes],
        [pa.large_binary(), bytes],
        [pa.large_string(), str],
        [pa.list_(pa.int64()), list],
        [pa.large_list(pa.int64()), list],
        [pa.map_(pa.string(), pa.int64()), list],
        [pa.struct([("f1", pa.int8()), ("f2", pa.string())]), dict],
        [pa.dictionary(pa.int64(), pa.int64()), CategoricalDtypeType],
    ],
)
def test_arrow_dtype_type(arrow_dtype, expected_type):
    # [FUNCTION DOC]
    # GH 51845. Parametrized over binary (fixed and variable size, large), large string, list, large list, map, struct and dictionary arrow types: verifies the dtype's scalar type is bytes, string, list, dict, or the categorical type respectively. A TODO says it is redundant with the scalar-get test once the dtype exists in the data fixture.
    # GH 51845
    # TODO: Redundant with test_getitem_scalar once arrow_dtype exists in data fixture
    assert ArrowDtype(arrow_dtype).type == expected_type


def test_is_bool_dtype():
    # [FUNCTION DOC]
    # GH 22667. An arrow boolean array is recognized as a boolean dtype and as a boolean indexer, and indexing a Series with it equals indexing with the numpy conversion.
    # GH 22667
    data = ArrowExtensionArray(pa.array([True, False, True]))
    assert is_bool_dtype(data)
    assert pd.core.common.is_bool_indexer(data)
    s = pd.Series(range(len(data)))
    result = s[data]
    expected = s[np.asarray(data)]
    tm.assert_series_equal(result, expected)


def test_is_numeric_dtype(data):
    # [FUNCTION DOC]
    # GH 50563. For floating, integer and decimal arrow types the data is numeric; other types are not.
    # GH 50563
    pa_type = data.dtype.pyarrow_dtype
    if (
        pa.types.is_floating(pa_type)
        or pa.types.is_integer(pa_type)
        or pa.types.is_decimal(pa_type)
    ):
        assert is_numeric_dtype(data)
    else:
        assert not is_numeric_dtype(data)


def test_is_integer_dtype(data):
    # [FUNCTION DOC]
    # GH 50667. The data is an integer dtype exactly when the arrow type is an integer type.
    # GH 50667
    pa_type = data.dtype.pyarrow_dtype
    if pa.types.is_integer(pa_type):
        assert is_integer_dtype(data)
    else:
        assert not is_integer_dtype(data)


def test_is_signed_integer_dtype(data):
    # [FUNCTION DOC]
    # The data is a signed-integer dtype exactly when the arrow type is signed integer.
    pa_type = data.dtype.pyarrow_dtype
    if pa.types.is_signed_integer(pa_type):
        assert is_signed_integer_dtype(data)
    else:
        assert not is_signed_integer_dtype(data)


def test_is_unsigned_integer_dtype(data):
    # [FUNCTION DOC]
    # The data is an unsigned-integer dtype exactly when the arrow type is unsigned integer.
    pa_type = data.dtype.pyarrow_dtype
    if pa.types.is_unsigned_integer(pa_type):
        assert is_unsigned_integer_dtype(data)
    else:
        assert not is_unsigned_integer_dtype(data)


def test_is_datetime64_any_dtype(data):
    # [FUNCTION DOC]
    # The data is a datetime64-any dtype exactly when the arrow type is a timestamp or date.
    pa_type = data.dtype.pyarrow_dtype
    if pa.types.is_timestamp(pa_type) or pa.types.is_date(pa_type):
        assert is_datetime64_any_dtype(data)
    else:
        assert not is_datetime64_any_dtype(data)


def test_is_float_dtype(data):
    # [FUNCTION DOC]
    # The data is a float dtype exactly when the arrow type is floating.
    pa_type = data.dtype.pyarrow_dtype
    if pa.types.is_floating(pa_type):
        assert is_float_dtype(data)
    else:
        assert not is_float_dtype(data)


def test_pickle_roundtrip(data):
    # [FUNCTION DOC]
    # GH 42600. Pickling a Series of the data and its two-element head: the full pickle is larger than the sliced one, and both round trip to equal Series.
    # GH 42600
    expected = pd.Series(data)
    expected_sliced = expected.head(2)
    full_pickled = pickle.dumps(expected)
    sliced_pickled = pickle.dumps(expected_sliced)

    assert len(full_pickled) > len(sliced_pickled)

    result = pickle.loads(full_pickled)
    tm.assert_series_equal(result, expected)

    result_sliced = pickle.loads(sliced_pickled)
    tm.assert_series_equal(result_sliced, expected_sliced)


def test_astype_from_non_pyarrow(data):
    # [FUNCTION DOC]
    # GH49795. Casting a non-arrow array created from the data's numpy conversion to the arrow dtype gives an arrow-typed array equal to the data.
    # GH49795
    np_arr = data.to_numpy()
    pd_array = pd.array(np_arr, dtype=np_arr.dtype)
    result = pd_array.astype(data.dtype)
    assert not isinstance(pd_array.dtype, ArrowDtype)
    assert isinstance(result.dtype, ArrowDtype)
    tm.assert_extension_array_equal(result, data)


def test_astype_float_from_non_pyarrow_str():
    # [FUNCTION DOC]
    # GH50430. Casting a Series of numeric strings to the 64-bit float arrow dtype parses them into floats.
    # GH50430
    ser = pd.Series(["1.0"])
    result = ser.astype("float64[pyarrow]")
    expected = pd.Series([1.0], dtype="float64[pyarrow]")
    tm.assert_series_equal(result, expected)


def test_astype_errors_ignore():
    # [FUNCTION DOC]
    # GH 55399. Casting a 32-bit integer arrow frame to a float arrow type with errors set to ignore gives the original frame unchanged (the target dtype is not a valid type, so the ignore returns the input).
    # GH 55399
    expected = pd.DataFrame({"col": [17000000]}, dtype="int32[pyarrow]")
    result = expected.astype("float[pyarrow]", errors="ignore")
    tm.assert_frame_equal(result, expected)


def test_to_numpy_with_defaults(data, using_nan_is_na):
    # [FUNCTION DOC]
    # GH49973. Default numpy conversion: skipped for duration and timestamp (tested elsewhere); dates expect an object array of date objects, others the numpy view of the arrow array; if the data has nulls and either the dtype is not numeric or NaN is not treated as NA, the expected is converted to object with NA at the missing positions.
    # GH49973
    result = data.to_numpy()

    pa_type = data._pa_array.type
    if pa.types.is_duration(pa_type) or pa.types.is_timestamp(pa_type):
        pytest.skip("Tested in test_to_numpy_temporal")
    elif pa.types.is_date(pa_type):
        expected = np.array(list(data))
    else:
        expected = np.array(data._pa_array)

    if data._hasna and (not is_numeric_dtype(data.dtype) or not using_nan_is_na):
        expected = expected.astype(object)
        expected[pd.isna(data)] = pd.NA

    tm.assert_numpy_array_equal(result, expected)


def test_to_numpy_int_with_na(using_nan_is_na):
    # [FUNCTION DOC]
    # GH51227. A 64-bit integer arrow array with a null converts to a numpy object array with NA when NaN is not treated as NA, else to a float array with NaN whose first element is a float.
    # GH51227: ensure to_numpy does not convert int to float
    data = [1, None]
    arr = pd.array(data, dtype="int64[pyarrow]")
    result = arr.to_numpy()
    if not using_nan_is_na:
        expected = np.array([1, pd.NA], dtype=object)
    else:
        expected = np.array([1, np.nan])
        assert isinstance(result[0], float)
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("na_val, exp", [(lib.no_default, np.nan), (1, 1)])
def test_to_numpy_null_array(na_val, exp):
    # [FUNCTION DOC]
    # GH#52443. Parametrized over (missing-value argument unspecified/1, expected fill: NaN/1). Converting an all-NA null-typed arrow array to numpy 64-bit float uses NaN by default or the given value.
    # GH#52443
    arr = pd.array([pd.NA, pd.NA], dtype="null[pyarrow]")
    result = arr.to_numpy(dtype="float64", na_value=na_val)
    expected = np.array([exp] * 2, dtype="float64")
    tm.assert_numpy_array_equal(result, expected)


def test_to_numpy_null_array_no_dtype():
    # [FUNCTION DOC]
    # GH#52443. Converting an all-NA null-typed arrow array to numpy with no dtype gives an object array of NA values.
    # GH#52443
    arr = pd.array([pd.NA, pd.NA], dtype="null[pyarrow]")
    result = arr.to_numpy(dtype=None)
    expected = np.array([pd.NA] * 2, dtype="object")
    tm.assert_numpy_array_equal(result, expected)


def test_to_numpy_without_dtype():
    # [FUNCTION DOC]
    # GH 54808. Converting arrow boolean [true, NA] to numpy with an NA replacement of false gives a boolean array; converting arrow 32-bit float [1.0, NA] with replacement 0.0 gives a 32-bit float array.
    # GH 54808
    arr = pd.array([True, pd.NA], dtype="boolean[pyarrow]")
    result = arr.to_numpy(na_value=False)
    expected = np.array([True, False], dtype=np.bool_)
    tm.assert_numpy_array_equal(result, expected)

    arr = pd.array([1.0, pd.NA], dtype="float32[pyarrow]")
    result = arr.to_numpy(na_value=0.0)
    expected = np.array([1.0, 0.0], dtype=np.float32)
    tm.assert_numpy_array_equal(result, expected)


def test_setitem_null_slice(data):
    # [FUNCTION DOC]
    # GH50248. Assigning to a full slice of a copy of the data: a scalar (first element) fills every slot; the reversed data replaces it; a list of the data's values gives back the data.
    # GH50248
    orig = data.copy()

    result = orig.copy()
    result[:] = data[0]
    expected = ArrowExtensionArray._from_sequence(
        [data[0]] * len(data),
        dtype=data.dtype,
    )
    tm.assert_extension_array_equal(result, expected)

    result = orig.copy()
    result[:] = data[::-1]
    expected = data[::-1]
    tm.assert_extension_array_equal(result, expected)

    result = orig.copy()
    result[:] = data.tolist()
    expected = data
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize(
    "dtype, np_dtype",
    [
        ("int64[pyarrow]", "int64"),
        ("uint8[pyarrow]", "uint8"),
        ("double[pyarrow]", "float64"),
        ("timestamp[ns][pyarrow]", "M8[ns]"),
        ("duration[ns][pyarrow]", "m8[ns]"),
    ],
)
def test_setitem_null_slice_no_alias(dtype, np_dtype):
    # [FUNCTION DOC]
    # GH#67990. Parametrized over (arrow dtype string, numpy dtype) for int64, uint8, double, timestamp ns, duration ns. Assigning a numpy array or a Series to a full slice of an all-null arrow array copies the data: later mutation of the source (first element set to the second) must not change the array.
    # GH#67990 the null-slice fast path must not adopt a buffer the caller owns
    expected = pd.array(np.array([10, 20, 30]).astype(np_dtype), dtype=dtype)

    np_values = np.array([10, 20, 30]).astype(np_dtype)
    arr = pd.array([None] * 3, dtype=dtype)
    arr[:] = np_values
    np_values[0] = np_values[1]
    tm.assert_extension_array_equal(arr, expected)

    ser = pd.Series(np.array([10, 20, 30]).astype(np_dtype))
    arr = pd.array([None] * 3, dtype=dtype)
    arr[:] = ser
    ser.iloc[0] = ser.iloc[1]
    tm.assert_extension_array_equal(arr, expected)


def test_setitem_null_slice_no_alias_masked():
    # [FUNCTION DOC]
    # GH#67990. Assigning a masked 64-bit integer array to a full slice of an arrow array copies it (masked arrays are zero-copy through the arrow conversion), so mutating the masked array afterwards does not alter the arrow array.
    # GH#67990 masked arrays are zero-copy through __arrow_array__
    arr = pd.array([None] * 3, dtype="int64[pyarrow]")
    values = pd.array([10, 20, 30], dtype="Int64")
    arr[:] = values
    values[0] = -1
    expected = pd.array([10, 20, 30], dtype="int64[pyarrow]")
    tm.assert_extension_array_equal(arr, expected)


@pytest.mark.parametrize(
    "wrap",
    [
        lambda np_values: pa.array(np_values),
        lambda np_values: pa.chunked_array([pa.array(np_values)]),
        lambda np_values: pa.chunked_array(
            [pa.array(np_values[:1]), pa.array(np_values[1:])]
        ),
        lambda np_values: ArrowExtensionArray(pa.array(np_values)),
        lambda np_values: pd.Series(ArrowExtensionArray(pa.array(np_values))),
    ],
    ids=["array", "chunked", "chunked_multi", "extension_array", "series"],
)
def test_setitem_null_slice_no_alias_pyarrow(wrap):
    # [FUNCTION DOC]
    # GH#67990. Parametrized over ways to wrap numpy values (arrow array, one-chunk chunked array, two-chunk chunked array, arrow extension array, Series of arrow extension array). A pyarrow array is immutable but may be zero-copy over caller-owned numpy memory; after assigning to a full slice of an all-null arrow array and mutating the numpy array, the array still holds the original values, and the number of chunks matches the value's (extension comparison ignores chunking).
    # GH#67990 a pyarrow array is immutable, but its buffers can still be
    #  zero-copy over a numpy array the caller owns
    np_values = np.array([10, 20, 30], dtype="int64")
    arr = pd.array([None] * 3, dtype="int64[pyarrow]")
    value = wrap(np_values)
    arr[:] = value
    np_values[0] = -1
    expected = pd.array([10, 20, 30], dtype="int64[pyarrow]")
    tm.assert_extension_array_equal(arr, expected)
    # assert_extension_array_equal ignores chunking, so pin it separately
    assert arr._pa_array.num_chunks == getattr(value, "num_chunks", 1)


@pytest.mark.parametrize(
    "dtype",
    [
        pd.StringDtype("pyarrow", na_value=np.nan),
        "string[pyarrow]",
        "binary[pyarrow]",
        ArrowDtype(pa.large_string()),
        ArrowDtype(pa.large_binary()),
    ],
)
def test_setitem_null_slice_string_stays_zero_copy(dtype):
    # [FUNCTION DOC]
    # GH#67990. Parametrized over string and binary arrow dtypes (string with NaN, string, binary, large string, large binary). Assigning values to a full slice of an all-null array keeps the character data buffer zero-copy: the last buffer address of the first chunk is the same as the source's (the aliasing fix must not undo GH#64529/64530).
    # GH#67990 the aliasing fix must not undo GH#64529/GH#64530: pa.array never
    #  packs character data into caller-owned memory, so it is not copied here
    is_binary = "binary" in str(dtype)
    values = pd.array(
        [b"a", b"bb", b"ccc"] if is_binary else ["a", "bb", "ccc"], dtype=dtype
    )
    arr = pd.array([None] * 3, dtype=dtype)
    arr[:] = values
    # the character data is the last buffer for every one of these layouts
    assert (
        arr._pa_array.chunks[0].buffers()[-1].address
        == values._pa_array.chunks[0].buffers()[-1].address
    )


def test_setitem_null_slice_no_alias_dictionary():
    # [FUNCTION DOC]
    # GH#67990. For a dictionary arrow type with 32-bit indices and 64-bit values, assigning a dictionary array over itself then mutating the caller-owned numpy values does not change the array (the dictionary child, which concatenation reuses, needs copying too).
    # GH#67990 pa.concat_arrays reuses the dictionary child, so the values half
    #  of a dictionary type needs copying too
    dtype = ArrowDtype(pa.dictionary(pa.int32(), pa.int64()))
    np_values = np.array([100, 200], dtype="int64")
    indices = pa.array(np.array([0, 1, 0], dtype="int32"))
    value = pa.DictionaryArray.from_arrays(indices, pa.array(np_values))

    arr = pd.array(value, dtype=dtype)
    arr[:] = value
    np_values[0] = -1
    expected = pd.array(
        pa.DictionaryArray.from_arrays(indices, pa.array([100, 200], type=pa.int64())),
        dtype=dtype,
    )
    tm.assert_extension_array_equal(arr, expected)


@pytest.mark.parametrize("kind", ["list", "struct"])
def test_setitem_null_slice_no_alias_nested(kind):
    # [FUNCTION DOC]
    # GH#67990. Parametrized over list and struct kinds. Assigning a nested arrow array over itself and mutating the caller-owned child numpy data leaves the list-of-values unchanged (concatenation copies the children of nested types, so only dictionaries need a special case).
    # GH#67990 pa.concat_arrays does copy a nested type's children, which is why
    #  only dictionary needs the special case above
    np_values = np.array([1, 2, 3, 4], dtype="int64")
    child = pa.array(np_values)
    if kind == "list":
        value = pa.ListArray.from_arrays(pa.array([0, 2, 4], type=pa.int32()), child)
        expected_data = [[1, 2], [3, 4]]
    else:
        value = pa.StructArray.from_arrays([child], names=["x"])
        expected_data = [{"x": 1}, {"x": 2}, {"x": 3}, {"x": 4}]
    dtype = ArrowDtype(value.type)

    arr = pd.array(value, dtype=dtype)
    arr[:] = value
    np_values[0] = -1
    assert arr.tolist() == expected_data


def test_setitem_null_slice_cow():
    # [FUNCTION DOC]
    # GH#67990. Assigning another frame's column to a full column slice of an arrow-backed frame by label does not tie the two frames together: mutating the source afterwards leaves the target as the original values.
    # GH#67990 full-slice assignment must not tie the two frames together
    df = pd.DataFrame({"a": pd.array([1, 2, 3], dtype="int64[pyarrow]")})
    other = pd.DataFrame({"b": [10, 20, 30]})
    df.loc[:, "a"] = other["b"]
    other.iloc[0, 0] = -777
    expected = pd.DataFrame({"a": pd.array([10, 20, 30], dtype="int64[pyarrow]")})
    tm.assert_frame_equal(df, expected)


def test_setitem_invalid_dtype(data):
    # [FUNCTION DOC]
    # GH50248. Assigning an invalid fill value to a full slice of the data raises: strings and binary with the integer 123 give a type error about invalid value for dtype; integer, floating and boolean with the string foo give an arrow invalid error saying it could not convert; other types with foo give a type error about invalid value.
    # GH50248
    pa_type = data._pa_array.type
    if pa.types.is_string(pa_type) or pa.types.is_binary(pa_type):
        fill_value = 123
        err = TypeError
        msg = "Invalid value '123' for dtype"
    elif (
        pa.types.is_integer(pa_type)
        or pa.types.is_floating(pa_type)
        or pa.types.is_boolean(pa_type)
    ):
        fill_value = "foo"
        err = pa.ArrowInvalid
        msg = "Could not convert"
    else:
        fill_value = "foo"
        err = TypeError
        msg = "Invalid value 'foo' for dtype"
    with pytest.raises(err, match=msg):
        data[:] = fill_value


def test_from_arrow_respecting_given_dtype():
    # [FUNCTION DOC]
    # The arrow-to-pandas conversion with a types mapper that maps a 32-bit date to a 64-bit date arrow dtype respects the given dtype.
    date_array = pa.array(
        [pd.Timestamp("2019-12-31"), pd.Timestamp("2019-12-31")], type=pa.date32()
    )
    result = date_array.to_pandas(
        types_mapper={pa.date32(): ArrowDtype(pa.date64())}.get
    )
    expected = pd.Series(
        [pd.Timestamp("2019-12-31"), pd.Timestamp("2019-12-31")],
        dtype=ArrowDtype(pa.date64()),
    )
    tm.assert_series_equal(result, expected)


def test_from_arrow_respecting_given_dtype_unsafe():
    # [FUNCTION DOC]
    # Converting arrow floats 1.5 and 2.5 with a mapper to a 64-bit integer arrow dtype raises an arrow invalid error (an external error from arrow, unsafe cast).
    array = pa.array([1.5, 2.5], type=pa.float64())
    with tm.external_error_raised(pa.ArrowInvalid):
        array.to_pandas(types_mapper={pa.float64(): ArrowDtype(pa.int64())}.get)


def test_round():
    # [FUNCTION DOC]
    # Rounding a 64-bit float arrow Series to one decimal place and to the tens place (negative digits) gives expected values with NA preserved.
    dtype = "float64[pyarrow]"

    ser = pd.Series([0.0, 1.23, 2.56, pd.NA], dtype=dtype)
    result = ser.round(1)
    expected = pd.Series([0.0, 1.2, 2.6, pd.NA], dtype=dtype)
    tm.assert_series_equal(result, expected)

    ser = pd.Series([123.4, pd.NA, 56.78], dtype=dtype)
    result = ser.round(-1)
    expected = pd.Series([120.0, pd.NA, 60.0], dtype=dtype)
    tm.assert_series_equal(result, expected)


def test_searchsorted_with_na_raises(data_for_sorting, as_series):
    # [FUNCTION DOC]
    # GH50447. Searching in an arrow array (or Series) with a trailing NA raises a value error saying searchsorted requires an array sorted, impossible with NAs present.
    # GH50447
    b, c, a = data_for_sorting
    arr = data_for_sorting.take([2, 0, 1])  # to get [a, b, c]
    arr[-1] = pd.NA

    if as_series:
        arr = pd.Series(arr)

    msg = (
        "searchsorted requires array to be sorted, "
        "which is impossible with NAs present."
    )
    with pytest.raises(ValueError, match=msg):
        arr.searchsorted(b)


def test_sort_values_dictionary():
    # [FUNCTION DOC]
    # Sorting a frame with a dictionary-typed arrow string column and an integer column by both leaves an already-sorted frame unchanged.
    df = pd.DataFrame(
        {
            "a": pd.Series(
                ["x", "y"], dtype=ArrowDtype(pa.dictionary(pa.int32(), pa.string()))
            ),
            "b": [1, 2],
        },
    )
    expected = df.copy()
    result = df.sort_values(by=["a", "b"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("pat", ["abc", "a[a-z]{2}"])
def test_str_count(pat):
    # [FUNCTION DOC]
    # Parametrized over patterns abc and a regular expression. Counting occurrences in an arrow string Series [abc, null] gives [1, null] with arrow 32-bit integer dtype.
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.count(pat)
    expected = pd.Series([1, None], dtype=ArrowDtype(pa.int32()))
    tm.assert_series_equal(result, expected)


def test_str_count_flags_unsupported():
    # [FUNCTION DOC]
    # Counting with flags set raises not-implemented with message that count is not supported.
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    with pytest.raises(NotImplementedError, match="count not"):
        ser.str.count("abc", flags=1)


@pytest.mark.parametrize(
    "side, str_func", [["left", "rjust"], ["right", "ljust"], ["both", "center"]]
)
def test_str_pad(side, str_func):
    # [FUNCTION DOC]
    # Parametrized over (side, matching string method): left/right-justify, right/left-justify, both/center. Padding an arrow string Series to width three with fill x gives results matching the corresponding Python string method on a, with null kept.
    ser = pd.Series(["a", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.pad(width=3, side=side, fillchar="x")
    expected = pd.Series(
        [getattr("a", str_func)(3, "x"), None], dtype=ArrowDtype(pa.string())
    )
    tm.assert_series_equal(result, expected)


def test_str_pad_invalid_side():
    # [FUNCTION DOC]
    # Padding with an invalid side raises a value error saying invalid side foo.
    ser = pd.Series(["a", None], dtype=ArrowDtype(pa.string()))
    with pytest.raises(ValueError, match="Invalid side: foo"):
        ser.str.pad(3, "foo", "x")


@pytest.mark.parametrize(
    "pat, case, na, regex, exp",
    [
        ["ab", False, None, False, [True, None]],
        ["Ab", True, None, False, [False, None]],
        ["ab", False, True, False, [True, True]],
        ["a[a-z]{1}", False, None, True, [True, None]],
        ["A[a-z]{1}", True, None, True, [False, None]],
    ],
)
def test_str_contains(pat, case, na, regex, exp):
    # [FUNCTION DOC]
    # Parametrized over (pattern, case sensitive, NA fill, regex, expected). Verifies contains on an arrow string Series [abc, null] gives arrow booleans as expected.
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.contains(pat, case=case, na=na, regex=regex)
    expected = pd.Series(exp, dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)


def test_str_contains_flags_unsupported():
    # [FUNCTION DOC]
    # Contains with flags set raises not-implemented with the message that contains is not supported.
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    with pytest.raises(NotImplementedError, match="contains not"):
        ser.str.contains("a", flags=1)


def test_str_contains_re2_unicode_escape():
    # [FUNCTION DOC]
    # GH 63901. Contains with a regular expression using a unicode range escape finds the Thai character: [false, true, null].
    # GH 63901
    ser = pd.Series(["a", "\u0e01", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.contains(r"[\x{0e00}-\x{0e7f}]")
    expected = pd.Series([False, True, None], dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "side, pat, na, exp",
    [
        ["startswith", "ab", None, [True, None, False]],
        ["startswith", "b", False, [False, False, False]],
        ["endswith", "b", True, [False, True, False]],
        ["endswith", "bc", None, [True, None, False]],
        ["startswith", ("a", "e", "g"), None, [True, None, True]],
        ["endswith", ("a", "c", "g"), None, [True, None, True]],
        ["startswith", (), None, [False, None, False]],
        ["endswith", (), None, [False, None, False]],
    ],
)
def test_str_start_ends_with(side, pat, na, exp):
    # [FUNCTION DOC]
    # Parametrized over (startswith/endswith, pattern, NA fill, expected). Verifies prefix/suffix tests on [abc, null, efg] including tuple patterns and the empty tuple (always false).
    ser = pd.Series(["abc", None, "efg"], dtype=ArrowDtype(pa.string()))
    result = getattr(ser.str, side)(pat, na=na)
    expected = pd.Series(exp, dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("side", ("startswith", "endswith"))
def test_str_starts_ends_with_all_nulls_empty_tuple(side):
    # [FUNCTION DOC]
    # Parametrized over startswith and endswith: with an empty tuple on an all-null arrow string Series, the result is all null but retains the arrow boolean dtype.
    ser = pd.Series([None, None], dtype=ArrowDtype(pa.string()))
    result = getattr(ser.str, side)(())

    # bool datatype preserved for all nulls.
    expected = pd.Series([None, None], dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "arg_name, arg",
    [["pat", re.compile("b")], ["repl", str], ["case", False], ["flags", 1]],
)
def test_str_replace_unsupported(arg_name, arg):
    # [FUNCTION DOC]
    # Parametrized over unsupported replace arguments (compiled pattern, callable replacement, case-insensitive flag, flags): replacing raises not-implemented with a message that replace is not supported.
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    kwargs = {"pat": "b", "repl": "x", "regex": True}
    kwargs[arg_name] = arg
    with pytest.raises(NotImplementedError, match="replace is not supported"):
        ser.str.replace(**kwargs)


@pytest.mark.parametrize(
    "pat, repl, n, regex, exp",
    [
        ["a", "x", -1, False, ["xbxc", None]],
        ["a", "x", 1, False, ["xbac", None]],
        ["[a-b]", "x", -1, True, ["xxxc", None]],
    ],
)
def test_str_replace(pat, repl, n, regex, exp):
    # [FUNCTION DOC]
    # Parametrized over (pattern, replacement, count, regex, expected). Verifies string replace on [abac, null] returns arrow strings as expected.
    ser = pd.Series(["abac", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.replace(pat, repl, n=n, regex=regex)
    expected = pd.Series(exp, dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)


def test_str_replace_re2_unicode_property():
    # [FUNCTION DOC]
    # Replacing with a regular expression using a unicode uppercase property class gives the expected result with null kept.
    ser = pd.Series(["Jan", "Feb", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.replace(r"\p{Lu}", "U", regex=True)
    expected = pd.Series(["Uan", "Ueb", None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)


def test_str_replace_negative_n():
    # [FUNCTION DOC]
    # GH 56404. Replacing with a negative count replaces every match for arrow strings, and (GH#59628) for pyarrow-backed string dtypes with NA and with NaN missing values.
    # GH 56404
    ser = pd.Series(["abc", "aaaaaa"], dtype=ArrowDtype(pa.string()))
    actual = ser.str.replace("a", "", -3, True)
    expected = pd.Series(["bc", ""], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(expected, actual)

    # Same bug for pyarrow-backed StringArray GH#59628
    ser2 = ser.astype(pd.StringDtype(storage="pyarrow"))
    actual2 = ser2.str.replace("a", "", -3, True)
    expected2 = expected.astype(ser2.dtype)
    tm.assert_series_equal(expected2, actual2)

    ser3 = ser.astype(pd.StringDtype(storage="pyarrow", na_value=np.nan))
    actual3 = ser3.str.replace("a", "", -3, True)
    expected3 = expected.astype(ser3.dtype)
    tm.assert_series_equal(expected3, actual3)


def test_str_replace_empty_pattern():
    # [FUNCTION DOC]
    # GH issue 64941. Replacing the empty pattern with an empty string leaves the string unchanged; replacing it with X inserts X before every character and at the end.
    # https://github.com/pandas-dev/pandas/issues/64941
    ser = pd.Series(["abcd"], dtype=ArrowDtype(pa.string()))

    result = ser.str.replace("", "")
    expected = pd.Series(["abcd"], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)

    result = ser.str.replace("", "X")
    expected = pd.Series(["XaXbXcXdX"], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)


def test_str_repeat_unsupported():
    # [FUNCTION DOC]
    # Repeating with a list of repeat counts raises not-implemented with a message that repeat is not supported.
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    with pytest.raises(NotImplementedError, match="repeat is not"):
        ser.str.repeat([1, 2])


def test_str_repeat():
    # [FUNCTION DOC]
    # Repeating each string of an arrow string Series twice, with null preserved.
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.repeat(2)
    expected = pd.Series(["abcabc", None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "pat, case, na, exp",
    [
        ["ab", False, None, [True, None]],
        ["Ab", True, None, [False, None]],
        ["bc", True, None, [False, None]],
        ["ab", False, True, [True, True]],
        ["a[a-z]{1}", False, None, [True, None]],
        ["A[a-z]{1}", True, None, [False, None]],
    ],
)
def test_str_match(pat, case, na, exp):
    # [FUNCTION DOC]
    # Parametrized over (pattern, case sensitive, NA fill, expected). Match anchors at the start of the string; verifies results on [abc, null] including regular expression patterns and the NA fill (arrow boolean dtype).
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.match(pat, case=case, na=na)
    expected = pd.Series(exp, dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "pat, case, na, exp",
    # Note: keep cases in sync with
    # pandas/tests/strings/test_find_replace.py::test_str_fullmatch_extra_cases
    [
        ["abc", False, None, [True, False, False, None]],
        ["Abc", True, None, [False, False, False, None]],
        ["bc", True, None, [False, False, False, None]],
        ["ab", False, None, [False, False, False, None]],
        ["a[a-z]{2}", False, None, [True, False, False, None]],
        ["A[a-z]{1}", True, None, [False, False, False, None]],
        # GH Issue: #56652
        ["abc$", False, None, [True, False, False, None]],
        ["abc\\$", False, None, [False, True, False, None]],
        ["Abc$", True, None, [False, False, False, None]],
        ["Abc\\$", True, None, [False, False, False, None]],
        # https://github.com/pandas-dev/pandas/issues/61072
        ["(abc)|(abx)", True, None, [True, False, False, None]],
        ["((abc)|(abx))", True, None, [True, False, False, None]],
    ],
)
def test_str_fullmatch(pat, case, na, exp):
    # [FUNCTION DOC]
    # Parametrized over (pattern, case sensitive, NA fill, expected) with cases kept in sync with the string-method tests of the same name (including dollar-anchored patterns, GH Issue #56652 and alternation patterns, issue 61072). Fullmatch on [abc, abc$, $abc, null] gives the expected arrow booleans.
    ser = pd.Series(["abc", "abc$", "$abc", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.fullmatch(pat, case=case, na=na)
    expected = pd.Series(exp, dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "sub, start, end, exp",
    [
        ["ab", 0, None, [0, None]],
        ["bc", 1, 3, [1, None]],
        ["ab", 1, 3, [-1, None]],
        ["ab", -3, -3, [-1, None]],
    ],
)
def test_str_find(sub, start, end, exp):
    # [FUNCTION DOC]
    # Parametrized over (substring, start, end, expected). Find on [abc, null] with the arrow 64-bit integer dtype and nulls preserved, including out-of-range negative bounds.
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.find(sub, start=start, end=end)
    expected = pd.Series(exp, dtype=ArrowDtype(pa.int64()))
    tm.assert_series_equal(result, expected)


def test_str_find_negative_start():
    # [FUNCTION DOC]
    # GH 56411. Find with a very negative start and end of three finds the substring at index one.
    # GH 56411
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.find(sub="b", start=-1000, end=3)
    expected = pd.Series([1, None], dtype=ArrowDtype(pa.int64()))
    tm.assert_series_equal(result, expected)


def test_str_find_no_end():
    # [FUNCTION DOC]
    # Find with a start but no end returns negative one when the substring is before start.
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.find("ab", start=1)
    expected = pd.Series([-1, None], dtype="int64[pyarrow]")
    tm.assert_series_equal(result, expected)


def test_str_find_negative_start_negative_end():
    # [FUNCTION DOC]
    # GH 56791. Find with negative start and end indexes counts from the end and finds the substring (index three).
    # GH 56791
    ser = pd.Series(["abcdefg", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.find(sub="d", start=-6, end=-3)
    expected = pd.Series([3, None], dtype=ArrowDtype(pa.int64()))
    tm.assert_series_equal(result, expected)


def test_str_find_large_start():
    # [FUNCTION DOC]
    # GH 56791. Find with a start beyond the string length returns negative one.
    # GH 56791
    ser = pd.Series(["abcdefg", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.find(sub="d", start=16)
    expected = pd.Series([-1, None], dtype=ArrowDtype(pa.int64()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("start", [-15, -3, 0, 1, 15, None])
@pytest.mark.parametrize("end", [-15, -1, 0, 3, 15, None])
@pytest.mark.parametrize("sub", ["", "az", "abce", "a", "caa"])
def test_str_find_e2e(start, end, sub):
    # [FUNCTION DOC]
    # Parametrized over start (six values), end (six values) and substring (five values). End-to-end comparison: find on an arrow string Series equals the python-storage string dtype result cast to the result's dtype, and the pyarrow-storage string dtype result agrees as well.
    s = pd.Series(
        ["abcaadef", "abc", "abcdeddefgj8292", "ab", "a", ""],
        dtype=ArrowDtype(pa.string()),
    )
    object_series = s.astype(pd.StringDtype(storage="python"))
    result = s.str.find(sub, start, end)
    expected = object_series.str.find(sub, start, end).astype(result.dtype)
    tm.assert_series_equal(result, expected)

    arrow_str_series = s.astype(pd.StringDtype(storage="pyarrow"))
    result2 = arrow_str_series.str.find(sub, start, end).astype(result.dtype)
    tm.assert_series_equal(result2, expected)


def test_str_find_negative_start_negative_end_no_match():
    # [FUNCTION DOC]
    # GH 56791. Find with a negative start after a negative end (empty range) returns negative one.
    # GH 56791
    ser = pd.Series(["abcdefg", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.find(sub="d", start=-3, end=-6)
    expected = pd.Series([-1, None], dtype=ArrowDtype(pa.int64()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "idx, expected_values",
    [
        [0, ["a", "d", None]],
        [1, ["b", "e", None]],
        [-1, ["c", "e", None]],
        [2, ["c", None, None]],
        [-3, ["a", None, None]],
        [4, [None, None, None]],
    ],
)
def test_str_get(idx, expected_values):
    # [FUNCTION DOC]
    # Parametrized over (index, expected). Get character at positions (including negative and out-of-range) for [abc, de, null] gives arrow strings with null for missing positions.
    ser = pd.Series(["abc", "de", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.get(idx)
    expected = pd.Series(expected_values, dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "idx, expected_values",
    [
        [0, ["a", "d", None]],
        [1, ["b", "e", None]],
        [-1, ["c", "e", None]],
        [2, ["c", None, None]],
        [-3, ["a", None, None]],
        [4, [None, None, None]],
    ],
)
def test_str_getitem(idx, expected_values):
    # [FUNCTION DOC]
    # GH 65112. Same cases as the get test but using item lookup on the string accessor with an integer index.
    # GH 65112
    ser = pd.Series(["abc", "de", None], dtype=ArrowDtype(pa.string()))
    result = ser.str[idx]
    expected = pd.Series(expected_values, dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)


@pytest.mark.xfail(
    reason="TODO: StringMethods._validate should support Arrow list types",
    raises=AttributeError,
)
def test_str_join():
    # [FUNCTION DOC]
    # Expected-to-fail with an attribute error (the string-method validation should support arrow list types, a TODO): joining list elements of an arrow list array with an equal sign.
    ser = pd.Series(ArrowExtensionArray(pa.array([list("abc"), list("123"), None])))
    result = ser.str.join("=")
    expected = pd.Series(["a=b=c", "1=2=3", None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)


def test_str_join_string_type():
    # [FUNCTION DOC]
    # Joining the characters of each arrow string (null preserved) with an equal sign.
    ser = pd.Series(ArrowExtensionArray(pa.array(["abc", "123", None])))
    result = ser.str.join("=")
    expected = pd.Series(["a=b=c", "1=2=3", None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "start, stop, step, expected_values",
    [
        [None, 2, None, ["ab", None]],
        [None, 2, 1, ["ab", None]],
        [1, 3, 1, ["bc", None]],
        (None, None, -1, ["dcba", None]),
        (1, None, 2, ["bd", None]),
        (None, None, None, ["abcd", None]),
    ],
)
def test_str_slice(start, stop, step, expected_values):
    # [FUNCTION DOC]
    # Parametrized over (start, stop, step, expected) including negative steps. Slicing each string of [abcd, null].
    ser = pd.Series(["abcd", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.slice(start, stop, step)
    expected = pd.Series(expected_values, dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "start, stop, step, expected_values",
    [
        [None, 2, None, ["ab", None]],
        [None, 2, 1, ["ab", None]],
        [1, 3, 1, ["bc", None]],
        (None, None, -1, ["dcba", None]),
        (1, None, 2, ["bd", None]),
        (None, None, None, ["abcd", None]),
    ],
)
def test_str_getitem_range(start, stop, step, expected_values):
    # [FUNCTION DOC]
    # GH 65112. Same cases as the slice test but using item lookup on the string accessor with a slice object.
    # GH 65112
    ser = pd.Series(["abcd", None], dtype=ArrowDtype(pa.string()))
    result = ser.str[slice(start, stop, step)]
    expected = pd.Series(expected_values, dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "start, stop, repl, exp",
    [
        [1, 2, "x", ["axcd", None]],
        [None, 2, "x", ["xcd", None]],
        [None, 2, None, ["cd", None]],
    ],
)
def test_str_slice_replace(start, stop, repl, exp):
    # [FUNCTION DOC]
    # Parametrized over (start, stop, replacement, expected). Replacing a slice of each string (a missing replacement deletes the slice).
    ser = pd.Series(["abcd", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.slice_replace(start, stop, repl)
    expected = pd.Series(exp, dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "value, method, exp",
    [
        ["a1c", "isalnum", True],
        ["!|,", "isalnum", False],
        ["aaa", "isalpha", True],
        ["!!!", "isalpha", False],
        ["٠", "isdecimal", True],  # noqa: RUF001
        ["~!", "isdecimal", False],
        ["2", "isdigit", True],
        ["~", "isdigit", False],
        ["aaa", "islower", True],
        ["aaA", "islower", False],
        ["123", "isnumeric", True],
        ["11I", "isnumeric", False],
        [" ", "isspace", True],
        ["", "isspace", False],
        ["The That", "istitle", True],
        ["the That", "istitle", False],
        ["AAA", "isupper", True],
        ["AAc", "isupper", False],
    ],
)
def test_str_is_functions(value, method, exp):
    # [FUNCTION DOC]
    # Parametrized over (value, predicate method, expected). The predicate methods (alphanumeric, alphabetic, decimal, digit, lower, numeric, space, title, upper) give the expected arrow boolean for the value, with null for missing.
    ser = pd.Series([value, None], dtype=ArrowDtype(pa.string()))
    result = getattr(ser.str, method)()
    expected = pd.Series([exp, None], dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "method, exp",
    [
        ["capitalize", "Abc def"],
        ["title", "Abc Def"],
        ["swapcase", "AbC Def"],
        ["lower", "abc def"],
        ["upper", "ABC DEF"],
        ["casefold", "abc def"],
    ],
)
def test_str_transform_functions(method, exp):
    # [FUNCTION DOC]
    # Parametrized over (case method, expected). Applying capitalize, title, swapcase, lower, upper, casefold to the mixed-case string gives the expected string with null preserved.
    ser = pd.Series(["aBc dEF", None], dtype=ArrowDtype(pa.string()))
    result = getattr(ser.str, method)()
    expected = pd.Series([exp, None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)


def test_str_len():
    # [FUNCTION DOC]
    # String lengths of [abcd, null] are [4, null] with the arrow 32-bit integer dtype.
    ser = pd.Series(["abcd", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.len()
    expected = pd.Series([4, None], dtype=ArrowDtype(pa.int32()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "method, to_strip, val",
    [
        ["strip", None, " abc "],
        ["strip", "x", "xabcx"],
        ["lstrip", None, " abc"],
        ["lstrip", "x", "xabc"],
        ["rstrip", None, "abc "],
        ["rstrip", "x", "abcx"],
    ],
)
def test_str_strip(method, to_strip, val):
    # [FUNCTION DOC]
    # Parametrized over (strip method, characters to strip, input). Strip, left-strip and right-strip of whitespace or x yield abc with null preserved.
    ser = pd.Series([val, None], dtype=ArrowDtype(pa.string()))
    result = getattr(ser.str, method)(to_strip=to_strip)
    expected = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("val", ["abc123", "abc"])
def test_str_removesuffix(val):
    # [FUNCTION DOC]
    # Parametrized over strings with and without the 123 suffix. Removing that suffix yields abc with null preserved.
    ser = pd.Series([val, None], dtype=ArrowDtype(pa.string()))
    result = ser.str.removesuffix("123")
    expected = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("val", ["123abc", "abc"])
def test_str_removeprefix(val):
    # [FUNCTION DOC]
    # Parametrized over strings with and without the 123 prefix. Removing that prefix yields abc with null preserved.
    ser = pd.Series([val, None], dtype=ArrowDtype(pa.string()))
    result = ser.str.removeprefix("123")
    expected = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("errors", ["ignore", "strict"])
@pytest.mark.parametrize(
    "encoding, exp",
    [
        ("utf8", {"little": b"abc", "big": "abc"}),
        (
            "utf32",
            {
                "little": b"\xff\xfe\x00\x00a\x00\x00\x00b\x00\x00\x00c\x00\x00\x00",
                "big": b"\x00\x00\xfe\xff\x00\x00\x00a\x00\x00\x00b\x00\x00\x00c",
            },
        ),
    ],
    ids=["utf8", "utf32"],
)
def test_str_encode(errors, encoding, exp):
    # [FUNCTION DOC]
    # Parametrized over errors mode (ignore, strict) and encoding (utf8, utf32). Encoding [abc, null] gives arrow binary values; the expected bytes depend on the platform byte order.
    # KNOWN ISSUE: for the little-endian utf8 case the expected bytes are bytes while the big-endian entry for utf8 is a plain string rather than bytes, so the big-endian expectation would not equal the binary result.
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.encode(encoding, errors)
    expected = pd.Series([exp[sys.byteorder], None], dtype=ArrowDtype(pa.binary()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("flags", [0, 2])
def test_str_findall(flags):
    # [FUNCTION DOC]
    # Parametrized over regular expression flags 0 and 2. Findall of b gives lists of matches as arrow list of strings, [[b], [], null].
    ser = pd.Series(["abc", "efg", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.findall("b", flags=flags)
    expected = pd.Series([["b"], [], None], dtype=ArrowDtype(pa.list_(pa.string())))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("method", ["index", "rindex"])
@pytest.mark.parametrize(
    "start, end",
    [
        [0, None],
        [1, 4],
    ],
)
def test_str_r_index(method, start, end):
    # [FUNCTION DOC]
    # Parametrized over index or rindex and (start, end). The index of c is two (null preserved, 64-bit integer); searching for a missing substring raises a value error saying substring not found.
    ser = pd.Series(["abcba", None], dtype=ArrowDtype(pa.string()))
    result = getattr(ser.str, method)("c", start, end)
    expected = pd.Series([2, None], dtype=ArrowDtype(pa.int64()))
    tm.assert_series_equal(result, expected)

    with pytest.raises(ValueError, match="substring not found"):
        getattr(ser.str, method)("foo", start, end)


@pytest.mark.parametrize("form", ["NFC", "NFKC"])
def test_str_normalize(form):
    # [FUNCTION DOC]
    # Parametrized over normalization forms NFC and NFKC. Normalizing an ASCII string leaves the Series unchanged.
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.normalize(form)
    expected = ser.copy()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "start, end",
    [
        [0, None],
        [1, 4],
    ],
)
def test_str_rfind(start, end):
    # [FUNCTION DOC]
    # Parametrized over (start, end). Right-find of c in [abcba, foo, null] gives [2, -1, null] with 64-bit integer dtype.
    ser = pd.Series(["abcba", "foo", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.rfind("c", start, end)
    expected = pd.Series([2, -1, None], dtype=ArrowDtype(pa.int64()))
    tm.assert_series_equal(result, expected)


def test_str_translate():
    # [FUNCTION DOC]
    # Translating by a mapping of character code ninety-seven to b turns a to b in the arrow strings, null preserved.
    ser = pd.Series(["abcba", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.translate({97: "b"})
    expected = pd.Series(["bbcbb", None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)


def test_str_wrap():
    # [FUNCTION DOC]
    # Wrapping at width three splits abcba into abc and ba separated by a newline; null preserved.
    ser = pd.Series(["abcba", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.wrap(3)
    expected = pd.Series(["abc\nba", None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)


def test_get_dummies():
    # [FUNCTION DOC]
    # Expanding the pipe-separated strings [a|b, null, a|c] into indicator columns gives a boolean arrow frame with columns a, b, c (null row all false).
    ser = pd.Series(["a|b", None, "a|c"], dtype=ArrowDtype(pa.string()))
    result = ser.str.get_dummies()
    expected = pd.DataFrame(
        [[True, True, False], [False, False, False], [True, False, True]],
        dtype=ArrowDtype(pa.bool_()),
        columns=["a", "b", "c"],
    )
    tm.assert_frame_equal(result, expected)


def test_str_partition():
    # [FUNCTION DOC]
    # Partition and right-partition of [abcba, null] on b: expanded output is a frame of three arrow-string columns (before, separator, after; all null for the null row; default range column index); with expand disabled the output is a Series of arrow lists of three strings. Right-partition splits at the last separator.
    ser = pd.Series(["abcba", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.partition("b")
    expected = pd.DataFrame(
        [["a", "b", "cba"], [None, None, None]],
        dtype=ArrowDtype(pa.string()),
        columns=pd.RangeIndex(3),
    )
    tm.assert_frame_equal(result, expected, check_column_type=True)

    result = ser.str.partition("b", expand=False)
    expected = pd.Series(ArrowExtensionArray(pa.array([["a", "b", "cba"], None])))
    tm.assert_series_equal(result, expected)

    result = ser.str.rpartition("b")
    expected = pd.DataFrame(
        [["abc", "b", "a"], [None, None, None]],
        dtype=ArrowDtype(pa.string()),
        columns=pd.RangeIndex(3),
    )
    tm.assert_frame_equal(result, expected, check_column_type=True)

    result = ser.str.rpartition("b", expand=False)
    expected = pd.Series(ArrowExtensionArray(pa.array([["abc", "b", "a"], None])))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("method", ["rsplit", "split"])
def test_str_split_pat_none(method):
    # [FUNCTION DOC]
    # GH 56271. Parametrized over split and right-split. With no pattern, splitting on whitespace (including newline) gives a list of fields; null stays null (arrow list Series).
    # GH 56271
    ser = pd.Series(["a1 cbc\nb", None], dtype=ArrowDtype(pa.string()))
    result = getattr(ser.str, method)()
    expected = pd.Series(ArrowExtensionArray(pa.array([["a1", "cbc", "b"], None])))
    tm.assert_series_equal(result, expected)


def test_str_split():
    # [FUNCTION DOC]
    # GH 52401. Split of [a1cbcb, a2cbcb, null] on c with no limit; with a limit of one; on a regular expression of digits (list result, and expanded into two columns of arrow strings); on a literal one with expansion (shorter rows padded with null).
    # GH 52401
    ser = pd.Series(["a1cbcb", "a2cbcb", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.split("c")
    expected = pd.Series(
        ArrowExtensionArray(pa.array([["a1", "b", "b"], ["a2", "b", "b"], None]))
    )
    tm.assert_series_equal(result, expected)

    result = ser.str.split("c", n=1)
    expected = pd.Series(
        ArrowExtensionArray(pa.array([["a1", "bcb"], ["a2", "bcb"], None]))
    )
    tm.assert_series_equal(result, expected)

    result = ser.str.split("[1-2]", regex=True)
    expected = pd.Series(
        ArrowExtensionArray(pa.array([["a", "cbcb"], ["a", "cbcb"], None]))
    )
    tm.assert_series_equal(result, expected)

    result = ser.str.split("[1-2]", regex=True, expand=True)
    expected = pd.DataFrame(
        {
            0: ArrowExtensionArray(pa.array(["a", "a", None])),
            1: ArrowExtensionArray(pa.array(["cbcb", "cbcb", None])),
        }
    )
    tm.assert_frame_equal(result, expected)

    result = ser.str.split("1", expand=True)
    expected = pd.DataFrame(
        {
            0: ArrowExtensionArray(pa.array(["a", "a2cbcb", None])),
            1: ArrowExtensionArray(pa.array(["cbcb", None, None])),
        }
    )
    tm.assert_frame_equal(result, expected)


def test_str_rsplit():
    # [FUNCTION DOC]
    # GH 52401. Right-split on c with no limit, with a limit of one (splitting at the last separator), expanded with a limit, and on a literal one with expansion.
    # GH 52401
    ser = pd.Series(["a1cbcb", "a2cbcb", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.rsplit("c")
    expected = pd.Series(
        ArrowExtensionArray(pa.array([["a1", "b", "b"], ["a2", "b", "b"], None]))
    )
    tm.assert_series_equal(result, expected)

    result = ser.str.rsplit("c", n=1)
    expected = pd.Series(
        ArrowExtensionArray(pa.array([["a1cb", "b"], ["a2cb", "b"], None]))
    )
    tm.assert_series_equal(result, expected)

    result = ser.str.rsplit("c", n=1, expand=True)
    expected = pd.DataFrame(
        {
            0: ArrowExtensionArray(pa.array(["a1cb", "a2cb", None])),
            1: ArrowExtensionArray(pa.array(["b", "b", None])),
        }
    )
    tm.assert_frame_equal(result, expected)

    result = ser.str.rsplit("1", expand=True)
    expected = pd.DataFrame(
        {
            0: ArrowExtensionArray(pa.array(["a", "a2cbcb", None])),
            1: ArrowExtensionArray(pa.array(["cbcb", None, None])),
        }
    )
    tm.assert_frame_equal(result, expected)


def test_str_extract_non_symbolic():
    # [FUNCTION DOC]
    # Extracting with a pattern lacking a named group raises a value error saying the pattern must contain a symbolic group name.
    ser = pd.Series(["a1", "b2", "c3"], dtype=ArrowDtype(pa.string()))
    with pytest.raises(ValueError, match="pat=.* must contain a symbolic group name."):
        ser.str.extract(r"[ab](\d)")


@pytest.mark.parametrize("expand", [True, False])
def test_str_extract(expand):
    # [FUNCTION DOC]
    # Parametrized over expand true/false. Extracting two named groups from [a1, b2, c3] gives a frame with columns letter and digit as arrow strings, null where not matching.
    ser = pd.Series(["a1", "b2", "c3"], dtype=ArrowDtype(pa.string()))
    result = ser.str.extract(r"(?P<letter>[ab])(?P<digit>\d)", expand=expand)
    expected = pd.DataFrame(
        {
            "letter": ArrowExtensionArray(pa.array(["a", "b", None])),
            "digit": ArrowExtensionArray(pa.array(["1", "2", None])),
        }
    )
    tm.assert_frame_equal(result, expected)


def test_str_extract_expand():
    # [FUNCTION DOC]
    # Extracting one named group: with expansion a one-column frame named digit; without expansion a Series named digit (both arrow strings with null for non-matches).
    ser = pd.Series(["a1", "b2", "c3"], dtype=ArrowDtype(pa.string()))
    result = ser.str.extract(r"[ab](?P<digit>\d)", expand=True)
    expected = pd.DataFrame(
        {
            "digit": ArrowExtensionArray(pa.array(["1", "2", None])),
        }
    )
    tm.assert_frame_equal(result, expected)

    result = ser.str.extract(r"[ab](?P<digit>\d)", expand=False)
    expected = pd.Series(ArrowExtensionArray(pa.array(["1", "2", None])), name="digit")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
def test_duration_from_strings_with_nat(unit):
    # [FUNCTION DOC]
    # GH51175. Parametrized over duration units ns, us, ms, s. Building from the strings 1000 and NaT gives an arrow duration array of value 1000 and null.
    # GH51175
    strings = ["1000", "NaT"]
    pa_type = pa.duration(unit)
    dtype = ArrowDtype(pa_type)
    result = ArrowExtensionArray._from_sequence_of_strings(strings, dtype=dtype)
    expected = ArrowExtensionArray(pa.array([1000, None], type=pa_type))
    tm.assert_extension_array_equal(result, expected)


def test_unsupported_dt(data):
    # [FUNCTION DOC]
    # For non-temporal arrow types, accessing the datetime accessor on a Series raises an attribute error saying it can only be used with datetimelike values; temporal types are not checked.
    pa_dtype = data.dtype.pyarrow_dtype
    if not pa.types.is_temporal(pa_dtype):
        with pytest.raises(
            AttributeError, match="Can only use .dt accessor with datetimelike values"
        ):
            pd.Series(data).dt


@pytest.mark.parametrize(
    "prop, expected",
    [
        ["year", 2023],
        ["day", 2],
        ["day_of_week", 0],
        ["dayofweek", 0],
        ["weekday", 0],
        ["day_of_year", 2],
        ["dayofyear", 2],
        ["hour", 3],
        ["minute", 4],
        ["is_leap_year", False],
        ["microsecond", 2000],
        ["month", 1],
        ["nanosecond", 6],
        ["quarter", 1],
        ["second", 7],
        ["date", date(2023, 1, 2)],
        ["time", time(3, 4, 7, 2000)],
    ],
)
def test_dt_properties(prop, expected):
    # [FUNCTION DOC]
    # Parametrized over (datetime property, expected): year, day, day of week (three aliases), day of year (two aliases), hour, minute, leap year flag, microsecond, month, nanosecond, quarter, second, date and time. A Series of a nanosecond timestamp (2023-01-02 03:04:07.002000006) and a null; the property result equals an arrow Series of the expected and null (date32 type for date, 64-bit nanosecond time for time).
    ser = pd.Series(
        [
            pd.Timestamp(
                year=2023,
                month=1,
                day=2,
                hour=3,
                minute=4,
                second=7,
                microsecond=2000,
                nanosecond=6,
            ),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    result = getattr(ser.dt, prop)
    exp_type = None
    if isinstance(expected, date):
        exp_type = pa.date32()
    elif isinstance(expected, time):
        exp_type = pa.time64("ns")
    expected = pd.Series(ArrowExtensionArray(pa.array([expected, None], type=exp_type)))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("microsecond", [2000, 5, 0])
def test_dt_microsecond(microsecond):
    # [FUNCTION DOC]
    # GH 59183. Parametrized over microsecond values 2000, 5, 0: the microsecond property of a nanosecond arrow timestamp returns it as arrow 64-bit integer with null kept.
    # GH 59183
    ser = pd.Series(
        [
            pd.Timestamp(
                year=2024,
                month=7,
                day=7,
                second=5,
                microsecond=microsecond,
                nanosecond=6,
            ),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    result = ser.dt.microsecond
    expected = pd.Series([microsecond, None], dtype="int64[pyarrow]")
    tm.assert_series_equal(result, expected)


def test_dt_is_month_start_end():
    # [FUNCTION DOC]
    # The month-start and month-end flags of three microsecond timestamps and a null are arrow booleans [false, true, false, null] and [false, false, true, null].
    ser = pd.Series(
        [
            datetime(year=2023, month=12, day=2, hour=3),
            datetime(year=2023, month=1, day=1, hour=3),
            datetime(year=2023, month=3, day=31, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("us")),
    )
    result = ser.dt.is_month_start
    expected = pd.Series([False, True, False, None], dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)

    result = ser.dt.is_month_end
    expected = pd.Series([False, False, True, None], dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)


def test_dt_is_year_start_end():
    # [FUNCTION DOC]
    # The year-start and year-end flags of three timestamps and a null: [false, true, false, null] and [true, false, false, null].
    ser = pd.Series(
        [
            datetime(year=2023, month=12, day=31, hour=3),
            datetime(year=2023, month=1, day=1, hour=3),
            datetime(year=2023, month=3, day=31, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("us")),
    )
    result = ser.dt.is_year_start
    expected = pd.Series([False, True, False, None], dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)

    result = ser.dt.is_year_end
    expected = pd.Series([True, False, False, None], dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)


def test_dt_is_quarter_start_end():
    # [FUNCTION DOC]
    # The quarter-start and quarter-end flags of three timestamps and a null: [false, true, false, null] and [false, false, true, null].
    ser = pd.Series(
        [
            datetime(year=2023, month=11, day=30, hour=3),
            datetime(year=2023, month=1, day=1, hour=3),
            datetime(year=2023, month=3, day=31, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("us")),
    )
    result = ser.dt.is_quarter_start
    expected = pd.Series([False, True, False, None], dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)

    result = ser.dt.is_quarter_end
    expected = pd.Series([False, False, True, None], dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("method", ["days_in_month", "daysinmonth"])
def test_dt_days_in_month(method):
    # [FUNCTION DOC]
    # Parametrized over the two property names days_in_month and daysinmonth: values 31, 30, 28 and null as arrow 64-bit integers.
    ser = pd.Series(
        [
            datetime(year=2023, month=3, day=30, hour=3),
            datetime(year=2023, month=4, day=1, hour=3),
            datetime(year=2023, month=2, day=3, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("us")),
    )
    result = getattr(ser.dt, method)
    expected = pd.Series([31, 30, 28, None], dtype=ArrowDtype(pa.int64()))
    tm.assert_series_equal(result, expected)


def test_dt_normalize():
    # [FUNCTION DOC]
    # Normalizing timestamps to midnight keeps the microsecond arrow timestamp dtype and null.
    ser = pd.Series(
        [
            datetime(year=2023, month=3, day=30),
            datetime(year=2023, month=4, day=1, hour=3),
            datetime(year=2023, month=2, day=3, hour=23, minute=59, second=59),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("us")),
    )
    result = ser.dt.normalize()
    expected = pd.Series(
        [
            datetime(year=2023, month=3, day=30),
            datetime(year=2023, month=4, day=1),
            datetime(year=2023, month=2, day=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("us")),
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("unit", ["us", "ns"])
def test_dt_time_preserve_unit(unit):
    # [FUNCTION DOC]
    # Parametrized over timestamp units us and ns. The accessor's unit equals the dtype unit, and the time-of-day property preserves the unit as a 64-bit time with the same unit.
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp(unit)),
    )
    assert ser.dt.unit == unit

    result = ser.dt.time
    expected = pd.Series(
        ArrowExtensionArray(pa.array([time(3, 0), None], type=pa.time64(unit)))
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("tz", [None, "UTC", "US/Pacific"])
def test_dt_tz(tz):
    # [FUNCTION DOC]
    # Parametrized over no time zone, UTC and US/Pacific. The time zone property of the accessor equals the time zone object resolved from the name.
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp("ns", tz=tz)),
    )
    result = ser.dt.tz
    assert result == timezones.maybe_get_tz(tz)


def test_dt_isocalendar():
    # [FUNCTION DOC]
    # The ISO calendar of a timestamp (2023-01-02) and null gives a frame with columns year, week, day of arrow 64-bit integers, with the null row as zeros.
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    result = ser.dt.isocalendar()
    expected = pd.DataFrame(
        [[2023, 1, 1], [0, 0, 0]],
        columns=["year", "week", "day"],
        dtype="int64[pyarrow]",
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "method, exp", [["day_name", "Sunday"], ["month_name", "January"]]
)
def test_dt_day_month_name(method, exp, request):
    # [FUNCTION DOC]
    # GH 52388. Parametrized over day name (Sunday) and month name (January). After applying the time zone database expected-failure helper, naming a millisecond timestamp 2023-01-01 gives an arrow string, null kept.
    # GH 52388
    _require_timezone_database(request)

    ser = pd.Series([datetime(2023, 1, 1), None], dtype=ArrowDtype(pa.timestamp("ms")))
    result = getattr(ser.dt, method)()
    expected = pd.Series([exp, None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)


def test_dt_strftime(request):
    # [FUNCTION DOC]
    # After applying the time zone database expected-failure helper, strftime with an ISO format on a nanosecond timestamp gives the string with nine fractional digits and null for null.
    _require_timezone_database(request)

    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    result = ser.dt.strftime("%Y-%m-%dT%H:%M:%S")
    expected = pd.Series(
        ["2023-01-02T03:00:00.000000000", None], dtype=ArrowDtype(pa.string())
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("method", ["ceil", "floor", "round"])
def test_dt_roundlike_tz_options_not_supported(method):
    # [FUNCTION DOC]
    # Parametrized over ceil, floor, round. Passing the ambiguous or nonexistent options raises not-implemented with messages saying each option is not supported.
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    with pytest.raises(NotImplementedError, match="ambiguous is not supported."):
        getattr(ser.dt, method)("1h", ambiguous="NaT")

    with pytest.raises(NotImplementedError, match="nonexistent is not supported."):
        getattr(ser.dt, method)("1h", nonexistent="NaT")


@pytest.mark.parametrize("method", ["ceil", "floor", "round"])
def test_dt_roundlike_unsupported_freq(method):
    # [FUNCTION DOC]
    # Parametrized over ceil, floor, round. An unsupported business-day frequency raises a value error saying that frequency is not supported; a frequency of none raises a value error saying a valid frequency must be specified.
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    with pytest.raises(ValueError, match="freq='1B' is not supported"):
        getattr(ser.dt, method)("1B")

    with pytest.raises(ValueError, match="Must specify a valid frequency: None"):
        getattr(ser.dt, method)(None)


@pytest.mark.parametrize("freq", ["D", "h", "min", "s", "ms", "us", "ns"])
@pytest.mark.parametrize("method", ["ceil", "floor", "round"])
def test_dt_ceil_year_floor(freq, method):
    # [FUNCTION DOC]
    # Parametrized over frequency (day, hour, minute, second, ms, us, ns) and ceil/floor/round. Rounding a nanosecond arrow timestamp Series at the frequency equals the numpy-backed result cast to the arrow dtype.
    ser = pd.Series(
        [datetime(year=2023, month=1, day=1), None],
    )
    pa_dtype = ArrowDtype(pa.timestamp("ns"))
    expected = getattr(ser.dt, method)(f"1{freq}").astype(pa_dtype)
    result = getattr(ser.astype(pa_dtype).dt, method)(f"1{freq}")
    tm.assert_series_equal(result, expected)


def test_dt_to_pydatetime():
    # [FUNCTION DOC]
    # GH 51859. Converting an arrow timestamp Series to python datetimes gives an object Series of datetime objects, equal to the numpy-backed result.
    # GH 51859
    data = [datetime(2022, 1, 1), datetime(2023, 1, 1)]
    ser = pd.Series(data, dtype=ArrowDtype(pa.timestamp("ns")))
    result = ser.dt.to_pydatetime()
    expected = pd.Series(data, dtype=object)
    tm.assert_series_equal(result, expected)
    assert all(type(expected.iloc[i]) is datetime for i in range(len(expected)))

    expected = ser.astype("datetime64[ns]").dt.to_pydatetime()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("date_type", [32, 64])
def test_dt_to_pydatetime_date_error(date_type):
    # [FUNCTION DOC]
    # GH 52812. Parametrized over 32-bit and 64-bit date types. Converting dates to python datetimes raises a value error saying the conversion cannot be called with the date type.
    # GH 52812
    ser = pd.Series(
        [date(2022, 12, 31)],
        dtype=ArrowDtype(getattr(pa, f"date{date_type}")()),
    )
    with pytest.raises(ValueError, match="to_pydatetime cannot be called with"):
        ser.dt.to_pydatetime()


def test_dt_tz_localize_unsupported_tz_options():
    # [FUNCTION DOC]
    # Localizing time zone with ambiguous or nonexistent set to NaT raises not-implemented with messages that those options are not supported.
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    with pytest.raises(NotImplementedError, match="ambiguous='NaT' is not supported"):
        ser.dt.tz_localize("UTC", ambiguous="NaT")

    with pytest.raises(NotImplementedError, match="nonexistent='NaT' is not supported"):
        ser.dt.tz_localize("UTC", nonexistent="NaT")


def test_dt_tz_localize_none(request):
    # [FUNCTION DOC]
    # After applying the time zone database expected-failure helper: removing the time zone from a US/Pacific nanosecond arrow Series gives the naive timestamps (equal to the scalar's naive version) with null kept.
    _require_timezone_database(request)

    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp("ns", tz="US/Pacific")),
    )
    result = ser.dt.tz_localize(None)
    expected = pd.Series(
        [ser[0].tz_localize(None), None],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("unit", ["us", "ns"])
def test_dt_tz_localize(unit, request):
    # [FUNCTION DOC]
    # Parametrized over timestamp units us and ns. After the time zone helper, localizing a naive Series to US/Pacific equals assuming the time zone with the arrow compute function on the same array.
    _require_timezone_database(request)

    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp(unit)),
    )
    result = ser.dt.tz_localize("US/Pacific")
    exp_data = pa.array(
        [datetime(year=2023, month=1, day=2, hour=3), None], type=pa.timestamp(unit)
    )
    exp_data = pa.compute.assume_timezone(exp_data, "US/Pacific")
    expected = pd.Series(ArrowExtensionArray(exp_data))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "nonexistent, exp_date",
    [
        ["shift_forward", datetime(year=2023, month=3, day=12, hour=3)],
        ["shift_backward", pd.Timestamp("2023-03-12 01:59:59.999999999")],
    ],
)
def test_dt_tz_localize_nonexistent(nonexistent, exp_date, request):
    # [FUNCTION DOC]
    # Parametrized over shift_forward and shift_backward with expected timestamps. After the time zone helper, localizing a nonexistent daylight-saving wall time (2023-03-12 02:30) to US/Pacific shifts it forward to 03:00 or backward to 01:59:59.999999999, null kept.
    _require_timezone_database(request)

    ser = pd.Series(
        [datetime(year=2023, month=3, day=12, hour=2, minute=30), None],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    result = ser.dt.tz_localize("US/Pacific", nonexistent=nonexistent)
    exp_data = pa.array([exp_date, None], type=pa.timestamp("ns"))
    exp_data = pa.compute.assume_timezone(exp_data, "US/Pacific")
    expected = pd.Series(ArrowExtensionArray(exp_data))
    tm.assert_series_equal(result, expected)


def test_dt_tz_convert_not_tz_raises():
    # [FUNCTION DOC]
    # Converting a naive timestamp Series to a time zone raises a type error saying it cannot convert tz-naive timestamps.
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    with pytest.raises(TypeError, match="Cannot convert tz-naive timestamps"):
        ser.dt.tz_convert("UTC")


def test_dt_tz_convert_none():
    # [FUNCTION DOC]
    # Converting a US/Pacific Series to no time zone gives naive timestamps in UTC, equal to the scalar's conversion, null kept.
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp("ns", "US/Pacific")),
    )
    result = ser.dt.tz_convert(None)
    expected = pd.Series(
        [ser[0].tz_convert(None), None],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("unit", ["us", "ns"])
def test_dt_tz_convert(unit):
    # [FUNCTION DOC]
    # Parametrized over units us and ns. Converting a US/Pacific Series to US/Eastern gives the scalar's conversion in that time zone, null kept.
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp(unit, "US/Pacific")),
    )
    result = ser.dt.tz_convert("US/Eastern")
    expected = pd.Series(
        [ser[0].tz_convert("US/Eastern"), None],
        dtype=ArrowDtype(pa.timestamp(unit, "US/Eastern")),
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("dtype", ["timestamp[ms][pyarrow]", "duration[ms][pyarrow]"])
def test_as_unit(dtype):
    # [FUNCTION DOC]
    # GH 52284. Parametrized over millisecond timestamp and duration. Changing the unit to nanoseconds equals casting to the nanosecond dtype.
    # GH 52284
    ser = pd.Series([1000, None], dtype=dtype)
    result = ser.dt.as_unit("ns")
    expected = ser.astype(dtype.replace("ms", "ns"))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "from_unit,to_unit",
    [
        ("ns", "us"),
        ("ns", "ms"),
        ("ns", "s"),
        ("us", "ms"),
        ("us", "s"),
        ("ms", "s"),
        ("s", "ms"),
        ("s", "us"),
        ("s", "ns"),
        ("ms", "us"),
        ("ms", "ns"),
        ("us", "ns"),
    ],
)
def test_as_unit_duration_truncation(from_unit, to_unit):
    # [FUNCTION DOC]
    # Parametrized over twelve (from-unit, to-unit) pairs. Changing the unit of an arrow duration Series equals the numpy-backed result (truncation toward the coarser unit).
    # Test that as_unit truncates correctly (matches NumPy behavior)
    # Value with sub-unit precision to test truncation
    ser_numpy = pd.Series(
        pd.to_timedelta([93784567890123, None], unit="ns").as_unit(from_unit)
    )
    ser_arrow = ser_numpy.astype(f"duration[{from_unit}][pyarrow]")

    result = ser_arrow.dt.as_unit(to_unit)
    expected = ser_numpy.dt.as_unit(to_unit).astype(f"duration[{to_unit}][pyarrow]")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "from_unit,to_unit",
    [
        ("ns", "us"),
        ("ns", "ms"),
        ("ns", "s"),
        ("s", "ns"),
        ("ms", "ns"),
        ("us", "ns"),
    ],
)
def test_as_unit_timestamp(from_unit, to_unit):
    # [FUNCTION DOC]
    # Parametrized over six unit pairs. Changing the unit of an arrow timestamp Series gives values equal to the numpy-backed result (comparing non-null values), and null counts are preserved.
    # Test timestamp as_unit matches NumPy behavior
    # Create Arrow series directly to preserve nulls correctly
    ser_arrow = pd.Series(
        [pd.Timestamp("2024-01-15 12:30:45.123456789"), None],
        dtype=f"timestamp[{from_unit}][pyarrow]",
    )
    ser_numpy = ser_arrow.astype(f"datetime64[{from_unit}]")

    result = ser_arrow.dt.as_unit(to_unit)
    expected_numpy = ser_numpy.dt.as_unit(to_unit)
    # Compare values (excluding null handling differences)
    tm.assert_almost_equal(
        result.dropna().to_numpy(dtype=f"datetime64[{to_unit}]"),
        expected_numpy.dropna().to_numpy(),
    )
    # Verify nulls are preserved
    assert result.isna().sum() == ser_arrow.isna().sum()


@pytest.mark.parametrize("to_unit", ["s", "ms", "us", "ns"])
def test_as_unit_timestamp_with_timezone(to_unit):
    # [FUNCTION DOC]
    # Parametrized over target units s, ms, us, ns. Changing the unit of a time-zone-aware (US/Eastern) arrow timestamp keeps the time zone and equals the numpy-backed result cast to the arrow dtype; the dtype string reflects the new unit and zone.
    # Test that timezone is preserved
    ser_numpy = pd.Series(
        pd.to_datetime(["2024-01-15 12:30:45.123456789"])
        .tz_localize("US/Eastern")
        .as_unit("ns")
    )
    ser_arrow = ser_numpy.astype("timestamp[ns, US/Eastern][pyarrow]")

    result = ser_arrow.dt.as_unit(to_unit)
    expected = ser_numpy.dt.as_unit(to_unit).astype(
        f"timestamp[{to_unit}, US/Eastern][pyarrow]"
    )
    tm.assert_series_equal(result, expected)
    assert str(result.dtype) == f"timestamp[{to_unit}, tz=US/Eastern][pyarrow]"


def test_as_unit_date_raises():
    # [FUNCTION DOC]
    # Changing the unit of date-typed data raises not-implemented with the message that as_unit is not implemented.
    # as_unit should raise for date types
    ser = pd.Series([1, 2], dtype=ArrowDtype(pa.date32()))
    with pytest.raises(NotImplementedError, match="as_unit not implemented"):
        ser.dt.as_unit("ns")


@pytest.mark.parametrize(
    "prop, expected",
    [
        ["days", 1],
        ["seconds", 2],
        ["microseconds", 3],
        ["nanoseconds", 4],
    ],
)
def test_dt_timedelta_properties(prop, expected):
    # [FUNCTION DOC]
    # GH 52284. Parametrized over (days, seconds, microseconds, nanoseconds) of a duration of one day, two seconds, three microseconds, four nanoseconds: the property returns the expected component as an arrow 32-bit integer Series, null kept.
    # GH 52284
    ser = pd.Series(
        [
            pd.Timedelta(
                days=1,
                seconds=2,
                microseconds=3,
                nanoseconds=4,
            ),
            None,
        ],
        dtype=ArrowDtype(pa.duration("ns")),
    )
    result = getattr(ser.dt, prop)
    expected = pd.Series(
        ArrowExtensionArray(pa.array([expected, None], type=pa.int32()))
    )
    tm.assert_series_equal(result, expected)


def test_dt_timedelta_total_seconds():
    # [FUNCTION DOC]
    # GH 52284. Total seconds of the same duration is 86402.000003 as an arrow 64-bit float, null kept.
    # GH 52284
    ser = pd.Series(
        [
            pd.Timedelta(
                days=1,
                seconds=2,
                microseconds=3,
                nanoseconds=4,
            ),
            None,
        ],
        dtype=ArrowDtype(pa.duration("ns")),
    )
    result = ser.dt.total_seconds()
    expected = pd.Series(
        ArrowExtensionArray(pa.array([86402.000003, None], type=pa.float64()))
    )
    tm.assert_series_equal(result, expected)


def test_dt_to_pytimedelta():
    # [FUNCTION DOC]
    # GH 52284. Converting arrow durations to python timedeltas gives an object array of timedelta objects and emits a deprecation warning (version four warning class) for the arrow accessor; the numpy-backed accessor emits its own deprecation warning and returns the same.
    # GH 52284
    data = [timedelta(1, 2, 3), timedelta(1, 2, 4)]
    ser = pd.Series(data, dtype=ArrowDtype(pa.duration("ns")))

    msg = "The behavior of ArrowTemporalProperties.to_pytimedelta is deprecated"
    with tm.assert_produces_warning(Pandas4Warning, match=msg):
        result = ser.dt.to_pytimedelta()
    expected = np.array(data, dtype=object)
    tm.assert_numpy_array_equal(result, expected)
    assert all(type(res) is timedelta for res in result)

    msg = "The behavior of TimedeltaProperties.to_pytimedelta is deprecated"
    with tm.assert_produces_warning(Pandas4Warning, match=msg):
        expected = ser.astype("timedelta64[ns]").dt.to_pytimedelta()
    tm.assert_numpy_array_equal(result, expected)


def test_dt_components():
    # [FUNCTION DOC]
    # GH 52284. The components of the one-day-two-seconds duration give a frame with columns days, hours, minutes, seconds, milliseconds, microseconds, nanoseconds of arrow 32-bit integers (1, 0, 0, 2, 0, 3, 4) with a missing row.
    # GH 52284
    ser = pd.Series(
        [
            pd.Timedelta(
                days=1,
                seconds=2,
                microseconds=3,
                nanoseconds=4,
            ),
            None,
        ],
        dtype=ArrowDtype(pa.duration("ns")),
    )
    result = ser.dt.components
    expected = pd.DataFrame(
        [[1, 0, 0, 2, 0, 3, 4], [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA]],
        columns=[
            "days",
            "hours",
            "minutes",
            "seconds",
            "milliseconds",
            "microseconds",
            "nanoseconds",
        ],
        dtype="int32[pyarrow]",
    )
    tm.assert_frame_equal(result, expected)


def test_dt_components_large_values():
    # [FUNCTION DOC]
    # Components of a 365-day, 23:59:59.999 duration: days 365, hours 23, minutes 59, seconds 59, milliseconds 999, microseconds and nanoseconds zero; missing row.
    ser = pd.Series(
        [
            pd.Timedelta("365 days 23:59:59.999000"),
            None,
        ],
        dtype=ArrowDtype(pa.duration("ns")),
    )
    result = ser.dt.components
    expected = pd.DataFrame(
        [
            [365, 23, 59, 59, 999, 0, 0],
            [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
        ],
        columns=[
            "days",
            "hours",
            "minutes",
            "seconds",
            "milliseconds",
            "microseconds",
            "nanoseconds",
        ],
        dtype="int32[pyarrow]",
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("skipna", [True, False])
def test_boolean_reduce_series_all_null(all_boolean_reductions, skipna):
    # [FUNCTION DOC]
    # GH51624. Parametrized over skipna and the boolean reductions. For an all-null float arrow Series: with skipna the result is true for all and false for any; without skipna the result is NA (identity checked).
    # GH51624
    ser = pd.Series([None], dtype="float64[pyarrow]")
    result = getattr(ser, all_boolean_reductions)(skipna=skipna)
    if skipna:
        expected = all_boolean_reductions == "all"
    else:
        expected = pd.NA
    assert result is expected


def test_from_sequence_of_strings_boolean():
    # [FUNCTION DOC]
    # Building a boolean arrow array from strings: true spellings (true, TRUE, True, 1, 1.0) give true, false spellings give false, a null stays null; an unparsable string raises an arrow invalid error saying it failed to parse.
    true_strings = ["true", "TRUE", "True", "1", "1.0"]
    false_strings = ["false", "FALSE", "False", "0", "0.0"]
    nulls = [None]
    strings = true_strings + false_strings + nulls
    bools = (
        [True] * len(true_strings) + [False] * len(false_strings) + [None] * len(nulls)
    )

    dtype = ArrowDtype(pa.bool_())
    result = ArrowExtensionArray._from_sequence_of_strings(strings, dtype=dtype)
    expected = pd.array(bools, dtype="boolean[pyarrow]")
    tm.assert_extension_array_equal(result, expected)

    strings = ["True", "foo"]
    with pytest.raises(pa.ArrowInvalid, match="Failed to parse"):
        ArrowExtensionArray._from_sequence_of_strings(strings, dtype=dtype)


def test_concat_empty_arrow_backed_series(dtype):
    # [FUNCTION DOC]
    # GH#51734. For every arrow dtype, concatenating an empty Series selected by an empty boolean mask returns an empty Series equal to the original.
    # GH#51734
    ser = pd.Series([], dtype=dtype)
    expected = ser.copy()
    result = pd.concat([ser[np.array([], dtype=np.bool_)]])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("dtype", ["string", "string[pyarrow]"])
def test_series_from_string_array(dtype):
    # [FUNCTION DOC]
    # Parametrized over string and pyarrow string dtype names. A Series built from an arrow string array with that dtype equals one built from the arrow extension array with that dtype.
    arr = pa.array("the quick brown fox".split())
    ser = pd.Series(arr, dtype=dtype)
    expected = pd.Series(ArrowExtensionArray(arr), dtype=dtype)
    tm.assert_series_equal(ser, expected)


# _data was renamed to _pa_data
class OldArrowExtensionArray(ArrowExtensionArray):
    # [CLASS DOC]
    # HELPER: test-only subclass of the arrow extension array that mimics the old pickle format, in which the underlying array was stored under the name data (it was renamed). Used for the pickle compatibility test.
    def __getstate__(self):
        # [FUNCTION DOC]
        # HELPER: returns the parent pickle state with the private arrow array entry renamed to the old name data.
        state = super().__getstate__()
        state["_data"] = state.pop("_pa_array")
        return state


def test_pickle_old_arrowextensionarray():
    # [FUNCTION DOC]
    # Verifies unpickling an old-format array (state with the old data name) loads correctly: the result equals the original, its arrow array is a chunked array of the data, and no old data attribute remains.
    data = pa.array([1])
    expected = OldArrowExtensionArray(data)
    result = pickle.loads(pickle.dumps(expected))
    tm.assert_extension_array_equal(result, expected)
    assert result._pa_array == pa.chunked_array(data)
    assert not hasattr(result, "_data")


def test_setitem_boolean_replace_with_mask_segfault():
    # [FUNCTION DOC]
    # GH#52059. Assigning false through an all-false mask of 145,000 elements to a chunked boolean array (large enough to have triggered a segmentation fault) leaves the array unchanged.
    # GH#52059
    N = 145_000
    arr = ArrowExtensionArray(pa.chunked_array([np.ones((N,), dtype=np.bool_)]))
    expected = arr.copy()
    arr[np.zeros((N,), dtype=np.bool_)] = False
    assert arr._pa_array == expected._pa_array


def test_setitem_na_chunked_string_if_else():
    # [FUNCTION DOC]
    # GH#64320. Frame concatenated from two chunks of string columns; five repeated assignments of NA to the first row by label do not corrupt: row zero is missing, the other rows of that column and the whole other column remain x.
    # GH#64320
    df = pd.concat(
        [
            pd.DataFrame({"a": ["x"] * 5, "b": ["x"] * 5}),
            pd.DataFrame({"a": ["x"] * 5, "b": ["x"] * 5}),
        ],
        ignore_index=True,
    )
    for _ in range(5):
        df.loc[[0], "a"] = pd.NA
    assert pd.isna(df["a"].iloc[0])
    assert (df["a"].iloc[1:] == "x").all()
    assert (df["b"] == "x").all()


@pytest.mark.parametrize(
    "data, arrow_dtype",
    [
        ([b"a", b"b"], pa.large_binary()),
        (["a", "b"], pa.large_string()),
    ],
)
def test_conversion_large_dtypes_from_numpy_array(data, arrow_dtype):
    # [FUNCTION DOC]
    # Parametrized over (values, arrow type): large binary and large string. Building an arrow array from a numpy array of the values with the large dtype equals building from the python list.
    dtype = ArrowDtype(arrow_dtype)
    result = pd.array(np.array(data), dtype=dtype)
    expected = pd.array(data, dtype=dtype)
    tm.assert_extension_array_equal(result, expected)


def test_concat_null_array():
    # [FUNCTION DOC]
    # Concatenating a frame with a null-typed column and a frame with a 64-bit arrow integer column gives an integer arrow column with nulls first.
    df = pd.DataFrame({"a": [None, None]}, dtype=ArrowDtype(pa.null()))
    df2 = pd.DataFrame({"a": [0, 1]}, dtype="int64[pyarrow]")

    result = pd.concat([df, df2], ignore_index=True)
    expected = pd.DataFrame({"a": [None, None, 0, 1]}, dtype="int64[pyarrow]")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("pa_type", tm.ALL_INT_PYARROW_DTYPES + tm.FLOAT_PYARROW_DTYPES)
def test_describe_numeric_data(pa_type):
    # [FUNCTION DOC]
    # GH 52470. Parametrized over every integer and floating arrow type. Describing the Series 1, 2, 3 gives count, mean, std, min, quartiles and max as arrow 64-bit float.
    # GH 52470
    data = pd.Series([1, 2, 3], dtype=ArrowDtype(pa_type))
    result = data.describe()
    expected = pd.Series(
        [3, 2, 1, 1, 1.5, 2.0, 2.5, 3],
        dtype=ArrowDtype(pa.float64()),
        index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("pa_type", tm.TIMEDELTA_PYARROW_DTYPES)
def test_describe_timedelta_data(pa_type):
    # [FUNCTION DOC]
    # GH53001. Parametrized over arrow duration types. Describing the Series 1..9 gives object-dtype results: count nine and timedelta values for mean, std, min and quartiles and max in the type's unit.
    # GH53001
    data = pd.Series(range(1, 10), dtype=ArrowDtype(pa_type))
    result = data.describe()
    expected = pd.Series(
        [9, *pd.to_timedelta([5, 2, 1, 3, 5, 7, 9], unit=pa_type.unit).tolist()],
        dtype=object,
        index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("pa_type", tm.DATETIME_PYARROW_DTYPES)
def test_describe_datetime_data(pa_type):
    # [FUNCTION DOC]
    # GH53001. Parametrized over arrow timestamp types. Describing the Series 1..9 gives object results: count nine and timestamps (with the type's time zone and unit) for mean, min, quartiles and max (no std).
    # GH53001
    data = pd.Series(range(1, 10), dtype=ArrowDtype(pa_type))
    result = data.describe()
    expected = pd.Series(
        [9]
        + [
            pd.Timestamp(v, tz=pa_type.tz, unit=pa_type.unit)
            for v in [5, 1, 3, 5, 7, 9]
        ],
        dtype=object,
        index=["count", "mean", "min", "25%", "50%", "75%", "max"],
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "pa_type", tm.DATETIME_PYARROW_DTYPES + tm.TIMEDELTA_PYARROW_DTYPES
)
def test_quantile_temporal(pa_type):
    # [FUNCTION DOC]
    # GH52678. Parametrized over datetime and duration arrow types. The 0.1 quantile of the Series 1, 2, 3 equals the first element.
    # GH52678
    data = [1, 2, 3]
    ser = pd.Series(data, dtype=ArrowDtype(pa_type))
    result = ser.quantile(0.1)
    expected = ser[0]
    assert result == expected


def test_date32_repr():
    # [FUNCTION DOC]
    # GH48238. The Series representation of a 32-bit date arrow value is the date followed by the dtype line date32[day][pyarrow].
    # GH48238
    arrow_dt = pa.array([date.fromisoformat("2020-01-01")], type=pa.date32())
    ser = pd.Series(arrow_dt, dtype=ArrowDtype(arrow_dt.type))
    assert repr(ser) == "0    2020-01-01\ndtype: date32[day][pyarrow]"


def test_duration_overflow_from_ndarray_containing_nat():
    # [FUNCTION DOC]
    # GH52843. Adding a nanosecond-timestamp arrow Series built from numpy datetimes containing NaT and a duration Series with a missing value gives timestamps 2 and null (no overflow).
    # GH52843
    data_ts = pd.to_datetime([1, None])
    data_td = pd.to_timedelta([1, None])
    ser_ts = pd.Series(data_ts, dtype=ArrowDtype(pa.timestamp("ns")))
    ser_td = pd.Series(data_td, dtype=ArrowDtype(pa.duration("ns")))
    result = ser_ts + ser_td
    expected = pd.Series([2, None], dtype=ArrowDtype(pa.timestamp("ns")))
    tm.assert_series_equal(result, expected)


def test_infer_dtype_pyarrow_dtype(data, request):
    # [FUNCTION DOC]
    # For every arrow dtype, inferring the data's type does not return unknown-array, and equals the inferred type of the python list of its elements (skipping missing values); data with missing values that infer as datetime64 or timedelta64 is expected to fail because NA is not ignored in the list check.
    res = lib.infer_dtype(data)
    assert res != "unknown-array"

    if data._hasna and res in ["datetime64", "timedelta64"]:
        mark = pytest.mark.xfail(
            reason="in infer_dtype pd.NA is not ignored in these cases "
            "even with skipna=True in the list(data) check below"
        )
        request.applymarker(mark)

    assert res == lib.infer_dtype(list(data), skipna=True)


@pytest.mark.parametrize(
    "pa_type", tm.DATETIME_PYARROW_DTYPES + tm.TIMEDELTA_PYARROW_DTYPES
)
def test_from_sequence_temporal(pa_type):
    # [FUNCTION DOC]
    # GH 53171. Parametrized over datetime and duration arrow types. Building from one timestamp or timedelta scalar in the same unit gives the same array as built from the integer value.
    # GH 53171
    val = 3
    unit = pa_type.unit
    if pa.types.is_duration(pa_type):
        seq = [pd.Timedelta(val, unit=unit).as_unit(unit)]
    else:
        seq = [pd.Timestamp(val, unit=unit, tz=pa_type.tz).as_unit(unit)]

    result = ArrowExtensionArray._from_sequence(seq, dtype=pa_type)
    expected = ArrowExtensionArray(pa.array([val], type=pa_type))
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize(
    "pa_type", tm.DATETIME_PYARROW_DTYPES + tm.TIMEDELTA_PYARROW_DTYPES
)
def test_setitem_temporal(pa_type):
    # [FUNCTION DOC]
    # GH 53171. Parametrized over temporal arrow types. Assigning a scalar timestamp or timedelta to a full slice of an array [1, 2, 3] gives [1, 1, 1].
    # GH 53171
    unit = pa_type.unit
    if pa.types.is_duration(pa_type):
        val = pd.Timedelta(1, unit=unit).as_unit(unit)
    else:
        val = pd.Timestamp(1, unit=unit, tz=pa_type.tz).as_unit(unit)

    arr = ArrowExtensionArray(pa.array([1, 2, 3], type=pa_type))

    result = arr.copy()
    result[:] = val
    expected = ArrowExtensionArray(pa.array([1, 1, 1], type=pa_type))
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize(
    "pa_type", tm.DATETIME_PYARROW_DTYPES + tm.TIMEDELTA_PYARROW_DTYPES
)
def test_arithmetic_temporal(pa_type, request):
    # [FUNCTION DOC]
    # GH 53171. Subtracting a one-unit timedelta from an array [1, 2, 3] of timestamps or durations gives [0, 1, 2] in the same arrow type. The request fixture is unused.
    # GH 53171
    arr = ArrowExtensionArray(pa.array([1, 2, 3], type=pa_type))
    unit = pa_type.unit
    result = arr - pd.Timedelta(1, unit=unit).as_unit(unit)
    expected = ArrowExtensionArray(pa.array([0, 1, 2], type=pa_type))
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize(
    "pa_type", tm.DATETIME_PYARROW_DTYPES + tm.TIMEDELTA_PYARROW_DTYPES
)
def test_comparison_temporal(pa_type):
    # [FUNCTION DOC]
    # GH 53171. Comparing greater-than a scalar of value one with an array [1, 2, 3] gives booleans [false, true, true].
    # GH 53171
    unit = pa_type.unit
    if pa.types.is_duration(pa_type):
        val = pd.Timedelta(1, unit=unit).as_unit(unit)
    else:
        val = pd.Timestamp(1, unit=unit, tz=pa_type.tz).as_unit(unit)

    arr = ArrowExtensionArray(pa.array([1, 2, 3], type=pa_type))

    result = arr > val
    expected = ArrowExtensionArray(pa.array([False, True, True], type=pa.bool_()))
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize(
    "pa_type", tm.DATETIME_PYARROW_DTYPES + tm.TIMEDELTA_PYARROW_DTYPES
)
def test_getitem_temporal(pa_type):
    # [FUNCTION DOC]
    # GH 53326. Item lookup returns a Timedelta (durations) or Timestamp (timestamps) scalar with the same unit and time zone as the type and the expected value.
    # GH 53326
    arr = ArrowExtensionArray(pa.array([1, 2, 3], type=pa_type))
    result = arr[1]
    if pa.types.is_duration(pa_type):
        expected = pd.Timedelta(2, unit=pa_type.unit).as_unit(pa_type.unit)
        assert isinstance(result, pd.Timedelta)
    else:
        expected = pd.Timestamp(2, unit=pa_type.unit, tz=pa_type.tz).as_unit(
            pa_type.unit
        )
        assert isinstance(result, pd.Timestamp)
    assert result.unit == expected.unit
    assert result == expected


@pytest.mark.parametrize(
    "pa_type", tm.DATETIME_PYARROW_DTYPES + tm.TIMEDELTA_PYARROW_DTYPES
)
def test_iter_temporal(pa_type):
    # [FUNCTION DOC]
    # GH 53326. Iterating over an array [1, null] yields the scalar Timedelta or Timestamp (matching unit) followed by NA.
    # GH 53326
    arr = ArrowExtensionArray(pa.array([1, None], type=pa_type))
    result = list(arr)
    if pa.types.is_duration(pa_type):
        expected = [
            pd.Timedelta(1, unit=pa_type.unit).as_unit(pa_type.unit),
            pd.NA,
        ]
        assert isinstance(result[0], pd.Timedelta)
    else:
        expected = [
            pd.Timestamp(1, unit=pa_type.unit, tz=pa_type.tz).as_unit(pa_type.unit),
            pd.NA,
        ]
        assert isinstance(result[0], pd.Timestamp)
    assert result[0].unit == expected[0].unit
    assert result == expected


def test_groupby_series_size_returns_pa_int(data):
    # [FUNCTION DOC]
    # GH 54132. Grouping the first three data elements by index level and taking sizes gives arrow 64-bit integers [2, 1] indexed by a and b.
    # GH 54132
    ser = pd.Series(data[:3], index=["a", "a", "b"])
    result = ser.groupby(level=0).size()
    expected = pd.Series([2, 1], dtype="int64[pyarrow]", index=["a", "b"])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "pa_type", tm.DATETIME_PYARROW_DTYPES + tm.TIMEDELTA_PYARROW_DTYPES, ids=repr
)
@pytest.mark.parametrize("dtype", [None, object])
def test_to_numpy_temporal(pa_type, dtype):
    # [FUNCTION DOC]
    # GH 53326 and GH 55997. Parametrized over datetime and duration arrow types and the numpy dtype requested (default or object). Converting [1, null] to numpy: with object dtype (NA) or tz-aware timestamps (NaT) gives an object array of the scalars with matching unit; otherwise a numpy datetime or timedelta array with NaT in the type's unit.
    # GH 53326
    # GH 55997: Return datetime64/timedelta64 types with NaT if possible
    arr = ArrowExtensionArray(pa.array([1, None], type=pa_type))
    result = arr.to_numpy(dtype=dtype)
    if pa.types.is_duration(pa_type):
        value = pd.Timedelta(1, unit=pa_type.unit).as_unit(pa_type.unit)
    else:
        value = pd.Timestamp(1, unit=pa_type.unit, tz=pa_type.tz).as_unit(pa_type.unit)

    if dtype == object or (pa.types.is_timestamp(pa_type) and pa_type.tz is not None):
        if dtype == object:
            na = pd.NA
        else:
            na = pd.NaT
        expected = np.array([value, na], dtype=object)
        assert result[0].unit == value.unit
    else:
        na = pa_type.to_pandas_dtype().type("nat", pa_type.unit)
        value = value.to_numpy()
        expected = np.array([value, na])
        assert np.datetime_data(result[0])[0] == pa_type.unit
    tm.assert_numpy_array_equal(result, expected)


def test_groupby_count_return_arrow_dtype(data_missing):
    # [FUNCTION DOC]
    # Counting non-missing values per group of a frame with two arrow columns [missing, valid] grouped by a constant key gives an arrow 64-bit integer frame.
    df = pd.DataFrame({"A": [1, 1], "B": data_missing, "C": data_missing})
    result = df.groupby("A").count()
    expected = pd.DataFrame(
        [[1, 1]],
        index=pd.Index([1], name="A"),
        columns=["B", "C"],
        dtype="int64[pyarrow]",
    )
    tm.assert_frame_equal(result, expected)


def test_fixed_size_list():
    # [FUNCTION DOC]
    # GH#55000. A Series of fixed-size list arrow type has the python list class as its dtype scalar type.
    # GH#55000
    ser = pd.Series(
        [[1, 2], [3, 4]], dtype=ArrowDtype(pa.list_(pa.int64(), list_size=2))
    )
    result = ser.dtype.type
    assert result == list


def test_arrowextensiondtype_dataframe_repr():
    # [FUNCTION DOC]
    # GH 54062. The frame representation of an arrow-extension dtype wrapping a period type shows the underlying ordinal integers. A TODO notes the representation may not be as expected.
    # GH 54062
    df = pd.DataFrame(
        pd.period_range("2012", periods=3),
        columns=["col"],
        dtype=ArrowDtype(ArrowPeriodType("D")),
    )
    result = repr(df)
    # TODO: repr value may not be expected; address how
    # pyarrow.ExtensionType values are displayed
    expected = "     col\n0  15340\n1  15341\n2  15342"
    assert result == expected


def test_pow_missing_operand():
    # [FUNCTION DOC]
    # GH 55512. The power method with a missing operand and a fill value of three on [2, null] gives [8, null] in arrow 64-bit integer.
    # GH 55512
    k = pd.Series([2, None], dtype="int64[pyarrow]")
    result = k.pow(None, fill_value=3)
    expected = pd.Series([8, None], dtype="int64[pyarrow]")
    tm.assert_series_equal(result, expected)


def test_decimal_parse_raises():
    # [FUNCTION DOC]
    # GH 56984. Casting the string 1.2345 to a decimal with precision 1 and scale 0 raises an arrow invalid error saying rescaling would lose data.
    # GH 56984
    ser = pd.Series(["1.2345"], dtype=ArrowDtype(pa.string()))
    with pytest.raises(
        pa.lib.ArrowInvalid, match="Rescaling Decimal(128)? value would cause data loss"
    ):
        ser.astype(ArrowDtype(pa.decimal128(1, 0)))


def test_decimal_parse_succeeds():
    # [FUNCTION DOC]
    # GH 56984. Casting the string 1.2345 to decimal with precision 5 and scale 4 gives the decimal value.
    # GH 56984
    ser = pd.Series(["1.2345"], dtype=ArrowDtype(pa.string()))
    dtype = ArrowDtype(pa.decimal128(5, 4))
    result = ser.astype(dtype)
    expected = pd.Series([Decimal("1.2345")], dtype=dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("pa_type", tm.TIMEDELTA_PYARROW_DTYPES)
def test_duration_fillna_numpy(pa_type):
    # [FUNCTION DOC]
    # GH 54707. Parametrized over arrow duration types. Filling nulls in an arrow duration Series with values from a numpy timedelta Series of the same unit gives [1, 2].
    # GH 54707
    ser1 = pd.Series([None, 2], dtype=ArrowDtype(pa_type))
    ser2 = pd.Series(np.array([1, 3], dtype=f"m8[{pa_type.unit}]"))
    result = ser1.fillna(ser2)
    expected = pd.Series([1, 2], dtype=ArrowDtype(pa_type))
    tm.assert_series_equal(result, expected)


def test_comparison_not_propagating_arrow_error():
    # [FUNCTION DOC]
    # GH#54944. Comparing an unsigned arrow value that does not fit signed 64-bit (2 to the 63rd) with a null signed Series raises an arrow invalid error (message about the integer value) instead of being swallowed.
    # GH#54944
    a = pd.Series([1 << 63], dtype="uint64[pyarrow]")
    b = pd.Series([None], dtype="int64[pyarrow]")
    with pytest.raises(pa.lib.ArrowInvalid, match="Integer value"):
        a < b


def test_factorize_chunked_dictionary():
    # [FUNCTION DOC]
    # GH 54844. Factorizing a Series made of two chunks each dictionary-encoded gives integer codes [0, 1] and uniques equal to the combined dictionary array as an index.
    # GH 54844
    pa_array = pa.chunked_array(
        [pa.array(["a"]).dictionary_encode(), pa.array(["b"]).dictionary_encode()]
    )
    ser = pd.Series(ArrowExtensionArray(pa_array))
    res_indices, res_uniques = ser.factorize()
    exp_indices = np.array([0, 1], dtype=np.intp)
    exp_uniques = pd.Index(ArrowExtensionArray(pa_array.combine_chunks()))
    tm.assert_numpy_array_equal(res_indices, exp_indices)
    tm.assert_index_equal(res_uniques, exp_uniques)


def test_factorize_dictionary_with_na():
    # [FUNCTION DOC]
    # GH#60567. Factorizing a dictionary-typed array of [a1, NA] without the NA sentinel gives codes [0, 1] and uniques [a1, null] as arrow strings.
    # GH#60567
    arr = pd.array(
        ["a1", pd.NA], dtype=ArrowDtype(pa.dictionary(pa.int32(), pa.utf8()))
    )
    indices, uniques = arr.factorize(use_na_sentinel=False)
    expected_indices = np.array([0, 1], dtype=np.intp)
    expected_uniques = pd.array(["a1", None], dtype=ArrowDtype(pa.string()))
    tm.assert_numpy_array_equal(indices, expected_indices)
    tm.assert_extension_array_equal(uniques, expected_uniques)


def test_dictionary_astype_categorical():
    # [FUNCTION DOC]
    # GH#56672. Casting a Series from chunked dictionary-encoded strings to category gives a categorical with categories in order of appearance, a, x, c, d (as an arrow-string index).
    # GH#56672
    arrs = [
        pa.array(np.array(["a", "x", "c", "a"])).dictionary_encode(),
        pa.array(np.array(["a", "d", "c"])).dictionary_encode(),
    ]
    ser = pd.Series(ArrowExtensionArray(pa.chunked_array(arrs)))
    result = ser.astype("category")
    categories = pd.Index(["a", "x", "c", "d"], dtype=ArrowDtype(pa.string()))
    expected = pd.Series(
        ["a", "x", "c", "a", "a", "d", "c"],
        dtype=pd.CategoricalDtype(categories=categories),
    )
    tm.assert_series_equal(result, expected)


def test_arrow_floordiv():
    # [FUNCTION DOC]
    # GH 55561. Floor division of -7 by 4 for arrow 64-bit integers gives -2 (rounding toward negative infinity).
    # GH 55561
    a = pd.Series([-7], dtype="int64[pyarrow]")
    b = pd.Series([4], dtype="int64[pyarrow]")
    expected = pd.Series([-2], dtype="int64[pyarrow]")
    result = a // b
    tm.assert_series_equal(result, expected)


def test_arrow_floordiv_large_values():
    # [FUNCTION DOC]
    # GH 56645. Floor division of a large nanosecond integer by one million gives the exact integer result.
    # GH 56645
    a = pd.Series([1425801600000000000], dtype="int64[pyarrow]")
    expected = pd.Series([1425801600000], dtype="int64[pyarrow]")
    result = a // 1_000_000
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("dtype", ["int64[pyarrow]", "uint64[pyarrow]"])
def test_arrow_floordiv_large_integral_result(dtype):
    # [FUNCTION DOC]
    # GH 56676. Parametrized over int64 and uint64. Floor division of a large integral value by one returns the same value without loss of precision.
    # GH 56676
    a = pd.Series([18014398509481983], dtype=dtype)
    result = a // 1
    tm.assert_series_equal(result, a)


@pytest.mark.parametrize("pa_type", tm.SIGNED_INT_PYARROW_DTYPES)
def test_arrow_floordiv_larger_divisor(pa_type):
    # [FUNCTION DOC]
    # GH 56676. Parametrized over signed integer arrow types. Floor division of -23 by 24 yields -1.
    # GH 56676
    dtype = ArrowDtype(pa_type)
    a = pd.Series([-23], dtype=dtype)
    result = a // 24
    expected = pd.Series([-1], dtype=dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("pa_type", tm.SIGNED_INT_PYARROW_DTYPES)
def test_arrow_floordiv_integral_invalid(pa_type):
    # [FUNCTION DOC]
    # GH 56676. Parametrized over signed integer types. Floor dividing the minimum value by negative one raises an arrow invalid error mentioning overflow or range; dividing by zero raises one mentioning divide by zero.
    # GH 56676
    min_value = np.iinfo(pa_type.to_pandas_dtype()).min
    a = pd.Series([min_value], dtype=ArrowDtype(pa_type))
    with pytest.raises(pa.lib.ArrowInvalid, match="overflow|not in range"):
        a // -1
    with pytest.raises(pa.lib.ArrowInvalid, match="divide by zero"):
        a // 0


@pytest.mark.parametrize("dtype", tm.FLOAT_PYARROW_DTYPES_STR_REPR)
def test_arrow_floordiv_floating_0_divisor(dtype):
    # [FUNCTION DOC]
    # GH 56676. Parametrized over floating arrow dtype strings. Floor dividing 2 by zero yields positive infinity of the same dtype.
    # [FUNCTION DOC]
    # GH 56676. Parametrized over floating dtype strings (continues).
    # GH 56676
    a = pd.Series([2], dtype=dtype)
    result = a // 0
    expected = pd.Series([float("inf")], dtype=dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("dtype", ["float64", "datetime64[ns]", "timedelta64[ns]"])
def test_astype_int_with_null_to_numpy_dtype(dtype):
    # [FUNCTION DOC]
    # GH 57093. Parametrized over float64, datetime64 and timedelta64 numpy dtypes. Casting an arrow integer Series with a null to that numpy dtype gives the matching values with NaN/NaT for the null.
    # GH 57093
    ser = pd.Series([1, None], dtype="int64[pyarrow]")
    result = ser.astype(dtype)
    expected = pd.Series([1, None], dtype=dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("pa_type", tm.ALL_INT_PYARROW_DTYPES)
def test_arrow_integral_floordiv_large_values(pa_type):
    # [FUNCTION DOC]
    # GH 56676. Parametrized over every integer arrow type. Floor dividing the type's maximum value by one returns the maximum unchanged.
    # GH 56676
    max_value = np.iinfo(pa_type.to_pandas_dtype()).max
    dtype = ArrowDtype(pa_type)
    a = pd.Series([max_value], dtype=dtype)
    b = pd.Series([1], dtype=dtype)
    result = a // b
    tm.assert_series_equal(result, a)


@pytest.mark.parametrize("dtype", ["int64[pyarrow]", "uint64[pyarrow]"])
def test_arrow_true_division_large_divisor(dtype):
    # [FUNCTION DOC]
    # GH 56706. Parametrized over int64 and uint64. True division of zero by a very large integer gives zero as arrow 64-bit float.
    # GH 56706
    a = pd.Series([0], dtype=dtype)
    b = pd.Series([18014398509481983], dtype=dtype)
    expected = pd.Series([0], dtype="float64[pyarrow]")
    result = a / b
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("dtype", ["int64[pyarrow]", "uint64[pyarrow]"])
def test_arrow_floor_division_large_divisor(dtype):
    # [FUNCTION DOC]
    # GH 56706. Parametrized over int64 and uint64. Floor division of zero by a very large integer gives zero in the same dtype.
    # GH 56706
    a = pd.Series([0], dtype=dtype)
    b = pd.Series([18014398509481983], dtype=dtype)
    expected = pd.Series([0], dtype=dtype)
    result = a // b
    tm.assert_series_equal(result, expected)


def test_string_to_datetime_parsing_cast():
    # [FUNCTION DOC]
    # GH 56266. Building a second-resolution arrow timestamp Series from date strings equals parsing the strings with the datetime parser, setting unit seconds and converting to arrow.
    # GH 56266
    string_dates = ["2020-01-01 04:30:00", "2020-01-02 00:00:00", "2020-01-03 00:00:00"]
    result = pd.Series(string_dates, dtype="timestamp[s][pyarrow]")

    pd_res = pd.to_datetime(string_dates).as_unit("s")
    expected = pd.Series(ArrowExtensionArray(pa.array(pd_res, from_pandas=True)))
    tm.assert_series_equal(result, expected)


def test_interpolate_not_numeric(data):
    # [FUNCTION DOC]
    # For non-numeric arrow types, interpolating a Series raises a type error with a message that it cannot interpolate with that dtype; numeric types are not tested.
    if not data.dtype._is_numeric:
        ser = pd.Series(data)
        msg = re.escape(f"Cannot interpolate with {ser.dtype} dtype")
        with pytest.raises(TypeError, match=msg):
            pd.Series(data).interpolate()


@pytest.mark.parametrize("dtype", ["int64[pyarrow]", "float64[pyarrow]"])
def test_interpolate_linear(dtype):
    # [FUNCTION DOC]
    # GH#65345. Parametrized over int64 and float64 arrow types. Linear interpolation of [null, 1, 2, null, 4, null] gives [null, 1.0, 2.0, 3.0, 4.0, 4.0] as arrow 64-bit float (results match the masked dtypes: upcast to float, fill the trailing null forward).
    # GH#65345 results should match the masked (e.g. Int64) dtypes:
    # upcast to float, and fill the trailing NA going forward
    ser = pd.Series([None, 1, 2, None, 4, None], dtype=dtype)
    result = ser.interpolate()
    expected = pd.Series([None, 1.0, 2.0, 3.0, 4.0, 4.0], dtype="float64[pyarrow]")
    tm.assert_series_equal(result, expected)


def test_interpolate_linear_consecutive_na():
    # [FUNCTION DOC]
    # GH#65345. Interpolating consecutive interior nulls in an integer arrow Series forward gives a continuous float sequence (they were previously left unfilled).
    # GH#65345 consecutive interior NAs were left unfilled
    ser = pd.Series([1, 2, 3, None, None, 6, 7], dtype="int64[pyarrow]")
    result = ser.interpolate(method="linear", limit_direction="forward")
    expected = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype="float64[pyarrow]")
    tm.assert_series_equal(result, expected)


def test_interpolate_linear_int_fractional():
    # [FUNCTION DOC]
    # GH#65345. Linear interpolation of an integer arrow Series [1, null, 2] gives 1.5 (not truncated) as float.
    # GH#65345 result should not truncate the interpolated value (1 instead of 1.5)
    ser = pd.Series([1, None, 2], dtype="int64[pyarrow]")
    result = ser.interpolate(method="linear")
    expected = pd.Series([1.0, 1.5, 2.0], dtype="float64[pyarrow]")
    tm.assert_series_equal(result, expected)


def test_string_to_time_parsing_cast():
    # [FUNCTION DOC]
    # GH 56463. A Series of a time string with microsecond precision built with the 64-bit time arrow dtype equals the arrow time array of the same value.
    # GH 56463
    string_times = ["11:41:43.076160"]
    result = pd.Series(string_times, dtype="time64[us][pyarrow]")
    expected = pd.Series(
        ArrowExtensionArray(pa.array([time(11, 41, 43, 76160)], from_pandas=True))
    )
    tm.assert_series_equal(result, expected)


def test_to_numpy_float():
    # [FUNCTION DOC]
    # GH#56267. Casting a float arrow Series with a null to numpy 64-bit float gives NaN for the null.
    # GH#56267
    ser = pd.Series([32, 40, None], dtype="float[pyarrow]")
    result = ser.astype("float64")
    expected = pd.Series([32, 40, np.nan], dtype="float64")
    tm.assert_series_equal(result, expected)


def test_to_numpy_timestamp_to_int():
    # [FUNCTION DOC]
    # GH 55997. Converting a nanosecond arrow timestamp to numpy 64-bit integer gives the number of nanoseconds since the epoch.
    # GH 55997
    ser = pd.Series(["2020-01-01 04:30:00"], dtype="timestamp[ns][pyarrow]")
    result = ser.to_numpy(dtype=np.int64)
    expected = np.array([1577853000000000000])
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("arrow_type", [pa.large_string(), pa.string()])
def test_cast_dictionary_different_value_dtype(arrow_type):
    # [FUNCTION DOC]
    # Parametrized over large string and string value types. Casting a pyarrow string frame column to a dictionary type whose values are the given string type gives that dtype.
    df = pd.DataFrame({"a": ["x", "y"]}, dtype="string[pyarrow]")
    data_type = ArrowDtype(pa.dictionary(pa.int32(), arrow_type))
    result = df.astype({"a": data_type})
    assert result.dtypes.iloc[0] == data_type


def test_map_numeric_na_action(using_nan_is_na):
    # [FUNCTION DOC]
    # Mapping a constant function with the ignore action over a 64-bit integer arrow Series with a null: valid values map to 42.0; the null stays NA (object result) or NaN (float64 result) depending on the NaN-is-NA mode.
    ser = pd.Series([32, 40, None], dtype="int64[pyarrow]")
    result = ser.map(lambda x: 42, na_action="ignore")
    if not using_nan_is_na:
        expected = pd.Series([42.0, 42.0, pd.NA], dtype="object")
    else:
        expected = pd.Series([42.0, 42.0, np.nan], dtype="float64")
    tm.assert_series_equal(result, expected)


def test_categorical_from_arrow_dictionary():
    # [FUNCTION DOC]
    # GH 60563. Counting value combinations (not dropping NA) in a frame with a dictionary-typed column gives counts of one for each, with a one-level multi-index of arrow strings and 64-bit integer count named count.
    # GH 60563
    df = pd.DataFrame(
        {"A": ["a1", "a2"]}, dtype=ArrowDtype(pa.dictionary(pa.int32(), pa.utf8()))
    )
    result = df.value_counts(dropna=False)
    expected = pd.Series(
        [1, 1],
        index=pd.MultiIndex.from_arrays(
            [pd.Index(["a1", "a2"], dtype=ArrowDtype(pa.string()), name="A")]
        ),
        name="count",
        dtype="int64",
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.skipif(
    pa_version_under19p0, reason="pa.json_ was introduced in pyarrow v19.0"
)
def test_arrow_json_type():
    # [FUNCTION DOC]
    # Skipped on pyarrow older than version 19 (the JSON type is not available). GH 60958: the scalar type of the arrow JSON dtype is the string class.
    # GH 60958
    dtype = ArrowDtype(pa.json_(pa.string()))
    result = dtype.type
    assert result == str


def test_timestamp_dtype_disallows_decimal():
    # [FUNCTION DOC]
    # GH#61773. Creating an arrow microsecond timestamp array containing a Timestamp and a decimal NaN raises a type error saying the decimal class is not convertible to datetime, the same as the datetime parser.
    # GH#61773 constructing with pyarrow timestamp dtype should disallow
    #  Decimal NaN, just like pd.to_datetime
    vals = [pd.Timestamp("2016-01-02 03:04:05"), Decimal("NaN")]

    msg = "<class 'decimal.Decimal'> is not convertible to datetime"
    with pytest.raises(TypeError, match=msg):
        # Check that the non-pyarrow version raises as expected
        pd.to_datetime(vals)

    with pytest.raises(TypeError, match=msg):
        pd.array(vals, dtype=ArrowDtype(pa.timestamp("us")))


def test_timestamp_dtype_matches_to_datetime():
    # [FUNCTION DOC]
    # GH#61775. A Series with the arrow tz-aware nanosecond timestamp dtype equals converting a numpy-backed tz-aware Series to the arrow backend.
    # GH#61775
    dtype1 = "datetime64[ns, US/Eastern]"
    dtype2 = "timestamp[ns, US/Eastern][pyarrow]"

    ts = pd.Timestamp("2025-07-03 18:10")

    result = pd.Series([ts], dtype=dtype2)
    expected = pd.Series([ts], dtype=dtype1).convert_dtypes(dtype_backend="pyarrow")

    tm.assert_series_equal(result, expected)


def test_timestamp_vs_dt64_comparison():
    # [FUNCTION DOC]
    # GH#60937. Equality between an arrow timestamp Series and the equal numpy datetime Series is [true] with arrow boolean dtype, in both operand orders.
    # GH#60937
    left = pd.Series(["2016-01-01"], dtype="timestamp[ns][pyarrow]")
    right = left.astype("datetime64[ns]")

    result = left == right
    expected = pd.Series([True], dtype="bool[pyarrow]")
    tm.assert_series_equal(result, expected)

    result = right == left
    tm.assert_series_equal(result, expected)


# TODO: reuse assert_invalid_comparison?
def test_date_vs_timestamp_scalar_comparison():
    # [FUNCTION DOC]
    # GH#62157. A date arrow Series never equals a timestamp scalar (and vice versa), nor does a timestamp Series equal a date scalar in either operand order; this matches non-pyarrow behavior. A TODO suggests reusing the invalid-comparison helper.
    # GH#62157 match non-pyarrow behavior
    ser = pd.Series(["2016-01-01"], dtype="date32[pyarrow]")
    ser2 = ser.astype("timestamp[ns][pyarrow]")

    ts = ser2[0]
    dt = ser[0]

    # date dtype don't match a Timestamp object
    assert not (ser == ts).any()
    assert not (ts == ser).any()

    # timestamp dtype doesn't match date object
    assert not (ser2 == dt).any()
    assert not (dt == ser2).any()


# TODO: reuse assert_invalid_comparison?
def test_date_vs_timestamp_array_comparison():
    # [FUNCTION DOC]
    # GH#62157. A date arrow Series compared with timestamp arrow and numpy datetime Series is never equal and always unequal, in both orders. A TODO suggests reusing the invalid-comparison helper; the issue comment in the source is incomplete.
    # GH#62157 match non-pyarrow behavior
    # GH#
    ser = pd.Series(["2016-01-01"], dtype="date32[pyarrow]")
    ser2 = ser.astype("timestamp[ns][pyarrow]")
    ser3 = ser.astype("datetime64[ns]")

    assert not (ser == ser2).any()
    assert not (ser2 == ser).any()
    assert (ser != ser2).all()
    assert (ser2 != ser).all()

    assert not (ser == ser3).any()
    assert not (ser3 == ser).any()
    assert (ser != ser3).all()
    assert (ser3 != ser).all()


def test_ops_with_nan_is_na(using_nan_is_na):
    # [FUNCTION DOC]
    # GH#61732. Subtracting or multiplying by NaN on a 64-bit integer arrow Series gives all-NA results when NaN is treated as NA, else non-NA values; division by zero yields NA at the zero element only in NA mode.
    # GH#61732
    ser = pd.Series([-1, 0, 1], dtype="int64[pyarrow]")

    result = ser - np.nan
    if using_nan_is_na:
        assert result.isna().all()
    else:
        assert not result.isna().any()

    result = ser * np.nan
    if using_nan_is_na:
        assert result.isna().all()
    else:
        assert not result.isna().any()

    result = ser / 0
    if using_nan_is_na:
        assert result.isna()[1]
    else:
        assert not result.isna()[1]


def test_setitem_float_nan_is_na(using_nan_is_na):
    # [FUNCTION DOC]
    # GH#61732. Setting NaN into an integer arrow Series makes the element NA in NA mode, else raises an arrow invalid error about converting nan to int64. In a float arrow Series, NaN is NA (and stays NA after setting) in NA mode, else it stays a float NaN, including after assignment.
    # GH#61732
    ser = pd.Series([-1, 0, 1], dtype="int64[pyarrow]")

    if using_nan_is_na:
        ser[1] = np.nan
        assert ser.isna()[1]
    else:
        msg = "Could not convert nan with type float: tried to convert to int64"
        with pytest.raises(pa.lib.ArrowInvalid, match=msg):
            ser[1] = np.nan

    ser = pd.Series([-1, np.nan, 1], dtype="float64[pyarrow]")
    if using_nan_is_na:
        assert ser.isna()[1]
        assert ser[1] is pd.NA

        ser[1] = np.nan
        assert ser[1] is pd.NA

    else:
        assert not ser.isna()[1]
        assert isinstance(ser[1], float)
        assert np.isnan(ser[1])

        ser[2] = np.nan
        assert isinstance(ser[2], float)
        assert np.isnan(ser[2])


def test_pow_with_all_na_float():
    # [FUNCTION DOC]
    # GH#62520. The power method with exponent two on an all-null float arrow Series gives all NA.
    # GH#62520

    s = pd.Series([None, None], dtype="float64[pyarrow]")
    result = s.pow(2)
    expected = pd.Series([pd.NA, pd.NA], dtype="float64[pyarrow]")
    tm.assert_series_equal(result, expected)


def test_mul_numpy_nullable_with_pyarrow_float():
    # [FUNCTION DOC]
    # GH#58602. Multiplying a masked 64-bit float Series by an arrow float Series (either order) gives squares as arrow 64-bit float; equality between them gives all true arrow booleans.
    # GH#58602
    left = pd.Series(range(5), dtype="Float64")
    right = pd.Series(range(5), dtype="float64[pyarrow]")

    expected = pd.Series([0, 1, 4, 9, 16], dtype="float64[pyarrow]")

    result = left * right
    tm.assert_series_equal(result, expected)

    result2 = right * left
    tm.assert_series_equal(result2, expected)

    # while we're here, let's check __eq__
    result3 = left == right
    expected3 = pd.Series([True] * 5, dtype="bool[pyarrow]")
    tm.assert_series_equal(result3, expected3)

    result4 = right == left
    tm.assert_series_equal(result4, expected3)


@pytest.mark.parametrize(
    "type_name, expected_size",
    [
        # Integer types
        ("int8", 1),
        ("int16", 2),
        ("int32", 4),
        ("int64", 8),
        ("uint8", 1),
        ("uint16", 2),
        ("uint32", 4),
        ("uint64", 8),
        # Floating point types
        ("float16", 2),
        ("float32", 4),
        ("float64", 8),
        # Boolean
        ("bool_", 1),
        # Date and timestamp types
        ("date32", 4),
        ("date64", 8),
        ("timestamp", 8),
        # Time types
        ("time32", 4),
        ("time64", 8),
        # Decimal types
        ("decimal128", 16),
        ("decimal256", 32),
    ],
)
def test_arrow_dtype_itemsize_fixed_width(type_name, expected_size):
    # [FUNCTION DOC]
    # GH 57948. Parametrized over fixed-width arrow type names with expected byte sizes (integers, floats, boolean, date, timestamp, time, decimal types). The dtype's item size equals the expected size (for boolean it is the numpy dtype item size), with a diagnostic message on failure.
    # GH 57948

    parametric_type_map = {
        "timestamp": pa.timestamp("ns"),
        "time32": pa.time32("s"),
        "time64": pa.time64("ns"),
        "decimal128": pa.decimal128(38, 10),
        "decimal256": pa.decimal256(76, 10),
    }

    if type_name in parametric_type_map:
        arrow_type = parametric_type_map.get(type_name)
    else:
        arrow_type = getattr(pa, type_name)()
    dtype = ArrowDtype(arrow_type)

    if type_name == "bool_":
        expected_size = dtype.numpy_dtype.itemsize

    assert dtype.itemsize == expected_size, (
        f"{type_name} expected {expected_size}, got {dtype.itemsize} "
        f"(bit_width={getattr(dtype.pyarrow_dtype, 'bit_width', 'N/A')})"
    )


@pytest.mark.parametrize("type_name", ["string", "binary", "large_string"])
def test_arrow_dtype_itemsize_variable_width(type_name):
    # [FUNCTION DOC]
    # GH 57948. Parametrized over string, binary and large string. The dtype's item size equals the numpy dtype's item size.
    # GH 57948

    arrow_type = getattr(pa, type_name)()
    dtype = ArrowDtype(arrow_type)

    assert dtype.itemsize == dtype.numpy_dtype.itemsize


def test_cast_pontwise_result_decimal_nan():
    # [FUNCTION DOC]
    # GH#62522. The array's private cast-pointwise-result applied to a list holding a decimal NaN gives an arrow decimal type (not a null type). The function name contains a spelling slip (pontwise).
    # GH#62522 we don't want to get back null[pyarrow] here
    ser = pd.Series([], dtype="float64[pyarrow]")
    arr = ser.array
    item = Decimal("NaN")

    result = arr._cast_pointwise_result([item])

    pa_type = result.dtype.pyarrow_dtype
    assert pa.types.is_decimal(pa_type)


def test_ufunc_retains_missing():
    # [FUNCTION DOC]
    # GH#62800. Applying a numpy sine function to an arrow float Series with a missing value gives the sine of the value and NA, retaining the dtype.
    # GH#62800
    ser = pd.Series([0.1, pd.NA], dtype="float64[pyarrow]")

    result = np.sin(ser)

    expected = pd.Series([np.sin(0.1), pd.NA], dtype="float64[pyarrow]")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("method", ["sum", "min", "max", "mean", "median"])
def test_duration_reduction_consistency(unit, method):
    # [FUNCTION DOC]
    # GH#63170. The unit fixture parametrizes durations; parametrized over sum, min, max, mean, median. Reductions of a duration arrow Series return a Timedelta whose unit equals the dtype's unit.
    # GH#63170
    dtype = f"duration[{unit}][pyarrow]"
    ser = pd.Series([timedelta(seconds=1), timedelta(seconds=2)], dtype=dtype)
    result = getattr(ser, method)()
    assert isinstance(result, pd.Timedelta), (
        f"{method} for {unit} returned {type(result)}"
    )
    assert result.unit == unit


@pytest.mark.parametrize("method", ["min", "max", "median"])
def test_timestamp_reduction_consistency(unit, method):
    # [FUNCTION DOC]
    # GH#63170. The unit fixture parametrizes timestamps; parametrized over min, max, median. Reductions of a timestamp arrow Series return a Timestamp whose unit equals the dtype's unit.
    # GH#63170
    dtype = f"timestamp[{unit}][pyarrow]"
    ser = pd.Series([datetime(2024, 1, 1), datetime(2024, 1, 3)], dtype=dtype)
    result = getattr(ser, method)()
    assert isinstance(result, pd.Timestamp), (
        f"{method} for {unit} returned {type(result)}"
    )
    assert result.unit == unit
