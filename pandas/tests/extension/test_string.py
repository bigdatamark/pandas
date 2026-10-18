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
# Compliance tests for the string extension dtype and arrays (python and pyarrow storage, with either NA
# or NaN as the missing value), run through the shared extension-array test suite. Fixtures build arrays optionally split
# into two pyarrow chunks. The test class overrides dtype name handling, expected exceptions, reductions and accumulations,
# fill and view tests, and a 2D-compat class skips pyarrow. Two module-level tests check that searching for a value in
# an array with missing values raises. Depends on the string-dtype argument fixture (storage, missing value) and on a
# helper from the string arithmetic tests. No import-time side effects.

from __future__ import annotations

import string
from typing import cast

import numpy as np
import pytest

from pandas.compat import HAS_PYARROW

from pandas.core.dtypes.base import StorageExtensionDtype

import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_string_dtype
from pandas.core.arrays import ArrowStringArray
from pandas.core.arrays.string_ import StringDtype
from pandas.tests.arithmetic.test_string import string_dtype_highest_priority
from pandas.tests.extension import base


def maybe_split_array(arr, chunked):
    # [FUNCTION DOC]
    # HELPER: optionally splits an array into two chunks. Parameters: arr (string array) and chunked (flag). If chunked is false,
    # or the array's storage is not pyarrow, returns the array unchanged. Otherwise (skipping the test if pyarrow is missing) splits the
    # underlying arrow data in half by length, builds a chunked array out of the chunks of both halves, asserts it has two chunks and
    # returns a new string array built from it.
    if not chunked:
        return arr
    elif arr.dtype.storage != "pyarrow":
        return arr

    pa = pytest.importorskip("pyarrow")

    arrow_array = arr._pa_array
    split = len(arrow_array) // 2
    arrow_array = pa.chunked_array(
        [*arrow_array[:split].chunks, *arrow_array[split:].chunks]
    )
    assert arrow_array.num_chunks == 2
    return arr._from_pyarrow_array(arrow_array)


@pytest.fixture(params=[True, False])
def chunked(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over true and false telling data fixtures whether to split into chunks.
    return request.param


@pytest.fixture
def dtype(string_dtype_arguments):
    # [FUNCTION DOC]
    # HELPER: fixture building the string dtype from the (storage, missing value) pair supplied by the string-dtype-arguments fixture.
    storage, na_value = string_dtype_arguments
    return StringDtype(storage=storage, na_value=na_value)


@pytest.fixture
def data(dtype, chunked):
    # [FUNCTION DOC]
    # HELPER: fixture returning ten random ASCII letters as a string array of the dtype (chunked per the fixture), ensuring the first two differ.
    # KNOWN ISSUE: the retry loop re-creates the generator with the same seed, so a first failing draw would repeat and never terminate.
    strings = np.random.default_rng(2).choice(list(string.ascii_letters), size=10)
    while strings[0] == strings[1]:
        strings = np.random.default_rng(2).choice(list(string.ascii_letters), size=10)

    arr = dtype.construct_array_type()._from_sequence(strings, dtype=dtype)
    return maybe_split_array(arr, chunked)


@pytest.fixture
def data_missing(dtype, chunked):
    # [FUNCTION DOC]
    # HELPER: fixture returning a two-element string array: NA then A (optionally chunked).
    """Length 2 array with [NA, Valid]"""
    arr = dtype.construct_array_type()._from_sequence([pd.NA, "A"], dtype=dtype)
    return maybe_split_array(arr, chunked)


@pytest.fixture
def data_for_sorting(dtype, chunked):
    # [FUNCTION DOC]
    # HELPER: fixture returning the string array B, C, A for sorting (optionally chunked).
    arr = dtype.construct_array_type()._from_sequence(["B", "C", "A"], dtype=dtype)
    return maybe_split_array(arr, chunked)


@pytest.fixture
def data_missing_for_sorting(dtype, chunked):
    # [FUNCTION DOC]
    # HELPER: fixture returning the string array B, NA, A for sorting with missing values (optionally chunked).
    arr = dtype.construct_array_type()._from_sequence(["B", pd.NA, "A"], dtype=dtype)
    return maybe_split_array(arr, chunked)


@pytest.fixture
def data_for_grouping(dtype, chunked):
    # [FUNCTION DOC]
    # HELPER: fixture returning the eight-element string array B, B, NA, NA, A, A, B, C for group-by tests (optionally chunked).
    arr = dtype.construct_array_type()._from_sequence(
        ["B", "B", pd.NA, pd.NA, "A", "A", "B", "C"], dtype=dtype
    )
    return maybe_split_array(arr, chunked)


class TestStringArray(base.ExtensionTests):
    # [CLASS DOC]
    # Extension-suite test class for the string array. Overrides dtype naming checks, view, transpose and views tests (skipped for pyarrow, where 2D is not implemented), fill
    # tests, expected exceptions for arithmetic, supported reductions and accumulations, the pointwise-cast helper and a few arithmetic and indexing tests.
    def test_combine_le(self, data_repeated):
        # [FUNCTION DOC]
        # Sets the instance expected dtype for the combine less-or-equal test to the pyarrow boolean for pyarrow storage with NA missing value, else plain bool, then returns the parent test result.
        dtype = next(iter(data_repeated(2))).dtype
        if dtype.storage == "pyarrow" and dtype.na_value is pd.NA:
            self._combine_le_expected_dtype = "bool[pyarrow]"
        else:
            self._combine_le_expected_dtype = "bool"
        return super().test_combine_le(data_repeated)

    def test_eq_with_str(self, dtype):
        # [FUNCTION DOC]
        # Runs the parent test, then checks the dtype equals the parametrized string alias with storage when the missing value is NA (only that variant supports it), or equals the plain str alias for pyarrow
        # storage with NaN missing value.
        super().test_eq_with_str(dtype)

        if dtype.na_value is pd.NA:
            # only the NA-variant supports parametrized string alias
            assert dtype == f"string[{dtype.storage}]"
        elif dtype.storage == "pyarrow":
            assert dtype == "str"

    def test_is_not_string_type(self, dtype):
        # [FUNCTION DOC]
        # Different from the base test: a string dtype is a string type, so asserts it is reported as a string dtype.
        # Different from BaseDtypeTests.test_is_not_string_type
        # because StringDtype is a string type
        assert is_string_dtype(dtype)

    def test_is_dtype_from_name(self, dtype, using_infer_string):
        # [FUNCTION DOC]
        # For a NaN-missing dtype when string inference is off, asserts that is-dtype-from-name on the dtype's name is false; otherwise delegates to the parent test.
        if dtype.na_value is np.nan and not using_infer_string:
            result = type(dtype).is_dtype(dtype.name)
            assert result is False
        else:
            super().test_is_dtype_from_name(dtype)

    def test_construct_from_string_own_name(self, dtype, using_infer_string):
        # [FUNCTION DOC]
        # For a NaN-missing dtype when string inference is off, constructing from the dtype's name raises a type error saying it cannot construct a string dtype; otherwise delegates to the parent test.
        if dtype.na_value is np.nan and not using_infer_string:
            with pytest.raises(TypeError, match="Cannot construct a 'StringDtype'"):
                dtype.construct_from_string(dtype.name)
        else:
            super().test_construct_from_string_own_name(dtype)

    def test_view(self, data):
        # [FUNCTION DOC]
        # Skips for pyarrow storage (2D support not implemented for the arrow string array); otherwise delegates to the parent view test.
        if data.dtype.storage == "pyarrow":
            pytest.skip(reason="2D support not implemented for ArrowStringArray")
        super().test_view(data)

    def test_from_dtype(self, data):
        # [FUNCTION DOC]
        # Overrides the parent test with an empty body (the base test uses the string representation of the dtype).
        # base test uses string representation of dtype
        pass

    def test_transpose(self, data):
        # [FUNCTION DOC]
        # Skips for pyarrow storage (2D support not implemented); otherwise delegates to the parent test.
        if data.dtype.storage == "pyarrow":
            pytest.skip(reason="2D support not implemented for ArrowStringArray")
        super().test_transpose(data)

    def test_setitem_preserves_views(self, data):
        # [FUNCTION DOC]
        # Skips for pyarrow storage (2D support not implemented); otherwise delegates to the parent test.
        if data.dtype.storage == "pyarrow":
            pytest.skip(reason="2D support not implemented for ArrowStringArray")
        super().test_setitem_preserves_views(data)

    def test_dropna_array(self, data_missing):
        # [FUNCTION DOC]
        # Verifies dropping missing values from the two-element array gives the array with only the valid element.
        result = data_missing.dropna()
        expected = data_missing[[1]]
        tm.assert_extension_array_equal(result, expected)

    def test_fillna_no_op_returns_copy(self, data):
        # [FUNCTION DOC]
        # Verifies filling a missing-free copy of the data with its first valid value returns a different object that is equal to the data.
        data = data[~data.isna()]

        valid = data[0]
        result = data.fillna(valid)
        assert result is not data
        tm.assert_extension_array_equal(result, data)

    def test_fillna_readonly(self, data_missing):
        # [FUNCTION DOC]
        # Makes a read-only copy of the missing-data array. Filling with the valid value (copy defaults to true) returns a result with the valid first element and leaves the original unchanged.
        # For pyarrow storage a second fill also works and the result is valid; for python storage filling with the no-copy option raises a value error about modifying a read-only array. The original stays unchanged.
        # KNOWN ISSUE: the comment says the no-copy option is not honored for arrow arrays, but the pyarrow branch repeats the default call without passing the no-copy option.
        data = data_missing.copy()
        data._readonly = True

        # by default fillna(copy=True), then this works fine
        result = data.fillna(data_missing[1])
        assert result[0] == data_missing[1]
        tm.assert_extension_array_equal(data, data_missing)

        # fillna(copy=False) is generally not honored by Arrow-backed array,
        # but always returns new data -> same result as above
        if data.dtype.storage == "pyarrow":
            result = data.fillna(data_missing[1])
            assert result[0] == data_missing[1]
        else:
            with pytest.raises(ValueError, match="Cannot modify read-only array"):
                data.fillna(data_missing[1], copy=False)
        tm.assert_extension_array_equal(data, data_missing)

    def _get_expected_exception(
        self, op_name: str, obj, other
    ) -> type[Exception] | tuple[type[Exception], ...] | None:
        # [FUNCTION DOC]
        # HELPER: the expected exception for an operation. Modulus, divmod, power, multiplication, true division, floor division and subtraction (including reflected forms)
        # all expect a type error (strings can only be multiplied by integers); other operations expect none. Parameters: op_name, obj, other.
        if op_name in [
            "__mod__",
            "__rmod__",
            "__divmod__",
            "__rdivmod__",
            "__pow__",
            "__rpow__",
        ]:
            return TypeError
        elif op_name in ["__mul__", "__rmul__"]:
            # Can only multiply strings by integers
            return TypeError
        elif op_name in [
            "__truediv__",
            "__rtruediv__",
            "__floordiv__",
            "__rfloordiv__",
            "__sub__",
            "__rsub__",
        ]:
            return TypeError

        return None

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        # [FUNCTION DOC]
        # HELPER: reduction support: min, max, sum; plus any and all only when the dtype's missing value is NaN.
        return op_name in ["min", "max", "sum"] or (
            ser.dtype.na_value is np.nan  # type: ignore[union-attr]
            and op_name in ("any", "all")
        )

    def _supports_accumulation(self, ser: pd.Series, op_name: str) -> bool:
        # [FUNCTION DOC]
        # HELPER: accumulation support: cummin, cummax, cumsum. Asserts the series dtype is a storage-aware extension dtype.
        assert isinstance(ser.dtype, StorageExtensionDtype)
        return op_name in ["cummin", "cummax", "cumsum"]

    def _cast_pointwise_result(self, op_name: str, obj, other, pointwise_result):
        # [FUNCTION DOC]
        # HELPER: casts the result of the pointwise combine path to the expected dtype. For addition and reflected addition: the left dtype, or the higher priority string dtype if the other operand is also a string-dtype
        # (not a plain str). Otherwise comparisons: numpy bool when the missing value is NaN, pyarrow boolean for pyarrow storage, else nullable boolean. Returns the result cast.
        dtype = cast(StringDtype, tm.get_dtype(obj))
        if op_name in ["__add__", "__radd__"]:
            cast_to = dtype
            dtype_other = tm.get_dtype(other) if not isinstance(other, str) else None
            if isinstance(dtype_other, StringDtype):
                cast_to = string_dtype_highest_priority(dtype, dtype_other)
        elif dtype.na_value is np.nan:
            cast_to = np.bool_  # type: ignore[assignment]
        elif dtype.storage == "pyarrow":
            cast_to = "bool[pyarrow]"  # type: ignore[assignment]
        else:
            cast_to = "boolean"  # type: ignore[assignment]
        return pointwise_result.astype(cast_to)

    def test_compare_scalar(self, data, comparison_op):
        # [FUNCTION DOC]
        # Compares the series from the data with the string abc using the comparison operator fixture, through the comparison helper.
        ser = pd.Series(data)
        self._compare_other(ser, data, comparison_op, "abc")

    def test_groupby_extension_apply(self, data_for_grouping, groupby_apply_op):
        # [FUNCTION DOC]
        # Delegates to the parent group-by extension apply test.
        super().test_groupby_extension_apply(data_for_grouping, groupby_apply_op)

    def test_combine_add(self, data_repeated, using_infer_string, request):
        # [FUNCTION DOC]
        # Marks expected-to-fail when string inference is off and storage is python (the pointwise result would be inferred as the NaN-valued pyarrow string, which differs from the input dtype), then delegates to the parent test.
        dtype = next(data_repeated(1)).dtype
        if not using_infer_string and dtype.storage == "python":
            mark = pytest.mark.xfail(
                reason="The pointwise operation result will be inferred to "
                "string[nan, pyarrow], which does not match the input dtype"
            )
            request.applymarker(mark)
        super().test_combine_add(data_repeated)

    def test_arith_series_with_array(
        self, data, all_arithmetic_operators, using_infer_string, request
    ):
        # [FUNCTION DOC]
        # Marks expected-to-fail for reflected addition when string inference is on, the missing value is NA and pyarrow is available or storage is pyarrow (same inference mismatch; a TODO for infer_string), then delegates to the parent test.
        dtype = data.dtype
        if (
            using_infer_string
            and all_arithmetic_operators == "__radd__"
            and dtype.na_value is pd.NA
            and (HAS_PYARROW or dtype.storage == "pyarrow")
        ):
            # TODO(infer_string)
            mark = pytest.mark.xfail(
                reason="The pointwise operation result will be inferred to "
                "string[nan, pyarrow], which does not match the input dtype"
            )
            request.applymarker(mark)
        super().test_arith_series_with_array(data, all_arithmetic_operators)

    def test_loc_setitem_with_expansion_preserves_ea_index_dtype(
        self, data, request, using_infer_string
    ):
        # [FUNCTION DOC]
        # Marks expected-to-fail (casts to object) when inference is off and storage is python; delegates to the parent test.
        if not using_infer_string and data.dtype.storage == "python":
            mark = pytest.mark.xfail(reason="Casts to object")
            request.applymarker(mark)
        super().test_loc_setitem_with_expansion_preserves_ea_index_dtype(data)


class Test2DCompat(base.Dim2CompatTests):
    # [CLASS DOC]
    # Runs the shared 2D compatibility tests for the string array.
    @pytest.fixture(autouse=True)
    def arrow_not_supported(self, data):
        # [FUNCTION DOC]
        # HELPER: autouse fixture that skips every test when the data is an arrow-backed string array (2D support not implemented).
        if isinstance(data, ArrowStringArray):
            pytest.skip(reason="2D support not implemented for ArrowStringArray")


@pytest.mark.parametrize("na_position", ["first", "last"])
def test_searchsorted_with_na_raises(data_for_sorting, as_series, na_position):
    # [FUNCTION DOC]
    # GH50447, GH65837. Parametrized over the NA position (first or last) and Series or array input. Sorts the three values into ascending order, places NA at the start or end,
    # and verifies searching for the middle value raises a value error saying the array must be sorted, which is impossible with NAs present.
    # GH50447, GH65837
    b, c, a = data_for_sorting
    arr = data_for_sorting.take([2, 0, 1])  # to get [a, b, c]
    if na_position == "last":
        arr[-1] = pd.NA
    else:
        arr[0] = pd.NA

    if as_series:
        arr = pd.Series(arr)

    msg = (
        "searchsorted requires array to be sorted, "
        "which is impossible with NAs present."
    )
    with pytest.raises(ValueError, match=msg):
        arr.searchsorted(b)


@pytest.mark.parametrize("na_position", ["first", "last"])
def test_searchsorted_with_na_and_sorter_raises(data_for_sorting, na_position):
    # [FUNCTION DOC]
    # GH65837. Parametrized over the NA position. Puts NA in the middle of a reordered array and passes a sorter permutation that places it first or last; verifies the search still raises the same value error about NAs.
    # GH65837
    b, c, a = data_for_sorting
    arr = data_for_sorting.take([0, 2, 1])
    arr[1] = pd.NA

    if na_position == "last":
        sorter = np.array([0, 2, 1])
    else:
        sorter = np.array([1, 0, 2])

    msg = (
        "searchsorted requires array to be sorted, "
        "which is impossible with NAs present."
    )
    with pytest.raises(ValueError, match=msg):
        arr.searchsorted(b, sorter=sorter)
