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

Note: we do not bother with base.BaseIndexTests because NumpyExtensionArray
will never be held in an Index.
"""
# [MODULE DOC]
# Compliance tests for the NumpyExtensionArray (wrapper of a numpy array as an extension array) with
# float and object (nested tuple) dtypes, run through the shared extension-array test suite. Defines a
# patched attribute-equality assertion, a fixture that lets the wrapper be used inside pandas
# constructors, data fixtures, a skip marker for nested-data cases and a test class that skips or expects
# failures for tests unsuited to nested data. Import-time side effect: stores the original attribute-equality
# assertion function in a module variable and defines a reusable marker applying the skip fixture.

import numpy as np
import pytest

from pandas.core.dtypes.dtypes import NumpyEADtype

import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_object_dtype
from pandas.core.arrays.numpy_ import NumpyExtensionArray
from pandas.tests.extension import base

orig_assert_attr_equal = tm.assert_attr_equal


def _assert_attr_equal(attr: str, left, right, obj: str = "Attributes"):
    # [FUNCTION DOC]
    # HELPER: replacement for the testing module's attribute-equality assertion (used while the allow fixture
    # is active). Parameters: attr name, left and right objects, obj label (default Attributes). For the dtype
    # attribute, if exactly one side has a numpy extension dtype, casts that side's object to the underlying
    # numpy dtype so an object extension dtype counts as close to the numpy object dtype. Then calls the
    # original assertion with the same arguments (raising an assertion error on mismatch).
    """
    patch tm.assert_attr_equal so NumpyEADtype("object") is closed enough to
    np.dtype("object")
    """
    if attr == "dtype":
        lattr = getattr(left, "dtype", None)
        rattr = getattr(right, "dtype", None)
        if isinstance(lattr, NumpyEADtype) and not isinstance(rattr, NumpyEADtype):
            left = left.astype(lattr.numpy_dtype)
        elif isinstance(rattr, NumpyEADtype) and not isinstance(lattr, NumpyEADtype):
            right = right.astype(rattr.numpy_dtype)

    orig_assert_attr_equal(attr, left, right, obj)


@pytest.fixture(params=["float", "object"])
def dtype(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over float and object; returns the numpy extension dtype wrapping that numpy dtype.
    return NumpyEADtype(np.dtype(request.param))


@pytest.fixture
def allow_in_pandas(monkeypatch):
    # [FUNCTION DOC]
    # HELPER: fixture that tells pandas to accept the wrapper in constructors. Within a monkeypatch context
    # it changes the wrapper's type tag to extension (to evade the abstract-class check that normally unboxes
    # it to a plain numpy array) and replaces the testing module's attribute-equality assertion with the patched
    # version; yields, and the patches are undone at teardown.
    """
    A monkeypatch to tells pandas to let us in.

    By default, passing a NumpyExtensionArray to an index / series / frame
    constructor will unbox that NumpyExtensionArray to an ndarray, and treat
    it as a non-EA column. We don't want people using EAs without
    reason.

    The mechanism for this is a check against ABCNumpyExtensionArray
    in each constructor.

    But, for testing, we need to allow them in pandas. So we patch
    the _typ of NumpyExtensionArray, so that we evade the ABCNumpyExtensionArray
    check.
    """
    with monkeypatch.context() as m:
        m.setattr(NumpyExtensionArray, "_typ", "extension")
        m.setattr(tm.asserters, "assert_attr_equal", _assert_attr_equal)
        yield


@pytest.fixture
def data(allow_in_pandas, dtype):
    # [FUNCTION DOC]
    # HELPER: fixture returning a wrapper array of ten values: tuples of one integer for object dtype (taken from a
    # Series' internal values) or the integers one through ten in the float dtype.
    if dtype.numpy_dtype == "object":
        arr = pd.Series([(i,) for i in range(10)])._values
    else:
        arr = np.arange(1, 11, dtype=dtype._dtype)
    return NumpyExtensionArray(arr)


@pytest.fixture
def data_missing(allow_in_pandas, dtype):
    # [FUNCTION DOC]
    # HELPER: fixture returning a two-element wrapper array: NaN then a one-tuple (object) or NaN then 1.0 (float).
    if dtype.numpy_dtype == "object":
        return NumpyExtensionArray(np.array([np.nan, (1,)], dtype=object))
    return NumpyExtensionArray(np.array([np.nan, 1.0]))


@pytest.fixture
def na_cmp():
    # [FUNCTION DOC]
    # HELPER: fixture returning a comparison function for missing values.
    def cmp(a, b):
        # [FUNCTION DOC]
        # HELPER: returns true only if both arguments are NaN.
        return np.isnan(a) and np.isnan(b)

    return cmp


@pytest.fixture
def data_for_sorting(allow_in_pandas, dtype):
    # [FUNCTION DOC]
    # HELPER: fixture returning a three-element wrapper array with known sort order. Object: tuples (2,), (3,), (1,)
    # built through an empty-tuple leading element that is then sliced off, to disable numpy shape inference;
    # otherwise the numbers one, two, zero.
    """Length-3 array with a known sort order.

    This should be three items [B, C, A] with
    A < B < C
    """
    if dtype.numpy_dtype == "object":
        # Use an empty tuple for first element, then remove,
        # to disable np.array's shape inference.
        return NumpyExtensionArray(np.array([(), (2,), (3,), (1,)], dtype=object)[1:])
    return NumpyExtensionArray(np.array([1, 2, 0]))


@pytest.fixture
def data_missing_for_sorting(allow_in_pandas, dtype):
    # [FUNCTION DOC]
    # HELPER: fixture returning a three-element wrapper array [B, NA, A] with A below B: object (1,), NaN, (0,);
    # otherwise 1, NaN, 0.
    """Length-3 array with a known sort order.

    This should be three items [B, NA, A] with
    A < B and NA missing.
    """
    if dtype.numpy_dtype == "object":
        return NumpyExtensionArray(np.array([(1,), np.nan, (0,)], dtype=object))
    return NumpyExtensionArray(np.array([1, np.nan, 0]))


@pytest.fixture
def data_for_grouping(allow_in_pandas, dtype):
    # [FUNCTION DOC]
    # HELPER: fixture returning an eight-element wrapper array [B, B, NA, NA, A, A, B, C] with A, B, C being
    # one-tuples one to three (object) or the integers zero to two, NaN for missing, in the dtype's numpy type.
    """Data for factorization, grouping, and unique tests.

    Expected to be like [B, B, NA, NA, A, A, B, C]

    Where A < B < C and NA is missing
    """
    if dtype.numpy_dtype == "object":
        a, b, c = (1,), (2,), (3,)
    else:
        a, b, c = np.arange(3)
    return NumpyExtensionArray(
        np.array([b, b, np.nan, np.nan, a, a, b, c], dtype=dtype.numpy_dtype)
    )


@pytest.fixture
def data_for_twos(dtype):
    # [FUNCTION DOC]
    # HELPER: fixture returning ten twos built through the sequence constructor with the dtype. Skips the test for
    # object dtype because it is not numeric.
    if dtype.kind == "O":
        pytest.skip(f"{dtype} is not a numeric dtype")
    arr = np.ones(10) * 2
    return NumpyExtensionArray._from_sequence(arr, dtype=dtype)


@pytest.fixture
def skip_numpy_object(dtype, request):
    # [FUNCTION DOC]
    # HELPER: fixture used via a marker for nested-data cases. When the dtype is the object dtype it applies an
    # expected-failure marker (reason: fails for object dtype) to the requesting test.
    # KNOWN ISSUE: the docstring says the tests are skipped but the code marks them expected to fail.
    """
    Tests for NumpyExtensionArray with nested data. Users typically won't create
    these objects via `pd.array`, but they can show up through `.array`
    on a Series with nested data. Many of the base tests fail, as they aren't
    appropriate for nested data.

    This fixture allows these tests to be skipped when used as a usefixtures
    marker to either an individual test or a test class.
    """
    if dtype == "object":
        mark = pytest.mark.xfail(reason="Fails for object dtype")
        request.applymarker(mark)


skip_nested = pytest.mark.usefixtures("skip_numpy_object")


class TestNumpyExtensionArray(base.ExtensionTests):
    # [CLASS DOC]
    # Extension-suite test class for the numpy extension array. Class attributes for expected exceptions
    # (divmod, series-scalar, frame-scalar, series-array) are none and set per test on the instance. Overrides
    # many tests that are inappropriate for nested data with skip-nested markers or expected failures.
    @pytest.mark.skip(reason="We don't register our dtype")
    # We don't want to register. This test should probably be split in two.
    def test_from_dtype(self, data):
        # [FUNCTION DOC]
        # Skipped because the dtype is not registered; the body does nothing. A comment says the test should
        # probably be split in two.
        pass

    @skip_nested
    def test_series_constructor_scalar_with_index(self, data, dtype):
        # [FUNCTION DOC]
        # Skip-nested (expected failure for object). Delegates to the parent test; the source notes the object case
        # fails with a length-mismatch value error.
        # ValueError: Length of passed values is 1, index implies 3.
        super().test_series_constructor_scalar_with_index(data, dtype)

    def test_check_dtype(self, data, request, using_infer_string):
        # [FUNCTION DOC]
        # Marks the test as expected-to-fail for object dtype (the wrapper clashes with a numpy name), then
        # delegates to the parent check-dtype test with the data.
        if data.dtype.numpy_dtype == "object":
            request.applymarker(
                pytest.mark.xfail(
                    reason=f"NumpyExtensionArray expectedly clashes with a "
                    f"NumPy name: {data.dtype.numpy_dtype}"
                )
            )
        super().test_check_dtype(data)

    def test_is_not_object_type(self, dtype, request):
        # [FUNCTION DOC]
        # For object dtype asserts the dtype is an object type (differs from the base test because the wrapped
        # object dtype is an object type); otherwise delegates to the parent test.
        if dtype.numpy_dtype == "object":
            # Different from BaseDtypeTests.test_is_not_object_type
            # because NumpyEADtype(object) is an object type
            assert is_object_dtype(dtype)
        else:
            super().test_is_not_object_type(dtype)

    @skip_nested
    def test_getitem_scalar(self, data):
        # [FUNCTION DOC]
        # Skip-nested; delegates to the parent scalar get test (source notes an assertion error for nested data).
        # AssertionError
        super().test_getitem_scalar(data)

    @skip_nested
    def test_shift_fill_value(self, data):
        # [FUNCTION DOC]
        # Skip-nested; delegates to the parent shift with fill value test (numpy shape inference breaks shifting).
        # np.array shape inference. Shift implementation fails.
        super().test_shift_fill_value(data)

    @skip_nested
    def test_fillna_limit_frame(self, data_missing):
        # [FUNCTION DOC]
        # Skip-nested; GH#58001: the scalar for this array is not a scalar. Delegates to the parent test.
        # GH#58001
        # The "scalar" for this array isn't a scalar.
        super().test_fillna_limit_frame(data_missing)

    @skip_nested
    def test_fillna_limit_series(self, data_missing):
        # [FUNCTION DOC]
        # Skip-nested; GH#58001: the scalar is not a scalar. Delegates to the parent test.
        # GH#58001
        # The "scalar" for this array isn't a scalar.
        super().test_fillna_limit_series(data_missing)

    @skip_nested
    def test_fillna_copy_frame(self, data_missing):
        # [FUNCTION DOC]
        # Skip-nested; the scalar is not a scalar. Delegates to the parent fill-copy frame test.
        # The "scalar" for this array isn't a scalar.
        super().test_fillna_copy_frame(data_missing)

    @skip_nested
    def test_fillna_copy_series(self, data_missing):
        # [FUNCTION DOC]
        # Skip-nested; the scalar is not a scalar. Delegates to the parent fill-copy series test.
        # The "scalar" for this array isn't a scalar.
        super().test_fillna_copy_series(data_missing)

    @skip_nested
    def test_searchsorted(self, data_for_sorting, as_series):
        # [FUNCTION DOC]
        # Skip-nested; delegates to the parent sorted-search test. A TODO says the array's search calls numpy's, which
        # is not right for nested data; something like an internal binary search is needed.
        # TODO: NumpyExtensionArray.searchsorted calls ndarray.searchsorted which
        #  isn't quite what we want in nested data cases. Instead we need to
        #  adapt something like libindex._bin_search.
        super().test_searchsorted(data_for_sorting, as_series)

    @pytest.mark.xfail(reason="NumpyExtensionArray.diff may fail on dtype")
    def test_diff(self, data, periods):
        # [FUNCTION DOC]
        # Expected-to-fail (diff may fail on dtype). Delegates to the parent diff test and returns its result.
        return super().test_diff(data, periods)

    def test_insert(self, data, request):
        # [FUNCTION DOC]
        # Marks the test expected-to-fail for object dtype (dimension mismatch when concatenating) and delegates to
        # the parent insert test.
        if data.dtype.numpy_dtype == object:
            mark = pytest.mark.xfail(reason="Dimension mismatch in np.concatenate")
            request.applymarker(mark)

        super().test_insert(data)

    @skip_nested
    def test_insert_invalid(self, data, invalid_scalar):
        # [FUNCTION DOC]
        # Skip-nested; the object array can hold anything so no invalid scalar exists. Delegates to the parent test.
        # NumpyExtensionArray[object] can hold anything, so skip
        super().test_insert_invalid(data, invalid_scalar)

    divmod_exc = None
    series_scalar_exc = None
    frame_scalar_exc = None
    series_array_exc = None

    def test_divmod(self, data):
        # [FUNCTION DOC]
        # Sets the instance divmod exception to type error for object-kind data (else none), then delegates to the parent divmod test.
        divmod_exc = None
        if data.dtype.kind == "O":
            divmod_exc = TypeError
        self.divmod_exc = divmod_exc
        super().test_divmod(data)

    def test_divmod_series_array(self, data):
        # [FUNCTION DOC]
        # Builds a Series of the data; for object-kind data sets the instance divmod exception to type error. Then runs the
        # divmod check helper with the Series, the divmod function and the data (does not call the parent test).
        ser = pd.Series(data)
        exc = None
        if data.dtype.kind == "O":
            exc = TypeError
            self.divmod_exc = exc
        self._check_divmod_op(ser, divmod, data)

    def test_arith_series_with_scalar(self, data, all_arithmetic_operators, request):
        # [FUNCTION DOC]
        # Sets the expected series-scalar exception to type error for object dtype, and for multiplication and reflected
        # multiplication marks the test expected-to-fail (combine raises but the Series method does not); stores it on the
        # instance and delegates to the parent test.
        opname = all_arithmetic_operators
        series_scalar_exc = None
        if data.dtype.numpy_dtype == object:
            if opname in ["__mul__", "__rmul__"]:
                mark = pytest.mark.xfail(
                    reason="the Series.combine step raises but not the Series method."
                )
                request.node.add_marker(mark)
            series_scalar_exc = TypeError
        self.series_scalar_exc = series_scalar_exc
        super().test_arith_series_with_scalar(data, all_arithmetic_operators)

    def test_arith_series_with_array(self, data, all_arithmetic_operators):
        # [FUNCTION DOC]
        # Sets the expected series-array exception to type error for object dtype except addition and reflected
        # addition; stores it on the instance and delegates to the parent test.
        opname = all_arithmetic_operators
        series_array_exc = None
        if data.dtype.numpy_dtype == object and opname not in ["__add__", "__radd__"]:
            series_array_exc = TypeError
        self.series_array_exc = series_array_exc
        super().test_arith_series_with_array(data, all_arithmetic_operators)

    def test_arith_frame_with_scalar(self, data, all_arithmetic_operators, request):
        # [FUNCTION DOC]
        # Sets the expected frame-scalar exception to type error for object dtype, with an expected failure for multiplication
        # and reflected multiplication (combine raises but the Series method does not); stores on the instance and delegates.
        opname = all_arithmetic_operators
        frame_scalar_exc = None
        if data.dtype.numpy_dtype == object:
            if opname in ["__mul__", "__rmul__"]:
                mark = pytest.mark.xfail(
                    reason="the Series.combine step raises but not the Series method."
                )
                request.node.add_marker(mark)
            frame_scalar_exc = TypeError
        self.frame_scalar_exc = frame_scalar_exc
        super().test_arith_frame_with_scalar(data, all_arithmetic_operators)

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        # [FUNCTION DOC]
        # HELPER: for object-kind arrays only sum, min, max, any and all are supported; everything else is supported.
        if ser.dtype.kind == "O":
            return op_name in ["sum", "min", "max", "any", "all"]
        return True

    def check_reduce(self, ser: pd.Series, op_name: str, skipna: bool):
        # [FUNCTION DOC]
        # HELPER: overrides the reduction check: casts the series to its actual numpy dtype (avoid turning integers into
        # floats) and compares the extension reduction with the numpy-backed one (count takes no skipna), using almost-equal.
        res_op = getattr(ser, op_name)
        # avoid coercing int -> float. Just cast to the actual numpy type.
        # error: Item "ExtensionDtype" of "dtype[Any] | ExtensionDtype" has
        # no attribute "numpy_dtype"
        cmp_dtype = ser.dtype.numpy_dtype  # type: ignore[union-attr]
        alt = ser.astype(cmp_dtype)
        exp_op = getattr(alt, op_name)
        if op_name == "count":
            result = res_op()
            expected = exp_op()
        else:
            result = res_op(skipna=skipna)
            expected = exp_op(skipna=skipna)
        tm.assert_almost_equal(result, expected)

    @pytest.mark.skip("TODO: tests not written yet")
    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_frame(self, data, all_numeric_reductions, skipna):
        # [FUNCTION DOC]
        # Skipped (tests not written yet), parametrized over skipna; body does nothing.
        pass

    @skip_nested
    def test_fillna_series(self, data_missing):
        # [FUNCTION DOC]
        # Skip-nested (non-scalar scalar values); delegates to the parent fill-series test.
        # Non-scalar "scalar" values.
        super().test_fillna_series(data_missing)

    @skip_nested
    def test_fillna_frame(self, data_missing):
        # [FUNCTION DOC]
        # Skip-nested (non-scalar scalar values); delegates to the parent fill-frame test.
        # Non-scalar "scalar" values.
        super().test_fillna_frame(data_missing)

    @skip_nested
    def test_fillna_readonly(self, data_missing):
        # [FUNCTION DOC]
        # Skip-nested (non-scalar scalar values); delegates to the parent read-only fill test.
        # Non-scalar "scalar" values.
        super().test_fillna_readonly(data_missing)

    @skip_nested
    def test_setitem_invalid(self, data, invalid_scalar):
        # [FUNCTION DOC]
        # Skip-nested; object dtype can hold anything so setting does not raise. Delegates to the parent test.
        # object dtype can hold anything, so doesn't raise
        super().test_setitem_invalid(data, invalid_scalar)

    @skip_nested
    def test_setitem_sequence_broadcasts(self, data, box_in_series):
        # [FUNCTION DOC]
        # Skip-nested; the source notes a value error about list-like indexer length mismatch. Delegates to the parent test.
        # ValueError: cannot set using a list-like indexer with a different
        # length than the value
        super().test_setitem_sequence_broadcasts(data, box_in_series)

    @skip_nested
    @pytest.mark.parametrize("setter", ["loc", None])
    def test_setitem_mask_broadcast(self, data, setter):
        # [FUNCTION DOC]
        # Skip-nested, parametrized over loc or default setter; same value error note. Delegates to the parent test.
        # ValueError: cannot set using a list-like indexer with a different
        # length than the value
        super().test_setitem_mask_broadcast(data, setter)

    @skip_nested
    def test_setitem_scalar_key_sequence_raise(self, data):
        # [FUNCTION DOC]
        # Skip-nested; the source notes the expected value error was not raised. Delegates to the parent test.
        # Failed: DID NOT RAISE <class 'ValueError'>
        super().test_setitem_scalar_key_sequence_raise(data)

    # TODO: there is some issue with NumpyExtensionArray, therefore,
    #   skip the setitem test for now, and fix it later (GH 31446)

    @skip_nested
    @pytest.mark.parametrize(
        "mask",
        [
            np.array([True, True, True, False, False]),
            pd.array([True, True, True, False, False], dtype="boolean"),
        ],
        ids=["numpy-array", "boolean-array"],
    )
    def test_setitem_mask(self, data, mask, box_in_series):
        # [FUNCTION DOC]
        # Skip-nested, parametrized over a numpy boolean mask and a nullable boolean mask (ids numpy-array and boolean-array).
        # Delegates to the parent mask-setting test. A TODO (GH 31446) says the wrapper has some issue so setitem tests are skipped.
        super().test_setitem_mask(data, mask, box_in_series)

    @skip_nested
    @pytest.mark.parametrize(
        "idx",
        [[0, 1, 2], pd.array([0, 1, 2], dtype="Int64"), np.array([0, 1, 2])],
        ids=["list", "integer-array", "numpy-array"],
    )
    def test_setitem_integer_array(self, data, idx, box_in_series):
        # [FUNCTION DOC]
        # Skip-nested, parametrized over a list, a nullable integer array and a numpy array of indices. Delegates to the parent test.
        super().test_setitem_integer_array(data, idx, box_in_series)

    @skip_nested
    def test_setitem_slice(self, data, box_in_series):
        # [FUNCTION DOC]
        # Skip-nested; delegates to the parent slice-setting test.
        super().test_setitem_slice(data, box_in_series)

    @skip_nested
    def test_setitem_loc_iloc_slice(self, data):
        # [FUNCTION DOC]
        # Skip-nested; delegates to the parent loc/iloc slice-setting test.
        super().test_setitem_loc_iloc_slice(data)

    def test_setitem_with_expansion_dataframe_column(self, data, full_indexer):
        # [FUNCTION DOC]
        # GH 32395. Enlarging an empty frame by label assignment of a column: the result is expected to equal the
        # frame itself for object dtype or full slice keys, otherwise the frame of the numpy-converted data (the wrapper behaves
        # oddly because its type tag is patched); column type is not checked.
        # https://github.com/pandas-dev/pandas/issues/32395
        df = expected = pd.DataFrame({"data": pd.Series(data)})
        result = pd.DataFrame(index=df.index)

        # because result has object dtype, the attempt to do setting inplace
        #  is successful, and object dtype is retained
        key = full_indexer(df)
        result.loc[key, "data"] = df["data"]

        # base class method has expected = df; NumpyExtensionArray behaves oddly because
        #  we patch _typ for these tests.
        if data.dtype.numpy_dtype != object:
            if not isinstance(key, slice) or key != slice(None):
                expected = pd.DataFrame({"data": data.to_numpy()})
        tm.assert_frame_equal(result, expected, check_column_type=False)

    @pytest.mark.xfail(reason="NumpyEADtype is unpacked")
    def test_index_from_listlike_with_dtype(self, data):
        # [FUNCTION DOC]
        # Expected-to-fail (the dtype is unpacked). Delegates to the parent test.
        super().test_index_from_listlike_with_dtype(data)

    @skip_nested
    @pytest.mark.parametrize("engine", ["c", "python"])
    def test_EA_types(self, engine, data, request):
        # [FUNCTION DOC]
        # Skip-nested, parametrized over the c and python parser engines; delegates to the parent test.
        super().test_EA_types(engine, data, request)

    def test_loc_setitem_with_expansion_preserves_ea_index_dtype(self, data, request):
        # [FUNCTION DOC]
        # Marks the test expected-to-fail if the last element is a tuple (the tuple is unpacked) then delegates to the parent
        # test of loc enlargement preserving the extension index dtype.
        if isinstance(data[-1], tuple):
            mark = pytest.mark.xfail(reason="Unpacks tuple")
            request.applymarker(mark)
        super().test_loc_setitem_with_expansion_preserves_ea_index_dtype(data)


class Test2DCompat(base.NDArrayBacked2DTests):
    # [CLASS DOC]
    # Runs the shared two-dimensional compatibility tests for array-backed extension arrays against the
    # wrapper without changes.
    pass
