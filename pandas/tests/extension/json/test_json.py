# [MODULE DOC]
# Test module applying the shared extension-array test suite to the JSON (mapping-valued) test
# array. The set-item base tests are intentionally not mixed in as a whole (the combined class is used, but
# many set-item tests are marked as expected failures because the internals struggle to set sequences
# into scalar positions). Contains: fixtures for the base suite (dtype, data, data_missing,
# data_for_sorting, data_missing_for_sorting, na_cmp, data_for_grouping), a module-level marker
# named unhashable that marks a test as an expected failure for unhashable values, the test class
# TestJSONArray with many overrides that mark known-unsupported behavior as expected failures,
# custom equality assertion helpers for series and frames (needed because numpy cannot build
# arrays from equal-length user mappings), and a test of those helpers. Depends on the JSON array
# module in the same package.
import collections
import operator
import sys

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.json.array import (
    JSONArray,
    JSONDtype,
    make_data,
)

# We intentionally don't run base.BaseSetitemTests because pandas'
# internals has trouble setting sequences of values into scalar positions.
unhashable = pytest.mark.xfail(reason="Unhashable")


@pytest.fixture
def dtype():
    # [FUNCTION DOC]
    # HELPER: fixture returning a JSON dtype instance.
    return JSONDtype()


@pytest.fixture
def data():
    # [FUNCTION DOC]
    # HELPER: fixture returning a length-ten JSON array. Draws random data repeatedly until the
    # first two elements have different lengths, because numpy cannot build an array from equal-length
    # sequences and many operations convert to an object array; several tests only look at the first
    # two elements.
    """Length-10 JSONArray for semantics test."""
    data = make_data(10)

    # Why the while loop? NumPy is unable to construct an ndarray from
    # equal-length ndarrays. Many of our operations involve coercing the
    # EA to an ndarray of objects. To avoid random test failures, we ensure
    # that our data is coercible to an ndarray. Several tests deal with only
    # the first two elements, so that's what we'll check.

    while len(data[0]) == len(data[1]):
        data = make_data(10)

    return JSONArray(data)


@pytest.fixture
def data_missing():
    # [FUNCTION DOC]
    # HELPER: fixture returning a two-element array: an empty mapping (missing) then a mapping with
    # key a and value 10.
    """Length 2 array with [NA, Valid]"""
    return JSONArray([{}, {"a": 10}])


@pytest.fixture
def data_for_sorting():
    # [FUNCTION DOC]
    # HELPER: fixture returning three mappings: key b=1; key c=4; keys a=2 and c=3 (the B, C, A
    # pattern).
    return JSONArray([{"b": 1}, {"c": 4}, {"a": 2, "c": 3}])


@pytest.fixture
def data_missing_for_sorting():
    # [FUNCTION DOC]
    # HELPER: fixture returning three elements: b=1, an empty mapping (missing), a=4.
    return JSONArray([{"b": 1}, {}, {"a": 4}])


@pytest.fixture
def na_cmp():
    # [FUNCTION DOC]
    # HELPER: fixture returning the equality operator as the missing-value comparison function.
    return operator.eq


@pytest.fixture
def data_for_grouping():
    # [FUNCTION DOC]
    # HELPER: fixture returning eight mappings in the B, B, NA, NA, A, A, B, C pattern: b=1, b=1, empty,
    # empty, (a=0 and c=2) twice, b=1, c=2.
    return JSONArray(
        [
            {"b": 1},
            {"b": 1},
            {},
            {},
            {"a": 0, "c": 2},
            {"a": 0, "c": 2},
            {"b": 1},
            {"c": 2},
        ]
    )


class TestJSONArray(base.ExtensionTests):
    # [CLASS DOC]
    # Runs the complete extension test suite for the JSON array via the combined test class.
    # Most methods are overrides that either mark the inherited test as an expected failure with a
    # reason (unsupported comparison, unhashable values, dicts treated as mappings, slice or sequence
    # assignment limits) or adjust expectations (unsupported limit area argument, fill with none).
    @pytest.mark.xfail(
        reason="comparison method not implemented for JSONArray (GH-37867)"
    )
    def test_contains(self, data):
        # [FUNCTION DOC]
        # Override: expected failure because the comparison method is not implemented for the array
        # (upstream issue). Takes only data and calls the base test.
        # GH-37867
        super().test_contains(data)

    @pytest.mark.xfail(reason="not implemented constructor from dtype")
    def test_from_dtype(self, data):
        # [FUNCTION DOC]
        # Override: expected failure because construction from the dtype is not implemented.
        # construct from our dtype & string dtype
        super().test_from_dtype(data)

    @pytest.mark.xfail(reason="RecursionError, GH-33900")
    def test_series_constructor_no_data_with_index(self, dtype, na_value):
        # [FUNCTION DOC]
        # Override: expected failure (infinite recursion, upstream issue). Temporarily lowers the
        # recursion limit to 100 to avoid stack overflow on Windows CI, runs the base test, and restores the
        # limit in a cleanup block. Mutates global interpreter state during the test.
        # RecursionError: maximum recursion depth exceeded in comparison
        rec_limit = sys.getrecursionlimit()
        try:
            # Limit to avoid stack overflow on Windows CI
            sys.setrecursionlimit(100)
            super().test_series_constructor_no_data_with_index(dtype, na_value)
        finally:
            sys.setrecursionlimit(rec_limit)

    @pytest.mark.xfail(reason="RecursionError, GH-33900")
    def test_series_constructor_scalar_na_with_index(self, dtype, na_value):
        # [FUNCTION DOC]
        # Override: expected failure (infinite recursion, upstream issue). Same recursion-limit handling as
        # the no-data test, wrapped around the base test.
        # RecursionError: maximum recursion depth exceeded in comparison
        rec_limit = sys.getrecursionlimit()
        try:
            # Limit to avoid stack overflow on Windows CI
            sys.setrecursionlimit(100)
            super().test_series_constructor_scalar_na_with_index(dtype, na_value)
        finally:
            sys.setrecursionlimit(rec_limit)

    @pytest.mark.xfail(reason="collection as scalar, GH-33901")
    def test_series_constructor_scalar_with_index(self, data, dtype):
        # [FUNCTION DOC]
        # Override: expected failure (a mapping scalar is treated as a collection, upstream issue).
        # Same recursion-limit handling around the base test.
        # TypeError: All values must be of type <class 'collections.abc.Mapping'>
        rec_limit = sys.getrecursionlimit()
        try:
            # Limit to avoid stack overflow on Windows CI
            sys.setrecursionlimit(100)
            super().test_series_constructor_scalar_with_index(data, dtype)
        finally:
            sys.setrecursionlimit(rec_limit)

    @pytest.mark.xfail(reason="Different definitions of NA")
    def test_stack(self):
        # [FUNCTION DOC]
        # Override: expected failure because the base test casts to object first and the array
        # considers an empty mapping to be missing while the object version does not, producing different
        # rows.
        # KNOWN ISSUE: this override takes no fixtures and calls the base test without the arguments
        # that it requires, so it would fail with an argument error rather than for the stated reason;
        # the expected-failure marker hides the discrepancy.
        """
        The test does .astype(object).stack(). If we happen to have
        any missing values in `data`, then we'll end up with different
        rows since we consider `{}` NA, but `.astype(object)` doesn't.
        """
        super().test_stack()

    @pytest.mark.xfail(reason="dict for NA")
    def test_unstack(self, data, index):
        # [FUNCTION DOC]
        # Override: expected failure because the array's missing value is an empty mapping whereas the
        # base test's expected frame uses NaN.
        # KNOWN ISSUE: the override takes data and index but omits the obj argument that the base
        # test requires, so the call to the base test fails for that reason; the expected-failure marker
        # hides it. It also returns the base result.
        # The base test has NaN for the expected NA value.
        # this matches otherwise
        return super().test_unstack(data, index)

    @pytest.mark.xfail(reason="Setting a dict as a scalar")
    def test_fillna_series(self):
        # [FUNCTION DOC]
        # Override: expected failure because a mapping fill value is treated as a mapping (fill by
        # label) rather than a scalar.
        # KNOWN ISSUE: calls the base test without its required fixture argument, which would fail
        # independently of the stated reason; hidden by the expected-failure marker.
        """We treat dictionaries as a mapping in fillna, not a scalar."""
        super().test_fillna_series()

    @pytest.mark.xfail(reason="Setting a dict as a scalar")
    def test_fillna_frame(self):
        # [FUNCTION DOC]
        # Override: expected failure because a mapping fill value is treated as a mapping rather than a
        # scalar.
        # KNOWN ISSUE: calls the base test without its required fixture argument, which would fail
        # independently of the stated reason; hidden by the expected-failure marker.
        """We treat dictionaries as a mapping in fillna, not a scalar."""
        super().test_fillna_frame()

    def test_fillna_with_none(self, data_missing):
        # [FUNCTION DOC]
        # Override. This array does not treat none as the missing value, so the base test must fail
        # with an assertion error (the test passes by expecting that failure).
        # GH#57723
        # EAs that don't have special logic for None will raise, unlike pandas'
        # which interpret None as the NA value for the dtype.
        with pytest.raises(AssertionError):
            super().test_fillna_with_none(data_missing)

    @pytest.mark.xfail(reason="fill value is a dictionary, takes incorrect code path")
    def test_fillna_limit_frame(self, data_missing):
        # [FUNCTION DOC]
        # Override: expected failure because the fill value is a mapping and takes the wrong code path.
        # GH#58001
        super().test_fillna_limit_frame(data_missing)

    @pytest.mark.xfail(reason="fill value is a dictionary, takes incorrect code path")
    def test_fillna_limit_series(self, data_missing):
        # [FUNCTION DOC]
        # Override: expected failure because the fill value is a mapping and takes the wrong code path.
        # KNOWN ISSUE: this series test calls the base class's frame version (the frame limit test)
        # instead of the series limit test.
        # GH#58001
        super().test_fillna_limit_frame(data_missing)

    @pytest.mark.parametrize(
        "limit_area, input_ilocs, expected_ilocs",
        [
            ("outside", [1, 0, 0, 0, 1], [1, 0, 0, 0, 1]),
            ("outside", [1, 0, 1, 0, 1], [1, 0, 1, 0, 1]),
            ("outside", [0, 1, 1, 1, 0], [0, 1, 1, 1, 1]),
            ("outside", [0, 1, 0, 1, 0], [0, 1, 0, 1, 1]),
            ("inside", [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]),
            ("inside", [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]),
            ("inside", [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]),
            ("inside", [0, 1, 0, 1, 0], [0, 1, 1, 1, 0]),
        ],
    )
    def test_ffill_limit_area(
        self, data_missing, limit_area, input_ilocs, expected_ilocs
    ):
        # [FUNCTION DOC]
        # Override. Parametrized the same as the base test. Because the array's fill hook has no limit
        # area argument, running the base test must raise a not-implemented error saying the array does
        # not implement limit area.
        # GH#56616
        msg = "JSONArray does not implement limit_area"
        with pytest.raises(NotImplementedError, match=msg):
            super().test_ffill_limit_area(
                data_missing, limit_area, input_ilocs, expected_ilocs
            )

    def test_value_counts(self, all_data, dropna, request):
        # [FUNCTION DOC]
        # Override. Takes dropna and request. Marks as expected failure (unhashable) when the data has ten
        # elements or dropna is true, then runs the base test.
        if len(all_data) == 10 or dropna:
            request.applymarker(unhashable)
        super().test_value_counts(all_data, dropna)

    @unhashable
    def test_sort_values_frame(self):
        # [FUNCTION DOC]
        # Override: marked unhashable expected failure; the comment suggests checking whether the
        # factorize values hook could make it work. Runs the base test.
        # KNOWN ISSUE: takes no fixtures but the base test requires the sorting data and ascending
        # arguments, so the call would raise an argument error; hidden by the expected-failure marker.
        # TODO (EA.factorize): see if _values_for_factorize allows this.
        super().test_sort_values_frame()

    @pytest.mark.xfail(reason="combine for JSONArray not supported")
    def test_combine_le(self, data_repeated):
        # [FUNCTION DOC]
        # Override: expected failure because combine is not supported for this array.
        super().test_combine_le(data_repeated)

    @pytest.mark.xfail(
        reason="combine for JSONArray not supported - "
        "may pass depending on random data",
        strict=False,
        raises=AssertionError,
    )
    def test_combine_first(self, data):
        # [FUNCTION DOC]
        # Override: non-strict expected failure (assertion error) because combine is not supported;
        # may pass depending on random data.
        super().test_combine_first(data)

    @pytest.mark.xfail(reason="broadcasting error")
    def test_where_series(self, data, na_value):
        # [FUNCTION DOC]
        # Override: expected failure because of a broadcasting error between arrays of mismatched
        # shapes.
        # Fails with
        # *** ValueError: operands could not be broadcast together
        # with shapes (4,) (4,) (0,)
        super().test_where_series(data, na_value)

    @pytest.mark.xfail(reason="Can't compare dicts.")
    def test_searchsorted(self, data_for_sorting):
        # [FUNCTION DOC]
        # Override: expected failure because mappings cannot be ordered.
        # KNOWN ISSUE: it takes only data_for_sorting while the base test also needs the as_series
        # argument, so the call would fail with an argument error regardless of the stated reason.
        super().test_searchsorted(data_for_sorting)

    @pytest.mark.xfail(reason="Can't compare dicts.")
    def test_equals(self, data, na_value, as_series):
        # [FUNCTION DOC]
        # Override: expected failure because mappings cannot be compared.
        # KNOWN ISSUE: it passes only three arguments to the base test, omitting the box argument the
        # base test requires, so it would fail with an argument error regardless of the stated reason.
        super().test_equals(data, na_value, as_series)

    @pytest.mark.skip("fill-value is interpreted as a dict of values")
    def test_fillna_copy_frame(self, data_missing):
        # [FUNCTION DOC]
        # Override: skipped (not run) because the fill value would be interpreted as a mapping of
        # values.
        super().test_fillna_copy_frame(data_missing)

    @pytest.mark.xfail(reason="Fails with CoW")
    def test_equals_same_data_different_object(self, data):
        # [FUNCTION DOC]
        # Override: expected failure under copy-on-write.
        super().test_equals_same_data_different_object(data)

    @pytest.mark.xfail(reason="failing on np.array(self, dtype=str)")
    def test_astype_str(self):
        # [FUNCTION DOC]
        # Override: expected failure because numpy cannot convert the array to a string array
        # (setting an element with a sequence).
        # KNOWN ISSUE: it takes no fixtures and calls the base test without the data argument, so it
        # would fail with an argument error regardless of the stated reason.
        """This currently fails in NumPy on np.array(self, dtype=str) with

        *** ValueError: setting an array element with a sequence
        """
        super().test_astype_str()

    @unhashable
    def test_groupby_extension_transform(self):
        # [FUNCTION DOC]
        # Override: unhashable expected failure; the name of the transformed series must be hashable
        # but is a mapping.
        # KNOWN ISSUE: takes no fixtures and calls the base test without the required data argument, hidden
        # by the expected-failure marker.
        """
        This currently fails in Series.name.setter, since the
        name must be hashable, but the value is a dictionary.
        I think this is what we want, i.e. `.name` should be the original
        values, and not the values for factorization.
        """
        super().test_groupby_extension_transform()

    @unhashable
    def test_groupby_extension_apply(self):
        # [FUNCTION DOC]
        # Override: unhashable expected failure; unique checking on an index hashes the mappings.
        # KNOWN ISSUE: takes no fixtures and calls the base test without its required arguments, hidden by
        # the expected-failure marker.
        """
        This fails in Index._do_unique_check with

        >   hash(val)
        E   TypeError: unhashable type: 'UserDict' with

        I suspect that once we support Index[ExtensionArray],
        we'll be able to dispatch unique.
        """
        super().test_groupby_extension_apply()

    @unhashable
    def test_groupby_extension_agg(self):
        # [FUNCTION DOC]
        # Override: unhashable expected failure; the comparison fails when the index contains mappings.
        # KNOWN ISSUE: takes no fixtures and calls the base test without its required arguments, hidden by
        # the expected-failure marker.
        """
        This fails when we get to tm.assert_series_equal when left.index
        contains dictionaries, which are not hashable.
        """
        super().test_groupby_extension_agg()

    @unhashable
    def test_groupby_extension_no_sort(self):
        # [FUNCTION DOC]
        # Override: unhashable expected failure; same cause as the aggregation test.
        # KNOWN ISSUE: takes no fixtures and calls the base test without its required argument, hidden by
        # the expected-failure marker.
        """
        This fails when we get to tm.assert_series_equal when left.index
        contains dictionaries, which are not hashable.
        """
        super().test_groupby_extension_no_sort()

    def test_arith_frame_with_scalar(self, data, all_arithmetic_operators, request):
        # [FUNCTION DOC]
        # Override. Marks an expected failure (raises while coercing to a series) when the first element
        # does not have exactly one entry, then runs the base test.
        if len(data[0]) != 1:
            mark = pytest.mark.xfail(reason="raises in coercing to Series")
            request.applymarker(mark)
        super().test_arith_frame_with_scalar(data, all_arithmetic_operators)

    def test_compare_array(self, data, comparison_op, request):
        # [FUNCTION DOC]
        # Override. Marks an expected failure (comparison methods not implemented) for equality and
        # inequality, then runs the base test.
        if comparison_op.__name__ in ["eq", "ne"]:
            mark = pytest.mark.xfail(reason="Comparison methods not implemented")
            request.applymarker(mark)
        super().test_compare_array(data, comparison_op)

    @pytest.mark.xfail(reason="ValueError: Must have equal len keys and value")
    def test_setitem_loc_scalar_mixed(self, data):
        # [FUNCTION DOC]
        # Override: expected failure (value error about keys and value lengths) then runs the base
        # test.
        super().test_setitem_loc_scalar_mixed(data)

    @pytest.mark.xfail(reason="ValueError: Must have equal len keys and value")
    def test_setitem_loc_scalar_multiple_homogoneous(self, data):
        # [FUNCTION DOC]
        # Override: expected failure (value error about keys and value lengths).
        super().test_setitem_loc_scalar_multiple_homogoneous(data)

    @pytest.mark.xfail(reason="ValueError: Must have equal len keys and value")
    def test_setitem_iloc_scalar_mixed(self, data):
        # [FUNCTION DOC]
        # Override: expected failure (value error about keys and value lengths).
        super().test_setitem_iloc_scalar_mixed(data)

    @pytest.mark.xfail(reason="ValueError: Must have equal len keys and value")
    def test_setitem_iloc_scalar_multiple_homogoneous(self, data):
        # [FUNCTION DOC]
        # Override: expected failure (value error about keys and value lengths).
        super().test_setitem_iloc_scalar_multiple_homogoneous(data)

    @pytest.mark.parametrize(
        "mask",
        [
            np.array([True, True, True, False, False]),
            pd.array([True, True, True, False, False], dtype="boolean"),
            pd.array([True, True, True, pd.NA, pd.NA], dtype="boolean"),
        ],
        ids=["numpy-array", "boolean-array", "boolean-array-na"],
    )
    def test_setitem_mask(self, data, mask, box_in_series, request):
        # [FUNCTION DOC]
        # Override, parametrized over three mask kinds. Marks expected failure (list-like indexer
        # length mismatch) when boxed in a series, and expected failure (unwanted deprecation warning)
        # when not boxed and the mask is not a numpy array; then runs the base test.
        if box_in_series:
            mark = pytest.mark.xfail(
                reason="cannot set using a list-like indexer with a different length"
            )
            request.applymarker(mark)
        elif not isinstance(mask, np.ndarray):
            mark = pytest.mark.xfail(reason="Issues unwanted DeprecationWarning")
            request.applymarker(mark)
        super().test_setitem_mask(data, mask, box_in_series)

    def test_setitem_mask_raises(self, data, box_in_series, request):
        # [FUNCTION DOC]
        # Override. Marks an expected failure (fails to raise) when not boxed in a series, then runs
        # the base test.
        if not box_in_series:
            mark = pytest.mark.xfail(reason="Fails to raise")
            request.applymarker(mark)

        super().test_setitem_mask_raises(data, box_in_series)

    @pytest.mark.xfail(
        reason="cannot set using a list-like indexer with a different length"
    )
    def test_setitem_mask_boolean_array_with_na(self, data, box_in_series):
        # [FUNCTION DOC]
        # Override: expected failure (list-like indexer length mismatch).
        super().test_setitem_mask_boolean_array_with_na(data, box_in_series)

    @pytest.mark.parametrize(
        "idx",
        [[0, 1, 2], pd.array([0, 1, 2], dtype="Int64"), np.array([0, 1, 2])],
        ids=["list", "integer-array", "numpy-array"],
    )
    def test_setitem_integer_array(self, data, idx, box_in_series, request):
        # [FUNCTION DOC]
        # Override, parametrized over indexer kinds. Marks an expected failure (list-like indexer length
        # mismatch) when boxed in a series, then runs the base test.
        if box_in_series:
            mark = pytest.mark.xfail(
                reason="cannot set using a list-like indexer with a different length"
            )
            request.applymarker(mark)
        super().test_setitem_integer_array(data, idx, box_in_series)

    @pytest.mark.parametrize(
        "idx",
        [
            [0, 1, 2, pd.NA],
            pd.array([0, 1, 2, pd.NA], dtype="Int64"),
        ],
        ids=["list", "integer-array"],
    )
    @pytest.mark.parametrize(
        "box_in_series",
        [
            True,
            pytest.param(
                False,
                marks=pytest.mark.xfail(
                    reason="list indices must be integers or slices, not NAType"
                ),
            ),
        ],
    )
    def test_setitem_integer_with_missing_raises(self, data, idx, box_in_series):
        # [FUNCTION DOC]
        # Override: re-parametrizes box_in_series so the unboxed case is an expected failure (list
        # indices must be integers or slices, not the NA type); then runs the base test.
        super().test_setitem_integer_with_missing_raises(data, idx, box_in_series)

    @pytest.mark.xfail(reason="Fails to raise")
    def test_setitem_scalar_key_sequence_raise(self, data):
        # [FUNCTION DOC]
        # Override: expected failure because the assignment fails to raise.
        super().test_setitem_scalar_key_sequence_raise(data)

    def test_setitem_with_expansion_dataframe_column(self, data, full_indexer, request):
        # [FUNCTION DOC]
        # Override. Marks an expected failure (slice is not iterable) when the test id contains the full
        # slice indexer name, then runs the base test.
        if "full_slice" in request.node.name:
            mark = pytest.mark.xfail(reason="slice is not iterable")
            request.applymarker(mark)
        super().test_setitem_with_expansion_dataframe_column(data, full_indexer)

    @pytest.mark.xfail(reason="slice is not iterable")
    def test_setitem_frame_2d_values(self, data):
        # [FUNCTION DOC]
        # Override: expected failure (slice is not iterable).
        super().test_setitem_frame_2d_values(data)

    @pytest.mark.xfail(
        reason="cannot set using a list-like indexer with a different length"
    )
    @pytest.mark.parametrize("setter", ["loc", None])
    def test_setitem_mask_broadcast(self, data, setter):
        # [FUNCTION DOC]
        # Override: expected failure (list-like indexer length mismatch), parametrized over setter.
        super().test_setitem_mask_broadcast(data, setter)

    @pytest.mark.xfail(
        reason="cannot set using a slice indexer with a different length"
    )
    def test_setitem_slice(self, data, box_in_series):
        # [FUNCTION DOC]
        # Override: expected failure (slice indexer length mismatch).
        super().test_setitem_slice(data, box_in_series)

    @pytest.mark.xfail(reason="slice object is not iterable")
    def test_setitem_loc_iloc_slice(self, data):
        # [FUNCTION DOC]
        # Override: expected failure (slice object is not iterable).
        super().test_setitem_loc_iloc_slice(data)

    @pytest.mark.xfail(reason="slice object is not iterable")
    def test_setitem_slice_mismatch_length_raises(self, data):
        # [FUNCTION DOC]
        # Override: expected failure (slice object is not iterable).
        super().test_setitem_slice_mismatch_length_raises(data)

    @pytest.mark.xfail(reason="slice object is not iterable")
    def test_setitem_slice_array(self, data):
        # [FUNCTION DOC]
        # Override: expected failure (slice object is not iterable).
        super().test_setitem_slice_array(data)

    @pytest.mark.xfail(reason="Fail to raise")
    def test_setitem_invalid(self, data, invalid_scalar):
        # [FUNCTION DOC]
        # Override: expected failure (fails to raise).
        super().test_setitem_invalid(data, invalid_scalar)

    @pytest.mark.xfail(
        reason="result readonly flag is incorrect and does not support na_value"
    )
    def test_readonly_propagates_to_numpy_array_method(self, data):
        # [FUNCTION DOC]
        # Override: expected failure (the read-only flag of the result is incorrect and the na_value
        # argument is unsupported).
        super().test_readonly_propagates_to_numpy_array_method(data)

    @pytest.mark.xfail(reason="only integer scalar arrays can be converted")
    def test_setitem_2d_values(self, data):
        # [FUNCTION DOC]
        # Override: expected failure (only integer scalar arrays can be converted).
        super().test_setitem_2d_values(data)

    @pytest.mark.xfail(reason="data type 'json' not understood")
    @pytest.mark.parametrize("engine", ["c", "python"])
    def test_EA_types(self, engine, data, request):
        # [FUNCTION DOC]
        # Override: expected failure (the data type name json is not understood by the parser),
        # parametrized over the C and Python engines.
        super().test_EA_types(engine, data, request)


def custom_assert_series_equal(left, right, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: series equality assertion that works with JSON arrays (numpy cannot build arrays
    # from equal-length user mappings, which the default assertion does via the values property).
    # Inputs: left, right series and extra positional/keyword arguments forwarded. If the left dtype
    # name is json: asserts the dtypes are equal, then rebuilds each series with the values converted
    # to an object array and wrapped back in the array type (keeping index and name). Then calls the
    # standard series assertion. Raises an assertion error on difference.
    # NumPy doesn't handle an array of equal-length UserDicts.
    # The default assert_series_equal eventually does a
    # Series.values, which raises. We work around it by
    # converting the UserDicts to dicts.
    if left.dtype.name == "json":
        assert left.dtype == right.dtype
        left = pd.Series(
            JSONArray(left.values.astype(object)), index=left.index, name=left.name
        )
        right = pd.Series(
            JSONArray(right.values.astype(object)),
            index=right.index,
            name=right.name,
        )
    tm.assert_series_equal(left, right, *args, **kwargs)


def custom_assert_frame_equal(left, right, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: frame equality assertion that works with JSON columns. Inputs: left, right frames
    # plus extra arguments forwarded. Asserts the column indexes are equal (using the check options
    # found in the keyword arguments and a default obj name for messages), compares each JSON-dtype
    # column of the left frame through the custom series assertion, drops those columns from both
    # frames and compares the remainder with the standard frame assertion.
    obj_type = kwargs.get("obj", "DataFrame")
    tm.assert_index_equal(
        left.columns,
        right.columns,
        exact=kwargs.get("check_column_type", "equiv"),
        check_names=kwargs.get("check_names", True),
        check_exact=kwargs.get("check_exact", False),
        check_categorical=kwargs.get("check_categorical", True),
        obj=f"{obj_type}.columns",
    )

    jsons = (left.dtypes == "json").index

    for col in jsons:
        custom_assert_series_equal(left[col], right[col], *args, **kwargs)

    left = left.drop(columns=jsons)
    right = right.drop(columns=jsons)
    tm.assert_frame_equal(left, right, *args, **kwargs)


def test_custom_asserts():
    # [FUNCTION DOC]
    # Tests the custom assertion helpers. Equal series and frames of three single-key mappings pass;
    # a series with different contents (take with repeated first element) must fail in the series and
    # frame helpers with an assertion error whose message says the series are different.
    # This would always trigger the KeyError from trying to put
    # an array of equal-length UserDicts inside an ndarray.
    data = JSONArray(
        [
            collections.UserDict({"a": 1}),
            collections.UserDict({"b": 2}),
            collections.UserDict({"c": 3}),
        ]
    )
    a = pd.Series(data)
    custom_assert_series_equal(a, a)
    custom_assert_frame_equal(a.to_frame(), a.to_frame())

    b = pd.Series(data.take([0, 0, 1]))
    msg = r"Series are different"
    with pytest.raises(AssertionError, match=msg):
        custom_assert_series_equal(a, b)

    with pytest.raises(AssertionError, match=msg):
        custom_assert_frame_equal(a.to_frame(), b.to_frame())
