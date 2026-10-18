# [MODULE DOC]
# Tests for assorted routines of the compiled general-purpose library (aliased
# as lib) and the writers library: maximum string length, multi-list unique,
# fast multi-get with timedelta keys, the conversion of integer indices or
# booleans into slices, reverse indexers, range-indexer checks, sentinel
# pickling and string-array coercion. Public contents are the TestMisc and
# TestIndexing classes plus module-level tests. No import-time side effects.

import pickle

import numpy as np
import pytest

from pandas._libs import (
    Timedelta,
    lib,
    writers as libwriters,
)
from pandas.compat import IS64

from pandas import Index
import pandas._testing as tm


class TestMisc:
    # [CLASS DOC]
    # Miscellaneous tests of library helpers (string length, unique over lists,
    # multi-get lookups).
    def test_max_len_string_array(self):
        # [FUNCTION DOC]
        # The maximum string length helper returns 3 for an object array of strings
        # (including NaN), for an array round-tripped through unicode, and through
        # bytes. A plain unicode (non-object) array is rejected with a type error
        # reporting that no matching signature was found.
        arr = a = np.array(["foo", "b", np.nan], dtype="object")
        assert libwriters.max_len_string_array(arr) == 3

        # unicode
        arr = a.astype("U").astype(object)
        assert libwriters.max_len_string_array(arr) == 3

        # bytes for python3
        arr = a.astype("S").astype(object)
        assert libwriters.max_len_string_array(arr) == 3

        # raises
        msg = "No matching signature found"
        with pytest.raises(TypeError, match=msg):
            libwriters.max_len_string_array(arr.astype("U"))

    def test_fast_unique_multiple_list_gen_sort(self):
        # [FUNCTION DOC]
        # The multi-list unique helper over a generator of lists returns the distinct
        # strings: sorted alphabetically when sorting is requested, and in first-seen
        # order when it is not.
        keys = [["p", "a"], ["n", "d"], ["a", "s"]]

        gen = (key for key in keys)
        expected = np.array(["a", "d", "n", "p", "s"])
        out = lib.fast_unique_multiple_list_gen(gen, sort=True)
        tm.assert_numpy_array_equal(np.array(out), expected)

        gen = (key for key in keys)
        expected = np.array(["p", "a", "n", "d", "s"])
        out = lib.fast_unique_multiple_list_gen(gen, sort=False)
        tm.assert_numpy_array_equal(np.array(out), expected)

    def test_fast_multiget_timedelta_resos(self):
        # [FUNCTION DOC]
        # Checks that the multi-get lookup finds a timedelta key regardless of the
        # resolution of the keys in the mapping (nanosecond default versus seconds),
        # using an index of multiples of one day. Repeats with a very large timedelta
        # that cannot be cast to nanosecond resolution, first asserting its hash
        # matches the millisecond and microsecond versions, and compares lookups in
        # mappings keyed at the default and at millisecond units.
        # This will become relevant for test_constructor_dict_timedelta64_index
        #  once Timedelta constructor preserves reso when passed a
        #  np.timedelta64 object
        td = Timedelta(days=1)

        mapping1 = {td: 1}
        mapping2 = {td.as_unit("s"): 1}

        oindex = Index([td * n for n in range(3)])._values.astype(object)

        expected = lib.fast_multiget(mapping1, oindex)
        result = lib.fast_multiget(mapping2, oindex)
        tm.assert_numpy_array_equal(result, expected)

        # case that can't be cast to td64ns
        td = Timedelta(np.timedelta64(146000, "D"))
        assert hash(td) == hash(td.as_unit("ms"))
        assert hash(td) == hash(td.as_unit("us"))
        mapping1 = {td: 1}
        mapping2 = {td.as_unit("ms"): 1}

        oindex = Index([td * n for n in range(3)])._values.astype(object)

        expected = lib.fast_multiget(mapping1, oindex)
        result = lib.fast_multiget(mapping2, oindex)
        tm.assert_numpy_array_equal(result, expected)


class TestIndexing:
    # [CLASS DOC]
    # Tests for the helpers converting integer positions or boolean arrays into a
    # slice when the pattern is regular.
    def test_maybe_indices_to_slice_left_edge(self):
        # [FUNCTION DOC]
        # An empty index array is converted to a slice, and selecting with either
        # gives equal results.
        target = np.arange(100)

        # slice
        indices = np.array([], dtype=np.intp)
        maybe_slice = lib.maybe_indices_to_slice(indices, len(target))

        assert isinstance(maybe_slice, slice)
        tm.assert_numpy_array_equal(target[indices], target[maybe_slice])

    @pytest.mark.parametrize("end", [1, 2, 5, 20, 99])
    @pytest.mark.parametrize("step", [1, 2, 4])
    def test_maybe_indices_to_slice_left_edge_not_slice_end_steps(self, end, step):
        # [FUNCTION DOC]
        # Parametrized over end and step. Index arrays starting at zero with positive
        # steps (and their reverse) are converted to a slice; selecting from a length
        # 100 array gives the same result with the indices and with the slice.
        target = np.arange(100)
        indices = np.arange(0, end, step, dtype=np.intp)
        maybe_slice = lib.maybe_indices_to_slice(indices, len(target))

        assert isinstance(maybe_slice, slice)
        tm.assert_numpy_array_equal(target[indices], target[maybe_slice])

        # reverse
        indices = indices[::-1]
        maybe_slice = lib.maybe_indices_to_slice(indices, len(target))

        assert isinstance(maybe_slice, slice)
        tm.assert_numpy_array_equal(target[indices], target[maybe_slice])

    @pytest.mark.parametrize(
        "case", [[2, 1, 2, 0], [2, 2, 1, 0], [0, 1, 2, 1], [-2, 0, 2], [2, 0, -2]]
    )
    def test_maybe_indices_to_slice_left_edge_not_slice(self, case):
        # [FUNCTION DOC]
        # Parametrized over irregular index lists (repeats or negatives, starting at the
        # left edge). The helper must return the original index array, not a slice, and
        # selecting gives the same result.
        # not slice
        target = np.arange(100)
        indices = np.array(case, dtype=np.intp)
        maybe_slice = lib.maybe_indices_to_slice(indices, len(target))

        assert not isinstance(maybe_slice, slice)
        tm.assert_numpy_array_equal(maybe_slice, indices)
        tm.assert_numpy_array_equal(target[indices], target[maybe_slice])

    @pytest.mark.parametrize("start", [0, 2, 5, 20, 97, 98])
    @pytest.mark.parametrize("step", [1, 2, 4])
    def test_maybe_indices_to_slice_right_edge(self, start, step):
        # [FUNCTION DOC]
        # Parametrized over start and step. Index arrays that run up to the position
        # before the last of a length 100 array (and their reverse) convert to a slice
        # with the same selection result.
        target = np.arange(100)

        # slice
        indices = np.arange(start, 99, step, dtype=np.intp)
        maybe_slice = lib.maybe_indices_to_slice(indices, len(target))

        assert isinstance(maybe_slice, slice)
        tm.assert_numpy_array_equal(target[indices], target[maybe_slice])

        # reverse
        indices = indices[::-1]
        maybe_slice = lib.maybe_indices_to_slice(indices, len(target))

        assert isinstance(maybe_slice, slice)
        tm.assert_numpy_array_equal(target[indices], target[maybe_slice])

    def test_maybe_indices_to_slice_right_edge_not_slice(self):
        # [FUNCTION DOC]
        # Index arrays that include the out-of-bounds position 100 (ascending and
        # descending) are not converted to a slice; the same array is returned.
        # Selecting with the indices or the returned array raises an index error saying
        # index 100 is out of bounds.
        # not slice
        target = np.arange(100)
        indices = np.array([97, 98, 99, 100], dtype=np.intp)
        maybe_slice = lib.maybe_indices_to_slice(indices, len(target))

        assert not isinstance(maybe_slice, slice)
        tm.assert_numpy_array_equal(maybe_slice, indices)

        msg = "index 100 is out of bounds for axis (0|1) with size 100"

        with pytest.raises(IndexError, match=msg):
            target[indices]
        with pytest.raises(IndexError, match=msg):
            target[maybe_slice]

        indices = np.array([100, 99, 98, 97], dtype=np.intp)
        maybe_slice = lib.maybe_indices_to_slice(indices, len(target))

        assert not isinstance(maybe_slice, slice)
        tm.assert_numpy_array_equal(maybe_slice, indices)

        with pytest.raises(IndexError, match=msg):
            target[indices]
        with pytest.raises(IndexError, match=msg):
            target[maybe_slice]

    @pytest.mark.parametrize(
        "case", [[99, 97, 99, 96], [99, 99, 98, 97], [98, 98, 97, 96]]
    )
    def test_maybe_indices_to_slice_right_edge_cases(self, case):
        # [FUNCTION DOC]
        # Parametrized irregular index lists near the right edge: the result is the
        # original array not a slice, and selecting is consistent.
        target = np.arange(100)
        indices = np.array(case, dtype=np.intp)
        maybe_slice = lib.maybe_indices_to_slice(indices, len(target))

        assert not isinstance(maybe_slice, slice)
        tm.assert_numpy_array_equal(maybe_slice, indices)
        tm.assert_numpy_array_equal(target[indices], target[maybe_slice])

    @pytest.mark.parametrize("step", [1, 2, 4, 5, 8, 9])
    def test_maybe_indices_to_slice_both_edges(self, step):
        # [FUNCTION DOC]
        # Parametrized over step using a length 10 array so the indices touch both
        # edges. Ascending and reversed arrays convert to a slice with equal selection.
        target = np.arange(10)

        # slice
        indices = np.arange(0, 9, step, dtype=np.intp)
        maybe_slice = lib.maybe_indices_to_slice(indices, len(target))
        assert isinstance(maybe_slice, slice)
        tm.assert_numpy_array_equal(target[indices], target[maybe_slice])

        # reverse
        indices = indices[::-1]
        maybe_slice = lib.maybe_indices_to_slice(indices, len(target))
        assert isinstance(maybe_slice, slice)
        tm.assert_numpy_array_equal(target[indices], target[maybe_slice])

    @pytest.mark.parametrize("case", [[4, 2, 0, -2], [2, 2, 1, 0], [0, 1, 2, 1]])
    def test_maybe_indices_to_slice_both_edges_not_slice(self, case):
        # [FUNCTION DOC]
        # Irregular index lists touching both edges (including a negative entry or
        # repeats) are returned unchanged and not converted to a slice.
        # not slice
        target = np.arange(10)
        indices = np.array(case, dtype=np.intp)
        maybe_slice = lib.maybe_indices_to_slice(indices, len(target))
        assert not isinstance(maybe_slice, slice)
        tm.assert_numpy_array_equal(maybe_slice, indices)
        tm.assert_numpy_array_equal(target[indices], target[maybe_slice])

    @pytest.mark.parametrize("start, end", [(2, 10), (5, 25), (65, 97)])
    @pytest.mark.parametrize("step", [1, 2, 4, 20])
    def test_maybe_indices_to_slice_middle(self, start, end, step):
        # [FUNCTION DOC]
        # Parametrized over (start, end) and step for ranges in the middle of a length
        # 100 array; ascending and reversed arrays convert to slices with equal
        # selection.
        target = np.arange(100)

        # slice
        indices = np.arange(start, end, step, dtype=np.intp)
        maybe_slice = lib.maybe_indices_to_slice(indices, len(target))

        assert isinstance(maybe_slice, slice)
        tm.assert_numpy_array_equal(target[indices], target[maybe_slice])

        # reverse
        indices = indices[::-1]
        maybe_slice = lib.maybe_indices_to_slice(indices, len(target))

        assert isinstance(maybe_slice, slice)
        tm.assert_numpy_array_equal(target[indices], target[maybe_slice])

    @pytest.mark.parametrize(
        "case", [[14, 12, 10, 12], [12, 12, 11, 10], [10, 11, 12, 11]]
    )
    def test_maybe_indices_to_slice_middle_not_slice(self, case):
        # [FUNCTION DOC]
        # Irregular index lists in the middle of the array (repeats, non-monotonic) are
        # returned unchanged rather than as a slice.
        # not slice
        target = np.arange(100)
        indices = np.array(case, dtype=np.intp)
        maybe_slice = lib.maybe_indices_to_slice(indices, len(target))

        assert not isinstance(maybe_slice, slice)
        tm.assert_numpy_array_equal(maybe_slice, indices)
        tm.assert_numpy_array_equal(target[indices], target[maybe_slice])

    def test_maybe_booleans_to_slice(self):
        # [FUNCTION DOC]
        # Boolean-to-slice helper on an unsigned 8-bit array of zeros and ones: for a
        # non-contiguous pattern it returns a boolean array; for an empty input it
        # returns the empty slice from zero to zero.
        arr = np.array([0, 0, 1, 1, 1, 0, 1], dtype=np.uint8)
        result = lib.maybe_booleans_to_slice(arr)
        assert result.dtype == np.bool_

        result = lib.maybe_booleans_to_slice(arr[:0])
        assert result == slice(0, 0)

    def test_get_reverse_indexer(self):
        # [FUNCTION DOC]
        # Reverse indexer helper: given an indexer with minus one for missing entries
        # and a count of 5, returns for each target position the index at which it
        # appears in the indexer (positions 4, 2, 3, 6, 7).
        indexer = np.array([-1, -1, 1, 2, 0, -1, 3, 4], dtype=np.intp)
        result = lib.get_reverse_indexer(indexer, 5)
        expected = np.array([4, 2, 3, 6, 7], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["int64", "int32"])
    def test_is_range_indexer(self, dtype):
        # [FUNCTION DOC]
        # GH#50592. A 64- or 32-bit integer array equal to the range 0..99 is
        # recognised by the range indexer check against length 100.
        # GH#50592
        left = np.arange(0, 100, dtype=dtype)
        assert lib.is_range_indexer(left, 100)

    @pytest.mark.skipif(
        not IS64,
        reason="2**31 is too big for Py_ssize_t on 32-bit. "
        "It doesn't matter though since you cannot create an array that long on 32-bit",
    )
    @pytest.mark.parametrize("dtype", ["int64", "int32"])
    def test_is_range_indexer_big_n(self, dtype):
        # [FUNCTION DOC]
        # GH53616. Skipped on non-64-bit platforms. Checking a length-100 range
        # against a length of 2 to the 31st power returns false (no overflow).
        # GH53616
        left = np.arange(0, 100, dtype=dtype)

        assert not lib.is_range_indexer(left, 2**31)

    @pytest.mark.parametrize("dtype", ["int64", "int32"])
    def test_is_range_indexer_not_equal(self, dtype):
        # [FUNCTION DOC]
        # GH#50592. An array that is not the identity range (starting at 1) is not a
        # range indexer.
        # GH#50592
        left = np.array([1, 2], dtype=dtype)
        assert not lib.is_range_indexer(left, 2)

    @pytest.mark.parametrize("dtype", ["int64", "int32"])
    def test_is_range_indexer_not_equal_shape(self, dtype):
        # [FUNCTION DOC]
        # GH#50592. An array of length 3 compared against length 2 is not a range
        # indexer.
        # GH#50592
        left = np.array([0, 1, 2], dtype=dtype)
        assert not lib.is_range_indexer(left, 2)


def test_cache_readonly_preserve_docstrings():
    # [FUNCTION DOC]
    # GH18197. Checks that the cached-property based attribute for missing-value
    # detection on Index keeps a non-empty docstring.
    # GH18197
    assert Index.hasnans.__doc__ is not None


def test_no_default_pickle(temp_file):
    # [FUNCTION DOC]
    # GH#40397. Takes the temp file fixture. Round-trips the library's no-default
    # sentinel through pickle via the testing helper and checks the identical
    # singleton object comes back.
    # GH#40397
    obj = tm.round_trip_pickle(lib.no_default, temp_file)
    assert obj is lib.no_default


def test_ensure_string_array_copy():
    # [FUNCTION DOC]
    # Regression for issue 54654. A pickle-round-tripped object array containing a
    # string and None is converted with copy disabled. Result must not share memory
    # with the input, the input still holds None, and the result holds NaN in
    # place of None.
    # ensure the original array is not modified in case of copy=False with
    # pickle-roundtripped object dtype array
    # https://github.com/pandas-dev/pandas/issues/54654
    arr = np.array(["a", None], dtype=object)
    arr = pickle.loads(pickle.dumps(arr))
    result = lib.ensure_string_array(arr, copy=False)
    assert not np.shares_memory(arr, result)
    assert arr[1] is None
    assert result[1] is np.nan


def test_ensure_string_array_list_of_lists():
    # [FUNCTION DOC]
    # GH#61155. Converting a list of lists to a string array keeps each element as
    # the string representation of the list, producing an object array of two
    # strings.
    # GH#61155: ensure list of lists doesn't get converted to string
    arr = [list("test"), list("word")]
    result = lib.ensure_string_array(arr)

    # Each item in result should still be a list, not a stringified version
    expected = np.array(["['t', 'e', 's', 't']", "['w', 'o', 'r', 'd']"], dtype=object)
    tm.assert_numpy_array_equal(result, expected)


def test_item_from_zerodim_for_subclasses():
    # [FUNCTION DOC]
    # GH#62981. Verifies the zero-dimensional unwrapping helper keeps ndarray
    # subclasses. Defines a local subclass. For plain arrays, a zero-dimension
    # array unwraps to a scalar and a one-dimension array is returned as is; for
    # the subclass, both zero- and one-dimension inputs stay instances of the
    # subclass.
    # GH#62981 Ensure item_from_zerodim preserves subclasses of ndarray
    # Define a custom ndarray subclass
    class TestArray(np.ndarray):
        # [CLASS DOC]
        # HELPER: local ndarray subclass used only by the zero-dimension unwrapping test.
        # Instances are created from any array-like and carry a marker attribute set
        # whenever the array is finalized.
        def __new__(cls, input_array):
            # [FUNCTION DOC]
            # HELPER: constructor for the local subclass. Takes input_array, converts it to
            # an array and returns it viewed as the subclass.
            return np.asarray(input_array).view(cls)

        def __array_finalize__(self, obj) -> None:
            # [FUNCTION DOC]
            # HELPER: finalization hook of the local subclass. Takes the source object and
            # sets a marker attribute on the new array. Returns nothing.
            self._is_test_array = True

    # Define test data
    val_0_dim = 1
    val_1_dim = [1, 2, 3]

    # 0-dim and 1-dim numpy arrays
    arr_0_dim = np.array(val_0_dim)
    arr_1_dim = np.array(val_1_dim)

    # 0-dim and 1-dim TestArray arrays
    test_arr_0_dim = TestArray(val_0_dim)
    test_arr_1_dim = TestArray(val_1_dim)

    # Check that behavior did not change for regular numpy arrays
    # Test with regular numpy 0-dim array
    result = lib.item_from_zerodim(arr_0_dim)
    expected = val_0_dim
    assert result == expected
    assert np.isscalar(result)

    # Test with regular numpy 1-dim array
    result = lib.item_from_zerodim(arr_1_dim)
    expected = arr_1_dim
    tm.assert_numpy_array_equal(result, expected)
    assert isinstance(result, np.ndarray)

    # Check that behaviour for subclasses now is as expected
    # Test with TestArray 0-dim array
    result = lib.item_from_zerodim(test_arr_0_dim)
    expected = test_arr_0_dim
    assert result == expected
    assert isinstance(result, TestArray)

    # Test with TestArray 1-dim array
    result = lib.item_from_zerodim(test_arr_1_dim)
    expected = test_arr_1_dim
    assert np.all(result == expected)
    assert isinstance(result, TestArray)
