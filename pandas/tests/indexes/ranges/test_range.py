# [MODULE DOC]
# Tests specific to the range-based integer index type: attributes, copying,
# representation, insertion and deletion (including when the result stays a
# range versus falls back to a general integer index), caching of materialized
# data, monotonicity, equality, slicing, appending, lookup, boolean-mask and
# integer-list getitem, unary and arithmetic operators, rounding, reindexing,
# take, argmin and argmax, value counts and searchsorted. A class holds the
# generic tests that share a fixture; module-level functions cover the rest.
# Imports a private helper from the range index implementation module for
# the min-fitting-element test.

import numpy as np
import pytest

from pandas.core.dtypes.common import ensure_platform_int

import pandas as pd
from pandas import (
    Index,
    RangeIndex,
)
import pandas._testing as tm
from pandas.core.indexes.range import min_fitting_element


class TestRangeIndex:
    # [CLASS DOC]
    # Test class for behavior of the range index; shares a fixture that builds a
    # range index from zero to twenty with step two.
    @pytest.fixture
    def simple_index(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a range index with start zero, stop twenty and
        # step two (ten elements: zero, two, ..., eighteen), used by tests in this class.
        return RangeIndex(start=0, stop=20, step=2)

    def test_constructor_unwraps_index(self):
        # [FUNCTION DOC]
        # Constructing a range index from one to three exposes an internal data array
        # of the two 64-bit integers one and two; checked with the numpy array equality assertion.
        result = RangeIndex(1, 3)
        expected = np.array([1, 2], dtype=np.int64)
        tm.assert_numpy_array_equal(result._data, expected)

    def test_can_hold_identifiers(self, simple_index):
        # [FUNCTION DOC]
        # The private identifier check reports false for the first element of a range
        # index (a numeric index never holds identifiers).
        idx = simple_index
        key = idx[0]
        assert idx._can_hold_identifiers_and_holds_name(key) is False

    def test_too_many_names(self, simple_index):
        # [FUNCTION DOC]
        # Assigning a names list longer than the one level of the index raises a value
        # error whose message starts with "Length".
        index = simple_index
        with pytest.raises(ValueError, match="^Length"):
            index.names = ["roger", "harold"]

    @pytest.mark.parametrize(
        "index, start, stop, step",
        [
            (RangeIndex(5), 0, 5, 1),
            (RangeIndex(0, 5), 0, 5, 1),
            (RangeIndex(5, step=2), 0, 5, 2),
            (RangeIndex(1, 5, 2), 1, 5, 2),
        ],
    )
    def test_start_stop_step_attrs(self, index, start, stop, step):
        # [FUNCTION DOC]
        # Parametrized over several range indexes with their expected start, stop and
        # step. Checks that the public start, stop and step attributes match.
        # GH 25710
        assert index.start == start
        assert index.stop == stop
        assert index.step == step

    def test_copy(self):
        # [FUNCTION DOC]
        # A copy of a named range index is a distinct object, is identical to the
        # original, keeps the same underlying range and the name.
        i = RangeIndex(5, name="Foo")
        i_copy = i.copy()
        assert i_copy is not i
        assert i_copy.identical(i)
        assert i_copy._range == range(0, 5, 1)
        assert i_copy.name == "Foo"

    def test_repr(self):
        # [FUNCTION DOC]
        # The text representation of a named range index (and of a descending one) has
        # the exact expected form with start, stop, step and name, and evaluating that
        # text reproduces an equal index with exact type matching.
        i = RangeIndex(5, name="Foo")
        result = repr(i)
        expected = "RangeIndex(start=0, stop=5, step=1, name='Foo')"
        assert result == expected

        result = eval(result)
        tm.assert_index_equal(result, i, exact=True)

        i = RangeIndex(5, 0, -1)
        result = repr(i)
        expected = "RangeIndex(start=5, stop=0, step=-1)"
        assert result == expected

        result = eval(result)
        tm.assert_index_equal(result, i, exact=True)

    def test_insert(self):
        # [FUNCTION DOC]
        # Inserting the first element of a slice at position zero reproduces the
        # prefix of the original; inserting any of the missing markers (NaN, None, NA)
        # yields a float index with NaN at the insertion point; inserting the missing
        # datetime marker (NaT) yields an object index containing it.
        idx = RangeIndex(5, name="Foo")
        result = idx[1:4]

        # test 0th element
        tm.assert_index_equal(idx[0:4], result.insert(0, idx[0]), exact="equiv")

        # GH 18295 (test missing)
        expected = Index([0, np.nan, 1, 2, 3, 4], dtype=np.float64)
        for na in [np.nan, None, pd.NA]:
            result = RangeIndex(5).insert(1, na)
            tm.assert_index_equal(result, expected)

        result = RangeIndex(5).insert(1, pd.NaT)
        expected = Index([0, pd.NaT, 1, 2, 3, 4], dtype=object)
        tm.assert_index_equal(result, expected)

    def test_insert_edges_preserves_rangeindex(self):
        # [FUNCTION DOC]
        # Inserting a value that extends a range at either end (position zero or the end)
        # returns a range index (exact type check) with the extended bounds.
        idx = Index(range(4, 9, 2))

        result = idx.insert(0, 2)
        expected = Index(range(2, 9, 2))
        tm.assert_index_equal(result, expected, exact=True)

        result = idx.insert(3, 10)
        expected = Index(range(4, 11, 2))
        tm.assert_index_equal(result, expected, exact=True)

    def test_insert_middle_preserves_rangeindex(self):
        # [FUNCTION DOC]
        # Inserting the missing middle value into a range with a gap gives a contiguous
        # range; also works after scaling the index by two with the insertion value
        # doubled. Exact type checks.
        # insert in the middle
        idx = Index(range(0, 3, 2))
        result = idx.insert(1, 1)
        expected = Index(range(3))
        tm.assert_index_equal(result, expected, exact=True)

        idx = idx * 2
        result = idx.insert(1, 2)
        expected = expected * 2
        tm.assert_index_equal(result, expected, exact=True)

    def test_delete(self):
        # [FUNCTION DOC]
        # Deleting the first and last position returns the sliced range (exact type)
        # and keeps the name. Deleting the position equal to the length raises an
        # index or value error (depending on numpy version) with the out-of-bounds message.
        idx = RangeIndex(5, name="Foo")
        expected = idx[1:]
        result = idx.delete(0)
        tm.assert_index_equal(result, expected, exact=True)
        assert result.name == expected.name

        expected = idx[:-1]
        result = idx.delete(-1)
        tm.assert_index_equal(result, expected, exact=True)
        assert result.name == expected.name

        msg = "index 5 is out of bounds for axis 0 with size 5"
        with pytest.raises((IndexError, ValueError), match=msg):
            # either depending on numpy version
            result = idx.delete(len(idx))

    def test_delete_preserves_rangeindex(self):
        # [FUNCTION DOC]
        # Deleting the last element of a two-element named range, by list or by scalar,
        # returns a one-element range index with the name preserved.
        idx = Index(range(2), name="foo")

        result = idx.delete([1])
        expected = Index(range(1), name="foo")
        tm.assert_index_equal(result, expected, exact=True)

        result = idx.delete(1)
        tm.assert_index_equal(result, expected, exact=True)

    def test_delete_preserves_rangeindex_middle(self):
        # [FUNCTION DOC]
        # Deleting the middle element of a three-element range (by positive or negative
        # position) returns a stepped range that skips it; exact type check.
        idx = Index(range(3), name="foo")
        result = idx.delete(1)
        expected = idx[::2]
        tm.assert_index_equal(result, expected, exact=True)

        result = idx.delete(-2)
        tm.assert_index_equal(result, expected, exact=True)

    def test_delete_preserves_rangeindex_list_at_end(self):
        # [FUNCTION DOC]
        # Deleting a list of trailing positions (in either order) gives the leading
        # slice as a range index.
        idx = RangeIndex(0, 6, 1)

        loc = [2, 3, 4, 5]
        result = idx.delete(loc)
        expected = idx[:2]
        tm.assert_index_equal(result, expected, exact=True)

        result = idx.delete(loc[::-1])
        tm.assert_index_equal(result, expected, exact=True)

    def test_delete_preserves_rangeindex_list_middle(self):
        # [FUNCTION DOC]
        # Deleting a list of interior positions (in either order) leaves only the two
        # ends, which is still a range index with step five.
        idx = RangeIndex(0, 6, 1)

        loc = [1, 2, 3, 4]
        result = idx.delete(loc)
        expected = RangeIndex(0, 6, 5)
        tm.assert_index_equal(result, expected, exact=True)

        result = idx.delete(loc[::-1])
        tm.assert_index_equal(result, expected, exact=True)

    def test_delete_all_preserves_rangeindex(self):
        # [FUNCTION DOC]
        # Deleting every position (in either order) gives an empty range index.
        idx = RangeIndex(0, 6, 1)

        loc = [0, 1, 2, 3, 4, 5]
        result = idx.delete(loc)
        expected = idx[:0]
        tm.assert_index_equal(result, expected, exact=True)

        result = idx.delete(loc[::-1])
        tm.assert_index_equal(result, expected, exact=True)

    def test_delete_not_preserving_rangeindex(self):
        # [FUNCTION DOC]
        # Deleting a list of positions that leaves irregular spacing returns a plain
        # integer index with the remaining values (exact type check), in either order.
        idx = RangeIndex(0, 6, 1)

        loc = [0, 3, 5]
        result = idx.delete(loc)
        expected = Index([1, 2, 4])
        tm.assert_index_equal(result, expected, exact=True)

        result = idx.delete(loc[::-1])
        tm.assert_index_equal(result, expected, exact=True)

    def test_view(self):
        # [FUNCTION DOC]
        # A default view of an empty named range index keeps the name; a view as 64-bit
        # integers returns an array equal to the index values.
        i = RangeIndex(0, name="Foo")
        i_view = i.view()
        assert i_view.name == "Foo"

        i_view = i.view("i8")
        tm.assert_numpy_array_equal(i.values, i_view)

    def test_dtype(self, simple_index):
        # [FUNCTION DOC]
        # The index dtype is 64-bit integer.
        index = simple_index
        assert index.dtype == np.int64

    def test_cache(self):
        # [FUNCTION DOC]
        # Verifies lazy materialization: operations that should not need the
        # materialized integer array (repr, str, scalar lookup, membership, all, any,
        # iteration, frame display, scalar and slice loc/iloc on a frame using this
        # index) leave the private cache dictionary empty. Operations using take or
        # list-like selectors may add keys but must not add the data entry. Finally,
        # accessing the private data attribute materializes a 64-bit integer array, is
        # cached (same object returned) and appears in the cache with the expected
        # values. Related GitHub issues are cited in the source.
        # GH 26565, GH26617, GH35432, GH53387
        # This test checks whether _cache has been set.
        # Calling RangeIndex._cache["_data"] creates an int64 array of the same length
        # as the RangeIndex and stores it in _cache.
        idx = RangeIndex(0, 100, 10)

        assert idx._cache == {}

        repr(idx)
        assert idx._cache == {}

        str(idx)
        assert idx._cache == {}

        idx.get_loc(20)
        assert idx._cache == {}

        90 in idx  # True
        assert idx._cache == {}

        91 in idx  # False
        assert idx._cache == {}

        idx.all()
        assert idx._cache == {}

        idx.any()
        assert idx._cache == {}

        for _ in idx:
            pass
        assert idx._cache == {}

        df = pd.DataFrame({"a": range(10)}, index=idx)

        # df.__repr__ should not populate index cache
        str(df)
        assert idx._cache == {}

        df.loc[50]
        assert idx._cache == {}

        with pytest.raises(KeyError, match="51"):
            df.loc[51]
        assert idx._cache == {}

        df.loc[10:50]
        assert idx._cache == {}

        df.iloc[5:10]
        assert idx._cache == {}

        # after calling take, _cache may contain other keys, but not "_data"
        idx.take([3, 0, 1])
        assert "_data" not in idx._cache

        df.loc[[50]]
        assert "_data" not in idx._cache

        df.iloc[[5, 6, 7, 8, 9]]
        assert "_data" not in idx._cache

        # idx._cache should contain a _data entry after call to idx._data
        idx._data
        assert isinstance(idx._data, np.ndarray)
        assert idx._data is idx._data  # check cached value is reused
        assert "_data" in idx._cache
        expected = np.arange(0, 100, 10, dtype="int64")
        tm.assert_numpy_array_equal(idx._cache["_data"], expected)

    def test_is_monotonic(self):
        # [FUNCTION DOC]
        # Checks the monotonic flags for several ranges: increasing step (both flags
        # increasing, not decreasing, strictly increasing); decreasing step; a single
        # element, an empty range with start after stop, and an empty range with equal
        # bounds (all of which are both monotonic increasing and decreasing and
        # strictly so in the private strict properties).
        index = RangeIndex(0, 20, 2)
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_decreasing is False
        assert index._is_strictly_monotonic_increasing is True
        assert index._is_strictly_monotonic_decreasing is False

        index = RangeIndex(4, 0, -1)
        assert index.is_monotonic_increasing is False
        assert index._is_strictly_monotonic_increasing is False
        assert index.is_monotonic_decreasing is True
        assert index._is_strictly_monotonic_decreasing is True

        index = RangeIndex(1, 2)
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_decreasing is True
        assert index._is_strictly_monotonic_increasing is True
        assert index._is_strictly_monotonic_decreasing is True

        index = RangeIndex(2, 1)
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_decreasing is True
        assert index._is_strictly_monotonic_increasing is True
        assert index._is_strictly_monotonic_decreasing is True

        index = RangeIndex(1, 1)
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_decreasing is True
        assert index._is_strictly_monotonic_increasing is True
        assert index._is_strictly_monotonic_decreasing is True

    @pytest.mark.parametrize(
        "left,right",
        [
            (RangeIndex(0, 9, 2), RangeIndex(0, 10, 2)),
            (RangeIndex(0), RangeIndex(1, -1, 3)),
            (RangeIndex(1, 2, 3), RangeIndex(1, 3, 4)),
            (RangeIndex(0, -9, -2), RangeIndex(0, -10, -2)),
        ],
    )
    def test_equals_range(self, left, right):
        # [FUNCTION DOC]
        # Parametrized over pairs of ranges with different stop/step representation but
        # identical contents; equality is true symmetrically.
        assert left.equals(right)
        assert right.equals(left)

    def test_logical_compat(self, simple_index):
        # [FUNCTION DOC]
        # The all and any reductions on the index equal those computed on its values array.
        idx = simple_index
        assert idx.all() == idx.values.all()
        assert idx.any() == idx.values.any()

    def test_identical(self, simple_index):
        # [FUNCTION DOC]
        # Identity checks: a copy is identical to the original. The function then returns early when the index is a range index.
        # KNOWN ISSUE: the fixture is always a range index, so the early return always happens and the remaining assertions about object-dtype copies and renamed indexes are unreachable dead code.
        # Intended remaining checks: object-dtype copies differ in identity from integer ones, and renaming breaks identity.
        index = simple_index
        i = Index(index.copy())
        assert i.identical(index)

        # we don't allow object dtype for RangeIndex
        if isinstance(index, RangeIndex):
            return

        same_values_different_type = Index(i, dtype=object)
        assert not i.identical(same_values_different_type)

        i = index.copy(dtype=object)
        i = i.rename("foo")
        same_values = Index(i, dtype=object)
        assert same_values.identical(index.copy(dtype=object))

        assert not i.identical(index)
        assert Index(same_values, name="foo", dtype=object).identical(i)

        assert not index.copy(dtype=object).identical(index.copy(dtype="int64"))

    def test_nbytes(self):
        # [FUNCTION DOC]
        # A large range index uses far less memory (under a tenth) than an ordinary integer
        # index of the same values, and memory use is independent of length.
        # memory savings vs int index
        idx = RangeIndex(0, 1000)
        assert idx.nbytes < Index(idx._values).nbytes / 10

        # constant memory usage
        i2 = RangeIndex(0, 10)
        assert idx.nbytes == i2.nbytes

    @pytest.mark.parametrize(
        "start,stop,step",
        [
            # can't
            ("foo", "bar", "baz"),
            # shouldn't
            ("0", "1", "2"),
        ],
    )
    def test_cant_or_shouldnt_cast(self, start, stop, step):
        # [FUNCTION DOC]
        # Parametrized over non-integer strings (non-numeric and numeric-looking).
        # Constructing a range index from them raises a type error with a message
        # stating the wrong type for the value.
        msg = f"Wrong type {type(start)} for value {start}"
        with pytest.raises(TypeError, match=msg):
            RangeIndex(start, stop, step)

    def test_view_index(self, simple_index):
        # [FUNCTION DOC]
        # Viewing a range index as the general index class raises a type error (message
        # variants for references/object arrays accepted).
        index = simple_index
        msg = (
            "Cannot change data-type for array of references.|"
            "Cannot change data-type for object array.|"
        )
        with pytest.raises(TypeError, match=msg):
            index.view(Index)

    def test_prevent_casting(self, simple_index):
        # [FUNCTION DOC]
        # Converting a range index to object dtype yields an object-dtype result.
        index = simple_index
        result = index.astype("O")
        assert result.dtype == np.object_

    def test_repr_roundtrip(self, simple_index):
        # [FUNCTION DOC]
        # Evaluating the text representation of the fixture index gives an equal index.
        index = simple_index
        tm.assert_index_equal(eval(repr(index)), index)

    def test_slice_keep_name(self):
        # [FUNCTION DOC]
        # Slicing a named one-element range index (from position one onwards) keeps the same name.
        idx = RangeIndex(1, 2, name="asdf")
        assert idx.name == idx[1:].name

    @pytest.mark.parametrize(
        "index",
        [
            RangeIndex(start=0, stop=20, step=2, name="foo"),
            RangeIndex(start=18, stop=-1, step=-2, name="bar"),
        ],
        ids=["index_inc", "index_dec"],
    )
    def test_has_duplicates(self, index):
        # [FUNCTION DOC]
        # Parametrized over an increasing and a decreasing named range; both are unique and report no duplicates.
        assert index.is_unique
        assert not index.has_duplicates

    def test_extended_gcd(self, simple_index):
        # [FUNCTION DOC]
        # Checks the private extended-gcd helper of the index: for the pairs six/ten and
        # ten/six the first returned value is the gcd (two) and equals the linear
        # combination of the inputs with the other two returned coefficients.
        index = simple_index
        result = index._extended_gcd(6, 10)
        assert result[0] == result[1] * 6 + result[2] * 10
        assert 2 == result[0]

        result = index._extended_gcd(10, 6)
        assert 2 == result[1] * 10 + result[2] * 6
        assert 2 == result[0]

    def test_min_fitting_element(self):
        # [FUNCTION DOC]
        # Checks the imported helper that finds the smallest element of an arithmetic
        # progression (given start and step) that is at least a lower bound. Cases with
        # positive and negative steps, and an extremely large bound that exceeds 64-bit
        # range (which must be returned exactly).
        result = min_fitting_element(0, 2, 1)
        assert 2 == result

        result = min_fitting_element(1, 1, 1)
        assert 1 == result

        result = min_fitting_element(18, -2, 1)
        assert 2 == result

        result = min_fitting_element(5, -1, 1)
        assert 1 == result

        big_num = 500000000000000000000000

        result = min_fitting_element(5, 1, big_num)
        assert big_num == result

    def test_slice_specialised(self, simple_index):
        # [FUNCTION DOC]
        # Named fixture index: scalar getitem (positive and negative) returns plain integers;
        # full, positive-step, negative-step, overshooting-stop and reversed slices
        # (including start beyond either end) produce the expected values and name,
        # compared with equivalent (loose type) checking.
        index = simple_index
        index.name = "foo"

        # scalar indexing
        res = index[1]
        expected = 2
        assert res == expected

        res = index[-1]
        expected = 18
        assert res == expected

        # slicing
        # slice value completion
        index_slice = index[:]
        expected = index
        tm.assert_index_equal(index_slice, expected)

        # positive slice values
        index_slice = index[7:10:2]
        expected = Index([14, 18], name="foo")
        tm.assert_index_equal(index_slice, expected, exact="equiv")

        # negative slice values
        index_slice = index[-1:-5:-2]
        expected = Index([18, 14], name="foo")
        tm.assert_index_equal(index_slice, expected, exact="equiv")

        # stop overshoot
        index_slice = index[2:100:4]
        expected = Index([4, 12], name="foo")
        tm.assert_index_equal(index_slice, expected, exact="equiv")

        # reverse
        index_slice = index[::-1]
        expected = Index(index.values[::-1], name="foo")
        tm.assert_index_equal(index_slice, expected, exact="equiv")

        index_slice = index[-8::-1]
        expected = Index([4, 2, 0], name="foo")
        tm.assert_index_equal(index_slice, expected, exact="equiv")

        index_slice = index[-40::-1]
        expected = Index(np.array([], dtype=np.int64), name="foo")
        tm.assert_index_equal(index_slice, expected, exact="equiv")

        index_slice = index[40::-1]
        expected = Index(index.values[40::-1], name="foo")
        tm.assert_index_equal(index_slice, expected, exact="equiv")

        index_slice = index[10::-1]
        expected = Index(index.values[::-1], name="foo")
        tm.assert_index_equal(index_slice, expected, exact="equiv")

    @pytest.mark.parametrize("step", set(range(-5, 6)) - {0})
    def test_len_specialised(self, step):
        # [FUNCTION DOC]
        # Parametrized over non-zero steps from minus five to five. The length of the
        # range index equals the length of the matching numpy sequence; with start and
        # stop swapped (so the direction is wrong) the length is zero.
        # make sure that our len is the same as np.arange calc
        start, stop = (0, 5) if step > 0 else (5, 0)

        arr = np.arange(start, stop, step)
        index = RangeIndex(start, stop, step)
        assert len(index) == len(arr)

        index = RangeIndex(stop, start, step)
        assert len(index) == 0

    @pytest.mark.parametrize(
        "indices, expected",
        [
            ([RangeIndex(1, 12, 5)], RangeIndex(1, 12, 5)),
            ([RangeIndex(0, 6, 4)], RangeIndex(0, 6, 4)),
            ([RangeIndex(1, 3), RangeIndex(3, 7)], RangeIndex(1, 7)),
            ([RangeIndex(1, 5, 2), RangeIndex(5, 6)], RangeIndex(1, 6, 2)),
            ([RangeIndex(1, 3, 2), RangeIndex(4, 7, 3)], RangeIndex(1, 7, 3)),
            ([RangeIndex(-4, 3, 2), RangeIndex(4, 7, 2)], RangeIndex(-4, 7, 2)),
            ([RangeIndex(-4, -8), RangeIndex(-8, -12)], RangeIndex(0, 0)),
            ([RangeIndex(-4, -8), RangeIndex(3, -4)], RangeIndex(0, 0)),
            ([RangeIndex(-4, -8), RangeIndex(3, 5)], RangeIndex(3, 5)),
            ([RangeIndex(-4, -2), RangeIndex(3, 5)], Index([-4, -3, 3, 4])),
            ([RangeIndex(-2), RangeIndex(3, 5)], RangeIndex(3, 5)),
            ([RangeIndex(2), RangeIndex(2)], Index([0, 1, 0, 1])),
            ([RangeIndex(2), RangeIndex(2, 5), RangeIndex(5, 8, 4)], RangeIndex(0, 6)),
            (
                [RangeIndex(2), RangeIndex(3, 5), RangeIndex(5, 8, 4)],
                Index([0, 1, 3, 4, 5]),
            ),
            (
                [RangeIndex(-2, 2), RangeIndex(2, 5), RangeIndex(5, 8, 4)],
                RangeIndex(-2, 6),
            ),
            ([RangeIndex(3), Index([-1, 3, 15])], Index([0, 1, 2, -1, 3, 15])),
            ([RangeIndex(3), Index([-1, 3.1, 15.0])], Index([0, 1, 2, -1, 3.1, 15.0])),
            ([RangeIndex(3), Index(["a", None, 14])], Index([0, 1, 2, "a", None, 14])),
            ([RangeIndex(3, 1), Index(["a", None, 14])], Index(["a", None, 14])),
        ],
    )
    def test_append(self, indices, expected):
        # [FUNCTION DOC]
        # Parametrized over lists of indexes and the expected result: consecutive or
        # compatible ranges concatenate into a single range index (with the right start,
        # stop and step); gaps, overlaps, duplicates or non-range operands give a plain
        # index. When exactly two indexes are given, appending a single index rather than a
        # list also produces the same result. Exact type checks.
        # GH16212
        result = indices[0].append(indices[1:])
        tm.assert_index_equal(result, expected, exact=True)

        if len(indices) == 2:
            # Append single item rather than list
            result2 = indices[0].append(indices[1])
            tm.assert_index_equal(result2, expected, exact=True)

    def test_engineless_lookup(self):
        # [FUNCTION DOC]
        # Basic scalar lookup (location, indexer, missing key error, wrong-type key
        # error) on a range index must not build the hash engine: the cache has no
        # engine entry afterwards.
        # GH 16685
        # Standard lookup on RangeIndex should not require the engine to be
        # created
        idx = RangeIndex(2, 10, 3)

        assert idx.get_loc(5) == 1
        tm.assert_numpy_array_equal(
            idx.get_indexer([2, 8]), ensure_platform_int(np.array([0, 2]))
        )
        with pytest.raises(KeyError, match="3"):
            idx.get_loc(3)

        assert "_engine" not in idx._cache

        # Different types of scalars can be excluded immediately, no need to
        #  use the _engine
        with pytest.raises(KeyError, match="'a'"):
            idx.get_loc("a")

        assert "_engine" not in idx._cache

    @pytest.mark.parametrize(
        "ri",
        [
            RangeIndex(0, -1, -1),
            RangeIndex(0, 1, 1),
            RangeIndex(1, 3, 2),
            RangeIndex(0, -1, -2),
            RangeIndex(-3, -5, -2),
        ],
    )
    def test_append_len_one(self, ri):
        # [FUNCTION DOC]
        # Parametrized over several one-element range indexes. Appending an empty list
        # returns an equal index of the same range type (exact).
        # GH39401
        result = ri.append([])
        tm.assert_index_equal(result, ri, exact=True)

    @pytest.mark.parametrize("base", [RangeIndex(0, 2), Index([0, 1])])
    def test_isin_range(self, base):
        # [FUNCTION DOC]
        # Parametrized over a range index and an equivalent integer index. isin against a
        # single-value range returns a boolean array marking only the matching position.
        # GH#41151
        values = RangeIndex(0, 1)
        result = base.isin(values)
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

    def test_sort_values_key(self):
        # [FUNCTION DOC]
        # Sorting a range index with a key function that maps values through a
        # dictionary orders by the mapped value, giving an integer index; the result
        # matches sorting the equivalent series by the same key.
        # GH#43666, GH#52764
        sort_order = {8: 2, 6: 0, 4: 8, 2: 10, 0: 12}
        values = RangeIndex(0, 10, 2)
        result = values.sort_values(key=lambda x: x.map(sort_order))
        expected = Index([6, 8, 4, 2, 0], dtype="int64")
        tm.assert_index_equal(result, expected, check_exact=True)

        # check this matches the Series.sort_values behavior
        ser = values.to_series()
        result2 = ser.sort_values(key=lambda x: x.map(sort_order))
        tm.assert_series_equal(result2, expected.to_series(), check_exact=True)

    def test_range_index_rsub_by_const(self):
        # [FUNCTION DOC]
        # Subtracting a range index from an integer constant (reverse subtraction) yields a range index with the expected reversed bounds.
        # GH#53255
        result = 3 - RangeIndex(0, 4, 1)
        expected = RangeIndex(3, -1, -1)
        tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "rng, decimals",
    [
        [range(5), 0],
        [range(5), 2],
        [range(10, 30, 10), -1],
        [range(30, 10, -10), -1],
    ],
)
def test_range_round_returns_rangeindex(rng, decimals):
    # [FUNCTION DOC]
    # Parametrized over ranges and decimals for which rounding does not change the values (for example multiples of ten with a negative
    # decimal count). The rounded result is an exact-type range index equal to a copy.
    ri = RangeIndex(rng)
    expected = ri.copy()
    result = ri.round(decimals=decimals)
    tm.assert_index_equal(result, expected, exact=True)


@pytest.mark.parametrize(
    "rng, decimals",
    [
        [range(10, 30, 1), -1],
        [range(30, 10, -1), -1],
        [range(11, 14), -10],
    ],
)
def test_range_round_returns_index(rng, decimals):
    # [FUNCTION DOC]
    # Parametrized over ranges and decimals where rounding does change the values.
    # The result is a plain integer index equal to rounding the equivalent integer
    # index (exact type).
    ri = RangeIndex(rng)
    expected = Index(list(rng)).round(decimals=decimals)
    result = ri.round(decimals=decimals)
    tm.assert_index_equal(result, expected, exact=True)


def test_reindex_1_value_returns_rangeindex():
    # [FUNCTION DOC]
    # Reindexing a stepped named range to a single existing label gives a one-element range with the name and indexer array containing its position.
    ri = RangeIndex(0, 10, 2, name="foo")
    result, result_indexer = ri.reindex([2])
    expected = RangeIndex(2, 4, 2, name="foo")
    tm.assert_index_equal(result, expected, exact=True)

    expected_indexer = np.array([1], dtype=np.intp)
    tm.assert_numpy_array_equal(result_indexer, expected_indexer)


def test_reindex_empty_returns_rangeindex():
    # [FUNCTION DOC]
    # Reindexing to an empty list gives an empty range index with the same step and name and an empty indexer array.
    ri = RangeIndex(0, 10, 2, name="foo")
    result, result_indexer = ri.reindex([])
    expected = RangeIndex(0, 0, 2, name="foo")
    tm.assert_index_equal(result, expected, exact=True)

    expected_indexer = np.array([], dtype=np.intp)
    tm.assert_numpy_array_equal(result_indexer, expected_indexer)


def test_insert_empty_0_loc():
    # [FUNCTION DOC]
    # Inserting a value at position zero of an empty range index with a step of ten gives a one-element range with that step, preserving the name.
    ri = RangeIndex(0, step=10, name="foo")
    result = ri.insert(0, 5)
    expected = RangeIndex(5, 15, 10, name="foo")
    tm.assert_index_equal(result, expected, exact=True)


def test_append_non_rangeindex_return_rangeindex():
    # [FUNCTION DOC]
    # Appending a general integer index that continues the range yields a range index (exact).
    ri = RangeIndex(1)
    result = ri.append(Index([1]))
    expected = RangeIndex(2)
    tm.assert_index_equal(result, expected, exact=True)


def test_append_non_rangeindex_return_index():
    # [FUNCTION DOC]
    # Appending an integer index that does not form a regular progression yields a plain integer index (exact).
    ri = RangeIndex(1)
    result = ri.append(Index([1, 3, 4]))
    expected = Index([0, 1, 3, 4])
    tm.assert_index_equal(result, expected, exact=True)


def test_reindex_returns_rangeindex():
    # [FUNCTION DOC]
    # Reindexing to a contiguous label list partly outside the index gives a range index with indexer entries for found and missing labels (minus one for missing).
    ri = RangeIndex(2, name="foo")
    result, result_indexer = ri.reindex([1, 2, 3])
    expected = RangeIndex(1, 4, name="foo")
    tm.assert_index_equal(result, expected, exact=True)

    expected_indexer = np.array([1, -1, -1], dtype=np.intp)
    tm.assert_numpy_array_equal(result_indexer, expected_indexer)


def test_reindex_returns_index():
    # [FUNCTION DOC]
    # Reindexing to a non-regular label list gives a plain integer index and the matching positions as indexer.
    ri = RangeIndex(4, name="foo")
    result, result_indexer = ri.reindex([0, 1, 3])
    expected = Index([0, 1, 3], name="foo")
    tm.assert_index_equal(result, expected, exact=True)

    expected_indexer = np.array([0, 1, 3], dtype=np.intp)
    tm.assert_numpy_array_equal(result_indexer, expected_indexer)


def test_take_return_rangeindex():
    # [FUNCTION DOC]
    # Taking an empty list gives an empty range; taking two adjacent positions gives a range index; name preserved (exact type).
    ri = RangeIndex(5, name="foo")
    result = ri.take([])
    expected = RangeIndex(0, name="foo")
    tm.assert_index_equal(result, expected, exact=True)

    result = ri.take([3, 4])
    expected = RangeIndex(3, 5, name="foo")
    tm.assert_index_equal(result, expected, exact=True)


def test__getitem__boolean_numpyextensionarray():
    # [FUNCTION DOC]
    # A single-element range indexed by a boolean extension array of one true value returns an equal index.
    ri = RangeIndex(1)
    result = ri[pd.arrays.NumpyExtensionArray(np.array([True]))]
    tm.assert_index_equal(ri, result)


@pytest.mark.parametrize(
    "container",
    [np.array, pd.Series, lambda x: pd.arrays.NumpyExtensionArray(np.array(x))],
    ids=["numpy-array", "series", "numpy-extension-array"],
)
def test__getitem__boolean_arraylike(container):
    # [FUNCTION DOC]
    # Parametrized over container types (numpy array, series, extension array wrapper). Indexing a five-element range with a boolean mask in that container returns the selected positions as a 64-bit integer index.
    ri = RangeIndex(5)
    result = ri[container([True, True, False, False, True])]
    expected = Index([0, 1, 4], dtype="int64")
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "rng, exp_rng",
    [
        [range(5), range(3, 4)],
        [range(0, -10, -2), range(-6, -8, -2)],
        [range(0, 10, 2), range(6, 8, 2)],
    ],
)
def test_take_1_value_returns_rangeindex(rng, exp_rng):
    # [FUNCTION DOC]
    # Parametrized over ranges and expected one-element ranges. Taking one position yields a range index covering that element, with the name preserved.
    ri = RangeIndex(rng, name="foo")
    result = ri.take([3])
    expected = RangeIndex(exp_rng, name="foo")
    tm.assert_index_equal(result, expected, exact=True)


def test_append_one_nonempty_preserve_step():
    # [FUNCTION DOC]
    # Appending a non-empty descending range to an empty range index returns the appended range with its step preserved.
    expected = RangeIndex(0, -1, -1)
    result = RangeIndex(0).append([expected])
    tm.assert_index_equal(result, expected, exact=True)


def test_getitem_boolmask_all_true():
    # [FUNCTION DOC]
    # An all-true boolean mask returns an equal range index (copy semantics, exact type).
    ri = RangeIndex(3, name="foo")
    expected = ri.copy()
    result = ri[[True] * 3]
    tm.assert_index_equal(result, expected, exact=True)


def test_getitem_boolmask_all_false():
    # [FUNCTION DOC]
    # An all-false boolean mask returns an empty range index keeping the name.
    ri = RangeIndex(3, name="foo")
    result = ri[[False] * 3]
    expected = RangeIndex(0, name="foo")
    tm.assert_index_equal(result, expected, exact=True)


def test_getitem_boolmask_returns_rangeindex():
    # [FUNCTION DOC]
    # Masks selecting a contiguous tail or evenly spaced positions give range indexes with the right bounds/step.
    ri = RangeIndex(3, name="foo")
    result = ri[[False, True, True]]
    expected = RangeIndex(1, 3, name="foo")
    tm.assert_index_equal(result, expected, exact=True)

    result = ri[[True, False, True]]
    expected = RangeIndex(0, 3, 2, name="foo")
    tm.assert_index_equal(result, expected, exact=True)


def test_getitem_boolmask_returns_index():
    # [FUNCTION DOC]
    # A mask selecting irregular positions gives a plain integer index.
    ri = RangeIndex(4, name="foo")
    result = ri[[True, True, False, True]]
    expected = Index([0, 1, 3], name="foo")
    tm.assert_index_equal(result, expected)


def test_getitem_boolmask_wrong_length():
    # [FUNCTION DOC]
    # A boolean mask whose length differs from the index raises an index error saying the boolean index has the wrong length.
    ri = RangeIndex(4, name="foo")
    with pytest.raises(IndexError, match="Boolean index has wrong length"):
        ri[[True]]


def test_pos_returns_rangeindex():
    # [FUNCTION DOC]
    # Unary plus on a range index returns an equal range index (exact type).
    ri = RangeIndex(2, name="foo")
    expected = ri.copy()
    result = +ri
    tm.assert_index_equal(result, expected, exact=True)


def test_neg_returns_rangeindex():
    # [FUNCTION DOC]
    # Unary negation of non-negative and mixed-sign ranges returns a range index with negated, step-reversed bounds and keeps the name.
    ri = RangeIndex(2, name="foo")
    result = -ri
    expected = RangeIndex(0, -2, -1, name="foo")
    tm.assert_index_equal(result, expected, exact=True)

    ri = RangeIndex(-2, 2, name="foo")
    result = -ri
    expected = RangeIndex(2, -2, -1, name="foo")
    tm.assert_index_equal(result, expected, exact=True)


@pytest.mark.parametrize(
    "rng, exp_rng",
    [
        [range(0), range(0)],
        [range(10), range(10)],
        [range(-2, 1, 1), range(2, -1, -1)],
        [range(0, -10, -1), range(0, 10, 1)],
    ],
)
def test_abs_returns_rangeindex(rng, exp_rng):
    # [FUNCTION DOC]
    # Parametrized over ranges whose absolute value is still a regular range (empty, non-negative, or monotonic sign-changing ending at zero). abs gives the expected range index with name.
    ri = RangeIndex(rng, name="foo")
    expected = RangeIndex(exp_rng, name="foo")
    result = abs(ri)
    tm.assert_index_equal(result, expected, exact=True)


def test_abs_returns_index():
    # [FUNCTION DOC]
    # Absolute value of a range that straddles zero gives a plain integer index of non-monotonic values.
    ri = RangeIndex(-2, 2, name="foo")
    result = abs(ri)
    expected = Index([2, 1, 0, 1], name="foo")
    tm.assert_index_equal(result, expected, exact=True)


@pytest.mark.parametrize(
    "rng",
    [
        range(0),
        range(5),
        range(0, -5, -1),
        range(-2, 2, 1),
        range(2, -2, -2),
        range(0, 5, 2),
    ],
)
def test_invert_returns_rangeindex(rng):
    # [FUNCTION DOC]
    # Parametrized over several ranges. Bitwise inversion returns a range index; its values equal the inversion of the equivalent integer index (loose type check).
    ri = RangeIndex(rng, name="foo")
    result = ~ri
    assert isinstance(result, RangeIndex)
    expected = ~Index(list(rng), name="foo")
    tm.assert_index_equal(result, expected, exact=False)


@pytest.mark.parametrize(
    "rng",
    [
        range(0, 5, 1),
        range(0, 5, 2),
        range(10, 15, 1),
        range(10, 5, -1),
        range(10, 5, -2),
        range(5, 0, -1),
    ],
)
@pytest.mark.parametrize("meth", ["argmax", "argmin"])
def test_arg_min_max(rng, meth):
    # [FUNCTION DOC]
    # Parametrized over ranges and the method name (argmax or argmin). The result for a range index equals the result for the equivalent integer index.
    ri = RangeIndex(rng)
    idx = Index(list(rng))
    assert getattr(ri, meth)() == getattr(idx, meth)()


@pytest.mark.parametrize("meth", ["argmin", "argmax"])
def test_empty_argmin_argmax_raises(meth):
    # [FUNCTION DOC]
    # Parametrized over argmin and argmax. On an empty range index each raises a value error saying it cannot get that reduction of an empty sequence.
    with pytest.raises(ValueError, match=f"attempt to get {meth} of an empty sequence"):
        getattr(RangeIndex(0), meth)()


def test_getitem_integers_return_rangeindex():
    # [FUNCTION DOC]
    # Integer-list getitem whose positions form an arithmetic progression (first and last, or a single one) returns a range index; name preserved.
    result = RangeIndex(0, 10, 2, name="foo")[[0, -1]]
    expected = RangeIndex(start=0, stop=16, step=8, name="foo")
    tm.assert_index_equal(result, expected, exact=True)

    result = RangeIndex(0, 10, 2, name="foo")[[3]]
    expected = RangeIndex(start=6, stop=8, step=2, name="foo")
    tm.assert_index_equal(result, expected, exact=True)


def test_getitem_empty_return_rangeindex():
    # [FUNCTION DOC]
    # Indexing with an empty list returns an empty range with step one and the name.
    result = RangeIndex(0, 10, 2, name="foo")[[]]
    expected = RangeIndex(start=0, stop=0, step=1, name="foo")
    tm.assert_index_equal(result, expected, exact=True)


def test_getitem_integers_return_index():
    # [FUNCTION DOC]
    # Integer-list getitem with irregular positions returns a plain 64-bit integer index with the name.
    result = RangeIndex(0, 10, 2, name="foo")[[0, 1, -1]]
    expected = Index([0, 2, 8], dtype="int64", name="foo")
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize(
    "rng",
    [
        range(3),
        range(0),
        range(0, 3, 2),
        range(3, -3, -2),
    ],
)
def test_value_counts(sort, dropna, ascending, normalize, rng):
    # [FUNCTION DOC]
    # Parametrized over several ranges, the normalize flag, and the sort, dropna and ascending fixtures. value_counts on a named range index must equal value_counts on the equivalent integer index (ignoring index type).
    ri = RangeIndex(rng, name="A")
    result = ri.value_counts(
        normalize=normalize, sort=sort, ascending=ascending, dropna=dropna
    )
    expected = Index(list(rng), name="A").value_counts(
        normalize=normalize, sort=sort, ascending=ascending, dropna=dropna
    )
    tm.assert_series_equal(result, expected, check_index_type=False)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("value", [0, -5, 5, -3, np.array([-5, -3, 0, 5])])
def test_searchsorted(side, value):
    # [FUNCTION DOC]
    # Parametrized over side (left or right) and values (scalars, and an array). searchsorted on a range index equals the same on the integer index; scalar results compared directly, array results with numpy equality.
    ri = RangeIndex(-3, 3, 2)
    result = ri.searchsorted(value=value, side=side)
    expected = Index(list(ri)).searchsorted(value=value, side=side)
    if isinstance(value, int):
        assert result == expected
    else:
        tm.assert_numpy_array_equal(result, expected)
