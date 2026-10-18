"""
test_indexing tests the following Index methods:
    __getitem__
    get_loc
    get_value
    __contains__
    take
    where
    get_indexer
    get_indexer_for
    slice_locs
    asof_locs

The corresponding tests.indexes.[index_type].test_indexing files
contain tests for the corresponding methods specific to those Index subclasses.
"""

# [MODULE DOC]
# Generic indexing tests for the base Index API: element selection by position (take), membership, label location, indexer computation with missing values and duplicates, slice conversion, mask-based replacement, and float-scalar getitem. Most tests use the shared index fixture so they run on every index type; subclass-specific indexing tests live in the per-type test packages. Grouped into one class per method family plus a few module-level tests.

import numpy as np
import pytest

from pandas.compat import PY314
from pandas.errors import InvalidIndexError

from pandas.core.dtypes.common import (
    is_float_dtype,
    is_scalar,
)

from pandas import (
    NA,
    DatetimeIndex,
    Index,
    IntervalIndex,
    MultiIndex,
    NaT,
    PeriodIndex,
    TimedeltaIndex,
)
import pandas._testing as tm


class TestTake:
    # [CLASS DOC]
    # Tests for taking elements by position.
    def test_take_invalid_kwargs(self, index):
        # [FUNCTION DOC]
        # Taking with an unknown keyword raises a type error naming it; passing the numpy-style output or mode parameters raises a value error saying they are not supported.
        indices = [1, 2]

        msg = r"take\(\) got an unexpected keyword argument 'foo'"
        with pytest.raises(TypeError, match=msg):
            index.take(indices, foo=2)

        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            index.take(indices, out=indices)

        msg = "the 'mode' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            index.take(indices, mode="clip")

    def test_take(self, index):
        # [FUNCTION DOC]
        # Skips indexes with fewer than five elements. Taking positions four, three, zero, two equals plain list indexing. For non-datetime-like indexes, reading a frequency attribute raises an attribute error.
        indexer = [4, 3, 0, 2]
        if len(index) < 5:
            pytest.skip("Test doesn't make sense since not enough elements")

        result = index.take(indexer)
        expected = index[indexer]
        assert result.equals(expected)

        if not isinstance(index, (DatetimeIndex, PeriodIndex, TimedeltaIndex)):
            # GH 10791
            msg = r"'(.*Index)' object has no attribute 'freq'"
            with pytest.raises(AttributeError, match=msg):
                index.freq

    def test_take_indexer_type(self):
        # [FUNCTION DOC]
        # Taking with a scalar instead of an array-like raises a type error saying that indices are expected to be array-like.
        # GH#42875
        integer_index = Index([0, 1, 2, 3])
        scalar_index = 1
        msg = "Expected indices to be array-like"
        with pytest.raises(TypeError, match=msg):
            integer_index.take(scalar_index)

    def test_take_minus1_without_fill(self, index):
        # [FUNCTION DOC]
        # Skips empty indexes. Without enabling fill, a position of minus one means the last element.
        # -1 does not get treated as NA unless allow_fill=True is passed
        if len(index) == 0:
            # Test is not applicable
            pytest.skip("Test doesn't make sense for empty index")

        result = index.take([0, 0, -1])

        expected = index.take([0, 0, len(index) - 1])
        tm.assert_index_equal(result, expected)


class TestContains:
    # [CLASS DOC]
    # Tests for membership testing.
    @pytest.mark.parametrize(
        "index,val",
        [
            ([0, 1, 2], 2),
            ([0, 1, "2"], "2"),
            ([0, 1, 2, np.inf, 4], 4),
            ([0, 1, 2, np.nan, 4], 4),
            ([0, 1, 2, np.inf], np.inf),
            ([0, 1, 2, np.nan], np.nan),
        ],
    )
    def test_index_contains(self, index, val):
        # [FUNCTION DOC]
        # Parametrized over (values, value). The listed values (including infinity or NaN) are found in the index.
        index = Index(index)
        assert val in index

    @pytest.mark.parametrize(
        "index,val",
        [
            (Index([0, 1, 2]), "2"),
            (Index([0, 1, "2"]), 2),
            (Index([0, 1, 2, np.inf]), 4),
            (Index([0, 1, 2, np.nan]), 4),
            (Index([0, 1, 2, np.inf]), np.nan),
            (Index([0, 1, 2, np.nan]), np.inf),
            # Checking if np.inf in int64 Index should not cause an OverflowError
            # Related to GH 16957
            (Index([0, 1, 2], dtype=np.int64), np.inf),
            (Index([0, 1, 2], dtype=np.int64), np.nan),
            (Index([0, 1, 2], dtype=np.uint64), np.inf),
            (Index([0, 1, 2], dtype=np.uint64), np.nan),
        ],
    )
    def test_index_not_contains(self, index, val):
        # [FUNCTION DOC]
        # Parametrized over (index, value). Strings versus integers, absent numbers, infinity versus NaN, and infinity or NaN in integer-typed indexes (signed and unsigned) are not found and must not overflow.
        assert val not in index

    @pytest.mark.parametrize("val", [0, "2"])
    def test_mixed_index_contains(self, val):
        # [FUNCTION DOC]
        # Parametrized over zero and the string "2". Both are members of a mixed integer/string index.
        # GH#19860
        index = Index([0, 1, "2"])
        assert val in index

    @pytest.mark.parametrize("val", ["1", 2])
    def test_mixed_index_not_contains(self, index, val):
        # [FUNCTION DOC]
        # KNOWN ISSUE: requests the shared index fixture but immediately overwrites it with a local mixed index, so the test body is repeated needlessly for every index type.
        # Parametrized over "1" and 2. Neither is a member of a mixed integer/string index.
        # GH#19860
        index = Index([0, 1, "2"])
        assert val not in index

    def test_contains_with_float_index(self, any_real_numpy_dtype):
        # [FUNCTION DOC]
        # Uses the real numpy dtype fixture. With float dtypes the index holds 0.1, 1.1, 2.2, 3.3 and contains 1.1 but not 1.0 or 1; for integer dtypes it holds 0 to 3 and contains 1.0 and 1 but not 1.1.
        # GH#22085
        dtype = any_real_numpy_dtype
        data = [0, 1, 2, 3] if not is_float_dtype(dtype) else [0.1, 1.1, 2.2, 3.3]
        index = Index(data, dtype=dtype)

        if not is_float_dtype(index.dtype):
            assert 1.1 not in index
            assert 1.0 in index
            assert 1 in index
        else:
            assert 1.1 in index
            assert 1.0 not in index
            assert 1 not in index

    def test_contains_requires_hashable_raises(self, index):
        # [FUNCTION DOC]
        # Returns early for multi-level indexes. Testing membership of an unhashable list raises a type error ("unhashable type"). Testing an empty dictionary against the private engine raises a type error whose message varies by engine type (several accepted patterns, one depending on whether the Python version is at least 3.14).
        if isinstance(index, MultiIndex):
            return  # TODO: do we want this to raise?

        msg = "unhashable type: 'list'"
        with pytest.raises(TypeError, match=msg):
            [] in index

        if PY314:
            container_or_iterable = "a container or iterable"
        else:
            container_or_iterable = "iterable"

        msg = "|".join(
            [
                r"unhashable type: 'dict'",
                r"must be real number, not dict",
                r"an integer is required",
                r"\{\}",
                r"pandas\._libs\.interval\.IntervalTree' is not "
                f"{container_or_iterable}",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            {} in index._engine


class TestGetLoc:
    # [CLASS DOC]
    # Tests for single-label location.
    def test_get_loc_non_hashable(self, index):
        # [FUNCTION DOC]
        # KNOWN ISSUE: the expected message pattern is a regular-expression character class matching any single one of the characters zero, one, space or comma rather than the literal list text, so it is a very weak message check.
        # Looking up a list raises the invalid-index error.
        with pytest.raises(InvalidIndexError, match="[0, 1]"):
            index.get_loc([0, 1])

    def test_get_loc_non_scalar_hashable(self, index):
        # [FUNCTION DOC]
        # Defines a local enumeration whose member is hashable but not a scalar. Looking it up raises a key error mentioning the member for most index types, but for datetime, timedelta, period and interval indexes raises the invalid-index error instead.
        # GH52877
        from enum import Enum

        class E(Enum):
            # [CLASS DOC]
            # HELPER: local enumeration with a single member, used as a hashable non-scalar key.
            X1 = "x1"

        assert not is_scalar(E.X1)

        exc = KeyError
        msg = "<E.X1: 'x1'>"
        if isinstance(
            index,
            (
                DatetimeIndex,
                TimedeltaIndex,
                PeriodIndex,
                IntervalIndex,
            ),
        ):
            # TODO: make these more consistent?
            exc = InvalidIndexError
            msg = "E.X1"
        with pytest.raises(exc, match=msg):
            index.get_loc(E.X1)

    def test_get_loc_generator(self, index):
        # [FUNCTION DOC]
        # Looking up a generator object raises a key error for most indexes, or the invalid-index error for datetime-like, interval and multi-level indexes; the message mentions a generator object.
        exc = KeyError
        if isinstance(
            index,
            (
                DatetimeIndex,
                TimedeltaIndex,
                PeriodIndex,
                IntervalIndex,
                MultiIndex,
            ),
        ):
            # TODO: make these more consistent?
            exc = InvalidIndexError
        with pytest.raises(exc, match="generator object"):
            # MultiIndex specifically checks for generator; others for scalar
            index.get_loc(x for x in range(5))

    def test_get_loc_masked_duplicated_na(self):
        # [FUNCTION DOC]
        # In a nullable-integer index with two NA values, looking up NA returns a boolean mask marking both.
        # GH#48411
        idx = Index([1, 2, NA, NA], dtype="Int64")
        result = idx.get_loc(NA)
        expected = np.array([False, False, True, True])
        tm.assert_numpy_array_equal(result, expected)


class TestGetIndexer:
    # [CLASS DOC]
    # Tests for computing integer indexers.
    def test_get_indexer_base(self, index):
        # [FUNCTION DOC]
        # For unique indexes, the indexer of an index against itself is the range of positions as platform integers; for non-unique indexes it raises the invalid-index error stating that reindexing is only valid with uniquely valued indexes. An unknown fill method raises a value error.
        if index._index_as_unique:
            expected = np.arange(index.size, dtype=np.intp)
            actual = index.get_indexer(index)
            tm.assert_numpy_array_equal(expected, actual)
        else:
            msg = "Reindexing only valid with uniquely valued Index objects"
            with pytest.raises(InvalidIndexError, match=msg):
                index.get_indexer(index)

        with pytest.raises(ValueError, match="Invalid fill method"):
            index.get_indexer(index, method="invalid")

    def test_get_indexer_consistency(self, index):
        # [FUNCTION DOC]
        # The indexer for the first two elements is a platform-integer array when the index is unique and raises for non-unique indexes; the non-unique variant of the indexer method always returns a platform-integer array as its first result.
        # See GH#16819

        if index._index_as_unique:
            indexer = index.get_indexer(index[0:2])
            assert isinstance(indexer, np.ndarray)
            assert indexer.dtype == np.intp
        else:
            msg = "Reindexing only valid with uniquely valued Index objects"
            with pytest.raises(InvalidIndexError, match=msg):
                index.get_indexer(index[0:2])

        indexer, _ = index.get_indexer_non_unique(index[0:2])
        assert isinstance(indexer, np.ndarray)
        assert indexer.dtype == np.intp

    def test_get_indexer_masked_duplicated_na(self):
        # [FUNCTION DOC]
        # In a nullable-integer index with duplicate NA values, the for-indexer method with targets one and NA returns positions zero, two, three.
        # GH#48411
        idx = Index([1, 2, NA, NA], dtype="Int64")
        result = idx.get_indexer_for(Index([1, NA], dtype="Int64"))
        expected = np.array([0, 2, 3], dtype=result.dtype)
        tm.assert_numpy_array_equal(result, expected)


class TestConvertSliceIndexer:
    # [CLASS DOC]
    # Tests for the private slice converter.
    def test_convert_almost_null_slice(self, index):
        # [FUNCTION DOC]
        # With a slice that has no start or stop but a string step: for interval indexes raises a value error (label-based slicing with step other than one unsupported); for other indexes raises a type error about comparing string with integer.
        # slice with None at both ends, but not step

        key = slice(None, None, "foo")

        if isinstance(index, IntervalIndex):
            msg = "label-based slicing with step!=1 is not supported for IntervalIndex"
            with pytest.raises(ValueError, match=msg):
                index._convert_slice_indexer(key, "loc")
        else:
            msg = "'>=' not supported between instances of 'str' and 'int'"
            with pytest.raises(TypeError, match=msg):
                index._convert_slice_indexer(key, "loc")


class TestPutmask:
    # [CLASS DOC]
    # Tests for mask-based replacement.
    def test_putmask_with_wrong_mask(self, index):
        # [FUNCTION DOC]
        # Skips empty indexes. A boolean mask that is longer or shorter than the index, or a non-array mask, raises a value error saying mask and data must be the same size.
        # GH#18368
        if not len(index):
            pytest.skip("Test doesn't make sense for empty index")

        fill = index[0]

        msg = "putmask: mask and data must be the same size"
        with pytest.raises(ValueError, match=msg):
            index.putmask(np.ones(len(index) + 1, np.bool_), fill)

        with pytest.raises(ValueError, match=msg):
            index.putmask(np.ones(len(index) - 1, np.bool_), fill)

        with pytest.raises(ValueError, match=msg):
            index.putmask("foo", fill)


@pytest.mark.parametrize("idx", [[1, 2, 3], [0.1, 0.2, 0.3], ["a", "b", "c"]])
def test_getitem_deprecated_float(idx):
    # [FUNCTION DOC]
    # Parametrized over integer, float and string index contents. Indexing with a float scalar raises an index error saying that indexing with a float is no longer supported.
    # https://github.com/pandas-dev/pandas/issues/34191
    idx = Index(idx)
    msg = "Indexing with a float is no longer supported"
    with pytest.raises(IndexError, match=msg):
        idx[1.0]


@pytest.mark.parametrize(
    "idx,target,expected",
    [
        ([np.nan, "var1", np.nan], [np.nan], np.array([0, 2], dtype=np.intp)),
        (
            [np.nan, "var1", np.nan],
            [np.nan, "var1"],
            np.array([0, 2, 1], dtype=np.intp),
        ),
        (
            np.array([np.nan, "var1", np.nan], dtype=object),
            [np.nan],
            np.array([0, 2], dtype=np.intp),
        ),
        (
            DatetimeIndex(["2020-08-05", NaT, NaT]),
            [NaT],
            np.array([1, 2], dtype=np.intp),
        ),
        (["a", "b", "a", np.nan], [np.nan], np.array([3], dtype=np.intp)),
        (
            np.array(["b", np.nan, float("NaN"), "b"], dtype=object),
            Index([np.nan], dtype=object),
            np.array([1, 2], dtype=np.intp),
        ),
    ],
)
def test_get_indexer_non_unique_multiple_nans(idx, target, expected):
    # [FUNCTION DOC]
    # Parametrized over (index values, target, expected indexer). The for-indexer method matches all missing values (NaN or NaT) in the index to the target missing value, including combined with other labels.
    # GH 35392
    axis = Index(idx)
    actual = axis.get_indexer_for(target)
    tm.assert_numpy_array_equal(actual, expected)


def test_get_indexer_non_unique_nans_in_object_dtype_target(nulls_fixture):
    # [FUNCTION DOC]
    # Uses the nulls fixture. An object-dtype target holding one and a missing value against a float index of 1.0 and 2.0 yields indexer zero and minus one and a missing-positions array of one.
    idx = Index([1.0, 2.0])
    target = Index([1, nulls_fixture], dtype="object")

    result_idx, result_missing = idx.get_indexer_non_unique(target)
    tm.assert_numpy_array_equal(result_idx, np.array([0, -1], dtype=np.intp))
    tm.assert_numpy_array_equal(result_missing, np.array([1], dtype=np.intp))
