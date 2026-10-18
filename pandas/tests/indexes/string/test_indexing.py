# [MODULE DOC]
# Tests for label-based lookup on string-dtype Index objects: single-label
# location, indexer computation (unique and non-unique, with fill methods and
# missing-value handling), and slice location including negative steps.
# All tests are parametrized over the any_string_dtype fixture, so they run
# for every string storage variant. Contains two private helpers for comparing
# missing-value sentinels.

import numpy as np
import pytest

import pandas as pd
from pandas import Index
import pandas._testing as tm


def _isnan(val):
    # [FUNCTION DOC]
    # HELPER: null-check utility for the missing-value tests in this module.
    # Takes a scalar val; returns true when it is a floating-point NaN and not the
    # pandas NA singleton, false otherwise (including when the numeric NaN check
    # raises a type error for non-numeric input).
    try:
        return val is not pd.NA and np.isnan(val)
    except TypeError:
        return False


def _equivalent_na(dtype, null):
    # [FUNCTION DOC]
    # HELPER: decides whether a null value is the same kind of missing value that a
    # string dtype uses natively. Takes a dtype object (needs its na_value
    # attribute) and a null scalar. Returns true if both are the pandas NA
    # singleton, or if both are NaN-like; false otherwise. Used to predict whether
    # lookups of that null in a string index will succeed.
    if dtype.na_value is pd.NA and null is pd.NA:
        return True
    elif _isnan(dtype.na_value) and _isnan(null):
        return True
    else:
        return False


class TestGetLoc:
    # [CLASS DOC]
    # Tests for single-label location lookup on string-dtype indexes.
    def test_get_loc(self, any_string_dtype):
        # [FUNCTION DOC]
        # Location of an existing label in a unique string index returns its integer position.
        index = Index(["a", "b", "c"], dtype=any_string_dtype)
        assert index.get_loc("b") == 1

    def test_get_loc_raises(self, any_string_dtype):
        # [FUNCTION DOC]
        # Looking up an absent label raises a key error whose message names the label.
        index = Index(["a", "b", "c"], dtype=any_string_dtype)
        with pytest.raises(KeyError, match="d"):
            index.get_loc("d")

    def test_get_loc_invalid_value(self, any_string_dtype):
        # [FUNCTION DOC]
        # Looking up a non-string (integer) label in a string index raises a key error.
        index = Index(["a", "b", "c"], dtype=any_string_dtype)
        with pytest.raises(KeyError, match="1"):
            index.get_loc(1)

    def test_get_loc_non_unique(self, any_string_dtype):
        # [FUNCTION DOC]
        # For an index with a repeated label, the location lookup returns a boolean
        # mask marking every matching position.
        index = Index(["a", "b", "a"], dtype=any_string_dtype)
        result = index.get_loc("a")
        expected = np.array([True, False, True])
        tm.assert_numpy_array_equal(result, expected)

    def test_get_loc_non_missing(self, any_string_dtype, nulls_fixture):
        # [FUNCTION DOC]
        # Parametrized over the null fixture: looking up any null value in an index
        # that contains no nulls raises a key error.
        index = Index(["a", "b", "c"], dtype=any_string_dtype)
        with pytest.raises(KeyError):
            index.get_loc(nulls_fixture)

    def test_get_loc_missing(self, any_string_dtype, nulls_fixture):
        # [FUNCTION DOC]
        # Parametrized over the null fixture: an index that has the null value as its
        # last element finds it at position two.
        index = Index(["a", "b", nulls_fixture], dtype=any_string_dtype)
        assert index.get_loc(nulls_fixture) == 2


class TestGetIndexer:
    # [CLASS DOC]
    # Tests for computing integer indexers (with fill methods) on string indexes.
    @pytest.mark.parametrize(
        "method,expected",
        [
            ("pad", [-1, 0, 1, 1]),
            ("backfill", [0, 0, 1, -1]),
        ],
    )
    def test_get_indexer_strings(self, any_string_dtype, method, expected):
        # [FUNCTION DOC]
        # Parametrized over the fill method (forward fill gives minus one, zero, one, one;
        # backward fill gives zero, zero, one, minus one). Builds a two-label index and
        # asks for an indexer for four labels with the given method; the result must
        # equal the expected array of platform integers.
        expected = np.array(expected, dtype=np.intp)
        index = Index(["b", "c"], dtype=any_string_dtype)
        actual = index.get_indexer(["a", "b", "c", "d"], method=method)

        tm.assert_numpy_array_equal(actual, expected)

    def test_get_indexer_strings_raises(self, any_string_dtype):
        # [FUNCTION DOC]
        # Nearest-neighbour matching, or a tolerance (scalar or list-like), on a string
        # index must raise a type error because subtraction is not supported between
        # strings. The accepted message is either the dtype-specific one or the generic
        # Python operand-type one.
        index = Index(["b", "c"], dtype=any_string_dtype)

        msg = "|".join(
            [
                "operation 'sub' not supported for dtype 'str",
                r"unsupported operand type\(s\) for -: 'str' and 'str'",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            index.get_indexer(["a", "b", "c", "d"], method="nearest")

        with pytest.raises(TypeError, match=msg):
            index.get_indexer(["a", "b", "c", "d"], method="pad", tolerance=2)

        with pytest.raises(TypeError, match=msg):
            index.get_indexer(
                ["a", "b", "c", "d"], method="pad", tolerance=[2, 2, 2, 2]
            )

    @pytest.mark.parametrize("null", [None, np.nan, float("nan"), pd.NA])
    def test_get_indexer_missing(self, any_string_dtype, null, using_infer_string):
        # [FUNCTION DOC]
        # Parametrized over several null values (None, NaN, float NaN, NA). Builds an
        # index with the null as its last label and looks up a list containing a label,
        # the null, and an absent label. Expected result depends on configuration: when
        # string inference is on, or the null matches the dtype's native missing value,
        # the null is found at position two; for the nullable string dtype with a
        # non-matching null it is not found (minus one). The absent label is always
        # minus one.
        # NaT and Decimal("NaN") from null_fixture are not supported for string dtype
        index = Index(["a", "b", null], dtype=any_string_dtype)
        result = index.get_indexer(["a", null, "c"])
        if using_infer_string:
            expected = np.array([0, 2, -1], dtype=np.intp)
        elif any_string_dtype == "string" and not _equivalent_na(
            any_string_dtype, null
        ):
            expected = np.array([0, -1, -1], dtype=np.intp)
        else:
            expected = np.array([0, 2, -1], dtype=np.intp)

        tm.assert_numpy_array_equal(result, expected)


class TestGetIndexerNonUnique:
    # [CLASS DOC]
    # Tests for the non-unique indexer computation on string indexes with nulls.
    @pytest.mark.parametrize("null", [None, np.nan, float("nan"), pd.NA])
    def test_get_indexer_non_unique_nas(
        self, any_string_dtype, null, using_infer_string
    ):
        # [FUNCTION DOC]
        # Parametrized over several null values. First part: unique index with the null
        # last; looks up a label and the null, checking both the indexer and the
        # missing-positions array. Found nulls give positions zero and two with no
        # missing entries; for the nullable string dtype with a non-matching null the
        # indexer has minus one and the missing array reports position one.
        # Second part: an index in which the null appears twice; expects positions zero,
        # one and three.
        # In the nullable-dtype non-matching case the expected values from the first part are reused unchanged (null not found).
        index = Index(["a", "b", null], dtype=any_string_dtype)
        indexer, missing = index.get_indexer_non_unique(["a", null])

        if using_infer_string:
            expected_indexer = np.array([0, 2], dtype=np.intp)
            expected_missing = np.array([], dtype=np.intp)
        elif any_string_dtype == "string" and not _equivalent_na(
            any_string_dtype, null
        ):
            expected_indexer = np.array([0, -1], dtype=np.intp)
            expected_missing = np.array([1], dtype=np.intp)
        else:
            expected_indexer = np.array([0, 2], dtype=np.intp)
            expected_missing = np.array([], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected_indexer)
        tm.assert_numpy_array_equal(missing, expected_missing)

        # actually non-unique
        index = Index(["a", null, "b", null], dtype=any_string_dtype)
        indexer, missing = index.get_indexer_non_unique(["a", null])

        if using_infer_string:
            expected_indexer = np.array([0, 1, 3], dtype=np.intp)
        elif any_string_dtype == "string" and not _equivalent_na(
            any_string_dtype, null
        ):
            pass
        else:
            expected_indexer = np.array([0, 1, 3], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected_indexer)
        tm.assert_numpy_array_equal(missing, expected_missing)


class TestSliceLocs:
    # [CLASS DOC]
    # Tests for locating slice bounds on string indexes.
    @pytest.mark.parametrize(
        "in_slice,expected",
        [
            # error: Slice index must be an integer or None
            (pd.IndexSlice[::-1], "yxdcb"),
            (pd.IndexSlice["b":"y":-1], ""),  # type: ignore[misc]
            (pd.IndexSlice["b"::-1], "b"),  # type: ignore[misc]
            (pd.IndexSlice[:"b":-1], "yxdcb"),  # type: ignore[misc]
            (pd.IndexSlice[:"y":-1], "y"),  # type: ignore[misc]
            (pd.IndexSlice["y"::-1], "yxdcb"),  # type: ignore[misc]
            (pd.IndexSlice["y"::-4], "yb"),  # type: ignore[misc]
            # absent labels
            (pd.IndexSlice[:"a":-1], "yxdcb"),  # type: ignore[misc]
            (pd.IndexSlice[:"a":-2], "ydb"),  # type: ignore[misc]
            (pd.IndexSlice["z"::-1], "yxdcb"),  # type: ignore[misc]
            (pd.IndexSlice["z"::-3], "yc"),  # type: ignore[misc]
            (pd.IndexSlice["m"::-1], "dcb"),  # type: ignore[misc]
            (pd.IndexSlice[:"m":-1], "yx"),  # type: ignore[misc]
            (pd.IndexSlice["a":"a":-1], ""),  # type: ignore[misc]
            (pd.IndexSlice["z":"z":-1], ""),  # type: ignore[misc]
            (pd.IndexSlice["m":"m":-1], ""),  # type: ignore[misc]
        ],
    )
    def test_slice_locs_negative_step(self, in_slice, expected, any_string_dtype):
        # [FUNCTION DOC]
        # Parametrized over slice objects (including absent bounds and various negative
        # steps) and the expected resulting labels. Uses the slice-location method of
        # an index of five sorted letters with the slice start, stop and step, slices
        # the index with the returned bounds and the step, and compares with an index
        # of the expected letters in the same dtype.
        index = Index(list("bcdxy"), dtype=any_string_dtype)

        s_start, s_stop = index.slice_locs(in_slice.start, in_slice.stop, in_slice.step)
        result = index[s_start : s_stop : in_slice.step]
        expected = Index(list(expected), dtype=any_string_dtype)
        tm.assert_index_equal(result, expected)

    def test_slice_locs_negative_step_oob(self, any_string_dtype):
        # [FUNCTION DOC]
        # Out-of-bounds integer slices still behave: a slice with a very negative
        # start and positive step returns the full index; a negative step with a very
        # negative stop returns the whole index reversed.
        index = Index(list("bcdxy"), dtype=any_string_dtype)

        result = index[-10:5:1]
        tm.assert_index_equal(result, index)

        result = index[4:-10:-1]
        expected = Index(list("yxdcb"), dtype=any_string_dtype)
        tm.assert_index_equal(result, expected)

    def test_slice_locs_dup(self, any_string_dtype):
        # [FUNCTION DOC]
        # For a sorted index with duplicates, slice location returns the leftmost start
        # and rightmost stop positions for labels present (checked with start and
        # stop, and stop-only), and the same for the reversed index with reversed
        # bounds.
        index = Index(["a", "a", "b", "c", "d", "d"], dtype=any_string_dtype)
        assert index.slice_locs("a", "d") == (0, 6)
        assert index.slice_locs(end="d") == (0, 6)
        assert index.slice_locs("a", "c") == (0, 4)
        assert index.slice_locs("b", "d") == (2, 6)

        index2 = index[::-1]
        assert index2.slice_locs("d", "a") == (0, 6)
        assert index2.slice_locs(end="a") == (0, 6)
        assert index2.slice_locs("d", "b") == (0, 4)
        assert index2.slice_locs("c", "a") == (2, 6)
