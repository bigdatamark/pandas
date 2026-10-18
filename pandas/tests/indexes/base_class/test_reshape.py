"""
Tests for ndarray-like method on the base Index class
"""
# [MODULE DOC]
# Tests for ndarray-like reshaping methods (repeat, insert, delete, append) on the base Index.

import numpy as np
import pytest

from pandas import Index
import pandas._testing as tm


class TestReshape:
    # [CLASS DOC]
    # Groups reshape-style method tests for the base Index; no state.
    def test_repeat(self):
        # [FUNCTION DOC]
        # Verifies that repeating each element twice yields each label duplicated consecutively.
        repeats = 2
        index = Index([1, 2, 3])
        expected = Index([1, 1, 2, 2, 3, 3])

        result = index.repeat(repeats)
        tm.assert_index_equal(result, expected)

    def test_insert(self):
        # [FUNCTION DOC]
        # Verifies insert at the front, at negative position (same as list semantics), equivalence of
        # positive and negative positions, and insertion into an empty Index.
        # GH 7256
        # validate neg/pos inserts
        result = Index(["b", "c", "d"])

        # test 0th element
        tm.assert_index_equal(Index(["a", "b", "c", "d"]), result.insert(0, "a"))

        # test Nth element that follows Python list behavior
        tm.assert_index_equal(Index(["b", "c", "e", "d"]), result.insert(-1, "e"))

        # test loc +/- neq (0, -1)
        tm.assert_index_equal(result.insert(1, "z"), result.insert(-2, "z"))

        # test empty
        null_index = Index([])
        tm.assert_index_equal(Index(["a"]), null_index.insert(0, "a"))

    def test_insert_missing(self, nulls_fixture, using_infer_string):
        # [FUNCTION DOC]
        # Uses the nulls_fixture. Verifies inserting a null value into an object Index keeps the exact
        # null object (no conversion of NA flavors).
        # GH#22295
        # test there is no mangling of NA values
        expected = Index(["a", nulls_fixture, "b", "c"], dtype=object)
        result = Index(list("abc"), dtype=object).insert(
            1, Index([nulls_fixture], dtype=object)
        )
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "val", [(1, 2), np.datetime64("2019-12-31"), np.timedelta64(1, "D")]
    )
    @pytest.mark.parametrize("loc", [-1, 2])
    def test_insert_datetime_into_object(self, loc, val):
        # [FUNCTION DOC]
        # Parametrized over location and value (tuple, datetime64, timedelta64). Verifies inserting
        # into a string Index gives an object Index preserving the inserted object type.
        # GH#44509
        idx = Index(["1", "2", "3"])
        result = idx.insert(loc, val)
        expected = Index(["1", "2", val, "3"])
        tm.assert_index_equal(result, expected)
        assert type(expected[2]) is type(val)

    def test_insert_none_into_string_numpy(self, string_dtype_no_object):
        # [FUNCTION DOC]
        # Uses the string_dtype_no_object fixture. Verifies inserting None into a string-dtype Index
        # keeps the string dtype with a missing value.
        # GH#55365
        index = Index(["a", "b", "c"], dtype=string_dtype_no_object)
        result = index.insert(-1, None)
        expected = Index(["a", "b", None, "c"], dtype=string_dtype_no_object)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "pos,expected",
        [
            (0, Index(["b", "c", "d"], name="index")),
            (-1, Index(["a", "b", "c"], name="index")),
        ],
    )
    def test_delete(self, pos, expected):
        # [FUNCTION DOC]
        # Parametrized over position (first, last) and expected Index. Verifies delete drops the
        # element and keeps the name.
        index = Index(["a", "b", "c", "d"], name="index")
        result = index.delete(pos)
        tm.assert_index_equal(result, expected)
        assert result.name == expected.name

    def test_delete_raises(self):
        # [FUNCTION DOC]
        # Verifies deleting a position past the end raises an index error stating the index is out of
        # bounds for the size.
        index = Index(["a", "b", "c", "d"], name="index")
        msg = "index 5 is out of bounds for axis 0 with size 4"
        with pytest.raises(IndexError, match=msg):
            index.delete(5)

    def test_append_multiple(self):
        # [FUNCTION DOC]
        # Verifies appending a list of Indexes reconstructs the original, and appending an empty list
        # returns an equal Index.
        index = Index(["a", "b", "c", "d", "e", "f"])

        foos = [index[:2], index[2:4], index[4:]]
        result = foos[0].append(foos[1:])
        tm.assert_index_equal(result, index)

        # empty
        result = index.append([])
        tm.assert_index_equal(result, index)
