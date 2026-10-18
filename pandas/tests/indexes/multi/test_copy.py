# [MODULE DOC]
# Tests for copying MultiIndex objects: copy, shallow copy, view, the copy
# module's copy and deepcopy, copy keyword arguments, and identity retention.
# Uses the shared idx fixture. Pure tests module.

from copy import (
    copy,
    deepcopy,
)

import pytest

from pandas import MultiIndex
import pandas._testing as tm


def assert_multiindex_copied(copy, original):
    # [FUNCTION DOC]
    # HELPER: asserts that a copy of a MultiIndex is properly independent of the
    # original: levels are copies (at least shallow), code arrays are equal but not the
    # same object, names are equal but a different list object, and the sort order
    # attribute is equal. The first parameter is the copy, the second the original.
    # KNOWN ISSUE: the code equality assertion is performed twice in a row (once under a levels comment and again under the labels comment), so the repeat is redundant.
    # Levels should be (at least, shallow copied)
    tm.assert_copy(copy.levels, original.levels)
    tm.assert_almost_equal(copy.codes, original.codes)

    # Labels doesn't matter which way copied
    tm.assert_almost_equal(copy.codes, original.codes)
    assert copy.codes is not original.codes

    # Names doesn't matter which way copied
    assert copy.names == original.names
    assert copy.names is not original.names

    # Sort order should be copied
    assert copy.sortorder == original.sortorder


def test_copy(idx):
    # [FUNCTION DOC]
    # Verifies a copy of the shared index passes the copy-independence helper.
    i_copy = idx.copy()

    assert_multiindex_copied(i_copy, idx)


def test_shallow_copy(idx):
    # [FUNCTION DOC]
    # Verifies the shallow internal copy of the shared index passes the helper.
    i_copy = idx._view()

    assert_multiindex_copied(i_copy, idx)


def test_view(idx):
    # [FUNCTION DOC]
    # Verifies a view of the shared index passes the helper.
    i_view = idx.view()
    assert_multiindex_copied(i_view, idx)


@pytest.mark.parametrize("func", [copy, deepcopy])
def test_copy_and_deepcopy(func):
    # [FUNCTION DOC]
    # Parametrized over the copy and deepcopy functions. Verifies the result is a
    # distinct object that equals the original multi-index.
    idx = MultiIndex(
        levels=[["foo", "bar"], ["fizz", "buzz"]],
        codes=[[0, 0, 0, 1], [0, 0, 1, 1]],
        names=["first", "second"],
    )
    idx_copy = func(idx)
    assert idx_copy is not idx
    assert idx_copy.equals(idx)


@pytest.mark.parametrize("deep", [True, False])
def test_copy_method(deep):
    # [FUNCTION DOC]
    # Parametrized over deep flag. Verifies the copy method result equals the original.
    idx = MultiIndex(
        levels=[["foo", "bar"], ["fizz", "buzz"]],
        codes=[[0, 0, 0, 1], [0, 0, 1, 1]],
        names=["first", "second"],
    )
    idx_copy = idx.copy(deep=deep)
    assert idx_copy.equals(idx)


@pytest.mark.parametrize("deep", [True, False])
@pytest.mark.parametrize(
    "kwarg, value",
    [
        ("names", ["third", "fourth"]),
    ],
)
def test_copy_method_kwargs(deep, kwarg, value):
    # [FUNCTION DOC]
    # Parametrized over deep flag and a keyword (names with new names). Verifies
    # (gh-12309) the copy method honors the name keyword, so the copy has the new names.
    # gh-12309: Check that the "name" argument as well other kwargs are honored
    idx = MultiIndex(
        levels=[["foo", "bar"], ["fizz", "buzz"]],
        codes=[[0, 0, 0, 1], [0, 0, 1, 1]],
        names=["first", "second"],
    )
    idx_copy = idx.copy(**{kwarg: value, "deep": deep})
    assert getattr(idx_copy, kwarg) == value


def test_copy_deep_false_retains_id():
    # [FUNCTION DOC]
    # Verifies (GH#47878) a non-deep copy keeps the same internal identity token as the original.
    # GH#47878
    idx = MultiIndex(
        levels=[["foo", "bar"], ["fizz", "buzz"]],
        codes=[[0, 0, 0, 1], [0, 0, 1, 1]],
        names=["first", "second"],
    )

    res = idx.copy(deep=False)
    assert res._id is idx._id
