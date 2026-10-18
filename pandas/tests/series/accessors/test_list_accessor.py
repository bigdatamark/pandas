# [MODULE DOC]
# Tests for the list accessor of Series with pyarrow list dtypes (variable, fixed-size
# and large lists): element selection by integer position or slice, length, flattening,
# and error handling. Skipped entirely when pyarrow is not installed (checked at import
# time).
import re

import pytest

from pandas import (
    ArrowDtype,
    Series,
)
import pandas._testing as tm

pa = pytest.importorskip("pyarrow")


@pytest.mark.parametrize(
    "list_dtype",
    (
        pa.list_(pa.int64()),
        pa.list_(pa.int64(), list_size=3),
        pa.large_list(pa.int64()),
    ),
)
def test_list_getitem(list_dtype):
    # [FUNCTION DOC]
    # Parametrized over three arrow list dtypes. Selecting position one from each list
    # gives an int64 arrow series with the second element, missing for the shorter list
    # and for the missing list; the name is kept.
    ser = Series(
        [[1, 2, 3], [4, None, 5], None],
        dtype=ArrowDtype(list_dtype),
        name="a",
    )
    actual = ser.list[1]
    expected = Series([2, None, None], dtype="int64[pyarrow]", name="a")
    tm.assert_series_equal(actual, expected)


def test_list_getitem_index():
    # [FUNCTION DOC]
    # Regression test (GH 58425). Same selection as above with a custom index; the index
    # is kept in the result.
    # GH 58425
    ser = Series(
        [[1, 2, 3], [4, None, 5], None],
        dtype=ArrowDtype(pa.list_(pa.int64())),
        index=[1, 3, 7],
        name="a",
    )
    actual = ser.list[1]
    expected = Series(
        [2, None, None],
        dtype="int64[pyarrow]",
        index=[1, 3, 7],
        name="a",
    )
    tm.assert_series_equal(actual, expected)


def test_list_getitem_slice():
    # [FUNCTION DOC]
    # Verifies slicing from position one with no stop or step returns list-typed results
    # (remaining elements), missing for a missing list; index and name are kept.
    ser = Series(
        [[1, 2, 3], [4, None, 5], None],
        dtype=ArrowDtype(pa.list_(pa.int64())),
        index=[1, 3, 7],
        name="a",
    )
    actual = ser.list[1:None:None]
    expected = Series(
        [[2, 3], [None, 5], None],
        dtype=ArrowDtype(pa.list_(pa.int64())),
        index=[1, 3, 7],
        name="a",
    )
    tm.assert_series_equal(actual, expected)


def test_list_len():
    # [FUNCTION DOC]
    # Verifies the list length method returns an arrow int32 series with counts and
    # missing for the missing list; name kept.
    ser = Series(
        [[1, 2, 3], [4, None], None],
        dtype=ArrowDtype(pa.list_(pa.int64())),
        name="a",
    )
    actual = ser.list.len()
    expected = Series([3, 2, None], dtype=ArrowDtype(pa.int32()), name="a")
    tm.assert_series_equal(actual, expected)


def test_list_flatten():
    # [FUNCTION DOC]
    # Verifies flattening produces one row per inner element (including a missing inner
    # element), skips empty and missing lists, and repeats the original index label for
    # each element, with arrow int64 dtype and the series name.
    ser = Series(
        [[1, 2, 3], None, [4, None], [], [7, 8]],
        dtype=ArrowDtype(pa.list_(pa.int64())),
        name="a",
    )
    actual = ser.list.flatten()
    expected = Series(
        [1, 2, 3, 4, None, 7, 8],
        dtype=ArrowDtype(pa.int64()),
        index=[0, 0, 0, 2, 2, 4, 4],
        name="a",
    )
    tm.assert_series_equal(actual, expected)


def test_list_getitem_slice_invalid():
    # [FUNCTION DOC]
    # Verifies a slice with step zero raises the arrow invalid error, passed through
    # from the external library.
    ser = Series(
        [[1, 2, 3], [4, None, 5], None],
        dtype=ArrowDtype(pa.list_(pa.int64())),
    )
    with tm.external_error_raised(pa.ArrowInvalid):
        ser.list[1:None:0]


def test_list_accessor_non_list_dtype():
    # [FUNCTION DOC]
    # Verifies the list accessor on an arrow integer series raises an attribute error
    # saying the accessor needs a list dtype and naming the actual dtype.
    # KNOWN ISSUE: the expression inside the raises block indexes with a slice but the
    # error comes from accessing the accessor itself, so the slice is irrelevant.
    ser = Series(
        [1, 2, 4],
        dtype=ArrowDtype(pa.int64()),
    )
    with pytest.raises(
        AttributeError,
        match=re.escape(
            "Can only use the '.list' accessor with 'list[pyarrow]' dtype, "
            "not int64[pyarrow]."
        ),
    ):
        ser.list[1:None:0]


@pytest.mark.parametrize(
    "list_dtype",
    (
        pa.list_(pa.int64()),
        pa.list_(pa.int64(), list_size=3),
        pa.large_list(pa.int64()),
    ),
)
def test_list_getitem_invalid_index(list_dtype):
    # [FUNCTION DOC]
    # Parametrized over three arrow list dtypes. Negative and out-of-range positions
    # raise the arrow invalid error; a string key raises a value error saying the key
    # must be an int or slice.
    ser = Series(
        [[1, 2, 3], [4, None, 5], None],
        dtype=ArrowDtype(list_dtype),
    )
    with tm.external_error_raised(pa.ArrowInvalid):
        ser.list[-1]
    with tm.external_error_raised(pa.ArrowInvalid):
        ser.list[5]
    with pytest.raises(ValueError, match="key must be an int or slice, got str"):
        ser.list["abc"]


def test_list_accessor_not_iterable():
    # [FUNCTION DOC]
    # Verifies iterating over the list accessor raises a type error saying the accessor
    # is not iterable.
    ser = Series(
        [[1, 2, 3], [4, None], None],
        dtype=ArrowDtype(pa.list_(pa.int64())),
    )
    with pytest.raises(TypeError, match="'ListAccessor' object is not iterable"):
        iter(ser.list)
