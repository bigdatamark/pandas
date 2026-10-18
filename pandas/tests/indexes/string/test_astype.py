# [MODULE DOC]
# Tests for converting an object-dtype Index (and Series) holding mixed text and
# bytes values to the string dtype. Single test function; uses the shared
# testing helpers for index and series equality.

from pandas import (
    Index,
    Series,
)
import pandas._testing as tm


def test_astype_str_from_bytes():
    # [FUNCTION DOC]
    # Verifies that converting an object index containing a non-ASCII text value and a
    # bytes value to str produces a string-dtype index whose bytes element is
    # rendered through ordinary string formatting (not decoded as bytes), and that
    # a Series built from the same index converts to the same values with string
    # dtype. Compares with the index and series equality assertions.
    # https://github.com/pandas-dev/pandas/issues/38607
    # GH#49658 pre-2.0 Index called .values.astype(str) here, which effectively
    #  did a .decode() on the bytes object.  In 2.0 we go through
    #  ensure_string_array which does f"{val}"
    idx = Index(["あ", b"a"], dtype="object")
    result = idx.astype(str)
    expected = Index(["あ", "a"], dtype="str")
    tm.assert_index_equal(result, expected)

    # while we're here, check that Series.astype behaves the same
    result = Series(idx).astype(str)
    expected = Series(expected, dtype="str")
    tm.assert_series_equal(result, expected)
