# [MODULE DOC]
# Tests for Series.add_prefix and Series.add_suffix.
# Covers label decoration of the index, the explicit axis argument, and the
# error for an axis that does not exist on a Series. Uses the string_series
# fixture (a float Series with string labels). No import-time side effects.

import pytest

from pandas import Index
import pandas._testing as tm


def test_add_prefix_suffix(string_series):
    # [FUNCTION DOC]
    # Checks that add_prefix and add_suffix prepend/append the given text to every
    # index label, including a percent sign (not treated as a format token).
    # Each result's index must equal an Index built by formatting every original label.
    with_prefix = string_series.add_prefix("foo#")
    expected = Index([f"foo#{c}" for c in string_series.index])
    tm.assert_index_equal(with_prefix.index, expected)

    with_suffix = string_series.add_suffix("#foo")
    expected = Index([f"{c}#foo" for c in string_series.index])
    tm.assert_index_equal(with_suffix.index, expected)

    with_pct_prefix = string_series.add_prefix("%")
    expected = Index([f"%{c}" for c in string_series.index])
    tm.assert_index_equal(with_pct_prefix.index, expected)

    with_pct_suffix = string_series.add_suffix("%")
    expected = Index([f"{c}%" for c in string_series.index])
    tm.assert_index_equal(with_pct_suffix.index, expected)


def test_add_prefix_suffix_axis(string_series):
    # [FUNCTION DOC]
    # Regression test (GH 47819): add_prefix and add_suffix accept an axis of 0 for a
    # Series and give the same label-decorated index as with the default axis.
    # GH 47819
    with_prefix = string_series.add_prefix("foo#", axis=0)
    expected = Index([f"foo#{c}" for c in string_series.index])
    tm.assert_index_equal(with_prefix.index, expected)

    with_pct_suffix = string_series.add_suffix("#foo", axis=0)
    expected = Index([f"{c}#foo" for c in string_series.index])
    tm.assert_index_equal(with_pct_suffix.index, expected)


def test_add_prefix_suffix_invalid_axis(string_series):
    # [FUNCTION DOC]
    # Verifies that passing an axis of 1 to add_prefix or add_suffix on a Series raises a
    # value error stating that no axis of that number exists for the Series type.
    with pytest.raises(ValueError, match="No axis named 1 for object type Series"):
        string_series.add_prefix("foo#", axis=1)

    with pytest.raises(ValueError, match="No axis named 1 for object type Series"):
        string_series.add_suffix("foo#", axis=1)
