# [MODULE DOC]
# Tests for the label-decorating methods of the frame class that add a prefix or suffix
# to every column label (default) or row label (axis argument). Checks results against
# indexes built from formatted labels, including labels with percent signs, and the
# invalid axis error. Uses the float frame fixture and the testing helpers module. No
# import-time side effects.

import pytest

from pandas import Index
import pandas._testing as tm


def test_add_prefix_suffix(float_frame):
    # [FUNCTION DOC]
    # Adds a prefix and a suffix (also strings containing a percent sign) to the column
    # labels of the float frame and compares the new column index with an index built by
    # string formatting of each original label.
    with_prefix = float_frame.add_prefix("foo#")
    expected = Index([f"foo#{c}" for c in float_frame.columns])
    tm.assert_index_equal(with_prefix.columns, expected)

    with_suffix = float_frame.add_suffix("#foo")
    expected = Index([f"{c}#foo" for c in float_frame.columns])
    tm.assert_index_equal(with_suffix.columns, expected)

    with_pct_prefix = float_frame.add_prefix("%")
    expected = Index([f"%{c}" for c in float_frame.columns])
    tm.assert_index_equal(with_pct_prefix.columns, expected)

    with_pct_suffix = float_frame.add_suffix("%")
    expected = Index([f"{c}%" for c in float_frame.columns])
    tm.assert_index_equal(with_pct_suffix.columns, expected)


def test_add_prefix_suffix_axis(float_frame):
    # [FUNCTION DOC]
    # Applies prefix and suffix with axis zero (rows) and axis one (columns) and checks
    # the matching index of the result equals an index of formatted labels. Variable
    # names for the suffix results say pct but the suffix is plain text.
    # GH 47819
    with_prefix = float_frame.add_prefix("foo#", axis=0)
    expected = Index([f"foo#{c}" for c in float_frame.index])
    tm.assert_index_equal(with_prefix.index, expected)

    with_prefix = float_frame.add_prefix("foo#", axis=1)
    expected = Index([f"foo#{c}" for c in float_frame.columns])
    tm.assert_index_equal(with_prefix.columns, expected)

    with_pct_suffix = float_frame.add_suffix("#foo", axis=0)
    expected = Index([f"{c}#foo" for c in float_frame.index])
    tm.assert_index_equal(with_pct_suffix.index, expected)

    with_pct_suffix = float_frame.add_suffix("#foo", axis=1)
    expected = Index([f"{c}#foo" for c in float_frame.columns])
    tm.assert_index_equal(with_pct_suffix.columns, expected)


def test_add_prefix_suffix_invalid_axis(float_frame):
    # [FUNCTION DOC]
    # An axis value of two raises a value error saying no such axis exists for a frame,
    # for both prefix and suffix methods.
    with pytest.raises(ValueError, match="No axis named 2 for object type DataFrame"):
        float_frame.add_prefix("foo#", axis=2)

    with pytest.raises(ValueError, match="No axis named 2 for object type DataFrame"):
        float_frame.add_suffix("foo#", axis=2)
