# [MODULE DOC]
# Tests for the label-or-level helper methods on DataFrame and Series
# (is-label-reference, is-level-reference,
# ambiguity check, value retrieval and dropping). Contains fixtures, shared assertion
# helpers, and tests
# for each helper in frame and series form. The axis fixture (from shared conftest)
# parametrizes over axis 0, 1 and aliases.
# For axis one the frame is transposed so the same expectations apply.
import pytest

from pandas.core.dtypes.missing import array_equivalent

import pandas as pd


# Fixtures
# ========
@pytest.fixture
def df():
    # [FUNCTION DOC]
    # Fixture: a three-row frame with integer columns L1 and L2 and string column L3.
    """DataFrame with columns 'L1', 'L2', and 'L3'"""
    return pd.DataFrame({"L1": [1, 2, 3], "L2": [11, 12, 13], "L3": ["A", "B", "C"]})


@pytest.fixture(params=[[], ["L1"], ["L1", "L2"], ["L1", "L2", "L3"]])
def df_levels(request, df):
    # [FUNCTION DOC]
    # Fixture parametrized over which columns to move into the index: none, L1, L1 and
    # L2, or all three.
    # Returns the frame from the df fixture with the chosen columns set as index levels.
    """DataFrame with columns or index levels 'L1', 'L2', and 'L3'"""
    levels = request.param

    if levels:
        df = df.set_index(levels)

    return df


@pytest.fixture
def df_ambig(df):
    # [FUNCTION DOC]
    # Fixture: frame with index levels L1 and L2 and a column also named L1 (copy of
    # L3), so L1 is ambiguous between a level and a label.
    """DataFrame with levels 'L1' and 'L2' and labels 'L1' and 'L3'"""
    df = df.set_index(["L1", "L2"])

    df["L1"] = df["L3"]

    return df


# Test is label/level reference
# =============================
def get_labels_levels(df_levels):
    # [FUNCTION DOC]
    # HELPER: given a frame, returns a pair of lists: its column names (labels) and its
    # non-missing index names (levels).
    expected_labels = list(df_levels.columns)
    expected_levels = [name for name in df_levels.index.names if name is not None]
    return expected_labels, expected_levels


def assert_label_reference(frame, labels, axis):
    # [FUNCTION DOC]
    # HELPER: asserts for each name in labels that along the given axis it is a label
    # reference, not a level reference, and a label-or-level reference.
    for label in labels:
        assert frame._is_label_reference(label, axis=axis)
        assert not frame._is_level_reference(label, axis=axis)
        assert frame._is_label_or_level_reference(label, axis=axis)


def assert_level_reference(frame, levels, axis):
    # [FUNCTION DOC]
    # HELPER: asserts for each name in levels that along the given axis it is a level
    # reference, not a label reference, and a label-or-level reference.
    for level in levels:
        assert frame._is_level_reference(level, axis=axis)
        assert not frame._is_label_reference(level, axis=axis)
        assert frame._is_label_or_level_reference(level, axis=axis)


# DataFrame
# ---------
def test_is_level_or_label_reference_df_simple(df_levels, axis):
    # [FUNCTION DOC]
    # Verifies label and level reference detection on frames with varying index levels;
    # for axis one the frame is transposed. Expected labels and levels are computed from
    # the original frame.
    axis = df_levels._get_axis_number(axis)
    # Compute expected labels and levels
    expected_labels, expected_levels = get_labels_levels(df_levels)

    # Transpose frame if axis == 1
    if axis == 1:
        df_levels = df_levels.T

    # Perform checks
    assert_level_reference(df_levels, expected_levels, axis=axis)
    assert_label_reference(df_levels, expected_labels, axis=axis)


def test_is_level_reference_df_ambig(df_ambig, axis):
    # [FUNCTION DOC]
    # Verifies for the ambiguous frame that L1 counts as a label reference (label wins),
    # L2 as a level, and L3 as a label. Parametrized over axis (transposed for axis
    # one).
    axis = df_ambig._get_axis_number(axis)

    # Transpose frame if axis == 1
    if axis == 1:
        df_ambig = df_ambig.T

    # df has both an on-axis level and off-axis label named L1
    # Therefore L1 should reference the label, not the level
    assert_label_reference(df_ambig, ["L1"], axis=axis)

    # df has an on-axis level named L2 and it is not ambiguous
    # Therefore L2 is a level reference
    assert_level_reference(df_ambig, ["L2"], axis=axis)

    # df has a column named L3 and it is not a level reference
    assert_label_reference(df_ambig, ["L3"], axis=axis)


# Series
# ------
def test_is_level_reference_series_simple_axis0(df):
    # [FUNCTION DOC]
    # Verifies level-reference detection on series with one and two index levels: the
    # index names are levels, the series' own column-like name is not.
    # Make series with L1 as index
    s = df.set_index("L1").L2
    assert_level_reference(s, ["L1"], axis=0)
    assert not s._is_level_reference("L2")

    # Make series with L1 and L2 as index
    s = df.set_index(["L1", "L2"]).L3
    assert_level_reference(s, ["L1", "L2"], axis=0)
    assert not s._is_level_reference("L3")


def test_is_level_reference_series_axis1_error(df):
    # [FUNCTION DOC]
    # Verifies asking a series for a level reference along axis one raises a value error
    # ("No axis named 1").
    # Make series with L1 as index
    s = df.set_index("L1").L2

    with pytest.raises(ValueError, match="No axis named 1"):
        s._is_level_reference("L1", axis=1)


# Test _check_label_or_level_ambiguity_df
# =======================================


# DataFrame
# ---------
def test_check_label_or_level_ambiguity_df(df_ambig, axis):
    # [FUNCTION DOC]
    # Verifies the ambiguity check raises a value error for L1 (message depends on axis:
    # index level vs column label for axis zero, column level vs index label after
    # transposing), and does not raise for the unambiguous L2 and returns a falsy result
    # for L3.
    axis = df_ambig._get_axis_number(axis)
    # Transpose frame if axis == 1
    if axis == 1:
        df_ambig = df_ambig.T
        msg = "'L1' is both a column level and an index label"

    else:
        msg = "'L1' is both an index level and a column label"
    # df_ambig has both an on-axis level and off-axis label named L1
    # Therefore, L1 is ambiguous.
    with pytest.raises(ValueError, match=msg):
        df_ambig._check_label_or_level_ambiguity("L1", axis=axis)

    # df_ambig has an on-axis level named L2,, and it is not ambiguous.
    df_ambig._check_label_or_level_ambiguity("L2", axis=axis)

    # df_ambig has an off-axis label named L3, and it is not ambiguous
    assert not df_ambig._check_label_or_level_ambiguity("L3", axis=axis)


# Series
# ------
def test_check_label_or_level_ambiguity_series(df):
    # [FUNCTION DOC]
    # Verifies the ambiguity check never raises for a series (no columns), for
    # index-level and non-level names.
    # A series has no columns and therefore references are never ambiguous

    # Make series with L1 as index
    s = df.set_index("L1").L2
    s._check_label_or_level_ambiguity("L1", axis=0)
    s._check_label_or_level_ambiguity("L2", axis=0)

    # Make series with L1 and L2 as index
    s = df.set_index(["L1", "L2"]).L3
    s._check_label_or_level_ambiguity("L1", axis=0)
    s._check_label_or_level_ambiguity("L2", axis=0)
    s._check_label_or_level_ambiguity("L3", axis=0)


def test_check_label_or_level_ambiguity_series_axis1_error(df):
    # [FUNCTION DOC]
    # Verifies the ambiguity check on a series with axis one raises a value error ("No
    # axis named 1").
    # Make series with L1 as index
    s = df.set_index("L1").L2

    with pytest.raises(ValueError, match="No axis named 1"):
        s._check_label_or_level_ambiguity("L1", axis=1)


# Test _get_label_or_level_values
# ===============================
def assert_label_values(frame, labels, axis):
    # [FUNCTION DOC]
    # HELPER: asserts that for each label the values returned by the label-or-level
    # retrieval equal the frame's column values (axis zero) or row values (axis one),
    # compared with the array-equivalence helper that treats missing values as equal.
    axis = frame._get_axis_number(axis)
    for label in labels:
        if axis == 0:
            expected = frame[label]._values
        else:
            expected = frame.loc[label]._values

        result = frame._get_label_or_level_values(label, axis=axis)
        assert array_equivalent(expected, result)


def assert_level_values(frame, levels, axis):
    # [FUNCTION DOC]
    # HELPER: asserts that for each level the retrieved values equal the values of that
    # index level (axis zero) or column level (axis one).
    axis = frame._get_axis_number(axis)
    for level in levels:
        if axis == 0:
            expected = frame.index.get_level_values(level=level)._values
        else:
            expected = frame.columns.get_level_values(level=level)._values

        result = frame._get_label_or_level_values(level, axis=axis)
        assert array_equivalent(expected, result)


# DataFrame
# ---------
def test_get_label_or_level_values_df_simple(df_levels, axis):
    # [FUNCTION DOC]
    # Verifies value retrieval for labels and levels on frames with varying index
    # levels, transposing for axis one.
    # Compute expected labels and levels
    expected_labels, expected_levels = get_labels_levels(df_levels)

    axis = df_levels._get_axis_number(axis)
    # Transpose frame if axis == 1
    if axis == 1:
        df_levels = df_levels.T

    # Perform checks
    assert_label_values(df_levels, expected_labels, axis=axis)
    assert_level_values(df_levels, expected_levels, axis=axis)


def test_get_label_or_level_values_df_ambig(df_ambig, axis):
    # [FUNCTION DOC]
    # Verifies value retrieval on the ambiguous frame for the unambiguous level L2 and
    # the label L3.
    axis = df_ambig._get_axis_number(axis)
    # Transpose frame if axis == 1
    if axis == 1:
        df_ambig = df_ambig.T

    # df has an on-axis level named L2, and it is not ambiguous.
    assert_level_values(df_ambig, ["L2"], axis=axis)

    # df has an off-axis label named L3, and it is not ambiguous.
    assert_label_values(df_ambig, ["L3"], axis=axis)


def test_get_label_or_level_values_df_duplabels(df, axis):
    # [FUNCTION DOC]
    # Verifies value retrieval when a label is duplicated: the level L1 and unique label
    # L3 work, while the duplicated label L2 raises a value error saying the column
    # label (axis zero) or index label (axis one) is not unique.
    df = df.set_index(["L1"])
    df_duplabels = pd.concat([df, df["L2"]], axis=1)
    axis = df_duplabels._get_axis_number(axis)
    # Transpose frame if axis == 1
    if axis == 1:
        df_duplabels = df_duplabels.T

    # df has unambiguous level 'L1'
    assert_level_values(df_duplabels, ["L1"], axis=axis)

    # df has unique label 'L3'
    assert_label_values(df_duplabels, ["L3"], axis=axis)

    # df has duplicate labels 'L2'
    if axis == 0:
        expected_msg = "The column label 'L2' is not unique"
    else:
        expected_msg = "The index label 'L2' is not unique"

    with pytest.raises(ValueError, match=expected_msg):
        assert_label_values(df_duplabels, ["L2"], axis=axis)


# Series
# ------
def test_get_label_or_level_values_series_axis0(df):
    # [FUNCTION DOC]
    # Verifies value retrieval of index levels from series with one and two index levels
    # along axis zero.
    # Make series with L1 as index
    s = df.set_index("L1").L2
    assert_level_values(s, ["L1"], axis=0)

    # Make series with L1 and L2 as index
    s = df.set_index(["L1", "L2"]).L3
    assert_level_values(s, ["L1", "L2"], axis=0)


def test_get_label_or_level_values_series_axis1_error(df):
    # [FUNCTION DOC]
    # Verifies value retrieval on a series with axis one raises a value error ("No axis
    # named 1").
    # Make series with L1 as index
    s = df.set_index("L1").L2

    with pytest.raises(ValueError, match="No axis named 1"):
        s._get_label_or_level_values("L1", axis=1)


# Test _drop_labels_or_levels
# ===========================
def assert_labels_dropped(frame, labels, axis):
    # [FUNCTION DOC]
    # HELPER: asserts that dropping each label along the axis removes it from the
    # frame's columns (axis zero) or index (axis one) in the returned frame while the
    # original still has it.
    axis = frame._get_axis_number(axis)
    for label in labels:
        df_dropped = frame._drop_labels_or_levels(label, axis=axis)

        if axis == 0:
            assert label in frame.columns
            assert label not in df_dropped.columns
        else:
            assert label in frame.index
            assert label not in df_dropped.index


def assert_levels_dropped(frame, levels, axis):
    # [FUNCTION DOC]
    # HELPER: asserts that dropping each level along the axis removes it from the
    # returned object's index names (axis zero) or column names (axis one) while the
    # original still has it.
    axis = frame._get_axis_number(axis)
    for level in levels:
        df_dropped = frame._drop_labels_or_levels(level, axis=axis)

        if axis == 0:
            assert level in frame.index.names
            assert level not in df_dropped.index.names
        else:
            assert level in frame.columns.names
            assert level not in df_dropped.columns.names


# DataFrame
# ---------
def test_drop_labels_or_levels_df(df_levels, axis):
    # [FUNCTION DOC]
    # Verifies dropping labels and levels from frames with varying index levels, and
    # that dropping an unknown name raises a value error ("not valid labels or levels").
    # Compute expected labels and levels
    expected_labels, expected_levels = get_labels_levels(df_levels)

    axis = df_levels._get_axis_number(axis)
    # Transpose frame if axis == 1
    if axis == 1:
        df_levels = df_levels.T

    # Perform checks
    assert_labels_dropped(df_levels, expected_labels, axis=axis)
    assert_levels_dropped(df_levels, expected_levels, axis=axis)

    with pytest.raises(ValueError, match="not valid labels or levels"):
        df_levels._drop_labels_or_levels("L4", axis=axis)


# Series
# ------
def test_drop_labels_or_levels_series(df):
    # [FUNCTION DOC]
    # Verifies dropping levels from series with one and two index levels, and that an
    # unknown name (L4) raises a value error ("not valid labels or levels").
    # Make series with L1 as index
    s = df.set_index("L1").L2
    assert_levels_dropped(s, ["L1"], axis=0)

    with pytest.raises(ValueError, match="not valid labels or levels"):
        s._drop_labels_or_levels("L4", axis=0)

    # Make series with L1 and L2 as index
    s = df.set_index(["L1", "L2"]).L3
    assert_levels_dropped(s, ["L1", "L2"], axis=0)

    with pytest.raises(ValueError, match="not valid labels or levels"):
        s._drop_labels_or_levels("L4", axis=0)
