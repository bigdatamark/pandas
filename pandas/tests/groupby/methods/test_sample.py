# [MODULE DOC]
# Tests for the sample method of DataFrameGroupBy and SeriesGroupBy: output shapes for balanced, unbalanced and oversampled groups, argument validation errors, weights, column selections and empty input.
import pytest

from pandas import (
    DataFrame,
    Index,
    Series,
)
import pandas._testing as tm


@pytest.mark.parametrize("n, frac", [(2, None), (None, 0.2)])
def test_groupby_sample_balanced_groups_shape(n, frac):
    # [FUNCTION DOC]
    # Parametrized over (n, frac) as either two rows or a 0.2 fraction. For a frame with two groups of ten rows, sampling per group must yield two rows per group, for both the frame and a single-column selection (named b). Only values and shape are checked; the index is taken from the result.
    values = [1] * 10 + [2] * 10
    df = DataFrame({"a": values, "b": values})

    result = df.groupby("a").sample(n=n, frac=frac)
    values = [1] * 2 + [2] * 2
    expected = DataFrame({"a": values, "b": values}, index=result.index)
    tm.assert_frame_equal(result, expected)

    result = df.groupby("a")["b"].sample(n=n, frac=frac)
    expected = Series(values, name="b", index=result.index)
    tm.assert_series_equal(result, expected)


def test_groupby_sample_unbalanced_groups_shape():
    # [FUNCTION DOC]
    # Groups of 10 and 20 rows sampled with n=5 must each return 5 rows, for the frame and the column selection. Index is taken from the result.
    values = [1] * 10 + [2] * 20
    df = DataFrame({"a": values, "b": values})

    result = df.groupby("a").sample(n=5)
    values = [1] * 5 + [2] * 5
    expected = DataFrame({"a": values, "b": values}, index=result.index)
    tm.assert_frame_equal(result, expected)

    result = df.groupby("a")["b"].sample(n=5)
    expected = Series(values, name="b", index=result.index)
    tm.assert_series_equal(result, expected)


def test_groupby_sample_index_value_spans_groups():
    # [FUNCTION DOC]
    # Frame with a non-unique index whose label 2 appears in both groups; sampling n=2 per group must still return two rows per group for frame and column selection.
    values = [1] * 3 + [2] * 3
    df = DataFrame({"a": values, "b": values}, index=[1, 2, 2, 2, 2, 2])

    result = df.groupby("a").sample(n=2)
    values = [1] * 2 + [2] * 2
    expected = DataFrame({"a": values, "b": values}, index=result.index)
    tm.assert_frame_equal(result, expected)

    result = df.groupby("a")["b"].sample(n=2)
    expected = Series(values, name="b", index=result.index)
    tm.assert_series_equal(result, expected)


def test_groupby_sample_n_and_frac_raises():
    # [FUNCTION DOC]
    # Passing both n and frac must raise a value error saying to enter a value for frac or n, not both, for both the frame and column selection forms.
    df = DataFrame({"a": [1, 2], "b": [1, 2]})
    msg = "Please enter a value for `frac` OR `n`, not both"

    with pytest.raises(ValueError, match=msg):
        df.groupby("a").sample(n=1, frac=1.0)

    with pytest.raises(ValueError, match=msg):
        df.groupby("a")["b"].sample(n=1, frac=1.0)


def test_groupby_sample_frac_gt_one_without_replacement_raises():
    # [FUNCTION DOC]
    # Passing a frac greater than one with replace false must raise a value error requiring replace to be true when upsampling, for frame and column selection forms.
    df = DataFrame({"a": [1, 2], "b": [1, 2]})
    msg = "Replace has to be set to `True` when upsampling the population `frac` > 1."

    with pytest.raises(ValueError, match=msg):
        df.groupby("a").sample(frac=1.5, replace=False)

    with pytest.raises(ValueError, match=msg):
        df.groupby("a")["b"].sample(frac=1.5, replace=False)


@pytest.mark.parametrize("n", [-1, 1.5])
def test_groupby_sample_invalid_n_raises(n):
    # [FUNCTION DOC]
    # Parametrized with n of -1 and 1.5. A negative n must raise a value error about a negative number of rows requested; a non-integer n must raise a value error that only integers are accepted. Checked for frame and column selection.
    df = DataFrame({"a": [1, 2], "b": [1, 2]})

    if n < 0:
        msg = "A negative number of rows requested. Please provide `n` >= 0."
    else:
        msg = "Only integers accepted as `n` values"

    with pytest.raises(ValueError, match=msg):
        df.groupby("a").sample(n=n)

    with pytest.raises(ValueError, match=msg):
        df.groupby("a")["b"].sample(n=n)


def test_groupby_sample_oversample():
    # [FUNCTION DOC]
    # Sampling with frac 2.0 and replace true must double each group (20 rows per group), for frame and column selection forms.
    values = [1] * 10 + [2] * 10
    df = DataFrame({"a": values, "b": values})

    result = df.groupby("a").sample(frac=2.0, replace=True)
    values = [1] * 20 + [2] * 20
    expected = DataFrame({"a": values, "b": values}, index=result.index)
    tm.assert_frame_equal(result, expected)

    result = df.groupby("a")["b"].sample(frac=2.0, replace=True)
    expected = Series(values, name="b", index=result.index)
    tm.assert_series_equal(result, expected)


def test_groupby_sample_without_n_or_frac():
    # [FUNCTION DOC]
    # With both n and frac None, the sample defaults to one row per group, producing the values 1 and 2 for the frame and the named b Series.
    values = [1] * 10 + [2] * 10
    df = DataFrame({"a": values, "b": values})

    result = df.groupby("a").sample(n=None, frac=None)
    expected = DataFrame({"a": [1, 2], "b": [1, 2]}, index=result.index)
    tm.assert_frame_equal(result, expected)

    result = df.groupby("a")["b"].sample(n=None, frac=None)
    expected = Series([1, 2], name="b", index=result.index)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "index, expected_index",
    [(["w", "x", "y", "z"], ["w", "w", "y", "y"]), ([3, 4, 5, 6], [3, 3, 5, 5])],
)
def test_groupby_sample_with_weights(index, expected_index):
    # [FUNCTION DOC]
    # GH 39927: parametrized over string and integer indexes. Sampling n=2 with replacement and weights that give weight only to the first row of each group must return that first row twice per group; the expected index repeats the first label of each group. Checked for frame and column selection.
    # GH 39927 - tests for integer index needed
    values = [1] * 2 + [2] * 2
    df = DataFrame({"a": values, "b": values}, index=Index(index))

    result = df.groupby("a").sample(n=2, replace=True, weights=[1, 0, 1, 0])
    expected = DataFrame({"a": values, "b": values}, index=Index(expected_index))
    tm.assert_frame_equal(result, expected)

    result = df.groupby("a")["b"].sample(n=2, replace=True, weights=[1, 0, 1, 0])
    expected = Series(values, name="b", index=Index(expected_index))
    tm.assert_series_equal(result, expected)


def test_groupby_sample_with_selections():
    # [FUNCTION DOC]
    # GH 39928: sampling from a list-selection of columns b and c with defaults returns one row per group with those two columns only.
    # GH 39928
    values = [1] * 10 + [2] * 10
    df = DataFrame({"a": values, "b": values, "c": values})

    result = df.groupby("a")[["b", "c"]].sample(n=None, frac=None)
    expected = DataFrame({"b": [1, 2], "c": [1, 2]}, index=result.index)
    tm.assert_frame_equal(result, expected)


def test_groupby_sample_with_empty_inputs():
    # [FUNCTION DOC]
    # GH48459: sampling from the groupby of an empty frame with default arguments returns an empty frame equal to the original.
    # GH48459
    df = DataFrame({"a": [], "b": []})
    groupby_df = df.groupby("a")

    result = groupby_df.sample()
    expected = df
    tm.assert_frame_equal(result, expected)
