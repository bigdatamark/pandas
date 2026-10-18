# [MODULE DOC]
# Tests for indexing a Series or DataFrame with boolean masks, including nullable boolean masks that contain missing values.
# Uses the shared fixtures for frame-or-series and the loc/iloc/bracket indexer wrappers.

import pytest

import pandas as pd
import pandas._testing as tm


@pytest.mark.parametrize(
    "values, dtype",
    [
        ([], "object"),
        ([1, 2, 3], "int64"),
        ([1.0, 2.0, 3.0], "float64"),
        (["a", "b", "c"], "object"),
        (["a", "b", "c"], "string"),
        ([1, 2, 3], "datetime64[ns]"),
        ([1, 2, 3], "datetime64[ns, CET]"),
        ([1, 2, 3], "timedelta64[ns]"),
        (["2000", "2001", "2002"], "Period[D]"),
        ([1, 0, 3], "Sparse"),
        ([pd.Interval(0, 1), pd.Interval(1, 2), pd.Interval(3, 4)], "interval"),
    ],
)
@pytest.mark.parametrize(
    "mask", [[True, False, False], [True, True, True], [False, False, False]]
)
@pytest.mark.parametrize("indexer_class", [list, pd.array, pd.Index, pd.Series])
@pytest.mark.parametrize("frame", [True, False])
def test_series_mask_boolean(values, dtype, mask, indexer_class, frame):
    # [FUNCTION DOC]
    # Verifies that a boolean mask works the same through bracket indexing, positional indexing and label indexing, for many value dtypes.
    # Parametrized over: value lists with dtypes (empty object, integer, float, object and string, datetime with and without timezone, timedelta, period, sparse, interval), three mask patterns (mixed, all true, all false), the mask container type (list, extension array, index, series) and whether the target is a series or a one-column frame.
    # Setup: builds a series with string labels (mask and labels truncated to the number of values; an empty frame is built directly with the dtype); the mask is converted to a nullable boolean array or series aligned on the index where relevant.
    # Expected: the result of each indexing route equals the result of bracket indexing with the same mask; positional indexing with a series mask must raise a value error saying positional boolean indexing cannot use an indexable as a mask.
    # KNOWN ISSUE: the expected value is computed with exactly the same bracket-indexing call as the first result, so that first comparison is tautological; only the positional and label indexing routes are really checked against it.
    # In case len(values) < 3
    index = ["a", "b", "c"][: len(values)]
    mask = mask[: len(values)]

    obj = pd.Series(values, dtype=dtype, index=index)
    if frame:
        if len(values) == 0:
            # Otherwise obj is an empty DataFrame with shape (0, 1)
            obj = pd.DataFrame(dtype=dtype, index=index)
        else:
            obj = obj.to_frame()

    if indexer_class is pd.array:
        mask = pd.array(mask, dtype="boolean")
    elif indexer_class is pd.Series:
        mask = pd.Series(mask, index=obj.index, dtype="boolean")
    else:
        mask = indexer_class(mask)

    expected = obj[mask]

    result = obj[mask]
    tm.assert_equal(result, expected)

    if indexer_class is pd.Series:
        msg = "iLocation based boolean indexing cannot use an indexable as a mask"
        with pytest.raises(ValueError, match=msg):
            result = obj.iloc[mask]
    else:
        result = obj.iloc[mask]
        tm.assert_equal(result, expected)

    result = obj.loc[mask]
    tm.assert_equal(result, expected)


def test_na_treated_as_false(frame_or_series, indexer_sli):
    # [FUNCTION DOC]
    # Verifies that missing values inside a nullable boolean mask are treated as false when indexing.
    # Parametrized by the frame-or-series and indexer fixtures (bracket, loc, iloc style wrappers). Setup: a three-element object and a nullable boolean mask with true, false and missing. Expected: indexing with the mask equals indexing with the same mask after filling missing entries with false (regression for issue 31503).
    # https://github.com/pandas-dev/pandas/issues/31503
    obj = frame_or_series([1, 2, 3])

    mask = pd.array([True, False, None], dtype="boolean")

    result = indexer_sli(obj)[mask]
    expected = indexer_sli(obj)[mask.fillna(False)]

    tm.assert_equal(result, expected)
