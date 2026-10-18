# [MODULE DOC]
# Regression test module for attribute preservation of FloatAttrArray through merge and
# concat when the column ends up entirely missing after reindexing. Depends on the array
# package in this directory and the public frame/testing APIs.
import numpy as np

import pandas as pd
import pandas._testing as tm
from pandas.tests.extension.array_with_attr import FloatAttrArray


def test_concat_with_all_na():
    # [FUNCTION DOC]
    # Verifies that the attribute stored on an extension array survives merge and concat when
    # reindexing introduces missing values (regression for a linked upstream pull request).
    # Setup: an array of two NaN values with attribute set to a string, placed in a frame with a key
    # column. Three scenarios: inner-style merge with full key overlap, merge with partial overlap
    # (only the first row matches), and column-wise concat of key-indexed frames producing a
    # missing entry. Expected: results equal the expected frames and the resulting column's array
    # still has the original attribute.
    # https://github.com/pandas-dev/pandas/pull/47762
    # ensure that attribute of the column array is preserved (when it gets
    # preserved in reindexing the array) during merge/concat
    arr = FloatAttrArray(np.array([np.nan, np.nan], dtype="float64"), attr="test")

    df1 = pd.DataFrame({"col": arr, "key": [0, 1]})
    df2 = pd.DataFrame({"key": [0, 1], "col2": [1, 2]})
    result = pd.merge(df1, df2, on="key")
    expected = pd.DataFrame({"col": arr, "key": [0, 1], "col2": [1, 2]})
    tm.assert_frame_equal(result, expected)
    assert result["col"].array.attr == "test"

    df1 = pd.DataFrame({"col": arr, "key": [0, 1]})
    df2 = pd.DataFrame({"key": [0, 2], "col2": [1, 2]})
    result = pd.merge(df1, df2, on="key")
    expected = pd.DataFrame({"col": arr.take([0]), "key": [0], "col2": [1]})
    tm.assert_frame_equal(result, expected)
    assert result["col"].array.attr == "test"

    result = pd.concat([df1.set_index("key"), df2.set_index("key")], axis=1)
    expected = pd.DataFrame(
        {"col": arr.take([0, 1, -1]), "col2": [1, np.nan, 2], "key": [0, 1, 2]}
    ).set_index("key")
    tm.assert_frame_equal(result, expected)
    assert result["col"].array.attr == "test"
