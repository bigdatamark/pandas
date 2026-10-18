# [MODULE DOC]
# Tests for named aggregation (keyword relabeling) on Series.agg.

import pandas as pd
import pandas._testing as tm


def test_relabel_no_duplicated_method():
    # [FUNCTION DOC]
    # Checks that named aggregation with distinct methods gives the same result as the equivalent
    # dict-based aggregation: single string function, two string functions, and a mix of Python builtin
    # and string functions. Compares via series equality.
    # this is to test there is no duplicated method used in agg
    df = pd.DataFrame({"A": [1, 2, 1, 2], "B": [1, 2, 3, 4]})

    result = df["A"].agg(foo="sum")
    expected = df["A"].agg({"foo": "sum"})
    tm.assert_series_equal(result, expected)

    result = df["B"].agg(foo="min", bar="max")
    expected = df["B"].agg({"foo": "min", "bar": "max"})
    tm.assert_series_equal(result, expected)

    result = df["B"].agg(foo=sum, bar=min, cat="max")
    expected = df["B"].agg({"foo": sum, "bar": min, "cat": "max"})
    tm.assert_series_equal(result, expected)


def test_relabel_duplicated_method():
    # [FUNCTION DOC]
    # Checks that the same method may be used under different new names in named aggregation: two
    # names for sum on column A give equal values, and a builtin min plus the string min give equal values.
    # Expected results are series indexed by the new names and carrying the original column name.
    # this is to test with nested renaming, duplicated method can be used
    # if they are assigned with different new names
    df = pd.DataFrame({"A": [1, 2, 1, 2], "B": [1, 2, 3, 4]})

    result = df["A"].agg(foo="sum", bar="sum")
    expected = pd.Series([6, 6], index=["foo", "bar"], name="A")
    tm.assert_series_equal(result, expected)

    result = df["B"].agg(foo=min, bar="min")
    expected = pd.Series([1, 1], index=["foo", "bar"], name="B")
    tm.assert_series_equal(result, expected)
