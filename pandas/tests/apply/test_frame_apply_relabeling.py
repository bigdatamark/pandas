# [MODULE DOC]
# Tests for named aggregation (keyword relabeling) on DataFrame.agg, plus a check on the helper
# that reconstructs aggregation arguments.

import numpy as np

import pandas as pd
import pandas._testing as tm


def test_agg_relabel():
    # [FUNCTION DOC]
    # GH 26513: named aggregation with tuples of column and function. A single column and function
    # yields a one-row frame; two functions on the same column yield two rows indexed by the new names.
    # GH 26513
    df = pd.DataFrame({"A": [1, 2, 1, 2], "B": [1, 2, 3, 4], "C": [3, 4, 5, 6]})

    # simplest case with one column, one func
    result = df.agg(foo=("B", "sum"))
    expected = pd.DataFrame({"B": [10]}, index=pd.Index(["foo"]))
    tm.assert_frame_equal(result, expected)

    # test on same column with different methods
    result = df.agg(foo=("B", "sum"), bar=("B", "min"))
    expected = pd.DataFrame({"B": [10, 1]}, index=pd.Index(["foo", "bar"]))

    tm.assert_frame_equal(result, expected)


def test_agg_relabel_multi_columns_multi_methods():
    # [FUNCTION DOC]
    # GH 26513: many names across several columns and methods. Expected is a frame with an
    # index of the new names, with missing values wherever a name does not apply to a column.
    # GH 26513, test on multiple columns with multiple methods
    df = pd.DataFrame({"A": [1, 2, 1, 2], "B": [1, 2, 3, 4], "C": [3, 4, 5, 6]})
    result = df.agg(
        foo=("A", "sum"),
        bar=("B", "mean"),
        cat=("A", "min"),
        dat=("B", "max"),
        f=("A", "max"),
        g=("C", "min"),
    )
    expected = pd.DataFrame(
        {
            "A": [6.0, np.nan, 1.0, np.nan, 2.0, np.nan],
            "B": [np.nan, 2.5, np.nan, 4.0, np.nan, np.nan],
            "C": [np.nan, np.nan, np.nan, np.nan, np.nan, 3.0],
        },
        index=pd.Index(["foo", "bar", "cat", "dat", "f", "g"]),
    )
    tm.assert_frame_equal(result, expected)


def test_agg_relabel_partial_functions():
    # [FUNCTION DOC]
    # GH 26513: named aggregation using numpy functions, builtins, strings and a lambda for the
    # function element; results for each column are checked, with missing values where not applicable.
    # GH 26513, test on partial, functools or more complex cases
    df = pd.DataFrame({"A": [1, 2, 1, 2], "B": [1, 2, 3, 4], "C": [3, 4, 5, 6]})
    result = df.agg(foo=("A", np.mean), bar=("A", "mean"), cat=("A", min))
    expected = pd.DataFrame(
        {"A": [1.5, 1.5, 1.0]}, index=pd.Index(["foo", "bar", "cat"])
    )
    tm.assert_frame_equal(result, expected)

    result = df.agg(
        foo=("A", min),
        bar=("B", np.min),
        cat=("B", max),
        dat=("C", "min"),
        f=("B", np.sum),
        kk=("B", lambda x: min(x)),
    )
    expected = pd.DataFrame(
        {
            "A": [1.0, np.nan, np.nan, np.nan, np.nan, np.nan],
            "B": [np.nan, 1.0, 4.0, np.nan, 10.0, 1.0],
            "C": [np.nan, np.nan, np.nan, 3.0, np.nan, np.nan],
        },
        index=pd.Index(["foo", "bar", "cat", "dat", "f", "kk"]),
    )
    tm.assert_frame_equal(result, expected)


def test_agg_namedtuple():
    # [FUNCTION DOC]
    # GH 26513: named aggregation using the NamedAgg helper, with positional and keyword forms, over
    # one column and then over two columns. Expected frames have missing values where not applicable.
    # GH 26513
    df = pd.DataFrame({"A": [0, 1], "B": [1, 2]})
    result = df.agg(
        foo=pd.NamedAgg("B", "sum"),
        bar=pd.NamedAgg("B", "min"),
        cat=pd.NamedAgg(column="B", aggfunc="count"),
        fft=pd.NamedAgg("B", aggfunc="max"),
    )

    expected = pd.DataFrame(
        {"B": [3, 1, 2, 2]}, index=pd.Index(["foo", "bar", "cat", "fft"])
    )
    tm.assert_frame_equal(result, expected)

    result = df.agg(
        foo=pd.NamedAgg("A", "min"),
        bar=pd.NamedAgg(column="B", aggfunc="max"),
        cat=pd.NamedAgg(column="A", aggfunc="max"),
    )
    expected = pd.DataFrame(
        {"A": [0.0, np.nan, 1.0], "B": [np.nan, 2.0, np.nan]},
        index=pd.Index(["foo", "bar", "cat"]),
    )
    tm.assert_frame_equal(result, expected)


def test_reconstruct_func():
    # [FUNCTION DOC]
    # GH 28472: ensures the reconstruct_func helper in the core apply module stays importable at that
    # location since other libraries use it. For the string min it must return a tuple of: relabeling flag
    # false, the function unchanged, no columns and no order.
    # GH 28472, test to ensure reconstruct_func isn't moved;
    # This method is used by other libraries (e.g. dask)
    result = pd.core.apply.reconstruct_func("min")
    expected = (False, "min", None, None)
    tm.assert_equal(result, expected)
