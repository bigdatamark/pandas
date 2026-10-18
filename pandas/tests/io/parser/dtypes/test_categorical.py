"""
Tests dtype specification during parsing
for all of the parsers defined in parsers.py
"""

# [MODULE DOC]
# Parser tests of categorical dtype handling during CSV reading: requesting categorical
# dtype for all or some columns, unsorted and missing values, high cardinality with
# small internal buffers, encodings, chunked reading with inferred and explicit
# categories, explicit category lists with order, and coercion of parsed values to
# numeric, datetime, timestamp, timedelta and boolean categories.
# KNOWN ISSUE: the module docstring says the parsers are defined in a file named
# parsers.py, but no such module exists here; the parser fixtures come from the
# package's shared fixtures module.
# Uses the pyarrow expected-failure marker alias.

from io import StringIO
import os

import numpy as np
import pytest

from pandas._libs import parsers as libparsers
from pandas.errors import Pandas4Warning

from pandas.core.dtypes.dtypes import CategoricalDtype

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Timestamp,
)
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")


@xfail_pyarrow  # AssertionError: Attributes of DataFrame.iloc[:, 0] are different
@pytest.mark.parametrize(
    "dtype",
    [
        "category",
        CategoricalDtype(),
        {"a": "category", "b": "category", "c": CategoricalDtype()},
    ],
)
def test_categorical_dtype(all_parsers, dtype):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies that requesting category dtype as a
    # string, as a dtype object, or as a per-column mapping converts every column to a
    # categorical whose categories are the string values (so even numbers are categories
    # of strings).
    # see gh-10153
    parser = all_parsers
    data = """a,b,c
1,a,3.4
1,a,3.4
2,b,4.5"""
    expected = DataFrame(
        {
            "a": Categorical(["1", "1", "2"]),
            "b": Categorical(["a", "a", "b"]),
            "c": Categorical(["3.4", "3.4", "4.5"]),
        }
    )
    actual = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("dtype", [{"b": "category"}, {1: "category"}])
def test_categorical_dtype_single(all_parsers, dtype, request):
    # [FUNCTION DOC]
    # Verifies that requesting category for only one column (by name or position)
    # converts just that column to a categorical and leaves the others numeric. For
    # pyarrow the test is marked as non-strictly expected to fail because it sometimes
    # yields object dtype.
    # see gh-10153
    parser = all_parsers
    data = """a,b,c
1,a,3.4
1,a,3.4
2,b,4.5"""
    expected = DataFrame(
        {"a": [1, 1, 2], "b": Categorical(["a", "a", "b"]), "c": [3.4, 3.4, 4.5]}
    )
    if parser.engine == "pyarrow":
        mark = pytest.mark.xfail(
            strict=False,
            reason="Flaky test sometimes gives object dtype instead of Categorical",
        )
        request.applymarker(mark)

    actual = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(actual, expected)


@xfail_pyarrow  # AssertionError: Attributes of DataFrame.iloc[:, 0] are different
def test_categorical_dtype_unsorted(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies categories are taken from the data
    # and the unsorted string column keeps its values.
    # see gh-10153
    parser = all_parsers
    data = """a,b,c
1,b,3.4
1,b,3.4
2,a,4.5"""
    expected = DataFrame(
        {
            "a": Categorical(["1", "1", "2"]),
            "b": Categorical(["b", "b", "a"]),
            "c": Categorical(["3.4", "3.4", "4.5"]),
        }
    )
    actual = parser.read_csv(StringIO(data), dtype="category")
    tm.assert_frame_equal(actual, expected)


@xfail_pyarrow  # AssertionError: Attributes of DataFrame.iloc[:, 0] are different
def test_categorical_dtype_missing(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies that text nan in a categorical
    # column becomes a missing value rather than a category.
    # see gh-10153
    parser = all_parsers
    data = """a,b,c
1,b,3.4
1,nan,3.4
2,a,4.5"""
    expected = DataFrame(
        {
            "a": Categorical(["1", "1", "2"]),
            "b": Categorical(["b", np.nan, "a"]),
            "c": Categorical(["3.4", "3.4", "4.5"]),
        }
    )
    actual = parser.read_csv(StringIO(data), dtype="category")
    tm.assert_frame_equal(actual, expected)


@xfail_pyarrow  # AssertionError: Attributes of DataFrame.iloc[:, 0] are different
@pytest.mark.slow
def test_categorical_dtype_high_cardinality_numeric(all_parsers, monkeypatch):
    # [FUNCTION DOC]
    # Marked slow and expected to fail for pyarrow. Regression for the C parser with
    # many distinct numeric strings across internal buffer boundaries: patches the
    # buffer heuristic to thirty-two, reads thirty-three sorted numeric strings as
    # category, reorders the result's categories to sorted order, and verifies equality
    # with an ordered categorical.
    # see gh-18186
    # was an issue with C parser, due to DEFAULT_BUFFER_HEURISTIC
    parser = all_parsers
    heuristic = 2**5
    data = np.sort([str(i) for i in range(heuristic + 1)])
    expected = DataFrame({"a": Categorical(data, ordered=True)})
    with monkeypatch.context() as m:
        m.setattr(libparsers, "DEFAULT_BUFFER_HEURISTIC", heuristic)
        actual = parser.read_csv(StringIO("a\n" + "\n".join(data)), dtype="category")
    actual["a"] = actual["a"].cat.reorder_categories(
        np.sort(actual.a.cat.categories), ordered=True
    )
    tm.assert_frame_equal(actual, expected)


def test_categorical_dtype_utf16(all_parsers, csv_dir_path):
    # [FUNCTION DOC]
    # Verifies that reading a UTF-16 tab-separated file with category dtype equals
    # converting each column of the normal read to categorical.
    # see gh-10153
    pth = os.path.join(csv_dir_path, "utf16_ex.txt")
    parser = all_parsers
    encoding = "utf-16"
    sep = "\t"

    expected = parser.read_csv(pth, sep=sep, encoding=encoding)
    expected = expected.apply(Categorical)

    actual = parser.read_csv(pth, sep=sep, encoding=encoding, dtype="category")
    tm.assert_frame_equal(actual, expected)


def test_categorical_dtype_chunksize_infer_categories(all_parsers):
    # [FUNCTION DOC]
    # Verifies chunked reading (size two) with category dtype for one column infers
    # categories per chunk (first chunk has a and b; second has b and c). For pyarrow,
    # verifies the unsupported chunk size error.
    # see gh-10153
    parser = all_parsers
    data = """a,b
1,a
1,b
1,b
2,c"""
    expecteds = [
        DataFrame({"a": [1, 1], "b": Categorical(["a", "b"])}),
        DataFrame({"a": [1, 2], "b": Categorical(["b", "c"])}, index=[2, 3]),
    ]

    if parser.engine == "pyarrow":
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), dtype={"b": "category"}, chunksize=2)
        return

    with parser.read_csv(
        StringIO(data), dtype={"b": "category"}, chunksize=2
    ) as actuals:
        for actual, expected in zip(actuals, expecteds):
            tm.assert_frame_equal(actual, expected)


def test_categorical_dtype_chunksize_explicit_categories(all_parsers):
    # [FUNCTION DOC]
    # Verifies chunked reading with an explicit categorical dtype (categories a, b, c)
    # gives each chunk the full category list. For pyarrow, verifies the unsupported
    # chunk size error.
    # see gh-10153
    parser = all_parsers
    data = """a,b
1,a
1,b
1,b
2,c"""
    cats = ["a", "b", "c"]
    expecteds = [
        DataFrame({"a": [1, 1], "b": Categorical(["a", "b"], categories=cats)}),
        DataFrame(
            {"a": [1, 2], "b": Categorical(["b", "c"], categories=cats)},
            index=[2, 3],
        ),
    ]
    dtype = CategoricalDtype(cats)

    if parser.engine == "pyarrow":
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), dtype={"b": dtype}, chunksize=2)
        return

    with parser.read_csv(StringIO(data), dtype={"b": dtype}, chunksize=2) as actuals:
        for actual, expected in zip(actuals, expecteds):
            tm.assert_frame_equal(actual, expected)


def test_categorical_dtype_latin1(all_parsers, csv_dir_path):
    # [FUNCTION DOC]
    # Verifies reading a latin-1 encoded file with category dtype for the second column
    # equals converting that column to categorical after a normal read.
    # see gh-10153
    pth = os.path.join(csv_dir_path, "unicode_series.csv")
    parser = all_parsers
    encoding = "latin-1"

    expected = parser.read_csv(pth, header=None, encoding=encoding)
    expected[1] = Categorical(expected[1])

    actual = parser.read_csv(pth, header=None, encoding=encoding, dtype={1: "category"})
    tm.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize(
    "categories",
    [["a", "b", "c"], ["a", "c", "b"], ["a", "b", "c", "d"], ["c", "b", "a"]],
)
def test_categorical_category_dtype(all_parsers, categories, ordered):
    # [FUNCTION DOC]
    # Verifies that explicit categorical dtypes with different category orders,
    # supersets of the data and the ordered flag on or off are applied exactly to the
    # column.
    parser = all_parsers
    data = """a,b
1,a
1,b
1,b
2,c"""
    expected = DataFrame(
        {
            "a": [1, 1, 1, 2],
            "b": Categorical(
                ["a", "b", "b", "c"], categories=categories, ordered=ordered
            ),
        }
    )

    dtype = {"b": CategoricalDtype(categories=categories, ordered=ordered)}
    result = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(result, expected)


def test_categorical_category_dtype_unsorted(all_parsers):
    # [FUNCTION DOC]
    # Verifies an explicit unsorted list of categories is preserved in that order.
    parser = all_parsers
    data = """a,b
1,a
1,b
1,b
2,c"""
    dtype = CategoricalDtype(["c", "b", "a"])
    expected = DataFrame(
        {
            "a": [1, 1, 1, 2],
            "b": Categorical(["a", "b", "b", "c"], categories=["c", "b", "a"]),
        }
    )

    result = parser.read_csv(StringIO(data), dtype={"b": dtype})
    tm.assert_frame_equal(result, expected)


def test_categorical_coerces_numeric(all_parsers):
    # [FUNCTION DOC]
    # Verifies string values in the data are coerced to match integer categories given
    # in the dtype.
    parser = all_parsers
    dtype = {"b": CategoricalDtype([1, 2, 3])}

    data = "b\n1\n1\n2\n3"
    expected = DataFrame({"b": Categorical([1, 1, 2, 3])})

    result = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(result, expected)


def test_categorical_coerces_datetime(all_parsers):
    # [FUNCTION DOC]
    # Verifies date strings are coerced to match datetime categories given in the dtype.
    parser = all_parsers
    dti = pd.DatetimeIndex(["2017-01-01", "2018-01-01", "2019-01-01"], freq=None)
    dtype = {"b": CategoricalDtype(dti)}

    data = "b\n2017-01-01\n2018-01-01\n2019-01-01"
    expected = DataFrame({"b": Categorical(dtype["b"].categories)})

    result = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(result, expected)


def test_categorical_coerces_timestamp(all_parsers):
    # [FUNCTION DOC]
    # Verifies date strings are coerced to match a single timestamp category.
    parser = all_parsers
    dtype = {"b": CategoricalDtype([Timestamp("2014")])}

    data = "b\n2014-01-01\n2014-01-01"
    expected = DataFrame({"b": Categorical([Timestamp("2014")] * 2)})

    result = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(result, expected)


def test_categorical_coerces_timedelta(all_parsers):
    # [FUNCTION DOC]
    # Verifies duration strings are coerced to match timedelta categories.
    parser = all_parsers
    dtype = {"b": CategoricalDtype(pd.to_timedelta(["1h", "2h", "3h"]))}

    data = "b\n1h\n2h\n3h"
    expected = DataFrame({"b": Categorical(dtype["b"].categories)})

    result = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data",
    [
        "b\nTrue\nFalse\nNA\nFalse",
        "b\ntrue\nfalse\nNA\nfalse",
        "b\nTRUE\nFALSE\nNA\nFALSE",
        "b\nTrue\nFalse\nNA\nFALSE",
    ],
)
def test_categorical_dtype_coerces_boolean(all_parsers, data):
    # [FUNCTION DOC]
    # Verifies boolean-like text in various case combinations, including NA, is coerced
    # to booleans with a missing value when the categories are false and true.
    # see gh-20498
    parser = all_parsers
    dtype = {"b": CategoricalDtype([False, True])}
    expected = DataFrame({"b": Categorical([True, False, None, False])})

    result = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(result, expected)


def test_categorical_unexpected_categories(all_parsers):
    # [FUNCTION DOC]
    # Verifies that a value not among the dtype's categories becomes a missing value,
    # and that a future-change warning (Pandas4Warning) about constructing a categorical
    # with values not in the dtype is emitted (stack level not checked).
    parser = all_parsers
    dtype = {"b": CategoricalDtype(["a", "b", "d", "e"])}

    data = "b\nd\na\nc\nd"  # Unexpected c
    expected = DataFrame({"b": Categorical(["d", "a", None, "d"], dtype=dtype["b"])})

    msg = "Constructing a Categorical with a dtype and values containing"
    with tm.assert_produces_warning(Pandas4Warning, match=msg, check_stacklevel=False):
        result = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(result, expected)
