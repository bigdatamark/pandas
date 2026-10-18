"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""

# [MODULE DOC]
# Parser tests (all engines via the parser fixture) of index handling in the CSV reader:
# names combined with an index column, multi-level indexes, empty frames with indexes,
# implicit indexes from short headers, and duplicate index labels.
# Defines two marker aliases: one marks a test as expected to fail for the pyarrow
# engine and one skips it for that engine.

from datetime import datetime
from io import StringIO
import os

import pytest

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
)
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")


@pytest.mark.parametrize(
    "data,kwargs,expected",
    [
        (
            """foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
""",
            {"index_col": 0, "names": ["index", "A", "B", "C", "D"]},
            DataFrame(
                [
                    [2, 3, 4, 5],
                    [7, 8, 9, 10],
                    [12, 13, 14, 15],
                    [12, 13, 14, 15],
                    [12, 13, 14, 15],
                    [12, 13, 14, 15],
                ],
                index=Index(["foo", "bar", "baz", "qux", "foo2", "bar2"], name="index"),
                columns=["A", "B", "C", "D"],
            ),
        ),
        (
            """foo,one,2,3,4,5
foo,two,7,8,9,10
foo,three,12,13,14,15
bar,one,12,13,14,15
bar,two,12,13,14,15
""",
            {"index_col": [0, 1], "names": ["index1", "index2", "A", "B", "C", "D"]},
            DataFrame(
                [
                    [2, 3, 4, 5],
                    [7, 8, 9, 10],
                    [12, 13, 14, 15],
                    [12, 13, 14, 15],
                    [12, 13, 14, 15],
                ],
                index=MultiIndex.from_tuples(
                    [
                        ("foo", "one"),
                        ("foo", "two"),
                        ("foo", "three"),
                        ("bar", "one"),
                        ("bar", "two"),
                    ],
                    names=["index1", "index2"],
                ),
                columns=["A", "B", "C", "D"],
            ),
        ),
    ],
)
def test_pass_names_with_index(all_parsers, data, kwargs, expected):
    # [FUNCTION DOC]
    # Verifies that explicit names combined with an index column given as a single
    # position or as two positions produce a frame whose index is a plain index or a
    # multi-level index with the named levels, and the remaining columns keep the given
    # names.
    parser = all_parsers
    result = parser.read_csv(StringIO(data), **kwargs)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("index_col", [[0, 1], [1, 0]])
def test_multi_index_no_level_names(
    request, all_parsers, index_col, using_infer_string
):
    # [FUNCTION DOC]
    # Verifies that reading headerless data with two index columns (in either order) and
    # explicit column names gives the same frame as reading the same data with its
    # header row, apart from the index level names being unset.
    data = """index1,index2,A,B,C,D
foo,one,2,3,4,5
foo,two,7,8,9,10
foo,three,12,13,14,15
bar,one,12,13,14,15
bar,two,12,13,14,15
"""
    headless_data = "\n".join(data.split("\n")[1:])

    names = ["A", "B", "C", "D"]
    parser = all_parsers

    result = parser.read_csv(
        StringIO(headless_data), index_col=index_col, header=None, names=names
    )
    expected = parser.read_csv(StringIO(data), index_col=index_col)

    # No index names in headless data.
    expected.index.names = [None] * 2
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
def test_multi_index_no_level_names_implicit(all_parsers):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Verifies that when data rows have two more fields than the
    # header names, the leading two fields form an unnamed multi-level index
    # automatically.
    parser = all_parsers
    data = """A,B,C,D
foo,one,2,3,4,5
foo,two,7,8,9,10
foo,three,12,13,14,15
bar,one,12,13,14,15
bar,two,12,13,14,15
"""

    result = parser.read_csv(StringIO(data))
    expected = DataFrame(
        [
            [2, 3, 4, 5],
            [7, 8, 9, 10],
            [12, 13, 14, 15],
            [12, 13, 14, 15],
            [12, 13, 14, 15],
        ],
        columns=["A", "B", "C", "D"],
        index=MultiIndex.from_tuples(
            [
                ("foo", "one"),
                ("foo", "two"),
                ("foo", "three"),
                ("bar", "one"),
                ("bar", "two"),
            ]
        ),
    )
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
@pytest.mark.parametrize(
    "data,columns,header",
    [
        ("a,b", ["a", "b"], [0]),
        (
            "a,b\nc,d",
            MultiIndex.from_tuples([("a", "c"), ("b", "d")]),
            [0, 1],
        ),
    ],
)
@pytest.mark.parametrize("round_trip", [True, False])
def test_multi_index_blank_df(all_parsers, data, columns, header, round_trip):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies that header-only input with a single
    # or two header rows (and the same content produced by writing an empty frame) gives
    # an empty frame with plain or multi-level columns.
    # see gh-14545
    parser = all_parsers
    expected = DataFrame(columns=columns)
    data = expected.to_csv(index=False) if round_trip else data

    result = parser.read_csv(StringIO(data), header=header)
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # AssertionError: DataFrame.columns are different
def test_no_unnamed_index(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies that space-separated data with a
    # leading space in the header line produces a first column named Unnamed with its
    # position and does not turn it into the index.
    parser = all_parsers
    data = """ id c0 c1 c2
0 1 0 a b
1 2 0 c d
2 2 2 e f
"""
    result = parser.read_csv(StringIO(data), sep=" ")
    expected = DataFrame(
        [[0, 1, 0, "a", "b"], [1, 2, 0, "c", "d"], [2, 2, 2, "e", "f"]],
        columns=["Unnamed: 0", "id", "c0", "c1", "c2"],
    )
    tm.assert_frame_equal(result, expected)


def test_read_duplicate_index_explicit(all_parsers):
    # [FUNCTION DOC]
    # Verifies that duplicate values in the column used as an explicit index are kept as
    # duplicate labels, with the index named after the header.
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo,12,13,14,15
bar,12,13,14,15
"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data), index_col=0)

    expected = DataFrame(
        [
            [2, 3, 4, 5],
            [7, 8, 9, 10],
            [12, 13, 14, 15],
            [12, 13, 14, 15],
            [12, 13, 14, 15],
            [12, 13, 14, 15],
        ],
        columns=["A", "B", "C", "D"],
        index=Index(["foo", "bar", "baz", "qux", "foo", "bar"], name="index"),
    )
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
def test_read_duplicate_index_implicit(all_parsers):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Same as the explicit case but the index is implicit (header
    # has one fewer name than data fields), giving an unnamed index with duplicates.
    data = """A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo,12,13,14,15
bar,12,13,14,15
"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data))

    expected = DataFrame(
        [
            [2, 3, 4, 5],
            [7, 8, 9, 10],
            [12, 13, 14, 15],
            [12, 13, 14, 15],
            [12, 13, 14, 15],
            [12, 13, 14, 15],
        ],
        columns=["A", "B", "C", "D"],
        index=Index(["foo", "bar", "baz", "qux", "foo", "bar"]),
    )
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
def test_read_csv_no_index_name(all_parsers, csv_dir_path):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Reads a sample file whose header lacks a name for the first
    # column, with that column as index and dates parsed, and verifies the five-row
    # frame with an unnamed microsecond datetime index and a string column.
    parser = all_parsers
    csv2 = os.path.join(csv_dir_path, "test2.csv")
    result = parser.read_csv(csv2, index_col=0, parse_dates=True)

    expected = DataFrame(
        [
            [0.980269, 3.685731, -0.364216805298, -1.159738, "foo"],
            [1.047916, -0.041232, -0.16181208307, 0.212549, "bar"],
            [0.498581, 0.731168, -0.537677223318, 1.346270, "baz"],
            [1.120202, 1.567621, 0.00364077397681, 0.675253, "qux"],
            [-0.487094, 0.571455, -1.6116394093, 0.103469, "foo2"],
        ],
        columns=["A", "B", "C", "D", "E"],
        index=Index(
            [
                datetime(2000, 1, 3),
                datetime(2000, 1, 4),
                datetime(2000, 1, 5),
                datetime(2000, 1, 6),
                datetime(2000, 1, 7),
            ],
            dtype="M8[us]",
        ),
    )
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
def test_empty_with_index(all_parsers):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Verifies header-only input with the first column as index
    # gives an empty frame with the second column and an empty index named after the
    # first.
    # see gh-10184
    data = "x,y"
    parser = all_parsers
    result = parser.read_csv(StringIO(data), index_col=0)

    expected = DataFrame(columns=["y"], index=Index([], name="x"))
    tm.assert_frame_equal(result, expected)


# CSV parse error: Empty CSV file or block: cannot infer number of columns
@skip_pyarrow
def test_empty_with_multi_index(all_parsers):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Verifies header-only input with the first two columns as a
    # named multi-level index gives an empty frame with one column and an empty
    # two-level index.
    # see gh-10467
    data = "x,y,z"
    parser = all_parsers
    result = parser.read_csv(StringIO(data), index_col=["x", "y"])

    expected = DataFrame(
        columns=["z"], index=MultiIndex.from_arrays([[]] * 2, names=["x", "y"])
    )
    tm.assert_frame_equal(result, expected)


# CSV parse error: Empty CSV file or block: cannot infer number of columns
@skip_pyarrow
def test_empty_with_reversed_multi_index(all_parsers):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Same as the previous test but the index columns are given in
    # reverse order, so the index level names follow that reversed order.
    data = "x,y,z"
    parser = all_parsers
    result = parser.read_csv(StringIO(data), index_col=[1, 0])

    expected = DataFrame(
        columns=["z"], index=MultiIndex.from_arrays([[]] * 2, names=["y", "x"])
    )
    tm.assert_frame_equal(result, expected)
