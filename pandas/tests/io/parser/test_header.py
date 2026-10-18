"""
Tests that the file header is properly handled or inferred
during parsing for all of the parsers defined in parsers.py
"""

# [MODULE DOC]
# Parser tests (all engines via the parser fixture) of header handling: invalid header
# arguments, header not on the first line, multi-level headers (valid, invalid, common
# formats and malformed variants), blank lines inside a multi-row header, interplay with
# names, skipped rows, index columns and implicit indexes, mangling of duplicate
# multi-level header entries, placeholder names for unnamed levels, and
# whitespace-separated headers.
# KNOWN ISSUE: the module docstring says the parsers are defined in a file named
# parsers.py, but no such module exists here; the parser fixtures come from the
# package's shared fixtures module.
# Defines two marker aliases (expected failure and skip for pyarrow) and a named tuple
# type with fields first and second, used to supply column names as tuples.

from collections import namedtuple
from io import StringIO

import numpy as np
import pytest

from pandas.errors import ParserError

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


@xfail_pyarrow  # TypeError: an integer is required
def test_read_with_bad_header(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies a header row index beyond the end of
    # the data raises a value error saying there were only a few lines in the file.
    parser = all_parsers
    msg = r"but only \d+ lines in file"

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(",,"), header=[10])


def test_negative_header(all_parsers):
    # [FUNCTION DOC]
    # Verifies a negative integer header raises a value error saying it is invalid and
    # suggesting using none for no header.
    # see gh-27779
    parser = all_parsers
    data = """1,2,3,4,5
6,7,8,9,10
11,12,13,14,15
"""
    with pytest.raises(
        ValueError,
        match="Passing negative integer to header is invalid. "
        "For no header, use header=None instead",
    ):
        parser.read_csv(StringIO(data), header=-1)


@pytest.mark.parametrize("header", [([-1, 2, 4]), ([-5, 0])])
def test_negative_multi_index_header(all_parsers, header):
    # [FUNCTION DOC]
    # Verifies lists of header rows that contain negative integers raise a value error
    # saying a multi-index header cannot specify negative integers.
    # see gh-27779
    parser = all_parsers
    data = """1,2,3,4,5
        6,7,8,9,10
        11,12,13,14,15
        """
    with pytest.raises(
        ValueError, match="cannot specify multi-index header with negative integers"
    ):
        parser.read_csv(StringIO(data), header=header)


@pytest.mark.parametrize("header", [True, False])
def test_bool_header_arg(all_parsers, header):
    # [FUNCTION DOC]
    # Verifies passing a boolean as header (true or false) raises a type error saying
    # passing a bool to header is invalid.
    # see gh-6114
    parser = all_parsers
    data = """\
MyColumn
a
b
a
b"""
    msg = "Passing a bool to header is invalid"
    with pytest.raises(TypeError, match=msg):
        parser.read_csv(StringIO(data), header=header)


@xfail_pyarrow  # AssertionError: DataFrame are different
def test_header_with_index_col(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies that with three names given for four
    # data fields, the first field becomes the (unnamed) index and the rest take the
    # given names.
    parser = all_parsers
    data = """foo,1,2,3
bar,4,5,6
baz,7,8,9
"""
    names = ["A", "B", "C"]
    result = parser.read_csv(StringIO(data), names=names)

    expected = DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        index=["foo", "bar", "baz"],
        columns=["A", "B", "C"],
    )
    tm.assert_frame_equal(result, expected)


def test_header_not_first_line(all_parsers):
    # [FUNCTION DOC]
    # Verifies that a header located at row two (with index column zero) after two junk
    # lines gives the same frame as reading the same data without the junk and with the
    # header at row zero.
    parser = all_parsers
    data = """got,to,ignore,this,line
got,to,ignore,this,line
index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
"""
    data2 = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
"""

    result = parser.read_csv(StringIO(data), header=2, index_col=0)
    expected = parser.read_csv(StringIO(data2), header=0, index_col=0)
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
def test_header_multi_index(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies a four-row header combined with two
    # index columns and a blank line after the first header row yields a four-level
    # column index and a two-level row index with names taken from the row below the
    # headers.
    parser = all_parsers

    data = """\
C0,,C_l0_g0,C_l0_g1,C_l0_g2

C1,,C_l1_g0,C_l1_g1,C_l1_g2
C2,,C_l2_g0,C_l2_g1,C_l2_g2
C3,,C_l3_g0,C_l3_g1,C_l3_g2
R0,R1,,,
R_l0_g0,R_l1_g0,R0C0,R0C1,R0C2
R_l0_g1,R_l1_g1,R1C0,R1C1,R1C2
R_l0_g2,R_l1_g2,R2C0,R2C1,R2C2
R_l0_g3,R_l1_g3,R3C0,R3C1,R3C2
R_l0_g4,R_l1_g4,R4C0,R4C1,R4C2
"""
    result = parser.read_csv(StringIO(data), header=[0, 1, 2, 3], index_col=[0, 1])
    data_gen_f = lambda r, c: f"R{r}C{c}"

    data = [[data_gen_f(r, c) for c in range(3)] for r in range(5)]
    index = MultiIndex.from_arrays(
        [[f"R_l0_g{i}" for i in range(5)], [f"R_l1_g{i}" for i in range(5)]],
        names=["R0", "R1"],
    )
    columns = MultiIndex.from_arrays(
        [
            [f"C_l0_g{i}" for i in range(3)],
            [f"C_l1_g{i}" for i in range(3)],
            [f"C_l2_g{i}" for i in range(3)],
            [f"C_l3_g{i}" for i in range(3)],
        ],
        names=["C0", "C1", "C2", "C3"],
    )
    expected = DataFrame(data, columns=columns, index=index)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "kwargs,msg",
    [
        (
            {"index_col": ["foo", "bar"]},
            (
                "index_col must only contain "
                "integers of column positions when specifying "
                "a multi-index header"
            ),
        ),
        (
            {"index_col": [0, 1], "names": ["foo", "bar"]},
            ("cannot specify names when specifying a multi-index header"),
        ),
        (
            {"index_col": [0, 1], "usecols": ["foo", "bar"]},
            ("cannot specify usecols when specifying a multi-index header"),
        ),
    ],
)
def test_header_multi_index_invalid(all_parsers, kwargs, msg):
    # [FUNCTION DOC]
    # Verifies three invalid option combinations with a multi-row header each raise a
    # value error: index columns given by name, names given, and column selection given.
    data = """\
C0,,C_l0_g0,C_l0_g1,C_l0_g2

C1,,C_l1_g0,C_l1_g1,C_l1_g2
C2,,C_l2_g0,C_l2_g1,C_l2_g2
C3,,C_l3_g0,C_l3_g1,C_l3_g2
R0,R1,,,
R_l0_g0,R_l1_g0,R0C0,R0C1,R0C2
R_l0_g1,R_l1_g1,R1C0,R1C1,R1C2
R_l0_g2,R_l1_g2,R2C0,R2C1,R2C2
R_l0_g3,R_l1_g3,R3C0,R3C1,R3C2
R_l0_g4,R_l1_g4,R4C0,R4C1,R4C2
"""
    parser = all_parsers

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), header=[0, 1, 2, 3], **kwargs)


_TestTuple = namedtuple("_TestTuple", ["first", "second"])


@xfail_pyarrow  # TypeError: an integer is required
@pytest.mark.parametrize(
    "kwargs",
    [
        {"header": [0, 1]},
        {
            "skiprows": 3,
            "names": [
                ("a", "q"),
                ("a", "r"),
                ("a", "s"),
                ("b", "t"),
                ("c", "u"),
                ("c", "v"),
            ],
        },
        {
            "skiprows": 3,
            "names": [
                _TestTuple("a", "q"),
                _TestTuple("a", "r"),
                _TestTuple("a", "s"),
                _TestTuple("b", "t"),
                _TestTuple("c", "u"),
                _TestTuple("c", "v"),
            ],
        },
    ],
)
def test_header_multi_index_common_format1(all_parsers, kwargs):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies a common spreadsheet-export layout
    # (index label row blank, two header rows, a blank separator row) is read, via a
    # two-row header or via skipping three rows with names given as plain tuples or
    # named tuples, into a two-row frame with two-level columns and index named by the
    # first column.
    parser = all_parsers
    expected = DataFrame(
        [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
        index=["one", "two"],
        columns=MultiIndex.from_tuples(
            [("a", "q"), ("a", "r"), ("a", "s"), ("b", "t"), ("c", "u"), ("c", "v")]
        ),
    )
    data = """,a,a,a,b,c,c
,q,r,s,t,u,v
,,,,,,
one,1,2,3,4,5,6
two,7,8,9,10,11,12"""

    result = parser.read_csv(StringIO(data), index_col=0, **kwargs)
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
@pytest.mark.parametrize(
    "kwargs",
    [
        {"header": [0, 1]},
        {
            "skiprows": 2,
            "names": [
                ("a", "q"),
                ("a", "r"),
                ("a", "s"),
                ("b", "t"),
                ("c", "u"),
                ("c", "v"),
            ],
        },
        {
            "skiprows": 2,
            "names": [
                _TestTuple("a", "q"),
                _TestTuple("a", "r"),
                _TestTuple("a", "s"),
                _TestTuple("b", "t"),
                _TestTuple("c", "u"),
                _TestTuple("c", "v"),
            ],
        },
    ],
)
def test_header_multi_index_common_format2(all_parsers, kwargs):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Same as the first format but without the
    # blank separator row, skipping two rows when names are given.
    parser = all_parsers
    expected = DataFrame(
        [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
        index=["one", "two"],
        columns=MultiIndex.from_tuples(
            [("a", "q"), ("a", "r"), ("a", "s"), ("b", "t"), ("c", "u"), ("c", "v")]
        ),
    )
    data = """,a,a,a,b,c,c
,q,r,s,t,u,v
one,1,2,3,4,5,6
two,7,8,9,10,11,12"""

    result = parser.read_csv(StringIO(data), index_col=0, **kwargs)
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
@pytest.mark.parametrize(
    "kwargs",
    [
        {"header": [0, 1]},
        {
            "skiprows": 2,
            "names": [
                ("a", "q"),
                ("a", "r"),
                ("a", "s"),
                ("b", "t"),
                ("c", "u"),
                ("c", "v"),
            ],
        },
        {
            "skiprows": 2,
            "names": [
                _TestTuple("a", "q"),
                _TestTuple("a", "r"),
                _TestTuple("a", "s"),
                _TestTuple("b", "t"),
                _TestTuple("c", "u"),
                _TestTuple("c", "v"),
            ],
        },
    ],
)
def test_header_multi_index_common_format3(all_parsers, kwargs):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Same header layout but without an index
    # column and without a blank line (index columns off): verifies the frame has a
    # default range index and two-level columns.
    parser = all_parsers
    expected = DataFrame(
        [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
        index=["one", "two"],
        columns=MultiIndex.from_tuples(
            [("a", "q"), ("a", "r"), ("a", "s"), ("b", "t"), ("c", "u"), ("c", "v")]
        ),
    )
    expected = expected.reset_index(drop=True)
    data = """a,a,a,b,c,c
q,r,s,t,u,v
1,2,3,4,5,6
7,8,9,10,11,12"""

    result = parser.read_csv(StringIO(data), index_col=None, **kwargs)
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
def test_header_multi_index_common_format_malformed1(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies a two-row header whose first-column
    # cells name both an index (from the first row) and a column-level name: the first
    # column becomes the index and the column levels get the names from the first column
    # of the header rows.
    parser = all_parsers
    expected = DataFrame(
        np.array([[2, 3, 4, 5, 6], [8, 9, 10, 11, 12]], dtype="int64"),
        index=Index([1, 7]),
        columns=MultiIndex(
            levels=[["a", "b", "c"], ["r", "s", "t", "u", "v"]],
            codes=[[0, 0, 1, 2, 2], [0, 1, 2, 3, 4]],
            names=["a", "q"],
        ),
    )
    data = """a,a,a,b,c,c
q,r,s,t,u,v
1,2,3,4,5,6
7,8,9,10,11,12"""

    result = parser.read_csv(StringIO(data), header=[0, 1], index_col=0)
    tm.assert_frame_equal(expected, result)


@xfail_pyarrow  # TypeError: an integer is required
def test_header_multi_index_common_format_malformed2(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Same as the previous malformed case but with
    # an empty first header cell, giving an unnamed first level and a default-named
    # index values 1 and 7.
    parser = all_parsers
    expected = DataFrame(
        np.array([[2, 3, 4, 5, 6], [8, 9, 10, 11, 12]], dtype="int64"),
        index=range(1, 13, 6),
        columns=MultiIndex(
            levels=[["a", "b", "c"], ["r", "s", "t", "u", "v"]],
            codes=[[0, 0, 1, 2, 2], [0, 1, 2, 3, 4]],
            names=[None, "q"],
        ),
    )

    data = """,a,a,b,c,c
q,r,s,t,u,v
1,2,3,4,5,6
7,8,9,10,11,12"""

    result = parser.read_csv(StringIO(data), header=[0, 1], index_col=0)
    tm.assert_frame_equal(expected, result)


@xfail_pyarrow  # TypeError: an integer is required
def test_header_multi_index_common_format_malformed3(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies a malformed layout with two index
    # columns and a two-row header gives a two-level row index and four columns with a
    # two-level header.
    parser = all_parsers
    expected = DataFrame(
        np.array([[3, 4, 5, 6], [9, 10, 11, 12]], dtype="int64"),
        index=MultiIndex(levels=[[1, 7], [2, 8]], codes=[[0, 1], [0, 1]]),
        columns=MultiIndex(
            levels=[["a", "b", "c"], ["s", "t", "u", "v"]],
            codes=[[0, 1, 2, 2], [0, 1, 2, 3]],
            names=[None, "q"],
        ),
    )
    data = """,a,a,b,c,c
q,r,s,t,u,v
1,2,3,4,5,6
7,8,9,10,11,12"""

    result = parser.read_csv(StringIO(data), header=[0, 1], index_col=[0, 1])
    tm.assert_frame_equal(expected, result)


@xfail_pyarrow  # TypeError: an integer is required
def test_header_multi_index_blank_line(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies a blank-valued row after a two-row
    # header becomes a data row of missing values rather than being skipped.
    # GH 40442
    parser = all_parsers
    data = [[None, None], [1, 2], [3, 4]]
    columns = MultiIndex.from_tuples([("a", "A"), ("b", "B")])
    expected = DataFrame(data, columns=columns)
    data = "a,b\nA,B\n,\n1,2\n3,4"
    result = parser.read_csv(StringIO(data), header=[0, 1])
    tm.assert_frame_equal(expected, result)


@pytest.mark.parametrize(
    "data,header", [("1,2,3\n4,5,6", None), ("foo,bar,baz\n1,2,3\n4,5,6", 0)]
)
def test_header_names_backward_compat(all_parsers, data, header, request):
    # [FUNCTION DOC]
    # Verifies that names given with a header at row zero or with no header produce the
    # same frame as reading the headerless data with names; for pyarrow the header case
    # is marked as expected to fail.
    # see gh-2539
    parser = all_parsers

    if parser.engine == "pyarrow" and header is not None:
        mark = pytest.mark.xfail(reason="DataFrame.columns are different")
        request.applymarker(mark)

    expected = parser.read_csv(StringIO("1,2,3\n4,5,6"), names=["a", "b", "c"])

    result = parser.read_csv(StringIO(data), names=["a", "b", "c"], header=header)
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Empty CSV file or block: cannot infer
@pytest.mark.parametrize("kwargs", [{}, {"index_col": False}])
def test_read_only_header_no_rows(all_parsers, kwargs):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Verifies header-only input (with or without the index column
    # option turned off) gives an empty frame with the three header columns.
    # See gh-7773
    parser = all_parsers
    expected = DataFrame(columns=["a", "b", "c"])

    result = parser.read_csv(StringIO("a,b,c"), **kwargs)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "kwargs,names",
    [
        ({}, [0, 1, 2, 3, 4]),
        (
            {"names": ["foo", "bar", "baz", "quux", "panda"]},
            ["foo", "bar", "baz", "quux", "panda"],
        ),
    ],
)
def test_no_header(all_parsers, kwargs, names):
    # [FUNCTION DOC]
    # Verifies that with no header, columns are named by integer positions by default,
    # or by the given names.
    parser = all_parsers
    data = """1,2,3,4,5
6,7,8,9,10
11,12,13,14,15
"""
    expected = DataFrame(
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]], columns=names
    )
    result = parser.read_csv(StringIO(data), header=None, **kwargs)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("header", [["a", "b"], "string_header"])
def test_non_int_header(all_parsers, header):
    # [FUNCTION DOC]
    # Verifies a list of strings or a plain string as header raises a value error that
    # the header must be an integer or list of integers.
    # see gh-16338
    msg = "header must be integer or list of integers"
    data = """1,2\n3,4"""
    parser = all_parsers

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), header=header)


@xfail_pyarrow  # TypeError: an integer is required
def test_singleton_header(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies a header given as a one-element list
    # behaves like a single header row.
    # see gh-7757
    data = """a,b,c\n0,1,2\n1,2,3"""
    parser = all_parsers

    expected = DataFrame({"a": [0, 1], "b": [1, 2], "c": [2, 3]})
    result = parser.read_csv(StringIO(data), header=[0])
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
@pytest.mark.parametrize(
    "data,expected",
    [
        (
            "A,A,A,B\none,one,one,two\n0,40,34,0.1",
            DataFrame(
                [[0, 40, 34, 0.1]],
                columns=MultiIndex.from_tuples(
                    [("A", "one"), ("A", "one.1"), ("A", "one.2"), ("B", "two")]
                ),
            ),
        ),
        (
            "A,A,A,B\none,one,one.1,two\n0,40,34,0.1",
            DataFrame(
                [[0, 40, 34, 0.1]],
                columns=MultiIndex.from_tuples(
                    [("A", "one"), ("A", "one.1"), ("A", "one.1.1"), ("B", "two")]
                ),
            ),
        ),
        (
            "A,A,A,B,B\none,one,one.1,two,two\n0,40,34,0.1,0.1",
            DataFrame(
                [[0, 40, 34, 0.1, 0.1]],
                columns=MultiIndex.from_tuples(
                    [
                        ("A", "one"),
                        ("A", "one.1"),
                        ("A", "one.1.1"),
                        ("B", "two"),
                        ("B", "two.1"),
                    ]
                ),
            ),
        ),
    ],
)
def test_mangles_multi_index(all_parsers, data, expected):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies that duplicate entries in the second
    # header level are mangled with numeric suffixes within each first-level group,
    # avoiding clashes with names already containing a suffix.
    # see gh-18062
    parser = all_parsers

    result = parser.read_csv(StringIO(data), header=[0, 1])
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
@pytest.mark.parametrize("index_col", [None, [0]])
@pytest.mark.parametrize(
    "columns", [None, (["", "Unnamed"]), (["Unnamed", ""]), (["Unnamed", "NotUnnamed"])]
)
def test_multi_index_unnamed(all_parsers, index_col, columns):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Regression for a multi-row header in which
    # one header row consists entirely of placeholder names. Verifies no error is raised
    # and the unnamed entries get placeholder names containing their column position
    # (shifted by one when an index column is used) and level.
    # see gh-23687
    #
    # When specifying a multi-index header, make sure that
    # we don't error just because one of the rows in our header
    # has ALL column names containing the string "Unnamed". The
    # correct condition to check is whether the row contains
    # ALL columns that did not have names (and instead were given
    # placeholder ones).
    parser = all_parsers
    header = [0, 1]

    if index_col is None:
        data = ",".join(columns or ["", ""]) + "\n0,1\n2,3\n4,5\n"
    else:
        data = ",".join([""] + (columns or ["", ""])) + "\n,0,1\n0,2,3\n1,4,5\n"

    result = parser.read_csv(StringIO(data), header=header, index_col=index_col)
    exp_columns = []

    if columns is None:
        columns = ["", "", ""]

    for i, col in enumerate(columns):
        if not col:  # Unnamed.
            col = f"Unnamed: {i if index_col is None else i + 1}_level_0"

        exp_columns.append(col)

    columns = MultiIndex.from_tuples(zip(exp_columns, ["0", "1"]))
    expected = DataFrame([[2, 3], [4, 5]], columns=columns)
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Expected 2 columns, got 3
def test_names_longer_than_header_but_equal_with_data_rows(all_parsers):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Verifies that three names given for a two-field header with
    # three-field data rows replace the header, and all three columns are read.
    # GH#38453
    parser = all_parsers
    data = """a, b
1,2,3
5,6,4
"""
    result = parser.read_csv(StringIO(data), header=0, names=["A", "B", "C"])
    expected = DataFrame({"A": [1, 5], "B": [2, 6], "C": [3, 4]})
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
def test_read_csv_multiindex_columns(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies a two-row header with duplicate
    # entries and spaces after commas is read to two-level columns with the second-level
    # duplicates mangled, for inputs with one and two data rows.
    # GH#6051
    parser = all_parsers

    s1 = "Male, Male, Male, Female, Female\nR, R, L, R, R\n.86, .67, .88, .78, .81"
    s2 = (
        "Male, Male, Male, Female, Female\n"
        "R, R, L, R, R\n"
        ".86, .67, .88, .78, .81\n"
        ".86, .67, .88, .78, .82"
    )

    mi = MultiIndex.from_tuples(
        [
            ("Male", "R"),
            (" Male", " R"),
            (" Male", " L"),
            (" Female", " R"),
            (" Female", " R.1"),
        ]
    )
    expected = DataFrame(
        [[0.86, 0.67, 0.88, 0.78, 0.81], [0.86, 0.67, 0.88, 0.78, 0.82]], columns=mi
    )

    df1 = parser.read_csv(StringIO(s1), header=[0, 1])
    tm.assert_frame_equal(df1, expected.iloc[:1])
    df2 = parser.read_csv(StringIO(s2), header=[0, 1])
    tm.assert_frame_equal(df2, expected)


@xfail_pyarrow  # TypeError: an integer is required
def test_read_csv_multi_header_length_check(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies that header rows of different field
    # counts (given header rows zero and two) raise a parser error saying header rows
    # must have an equal number of columns.
    # GH#43102
    parser = all_parsers

    case = """row11,row12,row13
row21,row22, row23
row31,row32
"""

    with pytest.raises(
        ParserError, match="Header rows must have an equal number of columns."
    ):
        parser.read_csv(StringIO(case), header=[0, 2])


@skip_pyarrow  # CSV parse error: Expected 3 columns, got 2
def test_header_none_and_implicit_index(all_parsers):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Verifies that with no header, two names and rows of three
    # fields followed by shorter rows, the first field becomes the index and missing
    # values pad short rows.
    # GH#22144
    parser = all_parsers
    data = "x,1,5\ny,2\nz,3\n"
    result = parser.read_csv(StringIO(data), names=["a", "b"], header=None)
    expected = DataFrame(
        {"a": [1, 2, 3], "b": [5, np.nan, np.nan]}, index=["x", "y", "z"]
    )
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # regex mismatch "CSV parse error: Expected 2 columns, got "
def test_header_none_and_implicit_index_in_second_row(all_parsers):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Verifies that with no header and two names, a second row with
    # three fields raises a parser error naming line two with three fields where two are
    # expected.
    # GH#22144
    parser = all_parsers
    data = "x,1\ny,2,5\nz,3\n"
    with pytest.raises(ParserError, match="Expected 2 fields in line 2, saw 3"):
        parser.read_csv(StringIO(data), names=["a", "b"], header=None)


def test_header_none_and_on_bad_lines_skip(all_parsers):
    # [FUNCTION DOC]
    # Verifies that with no header, two names and skipping of bad lines, the overlong
    # row is dropped and two rows remain.
    # GH#22144
    parser = all_parsers
    data = "x,1\ny,2,5\nz,3\n"
    result = parser.read_csv(
        StringIO(data), names=["a", "b"], header=None, on_bad_lines="skip"
    )
    expected = DataFrame({"a": ["x", "z"], "b": [1, 3]})
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
def test_header_missing_rows(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies that asking for a three-row header
    # when the file has two lines raises a value error mentioning the header list, its
    # length and the number of lines in the file.
    # GH#47400
    parser = all_parsers
    data = """a,b
1,2
"""
    msg = r"Passed header=\[0,1,2\], len of 3, but only 2 lines in file"
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), header=[0, 1, 2])


# ValueError: the 'pyarrow' engine does not support regex separators
@xfail_pyarrow
def test_header_multiple_whitespaces(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow (regex separator). Verifies that header and
    # data separated by runs of whitespace (with leading whitespace on the data line)
    # are read with a whitespace regex separator, keeping parentheses and commas inside
    # names.
    # GH#54931
    parser = all_parsers
    data = """aa    bb(1,1)   cc(1,1)
                0  2  3.5"""

    result = parser.read_csv(StringIO(data), sep=r"\s+")
    expected = DataFrame({"aa": [0], "bb(1,1)": 2, "cc(1,1)": 3.5})
    tm.assert_frame_equal(result, expected)


# ValueError: the 'pyarrow' engine does not support regex separators
@xfail_pyarrow
def test_header_delim_whitespace(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow (regex separator). Verifies that
    # comma-separated lines read with a whitespace regex separator yield a single column
    # named a,b with each line as a value (the trailing whitespace-only line is
    # dropped).
    # KNOWN ISSUE: the test name refers to the whitespace-delimiter option, but the
    # test uses a whitespace regex separator instead.
    # GH#54918
    parser = all_parsers
    data = """a,b
1,2
3,4
    """
    result = parser.read_csv(StringIO(data), sep=r"\s+")
    expected = DataFrame({"a,b": ["1,2", "3,4"]})
    tm.assert_frame_equal(result, expected)


def test_usecols_no_header_pyarrow(pyarrow_parser_only):
    # [FUNCTION DOC]
    # Uses the pyarrow-only parser fixture. Verifies that reading with no header, a
    # column selection of the first two columns, string pyarrow dtype and the pyarrow
    # dtype backend returns a two-column frame of arrow strings, skipping the leading
    # blank line.
    parser = pyarrow_parser_only
    data = """
a,i,x
b,j,y
"""
    result = parser.read_csv(
        StringIO(data),
        header=None,
        usecols=[0, 1],
        dtype="string[pyarrow]",
        dtype_backend="pyarrow",
        engine="pyarrow",
    )
    expected = DataFrame([["a", "i"], ["b", "j"]], dtype="string[pyarrow]")
    tm.assert_frame_equal(result, expected)
