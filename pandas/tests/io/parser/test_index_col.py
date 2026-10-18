"""
Tests that the specified index column (a.k.a "index_col")
is properly handled or inferred during parsing for all of
the parsers defined in parsers.py
"""

# [MODULE DOC]
# Parser tests (all engines via the parser fixture) of the index column option: index by
# name or position, rejected values, inferred index when data rows are wider than the
# header, index column handling with empty data, multi-level indexes and their naming,
# multi-level column headers combined with index columns, large inputs, and dtypes for
# index columns.
# KNOWN ISSUE: the module docstring says the parsers are defined in a file named
# parsers.py, but no such module exists here; the parser fixtures come from the
# package's shared fixtures module.
# Defines two marker aliases: one marks a test as expected to fail for the pyarrow
# engine and one skips it for that engine.

from io import StringIO

import numpy as np
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


@pytest.mark.parametrize("with_header", [True, False])
def test_index_col_named(all_parsers, with_header):
    # [FUNCTION DOC]
    # Verifies selecting the index column by name when a header is present equals
    # reading the frame and setting the index afterwards, and that without a header (no
    # such column name) a value error saying the index name is invalid is raised.
    parser = all_parsers
    no_header = """\
KORD1,19990127, 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000
KORD2,19990127, 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000
KORD3,19990127, 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000
KORD4,19990127, 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000
KORD5,19990127, 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000
KORD6,19990127, 23:00:00, 22:56:00, -0.5900, 1.7100, 4.6000, 0.0000, 280.0000"""
    header = "ID,date,NominalTime,ActualTime,TDew,TAir,Windspeed,Precip,WindDir\n"

    if with_header:
        data = header + no_header

        result = parser.read_csv(StringIO(data), index_col="ID")
        expected = parser.read_csv(StringIO(data), header=0).set_index("ID")
        tm.assert_frame_equal(result, expected)
    else:
        data = no_header
        msg = "Index ID invalid"

        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), index_col="ID")


def test_index_col_named2(all_parsers):
    # [FUNCTION DOC]
    # Verifies a name given in the names list can be used as the index column (a
    # one-element list), producing a named string index and the remaining columns.
    parser = all_parsers
    data = """\
1,2,3,4,hello
5,6,7,8,world
9,10,11,12,foo
"""

    expected = DataFrame(
        {"a": [1, 5, 9], "b": [2, 6, 10], "c": [3, 7, 11], "d": [4, 8, 12]},
        index=Index(["hello", "world", "foo"], name="message"),
    )
    names = ["a", "b", "c", "d", "message"]

    result = parser.read_csv(StringIO(data), names=names, index_col=["message"])
    tm.assert_frame_equal(result, expected)


def test_index_col_is_true(all_parsers):
    # [FUNCTION DOC]
    # Verifies passing true as the index column raises a value error saying the value
    # cannot be true.
    # see gh-9798
    data = "a,b\n1,2"
    parser = all_parsers

    msg = "The value of index_col couldn't be 'True'"
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), index_col=True)


@skip_pyarrow  # CSV parse error: Expected 3 columns, got 4
def test_infer_index_col(all_parsers):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Verifies that data rows with one more field than the header
    # produce an implicit unnamed index from the first field.
    data = """A,B,C
foo,1,2,3
bar,4,5,6
baz,7,8,9
"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data))

    expected = DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        index=["foo", "bar", "baz"],
        columns=["A", "B", "C"],
    )
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Empty CSV file or block
@pytest.mark.parametrize(
    "index_col,kwargs",
    [
        (None, {"columns": ["x", "y", "z"]}),
        (False, {"columns": ["x", "y", "z"]}),
        (0, {"columns": ["y", "z"], "index": Index([], name="x")}),
        (1, {"columns": ["x", "z"], "index": Index([], name="y")}),
        ("x", {"columns": ["y", "z"], "index": Index([], name="x")}),
        ("y", {"columns": ["x", "z"], "index": Index([], name="y")}),
        (
            [0, 1],
            {
                "columns": ["z"],
                "index": MultiIndex.from_arrays([[]] * 2, names=["x", "y"]),
            },
        ),
        (
            ["x", "y"],
            {
                "columns": ["z"],
                "index": MultiIndex.from_arrays([[]] * 2, names=["x", "y"]),
            },
        ),
        (
            [1, 0],
            {
                "columns": ["z"],
                "index": MultiIndex.from_arrays([[]] * 2, names=["y", "x"]),
            },
        ),
        (
            ["y", "x"],
            {
                "columns": ["z"],
                "index": MultiIndex.from_arrays([[]] * 2, names=["y", "x"]),
            },
        ),
    ],
)
def test_index_col_empty_data(all_parsers, index_col, kwargs):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Parametrized over index column choices (none, false,
    # position, name, lists of positions or names in both orders): verifies that
    # header-only input gives an empty frame with the right remaining columns and an
    # empty index (single or multi-level) named after the chosen columns in the chosen
    # order.
    data = "x,y,z"
    parser = all_parsers
    result = parser.read_csv(StringIO(data), index_col=index_col)

    expected = DataFrame(**kwargs)
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Empty CSV file or block
def test_empty_with_index_col_false(all_parsers):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Verifies header-only input with the index column turned off
    # gives an empty frame with both header columns.
    # see gh-10413
    data = "x,y"
    parser = all_parsers
    result = parser.read_csv(StringIO(data), index_col=False)

    expected = DataFrame(columns=["x", "y"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "index_names",
    [
        ["", ""],
        ["foo", ""],
        ["", "bar"],
        ["foo", "bar"],
        ["NotReallyUnnamed", "Unnamed: 0"],
    ],
)
def test_multi_index_naming(all_parsers, index_names, request):
    # [FUNCTION DOC]
    # Parametrized over index level name pairs (empty, partly empty, named, and names
    # resembling placeholders). Verifies that with two index columns the empty names
    # stay unset (they are not replaced by placeholder names) and the four-row frame has
    # the product index. For pyarrow, cases with an empty name are marked expected to
    # fail.
    parser = all_parsers

    if parser.engine == "pyarrow" and "" in index_names:
        mark = pytest.mark.xfail(reason="One case raises, others are wrong")
        request.applymarker(mark)

    # We don't want empty index names being replaced with "Unnamed: 0"
    data = ",".join([*index_names, "col\na,c,1\na,d,2\nb,c,3\nb,d,4"])
    result = parser.read_csv(StringIO(data), index_col=[0, 1])

    expected = DataFrame(
        {"col": [1, 2, 3, 4]}, index=MultiIndex.from_product([["a", "b"], ["c", "d"]])
    )
    expected.index.names = [name if name else None for name in index_names]
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # ValueError: Found non-unique column index
def test_multi_index_naming_not_all_at_beginning(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies two index columns that are not
    # adjacent (first and third) give a multi-level index without names and keep the
    # middle column (named with a placeholder-looking name) as the data column.
    parser = all_parsers
    data = ",Unnamed: 2,\na,c,1\na,d,2\nb,c,3\nb,d,4"
    result = parser.read_csv(StringIO(data), index_col=[0, 2])

    expected = DataFrame(
        {"Unnamed: 2": ["c", "d", "c", "d"]},
        index=MultiIndex(
            levels=[["a", "b"], [1, 2, 3, 4]], codes=[[0, 0, 1, 1], [0, 1, 2, 3]]
        ),
    )
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # ValueError: Found non-unique column index
def test_no_multi_index_level_names_empty(temp_file, all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Writes a random frame with a three-level
    # unnamed index (with a repeated entry) to a temporary CSV and verifies reading it
    # with the first three columns as index restores the frame.
    # Side effect: writes the temporary file.
    # GH 10984
    parser = all_parsers
    midx = MultiIndex.from_tuples([("A", 1, 2), ("A", 1, 2), ("B", 1, 2)])
    expected = DataFrame(
        np.random.default_rng(2).standard_normal((3, 3)),
        index=midx,
        columns=["x", "y", "z"],
    )
    expected.to_csv(temp_file)
    result = parser.read_csv(temp_file, index_col=[0, 1, 2])
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
def test_header_with_index_col(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Regression for header and index interplay
    # with duplicate column names: verifies a two-row header with the first column as
    # index gives two-level columns (second duplicate mangled) and index named by the
    # first header row's cell; and that a single header row with the index chosen by
    # name keeps the second row as data strings.
    # GH 33476
    parser = all_parsers
    data = """
I11,A,A
I12,B,B
I2,1,3
"""
    midx = MultiIndex.from_tuples([("A", "B"), ("A", "B.1")], names=["I11", "I12"])
    idx = Index(["I2"])
    expected = DataFrame([[1, 3]], index=idx, columns=midx)

    result = parser.read_csv(StringIO(data), index_col=0, header=[0, 1])
    tm.assert_frame_equal(result, expected)

    col_idx = Index(["A", "A.1"])
    idx = Index(["I12", "I2"], name="I11")
    expected = DataFrame([["B", "B"], ["1", "3"]], index=idx, columns=col_idx)

    result = parser.read_csv(StringIO(data), index_col="I11", header=0)
    tm.assert_frame_equal(result, expected)


@pytest.mark.slow
def test_index_col_large_csv(temp_file, all_parsers, monkeypatch):
    # [FUNCTION DOC]
    # Marked slow. Patches the minimum array length for a faster hashing path in the
    # algorithms module to a small value, writes 101 rows, and verifies reading with the
    # first column as index equals setting the index on the original frame (regression
    # for an issue on large inputs).
    # https://github.com/pandas-dev/pandas/issues/37094
    parser = all_parsers

    ARR_LEN = 100
    df = DataFrame(
        {
            "a": range(ARR_LEN + 1),
            "b": np.random.default_rng(2).standard_normal(ARR_LEN + 1),
        }
    )

    df.to_csv(temp_file, index=False)
    with monkeypatch.context() as m:
        m.setattr("pandas.core.algorithms._MINIMUM_COMP_ARR_LEN", ARR_LEN)
        result = parser.read_csv(temp_file, index_col=[0])

    tm.assert_frame_equal(result, df.set_index("a"))


@xfail_pyarrow  # TypeError: an integer is required
def test_index_col_multiindex_columns_no_data(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies two header rows and no data with the
    # first column as index gives an empty frame with an empty plain index and two-level
    # columns named after the first column's header cells.
    # GH#38292
    parser = all_parsers
    result = parser.read_csv(
        StringIO("a0,a1,a2\nb0,b1,b2\n"), header=[0, 1], index_col=0
    )
    expected = DataFrame(
        [],
        index=Index([]),
        columns=MultiIndex.from_arrays(
            [["a1", "a2"], ["b1", "b2"]], names=["a0", "b0"]
        ),
    )
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
def test_index_col_header_no_data(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies a one-element header list and index
    # column with no data rows gives an empty frame with an empty index named after the
    # first column.
    # GH#38292
    parser = all_parsers
    result = parser.read_csv(StringIO("a0,a1,a2\n"), header=[0], index_col=0)
    expected = DataFrame(
        [],
        columns=["a1", "a2"],
        index=Index([], name="a0"),
    )
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
def test_multiindex_columns_no_data(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies a two-row header and no data gives
    # an empty frame with three two-level columns.
    # GH#38292
    parser = all_parsers
    result = parser.read_csv(StringIO("a0,a1,a2\nb0,b1,b2\n"), header=[0, 1])
    expected = DataFrame(
        [], columns=MultiIndex.from_arrays([["a0", "a1", "a2"], ["b0", "b1", "b2"]])
    )
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
def test_multiindex_columns_index_col_with_data(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies a two-row header with one data row
    # and the first column as index gives a one-row frame, two-level columns with names
    # from the index column's header cells, and index of the first data value.
    # GH#38292
    parser = all_parsers
    result = parser.read_csv(
        StringIO("a0,a1,a2\nb0,b1,b2\ndata,data,data"), header=[0, 1], index_col=0
    )
    expected = DataFrame(
        [["data", "data"]],
        columns=MultiIndex.from_arrays(
            [["a1", "a2"], ["b1", "b2"]], names=["a0", "b0"]
        ),
        index=Index(["data"]),
    )
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Empty CSV file or block
def test_infer_types_boolean_sum(all_parsers):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Verifies an index column read with nullable unsigned 8-bit
    # dtype works (regression for boolean summation); the index type is not compared
    # because the C engine returns an object index while the python engine returns
    # int64.
    # GH#44079
    parser = all_parsers
    result = parser.read_csv(
        StringIO("0,1"),
        names=["a", "b"],
        index_col=["a"],
        dtype={"a": "UInt8"},
    )
    expected = DataFrame(
        data={
            "a": [
                0,
            ],
            "b": [1],
        }
    ).set_index("a")
    # Not checking index type now, because the C parser will return a
    # index column of dtype 'object', and the Python parser will return a
    # index column of dtype 'int64'.
    tm.assert_frame_equal(result, expected, check_index_type=False)


@pytest.mark.parametrize("dtype, val", [(object, "01"), ("int64", 1)])
def test_specify_dtype_for_index_col(all_parsers, dtype, val, request):
    # [FUNCTION DOC]
    # Verifies a dtype given for the index column is applied: object keeps the text with
    # leading zero, integer converts to one. For pyarrow the object case is marked
    # expected to fail (type inference cannot be disabled).
    # GH#9435
    data = "a,b\n01,2"
    parser = all_parsers
    if dtype == object and parser.engine == "pyarrow":
        request.applymarker(
            pytest.mark.xfail(reason="Cannot disable type-inference for pyarrow engine")
        )
    result = parser.read_csv(StringIO(data), index_col="a", dtype={"a": dtype})
    expected = DataFrame({"b": [2]}, index=Index([val], name="a", dtype=dtype))
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
def test_multiindex_columns_not_leading_index_col(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies a two-row header with an index
    # column that is not the first column (position one) gives the remaining columns
    # with two-level names taken from the index column's header cells.
    # GH#38549
    parser = all_parsers
    data = """a,b,c,d
e,f,g,h
x,y,1,2
"""
    result = parser.read_csv(
        StringIO(data),
        header=[0, 1],
        index_col=1,
    )
    cols = MultiIndex.from_tuples(
        [("a", "e"), ("c", "g"), ("d", "h")], names=["b", "f"]
    )
    expected = DataFrame([["x", 1, 2]], columns=cols, index=["y"])
    tm.assert_frame_equal(result, expected)
