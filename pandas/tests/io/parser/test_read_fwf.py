"""
Tests the 'read_fwf' function in parsers.py. This
test suite is independent of the others because the
engine is set to 'python-fwf' internally.
"""
# [MODULE DOC]
# Tests for read_fwf (fixed-width formatted reader), which always uses the python-fwf engine; independent of the all_parsers fixture. Covers column specs and widths, inference of column boundaries, delimiter/filler characters, comments, thousands, blank lines, skiprows/nrows/iteration, encodings and compression, usecols/names/index_col interplay, nullable dtype backends and URL input.
# Many tests compare inferred-width reading with explicit column specs on the same text, or against read_csv with a whitespace regex separator.

from io import (
    BytesIO,
    StringIO,
)

import numpy as np
import pytest

from pandas.errors import EmptyDataError

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
)
import pandas._testing as tm

from pandas.io.common import urlopen
from pandas.io.parsers import (
    read_csv,
    read_fwf,
)


def test_basic():
    # [FUNCTION DOC]
    # Reads whitespace-aligned text with an inferred layout; asserts an integer column and three float columns named from the header.
    data = """\
A         B            C            D
201158    360.242940   149.910199   11950.7
201159    444.953632   166.985655   11788.4
201160    364.136849   183.628767   11806.2
201161    413.836124   184.375703   11916.8
201162    502.953953   173.237159   12468.3
"""
    result = read_fwf(StringIO(data))
    expected = DataFrame(
        [
            [201158, 360.242940, 149.910199, 11950.7],
            [201159, 444.953632, 166.985655, 11788.4],
            [201160, 364.136849, 183.628767, 11806.2],
            [201161, 413.836124, 184.375703, 11916.8],
            [201162, 502.953953, 173.237159, 12468.3],
        ],
        columns=["A", "B", "C", "D"],
    )
    tm.assert_frame_equal(result, expected)


def test_colspecs():
    # [FUNCTION DOC]
    # Explicit column specs (start, end pairs) split the first column into two parts and the rest as given; asserts five columns with the first two numeric pieces of the first field.
    data = """\
A   B     C            D            E
201158    360.242940   149.910199   11950.7
201159    444.953632   166.985655   11788.4
201160    364.136849   183.628767   11806.2
201161    413.836124   184.375703   11916.8
201162    502.953953   173.237159   12468.3
"""
    colspecs = [(0, 4), (4, 8), (8, 20), (21, 33), (34, 43)]
    result = read_fwf(StringIO(data), colspecs=colspecs)

    expected = DataFrame(
        [
            [2011, 58, 360.242940, 149.910199, 11950.7],
            [2011, 59, 444.953632, 166.985655, 11788.4],
            [2011, 60, 364.136849, 183.628767, 11806.2],
            [2011, 61, 413.836124, 184.375703, 11916.8],
            [2011, 62, 502.953953, 173.237159, 12468.3],
        ],
        columns=["A", "B", "C", "D", "E"],
    )
    tm.assert_frame_equal(result, expected)


def test_widths():
    # [FUNCTION DOC]
    # Explicit widths list produces the same five-column frame as the colspecs case on differently aligned text.
    data = """\
A    B    C            D            E
2011 58   360.242940   149.910199   11950.7
2011 59   444.953632   166.985655   11788.4
2011 60   364.136849   183.628767   11806.2
2011 61   413.836124   184.375703   11916.8
2011 62   502.953953   173.237159   12468.3
"""
    result = read_fwf(StringIO(data), widths=[5, 5, 13, 13, 7])

    expected = DataFrame(
        [
            [2011, 58, 360.242940, 149.910199, 11950.7],
            [2011, 59, 444.953632, 166.985655, 11788.4],
            [2011, 60, 364.136849, 183.628767, 11806.2],
            [2011, 61, 413.836124, 184.375703, 11916.8],
            [2011, 62, 502.953953, 173.237159, 12468.3],
        ],
        columns=["A", "B", "C", "D", "E"],
    )
    tm.assert_frame_equal(result, expected)


def test_non_space_filler():
    # [FUNCTION DOC]
    # Data padded with tilde characters, read with colspecs and the tilde as delimiter (filler); asserts the same five-column frame (sourced from a user report on non-space fillers).
    # From Thomas Kluyver:
    #
    # Apparently, some non-space filler characters can be seen, this is
    # supported by specifying the 'delimiter' character:
    #
    # http://publib.boulder.ibm.com/infocenter/dmndhelp/v6r1mx/index.jsp?topic=/com.ibm.wbit.612.help.config.doc/topics/rfixwidth.html
    data = """\
A~~~~B~~~~C~~~~~~~~~~~~D~~~~~~~~~~~~E
201158~~~~360.242940~~~149.910199~~~11950.7
201159~~~~444.953632~~~166.985655~~~11788.4
201160~~~~364.136849~~~183.628767~~~11806.2
201161~~~~413.836124~~~184.375703~~~11916.8
201162~~~~502.953953~~~173.237159~~~12468.3
"""
    colspecs = [(0, 4), (4, 8), (8, 20), (21, 33), (34, 43)]
    result = read_fwf(StringIO(data), colspecs=colspecs, delimiter="~")

    expected = DataFrame(
        [
            [2011, 58, 360.242940, 149.910199, 11950.7],
            [2011, 59, 444.953632, 166.985655, 11788.4],
            [2011, 60, 364.136849, 183.628767, 11806.2],
            [2011, 61, 413.836124, 184.375703, 11916.8],
            [2011, 62, 502.953953, 173.237159, 12468.3],
        ],
        columns=["A", "B", "C", "D", "E"],
    )
    tm.assert_frame_equal(result, expected)


def test_over_specified():
    # [FUNCTION DOC]
    # Passing both colspecs and widths raises a value error saying only one may be specified.
    data = """\
A   B     C            D            E
201158    360.242940   149.910199   11950.7
201159    444.953632   166.985655   11788.4
201160    364.136849   183.628767   11806.2
201161    413.836124   184.375703   11916.8
201162    502.953953   173.237159   12468.3
"""
    colspecs = [(0, 4), (4, 8), (8, 20), (21, 33), (34, 43)]

    with pytest.raises(ValueError, match="must specify only one of"):
        read_fwf(StringIO(data), colspecs=colspecs, widths=[6, 10, 10, 7])


def test_under_specified():
    # [FUNCTION DOC]
    # Passing neither colspecs nor widths explicitly (both none) raises a value error saying one of them must be specified. (Note that the normal default of inferring colspecs is a separate sentinel, so explicit none is invalid.)
    data = """\
A   B     C            D            E
201158    360.242940   149.910199   11950.7
201159    444.953632   166.985655   11788.4
201160    364.136849   183.628767   11806.2
201161    413.836124   184.375703   11916.8
201162    502.953953   173.237159   12468.3
"""
    with pytest.raises(ValueError, match="Must specify either"):
        read_fwf(StringIO(data), colspecs=None, widths=None)


def test_read_csv_compat():
    # [FUNCTION DOC]
    # Asserts that reading fixed-width text with colspecs gives the same frame as reading the equivalent comma-separated text with read_csv (python engine).
    csv_data = """\
A,B,C,D,E
2011,58,360.242940,149.910199,11950.7
2011,59,444.953632,166.985655,11788.4
2011,60,364.136849,183.628767,11806.2
2011,61,413.836124,184.375703,11916.8
2011,62,502.953953,173.237159,12468.3
"""
    expected = read_csv(StringIO(csv_data), engine="python")

    fwf_data = """\
A   B     C            D            E
201158    360.242940   149.910199   11950.7
201159    444.953632   166.985655   11788.4
201160    364.136849   183.628767   11806.2
201161    413.836124   184.375703   11916.8
201162    502.953953   173.237159   12468.3
"""
    colspecs = [(0, 4), (4, 8), (8, 20), (21, 33), (34, 43)]
    result = read_fwf(StringIO(fwf_data), colspecs=colspecs)
    tm.assert_frame_equal(result, expected)


def test_bytes_io_input():
    # [FUNCTION DOC]
    # Reads a binary buffer containing Hebrew text with width two columns and utf8 encoding; asserts the header names and the single row are the two-character pieces.
    data = BytesIO("שלום\nשלום".encode())  # noqa: RUF001
    result = read_fwf(data, widths=[2, 2], encoding="utf8")
    expected = DataFrame([["של", "ום"]], columns=["של", "ום"])
    tm.assert_frame_equal(result, expected)


def test_fwf_colspecs_is_list_or_tuple():
    # [FUNCTION DOC]
    # Passing a mapping as colspecs raises a type error saying column specifications must be a list or tuple.
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""

    msg = "column specifications must be a list or tuple.+"

    with pytest.raises(TypeError, match=msg):
        read_fwf(StringIO(data), colspecs={"a": 1}, delimiter=",")


def test_fwf_colspecs_is_list_or_tuple_of_two_element_tuples():
    # [FUNCTION DOC]
    # A colspecs list whose element is not a pair of integers raises a type error saying each column specification must be of the right form.
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""

    msg = "Each column specification must be.+"

    with pytest.raises(TypeError, match=msg):
        read_fwf(StringIO(data), colspecs=[("a", 1)])


@pytest.mark.parametrize(
    "colspecs,exp_data",
    [
        ([(0, 3), (3, None)], [[123, 456], [456, 789]]),
        ([(None, 3), (3, 6)], [[123, 456], [456, 789]]),
        ([(0, None), (3, None)], [[123456, 456], [456789, 789]]),
        ([(None, None), (3, 6)], [[123456, 456], [456789, 789]]),
    ],
)
def test_fwf_colspecs_none(colspecs, exp_data):
    # [FUNCTION DOC]
    # See gh-7079. Parametrized over colspecs using open-ended (none) start or end positions; asserts that open ends extend to the start/end of the line and the resulting integer columns, with no header.
    # see gh-7079
    data = """\
123456
456789
"""
    expected = DataFrame(exp_data)

    result = read_fwf(StringIO(data), colspecs=colspecs, header=None)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "infer_nrows,exp_data",
    [
        # infer_nrows --> colspec == [(2, 3), (5, 6)]
        (1, [[1, 2], [3, 8]]),
        # infer_nrows > number of rows
        (10, [[1, 2], [123, 98]]),
    ],
)
def test_fwf_colspecs_infer_nrows(infer_nrows, exp_data):
    # [FUNCTION DOC]
    # See gh-15138. Parametrized over infer_nrows one (inference uses only the first line, giving narrower specs and truncated values) and ten (greater than the row count, so the whole file is used); asserts the corresponding two-column frames.
    # see gh-15138
    data = """\
  1  2
123 98
"""
    expected = DataFrame(exp_data)

    result = read_fwf(StringIO(data), infer_nrows=infer_nrows, header=None)
    tm.assert_frame_equal(result, expected)


def test_fwf_regression():
    # [FUNCTION DOC]
    # See gh-3594. Fixed-width timestamp-like first column plus seven float columns, with explicit widths, names, first column as index, date parsing and a day-of-year based date format; asserts a microsecond datetime index and float columns. The comment upstream notes a name such as T060 can be mistaken for a datetime slice.
    # see gh-3594
    #
    # Turns out "T060" is parsable as a datetime slice!
    tz_list = [1, 10, 20, 30, 60, 80, 100]
    widths = [16] + [8] * len(tz_list)
    names = ["SST"] + [f"T{z:03d}" for z in tz_list[1:]]

    data = """  2009164202000   9.5403  9.4105  8.6571  7.8372  6.0612  5.8843  5.5192
2009164203000   9.5435  9.2010  8.6167  7.8176  6.0804  5.8728  5.4869
2009164204000   9.5873  9.1326  8.4694  7.5889  6.0422  5.8526  5.4657
2009164205000   9.5810  9.0896  8.4009  7.4652  6.0322  5.8189  5.4379
2009164210000   9.6034  9.0897  8.3822  7.4905  6.0908  5.7904  5.4039
"""
    expected = DataFrame(
        [
            [9.5403, 9.4105, 8.6571, 7.8372, 6.0612, 5.8843, 5.5192],
            [9.5435, 9.2010, 8.6167, 7.8176, 6.0804, 5.8728, 5.4869],
            [9.5873, 9.1326, 8.4694, 7.5889, 6.0422, 5.8526, 5.4657],
            [9.5810, 9.0896, 8.4009, 7.4652, 6.0322, 5.8189, 5.4379],
            [9.6034, 9.0897, 8.3822, 7.4905, 6.0908, 5.7904, 5.4039],
        ],
        index=DatetimeIndex(
            [
                "2009-06-13 20:20:00",
                "2009-06-13 20:30:00",
                "2009-06-13 20:40:00",
                "2009-06-13 20:50:00",
                "2009-06-13 21:00:00",
            ],
            dtype="M8[us]",
        ),
        columns=["SST", "T010", "T020", "T030", "T060", "T080", "T100"],
    )
    result = read_fwf(
        StringIO(data),
        index_col=0,
        header=None,
        names=names,
        widths=widths,
        parse_dates=True,
        date_format="%Y%j%H%M%S",
    )
    tm.assert_frame_equal(result, expected)


def test_fwf_for_uint8():
    # [FUNCTION DOC]
    # Parses car-bus style log lines with colspecs, names and converters that convert hexadecimal fields to integers and count space-separated tokens; asserts the frame with a missing destination in the second row kept as an object column.
    # KNOWN ISSUE: the name refers to unsigned 8-bit integers but no such dtype is specified or asserted; the test checks converters and a missing field instead.
    data = """1421302965.213420    PRI=3 PGN=0xef00      DST=0x17 SRC=0x28    04 154 00 00 00 00 00 127
1421302964.226776    PRI=6 PGN=0xf002               SRC=0x47    243 00 00 255 247 00 00 71"""  # noqa: E501
    df = read_fwf(
        StringIO(data),
        colspecs=[(0, 17), (25, 26), (33, 37), (49, 51), (58, 62), (63, 1000)],
        names=["time", "pri", "pgn", "dst", "src", "data"],
        converters={
            "pgn": lambda x: int(x, 16),
            "src": lambda x: int(x, 16),
            "dst": lambda x: int(x, 16),
            "data": lambda x: len(x.split(" ")),
        },
    )

    expected = DataFrame(
        [
            [1421302965.213420, 3, 61184, 23, 40, 8],
            [1421302964.226776, 6, 61442, None, 71, 8],
        ],
        columns=["time", "pri", "pgn", "dst", "src", "data"],
    )
    expected["dst"] = expected["dst"].astype(object)
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize("comment", ["#", "~", "!"])
def test_fwf_comment(comment):
    # [FUNCTION DOC]
    # Parametrized over comment characters. Replaces the comment marker in the data, reads with colspecs, no header and the comment option; asserts comment text is ignored giving a frame with a missing value.
    data = """\
  1   2.   4  #hello world
  5  NaN  10.0
"""
    data = data.replace("#", comment)

    colspecs = [(0, 3), (4, 9), (9, 25)]
    expected = DataFrame([[1, 2.0, 4], [5, np.nan, 10.0]])

    result = read_fwf(StringIO(data), colspecs=colspecs, header=None, comment=comment)
    tm.assert_almost_equal(result, expected)


def test_fwf_skip_blank_lines():
    # [FUNCTION DOC]
    # With skipping of blank lines enabled, blank lines before the header and between rows are dropped; asserts three rows. With skipping disabled, blank lines in the body become rows of missing values; asserts five rows.
    data = """

A         B            C            D

201158    360.242940   149.910199   11950.7
201159    444.953632   166.985655   11788.4


201162    502.953953   173.237159   12468.3

"""
    result = read_fwf(StringIO(data), skip_blank_lines=True)
    expected = DataFrame(
        [
            [201158, 360.242940, 149.910199, 11950.7],
            [201159, 444.953632, 166.985655, 11788.4],
            [201162, 502.953953, 173.237159, 12468.3],
        ],
        columns=["A", "B", "C", "D"],
    )
    tm.assert_frame_equal(result, expected)

    data = """\
A         B            C            D
201158    360.242940   149.910199   11950.7
201159    444.953632   166.985655   11788.4


201162    502.953953   173.237159   12468.3
"""
    result = read_fwf(StringIO(data), skip_blank_lines=False)
    expected = DataFrame(
        [
            [201158, 360.242940, 149.910199, 11950.7],
            [201159, 444.953632, 166.985655, 11788.4],
            [np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
            [201162, 502.953953, 173.237159, 12468.3],
        ],
        columns=["A", "B", "C", "D"],
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("thousands", [",", "#", "~"])
def test_fwf_thousands(thousands):
    # [FUNCTION DOC]
    # Parametrized over thousands separators. Replaces the comma in the data by the separator, reads with explicit colspecs, no header and the thousands option; asserts the grouped number is parsed as a float 2334.0.
    data = """\
 1 2,334.0    5
10   13     10.
"""
    data = data.replace(",", thousands)

    colspecs = [(0, 3), (3, 11), (12, 16)]
    expected = DataFrame([[1, 2334.0, 5], [10, 13, 10.0]])

    result = read_fwf(
        StringIO(data), header=None, colspecs=colspecs, thousands=thousands
    )
    tm.assert_almost_equal(result, expected)


@pytest.mark.parametrize("header", [True, False])
def test_bool_header_arg(header):
    # [FUNCTION DOC]
    # See gh-6114. Parametrized over true and false: passing a boolean for header raises a type error saying that passing a bool to header is invalid.
    # see gh-6114
    data = """\
MyColumn
   a
   b
   a
   b"""

    msg = "Passing a bool to header is invalid"
    with pytest.raises(TypeError, match=msg):
        read_fwf(StringIO(data), header=header)


def test_full_file():
    # [FUNCTION DOC]
    # Fixed-width file with all values present: asserts inferred column boundaries give the same frame as the explicit colspecs.
    # File with all values.
    test = """index                             A    B    C
2000-01-03T00:00:00  0.980268513777    3  foo
2000-01-04T00:00:00  1.04791624281    -4  bar
2000-01-05T00:00:00  0.498580885705   73  baz
2000-01-06T00:00:00  1.12020151869     1  foo
2000-01-07T00:00:00  0.487094399463    0  bar
2000-01-10T00:00:00  0.836648671666    2  baz
2000-01-11T00:00:00  0.157160753327   34  foo"""
    colspecs = ((0, 19), (21, 35), (38, 40), (42, 45))
    expected = read_fwf(StringIO(test), colspecs=colspecs)

    result = read_fwf(StringIO(test))
    tm.assert_frame_equal(result, expected)


def test_full_file_with_missing():
    # [FUNCTION DOC]
    # File with some missing values in the fields: asserts inference matches explicit colspecs.
    # File with missing values.
    test = """index                             A    B    C
2000-01-03T00:00:00  0.980268513777    3  foo
2000-01-04T00:00:00  1.04791624281    -4  bar
                     0.498580885705   73  baz
2000-01-06T00:00:00  1.12020151869     1  foo
2000-01-07T00:00:00                    0  bar
2000-01-10T00:00:00  0.836648671666    2  baz
                                      34"""
    colspecs = ((0, 19), (21, 35), (38, 40), (42, 45))
    expected = read_fwf(StringIO(test), colspecs=colspecs)

    result = read_fwf(StringIO(test))
    tm.assert_frame_equal(result, expected)


def test_full_file_with_spaces():
    # [FUNCTION DOC]
    # File whose values contain embedded spaces (names): asserts inference matches explicit colspecs.
    # File with spaces in columns.
    test = """
Account                 Name  Balance     CreditLimit   AccountCreated
101     Keanu Reeves          9315.45     10000.00           1/17/1998
312     Gerard Butler         90.00       1000.00             8/6/2003
868     Jennifer Love Hewitt  0           17000.00           5/25/1985
761     Jada Pinkett-Smith    49654.87    100000.00          12/5/2006
317     Bill Murray           789.65      5000.00             2/5/2007
""".strip("\r\n")
    colspecs = ((0, 7), (8, 28), (30, 38), (42, 53), (56, 70))
    expected = read_fwf(StringIO(test), colspecs=colspecs)

    result = read_fwf(StringIO(test))
    tm.assert_frame_equal(result, expected)


def test_full_file_with_spaces_and_missing():
    # [FUNCTION DOC]
    # File with both embedded spaces and missing values: asserts inference matches explicit colspecs.
    # File with spaces and missing values in columns.
    test = """
Account               Name    Balance     CreditLimit   AccountCreated
101                           10000.00                       1/17/1998
312     Gerard Butler         90.00       1000.00             8/6/2003
868                                                          5/25/1985
761     Jada Pinkett-Smith    49654.87    100000.00          12/5/2006
317     Bill Murray           789.65
""".strip("\r\n")
    colspecs = ((0, 7), (8, 28), (30, 38), (42, 53), (56, 70))
    expected = read_fwf(StringIO(test), colspecs=colspecs)

    result = read_fwf(StringIO(test))
    tm.assert_frame_equal(result, expected)


def test_messed_up_data():
    # [FUNCTION DOC]
    # Irregularly indented file with a blank line and uneven columns: asserts inference matches the explicit colspecs.
    # Completely messed up file.
    test = """
   Account          Name             Balance     Credit Limit   Account Created
       101                           10000.00                       1/17/1998
       312     Gerard Butler         90.00       1000.00

       761     Jada Pinkett-Smith    49654.87    100000.00          12/5/2006
  317          Bill Murray           789.65
""".strip("\r\n")
    colspecs = ((2, 10), (15, 33), (37, 45), (49, 61), (64, 79))
    expected = read_fwf(StringIO(test), colspecs=colspecs)

    result = read_fwf(StringIO(test))
    tm.assert_frame_equal(result, expected)


def test_multiple_delimiters():
    # [FUNCTION DOC]
    # Text padded with a mix of several filler characters (space, plus, tilde, period, backslash) given together as the delimiter string; asserts that inference with the delimiter matches explicit colspecs with the same delimiter.
    test = r"""
col1~~~~~col2  col3++++++++++++++++++col4
~~22.....11.0+++foo~~~~~~~~~~Keanu Reeves
  33+++122.33\\\bar.........Gerard Butler
++44~~~~12.01   baz~~Jennifer Love Hewitt
~~55       11+++foo++++Jada Pinkett-Smith
..66++++++.03~~~bar           Bill Murray
""".strip("\r\n")
    delimiter = " +~.\\"
    colspecs = ((0, 4), (7, 13), (15, 19), (21, 41))
    expected = read_fwf(StringIO(test), colspecs=colspecs, delimiter=delimiter)

    result = read_fwf(StringIO(test), delimiter=delimiter)
    tm.assert_frame_equal(result, expected)


def test_variable_width_unicode():
    # [FUNCTION DOC]
    # Hebrew text with variable-width characters read with no header, from utf8 bytes: asserts the inferred layout matches the explicit colspecs.
    data = """
שלום שלום
ום   שלל
של   ום
""".strip("\r\n")
    encoding = "utf8"
    kwargs = {"header": None, "encoding": encoding}

    expected = read_fwf(
        BytesIO(data.encode(encoding)), colspecs=[(0, 4), (5, 9)], **kwargs
    )
    result = read_fwf(BytesIO(data.encode(encoding)), **kwargs)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", [{}, {"a": "float64", "b": str, "c": "int32"}])
def test_dtype(dtype):
    # [FUNCTION DOC]
    # Parametrized over no dtype and a mapping of floating, string and 32-bit integer dtypes. Reading with explicit colspecs; builds the expected frame from numbers and casts the mapped columns, and asserts equality.
    data = """ a    b    c
1    2    3.2
3    4    5.2
"""
    colspecs = [(0, 5), (5, 10), (10, None)]
    result = read_fwf(StringIO(data), colspecs=colspecs, dtype=dtype)

    expected = DataFrame(
        {"a": [1, 3], "b": [2, 4], "c": [3.2, 5.2]}, columns=["a", "b", "c"]
    )

    for col, dt in dtype.items():
        expected[col] = expected[col].astype(dt)

    tm.assert_frame_equal(result, expected)


def test_skiprows_inference():
    # [FUNCTION DOC]
    # See gh-11256. Skipping two header lines before inferring widths gives the same frame as read_csv with the same skiprows and a whitespace separator.
    # see gh-11256
    data = """
Text contained in the file header

DataCol1   DataCol2
     0.0        1.0
   101.6      956.1
""".strip()
    skiprows = 2

    expected = read_csv(StringIO(data), skiprows=skiprows, sep=r"\s+")
    result = read_fwf(StringIO(data), skiprows=skiprows)
    tm.assert_frame_equal(result, expected)


def test_skiprows_by_index_inference():
    # [FUNCTION DOC]
    # skiprows as a list of line indexes (zero and two) before inferring widths gives the same frame as read_csv with the same skiprows and whitespace separator.
    data = """
To be skipped
Not  To  Be  Skipped
Once more to be skipped
123  34   8      123
456  78   9      456
""".strip()
    skiprows = [0, 2]

    expected = read_csv(StringIO(data), skiprows=skiprows, sep=r"\s+")
    result = read_fwf(StringIO(data), skiprows=skiprows)
    tm.assert_frame_equal(result, expected)


def test_skiprows_inference_empty():
    # [FUNCTION DOC]
    # Skipping all lines raises an empty-data error saying there are no rows from which to infer column width.
    data = """
AA   BBB  C
12   345  6
78   901  2
""".strip()

    msg = "No rows from which to infer column width"
    with pytest.raises(EmptyDataError, match=msg):
        read_fwf(StringIO(data), skiprows=3)


def test_whitespace_preservation():
    # [FUNCTION DOC]
    # See gh-16772. Fixed-width text with surrounding spaces, explicit widths, tab and newline as delimiters and the first line skipped keeps the spaces in values; asserts equality with the equivalent comma-separated data read by read_csv.
    # see gh-16772
    header = None
    csv_data = """
 a ,bbb
 cc,dd """

    fwf_data = """
 a bbb
 ccdd """
    result = read_fwf(
        StringIO(fwf_data), widths=[3, 3], header=header, skiprows=[0], delimiter="\n\t"
    )
    expected = read_csv(StringIO(csv_data), header=header)
    tm.assert_frame_equal(result, expected)


def test_default_delimiter():
    # [FUNCTION DOC]
    # Fixed-width data with a tab inside a field: using default delimiters (space and tab) with explicit widths and the first line skipped, asserts equality with the equivalent comma-separated data read by read_csv.
    header = None
    csv_data = """
a,bbb
cc,dd"""

    fwf_data = """
a \tbbb
cc\tdd """
    result = read_fwf(StringIO(fwf_data), widths=[3, 3], header=header, skiprows=[0])
    expected = read_csv(StringIO(csv_data), header=header)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("infer", [True, False])
def test_fwf_compression(compression_only, infer, compression_to_extension, temp_file):
    # [FUNCTION DOC]
    # Parametrized over inferring compression from the extension or giving it explicitly, and over each compression format (fixtures). Writes the data compressed to a file in the temp directory (extension chosen from the format), reads with widths and names, and asserts equality with uncompressed reading.
    # KNOWN ISSUE: the check that the infer flag is not none is always true because the parametrization only supplies true or false, so the condition is redundant.
    data = """1111111111
    2222222222
    3333333333""".strip()

    compression = compression_only
    extension = compression_to_extension[compression]

    kwargs = {"widths": [5, 5], "names": ["one", "two"]}
    expected = read_fwf(StringIO(data), **kwargs)

    data = bytes(data, encoding="utf-8")

    path = temp_file.parent / f"tmp.{extension}"
    tm.write_to_compressed(compression, path, data)

    if infer is not None:
        kwargs["compression"] = "infer" if infer else compression

    result = read_fwf(path, **kwargs)
    tm.assert_frame_equal(result, expected)


def test_binary_mode(temp_file):
    # [FUNCTION DOC]
    # GH 18035. Writes a small file and opens it in binary mode; read_fwf on the handle returns a frame with duplicated header names made unique with numeric suffixes. Asserts equality with the reference.
    """
    read_fwf supports opening files in binary mode.

    GH 18035.
    """
    data = """aaa aaa aaa
bba bab b a"""
    df_reference = DataFrame(
        [["bba", "bab", "b a"]], columns=["aaa", "aaa.1", "aaa.2"], index=[0]
    )
    path = temp_file
    path.write_text(data, encoding="utf-8")
    with open(path, "rb") as file:
        df = read_fwf(file)
        file.seek(0)
        tm.assert_frame_equal(df, df_reference)


@pytest.mark.parametrize("memory_map", [True, False])
def test_encoding_mmap(memory_map, temp_file):
    # [FUNCTION DOC]
    # GH 23254. Parametrized over memory mapping on/off: a latin-1 encoded file with a non-ascii character read with the encoding and four widths; asserts the character is decoded properly.
    """
    encoding should be working, even when using a memory-mapped file.

    GH 23254.
    """
    encoding = "iso8859_1"
    path = temp_file
    path.write_bytes(" 1 A Ä 2\n".encode(encoding))
    df = read_fwf(
        path,
        header=None,
        widths=[2, 2, 2, 2],
        encoding=encoding,
        memory_map=memory_map,
    )
    df_reference = DataFrame([[1, "A", "Ä", 2]])
    tm.assert_frame_equal(df, df_reference)


@pytest.mark.parametrize(
    "colspecs, names, widths, index_col",
    [
        (
            [(0, 6), (6, 12), (12, 18), (18, None)],
            list("abcde"),
            None,
            None,
        ),
        (
            None,
            list("abcde"),
            [6] * 4,
            None,
        ),
        (
            [(0, 6), (6, 12), (12, 18), (18, None)],
            list("abcde"),
            None,
            True,
        ),
        (
            None,
            list("abcde"),
            [6] * 4,
            False,
        ),
        (
            None,
            list("abcde"),
            [6] * 4,
            True,
        ),
        (
            [(0, 6), (6, 12), (12, 18), (18, None)],
            list("abcde"),
            None,
            False,
        ),
    ],
)
def test_len_colspecs_len_names(colspecs, names, widths, index_col):
    # [FUNCTION DOC]
    # GH#40830. Parametrized over colspecs or widths with a name list of mismatching length, and index_col true, false or unset; asserts a value error that length of colspecs must match length of names.
    # GH#40830
    data = """col1  col2  col3  col4
    bab   ba    2"""
    msg = "Length of colspecs must match length of names"
    with pytest.raises(ValueError, match=msg):
        read_fwf(
            StringIO(data),
            colspecs=colspecs,
            names=names,
            widths=widths,
            index_col=index_col,
        )


@pytest.mark.parametrize(
    "colspecs, names, widths, index_col, expected",
    [
        (
            [(0, 6), (6, 12), (12, 18), (18, None)],
            list("abc"),
            None,
            0,
            DataFrame(
                index=["col1", "ba"],
                columns=["a", "b", "c"],
                data=[["col2", "col3", "col4"], ["b   ba", "2", np.nan]],
            ),
        ),
        (
            [(0, 6), (6, 12), (12, 18), (18, None)],
            list("ab"),
            None,
            [0, 1],
            DataFrame(
                index=[["col1", "ba"], ["col2", "b   ba"]],
                columns=["a", "b"],
                data=[["col3", "col4"], ["2", np.nan]],
            ),
        ),
        (
            [(0, 6), (6, 12), (12, 18), (18, None)],
            list("a"),
            None,
            [0, 1, 2],
            DataFrame(
                index=[["col1", "ba"], ["col2", "b   ba"], ["col3", "2"]],
                columns=["a"],
                data=[["col4"], [np.nan]],
            ),
        ),
        (
            None,
            list("abc"),
            [6] * 4,
            0,
            DataFrame(
                index=["col1", "ba"],
                columns=["a", "b", "c"],
                data=[["col2", "col3", "col4"], ["b   ba", "2", np.nan]],
            ),
        ),
        (
            None,
            list("ab"),
            [6] * 4,
            [0, 1],
            DataFrame(
                index=[["col1", "ba"], ["col2", "b   ba"]],
                columns=["a", "b"],
                data=[["col3", "col4"], ["2", np.nan]],
            ),
        ),
        (
            None,
            list("a"),
            [6] * 4,
            [0, 1, 2],
            DataFrame(
                index=[["col1", "ba"], ["col2", "b   ba"], ["col3", "2"]],
                columns=["a"],
                data=[["col4"], [np.nan]],
            ),
        ),
    ],
)
def test_len_colspecs_len_names_with_index_col(
    colspecs, names, widths, index_col, expected
):
    # [FUNCTION DOC]
    # GH#40830. Parametrized over colspecs or widths and index_col as a column or list of columns (names shorter than the number of columns, with the leading columns consumed as the index); asserts the expected frame with a single or multi-level index and missing values filling the short row.
    # GH#40830
    data = """col1  col2  col3  col4
    bab   ba    2"""
    result = read_fwf(
        StringIO(data),
        colspecs=colspecs,
        names=names,
        widths=widths,
        index_col=index_col,
    )
    tm.assert_frame_equal(result, expected)


def test_colspecs_with_comment():
    # [FUNCTION DOC]
    # GH 14135. A comment-only first line with colspecs and no header; asserts a one-row frame with parsed integer and string fields.
    # GH 14135
    result = read_fwf(
        StringIO("#\nA1K\n"), colspecs=[(1, 2), (2, 3)], comment="#", header=None
    )
    expected = DataFrame([[1, "K"]], columns=[0, 1])
    tm.assert_frame_equal(result, expected)


def test_skip_rows_and_n_rows():
    # [FUNCTION DOC]
    # GH#44021. Skipping rows two and four while limiting nrows to four returns the four rows that remain after skipping, with header taken from the first line.
    # GH#44021
    data = """a\tb
1\t a
2\t b
3\t c
4\t d
5\t e
6\t f
    """
    result = read_fwf(StringIO(data), nrows=4, skiprows=[2, 4])
    expected = DataFrame({"a": [1, 3, 5, 6], "b": ["a", "c", "e", "f"]})
    tm.assert_frame_equal(result, expected)


def test_skiprows_with_iterator():
    # [FUNCTION DOC]
    # GH#10261 and GH#56323. Chunked iteration (chunk size two) with a list of skipped rows; asserts three chunks of the remaining values with a continuing row index.
    # GH#10261, GH#56323
    data = """0
1
2
3
4
5
6
7
8
9
    """
    df_iter = read_fwf(
        StringIO(data),
        colspecs=[(0, 2)],
        names=["a"],
        iterator=True,
        chunksize=2,
        skiprows=[0, 1, 2, 6, 9],
    )
    expected_frames = [
        DataFrame({"a": [3, 4]}),
        DataFrame({"a": [5, 7]}, index=[2, 3]),
        DataFrame({"a": [8]}, index=[4]),
    ]
    for i, result in enumerate(df_iter):
        tm.assert_frame_equal(result, expected_frames[i])


def test_names_and_infer_colspecs():
    # [FUNCTION DOC]
    # GH#45337. With inferred colspecs, one skipped row, usecols zero and two and two supplied names; asserts the two selected columns carry the supplied names.
    # GH#45337
    data = """X   Y   Z
      959.0    345   22.2
    """
    result = read_fwf(StringIO(data), skiprows=1, usecols=[0, 2], names=["a", "b"])
    expected = DataFrame({"a": [959.0], "b": 22.2})
    tm.assert_frame_equal(result, expected)


def test_widths_and_usecols():
    # [FUNCTION DOC]
    # GH#46580. Explicit widths of five fields, no header, usecols selecting three of them, names and no index column; asserts the selected columns with their values (constant zero, integers, floats).
    # GH#46580
    data = """0  1    n -0.4100.1
0  2    p  0.2 90.1
0  3    n -0.3140.4"""
    result = read_fwf(
        StringIO(data),
        header=None,
        usecols=(0, 1, 3),
        widths=(3, 5, 1, 5, 5),
        index_col=False,
        names=("c0", "c1", "c3"),
    )
    expected = DataFrame(
        {
            "c0": 0,
            "c1": [1, 2, 3],
            "c3": [-0.4, 0.2, -0.3],
        }
    )
    tm.assert_frame_equal(result, expected)


def test_dtype_backend(string_storage, dtype_backend):
    # [FUNCTION DOC]
    # GH#50289. Parametrized over string storage and dtype backend (fixtures). Reads columns of integer, float, boolean, string and missing-heavy columns and asserts nullable extension dtypes (or pyarrow-backed arrays, with the all-missing column null-typed), ignoring column index type because string storage also affects it.
    # GH#50289
    data = """a  b    c      d  e     f  g    h  i
1  2.5  True  a
3  4.5  False b  True  6  7.5  a"""
    with pd.option_context("mode.string_storage", string_storage):
        result = read_fwf(StringIO(data), dtype_backend=dtype_backend)

    if dtype_backend == "pyarrow":
        pa = pytest.importorskip("pyarrow")
        string_dtype = pd.ArrowDtype(pa.string())
    else:
        string_dtype = pd.StringDtype(string_storage)

    expected = DataFrame(
        {
            "a": pd.Series([1, 3], dtype="Int64"),
            "b": pd.Series([2.5, 4.5], dtype="Float64"),
            "c": pd.Series([True, False], dtype="boolean"),
            "d": pd.Series(["a", "b"], dtype=string_dtype),
            "e": pd.Series([pd.NA, True], dtype="boolean"),
            "f": pd.Series([pd.NA, 6], dtype="Int64"),
            "g": pd.Series([pd.NA, 7.5], dtype="Float64"),
            "h": pd.Series([None, "a"], dtype=string_dtype),
            "i": pd.Series([pd.NA, pd.NA], dtype="Int64"),
        }
    )
    if dtype_backend == "pyarrow":
        pa = pytest.importorskip("pyarrow")
        from pandas.arrays import ArrowExtensionArray

        expected = DataFrame(
            {
                col: ArrowExtensionArray(pa.array(expected[col], from_pandas=True))
                for col in expected.columns
            }
        )
        expected["i"] = ArrowExtensionArray(pa.array([None, None]))

    # the storage of the str columns' Index is also affected by the
    # string_storage setting -> ignore that for checking the result
    tm.assert_frame_equal(result, expected, check_column_type=False)


def test_invalid_dtype_backend():
    # [FUNCTION DOC]
    # dtype_backend of numpy raises a value error that only numpy_nullable and pyarrow are allowed.
    msg = (
        "dtype_backend numpy is invalid, only 'numpy_nullable' and "
        "'pyarrow' are allowed."
    )
    with pytest.raises(ValueError, match=msg):
        read_fwf("test", dtype_backend="numpy")


@pytest.mark.network
@pytest.mark.single_cpu
def test_url_urlopen(httpserver):
    # [FUNCTION DOC]
    # Marked network and single_cpu. Serves the fixed-width text from a local HTTP server fixture, opens the URL with the library's urlopen helper and reads it; asserts the column labels are the four header letters.
    data = """\
A         B            C            D
201158    360.242940   149.910199   11950.7
201159    444.953632   166.985655   11788.4
201160    364.136849   183.628767   11806.2
201161    413.836124   184.375703   11916.8
201162    502.953953   173.237159   12468.3
"""
    httpserver.serve_content(content=data)
    expected = pd.Index(list("ABCD"))
    with urlopen(httpserver.url) as f:
        result = read_fwf(f).columns

    tm.assert_index_equal(result, expected)
