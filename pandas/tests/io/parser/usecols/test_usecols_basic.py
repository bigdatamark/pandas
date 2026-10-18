"""
Tests the usecols functionality during parsing
for all of the parsers defined in parsers.py
"""
# [MODULE DOC]
# Core tests of the usecols option of read_csv across all parsers: validation of invalid forms, selection by name, position, callable, numpy array and set, interaction with names, header, index_col, regex separators and dtype, and error messages for missing columns.
# Module-level constants: expected error message for invalid usecols argument, a format template for mismatched usecols names, the message for pyarrow rejecting integer positions, and xfail/skip markers for pyarrow.
# KNOWN ISSUE: the module-level filter-warnings mark is assigned twice (the second assignment, with a longer message pattern, replaces the first), so only the second applies.

from io import StringIO

import numpy as np
import pytest

from pandas.errors import ParserError

from pandas import (
    DataFrame,
    Index,
    array,
)
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

_msg_validate_usecols_arg = (
    "'usecols' must either be list-like "
    "of all strings, all unicode, all "
    "integers or a callable."
)
_msg_validate_usecols_names = (
    "Usecols do not match columns, columns expected but not found: {0}"
)
_msg_pyarrow_requires_names = (
    "The pyarrow engine does not allow 'usecols' to be integer column "
    "positions. Pass a list of string column names instead."
)

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame is deprecated:DeprecationWarning"
)


def test_raise_on_mixed_dtype_usecols(all_parsers):
    # [FUNCTION DOC]
    # See gh-12678. usecols mixing an integer and a string raises a value error with the invalid-argument message.
    # See gh-12678
    data = """a,b,c
        1000,2000,3000
        4000,5000,6000
        """
    usecols = [0, "b", 2]
    parser = all_parsers

    with pytest.raises(ValueError, match=_msg_validate_usecols_arg):
        parser.read_csv(StringIO(data), usecols=usecols)


@pytest.mark.parametrize("usecols", [(1, 2), ("b", "c")])
def test_usecols(all_parsers, usecols, request):
    # [FUNCTION DOC]
    # Parametrized over positional (one, two) and name-based usecols. Asserts a frame containing only the two selected columns. For pyarrow with integer positions, instead asserts a value error that integer positions are not allowed, then returns.
    data = """\
a,b,c
1,2,3
4,5,6
7,8,9
10,11,12"""
    parser = all_parsers
    if parser.engine == "pyarrow" and isinstance(usecols[0], int):
        with pytest.raises(ValueError, match=_msg_pyarrow_requires_names):
            parser.read_csv(StringIO(data), usecols=usecols)
        return

    result = parser.read_csv(StringIO(data), usecols=usecols)

    expected = DataFrame([[2, 3], [5, 6], [8, 9], [11, 12]], columns=["b", "c"])
    tm.assert_frame_equal(result, expected)


def test_usecols_with_names(all_parsers):
    # [FUNCTION DOC]
    # Header row 0 is replaced by two supplied names while usecols picks positions one and two; asserts the resulting columns carry the supplied names. For pyarrow asserts the integer-position value error instead.
    data = """\
a,b,c
1,2,3
4,5,6
7,8,9
10,11,12"""
    parser = all_parsers
    names = ["foo", "bar"]

    if parser.engine == "pyarrow":
        with pytest.raises(ValueError, match=_msg_pyarrow_requires_names):
            parser.read_csv(StringIO(data), names=names, usecols=[1, 2], header=0)
        return

    result = parser.read_csv(StringIO(data), names=names, usecols=[1, 2], header=0)

    expected = DataFrame([[2, 3], [5, 6], [8, 9], [11, 12]], columns=names)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "names,usecols", [(["b", "c"], [1, 2]), (["a", "b", "c"], ["b", "c"])]
)
def test_usecols_relative_to_names(all_parsers, names, usecols):
    # [FUNCTION DOC]
    # Parametrized over (names, usecols) pairs: two names with positional usecols, and three names with name-based usecols. With no header, asserts the selected columns b and c. Name-based selection is skipped on pyarrow (upstream key error).
    data = """\
1,2,3
4,5,6
7,8,9
10,11,12"""
    parser = all_parsers
    if parser.engine == "pyarrow" and not isinstance(usecols[0], int):
        # ArrowKeyError: Column 'fb' in include_columns does not exist
        pytest.skip(reason="https://github.com/apache/arrow/issues/38676")

    result = parser.read_csv(StringIO(data), names=names, header=None, usecols=usecols)

    expected = DataFrame([[2, 3], [5, 6], [8, 9], [11, 12]], columns=["b", "c"])
    tm.assert_frame_equal(result, expected)


def test_usecols_relative_to_names2(all_parsers):
    # [FUNCTION DOC]
    # See gh-5766. Two supplied names, no header, usecols positions zero and one; asserts first two columns returned with those names.
    # see gh-5766
    data = """\
1,2,3
4,5,6
7,8,9
10,11,12"""
    parser = all_parsers

    result = parser.read_csv(
        StringIO(data), names=["a", "b"], header=None, usecols=[0, 1]
    )

    expected = DataFrame([[1, 2], [4, 5], [7, 8], [10, 11]], columns=["a", "b"])
    tm.assert_frame_equal(result, expected)


# regex mismatch: "Length mismatch: Expected axis has 1 elements"
@xfail_pyarrow
def test_usecols_name_length_conflict(all_parsers):
    # [FUNCTION DOC]
    # Supplying two names with a single usecols entry on three-column data raises a value error that the number of passed names did not match the number of header fields. Expected to fail on pyarrow (message mismatch).
    data = """\
1,2,3
4,5,6
7,8,9
10,11,12"""
    parser = all_parsers
    msg = "Number of passed names did not match number of header fields in the file"
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), names=["a", "b"], header=None, usecols=[1])


def test_usecols_single_string(all_parsers):
    # [FUNCTION DOC]
    # See gh-20558. A plain string for usecols (not a list) raises a value error with the invalid-argument message.
    # see gh-20558
    parser = all_parsers
    data = """foo, bar, baz
1000, 2000, 3000
4000, 5000, 6000"""

    with pytest.raises(ValueError, match=_msg_validate_usecols_arg):
        parser.read_csv(StringIO(data), usecols="foo")


# ArrowKeyError: Column 'a' in include_columns does not exist in CSV file
@skip_pyarrow
@pytest.mark.parametrize(
    "data", ["a,b,c,d\n1,2,3,4\n5,6,7,8", "a,b,c,d\n1,2,3,4,\n5,6,7,8,"]
)
def test_usecols_index_col_false(all_parsers, data):
    # [FUNCTION DOC]
    # See gh-9082. Parametrized over data with and without trailing delimiters. With index_col disabled and three named columns selected, asserts those columns are returned with a default integer index. Skipped on pyarrow.
    # see gh-9082
    parser = all_parsers
    usecols = ["a", "c", "d"]
    expected = DataFrame({"a": [1, 5], "c": [3, 7], "d": [4, 8]})

    result = parser.read_csv(StringIO(data), usecols=usecols, index_col=False)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("index_col", ["b", 0])
@pytest.mark.parametrize("usecols", [["b", "c"], [1, 2]])
def test_usecols_index_col_conflict(all_parsers, usecols, index_col, request):
    # [FUNCTION DOC]
    # See gh-4201. Parametrized over name/position usecols and index_col given by name or position. Asserts the column b becomes the index and only column c remains, i.e. integer index_col is interpreted relative to usecols. For pyarrow with integer positions asserts the position-related value error.
    # see gh-4201: test that index_col as integer reflects usecols
    parser = all_parsers
    data = "a,b,c,d\nA,a,1,one\nB,b,2,two"

    if parser.engine == "pyarrow" and isinstance(usecols[0], int):
        with pytest.raises(ValueError, match=_msg_pyarrow_requires_names):
            parser.read_csv(StringIO(data), usecols=usecols, index_col=index_col)
        return

    expected = DataFrame({"c": [1, 2]}, index=Index(["a", "b"], name="b"))

    result = parser.read_csv(StringIO(data), usecols=usecols, index_col=index_col)
    tm.assert_frame_equal(result, expected)


def test_usecols_index_col_conflict2(all_parsers):
    # [FUNCTION DOC]
    # See gh-4201. usecols of three names with a two-column index_col list produces a frame indexed by a multi-index of columns b and c with column d remaining.
    # see gh-4201: test that index_col as integer reflects usecols
    parser = all_parsers
    data = "a,b,c,d\nA,a,1,one\nB,b,2,two"

    expected = DataFrame({"b": ["a", "b"], "c": [1, 2], "d": ("one", "two")})
    expected = expected.set_index(["b", "c"])

    result = parser.read_csv(
        StringIO(data), usecols=["b", "c", "d"], index_col=["b", "c"]
    )
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Expected 3 columns, got 4
def test_usecols_implicit_index_col(all_parsers):
    # [FUNCTION DOC]
    # See gh-2654. Rows have more fields than the header, so the leading field becomes an implicit index; selecting two names returns those columns indexed by the first values. Skipped on pyarrow.
    # see gh-2654
    parser = all_parsers
    data = "a,b,c\n4,apple,bat,5.7\n8,orange,cow,10"

    result = parser.read_csv(StringIO(data), usecols=["a", "b"])
    expected = DataFrame({"a": ["apple", "orange"], "b": ["bat", "cow"]}, index=[4, 8])
    tm.assert_frame_equal(result, expected)


def test_usecols_index_col_middle(all_parsers):
    # [FUNCTION DOC]
    # GH#9098. index_col naming a middle column among the selected columns yields that column as the named index and the other selected columns as data.
    # GH#9098
    parser = all_parsers
    data = """a,b,c,d
1,2,3,4
"""
    result = parser.read_csv(StringIO(data), usecols=["b", "c", "d"], index_col="c")
    expected = DataFrame({"b": [2], "d": [4]}, index=Index([3], name="c"))
    tm.assert_frame_equal(result, expected)


def test_usecols_index_col_end(all_parsers):
    # [FUNCTION DOC]
    # GH#9098. Same as the middle case but index_col names the last selected column.
    # GH#9098
    parser = all_parsers
    data = """a,b,c,d
1,2,3,4
"""
    result = parser.read_csv(StringIO(data), usecols=["b", "c", "d"], index_col="d")
    expected = DataFrame({"b": [2], "c": [3]}, index=Index([4], name="d"))
    tm.assert_frame_equal(result, expected)


def test_usecols_regex_sep(all_parsers):
    # [FUNCTION DOC]
    # See gh-2733. With a whitespace regex separator and two names selected, asserts the selected columns with an implicit index from the first field. For pyarrow asserts a value error that regex separators are unsupported.
    # see gh-2733
    parser = all_parsers
    data = "a  b  c\n4  apple  bat  5.7\n8  orange  cow  10"

    if parser.engine == "pyarrow":
        msg = "the 'pyarrow' engine does not support regex separators"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), sep=r"\s+", usecols=("a", "b"))
        return

    result = parser.read_csv(StringIO(data), sep=r"\s+", usecols=("a", "b"))

    expected = DataFrame({"a": ["apple", "orange"], "b": ["bat", "cow"]}, index=[4, 8])
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # Column 'a' in include_columns does not exist in CSV file
def test_usecols_with_whitespace(all_parsers):
    # [FUNCTION DOC]
    # Whitespace-separated data parsed with a regex separator and a tuple of names for usecols returns the two selected columns with an implicit index. Skipped on pyarrow.
    parser = all_parsers
    data = "a  b  c\n4  apple  bat  5.7\n8  orange  cow  10"

    result = parser.read_csv(StringIO(data), sep=r"\s+", usecols=("a", "b"))
    expected = DataFrame({"a": ["apple", "orange"], "b": ["bat", "cow"]}, index=[4, 8])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "usecols,expected",
    [
        # Column selection by index.
        ([0, 1], DataFrame(data=[[1000, 2000], [4000, 5000]], columns=["2", "0"])),
        # Column selection by name.
        (
            ["0", "1"],
            DataFrame(data=[[2000, 3000], [5000, 6000]], columns=["0", "1"]),
        ),
    ],
)
def test_usecols_with_integer_like_header(all_parsers, usecols, expected, request):
    # [FUNCTION DOC]
    # Parametrized over positional usecols and name-based usecols on a header consisting of numeric strings. Positional selection returns columns named "2" and "0"; name-based selection returns columns "0" and "1". For pyarrow with integer positions asserts the position value error.
    parser = all_parsers
    data = """2,0,1
1000,2000,3000
4000,5000,6000"""

    if parser.engine == "pyarrow" and isinstance(usecols[0], int):
        with pytest.raises(ValueError, match=_msg_pyarrow_requires_names):
            parser.read_csv(StringIO(data), usecols=usecols)
        return

    result = parser.read_csv(StringIO(data), usecols=usecols)
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # mismatched shape
def test_empty_usecols(all_parsers):
    # [FUNCTION DOC]
    # An empty set for usecols returns an empty frame with an empty column index. Expected to fail on pyarrow (shape mismatch).
    data = "a,b,c\n1,2,3\n4,5,6"
    expected = DataFrame(columns=Index([]))
    parser = all_parsers

    result = parser.read_csv(StringIO(data), usecols=set())
    tm.assert_frame_equal(result, expected)


def test_np_array_usecols(all_parsers):
    # [FUNCTION DOC]
    # See gh-12546. A numpy array of names is accepted as usecols and selects those columns.
    # see gh-12546
    parser = all_parsers
    data = "a,b,c\n1,2,3"
    usecols = np.array(["a", "b"])

    expected = DataFrame([[1, 2]], columns=usecols)
    result = parser.read_csv(StringIO(data), usecols=usecols)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "usecols,expected",
    [
        (
            lambda x: x.upper() in ["AAA", "BBB", "DDD"],
            DataFrame(
                {
                    "AaA": {
                        0: 0.056674972999999997,
                        1: 2.6132309819999997,
                        2: 3.5689350380000002,
                    },
                    "bBb": {0: 8, 1: 2, 2: 7},
                    "ddd": {0: "a", 1: "b", 2: "a"},
                }
            ),
        ),
        (lambda x: False, DataFrame(columns=Index([]))),
    ],
)
def test_callable_usecols(all_parsers, usecols, expected):
    # [FUNCTION DOC]
    # See gh-14154. Parametrized over a case-insensitive name predicate (selects three of four columns, preserving original header case) and a predicate that rejects everything (empty frame). For pyarrow asserts a value error that callable usecols are not allowed.
    # see gh-14154
    data = """AaA,bBb,CCC,ddd
0.056674973,8,True,a
2.613230982,2,False,b
3.568935038,7,False,a"""
    parser = all_parsers

    if parser.engine == "pyarrow":
        msg = "The pyarrow engine does not allow 'usecols' to be a callable"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), usecols=usecols)
        return

    result = parser.read_csv(StringIO(data), usecols=usecols)
    tm.assert_frame_equal(result, expected)


# ArrowKeyError: Column 'fa' in include_columns does not exist in CSV file
@skip_pyarrow
@pytest.mark.parametrize("usecols", [["a", "c"], lambda x: x in ["a", "c"]])
def test_incomplete_first_row(all_parsers, usecols):
    # [FUNCTION DOC]
    # See gh-6710. Parametrized over a name list and an equivalent predicate; with three names given and a short first row, asserts the selected columns a and c with missing values filled for the short row. Skipped on pyarrow.
    # see gh-6710
    data = "1,2\n1,2,3"
    parser = all_parsers
    names = ["a", "b", "c"]
    expected = DataFrame({"a": [1, 1], "c": [np.nan, 3]})

    result = parser.read_csv(StringIO(data), names=names, usecols=usecols)
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Expected 3 columns, got 4
@pytest.mark.parametrize(
    "data,usecols,kwargs,expected",
    [
        # see gh-8985
        (
            "19,29,39\n" * 2 + "10,20,30,40",
            [0, 1, 2],
            {"header": None},
            [[19, 29, 39], [19, 29, 39], [10, 20, 30]],
        ),
        # see gh-9549
        (
            ("A,B,C\n1,2,3\n3,4,5\n1,2,4,5,1,6\n1,2,3,,,1,\n1,2,3\n5,6,7"),
            ["A", "B", "C"],
            {},
            {
                "A": [1, 3, 1, 1, 1, 5],
                "B": [2, 4, 2, 2, 2, 6],
                "C": [3, 5, 4, 3, 3, 7],
            },
        ),
    ],
)
def test_uneven_length_cols(all_parsers, data, usecols, kwargs, expected):
    # [FUNCTION DOC]
    # See gh-8985 and gh-9549. Parametrized over data whose later rows have extra fields: header-less numeric rows with positional usecols, and named columns with a long row. Asserts the extra fields are dropped and the frame has the selected columns. Skipped on pyarrow.
    # see gh-8985
    parser = all_parsers
    result = parser.read_csv(StringIO(data), usecols=usecols, **kwargs)
    expected = DataFrame(expected)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "usecols,kwargs,expected,msg",
    [
        (
            ["a", "b", "c", "d"],
            {},
            DataFrame({"a": [1, 5], "b": [2, 6], "c": [3, 7], "d": [4, 8]}),
            None,
        ),
        (
            ["a", "b", "c", "f"],
            {},
            None,
            _msg_validate_usecols_names.format(r"\['f'\]"),
        ),
        (["a", "b", "f"], {}, None, _msg_validate_usecols_names.format(r"\['f'\]")),
        (
            ["a", "b", "f", "g"],
            {},
            None,
            _msg_validate_usecols_names.format(r"\[('f', 'g'|'g', 'f')\]"),
        ),
        # see gh-14671
        (
            None,
            {"header": 0, "names": ["A", "B", "C", "D"]},
            DataFrame({"A": [1, 5], "B": [2, 6], "C": [3, 7], "D": [4, 8]}),
            None,
        ),
        (
            ["A", "B", "C", "f"],
            {"header": 0, "names": ["A", "B", "C", "D"]},
            None,
            _msg_validate_usecols_names.format(r"\['f'\]"),
        ),
        (
            ["A", "B", "f"],
            {"names": ["A", "B", "C", "D"]},
            None,
            _msg_validate_usecols_names.format(r"\['f'\]"),
        ),
    ],
)
def test_raises_on_usecols_names_mismatch(
    all_parsers, usecols, kwargs, expected, msg, request
):
    # [FUNCTION DOC]
    # Parametrized over usecols/keyword/expected/message cases: all valid names (frame returned), names including one or two missing ones (value error naming the missing columns), and cases using supplied names with or without header (gh-14671). Mutates the parametrized keyword dictionary by adding the usecols entry before the call. For pyarrow, everything except the first (valid) case is skipped due to an upstream key error. Asserts either the value error message or the expected frame.
    data = "a,b,c,d\n1,2,3,4\n5,6,7,8"
    kwargs.update(usecols=usecols)
    parser = all_parsers

    if parser.engine == "pyarrow" and not (
        usecols is not None and expected is not None
    ):
        # everything but the first case
        # ArrowKeyError: Column 'f' in include_columns does not exist in CSV file
        pytest.skip(reason="https://github.com/apache/arrow/issues/38676")

    if expected is None:
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), **kwargs)
    else:
        result = parser.read_csv(StringIO(data), **kwargs)
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("usecols", [["A", "C"], [0, 2]])
def test_usecols_subset_names_mismatch_orig_columns(all_parsers, usecols, request):
    # [FUNCTION DOC]
    # Parametrized over name-based and positional usecols with supplied names replacing the header: selection by the new names or by position returns columns A and C. For pyarrow, positional selection asserts the position value error; name-based selection is skipped.
    data = "a,b,c,d\n1,2,3,4\n5,6,7,8"
    names = ["A", "B", "C", "D"]
    parser = all_parsers

    if parser.engine == "pyarrow":
        if isinstance(usecols[0], int):
            with pytest.raises(ValueError, match=_msg_pyarrow_requires_names):
                parser.read_csv(StringIO(data), header=0, names=names, usecols=usecols)
            return
        # "pyarrow.lib.ArrowKeyError: Column 'A' in include_columns does not exist"
        pytest.skip(reason="https://github.com/apache/arrow/issues/38676")

    result = parser.read_csv(StringIO(data), header=0, names=names, usecols=usecols)
    expected = DataFrame({"A": [1, 5], "C": [3, 7]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("names", [None, ["a", "b"]])
def test_usecols_indices_out_of_bounds(all_parsers, names):
    # [FUNCTION DOC]
    # GH#25623 and GH 41130 (enforced in 2.0). Parametrized over names None or two names: a positional usecols entry beyond the available columns raises a parser error about out-of-bounds usecols, or for pyarrow a value error that integer positions are unsupported.
    # GH#25623 & GH 41130; enforced in 2.0
    parser = all_parsers
    data = """
a,b
1,2
    """

    err = ParserError
    msg = "Defining usecols with out-of-bounds"
    if parser.engine == "pyarrow":
        err = ValueError
        msg = _msg_pyarrow_requires_names

    with pytest.raises(err, match=msg):
        parser.read_csv(StringIO(data), usecols=[0, 2], names=names, header=0)


def test_usecols_additional_columns(all_parsers):
    # [FUNCTION DOC]
    # GH#46997. A callable that strips whitespace from header names, applied to a header with fewer fields than a data row and with no index column, selects columns a and b and returns one row. For pyarrow asserts the callable-unsupported value error.
    # GH#46997
    parser = all_parsers
    usecols = lambda header: header.strip() in ["a", "b", "c"]

    if parser.engine == "pyarrow":
        msg = "The pyarrow engine does not allow 'usecols' to be a callable"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO("a,b\nx,y,z"), index_col=False, usecols=usecols)
        return
    result = parser.read_csv(StringIO("a,b\nx,y,z"), index_col=False, usecols=usecols)
    expected = DataFrame({"a": ["x"], "b": "y"})
    tm.assert_frame_equal(result, expected)


def test_usecols_additional_columns_integer_columns(all_parsers):
    # [FUNCTION DOC]
    # GH#46997. Same as the additional-columns test but with numeric-string headers; returns columns "0" and "1" with one row. For pyarrow asserts the callable-unsupported value error.
    # GH#46997
    parser = all_parsers
    usecols = lambda header: header.strip() in ["0", "1"]
    if parser.engine == "pyarrow":
        msg = "The pyarrow engine does not allow 'usecols' to be a callable"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO("0,1\nx,y,z"), index_col=False, usecols=usecols)
        return
    result = parser.read_csv(StringIO("0,1\nx,y,z"), index_col=False, usecols=usecols)
    expected = DataFrame({"0": ["x"], "1": "y"})
    tm.assert_frame_equal(result, expected)


def test_usecols_dtype(all_parsers):
    # [FUNCTION DOC]
    # Reading with usecols of two columns and a dtype mapping that also includes the unselected third column; asserts the result has just the two selected columns with the requested string and unsigned 8-bit integer dtypes (the unselected column's dtype is ignored).
    parser = all_parsers
    data = """
col1,col2,col3
a,1,x
b,2,y
"""
    result = parser.read_csv(
        StringIO(data),
        usecols=["col1", "col2"],
        dtype={"col1": "string", "col2": "uint8", "col3": "string"},
    )
    expected = DataFrame(
        {"col1": array(["a", "b"]), "col2": np.array([1, 2], dtype="uint8")}
    )
    tm.assert_frame_equal(result, expected)
