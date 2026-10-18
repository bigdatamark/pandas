"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""

# [MODULE DOC]
# Parser tests (all engines via the parser fixture) of integer and boolean parsing:
# float-looking integers, boolean text and custom true/false values, integers beyond
# floating-point precision or the 64-bit ranges, overflow handling with converters, and
# precision of large integers in wide rows.
# Defines two marker aliases: one marks a test as expected to fail for the pyarrow
# engine and one skips it for that engine.

from io import StringIO

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")


def test_int_conversion(all_parsers):
    # [FUNCTION DOC]
    # Verifies a column of values written with a decimal point is read as floats while a
    # column of plain integers is read as integers.
    data = """A,B
1.0,1
2.0,2
3.0,3
"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data))

    expected = DataFrame([[1.0, 1], [2.0, 2], [3.0, 3]], columns=["A", "B"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data,kwargs,expected",
    [
        (
            "A,B\nTrue,1\nFalse,2\nTrue,3",
            {},
            [[True, 1], [False, 2], [True, 3]],
        ),
        (
            "A,B\nYES,1\nno,2\nyes,3\nNo,3\nYes,3",
            {"true_values": ["yes", "Yes", "YES"], "false_values": ["no", "NO", "No"]},
            [[True, 1], [False, 2], [True, 3], [False, 3], [True, 3]],
        ),
        (
            "A,B\nTRUE,1\nFALSE,2\nTRUE,3",
            {},
            [[True, 1], [False, 2], [True, 3]],
        ),
        (
            "A,B\nfoo,bar\nbar,foo",
            {"true_values": ["foo"], "false_values": ["bar"]},
            [[True, False], [False, True]],
        ),
    ],
)
def test_parse_bool(all_parsers, data, kwargs, expected):
    # [FUNCTION DOC]
    # Verifies boolean parsing: True/False and TRUE/FALSE text become booleans by
    # default; custom true-values and false-values lists convert other words (case
    # variants of yes and no, or arbitrary words) to booleans; the second column stays
    # integer or becomes boolean as given.
    parser = all_parsers
    result = parser.read_csv(StringIO(data), **kwargs)
    expected = DataFrame(expected, columns=["A", "B"])
    tm.assert_frame_equal(result, expected)


def test_parse_integers_above_fp_precision(all_parsers):
    # [FUNCTION DOC]
    # Verifies ten integers around seventeen digits long, which cannot all be
    # represented exactly as floats, are read exactly as 64-bit integers.
    data = """Numbers
17007000002000191
17007000002000191
17007000002000191
17007000002000191
17007000002000192
17007000002000192
17007000002000192
17007000002000192
17007000002000192
17007000002000194"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data))
    expected = DataFrame(
        {
            "Numbers": [
                17007000002000191,
                17007000002000191,
                17007000002000191,
                17007000002000191,
                17007000002000192,
                17007000002000192,
                17007000002000192,
                17007000002000192,
                17007000002000192,
                17007000002000194,
            ]
        }
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("sep", [" ", r"\s+"])
def test_integer_overflow_bug(all_parsers, sep):
    # [FUNCTION DOC]
    # Verifies that rows of two fields, where the first looks like a huge exponent
    # number, parse (with a single-space or whitespace-regex separator) into a float and
    # an integer column with the expected values (regression for an integer overflow).
    # For pyarrow with the regex separator, verifies the regex separator unsupported
    # error.
    # see gh-2601
    data = "65248E10 11\n55555E55 22\n"
    parser = all_parsers
    if parser.engine == "pyarrow" and sep != " ":
        msg = "the 'pyarrow' engine does not support regex separators"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), header=None, sep=sep)
        return

    result = parser.read_csv(StringIO(data), header=None, sep=sep)
    expected = DataFrame([[6.5248e14, 11], [5.5555e59, 22]])
    tm.assert_frame_equal(result, expected)


def test_int64_min_issues(all_parsers):
    # [FUNCTION DOC]
    # Verifies a column with integer zero and an empty field is read as floats with NaN,
    # and the other column stays integer.
    # see gh-2599
    parser = all_parsers
    data = "A,B\n0,0\n0,"
    result = parser.read_csv(StringIO(data))

    expected = DataFrame({"A": [0, 0], "B": [0, np.nan]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("conv", [None, np.int64, np.uint64])
def test_int64_overflow(all_parsers, conv, request):
    # [FUNCTION DOC]
    # Uses ID values with leading zeros and more than the maximum unsigned 64-bit
    # magnitude. With no converter, verifies they are read as Python integers in an
    # object column (for pyarrow this is expected to fail since it parses floats). With
    # a signed or unsigned 64-bit converter, verifies an overflow error is raised (a
    # value error stating converters are unsupported for pyarrow).
    data = """ID
00013007854817840016671868
00013007854817840016749251
00013007854817840016754630
00013007854817840016781876
00013007854817840017028824
00013007854817840017963235
00013007854817840018860166"""
    parser = all_parsers

    if conv is None:
        # 13007854817840016671868 > UINT64_MAX, so this
        # will overflow and return object as the dtype.
        if parser.engine == "pyarrow":
            mark = pytest.mark.xfail(reason="parses to float64")
            request.applymarker(mark)

        result = parser.read_csv(StringIO(data))
        expected = DataFrame(
            [
                13007854817840016671868,
                13007854817840016749251,
                13007854817840016754630,
                13007854817840016781876,
                13007854817840017028824,
                13007854817840017963235,
                13007854817840018860166,
            ],
            columns=["ID"],
        )
        tm.assert_frame_equal(result, expected)
    else:
        # 13007854817840016671868 > UINT64_MAX, so attempts
        # to cast to either int64 or uint64 will result in
        # an OverflowError being raised.
        msg = "|".join(
            [
                "Python int too large to convert to C long",
                "long too big to convert",
                "int too big to convert",
            ]
        )
        err = OverflowError
        if parser.engine == "pyarrow":
            err = ValueError
            msg = "The 'converters' option is not supported with the 'pyarrow' engine"

        with pytest.raises(err, match=msg):
            parser.read_csv(StringIO(data), converters={"ID": conv})


@skip_pyarrow  # CSV parse error: Empty CSV file or block
@pytest.mark.parametrize(
    "val", [np.iinfo(np.uint64).max, np.iinfo(np.int64).max, np.iinfo(np.int64).min]
)
def test_int64_uint64_range(all_parsers, val):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Verifies the maximum unsigned 64-bit value and the maximum
    # and minimum signed 64-bit values, given as a single no-header value, are parsed
    # exactly as integers.
    # These numbers fall right inside the int64-uint64
    # range, so they should be parsed as integer.
    parser = all_parsers
    result = parser.read_csv(StringIO(str(val)), header=None)

    expected = DataFrame([val])
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Empty CSV file or block
@pytest.mark.parametrize(
    "val", [np.iinfo(np.uint64).max + 1, np.iinfo(np.int64).min - 1]
)
def test_outside_int64_uint64_range(all_parsers, val, request):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Verifies values just beyond the unsigned 64-bit maximum or
    # below the signed 64-bit minimum are parsed as Python integers in an object column.
    # The request fixture is accepted but unused.
    # These numbers fall just outside the int64-uint64
    # range, so they should be parsed as object.
    parser = all_parsers

    result = parser.read_csv(StringIO(str(val)), header=None)

    expected = DataFrame([val])
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Empty CSV file or block
@pytest.mark.parametrize(
    "val", [np.iinfo(np.uint64).max + 1, np.iinfo(np.int64).min - 1]
)
def test_outside_int64_uint64_range_follow_str(all_parsers, val):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Verifies that an out-of-range integer followed by a
    # non-numeric string yields an object column of the strings of both values.
    parser = all_parsers

    result = parser.read_csv(StringIO(f"{val}\nabc"), header=None)

    expected = DataFrame([str(val), "abc"])
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # gets float64 dtype instead of object
@pytest.mark.parametrize("exp_data", [[str(-1), str(2**63)], [str(2**63), str(-1)]])
def test_numeric_range_too_wide(all_parsers, exp_data):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow (yields float instead of object). Verifies
    # that a column holding a negative number and a number above the signed 64-bit
    # range, in either order, cannot fit one numeric dtype and is returned as an object
    # column of strings.
    # No numerical dtype can hold both negative and uint64
    # values, so they should be cast as string.
    parser = all_parsers
    data = "\n".join(exp_data)
    expected = DataFrame(exp_data)

    result = parser.read_csv(StringIO(data), header=None)
    tm.assert_frame_equal(result, expected)


def test_integer_precision(all_parsers):
    # [FUNCTION DOC]
    # Verifies that the fifth column of two wide rows (with semicolon-containing fields
    # and empty fields) is parsed as a series of exact 64-bit integers with name four,
    # without losing precision.
    # Gh 7072
    s = """1,1;0;0;0;1;1;3844;3844;3844;1;1;1;1;1;1;0;0;1;1;0;0,,,4321583677327450765
5,1;0;0;0;1;1;843;843;843;1;1;1;1;1;1;0;0;1;1;0;0,64.0,;,4321113141090630389"""
    parser = all_parsers
    result = parser.read_csv(StringIO(s), header=None)[4]
    expected = Series([4321583677327450765, 4321113141090630389], name=4)
    tm.assert_series_equal(result, expected)
