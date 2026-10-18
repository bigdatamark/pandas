"""
Tests that NA values are properly handled during
parsing for all of the parsers defined in parsers.py
"""

# [MODULE DOC]
# Parser tests (all engines via the parser fixture) of missing-value handling: default
# missing-value strings, custom missing values (scalar, list, per-column dictionary),
# keeping or dropping the defaults, the missing-value filter option, interactions with
# dtypes, index columns, multi-level headers, booleans and large integers.
# KNOWN ISSUE: the module docstring says the parsers are defined in a file named
# parsers.py, but no such module exists here; the parser fixtures come from the
# package's shared fixtures module.
# Imports the set of default missing-value strings from the parsers extension. Defines
# two marker aliases: one marks a test as expected to fail for the pyarrow engine and
# one skips it for that engine. The pyarrow engine requires string missing values and
# does not accept a per-column dictionary, so several tests expect errors for it.

from io import StringIO

import numpy as np
import pytest

from pandas._libs.parsers import STR_NA_VALUES

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


def test_string_nas(all_parsers):
    # [FUNCTION DOC]
    # Verifies empty fields in string columns are read as missing values (None for
    # pyarrow, NaN otherwise).
    parser = all_parsers
    data = """A,B,C
a,b,c
d,,f
,g,h
"""
    result = parser.read_csv(StringIO(data))
    expected = DataFrame(
        [["a", "b", "c"], ["d", np.nan, "f"], [np.nan, "g", "h"]],
        columns=["A", "B", "C"],
    )
    if parser.engine == "pyarrow":
        expected.loc[2, "A"] = None
        expected.loc[1, "B"] = None
    tm.assert_frame_equal(result, expected)


def test_detect_string_na(all_parsers):
    # [FUNCTION DOC]
    # Verifies the default missing-value spellings NA, NaN and nan in string columns are
    # converted to missing values (None for pyarrow).
    parser = all_parsers
    data = """A,B
foo,bar
NA,baz
NaN,nan
"""
    expected = DataFrame(
        [["foo", "bar"], [np.nan, "baz"], [np.nan, np.nan]], columns=["A", "B"]
    )
    if parser.engine == "pyarrow":
        expected.loc[[1, 2], "A"] = None
        expected.loc[2, "B"] = None
    result = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "na_values",
    [
        ["-999.0", "-999"],
        [-999, -999.0],
        [-999.0, -999],
        ["-999.0"],
        ["-999"],
        [-999.0],
        [-999],
    ],
)
@pytest.mark.parametrize(
    "data",
    [
        """A,B
-999,1.2
2,-999
3,4.5
""",
        """A,B
-999,1.200
2,-999.000
3,4.500
""",
    ],
)
def test_non_string_na_values(all_parsers, data, na_values, request):
    # [FUNCTION DOC]
    # Verifies numeric and string custom missing values (-999 given as ints, floats or
    # strings in various combinations) match -999 in the data written with or without
    # trailing zeros, producing NaN in a float frame. For pyarrow, non-string values
    # expect a type error that all values must be strings, and the trailing-zeros data
    # is marked expected to fail since equivalent floats are not recognized.
    # see gh-3611: with an odd float format, we can't match
    # the string "999.0" exactly but still need float matching
    parser = all_parsers
    expected = DataFrame([[np.nan, 1.2], [2.0, np.nan], [3.0, 4.5]], columns=["A", "B"])

    if parser.engine == "pyarrow" and not all(isinstance(x, str) for x in na_values):
        msg = "The 'pyarrow' engine requires all na_values to be strings"
        with pytest.raises(TypeError, match=msg):
            parser.read_csv(StringIO(data), na_values=na_values)
        return
    elif parser.engine == "pyarrow" and "-999.000" in data:
        # bc the pyarrow engine does not include the float-ified version
        #  of "-999" -> -999, it does not match the entry with the trailing
        #  zeros, so "-999.000" is not treated as null.
        mark = pytest.mark.xfail(
            reason="pyarrow engined does not recognize equivalent floats"
        )
        request.applymarker(mark)

    result = parser.read_csv(StringIO(data), na_values=na_values)
    tm.assert_frame_equal(result, expected)


def test_default_na_values(all_parsers):
    # [FUNCTION DOC]
    # Verifies the library's default missing-value string set equals a hard-coded list
    # of nineteen spellings (including NA variants, null, nan, infinity-marker forms and
    # the empty string), then builds a square block of data where each row has one of
    # those spellings on the diagonal and verifies every cell is read as NaN.
    _NA_VALUES = {
        "-1.#IND",
        "1.#QNAN",
        "1.#IND",
        "-1.#QNAN",
        "#N/A",
        "N/A",
        "n/a",
        "NA",
        "<NA>",
        "#NA",
        "NULL",
        "null",
        "NaN",
        "nan",
        "-NaN",
        "-nan",
        "#N/A N/A",
        "",
        "None",
    }
    assert _NA_VALUES == STR_NA_VALUES

    parser = all_parsers
    nv = len(_NA_VALUES)

    def f(i, v):
        # [FUNCTION DOC]
        # HELPER: builds one CSV row for the default-NA test. Takes the row number and a
        # missing-value spelling; returns the spelling preceded by that many commas and
        # followed by commas to fill out the total column count.
        if i == 0:
            buf = ""
        elif i > 0:
            buf = "".join([","] * i)

        buf = f"{buf}{v}"

        if i < nv - 1:
            joined = "".join([","] * (nv - i - 1))
            buf = f"{buf}{joined}"

        return buf

    data = StringIO("\n".join([f(i, v) for i, v in enumerate(_NA_VALUES)]))
    expected = DataFrame(np.nan, columns=range(nv), index=range(nv))

    result = parser.read_csv(data, header=None)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("na_values", ["baz", ["baz"]])
def test_custom_na_values(all_parsers, na_values):
    # [FUNCTION DOC]
    # Verifies a custom missing value given as a string or a one-element list is applied
    # together with the defaults, while skipping the second line (a junk row). For
    # pyarrow, verifies the value error that skiprows must be an integer.
    parser = all_parsers
    data = """A,B,C
ignore,this,row
1,NA,3
-1.#IND,5,baz
7,8,NaN
"""
    expected = DataFrame(
        [[1.0, np.nan, 3], [np.nan, 5, np.nan], [7, 8, np.nan]], columns=["A", "B", "C"]
    )
    if parser.engine == "pyarrow":
        msg = "skiprows argument must be an integer when using engine='pyarrow'"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), na_values=na_values, skiprows=[1])
        return

    result = parser.read_csv(StringIO(data), na_values=na_values, skiprows=[1])
    tm.assert_frame_equal(result, expected)


def test_bool_na_values(all_parsers):
    # [FUNCTION DOC]
    # Verifies columns of True/False with NA are read as object columns of booleans and
    # NaN (None for pyarrow), while a complete column is read as boolean.
    data = """A,B,C
True,False,True
NA,True,False
False,NA,True"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data))
    expected = DataFrame(
        {
            "A": np.array([True, np.nan, False], dtype=object),
            "B": np.array([False, True, np.nan], dtype=object),
            "C": [True, False, True],
        }
    )
    if parser.engine == "pyarrow":
        expected.loc[1, "A"] = None
        expected.loc[2, "B"] = None
    tm.assert_frame_equal(result, expected)


def test_na_value_dict(all_parsers):
    # [FUNCTION DOC]
    # Verifies a per-column dictionary of custom missing values makes the listed strings
    # missing in those columns while the default markers still apply to the others. For
    # pyarrow, verifies the dictionary-unsupported value error.
    data = """A,B,C
foo,bar,NA
bar,foo,foo
foo,bar,NA
bar,foo,foo"""
    parser = all_parsers

    if parser.engine == "pyarrow":
        msg = "pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), na_values={"A": ["foo"], "B": ["bar"]})
        return

    df = parser.read_csv(StringIO(data), na_values={"A": ["foo"], "B": ["bar"]})
    expected = DataFrame(
        {
            "A": [np.nan, "bar", np.nan, "bar"],
            "B": [np.nan, "foo", np.nan, "foo"],
            "C": [np.nan, "foo", np.nan, "foo"],
        }
    )
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    "index_col,expected",
    [
        (
            [0],
            DataFrame({"b": [np.nan], "c": [1], "d": [5]}, index=Index([0], name="a")),
        ),
        (
            [0, 2],
            DataFrame(
                {"b": [np.nan], "d": [5]},
                index=MultiIndex.from_tuples([(0, 1)], names=["a", "c"]),
            ),
        ),
        (
            ["a", "c"],
            DataFrame(
                {"b": [np.nan], "d": [5]},
                index=MultiIndex.from_tuples([(0, 1)], names=["a", "c"]),
            ),
        ),
    ],
)
def test_na_value_dict_multi_index(all_parsers, index_col, expected):
    # [FUNCTION DOC]
    # Verifies that, passing an empty set of custom missing values (the default markers
    # still apply) and index columns given by position or name, the NA entry in column
    # b is read as NaN and the result is a single-row frame with a plain or two-level
    # index.
    # KNOWN ISSUE: the test name mentions a dictionary of missing values, but an empty
    # set is passed; no dictionary is involved.
    data = """\
a,b,c,d
0,NA,1,5
"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data), na_values=set(), index_col=index_col)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        (
            {},
            {
                "A": ["a", "b", np.nan, "d", "e", np.nan, "g"],
                "B": [1, 2, 3, 4, 5, 6, 7],
                "C": ["one", "two", "three", np.nan, "five", np.nan, "seven"],
            },
        ),
        (
            {"na_values": {"A": [], "C": []}, "keep_default_na": False},
            {
                "A": ["a", "b", "", "d", "e", "nan", "g"],
                "B": [1, 2, 3, 4, 5, 6, 7],
                "C": ["one", "two", "three", "nan", "five", "", "seven"],
            },
        ),
        (
            {"na_values": ["a"], "keep_default_na": False},
            {
                "A": [np.nan, "b", "", "d", "e", "nan", "g"],
                "B": [1, 2, 3, 4, 5, 6, 7],
                "C": ["one", "two", "three", "nan", "five", "", "seven"],
            },
        ),
        (
            {"na_values": {"A": [], "C": []}},
            {
                "A": ["a", "b", np.nan, "d", "e", np.nan, "g"],
                "B": [1, 2, 3, 4, 5, 6, 7],
                "C": ["one", "two", "three", np.nan, "five", np.nan, "seven"],
            },
        ),
    ],
)
def test_na_values_keep_default(
    all_parsers, kwargs, expected, request, using_infer_string
):
    # [FUNCTION DOC]
    # Parametrized over: defaults only; a per-column dictionary of empty lists with the
    # defaults turned off (empty and nan strings kept); a list of one value with
    # defaults turned off; a dictionary of empty lists with defaults on. Verifies which
    # spellings become missing. For pyarrow, dictionary inputs expect the
    # unsupported-dictionary error and other cases are marked expected to fail depending
    # on string inference.
    data = """\
A,B,C
a,1,one
b,2,two
,3,three
d,4,nan
e,5,five
nan,6,
g,7,seven
"""
    parser = all_parsers
    if parser.engine == "pyarrow":
        if "na_values" in kwargs and isinstance(kwargs["na_values"], dict):
            msg = "The pyarrow engine doesn't support passing a dict for na_values"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(StringIO(data), **kwargs)
            return
        if not using_infer_string or "na_values" in kwargs:
            mark = pytest.mark.xfail()
            request.applymarker(mark)

    result = parser.read_csv(StringIO(data), **kwargs)
    expected = DataFrame(expected)
    tm.assert_frame_equal(result, expected)


def test_no_na_values_no_keep_default(all_parsers):
    # [FUNCTION DOC]
    # Verifies that with the default missing values turned off and no custom values, the
    # strings None, nan and empty stay as strings.
    # see gh-4318: passing na_values=None and
    # keep_default_na=False yields 'None" as a na_value
    data = """\
A,B,C
a,1,None
b,2,two
,3,None
d,4,nan
e,5,five
nan,6,
g,7,seven
"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data), keep_default_na=False)

    expected = DataFrame(
        {
            "A": ["a", "b", "", "d", "e", "nan", "g"],
            "B": [1, 2, 3, 4, 5, 6, 7],
            "C": ["None", "two", "None", "nan", "five", "", "seven"],
        }
    )
    tm.assert_frame_equal(result, expected)


def test_no_keep_default_na_dict_na_values(all_parsers):
    # [FUNCTION DOC]
    # Verifies a dictionary of custom missing values for one column with defaults turned
    # off makes only that column's listed value missing and keeps the empty string in
    # the other column. For pyarrow, verifies the unsupported-dictionary error.
    # see gh-19227
    data = "a,b\n,2"
    parser = all_parsers

    if parser.engine == "pyarrow":
        msg = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data), na_values={"b": ["2"]}, keep_default_na=False
            )
        return

    result = parser.read_csv(
        StringIO(data), na_values={"b": ["2"]}, keep_default_na=False
    )
    expected = DataFrame({"a": [""], "b": [np.nan]})
    tm.assert_frame_equal(result, expected)


def test_no_keep_default_na_dict_na_scalar_values(all_parsers):
    # [FUNCTION DOC]
    # Verifies a scalar (not list) value inside the per-column dictionary works with
    # defaults turned off. For pyarrow, verifies the unsupported-dictionary error.
    # see gh-19227
    #
    # Scalar values shouldn't cause the parsing to crash or fail.
    data = "a,b\n1,2"
    parser = all_parsers

    if parser.engine == "pyarrow":
        msg = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), na_values={"b": 2}, keep_default_na=False)
        return

    df = parser.read_csv(StringIO(data), na_values={"b": 2}, keep_default_na=False)
    expected = DataFrame({"a": [1], "b": [np.nan]})
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize("col_zero_na_values", [113125, "113125"])
def test_no_keep_default_na_dict_na_values_diff_reprs(all_parsers, col_zero_na_values):
    # [FUNCTION DOC]
    # Verifies a per-column dictionary where the same number is given as an integer or
    # as a string is matched against column values in either representation, along with
    # string and empty entries, with defaults turned off. For pyarrow, verifies the
    # unsupported-dictionary error.
    # see gh-19227
    data = """\
113125,"blah","/blaha",kjsdkj,412.166,225.874,214.008
729639,"qwer","",asdfkj,466.681,,252.373
"""
    parser = all_parsers
    expected = DataFrame(
        {
            0: [np.nan, 729639.0],
            1: [np.nan, "qwer"],
            2: ["/blaha", np.nan],
            3: ["kjsdkj", "asdfkj"],
            4: [412.166, 466.681],
            5: ["225.874", ""],
            6: [np.nan, 252.373],
        }
    )

    if parser.engine == "pyarrow":
        msg = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data),
                header=None,
                keep_default_na=False,
                na_values={2: "", 6: "214.008", 1: "blah", 0: col_zero_na_values},
            )
        return

    result = parser.read_csv(
        StringIO(data),
        header=None,
        keep_default_na=False,
        na_values={2: "", 6: "214.008", 1: "blah", 0: col_zero_na_values},
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "na_filter,row_data",
    [
        (True, [[1, "A"], [np.nan, np.nan], [3, "C"]]),
        (False, [["1", "A"], ["nan", "B"], ["3", "C"]]),
    ],
)
def test_na_values_na_filter_override(
    request, all_parsers, na_filter, row_data, using_infer_string
):
    # [FUNCTION DOC]
    # Verifies that with the missing-value filter on, the nan text and a custom marker B
    # become NaN; with the filter off nothing is converted and all values remain
    # strings. For pyarrow, most cases are marked expected to fail (unsupported or
    # mismatched dtypes).
    parser = all_parsers
    if parser.engine == "pyarrow":
        # mismatched dtypes in both cases, FutureWarning in the True case
        if not (using_infer_string and na_filter):
            mark = pytest.mark.xfail(reason="pyarrow doesn't support this.")
            request.applymarker(mark)
    data = """\
A,B
1,A
nan,B
3,C
"""
    result = parser.read_csv(StringIO(data), na_values=["B"], na_filter=na_filter)

    expected = DataFrame(row_data, columns=["A", "B"])
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Expected 8 columns, got 5:
def test_na_trailing_columns(all_parsers):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Verifies rows shorter than the header have the trailing
    # columns filled with NaN.
    parser = all_parsers
    data = """Date,Currency,Symbol,Type,Units,UnitPrice,Cost,Tax
2012-03-14,USD,AAPL,BUY,1000
2012-05-12,USD,SBUX,SELL,500"""

    # Trailing columns should be all NaN.
    result = parser.read_csv(StringIO(data))
    expected = DataFrame(
        [
            ["2012-03-14", "USD", "AAPL", "BUY", 1000, np.nan, np.nan, np.nan],
            ["2012-05-12", "USD", "SBUX", "SELL", 500, np.nan, np.nan, np.nan],
        ],
        columns=[
            "Date",
            "Currency",
            "Symbol",
            "Type",
            "Units",
            "UnitPrice",
            "Cost",
            "Tax",
        ],
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "na_values,row_data",
    [
        (1, [[np.nan, 2.0], [2.0, np.nan]]),
        ({"a": 2, "b": 1}, [[1.0, 2.0], [np.nan, np.nan]]),
    ],
)
def test_na_values_scalar(all_parsers, na_values, row_data):
    # [FUNCTION DOC]
    # Verifies a scalar missing value (integer) or a per-column dictionary of scalar
    # values marks matching numbers as NaN in the named columns. For pyarrow, verifies
    # the unsupported-dictionary error for the dictionary case and a type error
    # requiring string values for the scalar case.
    # see gh-12224
    parser = all_parsers
    names = ["a", "b"]
    data = "1,2\n2,1"

    if parser.engine == "pyarrow" and isinstance(na_values, dict):
        msg = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), names=names, na_values=na_values)
        return
    elif parser.engine == "pyarrow":
        msg = "The 'pyarrow' engine requires all na_values to be strings"
        with pytest.raises(TypeError, match=msg):
            parser.read_csv(StringIO(data), names=names, na_values=na_values)
        return

    result = parser.read_csv(StringIO(data), names=names, na_values=na_values)
    expected = DataFrame(row_data, columns=names)
    tm.assert_frame_equal(result, expected)


def test_na_values_dict_aliasing(all_parsers):
    # [FUNCTION DOC]
    # Verifies reading with a dictionary of missing values gives the expected NaNs and
    # does not modify the dictionary passed in. For pyarrow, verifies the
    # unsupported-dictionary error.
    parser = all_parsers
    na_values = {"a": 2, "b": 1}
    na_values_copy = na_values.copy()

    names = ["a", "b"]
    data = "1,2\n2,1"

    expected = DataFrame([[1.0, 2.0], [np.nan, np.nan]], columns=names)

    if parser.engine == "pyarrow":
        msg = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), names=names, na_values=na_values)
        return

    result = parser.read_csv(StringIO(data), names=names, na_values=na_values)

    tm.assert_frame_equal(result, expected)
    tm.assert_dict_equal(na_values, na_values_copy)


def test_na_values_dict_null_column_name(all_parsers):
    # [FUNCTION DOC]
    # Regression: with a column whose name is none, names including none, a dictionary
    # mapping each name to the default missing-value set, explicit dtypes and defaults
    # turned off, verifies the first column becomes the index and the remaining columns
    # have NaN for the empty value. For pyarrow, verifies the unsupported-dictionary
    # error.
    # see gh-57547
    parser = all_parsers
    data = ",x,y\n\nMA,1,2\nNA,2,1\nOA,,3"
    names = [None, "x", "y"]
    na_values = dict.fromkeys(names, STR_NA_VALUES)
    dtype = {None: "object", "x": "float64", "y": "float64"}

    if parser.engine == "pyarrow":
        msg = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data),
                index_col=0,
                header=0,
                dtype=dtype,
                names=names,
                na_values=na_values,
                keep_default_na=False,
            )
        return

    expected = DataFrame(
        {"x": [1.0, 2.0, np.nan], "y": [2.0, 1.0, 3.0]},
        index=Index(["MA", "NA", "OA"], dtype=object),
    )

    result = parser.read_csv(
        StringIO(data),
        index_col=0,
        header=0,
        dtype=dtype,
        names=names,
        na_values=na_values,
        keep_default_na=False,
    )

    tm.assert_frame_equal(result, expected)


def test_na_values_dict_col_index(all_parsers):
    # [FUNCTION DOC]
    # Verifies a dictionary keyed by column position (zero) applies its missing value to
    # that column. For pyarrow, verifies the unsupported-dictionary error.
    # see gh-14203
    data = "a\nfoo\n1"
    parser = all_parsers
    na_values = {0: "foo"}

    if parser.engine == "pyarrow":
        msg = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), na_values=na_values)
        return

    result = parser.read_csv(StringIO(data), na_values=na_values)
    expected = DataFrame({"a": [np.nan, 1]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data,kwargs,expected",
    [
        (
            str(2**63) + "\n" + str(2**63 + 1),
            {"na_values": [2**63]},
            [str(2**63), str(2**63 + 1)],
        ),
        (str(2**63) + ",1" + "\n,2", {}, [[str(2**63), 1], ["", 2]]),
        (str(2**63) + "\n1", {"na_values": [2**63]}, [np.nan, 1]),
    ],
)
def test_na_values_uint64(all_parsers, data, kwargs, expected, request):
    # [FUNCTION DOC]
    # Verifies handling of numbers at two to the 63rd: as a missing value (given as an
    # int) it is converted to NaN where it matches, and without custom values a mix of
    # such strings and empties are kept as object strings. For pyarrow, the custom value
    # cases expect a type error requiring strings and the other case is marked expected
    # to fail (float64 result).
    # see gh-14983
    parser = all_parsers

    if parser.engine == "pyarrow" and "na_values" in kwargs:
        msg = "The 'pyarrow' engine requires all na_values to be strings"
        with pytest.raises(TypeError, match=msg):
            parser.read_csv(StringIO(data), header=None, **kwargs)
        return
    elif parser.engine == "pyarrow":
        mark = pytest.mark.xfail(reason="Returns float64 instead of object")
        request.applymarker(mark)

    result = parser.read_csv(StringIO(data), header=None, **kwargs)
    expected = DataFrame(expected)
    tm.assert_frame_equal(result, expected)


def test_empty_na_values_no_default_with_index(all_parsers):
    # [FUNCTION DOC]
    # Verifies reading with defaults turned off and the first column as index works for
    # a two-row input, giving a named string index and integer column.
    # see gh-15835
    data = "a,1\nb,2"
    parser = all_parsers
    expected = DataFrame({"1": [2]}, index=Index(["b"], name="a"))

    result = parser.read_csv(StringIO(data), index_col=0, keep_default_na=False)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "na_filter,index_data", [(False, ["", "5"]), (True, [np.nan, 5.0])]
)
def test_no_na_filter_on_index(all_parsers, na_filter, index_data, request):
    # [FUNCTION DOC]
    # Verifies the index column converts empty values to NaN (float index) only when the
    # missing-value filter is on; with it off the index keeps the empty string. For
    # pyarrow with the filter off the test is marked expected to fail.
    # see gh-5239
    #
    # Don't parse NA-values in index unless na_filter=True
    parser = all_parsers
    data = "a,b,c\n1,,3\n4,5,6"

    if parser.engine == "pyarrow" and na_filter is False:
        mark = pytest.mark.xfail(reason="mismatched index result")
        request.applymarker(mark)

    expected = DataFrame({"a": [1, 4], "c": [3, 6]}, index=Index(index_data, name="b"))
    result = parser.read_csv(StringIO(data), index_col=[1], na_filter=na_filter)
    tm.assert_frame_equal(result, expected)


def test_inf_na_values_with_int_index(all_parsers):
    # [FUNCTION DOC]
    # Verifies that infinities declared as missing values do not cause an overflow error
    # when the index is an integer column (regression), producing NaN in the data
    # columns.
    # see gh-17128
    parser = all_parsers
    data = "idx,col1,col2\n1,3,4\n2,inf,-inf"

    # Don't fail with OverflowError with inf's and integer index column.
    out = parser.read_csv(StringIO(data), index_col=[0], na_values=["inf", "-inf"])
    expected = DataFrame(
        {"col1": [3, np.nan], "col2": [4, np.nan]}, index=Index([1, 2], name="idx")
    )
    tm.assert_frame_equal(out, expected)


@pytest.mark.parametrize("na_filter", [True, False])
def test_na_values_with_dtype_str_and_na_filter(
    all_parsers, na_filter, using_infer_string, request
):
    # [FUNCTION DOC]
    # Verifies reading with string dtype for all columns converts an empty field to NaN
    # with the filter on and keeps the empty string with the filter off. For pyarrow
    # most cases are marked expected to fail (mismatched shapes).
    # see gh-20377
    parser = all_parsers
    if parser.engine == "pyarrow" and (na_filter is False or not using_infer_string):
        mark = pytest.mark.xfail(reason="mismatched shape")
        request.applymarker(mark)

    data = "a,b,c\n1,,3\n4,5,6"

    # na_filter=True --> missing value becomes NaN.
    # na_filter=False --> missing value remains empty string.
    empty = np.nan if na_filter else ""
    expected = DataFrame({"a": ["1", "4"], "b": [empty, "5"], "c": ["3", "6"]})

    result = parser.read_csv(StringIO(data), na_filter=na_filter, dtype=str)
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # mismatched exception message
@pytest.mark.parametrize(
    "data, na_values",
    [
        ("false,1\n,1\ntrue", None),
        ("false,1\nnull,1\ntrue", None),
        ("false,1\nnan,1\ntrue", None),
        ("false,1\nfoo,1\ntrue", "foo"),
        ("false,1\nfoo,1\ntrue", ["foo"]),
        ("false,1\nfoo,1\ntrue", {"a": "foo"}),
    ],
)
def test_cast_NA_to_bool_raises_error(all_parsers, data, na_values):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow (message mismatch). Verifies that requesting
    # boolean dtype for a column that has missing values (empty, null, nan or a custom
    # marker given as scalar, list or dictionary) raises a value error about NA values
    # in a boolean column.
    parser = all_parsers
    msg = "|".join(
        [
            "Bool column has NA values in column [0a]",
            "cannot safely convert passed user dtype of "
            "bool for object dtyped data in column 0",
        ]
    )

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(
            StringIO(data),
            header=None,
            names=["a", "b"],
            dtype={"a": "bool"},
            na_values=na_values,
        )


# TODO: this test isn't about the na_values keyword, it is about the empty entries
#  being returned with NaN entries, whereas the pyarrow engine returns "nan"
@xfail_pyarrow  # mismatched shapes
def test_str_nan_dropped(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow (shape mismatch). Verifies that with string
    # dtype for all columns, rows whose empty fields are read as missing are removed by
    # dropping missing rows, leaving only complete rows with leading zeros preserved.
    # KNOWN ISSUE: a comment above the test says it is not really about the
    # missing-values option but about empty entries being returned as NaN here and as
    # the text nan by pyarrow.
    # see gh-21131
    parser = all_parsers

    data = """File: small.csv,,
10010010233,0123,654
foo,,bar
01001000155,4530,898"""

    result = parser.read_csv(
        StringIO(data),
        header=None,
        names=["col1", "col2", "col3"],
        dtype={"col1": str, "col2": str, "col3": str},
    ).dropna()

    expected = DataFrame(
        {
            "col1": ["10010010233", "01001000155"],
            "col2": ["0123", "4530"],
            "col3": ["654", "898"],
        },
        index=[1, 3],
    )

    tm.assert_frame_equal(result, expected)


def test_nan_multi_index(all_parsers):
    # [FUNCTION DOC]
    # Verifies a dictionary of missing values keyed by a tuple (two-level header) marks
    # the matching value as NaN in that column. For pyarrow, verifies the
    # unsupported-dictionary error.
    # GH 42446
    parser = all_parsers
    data = "A,B,B\nX,Y,Z\n1,2,inf"

    if parser.engine == "pyarrow":
        msg = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data), header=list(range(2)), na_values={("B", "Z"): "inf"}
            )
        return

    result = parser.read_csv(
        StringIO(data), header=list(range(2)), na_values={("B", "Z"): "inf"}
    )

    expected = DataFrame(
        {
            ("A", "X"): [1],
            ("B", "Y"): [2],
            ("B", "Z"): [np.nan],
        }
    )

    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # Failed: DID NOT RAISE <class 'ValueError'>; it casts the NaN to False
def test_bool_and_nan_to_bool(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow (it casts the NaN to false instead of
    # raising). Verifies a column with NaN, True and False requested as boolean raises a
    # value error mentioning NA values.
    # GH#42808
    parser = all_parsers
    data = """0
NaN
True
False
"""
    with pytest.raises(ValueError, match="NA values"):
        parser.read_csv(StringIO(data), dtype="bool")


def test_bool_and_nan_to_int(all_parsers):
    # [FUNCTION DOC]
    # Verifies the same column requested as integer raises a value error about NA values
    # (messages vary by engine: C engine safe-conversion message, python engine
    # unable-to-convert message, pyarrow integer-conversion message).
    # GH#42808
    parser = all_parsers
    data = """0
NaN
True
False
"""
    msg = (
        "cannot safely convert passed user dtype of int(64|32) for "
        "<class 'numpy.bool_?'> dtyped data in column 0 due to NA values"
    )
    if parser.engine == "python":
        msg = "Unable to convert column 0 to type int(64|32)"
    elif parser.engine == "pyarrow":
        msg = (
            r"int\(\) argument must be a string, a bytes-like object or a "
            "real number, not 'NoneType"
        )
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), dtype="int")


def test_bool_and_nan_to_float(all_parsers):
    # [FUNCTION DOC]
    # Verifies the same column requested as float is read as NaN, 1.0 and 0.0.
    # GH#42808
    parser = all_parsers
    data = """0
NaN
True
False
"""
    result = parser.read_csv(StringIO(data), dtype="float")
    expected = DataFrame.from_dict({"0": [np.nan, 1.0, 0.0]})
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
@pytest.mark.parametrize(
    "na_values",
    [[-99.0, -99], [-99, -99.0]],
)
def test_na_values_dict_without_dtype(all_parsers, na_values):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies that missing values given as a list
    # of both -99 as float and as int (in either order) mark all four entries (integer
    # and float spellings) as NaN without specifying a dtype.
    # KNOWN ISSUE: the name says dict, but the parametrized missing values are lists,
    # not dictionaries.
    parser = all_parsers
    data = """A
-99
-99
-99.0
-99.0"""

    result = parser.read_csv(StringIO(data), na_values=na_values)
    expected = DataFrame({"A": [np.nan, np.nan, np.nan, np.nan]})
    tm.assert_frame_equal(result, expected)
