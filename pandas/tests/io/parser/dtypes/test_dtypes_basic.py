"""
Tests dtype specification during parsing
for all of the parsers defined in parsers.py
"""

# [MODULE DOC]
# Parser tests of dtype specification during CSV reading: a single dtype for all
# columns, per-column dtypes (by name or position), invalid dtype names, nullable and
# boolean dtypes, converters versus dtypes, decimal and exponent handling with thousands
# separators, duplicated column names, multi-level headers, mapping-with-default dtypes,
# dtype backends (nullable and pyarrow), string inference, large integers, and dtypes
# with column selection or index columns.
# KNOWN ISSUE: the module docstring says the parsers are defined in a file named
# parsers.py, but no such module exists here; the parser fixtures come from the
# package's shared fixtures module.
# Many tests use the pyarrow expected-failure marker, applied either through the alias
# or the fixture name.

from collections import defaultdict
from io import StringIO

import numpy as np
import pytest

from pandas.errors import ParserWarning

import pandas as pd
from pandas import (
    DataFrame,
    Timestamp,
)
import pandas._testing as tm
from pandas.core.arrays import IntegerArray

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")


@pytest.mark.parametrize("dtype", [str, object])
@pytest.mark.parametrize("check_orig", [True, False])
@pytest.mark.usefixtures("pyarrow_xfail")
def test_dtype_all_columns(
    all_parsers, dtype, check_orig, using_infer_string, temp_file
):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Writes a random 5x2 float frame with string
    # index to a temporary CSV, reads it with a single dtype of string or object and the
    # first column as index, and verifies: with the original check the result converted
    # back to float equals the original; otherwise the result equals the original cast
    # to strings (kept as the string dtype when string inference is on and the dtype was
    # str, otherwise object).
    # Parametrized over dtype and the choice of check.
    # see gh-3795, gh-6607
    parser = all_parsers

    df = DataFrame(
        np.random.default_rng(2).random((5, 2)).round(4),
        columns=list("AB"),
        index=["1A", "1B", "1C", "1D", "1E"],
    )

    df.to_csv(temp_file)

    result = parser.read_csv(temp_file, dtype=dtype, index_col=0)

    if check_orig:
        expected = df.copy()
        result = result.astype(float)
    elif using_infer_string and dtype is str:
        expected = df.astype(str)
    else:
        expected = df.astype(str).astype(object)

    tm.assert_frame_equal(result, expected)


@pytest.mark.usefixtures("pyarrow_xfail")
def test_dtype_per_column(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies a dtype mapping with a float for one
    # column given by name and string for the other given by position converts the first
    # to floats and keeps the second as strings.
    parser = all_parsers
    data = """\
one,two
1,2.5
2,3.5
3,4.5
4,5.5"""
    expected = DataFrame(
        [[1, "2.5"], [2, "3.5"], [3, "4.5"], [4, "5.5"]], columns=["one", "two"]
    )
    expected["one"] = expected["one"].astype(np.float64)

    result = parser.read_csv(StringIO(data), dtype={"one": np.float64, 1: str})
    tm.assert_frame_equal(result, expected)


def test_invalid_dtype_per_column(all_parsers):
    # [FUNCTION DOC]
    # Verifies an unknown dtype name in a mapping raises a type error saying the data
    # type was not understood.
    parser = all_parsers
    data = """\
one,two
1,2.5
2,3.5
3,4.5
4,5.5"""

    with pytest.raises(TypeError, match="data type [\"']foo[\"'] not understood"):
        parser.read_csv(StringIO(data), dtype={"one": "foo", 1: "int"})


def test_raise_on_passed_int_dtype_with_nas(all_parsers):
    # [FUNCTION DOC]
    # Verifies that requesting an integer dtype for a column that has missing values
    # raises a value error (C engine: integer column has NA values; python engine:
    # unable to convert the column; pyarrow: the unsupported skip-initial-space option
    # error), with skip-initial-space turned on.
    # see gh-2631
    parser = all_parsers
    data = """YEAR, DOY, a
2001,106380451,10
2001,,11
2001,106380451,67"""

    if parser.engine == "c":
        msg = "Integer column has NA values"
    elif parser.engine == "pyarrow":
        msg = "The 'skipinitialspace' option is not supported with the 'pyarrow' engine"
    else:
        msg = "Unable to convert column DOY"

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), dtype={"DOY": np.int64}, skipinitialspace=True)


def test_dtype_with_converters(all_parsers):
    # [FUNCTION DOC]
    # Verifies that when both a converter and a dtype are given for a column, a parser
    # warning says only the converter will be used and the column holds converter output
    # (strings). For pyarrow, verifies the unsupported converters option error.
    parser = all_parsers
    data = """a,b
1.1,2.2
1.2,2.3"""

    if parser.engine == "pyarrow":
        msg = "The 'converters' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data), dtype={"a": "i8"}, converters={"a": lambda x: str(x)}
            )
        return

    # Dtype spec ignored if converted specified.
    result = parser.read_csv_check_warnings(
        ParserWarning,
        "Both a converter and dtype were specified for column a "
        "- only the converter will be used.",
        StringIO(data),
        dtype={"a": "i8"},
        converters={"a": lambda x: str(x)},
    )
    expected = DataFrame({"a": ["1.1", "1.2"], "b": [2.2, 2.3]})
    tm.assert_frame_equal(result, expected)


def test_numeric_dtype(all_parsers, any_real_numpy_dtype):
    # [FUNCTION DOC]
    # Verifies headerless two-row input read with each real numpy dtype yields a
    # one-column frame with that dtype.
    data = "0\n1"
    parser = all_parsers
    expected = DataFrame([0, 1], dtype=any_real_numpy_dtype)

    result = parser.read_csv(StringIO(data), header=None, dtype=any_real_numpy_dtype)
    tm.assert_frame_equal(expected, result, check_column_type=False)


@pytest.mark.usefixtures("pyarrow_xfail")
def test_boolean_dtype(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies the nullable boolean dtype maps
    # true-like spellings and the numbers one and one-point-zero to true, false-like
    # spellings and zero to false, and NaN, nan, NA, null and NULL to missing.
    parser = all_parsers
    data = "\n".join(
        [
            "a",
            "True",
            "TRUE",
            "true",
            "1",
            "1.0",
            "False",
            "FALSE",
            "false",
            "0",
            "0.0",
            "NaN",
            "nan",
            "NA",
            "null",
            "NULL",
        ]
    )

    result = parser.read_csv(StringIO(data), dtype="boolean")
    expected = DataFrame(
        {
            "a": pd.array(
                [
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
                dtype="boolean",
            )
        }
    )

    tm.assert_frame_equal(result, expected)


@pytest.mark.usefixtures("pyarrow_xfail")
def test_delimiter_with_usecols_and_parse_dates(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Reads a quoted line with comma decimal
    # marker, three selected columns by name and one date column to parse, and verifies
    # the numbers are floats and the date is a timestamp (regression for the python
    # engine with quoted decimals).
    # KNOWN ISSUE: the call passes an explicit python engine argument, but the parser
    # fixture's wrapper always overwrites the engine with its own, so the argument has
    # no effect.
    # GH#35873
    result = all_parsers.read_csv(
        StringIO('"dump","-9,1","-9,1",20101010'),
        engine="python",
        names=["col", "col1", "col2", "col3"],
        usecols=["col1", "col2", "col3"],
        parse_dates=["col3"],
        decimal=",",
    )
    expected = DataFrame(
        {"col1": [-9.1], "col2": [-9.1], "col3": [Timestamp("2010-10-10")]}
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("thousands", ["_", None])
def test_decimal_and_exponential(
    request, python_parser_only, numeric_decimal, thousands
):
    # [FUNCTION DOC]
    # Python-engine-only test, parametrized on the thousands separator (underscore or
    # none): runs the shared decimal check helper with no float precision for the
    # current numeric-decimal fixture value.
    # GH#31920
    decimal_number_check(request, python_parser_only, numeric_decimal, thousands, None)


@pytest.mark.parametrize("thousands", ["_", None])
@pytest.mark.parametrize("float_precision", [None, "legacy", "high", "round_trip"])
def test_1000_sep_decimal_float_precision(
    request, c_parser_only, numeric_decimal, float_precision, thousands
):
    # [FUNCTION DOC]
    # C-engine-only test, parametrized on the thousands separator and the float
    # precision (default, legacy, high, round trip): runs the decimal check helper on
    # the numeric-decimal fixture value, and again on the same value with a space added
    # at both ends (for string results the expectation also gets the spaces).
    # test decimal and thousand sep handling in across 'float_precision'
    # parsers
    decimal_number_check(
        request, c_parser_only, numeric_decimal, thousands, float_precision
    )
    text, value = numeric_decimal
    text = " " + text + " "
    if isinstance(value, str):  # the negative cases (parse as text)
        value = " " + value + " "
    decimal_number_check(
        request, c_parser_only, (text, value), thousands, float_precision
    )


def decimal_number_check(request, parser, numeric_decimal, thousands, float_precision):
    # [FUNCTION DOC]
    # HELPER: shared check for the decimal tests. Parameters: the pytest request (to
    # apply marks), a parser wrapper, a pair of input text and expected value, the
    # thousands separator and the float precision. For a few inputs that contain an
    # underscore comma pattern, applies an expected-failure mark when no thousands
    # separator is given. Reads the text with pipe separator, comma decimal marker, no
    # header and the given options, and asserts that the first cell equals the expected
    # value.
    # GH#31920
    value = numeric_decimal[0]
    if thousands is None and value in ("1_,", "1_234,56", "1_234,56e0"):
        request.applymarker(
            pytest.mark.xfail(reason=f"thousands={thousands} and sep is in {value}")
        )
    df = parser.read_csv(
        StringIO(value),
        float_precision=float_precision,
        sep="|",
        thousands=thousands,
        decimal=",",
        header=None,
    )
    val = df.iloc[0, 0]
    assert val == numeric_decimal[1]


@pytest.mark.parametrize("float_precision", [None, "legacy", "high", "round_trip"])
def test_skip_whitespace(c_parser_only, float_precision):
    # [FUNCTION DOC]
    # C-engine-only test over float precision settings: verifies a tab-separated file
    # with spaces around numbers and a trailing tab is read with the second column as
    # float64 with the whitespace stripped.
    DATA = """id\tnum\t
1\t1.2 \t
1\t 2.1\t
2\t 1\t
2\t 1.2 \t
"""
    df = c_parser_only.read_csv(
        StringIO(DATA),
        float_precision=float_precision,
        sep="\t",
        header=0,
        dtype={1: np.float64},
    )
    tm.assert_series_equal(df.iloc[:, 1], pd.Series([1.2, 2.1, 1.0, 1.2], name="num"))


@pytest.mark.usefixtures("pyarrow_xfail")
def test_true_values_cast_to_bool(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies that with yes/no given as true/false
    # values and nullable boolean dtype for a column, the values yes, no, one and zero
    # are converted to true, false, true and false.
    # GH#34655
    text = """a,b
yes,xxx
no,yyy
1,zzz
0,aaa
    """
    parser = all_parsers
    result = parser.read_csv(
        StringIO(text),
        true_values=["yes"],
        false_values=["no"],
        dtype={"a": "boolean"},
    )
    expected = DataFrame(
        {"a": [True, False, True, False], "b": ["xxx", "yyy", "zzz", "aaa"]}
    )
    expected["a"] = expected["a"].astype("boolean")
    tm.assert_frame_equal(result, expected)


@pytest.mark.usefixtures("pyarrow_xfail")
@pytest.mark.parametrize("dtypes, exp_value", [({}, "1"), ({"a.1": "int64"}, 1)])
def test_dtype_mangle_dup_cols(all_parsers, dtypes, exp_value):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies that with duplicated header names, a
    # dtype mapping on the original name applies to the first column and a mapping for
    # the mangled name (second column with suffix) applies to it, otherwise it takes the
    # value as a string; also checks the passed mapping is not mutated by the reader.
    # GH#35211
    parser = all_parsers
    data = """a,a\n1,1"""
    dtype_dict = {"a": str, **dtypes}
    # GH#42462
    dtype_dict_copy = dtype_dict.copy()
    result = parser.read_csv(StringIO(data), dtype=dtype_dict)
    expected = DataFrame({"a": ["1"], "a.1": [exp_value]})
    assert dtype_dict == dtype_dict_copy, "dtype dict changed"
    tm.assert_frame_equal(result, expected)


@pytest.mark.usefixtures("pyarrow_xfail")
def test_dtype_mangle_dup_cols_single_dtype(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies a single string dtype applies to
    # both columns of a duplicated header, with the second named with a suffix.
    # GH#42022
    parser = all_parsers
    data = """a,a\n1,1"""
    result = parser.read_csv(StringIO(data), dtype=str)
    expected = DataFrame({"a": ["1"], "a.1": ["1"]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.usefixtures("pyarrow_xfail")
def test_dtype_multi_index(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies a two-row header (multi-level
    # columns) with a dtype mapping keyed by tuples gives integer and float columns of
    # the requested widths.
    # GH 42446
    parser = all_parsers
    data = "A,B,B\nX,Y,Z\n1,2,3"

    result = parser.read_csv(
        StringIO(data),
        header=list(range(2)),
        dtype={
            ("A", "X"): np.int32,
            ("B", "Y"): np.int32,
            ("B", "Z"): np.float32,
        },
    )

    expected = DataFrame(
        {
            ("A", "X"): np.int32([1]),
            ("B", "Y"): np.int32([2]),
            ("B", "Z"): np.float32([3]),
        }
    )

    tm.assert_frame_equal(result, expected)


def test_nullable_int_dtype(all_parsers, any_int_ea_dtype):
    # [FUNCTION DOC]
    # Verifies each nullable integer dtype applied to all columns converts empty fields
    # to missing values.
    # GH 25472
    parser = all_parsers
    dtype = any_int_ea_dtype

    data = """a,b,c
,3,5
1,,6
2,4,"""
    expected = DataFrame(
        {
            "a": pd.array([pd.NA, 1, 2], dtype=dtype),
            "b": pd.array([3, pd.NA, 4], dtype=dtype),
            "c": pd.array([5, 6, pd.NA], dtype=dtype),
        }
    )
    actual = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(actual, expected)


@pytest.mark.usefixtures("pyarrow_xfail")
@pytest.mark.parametrize("default", ["float", "float64"])
def test_dtypes_defaultdict(all_parsers, default):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies a dictionary with default (via a
    # default-producing mapping) applies the default dtype to unlisted columns and the
    # explicit dtype to the listed column.
    # GH#41574
    data = """a,b
1,2
"""
    dtype = defaultdict(lambda: default, a="int64")
    parser = all_parsers
    result = parser.read_csv(StringIO(data), dtype=dtype)
    expected = DataFrame({"a": [1], "b": 2.0})
    tm.assert_frame_equal(result, expected)


@pytest.mark.usefixtures("pyarrow_xfail")
def test_dtypes_defaultdict_mangle_dup_cols(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies a mapping with a default dtype is
    # applied correctly across duplicated header names, including a name that already
    # looks like a mangled one.
    # GH#41574
    data = """a,b,a,b,b.1
1,2,3,4,5
"""
    dtype = defaultdict(lambda: "float64", a="int64")
    dtype["b.1"] = "int64"
    parser = all_parsers
    result = parser.read_csv(StringIO(data), dtype=dtype)
    expected = DataFrame({"a": [1], "b": [2.0], "a.1": [3], "b.2": [4.0], "b.1": [5]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.usefixtures("pyarrow_xfail")
def test_dtypes_defaultdict_invalid(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. Verifies a default dtype that is invalid
    # raises a type error with a message saying it is not understood.
    # GH#41574
    data = """a,b
1,2
"""
    dtype = defaultdict(lambda: "invalid_dtype", a="int64")
    parser = all_parsers
    with pytest.raises(TypeError, match="not understood"):
        parser.read_csv(StringIO(data), dtype=dtype)


def test_dtype_backend(all_parsers):
    # [FUNCTION DOC]
    # Verifies that the nullable-numpy backend gives nullable integer, float, boolean
    # and string dtypes for ten columns, including columns with missing values and an
    # all-missing column (nullable integer), while the parsed date column is a datetime.
    # GH#36712

    parser = all_parsers

    data = """a,b,c,d,e,f,g,h,i,j
1,2.5,True,a,,,,,12-31-2019,
3,4.5,False,b,6,7.5,True,a,12-31-2019,
"""
    result = parser.read_csv(
        StringIO(data), dtype_backend="numpy_nullable", parse_dates=["i"]
    )
    expected = DataFrame(
        {
            "a": pd.Series([1, 3], dtype="Int64"),
            "b": pd.Series([2.5, 4.5], dtype="Float64"),
            "c": pd.Series([True, False], dtype="boolean"),
            "d": pd.Series(["a", "b"], dtype="string"),
            "e": pd.Series([pd.NA, 6], dtype="Int64"),
            "f": pd.Series([pd.NA, 7.5], dtype="Float64"),
            "g": pd.Series([pd.NA, True], dtype="boolean"),
            "h": pd.Series([pd.NA, "a"], dtype="string"),
            "i": pd.Series([Timestamp("2019-12-31")] * 2),
            "j": pd.Series([pd.NA, pd.NA], dtype="Int64"),
        }
    )
    tm.assert_frame_equal(result, expected)


def test_dtype_backend_and_dtype(all_parsers):
    # [FUNCTION DOC]
    # Verifies an explicit float64 dtype takes precedence over the nullable backend,
    # giving NaN for missing values.
    # GH#36712

    parser = all_parsers

    data = """a,b
1,2.5
,
"""
    result = parser.read_csv(
        StringIO(data), dtype_backend="numpy_nullable", dtype="float64"
    )
    expected = DataFrame({"a": [1.0, np.nan], "b": [2.5, np.nan]})
    tm.assert_frame_equal(result, expected)


def test_dtype_backend_string(all_parsers, string_storage):
    # [FUNCTION DOC]
    # Verifies with the nullable backend under each string storage option that string
    # columns have the string dtype with that storage, with missing values as NA.
    # GH#36712
    with pd.option_context("mode.string_storage", string_storage):
        parser = all_parsers

        data = """a,b
a,x
b,
"""
        result = parser.read_csv(StringIO(data), dtype_backend="numpy_nullable")

        expected = DataFrame(
            {
                "a": pd.array(["a", "b"], dtype=pd.StringDtype(string_storage)),
                "b": pd.array(["x", pd.NA], dtype=pd.StringDtype(string_storage)),
            },
        )
    tm.assert_frame_equal(result, expected)


def test_dtype_backend_ea_dtype_specified(all_parsers):
    # [FUNCTION DOC]
    # Verifies an explicit extension integer dtype for all columns together with the
    # nullable backend yields nullable integer columns.
    # GH#491496
    data = """a,b
1,2
"""
    parser = all_parsers
    result = parser.read_csv(
        StringIO(data), dtype="Int64", dtype_backend="numpy_nullable"
    )
    expected = DataFrame({"a": [1], "b": 2}, dtype="Int64")
    tm.assert_frame_equal(result, expected)


def test_dtype_backend_pyarrow(all_parsers, request):
    # [FUNCTION DOC]
    # Skipped if pyarrow is not installed. Verifies the pyarrow dtype backend gives
    # arrow-backed integer, float, boolean, string, null and missing-value columns, with
    # the date column parsed to a normal datetime.
    # GH#36712
    pa = pytest.importorskip("pyarrow")
    parser = all_parsers

    data = """a,b,c,d,e,f,g,h,i,j
1,2.5,True,a,,,,,12-31-2019,
3,4.5,False,b,6,7.5,True,a,12-31-2019,
"""
    result = parser.read_csv(StringIO(data), dtype_backend="pyarrow", parse_dates=["i"])
    expected = DataFrame(
        {
            "a": pd.Series([1, 3], dtype="int64[pyarrow]"),
            "b": pd.Series([2.5, 4.5], dtype="float64[pyarrow]"),
            "c": pd.Series([True, False], dtype="bool[pyarrow]"),
            "d": pd.Series(["a", "b"], dtype=pd.ArrowDtype(pa.string())),
            "e": pd.Series([pd.NA, 6], dtype="int64[pyarrow]"),
            "f": pd.Series([pd.NA, 7.5], dtype="float64[pyarrow]"),
            "g": pd.Series([pd.NA, True], dtype="bool[pyarrow]"),
            "h": pd.Series(
                [pd.NA, "a"],
                dtype=pd.ArrowDtype(pa.string()),
            ),
            "i": pd.Series([Timestamp("2019-12-31")] * 2),
            "j": pd.Series([pd.NA, pd.NA], dtype="null[pyarrow]"),
        }
    )
    tm.assert_frame_equal(result, expected)


def test_ea_int_avoid_overflow(all_parsers):
    # [FUNCTION DOC]
    # Verifies a large 64-bit value in a column read as nullable integer is preserved
    # exactly and the empty field becomes missing (regression for overflow through
    # floats).
    # GH#32134
    parser = all_parsers
    data = """a,b
1,1
,1
1582218195625938945,1
"""
    result = parser.read_csv(StringIO(data), dtype={"a": "Int64"})
    expected = DataFrame(
        {
            "a": IntegerArray(
                np.array([1, 1, 1582218195625938945]), np.array([False, True, False])
            ),
            "b": 1,
        }
    )
    tm.assert_frame_equal(result, expected)


def test_string_inference(all_parsers):
    # [FUNCTION DOC]
    # Verifies with the string-inference option on that string columns and the column
    # labels get the NaN-based string dtype and missing values become missing.
    # GH#54430
    dtype = pd.StringDtype(na_value=np.nan)

    data = """a,b
x,1
y,2
,3"""
    parser = all_parsers
    with pd.option_context("future.infer_string", True):
        result = parser.read_csv(StringIO(data))

    expected = DataFrame(
        {"a": pd.Series(["x", "y", None], dtype=dtype), "b": [1, 2, 3]},
        columns=pd.Index(["a", "b"], dtype=dtype),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", ["O", object, "object", np.object_, str, np.str_])
def test_string_inference_object_dtype(all_parsers, dtype, using_infer_string):
    # [FUNCTION DOC]
    # Verifies with string inference on that requesting object or string dtype (several
    # spellings) for all columns or a single column gives object columns, or the
    # NaN-based string dtype when the requested type is the built-in string type;
    # unspecified columns and the labels use the inferred string dtype.
    # GH#56047
    data = """a,b
x,a
y,a
z,a"""
    parser = all_parsers
    with pd.option_context("future.infer_string", True):
        result = parser.read_csv(StringIO(data), dtype=dtype)

    expected_dtype = pd.StringDtype(na_value=np.nan) if dtype is str else object
    expected = DataFrame(
        {
            "a": pd.Series(["x", "y", "z"], dtype=expected_dtype),
            "b": pd.Series(["a", "a", "a"], dtype=expected_dtype),
        },
        columns=pd.Index(["a", "b"], dtype=pd.StringDtype(na_value=np.nan)),
    )
    tm.assert_frame_equal(result, expected)

    with pd.option_context("future.infer_string", True):
        result = parser.read_csv(StringIO(data), dtype={"a": dtype})

    expected = DataFrame(
        {
            "a": pd.Series(["x", "y", "z"], dtype=expected_dtype),
            "b": pd.Series(["a", "a", "a"], dtype=pd.StringDtype(na_value=np.nan)),
        },
        columns=pd.Index(["a", "b"], dtype=pd.StringDtype(na_value=np.nan)),
    )
    tm.assert_frame_equal(result, expected)


def test_accurate_parsing_of_large_integers(all_parsers):
    # [FUNCTION DOC]
    # Verifies that identifier values near the limit of exact floating-point integers,
    # read with nullable integer dtype for one column, remain distinguishable: counts of
    # rows equal to four specific values are one, one, two and two.
    # GH#52505
    data = """SYMBOL,MOMENT,ID,ID_DEAL
AAPL,20230301181139587,1925036343869802844,
AAPL,20230301181139587,2023552585717889863,2023552585717263358
NVDA,20230301181139587,2023552585717889863,2023552585717263359
AMC,20230301181139587,2023552585717889863,2023552585717263360
AMZN,20230301181139587,2023552585717889759,2023552585717263360
MSFT,20230301181139587,2023552585717889863,2023552585717263361
NVDA,20230301181139587,2023552585717889827,2023552585717263361"""
    orders = all_parsers.read_csv(StringIO(data), dtype={"ID_DEAL": pd.Int64Dtype()})
    assert len(orders.loc[orders["ID_DEAL"] == 2023552585717263358, "ID_DEAL"]) == 1
    assert len(orders.loc[orders["ID_DEAL"] == 2023552585717263359, "ID_DEAL"]) == 1
    assert len(orders.loc[orders["ID_DEAL"] == 2023552585717263360, "ID_DEAL"]) == 2
    assert len(orders.loc[orders["ID_DEAL"] == 2023552585717263361, "ID_DEAL"]) == 2


def test_dtypes_with_usecols(all_parsers):
    # [FUNCTION DOC]
    # Verifies a dtype for a selected column is applied alongside a column selection:
    # with object dtype the values are strings (for pyarrow they remain integers), and
    # the other selected column stays integer.
    # GH#54868

    parser = all_parsers
    data = """a,b,c
1,2,3
4,5,6"""

    result = parser.read_csv(StringIO(data), usecols=["a", "c"], dtype={"a": object})
    if parser.engine == "pyarrow":
        values = [1, 4]
    else:
        values = ["1", "4"]
    expected = DataFrame({"a": pd.Series(values, dtype=object), "c": [3, 6]})
    tm.assert_frame_equal(result, expected)


def test_index_col_with_dtype_no_rangeindex(all_parsers):
    # [FUNCTION DOC]
    # Verifies that when the index column is named and given an unsigned 32-bit dtype,
    # the resulting index has that dtype and name (not a range index).
    data = StringIO("345.5,519.5,0\n519.5,726.5,1")
    result = all_parsers.read_csv(
        data,
        header=None,
        names=["start", "stop", "bin_id"],
        dtype={"start": np.float32, "stop": np.float32, "bin_id": np.uint32},
        index_col="bin_id",
    ).index
    expected = pd.Index([0, 1], dtype=np.uint32, name="bin_id")
    tm.assert_index_equal(result, expected)
