"""
Tests date parsing functionality for all of the
parsers defined in parsers.py
"""
# [MODULE DOC]
# Tests for date parsing in read_csv (parse_dates, date_format, dayfirst, cache_dates, index_col interplay, time zones) across all parsers, with pyarrow differences marked via xfail/skip markers or inline conditionals.
# Uses the start_caching_at threshold from the datetime tools module to build inputs large enough to trigger the date cache. Module-level filter ignores the BlockManager deprecation warning.

from datetime import (
    datetime,
    timedelta,
    timezone,
)
from io import StringIO

import numpy as np
import pytest

from pandas.errors import Pandas4Warning

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    Timestamp,
)
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
from pandas.core.tools.datetimes import start_caching_at

from pandas.io.parsers import read_csv

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")


def test_date_col_as_index_col(all_parsers):
    # [FUNCTION DOC]
    # Headerless data with a date-time column parsed as dates and used as the index (with named columns); asserts a microsecond datetime index named X1 and remaining columns. For pyarrow, the time-only column X2 is expected as time objects (pyarrow infers time type, issue 44231).
    data = """\
KORD,19990127 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000
KORD,19990127 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000
KORD,19990127 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000
KORD,19990127 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000
KORD,19990127 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000
"""
    parser = all_parsers
    kwds = {
        "header": None,
        "parse_dates": [1],
        "index_col": 1,
        "names": ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X7"],
    }
    result = parser.read_csv(StringIO(data), **kwds)

    index = Index(
        [
            datetime(1999, 1, 27, 19, 0),
            datetime(1999, 1, 27, 20, 0),
            datetime(1999, 1, 27, 21, 0),
            datetime(1999, 1, 27, 21, 0),
            datetime(1999, 1, 27, 22, 0),
        ],
        dtype="M8[us]",
        name="X1",
    )
    expected = DataFrame(
        [
            ["KORD", " 18:56:00", 0.81, 2.81, 7.2, 0.0, 280.0],
            ["KORD", " 19:56:00", 0.01, 2.21, 7.2, 0.0, 260.0],
            ["KORD", " 20:56:00", -0.59, 2.21, 5.7, 0.0, 280.0],
            ["KORD", " 21:18:00", -0.99, 2.01, 3.6, 0.0, 270.0],
            ["KORD", " 21:56:00", -0.59, 1.71, 5.1, 0.0, 290.0],
        ],
        columns=["X0", "X2", "X3", "X4", "X5", "X6", "X7"],
        index=index,
    )
    if parser.engine == "pyarrow":
        # https://github.com/pandas-dev/pandas/issues/44231
        # pyarrow 6.0 starts to infer time type
        expected["X2"] = pd.to_datetime("1970-01-01" + expected["X2"]).dt.time

    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
def test_nat_parse(all_parsers, temp_file):
    # [FUNCTION DOC]
    # See gh-3062. Round-trips a frame containing missing numeric and timestamp values through a csv file and back with date parsing for column B; asserts equality with the original. Expected to fail on pyarrow.
    # see gh-3062
    parser = all_parsers
    df = DataFrame(
        {
            "A": np.arange(10, dtype="float64"),
            "B": Timestamp("20010101"),
        }
    )
    df.iloc[3:6, :] = np.nan

    path = temp_file
    df.to_csv(path)

    result = parser.read_csv(path, index_col=0, parse_dates=["B"])
    tm.assert_frame_equal(result, df)


@skip_pyarrow
def test_parse_dates_implicit_first_col(all_parsers):
    # [FUNCTION DOC]
    # Header with fewer names than data fields makes the first column an implicit index; asserts that enabling parse_dates as true gives the same result as explicitly using index_col zero with parse_dates true. Skipped on pyarrow.
    data = """A,B,C
20090101,a,1,2
20090102,b,3,4
20090103,c,4,5
"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data), parse_dates=True)

    expected = parser.read_csv(StringIO(data), index_col=0, parse_dates=True)
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
def test_parse_dates_string(all_parsers):
    # [FUNCTION DOC]
    # Parses a date column named in parse_dates and also used as the index; asserts a microsecond datetime index named date without frequency (frequency does not round-trip). Expected to fail on pyarrow.
    data = """date,A,B,C
20090101,a,1,2
20090102,b,3,4
20090103,c,4,5
"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data), index_col="date", parse_dates=["date"])
    # freq doesn't round-trip
    index = date_range("1/1/2009", periods=3, name="date", unit="us")._with_freq(None)

    expected = DataFrame(
        {"A": ["a", "b", "c"], "B": [1, 3, 4], "C": [2, 4, 5]}, index=index
    )
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
@pytest.mark.parametrize("parse_dates", [[0, 2], ["a", "c"]])
def test_parse_dates_column_list(all_parsers, parse_dates):
    # [FUNCTION DOC]
    # Parametrized over column positions and column names. Two date columns (dayfirst day/month/year) with a two-level index_col; asserts a frame indexed by a and b with microsecond datetimes and column c a datetime. Expected to fail on pyarrow.
    data = "a,b,c\n01/01/2010,1,15/02/2010"
    parser = all_parsers

    expected = DataFrame(
        {"a": [datetime(2010, 1, 1)], "b": [1], "c": [datetime(2010, 2, 15)]}
    )
    expected["a"] = expected["a"].astype("M8[us]")
    expected["c"] = expected["c"].astype("M8[us]")
    expected = expected.set_index(["a", "b"])

    result = parser.read_csv(
        StringIO(data), index_col=[0, 1], parse_dates=parse_dates, dayfirst=True
    )
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
@pytest.mark.parametrize("index_col", [[0, 1], [1, 0]])
def test_multi_index_parse_dates(all_parsers, index_col):
    # [FUNCTION DOC]
    # Parametrized over index_col orderings (zero then one, and one then zero). With parse_dates true, asserts a multi-index of a date level and a string level (swapped for the reversed order) and that a user warning about being unable to infer the format is emitted. Expected to fail on pyarrow.
    data = """index1,index2,A,B,C
20090101,one,a,1,2
20090101,two,b,3,4
20090101,three,c,4,5
20090102,one,a,1,2
20090102,two,b,3,4
20090102,three,c,4,5
20090103,one,a,1,2
20090103,two,b,3,4
20090103,three,c,4,5
"""
    parser = all_parsers
    dti = date_range("2009-01-01", periods=3, freq="D", unit="us")
    index = MultiIndex.from_product(
        [
            dti,
            ("one", "two", "three"),
        ],
        names=["index1", "index2"],
    )

    # Out of order.
    if index_col == [1, 0]:
        index = index.swaplevel(0, 1)

    expected = DataFrame(
        [
            ["a", 1, 2],
            ["b", 3, 4],
            ["c", 4, 5],
            ["a", 1, 2],
            ["b", 3, 4],
            ["c", 4, 5],
            ["a", 1, 2],
            ["b", 3, 4],
            ["c", 4, 5],
        ],
        columns=["A", "B", "C"],
        index=index,
    )
    result = parser.read_csv_check_warnings(
        UserWarning,
        "Could not infer format",
        StringIO(data),
        index_col=index_col,
        parse_dates=True,
    )
    tm.assert_frame_equal(result, expected)


def test_parse_tz_aware(all_parsers):
    # [FUNCTION DOC]
    # See gh-1693. A UTC-suffixed timestamp parsed as the index yields a timezone-aware UTC timestamp index named Date (second unit for pyarrow).
    # See gh-1693
    parser = all_parsers
    data = "Date,x\n2012-06-13T01:39:00Z,0.5"

    result = parser.read_csv(StringIO(data), index_col=0, parse_dates=True)
    expected = DataFrame(
        {"x": [0.5]}, index=Index([Timestamp("2012-06-13 01:39:00+00:00")], name="Date")
    )
    if parser.engine == "pyarrow":
        expected.index = expected.index.as_unit("s")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("kwargs", [{}, {"index_col": "C"}])
def test_read_with_parse_dates_scalar_non_bool(all_parsers, kwargs):
    # [FUNCTION DOC]
    # See gh-5636. Parametrized over no extra keywords and an index column. A scalar string for parse_dates raises a type error saying only booleans and lists are accepted.
    # see gh-5636
    parser = all_parsers
    msg = "Only booleans and lists are accepted for the 'parse_dates' parameter"
    data = """A,B,C
    1,2,2003-11-1"""

    with pytest.raises(TypeError, match=msg):
        parser.read_csv(StringIO(data), parse_dates="C", **kwargs)


@pytest.mark.parametrize("parse_dates", [(1,), np.array([4, 5]), {1, 3}])
def test_read_with_parse_dates_invalid_type(all_parsers, parse_dates):
    # [FUNCTION DOC]
    # Parametrized over a tuple, numpy array and set for parse_dates; each raises a type error saying only booleans and lists are accepted.
    parser = all_parsers
    msg = "Only booleans and lists are accepted for the 'parse_dates' parameter"
    data = """A,B,C
    1,2,2003-11-1"""

    with pytest.raises(TypeError, match=msg):
        parser.read_csv(StringIO(data), parse_dates=parse_dates)


@pytest.mark.parametrize("value", ["nan", ""])
def test_bad_date_parse(all_parsers, cache, value):
    # [FUNCTION DOC]
    # Parametrized over invalid date values (the string nan and empty) and the cache_dates flag (fixture). Builds input with more rows than the caching threshold and parses the first column as dates; passes if no exception is raised (no assertions).
    # if we have an invalid date make sure that we handle this with
    # and w/o the cache properly
    parser = all_parsers
    s = StringIO((f"{value},\n") * (start_caching_at + 1))

    parser.read_csv(
        s,
        header=None,
        names=["foo", "bar"],
        parse_dates=["foo"],
        cache_dates=cache,
    )


def test_bad_date_parse_with_warning(all_parsers, cache):
    # [FUNCTION DOC]
    # Rows of the single value zero exceeding the cache threshold are parsed as dates with caching on or off. Expects a could-not-infer-format user warning only when the cache is off and the engine is not pyarrow (pyarrow reads integers; with the cache there is a single unique value so no inconsistent parsing risk). Extra warnings are tolerated.
    # if we have an invalid date make sure that we handle this with
    # and w/o the cache properly.
    parser = all_parsers
    s = StringIO(("0,\n") * (start_caching_at + 1))

    if parser.engine == "pyarrow":
        # pyarrow reads "0" as 0 (of type int64), and so
        # pandas doesn't try to guess the datetime format
        # TODO: parse dates directly in pyarrow, see
        # https://github.com/pandas-dev/pandas/issues/48017
        warn = None
    elif cache:
        # Note: warning is not raised if 'cache_dates', because here there is only a
        # single unique date and hence no risk of inconsistent parsing.
        warn = None
    else:
        warn = UserWarning
    parser.read_csv_check_warnings(
        warn,
        "Could not infer format",
        s,
        header=None,
        names=["foo", "bar"],
        parse_dates=["foo"],
        cache_dates=cache,
        raise_on_extra_warnings=False,
    )


def test_parse_dates_empty_string(all_parsers):
    # [FUNCTION DOC]
    # See gh-2263. With missing-value filtering disabled, an empty date field becomes NaT; asserts a frame with a datetime and a missing timestamp.
    # see gh-2263
    parser = all_parsers
    data = "Date,test\n2012-01-01,1\n,2"
    result = parser.read_csv(StringIO(data), parse_dates=["Date"], na_filter=False)

    expected = DataFrame(
        [[datetime(2012, 1, 1), 1], [pd.NaT, 2]], columns=["Date", "test"]
    )
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
@pytest.mark.parametrize(
    "data,kwargs,expected",
    [
        (
            "a\n04.15.2016",
            {"parse_dates": ["a"]},
            DataFrame([datetime(2016, 4, 15)], columns=["a"], dtype="M8[us]"),
        ),
        (
            "a\n04.15.2016",
            {"parse_dates": True, "index_col": 0},
            DataFrame(
                index=DatetimeIndex(["2016-04-15"], dtype="M8[us]", name="a"),
                columns=[],
            ),
        ),
        (
            "a,b\n04.15.2016,09.16.2013",
            {"parse_dates": ["a", "b"]},
            DataFrame(
                [[datetime(2016, 4, 15), datetime(2013, 9, 16)]],
                dtype="M8[us]",
                columns=["a", "b"],
            ),
        ),
        (
            "a,b\n04.15.2016,09.16.2013",
            {"parse_dates": True, "index_col": [0, 1]},
            DataFrame(
                index=MultiIndex.from_tuples(
                    [
                        (
                            Timestamp(2016, 4, 15),
                            Timestamp(2013, 9, 16),
                        )
                    ],
                    names=["a", "b"],
                ),
                columns=[],
            ),
        ),
    ],
)
def test_parse_dates_no_convert_thousands(all_parsers, data, kwargs, expected):
    # [FUNCTION DOC]
    # See gh-14066. Parametrized over four cases of dotted dates (single column list, index, two columns, multi-index) with a period thousands separator; asserts that dates are parsed as microsecond datetimes and the dots are not treated as thousands separators. Expected to fail on pyarrow.
    # see gh-14066
    parser = all_parsers

    result = parser.read_csv(StringIO(data), thousands=".", **kwargs)
    tm.assert_frame_equal(result, expected)


def test_parse_date_column_with_empty_string(all_parsers):
    # [FUNCTION DOC]
    # See gh-6428. A date column containing a whitespace-only entry cannot be parsed, so the column stays as strings; asserts the original text including the blank.
    # see gh-6428
    parser = all_parsers
    data = "case,opdate\n7,10/18/2006\n7,10/18/2008\n621, "
    result = parser.read_csv(StringIO(data), parse_dates=["opdate"])

    expected_data = [[7, "10/18/2006"], [7, "10/18/2008"], [621, " "]]
    expected = DataFrame(expected_data, columns=["case", "opdate"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data,expected",
    [
        (
            "a\n135217135789158401\n1352171357E+5",
            [135217135789158401, 135217135700000],
        ),
        (
            "a\n99999999999\n123456789012345\n1234E+0",
            [99999999999, 123456789012345, 1234],
        ),
    ],
)
@pytest.mark.parametrize("parse_dates", [True, False])
def test_parse_date_float(all_parsers, data, expected, parse_dates):
    # [FUNCTION DOC]
    # See gh-2697. Parametrized over inputs with very large numbers and exponent notation, and over parse_dates true/false. Date parsing fails so values are left as floats with unchanged precision; asserts a float64 column.
    # see gh-2697
    #
    # Date parsing should fail, so we leave the data untouched
    # (i.e. float precision should remain unchanged).
    parser = all_parsers

    result = parser.read_csv(StringIO(data), parse_dates=parse_dates)
    expected = DataFrame({"a": expected}, dtype="float64")
    tm.assert_frame_equal(result, expected)


def test_parse_timezone(all_parsers):
    # [FUNCTION DOC]
    # See gh-22256. Timestamps with a fixed +09:00 offset parse into a timezone-aware microsecond column with that offset; asserts equality with a date range of five minutes without frequency.
    # see gh-22256
    parser = all_parsers
    data = """dt,val
              2018-01-04 09:01:00+09:00,23350
              2018-01-04 09:02:00+09:00,23400
              2018-01-04 09:03:00+09:00,23400
              2018-01-04 09:04:00+09:00,23400
              2018-01-04 09:05:00+09:00,23400"""
    result = parser.read_csv(StringIO(data), parse_dates=["dt"])

    dti = date_range(
        start="2018-01-04 09:01:00",
        end="2018-01-04 09:05:00",
        freq="1min",
        tz=timezone(timedelta(minutes=540)),
        unit="us",
    )._with_freq(None)
    expected_data = {"dt": dti, "val": [23350, 23400, 23400, 23400, 23400]}

    expected = DataFrame(expected_data)
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # pandas.errors.ParserError: CSV parse error
@pytest.mark.parametrize(
    "date_string",
    ["32/32/2019", "02/30/2019", "13/13/2019", "13/2019", "a3/11/2018", "10/11/2o17"],
)
def test_invalid_parse_delimited_date(all_parsers, date_string):
    # [FUNCTION DOC]
    # Parametrized over invalid date strings (impossible day or month values, wrong format, letters). A single-cell input with no header and date parsing requested stays as a string column with no conversion; asserts a string-typed frame. Skipped on pyarrow.
    parser = all_parsers
    expected = DataFrame({0: [date_string]}, dtype="str")
    result = parser.read_csv(
        StringIO(date_string),
        header=None,
        parse_dates=[0],
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "date_string,dayfirst,expected",
    [
        # %d/%m/%Y; month > 12 thus replacement
        ("13/02/2019", True, datetime(2019, 2, 13)),
        # %m/%d/%Y; day > 12 thus there will be no replacement
        ("02/13/2019", False, datetime(2019, 2, 13)),
        # %d/%m/%Y; dayfirst==True thus replacement
        ("04/02/2019", True, datetime(2019, 2, 4)),
    ],
)
def test_parse_delimited_date_swap_no_warning(
    all_parsers, date_string, dayfirst, expected, request
):
    # [FUNCTION DOC]
    # Parametrized over (date string, dayfirst, expected) cases where the dayfirst setting matches the data or a swap is unambiguous; asserts the expected microsecond datetime with no warning. For pyarrow with dayfirst true asserts a value error that the option is unsupported; pyarrow with dayfirst false is skipped.
    parser = all_parsers
    expected = DataFrame({0: [expected]}, dtype="datetime64[us]")
    if parser.engine == "pyarrow":
        if not dayfirst:
            # "CSV parse error: Empty CSV file or block"
            pytest.skip(reason="https://github.com/apache/arrow/issues/38676")
        msg = "The 'dayfirst' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(date_string), header=None, dayfirst=dayfirst, parse_dates=[0]
            )
        return

    result = parser.read_csv(
        StringIO(date_string), header=None, dayfirst=dayfirst, parse_dates=[0]
    )
    tm.assert_frame_equal(result, expected)


# ArrowInvalid: CSV parse error: Empty CSV file or block: cannot infer number of columns
@skip_pyarrow
@pytest.mark.parametrize(
    "date_string,dayfirst,expected",
    [
        # %d/%m/%Y; month > 12
        ("13/02/2019", False, datetime(2019, 2, 13)),
        # %m/%d/%Y; day > 12
        ("02/13/2019", True, datetime(2019, 2, 13)),
    ],
)
def test_parse_delimited_date_swap_with_warning(
    all_parsers, date_string, dayfirst, expected
):
    # [FUNCTION DOC]
    # Parametrized over cases where dayfirst conflicts with the data; asserts a user warning that dates were parsed in a different format than dayfirst specified, and the expected parsed datetime. Skipped on pyarrow.
    parser = all_parsers
    expected = DataFrame({0: [expected]}, dtype="datetime64[us]")
    warning_msg = (
        "Parsing dates in .* format when dayfirst=.* was specified. "
        "Pass `dayfirst=.*` or specify a format to silence this warning."
    )
    result = parser.read_csv_check_warnings(
        UserWarning,
        warning_msg,
        StringIO(date_string),
        header=None,
        dayfirst=dayfirst,
        parse_dates=[0],
    )
    tm.assert_frame_equal(result, expected)


def test_parse_multiple_delimited_dates_with_swap_warnings():
    # [FUNCTION DOC]
    # GH46210. Passes a list of mixed day-first and month-first date strings straight to the datetime conversion function (not read_csv) and asserts a value error stating the data does not match the inferred format and suggesting an alternative.
    # KNOWN ISSUE: the name refers to swap warnings and delimited-date parsing from csv, but the body asserts an error from the conversion function directly and does not exercise the csv reader.
    # GH46210
    with pytest.raises(
        ValueError,
        match=(
            r'^time data "31/05/2000" doesn\'t match format "%m/%d/%Y". '
            r"You might want to try:"
        ),
    ):
        pd.to_datetime(["01/01/2000", "31/05/2000", "31/05/2001", "01/02/2000"])


# ArrowKeyError: Column 'fdate1' in include_columns does not exist in CSV file
@skip_pyarrow
@pytest.mark.parametrize(
    "names, usecols, parse_dates, missing_cols",
    [
        (None, ["val"], ["date", "time"], "date, time"),
        (None, ["val"], [0, "time"], "time"),
        (["date1", "time1", "temperature"], None, ["date", "time"], "date, time"),
        (
            ["date1", "time1", "temperature"],
            ["date1", "temperature"],
            ["date1", "time"],
            "time",
        ),
    ],
)
def test_missing_parse_dates_column_raises(
    all_parsers, names, usecols, parse_dates, missing_cols
):
    # [FUNCTION DOC]
    # gh-31251. Parametrized over combinations of supplied names, usecols and parse_dates entries that reference nonexistent columns; asserts a value error naming the missing columns provided to parse_dates. Skipped on pyarrow.
    # gh-31251 column names provided in parse_dates could be missing.
    parser = all_parsers
    content = StringIO("date,time,val\n2020-01-31,04:20:32,32\n")
    msg = f"Missing column provided to 'parse_dates': '{missing_cols}'"

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(
            content, sep=",", names=names, usecols=usecols, parse_dates=parse_dates
        )


@xfail_pyarrow  # mismatched shape
def test_date_parser_and_names(all_parsers):
    # [FUNCTION DOC]
    # GH#33699. A single supplied name equal to the parse_dates column on two-line data: asserts the frame with header values as data and a could-not-infer-format user warning (pyarrow may also emit a Pandas4Warning). Expected to fail on pyarrow.
    # GH#33699
    parser = all_parsers
    data = StringIO("""x,y\n1,2""")
    warn = UserWarning
    if parser.engine == "pyarrow":
        # Pandas4Warning for passing a Manager object
        warn = (UserWarning, Pandas4Warning)
    result = parser.read_csv_check_warnings(
        warn,
        "Could not infer format",
        data,
        parse_dates=["B"],
        names=["B"],
    )
    expected = DataFrame({"B": ["y", "2"]}, index=["x", "1"])
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
def test_date_parser_multiindex_columns(all_parsers):
    # [FUNCTION DOC]
    # Two-row header with a tuple column parsed as date from a multi-level header; asserts a frame with timestamp column keyed by a tuple and an integer column. Expected to fail on pyarrow.
    parser = all_parsers
    data = """a,b
1,2
2019-12-31,6"""
    result = parser.read_csv(StringIO(data), parse_dates=[("a", "1")], header=[0, 1])
    expected = DataFrame({("a", "1"): Timestamp("2019-12-31"), ("b", "2"): [6]})
    tm.assert_frame_equal(result, expected)


def test_date_parser_usecols_thousands(all_parsers):
    # [FUNCTION DOC]
    # GH#39365. Combines parse_dates, usecols and a hyphen thousands separator. pyarrow asserts a value error that thousands is unsupported. Otherwise asserts a could-not-infer-format user warning and a frame with the integer column and a timestamp column.
    # KNOWN ISSUE: an inline comment in the pyarrow branch mentions a deprecation warning for passing a manager object, but that branch asserts the thousands-unsupported value error instead.
    # GH#39365
    data = """A,B,C
    1,3,20-09-01-01
    2,4,20-09-01-01
    """

    parser = all_parsers

    if parser.engine == "pyarrow":
        # DeprecationWarning for passing a Manager object
        msg = "The 'thousands' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data),
                parse_dates=[1],
                usecols=[1, 2],
                thousands="-",
            )
        return

    result = parser.read_csv_check_warnings(
        UserWarning,
        "Could not infer format",
        StringIO(data),
        parse_dates=[1],
        usecols=[1, 2],
        thousands="-",
    )
    expected = DataFrame({"B": [3, 4], "C": [Timestamp("20-09-2001 01:00:00")] * 2})
    tm.assert_frame_equal(result, expected)


def test_dayfirst_warnings():
    # [FUNCTION DOC]
    # GH 12585. Case one with consistent day-first strings: correct dayfirst gives the parsed index without warning; incorrect dayfirst gives a user warning but the same index. Case two with mixed formats: the strings cannot be parsed with one format and are returned unaltered as a string index, with a warning when dayfirst is false and none when true.
    # GH 12585

    # CASE 1: valid input
    input = "date\n31/12/2014\n10/03/2011"
    expected = DatetimeIndex(
        ["2014-12-31", "2011-03-10"], dtype="datetime64[us]", freq=None, name="date"
    )
    warning_msg = (
        "Parsing dates in .* format when dayfirst=.* was specified. "
        "Pass `dayfirst=.*` or specify a format to silence this warning."
    )

    # A. dayfirst arg correct, no warning
    res1 = read_csv(
        StringIO(input), parse_dates=["date"], dayfirst=True, index_col="date"
    ).index
    tm.assert_index_equal(expected, res1)

    # B. dayfirst arg incorrect, warning
    with tm.assert_produces_warning(UserWarning, match=warning_msg):
        res2 = read_csv(
            StringIO(input), parse_dates=["date"], dayfirst=False, index_col="date"
        ).index
    tm.assert_index_equal(expected, res2)

    # CASE 2: invalid input
    # cannot consistently process with single format
    # return to user unaltered

    # first in DD/MM/YYYY, second in MM/DD/YYYY
    input = "date\n31/12/2014\n03/30/2011"
    expected = Index(["31/12/2014", "03/30/2011"], dtype="str", name="date")

    # A. use dayfirst=True
    res5 = read_csv(
        StringIO(input), parse_dates=["date"], dayfirst=True, index_col="date"
    ).index
    tm.assert_index_equal(expected, res5)

    # B. use dayfirst=False
    with tm.assert_produces_warning(UserWarning, match=warning_msg):
        res6 = read_csv(
            StringIO(input), parse_dates=["date"], dayfirst=False, index_col="date"
        ).index
    tm.assert_index_equal(expected, res6)


@pytest.mark.parametrize(
    "date_string, dayfirst",
    [
        pytest.param(
            "31/1/2014",
            False,
            id="second date is single-digit",
        ),
        pytest.param(
            "1/31/2014",
            True,
            id="first date is single-digit",
        ),
    ],
)
def test_dayfirst_warnings_no_leading_zero(date_string, dayfirst):
    # [FUNCTION DOC]
    # GH47880. Parametrized over single-digit-day and single-digit-month dates with the opposite dayfirst setting; asserts the dayfirst-mismatch user warning and the parsed January date index.
    # GH47880
    initial_value = f"date\n{date_string}"
    expected = DatetimeIndex(
        ["2014-01-31"], dtype="datetime64[us]", freq=None, name="date"
    )
    warning_msg = (
        "Parsing dates in .* format when dayfirst=.* was specified. "
        "Pass `dayfirst=.*` or specify a format to silence this warning."
    )
    with tm.assert_produces_warning(UserWarning, match=warning_msg):
        res = read_csv(
            StringIO(initial_value),
            parse_dates=["date"],
            index_col="date",
            dayfirst=dayfirst,
        ).index
    tm.assert_index_equal(expected, res)


@skip_pyarrow  # CSV parse error: Expected 3 columns, got 4
def test_infer_first_column_as_index(all_parsers):
    # [FUNCTION DOC]
    # GH#11019. A data row with one more field than the header makes the first field the index while the named a column is also requested in parse_dates; asserts the string index and constant columns. Skipped on pyarrow.
    # GH#11019
    parser = all_parsers
    data = "a,b,c\n1970-01-01,2,3,4"
    result = parser.read_csv(
        StringIO(data),
        parse_dates=["a"],
    )
    expected = DataFrame({"a": "2", "b": 3, "c": 4}, index=["1970-01-01"])
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # pyarrow engine doesn't support passing a dict for na_values
def test_replace_nans_before_parsing_dates(all_parsers):
    # [FUNCTION DOC]
    # GH#26203. Per-column na_values (dict) are applied before date parsing with an explicit format, so the values marked as missing become NaT; asserts a microsecond datetime column. Expected to fail on pyarrow (no dict na_values).
    # GH#26203
    parser = all_parsers
    data = """Test
2012-10-01
0
2015-05-15
#
2017-09-09
"""
    result = parser.read_csv(
        StringIO(data),
        na_values={"Test": ["#", "0"]},
        parse_dates=["Test"],
        date_format="%Y-%m-%d",
    )
    expected = DataFrame(
        {
            "Test": [
                Timestamp("2012-10-01"),
                pd.NaT,
                Timestamp("2015-05-15"),
                pd.NaT,
                Timestamp("2017-09-09"),
            ]
        },
        dtype="M8[us]",
    )
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # string[python] instead of dt64[ns]
def test_parse_dates_and_string_dtype(all_parsers):
    # [FUNCTION DOC]
    # GH#34066. A string dtype for all columns combined with parse_dates on one column: asserts the other column is string dtype and the parsed column is datetime. Expected to fail on pyarrow.
    # GH#34066
    parser = all_parsers
    data = """a,b
1,2019-12-31
"""
    result = parser.read_csv(StringIO(data), dtype="string", parse_dates=["b"])
    expected = DataFrame({"a": ["1"], "b": [Timestamp("2019-12-31")]})
    expected["a"] = expected["a"].astype("string")
    tm.assert_frame_equal(result, expected)


def test_parse_dot_separated_dates(all_parsers):
    # [FUNCTION DOC]
    # Issue 2586. Dot-separated day.month.year timestamps as the index with parse_dates true. Non-pyarrow engines parse to a microsecond datetime index and emit a user warning about dayfirst being false; pyarrow leaves them as strings without a warning. Asserts the corresponding frame.
    # https://github.com/pandas-dev/pandas/issues/2586
    parser = all_parsers
    data = """a,b
27.03.2003 14:55:00.000,1
03.08.2003 15:20:00.000,2"""
    if parser.engine == "pyarrow":
        expected_index = Index(
            ["27.03.2003 14:55:00.000", "03.08.2003 15:20:00.000"],
            dtype="str",
            name="a",
        )
        warn = None
    else:
        expected_index = DatetimeIndex(
            ["2003-03-27 14:55:00", "2003-08-03 15:20:00"],
            dtype="datetime64[us]",
            name="a",
        )
        warn = UserWarning
    msg = r"when dayfirst=False \(the default\) was specified"
    result = parser.read_csv_check_warnings(
        warn,
        msg,
        StringIO(data),
        parse_dates=True,
        index_col=0,
        raise_on_extra_warnings=False,
    )
    expected = DataFrame({"b": [1, 2]}, index=expected_index)
    tm.assert_frame_equal(result, expected)


def test_parse_dates_dict_format(all_parsers):
    # [FUNCTION DOC]
    # GH#51240. Per-column date_format mapping with day-first and year-first formats for two parsed columns; asserts both columns equal the same microsecond timestamps.
    # GH#51240
    parser = all_parsers
    data = """a,b
2019-12-31,31-12-2019
2020-12-31,31-12-2020"""

    result = parser.read_csv(
        StringIO(data),
        date_format={"a": "%Y-%m-%d", "b": "%d-%m-%Y"},
        parse_dates=["a", "b"],
    )
    expected = DataFrame(
        {
            "a": [Timestamp("2019-12-31"), Timestamp("2020-12-31")],
            "b": [Timestamp("2019-12-31"), Timestamp("2020-12-31")],
        },
        dtype="M8[us]",
    )
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # object dtype index
def test_parse_dates_dict_format_index(all_parsers):
    # [FUNCTION DOC]
    # GH#51240. Per-column date_format only for column a with parse_dates true and index_col zero; asserts a datetime index named a and column b left as strings. Expected to fail on pyarrow (object-dtype index).
    # GH#51240
    parser = all_parsers
    data = """a,b
2019-12-31,31-12-2019
2020-12-31,31-12-2020"""

    result = parser.read_csv(
        StringIO(data), date_format={"a": "%Y-%m-%d"}, parse_dates=True, index_col=0
    )
    expected = DataFrame(
        {
            "b": ["31-12-2019", "31-12-2020"],
        },
        index=Index([Timestamp("2019-12-31"), Timestamp("2020-12-31")], name="a"),
    )
    tm.assert_frame_equal(result, expected)


def test_parse_dates_arrow_engine(all_parsers):
    # [FUNCTION DOC]
    # GH#53295. Parses a datetime column; asserts timestamps and a constant integer column, with the datetime column cast to second unit when the engine is pyarrow.
    # GH#53295
    parser = all_parsers
    data = """a,b
2000-01-01 00:00:00,1
2000-01-01 00:00:01,1"""

    result = parser.read_csv(StringIO(data), parse_dates=["a"])
    expected = DataFrame(
        {
            "a": [
                Timestamp("2000-01-01 00:00:00"),
                Timestamp("2000-01-01 00:00:01"),
            ],
            "b": 1,
        }
    )
    if parser.engine == "pyarrow":
        expected["a"] = expected["a"].astype("M8[s]")
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # object dtype index
def test_from_csv_with_mixed_offsets(all_parsers):
    # [FUNCTION DOC]
    # Timestamps with different UTC offsets in one column cannot be a single timezone-aware dtype, so the series remains strings; asserts equality with the original strings. Expected to fail on pyarrow (object-dtype index).
    parser = all_parsers
    data = "a\n2020-01-01T00:00:00+01:00\n2020-01-01T00:00:00+00:00"
    result = parser.read_csv(StringIO(data), parse_dates=["a"])["a"]
    expected = Series(
        [
            "2020-01-01T00:00:00+01:00",
            "2020-01-01T00:00:00+00:00",
        ],
        name="a",
        index=[0, 1],
    )
    tm.assert_series_equal(result, expected)
