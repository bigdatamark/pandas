# [MODULE DOC]
# Tests for the CSV writer (to_csv on frames and series): quoting and escaping,
# decimal and float formatting (old-style percent, new-style brace and callable
# formats), missing value representation, date formatting, multi-level indexes
# and columns, line terminators, writing to open file handles, standard output
# and in-memory buffers, compression options, and error handling. Most tests
# compare the produced text with expected rows built by a testing helper that
# joins rows with the platform line terminator, or compare bytes read back
# from a temporary file.

import io
import os
import sys
from zipfile import ZipFile

from _csv import Error
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    compat,
)
import pandas._testing as tm


class TestToCSV:
    # [CLASS DOC]
    # Collection of to_csv behaviour tests written as methods; each is independent
    # and uses temporary-file or capture fixtures where needed.
    def test_to_csv_with_single_column(self, temp_file):
        # [FUNCTION DOC]
        # See gh-18676. Writes a single-column frame with header and index suppressed,
        # where the first row is missing (second case: last row missing). The file
        # text must be an empty quoted field line followed by the number, or the number
        # followed by the empty quoted field, working around a standard-library quirk
        # that formerly emitted an extra quoted empty string.
        # see gh-18676, https://bugs.python.org/issue32255
        #
        # Python's CSV library adds an extraneous '""'
        # before the newline when the NaN-value is in
        # the first row. Otherwise, only the newline
        # character is added. This behavior is inconsistent
        # and was patched in https://bugs.python.org/pull_request4672.
        df1 = DataFrame([None, 1])
        expected1 = """\
""
1.0
"""
        df1.to_csv(temp_file, header=None, index=None)
        with open(temp_file, encoding="utf-8") as f:
            assert f.read() == expected1

        df2 = DataFrame([1, None])
        expected2 = """\
1.0
""
"""
        df2.to_csv(temp_file, header=None, index=None)
        with open(temp_file, encoding="utf-8") as f:
            assert f.read() == expected2

    def test_to_csv_default_encoding(self, temp_file):
        # [FUNCTION DOC]
        # GH17097. Writes a frame holding non-ASCII text in several scripts with default
        # settings and reads it back; the default encoding must be utf-8 so the frames
        # are equal.
        # GH17097
        df = DataFrame({"col": ["AAAAA", "ÄÄÄÄÄ", "ßßßßß", "聞聞聞聞聞"]})

        # the default to_csv encoding is uft-8.
        df.to_csv(temp_file)
        tm.assert_frame_equal(pd.read_csv(temp_file, index_col=0), df)

    def test_to_csv_quotechar(self, temp_file):
        # [FUNCTION DOC]
        # Writes a two-row frame with quote-all mode and expects every field, header
        # and index included, surrounded by double quotes; repeats with a custom quote
        # character; and finally expects a type error mentioning quotechar when quoting
        # is enabled but the quote character is none.
        df = DataFrame({"col": [1, 2]})
        expected = """\
"","col"
"0","1"
"1","2"
"""

        df.to_csv(temp_file, quoting=1)  # 1=QUOTE_ALL
        with open(temp_file, encoding="utf-8") as f:
            assert f.read() == expected

        expected = """\
$$,$col$
$0$,$1$
$1$,$2$
"""

        df.to_csv(temp_file, quoting=1, quotechar="$")
        with open(temp_file, encoding="utf-8") as f:
            assert f.read() == expected

        with pytest.raises(TypeError, match="quotechar"):
            df.to_csv(temp_file, quoting=1, quotechar=None)

    def test_to_csv_doublequote(self, temp_file):
        # [FUNCTION DOC]
        # Frame with embedded quote characters written with quote-all and doublequote
        # true: embedded quotes are doubled. With doublequote false and no escape
        # character the writer must raise the csv module error mentioning escapechar.
        df = DataFrame({"col": ['a"a', '"bb"']})
        expected = '''\
"","col"
"0","a""a"
"1","""bb"""
'''

        df.to_csv(temp_file, quoting=1, doublequote=True)  # QUOTE_ALL
        with open(temp_file, encoding="utf-8") as f:
            assert f.read() == expected

        with pytest.raises(Error, match="escapechar"):
            df.to_csv(temp_file, doublequote=False)  # no escapechar set

    def test_to_csv_escapechar(self, temp_file):
        # [FUNCTION DOC]
        # Quote-all with doublequote off and a backslash escape character: embedded
        # quotes are escaped with the backslash. Second case uses no-quoting mode with
        # the escape character: embedded commas are escaped by the backslash.
        df = DataFrame({"col": ['a"a', '"bb"']})
        expected = """\
"","col"
"0","a\\"a"
"1","\\"bb\\""
"""

        df.to_csv(temp_file, quoting=1, doublequote=False, escapechar="\\")
        with open(temp_file, encoding="utf-8") as f:
            assert f.read() == expected

        df = DataFrame({"col": ["a,a", ",bb,"]})
        expected = """\
,col
0,a\\,a
1,\\,bb\\,
"""

        df.to_csv(temp_file, quoting=3, escapechar="\\")  # QUOTE_NONE
        with open(temp_file, encoding="utf-8") as f:
            assert f.read() == expected

    def test_csv_to_string(self):
        # [FUNCTION DOC]
        # With no path, to_csv returns the CSV text; for a one-column frame it must equal
        # the header row and two data rows.
        df = DataFrame({"col": [1, 2]})
        expected_rows = [",col", "0,1", "1,2"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df.to_csv() == expected

    def test_to_csv_decimal(self):
        # [FUNCTION DOC]
        # See gh-781 and gh-11553. Checks default output, a comma decimal with semicolon
        # separator (European Excel style), float_format combined with it, and that the
        # decimal character is applied to floats whose value is zero, in columns, in a
        # single-level index and in a multi-level index.
        # see gh-781
        df = DataFrame({"col1": [1], "col2": ["a"], "col3": [10.1]})

        expected_rows = [",col1,col2,col3", "0,1,a,10.1"]
        expected_default = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df.to_csv() == expected_default

        expected_rows = [";col1;col2;col3", "0;1;a;10,1"]
        expected_european_excel = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df.to_csv(decimal=",", sep=";") == expected_european_excel

        expected_rows = [",col1,col2,col3", "0,1,a,10.10"]
        expected_float_format_default = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df.to_csv(float_format="%.2f") == expected_float_format_default

        expected_rows = [";col1;col2;col3", "0;1;a;10,10"]
        expected_float_format = tm.convert_rows_list_to_csv_str(expected_rows)
        assert (
            df.to_csv(decimal=",", sep=";", float_format="%.2f")
            == expected_float_format
        )

        # see gh-11553: testing if decimal is taken into account for '0.0'
        df = DataFrame({"a": [0, 1.1], "b": [2.2, 3.3], "c": 1})

        expected_rows = ["a,b,c", "0^0,2^2,1", "1^1,3^3,1"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df.to_csv(index=False, decimal="^") == expected

        # same but for an index
        assert df.set_index("a").to_csv(decimal="^") == expected

        # same for a multi-index
        assert df.set_index(["a", "b"]).to_csv(decimal="^") == expected

    def test_to_csv_float_format(self):
        # [FUNCTION DOC]
        # GH 11553. A float format must be applied to float values in a single-level
        # index and in a multi-level index as well as to columns; integer values stay
        # unformatted.
        # testing if float_format is taken into account for the index
        # GH 11553
        df = DataFrame({"a": [0, 1], "b": [2.2, 3.3], "c": 1})

        expected_rows = ["a,b,c", "0,2.20,1", "1,3.30,1"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df.set_index("a").to_csv(float_format="%.2f") == expected

        # same for a multi-index
        assert df.set_index(["a", "b"]).to_csv(float_format="%.2f") == expected

    def test_to_csv_na_rep(self):
        # [FUNCTION DOC]
        # See gh-11553. The missing value representation must be used for missing labels
        # in single and multi-level indexes, for an index of only missing values, must
        # not disturb data with no missing values, and for a series with a pandas NA
        # element must be the full given string.
        # see gh-11553
        #
        # Testing if NaN values are correctly represented in the index.
        df = DataFrame({"a": [0, np.nan], "b": [0, 1], "c": [2, 3]})
        expected_rows = ["a,b,c", "0.0,0,2", "_,1,3"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)

        assert df.set_index("a").to_csv(na_rep="_") == expected
        assert df.set_index(["a", "b"]).to_csv(na_rep="_") == expected

        # now with an index containing only NaNs
        df = DataFrame({"a": np.nan, "b": [0, 1], "c": [2, 3]})
        expected_rows = ["a,b,c", "_,0,2", "_,1,3"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)

        assert df.set_index("a").to_csv(na_rep="_") == expected
        assert df.set_index(["a", "b"]).to_csv(na_rep="_") == expected

        # check if na_rep parameter does not break anything when no NaN
        df = DataFrame({"a": 0, "b": [0, 1], "c": [2, 3]})
        expected_rows = ["a,b,c", "0,0,2", "0,1,3"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)

        assert df.set_index("a").to_csv(na_rep="_") == expected
        assert df.set_index(["a", "b"]).to_csv(na_rep="_") == expected

        csv = pd.Series(["a", pd.NA, "c"]).to_csv(na_rep="ZZZZZ")
        expected = tm.convert_rows_list_to_csv_str([",0", "0,a", "1,ZZZZZ", "2,c"])
        assert expected == csv

    def test_to_csv_na_rep_nullable_string(self, nullable_string_dtype):
        # [FUNCTION DOC]
        # GH 29975. For a series with a parametrized nullable string dtype, the full
        # missing value representation string (not truncated) is written.
        # GH 29975
        # Make sure full na_rep shows up when a dtype is provided
        expected = tm.convert_rows_list_to_csv_str([",0", "0,a", "1,ZZZZZ", "2,c"])
        csv = pd.Series(["a", pd.NA, "c"], dtype=nullable_string_dtype).to_csv(
            na_rep="ZZZZZ"
        )
        assert expected == csv

    def test_to_csv_date_format(self):
        # [FUNCTION DOC]
        # GH 10209 and gh-7791. Compares output for second-resolution and day-resolution
        # datetime columns with the default format and several date formats; the default
        # omits the time part only when all times are midnight. Also verifies the date
        # format is applied to a datetime level of a grouped (multi-index) mean.
        # GH 10209
        df_sec = DataFrame({"A": pd.date_range("20130101", periods=5, freq="s")})
        df_day = DataFrame({"A": pd.date_range("20130101", periods=5, freq="D")})

        expected_rows = [
            ",A",
            "0,2013-01-01 00:00:00",
            "1,2013-01-01 00:00:01",
            "2,2013-01-01 00:00:02",
            "3,2013-01-01 00:00:03",
            "4,2013-01-01 00:00:04",
        ]
        expected_default_sec = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df_sec.to_csv() == expected_default_sec

        expected_rows = [
            ",A",
            "0,2013-01-01 00:00:00",
            "1,2013-01-02 00:00:00",
            "2,2013-01-03 00:00:00",
            "3,2013-01-04 00:00:00",
            "4,2013-01-05 00:00:00",
        ]
        expected_ymdhms_day = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df_day.to_csv(date_format="%Y-%m-%d %H:%M:%S") == expected_ymdhms_day

        expected_rows = [
            ",A",
            "0,2013-01-01",
            "1,2013-01-01",
            "2,2013-01-01",
            "3,2013-01-01",
            "4,2013-01-01",
        ]
        expected_ymd_sec = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df_sec.to_csv(date_format="%Y-%m-%d") == expected_ymd_sec

        expected_rows = [
            ",A",
            "0,2013-01-01",
            "1,2013-01-02",
            "2,2013-01-03",
            "3,2013-01-04",
            "4,2013-01-05",
        ]
        expected_default_day = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df_day.to_csv() == expected_default_day
        assert df_day.to_csv(date_format="%Y-%m-%d") == expected_default_day

        # see gh-7791
        #
        # Testing if date_format parameter is taken into account
        # for multi-indexed DataFrames.
        df_sec["B"] = 0
        df_sec["C"] = 1

        expected_rows = ["A,B,C", "2013-01-01,0,1.0"]
        expected_ymd_sec = tm.convert_rows_list_to_csv_str(expected_rows)

        df_sec_grouped = df_sec.groupby([pd.Grouper(key="A", freq="1h"), "B"])
        assert df_sec_grouped.mean().to_csv(date_format="%Y-%m-%d") == expected_ymd_sec

    def test_to_csv_different_datetime_formats(self):
        # [FUNCTION DOC]
        # GH#21734. A frame with a scalar-broadcast date column and an hourly datetime
        # column written without an index: the date column prints as a date and the
        # datetime column with times, each formatted independently.
        # GH#21734
        df = DataFrame(
            {
                "date": pd.to_datetime("1970-01-01"),
                "datetime": pd.date_range("1970-01-01", periods=2, freq="h"),
            }
        )
        expected_rows = [
            "date,datetime",
            "1970-01-01,1970-01-01 00:00:00",
            "1970-01-01,1970-01-01 01:00:00",
        ]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df.to_csv(index=False) == expected

    def test_to_csv_date_format_in_categorical(self):
        # [FUNCTION DOC]
        # GH#40754. A categorical series of datetimes with a missing value, and one of
        # timezone-aware datetimes, written without index: the missing value becomes an
        # empty quoted field and, for the second case, the given date format applies to
        # the category values.
        # GH#40754
        ser = pd.Series(pd.to_datetime(["2021-03-27", pd.NaT], format="%Y-%m-%d"))
        ser = ser.astype("category")
        expected = tm.convert_rows_list_to_csv_str(["0", "2021-03-27", '""'])
        assert ser.to_csv(index=False) == expected

        ser = pd.Series(
            pd.date_range(
                start="2021-03-27", freq="D", periods=1, tz="Europe/Berlin"
            ).append(pd.DatetimeIndex([pd.NaT]))
        )
        ser = ser.astype("category")
        assert ser.to_csv(index=False, date_format="%Y-%m-%d") == expected

    def test_to_csv_float_ea_float_format(self):
        # [FUNCTION DOC]
        # GH#45991. A nullable Float64 column with a missing value is written with a
        # percent float format: values are formatted to five decimals and the missing
        # entry is empty.
        # GH#45991
        df = DataFrame({"a": [1.1, 2.02, pd.NA, 6.000006], "b": "c"})
        df["a"] = df["a"].astype("Float64")
        result = df.to_csv(index=False, float_format="%.5f")
        expected = tm.convert_rows_list_to_csv_str(
            ["a,b", "1.10000,c", "2.02000,c", ",c", "6.00001,c"]
        )
        assert result == expected

    def test_to_csv_float_ea_no_float_format(self):
        # [FUNCTION DOC]
        # GH#45991. The same nullable Float64 column written without a float format
        # prints values at their natural precision with an empty field for missing.
        # GH#45991
        df = DataFrame({"a": [1.1, 2.02, pd.NA, 6.000006], "b": "c"})
        df["a"] = df["a"].astype("Float64")
        result = df.to_csv(index=False)
        expected = tm.convert_rows_list_to_csv_str(
            ["a,b", "1.1,c", "2.02,c", ",c", "6.000006,c"]
        )
        assert result == expected

    def test_to_csv_float_ea_nan_distinguish(self, using_nan_is_na):
        # [FUNCTION DOC]
        # GH#61617, GH#65227. A nullable Float64 array holding a real NaN, a missing NA
        # and a number must not crash the writer. With the fixture indicating NaN is
        # treated as NA, both missing kinds print as empty fields; otherwise the NaN
        # prints as the text nan and NA as empty.
        # GH#61617, GH#65227 - to_csv should not crash when FloatingArray
        # contains unmasked NaN (with distinguish_nan_and_na=True)
        df = DataFrame({"a": pd.array([np.nan, pd.NA, 3.0], dtype="Float64"), "b": "c"})
        result = df.to_csv(index=False)
        if using_nan_is_na:
            expected = tm.convert_rows_list_to_csv_str(["a,b", ",c", ",c", "3.0,c"])
        else:
            expected = tm.convert_rows_list_to_csv_str(["a,b", "nan,c", ",c", "3.0,c"])
        assert result == expected

    def test_to_csv_float_ea_nan_distinguish_series(self, using_nan_is_na):
        # [FUNCTION DOC]
        # GH#65227. Same distinction for a nullable Float64 series built by dividing a
        # series by itself (producing 1, NA and NaN): the NaN prints as empty under NaN
        # is NA semantics, else as the text nan.
        # GH#65227 - Series.to_csv with FloatingArray containing both NaN and NA
        ser = pd.Series((1, pd.NA, 0), index=["a", "b", "c"], dtype="Float64", name="x")
        ser = ser / ser
        result = ser.to_csv()
        if using_nan_is_na:
            expected = tm.convert_rows_list_to_csv_str([",x", "a,1.0", "b,", "c,"])
        else:
            expected = tm.convert_rows_list_to_csv_str([",x", "a,1.0", "b,", "c,nan"])
        assert result == expected

    def test_to_csv_multi_index(self):
        # [FUNCTION DOC]
        # See gh-6618. Columns as a two-level multi-index are written as two header rows
        # (and with the index label blank cells in front); with index disabled only the
        # header levels and data are written. Checked with a multi-level row index too
        # and with string labels.
        # see gh-6618
        df = DataFrame([1], columns=pd.MultiIndex.from_arrays([[1], [2]]))

        exp_rows = [",1", ",2", "0,1"]
        exp = tm.convert_rows_list_to_csv_str(exp_rows)
        assert df.to_csv() == exp

        exp_rows = ["1", "2", "1"]
        exp = tm.convert_rows_list_to_csv_str(exp_rows)
        assert df.to_csv(index=False) == exp

        df = DataFrame(
            [1],
            columns=pd.MultiIndex.from_arrays([[1], [2]]),
            index=pd.MultiIndex.from_arrays([[1], [2]]),
        )

        exp_rows = [",,1", ",,2", "1,2,1"]
        exp = tm.convert_rows_list_to_csv_str(exp_rows)
        assert df.to_csv() == exp

        exp_rows = ["1", "2", "1"]
        exp = tm.convert_rows_list_to_csv_str(exp_rows)
        assert df.to_csv(index=False) == exp

        df = DataFrame([1], columns=pd.MultiIndex.from_arrays([["foo"], ["bar"]]))

        exp_rows = [",foo", ",bar", "0,1"]
        exp = tm.convert_rows_list_to_csv_str(exp_rows)
        assert df.to_csv() == exp

        exp_rows = ["foo", "bar", "1"]
        exp = tm.convert_rows_list_to_csv_str(exp_rows)
        assert df.to_csv(index=False) == exp

    @pytest.mark.parametrize(
        "ind,expected",
        [
            (
                pd.MultiIndex(levels=[[1.0]], codes=[[0]], names=["x"]),
                "x,data\n1.0,1\n",
            ),
            (
                pd.MultiIndex(
                    levels=[[1.0], [2.0]], codes=[[0], [0]], names=["x", "y"]
                ),
                "x,y,data\n1.0,2.0,1\n",
            ),
        ],
    )
    def test_to_csv_single_level_multi_index(self, ind, expected, frame_or_series):
        # [FUNCTION DOC]
        # See gh-19589. Parametrized on a one-level and a two-level multi-index with
        # named levels and on frame versus series wrapper. Writing with a newline
        # terminator and header gives the level names plus data column header followed by
        # the index values and data value.
        # see gh-19589
        obj = frame_or_series(pd.Series([1], ind, name="data"))

        result = obj.to_csv(lineterminator="\n", header=True)
        assert result == expected

    def test_to_csv_string_array_ascii(self, temp_file):
        # [FUNCTION DOC]
        # GH 10813. A frame containing lists of strings written with ascii encoding gives
        # the list text quoted in the file.
        # GH 10813
        str_array = [{"names": ["foo", "bar"]}, {"names": ["baz", "qux"]}]
        df = DataFrame(str_array)
        expected_ascii = """\
,names
0,"['foo', 'bar']"
1,"['baz', 'qux']"
"""
        df.to_csv(temp_file, encoding="ascii")
        with open(temp_file, encoding="utf-8") as f:
            assert f.read() == expected_ascii

    def test_to_csv_string_array_utf8(self, temp_file):
        # [FUNCTION DOC]
        # GH 10813. The same frame written with utf-8 gives identical text.
        # GH 10813
        str_array = [{"names": ["foo", "bar"]}, {"names": ["baz", "qux"]}]
        df = DataFrame(str_array)
        expected_utf8 = """\
,names
0,"['foo', 'bar']"
1,"['baz', 'qux']"
"""
        df.to_csv(temp_file, encoding="utf-8")
        with open(temp_file, encoding="utf-8") as f:
            assert f.read() == expected_utf8

    def test_to_csv_string_with_lf(self, temp_file):
        # [FUNCTION DOC]
        # GH 20353. Strings with embedded line feeds are quoted. Three cases compared
        # as bytes: default line terminator (the platform one, PR 21406), line feed, and
        # carriage return plus line feed; the line terminator never changes text inside
        # fields.
        # GH 20353
        data = {"int": [1, 2, 3], "str_lf": ["abc", "d\nef", "g\nh\n\ni"]}
        df = DataFrame(data)

        # case 1: The default line terminator(=os.linesep)(PR 21406)
        os_linesep = os.linesep.encode("utf-8")
        expected_noarg = (
            b"int,str_lf"
            + os_linesep
            + b"1,abc"
            + os_linesep
            + b'2,"d\nef"'
            + os_linesep
            + b'3,"g\nh\n\ni"'
            + os_linesep
        )
        df.to_csv(temp_file, index=False)
        with open(temp_file, "rb") as f:
            assert f.read() == expected_noarg

        # case 2: LF as line terminator
        expected_lf = b'int,str_lf\n1,abc\n2,"d\nef"\n3,"g\nh\n\ni"\n'
        df.to_csv(temp_file, lineterminator="\n", index=False)
        with open(temp_file, "rb") as f:
            assert f.read() == expected_lf

        # case 3: CRLF as line terminator
        # 'lineterminator' should not change inner element
        expected_crlf = b'int,str_lf\r\n1,abc\r\n2,"d\nef"\r\n3,"g\nh\n\ni"\r\n'
        df.to_csv(temp_file, lineterminator="\r\n", index=False)
        with open(temp_file, "rb") as f:
            assert f.read() == expected_crlf

    def test_to_csv_string_with_crlf(self, temp_file):
        # [FUNCTION DOC]
        # GH 20353. As the line-feed test, but the strings contain embedded carriage
        # return plus line feed sequences, which must be preserved inside quoted fields
        # for all three terminator choices.
        # GH 20353
        data = {"int": [1, 2, 3], "str_crlf": ["abc", "d\r\nef", "g\r\nh\r\n\r\ni"]}
        df = DataFrame(data)
        # case 1: The default line terminator(=os.linesep)(PR 21406)
        os_linesep = os.linesep.encode("utf-8")
        expected_noarg = (
            b"int,str_crlf"
            + os_linesep
            + b"1,abc"
            + os_linesep
            + b'2,"d\r\nef"'
            + os_linesep
            + b'3,"g\r\nh\r\n\r\ni"'
            + os_linesep
        )
        df.to_csv(temp_file, index=False)
        with open(temp_file, "rb") as f:
            assert f.read() == expected_noarg

        # case 2: LF as line terminator
        expected_lf = b'int,str_crlf\n1,abc\n2,"d\r\nef"\n3,"g\r\nh\r\n\r\ni"\n'
        df.to_csv(temp_file, lineterminator="\n", index=False)
        with open(temp_file, "rb") as f:
            assert f.read() == expected_lf

        # case 3: CRLF as line terminator
        # 'lineterminator' should not change inner element
        expected_crlf = (
            b'int,str_crlf\r\n1,abc\r\n2,"d\r\nef"\r\n3,"g\r\nh\r\n\r\ni"\r\n'
        )
        df.to_csv(temp_file, lineterminator="\r\n", index=False)
        with open(temp_file, "rb") as f:
            assert f.read() == expected_crlf

    def test_to_csv_stdout_file(self, capsys):
        # [FUNCTION DOC]
        # GH 21561. Writing to the standard output stream with ascii encoding emits the
        # expected text to captured output and leaves the stream open.
        # GH 21561
        df = DataFrame([["foo", "bar"], ["baz", "qux"]], columns=["name_1", "name_2"])
        expected_rows = [",name_1,name_2", "0,foo,bar", "1,baz,qux"]
        expected_ascii = tm.convert_rows_list_to_csv_str(expected_rows)

        df.to_csv(sys.stdout, encoding="ascii")
        captured = capsys.readouterr()

        assert captured.out == expected_ascii
        assert not sys.stdout.closed

    @pytest.mark.xfail(
        compat.is_platform_windows(),
        reason=(
            "Especially in Windows, file stream should not be passed"
            "to csv writer without newline='' option."
            "(https://docs.python.org/3/library/csv.html#csv.writer)"
        ),
    )
    def test_to_csv_write_to_open_file(self, temp_file):
        # [FUNCTION DOC]
        # GH 21696. Expected to fail on Windows. Writes a header line manually to a file
        # opened in text mode, then appends the frame with header and index disabled, and
        # expects the file to contain the manual header and rows.
        # GH 21696
        df = DataFrame({"a": ["x", "y", "z"]})
        expected = """\
manual header
x
y
z
"""
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write("manual header\n")
            df.to_csv(f, header=None, index=None)
        with open(temp_file, encoding="utf-8") as f:
            assert f.read() == expected

    def test_to_csv_write_to_open_file_with_newline_py3(self, temp_file):
        # [FUNCTION DOC]
        # See gh-21696, gh-20353. Same as the previous test but the file is opened with
        # an empty newline argument and compared as bytes with the expected text, which
        # uses the platform line terminator.
        # see gh-21696
        # see gh-20353
        df = DataFrame({"a": ["x", "y", "z"]})
        expected_rows = ["x", "y", "z"]
        expected = "manual header\n" + tm.convert_rows_list_to_csv_str(expected_rows)

        with open(temp_file, "w", newline="", encoding="utf-8") as f:
            f.write("manual header\n")
            df.to_csv(f, header=None, index=None)

        with open(temp_file, "rb") as f:
            assert f.read() == bytes(expected, "utf-8")

    @pytest.mark.parametrize("to_infer", [True, False])
    @pytest.mark.parametrize("read_infer", [True, False])
    def test_to_csv_compression(
        self,
        compression_only,
        read_infer,
        to_infer,
        compression_to_extension,
        temp_file,
    ):
        # [FUNCTION DOC]
        # See gh-15008. Parametrized on inferring compression on write and on read,
        # with a compression-only fixture. Writes a path with the matching extension,
        # reads it back with index column zero and expects equality.
        # see gh-15008
        compression = compression_only

        df = DataFrame({"A": [1]})

        to_compression = "infer" if to_infer else compression
        read_compression = "infer" if read_infer else compression

        path_ext = str(temp_file) + "." + compression_to_extension[compression]
        df.to_csv(path_ext, compression=to_compression)
        result = pd.read_csv(path_ext, index_col=0, compression=read_compression)
        tm.assert_frame_equal(result, df)

    def test_to_csv_compression_dict(self, compression_only, temp_file):
        # [FUNCTION DOC]
        # GH 26023. Compression given as a mapping with a method key; the file extension
        # maps gzip to gz and zstd to zst, other methods use their own name. Reading back
        # with inferred compression must give the original frame.
        # GH 26023
        method = compression_only
        df = DataFrame({"ABC": [1]})
        extension = {
            "gzip": "gz",
            "zstd": "zst",
        }.get(method, method)

        path = str(temp_file) + "." + extension
        df.to_csv(path, compression={"method": method})
        read_df = pd.read_csv(path, index_col=0)
        tm.assert_frame_equal(read_df, df)

    def test_to_csv_compression_dict_no_method_raises(self, temp_file):
        # [FUNCTION DOC]
        # GH 26023. A compression mapping without a method key must raise a value error
        # saying the mapping must have key method.
        # GH 26023
        df = DataFrame({"ABC": [1]})
        compression = {"some_option": True}
        msg = "must have key 'method'"

        with pytest.raises(ValueError, match=msg):
            df.to_csv(temp_file, compression=compression)

    @pytest.mark.parametrize("compression", ["zip", "infer"])
    @pytest.mark.parametrize("archive_name", ["test_to_csv.csv", "test_to_csv.zip"])
    def test_to_csv_zip_arguments(self, compression, archive_name, temp_file):
        # [FUNCTION DOC]
        # GH 26023. Parametrized on zip or infer as method and two archive names. Passing
        # an archive name in the compression mapping gives a zip with one member of
        # exactly that name.
        # GH 26023
        df = DataFrame({"ABC": [1]})

        path = str(temp_file) + ".zip"
        df.to_csv(
            path, compression={"method": compression, "archive_name": archive_name}
        )
        with ZipFile(path) as zp:
            assert len(zp.filelist) == 1
            archived_file = zp.filelist[0].filename
            assert archived_file == archive_name

    @pytest.mark.parametrize(
        "filename,expected_arcname",
        [
            ("archive.csv", "archive.csv"),
            ("archive.tsv", "archive.tsv"),
            ("archive.csv.zip", "archive.csv"),
            ("archive.tsv.zip", "archive.tsv"),
            ("archive.zip", "archive"),
        ],
    )
    def test_to_csv_zip_infer_name(self, tmp_path, filename, expected_arcname):
        # [FUNCTION DOC]
        # GH 39465. Parametrized on file names. With zip compression the member name
        # in the archive is inferred from the path: a trailing zip extension is removed
        # and other names are kept as is.
        # GH 39465
        df = DataFrame({"ABC": [1]})
        path = tmp_path / filename
        df.to_csv(path, compression="zip")
        with ZipFile(path) as zp:
            assert len(zp.filelist) == 1
            archived_file = zp.filelist[0].filename
            assert archived_file == expected_arcname

    @pytest.mark.parametrize("df_new_type", ["Int64"])
    def test_to_csv_na_rep_long_string(self, df_new_type):
        # [FUNCTION DOC]
        # See gh-25099. Parametrized (single value) on the nullable integer type. A
        # column of missing values written with a long missing representation and ascii
        # encoding yields that string in every row.
        # see gh-25099
        df = DataFrame({"c": [pd.NA] * 3})
        df = df.astype(df_new_type)
        expected_rows = ["c", "mynull", "mynull", "mynull"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)

        result = df.to_csv(index=False, na_rep="mynull", encoding="ascii")

        assert expected == result

    def test_to_csv_timedelta_precision(self):
        # [FUNCTION DOC]
        # GH 6783. A series of one-nanosecond timedeltas written to a buffer keeps full
        # nanosecond precision in the text.
        # GH 6783
        s = pd.Series([1, 1]).astype("timedelta64[ns]")
        buf = io.StringIO()
        s.to_csv(buf)
        result = buf.getvalue()
        expected_rows = [
            ",0",
            "0,0 days 00:00:00.000000001",
            "1,0 days 00:00:00.000000001",
        ]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert result == expected

    def test_na_rep_truncated(self):
        # [FUNCTION DOC]
        # Issue 31447. Where the series contains no missing values (integers, booleans,
        # floats) the na_rep value, however short, must not be used or truncate anything.
        # https://github.com/pandas-dev/pandas/issues/31447
        result = pd.Series(range(8, 12)).to_csv(na_rep="-")
        expected = tm.convert_rows_list_to_csv_str([",0", "0,8", "1,9", "2,10", "3,11"])
        assert result == expected

        result = pd.Series([True, False]).to_csv(na_rep="nan")
        expected = tm.convert_rows_list_to_csv_str([",0", "0,True", "1,False"])
        assert result == expected

        result = pd.Series([1.1, 2.2]).to_csv(na_rep=".")
        expected = tm.convert_rows_list_to_csv_str([",0", "0,1.1", "1,2.2"])
        assert result == expected

    @pytest.mark.parametrize("errors", ["surrogatepass", "ignore", "replace"])
    def test_to_csv_errors(self, errors, temp_file):
        # [FUNCTION DOC]
        # GH 22610. Parametrized on the encoding error handling modes surrogatepass,
        # ignore and replace. Writing a string containing a lone surrogate to a file
        # with each mode must not raise. There is no readback check.
        # GH 22610
        data = ["\ud800foo"]
        ser = pd.Series(data, index=Index(data, dtype=object), dtype=object)

        ser.to_csv(temp_file, errors=errors)
        # No use in reading back the data as it is not the same anymore
        # due to the error handling

    @pytest.mark.parametrize("mode", ["wb", "w"])
    def test_to_csv_binary_handle(self, mode, temp_file):
        # [FUNCTION DOC]
        # GH 35058 and GH 19827. Parametrized on mode wb or w. A frame is written to a
        # file object opened in binary read-write mode; reading the file back must give
        # an equal frame, so binary handles work regardless of mode letter.
        """
        Binary file objects should work (if 'mode' contains a 'b') or even without
        it in most cases.

        GH 35058 and GH 19827
        """
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD")),
            index=Index([f"i-{i}" for i in range(30)]),
        )

        with open(temp_file, mode="w+b") as handle:
            df.to_csv(handle, mode=mode)
        tm.assert_frame_equal(df, pd.read_csv(temp_file, index_col=0))

    @pytest.mark.parametrize("mode", ["wb", "w"])
    def test_to_csv_encoding_binary_handle(self, mode, temp_file):
        # [FUNCTION DOC]
        # GH 23854 and GH 13068. Parametrized on the mode. First, a byte stream with a
        # signature-prefixed utf-8 header is read into a frame and written to a binary
        # buffer with the same encoding; the output starts with the original bytes, and
        # the buffer remains open. Second, an empty frame written to a binary file with
        # that encoding starts with the byte-order mark followed by empty quoted header.
        """
        Binary file objects should honor a specified encoding.

        GH 23854 and GH 13068 with binary handles
        """
        # example from GH 23854
        content = "a, b, 🐟".encode("utf-8-sig")
        buffer = io.BytesIO(content)
        df = pd.read_csv(buffer, encoding="utf-8-sig")

        buffer = io.BytesIO()
        df.to_csv(buffer, mode=mode, encoding="utf-8-sig", index=False)
        buffer.seek(0)  # tests whether file handle wasn't closed
        assert buffer.getvalue().startswith(content)

        # example from GH 13068
        with open(temp_file, "w+b") as handle:
            DataFrame().to_csv(handle, mode=mode, encoding="utf-8-sig")

            handle.seek(0)
            assert handle.read().startswith(b'\xef\xbb\xbf""')


def test_to_csv_iterative_compression_name(compression, temp_file):
    # [FUNCTION DOC]
    # GH 38714. A frame written to a path with the parametrized compression in chunks
    # of one row and read back with the same compression must be equal.
    # GH 38714
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD")),
        index=Index([f"i-{i}" for i in range(30)]),
    )
    df.to_csv(temp_file, compression=compression, chunksize=1)
    tm.assert_frame_equal(
        pd.read_csv(temp_file, compression=compression, index_col=0), df
    )


def test_to_csv_iterative_compression_buffer(compression):
    # [FUNCTION DOC]
    # GH 38714. As the previous test but writing to an in-memory binary buffer; after
    # writing the buffer is rewound, read back equal, and must not have been closed.
    # GH 38714
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD")),
        index=Index([f"i-{i}" for i in range(30)]),
    )
    with io.BytesIO() as buffer:
        df.to_csv(buffer, compression=compression, chunksize=1)
        buffer.seek(0)
        tm.assert_frame_equal(
            pd.read_csv(buffer, compression=compression, index_col=0), df
        )
        assert not buffer.closed


def test_new_style_float_format_basic():
    # [FUNCTION DOC]
    # A brace-style float format with two decimals is accepted and applied to floats.
    df = DataFrame({"A": [1234.56789, 9876.54321]})
    result = df.to_csv(float_format="{:.2f}", lineterminator="\n")
    expected = ",A\n0,1234.57\n1,9876.54\n"
    assert result == expected


def test_new_style_float_format_thousands():
    # [FUNCTION DOC]
    # A brace-style format with thousands separators produces values containing
    # commas, so those fields are quoted.
    df = DataFrame({"A": [1234.56789, 9876.54321]})
    result = df.to_csv(float_format="{:,.2f}", lineterminator="\n")
    expected = ',A\n0,"1,234.57"\n1,"9,876.54"\n'
    assert result == expected


def test_new_style_scientific_format():
    # [FUNCTION DOC]
    # A brace-style scientific notation format with two decimals is applied.
    df = DataFrame({"A": [0.000123, 0.000456]})
    result = df.to_csv(float_format="{:.2e}", lineterminator="\n")
    expected = ",A\n0,1.23e-04\n1,4.56e-04\n"
    assert result == expected


def test_new_style_with_nan():
    # [FUNCTION DOC]
    # A brace-style format combined with na_rep: missing floats use na_rep and are
    # not formatted.
    df = DataFrame({"A": [1.23, np.nan, 4.56]})
    result = df.to_csv(float_format="{:.2f}", na_rep="NA", lineterminator="\n")
    expected = ",A\n0,1.23\n1,NA\n2,4.56\n"
    assert result == expected


def test_new_style_with_mixed_types():
    # [FUNCTION DOC]
    # A brace-style float format applies only to float columns; a string column is
    # untouched.
    df = DataFrame({"A": [1.23, 4.56], "B": ["x", "y"]})
    result = df.to_csv(float_format="{:.2f}", lineterminator="\n")
    expected = ",A,B\n0,1.23,x\n1,4.56,y\n"
    assert result == expected


def test_new_style_with_mixed_types_in_column():
    # [FUNCTION DOC]
    # In a column mixing floats and a string, only the float elements are formatted.
    df = DataFrame({"A": [1.23, "text", 4.56]})
    result = df.to_csv(float_format="{:.2f}", lineterminator="\n")
    expected = ",A\n0,1.23\n1,text\n2,4.56\n"
    assert result == expected


def test_invalid_new_style_format_missing_brace():
    # [FUNCTION DOC]
    # A float format missing its closing brace must raise a value error naming it
    # as an invalid new-style format string.
    df = DataFrame({"A": [1.23]})
    with pytest.raises(ValueError, match="Invalid new-style format string '{:.2f"):
        df.to_csv(float_format="{:.2f")


def test_invalid_new_style_format_specifier():
    # [FUNCTION DOC]
    # A brace-style format with an invalid specifier letter must raise a value error
    # naming it as an invalid new-style format string.
    df = DataFrame({"A": [1.23]})
    with pytest.raises(ValueError, match="Invalid new-style format string '{:.2z}'"):
        df.to_csv(float_format="{:.2z}")


def test_old_style_format_compatibility():
    # [FUNCTION DOC]
    # A percent-style float format still works and gives two-decimal values.
    df = DataFrame({"A": [1234.56789, 9876.54321]})
    result = df.to_csv(float_format="%.2f", lineterminator="\n")
    expected = ",A\n0,1234.57\n1,9876.54\n"
    assert result == expected


def test_callable_float_format_compatibility():
    # [FUNCTION DOC]
    # A callable float format is applied to each float; the callable inserts thousands
    # separators so the fields are quoted.
    df = DataFrame({"A": [1234.56789, 9876.54321]})
    result = df.to_csv(float_format=lambda x: f"{x:,.2f}", lineterminator="\n")
    expected = ',A\n0,"1,234.57"\n1,"9,876.54"\n'
    assert result == expected


def test_no_float_format():
    # [FUNCTION DOC]
    # A float format of none gives default float text.
    df = DataFrame({"A": [1.23, 4.56]})
    result = df.to_csv(float_format=None, lineterminator="\n")
    expected = ",A\n0,1.23\n1,4.56\n"
    assert result == expected


def test_large_numbers():
    # [FUNCTION DOC]
    # A scientific format applied to a huge finite value and a value that overflows
    # to infinity: the finite value is formatted and the overflow prints as inf.
    df = DataFrame({"A": [1e308, 2e308]})
    result = df.to_csv(float_format="{:.2e}", lineterminator="\n")
    expected = ",A\n0,1.00e+308\n1,inf\n"
    assert result == expected


def test_zero_and_negative():
    # [FUNCTION DOC]
    # A format with explicit sign shows plus for zero and minus for negatives.
    df = DataFrame({"A": [0.0, -1.23456]})
    result = df.to_csv(float_format="{:+.2f}", lineterminator="\n")
    expected = ",A\n0,+0.00\n1,-1.23\n"
    assert result == expected


def test_unicode_format():
    # [FUNCTION DOC]
    # A float format with a trailing non-ASCII currency symbol and utf-8 encoding
    # appends the symbol to each value.
    df = DataFrame({"A": [1.23, 4.56]})
    result = df.to_csv(float_format="{:.2f}€", encoding="utf-8", lineterminator="\n")
    expected = ",A\n0,1.23€\n1,4.56€\n"
    assert result == expected


def test_empty_dataframe():
    # [FUNCTION DOC]
    # An empty one-column frame with a float format writes just the header line.
    df = DataFrame({"A": []})
    result = df.to_csv(float_format="{:.2f}", lineterminator="\n")
    expected = ",A\n"
    assert result == expected


def test_multi_column_float():
    # [FUNCTION DOC]
    # The float format is applied to every float column.
    df = DataFrame({"A": [1.23, 4.56], "B": [7.89, 0.12]})
    result = df.to_csv(float_format="{:.2f}", lineterminator="\n")
    expected = ",A,B\n0,1.23,7.89\n1,4.56,0.12\n"
    assert result == expected


def test_invalid_float_format_type():
    # [FUNCTION DOC]
    # A float format that is neither string nor callable (an integer) must raise a
    # value error saying it must be a string or callable.
    df = DataFrame({"A": [1.23]})
    with pytest.raises(ValueError, match="float_format must be a string or callable"):
        df.to_csv(float_format=123)


def test_new_style_with_inf():
    # [FUNCTION DOC]
    # Infinite values with a brace-style format print as inf and negative inf and are
    # not treated as missing.
    df = DataFrame({"A": [1.23, np.inf, -np.inf]})
    result = df.to_csv(float_format="{:.2f}", na_rep="NA", lineterminator="\n")
    expected = ",A\n0,1.23\n1,inf\n2,-inf\n"
    assert result == expected


def test_new_style_with_precision_edge():
    # [FUNCTION DOC]
    # A brace-style format with ten decimals pads the value with trailing zeros.
    df = DataFrame({"A": [1.23456789]})
    result = df.to_csv(float_format="{:.10f}", lineterminator="\n")
    expected = ",A\n0,1.2345678900\n"
    assert result == expected


def test_new_style_with_template():
    # [FUNCTION DOC]
    # A brace-style format with surrounding text and thousands separator embeds the
    # value in the text; the result contains a comma so the field is quoted.
    df = DataFrame({"A": [1234.56789]})
    result = df.to_csv(float_format="Value: {:,.2f}", lineterminator="\n")
    expected = ',A\n0,"Value: 1,234.57"\n'
    assert result == expected
