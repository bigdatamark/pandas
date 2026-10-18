# [MODULE DOC]
# Tests for the dtype, converters and parse_dates options of the XML reader.
# Each positive test parses an in-memory XML string with the reader and also
# re-parses the same data through a temporary file using the streaming
# (iterparse) mode, and checks both results equal one expected frame.
# Parametrized over the available parsers (lxml when installed, and etree).
# Module-level XML strings hold shapes data with and without a date column.
# No import-time side effects other than defining fixtures and constants.

from __future__ import annotations

from io import StringIO

import pytest

from pandas.errors import ParserWarning
import pandas.util._test_decorators as td

from pandas import (
    NA,
    DataFrame,
    DatetimeIndex,
    Series,
    to_datetime,
)
import pandas._testing as tm

from pandas.io.xml import read_xml


@pytest.fixture(params=[pytest.param("lxml", marks=td.skip_if_no("lxml")), "etree"])
def parser(request):
    # [FUNCTION DOC]
    # Fixture parametrized over the XML parser backend: lxml (skipped when the
    # library is missing) and etree. Yields the parser name string.
    return request.param


@pytest.fixture(
    params=[None, {"book": ["category", "title", "author", "year", "price"]}]
)
def iterparse(request):
    # [FUNCTION DOC]
    # Fixture parametrized over the iterparse argument: either no value or a
    # mapping of the book element to its child names. Yields that value.
    return request.param


def read_xml_iterparse(data, temp_file, **kwargs):
    # [FUNCTION DOC]
    # HELPER: used by the tests to exercise the streaming reader on in-memory data.
    # Writes the given XML text to the given file path using utf-8 encoding and then
    # reads that file with the XML reader, forwarding all extra keyword arguments.
    # Returns the resulting frame. Side effect: creates or overwrites the file.
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(data)
    return read_xml(temp_file, **kwargs)


xml_types = """\
<?xml version='1.0' encoding='utf-8'?>
<data>
  <row>
    <shape>square</shape>
    <degrees>00360</degrees>
    <sides>4.0</sides>
   </row>
  <row>
    <shape>circle</shape>
    <degrees>00360</degrees>
    <sides/>
  </row>
  <row>
    <shape>triangle</shape>
    <degrees>00180</degrees>
    <sides>3.0</sides>
  </row>
</data>"""

xml_dates = """<?xml version='1.0' encoding='utf-8'?>
<data>
  <row>
    <shape>square</shape>
    <degrees>00360</degrees>
    <sides>4.0</sides>
    <date>2020-01-01</date>
   </row>
  <row>
    <shape>circle</shape>
    <degrees>00360</degrees>
    <sides/>
    <date>2021-01-01</date>
  </row>
  <row>
    <shape>triangle</shape>
    <degrees>00180</degrees>
    <sides>3.0</sides>
    <date>2022-01-01</date>
  </row>
</data>"""


# DTYPE


def test_dtype_single_str(parser, temp_file):
    # [FUNCTION DOC]
    # Reads the types data with the degrees column forced to string dtype. Expects
    # degrees to keep its leading zeros as text, while sides stays float with a
    # missing value, for both the in-memory and iterparse readers.
    df_result = read_xml(StringIO(xml_types), dtype={"degrees": "str"}, parser=parser)
    df_iter = read_xml_iterparse(
        xml_types,
        parser=parser,
        dtype={"degrees": "str"},
        iterparse={"row": ["shape", "degrees", "sides"]},
        temp_file=temp_file,
    )

    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": ["00360", "00360", "00180"],
            "sides": [4.0, float("nan"), 3.0],
        }
    )

    tm.assert_frame_equal(df_result, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_dtypes_all_str(parser, temp_file):
    # [FUNCTION DOC]
    # Reads the dates data with a single string extension dtype for every column.
    # Expects all columns, including sides and date, to be strings with the empty
    # sides entry as a missing value, for both reading modes.
    df_result = read_xml(StringIO(xml_dates), dtype="string", parser=parser)
    df_iter = read_xml_iterparse(
        xml_dates,
        parser=parser,
        dtype="string",
        iterparse={"row": ["shape", "degrees", "sides", "date"]},
        temp_file=temp_file,
    )

    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": ["00360", "00360", "00180"],
            "sides": ["4.0", None, "3.0"],
            "date": ["2020-01-01", "2021-01-01", "2022-01-01"],
        },
        dtype="string",
    )

    tm.assert_frame_equal(df_result, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_dtypes_with_names(parser, temp_file):
    # [FUNCTION DOC]
    # Reads the dates data with custom column names and a per-column dtype mapping
    # (string, nullable integer, datetime). Expects renamed columns with those
    # dtypes, the sides missing value becoming NA, for both reading modes.
    df_result = read_xml(
        StringIO(xml_dates),
        names=["Col1", "Col2", "Col3", "Col4"],
        dtype={"Col2": "string", "Col3": "Int64", "Col4": "datetime64[ns]"},
        parser=parser,
    )
    df_iter = read_xml_iterparse(
        xml_dates,
        parser=parser,
        names=["Col1", "Col2", "Col3", "Col4"],
        dtype={"Col2": "string", "Col3": "Int64", "Col4": "datetime64[ns]"},
        iterparse={"row": ["shape", "degrees", "sides", "date"]},
        temp_file=temp_file,
    )

    df_expected = DataFrame(
        {
            "Col1": ["square", "circle", "triangle"],
            "Col2": Series(["00360", "00360", "00180"], dtype="string"),
            "Col3": Series([4.0, NA, 3.0], dtype="Int64"),
            "Col4": DatetimeIndex(
                ["2020-01-01", "2021-01-01", "2022-01-01"], dtype="M8[ns]"
            ),
        }
    )

    tm.assert_frame_equal(df_result, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_dtype_nullable_int(parser, temp_file):
    # [FUNCTION DOC]
    # Reads the types data requesting nullable integer dtype for sides. Expects
    # sides as nullable integers with a missing value and degrees parsed as plain
    # integers, for both reading modes.
    df_result = read_xml(StringIO(xml_types), dtype={"sides": "Int64"}, parser=parser)
    df_iter = read_xml_iterparse(
        xml_types,
        parser=parser,
        dtype={"sides": "Int64"},
        iterparse={"row": ["shape", "degrees", "sides"]},
        temp_file=temp_file,
    )

    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": [360, 360, 180],
            "sides": Series([4.0, NA, 3.0], dtype="Int64"),
        }
    )

    tm.assert_frame_equal(df_result, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_dtype_float(parser, temp_file):
    # [FUNCTION DOC]
    # Reads the types data requesting float dtype for degrees. Expects degrees as
    # floats, sides unchanged, for both reading modes.
    df_result = read_xml(StringIO(xml_types), dtype={"degrees": "float"}, parser=parser)
    df_iter = read_xml_iterparse(
        xml_types,
        parser=parser,
        dtype={"degrees": "float"},
        iterparse={"row": ["shape", "degrees", "sides"]},
        temp_file=temp_file,
    )

    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": Series([360, 360, 180]).astype("float"),
            "sides": [4.0, float("nan"), 3.0],
        }
    )

    tm.assert_frame_equal(df_result, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_wrong_dtype(xml_books, parser, iterparse):
    # [FUNCTION DOC]
    # Requests integer dtype for the title column of the books data. Expects a
    # value error stating that the string cannot be parsed, for each parser and
    # iterparse option combination.
    with pytest.raises(
        ValueError, match=('Unable to parse string "Everyday Italian" at position 0')
    ):
        read_xml(
            xml_books, dtype={"title": "Int64"}, parser=parser, iterparse=iterparse
        )


def test_both_dtype_converters(parser, temp_file):
    # [FUNCTION DOC]
    # Passes both a dtype and a converter for the same column. Expects a parser
    # warning saying both were given; the converter wins so degrees stay strings.
    # The assertions on the results are executed inside the warning context.
    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": ["00360", "00360", "00180"],
            "sides": [4.0, float("nan"), 3.0],
        }
    )

    with tm.assert_produces_warning(ParserWarning, match="Both a converter and dtype"):
        df_result = read_xml(
            StringIO(xml_types),
            dtype={"degrees": "str"},
            converters={"degrees": str},
            parser=parser,
        )
        df_iter = read_xml_iterparse(
            xml_types,
            dtype={"degrees": "str"},
            converters={"degrees": str},
            parser=parser,
            iterparse={"row": ["shape", "degrees", "sides"]},
            temp_file=temp_file,
        )

        tm.assert_frame_equal(df_result, df_expected)
        tm.assert_frame_equal(df_iter, df_expected)


# CONVERTERS


def test_converters_str(parser, temp_file):
    # [FUNCTION DOC]
    # Reads the types data with a converter turning degrees into strings. Expects
    # degrees kept as text with leading zeros, for both reading modes.
    df_result = read_xml(
        StringIO(xml_types), converters={"degrees": str}, parser=parser
    )
    df_iter = read_xml_iterparse(
        xml_types,
        parser=parser,
        converters={"degrees": str},
        iterparse={"row": ["shape", "degrees", "sides"]},
        temp_file=temp_file,
    )

    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": ["00360", "00360", "00180"],
            "sides": [4.0, float("nan"), 3.0],
        }
    )

    tm.assert_frame_equal(df_result, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_converters_date(parser, temp_file):
    # [FUNCTION DOC]
    # Reads the dates data with a converter that converts the date column to
    # datetimes. Expects a datetime date column and integer degrees, for both modes.
    convert_to_datetime = lambda x: to_datetime(x)
    df_result = read_xml(
        StringIO(xml_dates), converters={"date": convert_to_datetime}, parser=parser
    )
    df_iter = read_xml_iterparse(
        xml_dates,
        parser=parser,
        converters={"date": convert_to_datetime},
        iterparse={"row": ["shape", "degrees", "sides", "date"]},
        temp_file=temp_file,
    )

    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": [360, 360, 180],
            "sides": [4.0, float("nan"), 3.0],
            "date": to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
        }
    )

    tm.assert_frame_equal(df_result, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_wrong_converters_type(xml_books, parser, iterparse):
    # [FUNCTION DOC]
    # Passes converters as a set instead of a mapping. Expects a type error saying
    # converters must be a dict or subclass.
    with pytest.raises(TypeError, match=("Type converters must be a dict or subclass")):
        read_xml(
            xml_books, converters={"year", str}, parser=parser, iterparse=iterparse
        )


def test_callable_func_converters(xml_books, parser, iterparse):
    # [FUNCTION DOC]
    # Passes a converter that is a float instance rather than a callable. Expects a
    # type error saying the float object is not callable.
    with pytest.raises(TypeError, match=("'float' object is not callable")):
        read_xml(
            xml_books, converters={"year": float()}, parser=parser, iterparse=iterparse
        )


def test_callable_str_converters(xml_books, parser, iterparse):
    # [FUNCTION DOC]
    # Passes a converter given as a string name instead of a callable. Expects a
    # type error saying the str object is not callable.
    with pytest.raises(TypeError, match=("'str' object is not callable")):
        read_xml(
            xml_books, converters={"year": "float"}, parser=parser, iterparse=iterparse
        )


# PARSE DATES


def test_parse_dates_column_name(parser, temp_file):
    # [FUNCTION DOC]
    # Parses dates by giving the column name in the parse_dates list. Expects the
    # date column as datetimes, for both reading modes.
    df_result = read_xml(StringIO(xml_dates), parse_dates=["date"], parser=parser)
    df_iter = read_xml_iterparse(
        xml_dates,
        parser=parser,
        parse_dates=["date"],
        iterparse={"row": ["shape", "degrees", "sides", "date"]},
        temp_file=temp_file,
    )

    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": [360, 360, 180],
            "sides": [4.0, float("nan"), 3.0],
            "date": to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
        }
    )

    tm.assert_frame_equal(df_result, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_parse_dates_column_index(parser, temp_file):
    # [FUNCTION DOC]
    # Parses dates by giving the column position in the parse_dates list. Expects
    # the fourth column converted to datetimes, for both reading modes.
    df_result = read_xml(StringIO(xml_dates), parse_dates=[3], parser=parser)
    df_iter = read_xml_iterparse(
        xml_dates,
        parser=parser,
        parse_dates=[3],
        iterparse={"row": ["shape", "degrees", "sides", "date"]},
        temp_file=temp_file,
    )

    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": [360, 360, 180],
            "sides": [4.0, float("nan"), 3.0],
            "date": to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
        }
    )

    tm.assert_frame_equal(df_result, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_parse_dates_true(parser, temp_file):
    # [FUNCTION DOC]
    # Passes parse_dates as true. Expects the date column to be left as strings,
    # since true only requests index parsing; both reading modes agree.
    df_result = read_xml(StringIO(xml_dates), parse_dates=True, parser=parser)

    df_iter = read_xml_iterparse(
        xml_dates,
        parser=parser,
        parse_dates=True,
        iterparse={"row": ["shape", "degrees", "sides", "date"]},
        temp_file=temp_file,
    )

    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": [360, 360, 180],
            "sides": [4.0, float("nan"), 3.0],
            "date": ["2020-01-01", "2021-01-01", "2022-01-01"],
        }
    )

    tm.assert_frame_equal(df_result, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_day_first_parse_dates(parser, temp_file):
    # [FUNCTION DOC]
    # Reads dates written day first with the date column requested in parse_dates.
    # Expects a user warning about the day-first format being inferred and the
    # dates parsed as the 31st of December in each year, for both reading modes.
    xml = """\
<?xml version='1.0' encoding='utf-8'?>
<data>
  <row>
    <shape>square</shape>
    <degrees>00360</degrees>
    <sides>4.0</sides>
    <date>31/12/2020</date>
   </row>
  <row>
    <shape>circle</shape>
    <degrees>00360</degrees>
    <sides/>
    <date>31/12/2021</date>
  </row>
  <row>
    <shape>triangle</shape>
    <degrees>00180</degrees>
    <sides>3.0</sides>
    <date>31/12/2022</date>
  </row>
</data>"""

    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": [360, 360, 180],
            "sides": [4.0, float("nan"), 3.0],
            "date": to_datetime(["2020-12-31", "2021-12-31", "2022-12-31"]),
        }
    )

    with tm.assert_produces_warning(
        UserWarning, match="Parsing dates in %d/%m/%Y format"
    ):
        df_result = read_xml(StringIO(xml), parse_dates=["date"], parser=parser)
        df_iter = read_xml_iterparse(
            xml,
            parse_dates=["date"],
            parser=parser,
            iterparse={"row": ["shape", "degrees", "sides", "date"]},
            temp_file=temp_file,
        )

        tm.assert_frame_equal(df_result, df_expected)
        tm.assert_frame_equal(df_iter, df_expected)


def test_wrong_parse_dates_type(xml_books, parser, iterparse):
    # [FUNCTION DOC]
    # Passes parse_dates as a set rather than a boolean or list. Expects a type
    # error stating that only booleans and lists are accepted.
    with pytest.raises(TypeError, match="Only booleans and lists are accepted"):
        read_xml(xml_books, parse_dates={"date"}, parser=parser, iterparse=iterparse)
