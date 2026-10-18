"""Tests for ExtensionDtype Table Schema integration."""

# [MODULE DOC]
# Tests for Table Schema JSON integration with extension dtypes: schema
# building, dtype to JSON type mapping, table-orient serialization and
# round-trip reading. Uses the test-only date and decimal extension arrays plus
# the built-in string and nullable integer dtypes; exercises the private table
# schema module's type mapping and schema builder.

from collections import OrderedDict
import datetime as dt
import decimal
from io import StringIO
import json

import pytest

from pandas import (
    NA,
    DataFrame,
    Index,
    array,
    read_json,
)
import pandas._testing as tm
from pandas.core.arrays.integer import Int64Dtype
from pandas.core.arrays.string_ import StringDtype
from pandas.core.series import Series
from pandas.tests.extension.date import (
    DateArray,
    DateDtype,
)
from pandas.tests.extension.decimal.array import (
    DecimalArray,
    DecimalDtype,
)

from pandas.io.json._table_schema import (
    as_json_table_type,
    build_table_schema,
)


class TestBuildSchema:
    # [CLASS DOC]
    # Tests for the schema builder applied to frames containing extension arrays.
    def test_build_table_schema(self):
        # [FUNCTION DOC]
        # Builds a frame with date, decimal, string and nullable integer extension
        # columns and builds its schema without version. Expects an index field of
        # integer type followed by one field per column carrying the JSON type (any,
        # number, string, integer) and an extension dtype name (the date dtype, decimal,
        # string, Int64), with the index as primary key. Building again with the
        # default version setting must include the pandas version entry.
        df = DataFrame(
            {
                "A": DateArray([dt.date(2021, 10, 10)]),
                "B": DecimalArray([decimal.Decimal(10)]),
                "C": array(["pandas"], dtype="string"),
                "D": array([10], dtype="Int64"),
            }
        )
        result = build_table_schema(df, version=False)
        expected = {
            "fields": [
                {"name": "index", "type": "integer"},
                {"name": "A", "type": "any", "extDtype": "DateDtype"},
                {"name": "B", "type": "number", "extDtype": "decimal"},
                {"name": "C", "type": "string", "extDtype": "string"},
                {"name": "D", "type": "integer", "extDtype": "Int64"},
            ],
            "primaryKey": ["index"],
        }
        assert result == expected
        result = build_table_schema(df)
        assert "pandas_version" in result


class TestTableSchemaType:
    @pytest.mark.parametrize("box", [lambda x: x, Series])
    # [CLASS DOC]
    # Tests for the mapping from extension dtypes to Table Schema JSON type names.
    def test_as_json_table_type_ext_date_array_dtype(self, box):
        # [FUNCTION DOC]
        # Parametrized on boxing the date array as itself or in a series. The dtype
        # of the data must map to the generic any type.
        date_data = box(DateArray([dt.date(2021, 10, 10)]))
        assert as_json_table_type(date_data.dtype) == "any"

    def test_as_json_table_type_ext_date_dtype(self):
        # [FUNCTION DOC]
        # The date extension dtype instance maps to the any type.
        assert as_json_table_type(DateDtype()) == "any"

    @pytest.mark.parametrize("box", [lambda x: x, Series])
    def test_as_json_table_type_ext_decimal_array_dtype(self, box):
        # [FUNCTION DOC]
        # Parametrized on array versus series boxing; a decimal array dtype maps to the
        # number type.
        decimal_data = box(DecimalArray([decimal.Decimal(10)]))
        assert as_json_table_type(decimal_data.dtype) == "number"

    def test_as_json_table_type_ext_decimal_dtype(self):
        # [FUNCTION DOC]
        # The decimal extension dtype instance maps to the number type.
        assert as_json_table_type(DecimalDtype()) == "number"

    @pytest.mark.parametrize("box", [lambda x: x, Series])
    def test_as_json_table_type_ext_string_array_dtype(self, box):
        # [FUNCTION DOC]
        # Parametrized on array versus series boxing; a string-dtype array maps to the
        # string type.
        string_data = box(array(["pandas"], dtype="string"))
        assert as_json_table_type(string_data.dtype) == "string"

    def test_as_json_table_type_ext_string_dtype(self):
        # [FUNCTION DOC]
        # The string dtype instance maps to the string type.
        assert as_json_table_type(StringDtype()) == "string"

    @pytest.mark.parametrize("box", [lambda x: x, Series])
    def test_as_json_table_type_ext_integer_array_dtype(self, box):
        # [FUNCTION DOC]
        # Parametrized on array versus series boxing; a nullable Int64 array dtype maps
        # to the integer type.
        integer_data = box(array([10], dtype="Int64"))
        assert as_json_table_type(integer_data.dtype) == "integer"

    def test_as_json_table_type_ext_integer_dtype(self):
        # [FUNCTION DOC]
        # The nullable Int64 dtype instance maps to the integer type.
        assert as_json_table_type(Int64Dtype()) == "integer"


class TestTableOrient:
    @pytest.fixture
    # [CLASS DOC]
    # Tests for writing and reading extension-dtype data with the table
    # orientation. Four fixtures provide one-element extension arrays (date,
    # decimal, string, Int64) shared by the tests.
    def da(self):
        # [FUNCTION DOC]
        # Fixture returning a date array holding a single date (2021-10-10).
        """Fixture for creating a DateArray."""
        return DateArray([dt.date(2021, 10, 10)])

    @pytest.fixture
    def dc(self):
        # [FUNCTION DOC]
        # Fixture returning a decimal array holding the single value ten.
        """Fixture for creating a DecimalArray."""
        return DecimalArray([decimal.Decimal(10)])

    @pytest.fixture
    def sa(self):
        # [FUNCTION DOC]
        # Fixture returning a string-dtype array holding the single value pandas.
        """Fixture for creating a StringDtype array."""
        return array(["pandas"], dtype="string")

    @pytest.fixture
    def ia(self):
        # [FUNCTION DOC]
        # Fixture returning a nullable Int64 array holding the single value ten.
        """Fixture for creating an Int64Dtype array."""
        return array([10], dtype="Int64")

    def test_build_date_series(self, da):
        # [FUNCTION DOC]
        # Wraps the date array in a named series with a named index, serializes with
        # table orientation and ISO dates, parses the output preserving order, removes
        # the version entry from the schema, and expects an integer index field, an
        # any-typed field with the date extension dtype name, primary key on the index,
        # and one data row with the ISO timestamp string.
        s = Series(da, name="a")
        s.index.name = "id"
        result = s.to_json(orient="table", date_format="iso")
        result = json.loads(result, object_pairs_hook=OrderedDict)

        assert "pandas_version" in result["schema"]
        result["schema"].pop("pandas_version")

        fields = [
            {"name": "id", "type": "integer"},
            {"name": "a", "type": "any", "extDtype": "DateDtype"},
        ]

        schema = {"fields": fields, "primaryKey": ["id"]}

        expected = OrderedDict(
            [
                ("schema", schema),
                ("data", [OrderedDict([("id", 0), ("a", "2021-10-10T00:00:00.000")])]),
            ]
        )

        assert result == expected

    def test_build_decimal_series(self, dc):
        # [FUNCTION DOC]
        # Same procedure as the date case for the decimal array: the value field is of
        # number type with extension dtype decimal, and the data value is serialized as
        # the string ten.
        s = Series(dc, name="a")
        s.index.name = "id"
        result = s.to_json(orient="table", date_format="iso")
        result = json.loads(result, object_pairs_hook=OrderedDict)

        assert "pandas_version" in result["schema"]
        result["schema"].pop("pandas_version")

        fields = [
            {"name": "id", "type": "integer"},
            {"name": "a", "type": "number", "extDtype": "decimal"},
        ]

        schema = {"fields": fields, "primaryKey": ["id"]}

        expected = OrderedDict(
            [
                ("schema", schema),
                ("data", [OrderedDict([("id", 0), ("a", "10")])]),
            ]
        )

        assert result == expected

    def test_build_string_series(self, sa):
        # [FUNCTION DOC]
        # Same procedure for the string array: the value field is string type with
        # extension dtype string and the data value is the text pandas.
        s = Series(sa, name="a")
        s.index.name = "id"
        result = s.to_json(orient="table", date_format="iso")
        result = json.loads(result, object_pairs_hook=OrderedDict)

        assert "pandas_version" in result["schema"]
        result["schema"].pop("pandas_version")

        fields = [
            {"name": "id", "type": "integer"},
            {"name": "a", "type": "string", "extDtype": "string"},
        ]

        schema = {"fields": fields, "primaryKey": ["id"]}

        expected = OrderedDict(
            [
                ("schema", schema),
                ("data", [OrderedDict([("id", 0), ("a", "pandas")])]),
            ]
        )

        assert result == expected

    def test_build_int64_series(self, ia):
        # [FUNCTION DOC]
        # Same procedure for the Int64 array: the field is integer type with extension
        # dtype Int64 and the data value is the number ten.
        s = Series(ia, name="a")
        s.index.name = "id"
        result = s.to_json(orient="table", date_format="iso")
        result = json.loads(result, object_pairs_hook=OrderedDict)

        assert "pandas_version" in result["schema"]
        result["schema"].pop("pandas_version")

        fields = [
            {"name": "id", "type": "integer"},
            {"name": "a", "type": "integer", "extDtype": "Int64"},
        ]

        schema = {"fields": fields, "primaryKey": ["id"]}

        expected = OrderedDict(
            [
                ("schema", schema),
                ("data", [OrderedDict([("id", 0), ("a", 10)])]),
            ]
        )

        assert result == expected

    def test_to_json(self, da, dc, sa, ia):
        # [FUNCTION DOC]
        # Builds a four-column frame from the four fixtures with a named index,
        # serializes with table orientation and ISO dates, parses the output, drops the
        # version, and compares against the expected ordered schema (index field plus the
        # four typed fields with extension dtype names, primary key the index) and one
        # data row holding the index and all four values.
        df = DataFrame(
            {
                "A": da,
                "B": dc,
                "C": sa,
                "D": ia,
            }
        )
        df.index.name = "idx"
        result = df.to_json(orient="table", date_format="iso")
        result = json.loads(result, object_pairs_hook=OrderedDict)

        assert "pandas_version" in result["schema"]
        result["schema"].pop("pandas_version")

        fields = [
            OrderedDict({"name": "idx", "type": "integer"}),
            OrderedDict({"name": "A", "type": "any", "extDtype": "DateDtype"}),
            OrderedDict({"name": "B", "type": "number", "extDtype": "decimal"}),
            OrderedDict({"name": "C", "type": "string", "extDtype": "string"}),
            OrderedDict({"name": "D", "type": "integer", "extDtype": "Int64"}),
        ]

        schema = OrderedDict({"fields": fields, "primaryKey": ["idx"]})
        data = [
            OrderedDict(
                [
                    ("idx", 0),
                    ("A", "2021-10-10T00:00:00.000"),
                    ("B", "10"),
                    ("C", "pandas"),
                    ("D", 10),
                ]
            )
        ]
        expected = OrderedDict([("schema", schema), ("data", data)])

        assert result == expected

    def test_json_ext_dtype_reading_roundtrip(self):
        # [FUNCTION DOC]
        # GH#40255. Builds a frame with nullable Int64, Float64 and boolean columns
        # containing a missing value and a nullable Int64 index with a missing label,
        # writes it with table orientation and indentation, and reads it back with table
        # orientation. The result must equal the original frame, preserving dtypes.
        # GH#40255
        df = DataFrame(
            {
                "a": Series([2, NA], dtype="Int64"),
                "b": Series([1.5, NA], dtype="Float64"),
                "c": Series([True, NA], dtype="boolean"),
            },
            index=Index([1, NA], dtype="Int64"),
        )
        expected = df.copy()
        data_json = df.to_json(orient="table", indent=4)
        result = read_json(StringIO(data_json), orient="table")
        tm.assert_frame_equal(result, expected)

    def test_json_ext_dtype_reading(self):
        # [FUNCTION DOC]
        # GH#40255. Reads a hand-written table-orient JSON string whose schema declares
        # one integer field with extension dtype Int64 and whose data has a value and a
        # null. The result must be a one-column frame with Int64 dtype containing the
        # value and a missing entry.
        # GH#40255
        data_json = """{
            "schema":{
                "fields":[
                    {
                        "name":"a",
                        "type":"integer",
                        "extDtype":"Int64"
                    }
                ],
            },
            "data":[
                {
                    "a":2
                },
                {
                    "a":null
                }
            ]
        }"""
        result = read_json(StringIO(data_json), orient="table")
        expected = DataFrame({"a": Series([2, NA], dtype="Int64")})
        tm.assert_frame_equal(result, expected)
