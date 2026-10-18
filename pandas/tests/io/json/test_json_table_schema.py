"""Tests for Table Schema integration."""

# [MODULE DOC]
# Tests for the Table Schema JSON integration: the schema builder, the mapping of
# pandas dtypes to Table Schema type names (and back), table-orient writing
# via the frame and series writers, and table-orient reading round trips. Uses
# the private table schema module's functions (type mapping, schema building,
# field conversion in both directions, default name assignment). Extension-dtype
# cases live in a sibling module. Two frame fixtures are defined.

from collections import OrderedDict
from io import StringIO
import json

import numpy as np
import pytest

from pandas.errors import Pandas4Warning

from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    PeriodDtype,
)

import pandas as pd
from pandas import DataFrame
import pandas._testing as tm

from pandas.io.json._table_schema import (
    as_json_table_type,
    build_table_schema,
    convert_json_field_to_pandas_type,
    convert_pandas_type_to_json_field,
    set_default_names,
)


@pytest.fixture
def df_schema():
    # [FUNCTION DOC]
    # HELPER: fixture returning a four-row frame with an integer, string, datetime
    # and timedelta column and an integer index named idx, used to test schema
    # building.
    return DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": ["a", "b", "c", "c"],
            "C": pd.date_range("2016-01-01", freq="D", periods=4),
            "D": pd.timedelta_range("1h", periods=4, freq="min"),
        },
        index=pd.Index(range(4), name="idx"),
    )


@pytest.fixture
def df_table():
    # [FUNCTION DOC]
    # HELPER: fixture returning a four-row frame like the schema fixture plus
    # unordered and ordered categorical columns, a float column and a
    # timezone-aware datetime column (US/Central), with an index named idx; used for
    # table-orient writing tests.
    return DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": ["a", "b", "c", "c"],
            "C": pd.date_range("2016-01-01", freq="D", periods=4),
            "D": pd.timedelta_range("1h", periods=4, freq="min"),
            "E": pd.Series(pd.Categorical(["a", "b", "c", "c"])),
            "F": pd.Series(pd.Categorical(["a", "b", "c", "c"], ordered=True)),
            "G": [1.0, 2.0, 3, 4.0],
            "H": pd.date_range("2016-01-01", freq="D", periods=4, tz="US/Central"),
        },
        index=pd.Index(range(4), name="idx"),
    )


class TestBuildSchema:
    # [CLASS DOC]
    # Tests for the schema builder on frames, series and multi-indexes.
    def test_build_table_schema(self, df_schema, using_infer_string):
        # [FUNCTION DOC]
        # Builds the schema of the schema fixture without a version: an index field plus
        # one field per column with JSON types integer, string, datetime and duration,
        # and a primary key on the index. When string inference is active the string
        # column field additionally carries an extension dtype entry naming the
        # string dtype. Building with the default version setting must include the
        # version entry.
        result = build_table_schema(df_schema, version=False)
        expected = {
            "fields": [
                {"name": "idx", "type": "integer"},
                {"name": "A", "type": "integer"},
                {"name": "B", "type": "string"},
                {"name": "C", "type": "datetime"},
                {"name": "D", "type": "duration"},
            ],
            "primaryKey": ["idx"],
        }
        if using_infer_string:
            expected["fields"][2] = {"name": "B", "type": "string", "extDtype": "str"}
        assert result == expected
        result = build_table_schema(df_schema)
        assert "pandas_version" in result

    def test_series(self):
        # [FUNCTION DOC]
        # A named integer series gives fields for the index and the named value, with the
        # primary key being the index; the version is included by default.
        s = pd.Series([1, 2, 3], name="foo")
        result = build_table_schema(s, version=False)
        expected = {
            "fields": [
                {"name": "index", "type": "integer"},
                {"name": "foo", "type": "integer"},
            ],
            "primaryKey": ["index"],
        }
        assert result == expected
        result = build_table_schema(s)
        assert "pandas_version" in result

    def test_series_unnamed(self):
        # [FUNCTION DOC]
        # An unnamed series uses the field name values for the data.
        result = build_table_schema(pd.Series([1, 2, 3]), version=False)
        expected = {
            "fields": [
                {"name": "index", "type": "integer"},
                {"name": "values", "type": "integer"},
            ],
            "primaryKey": ["index"],
        }
        assert result == expected

    def test_multiindex(self, df_schema, using_infer_string):
        # [FUNCTION DOC]
        # A two-level unnamed index gets generated level names and both levels are in the
        # primary key (with string-inference adjustments for the string level and the
        # string column). After naming the first level and leaving the second unnamed,
        # the first field takes the given name while the second keeps the generated name.
        df = df_schema
        idx = pd.MultiIndex.from_product([("a", "b"), (1, 2)])
        df.index = idx

        result = build_table_schema(df, version=False)
        expected = {
            "fields": [
                {"name": "level_0", "type": "string"},
                {"name": "level_1", "type": "integer"},
                {"name": "A", "type": "integer"},
                {"name": "B", "type": "string"},
                {"name": "C", "type": "datetime"},
                {"name": "D", "type": "duration"},
            ],
            "primaryKey": ["level_0", "level_1"],
        }
        if using_infer_string:
            expected["fields"][0] = {
                "name": "level_0",
                "type": "string",
                "extDtype": "str",
            }
            expected["fields"][3] = {"name": "B", "type": "string", "extDtype": "str"}
        assert result == expected

        df.index.names = ["idx0", None]
        expected["fields"][0]["name"] = "idx0"
        expected["primaryKey"] = ["idx0", "level_1"]
        result = build_table_schema(df, version=False)
        assert result == expected


class TestTableSchemaType:
    @pytest.mark.parametrize("int_type", [int, np.int16, np.int32, np.int64])
    # [CLASS DOC]
    # Tests for mapping dtypes and data to Table Schema type names.
    def test_as_json_table_type_int_data(self, int_type):
        # [FUNCTION DOC]
        # Parametrized on integer types. The dtype of an integer array maps to integer.
        int_data = [1, 2, 3]
        assert as_json_table_type(np.array(int_data, dtype=int_type).dtype) == "integer"

    @pytest.mark.parametrize("float_type", [float, np.float16, np.float32, np.float64])
    def test_as_json_table_type_float_data(self, float_type):
        # [FUNCTION DOC]
        # Parametrized on float types. The dtype of a float array maps to number.
        float_data = [1.0, 2.0, 3.0]
        assert (
            as_json_table_type(np.array(float_data, dtype=float_type).dtype) == "number"
        )

    @pytest.mark.parametrize("bool_type", [bool, np.bool_])
    def test_as_json_table_type_bool_data(self, bool_type):
        # [FUNCTION DOC]
        # Parametrized on boolean types. The dtype maps to boolean.
        bool_data = [True, False]
        assert (
            as_json_table_type(np.array(bool_data, dtype=bool_type).dtype) == "boolean"
        )

    @pytest.mark.parametrize(
        "date_data",
        [
            pd.to_datetime(["2016"]),
            pd.to_datetime(["2016"], utc=True),
            pd.Series(pd.to_datetime(["2016"])),
            pd.Series(pd.to_datetime(["2016"], utc=True)),
            pd.period_range("2016", freq="Y", periods=3),
        ],
    )
    def test_as_json_table_type_date_data(self, date_data):
        # [FUNCTION DOC]
        # Parametrized on datetime index and series, with and without UTC, and a period
        # range. The dtype maps to datetime.
        assert as_json_table_type(date_data.dtype) == "datetime"

    @pytest.mark.parametrize("klass", [pd.Series, pd.Index])
    def test_as_json_table_type_string_data(self, klass):
        # [FUNCTION DOC]
        # Parametrized on series and index of object dtype strings; maps to string.
        str_data = klass(["a", "b"], dtype=object)
        assert as_json_table_type(str_data.dtype) == "string"

    @pytest.mark.parametrize(
        "cat_data",
        [
            pd.Categorical(["a"]),
            pd.Categorical([1]),
            pd.Series(pd.Categorical([1])),
            pd.CategoricalIndex([1]),
        ],
    )
    def test_as_json_table_type_categorical_data(self, cat_data):
        # [FUNCTION DOC]
        # Parametrized on categorical array, series and index; maps to any.
        assert as_json_table_type(cat_data.dtype) == "any"

    # ------
    # dtypes
    # ------
    @pytest.mark.parametrize("int_dtype", [int, np.int16, np.int32, np.int64])
    def test_as_json_table_type_int_dtypes(self, int_dtype):
        # [FUNCTION DOC]
        # Parametrized on integer dtype specifiers passed directly; maps to integer.
        assert as_json_table_type(int_dtype) == "integer"

    @pytest.mark.parametrize("float_dtype", [float, np.float16, np.float32, np.float64])
    def test_as_json_table_type_float_dtypes(self, float_dtype):
        # [FUNCTION DOC]
        # Parametrized on float dtype specifiers; maps to number.
        assert as_json_table_type(float_dtype) == "number"

    @pytest.mark.parametrize("bool_dtype", [bool, np.bool_])
    def test_as_json_table_type_bool_dtypes(self, bool_dtype):
        # [FUNCTION DOC]
        # Parametrized on boolean dtype specifiers; maps to boolean.
        assert as_json_table_type(bool_dtype) == "boolean"

    @pytest.mark.parametrize(
        "date_dtype",
        [
            np.dtype("<M8[ns]"),
            PeriodDtype("D"),
            DatetimeTZDtype("ns", "US/Central"),
        ],
    )
    def test_as_json_table_type_date_dtypes(self, date_dtype):
        # [FUNCTION DOC]
        # Parametrized on a numpy datetime dtype, a period dtype and a timezone-aware
        # dtype; all map to datetime. A TODO notes that plain date and time objects are
        # not covered.
        # TODO: datedate.date? datetime.time?
        assert as_json_table_type(date_dtype) == "datetime"

    @pytest.mark.parametrize("td_dtype", [np.dtype("<m8[ns]")])
    def test_as_json_table_type_timedelta_dtypes(self, td_dtype):
        # [FUNCTION DOC]
        # The numpy nanosecond timedelta dtype maps to duration.
        assert as_json_table_type(td_dtype) == "duration"

    @pytest.mark.parametrize("str_dtype", [object])  # TODO(GH#14904) flesh out dtypes?
    def test_as_json_table_type_string_dtypes(self, str_dtype):
        # [FUNCTION DOC]
        # Parametrized on the object dtype only (a TODO references issue 14904 to extend
        # it); maps to string.
        assert as_json_table_type(str_dtype) == "string"

    def test_as_json_table_type_categorical_dtypes(self):
        # [FUNCTION DOC]
        # The categorical dtype of an array and a bare categorical dtype both map to any.
        assert as_json_table_type(pd.Categorical(["a"]).dtype) == "any"
        assert as_json_table_type(CategoricalDtype()) == "any"


class TestTableOrient:
    # [CLASS DOC]
    # Tests for table-orient serialization and for the field conversion helpers.
    def test_build_series(self):
        # [FUNCTION DOC]
        # A named integer series with a named index serialized with table orientation and
        # ISO dates: after dropping the version, the parsed output has the schema (index
        # and value fields, primary key) and one data record per element.
        s = pd.Series([1, 2], name="a")
        s.index.name = "id"
        result = s.to_json(orient="table", date_format="iso")
        result = json.loads(result, object_pairs_hook=OrderedDict)

        assert "pandas_version" in result["schema"]
        result["schema"].pop("pandas_version")

        fields = [{"name": "id", "type": "integer"}, {"name": "a", "type": "integer"}]

        schema = {"fields": fields, "primaryKey": ["id"]}

        expected = OrderedDict(
            [
                ("schema", schema),
                (
                    "data",
                    [
                        OrderedDict([("id", 0), ("a", 1)]),
                        OrderedDict([("id", 1), ("a", 2)]),
                    ],
                ),
            ]
        )

        assert result == expected

    def test_read_json_from_to_json_results(self):
        # [FUNCTION DOC]
        # GH32383. A frame with non-ASCII text values and row labels written with default
        # orientation reads back equal, both through the JSON reader and through
        # building a frame from the parsed dictionary.
        # GH32383
        df = DataFrame(
            {
                "_id": {"row_0": 0},
                "category": {"row_0": "Goods"},
                "recommender_id": {"row_0": 3},
                "recommender_name_jp": {"row_0": "浦田"},
                "recommender_name_en": {"row_0": "Urata"},
                "name_jp": {"row_0": "博多人形(松尾吉将まつお よしまさ)"},
                "name_en": {"row_0": "Hakata Dolls Matsuo"},
            }
        )

        result1 = pd.read_json(StringIO(df.to_json()))
        result2 = DataFrame.from_dict(json.loads(df.to_json()))
        tm.assert_frame_equal(result1, df)
        tm.assert_frame_equal(result2, df)

    def test_to_json(self, df_table, using_infer_string):
        # [FUNCTION DOC]
        # Serializes the table fixture with table orientation and ISO dates, parses it
        # preserving order, drops the version and compares to the expected schema and
        # data. Expected fields: integer, string (with extension dtype under string
        # inference), datetime, duration, unordered and ordered categorical with enum
        # constraints, number, and timezone-aware datetime with a time zone entry. Data
        # rows give ISO timestamps (timezone-aware ones converted to UTC with a trailing
        # Z) and ISO durations.
        df = df_table
        df.index.name = "idx"
        result = df.to_json(orient="table", date_format="iso")
        result = json.loads(result, object_pairs_hook=OrderedDict)

        assert "pandas_version" in result["schema"]
        result["schema"].pop("pandas_version")

        fields = [
            {"name": "idx", "type": "integer"},
            {"name": "A", "type": "integer"},
            {"name": "B", "type": "string"},
            {"name": "C", "type": "datetime"},
            {"name": "D", "type": "duration"},
            {
                "constraints": {"enum": ["a", "b", "c"]},
                "name": "E",
                "ordered": False,
                "type": "any",
            },
            {
                "constraints": {"enum": ["a", "b", "c"]},
                "name": "F",
                "ordered": True,
                "type": "any",
            },
            {"name": "G", "type": "number"},
            {"name": "H", "type": "datetime", "tz": "US/Central"},
        ]

        if using_infer_string:
            fields[2] = {"name": "B", "type": "string", "extDtype": "str"}

        schema = {"fields": fields, "primaryKey": ["idx"]}
        data = [
            OrderedDict(
                [
                    ("idx", 0),
                    ("A", 1),
                    ("B", "a"),
                    ("C", "2016-01-01T00:00:00.000"),
                    ("D", "P0DT1H0M0S"),
                    ("E", "a"),
                    ("F", "a"),
                    ("G", 1.0),
                    ("H", "2016-01-01T06:00:00.000Z"),
                ]
            ),
            OrderedDict(
                [
                    ("idx", 1),
                    ("A", 2),
                    ("B", "b"),
                    ("C", "2016-01-02T00:00:00.000"),
                    ("D", "P0DT1H1M0S"),
                    ("E", "b"),
                    ("F", "b"),
                    ("G", 2.0),
                    ("H", "2016-01-02T06:00:00.000Z"),
                ]
            ),
            OrderedDict(
                [
                    ("idx", 2),
                    ("A", 3),
                    ("B", "c"),
                    ("C", "2016-01-03T00:00:00.000"),
                    ("D", "P0DT1H2M0S"),
                    ("E", "c"),
                    ("F", "c"),
                    ("G", 3.0),
                    ("H", "2016-01-03T06:00:00.000Z"),
                ]
            ),
            OrderedDict(
                [
                    ("idx", 3),
                    ("A", 4),
                    ("B", "c"),
                    ("C", "2016-01-04T00:00:00.000"),
                    ("D", "P0DT1H3M0S"),
                    ("E", "c"),
                    ("F", "c"),
                    ("G", 4.0),
                    ("H", "2016-01-04T06:00:00.000Z"),
                ]
            ),
        ]
        expected = OrderedDict([("schema", schema), ("data", data)])

        assert result == expected

    def test_to_json_float_index(self):
        # [FUNCTION DOC]
        # A series with a float index serializes the index as number type with values.
        data = pd.Series(1, index=[1.0, 2.0])
        result = data.to_json(orient="table", date_format="iso")
        result = json.loads(result, object_pairs_hook=OrderedDict)
        result["schema"].pop("pandas_version")

        expected = OrderedDict(
            [
                (
                    "schema",
                    {
                        "fields": [
                            {"name": "index", "type": "number"},
                            {"name": "values", "type": "integer"},
                        ],
                        "primaryKey": ["index"],
                    },
                ),
                (
                    "data",
                    [
                        OrderedDict([("index", 1.0), ("values", 1)]),
                        OrderedDict([("index", 2.0), ("values", 1)]),
                    ],
                ),
            ]
        )

        assert result == expected

    def test_to_json_period_index(self):
        # [FUNCTION DOC]
        # A series with a quarterly period index: the index field has datetime type and a
        # frequency entry (quarterly ending in January), and data shows the period start
        # timestamps.
        idx = pd.period_range("2016", freq="Q-JAN", periods=2)
        data = pd.Series(1, idx)
        result = data.to_json(orient="table", date_format="iso")
        result = json.loads(result, object_pairs_hook=OrderedDict)
        result["schema"].pop("pandas_version")

        fields = [
            {"freq": "QE-JAN", "name": "index", "type": "datetime"},
            {"name": "values", "type": "integer"},
        ]

        schema = {"fields": fields, "primaryKey": ["index"]}
        data = [
            OrderedDict([("index", "2015-11-01T00:00:00.000"), ("values", 1)]),
            OrderedDict([("index", "2016-02-01T00:00:00.000"), ("values", 1)]),
        ]
        expected = OrderedDict([("schema", schema), ("data", data)])

        assert result == expected

    def test_to_json_categorical_index(self):
        # [FUNCTION DOC]
        # A categorical index yields an any-typed field with an enum constraint of the
        # categories and an unordered flag; data shows the category values.
        data = pd.Series(1, pd.CategoricalIndex(["a", "b"]))
        result = data.to_json(orient="table", date_format="iso")
        result = json.loads(result, object_pairs_hook=OrderedDict)
        result["schema"].pop("pandas_version")

        expected = OrderedDict(
            [
                (
                    "schema",
                    {
                        "fields": [
                            {
                                "name": "index",
                                "type": "any",
                                "constraints": {"enum": ["a", "b"]},
                                "ordered": False,
                            },
                            {"name": "values", "type": "integer"},
                        ],
                        "primaryKey": ["index"],
                    },
                ),
                (
                    "data",
                    [
                        OrderedDict([("index", "a"), ("values", 1)]),
                        OrderedDict([("index", "b"), ("values", 1)]),
                    ],
                ),
            ]
        )

        assert result == expected

    def test_date_format_raises(self, df_table):
        # [FUNCTION DOC]
        # Writing with table orientation and epoch dates must raise a value error saying
        # table schema requires ISO dates, and (inside it) emit the version-4 deprecation
        # warning for the epoch format. ISO and the default date format must work.
        error_msg = (
            "Trying to write with `orient='table'` and `date_format='epoch'`. Table "
            "Schema requires dates to be formatted with `date_format='iso'`"
        )
        warning_msg = (
            "'epoch' date format is deprecated and will be removed in a future "
            "version, please use 'iso' date format instead."
        )
        with pytest.raises(ValueError, match=error_msg):
            with tm.assert_produces_warning(Pandas4Warning, match=warning_msg):
                df_table.to_json(orient="table", date_format="epoch")

        # others work
        df_table.to_json(orient="table", date_format="iso")
        df_table.to_json(orient="table")

    def test_convert_pandas_type_to_json_field_int(self, index_or_series):
        # [FUNCTION DOC]
        # Parametrized by an index-or-series fixture. Converting a named integer object
        # gives a field with that name and type integer.
        kind = index_or_series
        data = [1, 2, 3]
        result = convert_pandas_type_to_json_field(kind(data, name="name"))
        expected = {"name": "name", "type": "integer"}
        assert result == expected

    def test_convert_pandas_type_to_json_field_float(self, index_or_series):
        # [FUNCTION DOC]
        # Same for floats; type number.
        kind = index_or_series
        data = [1.0, 2.0, 3.0]
        result = convert_pandas_type_to_json_field(kind(data, name="name"))
        expected = {"name": "name", "type": "number"}
        assert result == expected

    @pytest.mark.parametrize(
        "dt_args,extra_exp", [({}, {}), ({"utc": True}, {"tz": "UTC"})]
    )
    @pytest.mark.parametrize("wrapper", [None, pd.Series])
    def test_convert_pandas_type_to_json_field_datetime(
        self, dt_args, extra_exp, wrapper
    ):
        # [FUNCTION DOC]
        # Parametrized on to_datetime arguments (naive or UTC, giving a time zone entry
        # UTC) and on wrapping in a series. Converted field has the default name values,
        # type datetime and, for UTC, a tz entry.
        data = [1.0, 2.0, 3.0]
        data = pd.to_datetime(data, **dt_args)
        if wrapper is pd.Series:
            data = pd.Series(data, name="values")
        result = convert_pandas_type_to_json_field(data)
        expected = {"name": "values", "type": "datetime"}
        expected.update(extra_exp)
        assert result == expected

    def test_convert_pandas_type_to_json_period_range(self):
        # [FUNCTION DOC]
        # A yearly period range converts to a datetime field with the default name and a
        # frequency entry for year-end December.
        arr = pd.period_range("2016", freq="Y-DEC", periods=4)
        result = convert_pandas_type_to_json_field(arr)
        expected = {"name": "values", "type": "datetime", "freq": "YE-DEC"}
        assert result == expected

    @pytest.mark.parametrize("kind", [pd.Categorical, pd.CategoricalIndex])
    @pytest.mark.parametrize("ordered", [True, False])
    def test_convert_pandas_type_to_json_field_categorical(self, kind, ordered):
        # [FUNCTION DOC]
        # Parametrized on categorical series versus categorical index and on ordered or
        # not. The field has type any, an enum constraint with the categories and the
        # ordered flag.
        data = ["a", "b", "c"]
        if kind is pd.Categorical:
            arr = pd.Series(kind(data, ordered=ordered), name="cats")
        elif kind is pd.CategoricalIndex:
            arr = kind(data, ordered=ordered, name="cats")

        result = convert_pandas_type_to_json_field(arr)
        expected = {
            "name": "cats",
            "type": "any",
            "constraints": {"enum": data},
            "ordered": ordered,
        }
        assert result == expected

    @pytest.mark.parametrize(
        "inp,exp",
        [
            ({"type": "integer"}, "int64"),
            ({"type": "number"}, "float64"),
            ({"type": "boolean"}, "bool"),
            ({"type": "duration"}, "timedelta64"),
            ({"type": "datetime"}, "datetime64[ns]"),
            ({"type": "datetime", "tz": "US/Hawaii"}, "datetime64[ns, US/Hawaii]"),
            ({"type": "any"}, "object"),
            (
                {
                    "type": "any",
                    "constraints": {"enum": ["a", "b", "c"]},
                    "ordered": False,
                },
                CategoricalDtype(categories=["a", "b", "c"], ordered=False),
            ),
            (
                {
                    "type": "any",
                    "constraints": {"enum": ["a", "b", "c"]},
                    "ordered": True,
                },
                CategoricalDtype(categories=["a", "b", "c"], ordered=True),
            ),
            ({"type": "string"}, None),
        ],
    )
    def test_convert_json_field_to_pandas_type(self, inp, exp):
        # [FUNCTION DOC]
        # Parametrized on JSON field descriptions and expected pandas dtype names:
        # integer, number, boolean, duration, datetime (with or without a time zone),
        # any (object, or a categorical dtype when an enum constraint is given, with its
        # ordered flag) and string (no dtype, result none).
        field = {"name": "foo"}
        field.update(inp)
        assert convert_json_field_to_pandas_type(field) == exp

    @pytest.mark.parametrize("inp", ["geopoint", "geojson", "fake_type"])
    def test_convert_json_field_to_pandas_type_raises(self, inp):
        # [FUNCTION DOC]
        # Parametrized on unsupported type names (geopoint, geojson and a made-up one).
        # Raises a value error saying the field type is unsupported or invalid.
        field = {"type": inp}
        with pytest.raises(
            ValueError, match=f"Unsupported or invalid field type: {inp}"
        ):
            convert_json_field_to_pandas_type(field)

    def test_categorical(self):
        # [FUNCTION DOC]
        # A categorical series with a named index serializes to a schema with a
        # categorical field whose enum lists distinct categories, and three data records.
        s = pd.Series(pd.Categorical(["a", "b", "a"]))
        s.index.name = "idx"
        result = s.to_json(orient="table", date_format="iso")
        result = json.loads(result, object_pairs_hook=OrderedDict)
        result["schema"].pop("pandas_version")

        fields = [
            {"name": "idx", "type": "integer"},
            {
                "constraints": {"enum": ["a", "b"]},
                "name": "values",
                "ordered": False,
                "type": "any",
            },
        ]

        expected = OrderedDict(
            [
                ("schema", {"fields": fields, "primaryKey": ["idx"]}),
                (
                    "data",
                    [
                        OrderedDict([("idx", 0), ("values", "a")]),
                        OrderedDict([("idx", 1), ("values", "b")]),
                        OrderedDict([("idx", 2), ("values", "a")]),
                    ],
                ),
            ]
        )

        assert result == expected

    @pytest.mark.parametrize(
        "idx,nm,prop",
        [
            (pd.Index([1]), "index", "name"),
            (pd.Index([1], name="myname"), "myname", "name"),
            (
                pd.MultiIndex.from_product([("a", "b"), ("c", "d")]),
                ["level_0", "level_1"],
                "names",
            ),
            (
                pd.MultiIndex.from_product(
                    [("a", "b"), ("c", "d")], names=["n1", "n2"]
                ),
                ["n1", "n2"],
                "names",
            ),
            (
                pd.MultiIndex.from_product(
                    [("a", "b"), ("c", "d")], names=["n1", None]
                ),
                ["n1", "level_1"],
                "names",
            ),
        ],
    )
    def test_set_names_unset(self, idx, nm, prop):
        # [FUNCTION DOC]
        # Parametrized on indexes and the expected resulting name or names. Setting
        # default names on a series gives index for an unnamed flat index, preserves a
        # given name, and gives level_N names for unnamed levels of a multi-index while
        # keeping given names.
        data = pd.Series(1, idx)
        result = set_default_names(data)
        assert getattr(result.index, prop) == nm

    @pytest.mark.parametrize(
        "idx",
        [
            pd.Index([], name="index"),
            pd.MultiIndex.from_arrays([["foo"], ["bar"]], names=("level_0", "level_1")),
            pd.MultiIndex.from_arrays([["foo"], ["bar"]], names=("foo", "level_1")),
        ],
    )
    def test_warns_non_roundtrippable_names(self, idx):
        # [FUNCTION DOC]
        # GH 19130. Parametrized on indexes whose name is index or whose level names
        # collide with the generated ones. Setting default names must emit a user warning
        # saying the names are not round-trippable.
        # GH 19130
        df = DataFrame(index=idx)
        df.index.name = "index"
        with tm.assert_produces_warning(UserWarning, match="not round-trippable"):
            set_default_names(df)

    def test_timestamp_in_columns(self):
        # [FUNCTION DOC]
        # Column labels that are a timestamp and a timedelta are serialized as ISO text
        # names in the schema fields (positions one and two after the index).
        df = DataFrame(
            [[1, 2]], columns=[pd.Timestamp("2016"), pd.Timedelta(10, unit="s")]
        )
        result = df.to_json(orient="table")
        js = json.loads(result)
        assert js["schema"]["fields"][1]["name"] == "2016-01-01T00:00:00.000"
        assert js["schema"]["fields"][2]["name"] == "P0DT0H0M10S"

    @pytest.mark.parametrize(
        "case",
        [
            pd.Series([1], index=pd.Index([1], name="a"), name="a"),
            DataFrame({"A": [1]}, index=pd.Index([1], name="A")),
            DataFrame(
                {"A": [1]},
                index=pd.MultiIndex.from_arrays([["a"], [1]], names=["A", "a"]),
            ),
        ],
    )
    def test_overlapping_names(self, case):
        # [FUNCTION DOC]
        # Parametrized on a series, a frame and a frame with multi-index where the index
        # name equals a column name. Writing with table orientation must raise a value
        # error mentioning overlapping names.
        with pytest.raises(ValueError, match="Overlapping"):
            case.to_json(orient="table")

    def test_mi_falsey_name(self):
        # [FUNCTION DOC]
        # GH 16203. A frame with an unnamed two-level index and integer column labels
        # including zero: field names are the generated level names followed by the integer
        # labels, so a falsey label zero is kept.
        # GH 16203
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=pd.MultiIndex.from_product([("A", "B"), ("a", "b")]),
        )
        result = [x["name"] for x in build_table_schema(df)["fields"]]
        assert result == ["level_0", "level_1", 0, 1, 2, 3]


class TestTableOrientReader:
    @pytest.mark.parametrize(
        "index_nm",
        [None, "idx", pytest.param("index", marks=pytest.mark.xfail), "level_0"],
    )
    @pytest.mark.parametrize(
        "vals",
        [
            {"ints": [1, 2, 3, 4]},
            {"objects": ["a", "b", "c", "d"]},
            {"objects": ["1", "2", "3", "4"]},
            {
                "date_ranges": pd.date_range(
                    "2016-01-01", freq="D", periods=4, unit="ns"
                )
            },
            {"categoricals": pd.Series(pd.Categorical(["a", "b", "c", "c"]))},
            {
                "ordered_cats": pd.Series(
                    pd.Categorical(["a", "b", "c", "c"], ordered=True)
                )
            },
            {"floats": [1.0, 2.0, 3.0, 4.0]},
            {"floats": [1.1, 2.2, 3.3, 4.4]},
            {"bools": [True, False, False, True]},
            {
                "timezones": pd.date_range(
                    "2016-01-01", freq="D", periods=4, tz="US/Central", unit="ns"
                )  # added in # GH 35973
            },
        ],
    )
    # [CLASS DOC]
    # Round-trip tests for reading table-orient JSON back to frames.
    def test_read_json_table_orient(self, index_nm, vals):
        # [FUNCTION DOC]
        # Parametrized on index name (none, idx, level_0; index is expected to fail) and
        # on column content (ints, strings, numeric-like strings, dates, categoricals,
        # ordered categoricals, floats, bools, time zones). Writes with table orientation
        # and reads back; the frames must be equal.
        df = DataFrame(vals, index=pd.Index(range(4), name=index_nm))
        out = StringIO(df.to_json(orient="table"))
        result = pd.read_json(out, orient="table")
        tm.assert_frame_equal(df, result)

    @pytest.mark.parametrize(
        "index_nm",
        [
            None,
            "idx",
            pytest.param(
                "index",
                marks=pytest.mark.filterwarnings("ignore:Index name:UserWarning"),
            ),
        ],
    )
    def test_read_json_table_orient_raises(self, index_nm):
        # [FUNCTION DOC]
        # Parametrized on index name. Reading a table JSON with a timedelta column raises
        # a not-implemented error saying reading is not yet possible (the case where the
        # name is index ignores the name warning).
        vals = {"timedeltas": pd.timedelta_range("1h", periods=4, freq="min")}
        df = DataFrame(vals, index=pd.Index(range(4), name=index_nm))
        out = StringIO(df.to_json(orient="table"))
        with pytest.raises(NotImplementedError, match="can not yet read "):
            pd.read_json(out, orient="table")

    @pytest.mark.parametrize(
        "index_nm",
        [None, "idx", pytest.param("index", marks=pytest.mark.xfail), "level_0"],
    )
    @pytest.mark.parametrize(
        "vals",
        [
            {"ints": [1, 2, 3, 4]},
            {"objects": ["a", "b", "c", "d"]},
            {"objects": ["1", "2", "3", "4"]},
            {
                "date_ranges": pd.date_range(
                    "2016-01-01", freq="D", periods=4, unit="ns"
                )
            },
            {"categoricals": pd.Series(pd.Categorical(["a", "b", "c", "c"]))},
            {
                "ordered_cats": pd.Series(
                    pd.Categorical(["a", "b", "c", "c"], ordered=True)
                )
            },
            {"floats": [1.0, 2.0, 3.0, 4.0]},
            {"floats": [1.1, 2.2, 3.3, 4.4]},
            {"bools": [True, False, False, True]},
            {
                "timezones": pd.date_range(
                    "2016-01-01", freq="D", periods=4, tz="US/Central", unit="ns"
                )  # added in # GH 35973
            },
        ],
    )
    def test_read_json_table_period_orient(self, index_nm, vals):
        # [FUNCTION DOC]
        # Same parametrization as the first reader test, but with a quarterly period index;
        # round-trip equality is expected.
        df = DataFrame(
            vals,
            index=pd.Index(
                (pd.Period(f"2022Q{q}") for q in range(1, 5)), name=index_nm
            ),
        )
        out = StringIO(df.to_json(orient="table"))
        result = pd.read_json(out, orient="table")
        tm.assert_frame_equal(df, result)

    @pytest.mark.parametrize(
        "idx",
        [
            pd.Index(range(4)),
            pd.date_range(
                "2020-08-30",
                freq="D",
                periods=4,
                unit="ns",
            )._with_freq(None),
            pd.date_range(
                "2020-08-30", freq="D", periods=4, tz="US/Central", unit="ns"
            )._with_freq(None),
            pd.MultiIndex.from_product(
                [
                    pd.date_range(
                        "2020-08-30", freq="D", periods=2, tz="US/Central", unit="ns"
                    ),
                    ["x", "y"],
                ],
            ),
        ],
    )
    @pytest.mark.parametrize(
        "vals",
        [
            {"floats": [1.1, 2.2, 3.3, 4.4]},
            {"dates": pd.date_range("2020-08-30", freq="D", periods=4, unit="ns")},
            {
                "timezones": pd.date_range(
                    "2020-08-30", freq="D", periods=4, tz="Europe/London", unit="ns"
                )
            },
        ],
    )
    def test_read_json_table_timezones_orient(self, idx, vals):
        # [FUNCTION DOC]
        # GH 35973. Parametrized on index (range, naive datetime, timezone-aware datetime,
        # and a multi-index with a timezone level) and values (floats, dates, London time
        # zone). Round trip must give equal frames.
        # GH 35973
        df = DataFrame(vals, index=idx)
        out = StringIO(df.to_json(orient="table"))
        result = pd.read_json(out, orient="table")
        tm.assert_frame_equal(df, result)

    def test_comprehensive(self):
        # [FUNCTION DOC]
        # A frame with integer, string, datetime, categorical, ordered categorical, float,
        # timezone-aware datetime and boolean columns (no timedelta) and a named index
        # round-trips equal.
        df = DataFrame(
            {
                "A": [1, 2, 3, 4],
                "B": ["a", "b", "c", "c"],
                "C": pd.date_range("2016-01-01", freq="D", periods=4, unit="ns"),
                # 'D': pd.timedelta_range('1h', periods=4, freq='min'),
                "E": pd.Series(pd.Categorical(["a", "b", "c", "c"])),
                "F": pd.Series(pd.Categorical(["a", "b", "c", "c"], ordered=True)),
                "G": [1.1, 2.2, 3.3, 4.4],
                "H": pd.date_range(
                    "2016-01-01", freq="D", periods=4, tz="US/Central", unit="ns"
                ),
                "I": [True, False, False, True],
            },
            index=pd.Index(range(4), name="idx"),
        )

        out = StringIO(df.to_json(orient="table"))
        result = pd.read_json(out, orient="table")
        tm.assert_frame_equal(df, result)

    @pytest.mark.parametrize(
        "index_names",
        [[None, None], ["foo", "bar"], ["foo", None], [None, "foo"], ["index", "foo"]],
    )
    def test_multiindex(self, index_names):
        # [FUNCTION DOC]
        # GH 18912. Parametrized on index name lists, including missing names and the
        # name index. A frame with two-level index and list-valued column round-trips
        # equal.
        # GH 18912
        df = DataFrame(
            [["Arr", "alpha", [1, 2, 3, 4]], ["Bee", "Beta", [10, 20, 30, 40]]],
            index=[["A", "B"], ["Null", "Eins"]],
            columns=["Aussprache", "Griechisch", "Args"],
        )
        df.index.names = index_names
        out = StringIO(df.to_json(orient="table"))
        result = pd.read_json(out, orient="table")
        tm.assert_frame_equal(df, result)

    def test_empty_frame_roundtrip(self):
        # [FUNCTION DOC]
        # GH 21287. A frame with columns but no rows round-trips equal.
        # GH 21287
        df = DataFrame(columns=["a", "b", "c"])
        expected = df.copy()
        out = StringIO(df.to_json(orient="table"))
        result = pd.read_json(out, orient="table")
        tm.assert_frame_equal(expected, result)

    def test_read_json_orient_table_old_schema_version(self):
        # [FUNCTION DOC]
        # A hand-written table JSON with an old schema version (0.20.0) and a string field
        # holding mixed integer, float and text values is read as a single object column
        # of the mixed values.
        df_json = """
        {
            "schema":{
                "fields":[
                    {"name":"index","type":"integer"},
                    {"name":"a","type":"string"}
                ],
                "primaryKey":["index"],
                "pandas_version":"0.20.0"
            },
            "data":[
                {"index":0,"a":1},
                {"index":1,"a":2.0},
                {"index":2,"a":"s"}
            ]
        }
        """
        expected = DataFrame({"a": [1, 2.0, "s"]})
        result = pd.read_json(StringIO(df_json), orient="table")
        tm.assert_frame_equal(expected, result)

    @pytest.mark.parametrize("freq", ["M", "2M", "Q", "2Q", "Y", "2Y"])
    def test_read_json_table_orient_period_depr_freq(self, freq):
        # [FUNCTION DOC]
        # GH#9586. Parametrized on frequency aliases (month, quarter, year, with
        # multiples). A frame with a period index of that frequency round-trips equal. No
        # warning is asserted despite the name.
        # GH#9586
        df = DataFrame(
            {"ints": [1, 2]},
            index=pd.PeriodIndex(["2020-01", "2021-06"], freq=freq),
        )
        out = StringIO(df.to_json(orient="table"))
        result = pd.read_json(out, orient="table")
        tm.assert_frame_equal(df, result)
