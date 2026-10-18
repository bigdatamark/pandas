# [MODULE DOC]
# Tests for flattening nested JSON-like data into frames: the public
# normalization function (records, record paths, metadata, prefixes,
# separators, level limits, error handling, series and generator input) and the
# private nested-to-record flattening helper it relies on. Three data fixtures
# (a deeply nested country/state/city list, a state/county list and a list with
# missing metadata) are shared by the tests.

import json

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    Series,
    json_normalize,
)
import pandas._testing as tm

from pandas.io.json._normalize import nested_to_record


@pytest.fixture
def deep_nested():
    # [FUNCTION DOC]
    # HELPER: fixture returning two countries (USA and Germany), each with a list of
    # states, each state with a list of cities carrying names and populations.
    # deeply nested data
    return [
        {
            "country": "USA",
            "states": [
                {
                    "name": "California",
                    "cities": [
                        {"name": "San Francisco", "pop": 12345},
                        {"name": "Los Angeles", "pop": 12346},
                    ],
                },
                {
                    "name": "Ohio",
                    "cities": [
                        {"name": "Columbus", "pop": 1234},
                        {"name": "Cleveland", "pop": 1236},
                    ],
                },
            ],
        },
        {
            "country": "Germany",
            "states": [
                {"name": "Bayern", "cities": [{"name": "Munich", "pop": 12347}]},
                {
                    "name": "Nordrhein-Westfalen",
                    "cities": [
                        {"name": "Duesseldorf", "pop": 1238},
                        {"name": "Koeln", "pop": 1239},
                    ],
                },
            ],
        },
    ]


@pytest.fixture
def state_data():
    # [FUNCTION DOC]
    # HELPER: fixture returning two state records (Florida and Ohio), each with a
    # list of county records (name, population), an info mapping with a governor, a
    # short name and a state name.
    return [
        {
            "counties": [
                {"name": "Dade", "population": 12345},
                {"name": "Broward", "population": 40000},
                {"name": "Palm Beach", "population": 60000},
            ],
            "info": {"governor": "Rick Scott"},
            "shortname": "FL",
            "state": "Florida",
        },
        {
            "counties": [
                {"name": "Summit", "population": 1234},
                {"name": "Cuyahoga", "population": 1337},
            ],
            "info": {"governor": "John Kasich"},
            "shortname": "OH",
            "state": "Ohio",
        },
    ]


@pytest.fixture
def missing_metadata():
    # [FUNCTION DOC]
    # HELPER: fixture returning two person records with address lists and previous
    # residence city lists; the second record lacks the name key, so metadata for it
    # is missing.
    return [
        {
            "name": "Alice",
            "addresses": [
                {
                    "number": 9562,
                    "street": "Morris St.",
                    "city": "Massillon",
                    "state": "OH",
                    "zip": 44646,
                }
            ],
            "previous_residences": {"cities": [{"city_name": "Foo York City"}]},
        },
        {
            "addresses": [
                {
                    "number": 8449,
                    "street": "Spring St.",
                    "city": "Elizabethton",
                    "state": "TN",
                    "zip": 37643,
                }
            ],
            "previous_residences": {"cities": [{"city_name": "Barmingham"}]},
        },
    ]


class TestJSONNormalize:
    # [CLASS DOC]
    # Tests for the public normalization function.
    def test_simple_records(self):
        # [FUNCTION DOC]
        # A list of flat dictionaries normalizes to the same frame as building a frame
        # directly.
        recs = [
            {"a": 1, "b": 2, "c": 3},
            {"a": 4, "b": 5, "c": 6},
            {"a": 7, "b": 8, "c": 9},
            {"a": 10, "b": 11, "c": 12},
        ]

        result = json_normalize(recs)
        expected = DataFrame(recs)

        tm.assert_frame_equal(result, expected)

    def test_simple_normalize(self, state_data):
        # [FUNCTION DOC]
        # With a record path naming a list, one record gives a frame of its list elements;
        # multiple records are concatenated; adding a metadata key adds a column repeating
        # each parent's value for its children.
        result = json_normalize(state_data[0], "counties")
        expected = DataFrame(state_data[0]["counties"])
        tm.assert_frame_equal(result, expected)

        result = json_normalize(state_data, "counties")

        expected = []
        for rec in state_data:
            expected.extend(rec["counties"])
        expected = DataFrame(expected)

        tm.assert_frame_equal(result, expected)

        result = json_normalize(state_data, "counties", meta="state")
        expected["state"] = np.array(["Florida", "Ohio"]).repeat([3, 2])

        tm.assert_frame_equal(result, expected)

    def test_fields_list_type_normalize(self):
        # [FUNCTION DOC]
        # A record path of scalars with a metadata path whose value is a list: the
        # metadata column holds the same list in every row, and the record column is named
        # zero.
        parse_metadata_fields_list_type = [
            {"values": [1, 2, 3], "metadata": {"listdata": [1, 2]}}
        ]
        result = json_normalize(
            parse_metadata_fields_list_type,
            record_path=["values"],
            meta=[["metadata", "listdata"]],
        )
        expected = DataFrame(
            {0: [1, 2, 3], "metadata.listdata": [[1, 2], [1, 2], [1, 2]]}
        )
        tm.assert_frame_equal(result, expected)

    def test_empty_array(self):
        # [FUNCTION DOC]
        # An empty list normalizes to an empty frame.
        result = json_normalize([])
        expected = DataFrame()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "data, record_path, exception_type",
        [
            ([{"a": 0}, {"a": 1}], None, None),
            ({"a": [{"a": 0}, {"a": 1}]}, "a", None),
            ('{"a": [{"a": 0}, {"a": 1}]}', None, NotImplementedError),
            (None, None, NotImplementedError),
        ],
    )
    def test_accepted_input(self, data, record_path, exception_type):
        # [FUNCTION DOC]
        # Parametrized on data, record path and expected exception. A list of flat
        # dictionaries, or a dictionary with a record path, produce a one-column frame.
        # A JSON string or none raises a not-implemented error with an empty message.
        if exception_type is not None:
            with pytest.raises(exception_type, match="^$"):
                json_normalize(data, record_path=record_path)
        else:
            result = json_normalize(data, record_path=record_path)
            expected = DataFrame([0, 1], columns=["a"])
            tm.assert_frame_equal(result, expected)

    def test_simple_normalize_with_separator(self, deep_nested):
        # [FUNCTION DOC]
        # GH 14883. Nested keys are joined with a dot by default, and with a custom
        # separator including a non-ASCII one; and a deep record path with metadata and an
        # underscore separator yields the expected set of column names.
        # GH 14883
        result = json_normalize({"A": {"A": 1, "B": 2}})
        expected = DataFrame([[1, 2]], columns=["A.A", "A.B"])
        tm.assert_frame_equal(result.reindex_like(expected), expected)

        result = json_normalize({"A": {"A": 1, "B": 2}}, sep="_")
        expected = DataFrame([[1, 2]], columns=["A_A", "A_B"])
        tm.assert_frame_equal(result.reindex_like(expected), expected)

        result = json_normalize({"A": {"A": 1, "B": 2}}, sep="\u03c3")
        expected = DataFrame([[1, 2]], columns=["A\u03c3A", "A\u03c3B"])
        tm.assert_frame_equal(result.reindex_like(expected), expected)

        result = json_normalize(
            deep_nested,
            ["states", "cities"],
            meta=["country", ["states", "name"]],
            sep="_",
        )
        expected = Index(["name", "pop", "country", "states_name"]).sort_values()
        assert result.columns.sort_values().equals(expected)

    def test_normalize_with_multichar_separator(self):
        # [FUNCTION DOC]
        # GH #43831. A separator of more than one character is accepted; list and tuple
        # values are kept as cell values.
        # GH #43831
        data = {"a": [1, 2], "b": {"b_1": 2, "b_2": (3, 4)}}
        result = json_normalize(data, sep="__")
        expected = DataFrame([[[1, 2], 2, (3, 4)]], columns=["a", "b__b_1", "b__b_2"])
        tm.assert_frame_equal(result, expected)

    def test_value_array_record_prefix(self):
        # [FUNCTION DOC]
        # GH 21536. A record path of scalars with a record prefix names the single column
        # with the prefix followed by zero.
        # GH 21536
        result = json_normalize({"A": [1, 2]}, "A", record_prefix="Prefix.")
        expected = DataFrame([[1], [2]], columns=["Prefix.0"])
        tm.assert_frame_equal(result, expected)

    def test_nested_object_record_path(self):
        # [FUNCTION DOC]
        # GH 22706. A record path that goes through a nested mapping to a list of counties
        # produces a frame of the county records.
        # GH 22706
        data = {
            "state": "Florida",
            "info": {
                "governor": "Rick Scott",
                "counties": [
                    {"name": "Dade", "population": 12345},
                    {"name": "Broward", "population": 40000},
                    {"name": "Palm Beach", "population": 60000},
                ],
            },
        }
        result = json_normalize(data, record_path=["info", "counties"])
        expected = DataFrame(
            [["Dade", 12345], ["Broward", 40000], ["Palm Beach", 60000]],
            columns=["name", "population"],
        )
        tm.assert_frame_equal(result, expected)

    def test_more_deeply_nested(self, deep_nested):
        # [FUNCTION DOC]
        # A two-level record path with metadata for country and state name: seven city
        # rows with repeated parent values; column order taken from the result.
        result = json_normalize(
            deep_nested, ["states", "cities"], meta=["country", ["states", "name"]]
        )
        ex_data = {
            "country": ["USA"] * 4 + ["Germany"] * 3,
            "states.name": [
                "California",
                "California",
                "Ohio",
                "Ohio",
                "Bayern",
                "Nordrhein-Westfalen",
                "Nordrhein-Westfalen",
            ],
            "name": [
                "San Francisco",
                "Los Angeles",
                "Columbus",
                "Cleveland",
                "Munich",
                "Duesseldorf",
                "Koeln",
            ],
            "pop": [12345, 12346, 1234, 1236, 12347, 1238, 1239],
        }

        expected = DataFrame(ex_data, columns=result.columns)
        tm.assert_frame_equal(result, expected)

    def test_shallow_nested(self):
        # [FUNCTION DOC]
        # A record path of counties with metadata of state, short name and a nested
        # governor: five rows with repeated metadata and a dotted nested column name.
        data = [
            {
                "state": "Florida",
                "shortname": "FL",
                "info": {"governor": "Rick Scott"},
                "counties": [
                    {"name": "Dade", "population": 12345},
                    {"name": "Broward", "population": 40000},
                    {"name": "Palm Beach", "population": 60000},
                ],
            },
            {
                "state": "Ohio",
                "shortname": "OH",
                "info": {"governor": "John Kasich"},
                "counties": [
                    {"name": "Summit", "population": 1234},
                    {"name": "Cuyahoga", "population": 1337},
                ],
            },
        ]

        result = json_normalize(
            data, "counties", ["state", "shortname", ["info", "governor"]]
        )
        ex_data = {
            "name": ["Dade", "Broward", "Palm Beach", "Summit", "Cuyahoga"],
            "state": ["Florida"] * 3 + ["Ohio"] * 2,
            "shortname": ["FL", "FL", "FL", "OH", "OH"],
            "info.governor": ["Rick Scott"] * 3 + ["John Kasich"] * 2,
            "population": [12345, 40000, 60000, 1234, 1337],
        }
        expected = DataFrame(ex_data, columns=result.columns)
        tm.assert_frame_equal(result, expected)

    def test_nested_meta_path_with_nested_record_path(self, state_data):
        # [FUNCTION DOC]
        # GH 27220. Record path given as a one-element list and nested metadata with errors
        # ignored gives the same shape as the shallow case.
        # GH 27220
        result = json_normalize(
            data=state_data,
            record_path=["counties"],
            meta=["state", "shortname", ["info", "governor"]],
            errors="ignore",
        )

        ex_data = {
            "name": ["Dade", "Broward", "Palm Beach", "Summit", "Cuyahoga"],
            "population": [12345, 40000, 60000, 1234, 1337],
            "state": ["Florida"] * 3 + ["Ohio"] * 2,
            "shortname": ["FL"] * 3 + ["OH"] * 2,
            "info.governor": ["Rick Scott"] * 3 + ["John Kasich"] * 2,
        }

        expected = DataFrame(ex_data)
        tm.assert_frame_equal(result, expected)

    def test_meta_name_conflict(self):
        # [FUNCTION DOC]
        # A metadata key that collides with a record key raises a value error asking for a
        # distinguishing prefix; supplying a metadata prefix resolves it and both prefixed
        # and unprefixed names are present.
        data = [
            {
                "foo": "hello",
                "bar": "there",
                "data": [
                    {"foo": "something", "bar": "else"},
                    {"foo": "something2", "bar": "else2"},
                ],
            }
        ]

        msg = r"Conflicting metadata name (foo|bar), need distinguishing prefix"
        with pytest.raises(ValueError, match=msg):
            json_normalize(data, "data", meta=["foo", "bar"])

        result = json_normalize(data, "data", meta=["foo", "bar"], meta_prefix="meta")

        for val in ["metafoo", "metabar", "foo", "bar"]:
            assert val in result

    def test_meta_parameter_not_modified(self):
        # [FUNCTION DOC]
        # GH 18610. The metadata list passed in is not modified by the call, and the
        # prefixed columns are present.
        # GH 18610
        data = [
            {
                "foo": "hello",
                "bar": "there",
                "data": [
                    {"foo": "something", "bar": "else"},
                    {"foo": "something2", "bar": "else2"},
                ],
            }
        ]

        COLUMNS = ["foo", "bar"]
        result = json_normalize(data, "data", meta=COLUMNS, meta_prefix="meta")

        assert COLUMNS == ["foo", "bar"]
        for val in ["metafoo", "metabar", "foo", "bar"]:
            assert val in result

    def test_record_prefix(self, state_data):
        # [FUNCTION DOC]
        # Without metadata a record path gives plain record columns; with a record prefix
        # the record columns are prefixed but the metadata column is not.
        result = json_normalize(state_data[0], "counties")
        expected = DataFrame(state_data[0]["counties"])
        tm.assert_frame_equal(result, expected)

        result = json_normalize(
            state_data, "counties", meta="state", record_prefix="county_"
        )

        expected = []
        for rec in state_data:
            expected.extend(rec["counties"])
        expected = DataFrame(expected)
        expected = expected.rename(columns=lambda x: "county_" + x)
        expected["state"] = np.array(["Florida", "Ohio"]).repeat([3, 2])

        tm.assert_frame_equal(result, expected)

    def test_record_prefix_no_record_path_series(self):
        # [FUNCTION DOC]
        # A series of flat records with no record path and a record prefix gets the prefix
        # on all columns.
        # Ensure record_prefix is applied when record_path is None for Series input
        s = Series([{"k": f"{i}", "m": "q"} for i in range(3)])
        result = json_normalize(s, record_prefix="T.")
        expected = DataFrame(
            [
                {"T.k": "0", "T.m": "q"},
                {"T.k": "1", "T.m": "q"},
                {"T.k": "2", "T.m": "q"},
            ]
        )
        tm.assert_frame_equal(result, expected)

    def test_non_ascii_key(self):
        # [FUNCTION DOC]
        # Keys with non-ASCII characters and nested mapping keys are handled; the result
        # equals a frame with those column names.
        testjson = (
            b'[{"\xc3\x9cnic\xc3\xb8de":0,"sub":{"A":1, "B":2}},'
            b'{"\xc3\x9cnic\xc3\xb8de":1,"sub":{"A":3, "B":4}}]'
        ).decode("utf8")

        testdata = {
            b"\xc3\x9cnic\xc3\xb8de".decode("utf8"): [0, 1],
            "sub.A": [1, 3],
            "sub.B": [2, 4],
        }
        expected = DataFrame(testdata)

        result = json_normalize(json.loads(testjson))
        tm.assert_frame_equal(result, expected)

    def test_missing_field(self):
        # [FUNCTION DOC]
        # GH20030. A record where the info field is null and a record with nested info:
        # the null record gets missing values in every flattened column (including the
        # info column itself) while the second record keeps its own null-valued info entry.
        # GH20030:
        author_missing_data = [
            {"info": None},
            {
                "info": {"created_at": "11/08/1993", "last_updated": "26/05/2012"},
                "author_name": {"first": "Jane", "last_name": "Doe"},
            },
        ]
        result = json_normalize(author_missing_data)
        ex_data = [
            {
                "info": np.nan,
                "info.created_at": np.nan,
                "info.last_updated": np.nan,
                "author_name.first": np.nan,
                "author_name.last_name": np.nan,
            },
            {
                "info": None,
                "info.created_at": "11/08/1993",
                "info.last_updated": "26/05/2012",
                "author_name.first": "Jane",
                "author_name.last_name": "Doe",
            },
        ]
        expected = DataFrame(ex_data)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "max_level,expected",
        [
            (
                0,
                [
                    {
                        "TextField": "Some text",
                        "UserField": {"Id": "ID001", "Name": "Name001"},
                        "CreatedBy": {"Name": "User001"},
                        "Image": {"a": "b"},
                    },
                    {
                        "TextField": "Some text",
                        "UserField": {"Id": "ID001", "Name": "Name001"},
                        "CreatedBy": {"Name": "User001"},
                        "Image": {"a": "b"},
                    },
                ],
            ),
            (
                1,
                [
                    {
                        "TextField": "Some text",
                        "UserField.Id": "ID001",
                        "UserField.Name": "Name001",
                        "CreatedBy": {"Name": "User001"},
                        "Image": {"a": "b"},
                    },
                    {
                        "TextField": "Some text",
                        "UserField.Id": "ID001",
                        "UserField.Name": "Name001",
                        "CreatedBy": {"Name": "User001"},
                        "Image": {"a": "b"},
                    },
                ],
            ),
        ],
    )
    def test_max_level_with_records_path(self, max_level, expected):
        # [FUNCTION DOC]
        # GH23843. Parametrized on a maximum nesting level (0 or 1). With a record path
        # and mapping metadata, nested mappings inside records are flattened only up to
        # that level, remaining nested values stay as mappings.
        # GH23843: Enhanced JSON normalize
        test_input = [
            {
                "CreatedBy": {"Name": "User001"},
                "Lookup": [
                    {
                        "TextField": "Some text",
                        "UserField": {"Id": "ID001", "Name": "Name001"},
                    },
                    {
                        "TextField": "Some text",
                        "UserField": {"Id": "ID001", "Name": "Name001"},
                    },
                ],
                "Image": {"a": "b"},
                "tags": [
                    {"foo": "something", "bar": "else"},
                    {"foo": "something2", "bar": "else2"},
                ],
            }
        ]

        result = json_normalize(
            test_input,
            record_path=["Lookup"],
            meta=[["CreatedBy"], ["Image"]],
            max_level=max_level,
        )
        expected_df = DataFrame(data=expected, columns=result.columns.values)
        tm.assert_equal(expected_df, result)

    @pytest.mark.parametrize("max_level", [0, None])
    def test_json_normalize_non_dict_items(self, max_level):
        # [FUNCTION DOC]
        # gh-62829, gh-64188. Parametrized on max level (zero or none). A list containing
        # a missing value among dictionaries gives a missing row; a list with a non-NA-like
        # float raises a type error saying all items must be dict or NA-like.
        # gh-62829, gh-64188
        data_list = [np.nan, {"id": 12}, {"id": 13}]

        result = json_normalize(data_list, max_level=max_level)
        expected = DataFrame({"id": [np.nan, 12, 13]})
        tm.assert_frame_equal(result, expected)

        msg = "All items in data must be of type dict or NA-like, found float"
        with pytest.raises(TypeError, match=msg):
            json_normalize([1.0, {"id": 12}], max_level=max_level)

    def test_nested_flattening_consistent(self):
        # [FUNCTION DOC]
        # See gh-21537. A single nested record and the same record reached through a
        # record path produce equal frames.
        # see gh-21537
        df1 = json_normalize([{"A": {"B": 1}}])
        df2 = json_normalize({"dummy": [{"A": {"B": 1}}]}, "dummy")

        # They should be the same.
        tm.assert_frame_equal(df1, df2)

    def test_nonetype_record_path(self, nulls_fixture):
        # [FUNCTION DOC]
        # See gh-30148. Parametrized on null-like fixture values. A record whose value at
        # the record path is null-like is skipped without a type error, leaving only the
        # other record's rows.
        # see gh-30148
        # should not raise TypeError
        result = json_normalize(
            [
                {"state": "Texas", "info": nulls_fixture},
                {"state": "Florida", "info": [{"i": 2}]},
            ],
            record_path=["info"],
        )
        expected = DataFrame({"i": 2}, index=range(1))
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("value", ["false", "true", "{}", "1", '"text"'])
    def test_non_list_record_path_errors(self, value):
        # [FUNCTION DOC]
        # See gh-30148, GH 26284. Parametrized on JSON texts for boolean, mapping, number
        # and string. A record path value that is neither list nor null raises a type
        # error naming the actual type and the path.
        # see gh-30148, GH 26284
        parsed_value = json.loads(value)
        test_input = {"state": "Texas", "info": parsed_value}
        test_path = "info"
        msg = (
            f"Path must contain list or null, "
            f"but got {type(parsed_value).__name__} at 'info'"
        )
        with pytest.raises(TypeError, match=msg):
            json_normalize([test_input], record_path=[test_path])

    def test_meta_non_iterable(self):
        # [FUNCTION DOC]
        # GH 31507. A numeric metadata value is added as an object-dtype column.
        # GH 31507
        data = """[{"id": 99, "data": [{"one": 1, "two": 2}]}]"""

        result = json_normalize(json.loads(data), record_path=["data"], meta=["id"])
        expected = DataFrame(
            {"one": [1], "two": [2], "id": np.array([99], dtype=object)}
        )
        tm.assert_frame_equal(result, expected)

    def test_generator(self, state_data):
        # [FUNCTION DOC]
        # GH35923. A generator of records as input does not lose its first element; the
        # result equals the frame from the same records as a list.
        # GH35923 Fix pd.json_normalize to not skip the first element of a
        # generator input
        def generator_data():
            # [FUNCTION DOC]
            # HELPER: generator used by the enclosing test. Takes no inputs and yields each
            # county record of the first state fixture entry.
            yield from state_data[0]["counties"]

        result = json_normalize(generator_data())
        expected = DataFrame(state_data[0]["counties"])

        tm.assert_frame_equal(result, expected)

    def test_top_column_with_leading_underscore(self):
        # [FUNCTION DOC]
        # Issue 49861. A top-level key with a leading underscore and a custom separator of
        # underscore flattens correctly without losing the leading underscore.
        # 49861
        data = {"_id": {"a1": 10, "l2": {"l3": 0}}, "gg": 4}
        result = json_normalize(data, sep="_")
        expected = DataFrame([[4, 10, 0]], columns=["gg", "_id_a1", "_id_l2_l3"])

        tm.assert_frame_equal(result, expected)

    def test_series_index(self, state_data):
        # [FUNCTION DOC]
        # A series of records with a custom index keeps the index in the output, and with
        # a record path the index values are repeated per child record.
        idx = Index([7, 8])
        series = Series(state_data, index=idx)
        result = json_normalize(series)
        tm.assert_index_equal(result.index, idx)
        result = json_normalize(series, "counties")
        tm.assert_index_equal(result.index, idx.repeat([3, 2]))


class TestNestedToRecord:
    # [CLASS DOC]
    # Tests for the private flattening helper, plus several tests of the public
    # normalization function that cover missing-metadata handling and index handling.
    # KNOWN ISSUE: despite the class name, many tests here (error handling for missing metadata, missing nested metadata, series with a non-zero index, metadata validation) exercise the public normalization function rather than the flattening helper, so the grouping is misleading.
    def test_flat_stays_flat(self):
        # [FUNCTION DOC]
        # A list of flat dictionaries is returned unchanged.
        recs = [{"flat1": 1, "flat2": 2}, {"flat3": 3, "flat2": 4}]
        result = nested_to_record(recs)
        expected = recs
        assert result == expected

    def test_one_level_deep_flattens(self):
        # [FUNCTION DOC]
        # A nested mapping one level deep is flattened to dotted keys.
        data = {"flat1": 1, "dict1": {"c": 1, "d": 2}}

        result = nested_to_record(data)
        expected = {"dict1.c": 1, "dict1.d": 2, "flat1": 1}

        assert result == expected

    def test_nested_flattens(self):
        # [FUNCTION DOC]
        # Multiple levels of nesting are all flattened to dotted keys.
        data = {
            "flat1": 1,
            "dict1": {"c": 1, "d": 2},
            "nested": {"e": {"c": 1, "d": 2}, "d": 2},
        }

        result = nested_to_record(data)
        expected = {
            "dict1.c": 1,
            "dict1.d": 2,
            "flat1": 1,
            "nested.d": 2,
            "nested.e.c": 1,
            "nested.e.d": 2,
        }

        assert result == expected

    def test_json_normalize_errors(self, missing_metadata):
        # GH14583:
        # If meta keys are not always present a new option to set
        # errors='ignore' has been implemented

        # [FUNCTION DOC]
        # GH14583. Metadata missing in one record with the default error mode raises a key
        # error saying the key was not found and suggesting to ignore errors.
        msg = (
            "Key 'name' not found. To replace missing values of "
            "'name' with np.nan, pass in errors='ignore'"
        )
        with pytest.raises(KeyError, match=msg):
            json_normalize(
                data=missing_metadata,
                record_path="addresses",
                meta="name",
                errors="raise",
            )

    def test_missing_meta(self, missing_metadata):
        # [FUNCTION DOC]
        # GH25468. With errors ignored, missing metadata become missing values in the
        # metadata column.
        # GH25468
        # If metadata is nullable with errors set to ignore, the null values
        # should be numpy.nan values
        result = json_normalize(
            data=missing_metadata, record_path="addresses", meta="name", errors="ignore"
        )
        ex_data = [
            [9562, "Morris St.", "Massillon", "OH", 44646, "Alice"],
            [8449, "Spring St.", "Elizabethton", "TN", 37643, np.nan],
        ]
        columns = ["number", "street", "city", "state", "zip", "name"]
        expected = DataFrame(ex_data, columns=columns)
        tm.assert_frame_equal(result, expected)

    def test_missing_nested_meta(self):
        # [FUNCTION DOC]
        # GH44312. With errors ignored, nested metadata whose parent is null produces a
        # missing value in an object column; with errors raising it raises a key error
        # naming the missing leaf key.
        # GH44312
        # If errors="ignore" and nested metadata is null, we should return nan
        data = {"meta": "foo", "nested_meta": None, "value": [{"rec": 1}, {"rec": 2}]}
        result = json_normalize(
            data,
            record_path="value",
            meta=["meta", ["nested_meta", "leaf"]],
            errors="ignore",
        )
        ex_data = [[1, "foo", np.nan], [2, "foo", np.nan]]
        columns = ["rec", "meta", "nested_meta.leaf"]
        expected = DataFrame(ex_data, columns=columns).astype(
            {"nested_meta.leaf": object}
        )
        tm.assert_frame_equal(result, expected)

        # If errors="raise" and nested metadata is null, we should raise with the
        # key of the first missing level
        with pytest.raises(KeyError, match="'leaf' not found"):
            json_normalize(
                data,
                record_path="value",
                meta=["meta", ["nested_meta", "leaf"]],
                errors="raise",
            )

    def test_missing_meta_multilevel_record_path_errors_raise(self, missing_metadata):
        # [FUNCTION DOC]
        # GH41876. The key-error behaviour for missing metadata also holds when the record
        # path has more than one element.
        # GH41876
        # Ensure errors='raise' works as intended even when a record_path of length
        # greater than one is passed in
        msg = (
            "Key 'name' not found. To replace missing values of "
            "'name' with np.nan, pass in errors='ignore'"
        )
        with pytest.raises(KeyError, match=msg):
            json_normalize(
                data=missing_metadata,
                record_path=["previous_residences", "cities"],
                meta="name",
                errors="raise",
            )

    def test_missing_meta_multilevel_record_path_errors_ignore(self, missing_metadata):
        # [FUNCTION DOC]
        # GH41876. With errors ignored and a multi-element record path, missing metadata
        # become missing values.
        # GH41876
        # Ensure errors='ignore' works as intended even when a record_path of length
        # greater than one is passed in
        result = json_normalize(
            data=missing_metadata,
            record_path=["previous_residences", "cities"],
            meta="name",
            errors="ignore",
        )
        ex_data = [
            ["Foo York City", "Alice"],
            ["Barmingham", np.nan],
        ]
        columns = ["city_name", "name"]
        expected = DataFrame(ex_data, columns=columns)
        tm.assert_frame_equal(result, expected)

    def test_donot_drop_nonevalues(self):
        # [FUNCTION DOC]
        # GH21356. A null value for a key that would be expanded is kept as null under the
        # key itself, and other records are still flattened.
        # GH21356
        data = [
            {"info": None, "author_name": {"first": "Smith", "last_name": "Appleseed"}},
            {
                "info": {"created_at": "11/08/1993", "last_updated": "26/05/2012"},
                "author_name": {"first": "Jane", "last_name": "Doe"},
            },
        ]
        result = nested_to_record(data)
        expected = [
            {
                "info": None,
                "author_name.first": "Smith",
                "author_name.last_name": "Appleseed",
            },
            {
                "author_name.first": "Jane",
                "author_name.last_name": "Doe",
                "info.created_at": "11/08/1993",
                "info.last_updated": "26/05/2012",
            },
        ]

        assert result == expected

    def test_nonetype_top_level_bottom_level(self):
        # [FUNCTION DOC]
        # GH21158. Null values at various depths, and keys containing a dot, are flattened
        # without error (guards against removing a key twice).
        # GH21158: If inner level json has a key with a null value
        # make sure it does not do a new_d.pop twice and except
        data = {
            "id": None,
            "location": {
                "country": {
                    "state": {
                        "id": None,
                        "town.info": {
                            "id": None,
                            "region": None,
                            "x": 49.151580810546875,
                            "y": -33.148521423339844,
                            "z": 27.572303771972656,
                        },
                    }
                }
            },
        }
        result = nested_to_record(data)
        expected = {
            "id": None,
            "location.country.state.id": None,
            "location.country.state.town.info.id": None,
            "location.country.state.town.info.region": None,
            "location.country.state.town.info.x": 49.151580810546875,
            "location.country.state.town.info.y": -33.148521423339844,
            "location.country.state.town.info.z": 27.572303771972656,
        }
        assert result == expected

    def test_nonetype_multiple_levels(self):
        # [FUNCTION DOC]
        # GH21158. Null values at several levels are flattened with full dotted paths.
        # GH21158: If inner level json has a key with a null value
        # make sure it does not do a new_d.pop twice and except
        data = {
            "id": None,
            "location": {
                "id": None,
                "country": {
                    "id": None,
                    "state": {
                        "id": None,
                        "town.info": {
                            "region": None,
                            "x": 49.151580810546875,
                            "y": -33.148521423339844,
                            "z": 27.572303771972656,
                        },
                    },
                },
            },
        }
        result = nested_to_record(data)
        expected = {
            "id": None,
            "location.id": None,
            "location.country.id": None,
            "location.country.state.id": None,
            "location.country.state.town.info.region": None,
            "location.country.state.town.info.x": 49.151580810546875,
            "location.country.state.town.info.y": -33.148521423339844,
            "location.country.state.town.info.z": 27.572303771972656,
        }
        assert result == expected

    @pytest.mark.parametrize(
        "max_level, expected",
        [
            (
                None,
                [
                    {
                        "CreatedBy.Name": "User001",
                        "Lookup.TextField": "Some text",
                        "Lookup.UserField.Id": "ID001",
                        "Lookup.UserField.Name": "Name001",
                        "Image.a": "b",
                    }
                ],
            ),
            (
                0,
                [
                    {
                        "CreatedBy": {"Name": "User001"},
                        "Lookup": {
                            "TextField": "Some text",
                            "UserField": {"Id": "ID001", "Name": "Name001"},
                        },
                        "Image": {"a": "b"},
                    }
                ],
            ),
            (
                1,
                [
                    {
                        "CreatedBy.Name": "User001",
                        "Lookup.TextField": "Some text",
                        "Lookup.UserField": {"Id": "ID001", "Name": "Name001"},
                        "Image.a": "b",
                    }
                ],
            ),
        ],
    )
    def test_with_max_level(self, max_level, expected):
        # [FUNCTION DOC]
        # GH23843. Parametrized on max level none, 0 and 1. A record with nested mappings
        # is flattened to unlimited depth, not at all, or one level.
        # GH23843: Enhanced JSON normalize
        max_level_test_input_data = [
            {
                "CreatedBy": {"Name": "User001"},
                "Lookup": {
                    "TextField": "Some text",
                    "UserField": {"Id": "ID001", "Name": "Name001"},
                },
                "Image": {"a": "b"},
            }
        ]
        output = nested_to_record(max_level_test_input_data, max_level=max_level)
        assert output == expected

    def test_with_large_max_level(self):
        # [FUNCTION DOC]
        # GH23843. A max level of 100 flattens a deeply nested record completely.
        # GH23843: Enhanced JSON normalize
        max_level = 100
        input_data = [
            {
                "CreatedBy": {
                    "user": {
                        "name": {"firstname": "Leo", "LastName": "Thomson"},
                        "family_tree": {
                            "father": {
                                "name": "Father001",
                                "father": {
                                    "Name": "Father002",
                                    "father": {
                                        "name": "Father003",
                                        "father": {"Name": "Father004"},
                                    },
                                },
                            }
                        },
                    }
                }
            }
        ]
        expected = [
            {
                "CreatedBy.user.name.firstname": "Leo",
                "CreatedBy.user.name.LastName": "Thomson",
                "CreatedBy.user.family_tree.father.name": "Father001",
                "CreatedBy.user.family_tree.father.father.Name": "Father002",
                "CreatedBy.user.family_tree.father.father.father.name": "Father003",
                "CreatedBy.user.family_tree.father.father.father.father.Name": "Father004",  # noqa: E501
            }
        ]
        output = nested_to_record(input_data, max_level=max_level)
        assert output == expected

    def test_series_non_zero_index(self):
        # [FUNCTION DOC]
        # GH 19020. A series of records with index starting at one normalizes to a frame
        # with that index, with missing values for absent nested keys (float columns).
        # GH 19020
        data = {
            0: {"id": 1, "name": "Foo", "elements": {"a": 1}},
            1: {"id": 2, "name": "Bar", "elements": {"b": 2}},
            2: {"id": 3, "name": "Baz", "elements": {"c": 3}},
        }
        s = Series(data)
        s.index = [1, 2, 3]
        result = json_normalize(s)
        expected = DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Foo", "Bar", "Baz"],
                "elements.a": [1.0, np.nan, np.nan],
                "elements.b": [np.nan, 2.0, np.nan],
                "elements.c": [np.nan, np.nan, 3.0],
            },
            index=[1, 2, 3],
        )
        tm.assert_frame_equal(result, expected)

    def test_json_normalize_meta_string_validation(self):
        # [FUNCTION DOC]
        # GH 63019. Non-string metadata entries raise a type error saying they must be
        # strings, both without and with a record path.
        # GH 63019
        data = [{"a": 1, 12: "meta_value", "nested": [{"b": 2}]}]

        # Test non-string meta raises TypeError consistently
        with pytest.raises(TypeError, match="must be strings"):
            json_normalize(data, meta=[12])

        with pytest.raises(TypeError, match="must be strings"):
            json_normalize(data, record_path=["nested"], meta=[12])
