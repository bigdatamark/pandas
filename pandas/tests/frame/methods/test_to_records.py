# [MODULE DOC]
# Tests for DataFrame.to_records (conversion to a numpy record array): index handling, dtype mapping
# arguments (column_dtypes, index_dtypes), multi-level axes, timezone-aware data, categoricals and non-ascii names.

from collections import abc
import email
from email.parser import Parser

import numpy as np
import pytest

from pandas import (
    CategoricalDtype,
    DataFrame,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm


class TestDataFrameToRecords:
    # [CLASS DOC]
    # Test container for to_records.
    def test_to_records_timeseries(self):
        # [FUNCTION DOC]
        # A frame with a nanosecond datetime index converts with the index field named index and datetime dtype. A second
        # conversion without the index is performed.
        # KNOWN ISSUE: the result of the second conversion (index turned off) is never asserted on; the call only checks that no error is raised.
        index = date_range("1/1/2000", periods=10, unit="ns")
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)),
            index=index,
            columns=["a", "b", "c"],
        )

        result = df.to_records()
        assert result["index"].dtype == "M8[ns]"

        result = df.to_records(index=False)

    def test_to_records_dt64(self):
        # [FUNCTION DOC]
        # The index field of the first record equals the first value of the datetime index.
        df = DataFrame(
            [["one", "two", "three"], ["four", "five", "six"]],
            index=date_range("2012-01-01", "2012-01-02"),
        )

        expected = df.index.values[0]
        result = df.to_records()["index"][0]
        assert expected == result

    def test_to_records_dt64tz_column(self):
        # [FUNCTION DOC]
        # Regression test (GH 32535). A timezone-aware column is converted to object dtype field holding timestamp objects equal to the frame's original value, so the timezone is not lost.
        # GH#32535 dont less tz in to_records
        df = DataFrame({"A": date_range("2012-01-01", "2012-01-02", tz="US/Eastern")})

        result = df.to_records()

        assert result.dtype["A"] == object
        val = result[0][1]
        assert isinstance(val, Timestamp)
        assert val == df.loc[0, "A"]

    def test_to_records_with_multindex(self):
        # [FUNCTION DOC]
        # Regression test (GH 3189). With a two-level index, the first level is exposed as a field named level_0 and contains first-level labels but not second-level labels.
        # GH#3189
        index = [
            ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
        ]
        data = np.zeros((8, 4))
        df = DataFrame(data, index=index)
        r = df.to_records(index=True)["level_0"]
        assert "bar" in r
        assert "one" not in r

    def test_to_records_with_Mapping_type(self):
        # [FUNCTION DOC]
        # Registers an email message class as a mapping type (global side effect on the abstract mapping class) and builds a frame from records
        # containing a parsed message.
        # KNOWN ISSUE: the final all-membership check builds a boolean but its result is never asserted, so nothing is verified beyond not raising.
        abc.Mapping.register(email.message.Message)

        headers = Parser().parsestr(
            "From: <user@example.com>\n"
            "To: <someone_else@example.com>\n"
            "Subject: Test message\n"
            "\n"
            "Body would go here\n"
        )

        frame = DataFrame.from_records([headers])
        all(x in frame for x in ["Type", "Subject", "From"])

    def test_to_records_floats(self):
        # [FUNCTION DOC]
        # Smoke test: converting a random float frame does not raise; no assertions are made.
        df = DataFrame(np.random.default_rng(2).random((10, 10)))
        df.to_records()

    def test_to_records_index_name(self):
        # [FUNCTION DOC]
        # Index field naming: a named index uses its name, an unnamed index uses the label index, and for a two-level index
        # with names A and none the fields are A and level_1 (object dtype) followed by the column fields named by
        # column labels as strings with float dtype; compared with the expected record array.
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)))
        df.index.name = "X"
        rs = df.to_records()
        assert "X" in rs.dtype.fields

        df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)))
        rs = df.to_records()
        assert "index" in rs.dtype.fields

        df.index = MultiIndex.from_tuples([("a", "x"), ("a", "y"), ("b", "z")])
        df.index.names = ["A", None]
        result = df.to_records()
        expected = np.rec.fromarrays(
            [np.array(["a", "a", "b"]), np.array(["x", "y", "z"])]
            + [np.asarray(df.iloc[:, i]) for i in range(3)],
            dtype={
                "names": ["A", "level_1", "0", "1", "2"],
                "formats": [
                    "O",
                    "O",
                    f"{tm.ENDIAN}f8",
                    f"{tm.ENDIAN}f8",
                    f"{tm.ENDIAN}f8",
                ],
            },
        )
        tm.assert_numpy_array_equal(result, expected)

    def test_to_records_with_unicode_index(self):
        # [FUNCTION DOC]
        # Regression test (GH 13172). A string index named a gives an object-dtype record array with fields a and b.
        # GH#13172
        # unicode_literals conflict with to_records
        result = DataFrame([{"a": "x", "b": "y"}]).set_index("a").to_records()
        expected = np.rec.array([("x", "y")], dtype=[("a", "O"), ("b", "O")])
        tm.assert_almost_equal(result, expected)

    def test_to_records_index_dtype(self):
        # [FUNCTION DOC]
        # Regression test (GH 47263). Three datetime columns give the same record array whether the index is dropped, or the first column,
        # or the first two columns, are used as the (multi-)index and then included in the records: consistent datetime field dtypes.
        # GH 47263: consistent data types for Index and MultiIndex
        df = DataFrame(
            {
                1: date_range("2022-01-01", periods=2, unit="ns"),
                2: date_range("2022-01-01", periods=2, unit="ns"),
                3: date_range("2022-01-01", periods=2, unit="ns"),
            }
        )

        expected = np.rec.array(
            [
                ("2022-01-01", "2022-01-01", "2022-01-01"),
                ("2022-01-02", "2022-01-02", "2022-01-02"),
            ],
            dtype=[
                ("1", f"{tm.ENDIAN}M8[ns]"),
                ("2", f"{tm.ENDIAN}M8[ns]"),
                ("3", f"{tm.ENDIAN}M8[ns]"),
            ],
        )

        result = df.to_records(index=False)
        tm.assert_almost_equal(result, expected)

        result = df.set_index(1).to_records(index=True)
        tm.assert_almost_equal(result, expected)

        result = df.set_index([1, 2]).to_records(index=True)
        tm.assert_almost_equal(result, expected)

    def test_to_records_with_unicode_column_names(self):
        # [FUNCTION DOC]
        # Regression test (GH 11879). A column with a non-ascii label converts, giving fields for the index and that label.
        # xref issue: https://github.com/numpy/numpy/issues/2407
        # Issue GH#11879. to_records used to raise an exception when used
        # with column names containing non-ascii characters in Python 2
        result = DataFrame(data={"accented_name_é": [1.0]}).to_records()

        # Note that numpy allows for unicode field names but dtypes need
        # to be specified using dictionary instead of list of tuples.
        expected = np.rec.array(
            [(0, 1.0)],
            dtype={"names": ["index", "accented_name_é"], "formats": ["=i8", "=f8"]},
        )
        tm.assert_almost_equal(result, expected)

    def test_to_records_with_categorical(self):
        # [FUNCTION DOC]
        # Regression test (GH 8626). First checks that a categorical frame built from a mapping and from a list has categorical columns (construction checks only), then
        # verifies conversion coerces the categorical column to object dtype.
        # GH#8626

        # dict creation
        df = DataFrame({"A": list("abc")}, dtype="category")
        expected = Series(list("abc"), dtype="category", name="A")
        tm.assert_series_equal(df["A"], expected)

        # list-like creation
        df = DataFrame(list("abc"), dtype="category")
        expected = Series(list("abc"), dtype="category", name=0)
        tm.assert_series_equal(df[0], expected)

        # to record array
        # this coerces
        result = df.to_records()
        expected = np.rec.array(
            [(0, "a"), (1, "b"), (2, "c")], dtype=[("index", "=i8"), ("0", "O")]
        )
        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            # No dtypes --> default to array dtypes.
            (
                {},
                np.rec.array(
                    [(0, 1, 0.2, "a"), (1, 2, 1.5, "bc")],
                    dtype=[
                        ("index", f"{tm.ENDIAN}i8"),
                        ("A", f"{tm.ENDIAN}i8"),
                        ("B", f"{tm.ENDIAN}f8"),
                        ("C", "O"),
                    ],
                ),
            ),
            # Should have no effect in this case.
            (
                {"index": True},
                np.rec.array(
                    [(0, 1, 0.2, "a"), (1, 2, 1.5, "bc")],
                    dtype=[
                        ("index", f"{tm.ENDIAN}i8"),
                        ("A", f"{tm.ENDIAN}i8"),
                        ("B", f"{tm.ENDIAN}f8"),
                        ("C", "O"),
                    ],
                ),
            ),
            # Column dtype applied across the board. Index unaffected.
            (
                {"column_dtypes": f"{tm.ENDIAN}U4"},
                np.rec.array(
                    [("0", "1", "0.2", "a"), ("1", "2", "1.5", "bc")],
                    dtype=[
                        ("index", f"{tm.ENDIAN}i8"),
                        ("A", f"{tm.ENDIAN}U4"),
                        ("B", f"{tm.ENDIAN}U4"),
                        ("C", f"{tm.ENDIAN}U4"),
                    ],
                ),
            ),
            # Index dtype applied across the board. Columns unaffected.
            (
                {"index_dtypes": f"{tm.ENDIAN}U1"},
                np.rec.array(
                    [("0", 1, 0.2, "a"), ("1", 2, 1.5, "bc")],
                    dtype=[
                        ("index", f"{tm.ENDIAN}U1"),
                        ("A", f"{tm.ENDIAN}i8"),
                        ("B", f"{tm.ENDIAN}f8"),
                        ("C", "O"),
                    ],
                ),
            ),
            # Pass in a type instance.
            (
                {"column_dtypes": str},
                np.rec.array(
                    [("0", "1", "0.2", "a"), ("1", "2", "1.5", "bc")],
                    dtype=[
                        ("index", f"{tm.ENDIAN}i8"),
                        ("A", f"{tm.ENDIAN}U"),
                        ("B", f"{tm.ENDIAN}U"),
                        ("C", f"{tm.ENDIAN}U"),
                    ],
                ),
            ),
            # Pass in a dtype instance.
            (
                {"column_dtypes": np.dtype(np.str_)},
                np.rec.array(
                    [("0", "1", "0.2", "a"), ("1", "2", "1.5", "bc")],
                    dtype=[
                        ("index", f"{tm.ENDIAN}i8"),
                        ("A", f"{tm.ENDIAN}U"),
                        ("B", f"{tm.ENDIAN}U"),
                        ("C", f"{tm.ENDIAN}U"),
                    ],
                ),
            ),
            # Pass in a dictionary (name-only).
            (
                {
                    "column_dtypes": {
                        "A": np.int8,
                        "B": np.float32,
                        "C": f"{tm.ENDIAN}U2",
                    }
                },
                np.rec.array(
                    [("0", "1", "0.2", "a"), ("1", "2", "1.5", "bc")],
                    dtype=[
                        ("index", f"{tm.ENDIAN}i8"),
                        ("A", "i1"),
                        ("B", f"{tm.ENDIAN}f4"),
                        ("C", f"{tm.ENDIAN}U2"),
                    ],
                ),
            ),
            # Pass in a dictionary (indices-only).
            (
                {"index_dtypes": {0: "int16"}},
                np.rec.array(
                    [(0, 1, 0.2, "a"), (1, 2, 1.5, "bc")],
                    dtype=[
                        ("index", "i2"),
                        ("A", f"{tm.ENDIAN}i8"),
                        ("B", f"{tm.ENDIAN}f8"),
                        ("C", "O"),
                    ],
                ),
            ),
            # Ignore index mappings if index is not True.
            (
                {"index": False, "index_dtypes": f"{tm.ENDIAN}U2"},
                np.rec.array(
                    [(1, 0.2, "a"), (2, 1.5, "bc")],
                    dtype=[
                        ("A", f"{tm.ENDIAN}i8"),
                        ("B", f"{tm.ENDIAN}f8"),
                        ("C", "O"),
                    ],
                ),
            ),
            # Non-existent names / indices in mapping should not error.
            (
                {"index_dtypes": {0: "int16", "not-there": "float32"}},
                np.rec.array(
                    [(0, 1, 0.2, "a"), (1, 2, 1.5, "bc")],
                    dtype=[
                        ("index", "i2"),
                        ("A", f"{tm.ENDIAN}i8"),
                        ("B", f"{tm.ENDIAN}f8"),
                        ("C", "O"),
                    ],
                ),
            ),
            # Names / indices not in mapping default to array dtype.
            (
                {"column_dtypes": {"A": np.int8, "B": np.float32}},
                np.rec.array(
                    [("0", "1", "0.2", "a"), ("1", "2", "1.5", "bc")],
                    dtype=[
                        ("index", f"{tm.ENDIAN}i8"),
                        ("A", "i1"),
                        ("B", f"{tm.ENDIAN}f4"),
                        ("C", "O"),
                    ],
                ),
            ),
            # Names / indices not in dtype mapping default to array dtype.
            (
                {"column_dtypes": {"A": np.dtype("int8"), "B": np.dtype("float32")}},
                np.rec.array(
                    [("0", "1", "0.2", "a"), ("1", "2", "1.5", "bc")],
                    dtype=[
                        ("index", f"{tm.ENDIAN}i8"),
                        ("A", "i1"),
                        ("B", f"{tm.ENDIAN}f4"),
                        ("C", "O"),
                    ],
                ),
            ),
            # Mixture of everything.
            (
                {
                    "column_dtypes": {"A": np.int8, "B": np.float32},
                    "index_dtypes": f"{tm.ENDIAN}U2",
                },
                np.rec.array(
                    [("0", "1", "0.2", "a"), ("1", "2", "1.5", "bc")],
                    dtype=[
                        ("index", f"{tm.ENDIAN}U2"),
                        ("A", "i1"),
                        ("B", f"{tm.ENDIAN}f4"),
                        ("C", "O"),
                    ],
                ),
            ),
            # Invalid dype values.
            (
                {"index": False, "column_dtypes": []},
                (ValueError, "Invalid dtype \\[\\] specified for column A"),
            ),
            (
                {"index": False, "column_dtypes": {"A": "int32", "B": 5}},
                (ValueError, "Invalid dtype 5 specified for column B"),
            ),
            # Numpy can't handle EA types, so check error is raised
            (
                {
                    "index": False,
                    "column_dtypes": {"A": "int32", "B": CategoricalDtype(["a", "b"])},
                },
                (ValueError, "Invalid dtype category specified for column B"),
            ),
            # Check that bad types raise
            (
                {"index": False, "column_dtypes": {"A": "int32", "B": "foo"}},
                (TypeError, "data type [\"']foo[\"'] not understood"),
            ),
        ],
    )
    def test_to_records_dtype(self, kwargs, expected):
        # [FUNCTION DOC]
        # Regression test (GH 18146). Parametrized over keyword argument sets and expected output. Cases: defaults; index option;
        # a single column dtype string, type or dtype instance applied to all columns; a single index dtype string; per-column name
        # mappings and per-index position mappings; mappings with unknown keys are ignored; index dtype is ignored when the index is turned off;
        # columns missing from mappings keep their own dtype; and error cases expecting a value error for invalid dtype values (an empty list,
        # a non-dtype number, an extension dtype) or a type error for an unknown dtype name. When the expectation is a pair the
        # call must raise that exception with a matching message, otherwise the result is compared to the expected record array.
        # see GH#18146
        df = DataFrame({"A": [1, 2], "B": [0.2, 1.5], "C": ["a", "bc"]})

        if not isinstance(expected, np.rec.recarray):
            with pytest.raises(expected[0], match=expected[1]):
                df.to_records(**kwargs)
        else:
            result = df.to_records(**kwargs)
            tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize(
        "df,kwargs,expected",
        [
            # MultiIndex in the index.
            (
                DataFrame(
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=list("abc")
                ).set_index(["a", "b"]),
                {"column_dtypes": "float64", "index_dtypes": {0: "int32", 1: "int8"}},
                np.rec.array(
                    [(1, 2, 3.0), (4, 5, 6.0), (7, 8, 9.0)],
                    dtype=[
                        ("a", f"{tm.ENDIAN}i4"),
                        ("b", "i1"),
                        ("c", f"{tm.ENDIAN}f8"),
                    ],
                ),
            ),
            # MultiIndex in the columns.
            (
                DataFrame(
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    columns=MultiIndex.from_tuples(
                        [("a", "d"), ("b", "e"), ("c", "f")]
                    ),
                ),
                {
                    "column_dtypes": {0: f"{tm.ENDIAN}U1", 2: "float32"},
                    "index_dtypes": "float32",
                },
                np.rec.array(
                    [(0.0, "1", 2, 3.0), (1.0, "4", 5, 6.0), (2.0, "7", 8, 9.0)],
                    dtype=[
                        ("index", f"{tm.ENDIAN}f4"),
                        ("('a', 'd')", f"{tm.ENDIAN}U1"),
                        ("('b', 'e')", f"{tm.ENDIAN}i8"),
                        ("('c', 'f')", f"{tm.ENDIAN}f4"),
                    ],
                ),
            ),
            # MultiIndex in both the columns and index.
            (
                DataFrame(
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    columns=MultiIndex.from_tuples(
                        [("a", "d"), ("b", "e"), ("c", "f")], names=list("ab")
                    ),
                    index=MultiIndex.from_tuples(
                        [("d", -4), ("d", -5), ("f", -6)], names=list("cd")
                    ),
                ),
                {
                    "column_dtypes": "float64",
                    "index_dtypes": {0: f"{tm.ENDIAN}U2", 1: "int8"},
                },
                np.rec.array(
                    [
                        ("d", -4, 1.0, 2.0, 3.0),
                        ("d", -5, 4.0, 5.0, 6.0),
                        ("f", -6, 7, 8, 9.0),
                    ],
                    dtype=[
                        ("c", f"{tm.ENDIAN}U2"),
                        ("d", "i1"),
                        ("('a', 'd')", f"{tm.ENDIAN}f8"),
                        ("('b', 'e')", f"{tm.ENDIAN}f8"),
                        ("('c', 'f')", f"{tm.ENDIAN}f8"),
                    ],
                ),
            ),
        ],
    )
    def test_to_records_dtype_mi(self, df, kwargs, expected):
        # [FUNCTION DOC]
        # Regression test (GH 18146). Parametrized over frames with multi-level index, multi-level columns and both, with dtype mapping
        # arguments given by position; the resulting record array must match (multi-level column labels become tuple-formatted field names).
        # see GH#18146
        result = df.to_records(**kwargs)
        tm.assert_almost_equal(result, expected)

    def test_to_records_dict_like(self):
        # [FUNCTION DOC]
        # Regression test (GH 18146). Dtype mapping arguments that are dict-like objects (only item access, membership and keys supported, not true mappings) must be
        # accepted: per-column mapping from such an object and a single index dtype produce the expected record array.
        # see GH#18146
        class DictLike:
            # [CLASS DOC]
            # HELPER: minimal dict-like class (wraps a dictionary) used to test dtype mapping arguments that are not real mappings.
            def __init__(self, **kwargs) -> None:
                # [FUNCTION DOC]
                # HELPER: stores a copy of the given keyword arguments as the wrapped dictionary.
                self.d = kwargs.copy()

            def __getitem__(self, key):
                # [FUNCTION DOC]
                # HELPER: returns the wrapped dictionary's value for the key (propagating a missing key error).
                return self.d.__getitem__(key)

            def __contains__(self, key) -> bool:
                # [FUNCTION DOC]
                # HELPER: returns whether the key exists in the wrapped dictionary.
                return key in self.d

            def keys(self):
                # [FUNCTION DOC]
                # HELPER: returns the keys view of the wrapped dictionary.
                return self.d.keys()

        df = DataFrame({"A": [1, 2], "B": [0.2, 1.5], "C": ["a", "bc"]})

        dtype_mappings = {
            "column_dtypes": DictLike(A=np.int8, B=np.float32),
            "index_dtypes": f"{tm.ENDIAN}U2",
        }

        result = df.to_records(**dtype_mappings)
        expected = np.rec.array(
            [("0", "1", "0.2", "a"), ("1", "2", "1.5", "bc")],
            dtype=[
                ("index", f"{tm.ENDIAN}U2"),
                ("A", "i1"),
                ("B", f"{tm.ENDIAN}f4"),
                ("C", "O"),
            ],
        )
        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize("tz", ["UTC", "GMT", "US/Eastern"])
    def test_to_records_datetimeindex_with_tz(self, tz):
        # [FUNCTION DOC]
        # Regression test (GH 13937). Parametrized over UTC, GMT and US/Eastern. A frame with a timezone-aware column and index gives the same records
        # as the same frame first converted to UTC, since records are always UTC-normalized.
        # GH#13937
        dr = date_range("2016-01-01", periods=10, freq="s", tz=tz)

        df = DataFrame({"datetime": dr}, index=dr)

        expected = df.to_records()
        result = df.tz_convert("UTC").to_records()

        # both converted to UTC, so they are equal
        tm.assert_numpy_array_equal(result, expected)
