# [MODULE DOC]
# Tests for the DataFrame record-based alternate constructor (from-records) accepting
# structured and record arrays, lists of tuples, lists, dictionaries, Series, iterators and
# generators, with the index, columns, exclude, coerce-float and row-limit parameters.
# Covers dtype preservation, empty inputs, duplicates, missing-value keys, time zone
# aware values and error cases. One test is expected-to-fail under the string dtype mode.
# No import-time side effects.
from collections.abc import Iterator
from datetime import (
    datetime,
    timezone,
)
from decimal import Decimal

import numpy as np
import pytest

from pandas._config import using_string_dtype

from pandas.compat import is_platform_little_endian

import pandas as pd
from pandas import (
    CategoricalIndex,
    DataFrame,
    Index,
    Interval,
    RangeIndex,
    Series,
)
import pandas._testing as tm


class TestFromRecords:
    # [CLASS DOC]
    # Test class grouping all from-records constructor tests; no shared state.
    def test_from_records_dt64tz_frame(self):
        # [FUNCTION DOC]
        # KNOWN ISSUE: the name refers to datetime-with-time-zone data but the test passes an
        # ordinary integer frame with no time zone data.
        # Regression test for GH#51697: passing a DataFrame to from-records raises a type error
        # whose message says it is not supported.
        # GH#51697
        df = DataFrame({"a": [1, 2, 3]})
        with pytest.raises(TypeError, match="not supported"):
            DataFrame.from_records(df)

    def test_from_records_with_datetimes(self):
        # [FUNCTION DOC]
        # Verifies a record array with a microsecond datetime field containing a missing entry
        # (GH#6140) converts to a frame equal to one built from python datetimes and None. A second
        # array with minute resolution is coerced to the closest supported unit (seconds) in the
        # expected frame. Skipped on non-little-endian platforms because of a numpy issue.
        # this may fail on certain platforms because of a numpy issue
        # related GH#6140
        if not is_platform_little_endian():
            pytest.skip("known failure of test on non-little endian")

        # construction with a null in a recarray
        # GH#6140
        expected = DataFrame({"EXPIRY": [datetime(2005, 3, 1, 0, 0), None]})

        arrdata = [np.array([datetime(2005, 3, 1, 0, 0), None])]
        dtypes = [("EXPIRY", "<M8[us]")]

        recarray = np.rec.fromarrays(arrdata, dtype=dtypes)

        result = DataFrame.from_records(recarray)
        tm.assert_frame_equal(result, expected)

        # coercion should work too
        arrdata = [np.array([datetime(2005, 3, 1, 0, 0), None])]
        dtypes = [("EXPIRY", "<M8[m]")]
        recarray = np.rec.fromarrays(arrdata, dtype=dtypes)
        result = DataFrame.from_records(recarray)
        # we get the closest supported unit, "s"
        expected["EXPIRY"] = expected["EXPIRY"].astype("M8[s]")
        tm.assert_frame_equal(result, expected)

    @pytest.mark.xfail(using_string_dtype(), reason="dtype checking logic doesn't work")
    def test_from_records_sequencelike(self):
        # [FUNCTION DOC]
        # Expected-to-fail when the string dtype mode is on (dtype checking logic does not work).
        # Builds a frame with float64, int64, string, bool, float32 and int32 columns, derives
        # tuples, list rows, a hand-made record array and the frame's own record array (using the
        # internal block dictionary to align column order with dtypes). Verifies from-records on
        # tuples and lists reproduces the frame ignoring dtype, on record arrays reproduces it
        # exactly, that without columns the labels are a default range, and that the exclude
        # parameter keeps only the string and float32 columns with matching values.
        df = DataFrame(
            {
                "A": np.array(
                    np.random.default_rng(2).standard_normal(6), dtype=np.float64
                ),
                "A1": np.array(
                    np.random.default_rng(2).standard_normal(6), dtype=np.float64
                ),
                "B": np.array(np.arange(6), dtype=np.int64),
                "C": ["foo"] * 6,
                "D": np.array([True, False] * 3, dtype=bool),
                "E": np.array(
                    np.random.default_rng(2).standard_normal(6), dtype=np.float32
                ),
                "E1": np.array(
                    np.random.default_rng(2).standard_normal(6), dtype=np.float32
                ),
                "F": np.array(np.arange(6), dtype=np.int32),
            }
        )

        # this is actually tricky to create the recordlike arrays and
        # have the dtypes be intact
        blocks = df._to_dict_of_blocks()
        tuples = []
        columns = []
        dtypes = []
        for dtype, b in blocks.items():
            columns.extend(b.columns)
            dtypes.extend([(c, np.dtype(dtype).descr[0][1]) for c in b.columns])
        for i in range(len(df.index)):
            tup = []
            for _, b in blocks.items():
                tup.extend(b.iloc[i].values)
            tuples.append(tuple(tup))

        recarray = np.array(tuples, dtype=dtypes).view(np.rec.recarray)
        recarray2 = df.to_records()
        lists = [list(x) for x in tuples]

        # tuples (lose the dtype info)
        result = DataFrame.from_records(tuples, columns=columns).reindex(
            columns=df.columns
        )

        # created recarray and with to_records recarray (have dtype info)
        result2 = DataFrame.from_records(recarray, columns=columns).reindex(
            columns=df.columns
        )
        result3 = DataFrame.from_records(recarray2, columns=columns).reindex(
            columns=df.columns
        )

        # list of tuples (no dtype info)
        result4 = DataFrame.from_records(lists, columns=columns).reindex(
            columns=df.columns
        )

        tm.assert_frame_equal(result, df, check_dtype=False)
        tm.assert_frame_equal(result2, df)
        tm.assert_frame_equal(result3, df)
        tm.assert_frame_equal(result4, df, check_dtype=False)

        # tuples is in the order of the columns
        result = DataFrame.from_records(tuples)
        tm.assert_index_equal(result.columns, RangeIndex(8))

        # test exclude parameter & we are casting the results here (as we don't
        # have dtype info to recover)
        columns_to_test = [columns.index("C"), columns.index("E1")]

        exclude = list(set(range(8)) - set(columns_to_test))
        result = DataFrame.from_records(tuples, exclude=exclude)
        result.columns = [columns[i] for i in sorted(columns_to_test)]
        tm.assert_series_equal(result["C"], df["C"])
        tm.assert_series_equal(result["E1"], df["E1"])

    def test_from_records_sequencelike_empty(self):
        # [FUNCTION DOC]
        # Verifies from-records on an empty list: with columns gives zero rows and those column
        # labels; without columns gives zero rows and zero columns.
        # empty case
        result = DataFrame.from_records([], columns=["foo", "bar", "baz"])
        assert len(result) == 0
        tm.assert_index_equal(result.columns, Index(["foo", "bar", "baz"]))

        result = DataFrame.from_records([])
        assert len(result) == 0
        assert len(result.columns) == 0

    def test_from_records_sequencelike_empty_index(self):
        # [FUNCTION DOC]
        # Verifies from-records on an empty list with an explicit empty index gives an empty frame
        # with no rows or columns.
        result = DataFrame.from_records([], index=[])
        assert len(result) == 0
        assert len(result.columns) == 0
        assert len(result.index) == 0

    def test_from_records_dictlike(self):
        # [FUNCTION DOC]
        # Builds a mixed-dtype frame, then reconstructs it via from-records from a dictionary of
        # Series and from a dictionary of arrays, with and without a columns argument in the
        # block-grouped order, reindexing to the original order. All results equal the original
        # including dtypes.
        # test the dict methods
        df = DataFrame(
            {
                "A": np.array(
                    np.random.default_rng(2).standard_normal(6), dtype=np.float64
                ),
                "A1": np.array(
                    np.random.default_rng(2).standard_normal(6), dtype=np.float64
                ),
                "B": np.array(np.arange(6), dtype=np.int64),
                "C": ["foo"] * 6,
                "D": np.array([True, False] * 3, dtype=bool),
                "E": np.array(
                    np.random.default_rng(2).standard_normal(6), dtype=np.float32
                ),
                "E1": np.array(
                    np.random.default_rng(2).standard_normal(6), dtype=np.float32
                ),
                "F": np.array(np.arange(6), dtype=np.int32),
            }
        )

        # columns is in a different order here than the actual items iterated
        # from the dict
        blocks = df._to_dict_of_blocks()
        columns = []
        for b in blocks.values():
            columns.extend(b.columns)

        asdict = dict(df.items())
        asdict2 = {x: y.values for x, y in df.items()}

        # dict of series & dict of ndarrays (have dtype info)
        results = []
        results.append(DataFrame.from_records(asdict).reindex(columns=df.columns))
        results.append(
            DataFrame.from_records(asdict, columns=columns).reindex(columns=df.columns)
        )
        results.append(
            DataFrame.from_records(asdict2, columns=columns).reindex(columns=df.columns)
        )

        for r in results:
            tm.assert_frame_equal(r, df)

    def test_from_records_non_tuple(self):
        # [FUNCTION DOC]
        # Verifies from-records accepts a list of non-tuple record-like objects (supporting item
        # access and iteration) and gives the same frame as the equivalent list of tuples.
        class Record:
            # [CLASS DOC]
            # HELPER: local record-like class used by the enclosing test; wraps its positional
            # arguments, supports item lookup and iteration, but is not a tuple.
            def __init__(self, *args) -> None:
                # [FUNCTION DOC]
                # Stores all positional arguments as a tuple on the args attribute.
                self.args = args

            def __getitem__(self, i):
                # [FUNCTION DOC]
                # Returns the stored argument at the given position.
                return self.args[i]

            def __iter__(self) -> Iterator:
                # [FUNCTION DOC]
                # Returns an iterator over the stored arguments.
                return iter(self.args)

        recs = [Record(1, 2, 3), Record(4, 5, 6), Record(7, 8, 9)]
        tups = [tuple(rec) for rec in recs]

        result = DataFrame.from_records(recs)
        expected = DataFrame.from_records(tups)
        tm.assert_frame_equal(result, expected)

    def test_from_records_len0_with_columns(self):
        # [FUNCTION DOC]
        # Regression test for GH#2633: from-records on an empty list with an index name that is also
        # a column leaves zero rows, names the index after it and leaves only the other column.
        # GH#2633
        result = DataFrame.from_records([], index="foo", columns=["foo", "bar"])
        expected = Index(["bar"])

        assert len(result) == 0
        assert result.index.name == "foo"
        tm.assert_index_equal(result.columns, expected)

    def test_from_records_series_list_dict(self):
        # [FUNCTION DOC]
        # Regression test for GH#27358: a Series whose elements are lists of dictionaries converts
        # to a frame whose columns hold the dictionaries (equal to the transposed nested-list frame).
        # GH#27358
        expected = DataFrame([[{"a": 1, "b": 2}, {"a": 3, "b": 4}]]).T
        data = Series([[{"a": 1, "b": 2}], [{"a": 3, "b": 4}]])
        result = DataFrame.from_records(data)
        tm.assert_frame_equal(result, expected)

    def test_from_records_series_categorical_index(self):
        # [FUNCTION DOC]
        # Regression test for GH#32805: a Series of dictionaries with a categorical interval index,
        # converted with that index passed as the index argument, yields columns a and b with
        # missing values where keys are absent and the same index.
        # GH#32805
        index = CategoricalIndex(
            [Interval(-20, -10), Interval(-10, 0), Interval(0, 10)]
        )
        series_of_dicts = Series([{"a": 1}, {"a": 2}, {"b": 3}], index=index)
        frame = DataFrame.from_records(series_of_dicts, index=index)
        expected = DataFrame(
            {"a": [1, 2, np.nan], "b": [np.nan, np.nan, 3]}, index=index
        )
        tm.assert_frame_equal(frame, expected)

    def test_frame_from_records_utc(self):
        # [FUNCTION DOC]
        # Smoke test: from-records on a dictionary holding a time-zone-aware (UTC) datetime used as
        # the index column must not raise.
        rec = {"datum": 1.5, "begin_time": datetime(2006, 4, 27, tzinfo=timezone.utc)}

        # it works
        DataFrame.from_records([rec], index="begin_time")

    def test_from_records_to_records(self):
        # [FUNCTION DOC]
        # Round trip with a numpy structured array: from-records works with and without an explicit
        # reversed index, a plain two-dimensional array falls back to the normal constructor, an
        # index shorter than the data raises a value error about mismatched lengths, and setting
        # the index to a field then calling to-records gives three names, or two without the index.
        # from numpy documentation
        arr = np.zeros((2,), dtype=("i4,f4,S10"))
        arr[:] = [(1, 2.0, "Hello"), (2, 3.0, "World")]

        DataFrame.from_records(arr)

        index = Index(np.arange(len(arr))[::-1])
        indexed_frame = DataFrame.from_records(arr, index=index)
        tm.assert_index_equal(indexed_frame.index, index)

        # without names, it should go to last ditch
        arr2 = np.zeros((2, 3))
        tm.assert_frame_equal(DataFrame.from_records(arr2), DataFrame(arr2))

        # wrong length
        msg = "|".join(
            [
                r"Length of values \(2\) does not match length of index \(1\)",
            ]
        )
        with pytest.raises(ValueError, match=msg):
            DataFrame.from_records(arr, index=index[:-1])

        indexed_frame = DataFrame.from_records(arr, index="f1")

        # what to do?
        records = indexed_frame.to_records()
        assert len(records.dtype.names) == 3

        records = indexed_frame.to_records(index=False)
        assert len(records.dtype.names) == 2
        assert "index" not in records.dtype.names

    def test_from_records_nones(self):
        # [FUNCTION DOC]
        # Verifies None values in tuples become missing (not-a-number) in numeric columns.
        tuples = [(1, 2, None, 3), (1, 2, None, 3), (None, 2, 5, 3)]

        df = DataFrame.from_records(tuples, columns=["a", "b", "c", "d"])
        assert np.isnan(df["c"][0])

    def test_from_records_iterator(self):
        # [FUNCTION DOC]
        # Verifies from-records from an iterator of structured rows reads only the number of rows
        # requested by nrows, preserving field dtypes (compared after reindexing like the expected),
        # and for a plain list of tuples with column names the result matches ignoring dtype.
        arr = np.array(
            [(1.0, 1.0, 2, 2), (3.0, 3.0, 4, 4), (5.0, 5.0, 6, 6), (7.0, 7.0, 8, 8)],
            dtype=[
                ("x", np.float64),
                ("u", np.float32),
                ("y", np.int64),
                ("z", np.int32),
            ],
        )
        df = DataFrame.from_records(iter(arr), nrows=2)
        xp = DataFrame(
            {
                "x": np.array([1.0, 3.0], dtype=np.float64),
                "u": np.array([1.0, 3.0], dtype=np.float32),
                "y": np.array([2, 4], dtype=np.int64),
                "z": np.array([2, 4], dtype=np.int32),
            }
        )
        tm.assert_frame_equal(df.reindex_like(xp), xp)

        # no dtypes specified here, so just compare with the default
        arr = [(1.0, 2), (3.0, 4), (5.0, 6), (7.0, 8)]
        df = DataFrame.from_records(iter(arr), columns=["x", "y"], nrows=2)
        tm.assert_frame_equal(df, xp.reindex(columns=["x", "y"]), check_dtype=False)

    def test_from_records_tuples_generator(self):
        # [FUNCTION DOC]
        # Verifies from-records with a generator of tuples (integer, letter, fraction) and column
        # names builds the same frame as a dictionary of columns.
        def tuple_generator(length):
            # [FUNCTION DOC]
            # HELPER: generator yielding the given number of tuples of an integer, a cycling uppercase
            # letter and the fraction of the length.
            for i in range(length):
                letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                yield (i, letters[i % len(letters)], i / length)

        columns_names = ["Integer", "String", "Float"]
        columns = [
            [i[j] for i in tuple_generator(10)] for j in range(len(columns_names))
        ]
        data = {"Integer": columns[0], "String": columns[1], "Float": columns[2]}
        expected = DataFrame(data, columns=columns_names)

        generator = tuple_generator(10)
        result = DataFrame.from_records(generator, columns=columns_names)
        tm.assert_frame_equal(result, expected)

    def test_from_records_lists_generator(self):
        # [FUNCTION DOC]
        # Same as the tuple generator test but the generator yields lists instead of tuples.
        def list_generator(length):
            # [FUNCTION DOC]
            # HELPER: generator yielding the given number of lists of an integer, a cycling uppercase
            # letter and the fraction of the length.
            for i in range(length):
                letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                yield [i, letters[i % len(letters)], i / length]

        columns_names = ["Integer", "String", "Float"]
        columns = [
            [i[j] for i in list_generator(10)] for j in range(len(columns_names))
        ]
        data = {"Integer": columns[0], "String": columns[1], "Float": columns[2]}
        expected = DataFrame(data, columns=columns_names)

        generator = list_generator(10)
        result = DataFrame.from_records(generator, columns=columns_names)
        tm.assert_frame_equal(result, expected)

    def test_from_records_columns_not_modified(self):
        # [FUNCTION DOC]
        # Verifies from-records does not mutate the caller's list of column names when also given
        # an index column.
        tuples = [(1, 2, 3), (1, 2, 3), (2, 5, 3)]

        columns = ["a", "b", "c"]
        original_columns = list(columns)

        DataFrame.from_records(tuples, columns=columns, index="a")

        assert columns == original_columns

    def test_from_records_decimal(self):
        # [FUNCTION DOC]
        # Verifies decimal values with a None default to object dtype, and with coerce-float enabled
        # become 64-bit floats with the missing entry as not-a-number.
        tuples = [(Decimal("1.5"),), (Decimal("2.5"),), (None,)]

        df = DataFrame.from_records(tuples, columns=["a"])
        assert df["a"].dtype == object

        df = DataFrame.from_records(tuples, columns=["a"], coerce_float=True)
        assert df["a"].dtype == np.float64
        assert np.isnan(df["a"].values[-1])

    def test_from_records_duplicates(self):
        # [FUNCTION DOC]
        # Verifies duplicate column names in the columns argument are kept and the frame equals the
        # one from the normal constructor.
        result = DataFrame.from_records([(1, 2, 3), (4, 5, 6)], columns=["a", "b", "a"])

        expected = DataFrame([(1, 2, 3), (4, 5, 6)], columns=["a", "b", "a"])

        tm.assert_frame_equal(result, expected)

    def test_from_records_set_index_name(self):
        # [FUNCTION DOC]
        # Verifies from-records on a list of dictionaries (one lacking a key) sets the index name
        # from a single key and a tuple of names from a list of keys.
        def create_dict(order_id):
            # [FUNCTION DOC]
            # HELPER: builds a dictionary record with the given order id and two random integer values
            # (quantity and price) from a fixed-seed generator.
            return {
                "order_id": order_id,
                "quantity": np.random.default_rng(2).integers(1, 10),
                "price": np.random.default_rng(2).integers(1, 10),
            }

        documents = [create_dict(i) for i in range(10)]
        # demo missing data
        documents.append({"order_id": 10, "quantity": 5})

        result = DataFrame.from_records(documents, index="order_id")
        assert result.index.name == "order_id"

        # MultiIndex
        result = DataFrame.from_records(documents, index=["order_id", "quantity"])
        assert result.index.names == ("order_id", "quantity")

    def test_from_records_misc_brokenness(self):
        # [FUNCTION DOC]
        # Regression test for GH#2179: from-records on a dict with non-string keys and a different
        # columns list equals the plain constructor; a dict plus an index list that does not name
        # columns equals the constructor given the same index.
        # GH#2179

        data = {1: ["foo"], 2: ["bar"]}

        result = DataFrame.from_records(data, columns=["a", "b"])
        exp = DataFrame(data, columns=["a", "b"])
        tm.assert_frame_equal(result, exp)

        # overlap in index/index_names

        data = {"a": [1, 2, 3], "b": [4, 5, 6]}

        result = DataFrame.from_records(data, index=["a", "b", "c"])
        exp = DataFrame(data, index=["a", "b", "c"])
        tm.assert_frame_equal(result, exp)

    def test_from_records_misc_brokenness2(self):
        # [FUNCTION DOC]
        # Regression test for GH#2623: rows with a datetime and a value changing from integer to
        # string produce an object-dtype test column equal to the constructor result.
        # GH#2623
        rows = []
        rows.append([datetime(2010, 1, 1), 1])
        rows.append([datetime(2010, 1, 2), "hi"])  # test col upconverts to obj
        result = DataFrame.from_records(rows, columns=["date", "test"])
        expected = DataFrame(
            {"date": [row[0] for row in rows], "test": [row[1] for row in rows]}
        )
        tm.assert_frame_equal(result, expected)
        assert result.dtypes["test"] == np.dtype(object)

    def test_from_records_misc_brokenness3(self):
        # [FUNCTION DOC]
        # Companion case to the previous one with all integer values: from-records on rows equals
        # the constructor result.
        rows = []
        rows.append([datetime(2010, 1, 1), 1])
        rows.append([datetime(2010, 1, 2), 1])
        result = DataFrame.from_records(rows, columns=["date", "test"])
        expected = DataFrame(
            {"date": [row[0] for row in rows], "test": [row[1] for row in rows]}
        )
        tm.assert_frame_equal(result, expected)

    def test_from_records_empty(self):
        # [FUNCTION DOC]
        # Regression test for GH#3562: from-records on an empty list with column names, including a
        # repeated name, equals an empty frame with those columns.
        # GH#3562
        result = DataFrame.from_records([], columns=["a", "b", "c"])
        expected = DataFrame(columns=["a", "b", "c"])
        tm.assert_frame_equal(result, expected)

        result = DataFrame.from_records([], columns=["a", "b", "b"])
        expected = DataFrame(columns=["a", "b", "b"])
        tm.assert_frame_equal(result, expected)

    def test_from_records_empty_with_nonempty_fields_gh3682(self):
        # [FUNCTION DOC]
        # Regression test for GH#3682: a structured array with id and value fields indexed by id
        # gives a frame indexed by id with the value column; a zero-length slice gives an empty
        # frame equal to the first frame's empty slice.
        a = np.array([(1, 2)], dtype=[("id", np.int64), ("value", np.int64)])
        df = DataFrame.from_records(a, index="id")

        ex_index = Index([1], name="id")
        expected = DataFrame({"value": [2]}, index=ex_index, columns=["value"])
        tm.assert_frame_equal(df, expected)

        b = a[:0]
        df2 = DataFrame.from_records(b, index="id")
        tm.assert_frame_equal(df2, df.iloc[:0])

    def test_from_records_empty2(self):
        # [FUNCTION DOC]
        # Regression test for GH#42456: a zero-row structured array gives an empty frame with the
        # single field as an integer column, same through the plain constructor.
        # GH#42456
        dtype = [("prop", int)]
        shape = (0, len(dtype))
        arr = np.empty(shape, dtype=dtype)

        result = DataFrame.from_records(arr)
        expected = DataFrame({"prop": np.array([], dtype=int)})
        tm.assert_frame_equal(result, expected)

        alt = DataFrame(arr)
        tm.assert_frame_equal(alt, expected)

    def test_from_records_structured_array(self):
        # [FUNCTION DOC]
        # Regression test for GH 59717: from-records on a structured array with a columns subset
        # selects and orders the fields, keeping their original dtypes (32-bit ints) and string
        # values.
        # GH 59717
        data = np.array(
            [
                ("John", 25, "New York", 50000),
                ("Jane", 30, "San Francisco", 75000),
                ("Bob", 35, "Chicago", 65000),
                ("Alice", 28, "Los Angeles", 60000),
            ],
            dtype=[("name", "U10"), ("age", "i4"), ("city", "U15"), ("salary", "i4")],
        )

        actual_result = DataFrame.from_records(data, columns=["name", "salary", "city"])

        modified_data = {
            "name": ["John", "Jane", "Bob", "Alice"],
            "salary": np.array([50000, 75000, 65000, 60000], dtype="int32"),
            "city": ["New York", "San Francisco", "Chicago", "Los Angeles"],
        }
        expected_result = DataFrame(modified_data)

        tm.assert_frame_equal(actual_result, expected_result)

    def test_from_records_empty_iterator_with_preserve_columns(self):
        # [FUNCTION DOC]
        # Regression test for GH#61140: from-records on an empty iterator with an index, column
        # names and zero rows preserves the columns and index (equal to an empty frame built the
        # same way).
        # GH#61140
        rows = []
        result = DataFrame.from_records(
            iter(rows), index=[0, 1], columns=["col_1", "Col_2"], nrows=0
        )
        expected = DataFrame([], index=[0, 1], columns=["col_1", "Col_2"])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("missing_value", [None, np.nan, pd.NA])
    def test_from_records_missing_value_key(self, missing_value, using_infer_string):
        # [FUNCTION DOC]
        # Parametrized over a None, NaN and NA key. Verifies from-records on dictionaries with that
        # missing-value key keeps the data under a column label equal to the key (or NaN when the
        # string dtype inference mode is active, since a missing key becomes NaN).
        # https://github.com/pandas-dev/pandas/issues/63889
        # preserve values when None key is converted to NaN column name
        dict_data = [
            {"colA": 1, missing_value: 2},
            {"colA": 3, missing_value: 4},
        ]
        result = DataFrame.from_records(dict_data)
        expected = DataFrame(
            [[1, 2], [3, 4]],
            columns=["colA", np.nan if using_infer_string else missing_value],
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("missing_value", [None, np.nan, pd.NA])
    def test_from_records_missing_value_key_only(self, missing_value):
        # [FUNCTION DOC]
        # Parametrized over None, NaN and NA. Verifies from-records on dictionaries whose only key is
        # the missing value gives a single column labeled by that value.
        dict_data = [
            {missing_value: 1},
            {missing_value: 2},
        ]
        result = DataFrame.from_records(dict_data)
        expected = DataFrame([[1], [2]], columns=Index([missing_value]))
        tm.assert_frame_equal(result, expected)
