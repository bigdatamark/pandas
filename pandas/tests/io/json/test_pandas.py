# [MODULE DOC]
# Main tests for JSON reading and writing of frames and series (read_json and
# the to_json methods): round trips for every orientation, non-unique labels,
# dtype handling and conversion options, date formats and units (including the
# deprecated default epoch format, which emits a version-4 warning), timedelta
# handling, default handlers for unsupported objects, indentation, index
# inclusion options, nullable and Arrow dtype backends, lines (JSON Lines)
# mode, large numbers, URL and S3 access (network tests), and error messages.
# Many tests use the shared orient fixture from the sibling conftest and the
# fixtures for sample frames and series from the top-level test configuration.
# The module-level helper compares round-trip results and a few top-level
# tests follow the test class.

import datetime
from datetime import timedelta
from io import (
    BytesIO,
    StringIO,
)
import json
import os
import sys
import uuid

import numpy as np
import pytest

from pandas._config import using_string_dtype

from pandas.compat import IS64
from pandas.errors import Pandas4Warning
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    NA,
    DataFrame,
    DatetimeIndex,
    Index,
    RangeIndex,
    Series,
    Timestamp,
    date_range,
    read_json,
)
import pandas._testing as tm

from pandas.io.json import ujson_dumps


def test_literal_json_raises():
    # [FUNCTION DOC]
    # PR 53409. Passing literal JSON text (single or multi-line, with or without
    # lines mode, including escaped backslashes) to the reader is no longer
    # interpreted as data; each variant must raise a file-not-found error saying
    # the path does not exist.
    # PR 53409
    jsonl = """{"a": 1, "b": 2}
        {"a": 3, "b": 4}
        {"a": 5, "b": 6}
        {"a": 7, "b": 8}"""

    msg = r".* does not exist"

    with pytest.raises(FileNotFoundError, match=msg):
        read_json(jsonl, lines=False)

    with pytest.raises(FileNotFoundError, match=msg):
        read_json('{"a": 1, "b": 2}\n{"b":2, "a" :1}\n', lines=True)

    with pytest.raises(FileNotFoundError, match=msg):
        read_json(
            '{"a\\\\":"foo\\\\","b":"bar"}\n{"a\\\\":"foo\\"","b":"bar"}\n',
            lines=False,
        )

    with pytest.raises(FileNotFoundError, match=msg):
        read_json('{"a": 1, "b": 2}\n{"b":2, "a" :1}\n', lines=False)


def assert_json_roundtrip_equal(result, expected, orient):
    # [FUNCTION DOC]
    # HELPER: shared comparison for round-trip tests. Parameters: result, the frame
    # read back; expected, the original frame (modified in place for values orient);
    # orient, the orientation used. For records and values orientations the expected
    # index is replaced by a default range index (labels are not stored); for values
    # orientation the expected column labels are also replaced by a default range.
    # Then asserts the two frames are equal. Raises an assertion error on mismatch.
    if orient in ("records", "values"):
        expected = expected.reset_index(drop=True)
    if orient == "values":
        expected.columns = range(len(expected.columns))
    tm.assert_frame_equal(result, expected)


class TestPandasContainer:
    @pytest.fixture
    # [CLASS DOC]
    # Main collection of tests for JSON serialization and deserialization of frames
    # and series. Defines two fixtures with sample time-indexed data; the rest are
    # test methods.
    def datetime_series(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a ten-element float series named ts with a
        # nanosecond datetime index whose frequency is cleared (frequency does not
        # round-trip, GH#33711).
        # Same as usual datetime_series, but with index freq set to None,
        #  since that doesn't round-trip, see GH#33711
        ser = Series(
            1.1 * np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10, unit="ns"),
            name="ts",
        )
        ser.index = ser.index._with_freq(None)
        return ser

    @pytest.fixture
    def datetime_frame(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a 30 by 4 random normal frame with business-day
        # nanosecond datetime index whose frequency is cleared (does not round-trip,
        # GH#33711).
        # Same as usual datetime_frame, but with index freq set to None,
        #  since that doesn't round-trip, see GH#33711
        df = DataFrame(
            np.random.default_rng(2).standard_normal((30, 4)),
            columns=Index(list("ABCD")),
            index=date_range("2000-01-01", periods=30, freq="B", unit="ns"),
        )
        df.index = df.index._with_freq(None)
        return df

    def test_frame_double_encoded_labels(self, orient):
        # [FUNCTION DOC]
        # Frame whose labels contain quotes, slashes and backslashes round-trips for each
        # orientation (via the orient fixture), using the shared comparison helper.
        df = DataFrame(
            [["a", "b"], ["c", "d"]],
            index=['index " 1', "index / 2"],
            columns=["a \\ b", "y / z"],
        )

        data = StringIO(df.to_json(orient=orient))
        result = read_json(data, orient=orient)
        expected = df.copy()
        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.parametrize("orient", ["split", "records", "values"])
    def test_frame_non_unique_index(self, orient):
        # [FUNCTION DOC]
        # Parametrized on split, records and values orients. A frame with a duplicated
        # index round-trips; these orientations store the index as a list or not at all.
        df = DataFrame([["a", "b"], ["c", "d"]], index=[1, 1], columns=["x", "y"])
        data = StringIO(df.to_json(orient=orient))
        result = read_json(data, orient=orient)
        expected = df.copy()

        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.parametrize("orient", ["index", "columns"])
    def test_frame_non_unique_index_raises(self, orient):
        # [FUNCTION DOC]
        # Parametrized on index and columns orients. Writing a frame with duplicate index
        # labels raises a value error stating the index must be unique for that
        # orientation.
        df = DataFrame([["a", "b"], ["c", "d"]], index=[1, 1], columns=["x", "y"])
        msg = f"DataFrame index must be unique for orient='{orient}'"
        with pytest.raises(ValueError, match=msg):
            df.to_json(orient=orient)

    @pytest.mark.parametrize("orient", ["split", "values"])
    @pytest.mark.parametrize(
        "data",
        [
            [["a", "b"], ["c", "d"]],
            [[1.5, 2.5], [3.5, 4.5]],
            [[1, 2.5], [3, 4.5]],
            [
                [Timestamp("20130101").as_unit("s"), 3.5],
                [Timestamp("20130102").as_unit("s"), 4.5],
            ],
        ],
    )
    def test_frame_non_unique_columns(self, orient, data, request):
        # [FUNCTION DOC]
        # Parametrized on split/values orient and four data layouts (strings, floats,
        # mixed numbers, timestamps with floats). A frame with duplicated column names
        # is written and read with date conversion for the column. A pandas-4 deprecation
        # warning is expected only when the first column is second-resolution datetime.
        # For values orientation the expected frame has default column labels (and
        # timestamps are expressed as millisecond integers); for split orientation the
        # duplicate column gets a numeric suffix and datetimes become millisecond
        # resolution.
        # KNOWN ISSUE: the request fixture parameter is accepted but never used.
        df = DataFrame(data, index=[1, 2], columns=["x", "x"])

        expected_warning = None
        msg = (
            "The default 'epoch' date format is deprecated and will be removed "
            "in a future version, please use 'iso' date format instead."
        )
        if df.iloc[:, 0].dtype == "datetime64[s]":
            expected_warning = Pandas4Warning

        with tm.assert_produces_warning(expected_warning, match=msg):
            result = read_json(
                StringIO(df.to_json(orient=orient)), orient=orient, convert_dates=["x"]
            )
        if orient == "values":
            expected = DataFrame(data)
            if expected.iloc[:, 0].dtype == "datetime64[s]":
                # orient == "values" by default will write Timestamp objects out
                # in milliseconds;
                # TODO: a to_epoch method would also solve; see GH 14772
                dta = expected.iloc[:, 0]._values
                expected.isetitem(0, dta.as_unit("ms").astype(np.int64))
        elif orient == "split":
            expected = df
            expected.columns = ["x", "x.1"]
            if expected["x"].dtype.kind == "M":
                expected["x"] = expected["x"].astype("M8[ms]")

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("orient", ["index", "columns", "records"])
    def test_frame_non_unique_columns_raises(self, orient):
        # [FUNCTION DOC]
        # Parametrized on index, columns and records orients. Writing a frame with
        # duplicate column names raises a value error saying columns must be unique for
        # the orientation.
        df = DataFrame([["a", "b"], ["c", "d"]], index=[1, 2], columns=["x", "x"])

        msg = f"DataFrame columns must be unique for orient='{orient}'"
        with pytest.raises(ValueError, match=msg):
            df.to_json(orient=orient)

    def test_frame_default_orient(self, float_frame):
        # [FUNCTION DOC]
        # The default orientation of the frame writer is columns: output equals the
        # explicit columns orientation.
        assert float_frame.to_json() == float_frame.to_json(orient="columns")

    @pytest.mark.parametrize("dtype", [False, float])
    @pytest.mark.parametrize("convert_axes", [True, False])
    def test_roundtrip_simple(self, orient, convert_axes, dtype, float_frame):
        # [FUNCTION DOC]
        # Parametrized on orient fixture, convert_axes flag and dtype argument (false or
        # float). A float fixture frame round-trips equal.
        data = StringIO(float_frame.to_json(orient=orient))
        result = read_json(data, orient=orient, convert_axes=convert_axes, dtype=dtype)

        expected = float_frame

        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.parametrize("dtype", [False, np.int64])
    @pytest.mark.parametrize("convert_axes", [True, False])
    def test_roundtrip_intframe(self, orient, convert_axes, dtype, int_frame):
        # [FUNCTION DOC]
        # Parametrized on orient, convert_axes and dtype (false or 64-bit integer). An
        # integer fixture frame round-trips equal.
        data = StringIO(int_frame.to_json(orient=orient))
        result = read_json(data, orient=orient, convert_axes=convert_axes, dtype=dtype)
        expected = int_frame
        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.parametrize("dtype", [None, np.float64, int, "U3"])
    @pytest.mark.parametrize("convert_axes", [True, False])
    def test_roundtrip_str_axes(self, orient, convert_axes, dtype):
        # [FUNCTION DOC]
        # Parametrized on orient, convert_axes and dtype (none, float, int, short
        # string). A zero frame with numeric-looking string labels is written and read.
        # Expected result: values integer unless a dtype is given; when axes are
        # converted, labels that look numeric become integers for index/columns
        # orientation (both axes), records orientation (columns) and split orientation
        # (columns), because JSON keys are strings and numeric interpretation wins.
        df = DataFrame(
            np.zeros((200, 4)),
            columns=[str(i) for i in range(4)],
            index=[str(i) for i in range(200)],
            dtype=dtype,
        )

        data = StringIO(df.to_json(orient=orient))
        result = read_json(data, orient=orient, convert_axes=convert_axes, dtype=dtype)

        expected = df.copy()
        if not dtype:
            expected = expected.astype(np.int64)

        # index columns, and records orients cannot fully preserve the string
        # dtype for axes as the index and column labels are used as keys in
        # JSON objects. JSON keys are by definition strings, so there's no way
        # to disambiguate whether those keys actually were strings or numeric
        # beforehand and numeric wins out.
        if convert_axes and (orient in ("index", "columns")):
            expected.columns = expected.columns.astype(np.int64)
            expected.index = expected.index.astype(np.int64)
        elif orient == "records" and convert_axes:
            expected.columns = expected.columns.astype(np.int64)
        elif convert_axes and orient == "split":
            expected.columns = expected.columns.astype(np.int64)

        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.parametrize("convert_axes", [True, False])
    def test_roundtrip_categorical(
        self, request, orient, convert_axes, using_infer_string
    ):
        # [FUNCTION DOC]
        # Parametrized on orient and convert_axes. Expected to fail for index and columns
        # orients (duplicate index values). A frame with a categorical index (with
        # repeated categories) is written and read back; the categorical is not
        # preserved so the index becomes a string index (string dtype when inference is
        # on) and the index name is lost. A TODO asks for a better frame.
        # TODO: create a better frame to test with and improve coverage
        if orient in ("index", "columns"):
            request.applymarker(
                pytest.mark.xfail(
                    reason=f"Can't have duplicate index values for orient '{orient}')"
                )
            )

        data = {
            c: np.random.default_rng(i).standard_normal(30)
            for i, c in enumerate(list("ABCD"))
        }
        cat = ["bah"] * 5 + ["bar"] * 5 + ["baz"] * 5 + ["foo"] * 15
        data["E"] = list(reversed(cat))
        data["sort"] = np.arange(30, dtype="int64")
        categorical_frame = DataFrame(data, index=pd.CategoricalIndex(cat, name="E"))
        data = StringIO(categorical_frame.to_json(orient=orient))
        result = read_json(data, orient=orient, convert_axes=convert_axes)

        expected = categorical_frame.copy()
        expected.index = expected.index.astype(
            str if not using_infer_string else "str"
        )  # Categorical not preserved
        expected.index.name = None  # index names aren't preserved in JSON
        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.parametrize("convert_axes", [True, False])
    def test_roundtrip_empty(self, orient, convert_axes):
        # [FUNCTION DOC]
        # Parametrized on orient and convert_axes. An empty frame round-trips: for split
        # orientation to an empty frame with empty float (or object when axes are not
        # converted) labels, for index and columns to an empty frame, for others to the
        # original.
        empty_frame = DataFrame()
        data = StringIO(empty_frame.to_json(orient=orient))
        result = read_json(data, orient=orient, convert_axes=convert_axes)
        if orient == "split":
            idx = Index([], dtype=(float if convert_axes else object))
            expected = DataFrame(index=idx, columns=idx)
        elif orient in ["index", "columns"]:
            expected = DataFrame()
        else:
            expected = empty_frame.copy()

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "dtype_backend", [None, pytest.param("pyarrow", marks=td.skip_if_no("pyarrow"))]
    )
    @pytest.mark.parametrize("convert_axes", [True, False])
    def test_roundtrip_timestamp(
        self, orient, convert_axes, dtype_backend, datetime_frame
    ):
        # [FUNCTION DOC]
        # Parametrized on orient, convert_axes and dtype backend (none or pyarrow). A
        # frame with datetime index is written (optionally after converting the index
        # to an Arrow-backed type) and read. With convert_axes false the index arrives
        # as millisecond epoch integers (strings except for split orientation); with it
        # true, as datetimes of millisecond resolution. A TODO asks for coverage of the
        # date format parameter.
        # TODO: improve coverage with date_format parameter
        expected = datetime_frame.copy()
        if dtype_backend is not None:
            datetime_frame.index = Series(datetime_frame.index).convert_dtypes(
                dtype_backend=dtype_backend
            )
        data = StringIO(datetime_frame.to_json(orient=orient))
        result = read_json(data, orient=orient, convert_axes=convert_axes)

        if not convert_axes:  # one off for ts handling
            # DTI gets converted to epoch values
            idx = expected.index.view(np.int64) // 1000000
            if orient != "split":  # TODO: handle consistently across orients
                idx = idx.astype(str)

            expected.index = idx
        else:
            expected.index = expected.index.as_unit("ms")

        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.parametrize("convert_axes", [True, False])
    def test_roundtrip_mixed(self, orient, convert_axes):
        # [FUNCTION DOC]
        # Parametrized on orient and convert_axes. A frame of floats, strings and bools
        # with string index round-trips; numeric columns whose values are integral come
        # back as 64-bit integers.
        index = Index(["a", "b", "c", "d", "e"])
        values = {
            "A": [0.0, 1.0, 2.0, 3.0, 4.0],
            "B": [0.0, 1.0, 0.0, 1.0, 0.0],
            "C": ["foo1", "foo2", "foo3", "foo4", "foo5"],
            "D": [True, False, True, False, True],
        }

        df = DataFrame(data=values, index=index)

        data = StringIO(df.to_json(orient=orient))
        result = read_json(data, orient=orient, convert_axes=convert_axes)

        expected = df.copy()
        expected = expected.assign(**expected.select_dtypes("number").astype(np.int64))

        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.xfail(
        reason="#50456 Column multiindex is stored and loaded differently",
        raises=AssertionError,
    )
    @pytest.mark.parametrize(
        "columns",
        [
            [["2022", "2022"], ["JAN", "FEB"]],
            [["2022", "2023"], ["JAN", "JAN"]],
            [["2022", "2022"], ["JAN", "JAN"]],
        ],
    )
    def test_roundtrip_multiindex(self, columns):
        # [FUNCTION DOC]
        # Expected to fail (issue 50456; assertion error). Parametrized on three layouts
        # of two-level columns. Writing with split orientation and reading back does not
        # reproduce the multi-index columns.
        df = DataFrame(
            [[1, 2], [3, 4]],
            columns=pd.MultiIndex.from_arrays(columns),
        )
        data = StringIO(df.to_json(orient="split"))
        result = read_json(data, orient="split")
        tm.assert_frame_equal(result, df)

    @pytest.mark.parametrize(
        "data,msg,orient",
        [
            ('{"key":b:a:d}', "Expected object or value", "columns"),
            # too few indices
            (
                '{"columns":["A","B"],'
                '"index":["2","3"],'
                '"data":[[1.0,"1"],[2.0,"2"],[null,"3"]]}',
                "|".join(
                    [
                        r"Length of values \(3\) does not match length of index \(2\)",
                    ]
                ),
                "split",
            ),
            # too many columns
            (
                '{"columns":["A","B","C"],'
                '"index":["1","2","3"],'
                '"data":[[1.0,"1"],[2.0,"2"],[null,"3"]]}',
                "3 columns passed, passed data had 2 columns",
                "split",
            ),
            # bad key
            (
                '{"badkey":["A","B"],'
                '"index":["2","3"],'
                '"data":[[1.0,"1"],[2.0,"2"],[null,"3"]]}',
                r"unexpected key\(s\): badkey",
                "split",
            ),
        ],
    )
    def test_frame_from_json_bad_data_raises(self, data, msg, orient):
        # [FUNCTION DOC]
        # Parametrized on JSON text, expected message and orientation: malformed text,
        # data rows too many for the index, more column labels than data columns, and an
        # unexpected key in split format. Each must raise a value error with the message.
        with pytest.raises(ValueError, match=msg):
            read_json(StringIO(data), orient=orient)

    @pytest.mark.parametrize("dtype", [True, False])
    @pytest.mark.parametrize("convert_axes", [True, False])
    def test_frame_from_json_missing_data(self, orient, convert_axes, dtype):
        # [FUNCTION DOC]
        # Parametrized on orient, convert_axes and dtype flag. Ragged rows (numeric and
        # string) are written and read; the missing trailing entry of the short row is
        # NaN.
        num_df = DataFrame([[1, 2], [4, 5, 6]])

        result = read_json(
            StringIO(num_df.to_json(orient=orient)),
            orient=orient,
            convert_axes=convert_axes,
            dtype=dtype,
        )
        assert np.isnan(result.iloc[0, 2])

        obj_df = DataFrame([["1", "2"], ["4", "5", "6"]])
        result = read_json(
            StringIO(obj_df.to_json(orient=orient)),
            orient=orient,
            convert_axes=convert_axes,
            dtype=dtype,
        )
        assert np.isnan(result.iloc[0, 2])

    @pytest.mark.parametrize("dtype", [True, False])
    def test_frame_read_json_dtype_missing_value(self, dtype):
        # [FUNCTION DOC]
        # GH28501. Parametrized on dtype true/false. Reading a one-element list with a
        # null yields NaN; with dtype conversion off the column keeps object dtype.
        # GH28501 Parse missing values using read_json with dtype=False
        # to NaN instead of None
        result = read_json(StringIO("[null]"), dtype=dtype)
        expected = DataFrame([np.nan], dtype=object if not dtype else None)

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("inf", [np.inf, -np.inf])
    @pytest.mark.parametrize("dtype", [True, False])
    def test_frame_infinity(self, inf, dtype):
        # [FUNCTION DOC]
        # Parametrized on positive and negative infinity and dtype flag. An infinite
        # float is serialized as null and read back as NaN.
        # infinities get mapped to nulls which get mapped to NaNs during
        # deserialisation
        df = DataFrame([[1, 2], [4, 5, 6]])
        df.loc[0, 2] = inf

        data = StringIO(df.to_json())
        result = read_json(data, dtype=dtype)
        assert np.isnan(result.iloc[0, 2])

    @pytest.mark.skipif(not IS64, reason="not compliant on 32-bit, xref #15865")
    @pytest.mark.parametrize(
        "value,precision,expected_val",
        [
            (0.95, 1, 1.0),
            (1.95, 1, 2.0),
            (-1.95, 1, -2.0),
            (0.995, 2, 1.0),
            (0.9995, 3, 1.0),
            (0.99999999999999944, 15, 1.0),
        ],
    )
    def test_frame_to_json_float_precision(self, value, precision, expected_val):
        # [FUNCTION DOC]
        # Skipped on 32-bit platforms. Parametrized on value, double precision and
        # expected rounded value. Writing a float with limited decimal precision rounds
        # it as expected in the exact output text.
        df = DataFrame([{"a_float": value}])
        encoded = df.to_json(double_precision=precision)
        assert encoded == f'{{"a_float":{{"0":{expected_val}}}}}'

    def test_frame_to_json_except(self):
        # [FUNCTION DOC]
        # An invalid orient name raises a value error naming the invalid option value.
        df = DataFrame([1, 2, 3])
        msg = "Invalid value 'garbage' for option 'orient'"
        with pytest.raises(ValueError, match=msg):
            df.to_json(orient="garbage")

    def test_frame_empty(self):
        # [FUNCTION DOC]
        # A frame with columns but no rows is not mixed-type and round-trips when the
        # dtypes are provided back to the reader (index type check relaxed).
        df = DataFrame(columns=["jim", "joe"])
        assert not df._is_mixed_type

        data = StringIO(df.to_json())
        result = read_json(data, dtype=dict(df.dtypes))
        tm.assert_frame_equal(result, df, check_index_type=False)

    def test_frame_empty_to_json(self):
        # [FUNCTION DOC]
        # GH 7445. An empty frame with a column and empty index is serialized as an
        # object holding the column mapped to an empty mapping.
        # GH 7445
        df = DataFrame({"test": []}, index=[])
        result = df.to_json(orient="columns")
        expected = '{"test":{}}'
        assert result == expected

    def test_frame_empty_mixedtype(self):
        # [FUNCTION DOC]
        # An empty frame with one column cast to 64-bit integer is mixed-type and
        # round-trips when dtypes are passed to the reader.
        # mixed type
        df = DataFrame(columns=["jim", "joe"])
        df["joe"] = df["joe"].astype("i8")
        assert df._is_mixed_type
        data = df.to_json()
        tm.assert_frame_equal(
            read_json(StringIO(data), dtype=dict(df.dtypes)),
            df,
            check_index_type=False,
        )

    def test_frame_mixedtype_orient(self):  # GH10289
        # [FUNCTION DOC]
        # GH10289. A mixed-type frame round-trips for split, index and columns orients
        # without axis conversion; then for records orient with a default row index; then
        # for values orient with default row and column labels.
        vals = [
            [10, 1, "foo", 0.1, 0.01],
            [20, 2, "bar", 0.2, 0.02],
            [30, 3, "baz", 0.3, 0.03],
            [40, 4, "qux", 0.4, 0.04],
        ]

        df = DataFrame(
            vals, index=list("abcd"), columns=["1st", "2nd", "3rd", "4th", "5th"]
        )

        assert df._is_mixed_type
        right = df.copy()

        for orient in ["split", "index", "columns"]:
            inp = StringIO(df.to_json(orient=orient))
            left = read_json(inp, orient=orient, convert_axes=False)
            tm.assert_frame_equal(left, right)

        right.index = RangeIndex(len(df))
        inp = StringIO(df.to_json(orient="records"))
        left = read_json(inp, orient="records", convert_axes=False)
        tm.assert_frame_equal(left, right)

        right.columns = RangeIndex(df.shape[1])
        inp = StringIO(df.to_json(orient="values"))
        left = read_json(inp, orient="values", convert_axes=False)
        tm.assert_frame_equal(left, right)

    def test_v12_compat(self, datapath):
        # [FUNCTION DOC]
        # Reads two stored JSON files written by a very old version (epoch dates and ISO
        # dates) and compares with a frame constructed in code: the epoch file yields
        # nanosecond date columns, the ISO file yields microsecond units and lacks the
        # modified column. Depends on the data path fixture.
        dti = date_range("2000-01-03", "2000-01-07", unit="ns")
        # freq doesn't roundtrip
        dti = DatetimeIndex(np.asarray(dti), freq=None)
        df = DataFrame(
            [
                [1.56808523, 0.65727391, 1.81021139, -0.17251653],
                [-0.2550111, -0.08072427, -0.03202878, -0.17581665],
                [1.51493992, 0.11805825, 1.629455, -1.31506612],
                [-0.02765498, 0.44679743, 0.33192641, -0.27885413],
                [0.05951614, -2.69652057, 1.28163262, 0.34703478],
            ],
            columns=["A", "B", "C", "D"],
            index=dti,
        )
        df["date"] = Timestamp("19920106 18:21:32.12").as_unit("ns")
        df.iloc[3, df.columns.get_loc("date")] = Timestamp("20130101")
        df["modified"] = df["date"]
        df.iloc[1, df.columns.get_loc("modified")] = pd.NaT

        dirpath = datapath("io", "json", "data")
        v12_json = os.path.join(dirpath, "tsframe_v012.json")
        result = read_json(v12_json)
        expected = df.copy()
        expected["date"] = expected["date"].dt.as_unit("ns")
        expected["modified"] = expected["modified"].dt.as_unit("ns")
        expected.index = expected.index.as_unit("ns")
        tm.assert_frame_equal(result, expected)

        v12_iso_json = os.path.join(dirpath, "tsframe_iso_v012.json")
        result_iso = read_json(v12_iso_json)
        expected_iso = df.drop(["modified"], axis=1)
        expected_iso["date"] = expected_iso["date"].dt.as_unit("us")
        expected_iso.index = expected_iso.index.as_unit("us")
        tm.assert_frame_equal(result_iso, expected_iso, check_column_type=False)

    def test_blocks_compat_GH9037(self, using_infer_string):
        # [FUNCTION DOC]
        # A ten-row frame with floats, integers and strings (hex strings), hourly
        # datetime index, written with split orientation and read back; compared
        # strictly (exact values, block-wise, index and column types) after converting
        # the expected columns to string type and the index to millisecond resolution.
        index = date_range("20000101", periods=10, freq="h", unit="ns")
        # freq doesn't round-trip
        index = DatetimeIndex(list(index), freq=None)

        df_mixed = DataFrame(
            {
                "float_1": [
                    -0.92077639,
                    0.77434435,
                    1.25234727,
                    0.61485564,
                    -0.60316077,
                    0.24653374,
                    0.28668979,
                    -2.51969012,
                    0.95748401,
                    -1.02970536,
                ],
                "int_1": [
                    19680418,
                    75337055,
                    99973684,
                    65103179,
                    79373900,
                    40314334,
                    21290235,
                    4991321,
                    41903419,
                    16008365,
                ],
                "str_1": [
                    "78c608f1",
                    "64a99743",
                    "13d2ff52",
                    "ca7f4af2",
                    "97236474",
                    "bde7e214",
                    "1a6bde47",
                    "b1190be5",
                    "7a669144",
                    "8d64d068",
                ],
                "float_2": [
                    -0.0428278,
                    -1.80872357,
                    3.36042349,
                    -0.7573685,
                    -0.48217572,
                    0.86229683,
                    1.08935819,
                    0.93898739,
                    -0.03030452,
                    1.43366348,
                ],
                "str_2": [
                    "14f04af9",
                    "d085da90",
                    "4bcfac83",
                    "81504caf",
                    "2ffef4a9",
                    "08e2f5c4",
                    "07e1af03",
                    "addbd4a7",
                    "1f6a09ba",
                    "4bfc4d87",
                ],
                "int_2": [
                    86967717,
                    98098830,
                    51927505,
                    20372254,
                    12601730,
                    20884027,
                    34193846,
                    10561746,
                    24867120,
                    76131025,
                ],
            },
            index=index,
        )

        # JSON deserialisation always creates unicode strings
        df_mixed.columns = df_mixed.columns.astype(
            np.str_ if not using_infer_string else "str"
        )
        data = StringIO(df_mixed.to_json(orient="split"))
        df_roundtrip = read_json(data, orient="split")
        df_mixed.index = df_mixed.index.as_unit("ms")
        tm.assert_frame_equal(
            df_mixed,
            df_roundtrip,
            check_index_type=True,
            check_column_type=True,
            by_blocks=True,
            check_exact=True,
        )

    def test_frame_nonprintable_bytes(self):
        # GH14256: failing column caused segfaults, if it is not the last one

        # [FUNCTION DOC]
        # GH14256. Objects whose string form is hexadecimal text but whose binary payload
        # is not valid UTF-8: printable content serializes fine; an object column
        # holding such objects raises an overflow error for unsupported UTF-8 length,
        # also with multiple columns (formerly segfaulted); with a default handler of
        # string conversion the output contains the hex text.
        class BinaryThing:
            # [CLASS DOC]
            # HELPER: test double used by the enclosing test: an object that remembers a
            # hex text and the corresponding binary bytes, and whose string form is the hex
            # text.
            def __init__(self, hexed) -> None:
                # [FUNCTION DOC]
                # HELPER: stores the given hex text and the bytes decoded from it. Parameter
                # hexed: a string of hexadecimal digits; raises a value error if it is not
                # valid hexadecimal. Returns nothing.
                self.hexed = hexed
                self.binary = bytes.fromhex(hexed)

            def __str__(self) -> str:
                # [FUNCTION DOC]
                # HELPER: returns the stored hex text.
                return self.hexed

        hexed = "574b4454ba8c5eb4f98a8f45"
        binthing = BinaryThing(hexed)

        # verify the proper conversion of printable content
        df_printable = DataFrame({"A": [binthing.hexed]})
        assert df_printable.to_json() == f'{{"A":{{"0":"{hexed}"}}}}'

        # check if non-printable content throws appropriate Exception
        df_nonprintable = DataFrame({"A": [binthing]})
        msg = "Unsupported UTF-8 sequence length when encoding string"
        with pytest.raises(OverflowError, match=msg):
            df_nonprintable.to_json()

        # the same with multiple columns threw segfaults
        df_mixed = DataFrame({"A": [binthing], "B": [1]}, columns=["A", "B"])
        with pytest.raises(OverflowError, match=msg):
            df_mixed.to_json()

        # default_handler should resolve exceptions for non-string types
        result = df_nonprintable.to_json(default_handler=str)
        expected = f'{{"A":{{"0":"{hexed}"}}}}'
        assert result == expected
        assert (
            df_mixed.to_json(default_handler=str)
            == f'{{"A":{{"0":"{hexed}"}},"B":{{"0":1}}}}'
        )

    def test_label_overflow(self):
        # [FUNCTION DOC]
        # GH14256. A column label of 300000 characters is written without overflowing
        # the encoder's buffer; exact output compared.
        # GH14256: buffer length not checked when writing label
        result = DataFrame({"bar" * 100000: [1], "foo": [1337]}).to_json()
        expected = f'{{"{"bar" * 100000}":{{"0":1}},"foo":{{"0":1337}}}}'
        assert result == expected

    def test_series_non_unique_index(self):
        # [FUNCTION DOC]
        # A series with duplicate index: index orientation raises a value error (index
        # must be unique). Split orientation round-trips equal, records orientation
        # returns the same values.
        s = Series(["a", "b"], index=[1, 1])

        msg = "Series index must be unique for orient='index'"
        with pytest.raises(ValueError, match=msg):
            s.to_json(orient="index")

        tm.assert_series_equal(
            s,
            read_json(
                StringIO(s.to_json(orient="split")), orient="split", typ="series"
            ),
        )
        unserialized = read_json(
            StringIO(s.to_json(orient="records")), orient="records", typ="series"
        )
        tm.assert_equal(s.values, unserialized.values)

    def test_series_default_orient(self, string_series):
        # [FUNCTION DOC]
        # The default series orientation is index.
        assert string_series.to_json() == string_series.to_json(orient="index")

    def test_series_roundtrip_simple(self, orient, string_series, using_infer_string):
        # [FUNCTION DOC]
        # Parametrized on orient. A fixture string-indexed series round-trips; for
        # values and records the index is reset; for non-split orientations the name is
        # lost; under string inference the index becomes string dtype for orientations
        # that store labels as keys.
        data = StringIO(string_series.to_json(orient=orient))
        result = read_json(data, typ="series", orient=orient)

        expected = string_series
        if using_infer_string and orient in ("split", "index", "columns"):
            # These schemas don't contain dtypes, so we infer string
            expected.index = expected.index.astype("str")
        if orient in ("values", "records"):
            expected = expected.reset_index(drop=True)
        if orient != "split":
            expected.name = None

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("dtype", [False, None])
    def test_series_roundtrip_object(self, orient, dtype, object_series):
        # [FUNCTION DOC]
        # Parametrized on orient and dtype (false or none). An object series round-trips
        # with the same index and name rules; if the string dtype is the default the
        # expected is converted to string dtype.
        data = StringIO(object_series.to_json(orient=orient))
        result = read_json(data, typ="series", orient=orient, dtype=dtype)

        expected = object_series
        if orient in ("values", "records"):
            expected = expected.reset_index(drop=True)
        if orient != "split":
            expected.name = None

        if using_string_dtype():
            expected = expected.astype("str")

        tm.assert_series_equal(result, expected)

    def test_series_roundtrip_empty(self, orient):
        # [FUNCTION DOC]
        # Parametrized on orient. An empty float series round-trips; for split the
        # expected index is float typed.
        empty_series = Series([], index=[], dtype=np.float64)
        data = StringIO(empty_series.to_json(orient=orient))
        result = read_json(data, typ="series", orient=orient)

        expected = empty_series.reset_index(drop=True)
        if orient in ("split"):
            expected.index = expected.index.astype(np.float64)

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "dtype_backend", [None, pytest.param("pyarrow", marks=td.skip_if_no("pyarrow"))]
    )
    def test_series_roundtrip_timeseries(self, dtype_backend, orient, datetime_series):
        # [FUNCTION DOC]
        # Parametrized on dtype backend (none or pyarrow) and orient. A datetime-indexed
        # series round-trips; the index becomes millisecond resolution except for values
        # and records, where it is reset; the name is kept only for split.
        expected = datetime_series.copy()
        if dtype_backend is not None:
            datetime_series.index = Series(datetime_series.index).convert_dtypes(
                dtype_backend=dtype_backend
            )

        data = StringIO(datetime_series.to_json(orient=orient))
        result = read_json(data, typ="series", orient=orient)

        if orient in ("values", "records"):
            expected = expected.reset_index(drop=True)
        else:
            expected.index = expected.index.as_unit("ms")
        if orient != "split":
            expected.name = None

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.float64, int])
    def test_series_roundtrip_numeric(self, orient, dtype):
        # [FUNCTION DOC]
        # Parametrized on a dtype (float or int) and orient. A six-element integer series
        # with letter labels round-trips equal (index reset for values and records).
        # KNOWN ISSUE: the dtype parameter is never used in the body, so both parametrizations run identical code.
        s = Series(range(6), index=["a", "b", "c", "d", "e", "f"])
        data = StringIO(s.to_json(orient=orient))
        result = read_json(data, typ="series", orient=orient)

        expected = s.copy()
        if orient in ("values", "records"):
            expected = expected.reset_index(drop=True)

        tm.assert_series_equal(result, expected)

    def test_series_to_json_except(self):
        # [FUNCTION DOC]
        # An invalid orient name raises a value error naming the invalid option value.
        s = Series([1, 2, 3])
        msg = "Invalid value 'garbage' for option 'orient'"
        with pytest.raises(ValueError, match=msg):
            s.to_json(orient="garbage")

    def test_series_from_json_precise_float(self):
        # [FUNCTION DOC]
        # Reading a float series with the precise float option gives an equal series.
        s = Series([4.56, 4.56, 4.56])
        result = read_json(StringIO(s.to_json()), typ="series", precise_float=True)
        tm.assert_series_equal(result, s, check_index_type=False)

    def test_series_with_dtype(self):
        # [FUNCTION DOC]
        # GH 21986. Reading a float series with an integer dtype truncates to integers.
        # GH 21986
        s = Series([4.56, 4.56, 4.56])
        result = read_json(StringIO(s.to_json()), typ="series", dtype=np.int64)
        expected = Series([4] * 3)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "dtype_backend", [None, pytest.param("pyarrow", marks=td.skip_if_no("pyarrow"))]
    )
    @pytest.mark.parametrize(
        "dtype,expected",
        [
            (True, Series(["2000-01-01"], dtype="datetime64[ms]")),
            (False, Series([946684800000])),
        ],
    )
    def test_series_with_dtype_datetime(self, dtype, expected, dtype_backend):
        # [FUNCTION DOC]
        # Parametrized on dtype backend and on dtype flag. A datetime series written with
        # the default (deprecated epoch) format emits the pandas-4 warning; reading with
        # dtype true converts to millisecond datetimes, with false leaves the epoch
        # integer.
        s = Series(["2000-01-01"], dtype="datetime64[ns]")
        if dtype_backend is not None:
            s = s.convert_dtypes(dtype_backend=dtype_backend)
        msg = (
            "The default 'epoch' date format is deprecated and will be removed "
            "in a future version, please use 'iso' date format instead."
        )
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            data = StringIO(s.to_json())
        result = read_json(data, typ="series", dtype=dtype)
        tm.assert_series_equal(result, expected)

    def test_frame_from_json_precise_float(self):
        # [FUNCTION DOC]
        # Reading a float frame with the precise float option gives an equal frame.
        df = DataFrame([[4.56, 4.56, 4.56], [4.56, 4.56, 4.56]])
        result = read_json(StringIO(df.to_json()), precise_float=True)
        tm.assert_frame_equal(result, df)

    def test_typ(self):
        # [FUNCTION DOC]
        # Reading with series typ returns an equal 64-bit integer series.
        s = Series(range(6), index=["a", "b", "c", "d", "e", "f"], dtype="int64")
        result = read_json(StringIO(s.to_json()), typ="series")
        tm.assert_series_equal(result, s)

    def test_reconstruction_index(self):
        # [FUNCTION DOC]
        # Default frame round trips with default and with string index.
        df = DataFrame([[1, 2, 3], [4, 5, 6]])
        result = read_json(StringIO(df.to_json()))
        tm.assert_frame_equal(result, df)

        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=["A", "B", "C"])
        result = read_json(StringIO(df.to_json()))
        tm.assert_frame_equal(result, df)

    def test_path(self, float_frame, int_frame, datetime_frame, temp_file):
        # [FUNCTION DOC]
        # Float, integer and datetime fixture frames can be written to a path and read back
        # without error. Nothing is compared.
        for df in [float_frame, int_frame, datetime_frame]:
            df.to_json(temp_file)
            read_json(temp_file)

    def test_axis_dates(self, datetime_series, datetime_frame):
        # [FUNCTION DOC]
        # Datetime-indexed frame and series round-trip with default orientation; index
        # becomes millisecond resolution and the series name is lost.
        # frame
        json = StringIO(datetime_frame.to_json())
        result = read_json(json)
        expected = datetime_frame.copy()
        expected.index = expected.index.as_unit("ms")
        tm.assert_frame_equal(result, expected)

        # series
        json = StringIO(datetime_series.to_json())
        result = read_json(json, typ="series")
        expected = datetime_series.copy()
        expected.index = expected.index.as_unit("ms")
        tm.assert_series_equal(result, expected, check_names=False)
        assert result.name is None

    def test_convert_dates(self, datetime_series, datetime_frame):
        # [FUNCTION DOC]
        # A frame with an extra date column written with default settings (deprecated
        # epoch warning) reads back with millisecond dates in the column and index; with
        # date conversion disabled the epoch values stay integers; a series of dates
        # round-trips similarly.
        # frame
        df = datetime_frame
        df["date"] = Timestamp("20130101").as_unit("ns")

        msg = (
            "The default 'epoch' date format is deprecated and will be removed "
            "in a future version, please use 'iso' date format instead."
        )
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            json = StringIO(df.to_json())
        result = read_json(json)
        expected = df.copy()
        expected["date"] = expected["date"].dt.as_unit("ms")
        expected.index = expected.index.as_unit("ms")
        tm.assert_frame_equal(result, expected)

        df["foo"] = 1.0
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            json = StringIO(df.to_json(date_unit="ns"))

        result = read_json(json, convert_dates=False)
        expected = df.copy()
        expected["date"] = expected["date"].values.view("i8")
        expected["foo"] = expected["foo"].astype("int64")
        tm.assert_frame_equal(result, expected)

        # series
        ts = Series(Timestamp("20130101").as_unit("ns"), index=datetime_series.index)
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            json = StringIO(ts.to_json())
        result = read_json(json, typ="series")
        expected = ts.dt.as_unit("ms")
        expected.index = expected.index.as_unit("ms")
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("date_format", ["epoch", "iso"])
    @pytest.mark.parametrize("as_object", [True, False])
    @pytest.mark.parametrize("date_typ", [datetime.date, datetime.datetime, Timestamp])
    def test_date_index_and_values(self, date_format, as_object, date_typ):
        # [FUNCTION DOC]
        # Parametrized on date format (epoch, iso), whether data are object dtype, and
        # date type (date, datetime, timestamp). A series whose index and values are
        # dates with a missing value (and optionally a string) is serialized; the exact
        # expected JSON depends on the format (epoch emits the deprecation warning).
        data = [date_typ(year=2020, month=1, day=1), pd.NaT]
        if as_object:
            data.append("a")

        ser = Series(data, index=data)
        if not as_object:
            ser = ser.astype("M8[ns]")
            if isinstance(ser.index, DatetimeIndex):
                ser.index = ser.index.as_unit("ns")

        expected_warning = None
        if date_format == "epoch":
            expected = '{"1577836800000":1577836800000,"null":null}'
            expected_warning = Pandas4Warning
        else:
            expected = (
                '{"2020-01-01T00:00:00.000":"2020-01-01T00:00:00.000","null":null}'
            )

        msg = (
            "'epoch' date format is deprecated and will be removed in a future "
            "version, please use 'iso' date format instead."
        )
        with tm.assert_produces_warning(expected_warning, match=msg):
            result = ser.to_json(date_format=date_format)

        if as_object:
            expected = expected.replace("}", ',"a":"a"}')

        assert result == expected

    @pytest.mark.parametrize(
        "infer_word",
        [
            "trade_time",
            "date",
            "datetime",
            "sold_at",
            "modified",
            "timestamp",
            "timestamps",
        ],
    )
    def test_convert_dates_infer(self, infer_word):
        # GH10747

        # [FUNCTION DOC]
        # GH10747. Parametrized on column names that suggest dates (time, date,
        # datetime, sold_at, modified, timestamp, timestamps). An integer epoch value in
        # such a column is converted to a datetime (millisecond) with missing values for
        # absent rows, while the id column is left alone.
        data = [{"id": 1, infer_word: 1036713600000}, {"id": 2}]
        expected = DataFrame(
            [[1, Timestamp("2002-11-08")], [2, pd.NaT]], columns=["id", infer_word]
        )
        expected[infer_word] = expected[infer_word].astype("M8[ms]")

        result = read_json(StringIO(ujson_dumps(data)))[["id", infer_word]]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "date,date_unit",
        [
            ("20130101 20:43:42.123", None),
            ("20130101 20:43:42", "s"),
            ("20130101 20:43:42.123", "ms"),
            ("20130101 20:43:42.123456", "us"),
            ("20130101 20:43:42.123456789", "ns"),
        ],
    )
    def test_date_format_frame(self, date, date_unit, datetime_frame):
        # [FUNCTION DOC]
        # Parametrized on a date text and date unit (none, s, ms, us, ns). A frame with a
        # date column containing missing values is written in ISO format with that unit and
        # read back; with units other than nanoseconds the result has microsecond
        # resolution for the column and index.
        df = datetime_frame

        df["date"] = Timestamp(date).as_unit("ns")
        df.iloc[1, df.columns.get_loc("date")] = pd.NaT
        df.iloc[5, df.columns.get_loc("date")] = pd.NaT
        if date_unit:
            json = df.to_json(date_format="iso", date_unit=date_unit)
        else:
            json = df.to_json(date_format="iso")

        result = read_json(StringIO(json))
        expected = df.copy()
        if date_unit != "ns":
            expected["date"] = expected["date"].dt.as_unit("us")
            expected.index = expected.index.as_unit("us")
        tm.assert_frame_equal(result, expected)

    def test_date_format_frame_raises(self, datetime_frame):
        # [FUNCTION DOC]
        # An invalid date unit raises a value error naming the invalid option value.
        df = datetime_frame
        msg = "Invalid value 'foo' for option 'date_unit'"
        with pytest.raises(ValueError, match=msg):
            df.to_json(date_format="iso", date_unit="foo")

    @pytest.mark.parametrize(
        "date,date_unit",
        [
            ("20130101 20:43:42.123", None),
            ("20130101 20:43:42", "s"),
            ("20130101 20:43:42.123", "ms"),
            ("20130101 20:43:42.123456", "us"),
            ("20130101 20:43:42.123456789", "ns"),
        ],
    )
    def test_date_format_series(self, date, date_unit, datetime_series):
        # [FUNCTION DOC]
        # Same as the frame test for a series with datetime values and index.
        ts = Series(Timestamp(date), index=datetime_series.index)
        ts.iloc[1] = pd.NaT
        ts.iloc[5] = pd.NaT
        if date_unit:
            json = ts.to_json(date_format="iso", date_unit=date_unit)
        else:
            json = ts.to_json(date_format="iso")

        result = read_json(StringIO(json), typ="series")
        expected = ts.copy()
        if date_unit != "ns":
            expected.index = expected.index.as_unit("us")
        tm.assert_series_equal(result, expected)

    def test_date_format_series_raises(self, datetime_series):
        # [FUNCTION DOC]
        # An invalid date unit for a series raises a value error.
        ts = Series(Timestamp("20130101 20:43:42.123"), index=datetime_series.index)
        msg = "Invalid value 'foo' for option 'date_unit'"
        with pytest.raises(ValueError, match=msg):
            ts.to_json(date_format="iso", date_unit="foo")

    def test_date_unit(self, unit, datetime_frame):
        # [FUNCTION DOC]
        # Given the unit fixture. A frame whose date column has varied values including
        # dates before 1970 and after 2100 and a missing value is written with epoch
        # format and that date unit (deprecation warning expected); reading with automatic
        # unit detection must reproduce dates at that unit. A commented-out line shows an
        # earlier check with a forced unit.
        df = datetime_frame
        df["date"] = Timestamp("20130101 20:43:42").as_unit("ns")
        dl = df.columns.get_loc("date")
        df.iloc[1, dl] = Timestamp("19710101 20:43:42")
        df.iloc[2, dl] = Timestamp("21460101 20:43:42")
        df.iloc[4, dl] = pd.NaT

        msg = (
            "'epoch' date format is deprecated and will be removed in a future "
            "version, please use 'iso' date format instead."
        )
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            json = df.to_json(date_format="epoch", date_unit=unit)

        # force date unit
        # result = read_json(StringIO(json), date_unit=unit)
        expected = df.copy()
        expected["date"] = expected["date"].dt.as_unit(unit)
        expected.index = expected.index.as_unit(unit)
        # tm.assert_frame_equal(result, expected)

        # detect date unit
        result = read_json(StringIO(json), date_unit=None)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "df, warn",
        [
            (DataFrame({"A": ["a", "b", "c"], "B": np.arange(3)}), None),
            (DataFrame({"A": [True, False, False]}), None),
            (
                DataFrame(
                    {"A": ["a", "b", "c"], "B": pd.to_timedelta(np.arange(3), unit="D")}
                ),
                Pandas4Warning,
            ),
            (
                DataFrame(
                    {"A": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])}
                ),
                Pandas4Warning,
            ),
        ],
    )
    def test_default_epoch_date_format_deprecated(self, df, warn):
        # [FUNCTION DOC]
        # GH 57063. Parametrized on frame and expected warning. Writing with the default
        # date format emits the pandas-4 deprecation warning only when the frame holds
        # datetime or timedelta columns (not for strings, numbers, booleans).
        # GH 57063
        msg = (
            "The default 'epoch' date format is deprecated and will be removed "
            "in a future version, please use 'iso' date format instead."
        )
        with tm.assert_produces_warning(warn, match=msg):
            df.to_json()

    @pytest.mark.parametrize(
        "dtype_backend", [None, pytest.param("pyarrow", marks=td.skip_if_no("pyarrow"))]
    )
    @pytest.mark.parametrize("unit", ["s", "ms", "us"])
    def test_iso_non_nano_datetimes(self, dtype_backend, unit):
        # [FUNCTION DOC]
        # GH53686. Parametrized on dtype backend and unit (s, ms, us). A frame with
        # datetime columns and index of non-nanosecond resolution (one object-typed) is
        # written in ISO format with that unit and read with date conversion for both
        # columns; result equals the original converted to microsecond resolution.
        # Test that numpy datetimes
        # in an Index or a column with non-nano resolution can be serialized
        # correctly
        # GH53686
        index = DatetimeIndex(
            [np.datetime64("2023-01-01T11:22:33.123456", unit)],
            dtype=f"datetime64[{unit}]",
        )
        df = DataFrame(
            {
                "date": Series(
                    [np.datetime64("2022-01-01T11:22:33.123456", unit)],
                    dtype=f"datetime64[{unit}]",
                    index=index,
                ),
                "date_obj": Series(
                    [np.datetime64("2023-01-01T11:22:33.123456", unit)],
                    dtype=object,
                    index=index,
                ),
            },
        )
        expected = df.copy()
        expected.index = expected.index.as_unit("us")
        expected["date"] = expected["date"].dt.as_unit("us")
        expected["date_obj"] = expected["date_obj"].astype("datetime64[us]")

        if dtype_backend is not None:
            df = df.convert_dtypes(dtype_backend=dtype_backend)

        json = df.to_json(date_format="iso", date_unit=unit)
        result = read_json(StringIO(json), convert_dates=["date", "date_obj"])
        tm.assert_frame_equal(result, expected)

    def test_weird_nested_json(self):
        # [FUNCTION DOC]
        # A nested JSON document that used to crash the parser can be read without error.
        # this used to core dump the parser
        s = r"""{
        "status": "success",
        "data": {
        "posts": [
            {
            "id": 1,
            "title": "A blog post",
            "body": "Some useful content"
            },
            {
            "id": 2,
            "title": "Another blog post",
            "body": "More content"
            }
           ]
          }
        }"""
        read_json(StringIO(s))

    def test_doc_example(self):
        # [FUNCTION DOC]
        # Builds the documentation example frame (floats, a date, ints, bools, date index),
        # serializes it with the deprecated default (warning expected) and reads it with
        # specific dtypes.
        # KNOWN ISSUE: the final assertion compares the result with itself, so it can never fail and does not verify the round trip.
        dfj2 = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), columns=list("AB")
        )
        dfj2["date"] = Timestamp("20130101")
        dfj2["ints"] = range(5)
        dfj2["bools"] = True
        dfj2.index = date_range("20130101", periods=5, unit="ns")

        msg = (
            "The default 'epoch' date format is deprecated and will be removed "
            "in a future version, please use 'iso' date format instead."
        )
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            json = StringIO(dfj2.to_json())
        result = read_json(json, dtype={"ints": np.int64, "bools": np.bool_})
        tm.assert_frame_equal(result, result)

    def test_round_trip_exception(self, datapath):
        # [FUNCTION DOC]
        # GH 3867. A frame read from a stored CSV file is written to JSON and read back;
        # after reindexing to the original labels and filling missing values the frame
        # equals the original.
        # GH 3867
        path = datapath("io", "json", "data", "teams.csv")
        df = pd.read_csv(path)
        s = df.to_json()

        result = read_json(StringIO(s))
        res = result.reindex(index=df.index, columns=df.columns)
        res = res.fillna(np.nan)
        tm.assert_frame_equal(res, df)

    @pytest.mark.network
    @pytest.mark.single_cpu
    def test_url(self, httpserver):
        # [FUNCTION DOC]
        # Network, single-CPU test. Serves JSON text from a local test server and reads it
        # from the URL with date conversion; created and updated fields become UTC
        # timezone-aware microsecond datetimes and the closed field naive microsecond.
        data = '{"created_at": ["2023-06-23T18:21:36Z"], "closed_at": ["2023-06-23T18:21:36"], "updated_at": ["2023-06-23T18:21:36Z"]}\n'  # noqa: E501
        httpserver.serve_content(content=data)
        result = read_json(httpserver.url, convert_dates=True)
        for field, dtype in [
            ["created_at", pd.DatetimeTZDtype("us", tz="UTC")],
            ["closed_at", "datetime64[us]"],
            ["updated_at", pd.DatetimeTZDtype("us", tz="UTC")],
        ]:
            assert result[field].dtype == dtype

    def test_timedelta(self, unit):
        # [FUNCTION DOC]
        # Given the unit fixture. Timedelta series and frame at given units are written with
        # the default format (deprecation warning expected), read back and converted with
        # a millisecond converter; result equals the original cast to milliseconds.
        converter = lambda x: pd.to_timedelta(x, unit="ms")

        ser = Series([timedelta(23), timedelta(seconds=5)], dtype=f"m8[{unit}]")
        assert ser.dtype == f"timedelta64[{unit}]"

        msg = (
            "The default 'epoch' date format is deprecated and will be removed "
            "in a future version, please use 'iso' date format instead."
        )
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            result = read_json(StringIO(ser.to_json()), typ="series").apply(converter)
        tm.assert_series_equal(result, ser.astype("m8[ms]"))

        ser = Series(
            [timedelta(23), timedelta(seconds=5)], index=Index([0, 1]), dtype="m8[ns]"
        )
        assert ser.dtype == "timedelta64[ns]"
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            result = read_json(StringIO(ser.to_json()), typ="series").apply(converter)
        tm.assert_series_equal(result, ser.astype("m8[ms]"))

        frame = DataFrame([timedelta(23), timedelta(seconds=5)], dtype="m8[ns]")
        assert frame[0].dtype == "timedelta64[ns]"

        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            json = frame.to_json()
        result = read_json(StringIO(json)).apply(converter)
        tm.assert_frame_equal(frame.astype("m8[ms]"), result)

    def test_timedelta2(self):
        # [FUNCTION DOC]
        # A frame with timedelta, integer and datetime columns written with nanosecond date
        # unit (deprecation warning expected) and read back; after converting the
        # timedelta and date columns it equals the original.
        frame = DataFrame(
            {
                "a": [timedelta(days=23), timedelta(seconds=5)],
                "b": [1, 2],
                "c": date_range(start="20130101", periods=2, unit="ns"),
            }
        )
        frame["a"] = frame["a"].astype("m8[ns]")
        msg = (
            "The default 'epoch' date format is deprecated and will be removed "
            "in a future version, please use 'iso' date format instead."
        )
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            data = StringIO(frame.to_json(date_unit="ns"))
        result = read_json(data)
        result["a"] = pd.to_timedelta(result.a, unit="ns")
        result["c"] = pd.to_datetime(result.c)
        tm.assert_frame_equal(frame, result)

    def test_mixed_timedelta_datetime(self):
        # [FUNCTION DOC]
        # An object column mixing a timedelta and a timestamp, written with nanosecond unit
        # and read with an integer dtype, gives the integer nanosecond values of both.
        td = timedelta(23)
        ts = Timestamp("20130101")
        frame = DataFrame({"a": [td, ts]}, dtype=object)

        expected = DataFrame(
            {"a": [pd.Timedelta(td).as_unit("ns")._value, ts.as_unit("ns")._value]}
        )
        data = StringIO(frame.to_json(date_unit="ns"))
        result = read_json(data, dtype={"a": "int64"})
        tm.assert_frame_equal(result, expected, check_index_type=False)

    @pytest.mark.parametrize("as_object", [True, False])
    @pytest.mark.parametrize("date_format", ["iso", "epoch"])
    @pytest.mark.parametrize("timedelta_typ", [pd.Timedelta, timedelta])
    def test_timedelta_to_json(self, as_object, date_format, timedelta_typ):
        # [FUNCTION DOC]
        # GH28156. Parametrized on object versus timedelta dtype, date format and
        # timedelta class. Timedeltas as both values and index are serialized as ISO
        # duration strings (iso) or millisecond integers (epoch, with deprecation
        # warning); missing values become null; an extra string element is appended for
        # object data.
        # GH28156: to_json not correctly formatting Timedelta
        data = [timedelta_typ(days=1), timedelta_typ(days=2), pd.NaT]
        if as_object:
            data.append("a")

        ser = Series(data, index=data)
        if not as_object:
            ser = ser.astype("m8[ns]")
            ser.index = ser.index.astype("m8[ns]")
        expected_warning = None
        if date_format == "iso":
            expected = (
                '{"P1DT0H0M0S":"P1DT0H0M0S","P2DT0H0M0S":"P2DT0H0M0S","null":null}'
            )
        else:
            expected_warning = Pandas4Warning
            expected = '{"86400000":86400000,"172800000":172800000,"null":null}'

        if as_object:
            expected = expected.replace("}", ',"a":"a"}')

        msg = (
            "'epoch' date format is deprecated and will be removed in a future "
            "version, please use 'iso' date format instead."
        )
        with tm.assert_produces_warning(expected_warning, match=msg):
            result = ser.to_json(date_format=date_format)
        assert result == expected

    @pytest.mark.parametrize("as_object", [True, False])
    @pytest.mark.parametrize("timedelta_typ", [pd.Timedelta, timedelta])
    def test_timedelta_to_json_fractional_precision(self, as_object, timedelta_typ):
        # [FUNCTION DOC]
        # Parametrized on object versus timedelta dtype and timedelta class. A 42
        # millisecond timedelta as value and index is written as the integer 42; the
        # deprecation warning is expected only for timedelta dtype.
        data = [timedelta_typ(milliseconds=42)]
        ser = Series(data, index=data).astype("m8[ns]")
        ser.index = ser.index.astype("m8[ns]")
        warn = Pandas4Warning
        if as_object:
            ser = ser.astype(object)
            warn = None

        msg = (
            "The default 'epoch' date format is deprecated and will be removed "
            "in a future version, please use 'iso' date format instead."
        )
        with tm.assert_produces_warning(warn, match=msg):
            result = ser.to_json()
        expected = '{"42":42}'
        assert result == expected

    def test_default_handler(self):
        # [FUNCTION DOC]
        # An object with no JSON form in a column is converted to its string form when a
        # string conversion default handler is given; round trips to that string.
        value = object()
        frame = DataFrame({"a": [7, value]})
        expected = DataFrame({"a": [7, str(value)]})
        result = read_json(StringIO(frame.to_json(default_handler=str)))
        tm.assert_frame_equal(expected, result, check_index_type=False)

    def test_default_handler_indirect(self):
        # [FUNCTION DOC]
        # Using the low-level dump function with values orientation and a custom
        # default handler that encodes complex numbers as a list of pairs and other
        # objects as strings, a list holding a number and a frame with mixed content is
        # encoded to the exact expected text (the forward slash escaped).
        def default(obj):
            # [FUNCTION DOC]
            # HELPER: default handler used by the enclosing test. Parameter obj: any
            # unsupported object. Returns, for a complex number, a list of three key-value
            # pairs (a marker pair, real part, imaginary part); for anything else returns its
            # string form.
            if isinstance(obj, complex):
                return [("mathjs", "Complex"), ("re", obj.real), ("im", obj.imag)]
            return str(obj)

        df_list = [
            9,
            DataFrame(
                {"a": [1, "STR", complex(4, -5)], "b": [float("nan"), None, "N/A"]},
                columns=["a", "b"],
            ),
        ]
        expected = (
            '[9,[[1,null],["STR",null],[[["mathjs","Complex"],'
            '["re",4.0],["im",-5.0]],"N\\/A"]]]'
        )
        assert (
            ujson_dumps(df_list, default_handler=default, orient="values") == expected
        )

    def test_default_handler_numpy_unsupported_dtype(self):
        # [FUNCTION DOC]
        # GH12554. Complex-valued columns (an unsupported numpy dtype) are written with a
        # string default handler in values orientation, giving each element as its string
        # form.
        # GH12554 to_json raises 'Unhandled numpy dtype 15'
        df = DataFrame(
            {"a": [1, 2.3, complex(4, -5)], "b": [float("nan"), None, complex(1.2, 0)]},
            columns=["a", "b"],
        )
        expected = (
            '[["(1+0j)","(nan+0j)"],["(2.3+0j)","(nan+0j)"],["(4-5j)","(1.2+0j)"]]'
        )
        assert df.to_json(default_handler=str, orient="values") == expected

    def test_default_handler_raises(self):
        # [FUNCTION DOC]
        # An exception raised by a user-supplied default handler propagates, both for an
        # arbitrary object and for a complex number.
        msg = "raisin"

        def my_handler_raises(obj):
            # [FUNCTION DOC]
            # HELPER: default handler used by the enclosing test. Parameter obj: ignored.
            # Always raises a type error with the message defined in the enclosing test.
            raise TypeError(msg)

        with pytest.raises(TypeError, match=msg):
            DataFrame({"a": [1, 2, object()]}).to_json(
                default_handler=my_handler_raises
            )
        with pytest.raises(TypeError, match=msg):
            DataFrame({"a": [1, 2, complex(4, -5)]}).to_json(
                default_handler=my_handler_raises
            )

    def test_categorical(self):
        # [FUNCTION DOC]
        # GH4377. Writing a frame with a categorical column gives the same output as with
        # the plain column, and likewise for a series (previously segfaulted with
        # non-ndarray blocks).
        # GH4377 df.to_json segfaults with non-ndarray blocks
        df = DataFrame({"A": ["a", "b", "c", "a", "b", "b", "a"]})
        df["B"] = df["A"]
        expected = df.to_json()

        df["B"] = df["A"].astype("category")
        assert expected == df.to_json()

        s = df["A"]
        sc = df["B"]
        assert s.to_json() == sc.to_json()

    def test_datetime_tz(self):
        # [FUNCTION DOC]
        # GH4377. A frame and series with timezone-aware datetimes serialize the same as
        # the UTC-converted naive values, with the deprecation warning for the default
        # epoch format.
        # GH4377 df.to_json segfaults with non-ndarray blocks
        tz_range = date_range("20130101", periods=3, tz="US/Eastern", unit="ns")
        tz_naive = tz_range.tz_convert("utc").tz_localize(None)

        df = DataFrame(
            {"A": tz_range, "B": date_range("20130101", periods=3, unit="ns")}
        )

        df_naive = df.copy()
        df_naive["A"] = tz_naive
        msg = (
            "The default 'epoch' date format is deprecated and will be removed "
            "in a future version, please use 'iso' date format instead."
        )
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            expected = df_naive.to_json()
            assert expected == df.to_json()

        stz = Series(tz_range)
        s_naive = Series(tz_naive)
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            assert stz.to_json() == s_naive.to_json()

    def test_sparse(self):
        # [FUNCTION DOC]
        # GH4377. A mostly-missing sparse frame and series serialize to the same text as
        # their dense versions.
        # GH4377 df.to_json segfaults with non-ndarray blocks
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        df.loc[:8] = np.nan

        sdf = df.astype("Sparse")
        expected = df.to_json()
        assert expected == sdf.to_json()

        s = Series(np.random.default_rng(2).standard_normal(10))
        s.loc[:8] = np.nan
        ss = s.astype("Sparse")

        expected = s.to_json()
        assert expected == ss.to_json()

    @pytest.mark.parametrize(
        "ts",
        [
            Timestamp("2013-01-10 05:00:00Z"),
            Timestamp("2013-01-10 00:00:00", tz="US/Eastern"),
            Timestamp("2013-01-10 00:00:00-0500"),
        ],
    )
    def test_tz_is_utc(self, ts):
        # [FUNCTION DOC]
        # Parametrized on three timestamps representing the same instant (UTC, Eastern,
        # fixed offset). Low-level encoding with ISO dates gives the UTC string with a
        # trailing Z, for both the timestamp and the converted standard datetime.
        exp = '"2013-01-10T05:00:00.000Z"'

        assert ujson_dumps(ts, iso_dates=True) == exp
        dt = ts.to_pydatetime()
        assert ujson_dumps(dt, iso_dates=True) == exp

    def test_tz_is_naive(self):
        # [FUNCTION DOC]
        # A naive timestamp and standard datetime encode as an ISO string with no zone
        # suffix.
        ts = Timestamp("2013-01-10 05:00:00")
        exp = '"2013-01-10T05:00:00.000"'

        assert ujson_dumps(ts, iso_dates=True) == exp
        dt = ts.to_pydatetime()
        assert ujson_dumps(dt, iso_dates=True) == exp

    @pytest.mark.parametrize(
        "tz_range",
        [
            date_range("2013-01-01 05:00:00Z", periods=2, unit="ns"),
            date_range("2013-01-01 00:00:00", periods=2, tz="US/Eastern", unit="ns"),
            date_range("2013-01-01 00:00:00-0500", periods=2, unit="ns"),
        ],
    )
    def test_tz_range_is_utc(self, tz_range):
        # [FUNCTION DOC]
        # Parametrized on three equivalent timezone-aware date ranges. Low-level ISO
        # encoding of the range, of a datetime index, of the same as object array, and of
        # a frame holding it all gives UTC strings with Z. The last statement only checks
        # that encoding a frame with an object column returns something non-empty.
        # KNOWN ISSUE: the final assertion is only a truthiness check on the encoded text, not a comparison with the expected output.
        exp = '["2013-01-01T05:00:00.000Z","2013-01-02T05:00:00.000Z"]'
        dfexp = '{"DT":{"0":"2013-01-01T05:00:00.000Z","1":"2013-01-02T05:00:00.000Z"}}'

        assert ujson_dumps(tz_range, iso_dates=True) == exp
        dti = DatetimeIndex(tz_range)
        # Ensure datetimes in object array are serialized correctly
        # in addition to the normal DTI case
        assert ujson_dumps(dti, iso_dates=True) == exp
        assert ujson_dumps(dti.astype(object), iso_dates=True) == exp
        df = DataFrame({"DT": dti})
        result = ujson_dumps(df, iso_dates=True)
        assert result == dfexp
        assert ujson_dumps(df.astype({"DT": object}), iso_dates=True)

    def test_tz_range_is_naive(self):
        # [FUNCTION DOC]
        # A naive date range encodes as ISO strings without zone suffix, as an index,
        # object array, and in a frame.
        dti = date_range("2013-01-01 05:00:00", periods=2, unit="ns")

        exp = '["2013-01-01T05:00:00.000","2013-01-02T05:00:00.000"]'
        dfexp = '{"DT":{"0":"2013-01-01T05:00:00.000","1":"2013-01-02T05:00:00.000"}}'

        # Ensure datetimes in object array are serialized correctly
        # in addition to the normal DTI case
        assert ujson_dumps(dti, iso_dates=True) == exp
        assert ujson_dumps(dti.astype(object), iso_dates=True) == exp
        df = DataFrame({"DT": dti})
        result = ujson_dumps(df, iso_dates=True)
        assert result == dfexp
        assert ujson_dumps(df.astype({"DT": object}), iso_dates=True)

    def test_read_inline_jsonl(self):
        # GH9180

        # [FUNCTION DOC]
        # GH9180. Reading JSON Lines text from a buffer in lines mode yields a two-row
        # frame regardless of key order and spacing.
        result = read_json(StringIO('{"a": 1, "b": 2}\n{"b":2, "a" :1}\n'), lines=True)
        expected = DataFrame([[1, 2], [1, 2]], columns=["a", "b"])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.single_cpu
    @pytest.mark.network
    @td.skip_if_not_us_locale
    def test_read_s3_jsonl(self, s3_bucket_public_with_data, s3so):
        # [FUNCTION DOC]
        # GH17200. Network, single-CPU, US-locale test. Reads a JSON Lines object from a
        # mock S3 bucket using a non-standard URL scheme and storage options.
        # GH17200
        result = read_json(
            f"s3n://{s3_bucket_public_with_data.name}/items.jsonl",
            lines=True,
            storage_options=s3so,
        )
        expected = DataFrame([[1, 2], [1, 2]], columns=["a", "b"])
        tm.assert_frame_equal(result, expected)

    def test_read_local_jsonl(self, temp_file):
        # [FUNCTION DOC]
        # GH17200. Reading JSON Lines from a local file path in lines mode.
        # GH17200
        with open(temp_file, "w", encoding="utf-8") as infile:
            infile.write('{"a": 1, "b": 2}\n{"b":2, "a" :1}\n')
        result = read_json(temp_file, lines=True)
        expected = DataFrame([[1, 2], [1, 2]], columns=["a", "b"])
        tm.assert_frame_equal(result, expected)

    def test_read_jsonl_unicode_chars(self):
        # GH15132: non-ascii unicode characters
        # \u201d == RIGHT DOUBLE QUOTATION MARK

        # [FUNCTION DOC]
        # GH15132. JSON Lines text with a non-ASCII quotation mark is read correctly from
        # a buffer (tested twice, the second time with identical construction).
        # simulate file handle
        json = '{"a": "foo”", "b": "bar"}\n{"a": "foo", "b": "bar"}\n'
        json = StringIO(json)
        result = read_json(json, lines=True)
        expected = DataFrame([["foo\u201d", "bar"], ["foo", "bar"]], columns=["a", "b"])
        tm.assert_frame_equal(result, expected)

        # simulate string
        json = StringIO('{"a": "foo”", "b": "bar"}\n{"a": "foo", "b": "bar"}\n')
        result = read_json(json, lines=True)
        expected = DataFrame([["foo\u201d", "bar"], ["foo", "bar"]], columns=["a", "b"])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("bigNum", [sys.maxsize + 1, -(sys.maxsize + 2)])
    def test_to_json_large_numbers(self, bigNum):
        # [FUNCTION DOC]
        # GH34473. Parametrized on integers just beyond the 64-bit signed range. An
        # object-dtype series and frame holding them serialize with exact digits.
        # GH34473
        series = Series(bigNum, dtype=object, index=["articleId"])
        json = series.to_json()
        expected = '{"articleId":' + str(bigNum) + "}"
        assert json == expected

        df = DataFrame(bigNum, dtype=object, index=["articleId"], columns=[0])
        json = df.to_json()
        expected = '{"0":{"articleId":' + str(bigNum) + "}}"
        assert json == expected

    @pytest.mark.parametrize("bigNum", [-(2**63) - 1, 2**64])
    def test_read_json_large_numbers(self, bigNum):
        # [FUNCTION DOC]
        # GH20599, 26068. Parametrized on integers outside 64-bit range. Reading raises a
        # value error saying the value is too small or too big, both for series-like and
        # frame-like documents.
        # GH20599, 26068
        json = StringIO('{"articleId":' + str(bigNum) + "}")
        msg = r"Value is too small|Value is too big"
        with pytest.raises(ValueError, match=msg):
            read_json(json)

        json = StringIO('{"0":{"articleId":' + str(bigNum) + "}}")
        with pytest.raises(ValueError, match=msg):
            read_json(json)

    def test_read_json_large_numbers2(self):
        # [FUNCTION DOC]
        # GH18842. A numeric string too large for 64-bit integers is read as a float for
        # series and frame.
        # GH18842
        json = '{"articleId": "1404366058080022500245"}'
        json = StringIO(json)
        result = read_json(json, typ="series")
        expected = Series(1.404366e21, index=["articleId"])
        tm.assert_series_equal(result, expected)

        json = '{"0": {"articleId": "1404366058080022500245"}}'
        json = StringIO(json)
        result = read_json(json)
        expected = DataFrame(1.404366e21, index=["articleId"], columns=[0])
        tm.assert_frame_equal(result, expected)

    def test_to_jsonl(self):
        # [FUNCTION DOC]
        # GH9180 and GH15096. Records orientation with lines mode writes one object per
        # line with trailing newline; braces, quotes and backslashes in data and labels
        # are escaped correctly and read back equal.
        # GH9180
        df = DataFrame([[1, 2], [1, 2]], columns=["a", "b"])
        result = df.to_json(orient="records", lines=True)
        expected = '{"a":1,"b":2}\n{"a":1,"b":2}\n'
        assert result == expected

        df = DataFrame([["foo}", "bar"], ['foo"', "bar"]], columns=["a", "b"])
        result = df.to_json(orient="records", lines=True)
        expected = '{"a":"foo}","b":"bar"}\n{"a":"foo\\"","b":"bar"}\n'
        assert result == expected
        tm.assert_frame_equal(read_json(StringIO(result), lines=True), df)

        # GH15096: escaped characters in columns and data
        df = DataFrame([["foo\\", "bar"], ['foo"', "bar"]], columns=["a\\", "b"])
        result = df.to_json(orient="records", lines=True)
        expected = '{"a\\\\":"foo\\\\","b":"bar"}\n{"a\\\\":"foo\\"","b":"bar"}\n'
        assert result == expected

        tm.assert_frame_equal(read_json(StringIO(result), lines=True), df)

    # TODO: there is a near-identical test for pytables; can we share?
    @pytest.mark.xfail(reason="GH#13774 encoding kwarg not supported", raises=TypeError)
    @pytest.mark.parametrize(
        "val",
        [
            [b"E\xc9, 17", b"", b"a", b"b", b"c"],
            [b"E\xc9, 17", b"a", b"b", b"c"],
            [b"EE, 17", b"", b"a", b"b", b"c"],
            [b"E\xc9, 17", b"\xf8\xfc", b"a", b"b", b"c"],
            [b"", b"a", b"b", b"c"],
            [b"\xf8\xfc", b"a", b"b", b"c"],
            [b"A\xf8\xfc", b"", b"a", b"b", b"c"],
            [np.nan, b"", b"b", b"c"],
            [b"A\xf8\xfc", np.nan, b"", b"b", b"c"],
        ],
    )
    @pytest.mark.parametrize("dtype", ["category", object])
    def test_latin_encoding(self, dtype, val, temp_file):
        # [FUNCTION DOC]
        # GH 13774. Expected to fail with a type error (encoding keyword not supported).
        # Parametrized on lists of byte strings with Latin-1 characters and on category or
        # object dtype. Writes a series with Latin-1 encoding and attempts to read back.
        # A TODO mentions a similar test for the HDF store.
        # KNOWN ISSUE: the read-back wraps a file path in a text buffer constructor instead of opening the file, so the test would be wrong even if the encoding keyword were supported.
        # GH 13774
        ser = Series(
            [x.decode("latin-1") if isinstance(x, bytes) else x for x in val],
            dtype=dtype,
        )
        encoding = "latin-1"
        ser.to_json(temp_file, encoding=encoding)
        retr = read_json(StringIO(temp_file), encoding=encoding)
        tm.assert_series_equal(ser, retr, check_categorical=False)

    def test_data_frame_size_after_to_json(self):
        # [FUNCTION DOC]
        # GH15344. Serializing a frame does not change its deep memory usage.
        # GH15344
        df = DataFrame({"a": [str(1)]})

        size_before = df.memory_usage(index=True, deep=True).sum()
        df.to_json()
        size_after = df.memory_usage(index=True, deep=True).sum()

        assert size_before == size_after

    @pytest.mark.parametrize(
        "index", [None, [1, 2], [1.0, 2.0], ["a", "b"], ["1", "2"], ["1.", "2."]]
    )
    @pytest.mark.parametrize("columns", [["a", "b"], ["1", "2"], ["1.", "2."]])
    def test_from_json_to_json_table_index_and_columns(self, index, columns):
        # [FUNCTION DOC]
        # GH25433, GH25435. Parametrized on six index choices and three column label choices
        # (including numeric-looking strings and floats). A frame round-trips through the
        # table orientation preserving label types.
        # GH25433 GH25435
        expected = DataFrame([[1, 2], [3, 4]], index=index, columns=columns)
        dfjson = expected.to_json(orient="table")

        result = read_json(StringIO(dfjson), orient="table")
        tm.assert_frame_equal(result, expected)

    def test_from_json_to_json_table_dtypes(self):
        # [FUNCTION DOC]
        # GH21345. Integer, float and string-of-digits columns keep their dtypes in a
        # table orientation round trip.
        # GH21345
        expected = DataFrame({"a": [1, 2], "b": [3.0, 4.0], "c": ["5", "6"]})
        dfjson = expected.to_json(orient="table")
        result = read_json(StringIO(dfjson), orient="table")
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("orient", ["split", "records", "index", "columns"])
    def test_to_json_from_json_columns_dtypes(self, orient, using_infer_string):
        # [FUNCTION DOC]
        # GH21892, GH33205. Parametrized on split, records, index and columns orients.
        # A frame of integer, float with missing, object with missing, bool,
        # categorical with missing and datetime with missing is written (deprecation
        # warning for epoch) and read back with explicit dtypes for each column; the
        # result equals the original (with the missing object entry becoming NaN when
        # string inference is on).
        # GH21892 GH33205
        expected = DataFrame.from_dict(
            {
                "Integer": Series([1, 2, 3], dtype="int64"),
                "Float": Series([None, 2.0, 3.0], dtype="float64"),
                "Object": Series([None, "", "c"], dtype="object"),
                "Bool": Series([True, False, True], dtype="bool"),
                "Category": Series(["a", "b", None], dtype="category"),
                "Datetime": Series(
                    ["2020-01-01", None, "2020-01-03"], dtype="datetime64[ms]"
                ),
            }
        )
        msg = (
            "The default 'epoch' date format is deprecated and will be removed "
            "in a future version, please use 'iso' date format instead."
        )
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            dfjson = expected.to_json(orient=orient)

        if using_infer_string:
            # When this is read back in it is inferred to "str" dtype which
            #  uses NaN instead of None.
            expected.loc[0, "Object"] = np.nan

        result = read_json(
            StringIO(dfjson),
            orient=orient,
            dtype={
                "Integer": "int64",
                "Float": "float64",
                "Object": "object",
                "Bool": "bool",
                "Category": "category",
                "Datetime": "datetime64[ns]",
            },
        )
        tm.assert_frame_equal(result, expected)

    def test_to_json_with_index_as_a_column_name(self):
        # [FUNCTION DOC]
        # A frame with a column named index written with table orientation raises a value
        # error about overlapping names.
        df = DataFrame(data={"index": [1, 2], "a": [2, 3]})
        with pytest.raises(
            ValueError, match="Overlapping names between the index and columns"
        ):
            df.to_json(orient="table")

    @pytest.mark.parametrize("dtype", [True, {"b": int, "c": int}])
    def test_read_json_table_dtype_raises(self, dtype):
        # [FUNCTION DOC]
        # GH21345. Parametrized on dtype true or a mapping. Passing a dtype along with
        # table orientation raises a value error saying both cannot be passed.
        # GH21345
        df = DataFrame({"a": [1, 2], "b": [3.0, 4.0], "c": ["5", "6"]})
        dfjson = df.to_json(orient="table")
        msg = "cannot pass both dtype and orient='table'"
        with pytest.raises(ValueError, match=msg):
            read_json(dfjson, orient="table", dtype=dtype)

    @pytest.mark.parametrize("orient", ["index", "columns", "records", "values"])
    def test_read_json_table_empty_axes_dtype(self, orient):
        # GH28558

        # [FUNCTION DOC]
        # GH28558. Parametrized on four orients. Reading an empty object with axes
        # conversion gives empty default index and columns.
        expected = DataFrame()
        result = read_json(StringIO("{}"), orient=orient, convert_axes=True)
        tm.assert_index_equal(result.index, expected.index)
        tm.assert_index_equal(result.columns, expected.columns)

    def test_read_json_table_convert_axes_raises(self):
        # [FUNCTION DOC]
        # GH25433, GH25435. Passing convert_axes along with table orientation raises a
        # value error saying both cannot be passed.
        # GH25433 GH25435
        df = DataFrame([[1, 2], [3, 4]], index=[1.0, 2.0], columns=["1.", "2."])
        dfjson = df.to_json(orient="table")
        msg = "cannot pass both convert_axes and orient='table'"
        with pytest.raises(ValueError, match=msg):
            read_json(dfjson, orient="table", convert_axes=True)

    @pytest.mark.parametrize(
        "data, expected",
        [
            (
                DataFrame([[1, 2], [4, 5]], columns=["a", "b"]),
                {"columns": ["a", "b"], "data": [[1, 2], [4, 5]]},
            ),
            (
                DataFrame([[1, 2], [4, 5]], columns=["a", "b"]).rename_axis("foo"),
                {"columns": ["a", "b"], "data": [[1, 2], [4, 5]]},
            ),
            (
                DataFrame(
                    [[1, 2], [4, 5]], columns=["a", "b"], index=[["a", "b"], ["c", "d"]]
                ),
                {"columns": ["a", "b"], "data": [[1, 2], [4, 5]]},
            ),
            (Series([1, 2, 3], name="A"), {"name": "A", "data": [1, 2, 3]}),
            (
                Series([1, 2, 3], name="A").rename_axis("foo"),
                {"name": "A", "data": [1, 2, 3]},
            ),
            (
                Series([1, 2], name="A", index=[["a", "b"], ["c", "d"]]),
                {"name": "A", "data": [1, 2]},
            ),
        ],
    )
    def test_index_false_to_json_split(self, data, expected):
        # GH 17394
        # Testing index=False in to_json with orient='split'

        # [FUNCTION DOC]
        # GH 17394. Parametrized on frames and series (plain, named index, multi-index).
        # With split orientation and index off, the output has only columns (or name) and
        # data, no index.
        result = data.to_json(orient="split", index=False)
        result = json.loads(result)

        assert result == expected

    @pytest.mark.parametrize(
        "data",
        [
            (DataFrame([[1, 2], [4, 5]], columns=["a", "b"])),
            (DataFrame([[1, 2], [4, 5]], columns=["a", "b"]).rename_axis("foo")),
            (
                DataFrame(
                    [[1, 2], [4, 5]], columns=["a", "b"], index=[["a", "b"], ["c", "d"]]
                )
            ),
            (Series([1, 2, 3], name="A")),
            (Series([1, 2, 3], name="A").rename_axis("foo")),
            (Series([1, 2], name="A", index=[["a", "b"], ["c", "d"]])),
        ],
    )
    def test_index_false_to_json_table(self, data):
        # GH 17394
        # Testing index=False in to_json with orient='table'

        # [FUNCTION DOC]
        # GH 17394. Parametrized on the same objects. With table orientation and index
        # off, the parsed output equals a schema built without index plus records.
        result = data.to_json(orient="table", index=False)
        result = json.loads(result)

        expected = {
            "schema": pd.io.json.build_table_schema(data, index=False),
            "data": DataFrame(data).to_dict(orient="records"),
        }

        assert result == expected

    @pytest.mark.parametrize("orient", ["index", "columns"])
    def test_index_false_error_to_json(self, orient):
        # GH 17394, 25513
        # Testing error message from to_json with index=False

        # [FUNCTION DOC]
        # GH 17394, 25513. Parametrized on index and columns orients. Index false raises a
        # value error saying it is valid only for split, table, records or values.
        df = DataFrame([[1, 2], [4, 5]], columns=["a", "b"])

        msg = (
            "'index=False' is only valid when 'orient' is 'split', "
            "'table', 'records', or 'values'"
        )
        with pytest.raises(ValueError, match=msg):
            df.to_json(orient=orient, index=False)

    @pytest.mark.parametrize("orient", ["records", "values"])
    def test_index_true_error_to_json(self, orient):
        # GH 25513
        # Testing error message from to_json with index=True

        # [FUNCTION DOC]
        # GH 25513. Parametrized on records and values. Index true raises a value error
        # saying it is valid only for split, table, index or columns.
        df = DataFrame([[1, 2], [4, 5]], columns=["a", "b"])

        msg = (
            "'index=True' is only valid when 'orient' is 'split', "
            "'table', 'index', or 'columns'"
        )
        with pytest.raises(ValueError, match=msg):
            df.to_json(orient=orient, index=True)

    @pytest.mark.parametrize("orient", ["split", "table"])
    @pytest.mark.parametrize("index", [True, False])
    def test_index_false_from_json_to_json(self, orient, index):
        # [FUNCTION DOC]
        # GH25170. Parametrized on split/table orients and the index flag. Writing with the
        # flag and reading back without index equals the original frame.
        # GH25170
        # Test index=False in from_json to_json
        expected = DataFrame({"a": [1, 2], "b": [3, 4]})
        dfjson = expected.to_json(orient=orient, index=index)
        result = read_json(StringIO(dfjson), orient=orient)
        tm.assert_frame_equal(result, expected)

    def test_read_timezone_information(self):
        # [FUNCTION DOC]
        # GH 25546. A series keyed by an ISO timestamp with Z read with index orientation
        # gets a UTC timezone-aware microsecond index.
        # GH 25546
        result = read_json(
            StringIO('{"2019-01-01T11:00:00.000Z":88}'), typ="series", orient="index"
        )
        exp_dti = DatetimeIndex(["2019-01-01 11:00:00"], dtype="M8[us, UTC]")
        expected = Series([88], index=exp_dti)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "url",
        [
            "s3://example-fsspec/",
            "gcs://another-fsspec/file.json",
            "filecache::s3://yet-another-fsspec/file.json",
            "https://example-site.com/data",
            "some-protocol://data.txt",
        ],
    )
    def test_read_json_with_url_value(self, url):
        # [FUNCTION DOC]
        # GH 36271. Parametrized on URL-like strings (several protocols). A value that
        # looks like a URL inside a JSON document is treated as data, not fetched.
        # GH 36271
        result = read_json(StringIO(f'{{"url":{{"0":"{url}"}}}}'))
        expected = DataFrame({"url": [url]})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "compression",
        ["", ".gz", ".bz2", ".tar"],
    )
    def test_read_json_with_very_long_file_path(self, compression):
        # [FUNCTION DOC]
        # GH 46718. Parametrized on compression suffixes. A path of a thousand characters
        # raises a file-not-found error with the path in the message.
        # GH 46718
        long_json_path = f"{'a' * 1000}.json{compression}"
        with pytest.raises(
            FileNotFoundError, match=f"File {long_json_path} does not exist"
        ):
            # path too long for Windows is handled in file_exists() but raises in
            # _get_data_from_filepath()
            read_json(long_json_path)

    @pytest.mark.parametrize(
        "date_format,key", [("epoch", 86400000), ("iso", "1970-01-02T00:00:00.000")]
    )
    def test_datetime_as_label(self, date_format, key, unit):
        # [FUNCTION DOC]
        # Given the unit fixture. Parametrized on epoch versus ISO. A datetime used as a
        # column label is written as a quoted epoch number (with deprecation warning) or
        # ISO string key.
        df = DataFrame(
            [[1]], columns=[(Timestamp(0) + pd.Timedelta("1D")).as_unit(unit)]
        )
        expected = f'{{"{key}":{{"0":1}}}}'

        expected_warning = None
        if date_format == "epoch":
            expected_warning = Pandas4Warning

        msg = (
            "'epoch' date format is deprecated and will be removed in a future "
            "version, please use 'iso' date format instead."
        )
        with tm.assert_produces_warning(expected_warning, match=msg):
            result = df.to_json(date_format=date_format)

        assert result == expected

    @pytest.mark.parametrize(
        "date_format,key", [("epoch", 86400000), ("iso", "P1DT0H0M0S")]
    )
    def test_timedelta_as_label(self, date_format, key, unit):
        # [FUNCTION DOC]
        # Given the unit fixture. A timedelta column label is written as epoch
        # milliseconds (with deprecation warning) or an ISO duration key.
        df = DataFrame([[1]], columns=[pd.Timedelta("1D").as_unit(unit)])
        expected = f'{{"{key}":{{"0":1}}}}'

        expected_warning = None
        if date_format == "epoch":
            expected_warning = Pandas4Warning

        msg = (
            "'epoch' date format is deprecated and will be removed in a future "
            "version, please use 'iso' date format instead."
        )
        with tm.assert_produces_warning(expected_warning, match=msg):
            result = df.to_json(date_format=date_format)

        assert result == expected

    @pytest.mark.parametrize(
        "orient,expected",
        [
            ("index", "{\"('a', 'b')\":{\"('c', 'd')\":1}}"),
            ("columns", "{\"('c', 'd')\":{\"('a', 'b')\":1}}"),
            # TODO: the below have separate encoding procedures
            pytest.param(
                "split",
                "",
                marks=pytest.mark.xfail(
                    reason="Produces JSON but not in a consistent manner"
                ),
            ),
            pytest.param(
                "table",
                "",
                marks=pytest.mark.xfail(
                    reason="Produces JSON but not in a consistent manner"
                ),
            ),
        ],
    )
    def test_tuple_labels(self, orient, expected):
        # [FUNCTION DOC]
        # GH 20500. Parametrized on orients. Tuple labels are written as the tuple's text
        # for index and columns orients; split and table are expected to fail (output is
        # inconsistent), with empty expected text. A TODO mentions separate encoding
        # procedures.
        # GH 20500
        df = DataFrame([[1]], index=[("a", "b")], columns=[("c", "d")])
        result = df.to_json(orient=orient)
        assert result == expected

    @pytest.mark.parametrize("indent", [1, 2, 4])
    def test_to_json_indent(self, indent):
        # [FUNCTION DOC]
        # GH 12004. Parametrized on indentation of 1, 2, 4. Output is pretty-printed with
        # that many spaces per level.
        # GH 12004
        df = DataFrame([["foo", "bar"], ["baz", "qux"]], columns=["a", "b"])

        result = df.to_json(indent=indent)
        spaces = " " * indent
        expected = f"""{{
{spaces}"a":{{
{spaces}{spaces}"0":"foo",
{spaces}{spaces}"1":"baz"
{spaces}}},
{spaces}"b":{{
{spaces}{spaces}"0":"bar",
{spaces}{spaces}"1":"qux"
{spaces}}}
}}"""

        assert result == expected

    @pytest.mark.skipif(
        using_string_dtype(),
        reason="Adjust expected when infer_string is default, no bug here, "
        "just a complicated parametrization",
    )
    @pytest.mark.parametrize(
        "orient,expected",
        [
            (
                "split",
                """{
    "columns":[
        "a",
        "b"
    ],
    "index":[
        0,
        1
    ],
    "data":[
        [
            "foo",
            "bar"
        ],
        [
            "baz",
            "qux"
        ]
    ]
}""",
            ),
            (
                "records",
                """[
    {
        "a":"foo",
        "b":"bar"
    },
    {
        "a":"baz",
        "b":"qux"
    }
]""",
            ),
            (
                "index",
                """{
    "0":{
        "a":"foo",
        "b":"bar"
    },
    "1":{
        "a":"baz",
        "b":"qux"
    }
}""",
            ),
            (
                "columns",
                """{
    "a":{
        "0":"foo",
        "1":"baz"
    },
    "b":{
        "0":"bar",
        "1":"qux"
    }
}""",
            ),
            (
                "values",
                """[
    [
        "foo",
        "bar"
    ],
    [
        "baz",
        "qux"
    ]
]""",
            ),
            (
                "table",
                """{
    "schema":{
        "fields":[
            {
                "name":"index",
                "type":"integer"
            },
            {
                "name":"a",
                "type":"string"
            },
            {
                "name":"b",
                "type":"string"
            }
        ],
        "primaryKey":[
            "index"
        ],
        "pandas_version":"1.4.0"
    },
    "data":[
        {
            "index":0,
            "a":"foo",
            "b":"bar"
        },
        {
            "index":1,
            "a":"baz",
            "b":"qux"
        }
    ]
}""",
            ),
        ],
    )
    def test_json_indent_all_orients(self, orient, expected):
        # [FUNCTION DOC]
        # GH 12004. Skipped when string dtype is the default. Parametrized on all six
        # orients. Writing a small string frame with indentation of four spaces gives the
        # expected pretty-printed text; table orientation includes a schema with the
        # fixed version string.
        # GH 12004
        df = DataFrame([["foo", "bar"], ["baz", "qux"]], columns=["a", "b"])
        result = df.to_json(orient=orient, indent=4)
        assert result == expected

    def test_json_negative_indent_raises(self):
        # [FUNCTION DOC]
        # A negative indent raises a value error saying it must be a nonnegative integer.
        with pytest.raises(ValueError, match="must be a nonnegative integer"):
            DataFrame().to_json(indent=-1)

    def test_emca_262_nan_inf_support(self):
        # [FUNCTION DOC]
        # GH 12213. Reading a list including unquoted NaN, Infinity and negative Infinity
        # tokens gives missing and infinite numbers while quoted forms stay strings.
        # GH 12213
        data = StringIO(
            '["a", NaN, "NaN", Infinity, "Infinity", -Infinity, "-Infinity"]'
        )
        result = read_json(data)
        expected = DataFrame(
            ["a", None, "NaN", np.inf, "Infinity", -np.inf, "-Infinity"]
        )
        tm.assert_frame_equal(result, expected)

    def test_frame_int_overflow(self):
        # [FUNCTION DOC]
        # GH 30320. A numeric string too large for integers stays a string.
        # GH 30320
        encoded_json = json.dumps([{"col": "31900441201190696999"}, {"col": "Text"}])
        expected = DataFrame({"col": ["31900441201190696999", "Text"]})
        result = read_json(StringIO(encoded_json))
        tm.assert_frame_equal(result, expected)

    def test_json_multiindex(self):
        # [FUNCTION DOC]
        # A series with a two-level index (from stacking) written with index orientation
        # uses the text of each label tuple as the key.
        dataframe = DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        expected = (
            '{"(0, \'x\')":1,"(0, \'y\')":"a","(1, \'x\')":2,'
            '"(1, \'y\')":"b","(2, \'x\')":3,"(2, \'y\')":"c"}'
        )
        series = dataframe.stack()
        result = series.to_json(orient="index")
        assert result == expected

    @pytest.mark.single_cpu
    @pytest.mark.network
    def test_to_s3(self, s3_bucket_public, s3so):
        # [FUNCTION DOC]
        # GH 28375. Network, single-CPU test. Writing a frame to a mock S3 URI with storage
        # options and reading it back gives an equal frame.
        # GH 28375
        mock_bucket_name = s3_bucket_public.name
        target_file = f"{uuid.uuid4()}.json"
        df = DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
        uri = f"s3://{mock_bucket_name}/{target_file}"
        df.to_json(uri, storage_options=s3so)
        result = read_json(uri, storage_options=s3so)
        tm.assert_frame_equal(result, df)

    def test_json_pandas_nulls(self, nulls_fixture):
        # [FUNCTION DOC]
        # GH 31615. Parametrized on the null fixture. Any null-like value serializes as
        # null; the not-a-time value additionally emits the epoch deprecation warning.
        # GH 31615
        expected_warning = None
        msg = (
            "The default 'epoch' date format is deprecated and will be removed "
            "in a future version, please use 'iso' date format instead."
        )
        if nulls_fixture is pd.NaT:
            expected_warning = Pandas4Warning

        with tm.assert_produces_warning(expected_warning, match=msg):
            result = DataFrame([[nulls_fixture]]).to_json()
        assert result == '{"0":{"0":null}}'

    def test_readjson_bool_series(self):
        # [FUNCTION DOC]
        # GH31464. A JSON list of booleans read as a series gives a boolean series.
        # GH31464
        result = read_json(StringIO("[true, true, false]"), typ="series")
        expected = Series([True, True, False])
        tm.assert_series_equal(result, expected)

    def test_to_json_multiindex_escape(self):
        # [FUNCTION DOC]
        # GH 15273. A stacked series with a timestamp-and-string two-level index is written
        # with each key being the tuple's text, including timestamp reprs.
        # GH 15273
        df = DataFrame(
            True,
            index=date_range("2017-01-20", "2017-01-23", unit="ns"),
            columns=["foo", "bar"],
        ).stack()
        result = df.to_json()
        expected = (
            "{\"(Timestamp('2017-01-20 00:00:00'), 'foo')\":true,"
            "\"(Timestamp('2017-01-20 00:00:00'), 'bar')\":true,"
            "\"(Timestamp('2017-01-21 00:00:00'), 'foo')\":true,"
            "\"(Timestamp('2017-01-21 00:00:00'), 'bar')\":true,"
            "\"(Timestamp('2017-01-22 00:00:00'), 'foo')\":true,"
            "\"(Timestamp('2017-01-22 00:00:00'), 'bar')\":true,"
            "\"(Timestamp('2017-01-23 00:00:00'), 'foo')\":true,"
            "\"(Timestamp('2017-01-23 00:00:00'), 'bar')\":true}"
        )
        assert result == expected

    def test_to_json_series_of_objects(self):
        # [FUNCTION DOC]
        # Series of arbitrary objects: keys are all non-callable, non-underscore
        # attributes (GH-42768).
        class _TestObject:
            # [CLASS DOC]
            # HELPER: plain test object with public attributes a, b, d, a private attribute
            # and a method; used to show which attributes are serialized.
            def __init__(self, a, b, _c, d) -> None:
                # [FUNCTION DOC]
                # HELPER: stores the four given values as attributes with the same names (the third
                # one with a leading underscore). Returns nothing.
                self.a = a
                self.b = b
                self._c = _c
                self.d = d

            def e(self):
                # [FUNCTION DOC]
                # HELPER: a method that returns the constant five; present to show callable
                # attributes are not serialized.
                return 5

        # JSON keys should be all non-callable non-underscore attributes, see GH-42768
        series = Series([_TestObject(a=1, b=2, _c=3, d=4)])
        assert json.loads(series.to_json()) == {"0": {"a": 1, "b": 2, "d": 4}}

    @pytest.mark.parametrize(
        "data,expected",
        [
            (
                Series({0: -6 + 8j, 1: 0 + 1j, 2: 9 - 5j}),
                '{"0":{"imag":8.0,"real":-6.0},'
                '"1":{"imag":1.0,"real":0.0},'
                '"2":{"imag":-5.0,"real":9.0}}',
            ),
            (
                Series({0: -9.39 + 0.66j, 1: 3.95 + 9.32j, 2: 4.03 - 0.17j}),
                '{"0":{"imag":0.66,"real":-9.39},'
                '"1":{"imag":9.32,"real":3.95},'
                '"2":{"imag":-0.17,"real":4.03}}',
            ),
            (
                DataFrame([[-2 + 3j, -1 - 0j], [4 - 3j, -0 - 10j]]),
                '{"0":{"0":{"imag":3.0,"real":-2.0},'
                '"1":{"imag":-3.0,"real":4.0}},'
                '"1":{"0":{"imag":0.0,"real":-1.0},'
                '"1":{"imag":-10.0,"real":0.0}}}',
            ),
            (
                DataFrame(
                    [[-0.28 + 0.34j, -1.08 - 0.39j], [0.41 - 0.34j, -0.78 - 1.35j]]
                ),
                '{"0":{"0":{"imag":0.34,"real":-0.28},'
                '"1":{"imag":-0.34,"real":0.41}},'
                '"1":{"0":{"imag":-0.39,"real":-1.08},'
                '"1":{"imag":-1.35,"real":-0.78}}}',
            ),
        ],
    )
    def test_complex_data_tojson(self, data, expected):
        # [FUNCTION DOC]
        # GH41174. Parametrized on complex series and frames. Complex numbers are written
        # as objects with imaginary and real parts.
        # GH41174
        result = data.to_json()
        assert result == expected

    def test_json_uint64(self):
        # [FUNCTION DOC]
        # GH21073. Values above the signed 64-bit range are written exactly with split
        # orientation.
        # GH21073
        expected = (
            '{"columns":["col1"],"index":[0,1],'
            '"data":[[13342205958987758245],[12388075603347835679]]}'
        )
        df = DataFrame(data={"col1": [13342205958987758245, 12388075603347835679]})
        result = df.to_json(orient="split")
        assert result == expected

    def test_read_json_dtype_backend(
        self, string_storage, dtype_backend, orient, using_infer_string
    ):
        # [FUNCTION DOC]
        # GH#50750. Parametrized (fixtures) on string storage, dtype backend and orient. A
        # frame with nullable integers, floats, booleans and strings, with missing values,
        # is written and read with the dtype backend: nullable types for numpy_nullable,
        # Arrow-backed arrays for pyarrow. For values orientation column labels become
        # integers. Column index type check is relaxed.
        # GH#50750
        df = DataFrame(
            {
                "a": Series([1, NA, 3], dtype="Int64"),
                "b": Series([1, 2, 3], dtype="Int64"),
                "c": Series([1.5, NA, 2.5], dtype="Float64"),
                "d": Series([1.5, 2.0, 2.5], dtype="Float64"),
                "e": [True, False, None],
                "f": [True, False, True],
                "g": ["a", "b", "c"],
                "h": ["a", "b", None],
            }
        )

        out = df.to_json(orient=orient)
        with pd.option_context("mode.string_storage", string_storage):
            result = read_json(
                StringIO(out), dtype_backend=dtype_backend, orient=orient
            )

        if dtype_backend == "pyarrow":
            pa = pytest.importorskip("pyarrow")
            string_dtype = pd.ArrowDtype(pa.string())
        else:
            string_dtype = pd.StringDtype(string_storage)

        expected = DataFrame(
            {
                "a": Series([1, NA, 3], dtype="Int64"),
                "b": Series([1, 2, 3], dtype="Int64"),
                "c": Series([1.5, NA, 2.5], dtype="Float64"),
                "d": Series([1.5, 2.0, 2.5], dtype="Float64"),
                "e": Series([True, False, NA], dtype="boolean"),
                "f": Series([True, False, True], dtype="boolean"),
                "g": Series(["a", "b", "c"], dtype=string_dtype),
                "h": Series(["a", "b", None], dtype=string_dtype),
            }
        )

        if dtype_backend == "pyarrow":
            pa = pytest.importorskip("pyarrow")
            from pandas.arrays import ArrowExtensionArray

            expected = DataFrame(
                {
                    col: ArrowExtensionArray(pa.array(expected[col], from_pandas=True))
                    for col in expected.columns
                }
            )

        if orient == "values":
            expected.columns = list(range(8))

        # the storage of the str columns' Index is also affected by the
        # string_storage setting -> ignore that for checking the result
        tm.assert_frame_equal(result, expected, check_column_type=False)

    @td.skip_if_no("pyarrow")
    @pytest.mark.filterwarnings("ignore:Passing a BlockManager:DeprecationWarning")
    def test_read_json_pyarrow_with_dtype(self):
        # [FUNCTION DOC]
        # Requires pyarrow. Reading lines from bytes with the Arrow engine and Arrow
        # dtype backend with an explicit dtype mapping gives columns of exactly those
        # Arrow integer types.
        dtype = {"a": "int32[pyarrow]", "b": "int64[pyarrow]"}
        json = b'{"a": 1, "b": 2}\n'

        df = read_json(
            BytesIO(json),
            dtype=dtype,
            lines=True,
            engine="pyarrow",
            dtype_backend="pyarrow",
        )

        result = df.dtypes
        expected = Series(
            data=[
                pd.ArrowDtype.construct_from_string("int32[pyarrow]"),
                pd.ArrowDtype.construct_from_string("int64[pyarrow]"),
            ],
            index=["a", "b"],
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("orient", ["split", "records", "index"])
    def test_read_json_nullable_series(self, string_storage, dtype_backend, orient):
        # [FUNCTION DOC]
        # GH#50750. Parametrized on split, records, index orients. A nullable integer
        # series round-trips with the given dtype backend and string storage option; with
        # pyarrow backend the result is Arrow-backed.
        # GH#50750
        pa = pytest.importorskip("pyarrow")
        ser = Series([1, NA, 3], dtype="Int64")

        out = ser.to_json(orient=orient)
        with pd.option_context("mode.string_storage", string_storage):
            result = read_json(
                StringIO(out), dtype_backend=dtype_backend, orient=orient, typ="series"
            )

        expected = Series([1, NA, 3], dtype="Int64")

        if dtype_backend == "pyarrow":
            from pandas.arrays import ArrowExtensionArray

            expected = Series(ArrowExtensionArray(pa.array(expected, from_pandas=True)))

        tm.assert_series_equal(result, expected)

    def test_invalid_dtype_backend(self):
        # [FUNCTION DOC]
        # An unknown dtype backend name raises a value error listing the allowed values.
        msg = (
            "dtype_backend numpy is invalid, only 'numpy_nullable' and "
            "'pyarrow' are allowed."
        )
        with pytest.raises(ValueError, match=msg):
            read_json("test", dtype_backend="numpy")


def test_invalid_engine():
    # [FUNCTION DOC]
    # GH 48893. An unknown engine name raises a value error mentioning the engine type.
    # GH 48893
    ser = Series(range(1))
    out = ser.to_json()
    with pytest.raises(ValueError, match="The engine type foo"):
        read_json(out, engine="foo")


def test_pyarrow_engine_lines_false():
    # [FUNCTION DOC]
    # GH 48893. The Arrow engine without lines mode raises a value error saying only
    # lines is currently supported.
    # GH 48893
    ser = Series(range(1))
    out = ser.to_json()
    with pytest.raises(ValueError, match="currently pyarrow engine only supports"):
        read_json(out, engine="pyarrow", lines=False)


def test_json_roundtrip_string_inference(orient):
    # [FUNCTION DOC]
    # Parametrized by the orient fixture. Reading a string-only frame under the
    # future string inference option produces string dtype for data, index and
    # columns, matching an explicitly typed frame.
    # KNOWN ISSUE: the orient parameter is accepted but the writer is always called with the default orientation, so every parametrization runs the same test.
    df = DataFrame(
        [["a", "b"], ["c", "d"]], index=["row 1", "row 2"], columns=["col 1", "col 2"]
    )
    out = df.to_json()
    with pd.option_context("future.infer_string", True):
        result = read_json(StringIO(out))
    dtype = pd.StringDtype(na_value=np.nan)
    expected = DataFrame(
        [["a", "b"], ["c", "d"]],
        dtype=dtype,
        index=Index(["row 1", "row 2"], dtype=dtype),
        columns=Index(["col 1", "col 2"], dtype=dtype),
    )
    tm.assert_frame_equal(result, expected)


@td.skip_if_no("pyarrow")
def test_to_json_ea_null():
    # [FUNCTION DOC]
    # GH#57224. Requires pyarrow. Arrow-backed and nullable integer columns with a
    # missing value are written as lines; missing becomes null.
    # GH#57224
    df = DataFrame(
        {
            "a": Series([1, NA], dtype="int64[pyarrow]"),
            "b": Series([2, NA], dtype="Int64"),
        }
    )
    result = df.to_json(orient="records", lines=True)
    expected = """{"a":1,"b":2}
{"a":null,"b":null}
"""
    assert result == expected


def test_read_json_lines_rangeindex():
    # [FUNCTION DOC]
    # GH 57429. Reading lines mode produces a default range index (exact type
    # checked).
    # GH 57429
    data = """
{"a": 1, "b": 2}
{"a": 3, "b": 4}
"""
    result = read_json(StringIO(data), lines=True).index
    expected = RangeIndex(2)
    tm.assert_index_equal(result, expected, exact=True)


def test_large_number():
    # [FUNCTION DOC]
    # GH#20608. A list with a 16-digit number string read as a series with values
    # orientation and no date conversion gives a 64-bit integer series.
    # GH#20608
    result = read_json(
        StringIO('["9999999999999999"]'),
        orient="values",
        typ="series",
        convert_dates=False,
    )
    expected = Series([9999999999999999])
    tm.assert_series_equal(result, expected)
