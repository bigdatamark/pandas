# [MODULE DOC]
# Tests for the frame method that converts columns to the best possible nullable dtypes
# (convert_dtypes), including options for each conversion kind and the numpy-nullable
# and pyarrow backends. Specific per-type behavior is covered by the series tests; here
# the frame wrapper behavior is verified (block layout, column names, empty frames,
# arrow round trips, complex numbers). Uses string storage and NaN-semantics fixtures,
# optional dependency decorators and the testing helpers module. No import-time side
# effects.

import datetime

import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
import pandas._testing as tm


class TestConvertDtypes:
    # [CLASS DOC]
    # Stateless pytest class of convert_dtypes tests on frames.
    @pytest.mark.parametrize(
        "convert_integer, expected", [(False, np.dtype("int32")), (True, "Int32")]
    )
    def test_convert_dtypes(self, convert_integer, expected, string_storage):
        # [FUNCTION DOC]
        # Parametrized over (convert_integer flag, expected integer dtype) and the
        # string storage fixture. Under the string storage option, a frame with an int32
        # and an object string column is converted with positional arguments (infer
        # objects true, strings true, integers as given, floats false). Integer column
        # is converted to nullable Int32 only if requested, and the string column to the
        # string dtype of the chosen storage.
        # Specific types are tested in tests/series/test_dtypes.py
        # Just check that it works for DataFrame here
        df = pd.DataFrame(
            {
                "a": pd.Series([1, 2, 3], dtype=np.dtype("int32")),
                "b": pd.Series(["x", "y", "z"], dtype=np.dtype("O")),
            }
        )
        with pd.option_context("string_storage", string_storage):
            result = df.convert_dtypes(True, True, convert_integer, False)
        expected = pd.DataFrame(
            {
                "a": pd.Series([1, 2, 3], dtype=expected),
                "b": pd.Series(["x", "y", "z"], dtype=f"string[{string_storage}]"),
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_convert_empty(self):
        # [FUNCTION DOC]
        # An empty frame passes through conversion and equals the input.
        # Empty DataFrame can pass convert_dtypes, see GH#40393
        empty_df = pd.DataFrame()
        tm.assert_frame_equal(empty_df, empty_df.convert_dtypes())

    @td.skip_if_no("pyarrow")
    def test_convert_empty_categorical_to_pyarrow(self):
        # [FUNCTION DOC]
        # Requires pyarrow. Categorical columns with only missing values are unchanged
        # when converting with the arrow backend.
        # GH#59934
        df = pd.DataFrame(
            {
                "A": pd.Categorical([None] * 5),
                "B": pd.Categorical([None] * 5, categories=["B1", "B2"]),
            }
        )
        converted = df.convert_dtypes(dtype_backend="pyarrow")
        expected = df
        tm.assert_frame_equal(converted, expected)

    def test_convert_dtypes_retain_column_names(self):
        # [FUNCTION DOC]
        # The name of the column index is retained after conversion.
        # GH#41435
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df.columns.name = "cols"

        result = df.convert_dtypes()
        tm.assert_index_equal(result.columns, df.columns)
        assert result.columns.name == "cols"

    def test_pyarrow_dtype_backend(self, using_nan_is_na):
        # [FUNCTION DOC]
        # Requires pyarrow; uses nan-is-na fixture. Converting a frame of ints, strings
        # with None, objects with booleans, floats with NaN, datetimes (ns and tz-aware
        # s) and timedeltas with the arrow backend gives arrow-backed arrays with
        # matching arrow types (float NaN becomes a missing value only when the
        # NaN-is-NA mode is on; timedelta becomes microsecond duration).
        pa = pytest.importorskip("pyarrow")
        df = pd.DataFrame(
            {
                "a": pd.Series([1, 2, 3], dtype=np.dtype("int32")),
                "b": pd.Series(["x", "y", None], dtype=np.dtype("O")),
                "c": pd.Series([True, False, None], dtype=np.dtype("O")),
                "d": pd.Series([np.nan, 100.5, 200], dtype=np.dtype("float")),
                "e": pd.Series(pd.date_range("2022", periods=3, unit="ns")),
                "f": pd.Series(pd.date_range("2022", periods=3, tz="UTC").as_unit("s")),
                "g": pd.Series(pd.timedelta_range("1D", periods=3)),
            }
        )
        result = df.convert_dtypes(dtype_backend="pyarrow")

        item = None if using_nan_is_na else np.nan
        expected = pd.DataFrame(
            {
                "a": pd.arrays.ArrowExtensionArray(
                    pa.array([1, 2, 3], type=pa.int32())
                ),
                "b": pd.arrays.ArrowExtensionArray(pa.array(["x", "y", None])),
                "c": pd.arrays.ArrowExtensionArray(pa.array([True, False, None])),
                "d": pd.arrays.ArrowExtensionArray(pa.array([item, 100.5, 200.0])),
                "e": pd.arrays.ArrowExtensionArray(
                    pa.array(
                        [
                            datetime.datetime(2022, 1, 1),
                            datetime.datetime(2022, 1, 2),
                            datetime.datetime(2022, 1, 3),
                        ],
                        type=pa.timestamp(unit="ns"),
                    )
                ),
                "f": pd.arrays.ArrowExtensionArray(
                    pa.array(
                        [
                            datetime.datetime(2022, 1, 1),
                            datetime.datetime(2022, 1, 2),
                            datetime.datetime(2022, 1, 3),
                        ],
                        type=pa.timestamp(unit="s", tz="UTC"),
                    )
                ),
                "g": pd.arrays.ArrowExtensionArray(
                    pa.array(
                        [
                            datetime.timedelta(1),
                            datetime.timedelta(2),
                            datetime.timedelta(3),
                        ],
                        type=pa.duration("us"),
                    )
                ),
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_pyarrow_dtype_backend_already_pyarrow(self):
        # [FUNCTION DOC]
        # Requires pyarrow. A frame already holding arrow ints is unchanged by
        # arrow-backend conversion.
        pytest.importorskip("pyarrow")
        expected = pd.DataFrame([1, 2, 3], dtype="int64[pyarrow]")
        result = expected.convert_dtypes(dtype_backend="pyarrow")
        tm.assert_frame_equal(result, expected)

    def test_pyarrow_dtype_backend_from_pandas_nullable(self):
        # [FUNCTION DOC]
        # Requires pyarrow. Converting columns of nullable integer, python string,
        # boolean and Float64 types to the arrow backend yields arrow-backed arrays with
        # the same values and missing entries.
        pa = pytest.importorskip("pyarrow")
        df = pd.DataFrame(
            {
                "a": pd.Series([1, 2, None], dtype="Int32"),
                "b": pd.Series(["x", "y", None], dtype="string[python]"),
                "c": pd.Series([True, False, None], dtype="boolean"),
                "d": pd.Series([None, 100.5, 200], dtype="Float64"),
            }
        )
        result = df.convert_dtypes(dtype_backend="pyarrow")
        expected = pd.DataFrame(
            {
                "a": pd.arrays.ArrowExtensionArray(
                    pa.array([1, 2, None], type=pa.int32())
                ),
                "b": pd.arrays.ArrowExtensionArray(pa.array(["x", "y", None])),
                "c": pd.arrays.ArrowExtensionArray(pa.array([True, False, None])),
                "d": pd.arrays.ArrowExtensionArray(pa.array([None, 100.5, 200.0])),
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_pyarrow_dtype_empty_object(self):
        # [FUNCTION DOC]
        # Requires pyarrow. An empty frame with an object column converted with the
        # arrow backend is equal to itself.
        # GH 50970
        pytest.importorskip("pyarrow")
        expected = pd.DataFrame(columns=[0])
        result = expected.convert_dtypes(dtype_backend="pyarrow")
        tm.assert_frame_equal(result, expected)

    def test_pyarrow_engine_lines_false(self):
        # [FUNCTION DOC]
        # An unsupported backend name raises a value error stating only the
        # numpy-nullable and pyarrow backends are allowed.
        # KNOWN ISSUE: the test name refers to an engine lines option from the JSON
        # reader and does not describe what is tested (an invalid dtype backend value).
        # GH 48893
        df = pd.DataFrame({"a": [1, 2, 3]})
        msg = (
            "dtype_backend numpy is invalid, only 'numpy_nullable' and "
            "'pyarrow' are allowed."
        )
        with pytest.raises(ValueError, match=msg):
            df.convert_dtypes(dtype_backend="numpy")

    def test_pyarrow_backend_no_conversion(self):
        # [FUNCTION DOC]
        # Requires pyarrow. With all four conversion flags disabled and the arrow
        # backend selected, the frame is returned unchanged.
        # GH#52872
        pytest.importorskip("pyarrow")
        df = pd.DataFrame({"a": [1, 2], "b": 1.5, "c": True, "d": "x"})
        expected = df.copy()
        result = df.convert_dtypes(
            convert_floating=False,
            convert_integer=False,
            convert_boolean=False,
            convert_string=False,
            dtype_backend="pyarrow",
        )
        tm.assert_frame_equal(result, expected)

    def test_convert_dtypes_pyarrow_to_np_nullable(self):
        # [FUNCTION DOC]
        # Requires pyarrow. Converting an arrow int32 frame with the numpy-nullable
        # backend gives Int32.
        # GH 53648
        pytest.importorskip("pyarrow")
        ser = pd.DataFrame(range(2), dtype="int32[pyarrow]")
        result = ser.convert_dtypes(dtype_backend="numpy_nullable")
        expected = pd.DataFrame(range(2), dtype="Int32")
        tm.assert_frame_equal(result, expected)

    def test_convert_dtypes_pyarrow_timestamp(self):
        # [FUNCTION DOC]
        # Requires pyarrow. Converting an arrow timestamp series with the arrow backend
        # leaves it unchanged.
        # KNOWN ISSUE: this test lives in the frame tests but builds and converts a
        # series, and the variable of the original data is named ser while the
        # conversion is done on the expected object.
        # GH 54191
        pytest.importorskip("pyarrow")
        ser = pd.Series(pd.date_range("2020-01-01", "2020-01-02", freq="1min"))
        expected = ser.astype("timestamp[ms][pyarrow]")
        result = expected.convert_dtypes(dtype_backend="pyarrow")
        tm.assert_series_equal(result, expected)

    def test_convert_dtypes_avoid_block_splitting(self):
        # [FUNCTION DOC]
        # Converting with integer conversion disabled changes only the string column and
        # leaves the manager with two blocks (the two integer columns stay together).
        # GH#55341
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": "a"})
        result = df.convert_dtypes(convert_integer=False)
        expected = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
                "c": pd.Series(["a"] * 3, dtype="string"),
            }
        )
        tm.assert_frame_equal(result, expected)
        assert result._mgr.nblocks == 2

    def test_convert_dtypes_from_arrow(self):
        # [FUNCTION DOC]
        # Object column holding strings and a time object: the string column becomes the
        # string dtype and the time column is unchanged.
        # GH#56581
        df = pd.DataFrame([["a", datetime.time(18, 12)]], columns=["a", "b"])
        result = df.convert_dtypes()
        expected = df.astype({"a": "string"})
        tm.assert_frame_equal(result, expected)

    def test_convert_dtype_pyarrow_timezone_preserve(self):
        # [FUNCTION DOC]
        # Requires pyarrow. An arrow timestamp column with a UTC zone keeps its type
        # under arrow-backend conversion.
        # GH 60237
        pytest.importorskip("pyarrow")
        df = pd.DataFrame(
            {
                "timestamps": pd.Series(
                    pd.to_datetime(range(5), utc=True, unit="h"),
                    dtype="timestamp[ns, tz=UTC][pyarrow]",
                )
            }
        )
        result = df.convert_dtypes(dtype_backend="pyarrow")
        expected = df.copy()
        tm.assert_frame_equal(result, expected)

    def test_convert_dtypes_complex(self):
        # [FUNCTION DOC]
        # Complex floats are kept as complex128 and integers become nullable Int64.
        # GH 60129
        df = pd.DataFrame({"a": [1.0 + 5.0j, 1.5 - 3.0j], "b": [1, 2]})
        expected = pd.DataFrame(
            {
                "a": pd.array([1.0 + 5.0j, 1.5 - 3.0j], dtype="complex128"),
                "b": pd.array([1, 2], dtype="Int64"),
            }
        )
        result = df.convert_dtypes()
        tm.assert_frame_equal(result, expected)

    def test_convert_dtypes_mixed_column_after_slice(self):
        # [FUNCTION DOC]
        # After slicing rows out of a frame with mixed-type columns and copying, columns
        # that now hold a single kind are converted to nullable integer and string.
        # GH#64702
        df = pd.DataFrame(data=[[1, "a"], [2, "b"], ["c", 3]], columns=["col1", "col2"])
        df = df.loc[[0, 1]].copy()
        result = df.convert_dtypes()
        expected = pd.DataFrame(
            {
                "col1": pd.array([1, 2], dtype="Int64"),
                "col2": pd.array(["a", "b"], dtype="string"),
            }
        )
        tm.assert_frame_equal(result, expected)
