# [MODULE DOC]
# Tests for the frame attribute that reports each column's dtype (dtypes), covering
# empty frames, tz-aware datetimes, slices, timedeltas, derived columns and results
# after other operations. Uses the testing helpers module and frame fixtures. No
# import-time side effects.

from datetime import timedelta

import numpy as np
import pytest

from pandas.core.dtypes.dtypes import DatetimeTZDtype

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    date_range,
)
import pandas._testing as tm


class TestDataFrameDataTypes:
    # [CLASS DOC]
    # Stateless pytest class of dtypes tests.
    def test_empty_frame_dtypes(self):
        # [FUNCTION DOC]
        # Frames without rows or columns give empty object series (with column labels
        # where they exist); empty typed columns keep int32; a typed frame and its empty
        # slice report the same per-column dtypes.
        empty_df = DataFrame()
        tm.assert_series_equal(empty_df.dtypes, Series(dtype=object))

        nocols_df = DataFrame(index=[1, 2, 3])
        tm.assert_series_equal(nocols_df.dtypes, Series(dtype=object))

        norows_df = DataFrame(columns=list("abc"))
        tm.assert_series_equal(norows_df.dtypes, Series(object, index=list("abc")))

        norows_int_df = DataFrame(columns=list("abc")).astype(np.int32)
        tm.assert_series_equal(
            norows_int_df.dtypes, Series(np.dtype("int32"), index=list("abc"))
        )

        df = DataFrame({"a": 1, "b": True, "c": 1.0}, index=[1, 2, 3])
        ex_dtypes = Series({"a": np.int64, "b": np.bool_, "c": np.float64})
        tm.assert_series_equal(df.dtypes, ex_dtypes)

        # same but for empty slice of df
        tm.assert_series_equal(df[:0].dtypes, ex_dtypes)

    def test_datetime_with_tz_dtypes(self):
        # [FUNCTION DOC]
        # A frame of naive and two tz-aware datetime columns (with NaT set) reports the
        # datetime dtype with the right units and zones, sorted by label.
        tzframe = DataFrame(
            {
                "A": date_range("20130101", periods=3, unit="ns"),
                "B": date_range("20130101", periods=3, tz="US/Eastern", unit="ns"),
                "C": date_range("20130101", periods=3, tz="CET", unit="ns"),
            }
        )
        tzframe.iloc[1, 1] = pd.NaT
        tzframe.iloc[1, 2] = pd.NaT
        result = tzframe.dtypes.sort_index()
        expected = Series(
            [
                np.dtype("datetime64[ns]"),
                DatetimeTZDtype("ns", "US/Eastern"),
                DatetimeTZDtype("ns", "CET"),
            ],
            ["A", "B", "C"],
        )

        tm.assert_series_equal(result, expected)

    def test_dtypes_are_correct_after_column_slice(self):
        # [FUNCTION DOC]
        # Column slicing does not change the original dtypes and the slice reports only
        # its columns' dtypes.
        # GH6525
        df = DataFrame(index=range(5), columns=list("abc"), dtype=np.float64)
        tm.assert_series_equal(
            df.dtypes,
            Series({"a": np.float64, "b": np.float64, "c": np.float64}),
        )
        tm.assert_series_equal(df.iloc[:, 2:].dtypes, Series({"c": np.float64}))
        tm.assert_series_equal(
            df.dtypes,
            Series({"a": np.float64, "b": np.float64, "c": np.float64}),
        )

    @pytest.mark.parametrize(
        "data",
        [pd.NA, True],
    )
    def test_dtypes_are_correct_after_groupby_last(self, data):
        # [FUNCTION DOC]
        # Parametrized over missing marker and true. A nullable boolean column taken
        # after grouping and last keeps the boolean extension dtype and equals the
        # indexed column.
        # GH46409
        df = DataFrame(
            {"id": [1, 2, 3, 4], "test": [True, pd.NA, data, False]}
        ).convert_dtypes()
        result = df.groupby("id").last().test
        expected = df.set_index("id").test
        assert result.dtype == pd.BooleanDtype()
        tm.assert_series_equal(expected, result)

    def test_dtypes_gh8722(self, float_string_frame):
        # [FUNCTION DOC]
        # After adding a boolean column to the mixed frame, dtypes equals the per-column
        # dtypes taken from the items.
        float_string_frame["bool"] = float_string_frame["A"] > 0
        result = float_string_frame.dtypes
        expected = Series(
            {k: v.dtype for k, v in float_string_frame.items()}, index=result.index
        )
        tm.assert_series_equal(result, expected)

    def test_dtypes_timedeltas(self):
        # [FUNCTION DOC]
        # Datetime and timedelta columns report their numpy dtypes; the sum column is
        # datetime; a scalar integer column is int64.
        df = DataFrame(
            {
                "A": Series(date_range("2012-1-1", periods=3, freq="D", unit="ns")),
                "B": Series([timedelta(days=i) for i in range(3)]),
            }
        )
        result = df.dtypes
        expected = Series(
            [np.dtype("datetime64[ns]"), np.dtype("timedelta64[us]")], index=list("AB")
        )
        tm.assert_series_equal(result, expected)

        df["C"] = df["A"] + df["B"]
        result = df.dtypes
        expected = Series(
            [
                np.dtype("datetime64[ns]"),
                np.dtype("timedelta64[us]"),
                np.dtype("datetime64[ns]"),
            ],
            index=list("ABC"),
        )
        tm.assert_series_equal(result, expected)

        # mixed int types
        df["D"] = 1
        result = df.dtypes
        expected = Series(
            [
                np.dtype("datetime64[ns]"),
                np.dtype("timedelta64[us]"),
                np.dtype("datetime64[ns]"),
                np.dtype("int64"),
            ],
            index=list("ABCD"),
        )
        tm.assert_series_equal(result, expected)

    def test_frame_apply_np_array_return_type(self, using_infer_string):
        # [FUNCTION DOC]
        # Does not test dtypes: applying a function that returns a zero-dimensional
        # array per column gives a series equal to that array. The string inference
        # fixture is requested but not used.
        # KNOWN ISSUE: the test belongs to the apply behavior rather than dtypes, and
        # its fixture argument using_infer_string is unused.
        # GH 35517
        df = DataFrame([["foo"]])
        result = df.apply(lambda col: np.array("bar"))
        expected = Series(np.array("bar"))
        tm.assert_series_equal(result, expected)
