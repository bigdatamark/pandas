# [MODULE DOC]
# Tests for DataFrame.update (in-place alignment-based update from another frame, mapping or series):
# overwrite and filter options, error handling, dtype preservation and view/copy behavior.

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    date_range,
)
import pandas._testing as tm


class TestDataFrameUpdate:
    # [CLASS DOC]
    # Test container for DataFrame.update. All tests mutate a locally created frame in place and compare it with an expected frame.
    def test_update_nan(self):
        # [FUNCTION DOC]
        # Regression test (GH 15593, GH 15617). With overwrite turned off, missing values in the other frame do not
        # replace existing values (first case: frame unchanged) and fill missing values in the target (second case) while a datetime
        # column is preserved.
        # #15593 #15617
        # test 1
        df1 = DataFrame({"A": [1.0, 2, 3], "B": date_range("2000", periods=3)})
        df2 = DataFrame({"A": [None, 2, 3]})
        expected = df1.copy()
        df1.update(df2, overwrite=False)

        tm.assert_frame_equal(df1, expected)

        # test 2
        df1 = DataFrame({"A": [1.0, None, 3], "B": date_range("2000", periods=3)})
        df2 = DataFrame({"A": [None, 2, 3]})
        expected = DataFrame({"A": [1.0, 2, 3], "B": date_range("2000", periods=3)})
        df1.update(df2, overwrite=False)

        tm.assert_frame_equal(df1, expected)

    def test_update(self):
        # [FUNCTION DOC]
        # Updates a 4x3 frame from a smaller frame with two rows (labels 1 and 3) that contains missing values; only non-missing
        # values overwrite, aligned by label.
        df = DataFrame(
            [[1.5, np.nan, 3.0], [1.5, np.nan, 3.0], [1.5, np.nan, 3], [1.5, np.nan, 3]]
        )

        other = DataFrame([[3.6, 2.0, np.nan], [np.nan, np.nan, 7]], index=[1, 3])

        df.update(other)

        expected = DataFrame(
            [[1.5, np.nan, 3], [3.6, 2, 3], [1.5, np.nan, 3], [1.5, np.nan, 7.0]]
        )
        tm.assert_frame_equal(df, expected)

    def test_update_dtypes(self):
        # [FUNCTION DOC]
        # Regression test (GH 3016). Updating float columns with integers in a frame that also has boolean columns
        # keeps boolean columns and yields floats for the updated columns.
        # gh 3016
        df = DataFrame(
            [[1.0, 2.0, False, True], [4.0, 5.0, True, False]],
            columns=["A", "B", "bool1", "bool2"],
        )

        other = DataFrame([[45, 45]], index=[0], columns=["A", "B"])
        df.update(other)

        expected = DataFrame(
            [[45.0, 45.0, False, True], [4.0, 5.0, True, False]],
            columns=["A", "B", "bool1", "bool2"],
        )
        tm.assert_frame_equal(df, expected)

    def test_update_nooverwrite(self):
        # [FUNCTION DOC]
        # Same data as the basic update test but with overwrite off: only missing target cells are filled from the other frame.
        df = DataFrame(
            [[1.5, np.nan, 3.0], [1.5, np.nan, 3.0], [1.5, np.nan, 3], [1.5, np.nan, 3]]
        )

        other = DataFrame([[3.6, 2.0, np.nan], [np.nan, np.nan, 7]], index=[1, 3])

        df.update(other, overwrite=False)

        expected = DataFrame(
            [[1.5, np.nan, 3], [1.5, 2, 3], [1.5, np.nan, 3], [1.5, np.nan, 3.0]]
        )
        tm.assert_frame_equal(df, expected)

    def test_update_filtered(self):
        # [FUNCTION DOC]
        # Update with a filter function that selects target values greater than two: only target cells passing the filter are
        # overwritten by non-missing other values.
        df = DataFrame(
            [[1.5, np.nan, 3.0], [1.5, np.nan, 3.0], [1.5, np.nan, 3], [1.5, np.nan, 3]]
        )

        other = DataFrame([[3.6, 2.0, np.nan], [np.nan, np.nan, 7]], index=[1, 3])

        df.update(other, filter_func=lambda x: x > 2)

        expected = DataFrame(
            [[1.5, np.nan, 3], [1.5, np.nan, 3], [1.5, np.nan, 3], [1.5, np.nan, 7.0]]
        )
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "bad_kwarg, exception, msg",
        [
            # errors must be 'ignore' or 'raise'
            ({"errors": "something"}, ValueError, "The parameter errors must.*"),
            ({"join": "inner"}, NotImplementedError, "Only left join is supported"),
        ],
    )
    def test_update_raise_bad_parameter(self, bad_kwarg, exception, msg):
        # [FUNCTION DOC]
        # Parametrized over invalid keyword arguments: an unsupported errors value raises a value error (message starting with
        # that the parameter errors must be one of the allowed options) and a non-left join raises a not-implemented error
        # saying only left join is supported.
        df = DataFrame([[1.5, 1, 3.0]])
        with pytest.raises(exception, match=msg):
            df.update(df, **bad_kwarg)

    def test_update_raise_on_overlap(self):
        # [FUNCTION DOC]
        # With errors set to raise, updating where the other frame has non-missing values over existing non-missing target values
        # raises a value error saying data overlaps.
        df = DataFrame(
            [[1.5, 1, 3.0], [1.5, np.nan, 3.0], [1.5, np.nan, 3], [1.5, np.nan, 3]]
        )

        other = DataFrame([[2.0, np.nan], [np.nan, 7]], index=[1, 3], columns=[1, 2])
        with pytest.raises(ValueError, match="Data overlaps"):
            df.update(other, errors="raise")

    def test_update_from_non_df(self):
        # [FUNCTION DOC]
        # Update accepts a mapping of series, and a mapping of lists, as the other argument; the frame afterwards equals a frame
        # built from the changed mapping.
        d = {"a": Series([1, 2, 3, 4]), "b": Series([5, 6, 7, 8])}
        df = DataFrame(d)

        d["a"] = Series([5, 6, 7, 8])
        df.update(d)

        expected = DataFrame(d)

        tm.assert_frame_equal(df, expected)

        d = {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}
        df = DataFrame(d)

        d["a"] = [5, 6, 7, 8]
        df.update(d)

        expected = DataFrame(d)

        tm.assert_frame_equal(df, expected)

    def test_update_datetime_tz(self):
        # [FUNCTION DOC]
        # Regression test (GH 25807). Updating a timezone-aware datetime frame with itself emits no warning and leaves it unchanged.
        # GH 25807
        result = DataFrame([pd.Timestamp("2019", tz="UTC")])
        with tm.assert_produces_warning(None):
            result.update(result)
        expected = DataFrame([pd.Timestamp("2019", tz="UTC")])
        tm.assert_frame_equal(result, expected)

    def test_update_datetime_tz_in_place(self):
        # [FUNCTION DOC]
        # Regression issue 56227. Updating a timezone-aware frame from itself plus one day gives the shifted values in the frame
        # while a previously taken view of the frame keeps the original values (no write through).
        # https://github.com/pandas-dev/pandas/issues/56227
        result = DataFrame([pd.Timestamp("2019", tz="UTC")])
        orig = result.copy()
        view = result[:]
        result.update(result + pd.Timedelta(days=1))
        expected = DataFrame([pd.Timestamp("2019-01-02", tz="UTC")])
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(view, orig)

    def test_update_with_different_dtype(self):
        # [FUNCTION DOC]
        # Regression test (GH 3217). Updating an all-missing float column from a series of strings raises a type error for an invalid value.
        # GH#3217
        df = DataFrame({"a": [1, 3], "b": [np.nan, 2]})
        df["c"] = np.nan
        with pytest.raises(TypeError, match="Invalid value"):
            df.update({"c": Series(["foo"], index=[0])})

    @pytest.mark.parametrize("dtype", ["str", object])
    def test_update_modify_view(self, dtype):
        # [FUNCTION DOC]
        # Regression test (GH 47188). Parametrized over string dtype and object dtype. Updating a frame from another where the
        # other has missing entries only fills in non-missing entries and a pre-existing view of the target keeps its original values.
        # GH#47188
        df = DataFrame({"A": ["1", np.nan], "B": ["100", np.nan]}, dtype=dtype)
        df2 = DataFrame({"A": ["a", "x"], "B": ["100", "200"]}, dtype=dtype)
        df2_orig = df2.copy()
        result_view = df2[:]
        df2.update(df)
        expected = DataFrame({"A": ["1", "x"], "B": ["100", "200"]}, dtype=dtype)
        tm.assert_frame_equal(df2, expected)
        tm.assert_frame_equal(result_view, df2_orig)

    def test_update_dt_column_with_NaT_create_column(self):
        # [FUNCTION DOC]
        # Regression test (GH 16713). With overwrite off, updating a frame with a missing-value datetime column keeps that column
        # and fills the numeric column from the other frame, giving floats.
        # GH#16713
        df = DataFrame({"A": [1, None], "B": [pd.NaT, pd.to_datetime("2016-01-01")]})
        df2 = DataFrame({"A": [2, 3]})
        df.update(df2, overwrite=False)
        expected = DataFrame(
            {"A": [1.0, 3.0], "B": [pd.NaT, pd.to_datetime("2016-01-01")]}
        )
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "value_df, value_other, dtype",
        [
            (True, False, bool),
            (1, 2, int),
            (1.0, 2.0, float),
            (1.0 + 1j, 2.0 + 2j, complex),
            (np.uint64(1), np.uint(2), np.dtype("ubyte")),
            (np.uint64(1), np.uint(2), np.dtype("intc")),
            ("a", "b", pd.StringDtype()),
            (
                pd.to_timedelta("1 ms"),
                pd.to_timedelta("2 ms"),
                np.dtype("timedelta64[ns]"),
            ),
            (
                np.datetime64("2000-01-01T00:00:00"),
                np.datetime64("2000-01-02T00:00:00"),
                np.dtype("datetime64[ns]"),
            ),
            (1, 2, pd.Int64Dtype()),
        ],
    )
    def test_update_preserve_dtype(self, value_df, value_other, dtype):
        # [FUNCTION DOC]
        # Regression test (GH 55509). Parametrized over pairs of scalar values and dtypes (bool, int, float, complex, small
        # unsigned/integer numpy dtypes, string, timedelta, datetime, nullable integer). Updating a one-column frame at one
        # label from another frame of the same dtype must keep that dtype and take the new value at the matching label.
        # GH#55509
        df = DataFrame({"a": [value_df] * 2}, index=[1, 2], dtype=dtype)
        other = DataFrame({"a": [value_other]}, index=[1], dtype=dtype)
        expected = DataFrame({"a": [value_other, value_df]}, index=[1, 2], dtype=dtype)
        df.update(other)
        tm.assert_frame_equal(df, expected)

    def test_update_raises_on_duplicate_argument_index(self):
        # [FUNCTION DOC]
        # Regression test (GH 55509). When the other frame has duplicate index labels, update raises a value error mentioning a duplicate index.
        # GH#55509
        df = DataFrame({"a": [1, 1]}, index=[1, 2])
        other = DataFrame({"a": [2, 3]}, index=[1, 1])
        with pytest.raises(ValueError, match="duplicate index"):
            df.update(other)

    def test_update_without_intersection(self):
        # [FUNCTION DOC]
        # Regression test (GH 63452). Updating from a frame whose index shares no labels with the target leaves the target unchanged.
        # GH#63452
        orig = DataFrame({"a": [1]}, index=[1])
        df = orig.copy()
        other = DataFrame({"a": [2]}, index=[2])
        df.update(other)
        tm.assert_frame_equal(df, orig)

    def test_update_on_duplicate_frame_unique_argument_index(self):
        # [FUNCTION DOC]
        # Regression test (GH 55509). Target frame with duplicate labels and the other with unique labels: every matching duplicate row
        # takes the new value; the C int dtype is preserved.
        # GH#55509
        df = DataFrame({"a": [1, 1, 1]}, index=[1, 1, 2], dtype=np.dtype("intc"))
        other = DataFrame({"a": [2, 3]}, index=[1, 2], dtype=np.dtype("intc"))
        expected = DataFrame({"a": [2, 2, 3]}, index=[1, 1, 2], dtype=np.dtype("intc"))
        df.update(other)
        tm.assert_frame_equal(df, expected)

    def test_update_preserve_mixed_dtypes(self):
        # [FUNCTION DOC]
        # Regression test (GH 44104). Frame with nullable integer and string columns updated from a shorter frame of the same
        # dtypes: values for matching labels are replaced and both extension dtypes are preserved.
        # GH#44104
        dtype1 = pd.Int64Dtype()
        dtype2 = pd.StringDtype()
        df = DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        df = df.astype({"a": dtype1, "b": dtype2})

        other = DataFrame({"a": [4, 5], "b": ["a", "b"]})
        other = other.astype({"a": dtype1, "b": dtype2})

        expected = DataFrame({"a": [4, 5, 3], "b": ["a", "b", "z"]})
        expected = expected.astype({"a": dtype1, "b": dtype2})

        df.update(other)
        tm.assert_frame_equal(df, expected)
