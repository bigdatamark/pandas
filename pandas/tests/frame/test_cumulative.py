"""
Tests for DataFrame cumulative operations

See also
--------
tests.series.test_cumulative
"""
# [MODULE DOC]
# Tests for the cumulative reductions (cumulative min, max, sum, product) on the
# tabular container, covering smoke runs, agreement with the per-column
# one-dimensional implementation, dtype preservation, and the numeric-only flag.
# Uses shared fixtures from the test-suite conftest; no import-time side effects.

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
    Timestamp,
)
import pandas._testing as tm


class TestDataFrameCumulativeOps:
    # [CLASS DOC]
    # Test group for tabular cumulative operations. Stateless; each method builds
    # its own data or uses fixtures.
    # ---------------------------------------------------------------------
    # Cumulative Operations - cumsum, cummax, ...

    def test_cumulative_ops_smoke(self):
        # [FUNCTION DOC]
        # Smoke test: cumulative max, min and sum run without raising on an integer
        # frame, and cumulative sum runs on a 4x5 integer frame. Nothing is asserted
        # about the results.
        # it works
        df = DataFrame({"A": np.arange(20)}, index=np.arange(20))
        df.cummax()
        df.cummin()
        df.cumsum()

        dm = DataFrame(np.arange(20).reshape(4, 5), index=range(4), columns=range(5))
        # TODO(wesm): do something with this?
        dm.cumsum()

    def test_cumprod_smoke(self, datetime_frame):
        # [FUNCTION DOC]
        # Smoke test: cumulative product along both axes runs without raising on a
        # datetime-indexed fixture frame with some missing values, filled with zero and
        # cast to a 64-bit-default integer and to a 32-bit integer type. No result
        # assertions.
        datetime_frame.iloc[5:10, 0] = np.nan
        datetime_frame.iloc[10:15, 1] = np.nan
        datetime_frame.iloc[15:, 2] = np.nan

        # ints
        df = datetime_frame.fillna(0).astype(int)
        df.cumprod(0)
        df.cumprod(1)

        # ints32
        df = datetime_frame.fillna(0).astype(np.int32)
        df.cumprod(0)
        df.cumprod(1)

    def test_cumulative_ops_match_series_apply(
        self, datetime_frame, all_numeric_accumulations
    ):
        # [FUNCTION DOC]
        # Parametrized over every numeric accumulation name. Verifies that calling the
        # accumulation on the frame equals applying the same Series method column-wise
        # (axis zero) and row-wise (axis one), on a fixture frame with missing blocks in
        # three columns. Also checks the axis-one result keeps the input shape.
        datetime_frame.iloc[5:10, 0] = np.nan
        datetime_frame.iloc[10:15, 1] = np.nan
        datetime_frame.iloc[15:, 2] = np.nan

        # axis = 0
        result = getattr(datetime_frame, all_numeric_accumulations)()
        expected = datetime_frame.apply(getattr(Series, all_numeric_accumulations))
        tm.assert_frame_equal(result, expected)

        # axis = 1
        result = getattr(datetime_frame, all_numeric_accumulations)(axis=1)
        expected = datetime_frame.apply(
            getattr(Series, all_numeric_accumulations), axis=1
        )
        tm.assert_frame_equal(result, expected)

        # fix issue TODO: GH ref?
        assert np.shape(result) == np.shape(datetime_frame)

    def test_cumsum_preserve_dtypes(self):
        # [FUNCTION DOC]
        # Verifies cumulative sum does not upcast to object: an integer column stays
        # 64-bit integer, a float column stays float, and a boolean column matches the
        # cumulative sum of that column on its own (GH#19296).
        # GH#19296 dont incorrectly upcast to object
        df = DataFrame({"A": [1, 2, 3], "B": [1, 2, 3.0], "C": [True, False, False]})

        result = df.cumsum()

        expected = DataFrame(
            {
                "A": Series([1, 3, 6], dtype=np.int64),
                "B": Series([1, 3, 6], dtype=np.float64),
                "C": df["C"].cumsum(),
            }
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("method", ["cumsum", "cumprod", "cummin", "cummax"])
    @pytest.mark.parametrize("axis", [0, 1])
    def test_numeric_only_flag(self, method, axis):
        # [FUNCTION DOC]
        # Parametrized over the four cumulative methods and both axes. Builds a frame
        # with integer, boolean, string, float and datetime columns and checks that with
        # the numeric-only flag enabled the result equals the same method on the frame
        # with the string and datetime columns dropped.
        df = DataFrame(
            {
                "int": [1, 2, 3],
                "bool": [True, False, False],
                "string": ["a", "b", "c"],
                "float": [1.0, 3.5, 4.0],
                "datetime": [
                    Timestamp(2018, 1, 1),
                    Timestamp(2019, 1, 1),
                    Timestamp(2020, 1, 1),
                ],
            }
        )
        df_numeric_only = df.drop(["string", "datetime"], axis=1)

        result = getattr(df, method)(axis=axis, numeric_only=True)
        expected = getattr(df_numeric_only, method)(axis)
        tm.assert_frame_equal(result, expected)
