# [MODULE DOC]
# Tests for reorder_levels on DataFrame and Series with multi-level indexes.

import numpy as np
import pytest

from pandas import (
    DataFrame,
    MultiIndex,
)
import pandas._testing as tm


class TestReorderLevels:
    # [CLASS DOC]
    # Groups the reorder_levels tests.
    def test_reorder_levels(self, frame_or_series):
        # [FUNCTION DOC]
        # Parametrized over frame and series with a three-level index named L0, L1, L2. Verifies reordering by position or by name with the identity order returns an equal object, a rotated order gives the rotated levels and names, and repeating a level (by position or name) gives an index with the same level repeated and repeated names.
        index = MultiIndex(
            levels=[["bar"], ["one", "two", "three"], [0, 1]],
            codes=[[0, 0, 0, 0, 0, 0], [0, 1, 2, 0, 1, 2], [0, 1, 0, 1, 0, 1]],
            names=["L0", "L1", "L2"],
        )
        df = DataFrame({"A": np.arange(6), "B": np.arange(6)}, index=index)
        obj = tm.get_obj(df, frame_or_series)

        # no change, position
        result = obj.reorder_levels([0, 1, 2])
        tm.assert_equal(obj, result)

        # no change, labels
        result = obj.reorder_levels(["L0", "L1", "L2"])
        tm.assert_equal(obj, result)

        # rotate, position
        result = obj.reorder_levels([1, 2, 0])
        e_idx = MultiIndex(
            levels=[["one", "two", "three"], [0, 1], ["bar"]],
            codes=[[0, 1, 2, 0, 1, 2], [0, 1, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0]],
            names=["L1", "L2", "L0"],
        )
        expected = DataFrame({"A": np.arange(6), "B": np.arange(6)}, index=e_idx)
        expected = tm.get_obj(expected, frame_or_series)
        tm.assert_equal(result, expected)

        result = obj.reorder_levels([0, 0, 0])
        e_idx = MultiIndex(
            levels=[["bar"], ["bar"], ["bar"]],
            codes=[[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
            names=["L0", "L0", "L0"],
        )
        expected = DataFrame({"A": np.arange(6), "B": np.arange(6)}, index=e_idx)
        expected = tm.get_obj(expected, frame_or_series)
        tm.assert_equal(result, expected)

        result = obj.reorder_levels(["L0", "L0", "L0"])
        tm.assert_equal(result, expected)

    def test_reorder_levels_swaplevel_equivalence(
        self, multiindex_year_month_day_dataframe_random_data
    ):
        # [FUNCTION DOC]
        # Uses the year-month-day multi-index fixture. Verifies reorder_levels equals the equivalent chained swaplevel calls for a frame, a series and a transposed frame on the columns axis. Also verifies a type error mentioning a hierarchical axis when the axis is not a multi-index, and an index error mentioning too many levels when an index reorders with more level positions than it has.
        ymd = multiindex_year_month_day_dataframe_random_data

        result = ymd.reorder_levels(["month", "day", "year"])
        expected = ymd.swaplevel(0, 1).swaplevel(1, 2)
        tm.assert_frame_equal(result, expected)

        result = ymd["A"].reorder_levels(["month", "day", "year"])
        expected = ymd["A"].swaplevel(0, 1).swaplevel(1, 2)
        tm.assert_series_equal(result, expected)

        result = ymd.T.reorder_levels(["month", "day", "year"], axis=1)
        expected = ymd.T.swaplevel(0, 1, axis=1).swaplevel(1, 2, axis=1)
        tm.assert_frame_equal(result, expected)

        with pytest.raises(TypeError, match="hierarchical axis"):
            ymd.reorder_levels([1, 2], axis=1)

        with pytest.raises(IndexError, match="Too many levels"):
            ymd.index.reorder_levels([1, 2, 3])
