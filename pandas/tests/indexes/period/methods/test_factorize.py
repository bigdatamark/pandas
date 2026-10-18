# [MODULE DOC]
# Tests for PeriodIndex.factorize.
import numpy as np

from pandas import PeriodIndex
import pandas._testing as tm


class TestFactorize:
    # [CLASS DOC]
    # Groups factorize tests.
    def test_factorize_period(self):
        # [FUNCTION DOC]
        # Monotonic repeated monthly periods give codes and uniques in order, with or
        # without sorting.
        idx1 = PeriodIndex(
            ["2014-01", "2014-01", "2014-02", "2014-02", "2014-03", "2014-03"],
            freq="M",
        )

        exp_arr = np.array([0, 0, 1, 1, 2, 2], dtype=np.intp)
        exp_idx = PeriodIndex(["2014-01", "2014-02", "2014-03"], freq="M")

        arr, idx = idx1.factorize()
        tm.assert_numpy_array_equal(arr, exp_arr)
        tm.assert_index_equal(idx, exp_idx)

        arr, idx = idx1.factorize(sort=True)
        tm.assert_numpy_array_equal(arr, exp_arr)
        tm.assert_index_equal(idx, exp_idx)

    def test_factorize_period_nonmonotonic(self):
        # [FUNCTION DOC]
        # Non-monotonic input: sorting yields sorted uniques with codes pointing to
        # them; no sorting yields uniques in order of appearance.
        idx2 = PeriodIndex(
            ["2014-03", "2014-03", "2014-02", "2014-01", "2014-03", "2014-01"],
            freq="M",
        )
        exp_idx = PeriodIndex(["2014-01", "2014-02", "2014-03"], freq="M")

        exp_arr = np.array([2, 2, 1, 0, 2, 0], dtype=np.intp)
        arr, idx = idx2.factorize(sort=True)
        tm.assert_numpy_array_equal(arr, exp_arr)
        tm.assert_index_equal(idx, exp_idx)

        exp_arr = np.array([0, 0, 1, 2, 0, 2], dtype=np.intp)
        exp_idx = PeriodIndex(["2014-03", "2014-02", "2014-01"], freq="M")
        arr, idx = idx2.factorize()
        tm.assert_numpy_array_equal(arr, exp_arr)
        tm.assert_index_equal(idx, exp_idx)
