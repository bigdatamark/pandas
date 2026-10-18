# [MODULE DOC]
# Tests for reductions on SparseArray: all/any, sum, mean (also through the
# numpy functions with their unsupported-argument errors), min/max with various
# fill values and no valid values, and argmax/argmin. No import-time side effects.

import numpy as np
import pytest

from pandas import (
    NaT,
    SparseDtype,
    Timestamp,
    isna,
)
from pandas.core.arrays.sparse import SparseArray


class TestReductions:
    # [CLASS DOC]
    # Tests boolean and arithmetic reductions (all, any, sum, mean), both as methods
    # and via the numpy functions.
    @pytest.mark.parametrize(
        "data,pos,neg",
        [
            ([True, True, True], True, False),
            ([1, 2, 1], 1, 0),
            ([1.0, 2.0, 1.0], 1.0, 0.0),
        ],
    )
    def test_all(self, data, pos, neg):
        # [FUNCTION DOC]
        # Parametrized over (data, truthy fill value, falsy value) for bool, int and
        # float lists. Verifies the all method is true on all-truthy data with default
        # and truthy fill value, and false after one element is set falsy (the data list
        # is mutated in place by the test).
        # GH#17570
        out = SparseArray(data).all()
        assert out

        out = SparseArray(data, fill_value=pos).all()
        assert out

        data[1] = neg
        out = SparseArray(data).all()
        assert not out

        out = SparseArray(data, fill_value=pos).all()
        assert not out

    @pytest.mark.parametrize(
        "data,pos,neg",
        [
            ([True, True, True], True, False),
            ([1, 2, 1], 1, 0),
            ([1.0, 2.0, 1.0], 1.0, 0.0),
        ],
    )
    def test_numpy_all(self, data, pos, neg):
        # [FUNCTION DOC]
        # Same scenarios as the all method test but through the numpy all function.
        # Also verifies that passing the out argument raises a value error ("the 'out'
        # parameter is not supported").
        # KNOWN ISSUE: the comment about a different message on python 2 is stale; the
        # check applies to the current implementation only.
        # GH#17570
        out = np.all(SparseArray(data))
        assert out

        out = np.all(SparseArray(data, fill_value=pos))
        assert out

        data[1] = neg
        out = np.all(SparseArray(data))
        assert not out

        out = np.all(SparseArray(data, fill_value=pos))
        assert not out

        # raises with a different message on py2.
        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.all(SparseArray(data), out=np.array([]))

    @pytest.mark.parametrize(
        "data,pos,neg",
        [
            ([False, True, False], True, False),
            ([0, 2, 0], 2, 0),
            ([0.0, 2.0, 0.0], 2.0, 0.0),
        ],
    )
    def test_any(self, data, pos, neg):
        # [FUNCTION DOC]
        # Parametrized over (data, truthy fill value, falsy value). Verifies the any
        # method is true when some element is truthy (default and truthy fill) and false
        # after the only truthy element is set falsy (data list mutated in place).
        # GH#17570
        out = SparseArray(data).any()
        assert out

        out = SparseArray(data, fill_value=pos).any()
        assert out

        data[1] = neg
        out = SparseArray(data).any()
        assert not out

        out = SparseArray(data, fill_value=pos).any()
        assert not out

    @pytest.mark.parametrize(
        "data,pos,neg",
        [
            ([False, True, False], True, False),
            ([0, 2, 0], 2, 0),
            ([0.0, 2.0, 0.0], 2.0, 0.0),
        ],
    )
    def test_numpy_any(self, data, pos, neg):
        # [FUNCTION DOC]
        # Same scenarios as the any method test but through the numpy any function; also
        # verifies passing the out argument raises a value error ("the 'out' parameter
        # is not supported").
        # GH#17570
        out = np.any(SparseArray(data))
        assert out

        out = np.any(SparseArray(data, fill_value=pos))
        assert out

        data[1] = neg
        out = np.any(SparseArray(data))
        assert not out

        out = np.any(SparseArray(data, fill_value=pos))
        assert not out

        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.any(SparseArray(data), out=out)

    def test_sum(self):
        # [FUNCTION DOC]
        # Verifies sum of 0..9 as floats is 45, and 40 after one value is replaced by NaN
        # (NaN skipped), for fill value 2 and NaN fill value.
        data = np.arange(10).astype(float)
        out = SparseArray(data).sum()
        assert out == 45.0

        data[5] = np.nan
        out = SparseArray(data, fill_value=2).sum()
        assert out == 40.0

        out = SparseArray(data, fill_value=np.nan).sum()
        assert out == 40.0

    @pytest.mark.parametrize(
        "arr",
        [[0, 1, np.nan, 1], [0, 1, 1]],
    )
    @pytest.mark.parametrize("fill_value", [0, 1, np.nan])
    @pytest.mark.parametrize("min_count, expected", [(3, 2), (4, np.nan)])
    def test_sum_min_count(self, arr, fill_value, min_count, expected):
        # [FUNCTION DOC]
        # Regression test (GH#25777). Parametrized over data with and without NaN, fill
        # values 0, 1, NaN, and (min_count, expected) of (3, 2) and (4, NaN). Verifies
        # the sum is 2 when the number of valid values meets min_count and NaN otherwise.
        # GH#25777
        sparray = SparseArray(np.array(arr), fill_value=fill_value)
        result = sparray.sum(min_count=min_count)
        if np.isnan(expected):
            assert np.isnan(result)
        else:
            assert result == expected

    def test_bool_sum_min_count(self):
        # [FUNCTION DOC]
        # Verifies sum of a boolean SparseArray (fill value True) with min_count 1 is 5
        # and with min_count 11 (more than the length) is missing.
        spar_bool = SparseArray([False, True] * 5, dtype=np.bool_, fill_value=True)
        res = spar_bool.sum(min_count=1)
        assert res == 5
        res = spar_bool.sum(min_count=11)
        assert isna(res)

    def test_numpy_sum(self):
        # [FUNCTION DOC]
        # Like the sum method test but via the numpy sum function, and verifies that the
        # dtype argument and the out argument each raise a value error stating the
        # parameter is not supported.
        data = np.arange(10).astype(float)
        out = np.sum(SparseArray(data))
        assert out == 45.0

        data[5] = np.nan
        out = np.sum(SparseArray(data, fill_value=2))
        assert out == 40.0

        out = np.sum(SparseArray(data, fill_value=np.nan))
        assert out == 40.0

        msg = "the 'dtype' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.sum(SparseArray(data), dtype=np.int64)

        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.sum(SparseArray(data), out=out)

    def test_mean(self):
        # [FUNCTION DOC]
        # Verifies mean of 0..9 is 4.5, and 40/9 after one element becomes NaN.
        data = np.arange(10).astype(float)
        out = SparseArray(data).mean()
        assert out == 4.5

        data[5] = np.nan
        out = SparseArray(data).mean()
        assert out == 40.0 / 9

    def test_numpy_mean(self):
        # [FUNCTION DOC]
        # Like the mean method test via the numpy mean function, and verifies the dtype
        # and out arguments raise value errors ("the ... parameter is not supported").
        data = np.arange(10).astype(float)
        out = np.mean(SparseArray(data))
        assert out == 4.5

        data[5] = np.nan
        out = np.mean(SparseArray(data))
        assert out == 40.0 / 9

        msg = "the 'dtype' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.mean(SparseArray(data), dtype=np.int64)

        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.mean(SparseArray(data), out=out)


class TestMinMax:
    # [CLASS DOC]
    # Tests minimum and maximum reductions.
    @pytest.mark.parametrize(
        "raw_data,max_expected,min_expected",
        [
            (np.arange(5.0), [4], [0]),
            (-np.arange(5.0), [0], [-4]),
            (np.array([0, 1, 2, np.nan, 4]), [4], [0]),
            (np.array([np.nan] * 5), [np.nan], [np.nan]),
            (np.array([]), [np.nan], [np.nan]),
        ],
    )
    def test_nan_fill_value(self, raw_data, max_expected, min_expected):
        # [FUNCTION DOC]
        # Parametrized over float data (ramps, with NaN, all NaN, empty) with expected
        # max and min. With a NaN fill value, verifies max and min with skipna default
        # and, with skipna false, that NaN data gives NaN results while NaN-free data
        # gives the normal results.
        arr = SparseArray(raw_data)
        max_result = arr.max()
        min_result = arr.min()
        assert max_result in max_expected
        assert min_result in min_expected

        max_result = arr.max(skipna=False)
        min_result = arr.min(skipna=False)
        if np.isnan(raw_data).any():
            assert np.isnan(max_result)
            assert np.isnan(min_result)
        else:
            assert max_result in max_expected
            assert min_result in min_expected

    @pytest.mark.parametrize(
        "fill_value,max_expected,min_expected",
        [
            (100, 100, 0),
            (-100, 1, -100),
        ],
    )
    def test_fill_value(self, fill_value, max_expected, min_expected):
        # [FUNCTION DOC]
        # Parametrized over fill values (100 and -100) with expected max and min.
        # Builds an int SparseArray whose dtype fill value equals the first element and
        # verifies the max and min take the fill value into account.
        arr = SparseArray(
            np.array([fill_value, 0, 1]), dtype=SparseDtype("int", fill_value)
        )
        max_result = arr.max()
        assert max_result == max_expected

        min_result = arr.min()
        assert min_result == min_expected

    def test_only_fill_value(self):
        # [FUNCTION DOC]
        # Verifies that when every element equals the fill value (so no stored valid
        # values exist) max and min, with and without skipna, return the fill value.
        fv = 100
        arr = SparseArray(np.array([fv, fv, fv]), dtype=SparseDtype("int", fv))
        assert len(arr._valid_sp_values) == 0

        assert arr.max() == fv
        assert arr.min() == fv
        assert arr.max(skipna=False) == fv
        assert arr.min(skipna=False) == fv

    @pytest.mark.parametrize("func", ["min", "max"])
    @pytest.mark.parametrize("data", [np.array([]), np.array([np.nan, np.nan])])
    @pytest.mark.parametrize(
        "dtype,expected",
        [
            (SparseDtype(np.float64, np.nan), np.nan),
            (SparseDtype(np.float64, 5.0), np.nan),
            (SparseDtype("datetime64[ns]", NaT), NaT),
            (SparseDtype("datetime64[ns]", Timestamp("2018-05-05")), NaT),
        ],
    )
    def test_na_value_if_no_valid_values(self, func, data, dtype, expected):
        # [FUNCTION DOC]
        # Parametrized over function (min, max), data (empty, all NaN), and sparse dtypes
        # (float with NaN or non-NaN fill, datetime with NaT or a timestamp fill) with the
        # expected result (NaN or NaT). Verifies the reduction of an array with no valid
        # values returns the NA value of the dtype.
        # KNOWN ISSUE: a TODO notes the datetime case may return a numpy NaT rather than
        # the pandas NaT, and the assertion accepts either.
        arr = SparseArray(data, dtype=dtype)
        result = getattr(arr, func)()
        if expected is NaT:
            # TODO: pin down whether we wrap datetime64("NaT")
            assert result is NaT or np.isnat(result)
        else:
            assert np.isnan(result)


class TestArgmaxArgmin:
    # [CLASS DOC]
    # Tests argmax and argmin on SparseArray.
    @pytest.mark.parametrize(
        "arr,argmax_expected,argmin_expected",
        [
            (SparseArray([1, 2, 0, 1, 2]), 1, 2),
            (SparseArray([-1, -2, 0, -1, -2]), 2, 1),
            (SparseArray([np.nan, 1, 0, 0, np.nan, -1]), 1, 5),
            (SparseArray([np.nan, 1, 0, 0, np.nan, 2]), 5, 2),
            (SparseArray([np.nan, 1, 0, 0, np.nan, 2], fill_value=-1), 5, 2),
            (SparseArray([np.nan, 1, 0, 0, np.nan, 2], fill_value=0), 5, 2),
            (SparseArray([np.nan, 1, 0, 0, np.nan, 2], fill_value=1), 5, 2),
            (SparseArray([np.nan, 1, 0, 0, np.nan, 2], fill_value=2), 5, 2),
            (SparseArray([np.nan, 1, 0, 0, np.nan, 2], fill_value=3), 5, 2),
            (SparseArray([0] * 10 + [-1], fill_value=0), 0, 10),
            (SparseArray([0] * 10 + [-1], fill_value=-1), 0, 10),
            (SparseArray([0] * 10 + [-1], fill_value=1), 0, 10),
            (SparseArray([-1] + [0] * 10, fill_value=0), 1, 0),
            (SparseArray([1] + [0] * 10, fill_value=0), 0, 1),
            (SparseArray([-1] + [0] * 10, fill_value=-1), 1, 0),
            (SparseArray([1] + [0] * 10, fill_value=1), 0, 1),
        ],
    )
    def test_argmax_argmin(self, arr, argmax_expected, argmin_expected):
        # [FUNCTION DOC]
        # Parametrized over many arrays (different data and fill values, including NaN
        # and fill value greater/lower/equal to data) with expected positions. Verifies
        # argmax and argmin return the expected integer positions.
        argmax_result = arr.argmax()
        argmin_result = arr.argmin()
        assert argmax_result == argmax_expected
        assert argmin_result == argmin_expected

    @pytest.mark.parametrize("method", ["argmax", "argmin"])
    def test_empty_array(self, method):
        # [FUNCTION DOC]
        # Parametrized over method (argmax, argmin). Verifies calling it on an empty
        # SparseArray raises a value error saying it attempted to get the method of an
        # empty sequence.
        msg = f"attempt to get {method} of an empty sequence"
        arr = SparseArray([])
        with pytest.raises(ValueError, match=msg):
            getattr(arr, method)()
