# [MODULE DOC]
# Tests for Series mask, compared against where with an inverted condition, including
# dtype casting, array conditions and in-place use.
import numpy as np
import pytest

from pandas import Series
import pandas._testing as tm


def test_mask():
    # [FUNCTION DOC]
    # Compares mask to where with an inverted condition on random data: default
    # replacement, explicit NaN, a replacement series, and a partial condition series
    # that is shorter than the series. Passing a scalar or a mismatched-length array as
    # the condition raises a value error saying an array condition must be the same
    # shape as the series.
    # compare with tested results in test_where
    s = Series(np.random.default_rng(2).standard_normal(5))
    cond = s > 0

    rs = s.where(~cond, np.nan)
    tm.assert_series_equal(rs, s.mask(cond))

    rs = s.where(~cond)
    rs2 = s.mask(cond)
    tm.assert_series_equal(rs, rs2)

    rs = s.where(~cond, -s)
    rs2 = s.mask(cond, -s)
    tm.assert_series_equal(rs, rs2)

    cond = Series([True, False, False, True, False], index=s.index)
    s2 = -(s.abs())
    rs = s2.where(~cond[:3])
    rs2 = s2.mask(cond[:3])
    tm.assert_series_equal(rs, rs2)

    rs = s2.where(~cond[:3], -s2)
    rs2 = s2.mask(cond[:3], -s2)
    tm.assert_series_equal(rs, rs2)

    msg = "Array conditional must be same shape as self"
    with pytest.raises(ValueError, match=msg):
        s.mask(1)
    with pytest.raises(ValueError, match=msg):
        s.mask(cond[:3].values, -s)


def test_mask_casts():
    # [FUNCTION DOC]
    # Verifies masking integers with NaN as replacement converts the dtype to float with
    # NaN at masked positions.
    # dtype changes
    ser = Series([1, 2, 3, 4])
    result = ser.mask(ser > 2, np.nan)
    expected = Series([1, 2, np.nan, np.nan])
    tm.assert_series_equal(result, expected)


def test_mask_casts2():
    # [FUNCTION DOC]
    # Regression test (gh-21891). Masking an integer series with a boolean list
    # condition gives float with NaN at the first position.
    # see gh-21891
    ser = Series([1, 2])
    res = ser.mask([True, False])

    exp = Series([np.nan, 2])
    tm.assert_series_equal(res, exp)


def test_mask_inplace():
    # [FUNCTION DOC]
    # Verifies in-place masking (default fill, then replacement series) modifies the
    # series in place to equal the non-in-place result; the in-place default result with
    # missing dropped equals the non-masked values.
    s = Series(np.random.default_rng(2).standard_normal(5))
    cond = s > 0

    rs = s.copy()
    rs.mask(cond, inplace=True)
    tm.assert_series_equal(rs.dropna(), s[~cond])
    tm.assert_series_equal(rs, s.mask(cond))

    rs = s.copy()
    rs.mask(cond, -s, inplace=True)
    tm.assert_series_equal(rs, s.mask(cond, -s))
