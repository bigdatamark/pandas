# [MODULE DOC]
# Tests for the function that builds a one-dimensional array-like by repeating a
# scalar to a given dtype: categorical dtype handling, nanosecond precision for
# datetime-like scalars, and rejection of mismatched datetime-like types.

import numpy as np
import pytest

from pandas.core.dtypes.cast import construct_1d_arraylike_from_scalar
from pandas.core.dtypes.dtypes import CategoricalDtype

from pandas import (
    Categorical,
    Timedelta,
)
import pandas._testing as tm


def test_cast_1d_array_like_from_scalar_categorical():
    # [FUNCTION DOC]
    # Regression test for a GitHub issue: building from a scalar with an unordered
    # categorical dtype keeps the dtype's categories and ordering; the result equals
    # the expected categorical of two repeated values.
    # see gh-19565
    #
    # Categorical result from scalar did not maintain
    # categories and ordering of the passed dtype.
    cats = ["a", "b", "c"]
    cat_type = CategoricalDtype(categories=cats, ordered=False)
    expected = Categorical(["a", "a"], categories=cats)

    result = construct_1d_arraylike_from_scalar("a", len(expected), cat_type)
    tm.assert_categorical_equal(result, expected)


def test_cast_1d_array_like_from_timestamp(fixed_now_ts):
    # [FUNCTION DOC]
    # Fixture fixed_now_ts. A Timestamp with a one nanosecond offset repeated to
    # M8[ns] must retain the nanosecond: the first element equals the scalar.
    # check we dont lose nanoseconds
    ts = fixed_now_ts + Timedelta(1)
    res = construct_1d_arraylike_from_scalar(ts, 2, np.dtype("M8[ns]"))
    assert res[0] == ts


def test_cast_1d_array_like_from_timedelta():
    # [FUNCTION DOC]
    # A one-nanosecond Timedelta repeated to m8[ns] retains the nanosecond: first
    # element equals the scalar.
    # check we dont lose nanoseconds
    td = Timedelta(1)
    res = construct_1d_arraylike_from_scalar(td, 2, np.dtype("m8[ns]"))
    assert res[0] == td


def test_cast_1d_array_like_mismatched_datetimelike():
    # [FUNCTION DOC]
    # Using a timedelta NaT or value with a datetime64 dtype, or a datetime NaT or
    # value with a timedelta64 dtype, raises a type error saying it cannot cast
    # (four combinations).
    td = np.timedelta64("NaT", "ns")
    dt = np.datetime64("NaT", "ns")

    with pytest.raises(TypeError, match="Cannot cast"):
        construct_1d_arraylike_from_scalar(td, 2, dt.dtype)

    with pytest.raises(TypeError, match="Cannot cast"):
        construct_1d_arraylike_from_scalar(np.timedelta64(4, "ns"), 2, dt.dtype)

    with pytest.raises(TypeError, match="Cannot cast"):
        construct_1d_arraylike_from_scalar(dt, 2, td.dtype)

    with pytest.raises(TypeError, match="Cannot cast"):
        construct_1d_arraylike_from_scalar(np.datetime64(4, "ns"), 2, td.dtype)
