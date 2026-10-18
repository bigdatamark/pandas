# [MODULE DOC]
# Tests for Series where and related masked assignment: unsafe casting on boolean-mask
# setitem, alignment of the condition, error messages for invalid conditions, list-like
# assignment length errors, broadcasting, in-place use, duplicates in the index, mixed
# numeric and string replacement, and extension dtypes (datetime with timezone, sparse,
# categorical).
import numpy as np
import pytest

from pandas.core.dtypes.common import is_integer

import pandas as pd
from pandas import (
    Series,
    Timestamp,
    date_range,
    isna,
)
import pandas._testing as tm


def test_where_unsafe_int(any_signed_int_numpy_dtype):
    # [FUNCTION DOC]
    # Parametrized over signed integer dtypes. Assigning a range of the right length
    # through a boolean mask on an integer series keeps the dtype and writes the values.
    # KNOWN ISSUE: the name says where but the test uses mask-based item assignment.
    s = Series(np.arange(10), dtype=any_signed_int_numpy_dtype)
    mask = s < 5

    s[mask] = range(2, 7)
    expected = Series(
        list(range(2, 7)) + list(range(5, 10)),
        dtype=any_signed_int_numpy_dtype,
    )

    tm.assert_series_equal(s, expected)


def test_where_unsafe_float(float_numpy_dtype):
    # [FUNCTION DOC]
    # Parametrized over float dtypes. Assigning a range through a boolean mask on a
    # float series keeps the dtype and writes the values.
    # KNOWN ISSUE: the name says where but the test uses mask-based item assignment.
    s = Series(np.arange(10), dtype=float_numpy_dtype)
    mask = s < 5

    s[mask] = range(2, 7)
    data = list(range(2, 7)) + list(range(5, 10))
    expected = Series(data, dtype=float_numpy_dtype)

    tm.assert_series_equal(s, expected)


@pytest.mark.parametrize(
    "dtype,expected_dtype",
    [
        (np.int8, np.float64),
        (np.int16, np.float64),
        (np.int32, np.float64),
        (np.int64, np.float64),
        (np.float32, np.float32),
        (np.float64, np.float64),
    ],
)
def test_where_unsafe_upcast(dtype, expected_dtype):
    # [FUNCTION DOC]
    # Regression test (gh-9743). Parametrized over integer and float dtypes. When both
    # the source and the expected dtype are floats, assigning float values through the
    # mask works and keeps the dtype; for integer dtypes, assigning non-integral floats
    # raises a type error saying the value is invalid for the dtype.
    # see gh-9743
    s = Series(np.arange(10), dtype=dtype)
    values = [2.5, 3.5, 4.5, 5.5, 6.5]
    mask = s < 5
    if np.dtype(dtype).kind == np.dtype(expected_dtype).kind == "f":
        s[mask] = values
        expected = Series(values + list(range(5, 10)), dtype=expected_dtype)
        tm.assert_series_equal(s, expected)
    else:
        with pytest.raises(TypeError, match="Invalid value"):
            s[mask] = values


def test_where_unsafe():
    # [FUNCTION DOC]
    # Regression test (gh-9731, gh-3235, GH 4667). Assigning non-integral floats through
    # a mask on integers raises a type error; the same on float series works. Range and
    # list assignments with equal length work; value lists of a different length raise a
    # value error about the list-like indexer length. A where on integers with NaN
    # replacement produces floats. Setting None into a float series gives missing.
    # KNOWN ISSUE: the expected series built right after the first mask is never used
    # and the assignment on the float series is not checked; both are dead code.
    # see gh-9731
    s = Series(np.arange(10), dtype="int64")
    values = [2.5, 3.5, 4.5, 5.5]

    mask = s > 5
    expected = Series(list(range(6)) + values, dtype="float64")

    with pytest.raises(TypeError, match="Invalid value"):
        s[mask] = values
    s = s.astype("float64")
    s[mask] = values

    # see gh-3235
    s = Series(np.arange(10), dtype="int64")
    mask = s < 5
    s[mask] = range(2, 7)
    expected = Series(list(range(2, 7)) + list(range(5, 10)), dtype="int64")
    tm.assert_series_equal(s, expected)
    assert s.dtype == expected.dtype

    s = Series(np.arange(10), dtype="int64")
    mask = s > 5
    s[mask] = [0] * 4
    expected = Series([0, 1, 2, 3, 4, 5] + [0] * 4, dtype="int64")
    tm.assert_series_equal(s, expected)

    s = Series(np.arange(10))
    mask = s > 5

    msg = "cannot set using a list-like indexer with a different length than the value"
    with pytest.raises(ValueError, match=msg):
        s[mask] = [5, 4, 3, 2, 1]

    with pytest.raises(ValueError, match=msg):
        s[mask] = [0] * 5

    # dtype changes
    s = Series([1, 2, 3, 4])
    result = s.where(s > 2, np.nan)
    expected = Series([np.nan, np.nan, 3, 4])
    tm.assert_series_equal(result, expected)

    # GH 4667
    # setting with None changes dtype
    s = Series(range(10)).astype(float)
    s[8] = None
    result = s[8]
    assert isna(result)

    s = Series(range(10)).astype(float)
    s[s > 8] = None
    result = s[isna(s)]
    expected = Series(np.nan, index=[9])
    tm.assert_series_equal(result, expected)


def test_where():
    # [FUNCTION DOC]
    # Compares where against boolean selection plus drop of missing; with a replacement
    # series equals absolute value; the result is a new object of the same shape; a
    # condition series shorter than the data is aligned (missing positions become false,
    # giving NaN) and with a replacement series the unaligned positions take the
    # replacement.
    s = Series(np.random.default_rng(2).standard_normal(5))
    cond = s > 0

    rs = s.where(cond).dropna()
    rs2 = s[cond]
    tm.assert_series_equal(rs, rs2)

    rs = s.where(cond, -s)
    tm.assert_series_equal(rs, s.abs())

    rs = s.where(cond)
    assert s.shape == rs.shape
    assert rs is not s

    # test alignment
    cond = Series([True, False, False, True, False], index=s.index)
    s2 = -(s.abs())

    expected = s2[cond].reindex(s2.index[:3]).reindex(s2.index)
    rs = s2.where(cond[:3])
    tm.assert_series_equal(rs, expected)

    expected = s2.abs()
    expected.iloc[0] = s2[0]
    rs = s2.where(cond[:3], -s2)
    tm.assert_series_equal(rs, expected)


def test_where_error():
    # [FUNCTION DOC]
    # Verifies a scalar or mismatched-length array condition raises a value error about
    # the shape. Also (GH 2745) setting through a boolean list works, and a boolean list
    # with a value list of wrong length or an empty value list raises a value error
    # about the list-like indexer length.
    s = Series(np.random.default_rng(2).standard_normal(5))
    cond = s > 0

    msg = "Array conditional must be same shape as self"
    with pytest.raises(ValueError, match=msg):
        s.where(1)
    with pytest.raises(ValueError, match=msg):
        s.where(cond[:3].values, -s)

    # GH 2745
    s = Series([1, 2])
    s[[True, False]] = [0, 1]
    expected = Series([0, 2])
    tm.assert_series_equal(s, expected)

    # failures
    msg = "cannot set using a list-like indexer with a different length than the value"
    with pytest.raises(ValueError, match=msg):
        s[[True, False]] = [0, 2, 3]

    with pytest.raises(ValueError, match=msg):
        s[[True, False]] = []


@pytest.mark.parametrize("klass", [list, tuple, np.array, Series])
def test_where_array_like(klass):
    # [FUNCTION DOC]
    # Regression test (gh-15414). Parametrized over list, tuple, numpy array and Series
    # conditions of booleans; where gives NaN at false positions.
    # see gh-15414
    s = Series([1, 2, 3])
    cond = [False, True, True]
    expected = Series([np.nan, 2, 3])

    result = s.where(klass(cond))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "cond",
    [
        [1, 0, 1],
        Series([2, 5, 7]),
        ["True", "False", "True"],
        [Timestamp("2017-01-01"), pd.NaT, Timestamp("2017-01-02")],
    ],
)
def test_where_invalid_input(cond):
    # [FUNCTION DOC]
    # Regression test (gh-15414). Parametrized over non-boolean conditions (integers, an
    # integer series, strings, timestamps). where raises a type error saying a boolean
    # array is expected; a one-element list raises a value error about the shape.
    # see gh-15414: only boolean arrays accepted
    s = Series([1, 2, 3])
    msg = "Boolean array expected for the condition"

    with pytest.raises(TypeError, match=msg):
        s.where(cond)

    msg = "Array conditional must be same shape as self"
    with pytest.raises(ValueError, match=msg):
        s.where([True])


def test_where_ndframe_align():
    # [FUNCTION DOC]
    # Verifies conditions: a shorter list or longer array raises a value error about the
    # shape, but wrapping the same condition in a series aligns on the index (missing
    # become false) and gives NaN at the other positions.
    msg = "Array conditional must be same shape as self"
    s = Series([1, 2, 3])

    cond = [True]
    with pytest.raises(ValueError, match=msg):
        s.where(cond)

    expected = Series([1, np.nan, np.nan])

    out = s.where(Series(cond))
    tm.assert_series_equal(out, expected)

    cond = np.array([False, True, False, True])
    with pytest.raises(ValueError, match=msg):
        s.where(cond)

    expected = Series([np.nan, 2, np.nan])

    out = s.where(Series(cond))
    tm.assert_series_equal(out, expected)


def test_where_setitem_invalid():
    # [FUNCTION DOC]
    # Regression test (GH 2702). Assigning list values of the wrong length through a
    # slice (also with a step and negative slice) or a list indexer raises a value error
    # whose message names the indexer kind (slice or list-like); correct length
    # assignments work on object series; a scalar position can hold a list.
    # GH 2702
    # make sure correct exceptions are raised on invalid list assignment

    msg = lambda x: (
        f"cannot set using a {x} indexer with a different length than the value"
    )
    # slice
    s = Series(list("abc"), dtype=object)

    with pytest.raises(ValueError, match=msg("slice")):
        s[0:3] = list(range(27))

    s[0:3] = list(range(3))
    expected = Series([0, 1, 2])
    tm.assert_series_equal(s.astype(np.int64), expected)

    # slice with step
    s = Series(list("abcdef"), dtype=object)

    with pytest.raises(ValueError, match=msg("slice")):
        s[0:4:2] = list(range(27))

    s = Series(list("abcdef"), dtype=object)
    s[0:4:2] = list(range(2))
    expected = Series([0, "b", 1, "d", "e", "f"])
    tm.assert_series_equal(s, expected)

    # neg slices
    s = Series(list("abcdef"), dtype=object)

    with pytest.raises(ValueError, match=msg("slice")):
        s[:-1] = list(range(27))

    s[-3:-1] = list(range(2))
    expected = Series(["a", "b", "c", 0, 1, "f"])
    tm.assert_series_equal(s, expected)

    # list
    s = Series(list("abc"), dtype=object)

    with pytest.raises(ValueError, match=msg("list-like")):
        s[[0, 1, 2]] = list(range(27))

    s = Series(list("abc"), dtype=object)

    with pytest.raises(ValueError, match=msg("list-like")):
        s[[0, 1, 2]] = list(range(2))

    # scalar
    s = Series(list("abc"), dtype=object)
    s[0] = list(range(10))
    expected = Series([list(range(10)), "b", "c"])
    tm.assert_series_equal(s, expected)


@pytest.mark.parametrize("size", range(2, 6))
@pytest.mark.parametrize(
    "mask", [[True, False, False, False, False], [True, False], [False]]
)
@pytest.mark.parametrize(
    "item", [2.0, np.nan, np.finfo(float).max, np.finfo(float).min]
)
@pytest.mark.parametrize("box", [np.array, list, tuple])
def test_broadcast(size, mask, item, box):
    # [FUNCTION DOC]
    # Regression test (GH 8801, GH 4195). Parametrized over series size, boolean mask
    # pattern (resized to the size), a float item (including NaN and the float extremes)
    # and the container for the replacement (numpy array, list, tuple). Boolean-mask
    # assignment of the scalar, where with the inverted mask and a one-element
    # container, and mask with the mask and a one-element container all give the
    # expected series.
    # GH#8801, GH#4195
    selection = np.resize(mask, size)

    data = np.arange(size, dtype=float)

    # Construct the expected series by taking the source
    # data or item based on the selection
    expected = Series(
        [item if use_item else data[i] for i, use_item in enumerate(selection)]
    )

    s = Series(data)

    s[selection] = item
    tm.assert_series_equal(s, expected)

    s = Series(data)
    result = s.where(~selection, box([item]))
    tm.assert_series_equal(result, expected)

    s = Series(data)
    result = s.mask(selection, box([item]))
    tm.assert_series_equal(result, expected)


def test_where_inplace():
    # [FUNCTION DOC]
    # Verifies in-place where returns the series itself; with default replacement the
    # missing-dropped result equals the selection and the full result equals the
    # non-in-place; with a replacement series the in-place result equals the
    # non-in-place result.
    s = Series(np.random.default_rng(2).standard_normal(5))
    cond = s > 0

    rs = s.copy()

    result = rs.where(cond, inplace=True)
    assert result is rs
    tm.assert_series_equal(rs.dropna(), s[cond])
    tm.assert_series_equal(rs, s.where(cond))

    rs = s.copy()
    result = rs.where(cond, -s, inplace=True)
    assert result is rs
    tm.assert_series_equal(rs, s.where(cond, -s))


def test_where_dups():
    # [FUNCTION DOC]
    # Regression test (GH 4550, GH 4548). On a duplicated index (concatenation of two
    # series), where gives NaN where the condition is false; setting by condition works
    # on all duplicates; augmented assignment with a condition adds ten at matching
    # positions.
    # GH 4550
    # where crashes with dups in index
    s1 = Series(list(range(3)))
    s2 = Series(list(range(3)))
    comb = pd.concat([s1, s2])
    result = comb.where(comb < 2)
    expected = Series([0, 1, np.nan, 0, 1, np.nan], index=[0, 1, 2, 0, 1, 2])
    tm.assert_series_equal(result, expected)

    # GH 4548
    # inplace updating not working with dups
    comb[comb < 1] = 5
    expected = Series([5, 1, 2, 5, 1, 2], index=[0, 1, 2, 0, 1, 2])
    tm.assert_series_equal(comb, expected)

    comb[comb < 2] += 10
    expected = Series([5, 11, 2, 5, 11, 2], index=[0, 1, 2, 0, 1, 2])
    tm.assert_series_equal(comb, expected)


def test_where_numeric_with_string():
    # [FUNCTION DOC]
    # Regression test (GH 9280). Replacing with the string X (scalar, list or array)
    # turns an integer series into an object series where only the replaced element is a
    # string and the others remain integers.
    # GH 9280
    s = Series([1, 2, 3])
    w = s.where(s > 1, "X")

    assert not is_integer(w[0])
    assert is_integer(w[1])
    assert is_integer(w[2])
    assert isinstance(w[0], str)
    assert w.dtype == "object"

    w = s.where(s > 1, ["X", "Y", "Z"])
    assert not is_integer(w[0])
    assert is_integer(w[1])
    assert is_integer(w[2])
    assert isinstance(w[0], str)
    assert w.dtype == "object"

    w = s.where(s > 1, np.array(["X", "Y", "Z"]))
    assert not is_integer(w[0])
    assert is_integer(w[1])
    assert is_integer(w[2])
    assert isinstance(w[0], str)
    assert w.dtype == "object"


def test_where_datetimetz():
    # [FUNCTION DOC]
    # Regression test (GH 15701). Where on a UTC datetime series with a false condition
    # gives NaT in that position and keeps the dtype.
    # GH 15701
    timestamps = ["2016-12-31 12:00:04+00:00", "2016-12-31 12:00:04.010000+00:00"]
    ser = Series([Timestamp(t) for t in timestamps], dtype="datetime64[ns, UTC]")
    rs = ser.where(Series([False, True]))
    expected = Series([pd.NaT, ser[1]], dtype="datetime64[ns, UTC]")
    tm.assert_series_equal(rs, expected)


def test_where_sparse():
    # [FUNCTION DOC]
    # Regression test (GH 17198). Where with replacement zero on a sparse series gives
    # the expected sparse series, without an attribute error for the sparse index.
    # GH#17198 make sure we dont get an AttributeError for sp_index
    ser = Series(pd.arrays.SparseArray([1, 2]))
    result = ser.where(ser >= 2, 0)
    expected = Series(pd.arrays.SparseArray([0, 2]))
    tm.assert_series_equal(result, expected)


def test_where_empty_series_and_empty_cond_having_non_bool_dtypes():
    # [FUNCTION DOC]
    # Regression test (GH 34592). An empty float series with an empty list condition
    # returns an equal empty series.
    # https://github.com/pandas-dev/pandas/issues/34592
    ser = Series([], dtype=float)
    result = ser.where([])
    tm.assert_series_equal(result, ser)


def test_where_categorical(frame_or_series):
    # [FUNCTION DOC]
    # Regression test (GH 18888). Parametrized over series and frame. Masking out one
    # category value gives a categorical with that element missing and unchanged
    # categories.
    # https://github.com/pandas-dev/pandas/issues/18888
    exp = frame_or_series(
        pd.Categorical(["A", "A", "B", "B", np.nan], categories=["A", "B", "C"]),
        dtype="category",
    )
    df = frame_or_series(["A", "A", "B", "B", "C"], dtype="category")
    res = df.where(df != "C")
    tm.assert_equal(exp, res)


def test_where_datetimelike_categorical(tz_naive_fixture):
    # [FUNCTION DOC]
    # Regression test (GH 37682). Over the naive/timezone fixture, where with a
    # categorical replacement and a boolean mask gives the expected datetimes in four
    # containers: datetime index, datetime array (private method), series and frame.
    # GH#37682
    tz = tz_naive_fixture

    dr = date_range("2001-01-01", periods=3, tz=tz)._with_freq(None)
    lvals = pd.DatetimeIndex([dr[0], dr[1], pd.NaT])
    rvals = pd.Categorical([dr[0], pd.NaT, dr[2]])

    mask = np.array([True, True, False])

    # DatetimeIndex.where
    res = lvals.where(mask, rvals)
    tm.assert_index_equal(res, dr)

    # DatetimeArray.where
    res = lvals._data._where(mask, rvals)
    tm.assert_datetime_array_equal(res, dr._data)

    # Series.where
    res = Series(lvals).where(mask, rvals)
    tm.assert_series_equal(res, Series(dr))

    # DataFrame.where
    res = pd.DataFrame(lvals).where(mask[:, None], pd.DataFrame(rvals))

    tm.assert_frame_equal(res, pd.DataFrame(dr))
