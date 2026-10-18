# [MODULE DOC]
# Tests for IntervalArray: the is_empty attribute, closed-side changes, where, shift, unique (with negatives and infinities),
# setitem validation (NA and mismatched closed), and min/max reductions. Defines a fixture of left/right endpoint pairs of several
# dtypes. Uses fixtures closed, other_closed and index_or_series_or_array from the shared conftest. No import-time side effects.

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Index,
    Interval,
    IntervalIndex,
    Timedelta,
    Timestamp,
    date_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.arrays import IntervalArray


@pytest.fixture(
    params=[
        (Index([0, 2, 4]), Index([1, 3, 5])),
        (Index([0.0, 1.0, 2.0]), Index([1.0, 2.0, 3.0])),
        (timedelta_range("0 days", periods=3), timedelta_range("1 day", periods=3)),
        (date_range("20170101", periods=3), date_range("20170102", periods=3)),
        (
            date_range("20170101", periods=3, tz="US/Eastern"),
            date_range("20170102", periods=3, tz="US/Eastern"),
        ),
    ],
    ids=lambda x: str(x[0].dtype),
)
def left_right_dtypes(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over five pairs of endpoint indexes (integer, float, timedelta, naive datetime, tz-aware datetime) used to build interval arrays of several subtypes.
    """
    Fixture for building an IntervalArray from various dtypes
    """
    return request.param


class TestAttributes:
    # [CLASS DOC]
    # Groups tests of interval array attributes. Stateless.
    @pytest.mark.parametrize(
        "left, right",
        [
            (0, 1),
            (Timedelta("0 days"), Timedelta("1 day")),
            (Timestamp("2018-01-01"), Timestamp("2018-01-02")),
            (
                Timestamp("2018-01-01", tz="US/Eastern"),
                Timestamp("2018-01-02", tz="US/Eastern"),
            ),
        ],
    )
    @pytest.mark.parametrize("constructor", [IntervalArray, IntervalIndex])
    def test_is_empty(self, constructor, left, right, closed):
        # [FUNCTION DOC]
        # Parametrized over endpoint pairs (integer, timedelta, naive and tz-aware timestamp) and over IntervalArray and IntervalIndex, with the closed fixture. From tuples including a zero-length
        # interval, a normal one and NaN, is_empty is true only for the zero-length interval when the closed side is not both, false otherwise.
        # GH27219
        tuples = [(left, left), (left, right), np.nan]
        expected = np.array([closed != "both", False, False])
        result = constructor.from_tuples(tuples, closed=closed).is_empty
        tm.assert_numpy_array_equal(result, expected)


class TestMethods:
    # [CLASS DOC]
    # Groups tests of interval array methods. Stateless.
    def test_set_closed(self, closed, other_closed):
        # [FUNCTION DOC]
        # Using closed and other_closed fixtures. Changing the closed side returns an array equal to one built with that closed side from the same breaks.
        # GH 21670
        array = IntervalArray.from_breaks(range(10), closed=closed)
        result = array.set_closed(other_closed)
        expected = IntervalArray.from_breaks(range(10), closed=other_closed)
        tm.assert_extension_array_equal(result, expected)

    @pytest.mark.parametrize(
        "other",
        [
            Interval(0, 1, closed="right"),
            IntervalArray.from_breaks([1, 2, 3, 4], closed="right"),
        ],
    )
    def test_where_raises(self, other):
        # [FUNCTION DOC]
        # Parametrized over a right-closed interval and a right-closed array. The array-level private where method raises a value error when the replacement closed side differs
        # from the array (left); the Series where coerces to object and gives the same result as doing so explicitly.
        # GH#45768 The IntervalArray methods raises; the Series method coerces
        ser = pd.Series(IntervalArray.from_breaks([1, 2, 3, 4], closed="left"))
        mask = np.array([True, False, True])
        match = "'value.closed' is 'right', expected 'left'."
        with pytest.raises(ValueError, match=match):
            ser.array._where(mask, other)

        res = ser.where(mask, other=other)
        expected = ser.astype(object).where(mask, other)
        tm.assert_series_equal(res, expected)

    def test_shift(self):
        # [FUNCTION DOC]
        # Shifting by default inserts a missing interval and converts integer endpoints to float; a fill value of NaT raises a type error saying only intervals and NA can be inserted.
        # https://github.com/pandas-dev/pandas/issues/31495, GH#22428, GH#31502
        a = IntervalArray.from_breaks([1, 2, 3])
        result = a.shift()
        # int -> float
        expected = IntervalArray.from_tuples([(np.nan, np.nan), (1.0, 2.0)])
        tm.assert_interval_array_equal(result, expected)

        msg = "can only insert Interval objects and NA into an IntervalArray"
        with pytest.raises(TypeError, match=msg):
            a.shift(1, fill_value=pd.NaT)

    def test_shift_datetime(self):
        # [FUNCTION DOC]
        # Datetime interval array shifts match taking with fill (forward by two, backward by one); a numpy timedelta NaT fill value raises a type error.
        # GH#31502, GH#31504
        a = IntervalArray.from_breaks(date_range("2000", periods=4))
        result = a.shift(2)
        expected = a.take([-1, -1, 0], allow_fill=True)
        tm.assert_interval_array_equal(result, expected)

        result = a.shift(-1)
        expected = a.take([1, 2, -1], allow_fill=True)
        tm.assert_interval_array_equal(result, expected)

        msg = "can only insert Interval objects and NA into an IntervalArray"
        with pytest.raises(TypeError, match=msg):
            a.shift(1, fill_value=np.timedelta64("NaT", "ns"))

    def test_unique_with_negatives(self):
        # [FUNCTION DOC]
        # Unique on interval indexes with positive, negative and mixed endpoints keeps first occurrences in order (regression test).
        # GH#61917
        idx_pos = IntervalIndex.from_tuples(
            [(3, 4), (3, 4), (2, 3), (2, 3), (1, 2), (1, 2)]
        )
        result = idx_pos.unique()
        expected = IntervalIndex.from_tuples([(3, 4), (2, 3), (1, 2)])
        tm.assert_index_equal(result, expected)

        idx_neg = IntervalIndex.from_tuples(
            [(-4, -3), (-4, -3), (-3, -2), (-3, -2), (-2, -1), (-2, -1)]
        )
        result = idx_neg.unique()
        expected = IntervalIndex.from_tuples([(-4, -3), (-3, -2), (-2, -1)])
        tm.assert_index_equal(result, expected)

        idx_mix = IntervalIndex.from_tuples(
            [(1, 2), (0, 1), (-1, 0), (-2, -1), (-3, -2), (-3, -2)]
        )
        result = idx_mix.unique()
        expected = IntervalIndex.from_tuples(
            [(1, 2), (0, 1), (-1, 0), (-2, -1), (-3, -2)]
        )
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "data",
        [
            [Interval(-np.inf, 0), Interval(-np.inf, 1)],
            [Interval(0, np.inf), Interval(1, np.inf)],
        ],
    )
    def test_unique_with_infinty(self, data):
        # [FUNCTION DOC]
        # Parametrized over intervals with negative-infinite and positive-infinite endpoints. Unique returns the array unchanged, nunique is two and drop_duplicates returns the same Series.
        # https://github.com/pandas-dev/pandas/issues/63218
        s = pd.Series(data)
        tm.assert_interval_array_equal(s.unique(), s.array)
        assert s.nunique() == 2
        tm.assert_series_equal(s.drop_duplicates(), s)


class TestSetitem:
    # [CLASS DOC]
    # Groups tests of setting items on interval arrays. Stateless.
    def test_set_na(self, left_right_dtypes):
        # [FUNCTION DOC]
        # Using left_right_dtypes. Setting NaT into a non-datetimelike array raises a type error; setting NaN into an integer-backed array raises a type error; otherwise setting NaN makes the first
        # interval missing, matching an array built with the NA value at the first position.
        left, right = left_right_dtypes
        result = IntervalArray.from_arrays(left, right, copy=True)

        if result.dtype.subtype.kind not in ["m", "M"]:
            msg = "'value' should be an interval type, got <.*NaTType'> instead."
            with pytest.raises(TypeError, match=msg):
                result[0] = pd.NaT
        if result.dtype.subtype.kind in ["i", "u"]:
            msg = "Cannot set float NaN to integer-backed IntervalArray"
            # GH#45484 TypeError, not ValueError, matches what we get with
            # non-NA un-holdable value.
            with pytest.raises(TypeError, match=msg):
                result[0] = np.nan
            return

        result[0] = np.nan

        expected_left = Index([left._na_value, *list(left[1:])])
        expected_right = Index([right._na_value, *list(right[1:])])
        expected = IntervalArray.from_arrays(expected_left, expected_right)

        tm.assert_extension_array_equal(result, expected)

    def test_setitem_mismatched_closed(self):
        # [FUNCTION DOC]
        # Assigning intervals with a different closed side (single item, slices, empty slice, reversed array, list, object array, categorical) raises a value error; assigning an empty list is a no-op
        # and leaves the array unchanged.
        arr = IntervalArray.from_breaks(range(4))
        orig = arr.copy()
        other = arr.set_closed("both")

        msg = "'value.closed' is 'both', expected 'right'"
        with pytest.raises(ValueError, match=msg):
            arr[0] = other[0]
        with pytest.raises(ValueError, match=msg):
            arr[:1] = other[:1]
        with pytest.raises(ValueError, match=msg):
            arr[:0] = other[:0]
        with pytest.raises(ValueError, match=msg):
            arr[:] = other[::-1]
        with pytest.raises(ValueError, match=msg):
            arr[:] = list(other[::-1])
        with pytest.raises(ValueError, match=msg):
            arr[:] = other[::-1].astype(object)
        with pytest.raises(ValueError, match=msg):
            arr[:] = other[::-1].astype("category")

        # empty list should be no-op
        arr[:0] = []
        tm.assert_interval_array_equal(arr, orig)


class TestReductions:
    # [CLASS DOC]
    # Groups min and max reduction tests for interval arrays. Stateless.
    def test_min_max_invalid_axis(self, left_right_dtypes):
        # [FUNCTION DOC]
        # Using left_right_dtypes. An axis outside the dimensions raises a value error; a non-numeric axis raises a type error about comparison; for both min and max.
        left, right = left_right_dtypes
        arr = IntervalArray.from_arrays(left, right)

        msg = "`axis` must be fewer than the number of dimensions"
        for axis in [-2, 1]:
            with pytest.raises(ValueError, match=msg):
                arr.min(axis=axis)
            with pytest.raises(ValueError, match=msg):
                arr.max(axis=axis)

        msg = "'>=' not supported between"
        with pytest.raises(TypeError, match=msg):
            arr.min(axis="foo")
        with pytest.raises(TypeError, match=msg):
            arr.max(axis="foo")

    def test_min_max(self, left_right_dtypes, index_or_series_or_array):
        # [FUNCTION DOC]
        # Using left_right_dtypes and a container fixture (index, Series or array). On a shuffled monotonic array, min and max (with skipna true or false) return the first and last intervals; with an inserted NaN,
        # skipna false returns NaN while skipping missing returns the same extremes.
        # GH#44746
        left, right = left_right_dtypes
        arr = IntervalArray.from_arrays(left, right)

        # The expected results below are only valid if monotonic
        assert left.is_monotonic_increasing
        assert Index(arr).is_monotonic_increasing

        MIN = arr[0]
        MAX = arr[-1]

        indexer = np.arange(len(arr))
        np.random.default_rng(2).shuffle(indexer)
        arr = arr.take(indexer)

        arr_na = arr.insert(2, np.nan)

        arr = index_or_series_or_array(arr)
        arr_na = index_or_series_or_array(arr_na)

        for skipna in [True, False]:
            res = arr.min(skipna=skipna)
            assert res == MIN
            assert type(res) == type(MIN)

            res = arr.max(skipna=skipna)
            assert res == MAX
            assert type(res) == type(MAX)

        res = arr_na.min(skipna=False)
        assert np.isnan(res)
        res = arr_na.max(skipna=False)
        assert np.isnan(res)

        for kws in [{"skipna": True}, {}]:
            res = arr_na.min(**kws)
            assert res == MIN
            assert type(res) == type(MIN)
            res = arr_na.max(**kws)
            assert res == MAX
            assert type(res) == type(MAX)
