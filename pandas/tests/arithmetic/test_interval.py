# [MODULE DOC]
# Tests for comparison operators (equality and inequality) on interval arrays, interval indexes, and series of intervals against scalars, other intervals, and list-likes of many types.
# Defines fixtures that build interval arrays from several endpoint dtypes (integer, float, timedelta, naive and tz-aware datetime, with missing values where possible) and a set of constructors for native interval containers, plus an elementwise reference implementation used to compute expectations. Relies on shared fixtures (closed types, null values, box-with-array) and the upcast-box helper from the arithmetic common module.
import operator

import numpy as np
import pytest

from pandas.core.dtypes.common import is_list_like

import pandas as pd
from pandas import (
    Categorical,
    Index,
    Interval,
    IntervalIndex,
    Period,
    Series,
    Timedelta,
    Timestamp,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.arrays import (
    BooleanArray,
    IntervalArray,
)
from pandas.tests.arithmetic.common import get_upcast_box


@pytest.fixture(
    params=[
        (Index([0, 2, 4, 4]), Index([1, 3, 5, 8])),
        (Index([0.0, 1.0, 2.0, np.nan]), Index([1.0, 2.0, 3.0, np.nan])),
        (
            timedelta_range("0 days", periods=3).insert(3, pd.NaT),
            timedelta_range("1 day", periods=3).insert(3, pd.NaT),
        ),
        (
            date_range("20170101", periods=3).insert(3, pd.NaT),
            date_range("20170102", periods=3).insert(3, pd.NaT),
        ),
        (
            date_range("20170101", periods=3, tz="US/Eastern").insert(3, pd.NaT),
            date_range("20170102", periods=3, tz="US/Eastern").insert(3, pd.NaT),
        ),
    ],
    ids=lambda x: str(x[0].dtype),
)
def left_right_dtypes(request):
    # [FUNCTION DOC]
    # Parametrized fixture (ids are the left endpoint dtype names) returning a pair of left and right endpoint indexes of length four: integers, floats with NaN, timedeltas with NaT, naive datetimes with NaT, and tz-aware datetimes with NaT. The second is one step greater than the first.
    """
    Fixture for building an IntervalArray from various dtypes
    """
    return request.param


@pytest.fixture
def interval_array(left_right_dtypes):
    # [FUNCTION DOC]
    # Fixture building an interval array from the left/right endpoint pair of the parametrized dtype fixture, closed on the right; contains a missing interval wherever the endpoints include a missing value.
    """
    Fixture to generate an IntervalArray of various dtypes containing NA if possible
    """
    left, right = left_right_dtypes
    return IntervalArray.from_arrays(left, right)


def create_categorical_intervals(left, right, closed="right"):
    # [FUNCTION DOC]
    # HELPER: constructs a categorical whose categories are intervals built from left and right endpoints and the closed side (default right). Used as an interval constructor in the comparison fixtures. Returns a categorical.
    return Categorical(IntervalIndex.from_arrays(left, right, closed))


def create_series_intervals(left, right, closed="right"):
    # [FUNCTION DOC]
    # HELPER: constructs a series with interval extension dtype from left and right endpoints and the closed side (default right). Returns a series.
    return Series(IntervalArray.from_arrays(left, right, closed))


def create_series_categorical_intervals(left, right, closed="right"):
    # [FUNCTION DOC]
    # HELPER: constructs a series of categorical dtype whose categories are intervals from left and right endpoints and the closed side (default right). Returns a series.
    return Series(Categorical(IntervalIndex.from_arrays(left, right, closed)))


class TestComparison:
    # [CLASS DOC]
    # Test class for equality and inequality comparisons of interval arrays with scalars and list-likes; holds fixtures for the operator and for the right-hand-side constructors, a reference helper, and the test methods.
    @pytest.fixture(params=[operator.eq, operator.ne])
    def op(self, request):
        # [FUNCTION DOC]
        # Class fixture parametrized over the equality and not-equal operator functions; returns the chosen operator.
        return request.param

    @pytest.fixture(
        params=[
            IntervalArray.from_arrays,
            IntervalIndex.from_arrays,
            create_categorical_intervals,
            create_series_intervals,
            create_series_categorical_intervals,
        ],
        ids=[
            "IntervalArray",
            "IntervalIndex",
            "Categorical[Interval]",
            "Series[Interval]",
            "Series[Categorical[Interval]]",
        ],
    )
    def interval_constructor(self, request):
        # [FUNCTION DOC]
        # Class fixture parametrized over five interval-container constructors (interval array, interval index, categorical of intervals, series of intervals, series of categorical intervals); returns the chosen constructor, taking left, right and optional closed arguments. To be used to build the right-hand operand of comparisons.
        """
        Fixture for all pandas native interval constructors.
        To be used as the LHS of IntervalArray comparisons.
        """
        return request.param

    def elementwise_comparison(self, op, interval_array, other):
        # [FUNCTION DOC]
        # HELPER: reference implementation of the expected result of a comparison. Takes the operator, the interval array, and another operand; if the operand is not list-like it is repeated to the array's length. Applies the operator to each pair of elements (pairing strictly by length) and returns a boolean NumPy array, or a series carrying the operand's index when the operand is a series.
        """
        Helper that performs elementwise comparisons between `array` and `other`
        """
        other = other if is_list_like(other) else [other] * len(interval_array)
        expected = np.array(
            [op(x, y) for x, y in zip(interval_array, other, strict=True)]
        )
        if isinstance(other, Series):
            return Series(expected, index=other.index)
        return expected

    def test_compare_scalar_interval(self, op, interval_array):
        # [FUNCTION DOC]
        # Verifies comparing an interval array to a scalar interval: first with the array's own first element, then with an interval that matches only one endpoint of the first element; result must match the elementwise reference as a NumPy array.
        # matches first interval
        other = interval_array[0]
        result = op(interval_array, other)
        expected = self.elementwise_comparison(op, interval_array, other)
        tm.assert_numpy_array_equal(result, expected)

        # matches on a single endpoint but not both
        other = Interval(interval_array.left[0], interval_array.right[1])
        result = op(interval_array, other)
        expected = self.elementwise_comparison(op, interval_array, other)
        tm.assert_numpy_array_equal(result, expected)

    def test_compare_scalar_interval_mixed_closed(self, op, closed, other_closed):
        # [FUNCTION DOC]
        # Uses shared fixtures for the closed side of the array and of the scalar. Verifies comparing a two-element interval array with a scalar interval whose closed side may differ matches the elementwise reference.
        interval_array = IntervalArray.from_arrays(range(2), range(1, 3), closed=closed)
        other = Interval(0, 1, closed=other_closed)

        result = op(interval_array, other)
        expected = self.elementwise_comparison(op, interval_array, other)
        tm.assert_numpy_array_equal(result, expected)

    def test_compare_scalar_na(self, op, interval_array, nulls_fixture, box_with_array):
        # [FUNCTION DOC]
        # Uses fixtures for the null scalar and the container box. Wraps the interval array in the box and compares it against a null scalar in both operand orders. The expected value is the elementwise reference, except for the NA singleton where it is a fully masked boolean extension array; it is boxed in the upcast box unless the box is an index and the null is NA. Results must equal expected.
        box = box_with_array
        obj = tm.box_expected(interval_array, box)
        result = op(obj, nulls_fixture)

        if nulls_fixture is pd.NA:
            # GH#31882
            exp = np.ones(interval_array.shape, dtype=bool)
            expected = BooleanArray(exp, exp)
        else:
            expected = self.elementwise_comparison(op, interval_array, nulls_fixture)

        if not (box is Index and nulls_fixture is pd.NA):
            # don't cast expected from BooleanArray to ndarray[object]
            xbox = get_upcast_box(obj, nulls_fixture, True)
            expected = tm.box_expected(expected, xbox)

        tm.assert_equal(result, expected)

        rev = op(nulls_fixture, obj)
        tm.assert_equal(rev, expected)

    @pytest.mark.parametrize(
        "other",
        [
            0,
            1.0,
            True,
            "foo",
            Timestamp("2017-01-01"),
            Timestamp("2017-01-01", tz="US/Eastern"),
            Timedelta("0 days"),
            Period("2017-01-01", "D"),
        ],
    )
    def test_compare_scalar_other(self, op, interval_array, other):
        # [FUNCTION DOC]
        # Parametrized over scalars of unrelated types: integers, floats, bools, string, naive and tz-aware timestamps, timedelta, and period. Verifies comparing the interval array with each yields the elementwise reference result as a NumPy array (so such comparisons do not raise).
        result = op(interval_array, other)
        expected = self.elementwise_comparison(op, interval_array, other)
        tm.assert_numpy_array_equal(result, expected)

    def test_compare_list_like_interval(self, op, interval_array, interval_constructor):
        # [FUNCTION DOC]
        # Parametrized through the constructor fixture. Builds a right operand from the array's endpoints, from reversed endpoints, and from all-NaN endpoints and verifies the comparison equals the elementwise reference in each case.
        # same endpoints
        other = interval_constructor(interval_array.left, interval_array.right)
        result = op(interval_array, other)
        expected = self.elementwise_comparison(op, interval_array, other)
        tm.assert_equal(result, expected)

        # different endpoints
        other = interval_constructor(
            interval_array.left[::-1], interval_array.right[::-1]
        )
        result = op(interval_array, other)
        expected = self.elementwise_comparison(op, interval_array, other)
        tm.assert_equal(result, expected)

        # all nan endpoints
        other = interval_constructor([np.nan] * 4, [np.nan] * 4)
        result = op(interval_array, other)
        expected = self.elementwise_comparison(op, interval_array, other)
        tm.assert_equal(result, expected)

    def test_compare_list_like_interval_mixed_closed(
        self, op, interval_constructor, closed, other_closed
    ):
        # [FUNCTION DOC]
        # Uses the closed-side fixtures and the constructor fixture. Verifies comparing a two-element interval array with a same-sized container of intervals whose closed side may differ matches the elementwise reference.
        interval_array = IntervalArray.from_arrays(range(2), range(1, 3), closed=closed)
        other = interval_constructor(range(2), range(1, 3), closed=other_closed)

        result = op(interval_array, other)
        expected = self.elementwise_comparison(op, interval_array, other)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "other",
        [
            (
                Interval(0, 1),
                Interval(Timedelta("1 day"), Timedelta("2 days")),
                Interval(4, 5, "both"),
                Interval(10, 20, "neither"),
            ),
            (0, 1.5, Timestamp("20170103"), np.nan),
            (
                Timestamp("20170102", tz="US/Eastern"),
                Timedelta("2 days"),
                "baz",
                pd.NaT,
            ),
        ],
    )
    def test_compare_list_like_object(self, op, interval_array, other):
        # [FUNCTION DOC]
        # Parametrized over three length-four tuples of mixed objects (intervals of different kinds, mixed scalars including NaN, tz-aware timestamp with timedelta, string, and NaT). Verifies comparison with such object-like operands matches the elementwise reference, as a NumPy array.
        result = op(interval_array, other)
        expected = self.elementwise_comparison(op, interval_array, other)
        tm.assert_numpy_array_equal(result, expected)

    def test_compare_list_like_nan(self, op, interval_array, nulls_fixture):
        # [FUNCTION DOC]
        # Uses the null-values fixture. Verifies comparison with a list of four identical null values matches the elementwise reference.
        other = [nulls_fixture] * 4
        result = op(interval_array, other)
        expected = self.elementwise_comparison(op, interval_array, other)

        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "other",
        [
            np.arange(4, dtype="int64"),
            np.arange(4, dtype="float64"),
            date_range("2017-01-01", periods=4),
            date_range("2017-01-01", periods=4, tz="US/Eastern"),
            timedelta_range("0 days", periods=4),
            period_range("2017-01-01", periods=4, freq="D"),
            Categorical(list("abab")),
            Categorical(date_range("2017-01-01", periods=4)),
            pd.array(list("abcd")),
            pd.array(["foo", 3.14, None, object()], dtype=object),
        ],
        ids=lambda x: str(x.dtype),
    )
    def test_compare_list_like_other(self, op, interval_array, other):
        # [FUNCTION DOC]
        # Parametrized over many list-like operands of length four: integer and float arrays, datetime ranges (naive and tz-aware), timedelta and period ranges, categoricals of strings and dates, a string extension array, and an object array with mixed values. Verifies comparison matches the elementwise reference as a NumPy array.
        result = op(interval_array, other)
        expected = self.elementwise_comparison(op, interval_array, other)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("length", [1, 3, 5])
    @pytest.mark.parametrize("other_constructor", [IntervalArray, list])
    def test_compare_length_mismatch_errors(self, op, other_constructor, length):
        # [FUNCTION DOC]
        # Parametrized over operand type (interval array or list) and lengths one, three and five. Verifies comparing a length-four interval array to an operand of a different length raises a value error saying lengths must match to compare.
        interval_array = IntervalArray.from_arrays(range(4), range(1, 5))
        other = other_constructor([Interval(0, 1)] * length)
        with pytest.raises(ValueError, match="Lengths must match to compare"):
            op(interval_array, other)

    @pytest.mark.parametrize(
        "constructor, expected_type, assert_func",
        [
            (IntervalIndex, np.array, tm.assert_numpy_array_equal),
            (Series, Series, tm.assert_series_equal),
        ],
    )
    def test_index_series_compat(self, op, constructor, expected_type, assert_func):
        # [FUNCTION DOC]
        # Parametrized over (interval index, NumPy array, NumPy array assert) and (series, series, series assert) triples. Verifies that an interval index or series built from breaks compares with scalars (an interval, an integer), an interval array, and a mixed list exactly as the elementwise reference does, delegating to the interval array comparison.
        # IntervalIndex/Series that rely on IntervalArray for comparisons
        breaks = range(4)
        index = constructor(IntervalIndex.from_breaks(breaks))

        # scalar comparisons
        other = index[0]
        result = op(index, other)
        expected = expected_type(self.elementwise_comparison(op, index, other))
        assert_func(result, expected)

        other = breaks[0]
        result = op(index, other)
        expected = expected_type(self.elementwise_comparison(op, index, other))
        assert_func(result, expected)

        # list-like comparisons
        other = IntervalArray.from_breaks(breaks)
        result = op(index, other)
        expected = expected_type(self.elementwise_comparison(op, index, other))
        assert_func(result, expected)

        other = [index[0], breaks[0], "foo"]
        result = op(index, other)
        expected = expected_type(self.elementwise_comparison(op, index, other))
        assert_func(result, expected)

    @pytest.mark.parametrize("scalars", ["a", False, 1, 1.0, None])
    def test_comparison_operations(self, scalars):
        # [FUNCTION DOC]
        # Parametrized over scalars (string, False, integers, float, None). Verifies that equality of a series of two intervals with such a scalar yields an all-false series (regression test for a linked issue).
        # GH #28981
        expected = Series([False, False])
        s = Series([Interval(0, 1), Interval(1, 2)], dtype="interval")
        result = s == scalars
        tm.assert_series_equal(result, expected)
