"""Tests for Interval-Interval operations, such as overlaps, contains, etc."""

# [MODULE DOC]
# Tests for the overlaps method of interval containers (IntervalArray and IntervalIndex): overlap with a single interval for
# different closed sides and endpoint types, missing values, containers (not implemented) and invalid types. Defines fixtures for the
# container class and for endpoint start/shift pairs. Uses closed and other_closed fixtures from the shared conftest. No import-time side
# effects.

import numpy as np
import pytest

from pandas import (
    Interval,
    IntervalIndex,
    Timedelta,
    Timestamp,
)
import pandas._testing as tm
from pandas.core.arrays import IntervalArray


@pytest.fixture(params=[IntervalArray, IntervalIndex])
def constructor(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over the two interval container classes.
    """
    Fixture for testing both interval container classes.
    """
    return request.param


@pytest.fixture(
    params=[
        (Timedelta("0 days"), Timedelta("1 day")),
        (Timestamp("2018-01-01"), Timedelta("1 day")),
        (0, 1),
    ],
    ids=lambda x: type(x[0]).__name__,
)
def start_shift(request):
    # [FUNCTION DOC]
    # HELPER: fixture giving (start, shift) pairs for timedelta, timestamp and integer endpoints, where adding the shift to start yields an endpoint.
    """
    Fixture for generating intervals of different types from a start value
    and a shift value that can be added to start to generate an endpoint.
    """
    return request.param


class TestOverlaps:
    # [CLASS DOC]
    # Groups overlap tests. Stateless.
    def test_overlaps_interval(self, constructor, start_shift, closed, other_closed):
        # [FUNCTION DOC]
        # Builds six intervals (identical, nested, spanning, partial, adjacent, disjoint) with the closed fixture and tests overlap with an interval having the other_closed side. All but
        # the last are true; the adjacent one is true only if the probe's right side and the container's left side are both closed.
        start, shift = start_shift
        interval = Interval(start, start + 3 * shift, other_closed)

        # intervals: identical, nested, spanning, partial, adjacent, disjoint
        tuples = [
            (start, start + 3 * shift),
            (start + shift, start + 2 * shift),
            (start - shift, start + 4 * shift),
            (start + 2 * shift, start + 4 * shift),
            (start + 3 * shift, start + 4 * shift),
            (start + 4 * shift, start + 5 * shift),
        ]
        interval_container = constructor.from_tuples(tuples, closed)

        adjacent = interval.closed_right and interval_container.closed_left
        expected = np.array([True, True, True, True, adjacent, False])
        result = interval_container.overlaps(interval)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("other_constructor", [IntervalArray, IntervalIndex])
    def test_overlaps_interval_container(self, constructor, other_constructor):
        # [FUNCTION DOC]
        # Parametrized over the container class of the other argument. Passing a whole interval container raises a not-implemented error with an empty message.
        # KNOWN ISSUE: the test body carries a note to change it when the feature is implemented, so it asserts currently unimplemented behavior.
        # TODO: modify this test when implemented
        interval_container = constructor.from_breaks(range(5))
        other_container = other_constructor.from_breaks(range(5))
        with pytest.raises(NotImplementedError, match="^$"):
            interval_container.overlaps(other_container)

    def test_overlaps_na(self, constructor, start_shift):
        # [FUNCTION DOC]
        # Missing intervals are reported as not overlapping; only the first of three intervals overlaps.
        """NA values are marked as False"""
        start, shift = start_shift
        interval = Interval(start, start + shift)

        tuples = [
            (start, start + shift),
            np.nan,
            (start + 2 * shift, start + 3 * shift),
        ]
        interval_container = constructor.from_tuples(tuples)

        expected = np.array([True, False, False])
        result = interval_container.overlaps(interval)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "other",
        [10, True, "foo", Timedelta("1 day"), Timestamp("2018-01-01")],
        ids=lambda x: type(x).__name__,
    )
    def test_overlaps_invalid_type(self, constructor, other):
        # [FUNCTION DOC]
        # Parametrized over an integer, bool, string, timedelta and timestamp. A non-interval argument raises a type error saying it must be Interval-like and naming the type.
        interval_container = constructor.from_breaks(range(5))
        msg = f"`other` must be Interval-like, got {type(other).__name__}"
        with pytest.raises(TypeError, match=msg):
            interval_container.overlaps(other)
