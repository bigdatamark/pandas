# [MODULE DOC]
# Tests for joining IntervalIndex objects with each other and with MultiIndex
# objects containing overlapping intervals (GH-45661). Defines two local fixtures.

import pytest

from pandas import (
    IntervalIndex,
    MultiIndex,
    RangeIndex,
)
import pandas._testing as tm


@pytest.fixture
def range_index():
    # [FUNCTION DOC]
    # HELPER: fixture returning a named range index with three elements, used as the
    # second level of the multi-index in the join tests.
    return RangeIndex(3, name="range_index")


@pytest.fixture
def interval_index():
    # [FUNCTION DOC]
    # HELPER: fixture returning a named interval index of three overlapping float
    # intervals, used for the join tests.
    return IntervalIndex.from_tuples(
        [(0.0, 1.0), (1.0, 2.0), (1.5, 2.5)], name="interval_index"
    )


def test_join_overlapping_in_mi_to_same_intervalindex(range_index, interval_index):
    # [FUNCTION DOC]
    # Verifies joining a multi-index (interval level times range level) with its
    # own interval index returns the multi-index unchanged.
    #  GH-45661
    multi_index = MultiIndex.from_product([interval_index, range_index])
    result = multi_index.join(interval_index)

    tm.assert_index_equal(result, multi_index)


def test_join_overlapping_to_multiindex_with_same_interval(range_index, interval_index):
    # [FUNCTION DOC]
    # Verifies joining the interval index to the multi-index, in the opposite
    # direction, also returns the multi-index.
    #  GH-45661
    multi_index = MultiIndex.from_product([interval_index, range_index])
    result = interval_index.join(multi_index)

    tm.assert_index_equal(result, multi_index)


def test_join_overlapping_interval_to_another_intervalindex(interval_index):
    # [FUNCTION DOC]
    # Verifies joining an interval index with its reversed copy returns the original order.
    #  GH-45661
    flipped_interval_index = interval_index[::-1]
    result = interval_index.join(flipped_interval_index)

    tm.assert_index_equal(result, interval_index)
