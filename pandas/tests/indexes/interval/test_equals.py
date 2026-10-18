# [MODULE DOC]
# Tests for equality comparison of IntervalIndex objects.
# Uses the closed fixture. Pure tests module.

import numpy as np

from pandas import (
    IntervalIndex,
    date_range,
)


class TestEquals:
    # [CLASS DOC]
    # Groups equality tests.
    def test_equals(self, closed):
        # [FUNCTION DOC]
        # Parametrized over closed side. Verifies an index equals itself and a copy,
        # does not equal an object-dtype version, plain array, list, non-interval
        # values or a datetime index, ignores the name, and does not equal an index with
        # a different closed side.
        expected = IntervalIndex.from_breaks(np.arange(5), closed=closed)
        assert expected.equals(expected)
        assert expected.equals(expected.copy())

        assert not expected.equals(expected.astype(object))
        assert not expected.equals(np.array(expected))
        assert not expected.equals(list(expected))

        assert not expected.equals([1, 2])
        assert not expected.equals(np.array([1, 2]))
        assert not expected.equals(date_range("20130101", periods=2))

        expected_name1 = IntervalIndex.from_breaks(
            np.arange(5), closed=closed, name="foo"
        )
        expected_name2 = IntervalIndex.from_breaks(
            np.arange(5), closed=closed, name="bar"
        )
        assert expected.equals(expected_name1)
        assert expected_name1.equals(expected_name2)

        for other_closed in {"left", "right", "both", "neither"} - {closed}:
            expected_other_closed = IntervalIndex.from_breaks(
                np.arange(5), closed=other_closed
            )
            assert not expected.equals(expected_other_closed)
