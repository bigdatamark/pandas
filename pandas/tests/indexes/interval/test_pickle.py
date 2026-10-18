# [MODULE DOC]
# Tests for pickling IntervalIndex.
# Uses the closed and temp_file fixtures.

from pandas import IntervalIndex
import pandas._testing as tm


class TestPickle:
    # [CLASS DOC]
    # Groups pickle tests.
    def test_pickle_round_trip_closed(self, closed, temp_file):
        # [FUNCTION DOC]
        # Parametrized over closed side. Verifies (issue 35658) an interval index
        # round-trips through pickle, preserving the closed side.
        # https://github.com/pandas-dev/pandas/issues/35658
        idx = IntervalIndex.from_tuples([(1, 2), (2, 3)], closed=closed)
        result = tm.round_trip_pickle(idx, temp_file)
        tm.assert_index_equal(result, idx)
