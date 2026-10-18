# [MODULE DOC]
# Tests for the delete method of the timedelta index and its effect on the frequency attribute.

from pandas import (
    TimedeltaIndex,
    timedelta_range,
)
import pandas._testing as tm


class TestTimedeltaIndexDelete:
    # [CLASS DOC]
    # Test class for delete.
    def test_delete(self):
        # [FUNCTION DOC]
        # On a named daily range of five: deleting either end (by positive or negative position) preserves the frequency; deleting an interior element resets the frequency to none; the name is always preserved. Deleting a position past the end raises an index or value error depending on numpy version.
        idx = timedelta_range(start="1 Days", periods=5, freq="D", name="idx")

        # preserve freq
        expected_0 = timedelta_range(start="2 Days", periods=4, freq="D", name="idx")
        expected_4 = timedelta_range(start="1 Days", periods=4, freq="D", name="idx")

        # reset freq to None
        expected_1 = TimedeltaIndex(
            ["1 day", "3 day", "4 day", "5 day"], freq=None, name="idx"
        )

        cases = {
            0: expected_0,
            -5: expected_0,
            -1: expected_4,
            4: expected_4,
            1: expected_1,
        }
        for n, expected in cases.items():
            result = idx.delete(n)
            tm.assert_index_equal(result, expected)
            assert result.name == expected.name
            assert result.freq == expected.freq

        with tm.external_error_raised((IndexError, ValueError)):
            # either depending on numpy version
            idx.delete(5)

    def test_delete_slice(self):
        # [FUNCTION DOC]
        # On a ten-element daily range: deleting a tuple of leading or trailing positions, or a slice of the same extent, preserves the frequency, and deleting interior positions resets it; name preserved in all cases.
        idx = timedelta_range(start="1 days", periods=10, freq="D", name="idx")

        # preserve freq
        expected_0_2 = timedelta_range(start="4 days", periods=7, freq="D", name="idx")
        expected_7_9 = timedelta_range(start="1 days", periods=7, freq="D", name="idx")

        # reset freq to None
        expected_3_5 = TimedeltaIndex(
            ["1 D", "2 D", "3 D", "7 D", "8 D", "9 D", "10D"], freq=None, name="idx"
        )

        cases = {
            (0, 1, 2): expected_0_2,
            (7, 8, 9): expected_7_9,
            (3, 4, 5): expected_3_5,
        }
        for n, expected in cases.items():
            result = idx.delete(n)
            tm.assert_index_equal(result, expected)
            assert result.name == expected.name
            assert result.freq == expected.freq

            result = idx.delete(slice(n[0], n[-1] + 1))
            tm.assert_index_equal(result, expected)
            assert result.name == expected.name
            assert result.freq == expected.freq

    def test_delete_doesnt_infer_freq(self):
        # [FUNCTION DOC]
        # Deleting the missing element from an index with no frequency does not infer one: the result's frequency stays none.
        # GH#30655 behavior matches DatetimeIndex

        tdi = TimedeltaIndex(["1 Day", "2 Days", None, "3 Days", "4 Days"])
        result = tdi.delete(2)
        assert result.freq is None
