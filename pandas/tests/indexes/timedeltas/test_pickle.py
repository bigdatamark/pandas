# [MODULE DOC]
# Tests for serialization of the timedelta index.

from pandas import timedelta_range
import pandas._testing as tm


class TestPickle:
    # [CLASS DOC]
    # Test class for pickling.
    def test_pickle_after_set_freq(self, temp_file):
        # [FUNCTION DOC]
        # Uses the temporary file fixture. A range whose frequency was removed round-trips through pickle to an equal index.
        tdi = timedelta_range("1 day", periods=4, freq="s")
        tdi = tdi._with_freq(None)

        res = tm.round_trip_pickle(tdi, temp_file)
        tm.assert_index_equal(res, tdi)
