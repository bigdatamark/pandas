# [MODULE DOC]
# Tests for the seconds-since-epoch method of Timestamp (the method, not the class in
# general). Compares against the standard library datetime behavior and timezone
# conversion.
# NB: This is for the Timestamp.timestamp *method* specifically, not
# the Timestamp class in general.
from datetime import timezone

import pytest

from pandas._libs.tslibs import Timestamp
from pandas.compat import WASM
import pandas.util._test_decorators as td

import pandas._testing as tm


class TestTimestampMethod:
    # [CLASS DOC]
    # Groups the tests of the Timestamp method returning POSIX seconds. Stateless.
    @td.skip_if_windows
    @pytest.mark.skipif(WASM, reason="tzset is not available on WASM")
    def test_timestamp(self, fixed_now_ts):
        # [FUNCTION DOC]
        # Verifies the POSIX-seconds method (GH 17329). Skipped on Windows and on WASM
        # (timezone reset is unavailable). Setup: the fixed-now timestamp fixture, plus
        # a US/Central timestamp. Expected: a tz-naive timestamp is treated as UTC, so
        # it matches the same instant labeled UTC; a timestamp and its UTC conversion
        # give equal values; with the process timezone forced to UTC, the value agrees
        # with the standard library datetime obtained from the timestamp.
        # GH#17329
        # tz-naive --> treat it as if it were UTC for purposes of timestamp()
        ts = fixed_now_ts
        uts = ts.replace(tzinfo=timezone.utc)
        assert ts.timestamp() == uts.timestamp()

        tsc = Timestamp("2014-10-11 11:00:01.12345678", tz="US/Central")
        utsc = tsc.tz_convert("UTC")

        # utsc is a different representation of the same time
        assert tsc.timestamp() == utsc.timestamp()

        # datetime.timestamp() converts in the local timezone
        with tm.set_timezone("UTC"):
            # should agree with datetime.timestamp method
            dt = ts.to_pydatetime()
            assert dt.timestamp() == ts.timestamp()
