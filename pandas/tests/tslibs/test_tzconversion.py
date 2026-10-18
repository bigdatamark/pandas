# [MODULE DOC]
# Tests for the compiled routine that converts local wall-time integers to UTC integers for a timezone, focusing on ambiguity inference errors.

import zoneinfo

import numpy as np
import pytest

from pandas._libs.tslibs.tzconversion import tz_localize_to_utc


class TestTZLocalizeToUTC:
    # [CLASS DOC]
    # Tests for the local-to-UTC conversion routine.
    def test_tz_localize_to_utc_ambiguous_infer(self):
        # [FUNCTION DOC]
        # Uses a nanosecond integer value that is ambiguous in US/Eastern (the repeated hour on 2011-11-06) in an array of the value, the value minus one, and the value. With inference of ambiguous times: the three-element array raises a value error mentioning the ambiguous wall time; a one-element array raises a value error that there are no repeated times; after incrementing the middle value (making it equal the others) the array raises a value error saying there are two dst switches when there should be only one.
        # val is a timestamp that is ambiguous when localized to US/Eastern
        val = 1_320_541_200_000_000_000
        vals = np.array([val, val - 1, val], dtype=np.int64)

        with pytest.raises(ValueError, match="2011-11-06 01:00:00"):
            tz_localize_to_utc(vals, zoneinfo.ZoneInfo("US/Eastern"), ambiguous="infer")

        with pytest.raises(ValueError, match="are no repeated times"):
            tz_localize_to_utc(
                vals[:1], zoneinfo.ZoneInfo("US/Eastern"), ambiguous="infer"
            )

        vals[1] += 1
        msg = "There are 2 dst switches when there should only be 1"
        with pytest.raises(ValueError, match=msg):
            tz_localize_to_utc(vals, zoneinfo.ZoneInfo("US/Eastern"), ambiguous="infer")
