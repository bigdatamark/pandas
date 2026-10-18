# [MODULE DOC]
# Tests for the freq attribute of PeriodIndex.
import pytest

from pandas import (
    offsets,
    period_range,
)
import pandas._testing as tm


class TestFreq:
    # [CLASS DOC]
    # Groups freq attribute tests.
    def test_freq_setter_deprecated(self):
        # [FUNCTION DOC]
        # Reading freq raises no warning; assigning freq raises an attribute error
        # saying the property has no setter (GH#20678).
        # KNOWN ISSUE: the name and comments speak of a deprecation warning, but the code expects an attribute error because the setter was removed.
        # GH#20678
        idx = period_range("2018Q1", periods=4, freq="Q")

        # no warning for getter
        with tm.assert_produces_warning(None):
            idx.freq

        # warning for setter
        msg = "property 'freq' of 'PeriodArray' object has no setter"
        with pytest.raises(AttributeError, match=msg):
            idx.freq = offsets.Day()
