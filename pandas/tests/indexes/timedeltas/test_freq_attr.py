# [MODULE DOC]
# Tests for getting and setting the frequency attribute on a timedelta index, including validation and view safety.

import pytest

from pandas import TimedeltaIndex

from pandas.tseries.offsets import (
    DateOffset,
    Day,
    Hour,
    MonthEnd,
)


class TestFreq:
    # [CLASS DOC]
    # Test class for frequency attribute behavior.
    @pytest.mark.parametrize("values", [["0 days", "2 days", "4 days"], []])
    @pytest.mark.parametrize("freq", ["2D", Day(2), "48h", Hour(48)])
    def test_freq_setter(self, values, freq):
        # [FUNCTION DOC]
        # Parametrized over data (regular values or empty) and frequency forms (string, offset objects, equivalent 48-hour forms). Setting the frequency on the underlying array converts it to an offset and the index reports it; setting it back to none works.
        # GH#20678
        idx = TimedeltaIndex(values)

        # can set to an offset, converting from string if necessary
        idx._data.freq = freq
        assert idx.freq == freq
        assert isinstance(idx.freq, DateOffset)

        # can reset to None
        idx._data.freq = None
        assert idx.freq is None

    def test_with_freq_empty_requires_tick(self):
        # [FUNCTION DOC]
        # For an empty index, the private with-frequency method on both the index and the array raises a type error unless the offset is a fixed tick (a month-end offset fails with the message "TimedeltaArray/Index freq must be a Tick").
        idx = TimedeltaIndex([])

        off = MonthEnd(1)
        msg = "TimedeltaArray/Index freq must be a Tick"
        with pytest.raises(TypeError, match=msg):
            idx._with_freq(off)
        with pytest.raises(TypeError, match=msg):
            idx._data._with_freq(off)

    def test_freq_setter_errors(self):
        # [FUNCTION DOC]
        # Setting a frequency that does not match the data (five days for a two-day spacing) raises a value error describing the mismatch; a non-fixed frequency (two business days) raises a value error saying it is non-fixed; an invalid string raises a value error "Invalid frequency".
        # GH#20678
        idx = TimedeltaIndex(["0 days", "2 days", "4 days"])

        # setting with an incompatible freq
        msg = (
            "Inferred frequency 2D from passed values does not conform to "
            "passed frequency 5D"
        )
        with pytest.raises(ValueError, match=msg):
            idx._data.freq = "5D"

        # setting with a non-fixed frequency
        msg = r"<2 \* BusinessDays> is a non-fixed frequency"
        with pytest.raises(ValueError, match=msg):
            idx._data.freq = "2B"

        # setting with non-freq string
        with pytest.raises(ValueError, match="Invalid frequency"):
            idx._data.freq = "foo"

    def test_freq_view_safe(self):
        # [FUNCTION DOC]
        # A second index created from the same array and given a different (none) frequency does not change the frequency seen by the original index or the array.
        # Setting the freq for one TimedeltaIndex shouldn't alter the freq
        #  for another that views the same data

        tdi = TimedeltaIndex(["0 days", "2 days", "4 days"], freq="2D")
        tda = tdi._data

        tdi2 = TimedeltaIndex(tda)._with_freq(None)
        assert tdi2.freq is None

        # Original was not altered
        assert tdi.freq == "2D"
        assert tda.freq == "2D"
