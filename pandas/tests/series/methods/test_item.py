"""
Series.item method, mainly testing that we get python scalars as opposed to
numpy scalars.
"""

# [MODULE DOC]
# Test for Series.item (and Index.item), mainly confirming that Python scalars
# (or pandas Timestamp/Timedelta) rather than NumPy scalars are returned. No
# import-time side effects.

import pytest

from pandas import (
    Series,
    Timedelta,
    Timestamp,
    date_range,
)


class TestItem:
    # [CLASS DOC]
    # Groups the tests of item. Stateless.
    def test_item(self):
        # [FUNCTION DOC]
        # A one-element integer Series gives a plain Python int, and a float Series with
        # a non-default label gives a Python float. Series or index with two elements
        # raise a value error that only size-1 arrays can be converted. Datetime and
        # timedelta containers of length one give Timestamp or Timedelta objects (and
        # raise for two elements). A one-element slice of a Series with labels other
        # than zero works (positional, not label-based).
        # We are testing that we get python scalars as opposed to numpy scalars
        ser = Series([1])
        result = ser.item()
        assert result == 1
        assert result == ser.iloc[0]
        assert isinstance(result, int)  # i.e. not np.int64

        ser = Series([0.5], index=[3])
        result = ser.item()
        assert type(result) == float
        assert result == 0.5

        ser = Series([1, 2])
        msg = "can only convert an array of size 1"
        with pytest.raises(ValueError, match=msg):
            ser.item()

        dti = date_range("2016-01-01", periods=2)
        with pytest.raises(ValueError, match=msg):
            dti.item()
        with pytest.raises(ValueError, match=msg):
            Series(dti).item()

        val = dti[:1].item()
        assert isinstance(val, Timestamp)
        val = Series(dti)[:1].item()
        assert isinstance(val, Timestamp)

        tdi = dti - dti
        with pytest.raises(ValueError, match=msg):
            tdi.item()
        with pytest.raises(ValueError, match=msg):
            Series(tdi).item()

        val = tdi[:1].item()
        assert isinstance(val, Timedelta)
        val = Series(tdi)[:1].item()
        assert isinstance(val, Timedelta)

        # Case where ser[0] would not work
        ser = Series(dti, index=[5, 6])
        val = ser.iloc[:1].item()
        assert val == dti[0]
