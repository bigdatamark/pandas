# [MODULE DOC]
# Tests for Series.copy: deep and shallow copies, tz-aware data and name/index
# name behavior. Uses the datetime_series fixture. No import-time side effects.

import numpy as np
import pytest

from pandas import (
    Series,
    Timestamp,
)
import pandas._testing as tm


class TestCopy:
    # [CLASS DOC]
    # Groups the tests of Series.copy. Stateless.
    @pytest.mark.parametrize("deep", ["default", None, False, True])
    def test_copy(self, deep):
        # [FUNCTION DOC]
        # Parametrized over the deep argument (default, None, False, True). A shallow copy
        # (None or False) shares memory with the original while a deep copy does not,
        # but in every case setting values in the copy leaves the original unchanged
        # (copy-on-write semantics).
        ser = Series(np.arange(10), dtype="float64")

        # default deep is True
        if deep == "default":
            ser2 = ser.copy()
        else:
            ser2 = ser.copy(deep=deep)

        # INFO(CoW) a shallow copy doesn't yet copy the data
        # but parent will not be modified (CoW)
        if deep is None or deep is False:
            assert np.may_share_memory(ser.values, ser2.values)
        else:
            assert not np.may_share_memory(ser.values, ser2.values)

        ser2[::2] = np.nan

        # Did not modify original Series
        assert np.isnan(ser2[0])
        assert not np.isnan(ser[0])

    @pytest.mark.filterwarnings("ignore:Setting a value on a view:FutureWarning")
    @pytest.mark.parametrize("deep", ["default", None, False, True])
    def test_copy_tzaware(self, deep):
        # [FUNCTION DOC]
        # Regression test (GH#11794), parametrized like the copy test: for tz-aware data,
        # assigning a new timestamp into the copy does not change the original; shallow
        # copies share memory beforehand, deep copies do not. Ignores the warning about
        # setting a value on a view.
        # GH#11794
        # copy of tz-aware
        expected = Series([Timestamp("2012/01/01", tz="UTC")])
        expected2 = Series([Timestamp("1999/01/01", tz="UTC")])

        ser = Series([Timestamp("2012/01/01", tz="UTC")])

        if deep == "default":
            ser2 = ser.copy()
        else:
            ser2 = ser.copy(deep=deep)

        # INFO(CoW) a shallow copy doesn't yet copy the data
        # but parent will not be modified (CoW)
        if deep is None or deep is False:
            assert np.may_share_memory(ser.values, ser2.values)
        else:
            assert not np.may_share_memory(ser.values, ser2.values)

        ser2[0] = Timestamp("1999/01/01", tz="UTC")

        # Did not modify original Series
        tm.assert_series_equal(ser2, expected2)
        tm.assert_series_equal(ser, expected)

    def test_copy_name(self, datetime_series):
        # [FUNCTION DOC]
        # The copy of a Series has the same name as the original.
        result = datetime_series.copy()
        assert result.name == datetime_series.name

    def test_copy_index_name_checking(self, datetime_series):
        # [FUNCTION DOC]
        # After clearing the original's index name, renaming the copy's index must not
        # change the original's index name (the index is not shared by reference
        # for name changes).
        # KNOWN ISSUE: the statement asserting that the Series is itself is a
        # tautology that checks nothing.
        # don't want to be able to modify the index stored elsewhere after
        # making a copy

        datetime_series.index.name = None
        assert datetime_series.index.name is None
        assert datetime_series is datetime_series

        cp = datetime_series.copy()
        cp.index.name = "foo"
        assert datetime_series.index.name is None
