# [MODULE DOC]
# Tests for the insert method of the timedelta index: value insertion, frequency preservation or reset, coercion to object for non-timedelta items, handling of missing markers of several kinds, string handling and insertion into empty indexes.

from datetime import timedelta

import numpy as np
import pytest

from pandas._libs import lib

import pandas as pd
from pandas import (
    Index,
    Timedelta,
    TimedeltaIndex,
    timedelta_range,
)
import pandas._testing as tm


class TestTimedeltaIndexInsert:
    # [CLASS DOC]
    # Test class for inserting into timedelta indexes.
    def test_insert(self):
        # [FUNCTION DOC]
        # Inserting a timedelta in the middle keeps the name; inserting a non-timedelta string gives an object index with the name. For a second-frequency range, parametrized cases (via a list of position, item, expected) check that inserting at either end with a value continuing the sequence preserves the frequency while inserting elsewhere or a non-continuing value resets the frequency to none, with the name preserved.
        idx = TimedeltaIndex(["4day", "1day", "2day"], name="idx")

        result = idx.insert(2, timedelta(days=5))
        exp = TimedeltaIndex(["4day", "1day", "5day", "2day"], name="idx")
        tm.assert_index_equal(result, exp)

        # insertion of non-datetime should coerce to object index
        result = idx.insert(1, "inserted")
        expected = Index(
            [Timedelta("4day"), "inserted", Timedelta("1day"), Timedelta("2day")],
            name="idx",
        )
        assert not isinstance(result, TimedeltaIndex)
        tm.assert_index_equal(result, expected)
        assert result.name == expected.name

        idx = timedelta_range("1day 00:00:01", periods=3, freq="s", name="idx")

        # preserve freq
        expected_0 = TimedeltaIndex(
            ["1day", "1day 00:00:01", "1day 00:00:02", "1day 00:00:03"],
            name="idx",
            freq="s",
        )
        expected_3 = TimedeltaIndex(
            ["1day 00:00:01", "1day 00:00:02", "1day 00:00:03", "1day 00:00:04"],
            name="idx",
            freq="s",
        )

        # reset freq to None
        expected_1_nofreq = TimedeltaIndex(
            ["1day 00:00:01", "1day 00:00:01", "1day 00:00:02", "1day 00:00:03"],
            name="idx",
            freq=None,
        )
        expected_3_nofreq = TimedeltaIndex(
            ["1day 00:00:01", "1day 00:00:02", "1day 00:00:03", "1day 00:00:05"],
            name="idx",
            freq=None,
        )

        cases = [
            (0, Timedelta("1day"), expected_0),
            (-3, Timedelta("1day"), expected_0),
            (3, Timedelta("1day 00:00:04"), expected_3),
            (1, Timedelta("1day 00:00:01"), expected_1_nofreq),
            (3, Timedelta("1day 00:00:05"), expected_3_nofreq),
        ]

        for n, d, expected in cases:
            result = idx.insert(n, d)
            tm.assert_index_equal(result, expected)
            assert result.name == expected.name
            assert result.freq == expected.freq

    @pytest.mark.parametrize(
        "null", [None, np.nan, np.timedelta64("NaT", "ns"), pd.NaT, pd.NA]
    )
    def test_insert_nat(self, null):
        # [FUNCTION DOC]
        # Parametrized over missing markers (None, NaN, timedelta NaT of numpy, pandas NaT, NA). Inserting any at position one of a daily range gives a timedelta index with NaT there.
        # GH 18295 (test missing)
        idx = timedelta_range("1day", "3day")
        result = idx.insert(1, null)
        expected = TimedeltaIndex(["1day", pd.NaT, "2day", "3day"])
        tm.assert_index_equal(result, expected)

    def test_insert_invalid_na(self):
        # [FUNCTION DOC]
        # Inserting a datetime NaT (numpy) into a timedelta index gives an object index containing that NaT item with the name preserved; another instance of such an NaT gives the same result.
        idx = TimedeltaIndex(["4day", "1day", "2day"], name="idx")

        item = np.datetime64("NaT", "ns")
        result = idx.insert(0, item)

        expected = Index([item, *list(idx)], dtype=object, name="idx")
        tm.assert_index_equal(result, expected)

        # Also works if we pass a different dt64nat object
        item2 = np.datetime64("NaT", "ns")
        result = idx.insert(0, item2)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "item", [0, np.int64(0), np.float64(0), np.array(0), np.datetime64(456, "us")]
    )
    def test_insert_mismatched_types_raises(self, item):
        # [FUNCTION DOC]
        # KNOWN ISSUE: the name says it raises but the test asserts that an object index is returned.
        # Parametrized over zero as several numeric kinds, a zero-dimensional array and a numpy datetime. Inserting one at position one does not cast to timedelta; the result is an object index with the unwrapped item.
        # GH#33703 dont cast these to td64
        tdi = TimedeltaIndex(["4day", "1day", "2day"], name="idx")

        result = tdi.insert(1, item)

        expected = Index(
            [tdi[0], lib.item_from_zerodim(item), *list(tdi[1:])],
            dtype=object,
            name="idx",
        )
        tm.assert_index_equal(result, expected)

    def test_insert_castable_str(self):
        # [FUNCTION DOC]
        # Inserting a string that parses as a timedelta at the start gives a timedelta index with the parsed value first.
        idx = timedelta_range("1day", "3day")

        result = idx.insert(0, "1 Day")

        expected = TimedeltaIndex([idx[0], *list(idx)])
        tm.assert_index_equal(result, expected)

    def test_insert_non_castable_str(self):
        # [FUNCTION DOC]
        # Inserting an unparseable string gives an object index with the string first.
        idx = timedelta_range("1day", "3day")

        result = idx.insert(0, "foo")

        expected = Index(["foo", *list(idx)], dtype=object)
        tm.assert_index_equal(result, expected)

    def test_insert_empty(self):
        # [FUNCTION DOC]
        # Inserting the first element into an empty slice at position zero keeps a daily frequency; positions one or minus one on an empty index raise an index error saying loc must be an integer between bounds.
        # Corner case inserting with length zero doesn't raise IndexError
        # GH#33573 for freq preservation
        idx = timedelta_range("1 Day", periods=3)
        td = idx[0]

        result = idx[:0].insert(0, td)
        assert result.freq == "D"

        with pytest.raises(IndexError, match="loc must be an integer between"):
            result = idx[:0].insert(1, td)

        with pytest.raises(IndexError, match="loc must be an integer between"):
            result = idx[:0].insert(-1, td)
