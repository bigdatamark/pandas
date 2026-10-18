# [MODULE DOC]
# Tests for tz_convert on DataFrame and Series, plus shared tz_convert/tz_localize behavior with
# multi-level indexes and invalid input.

import zoneinfo

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    date_range,
)
import pandas._testing as tm


class TestTZConvert:
    # [CLASS DOC]
    # Test container for tz_convert (and combined tz_localize checks).
    def test_tz_convert(self, frame_or_series):
        # [FUNCTION DOC]
        # Converts a daily US/Eastern index to Europe/Berlin; result index has the Berlin zone key and equals the expected object.
        rng = date_range(
            "1/1/2011", periods=200, freq="D", tz=zoneinfo.ZoneInfo("US/Eastern")
        )

        obj = DataFrame({"a": 1}, index=rng)
        obj = tm.get_obj(obj, frame_or_series)

        berlin = zoneinfo.ZoneInfo("Europe/Berlin")
        result = obj.tz_convert(berlin)
        expected = DataFrame({"a": 1}, rng.tz_convert(berlin))
        expected = tm.get_obj(expected, frame_or_series)

        assert result.index.tz.key == "Europe/Berlin"
        tm.assert_equal(result, expected)

    def test_tz_convert_axis1(self):
        # [FUNCTION DOC]
        # Transposes the frame and converts the column axis (axis 1) to Berlin; columns carry that zone and
        # the result equals the transpose of the expected frame.
        rng = date_range(
            "1/1/2011", periods=200, freq="D", tz=zoneinfo.ZoneInfo("US/Eastern")
        )

        obj = DataFrame({"a": 1}, index=rng)

        obj = obj.T
        berlin = zoneinfo.ZoneInfo("Europe/Berlin")
        result = obj.tz_convert(berlin, axis=1)
        assert result.columns.tz.key == "Europe/Berlin"

        expected = DataFrame({"a": 1}, rng.tz_convert(berlin))

        tm.assert_equal(result, expected.T)

    def test_tz_convert_naive(self, frame_or_series):
        # [FUNCTION DOC]
        # Converting a tz-naive index must raise a type error saying a tz-naive index cannot be converted.
        # can't convert tz-naive
        rng = date_range("1/1/2011", periods=200, freq="D")
        ts = Series(1, index=rng)
        ts = frame_or_series(ts)

        with pytest.raises(TypeError, match="Cannot convert tz-naive"):
            ts.tz_convert("US/Eastern")

    @pytest.mark.parametrize("fn", ["tz_localize", "tz_convert"])
    def test_tz_convert_and_localize(self, fn):
        # [FUNCTION DOC]
        # Parametrized over method name (tz_localize or tz_convert). Two daily ranges are used (made UTC-aware for
        # tz_convert). For each, a frame has the method applied to a flat index and compared to the index method; for a two-level
        # index built from both ranges (GH 7846) the method applied to level 0 changes only that level (other level unchanged) and
        # likewise for level 1. Frequencies are dropped from expected levels because they are not kept in multi-index construction.
        # KNOWN ISSUE: the call on a frame with an integer first level and datetime second level has its result discarded (marked as untested in the code), and the assertions that follow re-check the earlier result, not that frame; also the two expected indexes l0_expected and l1_expected are computed identically.
        l0 = date_range("20140701", periods=5, freq="D")
        l1 = date_range("20140701", periods=5, freq="D")

        int_idx = Index(range(5))

        if fn == "tz_convert":
            l0 = l0.tz_localize("UTC")
            l1 = l1.tz_localize("UTC")

        for idx in [l0, l1]:
            l0_expected = getattr(idx, fn)("US/Pacific")
            l1_expected = getattr(idx, fn)("US/Pacific")

            df1 = DataFrame(np.ones(5), index=l0)
            df1 = getattr(df1, fn)("US/Pacific")
            tm.assert_index_equal(df1.index, l0_expected)

            # MultiIndex
            # GH7846
            df2 = DataFrame(np.ones(5), MultiIndex.from_arrays([l0, l1]))

            # freq is not preserved in MultiIndex construction
            l1_expected = l1_expected._with_freq(None)
            l0_expected = l0_expected._with_freq(None)
            l1 = l1._with_freq(None)
            l0 = l0._with_freq(None)

            df3 = getattr(df2, fn)("US/Pacific", level=0)
            assert not df3.index.levels[0].equals(l0)
            tm.assert_index_equal(df3.index.levels[0], l0_expected)
            tm.assert_index_equal(df3.index.levels[1], l1)
            assert not df3.index.levels[1].equals(l1_expected)

            df3 = getattr(df2, fn)("US/Pacific", level=1)
            tm.assert_index_equal(df3.index.levels[0], l0)
            assert not df3.index.levels[0].equals(l0_expected)
            tm.assert_index_equal(df3.index.levels[1], l1_expected)
            assert not df3.index.levels[1].equals(l1)

            df4 = DataFrame(np.ones(5), MultiIndex.from_arrays([int_idx, l0]))

            # TODO: untested
            getattr(df4, fn)("US/Pacific", level=1)

            tm.assert_index_equal(df3.index.levels[0], l0)
            assert not df3.index.levels[0].equals(l0_expected)
            tm.assert_index_equal(df3.index.levels[1], l1_expected)
            assert not df3.index.levels[1].equals(l1)

    @pytest.mark.parametrize("fn", ["tz_localize", "tz_convert"])
    def test_tz_convert_and_localize_bad_input(self, fn):
        # [FUNCTION DOC]
        # Parametrized over method name (tz_localize or tz_convert). Expects a type error mentioning DatetimeIndex for a
        # frame with an integer index and for a multi-index whose chosen level 0 is an integer level, and a value error
        # saying the level is not valid when level 1 is requested on a single-level index.
        int_idx = Index(range(5))
        l0 = date_range("20140701", periods=5, freq="D")
        # Not DatetimeIndex / PeriodIndex
        df = DataFrame(index=int_idx)
        with pytest.raises(TypeError, match="DatetimeIndex"):
            getattr(df, fn)("US/Pacific")

        # Not DatetimeIndex / PeriodIndex
        df = DataFrame(np.ones(5), MultiIndex.from_arrays([int_idx, l0]))
        with pytest.raises(TypeError, match="DatetimeIndex"):
            getattr(df, fn)("US/Pacific", level=0)

        # Invalid level
        df = DataFrame(index=l0)
        with pytest.raises(ValueError, match="not valid"):
            getattr(df, fn)("US/Pacific", level=1)

    def test_tz_convert_copy_inplace_mutate(self, frame_or_series):
        # [FUNCTION DOC]
        # Regression test (GH 6326). Converting a Berlin hourly object to UTC yields the expected object, leaves the original unchanged,
        # and the result object and its index are new objects.
        # GH#6326
        obj = frame_or_series(
            np.arange(0, 5),
            index=date_range("20131027", periods=5, freq="h", tz="Europe/Berlin"),
        )
        orig = obj.copy()
        result = obj.tz_convert("UTC")
        expected = frame_or_series(np.arange(0, 5), index=obj.index.tz_convert("UTC"))
        tm.assert_equal(result, expected)
        tm.assert_equal(obj, orig)
        assert result.index is not obj.index
        assert result is not obj
