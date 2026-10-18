# [MODULE DOC]
# Tests for tz_localize on DataFrame and Series (attach a timezone to a naive datetime axis).

from datetime import timezone

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
    date_range,
)
import pandas._testing as tm


class TestTZLocalize:
    # [CLASS DOC]
    # Test container for tz_localize. Related conversion tests live in the tz_convert test module.
    # See also:
    # test_tz_convert_and_localize in test_tz_convert

    def test_tz_localize(self, frame_or_series):
        # [FUNCTION DOC]
        # Localizing an hourly naive index to UTC gives an index whose tz is the standard library UTC
        # singleton and a result equal to the expected object built from the localized index.
        rng = date_range("1/1/2011", periods=100, freq="h")

        obj = DataFrame({"a": 1}, index=rng)
        obj = tm.get_obj(obj, frame_or_series)

        result = obj.tz_localize("utc")
        expected = DataFrame({"a": 1}, rng.tz_localize("UTC"))
        expected = tm.get_obj(expected, frame_or_series)

        assert result.index.tz is timezone.utc
        tm.assert_equal(result, expected)

    def test_tz_localize_axis1(self):
        # [FUNCTION DOC]
        # Transposes the frame and localizes the column axis to UTC; columns are UTC-aware and
        # result equals the transpose of the row-localized expected frame.
        rng = date_range("1/1/2011", periods=100, freq="h")

        df = DataFrame({"a": 1}, index=rng)

        df = df.T
        result = df.tz_localize("utc", axis=1)
        assert result.columns.tz is timezone.utc

        expected = DataFrame({"a": 1}, rng.tz_localize("UTC"))

        tm.assert_frame_equal(result, expected.T)

    def test_tz_localize_naive(self, frame_or_series):
        # [FUNCTION DOC]
        # Localizing an already timezone-aware index to another zone must raise a type error
        # saying the index is already tz-aware.
        # Can't localize if already tz-aware
        rng = date_range("1/1/2011", periods=100, freq="h", tz="utc")
        ts = Series(1, index=rng)
        ts = frame_or_series(ts)

        with pytest.raises(TypeError, match="Already tz-aware"):
            ts.tz_localize("US/Eastern")

    def test_tz_localize_copy_inplace_mutate(self, frame_or_series):
        # [FUNCTION DOC]
        # Regression test (GH 6326). Localizing a naive hourly index returns an expected UTC object,
        # leaves the original object unchanged, and the result is a new object with a new index object.
        # GH#6326
        obj = frame_or_series(
            np.arange(0, 5), index=date_range("20131027", periods=5, freq="1h", tz=None)
        )
        orig = obj.copy()
        result = obj.tz_localize("UTC")
        expected = frame_or_series(
            np.arange(0, 5),
            index=date_range("20131027", periods=5, freq="1h", tz="UTC"),
        )
        tm.assert_equal(result, expected)
        tm.assert_equal(obj, orig)
        assert result.index is not obj.index
        assert result is not obj
