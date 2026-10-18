# [MODULE DOC]
# Tests for fillna on DatetimeIndex with naive and timezone-aware data and mismatched fill values.
import pytest

import pandas as pd
import pandas._testing as tm


class TestDatetimeIndexFillNA:
    # [CLASS DOC]
    # Groups fillna tests for DatetimeIndex; no state.
    @pytest.mark.parametrize("tz", ["US/Eastern", "Asia/Tokyo"])
    def test_fillna_datetime64(self, tz):
        # [FUNCTION DOC]
        # Parametrized over timezone (US Eastern, Tokyo). For naive data: filling with a naive
        # Timestamp keeps datetime dtype; filling with a timezone-aware Timestamp or a string casts to
        # object. For timezone-aware data: filling with an aware Timestamp keeps the dtype; filling with
        # a naive Timestamp or string casts to object.
        # GH 11343
        idx = pd.DatetimeIndex(["2011-01-01 09:00", pd.NaT, "2011-01-01 11:00"])

        exp = pd.DatetimeIndex(
            ["2011-01-01 09:00", "2011-01-01 10:00", "2011-01-01 11:00"]
        )
        tm.assert_index_equal(idx.fillna(pd.Timestamp("2011-01-01 10:00")), exp)

        # tz mismatch
        exp = pd.Index(
            [
                pd.Timestamp("2011-01-01 09:00"),
                pd.Timestamp("2011-01-01 10:00", tz=tz),
                pd.Timestamp("2011-01-01 11:00"),
            ],
            dtype=object,
        )
        tm.assert_index_equal(idx.fillna(pd.Timestamp("2011-01-01 10:00", tz=tz)), exp)

        # object
        exp = pd.Index(
            [pd.Timestamp("2011-01-01 09:00"), "x", pd.Timestamp("2011-01-01 11:00")],
            dtype=object,
        )
        tm.assert_index_equal(idx.fillna("x"), exp)

        idx = pd.DatetimeIndex(["2011-01-01 09:00", pd.NaT, "2011-01-01 11:00"], tz=tz)

        exp = pd.DatetimeIndex(
            ["2011-01-01 09:00", "2011-01-01 10:00", "2011-01-01 11:00"], tz=tz
        )
        tm.assert_index_equal(idx.fillna(pd.Timestamp("2011-01-01 10:00", tz=tz)), exp)

        exp = pd.Index(
            [
                pd.Timestamp("2011-01-01 09:00", tz=tz),
                pd.Timestamp("2011-01-01 10:00"),
                pd.Timestamp("2011-01-01 11:00", tz=tz),
            ],
            dtype=object,
        )
        tm.assert_index_equal(idx.fillna(pd.Timestamp("2011-01-01 10:00")), exp)

        # object
        exp = pd.Index(
            [
                pd.Timestamp("2011-01-01 09:00", tz=tz),
                "x",
                pd.Timestamp("2011-01-01 11:00", tz=tz),
            ],
            dtype=object,
        )
        tm.assert_index_equal(idx.fillna("x"), exp)
