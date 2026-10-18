# [MODULE DOC]
# Tests for localizing the naive datetimes or index of a Series to a time zone, covering ambiguous times, nonexistent (DST gap) times and empty series.
# Relies on shared fixtures for a Warsaw time zone and a datetime resolution unit. No import-time side effects.

from datetime import timezone

import pytest

from pandas._libs.tslibs import timezones

from pandas import (
    DatetimeIndex,
    NaT,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm


class TestTZLocalize:
    # [CLASS DOC]
    # Groups tests of time-zone localization of Series data and indexes. Stateless.
    def test_series_tz_localize_ambiguous_bool(self):
        # [FUNCTION DOC]
        # Verifies that the ambiguous option of datetime-accessor localization accepts booleans and boolean lists (issue 14402): a naive timestamp during the autumn DST overlap in US Central raises a ValueError (from the underlying library) by default, while true or a list with true selects the daylight-time offset and false selects the standard-time offset.
        # make sure that we are correctly accepting bool values as ambiguous

        # GH#14402
        ts = Timestamp("2015-11-01 01:00:03")
        expected0 = Timestamp("2015-11-01 01:00:03-0500", tz="US/Central")
        expected1 = Timestamp("2015-11-01 01:00:03-0600", tz="US/Central")

        ser = Series([ts])
        expected0 = Series([expected0])
        expected1 = Series([expected1])

        with tm.external_error_raised(ValueError):
            ser.dt.tz_localize("US/Central")

        result = ser.dt.tz_localize("US/Central", ambiguous=True)
        tm.assert_series_equal(result, expected0)

        result = ser.dt.tz_localize("US/Central", ambiguous=[True])
        tm.assert_series_equal(result, expected0)

        result = ser.dt.tz_localize("US/Central", ambiguous=False)
        tm.assert_series_equal(result, expected1)

        result = ser.dt.tz_localize("US/Central", ambiguous=[False])
        tm.assert_series_equal(result, expected1)

    def test_series_tz_localize_matching_index(self):
        # [FUNCTION DOC]
        # Verifies that localizing a categorical-dtype datetime Series through the datetime accessor keeps the original non-default index on the result, which has a plain tz-aware datetime dtype (issue 43080).
        # Matching the index of the result with that of the original series
        # GH 43080
        dt_series = Series(
            date_range(start="2021-01-01T02:00:00", periods=5, freq="1D"),
            index=[2, 6, 7, 8, 11],
            dtype="category",
        )
        result = dt_series.dt.tz_localize("Europe/Berlin")
        expected = Series(
            date_range(
                start="2021-01-01T02:00:00", periods=5, freq="1D", tz="Europe/Berlin"
            ),
            index=[2, 6, 7, 8, 11],
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("year", ["2015", "2201"])
    @pytest.mark.parametrize(
        "method, exp",
        [
            ["shift_forward", "2015-03-29 03:00:00"],
            ["shift_backward", "2015-03-29 01:59:59.999999999"],
            ["NaT", NaT],
            ["raise", None],
            ["foo", "invalid"],
        ],
    )
    def test_tz_localize_nonexistent(self, warsaw, method, exp, year, unit):
        # [FUNCTION DOC]
        # Verifies the nonexistent option when localizing a one-hour minute-frequency range starting in a DST gap, for a Series, its one-column frame and a datetime index. Parametrized over the year (2015, 2201), the method and the expected value, plus the Warsaw time zone and resolution unit fixtures.
        # For raise, each object raises a ValueError. For an invalid method name, each raises a ValueError with the message listing allowed options. Otherwise (shift forward, shift backward, NaT) all 60 entries must localize to the single expected timestamp (year substituted), checked for Series, frame and index.
        # Skips the case for year 2201 unless the time zone is a non-dateutil zone object (zoneinfo passed as string is required).
        # GH 8917
        tz = warsaw
        if year == "2201" and (not isinstance(tz, str) or tz.startswith("dateutil/")):
            # zoneinfo tz passed as string
            pytest.skip("ZoneInfo required for year 2201")
        n = 60
        dti = date_range(
            start=f"{year}-03-29 02:00:00", periods=n, freq="min", unit=unit
        )
        ser = Series(1, index=dti)
        df = ser.to_frame()

        if method == "raise":
            with tm.external_error_raised(ValueError):
                dti.tz_localize(tz, nonexistent=method)
            with tm.external_error_raised(ValueError):
                ser.tz_localize(tz, nonexistent=method)
            with tm.external_error_raised(ValueError):
                df.tz_localize(tz, nonexistent=method)

        elif exp == "invalid":
            msg = (
                "The nonexistent argument must be one of "
                "'raise', 'NaT', 'shift_forward', 'shift_backward' "
                "or a timedelta object"
            )
            with pytest.raises(ValueError, match=msg):
                dti.tz_localize(tz, nonexistent=method)
            with pytest.raises(ValueError, match=msg):
                ser.tz_localize(tz, nonexistent=method)
            with pytest.raises(ValueError, match=msg):
                df.tz_localize(tz, nonexistent=method)

        else:
            if isinstance(exp, str):
                exp = exp.replace("2015", year)
            result = ser.tz_localize(tz, nonexistent=method)
            expected = Series(1, index=DatetimeIndex([exp] * n, tz=tz).as_unit(unit))
            tm.assert_series_equal(result, expected)

            result = df.tz_localize(tz, nonexistent=method)
            expected = expected.to_frame()
            tm.assert_frame_equal(result, expected)

            res_index = dti.tz_localize(tz, nonexistent=method)
            tm.assert_index_equal(res_index, expected.index)

    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_series_tz_localize_empty(self, tzstr):
        # [FUNCTION DOC]
        # Verifies localizing an empty object-dtype Series: using "utc" yields an index with the UTC timezone; localizing to a named zone (plain and dateutil variants) is compared with the resolved zone.
        # KNOWN ISSUE: the result of the zone comparison helper is never asserted, so the second check cannot fail on a mismatch.
        # GH#2248
        ser = Series(dtype=object)

        ser2 = ser.tz_localize("utc")
        assert ser2.index.tz == timezone.utc

        ser2 = ser.tz_localize(tzstr)
        timezones.tz_compare(ser2.index.tz, timezones.maybe_get_tz(tzstr))
