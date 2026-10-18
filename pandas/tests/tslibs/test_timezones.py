# [MODULE DOC]
# Tests for the compiled timezone helper library and timezone-aware behavior across timezone implementations: error messages when timezone data is missing,
# UTC detection and comparison, cache keys, local-timezone handling, timezone inference from datetimes, argument parsing of timezone specifiers, fixed-offset
# detection, and many zoneinfo regression tests (daylight saving rules after the year 2037 and 2100, far-future dates, cached-transition boundaries, nonexistent and
# ambiguous times, dates before the first transition, out-of-range years, odd transition files, and the transition cache). Uses fixtures for UTC aliases supplied elsewhere in the test suite.

from datetime import (
    UTC,
    datetime,
    timedelta,
    timezone,
)
import struct
import subprocess
import sys
import textwrap
import zoneinfo

import dateutil.tz
import numpy as np
import pytest

from pandas._libs.tslibs import (
    conversion,
    timezones,
)
from pandas.compat import (
    IS64,
    is_platform_windows,
)

import pandas as pd
from pandas import (
    Index,
    Timedelta,
    Timestamp,
    date_range,
    to_datetime,
)
import pandas._testing as tm


@pytest.mark.single_cpu
def test_no_timezone_data():
    # [FUNCTION DOC]
    # Marked single-process. Runs a small Python program in a subprocess in which the time zone data package is blocked and the zone search path is empty; localizing a timestamp to Europe/Brussels must raise the zone-not-found error whose message is exactly the expected key message. The subprocess exits with a failure (check_call raises) if its assertion fails. Regression for pull request 63335.
    # https://github.com/pandas-dev/pandas/pull/63335
    # Test error message when timezone data is not available.
    msg = "'No time zone found with key Europe/Brussels'"
    code = textwrap.dedent(
        f"""\
        import sys, zoneinfo, pandas as pd
        sys.modules['tzdata'] = None
        zoneinfo.reset_tzpath(['/path/to/nowhere'])
        try:
            pd.to_datetime('2012-01-01').tz_localize('Europe/Brussels')
        except zoneinfo.ZoneInfoNotFoundError as err:
            assert str(err) == "{msg}"
        """
    )
    subprocess.check_call([sys.executable, "-c", code])


def test_is_utc(utc_fixture):
    # [FUNCTION DOC]
    # For a UTC alias supplied by a fixture, the timezone obtained by the lookup function is recognized as UTC.
    tz = timezones.maybe_get_tz(utc_fixture)
    assert timezones.is_utc(tz)


def test_cache_keys_are_distinct_for_pytz_vs_dateutil():
    # [FUNCTION DOC]
    # Skips if pytz is missing. Intended to check that the cache keys for the pytz and dateutil versions of each common timezone differ.
    # KNOWN ISSUE: the skip check and the assertion are dedented outside the loop, so only the last timezone name in the list is ever examined (and the skip applies to just that one).
    pytz = pytest.importorskip("pytz")
    for tz_name in pytz.common_timezones:
        tz_p = timezones.maybe_get_tz(tz_name)
        tz_d = timezones.maybe_get_tz("dateutil/" + tz_name)

    if tz_d is None:
        pytest.skip(tz_name + ": dateutil does not know about this one")

    if not (tz_name == "UTC" and is_platform_windows()):
        # they both end up as tzwin("UTC") on windows
        assert timezones._p_tz_cache_key(tz_p) != timezones._p_tz_cache_key(tz_d)


def test_tzlocal_repr():
    # [FUNCTION DOC]
    # Regression test (gh-13583): a timestamp with the local dateutil timezone keeps it as its timezone and its representation shows a tzlocal timezone.
    # see gh-13583
    ts = Timestamp("2011-01-01", tz=dateutil.tz.tzlocal())
    assert ts.tz == dateutil.tz.tzlocal()
    assert "tz='tzlocal()')" in repr(ts)


def test_tzlocal_maybe_get_tz():
    # [FUNCTION DOC]
    # Regression test (gh-13583): the lookup of the string for the local timezone gives the local dateutil timezone.
    # see gh-13583
    tz = timezones.maybe_get_tz("tzlocal()")
    assert tz == dateutil.tz.tzlocal()


def test_tzlocal_offset():
    # [FUNCTION DOC]
    # Regression test (gh-13583): the integer value of a second-unit timestamp localized to the local timezone plus the local UTC offset (computed from a plain datetime) equals the value of the naive timestamp.
    # see gh-13583
    #
    # Get offset using normal datetime for test.
    ts = Timestamp("2011-01-01", tz=dateutil.tz.tzlocal()).as_unit("s")

    offset = dateutil.tz.tzlocal().utcoffset(datetime(2011, 1, 1))
    offset = offset.total_seconds()

    assert ts._value + offset == Timestamp("2011-01-01").as_unit("s")._value


def test_tzlocal_is_not_utc():
    # [FUNCTION DOC]
    # Even when the machine is in UTC, the local timezone object is not treated as UTC and does not compare equal to the dateutil UTC timezone.
    # even if the machine running the test is localized to UTC
    tz = dateutil.tz.tzlocal()
    assert not timezones.is_utc(tz)

    assert not timezones.tz_compare(tz, dateutil.tz.tzutc())


def test_tz_compare_utc(utc_fixture, utc_fixture2):
    # [FUNCTION DOC]
    # Two UTC aliases from fixtures, after lookup, compare as the same timezone with the compare function.
    tz = timezones.maybe_get_tz(utc_fixture)
    tz2 = timezones.maybe_get_tz(utc_fixture2)
    assert timezones.tz_compare(tz, tz2)


@pytest.fixture(
    params=[
        ("pytz/US/Eastern", lambda tz, x: tz.localize(x)),
        (dateutil.tz.gettz("US/Eastern"), lambda tz, x: x.replace(tzinfo=tz)),
    ]
)
def infer_setup(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over a pytz (skipped if unavailable) and a dateutil US/Eastern timezone, each with its localize function. Yields a tuple of the timezone, the localize function, aware start (2001-01-01) and end (2009-01-01) datetimes, and the naive start and end.
    eastern, localize = request.param
    if isinstance(eastern, str) and eastern.startswith("pytz/"):
        pytz = pytest.importorskip("pytz")
        eastern = pytz.timezone(eastern.removeprefix("pytz/"))

    start_naive = datetime(2001, 1, 1)
    end_naive = datetime(2009, 1, 1)

    start = localize(eastern, start_naive)
    end = localize(eastern, end_naive)

    return eastern, localize, start, end, start_naive, end_naive


def test_infer_tz_compat(infer_setup):
    # [FUNCTION DOC]
    # Inferring the timezone from two aware datetimes, or from one with the other missing, returns the same timezone object as localizing the corresponding naive datetime.
    eastern, _, start, end, start_naive, end_naive = infer_setup

    assert (
        timezones.infer_tzinfo(start, end)
        is conversion.localize_pydatetime(start_naive, eastern).tzinfo
    )
    assert (
        timezones.infer_tzinfo(start, None)
        is conversion.localize_pydatetime(start_naive, eastern).tzinfo
    )
    assert (
        timezones.infer_tzinfo(None, end)
        is conversion.localize_pydatetime(end_naive, eastern).tzinfo
    )


def test_infer_tz_utc_localize(infer_setup):
    # [FUNCTION DOC]
    # The start and end naive datetimes converted to UTC (with astimezone) infer to the standard UTC timezone.
    _, _, start, end, start_naive, end_naive = infer_setup
    utc = timezone.utc

    start = start_naive.astimezone(utc)
    end = end_naive.astimezone(utc)

    assert timezones.infer_tzinfo(start, end) is utc


@pytest.mark.parametrize("ordered", [True, False])
def test_infer_tz_mismatch(infer_setup, ordered):
    # [FUNCTION DOC]
    # Parametrized over argument order. Inferring from a UTC datetime and a differently-zoned datetime raises an assertion error saying inputs must both have the same timezone.
    eastern, _, _, _, start_naive, end_naive = infer_setup
    msg = "Inputs must both have the same timezone"

    utc = timezone.utc
    start = start_naive.astimezone(utc)
    end = conversion.localize_pydatetime(end_naive, eastern)

    args = (start, end) if ordered else (end, start)

    with pytest.raises(AssertionError, match=msg):
        timezones.infer_tzinfo(*args)


def test_maybe_get_tz_invalid_types():
    # [FUNCTION DOC]
    # The lookup raises type errors that mention the offending class for a float, a module, and a timestamp.
    with pytest.raises(TypeError, match="<class 'float'>"):
        timezones.maybe_get_tz(44.0)

    with pytest.raises(TypeError, match="<class 'module'>"):
        timezones.maybe_get_tz(pytest)

    msg = "<class 'pandas.Timestamp'>"
    with pytest.raises(TypeError, match=msg):
        timezones.maybe_get_tz(Timestamp("2021-01-01", tz="UTC"))


@pytest.mark.parametrize("tz_name", ["UTC", "GMT", "Etc/GMT+1", "Etc/GMT-5"])
def test_zoneinfo_fixed_offset(tz_name):
    # [FUNCTION DOC]
    # GH#64363. Parametrized over zone names UTC, GMT and fixed-offset Etc zones; the fixed-offset check returns true for the corresponding zoneinfo timezone.
    # GH#64363
    zoneinfo = pytest.importorskip("zoneinfo")
    tz = zoneinfo.ZoneInfo(tz_name)
    assert timezones.is_fixed_offset(tz)


def test_zoneinfo_not_fixed_offset_with_historical_transition():
    # [FUNCTION DOC]
    # GH#64363. A zone with a historical transition (Africa/Lusaka) is not a fixed offset.
    # GH#64363
    zoneinfo = pytest.importorskip("zoneinfo")
    tz = zoneinfo.ZoneInfo("Africa/Lusaka")
    assert not timezones.is_fixed_offset(tz)


def test_maybe_get_tz_offset_only():
    # [FUNCTION DOC]
    # Regression test (gh-36004). Lookup of the standard UTC object, offset-only strings (with and without a UTC sign prefix, positive and negative) gives fixed-offset timezones equal to the expected ones.
    # see gh-36004

    # timezone.utc
    tz = timezones.maybe_get_tz(timezone.utc)
    assert tz == timezone(timedelta(hours=0, minutes=0))

    # without UTC+- prefix
    tz = timezones.maybe_get_tz("+01:15")
    assert tz == timezone(timedelta(hours=1, minutes=15))

    tz = timezones.maybe_get_tz("-01:15")
    assert tz == timezone(-timedelta(hours=1, minutes=15))

    # with UTC+- prefix
    tz = timezones.maybe_get_tz("UTC+02:45")
    assert tz == timezone(timedelta(hours=2, minutes=45))

    tz = timezones.maybe_get_tz("UTC-02:45")
    assert tz == timezone(-timedelta(hours=2, minutes=45))


def test_zoneinfo_utc_to_local_post_2037():
    # [FUNCTION DOC]
    # GH#64363. A 24-hour UTC range on July 1 2040 converted to US/Pacific gives hours equal to those computed with the standard library per element (verifying DST transitions after 2037 from POSIX rule strings).
    # GH#64363 - verify that ZoneInfo DST transitions after 2037
    # (generated from POSIX TZ string rules) produce correct local times.
    tz = zoneinfo.ZoneInfo("US/Pacific")
    utc_times = date_range("2040-07-01", periods=24, freq="h", tz="UTC")
    local = utc_times.tz_convert(tz)

    expected_hours = np.array(
        [
            datetime(2040, 7, 1, hour, tzinfo=timezone.utc).astimezone(tz).hour
            for hour in range(24)
        ],
        dtype=np.int32,
    )
    tm.assert_numpy_array_equal(local.hour.to_numpy(), expected_hours)


@pytest.mark.parametrize(
    "tz_name", ["America/New_York", "Australia/Melbourne", "Europe/Kaliningrad"]
)
def test_zoneinfo_utc_to_local_post_2100(tz_name):
    # [FUNCTION DOC]
    # GH#65712. Parametrized over three zones (including Europe/Kaliningrad, which has no future DST rule and is not fixed offset). UTC datetimes in 2038, 2100 and 2200 (January and July) converted to the zone equal the standard library conversions as Python datetimes and have equal UTC offsets.
    # GH#65712 - verify zoneinfo-generated future DST transitions are used
    # beyond year 2100 for both hemispheres.
    # Europe/Kaliningrad is a special case because it has no rule for future DST
    # transitions (and is also not a fixed offset timezone)
    tz = zoneinfo.ZoneInfo(tz_name)
    # fmt: off
    data = [
        "2038-01-01", "2038-07-01",
        "2100-01-01", "2100-07-01",
        "2200-01-01", "2200-07-01",
    ]
    # fmt: on
    utc_times = to_datetime(data, utc=True)
    local = utc_times.tz_convert(tz)

    expected = [
        datetime.fromisoformat(date).replace(tzinfo=timezone.utc).astimezone(tz)
        for date in data
    ]

    tm.assert_numpy_array_equal(local.to_pydatetime(), np.array(expected))
    assert [ts.utcoffset() for ts in local] == [dt.utcoffset() for dt in expected]


def test_zoneinfo_utc_to_local_far_future_seconds_resolution():
    # [FUNCTION DOC]
    # GH#65712. Second-resolution UTC values in year 3000 converted to Europe/Brussels have dtype second with that zone and equal the standard library conversion (falling back to the zone API beyond cached transitions).
    # GH#65712 - for dates beyond cached transition data, we should fall back
    # to zoneinfo's API and preserve correct DST offsets.
    utc_times = to_datetime(
        np.array(["3000-01-01T00:00:00", "3000-07-01T00:00:00"], dtype="M8[s]"),
        utc=True,
    )
    local = utc_times.tz_convert("Europe/Brussels")

    tz = zoneinfo.ZoneInfo("Europe/Brussels")
    expected = [
        datetime(3000, 1, 1, tzinfo=timezone.utc).astimezone(tz),
        datetime(3000, 7, 1, tzinfo=timezone.utc).astimezone(tz),
    ]

    assert local.dtype == "datetime64[s, Europe/Brussels]"
    tm.assert_numpy_array_equal(local.to_pydatetime(), np.array(expected))


def test_zoneinfo_local_to_utc_far_future_seconds_resolution():
    # [FUNCTION DOC]
    # GH#65712. Localizing second-resolution naive values in year 3000 to Europe/Brussels and converting to UTC gives a UTC second dtype and values equal to the standard library conversion.
    # GH#65712 - localize should also preserve future DST rules when converting
    # from local wall times to UTC beyond cached transition data.
    local_times = to_datetime(
        np.array(["3000-01-01T00:00:00", "3000-07-01T00:00:00"], dtype="M8[s]")
    )
    localized = local_times.tz_localize("Europe/Brussels")
    result_utc = localized.tz_convert("UTC")

    tz = zoneinfo.ZoneInfo("Europe/Brussels")
    expected_utc = [
        datetime(3000, 1, 1, tzinfo=tz).astimezone(timezone.utc),
        datetime(3000, 7, 1, tzinfo=tz).astimezone(timezone.utc),
    ]

    assert result_utc.dtype == "datetime64[s, UTC]"
    tm.assert_numpy_array_equal(result_utc.to_pydatetime(), np.array(expected_utc))


@pytest.mark.skipif(
    not IS64,
    reason="stdlib datetime.fromtimestamp fails on 32-bit platforms with overflow",
)
@pytest.mark.parametrize(
    "tz_name",
    ["America/New_York", "America/Santiago", "Australia/Melbourne", "Europe/Brussels"],
)
def test_zoneinfo_boundary_at_last_cached_transition(tz_name):
    # [FUNCTION DOC]
    # GH#65733. Skipped on 32-bit platforms. Parametrized over four zones. Uses a private attribute of the standard library zone implementation to find the 2099 transition instant, builds half-hour UTC values for two days around it, converts to expected local values, then localizes the string forms back with ambiguous times inferred and checks: local to UTC equals the expected UTC datetimes, the string form equals expected strings, and the Python datetimes equal expected local values.
    # GH#65733 - target the boundary where UTC is beyond the cached transition
    # range while local wall time can still compare below that UTC cutoff.
    tz = zoneinfo.ZoneInfo(tz_name)

    from zoneinfo._zoneinfo import ZoneInfo

    trans = datetime.fromtimestamp(max(ZoneInfo(tz_name)._tz_after.transitions(2099)))
    start = (trans - timedelta(days=1)).replace(tzinfo=timezone.utc)
    expected_utc = [(start + timedelta(minutes=30 * i)) for i in range(24 * 2 * 2)]
    expected_tz = [ts.astimezone(tz) for ts in expected_utc]
    expected_str = [str(ts) for ts in expected_tz]

    local = to_datetime([v[:-6] for v in expected_str])

    # ambiguous keyword only needed for timezones of northern hemisphere, ignored for
    # the others
    result = local.tz_localize(tz_name, ambiguous="infer")

    # verify local to UTC
    tm.assert_equal(
        result.tz_convert("UTC").to_pydatetime(), np.array(expected_utc, dtype="O")
    )
    # verify UTC to local by converting to pydatetime/str repr of underlying UTC value
    tm.assert_equal(result.astype(str), Index(expected_str))
    tm.assert_equal(result.to_pydatetime(), np.array(expected_tz, dtype="O"))


@pytest.mark.parametrize(
    "tz_name",
    ["America/New_York", "America/Santiago", "Australia/Melbourne", "Europe/Brussels"],
)
def test_zoneinfo_nonexistent_at_last_cached_transition(tz_name):
    # [FUNCTION DOC]
    # GH#65733, parametrized over four zones. Using the private transition information for 2099: if the first transition is later than the second (spring forward), takes a timestamp half an hour after it; localizing raises a value error for a nonexistent time, nonexistent=NaT gives NaT, shift forward moves half an hour, shift backward moves back to just before the gap (minus one microsecond). Otherwise (fall back), takes a timestamp half an hour before the second transition: localizing raises a value error that the dst time cannot be inferred, ambiguous=NaT gives NaT, ambiguous true and false give the fold 0 and fold 1 datetimes respectively.
    # GH#65733 - target the boundary where UTC is beyond the cached transition
    # range while local wall time can still compare below that UTC cutoff.
    tz = zoneinfo.ZoneInfo(tz_name)

    from zoneinfo._zoneinfo import ZoneInfo

    start, end = ZoneInfo(tz_name)._tz_after.transitions(2099)
    if start > end:
        # to-dst nonexistent transition at the end of the year
        ts = Timestamp(start, unit="s") + Timedelta(minutes=30)

        with pytest.raises(ValueError, match="nonexistent time"):
            ts.tz_localize(tz_name)

        result = ts.tz_localize(tz_name, nonexistent="NaT")
        assert result is pd.NaT

        result_dst = ts.tz_localize(tz_name, nonexistent="shift_forward")
        expected_dst = (ts.to_pydatetime() + timedelta(minutes=30)).replace(tzinfo=tz)
        assert result_dst.to_pydatetime() == expected_dst

        result_std = ts.tz_localize(tz_name, nonexistent="shift_backward")
        expected_std = (
            ts.to_pydatetime() - timedelta(minutes=30, microseconds=1)
        ).replace(tzinfo=tz)
        assert result_std.to_pydatetime() == expected_std

    else:
        # to-std ambiguous transition at the end of the year
        ts = Timestamp(end, unit="s") - Timedelta(minutes=30)

        with pytest.raises(ValueError, match="Cannot infer dst time"):
            ts.tz_localize(tz_name)

        result = ts.tz_localize(tz_name, ambiguous="NaT")
        assert result is pd.NaT

        result_dst = ts.tz_localize(tz_name, ambiguous=True)
        expected_dst = ts.to_pydatetime().replace(fold=0, tzinfo=tz)
        assert result_dst.to_pydatetime() == expected_dst

        result_dst = ts.tz_localize(tz_name, ambiguous=False)
        expected_dst = ts.to_pydatetime().replace(fold=1, tzinfo=tz)
        assert result_dst.to_pydatetime() == expected_dst


@pytest.mark.parametrize("key", ["US/Eastern", "Africa/Lusaka", "Asia/Qyzylorda"])
def test_zoneinfo_utc_to_local_pre_first_transition(key):
    # [FUNCTION DOC]
    # GH#64363. Parametrized over three zones. A UTC timestamp in 1850 (before the first transition) converted to the zone has the same minute as the standard library conversion, which means the pre-first-transition offset is the first offset in the data, not the first standard-time transition offset.
    # GH#64363 - verify that ZoneInfo offsets before the first historical
    # transition (LMT era) match what ZoneInfo itself returns. The "before"
    # offset must be utcoff[0] from the TZ data (matching CPython's C
    # implementation), NOT the first non-DST transition offset.
    tz = zoneinfo.ZoneInfo(key)
    ts = Timestamp("1850-01-01", tz="UTC").tz_convert(tz)

    expected = datetime(1850, 1, 1, tzinfo=timezone.utc).astimezone(tz)
    assert ts.minute == expected.minute


def test_zoneinfo_conversion_outside_range_stdlib():
    # [FUNCTION DOC]
    # GH#65733. A timestamp in year 10000 (outside the standard datetime range): localizing it, and constructing a timestamp with a timezone from its value, each raise a not-implemented error that localizing timestamps outside the Python range is unsupported.
    # GH#65733 - verify that datetimes outside the range of Python's standard
    # library (year > 9999) raises a proper error message
    ts = Timestamp(np.datetime64("10000-01-01T09:00:00", "us"))

    msg = "Localizing Timestamps which are outside the range of Python"
    with pytest.raises(NotImplementedError, match=msg):
        ts.tz_localize("Europe/Brussels")

    with pytest.raises(NotImplementedError, match=msg):
        ts = Timestamp(ts._value, unit="us", tz="Europe/Brussels")


def test_normalize_pytz_timezone():
    # [FUNCTION DOC]
    # Skips without pytz. The io helper that normalizes pytz timezones maps the pytz UTC to the standard UTC, a pytz fixed offset to a standard fixed-offset timezone, and named pytz zones to the same-named zoneinfo zones.
    pytz = pytest.importorskip("pytz")

    from pandas.io._util import _normalize_pytz_timezone

    for tz, expected in [
        (pytz.UTC, timezone.utc),
        (pytz.FixedOffset(90), timezone(timedelta(minutes=90))),
        (pytz.timezone("America/New_York"), zoneinfo.ZoneInfo("America/New_York")),
        (pytz.timezone("Etc/GMT+1"), zoneinfo.ZoneInfo("Etc/GMT+1")),
    ]:
        result = _normalize_pytz_timezone(tz)
        assert result == expected


def _write_big_bang_tzif(path):
    # [FUNCTION DOC]
    # HELPER: for the big bang transition test.
    # Parameter path is a path object to write. Writes a binary timezone data file in version 2 format with an empty version-1 block and a 64-bit block whose first transition is the extreme big bang minus 2 to the 59 seconds (as zic produced between 2013 and 2018f), followed by local mean time until 1883, standard time, and the 1918 and 1919 daylight saving transitions; footer string EST5. Creates parent directories. Side effect: writes the file.
    """
    Write a TZif file opening with the -2**59 "big bang" transition, as zic
    did between 2013 and 2018f.
    """
    # (utoff, isdst, abbrev index)
    ttinfos = [(-17762, 0, 0), (-18000, 0, 4), (-14400, 1, 8)]
    abbrs = b"LMT\x00EST\x00EDT\x00"
    # LMT until 1883, then EST with the 1918 and 1919 DST transitions
    transitions = [
        (-(1 << 59), 0),
        (-2717650800, 1),
        (-1633280400, 2),
        (-1615140000, 1),
        (-1601830800, 2),
        (-1583690400, 1),
    ]

    def block(timecnt):
        # [FUNCTION DOC]
        # HELPER: nested function that returns the binary header for a data block: the magic text and version, 15 reserved zero bytes, then six big-endian 32-bit counts (three zeros, the given transition count, the number of local time types, the abbreviation byte length).
        return (
            b"TZif2"
            + b"\x00" * 15
            + struct.pack(">6i", 0, 0, 0, timecnt, len(ttinfos), len(abbrs))
        )

    def types():
        # [FUNCTION DOC]
        # HELPER: nested function that returns the binary local time type records (offset, daylight flag, abbreviation index) followed by the abbreviation bytes.
        return b"".join(struct.pack(">iBB", *info) for info in ttinfos) + abbrs

    # v1 block, kept empty; the v2 block below carries the 64-bit transitions
    data = block(0) + types()
    data += block(len(transitions))
    data += b"".join(struct.pack(">q", ts) for ts, _ in transitions)
    data += bytes(typ for _, typ in transitions)
    data += types()
    data += b"\nEST5\n"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_zoneinfo_big_bang_transition(tmp_path):
    # [FUNCTION DOC]
    # GH#67066. Writes the big bang data file in a temporary directory and points the zone search path at it. Converts UTC dates in 1850, 1900, 1918, 1920 to that zone and checks they match the standard library conversions (previously the 1883 to 1918 dates wrongly came out as local mean time), and that the cached transition array for that zone is strictly increasing. Always removes the cache entry and resets the search path afterwards.
    # GH#67066 - the "big bang" transition wrapped into the middle of the cached
    # transition array, so 1883-1918 dates came out as LMT instead of EST.
    _write_big_bang_tzif(tmp_path / "Test" / "BigBang")
    zoneinfo.reset_tzpath([str(tmp_path)])
    try:
        tz = zoneinfo.ZoneInfo("Test/BigBang")
        data = ["1850-01-01", "1900-01-01", "1918-07-01", "1920-01-01"]
        utc_times = to_datetime(data, utc=True)
        local = utc_times.tz_convert(tz)

        expected = [
            datetime.fromisoformat(date).replace(tzinfo=UTC).astimezone(tz)
            for date in data
        ]
        tm.assert_numpy_array_equal(local.to_pydatetime(), np.array(expected))

        trans = timezones.dst_cache["zoneinfo/Test/BigBang"][0]
        assert (np.diff(trans) > 0).all()
    finally:
        timezones.dst_cache.pop("zoneinfo/Test/BigBang", None)
        zoneinfo.reset_tzpath()


def test_zoneinfo_posix_rule_transitions_cached():
    # [FUNCTION DOC]
    # GH#67826. Creating a timestamp with a US/Eastern zoneinfo timezone populates the transition cache; the last cached transition is later than 2038-01-01 (the future transitions from the POSIX rule survive validation on 32-bit platforms; no-op on 64-bit).
    # GH#67826 - the POSIX-rule transitions must survive validation on 32-bit,
    # where the platform time_t overflows past 2038. A no-op on 64-bit.
    tz = zoneinfo.ZoneInfo("US/Eastern")
    Timestamp("2020-01-01", tz=tz)

    trans = timezones.dst_cache["zoneinfo/US/Eastern"][0]
    assert trans[-1] > Timestamp("2038-01-01").value
