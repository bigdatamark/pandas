# [MODULE DOC]
# Tests for frequency inference (the infer_freq function and the inferred_freq attribute of datetime indexes): regular
# and multiples of frequencies, business-day and business-hour patterns, ambiguity cases, time zones and daylight saving
# transitions, non-nanosecond units, Series and array inputs, invalid input types, legacy frequency aliases, and statelessness.
# Defines a module-level tuple of frequency strings (quarter-end, year-end, business year-end per month, month-end and business
# month variants, week-of-month and weekly per weekday) used by the range inference test. No import-time side effects.
from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

from pandas._libs.tslibs.ccalendar import (
    DAYS,
    MONTHS,
)
from pandas._libs.tslibs.offsets import _get_offset
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td

from pandas import (
    DatetimeIndex,
    Index,
    RangeIndex,
    Series,
    Timestamp,
    date_range,
    period_range,
)
from pandas.core.arrays import (
    DatetimeArray,
    TimedeltaArray,
)
from pandas.core.tools.datetimes import to_datetime

from pandas.tseries import (
    frequencies,
    offsets,
)


@pytest.fixture(
    params=[
        (timedelta(1), "D"),
        (timedelta(hours=1), "h"),
        (timedelta(minutes=1), "min"),
        (timedelta(seconds=1), "s"),
        (np.timedelta64(1, "ns"), "ns"),
        (timedelta(microseconds=1), "us"),
        (timedelta(microseconds=1000), "ms"),
    ]
)
def base_delta_code_pair(request):
    # [FUNCTION DOC]
    # HELPER: parametrized fixture providing (base time delta, expected frequency code) pairs: one day "D", one hour "h", one minute "min", one second "s", one nanosecond "ns", one microsecond "us" and one millisecond "ms". Returns the current parameter tuple.
    return request.param


freqs = (
    [f"QE-{month}" for month in MONTHS]
    + [f"{annual}-{month}" for annual in ["YE", "BYE"] for month in MONTHS]
    + ["ME", "BME", "BMS"]
    + [f"WOM-{count}{day}" for count in range(1, 5) for day in DAYS]
    + [f"W-{day}" for day in DAYS]
)


@pytest.mark.parametrize("freq", freqs)
@pytest.mark.parametrize("periods", [5, 7])
def test_infer_freq_range(periods, freq):
    # [FUNCTION DOC]
    # Parametrized over every entry in the frequency tuple and over periods 5 and 7. Builds a date range with the
    # upper-cased frequency, rebuilds an index from its raw values (dropping the stored frequency), and checks the inferred frequency equals the range's
    # frequency string. For quarter-end ranges the inferred frequency may be the equivalent anchor in the same family
    # (December, November or October based), so the test accepts any of the three consistent combinations of inferred and generating anchors.
    freq = freq.upper()

    gen = date_range("1/1/2000", periods=periods, freq=freq)
    index = DatetimeIndex(gen.values)

    if not freq.startswith("QE-"):
        assert frequencies.infer_freq(index) == gen.freqstr
    else:
        inf_freq = frequencies.infer_freq(index)
        is_dec_range = inf_freq == "QE-DEC" and gen.freqstr in (
            "QE",
            "QE-DEC",
            "QE-SEP",
            "QE-JUN",
            "QE-MAR",
        )
        is_nov_range = inf_freq == "QE-NOV" and gen.freqstr in (
            "QE-NOV",
            "QE-AUG",
            "QE-MAY",
            "QE-FEB",
        )
        is_oct_range = inf_freq == "QE-OCT" and gen.freqstr in (
            "QE-OCT",
            "QE-JUL",
            "QE-APR",
            "QE-JAN",
        )
        assert is_dec_range or is_nov_range or is_oct_range


def test_raise_if_period_index():
    # [FUNCTION DOC]
    # Inferring from a period index raises a type error telling the user to check the frequency attribute instead.
    index = period_range(start="1/1/1990", periods=20, freq="M")
    msg = "Check the `freq` attribute instead of using infer_freq"

    with pytest.raises(TypeError, match=msg):
        frequencies.infer_freq(index)


def test_raise_if_too_few():
    # [FUNCTION DOC]
    # An index with only two dates raises a value error "Need at least 3 dates to infer frequency".
    index = DatetimeIndex(["12/31/1998", "1/3/1999"])
    msg = "Need at least 3 dates to infer frequency"

    with pytest.raises(ValueError, match=msg):
        frequencies.infer_freq(index)


def test_business_daily():
    # [FUNCTION DOC]
    # Three dates spanning a weekend (Friday, Monday, Tuesday) infer the business day code "B".
    index = DatetimeIndex(["01/01/1999", "1/4/1999", "1/5/1999"])
    assert frequencies.infer_freq(index) == "B"


def test_business_daily_look_alike():
    # [FUNCTION DOC]
    # see gh-16624. A weekend-sized 2-day gap in the wrong place (Thursday, Sunday, Monday) must not infer business day; result is None.
    # see gh-16624
    #
    # Do not infer "B when "weekend" (2-day gap) in wrong place.
    index = DatetimeIndex(["12/31/1998", "1/3/1999", "1/4/1999"])
    assert frequencies.infer_freq(index) is None


def test_day_corner():
    # [FUNCTION DOC]
    # Three consecutive calendar days infer "D".
    index = DatetimeIndex(["1/1/2000", "1/2/2000", "1/3/2000"])
    assert frequencies.infer_freq(index) == "D"


def test_non_datetime_index():
    # [FUNCTION DOC]
    # Converting three date strings with the datetime conversion function and inferring from the result gives "D".
    dates = to_datetime(["1/1/2000", "1/2/2000", "1/3/2000"])
    assert frequencies.infer_freq(dates) == "D"


def test_fifth_week_of_month_infer():
    # [FUNCTION DOC]
    # see gh-9425. Dates on the fifth/last Monday of quarter months do not infer a week-of-month frequency (only up to the 4th week is inferred); result is None.
    # see gh-9425
    #
    # Only attempt to infer up to WOM-4.
    index = DatetimeIndex(["2014-03-31", "2014-06-30", "2015-03-30"])
    assert frequencies.infer_freq(index) is None


def test_week_of_month_fake():
    # [FUNCTION DOC]
    # Dates on the same weekday that are 4 or 5 weeks apart must not be inferred as the fourth-Tuesday week-of-month frequency.
    # All of these dates are on same day
    # of week and are 4 or 5 weeks apart.
    index = DatetimeIndex(["2013-08-27", "2013-10-01", "2013-10-29", "2013-11-26"])
    assert frequencies.infer_freq(index) != "WOM-4TUE"


def test_fifth_week_of_month():
    # [FUNCTION DOC]
    # see gh-9425. Creating a date range with a fifth-week-of-month frequency raises a value error "Invalid frequency: WOM-5MON".
    # see gh-9425
    #
    # Only supports freq up to WOM-4.
    msg = "Invalid frequency: WOM-5MON"

    with pytest.raises(ValueError, match=msg):
        date_range("2014-01-01", freq="WOM-5MON")


def test_monthly_ambiguous():
    # [FUNCTION DOC]
    # Month-end dates (Jan 31, Feb 29, Mar 31 of 2000) give the inferred frequency "ME" rather than an annual one.
    rng = DatetimeIndex(["1/31/2000", "2/29/2000", "3/31/2000"])
    assert rng.inferred_freq == "ME"


def test_annual_ambiguous():
    # [FUNCTION DOC]
    # Three January 31 dates in consecutive years give the inferred frequency "YE-JAN".
    rng = DatetimeIndex(["1/31/2000", "1/31/2001", "1/31/2002"])
    assert rng.inferred_freq == "YE-JAN"


@pytest.mark.parametrize("count", range(1, 5))
def test_infer_freq_delta(base_delta_code_pair, count):
    # [FUNCTION DOC]
    # Parametrized over the fixture pair and a count from 1 to 4. Builds three timestamps starting at the current time spaced by the base delta times count.
    # The inferred frequency must be the code alone for count one and the count prefixed to the code otherwise.
    b = Timestamp(datetime.now())
    base_delta, code = base_delta_code_pair

    inc = base_delta * count
    index = DatetimeIndex([b + inc * j for j in range(3)])

    exp_freq = f"{count:d}{code}" if count > 1 else code
    assert frequencies.infer_freq(index) == exp_freq


@pytest.mark.parametrize(
    "constructor",
    [
        lambda now, delta: DatetimeIndex(
            [now + delta * 7] + [now + delta * j for j in range(3)]
        ),
        lambda now, delta: DatetimeIndex(
            [now + delta * j for j in range(3)] + [now + delta * 7]
        ),
    ],
)
def test_infer_freq_custom(base_delta_code_pair, constructor):
    # [FUNCTION DOC]
    # Parametrized over two constructors that add an irregular extra point (seven steps) at the start or end of an otherwise regular three-point series, and over the fixture pair. Index with an irregular gap gives None.
    b = Timestamp(datetime.now())
    base_delta, _ = base_delta_code_pair

    index = constructor(b, base_delta)
    assert frequencies.infer_freq(index) is None


@pytest.mark.parametrize(
    "expected,dates",
    list(
        {
            "YS-JAN": ["2009-01-01", "2010-01-01", "2011-01-01", "2012-01-01"],
            "QE-OCT": ["2009-01-31", "2009-04-30", "2009-07-31", "2009-10-31"],
            "ME": ["2010-11-30", "2010-12-31", "2011-01-31", "2011-02-28"],
            "W-SAT": ["2010-12-25", "2011-01-01", "2011-01-08", "2011-01-15"],
            "D": ["2011-01-01", "2011-01-02", "2011-01-03", "2011-01-04"],
            "h": [
                "2011-12-31 22:00",
                "2011-12-31 23:00",
                "2012-01-01 00:00",
                "2012-01-01 01:00",
            ],
        }.items()
    ),
)
def test_infer_freq_tz(tz_naive_fixture, expected, dates, unit):
    # [FUNCTION DOC]
    # see gh-7310, GH#55609. Parametrized over (expected frequency, four date strings) for year-start, quarter-end, month-end, weekly Saturday, daily and hourly, plus the time zone fixture (including naive) and a unit fixture. Index built with the zone and cast to the unit has the expected inferred frequency.
    # see gh-7310, GH#55609
    tz = tz_naive_fixture
    idx = DatetimeIndex(dates, tz=tz).as_unit(unit)
    assert idx.inferred_freq == expected


def test_infer_freq_tz_series(tz_naive_fixture):
    # [FUNCTION DOC]
    # see gh-52456. A Series made from a four-day daily range (naive or aware per the fixture) with a reset index infers "D".
    # infer_freq should work with both tz-naive and tz-aware series. See gh-52456
    tz = tz_naive_fixture
    idx = date_range("2021-01-01", "2021-01-04", tz=tz)
    series = idx.to_series().reset_index(drop=True)
    inferred_freq = frequencies.infer_freq(series)
    assert inferred_freq == "D"


@pytest.mark.parametrize(
    "date_pair",
    [
        ["2013-11-02", "2013-11-5"],  # Fall DST
        ["2014-03-08", "2014-03-11"],  # Spring DST
        ["2014-01-01", "2014-01-03"],  # Regular Time
    ],
)
@pytest.mark.parametrize(
    "freq",
    ["h", "3h", "10min", "3601s", "3600001ms", "3600000001us", "3600000000001ns"],
)
def test_infer_freq_tz_transition(tz_naive_fixture, date_pair, freq):
    # [FUNCTION DOC]
    # see gh-8772. Parametrized over date pairs (spanning fall DST, spring DST, and no transition), a set of frequencies from hours down to nanoseconds, and the time zone fixture. A date range in that zone has an inferred frequency equal to the generating frequency, even across DST changes.
    # see gh-8772
    tz = tz_naive_fixture
    idx = date_range(date_pair[0], date_pair[1], freq=freq, tz=tz)
    assert idx.inferred_freq == freq


def test_infer_freq_tz_transition_custom():
    # [FUNCTION DOC]
    # A three-hour range starting 2013-11-03 localized after creation to America/Chicago (crossing the fall back transition so wall-clock gaps are irregular) gives inferred frequency None.
    index = date_range("2013-11-03", periods=5, freq="3h").tz_localize(
        "America/Chicago"
    )
    assert index.inferred_freq is None


@pytest.mark.parametrize(
    "data,expected",
    [
        # Hourly freq in a day must result in "h"
        (
            [
                "2014-07-01 09:00",
                "2014-07-01 10:00",
                "2014-07-01 11:00",
                "2014-07-01 12:00",
                "2014-07-01 13:00",
                "2014-07-01 14:00",
            ],
            "h",
        ),
        (
            [
                "2014-07-01 09:00",
                "2014-07-01 10:00",
                "2014-07-01 11:00",
                "2014-07-01 12:00",
                "2014-07-01 13:00",
                "2014-07-01 14:00",
                "2014-07-01 15:00",
                "2014-07-01 16:00",
                "2014-07-02 09:00",
                "2014-07-02 10:00",
                "2014-07-02 11:00",
            ],
            "bh",
        ),
        (
            [
                "2014-07-04 09:00",
                "2014-07-04 10:00",
                "2014-07-04 11:00",
                "2014-07-04 12:00",
                "2014-07-04 13:00",
                "2014-07-04 14:00",
                "2014-07-04 15:00",
                "2014-07-04 16:00",
                "2014-07-07 09:00",
                "2014-07-07 10:00",
                "2014-07-07 11:00",
            ],
            "bh",
        ),
        (
            [
                "2014-07-04 09:00",
                "2014-07-04 10:00",
                "2014-07-04 11:00",
                "2014-07-04 12:00",
                "2014-07-04 13:00",
                "2014-07-04 14:00",
                "2014-07-04 15:00",
                "2014-07-04 16:00",
                "2014-07-07 09:00",
                "2014-07-07 10:00",
                "2014-07-07 11:00",
                "2014-07-07 12:00",
                "2014-07-07 13:00",
                "2014-07-07 14:00",
                "2014-07-07 15:00",
                "2014-07-07 16:00",
                "2014-07-08 09:00",
                "2014-07-08 10:00",
                "2014-07-08 11:00",
                "2014-07-08 12:00",
                "2014-07-08 13:00",
                "2014-07-08 14:00",
                "2014-07-08 15:00",
                "2014-07-08 16:00",
            ],
            "bh",
        ),
    ],
)
def test_infer_freq_business_hour(data, expected):
    # [FUNCTION DOC]
    # see gh-7905. Parametrized over lists of timestamp strings with their expected result: continuous hourly stamps in one day give "h"; stamps that cover business hours 09:00-16:00 per day, skipping the weekend, give "bh". Builds an index and checks its inferred frequency.
    # see gh-7905
    idx = DatetimeIndex(data)
    assert idx.inferred_freq == expected


def test_not_monotonic():
    # [FUNCTION DOC]
    # A decreasing (reversed) series of annual January 31 dates gives the negative-step inferred frequency "-1YE-JAN".
    rng = DatetimeIndex(["1/31/2000", "1/31/2001", "1/31/2002"])
    rng = rng[::-1]

    assert rng.inferred_freq == "-1YE-JAN"


def test_non_datetime_index2():
    # [FUNCTION DOC]
    # Inferring from a plain array of Python datetime objects gives the same result as the inferred frequency attribute of the equivalent index.
    rng = DatetimeIndex(["1/31/2000", "1/31/2001", "1/31/2002"])
    vals = rng.to_pydatetime()

    result = frequencies.infer_freq(vals)
    assert result == rng.inferred_freq


@pytest.mark.parametrize(
    "idx",
    [
        Index(np.arange(5), dtype=np.int64),
        Index(np.arange(5), dtype=np.float64),
        period_range("2020-01-01", periods=5),
        RangeIndex(5),
    ],
)
def test_invalid_index_types(idx):
    # [FUNCTION DOC]
    # see gh-48439. Parametrized over integer, float, period and range indexes. Inferring raises a type error with either a "cannot infer freq from a non-convertible" message or the message telling the user to check the frequency attribute.
    # see gh-48439
    msg = "|".join(
        [
            "cannot infer freq from a non-convertible",
            "Check the `freq` attribute instead of using infer_freq",
        ]
    )

    with pytest.raises(TypeError, match=msg):
        frequencies.infer_freq(idx)


@pytest.mark.skipif(is_platform_windows(), reason="see gh-10822: Windows issue")
def test_invalid_index_types_unicode():
    # [FUNCTION DOC]
    # see gh-10822. Skipped on Windows. An index with a non-date string raises a value error "Unknown datetime string format".
    # see gh-10822
    #
    # Odd error message on conversions to datetime for unicode.
    msg = "Unknown datetime string format"

    with pytest.raises(ValueError, match=msg):
        frequencies.infer_freq(Index(["ZqgszYBfuL"]))


def test_string_datetime_like_compat():
    # [FUNCTION DOC]
    # see gh-6463. Inferring from a list of date-like month strings gives the same result as from an index holding those strings.
    # see gh-6463
    data = ["2004-01", "2004-02", "2004-03", "2004-04"]

    expected = frequencies.infer_freq(data)
    result = frequencies.infer_freq(Index(data))

    assert result == expected


def test_series():
    # [FUNCTION DOC]
    # see gh-6407. A Series from a daily range infers "D".
    # see gh-6407
    s = Series(date_range("20130101", "20130110"))
    inferred = frequencies.infer_freq(s)
    assert inferred == "D"


@pytest.mark.parametrize("end", [10, 10.0])
def test_series_invalid_type(end):
    # [FUNCTION DOC]
    # see gh-6407. Parametrized over end 10 and 10.0. A Series of integers or floats raises a type error "cannot infer freq from a non-convertible dtype on a Series".
    # see gh-6407
    msg = "cannot infer freq from a non-convertible dtype on a Series"
    s = Series(np.arange(end))

    with pytest.raises(TypeError, match=msg):
        frequencies.infer_freq(s)


def test_series_inconvertible_string(using_infer_string):
    # [FUNCTION DOC]
    # see gh-6407. Uses the infer-string fixture. A Series of non-date strings raises a type error ("cannot infer freq from") when the string inference mode is on, otherwise a value error "Unknown datetime string format".
    # see gh-6407
    if using_infer_string:
        msg = "cannot infer freq from"

        with pytest.raises(TypeError, match=msg):
            frequencies.infer_freq(Series(["foo", "bar"]))
    else:
        msg = "Unknown datetime string format"

        with pytest.raises(ValueError, match=msg):
            frequencies.infer_freq(Series(["foo", "bar"]))


@pytest.mark.parametrize("freq", [None, "ms"])
def test_series_period_index(freq):
    # [FUNCTION DOC]
    # see gh-6407. Parametrized over freq None and "ms". A Series of periods raises a type error "cannot infer freq from a non-convertible dtype on a Series".
    # see gh-6407
    #
    # Cannot infer on PeriodIndex
    msg = "cannot infer freq from a non-convertible dtype on a Series"
    s = Series(period_range("2013", periods=10, freq=freq))

    with pytest.raises(TypeError, match=msg):
        frequencies.infer_freq(s)


@pytest.mark.parametrize("freq", ["ME", "ms", "s"])
def test_series_datetime_index(freq):
    # [FUNCTION DOC]
    # Parametrized over month-end, millisecond and second. A Series of ten timestamps from a date range with that frequency infers that same frequency.
    s = Series(date_range("20130101", periods=10, freq=freq))
    inferred = frequencies.infer_freq(s)
    assert inferred == freq


@pytest.mark.parametrize(
    "offset_func",
    [
        _get_offset,
        lambda freq: date_range("2011-01-01", periods=5, freq=freq),
    ],
)
@pytest.mark.parametrize(
    "freq",
    [
        "WEEKDAY",
        "EOM",
        "W@MON",
        "W@TUE",
        "W@WED",
        "W@THU",
        "W@FRI",
        "W@SAT",
        "W@SUN",
        "QE@JAN",
        "QE@FEB",
        "QE@MAR",
        "YE@JAN",
        "YE@FEB",
        "YE@MAR",
        "YE@APR",
        "YE@MAY",
        "YE@JUN",
        "YE@JUL",
        "YE@AUG",
        "YE@SEP",
        "YE@OCT",
        "YE@NOV",
        "YE@DEC",
        "WOM@1MON",
        "WOM@2MON",
        "WOM@3MON",
        "WOM@4MON",
        "WOM@1TUE",
        "WOM@2TUE",
        "WOM@3TUE",
        "WOM@4TUE",
        "WOM@1WED",
        "WOM@2WED",
        "WOM@3WED",
        "WOM@4WED",
        "WOM@1THU",
        "WOM@2THU",
        "WOM@3THU",
        "WOM@4THU",
        "WOM@1FRI",
        "WOM@2FRI",
        "WOM@3FRI",
        "WOM@4FRI",
    ],
)
def test_legacy_offset_warnings(offset_func, freq):
    # [FUNCTION DOC]
    # Parametrized over two builders (the internal offset lookup, and a date range creator) and a long list of legacy frequency aliases (WEEKDAY, EOM, W@..., QE@..., YE@..., WOM@...).
    # Every combination must raise a value error with the invalid-frequency message.
    # KNOWN ISSUE: the name says warnings, but the test checks that an error is raised for the removed aliases.
    with pytest.raises(ValueError, match=INVALID_FREQ_ERR_MSG):
        offset_func(freq)


def test_ms_vs_capital_ms():
    # [FUNCTION DOC]
    # The offset lookup maps lowercase "ms" to the millisecond offset and uppercase "MS" to the month-begin offset (case sensitivity).
    left = _get_offset("ms")
    right = _get_offset("MS")

    assert left == offsets.Milli()
    assert right == offsets.MonthBegin()


def test_infer_freq_non_nano():
    # [FUNCTION DOC]
    # Builds a datetime array in second resolution and a timedelta array in millisecond resolution directly from raw integers (0 to 9) through the private simple constructors; inferred frequencies are "s" and "ms" respectively.
    arr = np.arange(10).astype(np.int64).view("M8[s]")
    dta = DatetimeArray._simple_new(arr, dtype=arr.dtype)
    res = frequencies.infer_freq(dta)
    assert res == "s"

    arr2 = arr.view("m8[ms]")
    tda = TimedeltaArray._simple_new(arr2, dtype=arr2.dtype)
    res2 = frequencies.infer_freq(tda)
    assert res2 == "ms"


def test_infer_freq_non_nano_tzaware(tz_aware_fixture):
    # [FUNCTION DOC]
    # Parametrized by the tz-aware fixture. A 365-period business-day range cast to second resolution infers "B".
    tz = tz_aware_fixture

    dti = date_range("2016-01-01", periods=365, freq="B", tz=tz)
    dta = dti._data.as_unit("s")

    res = frequencies.infer_freq(dta)
    assert res == "B"


@td.skip_if_no("pyarrow")
def test_infer_freq_pyarrow():
    # [FUNCTION DOC]
    # GH#58403. Skipped when pyarrow is missing. Three timestamps 30 seconds apart stored with a pyarrow timestamp dtype in both a Series and an Index infer "30s" whether passing the container or its underlying values.
    # GH#58403
    data = ["2022-01-01T10:00:00", "2022-01-01T10:00:30", "2022-01-01T10:01:00"]
    pd_series = Series(data).astype("timestamp[s][pyarrow]")
    pd_index = Index(data).astype("timestamp[s][pyarrow]")

    assert frequencies.infer_freq(pd_index.values) == "30s"
    assert frequencies.infer_freq(pd_series.values) == "30s"
    assert frequencies.infer_freq(pd_index) == "30s"
    assert frequencies.infer_freq(pd_series) == "30s"


def test_infer_freq_no_stateful_behavior():
    # [FUNCTION DOC]
    # GH#55794. Inference on the first three days gives "D", inference on the full index with a duplicate date gives None, and inference on the three-day slice again still gives "D" (no state is kept between calls).
    # GH#55794 infer_freq should not have stateful behavior
    # calling infer_freq on a full index with duplicates should not
    # affect the result of calling it on a slice without duplicates
    times = to_datetime(["2019-01-01", "2019-01-02", "2019-01-03", "2019-01-03"])

    # Before calling infer_freq on full index
    assert frequencies.infer_freq(times[:3]) == "D"

    # Call on full index (which has duplicates, so returns None)
    assert frequencies.infer_freq(times) is None

    # After calling on full index, slice should still return "D"
    assert frequencies.infer_freq(times[:3]) == "D"
