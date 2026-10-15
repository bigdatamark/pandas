# [MODULE DOC]
# Vectorized field extraction for datetime-like int64 arrays: values are counts of a
#   unit (reso, default ns) since 1970-01-01 UTC (already converted to local wall time
#   by callers), with the minimum int64 as the NaT marker. Each value is decomposed into
#   calendar fields (year, month, day, hour, minute, second, microsecond, picosecond)
#   via the pandas datetime C-API, and then a field, name, boolean flag or ISO calendar
#   triple is produced. Also timedelta component extraction, leap-year test,
#   month-position inference for frequency detection, localized day/month names, and
#   integer rounding of values to a multiple of a unit under several rounding modes.
# Side effects at import: initializes numpy and pandas datetime C-APIs.
"""
Functions for accessing attributes of Timestamp/datetime64/datetime-like
objects and arrays
"""
from locale import LC_TIME

from _strptime import LocaleTime

cimport cython
from cython cimport Py_ssize_t

import numpy as np

cimport numpy as cnp
from numpy cimport (
    int8_t,
    int32_t,
    int64_t,
    ndarray,
    uint32_t,
)

cnp.import_array()

from pandas._config.localization import set_locale

from pandas._libs.tslibs.ccalendar import (
    DAYS_FULL,
    MONTHS_FULL,
)

from pandas._libs.tslibs.ccalendar cimport (
    dayofweek,
    get_day_of_year,
    get_days_in_month,
    get_firstbday,
    get_iso_calendar,
    get_lastbday,
    get_week_of_year,
    iso_calendar_t,
)
from pandas._libs.tslibs.nattype cimport NPY_NAT
from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    NPY_FR_ns,
    import_pandas_datetime,
    npy_datetimestruct,
    pandas_datetime_to_datetimestruct,
    pandas_timedelta_to_timedeltastruct,
    pandas_timedeltastruct,
)

import_pandas_datetime()


@cython.wraparound(False)
@cython.boundscheck(False)
def build_field_sarray(const int64_t[:] dtindex, NPY_DATETIMEUNIT reso):
    # [FUNCTION DOC]
    # Decompose each int64 value into a structured array with int32 fields Y, M, D, h,
    #   m, s, u (microsecond). Inputs: dtindex - 1-D int64 values; reso - unit. Outputs:
    #   new structured ndarray of the same length. NaT is not special-cased (decomposed
    #   like any other integer). No bounds checks.
    """
    Datetime as int64 representation to a structured array of fields
    """
    cdef:
        Py_ssize_t i, count = len(dtindex)
        npy_datetimestruct dts
        ndarray[int32_t] years, months, days, hours, minutes, seconds, mus

    sa_dtype = [
        ("Y", "i4"),  # year
        ("M", "i4"),  # month
        ("D", "i4"),  # day
        ("h", "i4"),  # hour
        ("m", "i4"),  # min
        ("s", "i4"),  # second
        ("u", "i4"),  # microsecond
    ]

    out = np.empty(count, dtype=sa_dtype)

    years = out["Y"]
    months = out["M"]
    days = out["D"]
    hours = out["h"]
    minutes = out["m"]
    seconds = out["s"]
    mus = out["u"]

    for i in range(count):
        pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
        years[i] = dts.year
        months[i] = dts.month
        days[i] = dts.day
        hours[i] = dts.hour
        minutes[i] = dts.min
        seconds[i] = dts.sec
        mus[i] = dts.us

    return out


def month_position_check(fields, weekdays) -> str | None:
    # [FUNCTION DOC]
    # Infer whether all dates sit at a consistent month position, for frequency
    #   inference. Inputs: fields - structured array with Y, M, D int32 fields (as from
    #   build_field_sarray); weekdays - iterable of day-of-week ints (Monday=0), same
    #   length.
    # Tracks four flags over all rows: calendar start (day == 1), business start (day 1,
    #   or day <= 3 and Monday), calendar end (last day of month), business end (last
    #   day, or within the last 3 days and Friday). Stops early once all four are False.
    # Outputs: "ce", "be", "cs", "bs" (first true flag in that priority order) or None.
    #   Empty input returns "ce". Exceptions: ValueError if lengths differ (strict zip).
    cdef:
        int32_t daysinmonth, y, m, d
        bint calendar_end = True
        bint business_end = True
        bint calendar_start = True
        bint business_start = True
        bint cal
        int32_t[:] years = fields["Y"]
        int32_t[:] months = fields["M"]
        int32_t[:] days = fields["D"]

    for y, m, d, wd in zip(years, months, days, weekdays, strict=True):
        if calendar_start:
            calendar_start &= d == 1
        if business_start:
            business_start &= d == 1 or (d <= 3 and wd == 0)

        if calendar_end or business_end:
            daysinmonth = get_days_in_month(y, m)
            cal = d == daysinmonth
            if calendar_end:
                calendar_end &= cal
            if business_end:
                business_end &= cal or (daysinmonth - d < 3 and wd == 4)
        elif not calendar_start and not business_start:
            break

    if calendar_end:
        return "ce"
    elif business_end:
        return "be"
    elif calendar_start:
        return "cs"
    elif business_start:
        return "bs"
    else:
        return None


@cython.wraparound(False)
@cython.boundscheck(False)
def get_date_name_field(
    const int64_t[:] dtindex,
    str field,
    object locale=None,
    NPY_DATETIMEUNIT reso=NPY_FR_ns,
):
    # [FUNCTION DOC]
    # Day or month names for each value. Inputs: dtindex - 1-D int64; field - "day_name"
    #   or "month_name"; locale - None for English, else a locale string whose LC_TIME
    #   names are used; reso - unit. Outputs: object ndarray of capitalized names; NaT
    #   -> NaN float.
    # Exceptions: ValueError("Field ... not supported") for other fields; locale errors
    #   from set_locale. Side effects: temporarily switches the process LC_TIME locale
    #   when locale is given.
    """
    Given an int64-based datetime index, return array of strings of date
    name based on requested field (e.g. day_name)
    """
    cdef:
        Py_ssize_t i
        cnp.npy_intp count = dtindex.shape[0]
        ndarray[object] out, names
        npy_datetimestruct dts
        int dow

    out = cnp.PyArray_EMPTY(1, &count, cnp.NPY_OBJECT, 0)

    if field == "day_name":
        if locale is None:
            names = np.array(DAYS_FULL, dtype=np.object_)
        else:
            names = np.array(_get_locale_names("f_weekday", locale),
                             dtype=np.object_)
        for i in range(count):
            if dtindex[i] == NPY_NAT:
                out[i] = np.nan
                continue

            pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
            dow = dayofweek(dts.year, dts.month, dts.day)
            out[i] = names[dow].capitalize()

    elif field == "month_name":
        if locale is None:
            names = np.array(MONTHS_FULL, dtype=np.object_)
        else:
            names = np.array(_get_locale_names("f_month", locale),
                             dtype=np.object_)
        for i in range(count):
            if dtindex[i] == NPY_NAT:
                out[i] = np.nan
                continue

            pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
            out[i] = names[dts.month].capitalize()

    else:
        raise ValueError(f"Field {field} not supported")

    return out


cdef bint _is_on_month(int month, int compare_month, int modby) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Whether month matches an anchor month under a period. Inputs: month,
    #   compare_month - 1..12; modby - 1 (any month), 3 (same position within a
    #   quarter), otherwise exact match (yearly). Outputs: bint. Runs without the GIL.
    """
    Analogous to DateOffset.is_on_offset checking for the month part of a date.
    """
    if modby == 1:
        return True
    elif modby == 3:
        return (month - compare_month) % 3 == 0
    else:
        return month == compare_month


@cython.wraparound(False)
@cython.boundscheck(False)
def get_start_end_field(
    const int64_t[:] dtindex,
    str field,
    str freq_name=None,
    int month_kw=12,
    NPY_DATETIMEUNIT reso=NPY_FR_ns,
):
    # [FUNCTION DOC]
    # Boolean start/end flags for month, quarter or year. Inputs: dtindex - 1-D int64;
    #   field - one of is_month_start/end, is_quarter_start/end, is_year_start/end;
    #   freq_name - offset name of the index frequency or None; month_kw - anchor month
    #   of the offset (default 12); reso.
    # Anchors: with no freq_name, year ends in December and starts in January. With
    #   freq_name: "C" raises ValueError; a leading "B" means business-day semantics;
    #   names starting (after stripping leading B's) with QS or YS treat month_kw as the
    #   starting month (end = previous month); others treat month_kw as the ending month
    #   (start = next month). The compared month is the start anchor for "start" fields
    #   and the end anchor otherwise; period is 1 for month, 3 for quarter, 12 for year
    #   fields.
    # A value is flagged if the month matches and the day equals 1 (or the first
    #   business day) for start fields, or the last day of month (or last business day)
    #   for end fields. NaT -> False.
    # Outputs: bool ndarray. Exceptions: ValueError for unsupported field or "C" freq.
    """
    Given an int64-based datetime index return array of indicators
    of whether timestamps are at the start/end of the month/quarter/year
    (defined by frequency).

    Parameters
    ----------
    dtindex : ndarray[int64]
    field : str
    freq_name : str or None, default None
    month_kw : int, default 12
    reso : NPY_DATETIMEUNIT, default NPY_FR_ns

    Returns
    -------
    ndarray[bool]
    """
    cdef:
        Py_ssize_t i
        int count = dtindex.shape[0]
        bint is_business = 0
        int end_month = 12
        int start_month = 1
        ndarray[int8_t] out
        npy_datetimestruct dts
        int compare_month, modby

    out = np.zeros(count, dtype="int8")

    if freq_name:
        if freq_name == "C":
            raise ValueError(f"Custom business days is not supported by {field}")
        is_business = freq_name[0] == "B"

        # YearBegin(), BYearBegin() use month = starting month of year.
        # QuarterBegin(), BQuarterBegin() use startingMonth = starting
        # month of year. Other offsets use month, startingMonth as ending
        # month of year.

        if freq_name.lstrip("B")[0:2] in ["QS", "YS"]:
            end_month = 12 if month_kw == 1 else month_kw - 1
            start_month = month_kw

        else:
            end_month = month_kw
            start_month = (end_month % 12) + 1
    else:
        end_month = 12
        start_month = 1

    compare_month = start_month if "start" in field else end_month
    if "month" in field:
        modby = 1
    elif "quarter" in field:
        modby = 3
    else:
        modby = 12

    if field in ["is_month_start", "is_quarter_start", "is_year_start"]:
        if is_business:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = 0
                    continue

                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)

                if _is_on_month(dts.month, compare_month, modby) and (
                        dts.day == get_firstbday(dts.year, dts.month)):
                    out[i] = 1

        else:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = 0
                    continue

                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)

                if _is_on_month(dts.month, compare_month, modby) and dts.day == 1:
                    out[i] = 1

    elif field in ["is_month_end", "is_quarter_end", "is_year_end"]:
        if is_business:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = 0
                    continue

                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)

                if _is_on_month(dts.month, compare_month, modby) and (
                        dts.day == get_lastbday(dts.year, dts.month)):
                    out[i] = 1

        else:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = 0
                    continue

                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)

                if _is_on_month(dts.month, compare_month, modby) and (
                        dts.day == get_days_in_month(dts.year, dts.month)):
                    out[i] = 1

    else:
        raise ValueError(f"Field {field} not supported")

    return out.view(bool)


@cython.wraparound(False)
@cython.boundscheck(False)
def get_date_field(
    const int64_t[:] dtindex,
    str field,
    NPY_DATETIMEUNIT reso=NPY_FR_ns,
):
    # [FUNCTION DOC]
    # Extract one numeric calendar field per value as int32. Inputs: dtindex - 1-D
    #   int64; field - "Y", "M", "D", "h", "m", "s", "us" (microsecond), "ns"
    #   (nanosecond within the microsecond), "doy" (day of year, 1-based), "dow"
    #   (Monday=0), "woy" (ISO week), "q" (quarter 1-4), "dim" (days in month), or
    #   "is_leap_year"; reso.
    # Outputs: int32 ndarray with -1 for NaT; for "is_leap_year" a bool array (NaT ->
    #   False) computed from the years. Exceptions: ValueError for unknown fields. Side
    #   effects: releases the GIL in the main loops.
    """
    Given an int64-based datetime index, extract the year, month, etc.,
    field and return an array of these values.
    """
    cdef:
        Py_ssize_t i, count = len(dtindex)
        ndarray[int32_t] out
        npy_datetimestruct dts

    out = np.empty(count, dtype="i4")

    if field == "Y":
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
                out[i] = dts.year
        return out

    elif field == "M":
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
                out[i] = dts.month
        return out

    elif field == "D":
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
                out[i] = dts.day
        return out

    elif field == "h":
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
                out[i] = dts.hour
                # TODO: can we de-dup with period.pyx <accessor>s?
        return out

    elif field == "m":
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
                out[i] = dts.min
        return out

    elif field == "s":
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
                out[i] = dts.sec
        return out

    elif field == "us":
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
                out[i] = dts.us
        return out

    elif field == "ns":
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
                out[i] = dts.ps // 1000
        return out
    elif field == "doy":
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
                out[i] = get_day_of_year(dts.year, dts.month, dts.day)
        return out

    elif field == "dow":
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
                out[i] = dayofweek(dts.year, dts.month, dts.day)
        return out

    elif field == "woy":
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
                out[i] = get_week_of_year(dts.year, dts.month, dts.day)
        return out

    elif field == "q":
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
                out[i] = dts.month
                out[i] = ((out[i] - 1) // 3) + 1
        return out

    elif field == "dim":
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
                out[i] = get_days_in_month(dts.year, dts.month)
        return out
    elif field == "is_leap_year":
        return isleapyear_arr(get_date_field(dtindex, "Y", reso=reso))

    raise ValueError(f"Field {field} not supported")


@cython.wraparound(False)
@cython.boundscheck(False)
def get_timedelta_field(
    const int64_t[:] tdindex,
    str field,
    NPY_DATETIMEUNIT reso=NPY_FR_ns,
):
    # [FUNCTION DOC]
    # Extract a timedelta component per value as int32. Inputs: tdindex - 1-D int64
    #   durations; field - "seconds" (0-86399 within the day), "microseconds" (0-999999
    #   within the second), or "nanoseconds" (0-999 within the microsecond); reso.
    #   Outputs: int32 ndarray, -1 for NaT. Exceptions: ValueError for unknown field.
    #   Side effects: releases the GIL.
    """
    Given an int64-based timedelta index, extract the days, hrs, sec.,
    field and return an array of these values.
    """
    cdef:
        Py_ssize_t i, count = len(tdindex)
        ndarray[int32_t] out
        pandas_timedeltastruct tds

    out = np.empty(count, dtype="i4")

    if field == "seconds":
        with nogil:
            for i in range(count):
                if tdindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                pandas_timedelta_to_timedeltastruct(tdindex[i], reso, &tds)
                out[i] = tds.seconds
        return out

    elif field == "microseconds":
        with nogil:
            for i in range(count):
                if tdindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                pandas_timedelta_to_timedeltastruct(tdindex[i], reso, &tds)
                out[i] = tds.microseconds
        return out

    elif field == "nanoseconds":
        with nogil:
            for i in range(count):
                if tdindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                pandas_timedelta_to_timedeltastruct(tdindex[i], reso, &tds)
                out[i] = tds.nanoseconds
        return out

    raise ValueError(f"Field {field} not supported")


@cython.wraparound(False)
@cython.boundscheck(False)
def get_timedelta_days(
    const int64_t[:] tdindex,
    NPY_DATETIMEUNIT reso=NPY_FR_ns,
):
    # [FUNCTION DOC]
    # Whole days component (floored, can be negative) of each timedelta as int64.
    #   Inputs: tdindex - 1-D int64; reso. Outputs: int64 ndarray; NaT -> -1. Side
    #   effects: releases the GIL.
    # KNOWN ISSUE: NaT maps to -1, which is indistinguishable from a genuine duration of
    #   -1 day (callers must mask NaT themselves).
    """
    Given an int64-based timedelta index, extract the days,
    field and return an array of these values.
    """
    cdef:
        Py_ssize_t i, count = len(tdindex)
        ndarray[int64_t] out
        pandas_timedeltastruct tds

    out = np.empty(count, dtype="i8")

    with nogil:
        for i in range(count):
            if tdindex[i] == NPY_NAT:
                out[i] = -1
                continue

            pandas_timedelta_to_timedeltastruct(tdindex[i], reso, &tds)
            out[i] = tds.days
    return out


cpdef isleapyear_arr(ndarray years):
    # [FUNCTION DOC]
    # Vectorized Gregorian leap-year test. Inputs: years - integer ndarray. Outputs:
    #   bool ndarray, True where divisible by 400, or divisible by 4 and not by 100. A
    #   -1 NaT marker yields False.
    """vectorized version of isleapyear; NaT evaluates as False"""
    cdef:
        ndarray[int8_t] out

    out = np.zeros(len(years), dtype="int8")
    out[np.logical_or(years % 400 == 0,
                      np.logical_and(years % 4 == 0,
                                     years % 100 > 0))] = 1
    return out.view(bool)


@cython.wraparound(False)
@cython.boundscheck(False)
def build_isocalendar_sarray(const int64_t[:] dtindex, NPY_DATETIMEUNIT reso):
    # [FUNCTION DOC]
    # ISO 8601 calendar for each value as a structured array with uint32 fields year,
    #   week, day (Monday=1). Inputs: dtindex - 1-D int64; reso. Outputs: structured
    #   ndarray; NaT rows are (0, 0, 0). Side effects: releases the GIL.
    """
    Given an int64-based datetime array, return the ISO 8601 year, week, and day
    as a structured array.
    """
    cdef:
        Py_ssize_t i, count = len(dtindex)
        npy_datetimestruct dts
        ndarray[uint32_t] iso_years, iso_weeks, days
        iso_calendar_t ret_val

    sa_dtype = [
        ("year", "u4"),
        ("week", "u4"),
        ("day", "u4"),
    ]

    out = np.empty(count, dtype=sa_dtype)

    iso_years = out["year"]
    iso_weeks = out["week"]
    days = out["day"]

    with nogil:
        for i in range(count):
            if dtindex[i] == NPY_NAT:
                ret_val = 0, 0, 0
            else:
                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
                ret_val = get_iso_calendar(dts.year, dts.month, dts.day)

            iso_years[i] = ret_val[0]
            iso_weeks[i] = ret_val[1]
            days[i] = ret_val[2]
    return out


def _get_locale_names(name_type: str, locale: object = None):
    # [FUNCTION DOC]
    # HELPER: Localized day or month names. Inputs: name_type - LocaleTime attribute
    #   (e.g. "f_weekday", "f_month"); locale - locale string or None. Outputs: list of
    #   names (for months, index 0 is an empty string). Side effects: temporarily sets
    #   the process LC_TIME locale; raises if the locale is unavailable.
    """
    Returns an array of localized day or month names.

    Parameters
    ----------
    name_type : str
        Attribute of LocaleTime() in which to return localized names.
    locale : str

    Returns
    -------
    list of locale names
    """
    with set_locale(locale, LC_TIME):
        return getattr(LocaleTime(), name_type)


# ---------------------------------------------------------------------
# Rounding


class RoundTo:
    # [CLASS DOC]
    # Namespace of rounding mode codes: MINUS_INFTY (floor), PLUS_INFTY (ceil),
    #   NEAREST_HALF_EVEN, NEAREST_HALF_PLUS_INFTY, NEAREST_HALF_MINUS_INFTY.
    # KNOWN ISSUE: the codes are instance properties but are always accessed on the
    #   class, so RoundTo.X evaluates to the property object, not the integer the
    #   annotations promise; round_nsint64 works only because callers pass those same
    #   property objects and comparisons are by identity.
    """
    enumeration defining the available rounding modes

    Attributes
    ----------
    MINUS_INFTY
        round towards -∞, or floor [2]_
    PLUS_INFTY
        round towards +∞, or ceil [3]_
    NEAREST_HALF_EVEN
        round to nearest, tie-break half to even [6]_
    NEAREST_HALF_MINUS_INFTY
        round to nearest, tie-break half to -∞ [5]_
    NEAREST_HALF_PLUS_INFTY
        round to nearest, tie-break half to +∞ [4]_


    References
    ----------
    .. [1] "Rounding - Wikipedia"
           https://en.wikipedia.org/wiki/Rounding
    .. [2] "Rounding down"
           https://en.wikipedia.org/wiki/Rounding#Rounding_down
    .. [3] "Rounding up"
           https://en.wikipedia.org/wiki/Rounding#Rounding_up
    .. [4] "Round half up"
           https://en.wikipedia.org/wiki/Rounding#Round_half_up
    .. [5] "Round half down"
           https://en.wikipedia.org/wiki/Rounding#Round_half_down
    .. [6] "Round half to even"
           https://en.wikipedia.org/wiki/Rounding#Round_half_to_even
    """
    @property
    def MINUS_INFTY(self) -> int:
        # [FUNCTION DOC]
        # Mode code 0 (round toward minus infinity).
        return 0

    @property
    def PLUS_INFTY(self) -> int:
        # [FUNCTION DOC]
        # Mode code 1 (round toward plus infinity).
        return 1

    @property
    def NEAREST_HALF_EVEN(self) -> int:
        # [FUNCTION DOC]
        # Mode code 2 (round to nearest, ties to even multiple).
        return 2

    @property
    def NEAREST_HALF_PLUS_INFTY(self) -> int:
        # [FUNCTION DOC]
        # Mode code 3 (round to nearest, ties toward plus infinity).
        return 3

    @property
    def NEAREST_HALF_MINUS_INFTY(self) -> int:
        # [FUNCTION DOC]
        # Mode code 4 (round to nearest, ties toward minus infinity).
        return 4


cdef ndarray[int64_t] _floor_int64(const int64_t[:] values, int64_t unit):
    # [FUNCTION DOC]
    # HELPER: Floor each value to a multiple of unit (value minus floor-modulo). Inputs:
    #   values - 1-D int64; unit - positive int64. Outputs: new int64 ndarray; NaT
    #   preserved. Exceptions: OverflowError on int64 overflow.
    cdef:
        Py_ssize_t i, n = len(values)
        ndarray[int64_t] result = np.empty(n, dtype="i8")
        int64_t res, value

    with cython.overflowcheck(True):
        for i in range(n):
            value = values[i]
            if value == NPY_NAT:
                res = NPY_NAT
            else:
                res = value - value % unit
            result[i] = res

    return result


cdef ndarray[int64_t] _ceil_int64(const int64_t[:] values, int64_t unit):
    # [FUNCTION DOC]
    # HELPER: Ceil each value to a multiple of unit (unchanged if already a multiple,
    #   else add unit minus remainder). Inputs/outputs as the floor helper; NaT
    #   preserved. Exceptions: OverflowError on overflow.
    cdef:
        Py_ssize_t i, n = len(values)
        ndarray[int64_t] result = np.empty(n, dtype="i8")
        int64_t res, value, remainder

    with cython.overflowcheck(True):
        for i in range(n):
            value = values[i]

            if value == NPY_NAT:
                res = NPY_NAT
            else:
                remainder = value % unit
                if remainder == 0:
                    res = value
                else:
                    res = value + (unit - remainder)

            result[i] = res

    return result


cdef ndarray[int64_t] _rounddown_int64(values, int64_t unit):
    # [FUNCTION DOC]
    # HELPER: Round to nearest multiple of unit with ties toward minus infinity:
    #   subtracts half a unit (floor of unit/2) then ceils. Inputs: values - int64
    #   array-like; unit. Outputs: new int64 ndarray; NaT preserved. Exceptions:
    #   OverflowError on overflow. With an odd unit there are no ties, so this is exact
    #   nearest rounding.
    cdef:
        Py_ssize_t i, n = len(values)
        ndarray[int64_t] result = np.empty(n, dtype="i8")
        int64_t res, value, remainder, half

    half = unit // 2

    with cython.overflowcheck(True):
        for i in range(n):
            value = values[i]

            if value == NPY_NAT:
                res = NPY_NAT
            else:
                # This adjustment is the only difference between rounddown_int64
                #  and _ceil_int64
                value = value - half
                remainder = value % unit
                if remainder == 0:
                    res = value
                else:
                    res = value + (unit - remainder)

            result[i] = res
    return result


cdef ndarray[int64_t] _roundup_int64(values, int64_t unit):
    # [FUNCTION DOC]
    # HELPER: Round to nearest multiple of unit with ties toward plus infinity: adds
    #   half a unit (floor of unit/2) to the whole array, then floors.
    # KNOWN ISSUE: unlike the other helpers NaT is not preserved (half a unit is added
    #   to the NaT marker before flooring, producing a non-NaT value) and the array
    #   addition can wrap around on overflow without error; callers must mask NaT
    #   afterwards.
    return _floor_int64(values + unit // 2, unit)


cdef ndarray[int64_t] _round_nearest_int64(const int64_t[:] values, int64_t unit):
    # [FUNCTION DOC]
    # HELPER: Round to nearest multiple of unit with ties to the even multiple: split
    #   into quotient and floor-remainder; round up if remainder > half, or if remainder
    #   == half and the quotient is odd; else round down. Inputs: values - 1-D int64;
    #   unit (even, as called). Outputs: new int64 ndarray; NaT preserved. Exceptions:
    #   OverflowError on overflow.
    cdef:
        Py_ssize_t i, n = len(values)
        ndarray[int64_t] result = np.empty(n, dtype="i8")
        int64_t res, value, half, remainder, quotient

    half = unit // 2

    with cython.overflowcheck(True):
        for i in range(n):
            value = values[i]

            if value == NPY_NAT:
                res = NPY_NAT
            else:
                quotient, remainder = divmod(value, unit)
                if remainder > half:
                    res = value + (unit - remainder)
                elif remainder == half and quotient % 2:
                    res = value + (unit - remainder)
                else:
                    res = value - remainder

            result[i] = res

    return result


def round_nsint64(values: np.ndarray, mode: RoundTo, nanos: int) -> np.ndarray:
    # [FUNCTION DOC]
    # Round int64 values to a multiple of nanos according to mode. Inputs: values -
    #   int64 ndarray; mode - one of the RoundTo codes; nanos - the multiple, in the
    #   same unit as values (not necessarily nanoseconds despite the name). Dispatches
    #   to floor, ceil, half-down, half-up, or half-even (odd units use half-down since
    #   ties cannot occur).
    # Outputs: new int64 ndarray. Exceptions: ValueError for unknown mode; OverflowError
    #   from the helpers.
    """
    Applies rounding mode at given frequency

    Parameters
    ----------
    values : np.ndarray[int64_t]`
    mode : instance of `RoundTo` enumeration
    nanos : np.int64
        Freq to round to, expressed in nanoseconds

    Returns
    -------
    np.ndarray[int64_t]
    """
    cdef:
        int64_t unit = nanos

    if mode == RoundTo.MINUS_INFTY:
        return _floor_int64(values, unit)
    elif mode == RoundTo.PLUS_INFTY:
        return _ceil_int64(values, unit)
    elif mode == RoundTo.NEAREST_HALF_MINUS_INFTY:
        return _rounddown_int64(values, unit)
    elif mode == RoundTo.NEAREST_HALF_PLUS_INFTY:
        return _roundup_int64(values, unit)
    elif mode == RoundTo.NEAREST_HALF_EVEN:
        # for odd unit there is no need of a tie break
        if unit % 2:
            return _rounddown_int64(values, unit)
        return _round_nearest_int64(values, unit)

    # if/elif above should catch all rounding modes defined in enum 'RoundTo':
    # if flow of control arrives here, it is a bug
    raise ValueError("round_nsint64 called with an unrecognized rounding mode")
