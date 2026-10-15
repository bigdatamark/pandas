# [MODULE DOC]
# Period scalar and the integer 'period ordinal' arithmetic behind Period, PeriodArray
#   and PeriodIndex.
# Representation: a period is (ordinal, freq). freq is an integer PeriodDtypeCode
#   (dtypes.pyx) whose thousands part is the frequency group (FR_ANN=1000, FR_QTR=2000,
#   FR_MTH=3000, FR_WK=4000, FR_BUS=5000, FR_DAY=6000, FR_HR=7000, FR_MIN=8000,
#   FR_SEC=9000, FR_MS=10000, FR_US=11000, FR_NS=12000) and whose remainder is the
#   anchor (annual/quarterly: fiscal year-end month, 0 meaning December; weekly: end
#   weekday, 0=Sunday .. 6=Saturday). The ordinal counts periods since the epoch 1970:
#   year-1970 for annual (fiscal year labeled by the calendar year it ends in),
#   (year-1970)*4+q-1 for quarterly, (year-1970)*12+m-1 monthly, weeks counted so that
#   the week containing 1970-01-01 is 1-ish (see unix_date_to_week), business days
#   counted as 5 per 7-day week, and for daily and finer the number of units since
#   1970-01-01T00:00 (same as numpy datetime64 integer values). NPY_NAT (int64 min) is
#   the missing-value marker and propagates everywhere.
# Frequency conversion: get_asfreq_func picks a C function pointer
#   freq_conv_func(ordinal, asfreq_info*) for a (from, to) pair; most non-daily pairs go
#   through a daily (unix_date) intermediate. asfreq_info carries is_end (when
#   upsampling, pick the last sub-period instead of the first), from_end/to_end (anchor
#   month 1..12 or weekday code of source/target) and intraday_conversion_factor (ratio
#   of the coarser to the finer unit among day..ns, from a 7x7 lookup table).
#   Conversions are noexcept nogil and do no overflow checking.
# Also provides vectorized helpers used by pandas.core.arrays.period
#   (periodarr_to_dt64arr, period_asfreq_arr, get_period_field_arr,
#   period_array_strftime, extract_ordinals, extract_freq, from_ordinals), formatting
#   (period_format with extra directives %q %f %F %l %u %n), the IncompatibleFrequency
#   exception, PeriodMixin, the cdef base _Period and the public Period class with its
#   constructor parsing logic.

import re

cimport numpy as cnp
from cpython.object cimport (
    Py_EQ,
    Py_NE,
    PyObject,
    PyObject_RichCompare,
    PyObject_RichCompareBool,
)
from numpy cimport (
    int32_t,
    int64_t,
    ndarray,
)

import numpy as np

cnp.import_array()

cimport cython
from cpython.datetime cimport (
    PyDate_Check,
    PyDateTime_Check,
    datetime,
    import_datetime,
)
from libc.stdlib cimport (
    free,
    malloc,
)
from libc.string cimport (
    memset,
    strlen,
)
from libc.time cimport (
    strftime,
    tm,
)

from pandas._libs.tslibs.dtypes cimport (
    PeriodDtypeCode,
    c_OFFSET_TO_PERIOD_FREQSTR,
)

from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime

# import datetime C API
import_datetime()

cimport pandas._libs.tslibs.util as util
from pandas._libs.missing cimport C_NA
from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    NPY_FR_D,
    astype_overflowsafe,
    dts_to_iso_string,
    import_pandas_datetime,
    npy_datetimestruct,
    npy_datetimestruct_to_datetime,
    pandas_datetime_to_datetimestruct,
)

import_pandas_datetime()

from pandas._libs.tslibs.timestamps import Timestamp

from pandas._libs.tslibs.ccalendar cimport (
    dayofweek,
    get_day_of_year,
    get_days_in_month,
    get_week_of_year,
    is_leapyear,
)
from pandas._libs.tslibs.dtypes cimport (
    FR_ANN,
    FR_BUS,
    FR_DAY,
    FR_HR,
    FR_MIN,
    FR_MS,
    FR_MTH,
    FR_NS,
    FR_QTR,
    FR_SEC,
    FR_UND,
    FR_US,
    FR_WK,
    PeriodDtypeBase,
    attrname_to_abbrevs,
    freq_group_code_to_npy_unit,
)
from pandas._libs.tslibs.parsing cimport quarter_to_myear
from pandas._libs.tslibs.timedeltas cimport (
    delta_to_nanoseconds,
    is_any_td_scalar,
)

from pandas._libs.tslibs.parsing import parse_datetime_string_with_reso

from pandas._libs.tslibs.nattype cimport (
    NPY_NAT,
    c_NaT as NaT,
    c_nat_strings as nat_strings,
    checknull_with_nat,
)
from pandas._libs.tslibs.offsets cimport (
    BaseOffset,
    is_offset_object,
    to_offset,
)

from pandas._libs.tslibs.offsets import (
    INVALID_FREQ_ERR_MSG,
    BDay,
    Day,
)
from pandas.util._decorators import set_module

cdef:
    enum:
        INT32_MIN = -2_147_483_648LL


ctypedef struct asfreq_info:
    # [CLASS DOC]
    # Plain C struct describing one frequency conversion. intraday_conversion_factor:
    #   number of finer units per coarser unit between the source and target (both
    #   clamped to at least daily), 0 if either is coarser than daily index; is_end: 1
    #   to map to the end of the source period when upsampling, 0 for the start; to_end
    #   / from_end: target / source anchor (fiscal year-end month 1..12 for
    #   annual/quarterly, end weekday code 0..6 for weekly); left unset for other
    #   groups.
    int64_t intraday_conversion_factor
    int is_end
    int to_end
    int from_end

ctypedef int64_t (*freq_conv_func)(int64_t, asfreq_info*) noexcept nogil


cdef extern from *:
    """
    // must use npy typedef b/c int64_t is aliased in cython-generated c
    // unclear why we need LL for that row.
    // see https://github.com/pandas-dev/pandas/pull/34416/
    static npy_int64 daytime_conversion_factor_matrix[7][7] = {
        {1, 24, 1440, 86400, 86400000, 86400000000, 86400000000000},
        {0LL,  1LL,   60LL,  3600LL,  3600000LL,  3600000000LL,  3600000000000LL},
        {0,  0,   1,     60,    60000,    60000000,    60000000000},
        {0,  0,   0,      1,     1000,     1000000,     1000000000},
        {0,  0,   0,      0,        1,        1000,        1000000},
        {0,  0,   0,      0,        0,           1,           1000},
        {0,  0,   0,      0,        0,           0,              1}};
    """
    int64_t daytime_conversion_factor_matrix[7][7]


cdef int max_value(int left, int right) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Return the larger of two C ints.
    # Inputs: left, right - int. Outputs: int.
    # Side effects: none; runs without the GIL.
    if left > right:
        return left
    return right


cdef int min_value(int left, int right) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Return the smaller of two C ints.
    # Inputs: left, right - int. Outputs: int.
    # Side effects: none; runs without the GIL.
    if left < right:
        return left
    return right


cdef int64_t get_daytime_conversion_factor(int from_index, int to_index) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Look up how many units of the finer frequency make one unit of the coarser
    #   one, for frequency group indices (freq // 1000) 6 (day) .. 12 (ns). Uses row=min
    #   index, col=max index into the 7x7 upper-triangular table
    #   daytime_conversion_factor_matrix offset by 6.
    # Inputs: from_index, to_index - group indices (order irrelevant).
    # Outputs: int64 factor (1 when equal, e.g. 86400 for day->second); 0 if either
    #   index is below 6.
    # Side effects: none; runs without the GIL.
    cdef:
        int row = min_value(from_index, to_index)
        int col = max_value(from_index, to_index)
    # row or col < 6 means frequency strictly lower than Daily, which
    # do not use daytime_conversion_factors
    if row < 6:
        return 0
    elif col < 6:
        return 0
    return daytime_conversion_factor_matrix[row - 6][col - 6]


cdef int64_t nofunc(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Sentinel conversion for unsupported frequency pairs; ignores its inputs
    #   and returns INT32_MIN (-2**31).
    # Inputs: ordinal, af_info - unused. Outputs: int64 -2147483648.
    # Side effects: none; runs without the GIL.
    return INT32_MIN


cdef int64_t no_op(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Identity conversion; returns ordinal unchanged.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: ordinal.
    # Side effects: none; runs without the GIL.
    return ordinal


cdef freq_conv_func get_asfreq_func(int from_freq, int to_freq) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Select the conversion function pointer for from_freq -> to_freq based on
    #   their frequency groups (freq rounded down to a multiple of 1000; an undefined
    #   source group FR_UND is treated as daily). Business source takes precedence, then
    #   business target, then source group
    #   annual/quarterly/monthly/weekly/daily-or-finer, each dispatching on the target
    #   group to the matching asfreq_XtoY function. Intraday-to-intraday uses
    #   downsample_daytime when the source group is finer than the target (numerically
    #   larger) else upsample_daytime. Business->business and monthly->monthly use
    #   no_op; other same-group pairs (A->A, Q->Q, W->W) go via daily so that anchors
    #   may differ. Unrecognized pairs return nofunc.
    # Inputs: from_freq, to_freq - PeriodDtypeCode ints.
    # Outputs: freq_conv_func pointer.
    # Side effects: none; runs without the GIL.
    cdef:
        int from_group = get_freq_group(from_freq)
        int to_group = get_freq_group(to_freq)

    if from_group == FR_UND:
        from_group = FR_DAY

    if from_group == FR_BUS:
        if to_group == FR_ANN:
            return <freq_conv_func>asfreq_BtoA
        elif to_group == FR_QTR:
            return <freq_conv_func>asfreq_BtoQ
        elif to_group == FR_MTH:
            return <freq_conv_func>asfreq_BtoM
        elif to_group == FR_WK:
            return <freq_conv_func>asfreq_BtoW
        elif to_group == FR_BUS:
            return <freq_conv_func>no_op
        elif to_group in [FR_DAY, FR_HR, FR_MIN, FR_SEC, FR_MS, FR_US, FR_NS]:
            return <freq_conv_func>asfreq_BtoDT
        else:
            return <freq_conv_func>nofunc

    elif to_group == FR_BUS:
        if from_group == FR_ANN:
            return <freq_conv_func>asfreq_AtoB
        elif from_group == FR_QTR:
            return <freq_conv_func>asfreq_QtoB
        elif from_group == FR_MTH:
            return <freq_conv_func>asfreq_MtoB
        elif from_group == FR_WK:
            return <freq_conv_func>asfreq_WtoB
        elif from_group in [FR_DAY, FR_HR, FR_MIN, FR_SEC, FR_MS, FR_US, FR_NS]:
            return <freq_conv_func>asfreq_DTtoB
        else:
            return <freq_conv_func>nofunc

    elif from_group == FR_ANN:
        if to_group == FR_ANN:
            return <freq_conv_func>asfreq_AtoA
        elif to_group == FR_QTR:
            return <freq_conv_func>asfreq_AtoQ
        elif to_group == FR_MTH:
            return <freq_conv_func>asfreq_AtoM
        elif to_group == FR_WK:
            return <freq_conv_func>asfreq_AtoW
        elif to_group in [FR_DAY, FR_HR, FR_MIN, FR_SEC, FR_MS, FR_US, FR_NS]:
            return <freq_conv_func>asfreq_AtoDT
        else:
            return <freq_conv_func>nofunc

    elif from_group == FR_QTR:
        if to_group == FR_ANN:
            return <freq_conv_func>asfreq_QtoA
        elif to_group == FR_QTR:
            return <freq_conv_func>asfreq_QtoQ
        elif to_group == FR_MTH:
            return <freq_conv_func>asfreq_QtoM
        elif to_group == FR_WK:
            return <freq_conv_func>asfreq_QtoW
        elif to_group in [FR_DAY, FR_HR, FR_MIN, FR_SEC, FR_MS, FR_US, FR_NS]:
            return <freq_conv_func>asfreq_QtoDT
        else:
            return <freq_conv_func>nofunc

    elif from_group == FR_MTH:
        if to_group == FR_ANN:
            return <freq_conv_func>asfreq_MtoA
        elif to_group == FR_QTR:
            return <freq_conv_func>asfreq_MtoQ
        elif to_group == FR_MTH:
            return <freq_conv_func>no_op
        elif to_group == FR_WK:
            return <freq_conv_func>asfreq_MtoW
        elif to_group in [FR_DAY, FR_HR, FR_MIN, FR_SEC, FR_MS, FR_US, FR_NS]:
            return <freq_conv_func>asfreq_MtoDT
        else:
            return <freq_conv_func>nofunc

    elif from_group == FR_WK:
        if to_group == FR_ANN:
            return <freq_conv_func>asfreq_WtoA
        elif to_group == FR_QTR:
            return <freq_conv_func>asfreq_WtoQ
        elif to_group == FR_MTH:
            return <freq_conv_func>asfreq_WtoM
        elif to_group == FR_WK:
            return <freq_conv_func>asfreq_WtoW
        elif to_group in [FR_DAY, FR_HR, FR_MIN, FR_SEC, FR_MS, FR_US, FR_NS]:
            return <freq_conv_func>asfreq_WtoDT
        else:
            return <freq_conv_func>nofunc

    elif from_group in [FR_DAY, FR_HR, FR_MIN, FR_SEC, FR_MS, FR_US, FR_NS]:
        if to_group == FR_ANN:
            return <freq_conv_func>asfreq_DTtoA
        elif to_group == FR_QTR:
            return <freq_conv_func>asfreq_DTtoQ
        elif to_group == FR_MTH:
            return <freq_conv_func>asfreq_DTtoM
        elif to_group == FR_WK:
            return <freq_conv_func>asfreq_DTtoW
        elif to_group in [FR_DAY, FR_HR, FR_MIN, FR_SEC, FR_MS, FR_US, FR_NS]:
            if from_group > to_group:
                return <freq_conv_func>downsample_daytime
            else:
                return <freq_conv_func>upsample_daytime

        else:
            return <freq_conv_func>nofunc

    else:
        return <freq_conv_func>nofunc


# --------------------------------------------------------------------
# Frequency Conversion Helpers

cdef int64_t DtoB_weekday(int64_t unix_date) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Convert a unix date (days since 1970-01-01, assumed to fall on a weekday)
    #   to a business-day ordinal by counting 5 business days per 7-day week, aligned so
    #   weeks start on Monday (1970-01-01 is a Thursday, hence the +4/-4 shifts); uses
    #   floor division.
    # Inputs: unix_date - int64 day count. Outputs: int64 business-day ordinal.
    # Side effects: none; runs without the GIL.
    return ((unix_date + 4) // 7) * 5 + ((unix_date + 4) % 7) - 4


cdef int64_t DtoB(npy_datetimestruct *dts, int roll_back,
                  int64_t unix_date) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Convert a date to a business-day ordinal, first moving a weekend date to a
    #   weekday: if roll_back == 1 a Saturday/Sunday moves back to the preceding Friday,
    #   otherwise forward to the following Monday. Weekday is computed from dts
    #   (Monday=0).
    # Inputs: dts - broken-down date matching unix_date (only year/month/day read);
    #   roll_back - 1 to roll back, anything else rolls forward; unix_date - days since
    #   epoch.
    # Outputs: business-day ordinal (via DtoB_weekday).
    # Side effects: none; runs without the GIL.
    # calculate the current week (counting from 1970-01-01) treating
    # sunday as last day of a week
    cdef:
        int day_of_week = dayofweek(dts.year, dts.month, dts.day)

    if roll_back == 1:
        if day_of_week > 4:
            # change to friday before weekend
            unix_date -= (day_of_week - 4)
    else:
        if day_of_week > 4:
            # change to Monday after weekend
            unix_date += (7 - day_of_week)

    return DtoB_weekday(unix_date)


cdef int64_t upsample_daytime(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Convert an ordinal at a coarser day-or-finer unit to a finer one by
    #   multiplying by af_info.intraday_conversion_factor. If af_info.is_end returns the
    #   last finer unit in the period: (ordinal+1)*factor-1; otherwise ordinal*factor.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: int64 ordinal in the finer unit (no overflow check).
    # Side effects: none; runs without the GIL.
    if af_info.is_end:
        return (ordinal + 1) * af_info.intraday_conversion_factor - 1
    else:
        return ordinal * af_info.intraday_conversion_factor


cdef int64_t downsample_daytime(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Convert an ordinal at a finer day-or-finer unit to a coarser one via floor
    #   division by af_info.intraday_conversion_factor.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: int64 coarser ordinal. Exceptions: none (factor is >= 1 for valid
    #   pairs; a 0 factor would be undefined).
    # Side effects: none; runs without the GIL.
    return ordinal // af_info.intraday_conversion_factor


cdef int64_t transform_via_day(int64_t ordinal,
                               asfreq_info *af_info,
                               freq_conv_func first_func,
                               freq_conv_func second_func) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Compose two conversions: result = second_func(first_func(ordinal,
    #   af_info), af_info).
    # Inputs: ordinal; af_info shared by both calls; first_func, second_func -
    #   conversion function pointers. Outputs: int64 result.
    # Side effects: none; runs without the GIL.
    cdef:
        int64_t result

    result = first_func(ordinal, af_info)
    result = second_func(result, af_info)
    return result


# --------------------------------------------------------------------
# Conversion _to_ Daily Freq

cdef int64_t asfreq_AtoDT(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Annual ordinal -> day-or-finer ordinal. Adds is_end to the ordinal (so the
    #   end case targets the start of the next fiscal year), sets year=ordinal+1970,
    #   month=1, shifts to the fiscal-year start month with
    #   adjust_dts_for_month(from_end), takes the unix date of day 1 of that month,
    #   subtracts is_end (last day of the original year), then upsample_daytime.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: day/intraday ordinal at the start or end of the year.
    # Side effects: none; runs without the GIL.
    cdef:
        int64_t unix_date
        npy_datetimestruct dts

    ordinal += af_info.is_end

    dts.year = ordinal + 1970
    dts.month = 1
    adjust_dts_for_month(&dts, af_info.from_end)

    unix_date = unix_date_from_ymd(dts.year, dts.month, 1)
    unix_date -= af_info.is_end
    return upsample_daytime(unix_date, af_info)


cdef int64_t asfreq_QtoDT(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Quarterly ordinal -> day-or-finer ordinal. Same scheme as asfreq_AtoDT
    #   with year = ordinal//4 + 1970 and month = (ordinal%4)*3+1 (floor semantics),
    #   then adjusted for the fiscal year end from_end.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: day/intraday ordinal at the start or end of the quarter.
    # Side effects: none; runs without the GIL.
    cdef:
        int64_t unix_date
        npy_datetimestruct dts

    ordinal += af_info.is_end

    dts.year = ordinal // 4 + 1970
    dts.month = (ordinal % 4) * 3 + 1
    adjust_dts_for_month(&dts, af_info.from_end)

    unix_date = unix_date_from_ymd(dts.year, dts.month, 1)
    unix_date -= af_info.is_end
    return upsample_daytime(unix_date, af_info)


cdef int64_t asfreq_MtoDT(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Monthly ordinal -> day-or-finer ordinal: (ordinal+is_end) gives
    #   year=//12+1970 and month=%12+1; take day 1 of that month as a unix date,
    #   subtract is_end, then upsample_daytime.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: day/intraday ordinal at the start or end of the month.
    # Side effects: none; runs without the GIL.
    cdef:
        int64_t unix_date
        int year, month

    ordinal += af_info.is_end

    year = ordinal // 12 + 1970
    month = ordinal % 12 + 1

    unix_date = unix_date_from_ymd(year, month, 1)
    unix_date -= af_info.is_end
    return upsample_daytime(unix_date, af_info)


cdef int64_t asfreq_WtoDT(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Weekly ordinal -> day-or-finer ordinal. The end day of week ordinal w is
    #   unix date 7*w + from_end - 4; for the start (is_end=0) 6 days are subtracted.
    #   Then upsample_daytime.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: day/intraday ordinal at the start or end of the week.
    # Side effects: none; runs without the GIL.
    ordinal = (ordinal * 7 + af_info.from_end - 4 +
               (7 - 1) * (af_info.is_end - 1))
    return upsample_daytime(ordinal, af_info)


# --------------------------------------------------------------------
# Conversion _to_ BusinessDay Freq

cdef int64_t asfreq_AtoB(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Annual -> business day: compute the daily ordinal with asfreq_AtoDT
    #   (af_info factor is that of daily), decompose it, and call DtoB rolling back to
    #   Friday when is_end, forward to Monday otherwise.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: business-day ordinal.
    # Side effects: none; runs without the GIL.
    cdef:
        int roll_back
        npy_datetimestruct dts
        int64_t unix_date = asfreq_AtoDT(ordinal, af_info)

    pandas_datetime_to_datetimestruct(unix_date, NPY_FR_D, &dts)
    roll_back = af_info.is_end
    return DtoB(&dts, roll_back, unix_date)


cdef int64_t asfreq_QtoB(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Quarterly -> business day; as asfreq_AtoB but via asfreq_QtoDT.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: business-day ordinal.
    # Side effects: none; runs without the GIL.
    cdef:
        int roll_back
        npy_datetimestruct dts
        int64_t unix_date = asfreq_QtoDT(ordinal, af_info)

    pandas_datetime_to_datetimestruct(unix_date, NPY_FR_D, &dts)
    roll_back = af_info.is_end
    return DtoB(&dts, roll_back, unix_date)


cdef int64_t asfreq_MtoB(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Monthly -> business day; as asfreq_AtoB but via asfreq_MtoDT.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: business-day ordinal.
    # Side effects: none; runs without the GIL.
    cdef:
        int roll_back
        npy_datetimestruct dts
        int64_t unix_date = asfreq_MtoDT(ordinal, af_info)

    pandas_datetime_to_datetimestruct(unix_date, NPY_FR_D, &dts)
    roll_back = af_info.is_end
    return DtoB(&dts, roll_back, unix_date)


cdef int64_t asfreq_WtoB(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Weekly -> business day; as asfreq_AtoB but via asfreq_WtoDT.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: business-day ordinal.
    # Side effects: none; runs without the GIL.
    cdef:
        int roll_back
        npy_datetimestruct dts
        int64_t unix_date = asfreq_WtoDT(ordinal, af_info)

    pandas_datetime_to_datetimestruct(unix_date, NPY_FR_D, &dts)
    roll_back = af_info.is_end
    return DtoB(&dts, roll_back, unix_date)


cdef int64_t asfreq_DTtoB(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Day-or-finer -> business day: downsample to a unix date, then DtoB with
    #   roll_back = 1 - is_end (start: weekend rolls back to Friday; end: rolls forward
    #   to Monday; the opposite convention from the other XtoB).
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: business-day ordinal.
    # Side effects: none; runs without the GIL.
    cdef:
        int roll_back
        npy_datetimestruct dts
        int64_t unix_date = downsample_daytime(ordinal, af_info)

    pandas_datetime_to_datetimestruct(unix_date, NPY_FR_D, &dts)
    # This usage defines roll_back the opposite way from the others
    roll_back = 1 - af_info.is_end
    return DtoB(&dts, roll_back, unix_date)


# ----------------------------------------------------------------------
# Conversion _from_ Daily Freq

cdef int64_t asfreq_DTtoA(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Day-or-finer -> annual: downsample to a unix date, decompose, and return
    #   dts_to_year_ordinal with the target fiscal year end to_end.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: annual ordinal.
    # Side effects: none; runs without the GIL.
    cdef:
        npy_datetimestruct dts

    ordinal = downsample_daytime(ordinal, af_info)
    pandas_datetime_to_datetimestruct(ordinal, NPY_FR_D, &dts)
    return dts_to_year_ordinal(&dts, af_info.to_end)


cdef int DtoQ_yq(int64_t ordinal, asfreq_info *af_info,
                 npy_datetimestruct* dts) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Decompose unix date ordinal into dts, shift it into fiscal terms with
    #   adjust_dts_for_qtr(to_end) (dts.year becomes the fiscal year), and return the
    #   fiscal quarter 1..4.
    # Inputs: ordinal - unix date; af_info - uses to_end; dts - output struct.
    # Outputs: quarter int. Side effects: overwrites *dts; nogil.
    cdef:
        int quarter

    pandas_datetime_to_datetimestruct(ordinal, NPY_FR_D, dts)
    adjust_dts_for_qtr(dts, af_info.to_end)

    quarter = month_to_quarter(dts.month)
    return quarter


cdef int64_t asfreq_DTtoQ(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Day-or-finer -> quarterly: downsample to a unix date, compute fiscal
    #   year/quarter with DtoQ_yq, return (year-1970)*4 + quarter-1.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: quarterly ordinal.
    # Side effects: none; runs without the GIL.
    cdef:
        int quarter
        npy_datetimestruct dts

    ordinal = downsample_daytime(ordinal, af_info)

    quarter = DtoQ_yq(ordinal, af_info, &dts)
    return <int64_t>((dts.year - 1970) * 4 + quarter - 1)


cdef int64_t asfreq_DTtoM(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Day-or-finer -> monthly: downsample to a unix date, decompose, return
    #   (year-1970)*12 + month-1.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: monthly ordinal.
    # Side effects: none; runs without the GIL.
    cdef:
        npy_datetimestruct dts

    ordinal = downsample_daytime(ordinal, af_info)
    pandas_datetime_to_datetimestruct(ordinal, NPY_FR_D, &dts)
    return dts_to_month_ordinal(&dts)


cdef int64_t asfreq_DTtoW(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Day-or-finer -> weekly: downsample to a unix date and return
    #   unix_date_to_week(unix_date, to_end).
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: weekly ordinal.
    # Side effects: none; runs without the GIL.
    ordinal = downsample_daytime(ordinal, af_info)
    return unix_date_to_week(ordinal, af_info.to_end)


cdef int64_t unix_date_to_week(int64_t unix_date, int to_end) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Weekly ordinal of the week (ending on weekday code to_end,
    #   0=Sunday..6=Saturday) containing unix_date: floor((unix_date + 3 - to_end) / 7)
    #   + 1. Inverse of the end-day formula in asfreq_WtoDT.
    # Inputs: unix_date - days since epoch; to_end - end weekday code.
    # Outputs: int64 weekly ordinal.
    # Side effects: none; runs without the GIL.
    return (unix_date + 3 - to_end) // 7 + 1


# --------------------------------------------------------------------
# Conversion _from_ BusinessDay Freq

cdef int64_t asfreq_BtoDT(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Business-day ordinal -> unix date (5 business days per 7-day week, inverse
    #   of DtoB_weekday), then upsample_daytime to the target unit.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: day-or-finer ordinal.
    # Side effects: none; runs without the GIL.
    ordinal = ((ordinal + 3) // 5) * 7 + (ordinal + 3) % 5 - 3
    return upsample_daytime(ordinal, af_info)


cdef int64_t asfreq_BtoA(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Convert a business-day ordinal to annual by going business-day -> daily
    #   (with asfreq_BtoDT) -> annual (with asfreq_DTtoA) via transform_via_day, using
    #   the same af_info for both steps.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: target ordinal.
    # Side effects: none; runs without the GIL.
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_BtoDT,
                             <freq_conv_func>asfreq_DTtoA)


cdef int64_t asfreq_BtoQ(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Convert a business-day ordinal to quarterly by going business-day -> daily
    #   (with asfreq_BtoDT) -> quarterly (with asfreq_DTtoQ) via transform_via_day,
    #   using the same af_info for both steps.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: target ordinal.
    # Side effects: none; runs without the GIL.
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_BtoDT,
                             <freq_conv_func>asfreq_DTtoQ)


cdef int64_t asfreq_BtoM(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Convert a business-day ordinal to monthly by going business-day -> daily
    #   (with asfreq_BtoDT) -> monthly (with asfreq_DTtoM) via transform_via_day, using
    #   the same af_info for both steps.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: target ordinal.
    # Side effects: none; runs without the GIL.
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_BtoDT,
                             <freq_conv_func>asfreq_DTtoM)


cdef int64_t asfreq_BtoW(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Convert a business-day ordinal to weekly by going business-day -> daily
    #   (with asfreq_BtoDT) -> weekly (with asfreq_DTtoW) via transform_via_day, using
    #   the same af_info for both steps.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: target ordinal.
    # Side effects: none; runs without the GIL.
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_BtoDT,
                             <freq_conv_func>asfreq_DTtoW)


# ----------------------------------------------------------------------
# Conversion _from_ Annual Freq

cdef int64_t asfreq_AtoA(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Convert a annual ordinal to annual (possibly different year-end) by going
    #   annual -> daily (with asfreq_AtoDT) -> annual (possibly different year-end)
    #   (with asfreq_DTtoA) via transform_via_day, using the same af_info for both
    #   steps.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: target ordinal.
    # Side effects: none; runs without the GIL.
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_AtoDT,
                             <freq_conv_func>asfreq_DTtoA)


cdef int64_t asfreq_AtoQ(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Convert a annual ordinal to quarterly by going annual -> daily (with
    #   asfreq_AtoDT) -> quarterly (with asfreq_DTtoQ) via transform_via_day, using the
    #   same af_info for both steps.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: target ordinal.
    # Side effects: none; runs without the GIL.
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_AtoDT,
                             <freq_conv_func>asfreq_DTtoQ)


cdef int64_t asfreq_AtoM(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Convert a annual ordinal to monthly by going annual -> daily (with
    #   asfreq_AtoDT) -> monthly (with asfreq_DTtoM) via transform_via_day, using the
    #   same af_info for both steps.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: target ordinal.
    # Side effects: none; runs without the GIL.
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_AtoDT,
                             <freq_conv_func>asfreq_DTtoM)


cdef int64_t asfreq_AtoW(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Convert a annual ordinal to weekly by going annual -> daily (with
    #   asfreq_AtoDT) -> weekly (with asfreq_DTtoW) via transform_via_day, using the
    #   same af_info for both steps.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: target ordinal.
    # Side effects: none; runs without the GIL.
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_AtoDT,
                             <freq_conv_func>asfreq_DTtoW)


# ----------------------------------------------------------------------
# Conversion _from_ Quarterly Freq

cdef int64_t asfreq_QtoQ(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Convert a quarterly ordinal to quarterly (possibly different year-end) by
    #   going quarterly -> daily (with asfreq_QtoDT) -> quarterly (possibly different
    #   year-end) (with asfreq_DTtoQ) via transform_via_day, using the same af_info for
    #   both steps.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: target ordinal.
    # Side effects: none; runs without the GIL.
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_QtoDT,
                             <freq_conv_func>asfreq_DTtoQ)


cdef int64_t asfreq_QtoA(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Convert a quarterly ordinal to annual by going quarterly -> daily (with
    #   asfreq_QtoDT) -> annual (with asfreq_DTtoA) via transform_via_day, using the
    #   same af_info for both steps.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: target ordinal.
    # Side effects: none; runs without the GIL.
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_QtoDT,
                             <freq_conv_func>asfreq_DTtoA)


cdef int64_t asfreq_QtoM(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Convert a quarterly ordinal to monthly by going quarterly -> daily (with
    #   asfreq_QtoDT) -> monthly (with asfreq_DTtoM) via transform_via_day, using the
    #   same af_info for both steps.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: target ordinal.
    # Side effects: none; runs without the GIL.
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_QtoDT,
                             <freq_conv_func>asfreq_DTtoM)


cdef int64_t asfreq_QtoW(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Convert a quarterly ordinal to weekly by going quarterly -> daily (with
    #   asfreq_QtoDT) -> weekly (with asfreq_DTtoW) via transform_via_day, using the
    #   same af_info for both steps.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: target ordinal.
    # Side effects: none; runs without the GIL.
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_QtoDT,
                             <freq_conv_func>asfreq_DTtoW)


# ----------------------------------------------------------------------
# Conversion _from_ Monthly Freq

cdef int64_t asfreq_MtoA(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Convert a monthly ordinal to annual by going monthly -> daily (with
    #   asfreq_MtoDT) -> annual (with asfreq_DTtoA) via transform_via_day, using the
    #   same af_info for both steps.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: target ordinal.
    # Side effects: none; runs without the GIL.
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_MtoDT,
                             <freq_conv_func>asfreq_DTtoA)


cdef int64_t asfreq_MtoQ(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Convert a monthly ordinal to quarterly by going monthly -> daily (with
    #   asfreq_MtoDT) -> quarterly (with asfreq_DTtoQ) via transform_via_day, using the
    #   same af_info for both steps.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: target ordinal.
    # Side effects: none; runs without the GIL.
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_MtoDT,
                             <freq_conv_func>asfreq_DTtoQ)


cdef int64_t asfreq_MtoW(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Convert a monthly ordinal to weekly by going monthly -> daily (with
    #   asfreq_MtoDT) -> weekly (with asfreq_DTtoW) via transform_via_day, using the
    #   same af_info for both steps.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: target ordinal.
    # Side effects: none; runs without the GIL.
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_MtoDT,
                             <freq_conv_func>asfreq_DTtoW)


# ----------------------------------------------------------------------
# Conversion _from_ Weekly Freq

cdef int64_t asfreq_WtoA(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Convert a weekly ordinal to annual by going weekly -> daily (with
    #   asfreq_WtoDT) -> annual (with asfreq_DTtoA) via transform_via_day, using the
    #   same af_info for both steps.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: target ordinal.
    # Side effects: none; runs without the GIL.
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_WtoDT,
                             <freq_conv_func>asfreq_DTtoA)


cdef int64_t asfreq_WtoQ(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Convert a weekly ordinal to quarterly by going weekly -> daily (with
    #   asfreq_WtoDT) -> quarterly (with asfreq_DTtoQ) via transform_via_day, using the
    #   same af_info for both steps.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: target ordinal.
    # Side effects: none; runs without the GIL.
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_WtoDT,
                             <freq_conv_func>asfreq_DTtoQ)


cdef int64_t asfreq_WtoM(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Convert a weekly ordinal to monthly by going weekly -> daily (with
    #   asfreq_WtoDT) -> monthly (with asfreq_DTtoM) via transform_via_day, using the
    #   same af_info for both steps.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: target ordinal.
    # Side effects: none; runs without the GIL.
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_WtoDT,
                             <freq_conv_func>asfreq_DTtoM)


cdef int64_t asfreq_WtoW(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Convert a weekly ordinal to weekly (possibly different end weekday) by
    #   going weekly -> daily (with asfreq_WtoDT) -> weekly (possibly different end
    #   weekday) (with asfreq_DTtoW) via transform_via_day, using the same af_info for
    #   both steps.
    # Inputs: ordinal - source period ordinal (not NaT); af_info - conversion info.
    #   Outputs: target ordinal.
    # Side effects: none; runs without the GIL.
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_WtoDT,
                             <freq_conv_func>asfreq_DTtoW)


# ----------------------------------------------------------------------

@cython.cdivision
cdef char* c_strftime(npy_datetimestruct *dts, char *fmt):
    # [FUNCTION DOC]
    # HELPER: Format a broken-down datetime with the C library strftime. Fills a C tm
    #   struct from dts (sec, min, hour, mday, mon-1, year-1900, weekday converted to
    #   Sunday=0, yday 0-based, isdst=-1; sub-second fields ignored) and writes into a
    #   new buffer of strlen(fmt)+50 bytes.
    # Inputs: dts - pointer to the datetime struct; fmt - NUL-terminated byte format in
    #   the current locale encoding.
    # Outputs: newly malloc'd NUL-terminated char buffer; the caller must free it.
    # Exceptions: MemoryError if malloc fails.
    # Side effects: allocates heap memory; output depends on the C locale.
    # KNOWN ISSUE: the strftime return value is not checked; if the formatted result
    #   needs more than strlen(fmt)+50 bytes strftime returns 0 and the buffer contents
    #   are indeterminate (not guaranteed NUL-terminated).
    """
    Generate a nice string representation of the period
    object, originally from DateObject_strftime

    Parameters
    ----------
    dts : npy_datetimestruct*
    fmt : char*

    Returns
    -------
    result : char*
    """
    cdef:
        tm c_date
        char *result
        int result_len = strlen(fmt) + 50

    c_date.tm_sec = dts.sec
    c_date.tm_min = dts.min
    c_date.tm_hour = dts.hour
    c_date.tm_mday = dts.day
    c_date.tm_mon = dts.month - 1
    c_date.tm_year = dts.year - 1900
    c_date.tm_wday = (dayofweek(dts.year, dts.month, dts.day) + 1) % 7
    c_date.tm_yday = get_day_of_year(dts.year, dts.month, dts.day) - 1
    c_date.tm_isdst = -1

    result = <char*>malloc(result_len)
    if result is NULL:
        raise MemoryError()

    strftime(result, result_len, fmt, &c_date)

    return result


# ----------------------------------------------------------------------
# Conversion between date_info and npy_datetimestruct

cdef int get_freq_group(int freq) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Frequency group code of a PeriodDtypeCode: (freq // 1000) * 1000.
    # Inputs: freq - int. Outputs: int group (e.g. 2000 for any quarterly).
    # Side effects: none; runs without the GIL.
    # See also FreqGroup.get_freq_group
    return (freq // 1000) * 1000


cdef int get_freq_group_index(int freq) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Frequency group index freq // 1000 (1=annual .. 12=nanosecond).
    # Inputs: freq - int. Outputs: int.
    # Side effects: none; runs without the GIL.
    return freq // 1000


cdef void adjust_dts_for_month(npy_datetimestruct* dts, int from_end) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Turn (fiscal year Y, month 1) into the calendar start month of that fiscal
    #   year for year-end month from_end: if from_end != 12, month += from_end; if that
    #   exceeds 12 subtract 12, otherwise decrement the year. For from_end == 12 nothing
    #   changes.
    # Inputs: dts - year/month read and modified; from_end - year-end month 1..12.
    # Outputs: none. Side effects: mutates dts.year and dts.month; nogil.
    if from_end != 12:
        dts.month += from_end
        if dts.month > 12:
            dts.month -= 12
        else:
            dts.year -= 1


cdef void adjust_dts_for_qtr(npy_datetimestruct* dts, int to_end) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Convert a calendar year/month into fiscal terms for a fiscal year ending
    #   in month to_end: if to_end != 12, month -= to_end; if the result is <= 0 add 12,
    #   otherwise increment the year (so dts.year becomes the fiscal year label and
    #   month is the 1-based month within the fiscal year).
    # Inputs: dts - year/month read and modified; to_end - year-end month 1..12.
    # Outputs: none. Side effects: mutates dts.year and dts.month; nogil.
    if to_end != 12:
        dts.month -= to_end
        if dts.month <= 0:
            dts.month += 12
        else:
            dts.year += 1


# Find the unix_date (days elapsed since datetime(1970, 1, 1)
# for the given year/month/day.
# Assumes GREGORIAN_CALENDAR */
cdef int64_t unix_date_from_ymd(int year, int month, int day) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Days since 1970-01-01 of a proleptic Gregorian year/month/day, via a
    #   zeroed datetime struct and numpy's struct-to-datetime at day unit.
    # Inputs: year, month (1-12), day (1-31). Outputs: int64 unix date.
    # Side effects: none; runs without the GIL.
    # Calculate the absolute date
    cdef:
        npy_datetimestruct dts
        int64_t unix_date

    memset(&dts, 0, sizeof(npy_datetimestruct))
    dts.year = year
    dts.month = month
    dts.day = day
    unix_date = npy_datetimestruct_to_datetime(NPY_FR_D, &dts)
    return unix_date


cdef int64_t dts_to_month_ordinal(npy_datetimestruct* dts) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Monthly period ordinal (year-1970)*12 + month-1 of dts.
    # Inputs: dts - pointer, year/month read. Outputs: int64.
    # Side effects: none; runs without the GIL.
    # AKA: use npy_datetimestruct_to_datetime(NPY_FR_M, &dts)
    return <int64_t>((dts.year - 1970) * 12 + dts.month - 1)


cdef int64_t dts_to_year_ordinal(npy_datetimestruct *dts, int to_end) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Annual period ordinal for a fiscal year ending in month to_end: year-1970,
    #   plus 1 if dts.month > to_end (date belongs to the fiscal year ending next
    #   calendar year).
    # Inputs: dts - pointer (year/month used); to_end - year-end month 1..12.
    # Outputs: int64.
    # Side effects: none; runs without the GIL.
    cdef:
        int64_t result

    result = npy_datetimestruct_to_datetime(NPY_DATETIMEUNIT.NPY_FR_Y, dts)
    if dts.month > to_end:
        return result + 1
    else:
        return result


cdef int64_t dts_to_qtr_ordinal(npy_datetimestruct* dts, int to_end) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Quarterly period ordinal for a fiscal year ending in month to_end: applies
    #   adjust_dts_for_qtr then returns (fiscal year-1970)*4 + quarter-1.
    # Inputs: dts - pointer; to_end - year-end month.
    # Outputs: int64. Side effects: mutates dts year/month; nogil.
    cdef:
        int quarter

    adjust_dts_for_qtr(dts, to_end)
    quarter = month_to_quarter(dts.month)
    return <int64_t>((dts.year - 1970) * 4 + quarter - 1)


cdef int get_anchor_month(int freq, int freq_group) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Fiscal year-end month encoded in an annual/quarterly freq code: freq -
    #   freq_group, with 0 mapped to 12 (December).
    # Inputs: freq - code; freq_group - its group. Outputs: int 1..12.
    # Side effects: none; runs without the GIL.
    cdef:
        int fmonth
    fmonth = freq - freq_group
    if fmonth == 0:
        fmonth = 12
    return fmonth


# specifically _dont_ use cdvision or else ordinals near -1 are assigned to
# incorrect dates GH#19643
@cython.cdivision(False)
cdef int64_t get_period_ordinal(npy_datetimestruct *dts, int freq) noexcept nogil:
    # [FUNCTION DOC]
    # Compute the period ordinal at frequency freq containing the datetime dts. Annual
    #   -> dts_to_year_ordinal with the anchor month; quarterly -> dts_to_qtr_ordinal;
    #   weekly -> unix date then unix_date_to_week with end weekday freq-4000; business
    #   (exactly 5000) -> DtoB rolling weekends forward to Monday; otherwise (monthly,
    #   daily, intraday) numpy struct-to-datetime at the unit mapped from the frequency
    #   group. Python floor-division semantics are used (cdivision disabled) so negative
    #   ordinals are correct.
    # Inputs: dts - pointer to datetime struct (quarterly path mutates it); freq -
    #   PeriodDtypeCode.
    # Outputs: int64 ordinal.
    # Exceptions: none (noexcept). Side effects: may mutate dts; nogil.
    """
    Generate an ordinal in period space

    Parameters
    ----------
    dts : npy_datetimestruct*
    freq : int

    Returns
    -------
    period_ordinal : int64_t
    """
    cdef:
        int64_t unix_date
        int freq_group, fmonth
        NPY_DATETIMEUNIT unit

    freq_group = get_freq_group(freq)

    if freq_group == FR_ANN:
        fmonth = get_anchor_month(freq, freq_group)
        return dts_to_year_ordinal(dts, fmonth)

    elif freq_group == FR_QTR:
        fmonth = get_anchor_month(freq, freq_group)
        return dts_to_qtr_ordinal(dts, fmonth)

    elif freq_group == FR_WK:
        unix_date = npy_datetimestruct_to_datetime(NPY_FR_D, dts)
        return unix_date_to_week(unix_date, freq - FR_WK)

    elif freq == FR_BUS:
        unix_date = npy_datetimestruct_to_datetime(NPY_FR_D, dts)
        return DtoB(dts, 0, unix_date)

    unit = freq_group_code_to_npy_unit(freq)
    return npy_datetimestruct_to_datetime(unit, dts)


cdef void get_date_info(int64_t ordinal,
                        int freq, npy_datetimestruct *dts) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Fill dts with the date/time of a period's representative instant: the date
    #   is from get_unix_date (end day for coarser-than-daily periods), and
    #   hour/min/sec/us/ps come from get_time_nanos (start of an intraday period; zero
    #   for daily and coarser).
    # Inputs: ordinal - period ordinal; freq - code; dts - output pointer.
    # Outputs: none. Side effects: overwrites *dts; nogil.
    cdef:
        int64_t unix_date, nanos
        npy_datetimestruct dts2

    unix_date = get_unix_date(ordinal, freq)
    nanos = get_time_nanos(freq, unix_date, ordinal)

    pandas_datetime_to_datetimestruct(unix_date, NPY_FR_D, dts)

    pandas_datetime_to_datetimestruct(nanos, NPY_DATETIMEUNIT.NPY_FR_ns, &dts2)
    dts.hour = dts2.hour
    dts.min = dts2.min
    dts.sec = dts2.sec
    dts.us = dts2.us
    dts.ps = dts2.ps


cdef int64_t get_unix_date(int64_t period_ordinal, int freq) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Days since 1970-01-01 of a period. Daily returns the ordinal; otherwise
    #   converts to daily with is_end=True, i.e. the last day of coarser periods and the
    #   containing day (floor) for intraday periods.
    # Inputs: period_ordinal - int64; freq - code. Outputs: int64 unix date.
    # Side effects: none; runs without the GIL.
    """
    Returns the proleptic Gregorian ordinal of the date, as an integer.
    This corresponds to the number of days since Jan., 1st, 1970 AD.
    When the instance has a frequency less than daily, the proleptic date
    is calculated for the last day of the period.

    Parameters
    ----------
    period_ordinal : int64_t
    freq : int

    Returns
    -------
    unix_date : int64_t number of days since datetime(1970, 1, 1)
    """
    cdef:
        asfreq_info af_info
        freq_conv_func toDaily = NULL

    if freq == FR_DAY:
        return period_ordinal

    toDaily = get_asfreq_func(freq, FR_DAY)
    get_asfreq_info(freq, FR_DAY, True, &af_info)
    return toDaily(period_ordinal, &af_info)


@cython.cdivision
cdef int64_t get_time_nanos(int freq, int64_t unix_date,
                            int64_t ordinal) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Nanoseconds after midnight of unix_date represented by an intraday
    #   ordinal: (ordinal - unix_date * units_per_day) * unit_nanos. Returns 0 for daily
    #   and coarser groups; an unknown group above minute is treated as hourly.
    # Inputs: freq - code; unix_date - containing day; ordinal - period ordinal in freq
    #   units.
    # Outputs: int64 nanoseconds in [0, 86400e9).
    # Side effects: none; runs without the GIL.
    """
    Find the number of nanoseconds after midnight on the given unix_date
    that the ordinal represents in the given frequency.

    Parameters
    ----------
    freq : int
    unix_date : int64_t
    ordinal : int64_t

    Returns
    -------
    int64_t
    """
    cdef:
        int64_t sub, factor
        int64_t nanos_in_day = 24 * 3600 * 10**9

    freq = get_freq_group(freq)

    if freq <= FR_DAY:
        return 0

    elif freq == FR_NS:
        factor = 1

    elif freq == FR_US:
        factor = 10**3

    elif freq == FR_MS:
        factor = 10**6

    elif freq == FR_SEC:
        factor = 10 **9

    elif freq == FR_MIN:
        factor = 10**9 * 60

    else:
        # We must have freq == FR_HR
        factor = 10**9 * 3600

    sub = ordinal - unix_date * (nanos_in_day / factor)
    return sub * factor


cdef int get_yq(int64_t ordinal, int freq, npy_datetimestruct* dts):
    # [FUNCTION DOC]
    # HELPER: Fiscal year and quarter of a period: takes its unix date (end day for
    #   coarse periods), and converts with DtoQ_yq using freq's own anchor if freq is
    #   quarterly, else the December year-end quarterly code.
    # Inputs: ordinal; freq; dts - output pointer.
    # Outputs: quarter 1..4; dts.year is the fiscal year, dts.month the month within the
    #   fiscal year; other dts fields are from the date only.
    # Exceptions: none in practice (declared to propagate Python errors).
    # Side effects: overwrites *dts.
    """
    Find the year and quarter of a Period with the given ordinal and frequency

    Parameters
    ----------
    ordinal : int64_t
    freq : int
    dts : *npy_datetimestruct

    Returns
    -------
    quarter : int
        describes the implied quarterly frequency associated with `freq`

    Notes
    -----
    Sets dts.year in-place.
    """
    cdef:
        asfreq_info af_info
        int qtr_freq
        int64_t unix_date
        int quarter

    unix_date = get_unix_date(ordinal, freq)

    if get_freq_group(freq) == FR_QTR:
        qtr_freq = freq
    else:
        qtr_freq = FR_QTR

    get_asfreq_info(FR_DAY, qtr_freq, True, &af_info)

    quarter = DtoQ_yq(unix_date, &af_info, dts)
    return quarter


cdef int month_to_quarter(int month) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Map month 1..12 to quarter 1..4: (month-1)//3 + 1.
    # Inputs: month - int. Outputs: int.
    # Side effects: none; runs without the GIL.
    return (month - 1) // 3 + 1


# ----------------------------------------------------------------------
# Period logic

@cython.wraparound(False)
@cython.boundscheck(False)
def periodarr_to_dt64arr(const int64_t[:] periodarr, int freq):
    # [FUNCTION DOC]
    # Convert an array of period ordinals to datetime64 values at each period's
    #   representative instant. For frequencies coarser than daily (code < 6000: annual,
    #   quarterly, monthly, weekly, business) each element goes through
    #   period_ordinal_to_dt64 into a new int64 array viewed as M8[us]. For nanosecond
    #   freq the input buffer is reinterpreted as M8[ns] without copy. For
    #   daily..microsecond the input is viewed with the matching unit and cast to M8[us]
    #   with overflow checking.
    # Inputs: periodarr - 1-D int64 ordinals (NaT allowed); freq - code.
    # Outputs: 1-D datetime64[us] ndarray (datetime64[ns] for nanosecond freq, sharing
    #   memory with the input).
    # Exceptions: OutOfBoundsDatetime if a value does not fit.
    # Side effects: allocates output for most paths.
    # Note: for coarse freqs get_date_info yields the LAST day of the period, so callers
    #   first asfreq to daily start ordinals.
    # KNOWN ISSUE: an unrecognized freq >= 6000 (none exist today) would leave dta
    #   unbound and raise UnboundLocalError.
    """
    Convert array to datetime64 values from a set of ordinals corresponding to
    periods per period convention.
    """
    cdef:
        int64_t[::1] out
        Py_ssize_t i, N

    if freq < 6000:  # i.e. FR_DAY, hard-code to avoid need to cast
        N = len(periodarr)
        out = np.empty(N, dtype="i8")

        # We get here with freqs that do not correspond to a datetime64 unit
        for i in range(N):
            out[i] = period_ordinal_to_dt64(periodarr[i], freq)

        return out.base.view("M8[us]")  # .base to access underlying np.ndarray

    else:
        # Short-circuit for performance
        if freq == FR_NS:
            # TODO: copy?
            return periodarr.base.view("M8[ns]")

        if freq == FR_US:
            dta = periodarr.base.view("M8[us]")
        elif freq == FR_MS:
            dta = periodarr.base.view("M8[ms]")
        elif freq == FR_SEC:
            dta = periodarr.base.view("M8[s]")
        elif freq == FR_MIN:
            dta = periodarr.base.view("M8[m]")
        elif freq == FR_HR:
            dta = periodarr.base.view("M8[h]")
        elif freq == FR_DAY:
            dta = periodarr.base.view("M8[D]")
        # GH#63760 give microseconds for everything other than freq="ns"
        return astype_overflowsafe(dta, dtype=np.dtype("M8[us]"))


cdef void get_asfreq_info(int from_freq, int to_freq,
                          bint is_end, asfreq_info *af_info) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Populate af_info for converting from_freq -> to_freq: sets is_end;
    #   intraday_conversion_factor from the group indices with each group clamped to at
    #   least daily; from_end / to_end set from the anchor (weekly: end weekday via
    #   calc_week_end; annual and quarterly: year-end month via calc_a_year_end) only
    #   for those groups, other fields left untouched.
    # Inputs: from_freq, to_freq - codes; is_end - bool; af_info - output.
    # Outputs: none. Side effects: writes *af_info; nogil.
    """
    Construct the `asfreq_info` object used to convert an ordinal from
    `from_freq` to `to_freq`.

    Parameters
    ----------
    from_freq : int
    to_freq int
    is_end : bool
    af_info : *asfreq_info
    """
    cdef:
        int from_group = get_freq_group(from_freq)
        int to_group = get_freq_group(to_freq)

    af_info.is_end = is_end

    af_info.intraday_conversion_factor = get_daytime_conversion_factor(
        get_freq_group_index(max_value(from_group, FR_DAY)),
        get_freq_group_index(max_value(to_group, FR_DAY)))

    if from_group == FR_WK:
        af_info.from_end = calc_week_end(from_freq, from_group)
    elif from_group == FR_ANN:
        af_info.from_end = calc_a_year_end(from_freq, from_group)
    elif from_group == FR_QTR:
        af_info.from_end = calc_a_year_end(from_freq, from_group)

    if to_group == FR_WK:
        af_info.to_end = calc_week_end(to_freq, to_group)
    elif to_group == FR_ANN:
        af_info.to_end = calc_a_year_end(to_freq, to_group)
    elif to_group == FR_QTR:
        af_info.to_end = calc_a_year_end(to_freq, to_group)


@cython.cdivision
cdef int calc_a_year_end(int freq, int group) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Year-end month from an annual/quarterly code: (freq-group) % 12, with 0
    #   mapped to 12.
    # Inputs: freq, group. Outputs: int 1..12.
    # Side effects: none; runs without the GIL.
    cdef:
        int result = (freq - group) % 12
    if result == 0:
        return 12
    else:
        return result


cdef int calc_week_end(int freq, int group) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: End-weekday code of a weekly frequency: freq - group (0=Sunday).
    # Inputs: freq, group. Outputs: int 0..6.
    # Side effects: none; runs without the GIL.
    return freq - group


cpdef int64_t period_asfreq(int64_t ordinal, int freq1, int freq2, bint end):
    # [FUNCTION DOC]
    # Convert one period ordinal between frequencies; when upsampling, end selects the
    #   last (True) or first (False) sub-period. NaT maps to NaT.
    # Inputs: ordinal; freq1 - source code; freq2 - target code; end - bool.
    # Outputs: int64 target ordinal (INT32_MIN for unsupported pairs).
    # Exceptions/side effects: none.
    """
    Convert period ordinal from one frequency to another, and if upsampling,
    choose to use start ('S') or end ('E') of period.
    """
    cdef:
        int64_t retval

    _period_asfreq(&ordinal, &retval, 1, freq1, freq2, end)
    return retval


@cython.wraparound(False)
@cython.boundscheck(False)
def period_asfreq_arr(ndarray[int64_t] arr, int freq1, int freq2, bint end):
    # [FUNCTION DOC]
    # Vectorized period_asfreq over an int64 array of any shape: reads the input with a
    #   stride derived from its first-axis stride and writes a new array of the same
    #   shape. NaT preserved.
    # Inputs: arr - int64 ndarray; freq1, freq2 - codes; end - bool.
    # Outputs: new int64 ndarray, same shape.
    # Side effects: allocates the result.
    # KNOWN ISSUE: the flat loop uses only arr.strides[0]//8 as the element step and
    #   len(arr) as the count, so for multi-dimensional or non-contiguous inputs only
    #   the first-axis elements are converted correctly; callers pass 1-D arrays.
    """
    Convert int64-array of period ordinals from one frequency to another, and
    if upsampling, choose to use start ('S') or end ('E') of period.
    """
    cdef:
        Py_ssize_t n = len(arr)
        Py_ssize_t increment = arr.strides[0] // 8
        ndarray[int64_t] result = cnp.PyArray_EMPTY(
            arr.ndim, arr.shape, cnp.NPY_INT64, 0
        )

    _period_asfreq(
        <int64_t*>cnp.PyArray_DATA(arr),
        <int64_t*>cnp.PyArray_DATA(result),
        n,
        freq1,
        freq2,
        end,
        increment,
    )
    return result


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _period_asfreq(
    int64_t* ordinals,
    int64_t* out,
    Py_ssize_t length,
    int freq1,
    int freq2,
    bint end,
    Py_ssize_t increment=1,
) noexcept:
    # [FUNCTION DOC]
    # HELPER: Core loop for period_asfreq(_arr). Fast path: a single NaT input writes
    #   NaT without looking up a function. Otherwise builds the conversion function and
    #   af_info once, then for i in [0, length) reads ordinals[i*increment], converts it
    #   unless NaT, and writes out[i].
    # Inputs: ordinals - source buffer; out - destination buffer (length elements,
    #   contiguous); length; freq1, freq2; end; increment - input element stride
    #   (default 1).
    # Outputs: none. Side effects: writes out. Holds the GIL but raises nothing.
    """See period_asfreq.__doc__"""
    cdef:
        Py_ssize_t i
        freq_conv_func func
        asfreq_info af_info
        int64_t val

    if length == 1 and ordinals[0] == NPY_NAT:
        # fastpath avoid calling get_asfreq_func
        out[0] = NPY_NAT
        return

    func = get_asfreq_func(freq1, freq2)
    get_asfreq_info(freq1, freq2, end, &af_info)

    for i in range(length):
        val = ordinals[i * increment]
        if val != NPY_NAT:
            val = func(val, &af_info)
        out[i] = val


cpdef int64_t period_ordinal(int y, int m, int d, int h, int min,
                             int s, int us, int ps, int freq):
    # [FUNCTION DOC]
    # Ordinal at frequency freq of the period containing the datetime given by
    #   components (year, month, day, hour, minute, second, microsecond, picosecond) via
    #   get_period_ordinal.
    # Inputs: y, m, d, h, min, s, us, ps - ints; freq - code.
    # Outputs: int64 ordinal. Exceptions/side effects: none (no validation).
    """
    Find the ordinal representation of the given datetime components at the
    frequency `freq`.

    Parameters
    ----------
    y : int
    m : int
    d : int
    h : int
    min : int
    s : int
    us : int
    ps : int

    Returns
    -------
    ordinal : int64_t
    """
    cdef:
        npy_datetimestruct dts
    dts.year = y
    dts.month = m
    dts.day = d
    dts.hour = h
    dts.min = min
    dts.sec = s
    dts.us = us
    dts.ps = ps
    return get_period_ordinal(&dts, freq)


cdef int64_t period_ordinal_to_dt64(int64_t ordinal, int freq) except? -1:
    # [FUNCTION DOC]
    # HELPER: Convert one period ordinal to an int64 datetime64 value. NaT -> NaT;
    #   nanosecond freq returns the ordinal unchanged (already ns). Otherwise fills a
    #   struct via get_date_info and converts to MICROseconds since epoch.
    # Inputs: ordinal; freq - code.
    # Outputs: int64 (us units, or ns for nanosecond freq).
    # Exceptions: OutOfBoundsDatetime if out of microsecond range.
    cdef:
        npy_datetimestruct dts
        int64_t result

    if ordinal == NPY_NAT:
        return NPY_NAT

    if freq == PeriodDtypeCode.N:
        # We have to return nanosecond unit, but this is a no-op
        return ordinal

    get_date_info(ordinal, freq, &dts)

    try:
        result = npy_datetimestruct_to_datetime(NPY_DATETIMEUNIT.NPY_FR_us, &dts)
    except OverflowError as err:
        fmt = dts_to_iso_string(&dts)
        raise OutOfBoundsDatetime(
            f"Out of bounds microsecond timestamp: {fmt}"
        ) from err

    return result


cdef str period_format(int64_t value, int freq, object fmt=None):
    # [FUNCTION DOC]
    # Format a period ordinal as a string. NaT -> 'NaT'. With fmt None (or equal to the
    #   group's default pattern) uses fast f-string defaults: annual 'YYYY'; quarterly
    #   'YYYYQq' using fiscal year; monthly 'YYYY-MM'; weekly (fmt None only)
    #   'start/end' days in YYYY-MM-DD each; business/daily 'YYYY-MM-DD'; hourly
    #   'YYYY-MM-DD HH:00'; minute adds ':MM'; second adds ':SS'; ms/us/ns add
    #   '.fff'/'.ffffff'/'.fffffffff'. Otherwise encodes a str fmt in the locale
    #   encoding and delegates to _period_strftime.
    # Inputs: value - ordinal; freq - code; fmt - None, str or bytes.
    # Outputs: str.
    # Exceptions: ValueError('Unknown freq') for an unknown group with fmt None.
    # Side effects: none (strftime path allocates and frees a C buffer).

    cdef:
        int freq_group, quarter
        npy_datetimestruct dts
        bint is_fmt_none

    if value == NPY_NAT:
        return "NaT"

    # fill dts and freq group
    get_date_info(value, freq, &dts)
    freq_group = get_freq_group(freq)

    # use the appropriate default format depending on frequency group
    is_fmt_none = fmt is None
    if freq_group == FR_ANN and (is_fmt_none or fmt == "%Y"):
        return f"{dts.year}"

    elif freq_group == FR_QTR and (is_fmt_none or fmt == "%FQ%q"):
        # get quarter and modify dts.year to be the 'Fiscal' year
        quarter = get_yq(value, freq, &dts)
        return f"{dts.year}Q{quarter}"

    elif freq_group == FR_MTH and (is_fmt_none or fmt == "%Y-%m"):
        return f"{dts.year}-{dts.month:02d}"

    elif freq_group == FR_WK and is_fmt_none:
        # special: start_date/end_date. Recurse
        left = period_asfreq(value, freq, FR_DAY, 0)
        right = period_asfreq(value, freq, FR_DAY, 1)
        return f"{period_format(left, FR_DAY)}/{period_format(right, FR_DAY)}"

    elif (
        (freq_group == FR_BUS or freq_group == FR_DAY)
        and (is_fmt_none or fmt == "%Y-%m-%d")
    ):
        return f"{dts.year}-{dts.month:02d}-{dts.day:02d}"

    elif freq_group == FR_HR and (is_fmt_none or fmt == "%Y-%m-%d %H:00"):
        return f"{dts.year}-{dts.month:02d}-{dts.day:02d} {dts.hour:02d}:00"

    elif freq_group == FR_MIN and (is_fmt_none or fmt == "%Y-%m-%d %H:%M"):
        return (f"{dts.year}-{dts.month:02d}-{dts.day:02d} "
                f"{dts.hour:02d}:{dts.min:02d}")

    elif freq_group == FR_SEC and (is_fmt_none or fmt == "%Y-%m-%d %H:%M:%S"):
        return (f"{dts.year}-{dts.month:02d}-{dts.day:02d} "
                f"{dts.hour:02d}:{dts.min:02d}:{dts.sec:02d}")

    elif freq_group == FR_MS and (is_fmt_none or fmt == "%Y-%m-%d %H:%M:%S.%l"):
        return (f"{dts.year}-{dts.month:02d}-{dts.day:02d} "
                f"{dts.hour:02d}:{dts.min:02d}:{dts.sec:02d}"
                f".{(dts.us // 1_000):03d}")

    elif freq_group == FR_US and (is_fmt_none or fmt == "%Y-%m-%d %H:%M:%S.%u"):
        return (f"{dts.year}-{dts.month:02d}-{dts.day:02d} "
                f"{dts.hour:02d}:{dts.min:02d}:{dts.sec:02d}"
                f".{(dts.us):06d}")

    elif freq_group == FR_NS and (is_fmt_none or fmt == "%Y-%m-%d %H:%M:%S.%n"):
        return (f"{dts.year}-{dts.month:02d}-{dts.day:02d} "
                f"{dts.hour:02d}:{dts.min:02d}:{dts.sec:02d}"
                f".{((dts.us * 1000) + (dts.ps // 1000)):09d}")

    elif is_fmt_none:
        # `freq_group` is invalid, raise
        raise ValueError(f"Unknown freq: {freq}")

    else:
        # A custom format is requested
        if isinstance(fmt, str):
            # Encode using current locale, in case fmt contains non-utf8 chars
            fmt = <bytes>util.string_encode_locale(fmt)

        return _period_strftime(value, freq, fmt, dts)


cdef list extra_fmts = [(b"%q", b"^`AB`^"),
                        (b"%f", b"^`CD`^"),
                        (b"%F", b"^`EF`^"),
                        (b"%l", b"^`GH`^"),
                        (b"%u", b"^`IJ`^"),
                        (b"%n", b"^`KL`^")]

cdef list str_extra_fmts = ["^`AB`^", "^`CD`^", "^`EF`^",
                            "^`GH`^", "^`IJ`^", "^`KL`^"]

cdef str _period_strftime(int64_t value, int freq, bytes fmt, npy_datetimestruct dts):
    # [FUNCTION DOC]
    # HELPER: strftime with pandas extra directives. Replaces %q, %f, %F, %l, %u, %n in
    #   the byte fmt with unique placeholders (recording which were present), runs
    #   c_strftime on dts, decodes with the current locale and frees the buffer, then
    #   substitutes placeholders: %q quarter, %f 2-digit fiscal year, %F full fiscal
    #   year (fiscal values from get_yq, computed only if one of these three is used),
    #   %l milliseconds (3 digits), %u microseconds (6), %n nanoseconds (9) from the
    #   saved us/ps.
    # Inputs: value - ordinal; freq; fmt - bytes; dts - struct from get_date_info (by
    #   value).
    # Outputs: str.
    # Exceptions: MemoryError from c_strftime; decoding errors.
    # Side effects: temporary C allocation.
    cdef:
        Py_ssize_t i
        char *formatted
        bytes pat, brepl
        list found_pat = [False] * len(extra_fmts)
        int quarter
        int32_t us, ps
        str result, repl

    # Find our additional directives in the pattern and replace them with
    # placeholders that are not processed by c_strftime
    for i in range(len(extra_fmts)):
        pat = extra_fmts[i][0]
        brepl = extra_fmts[i][1]
        if pat in fmt:
            fmt = fmt.replace(pat, brepl)
            found_pat[i] = True

    # Execute c_strftime to process the usual datetime directives
    formatted = c_strftime(&dts, <char*>fmt)

    # Decode result according to current locale
    result = util.char_to_string_locale(formatted)
    free(formatted)

    # Now we will fill the placeholders corresponding to our additional directives

    # First prepare the contents
    # Save these to local vars as dts can be modified by get_yq below
    us = dts.us
    ps = dts.ps
    if any(found_pat[0:3]):
        # Note: this modifies `dts` in-place so that year becomes fiscal year
        # However it looses the us and ps
        quarter = get_yq(value, freq, &dts)
    else:
        quarter = 0

    # Now do the filling per se
    for i in range(len(extra_fmts)):
        if found_pat[i]:

            if i == 0:  # %q, 1-digit quarter.
                repl = f"{quarter}"
            elif i == 1:  # %f, 2-digit 'Fiscal' year
                repl = f"{(dts.year % 100):02d}"
            elif i == 2:  # %F, 'Fiscal' year with a century
                repl = str(dts.year)
            elif i == 3:  # %l, milliseconds
                repl = f"{(us // 1_000):03d}"
            elif i == 4:  # %u, microseconds
                repl = f"{(us):06d}"
            elif i == 5:  # %n, nanoseconds
                repl = f"{((us * 1000) + (ps // 1000)):09d}"

            result = result.replace(str_extra_fmts[i], repl)

    return result


def period_array_strftime(
    ndarray values, int dtype_code, object na_rep, str date_format
):
    # [FUNCTION DOC]
    # Vectorized period formatting for PeriodArray: for each element of an int64 ndarray
    #   of any shape, writes na_rep for NaT else period_format(ordinal, dtype_code,
    #   date_format).
    # Inputs: values - int64 ndarray; dtype_code - PeriodDtypeCode; na_rep - any object;
    #   date_format - str or None.
    # Outputs: new object ndarray of the same shape.
    # Exceptions: as period_format. Side effects: allocates the result.
    """
    Vectorized Period.strftime used for PeriodArray._format_native_types.

    Parameters
    ----------
    values : ndarray[int64_t], ndim unrestricted
    dtype_code : int
        Corresponds to PeriodDtype._dtype_code
    na_rep : any
    date_format : str or None
    """
    cdef:
        Py_ssize_t i, n = values.size
        int64_t ordinal
        object item_repr
        ndarray out = cnp.PyArray_EMPTY(
            values.ndim, values.shape, cnp.NPY_OBJECT, 0
        )
        object[::1] out_flat = out.ravel()
        cnp.broadcast mi = cnp.PyArray_MultiIterNew2(out, values)

    for i in range(n):
        # Analogous to: ordinal = values[i]
        ordinal = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

        if ordinal == NPY_NAT:
            item_repr = na_rep
        else:
            # This is equivalent to
            # freq = frequency_corresponding_to_dtype_code(dtype_code)
            # per = Period(ordinal, freq=freq)
            # if date_format:
            #     item_repr = per.strftime(date_format)
            # else:
            #     item_repr = str(per)
            item_repr = period_format(ordinal, dtype_code, date_format)

        # Analogous to: ordinals[i] = ordinal
        out_flat[i] = item_repr

        cnp.PyArray_MultiIter_NEXT(mi)

    return out


# ----------------------------------------------------------------------
# period accessors

ctypedef int (*accessor)(int64_t ordinal, int freq) except INT32_MIN


cdef int pyear(int64_t ordinal, int freq):
    # [FUNCTION DOC]
    # HELPER: Calendar year of the period's representative date (get_date_info; end day
    #   for coarse freqs).
    # Inputs: ordinal, freq. Outputs: int.
    cdef:
        npy_datetimestruct dts
    get_date_info(ordinal, freq, &dts)
    return dts.year


cdef int pqyear(int64_t ordinal, int freq):
    # [FUNCTION DOC]
    # HELPER: Fiscal year of the period (get_yq).
    # Inputs: ordinal, freq. Outputs: int.
    cdef:
        npy_datetimestruct dts

    get_yq(ordinal, freq, &dts)
    return dts.year


cdef int pquarter(int64_t ordinal, int freq):
    # [FUNCTION DOC]
    # HELPER: Fiscal quarter 1..4 of the period (get_yq).
    # Inputs: ordinal, freq. Outputs: int.
    cdef:
        int quarter
        npy_datetimestruct dts
    quarter = get_yq(ordinal, freq, &dts)
    return quarter


cdef int pmonth(int64_t ordinal, int freq):
    # [FUNCTION DOC]
    # HELPER: Month 1..12 of the period's representative date.
    # Inputs: ordinal, freq. Outputs: int.
    cdef:
        npy_datetimestruct dts
    get_date_info(ordinal, freq, &dts)
    return dts.month


cdef int pday(int64_t ordinal, int freq):
    # [FUNCTION DOC]
    # HELPER: Day of month of the period's representative date.
    # Inputs: ordinal, freq. Outputs: int.
    cdef:
        npy_datetimestruct dts
    get_date_info(ordinal, freq, &dts)
    return dts.day


cdef int pweekday(int64_t ordinal, int freq):
    # [FUNCTION DOC]
    # HELPER: Day of week (Monday=0) of the representative date.
    # Inputs: ordinal, freq. Outputs: int.
    cdef:
        npy_datetimestruct dts
    get_date_info(ordinal, freq, &dts)
    return dayofweek(dts.year, dts.month, dts.day)


cdef int pday_of_year(int64_t ordinal, int freq):
    # [FUNCTION DOC]
    # HELPER: Day of year (1-based) of the representative date.
    # Inputs: ordinal, freq. Outputs: int.
    cdef:
        npy_datetimestruct dts
    get_date_info(ordinal, freq, &dts)
    return get_day_of_year(dts.year, dts.month, dts.day)


cdef int pweek(int64_t ordinal, int freq):
    # [FUNCTION DOC]
    # HELPER: ISO week number of the representative date.
    # Inputs: ordinal, freq. Outputs: int.
    cdef:
        npy_datetimestruct dts
    get_date_info(ordinal, freq, &dts)
    return get_week_of_year(dts.year, dts.month, dts.day)


cdef int phour(int64_t ordinal, int freq):
    # [FUNCTION DOC]
    # HELPER: Hour of the period start (0 for daily and coarser).
    # Inputs: ordinal, freq. Outputs: int.
    cdef:
        npy_datetimestruct dts
    get_date_info(ordinal, freq, &dts)
    return dts.hour


cdef int pminute(int64_t ordinal, int freq):
    # [FUNCTION DOC]
    # HELPER: Minute of the period start (0 for hourly and coarser).
    # Inputs: ordinal, freq. Outputs: int.
    cdef:
        npy_datetimestruct dts
    get_date_info(ordinal, freq, &dts)
    return dts.min


cdef int psecond(int64_t ordinal, int freq):
    # [FUNCTION DOC]
    # HELPER: Second of the period start (0 for minutely and coarser).
    # Inputs: ordinal, freq. Outputs: int.
    cdef:
        npy_datetimestruct dts
    get_date_info(ordinal, freq, &dts)
    return <int>dts.sec


cdef int pdays_in_month(int64_t ordinal, int freq):
    # [FUNCTION DOC]
    # HELPER: Number of days in the month of the representative date.
    # Inputs: ordinal, freq. Outputs: int.
    cdef:
        npy_datetimestruct dts
    get_date_info(ordinal, freq, &dts)
    return get_days_in_month(dts.year, dts.month)


@cython.wraparound(False)
@cython.boundscheck(False)
def get_period_field_arr(str field, const int64_t[:] arr, int freq):
    # [FUNCTION DOC]
    # Vectorized field accessor: looks up the accessor for field and applies it to each
    #   ordinal; NaT entries produce -1.
    # Inputs: field - one of year, qyear, quarter, month, day, hour, minute, second,
    #   week, day_of_year, weekday/day_of_week, days_in_month; arr - 1-D int64 ordinals;
    #   freq - code.
    # Outputs: new 1-D int64 ndarray.
    # Exceptions: ValueError for an unrecognized field name.
    cdef:
        Py_ssize_t i, sz
        int64_t[::1] out

    func = _get_accessor_func(field)
    if func is NULL:
        raise ValueError(f"Unrecognized field name: {field}")

    sz = len(arr)
    out = np.empty(sz, dtype=np.int64)

    for i in range(sz):
        if arr[i] == NPY_NAT:
            out[i] = -1
            continue
        out[i] = func(arr[i], freq)

    return out.base  # .base to access underlying np.ndarray


cdef accessor _get_accessor_func(str field):
    # [FUNCTION DOC]
    # HELPER: Map a field name to the matching p* accessor function pointer ('weekday'
    #   and 'day_of_week' both map to pweekday).
    # Inputs: field - str. Outputs: function pointer or NULL if unknown.
    if field == "year":
        return <accessor>pyear
    elif field == "qyear":
        return <accessor>pqyear
    elif field == "quarter":
        return <accessor>pquarter
    elif field == "month":
        return <accessor>pmonth
    elif field == "day":
        return <accessor>pday
    elif field == "hour":
        return <accessor>phour
    elif field == "minute":
        return <accessor>pminute
    elif field == "second":
        return <accessor>psecond
    elif field == "week":
        return <accessor>pweek
    elif field == "day_of_year":
        return <accessor>pday_of_year
    elif field == "weekday" or field == "day_of_week":
        return <accessor>pweekday
    elif field == "days_in_month":
        return <accessor>pdays_in_month
    return NULL


@cython.wraparound(False)
@cython.boundscheck(False)
def from_ordinals(const int64_t[:] values, freq):
    # [FUNCTION DOC]
    # Build period ordinals from an integer array for a given freq. Despite the name,
    #   each non-NaT integer is passed as the value to Period(val, freq), which parses
    #   it as a string (e.g. 2000 -> the year 2000), so this re-interprets integers as
    #   date-like values rather than ordinals. NaT passes through.
    # Inputs: values - 1-D int64; freq - anything to_offset(is_period=True) accepts.
    # Outputs: new 1-D int64 ndarray of ordinals.
    # Exceptions: ValueError if freq is not an offset or values fail to parse.
    # Side effects: may emit the BDay FutureWarning via Period.
    cdef:
        Py_ssize_t i, n = len(values)
        int64_t[::1] result = np.empty(len(values), dtype="i8")
        int64_t val

    freq = to_offset(freq, is_period=True)
    if not isinstance(freq, BaseOffset):
        raise ValueError("freq not specified and cannot be inferred")

    for i in range(n):
        val = values[i]
        if val == NPY_NAT:
            result[i] = NPY_NAT
        else:
            result[i] = Period(val, freq=freq).ordinal

    return result.base


@cython.wraparound(False)
@cython.boundscheck(False)
def extract_ordinals(ndarray values, freq) -> np.ndarray:
    # [FUNCTION DOC]
    # Convert an object ndarray (any shape) of Period-like values into an int64 array of
    #   ordinals at freq, element by element via _extract_ordinal, comparing each
    #   Period's freqstr with the freqstr derived from freq's period dtype code and n.
    # Inputs: values - object-dtype ndarray; freq - period BaseOffset.
    # Outputs: new int64 ndarray of the same shape.
    # Exceptions: TypeError for non-object dtype or a non-NaT integer item;
    #   IncompatibleFrequency for mismatched freq; parsing errors from Period.
    # values is object-dtype, may be 2D

    cdef:
        Py_ssize_t i, n = values.size
        int64_t ordinal
        ndarray ordinals = cnp.PyArray_EMPTY(
            values.ndim, values.shape, cnp.NPY_INT64, 0
        )
        cnp.broadcast mi = cnp.PyArray_MultiIterNew2(ordinals, values)
        object p

    if values.descr.type_num != cnp.NPY_OBJECT:
        # if we don't raise here, we'll segfault later!
        raise TypeError("extract_ordinals values must be object-dtype")

    freqstr = PeriodDtypeBase(freq._period_dtype_code, freq.n)._freqstr

    for i in range(n):
        # Analogous to: p = values[i]
        p = <object>(<PyObject**>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

        ordinal = _extract_ordinal(p, freqstr, freq)

        # Analogous to: ordinals[i] = ordinal
        (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = ordinal

        cnp.PyArray_MultiIter_NEXT(mi)

    return ordinals


cdef int64_t _extract_ordinal(object item, str freqstr, freq) except? -1:
    # [FUNCTION DOC]
    # HELPER: Ordinal for one item: null/NaT/pd.NA -> NaT; integer equal to NaT value ->
    #   NaT; other integer -> TypeError; object with .ordinal -> that ordinal after
    #   checking its freqstr equals freqstr; otherwise Period(item, freq) and its
    #   ordinal (NaT if it parses to NaT).
    # Inputs: item; freqstr - expected freq string; freq - offset.
    # Outputs: int64 ordinal.
    # Exceptions: TypeError, IncompatibleFrequency, Period parse errors.
    """
    See extract_ordinals.
    """
    cdef:
        int64_t ordinal

    if checknull_with_nat(item) or item is C_NA:
        ordinal = NPY_NAT
    elif util.is_integer_object(item):
        if item == NPY_NAT:
            ordinal = NPY_NAT
        else:
            raise TypeError(item)
    else:
        try:
            ordinal = item.ordinal

            if item.freqstr != freqstr:
                msg = DIFFERENT_FREQ.format(cls="PeriodIndex",
                                            own_freq=freqstr,
                                            other_freq=item.freqstr)
                raise IncompatibleFrequency(msg)

        except AttributeError:
            item = Period(item, freq=freq)
            if item is NaT:
                # input may contain NaT-like string
                ordinal = NPY_NAT
            else:
                ordinal = item.ordinal

    return ordinal


def extract_freq(ndarray[object] values) -> BaseOffset:
    # [FUNCTION DOC]
    # Return the freq of the first Period object found in an object array.
    # Inputs: values - 1-D object ndarray. Outputs: BaseOffset.
    # Exceptions: ValueError if no Period is present.
    # TODO: Change type to const object[:] when Cython supports that.

    cdef:
        Py_ssize_t i, n = len(values)
        object value

    for i in range(n):
        value = values[i]

        if is_period_object(value):
            return value.freq

    raise ValueError("freq not specified and cannot be inferred")

# -----------------------------------------------------------------------
# period helpers


DIFFERENT_FREQ = ("Input has different freq={other_freq} "
                  "from {cls}(freq={own_freq})")


@set_module("pandas.errors")
class IncompatibleFrequency(TypeError):
    # [CLASS DOC]
    # Exception (subclass of TypeError, exposed as pandas.errors.IncompatibleFrequency)
    #   raised when combining periods/offsets whose frequencies do not match. No extra
    #   state.
    """
    Raised when trying to compare or operate between Periods with different frequencies.

    This error occurs when performing operations between Period objects or
    PeriodArrays that have different frequencies that cannot be aligned,
    such as comparing or doing arithmetic on periods with mismatched frequencies.

    See Also
    --------
    Period : Represents a period of time.
    PeriodIndex : Immutable ndarray holding ordinal values.
    PeriodDtype : An ExtensionDtype for Period data.

    Examples
    --------
    Trying to compare Period objects with different frequencies:

    >>> pd.Period("2024-01", freq="M") - pd.Period("2024-01-01", freq="D")
    Traceback (most recent call last):
    IncompatibleFrequency: Input has different freq=D from Period(freq=M)
    """
    pass


cdef class PeriodMixin:
    # [CLASS DOC]
    # Behavior shared by the Period scalar and PeriodArray: start_time / end_time via
    #   self.to_timestamp, and frequency-match validation. Subclasses must provide freq
    #   and to_timestamp. No state of its own.
    # Methods shared between Period and PeriodArray

    @property
    def start_time(self) -> Timestamp:
        # [FUNCTION DOC]
        # Timestamp at the start of the period: self.to_timestamp(how='start').
        # Outputs: Timestamp (or array equivalent for PeriodArray).
        """
        Get the Timestamp for the start of the period.

        Returns
        -------
        Timestamp

        See Also
        --------
        Period.end_time : Return the end Timestamp.
        Period.dayofyear : Return the day of year.
        Period.daysinmonth : Return the days in that month.
        Period.dayofweek : Return the day of the week.

        Examples
        --------
        >>> period = pd.Period('2012-1-1', freq='D')
        >>> period
        Period('2012-01-01', 'D')

        >>> period.start_time
        Timestamp('2012-01-01 00:00:00')

        >>> period.end_time
        Timestamp('2012-01-01 23:59:59.999999')
        """
        return self.to_timestamp(how="start")

    @property
    def end_time(self) -> Timestamp:
        # [FUNCTION DOC]
        # Timestamp of the last instant of the period: self.to_timestamp(how='end') (one
        #   unit before the next period's start).
        # Outputs: Timestamp (or array equivalent).
        """
        Get the Timestamp for the end of the period.

        Returns
        -------
        Timestamp

        See Also
        --------
        Period.start_time : Return the start Timestamp.
        Period.dayofyear : Return the day of year.
        Period.daysinmonth : Return the days in that month.
        Period.dayofweek : Return the day of the week.

        Examples
        --------
        For Period:

        >>> pd.Period('2020-01', 'D').end_time
        Timestamp('2020-01-01 23:59:59.999999')

        For Series:

        >>> period_index = pd.period_range('2020-1-1 00:00', '2020-3-1 00:00', freq='M')
        >>> s = pd.Series(period_index)
        >>> s
        0   2020-01
        1   2020-02
        2   2020-03
        dtype: period[M]
        >>> s.dt.end_time
        0   2020-01-31 23:59:59.999999
        1   2020-02-29 23:59:59.999999
        2   2020-03-31 23:59:59.999999
        dtype: datetime64[us]

        For PeriodIndex:

        >>> idx = pd.PeriodIndex(["2023-01", "2023-02", "2023-03"], freq="M")
        >>> idx.end_time
        DatetimeIndex(['2023-01-31 23:59:59.999999',
                       '2023-02-28 23:59:59.999999',
                       '2023-03-31 23:59:59.999999'],
                       dtype='datetime64[us]', freq=None)
        """
        return self.to_timestamp(how="end")

    def _require_matching_freq(self, other: BaseOffset, bint base=False):
        # [FUNCTION DOC]
        # HELPER: Raise unless frequencies match. With base=False compares self.freq !=
        #   other; with base=True compares self.freq.base != other.base (ignores the
        #   multiple n). On mismatch builds 'Input has different freq=<other> from
        #   <Class>(freq=<own>)' using period freq strings.
        # Inputs: other - BaseOffset; base - bool.
        # Outputs: None. Exceptions: IncompatibleFrequency.
        # See also arrays.period.raise_on_incompatible
        if base:
            condition = self.freq.base != other.base
        else:
            condition = self.freq != other

        if condition:
            freqstr = PeriodDtypeBase(
                self.freq._period_dtype_code, self.freq.n
            )._freqstr
            if hasattr(other, "_period_dtype_code"):
                other_freqstr = PeriodDtypeBase(
                    other._period_dtype_code, other.n
                )._freqstr
            else:
                other_freqstr = other.freqstr
            msg = DIFFERENT_FREQ.format(
                cls=type(self).__name__,
                own_freq=freqstr,
                other_freq=other_freqstr,
            )
            raise IncompatibleFrequency(msg)


cdef class _Period(PeriodMixin):
    # [CLASS DOC]
    # C-level base of Period storing _ordinal (int64), _freq (BaseOffset with n > 0) and
    #   _dtype (PeriodDtypeBase built from freq's period code and n). Implements
    #   comparison, hashing, arithmetic, frequency conversion, field properties
    #   (computed from the ordinal with the p* accessors at the dtype code), formatting
    #   and pickling. Instances are immutable.

    cdef readonly:
        int64_t _ordinal
        PeriodDtypeBase _dtype
        BaseOffset _freq

    @property
    def ordinal(self) -> int:
        # [FUNCTION DOC]
        # Return self._ordinal.
        # Outputs: int.
        """
        Return the integer ordinal for this Period.

        The ordinal is the internal integer representation of the Period,
        representing its position in the sequence of periods of the given
        frequency. It counts from an epoch (e.g., for daily frequency,
        ordinal 0 corresponds to January 1, 1970).

        See Also
        --------
        Period.freq : Return the frequency of the Period.
        Period.start_time : Return the start time of the Period.

        Examples
        --------
        >>> period = pd.Period('2020-01', freq='M')
        >>> period.ordinal
        600

        >>> period = pd.Period('2020-01-01', freq='D')
        >>> period.ordinal
        18262
        """
        return self._ordinal

    @property
    def freq(self):
        # [FUNCTION DOC]
        # Return self._freq.
        # Outputs: BaseOffset.
        """
        Return the frequency object for this Period.

        The frequency object represents the span of time that this Period covers.
        It is a DateOffset that defines the interval type (e.g., daily, monthly).

        See Also
        --------
        Period.freqstr : Return a string representation of the frequency.
        Period.asfreq : Convert Period to desired frequency.

        Examples
        --------
        >>> period = pd.Period('2020-01', freq='M')
        >>> period.freq
        <MonthEnd>
        """
        return self._freq

    # higher than np.ndarray, np.matrix, np.timedelta64
    __array_priority__ = 100

    dayofweek = _Period.day_of_week
    dayofyear = _Period.day_of_year

    def __cinit__(self, int64_t ordinal, BaseOffset freq):
        # [FUNCTION DOC]
        # Initialize the C fields: store ordinal and freq, build _dtype from
        #   freq._period_dtype_code and freq.n. No validation.
        # Inputs: ordinal - int64; freq - BaseOffset.
        # Exceptions: AttributeError if freq has no period dtype code.
        self._ordinal = ordinal
        self._freq = freq
        self._dtype = PeriodDtypeBase(freq._period_dtype_code, freq.n)

    @classmethod
    def _maybe_convert_freq(cls, object freq) -> BaseOffset:
        # [FUNCTION DOC]
        # Normalize freq to a period offset via to_offset(freq, is_period=True) and
        #   require freq.n > 0.
        # Inputs: freq - str/offset/timedelta-like. Outputs: BaseOffset.
        # Exceptions: ValueError if n <= 0 or freq invalid.
        """
        A Period's freq attribute must have `freq.n > 0`, which we check for here.

        Returns
        -------
        DateOffset
        """
        freq = to_offset(freq, is_period=True)

        if freq.n <= 0:
            raise ValueError("Frequency must be positive, because it "
                             f"represents span: {freq.freqstr}")

        return freq

    @classmethod
    def _from_ordinal(cls, ordinal: int64_t, freq: BaseOffset) -> "Period":
        # [FUNCTION DOC]
        # Fast constructor: NaT ordinal -> NaT singleton; otherwise normalizes freq with
        #   _maybe_convert_freq and creates cls via __cinit__ without parsing.
        # Inputs: ordinal - int; freq - offset. Outputs: Period or NaT.
        # Exceptions: as _maybe_convert_freq.
        """
        Fast creation from an ordinal and freq that are already validated!
        """
        if ordinal == NPY_NAT:
            return NaT
        else:
            freq = cls._maybe_convert_freq(freq)
            self = _Period.__new__(cls, ordinal, freq)
            return self

    def __richcmp__(self, other, op):
        # [FUNCTION DOC]
        # Rich comparison. Another Period: if dtypes (code and n) differ, == is False,
        #   != is True and ordering raises IncompatibleFrequency; else compare ordinals.
        #   NaT: only != is True. 0-dim ndarray: compare with its item; other ndarray:
        #   elementwise over the iteration, returning an ndarray of results. Otherwise
        #   NotImplemented.
        # Inputs: other; op - comparison opcode. Outputs: bool, ndarray or
        #   NotImplemented.
        if is_period_object(other):
            if other._dtype != self._dtype:
                if op == Py_EQ:
                    return False
                elif op == Py_NE:
                    return True
                self._require_matching_freq(other.freq)
            return PyObject_RichCompareBool(self._ordinal, other._ordinal, op)
        elif other is NaT:
            return op == Py_NE
        elif util.is_array(other):
            # GH#44285
            if cnp.PyArray_IsZeroDim(other):
                return PyObject_RichCompare(self, other.item(), op)
            else:
                # in particular ndarray[object]; see test_pi_cmp_period
                return np.array([PyObject_RichCompare(self, x, op) for x in other])
        return NotImplemented

    def __hash__(self):
        # [FUNCTION DOC]
        # Hash of the tuple (ordinal, freqstr).
        # Outputs: int.
        return hash((self.ordinal, self.freqstr))

    def _add_timedeltalike_scalar(self, other) -> "Period":
        # [FUNCTION DOC]
        # HELPER: Add a timedelta-like (timedelta, np.timedelta64, Tick, or Day treated
        #   as n days) to a tick-like (daily or finer) period. NaT timedelta64 -> NaT.
        #   Converts other to an integer count of the period's unit without rounding and
        #   adds it to the ordinal.
        # Inputs: other. Outputs: new Period (same freq) or NaT.
        # Exceptions: IncompatibleFrequency if the period is not tick-like or other is
        #   not an exact multiple of the unit; OverflowError on int64 overflow.
        cdef:
            int64_t inc, ordinal

        if not self._dtype._is_tick_like():
            raise IncompatibleFrequency("Input cannot be converted to "
                                        f"Period(freq={self.freqstr})")

        if (
            cnp.is_timedelta64_object(other) and
            cnp.get_timedelta64_value(other) == NPY_NAT
        ):
            # i.e. np.timedelta64("nat")
            return NaT

        if isinstance(other, Day):
            # Periods are timezone-naive, so we treat Day as Tick-like
            other = np.timedelta64(other.n, "D")

        try:
            inc = delta_to_nanoseconds(other, reso=self._dtype._creso, round_ok=False)
        except ValueError as err:
            raise IncompatibleFrequency("Input cannot be converted to "
                                        f"Period(freq={self.freqstr})") from err
        with cython.overflowcheck(True):
            ordinal = self._ordinal + inc
        return Period(ordinal=ordinal, freq=self._freq)

    def _add_offset(self, other) -> "Period":
        # [FUNCTION DOC]
        # HELPER: Add a non-tick DateOffset: requires the same base frequency (ignoring
        #   n), then adds other.n to the ordinal.
        # Inputs: other - offset. Outputs: new Period.
        # Exceptions: IncompatibleFrequency on base mismatch.
        # Non-Tick DateOffset other
        cdef:
            int64_t ordinal

        self._require_matching_freq(other, base=True)

        ordinal = self._ordinal + other.n
        return Period(ordinal=ordinal, freq=self._freq)

    @cython.overflowcheck(True)
    def __add__(self, other):
        # [FUNCTION DOC]
        # Addition. Timedelta-like or Day -> _add_timedeltalike_scalar; other offsets ->
        #   _add_offset; NaT -> NaT; integer k -> ordinal + k*n; Period -> TypeError;
        #   object ndarray -> elementwise object array; else NotImplemented.
        # Inputs: other. Outputs: Period, NaT, ndarray or NotImplemented.
        # Exceptions: TypeError, IncompatibleFrequency, OverflowError.
        if is_any_td_scalar(other) or isinstance(other, Day):
            return self._add_timedeltalike_scalar(other)
        elif is_offset_object(other):
            return self._add_offset(other)
        elif other is NaT:
            return NaT
        elif util.is_integer_object(other):
            ordinal = self._ordinal + other * self._dtype._n
            return Period(ordinal=ordinal, freq=self._freq)

        elif is_period_object(other):
            # can't add datetime-like
            # GH#17983; can't just return NotImplemented bc we get a RecursionError
            #  when called via np.add.reduce see TestNumpyReductions.test_add
            #  in npdev build
            sname = type(self).__name__
            oname = type(other).__name__
            raise TypeError(f"unsupported operand type(s) for +: '{sname}' "
                            f"and '{oname}'")

        elif util.is_array(other):
            if other.dtype == object:
                # GH#50162
                return np.array([self + x for x in other], dtype=object)

        return NotImplemented

    def __radd__(self, other):
        # [FUNCTION DOC]
        # Reflected addition; same as __add__.
        # Inputs: other. Outputs: as __add__.
        return self.__add__(other)

    def __sub__(self, other):
        # [FUNCTION DOC]
        # Subtraction. Timedelta-like, offset or integer -> self + (-other). Period ->
        #   require identical freq, return (ordinal difference) * freq.base as a
        #   DateOffset. NaT -> NaT. Object ndarray -> elementwise. Else NotImplemented.
        # Inputs: other. Outputs: Period, offset, NaT, ndarray or NotImplemented.
        # Exceptions: IncompatibleFrequency for a Period with different freq.
        if (
            is_any_td_scalar(other)
            or is_offset_object(other)
            or util.is_integer_object(other)
        ):
            return self + (-other)
        elif is_period_object(other):
            self._require_matching_freq(other.freq)
            # GH 23915 - mul by base freq since __add__ is agnostic of n
            return (self._ordinal - other._ordinal) * self._freq.base
        elif other is NaT:
            return NaT

        elif util.is_array(other):
            if other.dtype == object:
                # GH#50162
                return np.array([self - x for x in other], dtype=object)

        return NotImplemented

    def __rsub__(self, other):
        # [FUNCTION DOC]
        # Reflected subtraction: NaT -> NaT; object ndarray -> elementwise x - self;
        #   else NotImplemented.
        # Inputs: other. Outputs: NaT, ndarray or NotImplemented.
        if other is NaT:
            return NaT

        elif util.is_array(other):
            if other.dtype == object:
                # GH#50162
                return np.array([x - self for x in other], dtype=object)

        return NotImplemented

    def asfreq(self, freq, how="E") -> "Period":
        # [FUNCTION DOC]
        # Convert to another frequency. Validates freq (n > 0) and how ('S'/'E'
        #   aliases). For 'E' starts from the last base sub-period ordinal + n - 1
        #   (since self spans n base periods) and requests the end; for 'S' uses the
        #   ordinal and the start. Converts with period_asfreq between dtype codes.
        # Inputs: freq - str/offset; how - start/end alias.
        # Outputs: new Period with freq.
        # Exceptions: ValueError for invalid freq or how.
        # KNOWN ISSUE: the docstring example shows Period('2023-01-01 00:00', 'H') but
        #   hourly periods now repr with 'h'.
        """
        Convert Period to desired frequency, at the start or end of the interval.

        Parameters
        ----------
        freq : str, DateOffset
            The target frequency to convert the Period object to.
            If a string is provided,
            it must be a valid :ref:`period alias <timeseries.period_aliases>`.

        how : {'E', 'S', 'end', 'start'}, default 'end'
            Specifies whether to align the period to the start or end of the interval:
            - 'E' or 'end': Align to the end of the interval.
            - 'S' or 'start': Align to the start of the interval.

        Returns
        -------
        Period : Period object with the specified frequency, aligned to the parameter.

        See Also
        --------
        Period.end_time : Return the end Timestamp.
        Period.start_time : Return the start Timestamp.
        Period.dayofyear : Return the day of the year.
        Period.dayofweek : Return the day of the week.

        Examples
        --------
        Convert a daily period to an hourly period, aligning to the end of the day:

        >>> period = pd.Period('2023-01-01', freq='D')
        >>> period.asfreq('h')
        Period('2023-01-01 23:00', 'h')

        Convert a monthly period to a daily period, aligning to the start of the month:

        >>> period = pd.Period('2023-01', freq='M')
        >>> period.asfreq('D', how='start')
        Period('2023-01-01', 'D')

        Convert a yearly period to a monthly period, aligning to the last month:

        >>> period = pd.Period('2023', freq='Y')
        >>> period.asfreq('M', how='end')
        Period('2023-12', 'M')

        Convert a monthly period to an hourly period,
        aligning to the first day of the month:

        >>> period = pd.Period('2023-01', freq='M')
        >>> period.asfreq('h', how='start')
        Period('2023-01-01 00:00', 'H')

        Convert a weekly period to a daily period, aligning to the last day of the week:

        >>> period = pd.Period('2023-08-01', freq='W')
        >>> period.asfreq('D', how='end')
        Period('2023-08-04', 'D')
        """
        freq = self._maybe_convert_freq(freq)
        how = validate_end_alias(how)
        base1 = self._dtype._dtype_code
        base2 = freq_to_dtype_code(freq)

        # self.n can't be negative or 0
        end = how == "E"
        if end:
            ordinal = self._ordinal + self._dtype._n - 1
        else:
            ordinal = self._ordinal
        ordinal = period_asfreq(ordinal, base1, base2, end)

        return Period(ordinal=ordinal, freq=freq)

    def to_timestamp(self, freq=None, how="start") -> Timestamp:
        # [FUNCTION DOC]
        # Return a Timestamp for the start or end of the period. Unit is ns if the
        #   period is nanosecond-frequency or freq == 'ns', else us. how='E': for
        #   business freq returns start + 1 day - 1 unit; otherwise (self + freq) start
        #   minus 1 unit. how='S': target freq defaults to the dtype's to_timestamp base
        #   (daily for coarse freqs, own base otherwise); converts with asfreq(freq,
        #   'S') and period_ordinal_to_dt64, then wraps in Timestamp with that unit.
        # Inputs: freq - optional target frequency; how - start/end alias.
        # Outputs: Timestamp.
        # Exceptions: ValueError for bad how/freq; OutOfBoundsDatetime.
        """
        Return the Timestamp representation of the Period.

        Uses the target frequency specified at the part of the period specified
        by `how`, which is either `Start` or `Finish`.

        If possible, gives microsecond-unit Timestamp. Otherwise gives nanosecond
        unit.

        Parameters
        ----------
        freq : str or DateOffset
            Target frequency. Default is 'D' if self._freq is week or
            longer and 'S' otherwise.
        how : str, default 'S' (start)
            One of 'S', 'E'. Can be aliased as case insensitive
            'Start', 'Finish', 'Begin', 'End'.

        Returns
        -------
        Timestamp

        See Also
        --------
        Timestamp : A class representing a single point in time.
        Period : Represents a span of time with a fixed frequency.
        PeriodIndex.to_timestamp : Convert a `PeriodIndex` to a `DatetimeIndex`.

        Examples
        --------
        >>> period = pd.Period('2023-1-1', freq='D')
        >>> timestamp = period.to_timestamp()
        >>> timestamp
        Timestamp('2023-01-01 00:00:00')
        """
        how = validate_end_alias(how)

        if self._dtype._dtype_code == PeriodDtypeCode.N or freq == "ns":
            unit = "ns"
        else:
            unit = "us"

        end = how == "E"
        if end:
            if freq == "B" or self._freq == "B":
                # roll forward to ensure we land on B date
                adjust = np.timedelta64(1, "D") - np.timedelta64(1, unit)
                return self.to_timestamp(how="start") + adjust
            endpoint = (self + self._freq).to_timestamp(how="start")
            return endpoint - np.timedelta64(1, unit)

        if freq is None:
            freq_code = self._dtype._get_to_timestamp_base()
            dtype = PeriodDtypeBase(freq_code, 1)
            freq = dtype._freqstr
            base = freq_code
        else:
            freq = self._maybe_convert_freq(freq)
            base = freq._period_dtype_code

        val = self.asfreq(freq, how)

        dt64 = period_ordinal_to_dt64(val.ordinal, base)
        return Timestamp(dt64, unit=unit)

    @property
    def year(self) -> int:
        # [FUNCTION DOC]
        # Calendar year of the period's representative date (pyear at the dtype code).
        # Outputs: int.
        """
        Return the year this Period falls on.

        Returns
        -------
        int

        See Also
        --------
        period.month : Get the month of the year for the given Period.
        period.day : Return the day of the month the Period falls on.

        Notes
        -----
        The year is based on the `ordinal` and `base` attributes of the Period.

        Examples
        --------
        Create a Period object for January 2023 and get the year:

        >>> period = pd.Period('2023-01', 'M')
        >>> period.year
        2023

        Create a Period object for 01 January 2023 and get the year:

        >>> period = pd.Period('2023', 'D')
        >>> period.year
        2023

        Get the year for a period representing a quarter:

        >>> period = pd.Period('2023Q2', 'Q')
        >>> period.year
        2023

        Handle a case where the Period object is empty, which results in `NaN`:

        >>> period = pd.Period('nan', 'M')
        >>> period.year
        nan
        """
        base = self._dtype._dtype_code
        return pyear(self.ordinal, base)

    @property
    def month(self) -> int:
        # [FUNCTION DOC]
        # Month (pmonth).
        # Outputs: int.
        """
        Return the month this Period falls on.

        Returns
        -------
        int

        See Also
        --------
        period.week : Get the week of the year on the given Period.
        Period.year : Return the year this Period falls on.
        Period.day : Return the day of the month this Period falls on.

        Notes
        -----
        The month is based on the `ordinal` and `base` attributes of the Period.

        Examples
        --------
        Create a Period object for January 2022 and get the month:

        >>> period = pd.Period('2022-01', 'M')
        >>> period.month
        1

        Period object with no specified frequency, resulting in a default frequency:

        >>> period = pd.Period('2022', 'Y')
        >>> period.month
        12

        Create a Period object with a specified frequency but an incomplete date string:

        >>> period = pd.Period('2022', 'M')
        >>> period.month
        1

        Handle a case where the Period object is empty, which results in `NaN`:

        >>> period = pd.Period('nan', 'M')
        >>> period.month
        nan
        """
        base = self._dtype._dtype_code
        return pmonth(self.ordinal, base)

    @property
    def day(self) -> int:
        # [FUNCTION DOC]
        # Day of month (pday). Coarse periods use their LAST day.
        # Outputs: int.
        # KNOWN ISSUE: the docstring says yearly/quarterly periods return the first day
        #   of the period, but get_date_info uses the end day (e.g. Period('2020',
        #   'Y').day == 31).
        """
        Get day of the month that a Period falls on.

        The `day` property provides a simple way to access the day component
        of a `Period` object, which represents time spans in various frequencies
        (e.g., daily, hourly, monthly). If the period's frequency does not include
        a day component (e.g., yearly or quarterly periods), the returned day
        corresponds to the first day of that period.

        Returns
        -------
        int

        See Also
        --------
        Period.dayofweek : Get the day of the week.
        Period.dayofyear : Get the day of the year.

        Examples
        --------
        >>> p = pd.Period("2018-03-11", freq='h')
        >>> p.day
        11
        """
        base = self._dtype._dtype_code
        return pday(self.ordinal, base)

    @property
    def hour(self) -> int:
        # [FUNCTION DOC]
        # Hour of the period start (phour).
        # Outputs: int.
        """
        Get the hour of the day component of the Period.

        Returns
        -------
        int
            The hour as an integer, between 0 and 23.

        See Also
        --------
        Period.second : Get the second component of the Period.
        Period.minute : Get the minute component of the Period.

        Examples
        --------
        >>> p = pd.Period("2018-03-11 13:03:12.050000")
        >>> p.hour
        13

        Period longer than a day

        >>> p = pd.Period("2018-03-11", freq="M")
        >>> p.hour
        0
        """
        base = self._dtype._dtype_code
        return phour(self.ordinal, base)

    @property
    def minute(self) -> int:
        # [FUNCTION DOC]
        # Minute of the period start (pminute).
        # Outputs: int.
        """
        Get minute of the hour component of the Period.

        Returns
        -------
        int
            The minute as an integer, between 0 and 59.

        See Also
        --------
        Period.hour : Get the hour component of the Period.
        Period.second : Get the second component of the Period.

        Examples
        --------
        >>> p = pd.Period("2018-03-11 13:03:12.050000")
        >>> p.minute
        3
        """
        base = self._dtype._dtype_code
        return pminute(self.ordinal, base)

    @property
    def second(self) -> int:
        # [FUNCTION DOC]
        # Second of the period start (psecond).
        # Outputs: int.
        """
        Get the second component of the Period.

        Returns
        -------
        int
            The second of the Period (ranges from 0 to 59).

        See Also
        --------
        Period.hour : Get the hour component of the Period.
        Period.minute : Get the minute component of the Period.

        Examples
        --------
        >>> p = pd.Period("2018-03-11 13:03:12.050000")
        >>> p.second
        12
        """
        base = self._dtype._dtype_code
        return psecond(self.ordinal, base)

    @property
    def weekofyear(self) -> int:
        # [FUNCTION DOC]
        # ISO week of the representative date (pweek).
        # Outputs: int.
        """
        Get the week of the year on the given Period.

        Returns
        -------
        int

        See Also
        --------
        Period.dayofweek : Get the day component of the Period.
        Period.weekday : Get the day component of the Period.

        Examples
        --------
        >>> p = pd.Period("2018-03-11", "h")
        >>> p.weekofyear
        10

        >>> p = pd.Period("2018-02-01", "D")
        >>> p.weekofyear
        5

        >>> p = pd.Period("2018-01-06", "D")
        >>> p.weekofyear
        1
        """
        base = self._dtype._dtype_code
        return pweek(self.ordinal, base)

    @property
    def week(self) -> int:
        # [FUNCTION DOC]
        # Alias returning self.weekofyear.
        # Outputs: int.
        """
        Get the week of the year on the given Period.

        Returns
        -------
        int

        See Also
        --------
        Period.dayofweek : Get the day component of the Period.
        Period.weekday : Get the day component of the Period.

        Examples
        --------
        >>> p = pd.Period("2018-03-11", "h")
        >>> p.week
        10

        >>> p = pd.Period("2018-02-01", "D")
        >>> p.week
        5

        >>> p = pd.Period("2018-01-06", "D")
        >>> p.week
        1
        """
        return self.weekofyear

    @property
    def day_of_week(self) -> int:
        # [FUNCTION DOC]
        # Weekday, Monday=0 (pweekday).
        # Outputs: int.
        """
        Day of the week the period lies in, with Monday=0 and Sunday=6.

        If the period frequency is lower than daily (e.g. hourly), and the
        period spans over multiple days, the day at the start of the period is
        used.

        If the frequency is higher than daily (e.g. monthly), the last day
        of the period is used.

        Returns
        -------
        int
            Day of the week.

        See Also
        --------
        Period.day_of_week : Day of the week the period lies in.
        Period.weekday : Alias of Period.day_of_week.
        Period.day : Day of the month.
        Period.dayofyear : Day of the year.

        Examples
        --------
        >>> per = pd.Period('2017-12-31 22:00', 'h')
        >>> per.day_of_week
        6

        For periods that span over multiple days, the day at the beginning of
        the period is returned.

        >>> per = pd.Period('2017-12-31 22:00', '4h')
        >>> per.day_of_week
        6
        >>> per.start_time.day_of_week
        6

        For periods with a frequency higher than days, the last day of the
        period is returned.

        >>> per = pd.Period('2018-01', 'M')
        >>> per.day_of_week
        2
        >>> per.end_time.day_of_week
        2
        """
        base = self._dtype._dtype_code
        return pweekday(self.ordinal, base)

    @property
    def weekday(self) -> int:
        # [FUNCTION DOC]
        # Alias returning self.dayofweek.
        # Outputs: int.
        """
        Day of the week the period lies in, with Monday=0 and Sunday=6.

        If the period frequency is lower than daily (e.g. hourly), and the
        period spans over multiple days, the day at the start of the period is
        used.

        If the frequency is higher than daily (e.g. monthly), the last day
        of the period is used.

        Returns
        -------
        int
            Day of the week.

        See Also
        --------
        Period.dayofweek : Day of the week the period lies in.
        Period.weekday : Alias of Period.dayofweek.
        Period.day : Day of the month.
        Period.dayofyear : Day of the year.

        Examples
        --------
        >>> per = pd.Period('2017-12-31 22:00', 'h')
        >>> per.dayofweek
        6

        For periods that span over multiple days, the day at the beginning of
        the period is returned.

        >>> per = pd.Period('2017-12-31 22:00', '4h')
        >>> per.dayofweek
        6
        >>> per.start_time.dayofweek
        6

        For periods with a frequency higher than days, the last day of the
        period is returned.

        >>> per = pd.Period('2018-01', 'M')
        >>> per.dayofweek
        2
        >>> per.end_time.dayofweek
        2
        """
        # Docstring is a duplicate from dayofweek. Reusing docstrings with
        # Appender doesn't work for properties in Cython files, and setting
        # the __doc__ attribute is also not possible.
        return self.dayofweek

    @property
    def day_of_year(self) -> int:
        # [FUNCTION DOC]
        # Day of year (pday_of_year).
        # Outputs: int.
        """
        Return the day of the year.

        This attribute returns the day of the year on which the particular
        date occurs. The return value ranges between 1 to 365 for regular
        years and 1 to 366 for leap years.

        Returns
        -------
        int
            The day of year.

        See Also
        --------
        Period.day : Return the day of the month.
        Period.day_of_week : Return the day of week.
        PeriodIndex.day_of_year : Return the day of year of all indexes.

        Examples
        --------
        >>> period = pd.Period("2015-10-23", freq='h')
        >>> period.day_of_year
        296
        >>> period = pd.Period("2012-12-31", freq='D')
        >>> period.day_of_year
        366
        >>> period = pd.Period("2013-01-01", freq='D')
        >>> period.day_of_year
        1
        """
        base = self._dtype._dtype_code
        return pday_of_year(self.ordinal, base)

    @property
    def quarter(self) -> int:
        # [FUNCTION DOC]
        # Fiscal quarter (pquarter).
        # Outputs: int.
        """
        Return the quarter this Period falls on.

        See Also
        --------
        Timestamp.quarter : Return the quarter of the Timestamp.
        Period.year : Return the year of the period.
        Period.month : Return the month of the period.

        Examples
        --------
        >>> period = pd.Period('2022-04', 'M')
        >>> period.quarter
        2
        """
        base = self._dtype._dtype_code
        return pquarter(self.ordinal, base)

    @property
    def qyear(self) -> int:
        # [FUNCTION DOC]
        # Fiscal year (pqyear).
        # Outputs: int.
        """
        Fiscal year the Period lies in according to its starting-quarter.

        The `year` and the `qyear` of the period will be the same if the fiscal
        and calendar years are the same. When they are not, the fiscal year
        can be different from the calendar year of the period.

        Returns
        -------
        int
            The fiscal year of the period.

        See Also
        --------
        Period.year : Return the calendar year of the period.

        Examples
        --------
        If the natural and fiscal year are the same, `qyear` and `year` will
        be the same.

        >>> per = pd.Period('2018Q1', freq='Q')
        >>> per.qyear
        2018
        >>> per.year
        2018

        If the fiscal year starts in April (`Q-MAR`), the first quarter of
        2018 will start in April 2017. `year` will then be 2017, but `qyear`
        will be the fiscal year, 2018.

        >>> per = pd.Period('2018Q1', freq='Q-MAR')
        >>> per.start_time
        Timestamp('2017-04-01 00:00:00')
        >>> per.qyear
        2018
        >>> per.year
        2017
        """
        base = self._dtype._dtype_code
        return pqyear(self.ordinal, base)

    @property
    def days_in_month(self) -> int:
        # [FUNCTION DOC]
        # Days in the month of the representative date (pdays_in_month).
        # Outputs: int.
        """
        Get the total number of days in the month that this period falls on.

        Returns
        -------
        int

        See Also
        --------
        Period.daysinmonth : Gets the number of days in the month.
        DatetimeIndex.daysinmonth : Gets the number of days in the month.
        calendar.monthrange : Returns a tuple containing weekday
            (0-6 ~ Mon-Sun) and number of days (28-31).

        Examples
        --------
        >>> p = pd.Period('2018-2-17')
        >>> p.days_in_month
        28

        >>> pd.Period('2018-03-01').days_in_month
        31

        Handles the leap year case as well:

        >>> p = pd.Period('2016-2-17')
        >>> p.days_in_month
        29
        """
        base = self._dtype._dtype_code
        return pdays_in_month(self.ordinal, base)

    @property
    def daysinmonth(self) -> int:
        # [FUNCTION DOC]
        # Alias returning self.days_in_month.
        # Outputs: int.
        """
        Get the total number of days of the month that this period falls on.

        Returns
        -------
        int

        See Also
        --------
        Period.days_in_month : Return the days of the month.
        Period.dayofyear : Return the day of the year.

        Examples
        --------
        >>> p = pd.Period("2018-03-11", freq='h')
        >>> p.daysinmonth
        31
        """
        return self.days_in_month

    @property
    def is_leap_year(self) -> bool:
        # [FUNCTION DOC]
        # Whether self.year is a leap year.
        # Outputs: bool.
        """
        Return True if the period's year is in a leap year.

        See Also
        --------
        Timestamp.is_leap_year : Check if the year in a Timestamp is a leap year.
        DatetimeIndex.is_leap_year : Boolean indicator if the date belongs to a
            leap year.

        Examples
        --------
        >>> period = pd.Period('2022-01', 'M')
        >>> period.is_leap_year
        False

        >>> period = pd.Period('2020-01', 'M')
        >>> period.is_leap_year
        True
        """
        return bool(is_leapyear(self.year))

    @classmethod
    def now(cls, freq):
        # [FUNCTION DOC]
        # Period at freq containing the current local wall-clock time (datetime.now()).
        # Inputs: freq. Outputs: Period. Side effects: reads the system clock.
        """
        Return the period of now's date.

        The `now` method provides a convenient way to generate a period
        object for the current date and time. This can be particularly
        useful in financial and economic analysis, where data is often
        collected and analyzed in regular intervals (e.g., hourly, daily,
        monthly). By specifying the frequency, users can create periods
        that match the granularity of their data.

        Parameters
        ----------
        freq : str, DateOffset
            Frequency to use for the returned period.

        See Also
        --------
        to_datetime : Convert argument to datetime.
        Period : Represents a period of time.
        Period.to_timestamp : Return the Timestamp representation of the Period.

        Examples
        --------
        >>> pd.Period.now('h')  # doctest: +SKIP
        Period('2023-06-12 11:00', 'h')
        """
        return Period(datetime.now(), freq=freq)

    @property
    def freqstr(self) -> str:
        # [FUNCTION DOC]
        # Period frequency string from PeriodDtypeBase(code, n), e.g. '2M'.
        # Outputs: str.
        """
        Return a string representation of the frequency.

        This property provides the frequency string associated with the `Period`
        object. The frequency string describes the granularity of the time span
        represented by the `Period`. Common frequency strings include 'D' for
        daily, 'M' for monthly, 'Y' for yearly, etc.

        See Also
        --------
        Period.asfreq : Convert Period to desired frequency, at the start or end
            of the interval.
        period_range : Return a fixed frequency PeriodIndex.

        Examples
        --------
        >>> pd.Period('2020-01', 'D').freqstr
        'D'
        """
        freqstr = PeriodDtypeBase(self._freq._period_dtype_code, self._freq.n)._freqstr
        return freqstr

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Return "Period('<period_format default>', '<freqstr>')".
        # Outputs: str.
        base = self._dtype._dtype_code
        formatted = period_format(self.ordinal, base)
        return f"Period('{formatted}', '{self.freqstr}')"

    def __str__(self) -> str:
        # [FUNCTION DOC]
        # Return period_format(ordinal, code) with the default format.
        # Outputs: str.
        # KNOWN ISSUE: the docstring mentions 'a particular DataFrame', which is
        #   unrelated to this method.
        """
        Return a string representation for a particular DataFrame
        """
        base = self._dtype._dtype_code
        formatted = period_format(self.ordinal, base)
        value = str(formatted)
        return value

    def __setstate__(self, state):
        # [FUNCTION DOC]
        # Restore _freq and _ordinal from state[1] and state[2].
        # Inputs: state - tuple. Side effects: mutates self.
        # KNOWN ISSUE: _dtype is not updated; it is not reached by normal unpickling
        #   because __reduce__ returns no state.
        self._freq = state[1]
        self._ordinal = state[2]

    def __reduce__(self):
        # [FUNCTION DOC]
        # Pickle support: reconstruct as Period(None, freq, ordinal).
        # Outputs: (Period, (None, freq, ordinal)).
        object_state = None, self._freq, self.ordinal
        return (Period, object_state)

    def strftime(self, fmt: str | None) -> str:
        # [FUNCTION DOC]
        # Format using period_format(ordinal, code, fmt); see period_format and
        #   _period_strftime for supported directives.
        # Inputs: fmt - str or None. Outputs: str.
        r"""
        Returns a formatted string representation of the :class:`Period`.

        ``fmt`` must be ``None`` or a string containing one or several directives.
        When ``None``, the format will be determined from the frequency of the Period.
        The method recognizes the same directives as the :func:`time.strftime`
        function of the standard Python distribution, as well as the specific
        additional directives ``%f``, ``%F``, ``%q``, ``%l``, ``%u``, ``%n``.
        (formatting & docs originally from scikits.timeries).

        +-----------+--------------------------------+-------+
        | Directive | Meaning                        | Notes |
        +===========+================================+=======+
        | ``%a``    | Locale's abbreviated weekday   |       |
        |           | name.                          |       |
        +-----------+--------------------------------+-------+
        | ``%A``    | Locale's full weekday name.    |       |
        +-----------+--------------------------------+-------+
        | ``%b``    | Locale's abbreviated month     |       |
        |           | name.                          |       |
        +-----------+--------------------------------+-------+
        | ``%B``    | Locale's full month name.      |       |
        +-----------+--------------------------------+-------+
        | ``%c``    | Locale's appropriate date and  |       |
        |           | time representation.           |       |
        +-----------+--------------------------------+-------+
        | ``%d``    | Day of the month as a decimal  |       |
        |           | number [01,31].                |       |
        +-----------+--------------------------------+-------+
        | ``%f``    | 'Fiscal' year without a        | \(1)  |
        |           | century  as a decimal number   |       |
        |           | [00,99]                        |       |
        +-----------+--------------------------------+-------+
        | ``%F``    | 'Fiscal' year with a century   | \(2)  |
        |           | as a decimal number            |       |
        +-----------+--------------------------------+-------+
        | ``%H``    | Hour (24-hour clock) as a      |       |
        |           | decimal number [00,23].        |       |
        +-----------+--------------------------------+-------+
        | ``%I``    | Hour (12-hour clock) as a      |       |
        |           | decimal number [01,12].        |       |
        +-----------+--------------------------------+-------+
        | ``%j``    | Day of the year as a decimal   |       |
        |           | number [001,366].              |       |
        +-----------+--------------------------------+-------+
        | ``%m``    | Month as a decimal number      |       |
        |           | [01,12].                       |       |
        +-----------+--------------------------------+-------+
        | ``%M``    | Minute as a decimal number     |       |
        |           | [00,59].                       |       |
        +-----------+--------------------------------+-------+
        | ``%p``    | Locale's equivalent of either  | \(3)  |
        |           | AM or PM.                      |       |
        +-----------+--------------------------------+-------+
        | ``%q``    | Quarter as a decimal number    |       |
        |           | [1,4]                          |       |
        +-----------+--------------------------------+-------+
        | ``%S``    | Second as a decimal number     | \(4)  |
        |           | [00,61].                       |       |
        +-----------+--------------------------------+-------+
        | ``%l``    | Millisecond as a decimal number|       |
        |           | [000,999].                     |       |
        +-----------+--------------------------------+-------+
        | ``%u``    | Microsecond as a decimal number|       |
        |           | [000000,999999].               |       |
        +-----------+--------------------------------+-------+
        | ``%n``    | Nanosecond as a decimal number |       |
        |           | [000000000,999999999].         |       |
        +-----------+--------------------------------+-------+
        | ``%U``    | Week number of the year        | \(5)  |
        |           | (Sunday as the first day of    |       |
        |           | the week) as a decimal number  |       |
        |           | [00,53].  All days in a new    |       |
        |           | year preceding the first       |       |
        |           | Sunday are considered to be in |       |
        |           | week 0.                        |       |
        +-----------+--------------------------------+-------+
        | ``%w``    | Weekday as a decimal number    |       |
        |           | [0(Sunday),6].                 |       |
        +-----------+--------------------------------+-------+
        | ``%W``    | Week number of the year        | \(5)  |
        |           | (Monday as the first day of    |       |
        |           | the week) as a decimal number  |       |
        |           | [00,53].  All days in a new    |       |
        |           | year preceding the first       |       |
        |           | Monday are considered to be in |       |
        |           | week 0.                        |       |
        +-----------+--------------------------------+-------+
        | ``%x``    | Locale's appropriate date      |       |
        |           | representation.                |       |
        +-----------+--------------------------------+-------+
        | ``%X``    | Locale's appropriate time      |       |
        |           | representation.                |       |
        +-----------+--------------------------------+-------+
        | ``%y``    | Year without century as a      |       |
        |           | decimal number [00,99].        |       |
        +-----------+--------------------------------+-------+
        | ``%Y``    | Year with century as a decimal |       |
        |           | number.                        |       |
        +-----------+--------------------------------+-------+
        | ``%Z``    | Time zone name (no characters  |       |
        |           | if no time zone exists).       |       |
        +-----------+--------------------------------+-------+
        | ``%%``    | A literal ``'%'`` character.   |       |
        +-----------+--------------------------------+-------+

        The `strftime` method provides a way to represent a :class:`Period`
        object as a string in a specified format. This is particularly useful
        when displaying date and time data in different locales or customized
        formats, suitable for reports or user interfaces. It extends the standard
        Python string formatting capabilities with additional directives specific
        to `pandas`, accommodating features like fiscal years and precise
        sub-second components.

        Parameters
        ----------
        fmt : str or None
            String containing the desired format directives. If ``None``, the
            format is determined based on the Period's frequency.

        See Also
        --------
        Timestamp.strftime : Return a formatted string of the Timestamp.
        to_datetime : Convert argument to datetime.
        time.strftime : Format a time object as a string according to a
            specified format string in the standard Python library.

        Notes
        -----

        (1)
            The ``%f`` directive is the same as ``%y`` if the frequency is
            not quarterly.
            Otherwise, it corresponds to the 'fiscal' year, as defined by
            the :attr:`qyear` attribute.

        (2)
            The ``%F`` directive is the same as ``%Y`` if the frequency is
            not quarterly.
            Otherwise, it corresponds to the 'fiscal' year, as defined by
            the :attr:`qyear` attribute.

        (3)
            The ``%p`` directive only affects the output hour field
            if the ``%I`` directive is used to parse the hour.

        (4)
            The range really is ``0`` to ``61``; this accounts for leap
            seconds and the (very rare) double leap seconds.

        (5)
            The ``%U`` and ``%W`` directives are only used in calculations
            when the day of the week and the year are specified.

        Examples
        --------

        >>> from pandas import Period
        >>> a = Period(freq='Q-JUL', year=2006, quarter=1)
        >>> a.strftime('%F-Q%q')
        '2006-Q1'
        >>> # Output the last month in the quarter of this date
        >>> a.strftime('%b-%Y')
        'Oct-2005'
        >>>
        >>> a = Period(freq='D', year=2001, month=1, day=1)
        >>> a.strftime('%d-%b-%Y')
        '01-Jan-2001'
        >>> a.strftime('%b. %d, %Y was a %A')
        'Jan. 01, 2001 was a Monday'
        """
        base = self._dtype._dtype_code
        return period_format(self.ordinal, base, fmt)


@set_module("pandas")
class Period(_Period):
    # [CLASS DOC]
    # Public Period scalar (subclass of _Period); all logic is in __new__, which parses
    #   many input forms into (ordinal, freq).
    """
    Represents a period of time.

    A `Period` represents a specific time span rather than a point in time.
    Unlike `Timestamp`, which represents a single instant, a `Period` defines a
    duration, such as a month, quarter, or year. The exact representation is
    determined by the `freq` parameter.

    Parameters
    ----------
    value : Period, str, datetime, date or pandas.Timestamp, default None
        The time period represented (e.g., '4Q2005'). This represents neither
        the start or the end of the period, but rather the entire period itself.
    freq : str, default None
        One of pandas period strings or corresponding objects. Accepted
        strings are listed in the
        :ref:`period alias section <timeseries.period_aliases>` in the user docs.
        If value is datetime, freq is required.
    ordinal : int, default None
        The period offset from the proleptic Gregorian epoch.
    year : int, default None
        Year value of the period.
    month : int, default 1
        Month value of the period.
    quarter : int, default None
        Quarter value of the period.
    day : int, default 1
        Day value of the period.
    hour : int, default 0
        Hour value of the period.
    minute : int, default 0
        Minute value of the period.
    second : int, default 0
        Second value of the period.

    See Also
    --------
    Timestamp : Pandas replacement for python datetime.datetime object.
    date_range : Return a fixed frequency DatetimeIndex.
    timedelta_range : Generates a fixed frequency range of timedeltas.

    Examples
    --------
    >>> period = pd.Period('2012-1-1', freq='D')
    >>> period
    Period('2012-01-01', 'D')
    """

    def __new__(cls, value=None, freq=None, ordinal=None,
                year=None, month=None, quarter=None, day=None,
                hour=None, minute=None, second=None):
        # [FUNCTION DOC]
        # Construct a Period or return NaT. Steps: if freq given, normalize (n > 0) and
        #   require a period dtype code. ordinal and value both given -> ValueError;
        #   ordinal given -> must be an integer and freq required. value None: all
        #   fields None -> NaT; else freq required, defaults month=1, day=1,
        #   hour/minute/second=0, ordinal from _ordinal_from_fields. value Period: reuse
        #   its ordinal/freq if freq None or same code, else asfreq(freq). Null/NaT or
        #   NaT strings/empty string -> NaT. str or int (int converted to str, NaT value
        #   to 'NaT'): uppercase and parse with parse_datetime_string_with_reso; on
        #   ValueError try the 'YYYY-MM-DD/YYYY-MM-DD' weekly form; infer freq from the
        #   parsed resolution if freq None; keep nanoseconds. datetime / datetime64 /
        #   date: freq required. Finally compute the ordinal from the datetime
        #   components with period_ordinal if not already set, warn for BDay freq,
        #   return NaT for a NaT ordinal, else _from_ordinal.
        # Inputs: value, freq, ordinal, year, month, quarter, day, hour, minute, second.
        # Outputs: Period or NaT.
        # Exceptions: ValueError for missing/conflicting inputs or unparseable values;
        #   TypeError for unsupported freq types; parsing errors.
        # Side effects: FutureWarning for BDay frequency.
        # freq points to a tuple (base, mult);  base is one of the defined
        # periods such as Y, Q, etc. Every five minutes would be, e.g.,
        # ('min', 5) but may be passed in as a string like '5min'

        # ordinal is the period offset from the gregorian proleptic epoch

        if freq is not None:
            freq = cls._maybe_convert_freq(freq)
            try:
                period_dtype_code = freq._period_dtype_code
            except (AttributeError, TypeError):
                # AttributeError: _period_dtype_code might not exist
                # TypeError: _period_dtype_code might intentionally raise
                raise TypeError(
                    f"{(type(freq).__name__)} is not supported as period frequency"
                )
        nanosecond = 0

        if ordinal is not None and value is not None:
            raise ValueError("Only value or ordinal but not both should be "
                             "given but not both")
        elif ordinal is not None:
            if not util.is_integer_object(ordinal):
                raise ValueError("Ordinal must be an integer")
            if freq is None:
                raise ValueError("Must supply freq for ordinal value")

        elif value is None:
            if (year is None and month is None and
                    quarter is None and day is None and
                    hour is None and minute is None and second is None):
                ordinal = NPY_NAT
            else:
                if freq is None:
                    raise ValueError("If value is None, freq cannot be None")

                # set defaults
                month = 1 if month is None else month
                day = 1 if day is None else day
                hour = 0 if hour is None else hour
                minute = 0 if minute is None else minute
                second = 0 if second is None else second

                ordinal = _ordinal_from_fields(year, month, quarter, day,
                                               hour, minute, second, freq)

        elif is_period_object(value):
            other = value
            if freq is None or period_dtype_code == other._dtype._dtype_code:
                ordinal = other.ordinal
                freq = other.freq
            else:
                converted = other.asfreq(freq)
                ordinal = converted.ordinal

        elif checknull_with_nat(value) or (isinstance(value, str) and
                                           (value in nat_strings or len(value) == 0)):
            # explicit str check is necessary to avoid raising incorrectly
            #  if we have a non-hashable value.
            ordinal = NPY_NAT

        elif isinstance(value, str) or util.is_integer_object(value):
            if util.is_integer_object(value):
                if value == NPY_NAT:
                    value = "NaT"

                value = str(value)
            value = value.upper()

            freqstr = freq.rule_code if freq is not None else None
            try:
                dt, reso = parse_datetime_string_with_reso(value, freqstr)
            except ValueError as err:
                match = re.search(r"^\d{4}-\d{2}-\d{2}/\d{4}-\d{2}-\d{2}", value)
                if match:
                    # Case that cannot be parsed (correctly) by our datetime
                    #  parsing logic
                    dt, freq = _parse_weekly_str(value, freq)
                else:
                    raise err

            else:
                if reso == "nanosecond":
                    nanosecond = dt.nanosecond
                if dt is NaT:
                    ordinal = NPY_NAT

                if freq is None and ordinal != NPY_NAT:
                    # Skip NaT, since it doesn't have a resolution
                    freq = attrname_to_abbrevs[reso]
                    freq = c_OFFSET_TO_PERIOD_FREQSTR.get(freq, freq)
                    freq = to_offset(freq, is_period=True)

        elif PyDateTime_Check(value):
            dt = value
            if freq is None:
                raise ValueError("Must supply freq for datetime value")
            if isinstance(dt, Timestamp):
                nanosecond = dt.nanosecond
        elif cnp.is_datetime64_object(value):
            dt = Timestamp(value)
            if freq is None:
                raise ValueError("Must supply freq for datetime value")
            nanosecond = dt.nanosecond
        elif PyDate_Check(value):
            dt = datetime(year=value.year, month=value.month, day=value.day)
            if freq is None:
                raise ValueError("Must supply freq for datetime value")
        else:
            msg = "Value must be Period, string, integer, or datetime"
            raise ValueError(msg)

        if ordinal is None:
            base = freq_to_dtype_code(freq)
            ordinal = period_ordinal(dt.year, dt.month, dt.day,
                                     dt.hour, dt.minute, dt.second,
                                     dt.microsecond, 1000*nanosecond, base)

        if isinstance(freq, BDay):
            # GH#53446
            import warnings

            # TODO: Enforce in 3.0 (#53511)
            from pandas.util._exceptions import find_stack_level
            warnings.warn(
                "Period with BDay freq is deprecated and will be removed "
                "in a future version. Use a DatetimeIndex with BDay freq instead.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        if ordinal == NPY_NAT:
            return NaT
        return cls._from_ordinal(ordinal, freq)


cdef bint is_period_object(object obj):
    # [FUNCTION DOC]
    # True iff obj is an instance of _Period.
    # Inputs: obj. Outputs: bool.
    return isinstance(obj, _Period)


cpdef int freq_to_dtype_code(BaseOffset freq) except? -1:
    # [FUNCTION DOC]
    # Return freq._period_dtype_code.
    # Inputs: freq - BaseOffset. Outputs: int.
    # Exceptions: ValueError (invalid frequency) if the offset has none.
    try:
        return freq._period_dtype_code
    except AttributeError as err:
        raise ValueError(INVALID_FREQ_ERR_MSG.format(freq)) from err


cdef int64_t _ordinal_from_fields(int year, int month, quarter, int day,
                                  int hour, int minute, int second,
                                  BaseOffset freq):
    # [FUNCTION DOC]
    # HELPER: Ordinal from date fields at freq. If quarter given, converts (fiscal year,
    #   quarter) to the calendar year and starting month using freq's fiscal year end
    #   (quarter_to_myear). Microsecond/picosecond are 0.
    # Inputs: year, month, quarter (int or None), day, hour, minute, second, freq.
    #   Outputs: int64 ordinal.
    # Exceptions: ValueError from freq_to_dtype_code/quarter_to_myear.
    base = freq_to_dtype_code(freq)
    if quarter is not None:
        year, month = quarter_to_myear(year, quarter, freq.freqstr)

    return period_ordinal(year, month, day, hour,
                          minute, second, 0, 0, base)


def validate_end_alias(how: str) -> str:  # Literal["E", "S"]
    # [FUNCTION DOC]
    # Normalize a how argument: case-insensitively maps S/START/BEGIN -> 'S' and
    #   E/END/FINISH -> 'E'.
    # Inputs: how - any (str()-converted). Outputs: 'S' or 'E'.
    # Exceptions: ValueError otherwise.
    how_dict = {"S": "S", "E": "E",
                "START": "S", "FINISH": "E",
                "BEGIN": "S", "END": "E"}
    how = how_dict.get(str(how).upper())
    if how not in {"S", "E"}:
        raise ValueError("How must be one of S or E")
    return how


cdef _parse_weekly_str(value, BaseOffset freq):
    # [FUNCTION DOC]
    # HELPER: Parse 'start/end' strings produced by str() of weekly periods. Splits on
    #   '/', parses both as Timestamps, requires end - start == 6 days; if freq None,
    #   infers 'W-<end weekday abbreviation>'.
    # Inputs: value - str; freq - offset or None.
    # Outputs: (end Timestamp, freq).
    # Exceptions: ValueError if not exactly 6 days apart or unpacking fails; Timestamp
    #   parse errors.
    """
    Parse e.g. "2017-01-23/2017-01-29", which cannot be parsed by the general
    datetime-parsing logic.  This ensures that we can round-trip with
    Period.__str__ with weekly freq.
    """
    # GH#50803
    start, end = value.split("/")
    start = Timestamp(start)
    end = Timestamp(end)

    if (end - start).days != 6:
        # We are interested in cases where this is str(period)
        #  of a Week-freq period
        raise ValueError("Could not parse as weekly-freq Period")

    if freq is None:
        day_name = end.day_name()[:3].upper()
        freqstr = f"W-{day_name}"
        freq = to_offset(freqstr, is_period=True)
        # We _should_ have freq.is_on_offset(end)

    return end, freq
