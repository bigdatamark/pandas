# [MODULE DOC]
# Declarations for ccalendar.pyx: fast, GIL-free calendar arithmetic helpers on
# proleptic Gregorian (year, month, day) integers, plus month/day name lookup
# tables. iso_calendar_t is a C tuple of three 32-bit ints (iso year, iso week,
# iso weekday 1-7). All functions assume valid inputs and do no validation.
from cython cimport Py_ssize_t
from numpy cimport (
    int32_t,
    int64_t,
)

ctypedef (int32_t, int32_t, int32_t) iso_calendar_t

# [FUNCTION DOC]
# Weekday of date y-m-d, 0=Monday..6=Sunday (Gauss algorithm). No validation.
cdef int dayofweek(int y, int m, int d) noexcept nogil
# [FUNCTION DOC]
# True if year is a Gregorian leap year (divisible by 4, not by 100 unless by 400).
cdef bint is_leapyear(int64_t year) noexcept nogil
# [FUNCTION DOC]
# Number of days (28-31) in month (1-12) of year; out-of-range month reads memory
# out of bounds.
cpdef int32_t get_days_in_month(int year, Py_ssize_t month) noexcept nogil
# [FUNCTION DOC]
# ISO 8601 week number (1-53) of the given date.
cpdef int32_t get_week_of_year(int year, int month, int day) noexcept nogil
# [FUNCTION DOC]
# (ISO year, ISO week number, ISO weekday 1=Monday..7=Sunday) of the given date.
cpdef iso_calendar_t get_iso_calendar(int year, int month, int day) noexcept nogil
# [FUNCTION DOC]
# 1-based ordinal day of year (1-366) of the given date.
cpdef int32_t get_day_of_year(int year, int month, int day) noexcept nogil
# [FUNCTION DOC]
# Day of month (28-31) of the last weekday (Mon-Fri) in the given month.
cpdef int get_lastbday(int year, int month) noexcept nogil
# [FUNCTION DOC]
# Day of month (1-3) of the first weekday (Mon-Fri) in the given month.
cpdef int get_firstbday(int year, int month) noexcept nogil

# Lookup tables: c_MONTH_NUMBERS maps "JAN".."DEC" to 0..11;
# MONTH_TO_CAL_NUM maps them to 1..12.
cdef dict c_MONTH_NUMBERS, MONTH_TO_CAL_NUM

# Lookup table: month_offset, 26 cumulative day counts before month N (index N-1;
# entry 12 = year length); entries 0-12 for non-leap years, 13-25 for leap years.
cdef int32_t* month_offset
