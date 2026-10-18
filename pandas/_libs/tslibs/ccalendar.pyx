# [MODULE DOC]
# Pure-C calendar arithmetic for the proleptic Gregorian calendar, used throughout
# tslibs (field accessors, offsets, Timestamp properties). Holds static lookup
# tables (days per month for non-leap/leap years, cumulative month offsets, a
# Gauss-algorithm month table "em") and Python-visible name constants: MONTHS
# ("JAN".."DEC"), MONTHS_FULL (index 0 is "" so index = month number), DAYS
# ("MON".."SUN"), DAYS_FULL, MONTH_NUMBERS/c_MONTH_NUMBERS (abbrev -> 0..11),
# MONTH_ALIASES (1..12 -> abbrev), MONTH_TO_CAL_NUM (abbrev -> 1..12),
# int_to_weekday / weekday_to_int (0=Monday). All functions are nogil and noexcept,
# operate on plain ints, do no input validation, and have no side effects.
# Weekdays are 0=Monday..6=Sunday unless stated otherwise.
# cython: boundscheck=False
"""
Cython implementations of functions resembling the stdlib calendar module
"""
cimport cython
from numpy cimport (
    int32_t,
    int64_t,
)

# ----------------------------------------------------------------------
# Constants

# Slightly more performant cython lookups than a 2D table
# The first 12 entries correspond to month lengths for non-leap years.
# The remaining 12 entries give month lengths for leap years
cdef int32_t* days_per_month_array = [
    31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
    31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

cdef int* em = [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

# The first 13 entries give the month days elapsed as of the first of month N
# (or the total number of days in the year for N=13) in non-leap years.
# The remaining 13 entries give the days elapsed in leap years.
cdef int32_t* month_offset = [
    0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365,
    0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]

# Canonical location for other modules to find name constants
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL",
          "AUG", "SEP", "OCT", "NOV", "DEC"]
# The first blank line is consistent with calendar.month_name in the calendar
# standard library
MONTHS_FULL = ["", "January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November",
               "December"]
MONTH_NUMBERS = {name: num for num, name in enumerate(MONTHS)}
cdef dict c_MONTH_NUMBERS = MONTH_NUMBERS
MONTH_ALIASES = {(num + 1): name for num, name in enumerate(MONTHS)}
cdef dict MONTH_TO_CAL_NUM = {name: num + 1 for num, name in enumerate(MONTHS)}

DAYS = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
DAYS_FULL = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday"]
int_to_weekday = {num: name for num, name in enumerate(DAYS)}
weekday_to_int = {int_to_weekday[key]: key for key in int_to_weekday}


# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef int32_t get_days_in_month(int year, Py_ssize_t month) noexcept nogil:
    # [FUNCTION DOC]
    # Return days in month by indexing the 24-entry days-per-month table at
    # 12 * (1 if leap year else 0) + month - 1.
    # Inputs: year - int; month - int in 1..12.
    # Outputs: int32 day count 28..31.
    # Exceptions: none; bounds checking is disabled, so an invalid month reads
    # out-of-bounds memory (undefined behavior). Side effects: none.
    """
    Return the number of days in the given month of the given year.

    Parameters
    ----------
    year : int
    month : int

    Returns
    -------
    days_in_month : int

    Notes
    -----
    Assumes that the arguments are valid.  Passing a month not between 1 and 12
    risks a segfault.
    """
    return days_per_month_array[12 * is_leapyear(year) + month - 1]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef long quot(long a , long b) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Floor division of a by b (rounds toward negative infinity), using C
    # truncating division and subtracting 1 when a is negative and not divisible.
    # Inputs: a, b - C longs, b assumed > 0 (b == 0 is undefined behavior).
    # Outputs: floor(a / b) as long. Side effects: none.
    cdef long x
    x = a/b
    if (a < 0):
        x -= (a % b != 0)
    return x


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef int dayofweek(int y, int m, int d) noexcept nogil:
    # [FUNCTION DOC]
    # Day of week via Gauss's algorithm. If m < 3 the year is decremented (Jan/Feb are
    # treated as months of the prior year); c = floor(y/100), g = y - 100*c,
    # f = 5 * (c mod 4), e = em[m] (cumulative days before month m in a non-leap
    # year), minus 1 when m > 2. Result is (d - 1 + e + f + g + g/4) mod 7 with C
    # integer division.
    # Inputs: y, m (1..12), d - ints describing a valid date.
    # Outputs: int 0=Monday..6=Sunday.
    # Exceptions: none (no validation; invalid m reads out of bounds). Side effects:
    # none.
    """
    Find the day of week for the date described by the Y/M/D triple y, m, d
    using Gauss' method, from wikipedia.

    0 represents Monday.  See [1]_.

    Parameters
    ----------
    y : int
    m : int
    d : int

    Returns
    -------
    weekday : int

    Notes
    -----
    Assumes that y, m, d, represents a valid date.

    See Also
    --------
    [1] https://docs.python.org/3/library/calendar.html#calendar.weekday

    [2] https://en.wikipedia.org/wiki/\
    Determination_of_the_day_of_the_week#Gauss's_algorithm
    """
    # Note: this particular implementation comes from
    # http://berndt-schwerdtfeger.de/wp-content/uploads/pdf/cal.pdf
    cdef:
        long c
        int g
        int f
        int e

    if (m < 3):
        y -= 1

    c = quot(y, 100)
    g = y - c * 100
    f = 5 * (c - quot(c, 4) * 4)
    e = em[m]

    if (m > 2):
        e -= 1
    return (-1 + d + e + f + g + g/4) % 7

cdef bint is_leapyear(int64_t year) noexcept nogil:
    # [FUNCTION DOC]
    # Gregorian leap year test: year divisible by 4 (checked via low two bits) and
    # either not divisible by 100 or divisible by 400.
    # Inputs: year - 64-bit int (may be negative). Outputs: boolean.
    # Side effects: none.
    """
    Returns 1 if the given year is a leap year, 0 otherwise.

    Parameters
    ----------
    year : int

    Returns
    -------
    is_leap : bool
    """
    return ((year & 0x3) == 0 and  # year % 4 == 0
            ((year % 100) != 0 or (year % 400) == 0))


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef int32_t get_week_of_year(int year, int month, int day) noexcept nogil:
    # [FUNCTION DOC]
    # Return the ISO 8601 week number by taking element 1 of get_iso_calendar.
    # Inputs: year, month, day - ints of a valid date. Outputs: int32 week 1..53.
    # Side effects: none.
    """
    Return the ordinal week-of-year for the given day.

    Parameters
    ----------
    year : int
    month : int
    day : int

    Returns
    -------
    week_of_year : int32_t

    Notes
    -----
    Assumes the inputs describe a valid date.
    """
    return get_iso_calendar(year, month, day)[1]


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef iso_calendar_t get_iso_calendar(int year, int month, int day) noexcept nogil:
    # [FUNCTION DOC]
    # Compute the ISO 8601 calendar triple. doy = day of year, dow = weekday
    # (0=Mon). Estimate week = (doy - 1) - dow + 3, then if >= 0, week = week // 7 + 1.
    # If the estimate is negative the date belongs to the last week of the previous
    # ISO year: week 53 if estimate is -1, or -2 and the previous year is a leap year;
    # otherwise 52. If the estimate is 53 but fewer than 3 days of that week fall in
    # this year (31 - day + dow < 3), it is week 1 of next year. ISO year is year + 1
    # when week is 1 in December, year - 1 when week >= 52 in January.
    # Inputs: year, month, day - ints of a valid date.
    # Outputs: tuple (iso_year, iso_week, iso_weekday) with weekday 1=Mon..7=Sun.
    # Side effects: none.
    # KNOWN ISSUE: the docstring says it returns "year, week, and day of year", but
    # the third element is the ISO weekday (1-7), not the day of year.
    """
    Return the year, week, and day of year corresponding to ISO 8601

    Parameters
    ----------
    year : int
    month : int
    day : int

    Returns
    -------
    year : int32_t
    week : int32_t
    day : int32_t

    Notes
    -----
    Assumes the inputs describe a valid date.
    """
    cdef:
        int32_t doy, dow
        int32_t iso_year, iso_week

    doy = get_day_of_year(year, month, day)
    dow = dayofweek(year, month, day)

    # estimate
    iso_week = (doy - 1) - dow + 3
    if iso_week >= 0:
        iso_week = iso_week // 7 + 1

    # verify
    if iso_week < 0:
        if (iso_week > -2) or (iso_week == -2 and is_leapyear(year - 1)):
            iso_week = 53
        else:
            iso_week = 52
    elif iso_week == 53:
        if 31 - day + dow < 3:
            iso_week = 1

    iso_year = year
    if iso_week == 1 and month == 12:
        iso_year += 1

    elif iso_week >= 52 and month == 1:
        iso_year -= 1

    return iso_year, iso_week, dow + 1


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef int32_t get_day_of_year(int year, int month, int day) noexcept nogil:
    # [FUNCTION DOC]
    # Return the ordinal day of year: month_offset[13 * is_leap + month - 1] + day.
    # Inputs: year, month (1..12), day - ints of a valid date.
    # Outputs: int32 in 1..366. Exceptions: none (no validation; bounds checking is
    # off). Side effects: none.
    """
    Return the ordinal day-of-year for the given day.

    Parameters
    ----------
    year : int
    month : int
    day : int

    Returns
    -------
    day_of_year : int32_t

    Notes
    -----
    Assumes the inputs describe a valid date.
    """
    cdef:
        bint isleap
        int32_t mo_off
        int day_of_year

    isleap = is_leapyear(year)

    mo_off = month_offset[isleap * 13 + month - 1]

    day_of_year = mo_off + day
    return day_of_year


# ---------------------------------------------------------------------
# Business Helpers

cpdef int get_lastbday(int year, int month) noexcept nogil:
    # [FUNCTION DOC]
    # Return the day-of-month of the last business day (Mon-Fri) of the month. Finds
    # the weekday of the last day as (weekday_of_1st + days_in_month - 1) mod 7 and
    # steps back 1 day for Saturday (5) or 2 for Sunday (6).
    # Inputs: year, month (1..12). Outputs: int day number 26..31.
    # Side effects: none.
    """
    Find the last day of the month that is a business day.

    Parameters
    ----------
    year : int
    month : int

    Returns
    -------
    last_bday : int
    """
    cdef:
        int wkday, days_in_month

    wkday = dayofweek(year, month, 1)
    days_in_month = get_days_in_month(year, month)
    return days_in_month - max(((wkday + days_in_month - 1) % 7) - 4, 0)


cpdef int get_firstbday(int year, int month) noexcept nogil:
    # [FUNCTION DOC]
    # Return the day-of-month of the first business day (Mon-Fri): 1, or 3 if the
    # 1st is a Saturday, or 2 if it is a Sunday.
    # Inputs: year, month (1..12). Outputs: int 1..3. Side effects: none.
    """
    Find the first day of the month that is a business day.

    Parameters
    ----------
    year : int
    month : int

    Returns
    -------
    first_bday : int
    """
    cdef:
        int first, wkday

    wkday = dayofweek(year, month, 1)
    first = 1
    if wkday == 5:  # on Saturday
        first = 3
    elif wkday == 6:  # on Sunday
        first = 2
    return first
