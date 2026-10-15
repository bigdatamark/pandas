# [MODULE DOC]
# Implementation of all pandas DateOffset frequency objects (pd.offsets.*) and the
# string-to-offset parser to_offset. An offset represents a calendar-aware
# increment: n multiples of a rule (Day, Hour, MonthEnd, BusinessDay, FY5253...).
# Adding an offset to a datetime/Timestamp calls the offset's _apply method;
# adding to a datetime64 array calls _apply_array (vectorized, operating on
# int64 epoch values in a given resolution unit). Offsets are hashable and
# compare by a canonical params tuple. Fixed-width offsets (Tick subclasses:
# Hour..Nano) delegate arithmetic to Timedelta. Anchored offsets (month, quarter,
# half-year, year, week, fiscal year, easter) shift dates via month arithmetic
# on broken-down date structs and a "day_opt" rule (start/end/business start/
# business end of month). Business offsets use numpy busdaycalendar objects.
# Also provides low-level month/quarter shifting kernels used by DatetimeArray
# (shift_months, shift_quarters, shift_month, roll_qtrday, roll_convention).
# Module-level state: _relativedelta_kwds set, prefix_mapping/_lite_rule_alias/
# _dont_uppercase/opattern/INVALID_FREQ_ERR_MSG/_offset_map cache dict used by
# the string parser (_offset_map is mutated as a cache).

import re
import time
import warnings

from pandas.util._exceptions import find_stack_level

cimport cython
from cpython.datetime cimport (
    PyDate_Check,
    PyDateTime_Check,
    PyDelta_Check,
    date,
    datetime,
    import_datetime,
    time as dt_time,
    timedelta,
)

import warnings

import_datetime()

import numpy as np

cimport numpy as cnp
from numpy cimport (
    int64_t,
    ndarray,
)

cnp.import_array()

# TODO: formalize having _libs.properties "above" tslibs in the dependency structure

from typing import ClassVar

from pandas._libs.properties import cache_readonly

from pandas._libs.tslibs cimport util
from pandas._libs.tslibs.util cimport (
    is_float_object,
    is_integer_object,
)

from pandas._libs.tslibs.ccalendar import (
    MONTH_ALIASES,
    int_to_weekday,
    weekday_to_int,
)
from pandas.util._decorators import set_module
from pandas.util._exceptions import find_stack_level

from pandas._libs.tslibs.ccalendar cimport (
    MONTH_TO_CAL_NUM,
    dayofweek,
    get_days_in_month,
    get_firstbday,
    get_lastbday,
)
from pandas._libs.tslibs.conversion cimport localize_pydatetime
from pandas._libs.tslibs.dtypes cimport (
    c_OFFSET_RENAMED_FREQSTR,
    c_OFFSET_TO_PERIOD_FREQSTR,
    c_PERIOD_AND_OFFSET_DEPR_FREQSTR,
    c_PERIOD_TO_OFFSET_FREQSTR,
    periods_per_day,
)
from pandas._libs.tslibs.nattype cimport (
    NPY_NAT,
    c_NaT as NaT,
)
from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    get_unit_from_dtype,
    import_pandas_datetime,
    npy_datetimestruct,
    npy_datetimestruct_to_datetime,
    pandas_datetime_to_datetimestruct,
    pydate_to_dtstruct,
)

import_pandas_datetime()

from .dtypes cimport PeriodDtypeCode
from .timedeltas cimport (
    _Timedelta,
    delta_to_nanoseconds,
    is_any_td_scalar,
)

from .timedeltas import Timedelta

from .timestamps cimport _Timestamp

from .timestamps import Timestamp

# day_opt enum: avoids touching Python str objects inside nogil loops

# [CLASS DOC]
# Enumeration of day-of-month rules used by the C-level month shifting code:
# DAY_OPT_START (day 1), DAY_OPT_END (last calendar day), DAY_OPT_BUSINESS_START
# (first weekday), DAY_OPT_BUSINESS_END (last weekday). Avoids string compares
# inside GIL-free loops.
cdef enum _DayOpt:
    DAY_OPT_START = 0
    DAY_OPT_END = 1
    DAY_OPT_BUSINESS_START = 2
    DAY_OPT_BUSINESS_END = 3


cdef _DayOpt _str_to_day_opt(str day_opt) except? DAY_OPT_START:
    # [FUNCTION DOC]
    # HELPER: Map a day_opt string to the _DayOpt enum value.
    # Inputs: day_opt - one of "start", "end", "business_start", "business_end".
    # Outputs: matching enum value.
    # Exceptions: ValueError for any other value (including None).
    # Side effects: none.
    if day_opt == "start":
        return DAY_OPT_START
    elif day_opt == "end":
        return DAY_OPT_END
    elif day_opt == "business_start":
        return DAY_OPT_BUSINESS_START
    elif day_opt == "business_end":
        return DAY_OPT_BUSINESS_END
    else:
        raise ValueError(
            "day must be 'start', 'end', 'business_start', or "
            f"'business_end', got {day_opt}"
        )


# ---------------------------------------------------------------------
# Misc Helpers

cdef bint is_offset_object(object obj):
    # [FUNCTION DOC]
    # Return True if obj is an instance of BaseOffset (any pandas offset).
    # Inputs: obj - any object. Outputs: boolean. Exceptions/side effects: none.
    return isinstance(obj, BaseOffset)


cdef bint is_tick_object(object obj):
    # [FUNCTION DOC]
    # Return True if obj is an instance of Tick (fixed-width sub-daily offset).
    # Inputs: obj - any object. Outputs: boolean. Exceptions/side effects: none.
    return isinstance(obj, Tick)


cdef bint _is_normalized(datetime dt):
    # [FUNCTION DOC]
    # HELPER: Return True if dt is exactly midnight: hour, minute, second and
    # microsecond are zero and, if dt is a pandas Timestamp, nanosecond is zero too.
    # Inputs: dt - datetime or Timestamp. Outputs: boolean. Side effects: none.
    if dt.hour != 0 or dt.minute != 0 or dt.second != 0 or dt.microsecond != 0:
        # Regardless of whether dt is datetime vs Timestamp
        return False
    if isinstance(dt, _Timestamp):
        return dt.nanosecond == 0
    return True


def apply_wraps(func):
    # [FUNCTION DOC]
    # Decorator for offset _apply methods that adds common datetime handling.
    # Inputs: func - a method taking (self, other) and returning a datetime.
    # Outputs: wrapper function with func's __name__ and __doc__ copied over.
    # The wrapper: returns NaT for NaT; passes offsets/timedeltas/timedelta64
    # straight to func; converts datetime64/date/datetime to Timestamp; otherwise
    # raises ApplyTypeError (turned into NotImplemented by __add__). For datetimes it
    # strips the timezone (if self._adjust_dst) before calling func, converts the
    # result to a Timestamp in the input's unit if lossless, re-localizes to the
    # original tz, normalizes to midnight if self.normalize, and restores a lost
    # nanosecond component (for offsets without a 'nanoseconds' attribute) by
    # adding the original nanosecond to the (UTC-naive) value.
    # Exceptions: ApplyTypeError for unsupported types; tz localization errors
    # (e.g. nonexistent/ambiguous times) propagate.
    # Side effects: none.
    # Note: normally we would use `@functools.wraps(func)`, but this does
    # not play nicely with cython class methods

    def wrapper(self, other):
        # [FUNCTION DOC]
        # HELPER: Inner closure produced by apply_wraps; see apply_wraps for behavior.
        # Inputs: self - the offset; other - NaT, timedelta-like, offset, or date-like.
        # Outputs: NaT, func's result, or the adjusted Timestamp.
        # Exceptions: ApplyTypeError for unsupported other.
        # Side effects: none.

        if other is NaT:
            return NaT
        elif (
            isinstance(other, BaseOffset)
            or PyDelta_Check(other)
            or cnp.is_timedelta64_object(other)
        ):
            # timedelta path
            return func(self, other)
        elif cnp.is_datetime64_object(other) or PyDate_Check(other):
            # PyDate_Check includes date, datetime
            other = Timestamp(other)
        else:
            # This will end up returning NotImplemented back in __add__
            raise ApplyTypeError

        tz = other.tzinfo
        nano = other.nanosecond

        if self._adjust_dst:
            other = other.tz_localize(None)

        result = func(self, other)

        result2 = Timestamp(result).as_unit(other.unit)
        if result == result2:
            # i.e. the conversion is non-lossy, not the case for e.g.
            #  test_milliseconds_combination
            result = result2

        if self._adjust_dst:
            result = result.tz_localize(tz)

        if self.normalize:
            result = result.normalize()

        # If the offset object does not have a nanoseconds component,
        # the result's nanosecond component may be lost.
        if not self.normalize and nano != 0 and not hasattr(self, "nanoseconds"):
            if result.nanosecond != nano:
                if result.tz is not None:
                    # convert to UTC
                    res = result.tz_localize(None)
                else:
                    res = result
                value = res.as_unit("ns")._value
                result = Timestamp(value + nano)

        if tz is not None and result.tzinfo is None:
            result = result.tz_localize(tz)

        return result

    # do @functools.wraps(func) manually since it doesn't work on cdef funcs
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


cdef _wrap_timedelta_result(result):
    # [FUNCTION DOC]
    # HELPER: If result is a timedelta (incl. Timedelta), convert it to the
    # equivalent Tick via delta_to_tick; otherwise return result unchanged.
    # Inputs: result - any object from a Timedelta operation.
    # Outputs: Tick or the original object. Side effects: none.
    """
    Tick operations dispatch to their Timedelta counterparts.  Wrap the result
    of these operations in a Tick if possible.

    Parameters
    ----------
    result : object

    Returns
    -------
    object
    """
    if PyDelta_Check(result):
        # convert Timedelta back to a Tick
        return delta_to_tick(result)

    return result

# ---------------------------------------------------------------------
# Business Helpers


cdef _get_calendar(weekmask, holidays, calendar):
    # [FUNCTION DOC]
    # HELPER: Build a numpy busdaycalendar and a normalized holidays tuple.
    # Inputs: weekmask - numpy weekmask spec (str or sequence); holidays - None or
    # sequence of date-likes; calendar - np.busdaycalendar, an object with a
    # holidays() method (e.g. AbstractHolidayCalendar), or None.
    # Outputs: tuple (busdaycalendar, holidays_tuple). If calendar is already a
    # busdaycalendar it is returned as-is and holidays become its holidays if empty,
    # else are coerced to tuple. Otherwise holidays are concatenated with
    # calendar.holidays() (if available), converted to datetime64[D], sorted, and a
    # new busdaycalendar is constructed with weekmask and (if non-empty) holidays.
    # Exceptions: numpy errors for invalid weekmask/holidays; TypeError if holidays
    # is not a list and calendar has holidays() (list + list concatenation).
    # Side effects: none.
    """
    Generate busdaycalendar
    """
    if isinstance(calendar, np.busdaycalendar):
        if not holidays:
            holidays = tuple(calendar.holidays)
        elif not isinstance(holidays, tuple):
            holidays = tuple(holidays)
        else:
            # trust that calendar.holidays and holidays are
            # consistent
            pass
        return calendar, holidays

    if holidays is None:
        holidays = []
    try:
        holidays = holidays + calendar.holidays().tolist()
    except AttributeError:
        pass
    holidays = tuple(sorted(_to_dt64D(dt) for dt in holidays))

    kwargs = {"weekmask": weekmask}
    if holidays:
        kwargs["holidays"] = holidays

    busdaycalendar = np.busdaycalendar(**kwargs)
    return busdaycalendar, holidays


cdef _to_dt64D(dt):
    # [FUNCTION DOC]
    # HELPER: Convert a date-like to numpy datetime64 with day resolution. For
    # tz-aware inputs the local wall date is used (tzinfo dropped first, GH#49441).
    # Inputs: dt - date, datetime, Timestamp, datetime64 or date string.
    # Outputs: np.datetime64 with unit 'D'. Exceptions: numpy parse errors.
    # Side effects: none.
    # Currently
    # > np.datetime64(dt.datetime(2013,5,1),dtype='datetime64[D]')
    # numpy.datetime64('2013-05-01T02:00:00.000000+0200')
    # Thus astype is needed to cast datetime to datetime64[D]
    if getattr(dt, "tzinfo", None) is not None:
        # Get the nanosecond timestamp,
        #  equiv `Timestamp(dt).value` or `dt.timestamp() * 10**9`
        # The `naive` must be the `dt` naive wall time
        #  instead of the naive absolute time (GH#49441)
        naive = dt.replace(tzinfo=None)
        dt = np.datetime64(naive, "D")
    else:
        dt = np.datetime64(dt)
    if dt.dtype.name != "datetime64[D]":
        dt = dt.astype("datetime64[D]")
    return dt


# ---------------------------------------------------------------------
# Validation


cdef _validate_business_time(t_input):
    # [FUNCTION DOC]
    # HELPER: Validate/convert a business hour boundary.
    # Inputs: t_input - "HH:MM" string or datetime.time.
    # Outputs: datetime.time with only hour and minute set.
    # Exceptions: ValueError if string does not match "%H:%M", if a time has
    # nonzero seconds/microseconds, or for any other type.
    # Side effects: none.
    if isinstance(t_input, str):
        try:
            t = time.strptime(t_input, "%H:%M")
            return dt_time(hour=t.tm_hour, minute=t.tm_min)
        except ValueError:
            raise ValueError("time data must match '%H:%M' format")
    elif isinstance(t_input, dt_time):
        if t_input.second != 0 or t_input.microsecond != 0:
            raise ValueError(
                "time data must be specified only with hour and minute")
        return t_input
    else:
        raise ValueError("time data must be string or datetime.time")


# ---------------------------------------------------------------------
# Constructor Helpers

_relativedelta_kwds = {"years", "months", "weeks", "days", "year", "month",
                       "day", "weekday", "hour", "minute", "second",
                       "microsecond", "millisecond", "nanosecond",
                       "nanoseconds", "hours", "minutes", "seconds",
                       "milliseconds", "microseconds"}


cdef _determine_offset(kwds):
    # [FUNCTION DOC]
    # HELPER: Choose the underlying increment object for a DateOffset from its
    # keyword arguments.
    # Inputs: kwds - dict of DateOffset keyword arguments.
    # Outputs: tuple (offset, use_relativedelta). Empty kwds -> (1-day timedelta,
    # False). Only nanosecond(s) keys -> (zero timedelta, False) since nanos are
    # handled by apply_wraps. Only sub-daily plural keys (seconds, microseconds,
    # milliseconds, minutes, hours) -> (timedelta(**kwds), False). Otherwise
    # milliseconds are folded into microseconds and, if all keys are relativedelta
    # keys, returns (dateutil relativedelta(**kwds), True).
    # Exceptions: NotImplementedError if "millisecond" (singular) given;
    # ValueError for any other key combination.
    # Side effects: imports dateutil lazily; does not mutate kwds.
    if not kwds:
        # GH 45643/45890: (historically) defaults to 1 day
        return timedelta(days=1), False

    if "millisecond" in kwds:
        raise NotImplementedError(
            "Using DateOffset to replace `millisecond` component in "
            "datetime object is not supported. Use "
            "`microsecond=timestamp.microsecond % 1000 + ms * 1000` "
            "instead."
        )

    nanos = {"nanosecond", "nanoseconds"}

    # nanos are handled by apply_wraps
    if all(k in nanos for k in kwds):
        return timedelta(days=0), False

    kwds_no_nanos = {k: v for k, v in kwds.items() if k not in nanos}

    kwds_use_relativedelta = {
        "year", "month", "day", "hour", "minute",
        "second", "microsecond", "weekday", "years", "months", "weeks", "days",
        "hours", "minutes", "seconds", "microseconds"
    }

    # "weeks" and "days" are left out despite being valid args for timedelta,
    # because (historically) timedelta is used only for sub-daily.
    kwds_use_timedelta = {
        "seconds", "microseconds", "milliseconds", "minutes", "hours",
    }

    if all(k in kwds_use_timedelta for k in kwds_no_nanos):
        # Sub-daily offset - use timedelta (tz-aware)
        # This also handles "milliseconds" (plur): see GH 49897
        return timedelta(**kwds_no_nanos), False

    # convert milliseconds to microseconds, so relativedelta can parse it
    if "milliseconds" in kwds_no_nanos:
        micro = kwds_no_nanos.pop("milliseconds") * 1000
        kwds_no_nanos["microseconds"] = kwds_no_nanos.get("microseconds", 0) + micro

    if all(k in kwds_use_relativedelta for k in kwds_no_nanos):
        from dateutil.relativedelta import relativedelta

        return relativedelta(**kwds_no_nanos), True

    raise ValueError(
        f"Invalid argument/s or bad combination of arguments: {list(kwds.keys())}"
    )

# ---------------------------------------------------------------------
# Mixins & Singletons


class ApplyTypeError(TypeError):
    # [CLASS DOC]
    # Sentinel TypeError subclass raised by apply_wraps/_apply when the other
    # operand is unsupported; BaseOffset.__add__ catches it and returns
    # NotImplemented so Python can try the reflected operation.
    # sentinel class for catching the apply error to return NotImplemented
    pass


# ---------------------------------------------------------------------
# Base Classes

cdef class BaseOffset:
    # [CLASS DOC]
    # Root extension type for all date offsets. Stores _n (int64 multiple),
    # _normalize (bool: snap results to midnight) and _cache (dict for
    # cache_readonly properties). Class attributes: _day_opt (day rule for anchored
    # subclasses), _attributes (names that define the offset; drive equality,
    # repr, kwds and pickling), _use_relativedelta, _adjust_dst (if True, datetime
    # arithmetic is done on tz-naive wall time then re-localized),
    # __array_priority__ so numpy defers to offset operators. Provides arithmetic
    # (+, -, *, negation), equality/hash via _params, rollback/rollforward,
    # default is_on_offset, pickling, and naming (rule_code, freqstr, name).
    # Subclasses implement _apply (scalar) and optionally _apply_array (vectorized).
    """
    Base class for DateOffset methods that are not overridden by subclasses.

    Parameters
    ----------
    n : int
        Number of multiples of the frequency.

    normalize : bool
        Whether the frequency can align with midnight.

    Examples
    --------
    >>> pd.offsets.Hour(5).n
    5
    >>> pd.offsets.Hour(5).normalize
    False
    """
    # ensure that reversed-ops with numpy scalars return NotImplemented
    __array_priority__ = 1000

    _day_opt = None
    _attributes = tuple(["n", "normalize"])
    _use_relativedelta = False
    _adjust_dst = True

    # cdef readonly:
    #    int64_t n
    #    bint normalize
    #    dict _cache

    def __init__(self, n=1, normalize=False):
        # [FUNCTION DOC]
        # Initialize n (validated integer), normalize, and an empty cache dict.
        # Inputs: n - integer-like multiple; normalize - bool.
        # Exceptions: TypeError/ValueError from _validate_n. Side effects: sets fields.
        n = self._validate_n(n)
        self._n = n
        self._normalize = normalize
        self._cache = {}

    @property
    def n(self):
        # [FUNCTION DOC]
        # Property returning the integer multiple n. No inputs/exceptions/side effects.
        """
        Return the count of the number of periods.

        This represents how many multiples of the offset frequency should
        be applied. For example, ``Hour(5)`` has ``n=5``.

        See Also
        --------
        DateOffset.normalize : Return boolean whether the frequency can align
            with midnight.

        Examples
        --------
        >>> pd.offsets.Hour(5).n
        5

        >>> pd.offsets.Day(3).n
        3
        """
        return self._n

    @property
    def normalize(self) -> bool:
        # [FUNCTION DOC]
        # Property returning the normalize flag (bool). No exceptions/side effects.
        """
        Return boolean whether the frequency can align with midnight.

        This is True for offsets that round the result of a DateOffset
        addition down to the previous midnight.

        See Also
        --------
        tseries.offsets.DateOffset : The standard kind of date increment.

        Examples
        --------
        >>> pd.offsets.Hour(5).normalize
        False

        >>> pd.offsets.Day(5).normalize
        False
        """
        return self._normalize

    def __eq__(self, other) -> bool:
        # [FUNCTION DOC]
        # Equality: strings are first parsed with to_offset (unparseable -> False);
        # then compares _params tuples. Objects without _params compare False.
        # Inputs: other - any object. Outputs: bool. Side effects: may populate cache.
        if isinstance(other, str):
            try:
                # GH#23524 if to_offset fails, we are dealing with an
                #  incomparable type so == is False and != is True
                other = to_offset(other)
            except ValueError:
                # e.g. "infer"
                return False
        try:
            return self._params == other._params
        except AttributeError:
            # other is not a DateOffset object
            return False

    def __ne__(self, other):
        # [FUNCTION DOC]
        # Inequality: logical negation of __eq__. Inputs: other. Outputs: bool.
        return not self == other

    def __hash__(self) -> int:
        # [FUNCTION DOC]
        # Hash of the _params tuple, consistent with __eq__. Outputs: int.
        return hash(self._params)

    @cache_readonly
    def _params(self):
        # [FUNCTION DOC]
        # Cached property returning the canonical identity tuple: the string of the
        # type, followed by sorted (name, value) pairs drawn from __dict__ plus n,
        # normalize and every name in _attributes present on the object; excludes
        # "kwds", "name", "calendar", names starting with "_", and empty "holidays".
        # Outputs: tuple. Exceptions: TypeError if values are not sortable/hashable
        # together. Side effects: stored in self._cache.
        """
        Returns a tuple containing all of the attributes needed to evaluate
        equality between two DateOffset objects.
        """
        d = getattr(self, "__dict__", {})
        all_paras = d.copy()
        all_paras["n"] = self._n
        all_paras["normalize"] = self.normalize
        for attr in self._attributes:
            if hasattr(self, attr) and attr not in d:
                # cython attributes are not in __dict__
                all_paras[attr] = getattr(self, attr)

        if "holidays" in all_paras and not all_paras["holidays"]:
            all_paras.pop("holidays")
        exclude = {"kwds", "name", "calendar"}
        attrs = {(k, v) for k, v in all_paras.items()
                 if (k not in exclude) and (k[0] != "_")}
        params = tuple([str(type(self))] + sorted(attrs))
        return params

    @property
    def kwds(self) -> dict:
        # [FUNCTION DOC]
        # Property returning a dict of the _attributes other than n and normalize,
        # dropping entries whose value is None. Outputs: dict. Side effects: none.
        """
        Return a dict of extra parameters for the offset.

        These parameters exclude ``n`` and ``normalize``, which are common
        to all offsets. The returned dictionary can be passed to the offset
        constructor to recreate the offset with the same settings.

        See Also
        --------
        tseries.offsets.DateOffset : The base class for all pandas date offsets.
        tseries.offsets.WeekOfMonth : Represents the week of the month.
        tseries.offsets.LastWeekOfMonth : Represents the last week of the month.

        Examples
        --------
        >>> pd.DateOffset(5).kwds
        {}

        >>> pd.offsets.FY5253Quarter().kwds
        {'weekday': 0,
         'startingMonth': 1,
         'qtr_with_extra_week': 1,
         'variation': 'nearest'}
        """
        # for backwards-compatibility
        kwds = {name: getattr(self, name, None) for name in self._attributes
                if name not in ["n", "normalize"]}
        return {name: kwds[name] for name in kwds if kwds[name] is not None}

    @property
    def base(self):
        # [FUNCTION DOC]
        # Property returning a new offset of the same type with n=1 and the same
        # normalize/kwds. Outputs: offset instance.
        """
        Returns a copy of the calling offset object with n=1 and all other
        attributes equal.
        """
        return type(self)(n=1, normalize=self.normalize, **self.kwds)

    def __add__(self, other):
        # [FUNCTION DOC]
        # Addition. If other is an object-dtype numpy array, returns an array of
        # element-wise self + x. Otherwise returns self._apply(other), or
        # NotImplemented if _apply raises ApplyTypeError.
        # Inputs: other - datetime-like, timedelta-like, offset or object array.
        # Outputs: result of _apply or ndarray. Exceptions: propagated from _apply.
        if util.is_array(other) and other.dtype == object:
            return np.array([self + x for x in other])

        try:
            return self._apply(other)
        except ApplyTypeError:
            return NotImplemented

    def __radd__(self, other):
        # [FUNCTION DOC]
        # Reflected addition; identical to __add__ (offset addition is commutative).
        return self.__add__(other)

    def __sub__(self, other):
        # [FUNCTION DOC]
        # Subtraction self - other. Raises TypeError if other is a datetime; if other is
        # the same type returns type(self)(self.n - other.n, normalize, **kwds);
        # otherwise NotImplemented.
        # KNOWN ISSUE: kwds of other are ignored, so e.g. MonthEnd vs differently
        # anchored same-type offsets subtract using self's anchoring silently.
        if PyDateTime_Check(other):
            raise TypeError("Cannot subtract datetime from offset.")
        elif type(other) is type(self):
            return type(self)(self._n - other.n, normalize=self.normalize,
                              **self.kwds)
        else:
            # e.g. PeriodIndex
            return NotImplemented

    def __rsub__(self, other):
        # [FUNCTION DOC]
        # Reflected subtraction other - self, computed as (-self) + other.
        return (-self).__add__(other)

    def __mul__(self, other):
        # [FUNCTION DOC]
        # Multiplication. Array -> array of self * x per element; integer -> same type
        # with n multiplied; another offset -> TypeError; else NotImplemented.
        # Inputs: other. Outputs: offset, ndarray or NotImplemented.
        if util.is_array(other):
            return np.array([self * x for x in other])
        elif is_integer_object(other):
            return type(self)(n=other * self._n, normalize=self.normalize,
                              **self.kwds)
        elif isinstance(other, BaseOffset):
            # Otherwise raises RecurrsionError due to __rmul__
            raise TypeError(
                f"Cannot multiply {type(self).__name__} with "
                f"{type(other).__name__}."
            )
        return NotImplemented

    def __rmul__(self, other):
        # [FUNCTION DOC]
        # Reflected multiplication; same as __mul__.
        return self.__mul__(other)

    def __neg__(self):
        # [FUNCTION DOC]
        # Negation: returns self * -1 (same type, n negated).
        # Note: we are deferring directly to __mul__ instead of __rmul__, as
        # that allows us to use methods that can go in a `cdef class`
        return self * -1

    def copy(self):
        # [FUNCTION DOC]
        # Return a copy of the offset (self * 1, a new equal instance).
        # Note: we are deferring directly to __mul__ instead of __rmul__, as
        # that allows us to use methods that can go in a `cdef class`
        """
        Return a copy of the frequency.

        See Also
        --------
        tseries.offsets.Week.copy : Return a copy of Week offset.
        tseries.offsets.DateOffset.copy : Return a copy of date offset.
        tseries.offsets.MonthEnd.copy : Return a copy of MonthEnd offset.
        tseries.offsets.YearBegin.copy : Return a copy of YearBegin offset.

        Examples
        --------
        >>> freq = pd.DateOffset(1)
        >>> freq_copy = freq.copy()
        >>> freq is freq_copy
        False
        """
        return self * 1

    # ------------------------------------------------------------------
    # Name and Rendering Methods

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Repr like "<n * ClassNames: attr=value, ...>": uses _output_name if present,
        # pluralizes when |n| != 1, prefixes "n * " when n != 1, appends _repr_attrs().
        # Outputs: str.
        # _output_name used by B(Year|Quarter)(End|Begin) to
        #  expand "B" -> "Business"
        class_name = getattr(self, "_output_name", type(self).__name__)

        if abs(self._n) != 1:
            plural = "s"
        else:
            plural = ""

        n_str = ""
        if self._n != 1:
            n_str = f"{self._n} * "

        out = f"<{n_str}{class_name}{plural}{self._repr_attrs()}>"
        return out

    def _repr_attrs(self) -> str:
        # [FUNCTION DOC]
        # HELPER: Build the ": a=v, b=w" suffix for repr from sorted _attributes,
        # skipping private names, "kwds", missing attrs, and n/inc/normalize.
        # Outputs: str (empty if no attributes).
        exclude = {"n", "inc", "normalize"}
        attrs = []
        for attr in sorted(self._attributes):
            # _attributes instead of __dict__ because cython attrs are not in __dict__
            if attr.startswith("_") or attr == "kwds" or not hasattr(self, attr):
                # DateOffset may not have some of these attributes
                continue
            elif attr not in exclude:
                value = getattr(self, attr)
                attrs.append(f"{attr}={value}")

        out = ""
        if attrs:
            out += ": " + ", ".join(attrs)
        return out

    @property
    def name(self) -> str:
        # [FUNCTION DOC]
        # Property returning rule_code (the base frequency alias, without n).
        """
        Return a string representing the base frequency.

        See Also
        --------
        tseries.offsets.Week : Represents a weekly offset.
        DateOffset : Base class for all other offset classes.
        tseries.offsets.Day : Represents a single day offset.
        tseries.offsets.MonthEnd : Represents a monthly offset that
            snaps to the end of the month.

        Examples
        --------
        >>> pd.offsets.Hour().name
        'h'

        >>> pd.offsets.Hour(5).name
        'h'
        """
        return self.rule_code

    @property
    def _prefix(self) -> str:
        # [FUNCTION DOC]
        # Property for the frequency prefix alias; base implementation raises
        # NotImplementedError("Prefix not defined"). Subclasses set _prefix class attr.
        raise NotImplementedError("Prefix not defined")

    @property
    def rule_code(self) -> str:
        # [FUNCTION DOC]
        # Property returning the rule code; base implementation returns _prefix.
        # Exceptions: NotImplementedError if no prefix defined.
        """
        Return a string representing the base frequency.

        See Also
        --------
        tseries.offsets.Hour.rule_code :
            Returns a string representing the base frequency of 'h'.
        tseries.offsets.Day.rule_code :
            Returns a string representing the base frequency of 'D'.

        Examples
        --------
        >>> pd.offsets.Hour().rule_code
        'h'

        >>> pd.offsets.Week(5).rule_code
        'W'
        """
        return self._prefix

    @cache_readonly
    def freqstr(self) -> str:
        # [FUNCTION DOC]
        # Cached property returning the frequency string: "{n}{rule_code}" (n omitted
        # when 1), plus _offset_str() if the object has a truthy _offset attribute.
        # If rule_code raises NotImplementedError, returns repr(self).
        # Side effects: cached in self._cache.
        """
        Return a string representing the frequency.

        The string representation typically consists of the offset multiplier
        and a frequency alias (e.g., ``'D'`` for daily, ``'h'`` for hourly).

        See Also
        --------
        tseries.offsets.BusinessDay.freqstr :
            Return a string representing an offset frequency in Business Days.
        tseries.offsets.BusinessHour.freqstr :
            Return a string representing an offset frequency in Business Hours.
        tseries.offsets.Week.freqstr :
            Return a string representing an offset frequency in Weeks.
        tseries.offsets.Hour.freqstr :
            Return a string representing an offset frequency in Hours.

        Examples
        --------
        >>> pd.DateOffset(5).freqstr
        '<5 * DateOffsets>'

        >>> pd.offsets.BusinessHour(2).freqstr
        '2bh'

        >>> pd.offsets.Nano().freqstr
        'ns'

        >>> pd.offsets.Nano(-3).freqstr
        '-3ns'
        """
        try:
            code = self.rule_code
        except NotImplementedError:
            return str(repr(self))

        if self._n != 1:
            fstr = f"{self._n}{code}"
        else:
            fstr = code

        try:
            if self._offset:
                fstr += self._offset_str()
        except AttributeError:
            # TODO: standardize `_offset` vs `offset` naming convention
            pass

        return fstr

    def _offset_str(self) -> str:
        # [FUNCTION DOC]
        # HELPER: Suffix appended to freqstr for offsets with an _offset; base returns
        # empty string.
        return ""

    # ------------------------------------------------------------------

    def _apply(self, other):
        # [FUNCTION DOC]
        # Scalar application hook; base raises NotImplementedError. Subclasses return
        # other shifted by this offset.
        raise NotImplementedError("implemented by subclasses")

    def _apply_array(self, dtarr: np.ndarray) -> np.ndarray:
        # [FUNCTION DOC]
        # Vectorized application hook on a datetime64 ndarray; base raises
        # NotImplementedError. Does not handle normalize (the caller does).
        # NB: _apply_array does not handle respecting `self.normalize`, the
        #  caller (DatetimeArray) handles that in post-processing.
        raise NotImplementedError(
            f"DateOffset subclass {type(self).__name__} "
            "does not have a vectorized implementation"
        )

    def rollback(self, dt) -> datetime:
        # [FUNCTION DOC]
        # Roll dt backward to the nearest on-offset date if not already on it.
        # Inputs: dt - datetime-like (converted to Timestamp).
        # Outputs: Timestamp. If normalize is set and dt is not midnight it is first
        # normalized. If not is_on_offset, subtracts one unit of this offset.
        """
        Roll provided date backward to next offset only if not on offset.

        If the provided date is already on an offset, it is returned unchanged.
        Otherwise, the date is rolled backward to the previous offset boundary.

        Parameters
        ----------
        dt : datetime or Timestamp
            Timestamp to rollback.

        Returns
        -------
        Timestamp
            Rolled timestamp if not on offset, otherwise unchanged timestamp.

        See Also
        --------
        rollforward : Roll provided date forward to next offset only if not on offset.
        is_on_offset : Return boolean whether a timestamp intersects with this
            frequency.

        Examples
        --------
        >>> ts = pd.Timestamp("2025-01-15 09:00:00")
        >>> offset = pd.tseries.offsets.MonthEnd()

        Timestamp is not on the offset (not a month end), so it rolls backward:

        >>> offset.rollback(ts)
        Timestamp('2024-12-31 00:00:00')

        If the timestamp is already on the offset, it remains unchanged:

        >>> ts_on_offset = pd.Timestamp("2025-01-31")
        >>> offset.rollback(ts_on_offset)
        Timestamp('2025-01-31 00:00:00')
        """
        dt = Timestamp(dt)
        if self.normalize and (dt - dt.normalize())._value != 0:
            # GH#32616
            dt = dt.normalize()
        if not self.is_on_offset(dt):
            dt = dt - type(self)(1, normalize=self.normalize, **self.kwds)
        return dt

    def rollforward(self, dt) -> datetime:
        # [FUNCTION DOC]
        # Roll dt forward to the nearest on-offset date if not already on it.
        # Inputs: dt - datetime-like (converted to Timestamp).
        # Outputs: Timestamp; adds one unit of this offset when not on offset.
        """
        Roll provided date forward to next offset only if not on offset.

        If the provided date is already on an offset, it is returned unchanged.
        Otherwise, the date is rolled forward to the next offset boundary.

        Parameters
        ----------
        dt : datetime or Timestamp
            Timestamp to roll forward.

        Returns
        -------
        Timestamp
            Rolled timestamp if not on offset, otherwise unchanged timestamp.

        See Also
        --------
        rollback : Roll provided date backward to previous offset only if not
            on offset.
        is_on_offset : Return boolean whether a timestamp intersects with this
            frequency.

        Examples
        --------
        >>> ts = pd.Timestamp("2025-01-15 09:00:00")
        >>> offset = pd.tseries.offsets.MonthEnd()

        Timestamp is not on the offset (not a month end), so it rolls forward:

        >>> offset.rollforward(ts)
        Timestamp('2025-01-31 00:00:00')

        If the timestamp is already on the offset, it remains unchanged:

        >>> ts_on_offset = pd.Timestamp("2025-01-31")
        >>> offset.rollforward(ts_on_offset)
        Timestamp('2025-01-31 00:00:00')
        """
        dt = Timestamp(dt)
        if not self.is_on_offset(dt):
            dt = dt + type(self)(1, normalize=self.normalize, **self.kwds)
        return dt

    def _get_offset_day(self, other: datetime) -> int:
        # [FUNCTION DOC]
        # HELPER: Day of month that this offset anchors to for the month of other,
        # determined by self._day_opt (start, end, business start/end).
        # Inputs: other - datetime. Outputs: int day.
        # Exceptions: ValueError if _day_opt is not a valid day_opt string.
        # KNOWN ISSUE: the inline comment claims the base class implicitly uses
        # "business_end", but with _day_opt None _str_to_day_opt raises ValueError.
        # subclass must implement `_day_opt`; calling from the base class
        # will implicitly assume day_opt = "business_end", see get_day_of_month.
        cdef:
            npy_datetimestruct dts
        pydate_to_dtstruct(other, &dts)
        return get_day_of_month(&dts, _str_to_day_opt(self._day_opt))

    def is_on_offset(self, dt: datetime) -> bool:
        # [FUNCTION DOC]
        # Default membership test: False if normalize is set and dt is not midnight;
        # otherwise True iff (dt + self) - self == dt. Inputs: dt - datetime.
        # Outputs: bool.
        # KNOWN ISSUE: the docstring describes custom business month start semantics,
        # which do not apply to this generic base implementation.
        """
        Return boolean whether a timestamp intersects with this frequency.

        This method determines if a given timestamp aligns with the start
        of a custom business month, as defined by this offset. It accounts
        for custom rules, such as skipping weekends or other non-business days,
        and checks whether the provided datetime falls on a valid business day
        that marks the beginning of the custom business month.

        Parameters
        ----------
        dt : datetime.datetime
            Timestamp to check intersections with frequency.

        See Also
        --------
        tseries.offsets.CustomBusinessMonthBegin : Represents the start of a custom
            business month.
        tseries.offsets.CustomBusinessMonthEnd : Represents the end of a custom
            business month.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> freq = pd.offsets.Day(1)
        >>> freq.is_on_offset(ts)
        True

        >>> ts = pd.Timestamp(2022, 8, 6)
        >>> ts.day_name()
        'Saturday'
        >>> freq = pd.offsets.BusinessDay(1)
        >>> freq.is_on_offset(ts)
        False
        """
        if self.normalize and not _is_normalized(dt):
            return False

        # Default (slow) method for determining if some date is a member of the
        # date range generated by this offset. Subclasses may have this
        # re-implemented in a nicer way.
        a = dt
        b = (dt + self) - self
        return a == b

    # ------------------------------------------------------------------

    # Staticmethod so we can call from Tick.__init__, will be unnecessary
    #  once BaseOffset is a cdef class and is inherited by Tick
    @staticmethod
    def _validate_n(n) -> int:
        # [FUNCTION DOC]
        # Static method: validate n is integral. Outputs: int(n).
        # Exceptions: TypeError if n is a timedelta64 or int(n) fails; ValueError if
        # n != int(n) (e.g. 1.5).
        """
        Require that `n` be an integer.

        Parameters
        ----------
        n : int

        Returns
        -------
        nint : int

        Raises
        ------
        TypeError if `int(n)` raises
        ValueError if n != int(n)
        """
        if cnp.is_timedelta64_object(n):
            raise TypeError(f"`n` argument must be an integer, got {type(n)}")
        try:
            nint = int(n)
        except (ValueError, TypeError):
            raise TypeError(f"`n` argument must be an integer, got {type(n)}")
        if n != nint:
            raise ValueError(f"`n` argument must be an integer, got {n}")
        return nint

    def __setstate__(self, state):
        # [FUNCTION DOC]
        # Restore n, normalize and _cache from a pickled state dict; pops those keys,
        # mutating state (expected to be empty afterwards).
        """
        Reconstruct an instance from a pickled state
        """
        self._n = state.pop("n")
        self._normalize = state.pop("normalize")
        self._cache = state.pop("_cache", {})
        # At this point we expect state to be empty

    def __getstate__(self):
        # [FUNCTION DOC]
        # Return pickle state dict containing n and normalize.
        # KNOWN ISSUE: the calendar/kwds removal acts on a freshly created dict that
        # never contains those keys, so it is dead code.
        """
        Return a picklable state
        """
        state = {}
        state["n"] = self._n
        state["normalize"] = self.normalize

        # we don't want to actually pickle the calendar object
        # as its an np.busyday; we recreate on deserialization
        state.pop("calendar", None)
        if "kwds" in state:
            state["kwds"].pop("calendar", None)

        return state

    @property
    def nanos(self):
        # [FUNCTION DOC]
        # Property: total nanoseconds for fixed frequencies; base always raises
        # ValueError("... is a non-fixed frequency").
        """
        Returns an integer of the total number of nanoseconds for fixed frequencies.

        Raises
        ------
        ValueError
            If the frequency is non-fixed.

        See Also
        --------
        tseries.offsets.Hour.nanos :
            Returns an integer of the total number of nanoseconds.
        tseries.offsets.Day.nanos :
            Returns an integer of the total number of nanoseconds.

        Examples
        --------
        >>> pd.offsets.Week(n=1).nanos
        ValueError: Week: weekday=None is a non-fixed frequency
        """
        raise ValueError(f"{self} is a non-fixed frequency")

    # ------------------------------------------------------------------

    def is_month_start(self, _Timestamp ts):
        # [FUNCTION DOC]
        # Return whether ts is a month start, delegating to
        # ts._get_start_end_field("is_month_start", self) which honors this offset's
        # frequency (e.g. business month starts). Inputs: ts - Timestamp. Outputs: bool.
        """
        Return boolean whether a timestamp occurs on the month start.

        This method determines if a given timestamp falls on the first day
        of a month, considering the frequency of the offset.

        Parameters
        ----------
        ts : Timestamp
            The timestamp to check.

        See Also
        --------
        is_month_end : Return boolean whether a timestamp occurs on the month end.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> freq = pd.offsets.Hour(5)
        >>> freq.is_month_start(ts)
        True
        """
        return ts._get_start_end_field("is_month_start", self)

    def is_month_end(self, _Timestamp ts):
        # [FUNCTION DOC]
        # Return whether ts is a month end via ts._get_start_end_field with this
        # offset. Inputs: ts - Timestamp. Outputs: bool.
        """
        Return boolean whether a timestamp occurs on the month end.

        Parameters
        ----------
        ts : Timestamp
            The timestamp to check.

        See Also
        --------
        is_month_start : Return boolean whether a timestamp occurs on the month start.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> freq = pd.offsets.Hour(5)
        >>> freq.is_month_end(ts)
        False
        """
        return ts._get_start_end_field("is_month_end", self)

    def is_quarter_start(self, _Timestamp ts):
        # [FUNCTION DOC]
        # Return whether ts is a quarter start via ts._get_start_end_field with this
        # offset (respects quarter anchoring). Inputs: ts - Timestamp. Outputs: bool.
        """
        Return boolean whether a timestamp occurs on the quarter start.

        Parameters
        ----------
        ts : Timestamp
            The timestamp to check.

        See Also
        --------
        is_quarter_end : Return boolean whether a timestamp occurs on the quarter end.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> freq = pd.offsets.Hour(5)
        >>> freq.is_quarter_start(ts)
        True
        """
        return ts._get_start_end_field("is_quarter_start", self)

    def is_quarter_end(self, _Timestamp ts):
        # [FUNCTION DOC]
        # Return whether ts is a quarter end via ts._get_start_end_field with this
        # offset. Inputs: ts - Timestamp. Outputs: bool.
        """
        Return boolean whether a timestamp occurs on the quarter end.

        Parameters
        ----------
        ts : Timestamp
            The timestamp to check.

        See Also
        --------
        is_quarter_start : Return boolean whether a timestamp
            occurs on the quarter start.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> freq = pd.offsets.Hour(5)
        >>> freq.is_quarter_end(ts)
        False
        """
        return ts._get_start_end_field("is_quarter_end", self)

    def is_year_start(self, _Timestamp ts):
        # [FUNCTION DOC]
        # Return whether ts is a year start via ts._get_start_end_field with this
        # offset. Inputs: ts - Timestamp. Outputs: bool.
        """
        Return boolean whether a timestamp occurs on the year start.

        Parameters
        ----------
        ts : Timestamp
            The timestamp to check.

        See Also
        --------
        is_year_end : Return boolean whether a timestamp occurs on the year end.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> freq = pd.offsets.Hour(5)
        >>> freq.is_year_start(ts)
        True
        """
        return ts._get_start_end_field("is_year_start", self)

    def is_year_end(self, _Timestamp ts):
        # [FUNCTION DOC]
        # Return whether ts is a year end via ts._get_start_end_field with this
        # offset. Inputs: ts - Timestamp. Outputs: bool.
        """
        Return boolean whether a timestamp occurs on the year end.

        Parameters
        ----------
        ts : Timestamp
            The timestamp to check.

        See Also
        --------
        is_year_start : Return boolean whether a timestamp occurs on the year start.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> freq = pd.offsets.Hour(5)
        >>> freq.is_year_end(ts)
        False
        """
        return ts._get_start_end_field("is_year_end", self)


cdef class SingleConstructorOffset(BaseOffset):
    # [CLASS DOC]
    # Base for offsets fully reconstructible from their _attributes tuple. Adds the
    # default _from_name (string-alias construction) and a __reduce__ that pickles
    # via the constructor using attribute values in _attributes order.
    @classmethod
    def _from_name(cls, suffix=None):
        # [FUNCTION DOC]
        # Classmethod: construct the offset from a frequency-string suffix. Default
        # accepts no suffix and returns cls().
        # Inputs: suffix - str or None. Exceptions: ValueError("Bad freq suffix") if a
        # non-empty suffix is given.
        # default _from_name calls cls with no args
        if suffix:
            raise ValueError(f"Bad freq suffix {suffix}")
        return cls()

    def __reduce__(self):
        # [FUNCTION DOC]
        # Pickle support: returns (type(self), tuple of getattr(self, a) for each name
        # in _attributes), substituting None for "calendar" (numpy busdaycalendar does
        # not pickle; it is rebuilt from weekmask/holidays on construction).
        # This __reduce__ implementation is for all BaseOffset subclasses
        #  except for RelativeDeltaOffset
        # np.busdaycalendar objects do not pickle nicely, but we can reconstruct
        #  from attributes that do get pickled.
        tup = tuple(
            getattr(self, attr) if attr != "calendar" else None
            for attr in self._attributes
        )
        return type(self), tup


# ---------------------------------------------------------------------
# Tick Offsets

cdef class Tick(SingleConstructorOffset):
    # [CLASS DOC]
    # Base for fixed-duration sub-daily offsets (Hour, Minute, Second, Milli,
    # Micro, Nano). Each subclass defines _nanos_inc (nanoseconds per unit),
    # _prefix (alias), _period_dtype_code and _creso (numpy unit). normalize is
    # always False and _adjust_dst is False (arithmetic in absolute time).
    # Arithmetic and comparisons delegate to the equivalent pandas Timedelta and
    # results that are timedeltas are converted back to Tick objects.
    """
    Base class for fixed frequency offsets (Milli, Micro, Second, Minute, Hour).

    This class should not be instantiated directly. Use one of the specific
    Tick subclasses for a concrete offset.

    Attributes
    ----------
    n : int, default 1
        The number of units (hours, minutes, etc.) the offset represents.

    See Also
    --------
    Hour : Offset ``n`` hours.
    Minute : Offset ``n`` minutes.
    Second : Offset ``n`` seconds.
    Milli : Offset ``n`` milliseconds.
    Micro : Offset ``n`` microseconds.
    Nano : Offset ``n`` nanoseconds.

    Examples
    --------
    >>> from pandas.tseries.offsets import Hour, Minute
    >>> ts = pd.Timestamp(2022, 12, 9, 15)
    >>> ts
    Timestamp('2022-12-09 15:00:00')

    >>> ts + Hour(1)
    Timestamp('2022-12-09 16:00:00')

    >>> ts + Minute(30)
    Timestamp('2022-12-09 15:30:00')
    """
    _adjust_dst = False
    _prefix = "undefined"
    _attributes = tuple(["n", "normalize"])

    def __init__(self, n=1, normalize=False):
        # [FUNCTION DOC]
        # Initialize with validated n; normalize is forced False.
        # Inputs: n - integer-like; normalize - must be falsy.
        # Exceptions: TypeError/ValueError from _validate_n; ValueError if normalize
        # is truthy (GH#21427). Side effects: sets _n, _normalize, _cache.
        n = self._validate_n(n)
        self._n = n
        self._normalize = False
        self._cache = {}
        if normalize:
            # GH#21427
            raise ValueError(
                "Tick offset with `normalize=True` are not allowed."
            )

    # Note: Without making this cpdef, we get AttributeError when calling
    #  from __mul__
    cpdef Tick _next_higher_resolution(Tick self):
        # [FUNCTION DOC]
        # Return an equivalent Tick at the next finer unit (Hour->Minute*60,
        # Minute->Second*60, Second->Milli*1000, Milli->Micro*1000, Micro->Nano*1000).
        # Used when multiplying by a float yields a non-integer n.
        # Outputs: new Tick. Exceptions: ValueError for Nano (no finer unit).
        if type(self) is Hour:
            return Minute(self._n * 60)
        if type(self) is Minute:
            return Second(self._n * 60)
        if type(self) is Second:
            return Milli(self._n * 1000)
        if type(self) is Milli:
            return Micro(self._n * 1000)
        if type(self) is Micro:
            return Nano(self._n * 1000)
        raise ValueError("Could not convert to integer offset at any resolution")

    # --------------------------------------------------------------------

    def _repr_attrs(self) -> str:
        # [FUNCTION DOC]
        # Repr attribute suffix; Ticks have none, returns empty string.
        # Since cdef classes have no __dict__, we need to override
        return ""

    @cache_readonly
    def _as_pd_timedelta(self):
        # [FUNCTION DOC]
        # Cached property: the Timedelta equal to this tick (n * unit).
        # Side effects: cached in self._cache.
        return Timedelta(self)

    @property
    def nanos(self) -> int64_t:
        # [FUNCTION DOC]
        # Property: total nanoseconds, n * _nanos_inc. Outputs: integer.
        """
        Returns an integer of the total number of nanoseconds.

        See Also
        --------
        tseries.offsets.Hour.nanos :
            Returns an integer of the total number of nanoseconds.
        tseries.offsets.Day.nanos :
            Returns an integer of the total number of nanoseconds.

        Examples
        --------
        >>> pd.offsets.Hour(5).nanos
        18000000000000
        """
        return self._n * self._nanos_inc

    def is_on_offset(self, dt: datetime) -> bool:
        # [FUNCTION DOC]
        # Every timestamp is on a Tick offset; always returns True.
        return True

    # This is identical to BaseOffset.__hash__, but has to be redefined here
    # for Python 3, because we've redefined __eq__.
    def __hash__(self) -> int:
        # [FUNCTION DOC]
        # Hash of _params (redefined because __eq__ is overridden).
        return hash(self._params)

    # --------------------------------------------------------------------
    # Comparison and Arithmetic Methods

    def __eq__(self, other):
        # [FUNCTION DOC]
        # Equality: strings parsed via to_offset (failure -> False); then compares the
        # equivalent Timedelta with other, so e.g. Hour(1) == Minute(60) and
        # Hour(1) == timedelta(hours=1). Outputs: bool.
        # KNOWN ISSUE: equal objects of different types (Hour(1) vs Minute(60)) may
        # have different hashes since __hash__ uses _params (which includes the type).
        if isinstance(other, str):
            try:
                # GH#23524 if to_offset fails, we are dealing with an
                #  incomparable type so == is False and != is True
                other = to_offset(other)
            except ValueError:
                # e.g. "infer"
                return False
        return self._as_pd_timedelta == other

    def __ne__(self, other):
        # [FUNCTION DOC]
        # Inequality: negation of __eq__.
        return not (self == other)

    def __le__(self, other):
        # [FUNCTION DOC]
        # Less-or-equal comparison delegated to the equivalent Timedelta.
        return self._as_pd_timedelta.__le__(other)

    def __lt__(self, other):
        # [FUNCTION DOC]
        # Less-than comparison delegated to the equivalent Timedelta.
        return self._as_pd_timedelta.__lt__(other)

    def __ge__(self, other):
        # [FUNCTION DOC]
        # Greater-or-equal comparison delegated to the equivalent Timedelta.
        return self._as_pd_timedelta.__ge__(other)

    def __gt__(self, other):
        # [FUNCTION DOC]
        # Greater-than comparison delegated to the equivalent Timedelta.
        return self._as_pd_timedelta.__gt__(other)

    def __mul__(self, other):
        # [FUNCTION DOC]
        # Multiplication. For a float, computes n*other; if (approximately) integral
        # returns same type with that n, otherwise converts to the next finer
        # resolution and retries (recursively). Non-floats use BaseOffset.__mul__.
        # Exceptions: ValueError if no finer resolution makes it integral.
        if is_float_object(other):
            n = other * self._n
            # If the new `n` is an integer, we can represent it using the
            #  same Tick subclass as self, otherwise we need to move up
            #  to a higher-resolution subclass
            if np.isclose(n % 1, 0):
                return type(self)(int(n))
            new_self = self._next_higher_resolution()
            return new_self * other
        return BaseOffset.__mul__(self, other)

    def __rmul__(self, other):
        # [FUNCTION DOC]
        # Reflected multiplication; same as __mul__.
        return self.__mul__(other)

    def __truediv__(self, other):
        # [FUNCTION DOC]
        # True division delegated to Timedelta; timedelta results converted back to a
        # Tick. Handles Cython's possible argument swap (self not a Tick) by calling
        # other's reflected division. Outputs: Tick, float, or array.
        if not isinstance(self, Tick):
            # cython semantics mean the args are sometimes swapped
            result = other._as_pd_timedelta.__rtruediv__(self)
        else:
            result = self._as_pd_timedelta.__truediv__(other)
        return _wrap_timedelta_result(result)

    def __rtruediv__(self, other):
        # [FUNCTION DOC]
        # Reflected true division other / self via Timedelta; timedelta results become
        # Ticks.
        result = self._as_pd_timedelta.__rtruediv__(other)
        return _wrap_timedelta_result(result)

    def __add__(self, other):
        # [FUNCTION DOC]
        # Addition. Tick + same-type Tick -> same type with summed n; Tick + other Tick
        # -> delta_to_tick of the summed Timedeltas; otherwise self._apply(other).
        # Outputs: Tick, Timestamp, Timedelta, NaT or NotImplemented (ApplyTypeError).
        # Exceptions: OverflowError re-raised with a descriptive message.
        if isinstance(other, Tick):
            if type(self) is type(other):
                return type(self)(self._n + other.n)
            else:
                return delta_to_tick(self._as_pd_timedelta + other._as_pd_timedelta)
        try:
            return self._apply(other)
        except ApplyTypeError:
            # Includes pd.Period
            return NotImplemented
        except OverflowError as err:
            raise OverflowError(
                f"the add operation between {self} and {other} will overflow"
            ) from err

    def __radd__(self, other):
        # [FUNCTION DOC]
        # Reflected addition; same as __add__.
        return self.__add__(other)

    def _apply(self, other):
        # [FUNCTION DOC]
        # Apply tick to other without apply_wraps: Timestamp -> other + Timedelta;
        # NaT -> NaT; datetime64/date/datetime -> Timestamp(other) + self;
        # timedelta64/timedelta -> other + Timedelta.
        # Exceptions: ApplyTypeError for other types.
        # Timestamp can handle tz and nano sec, thus no need to use apply_wraps
        if isinstance(other, _Timestamp):
            # GH#15126
            return other + self._as_pd_timedelta
        elif other is NaT:
            return NaT
        elif cnp.is_datetime64_object(other) or PyDate_Check(other):
            # PyDate_Check includes date, datetime
            return Timestamp(other) + self

        if cnp.is_timedelta64_object(other) or PyDelta_Check(other):
            return other + self._as_pd_timedelta

        raise ApplyTypeError(f"Unhandled type: {type(other).__name__}")

    # --------------------------------------------------------------------
    # Pickle Methods

    def __setstate__(self, state):
        # [FUNCTION DOC]
        # Unpickle: set _n from state["n"] and force normalize False. Does not mutate
        # state.
        self._n = state["n"]
        self._normalize = False


cdef class Day(SingleConstructorOffset):
    # [CLASS DOC]
    # Calendar day offset (n days). Not a Tick: _adjust_dst is True, so adding to
    # tz-aware timestamps moves wall-clock days (23/25h across DST). normalize is
    # not allowed. _nanos_inc is 24h for nanos lookups.
    """
    Offset ``n`` days.

    Attributes
    ----------
    n : int, default 1
        The number of days represented.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    You can use the parameter ``n`` to represent a shift of n days.

    >>> from pandas.tseries.offsets import Day
    >>> ts = pd.Timestamp(2022, 12, 9, 15)
    >>> ts
    Timestamp('2022-12-09 15:00:00')

    >>> ts + Day()
    Timestamp('2022-12-10 15:00:00')
    >>> ts - Day(4)
    Timestamp('2022-12-05 15:00:00')

    >>> ts + Day(-4)
    Timestamp('2022-12-05 15:00:00')
    """
    _adjust_dst = True
    _attributes = tuple(["n", "normalize"])
    _nanos_inc = 24 * 3600 * 1_000_000_000
    _prefix = "D"
    _period_dtype_code = PeriodDtypeCode.D
    _creso = NPY_DATETIMEUNIT.NPY_FR_D

    def __init__(self, n=1, normalize=False):
        # [FUNCTION DOC]
        # Initialize with validated n (normalize False).
        # Exceptions: ValueError if normalize truthy; TypeError/ValueError from
        # _validate_n.
        BaseOffset.__init__(self, n)
        if normalize:
            # GH#21427
            raise ValueError(
                "Day offset with `normalize=True` are not allowed."
            )

    def is_on_offset(self, dt) -> bool:
        # [FUNCTION DOC]
        # Every timestamp is on a Day offset; returns True.
        return True

    @apply_wraps
    def _apply(self, other):
        # [FUNCTION DOC]
        # Apply (wrapped by apply_wraps): Day + Day -> Day(n sum); otherwise
        # other + n days as numpy timedelta64[D]. Timezone handled by apply_wraps.
        if isinstance(other, Day):
            # TODO: why isn't this handled in __add__?
            return Day(self._n + other.n)
        return other + np.timedelta64(self._n, "D")

    def _apply_array(self, dtarr):
        # [FUNCTION DOC]
        # Vectorized: dtarr + timedelta64(n, 'D'). Inputs: datetime64 ndarray.
        # Outputs: new ndarray.
        return dtarr + np.timedelta64(self._n, "D")

    @cache_readonly
    def freqstr(self) -> str:
        # [FUNCTION DOC]
        # Cached property: "D" if n == 1 else "{n}D".
        """
        Return a string representing the frequency.

        The frequency string is composed of a multiplier (if greater than 1)
        followed by the offset alias 'D' for days.

        See Also
        --------
        tseries.offsets.Hour.freqstr :
            Return a string representing an offset frequency in hours.
        tseries.offsets.Minute.freqstr :
            Return a string representing an offset frequency in minutes.
        tseries.offsets.Second.freqstr :
            Return a string representing an offset frequency in seconds.

        Examples
        --------
        >>> pd.Day(5).freqstr
        '5D'

        >>> pd.offsets.Day(1).freqstr
        'D'
        """
        if self._n != 1:
            return str(self._n) + "D"
        return "D"

    # Having this here isn't strictly-correct post-GH#61985
    #  but this gets called in timedelta.get_unit_for_round in cases where
    #  Day unambiguously means 24h.
    @property
    def nanos(self) -> int64_t:
        # [FUNCTION DOC]
        # Property: n * 24h in nanoseconds (treats a day as exactly 24 hours).
        """
        Returns an integer of the total number of nanoseconds.

        See Also
        --------
        tseries.offsets.Hour.nanos :
            Returns an integer of the total number of nanoseconds.
        tseries.offsets.Day.nanos :
            Returns an integer of the total number of nanoseconds.

        Examples
        --------
        >>> pd.offsets.Hour(5).nanos
        18000000000000
        """
        return self._n * self._nanos_inc


cdef class Hour(Tick):
    # [CLASS DOC]
    # Fixed offset of n hours (alias "h", 3.6e12 ns per unit).
    """
    Offset ``n`` hours.

    Attributes
    ----------
    n : int, default 1
        The number of hours represented.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    You can use the parameter ``n`` to represent a shift of n hours.

    >>> from pandas.tseries.offsets import Hour
    >>> ts = pd.Timestamp(2022, 12, 9, 15)
    >>> ts
    Timestamp('2022-12-09 15:00:00')

    >>> ts + Hour()
    Timestamp('2022-12-09 16:00:00')
    >>> ts - Hour(4)
    Timestamp('2022-12-09 11:00:00')

    >>> ts + Hour(-4)
    Timestamp('2022-12-09 11:00:00')
    """
    _nanos_inc = 3600 * 1_000_000_000
    _prefix = "h"
    _period_dtype_code = PeriodDtypeCode.H
    _creso = NPY_DATETIMEUNIT.NPY_FR_h


cdef class Minute(Tick):
    # [CLASS DOC]
    # Fixed offset of n minutes (alias "min", 6e10 ns per unit).
    """
    Offset ``n`` minutes.

    Attributes
    ----------
    n : int, default 1
        The number of minutes represented.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    You can use the parameter ``n`` to represent a shift of n minutes.

    >>> from pandas.tseries.offsets import Minute
    >>> ts = pd.Timestamp(2022, 12, 9, 15)
    >>> ts
    Timestamp('2022-12-09 15:00:00')

    >>> ts + Minute(n=10)
    Timestamp('2022-12-09 15:10:00')
    >>> ts - Minute(n=10)
    Timestamp('2022-12-09 14:50:00')

    >>> ts + Minute(n=-10)
    Timestamp('2022-12-09 14:50:00')
    """
    _nanos_inc = 60 * 1_000_000_000
    _prefix = "min"
    _period_dtype_code = PeriodDtypeCode.T
    _creso = NPY_DATETIMEUNIT.NPY_FR_m


cdef class Second(Tick):
    # [CLASS DOC]
    # Fixed offset of n seconds (alias "s", 1e9 ns per unit).
    """
    Offset ``n`` seconds.

    Attributes
    ----------
    n : int, default 1
        The number of seconds represented.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    You can use the parameter ``n`` to represent a shift of n seconds.

    >>> from pandas.tseries.offsets import Second
    >>> ts = pd.Timestamp(2022, 12, 9, 15)
    >>> ts
    Timestamp('2022-12-09 15:00:00')

    >>> ts + Second(n=10)
    Timestamp('2022-12-09 15:00:10')
    >>> ts - Second(n=10)
    Timestamp('2022-12-09 14:59:50')

    >>> ts + Second(n=-10)
    Timestamp('2022-12-09 14:59:50')
    """
    _nanos_inc = 1_000_000_000
    _prefix = "s"
    _period_dtype_code = PeriodDtypeCode.S
    _creso = NPY_DATETIMEUNIT.NPY_FR_s


cdef class Milli(Tick):
    # [CLASS DOC]
    # Fixed offset of n milliseconds (alias "ms", 1e6 ns per unit).
    """
    Offset ``n`` milliseconds.

    Attributes
    ----------
    n : int, default 1
        The number of milliseconds represented.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    You can use the parameter ``n`` to represent a shift of n milliseconds.

    >>> from pandas.tseries.offsets import Milli
    >>> ts = pd.Timestamp(2022, 12, 9, 15)
    >>> ts
    Timestamp('2022-12-09 15:00:00')

    >>> ts + Milli(n=10)
    Timestamp('2022-12-09 15:00:00.010000')

    >>> ts - Milli(n=10)
    Timestamp('2022-12-09 14:59:59.990000')

    >>> ts + Milli(n=-10)
    Timestamp('2022-12-09 14:59:59.990000')
    """
    _nanos_inc = 1_000_000
    _prefix = "ms"
    _period_dtype_code = PeriodDtypeCode.L
    _creso = NPY_DATETIMEUNIT.NPY_FR_ms


cdef class Micro(Tick):
    # [CLASS DOC]
    # Fixed offset of n microseconds (alias "us", 1000 ns per unit).
    """
    Offset ``n`` microseconds.

    Attributes
    ----------
    n : int, default 1
        The number of microseconds represented.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    You can use the parameter ``n`` to represent a shift of n microseconds.

    >>> from pandas.tseries.offsets import Micro
    >>> ts = pd.Timestamp(2022, 12, 9, 15)
    >>> ts
    Timestamp('2022-12-09 15:00:00')

    >>> ts + Micro(n=1000)
    Timestamp('2022-12-09 15:00:00.001000')

    >>> ts - Micro(n=1000)
    Timestamp('2022-12-09 14:59:59.999000')

    >>> ts + Micro(n=-1000)
    Timestamp('2022-12-09 14:59:59.999000')
    """
    _nanos_inc = 1000
    _prefix = "us"
    _period_dtype_code = PeriodDtypeCode.U
    _creso = NPY_DATETIMEUNIT.NPY_FR_us


cdef class Nano(Tick):
    # [CLASS DOC]
    # Fixed offset of n nanoseconds (alias "ns", 1 ns per unit).
    """
    Offset ``n`` nanoseconds.

    Attributes
    ----------
    n : int, default 1
        The number of nanoseconds represented.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    You can use the parameter ``n`` to represent a shift of n nanoseconds.

    >>> from pandas.tseries.offsets import Nano
    >>> ts = pd.Timestamp(2022, 12, 9, 15)
    >>> ts
    Timestamp('2022-12-09 15:00:00')

    >>> ts + Nano(n=1000)
    Timestamp('2022-12-09 15:00:00.000001')

    >>> ts - Nano(n=1000)
    Timestamp('2022-12-09 14:59:59.999999')

    >>> ts + Nano(n=-1000)
    Timestamp('2022-12-09 14:59:59.999999')
    """
    _nanos_inc = 1
    _prefix = "ns"
    _period_dtype_code = PeriodDtypeCode.N
    _creso = NPY_DATETIMEUNIT.NPY_FR_ns


def delta_to_tick(delta: timedelta) -> Tick:
    # [FUNCTION DOC]
    # Convert a timedelta to the coarsest exactly representing Tick.
    # Inputs: delta - datetime.timedelta or Timedelta.
    # Outputs: Hour, Minute or Second if no sub-second component (based on total
    # seconds), else Milli, Micro or Nano based on total nanoseconds.
    # Exceptions: OverflowError from nanosecond conversion for huge deltas.
    # Side effects: none.
    if delta.microseconds == 0 and getattr(delta, "nanoseconds", 0) == 0:
        # nanoseconds only for pd.Timedelta
        seconds = delta.days * 86400 + delta.seconds
        if seconds % 3600 == 0:
            return Hour(seconds / 3600)
        elif seconds % 60 == 0:
            return Minute(seconds / 60)
        else:
            return Second(seconds)
    else:
        nanos = delta_to_nanoseconds(delta)
        if nanos % 1_000_000 == 0:
            return Milli(nanos // 1_000_000)
        elif nanos % 1000 == 0:
            return Micro(nanos // 1000)
        else:  # pragma: no cover
            return Nano(nanos)


# --------------------------------------------------------------------

cdef class RelativeDeltaOffset(BaseOffset):
    # [CLASS DOC]
    # Offset backed by a timedelta or dateutil relativedelta (see
    # _determine_offset). Keyword arguments are stored as instance attributes;
    # _offset holds the increment and _use_relativedelta selects the path.
    # _adjust_dst False. Base of the public DateOffset class.
    """
    DateOffset subclass backed by a dateutil relativedelta object.
    """
    _attributes = tuple(["n", "normalize"] + list(_relativedelta_kwds))
    _adjust_dst = False

    def __init__(self, n=1, normalize=False, **kwds):
        # [FUNCTION DOC]
        # Initialize n/normalize then compute _offset/_use_relativedelta from kwds and
        # set each kwd as an attribute (bypassing DateOffset's immutability).
        # Exceptions: as _determine_offset and _validate_n.
        BaseOffset.__init__(self, n, normalize)
        off, use_rd = _determine_offset(kwds)
        object.__setattr__(self, "_offset", off)
        object.__setattr__(self, "_use_relativedelta", use_rd)
        for key in kwds:
            val = kwds[key]
            object.__setattr__(self, key, val)

    def __getstate__(self):
        # [FUNCTION DOC]
        # Return pickle state: copy of __dict__ plus n and normalize.
        """
        Return a picklable state
        """
        # RelativeDeltaOffset (technically DateOffset) is the only non-cdef
        #  class, so the only one with __dict__
        state = self.__dict__.copy()
        state["n"] = self._n
        state["normalize"] = self.normalize
        return state

    def __setstate__(self, state):
        # [FUNCTION DOC]
        # Unpickle: migrate legacy "offset" key to "_offset" (also into kwds), restore
        # n/normalize/_cache, then update __dict__ with remaining keys.
        # Exceptions: AssertionError if both "offset" and "_offset" exist.
        # Side effects: mutates state.
        """
        Reconstruct an instance from a pickled state
        """

        if "offset" in state:
            # Older (<0.22.0) versions have offset attribute instead of _offset
            if "_offset" in state:  # pragma: no cover
                raise AssertionError("Unexpected key `_offset`")
            state["_offset"] = state.pop("offset")
            state["kwds"]["offset"] = state["_offset"]

        self._n = state.pop("n")
        self._normalize = state.pop("normalize")
        self._cache = state.pop("_cache", {})

        self.__dict__.update(state)

    @apply_wraps
    def _apply(self, other: datetime) -> datetime:
        # [FUNCTION DOC]
        # Apply (wrapped by apply_wraps). With no kwds adds n days. Otherwise: for
        # relativedelta path, Timestamps are converted to pydatetime (saving the
        # nanosecond) and tz-aware values are made naive; adds _offset * n, adds n *
        # nanoseconds if set, re-adds the saved nanosecond, re-localizes the tz, and
        # returns a Timestamp.
        other_nanos = 0
        if self._use_relativedelta:
            if isinstance(other, _Timestamp):
                other_nanos = other.nanosecond
                other = other.to_pydatetime(warn=False)

        if len(self.kwds) > 0:
            tzinfo = getattr(other, "tzinfo", None)
            if tzinfo is not None and self._use_relativedelta:
                # perform calculation in UTC
                other = other.replace(tzinfo=None)

            other = other + (self._offset * self._n)

            if hasattr(self, "nanoseconds"):
                other = self._n * Timedelta(nanoseconds=self.nanoseconds) + other
            if other_nanos != 0:
                other = Timedelta(nanoseconds=other_nanos) + other

            if tzinfo is not None and self._use_relativedelta:
                # bring tz back from UTC calculation
                other = localize_pydatetime(other, tzinfo)

            return Timestamp(other)
        else:
            return other + timedelta(self._n)

    @cache_readonly
    def _pd_timedelta(self) -> Timedelta:
        # [FUNCTION DOC]
        # Cached property: Timedelta equivalent of the fixed part of the offset, used by
        # _apply_array. relativedelta path with only plural keys: Timedelta of
        # days(+7*weeks)/hours/minutes/seconds/ms/us times n (years/months excluded,
        # handled separately), unit us/ms/s by finest key. timedelta path: _offset
        # (plus nanoseconds) times n. Exceptions: NotImplementedError when replacement
        # keys (year, day, hour...) make vectorization impossible.
        # components of _offset that can be cast to pd.Timedelta

        kwds = self.kwds
        relativedelta_fast = {
            "years",
            "months",
            "weeks",
            "days",
            "hours",
            "minutes",
            "seconds",
            "microseconds",
            "milliseconds",
        }
        # relativedelta/_offset path only valid for base DateOffset
        if self._use_relativedelta and set(kwds).issubset(relativedelta_fast):
            td_args = {
                "days",
                "hours",
                "minutes",
                "seconds",
                "microseconds",
                "milliseconds"
            }
            td_kwds = {
                key: val
                for key, val in kwds.items()
                if key in td_args
            }
            if "weeks" in kwds:
                days = td_kwds.get("days", 0)
                td_kwds["days"] = days + 7 * kwds["weeks"]

            if td_kwds:
                delta = Timedelta(**td_kwds)
                if "microseconds" in kwds:
                    delta = delta.as_unit("us")
                elif "milliseconds" in kwds:
                    delta = delta.as_unit("ms")
                else:
                    delta = delta.as_unit("s")
            else:
                delta = Timedelta(0).as_unit("s")

            return delta * self._n

        elif not self._use_relativedelta and hasattr(self, "_offset"):
            # timedelta
            num_nano = getattr(self, "nanoseconds", 0)
            if num_nano != 0:
                rem_nano = Timedelta(nanoseconds=num_nano)
                delta = Timedelta((self._offset + rem_nano) * self._n)
            else:
                delta = Timedelta(self._offset * self._n)
                if "microseconds" in kwds:
                    delta = delta.as_unit("us")
                elif "milliseconds" in kwds:
                    delta = delta.as_unit("ms")
                else:
                    delta = delta.as_unit("s")
            return delta

        else:
            # relativedelta with other keywords
            kwd = set(kwds) - relativedelta_fast
            raise NotImplementedError(
                "DateOffset with relativedelta "
                f"keyword(s) {kwd} not able to be "
                "applied vectorized"
            )

    def _apply_array(self, dtarr: np.ndarray) -> np.ndarray:
        # [FUNCTION DOC]
        # Vectorized apply on datetime64 ndarray: shifts by (years*12 + months)*n
        # months via shift_months (in the array's resolution), then adds
        # _pd_timedelta. Exceptions: NotImplementedError from _pd_timedelta.
        # Outputs: new datetime64 ndarray.
        reso = get_unit_from_dtype(dtarr.dtype)
        dt64other = np.asarray(dtarr)

        delta = self._pd_timedelta  # may raise NotImplementedError

        kwds = self.kwds
        months = (kwds.get("years", 0) * 12 + kwds.get("months", 0)) * self._n
        if months:
            shifted = shift_months(dt64other.view("i8"), months, reso=reso)
            dt64other = shifted.view(dtarr.dtype)
        return dt64other + delta

    def is_on_offset(self, dt: datetime) -> bool:
        # [FUNCTION DOC]
        # On-offset test: False only when normalize is set and dt is not midnight;
        # otherwise True.
        if self.normalize and not _is_normalized(dt):
            return False
        return True


class OffsetMeta(type):
    # [CLASS DOC]
    # Metaclass making isinstance/issubclass checks against DateOffset succeed for
    # every BaseOffset subclass (backward compatibility).
    """
    Metaclass that allows us to pretend that all BaseOffset subclasses
    inherit from DateOffset (which is needed for backward-compatibility).
    """

    @classmethod
    def __instancecheck__(cls, obj) -> bool:
        # [FUNCTION DOC]
        # Return isinstance(obj, BaseOffset). Outputs: bool.
        return isinstance(obj, BaseOffset)

    @classmethod
    def __subclasscheck__(cls, obj) -> bool:
        # [FUNCTION DOC]
        # Return issubclass(obj, BaseOffset). Outputs: bool.
        # Exceptions: TypeError if obj is not a class.
        return issubclass(obj, BaseOffset)


# TODO: figure out a way to use a metaclass with a cdef class
@set_module("pandas")
class DateOffset(RelativeDeltaOffset, metaclass=OffsetMeta):
    # [CLASS DOC]
    # Public relativedelta-style offset (pd.DateOffset). Same behavior as
    # RelativeDeltaOffset; uses OffsetMeta so all offsets are "DateOffset"
    # instances, and is immutable after construction.
    """
    Standard kind of date increment used for a date range.

    Works exactly like the keyword argument form of relativedelta.
    Note that the positional argument form of relativedelta is not
    supported. Use of the keyword n is discouraged-- you would be better
    off specifying n in the keywords you use, but regardless it is
    there for you. n is needed for DateOffset subclasses.

    DateOffset works as follows.  Each offset specify a set of dates
    that conform to the DateOffset.  For example, Bday defines this
    set to be the set of dates that are weekdays (M-F).  To test if a
    date is in the set of a DateOffset dateOffset we can use the
    is_on_offset method: dateOffset.is_on_offset(date).

    If a date is not on a valid date, the rollback and rollforward
    methods can be used to roll the date to the nearest valid date
    before/after the date.

    DateOffsets can be created to move dates forward a given number of
    valid dates.  For example, Bday(2) can be added to a date to move
    it two business days forward.  If the date does not start on a
    valid date, first it is moved to a valid date.  Thus pseudo code
    is::

        def __add__(date):
          date = rollback(date) # does nothing if date is valid
          return date + <n number of periods>

    When a date offset is created for a negative number of periods,
    the date is first rolled forward.  The pseudo code is::

        def __add__(date):
          date = rollforward(date) # does nothing if date is valid
          return date + <n number of periods>

    Zero presents a problem.  Should it roll forward or back?  We
    arbitrarily have it rollforward:

    date + BDay(0) == BDay.rollforward(date)

    Since 0 is a bit weird, we suggest avoiding its use.

    Besides, adding a DateOffsets specified by the singular form of the date
    component can be used to replace certain component of the timestamp.

    Attributes
    ----------
    n : int, default 1
        The number of time periods the offset represents.
        If specified without a temporal pattern, defaults to n days.
    normalize : bool, default False
        Whether to round the result of a DateOffset addition down to the
        previous midnight.
    weekday : int {0, 1, ..., 6}, default 0

        A specific integer for the day of the week.

        - 0 is Monday
        - 1 is Tuesday
        - 2 is Wednesday
        - 3 is Thursday
        - 4 is Friday
        - 5 is Saturday
        - 6 is Sunday

        Instead Weekday type from dateutil.relativedelta can be used.

        - MO is Monday
        - TU is Tuesday
        - WE is Wednesday
        - TH is Thursday
        - FR is Friday
        - SA is Saturday
        - SU is Sunday.

    **kwds
        Temporal parameter that add to or replace the offset value.

        Parameters that **add** to the offset (like Timedelta):

        - years
        - months
        - weeks
        - days
        - hours
        - minutes
        - seconds
        - milliseconds
        - microseconds
        - nanoseconds

        Parameters that **replace** the offset value:

        - year
        - month
        - day
        - weekday
        - hour
        - minute
        - second
        - microsecond
        - nanosecond.

    See Also
    --------
    dateutil.relativedelta.relativedelta : The relativedelta type is designed
        to be applied to an existing datetime and can replace specific components of
        that datetime, or represents an interval of time.

    Examples
    --------
    >>> from pandas.tseries.offsets import DateOffset
    >>> ts = pd.Timestamp('2017-01-01 09:10:11')
    >>> ts + DateOffset(months=3)
    Timestamp('2017-04-01 09:10:11')

    >>> ts = pd.Timestamp('2017-01-01 09:10:11')
    >>> ts + DateOffset(months=2)
    Timestamp('2017-03-01 09:10:11')
    >>> ts + DateOffset(day=31)
    Timestamp('2017-01-31 09:10:11')

    >>> ts + pd.DateOffset(hour=8)
    Timestamp('2017-01-01 08:10:11')
    """
    def __setattr__(self, name, value):
        # [FUNCTION DOC]
        # Block attribute assignment: always raises AttributeError (immutable).
        raise AttributeError("DateOffset objects are immutable.")

# --------------------------------------------------------------------


cdef class BusinessMixin(SingleConstructorOffset):
    # [CLASS DOC]
    # Shared base for business offsets. Holds _offset (timedelta added after the
    # business shift) and, for Custom* subclasses only, _weekmask, _holidays
    # (tuple of datetime64[D]) and _calendar (np.busdaycalendar); these are None
    # for non-custom offsets.
    """
    Mixin to business types to provide related functions.
    """

    cdef readonly:
        timedelta _offset
        # Only Custom subclasses use weekmask, holiday, calendar
        object _weekmask, _holidays, _calendar

    def __init__(self, n=1, normalize=False, offset=timedelta(0)):
        # [FUNCTION DOC]
        # Initialize n/normalize and the extra time offset.
        # Inputs: n - int; normalize - bool; offset - timedelta (default zero).
        # Exceptions: from _validate_n; TypeError if offset is not a timedelta.
        BaseOffset.__init__(self, n, normalize)
        self._offset = offset

    cpdef _init_custom(self, weekmask, holidays, calendar):
        # [FUNCTION DOC]
        # Additional constructor for Custom* subclasses: build calendar and holidays via
        # _get_calendar and store weekmask, holidays, calendar.
        # Inputs: weekmask - str/sequence; holidays - sequence or None; calendar -
        # busdaycalendar, holiday calendar, or None. Side effects: sets fields.
        """
        Additional __init__ for Custom subclasses.
        """
        calendar, holidays = _get_calendar(
            weekmask=weekmask, holidays=holidays, calendar=calendar
        )
        # Custom offset instances are identified by the
        # following two attributes. See DateOffset._params()
        # holidays, weekmask
        self._weekmask = weekmask
        self._holidays = holidays
        self._calendar = calendar

    @property
    def offset(self):
        # [FUNCTION DOC]
        # Property alias for _offset (backward compatibility). Outputs: timedelta.
        """
        Alias for self._offset.
        """
        # Alias for backward compat
        return self._offset

    @property
    def holidays(self):
        # [FUNCTION DOC]
        # Property returning _holidays (tuple for custom offsets, else None).
        """
        Return the holidays used for custom business day calculations.

        This property returns a tuple or list of holidays used when calculating
        business days for custom business day offsets. For non-custom business
        offsets (e.g., standard BusinessDay, BusinessHour), this will be None.

        Returns
        -------
        tuple, list, or None
            Holidays used in business day calculations, or None if no custom
            holidays are specified.

        See Also
        --------
        BusinessDay.holidays : Holidays for standard business day offset.
        BusinessHour.holidays : Holidays for standard business hour offset.
        CustomBusinessDay.holidays : Holidays for custom business day offset.
        CustomBusinessHour.holidays : Holidays for custom business hour offset.
        CustomBusinessMonthEnd.holidays : Holidays for custom business month end offset.
        CustomBusinessMonthBegin.holidays : Holidays for custom business month begin
            offset.
        CustomBusinessDay.weekmask : Weekmask for custom business day offset.
        CustomBusinessDay.calendar : Calendar for custom business day offset.

        Examples
        --------
        For standard business offsets, holidays is None:

        >>> bd = pd.offsets.BusinessDay()
        >>> bd.holidays is None
        True

        >>> bh = pd.offsets.BusinessHour()
        >>> bh.holidays is None
        True

        For custom business day with explicit holidays:

        >>> holidays = [pd.Timestamp("2023-12-25"), pd.Timestamp("2024-01-01")]
        >>> cbd = pd.offsets.CustomBusinessDay(holidays=holidays)
        >>> cbd.holidays  # doctest: +SKIP
        (Timestamp('2023-12-25 00:00:00'), Timestamp('2024-01-01 00:00:00'))

        For custom business hour with explicit holidays:

        >>> cbh = pd.offsets.CustomBusinessHour(holidays=holidays)
        >>> cbh.holidays  # doctest: +SKIP
        (Timestamp('2023-12-25 00:00:00'), Timestamp('2024-01-01 00:00:00'))

        For custom business month end with explicit holidays:

        >>> cbme = pd.offsets.CustomBusinessMonthEnd(holidays=holidays)
        >>> cbme.holidays  # doctest: +SKIP
        (Timestamp('2023-12-25 00:00:00'), Timestamp('2024-01-01 00:00:00'))

        For custom business month begin with explicit holidays:

        >>> cbmb = pd.offsets.CustomBusinessMonthBegin(holidays=holidays)
        >>> cbmb.holidays  # doctest: +SKIP
        (Timestamp('2023-12-25 00:00:00'), Timestamp('2024-01-01 00:00:00'))

        For custom business offsets with a calendar:

        >>> from pandas.tseries.holiday import USFederalHolidayCalendar
        >>> cal = USFederalHolidayCalendar()
        >>> cbd_cal = pd.offsets.CustomBusinessDay(calendar=cal)
        >>> isinstance(cbd_cal.holidays, tuple)
        True

        >>> cbh_cal = pd.offsets.CustomBusinessHour(calendar=cal)
        >>> isinstance(cbh_cal.holidays, tuple)
        True
        """
        return self._holidays

    @property
    def calendar(self):
        # [FUNCTION DOC]
        # Property returning _calendar (np.busdaycalendar or None).
        """
        Return the calendar used for business day calculations.

        This property returns the numpy busdaycalendar object used for
        determining valid business days. For standard business day offsets
        (e.g., BusinessDay, BusinessHour), this returns None. For custom
        business day offsets, this returns the calendar that was either
        passed directly or constructed from weekmask and holidays.

        Returns
        -------
        np.busdaycalendar or None
            The business day calendar used for calculations, or None if
            using default business day rules.

        See Also
        --------
        BusinessDay.holidays : Holidays for standard business day offset.
        CustomBusinessDay.holidays : Holidays for custom business day offset.
        CustomBusinessDay.weekmask : Weekmask for custom business day offset.

        Examples
        --------
        For standard business offsets, calendar is None:

        >>> bd = pd.offsets.BusinessDay()
        >>> bd.calendar is None
        True

        >>> bh = pd.offsets.BusinessHour()
        >>> bh.calendar is None
        True

        For custom business day with explicit holidays:

        >>> holidays = [pd.Timestamp("2023-12-25"), pd.Timestamp("2024-01-01")]
        >>> cbd = pd.offsets.CustomBusinessDay(holidays=holidays)
        >>> isinstance(cbd.calendar, np.busdaycalendar)
        True
        """
        return self._calendar

    @property
    def weekmask(self):
        # [FUNCTION DOC]
        # Property returning _weekmask (str/sequence or None).
        """
        Return the weekmask used for custom business day calculations.

        This property returns the weekmask string that defines which days of the
        week are considered business days. For non-custom business offsets (e.g.,
        standard BusinessDay, BusinessHour), this will be None. For custom business
        day offsets, this returns the weekmask that was specified or the default
        'Mon Tue Wed Thu Fri'.

        Returns
        -------
        str or None
            String representing valid business days (e.g., 'Mon Tue Wed Thu Fri'),
            or None if using default business day rules.

        See Also
        --------
        BusinessDay.holidays : Holidays for standard business day offset.
        CustomBusinessDay.holidays : Holidays for custom business day offset.
        CustomBusinessDay.calendar : Calendar for custom business day offset.

        Examples
        --------
        For standard business offsets, weekmask is None:

        >>> bd = pd.offsets.BusinessDay()
        >>> bd.weekmask is None
        True

        >>> bh = pd.offsets.BusinessHour()
        >>> bh.weekmask is None
        True

        For custom business day with default weekmask:

        >>> cbd = pd.offsets.CustomBusinessDay()
        >>> cbd.weekmask
        'Mon Tue Wed Thu Fri'

        For custom business day with custom weekmask:

        >>> cbd = pd.offsets.CustomBusinessDay(weekmask="Mon Wed Fri")
        >>> cbd.weekmask
        'Mon Wed Fri'
        """
        return self._weekmask

    def _repr_attrs(self) -> str:
        # [FUNCTION DOC]
        # Repr suffix ": offset=<repr>" when offset is non-zero, else empty string.
        if self.offset:
            attrs = [f"offset={repr(self.offset)}"]
        else:
            attrs = []
        out = ""
        if attrs:
            out += ": " + ", ".join(attrs)
        return out

    cpdef __setstate__(self, state):
        # [FUNCTION DOC]
        # Unpickle: set _offset from "_offset" or legacy "offset"; for custom classes
        # (prefix starting with C/c) pop weekmask and holidays, rebuild calendar via
        # _get_calendar; then BaseOffset.__setstate__ for n/normalize/_cache.
        # Exceptions: KeyError if custom state lacks weekmask/holidays.
        # Side effects: mutates state (pops keys).
        # We need to use a cdef/cpdef method to set the readonly _offset attribute
        if "_offset" in state:
            self._offset = state.pop("_offset")
        elif "offset" in state:
            # Older (<0.22.0) versions have offset attribute instead of _offset
            self._offset = state.pop("offset")

        if self._prefix.startswith(("C", "c")):
            # i.e. this is a Custom class
            weekmask = state.pop("weekmask")
            holidays = state.pop("holidays")
            calendar, holidays = _get_calendar(weekmask=weekmask,
                                               holidays=holidays,
                                               calendar=None)
            self._weekmask = weekmask
            self._calendar = calendar
            self._holidays = holidays

        BaseOffset.__setstate__(self, state)


cdef class BusinessDay(BusinessMixin):
    # [CLASS DOC]
    # Offset of n business days (Mon-Fri), plus optional time offset. Weekend
    # starting points are rolled to the adjacent business day as part of the shift.
    """
    DateOffset subclass representing possibly n business days.

    Parameters
    ----------
    n : int, default 1
        The number of days represented.
    normalize : bool, default False
        Normalize start/end dates to midnight.
    offset : timedelta, default timedelta(0)
        Time offset to apply.

    Examples
    --------
    You can use the parameter ``n`` to represent a shift of n business days.

    >>> ts = pd.Timestamp(2022, 12, 9, 15)
    >>> ts.strftime('%a %d %b %Y %H:%M')
    'Fri 09 Dec 2022 15:00'
    >>> (ts + pd.offsets.BusinessDay(n=5)).strftime('%a %d %b %Y %H:%M')
    'Fri 16 Dec 2022 15:00'

    Passing the parameter ``normalize`` equal to True, you shift the start
    of the next business day to midnight.

    >>> ts = pd.Timestamp(2022, 12, 9, 15)
    >>> ts + pd.offsets.BusinessDay(normalize=True)
    Timestamp('2022-12-12 00:00:00')
    """
    _period_dtype_code = PeriodDtypeCode.B
    _prefix = "B"
    _attributes = tuple(["n", "normalize", "offset"])

    cpdef __setstate__(self, state):
        # [FUNCTION DOC]
        # Unpickle: pop n, normalize, _offset (or legacy offset) and _cache from state.
        # Side effects: mutates state.
        self._n = state.pop("n")
        self._normalize = state.pop("normalize")
        if "_offset" in state:
            self._offset = state.pop("_offset")
        elif "offset" in state:
            self._offset = state.pop("offset")
        self._cache = state.pop("_cache", {})

    def _offset_str(self) -> str:
        # [FUNCTION DOC]
        # HELPER: freqstr suffix for the time offset: "+" or "-" followed by components
        # like "1D2h3Min4s5us" (zero components omitted); if offset is not a timedelta,
        # "+" plus its repr.
        def get_str(td):
            # [FUNCTION DOC]
            # HELPER: Format a non-negative timedelta td as concatenated "<days>D",
            # "<hours>h", "<minutes>Min", "<seconds>s", "<micro>us", skipping zero
            # parts.
            # Outputs: str (empty for zero).
            off_str = ""
            if td.days > 0:
                off_str += str(td.days) + "D"
            if td.seconds > 0:
                s = td.seconds
                hrs = int(s / 3600)
                if hrs != 0:
                    off_str += str(hrs) + "h"
                    s -= hrs * 3600
                mts = int(s / 60)
                if mts != 0:
                    off_str += str(mts) + "Min"
                    s -= mts * 60
                if s != 0:
                    off_str += str(s) + "s"
            if td.microseconds > 0:
                off_str += str(td.microseconds) + "us"
            return off_str

        if PyDelta_Check(self.offset):
            zero = timedelta(0, 0, 0)
            if self.offset >= zero:
                off_str = "+" + get_str(self.offset)
            else:
                off_str = "-" + get_str(-self.offset)
            return off_str
        else:
            return "+" + repr(self.offset)

    @apply_wraps
    def _apply(self, other):
        # [FUNCTION DOC]
        # Apply (wrapped by apply_wraps). For datetimes: shift by 7*(n//5) days plus
        # _adjust_ndays(weekday, n//5) days, then add offset. For timedelta scalars:
        # returns a new BusinessDay with offset increased by other.
        # Exceptions: ApplyTypeError for other types.
        if PyDateTime_Check(other):
            n = self._n
            wday = other.weekday()

            # avoid slowness below by operating on weeks first
            weeks = n // 5
            days = self._adjust_ndays(wday, weeks)

            result = other + timedelta(days=7 * weeks + days)
            if self.offset:
                result = result + self.offset
            return result

        elif is_any_td_scalar(other):
            td = Timedelta(self.offset) + other
            return BusinessDay(
                self._n, offset=td.to_pytimedelta(), normalize=self.normalize
            )
        else:
            raise ApplyTypeError(
                "Only know how to combine business day with datetime or timedelta."
            )

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef ndarray _shift_bdays(
        self,
        ndarray i8other,
        NPY_DATETIMEUNIT reso=NPY_DATETIMEUNIT.NPY_FR_ns,
    ):
        # [FUNCTION DOC]
        # HELPER: Vectorized business-day shift on int64 epoch values.
        # Inputs: i8other - int64 ndarray (any shape) of datetimes in unit reso; reso -
        # numpy datetime unit (default ns).
        # Outputs: new int64 ndarray of same shape; NaT entries stay NaT. Each value is
        # shifted by (7*weeks + days) * units-per-day, where weeks = n//5 and days from
        # _adjust_ndays using the value's weekday. Time-of-day is preserved; does not
        # add self.offset.
        # Side effects: allocates result; bounds/wraparound checks disabled.
        # KNOWN ISSUE: docstring refers to a nonexistent "BusinessDay.apply_offset".
        """
        Implementation of BusinessDay.apply_offset.

        Parameters
        ----------
        i8other : const int64_t[:]
        reso : NPY_DATETIMEUNIT, default NPY_FR_ns

        Returns
        -------
        ndarray[int64_t]
        """
        cdef:
            int periods = self._n
            Py_ssize_t i, n = i8other.size
            ndarray result = cnp.PyArray_EMPTY(
                i8other.ndim, i8other.shape, cnp.NPY_INT64, 0
            )
            int64_t val, res_val
            int wday, days
            npy_datetimestruct dts
            int64_t DAY_PERIODS = periods_per_day(reso)
            cnp.broadcast mi = cnp.PyArray_MultiIterNew2(result, i8other)

        for i in range(n):
            # Analogous to: val = i8other[i]
            val = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

            if val == NPY_NAT:
                res_val = NPY_NAT
            else:
                # The rest of this is effectively a copy of BusinessDay.apply
                weeks = periods // 5
                pandas_datetime_to_datetimestruct(val, reso, &dts)
                wday = dayofweek(dts.year, dts.month, dts.day)

                days = self._adjust_ndays(wday, weeks)
                res_val = val + (7 * weeks + days) * DAY_PERIODS

            # Analogous to: out[i] = res_val
            (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = res_val

            cnp.PyArray_MultiIter_NEXT(mi)

        return result

    cdef int _adjust_ndays(self, int wday, int weeks):
        # [FUNCTION DOC]
        # HELPER: Number of extra calendar days (beyond 7*weeks) needed to move n
        # business days from a date whose weekday is wday (0=Mon..6=Sun).
        # Inputs: wday - weekday; weeks - n // 5 (floor). Outputs: int days.
        # Logic: if n <= 0 and starting on weekend, n += 1 (counts as roll forward);
        # remaining r = n - 5*weeks (0..5). If r == 0 on weekend, go back to Friday
        # (4 - wday); on weekend with r > 0, go to Monday then r-1 more; if
        # wday + r stays within the week, r; else r + 2 to skip the weekend.
        cdef:
            int n = self._n
            int days

        if n <= 0 and wday > 4:
            # roll forward
            n += 1

        n -= 5 * weeks

        # n is always >= 0 at this point
        if n == 0 and wday > 4:
            # roll back
            days = 4 - wday
        elif wday > 4:
            # roll forward
            days = (7 - wday) + (n - 1)
        elif wday + n <= 4:
            # shift by n days without leaving the current week
            days = n
        else:
            # shift by n days plus 2 to get past the weekend
            days = n + 2
        return days

    def _apply_array(self, dtarr: np.ndarray) -> np.ndarray:
        # [FUNCTION DOC]
        # Vectorized apply: _shift_bdays on the int64 view in the array's unit; if an
        # offset is set, view as the input dtype and add Timedelta(offset).
        # Outputs: int64 ndarray (no offset; caller re-views) or datetime64 ndarray.
        i8other = dtarr.view("i8")
        reso = get_unit_from_dtype(dtarr.dtype)
        res = self._shift_bdays(i8other, reso=reso)
        if self.offset:
            res = res.view(dtarr.dtype) + Timedelta(self.offset)
        return res

    def is_on_offset(self, dt: datetime) -> bool:
        # [FUNCTION DOC]
        # True if dt falls Monday-Friday (and is midnight when normalize is set).
        """
        Return boolean whether a timestamp intersects with this frequency.

        This method determines if a given timestamp falls on a business day
        (Monday through Friday). If ``normalize`` is True, it also checks that
        the time component is midnight.

        Parameters
        ----------
        dt : datetime
            Timestamp to check intersection with frequency.

        Returns
        -------
        bool
            True if the timestamp is on a business day, False otherwise.

        See Also
        --------
        tseries.offsets.BusinessDay : Represents business day offset.
        tseries.offsets.CustomBusinessDay : Represents custom business day offset.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 8, 5)
        >>> ts.day_name()
        'Friday'
        >>> pd.offsets.BusinessDay().is_on_offset(ts)
        True

        >>> ts = pd.Timestamp(2022, 8, 6)
        >>> ts.day_name()
        'Saturday'
        >>> pd.offsets.BusinessDay().is_on_offset(ts)
        False

        With ``normalize=True``, the timestamp must also be at midnight:

        >>> ts = pd.Timestamp(2022, 8, 5, 12, 0)
        >>> pd.offsets.BusinessDay(normalize=True).is_on_offset(ts)
        False
        >>> ts = pd.Timestamp(2022, 8, 5, 0, 0)
        >>> pd.offsets.BusinessDay(normalize=True).is_on_offset(ts)
        True
        """
        if self.normalize and not _is_normalized(dt):
            return False
        return dt.weekday() < 5


cdef class BusinessHour(BusinessMixin):
    # [CLASS DOC]
    # Offset of n business hours within daily opening intervals defined by
    # parallel tuples _start and _end of datetime.time (sorted by start; intervals
    # may cross midnight but must not overlap or touch). Business days are
    # determined by next_bday (BusinessDay, or CustomBusinessDay for the custom
    # subclass, detected by a prefix starting with "c"). _adjust_dst is False.
    """
    DateOffset subclass representing possibly n business hours.

    Parameters
    ----------
    n : int, default 1
        The number of hours represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    start : str, time, or list of str/time, default "09:00"
        Start time of your custom business hour in 24h format.
    end : str, time, or list of str/time, default: "17:00"
        End time of your custom business hour in 24h format.
    offset : timedelta, default timedelta(0)
        Time offset to apply.

    Examples
    --------
    You can use the parameter ``n`` to represent a shift of n hours.

    >>> ts = pd.Timestamp(2022, 12, 9, 8)
    >>> ts + pd.offsets.BusinessHour(n=5)
    Timestamp('2022-12-09 14:00:00')

    You can also change the start and the end of business hours.

    >>> ts = pd.Timestamp(2022, 8, 5, 16)
    >>> ts + pd.offsets.BusinessHour(start="11:00")
    Timestamp('2022-08-08 11:00:00')

    >>> from datetime import time as dt_time
    >>> ts = pd.Timestamp(2022, 8, 5, 22)
    >>> ts + pd.offsets.BusinessHour(end=dt_time(19, 0))
    Timestamp('2022-08-08 10:00:00')

    Passing the parameter ``normalize`` equal to True, you shift the start
    of the next business hour to midnight.

    >>> ts = pd.Timestamp(2022, 12, 9, 8)
    >>> ts + pd.offsets.BusinessHour(normalize=True)
    Timestamp('2022-12-09 00:00:00')

    You can divide your business day hours into several parts.

    >>> import datetime as dt
    >>> freq = pd.offsets.BusinessHour(start=["06:00", "10:00", "15:00"],
    ...                                end=["08:00", "12:00", "17:00"])
    >>> pd.date_range(dt.datetime(2022, 12, 9), dt.datetime(2022, 12, 13), freq=freq)
    DatetimeIndex(['2022-12-09 06:00:00', '2022-12-09 07:00:00',
                   '2022-12-09 10:00:00', '2022-12-09 11:00:00',
                   '2022-12-09 15:00:00', '2022-12-09 16:00:00',
                   '2022-12-12 06:00:00', '2022-12-12 07:00:00',
                   '2022-12-12 10:00:00', '2022-12-12 11:00:00',
                   '2022-12-12 15:00:00', '2022-12-12 16:00:00'],
                   dtype='datetime64[ns]', freq='bh')
    """

    _prefix = "bh"
    _anchor = 0
    _attributes = tuple(["n", "normalize", "start", "end", "offset"])
    _adjust_dst = False

    cdef readonly:
        tuple _start, _end

    @property
    def start(self):
        # [FUNCTION DOC]
        # Property: tuple of opening times (datetime.time).
        """
        Return the start time(s) of the business hour.

        This property returns a tuple of `datetime.time` objects representing
        the start times of business hours.

        See Also
        --------
        BusinessHour.end : Return the end time(s) of the business hour.
        CustomBusinessHour.end : Return the end time(s) of the custom business hour.

        Examples
        --------
        >>> pd.offsets.BusinessHour().start
        (datetime.time(9, 0),)
        """
        return self._start

    @property
    def end(self):
        # [FUNCTION DOC]
        # Property: tuple of closing times (datetime.time).
        """
        Return the end time(s) of the business hour.

        This property returns a tuple of `datetime.time` objects representing
        the end times of business hours.

        See Also
        --------
        BusinessHour.start : Return the start time(s) of the business hour.
        CustomBusinessHour.start : Return the start time(s) of the custom business hour.

        Examples
        --------
        >>> pd.offsets.BusinessHour().end
        (datetime.time(17, 0),)
        """
        return self._end

    def __init__(
            self, n=1, normalize=False, start="09:00", end="17:00", offset=timedelta(0)
    ):
        # [FUNCTION DOC]
        # Initialize: scalar start/end wrapped into lists; each validated via
        # _validate_business_time; sorted by start; stored as tuples. Validates that
        # the sum of all open intervals and gaps equals exactly 24 hours.
        # Inputs: n, normalize, start/end - "HH:MM"/time or lists thereof, offset.
        # Exceptions: ValueError for empty start/end, unequal lengths, bad formats, or
        # overlapping/touching intervals.
        BusinessMixin.__init__(self, n, normalize, offset)

        # must be validated here to equality check
        if np.ndim(start) == 0:
            # i.e. not is_list_like
            start = [start]
        if not len(start):
            raise ValueError("Must include at least 1 start time")

        if np.ndim(end) == 0:
            # i.e. not is_list_like
            end = [end]
        if not len(end):
            raise ValueError("Must include at least 1 end time")

        start = np.array([_validate_business_time(x) for x in start])
        end = np.array([_validate_business_time(x) for x in end])

        # Validation of input
        if len(start) != len(end):
            raise ValueError("number of starting time and ending time must be the same")
        num_openings = len(start)

        # sort starting and ending time by starting time
        index = np.argsort(start)

        # convert to tuple so that start and end are hashable
        start = tuple(start[index])
        end = tuple(end[index])

        total_secs = 0
        for i in range(num_openings):
            total_secs += self._get_business_hours_by_sec(start[i], end[i])
            total_secs += self._get_business_hours_by_sec(
                end[i], start[(i + 1) % num_openings]
            )
        if total_secs != 24 * 60 * 60:
            raise ValueError(
                "invalid starting and ending time(s): "
                "opening hours should not touch or overlap with "
                "one another"
            )

        self._start = start
        self._end = end

    cpdef __setstate__(self, state):
        # [FUNCTION DOC]
        # Unpickle: restore start/end as tuples, discard legacy "kwds" and "next_bday"
        # keys, then BusinessMixin.__setstate__. Side effects: mutates state.
        start = state.pop("start")
        start = (start,) if np.ndim(start) == 0 else tuple(start)
        end = state.pop("end")
        end = (end,) if np.ndim(end) == 0 else tuple(end)
        self._start = start
        self._end = end

        state.pop("kwds", {})
        state.pop("next_bday", None)
        BusinessMixin.__setstate__(self, state)

    def _repr_attrs(self) -> str:
        # [FUNCTION DOC]
        # Repr suffix: parent suffix plus ": bh=HH:MM-HH:MM,..." (prefix used as key).
        # KNOWN ISSUE: when offset is non-zero this yields two ": " separated groups
        # (": offset=..: bh=..") rather than a comma-separated list.
        out = super()._repr_attrs()
        # Use python string formatting to be faster than strftime
        hours = ",".join(
            f"{st.hour:02d}:{st.minute:02d}-{en.hour:02d}:{en.minute:02d}"
            for st, en in zip(self._start, self._end, strict=True)
        )
        attrs = [f"{self._prefix}={hours}"]
        out += ": " + ", ".join(attrs)
        return out

    def _get_business_hours_by_sec(self, start, end):
        # [FUNCTION DOC]
        # HELPER: Seconds from start to end time of one interval, treating end <= start
        # as being on the next day. Inputs: start, end - datetime.time. Outputs: int.
        """
        Return business hours in a day by seconds.
        """
        # create dummy datetime to calculate business hours in a day
        dtstart = datetime(2014, 4, 1, start.hour, start.minute)
        day = 1 if start < end else 2
        until = datetime(2014, 4, day, end.hour, end.minute)
        return int((until - dtstart).total_seconds())

    def _get_closing_time(self, dt: datetime) -> datetime:
        # [FUNCTION DOC]
        # HELPER: Closing datetime for the interval opening at dt (matched by
        # hour/minute against _start): dt + interval length.
        # Exceptions: AssertionError if dt is not an opening time.
        """
        Get the closing time of a business hour interval by its opening time.

        Parameters
        ----------
        dt : datetime
            Opening time of a business hour interval.

        Returns
        -------
        result : datetime
            Corresponding closing time.
        """
        for i, st in enumerate(self._start):
            if st.hour == dt.hour and st.minute == dt.minute:
                return dt + timedelta(
                    seconds=self._get_business_hours_by_sec(st, self._end[i])
                )
        assert False

    @cache_readonly
    def next_bday(self):
        # [FUNCTION DOC]
        # Cached property: business-day offset used to step days; sign follows n
        # (+1 or -1). CustomBusinessDay with same weekmask/holidays/calendar for custom
        # hours, otherwise BusinessDay. Side effects: cached.
        """
        Used for moving to next business day.
        """
        if self._n >= 0:
            nb_offset = 1
        else:
            nb_offset = -1
        if self._prefix.startswith(("c")):
            # CustomBusinessHour
            return CustomBusinessDay(
                n=nb_offset,
                weekmask=self.weekmask,
                holidays=self.holidays,
                calendar=self.calendar,
            )
        else:
            return BusinessDay(n=nb_offset)

    def _next_opening_time(self, other, sign=1):
        # [FUNCTION DOC]
        # HELPER: Find an opening time relative to other. If n and sign agree (n == 0
        # treated as positive), return the earliest opening at or after other;
        # otherwise the latest opening at or before other. If other is not on a
        # business day, steps one business day in direction sign and takes the
        # earliest/latest start of that day. Outputs: naive datetime (minute
        # precision).
        """
        If self._n and sign have the same sign, return the earliest opening time
        later than or equal to current time.
        Otherwise the latest opening time earlier than or equal to current
        time.

        Opening time always locates on BusinessDay.
        However, closing time may not if business hour extends over midnight.

        Parameters
        ----------
        other : datetime
            Current time.
        sign : int, default 1.
            Either 1 or -1. Going forward in time if it has the same sign as
            self._n. Going backward in time otherwise.

        Returns
        -------
        result : datetime
            Next opening time.
        """
        earliest_start = self._start[0]
        latest_start = self._start[-1]

        if self._n == 0:
            is_same_sign = sign > 0
        else:
            is_same_sign = self._n * sign >= 0

        if not self.next_bday.is_on_offset(other):
            # today is not business day
            other = other + sign * self.next_bday
            if is_same_sign:
                hour, minute = earliest_start.hour, earliest_start.minute
            else:
                hour, minute = latest_start.hour, latest_start.minute
        else:
            if is_same_sign:
                if latest_start < other.time():
                    # current time is after latest starting time in today
                    other = other + sign * self.next_bday
                    hour, minute = earliest_start.hour, earliest_start.minute
                else:
                    # find earliest starting time no earlier than current time
                    for st in self._start:
                        if other.time() <= st:
                            hour, minute = st.hour, st.minute
                            break
            else:
                if other.time() < earliest_start:
                    # current time is before earliest starting time in today
                    other = other + sign * self.next_bday
                    hour, minute = latest_start.hour, latest_start.minute
                else:
                    # find latest starting time no later than current time
                    for st in reversed(self._start):
                        if other.time() >= st:
                            hour, minute = st.hour, st.minute
                            break

        return datetime(other.year, other.month, other.day, hour, minute)

    def _prev_opening_time(self, other: datetime) -> datetime:
        # [FUNCTION DOC]
        # HELPER: _next_opening_time with sign=-1 (the reverse direction from n).
        """
        If n is positive, return the latest opening time earlier than or equal
        to current time.
        Otherwise the earliest opening time later than or equal to current
        time.

        Parameters
        ----------
        other : datetime
            Current time.

        Returns
        -------
        result : datetime
            Previous opening time.
        """
        return self._next_opening_time(other, sign=-1)

    @apply_wraps
    def rollback(self, dt: datetime) -> datetime:
        # [FUNCTION DOC]
        # Roll back (wrapped by apply_wraps): if dt is not within business hours,
        # move to the closing time of the previous interval (n>=0) or of the next
        # (n<0) opening. Outputs: datetime.
        """
        Roll provided date backward to next offset only if not on offset.
        """
        if not self.is_on_offset(dt):
            if self._n >= 0:
                dt = self._prev_opening_time(dt)
            else:
                dt = self._next_opening_time(dt)
            return self._get_closing_time(dt)
        return dt

    @apply_wraps
    def rollforward(self, dt: datetime) -> datetime:
        # [FUNCTION DOC]
        # Roll forward (wrapped by apply_wraps): if dt is not within business hours,
        # move to next opening time (n>=0) or previous one (n<0).
        """
        Roll provided date forward to next offset only if not on offset.
        """
        if not self.is_on_offset(dt):
            if self._n >= 0:
                return self._next_opening_time(dt)
            else:
                return self._prev_opening_time(dt)
        return dt

    @apply_wraps
    def _apply(self, other: datetime) -> datetime:
        # [FUNCTION DOC]
        # Apply n business hours (wrapped by apply_wraps). Drops tz/nanoseconds
        # (remembering nanosecond for an edge case). Normalizes the start: for n>=0
        # move to the next opening if at a closing time or outside hours; for n<0 step
        # back one second if at an opening time and, if outside hours, move to the
        # preceding closing time. Splits |n| hours into whole business days (bd) and
        # remaining minutes (r) using total daily business minutes; shifts by bd
        # business days (preserving time since the interval opening when after
        # midnight off a business day), then walks interval by interval consuming the
        # remaining time forward or backward. Outputs: naive datetime (apply_wraps
        # re-localizes).
        # used for detecting edge condition
        nanosecond = getattr(other, "nanosecond", 0)
        # reset timezone and nanosecond
        # other may be a Timestamp, thus not use replace
        other = datetime(
            other.year,
            other.month,
            other.day,
            other.hour,
            other.minute,
            other.second,
            other.microsecond,
        )
        n = self._n

        # adjust other to reduce number of cases to handle
        if n >= 0:
            if other.time() in self._end or not self._is_on_offset(other):
                other = self._next_opening_time(other)
        else:
            if other.time() in self._start:
                # adjustment to move to previous business day
                other = other - timedelta(seconds=1)
            if not self._is_on_offset(other):
                other = self._next_opening_time(other)
                other = self._get_closing_time(other)

        # get total business hours by sec in one business day
        businesshours = sum(
            self._get_business_hours_by_sec(st, en)
            for st, en in zip(self._start, self._end, strict=True)
        )

        bd, r = divmod(abs(n * 60), businesshours // 60)
        if n < 0:
            bd, r = -bd, -r

        # adjust by business days first
        if bd != 0:
            if self._prefix.startswith("c"):
                # GH#30593 this is a Custom offset
                skip_bd = CustomBusinessDay(
                    n=bd,
                    weekmask=self.weekmask,
                    holidays=self.holidays,
                    calendar=self.calendar,
                )
            else:
                skip_bd = BusinessDay(n=bd)
            # midnight business hour may not on BusinessDay
            if not self.next_bday.is_on_offset(other):
                prev_open = self._prev_opening_time(other)
                remain = other - prev_open
                other = prev_open + skip_bd + remain
            else:
                other = other + skip_bd

        # remaining business hours to adjust
        bhour_remain = timedelta(minutes=r)

        if n >= 0:
            while bhour_remain != timedelta(0):
                # business hour left in this business time interval
                bhour = (
                    self._get_closing_time(self._prev_opening_time(other)) - other
                )
                if bhour_remain < bhour:
                    # finish adjusting if possible
                    other += bhour_remain
                    bhour_remain = timedelta(0)
                else:
                    # go to next business time interval
                    bhour_remain -= bhour
                    other = self._next_opening_time(other + bhour)
        else:
            while bhour_remain != timedelta(0):
                # business hour left in this business time interval
                bhour = self._next_opening_time(other) - other
                if (
                    bhour_remain > bhour
                    or bhour_remain == bhour
                    and nanosecond != 0
                ):
                    # finish adjusting if possible
                    other += bhour_remain
                    bhour_remain = timedelta(0)
                else:
                    # go to next business time interval
                    bhour_remain -= bhour
                    other = self._get_closing_time(
                        self._next_opening_time(
                            other + bhour - timedelta(seconds=1)
                        )
                    )

        return other

    def is_on_offset(self, dt: datetime) -> bool:
        # [FUNCTION DOC]
        # True if dt falls within an opening interval (after normalize check); tz is
        # dropped before the check. Inputs: dt - datetime. Outputs: bool.
        """
        Return boolean whether a timestamp intersects with this frequency.

        This method determines if a given timestamp falls within business hours.
        Business hours are defined by the ``start`` and ``end`` parameters
        (default 09:00 to 17:00). The timestamp must also fall on a business day
        (Monday through Friday). If ``normalize`` is True, it also checks that
        the time component is midnight.

        Parameters
        ----------
        dt : datetime
            Timestamp to check intersection with frequency.

        Returns
        -------
        bool
            True if the timestamp is within business hours, False otherwise.

        See Also
        --------
        tseries.offsets.BusinessHour : Represents business hour offset.
        tseries.offsets.BusinessDay.is_on_offset : Check if a timestamp is on
            a business day.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 8, 5, 10)
        >>> ts.day_name()
        'Friday'
        >>> pd.offsets.BusinessHour().is_on_offset(ts)
        True

        >>> ts = pd.Timestamp(2022, 8, 5, 20)
        >>> pd.offsets.BusinessHour().is_on_offset(ts)
        False

        >>> ts = pd.Timestamp(2022, 8, 6, 10)
        >>> ts.day_name()
        'Saturday'
        >>> pd.offsets.BusinessHour().is_on_offset(ts)
        False
        """
        if self.normalize and not _is_normalized(dt):
            return False

        if dt.tzinfo is not None:
            dt = datetime(
                dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond
            )
        # Valid bh can be on the different BusinessDay during midnight
        # Distinguish by the time spent from previous opening time
        return self._is_on_offset(dt)

    def _is_on_offset(self, dt: datetime) -> bool:
        # [FUNCTION DOC]
        # HELPER: Membership test without normalize/tz handling: find the relevant
        # opening time (previous for n>=0, next for n<0) and return whether the
        # elapsed seconds since it are <= that interval's length.
        """
        Slight speedups using calculated values.
        """
        # if self.normalize and not _is_normalized(dt):
        #     return False
        # Valid bh can be on the different BusinessDay during midnight
        # Distinguish by the time spent from previous opening time
        if self._n >= 0:
            op = self._prev_opening_time(dt)
        else:
            op = self._next_opening_time(dt)
        span = (dt - op).total_seconds()
        businesshours = 0
        for i, st in enumerate(self._start):
            if op.hour == st.hour and op.minute == st.minute:
                businesshours = self._get_business_hours_by_sec(st, self._end[i])
        if span <= businesshours:
            return True
        else:
            return False


cdef class WeekOfMonthMixin(SingleConstructorOffset):
    # [CLASS DOC]
    # Base for WeekOfMonth and LastWeekOfMonth: offsets anchored to a particular
    # weekday (0=Mon..6=Sun) in a particular week of each month (week -1 means the
    # last such weekday). Subclasses implement _get_offset_day.
    """
    Mixin for methods common to WeekOfMonth and LastWeekOfMonth.
    """

    cdef readonly:
        int weekday, week

    def __init__(self, n=1, normalize=False, weekday=0):
        # [FUNCTION DOC]
        # Initialize n/normalize and weekday.
        # Exceptions: ValueError if weekday not in 0..6; from _validate_n.
        BaseOffset.__init__(self, n, normalize)
        self.weekday = weekday

        if weekday < 0 or weekday > 6:
            raise ValueError(f"Day must be 0<=day<=6, got {weekday}")

    @apply_wraps
    def _apply(self, other: datetime) -> datetime:
        # [FUNCTION DOC]
        # Apply (wrapped by apply_wraps): compute the anchor day in other's month, use
        # roll_convention to adjust n months depending on whether other is before/after
        # it, shift to the first day of the target month, then move to that month's
        # anchor day. Outputs: datetime.
        compare_day = self._get_offset_day(other)

        months = self._n
        months = roll_convention(other.day, months, compare_day)

        shifted = shift_month(other, months, "start")
        to_day = self._get_offset_day(shifted)
        return _shift_day(shifted, to_day - shifted.day)

    def is_on_offset(self, dt: datetime) -> bool:
        # [FUNCTION DOC]
        # True if dt's day equals the anchor day of its month (and midnight when
        # normalize is set).
        if self.normalize and not _is_normalized(dt):
            return False
        return dt.day == self._get_offset_day(dt)

    @property
    def rule_code(self) -> str:
        # [FUNCTION DOC]
        # Property: "{prefix}-{week+1}{WEEKDAY}" (e.g. "WOM-1MON"), or
        # "{prefix}-{WEEKDAY}" when week == -1 (last week of month).
        """
        Return a string representing the base frequency.

        See Also
        --------
        tseries.offsets.Hour.rule_code :
            Returns a string representing the base frequency of 'h'.
        tseries.offsets.Day.rule_code :
            Returns a string representing the base frequency of 'D'.

        Examples
        --------
        >>> pd.offsets.Week(5).rule_code
        'W'

        >>> pd.offsets.WeekOfMonth(n=1, week=0, weekday=0).rule_code
        'WOM-1MON'
        """
        weekday = int_to_weekday.get(self.weekday, "")
        if self.week == -1:
            # LastWeekOfMonth
            return f"{self._prefix}-{weekday}"
        return f"{self._prefix}-{self.week + 1}{weekday}"


# ----------------------------------------------------------------------
# Year-Based Offset Classes

cdef class YearOffset(SingleConstructorOffset):
    # [CLASS DOC]
    # Base for year-anchored offsets. _month is the anchor month (1..12, default
    # from subclass _default_month) and _day_opt selects the day within it.
    # Arithmetic uses roll_qtrday/shift_month (scalar) and shift_quarters with
    # modby=12 (vectorized).
    """
    DateOffset that just needs a month.
    """
    _attributes = tuple(["n", "normalize", "month"])

    _default_month: ClassVar[int]

    cdef readonly:
        int _month

    def __init__(self, n=1, normalize=False, month=None):
        # [FUNCTION DOC]
        # Initialize n/normalize and month (defaults to _default_month).
        # Exceptions: ValueError if month not in 1..12.
        BaseOffset.__init__(self, n, normalize)

        month = month if month is not None else self._default_month
        self._month = month

        if month < 1 or month > 12:
            raise ValueError("Month must go from 1 to 12")

    cpdef __setstate__(self, state):
        # [FUNCTION DOC]
        # Unpickle: pop month, n, normalize from state; reset cache.
        # Side effects: mutates state.
        self._month = state.pop("month")
        self._n = state.pop("n")
        self._normalize = state.pop("normalize")
        self._cache = {}

    @property
    def month(self) -> int:
        # [FUNCTION DOC]
        # Property: the anchor month (1..12).
        """
        Return the month of the year on which this offset applies.

        Returns an integer representing the month (1-12) that this offset
        targets. For year-based offsets, this determines which month is used
        for calculations.

        See Also
        --------
        tseries.offsets.YearEnd : Offset to end of year.
        tseries.offsets.YearBegin : Offset to start of year.
        tseries.offsets.BYearEnd : Offset to last business day of year.
        tseries.offsets.BYearBegin : Offset to first business day of year.

        Examples
        --------
        >>> pd.offsets.BYearBegin().month
        1

        >>> pd.offsets.BYearBegin(month=6).month
        6
        """
        return self._month

    @classmethod
    def _from_name(cls, suffix=None):
        # [FUNCTION DOC]
        # Classmethod: construct from alias suffix; a month abbreviation (e.g. "JUN")
        # maps to month number. Exceptions: KeyError for unknown suffix.
        kwargs = {}
        if suffix:
            kwargs["month"] = MONTH_TO_CAL_NUM[suffix]
        return cls(**kwargs)

    @property
    def rule_code(self) -> str:
        # [FUNCTION DOC]
        # Property: "{prefix}-{MON}" e.g. "YE-DEC".
        """
        Return a string representing the base frequency.

        See Also
        --------
        tseries.offsets.Hour.rule_code :
            Returns a string representing the base frequency of 'h'.
        tseries.offsets.Day.rule_code :
            Returns a string representing the base frequency of 'D'.

        Examples
        --------
        >>> pd.tseries.offsets.YearBegin(n=1, month=2).rule_code
        'YS-FEB'

        >>> pd.tseries.offsets.YearEnd(n=1, month=6).rule_code
        'YE-JUN'
        """
        month = MONTH_ALIASES[self._month]
        return f"{self._prefix}-{month}"

    def is_on_offset(self, dt: datetime) -> bool:
        # [FUNCTION DOC]
        # True if dt is in the anchor month on the anchor day (per _day_opt), and
        # midnight when normalize is set.
        """
        Return boolean whether a timestamp intersects with this frequency.

        This method checks if a given datetime falls on a valid year
        boundary as defined by this offset.

        Parameters
        ----------
        dt : datetime
            Timestamp to check intersections with frequency.

        Returns
        -------
        bool
            True if the timestamp is on the offset, False otherwise.

        See Also
        --------
        YearEnd.is_on_offset : Check if a timestamp is at the end of a year.
        YearBegin.is_on_offset : Check if a timestamp is at the start of a year.
        BYearEnd.is_on_offset : Check if a timestamp is at the end of a
            business year.
        BYearBegin.is_on_offset : Check if a timestamp is at the start of a
            business year.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> freq = pd.offsets.YearBegin()
        >>> freq.is_on_offset(ts)
        True

        >>> freq = pd.offsets.BYearBegin()
        >>> freq.is_on_offset(ts)
        False

        >>> ts = pd.Timestamp(2022, 1, 3)
        >>> freq.is_on_offset(ts)
        True
        """
        if self.normalize and not _is_normalized(dt):
            return False
        return dt.month == self._month and dt.day == self._get_offset_day(dt)

    def _get_offset_day(self, other: datetime) -> int:
        # [FUNCTION DOC]
        # HELPER: anchor day for the anchor month of other's year (overrides the base
        # to use self._month instead of other's month). Outputs: int day.
        # override BaseOffset method to use self._month instead of other.month
        cdef:
            npy_datetimestruct dts
        pydate_to_dtstruct(other, &dts)
        dts.month = self._month
        return get_day_of_month(&dts, _str_to_day_opt(self._day_opt))

    @apply_wraps
    def _apply(self, other: datetime) -> datetime:
        # [FUNCTION DOC]
        # Apply (wrapped by apply_wraps): years = roll_qtrday(other, n, month,
        # day_opt, modby=12); shift other by years*12 + (month - other.month) months
        # landing on the day_opt day.
        years = roll_qtrday(other, self._n, self._month, self._day_opt, modby=12)
        months = years * 12 + (self._month - other.month)
        return shift_month(other, months, self._day_opt)

    def _apply_array(self, dtarr: np.ndarray) -> np.ndarray:
        # [FUNCTION DOC]
        # Vectorized apply via shift_quarters with modby=12 in the array's unit.
        # Outputs: int64 ndarray of epoch values (caller re-views as datetime64).
        reso = get_unit_from_dtype(dtarr.dtype)
        shifted = shift_quarters(
            dtarr.view("i8"), self._n, self._month, self._day_opt, modby=12, reso=reso
        )
        return shifted


cdef class BYearEnd(YearOffset):
    # [CLASS DOC]
    # Business year end: last weekday of the anchor month (default December).
    # Alias "BYE".
    # KNOWN ISSUE: sets _outputName, but __repr__ reads _output_name, so the
    # intended "BusinessYearEnd" repr name is never used.
    """
    DateOffset increments between the last business day of the year.

    Attributes
    ----------
    n : int, default 1
        The number of years represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    month : int, default 12
        A specific integer for the month of the year.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    >>> from pandas.tseries.offsets import BYearEnd
    >>> ts = pd.Timestamp('2020-05-24 05:01:15')
    >>> ts - BYearEnd()
    Timestamp('2019-12-31 05:01:15')
    >>> ts + BYearEnd()
    Timestamp('2020-12-31 05:01:15')
    >>> ts + BYearEnd(3)
    Timestamp('2022-12-30 05:01:15')
    >>> ts + BYearEnd(-3)
    Timestamp('2017-12-29 05:01:15')
    >>> ts + BYearEnd(month=11)
    Timestamp('2020-11-30 05:01:15')
    """

    _outputName = "BusinessYearEnd"
    _default_month = 12
    _prefix = "BYE"
    _day_opt = "business_end"


cdef class BYearBegin(YearOffset):
    # [CLASS DOC]
    # Business year begin: first weekday of the anchor month (default January).
    # Alias "BYS".
    # KNOWN ISSUE: sets _outputName, but __repr__ reads _output_name, so the
    # intended "BusinessYearBegin" repr name is never used.
    """
    DateOffset increments between the first business day of the year.

    Attributes
    ----------
    n : int, default 1
        The number of years represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    month : int, default 1
        A specific integer for the month of the year.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    >>> from pandas.tseries.offsets import BYearBegin
    >>> ts = pd.Timestamp('2020-05-24 05:01:15')
    >>> ts + BYearBegin()
    Timestamp('2021-01-01 05:01:15')
    >>> ts - BYearBegin()
    Timestamp('2020-01-01 05:01:15')
    >>> ts + BYearBegin(-1)
    Timestamp('2020-01-01 05:01:15')
    >>> ts + BYearBegin(2)
    Timestamp('2022-01-03 05:01:15')
    >>> ts + BYearBegin(month=11)
    Timestamp('2020-11-02 05:01:15')
    """

    _outputName = "BusinessYearBegin"
    _default_month = 1
    _prefix = "BYS"
    _day_opt = "business_start"

# The pair of classes `_YearEnd` and `YearEnd` exist because of
# https://github.com/cython/cython/issues/3873

cdef class _YearEnd(YearOffset):
    # [CLASS DOC]
    # Cython-level implementation of YearEnd (last calendar day of anchor month,
    # default December, alias "YE"); split from YearEnd due to a Cython bug with
    # docstrings. Computes _period_dtype_code at construction.
    _default_month = 12
    _prefix = "YE"
    _day_opt = "end"

    cdef readonly:
        int _period_dtype_code

    def __init__(self, n=1, normalize=False, month=None):
        # [FUNCTION DOC]
        # Initialize as YearOffset and set _period_dtype_code to annual code plus
        # month % 12. Exceptions: as YearOffset.__init__.
        # Because YearEnd can be the freq for a Period, define its
        #  _period_dtype_code at construction for performance
        YearOffset.__init__(self, n, normalize, month)
        self._period_dtype_code = PeriodDtypeCode.A + self._month % 12


class YearEnd(_YearEnd):
    # [CLASS DOC]
    # Public year-end offset; pure-Python subclass of _YearEnd (holds docstring).
    """
    DateOffset increments between calendar year end dates.

    YearEnd goes to the next date which is the end of the year.

    Parameters
    ----------
    n : int, default 1
        The number of years represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    month : int, default 12
        A specific integer for the month of the year.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    >>> ts = pd.Timestamp(2022, 1, 1)
    >>> ts + pd.offsets.YearEnd()
    Timestamp('2022-12-31 00:00:00')

    >>> ts = pd.Timestamp(2022, 12, 31)
    >>> ts + pd.offsets.YearEnd()
    Timestamp('2023-12-31 00:00:00')

    >>> ts = pd.Timestamp(2022, 1, 1)
    >>> ts + pd.offsets.YearEnd(month=2)
    Timestamp('2022-02-28 00:00:00')

    If you want to get the end of the current year:

    >>> ts = pd.Timestamp(2022, 12, 31)
    >>> pd.offsets.YearEnd().rollforward(ts)
    Timestamp('2022-12-31 00:00:00')
    """

    def __new__(cls, n=1, normalize=False, month=None):
        # [FUNCTION DOC]
        # Construct via _YearEnd.__new__ with the given arguments. Outputs: instance.
        return _YearEnd.__new__(cls, n, normalize, month)


cdef class YearBegin(YearOffset):
    # [CLASS DOC]
    # Year begin: first calendar day of the anchor month (default January).
    # Alias "YS".
    """
    DateOffset increments between calendar year begin dates.

    YearBegin goes to the next date which is the start of the year.

    Attributes
    ----------
    n : int, default 1
        The number of years represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    month : int, default 1
        A specific integer for the month of the year.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    >>> ts = pd.Timestamp(2022, 12, 1)
    >>> ts + pd.offsets.YearBegin()
    Timestamp('2023-01-01 00:00:00')

    >>> ts = pd.Timestamp(2023, 1, 1)
    >>> ts + pd.offsets.YearBegin()
    Timestamp('2024-01-01 00:00:00')

    >>> ts = pd.Timestamp(2022, 1, 1)
    >>> ts + pd.offsets.YearBegin(month=2)
    Timestamp('2022-02-01 00:00:00')

    If you want to get the start of the current year:

    >>> ts = pd.Timestamp(2023, 1, 1)
    >>> pd.offsets.YearBegin().rollback(ts)
    Timestamp('2023-01-01 00:00:00')
    """

    _default_month = 1
    _prefix = "YS"
    _day_opt = "start"


# ----------------------------------------------------------------------
# Quarter-Based Offset Classes

cdef class QuarterOffset(SingleConstructorOffset):
    # [CLASS DOC]
    # Base for quarter-anchored offsets. _startingMonth (any month in the
    # quarterly cycle; default from subclass) and _day_opt define the anchor dates
    # (every 3 months). _from_name_starting_month is used when parsing an alias
    # without a month suffix. Arithmetic uses modby=3.
    _attributes = tuple(["n", "normalize", "startingMonth"])
    # TODO: Consider combining QuarterOffset and YearOffset __init__ at some
    #       point.  Also apply_index, is_on_offset, rule_code if
    #       startingMonth vs month attr names are resolved

    _default_starting_month: ClassVar[int]
    _from_name_starting_month: ClassVar[int]

    cdef readonly:
        int _startingMonth

    def __init__(self, n=1, normalize=False, startingMonth=None):
        # [FUNCTION DOC]
        # Initialize n/normalize and startingMonth (default _default_starting_month).
        # Note: startingMonth is not range-validated here.
        BaseOffset.__init__(self, n, normalize)

        if startingMonth is None:
            startingMonth = self._default_starting_month
        self._startingMonth = startingMonth

    cpdef __setstate__(self, state):
        # [FUNCTION DOC]
        # Unpickle: pop startingMonth, n, normalize. Side effects: mutates state.
        self._startingMonth = state.pop("startingMonth")
        self._n = state.pop("n")
        self._normalize = state.pop("normalize")

    @property
    def startingMonth(self) -> int:
        # [FUNCTION DOC]
        # Property: the starting month (1..12) of the quarterly cycle.
        """
        Return the month of the year from which quarters start.

        This value determines which month marks the beginning of a quarterly period.
        For example, with startingMonth=1, quarters start in January, April, July,
        and October.

        See Also
        --------
        QuarterOffset.rule_code : Return the rule code for the quarter offset.
        HalfYearOffset.startingMonth : Similar property for half-year-based offsets.

        Examples
        --------
        >>> pd.offsets.BQuarterBegin().startingMonth
        3

        >>> pd.offsets.QuarterEnd().startingMonth
        3

        >>> pd.offsets.QuarterBegin(startingMonth=1).startingMonth
        1
        """
        return self._startingMonth

    @classmethod
    def _from_name(cls, suffix=None):
        # [FUNCTION DOC]
        # Classmethod: construct from alias suffix (month abbreviation ->
        # startingMonth); without suffix uses _from_name_starting_month if set.
        # Exceptions: KeyError for unknown suffix.
        kwargs = {}
        if suffix:
            kwargs["startingMonth"] = MONTH_TO_CAL_NUM[suffix]
        else:
            if cls._from_name_starting_month is not None:
                kwargs["startingMonth"] = cls._from_name_starting_month
        return cls(**kwargs)

    @property
    def rule_code(self) -> str:
        # [FUNCTION DOC]
        # Property: "{prefix}-{MON}" using startingMonth, e.g. "QE-DEC".
        """
        Return a string representing the frequency with month suffix.

        This property generates a rule code string that combines the offset's
        prefix with the abbreviated month name of the starting month.

        Returns
        -------
        str
            Rule code string with format 'PREFIX-MONTH', where PREFIX is the
            offset's frequency abbreviation and MONTH is the three-letter
            abbreviation of the starting month.

        See Also
        --------
        BQuarterBegin.rule_code : Rule code for business quarter begin offset.
        BQuarterEnd.rule_code : Rule code for business quarter end offset.
        QuarterBegin.rule_code : Rule code for quarter begin offset.
        QuarterEnd.rule_code : Rule code for quarter end offset.
        FY5253Quarter.get_rule_code_suffix : Suffix component of rule code for
            FY5253Quarter.
        FY5253.get_rule_code_suffix : Suffix component of rule code for FY5253.

        Examples
        --------
        Business quarter begin with different starting months:

        >>> bqb = pd.offsets.BQuarterBegin(startingMonth=1)
        >>> bqb.rule_code
        'BQS-JAN'

        >>> bqb = pd.offsets.BQuarterBegin(startingMonth=2)
        >>> bqb.rule_code
        'BQS-FEB'

        >>> bqb = pd.offsets.BQuarterBegin(startingMonth=3)
        >>> bqb.rule_code
        'BQS-MAR'

        Business quarter end with different starting months:

        >>> bqe = pd.offsets.BQuarterEnd(startingMonth=1)
        >>> bqe.rule_code
        'BQE-JAN'

        >>> bqe = pd.offsets.BQuarterEnd(startingMonth=12)
        >>> bqe.rule_code
        'BQE-DEC'

        Quarter begin with different starting months:

        >>> qb = pd.offsets.QuarterBegin(startingMonth=1)
        >>> qb.rule_code
        'QS-JAN'

        >>> qb = pd.offsets.QuarterBegin(startingMonth=3)
        >>> qb.rule_code
        'QS-MAR'

        Quarter end with different starting months:

        >>> qe = pd.offsets.QuarterEnd(startingMonth=1)
        >>> qe.rule_code
        'QE-JAN'

        >>> qe = pd.offsets.QuarterEnd(startingMonth=3)
        >>> qe.rule_code
        'QE-MAR'
        """
        month = MONTH_ALIASES[self.startingMonth]
        return f"{self._prefix}-{month}"

    def is_on_offset(self, dt: datetime) -> bool:
        # [FUNCTION DOC]
        # True if dt's month is congruent to startingMonth mod 3 and its day is the
        # anchor day (and midnight when normalize is set).
        """
        Return boolean whether a timestamp intersects with this frequency.

        This method checks if the given datetime falls on a quarter boundary
        as defined by the offset's ``startingMonth`` and day option.

        Parameters
        ----------
        dt : datetime
            Timestamp to check intersections with frequency.

        Returns
        -------
        bool
            True if the timestamp is on the offset, False otherwise.

        See Also
        --------
        QuarterOffset : Parent class with quarterly offset logic.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> freq = pd.offsets.BQuarterBegin()
        >>> freq.is_on_offset(ts)
        False
        >>> ts = pd.Timestamp(2022, 3, 1)
        >>> freq.is_on_offset(ts)
        True
        """
        if self.normalize and not _is_normalized(dt):
            return False
        mod_month = (dt.month - self.startingMonth) % 3
        return mod_month == 0 and dt.day == self._get_offset_day(dt)

    @apply_wraps
    def _apply(self, other: datetime) -> datetime:
        # [FUNCTION DOC]
        # Apply (wrapped by apply_wraps): months_since = other.month%3 -
        # startingMonth%3; qtrs = roll_qtrday(..., modby=3); shift by
        # qtrs*3 - months_since months to the day_opt day.
        # months_since: find the calendar quarter containing other.month,
        # e.g. if other.month == 8, the calendar quarter is [Jul, Aug, Sep].
        # Then find the month in that quarter containing an is_on_offset date for
        # self.  `months_since` is the number of months to shift other.month
        # to get to this on-offset month.
        months_since = other.month % 3 - self.startingMonth % 3
        qtrs = roll_qtrday(
            other, self._n, self.startingMonth, day_opt=self._day_opt, modby=3
        )
        months = qtrs * 3 - months_since
        return shift_month(other, months, self._day_opt)

    def _apply_array(self, dtarr: np.ndarray) -> np.ndarray:
        # [FUNCTION DOC]
        # Vectorized apply via shift_quarters with modby=3. Outputs: int64 ndarray.
        reso = get_unit_from_dtype(dtarr.dtype)
        shifted = shift_quarters(
            dtarr.view("i8"),
            self._n,
            self.startingMonth,
            self._day_opt,
            modby=3,
            reso=reso,
        )
        return shifted


cdef class BQuarterEnd(QuarterOffset):
    # [CLASS DOC]
    # Business quarter end (last weekday of quarter-end months). Alias "BQE";
    # default startingMonth 3, alias default 12.
    """
    DateOffset increments between the last business day of each Quarter.

    startingMonth = 1 corresponds to dates like 1/31/2007, 4/30/2007, ...
    startingMonth = 2 corresponds to dates like 2/28/2007, 5/31/2007, ...
    startingMonth = 3 corresponds to dates like 3/30/2007, 6/29/2007, ...

    Attributes
    ----------
    n : int, default 1
        The number of quarters represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    startingMonth : int, default 3
        A specific integer for the month of the year from which we start quarters.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    >>> from pandas.tseries.offsets import BQuarterEnd
    >>> ts = pd.Timestamp('2020-05-24 05:01:15')
    >>> ts + BQuarterEnd()
    Timestamp('2020-06-30 05:01:15')
    >>> ts + BQuarterEnd(2)
    Timestamp('2020-09-30 05:01:15')
    >>> ts + BQuarterEnd(1, startingMonth=2)
    Timestamp('2020-05-29 05:01:15')
    >>> ts + BQuarterEnd(startingMonth=2)
    Timestamp('2020-05-29 05:01:15')
    """
    _output_name = "BusinessQuarterEnd"
    _default_starting_month = 3
    _from_name_starting_month = 12
    _prefix = "BQE"
    _day_opt = "business_end"


cdef class BQuarterBegin(QuarterOffset):
    # [CLASS DOC]
    # Business quarter begin (first weekday of quarter-start months). Alias "BQS";
    # default startingMonth 3, alias default 1.
    """
    DateOffset increments between the first business day of each Quarter.

    startingMonth = 1 corresponds to dates like 1/01/2007, 4/01/2007, ...
    startingMonth = 2 corresponds to dates like 2/01/2007, 5/01/2007, ...
    startingMonth = 3 corresponds to dates like 3/01/2007, 6/01/2007, ...

    Attributes
    ----------
    n : int, default 1
        The number of quarters represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    startingMonth : int, default 3
        A specific integer for the month of the year from which we start quarters.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    >>> from pandas.tseries.offsets import BQuarterBegin
    >>> ts = pd.Timestamp('2020-05-24 05:01:15')
    >>> ts + BQuarterBegin()
    Timestamp('2020-06-01 05:01:15')
    >>> ts + BQuarterBegin(2)
    Timestamp('2020-09-01 05:01:15')
    >>> ts + BQuarterBegin(startingMonth=2)
    Timestamp('2020-08-03 05:01:15')
    >>> ts + BQuarterBegin(-1)
    Timestamp('2020-03-02 05:01:15')
    """
    _output_name = "BusinessQuarterBegin"
    _default_starting_month = 3
    _from_name_starting_month = 1
    _prefix = "BQS"
    _day_opt = "business_start"


cdef class QuarterEnd(QuarterOffset):
    # [CLASS DOC]
    # Quarter end (last calendar day). Alias "QE"; default startingMonth 3.
    # Computes _period_dtype_code at construction.
    """
    DateOffset increments between Quarter end dates.

    startingMonth = 1 corresponds to dates like 1/31/2007, 4/30/2007, ...
    startingMonth = 2 corresponds to dates like 2/28/2007, 5/31/2007, ...
    startingMonth = 3 corresponds to dates like 3/31/2007, 6/30/2007, ...

    Attributes
    ----------
    n : int, default 1
        The number of quarters represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    startingMonth : int, default 3
        A specific integer for the month of the year from which we start quarters.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    >>> ts = pd.Timestamp(2022, 1, 1)
    >>> ts + pd.offsets.QuarterEnd()
    Timestamp('2022-03-31 00:00:00')
    """
    _default_starting_month = 3
    _prefix = "QE"
    _day_opt = "end"

    cdef readonly:
        int _period_dtype_code

    def __init__(self, n=1, normalize=False, startingMonth=None):
        # [FUNCTION DOC]
        # Initialize as QuarterOffset and set _period_dtype_code = Q_DEC code +
        # startingMonth % 12.
        # Because QuarterEnd can be the freq for a Period, define its
        #  _period_dtype_code at construction for performance
        QuarterOffset.__init__(self, n, normalize, startingMonth)
        self._period_dtype_code = PeriodDtypeCode.Q_DEC + self.startingMonth % 12


cdef class QuarterBegin(QuarterOffset):
    # [CLASS DOC]
    # Quarter begin (first calendar day). Alias "QS"; default startingMonth 3,
    # alias default 1.
    """
    DateOffset increments between Quarter start dates.

    startingMonth = 1 corresponds to dates like 1/01/2007, 4/01/2007, ...
    startingMonth = 2 corresponds to dates like 2/01/2007, 5/01/2007, ...
    startingMonth = 3 corresponds to dates like 3/01/2007, 6/01/2007, ...

    Attributes
    ----------
    n : int, default 1
        The number of quarters represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    startingMonth : int, default 3
        A specific integer for the month of the year from which we start quarters.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    >>> ts = pd.Timestamp(2022, 1, 1)
    >>> ts + pd.offsets.QuarterBegin()
    Timestamp('2022-03-01 00:00:00')
    """
    _default_starting_month = 3
    _from_name_starting_month = 1
    _prefix = "QS"
    _day_opt = "start"


# ----------------------------------------------------------------------
# HalfYear-Based Offset Classes

cdef class HalfYearOffset(SingleConstructorOffset):
    # [CLASS DOC]
    # Base for half-year-anchored offsets; same structure as QuarterOffset but with
    # a 6-month cycle (modby=6).
    _attributes = tuple(["n", "normalize", "startingMonth"])
    # TODO: Consider combining HalfYearOffset, QuarterOffset and YearOffset

    _default_starting_month: ClassVar[int]
    _from_name_starting_month: ClassVar[int]

    cdef readonly:
        int _startingMonth

    def __init__(self, n=1, normalize=False, startingMonth=None):
        # [FUNCTION DOC]
        # Initialize n/normalize and startingMonth (default _default_starting_month).
        BaseOffset.__init__(self, n, normalize)

        if startingMonth is None:
            startingMonth = self._default_starting_month
        self._startingMonth = startingMonth

    cpdef __setstate__(self, state):
        # [FUNCTION DOC]
        # Unpickle: pop startingMonth, n, normalize. Side effects: mutates state.
        self._startingMonth = state.pop("startingMonth")
        self._n = state.pop("n")
        self._normalize = state.pop("normalize")

    @property
    def startingMonth(self) -> int:
        # [FUNCTION DOC]
        # Property: starting month of the half-year cycle.
        """
        Return the month of the year from which half-years start.

        This value determines which month marks the beginning of a half-year period.
        For example, with startingMonth=1, half-years start in January and July.

        See Also
        --------
        HalfYearOffset.rule_code : Return the rule code for the half-year offset.
        QuarterOffset.startingMonth : Similar property for quarter-based offsets.

        Examples
        --------
        >>> pd.offsets.BHalfYearBegin().startingMonth
        1

        >>> pd.offsets.BHalfYearEnd().startingMonth
        6

        >>> pd.offsets.HalfYearBegin(startingMonth=3).startingMonth
        3
        """
        return self._startingMonth

    @classmethod
    def _from_name(cls, suffix=None):
        # [FUNCTION DOC]
        # Classmethod: construct from alias suffix (month abbreviation) or
        # _from_name_starting_month. Exceptions: KeyError for unknown suffix.
        kwargs = {}
        if suffix:
            kwargs["startingMonth"] = MONTH_TO_CAL_NUM[suffix]
        else:
            if cls._from_name_starting_month is not None:
                kwargs["startingMonth"] = cls._from_name_starting_month
        return cls(**kwargs)

    @property
    def rule_code(self) -> str:
        # [FUNCTION DOC]
        # Property: "{prefix}-{MON}" using startingMonth.
        """
        Return a string representing the frequency with month suffix.

        This property generates a rule code string that combines the offset's
        prefix with the abbreviated month name of the starting month.

        See Also
        --------
        FY5253Quarter.get_rule_code_suffix : Return the suffix component of the rule
            code for FY5253Quarter.
        FY5253.get_rule_code_suffix : Return the suffix component of the rule code for
            FY5253.

        Examples
        --------
        >>> offset = pd.offsets.BHalfYearEnd(startingMonth=6)
        >>> offset.rule_code
        'BHYE-JUN'
        """
        month = MONTH_ALIASES[self.startingMonth]
        return f"{self._prefix}-{month}"

    def is_on_offset(self, dt: datetime) -> bool:
        # [FUNCTION DOC]
        # True if dt's month is congruent to startingMonth mod 6 and its day is the
        # anchor day (and midnight when normalize is set).
        """
        Return boolean whether a timestamp intersects with this frequency.

        This method checks if a given datetime falls on a valid half-year
        boundary as defined by this offset.

        Parameters
        ----------
        dt : datetime
            Timestamp to check intersections with frequency.

        Returns
        -------
        bool
            True if the timestamp is on the offset, False otherwise.

        See Also
        --------
        HalfYearEnd.is_on_offset : Check if a timestamp is at the end of a
            half-year.
        HalfYearBegin.is_on_offset : Check if a timestamp is at the start of a
            half-year.
        BHalfYearEnd.is_on_offset : Check if a timestamp is at the end of a
            business half-year.
        BHalfYearBegin.is_on_offset : Check if a timestamp is at the start of a
            business half-year.

        Examples
        --------
        >>> freq = pd.offsets.BHalfYearBegin()
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> freq.is_on_offset(ts)
        False

        >>> ts = pd.Timestamp(2022, 1, 3)
        >>> freq.is_on_offset(ts)
        True
        """
        if self.normalize and not _is_normalized(dt):
            return False
        mod_month = (dt.month - self.startingMonth) % 6
        return mod_month == 0 and dt.day == self._get_offset_day(dt)

    @apply_wraps
    def _apply(self, other: datetime) -> datetime:
        # [FUNCTION DOC]
        # Apply (wrapped by apply_wraps): like QuarterOffset._apply with 6-month
        # periods: shift by hlvs*6 - (other.month%6 - startingMonth%6) months.
        # months_since: find the calendar half containing other.month,
        # e.g. if other.month == 8, the calendar half is [Jul, Aug, Sep, ..., Dec].
        # Then find the month in that half containing an is_on_offset date for
        # self.  `months_since` is the number of months to shift other.month
        # to get to this on-offset month.
        months_since = other.month % 6 - self.startingMonth % 6
        hlvs = roll_qtrday(
            other, self._n, self.startingMonth, day_opt=self._day_opt, modby=6
        )
        months = hlvs * 6 - months_since
        return shift_month(other, months, self._day_opt)

    def _apply_array(self, dtarr: np.ndarray) -> np.ndarray:
        # [FUNCTION DOC]
        # Vectorized apply via shift_quarters with modby=6. Outputs: int64 ndarray.
        reso = get_unit_from_dtype(dtarr.dtype)
        shifted = shift_quarters(
            dtarr.view("i8"),
            self._n,
            self.startingMonth,
            self._day_opt,
            modby=6,
            reso=reso,
        )
        return shifted


cdef class BHalfYearEnd(HalfYearOffset):
    # [CLASS DOC]
    # Business half-year end. Alias "BHYE"; default startingMonth 6, alias
    # default 12.
    """
    DateOffset increments between the last business day of each half-year.

    startingMonth = 1 corresponds to dates like 1/31/2007, 7/31/2007, ...
    startingMonth = 2 corresponds to dates like 2/28/2007, 8/31/2007, ...
    startingMonth = 6 corresponds to dates like 6/30/2007, 12/31/2007, ...

    Attributes
    ----------
    n : int, default 1
        The number of half-years represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    startingMonth : int, default 6
        A specific integer for the month of the year from which we start half-years.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    >>> from pandas.tseries.offsets import BHalfYearEnd
    >>> ts = pd.Timestamp('2020-05-24 05:01:15')
    >>> ts + BHalfYearEnd()
    Timestamp('2020-06-30 05:01:15')
    >>> ts + BHalfYearEnd(2)
    Timestamp('2020-12-31 05:01:15')
    >>> ts + BHalfYearEnd(1, startingMonth=2)
    Timestamp('2020-08-31 05:01:15')
    >>> ts + BHalfYearEnd(startingMonth=2)
    Timestamp('2020-08-31 05:01:15')
    """
    _output_name = "BusinessHalfYearEnd"
    _default_starting_month = 6
    _from_name_starting_month = 12
    _prefix = "BHYE"
    _day_opt = "business_end"


cdef class BHalfYearBegin(HalfYearOffset):
    # [CLASS DOC]
    # Business half-year begin. Alias "BHYS"; default startingMonth 1.
    """
    DateOffset increments between the first business day of each half-year.

    startingMonth = 1 corresponds to dates like 1/01/2007, 7/01/2007, ...
    startingMonth = 2 corresponds to dates like 2/01/2007, 8/01/2007, ...
    startingMonth = 3 corresponds to dates like 3/01/2007, 9/01/2007, ...

    Attributes
    ----------
    n : int, default 1
        The number of half-years represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    startingMonth : int, default 1
        A specific integer for the month of the year from which we start half-years.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    >>> from pandas.tseries.offsets import BHalfYearBegin
    >>> ts = pd.Timestamp('2020-05-24 05:01:15')
    >>> ts + BHalfYearBegin()
    Timestamp('2020-07-01 05:01:15')
    >>> ts + BHalfYearBegin(2)
    Timestamp('2021-01-01 05:01:15')
    >>> ts + BHalfYearBegin(startingMonth=2)
    Timestamp('2020-08-03 05:01:15')
    >>> ts + BHalfYearBegin(-1)
    Timestamp('2020-01-01 05:01:15')
    """
    _output_name = "BusinessHalfYearBegin"
    _default_starting_month = 1
    _from_name_starting_month = 1
    _prefix = "BHYS"
    _day_opt = "business_start"


cdef class HalfYearEnd(HalfYearOffset):
    # [CLASS DOC]
    # Half-year end (last calendar day). Alias "HYE"; default startingMonth 6,
    # alias default 12.
    """
    DateOffset increments between half-year end dates.

    startingMonth = 1 corresponds to dates like 1/31/2007, 7/31/2007, ...
    startingMonth = 2 corresponds to dates like 2/28/2007, 8/31/2007, ...
    startingMonth = 6 corresponds to dates like 6/30/2007, 12/31/2007, ...

    Attributes
    ----------
    n : int, default 1
        The number of half-years represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    startingMonth : int, default 6
        A specific integer for the month of the year from which we start half-years.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    >>> ts = pd.Timestamp(2022, 1, 1)
    >>> ts + pd.offsets.HalfYearEnd()
    Timestamp('2022-06-30 00:00:00')
    """
    _default_starting_month = 6
    _from_name_starting_month = 12
    _prefix = "HYE"
    _day_opt = "end"


cdef class HalfYearBegin(HalfYearOffset):
    # [CLASS DOC]
    # Half-year begin (first calendar day). Alias "HYS"; default startingMonth 1.
    """
    DateOffset increments between half-year start dates.

    startingMonth = 1 corresponds to dates like 1/01/2007, 7/01/2007, ...
    startingMonth = 2 corresponds to dates like 2/01/2007, 8/01/2007, ...
    startingMonth = 3 corresponds to dates like 3/01/2007, 9/01/2007, ...

    Attributes
    ----------
    n : int, default 1
        The number of half-years represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    startingMonth : int, default 1
        A specific integer for the month of the year from which we start half-years.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    >>> ts = pd.Timestamp(2022, 2, 1)
    >>> ts + pd.offsets.HalfYearBegin()
    Timestamp('2022-07-01 00:00:00')
    """
    _default_starting_month = 1
    _from_name_starting_month = 1
    _prefix = "HYS"
    _day_opt = "start"


# ----------------------------------------------------------------------
# Month-Based Offset Classes

cdef class MonthOffset(SingleConstructorOffset):
    # [CLASS DOC]
    # Base for month-anchored offsets; the anchor day of every month is chosen by
    # _day_opt.
    def is_on_offset(self, dt: datetime) -> bool:
        # [FUNCTION DOC]
        # True if dt's day equals the anchor day of its month (and midnight when
        # normalize is set).
        """
        Return boolean whether a timestamp intersects with this frequency.

        This method determines if a given timestamp aligns with the specific
        day of the month defined by this offset. For month-based offsets,
        it checks whether the timestamp's day matches the expected day
        according to the offset's configuration (e.g., start of month,
        end of month, or first/last business day).

        Parameters
        ----------
        dt : datetime
            Timestamp to check intersection with frequency.

        Returns
        -------
        bool
            True if the timestamp is on the offset, False otherwise.

        See Also
        --------
        tseries.offsets.MonthBegin : DateOffset of one month at beginning.
        tseries.offsets.MonthEnd : DateOffset of one month end.
        tseries.offsets.BusinessMonthBegin : DateOffset of one month at the first
            business day.
        tseries.offsets.BusinessMonthEnd : DateOffset of one month at the last
            business day.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> pd.offsets.MonthBegin().is_on_offset(ts)
        True

        >>> ts = pd.Timestamp(2022, 1, 15)
        >>> pd.offsets.MonthBegin().is_on_offset(ts)
        False

        >>> ts = pd.Timestamp(2022, 1, 31)
        >>> pd.offsets.MonthEnd().is_on_offset(ts)
        True
        """
        if self.normalize and not _is_normalized(dt):
            return False
        return dt.day == self._get_offset_day(dt)

    @apply_wraps
    def _apply(self, other: datetime) -> datetime:
        # [FUNCTION DOC]
        # Apply (wrapped by apply_wraps): adjust n with roll_convention relative to the
        # anchor day, then shift_month by that many months to the day_opt day.
        compare_day = self._get_offset_day(other)
        n = roll_convention(other.day, self._n, compare_day)
        return shift_month(other, n, self._day_opt)

    def _apply_array(self, dtarr: np.ndarray) -> np.ndarray:
        # [FUNCTION DOC]
        # Vectorized apply via shift_months with day_opt. Outputs: int64 ndarray.
        reso = get_unit_from_dtype(dtarr.dtype)
        shifted = shift_months(dtarr.view("i8"), self._n, self._day_opt, reso=reso)
        return shifted

    cpdef __setstate__(self, state):
        # [FUNCTION DOC]
        # Unpickle: discard legacy keys (_use_relativedelta, offset, _offset, kwds) then
        # BaseOffset.__setstate__. Side effects: mutates state.
        state.pop("_use_relativedelta", False)
        state.pop("offset", None)
        state.pop("_offset", None)
        state.pop("kwds", {})

        BaseOffset.__setstate__(self, state)


cdef class MonthEnd(MonthOffset):
    # [CLASS DOC]
    # Month end: last calendar day of each month. Alias "ME".
    """
    DateOffset of one month end.

    MonthEnd goes to the next date which is an end of the month.

    Attributes
    ----------
    n : int, default 1
        The number of months represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    >>> ts = pd.Timestamp(2022, 1, 30)
    >>> ts + pd.offsets.MonthEnd()
    Timestamp('2022-01-31 00:00:00')

    >>> ts = pd.Timestamp(2022, 1, 31)
    >>> ts + pd.offsets.MonthEnd()
    Timestamp('2022-02-28 00:00:00')

    If you want to get the end of the current month:

    >>> ts = pd.Timestamp(2022, 1, 31)
    >>> pd.offsets.MonthEnd().rollforward(ts)
    Timestamp('2022-01-31 00:00:00')
    """
    _period_dtype_code = PeriodDtypeCode.M
    _prefix = "ME"
    _day_opt = "end"


cdef class MonthBegin(MonthOffset):
    # [CLASS DOC]
    # Month begin: first calendar day of each month. Alias "MS".
    """
    DateOffset of one month at beginning.

    MonthBegin goes to the next date which is a start of the month.

    Attributes
    ----------
    n : int, default 1
        The number of months represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    >>> ts = pd.Timestamp(2022, 11, 30)
    >>> ts + pd.offsets.MonthBegin()
    Timestamp('2022-12-01 00:00:00')

    >>> ts = pd.Timestamp(2022, 12, 1)
    >>> ts + pd.offsets.MonthBegin()
    Timestamp('2023-01-01 00:00:00')

    If you want to get the start of the current month:

    >>> ts = pd.Timestamp(2022, 12, 1)
    >>> pd.offsets.MonthBegin().rollback(ts)
    Timestamp('2022-12-01 00:00:00')
    """
    _prefix = "MS"
    _day_opt = "start"


cdef class BusinessMonthEnd(MonthOffset):
    # [CLASS DOC]
    # Business month end: last weekday (Mon-Fri) of each month. Alias "BME".
    """
    DateOffset increments between the last business day of the month.

    BusinessMonthEnd goes to the next date which is the last business day of the month.

    Attributes
    ----------
    n : int, default 1
        The number of months represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    >>> ts = pd.Timestamp(2022, 11, 29)
    >>> ts + pd.offsets.BMonthEnd()
    Timestamp('2022-11-30 00:00:00')

    >>> ts = pd.Timestamp(2022, 11, 30)
    >>> ts + pd.offsets.BMonthEnd()
    Timestamp('2022-12-30 00:00:00')

    If you want to get the end of the current business month:

    >>> ts = pd.Timestamp(2022, 11, 30)
    >>> pd.offsets.BMonthEnd().rollforward(ts)
    Timestamp('2022-11-30 00:00:00')
    """
    _prefix = "BME"
    _day_opt = "business_end"


cdef class BusinessMonthBegin(MonthOffset):
    # [CLASS DOC]
    # Business month begin: first weekday of each month. Alias "BMS".
    """
    DateOffset of one month at the first business day.

    BusinessMonthBegin goes to the next date which is the first business day
    of the month.

    Attributes
    ----------
    n : int, default 1
        The number of months represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    >>> ts = pd.Timestamp(2022, 11, 30)
    >>> ts + pd.offsets.BMonthBegin()
    Timestamp('2022-12-01 00:00:00')

    >>> ts = pd.Timestamp(2022, 12, 1)
    >>> ts + pd.offsets.BMonthBegin()
    Timestamp('2023-01-02 00:00:00')

    If you want to get the start of the current business month:

    >>> ts = pd.Timestamp(2022, 12, 1)
    >>> pd.offsets.BMonthBegin().rollback(ts)
    Timestamp('2022-12-01 00:00:00')
    """
    _prefix = "BMS"
    _day_opt = "business_start"


# ---------------------------------------------------------------------
# Semi-Month Based Offsets

cdef class SemiMonthOffset(SingleConstructorOffset):
    # [CLASS DOC]
    # Base for twice-monthly offsets. Anchors are day_of_month (default 15) and
    # either the 1st (SemiMonthBegin) or the month end (SemiMonthEnd). Each step of
    # n moves to the next anchor (two per month). day_of_month must lie in
    # [_min_day_of_month, 27].
    _default_day_of_month = 15
    _min_day_of_month = 2
    _attributes = tuple(["n", "normalize", "day_of_month"])

    cdef readonly:
        int _day_of_month

    @property
    def day_of_month(self):
        # [FUNCTION DOC]
        # Property: the mid-month anchor day.
        """
        Return the day of the month for the semi-monthly offset.

        This specifies the day of the month on which the offset falls,
        in addition to the start (1st) or end (last day) of the month.

        See Also
        --------
        SemiMonthBegin : Offset at the start of the month and mid-month.
        SemiMonthEnd : Offset at mid-month and end of month.

        Examples
        --------
        >>> pd.offsets.SemiMonthBegin().day_of_month
        15

        >>> pd.offsets.SemiMonthBegin(day_of_month=20).day_of_month
        20
        """
        return self._day_of_month

    def __init__(self, n=1, normalize=False, day_of_month=None):
        # [FUNCTION DOC]
        # Initialize n/normalize and day_of_month (default 15, cast to int).
        # Exceptions: ValueError if out of [_min_day_of_month, 27]; TypeError/ValueError
        # from int() for non-numeric input.
        BaseOffset.__init__(self, n, normalize)

        if day_of_month is None:
            day_of_month = self._default_day_of_month

        self._day_of_month = int(day_of_month)
        if not self._min_day_of_month <= self._day_of_month <= 27:
            raise ValueError(
                "day_of_month must be "
                f"{self._min_day_of_month}<=day_of_month<=27, "
                f"got {self._day_of_month}"
            )

    cpdef __setstate__(self, state):
        # [FUNCTION DOC]
        # Unpickle: pop n, normalize, day_of_month. Side effects: mutates state.
        self._n = state.pop("n")
        self._normalize = state.pop("normalize")
        self._day_of_month = state.pop("day_of_month")

    @classmethod
    def _from_name(cls, suffix=None):
        # [FUNCTION DOC]
        # Classmethod: construct with day_of_month taken from the alias suffix (string
        # digits) or default when suffix is None.
        return cls(day_of_month=suffix)

    @property
    def rule_code(self) -> str:
        # [FUNCTION DOC]
        # Property: "{prefix}-{day_of_month}", e.g. "SME-15".
        suffix = f"-{self._day_of_month}"
        return self._prefix + suffix

    @apply_wraps
    def _apply(self, other: datetime) -> datetime:
        # [FUNCTION DOC]
        # Apply (wrapped by apply_wraps): adjust n with roll_convention relative to
        # day_of_month, with an extra decrement for SemiMonthBegin starting on day 1
        # with n <= 0 or increment for SemiMonthEnd starting on the last day with
        # n > 0. Then half-steps map to months/day: begin -> months = n//2 + n%2,
        # day = 1 if odd else day_of_month; end -> months = n//2, day = 31 (clipped to
        # month length) if odd else day_of_month. Uses shift_month with a numeric day.
        is_start = isinstance(self, SemiMonthBegin)

        # shift `other` to self._day_of_month, incrementing `n` if necessary
        n = roll_convention(other.day, self._n, self._day_of_month)

        days_in_month = get_days_in_month(other.year, other.month)
        # For SemiMonthBegin on other.day == 1 and
        # SemiMonthEnd on other.day == days_in_month,
        # shifting `other` to `self._day_of_month` _always_ requires
        # incrementing/decrementing `n`, regardless of whether it is
        # initially positive.
        if is_start and (self._n <= 0 and other.day == 1):
            n -= 1
        elif (not is_start) and (self._n > 0 and other.day == days_in_month):
            n += 1

        if is_start:
            months = n // 2 + n % 2
            to_day = 1 if n % 2 else self._day_of_month
        else:
            months = n // 2
            to_day = 31 if n % 2 else self._day_of_month

        return shift_month(other, months, to_day)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def _apply_array(self, dtarr: np.ndarray) -> np.ndarray:
        # [FUNCTION DOC]
        # Vectorized version of _apply over a datetime64 ndarray (any shape) in its
        # unit: same per-element logic on broken-down dates; NaT preserved; time of
        # day preserved. Outputs: new int64 ndarray. Side effects: releases the GIL
        # for the main loop; allocates output; bounds/wraparound checks disabled.
        cdef:
            ndarray i8other = dtarr.view("i8")
            Py_ssize_t i, count = dtarr.size
            int64_t val, res_val
            ndarray out = cnp.PyArray_EMPTY(
                i8other.ndim, i8other.shape, cnp.NPY_INT64, 0
            )
            npy_datetimestruct dts
            int months, to_day, nadj, n = self._n
            int days_in_month, day, anchor_dom = self._day_of_month
            bint is_start = isinstance(self, SemiMonthBegin)
            NPY_DATETIMEUNIT reso = get_unit_from_dtype(dtarr.dtype)
            cnp.broadcast mi = cnp.PyArray_MultiIterNew2(out, i8other)

        with nogil:
            for i in range(count):
                # Analogous to: val = i8other[i]
                val = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

                if val == NPY_NAT:
                    res_val = NPY_NAT

                else:
                    pandas_datetime_to_datetimestruct(val, reso, &dts)
                    day = dts.day

                    # Adjust so that we are always looking at self._day_of_month,
                    #  incrementing/decrementing n if necessary.
                    nadj = roll_convention(day, n, anchor_dom)

                    days_in_month = get_days_in_month(dts.year, dts.month)
                    # For SemiMonthBegin on other.day == 1 and
                    #  SemiMonthEnd on other.day == days_in_month,
                    #  shifting `other` to `self._day_of_month` _always_ requires
                    #  incrementing/decrementing `n`, regardless of whether it is
                    #  initially positive.
                    if is_start and (n <= 0 and day == 1):
                        nadj -= 1
                    elif (not is_start) and (n > 0 and day == days_in_month):
                        nadj += 1

                    if is_start:
                        # See also: SemiMonthBegin._apply
                        months = nadj // 2 + nadj % 2
                        to_day = 1 if nadj % 2 else anchor_dom

                    else:
                        # See also: SemiMonthEnd._apply
                        months = nadj // 2
                        to_day = 31 if nadj % 2 else anchor_dom

                    dts.year = year_add_months(dts, months)
                    dts.month = month_add_months(dts, months)
                    days_in_month = get_days_in_month(dts.year, dts.month)
                    dts.day = min(to_day, days_in_month)

                    res_val = npy_datetimestruct_to_datetime(reso, &dts)

                # Analogous to: out[i] = res_val
                (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = res_val

                cnp.PyArray_MultiIter_NEXT(mi)

        return out


cdef class SemiMonthEnd(SemiMonthOffset):
    # [CLASS DOC]
    # Semi-month end: anchors on day_of_month (default 15, min 1) and the last day
    # of the month. Alias "SME".
    """
    Two DateOffset's per month repeating on the last day of the month & day_of_month.

    This offset allows for flexibility in generating date ranges or adjusting dates
    to the end of a month or a specific day in the month, such as the 15th or the last
    day of the month. It is useful for financial or scheduling applications where
    events occur bi-monthly.

    Attributes
    ----------
    n : int, default 1
        The number of months represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    day_of_month : int, {1, 3,...,27}, default 15
        A specific integer for the day of the month.

    See Also
    --------
    tseries.offsets.SemiMonthBegin : Offset for semi-monthly frequencies, starting at
        the beginning of the month.
    tseries.offsets.MonthEnd : Offset to the last calendar day of the month.
    tseries.offsets.MonthBegin : Offset to the first calendar day of the month.

    Examples
    --------
    >>> ts = pd.Timestamp(2022, 1, 14)
    >>> ts + pd.offsets.SemiMonthEnd()
    Timestamp('2022-01-15 00:00:00')

    >>> ts = pd.Timestamp(2022, 1, 15)
    >>> ts + pd.offsets.SemiMonthEnd()
    Timestamp('2022-01-31 00:00:00')

    >>> ts = pd.Timestamp(2022, 1, 31)
    >>> ts + pd.offsets.SemiMonthEnd()
    Timestamp('2022-02-15 00:00:00')

    If you want to get the result for the current month:

    >>> ts = pd.Timestamp(2022, 1, 15)
    >>> pd.offsets.SemiMonthEnd().rollforward(ts)
    Timestamp('2022-01-15 00:00:00')
    """
    _prefix = "SME"
    _min_day_of_month = 1

    def is_on_offset(self, dt: datetime) -> bool:
        # [FUNCTION DOC]
        # True if dt's day is day_of_month or the month's last day (and midnight when
        # normalize is set).
        if self.normalize and not _is_normalized(dt):
            return False
        days_in_month = get_days_in_month(dt.year, dt.month)
        return dt.day in (self._day_of_month, days_in_month)


cdef class SemiMonthBegin(SemiMonthOffset):
    # [CLASS DOC]
    # Semi-month begin: anchors on the 1st and day_of_month (default 15, min 2).
    # Alias "SMS".
    """
    Two DateOffset's per month repeating on the first day of the month & day_of_month.

    This offset moves dates to the first day of the month and an additional specified
    day (typically the 15th by default), useful in scenarios where bi-monthly processing
    occurs on set days.

    Attributes
    ----------
    n : int, default 1
        The number of months represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    day_of_month : int, {1, 3,...,27}, default 15
        A specific integer for the day of the month.

    See Also
    --------
    tseries.offsets.SemiMonthEnd : Two DateOffset's per month repeating on the last day
        of the month & day_of_month.
    tseries.offsets.MonthEnd : Offset to the last calendar day of the month.
    tseries.offsets.MonthBegin : Offset to the first calendar day of the month.

    Examples
    --------
    >>> ts = pd.Timestamp(2022, 1, 1)
    >>> ts + pd.offsets.SemiMonthBegin()
    Timestamp('2022-01-15 00:00:00')
    """

    _prefix = "SMS"

    def is_on_offset(self, dt: datetime) -> bool:
        # [FUNCTION DOC]
        # True if dt's day is 1 or day_of_month (and midnight when normalize is set).
        if self.normalize and not _is_normalized(dt):
            return False
        return dt.day in (1, self._day_of_month)


# ---------------------------------------------------------------------
# Week-Based Offset Classes


cdef class Week(SingleConstructorOffset):
    # [CLASS DOC]
    # Weekly offset. With weekday None, a plain 7*n day shift; with weekday
    # 0..6, anchored to that weekday (and sets _period_dtype_code for Periods).
    """
    Weekly offset.

    Attributes
    ----------
    n : int, default 1
        The number of weeks represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    weekday : int or None, default None
        Always generate specific day of week.
        0 for Monday and 6 for Sunday.

    See Also
    --------
    pd.tseries.offsets.WeekOfMonth :
     Describes monthly dates like, the Tuesday of the
     2nd week of each month.

    Examples
    --------

    >>> date_object = pd.Timestamp("2023-01-13")
    >>> date_object
    Timestamp('2023-01-13 00:00:00')

    >>> date_plus_one_week = date_object + pd.tseries.offsets.Week(n=1)
    >>> date_plus_one_week
    Timestamp('2023-01-20 00:00:00')

    >>> date_next_monday = date_object + pd.tseries.offsets.Week(weekday=0)
    >>> date_next_monday
    Timestamp('2023-01-16 00:00:00')

    >>> date_next_sunday = date_object + pd.tseries.offsets.Week(weekday=6)
    >>> date_next_sunday
    Timestamp('2023-01-15 00:00:00')
    """

    _inc = timedelta(weeks=1)
    _prefix = "W"
    _attributes = tuple(["n", "normalize", "weekday"])

    cdef readonly:
        object _weekday  # int or None
        int _period_dtype_code

    @property
    def weekday(self):
        # [FUNCTION DOC]
        # Property: anchor weekday (0=Mon..6=Sun) or None.
        """
        Return the day of the week on which the offset is applied.

        This is the weekday on which the weekly offset is anchored.
        The value is an integer representing the day of the week,
        where Monday is 0 and Sunday is 6, or None if no specific
        weekday is set.

        See Also
        --------
        Week : The weekly offset class.
        tseries.offsets.WeekOfMonth : Describes monthly dates like the Tuesday
            of the 2nd week of each month.

        Examples
        --------
        >>> pd.offsets.Week(weekday=0).weekday
        0

        >>> pd.offsets.Week(weekday=6).weekday
        6

        >>> pd.offsets.Week().weekday is None
        True
        """
        return self._weekday

    def __init__(self, n=1, normalize=False, weekday=None):
        # [FUNCTION DOC]
        # Initialize n/normalize/weekday; if weekday given, validate and compute
        # _period_dtype_code = W_SUN code + (weekday+1) % 7.
        # Exceptions: ValueError if weekday outside 0..6.
        BaseOffset.__init__(self, n, normalize)
        self._weekday = weekday

        if self._weekday is not None:
            if self._weekday < 0 or self._weekday > 6:
                raise ValueError(f"Day must be 0<=day<=6, got {self._weekday}")

            self._period_dtype_code = PeriodDtypeCode.W_SUN + (weekday + 1) % 7

    cpdef __setstate__(self, state):
        # [FUNCTION DOC]
        # Unpickle: pop n, normalize, weekday, _cache. Side effects: mutates state.
        self._n = state.pop("n")
        self._normalize = state.pop("normalize")
        self._weekday = state.pop("weekday")
        self._cache = state.pop("_cache", {})

    @apply_wraps
    def _apply(self, other):
        # [FUNCTION DOC]
        # Apply (wrapped by apply_wraps): unanchored -> other + n weeks. Anchored ->
        # move forward to the next anchor weekday (counts as one step if n > 0), then
        # add remaining k weeks.
        # Exceptions: TypeError if anchored and other is not a datetime.
        if self.weekday is None:
            return other + self._n * self._inc

        if not PyDateTime_Check(other):
            raise TypeError(
                f"Cannot add {type(other).__name__} to {type(self).__name__}"
            )

        k = self._n
        otherDay = other.weekday()
        if otherDay != self.weekday:
            other = other + timedelta((self.weekday - otherDay) % 7)
            if k > 0:
                k -= 1

        return other + timedelta(weeks=k)

    def _apply_array(self, dtarr: np.ndarray) -> np.ndarray:
        # [FUNCTION DOC]
        # Vectorized apply: unanchored adds a timedelta64 of 7*n days in the array's
        # unit; anchored calls _end_apply_index on the int64 view.
        # Outputs: datetime64 ndarray (unanchored) or int64 ndarray (anchored).
        if self.weekday is None:
            td = timedelta(days=7 * self._n)
            unit = np.datetime_data(dtarr.dtype)[0]
            td64 = np.timedelta64(td, unit)
            return dtarr + td64
        else:
            reso = get_unit_from_dtype(dtarr.dtype)
            i8other = dtarr.view("i8")
            return self._end_apply_index(i8other, reso=reso)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef ndarray _end_apply_index(self, ndarray i8other, NPY_DATETIMEUNIT reso):
        # [FUNCTION DOC]
        # HELPER: Vectorized anchored-week shift on int64 epoch values in unit reso.
        # For each non-NaT value: days = (anchor - weekday) mod 7; weeks = n, minus
        # one if days > 0 and n > 0; result = value + (7*weeks + days) days.
        # Outputs: new int64 ndarray of same shape. Side effects: releases the GIL;
        # allocates output; bounds/wraparound checks disabled.
        """
        Add self to the given DatetimeIndex, specialized for case where
        self.weekday is non-null.

        Parameters
        ----------
        i8other : const int64_t[:]
        reso : NPY_DATETIMEUNIT

        Returns
        -------
        ndarray[int64_t]
        """
        cdef:
            Py_ssize_t i, count = i8other.size
            int64_t val, res_val
            ndarray out = cnp.PyArray_EMPTY(
                i8other.ndim, i8other.shape, cnp.NPY_INT64, 0
            )
            npy_datetimestruct dts
            int wday, days, weeks, n = self._n
            int anchor_weekday = self.weekday
            int64_t DAY_PERIODS = periods_per_day(reso)
            cnp.broadcast mi = cnp.PyArray_MultiIterNew2(out, i8other)

        with nogil:
            for i in range(count):
                # Analogous to: val = i8other[i]
                val = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

                if val == NPY_NAT:
                    res_val = NPY_NAT
                else:
                    pandas_datetime_to_datetimestruct(val, reso, &dts)
                    wday = dayofweek(dts.year, dts.month, dts.day)

                    days = 0
                    weeks = n
                    if wday != anchor_weekday:
                        days = (anchor_weekday - wday) % 7
                        if weeks > 0:
                            weeks -= 1

                    res_val = val + (7 * weeks + days) * DAY_PERIODS

                # Analogous to: out[i] = res_val
                (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = res_val

                cnp.PyArray_MultiIter_NEXT(mi)

        return out

    def is_on_offset(self, dt: datetime) -> bool:
        # [FUNCTION DOC]
        # True for any date if unanchored; otherwise dt's weekday equals the anchor
        # (and midnight when normalize is set).
        if self.normalize and not _is_normalized(dt):
            return False
        elif self.weekday is None:
            return True
        return dt.weekday() == self.weekday

    @property
    def rule_code(self) -> str:
        # [FUNCTION DOC]
        # Property: "W" or "W-{DAY}" (e.g. "W-SUN") when anchored.
        """
        Return a string representing the base frequency.

        See Also
        --------
        tseries.offsets.Hour.name :
            Returns a string representing the base frequency of 'h'.
        tseries.offsets.Day.name :
            Returns a string representing the base frequency of 'D'.

        Examples
        --------
        >>> pd.offsets.Hour().rule_code
        'h'

        >>> pd.offsets.Week(5).rule_code
        'W'
        """
        suffix = ""
        if self.weekday is not None:
            weekday = int_to_weekday[self.weekday]
            suffix = f"-{weekday}"
        return self._prefix + suffix

    @classmethod
    def _from_name(cls, suffix=None):
        # [FUNCTION DOC]
        # Classmethod: construct from optional weekday-abbreviation suffix.
        # Exceptions: KeyError for unknown weekday abbreviation.
        if not suffix:
            weekday = None
        else:
            weekday = weekday_to_int[suffix]
        return cls(weekday=weekday)


cdef class WeekOfMonth(WeekOfMonthMixin):
    # [CLASS DOC]
    # Offset to the given weekday of the given week (0..3) of each month, e.g.
    # "the 2nd Tuesday". Alias "WOM-{week+1}{DAY}".
    """
    Describes monthly dates like "the Tuesday of the 2nd week of each month".

    This offset allows for generating or adjusting dates by specifying
    a particular week and weekday within a month. The week is zero-indexed,
    where 0 corresponds to the first week of the month, and weekday follows
    a Monday=0 convention.

    Attributes
    ----------
    n : int, default 1
        The number of months represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    week : int {0, 1, 2, 3, ...}, default 0
        A specific integer for the week of the month.
        e.g. 0 is 1st week of month, 1 is the 2nd week, etc.
    weekday : int {0, 1, ..., 6}, default 0
        A specific integer for the day of the week.

        - 0 is Monday
        - 1 is Tuesday
        - 2 is Wednesday
        - 3 is Thursday
        - 4 is Friday
        - 5 is Saturday
        - 6 is Sunday.

    See Also
    --------
    offsets.Week : Describes weekly frequency adjustments.
    offsets.MonthEnd : Describes month-end frequency adjustments.
    date_range : Generates a range of dates based on a specific frequency.

    Examples
    --------
    >>> ts = pd.Timestamp(2022, 1, 1)
    >>> ts + pd.offsets.WeekOfMonth()
    Timestamp('2022-01-03 00:00:00')
    """

    _prefix = "WOM"
    _attributes = tuple(["n", "normalize", "week", "weekday"])

    def __init__(self, n=1, normalize=False, week=0, weekday=0):
        # [FUNCTION DOC]
        # Initialize with weekday (via mixin) and week.
        # Exceptions: ValueError if week not in 0..3 or weekday not in 0..6.
        WeekOfMonthMixin.__init__(self, n, normalize, weekday)
        self.week = week

        if self.week < 0 or self.week > 3:
            raise ValueError(f"Week must be 0<=week<=3, got {self.week}")

    cpdef __setstate__(self, state):
        # [FUNCTION DOC]
        # Unpickle: pop n, normalize, weekday, week. Side effects: mutates state.
        self._n = state.pop("n")
        self._normalize = state.pop("normalize")
        self.weekday = state.pop("weekday")
        self.week = state.pop("week")

    def _get_offset_day(self, other: datetime) -> int:
        # [FUNCTION DOC]
        # HELPER: Day of month of the (week+1)-th occurrence of weekday in other's
        # month: 1 + ((weekday - weekday_of_first) mod 7) + 7*week. Outputs: int.
        """
        Find the day in the same month as other that has the same
        weekday as self.weekday and is the self.week'th such day in the month.

        Parameters
        ----------
        other : datetime

        Returns
        -------
        day : int
        """
        mstart = datetime(other.year, other.month, 1)
        wday = mstart.weekday()
        shift_days = (self.weekday - wday) % 7
        return 1 + shift_days + self.week * 7

    @classmethod
    def _from_name(cls, suffix=None):
        # [FUNCTION DOC]
        # Classmethod: parse suffix like "2TUE" (single-digit week 1-4, then weekday
        # abbreviation). Exceptions: ValueError if no suffix or bad digit; KeyError for
        # unknown weekday.
        if not suffix:
            raise ValueError(f"Prefix {repr(cls._prefix)} requires a suffix.")
        # only one digit weeks (1 --> week 0, 2 --> week 1, etc.)
        week = int(suffix[0]) - 1
        weekday = weekday_to_int[suffix[1:]]
        return cls(week=week, weekday=weekday)


cdef class LastWeekOfMonth(WeekOfMonthMixin):
    # [CLASS DOC]
    # Offset to the last given weekday of each month. Alias "LWOM-{DAY}".
    # week is fixed to -1; n must be nonzero.
    """
    Describes monthly dates in last week of month.

    For example "the last Tuesday of each month".

    Attributes
    ----------
    n : int, default 1
        The number of months represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    weekday : int {0, 1, ..., 6}, default 0
        A specific integer for the day of the week.

        - 0 is Monday
        - 1 is Tuesday
        - 2 is Wednesday
        - 3 is Thursday
        - 4 is Friday
        - 5 is Saturday
        - 6 is Sunday.

    See Also
    --------
    tseries.offsets.WeekOfMonth :
        Date offset for a specific weekday in a month.
    tseries.offsets.MonthEnd :
        Date offset for the end of the month.
    tseries.offsets.BMonthEnd :
        Date offset for the last business day of the month.

    Examples
    --------
    >>> ts = pd.Timestamp(2022, 1, 1)
    >>> ts + pd.offsets.LastWeekOfMonth()
    Timestamp('2022-01-31 00:00:00')
    """

    _prefix = "LWOM"
    _attributes = tuple(["n", "normalize", "weekday"])

    def __init__(self, n=1, normalize=False, weekday=0):
        # [FUNCTION DOC]
        # Initialize weekday (via mixin) and set week = -1.
        # Exceptions: ValueError if n == 0 or weekday invalid.
        WeekOfMonthMixin.__init__(self, n, normalize, weekday)
        self.week = -1

        if self._n == 0:
            raise ValueError("N cannot be 0")

    cpdef __setstate__(self, state):
        # [FUNCTION DOC]
        # Unpickle: pop n, normalize, weekday; set week -1. Side effects: mutates
        # state.
        self._n = state.pop("n")
        self._normalize = state.pop("normalize")
        self.weekday = state.pop("weekday")
        self.week = -1

    def _get_offset_day(self, other: datetime) -> int:
        # [FUNCTION DOC]
        # HELPER: Day of month of the last occurrence of weekday in other's month:
        # days_in_month - ((weekday_of_last_day - weekday) mod 7). Outputs: int.
        """
        Find the day in the same month as other that has the same
        weekday as self.weekday and is the last such day in the month.

        Parameters
        ----------
        other: datetime

        Returns
        -------
        day: int
        """
        dim = get_days_in_month(other.year, other.month)
        mend = datetime(other.year, other.month, dim)
        wday = mend.weekday()
        shift_days = (wday - self.weekday) % 7
        return dim - shift_days

    @classmethod
    def _from_name(cls, suffix=None):
        # [FUNCTION DOC]
        # Classmethod: parse weekday abbreviation suffix.
        # Exceptions: ValueError if no suffix; KeyError for unknown weekday.
        if not suffix:
            raise ValueError(f"Prefix {repr(cls._prefix)} requires a suffix.")
        weekday = weekday_to_int[suffix]
        return cls(weekday=weekday)


# ---------------------------------------------------------------------
# Special Offset Classes

cdef class FY5253Mixin(SingleConstructorOffset):
    # [CLASS DOC]
    # Shared base for 52-53 week fiscal year offsets: stores startingMonth (the
    # month in which the fiscal year ends), _weekday (fiscal year ends on this
    # weekday) and variation ("nearest": the weekday nearest to the last day of
    # startingMonth; "last": the last such weekday in startingMonth).
    cdef readonly:
        int startingMonth
        int _weekday
        str variation

    @property
    def weekday(self):
        # [FUNCTION DOC]
        # Property: the anchor weekday (0=Mon..6=Sun).
        """
        Return the weekday used by the fiscal year.

        This is the day of the week on which the fiscal year ends.
        The value is an integer from 0 (Monday) to 6 (Sunday).

        See Also
        --------
        FY5253.startingMonth : Return the starting month of the fiscal year.
        FY5253.variation : Return the variation of the fiscal year.

        Examples
        --------
        >>> offset = pd.offsets.FY5253(weekday=4, startingMonth=12, variation="nearest")
        >>> offset.weekday
        4
        """
        return self._weekday

    def __init__(
        self, n=1, normalize=False, weekday=0, startingMonth=1, variation="nearest"
    ):
        # [FUNCTION DOC]
        # Initialize n, normalize, weekday, startingMonth, variation.
        # Exceptions: ValueError if n == 0 or variation not "nearest"/"last".
        # Note: weekday and startingMonth are not range-validated.
        BaseOffset.__init__(self, n, normalize)
        self.startingMonth = startingMonth
        self._weekday = weekday
        self.variation = variation

        if self._n == 0:
            raise ValueError("N cannot be 0")

        if self.variation not in ["nearest", "last"]:
            raise ValueError(f"{self.variation} is not a valid variation")

    cpdef __setstate__(self, state):
        # [FUNCTION DOC]
        # Unpickle: pop n, normalize, weekday, variation. Side effects: mutates state.
        # KNOWN ISSUE: startingMonth is not restored from state (left at 0) and the
        # "startingMonth" key is left in state; only legacy pickles reach this since
        # __reduce__ reconstructs via the constructor.
        self._n = state.pop("n")
        self._normalize = state.pop("normalize")
        self._weekday = state.pop("weekday")
        self.variation = state.pop("variation")

    # --------------------------------------------------------------------
    # Name-related methods

    @property
    def rule_code(self) -> str:
        # [FUNCTION DOC]
        # Property: "{prefix}-{suffix}" e.g. "RE-N-JAN-MON".
        prefix = self._prefix
        suffix = self.get_rule_code_suffix()
        return f"{prefix}-{suffix}"

    def _get_suffix_prefix(self) -> str:
        # [FUNCTION DOC]
        # HELPER: "N" for nearest variation, "L" otherwise.
        if self.variation == "nearest":
            return "N"
        else:
            return "L"

    def get_rule_code_suffix(self) -> str:
        # [FUNCTION DOC]
        # Return rule code suffix "{N|L}-{MON}-{DAY}" from variation, startingMonth
        # and weekday. Exceptions: KeyError if month/weekday out of range.
        """
        Return the suffix component of the rule code.

        The suffix combines the variation prefix, starting month abbreviation,
        and weekday abbreviation into a string format.

        Returns
        -------
        str
            Suffix string in the format "{variation}-{month}-{weekday}".
            The variation is "N" for nearest or "L" for last.

        See Also
        --------
        FY5253.rule_code : Return the complete rule code string.
        FY5253Quarter.rule_code : Return the complete rule code string for
            FY5253Quarter.

        Examples
        --------
        >>> offset = pd.offsets.FY5253(weekday=4, startingMonth=12, variation="nearest")
        >>> offset.get_rule_code_suffix()
        'N-DEC-FRI'

        >>> offset = pd.offsets.FY5253(weekday=0, startingMonth=1, variation="last")
        >>> offset.get_rule_code_suffix()
        'L-JAN-MON'
        """
        prefix = self._get_suffix_prefix()
        month = MONTH_ALIASES[self.startingMonth]
        weekday = int_to_weekday[self._weekday]
        return f"{prefix}-{month}-{weekday}"


cdef class FY5253(FY5253Mixin):
    # [CLASS DOC]
    # 52-53 week fiscal year (retail calendar) offset; each fiscal year ends on
    # the anchor weekday at/near the end of startingMonth. Alias "RE-...".
    """
    Describes 52-53 week fiscal year. This is also known as a 4-4-5 calendar.

    It is used by companies that desire that their
    fiscal year always end on the same day of the week.

    It is a method of managing accounting periods.
    It is a common calendar structure for some industries,
    such as retail, manufacturing and parking industry.

    For more information see:
    https://en.wikipedia.org/wiki/4-4-5_calendar

    The year may either:

    - end on the last X day of the Y month.
    - end on the last X day closest to the last day of the Y month.

    X is a specific day of the week.
    Y is a certain month of the year

    Attributes
    ----------
    n : int
        The number of fiscal years represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    weekday : int {0, 1, ..., 6}, default 0
        A specific integer for the day of the week.

        - 0 is Monday
        - 1 is Tuesday
        - 2 is Wednesday
        - 3 is Thursday
        - 4 is Friday
        - 5 is Saturday
        - 6 is Sunday.

    startingMonth : int {1, 2, ... 12}, default 1
        The month in which the fiscal year ends.

    variation : str, default "nearest"
        Method of employing 4-4-5 calendar.

        There are two options:

        - "nearest" means year end is **weekday** closest to last day of month in year.
        - "last" means year end is final **weekday** of the final month in fiscal year.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    In the example below the default parameters give the next 52-53 week fiscal year.

    >>> ts = pd.Timestamp(2022, 1, 1)
    >>> ts + pd.offsets.FY5253()
    Timestamp('2022-01-31 00:00:00')

    By the parameter ``startingMonth`` we can specify
    the month in which fiscal years end.

    >>> ts = pd.Timestamp(2022, 1, 1)
    >>> ts + pd.offsets.FY5253(startingMonth=3)
    Timestamp('2022-03-28 00:00:00')

    52-53 week fiscal year can be specified by
    ``weekday`` and ``variation`` parameters.

    >>> ts = pd.Timestamp(2022, 1, 1)
    >>> ts + pd.offsets.FY5253(weekday=5, startingMonth=12, variation="last")
    Timestamp('2022-12-31 00:00:00')
    """

    _prefix = "RE"
    _attributes = tuple(["n", "normalize", "weekday", "startingMonth", "variation"])

    def is_on_offset(self, dt: datetime) -> bool:
        # [FUNCTION DOC]
        # True if dt's date equals this calendar year's fiscal year end, or (nearest
        # variation) the previous year's end computed from one month before dt (since
        # a nearest year end can fall in early next month). Midnight check when
        # normalize is set.
        if self.normalize and not _is_normalized(dt):
            return False
        dt = datetime(dt.year, dt.month, dt.day)
        year_end = self.get_year_end(dt)

        if self.variation == "nearest":
            # We have to check the year end of "this" cal year AND the previous
            return year_end == dt or self.get_year_end(shift_month(dt, -1, None)) == dt
        else:
            return year_end == dt

    @apply_wraps
    def _apply(self, other: datetime) -> datetime:
        # [FUNCTION DOC]
        # Apply (wrapped by apply_wraps): compute the fiscal year ends for other's
        # previous, current and next calendar years (localized to other's tz),
        # compare normalized other against them to adjust n (on/before/between),
        # then return the fiscal year end of year other.year + n with other's time of
        # day. Exceptions: AssertionError on an unexpected configuration.
        norm = Timestamp(other).normalize()

        n = self._n
        prev_year = self.get_year_end(datetime(other.year - 1, self.startingMonth, 1))
        cur_year = self.get_year_end(datetime(other.year, self.startingMonth, 1))
        next_year = self.get_year_end(datetime(other.year + 1, self.startingMonth, 1))

        prev_year = localize_pydatetime(prev_year, other.tzinfo)
        cur_year = localize_pydatetime(cur_year, other.tzinfo)
        next_year = localize_pydatetime(next_year, other.tzinfo)

        # Note: next_year.year == other.year + 1, so we will always
        # have other < next_year
        if norm == prev_year:
            n -= 1
        elif norm == cur_year:
            pass
        elif n > 0:
            if norm < prev_year:
                n -= 2
            elif prev_year < norm < cur_year:
                n -= 1
            elif cur_year < norm < next_year:
                pass
        else:
            if cur_year < norm < next_year:
                n += 1
            elif prev_year < norm < cur_year:
                pass
            elif (
                norm.year == prev_year.year
                and norm < prev_year
                and prev_year - norm <= timedelta(6)
            ):
                # GH#14774, error when next_year.year == cur_year.year
                # e.g. prev_year == datetime(2004, 1, 3),
                # other == datetime(2004, 1, 1)
                n -= 1
            else:
                assert False

        shifted = datetime(other.year + n, self.startingMonth, 1)
        result = self.get_year_end(shifted)
        result = datetime(
            result.year,
            result.month,
            result.day,
            other.hour,
            other.minute,
            other.second,
            other.microsecond,
        )
        return result

    def get_year_end(self, dt: datetime) -> datetime:
        # [FUNCTION DOC]
        # Return the fiscal year end date for the calendar year of dt: start from the
        # last day of startingMonth; if its weekday matches return it; "last" moves
        # back to the previous anchor weekday; "nearest" moves forward up to 3 days or
        # back otherwise. Inputs: dt - tz-naive datetime.
        # Exceptions: AssertionError if dt is tz-aware.
        assert dt.tzinfo is None

        dim = get_days_in_month(dt.year, self.startingMonth)
        target_date = datetime(dt.year, self.startingMonth, dim)
        wkday_diff = self.weekday - target_date.weekday()
        if wkday_diff == 0:
            # year_end is the same for "last" and "nearest" cases
            return target_date

        if self.variation == "last":
            days_forward = (wkday_diff % 7) - 7

            # days_forward is always negative, so we always end up
            # in the same year as dt
            return target_date + timedelta(days=days_forward)
        else:
            # variation == "nearest":
            days_forward = wkday_diff % 7
            if days_forward <= 3:
                # The upcoming self.weekday is closer than the previous one
                return target_date + timedelta(days_forward)
            else:
                # The previous self.weekday is closer than the upcoming one
                return target_date + timedelta(days_forward - 7)

    @classmethod
    def _parse_suffix(cls, varion_code, startingMonth_code, weekday_code):
        # [FUNCTION DOC]
        # Classmethod: parse alias parts into constructor kwargs.
        # Inputs: varion_code - "N" or "L"; startingMonth_code - month abbreviation;
        # weekday_code - weekday abbreviation.
        # Outputs: dict with weekday, startingMonth, variation.
        # Exceptions: ValueError for bad variation code; KeyError for bad month/day.
        if varion_code == "N":
            variation = "nearest"
        elif varion_code == "L":
            variation = "last"
        else:
            raise ValueError(f"Unable to parse varion_code: {varion_code}")

        startingMonth = MONTH_TO_CAL_NUM[startingMonth_code]
        weekday = weekday_to_int[weekday_code]

        return {
            "weekday": weekday,
            "startingMonth": startingMonth,
            "variation": variation,
        }

    @classmethod
    def _from_name(cls, *args):
        # [FUNCTION DOC]
        # Classmethod: construct from alias parts via _parse_suffix.
        return cls(**cls._parse_suffix(*args))


cdef class FY5253Quarter(FY5253Mixin):
    # [CLASS DOC]
    # 52-53 week fiscal quarter offset. Uses an FY5253 year-end (_offset) with the
    # same weekday/startingMonth/variation; each fiscal year has four quarters of
    # 13 weeks, with qtr_with_extra_week (1..4) getting 14 weeks in 53-week years.
    # Alias "REQ-{N|L}-{MON}-{DAY}-{qtr}".
    """
    DateOffset increments between business quarter dates for 52-53 week fiscal year.

    Also known as a 4-4-5 calendar.

    It is used by companies that desire that their
    fiscal year always end on the same day of the week.

    It is a method of managing accounting periods.
    It is a common calendar structure for some industries,
    such as retail, manufacturing and parking industry.

    For more information see:
    https://en.wikipedia.org/wiki/4-4-5_calendar

    The year may either:

    - end on the last X day of the Y month.
    - end on the last X day closest to the last day of the Y month.

    X is a specific day of the week.
    Y is a certain month of the year

    startingMonth = 1 corresponds to dates like 1/31/2007, 4/30/2007, ...
    startingMonth = 2 corresponds to dates like 2/28/2007, 5/31/2007, ...
    startingMonth = 3 corresponds to dates like 3/30/2007, 6/29/2007, ...

    Attributes
    ----------
    n : int
        The number of business quarters represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    weekday : int {0, 1, ..., 6}, default 0
        A specific integer for the day of the week.

        - 0 is Monday
        - 1 is Tuesday
        - 2 is Wednesday
        - 3 is Thursday
        - 4 is Friday
        - 5 is Saturday
        - 6 is Sunday.

    startingMonth : int {1, 2, ..., 12}, default 1
        The month in which fiscal years end.

    qtr_with_extra_week : int {1, 2, 3, 4}, default 1
        The quarter number that has the leap or 14 week when needed.

    variation : str, default "nearest"
        Method of employing 4-4-5 calendar.

        There are two options:

        - "nearest" means year end is **weekday** closest to last day of month in year.
        - "last" means year end is final **weekday** of the final month in fiscal year.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    In the example below the default parameters give
    the next business quarter for 52-53 week fiscal year.

    >>> ts = pd.Timestamp(2022, 1, 1)
    >>> ts + pd.offsets.FY5253Quarter()
    Timestamp('2022-01-31 00:00:00')

    By the parameter ``startingMonth`` we can specify
    the month in which fiscal years end.

    >>> ts = pd.Timestamp(2022, 1, 1)
    >>> ts + pd.offsets.FY5253Quarter(startingMonth=3)
    Timestamp('2022-03-28 00:00:00')

    Business quarters for 52-53 week fiscal year can be specified by
    ``weekday`` and ``variation`` parameters.

    >>> ts = pd.Timestamp(2022, 1, 1)
    >>> ts + pd.offsets.FY5253Quarter(weekday=5, startingMonth=12, variation="last")
    Timestamp('2022-04-02 00:00:00')
    """

    _prefix = "REQ"
    _attributes = tuple(
        [
            "n",
            "normalize",
            "weekday",
            "startingMonth",
            "qtr_with_extra_week",
            "variation",
        ]
    )

    cdef readonly:
        int qtr_with_extra_week

    def __init__(
        self,
        n=1,
        normalize=False,
        weekday=0,
        startingMonth=1,
        qtr_with_extra_week=1,
        variation="nearest",
    ):
        # [FUNCTION DOC]
        # Initialize via FY5253Mixin and set qtr_with_extra_week.
        # Exceptions: as FY5253Mixin.__init__. Note: qtr_with_extra_week not
        # range-validated (values outside 1..4 break get_weeks).
        FY5253Mixin.__init__(
            self, n, normalize, weekday, startingMonth, variation
        )
        self.qtr_with_extra_week = qtr_with_extra_week

    cpdef __setstate__(self, state):
        # [FUNCTION DOC]
        # Unpickle: FY5253Mixin.__setstate__ then pop qtr_with_extra_week.
        # Side effects: mutates state. Inherits the startingMonth restore issue.
        FY5253Mixin.__setstate__(self, state)
        self.qtr_with_extra_week = state.pop("qtr_with_extra_week")

    @cache_readonly
    def _offset(self):
        # [FUNCTION DOC]
        # Cached property: the FY5253 year-end offset (n=1) matching this offset's
        # startingMonth, weekday and variation.
        return FY5253(
            startingMonth=self.startingMonth,
            weekday=self.weekday,
            variation=self.variation,
        )

    def _rollback_to_year(self, other: datetime):
        # [FUNCTION DOC]
        # HELPER: Roll other back to the most recent fiscal year end and count whole
        # fiscal quarters between it and other.
        # Inputs: other - datetime (tz is dropped). Outputs: tuple (start year-end
        # Timestamp, number of complete quarters, remaining Timedelta >= 0).
        # Exceptions: AssertionError if quarter lengths are inconsistent with the
        # year-end offset.
        """
        Roll `other` back to the most recent date that was on a fiscal year
        end.

        Return the date of that year-end, the number of full quarters
        elapsed between that year-end and other, and the remaining Timedelta
        since the most recent quarter-end.

        Parameters
        ----------
        other : datetime or Timestamp

        Returns
        -------
        tuple of
        prev_year_end : Timestamp giving most recent fiscal year end
        num_qtrs : int
        tdelta : Timedelta
        """
        num_qtrs = 0

        norm = Timestamp(other).tz_localize(None)
        start = self._offset.rollback(norm)
        # Note: start <= norm and self._offset.is_on_offset(start)

        if start < norm:
            # roll adjustment
            qtr_lens = self.get_weeks(norm)

            # check that qtr_lens is consistent with self._offset addition
            end = _shift_day(start, days=7 * sum(qtr_lens))
            assert self._offset.is_on_offset(end), (start, end, qtr_lens)

            tdelta = norm - start
            for qlen in qtr_lens:
                if qlen * 7 <= tdelta.days:
                    num_qtrs += 1
                    tdelta -= (
                        <_Timedelta>Timedelta(days=qlen * 7)
                    )._as_creso(norm._creso)
                else:
                    break
        else:
            tdelta = Timedelta(0)

        # Note: we always have tdelta._value>= 0
        return start, num_qtrs, tdelta

    @apply_wraps
    def _apply(self, other: datetime) -> datetime:
        # [FUNCTION DOC]
        # Apply (wrapped by apply_wraps): roll back to the last year end, add the
        # completed quarters to n (plus one if n <= 0 and other is past a boundary),
        # shift by whole years (n//4 FY5253 steps), then add the week lengths of the
        # remaining 0..3 quarters of the upcoming fiscal year.
        # Note: self._n == 0 is not allowed.

        n = self._n

        prev_year_end, num_qtrs, tdelta = self._rollback_to_year(other)
        res = prev_year_end
        n += num_qtrs
        if self._n <= 0 and tdelta._value > 0:
            n += 1

        # Possible speedup by handling years first.
        years = n // 4
        if years:
            res += self._offset * years
            n -= years * 4

        # Add an extra day to make *sure* we are getting the quarter lengths
        # for the upcoming year, not the previous year
        qtr_lens = self.get_weeks(res + Timedelta(days=1))

        # Note: we always have 0 <= n < 4
        weeks = sum(qtr_lens[:n])
        if weeks:
            res = _shift_day(res, days=weeks * 7)

        return res

    def get_weeks(self, dt: datetime):
        # [FUNCTION DOC]
        # Return a list of 4 quarter lengths in weeks for the fiscal year containing
        # dt: [13,13,13,13], with 14 for qtr_with_extra_week if the year has 53 weeks.
        ret = [13] * 4

        year_has_extra_week = self.year_has_extra_week(dt)

        if year_has_extra_week:
            ret[self.qtr_with_extra_week - 1] = 14

        return ret

    def year_has_extra_week(self, dt: datetime) -> bool:
        # [FUNCTION DOC]
        # True if the fiscal year containing dt (normalized, tz-naive) has 53 weeks:
        # distance from the previous year end to the next year end in weeks.
        # Exceptions: AssertionError if the week count is not 52 or 53.
        # Avoid round-down errors --> normalize to get
        # e.g. '370D' instead of '360D23H'
        norm = Timestamp(dt).normalize().tz_localize(None)

        next_year_end = self._offset.rollforward(norm)
        prev_year_end = norm - self._offset
        weeks_in_year = (next_year_end - prev_year_end).days / 7
        assert weeks_in_year in [52, 53], weeks_in_year
        return weeks_in_year == 53

    def is_on_offset(self, dt: datetime) -> bool:
        # [FUNCTION DOC]
        # True if dt is a fiscal year end, or equals one of the cumulative quarter
        # boundaries counted from the previous year end (dt - year offset); midnight
        # check when normalize is set.
        if self.normalize and not _is_normalized(dt):
            return False
        if self._offset.is_on_offset(dt):
            return True

        next_year_end = dt - self._offset

        qtr_lens = self.get_weeks(dt)

        current = next_year_end
        for qtr_len in qtr_lens:
            current = _shift_day(current, days=qtr_len * 7)
            if dt == current:
                return True
        return False

    @property
    def rule_code(self) -> str:
        # [FUNCTION DOC]
        # Property: FY5253 rule code plus "-{qtr_with_extra_week}".
        suffix = FY5253Mixin.rule_code.__get__(self)
        qtr = self.qtr_with_extra_week
        return f"{suffix}-{qtr}"

    @classmethod
    def _from_name(cls, *args):
        # [FUNCTION DOC]
        # Classmethod: construct from alias parts (variation, month, weekday, quarter);
        # quarter is parsed with int(). Exceptions: ValueError/KeyError on bad parts.
        return cls(
            **dict(FY5253._parse_suffix(*args[:-1]), qtr_with_extra_week=int(args[-1]))
        )


cdef class Easter(SingleConstructorOffset):
    # [CLASS DOC]
    # Offset to Easter Sunday using dateutil's easter algorithm; method selects
    # 1=Julian, 2=Orthodox, 3=Western (default).
    """
    DateOffset for the Easter holiday using logic defined in dateutil.

    Right now uses the revised method which is valid in years 1583-4099.

    Attributes
    ----------
    n : int, default 1
        The number of years represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    method : int, default 3
        The method used to calculate the date of Easter. Valid options are:
        - 1 (EASTER_JULIAN): Original calculation in Julian calendar
        - 2 (EASTER_ORTHODOX): Original method, date converted to Gregorian calendar
        - 3 (EASTER_WESTERN): Revised method, in Gregorian calendar
        These constants are defined in the `dateutil.easter` module.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    >>> ts = pd.Timestamp(2022, 1, 1)
    >>> ts + pd.offsets.Easter()
    Timestamp('2022-04-17 00:00:00')
    """

    _attributes = tuple(["n", "normalize", "method"])

    cdef readonly:
        int method

    from dateutil.easter import EASTER_WESTERN

    def __init__(self, n=1, normalize=False, method=EASTER_WESTERN):
        # [FUNCTION DOC]
        # Initialize n/normalize/method. Exceptions: ValueError if method not in 1..3.
        BaseOffset.__init__(self, n, normalize)

        self.method = method

        if method < 1 or method > 3:
            raise ValueError(f"Method must be 1<=method<=3, got {method}")

    cpdef __setstate__(self, state):
        # [FUNCTION DOC]
        # Unpickle: pop n, normalize, method (default Western). Side effects: mutates
        # state.
        from dateutil.easter import EASTER_WESTERN
        self._n = state.pop("n")
        self._normalize = state.pop("normalize")
        self.method = state.pop("method", EASTER_WESTERN)

    @apply_wraps
    def _apply(self, other: datetime) -> datetime:
        # [FUNCTION DOC]
        # Apply (wrapped by apply_wraps): compute this year's Easter (localized to
        # other's tz); decrement n if n >= 0 and other is before it, increment if
        # n < 0 and other is after it; return Easter of year other.year + n with
        # other's time of day.
        from dateutil.easter import easter

        current_easter = easter(other.year, method=self.method)
        current_easter = datetime(
            current_easter.year, current_easter.month, current_easter.day
        )
        current_easter = localize_pydatetime(current_easter, other.tzinfo)

        n = self._n
        if n >= 0 and other < current_easter:
            n -= 1
        elif n < 0 and other > current_easter:
            n += 1
        # TODO: Why does this handle the 0 case the opposite of others?

        # NOTE: easter returns a datetime.date so we have to convert to type of
        # other
        new = easter(other.year + n, method=self.method)
        new = datetime(
            new.year,
            new.month,
            new.day,
            other.hour,
            other.minute,
            other.second,
            other.microsecond,
        )
        return new

    def is_on_offset(self, dt: datetime) -> bool:
        # [FUNCTION DOC]
        # True if dt's date equals Easter of its year (and midnight when normalize is
        # set). Imports dateutil lazily.
        if self.normalize and not _is_normalized(dt):
            return False

        from dateutil.easter import easter

        return date(dt.year, dt.month, dt.day) == easter(dt.year, method=self.method)


# ----------------------------------------------------------------------
# Custom Offset classes


cdef class CustomBusinessDay(BusinessDay):
    # [CLASS DOC]
    # Business day offset with a custom weekmask, holidays and/or calendar; uses
    # numpy busday_offset. Alias "C". Not supported as a Period frequency, and
    # has no vectorized implementation (_apply_array is the base one that raises).
    """
    DateOffset subclass representing possibly n custom business days.

    In CustomBusinessDay we can use custom weekmask, holidays, and calendar.

    Parameters
    ----------
    n : int, default 1
        The number of days represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    weekmask : str, Default 'Mon Tue Wed Thu Fri'
        Weekmask of valid business days, passed to ``numpy.busdaycalendar``.
    holidays : list
        List/array of dates to exclude from the set of valid business days,
        passed to ``numpy.busdaycalendar``.
    calendar : np.busdaycalendar
        Calendar to integrate.
    offset : timedelta, default timedelta(0)
        Time offset to apply.

    Examples
    --------
    In the example below the default parameters give the next business day.

    >>> ts = pd.Timestamp(2022, 8, 5, 16)
    >>> ts + pd.offsets.CustomBusinessDay()
    Timestamp('2022-08-08 16:00:00')

    Business days can be specified by ``weekmask`` parameter. To convert
    the returned datetime object to its string representation
    the function strftime() is used in the next example.

    >>> import datetime as dt
    >>> freq = pd.offsets.CustomBusinessDay(weekmask="Mon Wed Fri")
    >>> pd.date_range(dt.datetime(2022, 12, 10), dt.datetime(2022, 12, 21),
    ...               freq=freq).strftime('%a %d %b %Y %H:%M')
    Index(['Mon 12 Dec 2022 00:00', 'Wed 14 Dec 2022 00:00',
           'Fri 16 Dec 2022 00:00', 'Mon 19 Dec 2022 00:00',
           'Wed 21 Dec 2022 00:00'],
           dtype='object')

    Using NumPy business day calendar you can define custom holidays.

    >>> import datetime as dt
    >>> bdc = np.busdaycalendar(holidays=['2022-12-12', '2022-12-14'])
    >>> freq = pd.offsets.CustomBusinessDay(calendar=bdc)
    >>> pd.date_range(dt.datetime(2022, 12, 10), dt.datetime(2022, 12, 25), freq=freq)
    DatetimeIndex(['2022-12-13', '2022-12-15', '2022-12-16', '2022-12-19',
                   '2022-12-20', '2022-12-21', '2022-12-22', '2022-12-23'],
                   dtype='datetime64[ns]', freq='C')

    If you want to shift the result on n day you can use the parameter ``offset``.

    >>> pd.Timestamp(2022, 8, 5, 16) + pd.offsets.CustomBusinessDay(1)
    Timestamp('2022-08-08 16:00:00')

    >>> import datetime as dt
    >>> ts = pd.Timestamp(2022, 8, 5, 16)
    >>> ts + pd.offsets.CustomBusinessDay(1, offset=dt.timedelta(days=1))
    Timestamp('2022-08-09 16:00:00')
    """

    _prefix = "C"
    _attributes = tuple(
        ["n", "normalize", "weekmask", "holidays", "calendar", "offset"]
    )

    @property
    def _period_dtype_code(self):
        # [FUNCTION DOC]
        # Property that always raises ValueError: custom business day is not a valid
        # Period frequency (GH#52534). Makes hasattr checks in to_offset fail safely.
        # GH#52534
        raise ValueError(f"{self.base} is not supported as period frequency")

    _apply_array = BaseOffset._apply_array

    def __init__(
        self,
        n=1,
        normalize=False,
        weekmask="Mon Tue Wed Thu Fri",
        holidays=None,
        calendar=None,
        offset=timedelta(0),
    ):
        # [FUNCTION DOC]
        # Initialize as BusinessDay then build weekmask/holidays/calendar.
        # Exceptions: numpy errors for invalid weekmask/holidays.
        BusinessDay.__init__(self, n, normalize, offset)
        self._init_custom(weekmask, holidays, calendar)

    cpdef __setstate__(self, state):
        # [FUNCTION DOC]
        # Unpickle: pop holidays and weekmask, then BusinessDay.__setstate__.
        # Side effects: mutates state.
        # KNOWN ISSUE: the busdaycalendar is not rebuilt (and BusinessDay.__setstate__
        # does not rebuild it), so _calendar stays None and holidays are not converted
        # to datetime64; only legacy pickles reach this (__reduce__ is used otherwise).
        self._holidays = state.pop("holidays")
        self._weekmask = state.pop("weekmask")
        BusinessDay.__setstate__(self, state)

    @apply_wraps
    def _apply(self, other):
        # [FUNCTION DOC]
        # Apply (wrapped by apply_wraps). For datetimes: numpy busday_offset on the
        # date with n steps (roll "forward" if n <= 0 else "backward") using the
        # calendar, recombined with the original time, then plus offset. For
        # timedelta scalars returns a new offset with offset increased by other.
        # Exceptions: ApplyTypeError for other types.
        # KNOWN ISSUE: the timedelta branch returns a plain BusinessDay (BDay),
        # silently dropping weekmask, holidays and calendar.
        if self._n <= 0:
            roll = "forward"
        else:
            roll = "backward"

        if PyDateTime_Check(other):
            date_in = other
            np_dt = np.datetime64(date_in.date())

            np_incr_dt = np.busday_offset(
                np_dt, self._n, roll=roll, busdaycal=self.calendar
            )

            dt_date = np_incr_dt.astype(datetime)
            result = datetime.combine(dt_date, date_in.time())

            if self.offset:
                result = result + self.offset
            return result

        elif is_any_td_scalar(other):
            td = Timedelta(self.offset) + other
            return BDay(self._n, offset=td.to_pytimedelta(), normalize=self.normalize)
        else:
            raise ApplyTypeError(
                "Only know how to combine trading day with "
                "datetime, datetime64 or timedelta."
            )

    def is_on_offset(self, dt: datetime) -> bool:
        # [FUNCTION DOC]
        # True if dt's date is a valid business day per the calendar (and midnight
        # when normalize is set).
        if self.normalize and not _is_normalized(dt):
            return False
        day64 = _to_dt64D(dt)
        return np.is_busday(day64, busdaycal=self.calendar)


cdef class CustomBusinessHour(BusinessHour):
    # [CLASS DOC]
    # Business hour offset whose business days come from a custom weekmask,
    # holidays and calendar (via CustomBusinessDay in next_bday). Alias "cbh".
    """
    DateOffset subclass representing possibly n custom business days.

    In CustomBusinessHour we can use custom weekmask, holidays, and calendar.

    Parameters
    ----------
    n : int, default 1
        The number of hours represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    weekmask : str, Default 'Mon Tue Wed Thu Fri'
        Weekmask of valid business days, passed to ``numpy.busdaycalendar``.
    holidays : list
        List/array of dates to exclude from the set of valid business days,
        passed to ``numpy.busdaycalendar``.
    calendar : np.busdaycalendar
        Calendar to integrate.
    start : str, time, or list of str/time, default "09:00"
        Start time of your custom business hour in 24h format.
    end : str, time, or list of str/time, default: "17:00"
        End time of your custom business hour in 24h format.
    offset : timedelta, default timedelta(0)
        Time offset to apply.

    Examples
    --------
    In the example below the default parameters give the next business hour.

    >>> ts = pd.Timestamp(2022, 8, 5, 16)
    >>> ts + pd.offsets.CustomBusinessHour()
    Timestamp('2022-08-08 09:00:00')

    We can also change the start and the end of business hours.

    >>> ts = pd.Timestamp(2022, 8, 5, 16)
    >>> ts + pd.offsets.CustomBusinessHour(start="11:00")
    Timestamp('2022-08-08 11:00:00')

    >>> from datetime import time as dt_time
    >>> ts = pd.Timestamp(2022, 8, 5, 16)
    >>> ts + pd.offsets.CustomBusinessHour(end=dt_time(19, 0))
    Timestamp('2022-08-05 17:00:00')

    >>> ts = pd.Timestamp(2022, 8, 5, 22)
    >>> ts + pd.offsets.CustomBusinessHour(end=dt_time(19, 0))
    Timestamp('2022-08-08 10:00:00')

    You can divide your business day hours into several parts.

    >>> import datetime as dt
    >>> freq = pd.offsets.CustomBusinessHour(start=["06:00", "10:00", "15:00"],
    ...                                      end=["08:00", "12:00", "17:00"])
    >>> pd.date_range(dt.datetime(2022, 12, 9), dt.datetime(2022, 12, 13), freq=freq)
    DatetimeIndex(['2022-12-09 06:00:00', '2022-12-09 07:00:00',
                   '2022-12-09 10:00:00', '2022-12-09 11:00:00',
                   '2022-12-09 15:00:00', '2022-12-09 16:00:00',
                   '2022-12-12 06:00:00', '2022-12-12 07:00:00',
                   '2022-12-12 10:00:00', '2022-12-12 11:00:00',
                   '2022-12-12 15:00:00', '2022-12-12 16:00:00'],
                   dtype='datetime64[ns]', freq='cbh')

    Business days can be specified by ``weekmask`` parameter. To convert
    the returned datetime object to its string representation
    the function strftime() is used in the next example.

    >>> import datetime as dt
    >>> freq = pd.offsets.CustomBusinessHour(weekmask="Mon Wed Fri",
    ...                                      start="10:00", end="13:00")
    >>> pd.date_range(dt.datetime(2022, 12, 10), dt.datetime(2022, 12, 18),
    ...               freq=freq).strftime('%a %d %b %Y %H:%M')
    Index(['Mon 12 Dec 2022 10:00', 'Mon 12 Dec 2022 11:00',
           'Mon 12 Dec 2022 12:00', 'Wed 14 Dec 2022 10:00',
           'Wed 14 Dec 2022 11:00', 'Wed 14 Dec 2022 12:00',
           'Fri 16 Dec 2022 10:00', 'Fri 16 Dec 2022 11:00',
           'Fri 16 Dec 2022 12:00'],
           dtype='object')

    Using NumPy business day calendar you can define custom holidays.

    >>> import datetime as dt
    >>> bdc = np.busdaycalendar(holidays=['2022-12-12', '2022-12-14'])
    >>> freq = pd.offsets.CustomBusinessHour(calendar=bdc, start="10:00", end="13:00")
    >>> pd.date_range(dt.datetime(2022, 12, 10), dt.datetime(2022, 12, 18), freq=freq)
    DatetimeIndex(['2022-12-13 10:00:00', '2022-12-13 11:00:00',
                   '2022-12-13 12:00:00', '2022-12-15 10:00:00',
                   '2022-12-15 11:00:00', '2022-12-15 12:00:00',
                   '2022-12-16 10:00:00', '2022-12-16 11:00:00',
                   '2022-12-16 12:00:00'],
                   dtype='datetime64[ns]', freq='cbh')
    """

    _prefix = "cbh"
    _anchor = 0
    _attributes = tuple(
        ["n", "normalize", "weekmask", "holidays", "calendar", "start", "end", "offset"]
    )

    def __init__(
        self,
        n=1,
        normalize=False,
        weekmask="Mon Tue Wed Thu Fri",
        holidays=None,
        calendar=None,
        start="09:00",
        end="17:00",
        offset=timedelta(0),
    ):
        # [FUNCTION DOC]
        # Initialize as BusinessHour then build weekmask/holidays/calendar.
        # Exceptions: as BusinessHour.__init__ plus numpy calendar errors.
        BusinessHour.__init__(self, n, normalize, start=start, end=end, offset=offset)
        self._init_custom(weekmask, holidays, calendar)


cdef class _CustomBusinessMonth(BusinessMixin):
    # [CLASS DOC]
    # Shared implementation for CustomBusinessMonthEnd/Begin: moves to the month
    # boundary (MonthEnd or MonthBegin, chosen by whether the prefix ends in "S")
    # and then rolls to the nearest valid custom business day inside the month.
    _attributes = tuple(
        ["n", "normalize", "weekmask", "holidays", "calendar", "offset"]
    )

    def __init__(
        self,
        n=1,
        normalize=False,
        weekmask="Mon Tue Wed Thu Fri",
        holidays=None,
        calendar=None,
        offset=timedelta(0),
    ):
        # [FUNCTION DOC]
        # Initialize as BusinessMixin then build weekmask/holidays/calendar.
        BusinessMixin.__init__(self, n, normalize, offset)
        self._init_custom(weekmask, holidays, calendar)

    @cache_readonly
    def cbday_roll(self):
        # [FUNCTION DOC]
        # Cached property: bound roll function of a CustomBusinessDay(n=1, same
        # weekmask/holidays/calendar, zero offset): rollforward for month-begin,
        # rollback for month-end.
        """
        Define default roll function to be called in apply method.
        """
        cbday_kwds = self.kwds.copy()
        cbday_kwds["offset"] = timedelta(0)

        cbday = CustomBusinessDay(n=1, normalize=False, **cbday_kwds)

        if self._prefix.endswith("S"):
            # MonthBegin
            roll_func = cbday.rollforward
        else:
            # MonthEnd
            roll_func = cbday.rollback
        return roll_func

    @cache_readonly
    def m_offset(self):
        # [FUNCTION DOC]
        # Cached property: MonthBegin(1) for the begin variant, MonthEnd(1) otherwise.
        """
        Return a MonthBegin or MonthEnd offset.

        This is used internally to compute the month-based component
        of the custom business month offset calculation.

        See Also
        --------
        tseries.offsets.MonthBegin : Represents the start of the month.
        tseries.offsets.MonthEnd : Represents the end of the month.

        Examples
        --------
        >>> pd.offsets.CustomBusinessMonthBegin().m_offset
        <MonthBegin>

        >>> pd.offsets.CustomBusinessMonthEnd().m_offset
        <MonthEnd>
        """
        if self._prefix.endswith("S"):
            # MonthBegin
            moff = MonthBegin(n=1, normalize=False)
        else:
            # MonthEnd
            moff = MonthEnd(n=1, normalize=False)
        return moff

    @cache_readonly
    def month_roll(self):
        # [FUNCTION DOC]
        # Cached property: bound roll function of m_offset: rollback for begin,
        # rollforward for end (moves a date to its own month's boundary).
        """
        Define default roll function to be called in apply method.
        """
        if self._prefix.endswith("S"):
            # MonthBegin
            roll_func = self.m_offset.rollback
        else:
            # MonthEnd
            roll_func = self.m_offset.rollforward
        return roll_func

    @apply_wraps
    def _apply(self, other: datetime) -> datetime:
        # [FUNCTION DOC]
        # Apply (wrapped by apply_wraps): roll other to its month boundary, find that
        # month's custom business anchor, adjust n with roll_convention comparing
        # other.day with the anchor day, move n months, roll to a valid business day,
        # and add offset.
        # First move to month offset
        cur_month_offset_date = self.month_roll(other)

        # Find this custom month offset
        compare_date = self.cbday_roll(cur_month_offset_date)
        n = roll_convention(other.day, self._n, compare_date.day)

        new = cur_month_offset_date + n * self.m_offset
        result = self.cbday_roll(new)

        if self.offset:
            result = result + self.offset
        return result


cdef class CustomBusinessMonthEnd(_CustomBusinessMonth):
    # [CLASS DOC]
    # Last custom business day of each month. Alias "CBME".
    """
    DateOffset subclass representing custom business month(s).

    Increments between end of month dates.

    Parameters
    ----------
    n : int, default 1
        The number of months represented.
    normalize : bool, default False
        Normalize end dates to midnight before generating date range.
    weekmask : str, Default 'Mon Tue Wed Thu Fri'
        Weekmask of valid business days, passed to ``numpy.busdaycalendar``.
    holidays : list
        List/array of dates to exclude from the set of valid business days,
        passed to ``numpy.busdaycalendar``.
    calendar : np.busdaycalendar
        Calendar to integrate.
    offset : timedelta, default timedelta(0)
        Time offset to apply.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    In the example below we use the default parameters.

    >>> ts = pd.Timestamp(2022, 8, 5)
    >>> ts + pd.offsets.CustomBusinessMonthEnd()
    Timestamp('2022-08-31 00:00:00')

    Custom business month end can be specified by ``weekmask`` parameter.
    To convert the returned datetime object to its string representation
    the function strftime() is used in the next example.

    >>> import datetime as dt
    >>> freq = pd.offsets.CustomBusinessMonthEnd(weekmask="Wed Thu")
    >>> pd.date_range(dt.datetime(2022, 7, 10), dt.datetime(2022, 12, 18),
    ...               freq=freq).strftime('%a %d %b %Y %H:%M')
    Index(['Thu 28 Jul 2022 00:00', 'Wed 31 Aug 2022 00:00',
           'Thu 29 Sep 2022 00:00', 'Thu 27 Oct 2022 00:00',
           'Wed 30 Nov 2022 00:00'],
           dtype='object')

    Using NumPy business day calendar you can define custom holidays.

    >>> import datetime as dt
    >>> bdc = np.busdaycalendar(holidays=['2022-08-01', '2022-09-30',
    ...                                   '2022-10-31', '2022-11-01'])
    >>> freq = pd.offsets.CustomBusinessMonthEnd(calendar=bdc)
    >>> pd.date_range(dt.datetime(2022, 7, 10), dt.datetime(2022, 11, 10), freq=freq)
    DatetimeIndex(['2022-07-29', '2022-08-31', '2022-09-29', '2022-10-28'],
                   dtype='datetime64[ns]', freq='CBME')
    """

    _prefix = "CBME"


cdef class CustomBusinessMonthBegin(_CustomBusinessMonth):
    # [CLASS DOC]
    # First custom business day of each month. Alias "CBMS".
    """
    DateOffset subclass representing custom business month(s).

    Increments between beginning of month dates.

    Parameters
    ----------
    n : int, default 1
        The number of months represented.
    normalize : bool, default False
        Normalize start dates to midnight before generating date range.
    weekmask : str, Default 'Mon Tue Wed Thu Fri'
        Weekmask of valid business days, passed to ``numpy.busdaycalendar``.
    holidays : list
        List/array of dates to exclude from the set of valid business days,
        passed to ``numpy.busdaycalendar``.
    calendar : np.busdaycalendar
        Calendar to integrate.
    offset : timedelta, default timedelta(0)
        Time offset to apply.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    In the example below we use the default parameters.

    >>> ts = pd.Timestamp(2022, 8, 5)
    >>> ts + pd.offsets.CustomBusinessMonthBegin()
    Timestamp('2022-09-01 00:00:00')

    Custom business month start can be specified by ``weekmask`` parameter.
    To convert the returned datetime object to its string representation
    the function strftime() is used in the next example.

    >>> import datetime as dt
    >>> freq = pd.offsets.CustomBusinessMonthBegin(weekmask="Wed Thu")
    >>> pd.date_range(dt.datetime(2022, 7, 10), dt.datetime(2022, 12, 18),
    ...               freq=freq).strftime('%a %d %b %Y %H:%M')
    Index(['Wed 03 Aug 2022 00:00', 'Thu 01 Sep 2022 00:00',
           'Wed 05 Oct 2022 00:00', 'Wed 02 Nov 2022 00:00',
           'Thu 01 Dec 2022 00:00'],
           dtype='object')

    Using NumPy business day calendar you can define custom holidays.

    >>> import datetime as dt
    >>> bdc = np.busdaycalendar(holidays=['2022-08-01', '2022-09-30',
    ...                                   '2022-10-31', '2022-11-01'])
    >>> freq = pd.offsets.CustomBusinessMonthBegin(calendar=bdc)
    >>> pd.date_range(dt.datetime(2022, 7, 10), dt.datetime(2022, 11, 10), freq=freq)
    DatetimeIndex(['2022-08-02', '2022-09-01', '2022-10-03', '2022-11-02'],
                   dtype='datetime64[ns]', freq='CBMS')
    """

    _prefix = "CBMS"


BDay = BusinessDay
BMonthEnd = BusinessMonthEnd
BMonthBegin = BusinessMonthBegin
CBMonthEnd = CustomBusinessMonthEnd
CBMonthBegin = CustomBusinessMonthBegin
CDay = CustomBusinessDay

# ----------------------------------------------------------------------
# to_offset helpers

prefix_mapping = {
    offset._prefix: offset
    for offset in [
        YearBegin,  # 'YS'
        YearEnd,  # 'YE'
        BYearBegin,  # 'BYS'
        BYearEnd,  # 'BYE'
        BusinessDay,  # 'B'
        BusinessMonthBegin,  # 'BMS'
        BusinessMonthEnd,  # 'BME'
        BQuarterEnd,  # 'BQE'
        BQuarterBegin,  # 'BQS'
        BHalfYearEnd,  # 'BHYE'
        BHalfYearBegin,  # 'BHYS'
        BusinessHour,  # 'bh'
        CustomBusinessDay,  # 'C'
        CustomBusinessMonthEnd,  # 'CBME'
        CustomBusinessMonthBegin,  # 'CBMS'
        CustomBusinessHour,  # 'cbh'
        MonthEnd,  # 'ME'
        MonthBegin,  # 'MS'
        Nano,  # 'ns'
        SemiMonthEnd,  # 'SME'
        SemiMonthBegin,  # 'SMS'
        Week,  # 'W'
        Second,  # 's'
        Minute,  # 'min'
        Micro,  # 'us'
        QuarterEnd,  # 'QE'
        QuarterBegin,  # 'QS'
        HalfYearEnd,  # 'HYE'
        HalfYearBegin,  # 'HYS'
        Milli,  # 'ms'
        Hour,  # 'h'
        Day,  # 'D'
        WeekOfMonth,  # 'WOM'
        LastWeekOfMonth,  # 'LWOM'
        FY5253,
        FY5253Quarter,
    ]
}

# hack to handle WOM-1MON
opattern = re.compile(
    r"([+\-]?\d*|[+\-]?\d*\.\d*)\s*([A-Za-z]+([\-][\dA-Za-z\-]+)?)"
)

_lite_rule_alias = {
    "W": "W-SUN",
    "QE": "QE-DEC",

    "YE": "YE-DEC",      # YearEnd(month=12),
    "YS": "YS-JAN",    # YearBegin(month=1),
    "BYE": "BYE-DEC",    # BYearEnd(month=12),
    "BYS": "BYS-JAN",  # BYearBegin(month=1),

    "Min": "min",
}

_dont_uppercase = {"min", "h", "bh", "cbh", "s", "ms", "us", "ns"}


INVALID_FREQ_ERR_MSG = "Invalid frequency: {0}"

# TODO: still needed?
# cache of previously seen offsets
_offset_map = {}


deprec_to_valid_alias = {
    "H": "h",
    "BH": "bh",
    "CBH": "cbh",
    "T": "min",
    "S": "s",
    "L": "ms",
    "U": "us",
    "N": "ns",
    "AS": "YS",
    "AS-JAN": "YS-JAN",
    "AS-FEB": "YS-FEB",
    "AS-MAR": "YS-MAR",
    "AS-APR": "YS-APR",
    "AS-MAY": "YS-MAY",
    "AS-JUN": "YS-JUN",
    "AS-JUL": "YS-JUL",
    "AS-AUG": "YS-AUG",
    "AS-SEP": "YS-SEP",
    "AS-OCT": "YS-OCT",
    "AS-NOV": "YS-NOV",
    "AS-DEC": "YS-DEC",
    "A": "Y",
    "A-JAN": "Y-JAN",
    "A-FEB": "Y-FEB",
    "A-MAR": "Y-MAR",
    "A-APR": "Y-APR",
    "A-MAY": "Y-MAY",
    "A-JUN": "Y-JUN",
    "A-JUL": "Y-JUL",
    "A-AUG": "Y-AUG",
    "A-SEP": "Y-SEP",
    "A-OCT": "Y-OCT",
    "A-NOV": "Y-NOV",
    "A-DEC": "Y-DEC",
}


def raise_invalid_freq(freq: str, extra_message: str | None = None) -> None:
    # [FUNCTION DOC]
    # Raise ValueError "Invalid frequency: {freq}." with an optional extra message
    # and a "Did you mean ...?" hint for known deprecated aliases.
    # Inputs: freq - the alias; extra_message - str or None.
    # Outputs: never returns. Exceptions: always ValueError.
    msg = f"Invalid frequency: {freq}."
    if extra_message is not None:
        msg += f" {extra_message}"
    if freq in deprec_to_valid_alias:
        msg += f" Did you mean {deprec_to_valid_alias[freq]}?"
    raise ValueError(msg)


def _warn_about_deprecated_aliases(name: str, is_period: bool) -> str:
    # [FUNCTION DOC]
    # HELPER: Map deprecated frequency aliases to current ones with a warning.
    # Inputs: name - alias component; is_period - unused.
    # Outputs: name unchanged if in _lite_rule_alias; the replacement for names in
    # the deprecated mapping; the lower/upper-case variant if that variant is a
    # valid replacement value; otherwise name.
    # Side effects: emits Pandas4Warning when a replacement is made.
    if name in _lite_rule_alias:
        return name
    if name in c_PERIOD_AND_OFFSET_DEPR_FREQSTR:
        from pandas.errors import Pandas4Warning

        # https://github.com/pandas-dev/pandas/pull/59240
        warnings.warn(
            f"\'{name}\' is deprecated and will be removed "
            f"in a future version, please use "
            f"\'{c_PERIOD_AND_OFFSET_DEPR_FREQSTR.get(name)}\' "
            f"instead.",
            Pandas4Warning,
            stacklevel=find_stack_level(),
            )
        return c_PERIOD_AND_OFFSET_DEPR_FREQSTR[name]

    for _name in (name.lower(), name.upper()):
        if name == _name:
            continue
        if _name in c_PERIOD_AND_OFFSET_DEPR_FREQSTR.values():
            from pandas.errors import Pandas4Warning

            # https://github.com/pandas-dev/pandas/pull/59240
            warnings.warn(
                f"\'{name}\' is deprecated and will be removed "
                f"in a future version, please use "
                f"\'{_name}\' "
                f"instead.",
                Pandas4Warning,
                stacklevel=find_stack_level(),
                )
            return _name

    return name


def _validate_to_offset_alias(alias: str, is_period: bool) -> None:
    # [FUNCTION DOC]
    # HELPER: Reject aliases invalid in the requested context.
    # Inputs: alias - alias string; is_period - whether parsing a Period freq.
    # For offsets: ValueError if alias was renamed (e.g. old "M"), or if it is not
    # all upper-case and (excluding s/ms/us/ns) its prefix ends in S or E. For
    # periods: ValueError if the alias has a different Period spelling.
    # Outputs: None.
    if not is_period:
        if alias.upper() in c_OFFSET_RENAMED_FREQSTR:
            raise ValueError(
                f"\'{alias}\' is no longer supported for offsets. Please "
                f"use \'{c_OFFSET_RENAMED_FREQSTR.get(alias.upper())}\' "
                f"instead."
            )
        if (alias.upper() != alias and
                alias.lower() not in {"s", "ms", "us", "ns"} and
                alias.upper().split("-")[0].endswith(("S", "E"))):
            raise ValueError(raise_invalid_freq(freq=alias))
    if (
        is_period and
        alias in c_OFFSET_TO_PERIOD_FREQSTR and
        alias != c_OFFSET_TO_PERIOD_FREQSTR[alias]
    ):
        alias_msg = c_OFFSET_TO_PERIOD_FREQSTR.get(alias)
        raise ValueError(
            f"for Period, please use \'{alias_msg}\' "
            f"instead of \'{alias}\'"
        )


# TODO: better name?
def _get_offset(name: str) -> BaseOffset:
    # [FUNCTION DOC]
    # HELPER: Construct (and cache) an offset from a single alias like "QE-DEC" or
    # "WOM-1MON": the prefix before the first "-" selects the class from
    # prefix_mapping and remaining parts go to its _from_name.
    # Outputs: offset with n=1. Exceptions: ValueError via raise_invalid_freq on
    # bad prefix/suffix. Side effects: inserts into module cache _offset_map.
    """
    Return DateOffset object associated with rule name.

    Examples
    --------
    _get_offset('EOM') --> BMonthEnd(1)
    """
    if name not in _offset_map:
        try:
            split = name.split("-")
            klass = prefix_mapping[split[0]]
            # handles case where there's no suffix (and will TypeError if too
            # many '-')
            offset = klass._from_name(*split[1:])
        except (ValueError, TypeError, KeyError) as err:
            # bad prefix or suffix
            raise_invalid_freq(
                freq=name,
                extra_message=f"Failed to parse with error message: {repr(err)}."
            )
        # cache
        _offset_map[name] = offset

    return _offset_map[name]


cpdef to_offset(freq, bint is_period=False):
    # [FUNCTION DOC]
    # Convert freq to a DateOffset.
    # Inputs: freq - None, BaseOffset, timedelta, or string such as "2h30min",
    # "-1D", "QE-DEC", "WOM-1MON"; is_period - validate as a Period frequency.
    # Outputs: None for None; offsets returned as-is; timedeltas via
    # delta_to_tick; strings parsed with opattern into (stride, name) parts: each
    # name is de-deprecated, validated, mapped (period aliases, _lite_rule_alias);
    # fixed units (D,h,min,s,ms,us,ns) build a Tick from float stride (so "1.5h"
    # works), others use _get_offset times integer stride; the first part's sign
    # applies to all parts; parts are summed. A result Hour divisible by 24 from a
    # string containing "d"/"D" is converted to Day.
    # Exceptions: TypeError for tuples; ValueError for unparseable strings, other
    # types, or (is_period) offsets lacking a period dtype code.
    # Side effects: may emit Pandas4Warning; populates _offset_map.
    """
    Return DateOffset object from string or datetime.timedelta object.

    Parameters
    ----------
    freq : str, datetime.timedelta, BaseOffset or None
        The frequency represented.
    is_period : bool, default False
        Convert string denoting period frequency to corresponding offsets
        frequency if is_period=True.

    Returns
    -------
    BaseOffset subclass or None

    Raises
    ------
    ValueError
        If freq is an invalid frequency

    See Also
    --------
    BaseOffset : Standard kind of date increment used for a date range.

    Examples
    --------
    >>> from pandas.tseries.frequencies import to_offset
    >>> to_offset("5min")
    <5 * Minutes>

    >>> to_offset("1D1h")
    <25 * Hours>

    >>> to_offset("2W")
    <2 * Weeks: weekday=6>

    >>> to_offset("2B")
    <2 * BusinessDays>

    >>> to_offset(pd.Timedelta(days=1))
    <Day>

    >>> to_offset(pd.offsets.Hour())
    <Hour>

    Passing the parameter ``is_period`` equal to True, you can use a string
    denoting period frequency:

    >>> freq = to_offset(freq="ME", is_period=False)
    >>> freq.rule_code
    'ME'

    >>> freq = to_offset(freq="M", is_period=True)
    >>> freq.rule_code
    'ME'
    """
    if freq is None:
        return None

    if isinstance(freq, tuple):
        raise TypeError(
            f"to_offset does not support tuples {freq}, pass as a string instead"
        )

    if isinstance(freq, BaseOffset):
        result = freq

    elif PyDelta_Check(freq):
        result = delta_to_tick(freq)

    elif isinstance(freq, str):
        result = None
        stride_sign = None

        try:
            split = opattern.split(freq)
            if split[-1] != "" and not split[-1].isspace():
                # the last element must be blank
                raise ValueError("last element must be blank")

            tups = zip(split[0::4], split[1::4], split[2::4], strict=False)
            for n, (sep, stride, name) in enumerate(tups):
                name = _warn_about_deprecated_aliases(name, is_period)
                _validate_to_offset_alias(name, is_period)
                if is_period:
                    if name.upper() in c_PERIOD_TO_OFFSET_FREQSTR:
                        if name.upper() != name:
                            raise ValueError(
                                f"\'{name}\' is no longer supported, "
                                f"please use \'{name.upper()}\' instead.",
                            )
                        name = c_PERIOD_TO_OFFSET_FREQSTR[name.upper()]
                name = _lite_rule_alias.get(name, name)

                if sep != "" and not sep.isspace():
                    raise ValueError("separator must be spaces")
                if stride_sign is None:
                    stride_sign = -1 if stride.startswith("-") else 1
                if not stride:
                    stride = 1

                if name in {"D", "h", "min", "s", "ms", "us", "ns"}:
                    # For these prefixes, we have something like "3h" or
                    #  "2.5min", so we can construct a Timedelta with the
                    #  matching unit and get our offset from delta_to_tick
                    td = Timedelta(1, unit=name)
                    off = delta_to_tick(td)
                    offset = off * float(stride)
                    if n != 0:
                        # If n==0, then stride_sign is already incorporated
                        #  into the offset
                        offset *= stride_sign
                else:
                    stride = int(stride)
                    offset = _get_offset(name)
                    offset = offset * int(np.fabs(stride) * stride_sign)

                if result is None:
                    result = offset
                else:
                    result = result + offset
        except (ValueError, TypeError) as err:
            raise_invalid_freq(
                freq=freq,
                extra_message=f"Failed to parse with error message: {repr(err)}"
            )

        # TODO(3.0?) once deprecation of "d" is enforced, the check for it here
        #  can be removed
        if (
                isinstance(result, Hour)
                and result.n % 24 == 0
                and ("d" in freq or "D" in freq)
        ):
            # Since Day is no longer a Tick, delta_to_tick returns Hour above,
            #  so we convert back here.
            result = Day(result.n // 24)
    else:
        result = None

    if result is None:
        raise_invalid_freq(freq=freq)

    try:
        has_period_dtype_code = hasattr(result, "_period_dtype_code")
    except ValueError:
        has_period_dtype_code = False

    if is_period and not has_period_dtype_code:
        if isinstance(freq, str):
            raise ValueError(f"{result.name} is not supported as period frequency")
        else:
            raise ValueError(f"{freq} is not supported as period frequency")

    return result


# ----------------------------------------------------------------------
# RelativeDelta Arithmetic

cdef datetime _shift_day(datetime other, int days):
    # [FUNCTION DOC]
    # HELPER: Add days calendar days to other; tz-aware inputs are shifted in wall
    # time and re-localized. Inputs: other - datetime; days - int.
    # Outputs: datetime (or Timestamp). Exceptions: localization errors.
    """
    Increment the datetime `other` by the given number of days, retaining
    the time-portion of the datetime.  For tz-naive datetimes this is
    equivalent to adding a timedelta.  For tz-aware datetimes it is similar to
    dateutil's relativedelta.__add__, but handles pytz tzinfo objects.

    Parameters
    ----------
    other : datetime or Timestamp
    days : int

    Returns
    -------
    shifted: datetime or Timestamp
    """
    if other.tzinfo is None:
        return other + timedelta(days=days)

    tz = other.tzinfo
    naive = other.replace(tzinfo=None)
    shifted = naive + timedelta(days=days)
    return localize_pydatetime(shifted, tz)


cdef int year_add_months(npy_datetimestruct dts, int months) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Year after adding months to dts's month (floor division handles
    # negative months). Inputs: dts - date struct by value; months - int.
    # Outputs: int year. No exceptions; GIL-free.
    """
    New year number after shifting npy_datetimestruct number of months.
    """
    return dts.year + (dts.month + months - 1) // 12


cdef int month_add_months(npy_datetimestruct dts, int months) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Month (1..12) after adding months to dts's month.
    # Inputs: dts - date struct; months - int. Outputs: int. GIL-free.
    """
    New month number after shifting npy_datetimestruct
    number of months.
    """
    cdef:
        int new_month = (dts.month + months) % 12
    return 12 if new_month == 0 else new_month


@cython.wraparound(False)
@cython.boundscheck(False)
cdef ndarray shift_quarters(
    ndarray dtindex,
    int quarters,
    int q1start_month,
    str day_opt,
    int modby=3,
    NPY_DATETIMEUNIT reso=NPY_DATETIMEUNIT.NPY_FR_ns,
):
    # [FUNCTION DOC]
    # Vectorized quarter/half-year/year shift of int64 epoch values.
    # Inputs: dtindex - int64 ndarray (any shape) in unit reso; quarters - number
    # of periods; q1start_month - anchor month; day_opt - start/end/business_start/
    # business_end; modby - period length in months (3, 6 or 12); reso - unit.
    # For each non-NaT value: months_since = (month - q1start_month) mod modby; n
    # adjusted by _roll_qtrday; shift by modby*n - months_since months; day set by
    # day_opt; time of day preserved. Outputs: new int64 ndarray (NaT kept).
    # Exceptions: ValueError for invalid day_opt.
    # Side effects: allocates output; releases the GIL in the main loop.
    # KNOWN ISSUE: the error message says None is allowed for day_opt, but None
    # is rejected.
    """
    Given an int64 array representing nanosecond timestamps, shift all elements
    by the specified number of quarters using DateOffset semantics.

    Parameters
    ----------
    dtindex : int64_t[:] timestamps for input dates
    quarters : int number of quarters to shift
    q1start_month : int month in which Q1 begins by convention
    day_opt : {'start', 'end', 'business_start', 'business_end'}
    modby : int (3 for quarters, 12 for years)
    reso : NPY_DATETIMEUNIT, default NPY_FR_ns

    Returns
    -------
    out : ndarray[int64_t]
    """
    if day_opt not in ["start", "end", "business_start", "business_end"]:
        raise ValueError("day must be None, 'start', 'end', "
                         "'business_start', or 'business_end'")

    cdef:
        Py_ssize_t count = dtindex.size
        ndarray out = cnp.PyArray_EMPTY(dtindex.ndim, dtindex.shape, cnp.NPY_INT64, 0)
        Py_ssize_t i
        int64_t val, res_val
        int months_since, n
        npy_datetimestruct dts
        cnp.broadcast mi = cnp.PyArray_MultiIterNew2(out, dtindex)
        _DayOpt day_opt_enum = _str_to_day_opt(day_opt)

    with nogil:
        for i in range(count):
            # Analogous to: val = dtindex[i]
            val = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

            if val == NPY_NAT:
                res_val = NPY_NAT
            else:
                pandas_datetime_to_datetimestruct(val, reso, &dts)
                n = quarters

                months_since = (dts.month - q1start_month) % modby
                n = _roll_qtrday(&dts, n, months_since, day_opt_enum)

                dts.year = year_add_months(dts, modby * n - months_since)
                dts.month = month_add_months(dts, modby * n - months_since)
                dts.day = get_day_of_month(&dts, day_opt_enum)

                res_val = npy_datetimestruct_to_datetime(reso, &dts)

            # Analogous to: out[i] = res_val
            (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = res_val

            cnp.PyArray_MultiIter_NEXT(mi)

    return out


@cython.wraparound(False)
@cython.boundscheck(False)
def shift_months(
    ndarray dtindex,  # int64_t, arbitrary ndim
    int months,
    str day_opt=None,
    NPY_DATETIMEUNIT reso=NPY_DATETIMEUNIT.NPY_FR_ns,
):
    # [FUNCTION DOC]
    # Vectorized month shift of int64 epoch values.
    # Inputs: dtindex - int64 ndarray (any shape); months - int; day_opt - None or
    # start/end/business_start/business_end; reso - unit.
    # Outputs: new int64 ndarray. With day_opt None: shift months keeping the day,
    # clipped to the month length. Otherwise: months adjusted via _roll_qtrday
    # (months_since 0) relative to the anchor day, then day set per day_opt.
    # NaT preserved; time of day preserved.
    # Exceptions: ValueError for invalid day_opt.
    # Side effects: allocates output; releases the GIL in the loops.
    """
    Given an int64-based datetime index, shift all elements
    specified number of months using DateOffset semantics

    day_opt: {None, 'start', 'end', 'business_start', 'business_end'}
       * None: day of month
       * 'start' 1st day of month
       * 'end' last day of month
    """
    cdef:
        Py_ssize_t i
        npy_datetimestruct dts
        int count = dtindex.size
        ndarray out = cnp.PyArray_EMPTY(dtindex.ndim, dtindex.shape, cnp.NPY_INT64, 0)
        int months_to_roll
        int64_t val, res_val
        _DayOpt day_opt_enum

        cnp.broadcast mi = cnp.PyArray_MultiIterNew2(out, dtindex)

    if day_opt is not None and day_opt not in {
            "start", "end", "business_start", "business_end"
    }:
        raise ValueError("day must be None, 'start', 'end', "
                         "'business_start', or 'business_end'")

    if day_opt is None:
        # TODO: can we combine this with the non-None case?
        with nogil:
            for i in range(count):
                # Analogous to: val = i8other[i]
                val = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

                if val == NPY_NAT:
                    res_val = NPY_NAT
                else:
                    pandas_datetime_to_datetimestruct(val, reso, &dts)
                    dts.year = year_add_months(dts, months)
                    dts.month = month_add_months(dts, months)

                    dts.day = min(dts.day, get_days_in_month(dts.year, dts.month))
                    res_val = npy_datetimestruct_to_datetime(reso, &dts)

                # Analogous to: out[i] = res_val
                (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = res_val

                cnp.PyArray_MultiIter_NEXT(mi)

    else:
        day_opt_enum = _str_to_day_opt(day_opt)
        with nogil:
            for i in range(count):

                # Analogous to: val = i8other[i]
                val = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

                if val == NPY_NAT:
                    res_val = NPY_NAT
                else:
                    pandas_datetime_to_datetimestruct(val, reso, &dts)
                    months_to_roll = months

                    months_to_roll = _roll_qtrday(
                        &dts, months_to_roll, 0, day_opt_enum
                    )

                    dts.year = year_add_months(dts, months_to_roll)
                    dts.month = month_add_months(dts, months_to_roll)
                    dts.day = get_day_of_month(&dts, day_opt_enum)

                    res_val = npy_datetimestruct_to_datetime(reso, &dts)

                # Analogous to: out[i] = res_val
                (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = res_val

                cnp.PyArray_MultiIter_NEXT(mi)

    return out


def shift_month(stamp: datetime, months: int, day_opt: object = None) -> datetime:
    # [FUNCTION DOC]
    # Scalar month shift: return stamp moved by months with day per day_opt (None:
    # keep day clipped to month length; start; end; business_start; business_end;
    # integer: that day clipped to month length).
    # Inputs: stamp - datetime or Timestamp (Timestamp uses its own resolution);
    # months - int; day_opt.
    # Outputs: stamp.replace(year, month, day), same type as input.
    # Exceptions: ValueError for an invalid day_opt.
    """
    Given a datetime (or Timestamp) `stamp`, an integer `months` and an
    option `day_opt`, return a new datetimelike that many months later,
    with day determined by `day_opt` using relativedelta semantics.

    Scalar analogue of shift_months.

    Parameters
    ----------
    stamp : datetime or Timestamp
    months : int
    day_opt : None, 'start', 'end', 'business_start', 'business_end', or int
        None: returned datetimelike has the same day as the input, or the
              last day of the month if the new month is too short
        'start': returned datetimelike has day=1
        'end': returned datetimelike has day on the last day of the month
        'business_start': returned datetimelike has day on the first
            business day of the month
        'business_end': returned datetimelike has day on the last
            business day of the month
        int: returned datetimelike has day equal to day_opt

    Returns
    -------
    shifted : datetime or Timestamp (same as input `stamp`)
    """
    cdef:
        int year, month, day
        int days_in_month, dy
        npy_datetimestruct dts

    if isinstance(stamp, _Timestamp):
        creso = (<_Timestamp>stamp)._creso
        val = (<_Timestamp>stamp)._value
        pandas_datetime_to_datetimestruct(val, creso, &dts)
    else:
        # Plain datetime/date
        pydate_to_dtstruct(stamp, &dts)

    dy = (dts.month + months) // 12
    month = (dts.month + months) % 12

    if month == 0:
        month = 12
        dy -= 1
    year = dts.year + dy

    if day_opt is None:
        days_in_month = get_days_in_month(year, month)
        day = min(dts.day, days_in_month)
    elif day_opt == "start":
        day = 1
    elif day_opt == "end":
        day = get_days_in_month(year, month)
    elif day_opt == "business_start":
        # first business day of month
        day = get_firstbday(year, month)
    elif day_opt == "business_end":
        # last business day of month
        day = get_lastbday(year, month)
    elif is_integer_object(day_opt):
        days_in_month = get_days_in_month(year, month)
        day = min(day_opt, days_in_month)
    else:
        raise ValueError(day_opt)
    return stamp.replace(year=year, month=month, day=day)


cdef int get_day_of_month(npy_datetimestruct* dts, _DayOpt day_opt) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Anchor day for dts's year/month per day_opt: 1, last day, first
    # weekday, or last weekday (anything else treated as business end).
    # Outputs: int. GIL-free, no exceptions.
    """
    Find the day in `other`'s month that satisfies a DateOffset's is_on_offset
    policy, as described by the `day_opt` argument.

    Parameters
    ----------
    dts : npy_datetimestruct*
    day_opt : _DayOpt enum value
        DAY_OPT_START: returns 1
        DAY_OPT_END: returns last day of the month
        DAY_OPT_BUSINESS_START: returns the first business day of the month
        DAY_OPT_BUSINESS_END: returns the last business day of the month

    Returns
    -------
    day_of_month : int

    Notes
    -----
    Caller is responsible for ensuring one of the four accepted day_opt values
    is passed.
    """

    if day_opt == DAY_OPT_START:
        return 1
    elif day_opt == DAY_OPT_END:
        return get_days_in_month(dts.year, dts.month)
    elif day_opt == DAY_OPT_BUSINESS_START:
        # first business day of month
        return get_firstbday(dts.year, dts.month)
    else:
        # i.e. day_opt == DAY_OPT_BUSINESS_END:
        # last business day of month
        return get_lastbday(dts.year, dts.month)


cpdef int roll_convention(int other, int n, int compare) noexcept nogil:
    # [FUNCTION DOC]
    # Adjust a period count for roll conventions.
    # Inputs: other - day (of current date); n - periods; compare - anchor day in
    # the same month. Outputs: n - 1 if n > 0 and other < compare (counts as the
    # first step); n + 1 if n <= 0 and other > compare; else n. GIL-free.
    """
    Possibly increment or decrement the number of periods to shift
    based on rollforward/rollbackward conventions.

    Parameters
    ----------
    other : int, generally the day component of a datetime
    n : number of periods to increment, before adjusting for rolling
    compare : int, generally the day component of a datetime, in the same
              month as the datetime form which `other` was taken.

    Returns
    -------
    n : int number of periods to increment
    """
    if n > 0 and other < compare:
        n -= 1
    elif n <= 0 and other > compare:
        # as if rolled forward already
        n += 1
    return n


def roll_qtrday(other: datetime, n: int, month: int,
                day_opt: str, modby: int) -> int:
    # [FUNCTION DOC]
    # Adjust a period count for quarter/year roll conventions.
    # Inputs: other - datetime; n - periods; month - anchor month; day_opt -
    # day rule; modby - 3, 6 or 12. months_since is other.month - month (modby 12)
    # or other.month % modby - month % modby; then delegates to _roll_qtrday.
    # Outputs: adjusted int. Exceptions: ValueError for invalid day_opt.
    """
    Possibly increment or decrement the number of periods to shift
    based on rollforward/rollbackward conventions.

    Parameters
    ----------
    other : datetime or Timestamp
    n : number of periods to increment, before adjusting for rolling
    month : int reference month giving the first month of the year
    day_opt : {'start', 'end', 'business_start', 'business_end'}
        The convention to use in finding the day in a given month against
        which to compare for rollforward/rollbackward decisions.
    modby : int 3 for quarters, 12 for years

    Returns
    -------
    n : int number of periods to increment

    See Also
    --------
    get_day_of_month : Find the day in a month provided an offset.
    """
    cdef:
        int months_since
        npy_datetimestruct dts

    if day_opt not in ["start", "end", "business_start", "business_end"]:
        raise ValueError(day_opt)

    pydate_to_dtstruct(other, &dts)

    if modby == 12:
        # We care about the month-of-year, not month-of-quarter, so skip mod
        months_since = other.month - month
    else:
        months_since = other.month % modby - month % modby

    return _roll_qtrday(&dts, n, months_since, _str_to_day_opt(day_opt))


cdef int _roll_qtrday(npy_datetimestruct* dts,
                      int n,
                      int months_since,
                      _DayOpt day_opt) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: For n > 0, decrement n if dts is in an earlier month of the period
    # (months_since < 0) or in the anchor month before the anchor day; for n <= 0,
    # increment n if later month (months_since > 0) or after the anchor day.
    # Inputs: dts - date struct pointer; n; months_since; day_opt enum.
    # Outputs: adjusted int. GIL-free.
    """
    See roll_qtrday.__doc__
    """

    if n > 0:
        if months_since < 0 or (months_since == 0 and
                                dts.day < get_day_of_month(dts, day_opt)):
            # pretend to roll back if on same month but
            # before compare_day
            n -= 1
    else:
        if months_since > 0 or (months_since == 0 and
                                dts.day > get_day_of_month(dts, day_opt)):
            # make sure to roll forward, so negate
            n += 1
    return n
