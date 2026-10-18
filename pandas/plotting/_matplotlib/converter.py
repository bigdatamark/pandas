# [MODULE DOC]
# Matplotlib unit converters, tick locators and tick formatters that teach
# matplotlib how to plot pandas date-like data (timestamps, periods, times of
# day, timedeltas).
# Public contents: converter classes for times, periods and datetimes;
# automatic date locator/formatter subclasses and a millisecond locator;
# frequency-aware tick finders for daily, monthly, quarterly and annual period
# data; period-axis locator and formatter classes and a timedelta formatter;
# registration functions (register and deregister) and a context manager and
# decorator that scope registration according to the plotting option for
# registering converters (true, false or auto).
# Relationships: used by the timeseries and core plotting modules and exposed
# through the public plotting package as register/deregister functions.
# Import-time side effects: none beyond imports; a module-level dictionary
# caches matplotlib converters that were replaced by registration, and the
# finder functions use a memoizing cache.
from __future__ import annotations

import contextlib
import datetime as pydt
from datetime import (
    datetime,
    tzinfo,
)
import functools
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)
import warnings

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.units as munits
import numpy as np

from pandas._libs import lib
from pandas._libs.tslibs import (
    Timestamp,
    to_offset,
)
from pandas._libs.tslibs.dtypes import (
    FreqGroup,
    periods_per_day,
)
from pandas._typing import (
    F,
    npt,
)

from pandas.core.dtypes.common import (
    is_float,
    is_float_dtype,
    is_integer,
    is_integer_dtype,
    is_nested_list_like,
)

from pandas import (
    Index,
    Series,
    get_option,
)
import pandas.core.common as com
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
    Period,
    PeriodIndex,
    period_range,
)
import pandas.core.tools.datetimes as tools

if TYPE_CHECKING:
    from collections.abc import Generator

    from matplotlib.axis import Axis

    from pandas._libs.tslibs.offsets import BaseOffset
    from pandas._typing import TimeUnit


_mpl_units: dict = {}  # Cache for units overwritten by us


def get_pairs() -> list[tuple[type, type[mdates.DateConverter]]]:
    # [FUNCTION DOC]
    # Returns the list of (type, converter class) pairs that registration installs:
    # timestamp, period, standard datetime, standard date and numpy datetime types
    # use the datetime converter (period uses the period converter); standard time
    # uses the time converter.
    pairs = [
        (Timestamp, DatetimeConverter),
        (Period, PeriodConverter),
        (pydt.datetime, DatetimeConverter),
        (pydt.date, DatetimeConverter),
        (pydt.time, TimeConverter),
        (np.datetime64, DatetimeConverter),
    ]
    return pairs


def register_pandas_matplotlib_converters(func: F) -> F:
    # [FUNCTION DOC]
    # Decorator factory: wraps a function so that every call runs inside the
    # pandas converters context manager. Input: the function to wrap. Returns a
    # wrapper preserving the original metadata and returning the function's result.
    """
    Decorator applying pandas_converters.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: the wrapper created by the decorator; forwards all arguments inside
        # the converters context and returns the wrapped function's result.
        with pandas_converters():
            return func(*args, **kwargs)

    return cast(F, wrapper)


@contextlib.contextmanager
def pandas_converters() -> Generator[None]:
    # [FUNCTION DOC]
    # Context manager that scopes converter registration for a plot. Reads the
    # option for registering matplotlib converters. If it is true or auto, registers
    # the pandas converters on entry. On exit (even after an exception) deregisters
    # only when the option is auto. Yields nothing.
    """
    Context manager registering pandas' converters for a plot.

    See Also
    --------
    register_pandas_matplotlib_converters : Decorator that applies this.
    """
    value = get_option("plotting.matplotlib.register_converters")

    if value:
        # register for True or "auto"
        register()
    try:
        yield
    finally:
        if value == "auto":
            # only deregister for "auto"
            deregister()


def register() -> None:
    # [FUNCTION DOC]
    # Registers the pandas converters with matplotlib's global unit registry. For
    # each pair, if a different converter is already present for the type it is saved
    # in the module-level cache of replaced converters; then a fresh instance of the
    # pandas converter replaces it. Mutates global state; returns nothing.
    pairs = get_pairs()
    for type_, cls in pairs:
        # Cache previous converter if present
        if type_ in munits.registry and not isinstance(munits.registry[type_], cls):
            previous = munits.registry[type_]
            _mpl_units[type_] = previous
        # Replace with pandas converter
        munits.registry[type_] = cls()


def deregister() -> None:
    # [FUNCTION DOC]
    # Undoes registration. Removes registry entries whose converter is exactly one of
    # the pandas classes (no subclasses), then restores previously saved converters
    # from the cache unless they are pandas converters themselves (keeping the
    # operation idempotent). Mutates global state; returns nothing. Exposed publicly
    # under a different name by the plotting package.
    # Renamed in pandas.plotting.__init__
    for type_, cls in get_pairs():
        # We use type to catch our classes directly, no inheritance
        if type(munits.registry.get(type_)) is cls:
            munits.registry.pop(type_)

    # restore the old keys
    for unit, formatter in _mpl_units.items():
        if type(formatter) not in {DatetimeConverter, PeriodConverter, TimeConverter}:
            # make it idempotent by excluding ours.
            munits.registry[unit] = formatter


def _to_ordinalf(tm: pydt.time) -> float:
    # [FUNCTION DOC]
    # HELPER: converts a time of day to a float number of seconds since midnight
    # (hours, minutes, seconds plus microseconds as a fraction). Returns the float.
    tot_sec = tm.hour * 3600 + tm.minute * 60 + tm.second + tm.microsecond / 10**6
    return tot_sec


def time2num(d):
    # [FUNCTION DOC]
    # Converts a time-like value to seconds since midnight. A string is parsed as a
    # timestamp and its time part used; a standard time object is converted directly;
    # any other value is returned unchanged.
    if isinstance(d, str):
        parsed = Timestamp(d)
        return _to_ordinalf(parsed.time())
    if isinstance(d, pydt.time):
        return _to_ordinalf(d)
    return d


class TimeConverter(munits.ConversionInterface):
    # [CLASS DOC]
    # Matplotlib unit converter for time-of-day values (subclass of matplotlib's
    # conversion interface; all methods are static, no state). Converts times and
    # time strings to float seconds and provides an axis description with a time
    # formatter and an automatic locator.
    @staticmethod
    def convert(value, unit, axis):
        # [FUNCTION DOC]
        # Converts values for plotting. Strings, standard times, integers and floats
        # become a float via the time conversion helper; an index is mapped element
        # wise and returned as an index; lists, tuples and arrays become a list of
        # converted elements; anything else is returned unchanged.
        # KNOWN ISSUE: the index case is checked before the later sequence check, which also names the index type, so that mention is redundant (dead alternative).
        valid_types = (str, pydt.time)
        if isinstance(value, valid_types) or is_integer(value) or is_float(value):
            return time2num(value)
        if isinstance(value, Index):
            return value.map(time2num)
        if isinstance(value, (list, tuple, np.ndarray, Index)):
            return [time2num(x) for x in value]
        return value

    @staticmethod
    def axisinfo(unit, axis) -> munits.AxisInfo | None:
        # [FUNCTION DOC]
        # Returns axis info for the unit. If the unit is not the string time, returns
        # none. Otherwise returns an axis description with an automatic major locator,
        # a time formatter bound to that locator, and the label time.
        if unit != "time":
            return None

        majloc = mpl.ticker.AutoLocator()  # pyright: ignore[reportAttributeAccessIssue]
        majfmt = TimeFormatter(majloc)
        return munits.AxisInfo(majloc=majloc, majfmt=majfmt, label="time")

    @staticmethod
    def default_units(x, axis) -> str:
        # [FUNCTION DOC]
        # Returns the default unit for any data on this axis: the constant string time.
        return "time"


# time formatter
class TimeFormatter(mpl.ticker.Formatter):  # pyright: ignore[reportAttributeAccessIssue]
    # [CLASS DOC]
    # Tick formatter for time-of-day axes (subclass of matplotlib's formatter).
    # Holds tick locations given at construction. Formats float seconds since
    # midnight as time text.
    def __init__(self, locs) -> None:
        # [FUNCTION DOC]
        # Constructor. Input: locs, the tick locations, passed to the formatter's
        # location setter. Returns nothing.
        self.set_locs(locs)

    def __call__(self, x, pos: int | None = 0) -> str:
        # [FUNCTION DOC]
        # Formats a tick value x (float seconds since midnight; the position argument
        # is unused) as text. Hours wrap modulo one day. Output is hours:minutes by
        # default; adds seconds when seconds are non-zero; adds milliseconds when
        # non-zero; and full microsecond precision when microseconds are non-zero.
        """
        Return the time of day as a formatted string.

        Parameters
        ----------
        x : float
            The time of day specified as seconds since 00:00 (midnight),
            with up to microsecond precision.
        pos
            Unused

        Returns
        -------
        str
            A string in HH:MM:SS.mmmuuu format. Microseconds,
            milliseconds and seconds are only displayed if non-zero.
        """
        fmt = "%H:%M:%S.%f"
        s = int(x)
        msus = round((x - s) * 10**6)
        ms = msus // 1000
        us = msus % 1000
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        _, h = divmod(h, 24)
        if us != 0:
            return pydt.time(h, m, s, msus).strftime(fmt)
        elif ms != 0:
            return pydt.time(h, m, s, msus).strftime(fmt)[:-3]
        elif s != 0:
            return pydt.time(h, m, s).strftime("%H:%M:%S")

        return pydt.time(h, m).strftime("%H:%M")


# Period Conversion


class PeriodConverter(mdates.DateConverter):
    # [CLASS DOC]
    # Matplotlib converter for period-frequency axes (subclass of matplotlib's date
    # converter; static methods only, no state). Turns dates, strings, periods and
    # sequences of them into period ordinals at the axis frequency.
    @staticmethod
    def convert(values, unit, axis: Axis):
        # [FUNCTION DOC]
        # Entry point used by matplotlib (for example when limits are set). Inputs:
        # values to convert, unit (unused), axis which must carry a frequency attribute.
        # Raises a type error when the axis has no frequency. Converts the axis
        # frequency to a period offset and returns the result of the from-frequency
        # conversion.
        # Reached via e.g. `ax.set_xlim`

        # In tests as of 2025-09-24, unit is always None except for 3 tests
        #  that directly call this with unit="";
        #  axis is always specifically a matplotlib.axis.XAxis

        if not hasattr(axis, "freq"):
            raise TypeError("Axis must have `freq` set to convert to Periods")
        freq = to_offset(axis.freq, is_period=True)  # pyright: ignore[reportAttributeAccessIssue]
        return PeriodConverter.convert_from_freq(values, freq)

    @staticmethod
    def convert_from_freq(values, freq: BaseOffset):
        # [FUNCTION DOC]
        # Converts values to ordinals at the given offset frequency. Nested list-like
        # input is converted element by element into a list; otherwise the values are
        # converted as a single one-dimensional input. Returns the converted structure.
        if is_nested_list_like(values):
            values = [PeriodConverter._convert_1d(v, freq) for v in values]
        else:
            values = PeriodConverter._convert_1d(values, freq)
        return values

    @staticmethod
    def _convert_1d(values, freq: BaseOffset):
        # [FUNCTION DOC]
        # HELPER: converts one-dimensional input to period ordinals. Silences the
        # future warnings about business-day periods while working. Scalars (string,
        # datetime, period, date, numpy datetime, integer, float) convert via the date
        # value helper. A period index is converted to the frequency and its integer
        # values returned. Another index is mapped element wise. An array of period
        # objects (detected by type inference) is turned into a period index and its
        # integer values returned. Lists, tuples and arrays are converted element wise
        # into a list. Anything else is returned unchanged.
        valid_types = (str, datetime, Period, pydt.date, np.datetime64)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "Period with BDay freq is deprecated", category=FutureWarning
            )
            warnings.filterwarnings(
                "ignore", r"PeriodDtype\[B\] is deprecated", category=FutureWarning
            )
            if (
                isinstance(values, valid_types)
                or is_integer(values)
                or is_float(values)
            ):
                return _get_datevalue(values, freq)
            elif isinstance(values, PeriodIndex):
                return values.asfreq(freq).asi8
            elif isinstance(values, Index):
                return values.map(lambda x: _get_datevalue(x, freq))
            elif lib.infer_dtype(values, skipna=False) == "period":
                # https://github.com/pandas-dev/pandas/issues/24304
                # convert ndarray[period] -> PeriodIndex
                return PeriodIndex(values, freq=freq).asi8
            elif isinstance(values, (list, tuple, np.ndarray)):
                return [_get_datevalue(x, freq) for x in values]
        return values


def _get_datevalue(date, freq: BaseOffset):
        # [FUNCTION DOC]
        # HELPER: converts a single date-like value to an ordinal at the frequency: a
        # period is re-expressed at the frequency; strings, datetimes, dates and numpy
        # datetimes are parsed as periods; numbers are returned unchanged; none stays
        # none. Raises a value error for any other type.
    if isinstance(date, Period):
        return date.asfreq(freq).ordinal
    elif isinstance(date, (str, datetime, pydt.date, np.datetime64)):
        return Period(date, freq).ordinal  # pyright: ignore[reportAttributeAccessIssue]
    elif is_integer(date) or is_float(date):
        return date
    elif date is None:
        return None
    raise ValueError(f"Unrecognizable date '{date}'")


# Datetime Conversion
class DatetimeConverter(mdates.DateConverter):
    # [CLASS DOC]
    # Matplotlib converter for datetime-valued axes (subclass of matplotlib's date
    # converter; static methods only, no state). Converts date-like input into
    # matplotlib's float date numbers and describes the axis with the pandas
    # auto locator and formatter.
    @staticmethod
    def convert(values, unit, axis: Axis):
        # [FUNCTION DOC]
        # Entry point used by matplotlib. Inputs: values (a one-dimensional input or a
        # nested list-like), unit (timezone or none), axis. Nested input is converted
        # element by element into a list; otherwise via the one-dimensional helper.
        # Returns the converted values.
        # Reached via e.g. `ax.set_xlim`

        # In tests as of 2025-09-24, unit is always None except for 3 tests
        #  that directly call this with unit="";
        #  axis is always specifically a matplotlib.axis.XAxis

        # values might be a 1-d array, or a list-like of arrays.
        if is_nested_list_like(values):
            values = [DatetimeConverter._convert_1d(v, unit, axis) for v in values]
        else:
            values = DatetimeConverter._convert_1d(values, unit, axis)
        return values

    @staticmethod
    def _convert_1d(values, unit, axis):
        # [FUNCTION DOC]
        # HELPER: converts one-dimensional date-like input to matplotlib date numbers.
        # Datetimes, dates, numpy datetimes and times convert directly. Numbers pass
        # through. Strings are parsed (returning the input unchanged on any failure).
        # Lists, tuples, arrays, indexes and series are normalized to an array (a series
        # via an index to work around a matplotlib issue); numeric arrays pass through,
        # otherwise a datetime parse is attempted (failures ignored) followed by the
        # conversion to date numbers. Other input is returned unchanged.
        def try_parse(values):
            # [FUNCTION DOC]
            # HELPER: parses text or values to datetimes and converts them to matplotlib
            # date numbers, returning the input unchanged if any exception occurs.
            try:
                return mdates.date2num(tools.to_datetime(values))
            except Exception:
                return values

        if isinstance(values, (datetime, pydt.date, np.datetime64, pydt.time)):
            return mdates.date2num(values)
        elif is_integer(values) or is_float(values):
            return values
        elif isinstance(values, str):
            return try_parse(values)
        elif isinstance(values, (list, tuple, np.ndarray, Index, Series)):
            if isinstance(values, Series):
                # https://github.com/matplotlib/matplotlib/issues/11391
                # Series was skipped. Convert to DatetimeIndex to get asi8
                values = Index(values)
            if isinstance(values, Index):
                values = values.values
            if not isinstance(values, np.ndarray):
                values = com.asarray_tuplesafe(values)

            if is_integer_dtype(values) or is_float_dtype(values):
                return values

            try:
                values = tools.to_datetime(values)
            except Exception:
                pass

            values = mdates.date2num(values)

        return values

    @staticmethod
    def axisinfo(unit: tzinfo | None, axis) -> munits.AxisInfo:
        # [FUNCTION DOC]
        # Builds the axis description for a timezone unit (none allowed; the axis
        # argument is unused). Uses the pandas auto date locator and formatter bound to
        # that timezone, an empty label, and default limits from the first day of 2000
        # to the first day of 2010.
        """
        Return the :class:`~matplotlib.units.AxisInfo` for *unit*.

        *unit* is a tzinfo instance or None.
        The *axis* argument is required but not used.
        """
        tz = unit

        majloc = PandasAutoDateLocator(tz=tz)
        majfmt = PandasAutoDateFormatter(majloc, tz=tz)
        datemin = pydt.date(2000, 1, 1)
        datemax = pydt.date(2010, 1, 1)

        return munits.AxisInfo(
            majloc=majloc, majfmt=majfmt, label="", default_limits=(datemin, datemax)
        )


class PandasAutoDateFormatter(mdates.AutoDateFormatter):
    # [CLASS DOC]
    # Automatic date formatter (subclass of matplotlib's) whose only change is a
    # different constructor signature with default format year-month-day.
    def __init__(self, locator, tz=None, defaultfmt: str = "%Y-%m-%d") -> None:
        # [FUNCTION DOC]
        # Constructor. Inputs: locator, tz (default none) and defaultfmt (default
        # year-month-day text). Delegates to the parent initializer.
        mdates.AutoDateFormatter.__init__(self, locator, tz, defaultfmt)


class PandasAutoDateLocator(mdates.AutoDateLocator):
    # [CLASS DOC]
    # Automatic date locator (subclass of matplotlib's) that supports sub-second
    # ranges by switching to the millisecond locator.
    def get_locator(self, dmin, dmax):
        # [FUNCTION DOC]
        # Chooses a locator for the range from dmin to dmax. When the span in seconds
        # is smaller than the minimum tick count, resets the frequency marker to the
        # sub-second value, builds a millisecond locator on the same timezone and axis,
        # copies the view and data intervals onto it, and returns it. Otherwise
        # defers to the parent behavior.
        """Pick the best locator based on a distance."""
        tot_sec = (dmax - dmin).total_seconds()

        if abs(tot_sec) < self.minticks:
            self._freq = -1
            locator = MilliSecondLocator(self.tz)
            locator.set_axis(self.axis)

            # error: Item "None" of "Axis | _DummyAxis | _AxisWrapper | None"
            # has no attribute "get_data_interval"
            locator.axis.set_view_interval(  # type: ignore[union-attr]
                *self.axis.get_view_interval()  # type: ignore[union-attr]
            )
            locator.axis.set_data_interval(  # type: ignore[union-attr]
                *self.axis.get_data_interval()  # type: ignore[union-attr]
            )
            return locator

        return mdates.AutoDateLocator.get_locator(self, dmin, dmax)

    def _get_unit(self):
        # [FUNCTION DOC]
        # Returns the unit size in days for the current frequency marker using the
        # millisecond locator's generic unit function.
        return MilliSecondLocator.get_unit_generic(self._freq)


class MilliSecondLocator(mdates.DateLocator):
    # [CLASS DOC]
    # Date locator placing ticks at millisecond-scale intervals (subclass of
    # matplotlib's date locator). State: the current tick interval in milliseconds.
    # Class constant UNIT is one millisecond expressed in days.
    UNIT = 1.0 / (24 * 3600 * 1000)

    def __init__(self, tz) -> None:
        # [FUNCTION DOC]
        # Constructor. Input: tz. Initializes the parent and sets the interval to 1.
        mdates.DateLocator.__init__(self, tz)
        self._interval = 1.0

    def _get_unit(self):
        # [FUNCTION DOC]
        # Returns the unit size in days for the sub-second case.
        return self.get_unit_generic(-1)

    @staticmethod
    def get_unit_generic(freq):
        # [FUNCTION DOC]
        # Static: returns the matplotlib unit size for a frequency code, substituting
        # the one-millisecond unit when the generic value is negative.
        unit = mdates.RRuleLocator.get_unit_generic(freq)
        if unit < 0:
            return MilliSecondLocator.UNIT
        return unit

    def __call__(self):
        # [FUNCTION DOC]
        # Computes tick locations for the current view. Returns an empty list when no
        # view limits are set. Converts the limits to date numbers and the span to
        # milliseconds, then picks the smallest interval among 1, 10, 50, 100, 200, 500
        # whose limit (interval times five) covers the span, else 1000.
        # KNOWN ISSUE: the fallback assignment of 1000 sits inside the loop rather than after it, so it runs at each non-matching iteration; the final outcome is still the same as a loop-else but the comment is misleading.
        # KNOWN ISSUE: the too-many-ticks error message formats the float estimate with an integer format specifier, which would itself raise a value error instead of the intended runtime error.
        # Estimates tick count; raises a runtime error if above twice the maximum tick
        # count (see known issue below).
        # Builds millisecond date range between the limits (timezone name of the
        # locator) and returns its date numbers checked against the maximum tick limit;
        # if anything fails or the range is empty, falls back to the two limits.
        # if no data have been set, this will tank with a ValueError
        try:
            dmin, dmax = self.viewlim_to_dt()
        except ValueError:
            return []

        # We need to cap at the endpoints of valid datetime
        nmax, nmin = mdates.date2num((dmax, dmin))

        num = (nmax - nmin) * 86400 * 1000
        max_millis_ticks = 6
        for interval in [1, 10, 50, 100, 200, 500]:
            if num <= interval * (max_millis_ticks - 1):
                self._interval = interval
                break
            # We went through the whole loop without breaking, default to 1
            self._interval = 1000.0

        estimate = (nmax - nmin) / (self._get_unit() * self._get_interval())

        if estimate > self.MAXTICKS * 2:
            raise RuntimeError(
                "MillisecondLocator estimated to generate "
                f"{estimate:d} ticks from {dmin} to {dmax}: exceeds Locator.MAXTICKS"
                f"* 2 ({self.MAXTICKS * 2:d}) "
            )

        interval = self._get_interval()
        freq = f"{interval}ms"
        tz = self.tz.tzname(None)
        st = dmin.replace(tzinfo=None)
        ed = dmax.replace(tzinfo=None)
        all_dates = date_range(start=st, end=ed, freq=freq, tz=tz).astype(object)

        try:
            if len(all_dates) > 0:
                locs = self.raise_if_exceeds(mdates.date2num(all_dates))
                return locs
        except Exception:  # pragma: no cover
            pass

        lims = mdates.date2num([dmin, dmax])
        return lims

    def _get_interval(self):
        # [FUNCTION DOC]
        # Returns the current tick interval in milliseconds.
        return self._interval

    def autoscale(self):
        # [FUNCTION DOC]
        # Sets view limits to include the data range: converts the data limits to date
        # numbers and returns the non-singular expansion of them.
        """
        Set the view limits to include the data range.
        """
        # We need to cap at the endpoints of valid datetime
        dmin, dmax = self.datalim_to_dt()

        vmin = mdates.date2num(dmin)
        vmax = mdates.date2num(dmax)

        return self.nonsingular(vmin, vmax)


# Fixed frequency dynamic tick locators and formatters

# -------------------------------------------------------------------------
# --- Locators ---
# -------------------------------------------------------------------------


def _get_default_annual_spacing(nyears) -> tuple[int, int]:
    # [FUNCTION DOC]
    # HELPER: tick spacing for annual data. Input: number of years. Returns a pair
    # (minor spacing, major spacing) in years from thresholds: under 11 years 1 and
    # 1; under 20, 1 and 2; under 50, 1 and 5; under 100, 5 and 10; under 200, 5 and
    # 25; under 600, 10 and 50; otherwise 20 and 100 each scaled by one plus the
    # integer thousands of years.
    """
    Returns a default spacing between consecutive ticks for annual data.
    """
    if nyears < 11:
        (min_spacing, maj_spacing) = (1, 1)
    elif nyears < 20:
        (min_spacing, maj_spacing) = (1, 2)
    elif nyears < 50:
        (min_spacing, maj_spacing) = (1, 5)
    elif nyears < 100:
        (min_spacing, maj_spacing) = (5, 10)
    elif nyears < 200:
        (min_spacing, maj_spacing) = (5, 25)
    elif nyears < 600:
        (min_spacing, maj_spacing) = (10, 50)
    else:
        factor = nyears // 1000 + 1
        (min_spacing, maj_spacing) = (factor * 20, factor * 100)
    return (min_spacing, maj_spacing)


def _period_break(dates: PeriodIndex, period: str) -> npt.NDArray[np.intp]:
    # [FUNCTION DOC]
    # HELPER: returns the indices (integer array) at which the named calendar
    # component of the periods differs from that of the preceding period. Inputs:
    # dates (period index), period (attribute name such as year, month, day).
    """
    Returns the indices where the given period changes.

    Parameters
    ----------
    dates : PeriodIndex
        Array of intervals to monitor.
    period : str
        Name of the period to monitor.
    """
    mask = _period_break_mask(dates, period)
    return np.nonzero(mask)[0]


def _period_break_mask(dates: PeriodIndex, period: str) -> npt.NDArray[np.bool_]:
    # [FUNCTION DOC]
    # HELPER: boolean array, true where the named component of each period differs
    # from the component of the period one step earlier.
    current = getattr(dates, period)
    previous = getattr(dates - 1 * dates.freq, period)
    return current != previous


def has_level_label(label_flags: npt.NDArray[np.intp], vmin: float) -> bool:
    # [FUNCTION DOC]
    # HELPER: tells whether a level of labels exists. Inputs: label_flags (array of
    # positions) and vmin (view minimum). Returns false when the array is empty or
    # holds only position zero while the view minimum is not an integer (the first
    # label would not be shown); true otherwise.
    """
    Returns true if the ``label_flags`` indicate there is at least one label
    for this level.

    if the minimum view limit is not an exact integer, then the first tick
    label won't be shown, so we must adjust for that.
    """
    if label_flags.size == 0 or (
        label_flags.size == 1 and label_flags[0] == 0 and vmin % 1 > 0.0
    ):
        return False
    else:
        return True


def _get_periods_per_ymd(freq: BaseOffset) -> tuple[int, int, int]:
    # [FUNCTION DOC]
    # HELPER: returns approximate counts (periods per day, per month, per year) for
    # a frequency. Day count is minus one for coarse frequencies (day and above).
    # Hour or finer: day count from the resolution, month 28 days and year 365 days
    # of it; business day: 19 and 261; day: 28 and 365; week: 3 and 52; month: 1 and
    # 12; quarter: unknown (minus one) and 4; annual: unknown and 1. Raises a
    # not-implemented error for other frequencies.
    # error: "BaseOffset" has no attribute "_period_dtype_code"
    dtype_code = freq._period_dtype_code  # type: ignore[attr-defined]
    freq_group = FreqGroup.from_period_dtype_code(dtype_code)

    ppd = -1  # placeholder for above-day freqs

    if dtype_code >= FreqGroup.FR_HR.value:  # pyright: ignore[reportAttributeAccessIssue]
        # error: "BaseOffset" has no attribute "_creso"
        ppd = periods_per_day(freq._creso)  # type: ignore[attr-defined]
        ppm = 28 * ppd
        ppy = 365 * ppd
    elif freq_group == FreqGroup.FR_BUS:
        ppm = 19
        ppy = 261
    elif freq_group == FreqGroup.FR_DAY:
        ppm = 28
        ppy = 365
    elif freq_group == FreqGroup.FR_WK:
        ppm = 3
        ppy = 52
    elif freq_group == FreqGroup.FR_MTH:
        ppm = 1
        ppy = 12
    elif freq_group == FreqGroup.FR_QTR:
        ppm = -1  # placerholder
        ppy = 4
    elif freq_group == FreqGroup.FR_ANN:
        ppm = -1  # placeholder
        ppy = 1
    else:
        raise NotImplementedError(f"Unsupported frequency: {dtype_code}")

    return ppd, ppm, ppy


@functools.cache
def _daily_finder(vmin: float, vmax: float, freq: BaseOffset) -> np.ndarray:
    # [FUNCTION DOC]
    # HELPER: memoized tick finder for sub-monthly through multi-year period data
    # (used for daily, business, weekly and finer frequencies). Inputs: vmin and
    # vmax (view limits as ordinals) and freq. Builds the period range between the
    # limits and a record array with value, major flag, minor flag and format text
    # per period, with both end points major. Picks one of several layouts by span:
    # within a month, choosing second, minute, hour or day-level labels via the nested
    # finders by span thresholds relative to periods per day; up to a quarter year;
    # up to 1.15 years; up to 2.5 years; up to 4 years; up to 11 years; longer,
    # using annual spacing. It sets major/minor flags and strftime-style format
    # strings, adding a year label on the first visible label when needed.
    # Returns the record array.
    # error: "BaseOffset" has no attribute "_period_dtype_code"
    dtype_code = freq._period_dtype_code  # type: ignore[attr-defined]

    periodsperday, periodspermonth, periodsperyear = _get_periods_per_ymd(freq)

    # save this for later usage
    vmin_orig = vmin
    (vmin, vmax) = (int(vmin), int(vmax))
    span = vmax - vmin + 1

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "Period with BDay freq is deprecated", category=FutureWarning
        )
        warnings.filterwarnings(
            "ignore", r"PeriodDtype\[B\] is deprecated", category=FutureWarning
        )
        dates_ = period_range(
            start=Period(ordinal=vmin, freq=freq),
            end=Period(ordinal=vmax, freq=freq),
            freq=freq,
        )

    # Initialize the output
    info = np.zeros(
        span, dtype=[("val", np.int64), ("maj", bool), ("min", bool), ("fmt", "|S20")]
    )
    info["val"][:] = dates_.asi8
    info["fmt"][:] = ""
    info["maj"][[0, -1]] = True
    # .. and set some shortcuts
    info_maj = info["maj"]
    info_min = info["min"]
    info_fmt = info["fmt"]

    def first_label(label_flags):
        # [FUNCTION DOC]
        # HELPER: given label positions returns the first usable one, skipping the zeroth
        # when the view minimum is fractional and more positions exist.
        if (label_flags[0] == 0) and (label_flags.size > 1) and ((vmin_orig % 1) > 0.0):
            return label_flags[1]
        else:
            return label_flags[0]

    # Case 1. Less than a month
    if span <= periodspermonth:
        day_start = _period_break(dates_, "day")
        month_start = _period_break(dates_, "month")
        year_start = _period_break(dates_, "year")

        def _hour_finder(label_interval: int, force_year_start: bool) -> None:
            # [FUNCTION DOC]
            # HELPER: marks hour-level ticks. Inputs: label_interval (hours between minor
            # ticks) and force_year_start (add a year to the first day label when no year
            # boundary is visible). Sets day starts major, minor ticks at multiples of the
            # interval with hour:minute format, day starts with day-month format, year
            # starts including the year. Mutates the enclosing record array.
            target = dates_.hour
            mask = _period_break_mask(dates_, "hour")
            info_maj[day_start] = True
            info_min[mask & (target % label_interval == 0)] = True
            info_fmt[mask & (target % label_interval == 0)] = "%H:%M"
            info_fmt[day_start] = "%H:%M\n%d-%b"
            info_fmt[year_start] = "%H:%M\n%d-%b\n%Y"
            if force_year_start and not has_level_label(year_start, vmin_orig):
                info_fmt[first_label(day_start)] = "%H:%M\n%d-%b\n%Y"

        def _minute_finder(label_interval: int) -> None:
            # [FUNCTION DOC]
            # HELPER: marks minute-level ticks. Input: label_interval in minutes. Hour starts
            # become major; minor ticks at multiples of the interval with hour:minute
            # format; day and year starts get fuller formats. Mutates the enclosing arrays.
            target = dates_.minute
            hour_start = _period_break(dates_, "hour")
            mask = _period_break_mask(dates_, "minute")
            info_maj[hour_start] = True
            info_min[mask & (target % label_interval == 0)] = True
            info_fmt[mask & (target % label_interval == 0)] = "%H:%M"
            info_fmt[day_start] = "%H:%M\n%d-%b"
            info_fmt[year_start] = "%H:%M\n%d-%b\n%Y"

        def _second_finder(label_interval: int) -> None:
            # [FUNCTION DOC]
            # HELPER: marks second-level ticks. Input: label_interval in seconds. Minute
            # starts become major; minor ticks at multiples of the interval with
            # hour:minute:second format; day and year starts get fuller formats. Mutates the
            # enclosing arrays.
            target = dates_.second
            minute_start = _period_break(dates_, "minute")
            mask = _period_break_mask(dates_, "second")
            info_maj[minute_start] = True
            info_min[mask & (target % label_interval == 0)] = True
            info_fmt[mask & (target % label_interval == 0)] = "%H:%M:%S"
            info_fmt[day_start] = "%H:%M:%S\n%d-%b"
            info_fmt[year_start] = "%H:%M:%S\n%d-%b\n%Y"

        if span < periodsperday / 12000:
            _second_finder(1)
        elif span < periodsperday / 6000:
            _second_finder(2)
        elif span < periodsperday / 2400:
            _second_finder(5)
        elif span < periodsperday / 1200:
            _second_finder(10)
        elif span < periodsperday / 800:
            _second_finder(15)
        elif span < periodsperday / 400:
            _second_finder(30)
        elif span < periodsperday / 150:
            _minute_finder(1)
        elif span < periodsperday / 70:
            _minute_finder(2)
        elif span < periodsperday / 24:
            _minute_finder(5)
        elif span < periodsperday / 12:
            _minute_finder(15)
        elif span < periodsperday / 6:
            _minute_finder(30)
        elif span < periodsperday / 2.5:
            _hour_finder(1, False)
        elif span < periodsperday / 1.5:
            _hour_finder(2, False)
        elif span < periodsperday * 1.25:
            _hour_finder(3, False)
        elif span < periodsperday * 2.5:
            _hour_finder(6, True)
        elif span < periodsperday * 4:
            _hour_finder(12, True)
        else:
            info_maj[month_start] = True
            info_min[day_start] = True
            info_fmt[day_start] = "%d"
            info_fmt[month_start] = "%d\n%b"
            info_fmt[year_start] = "%d\n%b\n%Y"
            if not has_level_label(year_start, vmin_orig):
                if not has_level_label(month_start, vmin_orig):
                    info_fmt[first_label(day_start)] = "%d\n%b\n%Y"
                else:
                    info_fmt[first_label(month_start)] = "%d\n%b\n%Y"

    # Case 2. Less than three months
    elif span <= periodsperyear // 4:
        month_start = _period_break(dates_, "month")
        info_maj[month_start] = True
        if dtype_code < FreqGroup.FR_HR.value:  # pyright: ignore[reportAttributeAccessIssue]
            info["min"] = True
        else:
            day_start = _period_break(dates_, "day")
            info["min"][day_start] = True
        week_start = _period_break(dates_, "week")
        year_start = _period_break(dates_, "year")
        info_fmt[week_start] = "%d"
        info_fmt[month_start] = "\n\n%b"
        info_fmt[year_start] = "\n\n%b\n%Y"
        if not has_level_label(year_start, vmin_orig):
            if not has_level_label(month_start, vmin_orig):
                info_fmt[first_label(week_start)] = "\n\n%b\n%Y"
            else:
                info_fmt[first_label(month_start)] = "\n\n%b\n%Y"
    # Case 3. Less than 14 months ...............
    elif span <= 1.15 * periodsperyear:
        year_start = _period_break(dates_, "year")
        month_start = _period_break(dates_, "month")
        week_start = _period_break(dates_, "week")
        info_maj[month_start] = True
        info_min[week_start] = True
        info_min[year_start] = False
        info_min[month_start] = False
        info_fmt[month_start] = "%b"
        info_fmt[year_start] = "%b\n%Y"
        if not has_level_label(year_start, vmin_orig):
            info_fmt[first_label(month_start)] = "%b\n%Y"
    # Case 4. Less than 2.5 years ...............
    elif span <= 2.5 * periodsperyear:
        year_start = _period_break(dates_, "year")
        quarter_start = _period_break(dates_, "quarter")
        month_start = _period_break(dates_, "month")
        info_maj[quarter_start] = True
        info_min[month_start] = True
        info_fmt[quarter_start] = "%b"
        info_fmt[year_start] = "%b\n%Y"
    # Case 4. Less than 4 years .................
    elif span <= 4 * periodsperyear:
        year_start = _period_break(dates_, "year")
        month_start = _period_break(dates_, "month")
        info_maj[year_start] = True
        info_min[month_start] = True
        info_min[year_start] = False

        month_break = dates_[month_start].month
        jan_or_jul = month_start[(month_break == 1) | (month_break == 7)]
        info_fmt[jan_or_jul] = "%b"
        info_fmt[year_start] = "%b\n%Y"
    # Case 5. Less than 11 years ................
    elif span <= 11 * periodsperyear:
        year_start = _period_break(dates_, "year")
        quarter_start = _period_break(dates_, "quarter")
        info_maj[year_start] = True
        info_min[quarter_start] = True
        info_min[year_start] = False
        info_fmt[year_start] = "%Y"
    # Case 6. More than 12 years ................
    else:
        year_start = _period_break(dates_, "year")
        year_break = dates_[year_start].year
        nyears = span / periodsperyear
        (min_anndef, maj_anndef) = _get_default_annual_spacing(nyears)
        major_idx = year_start[(year_break % maj_anndef == 0)]
        info_maj[major_idx] = True
        minor_idx = year_start[(year_break % min_anndef == 0)]
        info_min[minor_idx] = True
        info_fmt[major_idx] = "%Y"

    return info


@functools.cache
def _monthly_finder(vmin: float, vmax: float, freq: BaseOffset) -> np.ndarray:
    # [FUNCTION DOC]
    # HELPER: memoized tick finder for monthly data. Inputs: vmin, vmax (month
    # ordinals) and freq. Returns a record array (value, major, minor, format text)
    # for every month in range. Layout depends on span relative to years: up to
    # 1.15 years each month minor with month format and year starts major with year;
    # up to 2.5 years quarter labels; up to 4 years January/July labels; up to 11
    # years only year labels with quarter minor ticks; beyond, annual spacing.
    # Ensures a year label on the first label when no year start is visible.
    # KNOWN ISSUE: in the 2.5 year branch a boolean is assigned into the format field at quarter positions (flagged by a TODO in the code as probably meant for another field); it is overwritten immediately afterwards for those positions.
    _, _, periodsperyear = _get_periods_per_ymd(freq)

    vmin_orig = vmin
    (vmin, vmax) = (int(vmin), int(vmax))
    span = vmax - vmin + 1

    # Initialize the output
    info = np.zeros(
        span, dtype=[("val", int), ("maj", bool), ("min", bool), ("fmt", "|S8")]
    )
    info["val"] = np.arange(vmin, vmax + 1)
    dates_ = info["val"]
    info["fmt"] = ""
    year_start = (dates_ % 12 == 0).nonzero()[0]
    info_maj = info["maj"]
    info_fmt = info["fmt"]

    if span <= 1.15 * periodsperyear:
        info_maj[year_start] = True
        info["min"] = True

        info_fmt[:] = "%b"
        info_fmt[year_start] = "%b\n%Y"

        if not has_level_label(year_start, vmin_orig):
            if dates_.size > 1:
                idx = 1
            else:
                idx = 0
            info_fmt[idx] = "%b\n%Y"

    elif span <= 2.5 * periodsperyear:
        quarter_start = (dates_ % 3 == 0).nonzero()
        info_maj[year_start] = True
        # TODO: Check the following : is it really info['fmt'] ?
        #  2023-09-15 this is reached in test_finder_monthly
        info["fmt"][quarter_start] = True
        info["min"] = True

        info_fmt[quarter_start] = "%b"
        info_fmt[year_start] = "%b\n%Y"

    elif span <= 4 * periodsperyear:
        info_maj[year_start] = True
        info["min"] = True

        jan_or_jul = (dates_ % 12 == 0) | (dates_ % 12 == 6)
        info_fmt[jan_or_jul] = "%b"
        info_fmt[year_start] = "%b\n%Y"

    elif span <= 11 * periodsperyear:
        quarter_start = (dates_ % 3 == 0).nonzero()
        info_maj[year_start] = True
        info["min"][quarter_start] = True

        info_fmt[year_start] = "%Y"

    else:
        nyears = span / periodsperyear
        (min_anndef, maj_anndef) = _get_default_annual_spacing(nyears)
        years = dates_[year_start] // 12 + 1
        major_idx = year_start[(years % maj_anndef == 0)]
        info_maj[major_idx] = True
        info["min"][year_start[(years % min_anndef == 0)]] = True

        info_fmt[major_idx] = "%Y"

    return info


@functools.cache
def _quarterly_finder(vmin: float, vmax: float, freq: BaseOffset) -> np.ndarray:
    # [FUNCTION DOC]
    # HELPER: memoized tick finder for quarterly data. Inputs: vmin, vmax (quarter
    # ordinals) and freq. Returns a record array as for the other finders. Up to
    # 3.5 years: every quarter minor, labeled by quarter, year starts major with
    # year; up to 11 years: year starts major labeled by year; beyond: annual spacing
    # applied to years counted from 1970.
    _, _, periodsperyear = _get_periods_per_ymd(freq)
    vmin_orig = vmin
    (vmin, vmax) = (int(vmin), int(vmax))
    span = vmax - vmin + 1

    info = np.zeros(
        span, dtype=[("val", int), ("maj", bool), ("min", bool), ("fmt", "|S8")]
    )
    info["val"] = np.arange(vmin, vmax + 1)
    info["fmt"] = ""
    dates_ = info["val"]
    info_maj = info["maj"]
    info_fmt = info["fmt"]
    year_start = (dates_ % 4 == 0).nonzero()[0]

    if span <= 3.5 * periodsperyear:
        info_maj[year_start] = True
        info["min"] = True

        info_fmt[:] = "Q%q"
        info_fmt[year_start] = "Q%q\n%F"
        if not has_level_label(year_start, vmin_orig):
            if dates_.size > 1:
                idx = 1
            else:
                idx = 0
            info_fmt[idx] = "Q%q\n%F"

    elif span <= 11 * periodsperyear:
        info_maj[year_start] = True
        info["min"] = True
        info_fmt[year_start] = "%F"

    else:
        # https://github.com/pandas-dev/pandas/pull/47602
        years = dates_[year_start] // 4 + 1970
        nyears = span / periodsperyear
        (min_anndef, maj_anndef) = _get_default_annual_spacing(nyears)
        major_idx = year_start[(years % maj_anndef == 0)]
        info_maj[major_idx] = True
        info["min"][year_start[(years % min_anndef == 0)]] = True
        info_fmt[major_idx] = "%F"

    return info


@functools.cache
def _annual_finder(vmin: float, vmax: float, freq: BaseOffset) -> np.ndarray:
    # [FUNCTION DOC]
    # HELPER: memoized tick finder for annual data. Inputs: vmin, vmax (year
    # ordinals; the upper bound is extended by one, unlike the other finders).
    # Returns a record array of year values with major flags at multiples of the
    # major spacing (labeled with year format) and minor flags at multiples of the
    # minor spacing, spacings from the annual spacing helper.
    # Note: small difference here vs other finders in adding 1 to vmax
    (vmin, vmax) = (int(vmin), int(vmax + 1))
    span = vmax - vmin + 1

    info = np.zeros(
        span, dtype=[("val", int), ("maj", bool), ("min", bool), ("fmt", "|S8")]
    )
    info["val"] = np.arange(vmin, vmax + 1)
    info["fmt"] = ""
    dates_ = info["val"]

    (min_anndef, maj_anndef) = _get_default_annual_spacing(span)
    major_idx = dates_ % maj_anndef == 0
    minor_idx = dates_ % min_anndef == 0
    info["maj"][major_idx] = True
    info["min"][minor_idx] = True
    info["fmt"][major_idx] = "%Y"

    return info


def get_finder(freq: BaseOffset):
    # [FUNCTION DOC]
    # Returns the tick finder function matching a frequency: annual, quarterly,
    # monthly, or the daily finder for business, weekly and finer frequencies.
    # Raises a not-implemented error for unsupported frequencies.
    # error: "BaseOffset" has no attribute "_period_dtype_code"
    dtype_code = freq._period_dtype_code  # type: ignore[attr-defined]
    fgroup = FreqGroup.from_period_dtype_code(dtype_code)

    if fgroup == FreqGroup.FR_ANN:
        return _annual_finder
    elif fgroup == FreqGroup.FR_QTR:
        return _quarterly_finder
    elif fgroup == FreqGroup.FR_MTH:
        return _monthly_finder
    elif (dtype_code >= FreqGroup.FR_BUS.value) or fgroup == FreqGroup.FR_WK:  # pyright: ignore[reportAttributeAccessIssue]
        return _daily_finder
    else:  # pragma: no cover
        raise NotImplementedError(f"Unsupported frequency: {dtype_code}")


class TimeSeries_DateLocator(mpl.ticker.Locator):  # pyright: ignore[reportAttributeAccessIssue]
    # [CLASS DOC]
    # Matplotlib tick locator for period-indexed axes (subclass of matplotlib's
    # locator). State: frequency offset, base, quarter, month and day defaults,
    # minor flag, dynamic flag, a zero offset, the optional plot object and the
    # selected finder function. In dynamic mode ticks come from the finder for the
    # current view; the non-dynamic fallback uses plain multiples of base.
    """
    Locates the ticks along an axis controlled by a :class:`Series`.

    Parameters
    ----------
    freq : BaseOffset
        Valid frequency specifier.
    minor_locator : {False, True}, optional
        Whether the locator is for minor ticks (True) or not.
    dynamic_mode : {True, False}, optional
        Whether the locator should work in dynamic mode.
    base : {int}, optional
    quarter : {int}, optional
    month : {int}, optional
    day : {int}, optional
    """

    axis: Axis

    def __init__(
        self,
        freq: BaseOffset,
        minor_locator: bool = False,
        dynamic_mode: bool = True,
        base: int = 1,
        quarter: int = 1,
        month: int = 1,
        day: int = 1,
        plot_obj=None,
    ) -> None:
        # [FUNCTION DOC]
        # Constructor. Inputs: freq (frequency, converted to a period offset), minor_locator
        # (default false), dynamic_mode (default true), base, quarter, month, day
        # (each default 1), plot_obj (optional). Stores these and selects the finder.
        freq = to_offset(freq, is_period=True)
        self.freq = freq
        self.base = base
        (self.quarter, self.month, self.day) = (quarter, month, day)
        self.isminor = minor_locator
        self.isdynamic = dynamic_mode
        self.offset = 0
        self.plot_obj = plot_obj
        self.finder = get_finder(freq)

    def _get_default_locs(self, vmin, vmax):
        # [FUNCTION DOC]
        # HELPER: runs the finder for the given limits and returns the values flagged
        # minor (for a minor locator) or major.
        """Returns the default locations of ticks."""
        locator = self.finder(vmin, vmax, self.freq)

        if self.isminor:
            return np.compress(locator["min"], locator["val"])
        return np.compress(locator["maj"], locator["val"])

    def __call__(self):
        # [FUNCTION DOC]
        # Returns tick locations for the axis view interval (swapped if reversed). In
        # dynamic mode uses the default locations; otherwise returns a list of integers
        # spaced by base starting from the next multiple at or after the minimum.
        """Return the locations of the ticks."""
        # axis calls Locator.set_axis inside set_m<xxxx>_formatter

        vi = tuple(self.axis.get_view_interval())
        vmin, vmax = vi
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        if self.isdynamic:
            locs = self._get_default_locs(vmin, vmax)
        else:  # pragma: no cover
            base = self.base
            (d, m) = divmod(vmin, base)
            vmin = (d + 1) * base
            # error: No overload variant of "range" matches argument types "float",
            # "float", "int"
            locs = list(range(vmin, vmax + 1, base))  # type: ignore[call-overload]
        return locs

    def autoscale(self):
        # [FUNCTION DOC]
        # Returns view limits for autoscaling: the first and last default tick
        # locations for the data interval, widened by one on each side if they coincide,
        # passed through matplotlib's non-singular adjustment.
        """
        Sets the view limits to the nearest multiples of base that contain the
        data.
        """
        # requires matplotlib >= 0.98.0
        (vmin, vmax) = self.axis.get_data_interval()

        locs = self._get_default_locs(vmin, vmax)
        (vmin, vmax) = locs[[0, -1]]
        if vmin == vmax:
            vmin -= 1
            vmax += 1
        return mpl.transforms.nonsingular(vmin, vmax)


# -------------------------------------------------------------------------
# --- Formatter ---
# -------------------------------------------------------------------------


class TimeSeries_DateFormatter(mpl.ticker.Formatter):  # pyright: ignore[reportAttributeAccessIssue]
    # [CLASS DOC]
    # Tick formatter for period-indexed axes (subclass of matplotlib's formatter).
    # State: frequency offset, minor and dynamic flags, selected finder, a mapping
    # from tick value to format text (empty until locations are set), plot object.
    """
    Formats the ticks along an axis controlled by a :class:`PeriodIndex`.

    Parameters
    ----------
    freq : BaseOffset
        Valid frequency specifier.
    minor_locator : bool, default False
        Whether the current formatter should apply to minor ticks (True) or
        major ticks (False).
    dynamic_mode : bool, default True
        Whether the formatter works in dynamic mode or not.
    """

    axis: Axis

    def __init__(
        self,
        freq: BaseOffset,
        minor_locator: bool = False,
        dynamic_mode: bool = True,
        plot_obj=None,
    ) -> None:
        # [FUNCTION DOC]
        # Constructor. Inputs: freq, minor_locator (default false), dynamic_mode
        # (default true), plot_obj. Converts frequency, initializes empty locations in
        # the parent, and selects the finder.
        freq = to_offset(freq, is_period=True)
        self.format = None
        self.freq = freq
        super().set_locs([])  # unused, for matplotlib compat
        self.formatdict: dict[Any, Any] | None = None
        self.isminor = minor_locator
        self.isdynamic = dynamic_mode
        self.offset = 0
        self.plot_obj = plot_obj
        self.finder = get_finder(freq)

    def _set_default_format(self, vmin, vmax):
        # [FUNCTION DOC]
        # HELPER: runs the finder and stores a mapping from tick value to format text
        # (for minor-only or major ticks). Returns the mapping.
        """Returns the default ticks spacing."""
        info = self.finder(vmin, vmax, self.freq)

        if self.isminor:
            format = np.compress(info["min"] & np.logical_not(info["maj"]), info)
        else:
            format = np.compress(info["maj"], info)
        self.formatdict = {x: f for (x, _, _, f) in format}
        return self.formatdict

    def set_locs(self, locs) -> None:
        # [FUNCTION DOC]
        # Called by matplotlib with tick locations. Ignores them: records them in the
        # parent, then recomputes the format mapping from the current (ordered) view
        # interval.
        """Sets the locations of the ticks"""
        # don't actually use the locs. This is just needed to work with
        # matplotlib. Force to use vmin, vmax

        super().set_locs(locs)

        (vmin, vmax) = tuple(self.axis.get_view_interval())
        if vmax < vmin:
            (vmin, vmax) = (vmax, vmin)
        self._set_default_format(vmin, vmax)

    def __call__(self, x, pos: int | None = 0) -> str:
        # [FUNCTION DOC]
        # Formats a tick value as period text. Returns empty text if the mapping was never
        # built. Looks up and removes the format for the value (empty if absent),
        # decodes bytes, builds the period from the integer ordinal at the frequency
        # (ignoring a business-day deprecation warning) and formats it. Side effect:
        # mutates the mapping.
        if self.formatdict is None:
            return ""
        else:
            fmt = self.formatdict.pop(x, "")
            if isinstance(fmt, np.bytes_):
                fmt = fmt.decode("utf-8")
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    "Period with BDay freq is deprecated",
                    category=FutureWarning,
                )
                period = Period(ordinal=int(x), freq=self.freq)
            assert isinstance(period, Period)
            return period.strftime(fmt)


class TimeSeries_TimedeltaFormatter(mpl.ticker.Formatter):  # pyright: ignore[reportAttributeAccessIssue]
    # [CLASS DOC]
    # Tick formatter for timedelta axes (subclass of matplotlib's formatter). State:
    # the time unit (nanoseconds default; microseconds, milliseconds or seconds).
    """
    Formats the ticks along an axis controlled by a :class:`TimedeltaIndex`.
    """

    def __init__(self, unit: TimeUnit = "ns"):
        # [FUNCTION DOC]
        # Constructor. Input: unit, default nanoseconds. Initializes the parent.
        self.unit = unit
        super().__init__()

    axis: Axis

    @staticmethod
    def format_timedelta_ticks(x, pos, n_decimals: int, exp: int = 9) -> str:
        # [FUNCTION DOC]
        # Static: formats a tick value as days, hours, minutes, seconds text. Inputs: x
        # (value in the unit), pos (unused), n_decimals (fractional digits, zero for
        # none), exp (powers of ten per second, default 9). Output hours:minutes:seconds
        # with zero padding, optional fraction, prefixed by a days count when non-zero.
        # KNOWN ISSUE: the docstring says seconds are converted, but the input is in the configured unit and divided by the unit's power of ten.
        """
        Convert seconds to 'D days HH:MM:SS.F'
        """
        s, ns = divmod(x, 10**exp)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        decimals = int(ns * 10 ** (n_decimals - exp))
        s = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
        if n_decimals > 0:
            s += f".{decimals:0{n_decimals}d}"
        if d != 0:
            s = f"{int(d):d} days {s}"
        return s

    def __call__(self, x, pos: int | None = 0) -> str:
        # [FUNCTION DOC]
        # Formats a tick value. Determines the exponent from the unit, derives the
        # number of fractional digits from the current view range width (capped at the
        # exponent), and delegates to the static formatting function. Returns text.
        exp = {"ns": 9, "us": 6, "ms": 3, "s": 0}[self.unit]
        (vmin, vmax) = tuple(self.axis.get_view_interval())
        n_decimals = min(int(np.ceil(np.log10(100 * 10**exp / abs(vmax - vmin)))), exp)
        return self.format_timedelta_ticks(x, pos, n_decimals, exp)
