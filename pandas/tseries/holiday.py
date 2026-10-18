# [MODULE DOC]
# Holiday and holiday-calendar definitions for business-day computations. Public
# contents (exported list): re-exports of the dateutil weekday constants; observance
# functions that move a holiday date to a nearby workday according to a rule; the
# Holiday class describing one rule-based holiday; a registry of calendar classes
# with registration and lookup functions; an abstract calendar base class
# (with a metaclass registering every subclass by name); a US federal calendar
# class; a factory merging two calendars into a new class; and module-level
# Holiday instances for common US holidays, Good Friday and Easter Monday.
# Depends on the top-level pandas package (index, series, timestamp, concatenation,
# date range, DateOffset) and on the tseries offsets module. Import-time side
# effects: the metaclass registers the abstract calendar and the US federal
# calendar into the module-level registry dictionary when they are defined.
from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
)
from typing import (
    TYPE_CHECKING,
    Literal,
    overload,
)
import warnings

from dateutil.relativedelta import (
    FR,
    MO,
    SA,
    SU,
    TH,
    TU,
    WE,
)
import numpy as np

from pandas._libs.tslibs.offsets import BaseOffset
from pandas.errors import PerformanceWarning

from pandas import (
    DateOffset,
    DatetimeIndex,
    Series,
    Timestamp,
    concat,
    date_range,
)

from pandas.tseries.offsets import (
    Day,
    Easter,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def next_monday(dt: datetime) -> datetime:
    # [FUNCTION DOC]
    # Observance rule. Input: dt, a date. Saturday moves forward two days and Sunday
    # forward one day, both to Monday; other days are returned unchanged. Returns a date.
    """
    If holiday falls on Saturday, use following Monday instead;
    if holiday falls on Sunday, use Monday instead
    """
    if dt.weekday() == 5:
        return dt + timedelta(2)
    elif dt.weekday() == 6:
        return dt + timedelta(1)
    return dt


def next_monday_or_tuesday(dt: datetime) -> datetime:
    # [FUNCTION DOC]
    # Observance rule for the second of two adjacent holidays. Input: dt, a date.
    # Saturday or Sunday moves forward two days (to Monday or Tuesday respectively),
    # Monday moves forward one day to Tuesday, other days unchanged.
    """
    For second holiday of two adjacent ones!
    If holiday falls on Saturday, use following Monday instead;
    if holiday falls on Sunday or Monday, use following Tuesday instead
    (because Monday is already taken by adjacent holiday on the day before)
    """
    dow = dt.weekday()
    if dow in (5, 6):
        return dt + timedelta(2)
    if dow == 0:
        return dt + timedelta(1)
    return dt


def previous_friday(dt: datetime) -> datetime:
    # [FUNCTION DOC]
    # Observance rule. Input: dt, a date. Saturday moves back one day and Sunday back
    # two days, both to Friday; other days unchanged.
    """
    If holiday falls on Saturday or Sunday, use previous Friday instead.
    """
    if dt.weekday() == 5:
        return dt - timedelta(1)
    elif dt.weekday() == 6:
        return dt - timedelta(2)
    return dt


def sunday_to_monday(dt: datetime) -> datetime:
    # [FUNCTION DOC]
    # Observance rule. Input: dt, a date. Sunday moves forward one day to Monday; all
    # other days unchanged.
    """
    If holiday falls on Sunday, use day thereafter (Monday) instead.
    """
    if dt.weekday() == 6:
        return dt + timedelta(1)
    return dt


def weekend_to_monday(dt: datetime) -> datetime:
    # [FUNCTION DOC]
    # Observance rule. Input: dt, a date. Sunday moves forward one day and Saturday
    # forward two days, both to Monday; other days unchanged.
    """
    If holiday falls on Sunday or Saturday,
    use day thereafter (Monday) instead.
    Needed for holidays such as Christmas observation in Europe
    """
    if dt.weekday() == 6:
        return dt + timedelta(1)
    elif dt.weekday() == 5:
        return dt + timedelta(2)
    return dt


def nearest_workday(dt: datetime) -> datetime:
    # [FUNCTION DOC]
    # Observance rule. Input: dt, a date. Saturday moves back one day to Friday,
    # Sunday moves forward one day to Monday; other days unchanged.
    """
    If holiday falls on Saturday, use day before (Friday) instead;
    if holiday falls on Sunday, use day thereafter (Monday) instead.
    """
    if dt.weekday() == 5:
        return dt - timedelta(1)
    elif dt.weekday() == 6:
        return dt + timedelta(1)
    return dt


def next_workday(dt: datetime) -> datetime:
    # [FUNCTION DOC]
    # Returns the next Monday-to-Friday day strictly after dt (a date), skipping
    # weekends. Always advances at least one day. Does not know about holidays.
    """
    returns next workday used for observances
    """
    dt += timedelta(days=1)
    while dt.weekday() > 4:
        # Mon-Fri are 0-4
        dt += timedelta(days=1)
    return dt


def previous_workday(dt: datetime) -> datetime:
    # [FUNCTION DOC]
    # Returns the closest Monday-to-Friday day strictly before dt (a date), skipping
    # weekends. Always moves back at least one day. Does not know about holidays.
    """
    returns previous workday used for observances
    """
    dt -= timedelta(days=1)
    while dt.weekday() > 4:
        # Mon-Fri are 0-4
        dt -= timedelta(days=1)
    return dt


def before_nearest_workday(dt: datetime) -> datetime:
    # [FUNCTION DOC]
    # Applies the nearest-workday rule, then returns the previous workday before that
    # result.
    """
    returns previous workday before nearest workday
    """
    return previous_workday(nearest_workday(dt))


def after_nearest_workday(dt: datetime) -> datetime:
    # [FUNCTION DOC]
    # Applies the nearest-workday rule, then returns the next workday after that result
    # (used for days such as Boxing Day, or for series of consecutive holidays).
    """
    returns next workday after nearest workday
    needed for Boxing day or multiple holidays in a series
    """
    return next_workday(nearest_workday(dt))


class Holiday:
    # [CLASS DOC]
    # A single holiday rule: name, a month and day (and optionally a year for a
    # one-off holiday), an optional offset or list of offsets applied to the anchor date
    # or an observance function (the two are mutually exclusive), optional validity
    # start and end dates, optional restriction to certain weekdays, and optional
    # dates to exclude. State: all constructor arguments as attributes (start and end
    # are stored as timestamps). Main method: dates, returning the holiday dates in a
    # range; private helpers build the yearly anchor dates and apply the rule.
    """
    Class that defines a holiday with start/end dates and rules
    for observance.
    """

    start_date: Timestamp | None
    end_date: Timestamp | None
    days_of_week: tuple[int, ...] | None

    def __init__(
        self,
        name: str,
        year=None,
        month=None,
        day=None,
        offset: BaseOffset | list[BaseOffset] | None = None,
        observance: Callable | None = None,
        start_date=None,
        end_date=None,
        days_of_week: tuple | None = None,
        exclude_dates: DatetimeIndex | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Validates and stores the rule. Inputs: name, holiday name; year, month, day,
        # the anchor (year default none means recurring yearly); offset, one offset or a flat
        # list of offsets, default none; observance, callable mapping a date to a date,
        # default none; start_date and end_date, optional first and last date (converted to
        # timestamps); days_of_week, optional tuple of weekday numbers (Monday zero) to keep;
        # exclude_dates, optional date index of dates to drop. Raises not-implemented error
        # if both offset and observance are given, a value error if the offset is not an
        # offset or flat list of offsets, a value error if days_of_week is not none or a
        # tuple, and a value error if exclude_dates is not none or a date index.
        # KNOWN ISSUE: the docstring says the name defaults to the class name, but it is a
        # required argument with no default.
        """
        Parameters
        ----------
        name : str
            Name of the holiday , defaults to class name
        year : int, default None
            Year of the holiday
        month : int, default None
            Month of the holiday
        day : int, default None
            Day of the holiday
        offset : list of pandas.tseries.offsets or
                class from pandas.tseries.offsets, default None
            Computes offset from date
        observance : function, default None
            Computes when holiday is given a pandas Timestamp
        start_date : datetime-like, default None
            First date the holiday is observed
        end_date : datetime-like, default None
            Last date the holiday is observed
        days_of_week : tuple of int or dateutil.relativedelta weekday strs, default None
            Provide a tuple of days e.g  (0,1,2,3,) for Monday through Thursday
            Monday=0,..,Sunday=6
            Only instances of the holiday included in days_of_week will be computed
        exclude_dates : DatetimeIndex or default None
            Specific dates to exclude e.g. skipping a specific year's holiday

        Examples
        --------
        >>> from dateutil.relativedelta import MO

        >>> USMemorialDay = pd.tseries.holiday.Holiday(
        ...     "Memorial Day", month=5, day=31, offset=pd.DateOffset(weekday=MO(-1))
        ... )
        >>> USMemorialDay
        Holiday: Memorial Day (month=5, day=31, offset=<DateOffset: weekday=MO(-1)>)

        >>> USLaborDay = pd.tseries.holiday.Holiday(
        ...     "Labor Day", month=9, day=1, offset=pd.DateOffset(weekday=MO(1))
        ... )
        >>> USLaborDay
        Holiday: Labor Day (month=9, day=1, offset=<DateOffset: weekday=MO(+1)>)

        >>> July3rd = pd.tseries.holiday.Holiday("July 3rd", month=7, day=3)
        >>> July3rd
        Holiday: July 3rd (month=7, day=3, )

        >>> NewYears = pd.tseries.holiday.Holiday(
        ...     "New Years Day",
        ...     month=1,
        ...     day=1,
        ...     observance=pd.tseries.holiday.nearest_workday,
        ... )
        >>> NewYears  # doctest: +SKIP
        Holiday: New Years Day (
            month=1, day=1, observance=<function nearest_workday at 0x66545e9bc440>
        )

        >>> July3rd = pd.tseries.holiday.Holiday(
        ...     "July 3rd", month=7, day=3, days_of_week=(0, 1, 2, 3)
        ... )
        >>> July3rd
        Holiday: July 3rd (month=7, day=3, )
        """
        if offset is not None:
            if observance is not None:
                raise NotImplementedError("Cannot use both offset and observance.")
            if not (
                isinstance(offset, BaseOffset)
                or (
                    isinstance(offset, list)
                    and all(isinstance(off, BaseOffset) for off in offset)
                )
            ):
                raise ValueError(
                    "Only BaseOffsets and flat lists of them are supported for offset."
                )

        self.name = name
        self.year = year
        self.month = month
        self.day = day
        self.offset = offset
        self.start_date = (
            Timestamp(start_date) if start_date is not None else start_date
        )
        self.end_date = Timestamp(end_date) if end_date is not None else end_date
        self.observance = observance
        if not (days_of_week is None or isinstance(days_of_week, tuple)):
            raise ValueError("days_of_week must be None or tuple.")
        self.days_of_week = days_of_week
        if not (exclude_dates is None or isinstance(exclude_dates, DatetimeIndex)):
            raise ValueError("exclude_dates must be None or of type DatetimeIndex.")
        self.exclude_dates = exclude_dates

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Text form: the holiday name followed by parenthesized details: the year if
        # given, month and day, then the offset and/or the observance if present.
        # Offset and observance texts are appended without separators.
        info = ""
        if self.year is not None:
            info += f"year={self.year}, "
        info += f"month={self.month}, day={self.day}, "

        if self.offset is not None:
            info += f"offset={self.offset}"

        if self.observance is not None:
            info += f"observance={self.observance}"

        repr = f"Holiday: {self.name} ({info})"
        return repr

    @overload
    def dates(self, start_date, end_date, return_name: Literal[True]) -> Series: ...
        # [FUNCTION DOC]
        # Type-checking overload: with the name flag true, returns a series. Implementation below.

    @overload
    def dates(
        self, start_date, end_date, return_name: Literal[False]
    ) -> DatetimeIndex: ...
        # [FUNCTION DOC]
        # Type-checking overload: with the name flag false, returns a date index. Implementation below.

    @overload
    def dates(self, start_date, end_date) -> DatetimeIndex: ...
        # [FUNCTION DOC]
        # Type-checking overload: without the name flag, returns a date index. Implementation below.

    def dates(
        self, start_date, end_date, return_name: bool = False
    ) -> Series | DatetimeIndex:
        # [FUNCTION DOC]
        # Computes the dates on which the holiday is observed within a range. Inputs:
        # start_date, end_date, datetime-like bounds (inclusive); return_name, default
        # false. Returns a date index, or a series indexed by the dates whose values are
        # the holiday name when return_name is true. A rule with a fixed year ignores the
        # range and the other filters and returns just that single date. Otherwise: yearly
        # anchor dates spanning the range plus a year on each side are produced, the
        # offset or observance is applied, dates are filtered by weekday if requested,
        # the range is narrowed by the rule's own start and end dates (localized to the
        # range's time zone), and excluded dates are removed.
        """
        Calculate holidays observed between start date and end date

        Parameters
        ----------
        start_date : starting date, datetime-like, optional
        end_date : ending date, datetime-like, optional
        return_name : bool, optional, default=False
            If True, return a series that has dates and holiday names.
            False will only return dates.

        Returns
        -------
        Series or DatetimeIndex
            Series if return_name is True
        """
        start_date = Timestamp(start_date)
        end_date = Timestamp(end_date)

        filter_start_date = start_date
        filter_end_date = end_date

        if self.year is not None:
            dt = Timestamp(datetime(self.year, self.month, self.day))
            dti = DatetimeIndex([dt])
            if return_name:
                return Series(self.name, index=dti)
            else:
                return dti

        dates = self._reference_dates(start_date, end_date)
        holiday_dates = self._apply_rule(dates)
        if self.days_of_week is not None:
            holiday_dates = holiday_dates[
                np.isin(
                    # error: "DatetimeIndex" has no attribute "dayofweek"
                    holiday_dates.dayofweek,  # type: ignore[attr-defined]
                    self.days_of_week,
                ).ravel()
            ]

        if self.start_date is not None:
            filter_start_date = max(
                self.start_date.tz_localize(filter_start_date.tz), filter_start_date
            )
        if self.end_date is not None:
            filter_end_date = min(
                self.end_date.tz_localize(filter_end_date.tz), filter_end_date
            )
        holiday_dates = holiday_dates[
            (holiday_dates >= filter_start_date) & (holiday_dates <= filter_end_date)
        ]

        if self.exclude_dates is not None:
            holiday_dates = holiday_dates.difference(self.exclude_dates)
        if return_name:
            return Series(self.name, index=holiday_dates)
        return holiday_dates

    def _reference_dates(
        self, start_date: Timestamp, end_date: Timestamp
    ) -> DatetimeIndex:
        # [FUNCTION DOC]
        # HELPER: for the dates method. Inputs: start_date, end_date, timestamps. Returns a
        # date index of yearly anchor dates (this holiday's month and day) from the year
        # before the start through the year after the end, in the start's time zone. If the
        # rule has its own start or end date, it replaces the corresponding bound (localized
        # to the start's zone). Raises an error if month or day is none.
        """
        Get reference dates for the holiday.

        Return reference dates for the holiday also returning the year
        prior to the start_date and year following the end_date.  This ensures
        that any offsets to be applied will yield the holidays within
        the passed in dates.
        """
        if self.start_date is not None:
            start_date = self.start_date.tz_localize(start_date.tz)

        if self.end_date is not None:
            end_date = self.end_date.tz_localize(start_date.tz)

        year_offset = DateOffset(years=1)
        reference_start_date = Timestamp(
            datetime(start_date.year - 1, self.month, self.day)
        )

        reference_end_date = Timestamp(
            datetime(end_date.year + 1, self.month, self.day)
        )
        # Don't process unnecessary holidays
        dates = date_range(
            start=reference_start_date,
            end=reference_end_date,
            freq=year_offset,
            tz=start_date.tz,
        )

        return dates

    def _apply_rule(self, dates: DatetimeIndex) -> DatetimeIndex:
        # [FUNCTION DOC]
        # HELPER: for the dates method. Input: dates, a date index of anchors. Returns a
        # copy for an empty input; applies the observance function elementwise (through
        # an inline anonymous function) if set; otherwise adds each offset in turn
        # (a single offset is treated as a list of one), suppressing performance warnings
        # from non-vectorized offsets; with neither returns the input.
        """
        Apply the given offset/observance to a DatetimeIndex of dates.

        Parameters
        ----------
        dates : DatetimeIndex
            Dates to apply the given offset/observance rule

        Returns
        -------
        Dates with rules applied
        """
        if dates.empty:
            return dates.copy()

        if self.observance is not None:
            return dates.map(lambda d: self.observance(d))

        if self.offset is not None:
            if not isinstance(self.offset, list):
                offsets = [self.offset]
            else:
                offsets = self.offset
            for offset in offsets:
                # if we are adding a non-vectorized value
                # ignore the PerformanceWarnings:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", PerformanceWarning)
                    dates += offset
        return dates


holiday_calendars: dict[str, type[AbstractHolidayCalendar]] = {}


def register(cls) -> None:
    # [FUNCTION DOC]
    # Registers a calendar class in the module-level registry. Input: cls, a calendar
    # class. The key is its name attribute if it has one, otherwise the class name. A
    # later registration under the same key replaces the earlier one. Returns nothing.
    try:
        name = cls.name
    except AttributeError:
        name = cls.__name__
    holiday_calendars[name] = cls


def get_calendar(name: str) -> AbstractHolidayCalendar:
    # [FUNCTION DOC]
    # Returns a new instance of the calendar registered under a name. Input: name,
    # string. Raises a key error if no calendar was registered under that name.
    """
    Return an instance of a calendar based on its name.

    Parameters
    ----------
    name : str
        Calendar name to return an instance of
    """
    return holiday_calendars[name]()


class HolidayCalendarMetaClass(type):
    # [CLASS DOC]
    # Metaclass for calendar classes: every class created with it is automatically
    # registered in the module registry under its name. Applies to the abstract
    # calendar and all subclasses, including the base itself.
    def __new__(cls, clsname: str, bases, attrs):
        # [FUNCTION DOC]
        # Creates the class normally, registers it, and returns it. Inputs: the class name,
        # the base classes and the attribute mapping.
        calendar_class = super().__new__(cls, clsname, bases, attrs)
        register(calendar_class)
        return calendar_class


class AbstractHolidayCalendar(metaclass=HolidayCalendarMetaClass):
    # [CLASS DOC]
    # Base class for holiday calendars. Class state: rules, list of Holiday objects
    # (shared mutable default empty list); default start (1970-01-01) and end
    # (2200-12-31) timestamps; and a private cache of the last computed range and its
    # holiday series. Instances have a name (default class name) and may override rules.
    # Methods: find a rule by name, compute holidays over a range, and merge rule sets
    # (static class-level merge and instance merge). Uses the registering metaclass.
    """
    Abstract interface to create holidays following certain rules.
    """

    rules: list[Holiday] = []
    start_date = Timestamp(datetime(1970, 1, 1))
    end_date = Timestamp(datetime(2200, 12, 31))
    _cache: tuple[Timestamp, Timestamp, Series] | None = None

    def __init__(self, name: str = "", rules=None) -> None:
        # [FUNCTION DOC]
        # Initializes the calendar. Inputs: name, default empty string meaning use the
        # class name; rules, optional list of Holiday objects replacing the class's rules.
        """
        Initializes holiday object with a given set a rules.  Normally
        classes just have the rules defined within them.

        Parameters
        ----------
        name : str
            Name of the holiday calendar, defaults to class name
        rules : array of Holiday objects
            A set of rules used to create the holidays.
        """
        super().__init__()
        if not name:
            name = type(self).__name__
        self.name = name

        if rules is not None:
            self.rules = rules

    def rule_from_name(self, name: str) -> Holiday | None:
        # [FUNCTION DOC]
        # Returns the first rule whose name equals the given name, or none if there is none.
        for rule in self.rules:
            if rule.name == name:
                return rule

        return None

    def holidays(
        self, start=None, end=None, return_name: bool = False
    ) -> DatetimeIndex | Series:
        # [FUNCTION DOC]
        # Computes holiday dates over a range using all rules. Inputs: start, end,
        # datetime-like or none (defaults to the abstract base class's default bounds);
        # return_name, default false. Returns a series (dates mapped to holiday names) when
        # true, otherwise the index of dates. Uses a per-instance cache of the last result:
        # the rules are recomputed only when the requested range extends outside the cached
        # range; the cached series is sorted by date, then sliced to the range. With no rules
        # returns an empty object-valued series. Raises a general exception if rules is none.
        # KNOWN ISSUE: the default bounds are taken from the abstract base class rather than
        # from the class of this calendar, so a subclass that overrides its default start or end
        # date is ignored. The cache is also not invalidated when the rules change.
        # KNOWN ISSUE: the docstring's returns section does not mention the series variant.
        """
        Returns a curve with holidays between start_date and end_date

        Parameters
        ----------
        start : starting date, datetime-like, optional
        end : ending date, datetime-like, optional
        return_name : bool, optional
            If True, return a series that has dates and holiday names.
            False will only return a DatetimeIndex of dates.

        Returns
        -------
            DatetimeIndex of holidays
        """
        if self.rules is None:
            raise Exception(
                f"Holiday Calendar {self.name} does not have any rules specified"
            )

        if start is None:
            start = AbstractHolidayCalendar.start_date

        if end is None:
            end = AbstractHolidayCalendar.end_date

        start = Timestamp(start)
        end = Timestamp(end)

        # If we don't have a cache or the dates are outside the prior cache, we
        # get them again
        if self._cache is None or start < self._cache[0] or end > self._cache[1]:
            pre_holidays = [
                rule.dates(start, end, return_name=True) for rule in self.rules
            ]
            if pre_holidays:
                holidays = concat(pre_holidays)
            else:
                holidays = Series(index=DatetimeIndex([]), dtype=object)

            self._cache = (start, end, holidays.sort_index())

        holidays = self._cache[2]
        holidays = holidays[start:end]

        if return_name:
            return holidays
        else:
            return holidays.index

    @staticmethod
    def merge_class(base, other):
        # [FUNCTION DOC]
        # Static method merging two rule collections by holiday name. Inputs: base and
        # other, each a calendar instance or class (its rules are used), a single Holiday, or a
        # list of holidays. Returns a list of holidays in which entries from base override
        # entries of other with the same name.
        """
        Merge holiday calendars together. The base calendar
        will take precedence to other. The merge will be done
        based on each holiday's name.

        Parameters
        ----------
        base : AbstractHolidayCalendar
          instance/subclass or array of Holiday objects
        other : AbstractHolidayCalendar
          instance/subclass or array of Holiday objects
        """
        try:
            other = other.rules
        except AttributeError:
            pass

        if not isinstance(other, list):
            other = [other]
        other_holidays = {holiday.name: holiday for holiday in other}

        try:
            base = base.rules
        except AttributeError:
            pass

        if not isinstance(base, list):
            base = [base]
        base_holidays = {holiday.name: holiday for holiday in base}

        other_holidays.update(base_holidays)
        return list(other_holidays.values())

    def merge(self, other, inplace: bool = False):
        # [FUNCTION DOC]
        # Merges another calendar's (or rule list's) rules with this calendar's, with this
        # calendar's rules taking precedence by name. Inputs: other; inplace, default false.
        # When true the calendar's rules are replaced and nothing is returned; otherwise
        # returns the merged list of holidays.
        """
        Merge holiday calendars together.  The caller's class
        rules take precedence.  The merge will be done
        based on each holiday's name.

        Parameters
        ----------
        other : holiday calendar
        inplace : bool (default=False)
            If True set rule_table to holidays, else return array of Holidays
        """
        holidays = self.merge_class(self, other)
        if inplace:
            self.rules = holidays
        else:
            return holidays


USMemorialDay = Holiday(
    "Memorial Day", month=5, day=31, offset=DateOffset(weekday=MO(-1))
)
USLaborDay = Holiday("Labor Day", month=9, day=1, offset=DateOffset(weekday=MO(1)))
USColumbusDay = Holiday(
    "Columbus Day", month=10, day=1, offset=DateOffset(weekday=MO(2))
)
USThanksgivingDay = Holiday(
    "Thanksgiving Day", month=11, day=1, offset=DateOffset(weekday=TH(4))
)
USMartinLutherKingJr = Holiday(
    "Birthday of Martin Luther King, Jr.",
    start_date=datetime(1986, 1, 1),
    month=1,
    day=1,
    offset=DateOffset(weekday=MO(3)),
)
USPresidentsDay = Holiday(
    "Washington's Birthday", month=2, day=1, offset=DateOffset(weekday=MO(3))
)
GoodFriday = Holiday("Good Friday", month=1, day=1, offset=[Easter(), Day(-2)])

EasterMonday = Holiday("Easter Monday", month=1, day=1, offset=[Easter(), Day(1)])


class USFederalHolidayCalendar(AbstractHolidayCalendar):
    # [CLASS DOC]
    # Calendar of US federal holidays: new year, Martin Luther King day, Presidents day,
    # Memorial day, Juneteenth (from 2021), Independence day, Labor day, Columbus day,
    # Veterans day, Thanksgiving and Christmas. Fixed-date holidays use the nearest
    # workday observance; the others use the module-level rules with weekday offsets.
    # Defines rules only.
    """
    US Federal Government Holiday Calendar based on rules specified by:
    https://www.opm.gov/policy-data-oversight/pay-leave/federal-holidays/
    """

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        USMemorialDay,
        Holiday(
            "Juneteenth National Independence Day",
            month=6,
            day=19,
            start_date="2021-06-18",
            observance=nearest_workday,
        ),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USColumbusDay,
        Holiday("Veterans Day", month=11, day=11, observance=nearest_workday),
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


def HolidayCalendarFactory(name: str, base, other, base_class=AbstractHolidayCalendar):
    # [FUNCTION DOC]
    # Builds a new calendar class by merging the rules of two calendars. Inputs: name,
    # the new class name (also its name attribute); base and other, calendar classes,
    # instances, or rule lists to merge (base takes precedence); base_class, the parent of
    # the new class, default the abstract calendar. Returns the new class (not an
    # instance), which the metaclass registers under the name.
    rules = AbstractHolidayCalendar.merge_class(base, other)
    calendar_class = type(name, (base_class,), {"rules": rules, "name": name})
    return calendar_class


__all__ = [
    "FR",
    "MO",
    "SA",
    "SU",
    "TH",
    "TU",
    "WE",
    "HolidayCalendarFactory",
    "after_nearest_workday",
    "before_nearest_workday",
    "get_calendar",
    "nearest_workday",
    "next_monday",
    "next_monday_or_tuesday",
    "next_workday",
    "previous_friday",
    "previous_workday",
    "register",
    "sunday_to_monday",
    "weekend_to_monday",
]
