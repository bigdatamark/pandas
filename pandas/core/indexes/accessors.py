"""
datetimelike delegation
"""

# [MODULE DOC]
# Implements the datetime-like accessor of Series (the dt namespace). A
# combined accessor class acts as a factory choosing, by the Series dtype, one
# of: datetime accessor (naive or tz-aware datetimes), timedelta accessor,
# period accessor, or an Arrow-backed temporal accessor. All but the Arrow one
# share a base class that converts the Series to the matching index/array,
# delegates property and method access to it, and wraps array-like results
# back into a Series aligned to the original index. Delegation is wired by
# class decorators that generate forwarding attributes from the datetime-like
# operation and method name lists of the array classes.
# Relationships: used by the Series class to build its dt accessor; depends on
# the accessor delegation utilities, datetime-like arrays and indexes and the
# Arrow extension array. Import-time side effects: class decorators add many
# attributes to the accessor classes.
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    NoReturn,
    cast,
)
import warnings

import numpy as np

from pandas._libs import lib
from pandas.errors import Pandas4Warning
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    is_integer_dtype,
    is_list_like,
)
from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    CategoricalDtype,
    DatetimeTZDtype,
    PeriodDtype,
)
from pandas.core.dtypes.generic import ABCSeries

from pandas.core.accessor import (
    PandasDelegate,
    delegate_names,
)
from pandas.core.arrays import (
    DatetimeArray,
    PeriodArray,
    TimedeltaArray,
)
from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.base import (
    NoNewAttributesMixin,
    PandasObject,
)
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex

if TYPE_CHECKING:
    from pandas import (
        DataFrame,
        Series,
    )


class Properties(PandasDelegate, PandasObject, NoNewAttributesMixin):
    # [CLASS DOC]
    # Base class of the non-Arrow dt accessors. State: the parent Series, the
    # original categorical Series if the data was categorical (otherwise none),
    # and the Series name; attributes are frozen after construction. Combines
    # the delegation base (forwarding of generated properties and methods),
    # the common pandas object base and the no-new-attributes guard. The
    # orig and name attributes are hidden from tab completion. Key methods
    # fetch the underlying datetime-like values and implement the get, set and
    # method delegation hooks.
    _hidden_attrs = PandasObject._hidden_attrs | {
        "orig",
        "name",
    }

    def __init__(self, data: Series, orig) -> None:
        # [FUNCTION DOC]
        # Takes the Series whose values are datetime-like and orig, the
        # original categorical Series or none. Raises a type error if data is
        # not a Series. Stores the parent, orig and the Series name, then
        # freezes the instance against adding attributes.
        if not isinstance(data, ABCSeries):
            raise TypeError(
                f"cannot convert an object of type {type(data)} to a datetimelike index"
            )

        self._parent = data
        self.orig = orig
        self.name = getattr(data, "name", None)
        self._freeze()

    def _get_values(self):
        # [FUNCTION DOC]
        # HELPER: for the delegation hooks. Wraps the parent Series data
        # without copying: a datetime index for naive or tz-aware datetime
        # dtypes, a timedelta index for timedelta dtype (both carrying the
        # Series name), or a period array for period dtype. Raises a type
        # error for any other dtype.
        data = self._parent
        if lib.is_np_dtype(data.dtype, "M"):
            return DatetimeIndex(data, copy=False, name=self.name)

        elif isinstance(data.dtype, DatetimeTZDtype):
            return DatetimeIndex(data, copy=False, name=self.name)

        elif lib.is_np_dtype(data.dtype, "m"):
            return TimedeltaIndex(data, copy=False, name=self.name)

        elif isinstance(data.dtype, PeriodDtype):
            return PeriodArray(data, copy=False)

        raise TypeError(
            f"cannot convert an object of type {type(data)} to a datetimelike index"
        )

    def _delegate_property_get(self, name: str):
        # [FUNCTION DOC]
        # HELPER: delegation hook for reading a generated property. Reads the
        # named attribute from the underlying values. Non-list-like results
        # are returned as is. Integer arrays are widened to 64-bit integers;
        # list-likes are converted to a numpy array and wrapped in a Series with
        # the original (categorical source) Series' index if present, else
        # the parent's index, with the Series name, and metadata propagated
        # from the parent. Propagates errors from the attribute lookup.
        from pandas import Series

        values = self._get_values()

        result = getattr(values, name)

        # maybe need to upcast (ints)
        if isinstance(result, np.ndarray):
            if is_integer_dtype(result):
                result = result.astype("int64")
        elif not is_list_like(result):
            return result

        result = np.asarray(result)

        if self.orig is not None:
            index = self.orig.index
        else:
            index = self._parent.index
        # return the result as a Series
        return Series(result, index=index, name=self.name).__finalize__(self._parent)

    def _delegate_property_set(self, name: str, value, *args, **kwargs) -> NoReturn:
        # [FUNCTION DOC]
        # HELPER: delegation hook for assigning a property. Always raises a
        # value error saying modifications are unsupported and the original
        # data should be changed. Arguments are ignored.
        raise ValueError(
            "modifications to a property of a datetimelike object are not supported. "
            "Change values on the original."
        )

    def _delegate_method(self, name: str, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: delegation hook for generated methods. Looks up the named
        # method on the underlying values and calls it with the given
        # arguments. Non-list-like results are returned as is; list-like ones
        # are wrapped in a Series using the parent's index and name, with
        # metadata propagated from the parent.
        # KNOWN ISSUE: unlike the property hook, this uses the parent index
        # even when an original categorical Series was recorded.
        from pandas import Series

        values = self._get_values()

        method = getattr(values, name)
        result = method(*args, **kwargs)

        if not is_list_like(result):
            return result

        return Series(result, index=self._parent.index, name=self.name).__finalize__(
            self._parent
        )


@delegate_names(
    delegate=ArrowExtensionArray,
    accessors=TimedeltaArray._datetimelike_ops,
    typ="property",
    accessor_mapping=lambda x: f"_dt_{x}",
    raise_on_missing=False,
)
@delegate_names(
    delegate=ArrowExtensionArray,
    accessors=TimedeltaArray._datetimelike_methods,
    typ="method",
    accessor_mapping=lambda x: f"_dt_{x}",
    raise_on_missing=False,
)
@delegate_names(
    delegate=ArrowExtensionArray,
    accessors=DatetimeArray._datetimelike_ops,
    typ="property",
    accessor_mapping=lambda x: f"_dt_{x}",
    raise_on_missing=False,
)
@delegate_names(
    delegate=ArrowExtensionArray,
    accessors=DatetimeArray._datetimelike_methods,
    typ="method",
    accessor_mapping=lambda x: f"_dt_{x}",
    raise_on_missing=False,
)
class ArrowTemporalProperties(PandasDelegate, PandasObject, NoNewAttributesMixin):
    # [CLASS DOC]
    # Accessor for Series backed by Arrow timestamp/duration extension arrays.
    # Decorators generate forwarding properties and methods for the datetime
    # and timedelta operation names, routed to prefixed temporal methods on
    # the Arrow array (missing ones are tolerated at class creation).
    # State: the parent Series and the original categorical Series or none
    # (stored under the underscore-prefixed name, unlike the base accessors).
    # Does not derive from the common accessor base class.
    def __init__(self, data: Series, orig) -> None:
        # [FUNCTION DOC]
        # Takes the Arrow-backed Series and orig (original categorical Series
        # or none). Raises a type error if data is not a Series. Stores both
        # and freezes the instance against new attributes.
        if not isinstance(data, ABCSeries):
            raise TypeError(
                f"cannot convert an object of type {type(data)} to a datetimelike index"
            )

        self._parent = data
        self._orig = orig
        self._freeze()

    def _delegate_property_get(self, name: str):
        # [FUNCTION DOC]
        # HELPER: delegation hook for reading a generated property. Looks up
        # the prefixed temporal attribute on the parent's array; raises a
        # not-implemented error naming the dtype if the array lacks it.
        # Non-list-like results are returned as is; otherwise the result is
        # wrapped in a Series of the parent's type using the original
        # (categorical source) index if present, else the parent index, with
        # the parent name, and metadata propagated from the parent.
        if not hasattr(self._parent.array, f"_dt_{name}"):
            raise NotImplementedError(
                f"dt.{name} is not supported for {self._parent.dtype}"
            )
        result = getattr(self._parent.array, f"_dt_{name}")

        if not is_list_like(result):
            return result

        if self._orig is not None:
            index = self._orig.index
        else:
            index = self._parent.index
        # return the result as a Series, which is by definition a copy
        result = type(self._parent)(
            result, index=index, name=self._parent.name
        ).__finalize__(self._parent)

        return result

    def _delegate_method(self, name: str, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: delegation hook for generated methods. Raises a
        # not-implemented error naming the dtype if the parent's array lacks
        # the prefixed temporal method. Otherwise calls it with the given
        # arguments and always wraps the result (no list-like check) in a
        # Series of the parent's type, with the original (categorical source)
        # index if present else the parent index, the parent name, and
        # metadata propagated from the parent.
        if not hasattr(self._parent.array, f"_dt_{name}"):
            raise NotImplementedError(
                f"dt.{name} is not supported for {self._parent.dtype}"
            )

        result = getattr(self._parent.array, f"_dt_{name}")(*args, **kwargs)

        if self._orig is not None:
            index = self._orig.index
        else:
            index = self._parent.index
        # return the result as a Series, which is by definition a copy
        result = type(self._parent)(
            result, index=index, name=self._parent.name
        ).__finalize__(self._parent)

        return result

    def to_pytimedelta(self):
        # [FUNCTION DOC]
        # Emits a deprecation warning (the future-change warning category)
        # saying a future version will return a Series of timedelta objects,
        # then returns the Arrow array's conversion to native timedelta
        # objects (an object-dtype numpy array).
        # GH 57463
        warnings.warn(
            f"The behavior of {type(self).__name__}.to_pytimedelta is deprecated, "
            "in a future version this will return a Series containing python "
            "datetime.timedelta objects instead of an ndarray. To retain the "
            "old behavior, call `np.array` on the result",
            Pandas4Warning,
            stacklevel=find_stack_level(),
        )
        return cast(ArrowExtensionArray, self._parent.array)._dt_to_pytimedelta()

    def to_pydatetime(self) -> Series:
        # [FUNCTION DOC]
        # Returns the Arrow array's conversion of its values to native
        # datetime objects. No arguments; no warning. The annotation says
        # Series but whatever the array conversion returns is passed through.
        # GH#20306
        return cast(ArrowExtensionArray, self._parent.array)._dt_to_pydatetime()

    def isocalendar(self) -> DataFrame:
        # [FUNCTION DOC]
        # Returns a frame with columns year, week and day built from the
        # Arrow array's ISO calendar struct: the struct's chunks are combined
        # and each field is wrapped in the parent's array type. The frame
        # gets a default index (not the Series index).
        from pandas import DataFrame

        result = (
            cast(ArrowExtensionArray, self._parent.array)
            ._dt_isocalendar()
            ._pa_array.combine_chunks()
        )
        iso_calendar_df = DataFrame(
            {
                col: type(self._parent.array)(result.field(i))  # type: ignore[call-arg]
                for i, col in enumerate(["year", "week", "day"])
            }
        )
        return iso_calendar_df

    @property
    def components(self) -> DataFrame:
        # [FUNCTION DOC]
        # Property returning a frame with one column each for days, hours,
        # minutes, seconds, milliseconds, microseconds and nanoseconds, taken
        # from the prefixed temporal attributes of the parent's array. The
        # frame gets a default index.
        from pandas import DataFrame

        components_df = DataFrame(
            {
                col: getattr(self._parent.array, f"_dt_{col}")
                for col in [
                    "days",
                    "hours",
                    "minutes",
                    "seconds",
                    "milliseconds",
                    "microseconds",
                    "nanoseconds",
                ]
            }
        )
        return components_df


@delegate_names(
    delegate=DatetimeArray,
    accessors=[*DatetimeArray._datetimelike_ops, "unit"],
    typ="property",
)
@delegate_names(
    delegate=DatetimeArray,
    accessors=[*DatetimeArray._datetimelike_methods, "as_unit"],
    typ="method",
)
class DatetimeProperties(Properties):
    # [CLASS DOC]
    # Accessor for naive and tz-aware datetime Series. Subclass of the base
    # accessor; decorators forward datetime properties, methods and the unit
    # accessors to the datetime array. Adds explicit conversion to native
    # datetimes, an inferred frequency property and the ISO calendar frame.
    """
    Accessor object for datetimelike properties of the Series values.

    Examples
    --------
    >>> seconds_series = pd.Series(pd.date_range("2000-01-01", periods=3, freq="s"))
    >>> seconds_series
    0   2000-01-01 00:00:00
    1   2000-01-01 00:00:01
    2   2000-01-01 00:00:02
    dtype: datetime64[us]
    >>> seconds_series.dt.second
    0    0
    1    1
    2    2
    dtype: int32

    >>> hours_series = pd.Series(pd.date_range("2000-01-01", periods=3, freq="h"))
    >>> hours_series
    0   2000-01-01 00:00:00
    1   2000-01-01 01:00:00
    2   2000-01-01 02:00:00
    dtype: datetime64[us]
    >>> hours_series.dt.hour
    0    0
    1    1
    2    2
    dtype: int32

    >>> quarters_series = pd.Series(pd.date_range("2000-01-01", periods=3, freq="QE"))
    >>> quarters_series
    0   2000-03-31
    1   2000-06-30
    2   2000-09-30
    dtype: datetime64[us]
    >>> quarters_series.dt.quarter
    0    1
    1    2
    2    3
    dtype: int32

    Returns a Series indexed like the original Series.
    Raises TypeError if the Series does not contain datetimelike values.
    """

    def to_pydatetime(self) -> Series:
        # [FUNCTION DOC]
        # Converts the values to native datetime objects (timezone kept,
        # precision truncated to microseconds) and returns them as an
        # object-dtype Series with a default index (not the original index).
        # KNOWN ISSUE: the docstring's Returns section says a numpy array,
        # but a Series is returned.
        """
        Return the data as a Series of :class:`datetime.datetime` objects.

        Timezone information is retained if present.

        .. warning::

           Python's datetime uses microsecond resolution, which is lower than
           pandas (nanosecond). The values are truncated.

        Returns
        -------
        numpy.ndarray
            Object dtype array containing native Python datetime objects.

        See Also
        --------
        datetime.datetime : Standard library value for a datetime.

        Examples
        --------
        >>> s = pd.Series(pd.date_range("20180310", periods=2))
        >>> s
        0   2018-03-10
        1   2018-03-11
        dtype: datetime64[us]

        >>> s.dt.to_pydatetime()
        0    2018-03-10 00:00:00
        1    2018-03-11 00:00:00
        dtype: object

        pandas' nanosecond precision is truncated to microseconds.

        >>> s = pd.Series(pd.date_range("20180310", periods=2, freq="ns"))
        >>> s
        0   2018-03-10 00:00:00.000000000
        1   2018-03-10 00:00:00.000000001
        dtype: datetime64[ns]

        >>> s.dt.to_pydatetime()
        0    2018-03-10 00:00:00
        1    2018-03-10 00:00:00
        dtype: object
        """
        # GH#20306
        from pandas import Series

        return Series(self._get_values().to_pydatetime(), dtype=object)

    @property
    def freq(self):
        # [FUNCTION DOC]
        # Property returning the frequency string inferred from the values by
        # the underlying datetime index, or none if it cannot be inferred.
        """
        Tries to return a string representing a frequency generated by infer_freq.

        Returns None if it can't autodetect the frequency.

        See Also
        --------
        Series.dt.to_period : Cast to PeriodArray/PeriodIndex at a particular
            frequency.

        Examples
        --------
        >>> ser = pd.Series(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
        >>> ser = pd.to_datetime(ser)
        >>> ser.dt.freq
        'D'

        >>> ser = pd.Series(["2022-01-01", "2024-01-01", "2026-01-01", "2028-01-01"])
        >>> ser = pd.to_datetime(ser)
        >>> ser.dt.freq
        '2YS-JAN'
        """
        return self._get_values().inferred_freq

    def isocalendar(self) -> DataFrame:
        # [FUNCTION DOC]
        # Returns a frame with year, week and day columns (nullable unsigned
        # integers, missing for missing dates) computed by the underlying
        # datetime index, re-indexed with the parent Series index.
        """
        Calculate year, week, and day according to the ISO 8601 standard.

        Returns
        -------
        DataFrame
            With columns year, week and day.

        See Also
        --------
        Timestamp.isocalendar : Function return a 3-tuple containing ISO year,
            week number, and weekday for the given Timestamp object.
        datetime.date.isocalendar : Return a named tuple object with
            three components: year, week and weekday.

        Examples
        --------
        >>> ser = pd.to_datetime(pd.Series(["2010-01-01", pd.NaT]))
        >>> ser.dt.isocalendar()
           year  week  day
        0  2009    53     5
        1  <NA>  <NA>  <NA>
        >>> ser.dt.isocalendar().week
        0      53
        1    <NA>
        Name: week, dtype: UInt32
        """
        return self._get_values().isocalendar().set_index(self._parent.index)


@delegate_names(
    delegate=TimedeltaArray, accessors=TimedeltaArray._datetimelike_ops, typ="property"
)
@delegate_names(
    delegate=TimedeltaArray,
    accessors=TimedeltaArray._datetimelike_methods,
    typ="method",
)
class TimedeltaProperties(Properties):
    # [CLASS DOC]
    # Accessor for timedelta Series. Subclass of the base accessor; decorators
    # forward timedelta properties and methods to the timedelta array. Adds
    # the deprecated native-timedelta conversion, the components frame and an
    # inferred frequency property.
    """
    Accessor object for datetimelike properties of the Series values.

    Returns a Series indexed like the original Series.
    Raises TypeError if the Series does not contain datetimelike values.

    Examples
    --------
    >>> seconds_series = pd.Series(
    ...     pd.timedelta_range(start="1 second", periods=3, freq="s")
    ... )
    >>> seconds_series
    0   0 days 00:00:01
    1   0 days 00:00:02
    2   0 days 00:00:03
    dtype: timedelta64[us]
    >>> seconds_series.dt.seconds
    0    1
    1    2
    2    3
    dtype: int32
    """

    def to_pytimedelta(self) -> np.ndarray:
        # [FUNCTION DOC]
        # Emits a deprecation warning (future-change category) that a future
        # version will return a Series, then returns an object-dtype numpy
        # array of native timedelta objects, one per element.
        """
        Return an array of native :class:`datetime.timedelta` objects.

        Python's standard `datetime` library uses a different representation
        timedelta's. This method converts a Series of pandas Timedeltas
        to `datetime.timedelta` format with the same length as the original
        Series.

        Returns
        -------
        numpy.ndarray
            Array of 1D containing data with `datetime.timedelta` type.

        See Also
        --------
        datetime.timedelta : A duration expressing the difference
            between two date, time, or datetime.

        Examples
        --------
        >>> s = pd.Series(pd.to_timedelta(np.arange(5), unit="D"))
        >>> s
        0   0 days
        1   1 days
        2   2 days
        3   3 days
        4   4 days
        dtype: timedelta64[s]

        >>> s.dt.to_pytimedelta()
        array([datetime.timedelta(0), datetime.timedelta(days=1),
        datetime.timedelta(days=2), datetime.timedelta(days=3),
        datetime.timedelta(days=4)], dtype=object)
        """
        # GH 57463
        warnings.warn(
            f"The behavior of {type(self).__name__}.to_pytimedelta is deprecated, "
            "in a future version this will return a Series containing python "
            "datetime.timedelta objects instead of an ndarray. To retain the "
            "old behavior, call `np.array` on the result",
            Pandas4Warning,
            stacklevel=find_stack_level(),
        )
        return self._get_values().to_pytimedelta()

    @property
    def components(self) -> DataFrame:
        # [FUNCTION DOC]
        # Property returning a frame of the timedelta components (days through
        # nanoseconds), one row per element, indexed by the parent Series
        # index, with metadata propagated from the parent.
        """
        Return a Dataframe of the components of the Timedeltas.

        Each row of the DataFrame corresponds to a Timedelta in the original
        Series and contains the individual components (days, hours, minutes,
        seconds, milliseconds, microseconds, nanoseconds) of the Timedelta.

        Returns
        -------
        DataFrame

        See Also
        --------
        TimedeltaIndex.components : Return a DataFrame of the individual resolution
            components of the Timedeltas.
        Series.dt.total_seconds : Return the total number of seconds in the duration.

        Examples
        --------
        >>> s = pd.Series(pd.to_timedelta(np.arange(5), unit="s"))
        >>> s
        0   0 days 00:00:00
        1   0 days 00:00:01
        2   0 days 00:00:02
        3   0 days 00:00:03
        4   0 days 00:00:04
        dtype: timedelta64[s]
        >>> s.dt.components
           days  hours  minutes  seconds  milliseconds  microseconds  nanoseconds
        0     0      0        0        0             0             0            0
        1     0      0        0        1             0             0            0
        2     0      0        0        2             0             0            0
        3     0      0        0        3             0             0            0
        4     0      0        0        4             0             0            0
        """
        return (
            self._get_values()
            .components.set_index(self._parent.index)
            .__finalize__(self._parent)
        )

    @property
    def freq(self):
        # [FUNCTION DOC]
        # Property returning the frequency string inferred from the values by
        # the underlying timedelta index, or none if it cannot be inferred.
        return self._get_values().inferred_freq


@delegate_names(
    delegate=PeriodArray, accessors=PeriodArray._datetimelike_ops, typ="property"
)
@delegate_names(
    delegate=PeriodArray, accessors=PeriodArray._datetimelike_methods, typ="method"
)
class PeriodProperties(Properties):
    # [CLASS DOC]
    # Accessor for period Series. Subclass of the base accessor; decorators
    # forward period properties and methods to the period array. No explicit
    # methods of its own.
    """
    Accessor object for datetimelike properties of the Series values.

    Returns a Series indexed like the original Series.
    Raises TypeError if the Series does not contain datetimelike values.

    Examples
    --------
    >>> seconds_series = pd.Series(
    ...     pd.period_range(
    ...         start="2000-01-01 00:00:00", end="2000-01-01 00:00:03", freq="s"
    ...     )
    ... )
    >>> seconds_series
    0    2000-01-01 00:00:00
    1    2000-01-01 00:00:01
    2    2000-01-01 00:00:02
    3    2000-01-01 00:00:03
    dtype: period[s]
    >>> seconds_series.dt.second
    0    0
    1    1
    2    2
    3    3
    dtype: int64

    >>> hours_series = pd.Series(
    ...     pd.period_range(start="2000-01-01 00:00", end="2000-01-01 03:00", freq="h")
    ... )
    >>> hours_series
    0    2000-01-01 00:00
    1    2000-01-01 01:00
    2    2000-01-01 02:00
    3    2000-01-01 03:00
    dtype: period[h]
    >>> hours_series.dt.hour
    0    0
    1    1
    2    2
    3    3
    dtype: int64

    >>> quarters_series = pd.Series(
    ...     pd.period_range(start="2000-01-01", end="2000-12-31", freq="Q-DEC")
    ... )
    >>> quarters_series
    0    2000Q1
    1    2000Q2
    2    2000Q3
    3    2000Q4
    dtype: period[Q-DEC]
    >>> quarters_series.dt.quarter
    0    1
    1    2
    2    3
    3    4
    dtype: int64
    """


class CombinedDatetimelikeProperties(
    DatetimeProperties, TimedeltaProperties, PeriodProperties
):
    # [CLASS DOC]
    # Factory class backing the dt accessor of Series. Inherits from the
    # datetime, timedelta and period accessors only so that the combined
    # attribute names are visible for documentation and completion; it is
    # never really instantiated, because the constructor hook returns an
    # instance of the appropriate concrete accessor.
    """
    Accessor object for Series values' datetime-like, timedelta and period properties.

    See Also
    --------
    DatetimeIndex : Index of datetime64 data.

    Examples
    --------
    >>> dates = pd.Series(
    ...     ["2024-01-01", "2024-01-15", "2024-02-5"], dtype="datetime64[ns]"
    ... )
    >>> dates.dt.day
    0     1
    1    15
    2     5
    dtype: int32
    >>> dates.dt.month
    0    1
    1    1
    2    2
    dtype: int32

    >>> dates = pd.Series(
    ...     ["2024-01-01", "2024-01-15", "2024-02-5"], dtype="datetime64[ns, UTC]"
    ... )
    >>> dates.dt.day
    0     1
    1    15
    2     5
    dtype: int32
    >>> dates.dt.month
    0    1
    1    1
    2    2
    dtype: int32
    """

    def __new__(cls, data: Series):  # pyright: ignore[reportInconsistentConstructor]
        # [FUNCTION DOC]
        # Constructor hook that picks the concrete accessor for a Series.
        # data must be a Series, otherwise a type error is raised. A
        # categorical Series is unwrapped to a Series of its category dtype
        # (same index and name, no copy) and the original is remembered. Then:
        # Arrow timestamp or duration dtype gives the Arrow accessor; naive
        # or tz-aware datetime dtype the datetime accessor; timedelta dtype
        # the timedelta accessor; period dtype the period accessor. Any other
        # dtype raises an attribute error stating that the dt accessor needs
        # datetime-like values. Returns the new accessor instance.
        # CombinedDatetimelikeProperties isn't really instantiated. Instead
        # we need to choose which parent (datetime or timedelta) is
        # appropriate. Since we're checking the dtypes anyway, we'll just
        # do all the validation here.

        if not isinstance(data, ABCSeries):
            raise TypeError(
                f"cannot convert an object of type {type(data)} to a datetimelike index"
            )

        orig = data if isinstance(data.dtype, CategoricalDtype) else None
        if orig is not None:
            data = data._constructor(
                orig.array,
                name=orig.name,
                copy=False,
                dtype=orig._values.categories.dtype,
                index=orig.index,
            )

        if isinstance(data.dtype, ArrowDtype) and data.dtype.kind in "Mm":
            return ArrowTemporalProperties(data, orig)
        if lib.is_np_dtype(data.dtype, "M"):
            return DatetimeProperties(data, orig)
        elif isinstance(data.dtype, DatetimeTZDtype):
            return DatetimeProperties(data, orig)
        elif lib.is_np_dtype(data.dtype, "m"):
            return TimedeltaProperties(data, orig)
        elif isinstance(data.dtype, PeriodDtype):
            return PeriodProperties(data, orig)

        raise AttributeError("Can only use .dt accessor with datetimelike values")
