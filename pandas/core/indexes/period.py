# [MODULE DOC]
# Implementation of PeriodIndex, the index of regular time spans (periods), and
# its range-building function.
# Public contents: the PeriodIndex class (re-labelled as living in the top-level
# package namespace) and the period range function. Also holds a private
# unpickling reconstructor and shared documentation keyword dictionaries.
# Relationships: data lives in a PeriodArray; the class extends the datetime-like
# index mixin and borrows datetime index machinery (partial string indexing,
# slicing); field accessors are generated onto the class by the extension index
# attribute-pinning decorator; conversion to timestamps returns a DatetimeIndex.
# Import-time side effects: builds two module-level documentation kwargs
# dictionaries (a copy of the base index one extended with a target class text,
# and one naming the array class) and applies decorators to the class.

from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
)
from typing import (
    TYPE_CHECKING,
    Self,
)

import numpy as np

from pandas._libs import index as libindex
from pandas._libs.tslibs import (
    BaseOffset,
    Day,
    NaT,
    Period,
    Resolution,
    Tick,
)
from pandas._libs.tslibs.dtypes import OFFSET_TO_PERIOD_FREQSTR
from pandas.util._decorators import (
    cache_readonly,
    doc,
    set_module,
)

from pandas.core.dtypes.common import is_integer
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import is_valid_na_for_dtype

from pandas.core.arrays.period import (
    PeriodArray,
    period_array,
    raise_on_incompatible,
    validate_dtype_freq,
)
import pandas.core.common as com
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import maybe_extract_name
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
from pandas.core.indexes.datetimes import (
    DatetimeIndex,
    Index,
)
from pandas.core.indexes.extension import inherit_names

if TYPE_CHECKING:
    from collections.abc import Hashable

    from pandas._typing import (
        Dtype,
        DtypeObj,
        npt,
    )


_index_doc_kwargs = dict(ibase._index_doc_kwargs)
_index_doc_kwargs.update({"target_klass": "PeriodIndex or list of Periods"})
_shared_doc_kwargs = {
    "klass": "PeriodArray",
}

# --- Period index sketch


def _new_PeriodIndex(cls, **d):
    # [FUNCTION DOC]
    # HELPER: reconstructor used when unpickling indexes (GH13277). The class is
    # passed in; keyword arguments are the saved state and must contain the key data
    # (removed from the dictionary, raising a key error if absent). If the stored
    # values have int64 dtype, the optional freq entry is popped (default none), a
    # period dtype is built from it, the integers are wrapped in a PeriodArray and
    # the instance is made through the private simple constructor with the remaining
    # keywords. Otherwise the class is called normally with the values and the
    # remaining keywords. Returns the index.
    # GH13277 for unpickling
    values = d.pop("data")
    if values.dtype == "int64":
        freq = d.pop("freq", None)
        dtype = PeriodDtype(freq)
        values = PeriodArray(values, dtype=dtype)
        return cls._simple_new(values, **d)
    else:
        return cls(values, **d)


@inherit_names(
    ["strftime", "start_time", "end_time", *PeriodArray._field_ops],
    PeriodArray,
    wrap=True,
)
@inherit_names(["is_leap_year"], PeriodArray)
@set_module("pandas")
class PeriodIndex(DatetimeIndexOpsMixin):
    # [CLASS DOC]
    # Immutable index of period values, stored as int64 ordinals with a frequency but
    # boxed to Period scalars on access. Inherits from the datetime-like index mixin.
    # State: _data is a PeriodArray; freq and dtype (a period dtype) come from it.
    # Attributes for calendar fields (year, month, day, quarter, week, and so on),
    # start_time, end_time, strftime and is_leap_year are pinned from the array by
    # decorators (wrapping results as indexes); hour, minute and second are explicit
    # properties. Key methods: constructor, the from_fields and from_ordinals class
    # methods, asfreq, to_timestamp, shift, get_loc, asof_locs and is_full.
    # Supports partial string indexing. Reports its inferred type as period.
    """
    Immutable ndarray holding ordinal values indicating regular periods in time.

    Index keys are boxed to Period objects which carries the metadata (eg,
    frequency information).

    Parameters
    ----------
    data : array-like (1d int np.ndarray or PeriodArray), optional
        Optional period-like data to construct index with.
    freq : str or period object, optional
        One of pandas period strings or corresponding objects.
    dtype : str or PeriodDtype, default None
        A dtype from which to extract a freq.
    copy : bool, default None
        Whether to copy input data, only relevant for array, Series, and Index
        inputs (for other input, e.g. a list, a new array is created anyway).
        Defaults to True for array input and False for Index/Series.
        Set to False to avoid copying array input at your own risk (if you
        know the input data won't be modified elsewhere).
        Set to True to force copying Series/Index input up front.
    name : str, default None
        Name of the resulting PeriodIndex.

    Attributes
    ----------
    day
    dayofweek
    day_of_week
    dayofyear
    day_of_year
    days_in_month
    daysinmonth
    end_time
    freq
    freqstr
    hour
    is_leap_year
    minute
    month
    quarter
    qyear
    second
    start_time
    week
    weekday
    weekofyear
    year

    Methods
    -------
    asfreq
    strftime
    to_timestamp
    from_fields
    from_ordinals

    Raises
    ------
    ValueError
        Passing the parameter data as a list without specifying either freq or
        dtype will raise a ValueError: "freq not specified and cannot be inferred"

    See Also
    --------
    Index : The base pandas Index type.
    Period : Represents a period of time.
    DatetimeIndex : Index with datetime64 data.
    TimedeltaIndex : Index of timedelta64 data.
    period_range : Create a fixed-frequency PeriodIndex.

    Examples
    --------
    >>> idx = pd.PeriodIndex(data=["2000Q1", "2002Q3"], freq="Q")
    >>> idx
    PeriodIndex(['2000Q1', '2002Q3'], dtype='period[Q-DEC]')
    """

    _typ = "periodindex"

    _data: PeriodArray
    freq: BaseOffset
    dtype: PeriodDtype

    _data_cls = PeriodArray
    _supports_partial_string_indexing = True

    @property
    def _engine_type(self) -> type[libindex.PeriodEngine]:
        # [FUNCTION DOC]
        # HELPER: returns the Cython hash-table engine class used for lookups,
        # the period engine.
        return libindex.PeriodEngine

    @cache_readonly
    def _resolution_obj(self) -> Resolution:
        # [FUNCTION DOC]
        # HELPER: cached resolution of the index, taken from the period dtype; exists for
        # compatibility with the datetime index.
        # for compat with DatetimeIndex
        return self.dtype._resolution_obj

    # --------------------------------------------------------------------
    # methods that dispatch to array and wrap result in Index
    # These are defined here instead of via inherit_names for mypy

    @doc(
        PeriodArray.asfreq,
        other="arrays.PeriodArray",
        other_name="PeriodArray",
        **_shared_doc_kwargs,
    )
    def asfreq(self, freq=None, how: str = "E") -> Self:
        # [FUNCTION DOC]
        # Convert the periods to another frequency. Parameters: freq is the target
        # frequency (string or offset, default none); how selects the position within the
        # span used for the conversion, a string defaulting to the end ("E"; start
        # values such as "S" are also accepted by the array). Delegates to the array and
        # returns a new PeriodIndex of the same class with the same name. Documentation
        # is shared from the array method.
        arr = self._data.asfreq(freq, how)
        return type(self)._simple_new(arr, name=self.name)

    @doc(PeriodArray.to_timestamp)
    def to_timestamp(self, freq=None, how: str = "start") -> DatetimeIndex:
        # [FUNCTION DOC]
        # Convert the periods to timestamps. Parameters: freq is the target frequency of
        # the result (default none, uses the period frequency); how is which end of each
        # period to use, defaulting to the start. Delegates to the array and returns a
        # DatetimeIndex with the same name. Documentation is shared from the array
        # method.
        arr = self._data.to_timestamp(freq, how)
        return DatetimeIndex._simple_new(arr, name=self.name)

    @property
    @doc(PeriodArray.hour.fget)
    def hour(self) -> Index:
        # [FUNCTION DOC]
        # Property: the hour of each period as a generic index (not a period index),
        # carrying this index's name and without copying. Documentation shared from the
        # array property.
        return Index(self._data.hour, name=self.name, copy=False)

    @property
    @doc(PeriodArray.minute.fget)
    def minute(self) -> Index:
        # [FUNCTION DOC]
        # Property: the minute of each period as a generic index, with this index's name
        # and no copy. Documentation shared from the array property.
        return Index(self._data.minute, name=self.name, copy=False)

    @property
    @doc(PeriodArray.second.fget)
    def second(self) -> Index:
        # [FUNCTION DOC]
        # Property: the second of each period as a generic index, with this index's name
        # and no copy. Documentation shared from the array property.
        return Index(self._data.second, name=self.name, copy=False)

    # ------------------------------------------------------------------------
    # Index Constructors

    def __new__(
        cls,
        data=None,
        freq=None,
        dtype: Dtype | None = None,
        copy: bool | None = None,
        name: Hashable | None = None,
    ) -> Self:
        # [FUNCTION DOC]
        # Construct a PeriodIndex. Parameters: data is array-like period-like input
        # (default none); freq is a period string or offset (default none); dtype is a
        # dtype or string from which a frequency can be extracted (default none); copy
        # is a tri-state flag (none means copy for array input only, false/true force
        # the behavior); name is the index name (default none, taken from data if it has
        # one). Returns a new index of cls.
        # Steps: if copying is not requested and the input is an index or series, its
        # reference tracker is kept for copy-on-write; the name is resolved from the
        # data when not given; the frequency is validated against the dtype (raising on
        # conflict); the copy flag is resolved by the shared array-input helper (GH#63388);
        # when the input is already a PeriodIndex with a different frequency it is first
        # converted with the frequency-conversion method; the data is converted to a
        # PeriodArray; it is copied if the resolved flag asks for it; the instance is
        # built with the private simple constructor.
        # Raises a value error when data is a plain list and neither freq nor dtype
        # supplies a frequency (raised from the conversion helpers).
        refs = None
        if not copy and isinstance(data, (Index, ABCSeries)):
            refs = data._references

        name = maybe_extract_name(name, data, cls)

        freq = validate_dtype_freq(dtype, freq)

        # GH#63388
        data, copy = cls._maybe_copy_array_input(data, copy, dtype)

        # PeriodIndex allow PeriodIndex(period_index, freq=different)
        # Let's not encourage that kind of behavior in PeriodArray.

        if freq and isinstance(data, cls) and data.freq != freq:
            # TODO: We can do some of these with no-copy / coercion?
            # e.g. D -> 2D seems to be OK
            data = data.asfreq(freq)

        # don't pass copy here, since we copy later.
        data = period_array(data=data, freq=freq)

        if copy:
            data = data.copy()

        return cls._simple_new(data, name=name, refs=refs)

    @classmethod
    def from_fields(
        cls,
        *,
        year=None,
        quarter=None,
        month=None,
        day=None,
        hour=None,
        minute=None,
        second=None,
        freq=None,
    ) -> Self:
        # [FUNCTION DOC]
        # Class method building a PeriodIndex from calendar fields. All parameters are
        # keyword-only and default to none: year, quarter, month, day, hour, minute,
        # second (each an int, array or series) and freq (string or offset). Fields that
        # are none are dropped, and the rest are converted with the array's private
        # field-based constructor (which validates the combination and the frequency and
        # may raise a value error). Returns a new unnamed index.
        """
        Construct a PeriodIndex from fields (year, month, day, etc.).

        Parameters
        ----------
        year : int, array, or Series, default None
            Year for the PeriodIndex.
        quarter : int, array, or Series, default None
            Quarter for the PeriodIndex.
        month : int, array, or Series, default None
            Month for the PeriodIndex.
        day : int, array, or Series, default None
            Day for the PeriodIndex.
        hour : int, array, or Series, default None
            Hour for the PeriodIndex.
        minute : int, array, or Series, default None
            Minute for the PeriodIndex.
        second : int, array, or Series, default None
            Second for the PeriodIndex.
        freq : str or period object, optional
            One of pandas period strings or corresponding objects.

        Returns
        -------
        PeriodIndex

        See Also
        --------
        PeriodIndex.from_ordinals : Construct a PeriodIndex from ordinals.
        PeriodIndex.to_timestamp : Cast to DatetimeArray/Index.

        Examples
        --------
        >>> idx = pd.PeriodIndex.from_fields(year=[2000, 2002], quarter=[1, 3])
        >>> idx
        PeriodIndex(['2000Q1', '2002Q3'], dtype='period[Q-DEC]')
        """
        fields = {
            "year": year,
            "quarter": quarter,
            "month": month,
            "day": day,
            "hour": hour,
            "minute": minute,
            "second": second,
        }
        fields = {key: value for key, value in fields.items() if value is not None}
        arr = PeriodArray._from_fields(fields=fields, freq=freq)
        return cls._simple_new(arr)

    @classmethod
    def from_ordinals(cls, ordinals, *, freq, name=None) -> Self:
        # [FUNCTION DOC]
        # Class method building a PeriodIndex from integer ordinals (offsets from the
        # proleptic Gregorian epoch). Parameters: ordinals is array-like of ints, cast
        # to 64-bit integers; freq (required, keyword-only) is a period string or offset
        # used to build the period dtype; name is the index name (default none). Returns
        # a new index. No validation of the ordinal range is done beyond the integer
        # cast.
        """
        Construct a PeriodIndex from ordinals.

        Parameters
        ----------
        ordinals : array-like of int
            The period offsets from the proleptic Gregorian epoch.
        freq : str or period object
            One of pandas period strings or corresponding objects.
        name : str, default None
            Name of the resulting PeriodIndex.

        Returns
        -------
        PeriodIndex

        See Also
        --------
        PeriodIndex.from_fields : Construct a PeriodIndex from fields
            (year, month, day, etc.).
        PeriodIndex.to_timestamp : Cast to DatetimeArray/Index.

        Examples
        --------
        >>> idx = pd.PeriodIndex.from_ordinals([-1, 0, 1], freq="Q")
        >>> idx
        PeriodIndex(['1969Q4', '1970Q1', '1970Q2'], dtype='period[Q-DEC]')
        """
        ordinals = np.asarray(ordinals, dtype=np.int64)
        dtype = PeriodDtype(freq)
        data = PeriodArray._simple_new(ordinals, dtype=dtype)
        return cls._simple_new(data, name=name)

    # ------------------------------------------------------------------------
    # Data

    @property
    def values(self) -> npt.NDArray[np.object_]:
        # [FUNCTION DOC]
        # Property: the values as a numpy object array of Period scalars (the data is
        # boxed element by element).
        return np.asarray(self, dtype=object)

    def _maybe_convert_timedelta(self, other) -> int | npt.NDArray[np.int64]:
        # [FUNCTION DOC]
        # HELPER: converts a timedelta-like operand into an integer count of this index's
        # frequency units, for arithmetic and tolerance handling. Parameter other may be
        # a timedelta, numpy timedelta, offset, integer, or numpy array of timedeltas.
        # Returns an integer or an int64 array. Rules: timedelta-like inputs (including
        # Tick and arrays) are checked against the frequency by the array (only when the
        # index frequency is a fixed-size tick or a day; otherwise they fall through to
        # the final error); a base offset must have the same base as the frequency and
        # then its multiplier is returned, else an incompatible-frequency error (a
        # value error subclass) is raised; an integer is returned as is; anything else
        # raises the incompatible-frequency error for an input without a frequency.
        """
        Convert timedelta-like input to an integer multiple of self.freq

        Parameters
        ----------
        other : timedelta, np.timedelta64, DateOffset, int, np.ndarray

        Returns
        -------
        converted : int, np.ndarray[int64]

        Raises
        ------
        IncompatibleFrequency : if the input cannot be written as a multiple
            of self.freq.  Note IncompatibleFrequency subclasses ValueError.
        """
        if isinstance(other, (timedelta, np.timedelta64, Tick, np.ndarray)):
            if isinstance(self.freq, (Tick, Day)):
                # _check_timedeltalike_freq_compat will raise if incompatible
                delta = self._data._check_timedeltalike_freq_compat(other)
                return delta
        elif isinstance(other, BaseOffset):
            if other.base == self.freq.base:
                return other.n

            raise raise_on_incompatible(self, other)
        elif is_integer(other):
            assert isinstance(other, int)
            return other

        # raise when input doesn't have freq
        raise raise_on_incompatible(self, None)

    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        # [FUNCTION DOC]
        # HELPER: comparability check used by the base index machinery. Returns true only
        # when the given dtype equals this index's period dtype (including frequency).
        """
        Can we compare values of the given dtype to our own?
        """
        return self.dtype == dtype

    # ------------------------------------------------------------------------
    # Index Methods

    def asof_locs(self, where: Index, mask: npt.NDArray[np.bool_]) -> np.ndarray:
        # [FUNCTION DOC]
        # Find, for each label in where, the position of the last valid (non-missing)
        # entry at or before it. Parameters: where is a DatetimeIndex or PeriodIndex of
        # labels (a DatetimeIndex is converted to a PeriodIndex of this frequency
        # without copying); mask is a boolean array that is true where this index's data
        # is not missing. Raises a type error for any other type of where. Otherwise
        # returns the integer array computed by the base class implementation.
        """
        where : array of timestamps
        mask : np.ndarray[bool]
            Array of booleans where data is not NA.
        """
        if isinstance(where, DatetimeIndex):
            where = PeriodIndex(where._values, freq=self.freq, copy=False)
        elif not isinstance(where, PeriodIndex):
            raise TypeError("asof_locs `where` must be DatetimeIndex or PeriodIndex")

        return super().asof_locs(where, mask)

    @property
    def is_full(self) -> bool:
        # [FUNCTION DOC]
        # Property: true when the index is empty or when consecutive ordinals differ by
        # less than two, that is, no period between the first and last is missing.
        # Raises a value error if the index is not monotonic increasing and nonempty.
        """
        Returns True if this PeriodIndex is range-like in that all Periods
        between start and end are present, in order.
        """
        if len(self) == 0:
            return True
        if not self.is_monotonic_increasing:
            raise ValueError("Index is not monotonic")
        values = self.asi8
        return bool(((values[1:] - values[:-1]) < 2).all())

    @property
    def inferred_type(self) -> str:
        # [FUNCTION DOC]
        # Property: constant string "period". The comment records that since the data is
        # stored as integers this keeps indexing from being ambiguous.
        # b/c data is represented as ints make sure we can't have ambiguous
        # indexing
        return "period"

    # ------------------------------------------------------------------------
    # Indexing Methods

    def _convert_tolerance(self, tolerance, target):
        # [FUNCTION DOC]
        # HELPER: converts a user tolerance for nearest-style lookups into units
        # comparable to the int64 engine target. Parameters: tolerance (scalar or
        # array-like) and target (the index being matched). First applies the base
        # conversion; when the target has the same dtype as this index, the result is
        # further converted into an integer multiple of the frequency via the timedelta
        # conversion helper (which can raise the incompatible-frequency error). Returns
        # the converted tolerance.
        # Returned tolerance must be in dtype/units so that
        #  `|self._get_engine_target() - target._engine_target()| <= tolerance`
        #  is meaningful.  Since PeriodIndex returns int64 for engine_target,
        #  we may need to convert timedelta64 tolerance to int64.
        tolerance = super()._convert_tolerance(tolerance, target)

        if self.dtype == target.dtype:
            # convert tolerance to i8
            tolerance = self._maybe_convert_timedelta(tolerance)

        return tolerance

    def get_loc(self, key):
        # [FUNCTION DOC]
        # Return the integer location of a label. Parameter key may be a Period, a
        # missing-value marker, a parsable string or a datetime. Returns an integer, a
        # slice or a boolean/integer array as the base lookup yields (strings with lower
        # resolution than the index return a partial-slice result). Raises a key error
        # when the key is absent, a string cannot be parsed as a period, a string has a
        # coarser resolution that cannot partially slice, a string has a finer resolution
        # than the index, the key is a Period with a different dtype, a datetime cannot
        # be turned into a Period, or the key is of any other type (including integers).
        # Raises a type error for list-like or unhashable keys (from the indexing-error
        # check). Missing values are normalized to the NaT marker before lookup. Errors
        # from the base lookup are re-raised as a key error carrying the original key.
        """
        Get integer location for requested label.

        Parameters
        ----------
        key : Period, NaT, str, or datetime
            String or datetime key must be parsable as Period.

        Returns
        -------
        loc : int or ndarray[int64]

        Raises
        ------
        KeyError
            Key is not present in the index.
        TypeError
            If key is listlike or otherwise not hashable.
        """
        orig_key = key

        self._check_indexing_error(key)

        if is_valid_na_for_dtype(key, self.dtype):
            key = NaT

        elif isinstance(key, str):
            try:
                parsed, reso = self._parse_with_reso(key)
            except ValueError as err:
                # A string with invalid format
                raise KeyError(f"Cannot interpret '{key}' as period") from err

            if self._can_partial_date_slice(reso):
                try:
                    return self._partial_date_slice(reso, parsed)
                except KeyError as err:
                    raise KeyError(key) from err

            if reso == self._resolution_obj:
                # the reso < self._resolution_obj case goes
                #  through _get_string_slice
                key = self._cast_partial_indexing_scalar(parsed)
            else:
                raise KeyError(key)

        elif isinstance(key, Period):
            self._disallow_mismatched_indexing(key)

        elif isinstance(key, datetime):
            key = self._cast_partial_indexing_scalar(key)

        else:
            # in particular integer, which Period constructor would cast to string
            raise KeyError(key)

        try:
            return Index.get_loc(self, key)
        except KeyError as err:
            raise KeyError(orig_key) from err

    def _disallow_mismatched_indexing(self, key: Period) -> None:
        # [FUNCTION DOC]
        # HELPER: raises a key error carrying the key if the Period's dtype (frequency)
        # differs from this index's dtype; returns nothing otherwise.
        if key._dtype != self.dtype:
            raise KeyError(key)

    def _cast_partial_indexing_scalar(self, label: datetime) -> Period:
        # [FUNCTION DOC]
        # HELPER: converts a datetime label into a Period at this index's frequency.
        # Raises a key error carrying the label if the conversion fails with a value
        # error. Returns the Period.
        try:
            period = Period(label, freq=self.freq)
        except ValueError as err:
            # we cannot construct the Period
            raise KeyError(label) from err
        return period

    @doc(DatetimeIndexOpsMixin._maybe_cast_slice_bound)
    def _maybe_cast_slice_bound(self, label, side: str):
        # [FUNCTION DOC]
        # HELPER: slice-bound casting hook (documentation shared from the datetime-like
        # mixin). Parameters: label and side (a string for which end of a slice, as
        # used by the base class). A datetime label is first converted to a Period at
        # this frequency; the base class then does the rest and its result is returned.
        if isinstance(label, datetime):
            label = self._cast_partial_indexing_scalar(label)

        return super()._maybe_cast_slice_bound(label, side)

    def _parsed_string_to_bounds(self, reso: Resolution, parsed: datetime):
        # [FUNCTION DOC]
        # HELPER: converts a parsed string and its resolution into inclusive bounds of
        # partial string slicing. Parameters: reso is the resolution of the string;
        # parsed is the parsed datetime. The resolution abbreviation is mapped to a
        # period frequency string (falling back to the abbreviation itself), a Period is
        # made at that frequency, and it is converted to this index's frequency to get
        # the start and end. Returns a pair of Periods (lower, upper).
        freq = OFFSET_TO_PERIOD_FREQSTR.get(reso.attr_abbrev, reso.attr_abbrev)
        iv = Period(parsed, freq=freq)
        return (iv.asfreq(self.freq, how="start"), iv.asfreq(self.freq, how="end"))

    @doc(DatetimeIndexOpsMixin.shift)
    def shift(self, periods: int = 1, freq=None) -> Self:
        # [FUNCTION DOC]
        # Shift every period forward by a number of periods (documentation shared from
        # the datetime-like mixin). Parameters: periods is an integer, default 1; freq
        # must be left as none. Raises a type error if freq is given. Returns a new
        # index obtained by adding the integer to this index.
        if freq is not None:
            raise TypeError(
                f"`freq` argument is not supported for {type(self).__name__}.shift"
            )
        return self + periods


@set_module("pandas")
def period_range(
    start=None,
    end=None,
    periods: int | None = None,
    freq=None,
    name: Hashable | None = None,
) -> PeriodIndex:
    # [FUNCTION DOC]
    # Create a PeriodIndex with regular frequency. Parameters: start and end are
    # bounds (string, datetime, date, Timestamp or period-like, default none);
    # periods is the number of periods (default none); freq is a string or offset
    # (default none, meaning daily unless start or end is a Period, in which case
    # the array generator takes the frequency from it); name is the result's name
    # (default none). Raises a value error unless exactly two of start, end and
    # periods are given. Generates the ordinals and frequency through the array's
    # range generator, wraps them into a PeriodArray with the period dtype, and
    # returns a PeriodIndex without copying.
    """
    Return a fixed frequency PeriodIndex.

    The day (calendar) is the default frequency.

    Parameters
    ----------
    start : str, datetime, date, pandas.Timestamp, or period-like, default None
        Left bound for generating periods.
    end : str, datetime, date, pandas.Timestamp, or period-like, default None
        Right bound for generating periods.
    periods : int, default None
        Number of periods to generate.
    freq : str or DateOffset, optional
        Frequency alias. By default the freq is taken from `start` or `end`
        if those are Period objects. Otherwise, the default is ``"D"`` for
        daily frequency.
    name : str, default None
        Name of the resulting PeriodIndex.

    Returns
    -------
    PeriodIndex
        A PeriodIndex of fixed frequency periods.

    See Also
    --------
    date_range : Returns a fixed frequency DatetimeIndex.
    Period : Represents a period of time.
    PeriodIndex : Immutable ndarray holding ordinal values indicating regular periods
        in time.

    Notes
    -----
    Of the three parameters: ``start``, ``end``, and ``periods``, exactly two
    must be specified.

    To learn more about the frequency strings, please see
    :ref:`this link<timeseries.offset_aliases>`.

    Examples
    --------
    >>> pd.period_range(start="2017-01-01", end="2018-01-01", freq="M")
    PeriodIndex(['2017-01', '2017-02', '2017-03', '2017-04', '2017-05', '2017-06',
             '2017-07', '2017-08', '2017-09', '2017-10', '2017-11', '2017-12',
             '2018-01'],
            dtype='period[M]')

    If ``start`` or ``end`` are ``Period`` objects, they will be used as anchor
    endpoints for a ``PeriodIndex`` with frequency matching that of the
    ``period_range`` constructor.

    >>> pd.period_range(
    ...     start=pd.Period("2017Q1", freq="Q"),
    ...     end=pd.Period("2017Q2", freq="Q"),
    ...     freq="M",
    ... )
    PeriodIndex(['2017-03', '2017-04', '2017-05', '2017-06'],
                dtype='period[M]')
    """
    if com.count_not_none(start, end, periods) != 2:
        raise ValueError(
            "Of the three parameters: start, end, and periods, "
            "exactly two must be specified"
        )
    if freq is None and (not isinstance(start, Period) and not isinstance(end, Period)):
        freq = "D"

    data, freq = PeriodArray._generate_range(start, end, periods, freq)
    dtype = PeriodDtype(freq)
    data = PeriodArray(data, dtype=dtype)
    return PeriodIndex(data, name=name, copy=False)
