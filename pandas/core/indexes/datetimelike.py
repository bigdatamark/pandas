"""
Base and utility classes for tseries type pandas objects.
"""

# [MODULE DOC]
# Shared base classes for the datetime-like index types. Public contents:
# DatetimeIndexOpsMixin (behavior common to datetime, timedelta and period
# indexes: frequency access, equality, containment, partial-string slicing
# support, label casting) and DatetimeTimedeltaMixin (extra behavior shared by
# datetime and timedelta indexes but not period indexes: unit conversion,
# shifting, frequency-preserving set operations, joins, insert, delete, take).
# Both build on the array-backed extension index base and delegate storage to
# the datetime/timedelta/period arrays. The concrete index modules for each
# type subclass these mixins. Also defines a private copy of the shared index
# doc-format keyword dictionary taken from the base index module. No other
# import-time side effects.

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
    cast,
    final,
)

import numpy as np

from pandas._libs import (
    NaT,
    lib,
)
from pandas._libs.tslibs import (
    BaseOffset,
    Resolution,
    Tick,
    Timedelta,
    Timestamp,
    parsing,
    to_offset,
)
from pandas._libs.tslibs.dtypes import abbrev_to_npy_unit
from pandas.compat.numpy import function as nv
from pandas.errors import (
    InvalidIndexError,
    NullFrequencyError,
    OutOfBoundsDatetime,
    OutOfBoundsTimedelta,
)
from pandas.util._decorators import (
    cache_readonly,
)

from pandas.core.dtypes.common import (
    is_integer,
    is_list_like,
)
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    PeriodDtype,
)

from pandas.core.arrays import (
    DatetimeArray,
    ExtensionArray,
    PeriodArray,
    TimedeltaArray,
)
import pandas.core.common as com
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
    Index,
)
from pandas.core.indexes.extension import NDArrayBackedExtensionIndex
from pandas.core.indexes.range import RangeIndex
from pandas.core.tools.timedeltas import to_timedelta

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from pandas._typing import (
        Axis,
        JoinHow,
        TimeUnit,
        npt,
    )

    from pandas import CategoricalIndex

_index_doc_kwargs = dict(ibase._index_doc_kwargs)


class DatetimeIndexOpsMixin(NDArrayBackedExtensionIndex, ABC):
    # [CLASS DOC]
    # Abstract mixin giving a unified interface to datetime-like indexes
    # (datetime, timedelta, period). Inherits the array-backed extension
    # index; the wrapped data is a datetime, timedelta or period array and
    # most behavior forwards to it. Does not hold strings. Subclasses must
    # provide the resolution object property and the bounds parsing hook used
    # for partial string indexing. Key members: mean, the frequency
    # properties, equality, containment, label-to-bound casting, shift hook.
    """
    Common ops mixin to support a unified interface datetimelike Index.
    """

    _can_hold_strings = False
    _data: DatetimeArray | TimedeltaArray | PeriodArray

    def mean(self, *, skipna: bool = True, axis: int | None = 0):
        # [FUNCTION DOC]
        # Mean of the values. Keyword-only parameters: skipna (bool, default
        # true) ignores missing-time entries; axis (int or none, default 0)
        # is the axis to reduce. Delegates to the underlying array's mean and
        # returns a Timestamp or Timedelta scalar (or missing-time marker).
        # Not defined for period data (the array raises).
        """
        Return the mean value of the Array.

        Parameters
        ----------
        skipna : bool, default True
            Whether to ignore any NaT elements.
        axis : int, optional, default 0
            Axis for the function to be applied on.

        Returns
        -------
        scalar
            Timestamp or Timedelta.

        See Also
        --------
        numpy.ndarray.mean : Returns the average of array elements along a given axis.
        Series.mean : Return the mean value in a Series.

        Notes
        -----
        mean is only defined for Datetime and Timedelta dtypes, not for Period.

        Examples
        --------
        For :class:`pandas.DatetimeIndex`:

        >>> idx = pd.date_range("2001-01-01 00:00", periods=3)
        >>> idx
        DatetimeIndex(['2001-01-01', '2001-01-02', '2001-01-03'],
                      dtype='datetime64[us]', freq='D')
        >>> idx.mean()
        Timestamp('2001-01-02 00:00:00')

        For :class:`pandas.TimedeltaIndex`:

        >>> tdelta_idx = pd.to_timedelta([1, 2, 3], unit="D")
        >>> tdelta_idx
        TimedeltaIndex(['1 days', '2 days', '3 days'],
                        dtype='timedelta64[s]', freq=None)
        >>> tdelta_idx.mean()
        Timedelta('2 days 00:00:00')
        """
        return self._data.mean(skipna=skipna, axis=axis)

    @property
    def freq(self) -> BaseOffset | None:
        # [FUNCTION DOC]
        # Property getter. Returns the frequency offset stored on the
        # underlying array, or none if no frequency is set.
        """
        Return the frequency object if it is set, otherwise None.

        To learn more about the frequency strings, please see
        :ref:`this link<timeseries.offset_aliases>`.

        See Also
        --------
        DatetimeIndex.freq : Return the frequency object if it is set, otherwise None.
        PeriodIndex.freq : Return the frequency object if it is set, otherwise None.

        Examples
        --------
        >>> datetimeindex = pd.date_range(
        ...     "2022-02-22 02:22:22", periods=10, tz="America/Chicago", freq="h"
        ... )
        >>> datetimeindex
        DatetimeIndex(['2022-02-22 02:22:22-06:00', '2022-02-22 03:22:22-06:00',
                       '2022-02-22 04:22:22-06:00', '2022-02-22 05:22:22-06:00',
                       '2022-02-22 06:22:22-06:00', '2022-02-22 07:22:22-06:00',
                       '2022-02-22 08:22:22-06:00', '2022-02-22 09:22:22-06:00',
                       '2022-02-22 10:22:22-06:00', '2022-02-22 11:22:22-06:00'],
                      dtype='datetime64[us, America/Chicago]', freq='h')
        >>> datetimeindex.freq
        <Hour>
        """
        return self._data.freq

    @freq.setter
    def freq(self, value) -> None:
        # [FUNCTION DOC]
        # Property setter. Parameter value is the new frequency (offset,
        # string or none). Assigns it to the underlying array's frequency,
        # which validates it (the array may raise if it does not conform to
        # the values). Mutates the shared array in place; returns nothing.
        # error: Property "freq" defined in "PeriodArray" is read-only  [misc]
        self._data.freq = value  # type: ignore[misc]

    @property
    def asi8(self) -> npt.NDArray[np.int64]:
        # [FUNCTION DOC]
        # Property. Returns the values as a 64-bit integer array of the
        # underlying storage (units since the epoch for the array's unit,
        # with the sentinel minimum integer for missing entries).
        return self._data.asi8

    @property
    def freqstr(self) -> str:
        # [FUNCTION DOC]
        # Property. Returns the frequency as a string, or none when there is
        # no frequency. For period data (when the array has a frequency
        # string) the string is derived from the period dtype's frequency
        # string; otherwise the array's own frequency string is returned.
        # KNOWN ISSUE: the return annotation says string only, though none
        # is returned when no frequency is set (a type-ignore hides this).
        """
        Return the frequency object as a string if it's set, otherwise None.

        See Also
        --------
        DatetimeIndex.inferred_freq : Returns a string representing a frequency
            generated by infer_freq.

        Examples
        --------
        For DatetimeIndex:

        >>> idx = pd.DatetimeIndex(["1/1/2020 10:00:00+00:00"], freq="D")
        >>> idx.freqstr
        'D'

        The frequency can be inferred if there are more than 2 points:

        >>> idx = pd.DatetimeIndex(
        ...     ["2018-01-01", "2018-01-03", "2018-01-05"], freq="infer"
        ... )
        >>> idx.freqstr
        '2D'

        For PeriodIndex:

        >>> idx = pd.PeriodIndex(["2023-1", "2023-2", "2023-3"], freq="M")
        >>> idx.freqstr
        'M'
        """
        from pandas import PeriodIndex

        if self._data.freqstr is not None and isinstance(
            self._data, (PeriodArray, PeriodIndex)
        ):
            freq = PeriodDtype(self._data.freq)._freqstr
            return freq
        else:
            return self._data.freqstr  # type: ignore[return-value]

    @cache_readonly
    @abstractmethod
    def _resolution_obj(self) -> Resolution:
        # [FUNCTION DOC]
        # HELPER: abstract cached property that subclasses implement to
        # return the Resolution object describing the finest unit present
        # (used for partial string slicing). This stub has no behavior.
        ...

    @cache_readonly
    def resolution(self) -> str:
        # [FUNCTION DOC]
        # Cached property. Returns the name of the coarsest unit that
        # resolves all values (day, hour, minute, second, millisecond,
        # microsecond, nanosecond), as reported by the underlying array.
        """
        Returns day, hour, minute, second, millisecond or microsecond
        """
        return self._data.resolution

    # ------------------------------------------------------------------------

    @cache_readonly
    def hasnans(self) -> bool:
        # [FUNCTION DOC]
        # Cached property. Returns true if any element is missing (NaT),
        # using the array's cached missing-value flag.
        return self._data._hasna

    def equals(self, other: Any) -> bool:
        # [FUNCTION DOC]
        # Element-wise equality with other (any object). Returns true if
        # other is the identical object. False if other is not an Index or
        # has a numeric (integer, unsigned, float, complex) dtype. If other
        # is a different index class, it is converted to this class when it
        # is object dtype or categorical whose (inferred) contents are of a
        # kind convertible to this type; conversion failures (value, type or
        # overflow errors) give false. Different final classes give false.
        # Equal dtypes compare the integer representations. Datetimes with
        # the same timezone, or any timedeltas, with differing units are
        # compared after casting both to a common unit (out-of-bounds gives
        # false). All other cases give false.
        """
        Determines if two Index objects contain the same elements.
        """
        if self.is_(other):
            return True

        if not isinstance(other, Index):
            return False
        elif other.dtype.kind in "iufc":
            return False
        elif not isinstance(other, type(self)):
            should_try = False
            inferable = self._data._infer_matches
            if other.dtype == object:
                should_try = other.inferred_type in inferable
            elif isinstance(other.dtype, CategoricalDtype):
                other = cast("CategoricalIndex", other)
                should_try = other.categories.inferred_type in inferable

            if should_try:
                try:
                    other = type(self)(other)
                except (ValueError, TypeError, OverflowError):
                    # e.g.
                    #  ValueError -> cannot parse str entry, or OutOfBoundsDatetime
                    #  TypeError  -> trying to convert IntervalIndex to DatetimeIndex
                    #  OverflowError -> Index([very_large_timedeltas])
                    return False

        if type(self) != type(other):
            return False
        elif self.dtype == other.dtype:
            return np.array_equal(self.asi8, other.asi8)
        elif (self.dtype.kind == "M" and self.tz == other.tz) or self.dtype.kind == "m":  # type: ignore[attr-defined]
            # different units, otherwise matching
            try:
                # TODO: do this at the EA level?
                left, right = self._data._ensure_matching_resos(other._data)  # type: ignore[union-attr]
            except (OutOfBoundsDatetime, OutOfBoundsTimedelta):
                return False
            else:
                return np.array_equal(left.view("i8"), right.view("i8"))
        return False

    def __contains__(self, key: Any) -> bool:
        # [FUNCTION DOC]
        # Membership test for a label key. Hashes the key first (raising a
        # type error if unhashable), then tries a label location lookup;
        # returns false if that lookup raises a key, type, value or
        # invalid-indexer error, else true. Partial string keys therefore
        # count as members if they match any element.
        # KNOWN ISSUE: the docstring example uses a plain integer Index,
        # not a datetime-like index.
        """
        Return a boolean indicating whether the provided key is in the index.

        Parameters
        ----------
        key : label
            The key to check if it is present in the index.

        Returns
        -------
        bool
            Whether the key search is in the index.

        Raises
        ------
        TypeError
            If the key is not hashable.

        See Also
        --------
        Index.isin : Returns an ndarray of boolean dtype indicating whether the
            list-like key is in the index.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3, 4])
        >>> idx
        Index([1, 2, 3, 4], dtype='int64')
        >>> 2 in idx
        True
        >>> 6 in idx
        False
        """
        hash(key)
        try:
            self.get_loc(key)
        except (KeyError, TypeError, ValueError, InvalidIndexError):
            return False
        return True

    def _convert_tolerance(self, tolerance, target):
        # [FUNCTION DOC]
        # HELPER: for nearest-match indexer lookups. Converts tolerance
        # (scalar or list-like of time-delta-like values) into a numpy array
        # of timedeltas, then defers to the base implementation (which
        # validates its size against target and may raise a value error).
        # Returns the validated tolerance array.
        tolerance = np.asarray(to_timedelta(tolerance).to_numpy())
        return super()._convert_tolerance(tolerance, target)

    # --------------------------------------------------------------------
    # Rendering Methods
    _default_na_rep = "NaT"

    def _format_with_header(
        self, *, header: list[str], na_rep: str, date_format: str | None = None
    ) -> list[str]:
        # [FUNCTION DOC]
        # HELPER: for the repr. Keyword-only parameters: header is a list of
        # header strings, na_rep is the string for missing values, date_format
        # is an optional format string (default none). Returns the header
        # followed by the formatted element strings obtained from the
        # array's csv-style value formatting, without whitespace padding.
        # TODO: not reached in tests 2023-10-11
        # matches base class except for whitespace padding and date_format
        return header + list(
            self._get_values_for_csv(na_rep=na_rep, date_format=date_format)
        )

    @property
    def _formatter_func(self):
        # [FUNCTION DOC]
        # HELPER: property for the repr. Returns the per-element formatter
        # callable provided by the underlying array.
        return self._data._formatter()

    def _format_attrs(self):
        # [FUNCTION DOC]
        # HELPER: for the repr. Returns the base class's list of
        # (name, value) attribute pairs plus, if this class lists the
        # frequency among its attributes, a pair whose value is the quoted
        # frequency string (or none when unset).
        """
        Return a list of tuples of the (attr,formatted_value).
        """
        attrs = super()._format_attrs()
        for attrib in self._attributes:
            # iterating over _attributes prevents us from doing this for PeriodIndex
            if attrib == "freq":
                freq = self.freqstr
                if freq is not None:
                    freq = repr(freq)  # e.g. D -> 'D'
                attrs.append(("freq", freq))
        return attrs

    def _summary(self, name=None) -> str:
        # [FUNCTION DOC]
        # HELPER: for the info-style summary. Parameter name is an optional
        # label for the summary header. Returns the base summary text with an
        # extra line stating the frequency string appended when a frequency
        # is set.
        """
        Return a summarized representation.

        Parameters
        ----------
        name : str
            name to use in the summary representation

        Returns
        -------
        String with a summarized representation of the index
        """
        result = super()._summary(name=name)
        if self.freq:
            result += f"\nFreq: {self.freqstr}"

        return result

    # --------------------------------------------------------------------
    # Indexing Methods

    @final
    def _can_partial_date_slice(self, reso: Resolution) -> bool:
        # [FUNCTION DOC]
        # HELPER: final method. Parameter reso is the Resolution of a parsed
        # string label. Returns true when that resolution is strictly coarser
        # than this index's own resolution, meaning the label denotes a range
        # that can be partially sliced.
        # e.g. test_getitem_setitem_periodindex
        # History of conversation GH#3452, GH#3931, GH#2369, GH#14826
        return reso > self._resolution_obj
        # NB: for DTI/PI, not TDI

    def _parsed_string_to_bounds(self, reso: Resolution, parsed):
        # [FUNCTION DOC]
        # HELPER: abstract hook for subclasses. Given a resolution and the
        # parsed datetime, should return the (lower, upper) inclusive bounds
        # of the period it denotes. This base version always raises a
        # not-implemented error.
        raise NotImplementedError

    def _parse_with_reso(self, label: str) -> tuple[datetime, Resolution]:
        # [FUNCTION DOC]
        # HELPER: parses a string label into a datetime and its Resolution,
        # using this index's frequency string as a parsing hint. Parameter
        # label is the string (numpy string scalars are first converted to
        # builtin strings). Returns a pair (parsed datetime, Resolution).
        # Parser errors (date parse errors, a kind of value error) propagate.
        # KNOWN ISSUE: the frequency variable is only assigned when the
        # frequency is none or has a rule code (or when the property access
        # raises not-implemented); otherwise the later use would fail with an
        # unbound local variable error.
        # overridden by TimedeltaIndex
        try:
            if self.freq is None or hasattr(self.freq, "rule_code"):
                freq = self.freq
        except NotImplementedError:
            freq = getattr(self, "freqstr", getattr(self, "inferred_freq", None))

        freqstr: str | None
        if freq is not None and not isinstance(freq, str):
            freqstr = freq.rule_code
        else:
            freqstr = freq

        if isinstance(label, np.str_):
            # GH#45580
            label = str(label)

        parsed, reso_str = parsing.parse_datetime_string_with_reso(label, freqstr)
        reso = Resolution.from_attrname(reso_str)
        return parsed, reso

    def _get_string_slice(self, key: str) -> slice | npt.NDArray[np.intp]:
        # [FUNCTION DOC]
        # HELPER: partial-string indexing. Parses key into a datetime and
        # resolution, then returns the partial-date slice (a slice for
        # monotonic data, else an integer position array). A key error from
        # the slicing is re-raised as a key error carrying the original key.
        # overridden by TimedeltaIndex
        parsed, reso = self._parse_with_reso(key)
        try:
            return self._partial_date_slice(reso, parsed)
        except KeyError as err:
            raise KeyError(key) from err

    @final
    def _partial_date_slice(
        self,
        reso: Resolution,
        parsed: datetime,
    ) -> slice | npt.NDArray[np.intp]:
        # [FUNCTION DOC]
        # HELPER: final method selecting the elements that fall within the
        # period denoted by a parsed label. Parameters: reso (Resolution of
        # the label) and parsed (the parsed datetime). Raises a value error
        # when the resolution is not coarser than the index's. Computes the
        # lower and upper bounds via the subclass hook. For a monotonic
        # increasing index: raises a key error if both bounds lie wholly
        # before the first or after the last element, otherwise returns a
        # slice found by binary search (left of lower, right of upper). For
        # other indexes returns the integer positions whose values lie
        # between the bounds inclusive (possibly empty).
        # KNOWN ISSUE: the docstring lists no raised exceptions, but a value
        # error and a key error can be raised.
        """
        Parameters
        ----------
        reso : Resolution
        parsed : datetime

        Returns
        -------
        slice or ndarray[intp]
        """
        if not self._can_partial_date_slice(reso):
            raise ValueError

        t1, t2 = self._parsed_string_to_bounds(reso, parsed)
        vals = self._data._ndarray
        unbox = self._data._unbox

        if self.is_monotonic_increasing:
            if len(self) and (
                (t1 < self[0] and t2 < self[0]) or (t1 > self[-1] and t2 > self[-1])
            ):
                # we are out of range
                raise KeyError

            # TODO: does this depend on being monotonic _increasing_?

            # a monotonic (sorted) series can be sliced
            left = vals.searchsorted(unbox(t1), side="left")
            right = vals.searchsorted(unbox(t2), side="right")
            return slice(left, right)

        else:
            lhs_mask = vals >= unbox(t1)
            rhs_mask = vals <= unbox(t2)

            # try to find the dates
            return (lhs_mask & rhs_mask).nonzero()[0]

    def _maybe_cast_slice_bound(self, label, side: str):
        # [FUNCTION DOC]
        # HELPER: converts a slice endpoint. Parameters: label (any object)
        # and side (left or right, not validated here). A string label is
        # parsed to a resolution and mapped to the lower bound for left or
        # the upper bound for right of the period it denotes; parse failures
        # (value errors) are reported through the invalid-indexer error
        # (type error). Non-string labels that are not scalars recognized by
        # the array type also raise that invalid-indexer error. Otherwise the
        # label is returned unchanged.
        """
        If label is a string, cast it to scalar type according to resolution.

        Parameters
        ----------
        label : object
        side : {'left', 'right'}

        Returns
        -------
        label : object

        Notes
        -----
        Value of `side` parameter should be validated in caller.
        """
        if isinstance(label, str):
            try:
                parsed, reso = self._parse_with_reso(label)
            except ValueError as err:
                # DTI -> parsing.DateParseError
                # TDI -> 'unit abbreviation w/o a number'
                # PI -> string cannot be parsed as datetime-like
                self._raise_invalid_indexer("slice", label, err)

            lower, upper = self._parsed_string_to_bounds(reso, parsed)
            return lower if side == "left" else upper
        elif not isinstance(label, self._data._recognized_scalars):
            self._raise_invalid_indexer("slice", label)

        return label

    # --------------------------------------------------------------------
    # Arithmetic Methods

    def shift(self, periods: int = 1, freq=None) -> Self:
        # [FUNCTION DOC]
        # Abstract in this mixin: parameters periods (int, default 1) and
        # freq (offset, timedelta or string, default none) as documented;
        # always raises a not-implemented error. Subclasses provide behavior.
        """
        Shift index by desired number of time frequency increments.

        This method is for shifting the values of datetime-like indexes
        by a specified time increment a given number of times.

        Parameters
        ----------
        periods : int, default 1
            Number of periods (or increments) to shift by,
            can be positive or negative.
        freq : pandas.DateOffset, pandas.Timedelta or string, optional
            Frequency increment to shift by.
            If None, the index is shifted by its own `freq` attribute.
            Offset aliases are valid strings, e.g., 'D', 'W', 'M' etc.

        Returns
        -------
        pandas.DatetimeIndex
            Shifted index.

        See Also
        --------
        Index.shift : Shift values of Index.
        PeriodIndex.shift : Shift values of PeriodIndex.
        """
        raise NotImplementedError

    # --------------------------------------------------------------------

    def _maybe_cast_listlike_indexer(self, keyarr):
        # [FUNCTION DOC]
        # HELPER: list-like analogue of scalar label casting, used by indexer
        # computation. Tries to validate keyarr against the array type,
        # allowing object arrays; on a value or type error it falls back to
        # converting keyarr to an array (keeping extension arrays as they
        # are). Returns an Index of the result with its own dtype.
        """
        Analogue to maybe_cast_indexer for get_indexer instead of get_loc.
        """
        try:
            res = self._data._validate_listlike(keyarr, allow_object=True)
        except (ValueError, TypeError):
            if not isinstance(keyarr, ExtensionArray):
                # e.g. we don't want to cast DTA to ndarray[object]
                res = com.asarray_tuplesafe(keyarr)
                # TODO: com.asarray_tuplesafe shouldn't cast e.g. DatetimeArray
            else:
                res = keyarr
        return Index(res, dtype=res.dtype)


class DatetimeTimedeltaMixin(DatetimeIndexOpsMixin, ABC):
    # [CLASS DOC]
    # Abstract mixin for datetime and timedelta indexes (not period).
    # Extends the datetime-like mixin; the wrapped data is a datetime or
    # timedelta array. Comparable/attribute names are name and freq. Monotonic
    # and uniqueness checks are bound to the generic Index versions so that
    # frequency inference does not recurse. Adds unit conversion, shifting,
    # frequency-aware fast paths for intersection/union (including a range
    # index route for fixed-length ("tick") frequencies), join frequency
    # handling, and frequency-preserving delete, insert and take.
    """
    Mixin class for methods shared by DatetimeIndex and TimedeltaIndex,
    but not PeriodIndex
    """

    _data: DatetimeArray | TimedeltaArray
    _comparables = ["name", "freq"]
    _attributes = ["name", "freq"]

    # Compat for frequency inference, see GH#23789
    _is_monotonic_increasing = Index.is_monotonic_increasing
    _is_monotonic_decreasing = Index.is_monotonic_decreasing
    _is_unique = Index.is_unique

    @property
    def unit(self) -> TimeUnit:
        # [FUNCTION DOC]
        # Property. Returns the time unit string (s, ms, us or ns) of the
        # underlying array.
        return self._data.unit

    def as_unit(self, unit: TimeUnit) -> Self:
        # [FUNCTION DOC]
        # Parameter unit is one of s, ms, us, ns. Converts the underlying
        # array to that resolution and returns a new index of the same type
        # with the same name. The input is unchanged; overflow handling is
        # delegated to the array conversion (may raise out-of-bounds errors).
        """
        Convert to a dtype with the given unit resolution.

        This method is for converting the dtype of a ``DatetimeIndex`` or
        ``TimedeltaIndex`` to a new dtype with the given unit
        resolution/precision.

        Parameters
        ----------
        unit : {'s', 'ms', 'us', 'ns'}

        Returns
        -------
        same type as self
            Converted to the specified unit.

        See Also
        --------
        Timestamp.as_unit : Convert to the given unit.
        Timedelta.as_unit : Convert to the given unit.
        DatetimeIndex.as_unit : Convert to the given unit.
        TimedeltaIndex.as_unit : Convert to the given unit.

        Examples
        --------
        For :class:`pandas.DatetimeIndex`:

        >>> idx = pd.DatetimeIndex(["2020-01-02 01:02:03.004005006"])
        >>> idx
        DatetimeIndex(['2020-01-02 01:02:03.004005006'],
                      dtype='datetime64[ns]', freq=None)
        >>> idx.as_unit("s")
        DatetimeIndex(['2020-01-02 01:02:03'], dtype='datetime64[s]', freq=None)

        For :class:`pandas.TimedeltaIndex`:

        >>> tdelta_idx = pd.to_timedelta(["1 day 3 min 2 us 42 ns"])
        >>> tdelta_idx
        TimedeltaIndex(['1 days 00:03:00.000002042'],
                        dtype='timedelta64[ns]', freq=None)
        >>> tdelta_idx.as_unit("s")
        TimedeltaIndex(['1 days 00:03:00'], dtype='timedelta64[s]', freq=None)
        """
        arr = self._data.as_unit(unit)
        return type(self)._simple_new(arr, name=self.name)

    def _with_freq(self, freq):
        # [FUNCTION DOC]
        # HELPER: returns a new index of the same type and name whose data is
        # the array re-labelled with frequency freq (an offset, a string, the
        # infer marker, or none). Delegates validation to the array.
        arr = self._data._with_freq(freq)
        return type(self)._simple_new(arr, name=self._name)

    @property
    def values(self) -> np.ndarray:
        # [FUNCTION DOC]
        # Property. Returns a read-only view of the underlying datetime64 or
        # timedelta64 numpy array (timezone information is lost for
        # tz-aware data). The write flag is cleared on the view only.
        # NB: For Datetime64TZ this is lossy
        data = self._data._ndarray
        data = data.view()
        data.flags.writeable = False
        return data

    def shift(self, periods: int = 1, freq=None) -> Self:
        # [FUNCTION DOC]
        # Shifts the index. Parameters: periods (int, default 1; may be
        # negative) and freq (offset, timedelta or string; default none). If
        # freq is given and differs from the index's own, the string is
        # converted to an offset and the index plus periods times the offset
        # is returned. Otherwise, with zero periods or an empty index a copy
        # is returned. If no own frequency exists a null-frequency error is
        # raised. Otherwise the first and last values are shifted by periods
        # times the frequency and a new regular range between them is
        # generated (same unit, timezone inferred), returned as a new index of
        # the same type and name.
        """
        Shift index by desired number of time frequency increments.
        This method is for shifting the values of datetime-like indexes
        by a specified time increment a given number of times.

        Parameters
        ----------
        periods : int, default 1
            Number of periods (or increments) to shift by,
            can be positive or negative.
        freq : pandas.DateOffset, pandas.Timedelta or string, optional
            Frequency increment to shift by.
            If None, the index is shifted by its own `freq` attribute.
            Offset aliases are valid strings, e.g., 'D', 'W', 'M' etc.

        Returns
        -------
        pandas.DatetimeIndex
            Shifted index.

        See Also
        --------
        Index.shift : Shift values of Index.
        PeriodIndex.shift : Shift values of PeriodIndex.
        """
        if freq is not None and freq != self.freq:
            if isinstance(freq, str):
                freq = to_offset(freq)
            offset = periods * freq
            return self + offset

        if periods == 0 or len(self) == 0:
            # GH#14811 empty case
            return self.copy()

        if self.freq is None:
            raise NullFrequencyError("Cannot shift with no freq")

        start = self[0] + periods * self.freq
        end = self[-1] + periods * self.freq

        # Note: in the DatetimeTZ case, _generate_range will infer the
        #  appropriate timezone from `start` and `end`, so tz does not need
        #  to be passed explicitly.
        result = self._data._generate_range(
            start=start, end=end, periods=None, freq=self.freq, unit=self.unit
        )
        return type(self)._simple_new(result, name=self.name)

    @cache_readonly
    def inferred_freq(self) -> str | None:
        # [FUNCTION DOC]
        # Cached property. Returns the frequency string inferred from the
        # values by the array (via the frequency inference routine), or none
        # if it cannot be inferred.
        """
        Return the inferred frequency of the index.

        Returns
        -------
        str or None
            A string representing a frequency generated by ``infer_freq``.
            Returns ``None`` if the frequency cannot be inferred.

        See Also
        --------
        DatetimeIndex.freqstr : Return the frequency object as a string if it's set,
            otherwise ``None``.

        Examples
        --------
        For ``DatetimeIndex``:

        >>> idx = pd.DatetimeIndex(["2018-01-01", "2018-01-03", "2018-01-05"])
        >>> idx.inferred_freq
        '2D'

        For ``TimedeltaIndex``:

        >>> tdelta_idx = pd.to_timedelta(["0 days", "10 days", "20 days"])
        >>> tdelta_idx
        TimedeltaIndex(['0 days', '10 days', '20 days'],
                       dtype='timedelta64[us]', freq=None)
        >>> tdelta_idx.inferred_freq
        '10D'
        """
        return self._data.inferred_freq

    # --------------------------------------------------------------------
    # Set Operation Methods

    @cache_readonly
    def _as_range_index(self) -> RangeIndex:
        # [FUNCTION DOC]
        # HELPER: cached property for set operations. Requires a non-empty
        # index with a fixed-length (tick) frequency (caller must check).
        # Returns a RangeIndex over the integer representations from the
        # first to the last value, stepping by the frequency expressed in the
        # index's unit.
        # Convert our i8 representations to RangeIndex
        # Caller is responsible for checking isinstance(self.freq, Tick)
        freq = cast(Tick, self.freq)
        tick = Timedelta(freq).as_unit(self.unit)._value
        rng = range(self[0]._value, self[-1]._value + tick, tick)
        return RangeIndex(rng)

    def _can_range_setop(self, other) -> bool:
        # [FUNCTION DOC]
        # HELPER: returns true when both this index and other have
        # fixed-length (tick) frequencies, so set operations can run on range
        # indexes.
        return isinstance(self.freq, Tick) and isinstance(other.freq, Tick)

    def _wrap_range_setop(self, other, res_i8) -> Self:
        # [FUNCTION DOC]
        # HELPER: converts the integer-range result of a set operation back
        # to this index type. Parameters: other (the other operand) and
        # res_i8 (a RangeIndex or integer Index result). The new frequency is
        # this index's own when the result is empty, the range step expressed
        # as an offset when the result is a RangeIndex, else none. Builds a
        # new array with this dtype and that frequency from the integer
        # values viewed as the underlying datetime/timedelta storage, then
        # finalizes name/sort handling through the shared set-operation
        # wrapper. Returns the wrapped index.
        new_freq = None
        if not len(res_i8):
            # RangeIndex defaults to step=1, which we don't want.
            new_freq = self.freq
        elif isinstance(res_i8, RangeIndex):
            new_freq = to_offset(
                Timedelta(res_i8.step, unit=self.unit).as_unit(self.unit)
            )

        # TODO(GH#41493): we cannot just do
        #  type(self._data)(res_i8.values, dtype=self.dtype, freq=new_freq)
        # because test_setops_preserve_freq fails with _validate_frequency raising.
        # This raising is incorrect, as 'on_freq' is incorrect. This will
        # be fixed by GH#41493
        res_values = res_i8.values.view(self._data._ndarray.dtype)
        result = type(self._data)._simple_new(
            # error: Argument "dtype" to "_simple_new" of "DatetimeArray" has
            # incompatible type "Union[dtype[Any], ExtensionDtype]"; expected
            # "Union[dtype[datetime64], DatetimeTZDtype]"
            res_values,
            dtype=self.dtype,  # type: ignore[arg-type]
            freq=new_freq,  # type: ignore[arg-type]
        )
        return cast("Self", self._wrap_setop_result(other, result))

    def _range_intersect(self, other, sort) -> Self:
        # [FUNCTION DOC]
        # HELPER: intersection via range indexes. Converts both operands to
        # range indexes, intersects them using the given sort option, and
        # wraps the result back to this type. Both must have tick frequencies.
        # Dispatch to RangeIndex intersection logic.
        left = self._as_range_index
        right = other._as_range_index
        res_i8 = left.intersection(right, sort=sort)
        return self._wrap_range_setop(other, res_i8)

    def _range_union(self, other, sort) -> Self:
        # [FUNCTION DOC]
        # HELPER: union via range indexes. Converts both operands to range
        # indexes, takes their union with the given sort option, and wraps
        # the result back to this type. Both must have tick frequencies.
        # Dispatch to RangeIndex union logic.
        left = self._as_range_index
        right = other._as_range_index
        res_i8 = left.union(right, sort=sort)
        return self._wrap_range_setop(other, res_i8)

    def _intersection(self, other: Index, sort: bool = False) -> Index:
        # [FUNCTION DOC]
        # HELPER: implementation of intersection for same-dtype non-empty
        # operands. Parameters: other and sort (bool, default false). Uses the
        # range-index route if both have tick frequencies; else, if the fast
        # path is not allowed, runs the generic index intersection, wraps the
        # result, clears its frequency and re-infers it; else uses the fast
        # frequency-based path. Returns an index (frequency may be set).
        """
        intersection specialized to the case with matching dtypes and both non-empty.
        """
        other = cast("DatetimeTimedeltaMixin", other)

        if self._can_range_setop(other):
            return self._range_intersect(other, sort=sort)

        if not self._can_fast_intersect(other):
            result = Index._intersection(self, other, sort=sort)
            # We need to invalidate the freq because Index._intersection
            #  uses _shallow_copy on a view of self._data, which will preserve
            #  self.freq if we're not careful.
            # At this point we should have result.dtype == self.dtype
            #  and type(result) is type(self._data)
            result = self._wrap_setop_result(other, result)
            # error: "Index" has no attribute "_with_freq"; maybe "_with_infer"?
            return result._with_freq(None)._with_freq("infer")  # type: ignore[attr-defined]

        else:
            return self._fast_intersect(other, sort)

    def _fast_intersect(self, other, sort):
        # [FUNCTION DOC]
        # HELPER: intersection for two non-empty, same-frequency, ascending
        # regularly spaced indexes. The sort parameter is accepted but unused.
        # Orders the operands by first value; the overlap starts at the later
        # start and ends at the smaller of the two last values. Returns an
        # empty slice of self if there is no overlap, otherwise the
        # corresponding slice of the earlier-starting operand's values
        # (frequency is preserved via slicing).
        # to make our life easier, "sort" the two ranges
        if self[0] <= other[0]:
            left, right = self, other
        else:
            left, right = other, self

        # after sorting, the intersection always starts with the right index
        # and ends with the index of which the last elements is smallest
        end = min(left[-1], right[-1])
        start = right[0]

        if end < start:
            result = self[:0]
        else:
            lslice = slice(*left.slice_locs(start, end))
            result = left._values[lslice]

        return result

    def _can_fast_intersect(self, other: Self) -> bool:
        # [FUNCTION DOC]
        # HELPER: both operands must be non-empty. Returns true only when
        # this index has a frequency equal to other's, is monotonic
        # increasing, and the frequency multiple is exactly one; otherwise
        # false.
        # Note: we only get here with len(self) > 0 and len(other) > 0
        if self.freq is None:
            return False

        elif other.freq != self.freq:
            return False

        elif not self.is_monotonic_increasing:
            # Because freq is not None, we must then be monotonic decreasing
            return False

        # this along with matching freqs ensure that we "line up",
        #  so intersection will preserve freq
        # Note we are assuming away Ticks, as those go through _range_intersect
        # GH#42104
        return self.freq.n == 1

    def _can_fast_union(self, other: Self) -> bool:
        # [FUNCTION DOC]
        # HELPER: decides whether a union can be done by concatenation while
        # keeping the frequency. Requires the same type. Returns false if this
        # index has no frequency, the frequencies differ, or the index is not
        # increasing. Returns true if either operand is empty. Otherwise
        # orders the operands by first value and returns true if the later
        # one starts exactly one frequency step after the earlier one ends, or
        # starts at a value contained in the earlier one (overlap).
        # Assumes that type(self) == type(other), as per the annotation
        # The ability to fast_union also implies that `freq` should be
        #  retained on union.
        freq = self.freq

        if freq is None or freq != other.freq:
            return False

        if not self.is_monotonic_increasing:
            # Because freq is not None, we must then be monotonic decreasing
            # TODO: do union on the reversed indexes?
            return False

        if len(self) == 0 or len(other) == 0:
            # only reached via union_many
            return True

        # to make our life easier, "sort" the two ranges
        if self[0] <= other[0]:
            left, right = self, other
        else:
            left, right = other, self

        right_start = right[0]
        left_end = left[-1]

        # Only need to "adjoin", not overlap
        return (right_start == left_end + freq) or right_start in left

    def _fast_union(self, other: Self, sort=None) -> Self:
        # [FUNCTION DOC]
        # HELPER: union by concatenation. Both operands must be non-empty
        # (caller checks) and pass the fast-union check. Parameter sort is
        # none by default; when it is false and self starts later than other,
        # self is kept first and only the part of other before self's start
        # is appended (no reordering), returning a new index of this type
        # (without an explicit frequency). Otherwise the operands are ordered
        # by first value; if the later-ending one extends past the earlier,
        # the non-overlapping tail is appended and a new index returned
        # (asserting that frequency is retained); else the earlier-starting
        # (covering) operand is returned as is.
        # Caller is responsible for ensuring self and other are non-empty

        # to make our life easier, "sort" the two ranges
        if self[0] <= other[0]:
            left, right = self, other
        elif sort is False:
            # TDIs are not in the "correct" order and we don't want
            #  to sort but want to remove overlaps
            left, right = self, other
            left_start = left[0]
            loc = right.searchsorted(left_start, side="left")
            right_chunk = right._values[:loc]
            dates = concat_compat((left._values, right_chunk))
            result = type(self)._simple_new(dates, name=self.name)
            return result
        else:
            left, right = other, self

        left_end = left[-1]
        right_end = right[-1]

        # concatenate
        if left_end < right_end:
            loc = right.searchsorted(left_end, side="right")
            right_chunk = right._values[loc:]
            dates = concat_compat([left._values, right_chunk])
            # The can_fast_union check ensures that the result.freq
            #  should match self.freq
            assert isinstance(dates, type(self._data))
            # error: Item "ExtensionArray" of "ExtensionArray |
            # ndarray[Any, Any]" has no attribute "_freq"
            assert dates._freq == self.freq  # type: ignore[union-attr]
            result = type(self)._simple_new(dates)
            return result
        else:
            return left

    def _union(self, other, sort):
        # [FUNCTION DOC]
        # HELPER: union implementation called by the public union after
        # validation. Asserts other has the same class and dtype. Uses the
        # range-index route for tick frequencies, the fast concatenation
        # route when allowed, otherwise the generic union followed by
        # frequency inference. Parameters: other and sort (none, true or
        # false). Returns the union index.
        # We are called by `union`, which is responsible for this validation
        assert isinstance(other, type(self))
        assert self.dtype == other.dtype

        if self._can_range_setop(other):
            return self._range_union(other, sort=sort)

        if self._can_fast_union(other):
            result = self._fast_union(other, sort=sort)
            # in the case with sort=None, the _can_fast_union check ensures
            #  that result.freq == self.freq
            return result
        else:
            return super()._union(other, sort)._with_freq("infer")

    # --------------------------------------------------------------------
    # Join Methods

    def _get_join_freq(self, other):
        # [FUNCTION DOC]
        # HELPER: returns this index's frequency if a fast union with other
        # is possible (so the join result stays regular), otherwise none.
        """
        Get the freq to attach to the result of a join operation.
        """
        freq = None
        if self._can_fast_union(other):
            freq = self.freq
        return freq

    def _wrap_join_result(
        self,
        joined,
        other,
        lidx: npt.NDArray[np.intp] | None,
        ridx: npt.NDArray[np.intp] | None,
        how: JoinHow,
    ) -> tuple[Self, npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]:
        # [FUNCTION DOC]
        # HELPER: post-processes a join. Parameters: joined (raw result), other
        # (the other index, must have the same dtype, else assertion fails),
        # lidx and ridx (optional integer indexers), how (join kind). Calls
        # the base wrapper and then overwrites the frequency stored on the
        # resulting index's array with the join frequency. Returns the
        # triple (index, left indexer, right indexer). Mutates the result's
        # array frequency.
        assert other.dtype == self.dtype, (other.dtype, self.dtype)
        join_index, lidx, ridx = super()._wrap_join_result(
            joined, other, lidx, ridx, how
        )
        join_index._data._freq = self._get_join_freq(other)
        return join_index, lidx, ridx

    def _get_engine_target(self) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: returns the values viewed as 64-bit integers, which is the
        # form the lookup engine and join routines require.
        # engine methods and libjoin methods need dt64/td64 values cast to i8
        return self._data._ndarray.view("i8")

    def _from_join_target(self, result: np.ndarray):
        # [FUNCTION DOC]
        # HELPER: inverse of the engine-target conversion. Parameter result
        # is an integer array; it is viewed as the underlying datetime64 or
        # timedelta64 dtype and wrapped into a new array of the same type.
        # view e.g. i8 back to M8[ns]
        result = result.view(self._data._ndarray.dtype)
        return self._data._from_backing_data(result)

    def _searchsorted_monotonic(self, label, side: Literal["left", "right"] = "left"):
        # [FUNCTION DOC]
        # HELPER: binary-search position of label. Parameters: label and side
        # (left default, or right). For an increasing index, when label is a
        # Timestamp or Timedelta with a finer unit than the index, it is first
        # converted to the index unit: rounded down for side right, rounded up
        # (ceiling) for side left, so the comparison is exact. Then defers to
        # the base implementation and returns its integer position.
        if (
            self.is_monotonic_increasing
            and isinstance(label, (Timestamp, Timedelta))
            and abbrev_to_npy_unit(label.unit) > abbrev_to_npy_unit(self.unit)
        ):
            # For non-matching units we can safely round down (with side=right)
            # This is needed for GH#63262
            if side == "right":
                label = label.as_unit(self.unit)  # this should always be a round-down
            else:
                # round up
                label = label.ceil(self.unit).as_unit(self.unit)

        return super()._searchsorted_monotonic(label, side)

    # --------------------------------------------------------------------
    # List-like Methods

    def _get_delete_freq(self, loc: int | slice | Sequence[int]):
        # [FUNCTION DOC]
        # HELPER: computes the frequency of the result of deleting loc (an
        # integer, slice or integer sequence). Returns none if there is no
        # frequency. The frequency is kept for an integer location at either
        # end; a list-like is first collapsed to a slice when possible; a
        # step-one (or no-step) slice keeps the frequency if it starts at the
        # beginning or stops at the end. Otherwise returns none.
        """
        Find the `freq` for self.delete(loc).
        """
        freq = None
        if self.freq is not None:
            if is_integer(loc):
                if loc in (0, -len(self), -1, len(self) - 1):
                    freq = self.freq
            else:
                if is_list_like(loc):
                    # error: Incompatible types in assignment (expression has
                    # type "Union[slice, ndarray]", variable has type
                    # "Union[int, slice, Sequence[int]]")
                    loc = lib.maybe_indices_to_slice(  # type: ignore[assignment]
                        np.asarray(loc, dtype=np.intp), len(self)
                    )
                if isinstance(loc, slice) and loc.step in (1, None):
                    if loc.start in (0, None) or loc.stop in (len(self), None):
                        freq = self.freq
        return freq

    def _get_insert_freq(self, loc: int, item):
        # [FUNCTION DOC]
        # HELPER: computes the frequency of the result of inserting item at
        # position loc. The item is validated and boxed via the array (may
        # raise a type or value error for incompatible items). Returns none
        # when there is no frequency. For a non-empty index the frequency
        # is kept only if item is not missing and extends the index
        # regularly at the front or the back. For an empty index it is kept
        # for tick frequencies, or when the item lies on the offset.
        """
        Find the `freq` for self.insert(loc, item).
        """
        value = self._data._validate_scalar(item)
        item = self._data._box_func(value)

        freq = None
        if self.freq is not None:
            # freq can be preserved on edge cases
            if self.size:
                if item is NaT:
                    pass
                elif loc in (0, -len(self)) and item + self.freq == self[0]:
                    freq = self.freq
                elif (loc == len(self)) and item - self.freq == self[-1]:
                    freq = self.freq
            # Adding a single item to an empty index may preserve freq
            elif isinstance(self.freq, Tick):
                # all TimedeltaIndex cases go through here; is_on_offset
                #  would raise TypeError
                freq = self.freq
            elif self.freq.is_on_offset(item):
                freq = self.freq
        return freq

    def delete(self, loc) -> Self:
        # [FUNCTION DOC]
        # Parameter loc is an integer or list of integers (positions to
        # remove). Deletes them via the base implementation, then sets the
        # frequency of the result's array to the one computed for that
        # deletion (kept only for edge removals). Returns the new index.
        # KNOWN ISSUE: the docstring example uses a plain string Index, not
        # a datetime-like index.
        """
        Make new Index with passed location(-s) deleted.

        Parameters
        ----------
        loc : int or list of int
            Location of item(-s) which will be deleted.
            Use a list of locations to delete more than one value at the same time.

        Returns
        -------
        Index
            Will be same type as self, except for RangeIndex.

        See Also
        --------
        numpy.delete : Delete any rows and column from NumPy array (ndarray).

        Examples
        --------
        >>> idx = pd.Index(["a", "b", "c"])
        >>> idx.delete(1)
        Index(['a', 'c'], dtype='str')
        >>> idx = pd.Index(["a", "b", "c"])
        >>> idx.delete([0, 2])
        Index(['b'], dtype='str')
        """
        result = super().delete(loc)
        result._data._freq = self._get_delete_freq(loc)
        return result

    def insert(self, loc: int, item):
        # [FUNCTION DOC]
        # Parameters: loc (int position, negative allowed) and item (value to
        # insert). Delegates to the base insert; if the result is still the
        # same index class (no fallback to object dtype), sets the result's
        # frequency to the computed insert frequency. Returns the new index.
        # KNOWN ISSUE: the docstring example uses a plain string Index, not
        # a datetime-like index.
        """
        Make new Index inserting new item at location.
        Follows Python numpy.insert semantics for negative values.

        Parameters
        ----------
        loc : int
            The integer location where the new item will be inserted.
        item : object
            The new item to be inserted into the Index.

        Returns
        -------
        Index
            Returns a new Index object resulting from inserting the specified item at
            the specified location within the original Index.

        See Also
        --------
        Index.append : Append a collection of Indexes together.

        Examples
        --------
        >>> idx = pd.Index(["a", "b", "c"])
        >>> idx.insert(1, "x")
        Index(['a', 'x', 'b', 'c'], dtype='str')
        """
        result = super().insert(loc, item)
        if isinstance(result, type(self)):
            # i.e. parent class method did not cast
            result._data._freq = self._get_insert_freq(loc, item)
        return result

    # --------------------------------------------------------------------
    # NDArray-Like Methods

    def take(
        self,
        indices,
        axis: Axis = 0,
        allow_fill: bool = True,
        fill_value=None,
        **kwargs,
    ) -> Self:
        # [FUNCTION DOC]
        # Select elements by position. Parameters: indices (array-like of
        # integer positions, converted to a platform integer array), axis
        # (must be 0), allow_fill (bool, default true), fill_value (scalar,
        # default none), and extra keyword arguments that are only validated
        # for numpy compatibility (invalid ones raise). Calls the array-backed
        # index take, then, if the indices form a simple slice, copies the
        # corresponding slice-derived frequency onto the result's array.
        # Returns a new index of the same type.
        # KNOWN ISSUE: the docstring says negative indices count from the
        # right when allow_fill is false and calls that the default, but the
        # signature default for allow_fill is true.
        """
        Return a new Index of the values selected by the indices.
        For internal compatibility with numpy arrays.

        Parameters
        ----------
        indices : array-like
            Indices to be taken.
        axis : {0 or 'index'}, optional
            The axis over which to select values, always 0 or 'index'.
        allow_fill : bool, default True
            How to handle negative values in `indices`.
            * False: negative values in `indices` indicate positional indices
                from the right (the default). This is similar to
                :func:`numpy.take`.
            * True: negative values in `indices` indicate
                missing values. These values are set to `fill_value`. Any other
                other negative values raise a ``ValueError``.
        fill_value : scalar, default None
            If allow_fill=True and fill_value is not None, indices specified by
            -1 are regarded as NA. If Index doesn't hold NA, raise ValueError.
        **kwargs
            Required for compatibility with numpy.

        Returns
        -------
        Index
            An index formed of elements at the given indices. Will be the same
            type as self, except for RangeIndex.

        See Also
        --------
        numpy.ndarray.take: Return an array formed from the
            elements of a at the given indices.

        Examples
        --------
        >>> idx = pd.Index(["a", "b", "c"])
        >>> idx.take([2, 2, 1, 2])
        Index(['c', 'c', 'b', 'c'], dtype='str')
        """
        nv.validate_take((), kwargs)
        indices = np.asarray(indices, dtype=np.intp)

        result = NDArrayBackedExtensionIndex.take(
            self, indices, axis, allow_fill, fill_value, **kwargs
        )

        maybe_slice = lib.maybe_indices_to_slice(indices, len(self))
        if isinstance(maybe_slice, slice):
            freq = self._data._get_getitem_freq(maybe_slice)
            result._data._freq = freq
        return result
