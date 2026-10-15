# [MODULE DOC]
# Implements the pandas Timestamp scalar. Two layers: _Timestamp, a C extension type
# subclassing datetime (via ABCTimestamp) that stores the authoritative state and does
# heavy lifting, and Timestamp, a Python class handling construction, rounding, timezone
# handling and stdlib-compatible methods.
# Data model: _value is a 64-bit integer count of units since the Unix epoch in UTC
# (wall time if tz-naive); _creso is the storage resolution (seconds, milliseconds,
# microseconds or nanoseconds only); _year is the true year; _nanosecond (0-999) holds
# sub-microsecond nanoseconds (only nonzero at ns resolution). The inherited datetime
# fields hold month/day/hour/minute/second/microsecond/tzinfo/fold of the local wall
# time, but the inherited year is a fake 1970 (or 1972 for leap years) whenever the true
# year is outside 1..9999, so all year-dependent logic must use _year rather than the
# stdlib datetime machinery. The NaT singleton (c_NaT) is returned for missing values
# (NPY_NAT sentinel).
# Module constants: _zero_time (midnight time object), _no_input (sentinel meaning
# "argument not passed"). Also defines MinMaxReso (descriptor for min/max/resolution),
# integer_op_not_supported, _unpickle_timestamp, and the scalar normalize_i8_stamp
# helper. Imports numpy and the pandas C datetime API at load.
"""
_Timestamp is a c-defined subclass of datetime.datetime

_Timestamp is PITA. Because we inherit from datetime, which has very specific
construction requirements, we need to do object instantiation in python
(see Timestamp class below). This will serve as a C extension type that
shadows the python class, where we do any heavy lifting.
"""

import warnings

cimport cython

import numpy as np

cimport numpy as cnp
from numpy cimport (
    int64_t,
    ndarray,
    uint8_t,
)

cnp.import_array()

from cpython.datetime cimport (  # alias tzinfo_type bc `tzinfo` is a kwarg below
    PyDate_Check,
    PyDateTime_Check,
    PyDelta_Check,
    PyTZInfo_Check,
    datetime,
    import_datetime,
    time as dt_time,
    tzinfo as tzinfo_type,
)
from cpython.object cimport (
    Py_EQ,
    Py_GE,
    Py_GT,
    Py_LE,
    Py_LT,
    Py_NE,
    PyObject_RichCompare,
    PyObject_RichCompareBool,
)

import_datetime()

import datetime as dt

from pandas._libs.tslibs cimport ccalendar
from pandas._libs.tslibs.base cimport ABCTimestamp

from pandas.util._decorators import set_module
from pandas.util._exceptions import find_stack_level

from pandas._libs.tslibs.conversion cimport (
    _TSObject,
    convert_datetime_to_tsobject,
    convert_to_tsobject,
    maybe_localize_tso,
)
from pandas._libs.tslibs.dtypes cimport (
    npy_unit_to_abbrev,
    npy_unit_to_attrname,
    periods_per_day,
    periods_per_second,
)
from pandas._libs.tslibs.util cimport (
    is_array,
    is_float_object,
    is_integer_object,
)

from pandas._libs.tslibs.fields import (
    RoundTo,
    get_date_name_field,
    get_start_end_field,
    round_nsint64,
)

from pandas._libs.tslibs.nattype cimport (
    NPY_NAT,
    c_NaT as NaT,
)
from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    NPY_FR_ns,
    cmp_dtstructs,
    cmp_scalar,
    convert_reso,
    dts_to_iso_string,
    get_datetime64_unit,
    get_unit_from_dtype,
    import_pandas_datetime,
    npy_datetimestruct,
    npy_datetimestruct_to_datetime,
    pandas_datetime_to_datetimestruct,
    pydatetime_to_dtstruct,
)

import_pandas_datetime()

from pandas._libs.tslibs.np_datetime import (
    OutOfBoundsDatetime,
    OutOfBoundsTimedelta,
)

from pandas._libs.tslibs.offsets cimport to_offset
from pandas._libs.tslibs.timedeltas cimport (
    _Timedelta,
    get_unit_for_round,
    is_any_td_scalar,
)

from pandas._libs.tslibs.timedeltas import Timedelta

from pandas._libs.tslibs.timezones cimport (
    get_timezone,
    is_utc,
    maybe_get_tz,
    treat_tz_as_pytz,
    utc_stdlib as UTC,
)
from pandas._libs.tslibs.tzconversion cimport (
    tz_convert_from_utc_single,
    tz_localize_to_utc_single,
)

# ----------------------------------------------------------------------
# Constants
_zero_time = dt_time(0, 0)
_no_input = object()

# ----------------------------------------------------------------------


cdef _Timestamp create_timestamp_from_ts(
    int64_t value,
    npy_datetimestruct dts,
    tzinfo tz,
    bint fold,
    NPY_DATETIMEUNIT reso=NPY_FR_ns,
):
    # [FUNCTION DOC]
    # HELPER: Build a _Timestamp (actually of Python class Timestamp) from parts without
    # validation. Calls datetime's constructor with month/day/h/m/s/us from dts, tz and
    # fold; the year passed is dts.year if within 1..9999, else 1972 when dts.year is a
    # leap year (so Feb 29 stays valid) or 1970 otherwise. Then sets _value, _year (true
    # dts.year), _nanosecond (dts.ps // 1000) and _creso.
    # Inputs: value - int64 epoch count in resolution reso (UTC if tz-aware); dts -
    # broken-down local wall-time struct consistent with value; tz - tzinfo or None;
    # fold - 0/1; reso - resolution unit, default nanoseconds.
    # Outputs: new Timestamp. Exceptions: whatever datetime construction raises (e.g.
    # invalid tz type). Side effects: allocates a Python object.
    """ convenience routine to construct a Timestamp from its parts """
    cdef:
        _Timestamp ts_base
        int64_t pass_year = dts.year

    # We pass year=1970/1972 here and set year below because with non-nanosecond
    #  resolution we may have datetimes outside of the stdlib pydatetime
    #  implementation bounds, which would raise.
    # NB: this means the C-API macro PyDateTime_GET_YEAR is unreliable.
    if 1 <= pass_year <= 9999:
        # we are in-bounds for pydatetime
        pass
    elif ccalendar.is_leapyear(dts.year):
        pass_year = 1972
    else:
        pass_year = 1970

    ts_base = _Timestamp.__new__(Timestamp, pass_year, dts.month,
                                 dts.day, dts.hour, dts.min,
                                 dts.sec, dts.us, tz, fold=fold)

    ts_base._value = value
    ts_base._year = dts.year
    ts_base._nanosecond = dts.ps // 1000
    ts_base._creso = reso

    return ts_base


def _unpickle_timestamp(value, freq, tz, reso=NPY_FR_ns):
    # [FUNCTION DOC]
    # HELPER: Pickle reconstructor used by _Timestamp.__reduce__. Rebuilds a Timestamp
    # via _from_value_and_reso(value, reso, tz). freq is ignored (kept for backward
    # compatibility with old pickles carrying a freq).
    # Inputs: value - int epoch count; freq - ignored; tz - tzinfo or None; reso -
    # resolution unit (default nanoseconds).
    # Outputs: Timestamp (or NaT if value is the NaT sentinel).
    # Exceptions: NotImplementedError for unsupported reso. Side effects: none.
    # GH#41949 dont warn on unpickle if we have a freq
    ts = Timestamp._from_value_and_reso(value, reso, tz)
    return ts


# ----------------------------------------------------------------------

def integer_op_not_supported(obj):
    # [FUNCTION DOC]
    # Build (but do not raise) the TypeError used when integers or integer arrays are
    # added to/subtracted from a datetime-like object; message names the type of obj and
    # suggests using n * obj.freq.
    # Inputs: obj - any object (only its type name is used).
    # Outputs: a TypeError instance. Exceptions: none. Side effects: none.
    # GH#22535 add/sub of integers and int-arrays is no longer allowed
    # Note we return rather than raise the exception so we can raise in
    #  the caller; mypy finds this more palatable.
    cls = type(obj).__name__

    # GH#30886 using an fstring raises SystemError
    int_addsub_msg = (
        f"Addition/subtraction of integers and integer-arrays with {cls} is "
        "no longer supported.  Instead of adding/subtracting `n`, "
        "use `n * obj.freq`"
    )
    return TypeError(int_addsub_msg)


class MinMaxReso:
    # [CLASS DOC]
    # Descriptor providing Timestamp.min, .max and .resolution on both the class and
    # instances. Stores the attribute name ("min", "max" or "resolution") and a
    # docstring. On the class the value assumes nanosecond resolution; on an instance it
    # uses the instance's _creso.
    """
    We need to define min/max/resolution on both the Timestamp _instance_
    and Timestamp class.  On an instance, these depend on the object's _reso.
    On the class, we default to the values we would get with nanosecond _reso.

    See also: timedeltas.MinMaxReso
    """
    def __init__(self, name, docstring):
        # [FUNCTION DOC]
        # Store name (one of "min", "max", "resolution") and set the descriptor's
        # __doc__ to docstring. Inputs: name - str; docstring - str. Outputs: none.
        self._name = name
        self.__doc__ = docstring

    def __get__(self, obj, type=None):
        # [FUNCTION DOC]
        # Descriptor getter. Raw value: int64 min + 1 for "min" (int64 min is NaT),
        # int64 max for "max", 1 for "resolution". When accessed on the class (obj is
        # None) returns Timestamp(val) (nanoseconds) or Timedelta(1) for resolution. On
        # an instance returns Timestamp._from_value_and_reso(val, obj._creso, tz=None)
        # or Timedelta._from_value_and_reso(1, obj._creso). Sets result.__doc__ to the
        # stored docstring.
        # Inputs: obj - instance or None; type - owner class (unused).
        # Outputs: Timestamp or Timedelta.
        # Exceptions: AssertionError if name is not one of the three.
        # Side effects: mutates __doc__ on the returned object.
        cls = Timestamp
        if self._name == "min":
            val = np.iinfo(np.int64).min + 1
        elif self._name == "max":
            val = np.iinfo(np.int64).max
        else:
            assert self._name == "resolution"
            val = 1
            cls = Timedelta

        if obj is None:
            # i.e. this is on the class, default to nanos
            result = cls(val)
        elif self._name == "resolution":
            result = Timedelta._from_value_and_reso(val, obj._creso)
        else:
            result = Timestamp._from_value_and_reso(val, obj._creso, tz=None)

        result.__doc__ = self.__doc__

        return result

    def __set__(self, obj, value):
        # [FUNCTION DOC]
        # Descriptor setter; always raises AttributeError("<name> is not settable.").
        raise AttributeError(f"{self._name} is not settable.")


# ----------------------------------------------------------------------

cdef class _Timestamp(ABCTimestamp):
    # [CLASS DOC]
    # C extension base of Timestamp. Attributes (readonly C fields declared in the pxd):
    # _value (int64 epoch count, UTC if tz-aware), _nanosecond (0-999), _year (true
    # year), _creso (resolution unit). Inherits datetime's stored fields (with the
    # fake-year caveat described in the module doc). Class attributes:
    # __array_priority__ = 100 (so numpy defers binary ops to it), dayofweek and
    # dayofyear aliases, min/max/resolution MinMaxReso descriptors and their docstrings.
    # Provides comparison, arithmetic, field accessors, rendering, pickling and
    # conversions.

    # higher than np.ndarray and np.matrix
    __array_priority__ = 100
    dayofweek = _Timestamp.day_of_week
    dayofyear = _Timestamp.day_of_year

    _docstring_min = """
    Returns the minimum bound possible for Timestamp.

    This property provides access to the smallest possible value that
    can be represented by a Timestamp object.

    Returns
    -------
    Timestamp

    See Also
    --------
    Timestamp.max: Returns the maximum bound possible for Timestamp.
    Timestamp.resolution: Returns the smallest possible difference between
        non-equal Timestamp objects.

    Examples
    --------
    >>> pd.Timestamp.min
    Timestamp('1677-09-21 00:12:43.145224193')
    """

    _docstring_max = """
    Returns the maximum bound possible for Timestamp.

    This property provides access to the largest possible value that
    can be represented by a Timestamp object.

    Returns
    -------
    Timestamp

    See Also
    --------
    Timestamp.min: Returns the minimum bound possible for Timestamp.
    Timestamp.resolution: Returns the smallest possible difference between
        non-equal Timestamp objects.

    Examples
    --------
    >>> pd.Timestamp.max
    Timestamp('2262-04-11 23:47:16.854775807')
    """

    _docstring_reso = """
    Returns the smallest possible difference between non-equal Timestamp objects.

    The resolution value is determined by the underlying representation of time
    units and is equivalent to Timedelta(nanoseconds=1).

    Returns
    -------
    Timedelta

    See Also
    --------
    Timestamp.max: Returns the maximum bound possible for Timestamp.
    Timestamp.min: Returns the minimum bound possible for Timestamp.

    Examples
    --------
    >>> pd.Timestamp.resolution
    Timedelta('0 days 00:00:00.000000001')
    """

    min = MinMaxReso("min", _docstring_min)
    max = MinMaxReso("max", _docstring_max)
    resolution = MinMaxReso("resolution", _docstring_reso)  # GH#21336, GH#21365

    @property
    def value(self) -> int:
        # [FUNCTION DOC]
        # Property: _value converted to nanoseconds (convert_reso with round_ok False).
        # Outputs: Python int nanoseconds since epoch (UTC).
        # Exceptions: OverflowError (re-raised with a hint to use asm8) if the value
        # does not fit in int64 nanoseconds. Side effects: none.
        """
        Return the value of the Timestamp.

        Returns
        -------
        int
            The integer representation of the Timestamp object in nanoseconds
            since the Unix epoch (1970-01-01 00:00:00 UTC).

        See Also
        --------
        Timestamp.second : Return the second of the Timestamp.
        Timestamp.minute : Return the minute of the Timestamp.

        Examples
        --------
        >>> ts = pd.Timestamp("2024-08-31 16:16:30")
        >>> ts.value
        1725120990000000000
        """

        try:
            return convert_reso(self._value, self._creso, NPY_FR_ns, False)
        except OverflowError:
            raise OverflowError(
                "Cannot convert Timestamp to nanoseconds without overflow. "
                "Use `.asm8.view('i8')` to cast represent Timestamp in its own "
                f"unit (here, {self.unit})."
            )

    @property
    def unit(self) -> str:
        # [FUNCTION DOC]
        # Property: string abbreviation of _creso: "s", "ms", "us" or "ns".
        """
        The abbreviation associated with self._creso.

        This property returns a string representing the time unit of the Timestamp's
        resolution. It corresponds to the smallest time unit that can be represented
        by this Timestamp object. The possible values are:
        - 's' (second)
        - 'ms' (millisecond)
        - 'us' (microsecond)
        - 'ns' (nanosecond)

        Returns
        -------
        str
            A string abbreviation of the Timestamp's resolution unit:
            - 's' for second
            - 'ms' for millisecond
            - 'us' for microsecond
            - 'ns' for nanosecond

        See Also
        --------
        Timestamp.resolution : Return resolution of the Timestamp.
        Timedelta : A duration expressing the difference between two dates or times.

        Examples
        --------
        >>> pd.Timestamp("2020-01-01 12:34:56").unit
        's'

        >>> pd.Timestamp("2020-01-01 12:34:56.123").unit
        'ms'

        >>> pd.Timestamp("2020-01-01 12:34:56.123456").unit
        'us'

        >>> pd.Timestamp("2020-01-01 12:34:56.123456789").unit
        'ns'
        """
        return npy_unit_to_abbrev(self._creso)

    # -----------------------------------------------------------------
    # Constructors

    @classmethod
    def _from_value_and_reso(cls, int64_t value, NPY_DATETIMEUNIT reso, tzinfo tz):
        # [FUNCTION DOC]
        # Classmethod: construct from an epoch count and resolution. Returns NaT if
        # value is the NaT sentinel. Otherwise fills a _TSObject with value/creso,
        # converts value to a broken-down struct, applies tz via maybe_localize_tso
        # (which, for a non-UTC tz, shifts the struct to local wall time and sets
        # tzinfo/fold), and calls create_timestamp_from_ts.
        # Inputs: value - int64 (UTC-based if tz given); reso - resolution unit; tz -
        # tzinfo or None. Outputs: Timestamp or NaT.
        # Exceptions: NotImplementedError if reso is coarser than seconds or finer than
        # nanoseconds. Side effects: none.
        cdef:
            _TSObject obj = _TSObject()

        if value == NPY_NAT:
            return NaT

        if reso < NPY_DATETIMEUNIT.NPY_FR_s or reso > NPY_DATETIMEUNIT.NPY_FR_ns:
            raise NotImplementedError(
                "Only resolutions 's', 'ms', 'us', 'ns' are supported."
            )

        obj.value = value
        obj.creso = reso
        pandas_datetime_to_datetimestruct(value, reso, &obj.dts)
        maybe_localize_tso(obj, tz, reso)

        return create_timestamp_from_ts(
            value, obj.dts, tz=obj.tzinfo, fold=obj.fold, reso=reso
        )

    @classmethod
    def _from_dt64(cls, dt64: np.datetime64):
        # [FUNCTION DOC]
        # Classmethod: construct a tz-naive Timestamp from a numpy datetime64, keeping
        # its unit. Reads the unit and int64 payload and delegates to
        # _from_value_and_reso.
        # Inputs: dt64 - numpy datetime64 scalar. Outputs: Timestamp or NaT.
        # Exceptions: NotImplementedError for units other than s/ms/us/ns.
        # construct a Timestamp from an np.datetime64 object, keeping the
        #  resolution of the input.
        # This is here mainly so we can incrementally implement non-nano
        #  (e.g. only tznaive at first)
        cdef:
            int64_t value
            NPY_DATETIMEUNIT reso

        reso = get_datetime64_unit(dt64)
        value = cnp.get_datetime64_value(dt64)
        return cls._from_value_and_reso(value, reso, None)

    # -----------------------------------------------------------------

    def __hash__(_Timestamp self):
        # [FUNCTION DOC]
        # Hash. Uses hash(_value) when there are nonzero nanoseconds or the year is
        # outside 1..9999 (not representable as a datetime); if fold is 1, hashes the
        # fold=0 replacement via datetime.__hash__ (so both folds hash equal, matching
        # stdlib); otherwise datetime.__hash__(self), so equal datetime objects hash
        # equal. Outputs: int.
        if self._nanosecond:
            return hash(self._value)
        if not (1 <= self._year <= 9999):
            # out of bounds for pydatetime
            return hash(self._value)
        if self.fold:
            return datetime.__hash__(self.replace(fold=0))
        return datetime.__hash__(self)

    def __richcmp__(_Timestamp self, object other, int op):
        # [FUNCTION DOC]
        # Rich comparison dispatch. other handling:
        # - _Timestamp: compared directly.
        # - NaT: only != is True; all other ops False.
        # - numpy datetime64: converted with Timestamp(other).
        # - datetime: if self has no nanoseconds, compare self.to_pydatetime() with
        #   other using stdlib semantics; else convert via type(self)(other), falling
        #   back to _compare_outside_nanorange on ValueError.
        # - ndarray: datetime64 dtype -> if naive compare asm8 vs array elementwise
        #   (numpy), if aware return all-True for != / all-False for == else TypeError;
        #   object dtype -> elementwise Python comparison into bool array; other dtypes
        #   -> all-True/all-False for !=/== else NotImplemented.
        # - date (not datetime): == is False, != is True, ordering raises TypeError.
        # - anything else: NotImplemented. For two Timestamps: if tz-awareness differs,
        #   ==/!= return NotImplemented (the reflected op then yields the default
        #   identity result) and ordering raises TypeError; same resolution compares
        #   _value; otherwise _compare_mismatched_resos.
        # Inputs: other - any; op - comparison opcode. Outputs: bool, bool ndarray, or
        # NotImplemented. Exceptions: TypeError as above; conversion errors propagate.
        cdef:
            _Timestamp ots

        if isinstance(other, _Timestamp):
            ots = other
        elif other is NaT:
            return op == Py_NE
        elif cnp.is_datetime64_object(other):
            ots = Timestamp(other)
        elif PyDateTime_Check(other):
            if self._nanosecond == 0:
                val = self.to_pydatetime()
                return PyObject_RichCompareBool(val, other, op)

            try:
                ots = type(self)(other)
            except ValueError:
                return self._compare_outside_nanorange(other, op)

        elif is_array(other):
            # avoid recursion error GH#15183
            if other.dtype.kind == "M":
                if self.tz is None:
                    return PyObject_RichCompare(self.asm8, other, op)
                elif op == Py_NE:
                    return np.ones(other.shape, dtype=np.bool_)
                elif op == Py_EQ:
                    return np.zeros(other.shape, dtype=np.bool_)
                raise TypeError(
                    "Cannot compare tz-naive and tz-aware timestamps"
                )
            elif other.dtype.kind == "O":
                # Operate element-wise
                return np.array(
                    [PyObject_RichCompare(self, x, op) for x in other],
                    dtype=bool,
                )
            elif op == Py_NE:
                return np.ones(other.shape, dtype=np.bool_)
            elif op == Py_EQ:
                return np.zeros(other.shape, dtype=np.bool_)
            return NotImplemented

        elif PyDate_Check(other):
            # returning NotImplemented defers to the `date` implementation
            #  which incorrectly drops tz and normalizes to midnight
            #  before comparing
            # We follow the stdlib datetime behavior of never being equal
            if op == Py_EQ:
                return False
            elif op == Py_NE:
                return True
            raise TypeError(
                "Cannot compare Timestamp with datetime.date. "
                "Use ts == pd.Timestamp(date) or ts.date() == date instead."
            )
        else:
            return NotImplemented

        if not self._can_compare(ots):
            if op == Py_NE or op == Py_EQ:
                return NotImplemented
            raise TypeError(
                "Cannot compare tz-naive and tz-aware timestamps"
            )
        if self._creso == ots._creso:
            return cmp_scalar(self._value, ots._value, op)
        return self._compare_mismatched_resos(ots, op)

    # TODO: copied from Timedelta; try to de-duplicate
    cdef bint _compare_mismatched_resos(self, _Timestamp other, int op):
        # [FUNCTION DOC]
        # HELPER: Compare two Timestamps with different resolutions without overflow by
        # converting both _value fields to broken-down structs and comparing those
        # field-by-field (cmp_dtstructs). Inputs: other - _Timestamp; op - opcode.
        # Outputs: boolean result. Side effects: none.
        # Can't just dispatch to numpy as they silently overflow and get it wrong
        cdef:
            npy_datetimestruct dts_self
            npy_datetimestruct dts_other

        # dispatch to the datetimestruct utils instead of writing new ones!
        pandas_datetime_to_datetimestruct(self._value, self._creso, &dts_self)
        pandas_datetime_to_datetimestruct(other._value, other._creso, &dts_other)
        return cmp_dtstructs(&dts_self,  &dts_other, op)

    cdef bint _compare_outside_nanorange(_Timestamp self, datetime other,
                                         int op) except -1:
        # [FUNCTION DOC]
        # HELPER: Compare self (which has nonzero nanoseconds) with a datetime that
        # could not be converted to a Timestamp. Converts self to a datetime without
        # warning (dropping nanoseconds). If awareness differs returns NotImplemented.
        # If self has no nanoseconds, compares datetimes directly. Otherwise self is
        # strictly greater than its truncation: != True, == False, and ordering ops
        # decided only by comparing _year with other.year (<= for LT/LE, >= for GT/GE).
        # Inputs: other - datetime; op - opcode. Outputs: boolean.
        # Exceptions: errors from to_pydatetime (e.g. ValueError for out-of-range
        # years).
        # KNOWN ISSUE: return type is a C boolean, so "return NotImplemented" is coerced
        # to True rather than signalling NotImplemented to the caller.
        # KNOWN ISSUE: ordering by year only is a coarse approximation; e.g. for equal
        # years both < and > return True. The "_nanosecond == 0" branch is unreachable
        # because the only caller invokes this when _nanosecond != 0.
        cdef:
            datetime dtval = self.to_pydatetime(warn=False)

        if not self._can_compare(other):
            return NotImplemented

        if self._nanosecond == 0:
            return PyObject_RichCompareBool(dtval, other, op)

        # otherwise we have dtval < self
        if op == Py_NE:
            return True
        if op == Py_EQ:
            return False
        if op == Py_LE or op == Py_LT:
            return self._year <= other.year
        if op == Py_GE or op == Py_GT:
            return self._year >= other.year

    cdef bint _can_compare(self, datetime other):
        # [FUNCTION DOC]
        # HELPER: True if self and other are both tz-aware or both tz-naive.
        # Inputs: other - datetime. Outputs: boolean.
        if self.tzinfo is not None:
            return other.tzinfo is not None
        return other.tzinfo is None

    @cython.overflowcheck(True)
    def __add__(self, other):
        # [FUNCTION DOC]
        # Addition (overflow-checked). For timedelta-like scalars (Timedelta, timedelta,
        # numpy timedelta64, Tick offsets): convert to Timedelta, cast the
        # lower-resolution operand up to the higher resolution (rounding allowed), add
        # the int64 values and rebuild with the same tz via _from_value_and_reso.
        # Integers and integer arrays raise the integer_op_not_supported TypeError. A
        # timedelta64 array: naive -> asm8 + array (numpy datetime64 array); aware ->
        # object array of elementwise Timestamp sums. Otherwise NotImplemented (e.g.
        # DateOffsets handle via __radd__).
        # Inputs: other - any. Outputs: Timestamp, ndarray, or NotImplemented.
        # Exceptions: OutOfBoundsDatetime if the sum overflows; TypeError for ints.
        cdef:
            int64_t nanos = 0

        if is_any_td_scalar(other):
            other = Timedelta(other)

            # TODO: share this with __sub__, Timedelta.__add__
            # Matching numpy, we cast to the higher resolution. Unlike numpy,
            #  we raise instead of silently overflowing during this casting.
            if self._creso < other._creso:
                self = (<_Timestamp>self)._as_creso(other._creso, round_ok=True)
            elif self._creso > other._creso:
                other = (<_Timedelta>other)._as_creso(self._creso, round_ok=True)

            nanos = other._value

            try:
                new_value = self._value + nanos
                result = type(self)._from_value_and_reso(
                    new_value, reso=self._creso, tz=self.tzinfo
                )
            except OverflowError as err:
                new_value = int(self._value) + int(nanos)
                attrname = npy_unit_to_attrname[self._creso]
                raise OutOfBoundsDatetime(
                    f"Out of bounds {attrname} timestamp: {new_value}"
                ) from err

            return result

        elif is_integer_object(other):
            raise integer_op_not_supported(self)

        elif is_array(other):
            if other.dtype.kind in "iu":
                raise integer_op_not_supported(self)
            if other.dtype.kind == "m":
                if self.tz is None:
                    return self.asm8 + other
                return np.asarray(
                    [self + other[n] for n in range(len(other))],
                    dtype=object,
                )

        return NotImplemented

    def __radd__(self, other):
        # [FUNCTION DOC]
        # Reflected addition: for timedelta-like, integer or array other, delegates to
        # __add__ (avoids infinite NotImplemented recursion); else NotImplemented.
        # Have to duplicate checks to avoid infinite recursion due to NotImplemented
        if is_any_td_scalar(other) or is_integer_object(other) or is_array(other):
            return self.__add__(other)
        return NotImplemented

    def __sub__(self, other):
        # [FUNCTION DOC]
        # Subtraction. NaT -> NaT. Timedelta-like or integer -> self + (-other)
        # (integers then raise TypeError via __add__). Arrays: integer dtype raises
        # TypeError; timedelta64 -> naive: asm8 - array, aware: object array of
        # elementwise differences; other dtypes NotImplemented. datetime or datetime64
        # -> convert to Timestamp, raise TypeError if exactly one is tz-aware, cast to
        # the common finer resolution and return Timedelta of the difference of _value.
        # On overflow: if other was originally a Timestamp raise OutOfBoundsDatetime;
        # otherwise fall through to NotImplemented (letting stdlib datetime produce a
        # timedelta).
        # Inputs: other - any. Outputs: Timestamp, Timedelta, ndarray, NaT or
        # NotImplemented. Exceptions: TypeError, OutOfBoundsDatetime.
        if other is NaT:
            return NaT

        elif is_any_td_scalar(other) or is_integer_object(other):
            neg_other = -other
            return self + neg_other

        elif is_array(other):
            if other.dtype.kind in "iu":
                raise integer_op_not_supported(self)
            if other.dtype.kind == "m":
                if self.tz is None:
                    return self.asm8 - other
                return np.asarray(
                    [self - other[n] for n in range(len(other))],
                    dtype=object,
                )
            return NotImplemented

        # coerce if necessary if we are a Timestamp-like
        if PyDateTime_Check(other) or cnp.is_datetime64_object(other):
            # both_timestamps is to determine whether Timedelta(self - other)
            # should raise the OOB error, or fall back returning a timedelta.
            both_timestamps = isinstance(other, _Timestamp)
            other = type(self)(other)

            if (self.tzinfo is None) ^ (other.tzinfo is None):
                raise TypeError(
                    "Cannot subtract tz-naive and tz-aware datetime-like objects."
                )

            # Matching numpy, we cast to the higher resolution. Unlike numpy,
            #  we raise instead of silently overflowing during this casting.
            if self._creso < other._creso:
                self = (<_Timestamp>self)._as_creso(other._creso, round_ok=True)
            elif self._creso > other._creso:
                other = (<_Timestamp>other)._as_creso(self._creso, round_ok=True)

            # scalar Timestamp/datetime - Timestamp/datetime -> yields a
            # Timedelta
            try:
                res_value = self._value - other._value
                return Timedelta._from_value_and_reso(res_value, self._creso)
            except (OverflowError, OutOfBoundsDatetime, OutOfBoundsTimedelta) as err:
                if both_timestamps:
                    raise OutOfBoundsDatetime(
                        "Result is too large for pandas.Timedelta. Convert inputs "
                        "to datetime.datetime with 'Timestamp.to_pydatetime()' "
                        "before subtracting."
                    ) from err
                # We get here in stata tests, fall back to stdlib datetime
                #  method and return stdlib timedelta object
                pass

        return NotImplemented

    def __rsub__(self, other):
        # [FUNCTION DOC]
        # Reflected subtraction (other - self). datetime: Timestamp(other) - self,
        # returning NotImplemented if that overflows/out of bounds; datetime64:
        # Timestamp(other) - self (errors propagate); otherwise NotImplemented.
        if PyDateTime_Check(other):
            try:
                return type(self)(other) - self
            except (OverflowError, OutOfBoundsDatetime) as err:
                # We get here in stata tests, fall back to stdlib datetime
                #  method and return stdlib timedelta object
                pass
        elif cnp.is_datetime64_object(other):
            return type(self)(other) - self
        return NotImplemented

    # -----------------------------------------------------------------

    cdef int64_t _maybe_convert_value_to_local(self) except? -1:
        # [FUNCTION DOC]
        # HELPER: Return the epoch count of the local wall time in self's resolution. If
        # tz is set and not UTC, recompute from the stored wall-clock fields (true year)
        # plus _nanosecond; otherwise return _value.
        # Outputs: int64. Exceptions: OverflowError from struct conversion (unlikely).
        """Convert UTC i8 value to local i8 value if tz exists"""
        cdef:
            int64_t val
            tzinfo own_tz = self.tzinfo
            npy_datetimestruct dts

        if own_tz is not None and not is_utc(own_tz):
            pydatetime_to_dtstruct(self, &dts)
            val = npy_datetimestruct_to_datetime(self._creso, &dts) + self._nanosecond
        else:
            val = self._value
        return val

    @cython.boundscheck(False)
    cdef bint _get_start_end_field(self, str field, freq):
        # [FUNCTION DOC]
        # HELPER (used by DateOffset.is_month_start etc. in offsets.pyx): evaluate a
        # start/end field (e.g. "is_month_start", "is_quarter_end") relative to a given
        # offset. Reads the offset's "startingMonth" or "month" keyword (default 12) as
        # the fiscal year-end month and its name; with no freq uses month 12 and no
        # name. Runs the vectorized get_start_end_field on a length-1 array of the local
        # value.
        # Inputs: field - field name string; freq - DateOffset or None.
        # Outputs: boolean. Side effects: allocates a temporary 1-element array.
        cdef:
            int64_t val
            dict kwds
            ndarray[uint8_t, cast=True] out
            int month_kw

        if freq:
            kwds = freq.kwds
            month_kw = kwds.get("startingMonth", kwds.get("month", 12))
            freq_name = freq.name
        else:
            month_kw = 12
            freq_name = None

        val = self._maybe_convert_value_to_local()

        out = get_start_end_field(np.array([val], dtype=np.int64),
                                  field, freq_name, month_kw, self._creso)
        return out[0]

    @property
    def is_month_start(self) -> bool:
        # [FUNCTION DOC]
        # Property: True if day of month is 1. Outputs: bool.
        """
        Check if the date is the first day of the month.

        Returns
        -------
        bool
            True if the date is the first day of the month.

        See Also
        --------
        Timestamp.is_month_end : Similar property indicating the last day of the month.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.is_month_start
        False

        >>> ts = pd.Timestamp(2020, 1, 1)
        >>> ts.is_month_start
        True
        """
        return self.day == 1

    @property
    def is_month_end(self) -> bool:
        # [FUNCTION DOC]
        # Property: True if day equals the number of days in the month. Outputs: bool.
        """
        Check if the date is the last day of the month.

        Returns
        -------
        bool
            True if the date is the last day of the month.

        See Also
        --------
        Timestamp.is_month_start : Similar property indicating month start.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.is_month_end
        False

        >>> ts = pd.Timestamp(2020, 12, 31)
        >>> ts.is_month_end
        True
        """
        return self.day == self.days_in_month

    @property
    def is_quarter_start(self) -> bool:
        # [FUNCTION DOC]
        # Property: True if day is 1 and month is 1, 4, 7 or 10. Outputs: bool.
        """
        Check if the date is the first day of the quarter.

        Returns
        -------
        bool
            True if date is first day of the quarter.

        See Also
        --------
        Timestamp.is_quarter_end : Similar property indicating the quarter end.
        Timestamp.quarter : Return the quarter of the date.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.is_quarter_start
        False

        >>> ts = pd.Timestamp(2020, 4, 1)
        >>> ts.is_quarter_start
        True
        """
        return self.day == 1 and self.month % 3 == 1

    @property
    def is_quarter_end(self) -> bool:
        # [FUNCTION DOC]
        # Property: True if month is 3, 6, 9 or 12 and day is the last of the month.
        """
        Check if date is last day of the quarter.

        Returns
        -------
        bool
            True if date is last day of the quarter.

        See Also
        --------
        Timestamp.is_quarter_start : Similar property indicating the quarter start.
        Timestamp.quarter : Return the quarter of the date.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.is_quarter_end
        False

        >>> ts = pd.Timestamp(2020, 3, 31)
        >>> ts.is_quarter_end
        True
        """
        return (self.month % 3) == 0 and self.day == self.days_in_month

    @property
    def is_year_start(self) -> bool:
        # [FUNCTION DOC]
        # Property: True if the date is January 1. Outputs: bool.
        """
        Return True if date is first day of the year.

        Returns
        -------
        bool

        See Also
        --------
        Timestamp.is_year_end : Similar property indicating the end of the year.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.is_year_start
        False

        >>> ts = pd.Timestamp(2020, 1, 1)
        >>> ts.is_year_start
        True
        """
        return self.day == self.month == 1

    @property
    def is_year_end(self) -> bool:
        # [FUNCTION DOC]
        # Property: True if the date is December 31. Outputs: bool.
        """
        Return True if date is last day of the year.

        Returns
        -------
        bool

        See Also
        --------
        Timestamp.is_year_start : Similar property indicating the start of the year.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.is_year_end
        False

        >>> ts = pd.Timestamp(2020, 12, 31)
        >>> ts.is_year_end
        True
        """
        return self.month == 12 and self.day == 31

    @cython.boundscheck(False)
    cdef _get_date_name_field(self, str field, object locale):
        # [FUNCTION DOC]
        # HELPER: Compute a name field ("day_name" or "month_name") for the local wall
        # time via the vectorized get_date_name_field on a 1-element array, honoring
        # locale. Inputs: field - str; locale - locale string or None.
        # Outputs: str name. Exceptions: errors from locale lookup propagate.
        cdef:
            int64_t val
            object[::1] out

        val = self._maybe_convert_value_to_local()

        out = get_date_name_field(np.array([val], dtype=np.int64),
                                  field, locale=locale, reso=self._creso)
        return out[0]

    def day_name(self, locale=None) -> str:
        # [FUNCTION DOC]
        # Return the weekday name (e.g. "Monday") in the given locale (None = English).
        # Inputs: locale - str or None. Outputs: str.
        """
        Return the day name of the Timestamp with specified locale.

        Parameters
        ----------
        locale : str, default None (English locale)
            Locale determining the language in which to return the day name.

        Returns
        -------
        str

        See Also
        --------
        Timestamp.day_of_week : Return day of the week.
        Timestamp.day_of_year : Return day of the year.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts.day_name()
        'Saturday'

        Analogous for ``pd.NaT``:

        >>> pd.NaT.day_name()
        nan
        """
        return self._get_date_name_field("day_name", locale)

    def month_name(self, locale=None) -> str:
        # [FUNCTION DOC]
        # Return the month name (e.g. "January") in the given locale (None = English).
        # Inputs: locale - str or None. Outputs: str.
        """
        Return the month name of the Timestamp with specified locale.

        This method returns the full name of the month corresponding to the
        `Timestamp`, such as 'January', 'February', etc. The month name can
        be returned in a specified locale if provided; otherwise, it defaults
        to the English locale.

        Parameters
        ----------
        locale : str, default None (English locale)
            Locale determining the language in which to return the month name.

        Returns
        -------
        str
            The full month name as a string.

        See Also
        --------
        Timestamp.day_name : Returns the name of the day of the week.
        Timestamp.strftime : Returns a formatted string of the Timestamp.
        datetime.datetime.strftime : Returns a string representing the date and time.

        Examples
        --------
        Get the month name in English (default):

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts.month_name()
        'March'

        Analogous for ``pd.NaT``:

        >>> pd.NaT.month_name()
        nan
        """
        return self._get_date_name_field("month_name", locale)

    @property
    def is_leap_year(self) -> bool:
        # [FUNCTION DOC]
        # Property: True if the true year (_year) is a Gregorian leap year.
        """
        Return True if year is a leap year.

        A leap year is a year, which has 366 days (instead of 365) including 29th of
        February as an intercalary day. Leap years are years which are multiples of
        four with the exception of years divisible by 100 but not by 400.

        Returns
        -------
        bool
            True if year is a leap year, else False

        See Also
        --------
        Period.is_leap_year : Return True if the period’s year is in a leap year.
        DatetimeIndex.is_leap_year : Boolean indicator if the date belongs to a
            leap year.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.is_leap_year
        True
        """
        return bool(ccalendar.is_leapyear(self._year))

    @property
    def day_of_week(self) -> int:
        # [FUNCTION DOC]
        # Property: weekday, 0=Monday..6=Sunday (delegates to weekday()).
        """
        Return day of the week.

        Returns
        -------
        int

        See Also
        --------
        Timestamp.isoweekday : Return the ISO day of the week represented by the date.
        Timestamp.weekday : Return the day of the week represented by the date.
        Timestamp.day_of_year : Return day of the year.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.day_of_week
        5
        """
        return self.weekday()

    @property
    def day_of_year(self) -> int:
        # [FUNCTION DOC]
        # Property: ordinal day of year 1..366 computed from _year, month, day.
        """
        Return the day of the year.

        Returns
        -------
        int

        See Also
        --------
        Timestamp.day_of_week : Return day of the week.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.day_of_year
        74
        """
        return ccalendar.get_day_of_year(self._year, self.month, self.day)

    @property
    def quarter(self) -> int:
        # [FUNCTION DOC]
        # Property: calendar quarter 1..4, (month - 1) // 3 + 1.
        """
        Return the quarter of the year for the `Timestamp`.

        This property returns an integer representing the quarter of the year in
        which the `Timestamp` falls. The quarters are defined as follows:
        - Q1: January 1 to March 31
        - Q2: April 1 to June 30
        - Q3: July 1 to September 30
        - Q4: October 1 to December 31

        Returns
        -------
        int
            The quarter of the year (1 through 4).

        See Also
        --------
        Timestamp.month : Returns the month of the `Timestamp`.
        Timestamp.year : Returns the year of the `Timestamp`.

        Examples
        --------
        Get the quarter for a `Timestamp`:

        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.quarter
        1

        For a `Timestamp` in the fourth quarter:

        >>> ts = pd.Timestamp(2020, 10, 14)
        >>> ts.quarter
        4
        """
        return ((self.month - 1) // 3) + 1

    @property
    def day(self) -> int:
        # [FUNCTION DOC]
        # Property: day of month from the underlying datetime fields.
        """
        Return the day of the Timestamp.

        Returns
        -------
        int
            The day of the Timestamp.

        See Also
        --------
        Timestamp.week : Return the week number of the year.
        Timestamp.weekday : Return the day of the week.

        Examples
        --------
        >>> ts = pd.Timestamp("2024-08-31 16:16:30")
        >>> ts.day
        31
        """
        return super().day

    @property
    def fold(self) -> int:
        # [FUNCTION DOC]
        # Property: fold (0 or 1) from the underlying datetime, disambiguating repeated
        # wall times at DST transitions.
        """
        Return the fold value of the Timestamp.

        Returns
        -------
        int
            The fold value of the Timestamp, where 0 indicates the first occurrence
            of the ambiguous time, and 1 indicates the second.

        See Also
        --------
        Timestamp.dst : Return the daylight saving time (DST) adjustment.
        Timestamp.tzinfo : Return the timezone information associated.

        Examples
        --------
        >>> ts = pd.Timestamp("2024-11-03 01:30:00")
        >>> ts.fold
        0
        """
        return super().fold

    @property
    def year(self) -> int:
        # [FUNCTION DOC]
        # Property: the true year (_year), not the possibly fake datetime year.
        """
        Return the year of the Timestamp.

        Returns
        -------
        int
            The year of the Timestamp.

        See Also
        --------
        Timestamp.month : Return the month of the Timestamp.
        Timestamp.day : Return the day of the Timestamp.

        Examples
        --------
        >>> ts = pd.Timestamp("2024-08-31 16:16:30")
        >>> ts.year
        2024
        """
        return self._year

    @property
    def month(self) -> int:
        # [FUNCTION DOC]
        # Property: month 1..12 from the underlying datetime fields.
        """
        Return the month of the Timestamp.

        Returns
        -------
        int
            The month of the Timestamp.

        See Also
        --------
        Timestamp.day : Return the day of the Timestamp.
        Timestamp.year : Return the year of the Timestamp.

        Examples
        --------
        >>> ts = pd.Timestamp("2024-08-31 16:16:30")
        >>> ts.month
        8
        """
        return super().month

    @property
    def hour(self) -> int:
        # [FUNCTION DOC]
        # Property: hour 0..23 from the underlying datetime fields.
        """
        Return the hour of the Timestamp.

        Returns
        -------
        int
            The hour of the Timestamp.

        See Also
        --------
        Timestamp.minute : Return the minute of the Timestamp.
        Timestamp.second : Return the second of the Timestamp.

        Examples
        --------
        >>> ts = pd.Timestamp("2024-08-31 16:16:30")
        >>> ts.hour
        16
        """
        return super().hour

    @property
    def minute(self) -> int:
        # [FUNCTION DOC]
        # Property: minute 0..59 from the underlying datetime fields.
        """
        Return the minute of the Timestamp.

        Returns
        -------
        int
            The minute of the Timestamp.

        See Also
        --------
        Timestamp.hour : Return the hour of the Timestamp.
        Timestamp.second : Return the second of the Timestamp.

        Examples
        --------
        >>> ts = pd.Timestamp("2024-08-31 16:16:30")
        >>> ts.minute
        16
        """
        return super().minute

    @property
    def second(self) -> int:
        # [FUNCTION DOC]
        # Property: second 0..59 from the underlying datetime fields.
        """
        Return the second of the Timestamp.

        Returns
        -------
        int
            The second of the Timestamp.

        See Also
        --------
        Timestamp.microsecond : Return the microsecond of the Timestamp.
        Timestamp.minute : Return the minute of the Timestamp.

        Examples
        --------
        >>> ts = pd.Timestamp("2024-08-31 16:16:30")
        >>> ts.second
        30
        """
        return super().second

    @property
    def microsecond(self) -> int:
        # [FUNCTION DOC]
        # Property: microsecond 0..999999 from the underlying datetime fields.
        """
        Return the microsecond of the Timestamp.

        Returns
        -------
        int
            The microsecond of the Timestamp.

        See Also
        --------
        Timestamp.second : Return the second of the Timestamp.
        Timestamp.minute : Return the minute of the Timestamp.

        Examples
        --------
        >>> ts = pd.Timestamp("2024-08-31 16:16:30.2304")
        >>> ts.microsecond
        230400
        """
        return super().microsecond

    @property
    def nanosecond(self) -> int:
        # [FUNCTION DOC]
        # Property: nanosecond 0..999 (the _nanosecond field).
        """
        Return the nanosecond of the Timestamp.

        Returns
        -------
        int
            The nanosecond of the Timestamp.

        See Also
        --------
        Timestamp.second : Return the second of the Timestamp.
        Timestamp.microsecond : Return the microsecond of the Timestamp.

        Examples
        --------
        >>> ts = pd.Timestamp("2024-08-31 16:16:30.230400015")
        >>> ts.nanosecond
        15
        """
        return self._nanosecond

    @property
    def week(self) -> int:
        # [FUNCTION DOC]
        # Property: ISO 8601 week number computed from _year, month, day.
        """
        Return the week number of the year.

        Returns
        -------
        int

        See Also
        --------
        Timestamp.weekday : Return the day of the week.
        Timestamp.quarter : Return the quarter of the year.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.week
        11
        """
        return ccalendar.get_week_of_year(self._year, self.month, self.day)

    @property
    def days_in_month(self) -> int:
        # [FUNCTION DOC]
        # Property: number of days in the month for _year/month.
        """
        Return the number of days in the month.

        Returns
        -------
        int

        See Also
        --------
        Timestamp.month_name : Return the month name of the Timestamp with
            specified locale.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.days_in_month
        31
        """
        return ccalendar.get_days_in_month(self._year, self.month)

    # -----------------------------------------------------------------
    # Transformation Methods

    def normalize(self) -> "Timestamp":
        # [FUNCTION DOC]
        # Return a Timestamp at midnight of the same local date, keeping the tz. Takes
        # the local wall-time value, floors it to a multiple of periods-per-day for the
        # resolution, builds a naive Timestamp, then tz_localize(self.tzinfo) back.
        # Outputs: Timestamp. Exceptions: ValueError if flooring overflows; tz_localize
        # errors (e.g. nonexistent/ambiguous midnight raises with default "raise").
        """
        Normalize Timestamp to midnight, preserving tz information.

        This method sets the time component of the `Timestamp` to midnight (00:00:00),
        while preserving the date and time zone information. It is useful when you
        need to standardize the time across different `Timestamp` objects without
        altering the time zone or the date.

        Returns
        -------
        Timestamp

        See Also
        --------
        Timestamp.floor : Rounds `Timestamp` down to the nearest frequency.
        Timestamp.ceil : Rounds `Timestamp` up to the nearest frequency.
        Timestamp.round : Rounds `Timestamp` to the nearest frequency.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14, 15, 30)
        >>> ts.normalize()
        Timestamp('2020-03-14 00:00:00')
        """
        cdef:
            local_val = self._maybe_convert_value_to_local()
            int64_t normalized
            int64_t ppd = periods_per_day(self._creso)
            _Timestamp ts

        try:
            normalized = normalize_i8_stamp(local_val, ppd)
        except OverflowError as err:
            raise ValueError(
                "Cannot normalize Timestamp without integer overflow"
            ) from err
        ts = type(self)._from_value_and_reso(normalized, reso=self._creso, tz=None)
        return ts.tz_localize(self.tzinfo)

    # -----------------------------------------------------------------
    # Pickle Methods

    def __reduce_ex__(self, protocol):
        # [FUNCTION DOC]
        # Pickle protocol hook; ignores protocol and returns __reduce__().
        # python 3.6 compat
        # https://bugs.python.org/issue28730
        # now __reduce_ex__ is defined and higher priority than __reduce__
        return self.__reduce__()

    def __setstate__(self, state):
        # [FUNCTION DOC]
        # Legacy unpickle state setter: sets _value from state[0], tzinfo from state[2],
        # and _creso from state[4] (or nanoseconds if len(state) == 3).
        # Inputs: state - tuple. Side effects: mutates self.
        # KNOWN ISSUE: appears to be dead code, since __reduce__ returns a reconstructor
        # callable with no state. It would also fail if used: tzinfo is a read-only
        # datetime attribute, and __reduce__ produces a 4-tuple where the resolution is
        # at index 3, so state[4] would raise IndexError.
        self._value= state[0]
        self.tzinfo = state[2]

        if len(state) == 3:
            # pre-non-nano pickle
            # TODO: no tests get here 2022-05-10
            reso = NPY_FR_ns
        else:
            reso = state[4]
        self._creso = reso

    def __reduce__(self):
        # [FUNCTION DOC]
        # Pickle support: returns (_unpickle_timestamp, (_value, None, tzinfo, _creso)).
        # The None placeholder is the legacy freq slot.
        object_state = self._value, None, self.tzinfo, self._creso
        return (_unpickle_timestamp, object_state)

    # -----------------------------------------------------------------
    # Rendering Methods

    def isoformat(self, sep: str = "T", timespec: str = "auto") -> str:
        # [FUNCTION DOC]
        # ISO 8601 string. Formats via datetime.isoformat (timespec "nanoseconds" is
        # mapped to "microseconds"), then replaces the leading year with the true _year
        # zero-padded to 4 digits. If nanoseconds are present (timespec "auto") or
        # timespec is "nanoseconds", appends 3 nanosecond digits after the microseconds
        # (or, when microsecond is 0 and timespec is "auto", appends ".%09d" of the
        # nanoseconds), inserting before the trailing 6-char UTC offset if tz-aware.
        # Inputs: sep - date/time separator; timespec - stdlib timespec or
        # "nanoseconds". Outputs: str. Exceptions: ValueError for invalid timespec.
        # KNOWN ISSUE: assumes a tz offset string is exactly 6 characters ("+HH:MM"); an
        # offset with seconds (e.g. "+05:30:15") would be split incorrectly.
        """
        Return the time formatted according to ISO 8601.

        The full format looks like 'YYYY-MM-DD HH:MM:SS.mmmmmmnnn'.
        By default, the fractional part is omitted if self.microsecond == 0
        and self._nanosecond == 0.

        If self.tzinfo is not None, the UTC offset is also attached,
        giving a full format of 'YYYY-MM-DD HH:MM:SS.mmmmmmnnn+HH:MM'.

        Parameters
        ----------
        sep : str, default 'T'
            String used as the separator between the date and time.

        timespec : str, default 'auto'
            Specifies the number of additional terms of the time to include.
            The valid values are 'auto', 'hours', 'minutes', 'seconds',
            'milliseconds', 'microseconds', and 'nanoseconds'.

        Returns
        -------
        str

        See Also
        --------
        Timestamp.strftime : Return a formatted string.
        Timestamp.isocalendar : Return a tuple containing ISO year, week number and
            weekday.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts.isoformat()
        '2020-03-14T15:32:52.192548651'
        >>> ts.isoformat(timespec='microseconds')
        '2020-03-14T15:32:52.192548'
        """
        base_ts = "microseconds" if timespec == "nanoseconds" else timespec
        base = super(_Timestamp, self).isoformat(sep=sep, timespec=base_ts)
        # We need to replace the fake year 1970 with our real year
        base = f"{self._year:04d}-" + base.split("-", 1)[1]

        if self._nanosecond == 0 and timespec != "nanoseconds":
            return base

        if self.tzinfo is not None:
            base1, base2 = base[:-6], base[-6:]
        else:
            base1, base2 = base, ""

        if timespec == "nanoseconds" or (timespec == "auto" and self._nanosecond):
            if self.microsecond or timespec == "nanoseconds":
                base1 += f"{self._nanosecond:03d}"
            else:
                base1 += f".{self._nanosecond:09d}"

        return base1 + base2

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # repr: "Timestamp('<date> <time><offset> <zone abbrev>', tz='<zone>')". For
        # aware values appends strftime("%z") (falling back to the same value with year
        # 2000 if strftime raises ValueError, e.g. out-of-range years) and the tz name
        # from get_timezone; " %Z" is added when the zone object has strftime. Outputs:
        # str.
        stamp = self._repr_base
        zone = None

        if self.tzinfo is not None:
            try:
                stamp += self.strftime("%z")
            except ValueError:
                year2000 = self.replace(year=2000)
                stamp += year2000.strftime("%z")

            zone = get_timezone(self.tzinfo)
            try:
                stamp += zone.strftime(" %%Z")
            except AttributeError:
                # e.g. tzlocal has no `strftime`
                pass

        tz = f", tz='{zone}'" if zone is not None else ""

        return f"Timestamp('{stamp}'{tz})"

    @property
    def _repr_base(self) -> str:
        # [FUNCTION DOC]
        # HELPER property: "<_date_repr> <_time_repr>".
        return f"{self._date_repr} {self._time_repr}"

    @property
    def _date_repr(self) -> str:
        # [FUNCTION DOC]
        # HELPER property: "YYYY-MM-DD" using the true year (unpadded) and 2-digit
        # month/day. Avoids strftime, which is slow and limited in year range.
        # Ideal here would be self.strftime("%Y-%m-%d"), but
        # the datetime strftime() methods require year >= 1900 and is slower
        return f"{self._year}-{self.month:02d}-{self.day:02d}"

    @property
    def _time_repr(self) -> str:
        # [FUNCTION DOC]
        # HELPER property: "HH:MM:SS", plus ".fffffffff" (9 digits: microsecond*1000 +
        # nanosecond) if nanoseconds are nonzero, else ".ffffff" if microseconds
        # nonzero.
        result = f"{self.hour:02d}:{self.minute:02d}:{self.second:02d}"

        if self._nanosecond != 0:
            result += f".{self._nanosecond + 1000 * self.microsecond:09d}"
        elif self.microsecond != 0:
            result += f".{self.microsecond:06d}"

        return result

    # -----------------------------------------------------------------
    # Conversion Methods

    @cython.cdivision(False)
    cdef _Timestamp _as_creso(self, NPY_DATETIMEUNIT creso, bint round_ok=True):
        # [FUNCTION DOC]
        # HELPER: Return self converted to resolution creso (self if already equal).
        # Converts _value with convert_reso (round_ok controls whether lossy conversion
        # to a coarser unit rounds or raises) and rebuilds with the same tz.
        # Inputs: creso - target unit; round_ok - bool. Outputs: _Timestamp.
        # Exceptions: OutOfBoundsDatetime on overflow; ValueError from convert_reso if
        # lossy and round_ok is False.
        cdef:
            int64_t value

        if creso == self._creso:
            return self

        try:
            value = convert_reso(self._value, self._creso, creso, round_ok=round_ok)
        except OverflowError as err:
            unit = npy_unit_to_abbrev(creso)
            raise OutOfBoundsDatetime(
                f"Cannot cast {self} to unit='{unit}' without overflow."
            ) from err

        return type(self)._from_value_and_reso(value, reso=creso, tz=self.tzinfo)

    def as_unit(self, str unit, bint round_ok=True):
        # [FUNCTION DOC]
        # Public wrapper of _as_creso taking a unit string ("s", "ms", "us", "ns").
        # Parses the unit through numpy dtype "M8[unit]".
        # Inputs: unit - str; round_ok - bool. Outputs: Timestamp.
        # Exceptions: TypeError for invalid unit strings (from numpy),
        # OutOfBoundsDatetime on overflow, NotImplementedError for unsupported units.
        """
        Convert the underlying int64 representation to the given unit.

        Parameters
        ----------
        unit : {"ns", "us", "ms", "s"}
        round_ok : bool, default True
            If False and the conversion requires rounding, raise.

        Returns
        -------
        Timestamp

        See Also
        --------
        Timestamp.asm8 : Return numpy datetime64 format with same precision.
        Timestamp.to_pydatetime : Convert Timestamp object to a native
            Python datetime object.
        to_timedelta : Convert argument into timedelta object,
            which can represent differences in times.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 00:00:00.01')
        >>> ts
        Timestamp('2023-01-01 00:00:00.010000')
        >>> ts.unit
        'ms'
        >>> ts = ts.as_unit('s')
        >>> ts
        Timestamp('2023-01-01 00:00:00')
        >>> ts.unit
        's'
        """
        dtype = np.dtype(f"M8[{unit}]")
        reso = get_unit_from_dtype(dtype)
        try:
            return self._as_creso(reso, round_ok=round_ok)
        except OverflowError as err:
            raise OutOfBoundsDatetime(
                f"Cannot cast {self} to unit='{unit}' without overflow."
            ) from err

    @property
    def asm8(self) -> np.datetime64:
        # [FUNCTION DOC]
        # Property: numpy datetime64 equivalent (to_datetime64()); naive UTC/wall value.
        """
        Return numpy datetime64 format with same precision.

        See Also
        --------
        numpy.datetime64 : Numpy datatype for dates and times with high precision.
        Timestamp.to_numpy : Convert the Timestamp to a NumPy datetime64.
        to_datetime : Convert argument to datetime.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14, 15)
        >>> ts.asm8
        numpy.datetime64('2020-03-14T15:00:00.000000')
        """
        return self.to_datetime64()

    def timestamp(self):
        # [FUNCTION DOC]
        # POSIX timestamp as float seconds: _value / periods-per-second, rounded to 6
        # decimals. Naive values are treated as UTC (unlike stdlib, which uses local
        # time).
        # Outputs: float.
        """
        Return POSIX timestamp as float.

        This method converts the `Timestamp` object to a POSIX timestamp, which is
        the number of seconds since the Unix epoch (January 1, 1970). The returned
        value is a floating-point number, where the integer part represents the
        seconds, and the fractional part represents the microseconds.

        Returns
        -------
        float
            The POSIX timestamp representation of the `Timestamp` object.

        See Also
        --------
        Timestamp.fromtimestamp : Construct a `Timestamp` from a POSIX timestamp.
        datetime.datetime.timestamp : Equivalent method from the `datetime` module.
        Timestamp.to_pydatetime : Convert the `Timestamp` to a `datetime` object.
        Timestamp.to_datetime64 : Converts `Timestamp` to `numpy.datetime64`.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548')
        >>> ts.timestamp()
        1584199972.192548
        """
        # GH 17329
        # Note: Naive timestamps will not match datetime.stdlib

        denom = periods_per_second(self._creso)

        return round(self._value/ denom, 6)

    cpdef datetime to_pydatetime(_Timestamp self, bint warn=True):
        # [FUNCTION DOC]
        # Convert to datetime.datetime with true year, wall fields, tzinfo and fold,
        # dropping nanoseconds. Inputs: warn - if True and nanoseconds are nonzero,
        # emits UserWarning. Outputs: datetime. Exceptions: ValueError if year is
        # outside 1..9999. Side effects: may emit a warning.
        """
        Convert a Timestamp object to a native Python datetime object.

        This method is useful for when you need to utilize a pandas Timestamp
        object in contexts where native Python datetime objects are expected
        or required. The conversion discards the nanoseconds component, and a
        warning can be issued in such cases if desired.

        Parameters
        ----------
        warn : bool, default True
            If True, issues a warning when the timestamp includes nonzero
            nanoseconds, as these will be discarded during the conversion.

        Returns
        -------
        datetime.datetime or NaT
            Returns a datetime.datetime object representing the timestamp,
            with year, month, day, hour, minute, second, and microsecond components.
            If the timestamp is NaT (Not a Time), returns NaT.

        See Also
        --------
        datetime.datetime : The standard Python datetime class that this method
            returns.
        Timestamp.timestamp : Convert a Timestamp object to POSIX timestamp.
        Timestamp.to_datetime64 : Convert a Timestamp object to numpy.datetime64.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548')
        >>> ts.to_pydatetime()
        datetime.datetime(2020, 3, 14, 15, 32, 52, 192548)

        Analogous for ``pd.NaT``:

        >>> pd.NaT.to_pydatetime()
        NaT
        """
        if self._nanosecond != 0 and warn:
            warnings.warn("Discarding nonzero nanoseconds in conversion.",
                          UserWarning, stacklevel=find_stack_level())

        return datetime(self._year, self.month, self.day,
                        self.hour, self.minute, self.second,
                        self.microsecond, self.tzinfo, fold=self.fold)

    cpdef to_datetime64(self):
        # [FUNCTION DOC]
        # Return numpy datetime64 with payload _value and unit equal to the resolution
        # (tz information is dropped; value is UTC for aware timestamps).
        """
        Return a NumPy datetime64 object with same precision.

        This method returns a numpy.datetime64 object with the same
        date and time information and precision as the pd.Timestamp object.

        See Also
        --------
        numpy.datetime64 : Class to represent dates and times with high precision.
        Timestamp.to_numpy : Alias for this method.
        Timestamp.asm8 : Alias for this method.
        pd.to_datetime : Convert argument to datetime.

        Examples
        --------
        >>> ts = pd.Timestamp(year=2023, month=1, day=1,
        ...                   hour=10, second=15)
        >>> ts
        Timestamp('2023-01-01 10:00:15')
        >>> ts.to_datetime64()
        numpy.datetime64('2023-01-01T10:00:15.000000')
        """
        # TODO: find a way to construct dt64 directly from _reso
        abbrev = npy_unit_to_abbrev(self._creso)
        return np.datetime64(self._value, abbrev)

    def to_numpy(self, dtype=None, copy=False) -> np.datetime64:
        # [FUNCTION DOC]
        # Same as to_datetime64(). Inputs: dtype, copy - must be None/False.
        # Exceptions: ValueError if dtype is not None or copy is not False.
        """
        Convert the Timestamp to a NumPy datetime64.

        This is an alias method for `Timestamp.to_datetime64()`. The dtype and
        copy parameters are available here only for compatibility. Their values
        will not affect the return value.

        Parameters
        ----------
        dtype : dtype, optional
            Data type of the output, ignored in this method as the return type
            is always `numpy.datetime64`.
        copy : bool, default False
            Whether to ensure that the returned value is a new object. This
            parameter is also ignored as the method does not support copying.

        Returns
        -------
        numpy.datetime64

        See Also
        --------
        DatetimeIndex.to_numpy : Similar method for DatetimeIndex.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts.to_numpy()
        numpy.datetime64('2020-03-14T15:32:52.192548651')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.to_numpy()
        numpy.datetime64('NaT', 'ns')
        """
        if dtype is not None or copy is not False:
            raise ValueError(
                "Timestamp.to_numpy dtype and copy arguments are ignored."
            )
        return self.to_datetime64()

    def to_period(self, freq=None):
        # [FUNCTION DOC]
        # Convert to pandas Period of the given freq (None lets Period infer). Emits a
        # UserWarning when tz-aware because tz is dropped. Imports Period lazily.
        # Outputs: Period. Exceptions: as raised by Period.
        """
        Return a period of which this timestamp is an observation.

        This method converts the given Timestamp to a Period object,
        which represents a span of time,such as a year, month, etc.,
        based on the specified frequency.

        Parameters
        ----------
        freq : str, optional
            Frequency string for the period (e.g., 'Y', 'M', 'W'). Defaults to `None`.

        See Also
        --------
        Timestamp : Represents a specific timestamp.
        Period : Represents a span of time.
        to_period : Converts an object to a Period.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> # Year end frequency
        >>> ts.to_period(freq='Y')
        Period('2020', 'Y-DEC')

        >>> # Month end frequency
        >>> ts.to_period(freq='M')
        Period('2020-03', 'M')

        >>> # Weekly frequency
        >>> ts.to_period(freq='W')
        Period('2020-03-09/2020-03-15', 'W-SUN')

        >>> # Quarter end frequency
        >>> ts.to_period(freq='Q')
        Period('2020Q1', 'Q-DEC')
        """
        from pandas import Period

        if self.tz is not None:
            # GH#21333
            warnings.warn(
                "Converting to Period representation will drop timezone information.",
                UserWarning,
                stacklevel=find_stack_level(),
            )

        return Period(self, freq=freq)


# ----------------------------------------------------------------------

# Python front end to C extension type _Timestamp
# This serves as the box for datetime64

@set_module("pandas")
class Timestamp(_Timestamp):
    # [CLASS DOC]
    # Public Python Timestamp class (module reported as "pandas"). Adds the flexible
    # constructor (__new__), stdlib-style alternate constructors (now, today,
    # fromordinal, fromtimestamp, combine...), rounding (round/floor/ceil), timezone
    # handling (tz, tz_localize, tz_convert/astimezone), replace, and overrides of
    # datetime methods that would otherwise use the fake stored year. Aliases weekofyear
    # and daysinmonth are attached after the class.
    """
    Pandas replacement for python datetime.datetime object.

    Timestamp is the pandas equivalent of python's Datetime
    and is interchangeable with it in most cases. It's the type used
    for the entries that make up a DatetimeIndex, and other timeseries
    oriented data structures in pandas.

    Parameters
    ----------
    ts_input : datetime-like, str, int, float
        Value to be converted to Timestamp.
    year : int
        Value of year.
    month : int
        Value of month.
    day : int
        Value of day.
    hour : int, optional, default 0
        Value of hour.
    minute : int, optional, default 0
        Value of minute.
    second : int, optional, default 0
        Value of second.
    microsecond : int, optional, default 0
        Value of microsecond.
    tzinfo : datetime.tzinfo, optional, default None
        Timezone info.
    nanosecond : int, optional, default 0
        Value of nanosecond.
    tz : str, zoneinfo.ZoneInfo, pytz.timezone, dateutil.tz.tzfile or None
        Time zone for time which Timestamp will have.
    unit : str
        Unit used for conversion if ts_input is of type int or float. The
        valid values are 'W', 'D', 'h', 'm', 's', 'ms', 'us', and 'ns'. For
        example, 's' means seconds and 'ms' means milliseconds.

        For float inputs, the result will be stored in nanoseconds, and
        the unit attribute will be set as ``'ns'``.
    fold : {0, 1}, default None, keyword-only
        Due to daylight saving time, one wall clock time can occur twice
        when shifting from summer to winter time; fold describes whether the
        datetime-like corresponds  to the first (0) or the second time (1)
        the wall clock hits the ambiguous time.

    See Also
    --------
    Timedelta : Represents a duration, the difference between two dates or times.
    datetime.datetime : Python datetime.datetime object.

    Notes
    -----
    There are essentially three calling conventions for the constructor. The
    primary form accepts four parameters. They can be passed by position or
    keyword.

    The other two forms mimic the parameters from ``datetime.datetime``. They
    can be passed by either position or keyword, but not both mixed together.

    Examples
    --------
    Using the primary calling convention:

    This converts a datetime-like string

    >>> pd.Timestamp('2017-01-01T12')
    Timestamp('2017-01-01 12:00:00')

    This converts a float representing a Unix epoch in units of seconds

    >>> pd.Timestamp(1513393355.5, unit='s')
    Timestamp('2017-12-16 03:02:35.500000')

    This converts an int representing a Unix-epoch in units of weeks

    >>> pd.Timestamp(1535, unit='W')
    Timestamp('1999-06-03 00:00:00')

    This converts an int representing a Unix-epoch in units of seconds
    and for a particular timezone

    >>> pd.Timestamp(1513393355, unit='s', tz='US/Pacific')
    Timestamp('2017-12-15 19:02:35-0800', tz='US/Pacific')

    Using the other two forms that mimic the API for ``datetime.datetime``:

    >>> pd.Timestamp(2017, 1, 1, 12)
    Timestamp('2017-01-01 12:00:00')

    >>> pd.Timestamp(year=2017, month=1, day=1, hour=12)
    Timestamp('2017-01-01 12:00:00')
    """

    @classmethod
    def fromordinal(cls, ordinal, tz=None):
        # [FUNCTION DOC]
        # Classmethod: Timestamp from proleptic Gregorian ordinal (day 1 = 0001-01-01)
        # via datetime.fromordinal, then localized/interpreted with tz by the
        # constructor.
        # Inputs: ordinal - int; tz - tz or None. Outputs: Timestamp.
        # Exceptions: ValueError for invalid ordinals.
        """
        Construct a timestamp from a proleptic Gregorian ordinal.

        This method creates a `Timestamp` object corresponding to the given
        proleptic Gregorian ordinal, which is a count of days from January 1,
        0001 (using the proleptic Gregorian calendar). The time part of the
        `Timestamp` is set to midnight (00:00:00) by default.

        Parameters
        ----------
        ordinal : int
            Date corresponding to a proleptic Gregorian ordinal.
        tz : str, zoneinfo.ZoneInfo, pytz.timezone, dateutil.tz.tzfile or None
            Time zone for the Timestamp.

        Returns
        -------
        Timestamp
            A `Timestamp` object representing the specified ordinal date.

        See Also
        --------
        Timestamp : Represents a single timestamp, similar to `datetime`.
        to_datetime : Converts various types of data to datetime.

        Notes
        -----
        By definition there cannot be any tz info on the ordinal itself.

        Examples
        --------
        Convert an ordinal to a `Timestamp`:

        >>> pd.Timestamp.fromordinal(737425)
        Timestamp('2020-01-01 00:00:00')

        Create a `Timestamp` from an ordinal with timezone information:

        >>> pd.Timestamp.fromordinal(737425, tz='UTC')
        Timestamp('2020-01-01 00:00:00+0000', tz='UTC')
        """
        return cls(datetime.fromordinal(ordinal), tz=tz)

    @classmethod
    def now(cls, tz=None):
        # [FUNCTION DOC]
        # Classmethod: current time. String tz is resolved with maybe_get_tz first, then
        # datetime.now(tz) is wrapped. Inputs: tz - str, tzinfo or None (local naive).
        # Outputs: Timestamp (microsecond resolution). Side effects: reads the clock.
        """
        Return new Timestamp object representing current time local to tz.

        This method returns a new `Timestamp` object that represents the current time.
        If a timezone is provided, either through a timezone object or an IANA
        standard timezone identifier, the current time will be localized to that
        timezone.
        Otherwise, it returns the current local time.

        Parameters
        ----------
        tz : str or timezone object, default None
            Timezone to localize to.

        See Also
        --------
        to_datetime : Convert argument to datetime.
        Timestamp.utcnow : Return a new Timestamp representing UTC day and time.
        Timestamp.today : Return the current time in the local timezone.

        Examples
        --------
        >>> pd.Timestamp.now()  # doctest: +SKIP
        Timestamp('2020-11-16 22:06:16.378782')

        If you want a specific timezone, in this case 'Brazil/East':

        >>> pd.Timestamp.now('Brazil/East')  # doctest: +SKIP
        Timestamp('2025-11-11 22:17:59.609943-03:00)

        Analogous for ``pd.NaT``:

        >>> pd.NaT.now()
        NaT
        """
        if isinstance(tz, str):
            tz = maybe_get_tz(tz)
        return cls(datetime.now(tz))

    @classmethod
    def today(cls, tz=None):
        # [FUNCTION DOC]
        # Classmethod: same as now(tz). Side effects: reads the clock.
        """
        Return the current time in the local timezone.

        This differs from datetime.today() in that it can be localized to a
        passed timezone.

        Parameters
        ----------
        tz : str or timezone object, default None
            Timezone to localize to.

        See Also
        --------
        datetime.datetime.today : Returns the current local date.
        Timestamp.now : Returns current time with optional timezone.
        Timestamp : A class representing a specific timestamp.

        Examples
        --------
        >>> pd.Timestamp.today()    # doctest: +SKIP
        Timestamp('2020-11-16 22:37:39.969883')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.today()
        NaT
        """
        return cls.now(tz)

    @classmethod
    def utcnow(cls):
        # [FUNCTION DOC]
        # Deprecated classmethod: current time in UTC (now(UTC)). Emits Pandas4Warning.
        """
        Timestamp.utcnow()

        Return a new Timestamp representing UTC day and time.

        See Also
        --------
        Timestamp : Constructs an arbitrary datetime.
        Timestamp.now : Return the current local date and time, which
            can be timezone-aware.
        Timestamp.today : Return the current local date and time with
            timezone information set to None.
        to_datetime : Convert argument to datetime.
        date_range : Return a fixed frequency DatetimeIndex.
        Timestamp.utctimetuple : Return UTC time tuple, compatible with
            time.localtime().

        Examples
        --------
        >>> pd.Timestamp.utcnow()   # doctest: +SKIP
        Timestamp('2020-11-16 22:50:18.092888+0000', tz='UTC')
        """
        from pandas.errors import Pandas4Warning

        warnings.warn(
            # The stdlib datetime.utcnow is deprecated, so we deprecate to match.
            #  GH#56680
            "Timestamp.utcnow is deprecated and will be removed in a future "
            "version. Use Timestamp.now('UTC') instead.",
            Pandas4Warning,
            stacklevel=find_stack_level(),
        )
        return cls.now(UTC)

    @classmethod
    def utcfromtimestamp(cls, ts):
        # [FUNCTION DOC]
        # Deprecated classmethod: fromtimestamp(ts, tz="UTC"). Emits Pandas4Warning.
        # Inputs: ts - POSIX seconds (float/int). Outputs: UTC-aware Timestamp.
        """
        Timestamp.utcfromtimestamp(ts)

        Construct a timezone-aware UTC datetime from a POSIX timestamp.

        This method creates a datetime object from a POSIX timestamp, keeping the
        Timestamp object's timezone.

        Parameters
        ----------
        ts : float
            POSIX timestamp.

        See Also
        --------
        Timezone.tzname : Return time zone name.
        Timestamp.utcnow : Return a new Timestamp representing UTC day and time.
        Timestamp.fromtimestamp : Transform timestamp[, tz] to tz's local
            time from POSIX timestamp.

        Notes
        -----
        Timestamp.utcfromtimestamp behavior differs from datetime.utcfromtimestamp
        in returning a timezone-aware object.

        Examples
        --------
        >>> pd.Timestamp.utcfromtimestamp(1584199972)
        Timestamp('2020-03-14 15:32:52+0000', tz='UTC')
        """
        from pandas.errors import Pandas4Warning

        # GH#22451
        warnings.warn(
            # The stdlib datetime.utcfromtimestamp is deprecated, so we deprecate
            #  to match. GH#56680
            "Timestamp.utcfromtimestamp is deprecated and will be removed in a "
            "future version. Use Timestamp.fromtimestamp(ts, 'UTC') instead.",
            Pandas4Warning,
            stacklevel=find_stack_level(),
        )
        return cls.fromtimestamp(ts, tz="UTC")

    @classmethod
    def fromtimestamp(cls, ts, tz=None):
        # [FUNCTION DOC]
        # Classmethod: Timestamp from POSIX seconds via datetime.fromtimestamp(ts, tz)
        # after resolving tz with maybe_get_tz; naive result uses the system local time.
        # Inputs: ts - number; tz - str/tzinfo/None. Outputs: Timestamp.
        # Exceptions: OverflowError/OSError/ValueError from the stdlib.
        """
        Create a `Timestamp` object from a POSIX timestamp.

        This method converts a POSIX timestamp (the number of seconds since
        January 1, 1970, 00:00:00 UTC) into a `Timestamp` object. The resulting
        `Timestamp` can be localized to a specific time zone if provided.

        Parameters
        ----------
        ts : float
            The POSIX timestamp to convert, representing seconds since
            the epoch (1970-01-01 00:00:00 UTC).
        tz : str, zoneinfo.ZoneInfo, pytz.timezone, dateutil.tz.tzfile, optional
            Time zone for the `Timestamp`. If not provided, the `Timestamp` will
            be timezone-naive (i.e., without time zone information).

        Returns
        -------
        Timestamp
            A `Timestamp` object representing the given POSIX timestamp.

        See Also
        --------
        Timestamp : Represents a single timestamp, similar to `datetime`.
        to_datetime : Converts various types of data to datetime.
        datetime.datetime.fromtimestamp : Returns a datetime from a POSIX timestamp.

        Examples
        --------
        Convert a POSIX timestamp to a `Timestamp`:

        >>> pd.Timestamp.fromtimestamp(1584199972)  # doctest: +SKIP
        Timestamp('2020-03-14 15:32:52')

        Note that the output may change depending on your local time and time zone:

        >>> pd.Timestamp.fromtimestamp(1584199972, tz='UTC')  # doctest: +SKIP
        Timestamp('2020-03-14 15:32:52+0000', tz='UTC')
        """
        tz = maybe_get_tz(tz)
        return cls(datetime.fromtimestamp(ts, tz))

    def strftime(self, format):
        # [FUNCTION DOC]
        # Format with stdlib strftime on an equivalent datetime (true year; nanoseconds
        # are dropped). Inputs: format - str. Outputs: str.
        # Exceptions: NotImplementedError if the year is outside the datetime range.
        """
        Return a formatted string of the Timestamp.

        Parameters
        ----------
        format : str
            Format string to convert Timestamp to string.
            See strftime documentation for more information on the format string:
            https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior.

        See Also
        --------
        Timestamp.isoformat : Return the time formatted according to ISO 8601.
        pd.to_datetime : Convert argument to datetime.
        Period.strftime : Format a single Period.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts.strftime('%Y-%m-%d %X')
        '2020-03-14 15:32:52'
        """
        try:
            _dt = datetime(self._year, self.month, self.day,
                           self.hour, self.minute, self.second,
                           self.microsecond, self.tzinfo, fold=self.fold)
        except ValueError as err:
            raise NotImplementedError(
                "strftime not yet supported on Timestamps which "
                "are outside the range of Python's standard library. "
                "For now, please call the components you need (such as `.year` "
                "and `.month`) and construct your string from there."
            ) from err
        return _dt.strftime(format)

    def ctime(self):
        # [FUNCTION DOC]
        # ctime() string of the equivalent datetime (true year).
        # Exceptions: NotImplementedError if the year is outside the datetime range.
        """
        Return a ctime() style string representing the Timestamp.

        This method returns a string representing the date and time
        in the format returned by the standard library's `time.ctime()`
        function, which is typically in the form 'Day Mon DD HH:MM:SS YYYY'.

        If the `Timestamp` is outside the range supported by Python's
        standard library, a `NotImplementedError` is raised.

        Returns
        -------
        str
            A string representing the Timestamp in ctime format.

        See Also
        --------
        time.ctime : Return a string representing time in ctime format.
        Timestamp : Represents a single timestamp, similar to `datetime`.
        datetime.datetime.ctime : Return a ctime style string from a datetime object.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00.00')
        >>> ts
        Timestamp('2023-01-01 10:00:00')
        >>> ts.ctime()
        'Sun Jan  1 10:00:00 2023'
        """
        try:
            _dt = datetime(self._year, self.month, self.day,
                           self.hour, self.minute, self.second,
                           self.microsecond, self.tzinfo, fold=self.fold)
        except ValueError as err:
            raise NotImplementedError(
                "ctime not yet supported on Timestamps which "
                "are outside the range of Python's standard library. "
                "For now, please call the components you need (such as `.year` "
                "and `.month`) and construct your string from there."
            ) from err
        return _dt.ctime()

    def date(self):
        # [FUNCTION DOC]
        # Return datetime.date(true year, month, day).
        # Exceptions: NotImplementedError if the year is outside the date range.
        """
        Returns `datetime.date` with the same year, month, and day.

        This method extracts the date component from the `Timestamp` and returns
        it as a `datetime.date` object, discarding the time information.

        Returns
        -------
        datetime.date
            The date part of the `Timestamp`.

        See Also
        --------
        Timestamp : Represents a single timestamp, similar to `datetime`.
        datetime.datetime.date : Extract the date component from a `datetime` object.

        Examples
        --------
        Extract the date from a Timestamp:

        >>> ts = pd.Timestamp('2023-01-01 10:00:00.00')
        >>> ts
        Timestamp('2023-01-01 10:00:00')
        >>> ts.date()
        datetime.date(2023, 1, 1)
        """
        try:
            _dt = dt.date(self._year, self.month, self.day)
        except ValueError as err:
            raise NotImplementedError(
                "date not yet supported on Timestamps which "
                "are outside the range of Python's standard library. "
            ) from err
        return _dt

    def dst(self):
        # [FUNCTION DOC]
        # Daylight saving offset from tzinfo.dst via datetime.dst (None if naive).
        # KNOWN ISSUE: uses the stored datetime fields, so for years outside 1..9999 the
        # tz is queried for the fake year (1970/1972), which may give a different DST
        # state than the true date.
        """
        Return the daylight saving time (DST) adjustment.

        This method returns the DST adjustment as a `datetime.timedelta` object
        if the Timestamp is timezone-aware and DST is applicable.

        See Also
        --------
        Timestamp.tz_localize : Localize the Timestamp to a timezone.
        Timestamp.tz_convert : Convert timezone-aware Timestamp to another time zone.

        Examples
        --------
        >>> ts = pd.Timestamp('2000-06-01 00:00:00', tz='Europe/Brussels')
        >>> ts
        Timestamp('2000-06-01 00:00:00+0200', tz='Europe/Brussels')
        >>> ts.dst()
        datetime.timedelta(seconds=3600)
        """
        return super().dst()

    def isocalendar(self):
        # [FUNCTION DOC]
        # ISO calendar (year, week, weekday) of the equivalent datetime (true year).
        # Exceptions: NotImplementedError if the year is outside the datetime range.
        """
        Return a named tuple containing ISO year, week number, and weekday.

        See Also
        --------
        DatetimeIndex.isocalendar : Return a 3-tuple containing ISO year,
            week number, and weekday for the given DatetimeIndex object.
        datetime.date.isocalendar : The equivalent method for `datetime.date` objects.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00')
        >>> ts
        Timestamp('2023-01-01 10:00:00')
        >>> ts.isocalendar()
        datetime.IsoCalendarDate(year=2022, week=52, weekday=7)
        """
        try:
            _dt = datetime(self._year, self.month, self.day,
                           self.hour, self.minute, self.second,
                           self.microsecond, self.tzinfo, fold=self.fold)
        except ValueError as err:
            raise NotImplementedError(
                "isocalendar not yet supported on Timestamps which "
                "are outside the range of Python's standard library. "
            ) from err
        return _dt.isocalendar()

    def tzname(self):
        # [FUNCTION DOC]
        # Time zone name via datetime.tzname (None if naive). Same fake-year caveat as
        # dst() for out-of-range years.
        """
        Return time zone name.

        This method returns the name of the Timestamp's time zone as a string.

        See Also
        --------
        Timestamp.tzinfo : Returns the timezone information of the Timestamp.
        Timestamp.tz_convert : Convert timezone-aware Timestamp to another time zone.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00', tz='Europe/Brussels')
        >>> ts
        Timestamp('2023-01-01 10:00:00+0100', tz='Europe/Brussels')
        >>> ts.tzname()
        'CET'
        """
        return super().tzname()

    @property
    def tzinfo(self):
        # [FUNCTION DOC]
        # Property: the tzinfo stored in the underlying datetime (None if naive).
        """
        Returns the timezone info of the Timestamp.

        This property returns a `datetime.tzinfo` object if the Timestamp
        is timezone-aware. If the Timestamp has no timezone, it returns `None`.
        If the Timestamp is in UTC or a fixed-offset timezone,
        it returns `datetime.timezone`. If the Timestamp uses an
        IANA timezone (e.g., "America/New_York"), it returns `zoneinfo.ZoneInfo`.

        See Also
        --------
        Timestamp.tz : Alias for `tzinfo`, may return a `zoneinfo.ZoneInfo` object.
        Timestamp.tz_convert : Convert timezone-aware Timestamp to another time zone.
        Timestamp.tz_localize : Localize the Timestamp to a specific timezone.

        Examples
        --------
        >>> ts = pd.Timestamp("2023-01-01 12:00:00", tz="UTC")
        >>> ts.tzinfo
        datetime.timezone.utc

        >>> ts_naive = pd.Timestamp("2023-01-01 12:00:00")
        >>> ts_naive.tzinfo
        """
        return super().tzinfo

    def utcoffset(self):
        # [FUNCTION DOC]
        # UTC offset via datetime.utcoffset (None if naive). Same fake-year caveat as
        # dst() for out-of-range years.
        """
        Return utc offset.

        This method returns the difference between UTC and the local time
        as a `timedelta` object. It is useful for understanding the time
        difference between the current timezone and UTC.

        Returns
        -------
        timedelta
            The difference between UTC and the local time as a `timedelta` object.

        See Also
        --------
        datetime.datetime.utcoffset :
            Standard library method to get the UTC offset of a datetime object.
        Timestamp.tzname : Return the name of the timezone.
        Timestamp.dst : Return the daylight saving time (DST) adjustment.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00', tz='Europe/Brussels')
        >>> ts
        Timestamp('2023-01-01 10:00:00+0100', tz='Europe/Brussels')
        >>> ts.utcoffset()
        datetime.timedelta(seconds=3600)
        """
        return super().utcoffset()

    def utctimetuple(self):
        # [FUNCTION DOC]
        # UTC time tuple via datetime.utctimetuple.
        # KNOWN ISSUE: unlike timetuple(), this does not rebuild a datetime with the
        # true year, so for years outside 1..9999 the tuple reports the fake year
        # 1970/1972.
        """
        Return UTC time tuple, compatible with `time.localtime()`.

        This method converts the Timestamp to UTC and returns a time tuple
        containing 9 components: year, month, day, hour, minute, second,
        weekday, day of year, and DST flag. This is particularly useful for
        converting a Timestamp to a format compatible with time module functions.

        Returns
        -------
        time.struct_time
            A time.struct_time object representing the UTC time.

        See Also
        --------
        datetime.datetime.utctimetuple :
            Return UTC time tuple, compatible with time.localtime().
        Timestamp.timetuple : Return time tuple of local time.
        time.struct_time : Time tuple structure used by time functions.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00', tz='Europe/Brussels')
        >>> ts
        Timestamp('2023-01-01 10:00:00+0100', tz='Europe/Brussels')
        >>> ts.utctimetuple()
        time.struct_time(tm_year=2023, tm_mon=1, tm_mday=1, tm_hour=9,
        tm_min=0, tm_sec=0, tm_wday=6, tm_yday=1, tm_isdst=0)
        """
        return super().utctimetuple()

    def time(self):
        # [FUNCTION DOC]
        # Return the wall time as datetime.time (no tzinfo; nanoseconds dropped).
        """
        Return time object with same time but with tzinfo=None.

        This method extracts the time part of the `Timestamp` object, excluding any
        timezone information. It returns a `datetime.time` object which only represents
        the time (hours, minutes, seconds, and microseconds).

        See Also
        --------
        Timestamp.date : Return date object with same year, month and day.
        Timestamp.tz_convert : Convert timezone-aware Timestamp to another time zone.
        Timestamp.tz_localize : Localize the Timestamp to a timezone.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00')
        >>> ts
        Timestamp('2023-01-01 10:00:00')
        >>> ts.time()
        datetime.time(10, 0)
        """
        return super().time()

    def timetuple(self):
        # [FUNCTION DOC]
        # time.struct_time of the equivalent datetime (true year).
        # Exceptions: NotImplementedError if the year is outside the datetime range.
        """
        Return time tuple, compatible with time.localtime().

        This method converts the `Timestamp` into a time tuple, which is compatible
        with functions like `time.localtime()`. The time tuple is a named tuple with
        attributes such as year, month, day, hour, minute, second, weekday,
        day of the year, and daylight savings indicator.

        See Also
        --------
        time.localtime : Converts a POSIX timestamp into a time tuple.
        Timestamp : The `Timestamp` that represents a specific point in time.
        datetime.datetime.timetuple : Equivalent method in the `datetime` module.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00')
        >>> ts
        Timestamp('2023-01-01 10:00:00')
        >>> ts.timetuple()
        time.struct_time(tm_year=2023, tm_mon=1, tm_mday=1,
        tm_hour=10, tm_min=0, tm_sec=0, tm_wday=6, tm_yday=1, tm_isdst=-1)
        """
        try:
            _dt = datetime(self._year, self.month, self.day,
                           self.hour, self.minute, self.second,
                           self.microsecond, self.tzinfo, fold=self.fold)
        except ValueError as err:
            raise NotImplementedError(
                "timetuple not yet supported on Timestamps which "
                "are outside the range of Python's standard library. "
            ) from err
        return _dt.timetuple()

    def timetz(self):
        # [FUNCTION DOC]
        # Return the wall time as datetime.time including tzinfo (nanoseconds dropped).
        """
        Return time object with same time and tzinfo.

        This method returns a datetime.time object with
        the time and tzinfo corresponding to the pd.Timestamp
        object, ignoring any information about the day/date.

        See Also
        --------
        datetime.datetime.timetz : Return datetime.time object with the
            same time attributes as the datetime object.
        datetime.time : Class to represent the time of day, independent
            of any particular day.
        datetime.datetime.tzinfo : Attribute of datetime.datetime objects
            representing the timezone of the datetime object.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00', tz='Europe/Brussels')
        >>> ts
        Timestamp('2023-01-01 10:00:00+0100', tz='Europe/Brussels')
        >>> ts.timetz()
        datetime.time(10, 0, tzinfo=<DstTzInfo 'Europe/Brussels' CET+1:00:00 STD>)
        """
        return super().timetz()

    def toordinal(self):
        # [FUNCTION DOC]
        # Proleptic Gregorian ordinal of the date (true year).
        # Exceptions: NotImplementedError if the year is outside the datetime range.
        """
        Return proleptic Gregorian ordinal. January 1 of year 1 is day 1.

        The proleptic Gregorian ordinal is a continuous count of days since
        January 1 of year 1, which is considered day 1. This method converts
        the `Timestamp` to its equivalent ordinal number, useful for date arithmetic
        and comparison operations.

        See Also
        --------
        datetime.datetime.toordinal : Equivalent method in the `datetime` module.
        Timestamp : The `Timestamp` that represents a specific point in time.
        Timestamp.fromordinal : Create a `Timestamp` from an ordinal.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:50')
        >>> ts
        Timestamp('2023-01-01 10:00:50')
        >>> ts.toordinal()
        738521
        """
        try:
            _dt = datetime(self._year, self.month, self.day,
                           self.hour, self.minute, self.second,
                           self.microsecond, self.tzinfo, fold=self.fold)
        except ValueError as err:
            raise NotImplementedError(
                "toordinal not yet supported on Timestamps which "
                "are outside the range of Python's standard library. "
            ) from err
        return _dt.toordinal()

    # Issue 25016.
    @classmethod
    def strptime(cls, date_string, format):
        # [FUNCTION DOC]
        # Classmethod placeholder: always raises NotImplementedError (use to_datetime).
        """
        Convert string argument to datetime.

        This method is not implemented; calling it will raise NotImplementedError.
        Use pd.to_datetime() instead.

        Parameters
        ----------
        date_string : str
            String to convert to a datetime.
        format : str, default None
            The format string to parse time, e.g. "%d/%m/%Y".

        See Also
        --------
        pd.to_datetime : Convert argument to datetime.
        datetime.datetime.strptime : Return a datetime corresponding to a string
            representing a date and time, parsed according to a separate
            format string.
        datetime.datetime.strftime : Return a string representing the date and
            time, controlled by an explicit format string.
        Timestamp.isoformat : Return the time formatted according to ISO 8601.

        Examples
        --------
        >>> pd.Timestamp.strptime("2023-01-01", "%d/%m/%y")
        Traceback (most recent call last):
        NotImplementedError
        """
        raise NotImplementedError(
            "Timestamp.strptime() is not implemented. "
            "Use to_datetime() to parse date strings."
        )

    @classmethod
    def combine(cls, date, time):
        # [FUNCTION DOC]
        # Classmethod: Timestamp from datetime.combine(date, time) (tz taken from time).
        # Inputs: date - datetime.date; time - datetime.time. Outputs: Timestamp.
        """
        Timestamp.combine(date, time)

        Combine a date and time into a single Timestamp object.

        This method takes a `date` object and a `time` object
        and combines them into a single `Timestamp`
        that has the same date and time fields.

        Parameters
        ----------
        date : datetime.date
            The date part of the Timestamp.
        time : datetime.time
            The time part of the Timestamp.

        Returns
        -------
        Timestamp
            A new `Timestamp` object representing the combined date and time.

        See Also
        --------
        Timestamp : Represents a single timestamp, similar to `datetime`.
        to_datetime : Converts various types of data to datetime.

        Examples
        --------
        >>> from datetime import date, time
        >>> pd.Timestamp.combine(date(2020, 3, 14), time(15, 30, 15))
        Timestamp('2020-03-14 15:30:15')
        """
        return cls(datetime.combine(date, time))

    def __new__(
        cls,
        object ts_input=_no_input,
        year=None,
        month=None,
        day=None,
        hour=None,
        minute=None,
        second=None,
        microsecond=None,
        tzinfo_type tzinfo=None,
        *,
        nanosecond=None,
        tz=_no_input,
        unit=None,
        fold=None,
    ):
        # [FUNCTION DOC]
        # Constructor. Steps:
        # 1. tz defaults to None via the _no_input sentinel; remembers whether tz=None
        #    was passed explicitly. tzinfo (must be a tzinfo object) is moved into tz;
        #    passing both raises ValueError.
        # 2. fold, if given: must be 0/1 (ValueError); only allowed when ts_input is
        #    absent or a naive datetime (ValueError); disallowed with pytz zones
        #    (ValueError); applied to ts_input via replace(fold=...) if it has fold.
        # 3. unit given with a non-numeric ts_input emits UserWarning (unit ignored).
        # 4. A _Timestamp with no other arguments is returned unchanged (identity). A
        #    str ts_input with any date attribute arguments raises ValueError. With no
        #    ts_input, a datetime is built from year/month/day keywords (missing ones
        #    produce the stdlib error) and time parts defaulting to 0. If year is an
        #    integer (positional form), ts_input is treated as the year and the
        #    following positional args shift by one into month/day/hour/... (so the
        #    "year" parameter carries the month, etc.), building a datetime; unit reset.
        # 5. ts_input carrying tzinfo plus tz raises ValueError. tz is resolved with
        #    maybe_get_tz; nanosecond must be in 0..999 (ValueError).
        # 6. convert_to_tsobject parses/converts ts_input (str, int/float with unit,
        #    datetime, date, datetime64, ...), localizing to tz; NaT input returns NaT;
        #    aware data with explicit tz=None raises ValueError.
        # 7. Builds the Timestamp with create_timestamp_from_ts.
        # Outputs: Timestamp or NaT. Exceptions: ValueError, TypeError,
        # OutOfBoundsDatetime, parsing errors. Side effects: may warn.
        # KNOWN ISSUE: in the positional form an 8th positional argument (documented as
        # tzinfo) lands in the microsecond parameter and is silently ignored, e.g.
        # Timestamp(2020, 1, 2, 3, 4, 5, 6, timezone.utc) returns a tz-naive value.
        # The parameter list folds together legacy parameter names (the first
        # four) and positional and keyword parameter names from pydatetime.
        #
        # There are three calling forms:
        #
        # - In the legacy form, the first parameter, ts_input, is required
        #   and may be datetime-like, str, int, or float. The second
        #   parameter, offset, is optional and may be str or DateOffset.
        #
        # - ints in the first, second, and third arguments indicate
        #   pydatetime positional arguments. Only the first 8 arguments
        #   (standing in for year, month, day, hour, minute, second,
        #   microsecond, tzinfo) may be non-None. As a shortcut, we just
        #   check that the second argument is an int.
        #
        # - Nones for the first four (legacy) arguments indicate pydatetime
        #   keyword arguments. year, month, and day are required. As a
        #   shortcut, we just check that the first argument was not passed.
        #
        # Mixing pydatetime positional and keyword arguments is forbidden!

        cdef:
            _TSObject ts
            tzinfo_type tzobj

        _date_attributes = [year, month, day, hour, minute, second,
                            microsecond, nanosecond]

        explicit_tz_none = tz is None
        if tz is _no_input:
            tz = None

        if tzinfo is not None:
            # GH#17690 tzinfo must be a datetime.tzinfo object, ensured
            #  by the cython annotation.
            if tz is not None:
                raise ValueError("Can provide at most one of tz, tzinfo")

            # User passed tzinfo instead of tz; avoid silently ignoring
            tz, tzinfo = tzinfo, None

        # Allow fold only for unambiguous input
        if fold is not None:
            if fold not in [0, 1]:
                raise ValueError(
                    "Valid values for the fold argument are None, 0, or 1."
                )

            if (ts_input is not _no_input and not (
                    PyDateTime_Check(ts_input) and
                    getattr(ts_input, "tzinfo", None) is None)):
                raise ValueError(
                    "Cannot pass fold with possibly unambiguous input: int, "
                    "float, numpy.datetime64, str, or timezone-aware "
                    "datetime-like. Pass naive datetime-like or build "
                    "Timestamp from components."
                )

            if tz is not None and PyTZInfo_Check(tz) and treat_tz_as_pytz(tz):
                raise ValueError(
                    "pytz timezones do not support fold. Please use dateutil "
                    "timezones."
                )

            if hasattr(ts_input, "fold"):
                ts_input = ts_input.replace(fold=fold)

        if (
            unit is not None
            and not (is_float_object(ts_input) or is_integer_object(ts_input))
        ):
            # GH#53198
            warnings.warn(
                "The 'unit' keyword is only used when the Timestamp input is "
                f"an integer or float, not {type(ts_input).__name__}. "
                "To specify the storage unit of the output use `ts.as_unit(unit)`",
                UserWarning,
                stacklevel=find_stack_level(),
            )

        # GH 30543 if pd.Timestamp already passed, return it
        # check that only ts_input is passed
        # checking verbosely, because cython doesn't optimize
        # list comprehensions (as of cython 0.29.x)
        if (isinstance(ts_input, _Timestamp) and
                tz is None and unit is None and year is None and
                month is None and day is None and hour is None and
                minute is None and second is None and
                microsecond is None and nanosecond is None and
                tzinfo is None):
            return ts_input
        elif isinstance(ts_input, str):
            # User passed a date string to parse.
            # Check that the user didn't also pass a date attribute kwarg.
            if any(arg is not None for arg in _date_attributes):
                raise ValueError(
                    "Cannot pass a date attribute keyword "
                    "argument when passing a date string; 'tz' is keyword-only"
                )

        elif ts_input is _no_input:
            # GH 31200
            # When year, month or day is not given, we call the datetime
            # constructor to make sure we get the same error message
            # since Timestamp inherits datetime
            datetime_kwargs = {
                "hour": hour or 0,
                "minute": minute or 0,
                "second": second or 0,
                "microsecond": microsecond or 0,
                "fold": fold or 0
            }
            if year is not None:
                datetime_kwargs["year"] = year
            if month is not None:
                datetime_kwargs["month"] = month
            if day is not None:
                datetime_kwargs["day"] = day

            ts_input = datetime(**datetime_kwargs)

        elif is_integer_object(year):
            # User passed positional arguments:
            # Timestamp(year, month, day[, hour[, minute[, second[,
            # microsecond[, tzinfo]]]]])
            ts_input = datetime(ts_input, year, month, day or 0,
                                hour or 0, minute or 0, second or 0, fold=fold or 0)
            unit = None

        if getattr(ts_input, "tzinfo", None) is not None and tz is not None:
            raise ValueError("Cannot pass a datetime or Timestamp with tzinfo with "
                             "the tz parameter. Use tz_convert instead.")

        tzobj = maybe_get_tz(tz)

        if nanosecond is None:
            nanosecond = 0
        elif not (999 >= nanosecond >= 0):
            raise ValueError("nanosecond must be in 0..999")

        ts = convert_to_tsobject(ts_input, tzobj, unit, 0, 0, nanosecond)

        if ts.value == NPY_NAT:
            return NaT

        if ts.tzinfo is not None and explicit_tz_none:
            raise ValueError(
                "Passed data is timezone-aware, incompatible with 'tz=None'."
            )

        return create_timestamp_from_ts(ts.value, ts.dts, ts.tzinfo, ts.fold, ts.creso)

    def _round(self, freq, mode, ambiguous="raise", nonexistent="raise"):
        # [FUNCTION DOC]
        # HELPER: Shared implementation of round/floor/ceil. Converts freq to a
        # non-period offset and gets its length in units of self's resolution. If that
        # is 0 (freq finer than the resolution) returns self, unless freq itself has
        # zero length (ValueError). Works on the naive wall-clock value
        # (tz_localize(None) when aware), rounds a 1-element int64 array with
        # round_nsint64 in the given mode, rebuilds a naive Timestamp and re-localizes
        # to the original tz using ambiguous/nonexistent.
        # Inputs: freq - offset or alias string (fixed-length only); mode - RoundTo
        # enum; ambiguous - bool, "raise" or "NaT"; nonexistent - "raise", "NaT",
        # "shift_forward", "shift_backward" or timedelta.
        # Outputs: Timestamp (or NaT via localization). Exceptions: ValueError for
        # non-fixed freqs or zero-length freq; OutOfBoundsDatetime on overflow;
        # ambiguous/nonexistent time errors from tz_localize.
        cdef:
            int64_t nanos

        freq = to_offset(freq, is_period=False)
        nanos = get_unit_for_round(freq, self._creso)
        if nanos == 0:
            if freq.nanos == 0:
                raise ValueError("Division by zero in rounding")

            # e.g. self.unit == "s" and sub-second freq
            return self

        if self.tz is not None:
            value = self.tz_localize(None)._value
        else:
            value = self._value

        value = np.array([value], dtype=np.int64)

        # Will only ever contain 1 element for timestamp
        try:
            r = round_nsint64(value, mode, nanos)[0]
        except OverflowError as err:
            raise OutOfBoundsDatetime(
                f"Cannot round {self} to freq={freq} without overflow"
            ) from err

        result = Timestamp._from_value_and_reso(r, self._creso, None)
        if self.tz is not None:
            result = result.tz_localize(
                self.tz, ambiguous=ambiguous, nonexistent=nonexistent
            )
        return result

    def round(self, freq, ambiguous="raise", nonexistent="raise"):
        # [FUNCTION DOC]
        # Round to nearest multiple of freq, ties to even (_round with
        # NEAREST_HALF_EVEN).
        """
        Round the Timestamp to the specified resolution.

        This method rounds the given Timestamp down to a specified frequency
        level. It is particularly useful in data analysis to normalize timestamps
        to regular frequency intervals. For instance, rounding to the nearest
        minute, hour, or day can help in time series comparisons or resampling
        operations.

        Parameters
        ----------
        freq : str
            Frequency string indicating the rounding resolution.
        ambiguous : bool or {'raise', 'NaT'}, default 'raise'
            The behavior is as follows:

            * bool contains flags to determine if time is dst or not (note
              that this flag is only applicable for ambiguous fall dst dates).
            * 'NaT' will return NaT for an ambiguous time.
            * 'raise' will raise a ValueError for an ambiguous time.

        nonexistent : {'raise', 'shift_forward', 'shift_backward', 'NaT', \
timedelta}, default 'raise'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            * 'shift_forward' will shift the nonexistent time forward to the
              closest existing time.
            * 'shift_backward' will shift the nonexistent time backward to the
              closest existing time.
            * 'NaT' will return NaT where there are nonexistent times.
            * timedelta objects will shift nonexistent times by the timedelta.
            * 'raise' will raise a ValueError if there are
              nonexistent times.

        Returns
        -------
        a new Timestamp rounded to the given resolution of `freq`

        Raises
        ------
        ValueError if the freq cannot be converted

        See Also
        --------
        datetime.round : Similar behavior in native Python datetime module.
        Timestamp.floor : Round the Timestamp downward to the nearest multiple
            of the specified frequency.
        Timestamp.ceil : Round the Timestamp upward to the nearest multiple of
            the specified frequency.

        Notes
        -----
        If the Timestamp has a timezone, rounding will take place relative to the
        local ("wall") time and re-localized to the same timezone. When rounding
        near daylight savings time, use ``nonexistent`` and ``ambiguous`` to
        control the re-localization behavior.

        Examples
        --------
        Create a timestamp object:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')

        A timestamp can be rounded using multiple frequency units:

        >>> ts.round(freq='h')  # hour
        Timestamp('2020-03-14 16:00:00')

        >>> ts.round(freq='min')  # minute
        Timestamp('2020-03-14 15:33:00')

        >>> ts.round(freq='s')  # seconds
        Timestamp('2020-03-14 15:32:52')

        >>> ts.round(freq='ms')  # milliseconds
        Timestamp('2020-03-14 15:32:52.193000')

        ``freq`` can also be a multiple of a single unit, like '5min' (i.e.  5 minutes):

        >>> ts.round(freq='5min')
        Timestamp('2020-03-14 15:35:00')

        or a combination of multiple units, like '1h30min' (i.e. 1 hour and 30 minutes):

        >>> ts.round(freq='1h30min')
        Timestamp('2020-03-14 15:00:00')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.round()
        NaT

        When rounding near a daylight savings time transition, use ``ambiguous`` or
        ``nonexistent`` to control how the timestamp should be re-localized.

        >>> ts_tz = pd.Timestamp("2021-10-31 01:30:00").tz_localize("Europe/Amsterdam")

        >>> ts_tz.round("h", ambiguous=False)
        Timestamp('2021-10-31 02:00:00+0100', tz='Europe/Amsterdam')

        >>> ts_tz.round("h", ambiguous=True)
        Timestamp('2021-10-31 02:00:00+0200', tz='Europe/Amsterdam')
        """
        return self._round(
            freq, RoundTo.NEAREST_HALF_EVEN, ambiguous, nonexistent
        )

    def floor(self, freq, ambiguous="raise", nonexistent="raise"):
        # [FUNCTION DOC]
        # Round down to a multiple of freq (_round with MINUS_INFTY).
        """
        Return a new Timestamp floored to this resolution.

        Parameters
        ----------
        freq : str
            Frequency string indicating the flooring resolution.
        ambiguous : bool or {'raise', 'NaT'}, default 'raise'
            The behavior is as follows:

            * bool contains flags to determine if time is dst or not (note
              that this flag is only applicable for ambiguous fall dst dates).
            * 'NaT' will return NaT for an ambiguous time.
            * 'raise' will raise a ValueError for an ambiguous time.

        nonexistent : {'raise', 'shift_forward', 'shift_backward', 'NaT', \
timedelta}, default 'raise'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            * 'shift_forward' will shift the nonexistent time forward to the
              closest existing time.
            * 'shift_backward' will shift the nonexistent time backward to the
              closest existing time.
            * 'NaT' will return NaT where there are nonexistent times.
            * timedelta objects will shift nonexistent times by the timedelta.
            * 'raise' will raise a ValueError if there are
              nonexistent times.

        Raises
        ------
        ValueError if the freq cannot be converted.

        See Also
        --------
        Timestamp.ceil : Round up a Timestamp to the specified resolution.
        Timestamp.round : Round a Timestamp to the specified resolution.
        Series.dt.floor : Round down the datetime values in a Series.

        Notes
        -----
        If the Timestamp has a timezone, flooring will take place relative to the
        local ("wall") time and re-localized to the same timezone. When flooring
        near daylight savings time, use ``nonexistent`` and ``ambiguous`` to
        control the re-localization behavior.

        Examples
        --------
        Create a timestamp object:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')

        A timestamp can be floored using multiple frequency units:

        >>> ts.floor(freq='h')  # hour
        Timestamp('2020-03-14 15:00:00')

        >>> ts.floor(freq='min')  # minute
        Timestamp('2020-03-14 15:32:00')

        >>> ts.floor(freq='s')  # seconds
        Timestamp('2020-03-14 15:32:52')

        >>> ts.floor(freq='ns')  # nanoseconds
        Timestamp('2020-03-14 15:32:52.192548651')

        ``freq`` can also be a multiple of a single unit, like '5min' (i.e.  5 minutes):

        >>> ts.floor(freq='5min')
        Timestamp('2020-03-14 15:30:00')

        or a combination of multiple units, like '1h30min' (i.e. 1 hour and 30 minutes):

        >>> ts.floor(freq='1h30min')
        Timestamp('2020-03-14 15:00:00')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.floor()
        NaT

        When rounding near a daylight savings time transition, use ``ambiguous`` or
        ``nonexistent`` to control how the timestamp should be re-localized.

        >>> ts_tz = pd.Timestamp("2021-10-31 03:30:00").tz_localize("Europe/Amsterdam")

        >>> ts_tz.floor("2h", ambiguous=False)
        Timestamp('2021-10-31 02:00:00+0100', tz='Europe/Amsterdam')

        >>> ts_tz.floor("2h", ambiguous=True)
        Timestamp('2021-10-31 02:00:00+0200', tz='Europe/Amsterdam')
        """
        return self._round(freq, RoundTo.MINUS_INFTY, ambiguous, nonexistent)

    def ceil(self, freq, ambiguous="raise", nonexistent="raise"):
        # [FUNCTION DOC]
        # Round up to a multiple of freq (_round with PLUS_INFTY).
        """
        Return a new Timestamp ceiled to this resolution.

        Parameters
        ----------
        freq : str
            Frequency string indicating the ceiling resolution.
        ambiguous : bool or {'raise', 'NaT'}, default 'raise'
            The behavior is as follows:

            * bool contains flags to determine if time is dst or not (note
              that this flag is only applicable for ambiguous fall dst dates).
            * 'NaT' will return NaT for an ambiguous time.
            * 'raise' will raise a ValueError for an ambiguous time.

        nonexistent : {'raise', 'shift_forward', 'shift_backward', 'NaT', \
timedelta}, default 'raise'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            * 'shift_forward' will shift the nonexistent time forward to the
              closest existing time.
            * 'shift_backward' will shift the nonexistent time backward to the
              closest existing time.
            * 'NaT' will return NaT where there are nonexistent times.
            * timedelta objects will shift nonexistent times by the timedelta.
            * 'raise' will raise a ValueError if there are
              nonexistent times.

        Raises
        ------
        ValueError if the freq cannot be converted.

        See Also
        --------
        Timestamp.floor : Round down a Timestamp to the specified resolution.
        Timestamp.round : Round a Timestamp to the specified resolution.
        Series.dt.ceil : Ceil the datetime values in a Series.

        Notes
        -----
        If the Timestamp has a timezone, ceiling will take place relative to the
        local ("wall") time and re-localized to the same timezone. When ceiling
        near daylight savings time, use ``nonexistent`` and ``ambiguous`` to
        control the re-localization behavior.

        Examples
        --------
        Create a timestamp object:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')

        A timestamp can be ceiled using multiple frequency units:

        >>> ts.ceil(freq='h')  # hour
        Timestamp('2020-03-14 16:00:00')

        >>> ts.ceil(freq='min')  # minute
        Timestamp('2020-03-14 15:33:00')

        >>> ts.ceil(freq='s')  # seconds
        Timestamp('2020-03-14 15:32:53')

        >>> ts.ceil(freq='us')  # microseconds
        Timestamp('2020-03-14 15:32:52.192549')

        ``freq`` can also be a multiple of a single unit, like '5min' (i.e.  5 minutes):

        >>> ts.ceil(freq='5min')
        Timestamp('2020-03-14 15:35:00')

        or a combination of multiple units, like '1h30min' (i.e. 1 hour and 30 minutes):

        >>> ts.ceil(freq='1h30min')
        Timestamp('2020-03-14 16:30:00')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.ceil()
        NaT

        When rounding near a daylight savings time transition, use ``ambiguous`` or
        ``nonexistent`` to control how the timestamp should be re-localized.

        >>> ts_tz = pd.Timestamp("2021-10-31 01:30:00").tz_localize("Europe/Amsterdam")

        >>> ts_tz.ceil("h", ambiguous=False)
        Timestamp('2021-10-31 02:00:00+0100', tz='Europe/Amsterdam')

        >>> ts_tz.ceil("h", ambiguous=True)
        Timestamp('2021-10-31 02:00:00+0200', tz='Europe/Amsterdam')
        """
        return self._round(freq, RoundTo.PLUS_INFTY, ambiguous, nonexistent)

    @property
    def tz(self):
        # [FUNCTION DOC]
        # Property: alias for tzinfo.
        """
        Alias for tzinfo.

        The `tz` property provides a simple and direct way to retrieve the timezone
        information of a `Timestamp` object. It is particularly useful when working
        with time series data that includes timezone information, allowing for easy
        access and manipulation of the timezone context.

        See Also
        --------
        Timestamp.tzinfo : Returns the timezone information of the Timestamp.
        Timestamp.tz_convert : Convert timezone-aware Timestamp to another time zone.
        Timestamp.tz_localize : Localize the Timestamp to a timezone.

        Examples
        --------
        >>> ts = pd.Timestamp(1584226800, unit='s', tz='Europe/Stockholm')
        >>> ts.tz
        zoneinfo.ZoneInfo(key='Europe/Stockholm')
        """
        return self.tzinfo

    @tz.setter
    def tz(self, value):
        # [FUNCTION DOC]
        # Setter for tz: always raises AttributeError (use tz_localize/tz_convert).
        # GH 3746: Prevent localizing or converting the index by setting tz
        raise AttributeError(
            "Cannot directly set timezone. "
            "Use tz_localize() or tz_convert() as appropriate"
        )

    def tz_localize(self, tz, ambiguous="raise", nonexistent="raise"):
        # [FUNCTION DOC]
        # Localize a naive Timestamp to tz, or remove tz when tz is None. Validates
        # ambiguous (bool, "NaT" or "raise") and nonexistent ("raise", "NaT",
        # "shift_forward", "shift_backward" or timedelta), else ValueError. Naive:
        # resolve tz, wrap non-string ambiguous in a 1-element list, compute the UTC
        # value with tz_localize_to_utc_single and rebuild with tz. Aware with tz None:
        # convert UTC value to local wall time and rebuild naive. Aware with tz given:
        # TypeError.
        # Inputs: tz - str/tzinfo/None; ambiguous; nonexistent.
        # Outputs: Timestamp or NaT (when "NaT" handling applies).
        # Exceptions: ValueError, TypeError, ambiguous/nonexistent time errors.
        """
        Localize the Timestamp to a timezone.

        Convert naive Timestamp to local time zone or remove
        timezone from timezone-aware Timestamp.

        Parameters
        ----------
        tz : str, zoneinfo.ZoneInfo, pytz.timezone, dateutil.tz.tzfile or None
            Time zone for time which Timestamp will be converted to.
            None will remove timezone holding local time.

        ambiguous : bool, 'NaT', default 'raise'
            When clocks moved backward due to DST, ambiguous times may arise.
            For example in Central European Time (UTC+01), when going from
            03:00 DST to 02:00 non-DST, 02:30:00 local time occurs both at
            00:30:00 UTC and at 01:30:00 UTC. In such a situation, the
            `ambiguous` parameter dictates how ambiguous times should be
            handled.

            The behavior is as follows:

            * bool contains flags to determine if time is dst or not (note
              that this flag is only applicable for ambiguous fall dst dates).
            * 'NaT' will return NaT for an ambiguous time.
            * 'raise' will raise a ValueError for an ambiguous time.

        nonexistent : 'shift_forward', 'shift_backward', 'NaT', timedelta, \
default 'raise'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            The behavior is as follows:

            * 'shift_forward' will shift the nonexistent time forward to the
              closest existing time.
            * 'shift_backward' will shift the nonexistent time backward to the
              closest existing time.
            * 'NaT' will return NaT where there are nonexistent times.
            * timedelta objects will shift nonexistent times by the timedelta.
            * 'raise' will raise a ValueError if there are
              nonexistent times.

        Returns
        -------
        localized : Timestamp

        Raises
        ------
        TypeError
            If the Timestamp is tz-aware and tz is not None.

        See Also
        --------
        Timestamp.tzinfo : Returns the timezone information of the Timestamp.
        Timestamp.tz_convert : Convert timezone-aware Timestamp to another time zone.
        DatetimeIndex.tz_localize : Localize a DatetimeIndex to a specific time zone.
        datetime.datetime.astimezone : Convert a datetime object to another time zone.

        Examples
        --------
        Create a naive timestamp object:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts
        Timestamp('2020-03-14 15:32:52.192548651')

        Add 'Europe/Stockholm' as timezone:

        >>> ts.tz_localize(tz='Europe/Stockholm')
        Timestamp('2020-03-14 15:32:52.192548651+0100', tz='Europe/Stockholm')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.tz_localize()
        NaT
        """
        if not isinstance(ambiguous, bool) and ambiguous not in {"NaT", "raise"}:
            raise ValueError(
                        "'ambiguous' parameter must be one of: "
                        "True, False, 'NaT', 'raise' (default)"
                    )

        nonexistent_options = ("raise", "NaT", "shift_forward", "shift_backward")
        if nonexistent not in nonexistent_options and not PyDelta_Check(nonexistent):
            raise ValueError(
                "The nonexistent argument must be one of 'raise', "
                "'NaT', 'shift_forward', 'shift_backward' or a timedelta object"
            )

        if self.tzinfo is None:
            # tz naive, localize
            tz = maybe_get_tz(tz)
            if not isinstance(ambiguous, str):
                ambiguous = [ambiguous]
            value = tz_localize_to_utc_single(self._value, tz,
                                              ambiguous=ambiguous,
                                              nonexistent=nonexistent,
                                              creso=self._creso)
        elif tz is None:
            # reset tz
            value = tz_convert_from_utc_single(self._value, self.tz, creso=self._creso)

        else:
            raise TypeError(
                "Cannot localize tz-aware Timestamp, use tz_convert for conversions"
            )

        out = type(self)._from_value_and_reso(value, self._creso, tz=tz)
        return out

    def tz_convert(self, tz):
        # [FUNCTION DOC]
        # Convert an aware Timestamp to another tz: same UTC _value and resolution,
        # rebuilt with the resolved tz (tz=None yields naive UTC wall time).
        # Exceptions: TypeError if self is naive. Also exposed as astimezone.
        """
        Convert timezone-aware Timestamp to another time zone.

        This method is used to convert a timezone-aware Timestamp object to a
        different time zone. The original UTC time remains the same; only the
        time zone information is changed. If the Timestamp is timezone-naive, a
        TypeError is raised.

        Parameters
        ----------
        tz : str, zoneinfo.ZoneInfo, pytz.timezone, dateutil.tz.tzfile or None
            Time zone for time which Timestamp will be converted to.
            None will remove timezone holding UTC time.

        Returns
        -------
        converted : Timestamp

        Raises
        ------
        TypeError
            If Timestamp is tz-naive.

        See Also
        --------
        Timestamp.tz_localize : Localize the Timestamp to a timezone.
        DatetimeIndex.tz_convert : Convert a DatetimeIndex to another time zone.
        DatetimeIndex.tz_localize : Localize a DatetimeIndex to a specific time zone.
        datetime.datetime.astimezone : Convert a datetime object to another time zone.

        Examples
        --------
        Create a timestamp object with UTC timezone:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651', tz='UTC')
        >>> ts
        Timestamp('2020-03-14 15:32:52.192548651+0000', tz='UTC')

        Change to Tokyo timezone:

        >>> ts.tz_convert(tz='Asia/Tokyo')
        Timestamp('2020-03-15 00:32:52.192548651+0900', tz='Asia/Tokyo')

        Can also use ``astimezone``:

        >>> ts.astimezone(tz='Asia/Tokyo')
        Timestamp('2020-03-15 00:32:52.192548651+0900', tz='Asia/Tokyo')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.tz_convert(tz='Asia/Tokyo')
        NaT
        """
        if self.tzinfo is None:
            # tz naive, use tz_localize
            raise TypeError(
                "Cannot convert tz-naive Timestamp, use tz_localize to localize"
            )
        else:
            # Same UTC timestamp, different time zone
            tz = maybe_get_tz(tz)
            out = type(self)._from_value_and_reso(self._value, reso=self._creso, tz=tz)
            return out

    astimezone = tz_convert

    def replace(
        self,
        year=None,
        month=None,
        day=None,
        hour=None,
        minute=None,
        second=None,
        microsecond=None,
        nanosecond=None,
        tzinfo=object,
        fold=None,
    ):
        # [FUNCTION DOC]
        # Return a copy with fields replaced. Converts the value to local wall time,
        # breaks it into a struct (with ps from _nanosecond), overwrites each provided
        # field after validation (validate requires integers), and keeps self's fold
        # unless given. Replacing microsecond raises the resolution to at least
        # microseconds and replacing nanosecond forces nanoseconds. tzinfo=object
        # (default sentinel) keeps the current tz, any other value (incl. None) replaces
        # it without converting wall time.
        # Reconstruction: naive -> struct converted directly to an epoch count (works
        # beyond the stdlib datetime range). pytz -> tz.localize(datetime(...),
        # is_dst=not fold), taking the possibly new tzinfo. Other tz -> stdlib datetime
        # with tzinfo and fold. Aware paths convert via convert_datetime_to_tsobject.
        # Inputs: year..nanosecond - int or None; tzinfo - tzinfo/None/object; fold.
        # Outputs: Timestamp. Exceptions: ValueError for non-integer fields or invalid
        # datetime fields; OutOfBoundsDatetime on overflow.
        """
        Implements datetime.replace, handles nanoseconds.

        This method creates a new `Timestamp` object by replacing the specified
        fields with new values. The new `Timestamp` retains the original fields
        that are not explicitly replaced. This method handles nanoseconds, and
        the `tzinfo` parameter allows for timezone replacement without conversion.

        Parameters
        ----------
        year : int, optional
            The year to replace. If `None`, the year is not changed.
        month : int, optional
            The month to replace. If `None`, the month is not changed.
        day : int, optional
            The day to replace. If `None`, the day is not changed.
        hour : int, optional
            The hour to replace. If `None`, the hour is not changed.
        minute : int, optional
            The minute to replace. If `None`, the minute is not changed.
        second : int, optional
            The second to replace. If `None`, the second is not changed.
        microsecond : int, optional
            The microsecond to replace. If `None`, the microsecond is not changed.
        nanosecond : int, optional
            The nanosecond to replace. If `None`, the nanosecond is not changed.
        tzinfo : tz-convertible, optional
            The timezone information to replace. If `None`, the timezone is not changed.
        fold : int, optional
            The fold information to replace. If `None`, the fold is not changed.

        Returns
        -------
        Timestamp
            A new `Timestamp` object with the specified fields replaced.

        See Also
        --------
        Timestamp : Represents a single timestamp, similar to `datetime`.
        to_datetime : Converts various types of data to datetime.

        Notes
        -----
        The `replace` method does not perform timezone conversions. If you need
        to convert the timezone, use the `tz_convert` method instead.

        Examples
        --------
        Create a timestamp object:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651', tz='UTC')
        >>> ts
        Timestamp('2020-03-14 15:32:52.192548651+0000', tz='UTC')

        Replace year and the hour:

        >>> ts.replace(year=1999, hour=10)
        Timestamp('1999-03-14 10:32:52.192548651+0000', tz='UTC')

        Replace timezone (not a conversion):

        >>> import zoneinfo
        >>> ts.replace(tzinfo=zoneinfo.ZoneInfo('US/Pacific'))
        Timestamp('2020-03-14 15:32:52.192548651-0700', tz='US/Pacific')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.replace(tzinfo=zoneinfo.ZoneInfo('US/Pacific'))
        NaT
        """

        cdef:
            npy_datetimestruct dts
            int64_t value
            object k, v
            datetime ts_input
            tzinfo_type tzobj
            _TSObject ts
            NPY_DATETIMEUNIT creso = self._creso

        # set to naive if needed
        tzobj = self.tzinfo
        value = self._value

        # GH 37610. Preserve fold when replacing.
        if fold is None:
            fold = self.fold

        if tzobj is not None:
            value = tz_convert_from_utc_single(value, tzobj, creso=self._creso)

        # setup components
        pandas_datetime_to_datetimestruct(value, self._creso, &dts)
        dts.ps = self._nanosecond * 1000

        # replace
        def validate(k, v):
            # [FUNCTION DOC]
            # HELPER (nested): return v if it is an integer, else raise ValueError
            # naming field k and v's type. Inputs: k - field name str; v - value.
            """ validate integers """
            if not is_integer_object(v):
                raise ValueError(
                    f"value must be an integer, received {type(v)} for {k}"
                )
            return v

        if year is not None:
            dts.year = validate("year", year)
        if month is not None:
            dts.month = validate("month", month)
        if day is not None:
            dts.day = validate("day", day)
        if hour is not None:
            dts.hour = validate("hour", hour)
        if minute is not None:
            dts.min = validate("minute", minute)
        if second is not None:
            dts.sec = validate("second", second)
        if microsecond is not None:
            dts.us = validate("microsecond", microsecond)
            if creso < NPY_DATETIMEUNIT.NPY_FR_us:
                # GH#57749
                creso = NPY_DATETIMEUNIT.NPY_FR_us
        if nanosecond is not None:
            dts.ps = validate("nanosecond", nanosecond) * 1000
            creso = NPY_FR_ns  # GH#57749
        if tzinfo is not object:
            tzobj = tzinfo

        # reconstruct & check bounds
        if tzobj is None:
            # We can avoid going through pydatetime paths, which is robust
            #  to datetimes outside of pydatetime range.
            ts = _TSObject()
            try:
                ts.value = npy_datetimestruct_to_datetime(creso, &dts)
            except OverflowError as err:
                fmt = dts_to_iso_string(&dts)
                raise OutOfBoundsDatetime(
                    f"Out of bounds timestamp: {fmt} with frequency '{self.unit}'"
                ) from err
            ts.dts = dts
            ts.creso = creso
            ts.fold = fold
            return create_timestamp_from_ts(
                ts.value, dts, tzobj, fold, reso=creso
            )

        elif tzobj is not None and treat_tz_as_pytz(tzobj):
            # replacing across a DST boundary may induce a new tzinfo object
            # see GH#18319
            ts_input = tzobj.localize(datetime(dts.year, dts.month, dts.day,
                                               dts.hour, dts.min, dts.sec,
                                               dts.us),
                                      is_dst=not bool(fold))
            tzobj = ts_input.tzinfo
        else:
            kwargs = {"year": dts.year, "month": dts.month, "day": dts.day,
                      "hour": dts.hour, "minute": dts.min, "second": dts.sec,
                      "microsecond": dts.us, "tzinfo": tzobj,
                      "fold": fold}
            ts_input = datetime(**kwargs)

        ts = convert_datetime_to_tsobject(
            ts_input, tzobj, nanos=dts.ps // 1000, reso=creso
        )
        return create_timestamp_from_ts(
            ts.value, dts, tzobj, fold, reso=creso
        )

    def to_julian_date(self) -> np.float64:
        # [FUNCTION DOC]
        # Julian Date as float: shifts Jan/Feb to months 13/14 of the previous year,
        # applies the standard Gregorian day-count formula plus 1721118.5, and adds the
        # time of day (hours, minutes, seconds, microseconds, nanoseconds) as a fraction
        # of a day. Uses the true year and wall-clock fields (tz offset is not applied).
        # Outputs: numpy float64.
        """
        Convert TimeStamp to a Julian Date.

        This method returns the number of days as a float since
        0 Julian date, which is noon January 1, 4713 BC.

        See Also
        --------
        Timestamp.toordinal : Return proleptic Gregorian ordinal.
        Timestamp.timestamp : Return POSIX timestamp as float.
        Timestamp : Represents a single timestamp.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52')
        >>> ts.to_julian_date()
        2458923.147824074
        """
        year = self._year
        month = self.month
        day = self.day
        if month <= 2:
            year -= 1
            month += 12
        return (day +
                np.trunc((153 * month - 457) / 5) +
                365 * year +
                np.floor(year / 4) -
                np.floor(year / 100) +
                np.floor(year / 400) +
                1721118.5 +
                (self.hour +
                 self.minute / 60.0 +
                 self.second / 3600.0 +
                 self.microsecond / 3600.0 / 1e+6 +
                 self._nanosecond / 3600.0 / 1e+9
                 ) / 24.0)

    def isoweekday(self):
        # [FUNCTION DOC]
        # ISO weekday 1=Monday..7=Sunday (weekday() + 1); avoids datetime's fake year.
        """
        Return the day of the week represented by the date.

        Monday == 1 ... Sunday == 7.

        See Also
        --------
        Timestamp.weekday : Return the day of the week with Monday=0, Sunday=6.
        Timestamp.isocalendar : Return a tuple containing ISO year, week number
            and weekday.
        datetime.date.isoweekday : Equivalent method in datetime module.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00')
        >>> ts
        Timestamp('2023-01-01 10:00:00')
        >>> ts.isoweekday()
        7
        """
        # same as super().isoweekday(), but that breaks because of how
        #  we have overridden year, see note in create_timestamp_from_ts

        return self.weekday() + 1

    def weekday(self):
        # [FUNCTION DOC]
        # Weekday 0=Monday..6=Sunday from _year/month/day via ccalendar.dayofweek
        # (stdlib weekday would use the fake year for out-of-range dates).
        """
        Return the day of the week represented by the date.

        Monday == 0 ... Sunday == 6.

        See Also
        --------
        Timestamp.dayofweek : Return the day of the week with Monday=0, Sunday=6.
        Timestamp.isoweekday : Return the day of the week with Monday=1, Sunday=7.
        datetime.date.weekday : Equivalent method in datetime module.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01')
        >>> ts
        Timestamp('2023-01-01  00:00:00')
        >>> ts.weekday()
        6
        """
        # same as super().weekday(), but that breaks because of how
        #  we have overridden year, see note in create_timestamp_from_ts
        return ccalendar.dayofweek(self._year, self.month, self.day)


# Aliases
Timestamp.weekofyear = Timestamp.week
Timestamp.daysinmonth = Timestamp.days_in_month


# ----------------------------------------------------------------------
# Scalar analogues to functions in vectorized.pyx


@cython.cdivision(False)
cdef int64_t normalize_i8_stamp(int64_t local_val, int64_t ppd):
    # [FUNCTION DOC]
    # HELPER: Floor a local epoch count to the start of its day: local_val - (local_val
    # mod ppd), with Python (floor) modulo semantics so negative values floor toward
    # earlier days. Overflow-checked.
    # Inputs: local_val - int64 wall-time count; ppd - units per day (> 0).
    # Outputs: int64. Exceptions: OverflowError if the subtraction overflows.
    """
    Round the localized nanosecond timestamp down to the previous midnight.

    Parameters
    ----------
    local_val : int64_t
    ppd : int64_t
        Periods per day in the Timestamp's resolution.

    Returns
    -------
    int64_t
    """
    with cython.overflowcheck(True):
        # GH#60583
        return local_val - (local_val % ppd)
