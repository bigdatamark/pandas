# [MODULE DOC]
# Implements pandas' duration scalar Timedelta and the parsing/conversion machinery
# behind it and behind to_timedelta / TimedeltaIndex construction. Representation: a
# Timedelta stores a signed 64-bit integer _value counting units of a resolution _creso
# (one of seconds, milliseconds, microseconds, nanoseconds; numpy NPY_DATETIMEUNIT
# codes, where a larger code means a finer unit). Missing values are represented by the
# NaT singleton and, in integer arrays, by NPY_NAT (minimum int64).
# Day/hour/minute/second/ms/us/ns components are computed lazily from _value and cached.
# Contents: module constants (Components namedtuple, timedelta_abbrevs mapping of unit
# spellings to canonical abbreviations); array conversion functions
# (ints_to_pytimedelta, array_to_timedelta64, truediv/floordiv_object_array); string
# parsers for the "1 days 02:03:04.5" style and ISO 8601 "P1DT2H" style; unit
# normalization (parse_timedelta_unit, emitting deprecation warnings for deprecated
# spellings); operator factories used to build Timedelta arithmetic; the C base class
# _Timedelta (comparison, hashing, components, conversions, unit casting) and the Python
# class Timedelta (constructor, pickling, rounding, arithmetic). Resolution handling:
# inputs keep their own resolution where supported; binary operations cast both operands
# to the finer resolution and raise OutOfBoundsTimedelta instead of silently
# overflowing.
import collections
import re
import warnings

from pandas.util._decorators import set_module
from pandas.util._exceptions import find_stack_level

cimport cython
from cpython.object cimport (
    Py_EQ,
    Py_GE,
    Py_GT,
    Py_LE,
    Py_LT,
    Py_NE,
    PyObject,
    PyObject_RichCompare,
)

from pandas._libs.tslibs.offsets cimport to_offset

import numpy as np

cimport numpy as cnp
from numpy cimport (
    int64_t,
    ndarray,
)

cnp.import_array()

from cpython.datetime cimport (
    PyDateTime_Check,
    PyDelta_Check,
    import_datetime,
    timedelta,
)

import_datetime()


cimport pandas._libs.tslibs.util as util
from pandas._libs.missing cimport checknull_with_nat_and_na
from pandas._libs.tslibs.base cimport ABCTimestamp
from pandas._libs.tslibs.conversion cimport (
    cast_from_unit,
)
from pandas._libs.tslibs.dtypes cimport (
    abbrev_to_npy_unit,
    c_DEPR_UNITS,
    get_supported_reso,
    is_supported_unit,
    npy_unit_to_abbrev,
)
from pandas._libs.tslibs.nattype cimport (
    NPY_NAT,
    c_NaT as NaT,
    c_nat_strings as nat_strings,
    checknull_with_nat,
)
from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    NPY_FR_ns,
    cmp_dtstructs,
    cmp_scalar,
    convert_reso,
    get_datetime64_unit,
    get_unit_from_dtype,
    import_pandas_datetime,
    npy_datetimestruct,
    pandas_datetime_to_datetimestruct,
    pandas_timedelta_to_timedeltastruct,
    pandas_timedeltastruct,
)

import_pandas_datetime()

from pandas._libs.tslibs.np_datetime import (
    OutOfBoundsDatetime,
    OutOfBoundsTimedelta,
)

from pandas._libs.tslibs.offsets cimport is_tick_object

from pandas._libs.tslibs.offsets import Day

from pandas._libs.tslibs.util cimport (
    is_array,
    is_bool_object,
    is_float_object,
    is_integer_object,
)

from pandas._libs.tslibs.fields import (
    RoundTo,
    round_nsint64,
)

# ----------------------------------------------------------------------
# Constants

# components named tuple
Components = collections.namedtuple(
    "Components",
    [
        "days",
        "hours",
        "minutes",
        "seconds",
        "milliseconds",
        "microseconds",
        "nanoseconds",
    ],
)

# This should be kept consistent with UnitChoices in pandas/_libs/tslibs/timedeltas.pyi
cdef dict timedelta_abbrevs = {
    "Y": "Y",
    "y": "Y",
    "M": "M",
    "W": "W",
    "w": "W",
    "D": "D",
    "d": "D",
    "days": "D",
    "day": "D",
    "hours": "h",
    "hour": "h",
    "hr": "h",
    "h": "h",
    "m": "m",
    "minute": "m",
    "min": "m",
    "minutes": "m",
    "s": "s",
    "seconds": "s",
    "sec": "s",
    "second": "s",
    "ms": "ms",
    "milliseconds": "ms",
    "millisecond": "ms",
    "milli": "ms",
    "millis": "ms",
    "us": "us",
    "microseconds": "us",
    "microsecond": "us",
    "µs": "us",
    "micro": "us",
    "micros": "us",
    "ns": "ns",
    "nanoseconds": "ns",
    "nano": "ns",
    "nanos": "ns",
    "nanosecond": "ns",
}

_no_input = object()

# ----------------------------------------------------------------------
# API


@cython.boundscheck(False)
@cython.wraparound(False)
def ints_to_pytimedelta(ndarray m8values, box=False):
    # [FUNCTION DOC]
    # Converts a timedelta64 array of any shape into an object array of the same shape
    # holding Python timedeltas (or pandas Timedeltas when box is true).
    # Inputs: m8values - ndarray of timedelta64 dtype with any supported unit, any
    # dimensionality; box - truthy to produce Timedelta objects.
    # Outputs: new C-ordered object ndarray with same shape; NPY_NAT entries become NaT.
    # With box, each element is a Timedelta keeping the array's resolution. Without box,
    # elements are datetime.timedelta built from the integer in the array's unit (W, D,
    # h, m, s, ms, us; for ns the value is divided by 1000 as a float of microseconds,
    # so datetime.timedelta rounds to the nearest microsecond).
    # Steps: iterate the int64 view in C order with a flat iterator, writing into the
    # flattened result.
    # Exceptions: NotImplementedError for units Y, M, generic or finer than ns (and for
    # non-s/ms/us/ns units when box is true, via the Timedelta factory); OverflowError
    # if a value exceeds datetime.timedelta range.
    # Side effects: allocates result. Bounds/wraparound checks disabled.
    # KNOWN ISSUE: docstring names the parameter "arr" but it is m8values.
    """
    convert an i8 repr to an ndarray of timedelta or Timedelta (if box ==
    True)

    Parameters
    ----------
    arr : ndarray[timedelta64]
    box : bool, default False

    Returns
    -------
    result : ndarray[object]
        array of Timedelta or timedeltas objects
    """
    cdef:
        NPY_DATETIMEUNIT reso = get_unit_from_dtype(m8values.dtype)
        Py_ssize_t i, n = m8values.size
        int64_t value
        object res_val

        # Note that `result` (and thus `result_flat`) is C-order and
        #  `it` iterates C-order as well, so the iteration matches
        #  See discussion at
        #  github.com/pandas-dev/pandas/pull/46886#discussion_r860261305
        ndarray result = cnp.PyArray_EMPTY(
            m8values.ndim, m8values.shape, cnp.NPY_OBJECT, 0
        )
        object[::1] res_flat = result.ravel()     # should NOT be a copy

        ndarray arr = m8values.view("i8")
        cnp.flatiter it = cnp.PyArray_IterNew(arr)

    for i in range(n):
        # Analogous to: value = arr[i]
        value = (<int64_t*>cnp.PyArray_ITER_DATA(it))[0]

        if value == NPY_NAT:
            res_val = <object>NaT
        else:
            if box:
                res_val = _timedelta_from_value_and_reso(Timedelta, value, reso=reso)
            elif reso == NPY_DATETIMEUNIT.NPY_FR_ns:
                res_val = timedelta(microseconds=int(value) / 1000)
            elif reso == NPY_DATETIMEUNIT.NPY_FR_us:
                res_val = timedelta(microseconds=value)
            elif reso == NPY_DATETIMEUNIT.NPY_FR_ms:
                res_val = timedelta(milliseconds=value)
            elif reso == NPY_DATETIMEUNIT.NPY_FR_s:
                res_val = timedelta(seconds=value)
            elif reso == NPY_DATETIMEUNIT.NPY_FR_m:
                res_val = timedelta(minutes=value)
            elif reso == NPY_DATETIMEUNIT.NPY_FR_h:
                res_val = timedelta(hours=value)
            elif reso == NPY_DATETIMEUNIT.NPY_FR_D:
                res_val = timedelta(days=value)
            elif reso == NPY_DATETIMEUNIT.NPY_FR_W:
                res_val = timedelta(weeks=value)
            else:
                # Month, Year, NPY_FR_GENERIC, pico, femto, atto
                raise NotImplementedError(reso)

        # Note: we can index result directly instead of using PyArray_MultiIter_DATA
        #  like we do for the other functions because result is known C-contiguous
        #  and is the first argument to PyArray_MultiIterNew2.  The usual pattern
        #  does not seem to work with object dtype.
        #  See discussion at
        #  github.com/pandas-dev/pandas/pull/46886#discussion_r860261305
        res_flat[i] = res_val

        cnp.PyArray_ITER_NEXT(it)

    return result


# ----------------------------------------------------------------------


cpdef int64_t delta_to_nanoseconds(
    delta,
    NPY_DATETIMEUNIT reso=NPY_FR_ns,
    bint round_ok=True,
) except? -1:
    # [FUNCTION DOC]
    # Converts a timedelta-like scalar into an integer count of reso units.
    # Inputs: delta - a Tick offset (uses .n and ._creso), a _Timedelta (uses _value and
    # _creso), a numpy timedelta64 (its own unit), or datetime.timedelta (computed as
    # microseconds from days/seconds/microseconds); reso - target unit (default ns);
    # round_ok - if false, a conversion to a coarser unit that loses information raises
    # instead of truncating/rounding.
    # Outputs: int64 count of reso units (via convert_reso).
    # Exceptions: ValueError for timedelta64 with Y or M unit; TypeError for any other
    # type; OutOfBoundsTimedelta when the result overflows int64; ValueError from
    # convert_reso when round_ok is false and rounding is required.
    # Side effects: none. Note: a NaT timedelta64 is converted as an ordinary integer.
    # Note: this will raise on timedelta64 with Y or M unit

    cdef:
        NPY_DATETIMEUNIT in_reso
        int64_t n

    if is_tick_object(delta):
        n = delta.n
        in_reso = delta._creso

    elif isinstance(delta, _Timedelta):
        n = delta._value
        in_reso = delta._creso

    elif cnp.is_timedelta64_object(delta):
        in_reso = get_datetime64_unit(delta)
        if in_reso == NPY_DATETIMEUNIT.NPY_FR_Y or in_reso == NPY_DATETIMEUNIT.NPY_FR_M:
            raise ValueError(
                "delta_to_nanoseconds does not support Y or M units, "
                "as their duration in nanoseconds is ambiguous."
            )
        n = cnp.get_timedelta64_value(delta)

    elif PyDelta_Check(delta):
        in_reso = NPY_DATETIMEUNIT.NPY_FR_us
        try:
            n = (
                delta.days * 24 * 3600 * 1_000_000
                + delta.seconds * 1_000_000
                + delta.microseconds
                )
        except OverflowError as err:
            raise OutOfBoundsTimedelta(*err.args) from err

    else:
        raise TypeError(type(delta))

    try:
        return convert_reso(n, in_reso, reso, round_ok=round_ok)
    except (OutOfBoundsDatetime, OverflowError) as err:
        # Catch OutOfBoundsDatetime bc convert_reso can call check_dts_bounds
        #  for Y/M-resolution cases
        unit_str = npy_unit_to_abbrev(reso)
        raise OutOfBoundsTimedelta(
            f"Cannot cast {str(delta)} to unit={unit_str} without overflow."
        ) from err


cdef int64_t _numeric_to_td64ns(
    object item, str unit, NPY_DATETIMEUNIT out_reso=NPY_FR_ns
):
    # [FUNCTION DOC]
    # HELPER: Converts an int or float in a given unit into an integer count of out_reso
    # units (default ns).
    # Inputs: item - Python/numpy int or float (caller guarantees numeric); unit -
    # canonical unit abbreviation, not Y/y/M (caller guarantees); out_reso - target
    # resolution.
    # Outputs: int64; NPY_NAT if item is an integer equal to NPY_NAT; otherwise the
    # result of cast_from_unit (float fractions are rounded to the target unit).
    # Exceptions: OutOfBoundsTimedelta if the result overflows int64.
    # Side effects: none.
    # caller is responsible for checking
    #  assert unit not in ["Y", "y", "M"]
    #  assert is_integer_object(item) or is_float_object(item)
    if is_integer_object(item) and item == NPY_NAT:
        return NPY_NAT

    try:
        ival = cast_from_unit(item, unit, out_reso)
    except OutOfBoundsDatetime as err:
        abbrev = npy_unit_to_abbrev(out_reso)
        raise OutOfBoundsTimedelta(
            f"Cannot cast {item} from {unit} to '{abbrev}' without overflow."
        ) from err

    return ival


# TODO: de-duplicate with DatetimeParseState
cdef class ResoState:
    # [CLASS DOC]
    # HELPER: Mutable tracker of the resolution to use while converting an array of
    # mixed timedelta-like objects. creso holds the finest resolution seen so far
    # (GENERIC means none seen yet); creso_ever_changed records whether a finer
    # resolution was encountered after a first one was fixed, meaning earlier elements
    # were converted at too coarse a resolution and the array must be re-parsed.
    cdef:
        NPY_DATETIMEUNIT creso
        bint creso_ever_changed

    def __cinit__(self, NPY_DATETIMEUNIT creso):
        # [FUNCTION DOC]
        # HELPER: Initializes the state with the starting resolution creso (GENERIC to
        # infer) and creso_ever_changed = False. Side effects: sets fields.
        self.creso = creso
        self.creso_ever_changed = False

    cdef bint update_creso(self, NPY_DATETIMEUNIT item_reso) noexcept:
        # [FUNCTION DOC]
        # HELPER: Records the resolution of one element.
        # Inputs: item_reso - resolution of the current element.
        # Outputs: True if the tracked resolution was bumped to a finer one (and sets
        # creso_ever_changed); False otherwise. If no resolution has been set (GENERIC),
        # adopts item_reso and returns False without marking a change.
        # Exceptions: none (noexcept). Side effects: mutates self.
        # Return a bool indicating whether we bumped to a higher resolution
        if self.creso == NPY_DATETIMEUNIT.NPY_FR_GENERIC:
            self.creso = item_reso
        elif item_reso > self.creso:
            self.creso = item_reso
            self.creso_ever_changed = True
            return True
        return False


@cython.boundscheck(False)
@cython.wraparound(False)
def array_to_timedelta64(
    ndarray values,
    str unit=None,
    str errors="raise",
    NPY_DATETIMEUNIT creso=NPY_DATETIMEUNIT.NPY_FR_GENERIC,
) -> ndarray:
    # [FUNCTION DOC]
    # Converts an object-dtype array of timedelta-like values into a timedelta64 array.
    # Inputs: values - object ndarray, any shape; unit - unit for numeric entries (None
    # means ns; must not be Y/y/M, asserted); errors - "raise", "coerce" or "ignore";
    # creso - output resolution, or GENERIC to infer it from the data.
    # Outputs: new timedelta64 ndarray with values' shape. Per-element handling:
    # null/NaT/NA -> NPY_NAT; timedelta64 -> its unit's supported resolution; _Timedelta
    # -> its resolution; datetime.timedelta -> us; strings (ISO if starting with "P" or
    # "-P", else the regular parser) -> us if no sub-microsecond content
    # (needs_nano_unit) else ns; Tick -> its resolution; integers -> unit's supported
    # resolution; integral floats -> like integers; other floats -> ns. When inferring,
    # the finest resolution seen is used; if it increased after the first element, the
    # function recurses once with that fixed resolution. If nothing non-NaT is seen, the
    # result unit is seconds.
    # Exceptions: TypeError if values is not object dtype or an element has an
    # unsupported type; ValueError for invalid errors, for strings combined with a non-
    # None unit (unless errors is "coerce"), unsupported timedelta64 units, or
    # unparseable strings (message rewritten for "unit abbreviation w/o a number");
    # OutOfBoundsTimedelta on overflow. With errors="coerce", ValueError (including
    # OutOfBoundsTimedelta) for an element sets it to NaT instead.
    # Side effects: allocates result; may emit deprecation warnings via unit parsing.
    # Bounds/wraparound checks disabled.
    # KNOWN ISSUE: docstring says the result is timedelta64[ns], but the resolution is
    # creso or inferred (s/ms/us/ns). errors="ignore" is accepted but behaves like
    # "raise". TypeError for unsupported element types is not coerced to NaT.
    # values is object-dtype, may be 2D
    """
    Convert an ndarray to an array of timedeltas. If errors == 'coerce',
    coerce non-convertible objects to NaT. Otherwise, raise.

    Returns
    -------
    np.ndarray[timedelta64ns]
    """
    # Caller is responsible for checking
    assert unit not in ["Y", "y", "M"]

    cdef:
        Py_ssize_t i, n = values.size
        ndarray result = np.empty((<object>values).shape, dtype="m8[ns]")
        object item
        int64_t ival
        cnp.broadcast mi = cnp.PyArray_MultiIterNew2(result, values)
        cnp.flatiter it
        str parsed_unit = parse_timedelta_unit(unit or "ns")
        NPY_DATETIMEUNIT item_reso, int_reso
        ResoState state = ResoState(creso)
        bint infer_reso = creso == NPY_DATETIMEUNIT.NPY_FR_GENERIC
        ndarray iresult = result.view("i8")

    if unit is None:
        int_reso = NPY_FR_ns
    else:
        int_reso = get_supported_reso(abbrev_to_npy_unit(parsed_unit))

    if values.descr.type_num != cnp.NPY_OBJECT:
        # raise here otherwise we segfault below
        raise TypeError("array_to_timedelta64 'values' must have object dtype")

    if errors not in {"ignore", "raise", "coerce"}:
        raise ValueError("errors must be one of {'ignore', 'raise', or 'coerce'}")

    if unit is not None and errors != "coerce":
        it = cnp.PyArray_IterNew(values)
        for i in range(n):
            # Analogous to: item = values[i]
            item = cnp.PyArray_GETITEM(values, cnp.PyArray_ITER_DATA(it))
            if isinstance(item, str):
                raise ValueError(
                    "unit must not be specified if the input contains a str"
                )
            cnp.PyArray_ITER_NEXT(it)

    for i in range(n):
        item = <object>(<PyObject**>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

        try:
            if checknull_with_nat_and_na(item):
                ival = NPY_NAT

            elif cnp.is_timedelta64_object(item):
                # TODO: de-duplicate this with Timedelta.__new__
                ival = cnp.get_timedelta64_value(item)
                dt64_reso = get_datetime64_unit(item)
                if not (
                    is_supported_unit(dt64_reso) or
                    dt64_reso in [
                        NPY_DATETIMEUNIT.NPY_FR_m,
                        NPY_DATETIMEUNIT.NPY_FR_h,
                        NPY_DATETIMEUNIT.NPY_FR_D,
                        NPY_DATETIMEUNIT.NPY_FR_W,
                        NPY_DATETIMEUNIT.NPY_FR_GENERIC
                    ]
                ):
                    err = npy_unit_to_abbrev(dt64_reso)
                    raise ValueError(
                        f"Unit {err} is not supported. "
                        "Only unambiguous timedelta values durations are supported. "
                        "Allowed units are 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns'")

                item_reso = get_supported_reso(dt64_reso)
                state.update_creso(item_reso)
                if infer_reso:
                    creso = state.creso
                if dt64_reso != NPY_DATETIMEUNIT.NPY_FR_GENERIC:
                    try:
                        ival = convert_reso(
                            ival,
                            dt64_reso,
                            creso,
                            round_ok=True,
                        )
                    except (OverflowError, OutOfBoundsDatetime) as err:
                        raise OutOfBoundsTimedelta(item) from err
                else:
                    # e.g. NaT
                    pass

            elif isinstance(item, _Timedelta):
                item_reso = item._creso
                state.update_creso(item_reso)
                if infer_reso:
                    creso = state.creso

                ival = (<_Timedelta>item)._as_creso(creso)._value

            elif PyDelta_Check(item):
                # i.e. isinstance(item, timedelta)
                item_reso = NPY_DATETIMEUNIT.NPY_FR_us
                state.update_creso(item_reso)
                if infer_reso:
                    creso = state.creso
                ival = delta_to_nanoseconds(item, reso=creso)

            elif isinstance(item, str):
                if (
                    (len(item) > 0 and item[0] == "P")
                    or (len(item) > 1 and item[:2] == "-P")
                ):
                    ival = parse_iso_format_string(item)
                else:
                    ival = parse_timedelta_string(item)
                if (
                    (infer_reso or creso == NPY_DATETIMEUNIT.NPY_FR_us)
                    and not needs_nano_unit(ival, item)
                ):
                    item_reso = NPY_DATETIMEUNIT.NPY_FR_us
                    ival = ival // 1000
                else:
                    item_reso = NPY_FR_ns

                if ival != NPY_NAT:
                    state.update_creso(item_reso)
                    if infer_reso:
                        creso = state.creso

            elif is_tick_object(item):
                item_reso = get_supported_reso(item._creso)
                state.update_creso(item_reso)
                if infer_reso:
                    creso = state.creso
                ival = delta_to_nanoseconds(item, reso=creso)

            elif is_integer_object(item):
                if item == NPY_NAT:
                    ival = NPY_NAT
                else:
                    # GH#65150 match the pattern in tslib.pyx: update creso
                    #  first, then convert using creso as the output resolution.
                    item_reso = int_reso
                    state.update_creso(item_reso)
                    if infer_reso:
                        creso = state.creso

                    ival = _numeric_to_td64ns(item, parsed_unit, creso)

            elif is_float_object(item):
                int_item = int(item)
                if item == int_item:
                    # round float -> treat this like an int, giving
                    #  int_reso where possible
                    item = int_item

                    if item == NPY_NAT:
                        ival = NPY_NAT
                    else:
                        item_reso = int_reso
                        state.update_creso(item_reso)
                        if infer_reso:
                            creso = state.creso

                        ival = _numeric_to_td64ns(item, parsed_unit, creso)
                else:
                    item_reso = NPY_FR_ns
                    state.update_creso(item_reso)
                    if infer_reso:
                        creso = state.creso

                    ival = _numeric_to_td64ns(item, parsed_unit, creso)

            else:
                raise TypeError(f"Invalid type for timedelta scalar: {type(item)}")

        except ValueError as err:
            if errors == "coerce":
                ival = NPY_NAT
            elif "unit abbreviation w/o a number" in str(err):
                # re-raise with more pertinent message
                msg = f"Could not convert '{item}' to NumPy timedelta"
                raise ValueError(msg) from err
            else:
                raise

        (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = ival

        cnp.PyArray_MultiIter_NEXT(mi)

    if infer_reso:
        if state.creso_ever_changed:
            # We encountered mismatched resolutions, need to re-parse with
            #  the correct one.
            return array_to_timedelta64(
                values,
                unit=unit,
                errors=errors,
                creso=state.creso,
            )
        elif creso == NPY_DATETIMEUNIT.NPY_FR_GENERIC:
            # i.e. we never encountered anything non-NaT, default to "s". This
            # ensures that insert and concat-like operations with NaT
            # do not upcast units
            result = iresult.view("m8[s]")
        else:
            # Otherwise we can use the single reso that we encountered and avoid
            #  a second pass.
            abbrev = npy_unit_to_abbrev(state.creso)
            result = iresult.view(f"m8[{abbrev}]")
    else:
        abbrev = npy_unit_to_abbrev(creso)
        result = result.view(f"m8[{abbrev}]")
    return result


@cython.cpow(True)
cdef int64_t parse_timedelta_string(str ts) except? -1:
    # [FUNCTION DOC]
    # Parses a non-ISO timedelta string into an integer number of nanoseconds.
    # Inputs: ts - string like "1 days 02:03:04.5", "-1d2h", "10ns", "1.5 s", "3:04",
    # "12" (bare number = ns).
    # Outputs: int64 nanoseconds; NPY_NAT for empty strings or NaT strings. Algorithm:
    # single left-to-right scan accumulating digits (number), fraction digits (frac) and
    # unit letters (unit). Spaces, commas and "+" are skipped. A leading "-" sets neg,
    # which applies to every subsequent component (a "-" after a value is an error).
    # When a digit follows unit letters, the pending number/fraction/unit is converted
    # via timedelta_from_spec and added. Colons switch to hh:mm:ss mode: each colon
    # converts the pending number as hours, then minutes, then seconds. A "." after
    # hh:mm marks seconds; trailing fraction digits are scaled to ns (truncated to 9
    # digits). At end, the pending component is converted; a lone number with no unit is
    # nanoseconds.
    # Exceptions: ValueError for non-leading "-", malformed hh:mm:ss, fraction without
    # units, unit without number ("unit abbreviation w/o a number"), symbols without
    # numbers, leftover units, invalid or ambiguous (M, Y, y) units; OverflowError for
    # huge values.
    # Side effects: may emit deprecation warnings via parse_timedelta_unit.
    """
    Parse a regular format timedelta string. Return an int64_t (in ns)
    or raise a ValueError on an invalid parse.
    """

    cdef:
        str c
        bint neg = 0, have_dot = 0, have_value = 0, have_hhmmss = 0
        str current_unit = None
        int64_t result = 0, m = 0, r
        list number = [], frac = [], unit = []

    # neg : tracks if we have a leading negative for the value
    # have_dot : tracks if we are processing a dot (either post hhmmss or
    #            inside an expression)
    # have_value : track if we have at least 1 leading unit
    # have_hhmmss : tracks if we have a regular format hh:mm:ss

    if len(ts) == 0 or ts in nat_strings:
        return NPY_NAT

    for c in ts:

        # skip whitespace / commas
        if c == " " or c == ",":
            pass

        # positive signs are ignored
        elif c == "+":
            pass

        # neg
        elif c == "-":

            if neg or have_value or have_hhmmss:
                raise ValueError("only leading negative signs are allowed")

            neg = 1

        # number (ascii codes)
        elif ord(c) >= 48 and ord(c) <= 57:

            if have_dot:

                # we found a dot, but now its just a fraction
                if len(unit):
                    number.append(c)
                    have_dot = 0
                else:
                    frac.append(c)

            elif not len(unit):
                number.append(c)

            else:
                r = timedelta_from_spec(number, frac, unit)
                unit, number, frac = [], [c], []

                result += timedelta_as_neg(r, neg)

        # hh:mm:ss.
        elif c == ":":

            # we flip this off if we have a leading value
            if have_value:
                neg = 0

            # we are in the pattern hh:mm:ss pattern
            if len(number):
                if current_unit is None:
                    current_unit = "h"
                    m = 1000000000 * 3600
                elif current_unit == "h":
                    current_unit = "m"
                    m = 1000000000 * 60
                elif current_unit == "m":
                    current_unit = "s"
                    m = 1000000000
                r = <int64_t>int("".join(number)) * m
                result += timedelta_as_neg(r, neg)
                have_hhmmss = 1
            else:
                raise ValueError(f"expecting hh:mm:ss format, received: {ts}")

            unit, number = [], []

        # after the decimal point
        elif c == ".":

            if len(number) and current_unit is not None:

                # by definition we had something like
                # so we need to evaluate the final field from a
                # hh:mm:ss (so current_unit is 'm')
                if current_unit != "m":
                    raise ValueError("expected hh:mm:ss format before .")
                m = 1000000000
                r = <int64_t>int("".join(number)) * m
                result += timedelta_as_neg(r, neg)
                have_value = 1
                unit, number, frac = [], [], []

            have_dot = 1

        # unit
        else:
            unit.append(c)
            have_value = 1
            have_dot = 0

    # we had a dot, but we have a fractional
    # value since we have a unit
    if have_dot and len(unit):
        r = timedelta_from_spec(number, frac, unit)
        result += timedelta_as_neg(r, neg)

    # we have a dot as part of a regular format
    # e.g. hh:mm:ss.fffffff
    elif have_dot:

        if ((len(number) or len(frac)) and not len(unit)
                and current_unit is None):
            raise ValueError("no units specified")

        if len(frac) > 0 and len(frac) <= 3:
            m = 10**(3 -len(frac)) * 1000 * 1000
        elif len(frac) > 3 and len(frac) <= 6:
            m = 10**(6 -len(frac)) * 1000
        elif len(frac) > 6 and len(frac) <= 9:
            m = 10**(9 -len(frac))
        else:
            m = 1
            frac = frac[:9]
        r = <int64_t>int("".join(frac)) * m
        result += timedelta_as_neg(r, neg)

    # we have a regular format
    # we must have seconds at this point (hence the unit is still 'm')
    elif current_unit is not None:
        if current_unit != "m":
            raise ValueError("expected hh:mm:ss format")
        m = 1000000000
        r = <int64_t>int("".join(number)) * m
        result += timedelta_as_neg(r, neg)

    # we have a last abbreviation
    elif len(unit):
        if len(number):
            r = timedelta_from_spec(number, frac, unit)
            result += timedelta_as_neg(r, neg)
        else:
            raise ValueError("unit abbreviation w/o a number")

    # we only have symbols and no numbers
    elif len(number) == 0:
        raise ValueError("symbols w/o a number")

    # treat as nanoseconds
    # but only if we don't have anything else
    else:
        if have_value:
            raise ValueError("have leftover units")
        if len(number):
            r = timedelta_from_spec(number, frac, "ns")
            result += timedelta_as_neg(r, neg)

    return result


cdef int64_t timedelta_as_neg(int64_t value, bint neg):
    # [FUNCTION DOC]
    # HELPER: Returns -value if neg is true, else value.
    # Inputs: value - int64; neg - boolean. Exceptions/side effects: none.
    """

    Parameters
    ----------
    value : int64_t of the timedelta value
    neg : bool if the a negative value
    """
    if neg:
        return -value
    return value


cdef timedelta_from_spec(object number, object frac, object unit):
    # [FUNCTION DOC]
    # HELPER: Converts one parsed component into nanoseconds.
    # Inputs: number - sequence of digit characters (integer part); frac - sequence or
    # string of fractional digits; unit - sequence of characters (or a string) forming a
    # unit abbreviation.
    # Outputs: integer nanoseconds, computed by joining number "." frac into a float and
    # scaling by the unit (cast_from_unit with ns target).
    # Exceptions: ValueError for units M, Y, y or unknown abbreviations; ValueError if
    # the joined number is not a valid float (e.g. empty).
    # Side effects: may emit a deprecation warning for deprecated unit spellings.
    # Note: goes through a float, so very large values may lose precision.
    """

    Parameters
    ----------
    number : a list of number digits
    frac : a list of frac digits
    unit : a list of unit characters
    """
    cdef:
        str n

    unit = "".join(unit)
    if unit in ["M", "Y", "y"]:
        raise ValueError(
            "Units 'M', 'Y' and 'y' do not represent unambiguous timedelta "
            "values and are not supported."
        )

    unit = parse_timedelta_unit(unit)

    n = "".join(number) + "." + "".join(frac)
    return cast_from_unit(float(n), unit)


cdef bint needs_nano_unit(int64_t ival, str item):
    # [FUNCTION DOC]
    # HELPER: Decides if a parsed string value must be stored at ns resolution rather
    # than us.
    # Inputs: ival - parsed value in ns; item - original string.
    # Outputs: True if ival has a sub-microsecond remainder, if the string has a dot
    # followed by 7+ digits, or if it contains "ns" or "nano" (case-insensitive for
    # "nano"). Otherwise False.
    # Exceptions/side effects: none.
    """
    Check if a passed string `item` needs to be stored with nano unit or can
    use microsecond instead. Needs nanoseconds if:

    - if the parsed value in nanoseconds has sub-microseconds content -> certainly
      needs nano
    - if the seconds part in the string contains more than 6 decimals, i.e. has
      trailing zeros beyond the microsecond part (e.g. "0.123456000 s") -> treat
      as nano for consistency
    - if the string explicitly contains an entry for nanoseconds (e.g. "1000 ns")
    """
    # TODO: more performant way of doing this check?
    if ival % 1000 != 0:
        return True
    return re.search(r"\.\d{7}", item) or "ns" in item or "nano" in item.lower()


cpdef inline str parse_timedelta_unit(str unit):
    # [FUNCTION DOC]
    # Normalizes a user unit spelling to the canonical abbreviation.
    # Inputs: unit - str or None.
    # Outputs: "ns" for None; "M" unchanged for "M"; otherwise the lookup of the lower-
    # cased unit in timedelta_abbrevs (e.g. "days" -> "D", "minute" -> "m"). Deprecated
    # spellings in c_DEPR_UNITS are first replaced by their new spelling.
    # Exceptions: ValueError "invalid unit abbreviation" for unknown units.
    # Side effects: emits Pandas4Warning for deprecated spellings.
    """
    Parameters
    ----------
    unit : str or None

    Returns
    -------
    str
        Canonical unit string.

    Raises
    ------
    ValueError : on non-parseable input
    """
    if unit is None:
        return "ns"
    elif unit == "M":
        return unit
    elif unit in c_DEPR_UNITS:
        from pandas.errors import Pandas4Warning

        # https://github.com/pandas-dev/pandas/pull/59240
        warnings.warn(
            f"\'{unit}\' is deprecated and will be removed in a "
            f"future version. Please use \'{c_DEPR_UNITS.get(unit)}\' "
            f"instead of \'{unit}\'.",
            Pandas4Warning,
            stacklevel=find_stack_level(),
        )
        unit = c_DEPR_UNITS[unit]
    try:
        return timedelta_abbrevs[unit.lower()]
    except KeyError:
        raise ValueError(f"invalid unit abbreviation: {unit}")

# ----------------------------------------------------------------------
# Timedelta ops utilities

cdef bint _validate_ops_compat(other):
    # [FUNCTION DOC]
    # HELPER: True if other can participate in Timedelta add/sub: a null (None, NaN,
    # NaT), a timedelta-like scalar (timedelta, timedelta64, Tick) or a string.
    # Exceptions/side effects: none.
    # return True if we are compat with operating
    if checknull_with_nat(other):
        return True
    elif is_any_td_scalar(other):
        return True
    elif isinstance(other, str):
        return True
    return False


def _op_unary_method(func, name):
    # [FUNCTION DOC]
    # HELPER: Factory for Timedelta unary operators (neg, pos, abs).
    # Inputs: func - function from int to int applied to _value; name - method name.
    # Outputs: a method returning a new Timedelta with func(_value) at the same
    # resolution.
    def f(self):
        # [FUNCTION DOC]
        # HELPER: Generated unary method; returns Timedelta(func(self._value)) keeping
        # self's resolution. Exceptions: OverflowError if the result does not fit int64.
        new_value = func(self._value)
        return _timedelta_from_value_and_reso(Timedelta, new_value, self._creso)
    f.__name__ = name
    return f


def _binary_op_method_timedeltalike(op, name):
    # [FUNCTION DOC]
    # HELPER: Factory for Timedelta binary operators that require a timedelta-like
    # operand (add, radd, sub, rsub).
    # Inputs: op - function of two arguments (operands already in matching order for
    # reflected variants); name - method name.
    # Outputs: the method f described below.
    # define a binary operation that only works if the other argument is
    # timedelta like or an array of timedeltalike
    def f(self, other):
        # [FUNCTION DOC]
        # HELPER: Generated binary method combining self (Timedelta) with other.
        # Outputs: NaT if other is NaT; for datetime64 or non-Timestamp datetime,
        # applies op to self and Timestamp(other); for 0-dim arrays, recurses on the
        # scalar; for timedelta64/datetime64 arrays, applies op to self as timedelta64
        # and the array; for object arrays, element-wise list result as ndarray; other
        # arrays and incompatible types -> NotImplemented. Otherwise converts other to
        # Timedelta (strings warn as deprecated; unparseable -> NotImplemented), returns
        # NaT if it is NaT, casts both to the finer resolution, applies op to the
        # integer values, and returns a Timedelta at that resolution (NaT if the result
        # equals NPY_NAT).
        # Exceptions: OutOfBoundsTimedelta when resolution casting overflows; errors
        # from int64 conversion if the result overflows.
        # Side effects: may emit Pandas4Warning for string operands.
        if other is NaT:
            return NaT

        elif cnp.is_datetime64_object(other) or (
            PyDateTime_Check(other) and not isinstance(other, ABCTimestamp)
        ):
            # this case is for a datetime object that is specifically
            # *not* a Timestamp, as the Timestamp case will be
            # handled after `_validate_ops_compat` returns False below
            from pandas._libs.tslibs.timestamps import Timestamp
            return op(self, Timestamp(other))
            # We are implicitly requiring the canonical behavior to be
            # defined by Timestamp methods.

        elif is_array(other):
            if other.ndim == 0:
                # see also: item_from_zerodim
                item = cnp.PyArray_ToScalar(cnp.PyArray_DATA(other), other)
                return f(self, item)

            elif other.dtype.kind in "mM":
                return op(self.to_timedelta64(), other)
            elif other.dtype.kind == "O":
                return np.array([op(self, x) for x in other])
            else:
                return NotImplemented

        elif not _validate_ops_compat(other):
            # Includes any of our non-cython classes
            return NotImplemented

        try:
            other = _wrapped_to_timedelta(other)
        except ValueError:
            # failed to parse as timedelta
            return NotImplemented

        if other is NaT:
            # e.g. if original other was timedelta64('NaT')
            return NaT

        # Matching numpy, we cast to the higher resolution. Unlike numpy,
        #  we raise instead of silently overflowing during this casting.
        if self._creso < other._creso:
            self = (<_Timedelta>self)._as_creso(other._creso, round_ok=True)
        elif self._creso > other._creso:
            other = (<_Timedelta>other)._as_creso(self._creso, round_ok=True)

        res = op(self._value, other._value)
        if res == NPY_NAT:
            # e.g. test_implementation_limits
            # TODO: more generally could do an overflowcheck in op?
            return NaT

        return _timedelta_from_value_and_reso(Timedelta, res, reso=self._creso)

    f.__name__ = name
    return f


# ----------------------------------------------------------------------
# Timedelta Construction

cpdef disallow_ambiguous_unit(unit):
    # [FUNCTION DOC]
    # Validates that unit is not one of the ambiguous units "Y", "y", "M".
    # Inputs: unit - any. Outputs: None.
    # Exceptions: ValueError for ambiguous units. Side effects: none.
    if unit in {"Y", "y", "M"}:
        raise ValueError(
            "Units 'M', 'Y', and 'y' are no longer supported, as they do not "
            "represent unambiguous timedelta values durations."
        )


cdef int64_t parse_iso_format_string(str ts) except? -1:
    # [FUNCTION DOC]
    # Parses an ISO 8601 duration string (e.g. "P1DT2H3M4.5S", "-P1W") into ns.
    # Inputs: ts - non-empty string starting with "P" or "-P".
    # Outputs: int64 nanoseconds. Algorithm: a leading "-" negates the whole result.
    # Scan characters: digits accumulate into number; P/T/"+" are ignored; "-" inside
    # sets a per-component negative flag (only before any digit has been seen); W, D, H,
    # M (H/M at most 2 digits) convert the pending number in weeks, days, hours,
    # minutes; "." flushes the pending number as seconds and starts fraction mode where
    # digits are grouped by threes into ms, us, ns; "S" converts the pending number as
    # seconds or as the current fraction unit (padding to 3 digits).
    # Exceptions: ValueError "Invalid ISO 8601 Duration format" for unknown characters,
    # misplaced "-", too many digits, fractions without digits or longer than 9 digits,
    # or strings with no digits; IndexError for an empty string.
    # Side effects: none.
    """
    Extracts and cleanses the appropriate values from a match object with
    groups for each component of an ISO 8601 duration

    Parameters
    ----------
    ts: str
        ISO 8601 Duration formatted string

    Returns
    -------
    ns: int64_t
        Precision in nanoseconds of matched ISO 8601 duration

    Raises
    ------
    ValueError
        If ``ts`` cannot be parsed
    """

    cdef:
        unicode c
        int64_t result = 0, r
        int p = 0, sign = 1
        object dec_unit = "ms", err_msg
        bint have_dot = 0, have_value = 0, neg = 0
        list number = [], unit = []

    err_msg = f"Invalid ISO 8601 Duration format - {ts}"

    if ts[0] == "-":
        sign = -1
        ts = ts[1:]

    for c in ts:
        # number (ascii codes)
        if 48 <= ord(c) <= 57:

            have_value = 1
            if have_dot:
                if p == 3 and dec_unit != "ns":
                    unit.append(dec_unit)
                    if dec_unit == "ms":
                        dec_unit = "us"
                    elif dec_unit == "us":
                        dec_unit = "ns"
                    p = 0
                p += 1

            if not len(unit):
                number.append(c)
            else:
                r = timedelta_from_spec(number, "0", unit)
                result += timedelta_as_neg(r, neg)

                neg = 0
                unit, number = [], [c]
        else:
            if c == "P" or c == "T":
                pass  # ignore marking characters P and T
            elif c == "-":
                if neg or have_value:
                    raise ValueError(err_msg)
                else:
                    neg = 1
            elif c == "+":
                pass
            elif c in ["W", "D", "H", "M"]:
                if c in ["H", "M"] and len(number) > 2:
                    raise ValueError(err_msg)
                if c in ["M", "H"]:
                    c = c.replace("M", "min").replace("H", "h")
                unit.append(c)
                r = timedelta_from_spec(number, "0", unit)
                result += timedelta_as_neg(r, neg)

                neg = 0
                unit, number = [], []
            elif c == ".":
                # append any seconds
                if len(number):
                    r = timedelta_from_spec(number, "0", "s")
                    result += timedelta_as_neg(r, neg)
                    unit, number = [], []
                have_dot = 1
            elif c == "S":
                if have_dot:  # ms, us, or ns
                    if not len(number) or p > 3:
                        raise ValueError(err_msg)
                    # pad to 3 digits as required
                    pad = 3 - p
                    while pad > 0:
                        number.append("0")
                        pad -= 1

                    r = timedelta_from_spec(number, "0", dec_unit)
                    result += timedelta_as_neg(r, neg)
                else:  # seconds
                    r = timedelta_from_spec(number, "0", "s")
                    result += timedelta_as_neg(r, neg)
            else:
                raise ValueError(err_msg)

    if not have_value:
        # Received string only - never parsed any values
        raise ValueError(err_msg)

    return sign*result


cdef _to_py_int_float(v):
    # [FUNCTION DOC]
    # HELPER: Converts a numeric constructor keyword value to a Python int or float.
    # Inputs: v - any. Outputs: int(v) for integer objects, float(v) for floats.
    # Exceptions: TypeError for other types. Side effects: none.
    # Note: This used to be defined inside Timedelta.__new__
    # but cython will not allow `cdef` functions to be defined dynamically.
    if is_integer_object(v):
        return int(v)
    elif is_float_object(v):
        return float(v)
    raise TypeError(f"Invalid type {type(v)}. Must be int or float.")


def _timedelta_unpickle(value, reso):
    # [FUNCTION DOC]
    # HELPER: Unpickling hook used by Timedelta.__reduce__; rebuilds a Timedelta from
    # its integer value and resolution code. Exceptions: as the factory below.
    return _timedelta_from_value_and_reso(Timedelta, value, reso)


cdef _timedelta_from_value_and_reso(cls, int64_t value, NPY_DATETIMEUNIT reso):
    # [FUNCTION DOC]
    # HELPER: Low-level factory creating a Timedelta (or subclass cls) instance from an
    # integer value and resolution without validation or conversion.
    # Inputs: cls - Timedelta class; value - int64, must not be NPY_NAT (asserted); reso
    # - s, ms, us or ns.
    # Outputs: new instance with _value, _creso set and _is_populated = 0. The
    # underlying datetime.timedelta part is initialized from the value (us from ns by
    # floor division, us directly; for ms/s only when within timedelta range, otherwise
    # zero) so plain datetime arithmetic works where possible.
    # Exceptions: NotImplementedError for other resolutions; AssertionError for NaT;
    # OverflowError if ns/us value exceeds datetime.timedelta range.
    # Side effects: allocates the object.
    # Could make this a classmethod if/when cython supports cdef classmethods
    cdef:
        _Timedelta td_base

    assert value != NPY_NAT
    # For millisecond and second resos, we cannot actually pass int(value) because
    #  many cases would fall outside of the pytimedelta implementation bounds.
    #  We pass 0 instead, and override seconds, microseconds, days.
    #  In principle we could pass 0 for ns and us too.
    if reso == NPY_FR_ns:
        td_base = _Timedelta.__new__(cls, microseconds=int(value) // 1000)
    elif reso == NPY_DATETIMEUNIT.NPY_FR_us:
        td_base = _Timedelta.__new__(cls, microseconds=int(value))
    elif reso == NPY_DATETIMEUNIT.NPY_FR_ms:
        if -86_399_999_913_600_000 <= value < 86_400_000_000_000_000:
            # i.e. we are in range for pytimedelta. By passing the
            #  'correct' value here we can
            #   make pydatetime + Timedelta operations work correctly,
            #  xref GH#53643
            td_base = _Timedelta.__new__(cls, milliseconds=value)
        else:
            td_base = _Timedelta.__new__(cls, milliseconds=0)
    elif reso == NPY_DATETIMEUNIT.NPY_FR_s:
        if -86_399_999_913_600 <= value < 86_400_000_000_000:
            # i.e. we are in range for pytimedelta. By passing the
            #  'correct' value here we can
            #   make pydatetime + Timedelta operations work correctly,
            #  xref GH#53643
            td_base = _Timedelta.__new__(cls, seconds=value)
        else:
            td_base = _Timedelta.__new__(cls, seconds=0)
    # Other resolutions are disabled but could potentially be implemented here:
    # elif reso == NPY_DATETIMEUNIT.NPY_FR_m:
    #    td_base = _Timedelta.__new__(Timedelta, minutes=int(value))
    # elif reso == NPY_DATETIMEUNIT.NPY_FR_h:
    #    td_base = _Timedelta.__new__(Timedelta, hours=int(value))
    # elif reso == NPY_DATETIMEUNIT.NPY_FR_D:
    #    td_base = _Timedelta.__new__(Timedelta, days=int(value))
    else:
        raise NotImplementedError(
            "Only resolutions 's', 'ms', 'us', 'ns' are supported."
        )

    td_base._value = value
    td_base._is_populated = 0
    td_base._creso = reso
    return td_base


class MinMaxReso:
    # [CLASS DOC]
    # HELPER: Descriptor implementing Timedelta.min, .max and .resolution so they depend
    # on the instance's resolution. On the class they are nanosecond values; on an
    # instance they use its _creso. Read-only.
    """
    We need to define min/max/resolution on both the Timedelta _instance_
    and Timedelta class.  On an instance, these depend on the object's _reso.
    On the class, we default to the values we would get with nanosecond _reso.
    """
    def __init__(self, name, docstring):
        # [FUNCTION DOC]
        # HELPER: Stores the attribute name ("min", "max" or "resolution") and
        # docstring.
        # Side effects: sets _name and __doc__.
        self._name = name
        self.__doc__ = docstring

    def __get__(self, obj, type=None):
        # [FUNCTION DOC]
        # HELPER: Descriptor getter.
        # Inputs: obj - instance or None (class access); type - owner class (unused).
        # Outputs: Timedelta with value min int64 + 1 (min), max int64 (max) or 1
        # (resolution); at ns when accessed on the class, else at obj's resolution. The
        # result's __doc__ is set to the descriptor docstring.
        # Exceptions: AssertionError for an unknown name.
        # Side effects: sets __doc__ on the returned instance.
        if self._name == "min":
            val = np.iinfo(np.int64).min + 1
        elif self._name == "max":
            val = np.iinfo(np.int64).max
        else:
            assert self._name == "resolution"
            val = 1

        if obj is None:
            # i.e. this is on the class, default to nanos
            result = Timedelta(val)
        else:
            result = Timedelta._from_value_and_reso(val, obj._creso)

        result.__doc__ = self.__doc__

        return result

    def __set__(self, obj, value):
        # [FUNCTION DOC]
        # HELPER: Descriptor setter; always raises AttributeError "<name> is not
        # settable.".
        raise AttributeError(f"{self._name} is not settable.")


# Similar to Timestamp/datetime, this is a construction requirement for
# timedeltas that we need to do object instantiation in python. This will
# serve as a C extension type that shadows the Python class, where we do any
# heavy lifting.
cdef class _Timedelta(timedelta):
    # [CLASS DOC]
    # C extension base class of Timedelta, subclassing datetime.timedelta. Fields
    # (declared in timedeltas.pxd): _value (int64 count of _creso units), _creso
    # (resolution), _is_populated and the cached components _d, _h, _m, _s, _ms, _us,
    # _ns. __array_priority__ = 100 makes numpy defer to Timedelta operators. The
    # days/seconds/microseconds attributes of datetime.timedelta are overridden to be
    # computed from _value so they are correct for all resolutions.
    # cdef readonly:
    #    int64_t value      # nanoseconds
    #    bint _is_populated  # are my components populated
    #    int64_t _d, _h, _m, _s, _ms, _us, _ns
    #    NPY_DATETIMEUNIT _reso

    # higher than np.ndarray and np.matrix
    __array_priority__ = 100

    _docstring_min = """
    Returns the minimum bound possible for Timedelta.

    This property provides access to the smallest possible value that
    can be represented by a Timedelta object.

    Returns
    -------
    Timedelta

    See Also
    --------
    Timedelta.max: Returns the maximum bound possible for Timedelta.
    Timedelta.resolution: Returns the smallest possible difference between
        non-equal Timedelta objects.

    Examples
    --------
    >>> pd.Timedelta.min
    -106752 days +00:12:43.145224193
    """

    _docstring_max = """
    Returns the maximum bound possible for Timedelta.

    This property provides access to the largest possible value that
    can be represented by a Timedelta object.

    Returns
    -------
    Timedelta

    See Also
    --------
    Timedelta.min: Returns the minimum bound possible for Timedelta.
    Timedelta.resolution: Returns the smallest possible difference between
        non-equal Timedelta objects.

    Examples
    --------
    >>> pd.Timedelta.max
    106751 days 23:47:16.854775807
    """

    _docstring_reso = """
    Returns the smallest possible difference between non-equal Timedelta objects.

    The resolution value is determined by the underlying representation of time
    units and is equivalent to Timedelta(nanoseconds=1).

    Returns
    -------
    Timedelta

    See Also
    --------
    Timedelta.max: Returns the maximum bound possible for Timedelta.
    Timedelta.min: Returns the minimum bound possible for Timedelta.

    Examples
    --------
    >>> pd.Timedelta.resolution
    0 days 00:00:00.000000001
    """

    min = MinMaxReso("min", _docstring_min)
    max = MinMaxReso("max", _docstring_max)
    resolution = MinMaxReso("resolution", _docstring_reso)

    @property
    def value(self):
        # [FUNCTION DOC]
        # Property: the duration as an integer number of nanoseconds, converting from
        # _creso without rounding.
        # Exceptions: OverflowError if the value cannot be represented in ns.
        """
        Return the value of Timedelta object in nanoseconds.

        Return the total seconds, milliseconds and microseconds
        of the timedelta as nanoseconds.

        Returns
        -------
        int

        See Also
        --------
        Timedelta.unit : Return the unit of Timedelta object.

        Examples
        --------
        >>> pd.Timedelta(1, "us").value
        1000
        """
        try:
            return convert_reso(self._value, self._creso, NPY_FR_ns, False)
        except OverflowError:
            raise OverflowError(
                "Cannot convert Timedelta to nanoseconds without overflow. "
                "Use `.asm8.view('i8')` to cast represent Timedelta in its own "
                f"unit (here, {self.unit})."
            )

    @property
    def _unit(self) -> str:
        # [FUNCTION DOC]
        # Property: abbreviation of the resolution (e.g. "ns", "us").
        """
        The abbreviation associated with self._creso.
        """
        return npy_unit_to_abbrev(self._creso)

    @property
    def days(self) -> int:  # TODO(cython3): make cdef property
        # [FUNCTION DOC]
        # Property: day component (floor of the duration in days; negative durations
        # have negative days and non-negative remaining components).
        # Side effects: populates the component cache.
        """
        Returns the days of the timedelta.

        The `days` attribute of a `pandas.Timedelta` object provides the number
        of days represented by the `Timedelta`. This is useful for extracting
        the day component from a `Timedelta` that may also include hours, minutes,
        seconds, and smaller time units. This attribute simplifies the process
        of working with durations where only the day component is of interest.

        Returns
        -------
        int

        See Also
        --------
        Timedelta.seconds : Returns the seconds component of the timedelta.
        Timedelta.microseconds : Returns the microseconds component of the timedelta.
        Timedelta.total_seconds : Returns the total duration in seconds.

        Examples
        --------
        >>> td = pd.Timedelta(1, "d")
        >>> td.days
        1

        >>> td = pd.Timedelta('4 min 3 us 42 ns')
        >>> td.days
        0
        """
        # NB: using the python C-API PyDateTime_DELTA_GET_DAYS will fail
        #  (or be incorrect)
        self._ensure_components()
        return self._d

    @property
    def seconds(self) -> int:  # TODO(cython3): make cdef property
        # [FUNCTION DOC]
        # Property: seconds within the day, hours * 3600 + minutes * 60 + seconds, in
        # [0, 86400). Side effects: populates the component cache.
        """
        Return the total hours, minutes, and seconds of the timedelta as seconds.

        Timedelta.seconds = hours * 3600 + minutes * 60 + seconds.

        Returns
        -------
        int
            Number of seconds.

        See Also
        --------
        Timedelta.components : Return all attributes with assigned values
            (i.e. days, hours, minutes, seconds, milliseconds, microseconds,
            nanoseconds).
        Timedelta.total_seconds : Express the Timedelta as total number of seconds.

        Examples
        --------
        **Using string input**

        >>> td = pd.Timedelta('1 days 2 min 3 us 42 ns')
        >>> td.seconds
        120

        **Using integer input**

        >>> td = pd.Timedelta(42, unit='s')
        >>> td.seconds
        42
        """
        # NB: using the python C-API PyDateTime_DELTA_GET_SECONDS will fail
        #  (or be incorrect)
        self._ensure_components()
        return self._h * 3600 + self._m * 60 + self._s

    @property
    def microseconds(self) -> int:  # TODO(cython3): make cdef property
        # [FUNCTION DOC]
        # Property: microseconds within the second, milliseconds * 1000 + microseconds,
        # in [0, 1000000). Side effects: populates the component cache.
        # KNOWN ISSUE: the docstring says 0 <= n < 1 millisecond, but the range is up to
        # one second (it includes the millisecond part).
        # NB: using the python C-API PyDateTime_DELTA_GET_MICROSECONDS will fail
        #  (or be incorrect)
        """
        Return the number of microseconds (n), where 0 <= n < 1 millisecond.

        Timedelta.microseconds = milliseconds * 1000 + microseconds.

        Returns
        -------
        int
            Number of microseconds.

        See Also
        --------
        Timedelta.components : Return all attributes with assigned values
            (i.e. days, hours, minutes, seconds, milliseconds, microseconds,
            nanoseconds).

        Examples
        --------
        **Using string input**

        >>> td = pd.Timedelta('1 days 2 min 3 us')

        >>> td.microseconds
        3

        **Using integer input**

        >>> td = pd.Timedelta(42, unit='us')
        >>> td.microseconds
        42
        """
        self._ensure_components()
        return self._ms * 1000 + self._us

    def total_seconds(self) -> float:
        # [FUNCTION DOC]
        # Returns the duration in seconds as a float: days * 86400 + seconds +
        # microseconds / 1e6.
        # KNOWN ISSUE: the nanosecond component is ignored (see the TODO), so values at
        # ns resolution with sub-microsecond parts are truncated to microseconds.
        """
        Total seconds in the duration.

        This method calculates the total duration in seconds by combining
        the days, seconds, and microseconds of the `Timedelta` object.

        See Also
        --------
        to_timedelta : Convert argument to timedelta.
        Timedelta : Represents a duration, the difference between two dates or times.
        Timedelta.seconds : Returns the seconds component of the timedelta.
        Timedelta.microseconds : Returns the microseconds component of the timedelta.

        Examples
        --------
        >>> td = pd.Timedelta('1min')
        >>> td
        Timedelta('0 days 00:01:00')
        >>> td.total_seconds()
        60.0
        """
        # We need to override bc we overrode days/seconds/microseconds
        # TODO: add nanos/1e9?
        return self.days * 24 * 3600 + self.seconds + self.microseconds / 1_000_000

    @property
    def unit(self) -> str:
        # [FUNCTION DOC]
        # Property: abbreviation of the resolution ("s", "ms", "us" or "ns").
        # KNOWN ISSUE: the docstring example omits the line accessing td.unit, and the
        # docstring says the unit is "ns" by default although construction often infers
        # "us" (e.g. from strings or keyword arguments).
        """
        Return the unit of Timedelta object.

        The unit of Timedelta object is nanosecond, i.e., 'ns' by default.

        Returns
        -------
        str

        See Also
        --------
        Timedelta.value : Return the value of Timedelta object in nanoseconds.
        Timedelta.as_unit : Convert the underlying int64 representation to
            the given unit.

        Examples
        --------
        >>> td = pd.Timedelta(42, unit='us')
        'ns'
        """
        return npy_unit_to_abbrev(self._creso)

    def __hash__(_Timedelta self):
        # [FUNCTION DOC]
        # Hash consistent with equality across resolutions where possible.
        # Outputs: if there is a sub-microsecond part, hash of the integer _value; if in
        # datetime.timedelta range at ns or us resolution, datetime.timedelta's hash
        # (matches equal Python timedeltas); otherwise converts to the next finer
        # resolution (unit code + 1, e.g. s -> ms) and hashes that, recursing until the
        # timedelta branch applies, or hashes _value if conversion overflows.
        # KNOWN ISSUE: the inline comment says "downcasting to the next-lowest
        # resolution", but unit code + 1 is a finer resolution. Also, as noted in code,
        # equal values at different resolutions with ns content may not hash equal.
        # Side effects: populates the component cache.
        if self._has_ns():
            # Note: this does *not* satisfy the invariance
            #  td1 == td2 \\Rightarrow hash(td1) == hash(td2)
            #  if td1 and td2 have different _resos. timedelta64 also has this
            #  non-invariant behavior.
            #  see GH#44504
            return hash(self._value)
        elif self._is_in_pytimedelta_bounds() and (
            self._creso == NPY_FR_ns or self._creso == NPY_DATETIMEUNIT.NPY_FR_us
        ):
            # If we can defer to timedelta.__hash__, do so, as that
            #  ensures the hash is invariant to our _reso.
            # We can only defer for ns and us, as for these two resos we
            #  call _Timedelta.__new__ with the correct input in
            #  _timedelta_from_value_and_reso; so timedelta.__hash__
            #  will be correct
            return timedelta.__hash__(self)
        else:
            # We want to ensure that two equivalent Timedelta objects
            #  have the same hash.  So we try downcasting to the next-lowest
            #  resolution.
            try:
                obj = (<_Timedelta>self)._as_creso(<NPY_DATETIMEUNIT>(self._creso + 1))
            except OutOfBoundsTimedelta:
                # Doesn't fit, so we're off the hook
                return hash(self._value)
            else:
                return hash(obj)

    def __richcmp__(_Timedelta self, object other, int op):
        # [FUNCTION DOC]
        # Rich comparison of a Timedelta with other.
        # Inputs: other - any object; op - comparison code.
        # Outputs: if other is a Timedelta or timedelta-like scalar (converted to
        # Timedelta), compares integer values directly when resolutions match, else via
        # _compare_mismatched_resos. If conversion of a datetime.timedelta overflows,
        # compares (days, seconds, microseconds, nanoseconds) tuples. NaT -> True only
        # for "!=". timedelta64 arrays: numpy comparison with self.asm8; object arrays:
        # element-wise boolean array; other arrays: all-False for "==", all-True for
        # "!=", NotImplemented otherwise. NotImplemented for other types.
        # Exceptions: OutOfBoundsTimedelta for overflowing timedelta64/Tick operands.
        cdef:
            _Timedelta ots

        if isinstance(other, _Timedelta):
            ots = other
        elif is_any_td_scalar(other):
            try:
                ots = Timedelta(other)
            except OutOfBoundsTimedelta as err:
                # GH#49021 pytimedelta.max overflows
                if not PyDelta_Check(other):
                    # TODO: handle this case
                    raise
                ltup = (self.days, self.seconds, self.microseconds, self.nanoseconds)
                rtup = (other.days, other.seconds, other.microseconds, 0)
                if op == Py_EQ:
                    return ltup == rtup
                elif op == Py_NE:
                    return ltup != rtup
                elif op == Py_LT:
                    return ltup < rtup
                elif op == Py_LE:
                    return ltup <= rtup
                elif op == Py_GT:
                    return ltup > rtup
                elif op == Py_GE:
                    return ltup >= rtup

        elif other is NaT:
            return op == Py_NE

        elif util.is_array(other):
            if other.dtype.kind == "m":
                return PyObject_RichCompare(self.asm8, other, op)
            elif other.dtype.kind == "O":
                # operate element-wise
                return np.array(
                    [PyObject_RichCompare(self, x, op) for x in other],
                    dtype=bool,
                )
            if op == Py_EQ:
                return np.zeros(other.shape, dtype=bool)
            elif op == Py_NE:
                return np.ones(other.shape, dtype=bool)
            return NotImplemented  # let other raise TypeError

        else:
            return NotImplemented

        if self._creso == ots._creso:
            return cmp_scalar(self._value, ots._value, op)
        return self._compare_mismatched_resos(ots, op)

    # TODO: reuse/share with Timestamp
    cdef bint _compare_mismatched_resos(self, _Timedelta other, op):
        # [FUNCTION DOC]
        # HELPER: Compares two Timedeltas with different resolutions without overflow by
        # converting each to a broken-down date/time struct (as if an epoch offset) and
        # comparing the structs field by field.
        # Inputs: other - Timedelta; op - comparison code. Outputs: boolean.
        # Can't just dispatch to numpy as they silently overflow and get it wrong
        cdef:
            npy_datetimestruct dts_self
            npy_datetimestruct dts_other

        # dispatch to the datetimestruct utils instead of writing new ones!
        pandas_datetime_to_datetimestruct(self._value, self._creso, &dts_self)
        pandas_datetime_to_datetimestruct(other._value, other._creso, &dts_other)
        return cmp_dtstructs(&dts_self,  &dts_other, op)

    cdef bint _has_ns(self):
        # [FUNCTION DOC]
        # HELPER: True if the value has a non-zero sub-microsecond part. Always False
        # for s/ms/us resolution.
        # Exceptions: NotImplementedError for resolutions finer than ns.
        if self._creso == NPY_FR_ns:
            return self._value % 1000 != 0
        elif self._creso < NPY_FR_ns:
            # i.e. seconds, millisecond, microsecond
            return False
        else:
            raise NotImplementedError(self._creso)

    cdef bint _is_in_pytimedelta_bounds(self):
        # [FUNCTION DOC]
        # HELPER: True if the day component is within datetime.timedelta's limits
        # (+-999999999 days). Side effects: populates the component cache.
        """
        Check if we are within the bounds of datetime.timedelta.
        """
        self._ensure_components()
        return -999999999 <= self._d and self._d <= 999999999

    @cython.critical_section
    cdef _ensure_components(_Timedelta self):
        # [FUNCTION DOC]
        # HELPER: Lazily computes and caches the components of _value at _creso (days,
        # hours, minutes, seconds, ms, us, ns, plus total seconds-in-day and
        # microseconds-in-second used by to_pytimedelta). Returns immediately if already
        # populated; re-checks after computing for thread safety.
        # Side effects: mutates self; runs under a critical section (per-object lock in
        # free-threaded builds).
        # KNOWN ISSUE: _seconds and _microseconds are not declared C fields of
        # _Timedelta; the assignment only works because Timedelta instances (Python
        # subclass) have an instance dictionary.
        """
        compute the components
        """
        if self._is_populated:
            return

        cdef:
            pandas_timedeltastruct tds

        pandas_timedelta_to_timedeltastruct(self._value, self._creso, &tds)
        if self._is_populated:
            return
        self._d = tds.days
        self._h = tds.hrs
        self._m = tds.min
        self._s = tds.sec
        self._ms = tds.ms
        self._us = tds.us
        self._ns = tds.ns
        self._seconds = tds.seconds
        self._microseconds = tds.microseconds

        self._is_populated = 1

    cpdef timedelta to_pytimedelta(_Timedelta self):
        # [FUNCTION DOC]
        # Converts to a datetime.timedelta.
        # Outputs: at ns resolution, microseconds = floor(value / 1000) rounded up when
        # the remainder is >= 500; otherwise built from days, seconds-in-day and
        # microseconds-in-second components.
        # Exceptions: OverflowError if outside datetime.timedelta range.
        # Side effects: may populate the component cache.
        # KNOWN ISSUE: docstring mentions returning an array of timedeltas, but this
        # scalar method only returns a single datetime.timedelta.
        """
        Convert a pandas Timedelta object into a python ``datetime.timedelta`` object.

        Timedelta objects are internally saved as numpy datetime64[ns] dtype.
        Use to_pytimedelta() to convert to object dtype.

        Returns
        -------
        datetime.timedelta or numpy.array of datetime.timedelta

        See Also
        --------
        to_timedelta : Convert argument to Timedelta type.

        Notes
        -----
        Any nanosecond resolution will be lost.

        Examples
        --------
        >>> td = pd.Timedelta('3D')
        >>> td
        Timedelta('3 days 00:00:00')
        >>> td.to_pytimedelta()
        datetime.timedelta(days=3)
        """
        if self._creso == NPY_FR_ns:
            us, remainder = divmod(self._value, 1000)
            if remainder >= 500:
                us += 1
            return timedelta(microseconds=us)

        # TODO(@WillAyd): is this the right way to use components?
        self._ensure_components()
        return timedelta(
            days=self._d, seconds=self._seconds, microseconds=self._microseconds
        )

    def to_timedelta64(self) -> np.timedelta64:
        # [FUNCTION DOC]
        # Returns numpy timedelta64(_value, unit) in the object's own resolution.
        # KNOWN ISSUE: docstring says 'ns' precision; the unit is the Timedelta's own.
        """
        Return a numpy.timedelta64 object with 'ns' precision.

        Since NumPy uses ``timedelta64`` objects for its time operations, converting
        a pandas ``Timedelta`` into a NumPy ``timedelta64`` provides seamless
        integration between the two libraries, especially when working in environments
        that heavily rely on NumPy for array-based calculations.

        See Also
        --------
        to_timedelta : Convert argument to timedelta.
        numpy.timedelta64 : A NumPy object for time duration.
        Timedelta : Represents a duration, the difference between two dates
            or times.

        Examples
        --------
        >>> td = pd.Timedelta('3D')
        >>> td
        Timedelta('3 days 00:00:00')
        >>> td.to_timedelta64()
        numpy.timedelta64(259200000000000,'ns')
        """
        cdef:
            str abbrev = npy_unit_to_abbrev(self._creso)
        # TODO: way to create an np.timedelta64 obj with the reso directly
        #  instead of having to get the abbrev?
        return np.timedelta64(self._value, abbrev)

    def to_numpy(self, dtype=None, copy=False) -> np.timedelta64:
        # [FUNCTION DOC]
        # Returns to_timedelta64().
        # Inputs: dtype, copy - must be None/False.
        # Exceptions: ValueError if dtype is not None or copy is not False.
        # KNOWN ISSUE: the docstring says dtype and copy are ignored, but non-default
        # values raise ValueError.
        """
        Convert the Timedelta to a NumPy timedelta64.

        This is an alias method for `Timedelta.to_timedelta64()`.

        Parameters
        ----------
        dtype : NoneType
            It is available here only for compatibility. Its value will not
            affect the return value.
        copy : bool, default False
            It is available here only for compatibility. Its value will not
            affect the return value.

        Returns
        -------
        numpy.timedelta64

        See Also
        --------
        Series.to_numpy : Similar method for Series.

        Examples
        --------
        >>> td = pd.Timedelta('3D')
        >>> td
        Timedelta('3 days 00:00:00')
        >>> td.to_numpy()
        numpy.timedelta64(259200000000000,'ns')
        """
        if dtype is not None or copy is not False:
            raise ValueError(
                "Timedelta.to_numpy dtype and copy arguments are ignored"
            )
        return self.to_timedelta64()

    def view(self, dtype):
        # [FUNCTION DOC]
        # Reinterprets the integer value as dtype via numpy (timedelta64 with generic
        # unit, then view).
        # Inputs: dtype - numpy dtype spec (e.g. int, "i8").
        # Outputs: numpy scalar of dtype with the raw integer bits of _value.
        # Exceptions: numpy errors for incompatible itemsize.
        # Note: the raw value is in units of _creso, not necessarily nanoseconds.
        """
        Array view compatibility.

        This method allows you to reinterpret the underlying data of a Timedelta
        object as a different dtype. The `view` method provides a way to reinterpret
        the internal representation of the `Timedelta` object without modifying its
        data. This is particularly useful when you need to work with the underlying
        data directly, such as for performance optimizations or interfacing with
        low-level APIs. The returned value is typically the number of nanoseconds
        since the epoch, represented as an integer or another specified dtype.

        Parameters
        ----------
        dtype : str or dtype
            The dtype to view the underlying data as.

        See Also
        --------
        Timedelta.asm8 : Return a numpy timedelta64 array scalar view.
        numpy.ndarray.view : Returns a view of an array with the same data.
        Timedelta.to_numpy : Converts the Timedelta to a NumPy timedelta64.
        Timedelta.total_seconds : Returns the total duration of the Timedelta
            object in seconds.

        Examples
        --------
        >>> td = pd.Timedelta('3D')
        >>> td
        Timedelta('3 days 00:00:00')
        >>> td.view(int)
        259200000000000
        """
        return np.timedelta64(self._value).view(dtype)

    @property
    def components(self):
        # [FUNCTION DOC]
        # Property: Components namedtuple (days, hours, minutes, seconds, milliseconds,
        # microseconds, nanoseconds). Side effects: populates the component cache.
        """
        Return a components namedtuple-like.

        Each component represents a different time unit, allowing you to access the
        breakdown of the total duration in terms of days, hours, minutes, seconds,
        milliseconds, microseconds, and nanoseconds.

        See Also
        --------
        Timedelta.total_seconds : Returns the total duration of the Timedelta in
            seconds.
        to_timedelta : Convert argument to Timedelta.
        Timedelta : Represents a duration, the difference between two dates or times.

        Examples
        --------
        >>> td = pd.Timedelta('2 day 4 min 3 us 42 ns')
        >>> td.components
        Components(days=2, hours=0, minutes=4, seconds=0, milliseconds=0,
            microseconds=3, nanoseconds=42)
        """
        self._ensure_components()
        # return the named tuple
        return Components(self._d, self._h, self._m, self._s,
                          self._ms, self._us, self._ns)

    @property
    def asm8(self) -> np.timedelta64:
        # [FUNCTION DOC]
        # Property: same as to_timedelta64() (numpy timedelta64 at own resolution).
        # KNOWN ISSUE: docstring says the view is in nanoseconds; it is in the
        # Timedelta's own unit.
        """
        Return a numpy timedelta64 array scalar view.

        Provides access to the array scalar view (i.e. a combination of the
        value and the units) associated with the numpy.timedelta64().view(),
        including a 64-bit integer representation of the timedelta in
        nanoseconds (Python int compatible).

        Returns
        -------
        numpy timedelta64 array scalar view
            Array scalar view of the timedelta in nanoseconds.

        See Also
        --------
            Timedelta.total_seconds : Return the total seconds in the duration.
            Timedelta.components : Return a namedtuple of the Timedelta's components.
            Timedelta.to_timedelta64 : Convert the Timedelta to a numpy.timedelta64.

        Examples
        --------
        >>> td = pd.Timedelta('1 days 2 min 3 us 42 ns')
        >>> td.asm8
        numpy.timedelta64(86520000003042,'ns')

        >>> td = pd.Timedelta('2 min 3 s')
        >>> td.asm8
        numpy.timedelta64(123000000000,'ns')

        >>> td = pd.Timedelta('3 ms 5 us')
        >>> td.asm8
        numpy.timedelta64(3005000,'ns')

        >>> td = pd.Timedelta(42, unit='ns')
        >>> td.asm8
        numpy.timedelta64(42,'ns')
        """
        return self.to_timedelta64()

    @property
    def resolution_string(self) -> str:
        # [FUNCTION DOC]
        # Property: string for the finest non-zero component: "ns", "us", "ms", "s",
        # "min", "h", else "D". Side effects: populates the component cache.
        """
        Return a string representing the lowest timedelta resolution.

        Each timedelta has a defined resolution that represents the lowest OR
        most granular level of precision. Each level of resolution is
        represented by a short string as defined below:

        Resolution:     Return value

        * Days:         'D'
        * Hours:        'h'
        * Minutes:      'min'
        * Seconds:      's'
        * Milliseconds: 'ms'
        * Microseconds: 'us'
        * Nanoseconds:  'ns'

        Returns
        -------
        str
            Timedelta resolution.

        Examples
        --------
        >>> td = pd.Timedelta('1 days 2 min 3 us 42 ns')
        >>> td.resolution_string
        'ns'

        >>> td = pd.Timedelta('1 days 2 min 3 us')
        >>> td.resolution_string
        'us'

        >>> td = pd.Timedelta('2 min 3 s')
        >>> td.resolution_string
        's'

        >>> td = pd.Timedelta(36, unit='us')
        >>> td.resolution_string
        'us'
        """
        self._ensure_components()
        if self._ns:
            return "ns"
        elif self._us:
            return "us"
        elif self._ms:
            return "ms"
        elif self._s:
            return "s"
        elif self._m:
            return "min"
        elif self._h:
            return "h"
        else:
            return "D"

    @property
    def nanoseconds(self):
        # [FUNCTION DOC]
        # Property: nanosecond component in [0, 1000). Side effects: populates cache.
        """
        Return the number of nanoseconds (n), where 0 <= n < 1 microsecond.

        Returns
        -------
        int
            Number of nanoseconds.

        See Also
        --------
        Timedelta.components : Return all attributes with assigned values
            (i.e. days, hours, minutes, seconds, milliseconds, microseconds,
            nanoseconds).

        Examples
        --------
        **Using string input**

        >>> td = pd.Timedelta('1 days 2 min 3 us 42 ns')

        >>> td.nanoseconds
        42

        **Using integer input**

        >>> td = pd.Timedelta(42, unit='ns')
        >>> td.nanoseconds
        42
        """
        self._ensure_components()
        return self._ns

    def _repr_base(self, format=None) -> str:
        # [FUNCTION DOC]
        # HELPER: Builds the textual representation.
        # Inputs: format - None, "all", "sub_day" or "long".
        # Outputs: "all": "D days[ +]HH:MM:SS.mmmuuunnn". Otherwise seconds with a
        # fractional part of 6 digits if ms/us non-zero (9 if ns non-zero); "sub_day"
        # with zero days gives "HH:MM:SS[.fff]"; if any sub-day component is non-zero or
        # format is "long", "D days HH:MM:SS[.fff]"; else "D days". The separator is "
        # +" when days are negative.
        # Side effects: populates the component cache.
        """

        Parameters
        ----------
        format : None|all|sub_day|long

        Returns
        -------
        converted : string of a Timedelta

        """
        cdef:
            str sign, fmt
            dict comp_dict
            object subs

        self._ensure_components()

        if self._d < 0:
            sign = " +"
        else:
            sign = " "

        if format == "all":
            fmt = ("{days} days{sign}{hours:02}:{minutes:02}:{seconds:02}."
                   "{milliseconds:03}{microseconds:03}{nanoseconds:03}")
        else:
            # if we have a partial day
            subs = (self._h or self._m or self._s or
                    self._ms or self._us or self._ns)

            if self._ms or self._us or self._ns:
                seconds_fmt = "{seconds:02}.{milliseconds:03}{microseconds:03}"
                if self._ns:
                    # GH#9309
                    seconds_fmt += "{nanoseconds:03}"
            else:
                seconds_fmt = "{seconds:02}"

            if format == "sub_day" and not self._d:
                fmt = "{hours:02}:{minutes:02}:" + seconds_fmt
            elif subs or format == "long":
                fmt = "{days} days{sign}{hours:02}:{minutes:02}:" + seconds_fmt
            else:
                fmt = "{days} days"

        comp_dict = self.components._asdict()
        comp_dict["sign"] = sign

        return fmt.format(**comp_dict)

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns "Timedelta('<long format>')".
        repr_based = self._repr_base(format="long")
        return f"Timedelta('{repr_based}')"

    def __str__(self) -> str:
        # [FUNCTION DOC]
        # Returns the long-format string (e.g. "1 days 00:00:00").
        return self._repr_base(format="long")

    def __bool__(self) -> bool:
        # [FUNCTION DOC]
        # True when _value is non-zero.
        return self._value!= 0

    def isoformat(self) -> str:
        # [FUNCTION DOC]
        # Returns ISO 8601 duration "P{d}DT{h}H{m}M{s}S" with seconds including up to 9
        # fractional digits, trailing zeros and a trailing dot stripped.
        # Note: negative durations use the floor-based components (negative days with
        # positive hours etc.).
        """
        Format the Timedelta as ISO 8601 Duration.

        ``P[n]Y[n]M[n]DT[n]H[n]M[n]S``, where the ``[n]`` s are replaced by the
        values. See Wikipedia:
        `ISO 8601 § Durations <https://en.wikipedia.org/wiki/ISO_8601#Durations>`_.

        Returns
        -------
        str

        See Also
        --------
        Timestamp.isoformat : Function is used to convert the given
            Timestamp object into the ISO format.

        Notes
        -----
        The longest component is days, whose value may be larger than
        365.
        Every component is always included, even if its value is 0.
        Pandas uses nanosecond precision, so up to 9 decimal places may
        be included in the seconds component.
        Trailing 0's are removed from the seconds component after the decimal.
        We do not 0 pad components, so it's `...T5H...`, not `...T05H...`

        Examples
        --------
        >>> td = pd.Timedelta(days=6, minutes=50, seconds=3,
        ...                   milliseconds=10, microseconds=10, nanoseconds=12)

        >>> td.isoformat()
        'P6DT0H50M3.010010012S'
        >>> pd.Timedelta(hours=1, seconds=10).isoformat()
        'P0DT1H0M10S'
        >>> pd.Timedelta(days=500.5).isoformat()
        'P500DT12H0M0S'
        """
        components = self.components
        seconds = (f"{components.seconds}."
                   f"{components.milliseconds:0>3}"
                   f"{components.microseconds:0>3}"
                   f"{components.nanoseconds:0>3}")
        # Trim unnecessary 0s, 1.000000000 -> 1
        seconds = seconds.rstrip("0").rstrip(".")
        tpl = (f"P{components.days}DT{components.hours}"
               f"H{components.minutes}M{seconds}S")
        return tpl

    # ----------------------------------------------------------------
    # Constructors

    @classmethod
    def _from_value_and_reso(cls, int64_t value, NPY_DATETIMEUNIT reso):
        # [FUNCTION DOC]
        # HELPER: Class method wrapper around _timedelta_from_value_and_reso (for tests
        # and Python-level callers). Same inputs/outputs/exceptions.
        # exposing as classmethod for testing
        return _timedelta_from_value_and_reso(cls, value, reso)

    def as_unit(self, str unit, bint round_ok=True):
        # [FUNCTION DOC]
        # Converts to another resolution.
        # Inputs: unit - "s", "ms", "us" or "ns" (any numpy timedelta unit string is
        # accepted by the dtype lookup); round_ok - if false, raise on lossy conversion.
        # Outputs: Timedelta at the new resolution (self if unchanged).
        # Exceptions: TypeError for invalid unit strings; OutOfBoundsTimedelta on
        # overflow; ValueError on lossy conversion when round_ok is false;
        # NotImplementedError for unsupported resolutions.
        """
        Convert the underlying int64 representation to the given unit.

        Parameters
        ----------
        unit : {"ns", "us", "ms", "s"}
        round_ok : bool, default True
            If False and the conversion requires rounding, raise.

        Returns
        -------
        Timedelta

        See Also
        --------
        Timedelta : Represents a duration, the difference between two dates or times.
        to_timedelta : Convert argument to timedelta.
        Timedelta.asm8 : Return a numpy timedelta64 array scalar view.

        Examples
        --------
        >>> td = pd.Timedelta('1001ms')
        >>> td
        Timedelta('0 days 00:00:01.001000')
        >>> td.as_unit('s')
        Timedelta('0 days 00:00:01')
        """
        dtype = np.dtype(f"m8[{unit}]")
        reso = get_unit_from_dtype(dtype)
        return self._as_creso(reso, round_ok=round_ok)

    @cython.cdivision(False)
    cdef _Timedelta _as_creso(self, NPY_DATETIMEUNIT reso, bint round_ok=True):
        # [FUNCTION DOC]
        # HELPER: Converts _value to resolution reso (convert_reso) and builds a new
        # instance of type(self). Returns self if reso matches.
        # Exceptions: OutOfBoundsTimedelta on overflow; ValueError if round_ok is false
        # and the conversion is lossy.
        cdef:
            int64_t value

        if reso == self._creso:
            return self

        try:
            value = convert_reso(self._value, self._creso, reso, round_ok=round_ok)
        except OverflowError as err:
            unit = npy_unit_to_abbrev(reso)
            raise OutOfBoundsTimedelta(
                f"Cannot cast {self} to unit='{unit}' without overflow."
            ) from err

        return type(self)._from_value_and_reso(value, reso=reso)

    cpdef _maybe_cast_to_matching_resos(self, _Timedelta other):
        # [FUNCTION DOC]
        # HELPER: Returns (self, other) where the coarser one is cast to the finer
        # resolution. Exceptions: OutOfBoundsTimedelta on overflow.
        """
        If _resos do not match, cast to the higher resolution, raising on overflow.
        """
        if self._creso > other._creso:
            other = other._as_creso(self._creso)
        elif self._creso < other._creso:
            self = self._as_creso(other._creso)
        return self, other


# Python front end to C extension type _Timedelta
# This serves as the box for timedelta64
@set_module("pandas")
class Timedelta(_Timedelta):
    # [CLASS DOC]
    # Public Python duration scalar (pandas.Timedelta), subclass of _Timedelta. Adds the
    # flexible constructor, pickling, round/floor/ceil and the arithmetic operators (+,
    # -, *, /, //, %, divmod, unary ops). Operations with NaT or NaN generally return
    # NaT or NaN.
    """
    Represents a duration, the difference between two dates or times.

    Timedelta is the pandas equivalent of python's ``datetime.timedelta``
    and is interchangeable with it in most cases.

    Parameters
    ----------
    value : Timedelta, timedelta, np.timedelta64, str, int or float
        Input value.
    unit : str, default 'ns'
        If input is an integer, denote the unit of the input.
        If input is a float, denote the unit of the integer parts.
        The decimal parts with resolution lower than 1 nanosecond are ignored.

        Possible values:

        * 'W', or 'D'
        * 'days', or 'day'
        * 'hours', 'hour', 'hr', or 'h'
        * 'minutes', 'minute', 'min', or 'm'
        * 'seconds', 'second', 'sec', or 's'
        * 'milliseconds', 'millisecond', 'millis', 'milli', or 'ms'
        * 'microseconds', 'microsecond', 'micros', 'micro', or 'us'
        * 'nanoseconds', 'nanosecond', 'nanos', 'nano', or 'ns'.

        .. deprecated:: 3.0.0

            Allowing the values `w`, `d`, `MIN`, `MS`, `US` and `NS` to denote units
            are deprecated in favour of the values `W`, `D`, `min`, `ms`, `us` and `ns`.

    **kwargs
        Available kwargs: {days, seconds, microseconds,
        milliseconds, minutes, hours, weeks}.
        Values for construction in compat with datetime.timedelta.
        Numpy ints and floats will be coerced to python ints and floats.

    See Also
    --------
    Timestamp : Represents a single timestamp in time.
    TimedeltaIndex : Immutable Index of timedelta64 data.
    DateOffset : Standard kind of date increment used for a date range.
    to_timedelta : Convert argument to timedelta.
    datetime.timedelta : Represents a duration in the datetime module.
    numpy.timedelta64 : Represents a duration compatible with NumPy.

    Notes
    -----
    The constructor may take in either both values of value and unit or
    kwargs as above. Either one of them must be used during initialization

    The ``.value`` attribute is always in ns.

    If the precision is higher than nanoseconds, the precision of the duration is
    truncated to nanoseconds.

    Examples
    --------
    Here we initialize Timedelta object with both value and unit

    >>> td = pd.Timedelta(1, "D")
    >>> td
    Timedelta('1 days 00:00:00')

    Here we initialize the Timedelta object with kwargs

    >>> td2 = pd.Timedelta(days=1)
    >>> td2
    Timedelta('1 days 00:00:00')

    We see that either way we get the same result
    """

    _req_any_kwargs_new = {"weeks", "days", "hours", "minutes", "seconds",
                           "milliseconds", "microseconds", "nanoseconds"}

    def __new__(cls, object value=_no_input, unit=None, **kwargs):
        # [FUNCTION DOC]
        # Constructs a Timedelta (or returns NaT).
        # Inputs: value - Timedelta, str, datetime.timedelta, numpy timedelta64, Tick,
        # int, float, or null; unit - unit for numeric value; kwargs - any of weeks,
        # days, hours, minutes, seconds, milliseconds, microseconds, nanoseconds
        # (int/float).
        # Outputs: keyword form: sums all keywords exactly into ns (seconds-scale fields
        # computed via float then int), result at us if divisible by 1000 and no
        # nanoseconds keyword, else ns. Timedelta input returned as-is. Strings parsed
        # (ISO or regular) to us unless ns precision is needed. datetime.timedelta ->
        # us. timedelta64 -> nearest supported resolution (NaT -> NaT). Tick -> its
        # resolution. Nulls -> NaT. Integers with a non-ns unit go through timedelta64
        # to keep a coarse resolution; ns integers stay ns. Integral floats are treated
        # as ints; other floats are scaled to ns. NPY_NAT -> NaT.
        # Exceptions: ValueError for unknown/missing keywords, both value and keywords
        # for a Timedelta input, unit with a string, ambiguous units, unsupported
        # timedelta64 units, unparseable strings or unsupported types; TypeError for
        # non-numeric keyword values; OutOfBoundsTimedelta on overflow.
        # Side effects: UserWarning when unit is given for non-numeric input;
        # Pandas4Warning for deprecated unit spellings.
        unsupported_kwargs = set(kwargs)
        unsupported_kwargs.difference_update(cls._req_any_kwargs_new)
        if unsupported_kwargs or (
            value is _no_input and
            not cls._req_any_kwargs_new.intersection(kwargs)
        ):
            raise ValueError(
                # GH#53801
                "cannot construct a Timedelta from the passed arguments, "
                "allowed keywords are "
                "[weeks, days, hours, minutes, seconds, "
                "milliseconds, microseconds, nanoseconds]"
            )

        if (
            unit is not None
            and not (is_float_object(value) or is_integer_object(value))
        ):
            # GH#53198
            warnings.warn(
                "The 'unit' keyword is only used when the Timedelta input is "
                f"an integer or float, not {type(value).__name__}. "
                "To specify the storage unit of the output use `td.as_unit(unit)`",
                UserWarning,
                stacklevel=find_stack_level(),
            )

        if value is _no_input:
            if not len(kwargs):
                raise ValueError("cannot construct a Timedelta without a "
                                 "value/unit or descriptive keywords "
                                 "(days,seconds....)")

            kwargs = {key: _to_py_int_float(kwargs[key]) for key in kwargs}

            # GH43764, convert any input to nanoseconds first and then
            # create the timedelta. This ensures that any potential
            # nanosecond contributions from kwargs parsed as floats
            # are taken into consideration.
            seconds = int((
                (
                    (kwargs.get("days", 0) + kwargs.get("weeks", 0) * 7) * 24
                    + kwargs.get("hours", 0)
                ) * 3600
                + kwargs.get("minutes", 0) * 60
                + kwargs.get("seconds", 0)
                ) * 1_000_000_000
            )

            ns = kwargs.get("nanoseconds", 0)
            us = kwargs.get("microseconds", 0)
            ms = kwargs.get("milliseconds", 0)
            try:
                value = np.timedelta64(
                    int(ns)
                    + int(us * 1_000)
                    + int(ms * 1_000_000)
                    + seconds, "ns"
                )
            except OverflowError as err:
                # GH#55503
                msg = (
                    f"seconds={seconds}, milliseconds={ms}, "
                    f"microseconds={us}, nanoseconds={ns}"
                )
                raise OutOfBoundsTimedelta(msg) from err

            if (
                "nanoseconds" not in kwargs
                and cnp.get_timedelta64_value(value) % 1000 == 0
            ):
                # If possible, give a microsecond unit
                value = value.astype("m8[us]")

        disallow_ambiguous_unit(unit)

        cdef:
            int64_t new_value

        # GH 30543 if pd.Timedelta already passed, return it
        # check that only value is passed
        if isinstance(value, _Timedelta):
            # 'unit' is benign in this case, but e.g. days or seconds
            #  doesn't make sense here.
            if len(kwargs):
                # GH#48898
                raise ValueError(
                    "Cannot pass both a Timedelta input and timedelta keyword "
                    "arguments, got "
                    f"{list(kwargs.keys())}"
                )
            return value
        elif isinstance(value, str):
            if unit is not None:
                raise ValueError("unit must not be specified if the value is a str")
            if (len(value) > 0 and value[0] == "P") or (
                len(value) > 1 and value[:2] == "-P"
            ):
                ival = parse_iso_format_string(value)
            else:
                ival = parse_timedelta_string(value)

            if not needs_nano_unit(ival, value):
                # If we don't specifically need nanosecond resolution, default
                #  to microsecond like we do for datetimes
                value = np.timedelta64(ival // 1000, "us")
                return cls(value)
            else:
                value = np.timedelta64(ival, "ns")
        elif PyDelta_Check(value):
            # pytimedelta object -> microsecond resolution
            new_value = delta_to_nanoseconds(
                value, reso=NPY_DATETIMEUNIT.NPY_FR_us
            )
            return cls._from_value_and_reso(
                new_value, reso=NPY_DATETIMEUNIT.NPY_FR_us
            )
        elif cnp.is_timedelta64_object(value):
            # Retain the resolution if possible, otherwise cast to the nearest
            #  supported resolution.
            new_value = cnp.get_timedelta64_value(value)
            if new_value == NPY_NAT:
                # i.e. np.timedelta64("NaT")
                return NaT

            reso = get_datetime64_unit(value)
            if not (is_supported_unit(reso) or
                    reso in [NPY_DATETIMEUNIT.NPY_FR_m,
                             NPY_DATETIMEUNIT.NPY_FR_h,
                             NPY_DATETIMEUNIT.NPY_FR_D,
                             NPY_DATETIMEUNIT.NPY_FR_W,
                             NPY_DATETIMEUNIT.NPY_FR_GENERIC]):
                err = npy_unit_to_abbrev(reso)
                raise ValueError(
                    f"Unit {err} is not supported. "
                    "Only unambiguous timedelta values durations are supported. "
                    "Allowed units are 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns'")

            new_reso = get_supported_reso(reso)
            if reso != NPY_DATETIMEUNIT.NPY_FR_GENERIC:
                try:
                    new_value = convert_reso(
                        new_value,
                        reso,
                        new_reso,
                        round_ok=True,
                    )
                except (OverflowError, OutOfBoundsDatetime) as err:
                    raise OutOfBoundsTimedelta(value) from err
            return cls._from_value_and_reso(new_value, reso=new_reso)

        elif is_tick_object(value):
            new_reso = get_supported_reso(value._creso)
            new_value = delta_to_nanoseconds(value, reso=new_reso)
            return cls._from_value_and_reso(new_value, reso=new_reso)

        elif checknull_with_nat_and_na(value):
            return NaT

        elif is_integer_object(value):
            # unit=None is de-facto 'ns'
            if value != NPY_NAT:
                unit = parse_timedelta_unit(unit)
                if unit != "ns":
                    # Return with the closest-to-supported unit by going through
                    #  the timedelta64 path
                    td = np.timedelta64(value, unit)
                    return cls(td)
                value = _numeric_to_td64ns(value, unit)

        elif is_float_object(value):
            int_item = int(value)
            if value == int_item:
                # round float -> treat like a int, try to preserve unit
                return cls(int_item, unit=unit)

            # unit=None is de-facto 'ns'
            unit = parse_timedelta_unit(unit)
            value = _numeric_to_td64ns(value, unit)

        else:
            raise ValueError(
                "Value must be Timedelta, string, integer, "
                f"float, timedelta or convertible, not {type(value).__name__}"
            )

        if cnp.is_timedelta64_object(value):
            value = value.view("i8")

        # nat
        if value == NPY_NAT:
            return NaT

        return _timedelta_from_value_and_reso(cls, value, NPY_FR_ns)

    def __setstate__(self, state):
        # [FUNCTION DOC]
        # Restores state from a pickle: state is (value,) for old pickles (ns) or
        # (value, reso). Side effects: mutates _value and _creso.
        # KNOWN ISSUE: does not reset _is_populated or the datetime.timedelta base part;
        # in practice unused since __reduce__ rebuilds via _timedelta_unpickle.
        if len(state) == 1:
            # older pickle, only supported nanosecond
            value = state[0]
            reso = NPY_FR_ns
        else:
            value, reso = state
        self._value= value
        self._creso = reso

    def __reduce__(self):
        # [FUNCTION DOC]
        # Pickle hook: returns (_timedelta_unpickle, (_value, _creso)).
        object_state = self._value, self._creso
        return (_timedelta_unpickle, object_state)

    @cython.cdivision(True)
    def _round(self, freq, mode):
        # [FUNCTION DOC]
        # HELPER: Rounds the value to a multiple of freq.
        # Inputs: freq - frequency string/offset (fixed duration; Day = 24h); mode -
        # RoundTo rounding mode.
        # Outputs: Timedelta at the same resolution; self if freq is finer than the
        # resolution (integer unit count is 0).
        # Exceptions: ValueError for non-fixed freq or a zero-length freq;
        # OutOfBoundsTimedelta on overflow.
        cdef:
            int64_t result, nanos
            ndarray[int64_t] arr
        freq_arg = freq
        freq = to_offset(freq, is_period=False)
        nanos = get_unit_for_round(freq, self._creso)
        if nanos == 0:
            if freq.nanos == 0:
                raise ValueError("Division by zero in rounding")

            # e.g. self.unit == "s" and sub-second freq
            return self

        arr = np.array([self._value], dtype="i8")
        try:
            result = round_nsint64(arr, mode, nanos)[0]
        except OverflowError as err:
            raise OutOfBoundsTimedelta(
                f"Cannot round {self} to freq={freq_arg} without overflow"
            ) from err
        return Timedelta._from_value_and_reso(result, self._creso)

    def round(self, freq):
        # [FUNCTION DOC]
        # Rounds to the nearest multiple of freq (ties to even). See _round.
        """
        Round the Timedelta to the specified resolution.

        Parameters
        ----------
        freq : str
            Frequency string indicating the rounding resolution.
            It uses the same units as class constructor :class:`~pandas.Timedelta`.

        Returns
        -------
        a new Timedelta rounded to the given resolution of `freq`

        Raises
        ------
        ValueError if the freq cannot be converted

        See Also
        --------
            Timedelta.floor : Floor the Timedelta to the specified resolution.
            Timedelta.round : Round the Timedelta to the nearest specified resolution.
            Timestamp.ceil : Similar method for Timestamp objects.

        Examples
        --------
        >>> td = pd.Timedelta('1001ms')
        >>> td
        Timedelta('0 days 00:00:01.001000')
        >>> td.round('s')
        Timedelta('0 days 00:00:01')
        """
        return self._round(freq, RoundTo.NEAREST_HALF_EVEN)

    def floor(self, freq):
        # [FUNCTION DOC]
        # Rounds down (toward negative infinity) to freq. See _round.
        """
        Return a new Timedelta floored to this resolution.

        Parameters
        ----------
        freq : str
            Frequency string indicating the flooring resolution.
            It uses the same units as class constructor :class:`~pandas.Timedelta`.

        Returns
        -------
        Timedelta
            A new Timedelta object floored to the specified resolution.

        See Also
        --------
            Timestamp.ceil : Round the Timestamp up to the nearest specified resolution.
            Timestamp.round : Round the Timestamp to the nearest specified resolution.

        Examples
        --------
        >>> td = pd.Timedelta('1001ms')
        >>> td
        Timedelta('0 days 00:00:01.001000')
        >>> td.floor('s')
        Timedelta('0 days 00:00:01')
        """
        return self._round(freq, RoundTo.MINUS_INFTY)

    def ceil(self, freq):
        # [FUNCTION DOC]
        # Rounds up (toward positive infinity) to freq. See _round.
        """
        Return a new Timedelta ceiled to this resolution.

        Parameters
        ----------
        freq : str
            Frequency string indicating the ceiling resolution. Must be a fixed
            frequency like 's' (second) not 'ME' (month end). See
            :ref:`frequency aliases <timeseries.offset_aliases>` for
            a list of possible `freq` values.

        Returns
        -------
        Timedelta
            A new Timedelta object ceiled to the specified resolution.

        See Also
        --------
            Timedelta.floor : Floor the Timedelta to the specified resolution.
            Timedelta.round : Round the Timedelta to the nearest specified resolution.

        Examples
        --------
        >>> td = pd.Timedelta('1001ms')
        >>> td
        Timedelta('0 days 00:00:01.001000')
        >>> td.ceil('s')
        Timedelta('0 days 00:00:02')
        """
        return self._round(freq, RoundTo.PLUS_INFTY)

    # ----------------------------------------------------------------
    # Arithmetic Methods
    # TODO: Can some of these be defined in the cython class?

    __neg__ = _op_unary_method(lambda x: -x, "__neg__")
    __pos__ = _op_unary_method(lambda x: x, "__pos__")
    __abs__ = _op_unary_method(lambda x: abs(x), "__abs__")

    __add__ = _binary_op_method_timedeltalike(lambda x, y: x + y, "__add__")
    __radd__ = _binary_op_method_timedeltalike(lambda x, y: x + y, "__radd__")
    __sub__ = _binary_op_method_timedeltalike(lambda x, y: x - y, "__sub__")
    __rsub__ = _binary_op_method_timedeltalike(lambda x, y: y - x, "__rsub__")

    def __mul__(self, other):
        # [FUNCTION DOC]
        # Timedelta * other (also used for reflected multiplication).
        # Outputs: for int/float: NaT if NaN, else Timedelta with value truncated
        # int64(other * value) at same resolution; 0-dim arrays recurse on the scalar;
        # other arrays: other * self as timedelta64 (numpy semantics); NotImplemented
        # otherwise.
        # Exceptions: TypeError for bool; OverflowError if the product exceeds int64.
        if is_integer_object(other) or is_float_object(other):
            if util.is_nan(other):
                # np.nan * timedelta -> np.timedelta64("NaT"), in this case NaT
                return NaT

            return _timedelta_from_value_and_reso(
                Timedelta,
                <int64_t>(other * self._value),
                reso=self._creso,
            )

        elif is_array(other):
            if other.ndim == 0:
                # see also: item_from_zerodim
                item = cnp.PyArray_ToScalar(cnp.PyArray_DATA(other), other)
                return self.__mul__(item)
            return other * self.to_timedelta64()

        elif is_bool_object(other):
            # GH#62316
            raise TypeError(
                "Cannot multiply Timedelta by bool. "
                "Explicitly cast to integer instead."
            )

        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other):
        # [FUNCTION DOC]
        # Timedelta / other.
        # Outputs: timedelta-like, None, NaT or str (str deprecated): float ratio of
        # values after matching resolutions, NaN for NaT/None. int/float: NaT for NaN,
        # else Timedelta of truncated value / other (numpy scalars converted to Python
        # scalars). Arrays: 0-dim recurse, else numpy division of self as timedelta64.
        # NotImplemented otherwise.
        # Exceptions: ZeroDivisionError for zero divisors; OutOfBoundsTimedelta.
        # Side effects: Pandas4Warning for string operands.
        if _should_cast_to_timedelta(other):
            # We interpret NaT as timedelta64("NaT")
            other = _wrapped_to_timedelta(other)
            if other is NaT:
                return np.nan
            if other._creso != self._creso:
                self, other = self._maybe_cast_to_matching_resos(other)
            return self._value/ float(other._value)

        elif is_integer_object(other) or is_float_object(other):
            # integers or floats
            if util.is_nan(other):
                return NaT
            # We want NumPy numeric scalars to behave like Python scalars
            # post NEP 50
            if isinstance(other, cnp.integer):
                other = int(other)
            if isinstance(other, cnp.floating):
                other = float(other)
            return Timedelta._from_value_and_reso(
                <int64_t>(self._value/ other), self._creso
            )

        elif is_array(other):
            if other.ndim == 0:
                # see also: item_from_zerodim
                item = cnp.PyArray_ToScalar(cnp.PyArray_DATA(other), other)
                return self.__truediv__(item)
            return self.to_timedelta64() / other

        return NotImplemented

    def __rtruediv__(self, other):
        # [FUNCTION DOC]
        # other / Timedelta.
        # Outputs: timedelta-like/None/NaT/str: float other / self (NaN for NaT/None);
        # arrays: 0-dim recurse, object arrays element-wise, else numpy division by self
        # as timedelta64; NotImplemented otherwise (numbers / Timedelta unsupported).
        # Exceptions: ZeroDivisionError if self is zero. Side effects: Pandas4Warning
        # for strings.
        if _should_cast_to_timedelta(other):
            # We interpret NaT as timedelta64("NaT")
            other = _wrapped_to_timedelta(other)
            if other is NaT:
                return np.nan
            if self._creso != other._creso:
                self, other = self._maybe_cast_to_matching_resos(other)
            return float(other._value) / self._value

        elif is_array(other):
            if other.ndim == 0:
                # see also: item_from_zerodim
                item = cnp.PyArray_ToScalar(cnp.PyArray_DATA(other), other)
                return self.__rtruediv__(item)
            elif other.dtype.kind == "O":
                # GH#31869
                return np.array([x / self for x in other])

            # TODO: if other.dtype.kind == "m" and other.dtype != self.asm8.dtype
            #  then should disallow for consistency with scalar behavior; requires
            #  deprecation cycle. (or changing scalar behavior)
            return other / self.to_timedelta64()

        return NotImplemented

    def __floordiv__(self, other):
        # [FUNCTION DOC]
        # Timedelta // other.
        # Outputs: timedelta-like/None/NaT/str: integer floor of value ratio (NaN for
        # NaT). int/float: NaT for NaN, else Timedelta of value // other at same
        # resolution. Arrays: 0-dim recurse; timedelta64 arrays -> numpy floor division
        # with NaT positions set to NaN (float result); numeric arrays -> timedelta64
        # array; others raise. NotImplemented otherwise.
        # Exceptions: TypeError for unsupported array dtypes; ZeroDivisionError.
        # Side effects: Pandas4Warning for string operands; suppresses numpy "invalid
        # value" RuntimeWarning.
        # numpy does not implement floordiv for timedelta64 dtype, so we cannot
        # just defer
        if _should_cast_to_timedelta(other):
            # We interpret NaT as timedelta64("NaT")
            other = _wrapped_to_timedelta(other)
            if other is NaT:
                return np.nan
            if self._creso != other._creso:
                self, other = self._maybe_cast_to_matching_resos(other)
            return self._value// other._value

        elif is_integer_object(other) or is_float_object(other):
            if util.is_nan(other):
                return NaT
            # We want NumPy numeric scalars to behave like Python scalars
            # post NEP 50
            if isinstance(other, cnp.integer):
                other = int(other)
            if isinstance(other, cnp.floating):
                other = float(other)
            return type(self)._from_value_and_reso(self._value// other, self._creso)

        elif is_array(other):
            if other.ndim == 0:
                # see also: item_from_zerodim
                item = cnp.PyArray_ToScalar(cnp.PyArray_DATA(other), other)
                return self.__floordiv__(item)

            if other.dtype.kind == "m":
                # also timedelta-like
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        "invalid value encountered in floor_divide",
                        RuntimeWarning
                    )
                    result = self.asm8 // other
                mask = other.view("i8") == NPY_NAT
                if mask.any():
                    # We differ from numpy here
                    result = result.astype("f8")
                    result[mask] = np.nan
                return result

            elif other.dtype.kind in "iuf":
                if other.ndim == 0:
                    return self // other.item()
                else:
                    return self.to_timedelta64() // other

            raise TypeError(f"Invalid dtype {other.dtype} for __floordiv__")

        return NotImplemented

    def __rfloordiv__(self, other):
        # [FUNCTION DOC]
        # other // Timedelta.
        # Outputs: timedelta-like/None/NaT/str: integer floor ratio (NaN for NaT);
        # arrays: 0-dim recurse, timedelta64 arrays -> floor division with NaT -> NaN;
        # other arrays raise. NotImplemented otherwise.
        # Exceptions: TypeError for other array dtypes (message says __floordiv__).
        # Side effects: Pandas4Warning for strings; suppresses numpy RuntimeWarning.
        # numpy does not implement floordiv for timedelta64 dtype, so we cannot
        # just defer
        if _should_cast_to_timedelta(other):
            # We interpret NaT as timedelta64("NaT")
            other = _wrapped_to_timedelta(other)
            if other is NaT:
                return np.nan
            if self._creso != other._creso:
                self, other = self._maybe_cast_to_matching_resos(other)
            return other._value// self._value

        elif is_array(other):
            if other.ndim == 0:
                # see also: item_from_zerodim
                item = cnp.PyArray_ToScalar(cnp.PyArray_DATA(other), other)
                return self.__rfloordiv__(item)

            if other.dtype.kind == "m":
                # also timedelta-like
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        "invalid value encountered in floor_divide",
                        RuntimeWarning
                    )
                    result = other // self.asm8
                mask = other.view("i8") == NPY_NAT
                if mask.any():
                    # We differ from numpy here
                    result = result.astype("f8")
                    result[mask] = np.nan
                return result

            # Includes integer array // Timedelta, disallowed in GH#19761
            raise TypeError(f"Invalid dtype {other.dtype} for __floordiv__")

        return NotImplemented

    def __mod__(self, other):
        # [FUNCTION DOC]
        # Timedelta % other: second element of divmod(self, other).
        # Naive implementation, room for optimization
        return self.__divmod__(other)[1]

    def __rmod__(self, other):
        # [FUNCTION DOC]
        # other % Timedelta: second element of divmod(other, self).
        # Naive implementation, room for optimization
        return self.__rdivmod__(other)[1]

    def __divmod__(self, other):
        # [FUNCTION DOC]
        # Returns (self // other, self - (self // other) * other). Errors propagate from
        # the underlying operators.
        # Naive implementation, room for optimization
        div = self // other
        return div, self - div * other

    def __rdivmod__(self, other):
        # [FUNCTION DOC]
        # Returns (other // self, other - (other // self) * self).
        # Naive implementation, room for optimization
        div = other // self
        return div, other - div * self


def truediv_object_array(ndarray left, ndarray right):
    # [FUNCTION DOC]
    # Element-wise true division of an object array of timedelta64 scalars by an object
    # array of operands (used by TimedeltaArray ops).
    # Inputs: left - 1-D object ndarray of numpy timedelta64; right - same-length object
    # ndarray.
    # Outputs: new object ndarray: where left is NaT, NaN if right is timedelta-like/
    # null/str (string still parsed, may warn/raise), else numpy td64 / right; else
    # Timedelta(left) / right.
    # Exceptions: propagated from division/parsing.
    # Side effects: allocates result; may emit Pandas4Warning.
    cdef:
        ndarray[object] result = np.empty((<object>left).shape, dtype=object)
        object td64  # really timedelta64 if we find a way to declare that
        object obj, res_value
        _Timedelta td
        Py_ssize_t i

    for i in range(len(left)):
        td64 = left[i]
        obj = right[i]

        if cnp.get_timedelta64_value(td64) == NPY_NAT:
            # td here should be interpreted as a td64 NaT
            if _should_cast_to_timedelta(obj):
                _wrapped_to_timedelta(obj)  # deprecate if allowing string
                res_value = np.nan
            else:
                # if its a number then let numpy handle division, otherwise
                #  numpy will raise
                res_value = td64 / obj
        else:
            td = Timedelta(td64)
            res_value = td / obj

        result[i] = res_value

    return result


def floordiv_object_array(ndarray left, ndarray right):
    # [FUNCTION DOC]
    # Element-wise floor division analogous to truediv_object_array, using // in place
    # of /. Same inputs, outputs, exceptions and side effects.
    cdef:
        ndarray[object] result = np.empty((<object>left).shape, dtype=object)
        object td64  # really timedelta64 if we find a way to declare that
        object obj, res_value
        _Timedelta td
        Py_ssize_t i

    for i in range(len(left)):
        td64 = left[i]
        obj = right[i]

        if cnp.get_timedelta64_value(td64) == NPY_NAT:
            # td here should be interpreted as a td64 NaT
            if _should_cast_to_timedelta(obj):
                _wrapped_to_timedelta(obj)  # deprecate allowing string
                res_value = np.nan
            else:
                # if its a number then let numpy handle division, otherwise
                #  numpy will raise
                res_value = td64 // obj
        else:
            td = Timedelta(td64)
            res_value = td // obj

        result[i] = res_value

    return result


cdef bint is_any_td_scalar(object obj):
    # [FUNCTION DOC]
    # True if obj is a datetime.timedelta (incl. Timedelta), numpy timedelta64 or a Tick
    # offset. No side effects.
    """
    Cython equivalent for `isinstance(obj, (timedelta, np.timedelta64, Tick))`

    Parameters
    ----------
    obj : object

    Returns
    -------
    bool
    """
    return (
        PyDelta_Check(obj) or cnp.is_timedelta64_object(obj) or is_tick_object(obj)
    )


cdef inline _wrapped_to_timedelta(object other):
    # [FUNCTION DOC]
    # HELPER: Converts other to a Timedelta (or NaT) via the constructor.
    # Exceptions: as Timedelta construction.
    # Side effects: emits Pandas4Warning when other is a str (deprecated).
    # Helper for deprecating cases where we cast str to Timedelta
    td = Timedelta(other)
    if isinstance(other, str):
        from pandas.errors import Pandas4Warning
        warnings.warn(
            # GH#59653
            "Scalar operations between Timedelta and string are "
            "deprecated and will raise in a future version. "
            "Explicitly cast to Timedelta first.",
            Pandas4Warning,
            stacklevel=find_stack_level(),
        )
        # When this is enforced, remove str from _should_cast_to_timedelta
    return td


cdef bint _should_cast_to_timedelta(object obj):
    # [FUNCTION DOC]
    # HELPER: True if a binary-op operand should be converted to Timedelta: timedelta-
    # like scalar, None, NaT or str. No side effects.
    """
    Should we treat this object as a Timedelta for the purpose of a binary op
    """
    return (
        is_any_td_scalar(obj) or obj is None or obj is NaT or isinstance(obj, str)
    )


cpdef int64_t get_unit_for_round(freq, NPY_DATETIMEUNIT creso) except? -1:
    # [FUNCTION DOC]
    # Converts a rounding frequency to an integer number of creso units.
    # Inputs: freq - frequency string or offset; creso - resolution.
    # Outputs: int64 unit count (0 if freq is finer than creso). Day is treated as
    # exactly 24 hours.
    # Exceptions: ValueError for non-fixed frequencies (accessing .nanos) or invalid
    # freq strings; OutOfBoundsTimedelta on overflow.
    from pandas._libs.tslibs.offsets import to_offset

    freq = to_offset(freq)
    if isinstance(freq, Day):
        # In the "round" context, Day unambiguously means 24h, not calendar-day
        freq = Timedelta(days=freq.n)
    else:
        freq.nanos  # raises on non-fixed freq
    return delta_to_nanoseconds(freq, creso)
