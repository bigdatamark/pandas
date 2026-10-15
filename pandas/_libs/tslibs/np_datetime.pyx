# [MODULE DOC]
# Low-level bridge between numpy datetime64/timedelta64 storage (64-bit integer counts
#   of a unit since 1970-01-01, with the minimum int64 value meaning NaT) and broken-
#   down calendar structs (npy_datetimestruct: year, month, day, hour, min, sec, us, ps,
#   as).
# Wraps C helpers from pandas/datetime/pd_datetime.h (struct<->int conversions, struct
#   comparison, per-unit min/max bound structs, the ISO-8601 parser).
# Provides: unit inspection of numpy scalars/dtypes, struct comparisons, bounds checks
#   raising OutOfBoundsDatetime, conversion of Python date/datetime to structs and
#   int64, string parsing to structs, overflow-safe unit casting (scalar and array),
#   overflow-safe array addition, and the public
#   OutOfBoundsDatetime/OutOfBoundsTimedelta exceptions.
# Units are NPY_DATETIMEUNIT enum values ordered coarse to fine (Y, M, W, D, B, h, m,
#   s, ms, us, ns, ps, fs, as, then GENERIC meaning unitless).
# Module import side effects: initialises the CPython datetime C-API, the pandas
#   datetime C-API capsule and the numpy C-API.
cimport cython
from cpython.datetime cimport (
    PyDateTime_CheckExact,
    PyDateTime_DATE_GET_HOUR,
    PyDateTime_DATE_GET_MICROSECOND,
    PyDateTime_DATE_GET_MINUTE,
    PyDateTime_DATE_GET_SECOND,
    PyDateTime_GET_DAY,
    PyDateTime_GET_MONTH,
    PyDateTime_GET_YEAR,
    import_datetime,
)
from cpython.object cimport (
    Py_EQ,
    Py_GE,
    Py_GT,
    Py_LE,
    Py_LT,
    Py_NE,
)
from cpython.unicode cimport PyUnicode_AsUTF8AndSize
from libc.stdint cimport INT64_MAX

import_datetime()
PandasDateTime_IMPORT

import operator

import numpy as np

cimport numpy as cnp

cnp.import_array()
from numpy cimport (
    PyArray_DatetimeMetaData,
    PyDatetimeScalarObject,
    int64_t,
    ndarray,
    uint8_t,
)

from pandas._libs.tslibs.dtypes cimport (
    get_supported_reso,
    is_supported_unit,
    npy_unit_to_abbrev,
    npy_unit_to_attrname,
)


cdef extern from "pandas/datetime/pd_datetime.h":
    int cmp_npy_datetimestruct(npy_datetimestruct *a,
                               npy_datetimestruct *b)

    # AS, FS, PS versions exist but are not imported because they are not used.
    npy_datetimestruct _NS_MIN_DTS, _NS_MAX_DTS
    npy_datetimestruct _US_MIN_DTS, _US_MAX_DTS
    npy_datetimestruct _MS_MIN_DTS, _MS_MAX_DTS
    npy_datetimestruct _S_MIN_DTS, _S_MAX_DTS
    npy_datetimestruct _M_MIN_DTS, _M_MAX_DTS

    PyArray_DatetimeMetaData get_datetime_metadata_from_dtype(cnp.PyArray_Descr *dtype)

    int parse_iso_8601_datetime(const char *str, int len, int want_exc,
                                npy_datetimestruct *out,
                                NPY_DATETIMEUNIT *out_bestunit,
                                int *out_local, int *out_tzoffset,
                                const char *format, int format_len,
                                FormatRequirement exact)

# ----------------------------------------------------------------------
# numpy object inspection


cdef NPY_DATETIMEUNIT get_datetime64_unit(object obj) noexcept nogil:
    # [FUNCTION DOC]
    # Return the time unit of a numpy datetime64 or timedelta64 scalar by reading the
    #   unit field of the scalar object's metadata directly.
    # Inputs: obj - must be a numpy datetime64/timedelta64 scalar (not checked; any
    #   other object leads to reading invalid memory).
    # Outputs: NPY_DATETIMEUNIT enum value.
    # Exceptions: none. Side effects: none; callable without the GIL.
    """
    returns the unit part of the dtype for a numpy datetime64 object.
    """
    return <NPY_DATETIMEUNIT>(<PyDatetimeScalarObject*>obj).obmeta.base


cdef NPY_DATETIMEUNIT get_unit_from_dtype(cnp.dtype dtype):
    # [FUNCTION DOC]
    # Return the time unit of a datetime64/timedelta64 numpy dtype via the C helper
    #   that reads the dtype's datetime metadata.
    # Inputs: dtype - numpy dtype; caller must ensure it is M8 or m8 (otherwise may
    #   crash).
    # Outputs: NPY_DATETIMEUNIT (NPY_FR_GENERIC for unitless dtypes).
    # Exceptions: none. Side effects: none.
    # NB: caller is responsible for ensuring this is *some* datetime64 or
    #  timedelta64 dtype, otherwise we can segfault
    cdef:
        cnp.PyArray_Descr* descr = <cnp.PyArray_Descr*>dtype
        PyArray_DatetimeMetaData meta
    meta = get_datetime_metadata_from_dtype(descr)
    return meta.base


def py_get_unit_from_dtype(dtype):
    # [FUNCTION DOC]
    # Python-visible wrapper around get_unit_from_dtype, used for testing.
    # Inputs: dtype - datetime64/timedelta64 numpy dtype. Outputs: the unit as an
    #   integer.
    # Exceptions: none (non-datetime dtypes are unsafe). Side effects: none.
    # for testing get_unit_from_dtype; adds 896 bytes to the .so file.
    return get_unit_from_dtype(dtype)


def get_supported_dtype(dtype: cnp.dtype) -> cnp.dtype:
    # [FUNCTION DOC]
    # Map a datetime64/timedelta64 dtype to the nearest pandas-supported resolution
    #   (s, ms, us or ns) of the same kind, e.g. M8[D] -> M8[s], m8[ps] -> m8[ns].
    # Inputs: dtype - M8 or m8 numpy dtype.
    # Outputs: new numpy dtype "M8[unit]" if the input type number is datetime,
    #   otherwise "m8[unit]" where unit is the supported reso's abbreviation.
    # Exceptions: none explicitly. Side effects: none.
    reso = get_unit_from_dtype(dtype)
    new_reso = get_supported_reso(reso)
    new_unit = npy_unit_to_abbrev(new_reso)

    # Accessing dtype.kind here incorrectly(?) gives "" instead of "m"/"M",
    #  so we check type_num instead
    if dtype.type_num == cnp.NPY_DATETIME:
        new_dtype = np.dtype(f"M8[{new_unit}]")
    else:
        new_dtype = np.dtype(f"m8[{new_unit}]")
    return new_dtype


def is_supported_dtype(dtype: cnp.dtype) -> bool:
    # [FUNCTION DOC]
    # Report whether a datetime64/timedelta64 dtype has a resolution pandas supports
    #   natively (s, ms, us, ns).
    # Inputs: dtype - numpy dtype. Outputs: bool.
    # Exceptions: ValueError if dtype is neither datetime64 nor timedelta64.
    # Side effects: none.
    # KNOWN ISSUE: the ValueError message says "is_unitless dtype must be ..." (copied
    #   from is_unitless) rather than naming is_supported_dtype.
    if dtype.type_num not in [cnp.NPY_DATETIME, cnp.NPY_TIMEDELTA]:
        raise ValueError("is_unitless dtype must be datetime64 or timedelta64")
    cdef:
        NPY_DATETIMEUNIT unit = get_unit_from_dtype(dtype)
    return is_supported_unit(unit)


def is_unitless(dtype: cnp.dtype) -> bool:
    # [FUNCTION DOC]
    # Report whether a datetime64/timedelta64 dtype has no unit (generic M8/m8).
    # Inputs: dtype - numpy dtype. Outputs: True if the unit is NPY_FR_GENERIC.
    # Exceptions: ValueError if dtype is neither datetime64 nor timedelta64.
    # Side effects: none.
    """
    Check if a datetime64 or timedelta64 dtype has no attached unit.
    """
    if dtype.type_num not in [cnp.NPY_DATETIME, cnp.NPY_TIMEDELTA]:
        raise ValueError("is_unitless dtype must be datetime64 or timedelta64")
    cdef:
        NPY_DATETIMEUNIT unit = get_unit_from_dtype(dtype)

    return unit == NPY_DATETIMEUNIT.NPY_FR_GENERIC


# ----------------------------------------------------------------------
# Comparison


cdef bint cmp_dtstructs(
    npy_datetimestruct* left, npy_datetimestruct* right, int op
):
    # [FUNCTION DOC]
    # Compare two datetime structs with a rich-comparison opcode.
    # Inputs: left, right - pointers to npy_datetimestruct; op - Python rich
    #   comparison code (EQ, NE, GT, LT, GE, LE).
    # Outputs: boolean result. The C comparator returns -1/0/1 (lexicographic over the
    #   fields); any op other than EQ/NE/GT/LT/GE is treated as LE.
    # Exceptions: none. Side effects: none.
    cdef:
        int cmp_res

    cmp_res = cmp_npy_datetimestruct(left, right)
    if op == Py_EQ:
        return cmp_res == 0
    if op == Py_NE:
        return cmp_res != 0
    if op == Py_GT:
        return cmp_res == 1
    if op == Py_LT:
        return cmp_res == -1
    if op == Py_GE:
        return cmp_res == 1 or cmp_res == 0
    else:
        # i.e. op == Py_LE
        return cmp_res == -1 or cmp_res == 0


cdef bint cmp_scalar(int64_t lhs, int64_t rhs, int op) except -1:
    # [FUNCTION DOC]
    # Fast rich comparison of two int64 values using a Python comparison opcode.
    # Inputs: lhs, rhs - int64; op - one of EQ, NE, LT, LE, GT, GE.
    # Outputs: boolean result; an unrecognised opcode falls through and returns false
    #   (0).
    # Exceptions: none raised (signature allows propagation). Side effects: none.
    """
    cmp_scalar is a more performant version of PyObject_RichCompare
    typed for int64_t arguments.
    """
    if op == Py_EQ:
        return lhs == rhs
    elif op == Py_NE:
        return lhs != rhs
    elif op == Py_LT:
        return lhs < rhs
    elif op == Py_LE:
        return lhs <= rhs
    elif op == Py_GT:
        return lhs > rhs
    elif op == Py_GE:
        return lhs >= rhs


class OutOfBoundsDatetime(ValueError):
    # [CLASS DOC]
    # Public exception (exposed as pandas.errors.OutOfBoundsDatetime), subclass of
    #   ValueError, raised when a datetime value lies outside the range representable as
    #   an int64 count of the target unit. No extra state or behavior.
    """
    Raised when the datetime is outside the range that can be represented.

    This error occurs when attempting to convert or parse a datetime value
    that exceeds the bounds supported by pandas' internal datetime
    representation.

    See Also
    --------
    to_datetime : Convert argument to datetime.
    Timestamp : Pandas replacement for python ``datetime.datetime`` object.

    Examples
    --------
    >>> pd.to_datetime("08335394550")
    Traceback (most recent call last):
    OutOfBoundsDatetime: Parsing "08335394550" to datetime overflows,
    at position 0
    """
    __module__ = "pandas.errors"
    pass


class OutOfBoundsTimedelta(ValueError):
    # [CLASS DOC]
    # Public exception (exposed as pandas.errors.OutOfBoundsTimedelta), subclass of
    #   ValueError, raised when a timedelta cannot be represented as an int64 count of
    #   the target unit. No extra state or behavior.
    """
    Raised when encountering a timedelta value that cannot be represented.

    Representation should be within a timedelta64[ns].

    See Also
    --------
    date_range : Return a fixed frequency DatetimeIndex.

    Examples
    --------
    >>> pd.date_range(start="1/1/1700", freq="B", periods=100000, unit="ns")
    Traceback (most recent call last):
    OutOfBoundsTimedelta: Cannot cast 139999 days 00:00:00
    to unit='ns' without overflow.
    """
    __module__ = "pandas.errors"
    # Timedelta analogue to OutOfBoundsDatetime
    pass


cdef get_implementation_bounds(
    NPY_DATETIMEUNIT reso,
    npy_datetimestruct *lower,
    npy_datetimestruct *upper,
):
    # [FUNCTION DOC]
    # Fill lower/upper with the minimum and maximum datetime structs representable as
    #   int64 in the given resolution, copied from C globals.
    # Inputs: reso - one of ns, us, ms, s, m (minute); lower, upper - output struct
    #   pointers.
    # Outputs: none (returns None). Exceptions: NotImplementedError(reso) for any
    #   other unit.
    # Side effects: writes into *lower and *upper.
    if reso == NPY_FR_ns:
        upper[0] = _NS_MAX_DTS
        lower[0] = _NS_MIN_DTS
    elif reso == NPY_FR_us:
        upper[0] = _US_MAX_DTS
        lower[0] = _US_MIN_DTS
    elif reso == NPY_FR_ms:
        upper[0] = _MS_MAX_DTS
        lower[0] = _MS_MIN_DTS
    elif reso == NPY_FR_s:
        upper[0] = _S_MAX_DTS
        lower[0] = _S_MIN_DTS
    elif reso == NPY_FR_m:
        upper[0] = _M_MAX_DTS
        lower[0] = _M_MIN_DTS
    else:
        raise NotImplementedError(reso)


cdef str dts_to_iso_string(npy_datetimestruct *dts):
    # [FUNCTION DOC]
    # Format a datetime struct as "Y-MM-DD HH:MM:SS" (year unpadded, sub-second fields
    #   dropped), mainly for error messages.
    # Inputs: dts - struct pointer. Outputs: str. Exceptions: none. Side effects:
    #   none.
    return (f"{dts.year}-{dts.month:02d}-{dts.day:02d} "
            f"{dts.hour:02d}:{dts.min:02d}:{dts.sec:02d}")


cdef check_dts_bounds(npy_datetimestruct *dts, NPY_DATETIMEUNIT unit=NPY_FR_ns):
    # [FUNCTION DOC]
    # Validate that a datetime struct lies within the int64-representable range of
    #   unit, comparing it against the per-unit lower/upper bound structs.
    # Inputs: dts - struct pointer; unit - resolution (default ns; must be
    #   ns/us/ms/s/m).
    # Outputs: None.
    # Exceptions: OutOfBoundsDatetime("Out of bounds <attrname> timestamp: <iso>") if
    #   below the lower or above the upper bound; NotImplementedError for unsupported
    #   units.
    # Side effects: none.
    # KNOWN ISSUE: the docstring says the check is against nanosecond resolution, but
    #   it uses the unit argument (ns only by default).
    """Raises OutOfBoundsDatetime if the given date is outside the range that
    can be represented by nanosecond-resolution 64-bit integers."""
    cdef:
        bint error = False
        npy_datetimestruct cmp_upper, cmp_lower

    get_implementation_bounds(unit, &cmp_lower, &cmp_upper)

    if cmp_npy_datetimestruct(dts, &cmp_lower) == -1:
        error = True
    elif cmp_npy_datetimestruct(dts, &cmp_upper) == 1:
        error = True

    if error:
        fmt = dts_to_iso_string(dts)
        attrname = npy_unit_to_attrname[unit]
        raise OutOfBoundsDatetime(f"Out of bounds {attrname} timestamp: {fmt}")


# ----------------------------------------------------------------------
# Conversion


# just exposed for testing at the moment
def py_td64_to_tdstruct(int64_t td64, NPY_DATETIMEUNIT unit):
    # [FUNCTION DOC]
    # Testing helper: break an int64 timedelta of the given unit into a timedelta
    #   struct (days, hrs, min, sec, ms, us, ns, seconds, microseconds, nanoseconds).
    # Inputs: td64 - int64 count; unit - NPY_DATETIMEUNIT. Outputs: dict of struct
    #   fields.
    # Exceptions: none. Side effects: none.
    cdef:
        pandas_timedeltastruct tds
    pandas_timedelta_to_timedeltastruct(td64, unit, &tds)
    return tds  # <- returned as a dict to python


cdef void pydatetime_to_dtstruct(datetime dt, npy_datetimestruct *dts) noexcept:
    # [FUNCTION DOC]
    # Fill a datetime struct from a Python datetime (wall-clock fields; tzinfo
    #   ignored). For exact datetime instances the year is read from the C struct; for
    #   subclasses (e.g. Timestamp, which overrides year) the year attribute is used.
    #   month, day, hour, minute, second, microsecond come from the C accessors; ps and
    #   as are zeroed.
    # Inputs: dt - datetime; dts - output struct pointer. Outputs: none.
    # Exceptions: none (noexcept). Side effects: writes *dts.
    if PyDateTime_CheckExact(dt):
        dts.year = PyDateTime_GET_YEAR(dt)
    else:
        # We use dt.year instead of PyDateTime_GET_YEAR because with Timestamp
        #  we override year such that PyDateTime_GET_YEAR is incorrect.
        dts.year = dt.year

    dts.month = PyDateTime_GET_MONTH(dt)
    dts.day = PyDateTime_GET_DAY(dt)
    dts.hour = PyDateTime_DATE_GET_HOUR(dt)
    dts.min = PyDateTime_DATE_GET_MINUTE(dt)
    dts.sec = PyDateTime_DATE_GET_SECOND(dt)
    dts.us = PyDateTime_DATE_GET_MICROSECOND(dt)
    dts.ps = dts.as = 0


cdef int64_t pydatetime_to_dt64(datetime val,
                                npy_datetimestruct *dts,
                                NPY_DATETIMEUNIT reso=NPY_FR_ns) except? -1:
    # [FUNCTION DOC]
    # Convert a (naive, or treated as naive) Python datetime to an int64 count of reso
    #   units since the epoch, via pydatetime_to_dtstruct and the struct-to-int C
    #   routine.
    # Inputs: val - datetime; dts - scratch/output struct pointer; reso - target unit
    #   (default ns).
    # Outputs: int64 value. Exceptions: OutOfBoundsDatetime (chained from
    #   OverflowError) if the value does not fit. Side effects: fills *dts.
    """
    Note we are assuming that the datetime object is timezone-naive.
    """
    cdef int64_t result
    pydatetime_to_dtstruct(val, dts)
    try:
        result = npy_datetimestruct_to_datetime(reso, dts)
    except OverflowError as err:
        attrname = npy_unit_to_attrname[reso]
        raise OutOfBoundsDatetime(
            f"Out of bounds {attrname} timestamp: {val}"
        ) from err

    return result


cdef void pydate_to_dtstruct(date val, npy_datetimestruct *dts) noexcept:
    # [FUNCTION DOC]
    # Fill a datetime struct from a Python date: year, month, day from the date; all
    #   time fields zeroed. Inputs: val - date; dts - output pointer. Outputs: none.
    # Exceptions: none. Side effects: writes *dts.
    dts.year = PyDateTime_GET_YEAR(val)
    dts.month = PyDateTime_GET_MONTH(val)
    dts.day = PyDateTime_GET_DAY(val)
    dts.hour = dts.min = dts.sec = dts.us = 0
    dts.ps = dts.as = 0
    return


cdef int64_t pydate_to_dt64(
    date val, npy_datetimestruct *dts, NPY_DATETIMEUNIT reso=NPY_FR_ns
) except? -1:
    # [FUNCTION DOC]
    # Convert a Python date (midnight) to an int64 count of reso units since the
    #   epoch.
    # Inputs: val - date; dts - scratch/output struct; reso - target unit (default
    #   ns).
    # Outputs: int64. Exceptions: OutOfBoundsDatetime (from OverflowError) on
    #   overflow.
    # Side effects: fills *dts.
    cdef int64_t result
    pydate_to_dtstruct(val, dts)

    try:
        result = npy_datetimestruct_to_datetime(reso, dts)
    except OverflowError as err:
        attrname = npy_unit_to_attrname[reso]
        raise OutOfBoundsDatetime(f"Out of bounds {attrname} timestamp: {val}") from err

    return result


cdef int string_to_dts(
    str val,
    npy_datetimestruct* dts,
    NPY_DATETIMEUNIT* out_bestunit,
    int* out_local,
    int* out_tzoffset,
    bint want_exc,
    str format=None,
    bint exact=True,
) except? -1:
    # [FUNCTION DOC]
    # Parse a string as an ISO-8601-style datetime into a struct using the C parser,
    #   optionally guided by a strftime-style format.
    # Inputs: val - string (UTF-8 encoded for the parser); dts - output struct;
    #   out_bestunit - receives the finest unit present in the string; out_local - set
    #   to 1 if a timezone offset was present; out_tzoffset - receives offset in
    #   minutes; want_exc - if true, parser sets a Python exception on failure, else
    #   just returns -1; format - optional format; if None the format is inferred
    #   (INFER_FORMAT); exact - when format given, true means EXACT_MATCH, false
    #   PARTIAL_MATCH.
    # Outputs: 0 on success, -1 on failure.
    # Exceptions: ValueError/OutOfBoundsDatetime from the parser when want_exc is
    #   true.
    # Side effects: writes the output pointers.
    cdef:
        Py_ssize_t length
        const char* buf
        Py_ssize_t format_length
        const char* format_buf
        FormatRequirement format_requirement

    buf = PyUnicode_AsUTF8AndSize(val, &length)
    if format is None:
        format_buf = b""
        format_length = 0
        format_requirement = INFER_FORMAT
    else:
        format_buf = PyUnicode_AsUTF8AndSize(format, &format_length)
        format_requirement = <FormatRequirement>exact
    return parse_iso_8601_datetime(buf, length, want_exc,
                                   dts, out_bestunit, out_local, out_tzoffset,
                                   format_buf, format_length,
                                   format_requirement)


cpdef ndarray astype_overflowsafe(
    ndarray values,
    cnp.dtype dtype,
    bint copy=True,
    bint round_ok=True,
    bint is_coerce=False,
):
    # [FUNCTION DOC]
    # Cast an array of datetime64[X] to datetime64[Y] (or timedelta64[X] to [Y])
    #   without silent overflow, preserving NaT.
    # Inputs: values - M8 or m8 ndarray of any shape; dtype - target dtype of the same
    #   kind; copy - whether to copy when units match; round_ok - allow lossy truncation
    #   when going to a coarser unit; is_coerce - turn out-of-bounds values into NaT
    #   instead of raising (only on the finer-unit path).
    # Algorithm: if units equal return values (copied if copy). If target coarser,
    #   delegate to _astype_overflowsafe_to_smaller_unit (floor division). Otherwise
    #   byte-swap big-endian input, then for each element: NaT stays NaT, else convert
    #   to struct in the source unit, check bounds for the target unit and convert back
    #   to int64 in the target unit. Result is a new array viewed as dtype.
    # Outputs: ndarray of dtype.
    # Exceptions: TypeError if kinds differ or the source is unitless; ValueError if
    #   the target is unitless or (coarser path, not round_ok) a value would be
    #   truncated; OutOfBoundsDatetime for out of range datetimes; OutOfBoundsTimedelta
    #   for timedeltas.
    # Side effects: allocates the result array.
    # KNOWN ISSUE: the big-endian byte swap (GH#29684) is applied only on the finer-
    #   unit path; the coarser-unit path views big-endian data as native int64 directly.
    """
    Convert an ndarray with datetime64[X] to datetime64[Y]
    or timedelta64[X] to timedelta64[Y],
    raising on overflow.
    """
    if values.descr.type_num == dtype.type_num == cnp.NPY_DATETIME:
        # i.e. dtype.kind == "M"
        dtype_name = "datetime64"
    elif values.descr.type_num == dtype.type_num == cnp.NPY_TIMEDELTA:
        # i.e. dtype.kind == "m"
        dtype_name = "timedelta64"
    else:
        raise TypeError(
            "astype_overflowsafe values.dtype and dtype must be either "
            "both-datetime64 or both-timedelta64."
        )

    cdef:
        NPY_DATETIMEUNIT from_unit = get_unit_from_dtype(values.dtype)
        NPY_DATETIMEUNIT to_unit = get_unit_from_dtype(dtype)

    if from_unit == NPY_DATETIMEUNIT.NPY_FR_GENERIC:
        raise TypeError(f"{dtype_name} values must have a unit specified")

    if to_unit == NPY_DATETIMEUNIT.NPY_FR_GENERIC:
        # without raising explicitly here, we end up with a SystemError
        # built-in function [...] returned a result with an error
        raise ValueError(
            f"{dtype_name} dtype must have a unit specified"
        )

    if from_unit == to_unit:
        # Check this before allocating result for perf, might save some memory
        if copy:
            return values.copy()
        return values

    elif from_unit > to_unit:
        iresult2 = _astype_overflowsafe_to_smaller_unit(
            values.view("i8"), from_unit, to_unit, round_ok=round_ok
        )
        return iresult2.view(dtype)

    if (<object>values).dtype.byteorder == ">":
        # GH#29684 we incorrectly get OutOfBoundsDatetime if we dont swap
        values = values.astype(values.dtype.newbyteorder("<"))

    cdef:
        ndarray i8values = values.view("i8")

        # equiv: result = np.empty((<object>values).shape, dtype="i8")
        ndarray iresult = cnp.PyArray_EMPTY(
            values.ndim, values.shape, cnp.NPY_INT64, 0
        )

        cnp.broadcast mi = cnp.PyArray_MultiIterNew2(iresult, i8values)
        Py_ssize_t i, N = values.size
        int64_t value, new_value
        npy_datetimestruct dts
        bint is_td = dtype.type_num == cnp.NPY_TIMEDELTA

    for i in range(N):
        # Analogous to: item = values[i]
        value = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

        if value == NPY_DATETIME_NAT:
            new_value = NPY_DATETIME_NAT
        else:
            pandas_datetime_to_datetimestruct(value, from_unit, &dts)

            try:
                check_dts_bounds(&dts, to_unit)
            except OutOfBoundsDatetime as err:
                if is_coerce:
                    new_value = NPY_DATETIME_NAT
                elif is_td:
                    from_abbrev = np.datetime_data(values.dtype)[0]
                    np_val = np.timedelta64(value, from_abbrev)
                    msg = (
                        "Cannot convert {np_val} to {dtype} without overflow"
                        .format(np_val=str(np_val), dtype=str(dtype))
                    )
                    raise OutOfBoundsTimedelta(msg) from err
                else:
                    raise
            else:
                new_value = npy_datetimestruct_to_datetime(to_unit, &dts)

        # Analogous to: iresult[i] = new_value
        (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = new_value

        cnp.PyArray_MultiIter_NEXT(mi)

    return iresult.view(dtype)


# TODO(numpy#16352): try to upstream this fix to numpy
def compare_mismatched_resolutions(ndarray left, ndarray right, op):
    # [FUNCTION DOC]
    # Elementwise comparison of two datetime64 (or two timedelta64) arrays with
    #   possibly different units, done by converting each element to a calendar struct
    #   so no overflow occurs (numpy upcasts and can overflow).
    # Inputs: left, right - ndarrays of the same kind and broadcast-compatible shape
    #   (result shape is left's); op - an operator module comparison function.
    # Outputs: bool ndarray; any pair involving NaT yields True only for "ne".
    # Exceptions: ValueError if kinds differ or are not m/M. Side effects: allocates
    #   result.
    """
    Overflow-safe comparison of timedelta64/datetime64 with mismatched resolutions.

    >>> left = np.array([500], dtype="M8[Y]")
    >>> right = np.array([0], dtype="M8[ns]")
    >>> left < right  # <- wrong!
    array([ True])
    """

    if left.dtype.kind != right.dtype.kind or left.dtype.kind not in "mM":
        raise ValueError("left and right must both be timedelta64 or both datetime64")

    cdef:
        int op_code = op_to_op_code(op)
        NPY_DATETIMEUNIT left_unit = get_unit_from_dtype(left.dtype)
        NPY_DATETIMEUNIT right_unit = get_unit_from_dtype(right.dtype)

        # equiv: result = np.empty((<object>left).shape, dtype="bool")
        ndarray result = cnp.PyArray_EMPTY(
            left.ndim, left.shape, cnp.NPY_BOOL, 0
        )

        ndarray lvalues = left.view("i8")
        ndarray rvalues = right.view("i8")

        cnp.broadcast mi = cnp.PyArray_MultiIterNew3(result, lvalues, rvalues)
        int64_t lval, rval
        bint res_value

        Py_ssize_t i, N = left.size
        npy_datetimestruct ldts, rdts

    for i in range(N):
        # Analogous to: lval = lvalues[i]
        lval = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

        # Analogous to: rval = rvalues[i]
        rval = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 2))[0]

        if lval == NPY_DATETIME_NAT or rval == NPY_DATETIME_NAT:
            res_value = op_code == Py_NE

        else:
            pandas_datetime_to_datetimestruct(lval, left_unit, &ldts)
            pandas_datetime_to_datetimestruct(rval, right_unit, &rdts)

            res_value = cmp_dtstructs(&ldts, &rdts, op_code)

        # Analogous to: result[i] = res_value
        (<uint8_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = res_value

        cnp.PyArray_MultiIter_NEXT(mi)

    return result


cdef int op_to_op_code(op):
    # [FUNCTION DOC]
    # HELPER: Map operator.eq/ne/le/lt/ge/gt (identity check) to the Python rich
    #   comparison opcode. Inputs: op - callable. Outputs: int opcode.
    # Exceptions: none. Side effects: none.
    # KNOWN ISSUE: an unrecognised op falls through and returns 0, which equals Py_LT,
    #   so it is silently treated as a less-than comparison.
    if op is operator.eq:
        return Py_EQ
    if op is operator.ne:
        return Py_NE
    if op is operator.le:
        return Py_LE
    if op is operator.lt:
        return Py_LT
    if op is operator.ge:
        return Py_GE
    if op is operator.gt:
        return Py_GT


cdef ndarray _astype_overflowsafe_to_smaller_unit(
    ndarray i8values,
    NPY_DATETIMEUNIT from_unit,
    NPY_DATETIMEUNIT to_unit,
    bint round_ok,
):
    # [FUNCTION DOC]
    # HELPER: Convert int64 values from a finer unit to a coarser unit (e.g. ns -> us)
    #   by floor-dividing by the conversion factor, preserving NaT.
    # Inputs: i8values - int64 view of the data; from_unit - finer unit; to_unit -
    #   coarser unit; round_ok - if false, a nonzero remainder is an error.
    # Outputs: new int64 ndarray of the same shape.
    # Exceptions: ValueError("Cannot losslessly cast ...") on remainder when not
    #   round_ok; errors from get_conversion_factor. Side effects: allocates result.
    """
    Overflow-safe conversion for cases with from_unit > to_unit, e.g. ns->us.
    In addition for checking for overflows (which can occur near the lower
    implementation bound, see numpy#22346), this checks for truncation,
    e.g. 1500ns->1us.
    """
    # e.g. test_astype_ns_to_ms_near_bounds is a case with round_ok=True where
    #  just using numpy's astype silently fails
    cdef:
        Py_ssize_t i, N = i8values.size

        # equiv: iresult = np.empty((<object>i8values).shape, dtype="i8")
        ndarray iresult = cnp.PyArray_EMPTY(
            i8values.ndim, i8values.shape, cnp.NPY_INT64, 0
        )
        cnp.broadcast mi = cnp.PyArray_MultiIterNew2(iresult, i8values)

        # Note the arguments to_unit, from unit are swapped vs how they
        #  are passed when going to a higher-frequency reso.
        int64_t mult = get_conversion_factor(to_unit, from_unit)
        int64_t value, mod

    for i in range(N):
        # Analogous to: item = i8values[i]
        value = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

        if value == NPY_DATETIME_NAT:
            new_value = NPY_DATETIME_NAT
        else:
            new_value, mod = divmod(value, mult)
            if not round_ok and mod != 0:
                from_abbrev = npy_unit_to_abbrev(from_unit)
                to_abbrev = npy_unit_to_abbrev(to_unit)
                raise ValueError(
                    f"Cannot losslessly cast '{value} {from_abbrev}' to {to_abbrev}"
                )

        # Analogous to: iresult[i] = new_value
        (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = new_value

        cnp.PyArray_MultiIter_NEXT(mi)

    return iresult


cdef int64_t get_conversion_factor(
    NPY_DATETIMEUNIT from_unit,
    NPY_DATETIMEUNIT to_unit
) except? -1:
    # [FUNCTION DOC]
    # Return the integer multiplier converting a count in from_unit to to_unit, where
    #   from_unit is coarser or equal (W=7 D, D=24 h, h=60 m, m=60 s, then factors of
    #   1000 down to as). Computed recursively by chaining one-step factors.
    # Inputs: from_unit, to_unit - NPY_DATETIMEUNIT. Outputs: int64 factor (1 if
    #   equal).
    # Exceptions: ValueError for generic units, if from_unit > to_unit, or for Y/M/B
    #   start units; OverflowError("result would overflow") if the factor exceeds int64.
    # Side effects: none.
    """
    Find the factor by which we need to multiply to convert from from_unit to to_unit.
    """
    cdef int64_t value, overflow_limit, factor
    if (
        from_unit == NPY_DATETIMEUNIT.NPY_FR_GENERIC
        or to_unit == NPY_DATETIMEUNIT.NPY_FR_GENERIC
    ):
        raise ValueError("unit-less resolutions are not supported")
    if from_unit > to_unit:
        raise ValueError("from_unit must be <= to_unit")

    if from_unit == to_unit:
        return 1

    if from_unit == NPY_DATETIMEUNIT.NPY_FR_W:
        value = get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_D, to_unit)
        factor = 7
    elif from_unit == NPY_DATETIMEUNIT.NPY_FR_D:
        value = get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_h, to_unit)
        factor = 24
    elif from_unit == NPY_DATETIMEUNIT.NPY_FR_h:
        value = get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_m, to_unit)
        factor = 60
    elif from_unit == NPY_DATETIMEUNIT.NPY_FR_m:
        value = get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_s, to_unit)
        factor = 60
    elif from_unit == NPY_DATETIMEUNIT.NPY_FR_s:
        value = get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_ms, to_unit)
        factor = 1000
    elif from_unit == NPY_DATETIMEUNIT.NPY_FR_ms:
        value = get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_us, to_unit)
        factor = 1000
    elif from_unit == NPY_DATETIMEUNIT.NPY_FR_us:
        value = get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_ns, to_unit)
        factor = 1000
    elif from_unit == NPY_DATETIMEUNIT.NPY_FR_ns:
        value = get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_ps, to_unit)
        factor = 1000
    elif from_unit == NPY_DATETIMEUNIT.NPY_FR_ps:
        value = get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_fs, to_unit)
        factor = 1000
    elif from_unit == NPY_DATETIMEUNIT.NPY_FR_fs:
        value = get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_as, to_unit)
        factor = 1000
    else:
        raise ValueError("Converting from M or Y units is not supported.")

    overflow_limit = INT64_MAX // factor
    if value > overflow_limit or value < -overflow_limit:
        raise OverflowError("result would overflow")

    return factor * value


cdef int64_t convert_reso(
    int64_t value,
    NPY_DATETIMEUNIT from_reso,
    NPY_DATETIMEUNIT to_reso,
    bint round_ok,
) except? -1:
    # [FUNCTION DOC]
    # Convert a single int64 value between resolutions.
    # Inputs: value - int64 count; from_reso, to_reso - units; round_ok - allow lossy
    #   truncation when going coarser.
    # Behavior: equal units return value. Coarser target: floor-divide by factor
    #   (rounds down like numpy). If either unit is Y or M: go through a calendar
    #   struct. Otherwise: multiply by factor after an overflow check.
    # Outputs: converted int64.
    # Exceptions: ValueError("Cannot losslessly convert units") if remainder and not
    #   round_ok; OverflowError on multiplication overflow (callers re-raise as needed);
    #   OutOfBoundsDatetime from the struct path. Side effects: none.
    cdef:
        int64_t res_value, mult, div, mod, overflow_limit

    if from_reso == to_reso:
        return value

    elif to_reso < from_reso:
        # e.g. ns -> us, no risk of overflow, but can be lossy rounding
        mult = get_conversion_factor(to_reso, from_reso)
        div, mod = divmod(value, mult)
        if mod > 0 and not round_ok:
            raise ValueError("Cannot losslessly convert units")

        # Note that when mod > 0, we follow np.timedelta64 in always
        #  rounding down.
        res_value = div

    elif (
        from_reso == NPY_FR_Y
        or from_reso == NPY_FR_M
        or to_reso == NPY_FR_Y
        or to_reso == NPY_FR_M
    ):
        # Converting by multiplying isn't _quite_ right bc the number of
        #  seconds in a month/year isn't fixed.
        res_value = _convert_reso_with_dtstruct(value, from_reso, to_reso)

    else:
        # e.g. ns -> us, risk of overflow, but no risk of lossy rounding
        mult = get_conversion_factor(from_reso, to_reso)
        overflow_limit = INT64_MAX // mult
        if value > overflow_limit or value < -overflow_limit:
            # Note: caller is responsible for re-raising as OutOfBoundsTimedelta
            raise OverflowError("result would overflow")

        res_value = value * mult

    return res_value


cdef int64_t _convert_reso_with_dtstruct(
    int64_t value,
    NPY_DATETIMEUNIT from_unit,
    NPY_DATETIMEUNIT to_unit,
) except? -1:
    # [FUNCTION DOC]
    # HELPER: Convert an int64 value between units by expanding it into a calendar
    #   struct in from_unit and re-encoding in to_unit (needed for month/year units).
    # Inputs: value - int64; from_unit, to_unit - units. Outputs: int64.
    # Exceptions: OutOfBoundsDatetime (from OverflowError) if out of range.
    # Side effects: none.
    cdef:
        npy_datetimestruct dts
        int64_t result

    pandas_datetime_to_datetimestruct(value, from_unit, &dts)
    try:
        result = npy_datetimestruct_to_datetime(to_unit, &dts)
    except OverflowError as err:
        raise OutOfBoundsDatetime from err

    return result


@cython.overflowcheck(True)
cpdef cnp.ndarray add_overflowsafe(cnp.ndarray left, cnp.ndarray right):
    # [FUNCTION DOC]
    # Elementwise addition of two int64-backed datetime64/timedelta64 arrays with
    #   overflow checking (compiled with integer overflow checks), propagating NaT.
    # Inputs: left - ndarray; right - ndarray broadcastable to left (zero-dim or same
    #   shape).
    # Outputs: new int64 ndarray of left's shape (caller re-views as needed); NaT
    #   where either operand is NaT, else the sum.
    # Exceptions: OverflowError("Overflow in int64 addition") on overflow.
    # Side effects: allocates result.
    """
    Overflow-safe addition for datetime64/timedelta64 dtypes.

    `right` may either be zero-dim or of the same shape as `left`.
    """
    cdef:
        Py_ssize_t N = left.size
        int64_t lval, rval, res_value
        ndarray iresult = cnp.PyArray_EMPTY(
            left.ndim, left.shape, cnp.NPY_INT64, 0
        )
        cnp.broadcast mi = cnp.PyArray_MultiIterNew3(iresult, left, right)

    # Note: doing this try/except outside the loop improves performance over
    #  doing it inside the loop.
    try:
        for i in range(N):
            # Analogous to: lval = lvalues[i]
            lval = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

            # Analogous to: rval = rvalues[i]
            rval = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 2))[0]

            if lval == NPY_DATETIME_NAT or rval == NPY_DATETIME_NAT:
                res_value = NPY_DATETIME_NAT
            else:
                res_value = lval + rval

            # Analogous to: result[i] = res_value
            (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = res_value

            cnp.PyArray_MultiIter_NEXT(mi)
    except OverflowError as err:
        raise OverflowError("Overflow in int64 addition") from err

    return iresult
