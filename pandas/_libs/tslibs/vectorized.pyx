# [MODULE DOC]
# Vectorized operations on arrays of int64 epoch values (UTC, unit given by reso,
#   minimum int64 = NaT) that need the local wall time in a timezone. Every function
#   builds a Localizer (tzconversion.pyx) once, then iterates over all elements of an
#   array of any dimension in C order, converting each UTC value to local time before
#   computing its result. Results have the same shape as the input.
# Provides: UTC -> local conversion, boxing to Python datetime/date/time/Timestamp
#   objects, finest-resolution detection, normalization to local midnight, normalization
#   check, and conversion to period ordinals.
# Side effects at import: initializes numpy and pandas datetime C-APIs.
cimport cython
cimport numpy as cnp
from cpython.datetime cimport (
    date,
    datetime,
    time,
    tzinfo,
)
from numpy cimport (
    int64_t,
    ndarray,
)

cnp.import_array()

from .dtypes import Resolution

from .dtypes cimport (
    c_Resolution,
    periods_per_day,
)
from .nattype cimport (
    NPY_NAT,
    c_NaT as NaT,
)
from .np_datetime cimport (
    NPY_DATETIMEUNIT,
    NPY_FR_ns,
    import_pandas_datetime,
    npy_datetimestruct,
    pandas_datetime_to_datetimestruct,
)

import_pandas_datetime()

from .period cimport get_period_ordinal
from .timestamps cimport create_timestamp_from_ts
from .timezones cimport is_utc
from .tzconversion cimport Localizer


@cython.boundscheck(False)
@cython.wraparound(False)
def tz_convert_from_utc(ndarray stamps, tzinfo tz, NPY_DATETIMEUNIT reso=NPY_FR_ns):
    # [FUNCTION DOC]
    # Convert UTC int64 values to local wall-time int64 values. Inputs: stamps - int64
    #   ndarray, any shape; tz - tzinfo or None; reso - unit (default ns). Outputs: new
    #   int64 array, same shape; NaT stays NaT. If tz is None/UTC or stamps is empty,
    #   returns a copy. Exceptions: as Localizer (NotImplementedError for unsupported
    #   reso).
    # stamps is int64_t, arbitrary ndim
    """
    Convert the values (in i8) from UTC to tz

    Parameters
    ----------
    stamps : ndarray[int64]
    tz : tzinfo

    Returns
    -------
    ndarray[int64]
    """
    if tz is None or is_utc(tz) or stamps.size == 0:
        # Much faster than going through the "standard" pattern below;
        #  do this before initializing Localizer.
        return stamps.copy()

    cdef:
        Localizer info = Localizer(tz, creso=reso)
        int64_t utc_val, local_val
        Py_ssize_t pos, i, n = stamps.size

        ndarray result
        cnp.broadcast mi

    result = cnp.PyArray_EMPTY(stamps.ndim, stamps.shape, cnp.NPY_INT64, 0)
    mi = cnp.PyArray_MultiIterNew2(result, stamps)

    for i in range(n):
        # Analogous to: utc_val = stamps[i]
        utc_val = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

        if utc_val == NPY_NAT:
            local_val = NPY_NAT
        else:
            local_val = info.utc_val_to_local_val(utc_val, &pos)

        # Analogous to: result[i] = local_val
        (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = local_val

        cnp.PyArray_MultiIter_NEXT(mi)

    return result


# -------------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
def ints_to_pydatetime(
    ndarray stamps,
    tzinfo tz=None,
    str box="datetime",
    NPY_DATETIMEUNIT reso=NPY_FR_ns,
) -> ndarray:
    # [FUNCTION DOC]
    # Box int64 UTC values into Python objects. Inputs: stamps - int64 ndarray, any
    #   shape; tz - tzinfo or None; box - "datetime", "date", "time" or "timestamp";
    #   reso - unit.
    # For each value: NaT -> the NaT singleton. Otherwise convert to local time (getting
    #   fold), and for pytz zones pick the pytz tzinfo variant for the transition in
    #   force; decompose into fields; build a Timestamp (from UTC value, fields, tz,
    #   fold, reso), a datetime (with tz and fold), a date (no tz), or a time (with tz
    #   and fold). Sub-microsecond parts are dropped except for Timestamp.
    # Outputs: object ndarray with input shape. Exceptions: AssertionError if box="date"
    #   with non-None tz; ValueError for an unknown box; errors from datetime
    #   constructors for out-of-range years.
    # stamps is int64, arbitrary ndim
    """
    Convert an i8 repr to an ndarray of datetimes, date, time or Timestamp.

    Parameters
    ----------
    stamps : array of i8
    tz : str, optional
         convert to this timezone
    box : {'datetime', 'timestamp', 'date', 'time'}, default 'datetime'
        * If datetime, convert to datetime.datetime
        * If date, convert to datetime.date
        * If time, convert to datetime.time
        * If Timestamp, convert to pandas.Timestamp

    reso : NPY_DATETIMEUNIT, default NPY_FR_ns

    Returns
    -------
    ndarray[object] of type specified by box
    """
    cdef:
        Localizer info = Localizer(tz, creso=reso)
        int64_t utc_val, local_val
        Py_ssize_t i, n = stamps.size
        Py_ssize_t pos = -1  # unused, avoid not-initialized warning

        npy_datetimestruct dts
        tzinfo new_tz
        bint use_date = False, use_ts = False, use_pydt = False
        object res_val
        bint fold = 0

        # Note that `result` (and thus `result_flat`) is C-order and
        #  `it` iterates C-order as well, so the iteration matches
        #  See discussion at
        #  github.com/pandas-dev/pandas/pull/46886#discussion_r860261305
        ndarray result = cnp.PyArray_EMPTY(stamps.ndim, stamps.shape, cnp.NPY_OBJECT, 0)
        object[::1] res_flat = result.ravel()     # should NOT be a copy
        cnp.flatiter it = cnp.PyArray_IterNew(stamps)

    if box == "date":
        assert (tz is None), "tz should be None when converting to date"
        use_date = True
    elif box == "timestamp":
        use_ts = True
    elif box == "datetime":
        use_pydt = True
    elif box != "time":
        raise ValueError(
            "box must be one of 'datetime', 'date', 'time' or 'timestamp'"
        )

    for i in range(n):
        # Analogous to: utc_val = stamps[i]
        utc_val = (<int64_t*>cnp.PyArray_ITER_DATA(it))[0]

        new_tz = tz

        if utc_val == NPY_NAT:
            res_val = <object>NaT

        else:

            local_val = info.utc_val_to_local_val(utc_val, &pos, &fold)
            if info.use_pytz:
                # find right representation of dst etc in pytz timezone
                new_tz = tz._tzinfos[tz._transition_info[pos]]

            pandas_datetime_to_datetimestruct(local_val, reso, &dts)

            if use_ts:
                res_val = create_timestamp_from_ts(
                    utc_val, dts, new_tz, fold, reso=reso
                )
            elif use_pydt:
                res_val = datetime(
                    dts.year, dts.month, dts.day, dts.hour, dts.min, dts.sec, dts.us,
                    new_tz, fold=fold,
                )
            elif use_date:
                res_val = date(dts.year, dts.month, dts.day)
            else:
                res_val = time(dts.hour, dts.min, dts.sec, dts.us, new_tz, fold=fold)

        # Note: we can index result directly instead of using PyArray_MultiIter_DATA
        #  like we do for the other functions because result is known C-contiguous
        #  and is the first argument to PyArray_MultiIterNew2.  The usual pattern
        #  does not seem to work with object dtype.
        #  See discussion at
        #  github.com/pandas-dev/pandas/pull/46886#discussion_r860261305
        res_flat[i] = res_val

        cnp.PyArray_ITER_NEXT(it)

    return result


# -------------------------------------------------------------------------


cdef c_Resolution _reso_stamp(npy_datetimestruct *dts):
    # [FUNCTION DOC]
    # HELPER: Finest resolution present in one datetime struct: nanosecond if any
    #   picoseconds (i.e. sub-us part), else millisecond if microseconds are a non-zero
    #   multiple of 1000, microsecond if non-zero, then second, minute, hour by first
    #   non-zero field, else day. Inputs: dts - pointer to fields. Outputs:
    #   c_Resolution.
    if dts.ps != 0:
        return c_Resolution.RESO_NS
    elif dts.us != 0:
        if dts.us % 1000 == 0:
            return c_Resolution.RESO_MS
        return c_Resolution.RESO_US
    elif dts.sec != 0:
        return c_Resolution.RESO_SEC
    elif dts.min != 0:
        return c_Resolution.RESO_MIN
    elif dts.hour != 0:
        return c_Resolution.RESO_HR
    return c_Resolution.RESO_DAY


@cython.wraparound(False)
@cython.boundscheck(False)
def get_resolution(
    ndarray stamps, tzinfo tz=None, NPY_DATETIMEUNIT reso=NPY_FR_ns
) -> Resolution:
    # [FUNCTION DOC]
    # Finest resolution needed to represent all values in local time. Inputs: stamps -
    #   int64 ndarray (any shape); tz; reso. Skips NaT; for each other value converts to
    #   local, decomposes into fields, and takes the minimum resolution from the
    #   per-value helper. Outputs: Resolution enum member; RESO_DAY for empty or all-NaT
    #   input.
    # stamps is int64_t, any ndim
    cdef:
        Localizer info = Localizer(tz, creso=reso)
        int64_t utc_val, local_val
        Py_ssize_t i, n = stamps.size
        Py_ssize_t pos = -1  # unused, avoid not-initialized warning
        cnp.flatiter it = cnp.PyArray_IterNew(stamps)

        npy_datetimestruct dts
        c_Resolution pd_reso = c_Resolution.RESO_DAY, curr_reso

    for i in range(n):
        # Analogous to: utc_val = stamps[i]
        utc_val = (<int64_t*>cnp.PyArray_ITER_DATA(it))[0]

        if utc_val == NPY_NAT:
            pass
        else:
            local_val = info.utc_val_to_local_val(utc_val, &pos)

            pandas_datetime_to_datetimestruct(local_val, reso, &dts)
            curr_reso = _reso_stamp(&dts)
            if curr_reso < pd_reso:
                pd_reso = curr_reso

        cnp.PyArray_ITER_NEXT(it)

    return Resolution(pd_reso)


# -------------------------------------------------------------------------


@cython.overflowcheck(True)
@cython.cdivision(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cpdef ndarray normalize_i8_timestamps(ndarray stamps, tzinfo tz, NPY_DATETIMEUNIT reso):
    # [FUNCTION DOC]
    # Floor each value to local midnight. Inputs: stamps - int64 ndarray (any shape);
    #   tz; reso. For non-NaT values: local = UTC converted to tz, result = local minus
    #   (local floor-modulo units-per-day). Note the result is a local wall-time value
    #   (not converted back to UTC). NaT stays NaT.
    # Outputs: new int64 array. Exceptions: ValueError("Cannot normalize Timestamp
    #   without integer overflow") on overflow; Localizer errors.
    # stamps is int64_t, arbitrary ndim
    """
    Normalize each of the (nanosecond) timezone aware timestamps in the given
    array by rounding down to the beginning of the day (i.e. midnight).
    This is midnight for timezone, `tz`.

    Parameters
    ----------
    stamps : int64 ndarray
    tz : tzinfo or None
    reso : NPY_DATETIMEUNIT

    Returns
    -------
    result : int64 ndarray of converted of normalized nanosecond timestamps
    """
    cdef:
        Localizer info = Localizer(tz, creso=reso)
        int64_t utc_val, local_val, res_val
        Py_ssize_t i, n = stamps.size
        Py_ssize_t pos = -1  # unused, avoid not-initialized warning

        ndarray result = cnp.PyArray_EMPTY(stamps.ndim, stamps.shape, cnp.NPY_INT64, 0)
        cnp.broadcast mi = cnp.PyArray_MultiIterNew2(result, stamps)
        int64_t ppd = periods_per_day(reso)

    for i in range(n):
        # Analogous to: utc_val = stamps[i]
        utc_val = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

        if utc_val == NPY_NAT:
            res_val = NPY_NAT
        else:
            local_val = info.utc_val_to_local_val(utc_val, &pos)
            try:
                res_val = local_val - (local_val % ppd)
            except OverflowError as err:
                raise ValueError(
                    "Cannot normalize Timestamp without integer overflow"
                ) from err
        # Analogous to: result[i] = res_val
        (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = res_val

        cnp.PyArray_MultiIter_NEXT(mi)

    return result


@cython.wraparound(False)
@cython.boundscheck(False)
def is_date_array_normalized(ndarray stamps, tzinfo tz, NPY_DATETIMEUNIT reso) -> bool:
    # [FUNCTION DOC]
    # True if every value, converted to local time, is an exact multiple of
    #   units-per-day (local midnight). Inputs: stamps; tz; reso. Returns False at the
    #   first non-midnight value; True for empty input.
    # KNOWN ISSUE: NaT values are not skipped; NaT (minimum int64) is not a multiple of
    #   a day, so any NaT makes the result False (and for DST zones NaT goes through the
    #   transition search).
    # stamps is int64_t, arbitrary ndim
    """
    Check if all of the given (nanosecond) timestamps are normalized to
    midnight, i.e. hour == minute == second == 0.  If the optional timezone
    `tz` is not None, then this is midnight for this timezone.

    Parameters
    ----------
    stamps : int64 ndarray
    tz : tzinfo or None
    reso : NPY_DATETIMEUNIT

    Returns
    -------
    is_normalized : bool True if all stamps are normalized
    """
    cdef:
        Localizer info = Localizer(tz, creso=reso)
        int64_t utc_val, local_val
        Py_ssize_t i, n = stamps.size
        Py_ssize_t pos = -1  # unused, avoid not-initialized warning
        cnp.flatiter it = cnp.PyArray_IterNew(stamps)
        int64_t ppd = periods_per_day(reso)

    for i in range(n):
        # Analogous to: utc_val = stamps[i]
        utc_val = (<int64_t*>cnp.PyArray_ITER_DATA(it))[0]

        local_val = info.utc_val_to_local_val(utc_val, &pos)

        if local_val % ppd != 0:
            return False

        cnp.PyArray_ITER_NEXT(it)

    return True


# -------------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
def dt64arr_to_periodarr(
    ndarray stamps, int freq, tzinfo tz, NPY_DATETIMEUNIT reso=NPY_FR_ns
):
    # [FUNCTION DOC]
    # Convert UTC int64 values to period ordinals. Inputs: stamps - int64 ndarray (any
    #   shape); freq - period dtype code int (e.g. 3000 monthly); tz; reso. For non-NaT
    #   values converts to local, decomposes into fields and computes the period ordinal
    #   for freq. Outputs: new int64 array; NaT stays NaT. Exceptions: Localizer errors.
    # stamps is int64_t, arbitrary ndim
    cdef:
        Localizer info = Localizer(tz, creso=reso)
        Py_ssize_t i, n = stamps.size
        Py_ssize_t pos = -1  # unused, avoid not-initialized warning
        int64_t utc_val, local_val, res_val

        npy_datetimestruct dts
        ndarray result = cnp.PyArray_EMPTY(stamps.ndim, stamps.shape, cnp.NPY_INT64, 0)
        cnp.broadcast mi = cnp.PyArray_MultiIterNew2(result, stamps)

    for i in range(n):
        # Analogous to: utc_val = stamps[i]
        utc_val = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

        if utc_val == NPY_NAT:
            res_val = NPY_NAT
        else:
            local_val = info.utc_val_to_local_val(utc_val, &pos)
            pandas_datetime_to_datetimestruct(local_val, reso, &dts)
            res_val = get_period_ordinal(&dts, freq)

        # Analogous to: result[i] = res_val
        (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = res_val

        cnp.PyArray_MultiIter_NEXT(mi)

    return result
