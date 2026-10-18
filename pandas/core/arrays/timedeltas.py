# [MODULE DOC]
# Extension array for timedelta data. Public contents: TimedeltaArray
# (published in the arrays namespace), a datetime-like array backed by a numpy
# timedelta64 array with a unit of seconds, milliseconds, microseconds or
# nanoseconds and an optional fixed frequency; and the constructor helper
# sequence_to_td64ns, which converts many input kinds (strings, objects,
# integers, floats, timedelta64 arrays) to a timedelta64 numpy array and
# infers a frequency when the input already carries one. Private helpers
# convert integers and objects and validate dtypes, and a factory builds
# read-only component properties (days, seconds, microseconds, nanoseconds).
# Relationships: subclasses the datetime-like timelike-ops base in the
# datetimelike module; used by the TimedeltaIndex, the Series dt accessor and
# conversion functions. No import-time side effects beyond imports and
# property creation on the class.
from __future__ import annotations

from datetime import timedelta
import operator
from typing import (
    TYPE_CHECKING,
    Self,
    cast,
)

import numpy as np

from pandas._libs import (
    lib,
    tslibs,
)
from pandas._libs.tslibs import (
    Day,
    NaT,
    NaTType,
    Tick,
    Timedelta,
    astype_overflowsafe,
    get_supported_dtype,
    iNaT,
    is_supported_dtype,
    periods_per_second,
    to_offset,
)
from pandas._libs.tslibs.conversion import cast_from_unit_vectorized
from pandas._libs.tslibs.fields import (
    get_timedelta_days,
    get_timedelta_field,
)
from pandas._libs.tslibs.timedeltas import (
    array_to_timedelta64,
    floordiv_object_array,
    ints_to_pytimedelta,
    parse_timedelta_unit,
    truediv_object_array,
)
from pandas.compat.numpy import function as nv
from pandas.util._decorators import set_module
from pandas.util._validators import validate_endpoints

from pandas.core.dtypes.common import (
    TD64NS_DTYPE,
    is_float_dtype,
    is_integer_dtype,
    is_object_dtype,
    is_scalar,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    BaseMaskedDtype,
    ExtensionDtype,
)
from pandas.core.dtypes.missing import isna

from pandas.core import (
    nanops,
    roperator,
)
from pandas.core.array_algos import datetimelike_accumulations
from pandas.core.arrays import datetimelike as dtl
from pandas.core.arrays._ranges import generate_regular_range
import pandas.core.common as com
from pandas.core.ops.common import unpack_zerodim_and_defer

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from pandas._typing import (
        AxisInt,
        DateTimeErrorChoices,
        DtypeObj,
        NpDtype,
        npt,
        TimeUnit,
    )

    from pandas import DataFrame

import textwrap


def _field_accessor(name: str, alias: str, docstring: str):
    # [FUNCTION DOC]
    # HELPER: factory for the read-only component properties of the timedelta
    # array. Inputs: name, the property name; alias, the field to extract
    # ("days" uses the dedicated day extraction, others the generic field
    # extraction); docstring, the documentation text. Returns a property whose
    # getter computes the field from the integer representation at the array's
    # resolution; if the array has missing values, they are masked and the result
    # is float with NaN there. The getter's name and documentation are set from
    # name and docstring.
    def f(self) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: the generated getter; returns the component values (integer
        # array, or float array when missing values are present).
        values = self.asi8
        if alias == "days":
            result = get_timedelta_days(values, reso=self._creso)
        else:
            # error: Incompatible types in assignment (
            # expression has type "ndarray[Any, dtype[signedinteger[_32Bit]]]",
            # variable has type "ndarray[Any, dtype[signedinteger[_64Bit]]]
            result = get_timedelta_field(values, alias, reso=self._creso)  # type: ignore[assignment]
        if self._hasna:
            result = self._maybe_mask_results(
                result, fill_value=None, convert="float64"
            )

        return result

    f.__name__ = name
    f.__doc__ = f"\n{docstring}\n"
    return property(f)


@set_module("pandas.arrays")
class TimedeltaArray(dtl.TimelikeOps):
    # [CLASS DOC]
    # Extension array for timedeltas (subclass of the datetime-like
    # timelike-operations base). State: the backing numpy timedelta64 array (unit
    # s, ms, us or ns), and _freq, an optional fixed frequency (tick or day
    # offset) that is kept valid across operations that preserve regular spacing.
    # Class attributes: the type tag "timedeltaarray", recognised scalar types
    # (standard timedelta, numpy timedelta64, tick offsets), a predicate for
    # recognised dtypes (a function defined by assignment: true for numpy
    # timedelta dtypes), inference tags, a high array priority, and lists of
    # delegated field names and methods. The days, seconds, microseconds and
    # nanoseconds properties are generated at class creation by the field
    # factory. Key behaviour: construction helpers, unit conversion, scalar and
    # vector arithmetic (multiplication, true and floor division, modulo,
    # divmod, negation, absolute value) with frequency tracking, sum and
    # standard deviation reductions, cumulative sum, conversion to total seconds,
    # Python timedeltas and component tables.
    """
    Pandas ExtensionArray for timedelta data.

    .. warning::

       TimedeltaArray is currently experimental, and its API may change
       without warning. In particular, :attr:`TimedeltaArray.dtype` is
       expected to change to be an instance of an ``ExtensionDtype``
       subclass.

    Parameters
    ----------
    data : array-like
        The timedelta data.
    dtype : numpy.dtype
        Currently, only ``numpy.dtype("timedelta64[ns]")`` is accepted.
    freq : Offset, optional
        Frequency of the data.
    copy : bool, default False
        Whether to copy the underlying array of data.

    Attributes
    ----------
    None

    Methods
    -------
    None

    See Also
    --------
    Timedelta : Represents a duration, the difference between two dates or times.
    TimedeltaIndex : Immutable Index of timedelta64 data.
    to_timedelta : Convert argument to timedelta.

    Examples
    --------
    >>> pd.arrays.TimedeltaArray._from_sequence(pd.TimedeltaIndex(["1h", "2h"]))
    <TimedeltaArray>
    ['0 days 01:00:00', '0 days 02:00:00']
    Length: 2, dtype: timedelta64[us]
    """

    _typ = "timedeltaarray"
    _recognized_scalars = (timedelta, np.timedelta64, Tick)
    _is_recognized_dtype: Callable[[DtypeObj], bool] = lambda x: lib.is_np_dtype(x, "m")
    _infer_matches = ("timedelta", "timedelta64")

    @property
    def _internal_fill_value(self) -> np.timedelta64:
        # [FUNCTION DOC]
        # Property: the missing-value marker (not-a-time) as a numpy timedelta
        # scalar in this array's unit.
        return np.timedelta64("NaT", self.unit)

    @property
    def _scalar_type(self) -> type[Timedelta]:
        # [FUNCTION DOC]
        # Property: the scalar class of the elements, Timedelta.
        return Timedelta

    __array_priority__ = 1000
    # define my properties & methods for delegation
    _other_ops: list[str] = []
    _bool_ops: list[str] = []
    _field_ops: list[str] = ["days", "seconds", "microseconds", "nanoseconds"]
    _datetimelike_ops: list[str] = _field_ops + _bool_ops + ["unit", "freq"]
    _datetimelike_methods: list[str] = [
        "to_pytimedelta",
        "total_seconds",
        "round",
        "floor",
        "ceil",
        "as_unit",
    ]

    # Note: ndim must be defined to ensure NaT.__richcmp__(TimedeltaArray)
    #  operates pointwise.

    def _box_func(self, x: np.timedelta64) -> Timedelta | NaTType:
        # [FUNCTION DOC]
        # HELPER: boxes a numpy timedelta scalar. Reinterprets x as an integer;
        # the missing-value integer gives the not-a-time singleton, otherwise a
        # Timedelta with the same integer value at this array's resolution.
        y = x.view("i8")
        if y == NaT._value:
            return NaT
        return Timedelta._from_value_and_reso(y, reso=self._creso)

    @property
    # error: Return type "dtype" of "dtype" incompatible with return type
    # "ExtensionDtype" in supertype "ExtensionArray"
    def dtype(self) -> np.dtype[np.timedelta64]:  # type: ignore[override]
        # [FUNCTION DOC]
        # Property: the numpy timedelta64 dtype (with unit) of the backing array.
        """
        The dtype for the TimedeltaArray.

        .. warning::

           A future version of pandas will change dtype to be an instance
           of a :class:`pandas.api.extensions.ExtensionDtype` subclass,
           not a ``numpy.dtype``.

        Returns
        -------
        numpy.dtype
        """
        return self._ndarray.dtype

    # ----------------------------------------------------------------
    # Constructors

    _freq: Tick | Day | None = None

    @classmethod
    def _validate_dtype(cls, values, dtype):
        # [FUNCTION DOC]
        # HELPER (class method) called by the base initialiser. Inputs: values,
        # the backing array; dtype, requested dtype. Validates the requested dtype
        # (supported timedelta resolution with a unit) and the values' dtype, and
        # raises a value error if the two differ. Returns the validated dtype.
        # used in TimeLikeOps.__init__
        dtype = _validate_td64_dtype(dtype)
        _validate_td64_dtype(values.dtype)
        if dtype != values.dtype:
            raise ValueError("Values resolution does not match dtype.")
        return dtype

    # error: Signature of "_simple_new" incompatible with supertype "NDArrayBacked"
    @classmethod
    def _simple_new(  # type: ignore[override]
        cls,
        values: npt.NDArray[np.timedelta64],
        freq: Tick | Day | None = None,
        dtype: np.dtype[np.timedelta64] = TD64NS_DTYPE,
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER (class method): low-level constructor that skips validation of
        # conversions. Inputs: values, a numpy timedelta64 array; freq (default
        # none), a tick or day offset; dtype (default nanosecond timedelta64) which
        # must equal the values' dtype. Asserts (assertion error) that the dtype is a
        # timedelta dtype with a unit equal to the values' dtype, that values is a
        # numpy array and that freq is none or a tick/day offset. Returns a new
        # array wrapping values with the given freq.
        # Require td64 dtype, not unit-less, matching values.dtype
        assert lib.is_np_dtype(dtype, "m")
        assert not tslibs.is_unitless(dtype)
        assert isinstance(values, np.ndarray), type(values)
        assert dtype == values.dtype
        assert freq is None or isinstance(freq, (Tick, Day))

        result = super()._simple_new(values=values, dtype=dtype)
        result._freq = freq
        return result

    @classmethod
    def _from_sequence(cls, data, *, dtype=None, copy: bool = False) -> Self:
        # [FUNCTION DOC]
        # Class method building an array from a sequence. Inputs: data, any
        # array-like; dtype (keyword-only, default none), a supported timedelta
        # dtype (validated; integer-valued data are then interpreted in that
        # dtype's unit); copy (keyword-only, default false). Converts with the
        # sequence helper, then casts to dtype safely against overflow if given.
        # The frequency inferred from the input (if it was already an array with
        # a frequency) is kept. Returns the new array.
        unit = None
        if dtype:
            dtype = _validate_td64_dtype(dtype)
            if lib.infer_dtype(data) == "integer":
                unit = np.datetime_data(dtype)[0]

        data, freq = sequence_to_td64ns(data, copy=copy, unit=unit)

        if dtype is not None:
            data = astype_overflowsafe(data, dtype=dtype, copy=False)

        return cls._simple_new(data, dtype=data.dtype, freq=freq)

    @classmethod
    def _from_sequence_not_strict(
        cls,
        data,
        *,
        dtype=None,
        copy: bool = False,
        freq=lib.no_default,
        unit=None,
    ) -> Self:
        # [FUNCTION DOC]
        # Class method: like the strict sequence constructor but additionally
        # accepts freq and a unit for integer data, and leaves frequency discovery
        # to the caller. Inputs: data; dtype (validated; supplies the unit for
        # integer data when unit is none); copy; freq (default: not specified),
        # frequency to pin to the result; unit (default none), the unit integers
        # are multiples of. Asserts that unit is not a year or month unit (caller's
        # responsibility). Converts, optionally casts to dtype, builds the array
        # with the inferred frequency and then pins/validates the requested freq.
        # Returns the array.
        """
        _from_sequence_not_strict but without responsibility for finding the
        result's `freq`.
        """
        if dtype:
            dtype = _validate_td64_dtype(dtype)
            if unit is None and lib.infer_dtype(data) == "integer":
                unit = np.datetime_data(dtype)[0]

        assert unit not in ["Y", "y", "M"]  # caller is responsible for checking

        data, inferred_freq = sequence_to_td64ns(data, copy=copy, unit=unit)

        if dtype is not None:
            data = astype_overflowsafe(data, dtype=dtype, copy=False)

        result = cls._simple_new(data, dtype=data.dtype, freq=inferred_freq)

        result._maybe_pin_freq(freq, {})
        return result

    @classmethod
    def _generate_range(
        cls, start, end, periods, freq, closed=None, *, unit: TimeUnit
    ) -> Self:
        # [FUNCTION DOC]
        # Class method generating a regular range of timedeltas. Inputs: start,
        # end, periods, freq (any three of the four must be given, else a value
        # error); closed (default none: both ends included, or left/right);
        # unit (keyword-only), one of s, ms, us, ns (value error otherwise).
        # Start and end are converted to Timedelta and to the unit without
        # rounding (an error if rounding would be needed). With freq, the regular
        # range generator produces the values; without freq, values are evenly
        # spaced between start and end (linear spacing, integer truncated). Open
        # ends drop the first/last value. Returns an array with the freq.
        periods = dtl.validate_periods(periods)
        if freq is None and any(x is None for x in [periods, start, end]):
            raise ValueError("Must provide freq argument if no data is supplied")

        if com.count_not_none(start, end, periods, freq) != 3:
            raise ValueError(
                "Of the four parameters: start, end, periods, "
                "and freq, exactly three must be specified"
            )

        if start is not None:
            start = Timedelta(start).as_unit("ns")

        if end is not None:
            end = Timedelta(end).as_unit("ns")

        if unit not in ["s", "ms", "us", "ns"]:
            raise ValueError("'unit' must be one of 's', 'ms', 'us', 'ns'")

        if start is not None and unit is not None:
            start = start.as_unit(unit, round_ok=False)
        if end is not None and unit is not None:
            end = end.as_unit(unit, round_ok=False)

        left_closed, right_closed = validate_endpoints(closed)

        if freq is not None:
            index = generate_regular_range(start, end, periods, freq, unit=unit)
        else:
            index = np.linspace(start._value, end._value, periods).astype("i8")

        if not left_closed:
            index = index[1:]
        if not right_closed:
            index = index[:-1]

        td64values = index.view(f"m8[{unit}]")
        return cls._simple_new(td64values, dtype=td64values.dtype, freq=freq)

    # ----------------------------------------------------------------
    # DatetimeLike Interface

    def _unbox_scalar(self, value) -> np.timedelta64:
        # [FUNCTION DOC]
        # HELPER: converts a scalar to this array's numpy representation. Input
        # value: a Timedelta or the not-a-time marker; anything else raises a value
        # error. The not-a-time marker becomes a numpy missing timedelta in this
        # unit; a Timedelta is converted exactly to this unit (an error if it would
        # lose precision) and returned as a numpy scalar.
        if not isinstance(value, self._scalar_type) and value is not NaT:
            raise ValueError("'value' should be a Timedelta.")
        self._check_compatible_with(value)
        if value is NaT:
            return np.timedelta64(value._value, self.unit)
        else:
            return value.as_unit(self.unit, round_ok=False).asm8

    def _scalar_from_string(self, value) -> Timedelta | NaTType:
        # [FUNCTION DOC]
        # HELPER: parses a string into a Timedelta (or the not-a-time marker).
        return Timedelta(value)

    def _check_compatible_with(self, other) -> None:
        # [FUNCTION DOC]
        # HELPER: compatibility hook required by the base class; nothing to check
        # for timedeltas, so it does nothing.
        # we don't have anything to validate.
        pass

    # ----------------------------------------------------------------
    # Array-Like / EA-Interface Methods

    def astype(self, dtype, copy: bool = True):
        # [FUNCTION DOC]
        # Converts to another dtype. Inputs: dtype, anything convertible; copy
        # (default true). A timedelta dtype equal to the current one returns a copy
        # (or self). A different supported timedelta unit converts the values with
        # overflow checking, preserving the frequency. An unsupported timedelta
        # unit raises a value error. Everything else goes to the base class.
        # We handle
        #   --> timedelta64[ns]
        #   --> timedelta64
        # DatetimeLikeArrayMixin super call handles other cases
        dtype = pandas_dtype(dtype)

        if lib.is_np_dtype(dtype, "m"):
            if dtype == self.dtype:
                if copy:
                    return self.copy()
                return self

            if is_supported_dtype(dtype):
                # unit conversion e.g. timedelta64[s]
                res_values = astype_overflowsafe(self._ndarray, dtype, copy=False)
                return type(self)._simple_new(
                    res_values, dtype=res_values.dtype, freq=self.freq
                )
            else:
                raise ValueError(
                    f"Cannot convert from {self.dtype} to {dtype}. "
                    "Supported resolutions are 's', 'ms', 'us', 'ns'"
                )

        return dtl.DatetimeLikeArrayMixin.astype(self, dtype, copy=copy)

    def __iter__(self) -> Iterator:
        # [FUNCTION DOC]
        # Generator over elements. For multi-dimensional arrays yields the rows by
        # position. For one dimension, converts the integers to boxed Timedelta
        # objects in chunks of ten thousand and yields them one at a time (the
        # chunk count has one extra, possibly empty, chunk).
        if self.ndim > 1:
            for i in range(len(self)):
                yield self[i]
        else:
            # convert in chunks of 10k for efficiency
            data = self._ndarray
            length = len(self)
            chunksize = 10000
            chunks = (length // chunksize) + 1
            for i in range(chunks):
                start_i = i * chunksize
                end_i = min((i + 1) * chunksize, length)
                converted = ints_to_pytimedelta(data[start_i:end_i], box=True)
                yield from converted

    # ----------------------------------------------------------------
    # Reductions

    def sum(
        self,
        *,
        axis: AxisInt | None = None,
        dtype: NpDtype | None = None,
        out=None,
        keepdims: bool = False,
        initial=None,
        skipna: bool = True,
        min_count: int = 0,
    ):
        # [FUNCTION DOC]
        # Sum reduction. Inputs (all keyword-only): axis (default none: all);
        # dtype, out, keepdims, initial (numpy-compatibility arguments, validated
        # to be defaults); skipna (default true); min_count (default 0), minimum
        # number of non-missing values needed for a non-missing result. Computes
        # with the nan-aware sum on the integer values and boxes the outcome as a
        # Timedelta (axis none) or an array of timedeltas.
        nv.validate_sum(
            (), {"dtype": dtype, "out": out, "keepdims": keepdims, "initial": initial}
        )

        result = nanops.nansum(
            self._ndarray, axis=axis, skipna=skipna, min_count=min_count
        )
        return self._wrap_reduction_result(axis, result)

    def std(
        self,
        *,
        axis: AxisInt | None = None,
        dtype: NpDtype | None = None,
        out=None,
        ddof: int = 1,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        # [FUNCTION DOC]
        # Standard deviation reduction. Inputs (keyword-only): axis (default none),
        # dtype/out/keepdims (compatibility arguments, validated), ddof (default
        # 1), skipna (default true). Computes on the integer values; returns a
        # Timedelta for a full or one-dimensional reduction, otherwise a timedelta
        # array built from the result.
        nv.validate_stat_ddof_func(
            (), {"dtype": dtype, "out": out, "keepdims": keepdims}, fname="std"
        )

        result = nanops.nanstd(self._ndarray, axis=axis, skipna=skipna, ddof=ddof)
        if axis is None or self.ndim == 1:
            return self._box_func(result)
        return self._from_backing_data(result)

    # ----------------------------------------------------------------
    # Accumulations

    def _accumulate(self, name: str, *, skipna: bool = True, **kwargs):
        # [FUNCTION DOC]
        # Cumulative operations. Inputs: name; skipna (keyword-only, default true);
        # extra keywords for the underlying routine. Cumulative sum runs on a copy
        # of the data and returns a new array without a frequency. Cumulative
        # product raises a type error. Other names go to the base class.
        if name == "cumsum":
            op = getattr(datetimelike_accumulations, name)
            result = op(self._ndarray.copy(), skipna=skipna, **kwargs)

            return type(self)._simple_new(result, freq=None, dtype=self.dtype)
        elif name == "cumprod":
            raise TypeError("cumprod not supported for Timedelta.")

        else:
            return super()._accumulate(name, skipna=skipna, **kwargs)

    # ----------------------------------------------------------------
    # Rendering Methods

    def _formatter(self, boxed: bool = False):
        # [FUNCTION DOC]
        # Returns the element formatter used for display, delegating to the
        # timedelta formatting factory.
        # KNOWN ISSUE: the boxed argument is ignored; the factory is always called
        # with boxing enabled.
        from pandas.io.formats.format import get_format_timedelta64

        return get_format_timedelta64(self, box=True)

    def _format_native_types(
        self, *, na_rep: str | float = "NaT", date_format=None, **kwargs
    ) -> npt.NDArray[np.object_]:
        # [FUNCTION DOC]
        # Converts elements to text for output. Inputs (keyword-only): na_rep
        # (default "NaT"), text for missing values; date_format (unused); extra
        # keywords ignored. Applies the timedelta formatter element-wise over the
        # backing array (any dimensionality) and returns an object array of strings.
        from pandas.io.formats.format import get_format_timedelta64

        # Relies on TimeDelta._repr_base
        formatter = get_format_timedelta64(self, na_rep)
        # equiv: np.array([formatter(x) for x in self._ndarray])
        #  but independent of dimension
        return np.frompyfunc(formatter, 1, 1)(self._ndarray)

    # ----------------------------------------------------------------
    # Arithmetic Methods

    def _add_offset(self, other):
        # [FUNCTION DOC]
        # HELPER: addition of a non-fixed (calendar) offset is unsupported. Asserts
        # that other is not a tick or day offset, then always raises a type error
        # naming both types.
        assert not isinstance(other, (Tick, Day))
        raise TypeError(
            f"cannot add the type {type(other).__name__} to a {type(self).__name__}"
        )

    @unpack_zerodim_and_defer("__mul__")
    def __mul__(self, other) -> Self:
        # [FUNCTION DOC]
        # Multiplication by a number (also used as the reflected form). Input
        # other: scalar, list, array or extension array. Boolean operands raise a
        # type error (explicit cast required). A numeric scalar scales the values;
        # if numpy does not return a timedelta result a type error is raised; the
        # frequency is scaled too, unless the scalar is missing or the new
        # frequency would be zero. Sequences are converted to arrays; arrow or
        # masked-dtype arrays defer by returning the not-implemented marker; a
        # length mismatch (except for timedelta operands, which fail with a type
        # error) raises a value error; object arrays are multiplied element-wise;
        # other arrays use numpy and must yield a timedelta result (else type
        # error). Array results carry no frequency. Wrapped to defer on unsupported
        # operand types and unpack zero-dimensional arrays.
        if is_scalar(other):
            if lib.is_bool(other):
                raise TypeError(
                    f"Cannot multiply '{self.dtype}' by bool, explicitly cast to "
                    "integers instead"
                )
            # numpy will accept float and int, raise TypeError for others
            result = self._ndarray * other
            if result.dtype.kind != "m":
                # numpy >= 2.1 may not raise a TypeError
                # and seems to dispatch to others.__rmul__?
                raise TypeError(f"Cannot multiply with {type(other).__name__}")
            freq = None
            if self.freq is not None and not isna(other):
                freq = self.freq * other
                if freq.n == 0:
                    # GH#51575 Better to have no freq than an incorrect one
                    freq = None
            return type(self)._simple_new(result, dtype=result.dtype, freq=freq)

        if not hasattr(other, "dtype"):
            # list, tuple
            other = np.array(other)

        if other.dtype.kind == "b":
            # GH#58054
            raise TypeError(
                f"Cannot multiply '{self.dtype}' by bool, explicitly cast to "
                "integers instead"
            )
        if isinstance(other.dtype, (ArrowDtype, BaseMaskedDtype)):
            # GH#58054
            return NotImplemented

        if len(other) != len(self) and not lib.is_np_dtype(other.dtype, "m"):
            # Exclude timedelta64 here so we correctly raise TypeError
            #  for that instead of ValueError
            raise ValueError("Cannot multiply with unequal lengths")

        if is_object_dtype(other.dtype):
            # this multiplication will succeed only if all elements of other
            #  are int or float scalars, so we will end up with
            #  timedelta64[ns]-dtyped result
            arr = self._ndarray
            result = [arr[n] * other[n] for n in range(len(self))]
            result = np.array(result)
            return type(self)._simple_new(result, dtype=result.dtype)

        # numpy will accept float or int dtype, raise TypeError for others
        result = self._ndarray * other
        if result.dtype.kind != "m":
            # numpy >= 2.1 may not raise a TypeError
            # and seems to dispatch to others.__rmul__?
            raise TypeError(f"Cannot multiply with {type(other).__name__}")
        return type(self)._simple_new(result, dtype=result.dtype)

    __rmul__ = __mul__

    def _scalar_divlike_op(self, other, op):
        # [FUNCTION DOC]
        # HELPER: shared scalar division logic. Inputs: other, a scalar; op, one of
        # the division operator functions (true or floor, normal or reflected).
        # Timedelta-like scalars are converted to Timedelta: a missing one gives a
        # float array of NaN; otherwise numpy applies op to the values and the
        # scalar. Numeric scalars: reflected operations raise a type error; others
        # apply op to the values and the frequency is divided (true division even
        # for floor division; day frequencies stay exact when divisible, otherwise
        # converted to a finer offset); a frequency that collapses to zero is
        # dropped. Returns an array (timedelta for numeric divisors, numeric for
        # timedelta divisors).
        """
        Shared logic for __truediv__, __rtruediv__, __floordiv__, __rfloordiv__
        with scalar 'other'.
        """
        if isinstance(other, self._recognized_scalars):
            other = Timedelta(other)
            # mypy assumes that __new__ returns an instance of the class
            # github.com/python/mypy/issues/1020
            if cast("Timedelta | NaTType", other) is NaT:
                # specifically timedelta64-NaT
                res = np.empty(self.shape, dtype=np.float64)
                res.fill(np.nan)
                return res

            # otherwise, dispatch to Timedelta implementation
            return op(self._ndarray, other)

        else:
            # caller is responsible for checking lib.is_scalar(other)
            # assume other is numeric, otherwise numpy will raise

            if op in [roperator.rtruediv, roperator.rfloordiv]:
                raise TypeError(
                    f"Cannot divide {type(other).__name__} by {type(self).__name__}"
                )

            result = op(self._ndarray, other)
            freq = None

            if self.freq is not None:
                # Note: freq gets division, not floor-division, even if op
                #  is floordiv.
                if isinstance(self.freq, Day):
                    if self.freq.n % other == 0:
                        freq = Day(self.freq.n // other)
                    else:
                        freq = to_offset(Timedelta(days=self.freq.n)) / other
                else:
                    freq = self.freq / other
                if freq.nanos == 0 and self.freq.nanos != 0:
                    # e.g. if self.freq is Nano(1) then dividing by 2
                    #  rounds down to zero
                    freq = None

            return type(self)._simple_new(result, dtype=result.dtype, freq=freq)

    def _cast_divlike_op(self, other):
        # [FUNCTION DOC]
        # HELPER: prepares a non-scalar divisor/dividend. Converts lists and
        # tuples to arrays and raises a value error if its length differs from this
        # array's. Returns the array-like.
        if not hasattr(other, "dtype"):
            # e.g. list, tuple
            other = np.array(other)

        if len(other) != len(self):
            raise ValueError("Cannot divide vectors with unequal lengths")
        return other

    def _vector_divlike_op(self, other, op) -> np.ndarray | Self:
        # [FUNCTION DOC]
        # HELPER: shared array division logic. Inputs: other, an array with
        # timedelta, integer or float dtype; op, a division operator function.
        # Numpy applies op. Division of timedeltas by integers or floats (true or
        # floor, non-reflected) gives a timedelta array without a frequency. For
        # floor division (normal or reflected), positions that are missing in either
        # operand become NaN in a float result. Otherwise returns the numpy result.
        """
        Shared logic for __truediv__, __floordiv__, and their reversed versions
        with timedelta64-dtype ndarray other.
        """
        # Let numpy handle it
        result = op(self._ndarray, np.asarray(other))

        if (is_integer_dtype(other.dtype) or is_float_dtype(other.dtype)) and op in [
            operator.truediv,
            operator.floordiv,
        ]:
            return type(self)._simple_new(result, dtype=result.dtype)

        if op in [operator.floordiv, roperator.rfloordiv]:
            mask = self.isna() | isna(other)
            if mask.any():
                result = result.astype(np.float64)
                np.putmask(result, mask, np.nan)

        return result

    @unpack_zerodim_and_defer("__truediv__")
    def __truediv__(self, other):
        # [FUNCTION DOC]
        # True division by a scalar or array. Scalars use the scalar helper. Arrays
        # of timedelta, integer or float dtype use the vector helper (floats and
        # integers yield timedeltas, timedeltas yield floats). Object arrays divide
        # element-wise (row by row for multi-dimensional arrays, otherwise via a
        # specialised routine). Anything else returns the not-implemented marker.
        # Length mismatches raise a value error. Wrapped to unpack zero-dimensional
        # arrays and defer to unsupported types.
        # timedelta / X is well-defined for timedelta-like or numeric X
        op = operator.truediv
        if is_scalar(other):
            return self._scalar_divlike_op(other, op)

        other = self._cast_divlike_op(other)
        if (
            lib.is_np_dtype(other.dtype, "m")
            or is_integer_dtype(other.dtype)
            or is_float_dtype(other.dtype)
        ):
            return self._vector_divlike_op(other, op)

        if is_object_dtype(other.dtype):
            other = np.asarray(other)
            if self.ndim > 1:
                res_cols = [
                    left / right for left, right in zip(self, other, strict=True)
                ]
                res_cols2 = [x.reshape(1, -1) for x in res_cols]
                result = np.concatenate(res_cols2, axis=0)
            else:
                result = truediv_object_array(self._ndarray, other)

            return result

        else:
            return NotImplemented

    @unpack_zerodim_and_defer("__rtruediv__")
    def __rtruediv__(self, other):
        # [FUNCTION DOC]
        # Reflected true division (other divided by this array); defined only for
        # timedelta-like numerators. Scalars use the scalar helper (numeric
        # scalars raise a type error); timedelta arrays use the vector helper;
        # object arrays divide element-wise into a numeric array; anything else
        # returns the not-implemented marker. Length mismatches raise a value error.
        # X / timedelta is defined only for timedelta-like X
        op = roperator.rtruediv
        if is_scalar(other):
            return self._scalar_divlike_op(other, op)

        other = self._cast_divlike_op(other)
        if lib.is_np_dtype(other.dtype, "m"):
            return self._vector_divlike_op(other, op)

        elif is_object_dtype(other.dtype):
            # Note: unlike in __truediv__, we do not _need_ to do type
            #  inference on the result.  It does not raise, a numeric array
            #  is returned.  GH#23829
            result_list = [other[n] / self[n] for n in range(len(self))]
            return np.array(result_list)

        else:
            return NotImplemented

    @unpack_zerodim_and_defer("__floordiv__")
    def __floordiv__(self, other):
        # [FUNCTION DOC]
        # Floor division by a scalar or array; same dispatch as true division, but
        # object arrays use the specialised floor-division routine (or row-wise
        # division) and must yield an object array (asserted). Timedelta divisors
        # produce integers (or floats with NaN where either side is missing);
        # integer or float divisors produce timedeltas.
        op = operator.floordiv
        if is_scalar(other):
            return self._scalar_divlike_op(other, op)

        other = self._cast_divlike_op(other)
        if (
            lib.is_np_dtype(other.dtype, "m")
            or is_integer_dtype(other.dtype)
            or is_float_dtype(other.dtype)
        ):
            return self._vector_divlike_op(other, op)

        elif is_object_dtype(other.dtype):
            other = np.asarray(other)
            if self.ndim > 1:
                res_cols = [
                    left // right for left, right in zip(self, other, strict=True)
                ]
                res_cols2 = [x.reshape(1, -1) for x in res_cols]
                result = np.concatenate(res_cols2, axis=0)
            else:
                result = floordiv_object_array(self._ndarray, other)

            assert result.dtype == object
            return result

        else:
            return NotImplemented

    @unpack_zerodim_and_defer("__rfloordiv__")
    def __rfloordiv__(self, other):
        # [FUNCTION DOC]
        # Reflected floor division (other floor-divided by this array); defined
        # only for timedelta-like numerators: scalars via the scalar helper, timedelta
        # arrays via the vector helper, object arrays element-wise into a numpy
        # array; anything else returns the not-implemented marker.
        op = roperator.rfloordiv
        if is_scalar(other):
            return self._scalar_divlike_op(other, op)

        other = self._cast_divlike_op(other)
        if lib.is_np_dtype(other.dtype, "m"):
            return self._vector_divlike_op(other, op)

        elif is_object_dtype(other.dtype):
            result_list = [other[n] // self[n] for n in range(len(self))]
            result = np.array(result_list)
            return result

        else:
            return NotImplemented

    @unpack_zerodim_and_defer("__mod__")
    def __mod__(self, other):
        # [FUNCTION DOC]
        # Remainder of this array divided by other: recognised scalars are first
        # converted to Timedelta, then the result is this array minus the floor
        # quotient times other (naive implementation). Returns a timedelta array.
        # Note: This is a naive implementation, can likely be optimized
        if isinstance(other, self._recognized_scalars):
            other = Timedelta(other)
        return self - (self // other) * other

    @unpack_zerodim_and_defer("__rmod__")
    def __rmod__(self, other):
        # [FUNCTION DOC]
        # Reflected remainder (other modulo this array): recognised scalars are
        # converted to Timedelta, then computed as other minus the floor quotient
        # times this array (naive implementation).
        # Note: This is a naive implementation, can likely be optimized
        if isinstance(other, self._recognized_scalars):
            other = Timedelta(other)
        return other - (other // self) * self

    @unpack_zerodim_and_defer("__divmod__")
    def __divmod__(self, other):
        # [FUNCTION DOC]
        # Quotient and remainder of this array by other (recognised scalars
        # converted to Timedelta). Returns the pair (floor quotient, remainder),
        # computed naively from floor division and multiplication.
        # Note: This is a naive implementation, can likely be optimized
        if isinstance(other, self._recognized_scalars):
            other = Timedelta(other)

        res1 = self // other
        res2 = self - res1 * other
        return res1, res2

    @unpack_zerodim_and_defer("__rdivmod__")
    def __rdivmod__(self, other):
        # [FUNCTION DOC]
        # Reflected divmod: quotient and remainder of other by this array
        # (recognised scalars converted to Timedelta). Returns the pair (floor
        # quotient, remainder).
        # Note: This is a naive implementation, can likely be optimized
        if isinstance(other, self._recognized_scalars):
            other = Timedelta(other)

        res1 = other // self
        res2 = other - res1 * self
        return res1, res2

    def __neg__(self) -> TimedeltaArray:
        # [FUNCTION DOC]
        # Element-wise negation; returns a new array whose frequency (if any) is
        # negated.
        freq = None
        if self.freq is not None:
            freq = -self.freq
        return type(self)._simple_new(-self._ndarray, dtype=self.dtype, freq=freq)

    def __pos__(self) -> TimedeltaArray:
        # [FUNCTION DOC]
        # Unary plus; returns a new array with a copy of the data and the same
        # frequency.
        return type(self)._simple_new(
            self._ndarray.copy(), dtype=self.dtype, freq=self.freq
        )

    def __abs__(self) -> TimedeltaArray:
        # [FUNCTION DOC]
        # Element-wise absolute value; returns a new array. The frequency is not
        # preserved.
        # Note: freq is not preserved
        return type(self)._simple_new(np.abs(self._ndarray), dtype=self.dtype)

    # ----------------------------------------------------------------
    # Conversion Methods - Vectorized analogues of Timedelta methods

    def total_seconds(self) -> npt.NDArray[np.float64]:
        # [FUNCTION DOC]
        # Returns each duration in seconds as floating-point numbers, computed from
        # the integer values and the number of periods per second at this unit;
        # missing elements become NaN. The result is a numpy array for this class.
        """
        Return total duration of each element expressed in seconds.

        This method is available directly on TimedeltaArray, TimedeltaIndex
        and on Series containing timedelta values under the ``.dt`` namespace.

        Returns
        -------
        ndarray, Index or Series
            When the calling object is a TimedeltaArray, the return type
            is ndarray.  When the calling object is a TimedeltaIndex,
            the return type is an Index with a float64 dtype. When the calling object
            is a Series, the return type is Series of type `float64` whose
            index is the same as the original.

        See Also
        --------
        datetime.timedelta.total_seconds : Standard library version
            of this method.
        TimedeltaIndex.components : Return a DataFrame with components of
            each Timedelta.

        Examples
        --------
        **Series**

        >>> s = pd.Series(pd.to_timedelta(np.arange(5), unit="D"))
        >>> s
        0   0 days
        1   1 days
        2   2 days
        3   3 days
        4   4 days
        dtype: timedelta64[s]

        >>> s.dt.total_seconds()
        0         0.0
        1     86400.0
        2    172800.0
        3    259200.0
        4    345600.0
        dtype: float64

        **TimedeltaIndex**

        >>> idx = pd.to_timedelta(np.arange(5), unit="D")
        >>> idx
        TimedeltaIndex(['0 days', '1 days', '2 days', '3 days', '4 days'],
                       dtype='timedelta64[s]', freq=None)

        >>> idx.total_seconds()
        Index([0.0, 86400.0, 172800.0, 259200.0, 345600.0], dtype='float64')
        """
        pps = periods_per_second(self._creso)
        return self._maybe_mask_results(self.asi8 / pps, fill_value=None)

    def to_pytimedelta(self) -> npt.NDArray[np.object_]:
        # [FUNCTION DOC]
        # Returns an object array of standard-library timedelta objects, one per
        # element, converted from the integer values (missing values become the
        # not-a-time marker).
        # KNOWN ISSUE: the docstring's Returns section describes a numpy
        # timedelta64 value with nanosecond precision, but the method returns an
        # object-dtype array of standard timedelta objects at their native
        # (microsecond-limited) precision.
        """
        Return an ndarray of datetime.timedelta objects.

        Returns
        -------
        numpy.ndarray
            A NumPy ``timedelta64`` object representing the same duration as the
            original pandas ``Timedelta`` object. The precision of the resulting
            object is in nanoseconds, which is the default
            time resolution used by pandas for ``Timedelta`` objects, ensuring
            high precision for time-based calculations.

        See Also
        --------
        to_timedelta : Convert argument to timedelta format.
        Timedelta : Represents a duration between two dates or times.
        DatetimeIndex: Index of datetime64 data.
        Timedelta.components : Return a components namedtuple-like
                               of a single timedelta.

        Examples
        --------
        >>> tdelta_idx = pd.to_timedelta([1, 2, 3], unit="D")
        >>> tdelta_idx
        TimedeltaIndex(['1 days', '2 days', '3 days'],
                        dtype='timedelta64[s]', freq=None)
        >>> tdelta_idx.to_pytimedelta()
        array([datetime.timedelta(days=1), datetime.timedelta(days=2),
               datetime.timedelta(days=3)], dtype=object)

        >>> tidx = pd.TimedeltaIndex(data=["1 days 02:30:45", "3 days 04:15:10"])
        >>> tidx
        TimedeltaIndex(['1 days 02:30:45', '3 days 04:15:10'],
               dtype='timedelta64[us]', freq=None)
        >>> tidx.to_pytimedelta()
        array([datetime.timedelta(days=1, seconds=9045),
                datetime.timedelta(days=3, seconds=15310)], dtype=object)
        """
        return ints_to_pytimedelta(self._ndarray)

    days_docstring = textwrap.dedent(
        """Number of days for each element.

    See Also
    --------
    Series.dt.seconds : Return number of seconds for each element.
    Series.dt.microseconds : Return number of microseconds for each element.
    Series.dt.nanoseconds : Return number of nanoseconds for each element.

    Examples
    --------
    For Series:

    >>> ser = pd.Series(pd.to_timedelta([1, 2, 3], unit='D'))
    >>> ser
    0   1 days
    1   2 days
    2   3 days
    dtype: timedelta64[s]
    >>> ser.dt.days
    0    1
    1    2
    2    3
    dtype: int64

    For TimedeltaIndex:

    >>> tdelta_idx = pd.to_timedelta(["0 days", "10 days", "20 days"])
    >>> tdelta_idx
    TimedeltaIndex(['0 days', '10 days', '20 days'],
                    dtype='timedelta64[us]', freq=None)
    >>> tdelta_idx.days
    Index([0, 10, 20], dtype='int64')"""
    )
    days = _field_accessor("days", "days", days_docstring)

    seconds_docstring = textwrap.dedent(
        """Number of seconds (>= 0 and less than 1 day) for each element.

    See Also
    --------
    Series.dt.seconds : Return number of seconds for each element.
    Series.dt.nanoseconds : Return number of nanoseconds for each element.

    Examples
    --------
    For Series:

    >>> ser = pd.Series(pd.to_timedelta([1, 2, 3], unit='s'))
    >>> ser
    0   0 days 00:00:01
    1   0 days 00:00:02
    2   0 days 00:00:03
    dtype: timedelta64[s]
    >>> ser.dt.seconds
    0    1
    1    2
    2    3
    dtype: int32

    For TimedeltaIndex:

    >>> tdelta_idx = pd.to_timedelta([1, 2, 3], unit='s')
    >>> tdelta_idx
    TimedeltaIndex(['0 days 00:00:01', '0 days 00:00:02', '0 days 00:00:03'],
                   dtype='timedelta64[s]', freq=None)
    >>> tdelta_idx.seconds
    Index([1, 2, 3], dtype='int32')"""
    )
    seconds = _field_accessor(
        "seconds",
        "seconds",
        seconds_docstring,
    )

    microseconds_docstring = textwrap.dedent(
        """Number of microseconds (>= 0 and less than 1 second) for each element.

    See Also
    --------
    pd.Timedelta.microseconds : Number of microseconds (>= 0 and less than 1 second).
    pd.Timedelta.to_pytimedelta.microseconds : Number of microseconds (>= 0 and less
        than 1 second) of a datetime.timedelta.

    Examples
    --------
    For Series:

    >>> ser = pd.Series(pd.to_timedelta([1, 2, 3], unit='us'))
    >>> ser
    0   0 days 00:00:00.000001
    1   0 days 00:00:00.000002
    2   0 days 00:00:00.000003
    dtype: timedelta64[us]
    >>> ser.dt.microseconds
    0    1
    1    2
    2    3
    dtype: int32

    For TimedeltaIndex:

    >>> tdelta_idx = pd.to_timedelta([1, 2, 3], unit='us')
    >>> tdelta_idx
    TimedeltaIndex(['0 days 00:00:00.000001', '0 days 00:00:00.000002',
                    '0 days 00:00:00.000003'],
                   dtype='timedelta64[us]', freq=None)
    >>> tdelta_idx.microseconds
    Index([1, 2, 3], dtype='int32')"""
    )
    microseconds = _field_accessor(
        "microseconds",
        "microseconds",
        microseconds_docstring,
    )

    nanoseconds_docstring = textwrap.dedent(
        """Number of nanoseconds (>= 0 and less than 1 microsecond) for each element.

    See Also
    --------
    Series.dt.seconds : Return number of seconds for each element.
    Series.dt.microseconds : Return number of nanoseconds for each element.

    Examples
    --------
    For Series:

    >>> ser = pd.Series(pd.to_timedelta([1, 2, 3], unit='ns'))
    >>> ser
    0   0 days 00:00:00.000000001
    1   0 days 00:00:00.000000002
    2   0 days 00:00:00.000000003
    dtype: timedelta64[ns]
    >>> ser.dt.nanoseconds
    0    1
    1    2
    2    3
    dtype: int32

    For TimedeltaIndex:

    >>> tdelta_idx = pd.to_timedelta([1, 2, 3], unit='ns')
    >>> tdelta_idx
    TimedeltaIndex(['0 days 00:00:00.000000001', '0 days 00:00:00.000000002',
                    '0 days 00:00:00.000000003'],
                   dtype='timedelta64[ns]', freq=None)
    >>> tdelta_idx.nanoseconds
    Index([1, 2, 3], dtype='int32')"""
    )
    nanoseconds = _field_accessor(
        "nanoseconds",
        "nanoseconds",
        nanoseconds_docstring,
    )

    @property
    def components(self) -> DataFrame:
        # [FUNCTION DOC]
        # Property: a data frame with one row per element and columns days, hours,
        # minutes, seconds, milliseconds, microseconds, nanoseconds. Each element's
        # components are taken from its Timedelta; missing elements give NaN in all
        # columns and the frame is floating-point; with no missing values the
        # columns are cast to 64-bit integers.
        """
        Return a DataFrame of the individual resolution components of the Timedeltas.

        The components (days, hours, minutes seconds, milliseconds, microseconds,
        nanoseconds) are returned as columns in a DataFrame.

        Returns
        -------
        DataFrame

        See Also
        --------
        TimedeltaIndex.total_seconds : Return total duration expressed in seconds.
        Timedelta.components : Return a components namedtuple-like of a single
            timedelta.

        Examples
        --------
        >>> tdelta_idx = pd.to_timedelta(["1 day 3 min 2 us 42 ns"])
        >>> tdelta_idx
        TimedeltaIndex(['1 days 00:03:00.000002042'],
                       dtype='timedelta64[ns]', freq=None)
        >>> tdelta_idx.components
           days  hours  minutes  seconds  milliseconds  microseconds  nanoseconds
        0     1      0        3        0             0             2           42
        """
        from pandas import DataFrame

        columns = [
            "days",
            "hours",
            "minutes",
            "seconds",
            "milliseconds",
            "microseconds",
            "nanoseconds",
        ]
        hasnans = self._hasna
        if hasnans:

            def f(x):
                # [FUNCTION DOC]
                # HELPER: component list of one element, or all-NaN for a missing
                # element (used when the array has missing values).
                if isna(x):
                    return [np.nan] * len(columns)
                return x.components

        else:

            def f(x):
                # [FUNCTION DOC]
                # HELPER: component list of one element (used when the array has
                # no missing values).
                return x.components

        result = DataFrame([f(x) for x in self], columns=columns)
        if not hasnans:
            result = result.astype("int64")
        return result


# ---------------------------------------------------------------------
# Constructor Helpers


def sequence_to_td64ns(
    data,
    copy: bool = False,
    unit=None,
    errors: DateTimeErrorChoices = "raise",
) -> tuple[np.ndarray, Tick | Day | None]:
    # [FUNCTION DOC]
    # Converts many input kinds to a timedelta64 numpy array. Inputs: data,
    # list-like; copy (default false); unit (default none), the unit numeric data
    # are multiples of (disallowed with string data when errors are raised;
    # year and month units fail an assertion); errors, "raise" (default), "coerce"
    # or "ignore" (passed to the object parser; the caller may ignore errors).
    # Returns (array, inferred frequency): the frequency is taken from the input
    # if it was already a timedelta array.
    # Behaviour by input dtype: object or string data are parsed; integers are
    # scaled by the unit; floats are converted with separate whole/fractional
    # handling to limit precision loss (if a non-nanosecond unit is requested and
    # all non-missing floats are whole numbers they are treated as integers and
    # missing values are set to the missing marker); timedelta arrays in an
    # unsupported unit are converted to the closest supported unit; any other
    # dtype (such as datetimes) raises a type error. The result always has a
    # supported timedelta unit and a copy is made only if requested and not
    # already made.
    # KNOWN ISSUE: the docstring says the result has nanosecond resolution, but
    # integer data with a coarser unit and timedelta input keep (or convert to) a
    # coarser supported unit such as seconds, so the dtype is not always
    # nanosecond.
    """
    Parameters
    ----------
    data : list-like
    copy : bool, default False
    unit : str, optional
        The timedelta unit to treat integers as multiples of. For numeric
        data this defaults to ``'ns'``.
        Must be un-specified if the data contains a str and ``errors=="raise"``.
    errors : {"raise", "coerce", "ignore"}, default "raise"
        How to handle elements that cannot be converted to timedelta64[ns].
        See ``pandas.to_timedelta`` for details.

    Returns
    -------
    converted : numpy.ndarray
        The sequence converted to a numpy array with dtype ``timedelta64[ns]``.
    inferred_freq : Tick, Day, or None
        The inferred frequency of the sequence.

    Raises
    ------
    ValueError : Data cannot be converted to timedelta64[ns].

    Notes
    -----
    Unlike `pandas.to_timedelta`, if setting ``errors=ignore`` will not cause
    errors to be ignored; they are caught and subsequently ignored at a
    higher level.
    """
    assert unit not in ["Y", "y", "M"]  # caller is responsible for checking

    inferred_freq = None
    if unit is not None:
        unit = parse_timedelta_unit(unit)

    data, copy = dtl.ensure_arraylike_for_datetimelike(
        data, copy, cls_name="TimedeltaArray"
    )

    if isinstance(data, TimedeltaArray):
        inferred_freq = data.freq

    # Convert whatever we have into timedelta64[ns] dtype
    if data.dtype == object or is_string_dtype(data.dtype):
        # no need to make a copy, need to convert if string-dtyped
        data = _objects_to_td64ns(data, unit=unit, errors=errors)
        copy = False

    elif is_integer_dtype(data.dtype):
        # treat as multiples of the given unit
        data, copy_made = _ints_to_td64ns(data, unit=unit)
        copy = copy and not copy_made

    elif is_float_dtype(data.dtype):
        # cast the unit, multiply base/frac separately
        # to avoid precision issues from float -> int
        if isinstance(data.dtype, ExtensionDtype):
            mask = data._mask
            data = data._data
        else:
            mask = np.isnan(data)

        if unit is not None and unit != "ns":
            # if all non-NaN entries are round, treat these like ints and give
            #  back the requested unit (or closest-supported)
            with np.errstate(invalid="ignore"):
                int_data = data.astype(np.int64)
            all_round = (mask | (data == int_data)).all()
            if all_round:
                result, _ = sequence_to_td64ns(
                    int_data, copy=False, unit=unit, errors=errors
                )
                result[mask] = iNaT
                return result, inferred_freq

        data = cast_from_unit_vectorized(data, unit or "ns")
        data[mask] = iNaT
        data = data.view("m8[ns]")
        copy = False

    elif lib.is_np_dtype(data.dtype, "m"):
        if not is_supported_dtype(data.dtype):
            # cast to closest supported unit, i.e. s or ns
            new_dtype = get_supported_dtype(data.dtype)
            data = astype_overflowsafe(data, dtype=new_dtype, copy=False)
            copy = False

    else:
        # This includes datetime64-dtype, see GH#23539, GH#29794
        raise TypeError(f"dtype {data.dtype} cannot be converted to timedelta64[ns]")

    if not copy:
        data = np.asarray(data)
    else:
        data = np.array(data, copy=copy)

    assert data.dtype.kind == "m"
    assert data.dtype != "m8"  # i.e. not unit-less

    return data, inferred_freq


def _ints_to_td64ns(data, unit: str = "ns") -> tuple[np.ndarray, bool]:
    # [FUNCTION DOC]
    # HELPER: integer conversion for the sequence helper. Inputs: data, integer
    # array; unit (default "ns"; none is treated as nanoseconds). Casts to 64-bit
    # integers if needed (a copy), reinterprets as timedelta64 in the unit, and if
    # that unit is unsupported converts it to the closest supported one (another
    # copy). Returns (timedelta array, whether a copy was made).
    # KNOWN ISSUE: the docstring and return description say the result is
    # nanosecond resolution, but non-nanosecond units are kept (or converted only
    # to the nearest supported unit).
    """
    Convert an ndarray with integer-dtype to timedelta64[ns] dtype, treating
    the integers as multiples of the given timedelta unit.

    Parameters
    ----------
    data : numpy.ndarray with integer-dtype
    unit : str, default "ns"
        The timedelta unit to treat integers as multiples of.

    Returns
    -------
    numpy.ndarray : timedelta64[ns] array converted from data
    bool : whether a copy was made
    """
    copy_made = False
    unit = unit if unit is not None else "ns"

    if data.dtype != np.int64:
        # converting to int64 makes a copy, so we can avoid
        # re-copying later
        data = data.astype(np.int64)
        copy_made = True

    if unit != "ns":
        dtype_str = f"timedelta64[{unit}]"
        data = data.view(dtype_str)

        new_dtype = get_supported_dtype(data.dtype)
        if new_dtype != data.dtype:
            data = astype_overflowsafe(data, dtype=new_dtype)

            # the astype conversion makes a copy, so we can avoid re-copying later
            copy_made = True

    else:
        data = data.view("timedelta64[ns]")

    return data, copy_made


def _objects_to_td64ns(
    data, unit=None, errors: DateTimeErrorChoices = "raise"
) -> np.ndarray:
    # [FUNCTION DOC]
    # HELPER: parses object-dtype or string-dtype data. Inputs: data, array or
    # index; unit (default none), the unit for numeric items (must not be given if
    # strings are present); errors, "raise", "coerce" or "ignore" forwarded to the
    # parser. Converts data to an object array and parses it into a timedelta64
    # array. Parsing failures raise a value error (when errors are raised).
    """
    Convert an object-dtyped or string-dtyped array into a
    timedelta64[ns]-dtyped array.

    Parameters
    ----------
    data : ndarray or Index
    unit : str, default "ns"
        The timedelta unit to treat integers as multiples of.
        Must not be specified if the data contains a str.
    errors : {"raise", "coerce", "ignore"}, default "raise"
        How to handle elements that cannot be converted to timedelta64[ns].
        See ``pandas.to_timedelta`` for details.

    Returns
    -------
    numpy.ndarray : timedelta64[ns] array converted from data

    Raises
    ------
    ValueError : Data cannot be converted to timedelta64[ns].

    Notes
    -----
    Unlike `pandas.to_timedelta`, if setting `errors=ignore` will not cause
    errors to be ignored; they are caught and subsequently ignored at a
    higher level.
    """
    # coerce Index to np.ndarray, converting string-dtype if necessary
    values = np.asarray(data, dtype=np.object_)

    result = array_to_timedelta64(values, unit=unit, errors=errors)
    return result


def _validate_td64_dtype(dtype) -> DtypeObj:
    # [FUNCTION DOC]
    # HELPER: validates a requested timedelta dtype. Input dtype: anything
    # convertible to a dtype. Raises a value error for a unit-less timedelta
    # dtype, for non-timedelta dtypes, and for timedelta units other than
    # seconds, milliseconds, microseconds and nanoseconds. Returns the resolved
    # dtype.
    dtype = pandas_dtype(dtype)
    if dtype == np.dtype("m8"):
        # no precision disallowed GH#24806
        msg = (
            "Passing in 'timedelta' dtype with no precision is not allowed. "
            "Please pass in 'timedelta64[ns]' instead."
        )
        raise ValueError(msg)

    if not lib.is_np_dtype(dtype, "m"):
        raise ValueError(f"dtype '{dtype}' is invalid, should be np.timedelta64 dtype")
    elif not is_supported_dtype(dtype):
        raise ValueError("Supported timedelta64 resolutions are 's', 'ms', 'us', 'ns'")

    return dtype
