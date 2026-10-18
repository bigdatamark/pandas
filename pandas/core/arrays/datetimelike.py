# [MODULE DOC]
# Shared base classes for the three date/time-like extension arrays (datetime,
# time delta and period). Public contents: the mixin that supplies behavior
# common to all three (boxing/unboxing of scalars, indexing that tracks the
# frequency, validation of scalars and list-likes, comparison and arithmetic
# operators, null handling, reductions, shifting, repetition, unique/factorize
# style helpers), the datelike mixin (string formatting and strftime), the
# timelike mixin (frequency validation, resolution units, rounding/flooring/
# ceiling, conversion between units), and module-level helpers: a frequency
# validator/pinning hook, a unit-from-dtype resolver, a period-count validator,
# array-like coercion for constructors, a decorator that flattens
# multi-dimensional arrays, plus a few helpers for invalid operations and
# period dispatch.
# Relationships: concrete arrays live in the datetime, timedelta and period
# modules and derive from these mixins; this module builds on the
# numpy-backed extension array base and the operator mixin; it relies on the
# compiled time-series library for scalar types, offsets and overflow-safe
# arithmetic.
# Import-time side effects: none beyond defining names.
from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
)
from functools import wraps
import operator
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
    TypeAlias,
    Union,
    cast,
    final,
    overload,
)
import warnings

import numpy as np

from pandas._config import using_string_dtype
from pandas._config.config import get_option

from pandas._libs import (
    algos,
    lib,
)
from pandas._libs.tslibs import (
    BaseOffset,
    Day,
    IncompatibleFrequency,
    NaT,
    NaTType,
    Period,
    Resolution,
    Tick,
    Timedelta,
    Timestamp,
    add_overflowsafe,
    astype_overflowsafe,
    get_unit_from_dtype,
    iNaT,
    ints_to_pydatetime,
    ints_to_pytimedelta,
    periods_per_day,
    timezones,
    to_offset,
)
from pandas._libs.tslibs.fields import (
    RoundTo,
    round_nsint64,
)
from pandas._libs.tslibs.np_datetime import compare_mismatched_resolutions
from pandas._libs.tslibs.timedeltas import get_unit_for_round
from pandas._libs.tslibs.timestamps import integer_op_not_supported
from pandas._typing import (
    ArrayLike,
    AxisInt,
    DatetimeLikeScalar,
    Dtype,
    DtypeObj,
    F,
    InterpolateOptions,
    NpDtype,
    PositionalIndexer2D,
    PositionalIndexerTuple,
    ScalarIndexer,
    SequenceIndexer,
    TakeIndexer,
    TimeAmbiguous,
    TimeNonexistent,
    npt,
)
from pandas.compat.numpy import function as nv
from pandas.errors import (
    AbstractMethodError,
    InvalidComparison,
    PerformanceWarning,
)
from pandas.util._decorators import (
    cache_readonly,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import (
    is_all_strings,
    is_integer_dtype,
    is_list_like,
    is_object_dtype,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    CategoricalDtype,
    DatetimeTZDtype,
    ExtensionDtype,
    PeriodDtype,
)
from pandas.core.dtypes.generic import (
    ABCCategorical,
    ABCMultiIndex,
)
from pandas.core.dtypes.missing import (
    is_valid_na_for_dtype,
    isna,
)

from pandas.core import (
    algorithms,
    missing,
    nanops,
    ops,
)
from pandas.core.algorithms import (
    isin,
    map_array,
    unique1d,
)
from pandas.core.array_algos import datetimelike_accumulations
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._mixins import (
    NDArrayBackedExtensionArray,
    ravel_compat,
)
from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.integer import IntegerArray
import pandas.core.common as com
from pandas.core.construction import (
    array as pd_array,
    ensure_wrapped_if_datetimelike,
    extract_array,
)
from pandas.core.indexers import (
    check_array_indexer,
    check_setitem_lengths,
)
from pandas.core.ops.common import unpack_zerodim_and_defer
from pandas.core.ops.invalid import (
    invalid_comparison,
    make_invalid_op,
)

from pandas.tseries import frequencies

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterator,
        Sequence,
    )

    from pandas._typing import TimeUnit

    from pandas import Index
    from pandas.core.arrays import (
        DatetimeArray,
        PeriodArray,
        TimedeltaArray,
    )

DTScalarOrNaT: TypeAlias = DatetimeLikeScalar | NaTType


def _make_unpacked_invalid_op(op_name: str):
    # [FUNCTION DOC]
    # HELPER: builds an operator implementation that always fails. Parameter
    # op_name is the operator's name used in the error message. Creates the
    # invalid-operation function (which raises a type error saying the
    # operation is not supported for the type) and wraps it so that zero-
    # dimensional arrays are unpacked and unsupported right-hand sides defer to
    # the other operand. Returns the wrapped function, meant to be assigned as
    # a method on the mixin classes.
    op = make_invalid_op(op_name)
    return unpack_zerodim_and_defer(op_name)(op)


def _period_dispatch(meth: F) -> F:
    # [FUNCTION DOC]
    # HELPER: decorator factory-style wrapper used on mixin methods so that
    # period arrays can reuse datetime implementations. Parameter meth is the
    # method to wrap. Returns a wrapper with the same signature (function
    # metadata copied). No side effects at decoration time.
    """
    For PeriodArray methods, dispatch to DatetimeArray and re-wrap the results
    in PeriodArray.  We cannot use ._ndarray directly for the affected
    methods because the i8 data has different semantics on NaT values.
    """

    @wraps(meth)
    def new_meth(self, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: wrapper produced by the period dispatch decorator. Arguments
        # are forwarded unchanged. For non-period arrays it just calls the
        # wrapped method. For period arrays it views the data as
        # nanosecond datetimes, calls the wrapped method on that view, and maps
        # the result back: a missing marker stays missing, a single timestamp
        # is re-boxed as a period scalar from its integer value, and any array
        # result is viewed as integers and rebuilt as an array of the original
        # period type. Returns that converted result.
        if not isinstance(self.dtype, PeriodDtype):
            return meth(self, *args, **kwargs)

        arr = self.view("M8[ns]")
        result = meth(arr, *args, **kwargs)
        if result is NaT:
            return NaT
        elif isinstance(result, Timestamp):
            return self._box_func(result._value)

        res_i8 = result.view("i8")
        return self._from_backing_data(res_i8)

    return cast(F, new_meth)


class DatetimeLikeArrayMixin(OpsMixin, NDArrayBackedExtensionArray):
    # [CLASS DOC]
    # Common base of the datetime, time delta and period arrays. Combines the
    # operator-dispatch mixin with the numpy-backed extension array base.
    # State: the backing numpy array (holding 64-bit-integer-like values where
    # the minimum integer encodes missing), and a frequency attribute that
    # subclasses must provide (an offset or none). Class-level declarations that
    # subclasses fill in: which type-inference labels count as compatible, a
    # predicate for recognized dtypes, the tuple of recognized scalar types.
    # Many methods here are abstract hooks that raise an abstract-method error
    # (constructor, scalar type, scalar parsing, unboxing, compatibility
    # check, boxing function, native-type formatting). Concrete behavior:
    # indexing with frequency propagation, setitem with frequency
    # invalidation, astype, validation helpers, comparison and arithmetic
    # operators, reductions, shifting, repeating, and conversion helpers.
    # Invariant: the stored integer view equals the minimum 64-bit integer
    # exactly where a value is missing.
    """
    Shared Base/Mixin class for DatetimeArray, TimedeltaArray, PeriodArray

    Assumes that __new__/__init__ defines:
        _ndarray

    and that inheriting subclass implements:
        freq
    """

    # _infer_matches -> which infer_dtype strings are close enough to our own
    _infer_matches: tuple[str, ...]
    _is_recognized_dtype: Callable[[DtypeObj], bool]
    _recognized_scalars: tuple[type, ...]
    _ndarray: np.ndarray
    freq: BaseOffset | None

    @cache_readonly
    def _can_hold_na(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: cached property; always true because these arrays have a
        # missing-value marker.
        return True

    def __init__(
        self, data, dtype: Dtype | None = None, freq=None, copy: bool = False
    ) -> None:
        # [FUNCTION DOC]
        # Abstract constructor. Parameters data, dtype, freq, copy are accepted
        # for signature documentation only. Always raises the abstract-method
        # error; subclasses define the real constructors.
        raise AbstractMethodError(self)

    @property
    def _scalar_type(self) -> type[DatetimeLikeScalar]:
        # [FUNCTION DOC]
        # HELPER: abstract property naming the scalar class boxed values use.
        # Always raises the abstract-method error in this base class.
        """
        The scalar associated with this datelike

        * PeriodArray : Period
        * DatetimeArray : Timestamp
        * TimedeltaArray : Timedelta
        """
        raise AbstractMethodError(self)

    def _scalar_from_string(self, value: str) -> DTScalarOrNaT:
        # [FUNCTION DOC]
        # HELPER: abstract hook to parse one string into this array's scalar type
        # (or the missing marker). Parameter value is the string. Always raises
        # the abstract-method error here.
        """
        Construct a scalar type from a string.

        Parameters
        ----------
        value : str

        Returns
        -------
        Period, Timestamp, or Timedelta, or NaT
            Whatever the type of ``self._scalar_type`` is.

        Notes
        -----
        This should call ``self._check_compatible_with`` before
        unboxing the result.
        """
        raise AbstractMethodError(self)

    def _unbox_scalar(
        self, value: DTScalarOrNaT
    ) -> np.int64 | np.datetime64 | np.timedelta64:
        # [FUNCTION DOC]
        # HELPER: abstract hook converting a boxed scalar (or the missing
        # marker) to the raw stored value. Parameter value is the scalar. Always
        # raises the abstract-method error here.
        """
        Unbox the integer value of a scalar `value`.

        Parameters
        ----------
        value : Period, Timestamp, Timedelta, or NaT
            Depending on subclass.

        Returns
        -------
        int

        Examples
        --------
        >>> arr = pd.array(np.array(["1970-01-01"], "datetime64[ns]"))
        >>> arr._unbox_scalar(arr[0])
        np.datetime64('1970-01-01T00:00:00.000000000')
        """
        raise AbstractMethodError(self)

    def _check_compatible_with(self, other: DTScalarOrNaT) -> None:
        # [FUNCTION DOC]
        # HELPER: abstract hook verifying that a scalar can be combined with this
        # array (timezone match, frequency match, or no check, depending on the
        # subclass). Parameter other is the scalar. Always raises the
        # abstract-method error here.
        """
        Verify that `self` and `other` are compatible.

        * DatetimeArray verifies that the timezones (if any) match
        * PeriodArray verifies that the freq matches
        * Timedelta has no verification

        In each case, NaT is considered compatible.

        Parameters
        ----------
        other

        Raises
        ------
        Exception
        """
        raise AbstractMethodError(self)

    # ------------------------------------------------------------------

    def _box_func(self, x):
        # [FUNCTION DOC]
        # HELPER: abstract hook converting one raw stored value to a boxed
        # scalar. Parameter x is the raw value. Always raises the abstract-method
        # error here.
        """
        box function to get object from internal representation
        """
        raise AbstractMethodError(self)

    def _box_values(self, values) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: boxes a whole array of raw values. Parameter values is an
        # array of raw stored values. Applies the boxing function element-wise
        # and returns an object array (no type conversion of the result).
        """
        apply box func to passed values
        """
        return lib.map_infer(values, self._box_func, convert=False)

    def __iter__(self) -> Iterator:
        # [FUNCTION DOC]
        # Iterates lazily over boxed scalars. For multi-dimensional arrays yields
        # sub-arrays by position along the first axis; for one-dimensional
        # arrays yields each raw integer value boxed through the boxing function.
        # Returns a generator expression.
        if self.ndim > 1:
            return (self[n] for n in range(len(self)))
        else:
            return (self._box_func(v) for v in self.asi8)

    @property
    def asi8(self) -> npt.NDArray[np.int64]:
        # [FUNCTION DOC]
        # Property returning a 64-bit integer view of the stored values (shares
        # memory, never cached to avoid a memory leak). Missing entries are the
        # minimum 64-bit integer.
        """
        Integer representation of the values.

        Returns
        -------
        ndarray
            An ndarray with int64 dtype.
        """
        # do not cache or you'll create a memory leak
        return self._ndarray.view("i8")

    # ----------------------------------------------------------------
    # Rendering Methods

    def _format_native_types(
        self, *, na_rep: str | float = "NaT", date_format=None
    ) -> npt.NDArray[np.object_]:
        # [FUNCTION DOC]
        # HELPER: abstract hook that renders values as strings for conversion to
        # text dtypes. Parameters (keyword only): na_rep is the text for missing
        # values, default the not-a-time text; date_format is an optional
        # format pattern. Always raises the abstract-method error here.
        """
        Helper method for astype when converting to strings.

        Returns
        -------
        ndarray[str]
        """
        raise AbstractMethodError(self)

    def _formatter(self, boxed: bool = False) -> Callable[[object], str]:
        # [FUNCTION DOC]
        # HELPER: formatter used when printing the array's elements. Parameter
        # boxed is ignored. Returns a callable that wraps any value's text in
        # single quotes.
        # TODO: Remove Datetime & DatetimeTZ formatters.
        return "'{}'".format

    # ----------------------------------------------------------------
    # Array-Like / EA-Interface Methods

    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        # [FUNCTION DOC]
        # Numpy array-conversion protocol (time delta and datetime arrays; the
        # period array overrides it). Parameters: dtype is the requested numpy
        # dtype; copy is the numpy copy flag (true forces a copy, false forbids
        # one, none allows either). For an object dtype request, builds an object
        # array of boxed scalars and raises a value error if copy is false
        # (a copy is unavoidable). If copy is true, returns a fresh array cast to
        # dtype. Otherwise returns the backing array itself, as a read-only view
        # when this array is read-only. Note the requested dtype is ignored
        # (apart from the object case and forced copies) when copy is not true.
        # used for Timedelta/DatetimeArray, overwritten by PeriodArray
        if is_object_dtype(dtype):
            if copy is False:
                raise ValueError(
                    "Unable to avoid copy while creating an array as requested."
                )
            return np.array(list(self), dtype=object)

        if copy is True:
            return np.array(self._ndarray, dtype=dtype)

        result = self._ndarray
        if self._readonly:
            result = result.view()
            result.flags.writeable = False
        return result

    @overload
    def __getitem__(self, key: ScalarIndexer) -> DTScalarOrNaT:
        # [FUNCTION DOC]
        # Typing overload: a scalar position returns one boxed scalar. See the
        # implementation below.
        ...

    @overload
    def __getitem__(
        self,
        key: SequenceIndexer | PositionalIndexerTuple,
    ) -> Self:
        # [FUNCTION DOC]
        # Typing overload: a sequence, slice or tuple position returns a new
        # array. See the implementation below.
        ...

    def __getitem__(self, key: PositionalIndexer2D) -> Self | DTScalarOrNaT:
        # [FUNCTION DOC]
        # Positional lookup. Parameter key is a scalar integer, slice,
        # sequence, boolean mask or (for two-dimensional data) tuple. Defers to
        # the numpy-backed parent; scalar results come back boxed. Array
        # results get their frequency attribute assigned from the frequency-
        # lookup helper (so slices can keep or scale the frequency).
        """
        This getitem defers to the underlying array, which by-definition can
        only handle list-likes, slices, and integer scalars
        """
        # Use cast as we know we will get back a DatetimeLikeArray or DTScalar,
        # but skip evaluating the Union at runtime for performance
        # (see https://github.com/pandas-dev/pandas/pull/44624)
        result = cast(Union[Self, DTScalarOrNaT], super().__getitem__(key))
        if lib.is_scalar(result):
            return result
        else:
            # At this point we know the result is an array.
            result = cast(Self, result)
        # error: Incompatible types in assignment (expression has type
        # "BaseOffset | None", variable has type "None")
        result._freq = self._get_getitem_freq(key)  # type: ignore[assignment]
        return result

    def _get_getitem_freq(self, key) -> BaseOffset | None:
        # [FUNCTION DOC]
        # HELPER: decides the frequency of the result of an indexing operation.
        # Parameter key is the indexer used. Period arrays always keep their
        # frequency. For other arrays: multi-dimensional input yields none;
        # otherwise the key is normalized as an array indexer; a slice keeps the
        # frequency (multiplied by the slice step when a step is given and a
        # frequency exists); an ellipsis keeps it; a boolean mask that is
        # equivalent to a slice is re-evaluated as that slice; anything else
        # yields none. Returns an offset or none.
        """
        Find the `freq` attribute to assign to the result of a __getitem__ lookup.
        """
        is_period = isinstance(self.dtype, PeriodDtype)
        if is_period:
            freq = self.freq
        elif self.ndim != 1:
            freq = None
        else:
            key = check_array_indexer(self, key)  # maybe ndarray[bool] -> slice
            freq = None
            if isinstance(key, slice):
                if self.freq is not None and key.step is not None:
                    freq = key.step * self.freq
                else:
                    freq = self.freq
            elif key is Ellipsis:
                # GH#21282 indexing with Ellipsis is similar to a full slice,
                #  should preserve `freq` attribute
                freq = self.freq
            elif com.is_bool_indexer(key):
                new_key = lib.maybe_booleans_to_slice(key.view(np.uint8))
                if isinstance(new_key, slice):
                    return self._get_getitem_freq(new_key)
        return freq

    # error: Argument 1 of "__setitem__" is incompatible with supertype
    # "ExtensionArray"; supertype defines the argument type as "Union[int,
    # ndarray]"
    def __setitem__(
        self,
        key: int | Sequence[int] | Sequence[bool] | slice,
        value: NaTType | Any | Sequence[Any],
    ) -> None:
        # [FUNCTION DOC]
        # In-place assignment. Parameters: key is an integer, sequence of
        # integers or booleans, or slice; value is a scalar, the missing marker,
        # or a sequence of them. Checks that lengths line up (raising on
        # mismatch) and calls the parent assignment first, so invalid values
        # raise even when the assignment would be a no-op (for example an empty
        # selection). If something was actually assigned, the frequency
        # invalidation hook runs. Mutates self; returns nothing.
        # I'm fudging the types a bit here. "Any" above really depends
        # on type(self). For PeriodArray, it's Period (or stuff coercible
        # to a period in from_sequence). For DatetimeArray, it's Timestamp...
        # I don't know if mypy can do that, possibly with Generics.
        # https://mypy.readthedocs.io/en/latest/generics.html

        no_op = check_setitem_lengths(key, value, self)

        # Calling super() before the no_op short-circuit means that we raise
        #  on invalid 'value' even if this is a no-op, e.g. wrong-dtype empty array.
        super().__setitem__(key, value)

        if no_op:
            return

        self._maybe_clear_freq()

    def _maybe_clear_freq(self) -> None:
        # [FUNCTION DOC]
        # HELPER: hook called after in-place mutation. Does nothing in this base
        # class; datetime and time delta arrays override it to drop their
        # frequency, which may no longer hold.
        # inplace operations like __setitem__ may invalidate the freq of
        # DatetimeArray and TimedeltaArray
        pass

    def astype(self, dtype, copy: bool = True):
        # [FUNCTION DOC]
        # Generic conversion for datetime-like arrays (subclasses handle
        # same-kind cases first). Parameters: dtype is anything resolvable to a
        # dtype; copy (default true) matters for integer and extension targets.
        # Object target: datetimes become an object array of timestamps built
        # quickly from the integers with timezone; time deltas become an object
        # array of time delta scalars; other kinds are boxed element-wise with
        # the shape preserved. String target: with an extension string dtype,
        # formats values using that dtype's missing value and builds its array
        # type; with a plain numpy string dtype returns the formatted object
        # array. Other extension targets defer to the parent. Integer targets
        # must be exactly 64-bit signed (otherwise a type error telling the
        # caller to go through int64) and return the integer view, copied when
        # copy is true. Raises a type error for datetime/time delta targets that
        # differ from the current dtype and for any float target. Everything
        # else is converted by numpy from the array representation.
        # Some notes on cases we don't have to handle here in the base class:
        #   1. PeriodArray.astype handles period -> period
        #   2. DatetimeArray.astype handles conversion between tz.
        #   3. DatetimeArray.astype handles datetime -> period
        dtype = pandas_dtype(dtype)

        if dtype == object:
            if self.dtype.kind == "M":
                self = cast("DatetimeArray", self)
                # *much* faster than self._box_values
                #  for e.g. test_get_loc_tuple_monotonic_above_size_cutoff
                i8data = self.asi8
                converted = ints_to_pydatetime(
                    i8data,
                    tz=self.tz,
                    box="timestamp",
                    reso=self._creso,
                )
                return converted

            elif self.dtype.kind == "m":
                return ints_to_pytimedelta(self._ndarray, box=True)

            return self._box_values(self.asi8.ravel()).reshape(self.shape)

        elif is_string_dtype(dtype):
            if isinstance(dtype, ExtensionDtype):
                arr_object = self._format_native_types(na_rep=dtype.na_value)  # type: ignore[arg-type]
                cls = dtype.construct_array_type()
                return cls._from_sequence(arr_object, dtype=dtype, copy=False)
            else:
                return self._format_native_types()

        elif isinstance(dtype, ExtensionDtype):
            return super().astype(dtype, copy=copy)
        elif dtype.kind in "iu":
            # we deliberately ignore int32 vs. int64 here.
            # See https://github.com/pandas-dev/pandas/issues/24381 for more.
            values = self.asi8
            if dtype != np.int64:
                raise TypeError(
                    f"Converting from {self.dtype} to {dtype} is not supported. "
                    "Do obj.astype('int64').astype(dtype) instead"
                )

            if copy:
                values = values.copy()
            return values
        elif (dtype.kind in "mM" and self.dtype != dtype) or dtype.kind == "f":
            # disallow conversion between datetime/timedelta,
            # and conversions for any datetimelike to float
            msg = f"Cannot cast {type(self).__name__} to dtype {dtype}"
            raise TypeError(msg)
        else:
            return np.asarray(self, dtype=dtype)

    @overload  # type: ignore[override]
    def view(self) -> Self:
        # [FUNCTION DOC]
        # Typing overload: no argument returns the same array type. See the
        # implementation below.
        ...

    @overload
    def view(self, dtype: Literal["M8[ns]"]) -> DatetimeArray:
        # [FUNCTION DOC]
        # Typing overload: viewing as nanosecond datetimes returns a datetime
        # array. See the implementation below.
        ...

    @overload
    def view(self, dtype: Literal["m8[ns]"]) -> TimedeltaArray:
        # [FUNCTION DOC]
        # Typing overload: viewing as nanosecond time deltas returns a time
        # delta array. See the implementation below.
        ...

    @overload
    def view(self, dtype: Dtype | None = ...) -> ArrayLike:
        # [FUNCTION DOC]
        # Typing overload: any other dtype returns a generic array. See the
        # implementation below.
        ...

    def view(self, dtype: Dtype | None = None) -> ArrayLike:
        # [FUNCTION DOC]
        # Reinterprets the same memory as another dtype. Parameter dtype is the
        # target dtype, default none (same dtype). Simply delegates to the
        # numpy-backed parent; this definition exists only so the typing
        # overloads above have an implementation. Returns an array sharing
        # memory.
        # we need to explicitly call super() method as long as the `@overload`s
        #  are present in this file.
        return super().view(dtype)

    def _putmask(self, mask: npt.NDArray[np.bool_], value) -> None:
        # [FUNCTION DOC]
        # HELPER: in-place masked assignment. Parameters: mask is a boolean array
        # of the same length; value is the replacement (scalar or array-like).
        # Delegates to the parent implementation, then unconditionally clears
        # the stored frequency (even if the mask selects nothing). Mutates self.
        super()._putmask(mask, value)
        self._freq = None  # GH#24555

    # ------------------------------------------------------------------
    # Validation Methods
    # TODO: try to de-duplicate these, ensure identical behavior

    def _validate_comparison_value(self, other):
        # [FUNCTION DOC]
        # HELPER: prepares the right-hand side of a comparison. Parameter other
        # is a string, scalar, list-like or anything else. Strings are parsed
        # into this array's scalar (failure becomes an invalid-comparison
        # error, and string parsing is exempt from timezone-awareness checks).
        # Recognized scalars and the missing marker are converted to the scalar
        # type and checked for compatibility (a type error becomes an
        # invalid-comparison error). Non-list-likes raise invalid-comparison;
        # list-likes of the wrong length raise a value error saying lengths
        # must match; otherwise the list-like is validated allowing object
        # dtype, and compatibility is checked, where a type error is tolerated
        # only for object-dtype input (compared element-wise later) and
        # otherwise becomes invalid-comparison. Returns the converted operand.
        if isinstance(other, str):
            try:
                # GH#18435 strings get a pass from tzawareness compat
                other = self._scalar_from_string(other)
            except (ValueError, IncompatibleFrequency) as err:
                # failed to parse as Timestamp/Timedelta/Period
                raise InvalidComparison(other) from err

        if isinstance(other, self._recognized_scalars) or other is NaT:
            other = self._scalar_type(other)
            try:
                self._check_compatible_with(other)
            except TypeError as err:
                # e.g. tzawareness mismatch
                raise InvalidComparison(other) from err

        elif not is_list_like(other):
            raise InvalidComparison(other)

        elif len(other) != len(self):
            raise ValueError("Lengths must match")

        else:
            try:
                other = self._validate_listlike(other, allow_object=True)
                self._check_compatible_with(other)
            except TypeError as err:
                if is_object_dtype(getattr(other, "dtype", None)):
                    # We will have to operate element-wise
                    pass
                else:
                    raise InvalidComparison(other) from err

        return other

    def _validate_scalar(
        self,
        value,
        *,
        allow_listlike: bool = False,
        unbox: bool = True,
    ):
        # [FUNCTION DOC]
        # HELPER: validates and converts one scalar. Parameters: value is the
        # candidate; allow_listlike (keyword, default false) only changes the
        # wording of the error message; unbox (keyword, default true) controls
        # whether the scalar is converted to the raw stored value. Accepts
        # instances of the scalar type, strings (parsed; parse failure becomes a
        # type error), values that count as a valid missing marker for the dtype
        # (converted to the missing marker), recognized scalar types (converted).
        # Any other missing-like value, or anything unrecognized, raises a type
        # error with a descriptive message. When unbox is false the boxed scalar
        # is returned without the unboxing/compatibility check. Returns the raw
        # value or the boxed scalar.
        """
        Validate that the input value can be cast to our scalar_type.

        Parameters
        ----------
        value : object
        allow_listlike: bool, default False
            When raising an exception, whether the message should say
            listlike inputs are allowed.
        unbox : bool, default True
            Whether to unbox the result before returning.  Note: unbox=False
            skips the setitem compatibility check.

        Returns
        -------
        self._scalar_type or NaT
        """
        if isinstance(value, self._scalar_type):
            pass

        elif isinstance(value, str):
            # NB: Careful about tzawareness
            try:
                value = self._scalar_from_string(value)
            except ValueError as err:
                msg = self._validation_error_message(value, allow_listlike)
                raise TypeError(msg) from err

        elif is_valid_na_for_dtype(value, self.dtype):
            # GH#18295
            value = NaT

        elif isna(value):
            # if we are dt64tz and value is dt64("NaT"), dont cast to NaT,
            #  or else we'll fail to raise in _unbox_scalar
            msg = self._validation_error_message(value, allow_listlike)
            raise TypeError(msg)

        elif isinstance(value, self._recognized_scalars):
            # error: Argument 1 to "Timestamp" has incompatible type "object"; expected
            # "integer[Any] | float | str | date | datetime | datetime64"
            value = self._scalar_type(value)  # type: ignore[arg-type]

        else:
            msg = self._validation_error_message(value, allow_listlike)
            raise TypeError(msg)

        if not unbox:
            # NB: In general NDArrayBackedExtensionArray will unbox here;
            #  this option exists to prevent a performance hit in
            #  TimedeltaIndex.get_loc
            return value
        return self._unbox_scalar(value)

    def _validation_error_message(self, value, allow_listlike: bool = False) -> str:
        # [FUNCTION DOC]
        # HELPER: builds the text for validation failures. Parameters: value is
        # the offending input; allow_listlike (default false) selects wording
        # that also permits arrays. Describes the input as "<dtype> array" for
        # array-like values with a dimension, otherwise by its type name, and
        # states which scalar type or missing marker was expected. Returns the
        # message string; does not raise.
        """
        Construct an exception message on validation error.

        Some methods allow only scalar inputs, while others allow either scalar
        or listlike.

        Parameters
        ----------
        allow_listlike: bool, default False

        Returns
        -------
        str
        """
        if hasattr(value, "dtype") and getattr(value, "ndim", 0) > 0:
            msg_got = f"{value.dtype} array"
        else:
            msg_got = f"'{type(value).__name__}'"
        if allow_listlike:
            msg = (
                f"value should be a '{self._scalar_type.__name__}', 'NaT', "
                f"or array of those. Got {msg_got} instead."
            )
        else:
            msg = (
                f"value should be a '{self._scalar_type.__name__}' or 'NaT'. "
                f"Got {msg_got} instead."
            )
        return msg

    def _validate_listlike(self, value, allow_object: bool = False):
        # [FUNCTION DOC]
        # HELPER: coerces a list-like into an array of this class's type.
        # Parameters: value is any list-like; allow_object (default false) lets
        # object-dtype results through unconverted. Steps: an instance of the
        # same class is returned (cast to this unit for datetime/time delta kinds
        # unless allow_object, without rounding); an empty list becomes an empty
        # array of this dtype; object arrays whose inferred type is compatible
        # are parsed (failures return the input when allow_object, otherwise
        # raise a type error); lists become object arrays; object arrays are
        # converted by the generic object converter with all-missing results
        # given this dtype; then general array construction runs. An all-string
        # result is parsed to this dtype (errors ignored). A categorical whose
        # categories have this dtype is replaced by its values. Raises a type
        # error when the resulting dtype is not recognized (unless object is
        # allowed). Finally non-allow_object datetime/time delta values are cast
        # to this unit without rounding. Returns the converted array.
        if isinstance(value, type(self)):
            if self.dtype.kind in "mM" and not allow_object and self.unit != value.unit:  # type: ignore[attr-defined]
                # error: "DatetimeLikeArrayMixin" has no attribute "as_unit"
                value = value.as_unit(self.unit, round_ok=False)  # type: ignore[attr-defined]
            return value

        if isinstance(value, list) and len(value) == 0:
            # We treat empty list as our own dtype.
            return type(self)._from_sequence([], dtype=self.dtype)

        if hasattr(value, "dtype") and value.dtype == object:
            # `array` below won't do inference if value is an Index or Series.
            #  so do so here.  in the Index case, inferred_type may be cached.
            if lib.infer_dtype(value) in self._infer_matches:
                try:
                    value = type(self)._from_sequence(value)
                except (ValueError, TypeError) as err:
                    if allow_object:
                        return value
                    msg = self._validation_error_message(value, True)
                    raise TypeError(msg) from err

        if isinstance(value, list):
            value = construct_1d_object_array_from_listlike(value)
        if isinstance(value, np.ndarray) and value.dtype == object:
            # We need to call maybe_convert_objects here instead of in pd_array
            #  so we can specify dtype_if_all_nat.
            value = lib.maybe_convert_objects(
                value, convert_non_numeric=True, dtype_if_all_nat=self.dtype
            )
        # Do type inference if necessary up front (after unpacking
        # NumpyExtensionArray)
        # e.g. we passed PeriodIndex.values and got an ndarray of Periods
        value = extract_array(value, extract_numpy=True)
        value = pd_array(value)
        value = extract_array(value, extract_numpy=True)

        if is_all_strings(value):
            # We got a StringArray
            try:
                # TODO: Could use from_sequence_of_strings if implemented
                # Note: passing dtype is necessary for PeriodArray tests
                value = type(self)._from_sequence(value, dtype=self.dtype)
            except ValueError:
                pass

        if isinstance(value.dtype, CategoricalDtype):
            # e.g. we have a Categorical holding self.dtype
            if value.categories.dtype == self.dtype:
                # TODO: do we need equal dtype or just comparable?
                value = value._internal_get_values()
                value = extract_array(value, extract_numpy=True)

        if allow_object and is_object_dtype(value.dtype):
            pass

        elif not type(self)._is_recognized_dtype(value.dtype):
            msg = self._validation_error_message(value, True)
            raise TypeError(msg)

        if self.dtype.kind in "mM" and not allow_object:
            # error: "DatetimeLikeArrayMixin" has no attribute "as_unit"
            value = value.as_unit(self.unit, round_ok=False)  # type: ignore[attr-defined]
        return value

    def _validate_setitem_value(self, value):
        # [FUNCTION DOC]
        # HELPER: validates a value being assigned. Parameter value is a scalar
        # or list-like. Scalars go through the scalar validator (returning the
        # raw stored value, with error wording that mentions array inputs);
        # list-likes are validated into an array of this class and then
        # unboxed to the backing numpy array. Raises type errors from the
        # validators.
        if is_list_like(value):
            value = self._validate_listlike(value)
        else:
            return self._validate_scalar(value, allow_listlike=True)

        return self._unbox(value)

    @final
    def _unbox(self, other) -> np.int64 | np.datetime64 | np.timedelta64 | np.ndarray:
        # [FUNCTION DOC]
        # HELPER (final, not overridable): unboxes an operand. Parameter other is
        # either a scalar or an instance of this array type. A scalar is
        # converted with the scalar unboxing hook; an array is checked for
        # compatibility (may raise) and its backing numpy array is returned.
        """
        Unbox either a scalar with _unbox_scalar or an instance of our own type.
        """
        if lib.is_scalar(other):
            other = self._unbox_scalar(other)
        else:
            # same type as self
            self._check_compatible_with(other)
            other = other._ndarray
        return other

    # ------------------------------------------------------------------
    # Additional array methods
    #  These are not part of the EA API, but we implement them because
    #  pandas assumes they're there.

    @ravel_compat
    def map(self, mapper, na_action: Literal["ignore"] | None = None):
        # [FUNCTION DOC]
        # Applies a mapping to every element (decorated so multi-dimensional
        # arrays are flattened and restored). Parameters: mapper is a function,
        # dict or series; na_action is none or "ignore" (skip missing values).
        # The result of the generic array mapper is wrapped in an index so types
        # are inferred. Returns a numpy array of tuples-as-objects if the index
        # is a multi-level index, otherwise the index's underlying array (which
        # may be a datetime-like or other array).
        from pandas import Index

        result = map_array(self, mapper, na_action=na_action)
        result = Index(result)

        if isinstance(result, ABCMultiIndex):
            return result.to_numpy()
        else:
            return result.array

    def isin(self, values: ArrayLike) -> npt.NDArray[np.bool_]:
        # [FUNCTION DOC]
        # Membership test. Parameter values is an array or extension array.
        # Returns a boolean array shaped like self. Numeric kinds (float,
        # integer, unsigned, complex) never match, giving all false. Other
        # inputs are wrapped as datetime-like if possible; non-matching types
        # give all false except object arrays, which are first converted to a
        # datetime-like array if possible (then retried) or else compared
        # element-wise as objects. For same-type values, the unit is aligned for
        # datetime/time delta; incompatible timezones/frequencies give all
        # false; otherwise the integer representations are compared.
        """
        Compute boolean array of whether each value is found in the
        passed set of values.

        Parameters
        ----------
        values : np.ndarray or ExtensionArray

        Returns
        -------
        ndarray[bool]
        """
        if values.dtype.kind in "fiuc":
            # TODO: de-duplicate with equals, validate_comparison_value
            return np.zeros(self.shape, dtype=bool)

        values = ensure_wrapped_if_datetimelike(values)

        if not isinstance(values, type(self)):
            if values.dtype == object:
                values = lib.maybe_convert_objects(
                    values,  # type: ignore[arg-type]
                    convert_non_numeric=True,
                    dtype_if_all_nat=self.dtype,
                )
                if values.dtype != object:
                    return self.isin(values)
                else:
                    # TODO: Deprecate this case
                    # https://github.com/pandas-dev/pandas/pull/58645/files#r1604055791
                    return isin(self.astype(object), values)
            return np.zeros(self.shape, dtype=bool)

        if self.dtype.kind in "mM":
            self = cast("DatetimeArray | TimedeltaArray", self)
            # error: "DatetimeLikeArrayMixin" has no attribute "as_unit"
            values = values.as_unit(self.unit)  # type: ignore[attr-defined]

        try:
            # error: Argument 1 to "_check_compatible_with" of "DatetimeLikeArrayMixin"
            # has incompatible type "ExtensionArray | ndarray[Any, Any]"; expected
            # "Period | Timestamp | Timedelta | NaTType"
            self._check_compatible_with(values)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            # Includes tzawareness mismatch and IncompatibleFrequencyError
            return np.zeros(self.shape, dtype=bool)

        # error: Item "ExtensionArray" of "ExtensionArray | ndarray[Any, Any]"
        # has no attribute "asi8"
        return isin(self.asi8, values.asi8)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Null Handling

    def isna(self) -> npt.NDArray[np.bool_]:
        # [FUNCTION DOC]
        # Returns the boolean missing-value mask (true where missing), the same
        # object as the internal missing-mask property.
        return self._isnan

    @property  # NB: override with cache_readonly in immutable subclasses
    def _isnan(self) -> npt.NDArray[np.bool_]:
        # [FUNCTION DOC]
        # HELPER: property computing the missing mask by comparing the integer
        # view to the missing sentinel (the minimum 64-bit integer). Returns a
        # new boolean array each access unless overridden with a cache.
        """
        return if each value is nan
        """
        return self.asi8 == iNaT

    @property  # NB: override with cache_readonly in immutable subclasses
    def _hasna(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: property that is true if any element is missing; lets callers
        # skip masking work when there are none.
        """
        return if I have any nans; enables various perf speedups
        """
        return bool(self._isnan.any())

    def _maybe_mask_results(
        self, result: np.ndarray, fill_value=iNaT, convert=None
    ) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: overwrites entries of a computed result at positions where this
        # array is missing. Parameters: result is a numpy array of the same
        # length; fill_value is the value written, default the integer missing
        # sentinel (none means not-a-number); convert is an optional dtype the
        # result is cast to before masking (only when there is any missing
        # value). If nothing is missing the result is returned untouched
        # (note: then convert is not applied either). Otherwise the (possibly
        # cast) result is modified in place and returned.
        # KNOWN ISSUE: the docstring's summary lines about masking appear after
        # the Returns section and the described return type is loose; behavior
        # differs from the "convert if not None" description when there are no
        # missing values, because conversion is skipped.
        """
        Parameters
        ----------
        result : np.ndarray
        fill_value : object, default iNaT
        convert : str, dtype or None

        Returns
        -------
        result : ndarray with values replace by the fill_value

        mask the result if needed, convert to the provided dtype if its not
        None

        This is an internal routine.
        """
        if self._hasna:
            if convert:
                result = result.astype(convert)
            if fill_value is None:
                fill_value = np.nan
            np.putmask(result, self._isnan, fill_value)
        return result

    # ------------------------------------------------------------------
    # Frequency Properties/Methods

    @property
    def freqstr(self) -> str | None:
        # [FUNCTION DOC]
        # Property returning the frequency's string alias, or none when no
        # frequency is set.
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
        if self.freq is None:
            return None
        return self.freq.freqstr

    @property  # NB: override with cache_readonly in immutable subclasses
    def inferred_freq(self) -> str | None:
        # [FUNCTION DOC]
        # Property that tries to infer a frequency from the values. Returns none
        # for non one-dimensional data or when the inference helper raises a
        # value error (too few points, irregular spacing); otherwise returns the
        # frequency alias string. Immutable subclasses cache the result.
        """
        Tries to return a string representing a frequency generated by infer_freq.

        Returns None if it can't autodetect the frequency.

        See Also
        --------
        DatetimeIndex.freqstr : Return the frequency object as a string if it's set,
            otherwise None.

        Examples
        --------
        For DatetimeIndex:

        >>> idx = pd.DatetimeIndex(["2018-01-01", "2018-01-03", "2018-01-05"])
        >>> idx.inferred_freq
        '2D'

        For TimedeltaIndex:

        >>> tdelta_idx = pd.to_timedelta(["0 days", "10 days", "20 days"])
        >>> tdelta_idx
        TimedeltaIndex(['0 days', '10 days', '20 days'],
                       dtype='timedelta64[us]', freq=None)
        >>> tdelta_idx.inferred_freq
        '10D'
        """
        if self.ndim != 1:
            return None
        try:
            return frequencies.infer_freq(self)
        except ValueError:
            return None

    @property  # NB: override with cache_readonly in immutable subclasses
    def _resolution_obj(self) -> Resolution | None:
        # [FUNCTION DOC]
        # HELPER: property deriving the resolution enum from the frequency
        # string. Returns none when there is no frequency or the frequency alias
        # is unknown (lookup failure is swallowed); otherwise the resolution
        # object. Subclasses may override with a value-based computation.
        freqstr = self.freqstr
        if freqstr is None:
            return None
        try:
            return Resolution.get_reso_from_freqstr(freqstr)
        except KeyError:
            return None

    @property  # NB: override with cache_readonly in immutable subclasses
    def resolution(self) -> str:
        # [FUNCTION DOC]
        # Property returning the name of the finest resolution needed to
        # represent the values (as a lowercase word such as day or minute),
        # taken from the resolution object's name attribute.
        # KNOWN ISSUE: the docstring lists only down to microsecond, but
        # nanosecond is also possible; and when the resolution object is none
        # (for example no frequency in the base implementation) accessing the
        # name attribute fails with an attribute error (suppressed for the type
        # checker only).
        """
        Returns day, hour, minute, second, millisecond or microsecond
        """
        # error: Item "None" of "Optional[Any]" has no attribute "attrname"
        return self._resolution_obj.attrname  # type: ignore[union-attr]

    # monotonicity/uniqueness properties are called via frequencies.infer_freq,
    #  see GH#23789

    @property
    def _is_monotonic_increasing(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: property; true when the values are non-decreasing, using the
        # compiled monotonicity check on the integer view in time-like mode
        # (missing values handled by that routine). Used by frequency inference.
        return algos.is_monotonic(self.asi8, timelike=True)[0]

    @property
    def _is_monotonic_decreasing(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: property; true when the values are non-increasing, using the
        # compiled monotonicity check on the integer view in time-like mode.
        return algos.is_monotonic(self.asi8, timelike=True)[1]

    @property
    def _is_unique(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: property; true when no value repeats. Flattens the integer view
        # in memory order, counts distinct entries and compares the count to the
        # array size (all missing values count as one value).
        return len(unique1d(self.asi8.ravel("K"))) == self.size

    # ------------------------------------------------------------------
    # Arithmetic Methods

    def _cmp_method(self, other, op):
        # [FUNCTION DOC]
        # HELPER: implements all six comparison operators for the operator mixin.
        # Parameters: other is the right-hand operand; op is the binary
        # comparison function. Same-shaped multi-dimensional operands are
        # flattened, compared, and reshaped. The operand is validated; an
        # invalid comparison yields the "not implemented" signal for arrow-
        # backed dtypes, else the generic invalid-comparison result (false for
        # equality, true for inequality, or a type error for ordering).
        # Object-dtype operands are compared element-wise via the object
        # comparison helper. Comparing with the missing marker gives all true for
        # not-equal and all false otherwise. Differing resolutions (non-period):
        # a scalar is cast losslessly to this unit if possible, else both sides
        # are compared with the mismatched-resolution routine; arrays of
        # different resolution use that routine directly. Same resolution: the
        # integer views are compared (faster), and positions where either side
        # is missing are overwritten with true only for not-equal. Returns a
        # boolean array shaped like self.
        if self.ndim > 1 and getattr(other, "shape", None) == self.shape:
            # TODO: handle 2D-like listlikes
            return op(self.ravel(), other.ravel()).reshape(self.shape)

        try:
            other = self._validate_comparison_value(other)
        except InvalidComparison:
            if hasattr(other, "dtype") and isinstance(other.dtype, ArrowDtype):
                return NotImplemented
            return invalid_comparison(self, other, op)

        dtype = getattr(other, "dtype", None)
        if is_object_dtype(dtype):
            # We have to use comp_method_OBJECT_ARRAY instead of numpy
            #  comparison otherwise it would raise when comparing to None
            result = ops.comp_method_OBJECT_ARRAY(
                op, np.asarray(self.astype(object)), other
            )
            return result
        if other is NaT:
            if op is operator.ne:
                result = np.ones(self.shape, dtype=bool)
            else:
                result = np.zeros(self.shape, dtype=bool)
            return result

        if not isinstance(self.dtype, PeriodDtype):
            self = cast(TimelikeOps, self)
            if self._creso != other._creso:
                if not isinstance(other, type(self)):
                    # i.e. Timedelta/Timestamp, cast to ndarray and let
                    #  compare_mismatched_resolutions handle broadcasting
                    try:
                        # GH#52080 see if we can losslessly cast to shared unit
                        other = other.as_unit(self.unit, round_ok=False)
                    except ValueError:
                        other_arr = np.array(other.asm8)
                        return compare_mismatched_resolutions(
                            self._ndarray, other_arr, op
                        )
                else:
                    other_arr = other._ndarray
                    return compare_mismatched_resolutions(self._ndarray, other_arr, op)

        other_vals = self._unbox(other)
        # GH#37462 comparison on i8 values is almost 2x faster than M8/m8
        result = op(self._ndarray.view("i8"), other_vals.view("i8"))

        o_mask = isna(other)
        mask = self._isnan | o_mask
        if mask.any():
            nat_result = op is operator.ne
            np.putmask(result, mask, nat_result)

        return result

    # pow is invalid for all three subclasses; TimedeltaArray will override
    #  the multiplication and division ops
    __pow__ = _make_unpacked_invalid_op("__pow__")
    __rpow__ = _make_unpacked_invalid_op("__rpow__")
    __mul__ = _make_unpacked_invalid_op("__mul__")
    __rmul__ = _make_unpacked_invalid_op("__rmul__")
    __truediv__ = _make_unpacked_invalid_op("__truediv__")
    __rtruediv__ = _make_unpacked_invalid_op("__rtruediv__")
    __floordiv__ = _make_unpacked_invalid_op("__floordiv__")
    __rfloordiv__ = _make_unpacked_invalid_op("__rfloordiv__")
    __mod__ = _make_unpacked_invalid_op("__mod__")
    __rmod__ = _make_unpacked_invalid_op("__rmod__")
    __divmod__ = _make_unpacked_invalid_op("__divmod__")
    __rdivmod__ = _make_unpacked_invalid_op("__rdivmod__")

    @final
    def _get_i8_values_and_mask(
        self, other
    ) -> tuple[int | npt.NDArray[np.int64], None | npt.NDArray[np.bool_]]:
        # [FUNCTION DOC]
        # HELPER (final): extracts integer data for overflow-safe arithmetic.
        # Parameter other is a period, timestamp, time delta, or a datetime-like
        # array. Returns a pair: the integer value(s) (ordinal for a period,
        # stored integer for a timestamp/time delta, the integer view for
        # arrays) and a boolean missing mask (none for scalars, the array's mask
        # for arrays).
        """
        Get the int64 values and b_mask to pass to add_overflowsafe.
        """
        if isinstance(other, Period):
            i8values = other.ordinal
            mask = None
        elif isinstance(other, (Timestamp, Timedelta)):
            i8values = other._value
            mask = None
        else:
            # PeriodArray, DatetimeArray, TimedeltaArray
            mask = other._isnan
            i8values = other.asi8
        return i8values, mask

    @final
    def _get_arithmetic_result_freq(self, other) -> BaseOffset | None:
        # [FUNCTION DOC]
        # HELPER (final): decides whether the frequency survives an addition or
        # subtraction. Parameter other is the operand. Period arrays always keep
        # their frequency. Non-scalar operands drop it (none). For scalars it is
        # kept when: the frequency is a fixed-duration offset; the array is time
        # deltas and the operand is a time delta; the array is time deltas and
        # the operand is a timestamp that is naive or UTC; the array is
        # datetimes with a day frequency and the operand is a time delta or a
        # timestamp. Otherwise none.
        """
        Check if we can preserve self.freq in addition or subtraction.
        """
        # Adding or subtracting a Timedelta/Timestamp scalar is freq-preserving
        #  whenever self.freq is a Tick
        if isinstance(self.dtype, PeriodDtype):
            return self.freq
        elif not lib.is_scalar(other):
            return None
        elif isinstance(self.freq, Tick):
            # In these cases
            return self.freq
        elif self.dtype.kind == "m" and isinstance(other, Timedelta):
            return self.freq
        elif (
            self.dtype.kind == "m"
            and isinstance(other, Timestamp)
            and (other.tz is None or timezones.is_utc(other.tz))
        ):
            # e.g. test_td64arr_add_sub_datetimelike_scalar tdarr + timestamp
            #  gives a DatetimeArray. As long as the timestamp has no timezone
            #  or UTC, the result can retain a Day freq.
            return self.freq
        elif (
            lib.is_np_dtype(self.dtype, "M")
            and isinstance(self.freq, Day)
            and isinstance(other, Timedelta)
        ):
            # e.g. TestTimedelta64ArithmeticUnsorted::test_timedelta
            # Day is unambiguously 24h
            return self.freq
        elif (
            lib.is_np_dtype(self.dtype, "M")
            and isinstance(other, Timestamp)
            and isinstance(self.freq, Day)
        ):
            return self.freq

        return None

    @final
    def _add_datetimelike_scalar(self, other) -> DatetimeArray:
        # [FUNCTION DOC]
        # HELPER (final): time delta array plus a datetime scalar. Parameter
        # other is a datetime, timestamp or numpy datetime. Raises a type error
        # if self is not a time delta array. A numpy not-a-time is treated as a
        # datetime (not as a time delta) and yields an all-missing datetime
        # array at self's unit. Otherwise the scalar is converted to a
        # timestamp, units are matched, an overflow-safe integer addition is
        # done, and the result is a datetime array with the scalar's timezone
        # and a frequency decided by the arithmetic frequency helper.
        # KNOWN ISSUE: the code asserts the operand is not the missing marker
        # yet handles missing values just before; and it computes the
        # reinterpreted result values twice (redundant), and the mask returned
        # by the integer extraction helper is unused.
        if not lib.is_np_dtype(self.dtype, "m"):
            raise TypeError(
                f"cannot add {type(self).__name__} and {type(other).__name__}"
            )

        self = cast("TimedeltaArray", self)

        from pandas.core.arrays import DatetimeArray
        from pandas.core.arrays.datetimes import tz_to_dtype

        assert other is not NaT
        if isna(other):
            # i.e. np.datetime64("NaT")
            # In this case we specifically interpret NaT as a datetime, not
            # the timedelta interpretation we would get by returning self + NaT
            result = self._ndarray + NaT.to_datetime64().astype(f"M8[{self.unit}]")
            # Preserve our resolution
            return DatetimeArray._simple_new(result, dtype=result.dtype)

        other = Timestamp(other)
        self, other = self._ensure_matching_resos(other)
        self = cast("TimedeltaArray", self)

        other_i8, o_mask = self._get_i8_values_and_mask(other)
        result = add_overflowsafe(self.asi8, np.asarray(other_i8, dtype="i8"))
        res_values = result.view(f"M8[{self.unit}]")

        dtype = tz_to_dtype(tz=other.tz, unit=self.unit)
        res_values = result.view(f"M8[{self.unit}]")
        new_freq = self._get_arithmetic_result_freq(other)
        return DatetimeArray._simple_new(res_values, dtype=dtype, freq=new_freq)

    @final
    def _add_datetime_arraylike(self, other: DatetimeArray) -> DatetimeArray:
        # [FUNCTION DOC]
        # HELPER (final): time delta array plus a datetime array. Raises a type
        # error if self is not a time delta array; otherwise defers to the
        # datetime array's own addition (operands swapped) and returns its
        # result.
        if not lib.is_np_dtype(self.dtype, "m"):
            raise TypeError(
                f"cannot add {type(self).__name__} and {type(other).__name__}"
            )

        # defer to DatetimeArray.__add__
        return other + self

    @final
    def _sub_datetimelike_scalar(
        self, other: datetime | np.datetime64
    ) -> TimedeltaArray:
        # [FUNCTION DOC]
        # HELPER (final): datetime array minus a datetime scalar. Parameter other
        # is a datetime or numpy datetime. Raises a type error if self is not a
        # datetime array. A missing-like scalar is handled by subtracting the
        # missing marker (all-missing time deltas). Otherwise the scalar becomes
        # a timestamp, resolutions are matched, and the shared datetime
        # subtraction runs. Returns a time delta array.
        if self.dtype.kind != "M":
            raise TypeError(f"cannot subtract a datelike from a {type(self).__name__}")

        self = cast("DatetimeArray", self)
        # subtract a datetime from myself, yielding an ndarray[timedelta64[ns]]

        if isna(other):
            # i.e. np.datetime64("NaT")
            return self - NaT

        ts = Timestamp(other)

        self, ts = self._ensure_matching_resos(ts)
        return self._sub_datetimelike(ts)

    @final
    def _sub_datetime_arraylike(self, other: DatetimeArray) -> TimedeltaArray:
        # [FUNCTION DOC]
        # HELPER (final): datetime array minus a datetime array. Raises a type
        # error if self is not a datetime array and a value error if the lengths
        # differ (the message wrongly says "add"). Matches resolutions and runs
        # the shared datetime subtraction. Returns a time delta array.
        # KNOWN ISSUE: the length-mismatch error message says "cannot add
        # indices of unequal length" although this is a subtraction.
        if self.dtype.kind != "M":
            raise TypeError(f"cannot subtract a datelike from a {type(self).__name__}")

        if len(self) != len(other):
            raise ValueError("cannot add indices of unequal length")

        self = cast("DatetimeArray", self)

        self, other = self._ensure_matching_resos(other)
        return self._sub_datetimelike(other)

    @final
    def _sub_datetimelike(self, other: Timestamp | DatetimeArray) -> TimedeltaArray:
        # [FUNCTION DOC]
        # HELPER (final): shared core of datetime subtraction; resolutions must
        # already match. Parameter other is a timestamp or datetime array. A
        # naive/aware mismatch is re-raised with the word "compare" replaced by
        # "subtract" in the message. Computes an overflow-safe integer
        # difference and wraps it as a time delta array of self's unit, with a
        # frequency chosen by the arithmetic frequency helper. The returned
        # mask from the integer extraction helper is unused.
        self = cast("DatetimeArray", self)

        from pandas.core.arrays import TimedeltaArray

        try:
            self._assert_tzawareness_compat(other)
        except TypeError as err:
            new_message = str(err).replace("compare", "subtract")
            raise type(err)(new_message) from err

        other_i8, o_mask = self._get_i8_values_and_mask(other)
        res_values = add_overflowsafe(self.asi8, np.asarray(-other_i8, dtype="i8"))
        res_m8 = res_values.view(f"timedelta64[{self.unit}]")

        new_freq = self._get_arithmetic_result_freq(other)
        new_freq = cast("Tick | None", new_freq)
        return TimedeltaArray._simple_new(res_m8, dtype=res_m8.dtype, freq=new_freq)

    @final
    def _add_period(self, other: Period) -> PeriodArray:
        # [FUNCTION DOC]
        # HELPER (final): time delta array plus a period scalar. Raises a type
        # error if self is not a time delta array. Builds a period array of the
        # same shape filled with the scalar's ordinal and frequency and returns
        # the result of adding self to it (the period array's addition does the
        # work).
        if not lib.is_np_dtype(self.dtype, "m"):
            raise TypeError(f"cannot add Period to a {type(self).__name__}")

        # We will wrap in a PeriodArray and defer to the reversed operation
        from pandas.core.arrays.period import PeriodArray

        i8vals = np.broadcast_to(other.ordinal, self.shape)
        dtype = PeriodDtype(other.freq)
        parr = PeriodArray(i8vals, dtype=dtype)
        return parr + self

    def _add_offset(self, offset):
        # [FUNCTION DOC]
        # HELPER: abstract hook for adding a calendar offset. Parameter offset is
        # the offset. Always raises the abstract-method error; datetime and
        # period arrays override it.
        raise AbstractMethodError(self)

    def _add_timedeltalike_scalar(self, other):
        # [FUNCTION DOC]
        # HELPER: adds a time-delta-like scalar (time delta, numpy time delta or
        # fixed-duration offset). Parameter other is the scalar. A missing-like
        # scalar produces an all-missing array of self's dtype. Otherwise the
        # scalar is converted to a time delta, units are matched, and the
        # shared time-delta addition runs. Period arrays override this.
        # Returns an array of the same type as self.
        """
        Add a delta of a timedeltalike

        Returns
        -------
        Same type as self
        """
        if isna(other):
            # i.e np.timedelta64("NaT")
            new_values = np.empty(self.shape, dtype="i8").view(self._ndarray.dtype)
            new_values.fill(iNaT)
            return type(self)._simple_new(new_values, dtype=self.dtype)

        # PeriodArray overrides, so we only get here with DTA/TDA
        self = cast("DatetimeArray | TimedeltaArray", self)
        other = Timedelta(other)
        self, other = self._ensure_matching_resos(other)
        return self._add_timedeltalike(other)

    def _add_timedelta_arraylike(self, other: TimedeltaArray) -> Self:
        # [FUNCTION DOC]
        # HELPER: adds a time delta array element-wise. Parameter other is a
        # time delta array. Raises a value error if lengths differ (period
        # arrays override this method). Matches resolutions and runs the shared
        # time-delta addition. Returns an array of the same type as self.
        """
        Add a delta of a TimedeltaIndex

        Returns
        -------
        Same type as self
        """
        # overridden by PeriodArray

        if len(self) != len(other):
            raise ValueError("cannot add indices of unequal length")

        self, other = cast(
            "DatetimeArray | TimedeltaArray", self
        )._ensure_matching_resos(other)
        return self._add_timedeltalike(other)

    @final
    def _add_timedeltalike(self, other: Timedelta | TimedeltaArray) -> Self:
        # [FUNCTION DOC]
        # HELPER (final): shared integer addition of a time delta or time delta
        # array whose resolution already matches self. Uses overflow-safe
        # addition on the integer views (raising on overflow), reinterprets as
        # self's backing dtype, and builds a new array of self's type and dtype
        # with the frequency from the arithmetic frequency helper. The returned
        # mask from the integer extraction helper is unused.
        other_i8, o_mask = self._get_i8_values_and_mask(other)
        new_values = add_overflowsafe(self.asi8, np.asarray(other_i8, dtype="i8"))
        res_values = new_values.view(self._ndarray.dtype)

        new_freq = self._get_arithmetic_result_freq(other)

        # error: Unexpected keyword argument "freq" for "_simple_new" of "NDArrayBacked"
        return type(self)._simple_new(
            res_values,
            dtype=self.dtype,
            freq=new_freq,  # type: ignore[call-arg]
        )

    @final
    def _add_nat(self) -> Self:
        # [FUNCTION DOC]
        # HELPER (final): adds the missing marker. Raises a type error for period
        # arrays. For datetime and time delta arrays (where the marker acts like
        # a time delta) returns a new array of the same dtype, all missing, with
        # no frequency.
        """
        Add pd.NaT to self
        """
        if isinstance(self.dtype, PeriodDtype):
            raise TypeError(
                f"Cannot add {type(self).__name__} and {type(NaT).__name__}"
            )

        # GH#19124 pd.NaT is treated like a timedelta for both timedelta
        # and datetime dtypes
        result = np.empty(self.shape, dtype=np.int64)
        result.fill(iNaT)
        result = result.view(self._ndarray.dtype)  # preserve reso
        # error: Unexpected keyword argument "freq" for "_simple_new" of "NDArrayBacked"
        return type(self)._simple_new(
            result,
            dtype=self.dtype,
            freq=None,  # type: ignore[call-arg]
        )

    @final
    def _sub_nat(self) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER (final): subtracts the missing marker. Returns a plain numpy
        # time delta array, all missing, shaped like self: with self's unit for
        # datetime and time delta arrays, nanoseconds for periods.
        """
        Subtract pd.NaT from self
        """
        # GH#19124 Timedelta - datetime is not in general well-defined.
        # We make an exception for pd.NaT, which in this case quacks
        # like a timedelta.
        # For datetime64 dtypes by convention we treat NaT as a datetime, so
        # this subtraction returns a timedelta64 dtype.
        # For period dtype, timedelta64 is a close-enough return dtype.
        result = np.empty(self.shape, dtype=np.int64)
        result.fill(iNaT)
        if self.dtype.kind in "mM":
            # We can retain unit in dtype
            self = cast("DatetimeArray| TimedeltaArray", self)
            return result.view(f"timedelta64[{self.unit}]")
        else:
            return result.view("timedelta64[ns]")

    @final
    def _sub_periodlike(self, other: Period | PeriodArray) -> npt.NDArray[np.object_]:
        # [FUNCTION DOC]
        # HELPER (final): period array minus a period or period array. Raises a
        # type error if self is not a period array, and whatever the
        # compatibility check raises for a frequency mismatch. Computes
        # overflow-safe ordinal differences and turns each into a multiple of
        # the base offset of the frequency. Positions where either side is
        # missing are set to the missing marker. Returns an object array of
        # offsets.
        # If the operation is well-defined, we return an object-dtype ndarray
        # of DateOffsets.  Null entries are filled with pd.NaT
        if not isinstance(self.dtype, PeriodDtype):
            raise TypeError(
                f"cannot subtract {type(other).__name__} from {type(self).__name__}"
            )

        self = cast("PeriodArray", self)
        self._check_compatible_with(other)

        other_i8, o_mask = self._get_i8_values_and_mask(other)
        new_i8_data = add_overflowsafe(self.asi8, np.asarray(-other_i8, dtype="i8"))
        new_data = np.array([self.freq.base * x for x in new_i8_data])

        if o_mask is None:
            # i.e. Period scalar
            mask = self._isnan
        else:
            # i.e. PeriodArray
            mask = self._isnan | o_mask
        new_data[mask] = NaT
        return new_data

    @final
    def _addsub_object_array(self, other: npt.NDArray[np.object_], op) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER (final): add or subtract an object array (typically offsets).
        # Parameters: other is an object array; op is the addition or
        # subtraction function (asserted). If other has one element and self is
        # one-dimensional, applies op against that single element (so the result
        # may be an array of the array type rather than an object array).
        # Otherwise optionally emits a performance warning (when enabled),
        # asserts the shapes are equal (caller must broadcast), and applies op
        # between self converted to objects and other, returning the object
        # array.
        """
        Add or subtract array-like of DateOffset objects

        Parameters
        ----------
        other : np.ndarray[object]
        op : {operator.add, operator.sub}

        Returns
        -------
        np.ndarray[object]
            Except in fastpath case with length 1 where we operate on the
            contained scalar.
        """
        assert op in [operator.add, operator.sub]
        if len(other) == 1 and self.ndim == 1:
            # Note: without this special case, we could annotate return type
            #  as ndarray[object]
            # If both 1D then broadcasting is unambiguous
            return op(self, other[0])

        if get_option("performance_warnings"):
            warnings.warn(
                "Adding/subtracting object-dtype array to "
                f"{type(self).__name__} not vectorized.",
                PerformanceWarning,
                stacklevel=find_stack_level(),
            )

        # Caller is responsible for broadcasting if necessary
        assert self.shape == other.shape, (self.shape, other.shape)

        res_values = op(self.astype("O"), np.asarray(other))
        return res_values

    def _accumulate(self, name: str, *, skipna: bool = True, **kwargs) -> Self:
        # [FUNCTION DOC]
        # HELPER: running minimum/maximum. Parameters: name must be the
        # cumulative-minimum or cumulative-maximum name, otherwise a type error
        # is raised; skipna (keyword, default true) skips missing values; extra
        # keyword arguments are forwarded. Applies the matching accumulation
        # routine to a copy and returns a new array of the same type and dtype.
        if name not in {"cummin", "cummax"}:
            raise TypeError(f"Accumulation {name} not supported for {type(self)}")

        op = getattr(datetimelike_accumulations, name)
        result = op(self.copy(), skipna=skipna, **kwargs)

        return type(self)._simple_new(result, dtype=self.dtype)

    @unpack_zerodim_and_defer("__add__")
    def __add__(self, other):
        # [FUNCTION DOC]
        # Addition operator (decorated: zero-dimensional operands unpacked, and
        # unsupported operands defer). Parameter other is the right operand.
        # Dispatch order for scalars: missing marker, fixed-duration offsets /
        # time deltas, day offsets (treated as fixed 24 hours for datetime and
        # time delta arrays), other calendar offsets, datetimes, periods (time
        # delta arrays only), integers (period arrays only, scaled by the
        # frequency multiple; otherwise raises the integer-operation error). For
        # arrays: time delta dtype, object dtype (offsets), datetime dtypes
        # (returned directly), integer dtype (period only). Anything else
        # returns the "not implemented" signal. A numpy time delta result is
        # wrapped into a time delta array. Returns an array of the appropriate
        # type; may raise type errors from the helpers.
        other_dtype = getattr(other, "dtype", None)
        other = ensure_wrapped_if_datetimelike(other)

        # scalar others
        if other is NaT:
            result: np.ndarray | DatetimeLikeArrayMixin = self._add_nat()
        elif isinstance(other, (Tick, timedelta, np.timedelta64)):
            result = self._add_timedeltalike_scalar(other)
        elif isinstance(other, Day) and lib.is_np_dtype(self.dtype, "Mm"):
            # We treat this as Tick-like
            td = Timedelta(days=other.n).as_unit("s")
            result = self._add_timedeltalike_scalar(td)
        elif isinstance(other, BaseOffset):
            # specifically _not_ a Tick
            result = self._add_offset(other)
        elif isinstance(other, (datetime, np.datetime64)):
            result = self._add_datetimelike_scalar(other)
        elif isinstance(other, Period) and lib.is_np_dtype(self.dtype, "m"):
            result = self._add_period(other)
        elif lib.is_integer(other):
            # This check must come after the check for np.timedelta64
            # as is_integer returns True for these
            if not isinstance(self.dtype, PeriodDtype):
                raise integer_op_not_supported(self)
            obj = cast("PeriodArray", self)
            result = obj._addsub_int_array_or_scalar(other * obj.dtype._n, operator.add)

        # array-like others
        elif lib.is_np_dtype(other_dtype, "m"):
            # TimedeltaIndex, ndarray[timedelta64]
            result = self._add_timedelta_arraylike(other)
        elif is_object_dtype(other_dtype):
            # e.g. Array/Index of DateOffset objects
            result = self._addsub_object_array(other, operator.add)
        elif lib.is_np_dtype(other_dtype, "M") or isinstance(
            other_dtype, DatetimeTZDtype
        ):
            # DatetimeIndex, ndarray[datetime64]
            return self._add_datetime_arraylike(other)
        elif is_integer_dtype(other_dtype):
            if not isinstance(self.dtype, PeriodDtype):
                raise integer_op_not_supported(self)
            obj = cast("PeriodArray", self)
            result = obj._addsub_int_array_or_scalar(other * obj.dtype._n, operator.add)
        else:
            # Includes Categorical, other ExtensionArrays
            # For PeriodDtype, if self is a TimedeltaArray and other is a
            #  PeriodArray with  a timedelta-like (i.e. Tick) freq, this
            #  operation is valid.  Defer to the PeriodArray implementation.
            #  In remaining cases, this will end up raising TypeError.
            return NotImplemented

        if isinstance(result, np.ndarray) and lib.is_np_dtype(result.dtype, "m"):
            from pandas.core.arrays import TimedeltaArray

            return TimedeltaArray._from_sequence(result, dtype=result.dtype)
        return result

    def __radd__(self, other):
        # [FUNCTION DOC]
        # Reflected addition; addition is treated as commutative, so this simply
        # calls the forward addition and returns its result.
        # alias for __add__
        return self.__add__(other)

    @unpack_zerodim_and_defer("__sub__")
    def __sub__(self, other):
        # [FUNCTION DOC]
        # Subtraction operator (decorated like addition). Parameter other is the
        # right operand. Scalars: missing marker; fixed-duration offsets / time
        # deltas (added negated); day offsets (as 24 hours, negated); calendar
        # offsets (negated); datetimes; integers (period arrays only);
        # periods. Arrays: time delta dtype (added negated), object dtype
        # (offsets), datetime dtypes, period dtype, integer dtype (period only).
        # Anything else returns the "not implemented" signal. A numpy time
        # delta result is wrapped into a time delta array. Raises type errors
        # from helpers for unsupported combinations.
        other_dtype = getattr(other, "dtype", None)
        other = ensure_wrapped_if_datetimelike(other)

        # scalar others
        if other is NaT:
            result: np.ndarray | DatetimeLikeArrayMixin = self._sub_nat()
        elif isinstance(other, (Tick, timedelta, np.timedelta64)):
            result = self._add_timedeltalike_scalar(-other)
        elif isinstance(other, Day) and lib.is_np_dtype(self.dtype, "Mm"):
            # We treat this as Tick-like
            td = Timedelta(days=other.n).as_unit("s")
            result = self._add_timedeltalike_scalar(-td)
        elif isinstance(other, BaseOffset):
            # specifically _not_ a Tick
            result = self._add_offset(-other)
        elif isinstance(other, (datetime, np.datetime64)):
            result = self._sub_datetimelike_scalar(other)
        elif lib.is_integer(other):
            # This check must come after the check for np.timedelta64
            # as is_integer returns True for these
            if not isinstance(self.dtype, PeriodDtype):
                raise integer_op_not_supported(self)
            obj = cast("PeriodArray", self)
            result = obj._addsub_int_array_or_scalar(other * obj.dtype._n, operator.sub)

        elif isinstance(other, Period):
            result = self._sub_periodlike(other)

        # array-like others
        elif lib.is_np_dtype(other_dtype, "m"):
            # TimedeltaIndex, ndarray[timedelta64]
            result = self._add_timedelta_arraylike(-other)
        elif is_object_dtype(other_dtype):
            # e.g. Array/Index of DateOffset objects
            result = self._addsub_object_array(other, operator.sub)
        elif lib.is_np_dtype(other_dtype, "M") or isinstance(
            other_dtype, DatetimeTZDtype
        ):
            # DatetimeIndex, ndarray[datetime64]
            result = self._sub_datetime_arraylike(other)
        elif isinstance(other_dtype, PeriodDtype):
            # PeriodIndex
            result = self._sub_periodlike(other)
        elif is_integer_dtype(other_dtype):
            if not isinstance(self.dtype, PeriodDtype):
                raise integer_op_not_supported(self)
            obj = cast("PeriodArray", self)
            result = obj._addsub_int_array_or_scalar(other * obj.dtype._n, operator.sub)
        else:
            # Includes ExtensionArrays, float_dtype
            return NotImplemented

        if isinstance(result, np.ndarray) and lib.is_np_dtype(result.dtype, "m"):
            from pandas.core.arrays import TimedeltaArray

            return TimedeltaArray._from_sequence(result, dtype=result.dtype)
        return result

    def __rsub__(self, other):
        # [FUNCTION DOC]
        # Reflected subtraction (other minus self). Parameter other is the left
        # operand. If other is datetime data and self is time deltas, other is
        # wrapped as a datetime array (unless it already is a datetime-like
        # array) and the subtraction is flipped; a scalar numpy datetime is
        # converted to a timestamp first. If self is datetimes and other has a
        # non-datetime dtype, raises a type error. If self is periods and other
        # is time-delta typed, raises a type error. If self is time deltas, the
        # result is the negation of self plus other. Otherwise computes self
        # minus other and negates it, raising a type error if that forward
        # result is datetime-typed (unsupported reversed subtraction).
        other_dtype = getattr(other, "dtype", None)
        other_is_dt64 = lib.is_np_dtype(other_dtype, "M") or isinstance(
            other_dtype, DatetimeTZDtype
        )

        if other_is_dt64 and lib.is_np_dtype(self.dtype, "m"):
            # ndarray[datetime64] cannot be subtracted from self, so
            # we need to wrap in DatetimeArray/Index and flip the operation
            if lib.is_scalar(other):
                # i.e. np.datetime64 object
                return Timestamp(other) - self
            if not isinstance(other, DatetimeLikeArrayMixin):
                # Avoid down-casting DatetimeIndex
                from pandas.core.arrays import DatetimeArray

                other = DatetimeArray._from_sequence(other, dtype=other.dtype)
            return other - self
        elif self.dtype.kind == "M" and hasattr(other, "dtype") and not other_is_dt64:
            # GH#19959 datetime - datetime is well-defined as timedelta,
            # but any other type - datetime is not well-defined.
            raise TypeError(
                f"cannot subtract {type(self).__name__} from "
                f"{type(other).__name__}[{other.dtype}]"
            )
        elif isinstance(self.dtype, PeriodDtype) and lib.is_np_dtype(other_dtype, "m"):
            # TODO: Can we simplify/generalize these cases at all?
            raise TypeError(f"cannot subtract {type(self).__name__} from {other.dtype}")
        elif lib.is_np_dtype(self.dtype, "m"):
            self = cast("TimedeltaArray", self)
            return (-self) + other

        flipped = self - other
        if flipped.dtype.kind == "M":
            # GH#59571 give a more helpful exception message
            raise TypeError(
                f"cannot subtract {type(self).__name__} from {type(other).__name__}"
            )
        # We get here with e.g. datetime objects
        return -flipped

    def __iadd__(self, other) -> Self:
        # [FUNCTION DOC]
        # In-place addition. Parameter other is the right operand. Computes the
        # sum, writes it into self's storage (which clears the frequency), then
        # restores the frequency from the sum for non-period arrays. Mutates and
        # returns self.
        result = self + other
        self[:] = result[:]

        if not isinstance(self.dtype, PeriodDtype):
            # restore freq, which is invalidated by setitem
            self._freq = result.freq
        return self

    def __isub__(self, other) -> Self:
        # [FUNCTION DOC]
        # In-place subtraction. Parameter other is the right operand. Computes
        # the difference, writes it into self's storage (clearing the
        # frequency), then restores the frequency from the difference for
        # non-period arrays. Mutates and returns self.
        result = self - other
        self[:] = result[:]

        if not isinstance(self.dtype, PeriodDtype):
            # restore freq, which is invalidated by setitem
            self._freq = result.freq
        return self

    # --------------------------------------------------------------
    # Reductions

    @_period_dispatch
    def _quantile(
        self,
        qs: npt.NDArray[np.float64],
        interpolation: str,
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: quantile computation (for period arrays this runs via the
        # datetime-view dispatch decorator). Parameters: qs is a float array of
        # quantiles in zero-to-one; interpolation is the interpolation method
        # name. Delegates to the numpy-backed parent and returns an array of the
        # same type.
        return super()._quantile(qs=qs, interpolation=interpolation)

    @_period_dispatch
    def min(self, *, axis: AxisInt | None = None, skipna: bool = True, **kwargs):
        # [FUNCTION DOC]
        # Minimum (period arrays go through the datetime-view dispatch
        # decorator). Parameters (keyword only): axis is none or an axis number
        # validated against the dimension count; skipna (default true) ignores
        # missing values; other keyword arguments are validated against numpy's
        # accepted ones and otherwise rejected. Uses the nan-aware minimum on the
        # backing array and wraps the result as a scalar (or array for an axis
        # reduction). Returns the missing marker if empty or all-missing with
        # skipna.
        """
        Return the minimum value of the Array or minimum along
        an axis.

        See Also
        --------
        numpy.ndarray.min
        Index.min : Return the minimum value in an Index.
        Series.min : Return the minimum value in a Series.
        """
        nv.validate_min((), kwargs)
        nv.validate_minmax_axis(axis, self.ndim)

        result = nanops.nanmin(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    @_period_dispatch
    def max(self, *, axis: AxisInt | None = None, skipna: bool = True, **kwargs):
        # [FUNCTION DOC]
        # Maximum; mirrors the minimum method (same parameters and validation)
        # but uses the nan-aware maximum on the backing array. Returns a scalar,
        # or the missing marker when empty or all-missing with skipna.
        """
        Return the maximum value of the Array or maximum along
        an axis.

        See Also
        --------
        numpy.ndarray.max
        Index.max : Return the maximum value in an Index.
        Series.max : Return the maximum value in a Series.
        """
        nv.validate_max((), kwargs)
        nv.validate_minmax_axis(axis, self.ndim)

        result = nanops.nanmax(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def mean(self, *, skipna: bool = True, axis: AxisInt | None = 0):
        # [FUNCTION DOC]
        # Arithmetic mean of datetimes or time deltas. Parameters (keyword only):
        # skipna (default true) ignores missing values; axis (default 0) is the
        # axis to reduce. Raises a type error for period arrays (ambiguous
        # meaning, suggesting conversion to timestamps). Uses the nan-aware mean
        # on the backing array with the missing mask and wraps the result as a
        # scalar (or array).
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
        if isinstance(self.dtype, PeriodDtype):
            # See discussion in GH#24757
            raise TypeError(
                f"mean is not implemented for {type(self).__name__} since the "
                "meaning is ambiguous.  An alternative is "
                "obj.to_timestamp(how='start').mean()"
            )

        result = nanops.nanmean(
            self._ndarray, axis=axis, skipna=skipna, mask=self.isna()
        )
        return self._wrap_reduction_result(axis, result)

    @_period_dispatch
    def median(self, *, axis: AxisInt | None = None, skipna: bool = True, **kwargs):
        # [FUNCTION DOC]
        # Median (period arrays via the datetime-view dispatch decorator).
        # Parameters (keyword only): axis is none or an axis number; skipna
        # (default true) ignores missing values; other keyword arguments are
        # validated against numpy's accepted set. Raises a value error when the
        # absolute axis is not smaller than the dimension count. Returns the
        # wrapped scalar/array result of the nan-aware median.
        nv.validate_median((), kwargs)

        if axis is not None and abs(axis) >= self.ndim:
            raise ValueError("abs(axis) must be less than ndim")

        result = nanops.nanmedian(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def _mode(self, dropna: bool = True):
        # [FUNCTION DOC]
        # HELPER: most frequent value(s). Parameter dropna (default true) excludes
        # missing values. Computes the mode of the integer view (using the
        # missing mask when dropping) and rebuilds an array of the same type
        # from the result. Returns that array (possibly several modes, sorted by
        # the mode routine).
        mask = None
        if dropna:
            mask = self.isna()

        i8modes, _ = algorithms.mode(self.view("i8"), mask=mask)
        npmodes = i8modes.view(self._ndarray.dtype)
        npmodes = cast(np.ndarray, npmodes)
        return self._from_backing_data(npmodes)

    # ------------------------------------------------------------------
    # GroupBy Methods

    def _groupby_op(
        self,
        *,
        how: str,
        has_dropped_na: bool,
        min_count: int,
        ngroups: int,
        ids: npt.NDArray[np.intp],
        **kwargs,
    ):
        # [FUNCTION DOC]
        # HELPER: group-wise aggregation entry point used by the groupby engine.
        # Parameters (keyword only): how is the operation name; has_dropped_na
        # says whether missing group keys were dropped; min_count is the minimum
        # non-missing count; ngroups is the number of groups; ids is the integer
        # group label for each row; extra keyword arguments are forwarded to the
        # compiled operation. Rejects unsupported operations with type errors:
        # for datetimes and periods, sum/product/cumulative sum and product/
        # variance/skew/kurtosis plus any/all; for time deltas, product,
        # cumulative product, skew, kurtosis, variance. All supported
        # operations are order-based, so the data is viewed as naive nanosecond
        # datetimes, run through the compiled groupby operator, and the result
        # is mapped back: operations on the no-cast list (rank) return the raw
        # result; standard deviation / standard error give a time delta array
        # (a type error for periods); everything else is reinterpreted as the
        # original dtype and rebuilt as an array of the same type.
        dtype = self.dtype
        if dtype.kind == "M":
            # Adding/multiplying datetimes is not valid
            if how in ["sum", "prod", "cumsum", "cumprod", "var", "skew", "kurt"]:
                raise TypeError(f"datetime64 type does not support operation '{how}'")
            if how in ["any", "all"]:
                # GH#34479
                raise TypeError(
                    f"'{how}' with datetime64 dtypes is no longer supported. "
                    f"Use (obj != pd.Timestamp(0)).{how}() instead."
                )

        elif isinstance(dtype, PeriodDtype):
            # Adding/multiplying Periods is not valid
            if how in ["sum", "prod", "cumsum", "cumprod", "var", "skew", "kurt"]:
                raise TypeError(f"Period type does not support {how} operations")
            if how in ["any", "all"]:
                # GH#34479
                raise TypeError(
                    f"'{how}' with PeriodDtype is no longer supported. "
                    f"Use (obj != pd.Period(0, freq)).{how}() instead."
                )
        # timedeltas we can add but not multiply
        elif how in ["prod", "cumprod", "skew", "kurt", "var"]:
            raise TypeError(f"timedelta64 type does not support {how} operations")

        # All of the functions implemented here are ordinal, so we can
        #  operate on the tz-naive equivalents
        npvalues = self._ndarray.view("M8[ns]")

        from pandas.core.groupby.ops import WrappedCythonOp

        kind = WrappedCythonOp.get_kind_from_how(how)
        op = WrappedCythonOp(how=how, kind=kind, has_dropped_na=has_dropped_na)

        res_values = op._cython_op_ndim_compat(
            npvalues,
            min_count=min_count,
            ngroups=ngroups,
            comp_ids=ids,
            mask=None,
            **kwargs,
        )

        if op.how in op.cast_blocklist:
            # i.e. how in ["rank"], since other cast_blocklist methods don't go
            #  through cython_operation
            return res_values

        # We did a view to M8[ns] above, now we go the other direction
        assert res_values.dtype == "M8[ns]"
        if how in ["std", "sem"]:
            from pandas.core.arrays import TimedeltaArray

            if isinstance(self.dtype, PeriodDtype):
                raise TypeError("'std' and 'sem' are not valid for PeriodDtype")
            self = cast("DatetimeArray | TimedeltaArray", self)
            new_dtype = f"m8[{self.unit}]"
            res_values = res_values.view(new_dtype)
            return TimedeltaArray._simple_new(res_values, dtype=res_values.dtype)

        res_values = res_values.view(self._ndarray.dtype)
        return self._from_backing_data(res_values)


class DatelikeOps(DatetimeLikeArrayMixin):
    # [CLASS DOC]
    # Mixin for arrays holding calendar dates (datetime and period arrays, not
    # time deltas). Subclass of the datetime-like mixin; adds the string
    # formatting method (strftime). No extra state.
    """
    Common ops for DatetimeIndex/PeriodIndex, but not TimedeltaIndex.
    """

    def strftime(self, date_format: str) -> npt.NDArray[np.object_]:
        # [FUNCTION DOC]
        # Formats each value as text. Parameter date_format is a standard-library
        # strftime pattern (period arrays also accept their extra directives).
        # Calls the native-type formatter with that pattern and not-a-number as
        # the missing-value text. Returns an object array of strings, or, when
        # the global string-dtype option is on, an array of the string extension
        # dtype using not-a-number as its missing value.
        """
        Convert to Index using specified date_format.

        Return an Index of formatted strings specified by date_format, which
        supports the same string format as the python standard library. Details
        of the string format can be found in `python string format
        doc <https://docs.python.org/3/library/datetime.html
        #strftime-and-strptime-behavior>`__.

        Formats supported by the C `strftime` API but not by the python string format
        doc (such as `"%R"`, `"%r"`) are not officially supported and should be
        preferably replaced with their supported equivalents (such as `"%H:%M"`,
        `"%I:%M:%S %p"`).

        Note that `PeriodIndex` support additional directives, detailed in
        `Period.strftime`.

        Parameters
        ----------
        date_format : str
            Date format string (e.g. "%%Y-%%m-%%d").

        Returns
        -------
        ndarray[object]
            NumPy ndarray of formatted strings.

        See Also
        --------
        to_datetime : Convert the given argument to datetime.
        DatetimeIndex.normalize : Return DatetimeIndex with times to midnight.
        DatetimeIndex.round : Round the DatetimeIndex to the specified freq.
        DatetimeIndex.floor : Floor the DatetimeIndex to the specified freq.
        Timestamp.strftime : Format a single Timestamp.
        Period.strftime : Format a single Period.

        Examples
        --------
        >>> rng = pd.date_range(pd.Timestamp("2018-03-10 09:00"), periods=3, freq="s")
        >>> rng.strftime("%B %d, %Y, %r")
        Index(['March 10, 2018, 09:00:00 AM', 'March 10, 2018, 09:00:01 AM',
               'March 10, 2018, 09:00:02 AM'],
              dtype='str')
        """
        result = self._format_native_types(date_format=date_format, na_rep=np.nan)
        if using_string_dtype():
            from pandas import StringDtype

            return pd_array(result, dtype=StringDtype(na_value=np.nan))  # type: ignore[return-value]
        return result.astype(object, copy=False)


class TimelikeOps(DatetimeLikeArrayMixin):
    # [CLASS DOC]
    # Mixin for arrays with a numpy datetime/time-delta backing and an optional
    # fixed frequency (datetime and time delta arrays, not periods). Subclass of
    # the datetime-like mixin. State: the stored frequency offset (or none),
    # accessed through a validating property; the unit and the compiled
    # resolution code, both derived from the dtype and cached. Key methods:
    # frequency property, pinning/validating frequencies in constructors, unit
    # conversion, rounding (round, floor, ceil), any/all, factorize, take,
    # concatenation, copy, and interpolation. Subclasses must implement the
    # dtype validation and range generation hooks.
    # Invariant: when a frequency is set, the values lie exactly on it (checked
    # at assignment); mutation clears it.
    """
    Common ops for TimedeltaIndex/DatetimeIndex, but not PeriodIndex.
    """

    @classmethod
    def _validate_dtype(cls, values, dtype):
        # [FUNCTION DOC]
        # HELPER: abstract hook that validates a dtype against a data buffer
        # during construction. Parameters values and dtype. Always raises the
        # abstract-method error here.
        raise AbstractMethodError(cls)

    @property
    def freq(self):
        # [FUNCTION DOC]
        # Property returning the stored frequency offset, or none.
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
        return self._freq

    @freq.setter
    def freq(self, value) -> None:
        # [FUNCTION DOC]
        # Frequency setter with validation. Parameter value is an offset, a
        # frequency string, or none (clears). Non-none values are converted to an
        # offset and validated against the data (a value error if the values do
        # not conform). For time deltas the offset must be a fixed-duration or
        # day offset (otherwise a type error). Multi-dimensional arrays cannot
        # have a frequency (value error). On success stores the offset.
        if value is not None:
            value = to_offset(value)
            self._validate_frequency(self, value)
            if self.dtype.kind == "m" and not isinstance(value, (Tick, Day)):
                raise TypeError("TimedeltaArray/Index freq must be a Tick")

            if self.ndim > 1:
                raise ValueError("Cannot set freq with ndim > 1")

        self._freq = value

    @final
    def _maybe_pin_freq(self, freq, validate_kwds: dict) -> None:
        # [FUNCTION DOC]
        # HELPER (final): constructor helper that decides the stored frequency.
        # Parameters: freq is what the caller asked for: none (force no
        # frequency), the string "infer" (infer if not already known), the
        # not-given sentinel (keep whatever was inferred from the input), or an
        # offset/string; validate_kwds are extra keyword arguments passed to the
        # validator (such as the ambiguity mode). With an explicit frequency:
        # if none is currently stored it is validated against the values and
        # stored; otherwise it is only checked for conflict with the stored one
        # (value error on mismatch). Mutates self; returns nothing.
        """
        Constructor helper to pin the appropriate `freq` attribute.  Assumes
        that self._freq is currently set to any freq inferred in
        _from_sequence_not_strict.
        """
        if freq is None:
            # user explicitly passed None -> override any inferred_freq
            self._freq = None
        elif freq == "infer":
            # if self._freq is *not* None then we already inferred a freq
            #  and there is nothing left to do
            if self._freq is None:
                # Set _freq directly to bypass duplicative _validate_frequency
                # check.
                self._freq = to_offset(self.inferred_freq)  # type: ignore[assignment]
        elif freq is lib.no_default:
            # user did not specify anything, keep inferred freq if the original
            #  data had one, otherwise do nothing
            pass
        elif self._freq is None:
            # We cannot inherit a freq from the data, so we need to validate
            #  the user-passed freq
            freq = to_offset(freq)
            type(self)._validate_frequency(self, freq, **validate_kwds)
            self._freq = freq
        else:
            # Otherwise we just need to check that the user-passed freq
            #  doesn't conflict with the one we already have.
            freq = to_offset(freq)
            _validate_inferred_freq(freq, self._freq)

    @final
    @classmethod
    def _validate_frequency(cls, index, freq: BaseOffset, **kwargs) -> None:
        # [FUNCTION DOC]
        # HELPER (final class method): checks that a frequency matches the data.
        # Parameters: index is the array/index of values; freq is the offset to
        # validate; extra keyword arguments are forwarded to the range
        # generator. Returns nothing. Succeeds immediately for empty data or
        # when the inferred frequency string equals the offset's string.
        # Otherwise generates a range from the first value with the same length
        # and offset and requires it to equal the data; any mismatch (or a
        # failure such as a missing first value) raises a value error saying the
        # inferred frequency does not conform to the passed one, except an error
        # mentioning non-fixed frequencies is re-raised unchanged.
        """
        Validate that a frequency is compatible with the values of a given
        Datetime Array/Index or Timedelta Array/Index

        Parameters
        ----------
        index : DatetimeIndex or TimedeltaIndex
            The index on which to determine if the given frequency is valid
        freq : DateOffset
            The frequency to validate
        """
        inferred = index.inferred_freq
        if index.size == 0 or inferred == freq.freqstr:
            return None

        try:
            on_freq = cls._generate_range(
                start=index[0],
                end=None,
                periods=len(index),
                freq=freq,
                unit=index.unit,
                **kwargs,
            )
            if not np.array_equal(index.asi8, on_freq.asi8):
                raise ValueError
        except ValueError as err:
            if "non-fixed" in str(err):
                # non-fixed frequencies are not meaningful for timedelta64;
                #  we retain that error message
                raise err
            # GH#11587 the main way this is reached is if the `np.array_equal`
            #  check above is False.  This can also be reached if index[0]
            #  is `NaT`, in which case the call to `cls._generate_range` will
            #  raise a ValueError, which we re-raise with a more targeted
            #  message.
            raise ValueError(
                f"Inferred frequency {inferred} from passed values "
                f"does not conform to passed frequency {freq.freqstr}"
            ) from err

    @classmethod
    def _generate_range(
        cls, start, end, periods: int | None, freq, *args, **kwargs
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: abstract hook that generates a regular range. Parameters: start,
        # end, periods, freq, plus extra arguments. Always raises the
        # abstract-method error here; datetime and time delta arrays implement it.
        raise AbstractMethodError(cls)

    # --------------------------------------------------------------

    @cache_readonly
    def _creso(self) -> int:
        # [FUNCTION DOC]
        # HELPER: cached property giving the compiled library's integer code of
        # the backing dtype's time unit.
        return get_unit_from_dtype(self._ndarray.dtype)

    @cache_readonly
    def unit(self) -> TimeUnit:
        # [FUNCTION DOC]
        # Cached property returning the resolution of the data as one of the
        # strings for seconds, milliseconds, microseconds or nanoseconds,
        # derived from the dtype by the module-level unit helper.
        """
        The precision unit of the datetime data.

        Returns the precision unit for the dtype.
        It means the smallest time frame that can be stored within this dtype.

        Returns
        -------
        str
            Unit string representation (e.g. "ns").

        See Also
        --------
        TimelikeOps.as_unit : Converts to a specific unit.

        Examples
        --------
        >>> idx = pd.DatetimeIndex(["2020-01-02 01:02:03.004005006"])
        >>> idx.unit
        'ns'
        >>> idx.as_unit("s").unit
        's'
        """
        # error: Incompatible return value type (got "str", expected
        # "Literal['s', 'ms', 'us', 'ns']")  [return-value]
        return dtype_to_unit(self.dtype)  # type: ignore[return-value,arg-type]

    def as_unit(self, unit: TimeUnit, round_ok: bool = True) -> Self:
        # [FUNCTION DOC]
        # Converts to a different resolution. Parameters: unit must be one of the
        # seconds, milliseconds, microseconds, nanoseconds strings (otherwise a
        # value error); round_ok (default true) allows lossy conversion, and if
        # false a conversion needing rounding raises a value error. Overflow
        # during the cast raises an out-of-bounds error from the compiled cast.
        # The new dtype keeps the kind and, for timezone-aware data, the
        # timezone. Returns a new array of the same type with the same
        # frequency.
        """
        Convert to a dtype with the given unit resolution.

        The limits of timestamp representation depend on the chosen resolution.
        Different resolutions can be converted to each other through as_unit.

        Parameters
        ----------
        unit : {'s', 'ms', 'us', 'ns'}
        round_ok : bool, default True
            If False and the conversion requires rounding, raise ValueError.

        Returns
        -------
        same type as self
            Converted to the specified unit.

        See Also
        --------
        Timestamp.as_unit : Convert to the given unit.

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
        if unit not in ["s", "ms", "us", "ns"]:
            raise ValueError("Supported units are 's', 'ms', 'us', 'ns'")

        dtype = np.dtype(f"{self.dtype.kind}8[{unit}]")
        new_values = astype_overflowsafe(self._ndarray, dtype, round_ok=round_ok)

        if isinstance(self.dtype, np.dtype):
            new_dtype = new_values.dtype
        else:
            tz = cast("DatetimeArray", self).tz
            new_dtype = DatetimeTZDtype(tz=tz, unit=unit)

        # error: Unexpected keyword argument "freq" for "_simple_new" of
        # "NDArrayBacked"  [call-arg]
        return type(self)._simple_new(
            new_values,
            dtype=new_dtype,
            freq=self.freq,  # type: ignore[call-arg]
        )

    # TODO: annotate other as DatetimeArray | TimedeltaArray | Timestamp | Timedelta
    #  with the return type matching input type.  TypeVar?
    def _ensure_matching_resos(self, other):
        # [FUNCTION DOC]
        # HELPER: brings two operands to the finer of their two resolutions.
        # Parameter other is a datetime-like array or scalar with a resolution
        # attribute. Whichever operand has the coarser unit is converted to the
        # other's unit (round-trip lossless since finer). Returns a pair
        # (possibly new self, possibly new other).
        if self._creso != other._creso:
            # Just as with Timestamp/Timedelta, we cast to the higher resolution
            if self._creso < other._creso:
                self = self.as_unit(other.unit)
            else:
                other = other.as_unit(self.unit)
        return self, other

    # --------------------------------------------------------------

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        # [FUNCTION DOC]
        # Numpy universal-function protocol. Parameters: ufunc is the numpy
        # function; method is how it was invoked; inputs are the operands; extra
        # keyword arguments are forwarded. The single-operand missing/infinite/
        # finite checks applied to self run directly on the backing array
        # (newer numpy no longer raises for these on datetime kinds). Everything
        # else defers to the parent protocol. Returns the ufunc result.
        if (
            ufunc in [np.isnan, np.isinf, np.isfinite]
            and len(inputs) == 1
            and inputs[0] is self
        ):
            # numpy 1.18 changed isinf and isnan to not raise on dt64/td64
            return getattr(ufunc, method)(self._ndarray, **kwargs)

        return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)

    def _round(self, freq, mode, ambiguous, nonexistent):
        # [FUNCTION DOC]
        # HELPER: shared implementation of rounding, flooring and ceiling.
        # Parameters: freq is a fixed frequency (string or offset); mode is the
        # rounding mode enum (nearest-even, toward minus infinity, toward plus
        # infinity); ambiguous and nonexistent are the localization modes used
        # for timezone-aware data. Aware data is rounded in local wall time
        # (convert to naive, round, localize back with the given modes). Naive
        # data: the frequency is converted to a count of the array's units; a
        # zero count returns a copy; otherwise the integer values are rounded to
        # a multiple of that count, positions that were missing are restored,
        # and a new array is built (no frequency kept). Errors for non-fixed
        # frequencies come from the unit conversion. Returns a new array.
        # round the local times
        if isinstance(self.dtype, DatetimeTZDtype):
            # operate on naive timestamps, then convert back to aware
            self = cast("DatetimeArray", self)
            naive = self.tz_localize(None)
            result = naive._round(freq, mode, ambiguous, nonexistent)
            return result.tz_localize(
                self.tz, ambiguous=ambiguous, nonexistent=nonexistent
            )

        values = self.view("i8")
        values = cast(np.ndarray, values)
        nanos = get_unit_for_round(freq, self._creso)
        if nanos == 0:
            # GH 52761
            return self.copy()
        result_i8 = round_nsint64(values, mode, nanos)
        result = self._maybe_mask_results(result_i8, fill_value=iNaT)
        result = result.view(self._ndarray.dtype)
        return self._simple_new(result, dtype=self.dtype)

    def round(
        self,
        freq,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ) -> Self:
        # [FUNCTION DOC]
        # Rounds to the nearest multiple of a fixed frequency, ties to even.
        # Parameters: freq is a fixed frequency string/offset; ambiguous and
        # nonexistent (default raise) only matter for timezone-aware data and
        # control re-localization after rounding in local time. Delegates to the
        # shared rounding helper. Raises a value error for non-fixed
        # frequencies. Returns a new array.
        """
        Perform round operation on the data to the specified `freq`.

        Parameters
        ----------
        freq : str or Offset
            The frequency level to round the index to. Must be a fixed
            frequency like 's' (second) not 'ME' (month end). See
            :ref:`frequency aliases <timeseries.offset_aliases>` for
            a list of possible `freq` values.
        ambiguous : 'infer', bool-ndarray, 'NaT', default 'raise'
            Only relevant for DatetimeIndex:

            - 'infer' will attempt to infer fall dst-transition hours based on
              order
            - bool-ndarray where True signifies a DST time, False designates
              a non-DST time (note that this flag is only applicable for
              ambiguous times)
            - 'NaT' will return NaT where there are ambiguous times
            - 'raise' will raise a ValueError if there are ambiguous
              times.

        nonexistent : 'shift_forward', 'shift_backward', 'NaT', timedelta, \
            default 'raise'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            - 'shift_forward' will shift the nonexistent time forward to the
              closest existing time
            - 'shift_backward' will shift the nonexistent time backward to the
              closest existing time
            - 'NaT' will return NaT where there are nonexistent times
            - timedelta objects will shift nonexistent times by the timedelta
            - 'raise' will raise a ValueError if there are
              nonexistent times.

        Returns
        -------
        DatetimeIndex, TimedeltaIndex, or Series
            Index of the same type for a DatetimeIndex or TimedeltaIndex,
            or a Series with the same index for a Series.

        Raises
        ------
        ValueError if the `freq` cannot be converted.

        See Also
        --------
        DatetimeIndex.floor :
            Perform floor operation on the data to the specified `freq`.
        DatetimeIndex.snap :
            Snap time stamps to nearest occurring frequency.

        Notes
        -----
        If the timestamps have a timezone, rounding will take place relative to the
        local ("wall") time and re-localized to the same timezone. When rounding
        near daylight savings time, use ``nonexistent`` and ``ambiguous`` to
        control the re-localization behavior.

        Examples
        --------
        **DatetimeIndex**

        >>> rng = pd.date_range("1/1/2018 11:59:00", periods=3, freq="min")
        >>> rng
        DatetimeIndex(['2018-01-01 11:59:00', '2018-01-01 12:00:00',
                       '2018-01-01 12:01:00'],
                      dtype='datetime64[us]', freq='min')

        >>> rng.round('h')
        DatetimeIndex(['2018-01-01 12:00:00', '2018-01-01 12:00:00',
                       '2018-01-01 12:00:00'],
                      dtype='datetime64[us]', freq=None)

        **Series**

        >>> pd.Series(rng).dt.round("h")
        0   2018-01-01 12:00:00
        1   2018-01-01 12:00:00
        2   2018-01-01 12:00:00
        dtype: datetime64[us]

        When rounding near a daylight savings time transition, use ``ambiguous`` or
        ``nonexistent`` to control how the timestamp should be re-localized.

        >>> rng_tz = pd.DatetimeIndex(["2021-10-31 03:30:00"], tz="Europe/Amsterdam")

        >>> rng_tz.floor("2h", ambiguous=False)
        DatetimeIndex(['2021-10-31 02:00:00+01:00'],
                      dtype='datetime64[us, Europe/Amsterdam]', freq=None)

        >>> rng_tz.floor("2h", ambiguous=True)
        DatetimeIndex(['2021-10-31 02:00:00+02:00'],
                      dtype='datetime64[us, Europe/Amsterdam]', freq=None)
        """
        return self._round(freq, RoundTo.NEAREST_HALF_EVEN, ambiguous, nonexistent)

    def floor(
        self,
        freq,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ) -> Self:
        # [FUNCTION DOC]
        # Rounds down (toward minus infinity) to a multiple of a fixed
        # frequency. Same parameters, errors and result as the round method, but
        # with the floor rounding mode.
        """
        Perform floor operation on the data to the specified `freq`.

        Parameters
        ----------
        freq : str or Offset
            The frequency level to floor the index to. Must be a fixed
            frequency like 's' (second) not 'ME' (month end). See
            :ref:`frequency aliases <timeseries.offset_aliases>` for
            a list of possible `freq` values.
        ambiguous : 'infer', bool-ndarray, 'NaT', default 'raise'
            Only relevant for DatetimeIndex:

            - 'infer' will attempt to infer fall dst-transition hours based on
              order
            - bool-ndarray where True signifies a DST time, False designates
              a non-DST time (note that this flag is only applicable for
              ambiguous times)
            - 'NaT' will return NaT where there are ambiguous times
            - 'raise' will raise a ValueError if there are ambiguous
              times.

        nonexistent : 'shift_forward', 'shift_backward', 'NaT', timedelta, \
            default 'raise'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            - 'shift_forward' will shift the nonexistent time forward to the
              closest existing time
            - 'shift_backward' will shift the nonexistent time backward to the
              closest existing time
            - 'NaT' will return NaT where there are nonexistent times
            - timedelta objects will shift nonexistent times by the timedelta
            - 'raise' will raise a ValueError if there are
              nonexistent times.

        Returns
        -------
        DatetimeIndex, TimedeltaIndex, or Series
            Index of the same type for a DatetimeIndex or TimedeltaIndex,
            or a Series with the same index for a Series.

        Raises
        ------
        ValueError if the `freq` cannot be converted.

        See Also
        --------
        DatetimeIndex.floor :
            Perform floor operation on the data to the specified `freq`.
        DatetimeIndex.snap :
            Snap time stamps to nearest occurring frequency.

        Notes
        -----
        If the timestamps have a timezone, flooring will take place relative to the
        local ("wall") time and re-localized to the same timezone. When flooring
        near daylight savings time, use ``nonexistent`` and ``ambiguous`` to
        control the re-localization behavior.

        Examples
        --------
        **DatetimeIndex**

        >>> rng = pd.date_range("1/1/2018 11:59:00", periods=3, freq="min")
        >>> rng
        DatetimeIndex(['2018-01-01 11:59:00', '2018-01-01 12:00:00',
                       '2018-01-01 12:01:00'],
                      dtype='datetime64[us]', freq='min')

        >>> rng.floor('h')
        DatetimeIndex(['2018-01-01 11:00:00', '2018-01-01 12:00:00',
                       '2018-01-01 12:00:00'],
                      dtype='datetime64[us]', freq=None)

        **Series**

        >>> pd.Series(rng).dt.floor("h")
        0   2018-01-01 11:00:00
        1   2018-01-01 12:00:00
        2   2018-01-01 12:00:00
        dtype: datetime64[us]

        When rounding near a daylight savings time transition, use ``ambiguous`` or
        ``nonexistent`` to control how the timestamp should be re-localized.

        >>> rng_tz = pd.DatetimeIndex(["2021-10-31 03:30:00"], tz="Europe/Amsterdam")

        >>> rng_tz.floor("2h", ambiguous=False)
        DatetimeIndex(['2021-10-31 02:00:00+01:00'],
                     dtype='datetime64[us, Europe/Amsterdam]', freq=None)

        >>> rng_tz.floor("2h", ambiguous=True)
        DatetimeIndex(['2021-10-31 02:00:00+02:00'],
                      dtype='datetime64[us, Europe/Amsterdam]', freq=None)
        """
        return self._round(freq, RoundTo.MINUS_INFTY, ambiguous, nonexistent)

    def ceil(
        self,
        freq,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ) -> Self:
        # [FUNCTION DOC]
        # Rounds up (toward plus infinity) to a multiple of a fixed frequency.
        # Same parameters, errors and result as the round method, but with the
        # ceiling rounding mode.
        """
        Perform ceil operation on the data to the specified `freq`.

        Parameters
        ----------
        freq : str or Offset
            The frequency level to ceil the index to. Must be a fixed
            frequency like 's' (second) not 'ME' (month end). See
            :ref:`frequency aliases <timeseries.offset_aliases>` for
            a list of possible `freq` values.
        ambiguous : 'infer', bool-ndarray, 'NaT', default 'raise'
            Only relevant for DatetimeIndex:

            - 'infer' will attempt to infer fall dst-transition hours based on
              order
            - bool-ndarray where True signifies a DST time, False designates
              a non-DST time (note that this flag is only applicable for
              ambiguous times)
            - 'NaT' will return NaT where there are ambiguous times
            - 'raise' will raise a ValueError if there are ambiguous
              times.

        nonexistent : 'shift_forward', 'shift_backward', 'NaT', timedelta, \
            default 'raise'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            - 'shift_forward' will shift the nonexistent time forward to the
              closest existing time
            - 'shift_backward' will shift the nonexistent time backward to the
              closest existing time
            - 'NaT' will return NaT where there are nonexistent times
            - timedelta objects will shift nonexistent times by the timedelta
            - 'raise' will raise a ValueError if there are
              nonexistent times.

        Returns
        -------
        DatetimeIndex, TimedeltaIndex, or Series
            Index of the same type for a DatetimeIndex or TimedeltaIndex,
            or a Series with the same index for a Series.

        Raises
        ------
        ValueError if the `freq` cannot be converted.

        See Also
        --------
        DatetimeIndex.floor :
            Perform floor operation on the data to the specified `freq`.
        DatetimeIndex.snap :
            Snap time stamps to nearest occurring frequency.

        Notes
        -----
        If the timestamps have a timezone, ceiling will take place relative to the
        local ("wall") time and re-localized to the same timezone. When ceiling
        near daylight savings time, use ``nonexistent`` and ``ambiguous`` to
        control the re-localization behavior.

        Examples
        --------
        **DatetimeIndex**

        >>> rng = pd.date_range("1/1/2018 11:59:00", periods=3, freq="min")
        >>> rng
        DatetimeIndex(['2018-01-01 11:59:00', '2018-01-01 12:00:00',
                       '2018-01-01 12:01:00'],
                      dtype='datetime64[us]', freq='min')

        >>> rng.ceil('h')
        DatetimeIndex(['2018-01-01 12:00:00', '2018-01-01 12:00:00',
                       '2018-01-01 13:00:00'],
                      dtype='datetime64[us]', freq=None)

        **Series**

        >>> pd.Series(rng).dt.ceil("h")
        0   2018-01-01 12:00:00
        1   2018-01-01 12:00:00
        2   2018-01-01 13:00:00
        dtype: datetime64[us]

        When rounding near a daylight savings time transition, use ``ambiguous`` or
        ``nonexistent`` to control how the timestamp should be re-localized.

        >>> rng_tz = pd.DatetimeIndex(["2021-10-31 01:30:00"], tz="Europe/Amsterdam")

        >>> rng_tz.ceil("h", ambiguous=False)
        DatetimeIndex(['2021-10-31 02:00:00+01:00'],
                      dtype='datetime64[us, Europe/Amsterdam]', freq=None)

        >>> rng_tz.ceil("h", ambiguous=True)
        DatetimeIndex(['2021-10-31 02:00:00+02:00'],
                      dtype='datetime64[us, Europe/Amsterdam]', freq=None)
        """
        return self._round(freq, RoundTo.PLUS_INFTY, ambiguous, nonexistent)

    # --------------------------------------------------------------
    # Reductions

    def any(self, *, axis: AxisInt | None = None, skipna: bool = True) -> bool:
        # [FUNCTION DOC]
        # Reduction: true if any element is truthy (nonzero). Parameters
        # (keyword only): axis (default none); skipna (default true) uses the
        # missing mask to skip missing entries. Delegates to the nan-aware any
        # on the backing array, which raises a type error for non time-delta
        # dtypes. Returns a boolean.
        # GH#34479 the nanops call will raise a TypeError for non-td64 dtype
        return nanops.nanany(self._ndarray, axis=axis, skipna=skipna, mask=self.isna())

    def all(self, *, axis: AxisInt | None = None, skipna: bool = True) -> bool:
        # [FUNCTION DOC]
        # Reduction: true if all elements are truthy (nonzero). Same parameters
        # and error behavior as the any method, using the nan-aware all.
        # GH#34479 the nanops call will raise a TypeError for non-td64 dtype

        return nanops.nanall(self._ndarray, axis=axis, skipna=skipna, mask=self.isna())

    # --------------------------------------------------------------
    # Frequency Methods

    def _maybe_clear_freq(self) -> None:
        # [FUNCTION DOC]
        # HELPER: override of the mutation hook; unconditionally sets the stored
        # frequency to none.
        self._freq = None

    def _with_freq(self, freq) -> Self:
        # [FUNCTION DOC]
        # HELPER: returns a view of the same data carrying a different frequency.
        # Parameter freq is none (always allowed), the string "infer" (frequency
        # inferred from the values, possibly none), or an offset when the array
        # is empty (time delta arrays then require a fixed-duration or day
        # offset, else a type error). The frequency is not validated against
        # the data.
        # KNOWN ISSUE: the docstring says an offset is accepted, but for a
        # non-empty array an offset reaches an assertion that requires the
        # string "infer", so an offset raises an assertion error there.
        """
        Helper to get a view on the same data, with a new freq.

        Parameters
        ----------
        freq : DateOffset, None, or "infer"

        Returns
        -------
        Same type as self
        """
        # GH#29843
        if freq is None:
            # Always valid
            pass
        elif len(self) == 0 and isinstance(freq, BaseOffset):
            # Always valid.  In the TimedeltaArray case, we require a Tick offset
            if self.dtype.kind == "m" and not isinstance(freq, (Tick, Day)):
                raise TypeError("TimedeltaArray/Index freq must be a Tick")
        else:
            # As an internal method, we can ensure this assertion always holds
            assert freq == "infer"
            freq = to_offset(self.inferred_freq)

        arr = self.view()
        arr._freq = freq
        return arr

    # --------------------------------------------------------------
    # ExtensionArray Interface

    def _values_for_json(self) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: values handed to the JSON writer. For numpy-dtype data returns
        # the backing array directly (avoids a conversion); otherwise (aware
        # datetimes) defers to the parent. Returns a numpy array.
        # Small performance bump vs the base class which calls np.asarray(self)
        if isinstance(self.dtype, np.dtype):
            return self._ndarray
        return super()._values_for_json()

    def factorize(
        self,
        use_na_sentinel: bool = True,
        sort: bool = False,
    ):
        # [FUNCTION DOC]
        # Encodes values as integer codes plus unique values. Parameters:
        # use_na_sentinel (default true) encodes missing as the sentinel;
        # sort (default false) sorts the uniques. With a frequency set, values
        # are necessarily unique, so it short-circuits: codes are a simple range
        # (reversed, with reversed uniques, when sorting a decreasing frequency),
        # uniques are a copy (or a reversed slice, keeping the frequency).
        # Without a frequency, sort=true raises a not-implemented error
        # (callers are expected to use the top-level function), else defers to
        # the parent. Returns the pair (codes, uniques).
        if self.freq is not None:
            # We must be unique, so can short-circuit (and retain freq)
            if sort and self.freq.n < 0:
                codes = np.arange(len(self) - 1, -1, -1, dtype=np.intp)
                uniques = self[::-1]
            else:
                codes = np.arange(len(self), dtype=np.intp)
                uniques = self.copy()  # TODO: copy or view?
            return codes, uniques

        if sort:
            # algorithms.factorize only passes sort=True here when freq is
            #  not None, so this should not be reached.
            raise NotImplementedError(
                f"The 'sort' keyword in {type(self).__name__}.factorize is "
                "ignored unless arr.freq is not None. To factorize with sort, "
                "call pd.factorize(obj, sort=True) instead."
            )
        return super().factorize(use_na_sentinel=use_na_sentinel)

    @classmethod
    def _concat_same_type(
        cls,
        to_concat: Sequence[Self],
        axis: AxisInt = 0,
    ) -> Self:
        # [FUNCTION DOC]
        # Class method concatenating same-typed arrays. Parameters: to_concat is
        # a sequence of arrays (non-empty); axis (default 0). Delegates to the
        # parent. Along axis 0, the first array's frequency is kept on the
        # result if all non-empty inputs share it and each input starts exactly
        # one frequency step after the previous input ends. Returns a new array.
        new_obj = super()._concat_same_type(to_concat, axis)

        obj = to_concat[0]

        if axis == 0:
            # GH 3232: If the concat result is evenly spaced, we can retain the
            # original frequency
            to_concat = [x for x in to_concat if len(x)]

            if obj.freq is not None and all(x.freq == obj.freq for x in to_concat):
                pairs = zip(to_concat[:-1], to_concat[1:], strict=True)
                if all(pair[0][-1] + obj.freq == pair[1][0] for pair in pairs):
                    new_freq = obj.freq
                    new_obj._freq = new_freq
        return new_obj

    def copy(self, order: str = "C") -> Self:
        # [FUNCTION DOC]
        # Returns a copy. Parameter order is the memory layout code, default C
        # order. Delegates to the parent and carries over the frequency.
        new_obj = super().copy(order=order)
        new_obj._freq = self.freq
        return new_obj

    def interpolate(
        self,
        *,
        method: InterpolateOptions,
        axis: int,
        index: Index,
        limit,
        limit_direction,
        limit_area,
        copy: bool,
        **kwargs,
    ) -> Self:
        # [FUNCTION DOC]
        # Fills missing values by interpolation. Parameters (keyword only):
        # method must be the linear method (any other raises a not-implemented
        # error); axis is the axis to interpolate along; index is the index
        # providing coordinates; limit, limit_direction, limit_area restrict
        # how many/where gaps are filled; copy chooses between working on a copy
        # and in place; extra keyword arguments are forwarded. Uses the 2-D
        # in-place interpolation routine on the backing numpy array. With
        # copy false the array itself is modified and returned; with copy true a
        # new array of the same dtype is returned (no frequency carried).
        """
        See NDFrame.interpolate.__doc__.
        """
        # NB: we return type(self) even if copy=False
        if method != "linear":
            raise NotImplementedError

        if not copy:
            out_data = self._ndarray
        else:
            out_data = self._ndarray.copy()

        missing.interpolate_2d_inplace(
            out_data,
            method=method,
            axis=axis,
            index=index,
            limit=limit,
            limit_direction=limit_direction,
            limit_area=limit_area,
            **kwargs,
        )
        if not copy:
            return self
        return type(self)._simple_new(out_data, dtype=self.dtype)

    def take(
        self,
        indices: TakeIndexer,
        *,
        allow_fill: bool = False,
        fill_value: Any = None,
        axis: AxisInt = 0,
    ) -> Self:
        # [FUNCTION DOC]
        # Selects elements by position. Parameters: indices is a sequence of
        # integer positions; allow_fill (keyword, default false) treats negative
        # indices as missing when true; fill_value (default none) is the
        # replacement for filled slots; axis (default 0). Delegates to the
        # parent, then, if the indices form a slice-equivalent sequence, assigns
        # the frequency that the equivalent slice would have. Returns the result.
        result = super().take(
            indices=indices, allow_fill=allow_fill, fill_value=fill_value, axis=axis
        )

        indices = np.asarray(indices, dtype=np.intp)
        maybe_slice = lib.maybe_indices_to_slice(indices, len(self))  # type: ignore[arg-type]

        if isinstance(maybe_slice, slice):
            freq = self._get_getitem_freq(maybe_slice)
            result._freq = freq  # type: ignore[assignment]

        return result

    # --------------------------------------------------------------
    # Unsorted

    @property
    def _is_dates_only(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: property; true if every non-missing value is a whole multiple of
        # a day (midnight for datetimes, whole days for time deltas) and the
        # data has no timezone (aware data returns false). Used to choose
        # a compact display. Computed with integer arithmetic on the integer view
        # using the number of periods per day for the unit.
        """
        Check if we are round times at midnight (and no timezone), which will
        be given a more compact __repr__ than other cases. For TimedeltaArray
        we are checking for multiples of 24H.
        """
        if not lib.is_np_dtype(self.dtype):
            # i.e. we have a timezone
            return False

        values_int = self.asi8
        consider_values = values_int != iNaT
        reso = get_unit_from_dtype(self.dtype)
        ppd = periods_per_day(reso)

        # TODO: can we reuse is_date_array_normalized?  would need a skipna kwd
        #  (first attempt at this was less performant than this implementation)
        even_days = np.logical_and(consider_values, values_int % ppd != 0).sum() == 0
        return even_days


# -------------------------------------------------------------------
# Shared Constructor Helpers


def ensure_arraylike_for_datetimelike(
    data, copy: bool, cls_name: str
) -> tuple[ArrayLike, bool]:
    # [FUNCTION DOC]
    # Coerces constructor input into a numpy or extension array. Parameters: data
    # is any input; copy is the caller's copy flag; cls_name names the target
    # class for error messages. Inputs without a dtype (lists, tuples,
    # generators) become one-dimensional object arrays (copy flag cleared;
    # zero-dimensional non-list inputs such as generators are listed first).
    # Multi-level indexes raise a type error. Otherwise the underlying array is
    # extracted. Nullable integer arrays and integer arrow arrays become 64-bit
    # integers with the missing sentinel; other arrow arrays are converted to
    # numpy after datelike conversion; foreign array types are converted with
    # numpy; categoricals are expanded to their category values (keeping
    # timezone). Returns the array and the possibly updated copy flag.
    if not hasattr(data, "dtype"):
        # e.g. list, tuple
        if not isinstance(data, (list, tuple)) and np.ndim(data) == 0:
            # i.e. generator
            data = list(data)

        data = construct_1d_object_array_from_listlike(data)
        copy = False
    elif isinstance(data, ABCMultiIndex):
        raise TypeError(f"Cannot create a {cls_name} from a MultiIndex.")
    else:
        data = extract_array(data, extract_numpy=True)

    if isinstance(data, IntegerArray) or (
        isinstance(data, ArrowExtensionArray) and data.dtype.kind in "iu"
    ):
        data = data.to_numpy("int64", na_value=iNaT)
        copy = False
    elif isinstance(data, ArrowExtensionArray):
        data = data._maybe_convert_datelike_array()
        data = data.to_numpy()
        copy = False
    elif not isinstance(data, (np.ndarray, ExtensionArray)):
        # GH#24539 e.g. xarray, dask object
        data = np.asarray(data)

    elif isinstance(data, ABCCategorical):
        # GH#18664 preserve tz in going DTI->Categorical->DTI
        # TODO: cases where we need to do another pass through maybe_convert_dtype,
        #  e.g. the categories are timedelta64s
        data = data.categories.take(data.codes, fill_value=NaT)._values
        copy = False

    return data, copy


@overload
def validate_periods(periods: None) -> None:
    # [FUNCTION DOC]
    # Typing overload: none in, none out. See the implementation below.
    ...


@overload
def validate_periods(periods: int) -> int:
    # [FUNCTION DOC]
    # Typing overload: an integer in, an integer out. See the implementation
    # below.
    ...


def validate_periods(periods: int | None) -> int | None:
    # [FUNCTION DOC]
    # Validates the period count passed to range constructors. Parameter periods
    # is none or an integer (numpy integers count). Raises a type error if it is
    # anything else. Returns the input unchanged (despite the docstring saying
    # it casts to an integer, no cast is performed).
    # KNOWN ISSUE: the docstring says the value is cast to an integer, but the
    # function only validates and returns it as given.
    """
    If a `periods` argument is passed to the Datetime/Timedelta Array/Index
    constructor, cast it to an integer.

    Parameters
    ----------
    periods : None, int

    Returns
    -------
    periods : None or int

    Raises
    ------
    TypeError
        if periods is not None or int
    """
    if periods is not None and not lib.is_integer(periods):
        raise TypeError(f"periods must be an integer, got {periods}")
    # error: Incompatible return value type (got "int | integer[Any] | None",
    # expected "int | None")
    return periods  # type: ignore[return-value]


def _validate_inferred_freq(
    freq: BaseOffset | None, inferred_freq: BaseOffset | None
) -> BaseOffset | None:
    # [FUNCTION DOC]
    # HELPER: reconciles a user frequency with an inferred one. Parameters: freq
    # is the user's offset or none; inferred_freq is the inferred offset or none.
    # If an inferred frequency exists and the user gave a different one, raises
    # a value error saying the inferred frequency does not conform. If the user
    # gave none, the inferred one is returned; otherwise the user's frequency
    # is returned (also when there is no inferred one).
    """
    If the user passes a freq and another freq is inferred from passed data,
    require that they match.

    Parameters
    ----------
    freq : DateOffset or None
    inferred_freq : DateOffset or None

    Returns
    -------
    freq : DateOffset or None
    """
    if inferred_freq is not None:
        if freq is not None and freq != inferred_freq:
            raise ValueError(
                f"Inferred frequency {inferred_freq} from passed "
                "values does not conform to passed frequency "
                f"{freq.freqstr}"
            )
        if freq is None:
            freq = inferred_freq

    return freq


def dtype_to_unit(dtype: DatetimeTZDtype | np.dtype | ArrowDtype) -> str:
    # [FUNCTION DOC]
    # Returns the time-unit string of a datetime-like dtype. Parameter dtype is
    # a timezone-aware datetime dtype (its unit attribute is returned), an
    # arrow-backed dtype (must be of datetime or time-delta kind, otherwise a
    # value error saying it has no resolution; the pyarrow unit is returned), or
    # a numpy datetime/time-delta dtype (unit extracted from the dtype data).
    # Used by the unit property and constructors.
    """
    Return the unit str corresponding to the dtype's resolution.

    Parameters
    ----------
    dtype : DatetimeTZDtype or np.dtype
        If np.dtype, we assume it is a datetime64 dtype.

    Returns
    -------
    str
    """
    if isinstance(dtype, DatetimeTZDtype):
        return dtype.unit
    elif isinstance(dtype, ArrowDtype):
        if dtype.kind not in "mM":
            raise ValueError(f"{dtype=} does not have a resolution.")
        return dtype.pyarrow_dtype.unit
    return np.datetime_data(dtype)[0]
