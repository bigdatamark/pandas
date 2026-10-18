# [MODULE DOC]
# Defines the extension array that wraps a plain numpy array, exposed publicly
# as the numpy extension array in the public arrays namespace (the module name
# is rewritten by the module-setting decorator). It is mostly an internal
# compatibility layer: it lets numpy-backed data participate wherever an
# extension array is required (for example as the base of the string array
# classes, which reuse its ufunc, reduction and ops machinery).
# Relationships: builds on the ndarray-backed extension array mixin, the
# operator mixin and the object-string method mixin; delegates reductions to
# the nan-aware reduction module, filling to the missing-data module, and
# comparison/arithmetic to the ops module. Timedelta results are re-boxed as
# timedelta arrays.
# No import-time side effects besides applying the module-setting decorator.
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
    cast,
)

import numpy as np

from pandas._libs import lib
from pandas._libs.tslibs import is_supported_dtype
from pandas.compat.numpy import function as nv
from pandas.util._decorators import set_module

from pandas.core.dtypes.astype import (
    astype_array,
    astype_is_view,
)
from pandas.core.dtypes.cast import (
    construct_1d_object_array_from_listlike,
    maybe_downcast_to_dtype,
)
from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.missing import isna

from pandas.core import (
    arraylike,
    missing,
    nanops,
    ops,
)
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.strings.object_array import ObjectStringArrayMixin

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandas._typing import (
        ArrayLike,
        AxisInt,
        Dtype,
        FillnaOptions,
        InterpolateOptions,
        NpDtype,
        Scalar,
        TakeIndexer,
        npt,
    )

    from pandas import Index
    from pandas.arrays import StringArray


@set_module("pandas.arrays")
class NumpyExtensionArray(
    OpsMixin,
    NDArrayBackedExtensionArray,
    ObjectStringArrayMixin,
):
    # [CLASS DOC]
    # Thin extension array around a one-dimensional numpy array. State: the
    # wrapped array (kept in the backing-array attribute) and the dtype, a
    # numpy-extension dtype built from the array dtype. Inherits storage,
    # indexing, take/copy/etc. from the ndarray-backed mixin, operator dunder
    # plumbing from the ops mixin, and string accessor support from the
    # object-string mixin. Key methods: constructors, the numpy array and ufunc
    # protocols, cast/NA handling, fill and interpolate, take, reductions
    # (all delegated to the nan-aware reductions), to_numpy, unary operators and
    # the shared comparison/arithmetic method. Invariant: the backing array is
    # not zero-dimensional. The class marker type is distinct so the series
    # constructors do not place this array in an extension block. Missing value
    # is NaN by default for the internal fill value.
    # If you're wondering why pd.Series(cls) doesn't put the array in an
    # ExtensionBlock, search for `ABCNumpyExtensionArray`. We check for
    # that _typ to ensure that users don't unnecessarily use EAs inside
    # pandas internals, which turns off things like block consolidation.
    """
    A pandas ExtensionArray for NumPy data.

    This is mostly for internal compatibility, and is not especially
    useful on its own.

    Parameters
    ----------
    values : ndarray
        The NumPy ndarray to wrap. Must be 1-dimensional.
    copy : bool, default False
        Whether to copy `values`.

    Attributes
    ----------
    None

    Methods
    -------
    None

    See Also
    --------
    array : Create an array.
    Series.to_numpy : Convert a Series to a NumPy array.

    Examples
    --------
    >>> pd.arrays.NumpyExtensionArray(np.array([0, 1, 2, 3]))
    <NumpyExtensionArray>
    [0, 1, 2, 3]
    Length: 4, dtype: int64
    """

    _typ = "npy_extension"
    __array_priority__ = 1000
    _ndarray: np.ndarray
    _dtype: NumpyEADtype
    _internal_fill_value = np.nan

    # ------------------------------------------------------------------------
    # Constructors

    def __init__(
        self, values: np.ndarray | NumpyExtensionArray, copy: bool = False
    ) -> None:
        # [FUNCTION DOC]
        # Wraps a numpy array. values is a numpy array or another instance of
        # this class (its backing array is unwrapped); copy (default false)
        # copies the array first. Raises a value error when values is not a
        # numpy array, and when it is zero-dimensional.
        # KNOWN ISSUE: the class docstring and the error message say values
        # must be one-dimensional, but only zero-dimensional input is rejected;
        # arrays with two or more dimensions are accepted.
        # Sets the dtype from the array dtype and calls the
        # parent constructor.
        if isinstance(values, type(self)):
            values = values._ndarray
        if not isinstance(values, np.ndarray):
            raise ValueError(
                f"'values' must be a NumPy array, not {type(values).__name__}"
            )

        if values.ndim == 0:
            # Technically we support 2, but do not advertise that fact.
            raise ValueError("NumpyExtensionArray must be 1-dimensional.")

        if copy:
            values = values.copy()

        dtype = NumpyEADtype(values.dtype)
        super().__init__(values, dtype)

    @classmethod
    def _from_sequence(
        cls, scalars, *, dtype: Dtype | None = None, copy: bool = False
    ) -> NumpyExtensionArray:
        # [FUNCTION DOC]
        # HELPER: class-method constructor used by the generic array factory.
        # scalars is any sequence; dtype (keyword-only, default none) may be a
        # numpy-extension dtype (unwrapped to its numpy dtype) or anything the
        # numpy array constructor accepts; copy (keyword-only, default false)
        # copies only if the conversion returned the very same object.
        # Behavior: converts to a numpy array; if the result has more than one
        # dimension, the input has no dtype attribute and the requested dtype
        # is none or object (for example a list of tuples), a one-dimensional
        # object array holding each element is built instead.
        # Returns a new instance of the class.
        if isinstance(dtype, NumpyEADtype):
            dtype = dtype._dtype

        # error: Argument "dtype" to "asarray" has incompatible type
        # "Union[ExtensionDtype, str, dtype[Any], dtype[floating[_64Bit]], Type[object],
        # None]"; expected "Union[dtype[Any], None, type, _SupportsDType, str,
        # Union[Tuple[Any, int], Tuple[Any, Union[int, Sequence[int]]], List[Any],
        # _DTypeDict, Tuple[Any, Any]]]"
        result = np.asarray(scalars, dtype=dtype)  # type: ignore[arg-type]
        if (
            result.ndim > 1
            and not hasattr(scalars, "dtype")
            and (dtype is None or dtype == object)
        ):
            # e.g. list-of-tuples
            result = construct_1d_object_array_from_listlike(scalars)

        if copy and result is scalars:
            result = result.copy()
        return cls(result)

    def _cast_pointwise_result(self, values) -> ArrayLike:
        # [FUNCTION DOC]
        # HELPER: post-processes the output of an element-wise mapping
        # operation. Takes the sequence of results and first applies the parent
        # conversion. If both result and own kinds are integer-like, both
        # float, or both complex, the result is downcast to this array's numpy
        # dtype when that is lossless. If the result holds datetimes, it is
        # converted to a proper datetime extension array so later conversion
        # to object keeps timestamps. Otherwise returned as is. Returns the
        # array-like result.
        result = super()._cast_pointwise_result(values)
        lkind = self.dtype.kind
        rkind = result.dtype.kind
        if (
            (lkind in "iu" and rkind in "iu")
            or (lkind == "f" and rkind == "f")
            or (lkind == rkind == "c")
        ):
            result = maybe_downcast_to_dtype(result, self.dtype.numpy_dtype)
        elif rkind == "M":
            # Ensure potential subsequent .astype(object) doesn't incorrectly
            #  convert Timestamps to ints
            from pandas import array as pd_array

            result = pd_array(result, copy=False)
        return result

    # ------------------------------------------------------------------------
    # Data

    @property
    def dtype(self) -> NumpyEADtype:
        # [FUNCTION DOC]
        # Property returning the stored numpy-extension dtype wrapper.
        return self._dtype

    # ------------------------------------------------------------------------
    # NumPy Array Interface

    def __array__(
        self, dtype: np.dtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        # [FUNCTION DOC]
        # Numpy array protocol. dtype is an optional target numpy dtype;
        # copy is none (copy only if needed), true (always copy) or false.
        # Returns the backing array converted to the dtype. If the array is
        # read-only and no copy was requested and the conversion is a view, the
        # returned object is a view flagged non-writeable so the protection
        # carries over.
        if copy is not None:
            # Note: branch avoids `copy=None` for NumPy 1.x support
            result = np.array(self._ndarray, dtype=dtype, copy=copy)
        else:
            result = np.asarray(self._ndarray, dtype=dtype)

        if (
            self._readonly
            and not copy
            and (dtype is None or astype_is_view(self.dtype, dtype))
        ):
            result = result.view()
            result.flags.writeable = False

        return result

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        # [FUNCTION DOC]
        # Numpy universal-function protocol. ufunc is the function object,
        # method the call style name (call, reduce, at, and so on), inputs the
        # operands and kwargs the options (possibly including output arrays).
        # Order of handling: first try dispatching to the equivalent operator
        # dunder method; then a call with output arguments is dispatched
        # through the helper for outputs; a reduce call is tried through the
        # reduction dispatcher. If none applies, wrapped inputs and outputs are
        # unwrapped to plain arrays and the numpy function is invoked.
        # Result boxing: multiple outputs are each re-boxed in a tuple; the at
        # method returns nothing; reduce returns the raw scalar, or a boxed
        # array if the reduction produced an array; otherwise a single array
        # result is re-boxed (for string dtype arrays, boxed with the same
        # string dtype, and if that validation raises a value error the raw
        # numpy result is returned instead).
        # Scalars are never boxed because extension arrays are one-dimensional.
        # Lightly modified version of
        # https://numpy.org/doc/stable/reference/generated/numpy.lib.mixins.NDArrayOperatorsMixin.html
        # The primary modification is not boxing scalar return values
        # in NumpyExtensionArray, since pandas' ExtensionArrays are 1-d.
        out = kwargs.get("out", ())

        result = arraylike.maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )
        if result is not NotImplemented:
            return result

        if "out" in kwargs:
            # e.g. test_ufunc_unary
            return arraylike.dispatch_ufunc_with_out(
                self, ufunc, method, *inputs, **kwargs
            )

        if method == "reduce":
            result = arraylike.dispatch_reduction_ufunc(
                self, ufunc, method, *inputs, **kwargs
            )
            if result is not NotImplemented:
                # e.g. tests.series.test_ufunc.TestNumpyReductions
                return result

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(
            x._ndarray if isinstance(x, NumpyExtensionArray) else x for x in inputs
        )
        if out:
            kwargs["out"] = tuple(
                x._ndarray if isinstance(x, NumpyExtensionArray) else x for x in out
            )
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if ufunc.nout > 1:
            # multiple return values; re-box array-like results
            return tuple(type(self)(x) for x in result)
        elif method == "at":
            # no return value
            return None
        elif method == "reduce":
            if isinstance(result, np.ndarray):
                # e.g. test_np_reduce_2d
                return type(self)(result)

            # e.g. test_np_max_nested_tuples
            return result
        else:
            if self.dtype.type is str:  # type: ignore[comparison-overlap]
                # StringDtype
                self = cast("StringArray", self)
                try:
                    # specify dtype to preserve storage/na_value
                    return type(self)(result, dtype=self.dtype)
                except ValueError:
                    # if validation of input fails (no strings)
                    # -> fallback to returning raw numpy array
                    return result
            # one return value; re-box array-like results
            return type(self)(result)

    # ------------------------------------------------------------------------
    # Pandas ExtensionArray Interface

    def astype(self, dtype, copy: bool = True):
        # [FUNCTION DOC]
        # Casts to the given dtype (anything the dtype resolver accepts);
        # copy (default true) controls copying. If the resolved dtype equals
        # the current one, returns a copy or the same object depending on
        # copy; otherwise delegates to the generic array cast on the backing
        # array and returns its result (which may be a numpy array or an
        # extension array, not necessarily this class).
        dtype = pandas_dtype(dtype)

        if dtype == self.dtype:
            if copy:
                return self.copy()
            return self

        result = astype_array(self._ndarray, dtype=dtype, copy=copy)
        return result

    def isna(self) -> np.ndarray:
        # [FUNCTION DOC]
        # Returns a boolean numpy array, true where the backing element is
        # missing according to the generic missing-value check (NaN, none,
        # NaT and so on).
        return isna(self._ndarray)

    def _validate_scalar(self, fill_value):
        # [FUNCTION DOC]
        # HELPER: validation hook for fill values (used by take and fill
        # operations). A none fill value is replaced with the dtype's NA
        # value (mostly for subclasses); anything else is returned unchanged,
        # no actual validation is performed.
        if fill_value is None:
            # Primarily for subclasses
            fill_value = self.dtype.na_value
        return fill_value

    def _values_for_factorize(self) -> tuple[np.ndarray, float | None]:
        # [FUNCTION DOC]
        # HELPER: hook used by factorize. Returns a pair of the backing array
        # and the value to treat as missing: none for integer, unsigned and
        # boolean kinds (no missing possible), otherwise NaN.
        if self.dtype.kind in "iub":
            fv = None
        else:
            fv = np.nan
        return self._ndarray, fv

    # Base EA class (and all other EA classes) don't have limit_area keyword
    # This can be removed here as well when the interpolate ffill/bfill method
    # deprecation is enforced
    def _pad_or_backfill(
        self,
        *,
        method: FillnaOptions,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        copy: bool = True,
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: forward/backward fill hook. Keyword-only: method is the fill
        # direction name; limit (default none) caps consecutive fills;
        # limit_area (default none) restricts to inside or outside the valid
        # range; copy (default true) fills a copy instead of in place.
        # Operates on the transpose of the backing array along axis zero via
        # the missing-data module in place. Returns a new array of the same
        # class when copy is true, otherwise returns self (mutated).
        # Side effect: when copy is false the backing array is modified.
        """
        ffill or bfill along axis=0.
        """
        if copy:
            out_data = self._ndarray.copy()
        else:
            out_data = self._ndarray

        meth = missing.clean_fill_method(method)
        missing.pad_or_backfill_inplace(
            out_data.T,
            method=meth,
            axis=0,
            limit=limit,
            limit_area=limit_area,
        )

        if not copy:
            return self
        return type(self)._simple_new(out_data, dtype=self.dtype)

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
        # Fills missing values by interpolation. Keyword-only inputs: method
        # (interpolation algorithm name), axis, index (the index labels used by
        # index-based methods), limit, limit_direction, limit_area (fill
        # restrictions), copy (work on a copy, else in place), and extra
        # keyword arguments forwarded to the algorithm.
        # Raises a type error for non-numeric dtypes. The work is done in place
        # by the missing-data module on either a copy or the backing array.
        # Returns a new array of this class if copy is true, otherwise self
        # (mutated).
        # KNOWN ISSUE: a source comment notes there is no assertion that the
        # dtype is floating, so integer data would be interpolated in place
        # without a check.
        """
        See NDFrame.interpolate.__doc__.
        """
        # NB: we return type(self) even if copy=False
        if not self.dtype._is_numeric:
            raise TypeError(f"Cannot interpolate with {self.dtype} dtype")

        if not copy:
            out_data = self._ndarray
        else:
            out_data = self._ndarray.copy()

        # TODO: assert we have floating dtype?
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
        # Takes entries by position. indices is an integer sequence;
        # keyword-only allow_fill (default false) treats negative indices as
        # missing markers, fill_value (default none) is used for those, and
        # axis (default zero). Delegates to the parent implementation, then,
        # for integer, unsigned or boolean kinds, re-wraps the result array
        # without copying (works around a known issue with the result type).
        # Returns the new array. Errors for out-of-bounds indices come from
        # the parent.
        """
        Take entries from this array at each index in a list of indices,
        producing an array containing only those entries.
        """
        result = super().take(
            indices, allow_fill=allow_fill, fill_value=fill_value, axis=axis
        )
        # See GH#62448.
        if self.dtype.kind in "iub":
            return type(self)(result._ndarray, copy=False)

        return result

    # ------------------------------------------------------------------------
    # Reductions

    def any(
        self,
        *,
        axis: AxisInt | None = None,
        out=None,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        # [FUNCTION DOC]
        # Reduction: true if any element is truthy. Keyword-only: axis
        # (default none, reduce everything), out and keepdims (accepted only
        # for numpy compatibility; validated and must be defaults), skipna
        # (default true, ignore missing). Raises a value error through the
        # compatibility validator for non-default out or keepdims. Returns the
        # reduction result, wrapped via the parent helper for the axis.
        nv.validate_any((), {"out": out, "keepdims": keepdims})
        result = nanops.nanany(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def all(
        self,
        *,
        axis: AxisInt | None = None,
        out=None,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        # [FUNCTION DOC]
        # Reduction: true if all elements are truthy. Same parameters and
        # behavior as the any reduction (out and keepdims only validated,
        # skipna default true, axis default none), using the nan-aware all.
        nv.validate_all((), {"out": out, "keepdims": keepdims})
        result = nanops.nanall(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def min(
        self, *, axis: AxisInt | None = None, skipna: bool = True, **kwargs
    ) -> Scalar:
        # [FUNCTION DOC]
        # Reduction: minimum. Keyword-only axis (default none) and skipna
        # (default true); extra keyword arguments are only validated for numpy
        # compatibility. Uses the own missing mask. Returns the wrapped result.
        nv.validate_min((), kwargs)
        result = nanops.nanmin(
            values=self._ndarray, axis=axis, mask=self.isna(), skipna=skipna
        )
        return self._wrap_reduction_result(axis, result)

    def max(
        self, *, axis: AxisInt | None = None, skipna: bool = True, **kwargs
    ) -> Scalar:
        # [FUNCTION DOC]
        # Reduction: maximum. Same parameters and behavior as the minimum
        # reduction.
        nv.validate_max((), kwargs)
        result = nanops.nanmax(
            values=self._ndarray, axis=axis, mask=self.isna(), skipna=skipna
        )
        return self._wrap_reduction_result(axis, result)

    def sum(
        self,
        *,
        axis: AxisInt | None = None,
        skipna: bool = True,
        min_count: int = 0,
        **kwargs,
    ) -> Scalar:
        # [FUNCTION DOC]
        # Reduction: sum. Keyword-only axis (default none), skipna (default
        # true), min_count (default zero; minimum number of valid values
        # needed, otherwise the result is missing); extra keywords only
        # validated. Returns the wrapped result.
        nv.validate_sum((), kwargs)
        result = nanops.nansum(
            self._ndarray, axis=axis, skipna=skipna, min_count=min_count
        )
        return self._wrap_reduction_result(axis, result)

    def prod(
        self,
        *,
        axis: AxisInt | None = None,
        skipna: bool = True,
        min_count: int = 0,
        **kwargs,
    ) -> Scalar:
        # [FUNCTION DOC]
        # Reduction: product. Same parameters and behavior as the sum
        # reduction (axis, skipna, min_count).
        nv.validate_prod((), kwargs)
        result = nanops.nanprod(
            self._ndarray, axis=axis, skipna=skipna, min_count=min_count
        )
        return self._wrap_reduction_result(axis, result)

    def mean(
        self,
        *,
        axis: AxisInt | None = None,
        dtype: NpDtype | None = None,
        out=None,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        # [FUNCTION DOC]
        # Reduction: arithmetic mean. Keyword-only axis (default none), skipna
        # (default true); dtype, out and keepdims exist only for numpy
        # compatibility and are validated (non-default values rejected), not
        # used. Returns the wrapped result.
        nv.validate_mean((), {"dtype": dtype, "out": out, "keepdims": keepdims})
        result = nanops.nanmean(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def median(
        self,
        *,
        axis: AxisInt | None = None,
        out=None,
        overwrite_input: bool = False,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        # [FUNCTION DOC]
        # Reduction: median. Keyword-only axis (default none), skipna (default
        # true); out, overwrite_input and keepdims are only validated for
        # numpy compatibility. Returns the wrapped result.
        nv.validate_median(
            (), {"out": out, "overwrite_input": overwrite_input, "keepdims": keepdims}
        )
        result = nanops.nanmedian(self._ndarray, axis=axis, skipna=skipna)
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
        # Reduction: standard deviation. Keyword-only axis (default none),
        # ddof (delta degrees of freedom, default one), skipna (default true);
        # dtype, out and keepdims are validated only. Returns the wrapped
        # result.
        nv.validate_stat_ddof_func(
            (), {"dtype": dtype, "out": out, "keepdims": keepdims}, fname="std"
        )
        result = nanops.nanstd(self._ndarray, axis=axis, skipna=skipna, ddof=ddof)
        return self._wrap_reduction_result(axis, result)

    def var(
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
        # Reduction: variance. Same parameters and behavior as the standard
        # deviation reduction (ddof default one).
        nv.validate_stat_ddof_func(
            (), {"dtype": dtype, "out": out, "keepdims": keepdims}, fname="var"
        )
        result = nanops.nanvar(self._ndarray, axis=axis, skipna=skipna, ddof=ddof)
        return self._wrap_reduction_result(axis, result)

    def sem(
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
        # Reduction: standard error of the mean. Same parameters and behavior
        # as the standard deviation reduction (ddof default one).
        nv.validate_stat_ddof_func(
            (), {"dtype": dtype, "out": out, "keepdims": keepdims}, fname="sem"
        )
        result = nanops.nansem(self._ndarray, axis=axis, skipna=skipna, ddof=ddof)
        return self._wrap_reduction_result(axis, result)

    def kurt(
        self,
        *,
        axis: AxisInt | None = None,
        dtype: NpDtype | None = None,
        out=None,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        # [FUNCTION DOC]
        # Reduction: excess kurtosis. Keyword-only axis (default none) and
        # skipna (default true); dtype, out and keepdims are validated only.
        # Returns the wrapped result.
        nv.validate_stat_ddof_func(
            (), {"dtype": dtype, "out": out, "keepdims": keepdims}, fname="kurt"
        )
        result = nanops.nankurt(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def skew(
        self,
        *,
        axis: AxisInt | None = None,
        dtype: NpDtype | None = None,
        out=None,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        # [FUNCTION DOC]
        # Reduction: skewness. Same parameters and behavior as the kurtosis
        # reduction.
        nv.validate_stat_ddof_func(
            (), {"dtype": dtype, "out": out, "keepdims": keepdims}, fname="skew"
        )
        result = nanops.nanskew(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    # ------------------------------------------------------------------------
    # Additional Methods

    def to_numpy(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        na_value: object = lib.no_default,
    ) -> np.ndarray:
        # [FUNCTION DOC]
        # Converts to a numpy array. dtype is the optional target dtype; copy
        # (default false) forces a copy; na_value (default: not specified)
        # replaces missing entries.
        # Behavior: if na_value is given and there are missing entries, a copy
        # of the data with those entries replaced is made. Otherwise the
        # backing array is used, and for read-only arrays without copy a
        # non-writeable view is returned. The result is converted to dtype; if
        # copy was requested and the result is still the backing array, it is
        # copied. Returns the numpy array.
        mask = self.isna()
        if na_value is not lib.no_default and mask.any():
            result = self._ndarray.copy()
            result[mask] = na_value
        else:
            result = self._ndarray
            if not copy and self._readonly:
                result = result.view()
                result.flags.writeable = False

        result = np.asarray(result, dtype=dtype)

        if copy and result is self._ndarray:
            result = result.copy()

        return result

    # ------------------------------------------------------------------------
    # Ops

    def __invert__(self) -> NumpyExtensionArray:
        # [FUNCTION DOC]
        # Unary bitwise/logical inversion; returns a new array of this class
        # holding the element-wise inversion of the backing array.
        return type(self)(~self._ndarray)

    def __neg__(self) -> NumpyExtensionArray:
        # [FUNCTION DOC]
        # Unary negation; returns a new array of this class with every element
        # negated.
        return type(self)(-self._ndarray)

    def __pos__(self) -> NumpyExtensionArray:
        # [FUNCTION DOC]
        # Unary plus; returns a new array of this class with the unary plus
        # applied to the backing array.
        return type(self)(+self._ndarray)

    def __abs__(self) -> NumpyExtensionArray:
        # [FUNCTION DOC]
        # Absolute value; returns a new array of this class with every element
        # replaced by its absolute value.
        return type(self)(abs(self._ndarray))

    def _cmp_method(self, other, op):
        # [FUNCTION DOC]
        # HELPER: shared implementation behind comparison and arithmetic
        # operators (also bound as the arithmetic method, so both operator
        # families use identical logic). other is a scalar, array-like or
        # another instance (unwrapped); op is the operator function.
        # The operand is prepared for the operation, datetime-like operands are
        # wrapped, and the array-op for the operator is applied to the backing
        # array and operand. For the paired quotient-and-remainder operators
        # (normal and reflected) the two results are each wrapped. A numpy
        # result is wrapped via the wrapping helper (which makes timedelta
        # arrays where appropriate); results that already are extension
        # arrays are returned as is. Returns the wrapped result or tuple.
        if isinstance(other, NumpyExtensionArray):
            other = other._ndarray

        other = ops.maybe_prepare_scalar_for_op(other, (len(self),))
        pd_op = ops.get_array_op(op)
        other = ensure_wrapped_if_datetimelike(other)
        result = pd_op(self._ndarray, other)

        if op is divmod or op is ops.rdivmod:
            a, b = result
            if isinstance(a, np.ndarray):
                # for e.g. op vs TimedeltaArray, we may already
                #  have an ExtensionArray, in which case we do not wrap
                return self._wrap_ndarray_result(a), self._wrap_ndarray_result(b)
            return a, b

        if isinstance(result, np.ndarray):
            # for e.g. multiplication vs TimedeltaArray, we may already
            #  have an ExtensionArray, in which case we do not wrap
            return self._wrap_ndarray_result(result)
        return result

    _arith_method = _cmp_method

    def _wrap_ndarray_result(self, result: np.ndarray):
        # [FUNCTION DOC]
        # HELPER: boxes the numpy result of an operator. If the result has a
        # timedelta kind with a supported resolution, returns a timedelta array
        # (built without validation, imported lazily); otherwise a new array
        # of this class.
        # If we have timedelta64[ns] result, return a TimedeltaArray instead
        #  of a NumpyExtensionArray
        if result.dtype.kind == "m" and is_supported_dtype(result.dtype):
            from pandas.core.arrays import TimedeltaArray

            return TimedeltaArray._simple_new(result, dtype=result.dtype)
        return type(self)(result)

    def _formatter(self, boxed: bool = False) -> Callable[[Any], str | None]:
        # [FUNCTION DOC]
        # HELPER: returns the per-element formatting function used when
        # printing. boxed (default false) is accepted but not used. For
        # bytes/unicode string kinds, returns a function that wraps the value in
        # single quotes (to follow the numpy repr change); for object dtype the
        # repr function; otherwise the plain string conversion.
        # NEP 51: https://github.com/numpy/numpy/pull/22449
        if self.dtype.kind in "SU":
            return "'{}'".format
        elif self.dtype == "object":
            return repr
        else:
            return str
