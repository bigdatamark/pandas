"""
Provide classes to perform the groupby aggregate operations.

These are not exposed to the user and provide implementations of the grouping
operations, primarily in cython. These classes (BaseGrouper and BinGrouper)
are contained *in* the SeriesGroupBy and DataFrameGroupBy objects.
"""

# [MODULE DOC]
# Internal implementation of groupby computation. Contents: two small result
# helpers (validating that a user function aggregated, and unwrapping a scalar
# from array-like results); a dispatcher class for the compiled groupby
# routines (choosing function, input casting, output shape and dtype,
# post-processing); the base grouper (holds groupings, codes, result index,
# group sizes, iteration, pure-python aggregation and per-group apply); a bin
# grouper for contiguous bins (resampling); and splitter classes that cut a
# group-sorted series or frame into per-group objects.
# Relationships: built by grouper.py (get_grouper) and held by SeriesGroupBy
# and DataFrameGroupBy; depends on the compiled groupby library, sorting
# utilities and extension array group operations. No import-time side effects.
from __future__ import annotations

import collections
import functools
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    final,
)

import numpy as np

from pandas._libs import (
    NaT,
    lib,
)
import pandas._libs.groupby as libgroupby
from pandas._typing import (
    ArrayLike,
    AxisInt,
    NDFrameT,
    Shape,
    npt,
)
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly

from pandas.core.dtypes.cast import (
    maybe_downcast_to_dtype,
)
from pandas.core.dtypes.common import (
    ensure_float64,
    ensure_int64,
    ensure_platform_int,
    ensure_uint64,
    is_1d_only_ea_dtype,
)
from pandas.core.dtypes.missing import (
    isna,
    maybe_fill,
)

from pandas.core.arrays import Categorical
from pandas.core.frame import DataFrame
from pandas.core.groupby import grouper
from pandas.core.indexes.api import (
    CategoricalIndex,
    Index,
    MultiIndex,
    ensure_index,
)
from pandas.core.series import Series
from pandas.core.sorting import (
    compress_group_index,
    decons_obs_group_ids,
    get_group_index,
    get_group_index_sorter,
    get_indexer_dict,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Generator,
        Hashable,
        Iterator,
    )

    from pandas.core.generic import NDFrame


def check_result_array(obj, dtype) -> None:
    # [FUNCTION DOC]
    # HELPER: sanity check for user aggregation functions. obj is the value a
    # function returned for one group, dtype the dtype of the group's data.
    # Raises a value error (must produce aggregated value) when obj is a numpy
    # array and dtype is not object; object data may legitimately aggregate to
    # arrays. Returns nothing.
    # Our operation is supposed to be an aggregation/reduction. If
    #  it returns an ndarray, this likely means an invalid operation has
    #  been passed. See test_apply_without_aggregation, test_agg_must_agg
    if isinstance(obj, np.ndarray):
        if dtype != object:
            # If it is object dtype, the function can be a reduction/aggregation
            #  and still return an ndarray e.g. test_agg_over_numpy_arrays
            raise ValueError("Must produce aggregated value")


def extract_result(res):
    # [FUNCTION DOC]
    # HELPER: unwraps a user function result. If res exposes underlying values
    # (series, index, extension array), takes those (keeping extension types);
    # when the values are one-dimensional with exactly one element, returns that
    # single element. Anything without such values is returned unchanged.
    """
    Extract the result object, it might be a 0-dim ndarray
    or a len-1 0-dim, or a scalar
    """
    if hasattr(res, "_values"):
        # Preserve EA
        res = res._values
        if res.ndim == 1 and len(res) == 1:
            # see test_agg_lambda_with_timezone, test_resampler_grouper.py::test_apply
            res = res[0]
    return res


class WrappedCythonOp:
    # [CLASS DOC]
    # Dispatcher around the compiled groupby routines. State: kind (aggregate
    # or transform), how (operation name such as mean) and has_dropped_na (true
    # exactly when null group keys were dropped). Class data: a blocklist of
    # operations whose compiled result is never cast back to the input dtype,
    # a table mapping kind and operation name to a routine name or partial
    # function, and an arity table (only open-high-low-close has more than
    # one output column). Key methods: choose the routine for a dtype, cast
    # inputs, work out output shape and dtypes, and run the routine with
    # pre- and post-processing (the public entry point is the cython
    # operation method).
    """
    Dispatch logic for functions defined in _libs.groupby

    Parameters
    ----------
    kind: str
        Whether the operation is an aggregate or transform.
    how: str
        Operation name, e.g. "mean".
    has_dropped_na: bool
        True precisely when dropna=True and the grouper contains a null value.
    """

    # Functions for which we do _not_ attempt to cast the cython result
    #  back to the original dtype.
    cast_blocklist = frozenset(
        ["any", "all", "rank", "count", "size", "idxmin", "idxmax"]
    )

    def __init__(self, kind: str, how: str, has_dropped_na: bool) -> None:
        # [FUNCTION DOC]
        # Stores kind (aggregate or transform), how (operation name) and
        # has_dropped_na. No validation; unknown names fail later at lookup.
        self.kind = kind
        self.how = how
        self.has_dropped_na = has_dropped_na

    _CYTHON_FUNCTIONS: dict[str, dict] = {
        "aggregate": {
            "any": functools.partial(libgroupby.group_any_all, val_test="any"),
            "all": functools.partial(libgroupby.group_any_all, val_test="all"),
            "sum": "group_sum",
            "prod": "group_prod",
            "idxmin": functools.partial(libgroupby.group_idxmin_idxmax, name="idxmin"),
            "idxmax": functools.partial(libgroupby.group_idxmin_idxmax, name="idxmax"),
            "min": "group_min",
            "max": "group_max",
            "mean": "group_mean",
            "median": "group_median_float64",
            "var": "group_var",
            "std": functools.partial(libgroupby.group_var, name="std"),
            "sem": functools.partial(libgroupby.group_var, name="sem"),
            "skew": "group_skew",
            "kurt": "group_kurt",
            "first": "group_nth",
            "last": "group_last",
            "ohlc": "group_ohlc",
        },
        "transform": {
            "cumprod": "group_cumprod",
            "cumsum": "group_cumsum",
            "cummin": "group_cummin",
            "cummax": "group_cummax",
            "rank": "group_rank",
        },
    }

    _cython_arity = {"ohlc": 4}  # OHLC

    @classmethod
    def get_kind_from_how(cls, how: str) -> str:
        # [FUNCTION DOC]
        # Class method. Returns the string aggregate if the operation name is
        # in the aggregate table, otherwise the string transform (without
        # checking that the name is a known transform).
        if how in cls._CYTHON_FUNCTIONS["aggregate"]:
            return "aggregate"
        return "transform"

    # Note: we make this a classmethod and pass kind+how so that caching
    #  works at the class level and not the instance level
    @classmethod
    @functools.cache
    def _get_cython_function(
        cls, kind: str, how: str, dtype: np.dtype, is_numeric: bool
    ):
        # [FUNCTION DOC]
        # HELPER: class-level cached lookup (all arguments hashable) of the
        # compiled routine for kind, how, input dtype and whether the dtype
        # is numeric. Table entries given as names are resolved from the compiled
        # groupby library; partial functions are used directly. Numeric input
        # returns the routine. Object dtype: raises a not-implemented error for
        # median and cumulative product, and for routines lacking an object
        # signature (except skew/kurt which are converted later, and the
        # variance-family and index-of-extreme partials which are returned
        # directly). Any other non-numeric dtype raises a not-implemented error
        # that asks the user to report a bug.
        dtype_str = dtype.name
        ftype = cls._CYTHON_FUNCTIONS[kind][how]

        # see if there is a fused-type version of function
        # only valid for numeric
        if callable(ftype):
            f = ftype
        else:
            f = getattr(libgroupby, ftype)
        if is_numeric:
            return f
        elif dtype == np.dtype(object):
            if how in ["median", "cumprod"]:
                # no fused types -> no __signatures__
                raise NotImplementedError(
                    f"function is not implemented for this dtype: "
                    f"[how->{how},dtype->{dtype_str}]"
                )
            elif how in ["std", "sem", "idxmin", "idxmax"]:
                # We have a partial object that does not have __signatures__
                return f
            elif how in ["skew", "kurt"]:
                # _get_cython_vals will convert to float64
                pass
            elif "object" not in f.__signatures__:
                # raise NotImplementedError here rather than TypeError later
                raise NotImplementedError(
                    f"function is not implemented for this dtype: "
                    f"[how->{how},dtype->{dtype_str}]"
                )
            return f
        else:
            raise NotImplementedError(
                "This should not be reached. Please report a bug at "
                "github.com/pandas-dev/pandas/",
                dtype,
            )

    def _get_cython_vals(self, values: np.ndarray) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: casts input to what the compiled routine supports. For
        # median, std, sem, skew and kurt returns float64 values. For integer
        # data: variance and mean (and any transform when null keys were
        # dropped) get float64; sum, open-high-low-close, product and
        # cumulative sum/product are widened to 64-bit signed or unsigned to
        # avoid overflow. Otherwise returns values unchanged.
        """
        Cast numeric dtypes to float64 for functions that only support that.

        Parameters
        ----------
        values : np.ndarray

        Returns
        -------
        values : np.ndarray
        """
        how = self.how

        if how in ["median", "std", "sem", "skew", "kurt"]:
            # median only has a float64 implementation
            # We should only get here with is_numeric, as non-numeric cases
            #  should raise in _get_cython_function
            values = ensure_float64(values)

        elif values.dtype.kind in "iu":
            if how in ["var", "mean"] or (
                self.kind == "transform" and self.has_dropped_na
            ):
                # has_dropped_na check need for test_null_group_str_transformer
                # result may still include NaN, so we have to cast
                values = ensure_float64(values)

            elif how in ["sum", "ohlc", "prod", "cumsum", "cumprod"]:
                # Avoid overflow during group op
                if values.dtype.kind == "i":
                    values = ensure_int64(values)
                else:
                    values = ensure_uint64(values)

        return values

    def _get_output_shape(self, ngroups: int, values: np.ndarray) -> Shape:
        # [FUNCTION DOC]
        # HELPER: shape of the output buffer. Open-high-low-close yields
        # ngroups rows by its arity (four) columns. An operation with arity
        # above one otherwise raises a not-implemented error. Transforms keep
        # the input shape; aggregations give ngroups followed by all but the
        # first input dimension.
        how = self.how
        kind = self.kind

        arity = self._cython_arity.get(how, 1)

        out_shape: Shape
        if how == "ohlc":
            out_shape = (ngroups, arity)
        elif arity > 1:
            raise NotImplementedError(
                "arity of more than 1 is not supported for the 'how' argument"
            )
        elif kind == "transform":
            out_shape = values.shape
        else:
            out_shape = (ngroups, *values.shape[1:])
        return out_shape

    def _get_out_dtype(self, dtype: np.dtype) -> np.dtype:
        # [FUNCTION DOC]
        # HELPER: dtype of the output buffer for the compiled routine given the
        # (already cast) input dtype. Rank gives 64-bit float; index-of-minimum
        # and maximum give the platform integer (positions); integer, unsigned,
        # float, complex and bool kinds keep kind and item size; everything
        # else gives object.
        how = self.how

        if how == "rank":
            out_dtype = "float64"
        elif how in ["idxmin", "idxmax"]:
            # The Cython implementation only produces the row number; we'll take
            # from the index using this in post processing
            out_dtype = "intp"
        elif dtype.kind in "iufcb":
            out_dtype = f"{dtype.kind}{dtype.itemsize}"
        else:
            out_dtype = "object"
        return np.dtype(out_dtype)

    def _get_result_dtype(self, dtype: np.dtype) -> np.dtype:
        # [FUNCTION DOC]
        # HELPER: dtype the final result should have given the original input
        # dtype. Bool input becomes 64-bit integer for sum, cumulative sum,
        # product and cumulative product. For mean, median, variance, standard
        # deviation and standard error, float and complex keep their dtype
        # while integer, unsigned and bool become 64-bit float. All others
        # return the input dtype.
        # KNOWN ISSUE: the sum operation name appears twice in the first
        # membership list, which is redundant.
        """
        Get the desired dtype of a result based on the
        input dtype and how it was computed.

        Parameters
        ----------
        dtype : np.dtype

        Returns
        -------
        np.dtype
            The desired dtype of the result.
        """
        how = self.how

        if how in ["sum", "cumsum", "sum", "prod", "cumprod"]:
            if dtype == np.dtype(bool):
                return np.dtype(np.int64)
        elif how in ["mean", "median", "var", "std", "sem"]:
            if dtype.kind in "fc":
                return dtype
            elif dtype.kind in "iub":
                return np.dtype(np.float64)
        return dtype

    @final
    def _cython_op_ndim_compat(
        self,
        values: np.ndarray,
        *,
        min_count: int,
        ngroups: int,
        comp_ids: np.ndarray,
        mask: npt.NDArray[np.bool_] | None = None,
        result_mask: npt.NDArray[np.bool_] | None = None,
        initial: Any = 0,
        **kwargs,
    ) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: lets the two-dimensional-only compiled call handle 1-D data.
        # Keyword-only inputs: values (1-D or 2-D), min_count (minimum valid
        # entries per group), ngroups, comp_ids (group id per entry), mask and
        # result_mask (optional boolean masks for nullable data), initial
        # (starting value, default zero, used only by sum), plus extra
        # keywords forwarded to the call. A 1-D input (and masks) is
        # expanded to one row, dispatched, and returned squeezed to 1-D, or
        # transposed when there are several output columns (open-high-low-
        # close). 2-D input is passed straight through. Returns the array.

            # expand to 2d, dispatch, then squeeze if appropriate
            values2d = values[None, :]
            if mask is not None:
                mask = mask[None, :]
            if result_mask is not None:
                result_mask = result_mask[None, :]
            res = self._call_cython_op(
                values2d,
                min_count=min_count,
                ngroups=ngroups,
                comp_ids=comp_ids,
                mask=mask,
                result_mask=result_mask,
                initial=initial,
                **kwargs,
            )
            if res.shape[0] == 1:
                return res[0]

            # otherwise we have OHLC
            return res.T

        return self._call_cython_op(
            values,
            min_count=min_count,
            ngroups=ngroups,
            comp_ids=comp_ids,
            mask=mask,
            result_mask=result_mask,
            initial=initial,
            **kwargs,
        )

    @final
    def _call_cython_op(
        self,
        values: np.ndarray,  # np.ndarray[ndim=2]
        *,
        min_count: int,
        ngroups: int,
        comp_ids: np.ndarray,
        mask: npt.NDArray[np.bool_] | None,
        result_mask: npt.NDArray[np.bool_] | None,
        initial: Any = 0,
        **kwargs,
    ) -> np.ndarray:  # np.ndarray[ndim=2]
        # [FUNCTION DOC]
        # HELPER: runs the compiled routine on 2-D data (variables by rows).
        # Inputs as for the dimension-compat wrapper (values, min_count,
        # ngroups, comp_ids, masks, initial, extra keywords; the keywords may
        # include a skipna flag required for any/all on object data).
        # Steps: datetime-like data is viewed as 64-bit integers (and treated
        # as numeric), bool is viewed as small unsigned integers, half floats
        # widen to single. For any/all a null mask is computed if absent, object
        # data with skipna has masked entries set true, and values become int8
        # truth flags. Data is transposed, the routine for the dtype is selected,
        # inputs cast, output buffer allocated, and the routine called with an
        # argument set depending on the operation: aggregation families differ
        # in whether they take counts, an initial value (sum only), or the
        # datetime-like flag; any/all result is converted to bool; skew/kurt on
        # object input gives object output; an unknown aggregation raises a
        # not-implemented error. Transforms pass the group count and the
        # datetime-like flag (and the result mask for all but rank).
        # After the call, for aggregations other than index-of-extreme with
        # integer output (non datetime-like), groups with too few entries
        # (below the larger of min_count and zero for sum/product or one) are
        # filled with missing values by widening to float, unless a result mask is
        # supplied (then asserted already masked). The result is transposed back
        # and, except for blocklisted operations, cast toward the dtype implied by
        # the original input (restoring datetime types, etc.).
        # KNOWN ISSUE: unresolved TODOs note min_count is ignored for transforms
        # and that rank may need to accept a result mask.
        # KNOWN ISSUE: the inline comment says rank is the only blocklist member
        # reaching the cast step, but the cast is skipped for all blocklisted
        # names, so rank is actually skipped too; the comment is misleading.
        orig_values = values

        dtype = values.dtype
        is_numeric = dtype.kind in "iufcb"

        is_datetimelike = dtype.kind in "mM"

        if self.how in ["any", "all"]:
            if mask is None:
                mask = isna(values)

        if is_datetimelike:
            values = values.view("int64")
            is_numeric = True
        elif dtype.kind == "b":
            values = values.view("uint8")
        if values.dtype == "float16":
            values = values.astype(np.float32)

        if self.how in ["any", "all"]:
            if dtype == object:
                if kwargs["skipna"]:
                    # GH#37501: don't raise on pd.NA when skipna=True
                    if mask is not None and mask.any():
                        # mask on original values computed separately
                        values = values.copy()
                        values[mask] = True
            values = values.astype(bool, copy=False).view(np.int8)
            is_numeric = True

        values = values.T
        if mask is not None:
            mask = mask.T
            if result_mask is not None:
                result_mask = result_mask.T

        out_shape = self._get_output_shape(ngroups, values)
        func = self._get_cython_function(self.kind, self.how, values.dtype, is_numeric)
        values = self._get_cython_vals(values)
        out_dtype = self._get_out_dtype(values.dtype)

        result = maybe_fill(np.empty(out_shape, dtype=out_dtype))
        if self.kind == "aggregate":
            counts = np.zeros(ngroups, dtype=np.int64)
            if self.how in [
                "idxmin",
                "idxmax",
                "min",
                "max",
                "mean",
                "last",
                "first",
                "sum",
                "median",
            ]:
                if self.how == "sum":
                    # pass in through kwargs only for sum (other functions don't have
                    # the keyword)
                    kwargs["initial"] = initial
                func(
                    out=result,
                    counts=counts,
                    values=values,
                    labels=comp_ids,
                    min_count=min_count,
                    mask=mask,
                    result_mask=result_mask,
                    is_datetimelike=is_datetimelike,
                    **kwargs,
                )
            elif self.how in ["sem", "std", "var", "ohlc", "prod"]:
                if self.how in ["std", "sem"]:
                    kwargs["is_datetimelike"] = is_datetimelike
                func(
                    result,
                    counts,
                    values,
                    comp_ids,
                    min_count=min_count,
                    mask=mask,
                    result_mask=result_mask,
                    **kwargs,
                )
            elif self.how in ["any", "all"]:
                func(
                    out=result,
                    values=values,
                    labels=comp_ids,
                    mask=mask,
                    result_mask=result_mask,
                    **kwargs,
                )
                result = result.astype(bool, copy=False)
            elif self.how in ["skew", "kurt"]:
                func(
                    out=result,
                    counts=counts,
                    values=values,
                    labels=comp_ids,
                    mask=mask,
                    result_mask=result_mask,
                    **kwargs,
                )
                if dtype == object:
                    result = result.astype(object)

            else:
                raise NotImplementedError(f"{self.how} is not implemented")
        else:
            # TODO: min_count
            if self.how != "rank":
                # TODO: should rank take result_mask?
                kwargs["result_mask"] = result_mask
            func(
                out=result,
                values=values,
                labels=comp_ids,
                ngroups=ngroups,
                is_datetimelike=is_datetimelike,
                mask=mask,
                **kwargs,
            )

        if self.kind == "aggregate" and self.how not in ["idxmin", "idxmax"]:
            # i.e. counts is defined.  Locations where count<min_count
            # need to have the result set to np.nan, which may require casting,
            # see GH#40767. For idxmin/idxmax is handled specially via post-processing
            if result.dtype.kind in "iu" and not is_datetimelike:
                # if the op keeps the int dtypes, we have to use 0
                cutoff = max(0 if self.how in ["sum", "prod"] else 1, min_count)
                empty_groups = counts < cutoff
                if empty_groups.any():
                    if result_mask is not None:
                        assert result_mask[empty_groups].all()
                    else:
                        # Note: this conversion could be lossy, see GH#40767
                        result = result.astype("float64")
                        result[empty_groups] = np.nan

        result = result.T

        if self.how not in self.cast_blocklist:
            # e.g. if we are int64 and need to restore to datetime64/timedelta64
            # "rank" is the only member of cast_blocklist we get here
            # Casting only needed for float16, bool, datetimelike,
            #  and self.how in ["sum", "prod", "ohlc", "cumprod"]
            res_dtype = self._get_result_dtype(orig_values.dtype)
            op_result = maybe_downcast_to_dtype(result, res_dtype)
        else:
            op_result = result

        return op_result

    @final
    def _validate_axis(self, axis: AxisInt, values: ArrayLike) -> None:
        # [FUNCTION DOC]
        # HELPER: checks that axis is consistent with the data. Raises a
        # not-implemented error if values has more than two dimensions. For
        # 2-D data asserts the axis is one; for 1-D data that is not a 1-D-only
        # extension dtype, asserts the axis is zero. Returns nothing.
        if values.ndim > 2:
            raise NotImplementedError("number of dimensions is currently limited to 2")
        if values.ndim == 2:
            assert axis == 1, axis
        elif not is_1d_only_ea_dtype(values.dtype):
            # Note: it is *not* the case that axis is always 0 for 1-dim values,
            #  as we can have 1D ExtensionArrays that we need to treat as 2D
            assert axis == 0

    @final
    def cython_operation(
        self,
        *,
        values: ArrayLike,
        axis: AxisInt,
        min_count: int = -1,
        comp_ids: np.ndarray,
        ngroups: int,
        **kwargs,
    ) -> ArrayLike:
        # [FUNCTION DOC]
        # Public entry point. Keyword-only inputs: values (numpy or extension
        # array), axis (validated), min_count (default minus one), comp_ids
        # (group id per row), ngroups, and extra keywords forwarded on.
        # Validates the axis, then: extension arrays delegate to their own
        # group operation method (passing the operation name, dropped-null flag,
        # min_count, group count and ids); numpy arrays go through the
        # dimension-compat wrapper with no mask. Returns the result array.
        """
        Call our cython function, with appropriate pre- and post- processing.
        """
        self._validate_axis(axis, values)

        if not isinstance(values, np.ndarray):
            # i.e. ExtensionArray
            return values._groupby_op(
                how=self.how,
                has_dropped_na=self.has_dropped_na,
                min_count=min_count,
                ngroups=ngroups,
                ids=comp_ids,
                **kwargs,
            )

        return self._cython_op_ndim_compat(
            values,
            min_count=min_count,
            ngroups=ngroups,
            comp_ids=comp_ids,
            mask=None,
            **kwargs,
        )


class BaseGrouper:
    # [CLASS DOC]
    # Internal grouper holding the computed groups. State: the axis index being
    # grouped, a list of groupings (one per key), the sort flag and the dropna
    # flag. Most information is derived lazily and cached: the result index
    # with the group id of every row (a combined cache), group count, sizes,
    # label-to-position mappings, the sorting permutation, whether null keys
    # were dropped, and an observed-only version. Key methods: iteration over
    # groups, aggregation of a series with a python function, per-group
    # application of a function, compiled-operation dispatch and creation of
    # the splitters that cut data per group. Subclassed by the bin grouper.
    """
    This is an internal Grouper class, which actually holds
    the generated groups

    Parameters
    ----------
    axis : Index
    groupings : Sequence[Grouping]
        all the grouping instances to handle in this grouper
        for example for grouper list to groupby, need to pass the list
    sort : bool, default True
        whether this grouper will give sorted result or not

    """

    axis: Index

    def __init__(
        self,
        axis: Index,
        groupings: list[grouper.Grouping],
        sort: bool = True,
        dropna: bool = True,
    ) -> None:
        # [FUNCTION DOC]
        # Parameters: axis (index being grouped; an assertion error is raised if
        # it is not an index), groupings (list of grouping objects), sort
        # (default true) and dropna (default true; whether null keys are
        # excluded). Stores them; nothing is computed yet.
        # KNOWN ISSUE: the class docstring does not document the dropna parameter.
        assert isinstance(axis, Index), axis

        self.axis = axis
        self._groupings = groupings
        self._sort = sort
        self.dropna = dropna

    @property
    def groupings(self) -> list[grouper.Grouping]:
        # [FUNCTION DOC]
        # Property: the list of grouping objects, one per grouping key.
        return self._groupings

    def __iter__(self) -> Iterator[Hashable]:
        # [FUNCTION DOC]
        # Iterates over the group labels (keys of the label-to-positions mapping).
        return iter(self.indices)

    @property
    def nkeys(self) -> int:
        # [FUNCTION DOC]
        # Property: the number of grouping keys.
        return len(self.groupings)

    def get_iterator(self, data: NDFrameT) -> Iterator[tuple[Hashable, NDFrameT]]:
        # [FUNCTION DOC]
        # Generator over the groups of data (series or frame). Builds a
        # splitter and pairs each result-index label with the corresponding
        # subset, in result order (unobserved categorical groups give empty
        # subsets). Yields (label, subset) tuples; lengths must match or an error
        # is raised by the strict pairing.
        """
        Groupby iterator

        Returns
        -------
        Generator yielding sequence of (name, subsetted object)
        for each group
        """
        splitter = self._get_splitter(data)
        # TODO: Would be more efficient to skip unobserved for transforms
        keys = self.result_index
        yield from zip(keys, splitter, strict=True)

    @final
    def _get_splitter(self, data: NDFrame) -> DataSplitter:
        # [FUNCTION DOC]
        # HELPER: creates the splitter for data: a series splitter for a
        # series, a frame splitter otherwise. It is built with the group count,
        # the group ids in sorted order and the sorting permutation of rows.
        # Returns the (lazily iterable) splitter.
        # KNOWN ISSUE: the docstring's Returns section describes a generator of
        # subsets; the function actually returns the splitter object.
        """
        Returns
        -------
        Generator yielding subsetted objects
        """
        if isinstance(data, Series):
            klass: type[DataSplitter] = SeriesSplitter
        else:
            # i.e. DataFrame
            klass = FrameSplitter

        return klass(
            data,
            self.ngroups,
            sorted_ids=self._sorted_ids,
            sort_idx=self.result_ilocs,
        )

    @cache_readonly
    def indices(self) -> dict[Hashable, npt.NDArray[np.intp]]:
        # [FUNCTION DOC]
        # Cached mapping from group label to the integer positions of its rows.
        # With a single grouping and a categorical result index, uses the
        # grouping's own mapping (so unused categories appear). Otherwise builds
        # it from the groupings' codes and the levels. When dropna is false,
        # missing keys are normalized to the float not-a-number value (each
        # tuple component for multi-level keys) so missing groups collapse to
        # one key.
        """dict {group name -> group indices}"""
        if len(self.groupings) == 1 and isinstance(self.result_index, CategoricalIndex):
            # This shows unused categories in indices GH#38642
            result = self.groupings[0].indices
        else:
            codes_list = [ping.codes for ping in self.groupings]
            result = get_indexer_dict(codes_list, self.levels)
        if not self.dropna:
            has_mi = isinstance(self.result_index, MultiIndex)
            if not has_mi and self.result_index.hasnans:
                result = {
                    np.nan if isna(key) else key: value for key, value in result.items()
                }
            elif has_mi:
                # MultiIndex has no efficient way to tell if there are NAs
                result = {
                    # error: "Hashable" has no attribute "__iter__" (not iterable)
                    tuple(np.nan if isna(comp) else comp for comp in key): value  # type: ignore[attr-defined]
                    for key, value in result.items()
                }

        return result

    @final
    @cache_readonly
    def result_ilocs(self) -> npt.NDArray[np.intp]:
        # [FUNCTION DOC]
        # Cached permutation of the input rows that orders them by group id
        # (stable per group). When null keys were dropped, rows with negative
        # ids are excluded before sorting, and the sorter output is shifted
        # by the number of dropped rows preceding each position so the positions
        # refer to the original, unfiltered rows.
        """
        Get the original integer locations of result_index in the input.
        """
        # Original indices are where group_index would go via sorting.
        # But when dropna is true, we need to remove null values while accounting for
        # any gaps that then occur because of them.
        ids = self.ids

        if self.has_dropped_na:
            mask = np.where(ids >= 0)
            # Count how many gaps are caused by previous null values for each position
            null_gaps = np.cumsum(ids == -1)[mask]
            ids = ids[mask]

        result = get_group_index_sorter(ids, self.ngroups)

        if self.has_dropped_na:
            # Shift by the number of prior null gaps
            result += np.take(null_gaps, result)

        return result

    @property
    def codes(self) -> list[npt.NDArray[np.signedinteger]]:
        # [FUNCTION DOC]
        # Property: the per-grouping code arrays, one array per key.
        return [ping.codes for ping in self.groupings]

    @property
    def levels(self) -> list[Index]:
        # [FUNCTION DOC]
        # Property: the result index levels. For several groupings, the levels
        # of the multi-level result index; for one grouping, a one-element
        # list holding the result index.
        if len(self.groupings) > 1:
            # mypy doesn't know result_index must be a MultiIndex
            return list(self.result_index.levels)  # type: ignore[attr-defined]
        else:
            return [self.result_index]

    @property
    def names(self) -> list[Hashable]:
        # [FUNCTION DOC]
        # Property: the name of each grouping, in order.
        return [ping.name for ping in self.groupings]

    @final
    def size(self) -> Series:
        # [FUNCTION DOC]
        # Returns a 64-bit integer series indexed by the result index giving the
        # row count of each group. Rows with negative ids (dropped nulls) are not
        # counted; with zero groups the series is empty.
        """
        Compute group sizes.
        """
        ids = self.ids
        ngroups = self.ngroups
        out: np.ndarray | list
        if ngroups:
            out = np.bincount(ids[ids != -1], minlength=ngroups)
        else:
            out = []
        return Series(out, index=self.result_index, dtype="int64", copy=False)

    @cache_readonly
    def groups(self) -> dict[Hashable, Index]:
        # [FUNCTION DOC]
        # Cached mapping from group label to the axis labels of its rows. One
        # grouping delegates to that grouping. Several groupings take the
        # result index values and, for each group id, the axis labels at the
        # row positions found by the reverse indexer of the ids.
        """dict {group name -> group labels}"""
        if len(self.groupings) == 1:
            return self.groupings[0].groups
        result_index, ids = self.result_index_and_ids
        values = result_index._values
        categories = Categorical.from_codes(ids, categories=range(len(result_index)))
        result = {
            # mypy is not aware that group has to be an integer
            values[group]: self.axis.take(axis_ilocs)  # type: ignore[call-overload]
            for group, axis_ilocs in categories._reverse_indexer().items()
        }
        return result

    @final
    @cache_readonly
    def is_monotonic(self) -> bool:
        # [FUNCTION DOC]
        # Cached: true if the group ids of rows are monotonically increasing.
        # return if my group orderings are monotonic
        return Index(self.ids, copy=False).is_monotonic_increasing

    @final
    @cache_readonly
    def has_dropped_na(self) -> bool:
        # [FUNCTION DOC]
        # Cached: true if any row has a negative group id (a dropped null key).
        """
        Whether grouper has null value(s) that are dropped.
        """
        return bool((self.ids < 0).any())

    @cache_readonly
    def codes_info(self) -> npt.NDArray[np.intp]:
        # [FUNCTION DOC]
        # Cached: group id of each item in the original grouped axis (same
        # as the ids; the bin grouper overrides this to reorder).
        # return the codes of items in original grouped axis
        return self.ids

    @final
    @cache_readonly
    def ngroups(self) -> int:
        # [FUNCTION DOC]
        # Cached: the number of groups, i.e. the length of the result index.
        return len(self.result_index)

    @property
    def result_index(self) -> Index:
        # [FUNCTION DOC]
        # Property: the index of the result, one entry per group (first element
        # of the combined result index and ids computation).
        return self.result_index_and_ids[0]

    @property
    def ids(self) -> npt.NDArray[np.intp]:
        # [FUNCTION DOC]
        # Property: for each row of the axis, the position of its group in the
        # result index, or minus one for dropped null keys (second element of
        # the combined computation).
        return self.result_index_and_ids[1]

    @cache_readonly
    def result_index_and_ids(self) -> tuple[Index, npt.NDArray[np.intp]]:
        # [FUNCTION DOC]
        # Cached core computation returning (result index, row ids). Each
        # grouping's unique values form an index level (categorical groupings
        # get all original categories). One grouping: that level, named after
        # the grouping, with the grouping's codes as ids. Several groupings,
        # all observed: a multi-level index built only from combinations that
        # occur, via the observed helper. None observed (all unobserved
        # categoricals): the full cartesian product via the unobserved helper.
        # A mix: the observed and unobserved parts are computed separately,
        # combined as a product with unobserved levels varying slowest by
        # construction, levels reordered to the original order and ids combined
        # arithmetically. If any grouping sorts, the combined index is
        # sorted (only by levels whose sort flag is on) and ids are recoded;
        # if none sort, ids are compressed in order of appearance and the index
        # reordered with occurring groups first, unobserved last. When dropna
        # is on, rows whose observed or unobserved part id is negative get id
        # minus one.
        levels = [
            Index._with_infer(ping.uniques, copy=False) for ping in self.groupings
        ]
        obs = [
            ping._observed or not ping._passed_categorical for ping in self.groupings
        ]
        sorts = [ping._sort for ping in self.groupings]
        # When passed a categorical grouping, keep all categories
        for k, (ping, level) in enumerate(zip(self.groupings, levels, strict=True)):
            if ping._passed_categorical:
                levels[k] = level.set_categories(ping._orig_cats)

        if len(self.groupings) == 1:
            result_index = levels[0]
            result_index.name = self.names[0]
            ids = ensure_platform_int(self.codes[0])
        elif all(obs):
            result_index, ids = self._ob_index_and_ids(
                levels, self.codes, self.names, sorts
            )
        elif not any(obs):
            result_index, ids = self._unob_index_and_ids(levels, self.codes, self.names)
        else:
            # Combine unobserved and observed parts
            names = self.names
            codes = [ping.codes for ping in self.groupings]
            ob_indices = [idx for idx, ob in enumerate(obs) if ob]
            unob_indices = [idx for idx, ob in enumerate(obs) if not ob]
            ob_index, ob_ids = self._ob_index_and_ids(
                levels=[levels[idx] for idx in ob_indices],
                codes=[codes[idx] for idx in ob_indices],
                names=[names[idx] for idx in ob_indices],
                sorts=[sorts[idx] for idx in ob_indices],
            )
            unob_index, unob_ids = self._unob_index_and_ids(
                levels=[levels[idx] for idx in unob_indices],
                codes=[codes[idx] for idx in unob_indices],
                names=[names[idx] for idx in unob_indices],
            )

            result_index_codes = np.concatenate(
                [
                    np.tile(unob_index.codes, len(ob_index)),
                    np.repeat(ob_index.codes, len(unob_index), axis=1),
                ],
                axis=0,
            )
            _, index = np.unique(unob_indices + ob_indices, return_index=True)
            result_index = MultiIndex(
                levels=list(unob_index.levels) + list(ob_index.levels),
                codes=result_index_codes,
                names=list(unob_index.names) + list(ob_index.names),
            ).reorder_levels(index)

            # The sum here will get -1 values wrong when dropna=True;
            # we will fix at the end.
            ids = len(unob_index) * ob_ids + unob_ids

            if any(sorts):
                # Sort result_index and recode ids using the new order
                n_levels = len(sorts)
                drop_levels = [
                    n_levels - idx
                    for idx, sort in enumerate(reversed(sorts), 1)
                    if not sort
                ]
                if len(drop_levels) > 0:
                    sorter = result_index._drop_level_numbers(drop_levels).argsort()
                else:
                    sorter = result_index.argsort()
                result_index = result_index.take(sorter)
                _, index = np.unique(sorter, return_index=True)
                ids = ensure_platform_int(ids)
                ids = index.take(ids)
            else:
                # Recode ids and reorder result_index with observed groups up front,
                # unobserved at the end
                ids, uniques = compress_group_index(ids, sort=False)
                ids = ensure_platform_int(ids)
                taker = np.concatenate(
                    [uniques, np.delete(np.arange(len(result_index)), uniques)]
                )
                result_index = result_index.take(taker)

            if self.dropna:
                ids = np.where((ob_ids < 0) | (unob_ids < 0), -1, ids)

        return result_index, ids

    @property
    def observed_grouper(self) -> BaseGrouper:
        # [FUNCTION DOC]
        # Property: this grouper if every grouping is already observed-only,
        # otherwise the cached observed-only grouper.
        if all(ping._observed for ping in self.groupings):
            return self

        return self._observed_grouper

    @cache_readonly
    def _observed_grouper(self) -> BaseGrouper:
        # [FUNCTION DOC]
        # HELPER: cached; a new base grouper over the same axis whose groupings
        # are the observed-only versions, with the same sort and dropna.
        groupings = [ping.observed_grouping for ping in self.groupings]
        grouper = BaseGrouper(self.axis, groupings, sort=self._sort, dropna=self.dropna)
        return grouper

    def _ob_index_and_ids(
        self,
        levels: list[Index],
        codes: list[npt.NDArray[np.intp]],
        names: list[Hashable],
        sorts: list[bool],
    ) -> tuple[MultiIndex, npt.NDArray[np.intp]]:
        # [FUNCTION DOC]
        # HELPER: for the result index computation. Builds the multi-level
        # index of observed combinations. Inputs: levels, per-key codes, names
        # and per-key sort flags. Combines codes into one id (missing keys
        # propagate as null), compresses to occurring combinations (sorted when
        # all flags agree and are true), and deconstructs back into level codes.
        # Returns the multi-level index (no integrity check) and the platform
        # integer ids per row (negative for null). If the flags disagree and
        # the index is non-empty, the index is sorted using only the levels
        # whose flag is true, and ids are remapped (nulls stay minus one).

        sort_in_compress = sorts[0] if consistent_sorting else False
        shape = tuple(len(level) for level in levels)
        group_index = get_group_index(codes, shape, sort=True, xnull=True)
        ob_ids, obs_group_ids = compress_group_index(group_index, sort=sort_in_compress)
        ob_ids = ensure_platform_int(ob_ids)
        ob_index_codes = decons_obs_group_ids(
            ob_ids, obs_group_ids, shape, codes, xnull=True
        )
        ob_index = MultiIndex(
            levels=levels,
            codes=ob_index_codes,
            names=names,
            verify_integrity=False,
        )
        if not consistent_sorting and len(ob_index) > 0:
            # Sort by the levels where the corresponding sort argument is True
            n_levels = len(sorts)
            drop_levels = [
                n_levels - idx
                for idx, sort in enumerate(reversed(sorts), 1)
                if not sort
            ]
            if len(drop_levels) > 0:
                sorter = ob_index._drop_level_numbers(drop_levels).argsort()
            else:
                sorter = ob_index.argsort()
            ob_index = ob_index.take(sorter)
            _, index = np.unique(sorter, return_index=True)
            ob_ids = np.where(ob_ids == -1, -1, index.take(ob_ids))
        ob_ids = ensure_platform_int(ob_ids)
        return ob_index, ob_ids

    def _unob_index_and_ids(
        self,
        levels: list[Index],
        codes: list[npt.NDArray[np.intp]],
        names: list[Hashable],
    ) -> tuple[MultiIndex, npt.NDArray[np.intp]]:
        # [FUNCTION DOC]
        # HELPER: for the result index computation. Returns the full cartesian
        # product of the levels as a named multi-level index, and per-row ids
        # computed from the codes against the level sizes (sorted layout, nulls
        # propagate as null), as platform integers.
        shape = tuple(len(level) for level in levels)
        unob_ids = get_group_index(codes, shape, sort=True, xnull=True)
        unob_index = MultiIndex.from_product(levels, names=names)
        unob_ids = ensure_platform_int(unob_ids)
        return unob_index, unob_ids

    @final
    def get_group_levels(self) -> Generator[Index]:
        # [FUNCTION DOC]
        # Generator of per-key result levels. One grouping yields the result
        # index once. Several groupings yield the level values of the
        # multi-level result index from the last level to the first, each
        # having full length (one value per group). Only used for base groupers.
        # Note: only called from _insert_inaxis_grouper, which
        #  is only called for BaseGrouper, never for BinGrouper
        result_index = self.result_index
        if len(self.groupings) == 1:
            yield result_index
        else:
            for level in range(result_index.nlevels - 1, -1, -1):
                yield result_index.get_level_values(level)

    # ------------------------------------------------------------
    # Aggregation functions

    @final
    def _cython_operation(
        self,
        kind: str,
        values,
        how: str,
        axis: AxisInt,
        min_count: int = -1,
        **kwargs,
    ) -> ArrayLike:
        # [FUNCTION DOC]
        # HELPER: runs a compiled groupby operation. Inputs: kind (must be
        # transform or aggregate, else an assertion error), values, how
        # (operation name), axis, min_count (default minus one) and extra
        # keywords. Creates the dispatcher with this grouper's dropped-null
        # flag and calls it with the row ids and group count. Returns the result
        # array.
        """
        Returns the values of a cython operation.
        """
        assert kind in ["transform", "aggregate"]

        cy_op = WrappedCythonOp(kind=kind, how=how, has_dropped_na=self.has_dropped_na)

        return cy_op.cython_operation(
            values=values,
            axis=axis,
            min_count=min_count,
            comp_ids=self.ids,
            ngroups=self.ngroups,
            **kwargs,
        )

    @final
    def agg_series(
        self, obj: Series, func: Callable, preserve_dtype: bool = False
    ) -> ArrayLike:
        # [FUNCTION DOC]
        # Aggregates a series group by group with a python function. obj is
        # the series; func takes a group series and returns a scalar-like;
        # preserve_dtype is accepted but unused. Runs the pure-python
        # aggregation helper and asks the series' array to cast the object
        # array of results to its appropriate type. Returns a numpy or
        # extension array with one entry per group.
        # KNOWN ISSUE: preserve_dtype is documented as controlling dtype
        # preservation but the body never reads it.
        """
        Parameters
        ----------
        obj : Series
        func : function taking a Series and returning a scalar-like
        preserve_dtype : bool
            Whether the aggregation is known to be dtype-preserving.

        Returns
        -------
        np.ndarray or ExtensionArray
        """
        result = self._aggregate_series_pure_python(obj, func)
        return obj.array._cast_pointwise_result(result)

    @final
    def _aggregate_series_pure_python(
        self, obj: Series, func: Callable
    ) -> npt.NDArray[np.object_]:
        # [FUNCTION DOC]
        # HELPER: for the series aggregation method. Applies func to each group
        # (obtained from a splitter) in result order, unwraps each result with
        # the extractor, validates only the first result (raises a value error
        # if an aggregation returned a non-object array) and stores everything
        # in an object array of length equal to the group count, which it returns.
        result = np.empty(self.ngroups, dtype="O")
        initialized = False

        splitter = self._get_splitter(obj)

        for i, group in enumerate(splitter):
            res = func(group)
            res = extract_result(res)

            if not initialized:
                # We only do this validation on the first iteration
                check_result_array(res, group.dtype)
                initialized = True

            result[i] = res

        return result

    @final
    def apply_groupwise(
        self, f: Callable, data: DataFrame | Series
    ) -> tuple[list, bool]:
        # [FUNCTION DOC]
        # Applies f to every group of data (series or frame). Each group, in
        # result order, has its name attribute set to the group label (bypassing
        # normal attribute setting), then f is called on it. Records whether any
        # result is not indexed like its group (mutated flag; computed against the
        # group's axes captured before the call). If there are no groups and f
        # is named skew, kurt, sum or product, calls f once on an empty slice
        # of data so the usual dtype error surfaces. Returns (list of results,
        # mutated flag).
        mutated = False
        splitter = self._get_splitter(data)
        group_keys = self.result_index
        result_values = []

        # This calls DataSplitter.__iter__
        zipped = zip(group_keys, splitter, strict=True)

        for key, group in zipped:
            # Pinning name is needed for
            #  test_group_apply_once_per_group,
            #  test_inconsistent_return_type, test_set_group_name,
            #  test_group_name_available_in_inference_pass,
            #  test_groupby_multi_timezone
            object.__setattr__(group, "name", key)

            # group might be modified
            group_axes = group.axes
            res = f(group)
            if not mutated and not _is_indexed_like(res, group_axes):
                mutated = True
            result_values.append(res)
        # getattr pattern for __name__ is needed for functools.partial objects
        if len(group_keys) == 0 and getattr(f, "__name__", None) in [
            "skew",
            "kurt",
            "sum",
            "prod",
        ]:
            #  If group_keys is empty, then no function calls have been made,
            #  so we will not have raised even if this is an invalid dtype.
            #  So do one dummy call here to raise appropriate TypeError.
            f(data.iloc[:0])

        return result_values, mutated

    # ------------------------------------------------------------
    # Methods for sorting subsets of our GroupBy's object

    @final
    @cache_readonly
    def _sorted_ids(self) -> npt.NDArray[np.intp]:
        # [FUNCTION DOC]
        # HELPER: cached; group ids arranged in the sorting permutation order,
        # with negative ids (dropped nulls) removed when dropna applies (the
        # attribute defaults to true if absent, as the bin grouper lacks it).
        result = self.ids.take(self.result_ilocs)
        if getattr(self, "dropna", True):
            # BinGrouper has no dropna
            result = result[result >= 0]
        return result


class BinGrouper(BaseGrouper):
    # [CLASS DOC]
    # Internal grouper for contiguous, ordered bins of the axis (used by
    # time-based resampling). State: bins (end positions of each bin as 64-bit
    # integers), binlabels (index of bin labels, same length as bins), and an
    # optional indexer from a sorting grouper. Subclass of the base grouper
    # but does not call its initializer, so axis, stored groupings, sort and
    # dropna attributes are absent; groupings are synthesized on demand. A
    # missing (not-a-time) first label marks leading rows that belong to no bin.
    """
    This is an internal Grouper class

    Parameters
    ----------
    bins : the split index of binlabels to group the item of axis
    binlabels : the label list
    indexer : np.ndarray[np.intp], optional
        the indexer created by Grouper
        some groupers (TimeGrouper) will sort its axis and its
        group_info is also sorted, so need the indexer to reorder

    Examples
    --------
    bins: [2, 4, 6, 8, 10]
    binlabels: DatetimeIndex(['2005-01-01', '2005-01-03',
        '2005-01-05', '2005-01-07', '2005-01-09'],
        dtype='datetime64[ns]', freq='2D')

    the group_info, which contains the label of each item in grouped
    axis, the index of label in label list, group number, is

    (array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]), array([0, 1, 2, 3, 4]), 5)

    means that, the grouped axis has 10 items, can be grouped into 5
    labels, the first and second items belong to the first label, the
    third and forth items belong to the second label, and so on

    """

    bins: npt.NDArray[np.int64]
    binlabels: Index

    def __init__(
        self,
        bins,
        binlabels,
        indexer=None,
    ) -> None:
        # [FUNCTION DOC]
        # Parameters: bins (array-like of bin end positions, converted to
        # 64-bit integers), binlabels (labels, converted to an index) and indexer
        # (optional sort permutation from the creating grouper, default none).
        # Raises an assertion error if the label and bin counts differ.
        self.bins = ensure_int64(bins)
        self.binlabels = ensure_index(binlabels)
        self.indexer = indexer

        # These lengths must match, otherwise we could call agg_series
        #  with empty self.bins, which would raise later.
        assert len(self.binlabels) == len(self.bins)

    @cache_readonly
    def groups(self):
        # [FUNCTION DOC]
        # Cached mapping from each bin label to the end position of its bin,
        # skipping missing (not-a-time) labels. Kept for compatibility.
        # KNOWN ISSUE: the docstring says the values are group labels, but the
        # values are bin end positions (integers).
        """dict {group name -> group labels}"""
        # this is mainly for compat
        # GH 3881
        result = {
            key: value
            for key, value in zip(self.binlabels, self.bins, strict=True)
            if key is not NaT
        }
        return result

    @property
    def nkeys(self) -> int:
        # [FUNCTION DOC]
        # Property: always one key (hard-coded).
        # still matches len(self.groupings), but we can hard-code
        return 1

    @cache_readonly
    def codes_info(self) -> npt.NDArray[np.intp]:
        # [FUNCTION DOC]
        # Cached: the ids of items. If a sort indexer exists, the ids are
        # reordered by a lexicographic sort keyed first on the indexer and then
        # on the ids; otherwise the ids are returned as is.
        # return the codes of items in original grouped axis
        ids = self.ids
        if self.indexer is not None:
            sorter = np.lexsort((ids, self.indexer))
            ids = ids[sorter]
        return ids

    def get_iterator(self, data: NDFrame):
        # [FUNCTION DOC]
        # Generator over bins of data (series or frame), slicing by position.
        # For each bin end and label, yields (label, slice from the previous end
        # to this end), skipping missing labels (but still advancing the start).
        # If rows remain after the last bin end, yields them with the last bin
        # label. Slicing is positional.
        """
        Groupby iterator

        Returns
        -------
        Generator yielding sequence of (name, subsetted object)
        for each group
        """
        slicer = lambda start, edge: data.iloc[start:edge]

        start: np.int64 | int = 0
        for edge, label in zip(self.bins, self.binlabels, strict=True):
            if label is not NaT:
                yield label, slicer(start, edge)
            start = edge

        if start < len(data):
            yield self.binlabels[-1], slicer(start, None)

    @cache_readonly
    def indices(self):
        # [FUNCTION DOC]
        # Cached mapping (a default dictionary of lists) from bin label to the
        # list of row positions in that bin. Empty bins and missing labels are
        # omitted; a position cursor advances to each non-empty bin's end.
        indices = collections.defaultdict(list)

        i: np.int64 | int = 0
        for label, bin in zip(self.binlabels, self.bins, strict=True):
            if i < bin:
                if label is not NaT:
                    indices[label] = list(range(i, bin))
                i = bin
        return indices

    @cache_readonly
    def codes(self) -> list[npt.NDArray[np.intp]]:
        # [FUNCTION DOC]
        # Cached: a one-element list holding the row ids.
        return [self.ids]

    @cache_readonly
    def result_index_and_ids(self):
        # [FUNCTION DOC]
        # Cached (result index, ids) pair. The result index is the bin labels,
        # minus a leading missing label if present. Ids repeat each bin number
        # by the bin's width (differences of bin ends from zero); when a leading
        # missing label was removed, the first bin's rows get id minus one.
        # Returns platform integer ids.
        result_index = self.binlabels
        if len(self.binlabels) != 0 and isna(self.binlabels[0]):
            result_index = result_index[1:]

        ngroups = len(result_index)
        rep = np.diff(np.r_[0, self.bins])

        rep = ensure_platform_int(rep)
        if ngroups == len(self.bins):
            ids = np.repeat(np.arange(ngroups), rep)
        else:
            ids = np.repeat(np.r_[-1, np.arange(ngroups)], rep)
        ids = ensure_platform_int(ids)

        return result_index, ids

    @property
    def levels(self) -> list[Index]:
        # [FUNCTION DOC]
        # Property: a one-element list holding the bin labels.
        return [self.binlabels]

    @property
    def names(self) -> list[Hashable]:
        # [FUNCTION DOC]
        # Property: a one-element list holding the bin labels' name.
        return [self.binlabels.name]

    @property
    def groupings(self) -> list[grouper.Grouping]:
        # [FUNCTION DOC]
        # Property: builds on each access a one-element list with a grouping
        # whose axis and key are the bin labels taken at the row ids, with the
        # bin label values as the explicit uniques (so empty bins are kept).
        # Not cached.
        lev = self.binlabels
        codes = self.ids
        labels = lev.take(codes)
        ping = grouper.Grouping(
            labels, labels, in_axis=False, level=None, uniques=lev._values
        )
        return [ping]

    @property
    def observed_grouper(self) -> BinGrouper:
        # [FUNCTION DOC]
        # Property: returns this grouper itself (bins are always kept).
        return self


def _is_indexed_like(obj, axes) -> bool:
    # [FUNCTION DOC]
    # HELPER: for the groupwise apply. True if a function result keeps the
    # shape of its input group: for a series result, only when the group has a
    # single axis and the series index equals it; for a frame result, when its
    # index equals the first axis. Any other result type gives false.
    if isinstance(obj, Series):
        if len(axes) > 1:
            return False
        return obj.index.equals(axes[0])
    elif isinstance(obj, DataFrame):
        return obj.index.equals(axes[0])

    return False


# ----------------------------------------------------------------------
# Splitting / application


class DataSplitter(Generic[NDFrameT]):
    # [CLASS DOC]
    # Abstract, iterable splitter that cuts a group-sorted object into one
    # piece per group. State: the data, the group count, the group ids in
    # sorted order, and the sorting permutation of rows. Iteration yields the
    # pieces in group order; the data is reordered once (cached). Subclasses
    # define how a contiguous slice is turned into a series or frame.
    def __init__(
        self,
        data: NDFrameT,
        ngroups: int,
        *,
        sort_idx: npt.NDArray[np.intp],
        sorted_ids: npt.NDArray[np.intp],
    ) -> None:
        # [FUNCTION DOC]
        # Parameters: data (object to split), ngroups (number of groups),
        # sort_idx (keyword only; row permutation ordering rows by group) and
        # sorted_ids (keyword only; group id of each row in that order). Stores
        # them; no validation.
        self.data = data
        self.ngroups = ngroups

        self._slabels = sorted_ids
        self._sort_idx = sort_idx

    def __iter__(self) -> Iterator:
        # [FUNCTION DOC]
        # Generator yielding one piece per group, in group order. Yields nothing
        # when there are no groups. Otherwise computes start and end offsets of
        # each group from the sorted ids, reorders the data once, and cuts each
        # contiguous slice with the subclass chop method.
        if self.ngroups == 0:
            # we are inside a generator, rather than raise StopIteration
            # we merely return signal the end
            return

        starts, ends = lib.generate_slices(self._slabels, self.ngroups)
        sdata = self._sorted_data
        for start, end in zip(starts, ends, strict=True):
            yield self._chop(sdata, slice(start, end))

    @cache_readonly
    def _sorted_data(self) -> NDFrameT:
        # [FUNCTION DOC]
        # HELPER: cached; the data with rows taken in the sorting permutation
        # order along the row axis (a reordered copy).
        return self.data.take(self._sort_idx, axis=0)

    def _chop(self, sdata, slice_obj: slice) -> NDFrame:
        # [FUNCTION DOC]
        # HELPER: abstract hook. Subclasses return the piece of the sorted data
        # for the given slice; this base version raises the abstract-method error.
        raise AbstractMethodError(self)


class SeriesSplitter(DataSplitter):
    # [CLASS DOC]
    # Splitter for series data. Subclass of the base splitter; only defines
    # the slicing step.
    def _chop(self, sdata: Series, slice_obj: slice) -> Series:
        # [FUNCTION DOC]
        # HELPER: fast equivalent of positional slicing of a series. Slices
        # the underlying manager, builds a series of the same type from it,
        # copies the name, and propagates metadata marked as from groupby.
        # Returns the new series.
        # fastpath equivalent to `sdata.iloc[slice_obj]`
        mgr = sdata._mgr.get_slice(slice_obj)
        ser = sdata._constructor_from_mgr(mgr, axes=mgr.axes)
        ser._name = sdata.name
        return ser.__finalize__(sdata, method="groupby")


class FrameSplitter(DataSplitter):
    # [CLASS DOC]
    # Splitter for frame data. Subclass of the base splitter; only defines
    # the slicing step.
    def _chop(self, sdata: DataFrame, slice_obj: slice) -> DataFrame:
        # [FUNCTION DOC]
        # HELPER: fast equivalent of positional row slicing of a frame. Slices
        # the manager along its row axis, builds a frame of the same type from it
        # and propagates metadata marked as from groupby. Returns the new frame.
        # Fastpath equivalent to:
        # return sdata.iloc[slice_obj]
        mgr = sdata._mgr.get_slice(slice_obj, axis=1)
        df = sdata._constructor_from_mgr(mgr, axes=mgr.axes)
        return df.__finalize__(sdata, method="groupby")
