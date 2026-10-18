"""
SparseArray data structure
"""

# [MODULE DOC]
# Sparse extension array implementation. Public contents: the sparse array class
# (exposed in the arrays namespace) plus re-exports of the compiled sparse index
# classes (block layout and integer layout, with the common base) and the index
# factory function. Private helpers: a function giving the fill value as a 0-
# dimensional array, the binary-operation engine for two sparse arrays, the
# result wrapper, the sparse construction routine from dense data, and the
# sparse index builder.
# Storage model: an array stores only the values that differ from a fill value
# (kept in a dense array of stored values), plus a sparse index recording the
# positions of those values within the full length; everything else is implicit.
# Relationships: the sparse dtype (carrying subtype and fill value) lives in the
# dtypes module; compiled sparse algorithms live in the sparse library module
# (operations named by operator and subtype); the sibling accessor module
# exposes the series/frame accessors and the scipy module converts to scipy
# matrices. Operator dispatch uses the operator mixin and the ufunc protocol.
# No import-time side effects beyond decorator application.
from __future__ import annotations

from collections import abc
import numbers
import operator
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
    cast,
    overload,
)
import warnings

import numpy as np

from pandas._config.config import get_option

from pandas._libs import lib
import pandas._libs.sparse as splib
from pandas._libs.sparse import (
    BlockIndex,
    IntIndex,
    SparseIndex,
)
from pandas._libs.tslibs import NaT
from pandas.compat.numpy import function as nv
from pandas.errors import PerformanceWarning
from pandas.util._decorators import (
    doc,
    set_module,
)
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
    validate_bool_kwarg,
    validate_insert_loc,
)

from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.cast import (
    find_common_type,
    maybe_box_datetimelike,
)
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_integer,
    is_list_like,
    is_object_dtype,
    is_scalar,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    SparseDtype,
)
from pandas.core.dtypes.generic import (
    ABCIndex,
    ABCSeries,
)
from pandas.core.dtypes.missing import (
    isna,
    na_value_for_dtype,
    notna,
)

from pandas.core import arraylike
import pandas.core.algorithms as algos
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import ExtensionArray
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.construction import (
    ensure_wrapped_if_datetimelike,
    extract_array,
    sanitize_array,
)
from pandas.core.indexers import (
    check_array_indexer,
    unpack_tuple_and_ellipses,
)
from pandas.core.nanops import check_below_min_count

from pandas.io.formats import printing

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Sequence,
    )
    from types import EllipsisType
    from typing import (
        Protocol,
        type_check_only,
    )

    from scipy.sparse import (
        csc_array,
        csc_matrix,
    )

    @type_check_only
    class _SparseMatrixLike(Protocol):
        # [CLASS DOC]
        # HELPER: typing-only protocol (exists only for static type checking,
        # never at run time) describing the minimal scipy sparse matrix interface
        # needed by the matrix-import constructor: a two-element shape and a
        # conversion to a compressed-column matrix.
        @property
        def shape(self, /) -> tuple[int, int]:
            # [FUNCTION DOC]
            # HELPER: typing-only protocol member: (rows, columns) of the matrix.
            ...

        def tocsc(self, /) -> csc_array | csc_matrix:
            # [FUNCTION DOC]
            # HELPER: typing-only protocol member: convert to a compressed
            # sparse column matrix or array.
            ...

    from pandas._typing import NumpySorter

    SparseIndexKind = Literal["integer", "block"]

    from pandas._typing import (
        ArrayLike,
        AstypeArg,
        Axis,
        AxisInt,
        Dtype,
        NpDtype,
        PositionalIndexer,
        Scalar,
        ScalarIndexer,
        SequenceIndexer,
        npt,
    )

    from pandas import Series


# ----------------------------------------------------------------------------
# Array

_sparray_doc_kwargs = {"klass": "SparseArray"}


def _get_fill(arr: SparseArray) -> np.ndarray:
    # [FUNCTION DOC]
    # HELPER: gives the fill value of a sparse array as a zero-dimensional numpy
    # array. arr is the sparse array. Tries to cast the fill value to the subtype
    # (so e.g. integer data can still carry a NaN fill value only when the cast
    # fails with a value error, in which case the fill value is converted without
    # a target dtype). Used by the binary-operation engine.
    """
    Create a 0-dim ndarray containing the fill value

    Parameters
    ----------
    arr : SparseArray

    Returns
    -------
    fill_value : ndarray
        0-dim ndarray with just the fill value.

    Notes
    -----
    coerce fill_value to arr dtype if possible
    int64 SparseArray can have NaN as fill_value if there is no missing
    """
    try:
        return np.asarray(arr.fill_value, dtype=arr.dtype.subtype)
    except ValueError:
        return np.asarray(arr.fill_value)


def _sparse_array_op(
    left: SparseArray, right: SparseArray, op: Callable, name: str
) -> SparseArray:
    # [FUNCTION DOC]
    # HELPER: binary-operation engine for two sparse arrays. left and right are
    # sparse arrays; op is the operator function; name is the operator name
    # (dunder names are stripped to the bare name, e.g. "add", "rsub", "eq",
    # "divmod").
    # Steps: if the stored subtypes differ, both are cast to their common type
    # (keeping their own fill values). Then three cases, in this order: (1) one
    # side has no gaps (is fully dense): the operation is done densely on the full
    # arrays, the fill is the operation on the two fill values, and the result
    # index is the dense side's index; (2) both have identical index: the
    # operation is applied directly to the stored values and fill values; (3)
    # otherwise the compiled sparse routine matching the operator and subtype is
    # called (reflected names swap the operands; logical operators on booleans use
    # an unsigned 8-bit routine and force boolean results; integer floor division
    # and modulo by zero fall back to the 64-bit float routine to mimic the dense
    # behavior), returning values, index and fill value. Numpy floating-point
    # warnings are suppressed.
    # Output: wrapped result array (comparisons forced to boolean); for the
    # paired quotient-and-remainder operation a tuple of two wrapped arrays.
    # KNOWN ISSUE: the docstring says the operands may be plain numpy arrays, but
    # the code reads sparse-only attributes (stored values, index, fill value),
    # so both must be sparse arrays; the declared return type also omits the
    # tuple result of the paired operation. The docstring also has a typo in the
    # parameter list for the name.
    """
    Perform a binary operation between two arrays.

    Parameters
    ----------
    left : Union[SparseArray, ndarray]
    right : Union[SparseArray, ndarray]
    op : Callable
        The binary operation to perform
    name str
        Name of the callable.

    Returns
    -------
    SparseArray
    """
    if name.startswith("__"):
        # For lookups in _libs.sparse we need non-dunder op name
        name = name[2:-2]

    # dtype used to find corresponding sparse method
    ltype = left.dtype.subtype
    rtype = right.dtype.subtype

    if ltype != rtype:
        subtype = find_common_type([ltype, rtype])
        ltype = SparseDtype(subtype, left.fill_value)
        rtype = SparseDtype(subtype, right.fill_value)

        left = left.astype(ltype, copy=False)
        right = right.astype(rtype, copy=False)
        dtype = ltype.subtype
    else:
        dtype = ltype

    # dtype the result must have
    result_dtype = None

    if left.sp_index.ngaps == 0 or right.sp_index.ngaps == 0:
        with np.errstate(all="ignore"):
            result = op(left.to_dense(), right.to_dense())
            fill = op(_get_fill(left), _get_fill(right))

        if left.sp_index.ngaps == 0:
            index = left.sp_index
        else:
            index = right.sp_index
    elif left.sp_index.equals(right.sp_index):
        with np.errstate(all="ignore"):
            result = op(left.sp_values, right.sp_values)
            fill = op(_get_fill(left), _get_fill(right))
        index = left.sp_index
    else:
        if name[0] == "r":
            left, right = right, left
            name = name[1:]

        if name in ("and", "or", "xor") and dtype == "bool":
            opname = f"sparse_{name}_uint8"
            # to make template simple, cast here
            left_sp_values = left.sp_values.view(np.uint8)
            right_sp_values = right.sp_values.view(np.uint8)
            result_dtype = bool
        else:
            opname = f"sparse_{name}_{dtype}"
            left_sp_values = left.sp_values
            right_sp_values = right.sp_values

        if (
            name in ["floordiv", "mod"]
            and (right == 0).any()
            and left.dtype.kind in "iu"
        ):
            # Match the non-Sparse Series behavior
            opname = f"sparse_{name}_float64"
            left_sp_values = left_sp_values.astype("float64")
            right_sp_values = right_sp_values.astype("float64")

        sparse_op = getattr(splib, opname)

        with np.errstate(all="ignore"):
            result, index, fill = sparse_op(
                left_sp_values,
                left.sp_index,
                left.fill_value,
                right_sp_values,
                right.sp_index,
                right.fill_value,
            )

    if name == "divmod":
        # result is a 2-tuple
        # error: Incompatible return value type (got "Tuple[SparseArray,
        # SparseArray]", expected "SparseArray")
        return (  # type: ignore[return-value]
            _wrap_result(name, result[0], index, fill[0], dtype=result_dtype),
            _wrap_result(name, result[1], index, fill[1], dtype=result_dtype),
        )

    if result_dtype is None:
        result_dtype = result.dtype

    return _wrap_result(name, result, index, fill, dtype=result_dtype)


def _wrap_result(
    name: str, data, sparse_index, fill_value, dtype: Dtype | None = None
) -> SparseArray:
    # [FUNCTION DOC]
    # HELPER: builds the sparse array result of an operation. name is the
    # operator name (dunder stripped); data the stored values; sparse_index the
    # result index; fill_value the fill (possibly a 0-dimensional array, which is
    # unwrapped to a scalar); dtype (default none) the requested value dtype.
    # Comparison operators force a boolean dtype; a boolean dtype converts the
    # fill value to a plain boolean. Returns a new sparse array built with the
    # public constructor (so inputs are re-validated).
    """
    wrap op result to have correct dtype
    """
    if name.startswith("__"):
        # e.g. __eq__ --> eq
        name = name[2:-2]

    if name in ("eq", "ne", "lt", "gt", "le", "ge"):
        dtype = bool

    fill_value = lib.item_from_zerodim(fill_value)

    if is_bool_dtype(dtype):
        # fill_value may be np.bool_
        fill_value = bool(fill_value)
    return SparseArray(
        data, sparse_index=sparse_index, fill_value=fill_value, dtype=dtype
    )


@set_module("pandas.arrays")
class SparseArray(OpsMixin, PandasObject, ExtensionArray):
    # [CLASS DOC]
    # Extension array holding data in sparse form. Inherits the operator mixin
    # (binary operators), the base object for pandas objects (hidden attributes,
    # display helpers) and the generic extension array. State: the array of
    # stored values (the elements different from the fill value), the sparse index
    # (positions of those values; either block layout or integer layout) and the
    # sparse dtype (subtype of the stored values plus the fill value). Invariants:
    # the number of stored values equals the number of points in the index; the
    # array length is the length recorded by the index; the dtype subtype equals
    # the stored values dtype. The array is immutable for element assignment
    # (assignment always raises). Key members: the constructors, conversion to
    # dense/numpy, indexing and slicing, take, fill/unique/factorize/value counts,
    # shifting, cast, reductions, cumulative sum, arithmetic and comparison
    # dispatch, universal-function support, formatting and density properties.
    """
    An ExtensionArray for storing sparse data.

    SparseArray efficiently stores data with a high frequency of a
    specific fill value (e.g., zeros), saving memory by only retaining
    non-fill elements and their indices. This class is particularly
    useful for large datasets where most values are redundant.

    Parameters
    ----------
    data : array-like or scalar
        A dense array of values to store in the SparseArray. This may contain
        `fill_value`.
    sparse_index : SparseIndex, optional
        Index indicating the locations of sparse elements.
    fill_value : scalar, optional
        Elements in data that are ``fill_value`` are not stored in the
        SparseArray. For memory savings, this should be the most common value
        in `data`. By default, `fill_value` depends on the dtype of `data`:

        =========== ==========
        data.dtype  na_value
        =========== ==========
        float       ``np.nan``
        int         ``0``
        bool        False
        datetime64  ``pd.NaT``
        timedelta64 ``pd.NaT``
        =========== ==========

        The fill value is potentially specified in three ways. In order of
        precedence, these are

        1. The `fill_value` argument
        2. ``dtype.fill_value`` if `fill_value` is None and `dtype` is
           a ``SparseDtype``
        3. ``data.dtype.fill_value`` if `fill_value` is None and `dtype`
           is not a ``SparseDtype`` and `data` is a ``SparseArray``.

    kind : str
        Can be 'integer' or 'block', default is 'integer'.
        The type of storage for sparse locations.

        * 'block': Stores a `block` and `block_length` for each
          contiguous *span* of sparse values. This is best when
          sparse data tends to be clumped together, with large
          regions of ``fill-value`` values between sparse values.
        * 'integer': uses an integer to store the location of
          each sparse value.

    dtype : np.dtype or SparseDtype, optional
        The dtype to use for the SparseArray. For numpy dtypes, this
        determines the dtype of ``self.sp_values``. For SparseDtype,
        this determines ``self.sp_values`` and ``self.fill_value``.
    copy : bool, default False
        Whether to explicitly copy the incoming `data` array.

    Attributes
    ----------
    None

    Methods
    -------
    None

    See Also
    --------
    SparseDtype : Dtype for sparse data.

    Examples
    --------
    >>> from pandas.arrays import SparseArray
    >>> arr = SparseArray([0, 0, 1, 2])
    >>> arr
    [0, 0, 1, 2]
    Fill: 0
    IntIndex
    Indices: array([2, 3], dtype=int32)
    """

    _subtyp = "sparse_array"  # register ABCSparseArray
    _hidden_attrs = PandasObject._hidden_attrs | frozenset([])
    _sparse_index: SparseIndex
    _sparse_values: np.ndarray
    _dtype: SparseDtype

    def __init__(
        self,
        data,
        sparse_index=None,
        fill_value=None,
        kind: SparseIndexKind = "integer",
        dtype: Dtype | None = None,
        copy: bool = False,
    ) -> None:
        # [FUNCTION DOC]
        # Constructor from dense data or from stored values plus an index.
        # data: array-like (or another sparse array, or none for an empty float or
        # given-dtype array); scalar data raises a type error. sparse_index
        # (default none): when given, data is taken as the stored values and its
        # length must equal the index's number of points (assertion error
        # otherwise). fill_value (default none): resolved by precedence from the
        # argument, then a sparse dtype argument, then a sparse input array, then
        # the natural missing value of the data dtype. kind (default "integer"; or
        # "block"): layout of the index to build from dense data. dtype (default
        # none): numpy dtype, sparse dtype, or text for either; for a sparse dtype
        # its subtype becomes the dtype and its fill value the default fill.
        # copy (default false): copy dense input.
        # Behavior: input is sanitized to an array (retrying as object dtype if
        # numpy cannot build it and no dtype was given); with no index given,
        # time-zone aware datetime extension data is converted to naive datetime
        # with a user warning, then the dense data is sparsified around the fill
        # value; with an index given the values are simply cast. The final dtype is
        # a sparse dtype of the stored values dtype and the (possibly adjusted)
        # fill value.
        # KNOWN ISSUE: the branch that copies a sparse-array input into new
        # storage (data being an instance of the class after sanitizing) is
        # effectively unreachable when the input was a sparse array, because the
        # constructor unwraps such inputs to their stored values earlier; and when
        # copy is requested together with a dtype that forces a cast, data may be
        # copied twice (a source TODO).
        if fill_value is None and isinstance(dtype, SparseDtype):
            fill_value = dtype.fill_value

        if isinstance(data, type(self)):
            # disable normal inference on dtype, sparse_index, & fill_value
            if sparse_index is None:
                sparse_index = data.sp_index
            if fill_value is None:
                fill_value = data.fill_value
            if dtype is None:
                dtype = data.dtype
            # TODO: make kind=None, and use data.kind?
            data = data.sp_values

        # Handle use-provided dtype
        if isinstance(dtype, str):
            # Two options: dtype='int', regular numpy dtype
            # or dtype='Sparse[int]', a sparse dtype
            try:
                dtype = SparseDtype.construct_from_string(dtype)
            except TypeError:
                dtype = pandas_dtype(dtype)

        if isinstance(dtype, SparseDtype):
            if fill_value is None:
                fill_value = dtype.fill_value
            dtype = dtype.subtype

        if is_scalar(data):
            raise TypeError(
                f"Cannot construct {type(self).__name__} from scalar data. "
                "Pass a sequence instead."
            )

        if dtype is not None:
            dtype = pandas_dtype(dtype)

        # TODO: disentangle the fill_value dtype inference from
        # dtype inference
        if data is None:
            # TODO: What should the empty dtype be? Object or float?

            # error: Argument "dtype" to "array" has incompatible type
            # "Union[ExtensionDtype, dtype[Any], None]"; expected "Union[dtype[Any],
            # None, type, _SupportsDType, str, Union[Tuple[Any, int], Tuple[Any,
            # Union[int, Sequence[int]]], List[Any], _DTypeDict, Tuple[Any, Any]]]"
            data = np.array([], dtype=dtype)  # type: ignore[arg-type]

        try:
            data = sanitize_array(data, index=None)
        except ValueError:
            # NumPy may raise a ValueError on data like [1, []]
            # we retry with object dtype here.
            if dtype is None:
                dtype = np.dtype(object)
                data = np.atleast_1d(np.asarray(data, dtype=dtype))
            else:
                raise

        if copy:
            # TODO: avoid double copy when dtype forces cast.
            data = data.copy()

        if fill_value is None:
            fill_value_dtype = data.dtype if dtype is None else dtype
            if fill_value_dtype is None:
                fill_value = np.nan
            else:
                fill_value = na_value_for_dtype(fill_value_dtype)

        if isinstance(data, type(self)) and sparse_index is None:
            sparse_index = data._sparse_index
            # error: Argument "dtype" to "asarray" has incompatible type
            # "Union[ExtensionDtype, dtype[Any], None]"; expected "None"
            sparse_values = np.asarray(
                data.sp_values,
                dtype=dtype,  # type: ignore[arg-type]
            )
        elif sparse_index is None:
            data = extract_array(data, extract_numpy=True)
            if not isinstance(data, np.ndarray):
                # EA
                if isinstance(data.dtype, DatetimeTZDtype):
                    warnings.warn(
                        f"Creating SparseArray from {data.dtype} data "
                        "loses timezone information. Cast to object before "
                        "sparse to retain timezone information.",
                        UserWarning,
                        stacklevel=find_stack_level(),
                    )
                    data = np.asarray(data, dtype="datetime64[ns]")
                    if fill_value is NaT:
                        fill_value = np.datetime64("NaT", "ns")
                data = np.asarray(data)
            sparse_values, sparse_index, fill_value = _make_sparse(
                # error: Argument "dtype" to "_make_sparse" has incompatible type
                # "Union[ExtensionDtype, dtype[Any], None]"; expected
                # "Optional[dtype[Any]]"
                data,
                kind=kind,
                fill_value=fill_value,
                dtype=dtype,  # type: ignore[arg-type]
            )
        else:
            # error: Argument "dtype" to "asarray" has incompatible type
            # "Union[ExtensionDtype, dtype[Any], None]"; expected "None"
            sparse_values = np.asarray(data, dtype=dtype)  # type: ignore[arg-type]
            if len(sparse_values) != sparse_index.npoints:
                raise AssertionError(
                    f"Non array-like type {type(sparse_values)} must "
                    "have the same length as the index"
                )
        self._sparse_index = sparse_index
        self._sparse_values = sparse_values
        self._dtype = SparseDtype(sparse_values.dtype, fill_value)

    @classmethod
    def _simple_new(
        cls,
        sparse_array: np.ndarray,
        sparse_index: SparseIndex,
        dtype: SparseDtype,
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: fast internal constructor without validation or copying.
        # sparse_array is the numpy array of stored values, sparse_index the
        # index describing their positions and the length, dtype the sparse
        # dtype. Returns the new instance with the three private attributes set.
        new = object.__new__(cls)
        new._sparse_index = sparse_index
        new._sparse_values = sparse_array
        new._dtype = dtype
        return new

    @classmethod
    def from_spmatrix(cls, data: _SparseMatrixLike) -> Self:
        # [FUNCTION DOC]
        # Class method building a sparse array from a single-column scipy sparse
        # matrix. data must have exactly one column (value error otherwise). The
        # matrix is converted to compressed-column form and its stored indices
        # are sorted in place (mutating the converted matrix, which is the
        # caller's object if already in that format) because the integer sparse
        # index requires strictly increasing positions. The fill value is the
        # zero of the matrix dtype; the stored values are the matrix data array.
        # Returns the new array of the matrix's number of rows.
        """
        Create a SparseArray from a scipy.sparse matrix.

        Parameters
        ----------
        data : scipy.sparse.sp_matrix
            This should be a SciPy sparse matrix where the size
            of the second dimension is 1. In other words, a
            sparse matrix with a single column.

        Returns
        -------
        SparseArray

        Examples
        --------
        >>> import scipy.sparse
        >>> mat = scipy.sparse.coo_matrix((4, 1))
        >>> pd.arrays.SparseArray.from_spmatrix(mat)
        [0.0, 0.0, 0.0, 0.0]
        Fill: 0.0
        IntIndex
        Indices: array([], dtype=int32)
        """
        length, ncol = data.shape

        if ncol != 1:
            raise ValueError(f"'data' must have a single column, not '{ncol}'")

        # our sparse index classes require that the positions be strictly
        # increasing. So we need to sort loc, and arr accordingly.
        data_csc = data.tocsc()
        data_csc.sort_indices()
        arr = data_csc.data
        idx = data_csc.indices

        zero = np.array(0, dtype=arr.dtype).item()
        dtype = SparseDtype(arr.dtype, zero)
        index = IntIndex(length, idx)

        return cls._simple_new(arr, index, dtype)

    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        # [FUNCTION DOC]
        # Numpy array protocol: densifies the array. dtype is the requested numpy
        # dtype (default none: inferred); copy is none, true or false.
        # With no gaps (fully stored), returns the stored values (a copy when copy
        # is true, otherwise the shared array as a non-writeable view for read-only
        # arrays). With gaps, copy=false raises a value error. Otherwise the dtype
        # defaults to the result type of the stored dtype and the fill value type
        # (not-a-time fill with datetime data is turned into a numpy not-a-time;
        # object when numpy cannot represent the combination); a dense array of the
        # fill value is allocated and the stored values are scattered into it.
        # Returns the dense numpy array.
        # KNOWN ISSUE: in the fully-stored case the requested dtype is ignored and
        # the stored values are returned as they are.
        if self.sp_index.ngaps == 0:
            # Compat for na dtype and int values.
            if copy is True:
                return np.array(self.sp_values)
            else:
                result = self.sp_values
                if self._readonly:
                    result = result.view()
                    result.flags.writeable = False
                return result

        if copy is False:
            raise ValueError(
                "Unable to avoid copy while creating an array as requested."
            )

        fill_value = self.fill_value

        if dtype is None:
            # Can NumPy represent this type?
            # If not, `np.result_type` will raise. We catch that
            # and return object.
            if self.sp_values.dtype.kind == "M":
                # However, we *do* special-case the common case of
                # a datetime64 with pandas NaT.
                if fill_value is NaT:
                    # Can't put pd.NaT in a datetime64[ns]
                    unit = np.datetime_data(self.sp_values.dtype)[0]
                    fill_value = np.datetime64("NaT", unit)  # type: ignore[call-overload]
            try:
                dtype = np.result_type(self.sp_values.dtype, type(fill_value))
            except TypeError:
                dtype = object

        out = np.full(self.shape, fill_value, dtype=dtype)
        out[self.sp_index.indices] = self.sp_values
        return out

    def __setitem__(self, key, value) -> None:
        # [FUNCTION DOC]
        # Item assignment is not supported. Raises a value error if the array is
        # read-only, otherwise always raises a type error stating that sparse
        # arrays do not support item assignment. key and value are ignored.
        if self._readonly:
            raise ValueError("Cannot modify read-only array")
        # I suppose we could allow setting of non-fill_value elements.
        # TODO(SparseArray.__setitem__): remove special cases in
        # ExtensionBlock.where
        msg = "SparseArray does not support item assignment via setitem"
        raise TypeError(msg)

    @classmethod
    def _from_sequence(
        cls, scalars, *, dtype: Dtype | None = None, copy: bool = False
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: class-method constructor used by the generic array factory.
        # scalars is a sequence; keyword-only dtype (default none) is passed on;
        # copy (default false) is accepted but ignored.
        # KNOWN ISSUE: the copy argument is not forwarded to the constructor.
        return cls(scalars, dtype=dtype)

    @classmethod
    def _from_factorized(cls, values, original) -> Self:
        # [FUNCTION DOC]
        # HELPER: rebuilds an array from factorized unique values: values is the
        # array of uniques; original supplies the (sparse) dtype. Returns a new
        # array with that dtype.
        return cls(values, dtype=original.dtype)

    def _cast_pointwise_result(self, values):
        # [FUNCTION DOC]
        # HELPER: post-processes the sequence of results of an element-wise
        # mapping. The values are converted to an object array and a better dtype
        # is inferred (including non-numeric types). If the inferred kind equals
        # this array's kind, the result is built with this array's sparse dtype and
        # accepted only if it round-trips equal to the inferred values (missing
        # values matching); otherwise (or if construction fails with a value or
        # type error) it is built with an inferred sparse dtype. A different kind
        # also gives the inferred dtype. Returns a sparse array.
        values = np.asarray(values, dtype=object)
        result = lib.maybe_convert_objects(values, convert_non_numeric=True)
        if result.dtype.kind == self.dtype.kind:
            try:
                # e.g. test_groupby_agg_extension
                res = type(self)._from_sequence(result, dtype=self.dtype)
                if ((res == result) | (isna(result) & res.isna())).all():
                    # This does not hold for e.g.
                    #  test_arith_frame_with_scalar[0-__truediv__]
                    return res
                return type(self)._from_sequence(result)
            except (ValueError, TypeError):
                return type(self)._from_sequence(result)
        else:
            # e.g. test_combine_le avoid casting bools to Sparse[float64, nan]
            return type(self)._from_sequence(result)

    # ------------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------------
    @property
    def sp_index(self) -> SparseIndex:
        # [FUNCTION DOC]
        # Property returning the sparse index object describing the positions of
        # the stored (non-fill) values and the total length.
        """
        The SparseIndex containing the location of non- ``fill_value`` points.
        """
        return self._sparse_index

    @property
    def sp_values(self) -> np.ndarray:
        # [FUNCTION DOC]
        # Property returning the numpy array of stored values (those not equal to
        # the fill value); the actual internal array, not a copy.
        """
        An ndarray containing the non- ``fill_value`` values.

        This property returns the actual data values stored in the sparse
        representation, excluding the values that are equal to the ``fill_value``.
        The result is an ndarray of the underlying values, preserving the sparse
        structure by omitting the default ``fill_value`` entries.

        See Also
        --------
        Series.sparse.to_dense : Convert a Series from sparse values to dense.
        Series.sparse.fill_value : Elements in `data` that are `fill_value` are
            not stored.
        Series.sparse.density : The percent of non- ``fill_value`` points, as decimal.

        Examples
        --------
        >>> from pandas.arrays import SparseArray
        >>> s = SparseArray([0, 0, 1, 0, 2], fill_value=0)
        >>> s.sp_values
        array([1, 2])
        """
        return self._sparse_values

    @property
    def dtype(self) -> SparseDtype:
        # [FUNCTION DOC]
        # Property returning the sparse dtype (stored subtype plus fill value).
        return self._dtype

    @property
    def fill_value(self):
        # [FUNCTION DOC]
        # Property returning the fill value from the dtype. The setter below
        # replaces the dtype with one that has the new fill value.
        """
        Elements in `data` that are `fill_value` are not stored.

        For memory savings, this should be the most common value in the array.

        See Also
        --------
        SparseDtype : Dtype for data stored in :class:`SparseArray`.
        Series.value_counts : Return a Series containing counts of unique values.
        Series.fillna : Fill NA/NaN in a Series with a specified value.

        Examples
        --------
        >>> ser = pd.Series([0, 0, 2, 2, 2], dtype="Sparse[int]")
        >>> ser.sparse.fill_value
        0
        >>> spa_dtype = pd.SparseDtype(dtype=np.int32, fill_value=2)
        >>> ser = pd.Series([0, 0, 2, 2, 2], dtype=spa_dtype)
        >>> ser.sparse.fill_value
        2
        """
        return self.dtype.fill_value

    @fill_value.setter
    def fill_value(self, value) -> None:
        # [FUNCTION DOC]
        # Setter for the fill value property: replaces the dtype with a sparse
        # dtype of the same subtype and the new fill value. The stored values and
        # index are NOT adjusted, so positions equal to the new fill value that
        # are stored stay stored and positions previously implicit now represent
        # the new fill value. Mutates this array.
        self._dtype = SparseDtype(self.dtype.subtype, value)

    @property
    def kind(self) -> SparseIndexKind:
        # [FUNCTION DOC]
        # Property: the layout of the sparse index as text, "integer" for the
        # integer index class and "block" for any other (block) index.
        """
        The kind of sparse index for this array. One of {'integer', 'block'}.
        """
        if isinstance(self.sp_index, IntIndex):
            return "integer"
        else:
            return "block"

    @property
    def _valid_sp_values(self) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: property returning the stored values with missing entries
        # removed (a new array).
        sp_vals = self.sp_values
        mask = notna(sp_vals)
        return sp_vals[mask]

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Total length of the array including implicit fill positions, taken from
        # the sparse index.
        return self.sp_index.length

    @property
    def _null_fill_value(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: property telling whether the fill value is a missing marker
        # (NaN, none or not-a-time), taken from the dtype.
        return self._dtype._is_na_fill_value

    @property
    def nbytes(self) -> int:
        # [FUNCTION DOC]
        # Property: memory in bytes of the stored values plus the sparse index.
        return self.sp_values.nbytes + self.sp_index.nbytes

    @property
    def density(self) -> float:
        # [FUNCTION DOC]
        # Property: fraction of positions that are stored (number of stored points
        # divided by the length). An empty array divides by zero (floating result
        # not-a-number or error depending on the index types).
        """
        The percent of non- ``fill_value`` points, as decimal.

        See Also
        --------
        DataFrame.sparse.from_spmatrix : Create a new DataFrame from a
            scipy sparse matrix.

        Examples
        --------
        >>> from pandas.arrays import SparseArray
        >>> s = SparseArray([0, 0, 1, 1, 1], fill_value=0)
        >>> s.density
        0.6
        """
        return self.sp_index.npoints / self.sp_index.length

    @property
    def npoints(self) -> int:
        # [FUNCTION DOC]
        # Property: the number of stored points, i.e. entries different from the
        # fill value, taken from the sparse index.
        """
        The number of non- ``fill_value`` points.

        This property returns the number of elements in the sparse series that are
        not equal to the ``fill_value``. Sparse data structures store only the
        non-``fill_value`` elements, reducing memory usage when the majority of
        values are the same.

        See Also
        --------
        Series.sparse.to_dense : Convert a Series from sparse values to dense.
        Series.sparse.fill_value : Elements in ``data`` that are ``fill_value`` are
            not stored.
        Series.sparse.density : The percent of non- ``fill_value`` points, as decimal.

        Examples
        --------
        >>> from pandas.arrays import SparseArray
        >>> s = SparseArray([0, 0, 1, 1, 1], fill_value=0)
        >>> s.npoints
        3
        """
        return self.sp_index.npoints

    # error: Return type "SparseArray" of "isna" incompatible with return type
    # "ndarray[Any, Any] | ExtensionArraySupportsAnyAll" in supertype "ExtensionArray"
    def isna(self) -> Self:  # type: ignore[override]
        # [FUNCTION DOC]
        # Returns a SPARSE boolean array (not a numpy array) marking missing
        # entries. If the fill value is missing, the result is sparse with a true
        # fill value and stores the missing flags of the stored values (same
        # index), preserving memory use. Otherwise the result has a false fill
        # value, and is built from a dense boolean array with the stored positions
        # flagged by their missingness.
        # If null fill value, we want SparseDtype[bool, true]
        # to preserve the same memory usage.
        dtype = SparseDtype(bool, self._null_fill_value)
        if self._null_fill_value:
            return type(self)._simple_new(isna(self.sp_values), self.sp_index, dtype)
        mask = np.full(len(self), False, dtype=np.bool_)
        mask[self.sp_index.indices] = isna(self.sp_values)
        return type(self)(mask, fill_value=False, dtype=dtype)

    def fillna(
        self,
        value,
        limit: int | None = None,
        copy: bool = True,
    ) -> Self:
        # [FUNCTION DOC]
        # Fills missing values. value is the replacement scalar (not validated);
        # limit (default none) must be none (value error otherwise); copy
        # (default true) is ignored. Missing stored values are replaced by value.
        # If the fill value is itself missing, the result dtype's fill value
        # becomes value (so the implicit positions are filled too); otherwise the
        # dtype is unchanged. The index is shared with this array. Returns a new
        # sparse array.
        """
        Fill missing values with `value`.

        Parameters
        ----------
        value : scalar
        limit : int, optional
            Not supported for SparseArray, must be None.
        copy: bool, default True
            Ignored for SparseArray.

        Returns
        -------
        SparseArray

        Notes
        -----
        When `value` is specified, the result's ``fill_value`` depends on
        ``self.fill_value``. The goal is to maintain low-memory use.

        If ``self.fill_value`` is NA, the result dtype will be
        ``SparseDtype(self.dtype, fill_value=value)``. This will preserve
        amount of memory used before and after filling.

        When ``self.fill_value`` is not NA, the result dtype will be
        ``self.dtype``. Again, this preserves the amount of memory used.
        """
        if limit is not None:
            raise ValueError("limit must be None")
        new_values = np.where(isna(self.sp_values), value, self.sp_values)

        if self._null_fill_value:
            # This is essentially just updating the dtype.
            new_dtype = SparseDtype(self.dtype.subtype, fill_value=value)
        else:
            new_dtype = self.dtype

        return self._simple_new(new_values, self._sparse_index, new_dtype)

    def shift(self, periods: int = 1, fill_value=None) -> Self:
        # [FUNCTION DOC]
        # Shifts entries by periods positions (default one; positive moves toward
        # the end). fill_value (default none, meaning the dtype's missing value)
        # fills vacated slots. An empty array or zero shift returns a copy.
        # If the fill value does not fit the stored subtype, the array is first
        # cast to the common numpy subtype (keeping its fill value). A block of
        # fill entries (at most the array length) is built and concatenated with
        # the retained slice. Returns a new sparse array.
        if not len(self) or periods == 0:
            return self.copy()

        if isna(fill_value):
            fill_value = self.dtype.na_value

        subtype = np.result_type(fill_value, self.dtype.subtype)

        if subtype != self.dtype.subtype:
            # just coerce up front
            arr = self.astype(SparseDtype(subtype, self.fill_value))
        else:
            arr = self

        empty = self._from_sequence(
            [fill_value] * min(abs(periods), len(self)), dtype=arr.dtype
        )

        if periods > 0:
            a = empty
            b = arr[:-periods]
        else:
            a = arr[abs(periods) :]
            b = empty
        return arr._concat_same_type([a, b])

    def _first_fill_value_loc(self):
        # [FUNCTION DOC]
        # HELPER: position of the first implicit (fill value) element. Returns
        # minus one if the array is empty or fully stored (no fill positions).
        # Returns zero if there are no stored points or the first stored point is
        # not at position zero. Otherwise finds the first gap between consecutive
        # stored positions (a sentinel gap is appended so that a fill region at the
        # tail is found) and returns the position right after the stored position
        # preceding that gap.
        """
        Get the location of the first fill value.

        Returns
        -------
        int
        """
        if len(self) == 0 or self.sp_index.npoints == len(self):
            return -1

        indices = self.sp_index.indices
        if not len(indices) or indices[0] > 0:
            return 0

        # a number larger than 1 should be appended to
        # the last in case of fill value only appears
        # in the tail of array
        diff = np.r_[np.diff(indices), 2]
        return indices[(diff > 1).argmax()] + 1

    @doc(ExtensionArray.duplicated)
    def duplicated(
        self, keep: Literal["first", "last", False] = "first"
    ) -> npt.NDArray[np.bool_]:
        # [FUNCTION DOC]
        # Marks repeated values (documentation is inherited from the base class
        # through a decorator). keep (default first) says which occurrence is not
        # marked: first, last, or false meaning all repeats are marked. The array
        # is fully densified and missing values (from the sparse missing-flag
        # array) are treated as equal. Returns a boolean numpy array.
        values = np.asarray(self)
        mask = np.asarray(self.isna())
        return algos.duplicated(values, keep=keep, mask=mask)

    def unique(self) -> Self:
        # [FUNCTION DOC]
        # Returns a new sparse array of the distinct values in order of first
        # appearance. The stored values are de-duplicated; if the array has any
        # implicit positions, the fill value is inserted at the position
        # corresponding to the first fill element (computed by de-duplicating the
        # stored values that precede it). The result is built with this dtype.
        uniques = algos.unique(self.sp_values)
        if len(self.sp_values) != len(self):
            fill_loc = self._first_fill_value_loc()
            # Inorder to align the behavior of pd.unique or
            # pd.Series.unique, we should keep the original
            # order, here we use unique again to find the
            # insertion place. Since the length of sp_values
            # is not large, maybe minor performance hurt
            # is worthwhile to the correctness.
            insert_loc = len(algos.unique(self.sp_values[:fill_loc]))
            uniques = np.insert(uniques, insert_loc, self.fill_value)
        return type(self)._from_sequence(uniques, dtype=self.dtype)

    def _values_for_factorize(self):
        # [FUNCTION DOC]
        # HELPER: hook used by hashing/factorizing. Returns the pair of the fully
        # densified numpy array and the fill value (as the missing marker).
        # Still override this for hash_pandas_object
        return np.asarray(self), self.fill_value

    def factorize(
        self,
        use_na_sentinel: bool = True,
    ) -> tuple[np.ndarray, SparseArray]:
        # [FUNCTION DOC]
        # Encodes values as integer codes plus the distinct values. use_na_sentinel
        # (default true) gives missing values code minus one and leaves them out of
        # the uniques; false gives them a code of their own. The array is
        # densified and the generic factorizer is used (no sparse efficiency).
        # Returns the pair (dense integer code array, sparse array of uniques with
        # this dtype).
        # Currently, ExtensionArray.factorize -> Tuple[ndarray, EA]
        # The sparsity on this is backwards from what Sparse would want. Want
        # ExtensionArray.factorize -> Tuple[EA, EA]
        # Given that we have to return a dense array of codes, why bother
        # implementing an efficient factorize?
        codes, uniques = algos.factorize(
            np.asarray(self), use_na_sentinel=use_na_sentinel
        )
        uniques_sp = SparseArray(uniques, dtype=self.dtype)
        return codes, uniques_sp

    def value_counts(self, dropna: bool = True) -> Series:
        # [FUNCTION DOC]
        # Counts occurrences of each distinct value. dropna (default true) leaves
        # out missing values. Counts are computed on the stored values only; the
        # number of implicit positions is then added to the count of the fill
        # value (when the fill value is already among the keys), or inserted as a
        # new first entry (fill value, number of gaps) when it is absent. This is
        # skipped when the fill value is missing and dropna is true. Returns a
        # series of counts indexed by value.
        """
        Returns a Series containing counts of unique values.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of NaN, even if NaN is in sp_values.

        Returns
        -------
        counts : Series
        """
        from pandas import (
            Index,
            Series,
        )

        keys, counts, _ = algos.value_counts_arraylike(self.sp_values, dropna=dropna)
        fcounts = self.sp_index.ngaps
        if fcounts > 0 and (not self._null_fill_value or not dropna):
            mask = isna(keys) if self._null_fill_value else keys == self.fill_value
            if mask.any():
                counts[mask] += fcounts
            else:
                # error: Argument 1 to "insert" has incompatible type "Union[
                # ExtensionArray,ndarray[Any, Any]]"; expected "Union[
                # _SupportsArray[dtype[Any]], Sequence[_SupportsArray[dtype
                # [Any]]], Sequence[Sequence[_SupportsArray[dtype[Any]]]],
                # Sequence[Sequence[Sequence[_SupportsArray[dtype[Any]]]]], Sequence
                # [Sequence[Sequence[Sequence[_SupportsArray[dtype[Any]]]]]]]"
                keys = np.insert(keys, 0, self.fill_value)  # type: ignore[arg-type]
                counts = np.insert(counts, 0, fcounts)

        if not isinstance(keys, ABCIndex):
            index = Index(keys, copy=False)
        else:
            index = keys
        return Series(counts, index=index, copy=False)

    # --------
    # Indexing
    # --------
    @overload
    def __getitem__(self, key: ScalarIndexer) -> Any:
        # [FUNCTION DOC]
        # Typing stub only: scalar position gives a scalar. See implementation.
        ...

    @overload
    def __getitem__(
        self,
        key: SequenceIndexer | tuple[int | EllipsisType, ...],
    ) -> Self:
        # [FUNCTION DOC]
        # Typing stub only: sequence, slice, mask or tuple key gives an array. See
        # implementation.
        ...

    def __getitem__(
        self,
        key: PositionalIndexer | tuple[int | EllipsisType, ...],
    ) -> Self | Any:
        # [FUNCTION DOC]
        # Positional indexing. key is an integer, slice, ellipsis-containing
        # tuple, list-like of positions, or boolean mask (sparse masks too).
        # Tuples are unpacked of ellipses (a bare ellipsis raises a value error).
        # An integer returns the scalar (fill value for implicit positions). A
        # full slice returns a new array sharing the stored values and index and
        # the read-only flag. A unit-step slice is done without densifying: the
        # stored points inside the range are kept, their positions shifted to the
        # new origin, and an index of the same kind is built for the new length. A
        # stepped slice is converted to explicit positions and goes through take.
        # A remaining tuple form is applied to the dense array and the result
        # re-sparsified with the same kind. A non-list-like key raises an index
        # error. Sparse boolean masks use fast paths by their fill value (missing
        # fill: positions of true values; false fill: positions of stored points;
        # true fill: all positions except the stored ones, which are false); other
        # keys are validated as array indexers, boolean masks and integer arrays
        # go through take; anything else without a length raises a value error.
        if isinstance(key, tuple):
            key = unpack_tuple_and_ellipses(key)
            if key is ...:
                raise ValueError("Cannot slice with Ellipsis")

        if is_integer(key):
            return self._get_val_at(key)
        elif isinstance(key, tuple):
            data_slice = self.to_dense()[key]
        elif isinstance(key, slice):
            if key == slice(None):
                # to ensure arr[:] (used by view()) does not make a copy
                result = type(self)._simple_new(
                    self.sp_values, self.sp_index, self.dtype
                )
                result._readonly = self._readonly
                return result
            # Avoid densifying when handling contiguous slices
            if key.step is None or key.step == 1:
                start = 0 if key.start is None else key.start
                if start < 0:
                    start += len(self)

                end = len(self) if key.stop is None else key.stop
                if end < 0:
                    end += len(self)

                indices = self.sp_index.indices
                keep_inds = np.flatnonzero((indices >= start) & (indices < end))
                sp_vals = self.sp_values[keep_inds]

                sp_index = indices[keep_inds].copy()

                # If we've sliced to not include the start of the array, all our indices
                # should be shifted. NB: here we are careful to also not shift by a
                # negative value for a case like [0, 1][-100:] where the start index
                # should be treated like 0
                if start > 0:
                    sp_index -= start

                # Length of our result should match applying this slice to a range
                # of the length of our original array
                new_len = len(range(len(self))[key])
                new_sp_index = make_sparse_index(new_len, sp_index, self.kind)
                return type(self)._simple_new(sp_vals, new_sp_index, self.dtype)
            else:
                indices = np.arange(len(self), dtype=np.int32)[key]
                return self.take(indices)

        elif not is_list_like(key):
            # e.g. "foo" or 2.5
            # exception message copied from numpy
            raise IndexError(
                r"only integers, slices (`:`), ellipsis (`...`), numpy.newaxis "
                r"(`None`) and integer or boolean arrays are valid indices"
            )

        else:
            if isinstance(key, SparseArray):
                # NOTE: If we guarantee that SparseDType(bool)
                # has only fill_value - true, false or nan
                # (see GH PR 44955)
                # we can apply mask very fast:
                if is_bool_dtype(key):
                    if isna(key.fill_value):
                        return self.take(key.sp_index.indices[key.sp_values])
                    if not key.fill_value:
                        return self.take(key.sp_index.indices)
                    n = len(self)
                    mask = np.full(n, True, dtype=np.bool_)
                    mask[key.sp_index.indices] = False
                    return self.take(np.arange(n)[mask])
                else:
                    key = np.asarray(key)

            key = check_array_indexer(self, key)

            if com.is_bool_indexer(key):
                # mypy doesn't know we have an array here
                key = cast(np.ndarray, key)
                return self.take(np.arange(len(key), dtype=np.int32)[key])
            elif hasattr(key, "__len__"):
                return self.take(key)
            else:
                raise ValueError(f"Cannot slice with '{key}'")

        return type(self)(data_slice, kind=self.kind)

    def _get_val_at(self, loc):
        # [FUNCTION DOC]
        # HELPER: scalar lookup. loc is an integer position, negative counting from
        # the end; out-of-range positions raise an index error (via the location
        # validator). Returns the fill value if the position is not stored,
        # otherwise the stored value boxed as a pandas scalar for datetime-like
        # types.
        loc = validate_insert_loc(loc, len(self))

        sp_loc = self.sp_index.lookup(loc)
        if sp_loc == -1:
            return self.fill_value
        else:
            val = self.sp_values[sp_loc]
            val = maybe_box_datetimelike(val, self.sp_values.dtype)
            return val

    def take(self, indices, *, allow_fill: bool = False, fill_value=None) -> Self:
        # [FUNCTION DOC]
        # Positional take. indices is an integer sequence (a scalar raises a value
        # error); keyword-only allow_fill (default false: negatives count from the
        # end; true: minus one marks positions to fill) and fill_value (default
        # none, the dtype's missing value). Indices are converted to 32-bit
        # integers. Empty indices give an empty array of this dtype. With fill, the
        # dense taken values are computed by the fill helper and re-sparsified
        # with this fill value and kind; without fill, the fast sparse path is
        # used. Out-of-range positions raise an index error; invalid negative
        # values raise a value error when filling.
        if is_scalar(indices):
            raise ValueError(f"'indices' must be an array, not a scalar '{indices}'.")
        indices = np.asarray(indices, dtype=np.int32)

        dtype = None
        if indices.size == 0:
            result = np.array([], dtype="object")
            dtype = self.dtype
        elif allow_fill:
            result = self._take_with_fill(indices, fill_value=fill_value)
        else:
            return self._take_without_fill(indices)

        return type(self)(
            result, fill_value=self.fill_value, kind=self.kind, dtype=dtype
        )

    def _take_with_fill(self, indices, fill_value=None) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: take with fill semantics, returning a DENSE numpy array. indices
        # is a 32-bit integer array where minus one means "insert a fill"; fill_value
        # (default none, the dtype's missing value) is the value used there.
        # Raises a value error for indices below minus one, an index error for
        # indices at or beyond the length, and (for an empty source) allows only
        # all-minus-one requests, giving an array of the fill value. Positions
        # are looked up in the sparse index; the taken array starts from the
        # stored values (or from the fill values for an all-gap source) and is
        # widened to a common type when the original fill value or the new fill
        # value does not fit, then both kinds of fill are written. Returns the
        # dense array.
        if fill_value is None:
            fill_value = self.dtype.na_value

        if indices.min() < -1:
            raise ValueError(
                "Invalid value in 'indices'. Must be between -1 "
                "and the length of the array."
            )

        if indices.max() >= len(self):
            raise IndexError("out of bounds value in 'indices'.")

        if len(self) == 0:
            # Empty... Allow taking only if all empty
            if (indices == -1).all():
                dtype = np.result_type(self.sp_values, type(fill_value))
                taken = np.empty_like(indices, dtype=dtype)
                taken.fill(fill_value)
                return taken
            else:
                raise IndexError("cannot do a non-empty take from an empty axes.")

        # sp_indexer may be -1 for two reasons
        # 1.) we took for an index of -1 (new)
        # 2.) we took a value that was self.fill_value (old)
        sp_indexer = self.sp_index.lookup_array(indices)
        new_fill_indices = indices == -1
        old_fill_indices = (sp_indexer == -1) & ~new_fill_indices

        if self.sp_index.npoints == 0 and old_fill_indices.all():
            # We've looked up all valid points on an all-sparse array.
            taken = np.full(
                sp_indexer.shape, fill_value=self.fill_value, dtype=self.dtype.subtype
            )

        elif self.sp_index.npoints == 0:
            # Use the old fill_value unless we took for an index of -1
            _dtype = np.result_type(self.dtype.subtype, type(fill_value))
            taken = np.full(sp_indexer.shape, fill_value=fill_value, dtype=_dtype)
            taken[old_fill_indices] = self.fill_value
        else:
            taken = self.sp_values.take(sp_indexer)

            # Fill in two steps.
            # Old fill values
            # New fill values
            # potentially coercing to a new dtype at each stage.

            m0 = sp_indexer[old_fill_indices] < 0
            m1 = sp_indexer[new_fill_indices] < 0

            result_type = taken.dtype

            if m0.any():
                result_type = np.result_type(result_type, type(self.fill_value))
                taken = taken.astype(result_type)
                taken[old_fill_indices] = self.fill_value

            if m1.any():
                result_type = np.result_type(result_type, type(fill_value))
                taken = taken.astype(result_type)
                taken[new_fill_indices] = fill_value

        return taken

    def _take_without_fill(self, indices) -> Self:
        # [FUNCTION DOC]
        # HELPER: take with plain numpy semantics, staying sparse. indices is a
        # non-empty 32-bit integer array; negatives count from the end. Raises an
        # index error when any position is out of bounds (a distinct message for
        # an empty source). The positions are looked up in the sparse index; the
        # stored values at found positions are kept and a new index of the same
        # kind is built from the output positions that hit stored points. Returns
        # a new sparse array with this dtype.
        to_shift = indices < 0

        n = len(self)

        if (indices.max() >= n) or (indices.min() < -n):
            if n == 0:
                raise IndexError("cannot do a non-empty take from an empty axes.")
            raise IndexError("out of bounds value in 'indices'.")

        if to_shift.any():
            indices = indices.copy()
            indices[to_shift] += n

        sp_indexer = self.sp_index.lookup_array(indices)
        value_mask = sp_indexer != -1
        new_sp_values = self.sp_values[sp_indexer[value_mask]]

        value_indices = np.flatnonzero(value_mask).astype(np.int32, copy=False)

        new_sp_index = make_sparse_index(len(indices), value_indices, kind=self.kind)
        return type(self)._simple_new(new_sp_values, new_sp_index, dtype=self.dtype)

    def searchsorted(
        self,
        v: ArrayLike | object,
        side: Literal["left", "right"] = "left",
        sorter: NumpySorter | None = None,
    ) -> npt.NDArray[np.intp] | np.intp:
        # [FUNCTION DOC]
        # Binary search for insertion points. v is a scalar or array-like; side is
        # left or right (default left); sorter (default none) is an integer array
        # sorting the data. Emits a performance warning (if the global performance
        # warnings option is on) because the array is fully densified to the stored
        # subtype. Returns an integer array, or a single integer for a scalar.
        if get_option("performance_warnings"):
            msg = "searchsorted requires high memory usage."
            warnings.warn(msg, PerformanceWarning, stacklevel=find_stack_level())
        v = np.asarray(v)
        return np.asarray(self, dtype=self.dtype.subtype).searchsorted(v, side, sorter)

    def copy(self) -> Self:
        # [FUNCTION DOC]
        # Returns a new array with a copy of the stored values; the (immutable)
        # sparse index object and dtype are shared with this array.
        values = self.sp_values.copy()
        return self._simple_new(values, self.sp_index, self.dtype)

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[Self]) -> Self:
        # [FUNCTION DOC]
        # HELPER: class-method used by concatenation. to_concat is a sequence of
        # sparse arrays; the result takes the fill value and index kind of the
        # first. Integer kind: positions are shifted by the running length and
        # joined into one integer index. Block kind: each array is converted to a
        # block index, block starts are shifted by the running length and all
        # blocks are joined (adjacent blocks across arrays are not merged, so the
        # result may have more blocks than a fresh construction). Stored values are
        # joined. Returns the new array via the public constructor.
        # KNOWN ISSUE: the first element is read before the emptiness check, so an
        # empty sequence fails with an index error and the later default of
        # "integer" for an empty sequence is unreachable; also a source TODO notes
        # the shifted integer positions can wrap around for very large totals.
        fill_value = to_concat[0].fill_value

        values = []
        length = 0

        if to_concat:
            sp_kind = to_concat[0].kind
        else:
            sp_kind = "integer"

        sp_index: SparseIndex
        if sp_kind == "integer":
            indices = []

            for arr in to_concat:
                int_idx = arr.sp_index.indices.copy()
                int_idx += length  # TODO: wraparound
                length += arr.sp_index.length

                values.append(arr.sp_values)
                indices.append(int_idx)

            data = np.concatenate(values)
            indices_arr = np.concatenate(indices)
            sp_index = IntIndex(length, indices_arr)

        else:
            # when concatenating block indices, we don't claim that you'll
            # get an identical index as concatenating the values and then
            # creating a new index. We don't want to spend the time trying
            # to merge blocks across arrays in `to_concat`, so the resulting
            # BlockIndex may have more blocks.
            blengths = []
            blocs = []

            for arr in to_concat:
                block_idx = arr.sp_index.to_block_index()

                values.append(arr.sp_values)
                blocs.append(block_idx.blocs.copy() + length)
                blengths.append(block_idx.blengths)
                length += arr.sp_index.length

            data = np.concatenate(values)
            blocs_arr = np.concatenate(blocs)
            blengths_arr = np.concatenate(blengths)

            sp_index = BlockIndex(length, blocs_arr, blengths_arr)

        return cls(data, sparse_index=sp_index, fill_value=fill_value)

    def astype(self, dtype: AstypeArg | None = None, copy: bool = True):
        # [FUNCTION DOC]
        # Casts the array. dtype (default none) is a numpy dtype or a sparse dtype;
        # copy (default true) forces copying.
        # Same dtype: copy or self. A non-sparse target: the array is densified,
        # datetime-like data is wrapped, and the generic array cast is applied; the
        # result is a DENSE array of the target type (the docstring's claim that the
        # output is always sparse is not accurate). A sparse target: the dtype is
        # merged with the current one (a missing subtype or fill value is taken from
        # the current dtype), the stored values are cast to the new subtype and the
        # index is kept; the fill value comes from the merged dtype. Returns the new
        # array.
        # KNOWN ISSUE: the docstring says the output is always a SparseArray, but
        # casting to a non-sparse dtype returns a dense array.
        """
        Change the dtype of a SparseArray.

        The output will always be a SparseArray. To convert to a dense
        ndarray with a certain dtype, use :meth:`numpy.asarray`.

        Parameters
        ----------
        dtype : np.dtype or ExtensionDtype
            For SparseDtype, this changes the dtype of
            ``self.sp_values`` and the ``self.fill_value``.

            For other dtypes, this only changes the dtype of
            ``self.sp_values``.

        copy : bool, default True
            Whether to ensure a copy is made, even if not necessary.

        Returns
        -------
        SparseArray

        Examples
        --------
        >>> arr = pd.arrays.SparseArray([0, 0, 1, 2])
        >>> arr
        [0, 0, 1, 2]
        Fill: 0
        IntIndex
        Indices: array([2, 3], dtype=int32)

        >>> arr.astype(pd.SparseDtype(np.dtype("int32")))
        [0, 0, 1, 2]
        Fill: 0
        IntIndex
        Indices: array([2, 3], dtype=int32)

        Using a NumPy dtype with a different kind (e.g. float) will coerce
        just ``self.sp_values``.

        >>> arr.astype(pd.SparseDtype(np.dtype("float64")))
        ... # doctest: +NORMALIZE_WHITESPACE
        [nan, nan, 1.0, 2.0]
        Fill: nan
        IntIndex
        Indices: array([2, 3], dtype=int32)

        Using a SparseDtype, you can also change the fill value as well.

        >>> arr.astype(pd.SparseDtype("float64", fill_value=0.0))
        ... # doctest: +NORMALIZE_WHITESPACE
        [0.0, 0.0, 1.0, 2.0]
        Fill: 0.0
        IntIndex
        Indices: array([2, 3], dtype=int32)
        """
        if dtype == self._dtype:
            if not copy:
                return self
            else:
                return self.copy()

        future_dtype = pandas_dtype(dtype)
        if not isinstance(future_dtype, SparseDtype):
            # GH#34457
            values = np.asarray(self)
            values = ensure_wrapped_if_datetimelike(values)
            return astype_array(values, dtype=future_dtype, copy=False)

        dtype = self.dtype.update_dtype(dtype)
        subtype = pandas_dtype(dtype._subtype_with_str)
        subtype = cast(np.dtype, subtype)  # ensured by update_dtype
        values = ensure_wrapped_if_datetimelike(self.sp_values)
        sp_values = astype_array(values, subtype, copy=copy)
        sp_values = np.asarray(sp_values)

        return self._simple_new(sp_values, self.sp_index, dtype)

    def map(self, mapper, na_action: Literal["ignore"] | None = None) -> Self:
        # [FUNCTION DOC]
        # Maps the values through a function or mapping, keeping the sparse
        # structure. mapper is a function, mapping or series; na_action (default
        # none; or "ignore") skips mapping a missing fill value. The new fill value
        # is the mapped old fill value (not mapped when na_action is "ignore" and
        # the fill value is missing). Every stored value is mapped (missing keys of
        # a mapping give none); if a mapped stored value equals the new fill value
        # (identity or equality) a value error is raised since it would break the
        # sparse representation. Returns a new array with the same index and the
        # mapped stored values.
        # KNOWN ISSUE: the na_action setting only affects the fill value; stored
        # missing values are always passed to the mapper.
        """
        Map categories using an input mapping or function.

        Parameters
        ----------
        mapper : dict, Series, callable
            The correspondence from old values to new.
        na_action : {None, 'ignore'}, default None
            If 'ignore', propagate NA values, without passing them to the
            mapping correspondence.

        Returns
        -------
        SparseArray
            The output array will have the same density as the input.
            The output fill value will be the result of applying the
            mapping to ``self.fill_value``

        Examples
        --------
        >>> arr = pd.arrays.SparseArray([0, 1, 2])
        >>> arr.map(lambda x: x + 10)
        [10, 11, 12]
        Fill: 10
        IntIndex
        Indices: array([1, 2], dtype=int32)

        >>> arr.map({0: 10, 1: 11, 2: 12})
        [10, 11, 12]
        Fill: 10
        IntIndex
        Indices: array([1, 2], dtype=int32)

        >>> arr.map(pd.Series([10, 11, 12], index=[0, 1, 2]))
        [10, 11, 12]
        Fill: 10
        IntIndex
        Indices: array([1, 2], dtype=int32)
        """
        is_map = isinstance(mapper, (abc.Mapping, ABCSeries))

        fill_val = self.fill_value

        if na_action is None or notna(fill_val):
            fill_val = mapper.get(fill_val, fill_val) if is_map else mapper(fill_val)

        def func(sp_val):
            # [FUNCTION DOC]
            # HELPER: nested function of the mapping method. Maps one stored value
            # with the mapper (dictionary-style lookup with none for missing keys,
            # or a call). Raises a value error if the mapped value is the same
            # object as, or equal to, the new fill value. Returns the mapped value.
            new_sp_val = mapper.get(sp_val, None) if is_map else mapper(sp_val)
            # check identity and equality because nans are not equal to each other
            if new_sp_val is fill_val or new_sp_val == fill_val:
                msg = "fill value in the sparse values not supported"
                raise ValueError(msg)
            return new_sp_val

        sp_values = [func(x) for x in self.sp_values]

        return type(self)(sp_values, sparse_index=self.sp_index, fill_value=fill_val)

    def to_dense(self) -> np.ndarray:
        # [FUNCTION DOC]
        # Returns the dense numpy array of the stored subtype, with the fill value
        # written at every implicit position (a new array, except that a fully
        # stored array may return its stored values directly).
        """
        Convert SparseArray to a NumPy array.

        Returns
        -------
        arr : NumPy array
        """
        return np.asarray(self, dtype=self.sp_values.dtype)

    def _where(self, mask, value):
        # [FUNCTION DOC]
        # HELPER: conditional replacement hook. mask is a boolean array (true keeps
        # this array's element); value is the replacement. The work is done densely
        # with numpy and the result is re-sparsified with a sparse dtype made of the
        # result dtype and this array's fill value, so the subtype may change (e.g.
        # integer to float).
        # NB: may not preserve dtype, e.g. result may be Sparse[float64]
        #  while self is Sparse[int64]
        naive_implementation = np.where(mask, self, value)
        dtype = SparseDtype(naive_implementation.dtype, fill_value=self.fill_value)
        result = type(self)._from_sequence(naive_implementation, dtype=dtype)
        return result

    # ------------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------------
    def __setstate__(self, state) -> None:
        # [FUNCTION DOC]
        # Unpickling support. state is either a tuple (format from versions before
        # 0.24: a numpy array state for the stored values plus a pair of fill value
        # and sparse index), in which case the three private attributes are rebuilt
        # from it, or a dictionary of attributes that is merged into the instance
        # dictionary. Mutates this instance.
        """Necessary for making this object picklable"""
        if isinstance(state, tuple):
            # Compat for pandas < 0.24.0
            nd_state, (fill_value, sp_index) = state
            sparse_values = np.array([])
            sparse_values.__setstate__(nd_state)

            self._sparse_values = sparse_values
            self._sparse_index = sp_index
            self._dtype = SparseDtype(sparse_values.dtype, fill_value)
        else:
            self.__dict__.update(state)

    def nonzero(self) -> tuple[npt.NDArray[np.int32]]:
        # [FUNCTION DOC]
        # Positions of non-zero elements (numpy-style, a one-element tuple of an
        # integer array). If the fill value is zero, these are the stored
        # positions; otherwise the stored positions whose value is not zero.
        # KNOWN ISSUE: when the fill value is not zero, implicit positions (which
        # hold a non-zero fill value) are omitted from the result.
        if self.fill_value == 0:
            return (self.sp_index.indices,)
        else:
            return (self.sp_index.indices[self.sp_values != 0],)

    # ------------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------------

    def _reduce(
        self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs
    ):
        # [FUNCTION DOC]
        # Generic reduction entry point. name is the name of a method of this class
        # (otherwise a type error naming the dtype is raised); skipna (keyword-only,
        # default true); keepdims (default false) returns a one-element sparse
        # array; extra keywords go to the method.
        # The method is called on this array, or, when skipna is false, on a copy
        # with missing values removed. Returns the scalar result or the wrapped
        # one-element array.
        # KNOWN ISSUE: the skipna handling looks inverted: missing values are
        # dropped when skipna is false, and left in place when it is true (relying
        # on the called method's own default handling); skipna is never forwarded
        # to the method.
        method = getattr(self, name, None)

        if method is None:
            raise TypeError(f"cannot perform {name} with type {self.dtype}")

        if skipna:
            arr = self
        else:
            arr = self.dropna()

        result = getattr(arr, name)(**kwargs)

        if keepdims:
            return type(self)([result], dtype=self.dtype)
        else:
            return result

    def all(self, axis=None, *args, **kwargs):
        # [FUNCTION DOC]
        # True if all elements are truthy. axis (default none) and extra
        # positional/keyword arguments exist for numpy compatibility (extra ones
        # validated; axis unused). If there are implicit positions and the fill
        # value is not truthy, returns false; otherwise reduces the stored values.
        # Returns a numpy boolean (or a plain false).
        """
        Tests whether all elements evaluate True

        Returns
        -------
        all : bool

        See Also
        --------
        numpy.all
        """
        nv.validate_all(args, kwargs)

        values = self.sp_values

        if len(values) != len(self) and not np.all(self.fill_value):
            return False

        return values.all()

    def any(self, axis: AxisInt = 0, *args, **kwargs) -> bool:
        # [FUNCTION DOC]
        # True if any element is truthy. axis (default zero) and extra arguments are
        # for numpy compatibility (extra ones validated; axis unused). If there are
        # implicit positions and the fill value is truthy, returns true; otherwise
        # reduces the stored values. Returns a plain boolean.
        """
        Tests whether at least one of elements evaluate True

        Returns
        -------
        any : bool

        See Also
        --------
        numpy.any
        """
        nv.validate_any(args, kwargs)

        values = self.sp_values

        if len(values) != len(self) and np.any(self.fill_value):
            return True

        return values.any().item()

    def sum(
        self,
        axis: AxisInt = 0,
        min_count: int = 0,
        skipna: bool = True,
        *args,
        **kwargs,
    ) -> Scalar:
        # [FUNCTION DOC]
        # Sum of values. axis (default zero) is unused; min_count (default zero) is
        # the minimum number of valid values required, else the missing value of the
        # subtype is returned; skipna (default true) controls whether missing data
        # makes the result missing; extra arguments are validated for numpy
        # compatibility.
        # The stored non-missing values are summed. With a non-missing fill value the
        # fill value times the number of implicit positions is added (and those
        # positions count toward the minimum). If the fill value is missing the
        # minimum count applies to the stored valid values only.
        # KNOWN ISSUE: the "has missing data" test is true for a non-missing fill
        # value with implicit positions, so with skipna false such an array returns
        # the missing value although it holds no missing entries; and with skipna
        # false, missing stored values are still ignored. The docstring also omits
        # the skipna parameter.
        """
        Sum of non-NA/null values

        Parameters
        ----------
        axis : int, default 0
            Not Used. NumPy compatibility.
        min_count : int, default 0
            The required number of valid values to perform the summation. If fewer
            than ``min_count`` valid values are present, the result will be the missing
            value indicator for subarray type.
        *args, **kwargs
            Not Used. NumPy compatibility.

        Returns
        -------
        scalar
        """
        nv.validate_sum(args, kwargs)
        valid_vals = self._valid_sp_values
        sp_sum = valid_vals.sum()
        has_na = self.sp_index.ngaps > 0 and not self._null_fill_value

        if has_na and not skipna:
            return na_value_for_dtype(self.dtype.subtype, compat=False)

        if self._null_fill_value:
            if check_below_min_count(valid_vals.shape, None, min_count):
                return na_value_for_dtype(self.dtype.subtype, compat=False)
            return sp_sum
        else:
            nsparse = self.sp_index.ngaps
            if check_below_min_count(valid_vals.shape, None, min_count - nsparse):
                return na_value_for_dtype(self.dtype.subtype, compat=False)
            return sp_sum + self.fill_value * nsparse

    def cumsum(self, axis: AxisInt = 0, *args, **kwargs) -> SparseArray:
        # [FUNCTION DOC]
        # Cumulative sum. axis (default zero; none means flattened) must be less
        # than the number of dimensions (value error otherwise); extra arguments are
        # validated for numpy compatibility. If the fill value is not missing, the
        # array is densified and re-sparsified with a NaN fill value, then summed. If
        # the fill value is missing, the stored values are accumulated and the same
        # index and fill value kept. Returns a new sparse array.
        """
        Cumulative sum of non-NA/null values.

        When performing the cumulative summation, any non-NA/null values will
        be skipped. The resulting SparseArray will preserve the locations of
        NaN values, but the fill value will be `np.nan` regardless.

        Parameters
        ----------
        axis : int or None
            Axis over which to perform the cumulative summation. If None,
            perform cumulative summation over flattened array.

        Returns
        -------
        cumsum : SparseArray
        """
        nv.validate_cumsum(args, kwargs)

        if axis is not None and axis >= self.ndim:  # Mimic ndarray behaviour.
            raise ValueError(f"axis(={axis}) out of bounds")

        if not self._null_fill_value:
            return SparseArray(self.to_dense(), fill_value=np.nan).cumsum()

        return SparseArray(
            self.sp_values.cumsum(),
            sparse_index=self.sp_index,
            fill_value=self.fill_value,
        )

    def mean(self, axis: Axis = 0, *args, **kwargs):
        # [FUNCTION DOC]
        # Mean of the non-missing values. axis (default zero) is unused; extra
        # arguments validated for numpy compatibility. With a missing fill value,
        # the stored valid values are averaged; otherwise the implicit fill positions
        # contribute the fill value and count. Returns a float scalar; an empty
        # valid set gives a division result (not-a-number with a warning, or an
        # error).
        """
        Mean of non-NA/null values

        Returns
        -------
        mean : float
        """
        nv.validate_mean(args, kwargs)
        valid_vals = self._valid_sp_values
        sp_sum = valid_vals.sum()
        ct = len(valid_vals)

        if self._null_fill_value:
            return sp_sum / ct
        else:
            nsparse = self.sp_index.ngaps
            return (sp_sum + self.fill_value * nsparse) / (ct + nsparse)

    def max(self, *, axis: AxisInt | None = None, skipna: bool = True):
        # [FUNCTION DOC]
        # Maximum value. Keyword-only axis (default none; validated: only none or
        # zero allowed, else error) and skipna (default true). Delegates to the
        # shared min/max helper. Returns the scalar or the missing value.
        """
        Max of array values, ignoring NA values if specified.

        Parameters
        ----------
        axis : int, default 0
            Not Used. NumPy compatibility.
        skipna : bool, default True
            Whether to ignore NA values.

        Returns
        -------
        scalar
        """
        nv.validate_minmax_axis(axis, self.ndim)
        return self._min_max("max", skipna=skipna)

    def min(self, *, axis: AxisInt | None = None, skipna: bool = True):
        # [FUNCTION DOC]
        # Minimum value; same parameters and behavior as the maximum.
        """
        Min of array values, ignoring NA values if specified.

        Parameters
        ----------
        axis : int, default 0
            Not Used. NumPy compatibility.
        skipna : bool, default True
            Whether to ignore NA values.

        Returns
        -------
        scalar
        """
        nv.validate_minmax_axis(axis, self.ndim)
        return self._min_max("min", skipna=skipna)

    def _min_max(self, kind: Literal["min", "max"], skipna: bool) -> Scalar:
        # [FUNCTION DOC]
        # HELPER: shared implementation of minimum and maximum. kind is "min" or
        # "max"; skipna says whether missing data may be ignored.
        # If there are stored valid values: take their extreme; if there is a
        # non-missing fill value with implicit positions, return the more extreme of
        # that and the stored extreme; else with skipna or no implicit positions
        # return the stored extreme; otherwise (implicit missing positions and not
        # skipping) return the missing value. With no stored valid values: return
        # the fill value if it is non-missing and present at implicit positions,
        # else the missing value of the subtype.
        """
        Min/max of non-NA/null values

        Parameters
        ----------
        kind : {"min", "max"}
        skipna : bool

        Returns
        -------
        scalar
        """
        valid_vals = self._valid_sp_values
        has_nonnull_fill_vals = not self._null_fill_value and self.sp_index.ngaps > 0

        if len(valid_vals) > 0:
            sp_min_max = getattr(valid_vals, kind)()

            # If a non-null fill value is currently present, it might be the min/max
            if has_nonnull_fill_vals:
                func = max if kind == "max" else min
                return func(sp_min_max, self.fill_value)
            elif skipna:
                return sp_min_max
            elif self.sp_index.ngaps == 0:
                # No NAs present
                return sp_min_max
            else:
                return na_value_for_dtype(self.dtype.subtype, compat=False)
        elif has_nonnull_fill_vals:
            return self.fill_value
        else:
            return na_value_for_dtype(self.dtype.subtype, compat=False)

    def _argmin_argmax(self, kind: Literal["argmin", "argmax"]) -> int:
        # [FUNCTION DOC]
        # HELPER: shared implementation of position-of-minimum/maximum. kind is
        # "argmin" or "argmax". Finds the best stored non-missing value and its
        # position in the full array. If the fill value is missing, that is the
        # answer. Otherwise, if the stored candidate beats the fill value strictly
        # it is returned; if not, the position of the first implicit element (the
        # fill value) is returned, or the candidate when no implicit elements
        # exist. Returns an integer position. When there are no stored non-missing
        # values the underlying numpy call fails with a value error.
        values = self._sparse_values
        index = self._sparse_index.indices
        mask = np.asarray(isna(values))
        func = np.argmax if kind == "argmax" else np.argmin

        idx = np.arange(values.shape[0])
        non_nans = values[~mask]
        non_nan_idx = idx[~mask]

        _candidate = non_nan_idx[func(non_nans)]
        candidate = index[_candidate]

        if isna(self.fill_value):
            return candidate
        if kind == "argmin" and self[candidate] < self.fill_value:
            return candidate
        if kind == "argmax" and self[candidate] > self.fill_value:
            return candidate
        _loc = self._first_fill_value_loc()
        if _loc == -1:
            # fill_value doesn't exist
            return candidate
        else:
            return _loc

    def argmax(self, skipna: bool = True) -> int:
        # [FUNCTION DOC]
        # Position of the maximum. skipna (default true; must be a boolean) says
        # whether missing values may be ignored; with false and any missing value a
        # value error is raised. Returns the integer position from the shared
        # helper.
        validate_bool_kwarg(skipna, "skipna")
        if not skipna and self._hasna:
            raise ValueError("Encountered an NA value with skipna=False")
        return self._argmin_argmax("argmax")

    def argmin(self, skipna: bool = True) -> int:
        # [FUNCTION DOC]
        # Position of the minimum; same parameters and behavior as the argmax.
        validate_bool_kwarg(skipna, "skipna")
        if not skipna and self._hasna:
            raise ValueError("Encountered an NA value with skipna=False")
        return self._argmin_argmax("argmin")

    # ------------------------------------------------------------------------
    # Ufuncs
    # ------------------------------------------------------------------------

    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        # [FUNCTION DOC]
        # Numpy universal-function protocol. ufunc is the function, method the call
        # style, inputs the operands, kwargs the options (possibly output arrays).
        # Returns the not-implemented marker if any operand or output is not a
        # sparse array, a numpy array or a number. Binary operators are first
        # dispatched to the dunder methods; calls with outputs and reductions have
        # dedicated dispatchers. A single-input call is applied only to the stored
        # values and to the fill value, keeping the sparse index: multiple outputs
        # give a tuple of sparse arrays; a reduce returns the stored-values result
        # directly; otherwise a new sparse array with a sparse dtype of the result
        # dtype and the transformed fill value. Calls with several operands densify
        # all operands, apply the ufunc, and return the output argument if one was
        # given, a tuple of sparse arrays for multiple outputs, nothing for the
        # at method, or a new sparse array built from the dense result.
        out = kwargs.get("out", ())

        for x in inputs + out:
            if not isinstance(x, (*self._HANDLED_TYPES, SparseArray)):
                return NotImplemented

        # for binary ops, use our custom dunder methods
        result = arraylike.maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )
        if result is not NotImplemented:
            return result

        if "out" in kwargs:
            # e.g. tests.arrays.sparse.test_arithmetics.test_ndarray_inplace
            res = arraylike.dispatch_ufunc_with_out(
                self, ufunc, method, *inputs, **kwargs
            )
            return res

        if method == "reduce":
            result = arraylike.dispatch_reduction_ufunc(
                self, ufunc, method, *inputs, **kwargs
            )
            if result is not NotImplemented:
                # e.g. tests.series.test_ufunc.TestNumpyReductions
                return result

        if len(inputs) == 1:
            # No alignment necessary.
            sp_values = getattr(ufunc, method)(self.sp_values, **kwargs)
            fill_value = getattr(ufunc, method)(self.fill_value, **kwargs)

            if ufunc.nout > 1:
                # multiple outputs. e.g. modf
                arrays = tuple(
                    self._simple_new(
                        sp_value, self.sp_index, SparseDtype(sp_value.dtype, fv)
                    )
                    for sp_value, fv in zip(sp_values, fill_value, strict=True)
                )
                return arrays
            elif method == "reduce":
                # e.g. reductions
                return sp_values

            return self._simple_new(
                sp_values, self.sp_index, SparseDtype(sp_values.dtype, fill_value)
            )

        new_inputs = tuple(np.asarray(x) for x in inputs)
        result = getattr(ufunc, method)(*new_inputs, **kwargs)
        if out:
            if len(out) == 1:
                out = out[0]
            return out

        if ufunc.nout > 1:
            return tuple(type(self)(x) for x in result)
        elif method == "at":
            # no return value
            return None
        else:
            return type(self)(result)

    # ------------------------------------------------------------------------
    # Ops
    # ------------------------------------------------------------------------

    def _arith_method(self, other, op):
        # [FUNCTION DOC]
        # HELPER: implementation of arithmetic operators. other is a sparse array, a
        # scalar or an array-like; op is the operator function.
        # Sparse operand: the binary engine is used. Scalar operand: the operator is
        # applied to the stored values and to the fill value (numpy warnings
        # suppressed), keeping the index; the paired quotient-and-remainder
        # operation returns a pair. Other operands are converted to a numpy array (a
        # length mismatch raises an assertion error), turned into a sparse array
        # using this array's fill value and the operand's dtype, and the binary
        # engine is used. Returns the sparse result (or a pair).
        op_name = op.__name__

        if isinstance(other, SparseArray):
            return _sparse_array_op(self, other, op, op_name)

        elif is_scalar(other):
            with np.errstate(all="ignore"):
                fill = op(_get_fill(self), np.asarray(other))
                result = op(self.sp_values, other)

            if op_name == "divmod":
                left, right = result
                lfill, rfill = fill
                return (
                    _wrap_result(op_name, left, self.sp_index, lfill),
                    _wrap_result(op_name, right, self.sp_index, rfill),
                )

            return _wrap_result(op_name, result, self.sp_index, fill)

        else:
            other = np.asarray(other)
            with np.errstate(all="ignore"):
                if len(self) != len(other):
                    raise AssertionError(
                        f"length mismatch: {len(self)} vs. {len(other)}"
                    )
                if not isinstance(other, SparseArray):
                    dtype = getattr(other, "dtype", None)
                    other = SparseArray(other, fill_value=self.fill_value, dtype=dtype)
                return _sparse_array_op(self, other, op, op_name)

    def _cmp_method(self, other, op) -> SparseArray:
        # [FUNCTION DOC]
        # HELPER: implementation of comparison operators (also bound as the logical
        # operator implementation). other is a scalar, array-like or sparse array;
        # op is the operator function. Non-scalar, non-sparse operands become numpy
        # arrays and then sparse arrays with this array's fill value. Sparse
        # operands must have equal length (value error otherwise) and go through the
        # binary engine using the operator name. Scalars: the fill value is compared
        # to compute the new fill, the stored values are compared, and the dense
        # boolean result is built and sparsified. Returns a sparse boolean array.
        if not is_scalar(other) and not isinstance(other, type(self)):
            # convert list-like to ndarray
            other = np.asarray(other)

        if isinstance(other, np.ndarray):
            # TODO: make this more flexible than just ndarray...
            other = SparseArray(other, fill_value=self.fill_value)

        if isinstance(other, SparseArray):
            if len(self) != len(other):
                raise ValueError(
                    f"operands have mismatched length {len(self)} and {len(other)}"
                )

            op_name = op.__name__.strip("_")
            return _sparse_array_op(self, other, op, op_name)
        else:
            # scalar
            fill_value = op(self.fill_value, other)
            result = np.full(len(self), fill_value, dtype=np.bool_)
            result[self.sp_index.indices] = op(self.sp_values, other)

            return type(self)(
                result,
                fill_value=fill_value,
                dtype=np.bool_,
            )

    _logical_method = _cmp_method

    def _unary_method(self, op) -> SparseArray:
        # [FUNCTION DOC]
        # HELPER: shared implementation of the unary operators. op is the operator
        # function. The operator is applied to the fill value; if the fill value is
        # missing or unchanged, only the stored values are transformed and the index
        # and dtype are kept. Otherwise the array is densified, transformed and
        # re-sparsified around the new fill value. Returns a new sparse array.
        fill_value = op(np.array(self.fill_value)).item()
        dtype = SparseDtype(self.dtype.subtype, fill_value)
        # NOTE: if fill_value doesn't change
        # we just have to apply op to sp_values
        if isna(self.fill_value) or fill_value == self.fill_value:
            values = op(self.sp_values)
            return type(self)._simple_new(values, self.sp_index, self.dtype)
        # In the other case we have to recalc indexes
        return type(self)(op(self.to_dense()), dtype=dtype)

    def __pos__(self) -> SparseArray:
        # [FUNCTION DOC]
        # Unary plus via the shared unary helper; returns a sparse array.
        return self._unary_method(operator.pos)

    def __neg__(self) -> SparseArray:
        # [FUNCTION DOC]
        # Unary negation via the shared unary helper; returns a sparse array.
        return self._unary_method(operator.neg)

    def __invert__(self) -> SparseArray:
        # [FUNCTION DOC]
        # Unary inversion (bitwise/logical not) via the shared unary helper;
        # returns a sparse array.
        return self._unary_method(operator.invert)

    def __abs__(self) -> SparseArray:
        # [FUNCTION DOC]
        # Absolute value via the shared unary helper; returns a sparse array.
        return self._unary_method(operator.abs)

    # ----------
    # Formatting
    # -----------
    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Text representation made of three parts on separate lines: the pretty
        # printed values, the text "Fill: " with the pretty printed fill value, and
        # the pretty printed sparse index description. Returns the combined string.
        pp_str = printing.pprint_thing(self)
        pp_fill = printing.pprint_thing(self.fill_value)
        pp_index = printing.pprint_thing(self.sp_index)
        return f"{pp_str}\nFill: {pp_fill}\n{pp_index}"

    # error: Return type "None" of "_formatter" incompatible with return
    # type "Callable[[Any], str | None]" in supertype "ExtensionArray"
    def _formatter(self, boxed: bool = False) -> None:  # type: ignore[override]
        # [FUNCTION DOC]
        # HELPER: display hook. Always returns nothing, which tells the caller (the
        # generic array formatter) to infer the element formatter from the dtype of
        # the values itself. boxed is ignored.
        # Defer to the formatter from the GenericArrayFormatter calling us.
        # This will infer the correct formatter from the dtype of the values.
        return None


def _make_sparse(
    arr: np.ndarray,
    kind: SparseIndexKind = "block",
    fill_value=None,
    dtype: np.dtype | None = None,
):
    # [FUNCTION DOC]
    # HELPER: sparsifies a dense numpy array for the constructor. arr must be a
    # numpy array of at most one dimension (assertion error if not an array, type
    # error if more dimensions); kind is "block" or "integer" for the index layout
    # (the module default here is block, unlike the class constructor); fill_value
    # (default none, the natural missing value of the array dtype) is the value
    # that is not stored; dtype (default none) casts the stored values.
    # Steps: the mask of positions to store is not-missing (for a missing fill
    # value) or not-equal to the fill value (strings converted to object first;
    # object arrays compared by type and value through a compiled helper). The
    # stored positions become an index of the requested kind and the stored values
    # are cut out (cast when a dtype is given). Returns a triple (stored values,
    # sparse index, fill value).
    # KNOWN ISSUE: the docstring lists a copy parameter that does not exist, and
    # the branch handling a mask whose length differs from the array (the comment
    # says the array is a sparse array) is unreachable because the array is
    # asserted to be a numpy array; a source TODO also notes copy is not handled.
    """
    Convert ndarray to sparse format

    Parameters
    ----------
    arr : ndarray
    kind : {'block', 'integer'}
    fill_value : NaN or another value
    dtype : np.dtype, optional
    copy : bool, default False

    Returns
    -------
    (sparse_values, index, fill_value) : (ndarray, SparseIndex, Scalar)
    """
    assert isinstance(arr, np.ndarray)

    if arr.ndim > 1:
        raise TypeError("expected dimension <= 1 data")

    if fill_value is None:
        fill_value = na_value_for_dtype(arr.dtype)

    if isna(fill_value):
        mask = notna(arr)
    else:
        # cast to object comparison to be safe
        if is_string_dtype(arr.dtype):
            arr = arr.astype(object)

        if is_object_dtype(arr.dtype):
            # element-wise equality check method in numpy doesn't treat
            # each element type, eg. 0, 0.0, and False are treated as
            # same. So we have to check the both of its type and value.
            mask = splib.make_mask_object_ndarray(arr, fill_value)
        else:
            mask = arr != fill_value

    length = len(arr)
    if length != len(mask):
        # the arr is a SparseArray
        indices = mask.sp_index.indices
    else:
        indices = mask.nonzero()[0].astype(np.int32)

    index = make_sparse_index(length, indices, kind)
    sparsified_values = arr[mask]
    if dtype is not None:
        sparsified_values = ensure_wrapped_if_datetimelike(sparsified_values)
        sparsified_values = astype_array(sparsified_values, dtype=dtype)
        sparsified_values = np.asarray(sparsified_values)

    # TODO: copy
    return sparsified_values, index, fill_value


@overload
def make_sparse_index(length: int, indices, kind: Literal["block"]) -> BlockIndex:
    # [FUNCTION DOC]
    # Typing stub only: block kind gives a block index. See the implementation.
    ...


@overload
def make_sparse_index(length: int, indices, kind: Literal["integer"]) -> IntIndex:
    # [FUNCTION DOC]
    # Typing stub only: integer kind gives an integer index. See the
    # implementation.
    ...


def make_sparse_index(length: int, indices, kind: SparseIndexKind) -> SparseIndex:
    # [FUNCTION DOC]
    # Public factory for a sparse index. length is the total array length; indices
    # is a sorted integer array of stored positions; kind is "block" or "integer".
    # Block kind: contiguous runs of positions are converted to block starts and
    # lengths by a compiled helper and a block index is built. Integer kind: an
    # integer index is built directly. Any other kind raises a value error. Returns
    # the index.
    index: SparseIndex
    if kind == "block":
        locs, lens = splib.get_blocks(indices)
        index = BlockIndex(length, locs, lens)
    elif kind == "integer":
        index = IntIndex(length, indices)
    else:  # pragma: no cover
        raise ValueError("must be block or integer type")
    return index
