# [MODULE DOC]
# Interval extension array. Public contents: the interval array class (exposed
# in the arrays namespace) storing intervals that are all closed on the same
# side as two parallel arrays of left and right bounds plus an interval dtype
# carrying the bound subtype and the closed side; shared-docstring templates
# (a dictionary of format strings reused by the interval index class); two
# type aliases for bound arrays and elements; and private helpers that
# convert input to a platform-appropriate array and that validate bound
# arrays. Supported bound subtypes are numeric, datetime and timedelta types
# (category, object and string subtypes and periods are rejected).
# Relationships: builds on the interval mixin (scalar behavior, closed-side
# properties, from the compiled interval module) and the base extension
# array; bounds are stored as numpy arrays or datetime/timedelta arrays; the
# index class for intervals wraps this array; string array and arrow array
# classes are only used for type checks. Missing values are represented by
# NaN in both bounds.
# No import-time side effects besides populating the shared-docstring
# dictionary and applying decorators.
from __future__ import annotations

import operator
from operator import (
    le,
    lt,
)
import textwrap
from typing import (
    TYPE_CHECKING,
    Literal,
    Self,
    TypeAlias,
    overload,
)

import numpy as np

from pandas._libs import lib
from pandas._libs.interval import (
    VALID_CLOSED,
    Interval,
    IntervalMixin,
    intervals_to_interval_bounds,
)
from pandas._libs.missing import NA
from pandas._typing import (
    ArrayLike,
    AxisInt,
    Dtype,
    IntervalClosedType,
    NpDtype,
    PositionalIndexer,
    ScalarIndexer,
    SequenceIndexer,
    SortKind,
    TimeArrayLike,
    npt,
)
from pandas.compat.numpy import function as nv
from pandas.errors import IntCastingNaNError
from pandas.util._decorators import set_module

from pandas.core.dtypes.cast import (
    LossySetitemError,
    maybe_upcast_numeric_to_64bit,
)
from pandas.core.dtypes.common import (
    is_float_dtype,
    is_integer_dtype,
    is_list_like,
    is_object_dtype,
    is_scalar,
    is_string_dtype,
    needs_i8_conversion,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    IntervalDtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCDatetimeIndex,
    ABCIntervalIndex,
    ABCPeriodIndex,
)
from pandas.core.dtypes.missing import (
    is_valid_na_for_dtype,
    isna,
    notna,
)

from pandas.core.algorithms import (
    isin,
    take,
    unique,
)
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.arrays.base import (
    ExtensionArray,
)
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.arrays.timedeltas import TimedeltaArray
import pandas.core.common as com
from pandas.core.construction import (
    array as pd_array,
    ensure_wrapped_if_datetimelike,
    extract_array,
)
from pandas.core.indexers import (
    check_array_indexer,
    getitem_returns_view,
)
from pandas.core.ops import (
    invalid_comparison,
    unpack_zerodim_and_defer,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterator,
        Sequence,
    )

    from pandas import (
        Index,
    )


IntervalSide: TypeAlias = TimeArrayLike | np.ndarray
IntervalOrNA: TypeAlias = Interval | float

_interval_shared_docs: dict[str, str] = {}

_shared_docs_kwargs = {
    "klass": "IntervalArray",
    "qualname": "arrays.IntervalArray",
    "name": "",
}


_interval_shared_docs["class"] = """
%(summary)s

Parameters
----------
data : array-like (1-dimensional)
    Array-like (ndarray, :class:`DateTimeArray`, :class:`TimeDeltaArray`) containing
    Interval objects from which to build the %(klass)s.
closed : {'left', 'right', 'both', 'neither'}, default 'right'
    Whether the intervals are closed on the left-side, right-side, both or
    neither.
dtype : dtype or None, default None
    If None, dtype will be inferred.
copy : bool, default False
    Copy the input data.
%(name)s\
verify_integrity : bool, default True
    Verify that the %(klass)s is valid.

Attributes
----------
left
right
closed
mid
length
is_empty
is_non_overlapping_monotonic
%(extra_attributes)s\

Methods
-------
from_arrays
from_tuples
from_breaks
contains
overlaps
set_closed
to_tuples
%(extra_methods)s\

See Also
--------
Index : The base pandas Index type.
Interval : A bounded slice-like interval; the elements of an %(klass)s.
interval_range : Function to create a fixed frequency IntervalIndex.
cut : Bin values into discrete Intervals.
qcut : Bin values into equal-sized Intervals based on rank or sample quantiles.

Notes
-----
See the `user guide
<https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#intervalindex>`__
for more.

%(examples)s\
"""


@set_module("pandas.arrays")
class IntervalArray(IntervalMixin, ExtensionArray):
    # [CLASS DOC]
    # Array of intervals sharing the same closed side. Inherits the interval
    # mixin (closed-side helper properties and scalar-level operations) and the
    # generic extension array. State: the left bound array, the right bound
    # array (same length, same bound type; numpy arrays, or datetime/timedelta
    # arrays) and an interval dtype recording the bound subtype and the closed
    # side. Invariants (checked by validation unless skipped): the closed side
    # is one of the four valid values; each left bound is less than or equal to
    # the matching right bound; missing entries are missing in both bounds
    # simultaneously. Missing value is NaN. Always one-dimensional.
    # Constructors: the regular constructor (from intervals), from_breaks,
    # from_arrays and from_tuples; internal fast constructors skip validation.
    # Other groups: element access and assignment, comparison (equality only;
    # ordering is rejected), set_closed, interval geometry properties (mid,
    # length, emptiness, monotonicity), overlap and containment tests, take/
    # fill/unique/isin, conversion to tuples and arrow, and shared-docstring
    # templates.
    """
    Pandas array for interval data that are closed on the same side.

    Parameters
    ----------
    data : array-like (1-dimensional)
        Array-like (ndarray, :class:`DateTimeArray`, :class:`TimeDeltaArray`) containing
        Interval objects from which to build the IntervalArray.
    closed : {'left', 'right', 'both', 'neither'}, default 'right'
        Whether the intervals are closed on the left-side, right-side, both or
        neither.
    dtype : dtype or None, default None
        If None, dtype will be inferred.
    copy : bool, default False
        Copy the input data.
    verify_integrity : bool, default True
        Verify that the IntervalArray is valid.

    Attributes
    ----------
    left
    right
    closed
    mid
    length
    is_empty
    is_non_overlapping_monotonic

    Methods
    -------
    from_arrays
    from_tuples
    from_breaks
    contains
    overlaps
    set_closed
    to_tuples

    See Also
    --------
    Index : The base pandas Index type.
    Interval : A bounded slice-like interval; the elements of an IntervalArray.
    interval_range : Function to create a fixed frequency IntervalIndex.
    cut : Bin values into discrete Intervals.
    qcut : Bin values into equal-sized Intervals based on rank or sample quantiles.

    Notes
    -----
    See the `user guide
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#intervalindex>`__
    for more.

    Examples
    --------
    A new ``IntervalArray`` can be constructed directly from an array-like of
    ``Interval`` objects:
    >>> pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])
    <IntervalArray>
    [(0, 1], (1, 5]]
    Length: 2, dtype: interval[int64, right]

    It may also be constructed using one of the constructor
    methods: :meth:`IntervalArray.from_arrays`,
    :meth:`IntervalArray.from_breaks`, and :meth:`IntervalArray.from_tuples`.
    """

    can_hold_na = True
    _na_value = _fill_value = np.nan

    @property
    def ndim(self) -> Literal[1]:
        # [FUNCTION DOC]
        # Property: always the integer one (this array is one-dimensional).
        return 1

    # To make mypy recognize the fields
    _left: IntervalSide
    _right: IntervalSide
    _dtype: IntervalDtype

    # ---------------------------------------------------------------------
    # Constructors

    def __new__(
        cls,
        data,
        closed: IntervalClosedType | None = None,
        dtype: Dtype | None = None,
        copy: bool = False,
        verify_integrity: bool = True,
    ) -> Self:
        # [FUNCTION DOC]
        # Public constructor (a new-style constructor, no separate initializer).
        # data is a one-dimensional collection of Interval objects, an
        # existing interval array/index, or missing-only/empty data; closed
        # (default none) is left, right, both or neither and is inferred from the
        # data when omitted; dtype (default none) is an interval dtype to
        # enforce; copy (default false) copies the bounds; verify_integrity
        # (default true) runs the validation of the bounds.
        # Behavior: an existing interval array donates its bounds and the
        # closed side is the argument if given else its own (no check that the
        # argument matches). Scalars raise a type error. Other input is
        # converted to a platform array, bounds are extracted (the closed side is
        # inferred and its consistency is validated by the extractor only when
        # closed is not given), object bounds are converted to a better dtype,
        # and bounds/dtype are normalized by the input-ensuring helper.
        # Raises: type error for scalars or unsupported bound types, value error
        # from validation, normalization or the extractor. Returns the new array.
        data = extract_array(data, extract_numpy=True)

        if isinstance(data, cls):
            left: IntervalSide = data._left
            right: IntervalSide = data._right
            closed = closed or data.closed
            dtype = IntervalDtype(left.dtype, closed=closed)
        else:
            # don't allow scalars
            if is_scalar(data):
                msg = (
                    f"{cls.__name__}(...) must be called with a collection "
                    f"of some kind, {data} was passed"
                )
                raise TypeError(msg)

            # might need to convert empty or purely na data
            data = _maybe_convert_platform_interval(data)
            left, right, infer_closed = intervals_to_interval_bounds(
                data, validate_closed=closed is None
            )
            if left.dtype == object:
                left = lib.maybe_convert_objects(left)
                right = lib.maybe_convert_objects(right)
            closed = closed or infer_closed

            left, right, dtype = cls._ensure_simple_new_inputs(
                left,
                right,
                closed=closed,
                copy=copy,
                dtype=dtype,
            )

        if verify_integrity:
            cls._validate(left, right, dtype=dtype)

        return cls._simple_new(
            left,
            right,
            dtype=dtype,
        )

    @classmethod
    def _simple_new(
        cls,
        left: IntervalSide,
        right: IntervalSide,
        dtype: IntervalDtype,
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: fast internal constructor with no validation. left and right
        # are the already-normalized bound arrays and dtype the interval dtype.
        # Creates an instance bypassing the public constructor and sets the three
        # private attributes. Returns the instance.
        result = IntervalMixin.__new__(cls)
        result._left = left
        result._right = right
        result._dtype = dtype

        return result

    @classmethod
    def _ensure_simple_new_inputs(
        cls,
        left,
        right,
        closed: IntervalClosedType | None = None,
        copy: bool = False,
        dtype: Dtype | None = None,
    ) -> tuple[IntervalSide, IntervalSide, IntervalDtype]:
        # [FUNCTION DOC]
        # HELPER: normalizes bounds and dtype before the fast constructor. left
        # and right are array-likes of bounds; closed (default none) is the closed
        # side, falling back to the dtype's closed side, then to right; copy
        # (default false) copies the bounds; dtype (default none) is an interval
        # dtype whose subtype, when set, is used to cast both bounds.
        # Steps: bounds become index objects, small numeric types are upcast to
        # 64-bit; a non-interval dtype raises a type error; a dtype with no
        # closed side adopts the argument, while a differing closed side raises
        # a value error; float/integer mismatches between the bounds cast the
        # integer side to float; differing bound container types raise a value
        # error; category, object and string bounds raise a type error; period
        # bounds raise a value error; datetime bounds with different zones
        # raise a value error; timedelta/datetime bounds with different units
        # are cast to a common unit; bounds are rewrapped as datetime/timedelta
        # arrays and plain arrays; if the two bounds share memory the right bound
        # is copied so in-place assignment cannot corrupt the array (skipped for
        # arrow-backed bounds); finally a dtype is built from the left dtype and
        # closed side, and mixed signed/unsigned integer bounds raise a type
        # error. Returns the triple (left, right, dtype).
        """Ensure correctness of input parameters for cls._simple_new."""
        from pandas.core.indexes.base import ensure_index

        left = ensure_index(left, copy=copy)
        left = maybe_upcast_numeric_to_64bit(left)

        right = ensure_index(right, copy=copy)
        right = maybe_upcast_numeric_to_64bit(right)

        if closed is None and isinstance(dtype, IntervalDtype):
            closed = dtype.closed

        closed = closed or "right"

        if dtype is not None:
            # GH 19262: dtype must be an IntervalDtype to override inferred
            dtype = pandas_dtype(dtype)
            if isinstance(dtype, IntervalDtype):
                if dtype.subtype is not None:
                    left = left.astype(dtype.subtype)
                    right = right.astype(dtype.subtype)
            else:
                msg = f"dtype must be an IntervalDtype, got {dtype}"
                raise TypeError(msg)

            if dtype.closed is None:
                # possibly loading an old pickle
                dtype = IntervalDtype(dtype.subtype, closed)
            elif closed != dtype.closed:
                raise ValueError("closed keyword does not match dtype.closed")

        # coerce dtypes to match if needed
        if is_float_dtype(left.dtype) and is_integer_dtype(right.dtype):
            right = right.astype(left.dtype)
        elif is_float_dtype(right.dtype) and is_integer_dtype(left.dtype):
            left = left.astype(right.dtype)

        if type(left) != type(right):
            msg = (
                f"must not have differing left [{type(left).__name__}] and "
                f"right [{type(right).__name__}] types"
            )
            raise ValueError(msg)
        if isinstance(left.dtype, CategoricalDtype) or is_string_dtype(left.dtype):
            # GH 19016
            msg = (
                "category, object, and string subtypes are not supported "
                "for IntervalArray"
            )
            raise TypeError(msg)
        if isinstance(left, ABCPeriodIndex):
            msg = "Period dtypes are not supported, use a PeriodIndex instead"
            raise ValueError(msg)
        if isinstance(left, ABCDatetimeIndex) and str(left.tz) != str(right.tz):
            msg = (
                "left and right must have the same time zone, got "
                f"'{left.tz}' and '{right.tz}'"
            )
            raise ValueError(msg)
        elif needs_i8_conversion(left.dtype) and left.unit != right.unit:
            # e.g. m8[s] vs m8[ms], try to cast to a common dtype GH#55714
            left_arr, right_arr = left._data._ensure_matching_resos(right._data)
            left = ensure_index(left_arr)
            right = ensure_index(right_arr)

        # For dt64/td64 we want DatetimeArray/TimedeltaArray instead of ndarray
        left = ensure_wrapped_if_datetimelike(left)
        left = extract_array(left, extract_numpy=True)
        right = ensure_wrapped_if_datetimelike(right)
        right = extract_array(right, extract_numpy=True)

        if isinstance(left, ArrowExtensionArray) or isinstance(
            right, ArrowExtensionArray
        ):
            pass
        else:
            lbase = getattr(left, "_ndarray", left)
            lbase = getattr(lbase, "_data", lbase).base
            rbase = getattr(right, "_ndarray", right)
            rbase = getattr(rbase, "_data", rbase).base
            if lbase is not None and lbase is rbase:
                # If these share data, then setitem could corrupt our IA
                right = right.copy()

        dtype = IntervalDtype(left.dtype, closed=closed)

        # Check for mismatched signed/unsigned integer dtypes after casting
        left_dtype = left.dtype
        right_dtype = right.dtype
        if (
            left_dtype.kind in "iu"
            and right_dtype.kind in "iu"
            and left_dtype.kind != right_dtype.kind
        ):
            raise TypeError(
                f"Left and right arrays must have matching signedness. "
                f"Got {left_dtype} and {right_dtype}."
            )
        return left, right, dtype

    @classmethod
    def _from_sequence(
        cls,
        scalars,
        *,
        dtype: Dtype | None = None,
        copy: bool = False,
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: class-method constructor used by the generic array factory.
        # scalars is a collection of intervals; keyword-only dtype and copy
        # (default none and false) are forwarded to the public constructor.
        return cls(scalars, dtype=dtype, copy=copy)

    @classmethod
    def _from_factorized(cls, values: np.ndarray, original: IntervalArray) -> Self:
        # [FUNCTION DOC]
        # HELPER: rebuilds an array from unique values produced by factorize.
        # values is the array of unique intervals as objects; original supplies
        # the dtype. Returns a new array with that dtype.
        return cls._from_sequence(values, dtype=original.dtype)

    _interval_shared_docs["from_breaks"] = textwrap.dedent(
        """
        Construct an %(klass)s from an array of splits.

        Parameters
        ----------
        breaks : array-like (1-dimensional)
            Left and right bounds for each interval.
        closed : {'left', 'right', 'both', 'neither'}, default 'right'
            Whether the intervals are closed on the left-side, right-side, both
            or neither.\
        %(name)s
        copy : bool, default False
            Copy the data.
        dtype : dtype or None, default None
            If None, dtype will be inferred.

        Returns
        -------
        %(klass)s

        See Also
        --------
        interval_range : Function to create a fixed frequency IntervalIndex.
        %(klass)s.from_arrays : Construct from a left and right array.
        %(klass)s.from_tuples : Construct from a sequence of tuples.

        %(examples)s\
        """
    )

    @classmethod
    def from_breaks(
        cls,
        breaks,
        closed: IntervalClosedType | None = "right",
        copy: bool = False,
        dtype: Dtype | None = None,
    ) -> Self:
        # [FUNCTION DOC]
        # Builds intervals from consecutive break points. breaks is an
        # array-like of n boundaries (n minus one intervals result); closed
        # (default right) is the closed side; copy (default false); dtype
        # (default none). The breaks are converted to a platform array and
        # the left bounds are all but the last and the right bounds all but the
        # first, passed to the arrays constructor (which validates).
        # Returns the new array.
        """
        Construct an IntervalArray from an array of splits.

        Parameters
        ----------
        breaks : array-like (1-dimensional)
            Left and right bounds for each interval.
        closed : {'left', 'right', 'both', 'neither'}, default 'right'
            Whether the intervals are closed on the left-side, right-side, both
            or neither.
        copy : bool, default False
            Copy the data.
        dtype : dtype or None, default None
            If None, dtype will be inferred.

        Returns
        -------
        IntervalArray

        See Also
        --------
        interval_range : Function to create a fixed frequency IntervalIndex.
        IntervalArray.from_arrays : Construct from a left and right array.
        IntervalArray.from_tuples : Construct from a sequence of tuples.

        Examples
        --------
        >>> pd.arrays.IntervalArray.from_breaks([0, 1, 2, 3])
        <IntervalArray>
        [(0, 1], (1, 2], (2, 3]]
        Length: 3, dtype: interval[int64, right]
        """

        breaks = _maybe_convert_platform_interval(breaks)

        return cls.from_arrays(breaks[:-1], breaks[1:], closed, copy=copy, dtype=dtype)

    _interval_shared_docs["from_arrays"] = textwrap.dedent(
        """
        Construct from two arrays defining the left and right bounds.

        Parameters
        ----------
        left : array-like (1-dimensional)
            Left bounds for each interval.
        right : array-like (1-dimensional)
            Right bounds for each interval.
        closed : {'left', 'right', 'both', 'neither'}, default 'right'
            Whether the intervals are closed on the left-side, right-side, both
            or neither.\
        %(name)s
        copy : bool, default False
            Copy the data.
        dtype : dtype, optional
            If None, dtype will be inferred.

        Returns
        -------
        %(klass)s

        Raises
        ------
        ValueError
            When a value is missing in only one of `left` or `right`.
            When a value in `left` is greater than the corresponding value
            in `right`.

        See Also
        --------
        interval_range : Function to create a fixed frequency IntervalIndex.
        %(klass)s.from_breaks : Construct an %(klass)s from an array of
            splits.
        %(klass)s.from_tuples : Construct an %(klass)s from an
            array-like of tuples.

        Notes
        -----
        Each element of `left` must be less than or equal to the `right`
        element at the same position. If an element is missing, it must be
        missing in both `left` and `right`. A TypeError is raised when
        using an unsupported type for `left` or `right`. At the moment,
        'category', 'object', and 'string' subtypes are not supported.

        %(examples)s\
        """
    )

    @classmethod
    def from_arrays(
        cls,
        left,
        right,
        closed: IntervalClosedType | None = "right",
        copy: bool = False,
        dtype: Dtype | None = None,
    ) -> Self:
        # [FUNCTION DOC]
        # Builds intervals from separate left and right bound arrays. left and
        # right are equal-length array-likes; closed (default right); copy
        # (default false); dtype (default none). Both are converted to platform
        # arrays, normalized with the input-ensuring helper, validated (value
        # error if a bound is missing on only one side or left exceeds right; type
        # error for unsupported subtypes) and wrapped with the fast constructor.
        # Returns the new array.
        """
        Construct from two arrays defining the left and right bounds.

        Parameters
        ----------
        left : array-like (1-dimensional)
            Left bounds for each interval.
        right : array-like (1-dimensional)
            Right bounds for each interval.
        closed : {'left', 'right', 'both', 'neither'}, default 'right'
            Whether the intervals are closed on the left-side, right-side, both
            or neither.
        copy : bool, default False
            Copy the data.
        dtype : dtype, optional
            If None, dtype will be inferred.

        Returns
        -------
        IntervalArray

        Raises
        ------
        ValueError
            When a value is missing in only one of `left` or `right`.
            When a value in `left` is greater than the corresponding value
            in `right`.

        See Also
        --------
        interval_range : Function to create a fixed frequency IntervalIndex.
        IntervalArray.from_breaks : Construct an IntervalArray from an array of
            splits.
        IntervalArray.from_tuples : Construct an IntervalArray from an
            array-like of tuples.

        Notes
        -----
        Each element of `left` must be less than or equal to the `right`
        element at the same position. If an element is missing, it must be
        missing in both `left` and `right`. A TypeError is raised when
        using an unsupported type for `left` or `right`. At the moment,
        'category', 'object', and 'string' subtypes are not supported.

        Examples
        --------
        >>> pd.arrays.IntervalArray.from_arrays([0, 1, 2], [1, 2, 3])
        <IntervalArray>
        [(0, 1], (1, 2], (2, 3]]
        Length: 3, dtype: interval[int64, right]
        """
        left = _maybe_convert_platform_interval(left)
        right = _maybe_convert_platform_interval(right)

        left, right, dtype = cls._ensure_simple_new_inputs(
            left,
            right,
            closed=closed,
            copy=copy,
            dtype=dtype,
        )
        cls._validate(left, right, dtype=dtype)

        return cls._simple_new(left, right, dtype=dtype)

    _interval_shared_docs["from_tuples"] = textwrap.dedent(
        """
        Construct an %(klass)s from an array-like of tuples.

        Parameters
        ----------
        data : array-like (1-dimensional)
            Array of tuples.
        closed : {'left', 'right', 'both', 'neither'}, default 'right'
            Whether the intervals are closed on the left-side, right-side, both
            or neither.\
        %(name)s
        copy : bool, default False
            By-default copy the data, this is compat only and ignored.
        dtype : dtype or None, default None
            If None, dtype will be inferred.

        Returns
        -------
        %(klass)s

        See Also
        --------
        interval_range : Function to create a fixed frequency IntervalIndex.
        %(klass)s.from_arrays : Construct an %(klass)s from a left and
                                    right array.
        %(klass)s.from_breaks : Construct an %(klass)s from an array of
                                    splits.

        %(examples)s\
        """
    )

    @classmethod
    def from_tuples(
        cls,
        data,
        closed: IntervalClosedType | None = "right",
        copy: bool = False,
        dtype: Dtype | None = None,
    ) -> Self:
        # [FUNCTION DOC]
        # Builds intervals from a sequence of two-element tuples. data is an
        # array-like of (left, right) tuples; scalar missing entries (not tuples)
        # become a missing interval; closed (default right); copy (default false)
        # is accepted for compatibility but ignored; dtype (default none).
        # Each tuple is unpacked into the two bounds. A tuple of the wrong length
        # raises a value error and a non-iterable non-missing item raises a type
        # error (both naming the class and item). Empty input keeps its own
        # container as both bounds so the input dtype is preserved. The bounds are
        # passed to the arrays constructor (always with copy disabled), so
        # validation errors from there also apply. Returns the new array.
        """
        Construct an IntervalArray from an array-like of tuples.

        Parameters
        ----------
        data : array-like (1-dimensional)
            Array of tuples.
        closed : {'left', 'right', 'both', 'neither'}, default 'right'
            Whether the intervals are closed on the left-side, right-side, both
            or neither.
        copy : bool, default False
            By-default copy the data, this is compat only and ignored.
        dtype : dtype or None, default None
            If None, dtype will be inferred.

        Returns
        -------
        IntervalArray

        See Also
        --------
        interval_range : Function to create a fixed frequency IntervalIndex.
        IntervalArray.from_arrays : Construct an IntervalArray from a left and
                                    right array.
        IntervalArray.from_breaks : Construct an IntervalArray from an array of
                                    splits.

        Examples
        --------
        >>> pd.arrays.IntervalArray.from_tuples([(0, 1), (1, 2)])
        <IntervalArray>
        [(0, 1], (1, 2]]
        Length: 2, dtype: interval[int64, right]
        """
        if len(data):
            left, right = [], []
        else:
            # ensure that empty data keeps input dtype
            left = right = data

        for d in data:
            if not isinstance(d, tuple) and isna(d):
                lhs = rhs = np.nan
            else:
                name = cls.__name__
                try:
                    # need list of length 2 tuples, e.g. [(0, 1), (1, 2), ...]
                    lhs, rhs = d
                except ValueError as err:
                    msg = f"{name}.from_tuples requires tuples of length 2, got {d}"
                    raise ValueError(msg) from err
                except TypeError as err:
                    msg = f"{name}.from_tuples received an invalid item, {d}"
                    raise TypeError(msg) from err
            left.append(lhs)
            right.append(rhs)

        return cls.from_arrays(left, right, closed, copy=False, dtype=dtype)

    @classmethod
    def _validate(cls, left, right, dtype: IntervalDtype) -> None:
        # [FUNCTION DOC]
        # HELPER: class-method validation of normalized bounds. left and right
        # are the bound arrays; dtype must be an interval dtype. Raises a value
        # error when: dtype is not an interval dtype; lengths differ; the
        # missing-value positions of the two bounds differ; or, among the
        # non-missing positions, any left bound exceeds its right bound (equal
        # bounds are allowed). Returns nothing; no mutation.
        """
        Verify that the IntervalArray is valid.

        Checks that

        * dtype is correct
        * left and right match lengths
        * left and right have the same missing values
        * left is always below right
        """
        if not isinstance(dtype, IntervalDtype):
            msg = f"invalid dtype: {dtype}"
            raise ValueError(msg)
        if len(left) != len(right):
            msg = "left and right must have the same length"
            raise ValueError(msg)
        left_mask = notna(left)
        right_mask = notna(right)
        if not (left_mask == right_mask).all():
            msg = (
                "missing values must be missing in the same "
                "location both left and right sides"
            )
            raise ValueError(msg)
        if not (left[left_mask] <= right[left_mask]).all():
            msg = "left side of interval must be <= right side"
            raise ValueError(msg)

    def _shallow_copy(self, left, right) -> Self:
        # [FUNCTION DOC]
        # HELPER: builds a new array of the same closed side from replacement
        # bounds (index-like objects). The dtype is derived from the new left
        # bound's dtype and the current closed side; bounds are normalized by the
        # input-ensuring helper (so its errors apply), then the fast constructor
        # is used. Returns the new array; this array is unchanged.
        """
        Return a new IntervalArray with the replacement attributes

        Parameters
        ----------
        left : Index
            Values to be used for the left-side of the intervals.
        right : Index
            Values to be used for the right-side of the intervals.
        """
        dtype = IntervalDtype(left.dtype, closed=self.closed)
        left, right, dtype = self._ensure_simple_new_inputs(left, right, dtype=dtype)

        return self._simple_new(left, right, dtype=dtype)

    # ---------------------------------------------------------------------
    # Descriptive

    @property
    def dtype(self) -> IntervalDtype:
        # [FUNCTION DOC]
        # Property returning the stored interval dtype.
        return self._dtype

    @property
    def nbytes(self) -> int:
        # [FUNCTION DOC]
        # Property: memory in bytes, the sum of the left and right bound arrays.
        return self.left.nbytes + self.right.nbytes

    @property
    def size(self) -> int:
        # [FUNCTION DOC]
        # Property: number of elements, taken from the left bound so the object
        # array of intervals is not materialized.
        # Avoid materializing self.values
        return self.left.size

    # ---------------------------------------------------------------------
    # EA Interface

    def __iter__(self) -> Iterator:
        # [FUNCTION DOC]
        # Iterates over the elements: converts the array to an object ndarray of
        # interval scalars (missing as NaN) and iterates it.
        return iter(np.asarray(self))

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Number of intervals (the length of the left bound array).
        return len(self._left)

    @overload
    def __getitem__(self, key: ScalarIndexer) -> IntervalOrNA:
        # [FUNCTION DOC]
        # Typing stub only: scalar position gives one interval or NaN. See the
        # implementation below.
        ...

    @overload
    def __getitem__(self, key: SequenceIndexer) -> Self:
        # [FUNCTION DOC]
        # Typing stub only: slice/mask/position-array key gives a new array. See
        # the implementation below.
        ...

    def __getitem__(self, key: PositionalIndexer) -> Self | IntervalOrNA:
        # [FUNCTION DOC]
        # Positional indexing. key is an integer, slice, boolean mask or integer
        # array (validated as an array indexer). The same key is applied to both
        # bounds. A scalar key returns the missing value (NaN) if the bound is
        # missing, otherwise an interval scalar with this array's closed side.
        # Otherwise returns a new array (via the fast constructor); when the key
        # yields a view the new array inherits the read-only flag. Raises a
        # value error for keys producing multi-dimensional bounds.
        key = check_array_indexer(self, key)
        left = self._left[key]
        right = self._right[key]

        if not isinstance(left, (np.ndarray, ExtensionArray)):
            # scalar
            if is_scalar(left) and isna(left):
                return self._fill_value
            return Interval(left, right, self.closed)
        if np.ndim(left) > 1:
            # GH#30588 multi-dimensional indexer disallowed
            raise ValueError("multi-dimensional indexing not allowed")
        # Argument 2 to "_simple_new" of "IntervalArray" has incompatible type
        # "Union[Period, Timestamp, Timedelta, NaTType, DatetimeArray, TimedeltaArray,
        # ndarray[Any, Any]]"; expected "Union[Union[DatetimeArray, TimedeltaArray],
        # ndarray[Any, Any]]"
        result = self._simple_new(left, right, dtype=self.dtype)  # type: ignore[arg-type]
        if getitem_returns_view(self, key):
            result._readonly = self._readonly
        return result

    def __setitem__(self, key, value) -> None:
        # [FUNCTION DOC]
        # In-place assignment. key is an integer, slice, mask or position array;
        # value is an interval, a list-like of intervals or a missing value.
        # Raises a value error if the array is read-only; type errors come from
        # value validation (wrong type, mismatched closed side or bound type,
        # NaN into integer-backed bounds). The value is split into left and right
        # parts and written into both bound arrays at the key. Mutates this
        # array.
        if self._readonly:
            raise ValueError("Cannot modify read-only array")

        value_left, value_right = self._validate_setitem_value(value)
        key = check_array_indexer(self, key)

        self._left[key] = value_left
        self._right[key] = value_right

    def _cmp_method(self, other, op):
        # [FUNCTION DOC]
        # HELPER: implementation behind all six comparison operators. other is
        # an interval, an array-like (must match length, else value error), a
        # missing scalar, or another scalar; op is the operator function.
        # A missing-value scalar (the NA singleton) gives an all-missing nullable
        # boolean array; other non-interval scalars go to the generic
        # invalid-comparison handler (equality false, ordering raises a type
        # error). The element dtype of other is determined (categoricals defer to
        # their categories, and interval categories with the same closed side are
        # expanded to values). Interval-like operands with a different closed side
        # are invalid comparisons. For interval-like operands, a boolean ndarray is
        # produced: equality compares both bounds; ordering compares left bounds
        # first and breaks ties by right bounds; greater-or-equal and
        # less-or-equal combine equality with the strict ordering. Non-object
        # non-interval dtypes are invalid comparisons. Object operands are
        # compared element by element, producing booleans (or an object array
        # containing the NA singleton where a comparison with it fails); other
        # type errors propagate.
        # KNOWN ISSUE: the comment on the final branch of the operator chain says
        # it handles the less-than operator, but that was handled just above;
        # the branch actually handles less-than-or-equal.
        # ensure pandas array for list-like and eliminate non-interval scalars
        if is_list_like(other):
            if len(self) != len(other):
                raise ValueError("Lengths must match to compare")
            other = pd_array(other)
        elif not isinstance(other, Interval):
            # non-interval scalar -> no matches
            if other is NA:
                # GH#31882
                from pandas.core.arrays import BooleanArray

                arr = np.empty(self.shape, dtype=bool)
                mask = np.ones(self.shape, dtype=bool)
                return BooleanArray(arr, mask)
            return invalid_comparison(self, other, op)

        # determine the dtype of the elements we want to compare
        if isinstance(other, Interval):
            other_dtype = pandas_dtype("interval")
        elif not isinstance(other.dtype, CategoricalDtype):
            other_dtype = other.dtype
        else:
            # for categorical defer to categories for dtype
            other_dtype = other.categories.dtype

            # extract intervals if we have interval categories with matching closed
            if isinstance(other_dtype, IntervalDtype):
                if self.closed != other.categories.closed:
                    return invalid_comparison(self, other, op)

                other = other.categories._values.take(
                    other.codes, allow_fill=True, fill_value=other.categories._na_value
                )

        # interval-like -> need same closed and matching endpoints
        if isinstance(other_dtype, IntervalDtype):
            if self.closed != other.closed:
                return invalid_comparison(self, other, op)
            elif not isinstance(other, Interval):
                other = type(self)(other)

            if op is operator.eq:
                return (self._left == other.left) & (self._right == other.right)
            elif op is operator.ne:
                return (self._left != other.left) | (self._right != other.right)
            elif op is operator.gt:
                return (self._left > other.left) | (
                    (self._left == other.left) & (self._right > other.right)
                )
            elif op is operator.ge:
                return (self == other) | (self > other)
            elif op is operator.lt:
                return (self._left < other.left) | (
                    (self._left == other.left) & (self._right < other.right)
                )
            else:
                # operator.lt
                return (self == other) | (self < other)

        # non-interval/non-object dtype -> no matches
        if not is_object_dtype(other_dtype):
            return invalid_comparison(self, other, op)

        # object dtype -> iteratively check for intervals
        result = np.zeros(len(self), dtype=bool)
        for i, obj in enumerate(other):
            try:
                result[i] = op(self[i], obj)
            except TypeError:
                if obj is NA:
                    # comparison with np.nan returns NA
                    # github.com/pandas-dev/pandas/pull/37124#discussion_r509095092
                    result = result.astype(object)
                    result[i] = NA
                else:
                    raise
        return result

    @unpack_zerodim_and_defer("__eq__")
    def __eq__(self, other):
        # [FUNCTION DOC]
        # Element-wise equality. Wrapped by a decorator that unpacks zero-
        # dimensional arrays and defers to series/index/frame operands.
        # Delegates to the comparison implementation; returns a boolean array.
        return self._cmp_method(other, operator.eq)

    @unpack_zerodim_and_defer("__ne__")
    def __ne__(self, other):
        # [FUNCTION DOC]
        # Element-wise inequality; same wrapping and delegation as equality.
        return self._cmp_method(other, operator.ne)

    @unpack_zerodim_and_defer("__gt__")
    def __gt__(self, other):
        # [FUNCTION DOC]
        # Element-wise greater-than (left bound first, then right bound);
        # same wrapping and delegation as equality.
        return self._cmp_method(other, operator.gt)

    @unpack_zerodim_and_defer("__ge__")
    def __ge__(self, other):
        # [FUNCTION DOC]
        # Element-wise greater-or-equal; same wrapping and delegation as
        # equality.
        return self._cmp_method(other, operator.ge)

    @unpack_zerodim_and_defer("__lt__")
    def __lt__(self, other):
        # [FUNCTION DOC]
        # Element-wise less-than (left bound first, then right bound); same
        # wrapping and delegation as equality.
        return self._cmp_method(other, operator.lt)

    @unpack_zerodim_and_defer("__le__")
    def __le__(self, other):
        # [FUNCTION DOC]
        # Element-wise less-or-equal; same wrapping and delegation as equality.
        return self._cmp_method(other, operator.le)

    def argsort(
        self,
        *,
        ascending: bool = True,
        kind: SortKind = "quicksort",
        na_position: str = "last",
        **kwargs,
    ) -> np.ndarray:
        # [FUNCTION DOC]
        # Returns the integer positions that sort the array. Keyword-only:
        # ascending (default true), kind (default quicksort), na_position
        # (default last), plus numpy-compat keywords that are validated. The
        # fast path (ascending, quicksort, missing last) uses a multi-key sort by
        # left bound then right bound; every other case delegates to the generic
        # implementation. Returns an integer array.
        ascending = nv.validate_argsort_with_ascending(ascending, (), kwargs)

        if ascending and kind == "quicksort" and na_position == "last":
            # TODO: in an IntervalIndex we can reuse the cached
            #  IntervalTree.left_sorter
            return np.lexsort((self.right, self.left))

        # TODO: other cases we can use lexsort for?  much more performant.
        return super().argsort(
            ascending=ascending, kind=kind, na_position=na_position, **kwargs
        )

    def min(self, *, axis: AxisInt | None = None, skipna: bool = True) -> IntervalOrNA:
        # [FUNCTION DOC]
        # Smallest interval (by left bound, then right bound). Keyword-only: axis
        # (default none; validated, only zero or none allowed) and skipna (default
        # true). An empty array gives the missing value; with missing entries and
        # skipna false the missing value is returned; otherwise missing entries
        # are dropped. Returns an interval scalar or NaN.
        nv.validate_minmax_axis(axis, self.ndim)

        if not len(self):
            return self._na_value

        mask = self.isna()
        if mask.any():
            if not skipna:
                return self._na_value
            obj = self[~mask]
        else:
            obj = self

        indexer = obj.argsort()[0]
        return obj[indexer]

    def max(self, *, axis: AxisInt | None = None, skipna: bool = True) -> IntervalOrNA:
        # [FUNCTION DOC]
        # Largest interval (by left bound, then right bound); same parameters and
        # handling of empty and missing data as the minimum.
        nv.validate_minmax_axis(axis, self.ndim)

        if not len(self):
            return self._na_value

        mask = self.isna()
        if mask.any():
            if not skipna:
                return self._na_value
            obj = self[~mask]
        else:
            obj = self

        indexer = obj.argsort()[-1]
        return obj[indexer]

    def fillna(self, value, limit: int | None = None, copy: bool = True) -> Self:
        # [FUNCTION DOC]
        # Replaces missing entries. value is a single interval or missing value
        # (validated; must match the closed side and bound type); limit (default
        # none) must be none; copy (default true) must be true.
        # Raises a not-implemented error if copy is false and a value error if
        # limit is given. Both bounds are filled with the corresponding bound of
        # the value and a new array is returned; this array is unchanged.
        # KNOWN ISSUE: the docstring says value may be a dict or series, but the
        # scalar validator only accepts an interval or a missing value.
        """
        Fill NA/NaN values using the specified method.

        Parameters
        ----------
        value : scalar, dict, Series
            If a scalar value is passed it is used to fill all missing values.
            Alternatively, a Series or dict can be used to fill in different
            values for each index. The value should not be a list. The
            value(s) passed should be either Interval objects or NA/NaN.
        limit : int, default None
            (Not implemented yet for IntervalArray)
            The maximum number of entries where NA values will be filled.
        copy : bool, default True
            Whether to make a copy of the data before filling. If False, then
            the original should be modified and no new memory should be allocated.
            For ExtensionArray subclasses that cannot do this, it is at the
            author's discretion whether to ignore "copy=False" or to raise.

        Returns
        -------
        filled : IntervalArray with NA/NaN filled
        """
        if copy is False:
            raise NotImplementedError
        if limit is not None:
            raise ValueError("limit must be None")

        value_left, value_right = self._validate_scalar(value)

        left = self.left.fillna(value=value_left)
        right = self.right.fillna(value=value_right)
        return self._shallow_copy(left, right)

    def astype(self, dtype, copy: bool = True):
        # [FUNCTION DOC]
        # Casts to another dtype. dtype is anything resolvable to a dtype (none
        # allowed); copy (default true) matters only for the identical dtype.
        # Interval target: identical dtype returns a copy or self. A float
        # subtype cannot be cast to a datetime-like subtype (type error). Other
        # subtype changes cast both bounds using index casting rules (so NaN
        # into integer subtype raises the integer-cast NaN error unchanged); any
        # other type or value error becomes a type error that the subtypes are
        # incompatible. Returns a new array. Any non-interval target is delegated
        # to the parent cast, converting its type or value errors into a type
        # error naming the class and dtype.
        """
        Cast to an ExtensionArray or NumPy array with dtype 'dtype'.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.

        copy : bool, default True
            Whether to copy the data, even if not necessary. If False,
            a copy is made only if the old dtype does not match the
            new dtype.

        Returns
        -------
        array : ExtensionArray or ndarray
            ExtensionArray or NumPy ndarray with 'dtype' for its dtype.
        """
        from pandas import Index

        if dtype is not None:
            dtype = pandas_dtype(dtype)

        if isinstance(dtype, IntervalDtype):
            if dtype == self.dtype:
                return self.copy() if copy else self

            if is_float_dtype(self.dtype.subtype) and needs_i8_conversion(
                dtype.subtype
            ):
                # This is allowed on the Index.astype but we disallow it here
                msg = (
                    f"Cannot convert {self.dtype} to {dtype}; subtypes are incompatible"
                )
                raise TypeError(msg)

            # need to cast to different subtype
            try:
                # We need to use Index rules for astype to prevent casting
                #  np.nan entries to int subtypes
                new_left = Index(self._left, copy=False).astype(dtype.subtype)
                new_right = Index(self._right, copy=False).astype(dtype.subtype)
            except IntCastingNaNError:
                # e.g test_subtype_integer
                raise
            except (TypeError, ValueError) as err:
                # e.g. test_subtype_integer_errors f8->u8 can be lossy
                #  and raises ValueError
                msg = (
                    f"Cannot convert {self.dtype} to {dtype}; subtypes are incompatible"
                )
                raise TypeError(msg) from err
            return self._shallow_copy(new_left, new_right)
        else:
            try:
                return super().astype(dtype, copy=copy)
            except (TypeError, ValueError) as err:
                msg = f"Cannot cast {type(self).__name__} to dtype {dtype}"
                raise TypeError(msg) from err

    def equals(self, other) -> bool:
        # [FUNCTION DOC]
        # Whether other is an array of exactly the same type with the same
        # closed side and equal left and right bounds (bound equality treats
        # missing values in the same places as equal). Returns a boolean.
        if type(self) != type(other):
            return False

        return bool(
            self.closed == other.closed
            and self.left.equals(other.left)
            and self.right.equals(other.right)
        )

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[IntervalArray]) -> Self:
        # [FUNCTION DOC]
        # HELPER: class-method used by concatenation. to_concat is a sequence of
        # interval arrays that must all have the same closed side (value error
        # otherwise). The left bounds and right bounds are concatenated separately
        # and normalized with the input-ensuring helper (so other errors for
        # incompatible bound types apply). Returns the new array.
        """
        Concatenate multiple IntervalArray

        Parameters
        ----------
        to_concat : sequence of IntervalArray

        Returns
        -------
        IntervalArray
        """
        closed_set = {interval.closed for interval in to_concat}
        if len(closed_set) != 1:
            raise ValueError("Intervals must all be closed on the same side.")
        closed = closed_set.pop()

        left: IntervalSide = np.concatenate([interval.left for interval in to_concat])
        right: IntervalSide = np.concatenate([interval.right for interval in to_concat])

        left, right, dtype = cls._ensure_simple_new_inputs(left, right, closed=closed)

        return cls._simple_new(left, right, dtype=dtype)

    def copy(self) -> Self:
        # [FUNCTION DOC]
        # Returns a new array with copies of both bound arrays and the same dtype
        # (the read-only flag is not carried over).
        """
        Return a copy of the array.

        Returns
        -------
        IntervalArray
        """
        left = self._left.copy()
        right = self._right.copy()
        dtype = self.dtype
        return self._simple_new(left, right, dtype=dtype)

    def isna(self) -> np.ndarray:
        # [FUNCTION DOC]
        # Returns a boolean array, true where the interval is missing. Only the
        # left bound is inspected since missing values are required to coincide
        # in both bounds.
        return isna(self._left)

    def shift(self, periods: int = 1, fill_value: object = None) -> IntervalArray:
        # [FUNCTION DOC]
        # Shifts elements by periods positions (default one; positive moves
        # toward the end, negative toward the start), filling vacated slots with
        # fill_value (default none, meaning missing). A zero shift or an empty
        # array returns a plain copy. The fill value is validated first (type
        # error if it is neither an interval with this closed side nor a missing
        # value). The generic implementation is not used because the dtype's NA
        # value may be wrong and the sequence constructor only takes NaN for
        # missing. A block of fill entries (at most the array length) is built
        # and concatenated with the retained slice. Returns a new array.
        if not len(self) or periods == 0:
            return self.copy()

        self._validate_scalar(fill_value)

        # ExtensionArray.shift doesn't work for two reasons
        # 1. IntervalArray.dtype.na_value may not be correct for the dtype.
        # 2. IntervalArray._from_sequence only accepts NaN for missing values,
        #    not other values like NaT

        empty_len = min(abs(periods), len(self))
        if isna(fill_value):
            from pandas import Index

            fill_value = Index(self._left, copy=False)._na_value
            empty = IntervalArray.from_breaks(
                [fill_value] * (empty_len + 1), closed=self.closed
            )
        else:
            empty = self._from_sequence([fill_value] * empty_len, dtype=self.dtype)

        if periods > 0:
            a = empty
            b = self[:-periods]
        else:
            a = self[abs(periods) :]
            b = empty
        return self._concat_same_type([a, b])

    def take(
        self,
        indices,
        *,
        allow_fill: bool = False,
        fill_value=None,
        axis=None,
        **kwargs,
    ) -> Self:
        # [FUNCTION DOC]
        # Positional take. indices is an integer sequence; keyword-only
        # allow_fill (default false: negatives count from the end; true: minus one
        # means missing), fill_value (default none, meaning missing; validated
        # only when allow_fill is true, must be an interval with this closed side
        # or a missing value), axis (ignored, for index compatibility), and extra
        # keywords validated for numpy compatibility. Takes from each bound with
        # the corresponding fill part and builds a new array. Raises an index
        # error for out-of-bounds positions and a value error for other negative
        # values when filling.
        """
        Take elements from the IntervalArray.

        Parameters
        ----------
        indices : sequence of integers
            Indices to be taken.

        allow_fill : bool, default False
            How to handle negative values in `indices`.

            * False: negative values in `indices` indicate positional indices
              from the right (the default). This is similar to
              :func:`numpy.take`.

            * True: negative values in `indices` indicate
              missing values. These values are set to `fill_value`. Any other
              other negative values raise a ``ValueError``.

        fill_value : Interval or NA, optional
            Fill value to use for NA-indices when `allow_fill` is True.
            This may be ``None``, in which case the default NA value for
            the type, ``self.dtype.na_value``, is used.

            For many ExtensionArrays, there will be two representations of
            `fill_value`: a user-facing "boxed" scalar, and a low-level
            physical NA value. `fill_value` should be the user-facing version,
            and the implementation should handle translating that to the
            physical version for processing the take if necessary.

        axis : any, default None
            Present for compat with IntervalIndex; does nothing.

        Returns
        -------
        IntervalArray

        Raises
        ------
        IndexError
            When the indices are out of bounds for the array.
        ValueError
            When `indices` contains negative values other than ``-1``
            and `allow_fill` is True.
        """
        nv.validate_take((), kwargs)

        fill_left = fill_right = fill_value
        if allow_fill:
            fill_left, fill_right = self._validate_scalar(fill_value)

        left_take = take(
            self._left, indices, allow_fill=allow_fill, fill_value=fill_left
        )
        right_take = take(
            self._right, indices, allow_fill=allow_fill, fill_value=fill_right
        )

        return self._shallow_copy(left_take, right_take)

    def _validate_listlike(self, value):
        # [FUNCTION DOC]
        # HELPER: validates a list-like assignment value. value is converted to
        # an interval array (type error if not interval-like or missing), whose
        # closed side must match (check from the mixin), and its left bound must
        # be holdable by this array's left bound type without loss (type error
        # for incompatible interval types). Returns the pair (left bounds, right
        # bounds) of the converted value.
        # list-like of intervals
        try:
            array = IntervalArray(value)
            self._check_closed_matches(array, name="value")
            value_left, value_right = array.left, array.right
        except TypeError as err:
            # wrong type: not interval or NA
            msg = f"'value' should be an interval type, got {type(value)} instead."
            raise TypeError(msg) from err

        try:
            self.left._validate_fill_value(value_left)
        except (LossySetitemError, TypeError) as err:
            msg = (
                "'value' should be a compatible interval type, "
                f"got {type(value)} instead."
            )
            raise TypeError(msg) from err

        return value_left, value_right

    def _validate_scalar(self, value):
        # [FUNCTION DOC]
        # HELPER: validates a scalar fill/insert value. An interval must have the
        # same closed side (checked by the mixin; error otherwise) and yields its
        # two bounds (the bound subtype is not checked, see the source TODO). A
        # value that is a valid missing marker for the left dtype yields the
        # missing value twice. Anything else raises a type error. Returns the
        # pair (left, right).
        if isinstance(value, Interval):
            self._check_closed_matches(value, name="value")
            left, right = value.left, value.right
            # TODO: check subdtype match like _validate_setitem_value?
        elif is_valid_na_for_dtype(value, self.left.dtype):
            # GH#18295
            left = right = self.left._na_value
        else:
            raise TypeError(
                "can only insert Interval objects and NA into an IntervalArray"
            )
        return left, right

    def _validate_setitem_value(self, value):
        # [FUNCTION DOC]
        # HELPER: validates and splits an assignment value. A missing marker
        # becomes the left dtype's NA in both parts, but raises a type error when
        # the subtype is integer (numpy integer arrays cannot hold NaN). A single
        # interval must match the closed side and both bounds must be holdable by
        # the bound type. Anything else goes through the list-like validation.
        # Returns the pair (left part, right part).
        if is_valid_na_for_dtype(value, self.left.dtype):
            # na value: need special casing to set directly on numpy arrays
            value = self.left._na_value
            if is_integer_dtype(self.dtype.subtype):
                # can't set NaN on a numpy integer array
                # GH#45484 TypeError, not ValueError, matches what we get with
                #  non-NA un-holdable value.
                raise TypeError("Cannot set float NaN to integer-backed IntervalArray")
            value_left, value_right = value, value

        elif isinstance(value, Interval):
            # scalar interval
            self._check_closed_matches(value, name="value")
            value_left, value_right = value.left, value.right
            self.left._validate_fill_value(value_left)
            self.left._validate_fill_value(value_right)

        else:
            return self._validate_listlike(value)

        return value_left, value_right

    # ---------------------------------------------------------------------
    # Rendering Methods

    def _formatter(self, boxed: bool = False) -> Callable[[object], str]:
        # [FUNCTION DOC]
        # HELPER: returns the element display function: the plain text conversion
        # (so intervals show like a bracketed range). boxed is ignored.
        # returning 'str' here causes us to render as e.g. "(0, 1]" instead of
        #  "Interval(0, 1, closed='right')"
        return str

    # ---------------------------------------------------------------------
    # Vectorized Interval Properties/Attributes

    @property
    def left(self) -> Index:
        # [FUNCTION DOC]
        # Property returning the left bounds of all intervals as an index object
        # (wrapping the stored left array).
        # KNOWN ISSUE: the docstring's last sentence says the property returns
        # the midpoint of each interval, but it returns the left endpoints.
        """
        Return the left endpoints of each Interval in the IntervalArray as an Index.

        This property provides access to the left endpoints of the intervals
        contained within the IntervalArray. This can be useful for analyses where
        the starting point of each interval is of interest, such as in histogram
        creation, data aggregation, or any scenario requiring the identification
        of the beginning of defined ranges. This property returns a ``pandas.Index``
        object containing the midpoint for each interval.

        See Also
        --------
        arrays.IntervalArray.right : Return the right endpoints of each Interval in
            the IntervalArray as an Index.
        arrays.IntervalArray.mid : Return the midpoint of each Interval in the
            IntervalArray as an Index.
        arrays.IntervalArray.contains : Check elementwise if the Intervals contain
            the value.

        Examples
        --------

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(2, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (2, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.left
        Index([0, 2], dtype='int64')
        """
        from pandas import Index

        return Index(self._left, copy=False)

    @property
    def right(self) -> Index:
        # [FUNCTION DOC]
        # Property returning the right bounds of all intervals as an index object
        # (wrapping the stored right array without copying).
        """
        Return the right endpoints of each Interval in the IntervalArray as an Index.

        This property extracts the right endpoints from each interval contained within
        the IntervalArray. This can be helpful in use cases where you need to work
        with or compare only the upper bounds of intervals, such as when performing
        range-based filtering, determining interval overlaps, or visualizing the end
        boundaries of data segments.

        See Also
        --------
        arrays.IntervalArray.left : Return the left endpoints of each Interval in
            the IntervalArray as an Index.
        arrays.IntervalArray.mid : Return the midpoint of each Interval in the
            IntervalArray as an Index.
        arrays.IntervalArray.contains : Check elementwise if the Intervals contain
            the value.

        Examples
        --------

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(2, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (2, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.right
        Index([1, 5], dtype='int64')
        """
        from pandas import Index

        return Index(self._right, copy=False)

    @property
    def length(self) -> Index:
        # [FUNCTION DOC]
        # Property returning an index of interval lengths: the right bound minus
        # the left bound for every element (missing for missing intervals;
        # timedelta for datetime-like bounds).
        """
        Return an Index with entries denoting the length of each Interval.

        The length of an interval is calculated as the difference between
        its `right` and `left` bounds. This property is particularly useful
        when working with intervals where the size of the interval is an important
        attribute, such as in time-series analysis or spatial data analysis.

        See Also
        --------
        arrays.IntervalArray.left : Return the left endpoints of each Interval in
            the IntervalArray as an Index.
        arrays.IntervalArray.right : Return the right endpoints of each Interval in
            the IntervalArray as an Index.
        arrays.IntervalArray.mid : Return the midpoint of each Interval in the
            IntervalArray as an Index.

        Examples
        --------

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (1, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.length
        Index([1, 4], dtype='int64')
        """
        return self.right - self.left

    @property
    def mid(self) -> Index:
        # [FUNCTION DOC]
        # Property returning an index of interval midpoints. Normally half the
        # sum of the bounds (a float index for numeric bounds); if that sum is not
        # defined (datetime-like bounds raise a type error), uses the left bound
        # plus half the length instead.
        """
        Return the midpoint of each Interval in the IntervalArray as an Index.

        The midpoint of an interval is calculated as the average of its
        ``left`` and ``right`` bounds. This property returns a ``pandas.Index`` object
        containing the midpoint for each interval.

        See Also
        --------
        Interval.left : Return left bound for the interval.
        Interval.right : Return right bound for the interval.
        Interval.length : Return the length of each interval.

        Examples
        --------

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (1, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.mid
        Index([0.5, 3.0], dtype='float64')
        """
        try:
            return 0.5 * (self.left + self.right)
        except TypeError:
            # datetime safe version
            return self.left + 0.5 * self.length

    _interval_shared_docs["overlaps"] = textwrap.dedent(
        """
        Check elementwise if an Interval overlaps the values in the %(klass)s.

        Two intervals overlap if they share a common point, including closed
        endpoints. Intervals that only have an open endpoint in common do not
        overlap.

        Parameters
        ----------
        other : Interval
            Interval to check against for an overlap.

        Returns
        -------
        ndarray
            Boolean array positionally indicating where an overlap occurs.

        See Also
        --------
        Interval.overlaps : Check whether two Interval objects overlap.

        Examples
        --------
        %(examples)s
        >>> intervals.overlaps(pd.Interval(0.5, 1.5))
        array([ True,  True, False])

        Intervals that share closed endpoints overlap:

        >>> intervals.overlaps(pd.Interval(1, 3, closed='left'))
        array([ True,  True, True])

        Intervals that only have an open endpoint in common do not overlap:

        >>> intervals.overlaps(pd.Interval(1, 2, closed='right'))
        array([False,  True, False])
        """
    )

    def overlaps(self, other):
        # [FUNCTION DOC]
        # Element-wise test whether each interval shares a point with other.
        # other must be a single interval scalar: interval arrays or indexes raise
        # a not-implemented error and any other type raises a type error.
        # Two intervals overlap unless one lies entirely before the other: the
        # comparison of a left bound with the other's right bound uses
        # less-or-equal only when both touching ends are closed, otherwise strict
        # less-than; the two comparisons are combined with logical and. Returns a
        # boolean array (missing entries compare false).
        # KNOWN ISSUE: the parameter is documented as an interval array, but
        # only a single interval scalar is accepted.
        """
        Check elementwise if an Interval overlaps the values in the IntervalArray.

        Two intervals overlap if they share a common point, including closed
        endpoints. Intervals that only have an open endpoint in common do not
        overlap.

        Parameters
        ----------
        other : IntervalArray
            Interval to check against for an overlap.

        Returns
        -------
        ndarray
            Boolean array positionally indicating where an overlap occurs.

        See Also
        --------
        Interval.overlaps : Check whether two Interval objects overlap.

        Examples
        --------
        >>> data = [(0, 1), (1, 3), (2, 4)]
        >>> intervals = pd.arrays.IntervalArray.from_tuples(data)
        >>> intervals
        <IntervalArray>
        [(0, 1], (1, 3], (2, 4]]
        Length: 3, dtype: interval[int64, right]

        >>> intervals.overlaps(pd.Interval(0.5, 1.5))
        array([ True,  True, False])

        Intervals that share closed endpoints overlap:

        >>> intervals.overlaps(pd.Interval(1, 3, closed="left"))
        array([ True,  True, True])

        Intervals that only have an open endpoint in common do not overlap:

        >>> intervals.overlaps(pd.Interval(1, 2, closed="right"))
        array([False,  True, False])
        """
        if isinstance(other, (IntervalArray, ABCIntervalIndex)):
            raise NotImplementedError
        if not isinstance(other, Interval):
            msg = f"`other` must be Interval-like, got {type(other).__name__}"
            raise TypeError(msg)

        # equality is okay if both endpoints are closed (overlap at a point)
        op1 = le if (self.closed_left and other.closed_right) else lt
        op2 = le if (other.closed_left and self.closed_right) else lt

        # overlaps is equivalent negation of two interval being disjoint:
        # disjoint = (A.left > B.right) or (B.left > A.right)
        # (simplifying the negation allows this to be done in less operations)
        return op1(self.left, other.right) & op2(other.left, self.right)

    # ---------------------------------------------------------------------

    @property
    def closed(self) -> IntervalClosedType:
        # [FUNCTION DOC]
        # Property returning the closed side (left, right, both or neither), read
        # from the interval dtype.
        """
        String describing the inclusive side the intervals.

        Either ``left``, ``right``, ``both`` or ``neither``.

        See Also
        --------
        IntervalArray.closed : Returns inclusive side of the IntervalArray.
        Interval.closed : Returns inclusive side of the Interval.
        IntervalIndex.closed : Returns inclusive side of the IntervalIndex.

        Examples
        --------

        For arrays:

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (1, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.closed
        'right'

        For Interval Index:

        >>> interv_idx = pd.interval_range(start=0, end=2)
        >>> interv_idx
        IntervalIndex([(0, 1], (1, 2]], dtype='interval[int64, right]')
        >>> interv_idx.closed
        'right'
        """
        return self.dtype.closed

    _interval_shared_docs["set_closed"] = textwrap.dedent(
        """
        Return an identical %(klass)s closed on the specified side.

        Parameters
        ----------
        closed : {'left', 'right', 'both', 'neither'}
            Whether the intervals are closed on the left-side, right-side, both
            or neither.

        Returns
        -------
        %(klass)s

        %(examples)s\
        """
    )

    def set_closed(self, closed: IntervalClosedType) -> Self:
        # [FUNCTION DOC]
        # Returns a new array with the same bounds but a different closed side.
        # closed must be one of left, right, both, neither (value error
        # otherwise). The bound arrays are shared with this array (not copied)
        # and only the dtype is replaced; no overlap or ordering validation is
        # needed.
        """
        Return an identical IntervalArray closed on the specified side.

        Parameters
        ----------
        closed : {'left', 'right', 'both', 'neither'}
            Whether the intervals are closed on the left-side, right-side, both
            or neither.

        Returns
        -------
        IntervalArray
            A new IntervalArray with the specified side closures.

        See Also
        --------
        IntervalArray.closed : Returns inclusive side of the Interval.
        arrays.IntervalArray.closed : Returns inclusive side of the IntervalArray.

        Examples
        --------
        >>> index = pd.arrays.IntervalArray.from_breaks(range(4))
        >>> index
        <IntervalArray>
        [(0, 1], (1, 2], (2, 3]]
        Length: 3, dtype: interval[int64, right]
        >>> index.set_closed("both")
        <IntervalArray>
        [[0, 1], [1, 2], [2, 3]]
        Length: 3, dtype: interval[int64, both]
        """
        if closed not in VALID_CLOSED:
            msg = f"invalid option for 'closed': {closed}"
            raise ValueError(msg)

        left, right = self._left, self._right
        dtype = IntervalDtype(left.dtype, closed=closed)
        return self._simple_new(left, right, dtype=dtype)

    _interval_shared_docs["is_non_overlapping_monotonic"] = """
        Return a boolean whether the %(klass)s is non-overlapping and monotonic.

        Non-overlapping means (no Intervals share points), and monotonic means
        either monotonic increasing or monotonic decreasing.

        Examples
        --------
        For arrays:

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (1, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.is_non_overlapping_monotonic
        True

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1),
        ...                                       pd.Interval(-1, 0.1)])
        >>> interv_arr
        <IntervalArray>
        [(0.0, 1.0], (-1.0, 0.1]]
        Length: 2, dtype: interval[float64, right]
        >>> interv_arr.is_non_overlapping_monotonic
        False

        For Interval Index:

        >>> interv_idx = pd.interval_range(start=0, end=2)
        >>> interv_idx
        IntervalIndex([(0, 1], (1, 2]], dtype='interval[int64, right]')
        >>> interv_idx.is_non_overlapping_monotonic
        True

        >>> interv_idx = pd.interval_range(start=0, end=2, closed='both')
        >>> interv_idx
        IntervalIndex([[0, 1], [1, 2]], dtype='interval[int64, both]')
        >>> interv_idx.is_non_overlapping_monotonic
        False
        """

    @property
    def is_non_overlapping_monotonic(self) -> bool:
        # [FUNCTION DOC]
        # Property: true if no two intervals share a point and the sequence is
        # monotonic (either increasing or decreasing), comparing consecutive
        # elements only. For a both-closed array consecutive bounds must be
        # strictly separated; otherwise touching bounds are allowed. Returns a
        # plain boolean; arrays of length zero or one are true. Missing entries
        # are not treated specially (their comparisons are false).
        """
        Return a boolean whether the IntervalArray/IntervalIndex\
        is non-overlapping and monotonic.

        Non-overlapping means (no Intervals share points), and monotonic means
        either monotonic increasing or monotonic decreasing.

        See Also
        --------
        overlaps : Check if two IntervalIndex objects overlap.

        Examples
        --------
        For arrays:

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (1, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.is_non_overlapping_monotonic
        True

        >>> interv_arr = pd.arrays.IntervalArray(
        ...     [pd.Interval(0, 1), pd.Interval(-1, 0.1)]
        ... )
        >>> interv_arr
        <IntervalArray>
        [(0.0, 1.0], (-1.0, 0.1]]
        Length: 2, dtype: interval[float64, right]
        >>> interv_arr.is_non_overlapping_monotonic
        False

        For Interval Index:

        >>> interv_idx = pd.interval_range(start=0, end=2)
        >>> interv_idx
        IntervalIndex([(0, 1], (1, 2]], dtype='interval[int64, right]')
        >>> interv_idx.is_non_overlapping_monotonic
        True

        >>> interv_idx = pd.interval_range(start=0, end=2, closed="both")
        >>> interv_idx
        IntervalIndex([[0, 1], [1, 2]], dtype='interval[int64, both]')
        >>> interv_idx.is_non_overlapping_monotonic
        False
        """
        # must be increasing  (e.g., [0, 1), [1, 2), [2, 3), ... )
        # or decreasing (e.g., [-1, 0), [-2, -1), [-3, -2), ...)
        # we already require left <= right

        # strict inequality for closed == 'both'; equality implies overlapping
        # at a point when both sides of intervals are included
        if self.closed == "both":
            return bool(
                (self._right[:-1] < self._left[1:]).all()
                or (self._left[:-1] > self._right[1:]).all()
            )

        # non-strict inequality when closed != 'both'; at least one side is
        # not included in the intervals, so equality does not imply overlapping
        return bool(
            (self._right[:-1] <= self._left[1:]).all()
            or (self._left[:-1] >= self._right[1:]).all()
        )

    # ---------------------------------------------------------------------
    # Conversion

    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        # [FUNCTION DOC]
        # Numpy array protocol. dtype is the requested dtype and copy is none,
        # true or false.
        # Raises a value error if copy is false (a new object array is always
        # built). Builds an object array where each element is an interval scalar
        # with this array's closed side, or NaN where missing.
        # KNOWN ISSUE: the dtype argument is never used, so the result is always
        # object dtype regardless of the request.
        """
        Return the IntervalArray's data as a numpy array of Interval
        objects (with dtype='object')
        """
        if copy is False:
            raise ValueError(
                "Unable to avoid copy while creating an array as requested."
            )

        left = self._left
        right = self._right
        mask = self.isna()
        closed = self.closed

        result = np.empty(len(left), dtype=object)
        for i, left_value in enumerate(left):
            if mask[i]:
                result[i] = np.nan
            else:
                result[i] = Interval(left_value, right[i], closed)
        return result

    def __arrow_array__(self, type=None):
        # [FUNCTION DOC]
        # Arrow export protocol (pyarrow imported lazily). type (default none) is
        # the requested arrow type.
        # The bound subtype is converted to the arrow type (a type error if not
        # supported). Storage is a struct array with two fields (left and right
        # bounds); if any intervals are missing the struct-level validity bitmap
        # is set too. If type is the plain storage struct type, returns the
        # storage; if it is the interval extension type it must equal the one
        # derived from this array (type error on subtype or closed mismatch);
        # any other type raises a type error. Otherwise returns an arrow
        # extension array of the interval extension type wrapping the storage.
        """
        Convert myself into a pyarrow Array.
        """
        import pyarrow

        from pandas.core.arrays.arrow.extension_types import ArrowIntervalType

        try:
            subtype = pyarrow.from_numpy_dtype(self.dtype.subtype)
        except TypeError as err:
            raise TypeError(
                f"Conversion to arrow with subtype '{self.dtype.subtype}' "
                "is not supported"
            ) from err
        interval_type = ArrowIntervalType(subtype, self.closed)
        storage_array = pyarrow.StructArray.from_arrays(
            [
                pyarrow.array(self._left, type=subtype, from_pandas=True),
                pyarrow.array(self._right, type=subtype, from_pandas=True),
            ],
            names=["left", "right"],
        )
        mask = self.isna()
        if mask.any():
            # if there are missing values, set validity bitmap also on the array level
            null_bitmap = pyarrow.array(~mask).buffers()[1]
            storage_array = pyarrow.StructArray.from_buffers(
                storage_array.type,
                len(storage_array),
                [null_bitmap],
                children=[storage_array.field(0), storage_array.field(1)],
            )

        if type is not None:
            if type.equals(interval_type.storage_type):
                return storage_array
            elif isinstance(type, ArrowIntervalType):
                # ensure we have the same subtype and closed attributes
                if not type.equals(interval_type):
                    raise TypeError(
                        "Not supported to convert IntervalArray to type with "
                        f"different 'subtype' ({self.dtype.subtype} vs {type.subtype}) "
                        f"and 'closed' ({self.closed} vs {type.closed}) attributes"
                    )
            else:
                raise TypeError(
                    f"Not supported to convert IntervalArray to '{type}' type"
                )

        return pyarrow.ExtensionArray.from_storage(interval_type, storage_array)

    _interval_shared_docs["to_tuples"] = textwrap.dedent(
        """
        Return an %(return_type)s of tuples of the form (left, right).

        Parameters
        ----------
        na_tuple : bool, default True
            If ``True``, return ``NA`` as a tuple ``(nan, nan)``. If ``False``,
            just return ``NA`` as ``nan``.

        Returns
        -------
        tuples: %(return_type)s
        %(examples)s\
        """
    )

    def to_tuples(self, na_tuple: bool = True) -> np.ndarray:
        # [FUNCTION DOC]
        # Converts to an object array of (left, right) tuples. na_tuple (default
        # true) keeps missing intervals as a tuple of two NaN values; when false
        # they become a single NaN. Returns an object ndarray of tuples (elements
        # are numpy scalars for numeric bounds). The result is built by pairing
        # the two bound arrays element by element (equal lengths enforced).
        """
        Return an ndarray (if self is IntervalArray) or Index \
        (if self is IntervalIndex) of tuples of the form (left, right).

        Parameters
        ----------
        na_tuple : bool, default True
            If ``True``, return ``NA`` as a tuple ``(nan, nan)``. If ``False``,
            just return ``NA`` as ``nan``.

        Returns
        -------
        ndarray or Index
            An ndarray of tuples representing the intervals
                if `self` is an IntervalArray.
            An Index of tuples representing the intervals
                if `self` is an IntervalIndex.

        See Also
        --------
        IntervalArray.to_list : Convert IntervalArray to a list of tuples.
        IntervalArray.to_numpy : Convert IntervalArray to a numpy array.
        IntervalArray.unique : Find unique intervals in an IntervalArray.

        Examples
        --------
        For :class:`pandas.IntervalArray`:

        >>> idx = pd.arrays.IntervalArray.from_tuples([(0, 1), (1, 2)])
        >>> idx
        <IntervalArray>
        [(0, 1], (1, 2]]
        Length: 2, dtype: interval[int64, right]
        >>> idx.to_tuples()
        array([(np.int64(0), np.int64(1)), (np.int64(1), np.int64(2))],
              dtype=object)

        For :class:`pandas.IntervalIndex`:

        >>> idx = pd.interval_range(start=0, end=2)
        >>> idx
        IntervalIndex([(0, 1], (1, 2]], dtype='interval[int64, right]')
        >>> idx.to_tuples()
        Index([(0, 1), (1, 2)], dtype='object')
        """
        tuples = com.asarray_tuplesafe(zip(self._left, self._right, strict=True))
        if not na_tuple:
            # GH 18756
            tuples = np.where(~self.isna(), tuples, np.nan)
        return tuples

    # ---------------------------------------------------------------------

    def _putmask(self, mask: npt.NDArray[np.bool_], value) -> None:
        # [FUNCTION DOC]
        # HELPER: in-place conditional assignment. mask is a boolean array of
        # positions; value is an interval, list-like of intervals or missing
        # value (validated and split into left and right parts, with the usual
        # type errors). Both bound arrays are updated in place with the
        # corresponding parts, using numpy masked assignment for plain arrays and
        # the datetime-like array's own method otherwise. Mutates this array.
        value_left, value_right = self._validate_setitem_value(value)

        if isinstance(self._left, np.ndarray):
            np.putmask(self._left, mask, value_left)
            assert isinstance(self._right, np.ndarray)
            np.putmask(self._right, mask, value_right)
        else:
            self._left._putmask(mask, value_left)
            assert not isinstance(self._right, np.ndarray)
            self._right._putmask(mask, value_right)

    def insert(self, loc: int, item: Interval) -> Self:
        # [FUNCTION DOC]
        # Returns a new array with item inserted before position loc (negative
        # positions count from the end). item must be an interval with this
        # closed side or a missing value (type error otherwise). Insertion is done
        # on each bound index separately and the results are wrapped in a new
        # array. This array is unchanged.
        """
        Return a new IntervalArray inserting new item at location. Follows
        Python numpy.insert semantics for negative values.  Only Interval
        objects and NA can be inserted into an IntervalIndex

        Parameters
        ----------
        loc : int
        item : Interval

        Returns
        -------
        IntervalArray
        """
        left_insert, right_insert = self._validate_scalar(item)

        new_left = self.left.insert(loc, left_insert)
        new_right = self.right.insert(loc, right_insert)

        return self._shallow_copy(new_left, new_right)

    def delete(self, loc) -> Self:
        # [FUNCTION DOC]
        # Returns a new array with the element(s) at loc (an integer or sequence
        # or slice of positions) removed from both bounds. Uses numpy deletion for
        # plain arrays and the array's own delete otherwise. Out-of-bounds
        # positions raise the underlying error. This array is unchanged.
        new_left: np.ndarray | DatetimeArray | TimedeltaArray
        new_right: np.ndarray | DatetimeArray | TimedeltaArray
        if isinstance(self._left, np.ndarray):
            new_left = np.delete(self._left, loc)
            assert isinstance(self._right, np.ndarray)
            new_right = np.delete(self._right, loc)
        else:
            new_left = self._left.delete(loc)
            assert not isinstance(self._right, np.ndarray)
            new_right = self._right.delete(loc)
        return self._shallow_copy(left=new_left, right=new_right)

    def repeat(
        self,
        repeats: int | Sequence[int],
        axis: AxisInt | None = None,
    ) -> Self:
        # [FUNCTION DOC]
        # Repeats each element. repeats is a non-negative integer or a sequence
        # of integers (one per element); axis (default none) must be none (the
        # numpy-compat validator raises otherwise). Each bound is repeated
        # separately and a new array is built. Returns the new array.
        # KNOWN ISSUE: the docstring example demonstrates a categorical array
        # rather than an interval array.
        """
        Repeat elements of an IntervalArray.

        Returns a new IntervalArray where each element of the current IntervalArray
        is repeated consecutively a given number of times.

        Parameters
        ----------
        repeats : int or array of ints
            The number of repetitions for each element. This should be a
            non-negative integer. Repeating 0 times will return an empty
            IntervalArray.
        axis : None
            Must be ``None``. Has no effect but is accepted for compatibility
            with numpy.

        Returns
        -------
        IntervalArray
            Newly created IntervalArray with repeated elements.

        See Also
        --------
        Series.repeat : Equivalent function for Series.
        Index.repeat : Equivalent function for Index.
        numpy.repeat : Similar method for :class:`numpy.ndarray`.
        ExtensionArray.take : Take arbitrary positions.

        Examples
        --------
        >>> cat = pd.Categorical(["a", "b", "c"])
        >>> cat
        ['a', 'b', 'c']
        Categories (3, str): ['a', 'b', 'c']
        >>> cat.repeat(2)
        ['a', 'a', 'b', 'b', 'c', 'c']
        Categories (3, str): ['a', 'b', 'c']
        >>> cat.repeat([1, 2, 3])
        ['a', 'b', 'b', 'c', 'c', 'c']
        Categories (3, str): ['a', 'b', 'c']
        """
        nv.validate_repeat((), {"axis": axis})
        left_repeat = self.left.repeat(repeats)
        right_repeat = self.right.repeat(repeats)
        return self._shallow_copy(left=left_repeat, right=right_repeat)

    _interval_shared_docs["contains"] = textwrap.dedent(
        """
        Check elementwise if the Intervals contain the value.

        Return a boolean mask whether the value is contained in the Intervals
        of the %(klass)s.

        Parameters
        ----------
        other : scalar
            The value to check whether it is contained in the Intervals.

        Returns
        -------
        boolean array

        See Also
        --------
        Interval.contains : Check whether Interval object contains value.
        %(klass)s.overlaps : Check if an Interval overlaps the values in the
            %(klass)s.

        Examples
        --------
        %(examples)s
        >>> intervals.contains(0.5)
        array([ True, False, False])
    """
    )

    def contains(self, other):
        # [FUNCTION DOC]
        # Element-wise membership test of a scalar value inside each interval.
        # other is a scalar comparable with the bounds; an interval scalar raises
        # a not-implemented error. The left test is strict when the left side is
        # open and non-strict when closed; the right test is likewise. Returns a
        # boolean array combining both tests with logical and (missing entries
        # compare false).
        """
        Check elementwise if the Intervals contain the value.

        Return a boolean mask whether the value is contained in the Intervals
        of the IntervalArray.

        Parameters
        ----------
        other : scalar
            The value to check whether it is contained in the Intervals.

        Returns
        -------
        boolean array
            A boolean mask whether the value is contained in the Intervals.

        See Also
        --------
        Interval.contains : Check whether Interval object contains value.
        IntervalArray.overlaps : Check if an Interval overlaps the values in the
            IntervalArray.

        Examples
        --------
        >>> intervals = pd.arrays.IntervalArray.from_tuples([(0, 1), (1, 3), (2, 4)])
        >>> intervals
        <IntervalArray>
        [(0, 1], (1, 3], (2, 4]]
        Length: 3, dtype: interval[int64, right]

        >>> intervals.contains(0.5)
        array([ True, False, False])
        """
        if isinstance(other, Interval):
            raise NotImplementedError("contains not implemented for two intervals")

        return (self._left < other if self.open_left else self._left <= other) & (
            other < self._right if self.open_right else other <= self._right
        )

    def isin(self, values: ArrayLike) -> npt.NDArray[np.bool_]:
        # [FUNCTION DOC]
        # Membership test. values is an array-like of candidates. Returns a
        # boolean array of this array's length.
        # For an interval array with a different closed side, or whose bound types
        # are not comparable (one datetime-like, the other not), returns all
        # false. With identical dtype it compares the combined complex-number
        # representation of the bounds using numpy membership. Otherwise (and for
        # non-interval candidates) both sides are converted to object arrays and
        # compared with the generic membership routine.
        if isinstance(values, IntervalArray):
            if self.closed != values.closed:
                # not comparable -> no overlap
                return np.zeros(self.shape, dtype=bool)

            if self.dtype == values.dtype:
                left = self._combined
                right = values._combined
                return np.isin(left, right).ravel()

            elif needs_i8_conversion(self.left.dtype) ^ needs_i8_conversion(
                values.left.dtype
            ):
                # not comparable -> no overlap
                return np.zeros(self.shape, dtype=bool)

        return isin(self.astype(object), values.astype(object))

    @property
    def _combined(self) -> IntervalSide:
        # [FUNCTION DOC]
        # HELPER: property packing each (left, right) pair into one value so that
        # sets of intervals can be compared or de-duplicated quickly. Returns a
        # one-dimensional array of complex numbers, with the left bound as the
        # real part and the right bound as the imaginary part; for datetime-like
        # bounds the two columns are concatenated and their raw integer memory
        # is reinterpreted as complex values. Used by the membership and unique
        # operations and inverted by the matching rebuild helper.
        # error: Item "ExtensionArray" of "ExtensionArray | ndarray[Any, Any]"
        # has no attribute "reshape"  [union-attr]
        left = self.left._values.reshape(-1, 1)  # type: ignore[union-attr]
        right = self.right._values.reshape(-1, 1)  # type: ignore[union-attr]
        # GH#38353 instead of casting to object, operating on a
        # complex128 ndarray is much more performant.
        if needs_i8_conversion(left.dtype):
            # error: Item "ndarray[Any, Any]" of "Any | ndarray[Any, Any]" has
            # no attribute "_concat_same_type"
            comb = left._concat_same_type(  # type: ignore[union-attr]
                [left, right], axis=1
            )
            comb = comb.view("complex128")[:, 0]
        else:
            comb = np.asarray(left.ravel(), dtype="complex128")
            comb.imag = right.ravel()
        return comb

    def _from_combined(self, combined: np.ndarray) -> IntervalArray:
        # [FUNCTION DOC]
        # HELPER: inverse of the combined representation. combined is a complex
        # array (shape of one column per element, as produced by the combine
        # step after de-duplication). For datetime-like bounds the memory is
        # reinterpreted as pairs of 64-bit integers and each column is converted
        # back to the bound array type with the original dtype; for plain numeric
        # bounds the real and imaginary parts are cast to the original numpy
        # dtype and flattened. Returns a new array via the shallow-copy helper
        # (same closed side). Assertions guard the bound container types.
        """
        Create a new IntervalArray with our dtype from a 1D complex128 ndarray.
        """

        dtype = self._left.dtype
        if needs_i8_conversion(dtype):
            nc = combined.view("i8").reshape(-1, 2)
            assert isinstance(self._left, (DatetimeArray, TimedeltaArray))
            new_left: DatetimeArray | TimedeltaArray | np.ndarray = type(
                self._left
            )._from_sequence(nc[:, 0], dtype=dtype)
            assert isinstance(self._right, (DatetimeArray, TimedeltaArray))
            new_right: DatetimeArray | TimedeltaArray | np.ndarray = type(
                self._right
            )._from_sequence(nc[:, 1], dtype=dtype)
        else:
            assert isinstance(dtype, np.dtype)
            new_left = np.real(combined).astype(dtype).ravel()
            new_right = np.imag(combined).astype(dtype).ravel()
        return self._shallow_copy(left=new_left, right=new_right)

    def unique(self) -> IntervalArray:
        # [FUNCTION DOC]
        # Returns a new array of the distinct intervals in order of first
        # appearance, by de-duplicating the combined complex representation and
        # rebuilding the bounds. This array is unchanged.
        nc = unique(self._combined)
        return self._from_combined(np.asarray(nc)[:, None])


def _maybe_convert_platform_interval(values) -> ArrayLike:
    # [FUNCTION DOC]
    # HELPER: normalizes constructor input to an array-like with a suitable
    # dtype. values is any input. Empty lists or tuples become an empty 64-bit
    # integer array (object dtype is prohibited for interval bounds). Non-list-
    # like inputs and data frames are returned unchanged (they fail later).
    # Categorical-typed input becomes a numpy array. Inputs that are not list,
    # tuple, range, or dtype-carrying objects are returned unchanged. Other
    # inputs are unwrapped to arrays; if still without a dtype they are
    # converted to numpy, with any integer type other than 64-bit widened to
    # 64-bit. Returns the resulting array-like.
    # KNOWN ISSUE: the docstring says this wraps the platform conversion helper,
    # but that helper is never called; the conversion is done directly.
    """
    Try to do platform conversion, with special casing for IntervalArray.
    Wrapper around maybe_convert_platform that alters the default return
    dtype in certain cases to be compatible with IntervalArray.  For example,
    empty lists return with integer dtype instead of object dtype, which is
    prohibited for IntervalArray.

    Parameters
    ----------
    values : array-like

    Returns
    -------
    array
    """
    if isinstance(values, (list, tuple)) and len(values) == 0:
        # GH 19016
        # empty lists/tuples get object dtype by default, but this is
        # prohibited for IntervalArray, so coerce to integer instead
        return np.array([], dtype=np.int64)
    elif not is_list_like(values) or isinstance(values, ABCDataFrame):
        # This will raise later, but we avoid passing to maybe_convert_platform
        return values
    elif isinstance(getattr(values, "dtype", None), CategoricalDtype):
        values = np.asarray(values)
    elif not hasattr(values, "dtype") and not isinstance(values, (list, tuple, range)):
        # TODO: should we just cast these to list?
        return values
    else:
        values = extract_array(values, extract_numpy=True)

    if not hasattr(values, "dtype"):
        values = np.asarray(values)
        if values.dtype.kind in "iu" and values.dtype != np.int64:
            values = values.astype(np.int64)
    return values
