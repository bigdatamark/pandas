# pyright: reportPropertyTypeMismatch=false
# [MODULE DOC]
# Defines NDFrame, the shared base class of Series and DataFrame, plus a few
# module-level helpers.
# NDFrame holds a single internal data manager (the block or array manager), a
# free-form attribute dictionary and a flags object, and implements everything that is
# common to one- and N-dimensional labeled containers: axis bookkeeping, renaming,
# reindexing, sorting, selection helpers (head, tail, sample, filter, xs, take, get),
# missing-data handling (fillna, ffill, bfill, replace, interpolate, isna family,
# dropna-related helpers), type conversion (astype, convert_dtypes, infer_objects),
# alignment, where/mask, shift, time-series helpers (resample, asfreq, tz handling,
# at_time, between_time), reductions and accumulations, window entry points (rolling,
# expanding, ewm), in-place arithmetic operators, pickling, and the whole family of
# to_* serialization methods that delegate to the pandas.io modules.
# Concrete subclasses (Series in pandas.core.series, DataFrame in pandas.core.frame)
# supply the axis metadata (axis order, axis lengths, info axis) and the constructor
# hooks that this module only declares abstractly.
# Module level contents: a private copy of the shared docstring templates, a
# dictionary of default docstring substitution values used by the doc decorator, the
# NDFrame class, and a function that builds the docstring for a reduction or
# accumulation by name.
# Import-time side effects: makes a private copy of the shared docstring dictionary
# and builds a number of docstring template strings for reductions and accumulations.
# Many methods are wrapped by docstring-substitution decorators at class creation
# time.
# KNOWN ISSUE: the documentation generator function at the bottom of the module and
# several of the module-level reduction docstring templates are not referenced by any
# code in the repository, and the reduction methods defined here carry no docstrings
# of their own.
from __future__ import annotations

import collections
from copy import deepcopy
import datetime as dt
from functools import partial
from json import loads
import operator
import pickle
import re
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Concatenate,
    Literal,
    NoReturn,
    Self,
    cast,
    final,
    overload,
)
import warnings

import numpy as np

from pandas._config import config

from pandas._libs import lib
from pandas._libs.lib import is_range_indexer
from pandas._libs.tslibs import (
    Period,
    Timestamp,
    to_offset,
)
from pandas._typing import (
    AlignJoin,
    AnyArrayLike,
    ArrayLike,
    Axes,
    Axis,
    AxisInt,
    CompressionOptions,
    DtypeArg,
    DtypeBackend,
    DtypeObj,
    FilePath,
    FillnaOptions,
    FloatFormatType,
    FormattersType,
    Frequency,
    IgnoreRaise,
    IndexKeyFunc,
    IndexLabel,
    InterpolateOptions,
    IntervalClosedType,
    JSONSerializable,
    Level,
    ListLike,
    Manager,
    NaPosition,
    NDFrameT,
    OpenFileErrors,
    RandomState,
    ReindexMethod,
    Renamer,
    Scalar,
    SequenceNotStr,
    SortKind,
    StorageOptions,
    Suffixes,
    T,
    TimeAmbiguous,
    TimedeltaConvertibleTypes,
    TimeNonexistent,
    TimestampConvertibleTypes,
    TimeUnit,
    ValueKeyFunc,
    WriteBuffer,
    WriteExcelBuffer,
    npt,
)
from pandas.compat import CHAINED_WARNING_DISABLED
from pandas.compat._constants import (
    REF_COUNT_METHOD,
)
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import (
    AbstractMethodError,
    ChainedAssignmentError,
    InvalidIndexError,
    Pandas4Warning,
)
from pandas.errors.cow import _chained_assignment_method_msg
from pandas.util._decorators import (
    deprecate_kwarg,
    doc,
)
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
    check_dtype_backend,
    validate_ascending,
    validate_bool_kwarg,
    validate_inclusive,
)

from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.cast import can_hold_element
from pandas.core.dtypes.common import (
    ensure_object,
    ensure_platform_int,
    ensure_str,
    is_bool,
    is_bool_dtype,
    is_dict_like,
    is_extension_array_dtype,
    is_list_like,
    is_number,
    is_numeric_dtype,
    is_re_compilable,
    is_scalar,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    ExtensionDtype,
    PeriodDtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)
from pandas.core.dtypes.inference import (
    is_hashable,
    is_nested_list_like,
)
from pandas.core.dtypes.missing import (
    isna,
    notna,
)

from pandas.core import (
    algorithms as algos,
    arraylike,
    common,
    indexing,
    missing,
    nanops,
    sample,
)
from pandas.core.array_algos.replace import should_use_regex
from pandas.core.arrays import ExtensionArray
from pandas.core.base import PandasObject
from pandas.core.construction import extract_array
from pandas.core.flags import Flags
from pandas.core.indexes.api import (
    DatetimeIndex,
    Index,
    MultiIndex,
    PeriodIndex,
    default_index,
    ensure_index,
)
from pandas.core.internals import BlockManager
from pandas.core.methods.describe import describe_ndframe
from pandas.core.missing import (
    clean_fill_method,
    clean_reindex_fill_method,
    find_valid_index,
)
from pandas.core.reshape.concat import concat
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import get_indexer_indexer
from pandas.core.window import (
    Expanding,
    ExponentialMovingWindow,
    Rolling,
    Window,
)

from pandas.io.formats.format import (
    DataFrameFormatter,
    DataFrameRenderer,
)
from pandas.io.formats.printing import pprint_thing

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
        Iterator,
        Mapping,
        Sequence,
    )

    from pandas._libs.tslibs import BaseOffset
    from pandas._typing import P

    from pandas import (
        DataFrame,
        ExcelWriter,
        HDFStore,
        Series,
    )
    from pandas.core.indexers.objects import BaseIndexer
    from pandas.core.resample import Resampler


# goal is to be able to define the docs close to function, while still being
# able to share
_shared_docs = {**_shared_docs}
_shared_doc_kwargs = {
    "axes": "keywords for axes",
    "klass": "Series/DataFrame",
    "axes_single_arg": "{0 or 'index'} for Series, {0 or 'index', 1 or 'columns'} for DataFrame",  # noqa: E501
    "inplace": """
    inplace : bool, default False
        If True, performs operation inplace.""",
    "optional_by": """
        by : str or list of str
            Name or list of names to sort by""",
}


class NDFrame(PandasObject, indexing.IndexingMixin):
    # [CLASS DOC]
    # Base class for Series and DataFrame; inherits the generic object behavior from
    # PandasObject and the label and position based indexers from IndexingMixin.
    # State: the internal manager that stores the data (set through direct object
    # attribute assignment to bypass the custom attribute hooks), a dictionary of user
    # attributes, and a flags object bound to the instance. Class level attributes
    # list the internal attribute names, the accessor names, hidden attribute names
    # and the metadata attribute names propagated by the finalize hook; subclasses
    # override the axis description attributes (axis order, axis-name to number
    # mapping, info axis, number of axes).
    # The class is abstract in the sense that the constructor-related properties raise
    # an abstract-method error until a subclass overrides them.
    # Invariants: instances are unhashable, the array priority is high so that numpy
    # defers binary operations to pandas, and truth-value testing always raises.
    # Key method groups: construction and axis helpers, rename, equality, unary
    # operators, label-or-level helpers, iteration, pickling, rendering, I/O (to_*),
    # selection, reindex, sort, fill, replace, astype, time series, reductions,
    # windowing, in-place operators.
    """
    N-dimensional analogue of DataFrame. Store multi-dimensional in a
    size-mutable, labeled data structure

    Parameters
    ----------
    data : BlockManager
    axes : list
    copy : bool, default False
    """

    _internal_names: list[str] = [
        "_mgr",
        "_cache",
        "_name",
        "_metadata",
        "_flags",
    ]
    _internal_names_set: set[str] = set(_internal_names)
    _accessors: set[str] = set()
    _hidden_attrs: frozenset[str] = frozenset([])
    _metadata: list[str] = []
    _mgr: Manager
    _attrs: dict[Hashable, Any]
    _typ: str

    # ----------------------------------------------------------------------
    # Constructors

    def __init__(self, data: Manager) -> None:
        # [FUNCTION DOC]
        # Initialize the instance with an already-built data manager.
        # Parameters: data is the block or array manager to wrap (no validation or
        # copy is made).
        # Sets, by writing directly to the object dictionary to avoid the custom
        # attribute setter, the manager, an empty attribute dictionary, and a flags
        # object that allows duplicate labels.
        # Subclasses are expected to call this (or the from-manager constructor, which
        # calls it on a bare instance).
        object.__setattr__(self, "_mgr", data)
        object.__setattr__(self, "_attrs", {})
        object.__setattr__(self, "_flags", Flags(self, allows_duplicate_labels=True))

    @final
    @classmethod
    def _init_mgr(
        cls,
        mgr: Manager,
        axes: dict[Literal["index", "columns"], Axes | None],
        dtype: DtypeObj | None = None,
        copy: bool = False,
    ) -> Manager:
        # [FUNCTION DOC]
        # HELPER: used by the Series and DataFrame constructors to normalize a
        # manager.
        # Parameters: mgr is the manager to adjust; axes is a mapping from axis name
        # (index or columns) to labels or None; dtype is an optional dtype to coerce
        # to; copy says whether to deep copy the manager first.
        # For each axis with non-None labels the labels are converted to an Index and
        # the manager is reindexed along the matching manager axis (translated from
        # the user axis). If copy is true the manager is deep-copied. If dtype is
        # given, the manager is cast to it unless it is a block manager with exactly
        # one block already having that dtype.
        # Returns the resulting manager. No mutation of the input manager besides what
        # reindex and astype return as new objects.
        """passed a manager and a axes dict"""
        for a, axe in axes.items():
            if axe is not None:
                axe = ensure_index(axe)
                bm_axis = cls._get_block_manager_axis(a)
                mgr = mgr.reindex_axis(axe, axis=bm_axis)

        # make a copy if explicitly requested
        if copy:
            mgr = mgr.copy(deep=True)
        if dtype is not None:
            # avoid further copies if we can
            if (
                isinstance(mgr, BlockManager)
                and len(mgr.blocks) == 1
                and mgr.blocks[0].values.dtype == dtype
            ):
                pass
            else:
                mgr = mgr.astype(dtype=dtype)
        return mgr

    @final
    @classmethod
    def _from_mgr(cls, mgr: Manager, axes: list[Index]) -> Self:
        # [FUNCTION DOC]
        # HELPER: Class method that builds a new instance of the calling class from a
        # manager and a list of axes.
        # Parameters: mgr is the manager to wrap; axes is the list of axes (not used
        # by this base implementation, it is accepted for subclasses that need it).
        # Creates a bare instance without calling the subclass constructor and then
        # runs the base initializer on it with the manager. Returns the new object.
        """
        Construct a new object of this type from a Manager object and axes.

        Parameters
        ----------
        mgr : Manager
            Must have the same ndim as cls.
        axes : list[Index]

        Notes
        -----
        The axes must match mgr.axes, but are required for future-proofing
        in the event that axes are refactored out of the Manager objects.
        """
        obj = cls.__new__(cls)
        NDFrame.__init__(obj, mgr)
        return obj

    # ----------------------------------------------------------------------
    # attrs and flags

    @property
    def attrs(self) -> dict[Hashable, Any]:
        # [FUNCTION DOC]
        # Property getter returning the dictionary of global user attributes
        # (metadata) stored on the object.
        # Returns the live internal dictionary (not a copy), so in-place changes are
        # visible. Attributes are propagated by the finalize hook to results of many
        # operations.
        """
        Dictionary of global attributes of this dataset.

        .. warning::

           attrs is experimental and may change without warning.

        See Also
        --------
        DataFrame.flags : Global flags applying to this object.

        Notes
        -----
        Many operations that create new datasets will copy ``attrs``. Copies
        are always deep so that changing ``attrs`` will only affect the
        present dataset. :func:`pandas.concat` and :func:`pandas.merge` will
        only copy ``attrs`` if all input datasets have the same ``attrs``.

        Examples
        --------
        For Series:

        >>> ser = pd.Series([1, 2, 3])
        >>> ser.attrs = {"A": [10, 20, 30]}
        >>> ser.attrs
        {'A': [10, 20, 30]}

        For DataFrame:

        >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        >>> df.attrs = {"A": [10, 20, 30]}
        >>> df.attrs
        {'A': [10, 20, 30]}
        """
        return self._attrs

    @attrs.setter
    def attrs(self, value: Mapping[Hashable, Any]) -> None:
        # [FUNCTION DOC]
        # Property setter for the user attribute dictionary.
        # Parameters: value is any mapping; it is copied into a fresh dictionary which
        # replaces the stored one.
        self._attrs = dict(value)

    @final
    @property
    def flags(self) -> Flags:
        # [FUNCTION DOC]
        # Property getter returning the flags object bound to this instance (currently
        # exposing whether duplicate labels are allowed).
        # Returns the stored flags instance, not a copy.
        """
        Get the properties associated with this pandas object.

        The available flags are

        * :attr:`Flags.allows_duplicate_labels`

        See Also
        --------
        Flags : Flags that apply to pandas objects.
        DataFrame.attrs : Global metadata applying to this dataset.

        Notes
        -----
        "Flags" differ from "metadata". Flags reflect properties of the
        pandas object (the Series or DataFrame). Metadata refer to properties
        of the dataset, and should be stored in :attr:`DataFrame.attrs`.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2]})
        >>> df.flags
        <Flags(allows_duplicate_labels=True)>

        Flags can be get or set using ``.``

        >>> df.flags.allows_duplicate_labels
        True
        >>> df.flags.allows_duplicate_labels = False

        Or by slicing with a key

        >>> df.flags["allows_duplicate_labels"]
        False
        >>> df.flags["allows_duplicate_labels"] = True
        """
        return self._flags

    @final
    def set_flags(
        self,
        *,
        copy: bool | lib.NoDefault = lib.no_default,
        allows_duplicate_labels: bool | None = None,
    ) -> Self:
        # [FUNCTION DOC]
        # Return a new object (shallow copy) with updated flags.
        # Parameters (keyword only): copy is a deprecated placeholder that is
        # validated by the shared deprecation check (any explicitly passed value
        # triggers the deprecation handling); allows_duplicate_labels is an optional
        # bool that, when not None, is assigned to the flag of the same name on the
        # new object.
        # The original object is not modified. Setting the flag to disallow duplicates
        # raises through the flags machinery if the object currently has duplicate
        # labels. Returns the new object.
        """
        Return a new object with updated flags.

        This method creates a shallow copy of the original object, preserving its
        underlying data while modifying its global flags. In particular, it allows
        you to update properties such as whether duplicate labels are permitted. This
        behavior is especially useful in method chains, where one wishes to
        adjust DataFrame or Series characteristics without altering the original object.

        Parameters
        ----------
        copy : bool, default False
            This keyword is now ignored; changing its value will have no
            impact on the method.

            .. deprecated:: 3.0.0

                This keyword is ignored and will be removed in pandas 4.0. Since
                pandas 3.0, this method always returns a new object using a lazy
                copy mechanism that defers copies until necessary
                (Copy-on-Write). See the `user guide on Copy-on-Write
                <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                for more details.

        allows_duplicate_labels : bool, optional
            Whether the returned object allows duplicate labels.

        Returns
        -------
        Series or DataFrame
            The same type as the caller.

        See Also
        --------
        DataFrame.attrs : Global metadata applying to this dataset.
        DataFrame.flags : Global flags applying to this object.

        Notes
        -----
        This method returns a new object that's a view on the same data
        as the input. Mutating the input or the output values will be reflected
        in the other.

        This method is intended to be used in method chains.

        "Flags" differ from "metadata". Flags reflect properties of the
        pandas object (the Series or DataFrame). Metadata refer to properties
        of the dataset, and should be stored in :attr:`DataFrame.attrs`.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2]})
        >>> df.flags.allows_duplicate_labels
        True
        >>> df2 = df.set_flags(allows_duplicate_labels=False)
        >>> df2.flags.allows_duplicate_labels
        False
        """
        self._check_copy_deprecation(copy)
        df = self.copy(deep=False)
        if allows_duplicate_labels is not None:
            df.flags["allows_duplicate_labels"] = allows_duplicate_labels
        return df

    @final
    @classmethod
    def _validate_dtype(cls, dtype) -> DtypeObj | None:
        # [FUNCTION DOC]
        # HELPER: validates a dtype argument for the constructors.
        # Parameters: dtype is None or anything convertible to a dtype.
        # None is passed through. Otherwise the argument is converted to a pandas
        # dtype object. Compound (structured, kind V) numpy dtypes that are not
        # extension dtypes are rejected with a not-implemented error naming the class.
        # Returns the dtype object or None.
        """validate the passed dtype"""
        if dtype is not None:
            dtype = pandas_dtype(dtype)

            # a compound dtype
            if dtype.kind == "V" and not isinstance(dtype, ExtensionDtype):
                raise NotImplementedError(
                    "compound dtypes are not implemented "
                    f"in the {cls.__name__} constructor"
                )

        return dtype

    # ----------------------------------------------------------------------
    # Construction

    # error: Signature of "_constructor" incompatible with supertype "PandasObject"
    @property
    def _constructor(self) -> Callable[..., Self]:  # type: ignore[override]
        # [FUNCTION DOC]
        # HELPER: Abstract property: the callable used to build a result of the same
        # dimensionality as this object.
        # Always raises an abstract-method error in the base class; Series and
        # DataFrame override it to return their own class (or a subclass-aware
        # constructor).
        """
        Used when a manipulation result has the same dimensions as the
        original.
        """
        raise AbstractMethodError(self)

    # ----------------------------------------------------------------------
    # Axis
    _AXIS_ORDERS: list[Literal["index", "columns"]]
    _AXIS_TO_AXIS_NUMBER: dict[Axis, AxisInt] = {0: 0, "index": 0, "rows": 0}
    _info_axis_number: int
    _info_axis_name: Literal["index", "columns"]
    _AXIS_LEN: int

    @final
    def _construct_axes_dict(
        self, axes: Sequence[Axis] | None = None, **kwargs: AxisInt
    ) -> dict:
        # [FUNCTION DOC]
        # HELPER: builds a dictionary of axis name to axis labels for this object.
        # Parameters: axes is an optional sequence of axis identifiers (defaults to
        # all axes in the class axis order); extra keyword arguments are merged into
        # the result afterwards and override entries of the same key.
        # Returns a new dictionary mapping each requested axis to the corresponding
        # index object.
        """Return an axes dictionary for myself."""
        d = {a: self._get_axis(a) for a in (axes or self._AXIS_ORDERS)}
        # error: Argument 1 to "update" of "MutableMapping" has incompatible type
        # "Dict[str, Any]"; expected "SupportsKeysAndGetItem[Union[int, str], Any]"
        d.update(kwargs)  # type: ignore[arg-type]
        return d

    @final
    @classmethod
    def _get_axis_number(cls, axis: Axis) -> AxisInt:
        # [FUNCTION DOC]
        # HELPER: class method translating an axis identifier to its integer position.
        # Parameters: axis is a number or an alias accepted by the class level mapping
        # (for example an axis name or alias).
        # Returns the integer axis number. Raises a value error naming the class when
        # the identifier is not in the mapping.
        try:
            return cls._AXIS_TO_AXIS_NUMBER[axis]
        except KeyError as err:
            raise ValueError(
                f"No axis named {axis} for object type {cls.__name__}"
            ) from err

    @final
    @classmethod
    def _get_axis_name(cls, axis: Axis) -> Literal["index", "columns"]:
        # [FUNCTION DOC]
        # HELPER: class method translating an axis identifier to the canonical axis
        # name.
        # Parameters: axis is any identifier accepted by the axis number helper.
        # Returns the name (index or columns) at that position in the class axis
        # order. Propagates the value error of the number helper for invalid axes.
        axis_number = cls._get_axis_number(axis)
        return cls._AXIS_ORDERS[axis_number]

    @final
    def _get_axis(self, axis: Axis) -> Index:
        # [FUNCTION DOC]
        # HELPER: returns the Index object for an axis.
        # Parameters: axis is any axis identifier.
        # Converts it to a number, asserts that it is 0 or 1, and returns the index
        # for 0 and the columns for 1 (so for a Series only axis 0 is meaningful since
        # the columns attribute does not exist).
        axis_number = self._get_axis_number(axis)
        assert axis_number in {0, 1}
        return self.index if axis_number == 0 else self.columns

    @final
    @classmethod
    def _get_block_manager_axis(cls, axis: Axis) -> AxisInt:
        # [FUNCTION DOC]
        # HELPER: class method mapping a user-facing axis to the axis number used by
        # the internal manager.
        # Parameters: axis is any axis identifier.
        # For two-dimensional objects the manager stores axes reversed, so the result
        # is one minus the user axis number; for one-dimensional objects the number is
        # returned unchanged.
        """Map the axis to the block_manager axis."""
        axis = cls._get_axis_number(axis)
        ndim = cls._AXIS_LEN
        if ndim == 2:
            # i.e. DataFrame
            return 1 - axis
        return axis

    @final
    def _get_axis_resolvers(self, axis: str) -> dict[str, Series | MultiIndex]:
        # [FUNCTION DOC]
        # HELPER: builds the name resolution mapping used by expression evaluation
        # (query and eval) for one axis.
        # Parameters: axis is the attribute name of the axis (index or columns).
        # For each level of that axis, a series of the level values indexed by the
        # full axis is created, keyed by the level name, or by the first letter of the
        # axis followed by the text level_ and the level position when the level is
        # unnamed. The axis itself is added under the axis name (kept as a MultiIndex
        # if it is one, otherwise converted to a series). Returns the dictionary.
        # index or columns
        axis_index = getattr(self, axis)
        d = {}
        prefix = axis[0]

        for i, name in enumerate(axis_index.names):
            if name is not None:
                key = level = name
            else:
                # prefix with 'i' or 'c' depending on the input axis
                # e.g., you must do ilevel_0 for the 0th level of an unnamed
                # multiiindex
                key = f"{prefix}level_{i}"
                level = i

            level_values = axis_index.get_level_values(level)
            s = level_values.to_series()
            s.index = axis_index
            d[key] = s

        # put the index/columns itself in the dict
        if isinstance(axis_index, MultiIndex):
            dindex = axis_index
        else:
            dindex = axis_index.to_series()

        d[axis] = dindex
        return d

    @final
    def _get_index_resolvers(self) -> dict[Hashable, Series | MultiIndex]:
        # [FUNCTION DOC]
        # HELPER: collects the resolvers for all axes for expression evaluation.
        # Merges the per-axis resolver dictionaries, drops integer keys, and
        # normalizes every remaining key into a valid identifier using the column-name
        # cleaner from the computation parsing module (imported lazily). Returns the
        # resulting dictionary.
        from pandas.core.computation.parsing import clean_column_name

        d: dict[str, Series | MultiIndex] = {}
        for axis_name in self._AXIS_ORDERS:
            d.update(self._get_axis_resolvers(axis_name))

        return {clean_column_name(k): v for k, v in d.items() if not isinstance(k, int)}

    @final
    def _get_cleaned_column_resolvers(self) -> dict[Hashable, Series]:
        # [FUNCTION DOC]
        # HELPER: returns resolvers for expression evaluation that map
        # special-character-free column names to data.
        # For a Series, a single entry keyed by the cleaned series name is returned;
        # otherwise each column is keyed by its cleaned name. The cleaner is imported
        # lazily from the computation parsing module.
        """
        Return the special character free column resolvers of a DataFrame.

        Column names with special characters are 'cleaned up' so that they can
        be referred to by backtick quoting.
        Used in :meth:`DataFrame.eval`.
        """
        from pandas.core.computation.parsing import clean_column_name

        if isinstance(self, ABCSeries):
            return {clean_column_name(self.name): self}

        return {clean_column_name(k): v for k, v in self.items()}

    @final
    @property
    def _info_axis(self) -> Index:
        # [FUNCTION DOC]
        # HELPER property: the info axis, i.e. the Index that is iterated and used for
        # membership tests (index for Series, columns for DataFrame), looked up
        # through the class level info-axis name.
        return getattr(self, self._info_axis_name)

    @property
    def shape(self) -> tuple[int, ...]:
        # [FUNCTION DOC]
        # Property returning a tuple with the length of every axis in class axis
        # order.
        """
        Return a tuple of axis dimensions
        """
        return tuple(len(self._get_axis(a)) for a in self._AXIS_ORDERS)

    @property
    def axes(self) -> list[Index]:
        # [FUNCTION DOC]
        # Property returning a list of the axis index objects in class axis order
        # (index first, then columns for a DataFrame).
        """
        Return index label(s) of the internal NDFrame
        """
        # we do it this way because if we have reversed axes, then
        # the block manager shows then reversed
        return [self._get_axis(a) for a in self._AXIS_ORDERS]

    @final
    @property
    def ndim(self) -> int:
        # [FUNCTION DOC]
        # Property returning the number of dimensions as reported by the manager (1
        # for Series, 2 for DataFrame).
        """
        Return an int representing the number of axes / array dimensions.

        Return 1 if Series. Otherwise return 2 if DataFrame.

        See Also
        --------
        numpy.ndarray.ndim : Number of array dimensions.

        Examples
        --------
        >>> s = pd.Series({"a": 1, "b": 2, "c": 3})
        >>> s.ndim
        1

        >>> df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        >>> df.ndim
        2
        """
        return self._mgr.ndim

    @final
    @property
    def size(self) -> int:
        # [FUNCTION DOC]
        # Property returning the total number of elements, computed as the product of
        # the axis lengths converted to a plain int.
        """
        Return an int representing the number of elements in this object.

        Return the number of rows if Series. Otherwise return the number of
        rows times number of columns if DataFrame.

        See Also
        --------
        numpy.ndarray.size : Number of elements in the array.

        Examples
        --------
        >>> s = pd.Series({"a": 1, "b": 2, "c": 3})
        >>> s.size
        3

        >>> df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        >>> df.size
        4
        """

        return int(np.prod(self.shape))

    def set_axis(
        self,
        labels,
        *,
        axis: Axis = 0,
        copy: bool | lib.NoDefault = lib.no_default,
    ) -> Self:
        # [FUNCTION DOC]
        # Assign new labels to the given axis and return a new object.
        # Parameters (keyword only for axis and copy): labels is the new index-like
        # labels; axis selects the axis (default 0); copy is a deprecated placeholder
        # handled by the shared deprecation check.
        # The labels are validated for length by the underlying axis setter. The
        # original is not modified; a shallow copy with the labels replaced is
        # returned.
        """
        Assign desired index to given axis.

        Indexes for%(extended_summary_sub)s row labels can be changed by assigning
        a list-like or Index.

        Parameters
        ----------
        labels : list-like, Index
            The values for the new index.

        axis : %(axes_single_arg)s, default 0
            The axis to update. The value 0 identifies the rows. For `Series`
            this parameter is unused and defaults to 0.

        copy : bool, default False
            This keyword is now ignored; changing its value will have no
            impact on the method.

            .. deprecated:: 3.0.0

                This keyword is ignored and will be removed in pandas 4.0. Since
                pandas 3.0, this method always returns a new object using a lazy
                copy mechanism that defers copies until necessary
                (Copy-on-Write). See the `user guide on Copy-on-Write
                <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                for more details.

        Returns
        -------
        %(klass)s
            An object of type %(klass)s.

        See Also
        --------
        %(klass)s.rename_axis : Alter the name of the index%(see_also_sub)s.
        """
        self._check_copy_deprecation(copy)
        return self._set_axis_nocheck(labels, axis, inplace=False)

    @overload
    def _set_axis_nocheck(
        self, labels, axis: Axis, inplace: Literal[False]
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: Typing overload stub only (body is a placeholder); the behavior is
        # defined by the implementation that follows these overloads. Variant for
        # inplace false returning the same type.
        ...

    @overload
    def _set_axis_nocheck(self, labels, axis: Axis, inplace: Literal[True]) -> None:
        # [FUNCTION DOC]
        # HELPER: Typing overload stub only (body is a placeholder); the behavior is
        # defined by the implementation that follows these overloads. Variant for
        # inplace true returning nothing.
        ...

    @overload
    def _set_axis_nocheck(self, labels, axis: Axis, inplace: bool) -> Self | None:
        # [FUNCTION DOC]
        # HELPER: Typing overload stub only (body is a placeholder); the behavior is
        # defined by the implementation that follows these overloads. Variant for a
        # general boolean returning an optional object.
        ...

    @final
    def _set_axis_nocheck(self, labels, axis: Axis, inplace: bool) -> Self | None:
        # [FUNCTION DOC]
        # HELPER: sets labels on an axis with no extra validation or deprecation
        # handling.
        # Parameters: labels is the new label container; axis is any axis identifier;
        # inplace selects whether to modify self.
        # If inplace, the axis attribute (looked up by canonical name) is assigned on
        # self and None is returned. Otherwise a shallow copy is made, the attribute
        # assigned on the copy, and the copy returned.
        if inplace:
            setattr(self, self._get_axis_name(axis), labels)
            return None
        obj = self.copy(deep=False)
        setattr(obj, obj._get_axis_name(axis), labels)
        return obj

    @final
    def _set_axis(self, axis: AxisInt, labels: AnyArrayLike | list) -> None:
        # [FUNCTION DOC]
        # HELPER: low-level axis setter invoked from compiled code when the index or
        # columns attribute is assigned.
        # Parameters: axis is the integer manager-level axis; labels is any array-like
        # or list.
        # Converts the labels to an Index and passes them to the manager axis setter.
        # Mutates self. Returns nothing. The manager raises if the length does not
        # match.
        """
        This is called from the cython code when we set the `index` attribute
        directly, e.g. `series.index = [1, 2, 3]`.
        """
        labels = ensure_index(labels)
        self._mgr.set_axis(axis, labels)

    @final
    def droplevel(self, level: IndexLabel, axis: Axis = 0) -> Self:
        # [FUNCTION DOC]
        # Return a new object with the requested level(s) removed from the index or
        # the columns.
        # Parameters: level is an int, label or list of these identifying levels of a
        # MultiIndex; axis selects which axis is modified (default 0).
        # Takes the axis labels, drops the levels with the index droplevel operation,
        # and returns the object with the new labels set on that axis. Raises (from
        # the index) if levels are invalid or all levels would be removed.
        """
        Return Series/DataFrame with requested index / column level(s) removed.

        Parameters
        ----------
        level : int, str, or list-like
            If a string is given, must be the name of a level
            If list-like, elements must be names or positional indexes
            of levels.

        axis : {0 or 'index', 1 or 'columns'}, default 0
            Axis along which the level(s) is removed:

            * 0 or 'index': remove level(s) in column.
            * 1 or 'columns': remove level(s) in row.

            For `Series` this parameter is unused and defaults to 0.

        Returns
        -------
        Series/DataFrame
            Series/DataFrame with requested index / column level(s) removed.

        See Also
        --------
        DataFrame.replace : Replace values given in `to_replace` with `value`.
        DataFrame.pivot : Return reshaped DataFrame organized by given
            index / column values.

        Examples
        --------
        >>> df = (
        ...     pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        ...     .set_index([0, 1])
        ...     .rename_axis(["a", "b"])
        ... )

        >>> df.columns = pd.MultiIndex.from_tuples(
        ...     [("c", "e"), ("d", "f")], names=["level_1", "level_2"]
        ... )

        >>> df
        level_1   c   d
        level_2   e   f
        a b
        1 2      3   4
        5 6      7   8
        9 10    11  12

        >>> df.droplevel("a")
        level_1   c   d
        level_2   e   f
        b
        2        3   4
        6        7   8
        10      11  12

        >>> df.droplevel("level_2", axis=1)
        level_1   c   d
        a b
        1 2      3   4
        5 6      7   8
        9 10    11  12
        """
        labels = self._get_axis(axis)
        new_labels = labels.droplevel(level)
        return self.set_axis(new_labels, axis=axis)

    def pop(self, item: Hashable) -> Series | Any:
        # [FUNCTION DOC]
        # Remove an item from the info axis in place and return it.
        # Parameters: item is a label of the info axis (column for DataFrame, index
        # label for Series).
        # Looks up the item, deletes it from self through the item deletion operator,
        # and returns the looked up value. Raises a key error from lookup or deletion
        # if the label is absent. No docstring is defined for this method.
        result = self[item]
        del self[item]

        return result

    @final
    def squeeze(self, axis: Axis | None = None) -> Scalar | Series | DataFrame:
        # [FUNCTION DOC]
        # Reduce axes of length one to scalars.
        # Parameters: axis is None (all axes) or a single axis identifier restricting
        # the squeeze.
        # For each selected axis whose length is exactly one the positional indexer
        # takes position zero; other axes are kept whole. If the result is still an
        # NDFrame it is passed through the finalize hook with the method name squeeze.
        # Returns a scalar, a Series or the DataFrame depending on how many axes were
        # squeezed. Does not modify self.
        """
        Squeeze 1 dimensional axis objects into scalars.

        Series or DataFrames with a single element are squeezed to a scalar.
        DataFrames with a single column or a single row are squeezed to a
        Series. Otherwise the object is unchanged.

        This method is most useful when you don't know if your
        object is a Series or DataFrame, but you do know it has just a single
        column. In that case you can safely call `squeeze` to ensure you have a
        Series.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns', None}, default None
            A specific axis to squeeze. By default, all length-1 axes are
            squeezed. For `Series` this parameter is unused and defaults to `None`.

        Returns
        -------
        DataFrame, Series, or scalar
            The projection after squeezing `axis` or all the axes.

        See Also
        --------
        Series.iloc : Integer-location based indexing for selecting scalars.
        DataFrame.iloc : Integer-location based indexing for selecting Series.
        Series.to_frame : Inverse of DataFrame.squeeze for a
            single-column DataFrame.

        Examples
        --------
        >>> primes = pd.Series([2, 3, 5, 7])

        Slicing might produce a Series with a single value:

        >>> even_primes = primes[primes % 2 == 0]
        >>> even_primes
        0    2
        dtype: int64

        >>> even_primes.squeeze()
        np.int64(2)

        Squeezing objects with more than one value in every axis does nothing:

        >>> odd_primes = primes[primes % 2 == 1]
        >>> odd_primes
        1    3
        2    5
        3    7
        dtype: int64

        >>> odd_primes.squeeze()
        1    3
        2    5
        3    7
        dtype: int64

        Squeezing is even more effective when used with DataFrames.

        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
        >>> df
           a  b
        0  1  2
        1  3  4

        Slicing a single column will produce a DataFrame with the columns
        having only one value:

        >>> df_a = df[["a"]]
        >>> df_a
           a
        0  1
        1  3

        So the columns can be squeezed down, resulting in a Series:

        >>> df_a.squeeze("columns")
        0    1
        1    3
        Name: a, dtype: int64

        Slicing a single row from a single column will produce a single
        scalar DataFrame:

        >>> df_0a = df.loc[df.index < 1, ["a"]]
        >>> df_0a
           a
        0  1

        Squeezing the rows produces a single scalar Series:

        >>> df_0a.squeeze("rows")
        a    1
        Name: 0, dtype: int64

        Squeezing all axes will project directly into a scalar:

        >>> df_0a.squeeze()
        np.int64(1)
        """
        axes = range(self._AXIS_LEN) if axis is None else (self._get_axis_number(axis),)
        result = self.iloc[
            tuple(
                0 if i in axes and len(a) == 1 else slice(None)
                for i, a in enumerate(self.axes)
            )
        ]
        if isinstance(result, NDFrame):
            result = result.__finalize__(self, method="squeeze")
        return result

    # ----------------------------------------------------------------------
    # Rename

    @overload
    def _rename(
        self,
        mapper: Renamer | None = ...,
        *,
        index: Renamer | None = ...,
        columns: Renamer | None = ...,
        axis: Axis | None = ...,
        inplace: Literal[False] = ...,
        level: Level | None = ...,
        errors: str = ...,
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: Typing overload stub only (body is a placeholder); the behavior is
        # defined by the implementation that follows these overloads. Private rename;
        # variant for inplace false.
        ...

    @overload
    def _rename(
        self,
        mapper: Renamer | None = ...,
        *,
        index: Renamer | None = ...,
        columns: Renamer | None = ...,
        axis: Axis | None = ...,
        inplace: Literal[True],
        level: Level | None = ...,
        errors: str = ...,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Typing overload stub only (body is a placeholder); the behavior is
        # defined by the implementation that follows these overloads. Private rename;
        # variant for inplace true.
        ...

    @overload
    def _rename(
        self,
        mapper: Renamer | None = ...,
        *,
        index: Renamer | None = ...,
        columns: Renamer | None = ...,
        axis: Axis | None = ...,
        inplace: bool,
        level: Level | None = ...,
        errors: str = ...,
    ) -> Self | None:
        # [FUNCTION DOC]
        # HELPER: Typing overload stub only (body is a placeholder); the behavior is
        # defined by the implementation that follows these overloads. Private rename;
        # variant for a general boolean inplace.
        ...

    @final
    def _rename(
        self,
        mapper: Renamer | None = None,
        *,
        index: Renamer | None = None,
        columns: Renamer | None = None,
        axis: Axis | None = None,
        inplace: bool = False,
        level: Level | None = None,
        errors: str = "ignore",
    ) -> Self | None:
        # [FUNCTION DOC]
        # HELPER: shared implementation of the public rename methods of Series and
        # DataFrame.
        # Parameters (keyword only except mapper): mapper is a dict-like or function
        # applied to the axis chosen by axis; index and columns are alternative
        # per-axis mappers; axis selects the axis for mapper (default behavior applies
        # mapper to the index); inplace says whether to modify self; level restricts
        # renaming to one level of a MultiIndex; errors is ignore or raise, and when
        # raise a key error is raised for dict-like mappers with labels not found in
        # the axis.
        # Raises a type error if no mapper of any kind is given, or if axis is
        # combined with index or columns, or mapper with either of them. Raises a
        # value error if a Series mapper has a non-unique index. The in-place path
        # first checks that duplicate labels are allowed.
        # For each requested axis a rename function is derived from the mapper, the
        # new labels are computed by transforming the existing axis and assigned
        # without validation to a shallow copy (or self). If inplace the result is
        # merged back into self and None is returned; otherwise the copy is returned
        # after the finalize hook with method rename.
        # called by Series.rename and DataFrame.rename

        if mapper is None and index is None and columns is None:
            raise TypeError("must pass an index to rename")

        if index is not None or columns is not None:
            if axis is not None:
                raise TypeError(
                    "Cannot specify both 'axis' and any of 'index' or 'columns'"
                )
            if mapper is not None:
                raise TypeError(
                    "Cannot specify both 'mapper' and any of 'index' or 'columns'"
                )
        # use the mapper argument
        elif axis and self._get_axis_number(axis) == 1:
            columns = mapper
        else:
            index = mapper

        self._check_inplace_and_allows_duplicate_labels(inplace)
        result = self if inplace else self.copy(deep=False)

        for axis_no, replacements in enumerate((index, columns)):
            if replacements is None:
                continue

            ax = self._get_axis(axis_no)
            f = common.get_rename_function(replacements)

            if level is not None:
                level = ax._get_level_number(level)

            if isinstance(replacements, ABCSeries) and not replacements.index.is_unique:
                # GH#58621
                raise ValueError("Cannot rename with a Series with non-unique index.")

            # GH 13473
            if not callable(replacements):
                if ax._is_multi and level is not None:
                    indexer = ax.get_level_values(level).get_indexer_for(replacements)
                else:
                    indexer = ax.get_indexer_for(replacements)

                if errors == "raise" and len(indexer[indexer == -1]):
                    missing_labels = [
                        label
                        for index, label in enumerate(replacements)
                        if indexer[index] == -1
                    ]
                    raise KeyError(f"{missing_labels} not found in axis")

            new_index = ax._transform_index(f, level=level)
            result._set_axis_nocheck(new_index, axis=axis_no, inplace=True)

        if inplace:
            self._update_inplace(result)
            return None
        else:
            return result.__finalize__(self, method="rename")

    @overload
    def rename_axis(
        self,
        mapper: IndexLabel | lib.NoDefault = ...,
        *,
        index=...,
        columns=...,
        axis: Axis = ...,
        copy: bool | lib.NoDefault = lib.no_default,
        inplace: Literal[False] = ...,
    ) -> Self:
        # [FUNCTION DOC]
        # Typing overload stub only (body is a placeholder); the behavior is defined
        # by the implementation that follows these overloads. rename_axis variant for
        # inplace false.
        ...

    @overload
    def rename_axis(
        self,
        mapper: IndexLabel | lib.NoDefault = ...,
        *,
        index=...,
        columns=...,
        axis: Axis = ...,
        copy: bool | lib.NoDefault = lib.no_default,
        inplace: Literal[True],
    ) -> None:
        # [FUNCTION DOC]
        # Typing overload stub only (body is a placeholder); the behavior is defined
        # by the implementation that follows these overloads. rename_axis variant for
        # inplace true.
        ...

    @overload
    def rename_axis(
        self,
        mapper: IndexLabel | lib.NoDefault = ...,
        *,
        index=...,
        columns=...,
        axis: Axis = ...,
        copy: bool | lib.NoDefault = lib.no_default,
        inplace: bool = ...,
    ) -> Self | None:
        # [FUNCTION DOC]
        # Typing overload stub only (body is a placeholder); the behavior is defined
        # by the implementation that follows these overloads. rename_axis variant for
        # general boolean inplace.
        ...

    def rename_axis(
        self,
        mapper: IndexLabel | lib.NoDefault = lib.no_default,
        *,
        index=lib.no_default,
        columns=lib.no_default,
        axis: Axis = 0,
        copy: bool | lib.NoDefault = lib.no_default,
        inplace: bool = False,
    ) -> Self | None:
        # [FUNCTION DOC]
        # Set the name(s) of the index or column axis.
        # Parameters (keyword only except mapper): mapper is a scalar or list-like of
        # new names applied to the axis given by axis; index and columns give per-axis
        # values (scalar, list-like, or a dict-like/function transformer applied to
        # the existing names); axis is the axis for mapper (default 0); copy is a
        # deprecated placeholder; inplace chooses in-place modification.
        # When mapper is supplied: scalar or non-dict list-like values set the axis
        # names, anything else raises a value error telling the user to use rename.
        # Otherwise each of index and columns that was supplied is applied: scalars
        # and non-dict list-likes become the names directly, other values are turned
        # into a rename function applied to each current name of that axis. Axes left
        # at the sentinel default are skipped.
        # Returns the modified copy, or None when inplace.
        """
        Set the name of the axis for the index or columns.

        Parameters
        ----------
        mapper : scalar, list-like, optional
            Value to set the axis name attribute.

            Use either ``mapper`` and ``axis`` to
            specify the axis to target with ``mapper``, or ``index``
            and/or ``columns``.
        index : scalar, list-like, dict-like or function, optional
            A scalar, list-like, dict-like or functions transformations to
            apply to that axis' values.
        columns : scalar, list-like, dict-like or function, optional
            A scalar, list-like, dict-like or functions transformations to
            apply to that axis' values.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The axis to rename.
        copy : bool, default False
            This keyword is now ignored; changing its value will have no
            impact on the method.

            .. deprecated:: 3.0.0

                This keyword is ignored and will be removed in pandas 4.0. Since
                pandas 3.0, this method always returns a new object using a lazy
                copy mechanism that defers copies until necessary
                (Copy-on-Write). See the `user guide on Copy-on-Write
                <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                for more details.

        inplace : bool, default False
            Modifies the object directly, instead of creating a new Series
            or DataFrame.

        Returns
        -------
        DataFrame, or None
            The same type as the caller or None if ``inplace=True``.

        See Also
        --------
        Series.rename : Alter Series index labels or name.
        DataFrame.rename : Alter DataFrame index labels or name.
        Index.rename : Set new names on index.

        Notes
        -----
        ``DataFrame.rename_axis`` supports two calling conventions

        * ``(index=index_mapper, columns=columns_mapper, ...)``
        * ``(mapper, axis={'index', 'columns'}, ...)``

        The first calling convention will only modify the names of
        the index and/or the names of the Index object that is the columns.
        In this case, the parameter ``copy`` is ignored.

        The second calling convention will modify the names of the
        corresponding index if mapper is a list or a scalar.
        However, if mapper is dict-like or a function, it will use the
        deprecated behavior of modifying the axis *labels*.

        We *highly* recommend using keyword arguments to clarify your
        intent.

        Examples
        --------
        **DataFrame**

        >>> df = pd.DataFrame(
        ...     {"num_legs": [4, 4, 2], "num_arms": [0, 0, 2]}, ["dog", "cat", "monkey"]
        ... )
        >>> df
                num_legs  num_arms
        dog            4         0
        cat            4         0
        monkey         2         2
        >>> df = df.rename_axis("animal")
        >>> df
                num_legs  num_arms
        animal
        dog            4         0
        cat            4         0
        monkey         2         2
        >>> df = df.rename_axis("limbs", axis="columns")
        >>> df
        limbs   num_legs  num_arms
        animal
        dog            4         0
        cat            4         0
        monkey         2         2

        **MultiIndex**

        >>> df.index = pd.MultiIndex.from_product(
        ...     [["mammal"], ["dog", "cat", "monkey"]], names=["type", "name"]
        ... )
        >>> df
        limbs          num_legs  num_arms
        type   name
        mammal dog            4         0
               cat            4         0
               monkey         2         2

        >>> df.rename_axis(index={"type": "class"})
        limbs          num_legs  num_arms
        class  name
        mammal dog            4         0
               cat            4         0
               monkey         2         2

        >>> df.rename_axis(columns=str.upper)
        LIMBS          num_legs  num_arms
        type   name
        mammal dog            4         0
               cat            4         0
               monkey         2         2
        """
        self._check_copy_deprecation(copy)
        axes = {"index": index, "columns": columns}

        if axis is not None:
            axis = self._get_axis_number(axis)

        inplace = validate_bool_kwarg(inplace, "inplace")

        if mapper is not lib.no_default:
            # Use v0.23 behavior if a scalar or list
            non_mapper = is_scalar(mapper) or (
                is_list_like(mapper) and not is_dict_like(mapper)
            )
            if non_mapper:
                return self._set_axis_name(mapper, axis=axis, inplace=inplace)
            else:
                raise ValueError("Use `.rename` to alter labels with a mapper.")
        else:
            # Use new behavior.  Means that index and/or columns
            # is specified
            result = self if inplace else self.copy(deep=False)

            for axis in range(self._AXIS_LEN):
                v = axes.get(self._get_axis_name(axis))
                if v is lib.no_default:
                    continue
                non_mapper = is_scalar(v) or (is_list_like(v) and not is_dict_like(v))
                if non_mapper:
                    newnames = v
                else:
                    f = common.get_rename_function(v)
                    curnames = self._get_axis(axis).names
                    newnames = [f(name) for name in curnames]
                result._set_axis_name(newnames, axis=axis, inplace=True)
            if not inplace:
                return result
            return None

    @overload
    def _set_axis_name(
        self, name, axis: Axis = ..., *, inplace: Literal[False] = ...
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: Typing overload stub only (body is a placeholder); the behavior is
        # defined by the implementation that follows these overloads. Private name
        # setter; variant for inplace false.
        ...

    @overload
    def _set_axis_name(
        self, name, axis: Axis = ..., *, inplace: Literal[True]
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Typing overload stub only (body is a placeholder); the behavior is
        # defined by the implementation that follows these overloads. Private name
        # setter; variant for inplace true.
        ...

    @overload
    def _set_axis_name(
        self, name, axis: Axis = ..., *, inplace: bool
    ) -> Self | None:
        # [FUNCTION DOC]
        # HELPER: Typing overload stub only (body is a placeholder); the behavior is
        # defined by the implementation that follows these overloads. Private name
        # setter; variant for general boolean inplace.
        ...

    @final
    def _set_axis_name(
        self, name, axis: Axis = 0, *, inplace: bool = False
    ) -> Self | None:
        # [FUNCTION DOC]
        # HELPER: sets the name(s) of an axis, used by rename_axis and other callers.
        # Parameters: name is a scalar or list-like of names; axis is the axis
        # identifier (default 0); inplace (keyword only) selects in-place
        # modification.
        # Builds new axis labels with the names applied, validates the inplace flag as
        # a boolean, and assigns them to the index or columns of either self or a
        # shallow copy. Returns the copy, or None when inplace.
        """
        Set the name(s) of the axis.

        Parameters
        ----------
        name : str or list of str
            Name(s) to set.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The axis to set the label. The value 0 or 'index' specifies index,
            and the value 1 or 'columns' specifies columns.
        inplace : bool, default False
            If `True`, do operation inplace and return None.

        Returns
        -------
        Series, DataFrame, or None
            The same type as the caller or `None` if `inplace` is `True`.

        See Also
        --------
        DataFrame.rename : Alter the axis labels of :class:`DataFrame`.
        Series.rename : Alter the index labels or set the index name
            of :class:`Series`.
        Index.rename : Set the name of :class:`Index` or :class:`MultiIndex`.

        Examples
        --------
        >>> df = pd.DataFrame({"num_legs": [4, 4, 2]}, ["dog", "cat", "monkey"])
        >>> df
                num_legs
        dog            4
        cat            4
        monkey         2
        >>> df._set_axis_name("animal")
                num_legs
        animal
        dog            4
        cat            4
        monkey         2
        >>> df.index = pd.MultiIndex.from_product(
        ...     [["mammal"], ["dog", "cat", "monkey"]]
        ... )
        >>> df._set_axis_name(["type", "name"])
                       num_legs
        type   name
        mammal dog        4
               cat        4
               monkey     2
        """
        axis = self._get_axis_number(axis)
        idx = self._get_axis(axis).set_names(name)

        inplace = validate_bool_kwarg(inplace, "inplace")
        renamed = self if inplace else self.copy(deep=False)
        if axis == 0:
            renamed.index = idx
        else:
            renamed.columns = idx

        if not inplace:
            return renamed
        return None

    # ----------------------------------------------------------------------
    # Comparison Methods

    @final
    def _indexed_same(self, other) -> bool:
        # [FUNCTION DOC]
        # HELPER: returns whether every axis of self equals the corresponding axis of
        # other (labels and order). other must provide the same axes. Used by
        # alignment and comparison code.
        return all(
            self._get_axis(a).equals(other._get_axis(a)) for a in self._AXIS_ORDERS
        )

    @final
    def equals(self, other: object) -> bool:
        # [FUNCTION DOC]
        # Test whether two objects have the same shape and elements, treating missing
        # values in the same location as equal.
        # Parameters: other is any object. If it is not an instance of the same class
        # (checking in both directions of subclassing) False is returned. Otherwise
        # the managers are compared, which requires equal axes and equal values with
        # dtype-aware comparison. Returns a bool and never raises for mismatched
        # types.
        """
        Test whether two objects contain the same elements.

        This function allows two Series or DataFrames to be compared against
        each other to see if they have the same shape and elements. NaNs in
        the same location are considered equal.

        The row/column index do not need to have the same type, as long
        as the values are considered equal. Corresponding columns and
        index must be of the same dtype.

        Parameters
        ----------
        other : Series or DataFrame
            The other Series or DataFrame to be compared with the first.

        Returns
        -------
        bool
            True if all elements are the same in both objects, False
            otherwise.

        See Also
        --------
        Series.eq : Compare two Series objects of the same length
            and return a Series where each element is True if the element
            in each Series is equal, False otherwise.
        DataFrame.eq : Compare two DataFrame objects of the same shape and
            return a DataFrame where each element is True if the respective
            element in each DataFrame is equal, False otherwise.
        testing.assert_series_equal : Raises an AssertionError if left and
            right are not equal. Provides an easy interface to ignore
            inequality in dtypes, indexes and precision among others.
        testing.assert_frame_equal : Like assert_series_equal, but targets
            DataFrames.
        numpy.array_equal : Return True if two arrays have the same shape
            and elements, False otherwise.

        Examples
        --------
        >>> df = pd.DataFrame({1: [10], 2: [20]})
        >>> df
            1   2
        0  10  20

        DataFrames df and exactly_equal have the same types and values for
        their elements and column labels, which will return True.

        >>> exactly_equal = pd.DataFrame({1: [10], 2: [20]})
        >>> exactly_equal
            1   2
        0  10  20
        >>> df.equals(exactly_equal)
        True

        DataFrames df and different_column_type have the same element
        types and values, but have different types for the column labels,
        which will still return True.

        >>> different_column_type = pd.DataFrame({1.0: [10], 2.0: [20]})
        >>> different_column_type
           1.0  2.0
        0   10   20
        >>> df.equals(different_column_type)
        True

        DataFrames df and different_data_type have different types for the
        same values for their elements, and will return False even though
        their column labels are the same values and types.

        >>> different_data_type = pd.DataFrame({1: [10.0], 2: [20.0]})
        >>> different_data_type
              1     2
        0  10.0  20.0
        >>> df.equals(different_data_type)
        False

        DataFrames with NaN in the same locations compare equal.

        >>> df_nan1 = pd.DataFrame({"a": [1, np.nan], "b": [3, np.nan]})
        >>> df_nan2 = pd.DataFrame({"a": [1, np.nan], "b": [3, np.nan]})
        >>> df_nan1.equals(df_nan2)
        True

        If the NaN values are not in the same locations, they compare unequal.

        >>> df_nan3 = pd.DataFrame({"a": [1, np.nan], "b": [3, 4]})
        >>> df_nan1.equals(df_nan3)
        False
        """
        if not (isinstance(other, type(self)) or isinstance(self, type(other))):
            return False
        other = cast(NDFrame, other)
        return self._mgr.equals(other._mgr)

    # -------------------------------------------------------------------------
    # Unary Methods

    @final
    def __neg__(self) -> Self:
        # [FUNCTION DOC]
        # Unary minus. Applies a block-wise function through the manager: boolean
        # blocks are logically inverted, other blocks are arithmetically negated.
        # Wraps the new manager in an object of the same kind and runs the finalize
        # hook with method name for negation. Returns the new object; self is
        # unchanged.
        def blk_func(values: ArrayLike):
            # [FUNCTION DOC]
            # HELPER: block-level function used by the negation operator.
            # Parameters: values is the array of one block. Boolean dtype values are
            # inverted (logical not); all other dtypes are negated arithmetically.
            # Returns the new array.
            if is_bool_dtype(values.dtype):
                # error: Argument 1 to "inv" has incompatible type "Union
                # [ExtensionArray, ndarray[Any, Any]]"; expected
                # "_SupportsInversion[ndarray[Any, dtype[bool_]]]"
                return operator.inv(values)  # type: ignore[arg-type]
            else:
                # error: Argument 1 to "neg" has incompatible type "Union
                # [ExtensionArray, ndarray[Any, Any]]"; expected
                # "_SupportsNeg[ndarray[Any, dtype[Any]]]"
                return operator.neg(values)  # type: ignore[arg-type]

        new_data = self._mgr.apply(blk_func)
        res = self._constructor_from_mgr(new_data, axes=new_data.axes)
        return res.__finalize__(self, method="__neg__")

    @final
    def __pos__(self) -> Self:
        # [FUNCTION DOC]
        # Unary plus. Applies a block-wise function through the manager (copy for
        # booleans, positive operator otherwise), rebuilds an object of the same kind
        # and runs the finalize hook with the method name for unary plus. Returns the
        # new object.
        def blk_func(values: ArrayLike):
            # [FUNCTION DOC]
            # HELPER: block-level function used by the unary plus operator.
            # Parameters: values is the array of one block. Boolean values are
            # returned as a copy; other dtypes go through the numeric positive
            # operation. Returns the new array.
            if is_bool_dtype(values.dtype):
                return values.copy()
            else:
                # error: Argument 1 to "pos" has incompatible type "Union
                # [ExtensionArray, ndarray[Any, Any]]"; expected
                # "_SupportsPos[ndarray[Any, dtype[Any]]]"
                return operator.pos(values)  # type: ignore[arg-type]

        new_data = self._mgr.apply(blk_func)
        res = self._constructor_from_mgr(new_data, axes=new_data.axes)
        return res.__finalize__(self, method="__pos__")

    @final
    def __invert__(self) -> Self:
        # [FUNCTION DOC]
        # Bitwise inversion (the tilde operator). If the object has zero elements a
        # shallow copy is returned without applying the operation (because inversion
        # fails on empty data). Otherwise the invert operator is applied through the
        # manager block by block, and the result is finalized with the method name for
        # invert. Returns a new object of the same kind.
        if not self.size:
            # inv fails with 0 len
            return self.copy(deep=False)

        new_data = self._mgr.apply(operator.invert)
        res = self._constructor_from_mgr(new_data, axes=new_data.axes)
        return res.__finalize__(self, method="__invert__")

    @final
    def __bool__(self) -> NoReturn:
        # [FUNCTION DOC]
        # Truth-value testing is always an error: raises a value error stating that
        # the truth value of the object (naming its class) is ambiguous and suggesting
        # alternatives.
        # KNOWN ISSUE: the message suggests a bool method of the object, but that
        # method no longer exists on Series or DataFrame in current pandas, so the
        # advice is outdated.
        raise ValueError(
            f"The truth value of a {type(self).__name__} is ambiguous. "
            "Use a.empty, a.bool(), a.item(), a.any() or a.all()."
        )

    @final
    def abs(self) -> Self:
        # [FUNCTION DOC]
        # Return the absolute value of each element.
        # Applies the numpy absolute function to every block through the manager and
        # rebuilds an object of the same kind.
        # KNOWN ISSUE: the finalize hook is called with a keyword named name instead
        # of method, so the method argument stays None; the name keyword is swallowed
        # by the extra keyword arguments, so the operation name is never recorded.
        # Returns the new object.
        """
        Return a Series/DataFrame with absolute numeric value of each element.

        This function only applies to elements that are all numeric.

        Returns
        -------
        abs
            Series/DataFrame containing the absolute value of each element.

        See Also
        --------
        numpy.absolute : Calculate the absolute value element-wise.

        Notes
        -----
        For ``complex`` inputs, ``1.2 + 1j``, the absolute value is
        :math:`\\sqrt{ a^2 + b^2 }`.

        Examples
        --------
        Absolute numeric values in a Series.

        >>> s = pd.Series([-1.10, 2, -3.33, 4])
        >>> s.abs()
        0    1.10
        1    2.00
        2    3.33
        3    4.00
        dtype: float64

        Absolute numeric values in a Series with complex numbers.

        >>> s = pd.Series([1.2 + 1j])
        >>> s.abs()
        0    1.56205
        dtype: float64

        Absolute numeric values in a Series with a Timedelta element.

        >>> s = pd.Series([pd.Timedelta("1 days")])
        >>> s.abs()
        0   1 days
        dtype: timedelta64[us]

        Select rows with data closest to certain value using argsort (from
        `StackOverflow <https://stackoverflow.com/a/17758115>`__).

        >>> df = pd.DataFrame(
        ...     {"a": [4, 5, 6, 7], "b": [10, 20, 30, 40], "c": [100, 50, -30, -50]}
        ... )
        >>> df
             a    b    c
        0    4   10  100
        1    5   20   50
        2    6   30  -30
        3    7   40  -50
        >>> df.loc[(df.c - 43).abs().argsort()]
             a    b    c
        1    5   20   50
        0    4   10  100
        2    6   30  -30
        3    7   40  -50
        """
        res_mgr = self._mgr.apply(np.abs)
        return self._constructor_from_mgr(res_mgr, axes=res_mgr.axes).__finalize__(
            self, name="abs"
        )

    @final
    def __abs__(self) -> Self:
        # [FUNCTION DOC]
        # Implements the built-in absolute value operation by delegating to the abs
        # method. Returns a new object.
        return self.abs()

    @final
    def __round__(self, decimals: int = 0) -> Self:
        # [FUNCTION DOC]
        # Implements the built-in round operation.
        # Parameters: decimals is the number of decimals (default 0).
        # Delegates to the round method of the concrete class, then runs the finalize
        # hook with the method name for the built-in round. Returns the rounded
        # object.
        return self.round(decimals).__finalize__(self, method="__round__")

    # -------------------------------------------------------------------------
    # Label or Level Combination Helpers
    #
    # A collection of helper methods for DataFrame/Series operations that
    # accept a combination of column/index labels and levels.  All such
    # operations should utilize/extend these methods when possible so that we
    # have consistent precedence and validation logic throughout the library.

    @final
    def _is_level_reference(self, key: Level, axis: Axis = 0) -> bool:
        # [FUNCTION DOC]
        # HELPER: tests whether a key refers to an index level on the given axis.
        # Parameters: key is a level name; axis is the axis identifier (default 0).
        # True only if the key is not None, is hashable, is among the names of the
        # axis, and is not also a label on any other axis (labels take precedence over
        # levels in ambiguous cases). Returns a bool.
        """
        Test whether a key is a level reference for a given axis.

        To be considered a level reference, `key` must be a string that:
          - (axis=0): Matches the name of an index level and does NOT match
            a column label.
          - (axis=1): Matches the name of a column level and does NOT match
            an index label.

        Parameters
        ----------
        key : Hashable
            Potential level name for the given axis
        axis : int, default 0
            Axis that levels are associated with (0 for index, 1 for columns)

        Returns
        -------
        is_level : bool
        """
        axis_int = self._get_axis_number(axis)

        return (
            key is not None
            and is_hashable(key)
            and key in self.axes[axis_int].names
            and not self._is_label_reference(key, axis=axis_int)
        )

    @final
    def _is_label_reference(self, key: Level, axis: Axis = 0) -> bool:
        # [FUNCTION DOC]
        # HELPER: tests whether a key is a label of one of the axes other than the
        # given axis (for example a column label when axis is the index).
        # Parameters: key; axis (default 0). Requires the key to be hashable. Returns
        # a bool.
        """
        Test whether a key is a label reference for a given axis.

        To be considered a label reference, `key` must be a string that:
          - (axis=0): Matches a column label
          - (axis=1): Matches an index label

        Parameters
        ----------
        key : Hashable
            Potential label name, i.e. Index entry.
        axis : int, default 0
            Axis perpendicular to the axis that labels are associated with
            (0 means search for column labels, 1 means search for index labels)

        Returns
        -------
        is_label: bool
        """
        axis_int = self._get_axis_number(axis)
        other_axes = (ax for ax in range(self._AXIS_LEN) if ax != axis_int)

        return is_hashable(key) and any(key in self.axes[ax] for ax in other_axes)

    @final
    def _is_label_or_level_reference(self, key: Level, axis: AxisInt = 0) -> bool:
        # [FUNCTION DOC]
        # HELPER: returns true if the key is either a level reference or a label
        # reference for the given axis. Parameters: key; axis as an integer (default
        # 0).
        """
        Test whether a key is a label or level reference for a given axis.

        To be considered either a label or a level reference, `key` must be a
        string that:
          - (axis=0): Matches a column label or an index level
          - (axis=1): Matches an index label or a column level

        Parameters
        ----------
        key : Hashable
            Potential label or level name
        axis : int, default 0
            Axis that levels are associated with (0 for index, 1 for columns)

        Returns
        -------
        bool
        """
        return self._is_level_reference(key, axis=axis) or self._is_label_reference(
            key, axis=axis
        )

    @final
    def _check_label_or_level_ambiguity(self, key: Level, axis: Axis = 0) -> None:
        # [FUNCTION DOC]
        # HELPER: raises when a key is simultaneously a level name on the given axis
        # and a label on another axis.
        # Parameters: key; axis (default 0).
        # Returns None if there is no ambiguity. Raises a value error with a message
        # built around the correct article and the words index and column describing
        # both roles.
        """
        Check whether `key` is ambiguous.

        By ambiguous, we mean that it matches both a level of the input
        `axis` and a label of the other axis.

        Parameters
        ----------
        key : Hashable
            Label or level name.
        axis : int, default 0
            Axis that levels are associated with (0 for index, 1 for columns).

        Raises
        ------
        ValueError: `key` is ambiguous
        """

        axis_int = self._get_axis_number(axis)
        other_axes = (ax for ax in range(self._AXIS_LEN) if ax != axis_int)

        if (
            key is not None
            and is_hashable(key)
            and key in self.axes[axis_int].names
            and any(key in self.axes[ax] for ax in other_axes)
        ):
            # Build an informative and grammatical warning
            level_article, level_type = (
                ("an", "index") if axis_int == 0 else ("a", "column")
            )

            label_article, label_type = (
                ("a", "column") if axis_int == 0 else ("an", "index")
            )

            msg = (
                f"'{key}' is both {level_article} {level_type} level and "
                f"{label_article} {label_type} label, which is ambiguous."
            )
            raise ValueError(msg)

    @final
    def _get_label_or_level_values(self, key: Level, axis: AxisInt = 0) -> ArrayLike:
        # [FUNCTION DOC]
        # HELPER: returns a one-dimensional array of the values for a key that names a
        # label or a level of the given axis.
        # Parameters: key; axis as an integer (default 0).
        # If the key is a label reference, ambiguity is checked first and the values
        # are cross-sectioned from the first other axis; if it is a level reference
        # the level values of that axis are taken; otherwise a key error is raised. A
        # value error is raised if all axes match (no other axis exists) or if the
        # resulting values are multi-dimensional (the label is not unique), with an
        # extra hint for MultiIndex labels. Returns the underlying array.
        """
        Return a 1-D array of values associated with `key`, a label or level
        from the given `axis`.

        Retrieval logic:
          - (axis=0): Return column values if `key` matches a column label.
            Otherwise return index level values if `key` matches an index
            level.
          - (axis=1): Return row values if `key` matches an index label.
            Otherwise return column level values if 'key' matches a column
            level

        Parameters
        ----------
        key : Hashable
            Label or level name.
        axis : int, default 0
            Axis that levels are associated with (0 for index, 1 for columns)

        Returns
        -------
        np.ndarray or ExtensionArray

        Raises
        ------
        KeyError
            if `key` matches neither a label nor a level
        ValueError
            if `key` matches multiple labels
        """
        axis = self._get_axis_number(axis)
        first_other_axes = next(
            (ax for ax in range(self._AXIS_LEN) if ax != axis), None
        )

        if self._is_label_reference(key, axis=axis):
            self._check_label_or_level_ambiguity(key, axis=axis)
            if first_other_axes is None:
                raise ValueError("axis matched all axes")
            values = self.xs(key, axis=first_other_axes)._values
        elif self._is_level_reference(key, axis=axis):
            values = self.axes[axis].get_level_values(key)._values
        else:
            raise KeyError(key)

        # Check for duplicates
        if values.ndim > 1:
            if first_other_axes is not None and isinstance(
                self._get_axis(first_other_axes), MultiIndex
            ):
                multi_message = (
                    "\n"
                    "For a multi-index, the label must be a "
                    "tuple with elements corresponding to each level."
                )
            else:
                multi_message = ""

            label_axis_name = "column" if axis == 0 else "index"
            raise ValueError(
                f"The {label_axis_name} label '{key}' is not unique.{multi_message}"
            )

        return values

    @final
    def _drop_labels_or_levels(self, keys, axis: AxisInt = 0):
        # [FUNCTION DOC]
        # HELPER: drops the given labels and/or levels along an axis and returns a new
        # object.
        # Parameters: keys is a single key or list of keys (converted to a list); axis
        # as an integer (default 0).
        # Raises a value error listing keys that are not valid labels or levels.
        # Levels to drop are keys that are level references; the rest are labels. A
        # shallow copy is made once and modified in place: for axis 0, index levels
        # are removed by resetting the index with drop, and column labels are dropped;
        # for axis 1, levels are dropped from a MultiIndex of columns or, for a flat
        # column index, the columns are replaced with a default range index, and index
        # labels are dropped. Returns the modified copy.
        """
        Drop labels and/or levels for the given `axis`.

        For each key in `keys`:
          - (axis=0): If key matches a column label then drop the column.
            Otherwise if key matches an index level then drop the level.
          - (axis=1): If key matches an index label then drop the row.
            Otherwise if key matches a column level then drop the level.

        Parameters
        ----------
        keys : str or list of str
            labels or levels to drop
        axis : int, default 0
            Axis that levels are associated with (0 for index, 1 for columns)

        Returns
        -------
        dropped: DataFrame

        Raises
        ------
        ValueError
            if any `keys` match neither a label nor a level
        """
        axis = self._get_axis_number(axis)

        # Validate keys
        keys = common.maybe_make_list(keys)
        invalid_keys = [
            k for k in keys if not self._is_label_or_level_reference(k, axis=axis)
        ]

        if invalid_keys:
            raise ValueError(
                "The following keys are not valid labels or "
                f"levels for axis {axis}: {invalid_keys}"
            )

        # Compute levels and labels to drop
        levels_to_drop = [k for k in keys if self._is_level_reference(k, axis=axis)]

        labels_to_drop = [k for k in keys if not self._is_level_reference(k, axis=axis)]

        # Perform copy upfront and then use inplace operations below.
        # This ensures that we always perform exactly one copy.
        # ``copy`` and/or ``inplace`` options could be added in the future.
        dropped = self.copy(deep=False)

        if axis == 0:
            # Handle dropping index levels
            if levels_to_drop:
                dropped.reset_index(levels_to_drop, drop=True, inplace=True)

            # Handle dropping columns labels
            if labels_to_drop:
                dropped.drop(labels_to_drop, axis=1, inplace=True)
        else:
            # Handle dropping column levels
            if levels_to_drop:
                if isinstance(dropped.columns, MultiIndex):
                    # Drop the specified levels from the MultiIndex
                    dropped.columns = dropped.columns.droplevel(levels_to_drop)
                else:
                    # Drop the last level of Index by replacing with
                    # a RangeIndex
                    dropped.columns = default_index(dropped.columns.size)

            # Handle dropping index labels
            if labels_to_drop:
                dropped.drop(labels_to_drop, axis=0, inplace=True)

        return dropped

    # ----------------------------------------------------------------------
    # Iteration

    # https://github.com/python/typeshed/issues/2148#issuecomment-520783318
    # Incompatible types in assignment (expression has type "None", base class
    # "object" defined the type as "Callable[[object], int]")
    __hash__: ClassVar[None]  # type: ignore[assignment]

    def __iter__(self) -> Iterator:
        # [FUNCTION DOC]
        # Iterate over the info axis: returns an iterator over the labels (the column
        # labels of a DataFrame, the index values of a Series).
        """
        Iterate over info axis.

        Returns
        -------
        iterator
            Info axis as iterator.

        See Also
        --------
        DataFrame.items : Iterate over (column name, Series) pairs.
        DataFrame.itertuples : Iterate over DataFrame rows as namedtuples.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        >>> for x in df:
        ...     print(x)
        A
        B
        """
        return iter(self._info_axis)

    # can we get a better explanation of this?
    def keys(self) -> Index:
        # [FUNCTION DOC]
        # Return the info axis (the index for a Series, the columns for a DataFrame)
        # as an Index.
        """
        Get the 'info axis' (see Indexing for more).

        This is index for Series, columns for DataFrame.

        Returns
        -------
        Index
            Info axis.

        See Also
        --------
        DataFrame.index : The index (row labels) of the DataFrame.
        DataFrame.columns: The column labels of the DataFrame.

        Examples
        --------
        >>> d = pd.DataFrame(
        ...     data={"A": [1, 2, 3], "B": [0, 4, 8]}, index=["a", "b", "c"]
        ... )
        >>> d
           A  B
        a  1  0
        b  2  4
        c  3  8
        >>> d.keys()
        Index(['A', 'B'], dtype='str')
        """
        return self._info_axis

    def items(self):
        # [FUNCTION DOC]
        # Generator yielding pairs of each info axis label and the data stored under
        # it (a column for DataFrame, a scalar for Series). Looks every label up with
        # the item getter, so duplicate labels yield the looked-up result each time.
        """
        Iterate over (label, values) on info axis

        This is index for Series and columns for DataFrame.

        Returns
        -------
        Generator
        """
        for h in self._info_axis:
            yield h, self[h]

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Returns the length of the info axis.
        """Returns length of info axis"""
        return len(self._info_axis)

    @final
    def __contains__(self, key) -> bool:
        # [FUNCTION DOC]
        # Membership test: returns whether the key is in the info axis (labels, not
        # values).
        """True if the key is in the info axis"""
        return key in self._info_axis

    @property
    def empty(self) -> bool:
        # [FUNCTION DOC]
        # Property: true if any axis has length zero. Note a frame that holds only
        # missing values is not empty.
        """
        Indicator whether Series/DataFrame is empty.

        True if Series/DataFrame is entirely empty (no items), meaning any of the
        axes are of length 0.

        Returns
        -------
        bool
            If Series/DataFrame is empty, return True, if not return False.

        See Also
        --------
        Series.dropna : Return series without null values.
        DataFrame.dropna : Return DataFrame with labels on given axis omitted
            where (all or any) data are missing.

        Notes
        -----
        If Series/DataFrame contains only NaNs, it is still not considered empty. See
        the example below.

        Examples
        --------
        An example of an actual empty DataFrame. Notice the index is empty:

        >>> df_empty = pd.DataFrame({"A": []})
        >>> df_empty
        Empty DataFrame
        Columns: [A]
        Index: []
        >>> df_empty.empty
        True

        If we only have NaNs in our DataFrame, it is not considered empty! We
        will need to drop the NaNs to make the DataFrame empty:

        >>> df = pd.DataFrame({"A": [np.nan]})
        >>> df
            A
        0 NaN
        >>> df.empty
        False
        >>> df.dropna().empty
        True

        >>> ser_empty = pd.Series({"A": []})
        >>> ser_empty
        A    []
        dtype: object
        >>> ser_empty.empty
        False
        >>> ser_empty = pd.Series()
        >>> ser_empty.empty
        True
        """
        return any(len(self._get_axis(a)) == 0 for a in self._AXIS_ORDERS)

    # ----------------------------------------------------------------------
    # Array Interface

    # This is also set in IndexOpsMixin
    # GH#23114 Ensure ndarray.__op__(DataFrame) returns NotImplemented
    __array_priority__: int = 1000

    def __array__(
        self, dtype: npt.DTypeLike | None = None, copy: bool | None = None
    ) -> np.ndarray:
        # [FUNCTION DOC]
        # Implements the numpy array conversion protocol.
        # Parameters: dtype is an optional target numpy dtype; copy is None (copy if
        # needed), True (force) or False (never copy).
        # If copy is false and the data span several blocks and the object is
        # non-empty, raises a value error since a copy is unavoidable. The values are
        # fetched from the internal values property; without a copy flag they are
        # converted with the as-array function, otherwise with the array constructor
        # honoring the flag. When copying is not forced, the dtype change is a
        # view-compatible one and the object has a single block, a read-only view of
        # the result is returned so that later numpy writes cannot corrupt the pandas
        # data. Returns an ndarray.
        if copy is False and not self._mgr.is_single_block and not self.empty:
            # check this manually, otherwise ._values will already return a copy
            # and np.array(values, copy=False) will not raise an error
            raise ValueError(
                "Unable to avoid copy while creating an array as requested."
            )
        values = self._values
        if copy is None:
            # Note: branch avoids `copy=None` for NumPy 1.x support
            arr = np.asarray(values, dtype=dtype)
        else:
            arr = np.array(values, dtype=dtype, copy=copy)

        if (
            copy is not True
            and astype_is_view(values.dtype, arr.dtype)
            and self._mgr.is_single_block
        ):
            # Check if both conversions can be done without a copy
            if astype_is_view(self.dtypes.iloc[0], values.dtype) and astype_is_view(
                values.dtype, arr.dtype
            ):
                arr = arr.view()
                arr.flags.writeable = False
        return arr

    @final
    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any
    ):
        # [FUNCTION DOC]
        # Implements the numpy universal function protocol by delegating to the
        # array-like helper.
        # Parameters: ufunc is the numpy function; method is the call style name;
        # inputs are positional arguments; kwargs are keyword arguments. Returns
        # whatever the helper returns (a pandas object, tuple, or NotImplemented).
        return arraylike.array_ufunc(self, ufunc, method, *inputs, **kwargs)

    # ----------------------------------------------------------------------
    # Picklability

    @final
    def __getstate__(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Pickle support: returns the state dictionary used to serialize the object.
        # Contains the manager, the type tag, the list of metadata names, the user
        # attributes, a dictionary of flag values, and every metadata attribute value
        # (missing metadata attributes are stored as None).
        meta = {k: getattr(self, k, None) for k in self._metadata}
        return {
            "_mgr": self._mgr,
            "_typ": self._typ,
            "_metadata": self._metadata,
            "attrs": self.attrs,
            "_flags": {k: self.flags[k] for k in self.flags._keys},
            **meta,
        }

    @final
    def __setstate__(self, state) -> None:
        # [FUNCTION DOC]
        # Pickle support: restores the object from state.
        # Parameters: state is a manager (old style), or a dictionary produced by the
        # getstate method (or an older equivalent where the manager is stored under
        # the key data), or a legacy tuple.
        # Manager state is installed directly. For dictionary state with a type tag:
        # user attributes (a None value is replaced by an empty dictionary) and flags
        # are rebuilt, then internal names and metadata present in the state are set
        # first (in a stable-ish set order, skipping the flags entry) and all
        # remaining entries are set afterwards, all via direct attribute assignment.
        # Dictionaries without a type tag, and two-element legacy states, raise a
        # not-implemented error saying pickles older than 0.12 are unsupported.
        # Mutates self.
        if isinstance(state, BlockManager):
            self._mgr = state
        elif isinstance(state, dict):
            if "_data" in state and "_mgr" not in state:
                # compat for older pickles
                state["_mgr"] = state.pop("_data")
            typ = state.get("_typ")
            if typ is not None:
                attrs = state.get("_attrs", {})
                if attrs is None:  # should not happen, but better be on the safe side
                    attrs = {}
                object.__setattr__(self, "_attrs", attrs)
                flags = state.get("_flags", {"allows_duplicate_labels": True})
                object.__setattr__(self, "_flags", Flags(self, **flags))

                # set in the order of internal names
                # to avoid definitional recursion
                # e.g. say fill_value needing _mgr to be
                # defined
                meta = set(self._internal_names + self._metadata)
                for k in meta:
                    if k in state and k != "_flags":
                        v = state[k]
                        object.__setattr__(self, k, v)

                for k, v in state.items():
                    if k not in meta:
                        object.__setattr__(self, k, v)

            else:
                raise NotImplementedError("Pre-0.12 pickles are no longer supported")
        elif len(state) == 2:
            raise NotImplementedError("Pre-0.12 pickles are no longer supported")

    # ----------------------------------------------------------------------
    # Rendering Methods

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Fallback string representation: the class name followed by brackets
        # containing the comma separated printable forms of the iterated items of the
        # info axis. Concrete classes override this with richer output.
        # string representation based upon iterating over self
        # (since, by definition, `PandasContainers` are iterable)
        prepr = f"[{','.join(map(pprint_thing, self))}]"
        return f"{type(self).__name__}({prepr})"

    @final
    def _repr_latex_(self):
        # [FUNCTION DOC]
        # HELPER: notebook hook returning a LaTeX representation. Returns the output
        # of the latex export if the display option for the styler render
        # representation is set to latex, otherwise returns None. Intended for
        # nbconvert use.
        """
        Returns a LaTeX representation for a particular object.
        Mainly for use with nbconvert (jupyter notebook conversion to pdf).
        """
        if config.get_option("styler.render.repr") == "latex":
            return self.to_latex()
        else:
            return None

    @final
    def _repr_data_resource_(self):
        # [FUNCTION DOC]
        # HELPER: notebook-style data resource representation (not a real Jupyter hook
        # name). If the display option for the HTML table schema is enabled, takes the
        # first rows (limited by the maximum rows display option), serializes them to
        # JSON in table orientation and parses that JSON back into an ordered
        # dictionary, which is returned. Otherwise returns None.
        """
        Not a real Jupyter special repr method, but we use the same
        naming convention.
        """
        if config.get_option("display.html.table_schema"):
            data = self.head(config.get_option("display.max_rows"))

            as_json = data.to_json(orient="table")
            as_json = cast(str, as_json)
            return loads(as_json, object_pairs_hook=collections.OrderedDict)

    # ----------------------------------------------------------------------
    # I/O Methods

    @final
    def to_excel(
        self,
        excel_writer: FilePath | WriteExcelBuffer | ExcelWriter,
        *,
        sheet_name: str = "Sheet1",
        na_rep: str = "",
        float_format: str | None = None,
        columns: Sequence[Hashable] | None = None,
        header: Sequence[Hashable] | bool = True,
        index: bool = True,
        index_label: IndexLabel | None = None,
        startrow: int = 0,
        startcol: int = 0,
        engine: Literal["openpyxl", "xlsxwriter"] | None = None,
        merge_cells: bool = True,
        inf_rep: str = "inf",
        freeze_panes: tuple[int, int] | None = None,
        storage_options: StorageOptions | None = None,
        engine_kwargs: dict[str, Any] | None = None,
        autofilter: bool = False,
    ) -> None:
        # [FUNCTION DOC]
        # Write the object to an Excel sheet.
        # Parameters (all keyword-only except the first): excel_writer is a path,
        # buffer or an existing writer; sheet_name (default Sheet1); na_rep text for
        # missing values; float_format format string for floats; columns subset of
        # columns to write; header bool or alias list for column labels; index whether
        # to write the row labels; index_label column label(s) for index columns;
        # startrow and startcol give the zero-based upper-left cell; engine selects
        # openpyxl or xlsxwriter; merge_cells writes multi-level labels as merged
        # cells; inf_rep text for infinity; freeze_panes a tuple of rows and columns
        # to freeze; storage_options for remote file systems; engine_kwargs extra
        # options for the engine; autofilter whether to add a filter to the header
        # row.
        # A Series is first converted to a one-column frame. A formatter object from
        # the excel formatting module (imported lazily) is created from the frame and
        # the formatting arguments, and it performs the write. Returns None. Side
        # effect: writes a file or buffer.
        """
        Write object to an Excel sheet.

        To write a single object to an Excel .xlsx file it is only necessary to
        specify a target file name. To write to multiple sheets it is necessary to
        create an `ExcelWriter` object with a target file name, and specify a sheet
        in the file to write to.

        Multiple sheets may be written to by specifying unique `sheet_name`.
        With all data written to the file it is necessary to save the changes.
        Note that creating an `ExcelWriter` object with a file name that already exists
        will overwrite the existing file because the default mode is write.

        Parameters
        ----------
        excel_writer : path-like, file-like, or ExcelWriter object
            File path or existing ExcelWriter.
        sheet_name : str, default 'Sheet1'
            Name of sheet which will contain DataFrame.
        na_rep : str, default ''
            Missing data representation.
        float_format : str, optional
            Format string for floating point numbers. For example
            ``float_format="%.2f"`` will format 0.1234 to 0.12.
        columns : sequence or list of str, optional
            Columns to write.
        header : bool or list of str, default True
            Write out the column names. If a list of string is given it is
            assumed to be aliases for the column names.
        index : bool, default True
            Write row names (index).
        index_label : str or sequence, optional
            Column label for index column(s) if desired. If not specified, and
            `header` and `index` are True, then the index names are used. A
            sequence should be given if the DataFrame uses MultiIndex.
        startrow : int, default 0
            Upper left cell row to dump data frame.
        startcol : int, default 0
            Upper left cell column to dump data frame.
        engine : str, optional
            Write engine to use, 'openpyxl' or 'xlsxwriter'. You can also set this
            via the options ``io.excel.xlsx.writer`` or
            ``io.excel.xlsm.writer``.
        merge_cells : bool or 'columns', default False
            If True, write MultiIndex index and columns as merged cells.
            If 'columns', merge MultiIndex column cells only.
        inf_rep : str, default 'inf'
            Representation for infinity (there is no native representation for
            infinity in Excel).
        freeze_panes : tuple of int (length 2), optional
            Specifies the one-based bottommost row and rightmost column that
            is to be frozen.
        storage_options : dict, optional
            Extra options that make sense for a particular storage connection, e.g.
            host, port, username, password, etc. For HTTP(S) URLs the key-value pairs
            are forwarded to ``urllib.request.Request`` as header options. For other
            URLs (e.g. starting with "s3://", and "gcs://") the key-value pairs are
            forwarded to ``fsspec.open``. Please see ``fsspec`` and ``urllib`` for more
            details, and for more examples on storage options refer `here
            <https://pandas.pydata.org/docs/user_guide/io.html?
            highlight=storage_options#reading-writing-remote-files>`_.

        engine_kwargs : dict, optional
            Arbitrary keyword arguments passed to excel engine.
        autofilter : bool, default False
            If True, add automatic filters to all columns.

        See Also
        --------
        to_csv : Write DataFrame to a comma-separated values (csv) file.
        ExcelWriter : Class for writing DataFrame objects into excel sheets.
        read_excel : Read an Excel file into a pandas DataFrame.
        read_csv : Read a comma-separated values (csv) file into DataFrame.
        io.formats.style.Styler.to_excel : Add styles to Excel sheet.

        Notes
        -----
        For compatibility with :meth:`~DataFrame.to_csv`,
        to_excel serializes lists and dicts to strings before writing.

        Once a workbook has been saved it is not possible to write further
        data without rewriting the whole workbook.

        pandas will check the number of rows, columns,
        and cell character count does not exceed Excel's limitations.
        All other limitations must be checked by the user.

        Examples
        --------

        Create, write to and save a workbook:

        >>> df1 = pd.DataFrame(
        ...     [["a", "b"], ["c", "d"]],
        ...     index=["row 1", "row 2"],
        ...     columns=["col 1", "col 2"],
        ... )
        >>> df1.to_excel("output.xlsx")  # doctest: +SKIP

        To specify the sheet name:

        >>> df1.to_excel("output.xlsx", sheet_name="Sheet_name_1")  # doctest: +SKIP

        If you wish to write to more than one sheet in the workbook, it is
        necessary to specify an ExcelWriter object:

        >>> df2 = df1.copy()
        >>> with pd.ExcelWriter("output.xlsx") as writer:  # doctest: +SKIP
        ...     df1.to_excel(writer, sheet_name="Sheet_name_1")
        ...     df2.to_excel(writer, sheet_name="Sheet_name_2")

        ExcelWriter can also be used to append to an existing Excel file:

        >>> with pd.ExcelWriter("output.xlsx", mode="a") as writer:  # doctest: +SKIP
        ...     df1.to_excel(writer, sheet_name="Sheet_name_3")

        To set the library that is used to write the Excel file,
        you can pass the `engine` keyword (the default engine is
        automatically chosen depending on the file extension):

        >>> df1.to_excel("output1.xlsx", engine="xlsxwriter")  # doctest: +SKIP
        """
        if engine_kwargs is None:
            engine_kwargs = {}

        df = self if isinstance(self, ABCDataFrame) else self.to_frame()

        from pandas.io.formats.excel import ExcelFormatter

        formatter = ExcelFormatter(
            df,
            na_rep=na_rep,
            cols=columns,
            header=header,
            float_format=float_format,
            index=index,
            index_label=index_label,
            merge_cells=merge_cells,
            inf_rep=inf_rep,
            autofilter=autofilter,
        )
        formatter.write(
            excel_writer,
            sheet_name=sheet_name,
            startrow=startrow,
            startcol=startcol,
            freeze_panes=freeze_panes,
            engine=engine,
            storage_options=storage_options,
            engine_kwargs=engine_kwargs,
        )

    @final
    def to_json(
        self,
        path_or_buf: FilePath | WriteBuffer[bytes] | WriteBuffer[str] | None = None,
        *,
        orient: Literal["split", "records", "index", "table", "columns", "values"]
        | None = None,
        date_format: str | None = None,
        double_precision: int = 10,
        force_ascii: bool = True,
        date_unit: TimeUnit = "ms",
        default_handler: Callable[[Any], JSONSerializable] | None = None,
        lines: bool = False,
        compression: CompressionOptions = "infer",
        index: bool | None = None,
        indent: int | None = None,
        storage_options: StorageOptions | None = None,
        mode: Literal["a", "w"] = "w",
    ) -> str | None:
        # [FUNCTION DOC]
        # Convert the object to a JSON string or write it to a path or buffer.
        # Parameters (keyword only except the first): path_or_buf is a path, writable
        # buffer or None to return a string; orient selects the layout (split,
        # records, index, table, columns, values; default depends on the type);
        # date_format is iso or epoch (defaulted below); double_precision decimal
        # places for floats; force_ascii escape non-ascii characters; date_unit time
        # unit of epoch/iso output; default_handler callable for unserializable
        # objects; lines write line-delimited records (requires the records
        # orientation); compression, storage_options and mode for the output target;
        # index whether to include the index; indent number of spaces for pretty
        # printing.
        # Date format handling: when it is None, table orient selects iso, otherwise
        # epoch is selected and, if any column (or the Series) has a datetime or
        # timedelta dtype, a deprecation warning is emitted saying the default epoch
        # format will go away. An explicit epoch value also emits the deprecation
        # warning. Both warnings use the future-version warning class of pandas.
        # The indent is validated as a non-negative integer and replaced by zero when
        # None. Delegates to the JSON writer in the io package. Returns a string when
        # no target is given, otherwise None.
        """
        Convert the object to a JSON string.

        Note NaN's and None will be converted to null and datetime objects
        will be converted to UNIX timestamps.

        Parameters
        ----------
        path_or_buf : str, path object, file-like object, or None, default None
            String, path object (implementing os.PathLike[str]), or file-like
            object implementing a write() function. If None, the result is
            returned as a string.
        orient : str
            Indication of expected JSON string format.

            * Series:

                - default is 'index'
                - allowed values are: {'split', 'records', 'index', 'table'}.

            * DataFrame:

                - default is 'columns'
                - allowed values are: {'split', 'records', 'index', 'columns',
                  'values', 'table'}.

            * The format of the JSON string:

                - 'split' : dict like {'index' -> [index], 'columns' -> [columns],
                  'data' -> [values]}
                - 'records' : list like [{column -> value}, ... , {column -> value}]
                - 'index' : dict like {index -> {column -> value}}
                - 'columns' : dict like {column -> {index -> value}}
                - 'values' : just the values array
                - 'table' : dict like {'schema': {schema}, 'data': {data}}

                Describing the data, where data component is like ``orient='records'``.

        date_format : {None, 'epoch', 'iso'}
            Type of date conversion. 'epoch' = epoch milliseconds,
            'iso' = ISO8601. The default depends on the `orient`. For
            ``orient='table'``, the default is 'iso'. For all other orients,
            the default is 'epoch'.

            .. deprecated:: 3.0.0
                'epoch' date format is deprecated and will be removed in a future
                version, please use 'iso' instead.

        double_precision : int, default 10
            The number of decimal places to use when encoding
            floating point values. The possible maximal value is 15.
            Passing double_precision greater than 15 will raise a ValueError.
        force_ascii : bool, default True
            Force encoded string to be ASCII.
        date_unit : str, default 'ms' (milliseconds)
            The time unit to encode to, governs timestamp and ISO8601
            precision.  One of 's', 'ms', 'us', 'ns' for second, millisecond,
            microsecond, and nanosecond respectively.
        default_handler : callable, default None
            Handler to call if object cannot otherwise be converted to a
            suitable format for JSON. Should receive a single argument which is
            the object to convert and return a serialisable object.
        lines : bool, default False
            If 'orient' is 'records' write out line-delimited json format. Will
            throw ValueError if incorrect 'orient' since others are not
            list-like.

        compression : str or dict, default 'infer'
            For on-the-fly compression of the output data. If 'infer' and
            'path_or_buf' is path-like, then detect compression from the following
            extensions: '.gz',
            '.bz2', '.zip', '.xz', '.zst', '.tar', '.tar.gz', '.tar.xz' or '.tar.bz2'
            (otherwise no compression).
            Set to ``None`` for no compression.
            Can also be a dict with key ``'method'`` set to one of
            {``'zip'``, ``'gzip'``, ``'bz2'``, ``'zstd'``, ``'xz'``, ``'tar'``} and
            other key-value pairs are forwarded to
            ``zipfile.ZipFile``, ``gzip.GzipFile``,
            ``bz2.BZ2File``, ``zstandard.ZstdCompressor``, ``lzma.LZMAFile`` or
            ``tarfile.TarFile``, respectively.
            As an example, the following could be passed for faster compression and
            to create a reproducible gzip archive:
            ``compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1}``.

        index : bool or None, default None
            The index is only used when 'orient' is 'split', 'index', 'column',
            or 'table'. Of these, 'index' and 'column' do not support
            `index=False`. The string 'index' as a column name with empty :class:`Index`
            or if it is 'index' will raise a ``ValueError``.

        indent : int, optional
           Length of whitespace used to indent each record.

        storage_options : dict, optional
            Extra options that make sense for a particular storage connection, e.g.
            host, port, username, password, etc. For HTTP(S) URLs the key-value pairs
            are forwarded to ``urllib.request.Request`` as header options. For other
            URLs (e.g. starting with "s3://", and "gcs://") the key-value pairs are
            forwarded to ``fsspec.open``. Please see ``fsspec`` and ``urllib`` for more
            details, and for more examples on storage options refer `here
            <https://pandas.pydata.org/docs/user_guide/io.html?
            highlight=storage_options#reading-writing-remote-files>`_.

        mode : str, default 'w' (writing)
            Specify the IO mode for output when supplying a path_or_buf.
            Accepted args are 'w' (writing) and 'a' (append) only.
            mode='a' is only supported when lines is True and orient is 'records'.

        Returns
        -------
        None or str
            If path_or_buf is None, returns the resulting json format as a
            string. Otherwise returns None.

        See Also
        --------
        read_json : Convert a JSON string to pandas object.

        Notes
        -----
        The behavior of ``indent=0`` varies from the stdlib, which does not
        indent the output but does insert newlines. Currently, ``indent=0``
        and the default ``indent=None`` are equivalent in pandas, though this
        may change in a future release.

        ``orient='table'`` contains a 'pandas_version' field under 'schema'.
        This stores the version of `pandas` used in the latest revision of the
        schema.

        Examples
        --------
        >>> from json import loads, dumps
        >>> df = pd.DataFrame(
        ...     [["a", "b"], ["c", "d"]],
        ...     index=["row 1", "row 2"],
        ...     columns=["col 1", "col 2"],
        ... )

        >>> result = df.to_json(orient="split")
        >>> parsed = loads(result)
        >>> dumps(parsed, indent=4)  # doctest: +SKIP
        {
            "columns": [
                "col 1",
                "col 2"
            ],
            "index": [
                "row 1",
                "row 2"
            ],
            "data": [
                [
                    "a",
                    "b"
                ],
                [
                    "c",
                    "d"
                ]
            ]
        }

        Encoding/decoding a Dataframe using ``'records'`` formatted JSON.
        Note that index labels are not preserved with this encoding.

        >>> result = df.to_json(orient="records")
        >>> parsed = loads(result)
        >>> dumps(parsed, indent=4)  # doctest: +SKIP
        [
            {
                "col 1": "a",
                "col 2": "b"
            },
            {
                "col 1": "c",
                "col 2": "d"
            }
        ]

        Encoding/decoding a Dataframe using ``'index'`` formatted JSON:

        >>> result = df.to_json(orient="index")
        >>> parsed = loads(result)
        >>> dumps(parsed, indent=4)  # doctest: +SKIP
        {
            "row 1": {
                "col 1": "a",
                "col 2": "b"
            },
            "row 2": {
                "col 1": "c",
                "col 2": "d"
            }
        }

        Encoding/decoding a Dataframe using ``'columns'`` formatted JSON:

        >>> result = df.to_json(orient="columns")
        >>> parsed = loads(result)
        >>> dumps(parsed, indent=4)  # doctest: +SKIP
        {
            "col 1": {
                "row 1": "a",
                "row 2": "c"
            },
            "col 2": {
                "row 1": "b",
                "row 2": "d"
            }
        }

        Encoding/decoding a Dataframe using ``'values'`` formatted JSON:

        >>> result = df.to_json(orient="values")
        >>> parsed = loads(result)
        >>> dumps(parsed, indent=4)  # doctest: +SKIP
        [
            [
                "a",
                "b"
            ],
            [
                "c",
                "d"
            ]
        ]

        Encoding with Table Schema:

        >>> result = df.to_json(orient="table")
        >>> parsed = loads(result)
        >>> dumps(parsed, indent=4)  # doctest: +SKIP
        {
            "schema": {
                "fields": [
                    {
                        "name": "index",
                        "type": "string"
                    },
                    {
                        "name": "col 1",
                        "type": "string"
                    },
                    {
                        "name": "col 2",
                        "type": "string"
                    }
                ],
                "primaryKey": [
                    "index"
                ],
                "pandas_version": "1.4.0"
            },
            "data": [
                {
                    "index": "row 1",
                    "col 1": "a",
                    "col 2": "b"
                },
                {
                    "index": "row 2",
                    "col 1": "c",
                    "col 2": "d"
                }
            ]
        }
        """
        from pandas.io import json

        if date_format is None and orient == "table":
            date_format = "iso"
        elif date_format is None:
            date_format = "epoch"
            dtypes = self.dtypes if self.ndim == 2 else [self.dtype]
            if any(dtype.kind in "mM" for dtype in dtypes):
                warnings.warn(
                    "The default 'epoch' date format is deprecated and will be removed "
                    "in a future version, please use 'iso' date format instead.",
                    Pandas4Warning,
                    stacklevel=find_stack_level(),
                )
        elif date_format == "epoch":
            # GH#57063
            warnings.warn(
                "'epoch' date format is deprecated and will be removed in a future "
                "version, please use 'iso' date format instead.",
                Pandas4Warning,
                stacklevel=find_stack_level(),
            )

        config.is_nonnegative_int(indent)
        indent = indent or 0

        return json.to_json(
            path_or_buf=path_or_buf,
            obj=self,
            orient=orient,
            date_format=date_format,
            double_precision=double_precision,
            force_ascii=force_ascii,
            date_unit=date_unit,
            default_handler=default_handler,
            lines=lines,
            compression=compression,
            index=index,
            indent=indent,
            storage_options=storage_options,
            mode=mode,
        )

    @final
    def to_hdf(
        self,
        path_or_buf: FilePath | HDFStore,
        *,
        key: str,
        mode: Literal["a", "w", "r+"] = "a",
        complevel: int | None = None,
        complib: Literal["zlib", "lzo", "bzip2", "blosc"] | None = None,
        append: bool = False,
        format: Literal["fixed", "table"] | None = None,
        index: bool = True,
        min_itemsize: int | dict[str, int] | None = None,
        nan_rep=None,
        dropna: bool | None = None,
        data_columns: Literal[True] | list[str] | None = None,
        errors: OpenFileErrors = "strict",
        encoding: str = "UTF-8",
    ) -> None:
        # [FUNCTION DOC]
        # Write the object to an HDF5 file through an HDFStore.
        # Parameters: path_or_buf is a file path or an open store; key (keyword only,
        # required) is the group identifier; mode is a, w or r+; complevel and complib
        # set compression; append adds to an existing table; format is fixed or table;
        # index creates table indexes; min_itemsize sets minimum string column widths;
        # nan_rep is the string used for missing values; dropna drops all-missing rows
        # when appending; data_columns lists columns to make queryable; errors and
        # encoding control string encoding.
        # Pure delegation to the pytables module (imported lazily); all arguments are
        # forwarded unchanged. Returns None. Side effect: writes a file.
        """
        Write the contained data to an HDF5 file using HDFStore.

        Hierarchical Data Format (HDF) is self-describing, allowing an
        application to interpret the structure and contents of a file with
        no outside information. One HDF file can hold a mix of related objects
        which can be accessed as a group or as individual objects.

        In order to add another DataFrame or Series to an existing HDF file
        please use append mode and a different a key.

        .. warning::

           One can store a subclass of ``DataFrame`` or ``Series`` to HDF5,
           but the type of the subclass is lost upon storing.

        For more information see the :ref:`user guide <io.hdf5>`.

        Parameters
        ----------
        path_or_buf : str or pandas.HDFStore
            File path or HDFStore object.
        key : str
            Identifier for the group in the store.
        mode : {'a', 'w', 'r+'}, default 'a'
            Mode to open file:

            - 'w': write, a new file is created (an existing file with
              the same name would be deleted).
            - 'a': append, an existing file is opened for reading and
              writing, and if the file does not exist it is created.
            - 'r+': similar to 'a', but the file must already exist.
        complevel : {0-9}, default None
            Specifies a compression level for data.
            A value of 0 or None disables compression.
        complib : {'zlib', 'lzo', 'bzip2', 'blosc'}, default 'zlib'
            Specifies the compression library to be used.
            These additional compressors for Blosc are supported
            (default if no compressor specified: 'blosc:blosclz'):
            {'blosc:blosclz', 'blosc:lz4', 'blosc:lz4hc', 'blosc:snappy',
            'blosc:zlib', 'blosc:zstd'}.
            Specifying a compression library which is not available issues
            a ValueError.
        append : bool, default False
            For Table formats, append the input data to the existing.
        format : {'fixed', 'table', None}, default 'fixed'
            Possible values:

            - 'fixed': Fixed format. Fast writing/reading. Not-appendable,
              nor searchable.
            - 'table': Table format. Write as a PyTables Table structure
              which may perform worse but allow more flexible operations
              like searching / selecting subsets of the data.
            - If None, pd.get_option('io.hdf.default_format') is checked,
              followed by fallback to "fixed".
        index : bool, default True
            Write DataFrame index as a column.
        min_itemsize : dict or int, optional
            Map column names to minimum string sizes for columns.
        nan_rep : Any, optional
            How to represent null values as str.
            Not allowed with append=True.
        dropna : bool, default False, optional
            Remove missing values.
        data_columns : list of columns or True, optional
            List of columns to create as indexed data columns for on-disk
            queries, or True to use all columns. By default only the axes
            of the object are indexed. See
            :ref:`Query via data columns<io.hdf5-query-data-columns>`. for
            more information.
            Applicable only to format='table'.
        errors : str, default 'strict'
            Specifies how encoding and decoding errors are to be handled.
            See the errors argument for :func:`open` for a full list
            of options.
        encoding : str, default "UTF-8"
            Set character encoding.

        See Also
        --------
        read_hdf : Read from HDF file.
        DataFrame.to_orc : Write a DataFrame to the binary orc format.
        DataFrame.to_parquet : Write a DataFrame to the binary parquet format.
        DataFrame.to_sql : Write to a SQL table.
        DataFrame.to_feather : Write out feather-format for DataFrames.
        DataFrame.to_csv : Write out to a csv file.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "b", "c"]
        ... )  # doctest: +SKIP
        >>> df.to_hdf("data.h5", key="df", mode="w")  # doctest: +SKIP

        We can add another object to the same file:

        >>> s = pd.Series([1, 2, 3, 4])  # doctest: +SKIP
        >>> s.to_hdf("data.h5", key="s")  # doctest: +SKIP

        Reading from HDF file:

        >>> pd.read_hdf("data.h5", "df")  # doctest: +SKIP
        A  B
        a  1  4
        b  2  5
        c  3  6
        >>> pd.read_hdf("data.h5", "s")  # doctest: +SKIP
        0    1
        1    2
        2    3
        3    4
        dtype: int64
        """
        from pandas.io import pytables

        # Argument 3 to "to_hdf" has incompatible type "NDFrame"; expected
        # "Union[DataFrame, Series]" [arg-type]
        pytables.to_hdf(
            path_or_buf,
            key,
            self,  # type: ignore[arg-type]
            mode=mode,
            complevel=complevel,
            complib=complib,
            append=append,
            format=format,
            index=index,
            min_itemsize=min_itemsize,
            nan_rep=nan_rep,
            dropna=dropna,
            data_columns=data_columns,
            errors=errors,
            encoding=encoding,
        )

    @final
    def to_sql(
        self,
        name: str,
        con,
        *,
        schema: str | None = None,
        if_exists: Literal["fail", "replace", "append", "delete_rows"] = "fail",
        index: bool = True,
        index_label: IndexLabel | None = None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        method: Literal["multi"] | Callable | None = None,
    ) -> int | None:
        # [FUNCTION DOC]
        # Write the records of the object to a SQL database.
        # Parameters: name is the target table name; con is a SQLAlchemy connectable,
        # a DBAPI connection or similar; schema is an optional database schema;
        # if_exists says what to do if the table exists (fail, replace, append,
        # delete_rows); index whether to write the index as a column; index_label
        # column label(s) for the index; chunksize rows per batch; dtype column type
        # mapping; method selects the insertion strategy (None, multi, or a callable).
        # Delegates to the sql module (imported lazily). Returns the number of
        # affected rows when the driver reports it, otherwise None. Side effect:
        # executes statements on the database.
        """
        Write records stored in a DataFrame to a SQL database.

        Databases supported by SQLAlchemy [1]_ are supported. Tables can be
        newly created, appended to, or overwritten.

        .. warning::
            The pandas library does not attempt to sanitize inputs provided via a to_sql call.
            Please refer to the documentation for the underlying database driver to see if it
            will properly prevent injection, or alternatively be advised of a security risk when
            executing arbitrary commands in a to_sql call.

        Parameters
        ----------
        name : str
            Name of SQL table.
        con : ADBC connection, sqlalchemy.engine.(Engine or Connection) or sqlite3.Connection
            ADBC provides high performance I/O with native type support, where available.
            Using SQLAlchemy makes it possible to use any DB supported by that
            library. Legacy support is provided for sqlite3.Connection objects. The user
            is responsible for engine disposal and connection closure for the SQLAlchemy
            connectable. See `here \
                <https://docs.sqlalchemy.org/en/20/core/connections.html>`_.
            If passing a sqlalchemy.engine.Connection which is already in a transaction,
            the transaction will not be committed.  If passing a sqlite3.Connection,
            it will not be possible to roll back the record insertion.

        schema : str, optional
            Specify the schema (if database flavor supports this). If None, use
            default schema.
        if_exists : {'fail', 'replace', 'append', 'delete_rows'}, default 'fail'
            How to behave if the table already exists.

            * fail: Raise a ValueError.
            * replace: Drop the table before inserting new values.
            * append: Insert new values to the existing table.
            * delete_rows: If a table exists, delete all records and insert data.

        index : bool, default True
            Write DataFrame index as a column. Uses `index_label` as the column
            name in the table. Creates a table index for this column.
        index_label : str or sequence, default None
            Column label for index column(s). If None is given (default) and
            `index` is True, then the index names are used.
            A sequence should be given if the DataFrame uses MultiIndex.
        chunksize : int, optional
            Specify the number of rows in each batch to be written to the database connection at a time.
            By default, all rows will be written at once. Also see the method keyword.
        dtype : dict or scalar, optional
            Specifying the datatype for columns. If a dictionary is used, the
            keys should be the column names and the values should be the
            SQLAlchemy types or strings for the sqlite3 legacy mode. If a
            scalar is provided, it will be applied to all columns.
        method : {None, 'multi', callable}, optional
            Controls the SQL insertion clause used:

            * None : Uses standard SQL ``INSERT`` clause (one per row).
            * 'multi': Pass multiple values in a single ``INSERT`` clause.
            * callable with signature ``(pd_table, conn, keys, data_iter)``.

            Details and a sample callable implementation can be found in the
            section :ref:`insert method <io.sql.method>`.

        Returns
        -------
        None or int
            Number of rows affected by to_sql. None is returned if the callable
            passed into ``method`` does not return an integer number of rows.

            The number of returned rows affected is the sum of the ``rowcount``
            attribute of ``sqlite3.Cursor`` or SQLAlchemy connectable which may not
            reflect the exact number of written rows as stipulated in the
            `sqlite3 <https://docs.python.org/3/library/sqlite3.html#sqlite3.Cursor.rowcount>`__ or
            `SQLAlchemy <https://docs.sqlalchemy.org/en/20/core/connections.html#sqlalchemy.engine.CursorResult.rowcount>`__.

        Raises
        ------
        ValueError
            When the table already exists and `if_exists` is 'fail' (the
            default).

        See Also
        --------
        read_sql : Read a DataFrame from a table.

        Notes
        -----
        Timezone aware datetime columns will be written as
        ``Timestamp with timezone`` type with SQLAlchemy if supported by the
        database. Otherwise, the datetimes will be stored as timezone unaware
        timestamps local to the original timezone.

        Not all datastores support ``method="multi"``. Oracle, for example,
        does not support multi-value insert.

        References
        ----------
        .. [1] https://docs.sqlalchemy.org
        .. [2] https://www.python.org/dev/peps/pep-0249/

        Examples
        --------
        Create an in-memory SQLite database.

        >>> from sqlalchemy import create_engine
        >>> engine = create_engine('sqlite://', echo=False)

        Create a table from scratch with 3 rows.

        >>> df = pd.DataFrame({'name' : ['User 1', 'User 2', 'User 3']})
        >>> df
             name
        0  User 1
        1  User 2
        2  User 3

        >>> df.to_sql(name='users', con=engine)
        3
        >>> from sqlalchemy import text
        >>> with engine.connect() as conn:
        ...     conn.execute(text("SELECT * FROM users")).fetchall()
        [(0, 'User 1'), (1, 'User 2'), (2, 'User 3')]

        An `sqlalchemy.engine.Connection` can also be passed to `con`:

        >>> with engine.begin() as connection:
        ...     df1 = pd.DataFrame({'name' : ['User 4', 'User 5']})
        ...     df1.to_sql(name='users', con=connection, if_exists='append')
        2

        This is allowed to support operations that require that the same
        DBAPI connection is used for the entire operation.

        >>> df2 = pd.DataFrame({'name' : ['User 6', 'User 7']})
        >>> df2.to_sql(name='users', con=engine, if_exists='append')
        2
        >>> with engine.connect() as conn:
        ...     conn.execute(text("SELECT * FROM users")).fetchall()
        [(0, 'User 1'), (1, 'User 2'), (2, 'User 3'),
         (0, 'User 4'), (1, 'User 5'), (0, 'User 6'),
         (1, 'User 7')]

        Overwrite the table with just ``df2``.

        >>> df2.to_sql(name='users', con=engine, if_exists='replace',
        ...            index_label='id')
        2
        >>> with engine.connect() as conn:
        ...     conn.execute(text("SELECT * FROM users")).fetchall()
        [(0, 'User 6'), (1, 'User 7')]

        Delete all rows before inserting new records with ``df3``

        >>> df3 = pd.DataFrame({"name": ['User 8', 'User 9']})
        >>> df3.to_sql(name='users', con=engine, if_exists='delete_rows',
        ...            index_label='id')
        2
        >>> with engine.connect() as conn:
        ...     conn.execute(text("SELECT * FROM users")).fetchall()
        [(0, 'User 8'), (1, 'User 9')]

        Use ``method`` to define a callable insertion method to do nothing
        if there's a primary key conflict on a table in a PostgreSQL database.

        >>> from sqlalchemy.dialects.postgresql import insert
        >>> def insert_on_conflict_nothing(table, conn, keys, data_iter):
        ...     # "a" is the primary key in "conflict_table"
        ...     data = [dict(zip(keys, row)) for row in data_iter]
        ...     stmt = insert(table.table).values(data).on_conflict_do_nothing(index_elements=["a"])
        ...     result = conn.execute(stmt)
        ...     return result.rowcount
        >>> df_conflict.to_sql(name="conflict_table", con=conn, if_exists="append",  # noqa: F821
        ...                    method=insert_on_conflict_nothing)  # doctest: +SKIP
        0

        For MySQL, a callable to update columns ``b`` and ``c`` if there's a conflict
        on a primary key.

        >>> from sqlalchemy.dialects.mysql import insert   # noqa: F811
        >>> def insert_on_conflict_update(table, conn, keys, data_iter):
        ...     # update columns "b" and "c" on primary key conflict
        ...     data = [dict(zip(keys, row)) for row in data_iter]
        ...     stmt = (
        ...         insert(table.table)
        ...         .values(data)
        ...     )
        ...     stmt = stmt.on_duplicate_key_update(b=stmt.inserted.b, c=stmt.inserted.c)
        ...     result = conn.execute(stmt)
        ...     return result.rowcount
        >>> df_conflict.to_sql(name="conflict_table", con=conn, if_exists="append",  # noqa: F821
        ...                    method=insert_on_conflict_update)  # doctest: +SKIP
        2

        Specify the dtype (especially useful for integers with missing values).
        Notice that while pandas is forced to store the data as floating point,
        the database supports nullable integers. When fetching the data with
        Python, we get back integer scalars.

        >>> df = pd.DataFrame({"A": [1, None, 2]})
        >>> df
             A
        0  1.0
        1  NaN
        2  2.0

        >>> from sqlalchemy.types import Integer
        >>> df.to_sql(name='integers', con=engine, index=False,
        ...           dtype={"A": Integer()})
        3

        >>> with engine.connect() as conn:
        ...     conn.execute(text("SELECT * FROM integers")).fetchall()
        [(1,), (None,), (2,)]

        .. versionadded:: 2.2.0

           pandas now supports writing via ADBC drivers

        >>> df = pd.DataFrame({'name' : ['User 10', 'User 11', 'User 12']})
        >>> df
              name
        0  User 10
        1  User 11
        2  User 12

        >>> from adbc_driver_sqlite import dbapi  # doctest:+SKIP
        >>> with dbapi.connect("sqlite://") as conn:  # doctest:+SKIP
        ...     df.to_sql(name="users", con=conn)
        3
        """  # noqa: E501
        from pandas.io import sql

        return sql.to_sql(
            self,
            name,
            con,
            schema=schema,
            if_exists=if_exists,
            index=index,
            index_label=index_label,
            chunksize=chunksize,
            dtype=dtype,
            method=method,
        )

    @final
    def to_pickle(
        self,
        path: FilePath | WriteBuffer[bytes],
        *,
        compression: CompressionOptions = "infer",
        protocol: int = pickle.HIGHEST_PROTOCOL,
        storage_options: StorageOptions | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Serialize the object to a pickle file.
        # Parameters: path is a file path or a binary buffer; compression is infer or
        # a compression specification; protocol is the pickle protocol (default
        # highest available); storage_options for remote files.
        # Delegates to the pickle io module (imported lazily). Returns None. Side
        # effect: writes the file.
        """
        Pickle (serialize) object to file.

        Parameters
        ----------
        path : str, path object, or file-like object
            String, path object (implementing ``os.PathLike[str]``), or file-like
            object implementing a binary ``write()`` function. File path where
            the pickled object will be stored.

        compression : str or dict, default 'infer'
            For on-the-fly compression of the output data. If 'infer' and
            'path_or_buf' is path-like, then detect compression from the following
            extensions: '.gz',
            '.bz2', '.zip', '.xz', '.zst', '.tar', '.tar.gz', '.tar.xz' or '.tar.bz2'
            (otherwise no compression).
            Set to ``None`` for no compression.
            Can also be a dict with key ``'method'`` set to one of
            {``'zip'``, ``'gzip'``, ``'bz2'``, ``'zstd'``, ``'xz'``, ``'tar'``} and
            other key-value pairs are forwarded to
            ``zipfile.ZipFile``, ``gzip.GzipFile``,
            ``bz2.BZ2File``, ``zstandard.ZstdCompressor``, ``lzma.LZMAFile`` or
            ``tarfile.TarFile``, respectively.
            As an example, the following could be passed for faster compression and
            to create a reproducible gzip archive:
            ``compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1}``.

        protocol : int
            Int which indicates which protocol should be used by the pickler,
            default HIGHEST_PROTOCOL (see [1]_ paragraph 12.1.2). The possible
            values are 0, 1, 2, 3, 4, 5. A negative value for the protocol
            parameter is equivalent to setting its value to HIGHEST_PROTOCOL.

            .. [1] https://docs.python.org/3/library/pickle.html.

        storage_options : dict, optional
            Extra options that make sense for a particular storage connection, e.g.
            host, port, username, password, etc. For HTTP(S) URLs the key-value pairs
            are forwarded to ``urllib.request.Request`` as header options. For other
            URLs (e.g. starting with "s3://", and "gcs://") the key-value pairs are
            forwarded to ``fsspec.open``. Please see ``fsspec`` and ``urllib`` for more
            details, and for more examples on storage options refer `here
            <https://pandas.pydata.org/docs/user_guide/io.html?
            highlight=storage_options#reading-writing-remote-files>`_.

        See Also
        --------
        read_pickle : Load pickled pandas object (or any object) from file.
        DataFrame.to_hdf : Write DataFrame to an HDF5 file.
        DataFrame.to_sql : Write DataFrame to a SQL database.
        DataFrame.to_parquet : Write a DataFrame to the binary parquet format.

        Examples
        --------
        >>> original_df = pd.DataFrame(
        ...     {"foo": range(5), "bar": range(5, 10)}
        ... )  # doctest: +SKIP
        >>> original_df  # doctest: +SKIP
           foo  bar
        0    0    5
        1    1    6
        2    2    7
        3    3    8
        4    4    9
        >>> original_df.to_pickle("./dummy.pkl")  # doctest: +SKIP

        >>> unpickled_df = pd.read_pickle("./dummy.pkl")  # doctest: +SKIP
        >>> unpickled_df  # doctest: +SKIP
           foo  bar
        0    0    5
        1    1    6
        2    2    7
        3    3    8
        4    4    9
        """
        from pandas.io.pickle import to_pickle

        to_pickle(
            self,
            path,
            compression=compression,
            protocol=protocol,
            storage_options=storage_options,
        )

    @final
    def to_clipboard(
        self, *, excel: bool = True, sep: str | None = None, **kwargs
    ) -> None:
        # [FUNCTION DOC]
        # Copy the object to the system clipboard as text.
        # Parameters (keyword only): excel (default true) writes tab-separated text
        # that pastes into spreadsheets; sep is an optional field delimiter (None lets
        # the clipboard module choose); extra keyword arguments are passed through to
        # the underlying csv-style writer.
        # Delegates to the clipboards module (imported lazily). Returns None. Side
        # effect: modifies the clipboard.
        r"""
        Copy object to the system clipboard.

        Write a text representation of object to the system clipboard.
        This can be pasted into Excel, for example.

        Parameters
        ----------
        excel : bool, default True
            Produce output in a csv format for easy pasting into excel.

            - True, use the provided separator for csv pasting.
            - False, write a string representation of the object to the clipboard.

        sep : str, default ``'\t'``
            Field delimiter.
        **kwargs
            These parameters will be passed to DataFrame.to_csv.

        See Also
        --------
        DataFrame.to_csv : Write a DataFrame to a comma-separated values
            (csv) file.
        read_clipboard : Read text from clipboard and pass to read_csv.

        Notes
        -----
        Requirements for your platform.

          - Linux : `xclip`, or `xsel` (with `PyQt4` modules)
          - Windows : none
          - macOS : none

        This method uses the processes developed for the package `pyperclip`. A
        solution to render any output string format is given in the examples.

        Examples
        --------
        Copy the contents of a DataFrame to the clipboard.

        >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"])

        >>> df.to_clipboard(sep=",")  # doctest: +SKIP
        ... # Wrote the following to the system clipboard:
        ... # ,A,B,C
        ... # 0,1,2,3
        ... # 1,4,5,6

        We can omit the index by passing the keyword `index` and setting
        it to false.

        >>> df.to_clipboard(sep=",", index=False)  # doctest: +SKIP
        ... # Wrote the following to the system clipboard:
        ... # A,B,C
        ... # 1,2,3
        ... # 4,5,6

        Using the original `pyperclip` package for any string output format.

        .. code-block:: python

           import pyperclip

           html = df.style.to_html()
           pyperclip.copy(html)
        """
        from pandas.io import clipboards

        clipboards.to_clipboard(self, excel=excel, sep=sep, **kwargs)

    @final
    def to_xarray(self):
        # [FUNCTION DOC]
        # Convert the object to an xarray container.
        # Imports the optional xarray dependency (raising the standard
        # missing-dependency import error if unavailable). A one-dimensional object
        # becomes a data array built from the series; a two-dimensional one becomes a
        # dataset built from the frame. Returns that xarray object.
        """
        Return an xarray object from the pandas object.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            Data in the pandas structure converted to Dataset if the object is
            a DataFrame, or a DataArray if the object is a Series.

        See Also
        --------
        DataFrame.to_hdf : Write DataFrame to an HDF5 file.
        DataFrame.to_parquet : Write a DataFrame to the binary parquet format.

        Notes
        -----
        See the `xarray docs <https://xarray.pydata.org/en/stable/>`__

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     [
        ...         ("falcon", "bird", 389.0, 2),
        ...         ("parrot", "bird", 24.0, 2),
        ...         ("lion", "mammal", 80.5, 4),
        ...         ("monkey", "mammal", np.nan, 4),
        ...     ],
        ...     columns=["name", "class", "max_speed", "num_legs"],
        ... )
        >>> df
             name   class  max_speed  num_legs
        0  falcon    bird      389.0         2
        1  parrot    bird       24.0         2
        2    lion  mammal       80.5         4
        3  monkey  mammal        NaN         4

        >>> df.to_xarray()  # doctest: +SKIP
        <xarray.Dataset>
        Dimensions:    (index: 4)
        Coordinates:
          * index      (index) int64 32B 0 1 2 3
        Data variables:
            name       (index) object 32B 'falcon' 'parrot' 'lion' 'monkey'
            class      (index) object 32B 'bird' 'bird' 'mammal' 'mammal'
            max_speed  (index) float64 32B 389.0 24.0 80.5 nan
            num_legs   (index) int64 32B 2 2 4 4

        >>> df["max_speed"].to_xarray()  # doctest: +SKIP
        <xarray.DataArray 'max_speed' (index: 4)>
        array([389. ,  24. ,  80.5,   nan])
        Coordinates:
          * index    (index) int64 0 1 2 3

        >>> dates = pd.to_datetime(
        ...     ["2018-01-01", "2018-01-01", "2018-01-02", "2018-01-02"]
        ... )
        >>> df_multiindex = pd.DataFrame(
        ...     {
        ...         "date": dates,
        ...         "animal": ["falcon", "parrot", "falcon", "parrot"],
        ...         "speed": [350, 18, 361, 15],
        ...     }
        ... )
        >>> df_multiindex = df_multiindex.set_index(["date", "animal"])

        >>> df_multiindex
                           speed
        date       animal
        2018-01-01 falcon    350
                   parrot     18
        2018-01-02 falcon    361
                   parrot     15

        >>> df_multiindex.to_xarray()  # doctest: +SKIP
        <xarray.Dataset>
        Dimensions:  (date: 2, animal: 2)
        Coordinates:
          * date     (date) datetime64[s] 2018-01-01 2018-01-02
          * animal   (animal) object 'falcon' 'parrot'
        Data variables:
            speed    (date, animal) int64 350 18 361 15
        """
        xarray = import_optional_dependency("xarray")

        if self.ndim == 1:
            return xarray.DataArray.from_series(self)
        else:
            return xarray.Dataset.from_dataframe(self)

    @overload
    def to_latex(
        self,
        buf: None = ...,
        *,
        columns: Sequence[Hashable] | None = ...,
        header: bool | SequenceNotStr[str] = ...,
        index: bool = ...,
        na_rep: str = ...,
        formatters: FormattersType | None = ...,
        float_format: FloatFormatType | None = ...,
        sparsify: bool | None = ...,
        index_names: bool = ...,
        bold_rows: bool = ...,
        column_format: str | None = ...,
        longtable: bool | None = ...,
        escape: bool | None = ...,
        encoding: str | None = ...,
        decimal: str = ...,
        multicolumn: bool | None = ...,
        multicolumn_format: str | None = ...,
        multirow: bool | None = ...,
        caption: str | tuple[str, str] | None = ...,
        label: str | None = ...,
        position: str | None = ...,
    ) -> str:
        # [FUNCTION DOC]
        # Typing overload stub only (body is a placeholder); the behavior is defined
        # by the implementation that follows these overloads. to_latex variant with no
        # buffer; returns the LaTeX text as a string.
        ...

    @overload
    def to_latex(
        self,
        buf: FilePath | WriteBuffer[str],
        *,
        columns: Sequence[Hashable] | None = ...,
        header: bool | SequenceNotStr[str] = ...,
        index: bool = ...,
        na_rep: str = ...,
        formatters: FormattersType | None = ...,
        float_format: FloatFormatType | None = ...,
        sparsify: bool | None = ...,
        index_names: bool = ...,
        bold_rows: bool = ...,
        column_format: str | None = ...,
        longtable: bool | None = ...,
        escape: bool | None = ...,
        encoding: str | None = ...,
        decimal: str = ...,
        multicolumn: bool | None = ...,
        multicolumn_format: str | None = ...,
        multirow: bool | None = ...,
        caption: str | tuple[str, str] | None = ...,
        label: str | None = ...,
        position: str | None = ...,
    ) -> None:
        # [FUNCTION DOC]
        # Typing overload stub only (body is a placeholder); the behavior is defined
        # by the implementation that follows these overloads. to_latex variant with a
        # path or buffer; returns nothing.
        ...

    @final
    def to_latex(
        self,
        buf: FilePath | WriteBuffer[str] | None = None,
        *,
        columns: Sequence[Hashable] | None = None,
        header: bool | SequenceNotStr[str] = True,
        index: bool = True,
        na_rep: str = "NaN",
        formatters: FormattersType | None = None,
        float_format: FloatFormatType | None = None,
        sparsify: bool | None = None,
        index_names: bool = True,
        bold_rows: bool = False,
        column_format: str | None = None,
        longtable: bool | None = None,
        escape: bool | None = None,
        encoding: str | None = None,
        decimal: str = ".",
        multicolumn: bool | None = None,
        multicolumn_format: str | None = None,
        multirow: bool | None = None,
        caption: str | tuple[str, str] | None = None,
        label: str | None = None,
        position: str | None = None,
    ) -> str | None:
        # [FUNCTION DOC]
        # Render the object to a LaTeX table by configuring a Styler and delegating to
        # its LaTeX export.
        # Parameters: buf is a path or text buffer (None returns a string); all
        # remaining parameters are keyword only: columns subset to output; header
        # bool, or list of aliases whose length must equal the number of output
        # columns; index whether to show row labels; na_rep text for missing values
        # (default NaN); formatters is a per-column list, a dict keyed by column (with
        # optional special keys for the index and column labels), or a callable;
        # float_format is a format string or callable applied to floats and complex
        # numbers; sparsify, multicolumn, multicolumn_format, multirow, longtable,
        # escape control layout and escaping and default (when None) to the matching
        # styler configuration options; index_names whether to show index names;
        # bold_rows bold row labels; column_format the LaTeX column spec string;
        # encoding, decimal, caption, label and position are forwarded.
        # A Series is first converted to a frame. Raises a value error if
        # column_format is not a string, or if the number of header aliases differs
        # from the number of columns written.
        # Builds argument lists for hiding columns not selected, hiding the header or
        # index or index names, relabelling columns, formatting data and labels, and
        # render options (horizontal rules on, environment longtable if requested,
        # alignment variants prefixed with a naive marker when multicolumn or multirow
        # is disabled, cline setting only for multirow with a MultiIndex). A string
        # float_format is converted into a function applying percent formatting; when
        # float_format is given, floating values bypass the per-column formatter.
        # Side effect: when formatters is a dict, that caller-supplied dict is mutated
        # in place (the special index and columns keys are popped and entries for
        # float columns lacking a formatter are added).
        # Returns the string, or None when a buffer was supplied.
        r"""
        Render object to a LaTeX tabular, longtable, or nested table.

        Requires ``\usepackage{booktabs}``.  The output can be copy/pasted
        into a main LaTeX document or read from an external file
        with ``\input{table.tex}``.

        .. versionchanged:: 2.0.0
           Refactored to use the Styler implementation via jinja2 templating.

        Parameters
        ----------
        buf : str, Path or StringIO-like, optional, default None
            Buffer to write to. If None, the output is returned as a string.
        columns : list of label, optional
            The subset of columns to write. Writes all columns by default.
        header : bool or list of str, default True
            Write out the column names. If a list of strings is given,
            it is assumed to be aliases for the column names. Braces must be escaped.
        index : bool, default True
            Write row names (index).
        na_rep : str, default 'NaN'
            Missing data representation.
        formatters : list of functions or dict of {str: function}, optional
            Formatter functions to apply to columns' elements by position or
            name. The result of each function must be a unicode string.
            List must be of length equal to the number of columns.
        float_format : one-parameter function or str, optional, default None
            Formatter for floating point numbers. For example
            ``float_format="%.2f"`` and ``float_format="{:0.2f}".format`` will
            both result in 0.1234 being formatted as 0.12.
        sparsify : bool, optional
            Set to False for a DataFrame with a hierarchical index to print
            every multiindex key at each row. By default, the value will be
            read from the config module.
        index_names : bool, default True
            Prints the names of the indexes.
        bold_rows : bool, default False
            Make the row labels bold in the output.
        column_format : str, optional
            The columns format as specified in `LaTeX table format
            <https://en.wikibooks.org/wiki/LaTeX/Tables>`__ e.g. 'rcl' for 3
            columns. By default, 'l' will be used for all columns except
            columns of numbers, which default to 'r'.
        longtable : bool, optional
            Use a longtable environment instead of tabular. Requires
            adding a \usepackage{longtable} to your LaTeX preamble.
            By default, the value will be read from the pandas config
            module, and set to `True` if the option ``styler.latex.environment`` is
            `"longtable"`.

            .. versionchanged:: 2.0.0
               The pandas option affecting this argument has changed.
        escape : bool, optional
            By default, the value will be read from the pandas config
            module and set to `True` if the option ``styler.format.escape`` is
            `"latex"`. When set to False prevents from escaping latex special
            characters in column names.

            .. versionchanged:: 2.0.0
               The pandas option affecting this argument has changed, as has the
               default value to `False`.
        encoding : str, optional
            A string representing the encoding to use in the output file,
            defaults to 'utf-8'.
        decimal : str, default '.'
            Character recognized as decimal separator, e.g. ',' in Europe.
        multicolumn : bool, default True
            Use \multicolumn to enhance MultiIndex columns.
            The default will be read from the config module, and is set
            as the option ``styler.sparse.columns``.

            .. versionchanged:: 2.0.0
               The pandas option affecting this argument has changed.
        multicolumn_format : str, default 'r'
            The alignment for multicolumns, similar to `column_format`
            The default will be read from the config module, and is set as the option
            ``styler.latex.multicol_align``.

            .. versionchanged:: 2.0.0
               The pandas option affecting this argument has changed, as has the
               default value to "r".
        multirow : bool, default True
            Use \multirow to enhance MultiIndex rows. Requires adding a
            \usepackage{multirow} to your LaTeX preamble. Will print
            centered labels (instead of top-aligned) across the contained
            rows, separating groups via clines. The default will be read
            from the pandas config module, and is set as the option
            ``styler.sparse.index``.

            .. versionchanged:: 2.0.0
               The pandas option affecting this argument has changed, as has the
               default value to `True`.
        caption : str or tuple, optional
            Tuple (full_caption, short_caption),
            which results in ``\caption[short_caption]{full_caption}``;
            if a single string is passed, no short caption will be set.
        label : str, optional
            The LaTeX label to be placed inside ``\label{}`` in the output.
            This is used with ``\ref{}`` in the main ``.tex`` file.

        position : str, optional
            The LaTeX positional argument for tables, to be placed after
            ``\begin{}`` in the output.

        Returns
        -------
        str or None
            If buf is None, returns the result as a string. Otherwise returns None.

        See Also
        --------
        io.formats.style.Styler.to_latex : Render a DataFrame to LaTeX
            with conditional formatting.
        DataFrame.to_string : Render a DataFrame to a console-friendly
            tabular output.
        DataFrame.to_html : Render a DataFrame as an HTML table.

        Notes
        -----
        As of v2.0.0 this method has changed to use the Styler implementation as
        part of :meth:`.Styler.to_latex` via ``jinja2`` templating. This means
        that ``jinja2`` is a requirement, and needs to be installed, for this method
        to function. It is advised that users switch to using Styler, since that
        implementation is more frequently updated and contains much more
        flexibility with the output.

        Examples
        --------
        Convert a general DataFrame to LaTeX with formatting:

        >>> df = pd.DataFrame(dict(name=['Raphael', 'Donatello'],
        ...                        age=[26, 45],
        ...                        height=[181.23, 177.65]))
        >>> print(df.to_latex(index=False,
        ...                   formatters={"name": str.upper},
        ...                   float_format="{:.1f}".format,
        ...                   ))  # doctest: +SKIP
        \begin{tabular}{lrr}
        \toprule
        name & age & height \\
        \midrule
        RAPHAEL & 26 & 181.2 \\
        DONATELLO & 45 & 177.7 \\
        \bottomrule
        \end{tabular}
        """
        # Get defaults from the pandas config
        if self.ndim == 1:
            self = self.to_frame()
        if longtable is None:
            longtable = config.get_option("styler.latex.environment") == "longtable"
        if escape is None:
            escape = config.get_option("styler.format.escape") == "latex"
        if multicolumn is None:
            multicolumn = config.get_option("styler.sparse.columns")
        if multicolumn_format is None:
            multicolumn_format = config.get_option("styler.latex.multicol_align")
        if multirow is None:
            multirow = config.get_option("styler.sparse.index")

        if column_format is not None and not isinstance(column_format, str):
            raise ValueError("`column_format` must be str or unicode")
        length = len(self.columns) if columns is None else len(columns)
        if isinstance(header, (list, tuple)) and len(header) != length:
            raise ValueError(f"Writing {length} cols but got {len(header)} aliases")

        # Refactor formatters/float_format/decimal/na_rep/escape to Styler structure
        base_format_ = {
            "na_rep": na_rep,
            "escape": "latex" if escape else None,
            "decimal": decimal,
        }
        index_format_: dict[str, Any] = {"axis": 0, **base_format_}
        column_format_: dict[str, Any] = {"axis": 1, **base_format_}

        if isinstance(float_format, str):
            float_format_: Callable | None = lambda x: float_format % x
        else:
            float_format_ = float_format

        def _wrap(x, alt_format_):
            # [FUNCTION DOC]
            # HELPER: wraps a per-column formatter so that floats and complex values
            # use the float format function when one was given.
            # Parameters: x is the value to format; alt_format_ is the fallback
            # formatter used for all other values (or for floats when no float format
            # was given). Returns the formatted value.
            if isinstance(x, (float, complex)) and float_format_ is not None:
                return float_format_(x)
            else:
                return alt_format_(x)

        formatters_: list | tuple | dict | Callable | None = None
        if isinstance(formatters, list):
            formatters_ = {
                c: partial(_wrap, alt_format_=formatters[i])
                for i, c in enumerate(self.columns)
            }
        elif isinstance(formatters, dict):
            index_formatter = formatters.pop("__index__", None)
            column_formatter = formatters.pop("__columns__", None)
            if index_formatter is not None:
                index_format_.update({"formatter": index_formatter})
            if column_formatter is not None:
                column_format_.update({"formatter": column_formatter})

            formatters_ = formatters
            float_columns = self.select_dtypes(include="float").columns
            for col in float_columns:
                if col not in formatters.keys():
                    formatters_.update({col: float_format_})
        elif formatters is None and float_format is not None:
            formatters_ = partial(_wrap, alt_format_=lambda v: v)
        format_index_ = [index_format_, column_format_]
        format_index_names_ = [index_format_, column_format_]

        # Deal with hiding indexes and relabelling column names
        hide_: list[dict] = []
        relabel_index_: list[dict] = []
        if columns:
            hide_.append(
                {
                    "subset": [c for c in self.columns if c not in columns],
                    "axis": "columns",
                }
            )
        if header is False:
            hide_.append({"axis": "columns"})
        elif isinstance(header, (list, tuple)):
            relabel_index_.append({"labels": header, "axis": "columns"})
            format_index_ = [index_format_]  # column_format is overwritten

        if index is False:
            hide_.append({"axis": "index"})
        if index_names is False:
            hide_.append({"names": True, "axis": "index"})

        render_kwargs_ = {
            "hrules": True,
            "sparse_index": sparsify,
            "sparse_columns": sparsify,
            "environment": "longtable" if longtable else None,
            "multicol_align": multicolumn_format
            if multicolumn
            else f"naive-{multicolumn_format}",
            "multirow_align": "t" if multirow else "naive",
            "encoding": encoding,
            "caption": caption,
            "label": label,
            "position": position,
            "column_format": column_format,
            "clines": "skip-last;data"
            if (multirow and isinstance(self.index, MultiIndex))
            else None,
            "bold_rows": bold_rows,
        }

        return self._to_latex_via_styler(
            buf,
            hide=hide_,
            relabel_index=relabel_index_,
            format={"formatter": formatters_, **base_format_},
            format_index=format_index_,
            format_index_names=format_index_names_,
            render_kwargs=render_kwargs_,
        )

    @final
    def _to_latex_via_styler(
        self,
        buf=None,
        *,
        hide: dict | list[dict] | None = None,
        relabel_index: dict | list[dict] | None = None,
        format: dict | list[dict] | None = None,
        format_index: dict | list[dict] | None = None,
        format_index_names: dict | list[dict] | None = None,
        render_kwargs: dict | None = None,
    ):
        # [FUNCTION DOC]
        # HELPER: builds a Styler for the object and renders it as LaTeX; used by the
        # LaTeX export and shared with other callers.
        # Parameters: buf is the output target; hide, relabel_index, format,
        # format_index and format_index_names are each a dict of keyword arguments or
        # a list of such dicts, applied in order to the matching Styler method (hide,
        # relabel index, format, format index, format index names); render_kwargs is a
        # dict of keyword arguments for the Styler LaTeX export and must contain the
        # bold_rows entry, which is removed from it before rendering.
        # The object is treated as a DataFrame and given an empty uuid. The Styler
        # methods to call are looked up by name and the arguments located using the
        # local variable namespace. If bold rows is requested a map over index labels
        # adds a bold text-style directive. Returns the result of the Styler LaTeX
        # export (a string, or None when writing to a buffer).
        # Raises a key error if the bold_rows entry is missing from the render
        # options.
        """
        Render object to a LaTeX tabular, longtable, or nested table.

        Uses the ``Styler`` implementation with the following, ordered, method chaining:

        .. code-block:: python
           styler = Styler(DataFrame)
           styler.hide(**hide)
           styler.relabel_index(**relabel_index)
           styler.format(**format)
           styler.format_index(**format_index)
           styler.to_latex(buf=buf, **render_kwargs)

        Parameters
        ----------
        buf : str, Path or StringIO-like, optional, default None
            Buffer to write to. If None, the output is returned as a string.
        hide : dict, list of dict
            Keyword args to pass to the method call of ``Styler.hide``. If a list will
            call the method numerous times.
        relabel_index : dict, list of dict
            Keyword args to pass to the method of ``Styler.relabel_index``. If a list
            will call the method numerous times.
        format : dict, list of dict
            Keyword args to pass to the method call of ``Styler.format``. If a list will
            call the method numerous times.
        format_index : dict, list of dict
            Keyword args to pass to the method call of ``Styler.format_index``. If a
            list will call the method numerous times.
        render_kwargs : dict
            Keyword args to pass to the method call of ``Styler.to_latex``.

        Returns
        -------
        str or None
            If buf is None, returns the result as a string. Otherwise returns None.
        """
        from pandas.io.formats.style import Styler

        self = cast("DataFrame", self)
        styler = Styler(self, uuid="")

        for kw_name in [
            "hide",
            "relabel_index",
            "format",
            "format_index",
            "format_index_names",
        ]:
            kw = vars()[kw_name]
            if isinstance(kw, dict):
                getattr(styler, kw_name)(**kw)
            elif isinstance(kw, list):
                for sub_kw in kw:
                    getattr(styler, kw_name)(**sub_kw)

        # bold_rows is not a direct kwarg of Styler.to_latex
        render_kwargs = {} if render_kwargs is None else render_kwargs
        if render_kwargs.pop("bold_rows"):
            styler.map_index(lambda v: "textbf:--rwrap;")

        return styler.to_latex(buf=buf, **render_kwargs)

    @overload
    def to_csv(
        self,
        path_or_buf: None = ...,
        *,
        sep: str = ...,
        na_rep: str = ...,
        float_format: str | Callable | None = ...,
        columns: Sequence[Hashable] | None = ...,
        header: bool | list[str] = ...,
        index: bool = ...,
        index_label: IndexLabel | None = ...,
        mode: str = ...,
        encoding: str | None = ...,
        compression: CompressionOptions = ...,
        quoting: int | None = ...,
        quotechar: str = ...,
        lineterminator: str | None = ...,
        chunksize: int | None = ...,
        date_format: str | None = ...,
        doublequote: bool = ...,
        escapechar: str | None = ...,
        decimal: str = ...,
        errors: OpenFileErrors = ...,
        storage_options: StorageOptions = ...,
    ) -> str:
        # [FUNCTION DOC]
        # Typing overload stub only (body is a placeholder); the behavior is defined
        # by the implementation that follows these overloads. to_csv variant with no
        # target; returns the CSV text as a string.
        ...

    @overload
    def to_csv(
        self,
        path_or_buf: FilePath | WriteBuffer[bytes] | WriteBuffer[str],
        *,
        sep: str = ...,
        na_rep: str = ...,
        float_format: str | Callable | None = ...,
        columns: Sequence[Hashable] | None = ...,
        header: bool | list[str] = ...,
        index: bool = ...,
        index_label: IndexLabel | None = ...,
        mode: str = ...,
        encoding: str | None = ...,
        compression: CompressionOptions = ...,
        quoting: int | None = ...,
        quotechar: str = ...,
        lineterminator: str | None = ...,
        chunksize: int | None = ...,
        date_format: str | None = ...,
        doublequote: bool = ...,
        escapechar: str | None = ...,
        decimal: str = ...,
        errors: OpenFileErrors = ...,
        storage_options: StorageOptions = ...,
    ) -> None:
        # [FUNCTION DOC]
        # Typing overload stub only (body is a placeholder); the behavior is defined
        # by the implementation that follows these overloads. to_csv variant with a
        # path or buffer; returns nothing.
        ...

    @final
    def to_csv(
        self,
        path_or_buf: FilePath | WriteBuffer[bytes] | WriteBuffer[str] | None = None,
        *,
        sep: str = ",",
        na_rep: str = "",
        float_format: str | Callable | None = None,
        columns: Sequence[Hashable] | None = None,
        header: bool | list[str] = True,
        index: bool = True,
        index_label: IndexLabel | None = None,
        mode: str = "w",
        encoding: str | None = None,
        compression: CompressionOptions = "infer",
        quoting: int | None = None,
        quotechar: str = '"',
        lineterminator: str | None = None,
        chunksize: int | None = None,
        date_format: str | None = None,
        doublequote: bool = True,
        escapechar: str | None = None,
        decimal: str = ".",
        errors: OpenFileErrors = "strict",
        storage_options: StorageOptions | None = None,
    ) -> str | None:
        # [FUNCTION DOC]
        # Write the object to a comma-separated values file or return the text.
        # Parameters: path_or_buf is a path or binary or text buffer (None returns a
        # string); keyword only: sep field delimiter; na_rep missing-value text;
        # float_format format string or callable for floats; columns subset to write;
        # header bool or list of aliases; index include the row labels; index_label
        # header label(s) for index columns; mode file mode; encoding; compression;
        # quoting csv quoting constant; quotechar; lineterminator; chunksize rows per
        # write batch; date_format; doublequote; escapechar; decimal separator; errors
        # encoding error handling; storage_options for remote targets.
        # A Series is converted to a frame first. A formatter configured with header,
        # index, missing-value text, float format and decimal is built, and a renderer
        # object does the writing with the remaining arguments. Returns the text when
        # no target was given, otherwise None. Side effect: writes a file or buffer.
        r"""
        Write object to a comma-separated values (csv) file.

        Parameters
        ----------
        path_or_buf : str, path object, file-like object, or None, default None
            String, path object (implementing os.PathLike[str]), or file-like
            object implementing a write() function. If None, the result is
            returned as a string. If a non-binary file object is passed, it should
            be opened with `newline=''`, disabling universal newlines. If a binary
            file object is passed, `mode` might need to contain a `'b'`.
        sep : str, default ','
            String of length 1. Field delimiter for the output file.
        na_rep : str, default ''
            Missing data representation.
        float_format : str, Callable, default None
            Format string for floating point numbers. If a Callable is given, it takes
            precedence over other numeric formatting parameters, like decimal.
        columns : sequence, optional
            Columns to write.
        header : bool or list of str, default True
            Write out the column names. If a list of strings is given it is
            assumed to be aliases for the column names.
        index : bool, default True
            Write row names (index).
        index_label : str or sequence, or False, default None
            Column label for index column(s) if desired. If None is given, and
            `header` and `index` are True, then the index names are used. A
            sequence should be given if the object uses MultiIndex. If
            False do not print fields for index names. Use index_label=False
            for easier importing in R.
        mode : {'w', 'x', 'a'}, default 'w'
            Forwarded to either `open(mode=)` or `fsspec.open(mode=)` to control
            the file opening. Typical values include:

            - 'w', truncate the file first.
            - 'x', exclusive creation, failing if the file already exists.
            - 'a', append to the end of file if it exists.

        encoding : str, optional
            A string representing the encoding to use in the output file,
            defaults to 'utf-8'. `encoding` is not supported if `path_or_buf`
            is a non-binary file object.

        compression : str or dict, default 'infer'
            For on-the-fly compression of the output data. If 'infer' and
            'path_or_buf' is path-like, then detect compression from the following
            extensions: '.gz',
            '.bz2', '.zip', '.xz', '.zst', '.tar', '.tar.gz', '.tar.xz' or '.tar.bz2'
            (otherwise no compression).
            Set to ``None`` for no compression.
            Can also be a dict with key ``'method'`` set to one of
            {``'zip'``, ``'gzip'``, ``'bz2'``, ``'zstd'``, ``'xz'``, ``'tar'``} and
            other key-value pairs are forwarded to
            ``zipfile.ZipFile``, ``gzip.GzipFile``,
            ``bz2.BZ2File``, ``zstandard.ZstdCompressor``, ``lzma.LZMAFile`` or
            ``tarfile.TarFile``, respectively.
            As an example, the following could be passed for faster compression and
            to create a reproducible gzip archive:
            ``compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1}``.

               May be a dict with key 'method' as compression mode
               and other entries as additional compression options if
               compression mode is 'zip'.

               Passing compression options as keys in dict is
               supported for compression modes 'gzip', 'bz2', 'zstd', and 'zip'.
        quoting : optional constant from csv module
            Defaults to csv.QUOTE_MINIMAL. If you have set a `float_format`
            then floats are converted to strings and thus csv.QUOTE_NONNUMERIC
            will treat them as non-numeric.
        quotechar : str, default '\"'
            String of length 1. Character used to quote fields.
        lineterminator : str, optional
            The newline character or character sequence to use in the output
            file. Defaults to `os.linesep`, which depends on the OS in which
            this method is called ('\\n' for linux, '\\r\\n' for Windows, i.e.).
        chunksize : int or None
            Rows to write at a time.
        date_format : str, default None
            Format string for datetime objects.
        doublequote : bool, default True
            Control quoting of `quotechar` inside a field.
        escapechar : str, default None
            String of length 1. Character used to escape `sep` and `quotechar`
            when appropriate.
        decimal : str, default '.'
            Character recognized as decimal separator. E.g. use ',' for
            European data.
        errors : str, default 'strict'
            Specifies how encoding and decoding errors are to be handled.
            See the errors argument for :func:`open` for a full list
            of options.

        storage_options : dict, optional
            Extra options that make sense for a particular storage connection, e.g.
            host, port, username, password, etc. For HTTP(S) URLs the key-value pairs
            are forwarded to ``urllib.request.Request`` as header options. For other
            URLs (e.g. starting with "s3://", and "gcs://") the key-value pairs are
            forwarded to ``fsspec.open``. Please see ``fsspec`` and ``urllib`` for more
            details, and for more examples on storage options refer `here
            <https://pandas.pydata.org/docs/user_guide/io.html?
            highlight=storage_options#reading-writing-remote-files>`_.

        Returns
        -------
        None or str
            If path_or_buf is None, returns the resulting csv format as a
            string. Otherwise returns None.

        See Also
        --------
        read_csv : Load a CSV file into a DataFrame.
        to_excel : Write DataFrame to an Excel file.

        Examples
        --------
        Create 'out.csv' containing 'df' without indices

        >>> df = pd.DataFrame(
        ...     [["Raphael", "red", "sai"], ["Donatello", "purple", "bo staff"]],
        ...     columns=["name", "mask", "weapon"],
        ... )
        >>> df.to_csv("out.csv", index=False)  # doctest: +SKIP

        Create 'out.zip' containing 'out.csv'

        >>> df.to_csv(index=False)
        'name,mask,weapon\nRaphael,red,sai\nDonatello,purple,bo staff\n'
        >>> compression_opts = dict(
        ...     method="zip", archive_name="out.csv"
        ... )  # doctest: +SKIP
        >>> df.to_csv(
        ...     "out.zip", index=False, compression=compression_opts
        ... )  # doctest: +SKIP

        To write a csv file to a new folder or nested folder you will first
        need to create it using either Pathlib or os:

        >>> from pathlib import Path  # doctest: +SKIP
        >>> filepath = Path("folder/subfolder/out.csv")  # doctest: +SKIP
        >>> filepath.parent.mkdir(parents=True, exist_ok=True)  # doctest: +SKIP
        >>> df.to_csv(filepath)  # doctest: +SKIP

        >>> import os  # doctest: +SKIP
        >>> os.makedirs("folder/subfolder", exist_ok=True)  # doctest: +SKIP
        >>> df.to_csv("folder/subfolder/out.csv")  # doctest: +SKIP

        Format floats to two decimal places:

        >>> df.to_csv("out1.csv", float_format="%.2f")  # doctest: +SKIP

        Format floats using scientific notation:

        >>> df.to_csv("out2.csv", float_format="{:.2e}".format)  # doctest: +SKIP
        """
        df = self if isinstance(self, ABCDataFrame) else self.to_frame()

        formatter = DataFrameFormatter(
            frame=df,
            header=header,
            index=index,
            na_rep=na_rep,
            float_format=float_format,
            decimal=decimal,
        )

        return DataFrameRenderer(formatter).to_csv(
            path_or_buf,
            lineterminator=lineterminator,
            sep=sep,
            encoding=encoding,
            errors=errors,
            compression=compression,
            quoting=quoting,
            columns=columns,
            index_label=index_label,
            mode=mode,
            chunksize=chunksize,
            quotechar=quotechar,
            date_format=date_format,
            doublequote=doublequote,
            escapechar=escapechar,
            storage_options=storage_options,
        )

    # ----------------------------------------------------------------------
    # Indexing Methods

    @final
    def take(self, indices, axis: Axis = 0, **kwargs) -> Self:
        # [FUNCTION DOC]
        # Select elements by integer position along an axis.
        # Parameters: indices is an array-like of integer positions (negative values
        # count from the end); axis is the axis identifier (default 0); extra keyword
        # arguments exist only for numpy compatibility and are validated to be empty
        # or numpy-compatible defaults.
        # Raises a type error if indices is a slice. Positions are converted to a
        # platform integer array. When the axis is zero and the positions form the
        # identity range covering the whole length, a shallow copy is returned without
        # copying data. Otherwise the manager performs the take with bounds
        # verification (raising an index error if out of bounds) and the result is
        # finalized with the method name take. Returns a new object of the same kind.
        """
        Return the elements in the given *positional* indices along an axis.

        This means that we are not indexing according to actual values in
        the index attribute of the object. We are indexing according to the
        actual position of the element in the object.

        Parameters
        ----------
        indices : array-like
            An array of ints indicating which positions to take.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The axis on which to select elements. ``0`` means that we are
            selecting rows, ``1`` means that we are selecting columns.
            For `Series` this parameter is unused and defaults to 0.
        **kwargs
            For compatibility with :meth:`numpy.take`. Has no effect on the
            output.

        Returns
        -------
        same type as caller
            An array-like containing the elements taken from the object.

        See Also
        --------
        DataFrame.loc : Select a subset of a DataFrame by labels.
        DataFrame.iloc : Select a subset of a DataFrame by positions.
        numpy.take : Take elements from an array along an axis.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     [
        ...         ("falcon", "bird", 389.0),
        ...         ("parrot", "bird", 24.0),
        ...         ("lion", "mammal", 80.5),
        ...         ("monkey", "mammal", np.nan),
        ...     ],
        ...     columns=["name", "class", "max_speed"],
        ...     index=[0, 2, 3, 1],
        ... )
        >>> df
             name   class  max_speed
        0  falcon    bird      389.0
        2  parrot    bird       24.0
        3    lion  mammal       80.5
        1  monkey  mammal        NaN

        Take elements at positions 0 and 3 along the axis 0 (default).

        Note how the actual indices selected (0 and 1) do not correspond to
        our selected indices 0 and 3. That's because we are selecting the 0th
        and 3rd rows, not rows whose indices equal 0 and 3.

        >>> df.take([0, 3])
             name   class  max_speed
        0  falcon    bird      389.0
        1  monkey  mammal        NaN

        Take elements at indices 1 and 2 along the axis 1 (column selection).

        >>> df.take([1, 2], axis=1)
            class  max_speed
        0    bird      389.0
        2    bird       24.0
        3  mammal       80.5
        1  mammal        NaN

        We may take elements using negative integers for positive indices,
        starting from the end of the object, just like with Python lists.

        >>> df.take([-1, -2])
             name   class  max_speed
        1  monkey  mammal        NaN
        3    lion  mammal       80.5
        """

        nv.validate_take((), kwargs)

        if isinstance(indices, slice):
            raise TypeError(
                f"{type(self).__name__}.take requires a sequence of integers, "
                "not slice."
            )
        indices = np.asarray(indices, dtype=np.intp)
        if axis == 0 and indices.ndim == 1 and is_range_indexer(indices, len(self)):
            return self.copy(deep=False)

        new_data = self._mgr.take(
            indices,
            axis=self._get_block_manager_axis(axis),
            verify=True,
        )
        return self._constructor_from_mgr(new_data, axes=new_data.axes).__finalize__(
            self, method="take"
        )

    @final
    def xs(
        self,
        key: IndexLabel,
        axis: Axis = 0,
        level: IndexLabel | None = None,
        drop_level: bool = True,
    ) -> Self:
        # [FUNCTION DOC]
        # Return a cross-section of the object by label.
        # Parameters: key is a label or tuple of labels (a list is rejected); axis is
        # the axis to take from (default 0); level (for MultiIndex) names the level or
        # levels the key applies to; drop_level (default true) says whether the
        # matched levels are removed from the result.
        # Raises a type error for a list key, and a type error when level is given but
        # the axis is not a MultiIndex.
        # With level: the matching positions and the new axis are computed by the
        # MultiIndex level lookup, the rows or columns are selected by position and
        # the new axis is installed on the result.
        # Without level on columns with drop_level, the item getter is used. Otherwise
        # on a MultiIndex axis the first-level lookup is used (and, when not dropping,
        # the matching entries are kept as the new index), and on a flat axis the
        # label location is found. A location returned as an array (boolean mask or
        # positions) is routed to the take method. A scalar location on axis 0 yields
        # the underlying element for one-dimensional objects, or the row as a Series
        # named after the label (via the manager fast cross-section, with finalize)
        # for frames; a scalar location on axis 1 yields a one-column slice; other
        # non-scalar locations select by position, installing the new index for rows.
        # Returns a scalar, Series or DataFrame accordingly.
        """
        Return cross-section from the Series/DataFrame.

        This method takes a `key` argument to select data at a particular
        level of a MultiIndex.

        Parameters
        ----------
        key : label or tuple of label
            Label contained in the index, or partially in a MultiIndex.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Axis to retrieve cross-section on.
        level : object, defaults to first n levels (n=1 or len(key))
            In case of a key partially contained in a MultiIndex, indicate
            which levels are used. Levels can be referred by label or position.
        drop_level : bool, default True
            If False, returns object with same levels as self.

        Returns
        -------
        Series or DataFrame
            Cross-section from the original Series or DataFrame
            corresponding to the selected index levels.

        See Also
        --------
        DataFrame.loc : Access a group of rows and columns
            by label(s) or a boolean array.
        DataFrame.iloc : Purely integer-location based indexing
            for selection by position.

        Notes
        -----
        `xs` can not be used to set values.

        MultiIndex Slicers is a generic way to get/set values on
        any level or levels.
        It is a superset of `xs` functionality, see
        :ref:`MultiIndex Slicers <advanced.mi_slicers>`.

        Examples
        --------
        >>> d = {
        ...     "num_legs": [4, 4, 2, 2],
        ...     "num_wings": [0, 0, 2, 2],
        ...     "class": ["mammal", "mammal", "mammal", "bird"],
        ...     "animal": ["cat", "dog", "bat", "penguin"],
        ...     "locomotion": ["walks", "walks", "flies", "walks"],
        ... }
        >>> df = pd.DataFrame(data=d)
        >>> df = df.set_index(["class", "animal", "locomotion"])
        >>> df
                                   num_legs  num_wings
        class  animal  locomotion
        mammal cat     walks              4          0
               dog     walks              4          0
               bat     flies              2          2
        bird   penguin walks              2          2

        Get values at specified index

        >>> df.xs("mammal")
                           num_legs  num_wings
        animal locomotion
        cat    walks              4          0
        dog    walks              4          0
        bat    flies              2          2

        Get values at several indexes

        >>> df.xs(("mammal", "dog", "walks"))
        num_legs     4
        num_wings    0
        Name: (mammal, dog, walks), dtype: int64

        Get values at specified index and level

        >>> df.xs("cat", level=1)
                           num_legs  num_wings
        class  locomotion
        mammal walks              4          0

        Get values at several indexes and levels

        >>> df.xs(("bird", "walks"), level=[0, "locomotion"])
                 num_legs  num_wings
        animal
        penguin         2          2

        Get values at specified column and axis

        >>> df.xs("num_wings", axis=1)
        class   animal   locomotion
        mammal  cat      walks         0
                dog      walks         0
                bat      flies         2
        bird    penguin  walks         2
        Name: num_wings, dtype: int64
        """
        axis = self._get_axis_number(axis)
        labels = self._get_axis(axis)

        if isinstance(key, list):
            raise TypeError("list keys are not supported in xs, pass a tuple instead")

        if level is not None:
            if not isinstance(labels, MultiIndex):
                raise TypeError("Index must be a MultiIndex")
            loc, new_ax = labels.get_loc_level(key, level=level, drop_level=drop_level)

            # create the tuple of the indexer
            _indexer = [slice(None)] * self.ndim
            _indexer[axis] = loc
            indexer = tuple(_indexer)

            result = self.iloc[indexer]
            setattr(result, result._get_axis_name(axis), new_ax)
            return result

        if axis == 1:
            if drop_level:
                return self[key]
            index = self.columns
        else:
            index = self.index

        if isinstance(index, MultiIndex):
            loc, new_index = index._get_loc_level(key, level=0)
            if not drop_level:
                if lib.is_integer(loc):
                    # Slice index must be an integer or None
                    new_index = index[loc : loc + 1]
                else:
                    new_index = index[loc]
        else:
            loc = index.get_loc(key)

            if isinstance(loc, np.ndarray):
                if loc.dtype == np.bool_:
                    (inds,) = loc.nonzero()
                    return self.take(inds, axis=axis)
                else:
                    return self.take(loc, axis=axis)

            if not is_scalar(loc):
                new_index = index[loc]

        if is_scalar(loc) and axis == 0:
            # In this case loc should be an integer
            if self.ndim == 1:
                # if we encounter an array-like and we only have 1 dim
                # that means that their are list/ndarrays inside the Series!
                # so just return them (GH 6394)
                return self._values[loc]

            new_mgr = self._mgr.fast_xs(loc)

            result = self._constructor_sliced_from_mgr(new_mgr, axes=new_mgr.axes)
            result._name = self.index[loc]
            result = result.__finalize__(self)
        elif is_scalar(loc):
            result = self.iloc[:, slice(loc, loc + 1)]
        elif axis == 1:
            result = self.iloc[:, loc]
        else:
            result = self.iloc[loc]
            result.index = new_index

        return result

    def __getitem__(self, item):
        # [FUNCTION DOC]
        # Abstract item getter; always raises an abstract-method error. Series and
        # DataFrame provide the real implementation.
        raise AbstractMethodError(self)

    @final
    def _getitem_slice(self, key: slice) -> Self:
        # [FUNCTION DOC]
        # HELPER: item lookup for a slice key.
        # Parameters: key is a slice object of labels or positions.
        # The index decides whether the slice is positional or label based and
        # converts it to a positional slice or, for some index types (such as
        # datetime), a positional array. An array is condensed to a slice when
        # possible, otherwise take is used. Returns the sliced object.
        """
        __getitem__ for the case where the key is a slice object.
        """
        # _convert_slice_indexer to determine if this slice is positional
        #  or label based, and if the latter, convert to positional
        slobj = self.index._convert_slice_indexer(key, kind="getitem")
        if isinstance(slobj, np.ndarray):
            # reachable with DatetimeIndex
            indexer = lib.maybe_indices_to_slice(slobj.astype(np.intp), len(self))
            if isinstance(indexer, np.ndarray):
                # GH#43223 If we can not convert, use take
                return self.take(indexer, axis=0)
            slobj = indexer
        return self._slice(slobj)

    def _slice(self, slobj: slice, axis: AxisInt = 0) -> Self:
        # [FUNCTION DOC]
        # HELPER: constructs a positional slice of this object along an axis.
        # Parameters: slobj is a slice (asserted); axis is the user-facing axis number
        # (default 0), translated to the manager axis.
        # Takes a slice of the manager (a view under copy-on-write), builds an object
        # of the same kind and finalizes it. Returns the new object.
        """
        Construct a slice of this container.

        Slicing with this method is *always* positional.
        """
        assert isinstance(slobj, slice), type(slobj)
        axis = self._get_block_manager_axis(axis)
        new_mgr = self._mgr.get_slice(slobj, axis=axis)
        result = self._constructor_from_mgr(new_mgr, axes=new_mgr.axes)
        result = result.__finalize__(self)
        return result

    @final
    def __delitem__(self, key) -> None:
        # [FUNCTION DOC]
        # Delete an item from the info axis in place.
        # Parameters: key is a column label (or, for a frame with MultiIndex columns,
        # possibly a partial key).
        # For a frame with MultiIndex columns where the key is not itself a full key
        # of the column engine, the key is treated as a prefix: every column tuple
        # that starts with the key is deleted recursively. If nothing was deleted by
        # that path, or the shortcut did not apply, the location of the key on the
        # last axis is looked up (raising a key error if absent) and that position is
        # deleted from the manager. Mutates self.
        """
        Delete item
        """
        deleted = False

        maybe_shortcut = False
        if self.ndim == 2 and isinstance(self.columns, MultiIndex):
            try:
                # By using engine's __contains__ we effectively
                # restrict to same-length tuples
                maybe_shortcut = key not in self.columns._engine
            except TypeError:
                pass

        if maybe_shortcut:
            # Allow shorthand to delete all columns whose first len(key)
            # elements match key:
            if not isinstance(key, tuple):
                key = (key,)
            for col in self.columns:
                if isinstance(col, tuple) and col[: len(key)] == key:
                    del self[col]
                    deleted = True
        if not deleted:
            # If the above loop ran and didn't delete anything because
            # there was no match, this call should raise the appropriate
            # exception:
            loc = self.axes[-1].get_loc(key)
            self._mgr = self._mgr.idelete(loc)

    # ----------------------------------------------------------------------
    # Unsorted

    @final
    def _check_inplace_and_allows_duplicate_labels(self, inplace: bool) -> None:
        # [FUNCTION DOC]
        # HELPER: guard used by in-place operations.
        # Parameters: inplace is the caller's in-place flag. Raises a value error if
        # inplace is true and the flags forbid duplicate labels; otherwise returns
        # None.
        if inplace and not self.flags.allows_duplicate_labels:
            raise ValueError(
                "Cannot specify 'inplace=True' when "
                "'self.flags.allows_duplicate_labels' is False."
            )

    @final
    def get(self, key, default=None):
        # [FUNCTION DOC]
        # Get an item by key, returning a default if it cannot be retrieved.
        # Parameters: key is a label; default (None) is returned on failure.
        # Calls the item getter and returns its result, swallowing key errors, value
        # errors and index errors, in which case the default is returned.
        """
        Get item from object for given key (ex: DataFrame column).

        Returns ``default`` value if not found.

        Parameters
        ----------
        key : object
            Key for which item should be returned.
        default : object, default None
            Default value to return if key is not found.

        Returns
        -------
        same type as items contained in object
            Item for given key or ``default`` value, if key is not found.

        See Also
        --------
        DataFrame.get : Get item from object for given key (ex: DataFrame column).
        Series.get : Get item from object for given key (ex: DataFrame column).

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     [
        ...         [24.3, 75.7, "high"],
        ...         [31, 87.8, "high"],
        ...         [22, 71.6, "medium"],
        ...         [35, 95, "medium"],
        ...     ],
        ...     columns=["temp_celsius", "temp_fahrenheit", "windspeed"],
        ...     index=pd.date_range(start="2014-02-12", end="2014-02-15", freq="D"),
        ... )

        >>> df
                    temp_celsius  temp_fahrenheit windspeed
        2014-02-12          24.3             75.7      high
        2014-02-13          31.0             87.8      high
        2014-02-14          22.0             71.6    medium
        2014-02-15          35.0             95.0    medium

        >>> df.get(["temp_celsius", "windspeed"])
                    temp_celsius windspeed
        2014-02-12          24.3      high
        2014-02-13          31.0      high
        2014-02-14          22.0    medium
        2014-02-15          35.0    medium

        >>> ser = df["windspeed"]
        >>> ser.get("2014-02-13")
        'high'

        If the key isn't found, the default value will be used.

        >>> df.get(["temp_celsius", "temp_kelvin"], default="default_value")
        'default_value'

        >>> ser.get("2014-02-10", "[unknown]")
        '[unknown]'
        """
        try:
            return self[key]
        except (KeyError, ValueError, IndexError):
            return default

    @staticmethod
    def _check_copy_deprecation(copy):
        # [FUNCTION DOC]
        # HELPER: static method that implements the deprecation of the copy keyword.
        # Parameters: copy is the value the caller received. If it is anything other
        # than the no-default sentinel a deprecation warning (future-version class of
        # pandas) is emitted, attributed via the stack-level finder to the user's
        # call. Returns None.
        if copy is not lib.no_default:
            warnings.warn(
                "The copy keyword is deprecated and will be removed in a future "
                "version. Copy-on-Write is active in pandas since 3.0 which utilizes "
                "a lazy copy mechanism that defers copies until necessary. Use "
                ".copy() to make an eager copy if necessary.",
                Pandas4Warning,
                stacklevel=find_stack_level(),
            )

    # issue 58667
    @deprecate_kwarg(Pandas4Warning, "method", new_arg_name=None)
    @final
    def reindex_like(
        self,
        other,
        method: Literal["backfill", "bfill", "pad", "ffill", "nearest"] | None = None,
        copy: bool | lib.NoDefault = lib.no_default,
        limit: int | None = None,
        tolerance=None,
    ) -> Self:
        # [FUNCTION DOC]
        # Return an object conformed to the axes of another object.
        # Parameters: other is the object whose axes are used; method (deprecated,
        # through the decorator, with no replacement) is the fill method; copy is a
        # deprecated placeholder; limit is the maximum number of fills; tolerance is
        # the maximum label distance for inexact matches.
        # Collects the axes of other that match this object's axis order, together
        # with the method, limit and tolerance, and calls reindex with them. Returns
        # the reindexed object.
        """
        Return an object with matching indices as other object.

        Conform the object to the same index on all axes. Optional
        filling logic, placing NaN in locations having no value
        in the previous index. A new object is produced unless the
        new index is equivalent to the current one and copy=False.

        Parameters
        ----------
        other : Object of the same data type
            Its row and column indices are used to define the new indices
            of this object.
        method : {None, 'backfill'/'bfill', 'pad'/'ffill', 'nearest'}
            Method to use for filling holes in reindexed DataFrame.
            Please note: this is only applicable to DataFrames/Series with a
            monotonically increasing/decreasing index.

            .. deprecated:: 3.0.0

            * None (default): don't fill gaps
            * pad / ffill: propagate last valid observation forward to next
              valid
            * backfill / bfill: use next valid observation to fill gap
            * nearest: use nearest valid observations to fill gap.

        copy : bool, default False
            This keyword is now ignored; changing its value will have no
            impact on the method.

            .. deprecated:: 3.0.0

                This keyword is ignored and will be removed in pandas 4.0. Since
                pandas 3.0, this method always returns a new object using a lazy
                copy mechanism that defers copies until necessary
                (Copy-on-Write). See the `user guide on Copy-on-Write
                <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                for more details.

        limit : int, default None
            Maximum number of consecutive labels to fill for inexact matches.
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.

            Tolerance may be a scalar value, which applies the same tolerance
            to all values, or list-like, which applies variable tolerance per
            element. List-like includes list, tuple, array, Series, and must be
            the same size as the index and its dtype must exactly match the
            index's type.

        Returns
        -------
        Series or DataFrame
            Same type as caller, but with changed indices on each axis.

        See Also
        --------
        DataFrame.set_index : Set row labels.
        DataFrame.reset_index : Remove row labels or move them to new columns.
        DataFrame.reindex : Change to new indices or expand indices.

        Notes
        -----
        Same as calling
        ``.reindex(index=other.index, columns=other.columns,...)``.

        Examples
        --------
        >>> df1 = pd.DataFrame(
        ...     [
        ...         [24.3, 75.7, "high"],
        ...         [31, 87.8, "high"],
        ...         [22, 71.6, "medium"],
        ...         [35, 95, "medium"],
        ...     ],
        ...     columns=["temp_celsius", "temp_fahrenheit", "windspeed"],
        ...     index=pd.date_range(start="2014-02-12", end="2014-02-15", freq="D"),
        ... )

        >>> df1
                    temp_celsius  temp_fahrenheit windspeed
        2014-02-12          24.3             75.7      high
        2014-02-13          31.0             87.8      high
        2014-02-14          22.0             71.6    medium
        2014-02-15          35.0             95.0    medium

        >>> df2 = pd.DataFrame(
        ...     [[28, "low"], [30, "low"], [35.1, "medium"]],
        ...     columns=["temp_celsius", "windspeed"],
        ...     index=pd.DatetimeIndex(["2014-02-12", "2014-02-13", "2014-02-15"]),
        ... )

        >>> df2
                    temp_celsius windspeed
        2014-02-12          28.0       low
        2014-02-13          30.0       low
        2014-02-15          35.1    medium

        >>> df2.reindex_like(df1)
                    temp_celsius  temp_fahrenheit windspeed
        2014-02-12          28.0              NaN       low
        2014-02-13          30.0              NaN       low
        2014-02-14           NaN              NaN       NaN
        2014-02-15          35.1              NaN    medium
        """
        self._check_copy_deprecation(copy)
        d = other._construct_axes_dict(
            axes=self._AXIS_ORDERS,
            method=method,
            limit=limit,
            tolerance=tolerance,
        )

        return self.reindex(**d)

    @overload
    def drop(
        self,
        labels: IndexLabel | ListLike = ...,
        *,
        axis: Axis = ...,
        index: IndexLabel | ListLike = ...,
        columns: IndexLabel | ListLike = ...,
        level: Level | None = ...,
        inplace: Literal[True],
        errors: IgnoreRaise = ...,
    ) -> None:
        # [FUNCTION DOC]
        # Typing overload stub only (body is a placeholder); the behavior is defined
        # by the implementation that follows these overloads. drop variant for inplace
        # true; returns nothing.
        ...

    @overload
    def drop(
        self,
        labels: IndexLabel | ListLike = ...,
        *,
        axis: Axis = ...,
        index: IndexLabel | ListLike = ...,
        columns: IndexLabel | ListLike = ...,
        level: Level | None = ...,
        inplace: Literal[False] = ...,
        errors: IgnoreRaise = ...,
    ) -> Self:
        # [FUNCTION DOC]
        # Typing overload stub only (body is a placeholder); the behavior is defined
        # by the implementation that follows these overloads. drop variant for inplace
        # false; returns an object.
        ...

    @overload
    def drop(
        self,
        labels: IndexLabel | ListLike = ...,
        *,
        axis: Axis = ...,
        index: IndexLabel | ListLike = ...,
        columns: IndexLabel | ListLike = ...,
        level: Level | None = ...,
        inplace: bool = ...,
        errors: IgnoreRaise = ...,
    ) -> Self | None:
        # [FUNCTION DOC]
        # Typing overload stub only (body is a placeholder); the behavior is defined
        # by the implementation that follows these overloads. drop variant for general
        # boolean inplace.
        ...

    def drop(
        self,
        labels: IndexLabel | ListLike = None,
        *,
        axis: Axis = 0,
        index: IndexLabel | ListLike = None,
        columns: IndexLabel | ListLike = None,
        level: Level | None = None,
        inplace: bool = False,
        errors: IgnoreRaise = "raise",
    ) -> Self | None:
        # [FUNCTION DOC]
        # Remove labels from one or both axes (no docstring here; subclasses document
        # it).
        # Parameters (keyword only except labels): labels single label or list to drop
        # along axis; axis (default 0); index and columns are alternatives to specify
        # labels per axis; level is the MultiIndex level to match; inplace modifies
        # self; errors is raise (default, missing labels raise) or ignore.
        # Raises a value error if labels is combined with index or columns, if axis
        # equal to one is combined with index or columns, or if none of labels, index,
        # columns is given. Only the index axis is considered for one-dimensional
        # objects.
        # Each requested axis is processed in turn with the private drop helper. When
        # inplace, the result's internals replace self and None is returned; otherwise
        # the new object is returned.
        # Validates the inplace flag as a boolean.
        inplace = validate_bool_kwarg(inplace, "inplace")

        if labels is not None:
            if index is not None or columns is not None:
                raise ValueError("Cannot specify both 'labels' and 'index'/'columns'")
            axis_name = self._get_axis_name(axis)
            axes = {axis_name: labels}
        elif index is not None or columns is not None:
            if axis == 1:
                raise ValueError("Cannot specify both 'axis' and 'index'/'columns'")
            axes = {"index": index}
            if self.ndim == 2:
                axes["columns"] = columns
        else:
            raise ValueError(
                "Need to specify at least one of 'labels', 'index' or 'columns'"
            )

        obj = self

        for axis, labels in axes.items():
            if labels is not None:
                obj = obj._drop_axis(labels, axis, level=level, errors=errors)

        if inplace:
            self._update_inplace(obj)
            return None
        else:
            return obj

    @final
    def _drop_axis(
        self,
        labels,
        axis,
        level=None,
        errors: IgnoreRaise = "raise",
        only_slice: bool = False,
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: drops labels from a single axis; used by drop and other internals.
        # Parameters: labels to remove; axis identifier; level (for MultiIndex axes);
        # errors raise or ignore; only_slice says the manager may return a view when
        # the result is a slice.
        # For a unique axis the index drop operation computes the new axis and indexer
        # (a non-MultiIndex axis with a level raises an assertion error). For a
        # non-unique axis a boolean mask is computed manually: with a level, rows
        # whose level value is in labels are removed (raising a key error if errors is
        # raise and nothing matches); with a MultiIndex and non-tuple object labels
        # the first level is matched; otherwise membership on the axis is used and a
        # key error is raised if any label is missing and errors is raise. Extension
        # boolean masks are converted to numpy. The manager is reindexed with the
        # indexer allowing duplicates, the name is restored for a Series, and the
        # result is finalized from self. Returns the new object.
        """
        Drop labels from specified axis. Used in the ``drop`` method
        internally.

        Parameters
        ----------
        labels : single label or list-like
        axis : int or axis name
        level : int or level name, default None
            For MultiIndex
        errors : {'ignore', 'raise'}, default 'raise'
            If 'ignore', suppress error and existing labels are dropped.
        only_slice : bool, default False
            Whether indexing along columns should be view-only.

        """
        axis_num = self._get_axis_number(axis)
        axis = self._get_axis(axis)

        if axis.is_unique:
            if level is not None:
                if not isinstance(axis, MultiIndex):
                    raise AssertionError("axis must be a MultiIndex")
                new_axis = axis.drop(labels, level=level, errors=errors)
            else:
                new_axis = axis.drop(labels, errors=errors)
            indexer = axis.get_indexer(new_axis)

        # Case for non-unique axis
        else:
            is_tuple_labels = is_nested_list_like(labels) or isinstance(labels, tuple)
            labels = ensure_object(common.index_labels_to_array(labels))
            if level is not None:
                if not isinstance(axis, MultiIndex):
                    raise AssertionError("axis must be a MultiIndex")
                mask = ~axis.get_level_values(level).isin(labels)

                # GH 18561 MultiIndex.drop should raise if label is absent
                if errors == "raise" and mask.all():
                    raise KeyError(f"{labels} not found in axis")
            elif (
                isinstance(axis, MultiIndex)
                and labels.dtype == "object"
                and not is_tuple_labels
            ):
                # Set level to zero in case of MultiIndex and label is string,
                #  because isin can't handle strings for MultiIndexes GH#36293
                # In case of tuples we get dtype object but have to use isin GH#42771
                mask = ~axis.get_level_values(0).isin(labels)
            else:
                mask = ~axis.isin(labels)
                # Check if label doesn't exist along axis
                labels_missing = (axis.get_indexer_for(labels) == -1).any()
                if errors == "raise" and labels_missing:
                    raise KeyError(f"{labels} not found in axis")

            if isinstance(mask.dtype, ExtensionDtype):
                # GH#45860
                mask = mask.to_numpy(dtype=bool)

            indexer = mask.nonzero()[0]
            new_axis = axis.take(indexer)

        bm_axis = self.ndim - axis_num - 1
        new_mgr = self._mgr.reindex_indexer(
            new_axis,
            indexer,
            axis=bm_axis,
            allow_dups=True,
            only_slice=only_slice,
        )
        result = self._constructor_from_mgr(new_mgr, axes=new_mgr.axes)
        if self.ndim == 1:
            result._name = self.name

        return result.__finalize__(self)

    @final
    def _update_inplace(self, result) -> None:
        # [FUNCTION DOC]
        # HELPER: replaces the internal manager of self with the manager of result,
        # used by in-place variants. Deliberately does not propagate metadata through
        # the finalize hook. Mutates self; returns None.
        """
        Replace self internals with result.

        Parameters
        ----------
        result : same type as self
        """
        # NOTE: This does *not* call __finalize__ and that's an explicit
        # decision that we may revisit in the future.
        self._mgr = result._mgr

    @final
    def add_prefix(self, prefix: str, axis: Axis | None = None) -> Self:
        # [FUNCTION DOC]
        # Prefix labels of an axis with a string.
        # Parameters: prefix is the text to prepend (applied to the string form of
        # every label); axis selects the axis (default None means the info axis).
        # Builds a function formatting each label with the prefix and applies it
        # through the private rename on the selected axis. Returns a new object; self
        # is unchanged.
        """
        Prefix labels with string `prefix`.

        For Series, the row labels are prefixed.
        For DataFrame, the column labels are prefixed.

        Parameters
        ----------
        prefix : str
            The string to add before each label.
        axis : {0 or 'index', 1 or 'columns', None}, default None
            Axis to add prefix on

            .. versionadded:: 2.0.0

        Returns
        -------
        Series or DataFrame
            New Series or DataFrame with updated labels.

        See Also
        --------
        Series.add_suffix: Suffix row labels with string `suffix`.
        DataFrame.add_suffix: Suffix column labels with string `suffix`.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4])
        >>> s
        0    1
        1    2
        2    3
        3    4
        dtype: int64

        >>> s.add_prefix("item_")
        item_0    1
        item_1    2
        item_2    3
        item_3    4
        dtype: int64

        >>> df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [3, 4, 5, 6]})
        >>> df
           A  B
        0  1  3
        1  2  4
        2  3  5
        3  4  6

        >>> df.add_prefix("col_")
             col_A  col_B
        0       1       3
        1       2       4
        2       3       5
        3       4       6
        """
        f = lambda x: f"{prefix}{x}"

        axis_name = self._info_axis_name
        if axis is not None:
            axis_name = self._get_axis_name(axis)

        mapper = {axis_name: f}

        # error: Keywords must be strings
        # error: No overload variant of "_rename" of "NDFrame" matches
        # argument type "dict[Literal['index', 'columns'], Callable[[Any], str]]"
        return self._rename(**mapper)  # type: ignore[call-overload, misc]

    @final
    def add_suffix(self, suffix: str, axis: Axis | None = None) -> Self:
        # [FUNCTION DOC]
        # Suffix labels of an axis with a string.
        # Parameters: suffix is the text to append (applied to the string form of
        # every label); axis selects the axis (default None means the info axis).
        # Builds a function formatting each label with the suffix and applies it
        # through the private rename on the selected axis. Returns a new object; self
        # is unchanged.
        """
        Suffix labels with string `suffix`.

        For Series, the row labels are suffixed.
        For DataFrame, the column labels are suffixed.

        Parameters
        ----------
        suffix : str
            The string to add after each label.
        axis : {0 or 'index', 1 or 'columns', None}, default None
            Axis to add suffix on

            .. versionadded:: 2.0.0

        Returns
        -------
        Series or DataFrame
            New Series or DataFrame with updated labels.

        See Also
        --------
        Series.add_prefix: Prefix row labels with string `prefix`.
        DataFrame.add_prefix: Prefix column labels with string `prefix`.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4])
        >>> s
        0    1
        1    2
        2    3
        3    4
        dtype: int64

        >>> s.add_suffix("_item")
        0_item    1
        1_item    2
        2_item    3
        3_item    4
        dtype: int64

        >>> df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [3, 4, 5, 6]})
        >>> df
           A  B
        0  1  3
        1  2  4
        2  3  5
        3  4  6

        >>> df.add_suffix("_col")
             A_col  B_col
        0       1       3
        1       2       4
        2       3       5
        3       4       6
        """
        f = lambda x: f"{x}{suffix}"

        axis_name = self._info_axis_name
        if axis is not None:
            axis_name = self._get_axis_name(axis)

        mapper = {axis_name: f}
        # error: Keywords must be strings
        # error: No overload variant of "_rename" of "NDFrame" matches argument
        # type "dict[Literal['index', 'columns'], Callable[[Any], str]]"
        return self._rename(**mapper)  # type: ignore[call-overload, misc]

    @overload
    def sort_values(
        self,
        *,
        axis: Axis = ...,
        ascending: bool | Sequence[bool] = ...,
        inplace: Literal[False] = ...,
        kind: SortKind = ...,
        na_position: NaPosition = ...,
        ignore_index: bool = ...,
        key: ValueKeyFunc = ...,
    ) -> Self:
        # [FUNCTION DOC]
        # Typing overload stub only (body is a placeholder); the behavior is defined
        # by the implementation that follows these overloads. sort_values variant for
        # inplace false.
        ...

    @overload
    def sort_values(
        self,
        *,
        axis: Axis = ...,
        ascending: bool | Sequence[bool] = ...,
        inplace: Literal[True],
        kind: SortKind = ...,
        na_position: NaPosition = ...,
        ignore_index: bool = ...,
        key: ValueKeyFunc = ...,
    ) -> None:
        # [FUNCTION DOC]
        # Typing overload stub only (body is a placeholder); the behavior is defined
        # by the implementation that follows these overloads. sort_values variant for
        # inplace true.
        ...

    @overload
    def sort_values(
        self,
        *,
        axis: Axis = ...,
        ascending: bool | Sequence[bool] = ...,
        inplace: bool = ...,
        kind: SortKind = ...,
        na_position: NaPosition = ...,
        ignore_index: bool = ...,
        key: ValueKeyFunc = ...,
    ) -> Self | None:
        # [FUNCTION DOC]
        # Typing overload stub only (body is a placeholder); the behavior is defined
        # by the implementation that follows these overloads. sort_values variant for
        # general boolean inplace.
        ...

    def sort_values(
        self,
        *,
        axis: Axis = 0,
        ascending: bool | Sequence[bool] = True,
        inplace: bool = False,
        kind: SortKind = "quicksort",
        na_position: NaPosition = "last",
        ignore_index: bool = False,
        key: ValueKeyFunc | None = None,
    ) -> Self | None:
        # [FUNCTION DOC]
        # Abstract declaration of sorting by values; always raises an abstract-method
        # error. The parameters (axis, ascending, inplace, kind, na_position,
        # ignore_index, key) are documented for the concrete implementations in Series
        # and DataFrame, which override this method.
        """
        Sort by the values along either axis.

        Parameters
        ----------%(optional_by)s
        axis : %(axes_single_arg)s, default 0
             Axis to be sorted.
        ascending : bool or list of bool, default True
             Sort ascending vs. descending. Specify list for multiple sort
             orders.  If this is a list of bools, must match the length of
             the by.
        inplace : bool, default False
             If True, perform operation in-place.
        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, default 'quicksort'
             Choice of sorting algorithm. See also :func:`numpy.sort` for more
             information. `mergesort` and `stable` are the only stable algorithms. For
             DataFrames, this option is only applied when sorting on a single
             column or label.
        na_position : {'first', 'last'}, default 'last'
             Puts NaNs at the beginning if `first`; `last` puts NaNs at the
             end.
        ignore_index : bool, default False
             If True, the resulting axis will be labeled 0, 1, …, n - 1.
        key : callable, optional
            Apply the key function to the values
            before sorting. This is similar to the `key` argument in the
            builtin :meth:`sorted` function, with the notable difference that
            this `key` function should be *vectorized*. It should expect a
            ``Series`` and return a Series with the same shape as the input.
            It will be applied to each column in `by` independently. The values in the
            returned Series will be used as the keys for sorting.

        Returns
        -------
        DataFrame or None
            DataFrame with sorted values or None if ``inplace=True``.

        See Also
        --------
        DataFrame.sort_index : Sort a DataFrame by the index.
        Series.sort_values : Similar method for a Series.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "col1": ["A", "A", "B", np.nan, "D", "C"],
        ...         "col2": [2, 1, 9, 8, 7, 4],
        ...         "col3": [0, 1, 9, 4, 2, 3],
        ...         "col4": ["a", "B", "c", "D", "e", "F"],
        ...     }
        ... )
        >>> df
          col1  col2  col3 col4
        0    A     2     0    a
        1    A     1     1    B
        2    B     9     9    c
        3  NaN     8     4    D
        4    D     7     2    e
        5    C     4     3    F

        Sort by col1

        >>> df.sort_values(by=["col1"])
          col1  col2  col3 col4
        0    A     2     0    a
        1    A     1     1    B
        2    B     9     9    c
        5    C     4     3    F
        4    D     7     2    e
        3  NaN     8     4    D

        Sort by multiple columns

        >>> df.sort_values(by=["col1", "col2"])
          col1  col2  col3 col4
        1    A     1     1    B
        0    A     2     0    a
        2    B     9     9    c
        5    C     4     3    F
        4    D     7     2    e
        3  NaN     8     4    D

        Sort Descending

        >>> df.sort_values(by="col1", ascending=False)
          col1  col2  col3 col4
        4    D     7     2    e
        5    C     4     3    F
        2    B     9     9    c
        0    A     2     0    a
        1    A     1     1    B
        3  NaN     8     4    D

        Putting NAs first

        >>> df.sort_values(by="col1", ascending=False, na_position="first")
          col1  col2  col3 col4
        3  NaN     8     4    D
        4    D     7     2    e
        5    C     4     3    F
        2    B     9     9    c
        0    A     2     0    a
        1    A     1     1    B

        Sorting with a key function

        >>> df.sort_values(by="col4", key=lambda col: col.str.lower())
           col1  col2  col3 col4
        0    A     2     0    a
        1    A     1     1    B
        2    B     9     9    c
        3  NaN     8     4    D
        4    D     7     2    e
        5    C     4     3    F

        Natural sort with the key argument,
        using the `natsort <https://github.com/SethMMorton/natsort>` package.

        >>> df = pd.DataFrame(
        ...     {
        ...         "hours": ["0hr", "128hr", "0hr", "64hr", "64hr", "128hr"],
        ...         "mins": [
        ...             "10mins",
        ...             "40mins",
        ...             "40mins",
        ...             "40mins",
        ...             "10mins",
        ...             "10mins",
        ...         ],
        ...         "value": [10, 20, 30, 40, 50, 60],
        ...     }
        ... )
        >>> df
           hours    mins  value
        0    0hr  10mins     10
        1  128hr  40mins     20
        2    0hr  40mins     30
        3   64hr  40mins     40
        4   64hr  10mins     50
        5  128hr  10mins     60
        >>> from natsort import natsort_keygen
        >>> df.sort_values(
        ...     by=["hours", "mins"],
        ...     key=natsort_keygen(),
        ... )
           hours    mins  value
        0    0hr  10mins     10
        2    0hr  40mins     30
        4   64hr  10mins     50
        3   64hr  40mins     40
        5  128hr  10mins     60
        1  128hr  40mins     20
        """
        raise AbstractMethodError(self)

    @overload
    def sort_index(
        self,
        *,
        axis: Axis = ...,
        level: IndexLabel = ...,
        ascending: bool | Sequence[bool] = ...,
        inplace: Literal[True],
        kind: SortKind = ...,
        na_position: NaPosition = ...,
        sort_remaining: bool = ...,
        ignore_index: bool = ...,
        key: IndexKeyFunc = ...,
    ) -> None:
        # [FUNCTION DOC]
        # Typing overload stub only (body is a placeholder); the behavior is defined
        # by the implementation that follows these overloads. sort_index variant for
        # inplace true.
        ...

    @overload
    def sort_index(
        self,
        *,
        axis: Axis = ...,
        level: IndexLabel = ...,
        ascending: bool | Sequence[bool] = ...,
        inplace: Literal[False] = ...,
        kind: SortKind = ...,
        na_position: NaPosition = ...,
        sort_remaining: bool = ...,
        ignore_index: bool = ...,
        key: IndexKeyFunc = ...,
    ) -> Self:
        # [FUNCTION DOC]
        # Typing overload stub only (body is a placeholder); the behavior is defined
        # by the implementation that follows these overloads. sort_index variant for
        # inplace false.
        ...

    @overload
    def sort_index(
        self,
        *,
        axis: Axis = ...,
        level: IndexLabel = ...,
        ascending: bool | Sequence[bool] = ...,
        inplace: bool = ...,
        kind: SortKind = ...,
        na_position: NaPosition = ...,
        sort_remaining: bool = ...,
        ignore_index: bool = ...,
        key: IndexKeyFunc = ...,
    ) -> Self | None:
        # [FUNCTION DOC]
        # Typing overload stub only (body is a placeholder); the behavior is defined
        # by the implementation that follows these overloads. sort_index variant for
        # general boolean inplace.
        ...

    def sort_index(
        self,
        *,
        axis: Axis = 0,
        level: IndexLabel | None = None,
        ascending: bool | Sequence[bool] = True,
        inplace: bool = False,
        kind: SortKind = "quicksort",
        na_position: NaPosition = "last",
        sort_remaining: bool = True,
        ignore_index: bool = False,
        key: IndexKeyFunc | None = None,
    ) -> Self | None:
        # [FUNCTION DOC]
        # Sort the object by its index labels.
        # Parameters (keyword only): axis (default 0); level is a level or list of
        # levels for MultiIndex sorting; ascending bool or list of bools per level;
        # inplace; kind is the sorting algorithm; na_position is first or last;
        # sort_remaining sorts by remaining levels after the chosen ones; ignore_index
        # relabels the sorted axis with a default range; key is a function applied to
        # the labels before sorting.
        # Validates the inplace flag and ascending value. A sorting indexer is
        # computed by the sorting helper; if it is None (already sorted) the object
        # itself or a shallow copy is returned, with a fresh default index applied if
        # ignore_index (columns when axis is one). Otherwise the manager is taken with
        # the indexer, the sorted axis is rebuilt (with MultiIndex levels made
        # monotonic or replaced by a default index when ignoring the index) and the
        # result is built. Returns None when inplace, otherwise the finalized new
        # object with the method name sort_index.
        inplace = validate_bool_kwarg(inplace, "inplace")
        axis = self._get_axis_number(axis)
        ascending = validate_ascending(ascending)

        target = self._get_axis(axis)

        indexer = get_indexer_indexer(
            target, level, ascending, kind, na_position, sort_remaining, key
        )

        if indexer is None:
            if inplace:
                result = self
            else:
                result = self.copy(deep=False)

            if ignore_index:
                if axis == 1:
                    result.columns = default_index(len(self.columns))
                else:
                    result.index = default_index(len(self))
            if inplace:
                return None
            else:
                return result

        baxis = self._get_block_manager_axis(axis)
        new_data = self._mgr.take(indexer, axis=baxis, verify=False)

        # reconstruct axis if needed
        if not ignore_index:
            new_axis = new_data.axes[baxis]._sort_levels_monotonic()
        else:
            new_axis = default_index(len(indexer))
        new_data.set_axis(baxis, new_axis)

        result = self._constructor_from_mgr(new_data, axes=new_data.axes)

        if inplace:
            return self._update_inplace(result)
        else:
            return result.__finalize__(self, method="sort_index")

    def reindex(
        self,
        labels=None,
        *,
        index=None,
        columns=None,
        axis: Axis | None = None,
        method: ReindexMethod | None = None,
        copy: bool | lib.NoDefault = lib.no_default,
        level: Level | None = None,
        fill_value: Scalar | None = np.nan,
        limit: int | None = None,
        tolerance=None,
    ) -> Self:
        # [FUNCTION DOC]
        # Conform the object to new labels with optional fill logic.
        # Parameters: labels is the new labels for the axis chosen by axis; keyword
        # only: index and columns give per-axis labels; axis; method is the
        # fill/nearest method for inexact matching (None means no fill); copy is a
        # deprecated placeholder; level is the MultiIndex level to broadcast across;
        # fill_value is used for new missing entries (default missing marker); limit
        # caps consecutive fills; tolerance caps label distance.
        # Raises a type error if labels, index and columns are all given, or if axis
        # is combined with index or columns. Labels are mapped to the index (default)
        # or columns (when axis is one); when both index and columns are given along
        # with labels, labels fill whichever is missing.
        # When reindexing a non-MultiIndex index to a MultiIndex target whose level
        # names include the current index name and no level was given, that level is
        # selected automatically.
        # If every requested axis is identical (labels, names and dtype) to the
        # existing one a shallow copy is returned. If all axes are requested and no
        # method or level is used and a fast transpose is possible the multi-axis path
        # (abstract here, implemented by DataFrame) is used. Otherwise the axes are
        # reindexed one at a time. The method name is normalized by the missing-value
        # helper. Result is finalized with the method name reindex.
        """
        Conform Series/DataFrame to new index with optional filling logic.

        Places NA/NaN in locations having no value in the previous index. A new object
        is produced unless the new index is equivalent to the current one and
        ``copy=False``.

        Parameters
        ----------
        method : {None, 'backfill'/'bfill', 'pad'/'ffill', 'nearest'}
            Method to use for filling holes in reindexed DataFrame.
            Please note: this is only applicable to DataFrames/Series with a
            monotonically increasing/decreasing index.

            * None (default): don't fill gaps
            * pad / ffill: Propagate last valid observation forward to next
              valid.
            * backfill / bfill: Use next valid observation to fill gap.
            * nearest: Use nearest valid observations to fill gap.

        copy : bool, default False
            This keyword is now ignored; changing its value will have no
            impact on the method.

            .. deprecated:: 3.0.0

                This keyword is ignored and will be removed in pandas 4.0. Since
                pandas 3.0, this method always returns a new object using a lazy
                copy mechanism that defers copies until necessary
                (Copy-on-Write). See the `user guide on Copy-on-Write
                <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                for more details.

        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : scalar, default np.nan
            Value to use for missing values. Defaults to NaN, but can be any
            "compatible" value.
        limit : int, default None
            Maximum number of consecutive elements to forward or backward fill.
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations most
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.

            Tolerance may be a scalar value, which applies the same tolerance
            to all values, or list-like, which applies variable tolerance per
            element. List-like includes list, tuple, array, Series, and must be
            the same size as the index and its dtype must exactly match the
            index's type.

        Returns
        -------
        Series/DataFrame
            Series/DataFrame with changed index.

        See Also
        --------
        DataFrame.set_index : Set row labels.
        DataFrame.reset_index : Remove row labels or move them to new columns.
        DataFrame.reindex_like : Change to same indices as other DataFrame.

        Examples
        --------
        ``DataFrame.reindex`` supports two calling conventions

        * ``(index=index_labels, columns=column_labels, ...)``
        * ``(labels, axis={'index', 'columns'}, ...)``

        We *highly* recommend using keyword arguments to clarify your
        intent.

        Create a DataFrame with some fictional data.

        >>> index = ["Firefox", "Chrome", "Safari", "IE10", "Konqueror"]
        >>> columns = ["http_status", "response_time"]
        >>> df = pd.DataFrame(
        ...     [[200, 0.04], [200, 0.02], [404, 0.07], [404, 0.08], [301, 1.0]],
        ...     columns=columns,
        ...     index=index,
        ... )
        >>> df
                   http_status  response_time
        Firefox            200           0.04
        Chrome             200           0.02
        Safari             404           0.07
        IE10               404           0.08
        Konqueror          301           1.00

        Create a new index and reindex the DataFrame. By default
        values in the new index that do not have corresponding
        records in the DataFrame are assigned ``NaN``.

        >>> new_index = ["Safari", "Iceweasel", "Comodo Dragon", "IE10", "Chrome"]
        >>> df.reindex(new_index)
                       http_status  response_time
        Safari               404.0           0.07
        Iceweasel              NaN            NaN
        Comodo Dragon          NaN            NaN
        IE10                 404.0           0.08
        Chrome               200.0           0.02

        We can fill in the missing values by passing a value to
        the keyword ``fill_value``. Because the index is not monotonically
        increasing or decreasing, we cannot use arguments to the keyword
        ``method`` to fill the ``NaN`` values.

        >>> df.reindex(new_index, fill_value=0)
                       http_status  response_time
        Safari                 404           0.07
        Iceweasel                0           0.00
        Comodo Dragon            0           0.00
        IE10                   404           0.08
        Chrome                 200           0.02

        >>> df.reindex(new_index, fill_value="missing")
                      http_status response_time
        Safari                404          0.07
        Iceweasel         missing       missing
        Comodo Dragon     missing       missing
        IE10                  404          0.08
        Chrome                200          0.02

        We can also reindex the columns.

        >>> df.reindex(columns=["http_status", "user_agent"])
                   http_status  user_agent
        Firefox            200         NaN
        Chrome             200         NaN
        Safari             404         NaN
        IE10               404         NaN
        Konqueror          301         NaN

        Or we can use "axis-style" keyword arguments

        >>> df.reindex(["http_status", "user_agent"], axis="columns")
                   http_status  user_agent
        Firefox            200         NaN
        Chrome             200         NaN
        Safari             404         NaN
        IE10               404         NaN
        Konqueror          301         NaN

        To further illustrate the filling functionality in
        ``reindex``, we will create a DataFrame with a
        monotonically increasing index (for example, a sequence
        of dates).

        >>> date_index = pd.date_range("1/1/2010", periods=6, freq="D")
        >>> df2 = pd.DataFrame(
        ...     {"prices": [100, 101, np.nan, 100, 89, 88]}, index=date_index
        ... )
        >>> df2
                    prices
        2010-01-01   100.0
        2010-01-02   101.0
        2010-01-03     NaN
        2010-01-04   100.0
        2010-01-05    89.0
        2010-01-06    88.0

        Suppose we decide to expand the DataFrame to cover a wider
        date range.

        >>> date_index2 = pd.date_range("12/29/2009", periods=10, freq="D")
        >>> df2.reindex(date_index2)
                    prices
        2009-12-29     NaN
        2009-12-30     NaN
        2009-12-31     NaN
        2010-01-01   100.0
        2010-01-02   101.0
        2010-01-03     NaN
        2010-01-04   100.0
        2010-01-05    89.0
        2010-01-06    88.0
        2010-01-07     NaN

        The index entries that did not have a value in the original data frame
        (for example, '2009-12-29') are by default filled with ``NaN``.
        If desired, we can fill in the missing values using one of several
        options.

        For example, to back-propagate the last valid value to fill the ``NaN``
        values, pass ``bfill`` as an argument to the ``method`` keyword.

        >>> df2.reindex(date_index2, method="bfill")
                    prices
        2009-12-29   100.0
        2009-12-30   100.0
        2009-12-31   100.0
        2010-01-01   100.0
        2010-01-02   101.0
        2010-01-03     NaN
        2010-01-04   100.0
        2010-01-05    89.0
        2010-01-06    88.0
        2010-01-07     NaN

        Please note that the ``NaN`` value present in the original DataFrame
        (at index value 2010-01-03) will not be filled by any of the
        value propagation schemes. This is because filling while reindexing
        does not look at DataFrame values, but only compares the original and
        desired indexes. If you do want to fill in the ``NaN`` values present
        in the original DataFrame, use the ``fillna()`` method.

        See the :ref:`user guide <basics.reindexing>` for more.
        """
        # TODO: Decide if we care about having different examples for different
        # kinds

        # Automatically detect matching level when reindexing from Index to MultiIndex.
        # This prevents values from being incorrectly set to NaN when the source index
        # name matches a index name in the target MultiIndex
        if (
            level is None
            and index is not None
            and isinstance(index, MultiIndex)
            and not isinstance(self.index, MultiIndex)
            and self.index.name in index.names
        ):
            level = self.index.name
        self._check_copy_deprecation(copy)

        if index is not None and columns is not None and labels is not None:
            raise TypeError("Cannot specify all of 'labels', 'index', 'columns'.")
        elif index is not None or columns is not None:
            if axis is not None:
                raise TypeError(
                    "Cannot specify both 'axis' and any of 'index' or 'columns'"
                )
            if labels is not None:
                if index is not None:
                    columns = labels
                else:
                    index = labels
        elif axis and self._get_axis_number(axis) == 1:
            columns = labels
        else:
            index = labels
        axes: dict[Literal["index", "columns"], Any] = {
            "index": index,
            "columns": columns,
        }
        method = clean_reindex_fill_method(method)

        # if all axes that are requested to reindex are equal, then only copy
        # if indicated must have index names equal here as well as values
        if all(
            self._get_axis(axis_name).identical(ax)
            for axis_name, ax in axes.items()
            if ax is not None
        ):
            return self.copy(deep=False)

        # check if we are a multi reindex
        if self._needs_reindex_multi(axes, method, level):
            return self._reindex_multi(axes, fill_value)

        # perform the reindex on the axes
        return self._reindex_axes(
            axes, level, limit, tolerance, method, fill_value
        ).__finalize__(self, method="reindex")

    @final
    def _reindex_axes(
        self,
        axes,
        level: Level | None,
        limit: int | None,
        tolerance,
        method,
        fill_value: Scalar | None,
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: performs the reindex for each axis that has requested labels.
        # Parameters: axes is a mapping from axis name to new labels (or None to
        # skip); level, limit, tolerance, method, fill_value are forwarded.
        # For each axis in order, the axis reindex computes the new labels and an
        # indexer, and the indexer is applied to the data through the
        # reindex-with-indexers helper without allowing duplicates. Returns the final
        # object.
        """Perform the reindex for all the axes."""
        obj = self
        for a in self._AXIS_ORDERS:
            labels = axes[a]
            if labels is None:
                continue

            ax = self._get_axis(a)
            new_index, indexer = ax.reindex(
                labels, level=level, limit=limit, tolerance=tolerance, method=method
            )

            axis = self._get_axis_number(a)
            obj = obj._reindex_with_indexers(
                {axis: [new_index, indexer]},
                fill_value=fill_value,
                allow_dups=False,
            )

        return obj

    def _needs_reindex_multi(self, axes, method, level: Level | None) -> bool:
        # [FUNCTION DOC]
        # HELPER: decides whether the multi-axis reindex fast path applies.
        # Parameters: axes mapping of axis name to labels; method; level. True only if
        # labels were given for every axis, method and level are None, and the object
        # can be transposed cheaply (cheap access to the values array). Returns a
        # bool.
        """Check if we do need a multi reindex."""
        return (
            (common.count_not_none(*axes.values()) == self._AXIS_LEN)
            and method is None
            and level is None
            # reindex_multi calls self.values, so we only want to go
            #  down that path when doing so is cheap.
            and self._can_fast_transpose
        )

    def _reindex_multi(self, axes, fill_value):
        # [FUNCTION DOC]
        # HELPER: abstract hook for reindexing all axes at once; always raises an
        # abstract-method error. DataFrame provides the implementation.
        raise AbstractMethodError(self)

    @final
    def _reindex_with_indexers(
        self,
        reindexers,
        fill_value=None,
        allow_dups: bool = False,
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: applies precomputed indexers to the data, one axis at a time in
        # sorted axis order.
        # Parameters: reindexers maps an axis number to a pair of new labels and a
        # position indexer (either may be None; entries with None labels are skipped);
        # fill_value is inserted at positions with indexer -1; allow_dups says
        # duplicates in the labels are permitted (internal calls).
        # The labels are converted to an Index and the indexer to a platform integer
        # array before calling the manager reindex. If nothing changed, the manager is
        # shallow-copied. Returns a new object finalized from self.
        """allow_dups indicates an internal call here"""
        # reindex doing multiple operations on different axes if indicated
        new_data = self._mgr
        for axis in sorted(reindexers.keys()):
            index, indexer = reindexers[axis]
            baxis = self._get_block_manager_axis(axis)

            if index is None:
                continue

            index = ensure_index(index)
            if indexer is not None:
                indexer = ensure_platform_int(indexer)

            # TODO: speed up on homogeneous DataFrame objects (see _reindex_multi)
            new_data = new_data.reindex_indexer(
                index,
                indexer,
                axis=baxis,
                fill_value=fill_value,
                allow_dups=allow_dups,
            )

        if new_data is self._mgr:
            new_data = new_data.copy(deep=False)

        return self._constructor_from_mgr(new_data, axes=new_data.axes).__finalize__(
            self
        )

    def filter(
        self,
        items=None,
        like: str | None = None,
        regex: str | None = None,
        axis: Axis | None = None,
    ) -> Self:
        # [FUNCTION DOC]
        # Subset the object by labels of an axis.
        # Parameters: items is a list of labels to keep; like keeps labels containing
        # that substring; regex keeps labels matching the regular expression (search
        # semantics); axis defaults to the info axis (columns for frames, index for
        # Series).
        # Raises a type error if more than one of items, like, regex is given, or if
        # none is truthy-given (empty like or regex counts as not given for the
        # selection branches). Does not check that selected labels exist; items not
        # present are ignored via an intersection that preserves the dtype of the
        # labels when empty.
        # items go through reindex; like and regex map a predicate over the string
        # forms of the labels and select with a boolean mask via the label indexer on
        # that axis. Returns a new object.
        """
        Subset the DataFrame or Series according to the specified index labels.

        For DataFrame, filter rows or columns depending on ``axis`` argument.
        Note that this routine does not filter based on content.
        The filter is applied to the labels of the index.

        Parameters
        ----------
        items : list-like
            Keep labels from axis which are in items.
        like : str
            Keep labels from axis for which "like in label == True".
        regex : str (regular expression)
            Keep labels from axis for which re.search(regex, label) == True.
        axis : {0 or 'index', 1 or 'columns', None}, default None
            The axis to filter on, expressed either as an index (int)
            or axis name (str). By default this is the info axis, 'columns' for
            ``DataFrame``. For ``Series`` this parameter is unused and defaults to
            ``None``.

        Returns
        -------
        Same type as caller
            The filtered subset of the DataFrame or Series.

        See Also
        --------
        DataFrame.loc : Access a group of rows and columns
            by label(s) or a boolean array.

        Notes
        -----
        The ``items``, ``like``, and ``regex`` parameters are
        enforced to be mutually exclusive.

        ``axis`` defaults to the info axis that is used when indexing
        with ``[]``.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     np.array(([1, 2, 3], [4, 5, 6])),
        ...     index=["mouse", "rabbit"],
        ...     columns=["one", "two", "three"],
        ... )
        >>> df
                one  two  three
        mouse     1    2      3
        rabbit    4    5      6

        >>> # select columns by name
        >>> df.filter(items=["one", "three"])
                 one  three
        mouse     1      3
        rabbit    4      6

        >>> # select columns by regular expression
        >>> df.filter(regex="e$", axis=1)
                 one  three
        mouse     1      3
        rabbit    4      6

        >>> # select rows containing 'bbi'
        >>> df.filter(like="bbi", axis=0)
                 one  two  three
        rabbit    4    5      6
        """
        nkw = common.count_not_none(items, like, regex)
        if nkw > 1:
            raise TypeError(
                "Keyword arguments `items`, `like`, or `regex` are mutually exclusive"
            )

        if axis is None:
            axis = self._info_axis_name
        labels = self._get_axis(axis)

        if items is not None:
            name = self._get_axis_name(axis)
            items = Index(items).intersection(labels)
            if len(items) == 0:
                # Keep the dtype of labels when we are empty
                items = items.astype(labels.dtype)
            # error: Keywords must be strings
            return self.reindex(**{name: items})  # type: ignore[misc]
        elif like:

            def f(x) -> bool:
                # [FUNCTION DOC]
                # HELPER: predicate used by the filter method when like is given.
                # Parameters: x is a label. Returns whether the like substring is
                # contained in the string form of the label.
                assert like is not None  # needed for mypy
                return like in ensure_str(x)

            values = labels.map(f)
            return self.loc(axis=axis)[values]
        elif regex:

            def f(x) -> bool:
                # [FUNCTION DOC]
                # HELPER: predicate used by the filter method when regex is given.
                # Parameters: x is a label. Returns whether the compiled pattern finds
                # a match anywhere in the string form of the label.
                return matcher.search(ensure_str(x)) is not None

            matcher = re.compile(regex)
            values = labels.map(f)
            return self.loc(axis=axis)[values]
        else:
            raise TypeError("Must pass either `items`, `like`, or `regex`")

    @final
    def head(self, n: int = 5) -> Self:
        # [FUNCTION DOC]
        # Return the first n rows.
        # Parameters: n is the number of rows (default 5); a negative value means all
        # rows except the last absolute n.
        # Uses positional slicing and returns a copy (a lazily copied object under
        # copy-on-write).
        """
        Return the first `n` rows.

        This function exhibits the same behavior as ``df[:n]``, returning the
        first ``n`` rows based on position. It is useful for quickly checking
        if your object has the right type of data in it.

        When ``n`` is positive, it returns the first ``n`` rows. For ``n`` equal to 0,
        it returns an empty object. When ``n`` is negative, it returns
        all rows except the last ``|n|`` rows, mirroring the behavior of ``df[:n]``.

        If ``n`` is larger than the number of rows, this function returns all rows.

        Parameters
        ----------
        n : int, default 5
            Number of rows to select.

        Returns
        -------
        same type as caller
            The first `n` rows of the caller object.

        See Also
        --------
        DataFrame.tail: Returns the last `n` rows.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "animal": [
        ...             "alligator",
        ...             "bee",
        ...             "falcon",
        ...             "lion",
        ...             "monkey",
        ...             "parrot",
        ...             "shark",
        ...             "whale",
        ...             "zebra",
        ...         ]
        ...     }
        ... )
        >>> df
              animal
        0  alligator
        1        bee
        2     falcon
        3       lion
        4     monkey
        5     parrot
        6      shark
        7      whale
        8      zebra

        Viewing the first 5 lines

        >>> df.head()
              animal
        0  alligator
        1        bee
        2     falcon
        3       lion
        4     monkey

        Viewing the first `n` lines (three in this case)

        >>> df.head(3)
              animal
        0  alligator
        1        bee
        2     falcon

        For negative values of `n`

        >>> df.head(-3)
              animal
        0  alligator
        1        bee
        2     falcon
        3       lion
        4     monkey
        5     parrot
        """
        return self.iloc[:n].copy()

    @final
    def tail(self, n: int = 5) -> Self:
        # [FUNCTION DOC]
        # Return the last n rows.
        # Parameters: n is the number of rows (default 5); a negative value means all
        # rows except the first absolute n.
        # Zero returns an empty slice (copy); otherwise positional slicing from the
        # end is used and the result is copied.
        """
        Return the last `n` rows.

        This function returns last `n` rows from the object based on
        position. It is useful for quickly verifying data, for example,
        after sorting or appending rows.

        For negative values of `n`, this function returns all rows except
        the first `|n|` rows, equivalent to ``df[|n|:]``.

        If ``n`` is larger than the number of rows, this function returns all rows.

        Parameters
        ----------
        n : int, default 5
            Number of rows to select.

        Returns
        -------
        type of caller
            The last `n` rows of the caller object.

        See Also
        --------
        DataFrame.head : The first `n` rows of the caller object.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "animal": [
        ...             "alligator",
        ...             "bee",
        ...             "falcon",
        ...             "lion",
        ...             "monkey",
        ...             "parrot",
        ...             "shark",
        ...             "whale",
        ...             "zebra",
        ...         ]
        ...     }
        ... )
        >>> df
              animal
        0  alligator
        1        bee
        2     falcon
        3       lion
        4     monkey
        5     parrot
        6      shark
        7      whale
        8      zebra

        Viewing the last 5 lines

        >>> df.tail()
           animal
        4  monkey
        5  parrot
        6   shark
        7   whale
        8   zebra

        Viewing the last `n` lines (three in this case)

        >>> df.tail(3)
          animal
        6  shark
        7  whale
        8  zebra

        For negative values of `n`

        >>> df.tail(-3)
           animal
        3    lion
        4  monkey
        5  parrot
        6   shark
        7   whale
        8   zebra
        """
        if n == 0:
            return self.iloc[0:0].copy()
        return self.iloc[-n:].copy()

    @final
    def sample(
        self,
        n: int | None = None,
        frac: float | None = None,
        replace: bool = False,
        weights=None,
        random_state: RandomState | None = None,
        axis: Axis | None = None,
        ignore_index: bool = False,
    ) -> Self:
        # [FUNCTION DOC]
        # Return a random sample of items along an axis.
        # Parameters: n is the number of items (mutually exclusive with frac); frac is
        # a fraction of the axis length (rounded to the nearest integer count);
        # replace allows sampling the same item more than once; weights is a per-item
        # probability (a name, array-like or Series aligned on the axis); random_state
        # is a seed or generator; axis defaults to 0; ignore_index replaces the
        # sampled index with a default range.
        # The count is validated by the sampling helper module (which raises a value
        # error for invalid combinations such as both n and frac, or n larger than the
        # axis without replacement). Weights are preprocessed and validated. Random
        # positions are drawn using the helper and used with take. Returns a new
        # object.
        """
        Return a random sample of items from an axis of object.

        You can use `random_state` for reproducibility.

        Parameters
        ----------
        n : int, optional
            Number of items from axis to return. Cannot be used with `frac`.
            Default = 1 if `frac` = None.
        frac : float, optional
            Fraction of axis items to return. Cannot be used with `n`.
        replace : bool, default False
            Allow or disallow sampling of the same row more than once.
        weights : str or ndarray-like, optional
            Default ``None`` results in equal probability weighting.
            If passed a Series, will align with target object on index. Index
            values in weights not found in sampled object will be ignored and
            index values in sampled object not in weights will be assigned
            weights of zero.
            If called on a DataFrame, will accept the name of a column
            when axis = 0.
            Unless weights are a Series, weights must be same length as axis
            being sampled.
            If weights do not sum to 1, they will be normalized to sum to 1.
            Missing values in the weights column will be treated as zero.
            Infinite values not allowed.
            When replace = False will not allow ``(n * max(weights) / sum(weights)) > 1``
            in order to avoid biased results. See the Notes below for more details.
        random_state : int, array-like, BitGenerator, np.random.RandomState, np.random.Generator, optional
            If int, array-like, or BitGenerator, seed for random number generator.
            If np.random.RandomState or np.random.Generator, use as given.
            Default ``None`` results in sampling with the current state of np.random.
        axis : {0 or 'index', 1 or 'columns', None}, default None
            Axis to sample. Accepts axis number or name. Default is stat axis
            for given data type. For `Series` this parameter is unused and defaults to `None`.
        ignore_index : bool, default False
            If True, the resulting index will be labeled 0, 1, …, n - 1.

        Returns
        -------
        Series or DataFrame
            A new object of same type as caller containing `n` items randomly
            sampled from the caller object.

        See Also
        --------
        DataFrameGroupBy.sample: Generates random samples from each group of a
            DataFrame object.
        SeriesGroupBy.sample: Generates random samples from each group of a
            Series object.
        numpy.random.choice: Generates a random sample from a given 1-D numpy
            array.

        Notes
        -----
        If `frac` > 1, `replacement` should be set to `True`.

        When replace = False will not allow ``(n * max(weights) / sum(weights)) > 1``,
        since that would cause results to be biased. E.g. sampling 2 items without replacement
        with weights [100, 1, 1] would yield two last items in 1/2 of cases, instead of 1/102.
        This is similar to specifying `n=4` without replacement on a Series with 3 elements.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "num_legs": [2, 4, 8, 0],
        ...         "num_wings": [2, 0, 0, 0],
        ...         "num_specimen_seen": [10, 2, 1, 8],
        ...     },
        ...     index=["falcon", "dog", "spider", "fish"],
        ... )
        >>> df
                num_legs  num_wings  num_specimen_seen
        falcon         2          2                 10
        dog            4          0                  2
        spider         8          0                  1
        fish           0          0                  8

        Extract 3 random elements from the ``Series`` ``df['num_legs']``:
        Note that we use `random_state` to ensure the reproducibility of
        the examples.

        >>> df["num_legs"].sample(n=3, random_state=1)
        fish      0
        spider    8
        falcon    2
        Name: num_legs, dtype: int64

        A random 50% sample of the ``DataFrame`` with replacement:

        >>> df.sample(frac=0.5, replace=True, random_state=1)
              num_legs  num_wings  num_specimen_seen
        dog          4          0                  2
        fish         0          0                  8

        An upsample sample of the ``DataFrame`` with replacement:
        Note that `replace` parameter has to be `True` for `frac` parameter > 1.

        >>> df.sample(frac=2, replace=True, random_state=1)
                num_legs  num_wings  num_specimen_seen
        dog            4          0                  2
        fish           0          0                  8
        falcon         2          2                 10
        falcon         2          2                 10
        fish           0          0                  8
        dog            4          0                  2
        fish           0          0                  8
        dog            4          0                  2

        Using a DataFrame column as weights. Rows with larger value in the
        `num_specimen_seen` column are more likely to be sampled.

        >>> df.sample(n=2, weights="num_specimen_seen", random_state=1)
                num_legs  num_wings  num_specimen_seen
        falcon         2          2                 10
        fish           0          0                  8
        """  # noqa: E501
        if axis is None:
            axis = 0

        axis = self._get_axis_number(axis)
        obj_len = self.shape[axis]

        # Process random_state argument
        rs = common.random_state(random_state)

        size = sample.process_sampling_size(n, frac, replace)
        if size is None:
            assert frac is not None
            size = round(frac * obj_len)

        if weights is not None:
            weights = sample.preprocess_weights(self, weights, axis)

        sampled_indices = sample.sample(obj_len, size, replace, weights, rs)
        result = self.take(sampled_indices, axis=axis)

        if ignore_index:
            result.index = default_index(len(result))

        return result

    @overload
    def pipe(
        self,
        func: Callable[Concatenate[Self, P], T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        # [FUNCTION DOC]
        # Typing overload stub only (body is a placeholder); the behavior is defined
        # by the implementation that follows these overloads. pipe variant where func
        # is a callable.
        ...

    @overload
    def pipe(
        self,
        func: tuple[Callable[..., T], str],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        # [FUNCTION DOC]
        # Typing overload stub only (body is a placeholder); the behavior is defined
        # by the implementation that follows these overloads. pipe variant where func
        # is a pair of a callable and the keyword name receiving the object.
        ...

    @final
    def pipe(
        self,
        func: Callable[Concatenate[Self, P], T] | tuple[Callable[..., T], str],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        # [FUNCTION DOC]
        # Apply a function to the object in a chain.
        # Parameters: func is a callable taking the object as first argument, or a
        # pair of a callable and a string naming the keyword argument that should
        # receive the object; extra positional and keyword arguments are passed
        # through.
        # Calls the shared pipe helper with a shallow copy of the object. Returns
        # whatever the function returns. Raises a value error if the keyword name
        # given in the pair is also among the keyword arguments (from the helper).
        r"""
        Apply chainable functions that expect Series or DataFrames.

        Parameters
        ----------
        func : function
            Function to apply to the Series/DataFrame.
            ``args``, and ``kwargs`` are passed into ``func``.
            Alternatively a ``(callable, data_keyword)`` tuple where
            ``data_keyword`` is a string indicating the keyword of
            ``callable`` that expects the Series/DataFrame.
        *args : iterable, optional
            Positional arguments passed into ``func``.
        **kwargs : mapping, optional
            A dictionary of keyword arguments passed into ``func``.

        Returns
        -------
        The return type of ``func``.
            The result of applying ``func`` to the Series or DataFrame.

        See Also
        --------
        DataFrame.apply : Apply a function along input axis of DataFrame.
        DataFrame.map : Apply a function elementwise on a whole DataFrame.
        Series.map : Apply a mapping correspondence on a
            :class:`~pandas.Series`.

        Notes
        -----
        Use ``.pipe`` when chaining together functions that expect
        Series, DataFrames or GroupBy objects.

        Examples
        --------
        Constructing an income DataFrame from a dictionary.

        >>> data = [[8000, 1000], [9500, np.nan], [5000, 2000]]
        >>> df = pd.DataFrame(data, columns=["Salary", "Others"])
        >>> df
           Salary  Others
        0    8000  1000.0
        1    9500     NaN
        2    5000  2000.0

        Functions that perform tax reductions on an income DataFrame.

        >>> def subtract_federal_tax(df):
        ...     return df * 0.9
        >>> def subtract_state_tax(df, rate):
        ...     return df * (1 - rate)
        >>> def subtract_national_insurance(df, rate, rate_increase):
        ...     new_rate = rate + rate_increase
        ...     return df * (1 - new_rate)

        Instead of writing

        >>> subtract_national_insurance(
        ...     subtract_state_tax(subtract_federal_tax(df), rate=0.12),
        ...     rate=0.05,
        ...     rate_increase=0.02,
        ... )  # doctest: +SKIP

        You can write

        >>> (
        ...     df.pipe(subtract_federal_tax)
        ...     .pipe(subtract_state_tax, rate=0.12)
        ...     .pipe(subtract_national_insurance, rate=0.05, rate_increase=0.02)
        ... )
            Salary   Others
        0  5892.48   736.56
        1  6997.32      NaN
        2  3682.80  1473.12

        If you have a function that takes the data as (say) the second
        argument, pass a tuple indicating which keyword expects the
        data. For example, suppose ``national_insurance`` takes its data as ``df``
        in the second argument:

        >>> def subtract_national_insurance(rate, df, rate_increase):
        ...     new_rate = rate + rate_increase
        ...     return df * (1 - new_rate)
        >>> (
        ...     df.pipe(subtract_federal_tax)
        ...     .pipe(subtract_state_tax, rate=0.12)
        ...     .pipe(
        ...         (subtract_national_insurance, "df"), rate=0.05, rate_increase=0.02
        ...     )
        ... )
            Salary   Others
        0  5892.48   736.56
        1  6997.32      NaN
        2  3682.80  1473.12
        """
        return common.pipe(self.copy(deep=False), func, *args, **kwargs)

    # ----------------------------------------------------------------------
    # Attribute access

    @final
    def __finalize__(self, other, method: str | None = None, **kwargs) -> Self:
        # [FUNCTION DOC]
        # Propagate metadata from another object to self (the default hook used after
        # most operations).
        # Parameters: other is the source; method is an optional operation name for
        # subclass overrides (not used here); extra keyword arguments are accepted and
        # ignored.
        # If other is an NDFrame: its attrs are deep-copied onto self only when
        # non-empty; the duplicate-labels flag becomes the logical and of both flags;
        # and every metadata attribute name shared by both classes is copied by
        # reference. If other instead exposes a list of input objects (for example a
        # concat or merge operation record): attrs are propagated only if all inputs
        # have non-empty equal attrs, and the duplicate-labels flag becomes true only
        # if every input allows duplicates. Returns self (mutated).
        """
        Propagate metadata from other to self.

        This is the default implementation. Subclasses may override this method to
        implement their own metadata handling.

        Parameters
        ----------
        other : the object from which to get the attributes that we are going
            to propagate. If ``other`` has an ``input_objs`` attribute, then
            this attribute must contain an iterable of objects, each with an
            ``attrs`` attribute.
        method : str, optional
            A passed method name providing context on where ``__finalize__``
            was called.

            .. warning::

               The value passed as `method` are not currently considered
               stable across pandas releases.

        Notes
        -----
        In case ``other`` has an ``input_objs`` attribute, this method only
        propagates its metadata if each object in ``input_objs`` has the exact
        same metadata as the others.
        """
        if isinstance(other, NDFrame):
            if other.attrs:
                # We want attrs propagation to have minimal performance
                # impact if attrs are not used; i.e. attrs is an empty dict.
                # One could make the deepcopy unconditionally, but a deepcopy
                # of an empty dict is 50x more expensive than the empty check.
                self.attrs = deepcopy(other.attrs)
            self.flags.allows_duplicate_labels = (
                self.flags.allows_duplicate_labels
                and other.flags.allows_duplicate_labels
            )
            # For subclasses using _metadata.
            for name in set(self._metadata) & set(other._metadata):
                assert isinstance(name, str)
                object.__setattr__(self, name, getattr(other, name, None))

        elif hasattr(other, "input_objs"):
            objs = other.input_objs
            # propagate attrs only if all inputs have the same attrs
            if all(bool(obj.attrs) for obj in objs):
                # all inputs have non-empty attrs
                attrs = objs[0].attrs
                have_same_attrs = all(obj.attrs == attrs for obj in objs[1:])
                if have_same_attrs:
                    self.attrs = deepcopy(attrs)

            allows_duplicate_labels = all(x.flags.allows_duplicate_labels for x in objs)
            self.flags.allows_duplicate_labels = allows_duplicate_labels

        return self

    @final
    def __getattr__(self, name: str):
        # [FUNCTION DOC]
        # Attribute fallback giving access to items of the info axis (for example
        # columns) as attributes.
        # Parameters: name is the attribute name that regular lookup failed to find.
        # If the name is not an internal, metadata or accessor name and the info axis
        # can hold identifiers and contains the name, the item getter result is
        # returned; otherwise the ordinary object attribute lookup is run again, which
        # raises the standard attribute error.
        """
        After regular attribute access, try looking up the name
        This allows simpler access to columns for interactive use.
        """
        # Note: obj.x will always call obj.__getattribute__('x') prior to
        # calling obj.__getattr__('x').
        if (
            name not in self._internal_names_set
            and name not in self._metadata
            and name not in self._accessors
            and self._info_axis._can_hold_identifiers_and_holds_name(name)
        ):
            return self[name]
        return object.__getattribute__(self, name)

    @final
    def __setattr__(self, name: str, value) -> None:
        # [FUNCTION DOC]
        # Attribute assignment that also supports setting items of the info axis.
        # Parameters: name; value.
        # First tries ordinary assignment when the attribute already exists on the
        # object. Otherwise internal and metadata names are set directly; for other
        # names, the existing value is looked up: if it is an Index it is set as a
        # plain attribute, if the name is in the info axis it is assigned as an item,
        # otherwise set as a plain attribute. If the lookup fails with an attribute or
        # type error, a DataFrame given a list-like value gets a warning (default user
        # warning class) that columns cannot be created through new attribute names,
        # and the value is set as a plain attribute anyway. Mutates self.
        """
        After regular attribute access, try setting the name
        This allows simpler access to columns for interactive use.
        """
        # first try regular attribute access via __getattribute__, so that
        # e.g. ``obj.x`` and ``obj.x = 4`` will always reference/modify
        # the same attribute.

        try:
            object.__getattribute__(self, name)
            return object.__setattr__(self, name, value)
        except AttributeError:
            pass

        # if this fails, go on to more involved attribute setting
        # (note that this matches __getattr__, above).
        if name in self._internal_names_set:
            object.__setattr__(self, name, value)
        elif name in self._metadata:
            object.__setattr__(self, name, value)
        else:
            try:
                existing = getattr(self, name)
                if isinstance(existing, Index):
                    object.__setattr__(self, name, value)
                elif name in self._info_axis:
                    self[name] = value
                else:
                    object.__setattr__(self, name, value)
            except (AttributeError, TypeError):
                if isinstance(self, ABCDataFrame) and (is_list_like(value)):
                    warnings.warn(
                        "Pandas doesn't allow columns to be "
                        "created via a new attribute name - see "
                        "https://pandas.pydata.org/pandas-docs/"
                        "stable/indexing.html#attribute-access",
                        stacklevel=find_stack_level(),
                    )
                object.__setattr__(self, name, value)

    @final
    def _dir_additions(self) -> set[str]:
        # [FUNCTION DOC]
        # HELPER: extends the attribute names offered by directory listing and
        # completion with the string labels of the info axis (for a MultiIndex, its
        # first level values). Returns the combined set of names.
        """
        add the string-like attributes from the info_axis.
        If info_axis is a MultiIndex, its first level values are used.
        """
        additions = super()._dir_additions()
        if self._info_axis._can_hold_strings:
            additions.update(self._info_axis._dir_additions_for_owner)
        return additions

    # ----------------------------------------------------------------------
    # Consolidation of internals

    @final
    def _consolidate_inplace(self) -> None:
        # [FUNCTION DOC]
        # HELPER: consolidates the manager in place (merges same-dtype blocks) by
        # replacing the stored manager with the consolidated one. Mutates self;
        # returns None.
        """Consolidate data in place and return None"""

        self._mgr = self._mgr.consolidate()

    @final
    def _consolidate(self):
        # [FUNCTION DOC]
        # HELPER: returns a new object with consolidated internals (blocks of the same
        # dtype merged), finalized from self.
        """
        Compute NDFrame with "consolidated" internals (data of each dtype
        grouped together in a single ndarray).

        Returns
        -------
        consolidated : same type as caller
        """
        cons_data = self._mgr.consolidate()
        return self._constructor_from_mgr(cons_data, axes=cons_data.axes).__finalize__(
            self
        )

    @final
    @property
    def _is_mixed_type(self) -> bool:
        # [FUNCTION DOC]
        # HELPER property: true if the data is heterogeneous in a way that prevents
        # consolidation into one array. A single block manager is never mixed; a
        # manager containing extension types counts as mixed; otherwise it is true
        # when more than one distinct dtype is present among the columns.
        if self._mgr.is_single_block:
            # Includes all Series cases
            return False

        if self._mgr.any_extension_types:
            # Even if they have the same dtype, we can't consolidate them,
            #  so we pretend this is "mixed'"
            return True

        return self.dtypes.nunique() > 1

    @final
    def _get_numeric_data(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: returns a new object with only the numeric blocks of the manager
        # (the numeric subset of columns), finalized from self.
        new_mgr = self._mgr.get_numeric_data()
        return self._constructor_from_mgr(new_mgr, axes=new_mgr.axes).__finalize__(self)

    @final
    def _get_bool_data(self):
        # [FUNCTION DOC]
        # HELPER: returns a new object with only the boolean blocks of the manager,
        # finalized from self.
        new_mgr = self._mgr.get_bool_data()
        return self._constructor_from_mgr(new_mgr, axes=new_mgr.axes).__finalize__(self)

    # ----------------------------------------------------------------------
    # Internal Interface Methods

    @property
    def values(self):
        # [FUNCTION DOC]
        # Abstract property for the public values array; always raises an
        # abstract-method error here. Series and DataFrame override it.
        raise AbstractMethodError(self)

    @property
    def _values(self) -> ArrayLike:
        # [FUNCTION DOC]
        # HELPER: abstract property for the internal array of values (may be an
        # extension array); always raises an abstract-method error here. Subclasses
        # override it.
        """internal implementation"""
        raise AbstractMethodError(self)

    @property
    def dtypes(self):
        # [FUNCTION DOC]
        # Property returning the dtype of each column (or of the Series element) as a
        # Series.
        # Gets the dtypes from the manager and builds a Series of object dtype indexed
        # by the info axis using the sliced constructor (so a DataFrame returns a
        # Series, a Series returns a one-element container).
        """
        Return the dtypes in the DataFrame.

        This returns a Series with the data type of each column.
        The result's index is the original DataFrame's columns. Columns
        with mixed types are stored with the ``object`` dtype. See
        :ref:`the User Guide <basics.dtypes>` for more.

        Returns
        -------
        pandas.Series
            The data type of each column.

        See Also
        --------
        Series.dtypes : Return the dtype object of the underlying data.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "float": [1.0],
        ...         "int": [1],
        ...         "datetime": [pd.Timestamp("20180310")],
        ...         "string": ["foo"],
        ...     }
        ... )
        >>> df.dtypes
        float              float64
        int                  int64
        datetime    datetime64[us]
        string              str
        dtype: object
        """
        data = self._mgr.get_dtypes()
        return self._constructor_sliced(data, index=self._info_axis, dtype=np.object_)

    @final
    def astype(
        self,
        dtype,
        copy: bool | lib.NoDefault = lib.no_default,
        errors: IgnoreRaise = "raise",
    ) -> Self:
        # [FUNCTION DOC]
        # Cast to the given dtype.
        # Parameters: dtype is a dtype, a type, or a dict-like mapping column names to
        # dtypes (for a Series only its own name is valid as the key); copy is a
        # deprecated placeholder; errors is raise (default) or ignore for conversion
        # failures.
        # Raises a key error when a Series dict mapping has more than one key or a key
        # other than the series name, or when a frame mapping names a column that does
        # not exist. Value errors from per-column conversions get a message suffix
        # naming the column.
        # Dict-like dtype: for a Series the single mapped dtype is applied; for a
        # frame each column is cast separately to its mapped dtype and columns without
        # a mapping (missing values after reindexing the mapping to the columns) are
        # shallow copied, then the results are concatenated along the column axis,
        # wrapped in the original class and given the original columns (preserving
        # duplicate names).
        # Extension dtype on a frame: if every block already has that dtype a shallow
        # copy is returned, else each column is converted separately and concatenated.
        # Otherwise the manager casts all blocks at once. An empty list of per-column
        # results gives a shallow copy. Results are finalized with the method name
        # astype. Returns a new object.
        """
        Cast a pandas object to a specified dtype ``dtype``.

        This method allows the conversion of the data types of pandas objects,
        including DataFrames and Series, to the specified dtype. It supports casting
        entire objects to a single data type or applying different data types to
        individual columns using a mapping.

        Parameters
        ----------
        dtype : str, data type, Series or Mapping of column name -> data type
            Use a str, numpy.dtype, pandas.ExtensionDtype or Python type to
            cast entire pandas object to the same type. Alternatively, use a
            mapping, e.g. {col: dtype, ...}, where col is a column label and dtype is
            a numpy.dtype or Python type to cast one or more of the DataFrame's
            columns to column-specific types.
        copy : bool, default False
            This keyword is now ignored; changing its value will have no
            impact on the method.

            .. deprecated:: 3.0.0

                This keyword is ignored and will be removed in pandas 4.0. Since
                pandas 3.0, this method always returns a new object using a lazy
                copy mechanism that defers copies until necessary
                (Copy-on-Write). See the `user guide on Copy-on-Write
                <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                for more details.

        errors : {'raise', 'ignore'}, default 'raise'
            Control raising of exceptions on invalid data for provided dtype.

            - ``raise`` : allow exceptions to be raised
            - ``ignore`` : suppress exceptions. On error return original object.

        Returns
        -------
        same type as caller
            The pandas object casted to the specified ``dtype``.

        See Also
        --------
        to_datetime : Convert argument to datetime.
        to_timedelta : Convert argument to timedelta.
        to_numeric : Convert argument to a numeric type.
        numpy.ndarray.astype : Cast a numpy array to a specified type.

        Notes
        -----
        .. versionchanged:: 2.0.0

            Using ``astype`` to convert from timezone-naive dtype to
            timezone-aware dtype will raise an exception.
            Use :meth:`Series.dt.tz_localize` instead.

        Examples
        --------
        Create a DataFrame:

        >>> d = {"col1": [1, 2], "col2": [3, 4]}
        >>> df = pd.DataFrame(data=d)
        >>> df.dtypes
        col1    int64
        col2    int64
        dtype: object

        Cast all columns to int32:

        >>> df.astype("int32").dtypes
        col1    int32
        col2    int32
        dtype: object

        Cast col1 to int32 using a dictionary:

        >>> df.astype({"col1": "int32"}).dtypes
        col1    int32
        col2    int64
        dtype: object

        Create a series:

        >>> ser = pd.Series([1, 2], dtype="int32")
        >>> ser
        0    1
        1    2
        dtype: int32
        >>> ser.astype("int64")
        0    1
        1    2
        dtype: int64

        Convert to categorical type:

        >>> ser.astype("category")
        0    1
        1    2
        dtype: category
        Categories (2, int32): [1, 2]

        Convert to ordered categorical type with custom ordering:

        >>> from pandas.api.types import CategoricalDtype
        >>> cat_dtype = CategoricalDtype(categories=[2, 1], ordered=True)
        >>> ser.astype(cat_dtype)
        0    1
        1    2
        dtype: category
        Categories (2, int64): [2 < 1]

        Create a series of dates:

        >>> ser_date = pd.Series(pd.date_range("20200101", periods=3))
        >>> ser_date
        0   2020-01-01
        1   2020-01-02
        2   2020-01-03
        dtype: datetime64[us]
        """
        self._check_copy_deprecation(copy)
        if is_dict_like(dtype):
            if self.ndim == 1:  # i.e. Series
                if len(dtype) > 1 or self.name not in dtype:
                    raise KeyError(
                        "Only the Series name can be used for "
                        "the key in Series dtype mappings."
                    )
                new_type = dtype[self.name]
                return self.astype(new_type, errors=errors)

            # GH#44417 cast to Series so we can use .iat below, which will be
            #  robust in case we
            from pandas import Series

            dtype_ser = Series(dtype, dtype=object)

            for col_name in dtype_ser.index:
                if col_name not in self:
                    raise KeyError(
                        "Only a column name can be used for the "
                        "key in a dtype mappings argument. "
                        f"'{col_name}' not found in columns."
                    )

            dtype_ser = dtype_ser.reindex(self.columns, fill_value=None)

            results = []
            for i, (col_name, col) in enumerate(self.items()):
                cdt = dtype_ser.iat[i]
                if isna(cdt):
                    res_col = col.copy(deep=False)
                else:
                    try:
                        res_col = col.astype(dtype=cdt, errors=errors)
                    except ValueError as ex:
                        ex.args = (
                            f"{ex}: Error while type casting for column '{col_name}'",
                        )
                        raise
                results.append(res_col)

        elif is_extension_array_dtype(dtype) and self.ndim > 1:
            # TODO(EA2D): special case not needed with 2D EAs
            dtype = pandas_dtype(dtype)
            if isinstance(dtype, ExtensionDtype) and all(
                block.values.dtype == dtype for block in self._mgr.blocks
            ):
                return self.copy(deep=False)
            # GH 18099/22869: columnwise conversion to extension dtype
            # GH 24704: self.items handles duplicate column names
            results = [ser.astype(dtype, errors=errors) for _, ser in self.items()]

        else:
            # else, only a single dtype is given
            new_data = self._mgr.astype(dtype=dtype, errors=errors)
            res = self._constructor_from_mgr(new_data, axes=new_data.axes)
            return res.__finalize__(self, method="astype")

        # GH 33113: handle empty frame or series
        if not results:
            return self.copy(deep=False)

        # GH 19920: retain column metadata after concat
        result = concat(results, axis=1)
        # GH#40810 retain subclass
        # error: Incompatible types in assignment
        # (expression has type "Self", variable has type "DataFrame")
        result = self._constructor(result)  # type: ignore[assignment]
        result.columns = self.columns
        result = result.__finalize__(self, method="astype")
        # https://github.com/python/mypy/issues/8354
        return cast(Self, result)

    @final
    def copy(self, deep: bool = True) -> Self:
        # [FUNCTION DOC]
        # Return a copy of the object.
        # Parameters: deep (default true) says whether data are copied; under
        # copy-on-write semantics a shallow copy shares memory lazily and a deep copy
        # is eager. Returns a new object, finalized from self with the method name
        # copy.
        """
        Make a copy of this object's indices and data.

        When ``deep=True`` (default), a new object will be created with a
        copy of the calling object's data and indices. Modifications to
        the data or indices of the copy will not be reflected in the
        original object (see notes below).

        When ``deep=False``, a new object will be created without copying
        the calling object's data or index (only references to the data
        and index are copied). With Copy-on-Write, changes to the original
        will *not* be reflected in the shallow copy (and vice versa). The
        shallow copy uses a lazy (deferred) copy mechanism that copies the
        data only when any changes to the original or shallow copy are made,
        ensuring memory efficiency while maintaining data integrity.

        .. note::
            In pandas versions prior to 3.0, the default behavior without
            Copy-on-Write was different: changes to the original *were* reflected
            in the shallow copy (and vice versa). See the :ref:`Copy-on-Write
            user guide <copy_on_write>` for more information.

        Parameters
        ----------
        deep : bool, default True
            Make a deep copy, including a copy of the data and the indices.
            With ``deep=False`` neither the indices nor the data are copied.

        Returns
        -------
        Series or DataFrame
            Object type matches caller.

        See Also
        --------
        copy.copy : Return a shallow copy of an object.
        copy.deepcopy : Return a deep copy of an object.

        Notes
        -----
        When ``deep=True``, data is copied but actual Python objects
        will not be copied recursively, only the reference to the object.
        This is in contrast to `copy.deepcopy` in the Standard Library,
        which recursively copies object data (see examples below).

        While ``Index`` objects are copied when ``deep=True``, the underlying
        numpy array is not copied for performance reasons. Since ``Index`` is
        immutable, the underlying data can be safely shared and a copy
        is not needed.

        Since pandas is not thread safe, see the
        :ref:`gotchas <gotchas.thread-safety>` when copying in a threading
        environment.

        Copy-on-Write protects shallow copies against accidental modifications.
        This means that any changes to the copied data would make a new copy
        of the data upon write (and vice versa). Changes made to either the
        original or copied variable would not be reflected in the counterpart.
        See :ref:`Copy_on_Write <copy_on_write>` for more information.

        Examples
        --------
        >>> s = pd.Series([1, 2], index=["a", "b"])
        >>> s
        a    1
        b    2
        dtype: int64

        >>> s_copy = s.copy(deep=True)
        >>> s_copy
        a    1
        b    2
        dtype: int64

        Due to Copy-on-Write, shallow copies still protect data modifications.
        Note shallow does not get modified below.

        >>> s = pd.Series([1, 2], index=["a", "b"])
        >>> shallow = s.copy(deep=False)
        >>> s.iloc[1] = 200
        >>> shallow
        a    1
        b    2
        dtype: int64

        When the data has object dtype, even a deep copy does not copy the
        underlying Python objects. Updating a nested data object will be
        reflected in the deep copy.

        >>> s = pd.Series([[1, 2], [3, 4]])
        >>> deep = s.copy()
        >>> s[0][0] = 10
        >>> s
        0    [10, 2]
        1     [3, 4]
        dtype: object
        >>> deep
        0    [10, 2]
        1     [3, 4]
        dtype: object
        """
        data = self._mgr.copy(deep=deep)
        return self._constructor_from_mgr(data, axes=data.axes).__finalize__(
            self, method="copy"
        )

    @final
    def __copy__(self) -> Self:
        # [FUNCTION DOC]
        # Implements the standard library shallow copy protocol by returning a shallow
        # pandas copy.
        return self.copy(deep=False)

    @final
    def __deepcopy__(self, memo=None) -> Self:
        # [FUNCTION DOC]
        # Implements the standard library deep copy protocol.
        # Parameters: memo is the standard memo dictionary, ignored. Returns a deep
        # pandas copy.
        """
        Parameters
        ----------
        memo, default None
            Standard signature. Unused
        """
        return self.copy(deep=True)

    @final
    def infer_objects(self, copy: bool | lib.NoDefault = lib.no_default) -> Self:
        # [FUNCTION DOC]
        # Attempt to infer better dtypes for object columns.
        # Parameters: copy is a deprecated placeholder. Converts object blocks through
        # the manager conversion (soft conversion that tries to infer numeric,
        # datetime and similar dtypes) and returns a new object finalized with the
        # method name infer_objects. Non-object columns are left unchanged.
        """
        Attempt to infer better dtypes for object columns.

        Attempts soft conversion of object-dtyped
        columns, leaving non-object and unconvertible
        columns unchanged. The inference rules are the
        same as during normal Series/DataFrame construction.

        Parameters
        ----------
        copy : bool, default False
            This keyword is now ignored; changing its value will have no
            impact on the method.

            .. deprecated:: 3.0.0

                This keyword is ignored and will be removed in pandas 4.0. Since
                pandas 3.0, this method always returns a new object using a lazy
                copy mechanism that defers copies until necessary
                (Copy-on-Write). See the `user guide on Copy-on-Write
                <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                for more details.

        Returns
        -------
        same type as input object
            Returns an object of the same type as the input object.

        See Also
        --------
        to_datetime : Convert argument to datetime.
        to_timedelta : Convert argument to timedelta.
        to_numeric : Convert argument to numeric type.
        convert_dtypes : Convert argument to best possible dtype.

        Examples
        --------
        >>> df = pd.DataFrame({"A": ["a", 1, 2, 3]})
        >>> df = df.iloc[1:]
        >>> df
           A
        1  1
        2  2
        3  3

        >>> df.dtypes
        A    object
        dtype: object

        >>> df.infer_objects().dtypes
        A    int64
        dtype: object
        """
        self._check_copy_deprecation(copy)
        new_mgr = self._mgr.convert()
        res = self._constructor_from_mgr(new_mgr, axes=new_mgr.axes)
        return res.__finalize__(self, method="infer_objects")

    @final
    def convert_dtypes(
        self,
        infer_objects: bool = True,
        convert_string: bool = True,
        convert_integer: bool = True,
        convert_boolean: bool = True,
        convert_floating: bool = True,
        dtype_backend: DtypeBackend = "numpy_nullable",
    ) -> Self:
        # [FUNCTION DOC]
        # Convert to dtypes that support missing values natively.
        # Parameters: infer_objects (default true) first infer dtypes of object
        # columns; convert_string, convert_integer, convert_boolean, convert_floating
        # (all default true) enable conversion to the string, integer, boolean and
        # floating extension dtypes respectively; dtype_backend selects numpy_nullable
        # (default) or pyarrow flavored dtypes and is validated by the shared helper
        # (an invalid value raises a value error).
        # The manager does the conversion block by block. Returns a new object
        # finalized with the method name convert_dtypes.
        """
        Convert columns from numpy dtypes to the best dtypes that support ``pd.NA``.

        Parameters
        ----------
        infer_objects : bool, default True
            Whether object dtypes should be converted to the best possible types.
        convert_string : bool, default True
            Whether object dtypes should be converted to ``StringDtype()``.
        convert_integer : bool, default True
            Whether, if possible, conversion can be done to integer extension types.
        convert_boolean : bool, defaults True
            Whether object dtypes should be converted to ``BooleanDtypes()``.
        convert_floating : bool, defaults True
            Whether, if possible, conversion can be done to floating extension types.
            If `convert_integer` is also True, preference will be give to integer
            dtypes if the floats can be faithfully casted to integers.
        dtype_backend : {'numpy_nullable', 'pyarrow'}, default 'numpy_nullable'
            Back-end data type applied to the resultant :class:`DataFrame` or
            :class:`Series` (still experimental). Behaviour is as follows:

            * ``"numpy_nullable"``: returns nullable-dtype-backed
              :class:`DataFrame` or :class:`Serires`.
            * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
              :class:`DataFrame` or :class:`Series`.

            .. versionadded:: 2.0

        Returns
        -------
        Series or DataFrame
            Copy of input object with new dtype.

        See Also
        --------
        infer_objects : Infer dtypes of objects.
        to_datetime : Convert argument to datetime.
        to_timedelta : Convert argument to timedelta.
        to_numeric : Convert argument to a numeric type.

        Notes
        -----
        By default, ``convert_dtypes`` will attempt to convert a Series (or each
        Series in a DataFrame) to dtypes that support ``pd.NA``. By using the options
        ``convert_string``, ``convert_integer``, ``convert_boolean`` and
        ``convert_floating``, it is possible to turn off individual conversions
        to ``StringDtype``, the integer extension types, ``BooleanDtype``
        or floating extension types, respectively.

        For object-dtyped columns, if ``infer_objects`` is ``True``, use the inference
        rules as during normal Series/DataFrame construction.  Then, if possible,
        convert to ``StringDtype``, ``BooleanDtype`` or an appropriate integer
        or floating extension type, otherwise leave as ``object``.

        If the dtype is integer, convert to an appropriate integer extension type.

        If the dtype is numeric, and consists of all integers, convert to an
        appropriate integer extension type. Otherwise, convert to an
        appropriate floating extension type.

        In the future, as new dtypes are added that support ``pd.NA``, the results
        of this method will change to support those new dtypes.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "a": pd.Series([1, 2, 3], dtype=np.dtype("int32")),
        ...         "b": pd.Series(["x", "y", "z"], dtype=np.dtype("O")),
        ...         "c": pd.Series([True, False, np.nan], dtype=np.dtype("O")),
        ...         "d": pd.Series(["h", "i", np.nan], dtype=np.dtype("O")),
        ...         "e": pd.Series([10, np.nan, 20], dtype=np.dtype("float")),
        ...         "f": pd.Series([np.nan, 100.5, 200], dtype=np.dtype("float")),
        ...     }
        ... )

        Start with a DataFrame with default dtypes.

        >>> df
           a  b      c    d     e      f
        0  1  x   True    h  10.0    NaN
        1  2  y  False    i   NaN  100.5
        2  3  z    NaN  NaN  20.0  200.0

        >>> df.dtypes
        a      int32
        b     object
        c     object
        d     object
        e    float64
        f    float64
        dtype: object

        Convert the DataFrame to use best possible dtypes.

        >>> dfn = df.convert_dtypes()
        >>> dfn
           a  b      c     d     e      f
        0  1  x   True     h    10   <NA>
        1  2  y  False     i  <NA>  100.5
        2  3  z   <NA>  <NA>    20  200.0

        >>> dfn.dtypes
        a      Int32
        b     string
        c    boolean
        d     string
        e      Int64
        f    Float64
        dtype: object

        Start with a Series of strings and missing data represented by ``np.nan``.

        >>> s = pd.Series(["a", "b", np.nan])
        >>> s
        0      a
        1      b
        2    NaN
        dtype: str

        Obtain a Series with dtype ``StringDtype``.

        >>> s.convert_dtypes()
        0       a
        1       b
        2    <NA>
        dtype: string
        """
        check_dtype_backend(dtype_backend)
        new_mgr = self._mgr.convert_dtypes(
            infer_objects=infer_objects,
            convert_string=convert_string,
            convert_integer=convert_integer,
            convert_boolean=convert_boolean,
            convert_floating=convert_floating,
            dtype_backend=dtype_backend,
        )
        res = self._constructor_from_mgr(new_mgr, axes=new_mgr.axes)
        return res.__finalize__(self, method="convert_dtypes")

    # ----------------------------------------------------------------------
    # Filling NA's

    @final
    def _pad_or_backfill(
        self,
        method: Literal["ffill", "bfill", "pad", "backfill"],
        *,
        axis: None | Axis = None,
        inplace: bool = False,
        limit: None | int = None,
        limit_area: Literal["inside", "outside"] | None = None,
    ):
        # [FUNCTION DOC]
        # HELPER: shared implementation of forward and backward fill.
        # Parameters (keyword only except method): method is ffill, bfill, pad or
        # backfill (normalized by the missing-data helper); axis (default 0, None
        # treated as 0); inplace; limit maximum number of consecutive fills;
        # limit_area restricts filling to inside or outside the valid-value range.
        # Axis one: raises a not-implemented error if inplace is requested on a
        # multi-block object; otherwise the transpose is filled along its rows and
        # transposed back.
        # KNOWN ISSUE: on the axis one path the inplace flag is never forwarded to the
        # transposed call, so even for a single-block object a new object is returned
        # and self is not modified, and the result is not finalized.
        # Axis zero: the manager fills (in place if requested); if inplace the
        # internals of self are replaced and self is returned, otherwise the new
        # object is returned after finalizing.
        # KNOWN ISSUE: the finalize call uses the method name fillna for ffill and
        # bfill as well, so the recorded operation name is not the actual one.
        if axis is None:
            axis = 0
        axis = self._get_axis_number(axis)
        method = clean_fill_method(method)

        if axis == 1:
            if not self._mgr.is_single_block and inplace:
                raise NotImplementedError
            # e.g. test_align_fill_method
            result = self.T._pad_or_backfill(
                method=method, limit=limit, limit_area=limit_area
            ).T

            return result

        new_mgr = self._mgr.pad_or_backfill(
            method=method,
            limit=limit,
            limit_area=limit_area,
            inplace=inplace,
        )
        result = self._constructor_from_mgr(new_mgr, axes=new_mgr.axes)
        if inplace:
            self._update_inplace(result)
            return self
        else:
            return result.__finalize__(self, method="fillna")

    @final
    def fillna(
        self,
        value: Hashable | Mapping | Series | DataFrame,
        *,
        axis: Axis | None = None,
        inplace: bool = False,
        limit: int | None = None,
    ) -> Self:
        # [FUNCTION DOC]
        # Fill missing values with a given value.
        # Parameters (keyword only except value): value is a scalar, dict, Series (for
        # frames also a DataFrame) giving the replacements; axis (default 0) matters
        # for dict/Series values on frames and for scalar fill; inplace; limit is the
        # maximum number of fills per column (or consecutive positions for the value,
        # depending on manager).
        # When inplace is true and the chained-assignment warning has not been
        # disabled globally, the reference count of the object is inspected and, if it
        # indicates the call is made on a temporary (a chained assignment) rather than
        # a local variable of the caller, a chained-assignment warning is emitted at
        # stack level two.
        # Raises a type error for list or tuple values (and, for a Series, for any
        # list-like value other than dict or Series). Raises a value error for other
        # invalid fill value types, and, for dict or Series values on a frame with
        # axis one, if the columns have more than one dtype or the value cannot be
        # held by the frame dtype.
        # Series: dict or Series values are aligned to the index (an empty one is a
        # no-op copy); the manager performs the fill. Frame with dict or Series value:
        # columns are filled one by one (keys not present are skipped); in non-inplace
        # mode the column is replaced; in-place mode writes back into the existing
        # column when the dtype is unchanged (handling duplicate column names through
        # positional location), otherwise replaces the column; returns the result
        # directly. Frame with scalar: axis one fills the transpose; otherwise the
        # manager fills. Frame with a DataFrame value: a conditional replacement keeps
        # original non-missing values and takes the value elsewhere.
        # Finally the manager result is wrapped; in-place replaces the internals of
        # self and returns self; otherwise returns the new finalized object (method
        # name fillna).
        """
        Fill NA/NaN values with `value`.

        Parameters
        ----------
        value : scalar, dict, Series, or DataFrame
            Value to use to fill holes (e.g. 0), alternately a
            dict/Series/DataFrame of values specifying which value to use for
            each index (for a Series) or column (for a DataFrame).  Values not
            in the dict/Series/DataFrame will not be filled. This value cannot
            be a list.
        axis : {0 or 'index'} for Series, {0 or 'index', 1 or 'columns'} for DataFrame
            Axis along which to fill missing values. For `Series`
            this parameter is unused and defaults to 0.
        inplace : bool, default False
            If True, fill in-place. Note: this will modify any
            other views on this object (e.g., a no-copy slice for a column in a
            DataFrame).
        limit : int, default None
            This is the maximum number of entries along the entire axis
            where NaNs will be filled. Must be greater than 0 if not None.

        Returns
        -------
        Series/DataFrame
            Object with missing values filled.

        See Also
        --------
        ffill : Fill values by propagating the last valid observation to next valid.
        bfill : Fill values by using the next valid observation to fill the gap.
        interpolate : Fill NaN values using interpolation.
        reindex : Conform object to new index.
        asfreq : Convert TimeSeries to specified frequency.

        Notes
        -----
        For non-object dtype, ``value=None`` will use the NA value of the dtype.
        See more details in the :ref:`Filling missing data<missing_data.fillna>`
        section.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     [
        ...         [np.nan, 2, np.nan, 0],
        ...         [3, 4, np.nan, 1],
        ...         [np.nan, np.nan, np.nan, np.nan],
        ...         [np.nan, 3, np.nan, 4],
        ...     ],
        ...     columns=list("ABCD"),
        ... )
        >>> df
             A    B   C    D
        0  NaN  2.0 NaN  0.0
        1  3.0  4.0 NaN  1.0
        2  NaN  NaN NaN  NaN
        3  NaN  3.0 NaN  4.0

        Replace all NaN elements with 0s.

        >>> df.fillna(0)
             A    B    C    D
        0  0.0  2.0  0.0  0.0
        1  3.0  4.0  0.0  1.0
        2  0.0  0.0  0.0  0.0
        3  0.0  3.0  0.0  4.0

        Replace all NaN elements in column 'A', 'B', 'C', and 'D', with 0, 1,
        2, and 3 respectively.

        >>> values = {"A": 0, "B": 1, "C": 2, "D": 3}
        >>> df.fillna(value=values)
             A    B    C    D
        0  0.0  2.0  2.0  0.0
        1  3.0  4.0  2.0  1.0
        2  0.0  1.0  2.0  3.0
        3  0.0  3.0  2.0  4.0

        Only replace the first NaN element.

        >>> df.fillna(value=values, limit=1)
             A    B    C    D
        0  0.0  2.0  2.0  0.0
        1  3.0  4.0  NaN  1.0
        2  NaN  1.0  NaN  3.0
        3  NaN  3.0  NaN  4.0

        When filling using a DataFrame, replacement happens along
        the same column names and same indices

        >>> df2 = pd.DataFrame(np.zeros((4, 4)), columns=list("ABCE"))
        >>> df.fillna(df2)
             A    B    C    D
        0  0.0  2.0  0.0  0.0
        1  3.0  4.0  0.0  1.0
        2  0.0  0.0  0.0  NaN
        3  0.0  3.0  0.0  4.0

        Note that column D is not affected since it is not present in df2.
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        if inplace:
            if not CHAINED_WARNING_DISABLED:
                if sys.getrefcount(
                    self
                ) <= REF_COUNT_METHOD and not common.is_local_in_caller_frame(self):
                    warnings.warn(
                        _chained_assignment_method_msg,
                        ChainedAssignmentError,
                        stacklevel=2,
                    )

        if isinstance(value, (list, tuple)):
            raise TypeError(
                '"value" parameter must be a scalar or dict, but '
                f'you passed a "{type(value).__name__}"'
            )

        # set the default here, so functions examining the signature
        # can detect if something was set (e.g. in groupby) (GH9221)
        if axis is None:
            axis = 0
        axis = self._get_axis_number(axis)

        if self.ndim == 1:
            if isinstance(value, (dict, ABCSeries)):
                if not len(value):
                    # test_fillna_nonscalar
                    return self if inplace else self.copy(deep=False)
                from pandas import Series

                value = Series(value)
                value = value.reindex(self.index)
                value = value._values
            elif not is_list_like(value):
                pass
            else:
                raise TypeError(
                    '"value" parameter must be a scalar, dict '
                    "or Series, but you passed a "
                    f'"{type(value).__name__}"'
                )

            new_data = self._mgr.fillna(value=value, limit=limit, inplace=inplace)

        elif isinstance(value, (dict, ABCSeries)):
            result = self if inplace else self.copy(deep=False)
            if axis == 1:
                # Check that all columns in result have the same dtype
                # otherwise don't bother with fillna and losing accurate dtypes
                unique_dtypes = self._mgr.get_unique_dtypes()
                if len(unique_dtypes) > 1:
                    raise ValueError(
                        "All columns must have the same dtype, but got dtypes: "
                        f"{list(unique_dtypes)}"
                    )
                # Use the first column, which we have already validated has the
                # same dtypes as the other columns.
                if not can_hold_element(result.iloc[:, 0], value):
                    frame_dtype = unique_dtypes.item()
                    raise ValueError(
                        f"{value} not a suitable type to fill into {frame_dtype}"
                    )
                result = result.T.fillna(value=value).T
                if inplace:
                    self._update_inplace(result)
                    result = self
            else:
                for k, v in value.items():
                    if k not in result:
                        continue

                    res_k = result[k].fillna(v, limit=limit)

                    if not inplace:
                        result[k] = res_k
                    # We can write into our existing column(s) iff dtype
                    #  was preserved.
                    elif isinstance(res_k, ABCSeries):
                        # i.e. 'k' only shows up once in self.columns
                        if res_k.dtype == result[k].dtype:
                            result.loc[:, k] = res_k
                        else:
                            # Different dtype -> no way to do inplace.
                            result[k] = res_k
                    else:
                        # see test_fillna_dict_inplace_nonunique_columns
                        locs = result.columns.get_loc(k)
                        if isinstance(locs, slice):
                            locs = range(self.shape[1])[locs]
                        elif isinstance(locs, np.ndarray) and locs.dtype.kind == "b":
                            locs = locs.nonzero()[0]
                        elif not (
                            isinstance(locs, np.ndarray) and locs.dtype.kind == "i"
                        ):
                            # Should never be reached, but let's cover our bases
                            raise NotImplementedError(
                                "Unexpected get_loc result, please report a bug at "
                                "https://github.com/pandas-dev/pandas"
                            )

                        for i, loc in enumerate(locs):
                            res_loc = res_k.iloc[:, i]
                            target = self.iloc[:, loc]

                            if res_loc.dtype == target.dtype:
                                result.iloc[:, loc] = res_loc
                            else:
                                result.isetitem(loc, res_loc)
            return result

        elif not is_list_like(value):
            if axis == 1:
                result = self.T.fillna(value=value, limit=limit).T
                new_data = result._mgr
            else:
                new_data = self._mgr.fillna(value=value, limit=limit, inplace=inplace)
        elif isinstance(value, ABCDataFrame) and self.ndim == 2:
            new_data = self.where(self.notna(), value)._mgr
        else:
            raise ValueError(f"invalid fill value with a {type(value)}")

        result = self._constructor_from_mgr(new_data, axes=new_data.axes)
        if inplace:
            self._update_inplace(result)
            return self
        else:
            return result.__finalize__(self, method="fillna")

    @final
    def ffill(
        self,
        *,
        axis: None | Axis = None,
        inplace: bool = False,
        limit: None | int = None,
        limit_area: Literal["inside", "outside"] | None = None,
    ) -> Self:
        # [FUNCTION DOC]
        # Fill missing values by propagating the last valid observation forward.
        # Parameters (keyword only): axis (default None means 0); inplace; limit
        # maximum consecutive fills; limit_area inside or outside restriction.
        # When inplace is true and the chained-assignment warning has not been
        # disabled globally, the reference count of the object is inspected and, if it
        # indicates the call is made on a temporary (a chained assignment) rather than
        # a local variable of the caller, a chained-assignment warning is emitted at
        # stack level two. Delegates to the private pad-or-backfill helper with the
        # forward method. Returns the filled object, or self when in-place (see the
        # helper for the exception on axis one).
        """
        Fill NA/NaN values by propagating the last valid observation to next valid.

        Parameters
        ----------
        axis : {0 or 'index'} for Series, {0 or 'index', 1 or 'columns'} for DataFrame
            Axis along which to fill missing values. For `Series`
            this parameter is unused and defaults to 0.
        inplace : bool, default False
            If True, fill in-place. Note: this will modify any
            other views on this object (e.g., a no-copy slice for a column in a
            DataFrame).
        limit : int, default None
            If method is specified, this is the maximum number of consecutive
            NaN values to forward/backward fill. In other words, if there is
            a gap with more than this number of consecutive NaNs, it will only
            be partially filled. If method is not specified, this is the
            maximum number of entries along the entire axis where NaNs will be
            filled. Must be greater than 0 if not None.
        limit_area : {`None`, 'inside', 'outside'}, default None
            If limit is specified, consecutive NaNs will be filled with this
            restriction.

            * ``None``: No fill restriction.
            * 'inside': Only fill NaNs surrounded by valid values
              (interpolate).
            * 'outside': Only fill NaNs outside valid values (extrapolate).

            .. versionadded:: 2.2.0

        Returns
        -------
        Series/DataFrame
            Object with missing values filled.

        See Also
        --------
        DataFrame.bfill : Fill NA/NaN values by using the next valid observation
            to fill the gap.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     [
        ...         [np.nan, 2, np.nan, 0],
        ...         [3, 4, np.nan, 1],
        ...         [np.nan, np.nan, np.nan, np.nan],
        ...         [np.nan, 3, np.nan, 4],
        ...     ],
        ...     columns=list("ABCD"),
        ... )
        >>> df
             A    B   C    D
        0  NaN  2.0 NaN  0.0
        1  3.0  4.0 NaN  1.0
        2  NaN  NaN NaN  NaN
        3  NaN  3.0 NaN  4.0

        >>> df.ffill()
             A    B   C    D
        0  NaN  2.0 NaN  0.0
        1  3.0  4.0 NaN  1.0
        2  3.0  4.0 NaN  1.0
        3  3.0  3.0 NaN  4.0

        >>> ser = pd.Series([1, np.nan, 2, 3])
        >>> ser.ffill()
        0   1.0
        1   1.0
        2   2.0
        3   3.0
        dtype: float64
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        if inplace:
            if not CHAINED_WARNING_DISABLED:
                if sys.getrefcount(
                    self
                ) <= REF_COUNT_METHOD and not common.is_local_in_caller_frame(self):
                    warnings.warn(
                        _chained_assignment_method_msg,
                        ChainedAssignmentError,
                        stacklevel=2,
                    )

        return self._pad_or_backfill(
            "ffill",
            axis=axis,
            inplace=inplace,
            limit=limit,
            limit_area=limit_area,
        )

    @final
    def bfill(
        self,
        *,
        axis: None | Axis = None,
        inplace: bool = False,
        limit: None | int = None,
        limit_area: Literal["inside", "outside"] | None = None,
    ) -> Self:
        # [FUNCTION DOC]
        # Fill missing values using the next valid observation (backward fill).
        # Parameters (keyword only): axis (default None means 0); inplace; limit;
        # limit_area as for forward fill.
        # When inplace is true and the chained-assignment warning has not been
        # disabled globally, the reference count of the object is inspected and, if it
        # indicates the call is made on a temporary (a chained assignment) rather than
        # a local variable of the caller, a chained-assignment warning is emitted at
        # stack level two. Delegates to the private pad-or-backfill helper with the
        # backward method. Returns the filled object.
        """
        Fill NA/NaN values by using the next valid observation to fill the gap.

        This method fills missing values in a backward direction along the
        specified axis, propagating non-null values from later positions to
        earlier positions containing NaN.

        Parameters
        ----------
        axis : {0 or 'index'} for Series, {0 or 'index', 1 or 'columns'} for DataFrame
            Axis along which to fill missing values. For `Series`
            this parameter is unused and defaults to 0.
        inplace : bool, default False
            If True, fill in-place. Note: this will modify any
            other views on this object (e.g., a no-copy slice for a column in a
            DataFrame).
        limit : int, default None
            If method is specified, this is the maximum number of consecutive
            NaN values to forward/backward fill. In other words, if there is
            a gap with more than this number of consecutive NaNs, it will only
            be partially filled. If method is not specified, this is the
            maximum number of entries along the entire axis where NaNs will be
            filled. Must be greater than 0 if not None.
        limit_area : {`None`, 'inside', 'outside'}, default None
            If limit is specified, consecutive NaNs will be filled with this
            restriction.

            * ``None``: No fill restriction.
            * 'inside': Only fill NaNs surrounded by valid values
              (interpolate).
            * 'outside': Only fill NaNs outside valid values (extrapolate).

            .. versionadded:: 2.2.0

        Returns
        -------
        Series/DataFrame
            Object with missing values filled.

        See Also
        --------
        DataFrame.ffill : Fill NA/NaN values by propagating the last valid
            observation to next valid.

        Examples
        --------
        For Series:

        >>> s = pd.Series([1, None, None, 2])
        >>> s.bfill()
        0    1.0
        1    2.0
        2    2.0
        3    2.0
        dtype: float64
        >>> s.bfill(limit=1)
        0    1.0
        1    NaN
        2    2.0
        3    2.0
        dtype: float64

        With DataFrame:

        >>> df = pd.DataFrame({"A": [1, None, None, 4], "B": [None, 5, None, 7]})
        >>> df
              A     B
        0   1.0	  NaN
        1   NaN	  5.0
        2   NaN   NaN
        3   4.0   7.0
        >>> df.bfill()
              A     B
        0   1.0   5.0
        1   4.0   5.0
        2   4.0   7.0
        3   4.0   7.0
        >>> df.bfill(limit=1)
              A     B
        0   1.0   5.0
        1   NaN   5.0
        2   4.0   7.0
        3   4.0   7.0
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        if inplace:
            if not CHAINED_WARNING_DISABLED:
                if sys.getrefcount(
                    self
                ) <= REF_COUNT_METHOD and not common.is_local_in_caller_frame(self):
                    warnings.warn(
                        _chained_assignment_method_msg,
                        ChainedAssignmentError,
                        stacklevel=2,
                    )

        return self._pad_or_backfill(
            "bfill",
            axis=axis,
            inplace=inplace,
            limit=limit,
            limit_area=limit_area,
        )

    @final
    def replace(
        self,
        to_replace=None,
        value=lib.no_default,
        *,
        inplace: bool = False,
        regex: bool = False,
    ) -> Self:
        # [FUNCTION DOC]
        # Replace values with other values.
        # Parameters: to_replace is a scalar, list-like, dict-like, regular expression
        # string or None; value is the replacement (scalar, list-like or dict-like),
        # left unset when a mapping carries the values; keyword only: inplace; regex
        # says to interpret string to_replace values as regular expressions (a bool,
        # or, when to_replace is None, a pattern/list/dict).
        # Validation: a value error if regex is not a bool while to_replace is not
        # None; a type error if to_replace is not a scalar,
        # regular-expression-compilable object or list-like; a value error if value is
        # unset and neither to_replace nor regex is dict-like. The chained-assignment
        # check applies when inplace.
        # Unset value: a dict-like to_replace (or the regex dict, which is moved into
        # to_replace with regex enabled) is split into keys and values; a nested
        # mapping (all top-level values must be mappings, otherwise a type error)
        # becomes per-column replacement lists; the method then calls itself with
        # these.
        # Value given: empty objects are returned as copies. A dict-like to_replace
        # with dict-like value maps per column (value error for a Series; only columns
        # present in both and in the object are processed); with scalar value is
        # applied column-wise (value error for Series); list-like value raises a type
        # error. A list-like to_replace requires an equal-length list-like value (a
        # scalar value is broadcast) or raises a value error, and the manager replaces
        # the list. A None to_replace requires a valid regex and delegates with the
        # regex used as to_replace. A scalar to_replace with dict value acts per
        # column (value error for Series). A scalar value uses regex or plain
        # replacement according to whether regex use is warranted. Any other
        # combination raises a type error.
        # Returns a new finalized object (method name replace) or, when inplace, self.
        """
        Replace values given in `to_replace` with `value`.

        Values of the Series/DataFrame are replaced with other values dynamically.
        This differs from updating with ``.loc`` or ``.iloc``, which require
        you to specify a location to update with some value.

        Parameters
        ----------
        to_replace : str, regex, list, dict, Series, int, float, or None
            How to find the values that will be replaced.

            * numeric, str or regex:

                - numeric: numeric values equal to `to_replace` will be
                  replaced with `value`
                - str: string exactly matching `to_replace` will be replaced
                  with `value`
                - regex: regexes matching `to_replace` will be replaced with
                  `value`

            * list of str, regex, or numeric:

                - First, if `to_replace` and `value` are both lists, they
                  **must** be the same length.
                - Second, if ``regex=True`` then all of the strings in **both**
                  lists will be interpreted as regexes otherwise they will match
                  directly. This doesn't matter much for `value` since there
                  are only a few possible substitution regexes you can use.
                - str, regex and numeric rules apply as above.

            * dict:

                - Dicts can be used to specify different replacement values
                  for different existing values. For example,
                  ``{'a': 'b', 'y': 'z'}`` replaces the value 'a' with 'b' and
                  'y' with 'z'. To use a dict in this way, the optional `value`
                  parameter should not be given.
                - For a DataFrame a dict can specify that different values
                  should be replaced in different columns. For example,
                  ``{'a': 1, 'b': 'z'}`` looks for the value 1 in column 'a'
                  and the value 'z' in column 'b' and replaces these values
                  with whatever is specified in `value`. The `value` parameter
                  should not be ``None`` in this case. You can treat this as a
                  special case of passing two lists except that you are
                  specifying the column to search in.
                - For a DataFrame nested dictionaries, e.g.,
                  ``{'a': {'b': np.nan}}``, are read as follows: look in column
                  'a' for the value 'b' and replace it with NaN. The optional `value`
                  parameter should not be specified to use a nested dict in this
                  way. You can nest regular expressions as well. Note that
                  column names (the top-level dictionary keys in a nested
                  dictionary) **cannot** be regular expressions.

            * None:

                - This means that the `regex` argument must be a string,
                  compiled regular expression, or list, dict, ndarray or
                  Series of such elements. If `value` is also ``None`` then
                  this **must** be a nested dictionary or Series.

            See the examples section for examples of each of these.
        value : scalar, dict, list, str, regex, default None
            Value to replace any values matching `to_replace` with.
            For a DataFrame a dict of values can be used to specify which
            value to use for each column (columns not in the dict will not be
            filled). Regular expressions, strings and lists or dicts of such
            objects are also allowed.

        inplace : bool, default False
            If True, performs operation inplace.
        regex : bool or same types as `to_replace`, default False
            Whether to interpret `to_replace` and/or `value` as regular
            expressions. Alternatively, this could be a regular expression or a
            list, dict, or array of regular expressions in which case
            `to_replace` must be ``None``.

        Returns
        -------
        Series/DataFrame
            Object after replacement.

        Raises
        ------
        AssertionError
            * If `regex` is not a ``bool`` and `to_replace` is not
              ``None``.

        TypeError
            * If `to_replace` is not a scalar, array-like, ``dict``, or ``None``
            * If `to_replace` is a ``dict`` and `value` is not a ``list``,
              ``dict``, ``ndarray``, or ``Series``
            * If `to_replace` is ``None`` and `regex` is not compilable
              into a regular expression or is a list, dict, ndarray, or
              Series.
            * When replacing multiple ``bool`` or ``datetime64`` objects and
              the arguments to `to_replace` does not match the type of the
              value being replaced

        ValueError
            * If a ``list`` or an ``ndarray`` is passed to `to_replace` and
              `value` but they are not the same length.

        See Also
        --------
        Series.fillna : Fill NA values.
        DataFrame.fillna : Fill NA values.
        Series.where : Replace values based on boolean condition.
        DataFrame.where : Replace values based on boolean condition.
        DataFrame.map: Apply a function to a Dataframe elementwise.
        Series.map: Map values of Series according to an input mapping or function.
        Series.str.replace : Simple string replacement.

        Notes
        -----
        * Regex substitution is performed under the hood with ``re.sub``. The
          rules for substitution for ``re.sub`` are the same.
        * Regular expressions will only substitute on strings, meaning you
          cannot provide, for example, a regular expression matching floating
          point numbers and expect the columns in your frame that have a
          numeric dtype to be matched. However, if those floating point
          numbers *are* strings, then you can do this.
        * This method has *a lot* of options. You are encouraged to experiment
          and play with this method to gain intuition about how it works.
        * When dict is used as the `to_replace` value, it is like
          key(s) in the dict are the to_replace part and
          value(s) in the dict are the value parameter.

        Examples
        --------

        **Scalar `to_replace` and `value`**

        >>> s = pd.Series([1, 2, 3, 4, 5])
        >>> s.replace(1, 5)
        0    5
        1    2
        2    3
        3    4
        4    5
        dtype: int64

        >>> df = pd.DataFrame(
        ...     {
        ...         "A": [0, 1, 2, 3, 4],
        ...         "B": [5, 6, 7, 8, 9],
        ...         "C": ["a", "b", "c", "d", "e"],
        ...     }
        ... )
        >>> df.replace(0, 5)
            A  B  C
        0  5  5  a
        1  1  6  b
        2  2  7  c
        3  3  8  d
        4  4  9  e

        **List-like `to_replace`**

        >>> df.replace([0, 1, 2, 3], 4)
            A  B  C
        0  4  5  a
        1  4  6  b
        2  4  7  c
        3  4  8  d
        4  4  9  e

        >>> df.replace([0, 1, 2, 3], [4, 3, 2, 1])
            A  B  C
        0  4  5  a
        1  3  6  b
        2  2  7  c
        3  1  8  d
        4  4  9  e

        **dict-like `to_replace`**

        >>> df.replace({0: 10, 1: 100})
                A  B  C
        0   10  5  a
        1  100  6  b
        2    2  7  c
        3    3  8  d
        4    4  9  e

        >>> df.replace({"A": 0, "B": 5}, 100)
                A    B  C
        0  100  100  a
        1    1    6  b
        2    2    7  c
        3    3    8  d
        4    4    9  e

        >>> df.replace({"A": {0: 100, 4: 400}})
                A  B  C
        0  100  5  a
        1    1  6  b
        2    2  7  c
        3    3  8  d
        4  400  9  e

        **Regular expression `to_replace`**

        >>> df = pd.DataFrame({"A": ["bat", "foo", "bait"], "B": ["abc", "bar", "xyz"]})
        >>> df.replace(to_replace=r"^ba.$", value="new", regex=True)
                A    B
        0   new  abc
        1   foo  new
        2  bait  xyz

        >>> df.replace({"A": r"^ba.$"}, {"A": "new"}, regex=True)
                A    B
        0   new  abc
        1   foo  bar
        2  bait  xyz

        >>> df.replace(regex=r"^ba.$", value="new")
                A    B
        0   new  abc
        1   foo  new
        2  bait  xyz

        >>> df.replace(regex={r"^ba.$": "new", "foo": "xyz"})
                A    B
        0   new  abc
        1   xyz  new
        2  bait  xyz

        >>> df.replace(regex=[r"^ba.$", "foo"], value="new")
                A    B
        0   new  abc
        1   new  new
        2  bait  xyz

        Compare the behavior of ``s.replace({'a': None})`` and
        ``s.replace('a', None)`` to understand the peculiarities
        of the `to_replace` parameter:

        >>> s = pd.Series([10, "a", "a", "b", "a"])

        When one uses a dict as the `to_replace` value, it is like the
        value(s) in the dict are equal to the `value` parameter.
        ``s.replace({'a': None})`` is equivalent to
        ``s.replace(to_replace={'a': None}, value=None)``:

        >>> s.replace({"a": None})
        0      10
        1    None
        2    None
        3       b
        4    None
        dtype: object

        If ``None`` is explicitly passed for ``value``, it will be respected:

        >>> s.replace("a", None)
        0      10
        1    None
        2    None
        3       b
        4    None
        dtype: object

        When ``regex=True``, ``value`` is not ``None`` and `to_replace` is a string,
        the replacement will be applied in all columns of the DataFrame.

        >>> df = pd.DataFrame(
        ...     {
        ...         "A": [0, 1, 2, 3, 4],
        ...         "B": ["a", "b", "c", "d", "e"],
        ...         "C": ["f", "g", "h", "i", "j"],
        ...     }
        ... )

        >>> df.replace(to_replace="^[a-g]", value="e", regex=True)
            A  B  C
        0  0  e  e
        1  1  e  e
        2  2  e  h
        3  3  e  i
        4  4  e  j

        If ``value`` is not ``None`` and `to_replace` is a dictionary, the dictionary
        keys will be the DataFrame columns that the replacement will be applied.

        >>> df.replace(to_replace={"B": "^[a-c]", "C": "^[h-j]"}, value="e", regex=True)
            A  B  C
        0  0  e  f
        1  1  e  g
        2  2  e  e
        3  3  d  e
        4  4  e  e
        """
        if not is_bool(regex) and to_replace is not None:
            raise ValueError("'to_replace' must be 'None' if 'regex' is not a bool")

        if not (
            is_scalar(to_replace)
            or is_re_compilable(to_replace)
            or is_list_like(to_replace)
        ):
            raise TypeError(
                "Expecting 'to_replace' to be either a scalar, array-like, "
                "dict or None, got invalid type "
                f"{type(to_replace).__name__!r}"
            )

        if value is lib.no_default and not (
            is_dict_like(to_replace) or is_dict_like(regex)
        ):
            raise ValueError(
                # GH#33302
                f"{type(self).__name__}.replace must specify either 'value', "
                "a dict-like 'to_replace', or dict-like 'regex'."
            )

        inplace = validate_bool_kwarg(inplace, "inplace")
        if inplace:
            if not CHAINED_WARNING_DISABLED:
                if sys.getrefcount(
                    self
                ) <= REF_COUNT_METHOD and not common.is_local_in_caller_frame(self):
                    warnings.warn(
                        _chained_assignment_method_msg,
                        ChainedAssignmentError,
                        stacklevel=2,
                    )

        if value is lib.no_default:
            if not is_dict_like(to_replace):
                # In this case we have checked above that
                #  1) regex is dict-like and 2) to_replace is None
                to_replace = regex
                regex = True

            items = list(to_replace.items())
            if items:
                keys, values = zip(*items, strict=True)
            else:
                keys, values = ([], [])  # type: ignore[assignment]

            are_mappings = [is_dict_like(v) for v in values]

            if any(are_mappings):
                if not all(are_mappings):
                    raise TypeError(
                        "If a nested mapping is passed, all values "
                        "of the top level mapping must be mappings"
                    )
                # passed a nested dict/Series
                to_rep_dict = {}
                value_dict = {}

                for k, v in items:
                    # error: Incompatible types in assignment (expression has type
                    # "list[Never]", variable has type "tuple[Any, ...]")
                    keys, values = list(zip(*v.items(), strict=True)) or (  # type: ignore[assignment]
                        [],
                        [],
                    )

                    to_rep_dict[k] = list(keys)
                    value_dict[k] = list(values)

                to_replace, value = to_rep_dict, value_dict
            else:
                to_replace, value = keys, values

            return self.replace(to_replace, value, inplace=inplace, regex=regex)
        else:
            # need a non-zero len on all axes
            if not self.size:
                return self if inplace else self.copy(deep=False)
            if is_dict_like(to_replace):
                if is_dict_like(value):  # {'A' : NA} -> {'A' : 0}
                    if isinstance(self, ABCSeries):
                        raise ValueError(
                            "to_replace and value cannot be dict-like for "
                            "Series.replace"
                        )
                    # Note: Checking below for `in foo.keys()` instead of
                    #  `in foo` is needed for when we have a Series and not dict
                    mapping = {
                        col: (to_replace[col], value[col])
                        for col in to_replace.keys()
                        if col in value.keys() and col in self
                    }
                    return self._replace_columnwise(mapping, inplace, regex)

                # {'A': NA} -> 0
                elif not is_list_like(value):
                    # Operate column-wise
                    if self.ndim == 1:
                        raise ValueError(
                            "Series.replace cannot specify both a dict-like "
                            "'to_replace' and a 'value'"
                        )
                    mapping = {
                        col: (to_rep, value) for col, to_rep in to_replace.items()
                    }
                    return self._replace_columnwise(mapping, inplace, regex)
                else:
                    raise TypeError("value argument must be scalar, dict, or Series")

            elif is_list_like(to_replace):
                if not is_list_like(value):
                    # e.g. to_replace = [NA, ''] and value is 0,
                    #  so we replace NA with 0 and then replace '' with 0
                    value = [value] * len(to_replace)

                # e.g. we have to_replace = [NA, ''] and value = [0, 'missing']
                if len(to_replace) != len(value):
                    raise ValueError(
                        f"Replacement lists must match in length. "
                        f"Expecting {len(to_replace)} got {len(value)} "
                    )
                new_data = self._mgr.replace_list(
                    src_list=to_replace,
                    dest_list=value,
                    inplace=inplace,
                    regex=regex,
                )

            elif to_replace is None:
                if not (
                    is_re_compilable(regex)
                    or is_list_like(regex)
                    or is_dict_like(regex)
                ):
                    raise TypeError(
                        f"'regex' must be a string or a compiled regular expression "
                        f"or a list or dict of strings or regular expressions, "
                        f"you passed a {type(regex).__name__!r}"
                    )
                return self.replace(regex, value, inplace=inplace, regex=True)
            # dest iterable dict-like
            elif is_dict_like(value):  # NA -> {'A' : 0, 'B' : -1}
                # Operate column-wise
                if self.ndim == 1:
                    raise ValueError(
                        "Series.replace cannot use dict-value and non-None to_replace"
                    )
                mapping = {col: (to_replace, val) for col, val in value.items()}
                return self._replace_columnwise(mapping, inplace, regex)

            elif not is_list_like(value):  # NA -> 0
                regex = should_use_regex(regex, to_replace)
                if regex:
                    new_data = self._mgr.replace_regex(
                        to_replace=to_replace,
                        value=value,
                        inplace=inplace,
                    )
                else:
                    new_data = self._mgr.replace(
                        to_replace=to_replace, value=value, inplace=inplace
                    )
            else:
                raise TypeError(
                    f'Invalid "to_replace" type: {type(to_replace).__name__!r}'
                )

        result = self._constructor_from_mgr(new_data, axes=new_data.axes)
        if inplace:
            self._update_inplace(result)
            return self
        else:
            return result.__finalize__(self, method="replace")

    @final
    def interpolate(
        self,
        method: InterpolateOptions = "linear",
        *,
        axis: Axis = 0,
        limit: int | None = None,
        inplace: bool = False,
        limit_direction: Literal["forward", "backward", "both"] | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        **kwargs,
    ) -> Self:
        # [FUNCTION DOC]
        # Fill NaN values using an interpolation method.
        # Parameters: method names the technique (linear by default and many others);
        # keyword only: axis (default 0); limit maximum consecutive fills; inplace;
        # limit_direction forward, backward or both (default inferred from the
        # method); limit_area inside or outside; extra keyword arguments are passed to
        # the underlying method.
        # When inplace is true and the chained-assignment warning has not been
        # disabled globally, the reference count of the object is inspected and, if it
        # indicates the call is made on a temporary (a chained assignment) rather than
        # a local variable of the caller, a chained-assignment warning is emitted at
        # stack level two.
        # Empty objects are returned (self when inplace, otherwise a copy). Raises a
        # value error when method is not a string, and when the index is a MultiIndex
        # and the method is not linear. For axis one the transpose is interpolated and
        # transposed back. The direction default is inferred from the method, and the
        # interpolation index is derived from the method and the index. The manager
        # interpolates block by block. In-place replaces the internals and returns
        # self; otherwise returns the new object finalized with the method name
        # interpolate.
        """
        Fill NaN values using an interpolation method.

        Please note that only ``method='linear'`` is supported for
        DataFrame/Series with a MultiIndex.

        Parameters
        ----------
        method : str, default 'linear'
            Interpolation technique to use. One of:

            * 'linear': Ignore the index and treat the values as equally
              spaced. This is the only method supported on MultiIndexes.
            * 'time': Works on daily and higher resolution data to interpolate
              given length of interval. This interpolates values based on
              time interval between observations.
            * 'index': The interpolation uses the numerical values
              of the DataFrame's index to linearly calculate missing values.
            * 'values': Interpolation based on the numerical values
              in the DataFrame, treating them as equally spaced along the index.
            * 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
              'barycentric', 'polynomial': Passed to
              `scipy.interpolate.interp1d`, whereas 'spline' is passed to
              `scipy.interpolate.UnivariateSpline`. These methods use the numerical
              values of the index.  Both 'polynomial' and 'spline' require that
              you also specify an `order` (int), e.g.
              ``df.interpolate(method='polynomial', order=5)``. Note that,
              `slinear` method in Pandas refers to the Scipy first order `spline`
              instead of Pandas first order `spline`.
            * 'krogh', 'piecewise_polynomial', 'spline', 'pchip', 'akima',
              'cubicspline': Wrappers around the SciPy interpolation methods of
              similar names. See `Notes`.
            * 'from_derivatives': Refers to
              `scipy.interpolate.BPoly.from_derivatives`.

        axis : {0 or 'index', 1 or 'columns', None}, default None
            Axis to interpolate along. For `Series` this parameter is unused
            and defaults to 0.
        limit : int, optional
            Maximum number of consecutive NaNs to fill. Must be greater than
            0.
        inplace : bool, default False
            Update the data in place if possible.
        limit_direction : {'forward', 'backward', 'both'}, optional, default 'forward'
            Consecutive NaNs will be filled in this direction.

        limit_area : {`None`, 'inside', 'outside'}, default None
            If limit is specified, consecutive NaNs will be filled with this
            restriction.

            * ``None``: No fill restriction.
            * 'inside': Only fill NaNs surrounded by valid values
              (interpolate).
            * 'outside': Only fill NaNs outside valid values (extrapolate).

        **kwargs : optional
            Keyword arguments to pass on to the interpolating function.

        Returns
        -------
        Series or DataFrame
            Returns the same object type as the caller, interpolated at
            some or all ``NaN`` values.

        See Also
        --------
        fillna : Fill missing values using different methods.
        scipy.interpolate.Akima1DInterpolator : Piecewise cubic polynomials
            (Akima interpolator).
        scipy.interpolate.BPoly.from_derivatives : Piecewise polynomial in the
            Bernstein basis.
        scipy.interpolate.interp1d : Interpolate a 1-D function.
        scipy.interpolate.KroghInterpolator : Interpolate polynomial (Krogh
            interpolator).
        scipy.interpolate.PchipInterpolator : PCHIP 1-d monotonic cubic
            interpolation.
        scipy.interpolate.CubicSpline : Cubic spline data interpolator.

        Notes
        -----
        The 'krogh', 'piecewise_polynomial', 'spline', 'pchip' and 'akima'
        methods are wrappers around the respective SciPy implementations of
        similar names. These use the actual numerical values of the index.
        For more information on their behavior, see the
        `SciPy documentation
        <https://docs.scipy.org/doc/scipy/reference/interpolate.html#univariate-interpolation>`__.

        Examples
        --------
        Filling in ``NaN`` in a :class:`~pandas.Series` via linear
        interpolation.

        >>> s = pd.Series([0, 1, np.nan, 3])
        >>> s
        0    0.0
        1    1.0
        2    NaN
        3    3.0
        dtype: float64
        >>> s.interpolate()
        0    0.0
        1    1.0
        2    2.0
        3    3.0
        dtype: float64

        Filling in ``NaN`` in a Series via polynomial interpolation or splines:
        Both 'polynomial' and 'spline' methods require that you also specify
        an ``order`` (int).

        >>> s = pd.Series([0, 2, np.nan, 8])
        >>> s.interpolate(method="polynomial", order=2)
        0    0.000000
        1    2.000000
        2    4.666667
        3    8.000000
        dtype: float64

        Fill the DataFrame forward (that is, going down) along each column
        using linear interpolation.

        Note how the last entry in column 'a' is interpolated differently,
        because there is no entry after it to use for interpolation.
        Note how the first entry in column 'b' remains ``NaN``, because there
        is no entry before it to use for interpolation.

        >>> df = pd.DataFrame(
        ...     [
        ...         (0.0, np.nan, -1.0, 1.0),
        ...         (np.nan, 2.0, np.nan, np.nan),
        ...         (2.0, 3.0, np.nan, 9.0),
        ...         (np.nan, 4.0, -4.0, 16.0),
        ...     ],
        ...     columns=list("abcd"),
        ... )
        >>> df
             a    b    c     d
        0  0.0  NaN -1.0   1.0
        1  NaN  2.0  NaN   NaN
        2  2.0  3.0  NaN   9.0
        3  NaN  4.0 -4.0  16.0
        >>> df.interpolate(method="linear", limit_direction="forward", axis=0)
             a    b    c     d
        0  0.0  NaN -1.0   1.0
        1  1.0  2.0 -2.0   5.0
        2  2.0  3.0 -3.0   9.0
        3  2.0  4.0 -4.0  16.0

        Using polynomial interpolation.

        >>> df["d"].interpolate(method="polynomial", order=2)
        0     1.0
        1     4.0
        2     9.0
        3    16.0
        Name: d, dtype: float64
        """
        inplace = validate_bool_kwarg(inplace, "inplace")

        if inplace:
            if not CHAINED_WARNING_DISABLED:
                if sys.getrefcount(
                    self
                ) <= REF_COUNT_METHOD and not common.is_local_in_caller_frame(self):
                    warnings.warn(
                        _chained_assignment_method_msg,
                        ChainedAssignmentError,
                        stacklevel=2,
                    )

        axis = self._get_axis_number(axis)

        if self.empty:
            return self if inplace else self.copy()

        if not isinstance(method, str):
            raise ValueError("'method' should be a string, not None.")

        obj, should_transpose = (self.T, True) if axis == 1 else (self, False)

        if isinstance(obj.index, MultiIndex) and method != "linear":
            raise ValueError(
                "Only `method=linear` interpolation is supported on MultiIndexes."
            )

        limit_direction = missing.infer_limit_direction(limit_direction, method)

        index = missing.get_interp_index(method, obj.index)
        new_data = obj._mgr.interpolate(
            method=method,
            index=index,
            limit=limit,
            limit_direction=limit_direction,
            limit_area=limit_area,
            inplace=inplace,
            **kwargs,
        )

        result = self._constructor_from_mgr(new_data, axes=new_data.axes)
        if should_transpose:
            result = result.T
        if inplace:
            self._update_inplace(result)
            return self
        else:
            return result.__finalize__(self, method="interpolate")

    # ----------------------------------------------------------------------
    # Timeseries methods Methods

    @final
    def asof(self, where, subset=None):
        # [FUNCTION DOC]
        # Return the last row(s) with no missing values at or before the requested
        # label(s).
        # Parameters: where is a label, Timestamp-like string (converted to a
        # Timestamp) or list of labels; subset (frames only) restricts which columns
        # must be non-missing (default all).
        # Raises a value error if the index is not monotonic increasing, or if subset
        # is given for a Series. A scalar before the first index label yields missing
        # data (nan for a Series, an all-missing float Series for a frame). For a
        # PeriodIndex a scalar is converted to a Period with the index frequency.
        # A scalar on a Series uses a backwards scan from the right-most position not
        # after the label to the first non-missing value. Otherwise the labels become
        # an Index, a null mask is computed (any missing within subset for frames),
        # and if everything is missing the result is built entirely of missing values
        # with the matching shape. Otherwise the index finds the positions of the last
        # valid labels, the rows are taken, labelled with the requested labels, and
        # positions with no valid predecessor are set to missing. Returns the whole
        # result for list input, otherwise the last (single) row/element.
        """
        Return the last row(s) without any NaNs before `where`.

        The last row (for each element in `where`, if list) without any
        NaN is taken.
        In case of a :class:`~pandas.DataFrame`, the last row without NaN
        considering only the subset of columns (if not `None`)

        If there is no good value, NaN is returned for a Series or
        a Series of NaN values for a DataFrame

        Parameters
        ----------
        where : date or array-like of dates
            Date(s) before which the last row(s) are returned.
        subset : str or array-like of str, default `None`
            For DataFrame, if not `None`, only use these columns to
            check for NaNs.

        Returns
        -------
        scalar, Series, or DataFrame

            The return can be:

            * scalar : when `self` is a Series and `where` is a scalar
            * Series: when `self` is a Series and `where` is an array-like,
              or when `self` is a DataFrame and `where` is a scalar
            * DataFrame : when `self` is a DataFrame and `where` is an
              array-like

        See Also
        --------
        merge_asof : Perform an asof merge. Similar to left join.

        Notes
        -----
        Dates are assumed to be sorted. Raises if this is not the case.

        Examples
        --------
        A Series and a scalar `where`.

        >>> s = pd.Series([1, 2, np.nan, 4], index=[10, 20, 30, 40])
        >>> s
        10    1.0
        20    2.0
        30    NaN
        40    4.0
        dtype: float64

        >>> s.asof(20)
        np.float64(2.0)

        For a sequence `where`, a Series is returned. The first value is
        NaN, because the first element of `where` is before the first
        index value.

        >>> s.asof([5, 20])
        5     NaN
        20    2.0
        dtype: float64

        Missing values are not considered. The following is ``2.0``, not
        NaN, even though NaN is at the index location for ``30``.

        >>> s.asof(30)
        np.float64(2.0)

        Take all columns into consideration

        >>> df = pd.DataFrame(
        ...     {
        ...         "a": [10.0, 20.0, 30.0, 40.0, 50.0],
        ...         "b": [None, None, None, None, 500],
        ...     },
        ...     index=pd.DatetimeIndex(
        ...         [
        ...             "2018-02-27 09:01:00",
        ...             "2018-02-27 09:02:00",
        ...             "2018-02-27 09:03:00",
        ...             "2018-02-27 09:04:00",
        ...             "2018-02-27 09:05:00",
        ...         ]
        ...     ),
        ... )
        >>> df.asof(pd.DatetimeIndex(["2018-02-27 09:03:30", "2018-02-27 09:04:30"]))
                              a   b
        2018-02-27 09:03:30 NaN NaN
        2018-02-27 09:04:30 NaN NaN

        Take a single column into consideration

        >>> df.asof(
        ...     pd.DatetimeIndex(["2018-02-27 09:03:30", "2018-02-27 09:04:30"]),
        ...     subset=["a"],
        ... )
                                a   b
        2018-02-27 09:03:30  30.0 NaN
        2018-02-27 09:04:30  40.0 NaN
        """
        if isinstance(where, str):
            where = Timestamp(where)

        if not self.index.is_monotonic_increasing:
            raise ValueError("asof requires a sorted index")

        is_series = isinstance(self, ABCSeries)
        if is_series:
            if subset is not None:
                raise ValueError("subset is not valid for Series")
        else:
            if subset is None:
                subset = self.columns
            if not is_list_like(subset):
                subset = [subset]

        is_list = is_list_like(where)
        if not is_list:
            start = self.index[0]
            if isinstance(self.index, PeriodIndex):
                where = Period(where, freq=self.index.freq)

            if where < start:
                if not is_series:
                    return self._constructor_sliced(
                        index=self.columns, name=where, dtype=np.float64
                    )
                return np.nan

            # It's always much faster to use a *while* loop here for
            # Series than pre-computing all the NAs. However a
            # *while* loop is extremely expensive for DataFrame
            # so we later pre-compute all the NAs and use the same
            # code path whether *where* is a scalar or list.
            # See PR: https://github.com/pandas-dev/pandas/pull/14476
            if is_series:
                loc = self.index.searchsorted(where, side="right")
                if loc > 0:
                    loc -= 1

                values = self._values
                while loc > 0 and isna(values[loc]):
                    loc -= 1
                return values[loc]

        if not isinstance(where, Index):
            where = Index(where) if is_list else Index([where])

        nulls = self.isna() if is_series else self[subset].isna().any(axis=1)
        if nulls.all():
            if is_series:
                self = cast("Series", self)
                return self._constructor(np.nan, index=where, name=self.name)
            elif is_list:
                self = cast("DataFrame", self)
                return self._constructor(np.nan, index=where, columns=self.columns)
            else:
                self = cast("DataFrame", self)
                return self._constructor_sliced(
                    np.nan, index=self.columns, name=where[0]
                )

        # error: Unsupported operand type for
        # ~ ("ExtensionArray | ndarray[Any, Any] | Any")
        locs = self.index.asof_locs(where, ~nulls._values)  # type: ignore[operator]

        # mask the missing
        mask = locs == -1
        data = self.take(locs)
        data.index = where
        if mask.any():
            # GH#16063 only do this setting when necessary, otherwise
            #  we'd cast e.g. bools to floats
            data.loc[mask] = np.nan
        return data if is_list else data.iloc[-1]

    # ----------------------------------------------------------------------
    # Action Methods

    def isna(self) -> Self:
        # [FUNCTION DOC]
        # Detect missing values. Returns an object of the same shape of booleans (true
        # where missing), finalized with the method name isna.
        """
        Detect missing values.

        Return a boolean same-sized object indicating if the values are NA.
        NA values, such as None or :attr:`numpy.NaN`, gets mapped to True
        values.
        Everything else gets mapped to False values. Characters such as empty
        strings ``''`` or :attr:`numpy.inf` are not considered NA values.

        Returns
        -------
        Series/DataFrame
            Mask of bool values for each element in Series/DataFrame
            that indicates whether an element is an NA value.

        See Also
        --------
        Series.isnull : Alias of isna.
        DataFrame.isnull : Alias of isna.
        Series.notna : Boolean inverse of isna.
        DataFrame.notna : Boolean inverse of isna.
        Series.dropna : Omit axes labels with missing values.
        DataFrame.dropna : Omit axes labels with missing values.
        isna : Top-level isna.

        Examples
        --------
        Show which entries in a DataFrame are NA.

        >>> df = pd.DataFrame(
        ...     dict(
        ...         age=[5, 6, np.nan],
        ...         born=[
        ...             pd.NaT,
        ...             pd.Timestamp("1939-05-27"),
        ...             pd.Timestamp("1940-04-25"),
        ...         ],
        ...         name=["Alfred", "Batman", ""],
        ...         toy=[None, "Batmobile", "Joker"],
        ...     )
        ... )
        >>> df
           age       born    name        toy
        0  5.0        NaT  Alfred        NaN
        1  6.0 1939-05-27  Batman  Batmobile
        2  NaN 1940-04-25              Joker

        >>> df.isna()
             age   born   name    toy
        0  False   True  False   True
        1  False  False  False  False
        2   True  False  False  False

        Show which entries in a Series are NA.

        >>> ser = pd.Series([5, 6, np.nan])
        >>> ser
        0    5.0
        1    6.0
        2    NaN
        dtype: float64

        >>> ser.isna()
        0    False
        1    False
        2     True
        dtype: bool
        """
        return isna(self).__finalize__(self, method="isna")

    def isnull(self) -> Self:
        # [FUNCTION DOC]
        # Alias of the missing-value detector. Returns booleans (true where missing),
        # finalized with the method name isnull.
        """
        Detect missing values.

        Return a boolean same-sized object indicating if the values are NA.
        NA values, such as None or :attr:`numpy.NaN`, gets mapped to True
        values.
        Everything else gets mapped to False values. Characters such as empty
        strings ``''`` or :attr:`numpy.inf` are not considered NA values.

        Returns
        -------
        Series/DataFrame
            Mask of bool values for each element in Series/DataFrame
            that indicates whether an element is an NA value.

        See Also
        --------
        Series.isna : Alias of isnull.
        DataFrame.isna : Alias of isnull.
        Series.notna : Boolean inverse of isnull.
        DataFrame.notna : Boolean inverse of isnull.
        Series.dropna : Omit axes labels with missing values.
        DataFrame.dropna : Omit axes labels with missing values.
        isna : Top-level isna.

        Examples
        --------
        Show which entries in a DataFrame are NA.

        >>> df = pd.DataFrame(
        ...     dict(
        ...         age=[5, 6, np.nan],
        ...         born=[
        ...             pd.NaT,
        ...             pd.Timestamp("1939-05-27"),
        ...             pd.Timestamp("1940-04-25"),
        ...         ],
        ...         name=["Alfred", "Batman", ""],
        ...         toy=[None, "Batmobile", "Joker"],
        ...     )
        ... )
        >>> df
           age       born    name        toy
        0  5.0        NaT  Alfred        NaN
        1  6.0 1939-05-27  Batman  Batmobile
        2  NaN 1940-04-25              Joker

        >>> df.isna()
             age   born   name    toy
        0  False   True  False   True
        1  False  False  False  False
        2   True  False  False  False

        Show which entries in a Series are NA.

        >>> ser = pd.Series([5, 6, np.nan])
        >>> ser
        0    5.0
        1    6.0
        2    NaN
        dtype: float64

        >>> ser.isna()
        0    False
        1    False
        2     True
        dtype: bool
        """
        return isna(self).__finalize__(self, method="isnull")

    def notna(self) -> Self:
        # [FUNCTION DOC]
        # Detect non-missing values. Returns an object of the same shape of booleans
        # (true where not missing), finalized with the method name notna.
        """
        Detect existing (non-missing) values.

        Return a boolean same-sized object indicating if the values are not NA.
        Non-missing values get mapped to True. Characters such as empty
        strings ``''`` or :attr:`numpy.inf` are not considered NA values.
        NA values, such as None or :attr:`numpy.NaN`, get mapped to False
        values.

        Returns
        -------
        Series/DataFrame
            Mask of bool values for each element in Series/DataFrame
            that indicates whether an element is not an NA value.

        See Also
        --------
        Series.notnull : Alias of notna.
        DataFrame.notnull : Alias of notna.
        Series.isna : Boolean inverse of notna.
        DataFrame.isna : Boolean inverse of notna.
        Series.dropna : Omit axes labels with missing values.
        DataFrame.dropna : Omit axes labels with missing values.
        notna : Top-level notna.

        Examples
        --------
        Show which entries in a DataFrame are not NA.

        >>> df = pd.DataFrame(
        ...     dict(
        ...         age=[5, 6, np.nan],
        ...         born=[
        ...             pd.NaT,
        ...             pd.Timestamp("1939-05-27"),
        ...             pd.Timestamp("1940-04-25"),
        ...         ],
        ...         name=["Alfred", "Batman", ""],
        ...         toy=[None, "Batmobile", "Joker"],
        ...     )
        ... )
        >>> df
           age       born    name        toy
        0  5.0        NaT  Alfred        NaN
        1  6.0 1939-05-27  Batman  Batmobile
        2  NaN 1940-04-25              Joker

        >>> df.notna()
             age   born  name    toy
        0   True  False  True  False
        1   True   True  True   True
        2  False   True  True   True

        Show which entries in a Series are not NA.

        >>> ser = pd.Series([5, 6, np.nan])
        >>> ser
        0    5.0
        1    6.0
        2    NaN
        dtype: float64

        >>> ser.notna()
        0     True
        1     True
        2    False
        dtype: bool
        """
        return notna(self).__finalize__(self, method="notna")

    def notnull(self) -> Self:
        # [FUNCTION DOC]
        # Alias of the non-missing detector. Returns booleans (true where not
        # missing), finalized with the method name notnull.
        """
        Detect existing (non-missing) values.

        Return a boolean same-sized object indicating if the values are not NA.
        Non-missing values get mapped to True. Characters such as empty
        strings ``''`` or :attr:`numpy.inf` are not considered NA values.
        NA values, such as None or :attr:`numpy.NaN`, get mapped to False
        values.

        Returns
        -------
        Series/DataFrame
            Mask of bool values for each element in Series/DataFrame
            that indicates whether an element is not an NA value.

        See Also
        --------
        Series.notnull : Alias of notna.
        DataFrame.notnull : Alias of notna.
        Series.isna : Boolean inverse of notna.
        DataFrame.isna : Boolean inverse of notna.
        Series.dropna : Omit axes labels with missing values.
        DataFrame.dropna : Omit axes labels with missing values.
        notna : Top-level notna.

        Examples
        --------
        Show which entries in a DataFrame are not NA.

        >>> df = pd.DataFrame(
        ...     dict(
        ...         age=[5, 6, np.nan],
        ...         born=[
        ...             pd.NaT,
        ...             pd.Timestamp("1939-05-27"),
        ...             pd.Timestamp("1940-04-25"),
        ...         ],
        ...         name=["Alfred", "Batman", ""],
        ...         toy=[None, "Batmobile", "Joker"],
        ...     )
        ... )
        >>> df
           age       born    name        toy
        0  5.0        NaT  Alfred        NaN
        1  6.0 1939-05-27  Batman  Batmobile
        2  NaN 1940-04-25              Joker

        >>> df.notna()
             age   born  name    toy
        0   True  False  True  False
        1   True   True  True   True
        2  False   True  True   True

        Show which entries in a Series are not NA.

        >>> ser = pd.Series([5, 6, np.nan])
        >>> ser
        0    5.0
        1    6.0
        2    NaN
        dtype: float64

        >>> ser.notna()
        0     True
        1     True
        2    False
        dtype: bool
        """
        return notna(self).__finalize__(self, method="notnull")

    @final
    def _clip_with_scalar(self, lower, upper, inplace: bool = False):
        # [FUNCTION DOC]
        # HELPER: clip to scalar bounds.
        # Parameters: lower and upper are scalar thresholds or None; inplace.
        # Raises a value error if either bound is a missing value. Builds a mask of
        # existing missing entries; for each bound present, values violating it are
        # replaced by the bound using the conditional-replacement method while missing
        # entries are preserved. Returns the resulting object (possibly self when
        # inplace).
        if (lower is not None and np.any(isna(lower))) or (
            upper is not None and np.any(isna(upper))
        ):
            raise ValueError("Cannot use an NA value as a clip threshold")

        result = self
        mask = self.isna()

        if lower is not None:
            cond = mask | (self >= lower)
            result = result.where(cond, lower, inplace=inplace)
        if upper is not None:
            cond = mask | (self <= upper)
            result = result.where(cond, upper, inplace=inplace)

        return result

    @final
    def _clip_with_one_bound(self, threshold, method, axis, inplace):
        # [FUNCTION DOC]
        # HELPER: clip with a single bound that may be a scalar, array-like or labeled
        # object.
        # Parameters: threshold; method is the comparison method of the object
        # (less-or-equal for an upper bound, greater-or-equal for a lower bound;
        # identified through its name); axis; inplace.
        # Numeric scalars delegate to the scalar helper. List-likes that are not
        # Series are converted into a labeled object (a Series with the same index, or
        # by alignment along the axis for a frame). Missing threshold entries are
        # replaced by plus or minus infinity so they do not clip. Values already
        # satisfying the comparison, plus missing values, are kept; other values are
        # replaced by the threshold via conditional replacement. Returns the resulting
        # object.
        if axis is not None:
            axis = self._get_axis_number(axis)

        # method is self.le for upper bound and self.ge for lower bound
        if is_scalar(threshold) and is_number(threshold):
            if method.__name__ == "le":
                return self._clip_with_scalar(None, threshold, inplace=inplace)
            return self._clip_with_scalar(threshold, None, inplace=inplace)

        # GH #15390
        # In order for where method to work, the threshold must
        # be transformed to NDFrame from other array like structure.
        if (not isinstance(threshold, ABCSeries)) and is_list_like(threshold):
            if isinstance(self, ABCSeries):
                threshold = self._constructor(threshold, index=self.index)
            else:
                threshold = self._align_for_op(threshold, axis, flex=None)[1]

        # GH 40420
        # Treat missing thresholds as no bounds, not clipping the values
        if is_list_like(threshold):
            fill_value = np.inf if method.__name__ == "le" else -np.inf
            threshold_inf = threshold.fillna(fill_value)
        else:
            threshold_inf = threshold

        subset = method(threshold_inf, axis=axis) | isna(self)

        # GH 40420
        return self.where(subset, threshold, axis=axis, inplace=inplace)

    @final
    def clip(
        self,
        lower=None,
        upper=None,
        *,
        axis: Axis | None = None,
        inplace: bool = False,
        **kwargs,
    ) -> Self:
        # [FUNCTION DOC]
        # Trim values at the given thresholds.
        # Parameters: lower and upper are scalars, array-likes or None (missing bounds
        # are treated as no bound; a list-like bound is ignored only if all its
        # entries are missing); keyword only: axis along which to align array-like
        # bounds; inplace; extra keyword arguments exist for numpy compatibility and
        # are validated (the validation also may supply an axis).
        # When inplace is true and the chained-assignment warning has not been
        # disabled globally, the reference count of the object is inspected and, if it
        # indicates the call is made on a temporary (a chained assignment) rather than
        # a local variable of the caller, a chained-assignment warning is emitted at
        # stack level two.
        # If both bounds are scalars and reversed they are swapped. If both bounds are
        # None or numeric the scalar fast path is used. Otherwise lower and upper are
        # applied one after the other via the single-bound helper (in place the second
        # step uses self). Returns the clipped object, or the object itself when
        # inplace.
        """
        Trim values at input threshold(s).

        Assigns values outside boundary to boundary values. Thresholds
        can be singular values or array like, and in the latter case
        the clipping is performed element-wise in the specified axis.

        Parameters
        ----------
        lower : float or array-like, default None
            Minimum threshold value. All values below this
            threshold will be set to it. A missing
            threshold (e.g `NA`) will not clip the value.
        upper : float or array-like, default None
            Maximum threshold value. All values above this
            threshold will be set to it. A missing
            threshold (e.g `NA`) will not clip the value.
        axis : {0 or 'index', 1 or 'columns', None}, default None
            Align object with lower and upper along the given axis.
            For `Series` this parameter is unused and defaults to `None`.
        inplace : bool, default False
            Whether to perform the operation in place on the data.
        **kwargs
            Additional keywords have no effect but might be accepted
            for compatibility with numpy.

        Returns
        -------
        Series or DataFrame
            Same type as calling object with the values outside the
            clip boundaries replaced.

        See Also
        --------
        Series.clip : Trim values at input threshold in series.
        DataFrame.clip : Trim values at input threshold in DataFrame.
        numpy.clip : Clip (limit) the values in an array.

        Examples
        --------
        >>> data = {"col_0": [9, -3, 0, -1, 5], "col_1": [-2, -7, 6, 8, -5]}
        >>> df = pd.DataFrame(data)
        >>> df
           col_0  col_1
        0      9     -2
        1     -3     -7
        2      0      6
        3     -1      8
        4      5     -5

        Clips per column using lower and upper thresholds:

        >>> df.clip(-4, 6)
           col_0  col_1
        0      6     -2
        1     -3     -4
        2      0      6
        3     -1      6
        4      5     -4

        Clips using specific lower and upper thresholds per column:

        >>> df.clip([-2, -1], [4, 5])
            col_0  col_1
        0      4     -1
        1     -2     -1
        2      0      5
        3     -1      5
        4      4     -1

        Clips using specific lower and upper thresholds per column element:

        >>> t = pd.Series([2, -4, -1, 6, 3])
        >>> t
        0    2
        1   -4
        2   -1
        3    6
        4    3
        dtype: int64

        >>> df.clip(t, t + 4, axis=0)
           col_0  col_1
        0      6      2
        1     -3     -4
        2      0      3
        3      6      8
        4      5      3

        Clips using specific lower threshold per column element, with missing values:

        >>> t = pd.Series([2, -4, np.nan, 6, 3])
        >>> t
        0    2.0
        1   -4.0
        2    NaN
        3    6.0
        4    3.0
        dtype: float64

        >>> df.clip(t, axis=0)
        col_0  col_1
        0      9.0    2.0
        1     -3.0   -4.0
        2      0.0    6.0
        3      6.0    8.0
        4      5.0    3.0
        """
        inplace = validate_bool_kwarg(inplace, "inplace")

        if inplace:
            if not CHAINED_WARNING_DISABLED:
                if sys.getrefcount(
                    self
                ) <= REF_COUNT_METHOD and not common.is_local_in_caller_frame(self):
                    warnings.warn(
                        _chained_assignment_method_msg,
                        ChainedAssignmentError,
                        stacklevel=2,
                    )

        axis = nv.validate_clip_with_axis(axis, (), kwargs)
        if axis is not None:
            axis = self._get_axis_number(axis)

        # GH 17276
        # numpy doesn't like NaN as a clip value
        # so ignore
        # GH 19992
        # numpy doesn't drop a list-like bound containing NaN
        isna_lower = isna(lower)
        if not is_list_like(lower):
            if np.any(isna_lower):
                lower = None
        elif np.all(isna_lower):
            lower = None
        isna_upper = isna(upper)
        if not is_list_like(upper):
            if np.any(isna_upper):
                upper = None
        elif np.all(isna_upper):
            upper = None

        # GH 2747 (arguments were reversed)
        if (
            lower is not None
            and upper is not None
            and is_scalar(lower)
            and is_scalar(upper)
        ):
            lower, upper = min(lower, upper), max(lower, upper)

        # fast-path for scalars
        if (lower is None or is_number(lower)) and (upper is None or is_number(upper)):
            return self._clip_with_scalar(lower, upper, inplace=inplace)

        result = self
        if lower is not None:
            result = result._clip_with_one_bound(
                lower, method=self.ge, axis=axis, inplace=inplace
            )
        if upper is not None:
            if inplace:
                result = self
            result = result._clip_with_one_bound(
                upper, method=self.le, axis=axis, inplace=inplace
            )

        return result

    @final
    def asfreq(
        self,
        freq: Frequency,
        method: FillnaOptions | None = None,
        how: Literal["start", "end"] | None = None,
        normalize: bool = False,
        fill_value: Hashable | None = None,
    ) -> Self:
        # [FUNCTION DOC]
        # Convert a time series to the specified frequency.
        # Parameters: freq is an offset or frequency string; method is the fill method
        # for new missing entries (pad/ffill, backfill/bfill, or None); how is start
        # or end for period indexes; normalize resets times to midnight; fill_value
        # fills missing entries introduced by upsampling (not those already existing).
        # Delegates to the asfreq function in the resample module (imported lazily)
        # and returns its result, a reindexed object.
        """
        Convert time series to specified frequency.

        Returns the original data conformed to a new index with the specified
        frequency.

        If the index of this Series/DataFrame is a :class:`~pandas.PeriodIndex`, the
        new index is the result of transforming the original index with
        :meth:`PeriodIndex.asfreq <pandas.PeriodIndex.asfreq>` (so the original index
        will map one-to-one to the new index).

        Otherwise, the new index will be equivalent to ``pd.date_range(start, end,
        freq=freq)`` where ``start`` and ``end`` are, respectively, the min and
        max entries in the original index (see :func:`pandas.date_range`). The
        values corresponding to any timesteps in the new index which were not present
        in the original index will be null (``NaN``), unless a method for filling
        such unknowns is provided (see the ``method`` parameter below).

        The :meth:`resample` method is more appropriate if an operation on each group of
        timesteps (such as an aggregate) is necessary to represent the data at the new
        frequency.

        Parameters
        ----------
        freq : DateOffset or str
            Frequency DateOffset or string.
        method : {'backfill'/'bfill', 'pad'/'ffill'}, default None
            Method to use for filling holes in reindexed Series (note this
            does not fill NaNs that already were present):

            * 'pad' / 'ffill': propagate last valid observation forward to next
              valid based on the order of the index
            * 'backfill' / 'bfill': use NEXT valid observation to fill.
        how : {'start', 'end'}, default end
            For PeriodIndex only (see PeriodIndex.asfreq).
        normalize : bool, default False
            Whether to reset output index to midnight.
        fill_value : scalar, optional
            Value to use for missing values, applied during upsampling (note
            this does not fill NaNs that already were present).

        Returns
        -------
        Series/DataFrame
            Series/DataFrame object reindexed to the specified frequency.

        See Also
        --------
        reindex : Conform DataFrame to new index with optional filling logic.

        Notes
        -----
        To learn more about the frequency strings, please see
        :ref:`this link<timeseries.offset_aliases>`.

        Examples
        --------
        Start by creating a series with 4 one minute timestamps.

        >>> index = pd.date_range("1/1/2000", periods=4, freq="min")
        >>> series = pd.Series([0.0, None, 2.0, 3.0], index=index)
        >>> df = pd.DataFrame({"s": series})
        >>> df
                               s
        2000-01-01 00:00:00    0.0
        2000-01-01 00:01:00    NaN
        2000-01-01 00:02:00    2.0
        2000-01-01 00:03:00    3.0

        Upsample the series into 30 second bins.

        >>> df.asfreq(freq="30s")
                               s
        2000-01-01 00:00:00    0.0
        2000-01-01 00:00:30    NaN
        2000-01-01 00:01:00    NaN
        2000-01-01 00:01:30    NaN
        2000-01-01 00:02:00    2.0
        2000-01-01 00:02:30    NaN
        2000-01-01 00:03:00    3.0

        Upsample again, providing a ``fill value``.

        >>> df.asfreq(freq="30s", fill_value=9.0)
                               s
        2000-01-01 00:00:00    0.0
        2000-01-01 00:00:30    9.0
        2000-01-01 00:01:00    NaN
        2000-01-01 00:01:30    9.0
        2000-01-01 00:02:00    2.0
        2000-01-01 00:02:30    9.0
        2000-01-01 00:03:00    3.0

        Upsample again, providing a ``method``.

        >>> df.asfreq(freq="30s", method="bfill")
                               s
        2000-01-01 00:00:00    0.0
        2000-01-01 00:00:30    NaN
        2000-01-01 00:01:00    NaN
        2000-01-01 00:01:30    2.0
        2000-01-01 00:02:00    2.0
        2000-01-01 00:02:30    3.0
        2000-01-01 00:03:00    3.0
        """
        from pandas.core.resample import asfreq

        return asfreq(
            self,
            freq,
            method=method,
            how=how,
            normalize=normalize,
            fill_value=fill_value,
        )

    @final
    def at_time(self, time, asof: bool = False, axis: Axis | None = None) -> Self:
        # [FUNCTION DOC]
        # Select values at a particular time of day.
        # Parameters: time is a time object or string; asof (default false) uses the
        # as-of matching rule; axis (default 0).
        # Raises a type error if the chosen axis is not a datetime index. Takes the
        # rows (or columns) at the matching positions from the index time lookup.
        # Returns the subset.
        """
        Select values at particular time of day (e.g., 9:30AM).

        Parameters
        ----------
        time : datetime.time or str
            The values to select.
        asof : bool, default False
            This parameter is currently not supported.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            For `Series` this parameter is unused and defaults to 0.

        Returns
        -------
        Series or DataFrame
            The values with the specified time.

        Raises
        ------
        TypeError
            If the index is not  a :class:`DatetimeIndex`

        See Also
        --------
        between_time : Select values between particular times of the day.
        first : Select initial periods of time series based on a date offset.
        last : Select final periods of time series based on a date offset.
        DatetimeIndex.indexer_at_time : Get just the index locations for
            values at particular time of the day.

        Examples
        --------
        >>> i = pd.date_range("2018-04-09", periods=4, freq="12h")
        >>> ts = pd.DataFrame({"A": [1, 2, 3, 4]}, index=i)
        >>> ts
                             A
        2018-04-09 00:00:00  1
        2018-04-09 12:00:00  2
        2018-04-10 00:00:00  3
        2018-04-10 12:00:00  4

        >>> ts.at_time("12:00")
                             A
        2018-04-09 12:00:00  2
        2018-04-10 12:00:00  4
        """
        if axis is None:
            axis = 0
        axis = self._get_axis_number(axis)

        index = self._get_axis(axis)

        if not isinstance(index, DatetimeIndex):
            raise TypeError("Index must be DatetimeIndex")

        indexer = index.indexer_at_time(time, asof=asof)
        return self.take(indexer, axis=axis)

    @final
    def between_time(
        self,
        start_time,
        end_time,
        inclusive: IntervalClosedType = "both",
        axis: Axis | None = None,
    ) -> Self:
        # [FUNCTION DOC]
        # Select values between particular times of day.
        # Parameters: start_time and end_time are time objects or strings; inclusive
        # is both, neither, left or right (default both); axis (default 0).
        # Raises a type error if the chosen axis is not a datetime index; the
        # inclusive value is validated (a value error for an invalid value). Takes the
        # matching positions from the index time-range lookup. Returns the subset.
        """
        Select values between particular times of the day (e.g., 9:00-9:30 AM).

        By setting ``start_time`` to be later than ``end_time``,
        you can get the times that are *not* between the two times.

        Parameters
        ----------
        start_time : datetime.time or str
            Initial time as a time filter limit.
        end_time : datetime.time or str
            End time as a time filter limit.
        inclusive : {"both", "neither", "left", "right"}, default "both"
            Include boundaries; whether to set each bound as closed or open.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Determine range time on index or columns value.
            For `Series` this parameter is unused and defaults to 0.

        Returns
        -------
        Series or DataFrame
            Data from the original object filtered to the specified dates range.

        Raises
        ------
        TypeError
            If the index is not  a :class:`DatetimeIndex`

        See Also
        --------
        at_time : Select values at a particular time of the day.
        first : Select initial periods of time series based on a date offset.
        last : Select final periods of time series based on a date offset.
        DatetimeIndex.indexer_between_time : Get just the index locations for
            values between particular times of the day.

        Examples
        --------
        >>> i = pd.date_range("2018-04-09", periods=4, freq="1D20min")
        >>> ts = pd.DataFrame({"A": [1, 2, 3, 4]}, index=i)
        >>> ts
                             A
        2018-04-09 00:00:00  1
        2018-04-10 00:20:00  2
        2018-04-11 00:40:00  3
        2018-04-12 01:00:00  4

        >>> ts.between_time("0:15", "0:45")
                             A
        2018-04-10 00:20:00  2
        2018-04-11 00:40:00  3

        You get the times that are *not* between two times by setting
        ``start_time`` later than ``end_time``:

        >>> ts.between_time("0:45", "0:15")
                             A
        2018-04-09 00:00:00  1
        2018-04-12 01:00:00  4
        """
        if axis is None:
            axis = 0
        axis = self._get_axis_number(axis)

        index = self._get_axis(axis)
        if not isinstance(index, DatetimeIndex):
            raise TypeError("Index must be DatetimeIndex")

        left_inclusive, right_inclusive = validate_inclusive(inclusive)
        indexer = index.indexer_between_time(
            start_time,
            end_time,
            include_start=left_inclusive,
            include_end=right_inclusive,
        )
        return self.take(indexer, axis=axis)

    @final
    def resample(
        self,
        rule,
        closed: Literal["right", "left"] | None = None,
        label: Literal["right", "left"] | None = None,
        convention: Literal["start", "end", "s", "e"] = "start",
        on: Level | None = None,
        level: Level | None = None,
        origin: str | TimestampConvertibleTypes = "start_day",
        offset: TimedeltaConvertibleTypes | None = None,
        group_keys: bool = False,
    ) -> Resampler:
        # [FUNCTION DOC]
        # Create a resampler object for time-series resampling.
        # Parameters: rule is the offset or frequency string; closed and label are
        # left or right to choose which bin edge is closed/used as label (defaults
        # depend on the rule); convention (start or end, default start) is used when
        # resampling periods; on names a datetime-like column to use instead of the
        # index; level names a level of a MultiIndex to use; origin (default
        # start_day) and offset adjust bin boundaries; group_keys controls whether
        # group keys are included in apply results (default false).
        # Delegates to the resampler factory in the resample module (imported lazily)
        # and returns the Resampler object; no computation happens until an
        # aggregation is called.
        """
        Resample time-series data.

        Convenience method for frequency conversion and resampling of time series.
        The object must have a datetime-like index (`DatetimeIndex`, `PeriodIndex`,
        or `TimedeltaIndex`), or the caller must pass the label of a datetime-like
        series/index to the ``on``/``level`` keyword parameter.

        Parameters
        ----------
        rule : DateOffset, Timedelta or str
            The offset string or object representing target conversion.
        closed : {'right', 'left'}, default None
            Which side of bin interval is closed. The default is 'left'
            for all frequency offsets except for 'ME', 'YE', 'QE', 'BME',
            'BA', 'BQE', and 'W' which all have a default of 'right'.
        label : {'right', 'left'}, default None
            Which bin edge label to label bucket with. The default is 'left'
            for all frequency offsets except for 'ME', 'YE', 'QE', 'BME',
            'BA', 'BQE', and 'W' which all have a default of 'right'.
        convention : {'start', 'end', 's', 'e'}, default 'start'
            For `PeriodIndex` only, controls whether to use the start or
            end of `rule`.
        on : str, optional
            For a DataFrame, column to use instead of index for resampling.
            Column must be datetime-like.
        level : str or int, optional
            For a MultiIndex, level (name or number) to use for
            resampling. `level` must be datetime-like.
        origin : Timestamp or str, default 'start_day'
            The timestamp on which to adjust the grouping. The timezone of origin
            must match the timezone of the index.
            If string, must be Timestamp convertible or one of the following:

            - 'epoch': `origin` is 1970-01-01
            - 'start': `origin` is the first value of the timeseries
            - 'start_day': `origin` is the first day at midnight of the timeseries

            - 'end': `origin` is the last value of the timeseries
            - 'end_day': `origin` is the ceiling midnight of the last day

            .. note::

                Only takes effect for Tick-frequencies (i.e. fixed frequencies like
                days, hours, and minutes, rather than months or quarters).
        offset : Timedelta or str, default is None
            An offset timedelta added to the origin.

        group_keys : bool, default False
            Whether to include the group keys in the result index when using
            ``.apply()`` on the resampled object.

            .. versionchanged:: 2.0.0

                ``group_keys`` now defaults to ``False``.

        Returns
        -------
        pandas.api.typing.Resampler
            :class:`~pandas.core.Resampler` object.

        See Also
        --------
        Series.resample : Resample a Series.
        DataFrame.resample : Resample a DataFrame.
        groupby : Group Series/DataFrame by mapping, function, label, or list of labels.
        asfreq : Reindex a Series/DataFrame with the given frequency without grouping.

        Notes
        -----
        See the `user guide
        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling>`__
        for more.

        To learn more about the offset strings, please see `this link
        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>`__.

        Examples
        --------
        Start by creating a series with 9 one minute timestamps.

        >>> index = pd.date_range("1/1/2000", periods=9, freq="min")
        >>> series = pd.Series(range(9), index=index)
        >>> series
        2000-01-01 00:00:00    0
        2000-01-01 00:01:00    1
        2000-01-01 00:02:00    2
        2000-01-01 00:03:00    3
        2000-01-01 00:04:00    4
        2000-01-01 00:05:00    5
        2000-01-01 00:06:00    6
        2000-01-01 00:07:00    7
        2000-01-01 00:08:00    8
        Freq: min, dtype: int64

        Downsample the series into 3 minute bins and sum the values
        of the timestamps falling into a bin.

        >>> series.resample("3min").sum()
        2000-01-01 00:00:00     3
        2000-01-01 00:03:00    12
        2000-01-01 00:06:00    21
        Freq: 3min, dtype: int64

        Downsample the series into 3 minute bins as above, but label each
        bin using the right edge instead of the left. Please note that the
        value in the bucket used as the label is not included in the bucket,
        which it labels. For example, in the original series the
        bucket ``2000-01-01 00:03:00`` contains the value 3, but the summed
        value in the resampled bucket with the label ``2000-01-01 00:03:00``
        does not include 3 (if it did, the summed value would be 6, not 3).

        >>> series.resample("3min", label="right").sum()
        2000-01-01 00:03:00     3
        2000-01-01 00:06:00    12
        2000-01-01 00:09:00    21
        Freq: 3min, dtype: int64

        To include this value close the right side of the bin interval,
        as shown below.

        >>> series.resample("3min", label="right", closed="right").sum()
        2000-01-01 00:00:00     0
        2000-01-01 00:03:00     6
        2000-01-01 00:06:00    15
        2000-01-01 00:09:00    15
        Freq: 3min, dtype: int64

        Upsample the series into 30 second bins.

        >>> series.resample("30s").asfreq()[0:5]  # Select first 5 rows
        2000-01-01 00:00:00   0.0
        2000-01-01 00:00:30   NaN
        2000-01-01 00:01:00   1.0
        2000-01-01 00:01:30   NaN
        2000-01-01 00:02:00   2.0
        Freq: 30s, dtype: float64

        Upsample the series into 30 second bins and fill the ``NaN``
        values using the ``ffill`` method.

        >>> series.resample("30s").ffill()[0:5]
        2000-01-01 00:00:00    0
        2000-01-01 00:00:30    0
        2000-01-01 00:01:00    1
        2000-01-01 00:01:30    1
        2000-01-01 00:02:00    2
        Freq: 30s, dtype: int64

        Upsample the series into 30 second bins and fill the
        ``NaN`` values using the ``bfill`` method.

        >>> series.resample("30s").bfill()[0:5]
        2000-01-01 00:00:00    0
        2000-01-01 00:00:30    1
        2000-01-01 00:01:00    1
        2000-01-01 00:01:30    2
        2000-01-01 00:02:00    2
        Freq: 30s, dtype: int64

        Pass a custom function via ``apply``

        >>> def custom_resampler(arraylike):
        ...     return np.sum(arraylike) + 5
        >>> series.resample("3min").apply(custom_resampler)
        2000-01-01 00:00:00     8
        2000-01-01 00:03:00    17
        2000-01-01 00:06:00    26
        Freq: 3min, dtype: int64

        For a Series with a PeriodIndex, the keyword `convention` can be
        used to control whether to use the start or end of `rule`.

        Resample a year by quarter using 'start' `convention`. Values are
        assigned to the first quarter of the period.

        >>> s = pd.Series(
        ...     [1, 2], index=pd.period_range("2012-01-01", freq="Y", periods=2)
        ... )
        >>> s
        2012    1
        2013    2
        Freq: Y-DEC, dtype: int64
        >>> s.resample("Q", convention="start").asfreq()
        2012Q1    1.0
        2012Q2    NaN
        2012Q3    NaN
        2012Q4    NaN
        2013Q1    2.0
        2013Q2    NaN
        2013Q3    NaN
        2013Q4    NaN
        Freq: Q-DEC, dtype: float64

        Resample quarters by month using 'end' `convention`. Values are
        assigned to the last month of the period.

        >>> q = pd.Series(
        ...     [1, 2, 3, 4], index=pd.period_range("2018-01-01", freq="Q", periods=4)
        ... )
        >>> q
        2018Q1    1
        2018Q2    2
        2018Q3    3
        2018Q4    4
        Freq: Q-DEC, dtype: int64
        >>> q.resample("M", convention="end").asfreq()
        2018-03    1.0
        2018-04    NaN
        2018-05    NaN
        2018-06    2.0
        2018-07    NaN
        2018-08    NaN
        2018-09    3.0
        2018-10    NaN
        2018-11    NaN
        2018-12    4.0
        Freq: M, dtype: float64

        For DataFrame objects, the keyword `on` can be used to specify the
        column instead of the index for resampling.

        >>> df = pd.DataFrame([10, 11, 9, 13, 14, 18, 17, 19], columns=["price"])
        >>> df["volume"] = [50, 60, 40, 100, 50, 100, 40, 50]
        >>> df["week_starting"] = pd.date_range("01/01/2018", periods=8, freq="W")
        >>> df
           price  volume week_starting
        0     10      50    2018-01-07
        1     11      60    2018-01-14
        2      9      40    2018-01-21
        3     13     100    2018-01-28
        4     14      50    2018-02-04
        5     18     100    2018-02-11
        6     17      40    2018-02-18
        7     19      50    2018-02-25
        >>> df.resample("ME", on="week_starting").mean()
                       price  volume
        week_starting
        2018-01-31     10.75    62.5
        2018-02-28     17.00    60.0

        For a DataFrame with MultiIndex, the keyword `level` can be used to
        specify on which level the resampling needs to take place.

        >>> days = pd.date_range("1/1/2000", periods=4, freq="D")
        >>> df2 = pd.DataFrame(
        ...     [
        ...         [10, 50],
        ...         [11, 60],
        ...         [9, 40],
        ...         [13, 100],
        ...         [14, 50],
        ...         [18, 100],
        ...         [17, 40],
        ...         [19, 50],
        ...     ],
        ...     columns=["price", "volume"],
        ...     index=pd.MultiIndex.from_product([days, ["morning", "afternoon"]]),
        ... )
        >>> df2
                              price  volume
        2000-01-01 morning       10      50
                   afternoon     11      60
        2000-01-02 morning        9      40
                   afternoon     13     100
        2000-01-03 morning       14      50
                   afternoon     18     100
        2000-01-04 morning       17      40
                   afternoon     19      50
        >>> df2.resample("D", level=0).sum()
                    price  volume
        2000-01-01     21     110
        2000-01-02     22     140
        2000-01-03     32     150
        2000-01-04     36      90

        If you want to adjust the start of the bins based on a fixed timestamp:

        >>> start, end = "2000-10-01 23:30:00", "2000-10-02 00:30:00"
        >>> rng = pd.date_range(start, end, freq="7min")
        >>> ts = pd.Series(np.arange(len(rng)) * 3, index=rng)
        >>> ts
        2000-10-01 23:30:00     0
        2000-10-01 23:37:00     3
        2000-10-01 23:44:00     6
        2000-10-01 23:51:00     9
        2000-10-01 23:58:00    12
        2000-10-02 00:05:00    15
        2000-10-02 00:12:00    18
        2000-10-02 00:19:00    21
        2000-10-02 00:26:00    24
        Freq: 7min, dtype: int64

        >>> ts.resample("17min").sum()
        2000-10-01 23:14:00     0
        2000-10-01 23:31:00     9
        2000-10-01 23:48:00    21
        2000-10-02 00:05:00    54
        2000-10-02 00:22:00    24
        Freq: 17min, dtype: int64

        >>> ts.resample("17min", origin="epoch").sum()
        2000-10-01 23:18:00     0
        2000-10-01 23:35:00    18
        2000-10-01 23:52:00    27
        2000-10-02 00:09:00    39
        2000-10-02 00:26:00    24
        Freq: 17min, dtype: int64

        >>> ts.resample("17min", origin="2000-01-01").sum()
        2000-10-01 23:24:00     3
        2000-10-01 23:41:00    15
        2000-10-01 23:58:00    45
        2000-10-02 00:15:00    45
        Freq: 17min, dtype: int64

        If you want to adjust the start of the bins with an `offset` Timedelta, the two
        following lines are equivalent:

        >>> ts.resample("17min", origin="start").sum()
        2000-10-01 23:30:00     9
        2000-10-01 23:47:00    21
        2000-10-02 00:04:00    54
        2000-10-02 00:21:00    24
        Freq: 17min, dtype: int64

        >>> ts.resample("17min", offset="23h30min").sum()
        2000-10-01 23:30:00     9
        2000-10-01 23:47:00    21
        2000-10-02 00:04:00    54
        2000-10-02 00:21:00    24
        Freq: 17min, dtype: int64

        If you want to take the largest Timestamp as the end of the bins:

        >>> ts.resample("17min", origin="end").sum()
        2000-10-01 23:35:00     0
        2000-10-01 23:52:00    18
        2000-10-02 00:09:00    27
        2000-10-02 00:26:00    63
        Freq: 17min, dtype: int64

        In contrast with the `start_day`, you can use `end_day` to take the ceiling
        midnight of the largest Timestamp as the end of the bins and drop the bins
        not containing data:

        >>> ts.resample("17min", origin="end_day").sum()
        2000-10-01 23:38:00     3
        2000-10-01 23:55:00    15
        2000-10-02 00:12:00    45
        2000-10-02 00:29:00    45
        Freq: 17min, dtype: int64
        """
        from pandas.core.resample import get_resampler

        return get_resampler(
            cast("Series | DataFrame", self),
            freq=rule,
            label=label,
            closed=closed,
            convention=convention,
            key=on,
            level=level,
            origin=origin,
            offset=offset,
            group_keys=group_keys,
        )

    @final
    def rank(
        self,
        axis: Axis = 0,
        method: Literal["average", "min", "max", "first", "dense"] = "average",
        numeric_only: bool = False,
        na_option: Literal["keep", "top", "bottom"] = "keep",
        ascending: bool = True,
        pct: bool = False,
    ) -> Self:
        # [FUNCTION DOC]
        # Compute numerical data ranks along an axis.
        # Parameters: axis (default 0); method is average, min, max, first or dense
        # for ties; numeric_only restricts to numeric columns; na_option is keep, top
        # or bottom; ascending; pct returns percentile ranks.
        # Raises a value error for an invalid na_option, and a type error if
        # numeric_only is requested for a Series with a non-numeric dtype. Ranks are
        # computed by the nested function on the whole object (or on the numeric
        # subset). Returns an object of float ranks with the same axes.
        """
        Compute numerical data ranks (1 through n) along axis.

        By default, equal values are assigned a rank that is the average of the
        ranks of those values.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Index to direct ranking.
            For `Series` this parameter is unused and defaults to 0.
        method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'
            How to rank the group of records that have the same value (i.e. ties):

            * average: average rank of the group
            * min: lowest rank in the group
            * max: highest rank in the group
            * first: ranks assigned in order they appear in the array
            * dense: like 'min', but rank always increases by 1 between groups.

        numeric_only : bool, default False
            For DataFrame objects, rank only numeric columns if set to True.

            .. versionchanged:: 2.0.0
                The default value of ``numeric_only`` is now ``False``.

        na_option : {'keep', 'top', 'bottom'}, default 'keep'
            How to rank NaN values:

            * keep: assign NaN rank to NaN values
            * top: assign lowest rank to NaN values
            * bottom: assign highest rank to NaN values

        ascending : bool, default True
            Whether or not the elements should be ranked in ascending order.
        pct : bool, default False
            Whether or not to display the returned rankings in percentile
            form.

        Returns
        -------
        same type as caller
            Return a Series or DataFrame with data ranks as values.

        See Also
        --------
        core.groupby.DataFrameGroupBy.rank : Rank of values within each group.
        core.groupby.SeriesGroupBy.rank : Rank of values within each group.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     data={
        ...         "Animal": ["cat", "penguin", "dog", "spider", "snake"],
        ...         "Number_legs": [4, 2, 4, 8, np.nan],
        ...     }
        ... )
        >>> df
            Animal  Number_legs
        0      cat          4.0
        1  penguin          2.0
        2      dog          4.0
        3   spider          8.0
        4    snake          NaN

        Ties are assigned the mean of the ranks (by default) for the group.

        >>> s = pd.Series(range(5), index=list("abcde"))
        >>> s["d"] = s["b"]
        >>> s.rank()
        a    1.0
        b    2.5
        c    4.0
        d    2.5
        e    5.0
        dtype: float64

        The following example shows how the method behaves with the above
        parameters:

        * default_rank: this is the default behaviour obtained without using
          any parameter.
        * max_rank: setting ``method = 'max'`` the records that have the
          same values are ranked using the highest rank (e.g.: since 'cat'
          and 'dog' are both in the 2nd and 3rd position, rank 3 is assigned.)
        * NA_bottom: choosing ``na_option = 'bottom'``, if there are records
          with NaN values they are placed at the bottom of the ranking.
        * pct_rank: when setting ``pct = True``, the ranking is expressed as
          percentile rank.

        >>> df["default_rank"] = df["Number_legs"].rank()
        >>> df["max_rank"] = df["Number_legs"].rank(method="max")
        >>> df["NA_bottom"] = df["Number_legs"].rank(na_option="bottom")
        >>> df["pct_rank"] = df["Number_legs"].rank(pct=True)
        >>> df
            Animal  Number_legs  default_rank  max_rank  NA_bottom  pct_rank
        0      cat          4.0           2.5       3.0        2.5     0.625
        1  penguin          2.0           1.0       1.0        1.0     0.250
        2      dog          4.0           2.5       3.0        2.5     0.625
        3   spider          8.0           4.0       4.0        4.0     1.000
        4    snake          NaN           NaN       NaN        5.0       NaN
        """
        axis_int = self._get_axis_number(axis)

        if na_option not in {"keep", "top", "bottom"}:
            msg = "na_option must be one of 'keep', 'top', or 'bottom'"
            raise ValueError(msg)

        def ranker(data):
            # [FUNCTION DOC]
            # HELPER: nested function of the rank method that does the actual ranking
            # of an object using the closure parameters.
            # Parameters: data is a Series or DataFrame. Extension array values use
            # their own rank implementation; other arrays use the generic rank
            # algorithm, both along the closure axis. The ranks are wrapped in the
            # same class with the same axes and finalized with the method name rank.
            # Returns the ranked object.
            if data.ndim == 2:
                # i.e. DataFrame, we cast to ndarray
                values = data.values
            else:
                # i.e. Series, can dispatch to EA
                values = data._values

            if isinstance(values, ExtensionArray):
                ranks = values._rank(
                    axis=axis_int,
                    method=method,
                    ascending=ascending,
                    na_option=na_option,
                    pct=pct,
                )
            else:
                ranks = algos.rank(
                    values,
                    axis=axis_int,
                    method=method,
                    ascending=ascending,
                    na_option=na_option,
                    pct=pct,
                )

            ranks_obj = self._constructor(ranks, **data._construct_axes_dict())
            return ranks_obj.__finalize__(self, method="rank")

        if numeric_only:
            if self.ndim == 1 and not is_numeric_dtype(self.dtype):
                # GH#47500
                raise TypeError(
                    "Series.rank does not allow numeric_only=True with "
                    "non-numeric dtype."
                )
            data = self._get_numeric_data()
        else:
            data = self

        return ranker(data)

    def compare(
        self,
        other: Self,
        align_axis: Axis = 1,
        keep_shape: bool = False,
        keep_equal: bool = False,
        result_names: Suffixes = ("self", "other"),
    ):
        # [FUNCTION DOC]
        # Compare with another object of the same class and show the differences.
        # Parameters: other must have exactly the same class as self (otherwise a type
        # error naming both classes is raised); align_axis is 1 or columns to put
        # differences side by side (default) or 0 or index to stack them; keep_shape
        # keeps all rows and columns instead of only different ones; keep_equal keeps
        # equal values instead of marking them missing; result_names is a tuple of two
        # labels for the pieces (a type error for anything that is not a tuple).
        # A difference mask treats positions where both are missing as equal. Unless
        # equal values are kept, equal positions become missing in both. Unless the
        # shape is kept, frames are reduced to rows and columns containing differences
        # and Series are reduced to differing entries. The two pieces are concatenated
        # along the chosen axis using the result names as keys. If this adds a new
        # dimension (two Series side by side) it is returned. Otherwise the names of
        # the combined axis are temporarily replaced by positions, the key level is
        # moved to the innermost level, the original names are restored, and the
        # entries are reordered so that corresponding self and other entries
        # alternate. Returns the resulting object.
        """
        Compare to another Series/DataFrame and show the differences.

        Parameters
        ----------
        other : Series/DataFrame
            Object to compare with.

        align_axis : {0 or 'index', 1 or 'columns'}, default 1
            Determine which axis to align the comparison on.

            * 0, or 'index' : Resulting differences are stacked vertically
              with rows drawn alternately from self and other.
            * 1, or 'columns' : Resulting differences are aligned horizontally
              with columns drawn alternately from self and other.

        keep_shape : bool, default False
            If true, all rows and columns are kept.
            Otherwise, only the ones with different values are kept.

        keep_equal : bool, default False
            If true, the result keeps values that are equal.
            Otherwise, equal values are shown as NaNs.

        result_names : tuple, default ('self', 'other')
            Set the dataframes names in the comparison.
        """
        if type(self) is not type(other):
            cls_self, cls_other = type(self).__name__, type(other).__name__
            raise TypeError(
                f"can only compare '{cls_self}' (not '{cls_other}') with '{cls_self}'"
            )

        # error: Unsupported left operand type for & ("Self")
        mask = ~((self == other) | (self.isna() & other.isna()))  # type: ignore[operator]
        mask.fillna(True, inplace=True)

        if not keep_equal:
            self = self.where(mask)
            other = other.where(mask)

        if not keep_shape:
            if isinstance(self, ABCDataFrame):
                cmask = mask.any()
                rmask = mask.any(axis=1)
                self = self.loc[rmask, cmask]
                other = other.loc[rmask, cmask]
            else:
                self = self[mask]
                other = other[mask]
        if not isinstance(result_names, tuple):
            raise TypeError(
                f"Passing 'result_names' as a {type(result_names)} is not "
                "supported. Provide 'result_names' as a tuple instead."
            )

        if align_axis in (1, "columns"):  # This is needed for Series
            axis = 1
        else:
            axis = self._get_axis_number(align_axis)

        # error: List item 0 has incompatible type "NDFrame"; expected
        #  "Union[Series, DataFrame]"
        diff = concat(
            [self, other],  # type: ignore[list-item]
            axis=axis,
            keys=result_names,
        )

        if axis >= self.ndim:
            # No need to reorganize data if stacking on new axis
            # This currently applies for stacking two Series on columns
            return diff

        ax = diff._get_axis(axis)
        ax_names = np.array(ax.names)

        # set index names to positions to avoid confusion
        ax.names = np.arange(len(ax_names))

        # bring self-other to inner level
        order = [*range(1, ax.nlevels), 0]
        if isinstance(diff, ABCDataFrame):
            diff = diff.reorder_levels(order, axis=axis)
        else:
            diff = diff.reorder_levels(order)

        # restore the index names in order
        diff._get_axis(axis=axis).names = ax_names[order]

        # reorder axis to keep things organized
        indices = (
            np.arange(diff.shape[axis])
            .reshape([2, diff.shape[axis] // 2])
            .T.reshape(-1)
        )
        diff = diff.take(indices, axis=axis)

        return diff

    @final
    def align(
        self,
        other: NDFrameT,
        join: AlignJoin = "outer",
        axis: Axis | None = None,
        level: Level | None = None,
        copy: bool | lib.NoDefault = lib.no_default,
        fill_value: Hashable | None = None,
    ) -> tuple[Self, NDFrameT]:
        # [FUNCTION DOC]
        # Align two objects on their axes with the specified join method.
        # Parameters: other is a Series or DataFrame (anything else raises a type
        # error); join is outer (default), inner, left or right; axis restricts
        # alignment to one axis (None means both); level is the MultiIndex level to
        # broadcast across; copy is a deprecated placeholder; fill_value fills missing
        # entries introduced by alignment.
        # Dispatches to the frame or series alignment helper depending on the type of
        # other. When aligning along the index of timezone-aware datetimes with
        # differing time zones, both results get the join index (converted to UTC by
        # the join) on shallow copies. Each result is finalized from the corresponding
        # source. Returns a tuple of the aligned self and other.
        """
        Align two objects on their axes with the specified join method.

        Join method is specified for each axis Index.

        Parameters
        ----------
        other : DataFrame or Series
            The object to align with.
        join : {'outer', 'inner', 'left', 'right'}, default 'outer'
            Type of alignment to be performed.

            * left: use only keys from left frame, preserve key order.
            * right: use only keys from right frame, preserve key order.
            * outer: use union of keys from both frames, sort keys lexicographically.
            * inner: use intersection of keys from both frames,
              preserve the order of the left keys.

        axis : allowed axis of the other object, default None
            Align on index (0), columns (1), or both (None).
        level : int or level name, default None
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        copy : bool, default False
            This keyword is now ignored; changing its value will have no
            impact on the method.

            .. deprecated:: 3.0.0

                This keyword is ignored and will be removed in pandas 4.0. Since
                pandas 3.0, this method always returns a new object using a lazy
                copy mechanism that defers copies until necessary
                (Copy-on-Write). See the `user guide on Copy-on-Write
                <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                for more details.

        fill_value : scalar, default np.nan
            Value to use for missing values. Defaults to NaN, but can be any
            "compatible" value.

        Returns
        -------
        tuple of (Series/DataFrame, type of other)
            Aligned objects.

        See Also
        --------
        Series.align : Align two objects on their axes with specified join method.
        DataFrame.align : Align two objects on their axes with specified join method.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     [[1, 2, 3, 4], [6, 7, 8, 9]], columns=["D", "B", "E", "A"], index=[1, 2]
        ... )
        >>> other = pd.DataFrame(
        ...     [[10, 20, 30, 40], [60, 70, 80, 90], [600, 700, 800, 900]],
        ...     columns=["A", "B", "C", "D"],
        ...     index=[2, 3, 4],
        ... )
        >>> df
           D  B  E  A
        1  1  2  3  4
        2  6  7  8  9
        >>> other
            A    B    C    D
        2   10   20   30   40
        3   60   70   80   90
        4  600  700  800  900

        Align on columns:

        >>> left, right = df.align(other, join="outer", axis=1)
        >>> left
           A  B   C  D  E
        1  4  2 NaN  1  3
        2  9  7 NaN  6  8
        >>> right
            A    B    C    D   E
        2   10   20   30   40 NaN
        3   60   70   80   90 NaN
        4  600  700  800  900 NaN

        We can also align on the index:

        >>> left, right = df.align(other, join="outer", axis=0)
        >>> left
            D    B    E    A
        1  1.0  2.0  3.0  4.0
        2  6.0  7.0  8.0  9.0
        3  NaN  NaN  NaN  NaN
        4  NaN  NaN  NaN  NaN
        >>> right
            A      B      C      D
        1    NaN    NaN    NaN    NaN
        2   10.0   20.0   30.0   40.0
        3   60.0   70.0   80.0   90.0
        4  600.0  700.0  800.0  900.0

        Finally, the default `axis=None` will align on both index and columns:

        >>> left, right = df.align(other, join="outer", axis=None)
        >>> left
             A    B   C    D    E
        1  4.0  2.0 NaN  1.0  3.0
        2  9.0  7.0 NaN  6.0  8.0
        3  NaN  NaN NaN  NaN  NaN
        4  NaN  NaN NaN  NaN  NaN
        >>> right
               A      B      C      D   E
        1    NaN    NaN    NaN    NaN NaN
        2   10.0   20.0   30.0   40.0 NaN
        3   60.0   70.0   80.0   90.0 NaN
        4  600.0  700.0  800.0  900.0 NaN
        """
        self._check_copy_deprecation(copy)

        _right: DataFrame | Series
        if axis is not None:
            axis = self._get_axis_number(axis)
        if isinstance(other, ABCDataFrame):
            left, _right, join_index = self._align_frame(
                other,
                join=join,
                axis=axis,
                level=level,
                fill_value=fill_value,
            )

        elif isinstance(other, ABCSeries):
            left, _right, join_index = self._align_series(
                other,
                join=join,
                axis=axis,
                level=level,
                fill_value=fill_value,
            )
        else:  # pragma: no cover
            raise TypeError(f"unsupported type: {type(other)}")

        right = cast(NDFrameT, _right)
        if self.ndim == 1 or axis == 0:
            # If we are aligning timezone-aware DatetimeIndexes and the timezones
            #  do not match, convert both to UTC.
            if isinstance(left.index.dtype, DatetimeTZDtype):
                if left.index.tz != right.index.tz:
                    if join_index is not None:
                        # GH#33671 copy to ensure we don't change the index on
                        #  our original Series
                        left = left.copy(deep=False)
                        right = right.copy(deep=False)
                        left.index = join_index
                        right.index = join_index

        left = left.__finalize__(self)
        right = right.__finalize__(other)
        return left, right

    @final
    def _align_frame(
        self,
        other: DataFrame,
        join: AlignJoin = "outer",
        axis: Axis | None = None,
        level=None,
        fill_value=None,
    ) -> tuple[Self, DataFrame, Index | None]:
        # [FUNCTION DOC]
        # HELPER: aligns self with a DataFrame.
        # Parameters: other (a frame); join; axis (None, 0 or 1); level; fill_value.
        # For the index (axis None or 0) and for the columns (axis None or 1; not for
        # a Series) the labels are joined when they differ, producing the joined
        # labels and indexers for both sides. Both sides are reindexed with their
        # indexers allowing duplicates and the fill value. Returns the aligned self,
        # the aligned other and the joined index (None if the index did not need
        # joining).
        # defaults
        join_index, join_columns = None, None
        ilidx, iridx = None, None
        clidx, cridx = None, None

        is_series = isinstance(self, ABCSeries)

        if (axis is None or axis == 0) and not self.index.equals(other.index):
            join_index, ilidx, iridx = self.index.join(
                other.index, how=join, level=level, return_indexers=True
            )

        if (
            (axis is None or axis == 1)
            and not is_series
            and not self.columns.equals(other.columns)
        ):
            join_columns, clidx, cridx = self.columns.join(
                other.columns, how=join, level=level, return_indexers=True
            )

        if is_series:
            reindexers = {0: [join_index, ilidx]}
        else:
            reindexers = {0: [join_index, ilidx], 1: [join_columns, clidx]}

        left = self._reindex_with_indexers(
            reindexers, fill_value=fill_value, allow_dups=True
        )
        # other must be always DataFrame
        right = other._reindex_with_indexers(
            {0: [join_index, iridx], 1: [join_columns, cridx]},
            fill_value=fill_value,
            allow_dups=True,
        )
        return left, right, join_index

    @final
    def _align_series(
        self,
        other: Series,
        join: AlignJoin = "outer",
        axis: Axis | None = None,
        level=None,
        fill_value=None,
    ) -> tuple[Self, Series, Index | None]:
        # [FUNCTION DOC]
        # HELPER: aligns self with a Series.
        # Parameters: other (a Series); join; axis (0 or 1); level; fill_value.
        # Raises a value error if a frame is aligned without an axis or the axis is
        # not none, 0 or 1, and if a Series is aligned with axis one. When the axis is
        # zero or None, the indexes are joined if they differ, and both sides are
        # reindexed (a frame through its manager along the row axis; a Series through
        # its own reindex-by-indexer helper). Otherwise (axis one with a frame) the
        # frame columns are joined with the Series index and the frame manager is
        # reindexed along the column axis, while the Series is reindexed to the joined
        # labels. If fill_value is not missing both sides are filled with it. Returns
        # the aligned self, aligned other and the joined labels.
        is_series = isinstance(self, ABCSeries)

        if (not is_series and axis is None) or axis not in [None, 0, 1]:
            raise ValueError("Must specify axis=0 or 1")

        if is_series and axis == 1:
            raise ValueError("cannot align series to a series other than axis 0")

        # series/series compat, other must always be a Series
        if not axis:
            # equal
            if self.index.equals(other.index):
                join_index, lidx, ridx = None, None, None
            else:
                join_index, lidx, ridx = self.index.join(
                    other.index, how=join, level=level, return_indexers=True
                )

            if is_series:
                left = self._reindex_indexer(join_index, lidx)
            elif lidx is None or join_index is None:
                left = self.copy(deep=False)
            else:
                new_mgr = self._mgr.reindex_indexer(join_index, lidx, axis=1)
                left = self._constructor_from_mgr(new_mgr, axes=new_mgr.axes)

            right = other._reindex_indexer(join_index, ridx)

        else:
            # one has > 1 ndim
            fdata = self._mgr
            join_index = self.axes[1]
            lidx, ridx = None, None
            if not join_index.equals(other.index):
                join_index, lidx, ridx = join_index.join(
                    other.index, how=join, level=level, return_indexers=True
                )

            if lidx is not None:
                bm_axis = self._get_block_manager_axis(1)
                fdata = fdata.reindex_indexer(join_index, lidx, axis=bm_axis)

            left = self._constructor_from_mgr(fdata, axes=fdata.axes)

            right = other._reindex_indexer(join_index, ridx)

        # fill
        fill_na = notna(fill_value)
        if fill_na:
            left = left.fillna(fill_value)
            right = right.fillna(fill_value)

        return left, right, join_index

    @final
    def _where(
        self,
        cond,
        other=lib.no_default,
        *,
        inplace: bool = False,
        axis: Axis | None = None,
        level=None,
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: shared implementation of where and mask; also used by item
        # assignment. Unlike the public where, the replacement value is not called if
        # it is a callable.
        # Parameters: cond is a boolean-like object (callables are evaluated with
        # self); other is the replacement (scalar, array-like, Series or DataFrame;
        # unset means missing); keyword only: inplace; axis for aligning a
        # lower-dimensional other; level for MultiIndex alignment.
        # Condition processing: a labeled condition is aligned (a Series condition for
        # a frame is broadcast across the columns first) with a right join onto self;
        # an unlabeled condition must have exactly the shape of self or a value error
        # is raised. Missing entries in the condition are filled with the in-place
        # flag (so they count as true only when in place), then dtypes are inferred.
        # For non-empty conditions every block must be boolean, otherwise a type error
        # naming the dtype is raised (extension boolean frames are converted to plain
        # boolean arrays); empty conditions are cast to boolean. For in-place
        # operation the condition is inverted (since the put-mask operation overwrites
        # where the mask is true). Finally the condition is reindexed to the info
        # axis.
        # Replacement processing: a labeled other with at most the dimensionality of
        # self is aligned with a left join (an index mismatch with no axis given
        # raises an invalid-index error); a lower-dimensional other along an axis is
        # broadcast to the shape (for plain numpy arrays) or applied column by column
        # (for extension arrays, building the result frame directly and returning it,
        # in place or not). A higher-dimensional other raises a not-implemented error.
        # Other index-like inputs are converted to arrays. An array-like other of
        # different shape than self raises a value error for frames (one-dimensional
        # objects may pass differing length), and one of the same shape is wrapped as
        # a labeled object. If axis is None it defaults to 0, and alignment is flagged
        # when dimensionality matches or when the axis is one.
        # Execution: in place, the manager put-mask is applied, the internals replaced
        # and self returned; otherwise the manager where operation produces a new
        # object finalized from self (without a method name).
        """
        Equivalent to public method `where`, except that `other` is not
        applied as a function even if callable. Used in __setitem__.
        """
        inplace = validate_bool_kwarg(inplace, "inplace")

        if axis is not None:
            axis = self._get_axis_number(axis)

        # align the cond to same shape as myself
        cond = common.apply_if_callable(cond, self)
        if isinstance(cond, NDFrame):
            # CoW: Make sure reference is not kept alive
            if cond.ndim == 1 and self.ndim == 2:
                cond = cond._constructor_expanddim(
                    dict.fromkeys(range(len(self.columns)), cond),
                    copy=False,
                )
                cond.columns = self.columns
            cond = cond.align(self, join="right")[0]
        else:
            if not hasattr(cond, "shape"):
                cond = np.asanyarray(cond)
            if cond.shape != self.shape:
                raise ValueError("Array conditional must be same shape as self")
            cond = self._constructor(cond, **self._construct_axes_dict(), copy=False)

        # make sure we are boolean
        fill_value = bool(inplace)
        cond = cond.fillna(fill_value)
        cond = cond.infer_objects()

        msg = "Boolean array expected for the condition, not {dtype}"

        if not cond.empty:
            if not isinstance(cond, ABCDataFrame):
                # This is a single-dimensional object.
                if not is_bool_dtype(cond):
                    raise TypeError(msg.format(dtype=cond.dtype))
            else:
                for block in cond._mgr.blocks:
                    if not is_bool_dtype(block.dtype):
                        raise TypeError(msg.format(dtype=block.dtype))
                if cond._mgr.any_extension_types:
                    # GH51574: avoid object ndarray conversion later on
                    cond = cond._constructor(
                        cond.to_numpy(dtype=bool, na_value=fill_value),
                        **cond._construct_axes_dict(),
                    )
        else:
            # GH#21947 we have an empty DataFrame/Series, could be object-dtype
            cond = cond.astype(bool)

        cond = -cond if inplace else cond
        cond = cond.reindex(self._info_axis, axis=self._info_axis_number)

        # try to align with other
        if isinstance(other, NDFrame):
            # align with me
            if other.ndim <= self.ndim:
                # CoW: Make sure reference is not kept alive
                other = self.align(
                    other,
                    join="left",
                    axis=axis,
                    level=level,
                    fill_value=None,
                )[1]

                # if we are NOT aligned, raise as we cannot where index
                if axis is None and not other._indexed_same(self):
                    raise InvalidIndexError

                if other.ndim < self.ndim:
                    other = other._values
                    if isinstance(other, np.ndarray):
                        # TODO(EA2D): could also do this for NDArrayBackedEA cases?
                        if axis == 0:
                            other = np.reshape(other, (-1, 1))
                        elif axis == 1:
                            other = np.reshape(other, (1, -1))

                        other = np.broadcast_to(other, self.shape)
                    else:
                        # GH#38729, GH#62038 avoid lossy casting or object-casting
                        if axis == 0:
                            res_cols = [
                                self.iloc[:, i]._where(
                                    cond.iloc[:, i],
                                    other,
                                )
                                for i in range(self.shape[1])
                            ]
                        elif axis == 1:
                            # TODO: can we use a zero-copy alternative to "repeat"?
                            res_cols = [
                                self.iloc[:, i]._where(
                                    cond.iloc[:, i],
                                    other[i : i + 1].repeat(len(self)),
                                )
                                for i in range(self.shape[1])
                            ]
                        res = self._constructor(dict(enumerate(res_cols)))
                        res.index = self.index
                        res.columns = self.columns
                        if inplace:
                            self._update_inplace(res)
                            return self
                        return res.__finalize__(self)

            # slice me out of the other
            else:
                raise NotImplementedError(
                    "cannot align with a higher dimensional NDFrame"
                )

        elif not isinstance(other, (MultiIndex, NDFrame)):
            # mainly just catching Index here
            other = extract_array(other, extract_numpy=True)

        if isinstance(other, (np.ndarray, ExtensionArray)):
            if other.shape != self.shape:
                if self.ndim != 1:
                    # In the ndim == 1 case we may have
                    #  other length 1, which we treat as scalar (GH#2745, GH#4192)
                    #  or len(other) == icond.sum(), which we treat like
                    #  __setitem__ (GH#3235)
                    raise ValueError(
                        "other must be the same shape as self when an ndarray"
                    )

            # we are the same shape, so create an actual object for alignment
            else:
                other = self._constructor(
                    other, **self._construct_axes_dict(), copy=False
                )

        if axis is None:
            axis = 0

        if self.ndim == getattr(other, "ndim", 0):
            align = True
        else:
            align = self._get_axis_number(axis) == 1

        if inplace:
            # we may have different type blocks come out of putmask, so
            # reconstruct the block manager

            new_data = self._mgr.putmask(mask=cond, new=other, align=align)
            result = self._constructor_from_mgr(new_data, axes=new_data.axes)
            self._update_inplace(result)
            return self

        else:
            new_data = self._mgr.where(
                other=other,
                cond=cond,
                align=align,
            )
            result = self._constructor_from_mgr(new_data, axes=new_data.axes)
            return result.__finalize__(self)

    @final
    def where(
        self,
        cond,
        other=lib.no_default,
        *,
        inplace: bool = False,
        axis: Axis | None = None,
        level: Level | None = None,
    ) -> Self:
        # [FUNCTION DOC]
        # Keep values where the condition is true and replace the rest.
        # Parameters: cond is a boolean Series/DataFrame, array-like or callable
        # returning one; other is the replacement (scalar, labeled object, array-like
        # or callable evaluated with self; unset means missing); keyword only:
        # inplace; axis for alignment; level for MultiIndex alignment.
        # When inplace is true and the chained-assignment warning has not been
        # disabled globally, the reference count of the object is inspected and, if it
        # indicates the call is made on a temporary (a chained assignment) rather than
        # a local variable of the caller, a chained-assignment warning is emitted at
        # stack level two.
        # Evaluates a callable other and delegates to the shared where helper. Returns
        # the new object, or self when in place.
        """
        Replace values where the condition is False.

        This method allows conditional replacement of values. Where the
        condition evaluates to True, the original values are retained; where
        it evaluates to False, values are replaced with corresponding entries
        from ``other``.

        Parameters
        ----------
        cond : bool Series/DataFrame, array-like, or callable
            Where `cond` is True, keep the original value. Where
            False, replace with corresponding value from `other`.
            If `cond` is callable, it is computed on the Series/DataFrame and
            should return boolean Series/DataFrame or array. The callable must
            not change input Series/DataFrame (though pandas doesn't check it).
        other : scalar, Series/DataFrame, or callable
            Entries where `cond` is False are replaced with
            corresponding value from `other`.
            If other is callable, it is computed on the Series/DataFrame and
            should return scalar or Series/DataFrame. The callable must not
            change input Series/DataFrame (though pandas doesn't check it).
            If not specified, entries will be filled with the corresponding
            NULL value (``np.nan`` for numpy dtypes, ``pd.NA`` for extension
            dtypes).
        inplace : bool, default False
            Whether to perform the operation in place on the data.
        axis : int, default None
            Alignment axis if needed. For `Series` this parameter is
            unused and defaults to 0.
        level : int, default None
            Alignment level if needed.

        Returns
        -------
        Series or DataFrame
            When applied to a Series, the function will return a Series,
            and when applied to a DataFrame, it will return a DataFrame.

        See Also
        --------
        :func:`DataFrame.mask` : Return an object of same shape as caller.
        :func:`Series.mask` : Return an object of same shape as caller.

        Notes
        -----
        The where method is an application of the if-then idiom. For each
        element in the caller, if ``cond`` is ``True`` the
        element is used; otherwise the corresponding element from
        ``other`` is used. If the axis of ``other`` does not align with axis of
        ``cond`` Series/DataFrame, the values of ``cond`` on misaligned index positions
        will be filled with False.

        The signature for :func:`Series.where` or
        :func:`DataFrame.where` differs from :func:`numpy.where`.
        Roughly ``df1.where(m, df2)`` is equivalent to ``np.where(m, df1, df2)``.

        For further details and examples see the ``where`` documentation in
        :ref:`indexing <indexing.where_mask>`.

        The dtype of the object takes precedence. The fill value is casted to
        the object's dtype, if this can be done losslessly.

        Examples
        --------
        >>> s = pd.Series(range(5))
        >>> s.where(s > 0)
        0    NaN
        1    1.0
        2    2.0
        3    3.0
        4    4.0
        dtype: float64
        >>> s.mask(s > 0)
        0    0.0
        1    NaN
        2    NaN
        3    NaN
        4    NaN
        dtype: float64

        >>> s = pd.Series(range(5))
        >>> t = pd.Series([True, False])
        >>> s.where(t, 99)
        0     0
        1    99
        2    99
        3    99
        4    99
        dtype: int64
        >>> s.mask(t, 99)
        0    99
        1     1
        2    99
        3    99
        4    99
        dtype: int64

        >>> s.where(s > 1, 10)
        0    10
        1    10
        2    2
        3    3
        4    4
        dtype: int64
        >>> s.mask(s > 1, 10)
        0     0
        1     1
        2    10
        3    10
        4    10
        dtype: int64

        >>> df = pd.DataFrame(np.arange(10).reshape(-1, 2), columns=["A", "B"])
        >>> df
           A  B
        0  0  1
        1  2  3
        2  4  5
        3  6  7
        4  8  9
        >>> m = df % 3 == 0
        >>> df.where(m, -df)
           A  B
        0  0 -1
        1 -2  3
        2 -4 -5
        3  6 -7
        4 -8  9
        >>> df.where(m, -df) == np.where(m, df, -df)
              A     B
        0  True  True
        1  True  True
        2  True  True
        3  True  True
        4  True  True
        >>> df.where(m, -df) == df.mask(~m, -df)
              A     B
        0  True  True
        1  True  True
        2  True  True
        3  True  True
        4  True  True
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        if inplace:
            if not CHAINED_WARNING_DISABLED:
                if sys.getrefcount(
                    self
                ) <= REF_COUNT_METHOD and not common.is_local_in_caller_frame(self):
                    warnings.warn(
                        _chained_assignment_method_msg,
                        ChainedAssignmentError,
                        stacklevel=2,
                    )

        other = common.apply_if_callable(other, self)
        return self._where(cond, other, inplace=inplace, axis=axis, level=level)

    @final
    def mask(
        self,
        cond,
        other=lib.no_default,
        *,
        inplace: bool = False,
        axis: Axis | None = None,
        level: Level | None = None,
    ) -> Self:
        # [FUNCTION DOC]
        # Replace values where the condition is true (the inverse of the where
        # method).
        # Parameters: cond, other, inplace, axis, level as for where; cond may be a
        # callable.
        # When inplace is true and the chained-assignment warning has not been
        # disabled globally, the reference count of the object is inspected and, if it
        # indicates the call is made on a temporary (a chained assignment) rather than
        # a local variable of the caller, a chained-assignment warning is emitted at
        # stack level two.
        # Callable cond and other are evaluated with self. A condition lacking the
        # invert protocol (for example a plain list) is converted to a numpy array.
        # The inverted condition is passed to the shared where helper. Returns the new
        # object, or self when in place.
        """
        Replace values where the condition is True.

        Parameters
        ----------
        cond : bool Series/DataFrame, array-like, or callable
            Where `cond` is False, keep the original value. Where
            True, replace with corresponding value from `other`.
            If `cond` is callable, it is computed on the Series/DataFrame and
            should return boolean Series/DataFrame or array. The callable must
            not change input Series/DataFrame (though pandas doesn't check it).
        other : scalar, Series/DataFrame, or callable
            Entries where `cond` is True are replaced with
            corresponding value from `other`.
            If other is callable, it is computed on the Series/DataFrame and
            should return scalar or Series/DataFrame. The callable must not
            change input Series/DataFrame (though pandas doesn't check it).
            If not specified, entries will be filled with the corresponding
            NULL value (``np.nan`` for numpy dtypes, ``pd.NA`` for extension
            dtypes).
        inplace : bool, default False
            Whether to perform the operation in place on the data.
        axis : int, default None
            Alignment axis if needed. For `Series` this parameter is
            unused and defaults to 0.
        level : int, default None
            Alignment level if needed.

        Returns
        -------
        Series or DataFrame
            When applied to a Series, the function will return a Series,
            and when applied to a DataFrame, it will return a DataFrame.

        See Also
        --------
        :func:`DataFrame.where` : Return an object of same shape as caller.
        :func:`Series.where` : Return an object of same shape as caller.

        Notes
        -----
        The mask method is an application of the if-then idiom. For each
        element in the caller, if ``cond`` is ``False`` the
        element is used; otherwise the corresponding element from
        ``other`` is used. If the axis of ``other`` does not align with axis of
        ``cond`` Series/DataFrame, the values of ``cond`` on misaligned index positions
        will be filled with True.

        The signature for :func:`Series.where` or
        :func:`DataFrame.where` differs from :func:`numpy.where`.
        Roughly ``df1.where(m, df2)`` is equivalent to ``np.where(m, df1, df2)``.

        For further details and examples see the ``mask`` documentation in
        :ref:`indexing <indexing.where_mask>`.

        The dtype of the object takes precedence. The fill value is casted to
        the object's dtype, if this can be done losslessly.

        Examples
        --------
        >>> s = pd.Series(range(5))
        >>> s.where(s > 0)
        0    NaN
        1    1.0
        2    2.0
        3    3.0
        4    4.0
        dtype: float64
        >>> s.mask(s > 0)
        0    0.0
        1    NaN
        2    NaN
        3    NaN
        4    NaN
        dtype: float64

        >>> s = pd.Series(range(5))
        >>> t = pd.Series([True, False])
        >>> s.where(t, 99)
        0     0
        1    99
        2    99
        3    99
        4    99
        dtype: int64
        >>> s.mask(t, 99)
        0    99
        1     1
        2    99
        3    99
        4    99
        dtype: int64

        >>> s.where(s > 1, 10)
        0    10
        1    10
        2    2
        3    3
        4    4
        dtype: int64
        >>> s.mask(s > 1, 10)
        0     0
        1     1
        2    10
        3    10
        4    10
        dtype: int64

        >>> df = pd.DataFrame(np.arange(10).reshape(-1, 2), columns=["A", "B"])
        >>> df
           A  B
        0  0  1
        1  2  3
        2  4  5
        3  6  7
        4  8  9
        >>> m = df % 3 == 0
        >>> df.where(m, -df)
           A  B
        0  0 -1
        1 -2  3
        2 -4 -5
        3  6 -7
        4 -8  9
        >>> df.where(m, -df) == np.where(m, df, -df)
              A     B
        0  True  True
        1  True  True
        2  True  True
        3  True  True
        4  True  True
        >>> df.where(m, -df) == df.mask(~m, -df)
              A     B
        0  True  True
        1  True  True
        2  True  True
        3  True  True
        4  True  True
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        if inplace:
            if not CHAINED_WARNING_DISABLED:
                if sys.getrefcount(
                    self
                ) <= REF_COUNT_METHOD and not common.is_local_in_caller_frame(self):
                    warnings.warn(
                        _chained_assignment_method_msg,
                        ChainedAssignmentError,
                        stacklevel=2,
                    )

        cond = common.apply_if_callable(cond, self)
        other = common.apply_if_callable(other, self)

        # see gh-21891
        if not hasattr(cond, "__invert__"):
            cond = np.array(cond)

        return self._where(
            ~cond,
            other=other,
            inplace=inplace,
            axis=axis,
            level=level,
        )

    def shift(
        self,
        periods: int | Sequence[int] = 1,
        freq=None,
        axis: Axis = 0,
        fill_value: Hashable = lib.no_default,
        suffix: str | None = None,
    ) -> Self | DataFrame:
        # [FUNCTION DOC]
        # Shift data (or the index, when a frequency is given) by a number of periods.
        # Parameters: periods is an int (negative shifts backwards) or, for Series
        # only, a list of ints (a frame with one shifted column per period is
        # produced); freq is an offset or string, or infer, to shift the index instead
        # of the data; axis (default 0); fill_value is the value for newly created
        # positions (unset uses the dtype missing value); suffix is a column name
        # suffix used by subclasses in the multi-period case.
        # Raises a value error if both freq and fill_value are given. Zero periods
        # returns a shallow copy. A list of periods on a Series is handled by
        # converting to a frame and shifting that. Without freq the manager shifts the
        # data along the row axis (only axis zero is expected here; frames handle axis
        # one themselves) and the result is finalized with the method name shift. With
        # freq the index is shifted by the private helper.
        # KNOWN ISSUE: the suffix parameter is accepted but never used in this
        # implementation (it is not forwarded in the list-of-periods path either).
        # Returns the shifted object.
        """
        Shift index by desired number of periods with an optional time `freq`.

        When `freq` is not passed, shift the index without realigning the data.
        If `freq` is passed (in this case, the index must be date or datetime,
        or it will raise a `NotImplementedError`), the index will be
        increased using the periods and the `freq`. `freq` can be inferred
        when specified as "infer" as long as either freq or inferred_freq
        attribute is set in the index.

        Parameters
        ----------
        periods : int or Sequence
            Number of periods to shift. Can be positive or negative.
            If an iterable of ints, the data will be shifted once by each int.
            This is equivalent to shifting by one value at a time and
            concatenating all resulting frames. The resulting columns will have
            the shift suffixed to their column names. For multiple periods,
            axis must not be 1.
        freq : DateOffset, tseries.offsets, timedelta, or str, optional
            Offset to use from the tseries module or time rule (e.g. 'EOM').
            If `freq` is specified then the index values are shifted but the
            data is not realigned. That is, use `freq` if you would like to
            extend the index when shifting and preserve the original data.
            If `freq` is specified as "infer" then it will be inferred from
            the freq or inferred_freq attributes of the index. If neither of
            those attributes exist, a ValueError is thrown.
        axis : {0 or 'index', 1 or 'columns', None}, default None
            Shift direction. For `Series` this parameter is unused and defaults to 0.
        fill_value : object, optional
            The scalar value to use for newly introduced missing values.
            the default depends on the dtype of `self`.
            For Boolean and numeric NumPy data types, ``np.nan`` is used.
            For datetime, timedelta, or period data, etc. :attr:`NaT` is used.
            For extension dtypes, ``self.dtype.na_value`` is used.
        suffix : str, optional
            If str and periods is an iterable, this is added after the column
            name and before the shift value for each shifted column name.
            For `Series` this parameter is unused and defaults to `None`.

        Returns
        -------
        Series/DataFrame
            Copy of input object, shifted.

        See Also
        --------
        Index.shift : Shift values of Index.
        DatetimeIndex.shift : Shift values of DatetimeIndex.
        PeriodIndex.shift : Shift values of PeriodIndex.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     [[10, 13, 17], [20, 23, 27], [15, 18, 22], [30, 33, 37], [45, 48, 52]],
        ...     columns=["Col1", "Col2", "Col3"],
        ...     index=pd.date_range("2020-01-01", "2020-01-05"),
        ... )
        >>> df
                    Col1  Col2  Col3
        2020-01-01    10    13    17
        2020-01-02    20    23    27
        2020-01-03    15    18    22
        2020-01-04    30    33    37
        2020-01-05    45    48    52

        >>> df.shift(periods=3)
                    Col1  Col2  Col3
        2020-01-01   NaN   NaN   NaN
        2020-01-02   NaN   NaN   NaN
        2020-01-03   NaN   NaN   NaN
        2020-01-04  10.0  13.0  17.0
        2020-01-05  20.0  23.0  27.0

        >>> df.shift(periods=1, axis="columns")
                    Col1  Col2  Col3
        2020-01-01   NaN    10    13
        2020-01-02   NaN    20    23
        2020-01-03   NaN    15    18
        2020-01-04   NaN    30    33
        2020-01-05   NaN    45    48

        >>> df.shift(periods=3, fill_value=0)
                    Col1  Col2  Col3
        2020-01-01     0     0     0
        2020-01-02     0     0     0
        2020-01-03     0     0     0
        2020-01-04    10    13    17
        2020-01-05    20    23    27

        >>> df.shift(periods=3, freq="D")
                    Col1  Col2  Col3
        2020-01-04    10    13    17
        2020-01-05    20    23    27
        2020-01-06    15    18    22
        2020-01-07    30    33    37
        2020-01-08    45    48    52

        >>> df.shift(periods=3, freq="infer")
                    Col1  Col2  Col3
        2020-01-04    10    13    17
        2020-01-05    20    23    27
        2020-01-06    15    18    22
        2020-01-07    30    33    37
        2020-01-08    45    48    52

        >>> df["Col1"].shift(periods=[0, 1, 2])
                    Col1_0  Col1_1  Col1_2
        2020-01-01      10     NaN     NaN
        2020-01-02      20    10.0     NaN
        2020-01-03      15    20.0    10.0
        2020-01-04      30    15.0    20.0
        2020-01-05      45    30.0    15.0
        """
        axis = self._get_axis_number(axis)

        if freq is not None and fill_value is not lib.no_default:
            # GH#53832
            raise ValueError(
                "Passing a 'freq' together with a 'fill_value' is not allowed."
            )

        if periods == 0:
            return self.copy(deep=False)

        if is_list_like(periods) and isinstance(self, ABCSeries):
            return self.to_frame().shift(
                periods=periods, freq=freq, axis=axis, fill_value=fill_value
            )
        periods = cast(int, periods)

        if freq is None:
            # when freq is None, data is shifted, index is not
            axis = self._get_axis_number(axis)
            assert axis == 0  # axis == 1 cases handled in DataFrame.shift
            new_data = self._mgr.shift(periods=periods, fill_value=fill_value)
            return self._constructor_from_mgr(
                new_data, axes=new_data.axes
            ).__finalize__(self, method="shift")

        return self._shift_with_freq(periods, axis, freq)

    @final
    def _shift_with_freq(self, periods: int, axis: int, freq) -> Self:
        # [FUNCTION DOC]
        # HELPER: shifts the labels of an axis instead of the data.
        # Parameters: periods number of steps (int); axis integer axis; freq an
        # offset, string, or infer.
        # For infer the frequency is taken from the index freq attribute, then from
        # its inferred frequency, and a value error is raised if neither is available.
        # A string is converted to an offset (period-aware for PeriodIndex). For a
        # PeriodIndex a frequency not equal to the index frequency raises a value
        # error naming both; the index shift ignores freq in that case. Otherwise the
        # index is shifted by the given number of periods of the frequency. The new
        # axis is set on a copy, which is finalized with the method name shift and
        # returned.
        # see shift.__doc__
        # when freq is given, index is shifted, data is not
        index = self._get_axis(axis)

        if freq == "infer":
            freq = getattr(index, "freq", None)

            if freq is None:
                freq = getattr(index, "inferred_freq", None)

            if freq is None:
                msg = "Freq was not set in the index hence cannot be inferred"
                raise ValueError(msg)

        elif isinstance(freq, str):
            is_period = isinstance(index, PeriodIndex)
            freq = to_offset(freq, is_period=is_period)

        if isinstance(index, PeriodIndex):
            orig_freq = to_offset(index.freq)
            if freq != orig_freq:
                assert orig_freq is not None  # for mypy
                raise ValueError(
                    f"Given freq {PeriodDtype(freq)._freqstr} "
                    f"does not match PeriodIndex freq "
                    f"{PeriodDtype(orig_freq)._freqstr}"
                )
            new_ax: Index = index.shift(periods)
        else:
            new_ax = index.shift(periods, freq)

        result = self.set_axis(new_ax, axis=axis)
        return result.__finalize__(self, method="shift")

    @final
    def truncate(
        self,
        before=None,
        after=None,
        axis: Axis | None = None,
        copy: bool | lib.NoDefault = lib.no_default,
    ) -> Self:
        # [FUNCTION DOC]
        # Truncate to the rows (or columns) between two index values.
        # Parameters: before and after are the inclusive bounds (None means
        # unbounded); axis defaults to 0; copy is a deprecated placeholder.
        # Raises a value error if the axis is not monotonic (increasing or
        # decreasing), and if before is greater than after. If the axis consists of
        # date-like labels the bounds are converted with the datetime parser. For a
        # decreasing axis with more than one distinct value the bounds are swapped. A
        # label-based slice is taken along the axis; for a MultiIndex the axis is also
        # truncated by its own truncate method. Returns a shallow copy of the slice.
        """
        Truncate a Series or DataFrame before and after some index value.

        This is a useful shorthand for boolean indexing based on index
        values above or below certain thresholds.

        Parameters
        ----------
        before : date, str, int
            Truncate all rows before this index value.
        after : date, str, int
            Truncate all rows after this index value.
        axis : {0 or 'index', 1 or 'columns'}, optional
            Axis to truncate. Truncates the index (rows) by default.
            For `Series` this parameter is unused and defaults to 0.
        copy : bool, default False
            This keyword is now ignored; changing its value will have no
            impact on the method.

            .. deprecated:: 3.0.0

                This keyword is ignored and will be removed in pandas 4.0. Since
                pandas 3.0, this method always returns a new object using a lazy
                copy mechanism that defers copies until necessary
                (Copy-on-Write). See the `user guide on Copy-on-Write
                <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                for more details.

        Returns
        -------
        type of caller
            The truncated Series or DataFrame.

        See Also
        --------
        DataFrame.loc : Select a subset of a DataFrame by label.
        DataFrame.iloc : Select a subset of a DataFrame by position.

        Notes
        -----
        If the index being truncated contains only datetime values,
        `before` and `after` may be specified as strings instead of
        Timestamps.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "A": ["a", "b", "c", "d", "e"],
        ...         "B": ["f", "g", "h", "i", "j"],
        ...         "C": ["k", "l", "m", "n", "o"],
        ...     },
        ...     index=[1, 2, 3, 4, 5],
        ... )
        >>> df
           A  B  C
        1  a  f  k
        2  b  g  l
        3  c  h  m
        4  d  i  n
        5  e  j  o

        >>> df.truncate(before=2, after=4)
           A  B  C
        2  b  g  l
        3  c  h  m
        4  d  i  n

        The columns of a DataFrame can be truncated.

        >>> df.truncate(before="A", after="B", axis="columns")
           A  B
        1  a  f
        2  b  g
        3  c  h
        4  d  i
        5  e  j

        For Series, only rows can be truncated.

        >>> df["A"].truncate(before=2, after=4)
        2    b
        3    c
        4    d
        Name: A, dtype: str

        The index values in ``truncate`` can be datetimes or string
        dates.

        >>> dates = pd.date_range("2016-01-01", "2016-02-01", freq="s")
        >>> df = pd.DataFrame(index=dates, data={"A": 1})
        >>> df.tail()
                             A
        2016-01-31 23:59:56  1
        2016-01-31 23:59:57  1
        2016-01-31 23:59:58  1
        2016-01-31 23:59:59  1
        2016-02-01 00:00:00  1

        >>> df.truncate(
        ...     before=pd.Timestamp("2016-01-05"), after=pd.Timestamp("2016-01-10")
        ... ).tail()
                             A
        2016-01-09 23:59:56  1
        2016-01-09 23:59:57  1
        2016-01-09 23:59:58  1
        2016-01-09 23:59:59  1
        2016-01-10 00:00:00  1

        Because the index is a DatetimeIndex containing only dates, we can
        specify `before` and `after` as strings. They will be coerced to
        Timestamps before truncation.

        >>> df.truncate("2016-01-05", "2016-01-10").tail()
                             A
        2016-01-09 23:59:56  1
        2016-01-09 23:59:57  1
        2016-01-09 23:59:58  1
        2016-01-09 23:59:59  1
        2016-01-10 00:00:00  1

        Note that ``truncate`` assumes a 0 value for any unspecified time
        component (midnight). This differs from partial string slicing, which
        returns any partially matching dates.

        >>> df.loc["2016-01-05":"2016-01-10", :].tail()
                             A
        2016-01-10 23:59:55  1
        2016-01-10 23:59:56  1
        2016-01-10 23:59:57  1
        2016-01-10 23:59:58  1
        2016-01-10 23:59:59  1
        """
        self._check_copy_deprecation(copy)

        if axis is None:
            axis = 0
        axis = self._get_axis_number(axis)
        ax = self._get_axis(axis)

        # GH 17935
        # Check that index is sorted
        if not ax.is_monotonic_increasing and not ax.is_monotonic_decreasing:
            raise ValueError("truncate requires a sorted index")

        # if we have a date index, convert to dates, otherwise
        # treat like a slice
        if ax._is_all_dates:
            from pandas.core.tools.datetimes import to_datetime

            if before is not None:
                # Avoid converting to NaT
                before = to_datetime(before)
            if after is not None:
                # Avoid converting to NaT
                after = to_datetime(after)

        if before is not None and after is not None and before > after:
            raise ValueError(f"Truncate: {after} must be after {before}")

        if len(ax) > 1 and ax.is_monotonic_decreasing and ax.nunique() > 1:
            before, after = after, before

        slicer = [slice(None, None)] * self._AXIS_LEN
        slicer[axis] = slice(before, after)
        result = self.loc[tuple(slicer)]

        if isinstance(ax, MultiIndex):
            setattr(result, self._get_axis_name(axis), ax.truncate(before, after))

        result = result.copy(deep=False)

        return result

    @final
    def tz_convert(
        self,
        tz,
        axis: Axis = 0,
        level=None,
        copy: bool | lib.NoDefault = lib.no_default,
    ) -> Self:
        # [FUNCTION DOC]
        # Convert a time-zone aware axis to another time zone.
        # Parameters: tz is the target zone (string, tzinfo or None); axis (default
        # 0); level is the MultiIndex level to convert (required to be valid for a
        # MultiIndex); copy is a deprecated placeholder.
        # Raises a type error from the nested helper if a non-empty axis is not a
        # datetime-like index, and a value error if level is not None, 0 or the axis
        # name for a flat axis. Returns a shallow copy with the converted axis,
        # finalized with the method name tz_convert.
        """
        Convert tz-aware axis to target time zone.

        Parameters
        ----------
        tz : str or tzinfo object or None
            Target time zone. Passing ``None`` will convert to
            UTC and remove the timezone information.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The axis to convert
        level : int, str, default None
            If axis is a MultiIndex, convert a specific level. Otherwise
            must be None.
        copy : bool, default False
            This keyword is now ignored; changing its value will have no
            impact on the method.

            .. deprecated:: 3.0.0

                This keyword is ignored and will be removed in pandas 4.0. Since
                pandas 3.0, this method always returns a new object using a lazy
                copy mechanism that defers copies until necessary
                (Copy-on-Write). See the `user guide on Copy-on-Write
                <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                for more details.

        Returns
        -------
        Series/DataFrame
            Object with time zone converted axis.

        Raises
        ------
        TypeError
            If the axis is tz-naive.

        See Also
        --------
        DataFrame.tz_localize: Localize tz-naive index of DataFrame to target time zone.
        Series.tz_localize: Localize tz-naive index of Series to target time zone.

        Examples
        --------
        Change to another time zone:

        >>> s = pd.Series(
        ...     [1],
        ...     index=pd.DatetimeIndex(["2018-09-15 01:30:00+02:00"]),
        ... )
        >>> s.tz_convert("Asia/Shanghai")
        2018-09-15 07:30:00+08:00    1
        dtype: int64

        Pass None to convert to UTC and get a tz-naive index:

        >>> s = pd.Series([1], index=pd.DatetimeIndex(["2018-09-15 01:30:00+02:00"]))
        >>> s.tz_convert(None)
        2018-09-14 23:30:00    1
        dtype: int64
        """
        self._check_copy_deprecation(copy)
        axis = self._get_axis_number(axis)
        ax = self._get_axis(axis)

        def _tz_convert(ax, tz):
            # [FUNCTION DOC]
            # HELPER: nested function of tz_convert that converts one index to a time
            # zone.
            # Parameters: ax is an index (or MultiIndex level); tz is the target zone.
            # If the object has no convert method: an empty index becomes an empty
            # timezone-aware datetime index, otherwise a type error naming the axis is
            # raised. Returns the converted index.
            if not hasattr(ax, "tz_convert"):
                if len(ax) > 0:
                    ax_name = self._get_axis_name(axis)
                    raise TypeError(
                        f"{ax_name} is not a valid DatetimeIndex or PeriodIndex"
                    )
                ax = DatetimeIndex([], tz=tz)
            else:
                ax = ax.tz_convert(tz)
            return ax

        # if a level is given it must be a MultiIndex level or
        # equivalent to the axis name
        if isinstance(ax, MultiIndex):
            level = ax._get_level_number(level)
            new_level = _tz_convert(ax.levels[level], tz)
            ax = ax.set_levels(new_level, level=level)
        else:
            if level not in (None, 0, ax.name):
                raise ValueError(f"The level {level} is not valid")
            ax = _tz_convert(ax, tz)

        result = self.copy(deep=False)
        result = result.set_axis(ax, axis=axis)
        return result.__finalize__(self, method="tz_convert")

    @final
    def tz_localize(
        self,
        tz,
        axis: Axis = 0,
        level=None,
        copy: bool | lib.NoDefault = lib.no_default,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ) -> Self:
        # [FUNCTION DOC]
        # Localize a time-zone naive axis to a time zone.
        # Parameters: tz target zone (None removes localization); axis; level for
        # MultiIndex; copy deprecated placeholder; ambiguous is raise, infer, NaT, or
        # a boolean array handling repeated wall times; nonexistent is raise, NaT,
        # shift_forward, shift_backward or a timedelta handling skipped wall times.
        # Raises a value error for an invalid nonexistent option, for an invalid
        # level, and a type error from the nested helper if a non-empty axis is not
        # datetime-like. Returns a shallow copy with the localized axis, finalized
        # with the method name tz_localize.
        """
        Localize time zone naive index of a Series or DataFrame to target time zone.

        This operation localizes the Index. To localize the values in a
        time zone naive Series, use :meth:`Series.dt.tz_localize`.

        Parameters
        ----------
        tz : str or tzinfo or None
            Time zone to localize. Passing ``None`` will remove the
            time zone information and preserve local time.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The axis to localize
        level : int, str, default None
            If axis ia a MultiIndex, localize a specific level. Otherwise
            must be None.
        copy : bool, default False
            This keyword is now ignored; changing its value will have no
            impact on the method.

            .. deprecated:: 3.0.0

                This keyword is ignored and will be removed in pandas 4.0. Since
                pandas 3.0, this method always returns a new object using a lazy
                copy mechanism that defers copies until necessary
                (Copy-on-Write). See the `user guide on Copy-on-Write
                <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                for more details.

        ambiguous : 'infer', bool, bool-ndarray, 'NaT', default 'raise'
            When clocks moved backward due to DST, ambiguous times may arise.
            For example in Central European Time (UTC+01), when going from
            03:00 DST to 02:00 non-DST, 02:30:00 local time occurs both at
            00:30:00 UTC and at 01:30:00 UTC. In such a situation, the
            `ambiguous` parameter dictates how ambiguous times should be
            handled.

            - 'infer' will attempt to infer fall dst-transition hours based on
              order
            - bool (or bool-ndarray) where True signifies a DST time, False designates
              a non-DST time (note that this flag is only applicable for
              ambiguous times)
            - 'NaT' will return NaT where there are ambiguous times
            - 'raise' will raise a ValueError if there are ambiguous
              times.
        nonexistent : str, default 'raise'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST. Valid values are:

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
        Series/DataFrame
            Same type as the input, with time zone naive or aware index, depending on
            ``tz``.

        Raises
        ------
        TypeError
            If the TimeSeries is tz-aware and tz is not None.

        See Also
        --------
        Series.dt.tz_localize: Localize the values in a time zone naive Series.
        Timestamp.tz_localize: Localize the Timestamp to a timezone.

        Examples
        --------
        Localize local times:

        >>> s = pd.Series(
        ...     [1],
        ...     index=pd.DatetimeIndex(["2018-09-15 01:30:00"]),
        ... )
        >>> s.tz_localize("CET")
        2018-09-15 01:30:00+02:00    1
        dtype: int64

        Pass None to convert to tz-naive index and preserve local time:

        >>> s = pd.Series([1], index=pd.DatetimeIndex(["2018-09-15 01:30:00+02:00"]))
        >>> s.tz_localize(None)
        2018-09-15 01:30:00    1
        dtype: int64

        Be careful with DST changes. When there is sequential data, pandas
        can infer the DST time:

        >>> s = pd.Series(
        ...     range(7),
        ...     index=pd.DatetimeIndex(
        ...         [
        ...             "2018-10-28 01:30:00",
        ...             "2018-10-28 02:00:00",
        ...             "2018-10-28 02:30:00",
        ...             "2018-10-28 02:00:00",
        ...             "2018-10-28 02:30:00",
        ...             "2018-10-28 03:00:00",
        ...             "2018-10-28 03:30:00",
        ...         ]
        ...     ),
        ... )
        >>> s.tz_localize("CET", ambiguous="infer")
        2018-10-28 01:30:00+02:00    0
        2018-10-28 02:00:00+02:00    1
        2018-10-28 02:30:00+02:00    2
        2018-10-28 02:00:00+01:00    3
        2018-10-28 02:30:00+01:00    4
        2018-10-28 03:00:00+01:00    5
        2018-10-28 03:30:00+01:00    6
        dtype: int64

        In some cases, inferring the DST is impossible. In such cases, you can
        pass an ndarray to the ambiguous parameter to set the DST explicitly

        >>> s = pd.Series(
        ...     range(3),
        ...     index=pd.DatetimeIndex(
        ...         [
        ...             "2018-10-28 01:20:00",
        ...             "2018-10-28 02:36:00",
        ...             "2018-10-28 03:46:00",
        ...         ]
        ...     ),
        ... )
        >>> s.tz_localize("CET", ambiguous=np.array([True, True, False]))
        2018-10-28 01:20:00+02:00    0
        2018-10-28 02:36:00+02:00    1
        2018-10-28 03:46:00+01:00    2
        dtype: int64

        If the DST transition causes nonexistent times, you can shift these
        dates forward or backward with a timedelta object or `'shift_forward'`
        or `'shift_backward'`.

        >>> dti = pd.DatetimeIndex(
        ...     ["2015-03-29 02:30:00", "2015-03-29 03:30:00"], dtype="M8[ns]"
        ... )
        >>> s = pd.Series(range(2), index=dti)
        >>> s.tz_localize("Europe/Warsaw", nonexistent="shift_forward")
        2015-03-29 03:00:00+02:00    0
        2015-03-29 03:30:00+02:00    1
        dtype: int64
        >>> s.tz_localize("Europe/Warsaw", nonexistent="shift_backward")
        2015-03-29 01:59:59.999999999+01:00    0
        2015-03-29 03:30:00+02:00              1
        dtype: int64
        >>> s.tz_localize("Europe/Warsaw", nonexistent=pd.Timedelta("1h"))
        2015-03-29 03:30:00+02:00    0
        2015-03-29 03:30:00+02:00    1
        dtype: int64
        """
        self._check_copy_deprecation(copy)
        nonexistent_options = ("raise", "NaT", "shift_forward", "shift_backward")
        if nonexistent not in nonexistent_options and not isinstance(
            nonexistent, dt.timedelta
        ):
            raise ValueError(
                "The nonexistent argument must be one of 'raise', "
                "'NaT', 'shift_forward', 'shift_backward' or "
                "a timedelta object"
            )

        axis = self._get_axis_number(axis)
        ax = self._get_axis(axis)

        def _tz_localize(ax, tz, ambiguous, nonexistent):
            # [FUNCTION DOC]
            # HELPER: nested function of tz_localize that localizes one index.
            # Parameters: ax is an index; tz target zone; ambiguous and nonexistent
            # handling options forwarded.
            # If the object has no localization method: an empty index becomes an
            # empty timezone-aware datetime index, otherwise a type error naming the
            # axis is raised. Returns the localized index.
            if not hasattr(ax, "tz_localize"):
                if len(ax) > 0:
                    ax_name = self._get_axis_name(axis)
                    raise TypeError(
                        f"{ax_name} is not a valid DatetimeIndex or PeriodIndex"
                    )
                ax = DatetimeIndex([], tz=tz)
            else:
                ax = ax.tz_localize(tz, ambiguous=ambiguous, nonexistent=nonexistent)
            return ax

        # if a level is given it must be a MultiIndex level or
        # equivalent to the axis name
        if isinstance(ax, MultiIndex):
            level = ax._get_level_number(level)
            new_level = _tz_localize(ax.levels[level], tz, ambiguous, nonexistent)
            ax = ax.set_levels(new_level, level=level)
        else:
            if level not in (None, 0, ax.name):
                raise ValueError(f"The level {level} is not valid")
            ax = _tz_localize(ax, tz, ambiguous, nonexistent)

        result = self.copy(deep=False)
        result = result.set_axis(ax, axis=axis)
        return result.__finalize__(self, method="tz_localize")

    # ----------------------------------------------------------------------
    # Numeric Methods

    @final
    def describe(
        self,
        percentiles=None,
        include=None,
        exclude=None,
    ) -> Self:
        # [FUNCTION DOC]
        # Generate descriptive statistics.
        # Parameters: percentiles is a list of numbers between 0 and 1 (default
        # quartiles); include and exclude select dtypes (all, a list of dtypes or
        # None).
        # Delegates to the describe helper in the methods package and finalizes the
        # result with the method name describe. Returns a Series or DataFrame of
        # summary statistics.
        """
        Generate descriptive statistics.

        Descriptive statistics include those that summarize the central
        tendency, dispersion and shape of a
        dataset's distribution, excluding ``NaN`` values.

        Analyzes both numeric and object series, as well
        as ``DataFrame`` column sets of mixed data types. The output
        will vary depending on what is provided. Refer to the notes
        below for more detail.

        Parameters
        ----------
        percentiles : list-like of numbers, optional
            The percentiles to include in the output. All should
            fall between 0 and 1. The default, ``None``, will automatically
            return the 25th, 50th, and 75th percentiles.
        include : 'all', list-like of dtypes or None (default), optional
            A white list of data types to include in the result. Ignored
            for ``Series``. Here are the options:

            - 'all' : All columns of the input will be included in the output.
            - A list-like of dtypes : Limits the results to the
              provided data types.
              To limit the result to numeric types submit
              ``numpy.number``. To limit it instead to object columns submit
              the ``numpy.object`` data type. Strings
              can also be used in the style of
              ``select_dtypes`` (e.g. ``df.describe(include=['O'])``). To
              select pandas categorical columns, use ``'category'``
            - None (default) : The result will include all numeric columns.
        exclude : list-like of dtypes or None (default), optional,
            A black list of data types to omit from the result. Ignored
            for ``Series``. Here are the options:

            - A list-like of dtypes : Excludes the provided data types
              from the result. To exclude numeric types submit
              ``numpy.number``. To exclude object columns submit the data
              type ``numpy.object``. Strings can also be used in the style of
              ``select_dtypes`` (e.g. ``df.describe(exclude=['O'])``). To
              exclude pandas categorical columns, use ``'category'``
            - None (default) : The result will exclude nothing.

        Returns
        -------
        Series or DataFrame
            Summary statistics of the Series or Dataframe provided.

        See Also
        --------
        DataFrame.count: Count number of non-NA/null observations.
        DataFrame.max: Maximum of the values in the object.
        DataFrame.min: Minimum of the values in the object.
        DataFrame.mean: Mean of the values.
        DataFrame.std: Standard deviation of the observations.
        DataFrame.select_dtypes: Subset of a DataFrame including/excluding
            columns based on their dtype.

        Notes
        -----
        For numeric data, the result's index will include ``count``,
        ``mean``, ``std``, ``min``, ``max`` as well as lower, ``50`` and
        upper percentiles. By default the lower percentile is ``25`` and the
        upper percentile is ``75``. The ``50`` percentile is the
        same as the median.

        For object data (e.g. strings), the result's index
        will include ``count``, ``unique``, ``top``, and ``freq``. The ``top``
        is the most common value. The ``freq`` is the most common value's
        frequency.

        If multiple object values have the highest count, then the
        ``count`` and ``top`` results will be arbitrarily chosen from
        among those with the highest count.

        For mixed data types provided via a ``DataFrame``, the default is to
        return only an analysis of numeric columns. If the DataFrame consists
        only of object and categorical data without any numeric columns, the
        default is to return an analysis of both the object and categorical
        columns. If ``include='all'`` is provided as an option, the result
        will include a union of attributes of each type.

        The `include` and `exclude` parameters can be used to limit
        which columns in a ``DataFrame`` are analyzed for the output.
        The parameters are ignored when analyzing a ``Series``.

        Examples
        --------
        Describing a numeric ``Series``.

        >>> s = pd.Series([1, 2, 3])
        >>> s.describe()
        count    3.0
        mean     2.0
        std      1.0
        min      1.0
        25%      1.5
        50%      2.0
        75%      2.5
        max      3.0
        dtype: float64

        Describing a categorical ``Series``.

        >>> s = pd.Series(["a", "a", "b", "c"])
        >>> s.describe()
        count     4
        unique    3
        top       a
        freq      2
        dtype: object

        Describing a timestamp ``Series``.

        >>> s = pd.Series(
        ...     [
        ...         np.datetime64("2000-01-01"),
        ...         np.datetime64("2010-01-01"),
        ...         np.datetime64("2010-01-01"),
        ...     ]
        ... )
        >>> s.describe()
        count                      3
        mean     2006-09-01 08:00:00
        min      2000-01-01 00:00:00
        25%      2004-12-31 12:00:00
        50%      2010-01-01 00:00:00
        75%      2010-01-01 00:00:00
        max      2010-01-01 00:00:00
        dtype: object

        Describing a ``DataFrame``. By default only numeric fields
        are returned.

        >>> df = pd.DataFrame(
        ...     {
        ...         "categorical": pd.Categorical(["d", "e", "f"]),
        ...         "numeric": [1, 2, 3],
        ...         "object": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.describe()
               numeric
        count      3.0
        mean       2.0
        std        1.0
        min        1.0
        25%        1.5
        50%        2.0
        75%        2.5
        max        3.0

        Describing all columns of a ``DataFrame`` regardless of data type.

        >>> df.describe(include="all")  # doctest: +SKIP
               categorical  numeric object
        count            3      3.0      3
        unique           3      NaN      3
        top              f      NaN      a
        freq             1      NaN      1
        mean           NaN      2.0    NaN
        std            NaN      1.0    NaN
        min            NaN      1.0    NaN
        25%            NaN      1.5    NaN
        50%            NaN      2.0    NaN
        75%            NaN      2.5    NaN
        max            NaN      3.0    NaN

        Describing a column from a ``DataFrame`` by accessing it as
        an attribute.

        >>> df.numeric.describe()
        count    3.0
        mean     2.0
        std      1.0
        min      1.0
        25%      1.5
        50%      2.0
        75%      2.5
        max      3.0
        Name: numeric, dtype: float64

        Including only numeric columns in a ``DataFrame`` description.

        >>> df.describe(include=[np.number])
               numeric
        count      3.0
        mean       2.0
        std        1.0
        min        1.0
        25%        1.5
        50%        2.0
        75%        2.5
        max        3.0

        Including only string columns in a ``DataFrame`` description.

        >>> df.describe(include=[object])  # doctest: +SKIP
               object
        count       3
        unique      3
        top         a
        freq        1

        Including only categorical columns from a ``DataFrame`` description.

        >>> df.describe(include=["category"])
               categorical
        count            3
        unique           3
        top              d
        freq             1

        Excluding numeric columns from a ``DataFrame`` description.

        >>> df.describe(exclude=[np.number])  # doctest: +SKIP
               categorical object
        count            3      3
        unique           3      3
        top              f      a
        freq             1      1

        Excluding object columns from a ``DataFrame`` description.

        >>> df.describe(exclude=[object])  # doctest: +SKIP
               categorical  numeric
        count            3      3.0
        unique           3      NaN
        top              f      NaN
        freq             1      NaN
        mean           NaN      2.0
        std            NaN      1.0
        min            NaN      1.0
        25%            NaN      1.5
        50%            NaN      2.0
        75%            NaN      2.5
        max            NaN      3.0
        """
        return describe_ndframe(
            obj=self,
            include=include,
            exclude=exclude,
            percentiles=percentiles,
        ).__finalize__(self, method="describe")

    @final
    def pct_change(
        self,
        periods: int = 1,
        fill_method: None = None,
        freq=None,
        **kwargs,
    ) -> Self:
        # [FUNCTION DOC]
        # Fractional change between the current and a prior element.
        # Parameters: periods number of steps (default 1); fill_method must be None;
        # freq is an optional offset for time-based shifting; extra keyword arguments
        # may include axis (default index) and are passed to the shift.
        # Raises a value error if fill_method is anything other than None. Computes
        # the object divided by its shifted version minus one. With a frequency,
        # duplicated labels are dropped from the result and it is reindexed like the
        # original. Returns the new object finalized with the method name pct_change.
        """
        Fractional change between the current and a prior element.

        Computes the fractional change from the immediately previous row by
        default. This is useful in comparing the fraction of change in a time
        series of elements.

        .. note::

            Despite the name of this method, it calculates fractional change
            (also known as per unit change or relative change) and not
            percentage change. If you need the percentage change, multiply
            these values by 100.

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for forming percent change.
        fill_method : None
            Must be None. This argument will be removed in a future version of pandas.
        freq : DateOffset, timedelta, or str, optional
            Increment to use from time series API (e.g. 'ME' or BDay()).
        **kwargs
            Additional keyword arguments are passed into
            `DataFrame.shift` or `Series.shift`.

        Returns
        -------
        Series or DataFrame
            The same type as the calling object.

        See Also
        --------
        Series.diff : Compute the difference of two elements in a Series.
        DataFrame.diff : Compute the difference of two elements in a DataFrame.
        Series.shift : Shift the index by some number of periods.
        DataFrame.shift : Shift the index by some number of periods.

        Examples
        --------
        **Series**

        >>> s = pd.Series([90, 91, 85])
        >>> s
        0    90
        1    91
        2    85
        dtype: int64

        >>> s.pct_change()
        0         NaN
        1    0.011111
        2   -0.065934
        dtype: float64

        >>> s.pct_change(periods=2)
        0         NaN
        1         NaN
        2   -0.055556
        dtype: float64

        See the percentage change in a Series where filling NAs with last
        valid observation forward to next valid.

        >>> s = pd.Series([90, 91, None, 85])
        >>> s
        0    90.0
        1    91.0
        2     NaN
        3    85.0
        dtype: float64

        >>> s.ffill().pct_change()
        0         NaN
        1    0.011111
        2    0.000000
        3   -0.065934
        dtype: float64

        **DataFrame**

        Percentage change in French franc, Deutsche Mark, and Italian lira from
        1980-01-01 to 1980-03-01.

        >>> df = pd.DataFrame(
        ...     {
        ...         "FR": [4.0405, 4.0963, 4.3149],
        ...         "GR": [1.7246, 1.7482, 1.8519],
        ...         "IT": [804.74, 810.01, 860.13],
        ...     },
        ...     index=["1980-01-01", "1980-02-01", "1980-03-01"],
        ... )
        >>> df
                        FR      GR      IT
        1980-01-01  4.0405  1.7246  804.74
        1980-02-01  4.0963  1.7482  810.01
        1980-03-01  4.3149  1.8519  860.13

        >>> df.pct_change()
                          FR        GR        IT
        1980-01-01       NaN       NaN       NaN
        1980-02-01  0.013810  0.013684  0.006549
        1980-03-01  0.053365  0.059318  0.061876

        Percentage of change in GOOG and APPL stock volume. Shows computing
        the percentage change between columns.

        >>> df = pd.DataFrame(
        ...     {
        ...         "2016": [1769950, 30586265],
        ...         "2015": [1500923, 40912316],
        ...         "2014": [1371819, 41403351],
        ...     },
        ...     index=["GOOG", "APPL"],
        ... )
        >>> df
                  2016      2015      2014
        GOOG   1769950   1500923   1371819
        APPL  30586265  40912316  41403351

        >>> df.pct_change(axis="columns", periods=-1)
                  2016      2015  2014
        GOOG  0.179241  0.094112   NaN
        APPL -0.252395 -0.011860   NaN
        """
        # GH#53491
        if fill_method is not None:
            raise ValueError(f"fill_method must be None; got {fill_method=}.")

        axis = self._get_axis_number(kwargs.pop("axis", "index"))
        shifted = self.shift(periods=periods, freq=freq, axis=axis, **kwargs)
        # Unsupported left operand type for / ("Self")
        rs = self / shifted - 1  # type: ignore[operator]
        if freq is not None:
            # Shift method is implemented differently when freq is not None
            # We want to restore the original index
            rs = rs.loc[~rs.index.duplicated()]
            rs = rs.reindex_like(self)
        return rs.__finalize__(self, method="pct_change")

    @final
    def _logical_func(
        self,
        name: str,
        func,
        axis: Axis | None = 0,
        bool_only: bool = False,
        skipna: bool = True,
        **kwargs,
    ) -> Series | bool:
        # [FUNCTION DOC]
        # HELPER: shared implementation of any and all.
        # Parameters: name is the operation name (any or all); func is the null-aware
        # reduction function; axis (0, 1 or None); bool_only restrict to boolean data;
        # skipna; extra keyword arguments are validated for numpy compatibility (an
        # invalid one raises a type error).
        # For a frame with axis None the reduction is applied along axis 0 and then
        # again on the resulting Series to give a scalar. Axis None for
        # one-dimensional data is axis 0. A fast path for axis one avoids a transpose
        # when the frame has several blocks that are all two-dimensional and no extra
        # keywords exist. Otherwise the subclass reduction machinery is called with
        # the boolean filter type. Returns a Series or a bool.
        nv.validate_logical_func((), kwargs, fname=name)
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)

        if self.ndim > 1 and axis is None:
            # Reduce along one dimension then the other, to simplify DataFrame._reduce
            res = self._logical_func(
                name, func, axis=0, bool_only=bool_only, skipna=skipna, **kwargs
            )
            # error: Item "bool" of "Series | bool" has no attribute "_logical_func"
            return res._logical_func(  # type: ignore[union-attr]
                name, func, skipna=skipna, **kwargs
            )
        elif axis is None:
            axis = 0

        if (
            self.ndim > 1
            and axis == 1
            and len(self._mgr.blocks) > 1
            # TODO(EA2D): special-case not needed
            and all(block.values.ndim == 2 for block in self._mgr.blocks)
            and not kwargs
        ):
            # Fastpath avoiding potentially expensive transpose
            obj = self
            if bool_only:
                obj = self._get_bool_data()
            return obj._reduce_axis1(name, func, skipna=skipna)

        return self._reduce(
            func,
            name=name,
            axis=axis,
            skipna=skipna,
            numeric_only=bool_only,
            filter_type="bool",
        )

    def any(
        self,
        *,
        axis: Axis | None = 0,
        bool_only: bool = False,
        skipna: bool = True,
        **kwargs,
    ) -> Series | bool:
        # [FUNCTION DOC]
        # Return whether any element is true (no docstring here; it is attached by
        # subclasses). Parameters (keyword only): axis (default 0, None means all
        # axes); bool_only; skipna (default true); extra keywords for numpy
        # compatibility. Delegates to the shared logical helper with the null-aware
        # any reduction. Returns a Series or a bool.
        return self._logical_func(
            "any", nanops.nanany, axis, bool_only, skipna, **kwargs
        )

    def all(
        self,
        *,
        axis: Axis = 0,
        bool_only: bool = False,
        skipna: bool = True,
        **kwargs,
    ) -> Series | bool:
        # [FUNCTION DOC]
        # Return whether all elements are true. Parameters (keyword only): axis
        # (default 0); bool_only; skipna (default true); extra keywords for numpy
        # compatibility. Delegates to the shared logical helper with the null-aware
        # all reduction.
        # KNOWN ISSUE: the axis annotation excludes None whereas the corresponding any
        # method allows None; the shared helper handles None in both cases.
        # Returns a Series or a bool.
        return self._logical_func(
            "all", nanops.nanall, axis, bool_only, skipna, **kwargs
        )

    @final
    def _accum_func(
        self,
        name: str,
        func,
        axis: Axis | None = None,
        skipna: bool = True,
        *args,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # HELPER: shared implementation of the cumulative operations.
        # Parameters: name is the operation name; func is the numpy accumulate
        # function used for plain arrays; axis (default None means 0); skipna; extra
        # positional and keyword arguments are validated for numpy compatibility (also
        # extracting skipna).
        # Axis one is computed by transposing, accumulating along axis zero and
        # transposing back. Otherwise a nested function is applied to every block of
        # the manager. Returns a new object finalized with the method name equal to
        # the operation name.
        skipna = nv.validate_cum_func_with_skipna(skipna, args, kwargs, name)
        if axis is None:
            axis = 0
        else:
            axis = self._get_axis_number(axis)

        if axis == 1:
            return self.T._accum_func(
                name,
                func,
                axis=0,
                skipna=skipna,
                *args,  # noqa: B026
                **kwargs,
            ).T

        def block_accum_func(blk_values):
            # [FUNCTION DOC]
            # HELPER: nested function of the cumulative helper applied to each block.
            # Parameters: blk_values is the array of one block (transposed for
            # two-dimensional storage). Extension arrays use their own accumulate
            # method with the operation name and skipna; plain arrays use the
            # null-aware accumulate helper with the numpy function. The result is
            # transposed back if needed and returned.
            values = blk_values.T if hasattr(blk_values, "T") else blk_values

            result: np.ndarray | ExtensionArray
            if isinstance(values, ExtensionArray):
                result = values._accumulate(name, skipna=skipna, **kwargs)
            else:
                result = nanops.na_accum_func(values, func, skipna=skipna)

            result = result.T if hasattr(result, "T") else result
            return result

        result = self._mgr.apply(block_accum_func)

        return self._constructor_from_mgr(result, axes=result.axes).__finalize__(
            self, method=name
        )

    def cummax(self, axis: Axis = 0, skipna: bool = True, *args, **kwargs) -> Self:
        # [FUNCTION DOC]
        # Cumulative maximum (docstring attached by subclasses). Parameters: axis
        # (default 0); skipna (default true); extra positional and keyword arguments
        # for numpy compatibility. Delegates to the shared cumulative helper. Returns
        # an object of the same shape.
        return self._accum_func(
            "cummax", np.maximum.accumulate, axis, skipna, *args, **kwargs
        )

    def cummin(self, axis: Axis = 0, skipna: bool = True, *args, **kwargs) -> Self:
        # [FUNCTION DOC]
        # Cumulative minimum. Parameters: axis (default 0); skipna (default true);
        # extra positional and keyword arguments for numpy compatibility. Delegates to
        # the shared cumulative helper. Returns an object of the same shape.
        return self._accum_func(
            "cummin", np.minimum.accumulate, axis, skipna, *args, **kwargs
        )

    def cumsum(self, axis: Axis = 0, skipna: bool = True, *args, **kwargs) -> Self:
        # [FUNCTION DOC]
        # Cumulative sum. Parameters: axis (default 0); skipna (default true); extra
        # positional and keyword arguments for numpy compatibility. Delegates to the
        # shared cumulative helper. Returns an object of the same shape.
        return self._accum_func("cumsum", np.cumsum, axis, skipna, *args, **kwargs)

    def cumprod(self, axis: Axis = 0, skipna: bool = True, *args, **kwargs) -> Self:
        # [FUNCTION DOC]
        # Cumulative product. Parameters: axis (default 0); skipna (default true);
        # extra positional and keyword arguments for numpy compatibility. Delegates to
        # the shared cumulative helper. Returns an object of the same shape.
        return self._accum_func("cumprod", np.cumprod, axis, skipna, *args, **kwargs)

    @final
    def _stat_function_ddof(
        self,
        name: str,
        func,
        axis: Axis | None = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        **kwargs,
    ) -> Series | float:
        # [FUNCTION DOC]
        # HELPER: shared implementation of the reductions that take a
        # degrees-of-freedom argument (sem, var, std).
        # Parameters: name operation name; func null-aware reduction function; axis;
        # skipna (must be a real bool, None not allowed, otherwise a value error);
        # ddof divisor adjustment (default 1); numeric_only; extra keywords validated
        # for numpy compatibility.
        # Delegates to the subclass reduction with all arguments. Returns a Series or
        # a scalar.
        nv.validate_stat_ddof_func((), kwargs, fname=name)
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)

        return self._reduce(
            func, name, axis=axis, numeric_only=numeric_only, skipna=skipna, ddof=ddof
        )

    def sem(
        self,
        *,
        axis: Axis | None = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        **kwargs,
    ) -> Series | float:
        # [FUNCTION DOC]
        # Standard error of the mean. Keyword only parameters: axis (default 0),
        # skipna, ddof (default 1), numeric_only, extra keywords. Delegates to the
        # degrees-of-freedom helper with the null-aware sem function.
        return self._stat_function_ddof(
            "sem", nanops.nansem, axis, skipna, ddof, numeric_only, **kwargs
        )

    def var(
        self,
        *,
        axis: Axis | None = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        **kwargs,
    ) -> Series | float:
        # [FUNCTION DOC]
        # Variance. Keyword only parameters: axis (default 0), skipna, ddof (default
        # 1), numeric_only, extra keywords. Delegates to the degrees-of-freedom helper
        # with the null-aware var function.
        return self._stat_function_ddof(
            "var", nanops.nanvar, axis, skipna, ddof, numeric_only, **kwargs
        )

    def std(
        self,
        *,
        axis: Axis | None = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        **kwargs,
    ) -> Series | float:
        # [FUNCTION DOC]
        # Standard deviation. Keyword only parameters: axis (default 0), skipna, ddof
        # (default 1), numeric_only, extra keywords. Delegates to the
        # degrees-of-freedom helper with the null-aware std function.
        return self._stat_function_ddof(
            "std", nanops.nanstd, axis, skipna, ddof, numeric_only, **kwargs
        )

    @final
    def _stat_function(
        self,
        name: str,
        func,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # HELPER: shared implementation of median, mean, min, max, kurt and skew.
        # Parameters: name (asserted to be one of those six); func the null-aware
        # reduction; axis; skipna (must be a real bool); numeric_only; extra keywords
        # validated for numpy compatibility.
        # Delegates to the subclass reduction. Returns a Series or a scalar.
        assert name in ["median", "mean", "min", "max", "kurt", "skew"], name
        nv.validate_func(name, (), kwargs)

        validate_bool_kwarg(skipna, "skipna", none_allowed=False)

        return self._reduce(
            func, name=name, axis=axis, skipna=skipna, numeric_only=numeric_only
        )

    def min(
        self,
        *,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Minimum. Keyword only: axis (default 0), skipna, numeric_only, extra
        # keywords. Delegates to the shared statistic helper with the null-aware min.
        return self._stat_function(
            "min",
            nanops.nanmin,
            axis,
            skipna,
            numeric_only,
            **kwargs,
        )

    def max(
        self,
        *,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Maximum. Keyword only: axis (default 0), skipna, numeric_only, extra
        # keywords. Delegates to the shared statistic helper with the null-aware max.
        return self._stat_function(
            "max",
            nanops.nanmax,
            axis,
            skipna,
            numeric_only,
            **kwargs,
        )

    def mean(
        self,
        *,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ) -> Series | float:
        # [FUNCTION DOC]
        # Mean. Keyword only: axis (default 0), skipna, numeric_only, extra keywords.
        # Delegates to the shared statistic helper with the null-aware mean.
        return self._stat_function(
            "mean", nanops.nanmean, axis, skipna, numeric_only, **kwargs
        )

    def median(
        self,
        *,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ) -> Series | float:
        # [FUNCTION DOC]
        # Median. Keyword only: axis (default 0), skipna, numeric_only, extra
        # keywords. Delegates to the shared statistic helper with the null-aware
        # median.
        return self._stat_function(
            "median", nanops.nanmedian, axis, skipna, numeric_only, **kwargs
        )

    def skew(
        self,
        *,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ) -> Series | float:
        # [FUNCTION DOC]
        # Unbiased skewness. Keyword only: axis (default 0), skipna, numeric_only,
        # extra keywords. Delegates to the shared statistic helper with the null-aware
        # skew.
        return self._stat_function(
            "skew", nanops.nanskew, axis, skipna, numeric_only, **kwargs
        )

    def kurt(
        self,
        *,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ) -> Series | float:
        # [FUNCTION DOC]
        # Unbiased kurtosis (a module attribute below also exposes it under the longer
        # alias). Keyword only: axis (default 0), skipna, numeric_only, extra
        # keywords. Delegates to the shared statistic helper with the null-aware
        # kurtosis.
        return self._stat_function(
            "kurt", nanops.nankurt, axis, skipna, numeric_only, **kwargs
        )

    kurtosis = kurt

    @final
    def _min_count_stat_function(
        self,
        name: str,
        func,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        min_count: int = 0,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # HELPER: shared implementation of sum and product.
        # Parameters: name (asserted to be sum or prod); func the null-aware
        # reduction; axis; skipna (must be a real bool); numeric_only; min_count
        # required number of valid values for a non-missing result (default 0); extra
        # keywords validated for numpy compatibility.
        # Delegates to the subclass reduction passing min_count. Returns a Series or a
        # scalar.
        assert name in ["sum", "prod"], name
        nv.validate_func(name, (), kwargs)

        validate_bool_kwarg(skipna, "skipna", none_allowed=False)

        return self._reduce(
            func,
            name=name,
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            min_count=min_count,
        )

    def sum(
        self,
        *,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        min_count: int = 0,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Sum. Keyword only: axis (default 0), skipna, numeric_only, min_count
        # (default 0), extra keywords. Delegates to the min-count helper with the
        # null-aware sum.
        return self._min_count_stat_function(
            "sum", nanops.nansum, axis, skipna, numeric_only, min_count, **kwargs
        )

    def prod(
        self,
        *,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        min_count: int = 0,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Product. Keyword only: axis (default 0), skipna, numeric_only, min_count
        # (default 0), extra keywords. Delegates to the min-count helper with the
        # null-aware product. A module attribute below exposes it also under the alias
        # product.
        return self._min_count_stat_function(
            "prod",
            nanops.nanprod,
            axis,
            skipna,
            numeric_only,
            min_count,
            **kwargs,
        )

    product = prod

    @final
    def rolling(
        self,
        window: int | dt.timedelta | str | BaseOffset | BaseIndexer,
        min_periods: int | None = None,
        center: bool = False,
        win_type: str | None = None,
        on: str | None = None,
        closed: IntervalClosedType | None = None,
        step: int | None = None,
        method: str = "single",
    ) -> Window | Rolling:
        # [FUNCTION DOC]
        # Provide rolling window calculations.
        # Parameters: window is an integer size, a time-based size (timedelta, offset
        # or string), or a custom indexer object; min_periods is the minimum
        # observations needed (default the window size, or 1 for time-based windows);
        # center labels the window at its center; win_type names a weighting window
        # type (scipy); on names a column to use instead of the index for time-based
        # windows; closed says which window edges are closed; step evaluates every nth
        # window; method is single or table (execution method).
        # Creates a lazy window object. If win_type is given a weighted window object
        # is created; otherwise a plain rolling object. No computation occurs until an
        # aggregation is called. Argument validation happens in the window classes
        # (they raise value or type errors).
        """
        Provide rolling window calculations.

        Parameters
        ----------
        window : int, timedelta, str, offset, or BaseIndexer subclass
            Interval of the moving window.

            If an integer, the delta between the start and end of each window.
            The number of points in the window depends on the ``closed`` argument.

            If a timedelta, str, or offset, the time period of each window. Each
            window will be a variable sized based on the observations included in
            the time-period. This is only valid for datetimelike indexes.
            To learn more about the offsets & frequency strings, please see
            :ref:`this link<timeseries.offset_aliases>`.

            If a BaseIndexer subclass, the window boundaries
            based on the defined ``get_window_bounds`` method. Additional rolling
            keyword arguments, namely ``min_periods``, ``center``, ``closed`` and
            ``step`` will be passed to ``get_window_bounds``.

        min_periods : int, default None
            Minimum number of observations in window required to have a value;
            otherwise, result is ``np.nan``.

            For a window that is specified by an offset, ``min_periods`` will default
            to 1.

            For a window that is specified by an integer, ``min_periods`` will default
            to the size of the window.

        center : bool, default False
            If False, set the window labels as the right edge of the window index.

            If True, set the window labels as the center of the window index.

        win_type : str, default None
            If ``None``, all points are evenly weighted.

            If a string, it must be a valid `scipy.signal window function
            <https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows>`__.

            Certain Scipy window types require additional parameters to be passed
            in the aggregation function. The additional parameters must match
            the keywords specified in the Scipy window type method signature.

        on : str, optional
            For a DataFrame, a column label or Index level on which
            to calculate the rolling window, rather than the DataFrame's index.

            Provided integer column is ignored and excluded from result since
            an integer index is not used to calculate the rolling window.

        closed : str, default None
            Determines the inclusivity of points in the window

            If ``'right'``, uses the window (first, last] meaning the last point
            is included in the calculations.

            If ``'left'``, uses the window [first, last) meaning the first point
            is included in the calculations.

            If ``'both'``, uses the window [first, last] meaning all points in
            the window are included in the calculations.

            If ``'neither'``, uses the window (first, last) meaning the first
            and last points in the window are excluded from calculations.

            () and [] are referencing open and closed set
            notation respetively.

            Default ``None`` (``'right'``).

        step : int, default None
            Evaluate the window at every ``step`` result, equivalent to slicing as
            ``[::step]``. ``window`` must be an integer. Using a step argument other
            than None or 1 will produce a result with a different shape than the input.

        method : str {'single', 'table'}, default 'single'

            Execute the rolling operation per single column or row (``'single'``)
            or over the entire object (``'table'``).

            This argument is only implemented when specifying ``engine='numba'``
            in the method call.

        Returns
        -------
        pandas.api.typing.Window or pandas.api.typing.Rolling
            An instance of Window is returned if ``win_type`` is passed. Otherwise,
            an instance of Rolling is returned.

        See Also
        --------
        expanding : Provides expanding transformations.
        ewm : Provides exponential weighted functions.

        Notes
        -----
        See :ref:`Windowing Operations <window.generic>` for further usage details
        and examples.

        Examples
        --------
        >>> df = pd.DataFrame({"B": [0, 1, 2, np.nan, 4]})
        >>> df
             B
        0  0.0
        1  1.0
        2  2.0
        3  NaN
        4  4.0

        **window**

        Rolling sum with a window length of 2 observations.

        >>> df.rolling(2).sum()
             B
        0  NaN
        1  1.0
        2  3.0
        3  NaN
        4  NaN

        Rolling sum with a window span of 2 seconds.

        >>> df_time = pd.DataFrame(
        ...     {"B": [0, 1, 2, np.nan, 4]},
        ...     index=[
        ...         pd.Timestamp("20130101 09:00:00"),
        ...         pd.Timestamp("20130101 09:00:02"),
        ...         pd.Timestamp("20130101 09:00:03"),
        ...         pd.Timestamp("20130101 09:00:05"),
        ...         pd.Timestamp("20130101 09:00:06"),
        ...     ],
        ... )

        >>> df_time
                               B
        2013-01-01 09:00:00  0.0
        2013-01-01 09:00:02  1.0
        2013-01-01 09:00:03  2.0
        2013-01-01 09:00:05  NaN
        2013-01-01 09:00:06  4.0

        >>> df_time.rolling("2s").sum()
                               B
        2013-01-01 09:00:00  0.0
        2013-01-01 09:00:02  1.0
        2013-01-01 09:00:03  3.0
        2013-01-01 09:00:05  NaN
        2013-01-01 09:00:06  4.0

        Rolling sum with forward looking windows with 2 observations.

        >>> indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=2)
        >>> df.rolling(window=indexer, min_periods=1).sum()
             B
        0  1.0
        1  3.0
        2  2.0
        3  4.0
        4  4.0

        **min_periods**

        Rolling sum with a window length of 2 observations, but only needs a minimum
        of 1 observation to calculate a value.

        >>> df.rolling(2, min_periods=1).sum()
             B
        0  0.0
        1  1.0
        2  3.0
        3  2.0
        4  4.0

        **center**

        Rolling sum with the result assigned to the center of the window index.

        >>> df.rolling(3, min_periods=1, center=True).sum()
             B
        0  1.0
        1  3.0
        2  3.0
        3  6.0
        4  4.0

        >>> df.rolling(3, min_periods=1, center=False).sum()
             B
        0  0.0
        1  1.0
        2  3.0
        3  3.0
        4  6.0

        **step**

        Rolling sum with a window length of 2 observations, minimum of 1 observation to
        calculate a value, and a step of 2.

        >>> df.rolling(2, min_periods=1, step=2).sum()
             B
        0  0.0
        2  3.0
        4  4.0

        **win_type**

        Rolling sum with a window length of 2, using the Scipy ``'gaussian'``
        window type. ``std`` is required in the aggregation function.

        >>> df.rolling(2, win_type="gaussian").sum(std=3)
                  B
        0        NaN
        1   0.986207
        2   2.958621
        3        NaN
        4        NaN

        **on**

        Rolling sum with a window length of 2 days.

        >>> df = pd.DataFrame(
        ...     {
        ...         "A": [
        ...             pd.to_datetime("2020-01-01"),
        ...             pd.to_datetime("2020-01-01"),
        ...             pd.to_datetime("2020-01-02"),
        ...         ],
        ...         "B": [1, 2, 3],
        ...     },
        ...     index=pd.date_range("2020", periods=3),
        ... )

        >>> df
                            A  B
        2020-01-01 2020-01-01  1
        2020-01-02 2020-01-01  2
        2020-01-03 2020-01-02  3

        >>> df.rolling("2D", on="A").sum()
                            A    B
        2020-01-01 2020-01-01  1.0
        2020-01-02 2020-01-01  3.0
        2020-01-03 2020-01-02  6.0
        """
        if win_type is not None:
            return Window(
                self,
                window=window,
                min_periods=min_periods,
                center=center,
                win_type=win_type,
                on=on,
                closed=closed,
                step=step,
                method=method,
            )

        return Rolling(
            self,
            window=window,
            min_periods=min_periods,
            center=center,
            win_type=win_type,
            on=on,
            closed=closed,
            step=step,
            method=method,
        )

    @final
    def expanding(
        self,
        min_periods: int = 1,
        method: Literal["single", "table"] = "single",
    ) -> Expanding:
        # [FUNCTION DOC]
        # Provide expanding window calculations.
        # Parameters: min_periods minimum observations needed (default 1); method
        # single or table.
        # Returns a lazy expanding window object bound to self.
        """
        Provide expanding window calculations.

        An expanding window yields the value of an aggregation statistic with all
        the data available up to that point in time.

        Parameters
        ----------
        min_periods : int, default 1
            Minimum number of observations in window required to have a value;
            otherwise, result is ``np.nan``.

        method : str {'single', 'table'}, default 'single'
            Execute the rolling operation per single column or row (``'single'``)
            or over the entire object (``'table'``).

            This argument is only implemented when specifying ``engine='numba'``
            in the method call.

        Returns
        -------
        pandas.api.typing.Expanding
            An instance of Expanding for further expanding window calculations,
            e.g. using the ``sum`` method.

        See Also
        --------
        rolling : Provides rolling window calculations.
        ewm : Provides exponential weighted functions.

        Notes
        -----
        See :ref:`Windowing Operations <window.expanding>` for further usage details
        and examples.

        Examples
        --------
        >>> df = pd.DataFrame({"B": [0, 1, 2, np.nan, 4]})
        >>> df
             B
        0  0.0
        1  1.0
        2  2.0
        3  NaN
        4  4.0

        **min_periods**

        Expanding sum with 1 vs 3 observations needed to calculate a value.

        >>> df.expanding(1).sum()
             B
        0  0.0
        1  1.0
        2  3.0
        3  3.0
        4  7.0
        >>> df.expanding(3).sum()
             B
        0  NaN
        1  NaN
        2  3.0
        3  3.0
        4  7.0
        """
        return Expanding(self, min_periods=min_periods, method=method)

    @final
    @doc(ExponentialMovingWindow)
    def ewm(
        self,
        com: float | None = None,
        span: float | None = None,
        halflife: float | TimedeltaConvertibleTypes | None = None,
        alpha: float | None = None,
        min_periods: int | None = 0,
        adjust: bool = True,
        ignore_na: bool = False,
        times: np.ndarray | DataFrame | Series | None = None,
        method: Literal["single", "table"] = "single",
    ) -> ExponentialMovingWindow:
        # [FUNCTION DOC]
        # Provide exponentially weighted window calculations (docstring supplied by a
        # decorator copying it from the window class).
        # Parameters: com, span, halflife, alpha give mutually exclusive decay
        # specifications; min_periods minimum observations (default 0); adjust whether
        # to divide by decaying adjustment factor; ignore_na whether missing values
        # are ignored when computing weights; times optional timestamps for the
        # observations; method single or table.
        # Returns a lazy exponential window object bound to self. Argument validation
        # happens in the window class.
        return ExponentialMovingWindow(
            self,
            com=com,
            span=span,
            halflife=halflife,
            alpha=alpha,
            min_periods=min_periods,
            adjust=adjust,
            ignore_na=ignore_na,
            times=times,
            method=method,
        )

    # ----------------------------------------------------------------------
    # Arithmetic Methods

    @final
    def _inplace_method(self, other, op) -> Self:
        # [FUNCTION DOC]
        # HELPER: implements in-place arithmetic operators.
        # Parameters: other is the right operand; op is the unbound binary operation
        # of the class.
        # Computes the ordinary result, reindexes it to look like self, and replaces
        # the internals of self with it, so the object keeps its identity and its
        # axes. Returns self.
        """
        Wrap arithmetic method to operate inplace.
        """
        result = op(self, other)

        # this makes sure that we are aligned like the input
        # we are updating inplace
        self._update_inplace(result.reindex_like(self))
        return self

    @final
    def __iadd__(self, other) -> Self:
        # [FUNCTION DOC]
        # In-place addition. Delegates to the in-place helper with the class addition
        # operation. Returns self.
        # error: Unsupported left operand type for + ("Type[NDFrame]")
        return self._inplace_method(other, type(self).__add__)  # type: ignore[operator]

    @final
    def __isub__(self, other) -> Self:
        # [FUNCTION DOC]
        # In-place subtraction. Delegates to the in-place helper with the class
        # subtraction operation. Returns self.
        # error: Unsupported left operand type for - ("Type[NDFrame]")
        return self._inplace_method(other, type(self).__sub__)  # type: ignore[operator]

    @final
    def __imul__(self, other) -> Self:
        # [FUNCTION DOC]
        # In-place multiplication. Delegates to the in-place helper with the class
        # multiplication operation. Returns self.
        # error: Unsupported left operand type for * ("Type[NDFrame]")
        return self._inplace_method(other, type(self).__mul__)  # type: ignore[operator]

    @final
    def __itruediv__(self, other) -> Self:
        # [FUNCTION DOC]
        # In-place true division. Delegates to the in-place helper with the class true
        # division operation. Returns self.
        # error: Unsupported left operand type for / ("Type[NDFrame]")
        return self._inplace_method(
            other,
            type(self).__truediv__,  # type: ignore[operator]
        )

    @final
    def __ifloordiv__(self, other) -> Self:
        # [FUNCTION DOC]
        # In-place floor division. Delegates to the in-place helper with the class
        # floor division operation. Returns self.
        # error: Unsupported left operand type for // ("Type[NDFrame]")
        return self._inplace_method(
            other,
            type(self).__floordiv__,  # type: ignore[operator]
        )

    @final
    def __imod__(self, other) -> Self:
        # [FUNCTION DOC]
        # In-place modulo. Delegates to the in-place helper with the class modulo
        # operation. Returns self.
        # error: Unsupported left operand type for % ("Type[NDFrame]")
        return self._inplace_method(other, type(self).__mod__)  # type: ignore[operator]

    @final
    def __ipow__(self, other) -> Self:
        # [FUNCTION DOC]
        # In-place exponentiation. Delegates to the in-place helper with the class
        # power operation. Returns self.
        # error: Unsupported left operand type for ** ("Type[NDFrame]")
        return self._inplace_method(other, type(self).__pow__)  # type: ignore[operator]

    @final
    def __iand__(self, other) -> Self:
        # [FUNCTION DOC]
        # In-place bitwise and. Delegates to the in-place helper with the class and
        # operation. Returns self.
        # error: Unsupported left operand type for & ("Type[NDFrame]")
        return self._inplace_method(other, type(self).__and__)  # type: ignore[operator]

    @final
    def __ior__(self, other) -> Self:
        # [FUNCTION DOC]
        # In-place bitwise or. Delegates to the in-place helper with the class or
        # operation. Returns self.
        return self._inplace_method(other, type(self).__or__)

    @final
    def __ixor__(self, other) -> Self:
        # [FUNCTION DOC]
        # In-place bitwise exclusive or. Delegates to the in-place helper with the
        # class xor operation. Returns self.
        # error: Unsupported left operand type for ^ ("Type[NDFrame]")
        return self._inplace_method(other, type(self).__xor__)  # type: ignore[operator]

    # ----------------------------------------------------------------------
    # Misc methods

    @final
    def _find_valid_index(self, *, how: str) -> Hashable:
        # [FUNCTION DOC]
        # HELPER: finds the first or last label with a valid entry.
        # Parameters (keyword only): how is first or last.
        # Builds the array of non-missing flags, asks the missing-data helper for the
        # position of the first or last valid entry, and returns the index label at
        # that position, or None when there is none.
        """
        Retrieves the index of the first valid value.

        Parameters
        ----------
        how : {'first', 'last'}
            Use this parameter to change between the first or last valid index.

        Returns
        -------
        idx_first_valid : type of index
        """
        is_valid = self.notna().values
        idxpos = find_valid_index(how=how, is_valid=is_valid)
        if idxpos is None:
            return None
        return self.index[idxpos]

    @final
    def first_valid_index(self) -> Hashable:
        # [FUNCTION DOC]
        # Return the index label of the first non-missing value, or None if there is
        # none (also None for empty objects). Delegates to the private finder with the
        # first mode.
        """
        Return index for first non-missing value or None, if no value is found.

        See the :ref:`User Guide <missing_data>` for more information
        on which values are considered missing.

        Returns
        -------
        type of index
            Index of first non-missing value.

        See Also
        --------
        DataFrame.last_valid_index : Return index for last non-NA value or None, if
            no non-NA value is found.
        Series.last_valid_index : Return index for last non-NA value or None, if no
            non-NA value is found.
        DataFrame.isna : Detect missing values.

        Examples
        --------
        For Series:

        >>> s = pd.Series([None, 3, 4])
        >>> s.first_valid_index()
        1
        >>> s.last_valid_index()
        2

        >>> s = pd.Series([None, None])
        >>> print(s.first_valid_index())
        None
        >>> print(s.last_valid_index())
        None

        If all elements in Series are NA/null, returns None.

        >>> s = pd.Series()
        >>> print(s.first_valid_index())
        None
        >>> print(s.last_valid_index())
        None

        If Series is empty, returns None.

        For DataFrame:

        >>> df = pd.DataFrame({"A": [None, None, 2], "B": [None, 3, 4]})
        >>> df
             A      B
        0  NaN    NaN
        1  NaN    3.0
        2  2.0    4.0
        >>> df.first_valid_index()
        1
        >>> df.last_valid_index()
        2

        >>> df = pd.DataFrame({"A": [None, None, None], "B": [None, None, None]})
        >>> df
             A      B
        0  None   None
        1  None   None
        2  None   None
        >>> print(df.first_valid_index())
        None
        >>> print(df.last_valid_index())
        None

        If all elements in DataFrame are NA/null, returns None.

        >>> df = pd.DataFrame()
        >>> df
        Empty DataFrame
        Columns: []
        Index: []
        >>> print(df.first_valid_index())
        None
        >>> print(df.last_valid_index())
        None

        If DataFrame is empty, returns None.
        """
        return self._find_valid_index(how="first")

    @final
    def last_valid_index(self) -> Hashable:
        # [FUNCTION DOC]
        # Return the index label of the last non-missing value, or None if there is
        # none. Delegates to the private finder with the last mode.
        """
        Return index for last non-missing value or None, if no value is found.

        See the :ref:`User Guide <missing_data>` for more information
        on which values are considered missing.

        Returns
        -------
        type of index
            Index of last non-missing value.

        See Also
        --------
        DataFrame.first_valid_index : Return index for first non-NA value or None, if
            no non-NA value is found.
        Series.first_valid_index : Return index for first non-NA value or None, if no
            non-NA value is found.
        DataFrame.isna : Detect missing values.

        Examples
        --------
        For Series:

        >>> s = pd.Series([None, 3, 4])
        >>> s.first_valid_index()
        1
        >>> s.last_valid_index()
        2

        >>> s = pd.Series([None, None])
        >>> print(s.first_valid_index())
        None
        >>> print(s.last_valid_index())
        None

        If all elements in Series are NA/null, returns None.

        >>> s = pd.Series()
        >>> print(s.first_valid_index())
        None
        >>> print(s.last_valid_index())
        None

        If Series is empty, returns None.

        For DataFrame:

        >>> df = pd.DataFrame({"A": [None, None, 2], "B": [None, 3, 4]})
        >>> df
             A      B
        0  NaN    NaN
        1  NaN    3.0
        2  2.0    4.0
        >>> df.first_valid_index()
        1
        >>> df.last_valid_index()
        2

        >>> df = pd.DataFrame({"A": [None, None, None], "B": [None, None, None]})
        >>> df
             A      B
        0  None   None
        1  None   None
        2  None   None
        >>> print(df.first_valid_index())
        None
        >>> print(df.last_valid_index())
        None

        If all elements in DataFrame are NA/null, returns None.

        >>> df = pd.DataFrame()
        >>> df
        Empty DataFrame
        Columns: []
        Index: []
        >>> print(df.first_valid_index())
        None
        >>> print(df.last_valid_index())
        None

        If DataFrame is empty, returns None.
        """
        return self._find_valid_index(how="last")


_num_doc = """
{desc}

Parameters
----------
axis : {axis_descr}
    Axis for the function to be applied on.
    For `Series` this parameter is unused and defaults to 0.

    For DataFrames, specifying ``axis=None`` will apply the aggregation
    across both axes.

    .. versionadded:: 2.0.0

skipna : bool, default True
    Exclude NA/null values when computing the result.
numeric_only : bool, default False
    Include only float, int, boolean columns.

{min_count}\
**kwargs
    Additional keyword arguments to be passed to the function.

Returns
-------
{name1} or scalar\

    Value containing the calculation referenced in the description.\
{see_also}\
{examples}
"""

_sum_prod_doc = """
{desc}

Parameters
----------
axis : {axis_descr}
    Axis for the function to be applied on.
    For `Series` this parameter is unused and defaults to 0.

    .. warning::

        The behavior of DataFrame.{name} with ``axis=None`` is deprecated,
        in a future version this will reduce over both axes and return a scalar
        To retain the old behavior, pass axis=0 (or do not pass axis).

    .. versionadded:: 2.0.0

skipna : bool, default True
    Exclude NA/null values when computing the result.
numeric_only : bool, default False
    Include only float, int, boolean columns. Not implemented for Series.

{min_count}\
**kwargs
    Additional keyword arguments to be passed to the function.

Returns
-------
{name1} or scalar\

    Value containing the calculation referenced in the description.\
{see_also}\
{examples}
"""

_num_ddof_doc = """
{desc}

Parameters
----------
axis : {axis_descr}
    For `Series` this parameter is unused and defaults to 0.

    .. warning::

        The behavior of DataFrame.{name} with ``axis=None`` is deprecated,
        in a future version this will reduce over both axes and return a scalar
        To retain the old behavior, pass axis=0 (or do not pass axis).

skipna : bool, default True
    Exclude NA/null values. If an entire row/column is NA, the result
    will be NA.
ddof : int, default 1
    Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
    where N represents the number of elements.
numeric_only : bool, default False
    Include only float, int, boolean columns. Not implemented for Series.
**kwargs :
    Additional keywords have no effect but might be accepted
    for compatibility with NumPy.

Returns
-------
{name1} or {name2} (if level specified)
    {return_desc}

See Also
--------
{see_also}\
{notes}\
{examples}
"""

_sem_see_also = """\
scipy.stats.sem : Compute standard error of the mean.
{name2}.std : Return sample standard deviation over requested axis.
{name2}.var : Return unbiased variance over requested axis.
{name2}.mean : Return the mean of the values over the requested axis.
{name2}.median : Return the median of the values over the requested axis.
{name2}.mode : Return the mode(s) of the Series."""

_sem_return_desc = """\
Unbiased standard error of the mean over requested axis."""

_std_see_also = """\
numpy.std : Compute the standard deviation along the specified axis.
{name2}.var : Return unbiased variance over requested axis.
{name2}.sem : Return unbiased standard error of the mean over requested axis.
{name2}.mean : Return the mean of the values over the requested axis.
{name2}.median : Return the median of the values over the requested axis.
{name2}.mode : Return the mode(s) of the Series."""

_std_return_desc = """\
Standard deviation over requested axis."""

_std_notes = """

Notes
-----
To have the same behaviour as `numpy.std`, use `ddof=0` (instead of the
default `ddof=1`)"""

_std_examples = """

Examples
--------
>>> df = pd.DataFrame({'person_id': [0, 1, 2, 3],
...                    'age': [21, 25, 62, 43],
...                    'height': [1.61, 1.87, 1.49, 2.01]}
...                   ).set_index('person_id')
>>> df
           age  height
person_id
0           21    1.61
1           25    1.87
2           62    1.49
3           43    2.01

The standard deviation of the columns can be found as follows:

>>> df.std()
age       18.786076
height     0.237417
dtype: float64

Alternatively, `ddof=0` can be set to normalize by N instead of N-1:

>>> df.std(ddof=0)
age       16.269219
height     0.205609
dtype: float64"""

_var_examples = """

Examples
--------
>>> df = pd.DataFrame({'person_id': [0, 1, 2, 3],
...                    'age': [21, 25, 62, 43],
...                    'height': [1.61, 1.87, 1.49, 2.01]}
...                   ).set_index('person_id')
>>> df
           age  height
person_id
0           21    1.61
1           25    1.87
2           62    1.49
3           43    2.01

>>> df.var()
age       352.916667
height      0.056367
dtype: float64

Alternatively, ``ddof=0`` can be set to normalize by N instead of N-1:

>>> df.var(ddof=0)
age       264.687500
height      0.042275
dtype: float64"""

_bool_doc = """
{desc}

Parameters
----------
axis : {{0 or 'index', 1 or 'columns', None}}, default 0
    Indicate which axis or axes should be reduced. For `Series` this parameter
    is unused and defaults to 0.

    * 0 / 'index' : reduce the index, return a Series whose index is the
      original column labels.
    * 1 / 'columns' : reduce the columns, return a Series whose index is the
      original index.
    * None : reduce all axes, return a scalar.

bool_only : bool, default False
    Include only boolean columns. Not implemented for Series.
skipna : bool, default True
    Exclude NA/null values. If the entire row/column is NA and skipna is
    True, then the result will be {empty_value}, as for an empty row/column.
    If skipna is False, then NA are treated as True, because these are not
    equal to zero.
**kwargs : any, default None
    Additional keywords have no effect but might be accepted for
    compatibility with NumPy.

Returns
-------
{name2} or {name1}
    If axis=None, then a scalar boolean is returned.
    Otherwise a Series is returned with index matching the index argument.

{see_also}
{examples}"""

_all_desc = """\
Return whether all elements are True, potentially over an axis.

Returns True unless there at least one element within a series or
along a Dataframe axis that is False or equivalent (e.g. zero or
empty)."""

_all_examples = """\
Examples
--------
**Series**

>>> pd.Series([True, True]).all()
True
>>> pd.Series([True, False]).all()
False
>>> pd.Series([], dtype="float64").all()
True
>>> pd.Series([np.nan]).all()
True
>>> pd.Series([np.nan]).all(skipna=False)
True

**DataFrames**

Create a DataFrame from a dictionary.

>>> df = pd.DataFrame({'col1': [True, True], 'col2': [True, False]})
>>> df
   col1   col2
0  True   True
1  True  False

Default behaviour checks if values in each column all return True.

>>> df.all()
col1     True
col2    False
dtype: bool

Specify ``axis='columns'`` to check if values in each row all return True.

>>> df.all(axis='columns')
0     True
1    False
dtype: bool

Or ``axis=None`` for whether every value is True.

>>> df.all(axis=None)
False
"""

_all_see_also = """\
See Also
--------
Series.all : Return True if all elements are True.
DataFrame.any : Return True if one (or more) elements are True.
"""

_cnum_pd_doc = """
Return cumulative {desc} over a DataFrame or Series axis.

Returns a DataFrame or Series of the same size containing the cumulative
{desc}.

Parameters
----------
axis : {{0 or 'index', 1 or 'columns'}}, default 0
    The index or the name of the axis. 0 is equivalent to None or 'index'.
    For `Series` this parameter is unused and defaults to 0.
skipna : bool, default True
    Exclude NA/null values. If an entire row/column is NA, the result
    will be NA.
numeric_only : bool, default False
    Include only float, int, boolean columns.
*args, **kwargs
    Additional keywords have no effect but might be accepted for
    compatibility with NumPy.

Returns
-------
{name1} or {name2}
    Return cumulative {desc} of {name1} or {name2}.

See Also
--------
core.window.expanding.Expanding.{accum_func_name} : Similar functionality
    but ignores ``NaN`` values.
{name2}.{accum_func_name} : Return the {desc} over
    {name2} axis.
{name2}.cummax : Return cumulative maximum over {name2} axis.
{name2}.cummin : Return cumulative minimum over {name2} axis.
{name2}.cumsum : Return cumulative sum over {name2} axis.
{name2}.cumprod : Return cumulative product over {name2} axis.

{examples}"""

_cnum_series_doc = """
Return cumulative {desc} over a DataFrame or Series axis.

Returns a DataFrame or Series of the same size containing the cumulative
{desc}.

Parameters
----------
axis : {{0 or 'index', 1 or 'columns'}}, default 0
    The index or the name of the axis. 0 is equivalent to None or 'index'.
    For `Series` this parameter is unused and defaults to 0.
skipna : bool, default True
    Exclude NA/null values. If an entire row/column is NA, the result
    will be NA.
*args, **kwargs
    Additional keywords have no effect but might be accepted for
    compatibility with NumPy.

Returns
-------
{name1} or {name2}
    Return cumulative {desc} of {name1} or {name2}.

See Also
--------
core.window.expanding.Expanding.{accum_func_name} : Similar functionality
    but ignores ``NaN`` values.
{name2}.{accum_func_name} : Return the {desc} over
    {name2} axis.
{name2}.cummax : Return cumulative maximum over {name2} axis.
{name2}.cummin : Return cumulative minimum over {name2} axis.
{name2}.cumsum : Return cumulative sum over {name2} axis.
{name2}.cumprod : Return cumulative product over {name2} axis.

{examples}"""

_cummin_examples = """\
Examples
--------
**Series**

>>> s = pd.Series([2, np.nan, 5, -1, 0])
>>> s
0    2.0
1    NaN
2    5.0
3   -1.0
4    0.0
dtype: float64

By default, NA values are ignored.

>>> s.cummin()
0    2.0
1    NaN
2    2.0
3   -1.0
4   -1.0
dtype: float64

To include NA values in the operation, use ``skipna=False``

>>> s.cummin(skipna=False)
0    2.0
1    NaN
2    NaN
3    NaN
4    NaN
dtype: float64

**DataFrame**

>>> df = pd.DataFrame([[2.0, 1.0],
...                    [3.0, np.nan],
...                    [1.0, 0.0]],
...                   columns=list('AB'))
>>> df
     A    B
0  2.0  1.0
1  3.0  NaN
2  1.0  0.0

By default, iterates over rows and finds the minimum
in each column. This is equivalent to ``axis=None`` or ``axis='index'``.

>>> df.cummin()
     A    B
0  2.0  1.0
1  2.0  NaN
2  1.0  0.0

To iterate over columns and find the minimum in each row,
use ``axis=1``

>>> df.cummin(axis=1)
     A    B
0  2.0  1.0
1  3.0  NaN
2  1.0  0.0
"""

_cumsum_examples = """\
Examples
--------
**Series**

>>> s = pd.Series([2, np.nan, 5, -1, 0])
>>> s
0    2.0
1    NaN
2    5.0
3   -1.0
4    0.0
dtype: float64

By default, NA values are ignored.

>>> s.cumsum()
0    2.0
1    NaN
2    7.0
3    6.0
4    6.0
dtype: float64

To include NA values in the operation, use ``skipna=False``

>>> s.cumsum(skipna=False)
0    2.0
1    NaN
2    NaN
3    NaN
4    NaN
dtype: float64

**DataFrame**

>>> df = pd.DataFrame([[2.0, 1.0],
...                    [3.0, np.nan],
...                    [1.0, 0.0]],
...                   columns=list('AB'))
>>> df
     A    B
0  2.0  1.0
1  3.0  NaN
2  1.0  0.0

By default, iterates over rows and finds the sum
in each column. This is equivalent to ``axis=None`` or ``axis='index'``.

>>> df.cumsum()
     A    B
0  2.0  1.0
1  5.0  NaN
2  6.0  1.0

To iterate over columns and find the sum in each row,
use ``axis=1``

>>> df.cumsum(axis=1)
     A    B
0  2.0  3.0
1  3.0  NaN
2  1.0  1.0
"""

_cumprod_examples = """\
Examples
--------
**Series**

>>> s = pd.Series([2, np.nan, 5, -1, 0])
>>> s
0    2.0
1    NaN
2    5.0
3   -1.0
4    0.0
dtype: float64

By default, NA values are ignored.

>>> s.cumprod()
0     2.0
1     NaN
2    10.0
3   -10.0
4    -0.0
dtype: float64

To include NA values in the operation, use ``skipna=False``

>>> s.cumprod(skipna=False)
0    2.0
1    NaN
2    NaN
3    NaN
4    NaN
dtype: float64

**DataFrame**

>>> df = pd.DataFrame([[2.0, 1.0],
...                    [3.0, np.nan],
...                    [1.0, 0.0]],
...                   columns=list('AB'))
>>> df
     A    B
0  2.0  1.0
1  3.0  NaN
2  1.0  0.0

By default, iterates over rows and finds the product
in each column. This is equivalent to ``axis=None`` or ``axis='index'``.

>>> df.cumprod()
     A    B
0  2.0  1.0
1  6.0  NaN
2  6.0  0.0

To iterate over columns and find the product in each row,
use ``axis=1``

>>> df.cumprod(axis=1)
     A    B
0  2.0  2.0
1  3.0  NaN
2  1.0  0.0
"""

_cummax_examples = """\
Examples
--------
**Series**

>>> s = pd.Series([2, np.nan, 5, -1, 0])
>>> s
0    2.0
1    NaN
2    5.0
3   -1.0
4    0.0
dtype: float64

By default, NA values are ignored.

>>> s.cummax()
0    2.0
1    NaN
2    5.0
3    5.0
4    5.0
dtype: float64

To include NA values in the operation, use ``skipna=False``

>>> s.cummax(skipna=False)
0    2.0
1    NaN
2    NaN
3    NaN
4    NaN
dtype: float64

**DataFrame**

>>> df = pd.DataFrame([[2.0, 1.0],
...                    [3.0, np.nan],
...                    [1.0, 0.0]],
...                   columns=list('AB'))
>>> df
     A    B
0  2.0  1.0
1  3.0  NaN
2  1.0  0.0

By default, iterates over rows and finds the maximum
in each column. This is equivalent to ``axis=None`` or ``axis='index'``.

>>> df.cummax()
     A    B
0  2.0  1.0
1  3.0  NaN
2  3.0  1.0

To iterate over columns and find the maximum in each row,
use ``axis=1``

>>> df.cummax(axis=1)
     A    B
0  2.0  2.0
1  3.0  NaN
2  1.0  1.0
"""

_any_see_also = """\
See Also
--------
numpy.any : Numpy version of this method.
Series.any : Return whether any element is True.
Series.all : Return whether all elements are True.
DataFrame.any : Return whether any element is True over requested axis.
DataFrame.all : Return whether all elements are True over requested axis.
"""

_any_desc = """\
Return whether any element is True, potentially over an axis.

Returns False unless there is at least one element within a series or
along a Dataframe axis that is True or equivalent (e.g. non-zero or
non-empty)."""

_any_examples = """\
Examples
--------
**Series**

For Series input, the output is a scalar indicating whether any element
is True.

>>> pd.Series([False, False]).any()
False
>>> pd.Series([True, False]).any()
True
>>> pd.Series([], dtype="float64").any()
False
>>> pd.Series([np.nan]).any()
False
>>> pd.Series([np.nan]).any(skipna=False)
True

**DataFrame**

Whether each column contains at least one True element (the default).

>>> df = pd.DataFrame({"A": [1, 2], "B": [0, 2], "C": [0, 0]})
>>> df
   A  B  C
0  1  0  0
1  2  2  0

>>> df.any()
A     True
B     True
C    False
dtype: bool

Aggregating over the columns.

>>> df = pd.DataFrame({"A": [True, False], "B": [1, 2]})
>>> df
       A  B
0   True  1
1  False  2

>>> df.any(axis='columns')
0    True
1    True
dtype: bool

>>> df = pd.DataFrame({"A": [True, False], "B": [1, 0]})
>>> df
       A  B
0   True  1
1  False  0

>>> df.any(axis='columns')
0    True
1    False
dtype: bool

Aggregating over the entire DataFrame with ``axis=None``.

>>> df.any(axis=None)
True

`any` for an empty DataFrame is an empty Series.

>>> pd.DataFrame([]).any()
Series([], dtype: bool)
"""

_shared_docs["stat_func_example"] = """

Examples
--------
>>> idx = pd.MultiIndex.from_arrays([
...     ['warm', 'warm', 'cold', 'cold'],
...     ['dog', 'falcon', 'fish', 'spider']],
...     names=['blooded', 'animal'])
>>> s = pd.Series([4, 2, 0, 8], name='legs', index=idx)
>>> s
blooded  animal
warm     dog       4
         falcon    2
cold     fish      0
         spider    8
Name: legs, dtype: int64

>>> s.{stat_func}()
{default_output}"""

_sum_examples = _shared_docs["stat_func_example"].format(
    stat_func="sum", verb="Sum", default_output=14, level_output_0=6, level_output_1=8
)

_sum_examples += """

By default, the sum of an empty or all-NA Series is ``0``.

>>> pd.Series([], dtype="float64").sum()  # min_count=0 is the default
0.0

This can be controlled with the ``min_count`` parameter. For example, if
you'd like the sum of an empty series to be NaN, pass ``min_count=1``.

>>> pd.Series([], dtype="float64").sum(min_count=1)
nan

Thanks to the ``skipna`` parameter, ``min_count`` handles all-NA and
empty series identically.

>>> pd.Series([np.nan]).sum()
0.0

>>> pd.Series([np.nan]).sum(min_count=1)
nan"""

_max_examples: str = _shared_docs["stat_func_example"].format(
    stat_func="max", verb="Max", default_output=8, level_output_0=4, level_output_1=8
)

_min_examples: str = _shared_docs["stat_func_example"].format(
    stat_func="min", verb="Min", default_output=0, level_output_0=2, level_output_1=0
)

_skew_see_also = """

See Also
--------
Series.skew : Return unbiased skew over requested axis.
Series.var : Return unbiased variance over requested axis.
Series.std : Return unbiased standard deviation over requested axis."""

_stat_func_see_also = """

See Also
--------
Series.sum : Return the sum.
Series.min : Return the minimum.
Series.max : Return the maximum.
Series.idxmin : Return the index of the minimum.
Series.idxmax : Return the index of the maximum.
DataFrame.sum : Return the sum over the requested axis.
DataFrame.min : Return the minimum over the requested axis.
DataFrame.max : Return the maximum over the requested axis.
DataFrame.idxmin : Return the index of the minimum over the requested axis.
DataFrame.idxmax : Return the index of the maximum over the requested axis."""

_prod_examples = """

Examples
--------
By default, the product of an empty or all-NA Series is ``1``

>>> pd.Series([], dtype="float64").prod()
1.0

This can be controlled with the ``min_count`` parameter

>>> pd.Series([], dtype="float64").prod(min_count=1)
nan

Thanks to the ``skipna`` parameter, ``min_count`` handles all-NA and
empty series identically.

>>> pd.Series([np.nan]).prod()
1.0

>>> pd.Series([np.nan]).prod(min_count=1)
nan"""

_min_count_stub = """\
min_count : int, default 0
    The required number of valid values to perform the operation. If fewer than
    ``min_count`` non-NA values are present the result will be NA.
"""


def make_doc(name: str, ndim: int) -> str:
    # [FUNCTION DOC]
    # Generate the docstring text for a reduction or accumulation.
    # Parameters: name is one of any, all, min, max, sum, prod, median, mean, var,
    # std, sem, skew, kurt, cumsum, cumprod, cummin, cummax; ndim is 1 for Series
    # (scalar results) or anything else for DataFrame (Series results).
    # Selects the matching template, description, see-also, examples and extra
    # substitution values for the name and formats the template with the object kind
    # names and axis description. Raises a not-implemented error for an unknown name.
    # Returns the formatted string.
    # KNOWN ISSUE: nothing in the repository calls this function.
    # KNOWN ISSUE: for the variance, the chosen template contains a placeholder for
    # the return description but no value for it is supplied, so formatting would fail
    # with a missing-key error; the notes constant for the standard deviation template
    # is defined but an empty notes value is passed instead.
    """
    Generate the docstring for a Series/DataFrame reduction.
    """
    if ndim == 1:
        name1 = "scalar"
        name2 = "Series"
        axis_descr = "{index (0)}"
    else:
        name1 = "Series"
        name2 = "DataFrame"
        axis_descr = "{index (0), columns (1)}"

    if name == "any":
        base_doc = _bool_doc
        desc = _any_desc
        see_also = _any_see_also
        examples = _any_examples
        kwargs = {"empty_value": "False"}
    elif name == "all":
        base_doc = _bool_doc
        desc = _all_desc
        see_also = _all_see_also
        examples = _all_examples
        kwargs = {"empty_value": "True"}
    elif name == "min":
        base_doc = _num_doc
        desc = (
            "Return the minimum of the values over the requested axis.\n\n"
            "If you want the *index* of the minimum, use ``idxmin``. This is "
            "the equivalent of the ``numpy.ndarray`` method ``argmin``."
        )
        see_also = _stat_func_see_also
        examples = _min_examples
        kwargs = {"min_count": ""}
    elif name == "max":
        base_doc = _num_doc
        desc = (
            "Return the maximum of the values over the requested axis.\n\n"
            "If you want the *index* of the maximum, use ``idxmax``. This is "
            "the equivalent of the ``numpy.ndarray`` method ``argmax``."
        )
        see_also = _stat_func_see_also
        examples = _max_examples
        kwargs = {"min_count": ""}

    elif name == "sum":
        base_doc = _sum_prod_doc
        desc = (
            "Return the sum of the values over the requested axis.\n\n"
            "This is equivalent to the method ``numpy.sum``."
        )
        see_also = _stat_func_see_also
        examples = _sum_examples
        kwargs = {"min_count": _min_count_stub}

    elif name == "prod":
        base_doc = _sum_prod_doc
        desc = "Return the product of the values over the requested axis."
        see_also = _stat_func_see_also
        examples = _prod_examples
        kwargs = {"min_count": _min_count_stub}

    elif name == "median":
        base_doc = _num_doc
        desc = "Return the median of the values over the requested axis."
        see_also = _stat_func_see_also
        examples = """

            Examples
            --------
            >>> s = pd.Series([1, 2, 3])
            >>> s.median()
            2.0

            With a DataFrame

            >>> df = pd.DataFrame({'a': [1, 2], 'b': [2, 3]}, index=['tiger', 'zebra'])
            >>> df
                   a   b
            tiger  1   2
            zebra  2   3
            >>> df.median()
            a   1.5
            b   2.5
            dtype: float64

            Using axis=1

            >>> df.median(axis=1)
            tiger   1.5
            zebra   2.5
            dtype: float64

            In this case, `numeric_only` should be set to `True`
            to avoid getting an error.

            >>> df = pd.DataFrame({'a': [1, 2], 'b': ['T', 'Z']},
            ...                   index=['tiger', 'zebra'])
            >>> df.median(numeric_only=True)
            a   1.5
            dtype: float64"""
        kwargs = {"min_count": ""}

    elif name == "mean":
        base_doc = _num_doc
        desc = "Return the mean of the values over the requested axis."
        see_also = _stat_func_see_also
        examples = """

            Examples
            --------
            >>> s = pd.Series([1, 2, 3])
            >>> s.mean()
            2.0

            With a DataFrame

            >>> df = pd.DataFrame({'a': [1, 2], 'b': [2, 3]}, index=['tiger', 'zebra'])
            >>> df
                   a   b
            tiger  1   2
            zebra  2   3
            >>> df.mean()
            a   1.5
            b   2.5
            dtype: float64

            Using axis=1

            >>> df.mean(axis=1)
            tiger   1.5
            zebra   2.5
            dtype: float64

            In this case, `numeric_only` should be set to `True` to avoid
            getting an error.

            >>> df = pd.DataFrame({'a': [1, 2], 'b': ['T', 'Z']},
            ...                   index=['tiger', 'zebra'])
            >>> df.mean(numeric_only=True)
            a   1.5
            dtype: float64"""
        kwargs = {"min_count": ""}

    elif name == "var":
        base_doc = _num_ddof_doc
        desc = (
            "Return unbiased variance over requested axis.\n\nNormalized by "
            "N-1 by default. This can be changed using the ddof argument."
        )
        examples = _var_examples
        see_also = ""
        kwargs = {"notes": ""}

    elif name == "std":
        base_doc = _num_ddof_doc
        desc = (
            "Return sample standard deviation over requested axis."
            "\n\nNormalized by N-1 by default. This can be changed using the "
            "ddof argument."
        )
        examples = _std_examples
        see_also = _std_see_also.format(name2=name2)
        kwargs = {"notes": "", "return_desc": _std_return_desc}

    elif name == "sem":
        base_doc = _num_ddof_doc
        desc = (
            "Return unbiased standard error of the mean over requested "
            "axis.\n\nNormalized by N-1 by default. This can be changed "
            "using the ddof argument"
        )
        examples = """

            Examples
            --------
            >>> s = pd.Series([1, 2, 3])
            >>> round(s.sem(), 6)
            0.57735

            With a DataFrame

            >>> df = pd.DataFrame({'a': [1, 2], 'b': [2, 3]}, index=['tiger', 'zebra'])
            >>> df
                   a   b
            tiger  1   2
            zebra  2   3
            >>> df.sem()
            a   0.5
            b   0.5
            dtype: float64

            Using axis=1

            >>> df.sem(axis=1)
            tiger   0.5
            zebra   0.5
            dtype: float64

            In this case, `numeric_only` should be set to `True`
            to avoid getting an error.

            >>> df = pd.DataFrame({'a': [1, 2], 'b': ['T', 'Z']},
            ...                   index=['tiger', 'zebra'])
            >>> df.sem(numeric_only=True)
            a   0.5
            dtype: float64"""
        see_also = _sem_see_also.format(name2=name2)
        kwargs = {"notes": "", "return_desc": _sem_return_desc}

    elif name == "skew":
        base_doc = _num_doc
        desc = "Return unbiased skew over requested axis.\n\nNormalized by N-1."
        see_also = _skew_see_also
        examples = """

            Examples
            --------
            >>> s = pd.Series([1, 2, 3])
            >>> s.skew()
            0.0

            With a DataFrame

            >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4], 'c': [1, 3, 5]},
            ...                   index=['tiger', 'zebra', 'cow'])
            >>> df
                    a   b   c
            tiger   1   2   1
            zebra   2   3   3
            cow     3   4   5
            >>> df.skew()
            a   0.0
            b   0.0
            c   0.0
            dtype: float64

            Using axis=1

            >>> df.skew(axis=1)
            tiger   1.732051
            zebra  -1.732051
            cow     0.000000
            dtype: float64

            In this case, `numeric_only` should be set to `True` to avoid
            getting an error.

            >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': ['T', 'Z', 'X']},
            ...                   index=['tiger', 'zebra', 'cow'])
            >>> df.skew(numeric_only=True)
            a   0.0
            dtype: float64"""
        kwargs = {"min_count": ""}

    elif name == "kurt":
        base_doc = _num_doc
        desc = (
            "Return unbiased kurtosis over requested axis.\n\n"
            "Kurtosis obtained using Fisher's definition of\n"
            "kurtosis (kurtosis of normal == 0.0). Normalized "
            "by N-1."
        )
        see_also = ""
        examples = """

            Examples
            --------
            >>> s = pd.Series([1, 2, 2, 3], index=['cat', 'dog', 'dog', 'mouse'])
            >>> s
            cat    1
            dog    2
            dog    2
            mouse  3
            dtype: int64
            >>> s.kurt()
            1.5

            With a DataFrame

            >>> df = pd.DataFrame({'a': [1, 2, 2, 3], 'b': [3, 4, 4, 4]},
            ...                   index=['cat', 'dog', 'dog', 'mouse'])
            >>> df
                   a   b
              cat  1   3
              dog  2   4
              dog  2   4
            mouse  3   4
            >>> df.kurt()
            a   1.5
            b   4.0
            dtype: float64

            With axis=None

            >>> df.kurt(axis=None)
            -0.9886927196984727

            Using axis=1

            >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [3, 4], 'd': [1, 2]},
            ...                   index=['cat', 'dog'])
            >>> df.kurt(axis=1)
            cat   -6.0
            dog   -6.0
            dtype: float64"""
        kwargs = {"min_count": ""}

    elif name == "cumsum":
        if ndim == 1:
            base_doc = _cnum_series_doc
        else:
            base_doc = _cnum_pd_doc

        desc = "sum"
        see_also = ""
        examples = _cumsum_examples
        kwargs = {"accum_func_name": "sum"}

    elif name == "cumprod":
        if ndim == 1:
            base_doc = _cnum_series_doc
        else:
            base_doc = _cnum_pd_doc

        desc = "product"
        see_also = ""
        examples = _cumprod_examples
        kwargs = {"accum_func_name": "prod"}

    elif name == "cummin":
        if ndim == 1:
            base_doc = _cnum_series_doc
        else:
            base_doc = _cnum_pd_doc

        desc = "minimum"
        see_also = ""
        examples = _cummin_examples
        kwargs = {"accum_func_name": "min"}

    elif name == "cummax":
        if ndim == 1:
            base_doc = _cnum_series_doc
        else:
            base_doc = _cnum_pd_doc

        desc = "maximum"
        see_also = ""
        examples = _cummax_examples
        kwargs = {"accum_func_name": "max"}

    else:
        raise NotImplementedError

    docstr = base_doc.format(
        desc=desc,
        name=name,
        name1=name1,
        name2=name2,
        axis_descr=axis_descr,
        see_also=see_also,
        examples=examples,
        **kwargs,
    )
    return docstr
