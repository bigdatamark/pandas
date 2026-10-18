# [MODULE DOC]
# Machinery behind apply, agg (aggregate) and transform for Series, DataFrame,
# groupby, window and resample objects.
# Public contents: BaseExecutionEngine (abstract base for pluggable map/apply
# execution engines, exposed in the executors namespace), frame_apply (factory
# choosing the row-wise or column-wise frame applier), the abstract Apply base
# class and its subclasses (NDFrameApply, FrameApply with FrameRowApply and
# FrameColumnApply, SeriesApply, GroupByApply, ResamplerWindowApply), and helper
# functions for relabeling named aggregations (reconstruct_func,
# is_multi_agg_with_relabel, normalize_keyword_aggregation, relabel_result,
# reconstruct_and_relabel_result, maybe_mangle_lambdas, validate_func_kwargs), plus
# private helpers for making repeated names unique and mangling anonymous
# functions, and include_axis. Also defines the ResType alias for result mappings.
# Relationships: used by the apply/agg/transform methods of the pandas
# objects; may delegate to the numba executor module. Import-time side effects: none.
from __future__ import annotations

import abc
from collections import defaultdict
from collections.abc import Callable
import functools
from functools import partial
import inspect
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeAlias,
    cast,
)

import numpy as np

from pandas._libs.internals import BlockValuesRefs
from pandas._typing import (
    AggFuncType,
    AggFuncTypeBase,
    AggFuncTypeDict,
    AggObjType,
    Axis,
    AxisInt,
    NDFrameT,
    npt,
)
from pandas.compat._optional import import_optional_dependency
from pandas.errors import SpecificationError
from pandas.util._decorators import (
    cache_readonly,
    set_module,
)

from pandas.core.dtypes.cast import is_nested_object
from pandas.core.dtypes.common import (
    is_dict_like,
    is_extension_array_dtype,
    is_list_like,
    is_numeric_dtype,
    is_sequence,
)
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCNDFrame,
    ABCSeries,
)

from pandas.core._numba.executor import generate_apply_looper
import pandas.core.common as com
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.util.numba_ import (
    get_jit_arguments,
    prepare_function_arguments,
)

if TYPE_CHECKING:
    from collections.abc import (
        Generator,
        Hashable,
        Iterable,
        MutableMapping,
        Sequence,
    )

    from pandas import (
        DataFrame,
        Index,
        Series,
    )
    from pandas.core.groupby import GroupBy
    from pandas.core.resample import Resampler
    from pandas.core.window.rolling import BaseWindow

ResType: TypeAlias = dict[int, Any]


@set_module("pandas.api.executors")
class BaseExecutionEngine(abc.ABC):
    # [CLASS DOC]
    # Abstract base class for third-party execution engines for the map and apply
    # methods. Stateless; subclasses implement two static methods: map (run a
    # function elementwise) and apply (run a function along an axis). Exposed
    # through the executors public namespace.
    """
    Base class for execution engines for map and apply methods.

    An execution engine receives all the parameters of a call to
    ``apply`` or ``map``, such as the data container, the function,
    etc. and takes care of running the execution.

    Supporting different engines allows functions to be JIT compiled,
    run in parallel, and others. Besides the default executor which
    simply runs the code with the Python interpreter and pandas.
    """

    @staticmethod
    @abc.abstractmethod
    def map(
        data: Series | DataFrame | np.ndarray,
        func: AggFuncType,
        args: tuple,
        kwargs: dict[str, Any],
        decorator: Callable | None,
        skip_na: bool,
    ):
        # [FUNCTION DOC]
        # Abstract static method; engines must implement elementwise execution of
        # the function over the data. Inputs: data, func, args, kwargs, decorator
        # (optional JIT decorator) and skip_na (whether missing values are skipped).
        # The base implementation has no behavior (returns nothing).
        """
        Executor method to run functions elementwise.

        In general, pandas uses ``map`` for running functions elementwise,
        but ``Series.apply`` with the default ``by_row='compat'`` will also
        call this executor function.

        Parameters
        ----------
        data : Series, DataFrame or NumPy ndarray
            The object to use for the data. Some methods implement a ``raw``
            parameter which will convert the original pandas object to a
            NumPy array, which will then be passed here to the executor.
        func : function or NumPy ufunc
            The function to execute.
        args : tuple
            Positional arguments to be passed to ``func``.
        kwargs : dict
            Keyword arguments to be passed to ``func``.
        decorator : function, optional
            For JIT compilers and other engines that need to decorate the
            function ``func``, this is the decorator to use. While the
            executor may already know which is the decorator to use, this
            is useful as for a single executor the user can specify for
            example ``numba.jit`` or ``numba.njit(nogil=True)``, and this
            decorator parameter will contain the exact decorator from the
            executor the user wants to use.
        skip_na : bool
            Whether the function should be called for missing values or not.
            This is specified by the pandas user as ``map(na_action=None)``
            or ``map(na_action='ignore')``.
        """

    @staticmethod
    @abc.abstractmethod
    def apply(
        data: Series | DataFrame | np.ndarray,
        func: AggFuncType,
        args: tuple,
        kwargs: dict[str, Any],
        decorator: Callable,
        axis: Axis,
    ):
        # [FUNCTION DOC]
        # Abstract static method; engines must implement execution of the function
        # along an axis of the data. Inputs: data, func, args, kwargs, decorator and
        # axis (zero means columns are passed to the function, one means rows). The
        # base implementation has no behavior (returns nothing).
        """
        Executor method to run functions by an axis.

        While we can see ``map`` as executing the function for each cell
        in a ``DataFrame`` (or ``Series``), ``apply`` will execute the
        function for each column (or row).

        Parameters
        ----------
        data : Series, DataFrame or NumPy ndarray
            The object to use for the data. Some methods implement a ``raw``
            parameter which will convert the original pandas object to a
            NumPy array, which will then be passed here to the executor.
        func : function or NumPy ufunc
            The function to execute.
        args : tuple
            Positional arguments to be passed to ``func``.
        kwargs : dict
            Keyword arguments to be passed to ``func``.
        decorator : function, optional
            For JIT compilers and other engines that need to decorate the
            function ``func``, this is the decorator to use. While the
            executor may already know which is the decorator to use, this
            is useful as for a single executor the user can specify for
            example ``numba.jit`` or ``numba.njit(nogil=True)``, and this
            decorator parameter will contain the exact decorator from the
            executor the user wants to use.
        axis : {0 or 'index', 1 or 'columns'}
            0 or 'index' should execute the function passing each column as
            parameter. 1 or 'columns' should execute the function passing
            each row as parameter. The default executor engine passes rows
            as pandas ``Series``. Other executor engines should probably
            expect functions to be implemented this way for compatibility.
            But passing rows as other data structures is technically possible
            as far as the function ``func`` is implemented accordingly.
        """


def frame_apply(
    obj: DataFrame,
    func: AggFuncType,
    axis: Axis = 0,
    raw: bool = False,
    result_type: str | None = None,
    by_row: Literal[False, "compat"] = "compat",
    engine: str = "python",
    engine_kwargs: dict[str, bool] | None = None,
    args=None,
    kwargs=None,
) -> FrameApply:
    # [FUNCTION DOC]
    # Factory for the frame applier. Inputs: obj (DataFrame), func (function, string,
    # list or dict), axis (default 0), raw (default false: pass raw arrays), result_type
    # (default none), by_row (false or the compatibility marker), engine (default
    # python), engine_kwargs, args (positional arguments for func) and kwargs
    # (keyword arguments; also searched for named-aggregation specifications).
    # Behavior: the func is run through the relabeling reconstruction to detect named
    # aggregation (columns); the axis is normalized to a number; axis zero builds a
    # row-wise applier, axis one a column-wise applier (named aggregation with axis
    # one raises a not-implemented error). Returns the applier instance.
    # KNOWN ISSUE: kwargs defaults to none but is unpacked unconditionally in the
    # first line, which would fail for that default; callers always pass a mapping.
    """construct and return a row or column based frame apply object"""
    _, func, columns, _ = reconstruct_func(func, **kwargs)

    axis = obj._get_axis_number(axis)
    klass: type[FrameApply]
    if axis == 0:
        klass = FrameRowApply
    elif axis == 1:
        if columns:
            raise NotImplementedError(
                f"Named aggregation is not supported when {axis=}."
            )
        klass = FrameColumnApply

    return klass(
        obj,
        func,
        raw=raw,
        result_type=result_type,
        by_row=by_row,
        engine=engine,
        engine_kwargs=engine_kwargs,
        args=args,
        kwargs=kwargs,
    )


class Apply(metaclass=abc.ABCMeta):
    # [CLASS DOC]
    # Abstract base class holding the shared logic for apply, aggregate and
    # transform. State: obj (the object being operated on), func (the user
    # function, string, list or dict), raw, result_type, by_row mode, args and
    # kwargs (extra arguments, defaulting to empty), engine and engine_kwargs, and
    # the axis annotation (set by subclasses). Subclasses implement apply and the
    # list-like/dict-like aggregation or apply hooks; this class provides the
    # dispatching for agg and transform, the list/dict computation and result
    # wrapping, and string-method application.
    axis: AxisInt

    def __init__(
        self,
        obj: AggObjType,
        func: AggFuncType,
        raw: bool,
        result_type: str | None,
        *,
        by_row: Literal[False, "compat", "_compat"] = "compat",
        engine: str = "python",
        engine_kwargs: dict[str, bool] | None = None,
        args,
        kwargs,
    ) -> None:
        # [FUNCTION DOC]
        # Stores the configuration. Inputs: obj, func, raw, result_type (must be
        # none or one of reduce, broadcast, expand; otherwise a value error is
        # raised), keyword-only by_row (false, compat or an internal compat marker;
        # anything else fails an assertion), engine (default python), engine_kwargs
        # (default none stored as an empty dict), and args and kwargs (none becomes
        # empty tuple / dict). No computation is done.
        self.obj = obj
        self.raw = raw

        assert by_row is False or by_row in ["compat", "_compat"]
        self.by_row = by_row

        self.args = args or ()
        self.kwargs = kwargs or {}

        self.engine = engine
        self.engine_kwargs = {} if engine_kwargs is None else engine_kwargs

        if result_type not in [None, "reduce", "broadcast", "expand"]:
            raise ValueError(
                "invalid value for result_type, must be one "
                "of {None, 'reduce', 'broadcast', 'expand'}"
            )

        self.result_type = result_type

        self.func = func

    @abc.abstractmethod
    def apply(self) -> DataFrame | Series:
        # [FUNCTION DOC]
        # Abstract: subclasses implement the main apply operation and return the
        # result as a Series or DataFrame.
        pass

    @abc.abstractmethod
    def agg_or_apply_list_like(
        self, op_name: Literal["agg", "apply"]
    ) -> DataFrame | Series:
        # [FUNCTION DOC]
        # Abstract: subclasses run a list-like of functions for the named operation
        # (aggregate or apply) and return the combined result.
        pass

    @abc.abstractmethod
    def agg_or_apply_dict_like(
        self, op_name: Literal["agg", "apply"]
    ) -> DataFrame | Series:
        # [FUNCTION DOC]
        # Abstract: subclasses run a dict-like of functions for the named operation
        # (aggregate or apply) and return the combined result.
        pass

    def agg(self) -> DataFrame | Series | None:
        # [FUNCTION DOC]
        # Dispatches aggregation on the type of the function. A string runs the
        # string-method path; a dict-like runs the dict aggregation; any other
        # list-like runs the list aggregation. Returns the result, or none for a
        # callable (or other) so the caller can handle it itself.
        """
        Provide an implementation for the aggregators.

        Returns
        -------
        Result of aggregation, or None if agg cannot be performed by
        this method.
        """
        func = self.func

        if isinstance(func, str):
            return self.apply_str()

        if is_dict_like(func):
            return self.agg_dict_like()
        elif is_list_like(func):
            # we require a list, but not a 'str'
            return self.agg_list_like()

        # caller can react
        return None

    def transform(self) -> DataFrame | Series:
        # [FUNCTION DOC]
        # Applies a transform (a result with the same index as the input).
        # Behavior: for a frame with axis one, transposes, transforms along zero and
        # transposes back. A list of functions becomes a dict (by function name for a
        # Series, or every column mapped to the list for a frame). Dict-likes use the
        # dict path. For a string or callable, the string-or-callable path runs; a
        # type error from it propagates but any other failure becomes a value error
        # saying the transform function failed. A Series/DataFrame result that is
        # empty while the input is not raises the same value error; a result that is
        # not a Series/DataFrame, or whose index differs from the input index, raises
        # a value error saying the function did not transform. Returns the result.
        """
        Transform a DataFrame or Series.

        Returns
        -------
        DataFrame or Series
            Result of applying ``func`` along the given axis of the
            Series or DataFrame.

        Raises
        ------
        ValueError
            If the transform function fails or does not transform.
        """
        obj = self.obj
        func = self.func
        axis = self.axis
        args = self.args
        kwargs = self.kwargs

        is_series = obj.ndim == 1

        if obj._get_axis_number(axis) == 1:
            assert not is_series
            return obj.T.transform(func, 0, *args, **kwargs).T

        if is_list_like(func) and not is_dict_like(func):
            func = cast(list[AggFuncTypeBase], func)
            # Convert func equivalent dict
            if is_series:
                func = {com.get_callable_name(v) or v: v for v in func}
            else:
                func = dict.fromkeys(obj, func)

        if is_dict_like(func):
            func = cast(AggFuncTypeDict, func)
            return self.transform_dict_like(func)

        # func is either str or callable
        func = cast(AggFuncTypeBase, func)
        try:
            result = self.transform_str_or_callable(func)
        except TypeError:
            raise
        except Exception as err:
            raise ValueError("Transform function failed") from err

        # Functions that transform may return empty Series/DataFrame
        # when the dtype is not appropriate
        if (
            isinstance(result, (ABCSeries, ABCDataFrame))
            and result.empty
            and not obj.empty
        ):
            raise ValueError("Transform function failed")
        if not isinstance(result, (ABCSeries, ABCDataFrame)) or not result.index.equals(
            obj.index
        ):
            raise ValueError("Function did not transform")

        return result

    def transform_dict_like(self, func) -> DataFrame:
        # [FUNCTION DOC]
        # Transform with a dict of column to function(s). Input: func (dict-like).
        # An empty dict raises a value error; the dict is validated (unknown columns
        # raise a key error, nested renamers a specification error). Each column is
        # selected as a Series and transformed with its function(s) along axis zero,
        # passing the stored args and kwargs. Returns the column-wise concatenation
        # of the per-column results.
        """
        Compute transform in the case of a dict-like func
        """
        from pandas.core.reshape.concat import concat

        obj = self.obj
        args = self.args
        kwargs = self.kwargs

        # transform is currently only for Series/DataFrame
        assert isinstance(obj, ABCNDFrame)

        if len(func) == 0:
            raise ValueError("No transform functions were provided")

        func = self.normalize_dictlike_arg("transform", obj, func)

        results: dict[Hashable, DataFrame | Series] = {}
        for name, how in func.items():
            colg = obj._gotitem(name, ndim=1)
            results[name] = colg.transform(how, 0, *args, **kwargs)
        return concat(results, axis=1)

    def transform_str_or_callable(self, func) -> DataFrame | Series:
        # [FUNCTION DOC]
        # Transform with a string or callable. A string is resolved as a method or
        # attribute of the object (with the stored args and kwargs). A callable is
        # first tried through the object's apply method; if that raises any
        # exception it is called directly on the whole object instead. Returns the
        # result.
        """
        Compute transform in the case of a string or callable func
        """
        obj = self.obj
        args = self.args
        kwargs = self.kwargs

        if isinstance(func, str):
            return self._apply_str(obj, func, *args, **kwargs)

        # Two possible ways to use a UDF - apply or call directly
        try:
            return obj.apply(func, args=args, **kwargs)
        except Exception:
            return func(obj, *args, **kwargs)

    def agg_list_like(self) -> DataFrame | Series:
        # [FUNCTION DOC]
        # Aggregation for a list of functions; delegates to the shared list-like
        # routine with the aggregate operation name and returns its result.
        """
        Compute aggregation in the case of a list-like argument.

        Returns
        -------
        Result of aggregation.
        """
        return self.agg_or_apply_list_like(op_name="agg")

    def compute_list_like(
        self,
        op_name: Literal["agg", "apply"],
        selected_obj: Series | DataFrame,
        kwargs: dict[str, Any],
    ) -> tuple[list[Hashable] | Index, list[Any]]:
        # [FUNCTION DOC]
        # Runs each function of the list on the selected data. Inputs: op_name (agg
        # or apply), selected_obj (Series or DataFrame) and kwargs (passed to each
        # call). For a one-dimensional selection, each function is run on the
        # selection; for a frame, the whole list is run on each column. The axis is
        # prepended to the positional arguments when the operation takes one. Keys
        # are the function names (falling back to the function itself) or the column
        # labels. Returns a pair: keys and the list of results.
        # KNOWN ISSUE: the docstring header says "like-like" (a typo for list-like).
        """
        Compute agg/apply results for like-like input.

        Parameters
        ----------
        op_name : {"agg", "apply"}
            Operation being performed.
        selected_obj : Series or DataFrame
            Data to perform operation on.
        kwargs : dict
            Keyword arguments to pass to the functions.

        Returns
        -------
        keys : list[Hashable] or Index
            Index labels for result.
        results : list
            Data for result. When aggregating with a Series, this can contain any
            Python objects.
        """
        func = cast(list[AggFuncTypeBase], self.func)
        obj = self.obj

        results = []
        keys = []

        # degenerate case
        if selected_obj.ndim == 1:
            for a in func:
                colg = obj._gotitem(selected_obj.name, ndim=1, subset=selected_obj)
                args = (
                    [self.axis, *self.args]
                    if include_axis(op_name, colg)
                    else self.args
                )
                new_res = getattr(colg, op_name)(a, *args, **kwargs)
                results.append(new_res)

                # make sure we find a good name
                name = com.get_callable_name(a) or a
                keys.append(name)

        else:
            indices = []
            for index, col in enumerate(selected_obj):
                colg = obj._gotitem(col, ndim=1, subset=selected_obj.iloc[:, index])
                args = (
                    [self.axis, *self.args]
                    if include_axis(op_name, colg)
                    else self.args
                )
                new_res = getattr(colg, op_name)(func, *args, **kwargs)
                results.append(new_res)
                indices.append(index)
            # error: Incompatible types in assignment (expression has type "Any |
            # Index", variable has type "list[Any | Callable[..., Any] | str]")
            keys = selected_obj.columns.take(indices)  # type: ignore[assignment]

        return keys, results

    def wrap_results_list_like(
        self, keys: Iterable[Hashable], results: list[Series | DataFrame]
    ):
        # [FUNCTION DOC]
        # Combines the per-function results of a list-like operation. Inputs: keys
        # (labels) and results. First tries a column-wise concatenation with the
        # keys. If that fails with a type error (results are not pandas objects), a
        # Series of the results indexed by the keys and named like the object is
        # built; if it contains nested objects a value error is raised saying
        # transform and aggregation operations cannot be combined. Returns the
        # combined result.
        from pandas.core.reshape.concat import concat

        obj = self.obj

        try:
            return concat(results, keys=keys, axis=1, sort=False)
        except TypeError as err:
            # we are concatting non-NDFrame objects,
            # e.g. a list of scalars
            from pandas import Series

            result = Series(results, index=keys, name=obj.name)
            if is_nested_object(result):
                raise ValueError(
                    "cannot combine transform and aggregation operations"
                ) from err
            return result

    def agg_dict_like(self) -> DataFrame | Series:
        # [FUNCTION DOC]
        # Aggregation for a dict of functions; delegates to the shared dict-like
        # routine with the aggregate operation name and returns its result.
        """
        Compute aggregation in the case of a dict-like argument.

        Returns
        -------
        Result of aggregation.
        """
        return self.agg_or_apply_dict_like(op_name="agg")

    def compute_dict_like(
        self,
        op_name: Literal["agg", "apply"],
        selected_obj: Series | DataFrame,
        selection: Hashable | Sequence[Hashable],
        kwargs: dict[str, Any],
    ) -> tuple[list[Hashable], list[Any]]:
        # [FUNCTION DOC]
        # Runs the function(s) of each dict entry on the matching selection.
        # Inputs: op_name (agg or apply), selected_obj, selection (used by groupby,
        # window and resample when a selection exists) and kwargs. The dict is first
        # normalized (validation and wrapping scalars into lists). Cases: a Series
        # selection runs each function on the Series; a frame with duplicate column
        # names selects all columns matching each key; groupby objects select the
        # column(s) per key (several output keys when the key matches multiple
        # columns); everything else runs each function on the column selection.
        # Returns a pair: the output keys and the list of results.
        """
        Compute agg/apply results for dict-like input.

        Parameters
        ----------
        op_name : {"agg", "apply"}
            Operation being performed.
        selected_obj : Series or DataFrame
            Data to perform operation on.
        selection : hashable or sequence of hashables
            Used by GroupBy, Window, and Resample if selection is applied to the object.
        kwargs : dict
            Keyword arguments to pass to the functions.

        Returns
        -------
        keys : list[hashable]
            Index labels for result.
        results : list
            Data for result. When aggregating with a Series, this can contain any
            Python object.
        """
        from pandas.core.groupby.generic import (
            DataFrameGroupBy,
            SeriesGroupBy,
        )

        obj = self.obj
        is_groupby = isinstance(obj, (DataFrameGroupBy, SeriesGroupBy))
        func = cast(AggFuncTypeDict, self.func)
        func = self.normalize_dictlike_arg(op_name, selected_obj, func)

        is_non_unique_col = (
            selected_obj.ndim == 2
            and selected_obj.columns.nunique() < len(selected_obj.columns)
        )

        if selected_obj.ndim == 1:
            # key only used for output
            colg = obj._gotitem(selection, ndim=1)
            results = [getattr(colg, op_name)(how, **kwargs) for _, how in func.items()]
            keys = list(func.keys())
        elif not is_groupby and is_non_unique_col:
            # key used for column selection and output
            # GH#51099
            results = []
            keys = []
            for key, how in func.items():
                indices = selected_obj.columns.get_indexer_for([key])
                labels = selected_obj.columns.take(indices)
                label_to_indices = defaultdict(list)
                for index, label in zip(indices, labels, strict=True):
                    label_to_indices[label].append(index)

                key_data = [
                    getattr(selected_obj._ixs(indice, axis=1), op_name)(how, **kwargs)
                    for label, indices in label_to_indices.items()
                    for indice in indices
                ]

                keys += [key] * len(key_data)
                results += key_data
        elif is_groupby:
            # key used for column selection and output

            df = selected_obj
            results, keys = [], []
            for key, how in func.items():
                cols = df[key]

                if cols.ndim == 1:
                    series = obj._gotitem(key, ndim=1, subset=cols)
                    results.append(getattr(series, op_name)(how, **kwargs))
                    keys.append(key)
                else:
                    for _, col in cols.items():
                        series = obj._gotitem(key, ndim=1, subset=col)
                        results.append(getattr(series, op_name)(how, **kwargs))
                        keys.append(key)
        else:
            results = [
                getattr(obj._gotitem(key, ndim=1), op_name)(how, **kwargs)
                for key, how in func.items()
            ]
            keys = list(func.keys())

        return keys, results

    def wrap_results_dict_like(
        self,
        selected_obj: Series | DataFrame,
        result_index: list[Hashable],
        result_data: list,
    ):
        # [FUNCTION DOC]
        # Combines the results of a dict-like operation. Inputs: selected_obj,
        # result_index (keys) and result_data (results). If every result is a
        # pandas object, empty ones are dropped (unless all are empty), keys for a
        # frame selection keep the column index names, and the results are
        # concatenated (rows for a Series object, columns otherwise) with the keys
        # and no sorting. A mix of pandas objects and scalars raises a value error
        # (cannot perform both aggregation and transformation operations at once).
        # Otherwise a Series of the scalars indexed by the keys is returned, named
        # after the object only when it is a Series.
        from pandas import Index
        from pandas.core.reshape.concat import concat

        obj = self.obj

        # Avoid making two isinstance calls in all and any below
        is_ndframe = [isinstance(r, ABCNDFrame) for r in result_data]

        if all(is_ndframe):
            results = [result for result in result_data if not result.empty]
            keys_to_use: Iterable[Hashable]
            keys_to_use = [
                k for k, v in zip(result_index, result_data, strict=True) if not v.empty
            ]
            # Have to check, if at least one DataFrame is not empty.
            if keys_to_use == []:
                keys_to_use = result_index
                results = result_data

            if selected_obj.ndim == 2:
                # keys are columns, so we can preserve names
                ktu = Index(keys_to_use)
                ktu._set_names(selected_obj.columns.names)
                keys_to_use = ktu

            axis: AxisInt = 0 if isinstance(obj, ABCSeries) else 1
            result = concat(
                results,
                axis=axis,
                keys=keys_to_use,
                sort=False,
            )
        elif any(is_ndframe):
            # There is a mix of NDFrames and scalars
            raise ValueError(
                "cannot perform both aggregation "
                "and transformation operations "
                "simultaneously"
            )
        else:
            from pandas import Series

            # we have a list of scalars
            # GH 36212 use name only if obj is a series
            if obj.ndim == 1:
                obj = cast("Series", obj)
                name = obj.name
            else:
                name = None

            result = Series(result_data, index=result_index, name=name)

        return result

    def apply_str(self) -> DataFrame | Series:
        # [FUNCTION DOC]
        # Applies a string-named method or attribute of the object. If the object has
        # a callable of that name, its signature is inspected: when the axis is not
        # zero and the method has no axis parameter (or is the correlation-with or
        # skew method) a value error says the operation does not support axis one; if
        # it accepts an axis (and the object is not a groupby) the current axis is
        # added to the stored keyword arguments. SIDE EFFECT: this mutates the stored
        # keyword dictionary. Then the shared string helper runs with the stored args
        # and kwargs. Returns its result.
        # KNOWN ISSUE: the comment says the caller must check the func attribute is a
        # string, referring to an attribute name that does not exist (func is the
        # attribute actually used).
        """
        Compute apply in case of a string.

        Returns
        -------
        result: Series or DataFrame
        """
        # Caller is responsible for checking isinstance(self.f, str)
        func = cast(str, self.func)

        obj = self.obj

        from pandas.core.groupby.generic import (
            DataFrameGroupBy,
            SeriesGroupBy,
        )

        # Support for `frame.transform('method')`
        # Some methods (shift, etc.) require the axis argument, others
        # don't, so inspect and insert if necessary.
        method = getattr(obj, func, None)
        if callable(method):
            sig = inspect.getfullargspec(method)
            arg_names = (*sig.args, *sig.kwonlyargs)
            if self.axis != 0 and (
                "axis" not in arg_names or func in ("corrwith", "skew")
            ):
                raise ValueError(f"Operation {func} does not support axis=1")
            if "axis" in arg_names and not isinstance(
                obj, (SeriesGroupBy, DataFrameGroupBy)
            ):
                self.kwargs["axis"] = self.axis
        return self._apply_str(obj, func, *self.args, **self.kwargs)

    def apply_list_or_dict_like(self) -> DataFrame | Series:
        # [FUNCTION DOC]
        # Apply with a list-like or dict-like of functions. The numba engine raises a
        # not-implemented error. For a frame with axis one the frame is transposed,
        # applied along zero and transposed back. Otherwise the dict-like or
        # list-like apply path runs and the result is passed through relabeling for
        # named aggregation using the stored kwargs. Returns the result.
        # KNOWN ISSUE: the docstring says the result may be none when func is neither
        # list-like nor dict-like, but no code path returns none (every non-dict func
        # is handled as a list).
        """
        Compute apply in case of a list-like or dict-like.

        Returns
        -------
        result: Series, DataFrame, or None
            Result when self.func is a list-like or dict-like, None otherwise.
        """

        if self.engine == "numba":
            raise NotImplementedError(
                "The 'numba' engine doesn't support list-like/"
                "dict likes of callables yet."
            )

        if self.axis == 1 and isinstance(self.obj, ABCDataFrame):
            return self.obj.T.apply(self.func, 0, args=self.args, **self.kwargs).T

        func = self.func
        kwargs = self.kwargs

        if is_dict_like(func):
            result = self.agg_or_apply_dict_like(op_name="apply")
        else:
            result = self.agg_or_apply_list_like(op_name="apply")

        result = reconstruct_and_relabel_result(result, func, **kwargs)

        return result

    def normalize_dictlike_arg(
        self, how: str, obj: DataFrame | Series, func: AggFuncTypeDict
    ) -> AggFuncTypeDict:
        # [FUNCTION DOC]
        # Validates and normalizes a dict of functions. Inputs: how (apply, agg or
        # transform; anything else fails an assertion), obj and func (dict). Raises a
        # specification error (nested renamer not supported) if, for a Series
        # aggregation, any value is list-like, or if any value is dict-like. For a
        # frame, keys missing from the columns raise a key error listing them. If
        # any value is a list, tuple or dict, all scalar values are wrapped in
        # one-element lists. Returns the (possibly new) dict.
        """
        Handler for dict-like argument.

        Ensures that necessary columns exist if obj is a DataFrame, and
        that a nested renamer is not passed. Also normalizes to all lists
        when values consists of a mix of list and non-lists.
        """
        assert how in ("apply", "agg", "transform")

        # Can't use func.values(); wouldn't work for a Series
        if (
            how == "agg"
            and isinstance(obj, ABCSeries)
            and any(is_list_like(v) for _, v in func.items())
        ) or (any(is_dict_like(v) for _, v in func.items())):
            # GH 15931 - deprecation of renaming keys
            raise SpecificationError("nested renamer is not supported")

        if obj.ndim != 1:
            # Check for missing columns on a frame
            from pandas import Index

            cols = Index(list(func.keys())).difference(obj.columns, sort=True)
            if len(cols) > 0:
                # GH 58474
                raise KeyError(f"Label(s) {list(cols)} do not exist")

        aggregator_types = (list, tuple, dict)

        # if we have a dict of any non-scalars
        # eg. {'A' : ['mean']}, normalize all to
        # be list-likes
        # Cannot use func.values() because arg may be a Series
        if any(isinstance(x, aggregator_types) for _, x in func.items()):
            new_func: AggFuncTypeDict = {}
            for k, v in func.items():
                if not isinstance(v, aggregator_types):
                    new_func[k] = [v]
                else:
                    new_func[k] = v
            func = new_func
        return func

    def _apply_str(self, obj, func: str, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: resolves a string function name against an object. Inputs: obj,
        # func (string; a non-string fails an assertion) and extra args/kwargs.
        # If the object has an attribute of that name: a callable is called with the
        # args and kwargs; a non-callable attribute is returned (assertion errors if
        # args or an axis keyword were passed). Else, if numpy has a function of that
        # name and the object is array-like, the numpy function is called with the
        # object first. Otherwise an attribute error says the string is not a valid
        # function for the object type.
        """
        if arg is a string, then try to operate on it:
        - try to find a function (or attribute) on obj
        - try to find a numpy function
        - raise
        """
        assert isinstance(func, str)

        if hasattr(obj, func):
            f = getattr(obj, func)
            if callable(f):
                return f(*args, **kwargs)

            # people may aggregate on a non-callable attribute
            # but don't let them think they can pass args to it
            assert len(args) == 0
            assert not any(kwarg == "axis" for kwarg in kwargs)
            return f
        elif hasattr(np, func) and hasattr(obj, "__array__"):
            # in particular exclude Window
            f = getattr(np, func)
            return f(obj, *args, **kwargs)
        else:
            msg = f"'{func}' is not a valid function for '{type(obj).__name__}' object"
            raise AttributeError(msg)


class NDFrameApply(Apply):
    # [CLASS DOC]
    # Intermediate base class for appliers of Series and DataFrame (not groupby or
    # window objects). Inherits all of Apply; adds the index and aggregation axis
    # properties and the list-like and dict-like hooks for frame-like objects.
    """
    Methods shared by FrameApply and SeriesApply but
    not GroupByApply or ResamplerWindowApply
    """

    obj: DataFrame | Series

    @property
    def index(self) -> Index:
        # [FUNCTION DOC]
        # Property returning the index of the wrapped object.
        return self.obj.index

    @property
    def agg_axis(self) -> Index:
        # [FUNCTION DOC]
        # Property returning the labels of the axis that results are aggregated
        # over, derived from the current axis of the applier.
        return self.obj._get_agg_axis(self.axis)

    def agg_or_apply_list_like(
        self, op_name: Literal["agg", "apply"]
    ) -> DataFrame | Series:
        # [FUNCTION DOC]
        # List-like operation for frames and Series. For apply, the row-wise mode is
        # added to the keyword arguments (the stored mode for frames, the internal
        # compat marker or false for Series). If the object has an axis attribute of
        # one, a not-implemented error is raised. Computes the per-function results
        # with the shared routine and wraps them. Returns the wrapped result.
        obj = self.obj
        kwargs = self.kwargs

        if op_name == "apply":
            if isinstance(self, FrameApply):
                by_row = self.by_row

            elif isinstance(self, SeriesApply):
                by_row = "_compat" if self.by_row else False
            else:
                by_row = False
            kwargs = {**kwargs, "by_row": by_row}

        if getattr(obj, "axis", 0) == 1:
            raise NotImplementedError("axis other than 0 is not supported")

        keys, results = self.compute_list_like(op_name, obj, kwargs)
        result = self.wrap_results_list_like(keys, results)
        return result

    def agg_or_apply_dict_like(
        self, op_name: Literal["agg", "apply"]
    ) -> DataFrame | Series:
        # [FUNCTION DOC]
        # Dict-like operation for frames and Series. op_name must be agg or apply
        # (assertion error otherwise). For apply, the by-row mode keyword is set to
        # the internal compat marker or false. An object with an axis of one raises
        # a not-implemented error. Results are computed with no selection and wrapped.
        # Returns the wrapped result.
        assert op_name in ["agg", "apply"]
        obj = self.obj

        kwargs = {}
        if op_name == "apply":
            by_row = "_compat" if self.by_row else False
            kwargs.update({"by_row": by_row})

        if getattr(obj, "axis", 0) == 1:
            raise NotImplementedError("axis other than 0 is not supported")

        selection = None
        result_index, result_data = self.compute_dict_like(
            op_name, obj, selection, kwargs
        )
        result = self.wrap_results_dict_like(obj, result_index, result_data)
        return result


class FrameApply(NDFrameApply):
    # [CLASS DOC]
    # Applier for DataFrame objects; abstract base of the row-wise and column-wise
    # appliers. Adds frame-specific state/behavior on top of NDFrameApply: the
    # frame, the function-dispatching apply method, raw and standard apply modes,
    # and result wrapping. Its constructor restricts by_row to false or compat.
    obj: DataFrame

    def __init__(
        self,
        obj: AggObjType,
        func: AggFuncType,
        raw: bool,
        result_type: str | None,
        *,
        by_row: Literal[False, "compat"] = False,
        engine: str = "python",
        engine_kwargs: dict[str, bool] | None = None,
        args,
        kwargs,
    ) -> None:
        # [FUNCTION DOC]
        # Validates by_row (only false or the string compat are allowed; anything
        # else raises a value error) and forwards all arguments to the base
        # constructor. Note the default for by_row is false here.
        if by_row is not False and by_row != "compat":
            raise ValueError(f"by_row={by_row} not allowed")
        super().__init__(
            obj,
            func,
            raw,
            result_type,
            by_row=by_row,
            engine=engine,
            engine_kwargs=engine_kwargs,
            args=args,
            kwargs=kwargs,
        )

    # ---------------------------------------------------------------
    # Abstract Methods

    @property
    @abc.abstractmethod
    def result_index(self) -> Index:
        # [FUNCTION DOC]
        # Abstract property: subclasses return the index labels of the result
        # produced when applying the function over the series generator.
        pass

    @property
    @abc.abstractmethod
    def result_columns(self) -> Index:
        # [FUNCTION DOC]
        # Abstract property: subclasses return the column labels to use for the
        # result when it is a DataFrame.
        pass

    @property
    @abc.abstractmethod
    def series_generator(self) -> Generator[Series]:
        # [FUNCTION DOC]
        # Abstract property: subclasses return a generator that yields, one at a time,
        # the Series (columns or rows) that the function is applied to.
        pass

    @staticmethod
    @functools.cache
    @abc.abstractmethod
    def generate_numba_apply_func(
        func, nogil=True, nopython=True, parallel=False
    ) -> Callable[[npt.NDArray, Index, Index], dict[int, Any]]:
        # [FUNCTION DOC]
        # Abstract cached static method: subclasses return a numba-jitted function
        # that applies the user function to each column or row and returns a
        # dictionary from position to result. Inputs: func and the jit options
        # nogil, nopython and parallel (defaults true, true, false). The result is
        # cached per distinct set of arguments.
        pass

    @abc.abstractmethod
    def apply_with_numba(self):
        # [FUNCTION DOC]
        # Abstract: subclasses run the numba-compiled apply and return a dictionary
        # from position to result.
        pass

    def validate_values_for_numba(self) -> None:
        # [FUNCTION DOC]
        # Checks that every column can be handled by the numba engine. Raises a
        # value error naming the column if its dtype is not numeric, or if it is
        # backed by an extension array. Returns nothing.
        # Validate column dtyps all OK
        for colname, dtype in self.obj.dtypes.items():
            if not is_numeric_dtype(dtype):
                raise ValueError(
                    f"Column {colname} must have a numeric dtype. "
                    f"Found '{dtype}' instead"
                )
            if is_extension_array_dtype(dtype):
                raise ValueError(
                    f"Column {colname} is backed by an extension array, "
                    f"which is not supported by the numba engine."
                )

    @abc.abstractmethod
    def wrap_results_for_axis(
        self, results: ResType, res_index: Index
    ) -> DataFrame | Series:
        # [FUNCTION DOC]
        # Abstract: subclasses combine the dictionary of per-element results into a
        # DataFrame or Series according to the axis. Inputs: results (position to
        # result) and res_index (labels for the result).
        pass

    # ---------------------------------------------------------------

    @property
    def res_columns(self) -> Index:
        # [FUNCTION DOC]
        # Property: alias for the result columns property.
        return self.result_columns

    @property
    def columns(self) -> Index:
        # [FUNCTION DOC]
        # Property returning the columns of the wrapped frame.
        return self.obj.columns

    @cache_readonly
    def values(self):
        # [FUNCTION DOC]
        # Cached property returning the underlying values array of the frame
        # (computed once per applier instance).
        return self.obj.values

    def apply(self) -> DataFrame | Series:
        # [FUNCTION DOC]
        # Main entry point for frame apply. Dispatch order: a list-like func goes to
        # the list/dict path (the numba engine raises a not-implemented error for
        # it); a frame with both axes empty goes to the empty-result handling; a
        # string func goes to the string path (not-implemented for numba); a numpy
        # ufunc is applied block by block with all numpy floating point warnings
        # suppressed (not-implemented for numba) and wrapped without change of
        # labels; the broadcast result type goes to the broadcast routine
        # (not-implemented for numba); a frame with any empty axis goes to the
        # empty-result handling; the raw option applies to the raw array; otherwise
        # the standard series-by-series application runs. Returns the result.
        """compute the results"""

        # dispatch to handle list-like or dict-like
        if is_list_like(self.func):
            if self.engine == "numba":
                raise NotImplementedError(
                    "the 'numba' engine doesn't support lists of callables yet"
                )
            return self.apply_list_or_dict_like()

        # all empty
        if len(self.columns) == 0 and len(self.index) == 0:
            return self.apply_empty_result()

        # string dispatch
        if isinstance(self.func, str):
            if self.engine == "numba":
                raise NotImplementedError(
                    "the 'numba' engine doesn't support using "
                    "a string as the callable function"
                )
            return self.apply_str()

        # ufunc
        elif isinstance(self.func, np.ufunc):
            if self.engine == "numba":
                raise NotImplementedError(
                    "the 'numba' engine doesn't support "
                    "using a numpy ufunc as the callable function"
                )
            with np.errstate(all="ignore"):
                results = self.obj._mgr.apply("apply", func=self.func)
            # _constructor will retain self.index and self.columns
            return self.obj._constructor_from_mgr(results, axes=results.axes)

        # broadcasting
        if self.result_type == "broadcast":
            if self.engine == "numba":
                raise NotImplementedError(
                    "the 'numba' engine doesn't support result_type='broadcast'"
                )
            return self.apply_broadcast(self.obj)

        # one axis empty
        elif not all(self.obj.shape):
            return self.apply_empty_result()

        # raw
        elif self.raw:
            return self.apply_raw(engine=self.engine, engine_kwargs=self.engine_kwargs)

        return self.apply_standard()

    def agg(self):
        # [FUNCTION DOC]
        # Frame aggregation. Temporarily replaces the stored object with its
        # transpose when the axis is one (and sets the axis to zero) so the base
        # aggregation works along rows; the original object and axis are restored
        # afterwards even on errors (this temporary state mutation is a known wart,
        # per a comment). The result of the transposed aggregation is transposed
        # back. If the base aggregation could not handle the function (returned
        # none), the function is run through the frame's apply with the original
        # axis, args and kwargs. Returns the result.
        obj = self.obj
        axis = self.axis

        # TODO: Avoid having to change state
        self.obj = self.obj if self.axis == 0 else self.obj.T
        self.axis = 0

        result = None
        try:
            result = super().agg()
        finally:
            self.obj = obj
            self.axis = axis

        if axis == 1:
            result = result.T if result is not None else result

        if result is None:
            result = self.obj.apply(self.func, axis, args=self.args, **self.kwargs)

        return result

    def apply_empty_result(self):
        # [FUNCTION DOC]
        # Handles apply on a frame with at least one empty axis. The func must be
        # callable (assertion error otherwise). If result_type is something other
        # than reduce or unspecified, returns a copy of the frame. Otherwise, unless
        # a reduction was requested, tries calling the function on an empty
        # (axis-appropriate) float Series; any exception is ignored; the function is
        # considered a reduction if the outcome is not a Series. For a reduction,
        # the function is called on an empty Series if the aggregation axis is
        # non-empty (else the result is NaN) and a Series indexed by the aggregation
        # axis is returned; for a non-reduction, a copy of the frame is returned.
        """
        we have an empty result; at least 1 axis is 0

        we will try to apply the function to an empty
        series in order to see if this is a reduction function
        """
        assert callable(self.func)

        # we are not asked to reduce or infer reduction
        # so just return a copy of the existing object
        if self.result_type not in ["reduce", None]:
            return self.obj.copy()

        # we may need to infer
        should_reduce = self.result_type == "reduce"

        from pandas import Series

        if not should_reduce:
            try:
                if self.axis == 0:
                    r = self.func(
                        Series([], dtype=np.float64), *self.args, **self.kwargs
                    )
                else:
                    r = self.func(
                        Series(index=self.columns, dtype=np.float64),
                        *self.args,
                        **self.kwargs,
                    )
            except Exception:
                pass
            else:
                should_reduce = not isinstance(r, Series)

        if should_reduce:
            if len(self.agg_axis):
                r = self.func(Series([], dtype=np.float64), *self.args, **self.kwargs)
            else:
                r = np.nan

            return self.obj._constructor_sliced(r, index=self.agg_axis)
        else:
            return self.obj.copy()

    def apply_raw(self, engine="python", engine_kwargs=None):
        # [FUNCTION DOC]
        # Applies the function to the raw values array. Inputs: engine (python by
        # default, or numba) and engine_kwargs (jit options for numba).
        # Python engine: uses numpy's apply-along-axis with a wrapper that converts
        # string results to object arrays, passing the stored args and kwargs.
        # Numba engine: prepares the arguments, builds (cached) a compiled looper and
        # runs it on the values along the axis, squeezing the result back to 1D if
        # needed. A two dimensional result becomes a frame with the original labels;
        # otherwise a Series indexed by the aggregation axis.
        # KNOWN ISSUE: a TODO notes the mixed-type case is not handled; also the
        # keyword arguments returned by the argument preparation helper are unused in
        # the numba branch.
        """apply to the values as a numpy array"""

        def wrap_function(func):
            # [FUNCTION DOC]
            # HELPER: returns a wrapper around the user function that turns a string
            # result into a zero-dimensional object array (a workaround for a numpy
            # issue with string results in apply-along-axis).
            """
            Wrap user supplied function to work around numpy issue.

            see https://github.com/numpy/numpy/issues/8352
            """

            def wrapper(*args, **kwargs):
                # [FUNCTION DOC]
                # HELPER: calls the wrapped function with all arguments and wraps
                # a string result into an object array; other results are returned
                # unchanged.
                result = func(*args, **kwargs)
                if isinstance(result, str):
                    result = np.array(result, dtype=object)
                return result

            return wrapper

        if engine == "numba":
            args, kwargs = prepare_function_arguments(
                self.func,  # type: ignore[arg-type]
                self.args,
                self.kwargs,
                num_required_args=1,
            )
            # error: Argument 1 to "__call__" of "_lru_cache_wrapper" has
            # incompatible type "Callable[..., Any] | str | list[Callable
            # [..., Any] | str] | dict[Hashable,Callable[..., Any] | str |
            # list[Callable[..., Any] | str]]"; expected "Hashable"
            nb_looper = generate_apply_looper(
                self.func,  # type: ignore[arg-type]
                **get_jit_arguments(engine_kwargs),
            )
            result = nb_looper(self.values, self.axis, *args)
            # If we made the result 2-D, squeeze it back to 1-D
            result = np.squeeze(result)
        else:
            result = np.apply_along_axis(
                wrap_function(self.func),
                self.axis,
                self.values,
                *self.args,
                **self.kwargs,
            )

        # TODO: mixed type case
        if result.ndim == 2:
            return self.obj._constructor(result, index=self.index, columns=self.columns)
        else:
            return self.obj._constructor_sliced(result, index=self.agg_axis)

    def apply_broadcast(self, target: DataFrame) -> DataFrame:
        # [FUNCTION DOC]
        # Applies the function column by column and broadcasts each result to the
        # shape of the input. Input: target (frame to iterate over; the func must be
        # callable, assertion error otherwise). Each column's result must be a
        # scalar or one dimensional with the same length as the rows, otherwise a
        # value error (too many dimensions to broadcast, or cannot broadcast result)
        # is raised. Returns a frame with the same dtype layout as target values
        # and the original index and columns of the target.
        assert callable(self.func)

        result_values = np.empty_like(target.values)

        # axis which we want to compare compliance
        result_compare = target.shape[0]

        for i, col in enumerate(target.columns):
            res = self.func(target[col], *self.args, **self.kwargs)
            ares = np.asarray(res).ndim

            # must be a scalar or 1d
            if ares > 1:
                raise ValueError("too many dims to broadcast")
            if ares == 1:
                # must match return dim
                if result_compare != len(res):
                    raise ValueError("cannot broadcast result")

            result_values[:, i] = res

        # we *always* preserve the original index / columns
        result = self.obj._constructor(
            result_values, index=target.index, columns=target.columns
        )
        return result

    def apply_standard(self):
        # [FUNCTION DOC]
        # Standard series-by-series apply. Uses the Python series generator for the
        # python engine, otherwise the numba path, then wraps the results. Returns
        # the wrapped Series or DataFrame.
        if self.engine == "python":
            results, res_index = self.apply_series_generator()
        else:
            results, res_index = self.apply_series_numba()

        # wrap results
        return self.wrap_results(results, res_index)

    def apply_series_generator(self) -> tuple[ResType, Index]:
        # [FUNCTION DOC]
        # Python-engine loop. The func must be callable (assertion error otherwise).
        # Calls the function on each Series from the generator with the stored args
        # and kwargs, storing results by position. A returned Series is
        # shallow-copied because the generator may reuse and swap data in the Series
        # it yields. Returns a pair: the dictionary of results and the result index.
        assert callable(self.func)

        series_gen = self.series_generator
        res_index = self.result_index

        results = {}

        for i, v in enumerate(series_gen):
            results[i] = self.func(v, *self.args, **self.kwargs)
            if isinstance(results[i], ABCSeries):
                # If we have a view on v, we need to make a copy because
                #  series_generator will swap out the underlying data
                results[i] = results[i].copy(deep=False)

        return results, res_index

    def apply_series_numba(self):
        # [FUNCTION DOC]
        # Numba-engine path for the series-by-series apply. Raises a not-implemented
        # error if the parallel option is set, or if the index or columns are not
        # unique. Validates column dtypes (value error for non-numeric or extension
        # backed columns), runs the compiled apply and returns a pair: the dictionary
        # of results and the result index.
        if self.engine_kwargs.get("parallel", False):
            raise NotImplementedError(
                "Parallel apply is not supported when raw=False and engine='numba'"
            )
        if not self.obj.index.is_unique or not self.columns.is_unique:
            raise NotImplementedError(
                "The index/columns must be unique when raw=False and engine='numba'"
            )
        self.validate_values_for_numba()
        results = self.apply_with_numba()
        return results, self.result_index

    def wrap_results(self, results: ResType, res_index: Index) -> DataFrame | Series:
        # [FUNCTION DOC]
        # Builds the final object from the per-position results. If the first result
        # is a sequence (list-like), the axis-specific wrapper decides. Otherwise the
        # results are scalars and are wrapped into the sliced (one-dimensional)
        # type indexed by res_index; an empty result for a plain Series is given a
        # 64-bit float dtype rather than object. Returns the Series or DataFrame.
        from pandas import Series

        # see if we can infer the results
        if len(results) > 0 and 0 in results and is_sequence(results[0]):
            return self.wrap_results_for_axis(results, res_index)

        # dict of scalars

        # the default dtype of an empty Series is `object`, but this
        # code can be hit by df.mean() where the result should have dtype
        # float64 even if it's an empty Series.
        constructor_sliced = self.obj._constructor_sliced
        if len(results) == 0 and constructor_sliced is Series:
            result = constructor_sliced(results, dtype=np.float64)
        else:
            result = constructor_sliced(results)
        result.index = res_index

        return result

    def apply_str(self) -> DataFrame | Series:
        # [FUNCTION DOC]
        # String apply for frames. The size method is special-cased because the
        # frame's size is a single number: it returns the length of the relevant
        # axis broadcast into a Series over the aggregation axis. Other strings use
        # the base implementation.
        # Caller is responsible for checking isinstance(self.func, str)
        # TODO: GH#39993 - Avoid special-casing by replacing with lambda
        if self.func == "size":
            # Special-cased because DataFrame.size returns a single scalar
            obj = self.obj
            value = obj.shape[self.axis]
            return obj._constructor_sliced(value, index=self.agg_axis)
        return super().apply_str()


class FrameRowApply(FrameApply):
    # [CLASS DOC]
    # Frame applier for axis zero: the function is called once per column (each
    # column passed as a Series indexed by the row index). Sets the axis class
    # attribute to zero; implements the generator, result labels, numba function and
    # result wrapping for that orientation.
    axis: AxisInt = 0

    @property
    def series_generator(self) -> Generator[Series]:
        # [FUNCTION DOC]
        # Property returning a generator expression yielding each column of the frame
        # as a Series, in order.
        return (self.obj._ixs(i, axis=1) for i in range(len(self.columns)))

    @staticmethod
    @functools.cache
    def generate_numba_apply_func(
        func, nogil=True, nopython=True, parallel=False
    ) -> Callable[[npt.NDArray, Index, Index], dict[int, Any]]:
        # [FUNCTION DOC]
        # Builds (and caches per argument set) the compiled function for applying
        # func to every column. Requires numba (import error otherwise). Inputs: func
        # and the jit options nogil and nopython (default true) and parallel
        # (default false; documented in a comment as effectively disabled because
        # numba dictionaries are not thread safe). Importing the numba extensions
        # module here registers the Series/Index support as a side effect. The
        # returned function takes the 2D values, the column names, the row index
        # and extra arguments, builds a Series per column named after the column and
        # collects the user function results into a dictionary by column position.
        numba = import_optional_dependency("numba")
        from pandas import Series

        # Import helper from extensions to cast string object -> np strings
        # Note: This also has the side effect of loading our numba extensions
        from pandas.core._numba.extensions import maybe_cast_str

        jitted_udf = numba.extending.register_jitable(func)

        # Currently the parallel argument doesn't get passed through here
        # (it's disabled) since the dicts in numba aren't thread-safe.
        @numba.jit(nogil=nogil, nopython=nopython, parallel=parallel)
        def numba_func(values, col_names, df_index, *args):
            # [FUNCTION DOC]
            # HELPER: the jitted loop; see the builder above. Returns a dictionary
            # from column position to the user function result.
            results = {}
            for j in range(values.shape[1]):
                # Create the series
                ser = Series(
                    values[:, j], index=df_index, name=maybe_cast_str(col_names[j])
                )
                results[j] = jitted_udf(ser, *args)
            return results

        return numba_func

    def apply_with_numba(self) -> dict[int, Any]:
        # [FUNCTION DOC]
        # Runs the numba column-wise apply. Prepares the function arguments, gets the
        # compiled function using the jit options from the engine kwargs, then runs
        # it with the index and columns temporarily annotated with numba-friendly
        # data (string-like labels only, otherwise a value error). Returns a plain
        # dictionary from column position to result (converted from numba's typed
        # dictionary).
        # KNOWN ISSUE: the keyword arguments returned by the argument preparation
        # step are discarded and never passed on.
        func = cast(Callable, self.func)
        args, kwargs = prepare_function_arguments(
            func, self.args, self.kwargs, num_required_args=1
        )
        nb_func = self.generate_numba_apply_func(
            func, **get_jit_arguments(self.engine_kwargs)
        )
        from pandas.core._numba.extensions import set_numba_data

        index = self.obj.index
        columns = self.obj.columns

        # Convert from numba dict to regular dict
        # Our isinstance checks in the df constructor don't pass for numbas typed dict
        with set_numba_data(index) as index, set_numba_data(columns) as columns:
            res = dict(nb_func(self.values, columns, index, *args))
        return res

    @property
    def result_index(self) -> Index:
        # [FUNCTION DOC]
        # Property: for column-wise apply the result labels are the frame columns.
        return self.columns

    @property
    def result_columns(self) -> Index:
        # [FUNCTION DOC]
        # Property: for column-wise apply the result columns are the frame row index.
        return self.index

    def wrap_results_for_axis(
        self, results: ResType, res_index: Index
    ) -> DataFrame | Series:
        # [FUNCTION DOC]
        # Combines sequence-like per-column results. With the reduce result type,
        # or when no result type was given and every result is a dictionary, builds a
        # Series-like object from the results labeled by res_index. Otherwise builds
        # a frame with the results as columns; if that fails because the result
        # arrays have different lengths, falls back to the Series-like form (any
        # other value error is re-raised). When the first result is not a Series and
        # the row count equals the result columns count, the row labels are set to
        # the result columns; if the column count matches res_index it is used for
        # the column labels. Returns the object.
        """return the results for the rows"""

        if self.result_type == "reduce":
            # e.g. test_apply_dict GH#8735
            res = self.obj._constructor_sliced(results)
            res.index = res_index
            return res

        elif self.result_type is None and all(
            isinstance(x, dict) for x in results.values()
        ):
            # Our operation was a to_dict op e.g.
            #  test_apply_dict GH#8735, test_apply_reduce_to_dict GH#25196 #37544
            res = self.obj._constructor_sliced(results)
            res.index = res_index
            return res

        try:
            result = self.obj._constructor(data=results)
        except ValueError as err:
            if "All arrays must be of the same length" in str(err):
                # e.g. result = [[2, 3], [1.5], ['foo', 'bar']]
                #  see test_agg_listlike_result GH#29587
                res = self.obj._constructor_sliced(results)
                res.index = res_index
                return res
            else:
                raise

        if not isinstance(results[0], ABCSeries):
            if len(result.index) == len(self.res_columns):
                result.index = self.res_columns

        if len(result.columns) == len(res_index):
            result.columns = res_index

        return result


class FrameColumnApply(FrameApply):
    # [CLASS DOC]
    # Frame applier for axis one: the function is called once per row (each row
    # passed as a Series indexed by the columns). Sets the axis class attribute to
    # one; implements the generator (which reuses a single Series object for speed),
    # result labels, numba function and result wrapping for that orientation.
    axis: AxisInt = 1

    def apply_broadcast(self, target: DataFrame) -> DataFrame:
        # [FUNCTION DOC]
        # Broadcast apply for axis one: applies the base broadcasting to the
        # transpose and transposes the result back. Input: target frame.
        result = super().apply_broadcast(target.T)
        return result.T

    @property
    def series_generator(self) -> Generator[Series]:
        # [FUNCTION DOC]
        # Property returning a generator over rows as Series. Datetime-like values
        # are wrapped first; the frame must have at least one row (assertion error
        # otherwise). For extension dtypes (where a shared value array would be
        # wrong) each row is produced by positional row access. Otherwise a single
        # Series is created and, for each row, its underlying values are swapped for
        # that row, its name set to the row label without validation, and (when the
        # Series is not a view) its reference tracking reset so copy-on-write does
        # not trigger needless copies inside the user function. The same object is
        # yielded every time, so consumers must copy results that alias it.
        values = self.values
        values = ensure_wrapped_if_datetimelike(values)
        assert len(values) > 0

        # We create one Series object, and will swap out the data inside
        #  of it.  Kids: don't do this at home.
        ser = self.obj._ixs(0, axis=0)
        mgr = ser._mgr

        is_view = mgr.blocks[0].refs.has_reference()

        if isinstance(ser.dtype, ExtensionDtype):
            # values will be incorrect for this block
            # TODO(EA2D): special case would be unnecessary with 2D EAs
            obj = self.obj
            for i in range(len(obj)):
                yield obj._ixs(i, axis=0)

        else:
            for arr, name in zip(values, self.index, strict=True):
                # GH#35462 re-pin mgr in case setitem changed it
                ser._mgr = mgr
                mgr.set_values(arr)
                object.__setattr__(ser, "_name", name)
                if not is_view:
                    # In apply_series_generator we store the a shallow copy of the
                    # result, which potentially increases the ref count of this reused
                    # `ser` object (depending on the result of the applied function)
                    # -> if that happened and `ser` is already a copy, then we reset
                    # the refs here to avoid triggering a unnecessary CoW inside the
                    # applied function (https://github.com/pandas-dev/pandas/pull/56212)
                    mgr.blocks[0].refs = BlockValuesRefs(mgr.blocks[0])
                yield ser

    @staticmethod
    @functools.cache
    def generate_numba_apply_func(
        func, nogil=True, nopython=True, parallel=False
    ) -> Callable[[npt.NDArray, Index, Index], dict[int, Any]]:
        # [FUNCTION DOC]
        # Builds (and caches per argument set) the compiled function for applying
        # func to every row. Requires numba (import error otherwise). Inputs: func and
        # jit options nogil, nopython (default true) and parallel (default false).
        # The returned function takes the 2D values, the column index, the row index
        # and extra arguments; for each row it builds a Series (from a copy of the
        # row, since values were corrupted without copying) named after the row
        # label and collects the user function results by row position.
        numba = import_optional_dependency("numba")
        from pandas import Series
        from pandas.core._numba.extensions import maybe_cast_str

        jitted_udf = numba.extending.register_jitable(func)

        @numba.jit(nogil=nogil, nopython=nopython, parallel=parallel)
        def numba_func(values, col_names_index, index, *args):
            # [FUNCTION DOC]
            # HELPER: the jitted loop; see the builder above. Returns a dictionary
            # from row position to the user function result.
            results = {}
            # Currently the parallel argument doesn't get passed through here
            # (it's disabled) since the dicts in numba aren't thread-safe.
            for i in range(values.shape[0]):
                # Create the series
                # TODO: values corrupted without the copy
                ser = Series(
                    values[i].copy(),
                    index=col_names_index,
                    name=maybe_cast_str(index[i]),
                )
                results[i] = jitted_udf(ser, *args)

            return results

        return numba_func

    def apply_with_numba(self) -> dict[int, Any]:
        # [FUNCTION DOC]
        # Runs the numba row-wise apply: prepares arguments, gets the compiled
        # function, and runs it with the index and columns temporarily annotated
        # with numba-friendly data (string-like labels only, otherwise a value
        # error). Returns a plain dictionary from row position to result.
        # KNOWN ISSUE: as in the column-wise version, the keyword arguments returned
        # by the argument preparation step are discarded.
        func = cast(Callable, self.func)
        args, kwargs = prepare_function_arguments(
            func, self.args, self.kwargs, num_required_args=1
        )
        nb_func = self.generate_numba_apply_func(
            func, **get_jit_arguments(self.engine_kwargs)
        )

        from pandas.core._numba.extensions import set_numba_data

        # Convert from numba dict to regular dict
        # Our isinstance checks in the df constructor don't pass for numbas typed dict
        with (
            set_numba_data(self.obj.index) as index,
            set_numba_data(self.columns) as columns,
        ):
            res = dict(nb_func(self.values, columns, index, *args))

        return res

    @property
    def result_index(self) -> Index:
        # [FUNCTION DOC]
        # Property: for row-wise apply the result labels are the frame row index.
        return self.index

    @property
    def result_columns(self) -> Index:
        # [FUNCTION DOC]
        # Property: for row-wise apply the result columns are the frame columns.
        return self.columns

    def wrap_results_for_axis(
        self, results: ResType, res_index: Index
    ) -> DataFrame | Series:
        # [FUNCTION DOC]
        # Combines per-row results. With the expand result type, or when the first
        # result is a Series, the results are assembled into a frame shaped like the
        # input; otherwise (non-Series results with no inference wanted) a Series is
        # built from the results indexed by res_index. Returns the object.
        """return the results for the columns"""
        result: DataFrame | Series

        # we have requested to expand
        if self.result_type == "expand":
            result = self.infer_to_same_shape(results, res_index)

        # we have a non-series and don't want inference
        elif not isinstance(results[0], ABCSeries):
            result = self.obj._constructor_sliced(results)
            result.index = res_index

        # we may want to infer results
        else:
            result = self.infer_to_same_shape(results, res_index)

        return result

    def infer_to_same_shape(self, results: ResType, res_index: Index) -> DataFrame:
        # [FUNCTION DOC]
        # Builds a frame from the per-row results: constructs a frame with the
        # results as columns, transposes it, assigns res_index as the row index and
        # re-infers dtypes of object columns. Returns the frame.
        """infer the results to the same shape as the input object"""
        result = self.obj._constructor(data=results)
        result = result.T

        # set the index
        result.index = res_index

        # infer dtypes
        result = result.infer_objects()

        return result


class SeriesApply(NDFrameApply):
    # [CLASS DOC]
    # Applier for Series objects. Fixed attributes: axis is zero; raw is false and
    # result_type none (both set in the constructor). by_row (false, compat or the
    # internal compat marker) controls whether functions are applied per element;
    # it is only relevant to apply. Implements apply, agg, empty handling, the
    # compatibility path for functions within list/dict arguments, and the standard
    # elementwise path.
    obj: Series
    axis: AxisInt = 0
    by_row: Literal[False, "compat", "_compat"]  # only relevant for apply()

    def __init__(
        self,
        obj: Series,
        func: AggFuncType,
        *,
        by_row: Literal[False, "compat", "_compat"] = "compat",
        args,
        kwargs,
    ) -> None:
        # [FUNCTION DOC]
        # Forwards to the base constructor with raw false, result type none and the
        # given by_row (default compat), args and kwargs.
        super().__init__(
            obj,
            func,
            raw=False,
            result_type=None,
            by_row=by_row,
            args=args,
            kwargs=kwargs,
        )

    def apply(self) -> DataFrame | Series:
        # [FUNCTION DOC]
        # Main entry point for Series apply. Empty series use the empty handling.
        # A list-like func goes to the list/dict path; a string func to the string
        # path; the internal compat mode to the compat path; any other callable to
        # the standard path. Returns the result.
        obj = self.obj

        if len(obj) == 0:
            return self.apply_empty_result()

        # dispatch to handle list-like or dict-like
        if is_list_like(self.func):
            return self.apply_list_or_dict_like()

        if isinstance(self.func, str):
            # if we are a string, try to dispatch
            return self.apply_str()

        if self.by_row == "_compat":
            return self.apply_compat()

        # self.func is Callable
        return self.apply_standard()

    def agg(self):
        # [FUNCTION DOC]
        # Series aggregation. Uses the base implementation (strings, lists, dicts);
        # if it returns none the func must be callable (assertion error otherwise)
        # and is called on the whole Series with the stored args and kwargs. Returns
        # the result.
        result = super().agg()
        if result is None:
            obj = self.obj
            func = self.func
            # string, list-like, and dict-like are entirely handled in super
            assert callable(func)
            result = func(obj, *self.args, **self.kwargs)
        return result

    def apply_empty_result(self) -> Series:
        # [FUNCTION DOC]
        # Result for an empty Series: a new empty Series with the same dtype and
        # index, carrying over metadata with the apply method marker.
        obj = self.obj
        return obj._constructor(dtype=obj.dtype, index=obj.index).__finalize__(
            obj, method="apply"
        )

    def apply_compat(self):
        # [FUNCTION DOC]
        # Compatibility apply used for each callable inside list-like or dict-like
        # arguments (to keep behavior of versions before 2.1). If func is a callable
        # that has a known optimized equivalent and no extra args or kwargs were
        # given, applies it to the whole Series without per-row mode. Otherwise tries
        # applying per element; if that raises a value, attribute or type error,
        # retries applying to the whole Series. Returns the result.
        """compat apply method for funcs in listlikes and dictlikes.

         Used for each callable when giving listlikes and dictlikes of callables to
         apply. Needed for compatibility with Pandas < v2.1.

        .. versionadded:: 2.1.0
        """
        obj = self.obj
        func = self.func

        if callable(func):
            f = com.get_cython_func(func)
            if f and not self.args and not self.kwargs:
                return obj.apply(func, by_row=False)

        try:
            result = obj.apply(func, by_row="compat")
        except (ValueError, AttributeError, TypeError):
            result = obj.apply(func, by_row=False)
        return result

    def apply_standard(self) -> DataFrame | Series:
        # [FUNCTION DOC]
        # Standard apply for a callable. A numpy ufunc is called on the whole Series
        # with floating point warnings suppressed. With per-row mode off, the func is
        # called once on the whole Series. Otherwise the function (curried over the
        # stored args and kwargs if present, since the mapping helper accepts neither)
        # is mapped elementwise over the values. If the mapped results are Series,
        # they are combined into an expanded (frame) result sharing the index;
        # otherwise a Series with the original index and metadata is returned.
        # caller is responsible for ensuring that f is Callable
        func = cast(Callable, self.func)
        obj = self.obj

        if isinstance(func, np.ufunc):
            with np.errstate(all="ignore"):
                return func(obj, *self.args, **self.kwargs)
        elif not self.by_row:
            return func(obj, *self.args, **self.kwargs)

        if self.args or self.kwargs:
            # _map_values does not support args/kwargs
            def curried(x):
                # [FUNCTION DOC]
                # HELPER: calls the user function on one element with the stored
                # extra arguments.
                return func(x, *self.args, **self.kwargs)

        else:
            curried = func
        mapped = obj._map_values(mapper=curried)

        if len(mapped) and isinstance(mapped[0], ABCSeries):
            # GH#43986 Need to do list(mapped) in order to get treated as nested
            #  See also GH#25959 regarding EA support
            return obj._constructor_expanddim(list(mapped), index=obj.index)
        else:
            return obj._constructor(mapped, index=obj.index).__finalize__(
                obj, method="apply"
            )


class GroupByApply(Apply):
    # [CLASS DOC]
    # Applier for groupby objects (also used as a base by window and resample
    # appliers). The axis attribute is derived from the axis keyword argument. Plain
    # apply and transform are not implemented; implements the list-like and
    # dict-like hooks by temporarily forcing the as_index setting (where present)
    # and delegating to the shared computation.
    obj: GroupBy | Resampler | BaseWindow

    def __init__(
        self,
        obj: GroupBy[NDFrameT],
        func: AggFuncType,
        *,
        args,
        kwargs,
    ) -> None:
        # [FUNCTION DOC]
        # Constructor. Copies kwargs (so the caller's mapping is untouched), sets the
        # axis number from the axis keyword (default zero) resolved against the
        # underlying object, and forwards to the base constructor with raw false and
        # no result type.
        kwargs = kwargs.copy()
        self.axis = obj.obj._get_axis_number(kwargs.get("axis", 0))
        super().__init__(
            obj,
            func,
            raw=False,
            result_type=None,
            args=args,
            kwargs=kwargs,
        )

    def apply(self):
        # [FUNCTION DOC]
        # Not supported for this applier; always raises a not-implemented error.
        raise NotImplementedError

    def transform(self):
        # [FUNCTION DOC]
        # Not supported for this applier; always raises a not-implemented error.
        raise NotImplementedError

    def agg_or_apply_list_like(
        self, op_name: Literal["agg", "apply"]
    ) -> DataFrame | Series:
        # [FUNCTION DOC]
        # List-like operation for groupby-like objects. For apply, per-row mode is
        # turned off in the kwargs. An object with axis one raises a not-implemented
        # error. The selection is the object's selected data (for a one-dimensional
        # selection) or the data without excluded columns. While computing, the
        # as-index option is temporarily forced on for objects that have it (not for
        # window or resample objects). Results are computed and wrapped. Returns the
        # wrapped result.
        obj = self.obj
        kwargs = self.kwargs
        if op_name == "apply":
            kwargs = {**kwargs, "by_row": False}

        if getattr(obj, "axis", 0) == 1:
            raise NotImplementedError("axis other than 0 is not supported")

        if obj._selected_obj.ndim == 1:
            # For SeriesGroupBy this matches _obj_with_exclusions
            selected_obj = obj._selected_obj
        else:
            selected_obj = obj._obj_with_exclusions

        # Only set as_index=True on groupby objects, not Window or Resample
        # that inherit from this class.
        with com.temp_setattr(
            obj, "as_index", True, condition=hasattr(obj, "as_index")
        ):
            keys, results = self.compute_list_like(op_name, selected_obj, kwargs)
        result = self.wrap_results_list_like(keys, results)
        return result

    def agg_or_apply_dict_like(
        self, op_name: Literal["agg", "apply"]
    ) -> DataFrame | Series:
        # [FUNCTION DOC]
        # Dict-like operation for groupby-like objects. op_name must be agg or apply
        # (assertion error otherwise). For apply, the by-row mode keyword is set to
        # the internal compat marker or false. An object with axis one raises a
        # not-implemented error. For groupby objects the numba engine and engine
        # options stored in the kwargs are forwarded to each call. The selected data
        # and selection of the object are used; while computing, the as-index option
        # is temporarily forced on for objects that have it. Results are computed
        # and wrapped. Returns the wrapped result.
        from pandas.core.groupby.generic import (
            DataFrameGroupBy,
            SeriesGroupBy,
        )

        assert op_name in ["agg", "apply"]

        obj = self.obj
        kwargs: dict[str, Any] = {}
        if op_name == "apply":
            by_row = "_compat" if self.by_row else False
            kwargs.update({"by_row": by_row})

        if getattr(obj, "axis", 0) == 1:
            raise NotImplementedError("axis other than 0 is not supported")

        selected_obj = obj._selected_obj
        selection = obj._selection

        is_groupby = isinstance(obj, (DataFrameGroupBy, SeriesGroupBy))

        # Numba Groupby engine/engine-kwargs passthrough
        if is_groupby:
            engine = self.kwargs.get("engine", None)
            engine_kwargs = self.kwargs.get("engine_kwargs", None)
            kwargs.update({"engine": engine, "engine_kwargs": engine_kwargs})

        with com.temp_setattr(
            obj, "as_index", True, condition=hasattr(obj, "as_index")
        ):
            result_index, result_data = self.compute_dict_like(
                op_name, selected_obj, selection, kwargs
            )
        result = self.wrap_results_dict_like(selected_obj, result_index, result_data)
        return result


class ResamplerWindowApply(GroupByApply):
    # [CLASS DOC]
    # Applier for resample and window (rolling, expanding, ewm) objects. Subclass of
    # the groupby applier with the axis fixed at zero (class attribute) and plain
    # apply and transform unsupported (raise not-implemented errors).
    axis: AxisInt = 0
    obj: Resampler | BaseWindow

    def __init__(
        self,
        obj: Resampler | BaseWindow,
        func: AggFuncType,
        *,
        args,
        kwargs,
    ) -> None:
        # [FUNCTION DOC]
        # Constructor. Deliberately skips the groupby applier constructor (it calls
        # the grandparent constructor directly) so the axis is not derived from the
        # kwargs; passes raw false, no result type and the given args and kwargs.
        super(GroupByApply, self).__init__(
            obj,
            func,
            raw=False,
            result_type=None,
            args=args,
            kwargs=kwargs,
        )

    def apply(self):
        # [FUNCTION DOC]
        # Not supported for this applier; always raises a not-implemented error.
        raise NotImplementedError

    def transform(self):
        # [FUNCTION DOC]
        # Not supported for this applier; always raises a not-implemented error.
        raise NotImplementedError


def reconstruct_func(
    func: AggFuncType | None, **kwargs
) -> tuple[bool, AggFuncType, tuple[str, ...] | None, npt.NDArray[np.intp] | None]:
    # [FUNCTION DOC]
    # Detects and normalizes named aggregation. Inputs: func (aggregation spec, or
    # none when named aggregation is used) and keyword arguments (named aggregation
    # specs: pairs of column and function, or named-aggregation objects).
    # Relabeling is on only if func is none and all kwargs are two-element tuples
    # (non-empty) or any kwarg is a named-aggregation object. Without relabeling: a
    # list func with duplicate entries raises a specification error (function names
    # must be unique without new names), and a none func raises a type error
    # (must provide func or column/function pairs). With relabeling, named
    # aggregation objects are converted to pairs (their extra args and kwargs are
    # bound into the function) and normalized.
    # Returns a four-tuple: whether relabeling is on, the (normalized) func, the
    # output column names (tuple, or none) and the order of columns (index array, or
    # none).
    """
    This is the internal function to reconstruct func given if there is relabeling
    or not and also normalize the keyword to get new order of columns.

    If named aggregation is applied, `func` will be None, and kwargs contains the
    column and aggregation function information to be parsed;
    If named aggregation is not applied, `func` is either string (e.g. 'min') or
    Callable, or list of them (e.g. ['min', np.max]), or the dictionary of column name
    and str/Callable/list of them (e.g. {'A': 'min'}, or {'A': [np.min, lambda x: x]})

    If relabeling is True, will return relabeling, reconstructed func, column
    names, and the reconstructed order of columns.
    If relabeling is False, the columns and order will be None.

    Parameters
    ----------
    func: agg function (e.g. 'min' or Callable) or list of agg functions
        (e.g. ['min', np.max]) or dictionary (e.g. {'A': ['min', np.max]}).
    **kwargs: dict, kwargs used in is_multi_agg_with_relabel and
        normalize_keyword_aggregation function for relabelling

    Returns
    -------
    relabelling: bool, if there is relabelling or not
    func: normalized and mangled func
    columns: tuple of column names
    order: array of columns indices

    Examples
    --------
    >>> reconstruct_func(None, **{"foo": ("col", "min")})
    (True, defaultdict(<class 'list'>, {'col': ['min']}), ('foo',), array([0]))

    >>> reconstruct_func("min")
    (False, 'min', None, None)
    """
    from pandas.core.groupby.generic import NamedAgg

    relabeling = func is None and (
        is_multi_agg_with_relabel(**kwargs)
        or any(isinstance(v, NamedAgg) for v in kwargs.values())
    )

    columns: tuple[str, ...] | None = None
    order: npt.NDArray[np.intp] | None = None

    if not relabeling:
        if isinstance(func, list) and len(func) > len(set(func)):
            # GH 28426 will raise error if duplicated function names are used and
            # there is no reassigned name
            raise SpecificationError(
                "Function names must be unique if there is no new column names assigned"
            )
        if func is None:
            # nicer error message
            raise TypeError("Must provide 'func' or tuples of '(column, aggfunc).")

    if relabeling:
        # error: Incompatible types in assignment (expression has type
        # "MutableMapping[Hashable, list[Callable[..., Any] | str]]", variable has type
        # "Callable[..., Any] | str | list[Callable[..., Any] | str] |
        # MutableMapping[Hashable, Callable[..., Any] | str | list[Callable[..., Any] |
        # str]] | None")
        converted_kwargs = {}
        for key, val in kwargs.items():
            if isinstance(val, NamedAgg):
                aggfunc = val.aggfunc
                if val.args or val.kwargs:
                    aggfunc = lambda x, func=aggfunc, a=val.args, kw=val.kwargs: func(
                        x, *a, **kw
                    )
                converted_kwargs[key] = (val.column, aggfunc)
            else:
                converted_kwargs[key] = val

        func, columns, order = normalize_keyword_aggregation(  # type: ignore[assignment]
            converted_kwargs
        )

    assert func is not None

    return relabeling, func, columns, order


def is_multi_agg_with_relabel(**kwargs) -> bool:
    # [FUNCTION DOC]
    # Returns true when there is at least one keyword argument and every value is a
    # tuple of exactly two elements (the shape of a named aggregation); false
    # otherwise, including for no arguments.
    """
    Check whether kwargs passed to .agg look like multi-agg with relabeling.

    Parameters
    ----------
    **kwargs : dict

    Returns
    -------
    bool

    Examples
    --------
    >>> is_multi_agg_with_relabel(a="max")
    False
    >>> is_multi_agg_with_relabel(a_max=("a", "max"), a_min=("a", "min"))
    True
    >>> is_multi_agg_with_relabel()
    False
    """
    return all(isinstance(v, tuple) and len(v) == 2 for v in kwargs.values()) and (
        len(kwargs) > 0
    )


def normalize_keyword_aggregation(
    kwargs: dict,
) -> tuple[
    MutableMapping[Hashable, list[AggFuncTypeBase]],
    tuple[str, ...],
    npt.NDArray[np.intp],
]:
    # [FUNCTION DOC]
    # Converts named aggregation (output name mapped to a pair of column and
    # function) to a mapping from column to list of functions. Input: kwargs
    # (dict of output name to (column, function) pair). Returns a triple: the
    # column-to-function-list mapping (grouped by column in first-appearance order),
    # the tuple of user-provided output names, and an index array giving, for each
    # output name, the position of its (column, function name) pair in the grouped
    # ordering; duplicate function names (such as several lambdas) are made unique
    # with numeric suffixes so positions are unambiguous.
    """
    Normalize user-provided "named aggregation" kwargs.
    Transforms from the new ``Mapping[str, NamedAgg]`` style kwargs
    to the old Dict[str, List[scalar]]].

    Parameters
    ----------
    kwargs : dict

    Returns
    -------
    aggspec : dict
        The transformed kwargs.
    columns : tuple[str, ...]
        The user-provided keys.
    col_idx_order : List[int]
        List of columns indices.

    Examples
    --------
    >>> normalize_keyword_aggregation({"output": ("input", "sum")})
    (defaultdict(<class 'list'>, {'input': ['sum']}), ('output',), array([0]))
    """
    from pandas.core.indexes.base import Index

    # Normalize the aggregation functions as Mapping[column, List[func]],
    # process normally, then fixup the names.
    # TODO: aggspec type: typing.Dict[str, List[AggScalar]]
    aggspec = defaultdict(list)
    order = []
    columns = tuple(kwargs.keys())

    for column, aggfunc in kwargs.values():
        aggspec[column].append(aggfunc)
        order.append((column, com.get_callable_name(aggfunc) or aggfunc))

    # uniquify aggfunc name if duplicated in order list
    uniquified_order = _make_unique_kwarg_list(order)

    # GH 25719, due to aggspec will change the order of assigned columns in aggregation
    # uniquified_aggspec will store uniquified order list and will compare it with order
    # based on index
    aggspec_order = [
        (column, com.get_callable_name(aggfunc) or aggfunc)
        for column, aggfuncs in aggspec.items()
        for aggfunc in aggfuncs
    ]
    uniquified_aggspec = _make_unique_kwarg_list(aggspec_order)

    # get the new index of columns by comparison
    col_idx_order = Index(uniquified_aggspec).get_indexer(uniquified_order)
    return aggspec, columns, col_idx_order


def _make_unique_kwarg_list(
    seq: Sequence[tuple[Any, Any]],
) -> Sequence[tuple[Any, Any]]:
    # [FUNCTION DOC]
    # HELPER: makes repeated (column, function name) pairs unique. Input: seq (a
    # sequence of pairs). A pair that occurs more than once gets a numeric suffix
    # (counting prior identical pairs, starting at zero) appended to its second
    # element; unique pairs are unchanged. Returns a new list; the input is not
    # modified.
    """
    Uniquify aggfunc name of the pairs in the order list

    Examples:
    --------
    >>> kwarg_list = [("a", "<lambda>"), ("a", "<lambda>"), ("b", "<lambda>")]
    >>> _make_unique_kwarg_list(kwarg_list)
    [('a', '<lambda>_0'), ('a', '<lambda>_1'), ('b', '<lambda>')]
    """
    return [
        (pair[0], f"{pair[1]}_{seq[:i].count(pair)}") if seq.count(pair) > 1 else pair
        for i, pair in enumerate(seq)
    ]


def relabel_result(
    result: DataFrame | Series,
    func: dict[str, list[Callable | str]],
    columns: Iterable[Hashable],
    order: Iterable[int],
) -> dict[Hashable, Series]:
    # [FUNCTION DOC]
    # Reorders an aggregation result to follow user-provided output names.
    # Inputs: result (aggregated Series or DataFrame whose index holds function
    # names), func (mapping from column to list of functions), columns (output
    # names) and order (for each output name, its position among all aggregated
    # (column, function) pairs).
    # Behavior: output names are sorted by that position; for each input column in
    # func order, the non-missing aggregated values are taken, re-ordered to match the
    # function order when the frame has several columns (function names resolved
    # from callables), relabeled with the corresponding output names, and
    # reindexed against the full list of output names (absent names become missing).
    # Returns a dictionary from column to Series.
    """
    Internal function to reorder result if relabelling is True for
    dataframe.agg, and return the reordered result in dict.

    Parameters:
    ----------
    result: Result from aggregation
    func: Dict of (column name, funcs)
    columns: New columns name for relabelling
    order: New order for relabelling

    Examples
    --------
    >>> from pandas.core.apply import relabel_result
    >>> result = pd.DataFrame(
    ...     {"A": [np.nan, 2, np.nan], "C": [6, np.nan, np.nan], "B": [np.nan, 4, 2.5]},
    ...     index=["max", "mean", "min"],
    ... )
    >>> funcs = {"A": ["max"], "C": ["max"], "B": ["mean", "min"]}
    >>> columns = ("foo", "aab", "bar", "dat")
    >>> order = [0, 1, 2, 3]
    >>> result_in_dict = relabel_result(result, funcs, columns, order)
    >>> pd.DataFrame(result_in_dict, index=columns)
           A    C    B
    foo  2.0  NaN  NaN
    aab  NaN  6.0  NaN
    bar  NaN  NaN  4.0
    dat  NaN  NaN  2.5
    """
    from pandas.core.indexes.base import Index

    reordered_indexes = [
        pair[0] for pair in sorted(zip(columns, order, strict=True), key=lambda t: t[1])
    ]
    reordered_result_in_dict: dict[Hashable, Series] = {}
    idx = 0

    reorder_mask = not isinstance(result, ABCSeries) and len(result.columns) > 1
    for col, fun in func.items():
        s = result[col].dropna()

        # In the `_aggregate`, the callable names are obtained and used in `result`, and
        # these names are ordered alphabetically. e.g.
        #           C2   C1
        # <lambda>   1  NaN
        # amax     NaN  4.0
        # max      NaN  4.0
        # sum     18.0  6.0
        # Therefore, the order of functions for each column could be shuffled
        # accordingly so need to get the callable name if it is not parsed names, and
        # reorder the aggregated result for each column.
        # e.g. if df.agg(c1=("C2", sum), c2=("C2", lambda x: min(x))), correct order is
        # [sum, <lambda>], but in `result`, it will be [<lambda>, sum], and we need to
        # reorder so that aggregated values map to their functions regarding the order.

        # However there is only one column being used for aggregation, not need to
        # reorder since the index is not sorted, and keep as is in `funcs`, e.g.
        #         A
        # min   1.0
        # mean  1.5
        # mean  1.5
        if reorder_mask:
            fun = [
                com.get_callable_name(f) if not isinstance(f, str) else f for f in fun
            ]
            col_idx_order = Index(s.index, copy=False).get_indexer(fun)
            valid_idx = col_idx_order != -1
            if valid_idx.any():
                s = s.iloc[col_idx_order[valid_idx]]
        # assign the new user-provided "named aggregation" as index names, and reindex
        # it based on the whole user-provided names.
        if not s.empty:
            s.index = reordered_indexes[idx : idx + len(fun)]
        reordered_result_in_dict[col] = s.reindex(columns)
        idx = idx + len(fun)
    return reordered_result_in_dict


def reconstruct_and_relabel_result(result, func, **kwargs) -> DataFrame | Series:
    # [FUNCTION DOC]
    # Applies named-aggregation relabeling to an already computed result. Inputs:
    # result (aggregation result), func (the original aggregation spec or none) and
    # keyword arguments (named aggregation specs). If relabeling is detected, the
    # result is reordered and returned as a frame whose index holds the output names;
    # otherwise the result is returned unchanged.
    from pandas import DataFrame

    relabeling, func, columns, order = reconstruct_func(func, **kwargs)

    if relabeling:
        # This is to keep the order to columns occurrence unchanged, and also
        # keep the order of new columns occurrence unchanged

        # For the return values of reconstruct_func, if relabeling is
        # False, columns and order will be None.
        assert columns is not None
        assert order is not None

        result_in_dict = relabel_result(result, func, columns, order)
        result = DataFrame(result_in_dict, index=columns)

    return result


# TODO: Can't use, because mypy doesn't like us setting __name__
#   error: "partial[Any]" has no attribute "__name__"
# the type is:
#   typing.Sequence[Callable[..., ScalarResult]]
#     -> typing.Sequence[Callable[..., ScalarResult]]:


def _managle_lambda_list(aggfuncs: Sequence[Any]) -> Sequence[Any]:
    # [FUNCTION DOC]
    # HELPER: gives anonymous functions unique names within an aggregation list (for
    # use by the lambda-mangling function). Input: aggfuncs, a sequence. Sequences of
    # length zero or one are returned as is. Otherwise returns a new list where each
    # anonymous function is replaced by a wrapper object named with an increasing
    # numeric suffix (starting at zero) and other entries are kept.
    # The function name is misspelled (mangle) in the source.
    """
    Possibly mangle a list of aggfuncs.

    Parameters
    ----------
    aggfuncs : Sequence

    Returns
    -------
    mangled: list-like
        A new AggSpec sequence, where lambdas have been converted
        to have unique names.

    Notes
    -----
    If just one aggfunc is passed, the name will not be mangled.
    """
    if len(aggfuncs) <= 1:
        # don't mangle for .agg([lambda x: .])
        return aggfuncs
    i = 0
    mangled_aggfuncs = []
    for aggfunc in aggfuncs:
        if com.get_callable_name(aggfunc) == "<lambda>":
            aggfunc = partial(aggfunc)
            # error: "partial[Any]" has no attribute "__name__"; maybe "__new__"?
            aggfunc.__name__ = f"<lambda_{i}>"  # type: ignore[attr-defined]
            i += 1
        mangled_aggfuncs.append(aggfunc)

    return mangled_aggfuncs


def maybe_mangle_lambdas(agg_spec: Any) -> Any:
    # [FUNCTION DOC]
    # Ensures anonymous functions in an aggregation spec have unique names. Input:
    # agg_spec (anything). Values that are neither dict-like nor list-like are
    # returned as is. A dict-like spec yields a new mapping of the same type where
    # each list-like value is mangled; a list-like spec yields a mangled list.
    # The input is not modified.
    # KNOWN ISSUE: the docstring example output refers to a different helper name
    # and is skipped in doctests, so it does not describe the actual wrapper names.
    """
    Make new lambdas with unique names.

    Parameters
    ----------
    agg_spec : Any
        An argument to GroupBy.agg.
        Non-dict-like `agg_spec` are pass through as is.
        For dict-like `agg_spec` a new spec is returned
        with name-mangled lambdas.

    Returns
    -------
    mangled : Any
        Same type as the input.

    Examples
    --------
    >>> maybe_mangle_lambdas("sum")
    'sum'
    >>> maybe_mangle_lambdas([lambda: 1, lambda: 2])  # doctest: +SKIP
    [<function __main__.<lambda_0>,
     <function pandas...._make_lambda.<locals>.f(*args, **kwargs)>]
    """
    is_dict = is_dict_like(agg_spec)
    if not (is_dict or is_list_like(agg_spec)):
        return agg_spec
    mangled_aggspec = type(agg_spec)()  # dict or OrderedDict

    if is_dict:
        for key, aggfuncs in agg_spec.items():
            if is_list_like(aggfuncs) and not is_dict_like(aggfuncs):
                mangled_aggfuncs = _managle_lambda_list(aggfuncs)
            else:
                mangled_aggfuncs = aggfuncs

            mangled_aggspec[key] = mangled_aggfuncs
    else:
        mangled_aggspec = _managle_lambda_list(agg_spec)

    return mangled_aggspec


def validate_func_kwargs(
    kwargs: dict,
) -> tuple[list[str], list[str | Callable[..., Any]]]:
    # [FUNCTION DOC]
    # Validates keyword aggregation arguments. Input: kwargs (dict of output name to
    # aggregation function). Raises a type error if any value is neither a string nor
    # callable (message names the offending type), or if the dict is empty (message
    # asks for a func or named aggregation keywords). Note that values are checked
    # before emptiness. Returns a pair: the list of names and the list of functions.
    """
    Validates types of user-provided "named aggregation" kwargs.
    `TypeError` is raised if aggfunc is not `str` or callable.

    Parameters
    ----------
    kwargs : dict

    Returns
    -------
    columns : List[str]
        List of user-provided keys.
    func : List[Union[str, callable[...,Any]]]
        List of user-provided aggfuncs

    Examples
    --------
    >>> validate_func_kwargs({"one": "min", "two": "max"})
    (['one', 'two'], ['min', 'max'])
    """
    tuple_given_message = "func is expected but received {} in **kwargs."
    columns = list(kwargs)
    func = []
    for col_func in kwargs.values():
        if not (isinstance(col_func, str) or callable(col_func)):
            raise TypeError(tuple_given_message.format(type(col_func).__name__))
        func.append(col_func)
    if not columns:
        no_arg_message = "Must provide 'func' or named aggregation **kwargs."
        raise TypeError(no_arg_message)
    return columns, func


def include_axis(op_name: Literal["agg", "apply"], colg: Series | DataFrame) -> bool:
    # [FUNCTION DOC]
    # HELPER: decides whether the axis should be passed as a positional argument
    # when calling the operation on a column selection. Returns true for a frame, or
    # for a Series when the operation is aggregation; false otherwise.
    return isinstance(colg, ABCDataFrame) or (
        isinstance(colg, ABCSeries) and op_name == "agg"
    )
