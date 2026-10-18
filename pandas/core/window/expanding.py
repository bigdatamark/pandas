# [MODULE DOC]
# Expanding window calculations. Public contents: the expanding window class
# (returned by the expanding method of series and data frames; presented in the
# public typing namespace) and its groupby variant. Both derive from the shared
# window machinery in the rolling module: the expanding class subclasses the
# rolling/expanding mixin and supplies an indexer that makes every window start
# at the first row; the groupby variant combines the window groupby base with
# the expanding class and uses a groupby-aware indexer wrapping the expanding
# indexer. Nearly all statistics methods here are thin wrappers that only exist
# to carry expanding-specific documentation and forward their arguments
# unchanged to the parent implementation.
# Import-time side effects: none.

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Concatenate,
    Literal,
    Self,
    final,
    overload,
)

from pandas.util._decorators import set_module

from pandas.core.indexers.objects import (
    BaseIndexer,
    ExpandingIndexer,
    GroupbyIndexer,
)
from pandas.core.window.rolling import (
    BaseWindowGroupby,
    RollingAndExpandingMixin,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandas._typing import (
        P,
        QuantileInterpolation,
        T,
        WindowingRankType,
    )

    from pandas import (
        DataFrame,
        Series,
    )
    from pandas.core.generic import NDFrame


@set_module("pandas.api.typing")
class Expanding(RollingAndExpandingMixin):
    # [CLASS DOC]
    # Expanding window over a series or data frame: each output position
    # aggregates all observations from the start up to that position.
    # State (via the parent): the wrapped object, min_periods (minimum
    # observations needed, default one) and method (single or table execution,
    # only meaningful with the numba engine). The attribute list used for
    # copying/repr names min_periods and method.
    # Inherits the computation machinery from the rolling/expanding mixin;
    # methods below override only to attach documentation and forward to it.
    # Invariant: the window indexer always yields windows starting at row zero.
    """
    Provide expanding window calculations.

    An expanding window yields the value of an aggregation statistic with all the data
    available up to that point in time.

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

    _attributes: list[str] = ["min_periods", "method"]

    def __init__(
        self,
        obj: NDFrame,
        min_periods: int = 1,
        method: str = "single",
        selection=None,
    ) -> None:
        # [FUNCTION DOC]
        # Builds an expanding window object. obj is the series or data frame
        # to compute over; min_periods (default one) is the minimum number of
        # observations for a non-missing result; method (default single)
        # chooses per-column or whole-table execution (table only valid with
        # the numba engine, validated by the parent); selection optionally
        # restricts to selected columns (used by groupby). All values are
        # forwarded to the parent initializer, which validates them.
        super().__init__(
            obj=obj,
            min_periods=min_periods,
            method=method,
            selection=selection,
        )

    def _get_window_indexer(self) -> BaseIndexer:
        # [FUNCTION DOC]
        # HELPER: hook used by the window machinery. Returns a new expanding
        # indexer instance, which computes window bounds with every start at
        # zero and each end just past the current row.
        """
        Return an indexer class that will compute the window start and end bounds
        """
        return ExpandingIndexer()

    def aggregate(self, func=None, *args, **kwargs):
        # [FUNCTION DOC]
        # Aggregate the expanding windows with one or more functions. func is
        # a function, function name, list of those, or mapping of column label
        # to those (default none); extra positional and keyword arguments are
        # passed through to the functions. Delegates entirely to the parent
        # and returns its result: a scalar-like series, series or data frame
        # depending on input and number of functions. Also exposed under the
        # shorter alias name.
        """
        Aggregate using one or more operations over the specified axis.

        Parameters
        ----------
        func : function, str, list or dict
            Function to use for aggregating the data. If a function, must either
            work when passed a Series/Dataframe or when passed to
            Series/Dataframe.apply.

            Accepted combinations are:

            - function
            - string function name
            - list of functions and/or function names, e.g. ``[np.sum, 'mean']``
            - dict of axis labels -> functions, function names or list of such.

        *args
            Positional arguments to pass to `func`.
        **kwargs
            Keyword arguments to pass to `func`.

        Returns
        -------
        scalar, Series or DataFrame

            The return can be:

            * scalar : when Series.agg is called with single function
            * Series : when DataFrame.agg is called with a single function
            * DataFrame : when DataFrame.agg is called with several functions

        See Also
        --------
        DataFrame.aggregate : Similar DataFrame method.
        Series.aggregate : Similar Series method.

        Notes
        -----
        The aggregation operations are always performed over an axis, either the
        index (default) or the column axis. This behavior is different from
        `numpy` aggregation functions (`mean`, `median`, `prod`, `sum`, `std`,
        `var`), where the default is to compute the aggregation of the flattened
        array, e.g., ``numpy.mean(arr_2d)`` as opposed to
        ``numpy.mean(arr_2d, axis=0)``.

        `agg` is an alias for `aggregate`. Use the alias.

        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.

        A passed user-defined-function will be passed a Series for evaluation.

        If ``func`` defines an index relabeling, ``axis`` must be ``0`` or ``index``.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        >>> df
           A  B  C
        0  1  4  7
        1  2  5  8
        2  3  6  9

        >>> df.expanding(2).sum()
             A     B     C
        0  NaN   NaN   NaN
        1  3.0   9.0  15.0
        2  6.0  15.0  24.0

        >>> df.expanding(2).agg({"A": "sum", "B": "min"})
             A    B
        0  NaN  NaN
        1  3.0  4.0
        2  6.0  4.0
        """
        return super().aggregate(func, *args, **kwargs)

    agg = aggregate

    def count(self, numeric_only: bool = False):
        # [FUNCTION DOC]
        # Expanding count of non-missing observations. numeric_only (default
        # false) restricts the computation to float, integer and boolean
        # columns. Forwards to the parent; returns an object of the input's
        # type with float results.
        """
        Calculate the expanding count of non NaN observations.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        Series.expanding : Calling expanding with Series data.
        DataFrame.expanding : Calling expanding with DataFrames.
        Series.count : Aggregating count for Series.
        DataFrame.count : Aggregating count for DataFrame.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"])
        >>> ser.expanding().count()
        a    1.0
        b    2.0
        c    3.0
        d    4.0
        dtype: float64
        """
        return super().count(numeric_only=numeric_only)

    def apply(
        self,
        func: Callable[..., Any],
        raw: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
    ):
        # [FUNCTION DOC]
        # Apply a custom function over each expanding window. func maps the
        # window values (an array when raw is true, otherwise a series) to a
        # single value (or a compiled function for the numba engine); raw
        # (default false) selects array input; engine (default none, meaning
        # the cython engine unless the global numba option is set) selects
        # cython or numba; engine_kwargs (default none) holds the numba jit
        # flags for nopython, nogil and parallel; args and kwargs (default
        # none) are extra positional and keyword arguments for func. Forwards
        # all to the parent and returns a float-valued object of the input's
        # type. The parent validates the engine combination.
        """
        Calculate the expanding custom aggregation function.

        Parameters
        ----------
        func : function
            Must produce a single value from an ndarray input if ``raw=True``
            or a single value from a Series if ``raw=False``. Can also accept a
            Numba JIT function with ``engine='numba'`` specified.

        raw : bool, default False
            * ``False`` : passes each row or column as a Series to the
              function.
            * ``True`` : the passed function will receive ndarray objects instead.

            If you are just applying a NumPy reduction function this will
            achieve much better performance.

        engine : str, default None
            * ``'cython'`` : Runs rolling apply through C-extensions from cython.
            * ``'numba'`` : Runs rolling apply through JIT compiled code from numba.
              Only available when ``raw`` is set to ``True``.
            * ``None`` : Defaults to ``'cython'`` or globally setting
              ``compute.use_numba``

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{'nopython': True, 'nogil': False, 'parallel': False}`` and will be
              applied to both the ``func`` and the ``apply`` rolling aggregation.

        args : tuple, default None
            Positional arguments to be passed into func.

        kwargs : dict, default None
            Keyword arguments to be passed into func.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        Series.expanding : Calling expanding with Series data.
        DataFrame.expanding : Calling expanding with DataFrames.
        Series.apply : Aggregating apply for Series.
        DataFrame.apply : Aggregating apply for DataFrame.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"])
        >>> ser.expanding().apply(lambda s: s.max() - 2 * s.min())
        a   -1.0
        b    0.0
        c    1.0
        d    2.0
        dtype: float64
        """
        return super().apply(
            func,
            raw=raw,
            engine=engine,
            engine_kwargs=engine_kwargs,
            args=args,
            kwargs=kwargs,
        )

    @overload
    def pipe(
        self,
        func: Callable[Concatenate[Self, P], T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T: ...
    # [FUNCTION DOC]
    # Overload stub (callable form): see the final implementation of the
    # pipe method below.

    @overload
    def pipe(
        self,
        func: tuple[Callable[..., T], str],
        *args: Any,
        **kwargs: Any,
    ) -> T: ...
    # [FUNCTION DOC]
    # Overload stub (callable plus data keyword form): see the final
    # implementation of the pipe method below.

    @final
    def pipe(
        self,
        func: Callable[Concatenate[Self, P], T] | tuple[Callable[..., T], str],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        # [FUNCTION DOC]
        # Implementation of the overloaded pipe method (marked final).
        # func is either a callable that receives this expanding object as its
        # first argument, or a pair of (callable, keyword name) where the
        # object is passed under that keyword. Extra positional and keyword
        # arguments are forwarded. Delegates to the parent and returns
        # whatever func returns (not necessarily an expanding object).
        """
        Apply a ``func`` with arguments to this Expanding object and return its result.

        Use `.pipe` when you want to improve readability by chaining together
        functions that expect Series, DataFrames, GroupBy, Rolling, Expanding or
        Resampler
        objects.
        Instead of writing

        >>> h = lambda x, arg2, arg3: x + 1 - arg2 * arg3
        >>> g = lambda x, arg1: x * 5 / arg1
        >>> f = lambda x: x**4
        >>> df = pd.DataFrame(
        ...     {"A": [1, 2, 3, 4]}, index=pd.date_range("2012-08-02", periods=4)
        ... )
        >>> h(g(f(df.rolling("2D")), arg1=1), arg2=2, arg3=3)  # doctest: +SKIP

        You can write

        >>> (
        ...     df.rolling("2D").pipe(f).pipe(g, arg1=1).pipe(h, arg2=2, arg3=3)
        ... )  # doctest: +SKIP

        which is much more readable.

        Parameters
        ----------
        func : callable or tuple of (callable, str)
            Function to apply to this Expanding object or, alternatively,
            a `(callable, data_keyword)` tuple where `data_keyword` is a
            string indicating the keyword of `callable` that expects the
            Expanding object.
        *args : iterable, optional
            Positional arguments passed into `func`.
        **kwargs : dict, optional
                A dictionary of keyword arguments passed into `func`.

        Returns
        -------
        Expanding
            The original object with the function `func` applied.

        See Also
        --------
        Series.pipe : Apply a function with arguments to a series.
        DataFrame.pipe: Apply a function with arguments to a dataframe.
        apply : Apply function to each group instead of to the
            full Expanding object.

        Notes
        -----
        See more `here
        <https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#piping-function-calls>`_

        Examples
        --------

        >>> df = pd.DataFrame(
        ...     {"A": [1, 2, 3, 4]}, index=pd.date_range("2012-08-02", periods=4)
        ... )
        >>> df
                    A
        2012-08-02  1
        2012-08-03  2
        2012-08-04  3
        2012-08-05  4

        To get the difference between each expanding window's maximum and minimum
        value in one pass, you can do

        >>> df.expanding().pipe(lambda x: x.max() - x.min())
                      A
        2012-08-02  0.0
        2012-08-03  1.0
        2012-08-04  2.0
        2012-08-05  3.0
        """
        return super().pipe(func, *args, **kwargs)

    def sum(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        # [FUNCTION DOC]
        # Expanding sum. numeric_only (default false) limits to float,
        # integer and boolean columns; engine (default none: cython unless the
        # global numba option is set) selects cython or numba; engine_kwargs
        # (default none) are numba jit flags (nopython, nogil, parallel), not
        # accepted by cython. Forwards to the parent, which validates them.
        # Returns a float-valued object of the input's type.
        """
        Calculate the expanding sum.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or globally setting
              ``compute.use_numba``

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{'nopython': True, 'nogil': False, 'parallel': False}``

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        Series.expanding : Calling expanding with Series data.
        DataFrame.expanding : Calling expanding with DataFrames.
        Series.sum : Aggregating sum for Series.
        DataFrame.sum : Aggregating sum for DataFrame.

        Notes
        -----
        See :ref:`window.numba_engine` and :ref:`enhancingperf.numba` for extended
        documentation and performance considerations for the Numba engine.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"])
        >>> ser.expanding().sum()
        a     1.0
        b     3.0
        c     6.0
        d    10.0
        dtype: float64
        """
        return super().sum(
            numeric_only=numeric_only,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )

    def max(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        # [FUNCTION DOC]
        # Expanding maximum; same parameters (numeric_only, engine,
        # engine_kwargs) and delegation behavior as the expanding sum.
        """
        Calculate the expanding maximum.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or globally setting
              ``compute.use_numba``

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{'nopython': True, 'nogil': False, 'parallel': False}``

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        Series.expanding : Calling expanding with Series data.
        DataFrame.expanding : Calling expanding with DataFrames.
        Series.max : Aggregating max for Series.
        DataFrame.max : Aggregating max for DataFrame.

        Notes
        -----
        See :ref:`window.numba_engine` and :ref:`enhancingperf.numba` for extended
        documentation and performance considerations for the Numba engine.

        Examples
        --------
        >>> ser = pd.Series([3, 2, 1, 4], index=["a", "b", "c", "d"])
        >>> ser.expanding().max()
        a    3.0
        b    3.0
        c    3.0
        d    4.0
        dtype: float64
        """
        return super().max(
            numeric_only=numeric_only,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )

    def min(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        # [FUNCTION DOC]
        # Expanding minimum; same parameters (numeric_only, engine,
        # engine_kwargs) and delegation behavior as the expanding sum.
        """
        Calculate the expanding minimum.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or globally setting
              ``compute.use_numba``

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{'nopython': True, 'nogil': False, 'parallel': False}``

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        Series.expanding : Calling expanding with Series data.
        DataFrame.expanding : Calling expanding with DataFrames.
        Series.min : Aggregating min for Series.
        DataFrame.min : Aggregating min for DataFrame.

        Notes
        -----
        See :ref:`window.numba_engine` and :ref:`enhancingperf.numba` for extended
        documentation and performance considerations for the Numba engine.

        Examples
        --------
        >>> ser = pd.Series([2, 3, 4, 1], index=["a", "b", "c", "d"])
        >>> ser.expanding().min()
        a    2.0
        b    2.0
        c    2.0
        d    1.0
        dtype: float64
        """
        return super().min(
            numeric_only=numeric_only,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )

    def mean(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        # [FUNCTION DOC]
        # Expanding mean; same parameters (numeric_only, engine,
        # engine_kwargs) and delegation behavior as the expanding sum.
        """
        Calculate the expanding mean.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or globally setting
              ``compute.use_numba``

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{'nopython': True, 'nogil': False, 'parallel': False}``

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        Series.expanding : Calling expanding with Series data.
        DataFrame.expanding : Calling expanding with DataFrames.
        Series.mean : Aggregating mean for Series.
        DataFrame.mean : Aggregating mean for DataFrame.

        Notes
        -----
        See :ref:`window.numba_engine` and :ref:`enhancingperf.numba` for extended
        documentation and performance considerations for the Numba engine.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"])
        >>> ser.expanding().mean()
        a    1.0
        b    1.5
        c    2.0
        d    2.5
        dtype: float64
        """
        return super().mean(
            numeric_only=numeric_only,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )

    def median(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        # [FUNCTION DOC]
        # Expanding median; same parameters (numeric_only, engine,
        # engine_kwargs) and delegation behavior as the expanding sum.
        """
        Calculate the expanding median.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or globally setting
              ``compute.use_numba``

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{'nopython': True, 'nogil': False, 'parallel': False}``

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        Series.expanding : Calling expanding with Series data.
        DataFrame.expanding : Calling expanding with DataFrames.
        Series.median : Aggregating median for Series.
        DataFrame.median : Aggregating median for DataFrame.

        Notes
        -----
        See :ref:`window.numba_engine` and :ref:`enhancingperf.numba` for extended
        documentation and performance considerations for the Numba engine.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"])
        >>> ser.expanding().median()
        a    1.0
        b    1.5
        c    2.0
        d    2.5
        dtype: float64
        """
        return super().median(
            numeric_only=numeric_only,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )

    def std(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        # [FUNCTION DOC]
        # Expanding standard deviation. ddof (default one) is the delta
        # degrees of freedom: the divisor is the observation count minus ddof.
        # numeric_only, engine and engine_kwargs behave as for the expanding
        # sum. Delegates to the parent; windows with too few observations give
        # missing results.
        """
        Calculate the expanding standard deviation.

        Parameters
        ----------
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.

        numeric_only : bool, default False
            Include only float, int, boolean columns.

        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or globally setting
              ``compute.use_numba``

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{'nopython': True, 'nogil': False, 'parallel': False}``

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        numpy.std : Equivalent method for NumPy array.
        Series.expanding : Calling expanding with Series data.
        DataFrame.expanding : Calling expanding with DataFrames.
        Series.std : Aggregating std for Series.
        DataFrame.std : Aggregating std for DataFrame.

        Notes
        -----
        The default ``ddof`` of 1 used in :meth:`Series.std` is different
        than the default ``ddof`` of 0 in :func:`numpy.std`.

        A minimum of one period is required for the rolling calculation.

        Examples
        --------
        >>> s = pd.Series([5, 5, 6, 7, 5, 5, 5])

        >>> s.expanding(3).std()
        0         NaN
        1         NaN
        2    0.577350
        3    0.957427
        4    0.894427
        5    0.836660
        6    0.786796
        dtype: float64
        """
        return super().std(
            ddof=ddof,
            numeric_only=numeric_only,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )

    def var(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        # [FUNCTION DOC]
        # Expanding variance; same parameters (ddof default one, numeric_only,
        # engine, engine_kwargs) and delegation as the expanding standard
        # deviation.
        """
        Calculate the expanding variance.

        Parameters
        ----------
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.

        numeric_only : bool, default False
            Include only float, int, boolean columns.

        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or globally setting
              ``compute.use_numba``

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{'nopython': True, 'nogil': False, 'parallel': False}``

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        numpy.var : Equivalent method for NumPy array.
        Series.expanding : Calling expanding with Series data.
        DataFrame.expanding : Calling expanding with DataFrames.
        Series.var : Aggregating var for Series.
        DataFrame.var : Aggregating var for DataFrame.

        Notes
        -----
        The default ``ddof`` of 1 used in :meth:`Series.var` is different
        than the default ``ddof`` of 0 in :func:`numpy.var`.

        A minimum of one period is required for the rolling calculation.

        Examples
        --------
        >>> s = pd.Series([5, 5, 6, 7, 5, 5, 5])

        >>> s.expanding(3).var()
        0         NaN
        1         NaN
        2    0.333333
        3    0.916667
        4    0.800000
        5    0.700000
        6    0.619048
        dtype: float64
        """
        return super().var(
            ddof=ddof,
            numeric_only=numeric_only,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )

    def sem(self, ddof: int = 1, numeric_only: bool = False):
        # [FUNCTION DOC]
        # Expanding standard error of the mean. ddof (default one) is the
        # delta degrees of freedom; numeric_only (default false) limits to
        # numeric/boolean columns. Delegates to the parent; no engine choice.
        """
        Calculate the expanding standard error of mean.

        Parameters
        ----------
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.

        numeric_only : bool, default False
            Include only float, int, boolean columns.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        Series.expanding : Calling expanding with Series data.
        DataFrame.expanding : Calling expanding with DataFrames.
        Series.sem : Aggregating sem for Series.
        DataFrame.sem : Aggregating sem for DataFrame.

        Notes
        -----
        A minimum of one period is required for the calculation.

        Examples
        --------
        >>> s = pd.Series([0, 1, 2, 3])

        >>> s.expanding().sem()
        0         NaN
        1    0.500000
        2    0.577350
        3    0.645497
        dtype: float64
        """
        return super().sem(ddof=ddof, numeric_only=numeric_only)

    def skew(self, numeric_only: bool = False):
        # [FUNCTION DOC]
        # Expanding unbiased skewness; needs at least three observations per
        # window for a value. numeric_only (default false) limits to numeric
        # columns. Delegates to the parent.
        """
        Calculate the expanding unbiased skewness.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        scipy.stats.skew : Third moment of a probability density.
        Series.expanding : Calling expanding with Series data.
        DataFrame.expanding : Calling expanding with DataFrames.
        Series.skew : Aggregating skew for Series.
        DataFrame.skew : Aggregating skew for DataFrame.

        Notes
        -----
        A minimum of three periods is required for the rolling calculation.

        Examples
        --------
        >>> ser = pd.Series([-1, 0, 2, -1, 2], index=["a", "b", "c", "d", "e"])
        >>> ser.expanding().skew()
        a         NaN
        b         NaN
        c    0.935220
        d    1.414214
        e    0.315356
        dtype: float64
        """
        return super().skew(numeric_only=numeric_only)

    def kurt(self, numeric_only: bool = False):
        # [FUNCTION DOC]
        # Expanding unbiased excess kurtosis (Fisher definition); needs at
        # least four observations per window. numeric_only (default false)
        # limits to numeric columns. Delegates to the parent.
        """
        Calculate the expanding Fisher's definition of kurtosis without bias.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        scipy.stats.kurtosis : Reference SciPy method.
        Series.expanding : Calling expanding with Series data.
        DataFrame.expanding : Calling expanding with DataFrames.
        Series.kurt : Aggregating kurt for Series.
        DataFrame.kurt : Aggregating kurt for DataFrame.

        Notes
        -----
        A minimum of four periods is required for the calculation.

        Examples
        --------
        The example below will show a rolling calculation with a window size of
        four matching the equivalent function call using `scipy.stats`.

        >>> arr = [1, 2, 3, 4, 999]
        >>> import scipy.stats
        >>> print(f"{scipy.stats.kurtosis(arr[:-1], bias=False):.6f}")
        -1.200000
        >>> print(f"{scipy.stats.kurtosis(arr, bias=False):.6f}")
        4.999874
        >>> s = pd.Series(arr)
        >>> s.expanding(4).kurt()
        0         NaN
        1         NaN
        2         NaN
        3   -1.200000
        4    4.999874
        dtype: float64
        """
        return super().kurt(numeric_only=numeric_only)

    def first(self, numeric_only: bool = False):
        # [FUNCTION DOC]
        # Expanding first (left-most) element of each window; subject to the
        # minimum periods setting. numeric_only (default false) limits to
        # numeric columns. Delegates to the parent.
        """
        Calculate the expanding First (left-most) element of the window.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        GroupBy.first : Similar method for GroupBy objects.
        Expanding.last : Method to get the last element in each window.

        Examples
        --------
        The example below will show an expanding calculation with a window size of
        three.

        >>> s = pd.Series(range(5))
        >>> s.expanding(3).first()
        0         NaN
        1         NaN
        2         0.0
        3         0.0
        4         0.0
        dtype: float64
        """
        return super().first(numeric_only=numeric_only)

    def last(self, numeric_only: bool = False):
        # [FUNCTION DOC]
        # Expanding last (right-most) element of each window; subject to the
        # minimum periods setting. numeric_only (default false) limits to
        # numeric columns. Delegates to the parent.
        """
        Calculate the expanding Last (right-most) element of the window.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        GroupBy.last : Similar method for GroupBy objects.
        Expanding.first : Method to get the first element in each window.

        Examples
        --------
        The example below will show an expanding calculation with a window size of
        three.

        >>> s = pd.Series(range(5))
        >>> s.expanding(3).last()
        0         NaN
        1         NaN
        2         2.0
        3         3.0
        4         4.0
        dtype: float64
        """
        return super().last(numeric_only=numeric_only)

    def quantile(
        self,
        q: float,
        interpolation: QuantileInterpolation = "linear",
        numeric_only: bool = False,
    ):
        # [FUNCTION DOC]
        # Expanding quantile. q is the quantile between zero and one;
        # interpolation (default linear) is one of linear, lower, higher,
        # midpoint or nearest and applies when the quantile falls between two
        # data points; numeric_only (default false) limits to numeric columns.
        # Delegates to the parent.
        """
        Calculate the expanding quantile.

        Parameters
        ----------
        q : float
            Quantile to compute. 0 <= quantile <= 1.

        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            This optional parameter specifies the interpolation method to use,
            when the desired quantile lies between two data points `i` and `j`:

                * linear: `i + (j - i) * fraction`, where `fraction` is the
                  fractional part of the index surrounded by `i` and `j`.
                * lower: `i`.
                * higher: `j`.
                * nearest: `i` or `j` whichever is nearest.
                * midpoint: (`i` + `j`) / 2.

        numeric_only : bool, default False
            Include only float, int, boolean columns.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        Series.expanding : Calling expanding with Series data.
        DataFrame.expanding : Calling expanding with DataFrames.
        Series.quantile : Aggregating quantile for Series.
        DataFrame.quantile : Aggregating quantile for DataFrame.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3, 4, 5, 6], index=["a", "b", "c", "d", "e", "f"])
        >>> ser.expanding(min_periods=4).quantile(0.25)
        a     NaN
        b     NaN
        c     NaN
        d    1.75
        e    2.00
        f    2.25
        dtype: float64
        """
        return super().quantile(
            q=q,
            interpolation=interpolation,
            numeric_only=numeric_only,
        )

    def rank(
        self,
        method: WindowingRankType = "average",
        ascending: bool = True,
        pct: bool = False,
        numeric_only: bool = False,
    ):
        # [FUNCTION DOC]
        # Expanding rank of the latest value within each window. method
        # (default average) resolves ties: average, min or max; ascending
        # (default true) sets the ranking direction; pct (default false)
        # returns percentile ranks; numeric_only (default false) limits to
        # numeric columns. Delegates to the parent.
        """
        Calculate the expanding rank.

        Parameters
        ----------
        method : {'average', 'min', 'max'}, default 'average'
            How to rank the group of records that have the same value (i.e. ties):

            * average: average rank of the group
            * min: lowest rank in the group
            * max: highest rank in the group

        ascending : bool, default True
            Whether or not the elements should be ranked in ascending order.
        pct : bool, default False
            Whether or not to display the returned rankings in percentile
            form.
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        Series.expanding : Calling expanding with Series data.
        DataFrame.expanding : Calling expanding with DataFrames.
        Series.rank : Aggregating rank for Series.
        DataFrame.rank : Aggregating rank for DataFrame.

        Examples
        --------
        >>> s = pd.Series([1, 4, 2, 3, 5, 3])
        >>> s.expanding().rank()
        0    1.0
        1    2.0
        2    2.0
        3    3.0
        4    5.0
        5    3.5
        dtype: float64

        >>> s.expanding().rank(method="max")
        0    1.0
        1    2.0
        2    2.0
        3    3.0
        4    5.0
        5    4.0
        dtype: float64

        >>> s.expanding().rank(method="min")
        0    1.0
        1    2.0
        2    2.0
        3    3.0
        4    5.0
        5    3.0
        dtype: float64
        """
        return super().rank(
            method=method,
            ascending=ascending,
            pct=pct,
            numeric_only=numeric_only,
        )

    def nunique(
        self,
        numeric_only: bool = False,
    ):
        # [FUNCTION DOC]
        # Expanding count of distinct values in each window. numeric_only
        # (default false) limits to numeric columns. Delegates to the parent;
        # returns float-valued results.
        """
        Calculate the expanding nunique.

        .. versionadded:: 3.0.0

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        Series.expanding : Calling expanding with Series data.
        DataFrame.expanding : Calling expanding with DataFrames.
        Series.nunique : Aggregating nunique for Series.
        DataFrame.nunique : Aggregating nunique for DataFrame.

        Examples
        --------
        >>> s = pd.Series([1, 4, 2, 3, 5, 3])
        >>> s.expanding().nunique()
        0    1.0
        1    2.0
        2    3.0
        3    4.0
        4    5.0
        5    5.0
        dtype: float64
        """
        return super().nunique(
            numeric_only=numeric_only,
        )

    def cov(
        self,
        other: DataFrame | Series | None = None,
        pairwise: bool | None = None,
        ddof: int = 1,
        numeric_only: bool = False,
    ):
        # [FUNCTION DOC]
        # Expanding sample covariance. other (default none) is the series or
        # frame to compare with; when absent, pairwise output with itself is
        # produced. pairwise (default none) selects between matching columns
        # only (false) and all column pair combinations as a multi-indexed
        # result (true), using only complete observations. ddof (default one)
        # is the delta degrees of freedom; numeric_only (default false) limits
        # to numeric columns. Delegates to the parent.
        """
        Calculate the expanding sample covariance.

        Parameters
        ----------
        other : Series or DataFrame, optional
            If not supplied then will default to self and produce pairwise
            output.
        pairwise : bool, default None
            If False then only matching columns between self and other will be
            used and the output will be a DataFrame.
            If True then all pairwise combinations will be calculated and the
            output will be a MultiIndexed DataFrame in the case of DataFrame
            inputs. In the case of missing elements, only complete pairwise
            observations will be used.
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        Series.expanding : Calling expanding with Series data.
        DataFrame.expanding : Calling expanding with DataFrames.
        Series.cov : Aggregating cov for Series.
        DataFrame.cov : Aggregating cov for DataFrame.

        Examples
        --------
        >>> ser1 = pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"])
        >>> ser2 = pd.Series([10, 11, 13, 16], index=["a", "b", "c", "d"])
        >>> ser1.expanding().cov(ser2)
        a         NaN
        b    0.500000
        c    1.500000
        d    3.333333
        dtype: float64
        """
        return super().cov(
            other=other,
            pairwise=pairwise,
            ddof=ddof,
            numeric_only=numeric_only,
        )

    def corr(
        self,
        other: DataFrame | Series | None = None,
        pairwise: bool | None = None,
        ddof: int = 1,
        numeric_only: bool = False,
    ):
        # [FUNCTION DOC]
        # Expanding Pearson correlation; same parameters (other, pairwise,
        # ddof, numeric_only) and delegation as the expanding covariance.
        # Equal-valued sequences give missing results because of a zero by
        # zero division.
        """
        Calculate the expanding correlation.

        Parameters
        ----------
        other : Series or DataFrame, optional
            If not supplied then will default to self and produce pairwise
            output.
        pairwise : bool, default None
            If False then only matching columns between self and other will be
            used and the output will be a DataFrame.
            If True then all pairwise combinations will be calculated and the
            output will be a MultiIndexed DataFrame in the case of DataFrame
            inputs. In the case of missing elements, only complete pairwise
            observations will be used.
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.

        numeric_only : bool, default False
            Include only float, int, boolean columns.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        cov : Similar method to calculate covariance.
        numpy.corrcoef : NumPy Pearson's correlation calculation.
        Series.expanding : Calling expanding with Series data.
        DataFrame.expanding : Calling expanding with DataFrames.
        Series.corr : Aggregating corr for Series.
        DataFrame.corr : Aggregating corr for DataFrame.

        Notes
        -----

        This function uses Pearson's definition of correlation
        (https://en.wikipedia.org/wiki/Pearson_correlation_coefficient).

        When `other` is not specified, the output will be self correlation (e.g.
        all 1's), except for :class:`~pandas.DataFrame` inputs with `pairwise`
        set to `True`.

        Function will return ``NaN`` for correlations of equal valued sequences;
        this is the result of a 0/0 division error.

        When `pairwise` is set to `False`, only matching columns between `self` and
        `other` will be used.

        When `pairwise` is set to `True`, the output will be a MultiIndex DataFrame
        with the original index on the first level, and the `other` DataFrame
        columns on the second level.

        In the case of missing elements, only complete pairwise observations
        will be used.

        Examples
        --------
        >>> ser1 = pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"])
        >>> ser2 = pd.Series([10, 11, 13, 16], index=["a", "b", "c", "d"])
        >>> ser1.expanding().corr(ser2)
        a         NaN
        b    1.000000
        c    0.981981
        d    0.975900
        dtype: float64
        """
        return super().corr(
            other=other,
            pairwise=pairwise,
            ddof=ddof,
            numeric_only=numeric_only,
        )


@set_module("pandas.api.typing")
class ExpandingGroupby(BaseWindowGroupby, Expanding):
    # [CLASS DOC]
    # Expanding window computed independently within each group of a groupby
    # object. Inherits group handling from the window groupby base and the
    # statistics from the expanding class (base order matters: group base
    # first). The attribute list is the concatenation of both parents' lists.
    # No new methods other than the indexer hook.
    """
    Provide an expanding groupby implementation.
    """

    _attributes = Expanding._attributes + BaseWindowGroupby._attributes

    def _get_window_indexer(self) -> GroupbyIndexer:
        # [FUNCTION DOC]
        # HELPER: hook for the window machinery. Returns a groupby indexer
        # built from the grouper's group-to-row-positions mapping and the
        # expanding indexer class, so the expanding bounds are computed per
        # group.
        """
        Return an indexer class that will compute the window start and end bounds

        Returns
        -------
        GroupbyIndexer
        """
        window_indexer = GroupbyIndexer(
            groupby_indices=self._grouper.indices,
            window_indexer=ExpandingIndexer,
        )
        return window_indexer
