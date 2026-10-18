# [MODULE DOC]
# Exponentially weighted (EW) window calculations for Series and DataFrame.
# Public contents: ExponentialMovingWindow (the object returned by the ewm method,
# with mean, sum, std, var, cov, corr and aggregate), ExponentialMovingWindowGroupby
# (the groupby variant) and OnlineExponentialMovingWindow (stateful incremental mean
# computation, numba only). Helper functions: get_center_of_mass (normalizes the four
# decay specifications to a center of mass) and a private function computing scaled
# time deltas.
# Relationships: subclasses the base window classes in the rolling module; uses the
# compiled window aggregation kernels, numba kernel generators from the window numba
# module, the online state class, the EW window indexers and the numba utilities.
# No import-time side effects.
from __future__ import annotations

import datetime
from functools import partial
from typing import (
    TYPE_CHECKING,
    cast,
)

import numpy as np

from pandas._libs.tslibs import Timedelta
import pandas._libs.window.aggregations as window_aggregations
from pandas.util._decorators import set_module

from pandas.core.dtypes.common import (
    is_datetime64_dtype,
    is_numeric_dtype,
)
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import isna

from pandas.core import common
from pandas.core.arrays.datetimelike import dtype_to_unit
from pandas.core.indexers.objects import (
    BaseIndexer,
    ExponentialMovingWindowIndexer,
    GroupbyIndexer,
)
from pandas.core.util.numba_ import (
    get_jit_arguments,
    maybe_use_numba,
)
from pandas.core.window.common import zsqrt
from pandas.core.window.numba_ import (
    generate_numba_ewm_func,
    generate_numba_ewm_table_func,
)
from pandas.core.window.online import (
    EWMMeanState,
    generate_online_numba_ewma_func,
)
from pandas.core.window.rolling import (
    BaseWindow,
    BaseWindowGroupby,
)

if TYPE_CHECKING:
    from pandas._typing import (
        TimedeltaConvertibleTypes,
        TimeUnit,
        npt,
    )

    from pandas import (
        DataFrame,
        Series,
    )
    from pandas.core.generic import NDFrame


def get_center_of_mass(
    comass: float | None,
    span: float | None,
    halflife: float | None,
    alpha: float | None,
) -> float:
    # [FUNCTION DOC]
    # Convert exactly one of the four EW decay specifications into a center of mass.
    # Inputs: comass, span, halflife (numeric, in observations) and alpha, each
    # optional (None means not given).
    # Raises a value error if more than one is given, if none is given, or if the one
    # given violates its domain (center of mass at least 0; span at least 1; halflife
    # strictly positive; alpha strictly greater than 0 and at most 1).
    # Conversions: from span, (span - 1) / 2; from halflife, the decay is one minus
    # exp(ln(0.5) / halflife) and the center of mass is the reciprocal of the decay
    # minus one; from alpha, (1 - alpha) / alpha; a given center of mass is returned
    # unchanged. Returns a Python float.
    valid_count = common.count_not_none(comass, span, halflife, alpha)
    if valid_count > 1:
        raise ValueError("comass, span, halflife, and alpha are mutually exclusive")

    # Convert to center of mass; domain checks ensure 0 < alpha <= 1
    if comass is not None:
        if comass < 0:
            raise ValueError("comass must satisfy: comass >= 0")
    elif span is not None:
        if span < 1:
            raise ValueError("span must satisfy: span >= 1")
        comass = (span - 1) / 2
    elif halflife is not None:
        if halflife <= 0:
            raise ValueError("halflife must satisfy: halflife > 0")
        decay = 1 - np.exp(np.log(0.5) / halflife)
        comass = 1 / decay - 1
    elif alpha is not None:
        if alpha <= 0 or alpha > 1:
            raise ValueError("alpha must satisfy: 0 < alpha <= 1")
        comass = (1 - alpha) / alpha
    else:
        raise ValueError("Must pass one of comass, span, halflife, or alpha")

    return float(comass)


def _calculate_deltas(
    times: np.ndarray | NDFrame,
    halflife: float | TimedeltaConvertibleTypes | None,
) -> npt.NDArray[np.float64]:
    # [FUNCTION DOC]
    # HELPER: ExponentialMovingWindow (time-based decay). Computes the spacing between
    # consecutive observation times in units of the half-life.
    # Inputs: times, a datetime-like array or Series (a Series is unwrapped to its
    # underlying values); halflife, anything convertible to a Timedelta. The
    # half-life is converted to the resolution of the times' dtype, and the integer
    # representation of the times is cast to float.
    # Returns a float64 array of length one less than times (the consecutive
    # differences divided by the half-life as a number). No monotonicity validation
    # is done here.
    unit = dtype_to_unit(times.dtype) divided by the half-life. These values are used in
    the calculation of the ewm mean.

    Parameters
    ----------
    times : np.ndarray, Series
        Times corresponding to the observations. Must be monotonically increasing
        and ``datetime64[ns]`` dtype.
    halflife : float, str, timedelta, optional
        Half-life specifying the decay

    Returns
    -------
    np.ndarray
        Diff of the times divided by the half-life
    """
    unit = dtype_to_unit(times.dtype)
    unit = cast("TimeUnit", unit)
    if isinstance(times, ABCSeries):
        times = times._values
    _times = np.asarray(times.view(np.int64), dtype=np.float64)
    _halflife = float(Timedelta(halflife).as_unit(unit)._value)
    return np.diff(_times) / _halflife


@set_module("pandas.api.typing")
class ExponentialMovingWindow(BaseWindow):
    # [CLASS DOC]
    # Window object for exponentially weighted calculations, created by the ewm method
    # of Series and DataFrame. Subclass of the base window class, with fixed settings:
    # no centering, no closed-side choice and no step.
    # State: the decay arguments as given (com, span, halflife, alpha), adjust,
    # ignore_na, times, plus derived private state: the normalized center of mass and
    # an array of scaled time deltas (all ones, one fewer than the number of rows, when
    # no times are given). The minimum periods is stored as at least 1.
    # Key methods: mean and sum (cython or numba engines), var, std, cov, corr,
    # aggregate (alias agg) and online (create the stateful online variant).
    # Invariants: validity of the decay arguments and times is checked at construction.
    r"""
    Provide exponentially weighted (EW) calculations.

    Exactly one of ``com``, ``span``, ``halflife``, or ``alpha`` must be
    provided if ``times`` is not provided. If ``times`` is provided and ``adjust=True``,
    ``halflife`` and one of ``com``, ``span`` or ``alpha`` may be provided.
    If ``times`` is provided and ``adjust=False``, ``halflife`` must be the only
    provided decay-specification parameter.

    Parameters
    ----------
    com : float, optional
        Specify decay in terms of center of mass

        :math:`\alpha = 1 / (1 + com)`, for :math:`com \geq 0`.

    span : float, optional
        Specify decay in terms of span

        :math:`\alpha = 2 / (span + 1)`, for :math:`span \geq 1`.

    halflife : float, str, timedelta, optional
        Specify decay in terms of half-life

        :math:`\alpha = 1 - \exp\left(-\ln(2) / halflife\right)`, for
        :math:`halflife > 0`.

        If ``times`` is specified, a timedelta convertible unit over which an
        observation decays to half its value. Only applicable to ``mean()``,
        and halflife value will not apply to the other functions.

    alpha : float, optional
        Specify smoothing factor :math:`\alpha` directly

        :math:`0 < \alpha \leq 1`.

    min_periods : int, default 0
        Minimum number of observations in window required to have a value;
        otherwise, result is ``np.nan``.

    adjust : bool, default True
        Divide by decaying adjustment factor in beginning periods to account
        for imbalance in relative weightings (viewing EWMA as a moving average).

        - When ``adjust=True`` (default), the EW function is calculated using weights
          :math:`w_i = (1 - \alpha)^i`. For example, the EW moving average of the series
          [:math:`x_0, x_1, ..., x_t`] would be:

        .. math::
            y_t = \frac{x_t + (1 - \alpha)x_{t-1} + (1 - \alpha)^2 x_{t-2} + ... + (1 -
            \alpha)^t x_0}{1 + (1 - \alpha) + (1 - \alpha)^2 + ... + (1 - \alpha)^t}

        - When ``adjust=False``, the exponentially weighted function is calculated
          recursively:

        .. math::
            \begin{split}
                y_0 &= x_0\\
                y_t &= (1 - \alpha) y_{t-1} + \alpha x_t,
            \end{split}
    ignore_na : bool, default False
        Ignore missing values when calculating weights.

        - When ``ignore_na=False`` (default), weights are based on absolute positions.
          For example, the weights of :math:`x_0` and :math:`x_2` used in calculating
          the final weighted average of [:math:`x_0`, None, :math:`x_2`] are
          :math:`(1-\alpha)^2` and :math:`1` if ``adjust=True``, and
          :math:`(1-\alpha)^2` and :math:`\alpha` if ``adjust=False``.

        - When ``ignore_na=True``, weights are based
          on relative positions. For example, the weights of :math:`x_0` and :math:`x_2`
          used in calculating the final weighted average of
          [:math:`x_0`, None, :math:`x_2`] are :math:`1-\alpha` and :math:`1` if
          ``adjust=True``, and :math:`1-\alpha` and :math:`\alpha` if ``adjust=False``.

    times : np.ndarray, Series, default None

        Only applicable to ``mean()``.

        Times corresponding to the observations. Must be monotonically increasing and
        ``datetime64[ns]`` dtype.

        If 1-D array like, a sequence with the same shape as the observations.

    method : str {'single', 'table'}, default 'single'
        Execute the rolling operation per single column or row (``'single'``)
        or over the entire object (``'table'``).

        This argument is only implemented when specifying ``engine='numba'``
        in the method call.

        Only applicable to ``mean()``

    Returns
    -------
    pandas.api.typing.ExponentialMovingWindow
        An instance of ExponentialMovingWindow for further exponentially weighted (EW)
        calculations, e.g. using the ``mean`` method.

    See Also
    --------
    rolling : Provides rolling window calculations.
    expanding : Provides expanding transformations.

    Notes
    -----
    See :ref:`Windowing Operations <window.exponentially_weighted>`
    for further usage details and examples.

    Examples
    --------
    >>> df = pd.DataFrame({'B': [0, 1, 2, np.nan, 4]})
    >>> df
         B
    0  0.0
    1  1.0
    2  2.0
    3  NaN
    4  4.0

    >>> df.ewm(com=0.5).mean()
              B
    0  0.000000
    1  0.750000
    2  1.615385
    3  1.615385
    4  3.670213
    >>> df.ewm(alpha=2 / 3).mean()
              B
    0  0.000000
    1  0.750000
    2  1.615385
    3  1.615385
    4  3.670213

    **adjust**

    >>> df.ewm(com=0.5, adjust=True).mean()
              B
    0  0.000000
    1  0.750000
    2  1.615385
    3  1.615385
    4  3.670213
    >>> df.ewm(com=0.5, adjust=False).mean()
              B
    0  0.000000
    1  0.666667
    2  1.555556
    3  1.555556
    4  3.650794

    **ignore_na**

    >>> df.ewm(com=0.5, ignore_na=True).mean()
              B
    0  0.000000
    1  0.750000
    2  1.615385
    3  1.615385
    4  3.225000
    >>> df.ewm(com=0.5, ignore_na=False).mean()
              B
    0  0.000000
    1  0.750000
    2  1.615385
    3  1.615385
    4  3.670213

    **times**

    Exponentially weighted mean with weights calculated with a timedelta ``halflife``
    relative to ``times``.

    >>> times = ['2020-01-01', '2020-01-03', '2020-01-10', '2020-01-15', '2020-01-17']
    >>> df.ewm(halflife='4 days', times=pd.DatetimeIndex(times)).mean()
              B
    0  0.000000
    1  0.585786
    2  1.523889
    3  1.523889
    4  3.233686
    """

    _attributes = [
        "com",
        "span",
        "halflife",
        "alpha",
        "min_periods",
        "adjust",
        "ignore_na",
        "times",
        "method",
    ]

    def __init__(
        self,
        obj: NDFrame,
        com: float | None = None,
        span: float | None = None,
        halflife: float | TimedeltaConvertibleTypes | None = None,
        alpha: float | None = None,
        min_periods: int | None = 0,
        adjust: bool = True,
        ignore_na: bool = False,
        times: np.ndarray | NDFrame | None = None,
        method: str = "single",
        *,
        selection=None,
    ) -> None:
        # [FUNCTION DOC]
        # Construct the EW window and validate its arguments.
        # Inputs: obj (Series or DataFrame); com, span, halflife, alpha (decay
        # specification, see the class docs and the center-of-mass helper); min_periods
        # (default 0; None or any value below one is stored as one); adjust (default
        # true); ignore_na (default false); times (optional datetime64 array or Series
        # aligned with the rows); method ("single" or "table"); selection (keyword-only,
        # the selected columns for groupby selections).
        # With times given: raises a value error if the dtype is not datetime64 (naive
        # or timezone-aware), if the length differs from obj, if halflife is not a
        # string, timedelta or numpy timedelta, or if times contain missing values;
        # computes the scaled deltas; if any of com, span or alpha is also given it
        # raises a not-implemented error when adjust is false, otherwise normalizes them
        # (ignoring halflife) to the center of mass; if none is given the center of mass
        # is 1.
        # Without times: raises a value error if halflife is a timedelta-like; the
        # deltas are all ones; the center of mass is computed from exactly one of the
        # four decay arguments (errors from that helper propagate).
        super().__init__(
            obj=obj,
            min_periods=1 if min_periods is None else max(int(min_periods), 1),
            on=None,
            center=False,
            closed=None,
            method=method,
            selection=selection,
        )
        self.com = com
        self.span = span
        self.halflife = halflife
        self.alpha = alpha
        self.adjust = adjust
        self.ignore_na = ignore_na
        self.times = times
        if self.times is not None:
            times_dtype = getattr(self.times, "dtype", None)
            if not (
                is_datetime64_dtype(times_dtype)
                or isinstance(times_dtype, DatetimeTZDtype)
            ):
                raise ValueError("times must be datetime64 dtype.")
            if len(self.times) != len(obj):
                raise ValueError("times must be the same length as the object.")
            if not isinstance(self.halflife, (str, datetime.timedelta, np.timedelta64)):
                raise ValueError("halflife must be a timedelta convertible object")
            if isna(self.times).any():
                raise ValueError("Cannot convert NaT values to integer")
            self._deltas = _calculate_deltas(self.times, self.halflife)
            # Halflife is no longer applicable when calculating COM
            # But allow COM to still be calculated if the user passes other decay args
            if common.count_not_none(self.com, self.span, self.alpha) > 0:
                if not self.adjust:
                    raise NotImplementedError(
                        "None of com, span, or alpha can be specified if "
                        "times is provided and adjust=False"
                    )
                self._com = get_center_of_mass(self.com, self.span, None, self.alpha)
            else:
                self._com = 1.0
        else:
            if self.halflife is not None and isinstance(
                self.halflife, (str, datetime.timedelta, np.timedelta64)
            ):
                raise ValueError(
                    "halflife can only be a timedelta convertible argument if "
                    "times is not None."
                )
            # Without times, points are equally spaced
            self._deltas = np.ones(max(self.obj.shape[0] - 1, 0), dtype=np.float64)
            self._com = get_center_of_mass(
                # error: Argument 3 to "get_center_of_mass" has incompatible type
                # "Union[float, Any, None, timedelta64, signedinteger[_64Bit]]";
                # expected "Optional[float]"
                self.com,
                self.span,
                self.halflife,  # type: ignore[arg-type]
                self.alpha,
            )

    def _check_window_bounds(
        self, start: np.ndarray, end: np.ndarray, num_vals: int
    ) -> None:
        # [FUNCTION DOC]
        # Override of the base window-bounds validation that intentionally does nothing.
        # Inputs (start, end, num_vals) are ignored. The EW algorithms iterate over
        # every point and their indexer's bounds always cover the whole series, so no
        # check applies. Returns nothing; no exceptions or side effects.
        # emw algorithms are iterative with each point
        # ExponentialMovingWindowIndexer "bounds" are the entire window
        pass

    def _get_window_indexer(self) -> BaseIndexer:
        # [FUNCTION DOC]
        # HELPER: window computation. Returns a new EW window indexer whose bounds
        # cover the entire series for every point. Takes no inputs besides self.
        """
        Return an indexer class that will compute the window start and end bounds
        """
        return ExponentialMovingWindowIndexer()

    def online(
        self, engine: str = "numba", engine_kwargs=None
    ) -> OnlineExponentialMovingWindow:
        # [FUNCTION DOC]
        # Create a stateful online EW window from this object's settings.
        # Inputs: engine (default "numba"; only numba is supported) and engine_kwargs
        # (optional mapping of numba jit options). Returns a new online EW window
        # constructed from the same object, decay arguments, min_periods, adjust,
        # ignore_na, times and column selection. Errors (such as a not-implemented error
        # when times is set, or a value error for an unsupported engine) come from the
        # online class constructor.
        """
        Return an ``OnlineExponentialMovingWindow`` object to calculate
        exponentially moving window aggregations in an online method.

        Parameters
        ----------
        engine: str, default ``'numba'``
            Execution engine to calculate online aggregations.
            Applies to all supported aggregation methods.

        engine_kwargs : dict, default None
            Applies to all supported aggregation methods.

            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{'nopython': True, 'nogil': False, 'parallel': False}`` and will be
              applied to the function

        Returns
        -------
        OnlineExponentialMovingWindow
        """
        return OnlineExponentialMovingWindow(
            obj=self.obj,
            com=self.com,
            span=self.span,
            halflife=self.halflife,
            alpha=self.alpha,
            min_periods=self.min_periods,
            adjust=self.adjust,
            ignore_na=self.ignore_na,
            times=self.times,
            engine=engine,
            engine_kwargs=engine_kwargs,
            selection=self._selection,
        )

    def aggregate(self, func=None, *args, **kwargs):
        # [FUNCTION DOC]
        # Aggregate using one or more functions; a thin pass-through to the base
        # window class. Inputs: func (function, name, list or mapping), plus extra
        # positional and keyword arguments forwarded to it. Returns whatever the base
        # implementation returns (scalar, Series or DataFrame). The agg alias is bound
        # to this method.
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
        pandas.DataFrame.rolling.aggregate

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

        >>> df.ewm(alpha=0.5).mean()
                  A         B         C
        0  1.000000  4.000000  7.000000
        1  1.666667  4.666667  7.666667
        2  2.428571  5.428571  8.428571
        """
        return super().aggregate(func, *args, **kwargs)

    agg = aggregate

    def mean(
        self,
        numeric_only: bool = False,
        engine=None,
        engine_kwargs=None,
    ):
        # [FUNCTION DOC]
        # Compute the exponentially weighted mean (weights normalized).
        # Inputs: numeric_only (default false); engine (None, "cython" or "numba";
        # None uses numba only if the global numba option is on); engine_kwargs (numba
        # jit options; must be None for cython).
        # Numba path: uses the single-column or table kernel according to the method
        # setting, passing the center of mass, adjust, ignore_na and the deltas as a
        # tuple. Cython path: raises a value error if engine_kwargs is given, then uses
        # the compiled kernel, passing deltas only when times were given. Any other
        # engine raises a value error. Returns a Series or DataFrame of float64.
        # KNOWN ISSUE: the numba path does not pass numeric_only on to the apply step,
        # so that argument is silently ignored with the numba engine.
        """
        Calculate the ewm (exponential weighted moment) mean.

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
        Series.ewm : Calling ewm with Series data.
        DataFrame.ewm : Calling ewm with DataFrames.
        Series.mean : Aggregating mean for Series.
        DataFrame.mean : Aggregating mean for DataFrame.

        Notes
        -----
        See :ref:`window.numba_engine` and :ref:`enhancingperf.numba` for
        extended documentation and performance considerations for the Numba engine.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3, 4])
        >>> ser.ewm(alpha=0.2).mean()
        0    1.000000
        1    1.555556
        2    2.147541
        3    2.775068
        dtype: float64
        """
        if maybe_use_numba(engine):
            if self.method == "single":
                func = generate_numba_ewm_func
            else:
                func = generate_numba_ewm_table_func
            ewm_func = func(
                **get_jit_arguments(engine_kwargs),
                com=self._com,
                adjust=self.adjust,
                ignore_na=self.ignore_na,
                deltas=tuple(self._deltas),
                normalize=True,
            )
            return self._apply(ewm_func, name="mean")
        elif engine in ("cython", None):
            if engine_kwargs is not None:
                raise ValueError("cython engine does not accept engine_kwargs")

            deltas = None if self.times is None else self._deltas
            window_func = partial(
                window_aggregations.ewm,
                com=self._com,
                adjust=self.adjust,
                ignore_na=self.ignore_na,
                deltas=deltas,
                normalize=True,
            )
            return self._apply(window_func, name="mean", numeric_only=numeric_only)
        else:
            raise ValueError("engine must be either 'numba' or 'cython'")

    def sum(
        self,
        numeric_only: bool = False,
        engine=None,
        engine_kwargs=None,
    ):
        # [FUNCTION DOC]
        # Compute the exponentially weighted sum (weights not normalized).
        # Same inputs and engine handling as the mean method, except that it first
        # raises a not-implemented error if adjust is false or if times were given.
        # Returns a Series or DataFrame of float64.
        # KNOWN ISSUE: as in the mean method, the numba path does not forward
        # numeric_only, so it is ignored with that engine.
        """
        Calculate the ewm (exponential weighted moment) sum.

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
        Series.ewm : Calling ewm with Series data.
        DataFrame.ewm : Calling ewm with DataFrames.
        Series.sum : Aggregating sum for Series.
        DataFrame.sum : Aggregating sum for DataFrame.

        Notes
        -----
        See :ref:`window.numba_engine` and :ref:`enhancingperf.numba` for extended
        documentation and performance considerations for the Numba engine.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3, 4])
        >>> ser.ewm(alpha=0.2).sum()
        0    1.000
        1    2.800
        2    5.240
        3    8.192
        dtype: float64
        """
        if not self.adjust:
            raise NotImplementedError("sum is not implemented with adjust=False")
        if self.times is not None:
            raise NotImplementedError("sum is not implemented with times")
        if maybe_use_numba(engine):
            if self.method == "single":
                func = generate_numba_ewm_func
            else:
                func = generate_numba_ewm_table_func
            ewm_func = func(
                **get_jit_arguments(engine_kwargs),
                com=self._com,
                adjust=self.adjust,
                ignore_na=self.ignore_na,
                deltas=tuple(self._deltas),
                normalize=False,
            )
            return self._apply(ewm_func, name="sum")
        elif engine in ("cython", None):
            if engine_kwargs is not None:
                raise ValueError("cython engine does not accept engine_kwargs")

            deltas = None if self.times is None else self._deltas
            window_func = partial(
                window_aggregations.ewm,
                com=self._com,
                adjust=self.adjust,
                ignore_na=self.ignore_na,
                deltas=deltas,
                normalize=False,
            )
            return self._apply(window_func, name="sum", numeric_only=numeric_only)
        else:
            raise ValueError("engine must be either 'numba' or 'cython'")

    def std(self, bias: bool = False, numeric_only: bool = False):
        # [FUNCTION DOC]
        # Compute the EW standard deviation as the square root of the EW variance.
        # Inputs: bias (default false; use the biased estimator when true);
        # numeric_only (default false). Raises a not-implemented error when
        # numeric_only is set and the selected object is a non-numeric Series (so the
        # message names std rather than var), and when times were given. Negative
        # variances are clipped to zero before the square root. Returns float64 data.
        """
        Calculate the ewm (exponential weighted moment) standard deviation.

        Parameters
        ----------
        bias : bool, default False
            Use a standard estimation bias correction.
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        Series.ewm : Calling ewm with Series data.
        DataFrame.ewm : Calling ewm with DataFrames.
        Series.std : Aggregating std for Series.
        DataFrame.std : Aggregating std for DataFrame.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3, 4])
        >>> ser.ewm(alpha=0.2).std()
        0         NaN
        1    0.707107
        2    0.995893
        3    1.277320
        dtype: float64
        """
        if (
            numeric_only
            and self._selected_obj.ndim == 1
            and not is_numeric_dtype(self._selected_obj.dtype)
        ):
            # Raise directly so error message says std instead of var
            raise NotImplementedError(
                f"{type(self).__name__}.std does not implement numeric_only"
            )
        if self.times is not None:
            raise NotImplementedError("std is not implemented with times")
        return zsqrt(self.var(bias=bias, numeric_only=numeric_only))

    def var(self, bias: bool = False, numeric_only: bool = False):
        # [FUNCTION DOC]
        # Compute the EW variance as the EW covariance of each column with itself.
        # Inputs: bias (default false); numeric_only (default false). Raises a
        # not-implemented error when times were given. Returns float64 data.
        """
        Calculate the ewm (exponential weighted moment) variance.

        Parameters
        ----------
        bias : bool, default False
            Use a standard estimation bias correction.
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        Series.ewm : Calling ewm with Series data.
        DataFrame.ewm : Calling ewm with DataFrames.
        Series.var : Aggregating var for Series.
        DataFrame.var : Aggregating var for DataFrame.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3, 4])
        >>> ser.ewm(alpha=0.2).var()
        0         NaN
        1    0.500000
        2    0.991803
        3    1.631547
        dtype: float64
        """
        if self.times is not None:
            raise NotImplementedError("var is not implemented with times")
        window_func = window_aggregations.ewmcov
        wfunc = partial(
            window_func,
            com=self._com,
            adjust=self.adjust,
            ignore_na=self.ignore_na,
            bias=bias,
        )

        def var_func(values, begin, end, min_periods):
            # [FUNCTION DOC]
            # HELPER: var. Kernel adapter: calls the partially bound covariance kernel
            # with the same array as both operands. Inputs: values, window start and
            # end bound arrays, and the minimum periods; returns the kernel's output.
            return wfunc(values, begin, end, min_periods, values)

        return self._apply(var_func, name="var", numeric_only=numeric_only)

    def cov(
        self,
        other: DataFrame | Series | None = None,
        pairwise: bool | None = None,
        bias: bool = False,
        numeric_only: bool = False,
    ):
        # [FUNCTION DOC]
        # Compute the EW sample covariance with another object (or pairwise with itself).
        # Inputs: other (Series or DataFrame; default none meaning itself); pairwise
        # (default none; whether to compute all column pairs); bias (default false);
        # numeric_only (default false). Raises a not-implemented error when times were
        # given, and validates numeric_only. Computation is delegated to the base
        # pairwise machinery with a per-pair function that converts both inputs to
        # float arrays, obtains window bounds from the indexer and calls the compiled
        # covariance kernel, returning a float Series. Returns a Series or DataFrame.
        # KNOWN ISSUE: inside the per-pair function a local minimum-periods value is
        # computed (falling back to the indexer window size) and used for the bounds,
        # but the kernel is called with the stored minimum periods attribute instead.
        """
        Calculate the ewm (exponential weighted moment) sample covariance.

        Parameters
        ----------
        other : Series or DataFrame , optional
            If not supplied then will default to self and produce pairwise
            output.
        pairwise : bool, default None
            If False then only matching columns between self and other will be
            used and the output will be a DataFrame.
            If True then all pairwise combinations will be calculated and the
            output will be a MultiIndex DataFrame in the case of DataFrame
            inputs. In the case of missing elements, only complete pairwise
            observations will be used.
        bias : bool, default False
            Use a standard estimation bias correction.
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        Series.ewm : Calling ewm with Series data.
        DataFrame.ewm : Calling ewm with DataFrames.
        Series.cov : Aggregating cov for Series.
        DataFrame.cov : Aggregating cov for DataFrame.

        Examples
        --------
        >>> ser1 = pd.Series([1, 2, 3, 4])
        >>> ser2 = pd.Series([10, 11, 13, 16])
        >>> ser1.ewm(alpha=0.2).cov(ser2)
        0         NaN
        1    0.500000
        2    1.524590
        3    3.408836
        dtype: float64
        """
        if self.times is not None:
            raise NotImplementedError("cov is not implemented with times")

        from pandas import Series

        self._validate_numeric_only("cov", numeric_only)

        def cov_func(x, y):
            # [FUNCTION DOC]
            # HELPER: cov. Per-pair covariance. Inputs: x and y, two aligned Series.
            # Returns a float Series with the EW covariance (bias setting from the
            # enclosing call), indexed and named like x.
            x_array = self._prep_values(x)
            y_array = self._prep_values(y)
            window_indexer = self._get_window_indexer()
            min_periods = (
                self.min_periods
                if self.min_periods is not None
                else window_indexer.window_size
            )
            start, end = window_indexer.get_window_bounds(
                num_values=len(x_array),
                min_periods=min_periods,
                center=self.center,
                closed=self.closed,
                step=self.step,
            )
            result = window_aggregations.ewmcov(
                x_array,
                start,
                end,
                # error: Argument 4 to "ewmcov" has incompatible type
                # "Optional[int]"; expected "int"
                self.min_periods,  # type: ignore[arg-type]
                y_array,
                self._com,
                self.adjust,
                self.ignore_na,
                bias,
            )
            return Series(result, index=x.index, name=x.name, copy=False)

        return self._apply_pairwise(
            self._selected_obj, other, pairwise, cov_func, numeric_only
        )

    def corr(
        self,
        other: DataFrame | Series | None = None,
        pairwise: bool | None = None,
        numeric_only: bool = False,
    ):
        # [FUNCTION DOC]
        # Compute the EW sample correlation with another object (or pairwise with
        # itself).
        # Inputs: other (Series or DataFrame; default none meaning itself); pairwise
        # (default none); numeric_only (default false). Raises a not-implemented error
        # when times were given, and validates numeric_only. For each pair, the
        # covariance of the pair divided by the square root of the product of the two
        # variances is computed using the biased covariance kernel (errors suppressed,
        # negative products clipped to zero). Returns a Series or DataFrame of float64.
        """
        Calculate the ewm (exponential weighted moment) sample correlation.

        Parameters
        ----------
        other : Series or DataFrame, optional
            If not supplied then will default to self and produce pairwise
            output.
        pairwise : bool, default None
            If False then only matching columns between self and other will be
            used and the output will be a DataFrame.
            If True then all pairwise combinations will be calculated and the
            output will be a MultiIndex DataFrame in the case of DataFrame
            inputs. In the case of missing elements, only complete pairwise
            observations will be used.
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        Series.ewm : Calling ewm with Series data.
        DataFrame.ewm : Calling ewm with DataFrames.
        Series.corr : Aggregating corr for Series.
        DataFrame.corr : Aggregating corr for DataFrame.

        Examples
        --------
        >>> ser1 = pd.Series([1, 2, 3, 4])
        >>> ser2 = pd.Series([10, 11, 13, 16])
        >>> ser1.ewm(alpha=0.2).corr(ser2)
        0         NaN
        1    1.000000
        2    0.982821
        3    0.977802
        dtype: float64
        """
        if self.times is not None:
            raise NotImplementedError("corr is not implemented with times")

        from pandas import Series

        self._validate_numeric_only("corr", numeric_only)

        def cov_func(x, y):
            # [FUNCTION DOC]
            # HELPER: corr. Per-pair correlation (despite the name, it returns a
            # correlation). Inputs: x and y, two aligned Series. Returns a float Series
            # indexed and named like x.
            x_array = self._prep_values(x)
            y_array = self._prep_values(y)
            window_indexer = self._get_window_indexer()
            min_periods = (
                self.min_periods
                if self.min_periods is not None
                else window_indexer.window_size
            )
            start, end = window_indexer.get_window_bounds(
                num_values=len(x_array),
                min_periods=min_periods,
                center=self.center,
                closed=self.closed,
                step=self.step,
            )

            def _cov(X, Y):
                # [FUNCTION DOC]
                # HELPER: corr. Biased EW covariance of two float arrays over the
                # precomputed window bounds; returns a float array.
                return window_aggregations.ewmcov(
                    X,
                    start,
                    end,
                    min_periods,
                    Y,
                    self._com,
                    self.adjust,
                    self.ignore_na,
                    True,
                )

            with np.errstate(all="ignore"):
                cov = _cov(x_array, y_array)
                x_var = _cov(x_array, x_array)
                y_var = _cov(y_array, y_array)
                result = cov / zsqrt(x_var * y_var)
            return Series(result, index=x.index, name=x.name, copy=False)

        return self._apply_pairwise(
            self._selected_obj, other, pairwise, cov_func, numeric_only
        )


@set_module("pandas.api.typing")
class ExponentialMovingWindowGroupby(BaseWindowGroupby, ExponentialMovingWindow):
    # [CLASS DOC]
    # EW window applied within each group of a groupby. Inherits from the groupby base
    # window class and the EW window class (groupby behavior first). Its attribute list
    # is the union of both parents' lists. Adds no new public methods: the parents
    # supply the calculations and this class supplies the grouped indexer and
    # group-ordered deltas.
    """
    Provide an exponential moving window groupby implementation.
    """

    _attributes = ExponentialMovingWindow._attributes + BaseWindowGroupby._attributes

    def __init__(self, obj, *args, _grouper=None, **kwargs) -> None:
        # [FUNCTION DOC]
        # Construct the grouped EW window. Inputs: obj (the object to window), extra
        # positional and keyword arguments passed through to the parents, and _grouper
        # (keyword-only, the groupby grouper). If obj is not empty and times were given,
        # the times are reordered into group order (all group row positions
        # concatenated) and the scaled deltas are recomputed from the reordered times,
        # replacing those set by the parent constructor.
        super().__init__(obj, *args, _grouper=_grouper, **kwargs)

        if not obj.empty and self.times is not None:
            # sort the times and recalculate the deltas according to the groups
            groupby_order = np.concatenate(list(self._grouper.indices.values()))
            self._deltas = _calculate_deltas(
                self.times.take(groupby_order),
                self.halflife,
            )

    def _get_window_indexer(self) -> GroupbyIndexer:
        # [FUNCTION DOC]
        # HELPER: window computation. Returns a new grouped indexer built from the
        # group row positions of the grouper, wrapping the EW indexer class for use
        # within each group.
        """
        Return an indexer class that will compute the window start and end bounds

        Returns
        -------
        GroupbyIndexer
        """
        window_indexer = GroupbyIndexer(
            groupby_indices=self._grouper.indices,
            window_indexer=ExponentialMovingWindowIndexer,
        )
        return window_indexer


class OnlineExponentialMovingWindow(ExponentialMovingWindow):
    # [CLASS DOC]
    # Stateful EW window supporting incremental mean computation, created by the online
    # method of the EW window. Subclass of the EW window class, supporting only the
    # numba engine and no times.
    # State: the engine name and engine keyword arguments, and a private mean-state
    # object (holding the last EW values and weights) that persists between mean calls.
    # Public behavior: mean (initial computation, or continuation with an update) and
    # reset (clear the state). The aggregate, std, corr, cov and var methods are
    # overridden to raise a not-implemented error.
    def __init__(
        self,
        obj: NDFrame,
        com: float | None = None,
        span: float | None = None,
        halflife: float | TimedeltaConvertibleTypes | None = None,
        alpha: float | None = None,
        min_periods: int | None = 0,
        adjust: bool = True,
        ignore_na: bool = False,
        times: np.ndarray | NDFrame | None = None,
        engine: str = "numba",
        engine_kwargs: dict[str, bool] | None = None,
        *,
        selection=None,
    ) -> None:
        # [FUNCTION DOC]
        # Construct the online EW window. Inputs are as for the parent constructor,
        # except there is no method argument, plus engine (default "numba") and
        # engine_kwargs (optional numba jit options).
        # Raises a not-implemented error if times is given. After parent construction it
        # creates the mean-state object from the center of mass, adjust, ignore_na and
        # the object shape. It raises a value error unless the numba engine is selected
        # (explicitly, or by default through the global numba option).
        if times is not None:
            raise NotImplementedError(
                "times is not implemented with online operations."
            )
        super().__init__(
            obj=obj,
            com=com,
            span=span,
            halflife=halflife,
            alpha=alpha,
            min_periods=min_periods,
            adjust=adjust,
            ignore_na=ignore_na,
            times=times,
            selection=selection,
        )
        self._mean = EWMMeanState(self._com, self.adjust, self.ignore_na, obj.shape)
        if maybe_use_numba(engine):
            self.engine = engine
            self.engine_kwargs = engine_kwargs
        else:
            raise ValueError("'numba' is the only supported engine")

    def reset(self) -> None:
        # [FUNCTION DOC]
        # Clear the stored online state (last EW values and weights) so the next mean
        # call starts afresh. Takes no inputs, returns nothing; mutates the private
        # mean-state object.
        """
        Reset the state captured by `update` calls.
        """
        self._mean.reset()

    def aggregate(self, func=None, *args, **kwargs):
        # [FUNCTION DOC]
        # Unsupported override: always raises a not-implemented error; all arguments
        # are ignored.
        raise NotImplementedError("aggregate is not implemented.")

    def std(self, bias: bool = False, *args, **kwargs):
        # [FUNCTION DOC]
        # Unsupported override: always raises a not-implemented error; all arguments
        # are ignored.
        raise NotImplementedError("std is not implemented.")

    def corr(
        self,
        other: DataFrame | Series | None = None,
        pairwise: bool | None = None,
        numeric_only: bool = False,
    ):
        # [FUNCTION DOC]
        # Unsupported override: always raises a not-implemented error; all arguments
        # are ignored.
        raise NotImplementedError("corr is not implemented.")

    def cov(
        self,
        other: DataFrame | Series | None = None,
        pairwise: bool | None = None,
        bias: bool = False,
        numeric_only: bool = False,
    ):
        # [FUNCTION DOC]
        # Unsupported override: always raises a not-implemented error; all arguments
        # are ignored.
        raise NotImplementedError("cov is not implemented.")

    def var(self, bias: bool = False, numeric_only: bool = False):
        # [FUNCTION DOC]
        # Unsupported override: always raises a not-implemented error; all arguments
        # are ignored.
        raise NotImplementedError("var is not implemented.")

    def mean(self, *args, update=None, update_times=None, **kwargs):
        # [FUNCTION DOC]
        # Compute the online EW mean, optionally continuing from saved state.
        # Inputs: extra positional and keyword arguments (accepted and ignored);
        # update (keyword-only, optional Series or DataFrame of new float values to
        # append to the previous state; must be none on the first call); update_times
        # (unsupported; any non-none value raises a not-implemented error).
        # Raises a value error if update is given before any initial mean call has
        # stored state. Without update, the whole selected object is cast to float64 and
        # the state is initialized from it; with update, the last stored EW value is
        # prepended as a seed row and only the rows from the update are returned. The
        # kernel is built from the stored engine options, deltas are all ones, and the
        # minimum periods setting applies. Side effect: the internal mean state is
        # advanced by the call. Returns a Series or DataFrame (same kind as the
        # selected object) indexed like the update (or the original object) and with
        # matching column or series names.
        """
        Calculate an online exponentially weighted mean.

        Parameters
        ----------
        update: DataFrame or Series, default None
            New values to continue calculating the
            exponentially weighted mean from the last values and weights.
            Values should be float64 dtype.

            ``update`` needs to be ``None`` the first time the
            exponentially weighted mean is calculated.

        update_times: Series or 1-D np.ndarray, default None
            New times to continue calculating the
            exponentially weighted mean from the last values and weights.
            If ``None``, values are assumed to be evenly spaced
            in time.
            This feature is currently unsupported.

        Returns
        -------
        DataFrame or Series

        Examples
        --------
        >>> df = pd.DataFrame({"a": range(5), "b": range(5, 10)})
        >>> online_ewm = df.head(2).ewm(0.5).online()
        >>> online_ewm.mean()
              a     b
        0  0.00  5.00
        1  0.75  5.75
        >>> online_ewm.mean(update=df.tail(3))
                  a         b
        2  1.615385  6.615385
        3  2.550000  7.550000
        4  3.520661  8.520661
        >>> online_ewm.reset()
        >>> online_ewm.mean()
              a     b
        0  0.00  5.00
        1  0.75  5.75
        """
        result_kwargs = {}
        is_frame = self._selected_obj.ndim == 2
        if update_times is not None:
            raise NotImplementedError("update_times is not implemented.")
        update_deltas = np.ones(
            max(self._selected_obj.shape[-1] - 1, 0), dtype=np.float64
        )
        if update is not None:
            if self._mean.last_ewm is None:
                raise ValueError(
                    "Must call mean with update=None first before passing update"
                )
            result_from = 1
            result_kwargs["index"] = update.index
            if is_frame:
                last_value = self._mean.last_ewm[np.newaxis, :]
                result_kwargs["columns"] = update.columns
            else:
                last_value = self._mean.last_ewm
                result_kwargs["name"] = update.name
            np_array = np.concatenate((last_value, update.to_numpy()))
        else:
            result_from = 0
            result_kwargs["index"] = self._selected_obj.index
            if is_frame:
                result_kwargs["columns"] = self._selected_obj.columns
            else:
                result_kwargs["name"] = self._selected_obj.name
            np_array = self._selected_obj.astype(np.float64).to_numpy()
        ewma_func = generate_online_numba_ewma_func(
            **get_jit_arguments(self.engine_kwargs)
        )
        result = self._mean.run_ewm(
            np_array if is_frame else np_array[:, np.newaxis],
            update_deltas,
            self.min_periods,
            ewma_func,
        )
        if not is_frame:
            result = result.squeeze()
        result = result[result_from:]
        result = self._selected_obj._constructor(result, **result_kwargs)
        return result
