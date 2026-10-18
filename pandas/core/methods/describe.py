"""
Module responsible for execution of NDFrame.describe() method.

Method NDFrame.describe() delegates actual execution to function describe_ndframe().
"""

# [MODULE DOC]
# Implementation of the describe summary-statistics method shared by Series and DataFrame.
# Public entry point is the top-level describe function (called by the generic frame describe method). The module defines an abstract describer base class with Series and DataFrame subclasses, per-dtype 1D describing functions (numeric, categorical, timestamp), a selector that picks the right 1D function for a Series, and helpers to order result rows and normalize percentiles.
# Depends on the frame concatenation routine, percentile validation and formatting utilities, and dtype predicates. No import-time side effects.

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    TYPE_CHECKING,
    cast,
)

import numpy as np

from pandas._typing import (
    DtypeObj,
    NDFrameT,
    npt,
)
from pandas.util._validators import validate_percentile

from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_numeric_dtype,
)
from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    DatetimeTZDtype,
    ExtensionDtype,
)

from pandas.core.arrays.floating import Float64Dtype
from pandas.core.reshape.concat import concat

from pandas.io.formats.format import format_percentiles

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
        Sequence,
    )

    from pandas import (
        DataFrame,
        Series,
    )


def describe_ndframe(
    *,
    obj: NDFrameT,
    include: str | Sequence[str] | None,
    exclude: str | Sequence[str] | None,
    percentiles: Sequence[float] | np.ndarray | None,
) -> NDFrameT:
    # [FUNCTION DOC]
    # Entry point: compute the describe summary for a Series or a DataFrame.
    # Parameters (keyword only): obj is the object to describe; include and exclude are dtype selections (or the string "all" for include) that apply only to frames and are ignored for Series; percentiles is a list-like of numbers in the unit interval, or None for the default quartiles.
    # Normalizes the percentiles (validated, deduplicated, sorted), then builds a Series describer for one-dimensional input or a frame describer otherwise, and returns its result: a Series of statistics for a Series, a DataFrame of statistics (one column per described column) for a frame.
    # Raises a value error (via helpers) for invalid or duplicate percentiles, for a frame without columns, or for invalid include/exclude combinations.
    """Describe series or dataframe.

    Called from pandas.core.generic.NDFrame.describe()

    Parameters
    ----------
    obj: DataFrame or Series
        Either dataframe or series to be described.
    include : 'all', list-like of dtypes or None (default), optional
        A white list of data types to include in the result. Ignored for ``Series``.
    exclude : list-like of dtypes or None (default), optional,
        A black list of data types to omit from the result. Ignored for ``Series``.
    percentiles : list-like of numbers, optional
        The percentiles to include in the output. All should fall between 0 and 1.
        The default is ``[.25, .5, .75]``, which returns the 25th, 50th, and
        75th percentiles.

    Returns
    -------
    Dataframe or series description.
    """
    percentiles = _refine_percentiles(percentiles)

    describer: NDFrameDescriberAbstract
    if obj.ndim == 1:
        describer = SeriesDescriber(
            obj=cast("Series", obj),
        )
    else:
        describer = DataFrameDescriber(
            obj=cast("DataFrame", obj),
            include=include,
            exclude=exclude,
        )

    result = describer.describe(percentiles=percentiles)
    return cast(NDFrameT, result)


class NDFrameDescriberAbstract(ABC):
    # [CLASS DOC]
    # Abstract base for objects that describe a Series or DataFrame.
    # State: the attribute obj holds the object being described. Subclasses must implement the describe method, which takes normalized percentiles and returns the summary object.
    """Abstract class for describing dataframe or series.

    Parameters
    ----------
    obj : Series or DataFrame
        Object to be described.
    """

    def __init__(self, obj: DataFrame | Series) -> None:
        # [FUNCTION DOC]
        # Stores the object to describe in the attribute obj. No validation.
        self.obj = obj

    @abstractmethod
    def describe(self, percentiles: Sequence[float] | np.ndarray) -> DataFrame | Series:
        # [FUNCTION DOC]
        # Abstract method: subclasses return the description (a Series or DataFrame) of the stored object for the given percentiles (list-like of numbers in the unit interval). The base implementation has no behavior.
        """Do describe either series or dataframe.

        Parameters
        ----------
        percentiles : list-like of numbers
            The percentiles to include in the output.
        """


class SeriesDescriber(NDFrameDescriberAbstract):
    # [CLASS DOC]
    # Describer for a Series; inherits the abstract describer and uses the inherited initializer unchanged. Delegates to the per-dtype function chosen for the Series dtype.
    """Class responsible for creating series description."""

    obj: Series

    def describe(self, percentiles: Sequence[float] | np.ndarray) -> Series:
        # [FUNCTION DOC]
        # Describe the stored Series: select the 1D describing function matching its dtype and return its result, a Series of summary statistics indexed by statistic name. The percentiles are passed through.
        describe_func = select_describe_func(
            self.obj,
        )
        return describe_func(self.obj, percentiles)


class DataFrameDescriber(NDFrameDescriberAbstract):
    # [CLASS DOC]
    # Describer for a DataFrame; inherits the abstract describer.
    # State: obj (the frame), include and exclude (dtype selections, possibly None; include may be the string "all"). Describes each selected column separately and concatenates the results.
    """Class responsible for creating dataobj description.

    Parameters
    ----------
    obj : DataFrame
        DataFrame to be described.
    include : 'all', list-like of dtypes or None
        A white list of data types to include in the result.
    exclude : list-like of dtypes or None
        A black list of data types to omit from the result.
    """

    obj: DataFrame

    def __init__(
        self,
        obj: DataFrame,
        *,
        include: str | Sequence[str] | None,
        exclude: str | Sequence[str] | None,
    ) -> None:
        # [FUNCTION DOC]
        # Store include and exclude and the frame. Keyword-only include and exclude are dtype selections or None.
        # Raises a value error if the frame has no columns (checked before delegating to the base initializer).
        self.include = include
        self.exclude = exclude

        if obj.ndim == 2 and obj.columns.size == 0:
            raise ValueError("Cannot describe a DataFrame without columns")

        super().__init__(obj)

    def describe(self, percentiles: Sequence[float] | np.ndarray) -> DataFrame:
        # [FUNCTION DOC]
        # Describe the frame: select the columns via the selection helper, describe each column with the function matching its dtype, then concatenate the per-column results side by side (reindexed to a common, display-friendly row order from the ordering helper, without sorting). The result columns are set to a copy of the selected columns.
        # Returns a DataFrame with one column per described column. Percentiles are passed through to each 1D function.
        data = self._select_data()

        ldesc: list[Series] = []
        for _, series in data.items():
            describe_func = select_describe_func(series)
            ldesc.append(describe_func(series, percentiles))

        col_names = reorder_columns(ldesc)
        d = concat(
            [x.reindex(col_names) for x in ldesc],
            axis=1,
            ignore_index=True,
            sort=False,
        )
        d.columns = data.columns.copy()
        return d

    def _select_data(self) -> DataFrame:
        # [FUNCTION DOC]
        # HELPER: for describing a frame. Chooses the columns to describe.
        # With neither include nor exclude given: keeps only numeric and datetime columns, falling back to all columns if none match. With include equal to "all": uses all columns, and raises a value error if exclude is also given. Otherwise: selects columns by dtype using both include and exclude, and raises a value error if no columns match.
        # Returns the selected sub-frame.
        """Select columns to be described."""
        if (self.include is None) and (self.exclude is None):
            # when some numerics are found, keep only numerics
            default_include: list[npt.DTypeLike] = [np.number, "datetime"]
            data = self.obj.select_dtypes(include=default_include)
            if len(data.columns) == 0:
                data = self.obj
        elif self.include == "all":
            if self.exclude is not None:
                msg = "exclude must be None when include is 'all'"
                raise ValueError(msg)
            data = self.obj
        else:
            data = self.obj.select_dtypes(
                include=self.include,
                exclude=self.exclude,
            )
            if len(data.columns) == 0:
                msg = "No columns match the specified include or exclude data types"
                raise ValueError(msg)
        return data


def reorder_columns(ldesc: Sequence[Series]) -> list[Hashable]:
    # [FUNCTION DOC]
    # Compute a display-friendly row order for combining per-column descriptions.
    # Parameter: ldesc is a sequence of Series (each indexed by statistic names). The series indexes are visited from shortest to longest (stable for ties) and each statistic name is appended the first time it is seen.
    # Returns the ordered list of unique names.
    """Set a convenient order for rows for display."""
    names: list[Hashable] = []
    seen_names: set[Hashable] = set()
    ldesc_indexes = sorted((x.index for x in ldesc), key=len)
    for idxnames in ldesc_indexes:
        for name in idxnames:
            if name not in seen_names:
                seen_names.add(name)
                names.append(name)
    return names


def describe_numeric_1d(series: Series, percentiles: Sequence[float]) -> Series:
    # [FUNCTION DOC]
    # Describe a numeric (or timedelta) Series.
    # Parameters: series is the data; percentiles is a list-like of numbers in the unit interval, possibly empty.
    # Returns a Series named after the input, indexed by count, mean, std, min, the formatted percentile labels, and max, in that order; quantiles are skipped when no percentiles are given. The result dtype is float64 for numeric non-complex data (nullable Float64 for masked extension dtypes, Arrow float64 for Arrow-backed numeric dtypes), and is inferred (object-like) for complex data, timedeltas and Arrow timedeltas.
    """Describe series containing numerical data.

    Parameters
    ----------
    series : Series
        Series to be described.
    percentiles : list-like of numbers
        The percentiles to include in the output.
    """
    from pandas import Series

    formatted_percentiles = format_percentiles(percentiles)

    if len(percentiles) == 0:
        quantiles = []
    else:
        quantiles = series.quantile(percentiles).tolist()

    stat_index = ["count", "mean", "std", "min", *formatted_percentiles, "max"]
    d = [
        series.count(),
        series.mean(),
        series.std(),
        series.min(),
        *quantiles,
        series.max(),
    ]
    # GH#48340 - always return float on non-complex numeric data
    dtype: DtypeObj | None
    if isinstance(series.dtype, ExtensionDtype):
        if isinstance(series.dtype, ArrowDtype):
            if series.dtype.kind == "m":
                # GH53001: describe timedeltas with object dtype
                dtype = None
            else:
                import pyarrow as pa

                dtype = ArrowDtype(pa.float64())
        else:
            dtype = Float64Dtype()
    elif series.dtype.kind in "iufb":
        # i.e. numeric but exclude complex dtype
        dtype = np.dtype("float")
    else:
        dtype = None
    return Series(d, index=stat_index, name=series.name, dtype=dtype)


def describe_categorical_1d(
    data: Series,
    percentiles_ignored: Sequence[float],
) -> Series:
    # [FUNCTION DOC]
    # Describe categorical-like (also object, string and boolean) data. The percentiles_ignored argument is unused and exists only to unify the interface.
    # Returns a Series named after the input, indexed by count, unique, top and freq: non-null count, number of distinct values with nonzero frequency, the most frequent value and its frequency. When there are no values, top and freq are missing values and the result dtype is object.
    """Describe series containing categorical data.

    Parameters
    ----------
    data : Series
        Series to be described.
    percentiles_ignored : list-like of numbers
        Ignored, but in place to unify interface.
    """
    names = ["count", "unique", "top", "freq"]
    objcounts = data.value_counts()
    count_unique = len(objcounts[objcounts != 0])
    if count_unique > 0:
        top, freq = objcounts.index[0], objcounts.iloc[0]
        dtype = None
    else:
        # If the DataFrame is empty, set 'top' and 'freq' to None
        # to maintain output shape consistency
        top, freq = np.nan, np.nan
        dtype = "object"

    result = [data.count(), count_unique, top, freq]

    from pandas import Series

    return Series(result, index=names, name=data.name, dtype=dtype)


def describe_timestamp_1d(data: Series, percentiles: Sequence[float]) -> Series:
    # [FUNCTION DOC]
    # Describe datetime-like data (naive or timezone-aware datetimes).
    # Parameters: data is the Series; percentiles is a list-like of numbers in the unit interval.
    # Returns a Series named after the input, indexed by count, mean, min, the formatted percentile labels, and max (no standard deviation), with inferred dtype.
    """Describe series containing datetime64 dtype.

    Parameters
    ----------
    data : Series
        Series to be described.
    percentiles : list-like of numbers
        The percentiles to include in the output.
    """
    # GH-30164
    from pandas import Series

    formatted_percentiles = format_percentiles(percentiles)

    stat_index = ["count", "mean", "min", *formatted_percentiles, "max"]
    d = [
        data.count(),
        data.mean(),
        data.min(),
        *data.quantile(percentiles).tolist(),
        data.max(),
    ]
    return Series(d, index=stat_index, name=data.name)


def select_describe_func(
    data: Series,
) -> Callable:
    # [FUNCTION DOC]
    # Choose the 1D describing function for a Series based on its dtype.
    # Boolean dtype: categorical describer. Numeric dtype: numeric describer. Datetime64 or timezone-aware datetime: timestamp describer. Timedelta: numeric describer. Anything else: categorical describer.
    # Returns the selected callable; the order of checks matters because boolean is checked before numeric.
    """Select proper function for describing series based on data type.

    Parameters
    ----------
    data : Series
        Series to be described.
    """
    if is_bool_dtype(data.dtype):
        return describe_categorical_1d
    elif is_numeric_dtype(data):
        return describe_numeric_1d
    elif data.dtype.kind == "M" or isinstance(data.dtype, DatetimeTZDtype):
        return describe_timestamp_1d
    elif data.dtype.kind == "m":
        return describe_numeric_1d
    else:
        return describe_categorical_1d


def _refine_percentiles(
    percentiles: Sequence[float] | np.ndarray | None,
) -> npt.NDArray[np.float64]:
    # [FUNCTION DOC]
    # HELPER: for the top-level describe function. Normalizes percentiles.
    # Parameter: percentiles is a list-like of numbers or None. None yields the default of quarter, half and three quarters.
    # Otherwise converts to an array, validates every value lies in the unit interval (raising a value error if not), sorts and deduplicates, and raises a value error if duplicates were present.
    # Returns a sorted float array.
    # KNOWN ISSUE: the assertion that percentiles is not None is redundant (the None case has already returned), so it can never fail.
    """
    Ensure that percentiles are unique and sorted.

    Parameters
    ----------
    percentiles : list-like of numbers, optional
        The percentiles to include in the output.
    """
    if percentiles is None:
        return np.array([0.25, 0.5, 0.75])

    percentiles = np.asarray(percentiles)

    # get them all to be in [0, 1]
    validate_percentile(percentiles)

    # sort and check for duplicates
    unique_pcts = np.unique(percentiles)
    assert percentiles is not None
    if len(unique_pcts) < len(percentiles):
        raise ValueError("percentiles cannot contain duplicates")

    return unique_pcts
