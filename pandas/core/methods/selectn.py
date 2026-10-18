"""
Implementation of nlargest and nsmallest.
"""

# [MODULE DOC]
# Implementation of the n-largest and n-smallest selection used by the Series and DataFrame methods of those names.
# Contents: a generic base class holding the shared arguments, a Series implementation (partial selection using a k-th smallest kernel on a transformed array), and a DataFrame implementation (successive per-column selection with tie handling, then a final stable sort).
# Imports the compiled algorithms library and index helpers. Outside type checking, the generic frame and series type names are replaced by a plain type variable to avoid circular imports; no other import-time side effects.

from __future__ import annotations

from collections.abc import (
    Hashable,
    Sequence,
)
from typing import (
    TYPE_CHECKING,
    Generic,
    Literal,
    cast,
    final,
)

import numpy as np

from pandas._libs import algos as libalgos

from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_complex_dtype,
    is_integer_dtype,
    is_list_like,
    is_numeric_dtype,
    needs_i8_conversion,
)
from pandas.core.dtypes.dtypes import BaseMaskedDtype

from pandas.core.indexes.api import default_index

if TYPE_CHECKING:
    from pandas._typing import (
        DtypeObj,
        IndexLabel,
        NDFrameT,
    )

    from pandas import (
        DataFrame,
        Index,
        Series,
    )
else:
    # Generic[...] requires a non-str, provide it with a plain TypeVar at
    # runtime to avoid circular imports
    from pandas._typing import T

    NDFrameT = T
    DataFrame = T
    Series = T


class SelectN(Generic[NDFrameT]):
    # [CLASS DOC]
    # Generic base class for n-largest/n-smallest selectors, parameterized by the pandas object type.
    # State: obj (the object to select from), n (number of rows to select), keep (one of first, last or all, which decides tie handling).
    # Subclasses implement the compute method; the two public methods delegate to it with the method name.
    def __init__(
        self, obj: NDFrameT, n: int, keep: Literal["first", "last", "all"]
    ) -> None:
        # [FUNCTION DOC]
        # Store obj, n and keep. Raises a value error if keep is not one of first, last or all (validation occurs after assignment).
        self.obj = obj
        self.n = n
        self.keep = keep

        if self.keep not in ("first", "last", "all"):
            raise ValueError('keep must be either "first", "last" or "all"')

    def compute(self, method: str) -> NDFrameT:
        # [FUNCTION DOC]
        # Abstract hook: subclasses compute the result for the given method name (largest or smallest variant). The base class raises a not-implemented error.
        raise NotImplementedError

    @final
    def nlargest(self) -> NDFrameT:
        # [FUNCTION DOC]
        # Final method: return the n largest items by delegating to the compute method with the largest-selection name.
        return self.compute("nlargest")

    @final
    def nsmallest(self) -> NDFrameT:
        # [FUNCTION DOC]
        # Final method: return the n smallest items by delegating to the compute method with the smallest-selection name.
        return self.compute("nsmallest")

    @final
    @staticmethod
    def is_valid_dtype_n_method(dtype: DtypeObj) -> bool:
        # [FUNCTION DOC]
        # HELPER: static final check used by the compute implementations.
        # Returns True if the dtype supports n-largest/n-smallest: any numeric dtype except complex, or any dtype needing integer-based conversion (datetime-like). Otherwise False. No side effects.
        """
        Helper function to determine if dtype is valid for
        nsmallest/nlargest methods
        """
        if is_numeric_dtype(dtype):
            return not is_complex_dtype(dtype)
        return needs_i8_conversion(dtype)


class SelectNSeries(SelectN[Series]):
    # [CLASS DOC]
    # Selector for a Series. Inherits the base selector (same state and constructor).
    # The compute method returns a Series holding the n largest or smallest values with their original index labels.
    # KNOWN ISSUE: the docstring lists only first and last for keep, but the constructor and compute logic also accept all, which keeps ties beyond n.
    """
    Implement n largest/smallest for Series

    Parameters
    ----------
    obj : Series
    n : int
    keep : {'first', 'last'}, default 'first'

    Returns
    -------
    nordered : Series
    """

    def compute(self, method: str) -> Series:
        # [FUNCTION DOC]
        # Compute the selection for a Series; method is the largest or smallest name.
        # Raises a type error if the dtype is not valid (complex, or non-numeric non-datetime). If n is zero or negative, returns an empty selection.
        # The index is reset to a default one to avoid cost from duplicate labels, and restored on the result by taking from the original index. If n covers the whole length, performs a stable sort in the appropriate direction and takes the head. Otherwise: drops missing values, converts data to a plain numeric array (integer view for datetime-like, raw data for masked, unsigned view for boolean), negates it for the largest case (adjusting integers by one so boundaries order correctly, and mapping booleans so false orders before true), reverses it when keep is last, finds the k-th smallest value on a contiguous copy, selects positions at or below it, orders them stably, and trims according to keep (all keeps all ties; missing values are appended afterwards only if needed to reach n). For keep last, positions are mapped back to the original orientation. Result is the chosen rows concatenated with the missing-value rows, truncated to the computed length.
        # The local variable named default_index shadows the imported index helper within this method (harmless, since the helper is not used here).
        from pandas.core.reshape.concat import concat

        n = self.n
        dtype = self.obj.dtype
        if not self.is_valid_dtype_n_method(dtype):
            raise TypeError(f"Cannot use method '{method}' with dtype {dtype}")

        if n <= 0:
            return self.obj[[]]

        # Save index and reset to default index to avoid performance impact
        # from when index contains duplicates
        original_index: Index = self.obj.index
        default_index = self.obj.reset_index(drop=True)

        # Slower method used when taking the full length of the series
        # In this case, it is equivalent to a sort.
        if n >= len(default_index):
            ascending = method == "nsmallest"
            result = default_index.sort_values(ascending=ascending, kind="stable").head(
                n
            )
            result.index = original_index.take(result.index)
            return result

        # Fast method used in the general case
        dropped = default_index.dropna()
        nan_index = default_index.drop(dropped.index)

        new_dtype = dropped.dtype

        # Similar to algorithms._ensure_data
        arr = dropped._values
        if needs_i8_conversion(arr.dtype):
            arr = arr.view("i8")
        elif isinstance(arr.dtype, BaseMaskedDtype):
            arr = arr._data
        else:
            arr = np.asarray(arr)
        if arr.dtype.kind == "b":
            arr = arr.view(np.uint8)

        if method == "nlargest":
            arr = -arr
            if is_integer_dtype(new_dtype):
                # GH 21426: ensure reverse ordering at boundaries
                arr -= 1

            elif is_bool_dtype(new_dtype):
                # GH 26154: ensure False is smaller than True
                arr = 1 - (-arr)

        if self.keep == "last":
            arr = arr[::-1]

        nbase = n
        narr = len(arr)
        n = min(n, narr)

        # arr passed into kth_smallest must be contiguous. We copy
        # here because kth_smallest will modify its input
        # avoid OOB access with kth_smallest_c when n <= 0
        if len(arr) > 0:
            kth_val = libalgos.kth_smallest(arr.copy(order="C"), n - 1)
        else:
            kth_val = np.nan
        (ns,) = np.nonzero(arr <= kth_val)
        inds = ns[arr[ns].argsort(kind="stable")]

        if self.keep != "all":
            inds = inds[:n]
            findex = nbase
        elif len(inds) < nbase <= len(nan_index) + len(inds):
            findex = len(nan_index) + len(inds)
        else:
            findex = len(inds)

        if self.keep == "last":
            # reverse indices
            inds = narr - 1 - inds

        result = concat([dropped.iloc[inds], nan_index]).iloc[:findex]
        result.index = original_index.take(result.index)
        return result


class SelectNFrame(SelectN[DataFrame]):
    # [CLASS DOC]
    # Selector for a DataFrame. Inherits the base selector and adds a list of column labels to rank by. The compute method returns a DataFrame with the n largest or smallest rows ordered by those columns.
    # KNOWN ISSUE: the docstring lists only first and last for keep, but all is also accepted.
    """
    Implement n largest/smallest for DataFrame

    Parameters
    ----------
    obj : DataFrame
    n : int
    keep : {'first', 'last'}, default 'first'
    columns : list or str

    Returns
    -------
    nordered : DataFrame
    """

    def __init__(
        self,
        obj: DataFrame,
        n: int,
        keep: Literal["first", "last", "all"],
        columns: IndexLabel,
    ) -> None:
        # [FUNCTION DOC]
        # Store arguments via the base constructor (so keep is validated), plus columns: a non-list-like value or a tuple is treated as a single column label; the result is stored as a list under the attribute columns.
        super().__init__(obj, n, keep)
        if not is_list_like(columns) or isinstance(columns, tuple):
            columns = [columns]

        columns = cast(Sequence[Hashable], columns)
        columns = list(columns)
        self.columns = columns

    def compute(self, method: str) -> DataFrame:
        # [FUNCTION DOC]
        # Compute the selection for a DataFrame; method is the largest or smallest name.
        # Raises a type error naming the first column whose dtype is invalid for the method. The frame index is reset (restored at the end) to cope with duplicates. For each column in order, the matching Series method is applied with the remaining count (keep all unless it is the last column). If it is the last column or the result is not larger than the remaining count, its labels are added to the collected row indexer and the loop ends. Otherwise rows strictly better than the border value are added as safe, and only rows tied at the border are examined on the next column with a reduced count. Finally takes the collected rows, restores the original labels, and, with multiple columns, stably sorts by all columns in the direction of the method. Returns the new frame.
        n = self.n
        frame = self.obj
        columns = self.columns

        for column in columns:
            dtype = frame[column].dtype
            if not self.is_valid_dtype_n_method(dtype):
                raise TypeError(
                    f"Column {column!r} has dtype {dtype}, "
                    f"cannot use method {method!r} with this dtype"
                )

        def get_indexer(current_indexer: Index, other_indexer: Index) -> Index:
            # [FUNCTION DOC]
            # HELPER: nested function for compute. Concatenates two index objects: for the smallest method, the current indexer followed by the other; otherwise the other followed by the current. Returns the combined index; no mutation.
            """
            Helper function to concat `current_indexer` and `other_indexer`
            depending on `method`
            """
            if method == "nsmallest":
                return current_indexer.append(other_indexer)
            else:
                return other_indexer.append(current_indexer)

        # Below we save and reset the index in case index contains duplicates
        original_index = frame.index
        cur_frame = frame = frame.reset_index(drop=True)
        cur_n = n
        indexer: Index = default_index(0)

        for i, column in enumerate(columns):
            # For each column we apply method to cur_frame[column].
            # If it's the last column or if we have the number of
            # results desired we are done.
            # Otherwise there are duplicates of the largest/smallest
            # value and we need to look at the rest of the columns
            # to determine which of the rows with the largest/smallest
            # value in the column to keep.
            series = cur_frame[column]
            is_last_column = len(columns) - 1 == i
            values = getattr(series, method)(
                cur_n, keep=self.keep if is_last_column else "all"
            )

            if is_last_column or len(values) <= cur_n:
                indexer = get_indexer(indexer, values.index)
                break

            # Now find all values which are equal to
            # the (nsmallest: largest)/(nlargest: smallest)
            # from our series.
            border_value = values == values[values.index[-1]]

            # Some of these values are among the top-n
            # some aren't.
            unsafe_values = values[border_value]

            # These values are definitely among the top-n
            safe_values = values[~border_value]
            indexer = get_indexer(indexer, safe_values.index)

            # Go on and separate the unsafe_values on the remaining
            # columns.
            cur_frame = cur_frame.loc[unsafe_values.index]
            cur_n = n - len(indexer)

        frame = frame.take(indexer)

        # Restore the index on frame
        frame.index = original_index.take(indexer)

        # If there is only one column, the frame is already sorted.
        if len(columns) == 1:
            return frame

        ascending = method == "nsmallest"

        return frame.sort_values(columns, ascending=ascending, kind="stable")
