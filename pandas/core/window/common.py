"""Common utility functions for rolling operations"""
# [MODULE DOC]
# HELPER module: shared utilities for the window (rolling, expanding, exponentially
# weighted) implementations, mainly for binary moments such as covariance and
# correlation.
# Contents: flex_binary_moment (apply a pairwise binary kernel across two Series /
# DataFrames, optionally pairwise over columns), zsqrt (square root clipped to zero
# for negative inputs) and prep_binary (align two objects and mask them to a common
# set of missing values).
# Relationships: used by the rolling and ewm modules; builds frames with the public
# DataFrame, concat and MultiIndex tools (imported lazily inside functions).
# No import-time side effects.

from __future__ import annotations

from collections import defaultdict
from typing import cast

import numpy as np

from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)

from pandas.core.indexes.api import MultiIndex


def flex_binary_moment(arg1, arg2, f, pairwise: bool = False):
    # [FUNCTION DOC]
    # HELPER: window binary moments (covariance, correlation). Applies the binary
    # kernel f to Series/DataFrame operands and assembles the result.
    # Inputs: arg1 and arg2, each a Series or DataFrame; f, a callable taking two
    # aligned Series and returning a Series; pairwise (default false), whether to
    # compute f for every pair of columns between the two frames.
    # Cases: two Series are masked to a common index and passed to f directly. If arg1
    # is a DataFrame and arg2 a DataFrame: with pairwise false, identical frames are
    # processed column by column (supporting duplicate column names) and different
    # frames must each have unique columns (otherwise a value error is raised), are
    # outer-aligned and combined over the union of columns, keeping only columns
    # present in both; with pairwise true, every column pair is computed (reusing the
    # transposed result when both frames are the same object), the pieces are
    # concatenated and indexed by a (row label, arg2 column) multi-level index, with
    # column names taken from arg1 and index names from the row index plus arg2
    # column names; an empty union of row labels gives an empty float frame.
    # If arg1 is a DataFrame and arg2 is not, each column of arg1 is combined with arg2
    # and the columns of arg1 are kept. If arg1 is neither (a Series with a
    # DataFrame), the arguments are swapped and the call repeated without pairwise.
    # Returns the Series or DataFrame result.
    # KNOWN ISSUE: with two DataFrames, a pairwise value that is neither true nor
    # false (for example None) matches neither branch, so the function falls through
    # and returns None.
    if isinstance(arg1, ABCSeries) and isinstance(arg2, ABCSeries):
        X, Y = prep_binary(arg1, arg2)
        return f(X, Y)

    elif isinstance(arg1, ABCDataFrame):
        from pandas import DataFrame

        def dataframe_from_int_dict(data, frame_template) -> DataFrame:
            # [FUNCTION DOC]
            # HELPER: flex_binary_moment. Builds a frame from a mapping of integer
            # column positions to results, indexed like frame_template.
            # Columns are relabeled by looking up the integer positions in the
            # template's columns; if the result has no columns, the template's columns
            # are copied instead. Returns the new frame.
            result = DataFrame(data, index=frame_template.index)
            if len(result.columns) > 0:
                result.columns = frame_template.columns[result.columns]
            else:
                result.columns = frame_template.columns.copy()
            return result

        results = {}
        if isinstance(arg2, ABCDataFrame):
            if pairwise is False:
                if arg1 is arg2:
                    # special case in order to handle duplicate column names
                    for i in range(len(arg1.columns)):
                        results[i] = f(arg1.iloc[:, i], arg2.iloc[:, i])
                    return dataframe_from_int_dict(results, arg1)
                else:
                    if not arg1.columns.is_unique:
                        raise ValueError("'arg1' columns are not unique")
                    if not arg2.columns.is_unique:
                        raise ValueError("'arg2' columns are not unique")
                    X, Y = arg1.align(arg2, join="outer")
                    X, Y = prep_binary(X, Y)
                    res_columns = arg1.columns.union(arg2.columns)
                    for col in res_columns:
                        if col in X and col in Y:
                            results[col] = f(X[col], Y[col])
                    return DataFrame(results, index=X.index, columns=res_columns)
            elif pairwise is True:
                results = defaultdict(dict)
                for i in range(len(arg1.columns)):
                    for j in range(len(arg2.columns)):
                        if j < i and arg2 is arg1:
                            # Symmetric case
                            results[i][j] = results[j][i]
                        else:
                            results[i][j] = f(
                                *prep_binary(arg1.iloc[:, i], arg2.iloc[:, j])
                            )

                from pandas import concat

                result_index = arg1.index.union(arg2.index)
                if len(result_index):
                    # construct result frame
                    result = concat(
                        [
                            concat(
                                [results[i][j] for j in range(len(arg2.columns))],
                                ignore_index=True,
                            )
                            for i in range(len(arg1.columns))
                        ],
                        ignore_index=True,
                        axis=1,
                    )
                    result.columns = arg1.columns

                    # set the index and reorder
                    if arg2.columns.nlevels > 1:
                        # mypy needs to know columns is a MultiIndex, Index doesn't
                        # have levels attribute
                        arg2.columns = cast(MultiIndex, arg2.columns)
                        # GH 21157: Equivalent to MultiIndex.from_product(
                        #  [result_index], <unique combinations of arg2.columns.levels>,
                        # )
                        # A normal MultiIndex.from_product will produce too many
                        # combinations.
                        result_level = np.tile(
                            result_index, len(result) // len(result_index)
                        )
                        arg2_levels = (
                            np.repeat(
                                arg2.columns.get_level_values(i),
                                len(result) // len(arg2.columns),
                            )
                            for i in range(arg2.columns.nlevels)
                        )
                        result_names = [*arg2.columns.names, result_index.name]
                        result.index = MultiIndex.from_arrays(
                            [*arg2_levels, result_level], names=result_names
                        )
                        # GH 34440
                        num_levels = len(
                            result.index.levels  # pyright: ignore[reportAttributeAccessIssue]
                        )
                        new_order = [num_levels - 1, *range(num_levels - 1)]
                        result = result.reorder_levels(new_order).sort_index()
                    else:
                        result.index = MultiIndex.from_product(
                            [range(len(arg2.columns)), range(len(result_index))]
                        )
                        result = result.swaplevel(1, 0).sort_index()
                        result.index = MultiIndex.from_product(
                            [result_index, arg2.columns]
                        )
                else:
                    # empty result
                    result = DataFrame(
                        index=MultiIndex(
                            levels=[arg1.index, arg2.columns], codes=[[], []]
                        ),
                        columns=arg2.columns,
                        dtype="float64",
                    )

                # reset our index names to arg1 names
                # reset our column names to arg2 names
                # careful not to mutate the original names
                result.columns = result.columns.set_names(arg1.columns.names)
                result.index = result.index.set_names(
                    result_index.names + arg2.columns.names
                )

                return result
        else:
            results = {
                i: f(*prep_binary(arg1.iloc[:, i], arg2))
                for i in range(len(arg1.columns))
            }
            return dataframe_from_int_dict(results, arg1)

    else:
        return flex_binary_moment(arg2, arg1, f)


def zsqrt(x):
    # [FUNCTION DOC]
    # HELPER: window variance/standard-deviation computations. Square root that maps
    # negative inputs to zero instead of not-a-number.
    # Input: x, a numpy array, Series or DataFrame. Floating-point warnings are
    # suppressed during the computation. Returns the element-wise square root with
    # entries where x was negative set to zero (missing values stay missing). The
    # result is a new object; x is not mutated.
    with np.errstate(all="ignore"):
        result = np.sqrt(x)
        mask = x < 0

    if isinstance(x, ABCDataFrame):
        if mask._values.any():
            result[mask] = 0
    elif mask.any():
        result[mask] = 0

    return result


def prep_binary(arg1, arg2):
    # [FUNCTION DOC]
    # HELPER: window binary moments. Align two objects to a common index and mask
    # each so that a value is missing wherever either input is missing.
    # Inputs: arg1 and arg2, Series or DataFrames (or scalars supporting arithmetic).
    # Each result is its input plus zero times the other, which broadcasts/aligns the
    # indexes and propagates missing values. Returns the pair of new objects.
    # mask out values, this also makes a common index...
    X = arg1 + 0 * arg2
    Y = arg2 + 0 * arg1

    return X, Y
