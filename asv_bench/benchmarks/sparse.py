# [MODULE DOC]
# Benchmarks for sparse data structures: sparse series to frame, sparse array
# construction, frame construction from a scientific-library sparse matrix,
# conversion to and from coordinate-format matrices, arithmetic on sparse arrays
# and their index objects, minimum/maximum, take, item access and mask indexing.
# Public contents are the helper function make_array and the benchmark classes
# SparseSeriesToFrame, SparseArrayConstructor, SparseDataFrameConstructor,
# FromCoo, ToCoo, ToCooFrame, Arithmetic, ArithmeticBlock, MinMax, Take, GetItem
# and GetItemMask. Requires the optional scipy sparse package at import time. The
# final line re-exports the common asv setup hook.
import numpy as np
import scipy.sparse

import pandas as pd
from pandas import (
    MultiIndex,
    Series,
    date_range,
)
from pandas.arrays import SparseArray


def make_array(size, dense_proportion, fill_value, dtype):
    # [FUNCTION DOC]
    # HELPER: data factory shared by the sparse benchmark classes in this module.
    # Inputs: size, the array length; dense_proportion, the fraction of positions
    # that hold non-fill values; fill_value, the value in all other positions;
    # dtype, the numpy dtype of the result. Returns an array of the given size and
    # dtype filled with fill_value, where a random sample (without replacement) of
    # size times dense_proportion positions (rounded down) is overwritten with random
    # integers from 0 to 99 drawn with repeats, cast to the dtype. Uses the global
    # random state.
    dense_size = int(size * dense_proportion)
    arr = np.full(size, fill_value, dtype)
    indexer = np.random.choice(np.arange(size), dense_size, replace=False)
    arr[indexer] = np.random.choice(np.arange(100, dtype=dtype), dense_size)
    return arr


class SparseSeriesToFrame:
    # [CLASS DOC]
    # Benchmark of building a frame from a dictionary of 49 sparse series with
    # staggered indexes. No parameters. State: series (mapping of position to series).
    def setup(self):
        # [FUNCTION DOC]
        # Builds 49 sparse series (positions 1 to 49), each from 50001 minute
        # timestamps from 2000-01-01 with the last i entries removed; the values are
        # random normal with everything after the first 100 set to missing, stored as
        # sparse arrays.
        K = 50
        N = 50001
        rng = date_range("1/1/2000", periods=N, freq="min")
        self.series = {}
        for i in range(1, K):
            data = np.random.randn(N)[:-i]
            idx = rng[:-i]
            data[100:] = np.nan
            self.series[i] = Series(SparseArray(data), index=idx)

    def time_series_to_frame(self):
        # [FUNCTION DOC]
        # Times building a frame from the mapping of series (aligns the indexes).
        pd.DataFrame(self.series)


class SparseArrayConstructor:
    # [CLASS DOC]
    # Benchmark of building a sparse array from a dense array for different
    # densities, fill values and dtypes. Parametrized by dense proportion, fill
    # value (zero or NaN) and dtype (int64, float64, object). State: array.
    params = ([0.1, 0.01], [0, np.nan], [np.int64, np.float64, object])
    param_names = ["dense_proportion", "fill_value", "dtype"]

    def setup(self, dense_proportion, fill_value, dtype):
        # [FUNCTION DOC]
        # Builds a million-element dense array with the shared factory.
        N = 10**6
        self.array = make_array(N, dense_proportion, fill_value, dtype)

    def time_sparse_array(self, dense_proportion, fill_value, dtype):
        # [FUNCTION DOC]
        # Times building the sparse array with the fill value and dtype.
        SparseArray(self.array, fill_value=fill_value, dtype=dtype)


class SparseDataFrameConstructor:
    # [CLASS DOC]
    # Benchmark of building a sparse-dtype frame from a scientific-library sparse
    # matrix. No parameters. State: sparse.
    def setup(self):
        # [FUNCTION DOC]
        # Creates a random sparse matrix of 1000 by 1000 with density 0.005 as
        # sparse.
        N = 1000
        self.sparse = scipy.sparse.rand(N, N, 0.005)

    def time_from_scipy(self):
        # [FUNCTION DOC]
        # Times building a frame from the sparse matrix with the sparse accessor.
        pd.DataFrame.sparse.from_spmatrix(self.sparse)


class FromCoo:
    # [CLASS DOC]
    # Benchmark of building a sparse series from a coordinate-format matrix. No
    # parameters. State: matrix.
    def setup(self):
        # [FUNCTION DOC]
        # Creates a 100 by 100 coordinate matrix with three entries: 3.0 at row one,
        # column zero; 1.0 at row zero, column two; 2.0 at row zero, column three.
        self.matrix = scipy.sparse.coo_matrix(
            ([3.0, 1.0, 2.0], ([1, 0, 0], [0, 2, 3])), shape=(100, 100)
        )

    def time_sparse_series_from_coo(self):
        # [FUNCTION DOC]
        # Times building a sparse series from the matrix.
        Series.sparse.from_coo(self.matrix)


class ToCoo:
    # [CLASS DOC]
    # Benchmark of converting a sparse series with a multi-level index to a
    # coordinate matrix, with or without sorting labels. Parametrized by
    # sort_labels. State: ss_mult_lvl (four levels) and ss_two_lvl (two levels).
    params = [True, False]
    param_names = ["sort_labels"]

    def setup(self, sort_labels):
        # [FUNCTION DOC]
        # Builds a series of 10000 missing values with three non-missing entries
        # (3.0 at position 0, minus 1.0 at 100, 12.1 at 999), indexes it by a
        # four-level product of ranges of ten and again by a two-level product of
        # ranges of 100, and converts each to sparse dtype. The sort_labels flag is
        # unused here.
        s = Series([np.nan] * 10000)
        s[0] = 3.0
        s[100] = -1.0
        s[999] = 12.1

        s_mult_lvl = s.set_axis(MultiIndex.from_product([range(10)] * 4))
        self.ss_mult_lvl = s_mult_lvl.astype("Sparse")

        s_two_lvl = s.set_axis(MultiIndex.from_product([range(100)] * 2))
        self.ss_two_lvl = s_two_lvl.astype("Sparse")

    def time_sparse_series_to_coo(self, sort_labels):
        # [FUNCTION DOC]
        # Times converting the four-level series to a matrix with levels zero and one
        # as rows and levels two and three as columns.
        self.ss_mult_lvl.sparse.to_coo(
            row_levels=[0, 1], column_levels=[2, 3], sort_labels=sort_labels
        )

    def time_sparse_series_to_coo_single_level(self, sort_labels):
        # [FUNCTION DOC]
        # Times converting the two-level series to a matrix using the default level
        # split.
        self.ss_two_lvl.sparse.to_coo(sort_labels=sort_labels)


class ToCooFrame:
    # [CLASS DOC]
    # Benchmark of converting a sparse-dtype frame to a coordinate matrix. No
    # parameters. State: df.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a 10000 by 10 zero float array with three non-zero cells, and a
        # frame from it with sparse float dtype and fill value zero.
        N = 10000
        k = 10
        arr = np.zeros((N, k), dtype=float)
        arr[0, 0] = 3.0
        arr[12, 7] = -1.0
        arr[0, 9] = 11.2
        self.df = pd.DataFrame(arr, dtype=pd.SparseDtype("float", fill_value=0.0))

    def time_to_coo(self):
        # [FUNCTION DOC]
        # Times converting the frame to a coordinate matrix with the sparse accessor.
        self.df.sparse.to_coo()


class Arithmetic:
    # [CLASS DOC]
    # Benchmark of index union/intersection and element-wise arithmetic between two
    # sparse integer arrays of a million elements. Parametrized by dense proportion
    # and fill value. State: array1 and array2.
    params = ([0.1, 0.01], [0, np.nan])
    param_names = ["dense_proportion", "fill_value"]

    def setup(self, dense_proportion, fill_value):
        # [FUNCTION DOC]
        # Builds two independent dense arrays with the shared factory and converts
        # each to a sparse array with the fill value.
        N = 10**6
        arr1 = make_array(N, dense_proportion, fill_value, np.int64)
        self.array1 = SparseArray(arr1, fill_value=fill_value)
        arr2 = make_array(N, dense_proportion, fill_value, np.int64)
        self.array2 = SparseArray(arr2, fill_value=fill_value)

    def time_make_union(self, dense_proportion, fill_value):
        # [FUNCTION DOC]
        # Times the union of the sparse index structures of the two arrays.
        self.array1.sp_index.make_union(self.array2.sp_index)

    def time_intersect(self, dense_proportion, fill_value):
        # [FUNCTION DOC]
        # Times the intersection of the sparse index structures of the two arrays.
        self.array1.sp_index.intersect(self.array2.sp_index)

    def time_add(self, dense_proportion, fill_value):
        # [FUNCTION DOC]
        # Times adding the two arrays.
        self.array1 + self.array2

    def time_divide(self, dense_proportion, fill_value):
        # [FUNCTION DOC]
        # Times dividing the first array by the second.
        self.array1 / self.array2


class ArithmeticBlock:
    # [CLASS DOC]
    # Benchmark of index operations and arithmetic on sparse arrays made of 1000
    # blocks of 10 consecutive non-fill values. Parametrized by fill value (NaN or
    # zero). State: arr1 and arr2.
    params = [np.nan, 0]
    param_names = ["fill_value"]

    def setup(self, fill_value):
        # [FUNCTION DOC]
        # Builds two independent block-structured sparse arrays of a million
        # elements with the helper method.
        N = 10**6
        self.arr1 = self.make_block_array(
            length=N, num_blocks=1000, block_size=10, fill_value=fill_value
        )
        self.arr2 = self.make_block_array(
            length=N, num_blocks=1000, block_size=10, fill_value=fill_value
        )

    def make_block_array(self, length, num_blocks, block_size, fill_value):
        # [FUNCTION DOC]
        # HELPER: array factory for the setup method of this class. Inputs: length of
        # the array; num_blocks, the number of blocks; block_size, the length of each
        # block; fill_value, the value outside the blocks. Starts from an array
        # filled with the fill value, chooses num_blocks distinct block start
        # positions among multiples of the block size, writes random integers from 0
        # to 99 in each block, and returns the result converted to a sparse array with
        # the fill value. Uses the global random state.
        arr = np.full(length, fill_value)
        indices = np.random.choice(
            np.arange(0, length, block_size), num_blocks, replace=False
        )
        for ind in indices:
            arr[ind : ind + block_size] = np.random.randint(0, 100, block_size)
        return SparseArray(arr, fill_value=fill_value)

    def time_make_union(self, fill_value):
        # [FUNCTION DOC]
        # Times the union of the sparse index structures of the two arrays.
        self.arr1.sp_index.make_union(self.arr2.sp_index)

    def time_intersect(self, fill_value):
        # [FUNCTION DOC]
        # Times the intersection of a sparse index structure with another.
        # KNOWN ISSUE: both operands are the second array, so it intersects the
        # second array's index with itself rather than the two different arrays, which
        # differs from the neighbouring union benchmark.
        self.arr2.sp_index.intersect(self.arr2.sp_index)

    def time_addition(self, fill_value):
        # [FUNCTION DOC]
        # Times adding the two arrays.
        self.arr1 + self.arr2

    def time_division(self, fill_value):
        # [FUNCTION DOC]
        # Times dividing the first array by the second.
        self.arr1 / self.arr2


class MinMax:
    # [CLASS DOC]
    # Benchmark of the minimum and maximum of a very sparse array. Parametrized by
    # function name and fill value (0.0 or NaN). State: sp_arr.
    params = (["min", "max"], [0.0, np.nan])
    param_names = ["func", "fill_value"]

    def setup(self, func, fill_value):
        # [FUNCTION DOC]
        # Builds a million-element float array with a density of one in a hundred
        # thousand and converts it to a sparse array with the fill value. The func
        # parameter is unused here.
        N = 1_000_000
        arr = make_array(N, 1e-5, fill_value, np.float64)
        self.sp_arr = SparseArray(arr, fill_value=fill_value)

    def time_min_max(self, func, fill_value):
        # [FUNCTION DOC]
        # Times calling the named reduction on the sparse array.
        getattr(self.sp_arr, func)()


class Take:
    # [CLASS DOC]
    # Benchmark of position-based selection from a very sparse array for different
    # index arrays and allow-fill settings. Parametrized by the index array (one
    # element zero; the first 100000 positions; 100000 copies of minus one) and the
    # allow_fill flag. State: sp_arr.
    params = ([np.array([0]), np.arange(100_000), np.full(100_000, -1)], [True, False])
    param_names = ["indices", "allow_fill"]

    def setup(self, indices, allow_fill):
        # [FUNCTION DOC]
        # Builds a million-element float array with density one in a hundred
        # thousand and fill value zero, converted to a sparse array as sp_arr. The
        # parameters are unused here.
        N = 1_000_000
        fill_value = 0.0
        arr = make_array(N, 1e-5, fill_value, np.float64)
        self.sp_arr = SparseArray(arr, fill_value=fill_value)

    def time_take(self, indices, allow_fill):
        # [FUNCTION DOC]
        # Times selecting the given positions with the fill-handling flag (a minus one
        # position is interpreted as missing when filling is allowed).
        self.sp_arr.take(indices, allow_fill=allow_fill)


class GetItem:
    # [CLASS DOC]
    # Benchmark of integer and slice item access on a very sparse array with NaN as
    # the fill value. No parameters. State: sp_arr.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a million-element float array with density one in a hundred
        # thousand and NaN as fill value, converted to a sparse array as sp_arr.
        N = 1_000_000
        d = 1e-5
        arr = make_array(N, d, np.nan, np.float64)
        self.sp_arr = SparseArray(arr)

    def time_integer_indexing(self):
        # [FUNCTION DOC]
        # Times access to the element at position 78.
        self.sp_arr[78]

    def time_slice(self):
        # [FUNCTION DOC]
        # Times selecting everything but the first element with a slice.
        self.sp_arr[1:]


class GetItemMask:
    # [CLASS DOC]
    # Benchmark of selecting from a sparse array using a sparse boolean mask, for
    # different mask fill values. Parametrized by the mask fill value (true, false,
    # NaN). State: sp_arr and sp_b_arr.
    params = [True, False, np.nan]
    param_names = ["fill_value"]

    def setup(self, fill_value):
        # [FUNCTION DOC]
        # Builds a million-element float sparse array with NaN fill and density one
        # in a hundred thousand as sp_arr. Builds a boolean array filled with the
        # fill value (for NaN, numpy casts it to true), sets about one in a hundred
        # thousand random positions (distinct, from 32-bit draws) to the opposite
        # value (true if the fill value is missing, otherwise its negation), and
        # converts it to a sparse boolean array with that fill value as sp_b_arr.
        N = 1_000_000
        d = 1e-5
        arr = make_array(N, d, np.nan, np.float64)
        self.sp_arr = SparseArray(arr)
        b_arr = np.full(shape=N, fill_value=fill_value, dtype=np.bool_)
        fv_inds = np.unique(
            np.random.randint(low=0, high=N - 1, size=int(N * d), dtype=np.int32)
        )
        b_arr[fv_inds] = True if pd.isna(fill_value) else not fill_value
        self.sp_b_arr = SparseArray(b_arr, dtype=np.bool_, fill_value=fill_value)

    def time_mask(self, fill_value):
        # [FUNCTION DOC]
        # Times selecting the elements of the sparse array where the sparse mask is
        # true.
        self.sp_arr[self.sp_b_arr]


from .pandas_vb_common import setup  # noqa: F401 isort:skip
