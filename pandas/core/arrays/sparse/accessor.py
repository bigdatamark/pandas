"""Sparse accessor"""

# [MODULE DOC]
# Implements the "sparse" accessor namespaces attached to series and frames.
# Public contents: the series-level accessor class (properties delegated to the
# sparse array: stored-point count, density, fill value, stored values; plus
# coordinate-matrix import/export and densification) and the frame-level
# accessor class (build a frame from a scipy matrix, densify, export to a
# coordinate matrix, overall density). A shared private-ish base class stores
# the parent object and triggers validation.
# Relationships: uses the sparse array class and sparse dtype, the scipy
# conversion helpers in the sibling module (imported lazily), the generic
# accessor delegation utilities and common-type resolution. The accessors are
# registered on the series and frame classes elsewhere (in the frame/series
# modules), not in this file. scipy is only imported inside methods.
# No import-time side effects other than the delegation decorator adding
# properties to the series accessor class.
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pandas.compat._optional import import_optional_dependency

from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.dtypes import SparseDtype

from pandas.core.accessor import (
    PandasDelegate,
    delegate_names,
)
from pandas.core.arrays.sparse.array import SparseArray

if TYPE_CHECKING:
    from scipy.sparse import (
        coo_matrix,
        spmatrix,
    )

    from pandas import (
        DataFrame,
        Series,
    )


class BaseAccessor:
    # [CLASS DOC]
    # HELPER: common base of the two sparse accessors. State: the parent series
    # or frame (stored in the parent attribute) and a class-level error message
    # used when the parent is not sparse. Subclasses implement the validation
    # step; construction stores the parent and then validates it.
    _validation_msg = "Can only use the '.sparse' accessor with Sparse data."

    def __init__(self, data=None) -> None:
        # [FUNCTION DOC]
        # Stores the given series or frame as the parent, then runs the
        # subclass validation. Raises whatever the validation raises (an
        # attribute error for non-sparse data, so that attribute access on the
        # accessor property fails cleanly).
        self._parent = data
        self._validate(data)

    def _validate(self, data) -> None:
        # [FUNCTION DOC]
        # HELPER: abstract validation hook. Always raises a not-implemented
        # error in the base class; subclasses check that the data is sparse.
        raise NotImplementedError


@delegate_names(
    SparseArray, ["npoints", "density", "fill_value", "sp_values"], typ="property"
)
class SparseAccessor(BaseAccessor, PandasDelegate):
    # [CLASS DOC]
    # Series-level sparse accessor (the sparse namespace on a series).
    # Inherits the parent storage/validation protocol from the base accessor and
    # the delegation protocol from the pandas delegate class. The class
    # decorator creates read-only properties for stored-point count, density,
    # fill value and stored values, which read the same-named attributes from
    # the series' sparse array through the property delegate hook. Other
    # methods: import from a coordinate matrix (class method), export to a
    # coordinate matrix, and conversion to a dense series. Invariant: the
    # parent series has a sparse dtype (checked at construction).
    # KNOWN ISSUE: the docstring summary line reads "Accessor for SparseSparse
    # from other sparse matrix data types", which is garbled; it is a
    # series-level accessor for sparse-dtype data.
    """
    Accessor for SparseSparse from other sparse matrix data types.

    Parameters
    ----------
    data : Series or DataFrame
        The Series or DataFrame to which the SparseAccessor is attached.

    See Also
    --------
    Series.sparse.to_coo : Create a scipy.sparse.coo_matrix from a Series with
        MultiIndex.
    Series.sparse.from_coo : Create a Series with sparse values from a
        scipy.sparse.coo_matrix.

    Examples
    --------
    >>> ser = pd.Series([0, 0, 2, 2, 2], dtype="Sparse[int]")
    >>> ser.sparse.density
    0.6
    >>> ser.sparse.sp_values
    array([2, 2, 2])
    """

    def _validate(self, data) -> None:
        # [FUNCTION DOC]
        # HELPER: construction check. Raises an attribute error with the
        # shared message when the series dtype is not a sparse dtype; returns
        # nothing otherwise.
        if not isinstance(data.dtype, SparseDtype):
            raise AttributeError(self._validation_msg)

    def _delegate_property_get(self, name: str, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: property delegation hook used by the delegating decorator.
        # name is the property name; extra positional/keyword arguments are
        # ignored. Returns the same-named attribute of the sparse array held
        # by the parent series.
        return getattr(self._parent.array, name)

    def _delegate_method(self, name: str, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: method delegation hook. For the names of the matrix import
        # and export methods it forwards the arguments to those methods and
        # returns their result; any other name raises a bare value error.
        # KNOWN ISSUE: the class only delegates properties through the
        # decorator, so this hook appears to be unused in practice, and the
        # raised error carries no message.
        if name == "from_coo":
            return self.from_coo(*args, **kwargs)
        elif name == "to_coo":
            return self.to_coo(*args, **kwargs)
        else:
            raise ValueError

    @classmethod
    def from_coo(cls, A, dense_index: bool = False) -> Series:
        # [FUNCTION DOC]
        # Class method. Builds a sparse-valued series from a scipy coordinate
        # matrix A; dense_index (default false) chooses between an index of only
        # stored coordinates and an index with the full sorted set of
        # coordinates. Delegates to the conversion helper in the scipy module
        # (which raises a type error if A is not a coordinate matrix), then
        # re-wraps the resulting array and index in a fresh series without
        # copying. Returns a series with a two-level index.
        """
        Create a Series with sparse values from a scipy.sparse.coo_matrix.

        This method takes a ``scipy.sparse.coo_matrix`` (coordinate format) as input and
        returns a pandas ``Series`` where the non-zero elements are represented as
        sparse values. The index of the Series can either include only the coordinates
        of non-zero elements (default behavior) or the full sorted set of coordinates
        from the matrix if ``dense_index`` is set to `True`.

        Parameters
        ----------
        A : scipy.sparse.coo_matrix
            The sparse matrix in coordinate format from which the sparse Series
            will be created.
        dense_index : bool, default False
            If False (default), the index consists of only the
            coords of the non-null entries of the original coo_matrix.
            If True, the index consists of the full sorted
            (row, col) coordinates of the coo_matrix.

        Returns
        -------
        s : Series
            A Series with sparse values.

        See Also
        --------
        DataFrame.sparse.from_spmatrix : Create a new DataFrame from a scipy sparse
            matrix.
        scipy.sparse.coo_matrix : A sparse matrix in COOrdinate format.

        Examples
        --------
        >>> from scipy import sparse

        >>> A = sparse.coo_matrix(
        ...     ([3.0, 1.0, 2.0], ([1, 0, 0], [0, 2, 3])), shape=(3, 4)
        ... )
        >>> A
        <COOrdinate sparse matrix of dtype 'float64'
            with 3 stored elements and shape (3, 4)>

        >>> A.todense()
        matrix([[0., 0., 1., 2.],
        [3., 0., 0., 0.],
        [0., 0., 0., 0.]])

        >>> ss = pd.Series.sparse.from_coo(A)
        >>> ss
        0  2    1.0
           3    2.0
        1  0    3.0
        dtype: Sparse[float64, nan]
        """
        from pandas import Series
        from pandas.core.arrays.sparse.scipy_sparse import coo_to_sparse_series

        result = coo_to_sparse_series(A, dense_index=dense_index)
        result = Series(result.array, index=result.index, copy=False)

        return result

    def to_coo(
        self, row_levels=(0,), column_levels=(1,), sort_labels: bool = False
    ) -> tuple[coo_matrix, list, list]:
        # [FUNCTION DOC]
        # Exports the parent series (which must have a multi-index with at
        # least two levels and a unique index) to a scipy coordinate matrix.
        # row_levels and column_levels (defaults: first level for rows, second
        # for columns) are sequences of level names or numbers that together
        # partition all levels; sort_labels (default false) sorts the labels.
        # Returns a triple of the matrix, the list of row labels and the list of
        # column labels. Errors (value errors for bad index/levels) come from
        # the scipy helper module.
        """
        Create a scipy.sparse.coo_matrix from a Series with MultiIndex.

        Use row_levels and column_levels to determine the row and column
        coordinates respectively. row_levels and column_levels are the names
        (labels) or numbers of the levels. {row_levels, column_levels} must be
        a partition of the MultiIndex level names (or numbers).

        Parameters
        ----------
        row_levels : tuple/list
            MultiIndex levels to use for row coordinates, specified by name or index.
        column_levels : tuple/list
            MultiIndex levels to use for column coordinates, specified by name or index.
        sort_labels : bool, default False
            Sort the row and column labels before forming the sparse matrix.
            When `row_levels` and/or `column_levels` refer to a single level,
            set to `True` for a faster execution.

        Returns
        -------
        y : scipy.sparse.coo_matrix
            The sparse matrix in coordinate format.
        rows : list (row labels)
            Labels corresponding to the row coordinates.
        columns : list (column labels)
            Labels corresponding to the column coordinates.

        See Also
        --------
        Series.sparse.from_coo : Create a Series with sparse values from a
            scipy.sparse.coo_matrix.

        Examples
        --------
        >>> s = pd.Series([3.0, np.nan, 1.0, 3.0, np.nan, np.nan])
        >>> s.index = pd.MultiIndex.from_tuples(
        ...     [
        ...         (1, 2, "a", 0),
        ...         (1, 2, "a", 1),
        ...         (1, 1, "b", 0),
        ...         (1, 1, "b", 1),
        ...         (2, 1, "b", 0),
        ...         (2, 1, "b", 1),
        ...     ],
        ...     names=["A", "B", "C", "D"],
        ... )
        >>> s
        A  B  C  D
        1  2  a  0    3.0
                 1    NaN
           1  b  0    1.0
                 1    3.0
        2  1  b  0    NaN
                 1    NaN
        dtype: float64

        >>> ss = s.astype("Sparse")
        >>> ss
        A  B  C  D
        1  2  a  0    3.0
                 1    NaN
           1  b  0    1.0
                 1    3.0
        2  1  b  0    NaN
                 1    NaN
        dtype: Sparse[float64, nan]

        >>> A, rows, columns = ss.sparse.to_coo(
        ...     row_levels=["A", "B"], column_levels=["C", "D"], sort_labels=True
        ... )
        >>> A
        <COOrdinate sparse matrix of dtype 'float64'
            with 3 stored elements and shape (3, 4)>
        >>> A.todense()
        matrix([[0., 0., 1., 3.],
        [3., 0., 0., 0.],
        [0., 0., 0., 0.]])

        >>> rows
        [(1, 1), (1, 2), (2, 1)]
        >>> columns
        [('a', 0), ('a', 1), ('b', 0), ('b', 1)]
        """
        from pandas.core.arrays.sparse.scipy_sparse import sparse_series_to_coo

        A, rows, columns = sparse_series_to_coo(
            self._parent, row_levels, column_levels, sort_labels=sort_labels
        )
        return A, rows, columns

    def to_dense(self) -> Series:
        # [FUNCTION DOC]
        # Returns a new series with the same index and name whose values are
        # the dense (numpy) version of the sparse array, built without copying
        # the densified data again. The parent is not modified.
        """
        Convert a Series from sparse values to dense.

        Returns
        -------
        Series:
            A Series with the same values, stored as a dense array.

        Examples
        --------
        >>> series = pd.Series(pd.arrays.SparseArray([0, 1, 0]))
        >>> series
        0    0
        1    1
        2    0
        dtype: Sparse[int64, 0]

        >>> series.sparse.to_dense()
        0    0
        1    1
        2    0
        dtype: int64
        """
        from pandas import Series

        return Series(
            self._parent.array.to_dense(),
            index=self._parent.index,
            name=self._parent.name,
            copy=False,
        )


class SparseFrameAccessor(BaseAccessor, PandasDelegate):
    # [CLASS DOC]
    # Frame-level sparse accessor (the sparse namespace on a frame). Inherits
    # the parent/validation protocol from the base accessor and the delegate
    # class. The parent is a frame in which every column has a sparse dtype
    # (checked at construction). Methods: build a frame from a scipy matrix
    # (class method), convert to a dense frame, convert to a coordinate matrix,
    # the overall density property, and a private index-preparation helper.
    # KNOWN ISSUE: the docstring parameter section says the data is a scipy
    # matrix, but the constructor takes the data frame to which the accessor
    # is attached.
    """
    DataFrame accessor for sparse data.

    It allows users to interact with a `DataFrame` that contains sparse data types
    (`SparseDtype`). It provides methods and attributes to efficiently work with sparse
    storage, reducing memory usage while maintaining compatibility with standard pandas
    operations.

    Parameters
    ----------
    data : scipy.sparse.spmatrix
        Must be convertible to csc format.

    See Also
    --------
    DataFrame.sparse.density : Ratio of non-sparse points to total (dense) data points.

    Examples
    --------
    >>> df = pd.DataFrame({"a": [1, 2, 0, 0], "b": [3, 0, 0, 4]}, dtype="Sparse[int]")
    >>> df.sparse.density
    np.float64(0.5)
    """

    def _validate(self, data) -> None:
        # [FUNCTION DOC]
        # HELPER: construction check. Raises an attribute error with the shared
        # message unless every column dtype of the frame is a sparse dtype.
        dtypes = data.dtypes
        if not all(isinstance(t, SparseDtype) for t in dtypes):
            raise AttributeError(self._validation_msg)

    @classmethod
    def from_spmatrix(cls, data, index=None, columns=None) -> DataFrame:
        # [FUNCTION DOC]
        # Class method. Builds a frame of sparse columns from a scipy sparse
        # matrix. data is any scipy sparse matrix (converted to compressed
        # column format); index and columns (default none) are optional labels
        # for rows and columns, defaulting to a default integer range index.
        # Raises a value error if the label lengths do not match the matrix
        # shape (via the index preparation helper).
        # Behavior: the stored indices are sorted first (mutating the converted
        # matrix, which for a matrix already in that format is the caller's
        # object) because the integer sparse index is built without integrity
        # validation. Each column becomes a sparse array whose stored values
        # and positions are the column's slice of the compressed storage, with
        # a sparse dtype taken from the data dtype. Returns the frame, created
        # without integrity verification.
        """
        Create a new DataFrame from a scipy sparse matrix.

        Parameters
        ----------
        data : scipy.sparse.spmatrix
            Must be convertible to csc format.
        index, columns : Index, optional
            Row and column labels to use for the resulting DataFrame.
            Defaults to a RangeIndex.

        Returns
        -------
        DataFrame
            Each column of the DataFrame is stored as a
            :class:`arrays.SparseArray`.

        See Also
        --------
        DataFrame.sparse.to_coo : Return the contents of the frame as a
            sparse SciPy COO matrix.

        Examples
        --------
        >>> import scipy.sparse
        >>> mat = scipy.sparse.eye(3, dtype=int)
        >>> pd.DataFrame.sparse.from_spmatrix(mat)
             0    1    2
        0    1    0    0
        1    0    1    0
        2    0    0    1
        """
        from pandas._libs.sparse import IntIndex

        from pandas import DataFrame

        data = data.tocsc()
        index, columns = cls._prep_index(data, index, columns)
        n_rows, n_columns = data.shape
        # We need to make sure indices are sorted, as we create
        # IntIndex with no input validation (i.e. check_integrity=False ).
        # Indices may already be sorted in scipy in which case this adds
        # a small overhead.
        data.sort_indices()
        indices = data.indices
        indptr = data.indptr
        array_data = data.data
        dtype = SparseDtype(array_data.dtype)
        arrays = []
        for i in range(n_columns):
            sl = slice(indptr[i], indptr[i + 1])
            idx = IntIndex(n_rows, indices[sl], check_integrity=False)
            arr = SparseArray._simple_new(array_data[sl], idx, dtype)
            arrays.append(arr)
        return DataFrame._from_arrays(
            arrays, columns=columns, index=index, verify_integrity=False
        )

    def to_dense(self) -> DataFrame:
        # [FUNCTION DOC]
        # Returns a new frame (same constructor type as the parent, same index
        # and columns) in which every sparse column is replaced by its dense
        # array. The parent is not modified.
        """
        Convert a DataFrame with sparse values to dense.

        Returns
        -------
        DataFrame
            A DataFrame with the same values stored as dense arrays.

        See Also
        --------
        DataFrame.sparse.density : Ratio of non-sparse points to total
            (dense) data points.

        Examples
        --------
        >>> df = pd.DataFrame({"A": pd.arrays.SparseArray([0, 1, 0])})
        >>> df.sparse.to_dense()
           A
        0  0
        1  1
        2  0
        """
        data = {k: v.array.to_dense() for k, v in self._parent.items()}
        return self._parent._constructor(
            data, index=self._parent.index, columns=self._parent.columns
        )

    def to_coo(self) -> spmatrix:
        # [FUNCTION DOC]
        # Exports the frame to a scipy coordinate matrix with the frame's shape.
        # Requires scipy (an import error is raised when missing, through the
        # optional dependency helper). The matrix dtype is the common type of
        # all column dtypes (the subtype if the common type is itself sparse).
        # For each column the stored positions become row coordinates, the
        # column number the column coordinate, and the stored values are cast
        # to the common type; all pieces are concatenated. Fill values are not
        # stored, so the matrix only holds the stored entries.
        # KNOWN ISSUE: a frame with no columns makes the concatenation of an
        # empty list fail with a value error instead of returning an empty
        # matrix.
        """
        Return the contents of the frame as a sparse SciPy COO matrix.

        Returns
        -------
        scipy.sparse.spmatrix
            If the caller is heterogeneous and contains booleans or objects,
            the result will be of dtype=object. See Notes.

        See Also
        --------
        DataFrame.sparse.to_dense : Convert a DataFrame with sparse values to dense.

        Notes
        -----
        The dtype will be the lowest-common-denominator type (implicit
        upcasting); that is to say if the dtypes (even of numeric types)
        are mixed, the one that accommodates all will be chosen.

        e.g. If the dtypes are float16 and float32, dtype will be upcast to
        float32. By numpy.find_common_type convention, mixing int64 and
        and uint64 will result in a float64 dtype.

        Examples
        --------
        >>> df = pd.DataFrame({"A": pd.arrays.SparseArray([0, 1, 0, 1])})
        >>> df.sparse.to_coo()
        <COOrdinate sparse matrix of dtype 'int64'
            with 2 stored elements and shape (4, 1)>
        """
        import_optional_dependency("scipy")
        from scipy.sparse import coo_matrix

        dtype = find_common_type(self._parent.dtypes.to_list())
        if isinstance(dtype, SparseDtype):
            dtype = dtype.subtype

        cols, rows, data = [], [], []
        for col, (_, ser) in enumerate(self._parent.items()):
            sp_arr = ser.array

            row = sp_arr.sp_index.indices
            cols.append(np.repeat(col, len(row)))
            rows.append(row)
            data.append(sp_arr.sp_values.astype(dtype, copy=False))

        cols_arr = np.concatenate(cols)
        rows_arr = np.concatenate(rows)
        data_arr = np.concatenate(data)
        return coo_matrix((data_arr, (rows_arr, cols_arr)), shape=self._parent.shape)

    @property
    def density(self) -> float:
        # [FUNCTION DOC]
        # Property. Returns the mean of the per-column densities (ratio of
        # stored points to length) as a floating scalar; since all columns have
        # equal length this equals the overall ratio of stored points to total
        # cells. An empty frame yields NaN.
        """
        Ratio of non-sparse points to total (dense) data points.

        See Also
        --------
        DataFrame.sparse.from_spmatrix : Create a new DataFrame from a
            scipy sparse matrix.

        Examples
        --------
        >>> df = pd.DataFrame({"A": pd.arrays.SparseArray([0, 1, 0, 1])})
        >>> df.sparse.density
        np.float64(0.5)
        """
        tmp = np.mean([column.array.density for _, column in self._parent.items()])
        return tmp

    @staticmethod
    def _prep_index(data, index, columns):
        # [FUNCTION DOC]
        # HELPER: static helper for the matrix import method. data is the
        # matrix (its shape gives the row and column counts); index and
        # columns are optional label collections. Missing labels become default
        # integer range indexes; supplied ones are converted to index objects.
        # Raises a value error if the number of column labels or row labels does
        # not match the matrix shape. Returns the pair (index, columns).
        from pandas.core.indexes.api import (
            default_index,
            ensure_index,
        )

        N, K = data.shape
        if index is None:
            index = default_index(N)
        else:
            index = ensure_index(index)
        if columns is None:
            columns = default_index(K)
        else:
            columns = ensure_index(columns)

        if len(columns) != K:
            raise ValueError(f"Column length mismatch: {len(columns)} vs. {K}")
        if len(index) != N:
            raise ValueError(f"Index length mismatch: {len(index)} vs. {N}")
        return index, columns
