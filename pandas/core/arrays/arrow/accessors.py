"""Accessors for arrow-backed data."""

# [MODULE DOC]
# Defines the series accessors for arrow-backed list and struct data: an abstract
# validating base class and the list and struct accessor classes. They are
# registered on series as the list and struct accessors elsewhere in the library
# (and re-exported from the arrow package). Pyarrow and the arrow dtype are
# imported only when pyarrow is installed; without it, constructing an accessor
# raises an attribute error. No other import-time side effects.
from __future__ import annotations

from abc import (
    ABCMeta,
    abstractmethod,
)
from typing import (
    TYPE_CHECKING,
    cast,
)

from pandas.compat import HAS_PYARROW

from pandas.core.dtypes.common import is_list_like

if HAS_PYARROW:
    import pyarrow as pa
    import pyarrow.compute as pc

    from pandas.core.dtypes.dtypes import ArrowDtype

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pandas import (
        DataFrame,
        Series,
    )


class ArrowAccessor(metaclass=ABCMeta):
    # [CLASS DOC]
    # Abstract base class for the series accessors that expose pyarrow-specific
    # operations (list and struct). Uses an abstract-base metaclass. State: the wrapped
    # series and a message template used for validation errors. Construction validates
    # immediately: subclasses must supply the template and decide which pyarrow types
    # are acceptable. Provides a property exposing the underlying pyarrow chunked array
    # of the series.
    @abstractmethod
    def __init__(self, data, validation_msg: str) -> None:
        # [FUNCTION DOC]
        # Abstract constructor (subclasses must override and call it). Parameters: data
        # is the series to wrap; validation_msg is a message template with a placeholder
        # named dtype, used when validation fails. Stores both and validates the data,
        # which may raise an attribute error.
        self._data = data
        self._validation_msg = validation_msg
        self._validate(data)

    @abstractmethod
    def _is_valid_pyarrow_dtype(self, pyarrow_dtype) -> bool:
        # [FUNCTION DOC]
        # HELPER: abstract predicate implemented by subclasses. Takes a pyarrow data
        # type and returns true if the accessor supports it. The base body does nothing.
        pass

    def _validate(self, data) -> None:
        # [FUNCTION DOC]
        # HELPER: checks that the series has an arrow-backed dtype with a supported
        # pyarrow type. Parameter data is the series. Raises an attribute error (rather
        # than a type or value error, so that introspection tools treat the accessor as
        # absent) with the formatted message template when pyarrow is not installed, the
        # dtype is not an arrow dtype, or the subclass predicate rejects the pyarrow
        # type.
        dtype = data.dtype
        if not HAS_PYARROW or not isinstance(dtype, ArrowDtype):
            # Raise AttributeError so that inspect can handle non-struct Series.
            raise AttributeError(self._validation_msg.format(dtype=dtype))

        if not self._is_valid_pyarrow_dtype(dtype.pyarrow_dtype):
            # Raise AttributeError so that inspect can handle invalid Series.
            raise AttributeError(self._validation_msg.format(dtype=dtype))

    @property
    def _pa_array(self):
        # [FUNCTION DOC]
        # HELPER: property returning the pyarrow chunked array stored in the wrapped
        # series' backing array.
        return self._data.array._pa_array


class ListAccessor(ArrowAccessor):
    # [CLASS DOC]
    # Accessor for series holding pyarrow list data (list, fixed-size list and large
    # list types), exposed as the list accessor on series. Inherits the validating base
    # class. Methods: length of each list, element or slice extraction by subscripting,
    # and flattening. Iteration is deliberately disabled.
    """
    Accessor object for list data properties of the Series values.

    Parameters
    ----------
    data : Series
        Series containing Arrow list data.
    """

    def __init__(self, data=None) -> None:
        # [FUNCTION DOC]
        # Constructor. Parameter data is the series to wrap (default None, which will
        # fail validation). Validates that the series has a pyarrow list-like dtype;
        # raises an attribute error with a message mentioning the list accessor
        # otherwise.
        super().__init__(
            data,
            validation_msg="Can only use the '.list' accessor with "
            "'list[pyarrow]' dtype, not {dtype}.",
        )

    def _is_valid_pyarrow_dtype(self, pyarrow_dtype) -> bool:
        # [FUNCTION DOC]
        # HELPER: returns true when the pyarrow type is a list, fixed-size list or large
        # list type.
        return (
            pa.types.is_list(pyarrow_dtype)
            or pa.types.is_fixed_size_list(pyarrow_dtype)
            or pa.types.is_large_list(pyarrow_dtype)
        )

    def len(self) -> Series:
        # [FUNCTION DOC]
        # Length of each list in the series. Returns a series (same index and name) with
        # an arrow-backed integer dtype, whose values are the list lengths from the
        # pyarrow length kernel; missing lists stay missing.
        """
        Return the length of each list in the Series.

        Returns
        -------
        pandas.Series
            The length of each list.

        See Also
        --------
        str.len : Python built-in function returning the length of an object.
        Series.size : Returns the length of the Series.
        StringMethods.len : Compute the length of each element in the Series/Index.

        Examples
        --------
        >>> import pyarrow as pa
        >>> s = pd.Series(
        ...     [
        ...         [1, 2, 3],
        ...         [3],
        ...     ],
        ...     dtype=pd.ArrowDtype(pa.list_(pa.int64())),
        ... )
        >>> s.list.len()
        0    3
        1    1
        dtype: int32[pyarrow]
        """
        from pandas import Series

        value_lengths = pc.list_value_length(self._pa_array)
        return Series(
            value_lengths,
            dtype=ArrowDtype(value_lengths.type),
            index=self._data.index,
            name=self._data.name,
        )

    def __getitem__(self, key: int | slice) -> Series:
        # [FUNCTION DOC]
        # Extract an element or slice from every list. Parameter key is an integer or a
        # slice. An integer selects that element from each list using the pyarrow
        # element kernel and returns a series with the element's arrow dtype. A slice
        # selects a sub-list from each list (start defaults to 0 and step to 1) and
        # returns a series with the sliced list's arrow dtype. Result keeps the index
        # and name. Raises a value error naming the type when key is neither.
        # KNOWN ISSUE: negative integer keys and negative slice bounds or steps are not
        # handled specially (the author's TODOs note the pyarrow limitation), so they
        # are passed straight to the kernel; in particular a missing start with a
        # negative step still starts at zero instead of the last element.
        """
        Index or slice lists in the Series.

        Parameters
        ----------
        key : int | slice
            Index or slice of indices to access from each list.

        Returns
        -------
        pandas.Series
            The list at requested index.

        See Also
        --------
        ListAccessor.flatten : Flatten list values.

        Examples
        --------
        >>> import pyarrow as pa
        >>> s = pd.Series(
        ...     [
        ...         [1, 2, 3],
        ...         [3],
        ...     ],
        ...     dtype=pd.ArrowDtype(pa.list_(pa.int64())),
        ... )
        >>> s.list[0]
        0    1
        1    3
        dtype: int64[pyarrow]
        """
        from pandas import Series

        if isinstance(key, int):
            # TODO: Support negative key but pyarrow does not allow
            # element index to be an array.
            # if key < 0:
            #     key = pc.add(key, pc.list_value_length(self._pa_array))
            element = pc.list_element(self._pa_array, key)
            return Series(
                element,
                dtype=ArrowDtype(element.type),
                index=self._data.index,
                name=self._data.name,
            )
        elif isinstance(key, slice):
            # TODO: Support negative start/stop/step, ideally this would be added
            # upstream in pyarrow.
            start, stop, step = key.start, key.stop, key.step
            if start is None:
                # TODO: When adding negative step support
                #  this should be setto last element of array
                # when step is negative.
                start = 0
            if step is None:
                step = 1
            sliced = pc.list_slice(self._pa_array, start, stop, step)
            return Series(
                sliced,
                dtype=ArrowDtype(sliced.type),
                index=self._data.index,
                name=self._data.name,
            )
        else:
            raise ValueError(f"key must be an int or slice, got {type(key).__name__}")

    def __iter__(self) -> Iterator:
        # [FUNCTION DOC]
        # Disables iteration. Always raises a type error saying the accessor object is
        # not iterable.
        raise TypeError(f"'{type(self).__name__}' object is not iterable")

    def flatten(self) -> Series:
        # [FUNCTION DOC]
        # Flatten all lists into a single series. Computes each list's length and the
        # concatenated elements; the index is the original index repeated by each list's
        # length (missing lists count as length zero). Returns a series with the element
        # arrow dtype and the original name.
        """
        Flatten list values.

        Returns
        -------
        pandas.Series
            The data from all lists in the series flattened.

        See Also
        --------
        ListAccessor.__getitem__ : Index or slice values in the Series.

        Examples
        --------
        >>> import pyarrow as pa
        >>> s = pd.Series(
        ...     [
        ...         [1, 2, 3],
        ...         [3],
        ...     ],
        ...     dtype=pd.ArrowDtype(pa.list_(pa.int64())),
        ... )
        >>> s.list.flatten()
        0    1
        0    2
        0    3
        1    3
        dtype: int64[pyarrow]
        """
        from pandas import Series

        counts = pa.compute.list_value_length(self._pa_array)
        flattened = pa.compute.list_flatten(self._pa_array)
        index = self._data.index.repeat(counts.fill_null(pa.scalar(0, counts.type)))
        return Series(
            flattened,
            dtype=ArrowDtype(flattened.type),
            index=index,
            name=self._data.name,
        )


class StructAccessor(ArrowAccessor):
    # [CLASS DOC]
    # Accessor for series holding pyarrow struct data, exposed as the struct accessor on
    # series. Inherits the validating base class. Provides the dtypes of the child
    # fields, extraction of a single (possibly nested) field as a series, and expansion
    # of all fields into a dataframe.
    """
    Accessor object for structured data properties of the Series values.

    Parameters
    ----------
    data : Series
        Series containing Arrow struct data.
    """

    def __init__(self, data=None) -> None:
        # [FUNCTION DOC]
        # Constructor. Parameter data is the series to wrap (default None, which will
        # fail validation). Validates that the series has a pyarrow struct dtype; raises
        # an attribute error with a message mentioning the struct accessor otherwise.
        super().__init__(
            data,
            validation_msg=(
                "Can only use the '.struct' accessor with 'struct[pyarrow]' "
                "dtype, not {dtype}."
            ),
        )

    def _is_valid_pyarrow_dtype(self, pyarrow_dtype) -> bool:
        # [FUNCTION DOC]
        # HELPER: returns true when the pyarrow type is a struct type.
        return pa.types.is_struct(pyarrow_dtype)

    @property
    def dtypes(self) -> Series:
        # [FUNCTION DOC]
        # Property: the dtype of each child field. Returns a series of arrow dtypes
        # indexed by the field names, in field order.
        """
        Return the dtype object of each child field of the struct.

        Returns
        -------
        pandas.Series
            The data type of each child field.

        See Also
        --------
        Series.dtype: Return the dtype object of the underlying data.

        Examples
        --------
        >>> import pyarrow as pa
        >>> s = pd.Series(
        ...     [
        ...         {"version": 1, "project": "pandas"},
        ...         {"version": 2, "project": "pandas"},
        ...         {"version": 1, "project": "numpy"},
        ...     ],
        ...     dtype=pd.ArrowDtype(
        ...         pa.struct([("version", pa.int64()), ("project", pa.string())])
        ...     ),
        ... )
        >>> s.struct.dtypes
        version     int64[pyarrow]
        project    string[pyarrow]
        dtype: object
        """
        from pandas import (
            Index,
            Series,
        )

        pa_type = self._data.dtype.pyarrow_dtype
        types = [ArrowDtype(struct.type) for struct in pa_type]
        names = [struct.name for struct in pa_type]
        return Series(types, index=Index(names))

    def field(
        self,
        name_or_index: list[str]
        | list[bytes]
        | list[int]
        | pc.Expression
        | bytes
        | str
        | int,
    ) -> Series:
        # [FUNCTION DOC]
        # Extract a child field as a series. Parameter name_or_index is a field name
        # (string or bytes), a field position (integer), a pyarrow compute expression,
        # or a list of names and positions describing a path into nested structs.
        # Returns a series with the field's arrow dtype and the original index. The
        # series name is the field name for names and positions (the name of the last
        # field for a path) and the string form of an expression. The name is determined
        # first by a nested helper; the data is extracted with the pyarrow struct-field
        # kernel using the original argument. Raises a value error (from the helper) for
        # unsupported argument types; invalid fields raise errors from pyarrow.
        """
        Extract a child field of a struct as a Series.

        Parameters
        ----------
        name_or_index : str | bytes | int | expression | list
            Name or index of the child field to extract.

            For list-like inputs, this will index into a nested
            struct.

        Returns
        -------
        pandas.Series
            The data corresponding to the selected child field.

        See Also
        --------
        Series.struct.explode : Return all child fields as a DataFrame.

        Notes
        -----
        The name of the resulting Series will be set using the following
        rules:

        - For string, bytes, or integer `name_or_index` (or a list of these, for
          a nested selection), the Series name is set to the selected
          field's name.
        - For a :class:`pyarrow.compute.Expression`, this is set to
          the string form of the expression.
        - For list-like `name_or_index`, the name will be set to the
          name of the final field selected.

        Examples
        --------
        >>> import pyarrow as pa
        >>> s = pd.Series(
        ...     [
        ...         {"version": 1, "project": "pandas"},
        ...         {"version": 2, "project": "pandas"},
        ...         {"version": 1, "project": "numpy"},
        ...     ],
        ...     dtype=pd.ArrowDtype(
        ...         pa.struct([("version", pa.int64()), ("project", pa.string())])
        ...     ),
        ... )

        Extract by field name.

        >>> s.struct.field("project")
        0    pandas
        1    pandas
        2     numpy
        Name: project, dtype: string[pyarrow]

        Extract by field index.

        >>> s.struct.field(0)
        0    1
        1    2
        2    1
        Name: version, dtype: int64[pyarrow]

        Or an expression

        >>> import pyarrow.compute as pc
        >>> s.struct.field(pc.field("project"))
        0    pandas
        1    pandas
        2     numpy
        Name: project, dtype: string[pyarrow]

        For nested struct types, you can pass a list of values to index
        multiple levels:

        >>> version_type = pa.struct(
        ...     [
        ...         ("major", pa.int64()),
        ...         ("minor", pa.int64()),
        ...     ]
        ... )
        >>> s = pd.Series(
        ...     [
        ...         {"version": {"major": 1, "minor": 5}, "project": "pandas"},
        ...         {"version": {"major": 2, "minor": 1}, "project": "pandas"},
        ...         {"version": {"major": 1, "minor": 26}, "project": "numpy"},
        ...     ],
        ...     dtype=pd.ArrowDtype(
        ...         pa.struct([("version", version_type), ("project", pa.string())])
        ...     ),
        ... )
        >>> s.struct.field(["version", "minor"])
        0     5
        1     1
        2    26
        Name: minor, dtype: int64[pyarrow]
        >>> s.struct.field([0, 0])
        0    1
        1    2
        2    1
        Name: major, dtype: int64[pyarrow]
        """
        from pandas import Series

        def get_name(
            level_name_or_index: list[str]
            | list[bytes]
            | list[int]
            | pc.Expression
            | bytes
            | str
            | int,
            data: pa.ChunkedArray,
        ):
            # [FUNCTION DOC]
            # HELPER: nested function that determines the result series name for the
            # field selection. Parameters: level_name_or_index is the selector
            # (position, name, bytes, expression or list) and data is the pyarrow
            # chunked array of structs at that level. A position yields the name of the
            # field at that position; a string or bytes yields itself; an expression
            # yields its string form; a list is walked level by level (recursively
            # resolving each element against the current struct type and descending into
            # that field), and the name of the final field is returned. Anything else
            # raises a value error listing the accepted types.
            if isinstance(level_name_or_index, int):
                name = data.type.field(level_name_or_index).name
            elif isinstance(level_name_or_index, (str, bytes)):
                name = level_name_or_index
            elif isinstance(level_name_or_index, pc.Expression):
                name = str(level_name_or_index)
            elif is_list_like(level_name_or_index):
                # For nested input like [2, 1, 2]
                # iteratively get the struct and field name. The last
                # one is used for the name of the index.
                level_name_or_index = list(reversed(level_name_or_index))
                selected = data
                while level_name_or_index:
                    # we need the cast, otherwise mypy complains about
                    # getting ints, bytes, or str here, which isn't possible.
                    level_name_or_index = cast(list, level_name_or_index)
                    name_or_index = level_name_or_index.pop()
                    name = get_name(name_or_index, selected)
                    selected = selected.type.field(selected.type.get_field_index(name))
                    name = selected.name
            else:
                raise ValueError(
                    "name_or_index must be an int, str, bytes, "
                    "pyarrow.compute.Expression, or list of those"
                )
            return name

        pa_arr = self._data.array._pa_array
        name = get_name(name_or_index, pa_arr)
        field_arr = pc.struct_field(pa_arr, name_or_index)

        return Series(
            field_arr,
            dtype=ArrowDtype(field_arr.type),
            index=self._data.index,
            name=name,
        )

    def explode(self) -> DataFrame:
        # [FUNCTION DOC]
        # Expand all child fields into columns. Returns a dataframe made by extracting
        # each field by position and concatenating them side by side; the column names
        # come from the field names.
        """
        Extract all child fields of a struct as a DataFrame.

        Returns
        -------
        pandas.DataFrame
            The data corresponding to all child fields.

        See Also
        --------
        Series.struct.field : Return a single child field as a Series.

        Examples
        --------
        >>> import pyarrow as pa
        >>> s = pd.Series(
        ...     [
        ...         {"version": 1, "project": "pandas"},
        ...         {"version": 2, "project": "pandas"},
        ...         {"version": 1, "project": "numpy"},
        ...     ],
        ...     dtype=pd.ArrowDtype(
        ...         pa.struct([("version", pa.int64()), ("project", pa.string())])
        ...     ),
        ... )

        >>> s.struct.explode()
           version project
        0        1  pandas
        1        2  pandas
        2        1   numpy
        """
        from pandas import concat

        pa_type = self._pa_array.type
        return concat(
            [self.field(i) for i in range(pa_type.num_fields)], axis="columns"
        )
