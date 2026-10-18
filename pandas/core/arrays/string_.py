# [MODULE DOC]
# Core of the string extension type. Public contents: the string dtype class
# (registered with the extension dtype registry, publicly exposed under the
# top-level namespace), the python-storage string array class (publicly exposed
# under the arrays namespace), and a base mixin class shared by all string
# arrays (including the pyarrow-backed one defined in the sibling arrow-string
# module). The dtype holds two settings: a storage backend (python objects or
# pyarrow) and the missing-value sentinel (the pandas NA singleton or NaN),
# which also selects the semantics of comparison, reductions and string
# method results.
# Relationships: the python-storage array builds on the numpy extension array
# (object ndarray backing) and the base mixin; arithmetic/comparison, masked
# reductions and nan-aware reductions are delegated to the ops, masked
# reduction and nanops modules; boolean/integer/floating nullable arrays are
# produced by string methods and casts. The arrow-string module is imported
# lazily by the dtype to avoid circular imports.
# Import-time side effects: registers the string dtype class with the global
# extension dtype registry; imports pyarrow and its compute module only if
# available.
from __future__ import annotations

from functools import partial
import operator
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
    cast,
)
import warnings

import numpy as np

from pandas._config import (
    get_option,
    using_string_dtype,
)

from pandas._libs import (
    lib,
    missing as libmissing,
)
from pandas._libs.arrays import NDArrayBacked
from pandas._libs.lib import ensure_string_array
from pandas.compat import (
    HAS_PYARROW,
    PYARROW_MIN_VERSION,
)
from pandas.compat.numpy import function as nv
from pandas.errors import Pandas4Warning
from pandas.util._decorators import (
    set_module,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.base import (
    ExtensionDtype,
    StorageExtensionDtype,
    register_extension_dtype,
)
from pandas.core.dtypes.common import (
    is_array_like,
    is_bool_dtype,
    is_integer_dtype,
    is_object_dtype,
    is_string_dtype,
    pandas_dtype,
)

from pandas.core import (
    missing,
    nanops,
    ops,
    roperator,
)
from pandas.core.algorithms import isin
from pandas.core.array_algos import masked_reductions
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.floating import (
    FloatingArray,
    FloatingDtype,
)
from pandas.core.arrays.integer import (
    IntegerArray,
    IntegerDtype,
)
from pandas.core.arrays.numpy_ import NumpyExtensionArray
from pandas.core.construction import extract_array
from pandas.core.indexers import check_array_indexer
from pandas.core.missing import isna

from pandas.io.formats import printing

if HAS_PYARROW:
    import pyarrow as pa
    import pyarrow.compute as pc

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    import pyarrow

    from pandas._typing import (
        ArrayLike,
        AxisInt,
        Dtype,
        DtypeObj,
        NumpySorter,
        NumpyValueArrayLike,
        Scalar,
        npt,
        type_t,
    )

    from pandas import Series


@set_module("pandas")
@register_extension_dtype
class StringDtype(StorageExtensionDtype):
    # [CLASS DOC]
    # Dtype describing string data. Inherits from the storage-aware extension
    # dtype base. State: the storage backend name (either python or pyarrow)
    # and the missing-value sentinel (the NA singleton or NaN); both take part
    # in equality and pickling (declared in the metadata tuple). Key methods:
    # name and na_value properties, string construction, the array-type
    # resolver (returns the python-storage or the arrow-storage array class),
    # common-type resolution among string dtypes and conversion from arrow
    # data. Invariant: the NaN sentinel is always normalized to the single numpy
    # NaN object so identity checks work.
    """
    Extension dtype for string data.

    .. warning::

       StringDtype is considered experimental. The implementation and
       parts of the API may change without warning.

    Parameters
    ----------
    storage : {"python", "pyarrow"}, optional
        If not given, the value of ``pd.options.mode.string_storage``.
    na_value : {np.nan, pd.NA}, default pd.NA
        Whether the dtype follows NaN or NA missing value semantics.

    Attributes
    ----------
    storage
    na_value

    Methods
    -------
    None

    See Also
    --------
    BooleanDtype : Extension dtype for boolean data.

    Examples
    --------
    >>> pd.StringDtype()
    <StringDtype(na_value=<NA>)>

    >>> pd.StringDtype(storage="python")
    <StringDtype(storage='python', na_value=<NA>)>
    """

    @property
    def name(self) -> str:  # type: ignore[override]
        # [FUNCTION DOC]
        # Property returning the dtype name: the text "string" when the
        # missing-value sentinel is NA, and the text "str" when it is NaN.
        if self._na_value is libmissing.NA:
            return "string"
        else:
            return "str"

    #: StringDtype().na_value uses pandas.NA except the implementation that
    # follows NumPy semantics, which uses nan.
    @property
    def na_value(self) -> libmissing.NAType | float:  # type: ignore[override]
        # [FUNCTION DOC]
        # Property returning the stored missing-value sentinel (the NA
        # singleton or NaN).
        """
        The missing value representation for this dtype.

        This value indicates which missing value semantics are used by this dtype.
        Returns ``np.nan`` for the default string dtype with NumPy semantics,
        and ``pd.NA`` for the opt-in string dtype with pandas NA semantics.

        See Also
        --------
        isna : Detect missing values.
        NA : Missing value indicator for nullable dtypes.

        Examples
        --------
        >>> ser = pd.Series(["a", "b"])
        >>> ser.dtype
        <StringDtype(na_value=nan)>
        >>> ser.dtype.na_value
        nan
        """
        return self._na_value

    @property
    def storage(self) -> str:
        # [FUNCTION DOC]
        # Property returning the storage backend name, either python or
        # pyarrow.
        """
        The storage backend for this dtype.

        Can be either "pyarrow" or "python".

        See Also
        --------
        StringDtype.na_value : The missing value for this dtype.

        Examples
        --------
        >>> ser = pd.Series(["a", "b"])
        >>> ser.dtype
        <StringDtype(na_value=nan)>
        >>> ser.dtype.storage
        'pyarrow'
        """
        return self._storage

    _metadata = ("storage", "_na_value")  # type: ignore[assignment]

    def __init__(
        self,
        storage: str | None = None,
        na_value: libmissing.NAType | float = libmissing.NA,
    ) -> None:
        # [FUNCTION DOC]
        # Creates a string dtype. storage (default none) is the backend name,
        # python or pyarrow; when none, the global string-storage option is
        # read, and its automatic setting resolves to pyarrow if pyarrow is
        # installed, else python. na_value (default the NA singleton) must be
        # the NA singleton or a float NaN (any NaN is replaced by the canonical
        # numpy NaN).
        # Raises a value error for an unknown storage name or for any other
        # na_value; raises an import error if pyarrow storage is requested but
        # pyarrow is not installed. Stores both settings on the instance.
        # infer defaults
        if storage is None:
            storage = get_option("mode.string_storage")
            if storage == "auto":
                if HAS_PYARROW:
                    storage = "pyarrow"
                else:
                    storage = "python"

        # validate options
        if storage not in {"python", "pyarrow"}:
            raise ValueError(
                f"Storage must be 'python' or 'pyarrow'. Got {storage} instead."
            )
        if storage == "pyarrow" and not HAS_PYARROW:
            raise ImportError(
                f"pyarrow>={PYARROW_MIN_VERSION} is required for PyArrow "
                "backed StringArray."
            )

        if isinstance(na_value, float) and np.isnan(na_value):
            # when passed a NaN value, always set to np.nan to ensure we use
            # a consistent NaN value (and we can use `dtype.na_value is np.nan`)
            na_value = np.nan
        elif na_value is not libmissing.NA:
            raise ValueError(f"'na_value' must be np.nan or pd.NA, got {na_value}")

        self._storage = cast(str, storage)
        self._na_value = na_value

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns a text such as the class name with its settings in angle
        # brackets: the storage is shown only when it is python (pyarrow is
        # treated as the default and omitted), and the missing-value sentinel is
        # always shown.
        storage = "" if self.storage == "pyarrow" else "storage='python', "
        return f"<StringDtype({storage}na_value={self._na_value})>"

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # Equality. A text other is equal if it is "string" or this dtype's own
        # name; otherwise it is parsed with the string constructor and, if
        # parsing fails with a type error or import error, the answer is false.
        # Another string dtype is equal when storage matches and the sentinel is
        # the identical object (NA versus NaN cannot be compared with normal
        # equality). Anything else is not equal. Returns a boolean.
        # KNOWN ISSUE: the text "string" compares equal for the NaN-semantics
        # dtype as well (a source TODO questions this), so equality with that
        # text is true regardless of the sentinel.
        # we need to override the base class __eq__ because na_value (NA or NaN)
        # cannot be checked with normal `==`
        if isinstance(other, str):
            # TODO should dtype == "string" work for the NaN variant?
            if other == "string" or other == self.name:  # noqa: PLR1714 (repeated-equality-comparison)
                return True
            try:
                other = self.construct_from_string(other)
            except (TypeError, ImportError):
                # TypeError if `other` is not a valid string for StringDtype
                # ImportError if pyarrow is not installed for "string[pyarrow]"
                return False
        if isinstance(other, type(self)):
            return self.storage == other.storage and self.na_value is other.na_value
        return False

    def __setstate__(self, state: MutableMapping[str, Any]) -> None:
        # [FUNCTION DOC]
        # Unpickling support. state is a mapping with optional storage and
        # sentinel entries (entries are removed from the mapping as they are
        # read); missing entries default to python storage and the NA sentinel,
        # which handles pickles from older versions without a sentinel.
        # back-compat for pandas < 2.3, where na_value did not yet exist
        self._storage = state.pop("storage", "python")
        self._na_value = state.pop("_na_value", libmissing.NA)

    def __hash__(self) -> int:
        # [FUNCTION DOC]
        # Returns the hash from the parent implementation (based on the
        # metadata tuple); redefined only because defining equality removes
        # the inherited hash.
        # need to override __hash__ as well because of overriding __eq__
        return super().__hash__()

    def __reduce__(self):
        # [FUNCTION DOC]
        # Pickling support: returns the class and the tuple of (storage,
        # sentinel) used to recreate the dtype.
        return StringDtype, (self.storage, self.na_value)

    @property
    def type(self) -> type[str]:
        # [FUNCTION DOC]
        # Property returning the scalar type of the elements: the built-in
        # text type.
        return str

    @classmethod
    def construct_from_string(cls, string) -> Self:
        # [FUNCTION DOC]
        # Class method parsing a dtype name. string must be a text value
        # (otherwise a type error is raised). Mapping: "string" gives the
        # default dtype (option-driven storage, NA sentinel); "str" gives the
        # NaN-sentinel dtype only when the global string-dtype inference option
        # is on; the names with an explicit python or pyarrow storage suffix in brackets give the NA-sentinel
        # dtype with that explicit storage. Any other text raises a type error
        # naming the class and the text.
        # KNOWN ISSUE: the docstring header says "Raise" instead of "Raises",
        # and its table omits the "str" case that the code accepts.
        """
        Construct a StringDtype from a string.

        Parameters
        ----------
        string : str
            The type of the name. The storage type will be taking from `string`.
            Valid options and their storage types are

            ========================== ==============================================
            string                     result storage
            ========================== ==============================================
            ``'string'``               pd.options.mode.string_storage, default python
            ``'string[python]'``       python
            ``'string[pyarrow]'``      pyarrow
            ========================== ==============================================

        Returns
        -------
        StringDtype

        Raise
        -----
        TypeError
            If the string is not a valid option.
        """
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
        if string == "string":
            return cls()
        elif string == "str" and using_string_dtype():
            return cls(na_value=np.nan)
        elif string == "string[python]":
            return cls(storage="python")
        elif string == "string[pyarrow]":
            return cls(storage="pyarrow")
        else:
            raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")

    def construct_array_type(self) -> type_t[BaseStringArray]:
        # [FUNCTION DOC]
        # Returns the array class for this dtype: the python-storage array
        # class when storage is python, otherwise the arrow-backed string array
        # class (imported lazily from the sibling module). The branches
        # regarding the sentinel are redundant: the result depends only on
        # storage.
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from pandas.core.arrays.string_arrow import (
            ArrowStringArray,
        )

        if self.storage == "python" and self._na_value is libmissing.NA:
            return StringArray
        elif self.storage == "pyarrow" and self._na_value is libmissing.NA:
            return ArrowStringArray
        elif self.storage == "python":
            return StringArray
        else:
            return ArrowStringArray

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        # [FUNCTION DOC]
        # HELPER: hook for concatenation / common-type resolution. dtypes is a
        # list of dtypes to combine. If any entry is neither a string dtype nor
        # a numpy fixed or variable-width text dtype, returns none (no common
        # type known here). Otherwise returns a string dtype: storage is
        # pyarrow if both storages occur (else the single one), and the
        # sentinel is NA if both sentinels occur (else the single one).
        # Numpy text dtypes are ignored when choosing. If the list has only
        # numpy text dtypes the selection step fails (no string dtype present),
        # which cannot happen when called on a string dtype's own list.
        storages = set()
        na_values = set()

        for dtype in dtypes:
            if isinstance(dtype, StringDtype):
                storages.add(dtype.storage)
                na_values.add(dtype.na_value)
            elif isinstance(dtype, np.dtype) and dtype.kind in ("U", "T"):
                continue
            else:
                return None

        if len(storages) == 2:
            # if both python and pyarrow storage -> priority to pyarrow
            storage = "pyarrow"
        else:
            storage = next(iter(storages))

        na_value: libmissing.NAType | float
        if len(na_values) == 2:
            # if both NaN and NA -> priority to NA
            na_value = libmissing.NA
        else:
            na_value = next(iter(na_values))

        return StringDtype(storage=storage, na_value=na_value)

    def __from_arrow__(
        self, array: pyarrow.Array | pyarrow.ChunkedArray
    ) -> BaseStringArray:
        # [FUNCTION DOC]
        # Converts pyarrow data (array or chunked array) to a string array of
        # this dtype. For pyarrow storage: checks pyarrow availability, casts
        # to the large string type unless already that, and wraps it in the
        # arrow-backed string array with this dtype. For python storage: each
        # chunk is converted separately to a numpy object array of strings
        # (missing values replaced by this dtype's sentinel) to avoid overflow
        # when joining large arrow data, then the pieces are concatenated (an
        # empty object array if there are no chunks). The array object is built
        # by bypassing the validating constructor (a known performance
        # shortcut). Returns the new array.
        """
        Construct StringArray from pyarrow Array/ChunkedArray.
        """
        if self.storage == "pyarrow":
            from pandas.core.arrays.string_arrow import (
                ArrowStringArray,
                _check_pyarrow_available,
            )

            _check_pyarrow_available()

            if not pa.types.is_large_string(array.type):
                array = pc.cast(array, pa.large_string())

            return ArrowStringArray(array, dtype=self)

        else:
            import pyarrow

            if isinstance(array, pyarrow.Array):
                chunks = [array]
            else:
                # pyarrow.ChunkedArray
                chunks = array.chunks

            results = []
            for arr in chunks:
                # convert chunk by chunk to numpy and concatenate then, to avoid
                # overflow for large string data when concatenating the pyarrow arrays
                arr = arr.to_numpy(zero_copy_only=False)
                arr = ensure_string_array(arr, na_value=self.na_value)
                results.append(arr)

        if len(chunks) == 0:
            arr = np.array([], dtype=object)
        else:
            arr = np.concatenate(results)

        # Bypass validation inside StringArray constructor, see GH#47781
        new_string_array = StringArray.__new__(StringArray)
        NDArrayBacked.__init__(new_string_array, arr, self)
        return new_string_array


class BaseStringArray(ExtensionArray):
    # [CLASS DOC]
    # HELPER: shared base for the python-storage and arrow-storage string
    # arrays. Inherits the generic extension array interface and declares that
    # the dtype attribute is a string dtype. Provides: a deprecated logical
    # operator fallback, list conversion, the display formatter, the
    # element-wise string-method mapping machinery (separate paths for NA and
    # NaN semantics) and a view restriction. Concrete subclasses supply storage
    # and construction.
    """
    Mixin class for StringArray, ArrowStringArray.
    """

    dtype: StringDtype

    # TODO(4.0): Once the deprecation here is enforced, this method can be
    #  removed and we use the parent class method instead.
    def _logical_method(self, other, op):
        # [FUNCTION DOC]
        # HELPER: hook for the logical operators (and, or, xor). other is the
        # right operand; op is the operator function. Only the reflected
        # operators with a boolean numpy array operand are supported, for
        # backward compatibility: they emit a deprecation warning (category for
        # the next major release) and compute the operation after casting this
        # array to boolean. In all other cases returns the not-implemented
        # marker so the caller falls back to default behavior.
        # Side effect: emits the warning.
        if (
            op in (roperator.ror_, roperator.rand_, roperator.rxor)
            and isinstance(other, np.ndarray)
            and other.dtype == bool
        ):
            # GH#60234 backward compatibility for the move to StringDtype in 3.0
            op_name = op.__name__[1:].strip("_")
            warnings.warn(
                f"'{op_name}' operations between boolean dtype and {self.dtype} are "
                "deprecated and will raise in a future version. Explicitly "
                "cast the strings to a boolean dtype before operating instead.",
                Pandas4Warning,
                stacklevel=find_stack_level(),
            )
            return op(other, self.astype(bool))
        return NotImplemented

    def tolist(self) -> list:
        # [FUNCTION DOC]
        # Returns a plain list of the elements. For arrays with more than one
        # dimension returns a nested list built by converting each sub-array
        # recursively; otherwise a list of the elements of the numpy
        # conversion (strings and the missing-value sentinel).
        """
        Return a list of the value.

        These are each a scalar type, which is a Python scalar
        (for str, int, float) or pandas scalar
        (for Timestamp/Timedelta/Interval/Period)

        Returns
        ----------
        list

        Examples
        ----------
        >>> arr = pd.array(["a", "b", "c"])
        >>> arr.tolist()
        ['a', 'b', 'c']
        """
        if self.ndim > 1:
            return [x.tolist() for x in self]
        return list(self.to_numpy())

    def _formatter(self, boxed: bool = False):
        # [FUNCTION DOC]
        # HELPER: returns the per-element display function used for printing.
        # boxed (default false) means the value is shown inside a container
        # cell: strings are quoted only when boxed is false. The function
        # escapes tab, carriage return and newline characters.
        formatter = partial(
            printing.pprint_thing,
            escape_chars=("\t", "\r", "\n"),
            quote_strings=not boxed,
        )
        return formatter

    def _str_map(
        self,
        f,
        na_value=lib.no_default,
        dtype: Dtype | None = None,
        convert: bool = True,
    ):
        # [FUNCTION DOC]
        # HELPER: maps a function over the elements for the string accessor
        # methods, with NA semantics. f is applied to each non-missing string;
        # na_value (default: use the dtype sentinel) is the value for missing
        # inputs; dtype (default: this dtype) is the requested result dtype;
        # convert is accepted but not used.
        # If the dtype sentinel is NaN, the NaN-semantics variant is used
        # instead. Otherwise: for integer or boolean result dtypes, the mapped
        # values are placed in a nullable integer or boolean array whose mask is
        # the input missing mask (when na_value is itself missing, a dummy value
        # fills those slots and the mask is kept; otherwise the given na_value is
        # used, booleans coerced, and the mask is cleared). For other dtypes the
        # result comes from the shared string-or-object helper.
        # Returns an array (nullable integer/boolean, string array, or object
        # ndarray).
        # KNOWN ISSUE: the convert parameter is never used.
        if self.dtype.na_value is np.nan:
            return self._str_map_nan_semantics(f, na_value=na_value, dtype=dtype)

        from pandas.arrays import BooleanArray

        if dtype is None:
            dtype = self.dtype
        if na_value is lib.no_default:
            na_value = self.dtype.na_value

        mask = isna(self)
        arr = np.asarray(self)

        if is_integer_dtype(dtype) or is_bool_dtype(dtype):
            constructor: type[IntegerArray | BooleanArray]
            if is_integer_dtype(dtype):
                constructor = IntegerArray
            else:
                constructor = BooleanArray

            na_value_is_na = isna(na_value)
            if na_value_is_na:
                na_value = 1
            elif dtype == np.dtype("bool"):
                # GH#55736
                na_value = bool(na_value)
            result = lib.map_infer_mask(
                arr,
                f,
                mask.view("uint8"),
                convert=False,
                na_value=na_value,
                # error: Argument 1 to "dtype" has incompatible type
                # "Union[ExtensionDtype, str, dtype[Any], Type[object]]"; expected
                # "Type[object]"
                dtype=np.dtype(cast(type, dtype)),
            )

            if not na_value_is_na:
                mask[:] = False

            return constructor(result, mask)

        else:
            return self._str_map_str_or_object(dtype, na_value, arr, f, mask)

    def _str_map_str_or_object(
        self,
        dtype,
        na_value,
        arr: np.ndarray,
        f,
        mask: npt.NDArray[np.bool_],
    ):
        # [FUNCTION DOC]
        # HELPER: the shared tail of the string mapping methods for result
        # dtypes that are a string dtype or object. dtype is the requested
        # result dtype; na_value fills masked positions; arr is the object
        # ndarray of values; f is the function; mask is the boolean missing
        # mask (viewed as bytes for the cython mapper).
        # For a string (non-object) dtype: maps without type inference, then
        # for pyarrow storage builds a large-string arrow array with the mask
        # and wraps it using the subclass hook, for python storage constructs
        # the same array class with this dtype. For object dtype (result type
        # truly object or unknown): returns the inferred-type mapping result
        # as an ndarray without wrapping.
        # _str_map helper for case where dtype is either string dtype or object
        if is_string_dtype(dtype) and not is_object_dtype(dtype):
            # i.e. StringDtype
            result = lib.map_infer_mask(
                arr, f, mask.view("uint8"), convert=False, na_value=na_value
            )
            if self.dtype.storage == "pyarrow":
                import pyarrow as pa

                # TODO: shouldn't this already be caught my passed mask?
                #  it isn't in test_extract_expand_capture_groups_index
                # mask = mask | np.array(
                #    [x is libmissing.NA for x in result], dtype=bool
                #    )

                result = pa.array(
                    result, mask=mask, type=pa.large_string(), from_pandas=True
                )
                # error: "BaseStringArray" has no attribute "_from_pyarrow_array"
                return self._from_pyarrow_array(result)  # type: ignore[attr-defined]
            else:
                # StringArray
                # error: Too many arguments for "BaseStringArray"
                return type(self)(result, dtype=self.dtype)  # type: ignore[call-arg]

        else:
            # This is when the result type is object. We reach this when
            # -> We know the result type is truly object (e.g. .encode returns bytes
            #    or .findall returns a list).
            # -> We don't know the result type. E.g. `.get` can return anything.
            return lib.map_infer_mask(arr, f, mask.view("uint8"))

    def _str_map_nan_semantics(
        self, f, na_value=lib.no_default, dtype: Dtype | None = None
    ):
        # [FUNCTION DOC]
        # HELPER: NaN-semantics variant of the string mapping routine. f is the
        # function applied to non-missing strings; na_value (default: false for
        # boolean results, otherwise the dtype sentinel) is the output for
        # missing inputs; dtype (default: this dtype) is the result dtype.
        # For integer or boolean result dtypes returns a plain numpy array (not
        # a nullable array): a missing na_value becomes zero for integers and
        # false for booleans during mapping, and for integer results with
        # missing inputs and a missing na_value the array is converted to
        # float with NaN at the missing positions afterwards. Other dtypes use
        # the shared string-or-object helper.
        if dtype is None:
            dtype = self.dtype
        if na_value is lib.no_default:
            if is_bool_dtype(dtype):
                # NaN propagates as False
                na_value = False
            else:
                na_value = self.dtype.na_value

        mask = isna(self)
        arr = np.asarray(self)

        if is_integer_dtype(dtype) or is_bool_dtype(dtype):
            na_value_is_na = isna(na_value)
            if na_value_is_na:
                if is_integer_dtype(dtype):
                    na_value = 0
                else:
                    # NaN propagates as False
                    na_value = False

            result = lib.map_infer_mask(
                arr,
                f,
                mask.view("uint8"),
                convert=False,
                na_value=na_value,
                dtype=np.dtype(cast(type, dtype)),
            )
            if na_value_is_na and is_integer_dtype(dtype) and mask.any():
                # TODO: we could alternatively do this check before map_infer_mask
                #  and adjust the dtype/na_value we pass there. Which is more
                #  performant?
                result = result.astype("float64")
                result[mask] = np.nan

            return result

        else:
            return self._str_map_str_or_object(dtype, na_value, arr, f, mask)

    def view(self, dtype: Dtype | None = None) -> Self:
        # [FUNCTION DOC]
        # Returns a view of the array. dtype (default none) must be none:
        # reinterpreting string data is not allowed and any other value raises
        # a type error. Otherwise delegates to the parent view.
        if dtype is not None:
            raise TypeError("Cannot change data-type for string array.")
        return super().view()


@set_module("pandas.arrays")
# error: Definition of "_concat_same_type" in base class "NDArrayBacked" is
# incompatible with definition in base class "ExtensionArray"
class StringArray(BaseStringArray, NumpyExtensionArray):  # type: ignore[misc]
    # [CLASS DOC]
    # Python-storage string array: an object-dtype numpy array holding strings
    # and the missing-value sentinel of its dtype. Inherits the shared string
    # mixin first and the numpy extension array second (storage, indexing,
    # ufunc machinery). State: the backing object ndarray and a string dtype
    # (python storage; sentinel NA or NaN). Invariant (checked by the
    # validating constructor, bypassed by internal fast constructors): every
    # element is a string or the sentinel. Comparison returns a nullable boolean
    # array for NA semantics and a plain boolean ndarray for NaN semantics;
    # arithmetic is limited to concatenation-like operations; reductions are
    # limited to min, max, sum and (for NaN semantics) any/all.
    # undo the NumpyExtensionArray hack
    """
    Extension array for string data.

    .. warning::

       StringArray is considered experimental. The implementation and
       parts of the API may change without warning.

    Parameters
    ----------
    values : array-like
        The array of data.

        .. warning::

           Currently, this expects an object-dtype ndarray
           where the elements are Python strings
           or nan-likes (``None``, ``np.nan``, ``NA``).
           This may change without warning in the future. Use
           :meth:`pandas.array` with ``dtype="string"`` for a stable way of
           creating a `StringArray` from any sequence.

           StringArray accepts array-likes containing
           nan-likes(``None``, ``np.nan``) for the ``values`` parameter
           in addition to strings and :attr:`pandas.NA`

    dtype : StringDtype
        Dtype for the array.
    copy : bool, default False
        Whether to copy the array of data.

    Attributes
    ----------
    None

    Methods
    -------
    None

    See Also
    --------
    :func:`array`
        The recommended function for creating a StringArray.
    Series.str
        The string methods are available on Series backed by
        a StringArray.

    Notes
    -----
    StringArray returns a BooleanArray for comparison methods.

    Examples
    --------
    >>> pd.array(["This is", "some text", None, "data."], dtype="string")
    <ArrowStringArray>
    ['This is', 'some text', <NA>, 'data.']
    Length: 4, dtype: string

    Unlike arrays instantiated with ``dtype="object"``, ``StringArray``
    will convert the values to strings.

    >>> pd.array(["1", 1], dtype="object")
    <NumpyExtensionArray>
    ['1', 1]
    Length: 2, dtype: object
    >>> pd.array(["1", 1], dtype="string")
    <ArrowStringArray>
    ['1', '1']
    Length: 2, dtype: string

    However, instantiating StringArrays directly with non-strings will raise an error.

    For comparison methods, `StringArray` returns a :class:`pandas.BooleanArray`:

    >>> pd.array(["a", None, "c"], dtype="string[python]") == "a"
    <BooleanArray>
    [True, <NA>, False]
    Length: 3, dtype: boolean
    """

    _typ = "extension"

    def __init__(
        self, values, *, dtype: StringDtype | None = None, copy: bool = False
    ) -> None:
        # [FUNCTION DOC]
        # Validating constructor. values is an array-like (an object ndarray
        # of strings or missing values is expected; extension wrappers are
        # unwrapped); dtype (keyword-only, default none) is a string dtype,
        # defaulting to a new default string dtype; copy (keyword-only, default
        # false) copies the data.
        # Behavior: stores the data through the parent constructor, validates
        # content against the dtype unless the input already was this class,
        # then sets the dtype. Raises a value error for non-string content or a
        # non-object ndarray (see the validation helper).
        if dtype is None:
            dtype = StringDtype()
        values = extract_array(values)

        super().__init__(values, copy=copy)
        if not isinstance(values, type(self)):
            self._validate(dtype)
        NDArrayBacked.__init__(
            self,
            self._ndarray,
            dtype,
        )

    def _validate(self, dtype: StringDtype) -> None:
        # [FUNCTION DOC]
        # HELPER: content check used by the validating constructor. dtype is
        # the string dtype whose sentinel decides the rule. For the NA
        # sentinel: raises a value error if the data (non-empty) is not all
        # strings/missing, or if the backing array is not object dtype; then
        # converts every NaN-like missing value in place to the NA singleton
        # (flattening arrays with more than two dimensions first). For the NaN
        # sentinel: raises a value error under the same two conditions but does
        # not normalize missing values.
        # Side effect: mutates the backing array in the NA case.
        # KNOWN ISSUE: in the NaN branch the second part of the dtype error
        # message lacks the leading marker for formatted strings, so the text
        # shows the literal placeholder instead of the actual dtype.
        """Validate that we only store NA or strings."""

        if dtype._na_value is libmissing.NA:
            if len(self._ndarray) and not lib.is_string_array(
                self._ndarray, skipna=True
            ):
                raise ValueError(
                    "StringArray requires a sequence of strings or pandas.NA"
                )
            if self._ndarray.dtype != "object":
                raise ValueError(
                    "StringArray requires a sequence of strings or pandas.NA. Got "
                    f"'{self._ndarray.dtype}' dtype instead."
                )
            # Check to see if need to convert Na values to pd.NA
            if self._ndarray.ndim > 2:
                # Ravel if ndims > 2 b/c no cythonized version available
                lib.convert_nans_to_NA(self._ndarray.ravel("K"))
            else:
                lib.convert_nans_to_NA(self._ndarray)
        else:
            # Validate that we only store NaN or strings.
            if len(self._ndarray) and not lib.is_string_array(
                self._ndarray, skipna=True
            ):
                raise ValueError("StringArray requires a sequence of strings or NaN")
            if self._ndarray.dtype != "object":
                raise ValueError(
                    "StringArray requires a sequence of strings "
                    "or NaN. Got '{self._ndarray.dtype}' dtype instead."
                )
            # TODO validate or force NA/None to NaN

    def _validate_scalar(self, value):
        # [FUNCTION DOC]
        # HELPER: validates one scalar for insertion. A missing value (any
        # missing-like) is replaced with the dtype sentinel; a non-string
        # value raises a type error naming the value, the dtype and the actual
        # type; a string is returned unchanged.
        # used by NDArrayBackedExtensionIndex.insert
        if isna(value):
            return self.dtype.na_value
        elif not isinstance(value, str):
            raise TypeError(
                f"Invalid value '{value}' for dtype '{self.dtype}'. Value should be a "
                f"string or missing value, got '{type(value).__name__}' instead."
            )
        return value

    @classmethod
    def _from_sequence(
        cls, scalars, *, dtype: Dtype | None = None, copy: bool = False
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: class-method constructor used by the generic array factory.
        # scalars is any sequence (including masked arrays and pyarrow
        # arrays); dtype (keyword-only, default none) is a string dtype or its
        # name; copy (keyword-only, default false) copies the converted data.
        # Dtype resolution: a given dtype other than the plain name "string" is
        # resolved and must be a python-storage string dtype (enforced by an
        # assertion, so other dtypes raise an assertion error); otherwise,
        # with the global string-dtype inference on, the NaN-sentinel python
        # dtype is used, else the NA-sentinel python dtype.
        # Conversion: masked arrays are converted from their data directly,
        # with masked positions set to the sentinel; other inputs (pyarrow
        # arrays first converted to numpy) are converted to an object array of
        # strings with missing-likes replaced by the sentinel. The array is
        # created without validation for speed. Returns the new array.
        if dtype and not (isinstance(dtype, str) and dtype == "string"):
            dtype = pandas_dtype(dtype)
            assert isinstance(dtype, StringDtype) and dtype.storage == "python"
        elif using_string_dtype():
            dtype = StringDtype(storage="python", na_value=np.nan)
        else:
            dtype = StringDtype(storage="python")

        from pandas.core.arrays.masked import BaseMaskedArray

        na_value = dtype.na_value
        if isinstance(scalars, BaseMaskedArray):
            # avoid costly conversion to object dtype
            na_values = scalars._mask
            result = scalars._data
            result = lib.ensure_string_array(
                result, copy=copy, convert_na_value=False, skipna=False
            )
            result[na_values] = na_value

        else:
            if lib.is_pyarrow_array(scalars):
                # pyarrow array; we cannot rely on the "to_numpy" check in
                #  ensure_string_array because calling scalars.to_numpy would set
                #  zero_copy_only to True which caused problems see GH#52076
                scalars = np.array(scalars)
            # convert non-na-likes to str, and nan-likes to StringDtype().na_value
            result = lib.ensure_string_array(scalars, na_value=na_value, copy=copy)

        # Manually creating new array avoids the validation step in the __init__, so is
        # faster. Refactor need for validation?
        new_string_array = cls.__new__(cls)
        NDArrayBacked.__init__(new_string_array, result, dtype)

        return new_string_array

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, *, dtype: ExtensionDtype, copy: bool = False
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: class-method constructor for parsed text input (readers).
        # Strings are already text, so this simply forwards to the sequence
        # constructor with the keyword-only dtype and copy flag (default
        # false).
        return cls._from_sequence(strings, dtype=dtype, copy=copy)

    def _cast_pointwise_result(self, values) -> ArrayLike:
        # [FUNCTION DOC]
        # HELPER: post-processes the result of an element-wise mapping. Applies
        # the parent conversion; if the result has a string dtype, it is cast
        # to this array's dtype so storage and sentinel are preserved. Returns
        # the array-like result.
        result = super()._cast_pointwise_result(values)
        if isinstance(result.dtype, StringDtype):
            # Ensure we retain our same na_value/storage
            result = result.astype(self.dtype)
        return result

    @classmethod
    def _empty(cls, shape, dtype) -> StringArray:
        # [FUNCTION DOC]
        # HELPER: class-method creating an array of the given shape (integer or
        # tuple) in which every element is the dtype sentinel. dtype is the
        # string dtype. Returns the array cast to that dtype without copying.
        values = np.empty(shape, dtype=object)
        values[:] = dtype.na_value
        return cls(values, dtype=dtype).astype(dtype, copy=False)

    def __arrow_array__(self, type=None):
        # [FUNCTION DOC]
        # Arrow export protocol. type (default none) is the requested arrow
        # type, defaulting to the standard string type. Copies the data,
        # replaces missing entries with none, and builds an arrow array of the
        # type. Returns the arrow array. Requires pyarrow (imported lazily).
        """
        Convert myself into a pyarrow Array.
        """
        import pyarrow as pa

        if type is None:
            type = pa.string()

        values = self._ndarray.copy()
        values[self.isna()] = None
        return pa.array(values, type=type)

    def _values_for_factorize(self) -> tuple[np.ndarray, libmissing.NAType | float]:  # type: ignore[override]
        # [FUNCTION DOC]
        # HELPER: hook used by factorize. Returns the backing object array and
        # the dtype sentinel as the value to treat as missing.
        arr = self._ndarray

        return arr, self.dtype.na_value

    def _maybe_convert_setitem_value(self, value):
        # [FUNCTION DOC]
        # HELPER: normalizes the right-hand side of element assignment. A
        # scalar: missing values become the dtype sentinel; non-strings raise a
        # type error. A non-scalar: unwrapped to an array (lists become object
        # arrays); arrays that already have this dtype's class of dtype are
        # returned unchanged; other array-likes (for example categoricals) are
        # converted to numpy. A non-empty result containing anything other than
        # strings or missing values raises a type error.
        # Returns the converted value.
        # KNOWN ISSUE: the array-case error message always names the dtype as
        # "str", whatever the actual dtype, unlike the scalar case.
        """Maybe convert value to be StringArray compatible."""
        if lib.is_scalar(value):
            if isna(value):
                value = self.dtype.na_value
            elif not isinstance(value, str):
                raise TypeError(
                    f"Invalid value '{value}' for dtype '{self.dtype}'. Value should "
                    f"be a string or missing value, got '{type(value).__name__}' "
                    "instead."
                )
        else:
            value = extract_array(value, extract_numpy=True)
            if not is_array_like(value):
                value = np.asarray(value, dtype=object)
            elif isinstance(value.dtype, type(self.dtype)):
                return value
            else:
                # cast categories and friends to arrays to see if values are
                # compatible, compatibility with arrow backed strings
                value = np.asarray(value)
            if len(value) and not lib.is_string_array(value, skipna=True):
                raise TypeError(
                    "Invalid value for dtype 'str'. Value should be a "
                    "string or missing value (or array of those)."
                )
        return value

    def __setitem__(self, key, value) -> None:
        # [FUNCTION DOC]
        # Element assignment. key is an integer, slice, mask or position array;
        # value is a string, missing value, or array-like of those.
        # Raises a value error if the array is read-only, or when a scalar key
        # is given with a non-scalar value; type errors from value
        # normalization propagate. The value is normalized, the key validated;
        # array values with this dtype are unwrapped to their backing data,
        # other arrays are converted to numpy with missing entries replaced by
        # the sentinel (on a copy). Then the parent assignment writes in place
        # (mutating this array).
        if self._readonly:
            raise ValueError("Cannot modify read-only array")

        value = self._maybe_convert_setitem_value(value)

        key = check_array_indexer(self, key)
        scalar_key = lib.is_scalar(key)
        scalar_value = lib.is_scalar(value)
        if scalar_key and not scalar_value:
            raise ValueError("setting an array element with a sequence.")

        if not scalar_value:
            if value.dtype == self.dtype:
                value = value._ndarray
            else:
                value = np.asarray(value)
                mask = isna(value)
                if mask.any():
                    value = value.copy()
                    value[isna(value)] = self.dtype.na_value

        super().__setitem__(key, value)

    def _putmask(self, mask: npt.NDArray[np.bool_], value) -> None:
        # [FUNCTION DOC]
        # HELPER: in-place conditional assignment. mask is a boolean array
        # selecting positions; value is the replacement. Uses the generic
        # extension array implementation, which goes through element assignment
        # (so missing values are handled), instead of the numpy-based
        # implementation of the parent. Mutates this array; returns nothing.
        # the super() method NDArrayBackedExtensionArray._putmask uses
        # np.putmask which doesn't properly handle None/pd.NA, so using the
        # base class implementation that uses __setitem__
        ExtensionArray._putmask(self, mask, value)

    def _where(self, mask: npt.NDArray[np.bool_], value) -> Self:
        # [FUNCTION DOC]
        # HELPER: conditional replacement returning a new array. mask is a
        # boolean array of positions to replace by value. Delegates to the
        # generic extension array implementation (based on element
        # assignment) for correct handling of missing values.
        # the super() method NDArrayBackedExtensionArray._where uses
        # np.putmask which doesn't properly handle None/pd.NA, so using the
        # base class implementation that uses __setitem__
        return ExtensionArray._where(self, mask, value)

    def isin(self, values: ArrayLike) -> npt.NDArray[np.bool_]:
        # [FUNCTION DOC]
        # Membership test. values is an array-like of candidates. Returns a
        # boolean numpy array of this array's length, true where the element
        # occurs in values. String-typed candidates are cast to this dtype;
        # other candidates that are not all strings are filtered down to the
        # strings and missing values (an all-removed candidate set returns all
        # false) and then converted to this array type; the comparison uses the
        # generic hashing-based membership routine on the object data.
        if isinstance(values, BaseStringArray) or (
            isinstance(values, ExtensionArray) and is_string_dtype(values.dtype)
        ):
            values = values.astype(self.dtype, copy=False)
        else:
            if not lib.is_string_array(np.asarray(values), skipna=True):
                values = np.array(
                    [val for val in values if isinstance(val, str) or isna(val)],
                    dtype=object,
                )
                if not len(values):
                    return np.zeros(self.shape, dtype=bool)

            values = self._from_sequence(values, dtype=self.dtype)

        return isin(np.asarray(self), np.asarray(values))

    def astype(self, dtype, copy: bool = True):
        # [FUNCTION DOC]
        # Casts to another dtype. dtype is anything the dtype resolver accepts;
        # copy (default true) is honored only for the same-dtype case and
        # delegated elsewhere. Same dtype: copy or self. Nullable integer
        # target: missing entries are set to zero in a copy, the data is cast
        # via numpy and wrapped with the missing mask (so the text must be
        # parseable, otherwise numpy raises). Nullable floating target: a copy
        # of this array has missing entries set to the text zero, is cast to
        # the numpy float dtype and wrapped with the mask. Any other extension
        # dtype: uses the generic extension array cast, skipping the numpy
        # extension array implementation. Numpy float target: missing entries
        # are zeroed, the object data cast, then missing positions set to NaN.
        # Everything else falls to the parent implementation.
        dtype = pandas_dtype(dtype)

        if dtype == self.dtype:
            if copy:
                return self.copy()
            return self

        elif isinstance(dtype, IntegerDtype):
            arr = self._ndarray.copy()
            mask = self.isna()
            arr[mask] = 0
            values = arr.astype(dtype.numpy_dtype)
            return IntegerArray(values, mask, copy=False)
        elif isinstance(dtype, FloatingDtype):
            arr_ea = self.copy()
            mask = self.isna()
            arr_ea[mask] = "0"
            values = arr_ea.astype(dtype.numpy_dtype)
            return FloatingArray(values, mask, copy=False)
        elif isinstance(dtype, ExtensionDtype):
            # Skip the NumpyExtensionArray.astype method
            return ExtensionArray.astype(self, dtype, copy)
        elif np.issubdtype(dtype, np.floating):
            arr = self._ndarray.copy()
            mask = self.isna()
            arr[mask] = 0
            values = arr.astype(dtype)
            values[mask] = np.nan
            return values

        return super().astype(dtype, copy)

    def _reduce(
        self,
        name: str,
        *,
        skipna: bool = True,
        keepdims: bool = False,
        axis: AxisInt | None = 0,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Generic reduction entry point. name is the reduction name; skipna
        # (keyword-only, default true) skips missing values; keepdims (default
        # false) returns a one-element array instead of a scalar; axis (default
        # zero) and extra keywords are passed to the named method.
        # Supported: any and all for the NaN-sentinel dtype (nan-aware
        # reduction on the object data, ignoring keepdims and axis); minimum,
        # maximum, positions of minimum and maximum, and sum for both
        # sentinels. Anything else raises a type error naming the reduction.
        if self.dtype.na_value is np.nan and name in ["any", "all"]:
            if name == "any":
                return nanops.nanany(self._ndarray, skipna=skipna)
            else:
                return nanops.nanall(self._ndarray, skipna=skipna)

        if name in ["min", "max", "argmin", "argmax", "sum"]:
            result = getattr(self, name)(skipna=skipna, axis=axis, **kwargs)
            if keepdims:
                return self._from_sequence([result], dtype=self.dtype)
            return result

        raise TypeError(f"Cannot perform reduction '{name}' with string dtype")

    def _accumulate(self, name: str, *, skipna: bool = True, **kwargs) -> StringArray:
        # [FUNCTION DOC]
        # Cumulative operation along the array. name is cumsum, cummin or
        # cummax (cumprod raises a type error because products of strings are
        # undefined; any other name fails with a lookup error when selecting
        # the numpy function); skipna (keyword-only, default true) controls
        # missing handling; extra keywords are ignored.
        # KNOWN ISSUE: the docstring says it raises a not-implemented error for
        # subclasses without accumulations, but the code raises a type error for
        # the unsupported product operation and nothing else; it also lists the
        # cumulative product among supported names.
        # Behavior: all-missing data is returned unchanged. With skipna and
        # cumsum, missing entries are replaced by empty text before
        # accumulating and restored as missing in the output; with skipna and
        # min/max, missing entries are filled forward then backward so the
        # running value carries over, and restored as missing afterwards.
        # Without skipna, everything from the first missing entry onward is the
        # sentinel. Returns a new array of this class and dtype.
        """
        Return an ExtensionArray performing an accumulation operation.

        The underlying data type might change.

        Parameters
        ----------
        name : str
            Name of the function, supported values are:
            - cummin
            - cummax
            - cumsum
            - cumprod
        skipna : bool, default True
            If True, skip NA values.
        **kwargs
            Additional keyword arguments passed to the accumulation function.
            Currently, there is no supported kwarg.

        Returns
        -------
        array

        Raises
        ------
        NotImplementedError : subclass does not define accumulations
        """
        if name == "cumprod":
            msg = f"operation '{name}' not supported for dtype '{self.dtype}'"
            raise TypeError(msg)

        # We may need to strip out trailing NA values
        tail: np.ndarray | None = None
        na_mask: np.ndarray | None = None
        ndarray = self._ndarray
        np_func = {
            "cumsum": np.cumsum,
            "cummin": np.minimum.accumulate,
            "cummax": np.maximum.accumulate,
        }[name]

        if self._hasna:
            na_mask = cast("npt.NDArray[np.bool_]", isna(ndarray))
            if np.all(na_mask):
                return type(self)(ndarray, dtype=self.dtype)
            if skipna:
                if name == "cumsum":
                    ndarray = np.where(na_mask, "", ndarray)
                else:
                    # We can retain the running min/max by forward/backward filling.
                    ndarray = ndarray.copy()
                    missing.pad_or_backfill_inplace(
                        ndarray,
                        method="pad",
                        axis=0,
                    )
                    missing.pad_or_backfill_inplace(
                        ndarray,
                        method="backfill",
                        axis=0,
                    )
            else:
                # When not skipping NA values, the result should be null from
                # the first NA value onward.
                idx = np.argmax(na_mask)
                tail = np.empty(len(ndarray) - idx, dtype="object")
                tail[:] = self.dtype.na_value
                ndarray = ndarray[:idx]

        # mypy: Cannot call function of unknown type
        np_result = np_func(ndarray)  # type: ignore[operator]

        if tail is not None:
            np_result = np.hstack((np_result, tail))
        elif na_mask is not None:
            # Argument 2 to "where" has incompatible type "NAType | float"
            np_result = np.where(na_mask, self.dtype.na_value, np_result)  # type: ignore[arg-type]

        result = type(self)(np_result, dtype=self.dtype)
        return result

    def _wrap_reduction_result(self, axis: AxisInt | None, result) -> Any:
        # [FUNCTION DOC]
        # HELPER: finalizes a reduction. axis is the reduced axis, result the
        # raw result. For the NaN-sentinel dtype a result equal to the NA
        # singleton (produced by the masked reductions) becomes NaN; otherwise
        # defers to the parent wrapping.
        if self.dtype.na_value is np.nan and result is libmissing.NA:
            # the masked_reductions use pd.NA -> convert to np.nan
            return np.nan
        return super()._wrap_reduction_result(axis, result)

    def min(self, axis=None, skipna: bool = True, **kwargs) -> Scalar:
        # [FUNCTION DOC]
        # Minimum string. axis (default none) and skipna (default true);
        # extra keywords only validated for numpy compatibility. Uses the
        # masked reduction on the converted data and the missing mask; returns
        # the smallest string, or the missing value if empty/all missing (or
        # when skipna is false and any is missing).
        nv.validate_min((), kwargs)
        result = masked_reductions.min(
            values=self.to_numpy(), mask=self.isna(), skipna=skipna
        )
        return self._wrap_reduction_result(axis, result)

    def max(self, axis=None, skipna: bool = True, **kwargs) -> Scalar:
        # [FUNCTION DOC]
        # Maximum string; same parameters and behavior as the minimum.
        nv.validate_max((), kwargs)
        result = masked_reductions.max(
            values=self.to_numpy(), mask=self.isna(), skipna=skipna
        )
        return self._wrap_reduction_result(axis, result)

    def sum(
        self,
        *,
        axis: AxisInt | None = None,
        skipna: bool = True,
        min_count: int = 0,
        **kwargs,
    ) -> Scalar:
        # [FUNCTION DOC]
        # Concatenation of all strings. Keyword-only axis (default none), skipna
        # (default true), min_count (default zero; minimum number of valid
        # values required); extra keywords only validated. Uses the masked
        # sum with an empty-text initial value. Returns the joined text or the
        # missing value.
        nv.validate_sum((), kwargs)
        result = masked_reductions.sum(
            values=self._ndarray,
            mask=self.isna(),
            skipna=skipna,
            min_count=min_count,
            initial="",
        )
        return self._wrap_reduction_result(axis, result)

    def value_counts(self, dropna: bool = True) -> Series:
        # [FUNCTION DOC]
        # Counts occurrences of each distinct value. dropna (default true)
        # excludes missing values. Returns a series of counts indexed by value;
        # for the NA-sentinel dtype the counts are cast to the nullable 64-bit
        # integer dtype, otherwise the parent dtype is kept.
        result = super().value_counts(dropna=dropna)

        if self.dtype.na_value is libmissing.NA:
            result = result.astype("Int64")
        return result

    def memory_usage(self, deep: bool = False) -> int:
        # [FUNCTION DOC]
        # Returns the number of bytes used. Shallow: the size of the pointer
        # array. deep (default false): additionally adds the memory of the
        # string objects themselves.
        result = self._ndarray.nbytes
        if deep:
            return result + lib.memory_usage_of_objects(self._ndarray)
        return result

    def searchsorted(
        self,
        value: NumpyValueArrayLike | ExtensionArray,
        side: Literal["left", "right"] = "left",
        sorter: NumpySorter | None = None,
    ) -> npt.NDArray[np.intp] | np.intp:
        # [FUNCTION DOC]
        # Binary search for insertion points. value is a scalar or array-like
        # to insert; side is left or right (default left); sorter (default none)
        # is an integer array of positions that sorts this array.
        # Raises a value error if the first or last element (in sorted order,
        # via the sorter if given) is missing, because a sorted array with
        # missing entries is impossible; only the ends are checked for speed.
        # Otherwise delegates to the parent search and returns an integer array
        # or single integer.
        # KNOWN ISSUE: the docstring example uses numeric data, not strings.
        """
        Find indices where elements should be inserted to maintain order.

        Find the indices into a sorted array `self` (a) such that, if the
        corresponding elements in `value` were inserted before the indices,
        the order of `self` would be preserved.

        Assuming that `self` is sorted:

        ======  ================================
        `side`  returned index `i` satisfies
        ======  ================================
        left    ``self[i-1] < value <= self[i]``
        right   ``self[i-1] <= value < self[i]``
        ======  ================================

        Parameters
        ----------
        value : array-like, list or scalar
            Value(s) to insert into `self`.
        side : {'left', 'right'}, optional
            If 'left', the index of the first suitable location found is given.
            If 'right', return the last such index.  If there is no suitable
            index, return either 0 or N (where N is the length of `self`).
        sorter : 1-D array-like, optional
            Optional array of integer indices that sort array a into ascending
            order. They are typically the result of argsort.

        Returns
        -------
        array of ints or int
            If value is array-like, array of insertion points.
            If value is scalar, a single integer.

        See Also
        --------
        numpy.searchsorted : Similar method from NumPy.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3, 5])
        >>> arr.searchsorted([4])
        array([3])
        """

        # GH#65837: avoid O(n) scan; NA confined to array ends in sorted data.
        # When sorter is given, the sorted order is ndarray[sorter], so check
        # the first/last positions via sorter instead of raw ndarray positions.
        ndarray = self._ndarray
        if len(ndarray):
            if sorter is None:
                has_na = libmissing.checknull(ndarray[0]) or libmissing.checknull(
                    ndarray[-1]
                )
            else:
                has_na = libmissing.checknull(
                    ndarray[sorter[0]]
                ) or libmissing.checknull(ndarray[sorter[-1]])
        else:
            has_na = False
        if has_na:
            raise ValueError(
                "searchsorted requires array to be sorted, which is impossible "
                "with NAs present."
            )
        return super().searchsorted(value=value, side=side, sorter=sorter)

    def _cmp_method(self, other, op):
        # [FUNCTION DOC]
        # HELPER: shared implementation of comparison and arithmetic operators
        # (also bound as the arithmetic method). other is a scalar, array-like
        # or array; op is the operator function.
        # Dispatch rules: if other is a string array with NA sentinel while this
        # one has NaN, this array is cast to other's dtype and the operation
        # repeated. If other is arrow-backed, the not-implemented marker is
        # returned (letting the arrow side handle it) except when other is a
        # string array and this array has NA semantics but other does not.
        # A python string array operand is unwrapped to its data.
        # Missing positions are the union of both operands' missing; list-like
        # operands must have equal length (value error otherwise) and are
        # filtered to valid positions. Multiplying by boolean values raises a
        # type error. Arithmetic operators produce an object array with the
        # sentinel at missing positions; if all entries end up strings the
        # result is wrapped in this array class, else the raw object array is
        # returned. Comparison operators produce a nullable boolean array
        # (missing where either side is missing); for the NaN-sentinel dtype
        # this becomes a plain boolean numpy array where missing counts as
        # true for not-equal and false for all other comparisons.
        from pandas.arrays import (
            ArrowExtensionArray,
            BooleanArray,
        )

        if (
            isinstance(other, BaseStringArray)
            and self.dtype.na_value is not libmissing.NA
            and other.dtype.na_value is libmissing.NA
        ):
            # NA has priority of NaN semantics
            return op(self.astype(other.dtype, copy=False), other)

        if isinstance(other, ArrowExtensionArray):
            if isinstance(other, BaseStringArray):
                # pyarrow storage has priority over python storage
                # (except if we have NA semantics and other not)
                if not (
                    self.dtype.na_value is libmissing.NA
                    and other.dtype.na_value is not libmissing.NA
                ):
                    return NotImplemented
            else:
                return NotImplemented

        if isinstance(other, StringArray):
            other = other._ndarray

        mask = isna(self) | isna(other)
        valid = ~mask

        if lib.is_list_like(other):
            if len(other) != len(self):
                # prevent improper broadcasting when other is 2D
                raise ValueError(
                    f"Lengths of operands do not match: {len(self)} != {len(other)}"
                )

            # for array-likes, first filter out NAs before converting to numpy
            if not is_array_like(other):
                other = np.asarray(other)
            other = other[valid]

        other_dtype = getattr(other, "dtype", None)
        if op.__name__.strip("_") in ["mul", "rmul"] and (
            lib.is_bool(other) or lib.is_np_dtype(other_dtype, "b")
        ):
            # GH#62595
            raise TypeError(
                "Cannot multiply StringArray by bools. "
                "Explicitly cast to integers instead."
            )

        if op.__name__ in ops.ARITHMETIC_BINOPS:
            result = np.empty_like(self._ndarray, dtype="object")
            result[mask] = self.dtype.na_value
            result[valid] = op(self._ndarray[valid], other)
            if not lib.is_string_array(result, skipna=True):
                return result
            return self._from_backing_data(result)
        else:
            # logical
            result = np.zeros(len(self._ndarray), dtype="bool")
            result[valid] = op(self._ndarray[valid], other)
            res_arr = BooleanArray(result, mask)
            if self.dtype.na_value is np.nan:
                if op == operator.ne:
                    return res_arr.to_numpy(np.bool_, na_value=True)
                else:
                    return res_arr.to_numpy(np.bool_, na_value=False)
            return res_arr

    _arith_method = _cmp_method

    def _str_zfill(self, width: int) -> Self:
        # [FUNCTION DOC]
        # HELPER: backs the string-accessor zero-padding method. width is the
        # minimum total length; each non-missing string is left-padded with
        # zeros (keeping a leading sign in front) by mapping over elements via
        # the shared mapping helper. Returns the mapped array.
        return self._str_map(lambda x: x.zfill(width))
