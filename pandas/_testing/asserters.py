# [MODULE DOC]
# HELPER: test support module containing the equality assertion functions of
# pandas' testing utilities. All raise assertion errors with a detailed
# message on mismatch. Public contents (re-exported by the testing package;
# the index, series, frame and extension array ones are also exposed publicly
# with a patched module name): assert_almost_equal, assert_dict_equal,
# assert_index_equal, assert_class_equal, assert_attr_equal, assert_is_sorted,
# assert_categorical_equal, assert_interval_array_equal,
# assert_period_array_equal, assert_datetime_array_equal,
# assert_timedelta_array_equal, raise_assert_detail,
# assert_numpy_array_equal, assert_extension_array_equal, assert_series_equal,
# assert_frame_equal, assert_equal (dispatcher), assert_sp_array_equal,
# assert_contains_all, assert_copy, assert_indexing_slices_equivalent and
# assert_metadata_equivalent. Private helpers: _check_isinstance and
# is_extension_array_dtype_and_needs_i8_conversion.
# Relationships: low-level value comparison is delegated to the compiled
# testing extension (aliased here as _testing); frame comparison is built on the
# series comparison, which is built on the index and array comparisons.
# Several functions hide their frames from tracebacks. Import-time side
# effects: none beyond importing pandas.
from __future__ import annotations

import operator
from typing import (
    TYPE_CHECKING,
    Literal,
    NoReturn,
    cast,
)
import warnings

import numpy as np

from pandas._libs import lib
from pandas._libs.missing import is_matching_na
from pandas._libs.sparse import SparseIndex
import pandas._libs.testing as _testing
from pandas._libs.tslibs.np_datetime import compare_mismatched_resolutions
from pandas.errors import Pandas4Warning
from pandas.util._decorators import (
    deprecate_kwarg,
    set_module,
)

from pandas.core.dtypes.common import (
    is_bool,
    is_float_dtype,
    is_integer_dtype,
    is_number,
    is_numeric_dtype,
    needs_i8_conversion,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    ExtensionDtype,
    NumpyEADtype,
)
from pandas.core.dtypes.missing import array_equivalent

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    DatetimeIndex,
    Index,
    IntervalDtype,
    IntervalIndex,
    MultiIndex,
    PeriodIndex,
    RangeIndex,
    Series,
    TimedeltaIndex,
)
from pandas.core.arrays import (
    DatetimeArray,
    ExtensionArray,
    IntervalArray,
    PeriodArray,
    TimedeltaArray,
)
from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin
from pandas.core.arrays.string_ import StringDtype
from pandas.core.indexes.api import safe_sort_index

from pandas.io.formats.printing import pprint_thing

if TYPE_CHECKING:
    from pandas._typing import DtypeObj


def assert_almost_equal(
    left,
    right,
    check_dtype: bool | Literal["equiv"] = "equiv",
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    **kwargs,
) -> None:
    # [FUNCTION DOC]
    # HELPER: test assertion. Compares two objects for approximate equality and
    # raises an assertion error if they differ. Parameters: left and right (any
    # objects); check_dtype (true, false or the word equiv, default equiv);
    # rtol (relative tolerance, default 1e-5); atol (absolute tolerance, default
    # 1e-8); extra keyword arguments are passed to the underlying comparison.
    # Dispatch on the type of left: for an index, the index assertion is called
    # with exactness disabled, the dtype setting used as the exactness level; for
    # a series or a frame, the respective assertion with exactness disabled;
    # otherwise (scalars, arrays, sequences): if the dtype check is on, first
    # asserts the classes match (the object label is numpy array when either is
    # an array, else input), except that two numbers, or two booleans, of any
    # classes are not compared by class; then calls the compiled low-level
    # comparison with the dtype flag converted to a boolean.
    """
    Check that the left and right objects are approximately equal.

    By approximately equal, we refer to objects that are numbers or that
    contain numbers which may be equivalent to specific levels of precision.

    Parameters
    ----------
    left : object
    right : object
    check_dtype : bool or {'equiv'}, default 'equiv'
        Check dtype if both a and b are the same type. If 'equiv' is passed in,
        then `RangeIndex` and `Index` with int64 dtype are also considered
        equivalent when doing type checking.
    rtol : float, default 1e-5
        Relative tolerance.
    atol : float, default 1e-8
        Absolute tolerance.
    """
    if isinstance(left, Index):
        assert_index_equal(
            left,
            right,
            check_exact=False,
            exact=check_dtype,
            rtol=rtol,
            atol=atol,
            **kwargs,
        )

    elif isinstance(left, Series):
        assert_series_equal(
            left,
            right,
            check_exact=False,
            check_dtype=check_dtype,
            rtol=rtol,
            atol=atol,
            **kwargs,
        )

    elif isinstance(left, DataFrame):
        assert_frame_equal(
            left,
            right,
            check_exact=False,
            check_dtype=check_dtype,
            rtol=rtol,
            atol=atol,
            **kwargs,
        )

    else:
        # Other sequences.
        if check_dtype:
            if is_number(left) and is_number(right):
                # Do not compare numeric classes, like np.float64 and float.
                pass
            elif is_bool(left) and is_bool(right):
                # Do not compare bool classes, like np.bool_ and bool.
                pass
            else:
                if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
                    obj = "numpy array"
                else:
                    obj = "Input"
                assert_class_equal(left, right, obj=obj)

        # if we have "equiv", this becomes True
        _testing.assert_almost_equal(
            left, right, check_dtype=bool(check_dtype), rtol=rtol, atol=atol, **kwargs
        )


def _check_isinstance(left, right, cls) -> None:
    # [FUNCTION DOC]
    # HELPER: for the assert functions. Parameters: left and right (objects to
    # check) and cls (the required class). Raises an assertion error naming the
    # class and the actual type if left, checked first, and then right, is not
    # an instance of cls (subclasses are accepted). Returns nothing.
    """
    Helper method for our assert_* methods that ensures that
    the two objects being compared have the right type before
    proceeding with the comparison.

    Parameters
    ----------
    left : The first object being compared.
    right : The second object being compared.
    cls : The class type to check against.

    Raises
    ------
    AssertionError : Either `left` or `right` is not an instance of `cls`.
    """
    cls_name = cls.__name__

    if not isinstance(left, cls):
        raise AssertionError(
            f"{cls_name} Expected type {cls}, found {type(left)} instead"
        )
    if not isinstance(right, cls):
        raise AssertionError(
            f"{cls_name} Expected type {cls}, found {type(right)} instead"
        )


def assert_dict_equal(left, right, compare_keys: bool = True) -> None:
    # [FUNCTION DOC]
    # HELPER: test assertion. Parameters: left and right (must both be
    # dictionaries, otherwise an assertion error about the type is raised) and
    # compare_keys (default true; whether the key sets must match). Delegates the
    # comparison to the compiled testing helper. No docstring exists for this
    # function.
    _check_isinstance(left, right, dict)
    _testing.assert_dict_equal(left, right, compare_keys=compare_keys)


@set_module("pandas.testing")
def assert_index_equal(
    left: Index,
    right: Index,
    exact: bool | str = "equiv",
    check_names: bool = True,
    check_exact: bool = True,
    check_categorical: bool = True,
    check_order: bool = True,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    obj: str | None = None,
) -> None:
    # [FUNCTION DOC]
    # HELPER: test assertion (public as a testing function). Compares two
    # indexes. Parameters: left and right (indexes); exact (true, false or
    # equiv; default equiv: whether classes, dtypes and inferred types must be
    # identical, with a range index equivalent to a 64-bit integer index under
    # equiv); check_names (compare the names attribute, default true);
    # check_exact (compare numbers exactly, default true); check_categorical
    # (compare categorical content, default true); check_order (default true;
    # if false both are first sorted with a safe sort); rtol and atol
    # (tolerances used only when not exact); obj (label used in messages;
    # defaults to the multi-index or index label by type of left).
    # Steps: verify both are indexes; check types (class equality, inferred type,
    # and dtype, with special handling of categorical dtypes whose categories are
    # compared recursively unless check_categorical is false; skipped entirely
    # when exact is false); compare number of levels, then lengths; for a
    # multi-index compare each level (the levels and codes first, falling back to
    # level values on failure) and then the level types; otherwise, when both
    # exact and categorical checks are on, require equality and on mismatch
    # build an error giving the percentage of differing values (an error is
    # raised if the values are not comparable); otherwise compare approximately
    # by values. Finally compare names if requested; the dtype for period
    # indexes; interval content for interval indexes; and categorical content
    # when requested. Raises assertion errors via the shared detail raiser.
    """
    Check that left and right Index are equal.

    Parameters
    ----------
    left : Index
        The first index to compare.
    right : Index
        The second index to compare.
    exact : bool or {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical. If 'equiv', then RangeIndex can be substituted for
        Index with an int64 dtype as well.
    check_names : bool, default True
        Whether to check the names attribute.
    check_exact : bool, default True
        Whether to compare number exactly.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    check_order : bool, default True
        Whether to compare the order of index entries as well as their values.
        If True, both indexes must contain the same elements, in the same order.
        If False, both indexes must contain the same elements, but in any order.
    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.
    obj : str, default 'Index' or 'MultiIndex'
        Specify object name being compared, internally used to show appropriate
        assertion message.

    See Also
    --------
    testing.assert_series_equal : Check that two Series are equal.
    testing.assert_frame_equal : Check that two DataFrames are equal.

    Examples
    --------
    >>> from pandas import testing as tm
    >>> a = pd.Index([1, 2, 3])
    >>> b = pd.Index([1, 2, 3])
    >>> tm.assert_index_equal(a, b)
    """
    __tracebackhide__ = True

    if obj is None:
        obj = "MultiIndex" if isinstance(left, MultiIndex) else "Index"

    def _check_types(left, right, obj: str = "Index") -> None:
        # [FUNCTION DOC]
        # HELPER: for the index assertion. Parameters: left, right (indexes) and
        # obj (label, default Index). Does nothing when exactness is disabled in
        # the enclosing call. Otherwise asserts class equality (honoring the
        # equiv mode) and equal inferred type; if both have categorical dtype,
        # when categorical checking is on asserts the dtypes equal and the
        # categories are equal (recursive call), then returns without a further
        # dtype check; in all other cases asserts the dtypes are equal.
        if not exact:
            return

        assert_class_equal(left, right, exact=exact, obj=obj)
        assert_attr_equal("inferred_type", left, right, obj=obj)

        # Skip exact dtype checking when `check_categorical` is False
        if isinstance(left.dtype, CategoricalDtype) and isinstance(
            right.dtype, CategoricalDtype
        ):
            if check_categorical:
                assert_attr_equal("dtype", left, right, obj=obj)
                assert_index_equal(left.categories, right.categories, exact=exact)
            return

        assert_attr_equal("dtype", left, right, obj=obj)

    # instance validation
    _check_isinstance(left, right, Index)

    # class / dtype comparison
    _check_types(left, right, obj=obj)

    # level comparison
    if left.nlevels != right.nlevels:
        msg1 = f"{obj} levels are different"
        msg2 = f"{left.nlevels}, {left}"
        msg3 = f"{right.nlevels}, {right}"
        raise_assert_detail(obj, msg1, msg2, msg3)

    # length comparison
    if len(left) != len(right):
        msg1 = f"{obj} length are different"
        msg2 = f"{len(left)}, {left}"
        msg3 = f"{len(right)}, {right}"
        raise_assert_detail(obj, msg1, msg2, msg3)

    # If order doesn't matter then sort the index entries
    if not check_order:
        left = safe_sort_index(left)
        right = safe_sort_index(right)

    # MultiIndex special comparison for little-friendly error messages
    if isinstance(left, MultiIndex):
        right = cast(MultiIndex, right)

        for level in range(left.nlevels):
            lobj = f"{obj} level [{level}]"
            try:
                # try comparison on levels/codes to avoid densifying MultiIndex
                assert_index_equal(
                    left.levels[level],
                    right.levels[level],
                    exact=exact,
                    check_names=check_names,
                    check_exact=check_exact,
                    check_categorical=check_categorical,
                    rtol=rtol,
                    atol=atol,
                    obj=lobj,
                )
                assert_numpy_array_equal(left.codes[level], right.codes[level])
            except AssertionError:
                llevel = left.get_level_values(level)
                rlevel = right.get_level_values(level)

                assert_index_equal(
                    llevel,
                    rlevel,
                    exact=exact,
                    check_names=check_names,
                    check_exact=check_exact,
                    check_categorical=check_categorical,
                    rtol=rtol,
                    atol=atol,
                    obj=lobj,
                )
            # get_level_values may change dtype
            _check_types(left.levels[level], right.levels[level], obj=lobj)

    # skip exact index checking when `check_categorical` is False
    elif check_exact and check_categorical:
        if not left.equals(right):
            # _values compare can raise TypeError (non-comparable
            # categoricals (GH#61935)
            try:
                mismatch = left._values != right._values
            except TypeError:
                raise_assert_detail(
                    obj,
                    "types are not comparable (non-matching categorical categories)",
                    left,
                    right,
                )

            if not isinstance(mismatch, np.ndarray):
                mismatch = cast("ExtensionArray", mismatch).fillna(True)

            diff = np.sum(mismatch.astype(int)) * 100.0 / len(left)
            msg = f"{obj} values are different ({np.round(diff, 5)} %)"
            raise_assert_detail(obj, msg, left, right)
    else:
        # if we have "equiv", this becomes True
        exact_bool = bool(exact)
        _testing.assert_almost_equal(
            left.values,
            right.values,
            rtol=rtol,
            atol=atol,
            check_dtype=exact_bool,
            obj=obj,
            lobj=left,
            robj=right,
        )

    # metadata comparison
    if check_names:
        assert_attr_equal("names", left, right, obj=obj)
    if isinstance(left, PeriodIndex) or isinstance(right, PeriodIndex):
        assert_attr_equal("dtype", left, right, obj=obj)
    if isinstance(left, IntervalIndex) or isinstance(right, IntervalIndex):
        assert_interval_array_equal(left._values, right._values)

    if check_categorical:
        if isinstance(left.dtype, CategoricalDtype) or isinstance(
            right.dtype, CategoricalDtype
        ):
            assert_categorical_equal(left._values, right._values, obj=f"{obj} category")


def assert_class_equal(
    left, right, exact: bool | str = True, obj: str = "Input"
) -> None:
    # [FUNCTION DOC]
    # HELPER: test assertion. Parameters: left and right (objects), exact (true or
    # the word equiv; default true) and obj (label, default Input). Returns
    # normally if the exact types are identical. In equiv mode, a plain index
    # class and a range index (or subclass) are also treated as equal to each
    # other. Otherwise raises an assertion error via the detail raiser, showing
    # the index itself (for indexes, to include values) or the class name.
    # Any value of exact other than the equiv word behaves like true.
    """
    Checks classes are equal.
    """
    __tracebackhide__ = True

    def repr_class(x):
        # [FUNCTION DOC]
        # HELPER: for the class assertion. Returns the object itself if it is an
        # index (so its values appear in the message), otherwise the name of its
        # class.
        if isinstance(x, Index):
            # return Index as it is to include values in the error message
            return x

        return type(x).__name__

    def is_class_equiv(idx: Index) -> bool:
        # [FUNCTION DOC]
        # HELPER: for the class assertion in equiv mode. Returns true if idx is
        # exactly the plain index class or is an instance of the range index
        # class (including subclasses). Class equivalence only; the integer
        # dtype is checked elsewhere.
        """Classes that are a RangeIndex (sub-)instance or exactly an `Index` .

        This only checks class equivalence. There is a separate check that the
        dtype is int64.
        """
        return type(idx) is Index or isinstance(idx, RangeIndex)

    if type(left) == type(right):
        return

    if exact == "equiv":
        if is_class_equiv(left) and is_class_equiv(right):
            return

    msg = f"{obj} classes are different"
    raise_assert_detail(obj, msg, repr_class(left), repr_class(right))


def assert_attr_equal(attr: str, left, right, obj: str = "Attributes") -> None:
    # [FUNCTION DOC]
    # HELPER: test assertion. Parameters: attr (attribute name), left and right
    # (objects that must both have the attribute; a missing one raises an
    # attribute error) and obj (label, default Attributes). Passes if the two
    # attribute values are the same object or are matching missing values (both
    # NaN, both not-a-time, both the missing marker, and so on). Otherwise
    # compares with equality; a type error during comparison counts as unequal; if
    # exactly one side is the missing marker the result is unequal; a
    # non-boolean comparison result (array-like) is reduced with an all test.
    # Raises an assertion error stating the attribute differs, with both values.
    """
    Check attributes are equal. Both objects must have attribute.

    Parameters
    ----------
    attr : str
        Attribute name being compared.
    left : object
    right : object
    obj : str, default 'Attributes'
        Specify object name being compared, internally used to show appropriate
        assertion message
    """
    __tracebackhide__ = True

    left_attr = getattr(left, attr)
    right_attr = getattr(right, attr)

    if left_attr is right_attr or is_matching_na(left_attr, right_attr):
        # e.g. both np.nan, both NaT, both pd.NA, ...
        return None

    try:
        result = left_attr == right_attr
    except TypeError:
        # datetimetz on rhs may raise TypeError
        result = False
    if (left_attr is pd.NA) ^ (right_attr is pd.NA):
        result = False
    elif not isinstance(result, bool):
        result = result.all()

    if not result:
        msg = f'Attribute "{attr}" are different'
        raise_assert_detail(obj, msg, left_attr, right_attr)
    return None


def assert_is_sorted(seq) -> None:
    # [FUNCTION DOC]
    # HELPER: test assertion. Parameter seq is an index, series, numpy array or
    # extension array. Index and series are reduced to their values. A numpy
    # array is compared to a sorted copy using the numpy array assertion; any
    # other array is compared to itself reordered by its sort order using the
    # extension array assertion. Raises an assertion error if not sorted
    # ascending.
    """Assert that the sequence is sorted."""
    if isinstance(seq, (Index, Series)):
        seq = seq.values
    # sorting does not change precisions
    if isinstance(seq, np.ndarray):
        assert_numpy_array_equal(seq, np.sort(np.array(seq)))
    else:
        assert_extension_array_equal(seq, seq[seq.argsort()])


def assert_categorical_equal(
    left,
    right,
    check_dtype: bool = True,
    check_category_order: bool = True,
    obj: str = "Categorical",
) -> None:
    # [FUNCTION DOC]
    # HELPER: test assertion. Parameters: left and right (must be categoricals,
    # else an assertion error), check_dtype (compare the integer dtype of the
    # codes, default true), check_category_order (default true) and obj (label).
    # The categories comparison mode is equiv if either categories is a range
    # index, else exact. With category order checked: the categories must be
    # equal in order and the codes arrays must be equal. Otherwise: categories
    # are compared after sorting (unsorted if sorting fails with a type error)
    # and the materialized values (categories taken by codes) are compared. In
    # both cases the ordered flag is compared at the end.
    """
    Test that Categoricals are equivalent.

    Parameters
    ----------
    left : Categorical
    right : Categorical
    check_dtype : bool, default True
        Check that integer dtype of the codes are the same.
    check_category_order : bool, default True
        Whether the order of the categories should be compared, which
        implies identical integer codes.  If False, only the resulting
        values are compared.  The ordered attribute is
        checked regardless.
    obj : str, default 'Categorical'
        Specify object name being compared, internally used to show appropriate
        assertion message.
    """
    _check_isinstance(left, right, Categorical)

    exact: bool | str
    if isinstance(left.categories, RangeIndex) or isinstance(
        right.categories, RangeIndex
    ):
        exact = "equiv"
    else:
        # We still want to require exact matches for Index
        exact = True

    if check_category_order:
        assert_index_equal(
            left.categories, right.categories, obj=f"{obj}.categories", exact=exact
        )
        assert_numpy_array_equal(
            left.codes, right.codes, check_dtype=check_dtype, obj=f"{obj}.codes"
        )
    else:
        try:
            lc = left.categories.sort_values()
            rc = right.categories.sort_values()
        except TypeError:
            # e.g. '<' not supported between instances of 'int' and 'str'
            lc, rc = left.categories, right.categories
        assert_index_equal(lc, rc, obj=f"{obj}.categories", exact=exact)
        assert_index_equal(
            left.categories.take(left.codes),
            right.categories.take(right.codes),
            obj=f"{obj}.values",
            exact=exact,
        )

    assert_attr_equal("ordered", left, right, obj=obj)


def assert_interval_array_equal(
    left, right, exact: bool | Literal["equiv"] = "equiv", obj: str = "IntervalArray"
) -> None:
    # [FUNCTION DOC]
    # HELPER: test assertion. Parameters: left and right (must be interval
    # arrays, else an assertion error), exact (documented as an index exactness
    # level; default equiv) and obj (label). Compares the left endpoint arrays,
    # then the right endpoint arrays, using the generic dispatcher (with
    # frequency checking turned off when the endpoints are datetimes or
    # timedeltas), then the closed-side attribute.
    # KNOWN ISSUE: the parameter exact is accepted and documented but never used.
    """
    Test that two IntervalArrays are equivalent.

    Parameters
    ----------
    left, right : IntervalArray
        The IntervalArrays to compare.
    exact : bool or {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical. If 'equiv', then RangeIndex can be substituted for
        Index with an int64 dtype as well.
    obj : str, default 'IntervalArray'
        Specify object name being compared, internally used to show appropriate
        assertion message
    """
    _check_isinstance(left, right, IntervalArray)

    kwargs = {}
    if left._left.dtype.kind in "mM":
        # We have a DatetimeArray or TimedeltaArray
        kwargs["check_freq"] = False

    assert_equal(left._left, right._left, obj=f"{obj}.left", **kwargs)
    assert_equal(left._right, right._right, obj=f"{obj}.right", **kwargs)

    assert_attr_equal("closed", left, right, obj=obj)


def assert_period_array_equal(left, right, obj: str = "PeriodArray") -> None:
    # [FUNCTION DOC]
    # HELPER: test assertion. Parameters: left and right (must be period
    # arrays) and obj (label). Compares the underlying integer ordinals using the
    # numpy array assertion, then the dtype attribute (which carries the
    # frequency). No docstring exists for this function.
    _check_isinstance(left, right, PeriodArray)

    assert_numpy_array_equal(left._ndarray, right._ndarray, obj=f"{obj}._ndarray")
    assert_attr_equal("dtype", left, right, obj=obj)


def assert_datetime_array_equal(
    left, right, obj: str = "DatetimeArray", check_freq: bool = True
) -> None:
    # [FUNCTION DOC]
    # HELPER: test assertion. Parameters: left and right (must be datetime
    # arrays), obj (label) and check_freq (default true). Compares the
    # underlying datetime64 data arrays, the frequency if requested, and the
    # timezone attribute. No docstring exists for this function.
    __tracebackhide__ = True
    _check_isinstance(left, right, DatetimeArray)

    assert_numpy_array_equal(left._ndarray, right._ndarray, obj=f"{obj}._ndarray")
    if check_freq:
        assert_attr_equal("freq", left, right, obj=obj)
    assert_attr_equal("tz", left, right, obj=obj)


def assert_timedelta_array_equal(
    left, right, obj: str = "TimedeltaArray", check_freq: bool = True
) -> None:
    # [FUNCTION DOC]
    # HELPER: test assertion. Parameters: left and right (must be timedelta
    # arrays), obj (label) and check_freq (default true). Compares the
    # underlying timedelta64 data arrays and, if requested, the frequency.
    # No docstring exists for this function.
    __tracebackhide__ = True
    _check_isinstance(left, right, TimedeltaArray)
    assert_numpy_array_equal(left._ndarray, right._ndarray, obj=f"{obj}._ndarray")
    if check_freq:
        assert_attr_equal("freq", left, right, obj=obj)


def raise_assert_detail(
    obj, message, left, right, diff=None, first_diff=None, index_values=None
) -> NoReturn:
    # [FUNCTION DOC]
    # HELPER: shared failure raiser for the assert functions; never returns.
    # Parameters: obj (label), message (explanation), left and right (the
    # differing values), diff (optional extra diff text), first_diff (optional
    # extra line) and index_values (optional index or array of labels).
    # Builds the message: first line is the label followed by the words are
    # different, blank line, the message; then, if index_values is given, a line
    # listing it; a left line and a right line (arrays pretty-printed; categorical,
    # string and numpy-backed dtypes shown by their repr); then optional diff and
    # first-diff lines. Always raises an assertion error with that text.
    __tracebackhide__ = True

    msg = f"""{obj} are different

{message}"""

    if isinstance(index_values, Index):
        index_values = np.asarray(index_values)

    if isinstance(index_values, np.ndarray):
        msg += f"\n[index]: {pprint_thing(index_values)}"

    if isinstance(left, np.ndarray):
        left = pprint_thing(left)
    elif isinstance(left, (CategoricalDtype, StringDtype, NumpyEADtype)):
        left = repr(left)

    if isinstance(right, np.ndarray):
        right = pprint_thing(right)
    elif isinstance(right, (CategoricalDtype, StringDtype, NumpyEADtype)):
        right = repr(right)

    msg += f"""
[left]:  {left}
[right]: {right}"""

    if diff is not None:
        msg += f"\n[diff]: {diff}"

    if first_diff is not None:
        msg += f"\n{first_diff}"

    raise AssertionError(msg)


def assert_numpy_array_equal(
    left,
    right,
    strict_nan: bool = False,
    check_dtype: bool | Literal["equiv"] = True,
    err_msg=None,
    check_same=None,
    obj: str = "numpy array",
    index_values=None,
) -> None:
    # [FUNCTION DOC]
    # HELPER: test assertion. Parameters: left and right (numpy arrays);
    # strict_nan (treat NaN and none as different, default false); check_dtype
    # (default true; compare dtypes when both are arrays); err_msg (custom
    # failure message); check_same (none, the word same or the word copy: require
    # that the two share, or do not share, the same underlying base object);
    # obj (label); index_values (optional labels shown in messages).
    # Steps: assert class equality, then that both are numpy arrays; apply the
    # identity check on bases if requested (assertion error stating they are or
    # are not the same); test equivalence with the library's array equivalence
    # function, and if different, raise: the custom message if given, otherwise a
    # shape-difference error or an error with the percentage of differing rows
    # (counted along the first axis); finally compare dtypes if requested.
    # KNOWN ISSUE: the parameter check_dtype is annotated to allow the word equiv
    # but only truthiness is used.
    # KNOWN ISSUE: the percentage divides the number of differing first-axis
    # entries by the total element count rather than by the number of entries, so
    # it is understated for multi-dimensional arrays.
    """
    Check that 'np.ndarray' is equivalent.

    Parameters
    ----------
    left, right : numpy.ndarray or iterable
        The two arrays to be compared.
    strict_nan : bool, default False
        If True, consider NaN and None to be different.
    check_dtype : bool, default True
        Check dtype if both a and b are np.ndarray.
    err_msg : str, default None
        If provided, used as assertion message.
    check_same : None|'copy'|'same', default None
        Ensure left and right refer/do not refer to the same memory area.
    obj : str, default 'numpy array'
        Specify object name being compared, internally used to show appropriate
        assertion message.
    index_values : Index | numpy.ndarray, default None
        optional index (shared by both left and right), used in output.
    """
    __tracebackhide__ = True

    # instance validation
    # Show a detailed error message when classes are different
    assert_class_equal(left, right, obj=obj)
    # both classes must be an np.ndarray
    _check_isinstance(left, right, np.ndarray)

    def _get_base(obj):
        # [FUNCTION DOC]
        # HELPER: for the same-memory check. Returns the base object that owns the
        # array's memory if it has one, otherwise the object itself.
        return obj.base if getattr(obj, "base", None) is not None else obj

    left_base = _get_base(left)
    right_base = _get_base(right)

    if check_same == "same":
        if left_base is not right_base:
            raise AssertionError(f"{left_base!r} is not {right_base!r}")
    elif check_same == "copy":
        if left_base is right_base:
            raise AssertionError(f"{left_base!r} is {right_base!r}")

    def _raise(left, right, err_msg) -> NoReturn:
        # [FUNCTION DOC]
        # HELPER: for the numpy array assertion; never returns. If no custom
        # message is given: raises a shape-mismatch error if shapes differ;
        # otherwise counts first-axis entries that are not equivalent, converts to
        # a percentage of the total element count, and raises a values-differ error
        # including the optional index labels. If a custom message is given, raises
        # an assertion error with exactly that text.
        if err_msg is None:
            if left.shape != right.shape:
                raise_assert_detail(
                    obj, f"{obj} shapes are different", left.shape, right.shape
                )

            diff = 0
            for left_arr, right_arr in zip(left, right, strict=True):
                # count up differences
                if not array_equivalent(left_arr, right_arr, strict_nan=strict_nan):
                    diff += 1

            diff = diff * 100.0 / left.size
            msg = f"{obj} values are different ({np.round(diff, 5)} %)"
            raise_assert_detail(obj, msg, left, right, index_values=index_values)

        raise AssertionError(err_msg)

    # compare shape and values
    if not array_equivalent(left, right, strict_nan=strict_nan):
        _raise(left, right, err_msg)

    if check_dtype:
        if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            assert_attr_equal("dtype", left, right, obj=obj)


@set_module("pandas.testing")
def assert_extension_array_equal(
    left,
    right,
    check_dtype: bool | Literal["equiv"] = True,
    index_values=None,
    check_exact: bool | lib.NoDefault = lib.no_default,
    rtol: float | lib.NoDefault = lib.no_default,
    atol: float | lib.NoDefault = lib.no_default,
    obj: str = "ExtensionArray",
) -> None:
    # [FUNCTION DOC]
    # HELPER: test assertion (public as a testing function). Parameters: left and
    # right (extension arrays; an assertion error is raised if either is not
    # one); check_dtype (default true; may be the word equiv); index_values
    # (optional labels for messages); check_exact, rtol, atol (default: not
    # given); obj (label).
    # Tolerance resolution: if none of check_exact, rtol, atol is given,
    # check_exact becomes true when either array has a numeric non-float dtype
    # (so integers compare exactly); if only check_exact is missing it becomes
    # false; rtol and atol default to 1e-5 and 1e-8.
    # Steps: optionally compare dtypes; for two arrays of the same datetime-like
    # class, if dtype checking is off and the units differ, return success when
    # all values are equal across resolutions, otherwise compare the underlying
    # int64 views and return; otherwise compare the missing-value masks; for the
    # python-storage string dtype with a NaN missing marker, assert that missing
    # slots really hold NaN (separately for left and right); then convert the
    # non-missing values to object arrays and compare them exactly or
    # approximately depending on check_exact.
    # KNOWN ISSUE: the existing docstring states check_exact defaults to False,
    # but the code chooses it from dtypes when no tolerance arguments are given.
    """
    Check that left and right ExtensionArrays are equal.

    This method compares two ``ExtensionArray`` instances for equality,
    including checks for missing values, the dtype of the arrays, and
    the exactness of the comparison (or tolerance when comparing floats).

    Parameters
    ----------
    left, right : ExtensionArray
        The two arrays to compare.
    check_dtype : bool, default True
        Whether to check if the ExtensionArray dtypes are identical.
    index_values : Index | numpy.ndarray, default None
        Optional index (shared by both left and right), used in output.
    check_exact : bool, default False
        Whether to compare number exactly.

        .. versionchanged:: 2.2.0

            Defaults to True for integer dtypes if none of
            ``check_exact``, ``rtol`` and ``atol`` are specified.
    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.
    obj : str, default 'ExtensionArray'
        Specify object name being compared, internally used to show appropriate
        assertion message.

        .. versionadded:: 2.0.0

    See Also
    --------
    testing.assert_series_equal : Check that left and right ``Series`` are equal.
    testing.assert_frame_equal : Check that left and right ``DataFrame`` are equal.
    testing.assert_index_equal : Check that left and right ``Index`` are equal.

    Notes
    -----
    Missing values are checked separately from valid values.
    A mask of missing values is computed for each and checked to match.
    The remaining all-valid values are cast to object dtype and checked.

    Examples
    --------
    >>> from pandas import testing as tm
    >>> a = pd.Series([1, 2, 3, 4])
    >>> b, c = a.array, a.array
    >>> tm.assert_extension_array_equal(b, c)
    """
    if (
        check_exact is lib.no_default
        and rtol is lib.no_default
        and atol is lib.no_default
    ):
        check_exact = (
            is_numeric_dtype(left.dtype) and not is_float_dtype(left.dtype)
        ) or (is_numeric_dtype(right.dtype) and not is_float_dtype(right.dtype))
    elif check_exact is lib.no_default:
        check_exact = False

    rtol = rtol if rtol is not lib.no_default else 1.0e-5
    atol = atol if atol is not lib.no_default else 1.0e-8

    assert isinstance(left, ExtensionArray), "left is not an ExtensionArray"
    assert isinstance(right, ExtensionArray), "right is not an ExtensionArray"
    if check_dtype:
        assert_attr_equal("dtype", left, right, obj=f"Attributes of {obj}")

    if (
        isinstance(left, DatetimeLikeArrayMixin)
        and isinstance(right, DatetimeLikeArrayMixin)
        and type(right) == type(left)
    ):
        # GH 52449
        if not check_dtype and left.dtype.kind in "mM":
            if not isinstance(left.dtype, np.dtype):
                l_unit = cast(DatetimeTZDtype, left.dtype).unit
            else:
                l_unit = np.datetime_data(left.dtype)[0]
            if not isinstance(right.dtype, np.dtype):
                r_unit = cast(DatetimeTZDtype, right.dtype).unit
            else:
                r_unit = np.datetime_data(right.dtype)[0]
            if (
                l_unit != r_unit
                and compare_mismatched_resolutions(
                    left._ndarray, right._ndarray, operator.eq
                ).all()
            ):
                return
        # Avoid slow object-dtype comparisons
        # np.asarray for case where we have an np.MaskedArray
        assert_numpy_array_equal(
            np.asarray(left.asi8),
            np.asarray(right.asi8),
            index_values=index_values,
            obj=obj,
        )
        return

    left_na = np.asarray(left.isna())
    right_na = np.asarray(right.isna())
    assert_numpy_array_equal(
        left_na, right_na, obj=f"{obj} NA mask", index_values=index_values
    )

    # Specifically for StringArrayNumpySemantics, validate here we have a valid array
    if (
        isinstance(left.dtype, StringDtype)
        and left.dtype.storage == "python"
        and left.dtype.na_value is np.nan
    ):
        assert np.all(
            [np.isnan(val) for val in left._ndarray[left_na]]  # type: ignore[attr-defined]
        ), "wrong missing value sentinels"
    if (
        isinstance(right.dtype, StringDtype)
        and right.dtype.storage == "python"
        and right.dtype.na_value is np.nan
    ):
        assert np.all(
            [np.isnan(val) for val in right._ndarray[right_na]]  # type: ignore[attr-defined]
        ), "wrong missing value sentinels"

    left_valid = left[~left_na].to_numpy(dtype=object)
    right_valid = right[~right_na].to_numpy(dtype=object)
    if check_exact:
        assert_numpy_array_equal(
            left_valid, right_valid, obj=obj, index_values=index_values
        )
    else:
        _testing.assert_almost_equal(
            left_valid,
            right_valid,
            check_dtype=bool(check_dtype),
            rtol=rtol,
            atol=atol,
            obj=obj,
            index_values=index_values,
        )


# This could be refactored to use the NDFrame.equals method
@set_module("pandas.testing")
@deprecate_kwarg(Pandas4Warning, "check_datetimelike_compat", new_arg_name=None)
def assert_series_equal(
    left,
    right,
    check_dtype: bool | Literal["equiv"] = True,
    check_index_type: bool | Literal["equiv"] = "equiv",
    check_series_type: bool = True,
    check_names: bool = True,
    check_exact: bool | lib.NoDefault = lib.no_default,
    check_datetimelike_compat: bool = False,
    check_categorical: bool = True,
    check_category_order: bool = True,
    check_freq: bool = True,
    check_flags: bool = True,
    rtol: float | lib.NoDefault = lib.no_default,
    atol: float | lib.NoDefault = lib.no_default,
    obj: str = "Series",
    *,
    check_index: bool = True,
    check_like: bool = False,
) -> None:
    # [FUNCTION DOC]
    # HELPER: test assertion (public as a testing function; one deprecated
    # keyword is accepted and ignored by a decorator that warns). Compares two
    # series. Parameters are as in the existing docstring: check_dtype,
    # check_index_type, check_series_type, check_names, check_exact (default:
    # not given), check_datetimelike_compat (deprecated, default false),
    # check_categorical, check_category_order, check_freq, check_flags, rtol,
    # atol (default: not given), obj, and the keyword-only check_index (default
    # true) and check_like (default false).
    # Tolerance resolution: when none of check_exact, rtol and atol is given,
    # exact comparison is used for values if either series has a numeric
    # non-float dtype, and exact comparison is used for the indexes if all
    # levels of either index have integer dtypes; when only check_exact is
    # missing, both are false; when check_exact is given, it applies to the index
    # too. rtol and atol default to 1e-5 and 1e-8.
    # Raises a value error if check_like is true while check_index is false.
    # Flow: instance and class checks, length check, flags check, index
    # comparison (order-insensitive when check_like) and then reindexing of left
    # like right when check_like; frequency check for datetime-like indexes;
    # dtype check (skipped when both are categorical and categorical checking
    # is off); then the values comparison chosen by the first matching case in
    # order: exact comparison (extension arrays with each other, else numpy
    # conversion); deprecated datetime-like compatibility comparison via indexes;
    # interval arrays; categorical on either side (approximate comparison);
    # both extension dtypes; an extension dtype versus a datetime-like dtype;
    # both datetime-like; and finally approximate comparison. Then names
    # (if requested) and categorical content (if requested) are compared.
    # KNOWN ISSUE: the existing docstring says check_exact defaults to False,
    # but the code derives it from dtypes when tolerances are not supplied.
    # KNOWN ISSUE: the frequency check uses an assert statement, so it is
    # skipped when python runs with assertions disabled, and the flags check
    # likewise.
    """
    Check that left and right Series are equal.

    Parameters
    ----------
    left : Series
        First Series to compare.
    right : Series
        Second Series to compare.
    check_dtype : bool, default True
        Whether to check the Series dtype is identical.
    check_index_type : bool or {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical.
    check_series_type : bool, default True
         Whether to check the Series class is identical.
    check_names : bool, default True
        Whether to check the Series and Index names attribute.
    check_exact : bool, default False
        Whether to compare number exactly. This also applies when checking
        Index equivalence.

        .. versionchanged:: 2.2.0

            Defaults to True for integer dtypes if none of
            ``check_exact``, ``rtol`` and ``atol`` are specified.

        .. versionchanged:: 3.0.0

            check_exact for comparing the Indexes defaults to True by
            checking if an Index is of integer dtypes.

    check_datetimelike_compat : bool, default False
        Compare datetime-like which is comparable ignoring dtype.

        .. deprecated:: 3.0

    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    check_category_order : bool, default True
        Whether to compare category order of internal Categoricals.
    check_freq : bool, default True
        Whether to check the `freq` attribute on a DatetimeIndex or TimedeltaIndex.
    check_flags : bool, default True
        Whether to check the `flags` attribute.
    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.
    obj : str, default 'Series'
        Specify object name being compared, internally used to show appropriate
        assertion message.
    check_index : bool, default True
        Whether to check index equivalence. If False, then compare only values.
    check_like : bool, default False
        If True, ignore the order of the index. Must be False if check_index is False.
        Note: same labels must be with the same data.

    See Also
    --------
    testing.assert_index_equal : Check that two Indexes are equal.
    testing.assert_frame_equal : Check that two DataFrames are equal.

    Examples
    --------
    >>> from pandas import testing as tm
    >>> a = pd.Series([1, 2, 3, 4])
    >>> b = pd.Series([1, 2, 3, 4])
    >>> tm.assert_series_equal(a, b)
    """
    __tracebackhide__ = True
    if (
        check_exact is lib.no_default
        and rtol is lib.no_default
        and atol is lib.no_default
    ):
        check_exact = (
            is_numeric_dtype(left.dtype) and not is_float_dtype(left.dtype)
        ) or (is_numeric_dtype(right.dtype) and not is_float_dtype(right.dtype))
        left_index_dtypes = (
            [left.index.dtype] if left.index.nlevels == 1 else left.index.dtypes
        )
        right_index_dtypes = (
            [right.index.dtype] if right.index.nlevels == 1 else right.index.dtypes
        )
        check_exact_index = all(
            dtype.kind in "iu" for dtype in left_index_dtypes
        ) or all(dtype.kind in "iu" for dtype in right_index_dtypes)
    elif check_exact is lib.no_default:
        check_exact = False
        check_exact_index = False
    else:
        check_exact_index = check_exact

    rtol = rtol if rtol is not lib.no_default else 1.0e-5
    atol = atol if atol is not lib.no_default else 1.0e-8

    if not check_index and check_like:
        raise ValueError("check_like must be False if check_index is False")

    # instance validation
    _check_isinstance(left, right, Series)

    if check_series_type:
        assert_class_equal(left, right, obj=obj)

    # length comparison
    if len(left) != len(right):
        msg1 = f"{len(left)}, {left.index}"
        msg2 = f"{len(right)}, {right.index}"
        raise_assert_detail(obj, "Series length are different", msg1, msg2)

    if check_flags:
        assert left.flags == right.flags, f"{left.flags!r} != {right.flags!r}"

    if check_index:
        # GH #38183
        assert_index_equal(
            left.index,
            right.index,
            exact=check_index_type,
            check_names=check_names,
            check_exact=check_exact_index,
            check_categorical=check_categorical,
            check_order=not check_like,
            rtol=rtol,
            atol=atol,
            obj=f"{obj}.index",
        )

    if check_like:
        left = left.reindex_like(right)

    if check_freq and isinstance(left.index, (DatetimeIndex, TimedeltaIndex)):
        lidx = left.index
        ridx = right.index
        assert lidx.freq == ridx.freq, (lidx.freq, ridx.freq)

    if check_dtype:
        # We want to skip exact dtype checking when `check_categorical`
        # is False. We'll still raise if only one is a `Categorical`,
        # regardless of `check_categorical`
        if (
            isinstance(left.dtype, CategoricalDtype)
            and isinstance(right.dtype, CategoricalDtype)
            and not check_categorical
        ):
            pass
        else:
            assert_attr_equal("dtype", left, right, obj=f"Attributes of {obj}")
    if check_exact:
        left_values = left._values
        right_values = right._values
        # Only check exact if dtype is numeric
        if isinstance(left_values, ExtensionArray) and isinstance(
            right_values, ExtensionArray
        ):
            assert_extension_array_equal(
                left_values,
                right_values,
                check_dtype=check_dtype,
                index_values=left.index,
                obj=str(obj),
            )
        else:
            # convert both to NumPy if not, check_dtype would raise earlier
            lv, rv = left_values, right_values
            if isinstance(left_values, ExtensionArray):
                lv = left_values.to_numpy()
            if isinstance(right_values, ExtensionArray):
                rv = right_values.to_numpy()
            assert_numpy_array_equal(
                lv,
                rv,
                check_dtype=check_dtype,
                obj=str(obj),
                index_values=left.index,
            )
    elif check_datetimelike_compat and (
        needs_i8_conversion(left.dtype) or needs_i8_conversion(right.dtype)
    ):
        # we want to check only if we have compat dtypes
        # e.g. integer and M|m are NOT compat, but we can simply check
        # the values in that case

        # datetimelike may have different objects (e.g. datetime.datetime
        # vs Timestamp) but will compare equal
        if not Index(left._values).equals(Index(right._values)):
            msg = (
                f"[datetimelike_compat=True] {left._values} "
                f"is not equal to {right._values}."
            )
            raise AssertionError(msg)
    elif isinstance(left.dtype, IntervalDtype) and isinstance(
        right.dtype, IntervalDtype
    ):
        assert_interval_array_equal(left.array, right.array)
    elif isinstance(left.dtype, CategoricalDtype) or isinstance(
        right.dtype, CategoricalDtype
    ):
        _testing.assert_almost_equal(
            left._values,
            right._values,
            rtol=rtol,
            atol=atol,
            check_dtype=bool(check_dtype),
            obj=str(obj),
            index_values=left.index,
        )
    elif isinstance(left.dtype, ExtensionDtype) and isinstance(
        right.dtype, ExtensionDtype
    ):
        assert_extension_array_equal(
            left._values,
            right._values,
            rtol=rtol,
            atol=atol,
            check_dtype=check_dtype,
            index_values=left.index,
            obj=str(obj),
        )
    elif is_extension_array_dtype_and_needs_i8_conversion(
        left.dtype, right.dtype
    ) or is_extension_array_dtype_and_needs_i8_conversion(right.dtype, left.dtype):
        assert_extension_array_equal(
            left._values,
            right._values,
            check_dtype=check_dtype,
            index_values=left.index,
            obj=str(obj),
        )
    elif needs_i8_conversion(left.dtype) and needs_i8_conversion(right.dtype):
        # DatetimeArray or TimedeltaArray
        assert_extension_array_equal(
            left._values,
            right._values,
            check_dtype=check_dtype,
            index_values=left.index,
            obj=str(obj),
        )
    else:
        _testing.assert_almost_equal(
            left._values,
            right._values,
            rtol=rtol,
            atol=atol,
            check_dtype=bool(check_dtype),
            obj=str(obj),
            index_values=left.index,
        )

    # metadata comparison
    if check_names:
        assert_attr_equal("name", left, right, obj=obj)

    if check_categorical:
        if isinstance(left.dtype, CategoricalDtype) or isinstance(
            right.dtype, CategoricalDtype
        ):
            assert_categorical_equal(
                left._values,
                right._values,
                obj=f"{obj} category",
                check_category_order=check_category_order,
            )


# This could be refactored to use the NDFrame.equals method
@set_module("pandas.testing")
@deprecate_kwarg(Pandas4Warning, "check_datetimelike_compat", new_arg_name=None)
def assert_frame_equal(
    left,
    right,
    check_dtype: bool | Literal["equiv"] = True,
    check_index_type: bool | Literal["equiv"] = "equiv",
    check_column_type: bool | Literal["equiv"] = "equiv",
    check_frame_type: bool = True,
    check_names: bool = True,
    by_blocks: bool = False,
    check_exact: bool | lib.NoDefault = lib.no_default,
    check_datetimelike_compat: bool = False,
    check_categorical: bool = True,
    check_like: bool = False,
    check_freq: bool = True,
    check_flags: bool = True,
    rtol: float | lib.NoDefault = lib.no_default,
    atol: float | lib.NoDefault = lib.no_default,
    obj: str = "DataFrame",
) -> None:
    # [FUNCTION DOC]
    # HELPER: test assertion (public as a testing function; the deprecated
    # keyword is handled by a decorator). Compares two frames. Parameters as in
    # the existing docstring (check_dtype, check_index_type, check_column_type,
    # check_frame_type, check_names, by_blocks, check_exact, the deprecated
    # check_datetimelike_compat, check_categorical, check_like, check_freq,
    # check_flags, rtol, atol, obj).
    # Tolerance handling: for the index and column comparisons, check_exact
    # defaults to false and rtol and atol to 1e-5 and 1e-8 when not given; the
    # original (possibly unspecified) values are passed on to the per-column
    # series comparison so integer columns are still compared exactly there.
    # Flow: both must be frames; if check_frame_type the left must be an instance
    # of the type of right; shapes compared; flags compared if requested; index
    # and columns compared (order-insensitive when check_like; then left is
    # reindexed like right); then either by blocks (frames of each dtype group
    # compared recursively; asserting both have the same dtype groups) or by
    # column position, comparing each column pair with the series assertion
    # (index comparison disabled and flags unchecked since already done, with the
    # deprecation warning of the compat keyword suppressed). The object label of
    # each column includes its position and name.
    # KNOWN ISSUE: the frame class check uses an assert statement without
    # message and the commented-out call to the class assertion is dead code, and
    # asserts are skipped under optimized execution.
    # KNOWN ISSUE: the existing docstring says check_exact defaults to False and
    # to True for integer dtypes; the index and column comparisons always use
    # false when it is not given.
    """
    Check that left and right DataFrame are equal.

    This function is intended to compare two DataFrames and output any
    differences. It is mostly intended for use in unit tests.
    Additional parameters allow varying the strictness of the
    equality checks performed.

    Parameters
    ----------
    left : DataFrame
        First DataFrame to compare.
    right : DataFrame
        Second DataFrame to compare.
    check_dtype : bool, default True
        Whether to check the DataFrame dtype is identical.
    check_index_type : bool or {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical.
    check_column_type : bool or {'equiv'}, default 'equiv'
        Whether to check the columns class, dtype and inferred_type
        are identical. Is passed as the ``exact`` argument of
        :func:`assert_index_equal`.
    check_frame_type : bool, default True
        Whether to check the DataFrame class is identical.
    check_names : bool, default True
        Whether to check that the `names` attribute for both the `index`
        and `column` attributes of the DataFrame is identical.
    by_blocks : bool, default False
        Specify how to compare internal data. If False, compare by columns.
        If True, compare by blocks.
    check_exact : bool, default False
        Whether to compare number exactly. If False, the comparison uses the
        relative tolerance (``rtol``) and absolute tolerance (``atol``)
        parameters to determine if two values are considered close,
        according to the formula: ``|a - b| <= (atol + rtol * |b|)``.

        .. versionchanged:: 2.2.0

            Defaults to True for integer dtypes if none of
            ``check_exact``, ``rtol`` and ``atol`` are specified.
    check_datetimelike_compat : bool, default False
        Compare datetime-like which is comparable ignoring dtype.

        .. deprecated:: 3.0

    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    check_like : bool, default False
        If True, ignore the order of index & columns.
        Note: index labels must match their respective rows
        (same as in columns) - same labels must be with the same data.
    check_freq : bool, default True
        Whether to check the `freq` attribute on a DatetimeIndex or TimedeltaIndex.
    check_flags : bool, default True
        Whether to check the `flags` attribute.
    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.
    obj : str, default 'DataFrame'
        Specify object name being compared, internally used to show appropriate
        assertion message.

    See Also
    --------
    assert_series_equal : Equivalent method for asserting Series equality.
    DataFrame.equals : Check DataFrame equality.

    Examples
    --------
    This example shows comparing two DataFrames that are equal
    but with columns of differing dtypes.

    >>> from pandas.testing import assert_frame_equal
    >>> df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    >>> df2 = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})

    df1 equals itself.

    >>> assert_frame_equal(df1, df1)

    df1 differs from df2 as column 'b' is of a different type.

    >>> assert_frame_equal(df1, df2)
    Traceback (most recent call last):
    ...
    AssertionError: Attributes of DataFrame.iloc[:, 1] (column name="b") are different

    Attribute "dtype" are different
    [left]:  int64
    [right]: float64

    Ignore differing dtypes in columns with check_dtype.

    >>> assert_frame_equal(df1, df2, check_dtype=False)
    """
    __tracebackhide__ = True
    _rtol = rtol if rtol is not lib.no_default else 1.0e-5
    _atol = atol if atol is not lib.no_default else 1.0e-8
    _check_exact = check_exact if check_exact is not lib.no_default else False

    # instance validation
    _check_isinstance(left, right, DataFrame)

    if check_frame_type:
        assert isinstance(left, type(right))
        # assert_class_equal(left, right, obj=obj)

    # shape comparison
    if left.shape != right.shape:
        raise_assert_detail(
            obj, f"{obj} shape mismatch", f"{left.shape!r}", f"{right.shape!r}"
        )

    if check_flags:
        assert left.flags == right.flags, f"{left.flags!r} != {right.flags!r}"

    # index comparison
    assert_index_equal(
        left.index,
        right.index,
        exact=check_index_type,
        check_names=check_names,
        check_exact=_check_exact,
        check_categorical=check_categorical,
        check_order=not check_like,
        rtol=_rtol,
        atol=_atol,
        obj=f"{obj}.index",
    )

    # column comparison
    assert_index_equal(
        left.columns,
        right.columns,
        exact=check_column_type,
        check_names=check_names,
        check_exact=_check_exact,
        check_categorical=check_categorical,
        check_order=not check_like,
        rtol=_rtol,
        atol=_atol,
        obj=f"{obj}.columns",
    )

    if check_like:
        left = left.reindex_like(right)

    # compare by blocks
    if by_blocks:
        rblocks = right._to_dict_of_blocks()
        lblocks = left._to_dict_of_blocks()
        for dtype in list(set(list(lblocks.keys()) + list(rblocks.keys()))):
            assert dtype in lblocks
            assert dtype in rblocks
            assert_frame_equal(
                lblocks[dtype], rblocks[dtype], check_dtype=check_dtype, obj=obj
            )

    # compare by columns
    else:
        for i, col in enumerate(left.columns):
            # We have already checked that columns match, so we can do
            #  fast location-based lookups
            lcol = left._ixs(i, axis=1)
            rcol = right._ixs(i, axis=1)

            # GH #38183
            # use check_index=False, because we do not want to run
            # assert_index_equal for each column,
            # as we already checked it for the whole dataframe before.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="the 'check_datetimelike_compat' keyword",
                    category=Pandas4Warning,
                )
                assert_series_equal(
                    lcol,
                    rcol,
                    check_dtype=check_dtype,
                    check_index_type=check_index_type,
                    check_exact=check_exact,
                    check_names=check_names,
                    check_datetimelike_compat=check_datetimelike_compat,
                    check_categorical=check_categorical,
                    check_freq=check_freq,
                    obj=f'{obj}.iloc[:, {i}] (column name="{col}")',
                    rtol=rtol,
                    atol=atol,
                    check_index=False,
                    check_flags=False,
                )


def assert_equal(left, right, **kwargs) -> None:
    # [FUNCTION DOC]
    # HELPER: test assertion dispatcher. Parameters: left and right (objects to
    # compare) and keyword arguments forwarded to the specific assertion.
    # Dispatch on the type of left, in this order: index (also asserts equal
    # frequency for datetime and timedelta indexes); series; frame; interval,
    # period, datetime, timedelta array assertions; other extension arrays; numpy
    # arrays; strings (no keywords allowed; plain equality); everything else (no
    # keywords allowed; approximate equality). Raises assertion errors on
    # mismatch or when unexpected keywords are given for the last two cases.
    """
    Wrapper for tm.assert_*_equal to dispatch to the appropriate test function.

    Parameters
    ----------
    left, right : Index, Series, DataFrame, ExtensionArray, or np.ndarray
        The two items to be compared.
    **kwargs
        All keyword arguments are passed through to the underlying assert method.
    """
    __tracebackhide__ = True

    if isinstance(left, Index):
        assert_index_equal(left, right, **kwargs)
        if isinstance(left, (DatetimeIndex, TimedeltaIndex)):
            assert left.freq == right.freq, (left.freq, right.freq)
    elif isinstance(left, Series):
        assert_series_equal(left, right, **kwargs)
    elif isinstance(left, DataFrame):
        assert_frame_equal(left, right, **kwargs)
    elif isinstance(left, IntervalArray):
        assert_interval_array_equal(left, right, **kwargs)
    elif isinstance(left, PeriodArray):
        assert_period_array_equal(left, right, **kwargs)
    elif isinstance(left, DatetimeArray):
        assert_datetime_array_equal(left, right, **kwargs)
    elif isinstance(left, TimedeltaArray):
        assert_timedelta_array_equal(left, right, **kwargs)
    elif isinstance(left, ExtensionArray):
        assert_extension_array_equal(left, right, **kwargs)
    elif isinstance(left, np.ndarray):
        assert_numpy_array_equal(left, right, **kwargs)
    elif isinstance(left, str):
        assert kwargs == {}
        assert left == right
    else:
        assert kwargs == {}
        assert_almost_equal(left, right)


def assert_sp_array_equal(left, right) -> None:
    # [FUNCTION DOC]
    # HELPER: test assertion. Parameters: left and right (must be sparse
    # arrays). Compares the stored values, verifies the sparse index objects
    # are of the sparse index class and equal (raising a detail error naming the
    # index otherwise), then the fill value, dtype, and the densified arrays.
    # KNOWN ISSUE: the else branch after the index equality check contains only
    # a no-op statement and an incomplete comment, i.e. dead code.
    """
    Check that the left and right SparseArray are equal.

    Parameters
    ----------
    left : SparseArray
    right : SparseArray
    """
    _check_isinstance(left, right, pd.arrays.SparseArray)

    assert_numpy_array_equal(left.sp_values, right.sp_values)

    # SparseIndex comparison
    assert isinstance(left.sp_index, SparseIndex)
    assert isinstance(right.sp_index, SparseIndex)

    left_index = left.sp_index
    right_index = right.sp_index

    if not left_index.equals(right_index):
        raise_assert_detail(
            "SparseArray.index", "index are not equal", left_index, right_index
        )
    else:
        # Just ensure a
        pass

    assert_attr_equal("fill_value", left, right)
    assert_attr_equal("dtype", left, right)
    assert_numpy_array_equal(left.to_dense(), right.to_dense())


def assert_contains_all(iterable, dic) -> None:
    # [FUNCTION DOC]
    # HELPER: test assertion. Parameters: iterable (items to look for) and dic
    # (a container supporting membership tests, e.g. a dictionary). Raises an
    # assertion error naming the first item not contained. No docstring exists
    # for this function.
    for k in iterable:
        assert k in dic, f"Did not contain item: {k!r}"


def assert_copy(iter1, iter2, **eql_kwargs) -> None:
    # [FUNCTION DOC]
    # HELPER: test assertion. Parameters: iter1 and iter2 (iterables of the same
    # length; a length mismatch raises a value error) and keyword arguments passed
    # to the approximate comparison. For each pair of elements asserts they are
    # approximately equal and that they are not the same object, raising an
    # assertion error naming the types otherwise.
    """
    iter1, iter2: iterables that produce elements
    comparable with assert_almost_equal

    Checks that the elements are equal, but not
    the same object. (Does not check that items
    in sequences are also not the same object)
    """
    for elem1, elem2 in zip(iter1, iter2, strict=True):
        assert_almost_equal(elem1, elem2, **eql_kwargs)
        msg = (
            f"Expected object {type(elem1)!r} and object {type(elem2)!r} to be "
            "different objects, but they were the same object."
        )
        assert elem1 is not elem2, msg


def is_extension_array_dtype_and_needs_i8_conversion(
    left_dtype: DtypeObj, right_dtype: DtypeObj
) -> bool:
    # [FUNCTION DOC]
    # HELPER: for the series assertion. Parameters: left_dtype and right_dtype.
    # Returns true when the left dtype is an extension dtype and the right dtype
    # is a kind that needs conversion to 64-bit integers (datetime-like). The
    # test is not symmetric; callers call it in both orders (related to issue
    # 37609).
    """
    Checks that we have the combination of an ExtensionArraydtype and
    a dtype that should be converted to int64

    Returns
    -------
    bool

    Related to issue #37609
    """
    return isinstance(left_dtype, ExtensionDtype) and needs_i8_conversion(right_dtype)


def assert_indexing_slices_equivalent(ser: Series, l_slc: slice, i_slc: slice) -> None:
    # [FUNCTION DOC]
    # HELPER: test assertion. Parameters: ser (series), l_slc (label slice) and
    # i_slc (positional slice). Takes the positional slice as the expected
    # result and asserts the label-based slice gives an equal series; also asserts
    # plain bracket slicing with the label slice does, but only when the index is
    # not of integer dtype (where bracket slicing is positional).
    """
    Check that ser.iloc[i_slc] matches ser.loc[l_slc] and, if applicable,
    ser[l_slc].
    """
    expected = ser.iloc[i_slc]

    assert_series_equal(ser.loc[l_slc], expected)

    if not is_integer_dtype(ser.index):
        # For integer indices, .loc and plain getitem are position-based.
        assert_series_equal(ser[l_slc], expected)


def assert_metadata_equivalent(
    left: DataFrame | Series, right: DataFrame | Series | None = None
) -> None:
    # [FUNCTION DOC]
    # HELPER: test assertion. Parameters: left (frame or series) and right
    # (frame, series or none). For each name in the metadata list of left, reads
    # the attribute from left (missing treated as none). If right is none,
    # asserts the value is none; otherwise asserts it equals the same attribute of
    # right (missing treated as none). Raises assertion errors without messages.
    """
    Check that ._metadata attributes are equivalent.
    """
    for attr in left._metadata:
        val = getattr(left, attr, None)
        if right is None:
            assert val is None
        else:
            assert val == getattr(right, attr, None)
