# [MODULE DOC]
# Tests for the dtype helper functions in the common dtypes module (accessed as
# com): converting objects to dtypes, comparing dtypes, the family of is-type
# predicates for numpy and extension dtypes, internal dtype lookup helpers, array
# casting, hashability validation and string dtype resolution. Defines module
# level mappings and lists of dtypes used for parametrization.

from __future__ import annotations

import numpy as np
import pytest

from pandas.compat import HAS_PYARROW
from pandas.errors import Pandas4Warning
import pandas.util._test_decorators as td

from pandas.core.dtypes.astype import astype_array
import pandas.core.dtypes.common as com
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    CategoricalDtypeType,
    DatetimeTZDtype,
    ExtensionDtype,
    IntervalDtype,
    PeriodDtype,
)
from pandas.core.dtypes.missing import isna

import pandas as pd
import pandas._testing as tm
from pandas.api.types import pandas_dtype
from pandas.arrays import SparseArray
from pandas.util.version import Version


# EA & Actual Dtypes
def to_ea_dtypes(dtypes):
    # [FUNCTION DOC]
    # HELPER: test parametrization helper. Takes a list of dtype name strings and
    # returns a list of the corresponding pandas extension dtype classes, found by
    # appending the dtype suffix to each name on the top-level pandas namespace.
    """convert list of string dtypes to EA dtype"""
    return [getattr(pd, dt + "Dtype") for dt in dtypes]


def to_numpy_dtypes(dtypes):
    # [FUNCTION DOC]
    # HELPER: test parametrization helper. Takes a list of dtype names and returns a
    # list of numpy type objects for those entries that are strings (non-strings are
    # skipped), looked up by name on numpy.
    """convert list of string dtypes to numpy dtype"""
    return [getattr(np, dt) for dt in dtypes if isinstance(dt, str)]


class TestNumpyEADtype:
    # [CLASS DOC]
    # Group of tests for converting values to dtype objects with pandas_dtype
    # (numpy dtypes, datetime tz, categorical, period strings, and invalid input).
    # No state.
    # Passing invalid dtype, both as a string or object, must raise TypeError
    # Per issue GH15520
    @pytest.mark.parametrize("box", [pd.Timestamp, "pd.Timestamp", list])
    def test_invalid_dtype_error(self, box):
        # [FUNCTION DOC]
        # Parametrized over a Timestamp class, its name as a string and the list type.
        # Converting them raises a type error saying the dtype is not understood
        # (GitHub issue regarding invalid dtypes).
        with pytest.raises(TypeError, match="not understood"):
            com.pandas_dtype(box)

    @pytest.mark.parametrize(
        "dtype",
        [
            object,
            "float64",
            np.object_,
            np.dtype("object"),
            "O",
            np.float64,
            float,
            np.dtype("float64"),
            "object_",
        ],
    )
    def test_pandas_dtype_valid(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over many equivalent spellings of float64 and object (types,
        # strings, numpy dtypes). The converted dtype compares equal to the input.
        assert com.pandas_dtype(dtype) == dtype

    @pytest.mark.parametrize(
        "dtype", ["M8[ns]", "m8[ns]", "object", "float64", "int64"]
    )
    def test_numpy_dtype(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over string spellings of numpy dtypes. The conversion equals the
        # numpy dtype built from the same string.
        assert com.pandas_dtype(dtype) == np.dtype(dtype)

    def test_numpy_string_dtype(self):
        # [FUNCTION DOC]
        # The strings for unicode and bytes kinds convert to the numpy dtypes of those
        # kinds and are not parsed as period frequency strings.
        # do not parse freq-like string as period dtype
        assert com.pandas_dtype("U") == np.dtype("U")
        assert com.pandas_dtype("S") == np.dtype("S")

    @pytest.mark.parametrize(
        "dtype",
        [
            "datetime64[ns, US/Eastern]",
            "datetime64[ns, Asia/Tokyo]",
            "datetime64[ns, UTC]",
            # GH#33885 check that the M8 alias is understood
            "M8[ns, US/Eastern]",
            "M8[ns, Asia/Tokyo]",
            "M8[ns, UTC]",
        ],
    )
    def test_datetimetz_dtype(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over datetime64 strings with timezones, including M8 aliases
        # (regression for a GitHub issue). Conversion equals the dtype parsed by the
        # tz-aware dtype's own string constructor and equals the string itself.
        assert com.pandas_dtype(dtype) == DatetimeTZDtype.construct_from_string(dtype)
        assert com.pandas_dtype(dtype) == dtype

    def test_categorical_dtype(self):
        # [FUNCTION DOC]
        # The string category converts to a default categorical dtype.
        assert com.pandas_dtype("category") == CategoricalDtype()

    @pytest.mark.parametrize(
        "dtype",
        [
            "period[D]",
            "period[3M]",
            "period[us]",
            "Period[D]",
            "Period[3M]",
            "Period[us]",
        ],
    )
    def test_period_dtype(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over period strings (lower and capitalized names, with
        # frequencies). Conversion returns a dtype that equals a freshly built period
        # dtype and equals the string but is not the same object as a new instance.
        assert com.pandas_dtype(dtype) is not PeriodDtype(dtype)
        assert com.pandas_dtype(dtype) == PeriodDtype(dtype)
        assert com.pandas_dtype(dtype) == dtype


dtypes = {
    "datetime_tz": com.pandas_dtype("datetime64[ns, US/Eastern]"),
    "datetime": com.pandas_dtype("datetime64[ns]"),
    "timedelta": com.pandas_dtype("timedelta64[ns]"),
    "period": PeriodDtype("D"),
    "integer": np.dtype(np.int64),
    "float": np.dtype(np.float64),
    "object": np.dtype(object),
    "category": com.pandas_dtype("category"),
    "string": pd.StringDtype("python"),
}


@pytest.mark.parametrize("name1,dtype1", list(dtypes.items()), ids=lambda x: str(x))
@pytest.mark.parametrize("name2,dtype2", list(dtypes.items()), ids=lambda x: str(x))
def test_dtype_equal(name1, dtype1, name2, dtype2):
    # [FUNCTION DOC]
    # Parametrized over every pair from the module-level mapping of sample dtypes
    # (tz-aware datetime, datetime, timedelta, period, integer, float, object,
    # category, python string). A dtype is equal to itself; for different names the
    # two are not equal.
    # match equal to self, but not equal to other
    assert com.is_dtype_equal(dtype1, dtype1)
    if name1 != name2:
        assert not com.is_dtype_equal(dtype1, dtype2)


@pytest.mark.parametrize("name,dtype", list(dtypes.items()), ids=lambda x: str(x))
def test_pyarrow_string_import_error(name, dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue; parametrized over the sample dtypes. None
    # of them is equal to the pyarrow string dtype name, and the comparison does not
    # raise even when pyarrow is unavailable.
    # GH-44276
    assert not com.is_dtype_equal(dtype, "string[pyarrow]")


@pytest.mark.parametrize(
    "dtype1,dtype2",
    [
        (np.int8, np.int64),
        (np.int16, np.int64),
        (np.int32, np.int64),
        (np.float32, np.float64),
        (PeriodDtype("D"), PeriodDtype("2D")),  # PeriodType
        (
            com.pandas_dtype("datetime64[ns, US/Eastern]"),
            com.pandas_dtype("datetime64[ns, CET]"),
        ),  # Datetime
        (None, None),  # gh-15941: no exception should be raised.
    ],
)
def test_dtype_equal_strict(dtype1, dtype2):
    # [FUNCTION DOC]
    # Parametrized over pairs that differ (integer widths, float widths, period
    # frequency, timezone, and a pair of none values). The equality function returns
    # false for each and raises nothing (a GitHub issue about none input).
    assert not com.is_dtype_equal(dtype1, dtype2)


def get_is_dtype_funcs():
    # [FUNCTION DOC]
    # HELPER: collects all predicates in the common module whose names start with
    # is and end with dtype, excluding the string-or-object numpy-dtype fast-path one
    # (which requires a real numpy dtype). Returns them as a list of functions. Used to
    # parametrize the error-catching test at import time.
    """
    Get all functions in pandas.core.dtypes.common that
    begin with 'is_' and end with 'dtype'

    """
    fnames = [f for f in dir(com) if (f.startswith("is_") and f.endswith("dtype"))]
    fnames.remove("is_string_or_object_np_dtype")  # fastpath requires np.dtype obj
    return [getattr(com, fname) for fname in fnames]


@pytest.mark.filterwarnings(
    "ignore:is_categorical_dtype is deprecated:DeprecationWarning"
)
@pytest.mark.parametrize("func", get_is_dtype_funcs(), ids=lambda x: x.__name__)
def test_get_dtype_error_catch(func):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue; parametrized over all is-dtype predicates.
    # Calling each with none must return false and not raise. For the deprecated
    # ones (int64, interval, datetime tz, categorical, period) a deprecation warning
    # naming the function is expected; other deprecation warnings about categorical
    # are ignored by a mark.
    # see gh-15941
    #
    # No exception should be raised.

    msg = f"{func.__name__} is deprecated"
    warn = None
    if (
        func is com.is_int64_dtype
        or func is com.is_interval_dtype
        or func is com.is_datetime64tz_dtype
        or func is com.is_categorical_dtype
        or func is com.is_period_dtype
    ):
        warn = Pandas4Warning

    with tm.assert_produces_warning(warn, match=msg):
        assert not func(None)


def test_is_object():
    # [FUNCTION DOC]
    # The object-dtype predicate is true for the object type and an empty object
    # array, false for int, an int array and a list.
    assert com.is_object_dtype(object)
    assert com.is_object_dtype(np.array([], dtype=object))

    assert not com.is_object_dtype(int)
    assert not com.is_object_dtype(np.array([], dtype=int))
    assert not com.is_object_dtype([1, 2, 3])


@pytest.mark.parametrize(
    "check_scipy", [False, pytest.param(True, marks=td.skip_if_no("scipy"))]
)
def test_is_sparse(check_scipy):
    # [FUNCTION DOC]
    # Parametrized over whether to also check a scipy sparse matrix (skipped if scipy
    # is missing). The deprecated is-sparse predicate emits a deprecation warning, is
    # true for a sparse array, and false for a dense array and a scipy sparse matrix.
    msg = "is_sparse is deprecated"
    with tm.assert_produces_warning(Pandas4Warning, match=msg):
        assert com.is_sparse(SparseArray([1, 2, 3]))

        assert not com.is_sparse(np.array([1, 2, 3]))

        if check_scipy:
            import scipy.sparse

            assert not com.is_sparse(scipy.sparse.bsr_matrix([1, 2, 3]))


def test_is_scipy_sparse():
    # [FUNCTION DOC]
    # Skips if scipy is missing. Is true for a scipy sparse matrix, false for pandas
    # sparse array.
    sp_sparse = pytest.importorskip("scipy.sparse")

    assert com.is_scipy_sparse(sp_sparse.bsr_matrix([1, 2, 3]))

    assert not com.is_scipy_sparse(SparseArray([1, 2, 3]))


def test_is_datetime64_dtype():
    # [FUNCTION DOC]
    # False for object, a list, an int array; true for the numpy datetime type and an
    # empty datetime array.
    assert not com.is_datetime64_dtype(object)
    assert not com.is_datetime64_dtype([1, 2, 3])
    assert not com.is_datetime64_dtype(np.array([], dtype=int))

    assert com.is_datetime64_dtype(np.datetime64)
    assert com.is_datetime64_dtype(np.array([], dtype=np.datetime64))


def test_is_datetime64tz_dtype():
    # [FUNCTION DOC]
    # The deprecated tz predicate emits a deprecation warning; false for object, list,
    # a naive index; true for a tz-aware index.
    msg = "is_datetime64tz_dtype is deprecated"
    with tm.assert_produces_warning(Pandas4Warning, match=msg):
        assert not com.is_datetime64tz_dtype(object)
        assert not com.is_datetime64tz_dtype([1, 2, 3])
        assert not com.is_datetime64tz_dtype(pd.DatetimeIndex([1, 2, 3]))
        assert com.is_datetime64tz_dtype(pd.DatetimeIndex(["2000"], tz="US/Eastern"))


def test_custom_ea_kind_M_not_datetime64tz():
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Defines a local extension dtype whose kind
    # is the datetime character; the tz predicate (deprecated, warns) and the i8
    # conversion predicate both return false for it.
    # GH 34986
    class NotTZDtype(ExtensionDtype):
        # [CLASS DOC]
        # HELPER: nested extension dtype used only by the preceding test; reports kind
        # as the datetime kind character without being a tz dtype.
        @property
        def kind(self) -> str:
            # [FUNCTION DOC]
            # HELPER: property of the nested test dtype; returns the datetime kind character.
            return "M"

    not_tz_dtype = NotTZDtype()
    msg = "is_datetime64tz_dtype is deprecated"
    with tm.assert_produces_warning(Pandas4Warning, match=msg):
        assert not com.is_datetime64tz_dtype(not_tz_dtype)
        assert not com.needs_i8_conversion(not_tz_dtype)


def test_is_timedelta64_dtype():
    # [FUNCTION DOC]
    # False for object, none, lists, datetime arrays and timedelta-like strings;
    # true for the numpy timedelta type, a timedelta Series and a timedelta index.
    assert not com.is_timedelta64_dtype(object)
    assert not com.is_timedelta64_dtype(None)
    assert not com.is_timedelta64_dtype([1, 2, 3])
    assert not com.is_timedelta64_dtype(np.array([], dtype=np.datetime64))
    assert not com.is_timedelta64_dtype("0 days")
    assert not com.is_timedelta64_dtype("0 days 00:00:00")
    assert not com.is_timedelta64_dtype(["0 days 00:00:00"])
    assert not com.is_timedelta64_dtype("NO DATE")

    assert com.is_timedelta64_dtype(np.timedelta64)
    assert com.is_timedelta64_dtype(pd.Series([], dtype="timedelta64[ns]"))
    assert com.is_timedelta64_dtype(pd.to_timedelta(["0 days", "1 days"]))


def test_is_period_dtype():
    # [FUNCTION DOC]
    # The deprecated predicate emits a deprecation warning; false for object, a list
    # and a Period scalar; true for a period dtype and a period index.
    msg = "is_period_dtype is deprecated"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        assert not com.is_period_dtype(object)
        assert not com.is_period_dtype([1, 2, 3])
        assert not com.is_period_dtype(pd.Period("2017-01-01"))

        assert com.is_period_dtype(PeriodDtype(freq="D"))
        assert com.is_period_dtype(pd.PeriodIndex([], freq="Y"))


def test_is_interval_dtype():
    # [FUNCTION DOC]
    # The deprecated predicate emits a deprecation warning; false for object, a list
    # and an Interval scalar; true for an interval dtype and an interval index.
    msg = "is_interval_dtype is deprecated"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        assert not com.is_interval_dtype(object)
        assert not com.is_interval_dtype([1, 2, 3])

        assert com.is_interval_dtype(IntervalDtype())

        interval = pd.Interval(1, 2, closed="right")
        assert not com.is_interval_dtype(interval)
        assert com.is_interval_dtype(pd.IntervalIndex([interval]))


def test_is_categorical_dtype():
    # [FUNCTION DOC]
    # The deprecated predicate emits a deprecation warning; false for object and a list;
    # true for categorical dtype, Categorical and a categorical index.
    msg = "is_categorical_dtype is deprecated"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        assert not com.is_categorical_dtype(object)
        assert not com.is_categorical_dtype([1, 2, 3])

        assert com.is_categorical_dtype(CategoricalDtype())
        assert com.is_categorical_dtype(pd.Categorical([1, 2, 3]))
        assert com.is_categorical_dtype(pd.CategoricalIndex([1, 2, 3]))


@pytest.mark.parametrize(
    "dtype, expected",
    [
        (int, False),
        (pd.Series([1, 2]), False),
        (str, True),
        (object, True),
        (np.array(["a", "b"]), True),
        (pd.StringDtype(), True),
        (pd.Index([], dtype="O"), True),
    ],
)
def test_is_string_dtype(dtype, expected):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue; parametrized over (input, expected). True
    # for str, object, string arrays, the string dtype and an object index; false for
    # int and an integer Series. The result must be the exact boolean.
    # GH#54661

    result = com.is_string_dtype(dtype)
    assert result is expected


@pytest.mark.parametrize(
    "data",
    [[(0, 1), (1, 1)], pd.Categorical([1, 2, 3]), np.array([1, 2], dtype=object)],
)
def test_is_string_dtype_arraylike_with_object_elements_not_strings(data):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue; parametrized over a list of tuples, a
    # categorical and an object array of integers wrapped in a Series. The
    # predicate is false since elements are not strings.
    # GH 15585
    assert not com.is_string_dtype(pd.Series(data))


def test_is_string_dtype_nullable(nullable_string_dtype):
    # [FUNCTION DOC]
    # Fixture nullable_string_dtype. A nullable string array is a string dtype.
    assert com.is_string_dtype(pd.array(["a", "b"], dtype=nullable_string_dtype))


integer_dtypes: list = []


@pytest.mark.parametrize(
    "dtype",
    [
        *integer_dtypes,
        pd.Series([1, 2]),
        *tm.ALL_INT_NUMPY_DTYPES,
        *to_numpy_dtypes(tm.ALL_INT_NUMPY_DTYPES),
        *tm.ALL_INT_EA_DTYPES,
        *to_ea_dtypes(tm.ALL_INT_EA_DTYPES),
    ],
)
def test_is_integer_dtype(dtype):
    # [FUNCTION DOC]
    # Parametrized over integer Series, numpy and extension integer dtypes (as names
    # and types). All satisfy the integer predicate. An empty list of extra dtypes
    # is spliced in.
    assert com.is_integer_dtype(dtype)


@pytest.mark.parametrize(
    "dtype",
    [
        str,
        float,
        np.datetime64,
        np.timedelta64,
        pd.Index([1, 2.0]),
        np.array(["a", "b"]),
        np.array([], dtype="m8[ns]"),
    ],
)
def test_is_not_integer_dtype(dtype):
    # [FUNCTION DOC]
    # Parametrized over str, float, datetime/timedelta, a float index, a string array
    # and a timedelta array: none satisfy the integer predicate.
    assert not com.is_integer_dtype(dtype)


signed_integer_dtypes: list = []


@pytest.mark.parametrize(
    "dtype",
    [
        *signed_integer_dtypes,
        pd.Series([1, 2]),
        *tm.SIGNED_INT_NUMPY_DTYPES,
        *to_numpy_dtypes(tm.SIGNED_INT_NUMPY_DTYPES),
        *tm.SIGNED_INT_EA_DTYPES,
        *to_ea_dtypes(tm.SIGNED_INT_EA_DTYPES),
    ],
)
def test_is_signed_integer_dtype(dtype):
    # [FUNCTION DOC]
    # Parametrized over signed integer Series, numpy and extension dtypes.
    # KNOWN ISSUE: the assertion calls the generic integer predicate rather than the signed integer predicate, so signedness is not actually verified.
    assert com.is_integer_dtype(dtype)


@pytest.mark.parametrize(
    "dtype",
    [
        str,
        float,
        np.datetime64,
        np.timedelta64,
        pd.Index([1, 2.0]),
        np.array(["a", "b"]),
        np.array([], dtype="m8[ns]"),
        *tm.UNSIGNED_INT_NUMPY_DTYPES,
        *to_numpy_dtypes(tm.UNSIGNED_INT_NUMPY_DTYPES),
        *tm.UNSIGNED_INT_EA_DTYPES,
        *to_ea_dtypes(tm.UNSIGNED_INT_EA_DTYPES),
    ],
)
def test_is_not_signed_integer_dtype(dtype):
    # [FUNCTION DOC]
    # Parametrized over non-integer inputs and all unsigned integer dtypes: none
    # satisfy the signed integer predicate.
    assert not com.is_signed_integer_dtype(dtype)


unsigned_integer_dtypes: list = []


@pytest.mark.parametrize(
    "dtype",
    [
        *unsigned_integer_dtypes,
        pd.Series([1, 2], dtype=np.uint32),
        *tm.UNSIGNED_INT_NUMPY_DTYPES,
        *to_numpy_dtypes(tm.UNSIGNED_INT_NUMPY_DTYPES),
        *tm.UNSIGNED_INT_EA_DTYPES,
        *to_ea_dtypes(tm.UNSIGNED_INT_EA_DTYPES),
    ],
)
def test_is_unsigned_integer_dtype(dtype):
    # [FUNCTION DOC]
    # Parametrized over an unsigned Series and unsigned numpy and extension dtypes: all
    # satisfy the unsigned integer predicate.
    assert com.is_unsigned_integer_dtype(dtype)


@pytest.mark.parametrize(
    "dtype",
    [
        str,
        float,
        np.datetime64,
        np.timedelta64,
        pd.Index([1, 2.0]),
        np.array(["a", "b"]),
        np.array([], dtype="m8[ns]"),
        *tm.SIGNED_INT_NUMPY_DTYPES,
        *to_numpy_dtypes(tm.SIGNED_INT_NUMPY_DTYPES),
        *tm.SIGNED_INT_EA_DTYPES,
        *to_ea_dtypes(tm.SIGNED_INT_EA_DTYPES),
    ],
)
def test_is_not_unsigned_integer_dtype(dtype):
    # [FUNCTION DOC]
    # Parametrized over non-integer inputs and all signed integer dtypes: none
    # satisfy the unsigned integer predicate.
    assert not com.is_unsigned_integer_dtype(dtype)


@pytest.mark.parametrize(
    "dtype", [np.int64, np.array([1, 2], dtype=np.int64), "Int64", pd.Int64Dtype]
)
def test_is_int64_dtype(dtype):
    # [FUNCTION DOC]
    # Parametrized over int64 type, an int64 array, the nullable name and class. The
    # deprecated predicate warns and is true.
    msg = "is_int64_dtype is deprecated"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        assert com.is_int64_dtype(dtype)


def test_type_comparison_with_numeric_ea_dtype(any_numeric_ea_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue; fixture any_numeric_ea_dtype. Converting an
    # extension dtype with pandas_dtype gives something equal to the original.
    # GH#43038
    assert pandas_dtype(any_numeric_ea_dtype) == any_numeric_ea_dtype


def test_type_comparison_with_real_numpy_dtype(any_real_numpy_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue; fixture any_real_numpy_dtype. Converting a
    # numpy dtype with pandas_dtype gives something equal to the original.
    # GH#43038
    assert pandas_dtype(any_real_numpy_dtype) == any_real_numpy_dtype


def test_type_comparison_with_signed_int_ea_dtype_and_signed_int_numpy_dtype(
    any_signed_int_ea_dtype, any_signed_int_numpy_dtype
):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue; fixtures for signed extension and numpy
    # integer dtypes. A converted extension integer dtype is not equal to a numpy
    # integer dtype.
    # GH#43038
    assert not pandas_dtype(any_signed_int_ea_dtype) == any_signed_int_numpy_dtype


@pytest.mark.parametrize(
    "dtype",
    [
        str,
        float,
        np.int32,
        np.uint64,
        pd.Index([1, 2.0]),
        np.array(["a", "b"]),
        np.array([1, 2], dtype=np.uint32),
        "int8",
        "Int8",
        pd.Int8Dtype,
    ],
)
def test_is_not_int64_dtype(dtype):
    # [FUNCTION DOC]
    # Parametrized over other types, arrays, strings and classes (other widths,
    # nullable Int8). The deprecated predicate warns and is false.
    msg = "is_int64_dtype is deprecated"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        assert not com.is_int64_dtype(dtype)


def test_is_datetime64_any_dtype():
    # [FUNCTION DOC]
    # False for int, str and plain arrays; true for the numpy datetime type, a
    # datetime array, a tz-aware dtype and a datetime index.
    assert not com.is_datetime64_any_dtype(int)
    assert not com.is_datetime64_any_dtype(str)
    assert not com.is_datetime64_any_dtype(np.array([1, 2]))
    assert not com.is_datetime64_any_dtype(np.array(["a", "b"]))

    assert com.is_datetime64_any_dtype(np.datetime64)
    assert com.is_datetime64_any_dtype(np.array([], dtype=np.datetime64))
    assert com.is_datetime64_any_dtype(DatetimeTZDtype("ns", "US/Eastern"))
    assert com.is_datetime64_any_dtype(
        pd.DatetimeIndex([1, 2, 3], dtype="datetime64[ns]")
    )


def test_is_datetime64_ns_dtype():
    # [FUNCTION DOC]
    # False for non-datetime inputs, the generic datetime type, a picosecond array
    # and a non-nanosecond tz-aware dtype; true for a nanosecond tz-aware dtype and
    # a nanosecond datetime index.
    assert not com.is_datetime64_ns_dtype(int)
    assert not com.is_datetime64_ns_dtype(str)
    assert not com.is_datetime64_ns_dtype(np.datetime64)
    assert not com.is_datetime64_ns_dtype(np.array([1, 2]))
    assert not com.is_datetime64_ns_dtype(np.array(["a", "b"]))
    assert not com.is_datetime64_ns_dtype(np.array([], dtype=np.datetime64))

    # This datetime array has the wrong unit (ps instead of ns)
    assert not com.is_datetime64_ns_dtype(np.array([], dtype="datetime64[ps]"))

    assert com.is_datetime64_ns_dtype(DatetimeTZDtype("ns", "US/Eastern"))
    assert com.is_datetime64_ns_dtype(
        pd.DatetimeIndex([1, 2, 3], dtype=np.dtype("datetime64[ns]"))
    )

    # non-nano dt64tz
    assert not com.is_datetime64_ns_dtype(DatetimeTZDtype("us", "US/Eastern"))


@pytest.mark.filterwarnings(
    "ignore:.*'generic' unit for NumPy timedelta:DeprecationWarning"
)
def test_is_timedelta64_ns_dtype():
    # [FUNCTION DOC]
    # Ignores a deprecation warning about the generic unit. False for picosecond and
    # generic-unit timedelta; true for nanosecond dtype and array.
    assert not com.is_timedelta64_ns_dtype(np.dtype("m8[ps]"))
    assert not com.is_timedelta64_ns_dtype(np.array([1, 2], dtype="m8"))

    assert com.is_timedelta64_ns_dtype(np.dtype("m8[ns]"))
    assert com.is_timedelta64_ns_dtype(np.array([1, 2], dtype="m8[ns]"))


def test_is_numeric_v_string_like():
    # [FUNCTION DOC]
    # False when the pair is numeric/numeric, string/string or a numeric array with
    # a numeric scalar; true when a numeric array is paired with a string scalar or
    # a string array, in either order.
    assert not com.is_numeric_v_string_like(np.array([1]), 1)
    assert not com.is_numeric_v_string_like(np.array([1]), np.array([2]))
    assert not com.is_numeric_v_string_like(np.array(["foo"]), np.array(["foo"]))

    assert com.is_numeric_v_string_like(np.array([1]), "foo")
    assert com.is_numeric_v_string_like(np.array([1, 2]), np.array(["foo"]))
    assert com.is_numeric_v_string_like(np.array(["foo"]), np.array([1, 2]))


def test_needs_i8_conversion():
    # [FUNCTION DOC]
    # Predicate applies to dtypes only: false for str, integer type, Series, a string
    # array, the bare datetime type, a timedelta Series and a tz index (objects); true
    # for the numpy datetime dtype, the timedelta dtype and the tz-aware dtype.
    assert not com.needs_i8_conversion(str)
    assert not com.needs_i8_conversion(np.int64)
    assert not com.needs_i8_conversion(pd.Series([1, 2]))
    assert not com.needs_i8_conversion(np.array(["a", "b"]))

    assert not com.needs_i8_conversion(np.datetime64)
    assert com.needs_i8_conversion(np.dtype(np.datetime64))
    assert not com.needs_i8_conversion(pd.Series([], dtype="timedelta64[ns]"))
    assert com.needs_i8_conversion(pd.Series([], dtype="timedelta64[ns]").dtype)
    assert not com.needs_i8_conversion(pd.DatetimeIndex(["2000"], tz="US/Eastern"))
    assert com.needs_i8_conversion(pd.DatetimeIndex(["2000"], tz="US/Eastern").dtype)


def test_is_numeric_dtype():
    # [FUNCTION DOC]
    # False for str, datetime/timedelta types and string/timedelta arrays; true for
    # int, float, unsigned, a numeric Series and Index, and a custom extension
    # dtype that declares itself numeric.
    assert not com.is_numeric_dtype(str)
    assert not com.is_numeric_dtype(np.datetime64)
    assert not com.is_numeric_dtype(np.timedelta64)
    assert not com.is_numeric_dtype(np.array(["a", "b"]))
    assert not com.is_numeric_dtype(np.array([], dtype="m8[ns]"))

    assert com.is_numeric_dtype(int)
    assert com.is_numeric_dtype(float)
    assert com.is_numeric_dtype(np.uint64)
    assert com.is_numeric_dtype(pd.Series([1, 2]))
    assert com.is_numeric_dtype(pd.Index([1, 2.0]))

    class MyNumericDType(ExtensionDtype):
        # [CLASS DOC]
        # HELPER: nested extension dtype used only by the numeric dtype test; reports its
        # scalar type as str and declares itself numeric. Name lookup and array type
        # construction are not implemented and raise not-implemented errors if called.
        @property
        def type(self):
            # [FUNCTION DOC]
            # HELPER: property of the nested test dtype; returns the string type.
            return str

        @property
        def name(self):
            # [FUNCTION DOC]
            # HELPER: property of the nested test dtype; raises a not-implemented error.
            raise NotImplementedError

        def construct_array_type(self):
            # [FUNCTION DOC]
            # HELPER: method of the nested test dtype; raises a not-implemented error.
            raise NotImplementedError

        def _is_numeric(self) -> bool:
            # [FUNCTION DOC]
            # HELPER: private hook of the nested test dtype; returns true so the dtype is
            # treated as numeric.
            return True

    assert com.is_numeric_dtype(MyNumericDType())


def test_is_any_real_numeric_dtype():
    # [FUNCTION DOC]
    # False for str, bool, complex, object, datetime, complex-bearing arrays and
    # frames; true for int, float and a mixed int/float array.
    assert not com.is_any_real_numeric_dtype(str)
    assert not com.is_any_real_numeric_dtype(bool)
    assert not com.is_any_real_numeric_dtype(complex)
    assert not com.is_any_real_numeric_dtype(object)
    assert not com.is_any_real_numeric_dtype(np.datetime64)
    assert not com.is_any_real_numeric_dtype(np.array(["a", "b", complex(1, 2)]))
    assert not com.is_any_real_numeric_dtype(pd.DataFrame([complex(1, 2), True]))

    assert com.is_any_real_numeric_dtype(int)
    assert com.is_any_real_numeric_dtype(float)
    assert com.is_any_real_numeric_dtype(np.array([1, 2.5]))


def test_is_float_dtype():
    # [FUNCTION DOC]
    # False for str, int, an int Series and a string array; true for float and a
    # float index.
    assert not com.is_float_dtype(str)
    assert not com.is_float_dtype(int)
    assert not com.is_float_dtype(pd.Series([1, 2]))
    assert not com.is_float_dtype(np.array(["a", "b"]))

    assert com.is_float_dtype(float)
    assert com.is_float_dtype(pd.Index([1, 2.0]))


def test_is_bool_dtype():
    # [FUNCTION DOC]
    # False for ints, str, int Series, a categorical of strings, string arrays and
    # indexes, and the nullable integer name; true for bool types, a categorical of
    # booleans, bool arrays and indexes, the nullable boolean dtype, array and name.
    assert not com.is_bool_dtype(int)
    assert not com.is_bool_dtype(str)
    assert not com.is_bool_dtype(pd.Series([1, 2]))
    assert not com.is_bool_dtype(pd.Series(["a", "b"], dtype="category"))
    assert not com.is_bool_dtype(np.array(["a", "b"]))
    assert not com.is_bool_dtype(pd.Index(["a", "b"]))
    assert not com.is_bool_dtype("Int64")

    assert com.is_bool_dtype(bool)
    assert com.is_bool_dtype(np.bool_)
    assert com.is_bool_dtype(pd.Series([True, False], dtype="category"))
    assert com.is_bool_dtype(np.array([True, False]))
    assert com.is_bool_dtype(pd.Index([True, False]))

    assert com.is_bool_dtype(pd.BooleanDtype())
    assert com.is_bool_dtype(pd.array([True, False, None], dtype="boolean"))
    assert com.is_bool_dtype("boolean")


def test_is_bool_dtype_numpy_error():
    # [FUNCTION DOC]
    # Regression test for a GitHub issue: a string that numpy cannot parse as a dtype
    # yields false instead of raising.
    # GH39010
    assert not com.is_bool_dtype("0 - Name")


@pytest.mark.parametrize(
    "check_scipy", [False, pytest.param(True, marks=td.skip_if_no("scipy"))]
)
def test_is_extension_array_dtype(check_scipy):
    # [FUNCTION DOC]
    # Parametrized over whether to include scipy sparse (skipped if missing). False
    # for a list, numpy array, datetime index; true for a categorical (and its
    # Series), sparse array, tz-aware index and tz-aware Series; false for a scipy
    # sparse matrix.
    assert not com.is_extension_array_dtype([1, 2, 3])
    assert not com.is_extension_array_dtype(np.array([1, 2, 3]))
    assert not com.is_extension_array_dtype(pd.DatetimeIndex([1, 2, 3]))

    cat = pd.Categorical([1, 2, 3])
    assert com.is_extension_array_dtype(cat)
    assert com.is_extension_array_dtype(pd.Series(cat))
    assert com.is_extension_array_dtype(SparseArray([1, 2, 3]))
    assert com.is_extension_array_dtype(pd.DatetimeIndex(["2000"], tz="US/Eastern"))

    dtype = DatetimeTZDtype("ns", tz="US/Eastern")
    s = pd.Series([], dtype=dtype)
    assert com.is_extension_array_dtype(s)

    if check_scipy:
        import scipy.sparse

        assert not com.is_extension_array_dtype(scipy.sparse.bsr_matrix([1, 2, 3]))


def test_is_complex_dtype():
    # [FUNCTION DOC]
    # False for int, str, int Series and string array; true for complex types and a
    # complex array.
    assert not com.is_complex_dtype(int)
    assert not com.is_complex_dtype(str)
    assert not com.is_complex_dtype(pd.Series([1, 2]))
    assert not com.is_complex_dtype(np.array(["a", "b"]))

    assert com.is_complex_dtype(np.complex128)
    assert com.is_complex_dtype(complex)
    assert com.is_complex_dtype(np.array([1 + 1j, 5]))


@pytest.mark.parametrize(
    "input_param,result",
    [
        (int, np.dtype(int)),
        ("int32", np.dtype("int32")),
        (float, np.dtype(float)),
        ("float64", np.dtype("float64")),
        (np.dtype("float64"), np.dtype("float64")),
        (str, np.dtype(str)),
        (pd.Series([1, 2], dtype=np.dtype("int16")), np.dtype("int16")),
        (pd.Series(["a", "b"], dtype=object), np.dtype(object)),
        (pd.Index([1, 2]), np.dtype("int64")),
        (pd.Index(["a", "b"], dtype=object), np.dtype(object)),
        ("category", "category"),
        (pd.Categorical(["a", "b"]).dtype, CategoricalDtype(["a", "b"])),
        (pd.Categorical(["a", "b"]), CategoricalDtype(["a", "b"])),
        (pd.CategoricalIndex(["a", "b"]).dtype, CategoricalDtype(["a", "b"])),
        (pd.CategoricalIndex(["a", "b"]), CategoricalDtype(["a", "b"])),
        (CategoricalDtype(), CategoricalDtype()),
        (pd.DatetimeIndex([1, 2]), np.dtype("=M8[ns]")),
        (pd.DatetimeIndex([1, 2]).dtype, np.dtype("=M8[ns]")),
        ("<M8[ns]", np.dtype("<M8[ns]")),
        ("datetime64[ns, Europe/London]", DatetimeTZDtype("ns", "Europe/London")),
        (PeriodDtype(freq="D"), PeriodDtype(freq="D")),
        ("period[D]", PeriodDtype(freq="D")),
        (IntervalDtype(), IntervalDtype()),
    ],
)
def test_get_dtype(input_param, result):
    # [FUNCTION DOC]
    # Parametrized over many inputs (types, strings, dtypes, Series, Index,
    # categorical, datetime index, period, interval) and expected dtypes. The
    # private dtype lookup returns the expected dtype.
    assert com._get_dtype(input_param) == result


@pytest.mark.parametrize(
    "input_param,expected_error_message",
    [
        (None, "Cannot deduce dtype from null object"),
        (1, "data type not understood"),
        (1.2, "data type not understood"),
        # numpy dev changed from double-quotes to single quotes
        ("random string", "data type [\"']random string[\"'] not understood"),
        (pd.DataFrame([1, 2]), "data type not understood"),
        (
            np.typing.NDArray[np.float32],
            "data type not understood|Cannot interpret.*numpy.*as a data type",
        ),
    ],
)
def test_get_dtype_fails(input_param, expected_error_message):
    # [FUNCTION DOC]
    # Parametrized over invalid inputs (none, numbers, an unknown string, a frame, a
    # numpy typing alias) and message patterns. The private lookup raises a type error
    # matching the pattern, to which an alternative for newer numpy wording is appended.
    # python objects
    # 2020-02-02 npdev changed error message
    expected_error_message += f"|Cannot interpret '{input_param}' as a data type"
    with pytest.raises(TypeError, match=expected_error_message):
        com._get_dtype(input_param)


@pytest.mark.parametrize(
    "input_param,result",
    [
        (int, np.dtype(int).type),
        ("int32", np.int32),
        (float, np.dtype(float).type),
        ("float64", np.float64),
        (np.dtype("float64"), np.float64),
        (str, np.dtype(str).type),
        (pd.Series([1, 2], dtype=np.dtype("int16")), np.int16),
        (pd.Series(["a", "b"], dtype=object), np.object_),
        (pd.Index([1, 2], dtype="int64"), np.int64),
        (pd.Index(["a", "b"], dtype=object), np.object_),
        ("category", CategoricalDtypeType),
        (pd.Categorical(["a", "b"]).dtype, CategoricalDtypeType),
        (pd.Categorical(["a", "b"]), CategoricalDtypeType),
        (pd.CategoricalIndex(["a", "b"]).dtype, CategoricalDtypeType),
        (pd.CategoricalIndex(["a", "b"]), CategoricalDtypeType),
        (pd.DatetimeIndex([1, 2]), np.datetime64),
        (pd.DatetimeIndex([1, 2]).dtype, np.datetime64),
        ("<M8[ns]", np.datetime64),
        (pd.DatetimeIndex(["2000"], tz="Europe/London"), pd.Timestamp),
        (pd.DatetimeIndex(["2000"], tz="Europe/London").dtype, pd.Timestamp),
        ("datetime64[ns, Europe/London]", pd.Timestamp),
        (PeriodDtype(freq="D"), pd.Period),
        ("period[D]", pd.Period),
        (IntervalDtype(), pd.Interval),
        (None, type(None)),
        (1, type(None)),
        (1.2, type(None)),
        (pd.DataFrame([1, 2]), type(None)),  # composite dtype
    ],
)
def test__is_dtype_type(input_param, result):
    # [FUNCTION DOC]
    # Parametrized over inputs (types, strings, Series, Index, categoricals,
    # datetimes, periods, intervals, none, numbers, frame) and the expected scalar
    # type. The private helper applies the supplied predicate to the dtype's scalar
    # type; the predicate compares it to the expected type (the none type for
    # inputs with no dtype).
    assert com._is_dtype_type(input_param, lambda tipo: tipo == result)


def test_astype_nansafe_copy_false(any_int_numpy_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Fixture any_int_numpy_dtype. Casting an
    # integer array to float64 without copy returns a converted array, equal to the
    # expected floats.
    # GH#34457 use astype, not view
    arr = np.array([1, 2, 3], dtype=any_int_numpy_dtype)

    dtype = np.dtype("float64")
    result = astype_array(arr, dtype, copy=False)

    expected = np.array([1.0, 2.0, 3.0], dtype=dtype)
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("from_type", [np.datetime64, np.timedelta64])
def test_astype_object_preserves_datetime_na(from_type):
    # [FUNCTION DOC]
    # Parametrized over datetime64 and timedelta64. Casting an array holding a NaT to
    # object keeps it recognized as missing.
    arr = np.array([from_type("NaT", "ns")])
    result = astype_array(arr, dtype=np.dtype("object"))

    assert isna(result)[0]


def test_validate_allhashable():
    # [FUNCTION DOC]
    # Validation of all-hashable arguments returns none for hashable ones; raises a
    # type error ("All elements must be hashable") for an unhashable one; with an
    # error name given the message uses that name.
    assert com.validate_all_hashable(1, "a") is None

    with pytest.raises(TypeError, match="All elements must be hashable"):
        com.validate_all_hashable([])

    with pytest.raises(TypeError, match="list must be a hashable type"):
        com.validate_all_hashable([], error_name="list")


def test_pandas_dtype_numpy_warning():
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Converting the generic numpy integer type:
    # on older numpy a deprecation warning is expected (stacklevel not checked); on
    # newer numpy a type error from numpy is expected.
    # GH#51523
    if Version(np.__version__) < Version("2.3.0.dev0"):
        ctx = tm.assert_produces_warning(
            DeprecationWarning,
            check_stacklevel=False,
            match=(
                "Converting `np.integer` or `np.signedinteger` to a dtype is deprecated"
            ),
        )
    else:
        ctx = tm.external_error_raised(TypeError)

    with ctx:
        pandas_dtype(np.integer)


def test_pandas_dtype_ea_not_instance():
    # [FUNCTION DOC]
    # Regression test for GitHub issues. Passing the categorical dtype class instead
    # of an instance emits a user warning about missing arguments and returns a
    # default categorical dtype.
    # GH 31356 GH 54592
    with tm.assert_produces_warning(UserWarning, match="without any arguments"):
        assert pandas_dtype(CategoricalDtype) == CategoricalDtype()


def test_pandas_dtype_string_dtypes(string_storage):
    # [FUNCTION DOC]
    # Fixture string_storage. With the infer-string option on, the str name and the
    # str type resolve to the string dtype with NaN missing value, storage pyarrow
    # if installed else python, or the configured storage when set. With the option off,
    # str resolves to the numpy unicode dtype. The string name always resolves to
    # the NA-based string dtype with configured storage.
    with pd.option_context("future.infer_string", True):
        # with the default string_storage setting
        result = pandas_dtype("str")
    assert result == pd.StringDtype(
        "pyarrow" if HAS_PYARROW else "python", na_value=np.nan
    )

    with pd.option_context("future.infer_string", True):
        # with the default string_storage setting
        result = pandas_dtype(str)
    assert result == pd.StringDtype(
        "pyarrow" if HAS_PYARROW else "python", na_value=np.nan
    )

    with pd.option_context("future.infer_string", True):
        with pd.option_context("string_storage", string_storage):
            result = pandas_dtype("str")
    assert result == pd.StringDtype(string_storage, na_value=np.nan)

    with pd.option_context("future.infer_string", True):
        with pd.option_context("string_storage", string_storage):
            result = pandas_dtype(str)
    assert result == pd.StringDtype(string_storage, na_value=np.nan)

    with pd.option_context("future.infer_string", False):
        with pd.option_context("string_storage", string_storage):
            result = pandas_dtype("str")
    assert result == np.dtype("U")

    with pd.option_context("string_storage", string_storage):
        result = pandas_dtype("string")
    assert result == pd.StringDtype(string_storage, na_value=pd.NA)


def test_pandas_dtype_string_dtype_alias_with_storage():
    # [FUNCTION DOC]
    # The str alias with a storage suffix is not understood (type error). The string
    # alias with python storage works; with pyarrow storage works if installed,
    # otherwise raises an import error about pyarrow being required.
    with pytest.raises(TypeError, match="not understood"):
        pandas_dtype("str[python]")

    with pytest.raises(TypeError, match="not understood"):
        pandas_dtype("str[pyarrow]")

    result = pandas_dtype("string[python]")
    assert result == pd.StringDtype("python", na_value=pd.NA)

    if HAS_PYARROW:
        result = pandas_dtype("string[pyarrow]")
        assert result == pd.StringDtype("pyarrow", na_value=pd.NA)
    else:
        with pytest.raises(
            ImportError, match="required for PyArrow backed StringArray"
        ):
            pandas_dtype("string[pyarrow]")


@td.skip_if_installed("pyarrow")
def test_construct_from_string_without_pyarrow_installed():
    # [FUNCTION DOC]
    # Regression test for a GitHub issue; only runs when pyarrow is not installed.
    # Building a Series with a pyarrow-backed float dtype raises an import error
    # that says a minimum pyarrow version is required.
    # GH 57928
    with pytest.raises(ImportError, match="pyarrow>=.* is required"):
        pd.Series([-1.5, 0.2, None], dtype="float32[pyarrow]")
