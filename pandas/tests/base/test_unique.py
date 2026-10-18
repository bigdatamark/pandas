# [MODULE DOC]
# Tests for unique and nunique on Index and Series objects (via the
# index_or_series_obj fixture), including missing values and unusual strings.
# No import-time side effects.

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.tests.base.common import allow_na_ops


@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
def test_unique(index_or_series_obj):
    # [FUNCTION DOC]
    # Repeats each element of the object a growing number of times then verifies unique
    # returns the values in first-appearance order: as an Index with the dtype
    # (timezone-aware values normalized) or MultiIndex with names for indexes, and a numpy
    # array for series. Period deprecation warnings are ignored.
    obj = index_or_series_obj
    obj = np.repeat(obj, range(1, len(obj) + 1))
    result = obj.unique()

    # dict.fromkeys preserves the order
    unique_values = list(dict.fromkeys(obj.values))
    if isinstance(obj, pd.MultiIndex):
        expected = pd.MultiIndex.from_tuples(unique_values)
        expected.names = obj.names
        tm.assert_index_equal(result, expected, exact=True)
    elif isinstance(obj, pd.Index):
        expected = pd.Index(unique_values, dtype=obj.dtype)
        if isinstance(obj.dtype, pd.DatetimeTZDtype):
            expected = expected.normalize()
        tm.assert_index_equal(result, expected, exact=True)
    else:
        expected = np.array(unique_values)
        tm.assert_numpy_array_equal(result, expected)


@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
@pytest.mark.parametrize("null_obj", [np.nan, None])
def test_unique_null(null_obj, index_or_series_obj, using_nan_is_na):
    # [FUNCTION DOC]
    # Parametrized over null marker (NaN, None). Skips objects that cannot hold NA,
    # empty ones, MultiIndex, and dtype/NaN combinations where NaN is not a valid NA.
    # Replaces the first two values by the null marker, repeats, and verifies unique
    # returns the null marker first followed by the non-null values in appearance
    # order (the NaN and None cases differ in how duplicates collapse).
    obj = index_or_series_obj

    if not allow_na_ops(obj):
        pytest.skip("type doesn't allow for NA operations")
    elif len(obj) < 1:
        pytest.skip("Test doesn't make sense on empty data")
    elif isinstance(obj, pd.MultiIndex):
        pytest.skip(f"MultiIndex can't hold '{null_obj}'")
    elif (
        null_obj is not None
        and not using_nan_is_na
        and obj.dtype in ["Int64", "UInt16", "Float32"]
    ):
        pytest.skip("NaN is not a valid NA for this dtype.")

    obj = obj.copy(deep=True)
    values = obj._values
    values[0:2] = null_obj

    klass = type(obj)
    repeated_values = np.repeat(values, range(1, len(values) + 1))
    obj = klass(repeated_values, dtype=obj.dtype)
    result = obj.unique()

    unique_values_raw = dict.fromkeys(obj.values)
    # because np.nan == np.nan is False, but None == None is True
    # np.nan would be duplicated, whereas None wouldn't
    unique_values_not_null = [val for val in unique_values_raw if not pd.isnull(val)]
    unique_values = [null_obj, *unique_values_not_null]

    if isinstance(obj, pd.Index):
        expected = pd.Index(unique_values, dtype=obj.dtype)
        if isinstance(obj.dtype, pd.DatetimeTZDtype):
            result = result.normalize()
            expected = expected.normalize()
        tm.assert_index_equal(result, expected, exact=True)
    else:
        expected = np.array(unique_values, dtype=obj.dtype)
        tm.assert_numpy_array_equal(result, expected)


def test_nunique(index_or_series_obj):
    # [FUNCTION DOC]
    # Verifies nunique without dropping missing values equals the number of unique values.
    obj = index_or_series_obj
    obj = np.repeat(obj, range(1, len(obj) + 1))
    expected = len(obj.unique())
    assert obj.nunique(dropna=False) == expected


@pytest.mark.parametrize("null_obj", [np.nan, None])
def test_nunique_null(null_obj, index_or_series_obj):
    # [FUNCTION DOC]
    # Parametrized over null marker. Skips objects without NA support and MultiIndex.
    # Replaces the first two values by the marker and verifies nunique counts distinct
    # non-missing values by default and including missing with dropna false; for a
    # categorical index the category count (plus one when including missing).
    obj = index_or_series_obj

    if not allow_na_ops(obj):
        pytest.skip("type doesn't allow for NA operations")
    elif isinstance(obj, pd.MultiIndex):
        pytest.skip(f"MultiIndex can't hold '{null_obj}'")

    obj = obj.copy(deep=True)
    values = obj._values
    values[0:2] = null_obj

    klass = type(obj)
    repeated_values = np.repeat(values, range(1, len(values) + 1))
    obj = klass(repeated_values, dtype=obj.dtype)

    if isinstance(obj, pd.CategoricalIndex):
        assert obj.nunique() == len(obj.categories)
        assert obj.nunique(dropna=False) == len(obj.categories) + 1
    else:
        num_unique_values = len(obj.unique())
        assert obj.nunique() == max(0, num_unique_values - 1)
        assert obj.nunique(dropna=False) == max(0, num_unique_values)


@pytest.mark.single_cpu
def test_unique_bad_unicode(index_or_series):
    # [FUNCTION DOC]
    # Regression test (#34550). Single CPU. Verifies unique of an object container with a
    # lone surrogate string repeated twice gives one element.
    # regression test for #34550
    uval = "\ud83d"  # smiley emoji

    obj = index_or_series([uval] * 2, dtype=object)
    result = obj.unique()

    if isinstance(obj, pd.Index):
        expected = pd.Index(["\ud83d"], dtype=object)
        tm.assert_index_equal(result, expected, exact=True)
    else:
        expected = np.array(["\ud83d"], dtype=object)
        tm.assert_numpy_array_equal(result, expected)


def test_nunique_dropna(dropna):
    # [FUNCTION DOC]
    # Regression test (GH37566). Uses the dropna fixture. Verifies nunique of a Series
    # with a string and several kinds of missing markers.
    # KNOWN ISSUE: the assertion is parsed as a conditional expression, so when dropna
    # is false it asserts the truthy constant 5 and checks nothing; the intended
    # comparison for that case is never made.
    # GH37566
    ser = pd.Series(["yes", "yes", pd.NA, np.nan, None, pd.NaT])
    res = ser.nunique(dropna)
    assert res == 1 if dropna else 5
