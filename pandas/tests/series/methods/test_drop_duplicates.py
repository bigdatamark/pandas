# [MODULE DOC]
# Tests for Series.drop_duplicates and Series.duplicated: keep options
# (first, last, False), in-place operation, boolean, categorical (ordered or
# not, with unused categories, with NA), ignore_index and pyarrow-backed data.
# Uses the any_numpy_dtype, keep, ordered and nulls_fixture fixtures supplied
# by the shared conftest files. Defines a test class with its own dtype fixtures.
# No import-time side effects.

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Categorical,
    Series,
)
import pandas._testing as tm


@pytest.mark.parametrize(
    "keep, expected",
    [
        ("first", Series([False, False, False, False, True, True, False])),
        ("last", Series([False, True, True, False, False, False, False])),
        (False, Series([False, True, True, False, True, True, False])),
    ],
)
def test_drop_duplicates(any_numpy_dtype, keep, expected):
    # [FUNCTION DOC]
    # Parametrized over the keep option with the expected duplicated mask, and over
    # numpy dtypes (bool is skipped; covered by a separate test). For the seven-element
    # Series, duplicated gives the mask, drop_duplicates returns the elements where the
    # mask is false, and the in-place form returns None and mutates the copy to the same result.
    tc = Series([1, 0, 3, 5, 3, 0, 4], dtype=np.dtype(any_numpy_dtype))

    if tc.dtype == "bool":
        pytest.skip("tested separately in test_drop_duplicates_bool")

    tm.assert_series_equal(tc.duplicated(keep=keep), expected)
    tm.assert_series_equal(tc.drop_duplicates(keep=keep), tc[~expected])
    sc = tc.copy()
    return_value = sc.drop_duplicates(keep=keep, inplace=True)
    assert return_value is None
    tm.assert_series_equal(sc, tc[~expected])


@pytest.mark.parametrize(
    "keep, expected",
    [
        ("first", [False, False, True, True]),
        ("last", [True, True, False, False]),
        (False, [True, True, True, True]),
    ],
)
def test_drop_duplicates_bool(keep, expected):
    # [FUNCTION DOC]
    # Same checks as the numeric test for a boolean Series with two True and two False
    # values, for each keep option; the in-place form returns None.
    tc = Series([True, False, True, False])
    expected = Series(expected)
    tm.assert_series_equal(tc.duplicated(keep=keep), expected)
    tm.assert_series_equal(tc.drop_duplicates(keep=keep), tc[~expected])
    sc = tc.copy()
    return_value = sc.drop_duplicates(keep=keep, inplace=True)
    tm.assert_series_equal(sc, tc[~expected])
    assert return_value is None


@pytest.mark.parametrize("values", [[], list(range(5))])
def test_drop_duplicates_no_duplicates(any_numpy_dtype, keep, values):
    # [FUNCTION DOC]
    # Parametrized over empty and 0 to 4 values, numpy dtypes and the keep fixture.
    # For a Series with no duplicates duplicated is all False and drop_duplicates
    # returns an equal Series that is a different object. For bool dtype only the first two
    # values (false and true) are used since more would repeat.
    tc = Series(values, dtype=np.dtype(any_numpy_dtype))
    expected = Series([False] * len(tc), dtype="bool")

    if tc.dtype == "bool":
        # 0 -> False and 1-> True
        # any other value would be duplicated
        tc = tc[:2]
        expected = expected[:2]

    tm.assert_series_equal(tc.duplicated(keep=keep), expected)

    result_dropped = tc.drop_duplicates(keep=keep)
    tm.assert_series_equal(result_dropped, tc)

    # validate shallow copy
    assert result_dropped is not tc


class TestSeriesDropDuplicates:
    # [CLASS DOC]
    # Tests of drop_duplicates and duplicated for categorical Series and
    # miscellaneous cases. Defines fixtures: a dtype fixture and two categorical
    # Series fixtures.
    @pytest.fixture(
        params=["int_", "uint", "float64", "str_", "timedelta64[h]", "datetime64[D]"]
    )
    def dtype(self, request):
        # [FUNCTION DOC]
        # HELPER: fixture parametrized over a set of dtype names (integer, unsigned,
        # float, string, timedelta in hours, datetime in days) used to build categories.
        """
        Fixture that provides different data types for testing.
        """
        return request.param

    @pytest.fixture
    def cat_series_unused_category(self, dtype, ordered):
        # [FUNCTION DOC]
        # HELPER: fixture returning a categorical Series of values 1, 2, 3, 3 of the given
        # dtype with categories 1 to 5 (two unused) and the ordered setting from the
        # ordered fixture.
        """
        Fixture that creates a Categorical Series with some unused categories.
        """
        # Test case 1
        cat_array = np.array([1, 2, 3, 4, 5], dtype=np.dtype(dtype))

        input1 = np.array([1, 2, 3, 3], dtype=np.dtype(dtype))
        cat = Categorical(input1, categories=cat_array, ordered=ordered)
        tc1 = Series(cat)
        return tc1

    def test_drop_duplicates_categorical_non_bool(self, cat_series_unused_category):
        # [FUNCTION DOC]
        # For the categorical with unused categories, the default keep marks only the last
        # element as duplicated; drop_duplicates and the in-place variant remove it.
        tc1 = cat_series_unused_category

        expected = Series([False, False, False, True])

        result = tc1.duplicated()
        tm.assert_series_equal(result, expected)

        result = tc1.drop_duplicates()
        tm.assert_series_equal(result, tc1[~expected])

        sc = tc1.copy()
        return_value = sc.drop_duplicates(inplace=True)
        assert return_value is None
        tm.assert_series_equal(sc, tc1[~expected])

    def test_drop_duplicates_categorical_non_bool_keeplast(
        self, cat_series_unused_category
    ):
        # [FUNCTION DOC]
        # Same as above with keep set to last: the duplicate flagged is the third element.
        tc1 = cat_series_unused_category

        expected = Series([False, False, True, False])

        result = tc1.duplicated(keep="last")
        tm.assert_series_equal(result, expected)

        result = tc1.drop_duplicates(keep="last")
        tm.assert_series_equal(result, tc1[~expected])

        sc = tc1.copy()
        return_value = sc.drop_duplicates(keep="last", inplace=True)
        assert return_value is None
        tm.assert_series_equal(sc, tc1[~expected])

    def test_drop_duplicates_categorical_non_bool_keepfalse(
        self, cat_series_unused_category
    ):
        # [FUNCTION DOC]
        # Same as above with keep set to False: both of the repeated elements are flagged.
        tc1 = cat_series_unused_category

        expected = Series([False, False, True, True])

        result = tc1.duplicated(keep=False)
        tm.assert_series_equal(result, expected)

        result = tc1.drop_duplicates(keep=False)
        tm.assert_series_equal(result, tc1[~expected])

        sc = tc1.copy()
        return_value = sc.drop_duplicates(keep=False, inplace=True)
        assert return_value is None
        tm.assert_series_equal(sc, tc1[~expected])

    @pytest.fixture
    def cat_series(self, dtype, ordered):
        # [FUNCTION DOC]
        # HELPER: fixture returning a categorical Series of seven values (1, 2, 3, 5, 3, 2, 4)
        # of the given dtype with categories 1 to 5 and the ordered fixture's setting.
        """
        Fixture that creates a Categorical Series with no unused categories.
        """
        cat_array = np.array([1, 2, 3, 4, 5], dtype=np.dtype(dtype))

        input2 = np.array([1, 2, 3, 5, 3, 2, 4], dtype=np.dtype(dtype))
        cat = Categorical(input2, categories=cat_array, ordered=ordered)
        tc2 = Series(cat)
        return tc2

    def test_drop_duplicates_categorical_non_bool2(self, cat_series):
        # [FUNCTION DOC]
        # Default keep on the seven-element categorical: elements five and six are
        # flagged as duplicates; drop_duplicates and in-place removal agree.
        tc2 = cat_series

        expected = Series([False, False, False, False, True, True, False])

        result = tc2.duplicated()
        tm.assert_series_equal(result, expected)

        result = tc2.drop_duplicates()
        tm.assert_series_equal(result, tc2[~expected])

        sc = tc2.copy()
        return_value = sc.drop_duplicates(inplace=True)
        assert return_value is None
        tm.assert_series_equal(sc, tc2[~expected])

    def test_drop_duplicates_categorical_non_bool2_keeplast(self, cat_series):
        # [FUNCTION DOC]
        # Same with keep set to last: the second and third elements are flagged.
        tc2 = cat_series

        expected = Series([False, True, True, False, False, False, False])

        result = tc2.duplicated(keep="last")
        tm.assert_series_equal(result, expected)

        result = tc2.drop_duplicates(keep="last")
        tm.assert_series_equal(result, tc2[~expected])

        sc = tc2.copy()
        return_value = sc.drop_duplicates(keep="last", inplace=True)
        assert return_value is None
        tm.assert_series_equal(sc, tc2[~expected])

    def test_drop_duplicates_categorical_non_bool2_keepfalse(self, cat_series):
        # [FUNCTION DOC]
        # Same with keep set to False: every repeated value (second, third, fifth and
        # sixth elements) is flagged.
        tc2 = cat_series

        expected = Series([False, True, True, False, True, True, False])

        result = tc2.duplicated(keep=False)
        tm.assert_series_equal(result, expected)

        result = tc2.drop_duplicates(keep=False)
        tm.assert_series_equal(result, tc2[~expected])

        sc = tc2.copy()
        return_value = sc.drop_duplicates(keep=False, inplace=True)
        assert return_value is None
        tm.assert_series_equal(sc, tc2[~expected])

    def test_drop_duplicates_categorical_bool(self, ordered):
        # [FUNCTION DOC]
        # Categorical of booleans with ordered and unordered variants (ordered fixture):
        # for first, last and False keeps, the duplicated mask and results of
        # drop_duplicates (also in place, returning None) match the expected masks.
        tc = Series(
            Categorical(
                [True, False, True, False], categories=[True, False], ordered=ordered
            )
        )

        expected = Series([False, False, True, True])
        tm.assert_series_equal(tc.duplicated(), expected)
        tm.assert_series_equal(tc.drop_duplicates(), tc[~expected])
        sc = tc.copy()
        return_value = sc.drop_duplicates(inplace=True)
        assert return_value is None
        tm.assert_series_equal(sc, tc[~expected])

        expected = Series([True, True, False, False])
        tm.assert_series_equal(tc.duplicated(keep="last"), expected)
        tm.assert_series_equal(tc.drop_duplicates(keep="last"), tc[~expected])
        sc = tc.copy()
        return_value = sc.drop_duplicates(keep="last", inplace=True)
        assert return_value is None
        tm.assert_series_equal(sc, tc[~expected])

        expected = Series([True, True, True, True])
        tm.assert_series_equal(tc.duplicated(keep=False), expected)
        tm.assert_series_equal(tc.drop_duplicates(keep=False), tc[~expected])
        sc = tc.copy()
        return_value = sc.drop_duplicates(keep=False, inplace=True)
        assert return_value is None
        tm.assert_series_equal(sc, tc[~expected])

    def test_drop_duplicates_categorical_bool_na(self, nulls_fixture):
        # [FUNCTION DOC]
        # Regression test (GH#44351), parametrized over null scalars: a categorical with a
        # missing value does not treat the missing value as a duplicate; it is kept at
        # its position with the labels 0, 1 and 4.
        # GH#44351
        ser = Series(
            Categorical(
                [True, False, True, False, nulls_fixture],
                categories=[True, False],
                ordered=True,
            )
        )
        result = ser.drop_duplicates()
        expected = Series(
            Categorical([True, False, np.nan], categories=[True, False], ordered=True),
            index=[0, 1, 4],
        )
        tm.assert_series_equal(result, expected)

    def test_drop_duplicates_ignore_index(self):
        # [FUNCTION DOC]
        # Regression test (GH#48304): the ignore_index option renumbers the result from
        # zero.
        # GH#48304
        ser = Series([1, 2, 2, 3])
        result = ser.drop_duplicates(ignore_index=True)
        expected = Series([1, 2, 3])
        tm.assert_series_equal(result, expected)

    def test_duplicated_arrow_dtype(self):
        # [FUNCTION DOC]
        # Requires pyarrow. drop_duplicates on a pyarrow boolean Series with a missing value
        # keeps the first occurrences and the missing value.
        # KNOWN ISSUE: the test name says duplicated but the test exercises
        # drop_duplicates.
        pytest.importorskip("pyarrow")
        ser = Series([True, False, None, False], dtype="bool[pyarrow]")
        result = ser.drop_duplicates()
        expected = Series([True, False, None], dtype="bool[pyarrow]")
        tm.assert_series_equal(result, expected)

    def test_drop_duplicates_arrow_strings(self):
        # [FUNCTION DOC]
        # Regression test (GH#54904), requires pyarrow: drop_duplicates on a pyarrow
        # string Series of two equal values keeps one.
        # GH#54904
        pa = pytest.importorskip("pyarrow")
        ser = Series(["a", "a"], dtype=pd.ArrowDtype(pa.string()))
        result = ser.drop_duplicates()
        expecetd = Series(["a"], dtype=pd.ArrowDtype(pa.string()))
        tm.assert_series_equal(result, expecetd)
