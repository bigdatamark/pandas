# [MODULE DOC]
# Tests for the testing-utility function that asserts an attribute is equal on two objects, specifically its treatment of the different kinds of missing values. Uses fixtures supplying missing-value scalars from the shared test configuration.

from types import SimpleNamespace

import pytest

from pandas.core.dtypes.common import is_float

import pandas._testing as tm


def test_assert_attr_equal(nulls_fixture):
    # [FUNCTION DOC]
    # For any missing-value scalar from the fixture stored as an attribute on one namespace object, asserting the attribute equal between the object and itself passes.
    obj = SimpleNamespace()
    obj.na_value = nulls_fixture
    tm.assert_attr_equal("na_value", obj, obj)


def test_assert_attr_equal_different_nulls(nulls_fixture, nulls_fixture2):
    # [FUNCTION DOC]
    # Parametrized over pairs of missing-value scalars. Two objects each carry one scalar. Expected: passes if both scalars are the same object, or both are floats (a float NaN and a numpy float NaN are equivalent), or both have the same type (for example a decimal NaN); otherwise the assertion raises an assertion error saying the attribute is different.
    obj = SimpleNamespace()
    obj.na_value = nulls_fixture

    obj2 = SimpleNamespace()
    obj2.na_value = nulls_fixture2

    if nulls_fixture is nulls_fixture2:
        tm.assert_attr_equal("na_value", obj, obj2)
    elif is_float(nulls_fixture) and is_float(nulls_fixture2):
        # we consider float("nan") and np.float64("nan") to be equivalent
        tm.assert_attr_equal("na_value", obj, obj2)
    elif type(nulls_fixture) is type(nulls_fixture2):
        # e.g. Decimal("NaN")
        tm.assert_attr_equal("na_value", obj, obj2)
    else:
        with pytest.raises(AssertionError, match='"na_value" are different'):
            tm.assert_attr_equal("na_value", obj, obj2)
