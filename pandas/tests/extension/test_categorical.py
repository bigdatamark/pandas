"""
This file contains a minimal set of tests for compliance with the extension
array interface test suite, and should contain no other tests.
The test suite for the full functionality of the array is located in
`pandas/tests/arrays/`.

The tests in this file are inherited from the BaseExtensionTests, and only
minimal tweaks should be applied to get the tests passing (by overwriting a
parent method).

Additional tests should either be added to one of the BaseExtensionTests
classes (if they are relevant for the extension interface for all dtypes), or
be added to the array-specific tests in `pandas/tests/arrays/`.

"""
# [MODULE DOC]
# Compliance tests for the categorical extension dtype and array, run through the shared
# extension-array test suite. Defines the fixtures the suite expects (dtype, data, missing data,
# sorting and grouping data) built from categorical arrays, a test class overriding suite
# tests where categorical behavior differs, and a 2D-compat class from the array-backed 2D suite.
# Full categorical tests live elsewhere. No import-time side effects.

import string

import numpy as np
import pytest

from pandas._config import using_string_dtype

import pandas as pd
from pandas import Categorical
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.tests.extension import base


def make_data(n: int):
    # [FUNCTION DOC]
    # HELPER: generates the random data for the data fixture. Parameter n is the length. Repeatedly
    # draws n random ASCII letters from a fixed-seed generator until the first two values differ,
    # then returns the numpy array of letters.
    # KNOWN ISSUE: the generator is recreated with the same seed on every iteration, so a rejected draw
    # would repeat identically and the loop could never end; also the comment claims the first two
    # must be non-null but only inequality is checked.
    while True:
        values = np.random.default_rng(2).choice(list(string.ascii_letters), size=n)
        # ensure we meet the requirements
        # 1. first two not null
        # 2. first and second are different
        if values[0] != values[1]:
            break
    return values


@pytest.fixture
def dtype():
    # [FUNCTION DOC]
    # HELPER: fixture returning a categorical dtype with no categories specified.
    return CategoricalDtype()


@pytest.fixture
def data():
    # [FUNCTION DOC]
    # HELPER: fixture returning a categorical array built from random letters of length ten. First
    # two elements are non-missing and differ.
    # KNOWN ISSUE: the docstring says length 100 but the array has ten elements.
    """Length-100 array for this type.

    * data[0] and data[1] should both be non missing
    * data[0] and data[1] should not be equal
    """
    return Categorical(make_data(10))


@pytest.fixture
def data_missing():
    # [FUNCTION DOC]
    # HELPER: fixture returning a two-element categorical array with a missing value first and the
    # valid category A second.
    """Length 2 array with [NA, Valid]"""
    return Categorical([np.nan, "A"])


@pytest.fixture
def data_for_sorting():
    # [FUNCTION DOC]
    # HELPER: fixture returning an ordered categorical with values A, B, C and category order C, A, B
    # (so the sorted order is C, A, B by category position).
    return Categorical(["A", "B", "C"], categories=["C", "A", "B"], ordered=True)


@pytest.fixture
def data_missing_for_sorting():
    # [FUNCTION DOC]
    # HELPER: fixture returning an ordered categorical with values A, missing, B and category order B, A.
    return Categorical(["A", None, "B"], categories=["B", "A"], ordered=True)


@pytest.fixture
def data_for_grouping():
    # [FUNCTION DOC]
    # HELPER: fixture returning an eight-element unordered categorical for group-by tests: a, a,
    # missing, missing, b, b, a, c.
    return Categorical(["a", "a", None, None, "b", "b", "a", "c"])


class TestCategorical(base.ExtensionTests):
    # [CLASS DOC]
    # Extension-suite test class for categoricals. Overrides contains, empty, scalar get, combine,
    # mapping, arithmetic with scalars, comparison helper, repr and groupby aggregation tests.
    def test_contains(self, data, data_missing):
        # [FUNCTION DOC]
        # Regression test for GH-37867. Missing value membership: a non-missing element is contained in
        # its data; the dtype's missing value is contained in the data with a missing value and not in data
        # without any. Other null-like objects are not in the missing-free data and (unless the string
        # dtype mode is on) are contained in the missing data.
        # GH-37867
        # na value handling in Categorical.__contains__ is deprecated.
        # See base.BaseInterFaceTests.test_contains for more details.

        na_value = data.dtype.na_value
        # ensure data without missing values
        data = data[~data.isna()]

        # first elements are non-missing
        assert data[0] in data
        assert data_missing[0] in data_missing

        # check the presence of na_value
        assert na_value in data_missing
        assert na_value not in data

        # Categoricals can contain other nan-likes than na_value
        for na_value_obj in tm.NULL_OBJECTS:
            if na_value_obj is na_value:
                continue
            assert na_value_obj not in data
            # this section suffers from super method
            if not using_string_dtype():
                assert na_value_obj in data_missing

    def test_empty(self, dtype):
        # [FUNCTION DOC]
        # Verifies the empty-array constructor on the array type from the dtype returns an instance of that
        # array type of length four whose dtype equals a categorical dtype with no categories (not the dtype
        # passed, which is uninitialized).
        cls = dtype.construct_array_type()
        result = cls._empty((4,), dtype=dtype)

        assert isinstance(result, cls)
        # the dtype we passed is not initialized, so will not match the
        #  dtype on our result.
        assert result.dtype == CategoricalDtype([])

    @pytest.mark.skip(reason="Backwards compatibility")
    def test_getitem_scalar(self, data):
        # [FUNCTION DOC]
        # Skipped for backwards compatibility: the scalar type of the categorical dtype is not a parent of
        # the elements, so the inherited scalar-get test is not run (only delegates to the parent).
        # CategoricalDtype.type isn't "correct" since it should
        # be a parent of the elements (object). But don't want
        # to break things by changing.
        super().test_getitem_scalar(data)

    def test_combine_add(self, data_repeated):
        # [FUNCTION DOC]
        # Regression test for GH 20825: combining two series of categoricals with element addition gives a
        # series of string concatenations; combining with a scalar adds that scalar to each element.
        # GH 20825
        # When adding categoricals in combine, result is a string
        orig_data1, orig_data2 = data_repeated(2)
        s1 = pd.Series(orig_data1)
        s2 = pd.Series(orig_data2)
        result = s1.combine(s2, lambda x1, x2: x1 + x2)
        expected = pd.Series(
            [a + b for (a, b) in zip(list(orig_data1), list(orig_data2), strict=True)]
        )
        tm.assert_series_equal(result, expected)

        val = s1.iloc[0]
        result = s1.combine(val, lambda x1, x2: x1 + x2)
        expected = pd.Series([a + val for a in list(orig_data1)])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("na_action", [None, "ignore"])
    def test_map(self, data, na_action):
        # [FUNCTION DOC]
        # Parametrized over missing-value action (none, ignore). Verifies mapping an identity function
        # over the categorical returns an equal array.
        result = data.map(lambda x: x, na_action=na_action)
        tm.assert_extension_array_equal(result, data)

    def test_arith_frame_with_scalar(self, data, all_arithmetic_operators, request):
        # [FUNCTION DOC]
        # Overrides the suite: scalar arithmetic on a frame. For the reflected modulus operator it applies
        # an expected-failure marker (reflected modulus is never called when a string is the first
        # argument), then delegates to the parent test.
        # frame & scalar
        op_name = all_arithmetic_operators
        if op_name == "__rmod__":
            request.applymarker(
                pytest.mark.xfail(
                    reason="rmod never called when string is first argument"
                )
            )
        super().test_arith_frame_with_scalar(data, op_name)

    def test_arith_series_with_scalar(self, data, all_arithmetic_operators, request):
        # [FUNCTION DOC]
        # Same as the frame variant but for Series scalar arithmetic: marks the reflected-modulus case
        # expected to fail, then delegates.
        op_name = all_arithmetic_operators
        if op_name == "__rmod__":
            request.applymarker(
                pytest.mark.xfail(
                    reason="rmod never called when string is first argument"
                )
            )
        super().test_arith_series_with_scalar(data, op_name)

    def _compare_other(self, ser: pd.Series, data, op, other):
        # [FUNCTION DOC]
        # HELPER: overrides the suite's comparison helper. For ordering operators (not equality or inequality)
        # it asserts that comparing the data with the other raises a type error saying unordered categoricals
        # can only compare equality or not; for equality and inequality it delegates to the parent helper
        # and returns its result. Parameters: the series, the data, the operator, and the other operand.
        op_name = f"__{op.__name__}__"
        if op_name not in ["__eq__", "__ne__"]:
            msg = "Unordered Categoricals can only compare equality or not"
            with pytest.raises(TypeError, match=msg):
                op(data, other)
        else:
            return super()._compare_other(ser, data, op, other)

    @pytest.mark.xfail(reason="Categorical overrides __repr__")
    @pytest.mark.parametrize("size", ["big", "small"])
    def test_array_repr(self, data, size):
        # [FUNCTION DOC]
        # Expected-to-fail (the categorical overrides its repr). Parametrized over big and small sizes;
        # delegates to the parent test.
        super().test_array_repr(data, size)

    @pytest.mark.xfail(reason="TBD")
    @pytest.mark.parametrize("as_index", [True, False])
    def test_groupby_extension_agg(self, as_index, data_for_grouping):
        # [FUNCTION DOC]
        # Expected-to-fail (reason to be determined). Parametrized over whether the group key is an index;
        # delegates to the parent test.
        super().test_groupby_extension_agg(as_index, data_for_grouping)


class Test2DCompat(base.NDArrayBacked2DTests):
    # [CLASS DOC]
    # Runs the shared two-dimensional compatibility tests for array-backed extension arrays against
    # categoricals, adding one override.
    def test_repr_2d(self, data):
        # [FUNCTION DOC]
        # Verifies the repr of the data reshaped into a single row and into a single column each contains
        # the categories line exactly once (a categorical repr does not include its class name, so the
        # generic test is special-cased).
        # Categorical __repr__ doesn't include "Categorical", so we need
        #  to special-case
        res = repr(data.reshape(1, -1))
        assert res.count("\nCategories") == 1

        res = repr(data.reshape(-1, 1))
        assert res.count("\nCategories") == 1
