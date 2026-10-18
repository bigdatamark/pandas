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
# Compliance tests for the interval extension dtype and array, run through the shared
# extension-array test suite. Provides the fixtures the suite expects and a test class overriding
# inherited tests that need expected failures, warning filters or different exceptions, plus one
# extra test for filling with a non-scalar. Full interval array tests live elsewhere. No
# import-time side effects.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pandas.core.dtypes.dtypes import IntervalDtype

from pandas import Interval
from pandas.core.arrays import IntervalArray
from pandas.tests.extension import base

if TYPE_CHECKING:
    import pandas as pd


def make_data(n: int):
    # [FUNCTION DOC]
    # HELPER: builds test data for the data fixture. Parameter n is the length. Draws uniform values
    # from a fixed-seed generator, takes their cumulative sum as left edges and adds a second draw
    # (the same sequence, since the generator is re-seeded) for right edges, and returns a list of
    # n Interval objects.
    left_array = np.random.default_rng(2).uniform(size=n).cumsum()
    right_array = left_array + np.random.default_rng(2).uniform(size=n)
    return [
        Interval(left, right)
        for left, right in zip(left_array, right_array, strict=True)
    ]


@pytest.fixture
def dtype():
    # [FUNCTION DOC]
    # HELPER: fixture returning the default interval dtype.
    return IntervalDtype()


@pytest.fixture
def data():
    # [FUNCTION DOC]
    # HELPER: fixture returning an interval array of length ten from the random data helper.
    """Length-10 IntervalArray for semantics test."""
    return IntervalArray(make_data(10))


@pytest.fixture
def data_missing():
    # [FUNCTION DOC]
    # HELPER: fixture returning a two-element interval array: missing first, then the interval from
    # zero to one.
    """Length 2 array with [NA, Valid]"""
    return IntervalArray.from_tuples([None, (0, 1)])


@pytest.fixture
def data_for_twos():
    # [FUNCTION DOC]
    # HELPER: overrides the suite fixture; skips any test using it because interval is not numeric.
    pytest.skip("Interval is not a numeric dtype")


@pytest.fixture
def data_for_sorting():
    # [FUNCTION DOC]
    # HELPER: fixture returning an interval array of three intervals (1,2), (2,3), (0,1) for sorting.
    return IntervalArray.from_tuples([(1, 2), (2, 3), (0, 1)])


@pytest.fixture
def data_missing_for_sorting():
    # [FUNCTION DOC]
    # HELPER: fixture returning an interval array (1,2), missing, (0,1) for sorting with missing values.
    return IntervalArray.from_tuples([(1, 2), None, (0, 1)])


@pytest.fixture
def data_for_grouping():
    # [FUNCTION DOC]
    # HELPER: fixture returning an eight-element array [B, B, NA, NA, A, A, B, C] for group-by tests,
    # with A=(0,1), B=(1,2), C=(2,3).
    a = (0, 1)
    b = (1, 2)
    c = (2, 3)
    return IntervalArray.from_tuples([b, b, None, None, a, a, b, c])


class TestIntervalArray(base.ExtensionTests):
    # [CLASS DOC]
    # Extension-suite test class for the interval array. Class attribute divmod_exc is set to type
    # error, the exception expected for divmod. Overrides reduction support and several fill, hash and
    # repr-related tests.
    divmod_exc = TypeError

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        # [FUNCTION DOC]
        # HELPER: reports whether a reduction is supported: only min and max.
        return op_name in ["min", "max"]

    def test_fillna_limit_frame(self, data_missing):
        # [FUNCTION DOC]
        # Regression test for GH#58001: calling the inherited frame fill-with-limit test is expected to
        # raise a value error saying the limit must be none.
        # GH#58001
        with pytest.raises(ValueError, match="limit must be None"):
            super().test_fillna_limit_frame(data_missing)

    def test_fillna_limit_series(self, data_missing):
        # [FUNCTION DOC]
        # Regression test for GH#58001: expected to raise a value error saying the limit must be none.
        # KNOWN ISSUE: the series test calls the inherited frame variant of the fill-limit test, not the
        # series variant, so the series path is not exercised.
        # GH#58001
        with pytest.raises(ValueError, match="limit must be None"):
            super().test_fillna_limit_frame(data_missing)

    @pytest.mark.xfail(
        reason="Raises with incorrect message bc it disallows *all* listlikes "
        "instead of just wrong-length listlikes"
    )
    def test_fillna_length_mismatch(self, data_missing):
        # [FUNCTION DOC]
        # Expected-to-fail: it raises with an incorrect message because all list-likes are disallowed, not
        # just wrong-length ones. Delegates to the parent test.
        super().test_fillna_length_mismatch(data_missing)

    @pytest.mark.xfail(reason="copy=False is not Implemented")
    def test_fillna_readonly(self, data_missing):
        # [FUNCTION DOC]
        # Expected-to-fail because the no-copy option is not implemented. Delegates to the parent test.
        super().test_fillna_readonly(data_missing)

    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in cast:RuntimeWarning"
    )
    def test_hash_pandas_object(self, data):
        # [FUNCTION DOC]
        # Delegates to the parent hash test while ignoring a runtime warning about invalid values
        # encountered in a cast.
        super().test_hash_pandas_object(data)

    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in cast:RuntimeWarning"
    )
    def test_hash_pandas_object_works(self, data, as_frame):
        # [FUNCTION DOC]
        # Delegates to the parent hash-works test (parametrized by the as_frame fixture) while ignoring the
        # invalid-value-in-cast runtime warning.
        super().test_hash_pandas_object_works(data, as_frame)

    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in cast:RuntimeWarning"
    )
    @pytest.mark.parametrize("engine", ["c", "python"])
    def test_EA_types(self, engine, data, request):
        # [FUNCTION DOC]
        # Parametrized over the c and python parser engines. Delegates to the parent test of reading
        # extension types, ignoring the invalid-value-in-cast runtime warning.
        super().test_EA_types(engine, data, request)

    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in cast:RuntimeWarning"
    )
    def test_astype_str(self, data):
        # [FUNCTION DOC]
        # Delegates to the parent test of casting to string while ignoring the invalid-value-in-cast
        # runtime warning.
        super().test_astype_str(data)

    @pytest.mark.xfail(
        reason="Test is invalid for IntervalDtype, needs to be adapted for "
        "this dtype with an index with index._index_as_unique."
    )
    def test_loc_setitem_with_expansion_preserves_ea_index_dtype(self, data):
        # [FUNCTION DOC]
        # Expected-to-fail: the inherited test is invalid for the interval dtype and needs adapting to
        # an index that is unique. Delegates to the parent test.
        super().test_loc_setitem_with_expansion_preserves_ea_index_dtype(data)


# TODO: either belongs in tests.arrays.interval or move into base tests.
def test_fillna_non_scalar_raises(data_missing):
    # [FUNCTION DOC]
    # Verifies filling missing values with a list (not a scalar) raises a type error saying only
    # Interval objects and NA can be inserted into an interval array. A source TODO says it belongs
    # in the array tests or the base tests.
    msg = "can only insert Interval objects and NA into an IntervalArray"
    with pytest.raises(TypeError, match=msg):
        data_missing.fillna([1, 1])
