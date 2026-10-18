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
# Compliance tests for the period extension dtype and array (daily frequencies one and two
# days), run through the shared extension-array test suite. Provides the fixtures the suite
# expects and a test class adjusting exceptions, supported operations, reduction checks and the
# diff test, plus a 2D-compat class. Full period array tests live elsewhere. No import-time side effects.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pandas._libs import (
    Period,
    iNaT,
)

from pandas.core.dtypes.dtypes import PeriodDtype

import pandas._testing as tm
from pandas.core.arrays import PeriodArray
from pandas.tests.extension import base

if TYPE_CHECKING:
    import pandas as pd


@pytest.fixture(params=["D", "2D"])
def dtype(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over daily frequency of one day and of two days; returns a period
    # dtype with that frequency.
    return PeriodDtype(freq=request.param)


@pytest.fixture
def data(dtype):
    # [FUNCTION DOC]
    # HELPER: fixture returning a period array of the ordinals 1970 through 1979 with the dtype.
    return PeriodArray(np.arange(1970, 1980), dtype=dtype)


@pytest.fixture
def data_for_sorting(dtype):
    # [FUNCTION DOC]
    # HELPER: fixture returning a period array of ordinals 2018, 2019, 2017 for sorting.
    return PeriodArray([2018, 2019, 2017], dtype=dtype)


@pytest.fixture
def data_missing(dtype):
    # [FUNCTION DOC]
    # HELPER: fixture returning a two-element period array: missing (integer NaT code) then 2017.
    return PeriodArray([iNaT, 2017], dtype=dtype)


@pytest.fixture
def data_missing_for_sorting(dtype):
    # [FUNCTION DOC]
    # HELPER: fixture returning a period array 2018, missing, 2017 for sorting with missing values.
    return PeriodArray([2018, iNaT, 2017], dtype=dtype)


@pytest.fixture
def data_for_grouping(dtype):
    # [FUNCTION DOC]
    # HELPER: fixture returning an eight-element array [B, B, NA, NA, A, A, B, C] for group-by
    # tests with A=2017, B=2018, C=2019 and NA the integer NaT code.
    B = 2018
    NA = iNaT
    A = 2017
    C = 2019
    return PeriodArray([B, B, NA, NA, A, A, B, C], dtype=dtype)


class TestPeriodArray(base.ExtensionTests):
    # [CLASS DOC]
    # Extension-suite test class for the period array.
    def _get_expected_exception(self, op_name, obj, other):
        # [FUNCTION DOC]
        # HELPER: returns no expected exception for subtraction and reflected subtraction; otherwise
        # defers to the parent implementation.
        if op_name in ("__sub__", "__rsub__"):
            return None
        return super()._get_expected_exception(op_name, obj, other)

    def _supports_accumulation(self, ser, op_name: str) -> bool:
        # [FUNCTION DOC]
        # HELPER: accumulation support: only running minimum and maximum.
        return op_name in ["cummin", "cummax"]

    def _supports_reduction(self, obj, op_name: str) -> bool:
        # [FUNCTION DOC]
        # HELPER: reduction support: only min, max and median.
        return op_name in ["min", "max", "median"]

    def check_reduce(self, ser: pd.Series, op_name: str, skipna: bool):
        # [FUNCTION DOC]
        # HELPER: overrides the reduction check. For median, compares the series result with the median of
        # the series cast to 64-bit integers converted to a period from the ordinal in the series'
        # frequency (almost equal); other reductions defer to the parent check.
        if op_name == "median":
            res_op = getattr(ser, op_name)

            alt = ser.astype("int64")

            exp_op = getattr(alt, op_name)
            result = res_op(skipna=skipna)
            expected = exp_op(skipna=skipna)
            # error: Item "dtype[Any]" of "dtype[Any] | ExtensionDtype" has no
            # attribute "freq"
            freq = ser.dtype.freq  # type: ignore[union-attr]
            expected = Period._from_ordinal(int(expected), freq=freq)
            tm.assert_almost_equal(result, expected)

        else:
            return super().check_reduce(ser, op_name, skipna)

    @pytest.mark.parametrize("periods", [1, -2])
    # NOTE: RuntimeWarning on Windows(non-ARM) platforms (in CI)
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_diff(self, request, data, periods):
        # [FUNCTION DOC]
        # Parametrized over periods one and negative two. Delegates to the parent diff test while ignoring
        # runtime warnings (seen on Windows, non-ARM, in CI). The request fixture is unused.
        super().test_diff(data, periods)

    @pytest.mark.parametrize("na_action", [None, "ignore"])
    def test_map(self, data, na_action):
        # [FUNCTION DOC]
        # Parametrized over missing-value action (none, ignore). Verifies mapping the identity function
        # gives an equal extension array.
        result = data.map(lambda x: x, na_action=na_action)
        tm.assert_extension_array_equal(result, data)


class Test2DCompat(base.NDArrayBacked2DTests):
    # [CLASS DOC]
    # Runs the shared two-dimensional compatibility tests for array-backed extension arrays against
    # the period array without changes.
    pass
