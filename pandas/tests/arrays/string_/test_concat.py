# [MODULE DOC]
# Tests for the resulting dtype and values when concatenating string-dtype
# Series with each other, with object Series, and the common-type resolution
# between StringDtype and numpy dtypes. Uses the string_dtype_arguments fixture
# supplied by the test suite conftest. No import-time side effects.

import numpy as np
import pytest

from pandas.compat import HAS_PYARROW

from pandas.core.dtypes.cast import find_common_type

import pandas as pd
import pandas._testing as tm
from pandas.util.version import Version


@pytest.mark.parametrize(
    "to_concat_dtypes, result_dtype",
    [
        # same types
        ([("pyarrow", pd.NA), ("pyarrow", pd.NA)], ("pyarrow", pd.NA)),
        ([("pyarrow", np.nan), ("pyarrow", np.nan)], ("pyarrow", np.nan)),
        ([("python", pd.NA), ("python", pd.NA)], ("python", pd.NA)),
        ([("python", np.nan), ("python", np.nan)], ("python", np.nan)),
        # pyarrow preference
        ([("pyarrow", pd.NA), ("python", pd.NA)], ("pyarrow", pd.NA)),
        # NA preference
        ([("python", pd.NA), ("python", np.nan)], ("python", pd.NA)),
    ],
)
def test_concat_series(request, to_concat_dtypes, result_dtype):
    # [FUNCTION DOC]
    # Parametrized over lists of (storage, NA value) pairs for the inputs and the
    # (storage, NA value) of the expected result: same types stay the same, pyarrow
    # storage wins over python storage, and the pandas NA marker wins over the
    # NaN marker. Skips when pyarrow is needed but not installed. Builds a Series of
    # two strings and a missing value per input, concatenates ignoring the index,
    # and verifies the values and expected string dtype. The request argument is unused.
    # KNOWN ISSUE: the second check is commented as showing order does not matter,
    # but it slices with step one, which keeps the same order rather than reversing it.
    if any(storage == "pyarrow" for storage, _ in to_concat_dtypes) and not HAS_PYARROW:
        pytest.skip("Could not import 'pyarrow'")

    ser_list = [
        pd.Series(["a", "b", None], dtype=pd.StringDtype(storage, na_value))
        for storage, na_value in to_concat_dtypes
    ]

    result = pd.concat(ser_list, ignore_index=True)
    expected = pd.Series(
        ["a", "b", None, "a", "b", None], dtype=pd.StringDtype(*result_dtype)
    )
    tm.assert_series_equal(result, expected)

    # order doesn't matter for result
    result = pd.concat(ser_list[::1], ignore_index=True)
    tm.assert_series_equal(result, expected)


def test_concat_with_object(string_dtype_arguments):
    # [FUNCTION DOC]
    # Verifies that concatenating a string-dtype Series with an object-dtype Series
    # holding strings results in object dtype, because common-type resolution cannot
    # inspect values.
    # _get_common_dtype cannot inspect values, so object dtype with strings still
    # results in object dtype
    result = pd.concat(
        [
            pd.Series(["a", "b", None], dtype=pd.StringDtype(*string_dtype_arguments)),
            pd.Series(["a", "b", None], dtype=object),
        ]
    )
    assert result.dtype == np.dtype("object")


def test_concat_with_numpy(string_dtype_arguments):
    # [FUNCTION DOC]
    # Verifies the common-type finder: the string dtype combined with a numpy
    # unicode dtype (any width, either order) yields the string dtype; with a numpy
    # bytes or integer dtype it yields object. With numpy 2 or newer, the numpy
    # variable-width string dtype combined with the string dtype yields the string
    # dtype in either order.
    # common type with a numpy string dtype always preserves the pandas string dtype
    dtype = pd.StringDtype(*string_dtype_arguments)
    assert find_common_type([dtype, np.dtype("U")]) == dtype
    assert find_common_type([np.dtype("U"), dtype]) == dtype
    assert find_common_type([dtype, np.dtype("U10")]) == dtype
    assert find_common_type([np.dtype("U10"), dtype]) == dtype

    # with any other numpy dtype -> object
    assert find_common_type([dtype, np.dtype("S")]) == np.dtype("object")
    assert find_common_type([dtype, np.dtype("int64")]) == np.dtype("object")

    if Version(np.__version__) >= Version("2"):
        assert find_common_type([dtype, np.dtypes.StringDType()]) == dtype
        assert find_common_type([np.dtypes.StringDType(), dtype]) == dtype
