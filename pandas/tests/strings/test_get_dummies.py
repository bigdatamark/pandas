# [MODULE DOC]
# Tests for the string accessor method that splits strings on a separator and returns
# indicator (dummy) columns, for Series and Index, including the dtype argument.
import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    _testing as tm,
)


def test_get_dummies(any_string_dtype):
    # [FUNCTION DOC]
    # Uses the parametrized string dtype fixture. Verifies a Series with separator-split
    # strings and a missing value yields an integer indicator frame with sorted columns
    # and an all-zero row for the missing value; also checks a non-string element (a number)
    # is treated as its text form and sorted into the columns.
    s = Series(["a|b", "a|c", np.nan], dtype=any_string_dtype)
    result = s.str.get_dummies("|")
    expected = DataFrame([[1, 1, 0], [1, 0, 1], [0, 0, 0]], columns=list("abc"))
    tm.assert_frame_equal(result, expected)

    s = Series(["a;b", "a", 7], dtype=any_string_dtype)
    result = s.str.get_dummies(";")
    expected = DataFrame([[0, 1, 1], [0, 1, 0], [1, 0, 0]], columns=list("7ab"))
    tm.assert_frame_equal(result, expected)


def test_get_dummies_index():
    # [FUNCTION DOC]
    # Regression test for GitHub issues. Verifies calling the method on an Index returns
    # a MultiIndex of indicator tuples with the unique tokens as level names.
    # GH9980, GH8028
    idx = Index(["a|b", "a|c", "b|c"])
    result = idx.str.get_dummies("|")

    expected = MultiIndex.from_tuples(
        [(1, 1, 0), (1, 0, 1), (0, 1, 1)], names=("a", "b", "c")
    )
    tm.assert_index_equal(result, expected)


# GH#47872
@pytest.mark.parametrize(
    "dtype",
    [
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        bool,
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "boolean",
    ],
)
def test_get_dummies_with_dtype(any_string_dtype, dtype):
    # [FUNCTION DOC]
    # Parametrized over numeric, boolean and nullable dtypes, and the string dtype fixture.
    # Verifies the result frame has the requested dtype for every column with the expected
    # indicator values.
    s = Series(["a|b", "a|c", np.nan], dtype=any_string_dtype)
    result = s.str.get_dummies("|", dtype=dtype)
    expected = DataFrame(
        [[1, 1, 0], [1, 0, 1], [0, 0, 0]], columns=list("abc"), dtype=dtype
    )
    tm.assert_frame_equal(result, expected)


# GH#47872
@td.skip_if_no("pyarrow")
@pytest.mark.parametrize(
    "dtype",
    [
        "int8[pyarrow]",
        "uint8[pyarrow]",
        "int16[pyarrow]",
        "uint16[pyarrow]",
        "int32[pyarrow]",
        "uint32[pyarrow]",
        "int64[pyarrow]",
        "uint64[pyarrow]",
        "bool[pyarrow]",
    ],
)
def test_get_dummies_with_pyarrow_dtype(any_string_dtype, dtype):
    # [FUNCTION DOC]
    # Skipped if pyarrow is missing. Parametrized over pyarrow-backed numeric and boolean
    # dtypes. Verifies the result frame uses the requested dtype with the same indicator values.
    s = Series(["a|b", "a|c", np.nan], dtype=any_string_dtype)
    result = s.str.get_dummies("|", dtype=dtype)
    expected = DataFrame(
        [[1, 1, 0], [1, 0, 1], [0, 0, 0]],
        columns=list("abc"),
        dtype=dtype,
    )
    tm.assert_frame_equal(result, expected)


# GH#47872
def test_get_dummies_with_str_dtype(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies that a text dtype or a datetime dtype for the dtype argument raises a value
    # error saying only numeric or boolean dtypes are supported.
    s = Series(["a|b", "a|c", np.nan], dtype=any_string_dtype)

    msg = "Only numeric or boolean dtypes are supported for 'dtype'"
    with pytest.raises(ValueError, match=msg):
        s.str.get_dummies("|", dtype=str)

    with pytest.raises(ValueError, match=msg):
        s.str.get_dummies("|", dtype="datetime64[ns]")
