# [MODULE DOC]
# Tests that the Series method converting to a Python list yields elements of the expected built-in or pandas scalar type for many dtypes (numpy, nullable, string, interval, period, timedelta, datetime, and pyarrow-backed ones that are skipped when pyarrow is missing).
# No import-time side effects.

import pytest

import pandas.util._test_decorators as td

from pandas import (
    Interval,
    Period,
    Series,
    Timedelta,
    Timestamp,
)


@pytest.mark.parametrize(
    "values, dtype, expected_dtype",
    (
        ([1], "int64", int),
        ([1], "Int64", int),
        ([1.0], "float64", float),
        ([1.0], "Float64", float),
        (["abc"], "object", str),
        (["abc"], "string", str),
        ([Interval(1, 3)], "interval", Interval),
        ([Period("2000-01-01", "D")], "period[D]", Period),
        ([Timedelta(days=1)], "timedelta64[ns]", Timedelta),
        ([Timestamp("2000-01-01")], "datetime64[ns]", Timestamp),
        pytest.param([1], "int64[pyarrow]", int, marks=td.skip_if_no("pyarrow")),
        pytest.param([1.0], "float64[pyarrow]", float, marks=td.skip_if_no("pyarrow")),
        pytest.param(["abc"], "string[pyarrow]", str, marks=td.skip_if_no("pyarrow")),
    ),
)
def test_tolist_scalar_dtype(values, dtype, expected_dtype):
    # [FUNCTION DOC]
    # Verifies that, for a one-element Series built from the given values and dtype, the first element of the converted list has exactly the expected scalar type (issue 49890). Parametrized over (values, dtype, expected type) triples, e.g. integers give Python int, floats give float, strings give str, intervals, periods, timedeltas and timestamps give their pandas scalar classes. The type comparison is exact equality, not instance checking.
    # GH49890
    ser = Series(values, dtype=dtype)
    result_dtype = type(ser.tolist()[0])
    assert result_dtype == expected_dtype
