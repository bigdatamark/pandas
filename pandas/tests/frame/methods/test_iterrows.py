# [MODULE DOC]
# Regression test for DataFrame iterrows with string and timedelta columns.

from pandas import (
    DataFrame,
    Timedelta,
)


def test_no_overflow_of_freq_and_time_in_dataframe():
    # [FUNCTION DOC]
    # GH 35665. Builds a frame with a string column and a timedelta column and iterates rows. Verifies every row series has object dtype (no overflow or coercion of the mixed values).
    # GH 35665
    df = DataFrame(
        {
            "some_string": ["2222Y3"],
            "time": [Timedelta("0 days 00:00:00.990000")],
        }
    )
    for _, row in df.iterrows():
        assert row.dtype == "object"
