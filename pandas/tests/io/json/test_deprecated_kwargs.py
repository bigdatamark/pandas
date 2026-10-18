"""
Tests for the deprecated keyword arguments for `read_json`.
"""

# [MODULE DOC]
# Tests that read_json accepts the currently supported keyword arguments
# without emitting any warning. Despite the name, only the non-deprecated
# usage is exercised here. Depends on the testing helpers module.

from io import StringIO

import pandas as pd
import pandas._testing as tm

from pandas.io.json import read_json


def test_good_kwargs():
    # [FUNCTION DOC]
    # Builds a small two-column integer frame and, for each of the split, columns
    # and index orientations, serializes it to a JSON string, wraps it in a text
    # buffer and reads it back with the same orientation. All of this runs inside a
    # check that no warning is produced, and each round trip must equal the
    # original frame.
    df = pd.DataFrame({"A": [2, 4, 6], "B": [3, 6, 9]}, index=[0, 1, 2])

    with tm.assert_produces_warning(None):
        data1 = StringIO(df.to_json(orient="split"))
        tm.assert_frame_equal(df, read_json(data1, orient="split"))
        data2 = StringIO(df.to_json(orient="columns"))
        tm.assert_frame_equal(df, read_json(data2, orient="columns"))
        data3 = StringIO(df.to_json(orient="index"))
        tm.assert_frame_equal(df, read_json(data3, orient="index"))
