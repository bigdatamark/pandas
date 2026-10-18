# [MODULE DOC]
# Tests for DataFrame infer_objects, which converts object columns to a more specific dtype when all values allow it.

from datetime import datetime

from pandas import DataFrame
import pandas._testing as tm


class TestInferObjects:
    # [CLASS DOC]
    # Groups the infer_objects tests.
    def test_infer_objects(self):
        # [FUNCTION DOC]
        # GH#11221. Builds a frame of object columns whose first row holds a string and later rows hold ints, floats, datetimes, and for the last column ints followed by a string. After dropping the first row and calling infer_objects, expects int64, float64, microsecond datetime, and object dtypes respectively, and that the result equals a frame constructed directly from the remaining values.
        # GH#11221
        df = DataFrame(
            {
                "a": ["a", 1, 2, 3],
                "b": ["b", 2.0, 3.0, 4.1],
                "c": [
                    "c",
                    datetime(2016, 1, 1),
                    datetime(2016, 1, 2),
                    datetime(2016, 1, 3),
                ],
                "d": [1, 2, 3, "d"],
            },
            columns=["a", "b", "c", "d"],
        )
        df = df.iloc[1:].infer_objects()

        assert df["a"].dtype == "int64"
        assert df["b"].dtype == "float64"
        assert df["c"].dtype == "M8[us]"
        assert df["d"].dtype == "object"

        expected = DataFrame(
            {
                "a": [1, 2, 3],
                "b": [2.0, 3.0, 4.1],
                "c": [datetime(2016, 1, 1), datetime(2016, 1, 2), datetime(2016, 1, 3)],
                "d": [2, 3, "d"],
            },
            columns=["a", "b", "c", "d"],
        )
        # reconstruct frame to verify inference is same
        result = df.reset_index(drop=True)
        tm.assert_frame_equal(result, expected)
