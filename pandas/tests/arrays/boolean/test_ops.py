# [MODULE DOC]
# Tests for unary operators (invert and absolute value) on the nullable boolean array, series, and frame.
import pandas as pd
import pandas._testing as tm


class TestUnaryOps:
    # [CLASS DOC]
    # Groups tests of unary operators on boolean arrays.
    def test_invert(self):
        # [FUNCTION DOC]
        # Verifies logical inversion on an array with a missing value: true and false swap and missing stays missing.
        # Checks the same through a named, indexed series (index and name preserved) and through a frame with a boolean-extension column and a plain boolean column, where both columns are inverted.
        a = pd.array([True, False, None], dtype="boolean")
        expected = pd.array([False, True, None], dtype="boolean")
        tm.assert_extension_array_equal(~a, expected)

        expected = pd.Series(expected, index=["a", "b", "c"], name="name")
        result = ~pd.Series(a, index=["a", "b", "c"], name="name")
        tm.assert_series_equal(result, expected)

        df = pd.DataFrame({"A": a, "B": [True, False, False]}, index=["a", "b", "c"])
        result = ~df
        expected = pd.DataFrame(
            {"A": expected, "B": [False, True, True]}, index=["a", "b", "c"]
        )
        tm.assert_frame_equal(result, expected)

    def test_abs(self):
        # [FUNCTION DOC]
        # Verifies that absolute value of a boolean array with missing values returns an array equal to the input, matching NumPy where it is the identity for booleans.
        # matching numpy behavior, abs is the identity function
        arr = pd.array([True, False, None], dtype="boolean")
        result = abs(arr)

        tm.assert_extension_array_equal(result, arr)
