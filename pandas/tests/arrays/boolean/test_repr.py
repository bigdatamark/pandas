# [MODULE DOC]
# Tests for the string representation of the nullable boolean array inside a frame, a series, and on its own.
import pandas as pd


def test_repr():
    # [FUNCTION DOC]
    # Checks that the repr of a one-column frame, of that column as a series, and of the underlying array exactly match the expected text, with missing shown as the NA marker, right-aligned values, and the array repr giving the class name, element list, length, and dtype.
    df = pd.DataFrame({"A": pd.array([True, False, None], dtype="boolean")})
    expected = "       A\n0   True\n1  False\n2   <NA>"
    assert repr(df) == expected

    expected = "0     True\n1    False\n2     <NA>\nName: A, dtype: boolean"
    assert repr(df.A) == expected

    expected = "<BooleanArray>\n[True, False, <NA>]\nLength: 3, dtype: boolean"
    assert repr(df.A.array) == expected
