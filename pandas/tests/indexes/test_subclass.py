"""
Tests involving custom Index subclasses
"""

# [MODULE DOC]
# Tests involving a custom user-defined Index subclass that cannot hold strings. Defines the subclass and one regression test about inserting a string into it.

import numpy as np

from pandas import (
    DataFrame,
    Index,
)
import pandas._testing as tm


class CustomIndex(Index):
    # [CLASS DOC]
    # Index subclass used for testing, whose constructor rejects string members and stores everything as object-dtype values. Inherits all behavior from the base Index class.
    def __new__(cls, data, name=None):
        # [FUNCTION DOC]
        # Constructor. Takes data (an iterable) and an optional name. Raises a type error if any element is a string. If no name is given and the data has a name attribute, it is used. Converts the data to an object array and returns a new instance through the private simple-new constructor.
        # assert that this index class cannot hold strings
        if any(isinstance(val, str) for val in data):
            raise TypeError("CustomIndex cannot hold strings")

        if name is None and hasattr(data, "name"):
            name = data.name
        data = np.array(data, dtype="O")

        return cls._simple_new(data, name)


def test_insert_fallback_to_base_index():
    # [FUNCTION DOC]
    # Inserting a string into the custom index falls back to a plain object Index containing the string and the original values. A frame with that custom index as columns, after reset_index, has columns equal to the expected plain object index.
    # https://github.com/pandas-dev/pandas/issues/47071

    idx = CustomIndex([1, 2, 3])
    result = idx.insert(0, "string")
    expected = Index(["string", 1, 2, 3], dtype=object)
    tm.assert_index_equal(result, expected)

    df = DataFrame(
        np.random.default_rng(2).standard_normal((2, 3)),
        columns=idx,
        index=Index([1, 2], name="string"),
    )
    result = df.reset_index()
    tm.assert_index_equal(result.columns, expected)
