# [MODULE DOC]
# Tests for the list-valued test extension array. Currently holds only a CSV output
# regression test plus the fixtures it needs (dtype and data). Does not use the shared extension
# conftest fixtures for data.
import pytest

import pandas as pd
from pandas.tests.extension.list.array import (
    ListArray,
    ListDtype,
    make_data,
)


@pytest.fixture
def dtype():
    # [FUNCTION DOC]
    # HELPER: fixture returning a ListDtype instance.
    return ListDtype()


@pytest.fixture
def data():
    # [FUNCTION DOC]
    # HELPER: fixture returning a length-10 ListArray. Generates random list data and
    # regenerates until the first two elements have different lengths, so tests can rely on
    # them being distinguishable.
    """Length-10 ListArray for semantics test."""
    data = make_data(10)

    while len(data[0]) == len(data[1]):
        data = make_data(10)

    return ListArray(data)


def test_to_csv(data):
    # [FUNCTION DOC]
    # Verifies that writing a frame with a list-array column to CSV works (regression for an
    # upstream issue where converting nested elements to string failed). Builds a one-column frame
    # from the data fixture, writes it to a string, and asserts the text of the first element
    # appears in the output.
    # https://github.com/pandas-dev/pandas/issues/28840
    # array with list-likes fail when doing astype(str) on the numpy array
    # which was done in get_values_for_csv
    df = pd.DataFrame({"a": data})
    res = df.to_csv()
    assert str(data[0]) in res
