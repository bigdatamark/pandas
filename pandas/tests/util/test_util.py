# [MODULE DOC]
# Tests for miscellaneous testing utilities and the test-suite fixtures:
# numpy error-state defaults, the csv row conversion helper, the data path
# fixture, the external error context manager and the sortedness assertion.
# Relies on the testing helper module and the data path fixture from conftest.
import os

import pytest

from pandas import (
    array,
    compat,
)
import pandas._testing as tm


def test_numpy_err_state_is_default():
    # [FUNCTION DOC]
    # Verifies that importing numpy (after pandas is loaded) leaves numpy's
    # floating point error state at its defaults: warn for overflow, divide
    # and invalid operations, ignore for underflow. No inputs; passes if the
    # reported state equals that mapping.
    expected = {"over": "warn", "divide": "warn", "invalid": "warn", "under": "ignore"}
    import numpy as np

    # The error state should be unchanged after that import.
    assert np.geterr() == expected


def test_convert_rows_list_to_csv_str():
    # [FUNCTION DOC]
    # Verifies the helper that joins a list of row strings into one csv text.
    # Input is three simple rows; expected output is the rows joined with a
    # line terminator after each row, using carriage return plus newline on
    # Windows and a plain newline on other platforms.
    rows_list = ["aaa", "bbb", "ccc"]
    ret = tm.convert_rows_list_to_csv_str(rows_list)

    if compat.is_platform_windows():
        expected = "aaa\r\nbbb\r\nccc\r\n"
    else:
        expected = "aaa\nbbb\nccc\n"

    assert ret == expected


@pytest.mark.parametrize("strict_data_files", [True, False])
def test_datapath_missing(datapath):
    # [FUNCTION DOC]
    # Verifies the data path fixture raises a value error mentioning that the
    # file could not be found when asked for a nonexistent file. Parametrized
    # over the strict data files option (true and false); the fixture must
    # raise in both modes.
    with pytest.raises(ValueError, match="Could not find file"):
        datapath("not_a_file")


def test_datapath(datapath):
    # [FUNCTION DOC]
    # Verifies the data path fixture resolves path components for the bundled
    # iris csv file to the absolute path under the tests directory (the parent
    # of this test's directory) joined with the given components.
    args = ("io", "data", "csv", "iris.csv")

    result = datapath(*args)
    expected = os.path.join(os.path.dirname(os.path.dirname(__file__)), *args)

    assert result == expected


def test_external_error_raised():
    # [FUNCTION DOC]
    # Verifies the external error context manager swallows an error of the
    # given type regardless of its message: raising a type error with an
    # arbitrary message inside it must not fail the test.
    with tm.external_error_raised(TypeError):
        raise TypeError("Should not check this error message, so it will pass")


def test_is_sorted():
    # [FUNCTION DOC]
    # Verifies the sortedness assertion helper on a nullable integer array:
    # an ascending array passes; an unsorted array raises an assertion error
    # whose message says the extension arrays are different.
    arr = array([1, 2, 3], dtype="Int64")
    tm.assert_is_sorted(arr)

    arr = array([4, 2, 3], dtype="Int64")
    with pytest.raises(AssertionError, match="ExtensionArray are different"):
        tm.assert_is_sorted(arr)
