# [MODULE DOC]
# Tests for the function that inverts one-hot (dummy) encoding, turning indicator columns back into categorical columns, optionally using a separator to split column names into prefix and category and a default category for all-zero rows.
# Provides two fixtures of prefixed dummy frames and many module-level tests: error cases (types, unassigned, multi-assigned, null or non-dummy data, separators), round trips with the dummy-encoding function, and no-prefix / with-prefix behaviors including default categories and extension dtypes. No import-time side effects.
import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
    from_dummies,
    get_dummies,
)
import pandas._testing as tm


@pytest.fixture
def dummies_basic():
    # [FUNCTION DOC]
    # HELPER: fixture returning a three-row frame of indicator columns col1_a, col1_b and col2_a, col2_b, col2_c where every row has exactly one indicator set per prefix.
    return DataFrame(
        {
            "col1_a": [1, 0, 1],
            "col1_b": [0, 1, 0],
            "col2_a": [0, 1, 0],
            "col2_b": [1, 0, 0],
            "col2_c": [0, 0, 1],
        },
    )


@pytest.fixture
def dummies_with_unassigned():
    # [FUNCTION DOC]
    # HELPER: fixture like the basic dummies but with one row unassigned for each prefix: the last row of the first prefix has no indicator set (third row) and the first row of the second prefix has none.
    return DataFrame(
        {
            "col1_a": [1, 0, 0],
            "col1_b": [0, 1, 0],
            "col2_a": [0, 1, 0],
            "col2_b": [0, 0, 0],
            "col2_c": [0, 0, 1],
        },
    )


def test_error_wrong_data_type():
    # [FUNCTION DOC]
    # Passing a list instead of a frame raises a type error saying a frame was expected and naming the received type.
    dummies = [0, 1, 0]
    with pytest.raises(
        TypeError,
        match=r"Expected 'data' to be a 'DataFrame'; Received 'data' of type: list",
    ):
        from_dummies(dummies)


def test_error_no_prefix_contains_unassigned():
    # [FUNCTION DOC]
    # Without a separator, a row where no column is set raises a value error reporting unassigned values and the first offending row (index two).
    dummies = DataFrame({"a": [1, 0, 0], "b": [0, 1, 0]})
    with pytest.raises(
        ValueError,
        match=(
            r"Dummy DataFrame contains unassigned value\(s\); "
            r"First instance in row: 2"
        ),
    ):
        from_dummies(dummies)


def test_error_no_prefix_wrong_default_category_type():
    # [FUNCTION DOC]
    # A default category that is a list raises a type error listing the allowed kinds (none, hashable, or dict) and the received type.
    dummies = DataFrame({"a": [1, 0, 1], "b": [0, 1, 1]})
    with pytest.raises(
        TypeError,
        match=(
            r"Expected 'default_category' to be of type 'None', 'Hashable', or 'dict'; "
            r"Received 'default_category' of type: list"
        ),
    ):
        from_dummies(dummies, default_category=["c", "d"])


def test_error_no_prefix_multi_assignment():
    # [FUNCTION DOC]
    # A row with more than one indicator set raises a value error about multi-assignment, naming the first offending row.
    dummies = DataFrame({"a": [1, 0, 1], "b": [0, 1, 1]})
    with pytest.raises(
        ValueError,
        match=(
            r"Dummy DataFrame contains multi-assignment\(s\); "
            r"First instance in row: 2"
        ),
    ):
        from_dummies(dummies)


def test_error_no_prefix_contains_nan():
    # [FUNCTION DOC]
    # A null in an indicator column raises a value error naming that column.
    dummies = DataFrame({"a": [1, 0, 0], "b": [0, 1, np.nan]})
    with pytest.raises(
        ValueError, match=r"Dummy DataFrame contains NA value in column: 'b'"
    ):
        from_dummies(dummies)


def test_error_contains_non_dummies():
    # [FUNCTION DOC]
    # Columns with values other than zero and one (including strings) raise a type error saying the frame contains non-dummy data.
    dummies = DataFrame(
        {"a": [1, 6, 3, 1], "b": [0, 1, 0, 2], "c": ["c1", "c2", "c3", "c4"]}
    )
    with pytest.raises(
        TypeError,
        match=r"Passed DataFrame contains non-dummy data",
    ):
        from_dummies(dummies)


def test_error_with_prefix_multiple_separators():
    # [FUNCTION DOC]
    # A column whose name uses a different separator than the one given raises a value error saying the separator was not found for that column.
    dummies = DataFrame(
        {
            "col1_a": [1, 0, 1],
            "col1_b": [0, 1, 0],
            "col2-a": [0, 1, 0],
            "col2-b": [1, 0, 1],
        },
    )
    with pytest.raises(
        ValueError,
        match=(r"Separator not specified for column: col2-a"),
    ):
        from_dummies(dummies, sep="_")


def test_error_with_prefix_sep_wrong_type(dummies_basic):
    # [FUNCTION DOC]
    # A separator that is a list raises a type error saying it must be a string or none, naming the received type.
    with pytest.raises(
        TypeError,
        match=(
            r"Expected 'sep' to be of type 'str' or 'None'; "
            r"Received 'sep' of type: list"
        ),
    ):
        from_dummies(dummies_basic, sep=["_"])


def test_error_with_prefix_contains_unassigned(dummies_with_unassigned):
    # [FUNCTION DOC]
    # With a separator, a row unassigned in a prefix group raises the unassigned-values value error with the first row.
    with pytest.raises(
        ValueError,
        match=(
            r"Dummy DataFrame contains unassigned value\(s\); "
            r"First instance in row: 2"
        ),
    ):
        from_dummies(dummies_with_unassigned, sep="_")


def test_error_with_prefix_default_category_wrong_type(dummies_with_unassigned):
    # [FUNCTION DOC]
    # A list as default category with a separator raises the same type error as without a prefix.
    with pytest.raises(
        TypeError,
        match=(
            r"Expected 'default_category' to be of type 'None', 'Hashable', or 'dict'; "
            r"Received 'default_category' of type: list"
        ),
    ):
        from_dummies(dummies_with_unassigned, sep="_", default_category=["x", "y"])


def test_error_with_prefix_default_category_dict_not_complete(
    dummies_with_unassigned,
):
    # [FUNCTION DOC]
    # A default-category dict covering only one of two prefix groups raises a value error reporting the length mismatch (one versus two).
    with pytest.raises(
        ValueError,
        match=(
            r"Length of 'default_category' \(1\) did not match "
            r"the length of the columns being encoded \(2\)"
        ),
    ):
        from_dummies(dummies_with_unassigned, sep="_", default_category={"col1": "x"})


def test_error_with_prefix_contains_nan(dummies_basic):
    # Set float64 dtype to avoid upcast when setting np.nan
    # [FUNCTION DOC]
    # A null set in a float indicator column of a prefixed group raises the value error naming the full column.
    dummies_basic["col2_c"] = dummies_basic["col2_c"].astype("float64")
    dummies_basic.loc[2, "col2_c"] = np.nan
    with pytest.raises(
        ValueError, match=r"Dummy DataFrame contains NA value in column: 'col2_c'"
    ):
        from_dummies(dummies_basic, sep="_")


def test_error_with_prefix_contains_non_dummies(dummies_basic):
    # Set object dtype to avoid upcast when setting "str"
    # [FUNCTION DOC]
    # A string placed in an object indicator column raises the type error about non-dummy data.
    dummies_basic["col2_c"] = dummies_basic["col2_c"].astype(object)
    dummies_basic.loc[2, "col2_c"] = "str"
    with pytest.raises(TypeError, match=r"Passed DataFrame contains non-dummy data"):
        from_dummies(dummies_basic, sep="_")


def test_error_with_prefix_double_assignment():
    # [FUNCTION DOC]
    # A prefixed group with two indicators set in one row raises the multi-assignment value error naming that row (zero).
    dummies = DataFrame(
        {
            "col1_a": [1, 0, 1],
            "col1_b": [1, 1, 0],
            "col2_a": [0, 1, 0],
            "col2_b": [1, 0, 0],
            "col2_c": [0, 0, 1],
        },
    )
    with pytest.raises(
        ValueError,
        match=(
            r"Dummy DataFrame contains multi-assignment\(s\); "
            r"First instance in row: 0"
        ),
    ):
        from_dummies(dummies, sep="_")


def test_roundtrip_series_to_dataframe():
    # [FUNCTION DOC]
    # Dummies built from a series and converted back give a single column frame with an empty-string column name and the original values.
    categories = Series(["a", "b", "c", "a"])
    dummies = get_dummies(categories)
    result = from_dummies(dummies)
    expected = DataFrame({"": ["a", "b", "c", "a"]})
    tm.assert_frame_equal(result, expected)


def test_roundtrip_single_column_dataframe():
    # [FUNCTION DOC]
    # Dummies built from a one-column frame (empty column name) and converted back with an underscore separator equal the original frame.
    categories = DataFrame({"": ["a", "b", "c", "a"]})
    dummies = get_dummies(categories)
    result = from_dummies(dummies, sep="_")
    expected = categories
    tm.assert_frame_equal(result, expected)


def test_roundtrip_with_prefixes():
    # [FUNCTION DOC]
    # Dummies from a two-column frame converted back with an underscore separator reproduce the original frame.
    categories = DataFrame({"col1": ["a", "b", "a"], "col2": ["b", "a", "c"]})
    dummies = get_dummies(categories)
    result = from_dummies(dummies, sep="_")
    expected = categories
    tm.assert_frame_equal(result, expected)


def test_no_prefix_string_cats_basic():
    # [FUNCTION DOC]
    # Integer indicator columns named as strings without a separator are converted to one column (empty name) of those names.
    dummies = DataFrame({"a": [1, 0, 0, 1], "b": [0, 1, 0, 0], "c": [0, 0, 1, 0]})
    expected = DataFrame({"": ["a", "b", "c", "a"]})
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)


def test_no_prefix_string_cats_basic_bool_values():
    # [FUNCTION DOC]
    # Same with boolean indicators.
    dummies = DataFrame(
        {
            "a": [True, False, False, True],
            "b": [False, True, False, False],
            "c": [False, False, True, False],
        }
    )
    expected = DataFrame({"": ["a", "b", "c", "a"]})
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)


def test_no_prefix_string_cats_basic_mixed_bool_values():
    # [FUNCTION DOC]
    # Same with a mix of integer and boolean indicator columns.
    dummies = DataFrame(
        {"a": [1, 0, 0, 1], "b": [False, True, False, False], "c": [0, 0, 1, 0]}
    )
    expected = DataFrame({"": ["a", "b", "c", "a"]})
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)


def test_no_prefix_int_cats_basic():
    # [FUNCTION DOC]
    # Integer column labels (unsorted) are recovered as integer categories in row order.
    dummies = DataFrame(
        {1: [1, 0, 0, 0], 25: [0, 1, 0, 0], 2: [0, 0, 1, 0], 5: [0, 0, 0, 1]}
    )
    expected = DataFrame({"": [1, 25, 2, 5]})
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)


def test_no_prefix_float_cats_basic():
    # [FUNCTION DOC]
    # Float column labels are recovered as float categories.
    dummies = DataFrame(
        {1.0: [1, 0, 0, 0], 25.0: [0, 1, 0, 0], 2.5: [0, 0, 1, 0], 5.84: [0, 0, 0, 1]}
    )
    expected = DataFrame({"": [1.0, 25.0, 2.5, 5.84]})
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)


def test_no_prefix_mixed_cats_basic():
    # [FUNCTION DOC]
    # Mixed column labels (float, string, integer, boolean, none) give an object column of those values.
    dummies = DataFrame(
        {
            1.23: [1, 0, 0, 0, 0],
            "c": [0, 1, 0, 0, 0],
            2: [0, 0, 1, 0, 0],
            False: [0, 0, 0, 1, 0],
            None: [0, 0, 0, 0, 1],
        }
    )
    expected = DataFrame({"": [1.23, "c", 2, False, None]}, dtype="object")
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)


def test_no_prefix_string_cats_contains_get_dummies_NaN_column():
    # [FUNCTION DOC]
    # A column literally named NaN is treated as an ordinary category.
    dummies = DataFrame({"a": [1, 0, 0], "b": [0, 1, 0], "NaN": [0, 0, 1]})
    expected = DataFrame({"": ["a", "b", "NaN"]})
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "default_category, expected",
    [
        pytest.param(
            "c",
            {"": ["a", "b", "c"]},
            id="default_category is a str",
        ),
        pytest.param(
            1,
            {"": ["a", "b", 1]},
            id="default_category is an int",
        ),
        pytest.param(
            1.25,
            {"": ["a", "b", 1.25]},
            id="default_category is a float",
        ),
        pytest.param(
            0,
            {"": ["a", "b", 0]},
            id="default_category is a 0",
        ),
        pytest.param(
            False,
            {"": ["a", "b", False]},
            id="default_category is a bool",
        ),
        pytest.param(
            (1, 2),
            {"": ["a", "b", (1, 2)]},
            id="default_category is a tuple",
        ),
    ],
)
def test_no_prefix_string_cats_default_category(
    default_category, expected, using_infer_string
):
    # [FUNCTION DOC]
    # Parametrized on default category value (string, int, float, zero, bool, tuple): the all-zero row receives the default category. Expected is built with the dtype of the input columns.
    # KNOWN ISSUE: the string-inference fixture parameter is accepted but never used.
    dummies = DataFrame({"a": [1, 0, 0], "b": [0, 1, 0]})
    result = from_dummies(dummies, default_category=default_category)
    expected = DataFrame(expected, dtype=dummies.columns.dtype)
    tm.assert_frame_equal(result, expected)


def test_with_prefix_basic(dummies_basic):
    # [FUNCTION DOC]
    # Prefixed dummies with an underscore separator give one column per prefix with the decoded categories.
    expected = DataFrame({"col1": ["a", "b", "a"], "col2": ["b", "a", "c"]})
    result = from_dummies(dummies_basic, sep="_")
    tm.assert_frame_equal(result, expected)


def test_with_prefix_contains_get_dummies_NaN_column():
    # [FUNCTION DOC]
    # Indicator columns named with a trailing NaN category are decoded as the string NaN category.
    dummies = DataFrame(
        {
            "col1_a": [1, 0, 0],
            "col1_b": [0, 1, 0],
            "col1_NaN": [0, 0, 1],
            "col2_a": [0, 1, 0],
            "col2_b": [0, 0, 0],
            "col2_c": [0, 0, 1],
            "col2_NaN": [1, 0, 0],
        },
    )
    expected = DataFrame({"col1": ["a", "b", "NaN"], "col2": ["NaN", "a", "c"]})
    result = from_dummies(dummies, sep="_")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "default_category, expected",
    [
        pytest.param(
            "x",
            {"col1": ["a", "b", "x"], "col2": ["x", "a", "c"]},
            id="default_category is a str",
        ),
        pytest.param(
            0,
            {"col1": ["a", "b", 0], "col2": [0, "a", "c"]},
            id="default_category is a 0",
        ),
        pytest.param(
            False,
            {"col1": ["a", "b", False], "col2": [False, "a", "c"]},
            id="default_category is a False",
        ),
        pytest.param(
            {"col2": 1, "col1": 2.5},
            {"col1": ["a", "b", 2.5], "col2": [1, "a", "c"]},
            id="default_category is a dict with int and float values",
        ),
        pytest.param(
            {"col2": None, "col1": False},
            {"col1": ["a", "b", False], "col2": [None, "a", "c"]},
            id="default_category is a dict with bool and None values",
        ),
        pytest.param(
            {"col2": (1, 2), "col1": [1.25, False]},
            {"col1": ["a", "b", [1.25, False]], "col2": [(1, 2), "a", "c"]},
            id="default_category is a dict with list and tuple values",
        ),
    ],
)
def test_with_prefix_default_category(
    dummies_with_unassigned, default_category, expected, using_infer_string
):
    # [FUNCTION DOC]
    # Parametrized default categories as a single value (string, zero, false) or per-prefix dict (mixed numeric, bool, none, list and tuple values) applied to unassigned rows; expected converted to string dtype when string inference is on.
    result = from_dummies(
        dummies_with_unassigned, sep="_", default_category=default_category
    )
    expected = DataFrame(expected)
    if using_infer_string:
        expected = expected.astype("str")
    tm.assert_frame_equal(result, expected)


def test_ea_categories():
    # GH 54300
    # [FUNCTION DOC]
    # GH 54300. Column labels of the python string extension dtype yield a result column of that extension dtype.
    df = DataFrame({"a": [1, 0, 0, 1], "b": [0, 1, 0, 0], "c": [0, 0, 1, 0]})
    df.columns = df.columns.astype("string[python]")
    result = from_dummies(df)
    expected = DataFrame({"": Series(list("abca"), dtype="string[python]")})
    tm.assert_frame_equal(result, expected)


def test_ea_categories_with_sep():
    # GH 54300
    # [FUNCTION DOC]
    # GH 54300. Same with prefixes and a separator; the result columns also use the string extension dtype.
    df = DataFrame(
        {
            "col1_a": [1, 0, 1],
            "col1_b": [0, 1, 0],
            "col2_a": [0, 1, 0],
            "col2_b": [1, 0, 0],
            "col2_c": [0, 0, 1],
        }
    )
    df.columns = df.columns.astype("string[python]")
    result = from_dummies(df, sep="_")
    expected = DataFrame(
        {
            "col1": Series(list("aba"), dtype="string[python]"),
            "col2": Series(list("bac"), dtype="string[python]"),
        }
    )
    expected.columns = expected.columns.astype("string[python]")
    tm.assert_frame_equal(result, expected)


def test_maintain_original_index():
    # GH 54300
    # [FUNCTION DOC]
    # GH 54300. The input frame index is preserved on the result.
    df = DataFrame(
        {"a": [1, 0, 0, 1], "b": [0, 1, 0, 0], "c": [0, 0, 1, 0]}, index=list("abcd")
    )
    result = from_dummies(df)
    expected = DataFrame({"": list("abca")}, index=list("abcd"))
    tm.assert_frame_equal(result, expected)


def test_int_columns_with_float_default():
    # https://github.com/pandas-dev/pandas/pull/60694
    # [FUNCTION DOC]
    # Pull 60694. A float default category with integer column labels raises a value error about coercing floats to integers.
    df = DataFrame(
        {
            3: [1, 0, 0],
            4: [0, 1, 0],
        },
    )
    with pytest.raises(ValueError, match="Trying to coerce float values to integers"):
        from_dummies(df, default_category=0.5)


def test_object_dtype_preserved():
    # https://github.com/pandas-dev/pandas/pull/60694
    # When the input has object dtype, the result should as
    # well even when infer_string is True.
    # [FUNCTION DOC]
    # Pull 60694. Object-dtype column labels give an object-dtype result even when string inference is enabled.
    df = DataFrame(
        {
            "x": [1, 0, 0],
            "y": [0, 1, 0],
        },
    )
    df.columns = df.columns.astype("object")
    result = from_dummies(df, default_category="z")
    expected = DataFrame({"": ["x", "y", "z"]}, dtype="object")
    tm.assert_frame_equal(result, expected)
