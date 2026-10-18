# [MODULE DOC]
# Tests that DataFrame methods reject non-boolean values for the inplace argument.
# Contains a single test class; depends only on the DataFrame class and pytest.
import pytest

from pandas.core.frame import DataFrame


class TestDataFrameValidate:
    # [CLASS DOC]
    # Groups the argument-type validation tests for DataFrame methods. Stateless.
    """Tests for error handling related to data types of method arguments."""

    @pytest.mark.parametrize(
        "func",
        [
            "query",
            "eval",
            "set_index",
            "reset_index",
            "dropna",
            "drop_duplicates",
            "sort_values",
        ],
    )
    @pytest.mark.parametrize("inplace", [1, "True", [1, 2, 3], 5.0])
    def test_validate_bool_args(self, func, inplace):
        # [FUNCTION DOC]
        # Verifies that each listed DataFrame method raises a value error mentioning
        # that
        # the inplace argument must be a bool when given a non-bool.
        # Parametrized over method names (query, eval, set_index, reset_index, dropna,
        # drop_duplicates, sort_values) and over invalid inplace values (integer,
        # string,
        # list, float). Builds a small two-column frame and adds the required extra
        # keyword (expression, keys or by) for the methods that need one. Expected
        # result:
        # a value error whose message matches the bool-type complaint.
        dataframe = DataFrame({"a": [1, 2], "b": [3, 4]})
        msg = 'For argument "inplace" expected type bool'
        kwargs = {"inplace": inplace}

        if func == "query":
            kwargs["expr"] = "a > b"
        elif func == "eval":
            kwargs["expr"] = "a + b"
        elif func == "set_index":
            kwargs["keys"] = ["a"]
        elif func == "sort_values":
            kwargs["by"] = ["a"]

        with pytest.raises(ValueError, match=msg):
            getattr(dataframe, func)(**kwargs)
