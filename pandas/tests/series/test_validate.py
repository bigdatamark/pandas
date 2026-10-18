# [MODULE DOC]
# Tests argument validation for Series methods that take a boolean in-place flag.
# No import-time side effects.

import pytest


@pytest.mark.parametrize(
    "func",
    [
        "reset_index",
        "_set_name",
        "sort_values",
        "sort_index",
        "rename",
        "dropna",
        "drop_duplicates",
    ],
)
@pytest.mark.parametrize("inplace", [1, "True", [1, 2, 3], 5.0])
def test_validate_bool_args(string_series, func, inplace):
    # [FUNCTION DOC]
    # Verifies that Series methods (reset index, set name, sort values, sort index, rename, drop missing, drop duplicates) reject an in-place argument that is not a real boolean (integer, string, list, float), raising a value error that says the argument must be of type bool. The private name-setting method also receives a name argument. Parametrized over method name and invalid flag; uses the string series fixture.
    """Tests for error handling related to data types of method arguments."""
    msg = 'For argument "inplace" expected type bool'
    kwargs = {"inplace": inplace}

    if func == "_set_name":
        kwargs["name"] = "hello"

    with pytest.raises(ValueError, match=msg):
        getattr(string_series, func)(**kwargs)
