# [PACKAGE DOC]
# Test package defining an example extension array whose elements are mappings
# (JSON-like objects).
# Submodules: array (array, dtype and data-construction helper) and test_json
# (the extension test suite applied to the JSON array).
# Re-exports JSONArray, JSONDtype and make_data (the public names listed in the
# export list). Import has no side effects beyond importing the array module.
from pandas.tests.extension.json.array import (
    JSONArray,
    JSONDtype,
    make_data,
)

__all__ = ["JSONArray", "JSONDtype", "make_data"]
