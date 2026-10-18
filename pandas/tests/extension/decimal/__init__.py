# [PACKAGE DOC]
# Test package defining an example extension array backed by decimal values.
# Submodules: array (array, dtype and data-construction helpers) and
# test_decimal (the extension test suite applied to the decimal array).
# Re-exports DecimalArray, DecimalDtype, make_data and to_decimal (the public
# names listed in the export list). Import has no side effects beyond
# importing the array module.
from pandas.tests.extension.decimal.array import (
    DecimalArray,
    DecimalDtype,
    make_data,
    to_decimal,
)

__all__ = ["DecimalArray", "DecimalDtype", "make_data", "to_decimal"]
