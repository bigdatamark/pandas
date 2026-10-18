# [PACKAGE DOC]
# Test package defining an example extension array whose float data carries an
# extra attribute that is stored on the array rather than on the dtype.
# Submodules: array (the array and dtype implementations) and
# test_array_with_attr (tests of attribute preservation).
# Re-exports FloatAttrArray and FloatAttrDtype (the public names listed in the
# export list). Import has no side effects beyond importing the array module.
from pandas.tests.extension.array_with_attr.array import (
    FloatAttrArray,
    FloatAttrDtype,
)

__all__ = ["FloatAttrArray", "FloatAttrDtype"]
