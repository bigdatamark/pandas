# [PACKAGE DOC]
# Test package defining an example extension array that stores calendar dates.
# Submodule: array (the array and dtype implementations).
# Re-exports DateArray and DateDtype (the public names listed in the export
# list). Import has no side effects beyond importing the array module.
from pandas.tests.extension.date.array import (
    DateArray,
    DateDtype,
)

__all__ = ["DateArray", "DateDtype"]
