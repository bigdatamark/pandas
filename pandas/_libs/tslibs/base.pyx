# [MODULE DOC]
# Defines ABCTimestamp, an empty cdef subclass of the stdlib datetime type. It
# exists solely so that other tslibs modules can do fast C-level isinstance checks
# for pandas Timestamp objects without importing timestamps.pyx (which would cause
# circular cimports). Timestamp's base class _Timestamp inherits from it.
"""
We define base classes that will be inherited by Timestamp, Timedelta, etc
in order to allow for fast isinstance checks without circular dependency issues.

This is analogous to core.dtypes.generic.
"""

from cpython.datetime cimport datetime


cdef class ABCTimestamp(datetime):
    # [CLASS DOC]
    # Abstract marker base class for pandas Timestamp. Adds no attributes or methods
    # beyond datetime; used only as a type tag for isinstance checks.
    pass
