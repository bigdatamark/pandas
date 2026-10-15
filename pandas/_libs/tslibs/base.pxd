# [MODULE DOC]
# Cython declaration file exposing ABCTimestamp (see base.pyx) for cimport by other
# tslibs modules so they can type-check Timestamp objects at C level.
from cpython.datetime cimport datetime


cdef class ABCTimestamp(datetime):
    # [CLASS DOC]
    # Declaration of the marker base class for Timestamp; subclass of datetime with no
    # additional members.
    pass
