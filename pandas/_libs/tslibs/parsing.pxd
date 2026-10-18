# [MODULE DOC]
# Cython declarations of parsing.pyx functions cimported by other tslibs modules (e.g.
#   conversion.pyx for string-to-Timestamp, period.pyx for quarter handling). See
#   parsing.pyx for behavior.
from cpython.datetime cimport datetime
from numpy cimport int64_t

from pandas._libs.tslibs.np_datetime cimport NPY_DATETIMEUNIT


# [FUNCTION DOC]
# Declaration; see parsing.pyx (anchor month of a freq string, default DEC).
cpdef str get_rule_month(str source)
# [FUNCTION DOC]
# Declaration; see parsing.pyx (fiscal year/quarter to calendar year/month).
cpdef quarter_to_myear(int year, int quarter, str freq)

# [FUNCTION DOC]
# Declaration; see parsing.pyx (free-form string to datetime, reporting resolution and
#   extra nanoseconds via pointers).
cdef datetime parse_datetime_string(
    str date_string,
    bint dayfirst,
    bint yearfirst,
    NPY_DATETIMEUNIT* out_bestunit,
    int64_t* nanos,
)
