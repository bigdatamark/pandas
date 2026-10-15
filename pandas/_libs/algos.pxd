# [MODULE DOC]
# Cython declarations exported by pandas/_libs/algos.pyx for cimport by other extension
#   modules (groupby.pyx, window/aggregations.pyx): an in-place selection routine, the
#   rank tie-breaking enum, and the NaN fill-value helper used when ranking.
from pandas._libs.dtypes cimport (
    numeric_object_t,
    numeric_t,
)


# [FUNCTION DOC]
# Declaration of algos.kth_smallest_c: in-place quickselect (Hoare partition) returning
#   the k-th smallest (0-based) of the first n elements of the C array arr of a numeric
#   fused type. Reorders arr; no exception; GIL not required.
cdef numeric_t kth_smallest_c(numeric_t* arr, Py_ssize_t k, Py_ssize_t n) noexcept nogil

cdef enum TiebreakEnumType:
    # [CLASS DOC]
    # C enum of tie-breaking strategies for ranking equal values: TIEBREAK_AVERAGE (mean
    #   of the tied ranks), TIEBREAK_MIN (lowest), TIEBREAK_MAX (highest),
    #   TIEBREAK_FIRST (order of appearance), TIEBREAK_FIRST_DESCENDING (order of
    #   appearance when ranking descending; internal only), TIEBREAK_DENSE (like MIN but
    #   ranks increase by 1 between distinct values). Values are 0..5 in that order.
    TIEBREAK_AVERAGE
    TIEBREAK_MIN,
    TIEBREAK_MAX
    TIEBREAK_FIRST
    TIEBREAK_FIRST_DESCENDING
    TIEBREAK_DENSE


# [FUNCTION DOC]
# Declaration of algos.get_rank_nan_fill_val: returns the sentinel used to replace
#   missing values before sorting for ranking - the type maximum (or +inf / Infinity
#   object) when rank_nans_highest, else the type minimum (or -inf / NegInfinity, NaT
#   for int64, NaT+1 for datetimelike). val only selects the fused specialization;
#   is_datetimelike defaults to False.
cdef numeric_object_t get_rank_nan_fill_val(
    bint rank_nans_highest,
    numeric_object_t val,
    bint is_datetimelike=*,
)
