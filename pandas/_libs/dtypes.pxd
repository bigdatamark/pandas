# [MODULE DOC]
# Shared Cython fused types used across pandas._libs. numeric_t covers all
# signed/unsigned integer widths (8/16/32/64 bit) and float32/float64 (no complex, no
# bool). numeric_object_t extends numeric_t with Python object. Functions typed with
# these get one compiled specialization per member type. No functions or classes are
# defined here.
"""
Common location for shared fused types
"""

from numpy cimport (
    float32_t,
    float64_t,
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)

# All numeric types except complex
ctypedef fused numeric_t:
    int8_t
    int16_t
    int32_t
    int64_t

    uint8_t
    uint16_t
    uint32_t
    uint64_t

    float32_t
    float64_t

# All numeric types + object, doesn't include complex
ctypedef fused numeric_object_t:
    numeric_t
    object
