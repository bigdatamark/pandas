# [MODULE DOC]
# HELPER: shared constants for the apply test modules.
# Builds two sorted lists of group-by transformation kernel names (one for series tests and one for
# frame tests) from the groupby base module, excluding the cumulative-count kernel because neither
# Series nor DataFrame provides it. Both lists have identical contents. No import-time side effects.

from pandas.core.groupby.base import transformation_kernels

# There is no Series.cumcount or DataFrame.cumcount
series_transform_kernels = [
    x for x in sorted(transformation_kernels) if x != "cumcount"
]
frame_transform_kernels = [x for x in sorted(transformation_kernels) if x != "cumcount"]
