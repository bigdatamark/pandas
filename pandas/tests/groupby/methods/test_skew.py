# [MODULE DOC]
# Test of the grouped skew reduction on DataFrameGroupBy, checking the cython group-wise implementation against the per-group nan-aware skew computed through iteration.
import numpy as np

import pandas as pd
import pandas._testing as tm


def test_groupby_skew_equivalence():
    # [FUNCTION DOC]
    # Builds a 1000 by 2 random normal frame with about 5 percent of rows set to missing, groups it by random integers in three buckets, and compares the groupby skew result with the concatenation of each group's own skew (transposed to a row per group). The expected index is cast to the result's index dtype to accommodate 32-bit platforms. Expects frames equal.
    # Test that that groupby skew method (which uses libgroupby.group_skew)
    #  matches the results of operating group-by-group (which uses nanops.nanskew)
    nrows = 1000
    ngroups = 3
    ncols = 2
    nan_frac = 0.05

    arr = np.random.default_rng(2).standard_normal((nrows, ncols))
    arr[np.random.default_rng(2).random(nrows) < nan_frac] = np.nan

    df = pd.DataFrame(arr)
    grps = np.random.default_rng(2).integers(0, ngroups, size=nrows)
    gb = df.groupby(grps)

    result = gb.skew()

    grpwise = [grp.skew().to_frame(i).T for i, grp in gb]
    expected = pd.concat(grpwise, axis=0)
    expected.index = expected.index.astype(result.index.dtype)  # 32bit builds
    tm.assert_frame_equal(result, expected)
