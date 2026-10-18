# [MODULE DOC]
# Tests for the groupby kurtosis method (GH#40139): equivalence with per-group
# computation, nullable and arrow float dtypes, skipna handling, and constant data.
import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
import pandas._testing as tm


def test_groupby_kurt_equivalence():
    # [FUNCTION DOC]
    # Verifies the groupby kurtosis (compiled kernel) equals computing kurtosis group by
    # group (nan-aware reduction) on a 1000 by 2 random frame with 5 percent missing
    # rows, grouped by random labels from three groups; the expected index is cast to
    # 64-bit integers for 32-bit builds.
    # GH#40139
    # Test that that groupby kurt method (which uses libgroupby.group_kurt)
    #  matches the results of operating group-by-group (which uses nanops.nankurt)
    nrows = 1000
    ngroups = 3
    ncols = 2
    nan_frac = 0.05

    arr = np.random.default_rng(2).standard_normal((nrows, ncols))
    arr[np.random.default_rng(2).random(nrows) < nan_frac] = np.nan

    df = pd.DataFrame(arr)
    grps = np.random.default_rng(2).integers(0, ngroups, size=nrows)
    gb = df.groupby(grps)

    result = gb.kurt()

    grpwise = [grp.kurt().to_frame(i).T for i, grp in gb]
    expected = pd.concat(grpwise, axis=0)
    expected.index = expected.index.astype("int64")  # 32bit builds
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64[pyarrow]", marks=td.skip_if_no("pyarrow")),
        "Float64",
    ],
)
def test_groupby_kurt_arrow_float64(dtype):
    # [FUNCTION DOC]
    # Verifies groupby kurtosis for a Float64 or float64[pyarrow] frame containing a
    # missing value, grouped into a single group, matches known values (about 2.164 and
    # 0.151) approximately. Parametrized over dtype; the arrow case is skipped without
    # pyarrow.
    # GH#40139
    # Test groupby.kurt() with float64[pyarrow] and Float64 dtypes
    df = pd.DataFrame(
        {
            "x": [1.0, pd.NA, 3.2, 4.8, 2.3, 1.9, 8.9],
            "y": [1.6, 3.3, 3.2, 6.8, 1.3, 2.9, 9.0],
        },
        dtype=dtype,
    )
    gb = df.groupby(by=lambda x: 0)

    result = gb.kurt()
    expected = pd.DataFrame({"x": [2.1644713], "y": [0.1513969]}, dtype=dtype)
    tm.assert_almost_equal(result, expected)


def test_groupby_kurt_noskipna():
    # [FUNCTION DOC]
    # Verifies groupby kurtosis with skipna disabled gives missing for the column
    # containing a missing value and the same value as before for the complete column.
    # GH#40139
    # Test groupby.kurt() with skipna = False
    df = pd.DataFrame(
        {
            "x": [1.0, np.nan, 3.2, 4.8, 2.3, 1.9, 8.9],
            "y": [1.6, 3.3, 3.2, 6.8, 1.3, 2.9, 9.0],
        }
    )
    gb = df.groupby(by=lambda x: 0)

    result = gb.kurt(skipna=False)
    expected = pd.DataFrame({"x": [np.nan], "y": [0.1513969]})
    tm.assert_almost_equal(result, expected)


def test_groupby_kurt_all_ones():
    # [FUNCTION DOC]
    # Verifies groupby kurtosis (skipna disabled) of a constant column gives zero,
    # matching the frame-level kurtosis behavior.
    # GH#40139
    # Test groupby.kurt() with constant values
    df = pd.DataFrame(
        {
            "x": [1.0] * 10,
        }
    )
    gb = df.groupby(by=lambda x: 0)

    result = gb.kurt(skipna=False)
    expected = pd.DataFrame(
        {
            "x": [0.0],  # Same behavior as pd.DataFrame.kurt()
        }
    )
    tm.assert_almost_equal(result, expected)
