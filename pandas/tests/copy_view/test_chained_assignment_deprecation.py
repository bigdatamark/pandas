# [MODULE DOC]
# Tests that chained assignment (setting through an intermediate selection) emits
# a ChainedAssignmentError warning under the copy-on-write regime, for the
# Series/DataFrame setitem, iloc, loc, at and iat accessors.

import numpy as np
import pytest

from pandas.compat import CHAINED_WARNING_DISABLED
from pandas.errors import ChainedAssignmentError

from pandas import DataFrame
import pandas._testing as tm


@pytest.mark.parametrize(
    "indexer", [0, [0, 1], slice(0, 2), np.array([True, False, True])]
)
def test_series_setitem(indexer):
    # [FUNCTION DOC]
    # Parametrized over scalar, list, slice and boolean-mask indexers.
    # Chained setitem on a column Series must emit exactly one warning whose category
    # is ChainedAssignmentError (custom check because the standard helper does not
    # fail on multiple warnings). Returns early when chained warnings are disabled
    # at build time.
    # ensure we only get a single warning for those typical cases of chained
    # assignment
    df = DataFrame({"a": [1, 2, 3], "b": 1})

    # using custom check instead of tm.assert_produces_warning because that doesn't
    # fail if multiple warnings are raised
    if CHAINED_WARNING_DISABLED:
        return
    with pytest.warns() as record:  # noqa: TID251
        df["a"][indexer] = 0
    assert len(record) == 1
    assert record[0].category == ChainedAssignmentError


@pytest.mark.parametrize(
    "indexer", ["a", ["a", "b"], slice(0, 2), np.array([True, False, True])]
)
def test_frame_setitem(indexer):
    # [FUNCTION DOC]
    # Parametrized over column label, label list, slice and mask. Setting through a
    # row slice of a frame must raise the chained assignment warning.
    df = DataFrame({"a": [1, 2, 3, 4, 5], "b": 1})

    with tm.raises_chained_assignment_error():
        df[0:3][indexer] = 10


@pytest.mark.parametrize(
    "indexer", [0, [0, 1], slice(0, 2), np.array([True, False, True])]
)
def test_series_iloc_setitem(indexer):
    # [FUNCTION DOC]
    # Chained positional assignment via iloc on a column Series must warn, for
    # scalar, list, slice and mask indexers.
    df = DataFrame({"a": [1, 2, 3], "b": 1})

    with tm.raises_chained_assignment_error():
        df["a"].iloc[indexer] = 0


@pytest.mark.parametrize(
    "indexer", [0, [0, 1], slice(0, 2), np.array([True, False, True])]
)
def test_frame_iloc_setitem(indexer):
    # [FUNCTION DOC]
    # Chained positional assignment via iloc on a row-sliced frame must warn, for
    # scalar, list, slice and mask indexers.
    df = DataFrame({"a": [1, 2, 3, 4, 5], "b": 1})

    with tm.raises_chained_assignment_error():
        df[0:3].iloc[indexer] = 10


@pytest.mark.parametrize(
    "indexer", [0, [0, 1], slice(0, 2), np.array([True, False, True])]
)
def test_series_loc_setitem(indexer):
    # [FUNCTION DOC]
    # Chained label-based assignment via loc on a column Series must warn, for
    # scalar, list, slice and mask indexers.
    df = DataFrame({"a": [1, 2, 3], "b": 1})

    with tm.raises_chained_assignment_error():
        df["a"].loc[indexer] = 0


@pytest.mark.parametrize(
    "indexer", [0, [0, 1], (0, "a"), slice(0, 2), np.array([True, False, True])]
)
def test_frame_loc_setitem(indexer):
    # [FUNCTION DOC]
    # Chained label-based assignment via loc on a row-sliced frame must warn, for
    # scalar, list, row/column tuple, slice and mask indexers.
    df = DataFrame({"a": [1, 2, 3, 4, 5], "b": 1})

    with tm.raises_chained_assignment_error():
        df[0:3].loc[indexer] = 10


def test_series_at_setitem():
    # [FUNCTION DOC]
    # Chained scalar assignment via at on a column Series must warn.
    df = DataFrame({"a": [1, 2, 3], "b": 1})

    with tm.raises_chained_assignment_error():
        df["a"].at[0] = 0


def test_frame_at_setitem():
    # [FUNCTION DOC]
    # Chained scalar assignment via at on a row-sliced frame must warn.
    df = DataFrame({"a": [1, 2, 3, 4, 5], "b": 1})

    with tm.raises_chained_assignment_error():
        df[0:3].at[0, "a"] = 10


def test_series_iat_setitem():
    # [FUNCTION DOC]
    # Chained scalar assignment via iat on a column Series must warn.
    df = DataFrame({"a": [1, 2, 3], "b": 1})

    with tm.raises_chained_assignment_error():
        df["a"].iat[0] = 0


def test_frame_iat_setitem():
    # [FUNCTION DOC]
    # Chained scalar assignment via iat on a row-sliced frame must warn.
    df = DataFrame({"a": [1, 2, 3, 4, 5], "b": 1})

    with tm.raises_chained_assignment_error():
        df[0:3].iat[0, 0] = 10
