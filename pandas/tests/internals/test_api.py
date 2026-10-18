"""
Tests for the pseudo-public API implemented in internals/api.py and exposed
in core.internals
"""

# [MODULE DOC]
# Tests for the semi-public internals API: the block-construction shim module and its re-exports in the core internals namespace, the deprecations in that namespace, and the public function that builds a frame from raw blocks.
# Imports the internals package, its api module and the public internals module; no import-time side effects beyond imports.

import datetime

import numpy as np
import pytest

from pandas.errors import Pandas4Warning

import pandas as pd
import pandas._testing as tm
from pandas.api.internals import create_dataframe_from_blocks
from pandas.core import internals
from pandas.core.internals import api


def test_internals_api():
    # [FUNCTION DOC]
    # Verifies that the block factory exposed in the core internals namespace is the same object as the one defined in the api module.
    assert internals.make_block is api.make_block


def test_namespace():
    # [FUNCTION DOC]
    # Verifies the exact public namespace of the core internals package (marked subject to change).
    # Expected: the non-dunder names present equal the union of the submodule names (blocks, concat, managers, construction, api, ops) and the exported names (block factory, the two block manager classes and the manager concatenation function); on failure the symmetric difference is shown.
    # SUBJECT TO CHANGE

    modules = [
        "blocks",
        "concat",
        "managers",
        "construction",
        "api",
        "ops",
    ]
    expected = [
        "make_block",
        "BlockManager",
        "SingleBlockManager",
        "concatenate_managers",
    ]

    result = [x for x in dir(internals) if not x.startswith("__")]
    assert set(result) == set(expected + modules), set(result) ^ set(expected + modules)


@pytest.mark.parametrize(
    "name",
    [
        "Block",
        "ExtensionBlock",
    ],
)
def test_deprecations(name):
    # [FUNCTION DOC]
    # Verifies that accessing the legacy block class names in the internals namespace emits the deprecation warning (issue 55139).
    # Parametrized over the two names (generic block and extension block). The message must say the name is deprecated and to use public APIs instead.
    # GH#55139
    msg = f"{name} is deprecated.* Use public APIs instead"
    with tm.assert_produces_warning(Pandas4Warning, match=msg):
        getattr(internals, name)


def test_make_block_2d_with_dti():
    # [FUNCTION DOC]
    # Verifies the block factory is deprecated and still builds a two-dimensional block from a timezone-aware datetime index (issue 41168).
    # Expected: the deprecation warning about the factory is emitted, and the resulting block and its values both have shape one by three.
    # GH#41168
    dti = pd.date_range("2012", periods=3, tz="UTC")

    msg = "make_block is deprecated"
    with tm.assert_produces_warning(Pandas4Warning, match=msg):
        blk = api.make_block(dti, placement=[0])

    assert blk.shape == (1, 3)
    assert blk.values.shape == (1, 3)


def test_create_block_manager_from_blocks_deprecated():
    # [FUNCTION DOC]
    # Verifies that accessing the manager-from-blocks creator through the internals namespace emits the deprecation warning stating it will be removed and to use public APIs (issue 33892). The access alone triggers the warning.
    # GH#33892
    # If they must, downstream packages should get this from internals.api,
    #  not internals.
    msg = (
        "create_block_manager_from_blocks is deprecated and will be "
        "removed in a future version. Use public APIs instead"
    )
    with tm.assert_produces_warning(Pandas4Warning, match=msg):
        internals.create_block_manager_from_blocks


def test_maybe_infer_ndim_deprecated():
    # [FUNCTION DOC]
    # Verifies that the legacy ndim inference helper in the api module emits a deprecation warning (issue 40226).
    # Calls it with a five-element integer array, a block placement of one position and the integer one; expects a standard deprecation warning (not the project warning class) whose message says it will be removed in a future version.
    # GH#40226
    msg = "maybe_infer_ndim is deprecated and will be removed in a future version."
    arr = np.arange(5)
    bp = pd._libs.internals.BlockPlacement([1])
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        internals.api.maybe_infer_ndim(arr, bp, 1)


def test_create_dataframe_from_blocks(float_frame):
    # [FUNCTION DOC]
    # Verifies building a frame from a single raw block with the public function.
    # Uses the float frame fixture: takes its first block's values and placement array together with copies of the index and columns, and expects a frame equal to the fixture.
    block = float_frame._mgr.blocks[0]
    index = float_frame.index.copy()
    columns = float_frame.columns.copy()

    result = create_dataframe_from_blocks(
        [(block.values, block.mgr_locs.as_array)], index=index, columns=columns
    )
    tm.assert_frame_equal(result, float_frame)


def test_create_dataframe_from_blocks_types():
    # [FUNCTION DOC]
    # Verifies the public block-based constructor round trips frames of many dtypes.
    # Setup: a frame with integer, unsigned, float, bool, nullable boolean, string, datetime, timezone-aware datetime, timedelta, period, categorical and interval columns. Rebuilds the frame from the values and placements of all its blocks, with the same index and columns; expected equal to the original.
    df = pd.DataFrame(
        {
            "int": list(range(1, 4)),
            "uint": np.arange(3, 6).astype("uint8"),
            "float": [2.0, np.nan, 3.0],
            "bool": np.array([True, False, True]),
            "boolean": pd.array([True, False, None], dtype="boolean"),
            "string": list("abc"),
            "datetime": pd.date_range("20130101", periods=3),
            "datetimetz": pd.date_range("20130101", periods=3).tz_localize(
                "Europe/Brussels"
            ),
            "timedelta": pd.timedelta_range("1 day", periods=3),
            "period": pd.period_range("2012-01-01", periods=3, freq="D"),
            "categorical": pd.Categorical(["a", "b", "a"]),
            "interval": pd.IntervalIndex.from_tuples([(0, 1), (1, 2), (3, 4)]),
        }
    )

    result = create_dataframe_from_blocks(
        [(block.values, block.mgr_locs.as_array) for block in df._mgr.blocks],
        index=df.index,
        columns=df.columns,
    )
    tm.assert_frame_equal(result, df)


def test_create_dataframe_from_blocks_datetimelike():
    # [FUNCTION DOC]
    # Verifies that datetime-like data passed as plain numpy arrays (datetime and timedelta arrays whose dtypes match extension dtypes) is accepted.
    # Setup: two two-by-three arrays of day-spaced datetimes and timedeltas placed at columns zero/two and one/three of a four-column frame. Expected: columns of datetimes and timedeltas with microsecond resolution as listed.
    # extension dtypes that have an exact matching numpy dtype can also be
    # be passed as a numpy array
    index, columns = pd.RangeIndex(3), pd.Index(["a", "b", "c", "d"])

    block_array1 = np.arange(
        datetime.datetime(2020, 1, 1),
        datetime.datetime(2020, 1, 7),
        step=datetime.timedelta(1),
    ).reshape((2, 3))
    block_array2 = np.arange(
        datetime.timedelta(1), datetime.timedelta(7), step=datetime.timedelta(1)
    ).reshape((2, 3))
    result = create_dataframe_from_blocks(
        [(block_array1, np.array([0, 2])), (block_array2, np.array([1, 3]))],
        index=index,
        columns=columns,
    )
    expected = pd.DataFrame(
        {
            "a": pd.date_range("2020-01-01", periods=3, unit="us"),
            "b": pd.timedelta_range("1 days", periods=3, unit="us"),
            "c": pd.date_range("2020-01-04", periods=3, unit="us"),
            "d": pd.timedelta_range("4 days", periods=3, unit="us"),
        }
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "array",
    [
        pd.date_range("2020-01-01", periods=3),
        pd.date_range("2020-01-01", periods=3, tz="UTC"),
        pd.period_range("2012-01-01", periods=3, freq="D"),
        pd.timedelta_range("1 day", periods=3),
    ],
)
def test_create_dataframe_from_blocks_1dEA(array):
    # [FUNCTION DOC]
    # Verifies that one-dimensional extension arrays may be passed even though the block stores them as two-dimensional.
    # Parametrized over datetime, timezone-aware datetime, period and timedelta arrays. Builds a frame, then re-creates it from the first row of the block values and the placement; expected equal to the original.
    # ExtensionArrays can be passed as 1D even if stored under the hood as 2D
    df = pd.DataFrame({"a": array})

    block = df._mgr.blocks[0]
    result = create_dataframe_from_blocks(
        [(block.values[0], block.mgr_locs.as_array)], index=df.index, columns=df.columns
    )
    tm.assert_frame_equal(result, df)
