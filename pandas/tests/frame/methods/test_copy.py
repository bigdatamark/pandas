# [MODULE DOC]
# Tests for the frame copy method: independence of copied axes labels, data and internal
# block manager, and consolidation of blocks on copy. Uses frame fixtures from conftest.
# No import-time side effects.

import numpy as np
import pytest

from pandas import DataFrame


class TestCopy:
    # [CLASS DOC]
    # Stateless pytest class of copy tests.
    @pytest.mark.parametrize("attr", ["index", "columns"])
    def test_copy_index_name_checking(self, float_frame, attr):
        # [FUNCTION DOC]
        # Parametrized over index and columns. Setting a name on a copy's axis must not
        # change the name on the original frame's axis.
        # don't want to be able to modify the index stored elsewhere after
        # making a copy
        ind = getattr(float_frame, attr)
        ind.name = None
        cp = float_frame.copy()
        getattr(cp, attr).name = "foo"
        assert getattr(float_frame, attr).name is None

    def test_copy(self, float_frame, float_string_frame):
        # [FUNCTION DOC]
        # Adding a column to a copy leaves the original unchanged; copying a mixed frame
        # produces a distinct block manager object.
        cop = float_frame.copy()
        cop["E"] = cop["A"]
        assert "E" not in float_frame

        # copy objects
        copy = float_string_frame.copy()
        assert copy._mgr is not float_string_frame._mgr

    def test_copy_consolidates(self):
        # [FUNCTION DOC]
        # Builds a frame with many separately-added columns (eleven blocks) and checks
        # that copying consolidates them into a single block.
        # GH#42477
        df = DataFrame(
            {
                "a": np.random.default_rng(2).integers(0, 100, size=55),
                "b": np.random.default_rng(2).integers(0, 100, size=55),
            }
        )

        for i in range(10):
            df.loc[:, f"n_{i}"] = np.random.default_rng(2).integers(0, 100, size=55)

        assert len(df._mgr.blocks) == 11
        result = df.copy()
        assert len(result._mgr.blocks) == 1
