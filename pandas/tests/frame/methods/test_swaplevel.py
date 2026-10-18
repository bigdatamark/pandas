# [MODULE DOC]
# Tests for DataFrame.swaplevel on frames with a hierarchical index.
# Contains one test class using the shared multi-level random frame fixture.

import pytest

from pandas import DataFrame
import pandas._testing as tm


class TestSwaplevel:
    # [CLASS DOC]
    # Test container for the swaplevel method of DataFrame and Series.
    def test_swaplevel(self, multiindex_dataframe_random_data):
        # [FUNCTION DOC]
        # Verifies swapping the two levels of a multi-index, using the default arguments,
        # a single position, two positions and two level names, all of which must give
        # identical results that differ from the original index. Swapping back must
        # restore an index equal to the original. Also verifies swapping along the column
        # axis of the transposed frame matches the transpose of the row-swapped frame, and
        # that calling it on a frame with a flat index raises a type error stating that levels
        # can only be swapped on a hierarchical axis.
        frame = multiindex_dataframe_random_data

        swapped = frame["A"].swaplevel()
        swapped2 = frame["A"].swaplevel(0)
        swapped3 = frame["A"].swaplevel(0, 1)
        swapped4 = frame["A"].swaplevel("first", "second")
        assert not swapped.index.equals(frame.index)
        tm.assert_series_equal(swapped, swapped2)
        tm.assert_series_equal(swapped, swapped3)
        tm.assert_series_equal(swapped, swapped4)

        back = swapped.swaplevel()
        back2 = swapped.swaplevel(0)
        back3 = swapped.swaplevel(0, 1)
        back4 = swapped.swaplevel("second", "first")
        assert back.index.equals(frame.index)
        tm.assert_series_equal(back, back2)
        tm.assert_series_equal(back, back3)
        tm.assert_series_equal(back, back4)

        ft = frame.T
        swapped = ft.swaplevel("first", "second", axis=1)
        exp = frame.swaplevel("first", "second").T
        tm.assert_frame_equal(swapped, exp)

        msg = "Can only swap levels on a hierarchical axis."
        with pytest.raises(TypeError, match=msg):
            DataFrame(range(3)).swaplevel()
