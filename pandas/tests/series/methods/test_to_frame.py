# [MODULE DOC]
# Tests for the Series method converting a Series to a one-column DataFrame, covering column naming and subclass propagation.
# No import-time side effects.

import pytest

from pandas import (
    DataFrame,
    Index,
    Series,
)
import pandas._testing as tm


class TestToFrame:
    # [CLASS DOC]
    # Groups tests of converting a Series to a DataFrame. Stateless.
    def test_to_frame_respects_name_none(self):
        # [FUNCTION DOC]
        # Verifies that explicitly passing None as the name is respected: the resulting column index is an object-dtype index holding a single None, both for an unnamed and a named Series (issues 44212, 45448).
        # GH#44212 if we explicitly pass name=None, then that should be respected,
        #  not changed to 0
        # GH-45448 this is first deprecated & enforced in 2.0
        ser = Series(range(3))
        result = ser.to_frame(None)

        exp_index = Index([None], dtype=object)
        tm.assert_index_equal(result.columns, exp_index)

        result = ser.rename("foo").to_frame(None)
        exp_index = Index([None], dtype=object)
        tm.assert_index_equal(result.columns, exp_index)

    def test_to_frame(self, datetime_series):
        # [FUNCTION DOC]
        # Verifies column naming using the datetime series: with no name the single column is labeled by position zero, with a series name that name is used, and an explicit name argument overrides the series name. Each result is compared to an equivalent DataFrame built from the values and the original index.
        datetime_series.name = None
        rs = datetime_series.to_frame()
        xp = DataFrame(datetime_series.values, index=datetime_series.index)
        tm.assert_frame_equal(rs, xp)

        datetime_series.name = "testname"
        rs = datetime_series.to_frame()
        xp = DataFrame(
            {"testname": datetime_series.values}, index=datetime_series.index
        )
        tm.assert_frame_equal(rs, xp)

        rs = datetime_series.to_frame(name="testdifferent")
        xp = DataFrame(
            {"testdifferent": datetime_series.values}, index=datetime_series.index
        )
        tm.assert_frame_equal(rs, xp)

    @pytest.mark.filterwarnings(
        "ignore:Passing a BlockManager|Passing a SingleBlockManager:DeprecationWarning"
    )
    def test_to_frame_expanddim(self):
        # [FUNCTION DOC]
        # Verifies that a Series subclass overriding the expanded-dimension constructor property makes the conversion return the subclass's frame type, with the column labeled by the series name (issue 9762). Block-manager deprecation warnings are filtered.
        # GH#9762

        class SubclassedSeries(Series):
            # [CLASS DOC]
            # HELPER: local Series subclass whose expanded-dimension constructor points to the local frame subclass, used to test constructor propagation.
            @property
            def _constructor_expanddim(self):
                # [FUNCTION DOC]
                # HELPER: property returning the local DataFrame subclass as the type to construct when expanding dimensions.
                return SubclassedFrame

        class SubclassedFrame(DataFrame):
            # [CLASS DOC]
            # HELPER: empty DataFrame subclass used as the expected result type of the conversion.
            pass

        ser = SubclassedSeries([1, 2, 3], name="X")
        result = ser.to_frame()
        assert isinstance(result, SubclassedFrame)
        expected = SubclassedFrame({"X": [1, 2, 3]})
        tm.assert_frame_equal(result, expected)
