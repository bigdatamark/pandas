# [MODULE DOC]
# Tests for the private scalar setter on DataFrame, including enlargement, dtype effects and
# index changes. Uses the float_frame and using_infer_string fixtures. No import-time side effects.
import numpy as np
import pytest

from pandas.core.dtypes.common import is_float_dtype

from pandas import (
    DataFrame,
    isna,
)


class TestSetValue:
    # [CLASS DOC]
    # Test class for the private scalar value setter.
    def test_set_value(self, float_frame):
        # [FUNCTION DOC]
        # Verifies setting every row and column cell of the float frame to one via the private setter
        # makes each cell read back as one.
        for idx in float_frame.index:
            for col in float_frame.columns:
                float_frame._set_value(idx, col, 1)
                assert float_frame[col][idx] == 1

    def test_set_value_resize(self, float_frame, using_infer_string):
        # [FUNCTION DOC]
        # Verifies enlargement through the private setter: setting a new row label returns none and
        # appends the label at the end; assigning through label location adds a new column; setting a
        # string into a new column gives string (when inferring strings) or object dtype; a boolean
        # gives object; an integer gives a float column with missing values elsewhere; and setting a
        # string into that numeric column raises a type error about an invalid value.
        res = float_frame._set_value("foobar", "B", 0)
        assert res is None
        assert float_frame.index[-1] == "foobar"
        assert float_frame._get_value("foobar", "B") == 0

        float_frame.loc["foobar", "qux"] = 0
        assert float_frame._get_value("foobar", "qux") == 0

        res = float_frame.copy()
        res._set_value("foobar", "baz", "sam")
        if using_infer_string:
            assert res["baz"].dtype == "str"
        else:
            assert res["baz"].dtype == np.object_
        res = float_frame.copy()
        res._set_value("foobar", "baz", True)
        assert res["baz"].dtype == np.object_

        res = float_frame.copy()
        res._set_value("foobar", "baz", 5)
        assert is_float_dtype(res["baz"])
        assert isna(res["baz"].drop(["foobar"])).all()

        with pytest.raises(TypeError, match="Invalid value"):
            res._set_value("foobar", "baz", "sam")

    def test_set_value_with_index_dtype_change(self):
        # [FUNCTION DOC]
        # Verifies enlarging a frame with integer row labels using a new string row label appends it, for
        # the private setter and label location. With an integer column key the key is interpreted as
        # positional so no column is added (ambiguous case); with a new string column both row and
        # column are created. The commented-out column assertion in the source reflects this ambiguity.
        df_orig = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            index=range(3),
            columns=list("ABC"),
        )

        # this is actually ambiguous as the 2 is interpreted as a positional
        # so column is not created
        df = df_orig.copy()
        df._set_value("C", 2, 1.0)
        assert list(df.index) == [*list(df_orig.index), "C"]
        # assert list(df.columns) == list(df_orig.columns) + [2]

        df = df_orig.copy()
        df.loc["C", 2] = 1.0
        assert list(df.index) == [*list(df_orig.index), "C"]
        # assert list(df.columns) == list(df_orig.columns) + [2]

        # create both new
        df = df_orig.copy()
        df._set_value("C", "D", 1.0)
        assert list(df.index) == [*list(df_orig.index), "C"]
        assert list(df.columns) == [*list(df_orig.columns), "D"]

        df = df_orig.copy()
        df.loc["C", "D"] = 1.0
        assert list(df.index) == [*list(df_orig.index), "C"]
        assert list(df.columns) == [*list(df_orig.columns), "D"]
