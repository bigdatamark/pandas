# [MODULE DOC]
# DataFrame-specific tests living in the generic test package: axis naming,
# truth-value ambiguity, metadata propagation through groupby, resample, merge and
# concat,
# attribute versus column assignment, deep copy of empty frames, and argument
# validation.
# Contains two test classes.
from copy import deepcopy
from operator import methodcaller
from typing import Literal

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    MultiIndex,
    Series,
    date_range,
)
import pandas._testing as tm


class TestDataFrame:
    # [CLASS DOC]
    # Groups DataFrame tests for axis naming, metadata propagation, attribute setting
    # and copying. Stateless.
    @pytest.mark.parametrize("func", ["_set_axis_name", "rename_axis"])
    def test_set_axis_name(self, func):
        # [FUNCTION DOC]
        # Verifies that setting an axis name (through the private setter or the public
        # rename method) returns a
        # frame with the index name set, then with the column name set when axis is one,
        # leaving the original unchanged.
        # Parametrized over the two method names.
        df = DataFrame([[1, 2], [3, 4]])

        result = methodcaller(func, "foo")(df)
        assert df.index.name is None
        assert result.index.name == "foo"

        result = methodcaller(func, "cols", axis=1)(df)
        assert df.columns.name is None
        assert result.columns.name == "cols"

    @pytest.mark.parametrize("func", ["_set_axis_name", "rename_axis"])
    def test_set_axis_name_mi(self, func):
        # [FUNCTION DOC]
        # Verifies that setting axis names with a list of level names on a frame with
        # row and column multi-indexes only changes the targeted axis (rows by default,
        # columns for axis one);
        # the other axis keeps unnamed levels. Parametrized over the two method names.
        df = DataFrame(
            np.empty((3, 3)),
            index=MultiIndex.from_tuples([("A", x) for x in list("aBc")]),
            columns=MultiIndex.from_tuples([("C", x) for x in list("xyz")]),
        )

        level_names = ["L1", "L2"]

        result = methodcaller(func, level_names)(df)
        assert result.index.names == level_names
        assert result.columns.names == [None, None]

        result = methodcaller(func, level_names, axis=1)(df)
        assert result.columns.names == ["L1", "L2"]
        assert result.index.names == [None, None]

    def test_nonzero_single_element(self):
        # [FUNCTION DOC]
        # Verifies that converting a one-element boolean frame to a bool raises a value
        # error about ambiguous truth value.
        df = DataFrame([[False, False]])
        msg_err = "The truth value of a DataFrame is ambiguous"
        with pytest.raises(ValueError, match=msg_err):
            bool(df)

    def test_metadata_propagation_indiv_groupby(self):
        # [FUNCTION DOC]
        # Verifies that metadata of a frame survives a groupby sum, using the
        # metadata-equivalence assertion helper.
        # groupby
        df = DataFrame(
            {
                "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
                "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
                "C": np.random.default_rng(2).standard_normal(8),
                "D": np.random.default_rng(2).standard_normal(8),
            }
        )
        result = df.groupby("A").sum()
        tm.assert_metadata_equivalent(df, result)

    def test_metadata_propagation_indiv_resample(self):
        # [FUNCTION DOC]
        # Verifies that metadata of a frame is preserved on the resampler object of a
        # one-thousand-row
        # second-frequency frame resampled to one minute.
        # resample
        df = DataFrame(
            np.random.default_rng(2).standard_normal((1000, 2)),
            index=date_range("20130101", periods=1000, freq="s"),
        )
        result = df.resample("1min")
        tm.assert_metadata_equivalent(df, result)

    def test_metadata_propagation_indiv(self, monkeypatch):
        # [FUNCTION DOC]
        # Verifies customized metadata propagation (GH 6923, GH#6927) by temporarily
        # replacing the
        # frame metadata list with a single filename attribute and the finalize hook
        # with a custom one. Merge must
        # join the two inputs' filenames with a pipe; concat must join non-empty
        # filenames with a plus.
        # Uses monkeypatch so class attributes are restored afterwards.
        # merging with override
        # GH 6923

        def finalize(
            self: DataFrame,
            other: DataFrame,
            method: Literal["merge", "concat"] | None = None,
            **kwargs,
        ):
            # [FUNCTION DOC]
            # HELPER: replacement finalize hook installed by the enclosing test.
            # Parameters: self is the new frame, other is the
            # source (or an object holding the input objects for merge and concat),
            # method is merge, concat or none, extra keywords ignored.
            # For each metadata name: on merge, sets the pipe-joined values of the left
            # and right inputs (default empty string); on concat,
            # sets the plus-joined truthy values of the inputs; otherwise copies from
            # other (default empty string). Returns self.
            for name in self._metadata:
                if method == "merge":
                    left, right = other.input_objs
                    value = getattr(left, name, "") + "|" + getattr(right, name, "")
                    object.__setattr__(self, name, value)
                elif method == "concat":
                    value = "+".join(
                        [
                            getattr(o, name)
                            for o in other.input_objs
                            if getattr(o, name, None)
                        ]
                    )
                    object.__setattr__(self, name, value)
                else:
                    object.__setattr__(self, name, getattr(other, name, ""))

            return self

        with monkeypatch.context() as m:
            m.setattr(DataFrame, "_metadata", ["filename"])
            m.setattr(DataFrame, "__finalize__", finalize)

            df1 = DataFrame(
                np.random.default_rng(2).integers(0, 4, (3, 2)), columns=["a", "b"]
            )
            df2 = DataFrame(
                np.random.default_rng(2).integers(0, 4, (3, 2)), columns=["c", "d"]
            )
            DataFrame._metadata = ["filename"]
            df1.filename = "fname1.csv"
            df2.filename = "fname2.csv"

            result = df1.merge(df2, left_on=["a"], right_on=["c"], how="inner")
            assert result.filename == "fname1.csv|fname2.csv"

            # concat
            # GH#6927
            df1 = DataFrame(
                np.random.default_rng(2).integers(0, 4, (3, 2)), columns=list("ab")
            )
            df1.filename = "foo"

            result = pd.concat([df1, df1])
            assert result.filename == "foo+foo"

    def test_set_attribute(self):
        # [FUNCTION DOC]
        # Verifies attribute assignment when an attribute and a column share a name
        # (Issue 8994): the attribute and the
        # column coexist, the attribute keeps the last assigned scalar and the column
        # keeps its own data.
        # Test for consistent setattr behavior when an attribute and a column
        # have the same name (Issue #8994)
        df = DataFrame({"x": [1, 2, 3]})

        df.y = 2
        df["y"] = [2, 4, 6]
        df.y = 5

        assert df.y == 5
        tm.assert_series_equal(df["y"], Series([2, 4, 6], name="y"))

    def test_deepcopy_empty(self):
        # [FUNCTION DOC]
        # Verifies deep copying a frame with no rows but one column (GH15370) gives an
        # equal frame.
        # This test covers empty frame copying with non-empty column sets
        # as reported in issue GH15370
        empty_frame = DataFrame(data=[], index=[], columns=["A"])
        empty_frame_copy = deepcopy(empty_frame)

        tm.assert_frame_equal(empty_frame_copy, empty_frame)


# formerly in Generic but only test DataFrame
class TestDataFrame2:
    # [CLASS DOC]
    # Groups DataFrame tests formerly in the generic test module; they only exercise
    # frames. Stateless.
    @pytest.mark.parametrize("value", [1, "True", [1, 2, 3], 5.0])
    def test_validate_bool_args(self, value):
        # [FUNCTION DOC]
        # Verifies that a non-bool inplace value raises a value error ("expected type
        # bool") for rename_axis,
        # drop, fillna, replace, interpolate, the private where and mask methods.
        # Parametrized over invalid values (int,
        # string, list, float); each call is made on a copy.
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        msg = 'For argument "inplace" expected type bool, received type'
        with pytest.raises(ValueError, match=msg):
            df.copy().rename_axis(mapper={"a": "x", "b": "y"}, axis=1, inplace=value)

        with pytest.raises(ValueError, match=msg):
            df.copy().drop("a", axis=1, inplace=value)

        with pytest.raises(ValueError, match=msg):
            df.copy().fillna(value=0, inplace=value)

        with pytest.raises(ValueError, match=msg):
            df.copy().replace(to_replace=1, value=7, inplace=value)

        with pytest.raises(ValueError, match=msg):
            df.copy().interpolate(inplace=value)

        with pytest.raises(ValueError, match=msg):
            df.copy()._where(cond=df.a > 2, inplace=value)

        with pytest.raises(ValueError, match=msg):
            df.copy().mask(cond=df.a > 2, inplace=value)

    def test_unexpected_keyword(self):
        # [FUNCTION DOC]
        # Verifies that an unexpected keyword (misspelled in_place, or inplace on
        # reindex) raises a type error
        # mentioning unexpected keyword, for frame drop, frame reindex, categorical
        # fillna and series fillna (GH8597).
        # GH8597
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), columns=["jim", "joe"]
        )
        ca = pd.Categorical([0, 0, 2, 2, 3, np.nan])
        ts = df["joe"].copy()
        ts[2] = np.nan

        msg = "unexpected keyword"
        with pytest.raises(TypeError, match=msg):
            df.drop("joe", axis=1, in_place=True)

        with pytest.raises(TypeError, match=msg):
            df.reindex([1, 0], inplace=True)

        with pytest.raises(TypeError, match=msg):
            ca.fillna(0, inplace=True)

        with pytest.raises(TypeError, match=msg):
            ts.fillna(0, in_place=True)
