# [MODULE DOC]
# Tests for how the concatenate function builds and names the resulting index
# and columns: ignored index, index-name resolution, copying vs sharing of axes,
# default range indexes, duplicate labels, multi-level indexes with keys and
# levels, and dtype retention of the column index. Two classes (general index and
# multi-level index) are defined. Uses fixtures sort, axis, performance_warning,
# using_infer_string, any_numeric_ea_dtype and multiindex_dataframe_random_data
# from conftest.
from copy import deepcopy

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    concat,
)
import pandas._testing as tm


class TestIndexConcat:
    # [CLASS DOC]
    # Test collection for index handling in concatenation. Stateless.
    def test_concat_ignore_index(self, sort):
        # [FUNCTION DOC]
        # Parametrized by sort. Verifies column-wise concatenation with ignored index of
        # frames with partly overlapping row labels: the row index is the union (sorted
        # when sort is true, otherwise first-appearance order) while the columns are
        # renumbered.
        frame1 = DataFrame(
            {"test1": ["a", "b", "c"], "test2": [1, 2, 3], "test3": [4.5, 3.2, 1.2]}
        )
        frame2 = DataFrame({"test3": [5.2, 2.2, 4.3]})
        frame1.index = Index(["x", "y", "z"])
        frame2.index = Index(["x", "y", "q"])

        v1 = concat([frame1, frame2], axis=1, ignore_index=True, sort=sort)

        nan = np.nan
        expected = DataFrame(
            [
                [nan, nan, nan, 4.3],
                ["a", 1, 4.5, 5.2],
                ["b", 2, 3.2, 2.2],
                ["c", 3, 1.2, nan],
            ],
            index=Index(["q", "x", "y", "z"]),
        )
        if not sort:
            expected = expected.loc[["x", "y", "z", "q"]]

        tm.assert_frame_equal(v1, expected)

    @pytest.mark.parametrize(
        "name_in1,name_in2,name_in3,name_out",
        [
            ("idx", "idx", "idx", "idx"),
            ("idx", "idx", None, None),
            ("idx", None, None, None),
            ("idx1", "idx2", None, None),
            ("idx1", "idx1", "idx2", None),
            ("idx1", "idx2", "idx3", None),
            (None, None, None, None),
        ],
    )
    def test_concat_same_index_names(self, name_in1, name_in2, name_in3, name_out):
        # [FUNCTION DOC]
        # Parametrized over name triples and the expected output name. Verifies that for
        # column-wise concatenation of three frames with different indexes the union index
        # keeps a name only when all inputs share the same name; otherwise the name is
        # unset (GH 13475).
        # GH13475
        indices = [
            Index(["a", "b", "c"], name=name_in1),
            Index(["b", "c", "d"], name=name_in2),
            Index(["c", "d", "e"], name=name_in3),
        ]
        frames = [
            DataFrame({c: [0, 1, 2]}, index=i) for i, c in zip(indices, ["x", "y", "z"])
        ]
        result = concat(frames, axis=1)

        exp_ind = Index(["a", "b", "c", "d", "e"], name=name_out)
        expected = DataFrame(
            {
                "x": [0, 1, 2, np.nan, np.nan],
                "y": [np.nan, 0, 1, 2, np.nan],
                "z": [np.nan, np.nan, 0, 1, 2],
            },
            index=exp_ind,
        )

        tm.assert_frame_equal(result, expected)

    def test_concat_rename_index(self):
        # [FUNCTION DOC]
        # Verifies that when keys and two names are given, and the inputs' indexes have
        # different names, the result gets both level names (the second name replacing the
        # inner level name).
        a = DataFrame(
            np.random.default_rng(2).random((3, 3)),
            columns=list("ABC"),
            index=Index(list("abc"), name="index_a"),
        )
        b = DataFrame(
            np.random.default_rng(2).random((3, 3)),
            columns=list("ABC"),
            index=Index(list("abc"), name="index_b"),
        )

        result = concat([a, b], keys=["key0", "key1"], names=["lvl0", "lvl1"])

        exp = concat([a, b], keys=["key0", "key1"], names=["lvl0"])
        names = list(exp.index.names)
        names[1] = "lvl1"
        exp.index.set_names(names, inplace=True)

        tm.assert_frame_equal(result, exp)
        assert result.index.names == exp.index.names

    def test_concat_copy_index_series(self, axis):
        # [FUNCTION DOC]
        # Takes the axis fixture. Verifies that concatenating a series with itself yields
        # a new index along rows, but reuses the same index object along columns
        # (GH 29879).
        # GH 29879
        ser = Series([1, 2])
        comb = concat([ser, ser], axis=axis)
        if axis in [0, "index"]:
            assert comb.index is not ser.index
        else:
            assert comb.index is ser.index

    def test_concat_copy_index_frame(self, axis):
        # [FUNCTION DOC]
        # Takes the axis fixture. Verifies for frames that the concatenated axis is a new
        # index while the other axis is the identical object as the source (GH 29879).
        # GH 29879
        df = DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
        comb = concat([df, df], axis=axis)
        if axis in [0, "index"]:
            assert not comb.index.is_(df.index)
            assert comb.columns.is_(df.columns)
        elif axis in [1, "columns"]:
            assert comb.index.is_(df.index)
            assert not comb.columns.is_(df.columns)

    def test_default_index(self):
        # [FUNCTION DOC]
        # Verifies that column-wise concatenation of series with ignored index, or all
        # unnamed, gives a range-index columns, and row/column concatenation of frames
        # with ignored index gives range-index labels (checked via strict index/column
        # type checks).
        # is_series and ignore_index
        s1 = Series([1, 2, 3], name="x")
        s2 = Series([4, 5, 6], name="y")
        res = concat([s1, s2], axis=1, ignore_index=True)
        assert isinstance(res.columns, pd.RangeIndex)
        exp = DataFrame([[1, 4], [2, 5], [3, 6]])
        # use check_index_type=True to check the result have
        # RangeIndex (default index)
        tm.assert_frame_equal(res, exp, check_index_type=True, check_column_type=True)

        # is_series and all inputs have no names
        s1 = Series([1, 2, 3])
        s2 = Series([4, 5, 6])
        res = concat([s1, s2], axis=1, ignore_index=False)
        assert isinstance(res.columns, pd.RangeIndex)
        exp = DataFrame([[1, 4], [2, 5], [3, 6]])
        exp.columns = pd.RangeIndex(2)
        tm.assert_frame_equal(res, exp, check_index_type=True, check_column_type=True)

        # is_dataframe and ignore_index
        df1 = DataFrame({"A": [1, 2], "B": [5, 6]})
        df2 = DataFrame({"A": [3, 4], "B": [7, 8]})

        res = concat([df1, df2], axis=0, ignore_index=True)
        exp = DataFrame([[1, 5], [2, 6], [3, 7], [4, 8]], columns=["A", "B"])
        tm.assert_frame_equal(res, exp, check_index_type=True, check_column_type=True)

        res = concat([df1, df2], axis=1, ignore_index=True)
        exp = DataFrame([[1, 5, 3, 7], [2, 6, 4, 8]])
        tm.assert_frame_equal(res, exp, check_index_type=True, check_column_type=True)

    def test_dups_index(self):
        # [FUNCTION DOC]
        # Verifies concatenation of frames with duplicate column labels along both axes
        # (single and mixed dtypes) preserves the parts exactly (GH 4771).
        # GH 4771

        # single dtypes
        df = DataFrame(
            np.random.default_rng(2).integers(0, 10, size=40).reshape(10, 4),
            columns=["A", "A", "C", "C"],
        )

        result = concat([df, df], axis=1)
        tm.assert_frame_equal(result.iloc[:, :4], df)
        tm.assert_frame_equal(result.iloc[:, 4:], df)

        result = concat([df, df], axis=0)
        tm.assert_frame_equal(result.iloc[:10], df)
        tm.assert_frame_equal(result.iloc[10:], df)

        # multi dtypes
        df = concat(
            [
                DataFrame(
                    np.random.default_rng(2).standard_normal((10, 4)),
                    columns=["A", "A", "B", "B"],
                ),
                DataFrame(
                    np.random.default_rng(2).integers(0, 10, size=20).reshape(10, 2),
                    columns=["A", "C"],
                ),
            ],
            axis=1,
        )

        result = concat([df, df], axis=1)
        tm.assert_frame_equal(result.iloc[:, :6], df)
        tm.assert_frame_equal(result.iloc[:, 6:], df)

        result = concat([df, df], axis=0)
        tm.assert_frame_equal(result.iloc[:10], df)
        tm.assert_frame_equal(result.iloc[10:], df)


class TestMultiIndexConcat:
    # [CLASS DOC]
    # Test collection for concatenation involving multi-level indexes. Stateless.
    def test_concat_multiindex_with_keys(self, multiindex_dataframe_random_data):
        # [FUNCTION DOC]
        # Verifies that keys with a name add an outer level named iteration to a
        # multi-level index (three levels in total), and each key selects the original
        # frame.
        frame = multiindex_dataframe_random_data
        index = frame.index
        result = concat([frame, frame], keys=[0, 1], names=["iteration"])

        assert result.index.names == ("iteration", *index.names)
        tm.assert_frame_equal(result.loc[0], frame)
        tm.assert_frame_equal(result.loc[1], frame)
        assert result.index.nlevels == 3

    def test_concat_multiindex_with_none_in_index_names(self):
        # [FUNCTION DOC]
        # Verifies the result index when an input index level has no name and one name is
        # supplied for the keys level; also for inputs of different lengths (GH 15787).
        # GH 15787
        index = MultiIndex.from_product([[1], range(5)], names=["level1", None])
        df = DataFrame({"col": range(5)}, index=index, dtype=np.int32)

        result = concat([df, df], keys=[1, 2], names=["level2"])
        index = MultiIndex.from_product(
            [[1, 2], [1], range(5)], names=["level2", "level1", None]
        )
        expected = DataFrame({"col": list(range(5)) * 2}, index=index, dtype=np.int32)
        tm.assert_frame_equal(result, expected)

        result = concat([df, df[:2]], keys=[1, 2], names=["level2"])
        level2 = [1] * 5 + [2] * 2
        level1 = [1] * 7
        no_name = list(range(5)) + list(range(2))
        tuples = list(zip(level2, level1, no_name))
        index = MultiIndex.from_tuples(tuples, names=["level2", "level1", None])
        expected = DataFrame({"col": no_name}, index=index, dtype=np.int32)
        tm.assert_frame_equal(result, expected)

    def test_concat_multiindex_rangeindex(self):
        # [FUNCTION DOC]
        # Verifies row concatenation of selections of a frame whose multi-level index
        # levels are range indexes, including a length-one part, equals the combined
        # selection (GH 13542).
        # GH13542
        # when multi-index levels are RangeIndex objects
        # there is a bug in concat with objects of len 1

        df = DataFrame(np.random.default_rng(2).standard_normal((9, 2)))
        df.index = MultiIndex(
            levels=[pd.RangeIndex(3), pd.RangeIndex(3)],
            codes=[np.repeat(np.arange(3), 3), np.tile(np.arange(3), 3)],
        )

        res = concat([df.iloc[[2, 3, 4], :], df.iloc[[5], :]])
        exp = df.iloc[[2, 3, 4, 5], :]
        tm.assert_frame_equal(res, exp)

    def test_concat_multiindex_dfs_with_deepcopy(self):
        # [FUNCTION DOC]
        # Verifies that concatenating a dict of frames with multi-level indexes and a name
        # for the key level gives the expected three-level index, for both the dict and a
        # deep copy of it (GH 9967).
        # GH 9967
        example_multiindex1 = MultiIndex.from_product([["a"], ["b"]])
        example_dataframe1 = DataFrame([0], index=example_multiindex1)

        example_multiindex2 = MultiIndex.from_product([["a"], ["c"]])
        example_dataframe2 = DataFrame([1], index=example_multiindex2)

        example_dict = {"s1": example_dataframe1, "s2": example_dataframe2}
        expected_index = MultiIndex(
            levels=[["s1", "s2"], ["a"], ["b", "c"]],
            codes=[[0, 1], [0, 0], [0, 1]],
            names=["testname", None, None],
        )
        expected = DataFrame([[0], [1]], index=expected_index)
        result_copy = concat(deepcopy(example_dict), names=["testname"])
        tm.assert_frame_equal(result_copy, expected)
        result_no_copy = concat(example_dict, names=["testname"])
        tm.assert_frame_equal(result_no_copy, expected)

    @pytest.mark.parametrize(
        "mi1_list",
        [
            [["a"], range(2)],
            [["b"], np.arange(2.0, 4.0)],
            [["c"], ["A", "B"]],
            [["d"], pd.date_range(start="2017", end="2018", periods=2)],
        ],
    )
    @pytest.mark.parametrize(
        "mi2_list",
        [
            [["a"], range(2)],
            [["b"], np.arange(2.0, 4.0)],
            [["c"], ["A", "B"]],
            [["d"], pd.date_range(start="2017", end="2018", periods=2)],
        ],
    )
    def test_concat_with_various_multiindex_dtypes(
        self, mi1_list: list, mi2_list: list
    ):
        # [FUNCTION DOC]
        # Parametrized over pairs of descriptions for the product of two level lists
        # (integers, floats, strings, datetimes). Verifies column-wise concatenation of
        # frames with multi-level columns of various level dtypes gives the expected union
        # columns without warnings (GH 23478).
        # GitHub #23478
        mi1 = MultiIndex.from_product(mi1_list)
        mi2 = MultiIndex.from_product(mi2_list)

        df1 = DataFrame(np.zeros((1, len(mi1))), columns=mi1)
        df2 = DataFrame(np.zeros((1, len(mi2))), columns=mi2)

        if mi1_list[0] == mi2_list[0]:
            expected_mi = MultiIndex(
                levels=[mi1_list[0], list(mi1_list[1])],
                codes=[[0, 0, 0, 0], [0, 1, 0, 1]],
            )
        else:
            expected_mi = MultiIndex(
                levels=[
                    mi1_list[0] + mi2_list[0],
                    list(mi1_list[1]) + list(mi2_list[1]),
                ],
                codes=[[0, 0, 1, 1], [0, 1, 2, 3]],
            )

        expected_df = DataFrame(np.zeros((1, len(expected_mi))), columns=expected_mi)

        with tm.assert_produces_warning(None):
            result_df = concat((df1, df2), axis=1)

        tm.assert_frame_equal(expected_df, result_df)

    def test_concat_multiindex_(self):
        # [FUNCTION DOC]
        # Verifies that concatenating a single frame with a duplicate-labelled index under
        # one key gives a multi-level index equal to a product of the key and the index
        # values (GH 44786).
        # GitHub #44786
        df = DataFrame({"col": ["a", "b", "c"]}, index=["1", "2", "2"])
        df = concat([df], keys=["X"])

        iterables = [["X"], ["1", "2", "2"]]
        result_index = df.index
        expected_index = MultiIndex.from_product(iterables)

        tm.assert_index_equal(result_index, expected_index)

        result_df = df
        expected_df = DataFrame(
            {"col": ["a", "b", "c"]}, index=MultiIndex.from_product(iterables)
        )
        tm.assert_frame_equal(result_df, expected_df)

    def test_concat_with_key_not_unique(self, performance_warning):
        # [FUNCTION DOC]
        # Takes the performance_warning fixture. Verifies concatenation with repeated keys
        # produces a multi-level index which is not lexsorted; selecting a label emits a
        # performance warning about indexing past lexsort depth, and the selection equals
        # that of an equivalent hand-built frame, for single-row and multi-row parts
        # (GH 46519).
        # GitHub #46519
        df1 = DataFrame({"name": [1]})
        df2 = DataFrame({"name": [2]})
        df3 = DataFrame({"name": [3]})
        df_a = concat([df1, df2, df3], keys=["x", "y", "x"])
        # the warning is caused by indexing unsorted multi-index
        with tm.assert_produces_warning(
            performance_warning, match="indexing past lexsort depth"
        ):
            out_a = df_a.loc[("x", 0), :]
        df_b = DataFrame(
            {"name": [1, 2, 3]},
            index=MultiIndex(
                levels=[["x", "y"], range(1)], codes=[[0, 1, 0], [0, 0, 0]]
            ),
        )
        with tm.assert_produces_warning(
            performance_warning, match="indexing past lexsort depth"
        ):
            out_b = df_b.loc[("x", 0)]

        tm.assert_frame_equal(out_a, out_b)

        df1 = DataFrame({"name": ["a", "a", "b"]})
        df2 = DataFrame({"name": ["a", "b"]})
        df3 = DataFrame({"name": ["c", "d"]})
        df_a = concat([df1, df2, df3], keys=["x", "y", "x"])
        with tm.assert_produces_warning(
            performance_warning, match="indexing past lexsort depth"
        ):
            out_a = df_a.loc[("x", 0), :]

        df_b = DataFrame(
            {
                "a": ["x", "x", "x", "y", "y", "x", "x"],
                "b": [0, 1, 2, 0, 1, 0, 1],
                "name": list("aababcd"),
            }
        ).set_index(["a", "b"])
        df_b.index.names = [None, None]
        with tm.assert_produces_warning(
            performance_warning, match="indexing past lexsort depth"
        ):
            out_b = df_b.loc[("x", 0), :]

        tm.assert_frame_equal(out_a, out_b)

    def test_concat_with_duplicated_levels(self):
        # [FUNCTION DOC]
        # Verifies a value error saying level values are not unique when the passed
        # levels contain duplicates.
        # keyword levels should be unique
        df1 = DataFrame({"A": [1]}, index=["x"])
        df2 = DataFrame({"A": [1]}, index=["y"])
        msg = r"Level values not unique: \['x', 'y', 'y'\]"
        with pytest.raises(ValueError, match=msg):
            concat([df1, df2], keys=["x", "y"], levels=[["x", "y", "y"]])

    @pytest.mark.parametrize("levels", [[["x", "y"]], [["x", "y", "y"]]])
    def test_concat_with_levels_with_none_keys(self, levels):
        # [FUNCTION DOC]
        # Parametrized over two level lists. Verifies a value error saying levels are only
        # supported when keys is given.
        df1 = DataFrame({"A": [1]}, index=["x"])
        df2 = DataFrame({"A": [1]}, index=["y"])
        msg = "levels supported only when keys is not None"
        with pytest.raises(ValueError, match=msg):
            concat([df1, df2], levels=levels)

    def test_concat_range_index_result(self):
        # [FUNCTION DOC]
        # Verifies column-wise concatenation with sorting of frames with default indexes
        # returns a strictly range-typed index (GH 47501).
        # GH#47501
        df1 = DataFrame({"a": [1, 2]})
        df2 = DataFrame({"b": [1, 2]})

        result = concat([df1, df2], sort=True, axis=1)
        expected = DataFrame({"a": [1, 2], "b": [1, 2]})
        tm.assert_frame_equal(result, expected)
        expected_index = pd.RangeIndex(0, 2)
        tm.assert_index_equal(result.index, expected_index, exact=True)

    def test_concat_index_keep_dtype(self):
        # [FUNCTION DOC]
        # Verifies that outer concatenation with sorting of frames with object-dtype integer
        # column labels keeps the object column index dtype (GH 47329).
        # GH#47329
        df1 = DataFrame([[0, 1, 1]], columns=Index([1, 2, 3], dtype="object"))
        df2 = DataFrame([[0, 1]], columns=Index([1, 2], dtype="object"))
        result = concat([df1, df2], ignore_index=True, join="outer", sort=True)
        expected = DataFrame(
            [[0, 1, 1.0], [0, 1, np.nan]], columns=Index([1, 2, 3], dtype="object")
        )
        tm.assert_frame_equal(result, expected)

    def test_concat_index_keep_dtype_ea_numeric(self, any_numeric_ea_dtype):
        # [FUNCTION DOC]
        # Same as above but column labels use each numeric extension dtype from the
        # fixture; the result keeps that dtype (GH 47329).
        # GH#47329
        df1 = DataFrame(
            [[0, 1, 1]], columns=Index([1, 2, 3], dtype=any_numeric_ea_dtype)
        )
        df2 = DataFrame([[0, 1]], columns=Index([1, 2], dtype=any_numeric_ea_dtype))
        result = concat([df1, df2], ignore_index=True, join="outer", sort=True)
        expected = DataFrame(
            [[0, 1, 1.0], [0, 1, np.nan]],
            columns=Index([1, 2, 3], dtype=any_numeric_ea_dtype),
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["Int8", "Int16", "Int32"])
    def test_concat_index_find_common(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over smaller nullable integer dtypes for the first frame's columns.
        # Verifies the result's column index takes the common dtype with the 32-bit
        # nullable integer dtype (GH 47329).
        # GH#47329
        df1 = DataFrame([[0, 1, 1]], columns=Index([1, 2, 3], dtype=dtype))
        df2 = DataFrame([[0, 1]], columns=Index([1, 2], dtype="Int32"))
        result = concat([df1, df2], ignore_index=True, join="outer", sort=True)
        expected = DataFrame(
            [[0, 1, 1.0], [0, 1, np.nan]], columns=Index([1, 2, 3], dtype="Int32")
        )
        tm.assert_frame_equal(result, expected)

    def test_concat_axis_1_sort_false_rangeindex(self, using_infer_string):
        # [FUNCTION DOC]
        # Verifies column-wise outer concatenation without sorting of four string series
        # (including an empty one) gives a range-index result of the expected string/object
        # dtype with missing fill (GH 46675).
        # GH 46675
        s1 = Series(["a", "b", "c"])
        s2 = Series(["a", "b"])
        s3 = Series(["a", "b", "c", "d"])
        s4 = Series([], dtype=object if not using_infer_string else "str")
        result = concat(
            [s1, s2, s3, s4], sort=False, join="outer", ignore_index=False, axis=1
        )
        expected = DataFrame(
            [
                ["a"] * 3 + [np.nan],
                ["b"] * 3 + [np.nan],
                ["c", np.nan] * 2,
                [np.nan] * 2 + ["d"] + [np.nan],
            ],
            dtype=object if not using_infer_string else "str",
        )
        tm.assert_frame_equal(
            result, expected, check_index_type=True, check_column_type=True
        )
