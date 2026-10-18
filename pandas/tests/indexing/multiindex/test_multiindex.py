# [MODULE DOC]
# Miscellaneous MultiIndex behavior tests collected in a single class: performance
# warnings on unsorted lookups, hashtable size cutoff, NaN / NaT / complex levels,
# frequency preservation, renaming, alignment with NaN, dtype preservation, constructing
# from tuples with NaN, and column assignment rules. Imports the internal index extension
# module to patch a module constant.
import numpy as np
import pytest

import pandas._libs.index as libindex

import pandas as pd
from pandas import (
    CategoricalDtype,
    DataFrame,
    Index,
    MultiIndex,
    Series,
)
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype


class TestMultiIndexBasic:
    # [CLASS DOC]
    # Test collection for assorted MultiIndex regression behaviors (see module doc).
    def test_multiindex_perf_warn(self, performance_warning):
        # [FUNCTION DOC]
        # Using the performance-warning fixture: a full-tuple lookup on an unsorted two-level frame
        # and a one-element-tuple lookup on a reordered frame must each emit that warning.
        df = DataFrame(
            {
                "jim": [0, 0, 1, 1],
                "joe": ["x", "x", "z", "y"],
                "jolie": np.random.default_rng(2).random(4),
            }
        ).set_index(["jim", "joe"])

        with tm.assert_produces_warning(performance_warning):
            df.loc[(1, "z")]

        df = df.iloc[[2, 1, 3, 0]]
        with tm.assert_produces_warning(performance_warning):
            df.loc[(0,)]

    @pytest.mark.parametrize("offset", [-5, 5])
    def test_indexing_over_hashtable_size_cutoff(self, monkeypatch, offset):
        # [FUNCTION DOC]
        # Parametrized with an offset of -5 or 5 relative to a patched internal size cutoff of 20, so
        # the index is below or above the point where the engine switches strategy. While the
        # constant is patched (restored by the monkeypatch context), full-tuple getitem on a series
        # with ("a", n) keys returns the correct values for positions 5, 6, 7.
        size_cutoff = 20
        n = size_cutoff + offset

        with monkeypatch.context():
            monkeypatch.setattr(libindex, "_SIZE_CUTOFF", size_cutoff)
            s = Series(np.arange(n), MultiIndex.from_arrays((["a"] * n, np.arange(n))))

            # hai it works!
            assert s[("a", 5)] == 5
            assert s[("a", 6)] == 6
            assert s[("a", 7)] == 7

    def test_multi_nan_indexing(self):
        # GH 3588
        # [FUNCTION DOC]
        # GH 3588. Setting two columns (one with a missing value) as index while keeping them as
        # columns gives a frame whose index is a MultiIndex built from the two level indexes, with
        # the NaN preserved.
        df = DataFrame(
            {
                "a": ["R1", "R2", np.nan, "R4"],
                "b": ["C1", "C2", "C3", "C4"],
                "c": [10, 15, np.nan, 20],
            }
        )
        result = df.set_index(["a", "b"], drop=False)
        expected = DataFrame(
            {
                "a": ["R1", "R2", np.nan, "R4"],
                "b": ["C1", "C2", "C3", "C4"],
                "c": [10, 15, np.nan, 20],
            },
            index=[
                Index(["R1", "R2", np.nan, "R4"], name="a"),
                Index(["C1", "C2", "C3", "C4"], name="b"),
            ],
        )
        tm.assert_frame_equal(result, expected)

    def test_exclusive_nat_column_indexing(self):
        # GH 38025
        # test multi indexing when one column exclusively contains NaT values
        # [FUNCTION DOC]
        # GH 38025. When one index column holds only NaT values, set_index still produces a frame
        # equal to the expected one with that column as level.
        df = DataFrame(
            {
                "a": [pd.NaT, pd.NaT, pd.NaT, pd.NaT],
                "b": ["C1", "C2", "C3", "C4"],
                "c": [10, 15, np.nan, 20],
            }
        )
        df = df.set_index(["a", "b"])
        expected = DataFrame(
            {
                "c": [10, 15, np.nan, 20],
            },
            index=[
                Index([pd.NaT, pd.NaT, pd.NaT, pd.NaT], name="a"),
                Index(["C1", "C2", "C3", "C4"], name="b"),
            ],
        )
        tm.assert_frame_equal(df, expected)

    def test_nested_tuples_duplicates(self):
        # GH#30892

        # [FUNCTION DOC]
        # GH 30892. A date/string MultiIndex with duplicate first entries: assigning a scalar via loc
        # with a tuple row key, and with a list holding one tuple, both set the matching row(s) of the
        # second column, giving the same frame.
        dti = pd.to_datetime(["20190101", "20190101", "20190102"])
        idx = Index(["a", "a", "c"])
        mi = MultiIndex.from_arrays([dti, idx], names=["index1", "index2"])

        df = DataFrame({"c1": [1, 2, 3], "c2": [np.nan, np.nan, np.nan]}, index=mi)

        expected = DataFrame({"c1": df["c1"], "c2": [1.0, 1.0, np.nan]}, index=mi)

        df2 = df.copy(deep=True)
        df2.loc[(dti[0], "a"), "c2"] = 1.0
        tm.assert_frame_equal(df2, expected)

        df3 = df.copy(deep=True)
        df3.loc[[(dti[0], "a")], "c2"] = 1.0
        tm.assert_frame_equal(df3, expected)

    def test_multiindex_with_datatime_level_preserves_freq(self):
        # https://github.com/pandas-dev/pandas/issues/35563
        # [FUNCTION DOC]
        # Issue 35563. Selecting one first-level label from a product index with a daily date level
        # returns an index equal to the date index and keeps its frequency.
        idx = Index(range(2), name="A")
        dti = pd.date_range("2020-01-01", periods=7, freq="D", name="B")
        mi = MultiIndex.from_product([idx, dti])
        df = DataFrame(np.random.default_rng(2).standard_normal((14, 2)), index=mi)
        result = df.loc[0].index
        tm.assert_index_equal(result, dti)
        assert result.freq == dti.freq

    def test_multiindex_complex(self):
        # GH#42145
        # [FUNCTION DOC]
        # GH 42145. Complex-valued and integer columns set as index in place give an index equal to a
        # MultiIndex built from the arrays with the same names.
        complex_data = [1 + 2j, 4 - 3j, 10 - 1j]
        non_complex_data = [3, 4, 5]
        result = DataFrame(
            {
                "x": complex_data,
                "y": non_complex_data,
                "z": non_complex_data,
            }
        )
        result.set_index(["x", "y"], inplace=True)
        expected = DataFrame(
            {"z": non_complex_data},
            index=MultiIndex.from_arrays(
                [complex_data, non_complex_data],
                names=("x", "y"),
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_rename_multiindex_with_duplicates(self):
        # GH 38015
        # [FUNCTION DOC]
        # GH 38015. Renaming a label in level 0 of a MultiIndex with duplicate entries in the second
        # level replaces only matching labels and keeps duplicates.
        mi = MultiIndex.from_tuples([("A", "cat"), ("B", "cat"), ("B", "cat")])
        df = DataFrame(index=mi)
        df = df.rename(index={"A": "Apple"}, level=0)

        mi2 = MultiIndex.from_tuples([("Apple", "cat"), ("B", "cat"), ("B", "cat")])
        expected = DataFrame(index=mi2)
        tm.assert_frame_equal(df, expected)

    def test_series_align_multiindex_with_nan_overlap_only(self):
        # GH 38439
        # [FUNCTION DOC]
        # GH 38439. Aligning two series whose MultiIndexes only share NaN entries produces the union
        # index (sorted with NaN last), with float values and NaN where one side is missing.
        mi1 = MultiIndex.from_arrays([[81.0, np.nan], [np.nan, np.nan]])
        mi2 = MultiIndex.from_arrays([[np.nan, 82.0], [np.nan, np.nan]])
        ser1 = Series([1, 2], index=mi1)
        ser2 = Series([1, 2], index=mi2)
        result1, result2 = ser1.align(ser2)

        mi = MultiIndex.from_arrays([[81.0, 82.0, np.nan], [np.nan, np.nan, np.nan]])
        expected1 = Series([1.0, np.nan, 2.0], index=mi)
        expected2 = Series([np.nan, 2.0, 1.0], index=mi)

        tm.assert_series_equal(result1, expected1)
        tm.assert_series_equal(result2, expected2)

    def test_series_align_multiindex_with_nan(self):
        # GH 38439
        # [FUNCTION DOC]
        # GH 38439. Aligning two series with the same entries in different order (including NaN) gives
        # both sides on the first's index order, with values reordered accordingly (dtype integer).
        mi1 = MultiIndex.from_arrays([[81.0, np.nan], [np.nan, np.nan]])
        mi2 = MultiIndex.from_arrays([[np.nan, 81.0], [np.nan, np.nan]])
        ser1 = Series([1, 2], index=mi1)
        ser2 = Series([1, 2], index=mi2)
        result1, result2 = ser1.align(ser2)

        mi = MultiIndex.from_arrays([[81.0, np.nan], [np.nan, np.nan]])
        expected1 = Series([1, 2], index=mi)
        expected2 = Series([2, 1], index=mi)

        tm.assert_series_equal(result1, expected1)
        tm.assert_series_equal(result2, expected2)

    def test_nunique_smoke(self):
        # GH 34019
        # [FUNCTION DOC]
        # GH 34019. Smoke test: the number of unique entries of a two-level index made of duplicated
        # rows equals 1.
        n = DataFrame([[1, 2], [1, 2]]).set_index([0, 1]).index.nunique()
        assert n == 1

    def test_multiindex_repeated_keys(self):
        # GH19414
        # [FUNCTION DOC]
        # GH 19414. Selecting a single-level MultiIndex series by a list with repeated labels returns
        # repeated entries in the list order.
        tm.assert_series_equal(
            Series([1, 2], MultiIndex.from_arrays([["a", "b"]])).loc[
                ["a", "a", "b", "b"]
            ],
            Series([1, 1, 2, 2], MultiIndex.from_arrays([["a", "a", "b", "b"]])),
        )

    def test_multiindex_with_na_missing_key(self):
        # GH46173
        # [FUNCTION DOC]
        # GH 46173. A list holding a one-tuple column key not in the columns (which contain a
        # missing label) raises a key error mentioning the missing label.
        df = DataFrame.from_dict(
            {
                ("foo",): [1, 2, 3],
                ("bar",): [5, 6, 7],
                (None,): [8, 9, 0],
            }
        )
        with pytest.raises(KeyError, match="missing_key"):
            df[[("missing_key",)]]

    def test_multiindex_dtype_preservation(self):
        # GH51261
        # [FUNCTION DOC]
        # GH 51261. After converting to categorical, taking the first-level column from a MultiIndex
        # keeps the categorical dtype; also a frame with MultiIndex columns assigned a nullable
        # boolean column keeps that extension dtype (analogue of a geopandas issue).
        columns = MultiIndex.from_tuples([("A", "B")], names=["lvl1", "lvl2"])
        df = DataFrame(["value"], columns=columns).astype("category")
        df_no_multiindex = df["A"]
        assert isinstance(df_no_multiindex["B"].dtype, CategoricalDtype)

        # geopandas 1763 analogue
        df = DataFrame(
            [[1, 0], [0, 1]],
            columns=[
                ["foo", "foo"],
                ["location", "location"],
                ["x", "y"],
            ],
        ).assign(bools=Series([True, False], dtype="boolean"))
        assert isinstance(df["bools"].dtype, BooleanDtype)

    def test_multiindex_from_tuples_with_nan(self):
        # GH#23578
        # [FUNCTION DOC]
        # GH 23578. Building a MultiIndex from tuples where one element is a bare NaN treats it as an
        # all-NaN row of the same length as others.
        result = MultiIndex.from_tuples([("a", "b", "c"), np.nan, ("d", "", "")])
        expected = MultiIndex.from_tuples(
            [("a", "b", "c"), (np.nan, np.nan, np.nan), ("d", "", "")]
        )
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("operation", ["div", "mul", "add", "sub"])
    def test_groupyby_rename_categories_operation_with_multiindex(self, operation):
        # GH#51500
        # [FUNCTION DOC]
        # GH 51500. Parametrized over four arithmetic method names. After renaming categories (swapping
        # two) before grouping, an arithmetic op between the value-counts series and group-size series
        # aligns by label, so the result equals the op with the second operand in reverse order.
        # KNOWN ISSUE: the method name has a typo (groupyby instead of groupby); behavior is unaffected.
        data = DataFrame(
            [["C", "B", "B"], ["B", "A", "A"], ["B", "A", "B"]], columns=["0", "1", "2"]
        )
        data["0"] = data["0"].astype("category")
        data["0"] = data["0"].cat.rename_categories({"C": "B", "B": "C"})

        a = data.groupby(by=["0", "1"])["2"].value_counts()
        b = data.groupby(by=["0", "1"]).size()

        result = getattr(a, operation)(b)
        expected = getattr(a, operation)(b.sort_index(ascending=False))

        tm.assert_series_equal(result, expected)

    def test_multiindex_assign_aligns_as_implicit_tuple(self):
        # GH 61841
        # [FUNCTION DOC]
        # GH 61841. For frames with two-level columns, assigning a series to a plain key behaves like
        # assigning to a tuple key with an empty second element (equal frames), and a later
        # reassignment of a scaled series is consistent with the earlier results.
        cols = MultiIndex.from_tuples([("A", "B")])
        df1 = DataFrame([[i] for i in range(3)], columns=cols)
        df2 = df1.copy()
        df3 = df1.copy()
        s1 = df1["A"].rolling(2).mean()
        s2 = s1.copy()
        s3 = s1.copy()

        df2["C"] = s2
        df3[("C", "")] = s3
        tm.assert_frame_equal(df2, df3)

        df1["C"] = s1
        tm.assert_frame_equal(df1, df2)
        tm.assert_frame_equal(df1, df3)

        df1["C"] = s1 * 2
        tm.assert_series_equal(df1["C"], df2["C"] * 2)

    def test_multiindex_assign_alignment_with_non_string_dtype(self):
        # GH 62518
        # [FUNCTION DOC]
        # GH 62518. For a MultiIndex with categorical second level, converting a selected group of
        # columns to integer yields a frame equal to one built with per-column series of the matching
        # dtypes (empty frame with object and int64 columns).
        columns = MultiIndex.from_arrays(
            [["a", "a", "z", "z"], pd.Categorical([1, 2, 1, 2])]
        )

        meta = DataFrame(columns=columns, dtype=object)
        meta["z"] = meta["z"].astype("int64")

        result = DataFrame(
            data={
                ("a", 1): Series([], dtype=object),
                ("a", 2): Series([], dtype=object),
                ("z", 1): Series([], dtype="int64"),
                ("z", 2): Series([], dtype="int64"),
            },
            columns=columns,
        )

        tm.assert_frame_equal(meta, result)

    def test_multiindex_assign_alignment_with_object_dtype(self):
        # https://github.com/pandas-dev/pandas/issues/65118
        # second level of the multiindex is object dtype, and the value of
        # that level is 0 for the single column we are setting

        # [FUNCTION DOC]
        # Issue 65118. With second level of mixed string and integer labels (object dtype), assigning
        # a computed block back under each first-level key (including a key with a single column labeled
        # zero) writes values into the right columns; result equals the data divided by 100.
        columns = MultiIndex.from_tuples(
            [("A", "M"), ("A", 0), ("B", 1), ("B", 2), ("C", 0)]
        )
        df = DataFrame(np.arange(20, dtype=float).reshape(4, 5), columns=columns)

        df["A"] = df["A"] / 100
        df["B"] = df["B"] / 100
        # this case specifically was the buggy one
        df["C"] = df["C"] / 100

        expected = DataFrame(
            np.arange(20, dtype=float).reshape(4, 5) / 100, columns=columns
        )
        tm.assert_frame_equal(df, expected)
