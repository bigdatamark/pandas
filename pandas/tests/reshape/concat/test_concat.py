# [MODULE DOC]
# General tests for the concatenate function on series and frames: group keys and
# levels, mappings and iterables as input, None handling, memory sharing,
# dtype upcasting rules (numeric, extension, sparse, bool), subclass and
# attribute propagation, empty and all-missing inputs, duplicate keys and indexes,
# error messages, and mixed series/frame inputs. Relies on shared fixtures such as
# sort, frame_or_series, float_numpy_dtype, any_signed_int_numpy_dtype and
# non_dict_mapping_subclass from conftest, and the decimal extension test array.
from collections import (
    abc,
    deque,
)
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
import itertools

import numpy as np
import pytest

from pandas.errors import (
    InvalidIndexError,
    Pandas4Warning,
)

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    PeriodIndex,
    RangeIndex,
    Series,
    concat,
    date_range,
)
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tests.extension.decimal import to_decimal


class TestConcatenate:
    # [CLASS DOC]
    # Collects general behavior tests of concatenate. Stateless; each method builds
    # its own random or deterministic data.
    def test_append_concat(self):
        # [FUNCTION DOC]
        # Verifies that concatenating two series with period-converted date indexes keeps
        # a period index and the first element of the index (GH 1815).
        # GH#1815
        d1 = date_range("12/31/1990", "12/31/1999", freq="YE-DEC")
        d2 = date_range("12/31/2000", "12/31/2009", freq="YE-DEC")

        s1 = Series(np.random.default_rng(2).standard_normal(10), d1)
        s2 = Series(np.random.default_rng(2).standard_normal(10), d2)

        s1 = s1.to_period()
        s2 = s2.to_period()

        # drops index
        result = concat([s1, s2])
        assert isinstance(result.index, PeriodIndex)
        assert result.index[0] == s1.index[0]

    def test_concat_copy(self):
        # [FUNCTION DOC]
        # Verifies memory sharing: when concatenating along columns, the float and integer
        # blocks of the result share memory with the source blocks (no copy), and after
        # adding a second float frame (so blocks are consolidated) the float block is a
        # view on one of the sources, the integer block a view of the integer source and
        # the object block a view of the string source.
        # KNOWN ISSUE: for the string dtype
        # branch the shares-memory helper is called without an assertion, so that check
        # verifies nothing.
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 3)))
        df2 = DataFrame(np.random.default_rng(2).integers(0, 10, size=4).reshape(4, 1))
        df3 = DataFrame({5: "foo"}, index=range(4))

        result = concat([df, df2, df3], axis=1)

        for block in result._mgr.blocks:
            arr = block.values
            if arr.dtype.kind == "f":
                assert arr.base is df._mgr.blocks[0].values.base
            elif arr.dtype.kind in ["i", "u"]:
                assert arr.base is df2._mgr.blocks[0].values.base
            elif arr.dtype == object:
                assert arr.base is not None
            elif arr.dtype == "string":
                tm.shares_memory(arr, df3._mgr.blocks[0].values)

        # Float block was consolidated.
        df4 = DataFrame(np.random.default_rng(2).standard_normal((4, 1)))
        result = concat([df, df2, df3, df4], axis=1)
        for blocks in result._mgr.blocks:
            arr = blocks.values
            if arr.dtype.kind == "f":
                # this is a view on some array in either df or df4
                assert any(
                    np.shares_memory(arr, block.values)
                    for block in itertools.chain(df._mgr.blocks, df4._mgr.blocks)
                )
            elif arr.dtype.kind in ["i", "u"]:
                assert arr.base is df2._mgr.blocks[0].values.base
            elif arr.dtype == object:
                # this is a view on df3
                assert any(
                    np.shares_memory(arr, block.values) for block in df3._mgr.blocks
                )

    def test_concat_with_group_keys(self):
        # [FUNCTION DOC]
        # Verifies that supplying keys builds a hierarchical index (along rows) or
        # hierarchical columns (along columns) with the keys as outer level, for frames
        # of differing lengths and for the same frame repeated.
        # axis=0
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 4)))
        df2 = DataFrame(np.random.default_rng(2).standard_normal((4, 4)))

        result = concat([df, df2], keys=[0, 1])
        exp_index = MultiIndex.from_arrays(
            [[0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 0, 1, 2, 3]]
        )
        expected = DataFrame(np.r_[df.values, df2.values], index=exp_index)
        tm.assert_frame_equal(result, expected)

        result = concat([df, df], keys=[0, 1])
        exp_index2 = MultiIndex.from_arrays([[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]])
        expected = DataFrame(np.r_[df.values, df.values], index=exp_index2)
        tm.assert_frame_equal(result, expected)

        # axis=1
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 3)))
        df2 = DataFrame(np.random.default_rng(2).standard_normal((4, 4)))

        result = concat([df, df2], keys=[0, 1], axis=1)
        expected = DataFrame(np.c_[df.values, df2.values], columns=exp_index)
        tm.assert_frame_equal(result, expected)

        result = concat([df, df], keys=[0, 1], axis=1)
        expected = DataFrame(np.c_[df.values, df.values], columns=exp_index2)
        tm.assert_frame_equal(result, expected)

    def test_concat_keys_specific_levels(self):
        # [FUNCTION DOC]
        # Verifies that explicit levels and a names argument produce a column
        # multi-level index whose first level equals the passed level (in given order,
        # named by the names argument) and whose inner level is the original columns.
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        pieces = [df.iloc[:, [0, 1]], df.iloc[:, [2]], df.iloc[:, [3]]]
        level = ["three", "two", "one", "zero"]
        result = concat(
            pieces,
            axis=1,
            keys=["one", "two", "three"],
            levels=[level],
            names=["group_key"],
        )

        tm.assert_index_equal(result.columns.levels[0], Index(level, name="group_key"))
        tm.assert_index_equal(result.columns.levels[1], Index([0, 1, 2, 3]))

        assert result.columns.names == ["group_key", None]

    @pytest.mark.parametrize("mapping", ["mapping", "dict"])
    def test_concat_mapping(self, mapping, non_dict_mapping_subclass):
        # [FUNCTION DOC]
        # Parametrized by mapping type (a plain dict or a non-dict mapping subclass from
        # the fixture). Verifies concatenate accepts a mapping: keys become the group
        # keys in mapping order (also along columns), and an explicit key list selects
        # and orders the frames.
        constructor = dict if mapping == "dict" else non_dict_mapping_subclass
        frames = constructor(
            {
                "foo": DataFrame(np.random.default_rng(2).standard_normal((4, 3))),
                "bar": DataFrame(np.random.default_rng(2).standard_normal((4, 3))),
                "baz": DataFrame(np.random.default_rng(2).standard_normal((4, 3))),
                "qux": DataFrame(np.random.default_rng(2).standard_normal((4, 3))),
            }
        )

        sorted_keys = list(frames.keys())

        result = concat(frames)
        expected = concat([frames[k] for k in sorted_keys], keys=sorted_keys)
        tm.assert_frame_equal(result, expected)

        result = concat(frames, axis=1)
        expected = concat([frames[k] for k in sorted_keys], keys=sorted_keys, axis=1)
        tm.assert_frame_equal(result, expected)

        keys = ["baz", "foo", "bar"]
        result = concat(frames, keys=keys)
        expected = concat([frames[k] for k in keys], keys=keys)
        tm.assert_frame_equal(result, expected)

    def test_concat_keys_and_levels(self):
        # [FUNCTION DOC]
        # Verifies concatenation with tuple keys, explicit levels and names builds a
        # three-level index (last level from the original index); without names the level
        # names are unset, and without levels the first level is the sorted unique keys.
        df = DataFrame(np.random.default_rng(2).standard_normal((1, 3)))
        df2 = DataFrame(np.random.default_rng(2).standard_normal((1, 4)))

        levels = [["foo", "baz"], ["one", "two"]]
        names = ["first", "second"]
        result = concat(
            [df, df2, df, df2],
            keys=[("foo", "one"), ("foo", "two"), ("baz", "one"), ("baz", "two")],
            levels=levels,
            names=names,
        )
        expected = concat([df, df2, df, df2])
        exp_index = MultiIndex(
            levels=[*levels, [0]],
            codes=[[0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 0, 0]],
            names=[*names, None],
        )
        expected.index = exp_index

        tm.assert_frame_equal(result, expected)

        # no names
        result = concat(
            [df, df2, df, df2],
            keys=[("foo", "one"), ("foo", "two"), ("baz", "one"), ("baz", "two")],
            levels=levels,
        )
        assert result.index.names == (None,) * 3

        # no levels
        result = concat(
            [df, df2, df, df2],
            keys=[("foo", "one"), ("foo", "two"), ("baz", "one"), ("baz", "two")],
            names=["first", "second"],
        )
        assert result.index.names == ("first", "second", None)
        tm.assert_index_equal(
            result.index.levels[0], Index(["baz", "foo"], name="first")
        )

    def test_concat_keys_levels_no_overlap(self):
        # [FUNCTION DOC]
        # Verifies value errors when keys are not present in the given levels: one message
        # for values missing from the passed level and one saying a key is not in the
        # level (GH 1406).
        # GH #1406
        df = DataFrame(np.random.default_rng(2).standard_normal((1, 3)), index=["a"])
        df2 = DataFrame(np.random.default_rng(2).standard_normal((1, 4)), index=["b"])

        msg = "Values not found in passed level"
        with pytest.raises(ValueError, match=msg):
            concat([df, df], keys=["one", "two"], levels=[["foo", "bar", "baz"]])

        msg = "Key one not in level"
        with pytest.raises(ValueError, match=msg):
            concat([df, df2], keys=["one", "two"], levels=[["foo", "bar", "baz"]])

    def test_crossed_dtypes_weird_corner(self):
        # [FUNCTION DOC]
        # Verifies row concatenation of frames whose int and float column dtypes are
        # crossed equals the numpy concatenation of their values, and that passing keys
        # with names to frames of different widths names the resulting index levels.
        columns = ["A", "B", "C", "D"]
        df1 = DataFrame(
            {
                "A": np.array([1, 2, 3, 4], dtype="f8"),
                "B": np.array([1, 2, 3, 4], dtype="i8"),
                "C": np.array([1, 2, 3, 4], dtype="f8"),
                "D": np.array([1, 2, 3, 4], dtype="i8"),
            },
            columns=columns,
        )

        df2 = DataFrame(
            {
                "A": np.array([1, 2, 3, 4], dtype="i8"),
                "B": np.array([1, 2, 3, 4], dtype="f8"),
                "C": np.array([1, 2, 3, 4], dtype="i8"),
                "D": np.array([1, 2, 3, 4], dtype="f8"),
            },
            columns=columns,
        )

        appended = concat([df1, df2], ignore_index=True)
        expected = DataFrame(
            np.concatenate([df1.values, df2.values], axis=0), columns=columns
        )
        tm.assert_frame_equal(appended, expected)

        df = DataFrame(np.random.default_rng(2).standard_normal((1, 3)), index=["a"])
        df2 = DataFrame(np.random.default_rng(2).standard_normal((1, 4)), index=["b"])
        result = concat([df, df2], keys=["one", "two"], names=["first", "second"])
        assert result.index.names == ("first", "second")

    def test_with_mixed_tuples(self, sort):
        # [FUNCTION DOC]
        # Parametrized by sort. Smoke test: concatenating frames whose column labels
        # contain a tuple and strings must not raise (GH 10697).
        # 10697
        # columns have mixed tuples, so handle properly
        df1 = DataFrame({"A": "foo", ("B", 1): "bar"}, index=range(2))
        df2 = DataFrame({"B": "foo", ("B", 1): "bar"}, index=range(2))

        # it works
        concat([df1, df2], sort=sort)

    def test_concat_mixed_objs_columns(self):
        # [FUNCTION DOC]
        # Verifies column-wise concatenation of a mix of series and frames: default
        # column labels (positional for unnamed series, existing names retained, frame
        # columns kept as is), and ignored index renumbering columns (GH 2385).
        # Test column-wise concat for mixed series/frames (axis=1)
        # G2385

        index = date_range("01-Jan-2013", periods=10, freq="h")
        arr = np.arange(10, dtype="int64")
        s1 = Series(arr, index=index)
        s2 = Series(arr, index=index)
        df = DataFrame(arr.reshape(-1, 1), index=index)

        expected = DataFrame(
            np.repeat(arr, 2).reshape(-1, 2), index=index, columns=[0, 0]
        )
        result = concat([df, df], axis=1)
        tm.assert_frame_equal(result, expected)

        expected = DataFrame(
            np.repeat(arr, 2).reshape(-1, 2), index=index, columns=[0, 1]
        )
        result = concat([s1, s2], axis=1)
        tm.assert_frame_equal(result, expected)

        expected = DataFrame(
            np.repeat(arr, 3).reshape(-1, 3), index=index, columns=[0, 1, 2]
        )
        result = concat([s1, s2, s1], axis=1)
        tm.assert_frame_equal(result, expected)

        expected = DataFrame(
            np.repeat(arr, 5).reshape(-1, 5), index=index, columns=[0, 0, 1, 2, 3]
        )
        result = concat([s1, df, s2, s2, s1], axis=1)
        tm.assert_frame_equal(result, expected)

        # with names
        s1.name = "foo"
        expected = DataFrame(
            np.repeat(arr, 3).reshape(-1, 3), index=index, columns=["foo", 0, 0]
        )
        result = concat([s1, df, s2], axis=1)
        tm.assert_frame_equal(result, expected)

        s2.name = "bar"
        expected = DataFrame(
            np.repeat(arr, 3).reshape(-1, 3), index=index, columns=["foo", 0, "bar"]
        )
        result = concat([s1, df, s2], axis=1)
        tm.assert_frame_equal(result, expected)

        # ignore index
        expected = DataFrame(
            np.repeat(arr, 3).reshape(-1, 3), index=index, columns=[0, 1, 2]
        )
        result = concat([s1, df, s2], axis=1, ignore_index=True)
        tm.assert_frame_equal(result, expected)

    def test_concat_mixed_objs_index(self):
        # [FUNCTION DOC]
        # Verifies row-wise concatenation of series and frame with no names: result is a
        # one-column frame stacking all values with repeated index (GH 2385, GH 15047).
        # Test row-wise concat for mixed series/frames with a common name
        # GH2385, GH15047

        index = date_range("01-Jan-2013", periods=10, freq="h")
        arr = np.arange(10, dtype="int64")
        s1 = Series(arr, index=index)
        s2 = Series(arr, index=index)
        df = DataFrame(arr.reshape(-1, 1), index=index)

        expected = DataFrame(
            np.tile(arr, 3).reshape(-1, 1), index=index.tolist() * 3, columns=[0]
        )
        result = concat([s1, df, s2])
        tm.assert_frame_equal(result, expected)

    def test_concat_mixed_objs_index_names(self):
        # [FUNCTION DOC]
        # Verifies row-wise concatenation of series with different names and a frame:
        # result is a block-diagonal-like frame with one column per name and missing
        # values elsewhere, with repeated index or, with ignored index, a new integer
        # index (GH 60723, GH 56257).
        # Test row-wise concat for mixed series/frames with distinct names
        # GH2385, GH15047
        # GH #60723 & GH #56257 (Updated the test case,
        # as the above GH PR ones were incorrect)

        index = date_range("01-Jan-2013", periods=10, freq="h")
        arr = np.arange(10, dtype="int64")
        s1 = Series(arr, index=index, name="foo")
        s2 = Series(arr, index=index, name="bar")
        df = DataFrame(arr.reshape(-1, 1), index=index)

        expected = DataFrame(
            np.kron(np.where(np.identity(3) == 1, 1, np.nan), arr).T,
            index=index.tolist() * 3,
            columns=["foo", 0, "bar"],
        )
        result = concat([s1, df, s2])
        tm.assert_frame_equal(result, expected)

        expected = DataFrame(
            np.kron(np.where(np.identity(3) == 1, 1, np.nan), arr).T,
            index=np.arange(30, dtype=np.int64),
            columns=["foo", 0, "bar"],
        )
        result = concat([s1, df, s2], ignore_index=True)
        tm.assert_frame_equal(result, expected)

    def test_dtype_coercion(self):
        # [FUNCTION DOC]
        # Verifies that concatenating single-row slices preserves dtypes for tz-aware
        # datetimes with NaT, out-of-bounds-year datetimes and text with missing values
        # (GH 12411, GH 12045, GH 11594).
        # 12411
        df = DataFrame({"date": [pd.Timestamp("20130101").tz_localize("UTC"), pd.NaT]})

        result = concat([df.iloc[[0]], df.iloc[[1]]])
        tm.assert_series_equal(result.dtypes, df.dtypes)

        # 12045
        df = DataFrame({"date": [datetime(2012, 1, 1), datetime(1012, 1, 2)]})
        result = concat([df.iloc[[0]], df.iloc[[1]]])
        tm.assert_series_equal(result.dtypes, df.dtypes)

        # 11594
        df = DataFrame({"text": ["some words"] + [None] * 9})
        result = concat([df.iloc[[0]], df.iloc[[1]]])
        tm.assert_series_equal(result.dtypes, df.dtypes)

    def test_concat_single_with_key(self):
        # [FUNCTION DOC]
        # Verifies that concatenating a single-element list with one key equals the first
        # part of a two-frame concatenation with keys.
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))

        result = concat([df], keys=["foo"])
        expected = concat([df, df], keys=["foo", "bar"])
        tm.assert_frame_equal(result, expected[:10])

    def test_concat_no_items_raises(self):
        # [FUNCTION DOC]
        # Verifies an empty list raises a value error saying there are no objects to
        # concatenate.
        with pytest.raises(ValueError, match="No objects to concatenate"):
            concat([])

    def test_concat_exclude_none(self):
        # [FUNCTION DOC]
        # Verifies None entries are silently dropped and that a list consisting only of
        # None raises a value error saying all objects were None.
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))

        pieces = [df[:5], None, None, df[5:]]
        result = concat(pieces)
        tm.assert_frame_equal(result, df)
        with pytest.raises(ValueError, match="All objects passed were None"):
            concat([None, None])

    def test_concat_keys_with_none(self):
        # [FUNCTION DOC]
        # Verifies that None values in the input (mapping or list with keys) are dropped
        # together with their keys (GH 1649).
        # #1649
        df0 = DataFrame([[10, 20, 30], [10, 20, 30], [10, 20, 30]])

        result = concat({"a": None, "b": df0, "c": df0[:2], "d": df0[:1], "e": df0})
        expected = concat({"b": df0, "c": df0[:2], "d": df0[:1], "e": df0})
        tm.assert_frame_equal(result, expected)

        result = concat(
            [None, df0, df0[:2], df0[:1], df0], keys=["a", "b", "c", "d", "e"]
        )
        expected = concat([df0, df0[:2], df0[:1], df0], keys=["b", "c", "d", "e"])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("klass", [range, RangeIndex])
    @pytest.mark.parametrize("include_none", [True, False])
    def test_concat_preserves_rangeindex(self, klass, include_none):
        # [FUNCTION DOC]
        # Parametrized over key container type (range or range index) and whether None
        # entries are interleaved. Verifies the resulting multi-level index keeps range
        # index levels with the expected start, stop and step.
        df = DataFrame([1, 2])
        df2 = DataFrame([3, 4])
        data = [df, None, df2, None] if include_none else [df, df2]
        keys_length = 4 if include_none else 2
        result = concat(data, keys=klass(keys_length))
        expected = DataFrame(
            [1, 2, 3, 4],
            index=MultiIndex(
                levels=(
                    RangeIndex(start=0, stop=keys_length, step=keys_length / 2),
                    RangeIndex(start=0, stop=2, step=1),
                ),
                codes=(
                    np.array([0, 0, 1, 1], dtype=np.int8),
                    np.array([0, 1, 0, 1], dtype=np.int8),
                ),
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_concat_bug_1719(self):
        # [FUNCTION DOC]
        # Verifies outer join along columns of series of different lengths gives equal
        # length results in either order; the second order emits a Pandas4Warning about
        # default sorting of all-datetime indexes (GH 1719).
        ts1 = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        ts2 = ts1[::2]

        # to join with union
        # these two are of different length!
        left = concat([ts1, ts2], join="outer", axis=1)
        msg = "Sorting by default when concatenating all DatetimeIndex is deprecated"
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            right = concat([ts2, ts1], join="outer", axis=1)

        assert len(left) == len(right)

    def test_concat_bug_2972(self):
        # [FUNCTION DOC]
        # Verifies column-wise concatenation of two series with the same name keeps the
        # duplicate name as column labels (GH 2972).
        ts0 = Series(np.zeros(5))
        ts1 = Series(np.ones(5))
        ts0.name = ts1.name = "same name"
        result = concat([ts0, ts1], axis=1)

        expected = DataFrame({0: ts0, 1: ts1})
        expected.columns = ["same name", "same name"]
        tm.assert_frame_equal(result, expected)

    def test_concat_bug_3602(self):
        # [FUNCTION DOC]
        # Verifies column-wise concatenation of frames with an overlapping column name
        # keeps duplicate columns in order (GH 3602).
        # GH 3602, duplicate columns
        df1 = DataFrame(
            {
                "firmNo": [0, 0, 0, 0],
                "prc": [6, 6, 6, 6],
                "stringvar": ["rrr", "rrr", "rrr", "rrr"],
            }
        )
        df2 = DataFrame(
            {"C": [9, 10, 11, 12], "misc": [1, 2, 3, 4], "prc": [6, 6, 6, 6]}
        )
        expected = DataFrame(
            [
                [0, 6, "rrr", 9, 1, 6],
                [0, 6, "rrr", 10, 2, 6],
                [0, 6, "rrr", 11, 3, 6],
                [0, 6, "rrr", 12, 4, 6],
            ]
        )
        expected.columns = ["firmNo", "prc", "stringvar", "C", "misc", "prc"]

        result = concat([df1, df2], axis=1)
        tm.assert_frame_equal(result, expected)

    def test_concat_iterables(self):
        # [FUNCTION DOC]
        # Verifies that tuples, lists, generators, deque, an object with length and item
        # access, and an iterable subclass are all accepted as the input collection
        # (GH 8645).
        # GH8645 check concat works with tuples, list, generators, and weird
        # stuff like deque and custom iterables
        df1 = DataFrame([1, 2, 3])
        df2 = DataFrame([4, 5, 6])
        expected = DataFrame([1, 2, 3, 4, 5, 6])
        tm.assert_frame_equal(concat((df1, df2), ignore_index=True), expected)
        tm.assert_frame_equal(concat([df1, df2], ignore_index=True), expected)
        tm.assert_frame_equal(
            concat((df for df in (df1, df2)), ignore_index=True), expected
        )
        tm.assert_frame_equal(concat(deque((df1, df2)), ignore_index=True), expected)

        class CustomIterator1:
            # [CLASS DOC]
            # HELPER: local class for test_concat_iterables. A sequence-like object (length
            # two, indexable) yielding the two frames from the enclosing test.
            def __len__(self) -> int:
                # [FUNCTION DOC]
                # HELPER: returns the fixed length two for the sequence-like test helper.
                return 2

            def __getitem__(self, index):
                # [FUNCTION DOC]
                # HELPER: returns the first or second frame for index zero or one; raises an
                # index error for any other key, which ends sequence iteration.
                try:
                    return {0: df1, 1: df2}[index]
                except KeyError as err:
                    raise IndexError from err

        tm.assert_frame_equal(concat(CustomIterator1(), ignore_index=True), expected)

        class CustomIterator2(abc.Iterable):
            # [CLASS DOC]
            # HELPER: local class for test_concat_iterables, subclassing the abstract
            # iterable base; iteration yields the two frames from the enclosing test.
            def __iter__(self) -> Iterator:
                # [FUNCTION DOC]
                # HELPER: generator yielding the first then the second frame.
                yield df1
                yield df2

        tm.assert_frame_equal(concat(CustomIterator2(), ignore_index=True), expected)

    def test_concat_order(self):
        # [FUNCTION DOC]
        # Verifies that with sorting enabled, column labels of mixed types (integer,
        # string, None) across many frames are sorted into integer, string, None order
        # (GH 17344, GH 47331).
        # GH 17344, GH#47331
        dfs = [DataFrame(index=range(3), columns=["a", 1, None])]
        dfs += [DataFrame(index=range(3), columns=[None, 1, "a"]) for _ in range(100)]

        result = concat(dfs, sort=True).columns
        expected = Index([1, "a", None])
        tm.assert_index_equal(result, expected)

    def test_concat_different_extension_dtypes_upcasts(self):
        # [FUNCTION DOC]
        # Verifies concatenating a nullable integer series with a decimal extension series
        # upcasts to object holding integers and decimals.
        a = Series(pd.array([1, 2], dtype="Int64"))
        b = Series(to_decimal([1, 2]))

        result = concat([a, b], ignore_index=True)
        expected = Series([1, 2, Decimal(1), Decimal(2)], dtype=object)
        tm.assert_series_equal(result, expected)

    def test_concat_ordered_dict(self):
        # [FUNCTION DOC]
        # Verifies that a dict argument gives the same result as the list with keys in
        # insertion order (GH 21510).
        # GH 21510
        expected = concat(
            [Series(range(3)), Series(range(4))], keys=["First", "Another"]
        )
        result = concat({"First": Series(range(3)), "Another": Series(range(4))})
        tm.assert_series_equal(result, expected)

    def test_concat_duplicate_indices_raise(self):
        # [FUNCTION DOC]
        # Verifies that column-wise concatenation of frames with duplicate index values
        # raises InvalidIndexError saying reindexing is only valid with uniquely valued
        # indexes (GH 45888).
        # GH 45888: test raise for concat DataFrames with duplicate indices
        # https://github.com/pandas-dev/pandas/issues/36263
        df1 = DataFrame(
            np.random.default_rng(2).standard_normal(5),
            index=[0, 1, 2, 3, 3],
            columns=["a"],
        )
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal(5),
            index=[0, 1, 2, 2, 4],
            columns=["b"],
        )
        msg = "Reindexing only valid with uniquely valued Index objects"
        with pytest.raises(InvalidIndexError, match=msg):
            concat([df1, df2], axis=1)


def test_concat_no_unnecessary_upcast(float_numpy_dtype, frame_or_series):
    # [FUNCTION DOC]
    # Parametrized by float numpy dtype and frame_or_series fixtures. Verifies that
    # concatenating three one-element objects containing a missing value keeps the
    # float dtype (GH 13247).
    # GH 13247
    dims = frame_or_series(dtype=object).ndim
    dt = float_numpy_dtype

    dfs = [
        frame_or_series(np.array([1], dtype=dt, ndmin=dims)),
        frame_or_series(np.array([np.nan], dtype=dt, ndmin=dims)),
        frame_or_series(np.array([5], dtype=dt, ndmin=dims)),
    ]
    x = concat(dfs)
    assert x.values.dtype == dt


def test_concat_will_upcast(frame_or_series, any_signed_int_numpy_dtype):
    # [FUNCTION DOC]
    # Parametrized by frame_or_series and signed integer dtype. Verifies that
    # concatenating integer data with a float-NaN part upcasts the result to float64.
    dt = any_signed_int_numpy_dtype
    dims = frame_or_series().ndim
    dfs = [
        frame_or_series(np.array([1], dtype=dt, ndmin=dims)),
        frame_or_series(np.array([np.nan], ndmin=dims)),
        frame_or_series(np.array([5], dtype=dt, ndmin=dims)),
    ]
    x = concat(dfs)
    assert x.values.dtype == "float64"


def test_concat_empty_and_non_empty_frame_regression():
    # [FUNCTION DOC]
    # Verifies that concatenating a frame with an integer column and a frame with an
    # empty column of the same name gives a float column (GH 18178).
    # GH 18178 regression test
    df1 = DataFrame({"foo": [1]})
    df2 = DataFrame({"foo": []})
    expected = DataFrame({"foo": [1.0]})
    result = concat([df1, df2])
    tm.assert_frame_equal(result, expected)


def test_concat_sparse():
    # [FUNCTION DOC]
    # Verifies column-wise concatenation of sparse integer series keeps the sparse
    # dtype with fill value zero (GH 23557).
    # GH 23557
    a = Series(SparseArray([0, 1, 2]))
    expected = DataFrame(data=[[0, 0], [1, 1], [2, 2]]).astype(
        pd.SparseDtype(np.int64, 0)
    )
    result = concat([a, a], axis=1)
    tm.assert_frame_equal(result, expected)


def test_concat_dense_sparse():
    # [FUNCTION DOC]
    # Verifies that concatenating a sparse float series with a dense float series
    # returns a sparse float series (GH 30668).
    # GH 30668
    dtype = pd.SparseDtype(np.float64, None)
    a = Series(pd.arrays.SparseArray([1, None]), dtype=dtype)
    b = Series([1], dtype=float)
    expected = Series(data=[1, None, 1], index=[0, 1, 0]).astype(dtype)
    result = concat([a, b], axis=0)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("keys", [["e", "f", "f"], ["f", "e", "f"]])
def test_duplicate_keys(keys):
    # [FUNCTION DOC]
    # Parametrized over two key lists with duplicate keys. Verifies column-wise
    # concatenation of a frame and two named series: the result has a two-level
    # column index in which a frame contributes its columns under its key and each
    # series its name under its key (GH 33654).
    # GH 33654
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    s1 = Series([7, 8, 9], name="c")
    s2 = Series([10, 11, 12], name="d")
    result = concat([df, s1, s2], axis=1, keys=keys)
    expected_values = [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]]
    expected_columns = MultiIndex.from_tuples(
        [(keys[0], "a"), (keys[0], "b"), (keys[1], "c"), (keys[2], "d")]
    )
    expected = DataFrame(expected_values, columns=expected_columns)
    tm.assert_frame_equal(result, expected)


def test_duplicate_keys_same_frame():
    # [FUNCTION DOC]
    # Verifies that concatenating the same frame twice column-wise with identical
    # keys gives repeated two-level column labels (GH 43595).
    # GH 43595
    keys = ["e", "e"]
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = concat([df, df], axis=1, keys=keys)
    expected_values = [[1, 4, 1, 4], [2, 5, 2, 5], [3, 6, 3, 6]]
    expected_columns = MultiIndex.from_tuples(
        [(keys[0], "a"), (keys[0], "b"), (keys[1], "a"), (keys[1], "b")]
    )
    expected = DataFrame(expected_values, columns=expected_columns)
    tm.assert_frame_equal(result, expected)


@pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager|Passing a SingleBlockManager:DeprecationWarning"
)
@pytest.mark.parametrize(
    "obj",
    [
        tm.SubclassedDataFrame({"A": np.arange(0, 10)}),
        tm.SubclassedSeries(np.arange(0, 10), name="A"),
    ],
)
def test_concat_preserves_subclass(obj):
    # [FUNCTION DOC]
    # Parametrized over a frame subclass and a series subclass instance (ignoring
    # block manager deprecation warnings). Verifies the result of concatenation is an
    # instance of the subclass (GH 28330).
    # GH28330 -- preserve subclass

    result = concat([obj, obj])
    assert isinstance(result, type(obj))


def test_concat_frame_axis0_extension_dtypes():
    # [FUNCTION DOC]
    # Verifies that concatenating a nullable-integer column with a numpy integer
    # column in either order yields the nullable integer dtype.
    # preserve extension dtype (through common_dtype mechanism)
    df1 = DataFrame({"a": pd.array([1, 2, 3], dtype="Int64")})
    df2 = DataFrame({"a": np.array([4, 5, 6])})

    result = concat([df1, df2], ignore_index=True)
    expected = DataFrame({"a": [1, 2, 3, 4, 5, 6]}, dtype="Int64")
    tm.assert_frame_equal(result, expected)

    result = concat([df2, df1], ignore_index=True)
    expected = DataFrame({"a": [4, 5, 6, 1, 2, 3]}, dtype="Int64")
    tm.assert_frame_equal(result, expected)


def test_concat_preserves_extension_int64_dtype():
    # [FUNCTION DOC]
    # Verifies concatenating two nullable integer frames with different columns
    # keeps the nullable dtype with missing values (GH 24768).
    # GH 24768
    df_a = DataFrame({"a": [-1]}, dtype="Int64")
    df_b = DataFrame({"b": [1]}, dtype="Int64")
    result = concat([df_a, df_b], ignore_index=True)
    expected = DataFrame({"a": [-1, None], "b": [None, 1]}, dtype="Int64")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "dtype1,dtype2,expected_dtype",
    [
        ("bool", "bool", "bool"),
        ("boolean", "bool", "boolean"),
        ("bool", "boolean", "boolean"),
        ("boolean", "boolean", "boolean"),
    ],
)
def test_concat_bool_types(dtype1, dtype2, expected_dtype):
    # [FUNCTION DOC]
    # Parametrized over pairs of bool and nullable boolean dtypes. Verifies the result
    # is bool only for two plain bool inputs and nullable boolean otherwise
    # (GH 42800).
    # GH 42800
    ser1 = Series([True, False], dtype=dtype1)
    ser2 = Series([False, True], dtype=dtype2)
    result = concat([ser1, ser2], ignore_index=True)
    expected = Series([True, False, False, True], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("keys", "integrity"),
    [
        (["red"] * 3, True),
        (["red"] * 3, False),
        (["red", "blue", "red"], False),
        (["red", "blue", "red"], True),
    ],
)
def test_concat_repeated_keys(keys, integrity):
    # [FUNCTION DOC]
    # Parametrized over key lists with repeats and verify-integrity on/off. Verifies
    # the result index is a multi-level index built from keys and series labels
    # without raising (GH 20816).
    # GH: 20816
    series_list = [Series({"a": 1}), Series({"b": 2}), Series({"c": 3})]
    result = concat(series_list, keys=keys, verify_integrity=integrity)
    tuples = list(zip(keys, ["a", "b", "c"]))
    expected = Series([1, 2, 3], index=MultiIndex.from_tuples(tuples))
    tm.assert_series_equal(result, expected)


def test_concat_null_object_with_dti():
    # [FUNCTION DOC]
    # Verifies column-wise concatenation of a frame with an object index containing
    # None and one with a tz-aware datetime index gives an object index and
    # missing-value filled columns (GH 40841).
    # GH#40841
    dti = pd.DatetimeIndex(
        ["2021-04-08 21:21:14+00:00"], dtype="datetime64[ns, UTC]", name="Time (UTC)"
    )
    right = DataFrame(data={"C": [0.5274]}, index=dti)

    idx = Index([None], dtype="object", name="Maybe Time (UTC)")
    left = DataFrame(data={"A": [None], "B": [np.nan]}, index=idx)

    result = concat([left, right], axis="columns")

    exp_index = Index([None, dti[0]], dtype=object)
    expected = DataFrame(
        {
            "A": np.array([None, np.nan], dtype=object),
            "B": [np.nan, np.nan],
            "C": [np.nan, 0.5274],
        },
        index=exp_index,
    )
    tm.assert_frame_equal(result, expected)


def test_concat_multiindex_with_empty_rangeindex():
    # [FUNCTION DOC]
    # Verifies concatenating a frame with multi-level columns and an empty frame whose
    # columns are an empty range index keeps the multi-level columns and adds a
    # missing row (GH 41234).
    # GH#41234
    mi = MultiIndex.from_tuples([("B", 1), ("C", 1)])
    df1 = DataFrame([[1, 2]], columns=mi)
    df2 = DataFrame(index=[1], columns=RangeIndex(0))

    result = concat([df1, df2])
    expected = DataFrame([[1, 2], [np.nan, np.nan]], columns=mi)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data",
    [
        Series(data=[1, 2]),
        DataFrame(
            data={
                "col1": [1, 2],
            }
        ),
        DataFrame(dtype=float),
        Series(dtype=float),
    ],
)
def test_concat_drop_attrs(data):
    # [FUNCTION DOC]
    # Parametrized over series and frame instances (empty and not). Verifies that the
    # attributes dictionary is dropped when inputs have conflicting attribute values
    # (GH 41828).
    # GH#41828
    df1 = data.copy()
    df1.attrs = {1: 1}
    df2 = data.copy()
    df2.attrs = {1: 2}
    df = concat([df1, df2])
    assert len(df.attrs) == 0


@pytest.mark.parametrize(
    "data",
    [
        Series(data=[1, 2]),
        DataFrame(
            data={
                "col1": [1, 2],
            }
        ),
        DataFrame(dtype=float),
        Series(dtype=float),
    ],
)
def test_concat_retain_attrs(data):
    # [FUNCTION DOC]
    # Same parametrization as the drop test. Verifies the attributes dictionary is
    # kept when all inputs carry the same attributes (GH 41828).
    # GH#41828
    df1 = data.copy()
    df1.attrs = {1: 1}
    df2 = data.copy()
    df2.attrs = {1: 1}
    df = concat([df1, df2])
    assert df.attrs[1] == 1


@pytest.mark.parametrize("df_dtype", ["float64", "int64", "datetime64[ns]"])
@pytest.mark.parametrize("empty_dtype", [None, "float64", "object"])
def test_concat_ignore_empty_object_float(empty_dtype, df_dtype):
    # [FUNCTION DOC]
    # Parametrized over the dtype of a non-empty frame (float, integer, datetime) and
    # of an empty frame (unset, float, object). Verifies the result dtype of
    # concatenating the empty frame with the other: the data frame's dtype is kept,
    # except it becomes object for datetime or float-with-nonfloat-empty, and for
    # integer data becomes float if the empty frame is float, else object
    # (GH 40893, GH 45637). The test body contains a TODO noting integer behavior is
    # undecided.
    # https://github.com/pandas-dev/pandas/issues/45637
    df = DataFrame({"foo": [1, 2], "bar": [1, 2]}, dtype=df_dtype)
    empty = DataFrame(columns=["foo", "bar"], dtype=empty_dtype)

    needs_update = False
    if df_dtype == "datetime64[ns]" or (
        df_dtype == "float64" and empty_dtype != "float64"
    ):
        needs_update = True

    result = concat([empty, df])
    expected = df
    if df_dtype == "int64":
        # TODO what exact behaviour do we want for integer eventually?
        if empty_dtype == "float64":
            expected = df.astype("float64")
        else:
            expected = df.astype("object")

    if needs_update:
        # GH#40893 changed the expected here to retain dependence on empty
        expected = expected.astype(object)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("df_dtype", ["float64", "int64", "datetime64[ns]"])
@pytest.mark.parametrize("empty_dtype", [None, "float64", "object"])
def test_concat_ignore_all_na_object_float(empty_dtype, df_dtype):
    # [FUNCTION DOC]
    # Same parametrization as above but the second frame holds all-missing values.
    # Verifies the result dtype: the data dtype (integer data becomes float, or object
    # when the empty-side dtype is object) is kept unless dtypes differ or datetimes
    # are involved, in which case the result is object with the first row missing.
    df = DataFrame({"foo": [1, 2], "bar": [1, 2]}, dtype=df_dtype)
    empty = DataFrame({"foo": [np.nan], "bar": [np.nan]}, dtype=empty_dtype)

    if df_dtype == "int64":
        # TODO what exact behaviour do we want for integer eventually?
        if empty_dtype == "object":
            df_dtype = "object"
        else:
            df_dtype = "float64"

    needs_update = False
    if empty_dtype != df_dtype and empty_dtype is not None:
        needs_update = True
    elif df_dtype == "datetime64[ns]":
        needs_update = True

    result = concat([empty, df], ignore_index=True)

    expected = DataFrame({"foo": [np.nan, 1, 2], "bar": [np.nan, 1, 2]}, dtype=df_dtype)
    if needs_update:
        # GH#40893 changed the expected here to retain dependence on empty
        expected = expected.astype(object)
        expected.iloc[0] = np.nan
    tm.assert_frame_equal(result, expected)


def test_concat_ignore_empty_from_reindex():
    # [FUNCTION DOC]
    # Verifies that concatenating a frame having a datetime column with a frame
    # reindexed to add that column (all missing) gives an object column holding the
    # timestamp and a missing value, with the integer column preserved.
    # https://github.com/pandas-dev/pandas/pull/43507#issuecomment-920375856
    df1 = DataFrame({"a": [1], "b": [pd.Timestamp("2012-01-01")]})
    df2 = DataFrame({"a": [2]})

    aligned = df2.reindex(columns=df1.columns)

    result = concat([df1, aligned], ignore_index=True)

    expected = DataFrame(
        {
            "a": [1, 2],
            "b": pd.array([pd.Timestamp("2012-01-01"), np.nan], dtype=object),
        },
        dtype=object,
    )
    expected["a"] = expected["a"].astype("int64")
    tm.assert_frame_equal(result, expected)


def test_concat_mismatched_keys_length():
    # [FUNCTION DOC]
    # Verifies a value error about the length of the keys is raised when the keys list
    # is shorter than the objects, for both axes and for generator inputs.
    # GH#43485
    ser = Series(range(5))
    sers = [ser + n for n in range(4)]
    keys = ["A", "B", "C"]

    msg = r"The length of the keys"
    with pytest.raises(ValueError, match=msg):
        concat(sers, keys=keys, axis=1)
    with pytest.raises(ValueError, match=msg):
        concat(sers, keys=keys, axis=0)
    with pytest.raises(ValueError, match=msg):
        concat((x for x in sers), keys=(y for y in keys), axis=1)
    with pytest.raises(ValueError, match=msg):
        concat((x for x in sers), keys=(y for y in keys), axis=0)


def test_concat_multiindex_with_category():
    # [FUNCTION DOC]
    # Verifies concatenating frames indexed by two categorical levels keeps
    # categorical level dtypes in the result index.
    df1 = DataFrame(
        {
            "c1": Series(list("abc"), dtype="category"),
            "c2": Series(list("eee"), dtype="category"),
            "i2": Series([1, 2, 3]),
        }
    )
    df1 = df1.set_index(["c1", "c2"])
    df2 = DataFrame(
        {
            "c1": Series(list("abc"), dtype="category"),
            "c2": Series(list("eee"), dtype="category"),
            "i2": Series([4, 5, 6]),
        }
    )
    df2 = df2.set_index(["c1", "c2"])
    result = concat([df1, df2])
    expected = DataFrame(
        {
            "c1": Series(list("abcabc"), dtype="category"),
            "c2": Series(list("eeeeee"), dtype="category"),
            "i2": Series([1, 2, 3, 4, 5, 6]),
        }
    )
    expected = expected.set_index(["c1", "c2"])
    tm.assert_frame_equal(result, expected)


def test_concat_ea_upcast():
    # [FUNCTION DOC]
    # Verifies concatenating a string-dtype frame and a nullable integer frame gives an
    # object frame holding both values (GH 54848).
    # GH#54848
    df1 = DataFrame(["a"], dtype="string")
    df2 = DataFrame([1], dtype="Int64")
    result = concat([df1, df2])
    expected = DataFrame(["a", 1], index=[0, 0])
    tm.assert_frame_equal(result, expected)


def test_concat_none_with_timezone_timestamp():
    # [FUNCTION DOC]
    # Verifies that concatenating a frame holding None with one holding a tz-aware
    # timestamp gives an object column (GH 52093).
    # GH#52093
    df1 = DataFrame([{"A": None}])
    df2 = DataFrame([{"A": pd.Timestamp("1990-12-20 00:00:00+00:00")}])
    result = concat([df1, df2], ignore_index=True)
    expected = DataFrame(
        {"A": [None, pd.Timestamp("1990-12-20 00:00:00+00:00")]}, dtype=object
    )
    tm.assert_frame_equal(result, expected)


def test_concat_with_series_and_frame_returns_rangeindex_columns():
    # [FUNCTION DOC]
    # Verifies row-wise concatenation of an unnamed series and a one-column frame gives
    # a frame whose columns are a range index (checked with the column type check).
    ser = Series([0])
    df = DataFrame([1, 2])
    result = concat([ser, df])
    expected = DataFrame([0, 1, 2], index=[0, 0, 1])
    tm.assert_frame_equal(result, expected, check_column_type=True)


def test_concat_with_moot_ignore_index_and_keys():
    # [FUNCTION DOC]
    # Verifies that giving both ignore_index true and keys raises a value error saying
    # they cannot be combined.
    df1 = DataFrame([[0]])
    df2 = DataFrame([[42]])

    ignore_index = True
    keys = ["df1", "df2"]
    msg = f"Cannot set {ignore_index=} and specify keys. Either should be used."
    with pytest.raises(ValueError, match=msg):
        concat([df1, df2], keys=keys, ignore_index=ignore_index)


@pytest.mark.parametrize(
    "inputs, ignore_index, axis, expected",
    [
        # Concatenating DataFrame and named Series without ignore_index
        (
            [DataFrame({"a": [0, 1], "b": [2, 3]}), Series([4, 5], name="c")],
            False,
            0,
            DataFrame(
                {
                    "a": [0, 1, None, None],
                    "b": [2, 3, None, None],
                    "c": [None, None, 4, 5],
                },
                index=[0, 1, 0, 1],
            ),
        ),
        # Concatenating DataFrame and named Series with ignore_index
        (
            [DataFrame({"a": [0, 1], "b": [2, 3]}), Series([4, 5], name="c")],
            True,
            0,
            DataFrame(
                {
                    "a": [0, 1, None, None],
                    "b": [2, 3, None, None],
                    "c": [None, None, 4, 5],
                },
                index=[0, 1, 2, 3],
            ),
        ),
        # Concatenating DataFrame and unnamed Series along columns
        (
            [DataFrame({"a": [0, 1], "b": [2, 3]}), Series([4, 5]), Series([4, 5])],
            False,
            1,
            DataFrame({"a": [0, 1], "b": [2, 3], 0: [4, 5], 1: [4, 5]}, index=[0, 1]),
        ),
        # Concatenating DataFrame and unnamed Series along columns with ignore_index
        (
            [DataFrame({"a": [0, 1], "b": [2, 3]}), Series([4, 5]), Series([4, 5])],
            True,
            1,
            DataFrame({0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [4, 5]}, index=[0, 1]),
        ),
    ],
)
def test_concat_of_series_and_frame(inputs, ignore_index, axis, expected):
    # [FUNCTION DOC]
    # Parametrized over (inputs, ignore_index, axis, expected). Verifies concatenation of
    # a frame with named or unnamed series along either axis: a named series becomes a
    # new column with missing fill when stacking rows; unnamed series along columns get
    # positional labels, or all columns are renumbered when ignoring index
    # (GH 60723, GH 56257).
    # GH #60723 and #56257
    result = concat(inputs, ignore_index=ignore_index, axis=axis)
    tm.assert_frame_equal(result, expected)
