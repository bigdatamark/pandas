# [MODULE DOC]
# Tests for Series.rename: callables, dicts, Series, scalar or hashable labels
# (sets the name), index level renames, inplace, axis, errors, and shallow copy
# semantics. No import-time side effects.

from datetime import datetime
import re

import numpy as np
import pytest

from pandas import (
    Index,
    MultiIndex,
    Series,
    array,
)
import pandas._testing as tm


class TestRename:
    # [CLASS DOC]
    # Groups the tests of Series.rename. Stateless.
    def test_rename(self, datetime_series):
        # [FUNCTION DOC]
        # Renaming with a function (date formatting) or an equivalent dict gives the
        # same Series.
        ts = datetime_series
        renamer = lambda x: x.strftime("%Y%m%d")
        renamed = ts.rename(renamer)
        assert renamed.index[0] == renamer(ts.index[0])

        # dict
        rename_dict = dict(zip(ts.index, renamed.index, strict=True))
        renamed2 = ts.rename(rename_dict)
        tm.assert_series_equal(renamed, renamed2)

    def test_rename_partial_dict(self):
        # [FUNCTION DOC]
        # A dict with only some labels renames those and keeps the others.
        # partial dict
        ser = Series(np.arange(4), index=["a", "b", "c", "d"], dtype="int64")
        renamed = ser.rename({"b": "foo", "d": "bar"})
        tm.assert_index_equal(renamed.index, Index(["a", "foo", "c", "bar"]))

    def test_rename_retain_index_name(self):
        # [FUNCTION DOC]
        # Renaming with an empty dict preserves the index name.
        # index with name
        renamer = Series(
            np.arange(4), index=Index(["a", "b", "c", "d"], name="name"), dtype="int64"
        )
        renamed = renamer.rename({})
        assert renamed.index.name == renamer.index.name

    def test_rename_by_series(self):
        # [FUNCTION DOC]
        # A Series used as mapper renames labels using its index to values; unmatched
        # labels are kept.
        ser = Series(range(5), name="foo")
        renamer = Series({1: 10, 2: 20})
        result = ser.rename(renamer)
        expected = Series(range(5), index=[0, 10, 20, 3, 4], name="foo")
        tm.assert_series_equal(result, expected)

    def test_rename_set_name(self, using_infer_string):
        # [FUNCTION DOC]
        # A scalar or hashable argument (string, int, float, datetime, tuple) sets the Series
        # name, leaves the index untouched (also under the inferred-string option) and
        # does not change the original.
        ser = Series(range(4), index=list("abcd"))
        for name in ["foo", 123, 123.0, datetime(2001, 11, 11), ("foo",)]:
            result = ser.rename(name)
            assert result.name == name
            if using_infer_string:
                tm.assert_extension_array_equal(result.index.values, ser.index.values)
            else:
                tm.assert_numpy_array_equal(result.index.values, ser.index.values)
            assert ser.name is None

    def test_rename_set_name_inplace(self, using_infer_string):
        # [FUNCTION DOC]
        # Same as above in place: the name is set on the Series and the index values are
        # untouched.
        ser = Series(range(3), index=list("abc"))
        for name in ["foo", 123, 123.0, datetime(2001, 11, 11), ("foo",)]:
            ser.rename(name, inplace=True)
            assert ser.name == name
            exp = np.array(["a", "b", "c"], dtype=np.object_)
            if using_infer_string:
                exp = array(exp, dtype="str")
                tm.assert_extension_array_equal(ser.index.values, exp)
            else:
                tm.assert_numpy_array_equal(ser.index.values, exp)

    def test_rename_axis_supported(self):
        # [FUNCTION DOC]
        # GH-18589: axis 0 or "index" is accepted for compatibility; an invalid axis
        # raises a value error.
        # Supporting axis for compatibility, detailed in GH-18589
        ser = Series(range(5))
        ser.rename({}, axis=0)
        ser.rename({}, axis="index")

        with pytest.raises(ValueError, match="No axis named 5"):
            ser.rename({}, axis=5)

    def test_rename_inplace(self, datetime_series):
        # [FUNCTION DOC]
        # Renaming in place with a function changes the index labels of the Series.
        renamer = lambda x: x.strftime("%Y%m%d")
        expected = renamer(datetime_series.index[0])

        datetime_series.rename(renamer, inplace=True)
        assert datetime_series.index[0] == expected

    def test_rename_with_custom_indexer(self):
        # [FUNCTION DOC]
        # GH 27814: an arbitrary non-callable, non-mapping hashable object passed to
        # rename is used as the Series name.
        # GH 27814
        class MyIndexer:
            # [CLASS DOC]
            # HELPER: empty class whose instance serves as a hashable but non-mapper name.
            pass

        ix = MyIndexer()
        ser = Series([1, 2, 3]).rename(ix)
        assert ser.name is ix

    def test_rename_with_custom_indexer_inplace(self):
        # [FUNCTION DOC]
        # GH 27814: the same in place; the object becomes the Series name.
        # GH 27814
        class MyIndexer:
            # [CLASS DOC]
            # HELPER: empty class whose instance serves as a hashable but non-mapper name.
            pass

        ix = MyIndexer()
        ser = Series([1, 2, 3])
        ser.rename(ix, inplace=True)
        assert ser.name is ix

    def test_rename_callable(self):
        # [FUNCTION DOC]
        # GH 17407: renaming with the string conversion type equals renaming with a
        # lambda that does the same, and the name is preserved.
        # GH 17407
        ser = Series(range(1, 6), index=Index(range(2, 7), name="IntIndex"))
        result = ser.rename(str)
        expected = ser.rename(lambda i: str(i))
        tm.assert_series_equal(result, expected)

        assert result.name == expected.name

    def test_rename_none(self):
        # [FUNCTION DOC]
        # GH 40977: renaming with None removes the name.
        # GH 40977
        ser = Series([1, 2], name="foo")
        result = ser.rename(None)
        expected = Series([1, 2])
        tm.assert_series_equal(result, expected)

    def test_rename_series_with_multiindex(self):
        # [FUNCTION DOC]
        # Issue #43659: renaming values within one named level of a MultiIndex with the
        # errors option set to raise changes only that level.
        # issue #43659
        arrays = [
            ["bar", "baz", "baz", "foo", "qux"],
            ["one", "one", "two", "two", "one"],
        ]

        index = MultiIndex.from_arrays(arrays, names=["first", "second"])
        ser = Series(np.ones(5), index=index)
        result = ser.rename(index={"one": "yes"}, level="second", errors="raise")

        arrays_expected = [
            ["bar", "baz", "baz", "foo", "qux"],
            ["yes", "yes", "two", "two", "yes"],
        ]

        index_expected = MultiIndex.from_arrays(
            arrays_expected, names=["first", "second"]
        )
        series_expected = Series(np.ones(5), index=index_expected)

        tm.assert_series_equal(result, series_expected)

    def test_rename_series_with_multiindex_keeps_ea_dtypes(self):
        # [FUNCTION DOC]
        # GH21055: renaming a level of a MultiIndex built from extension dtypes
        # (categorical and Int64) keeps those dtypes.
        # GH21055
        arrays = [
            Index([1, 2, 3], dtype="Int64").astype("category"),
            Index([1, 2, 3], dtype="Int64"),
        ]
        mi = MultiIndex.from_arrays(arrays, names=["A", "B"])
        ser = Series(1, index=mi)
        result = ser.rename({1: 4}, level=1)

        arrays_expected = [
            Index([1, 2, 3], dtype="Int64").astype("category"),
            Index([4, 2, 3], dtype="Int64"),
        ]
        mi_expected = MultiIndex.from_arrays(arrays_expected, names=["A", "B"])
        expected = Series(1, index=mi_expected)

        tm.assert_series_equal(result, expected)

    def test_rename_error_arg(self):
        # [FUNCTION DOC]
        # GH 46889: rename with a key not found and errors set to raise raises a key
        # error saying the key is not found in the axis.
        # GH 46889
        ser = Series(["foo", "bar"])
        match = re.escape("[2] not found in axis")
        with pytest.raises(KeyError, match=match):
            ser.rename({2: 9}, errors="raise")

    def test_rename_copy_false(self):
        # [FUNCTION DOC]
        # GH 46889: the renamed Series shares data with the source, but modifying the
        # source does not change the renamed copy (copy-on-write).
        # KNOWN ISSUE: the test name mentions copy false but no copy argument is
        # passed; it tests the default shallow-copy behavior.
        # GH 46889
        ser = Series(["foo", "bar"])
        ser_orig = ser.copy()
        shallow_copy = ser.rename({1: 9})
        ser[0] = "foobar"
        assert ser_orig[0] == shallow_copy[0]
        assert ser_orig[1] == shallow_copy[9]
