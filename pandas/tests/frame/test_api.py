# [MODULE DOC]
# Miscellaneous DataFrame API tests: axis helper methods, attribute/tab completion, hashing, emptiness, in-place
# return values, attrs propagation, flags, and removed expansion constructor. One test class, all tests use shared fixtures such as the float frame.

from copy import deepcopy
import inspect
import pydoc

import numpy as np
import pytest

from pandas._config import using_string_dtype
from pandas._config.config import option_context

from pandas.compat import HAS_PYARROW

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    date_range,
    timedelta_range,
)
import pandas._testing as tm


class TestDataFrameMisc:
    # [CLASS DOC]
    # Test container for assorted DataFrame API behavior.
    def test_getitem_pop_assign_name(self, float_frame):
        # [FUNCTION DOC]
        # A column obtained by item access, by popping, by label-based selection, or by slicing that series keeps the column label as its name.
        s = float_frame["A"]
        assert s.name == "A"

        s = float_frame.pop("A")
        assert s.name == "A"

        s = float_frame.loc[:, "B"]
        assert s.name == "B"

        s2 = s.loc[:]
        assert s2.name == "B"

    def test_get_axis(self, float_frame):
        # [FUNCTION DOC]
        # The private axis helpers: the axis number for 0, 1, and the names index, rows and columns; the axis name for numbers and aliases (rows maps to index); the axis object (index or columns). Unknown axis values (2, None, an unknown name) raise a value error saying no axis with that name exists.
        f = float_frame
        assert f._get_axis_number(0) == 0
        assert f._get_axis_number(1) == 1
        assert f._get_axis_number("index") == 0
        assert f._get_axis_number("rows") == 0
        assert f._get_axis_number("columns") == 1

        assert f._get_axis_name(0) == "index"
        assert f._get_axis_name(1) == "columns"
        assert f._get_axis_name("index") == "index"
        assert f._get_axis_name("rows") == "index"
        assert f._get_axis_name("columns") == "columns"

        assert f._get_axis(0) is f.index
        assert f._get_axis(1) is f.columns

        with pytest.raises(ValueError, match="No axis named"):
            f._get_axis_number(2)

        with pytest.raises(ValueError, match="No axis.*foo"):
            f._get_axis_name("foo")

        with pytest.raises(ValueError, match="No axis.*None"):
            f._get_axis_name(None)

        with pytest.raises(ValueError, match="No axis named"):
            f._get_axis_number(None)

    def test_column_contains_raises(self, float_frame):
        # [FUNCTION DOC]
        # Testing membership of the columns index object in a frame raises a type error that Index is unhashable.
        with pytest.raises(TypeError, match="unhashable type: 'Index'"):
            float_frame.columns in float_frame

    def test_tab_completion(self):
        # [FUNCTION DOC]
        # Columns that are valid identifiers appear in the directory listing, and the item access returns a series. For multi-level columns only the first-level labels appear, and item access returns a frame.
        # DataFrame whose columns are identifiers shall have them in __dir__.
        df = DataFrame([list("abcd"), list("efgh")], columns=list("ABCD"))
        for key in list("ABCD"):
            assert key in dir(df)
        assert isinstance(df.__getitem__("A"), Series)

        # DataFrame whose first-level columns are identifiers shall have
        # them in __dir__.
        df = DataFrame(
            [list("abcd"), list("efgh")],
            columns=pd.MultiIndex.from_tuples(list(zip("ABCD", "EFGH"))),
        )
        for key in list("ABCD"):
            assert key in dir(df)
        for key in list("EFGH"):
            assert key not in dir(df)
        assert isinstance(df.__getitem__("A"), DataFrame)

    def test_display_max_dir_items(self):
        # [FUNCTION DOC]
        # The display.max_dir_items option bounds how many column names appear in the directory listing: default 100 (first 100 shown), 300 when set (via the option context), and all when set to none (420 columns).
        # display.max_dir_items increases the number of columns that are in __dir__.
        columns = ["a" + str(i) for i in range(420)]
        values = [range(420), range(420)]
        df = DataFrame(values, columns=columns)

        # The default value for display.max_dir_items is 100
        assert "a99" in dir(df)
        assert "a100" not in dir(df)

        with option_context("display.max_dir_items", 300):
            df = DataFrame(values, columns=columns)
            assert "a299" in dir(df)
            assert "a300" not in dir(df)

        with option_context("display.max_dir_items", None):
            df = DataFrame(values, columns=columns)
            assert "a419" in dir(df)

    def test_not_hashable(self):
        # [FUNCTION DOC]
        # Hashing a populated or empty frame raises a type error that DataFrame is unhashable.
        empty_frame = DataFrame()

        df = DataFrame([1])
        msg = "unhashable type: 'DataFrame'"
        with pytest.raises(TypeError, match=msg):
            hash(df)
        with pytest.raises(TypeError, match=msg):
            hash(empty_frame)

    @pytest.mark.xfail(
        using_string_dtype() and HAS_PYARROW, reason="surrogates not allowed"
    )
    def test_column_name_contains_unicode_surrogate(self):
        # [FUNCTION DOC]
        # Regression test (GH 25509). A column label that is a lone surrogate character does not crash the directory listing or label access. Expected failure when string dtype inference is on and pyarrow is installed (surrogates not allowed).
        # GH 25509
        colname = "\ud83d"
        df = DataFrame({colname: []})
        # this should not crash
        assert colname not in dir(df)
        assert df.columns[0] == colname

    def test_new_empty_index(self):
        # [FUNCTION DOC]
        # Setting a name on one empty frame's index does not affect the name of another (indexes are not shared).
        df1 = DataFrame(np.random.default_rng(2).standard_normal((0, 3)))
        df2 = DataFrame(np.random.default_rng(2).standard_normal((0, 3)))
        df1.index.name = "foo"
        assert df2.index.name is None

    def test_get_agg_axis(self, float_frame):
        # [FUNCTION DOC]
        # The private aggregation-axis helper returns the columns for axis 0 and the index for axis 1 (the same objects); any other axis raises a value error that the axis must be 0 or 1.
        cols = float_frame._get_agg_axis(0)
        assert cols is float_frame.columns

        idx = float_frame._get_agg_axis(1)
        assert idx is float_frame.index

        msg = r"Axis must be 0 or 1 \(got 2\)"
        with pytest.raises(ValueError, match=msg):
            float_frame._get_agg_axis(2)

    def test_empty(self, float_frame, float_string_frame):
        # [FUNCTION DOC]
        # An empty frame is empty; frames with data are not; a frame whose only numeric column was deleted but still has a string column is not empty.
        empty_frame = DataFrame()
        assert empty_frame.empty

        assert not float_frame.empty
        assert not float_string_frame.empty

        # corner case
        df = DataFrame({"A": [1.0, 2.0, 3.0], "B": ["a", "b", "c"]}, index=np.arange(3))
        del df["A"]
        assert not df.empty

    def test_len(self, float_frame):
        # [FUNCTION DOC]
        # The length equals the length of the index; selecting two columns by values and by reindexing gives equal arrays (a single-block corner case).
        assert len(float_frame) == len(float_frame.index)

        # single block corner case
        arr = float_frame[["A", "B"]].values
        expected = float_frame.reindex(columns=["A", "B"]).values
        tm.assert_almost_equal(arr, expected)

    def test_axis_aliases(self, float_frame):
        # [FUNCTION DOC]
        # Summing with the axis names index and columns gives the same series as with axes 0 and 1.
        f = float_frame

        # reg name
        expected = f.sum(axis=0)
        result = f.sum(axis="index")
        tm.assert_series_equal(result, expected)

        expected = f.sum(axis=1)
        result = f.sum(axis="columns")
        tm.assert_series_equal(result, expected)

    def test_class_axis(self):
        # [FUNCTION DOC]
        # Regression test (GH 18147). The class-level index and columns attributes have non-empty documentation and accessing it does not raise.
        # GH 18147
        # no exception and no empty docstring
        assert pydoc.getdoc(DataFrame.index)
        assert pydoc.getdoc(DataFrame.columns)

    def test_series_put_names(self, float_string_frame):
        # [FUNCTION DOC]
        # Every series in the private per-column series cache of the mixed frame has the column label as its name.
        series = float_string_frame._series
        for k, v in series.items():
            assert v.name == k

    def test_empty_nonzero(self):
        # [FUNCTION DOC]
        # A frame with data, a frame with an index and column but no data (all missing) are not empty; dropping missing rows makes it empty, as does its transpose.
        df = DataFrame([1, 2, 3])
        assert not df.empty
        df = DataFrame(index=[1], columns=[1])
        assert not df.empty
        df = DataFrame(index=["a", "b"], columns=["c", "d"]).dropna()
        assert df.empty
        assert df.T.empty

    @pytest.mark.parametrize(
        "df",
        [
            DataFrame(),
            DataFrame(index=[1]),
            DataFrame(columns=[1]),
            DataFrame({1: []}),
        ],
    )
    def test_empty_like(self, df):
        # [FUNCTION DOC]
        # Parametrized over four frames lacking rows, columns or both; each and its transpose are empty.
        assert df.empty
        assert df.T.empty

    def test_with_datetimelikes(self):
        # [FUNCTION DOC]
        # A frame with datetime and timedelta columns transposes to all-object dtype columns (ten of them).
        df = DataFrame(
            {
                "A": date_range("20130101", periods=10),
                "B": timedelta_range("1 day", periods=10),
            }
        )
        t = df.T

        result = t.dtypes.value_counts()
        expected = Series({np.dtype("object"): 10}, name="count")
        tm.assert_series_equal(result, expected)

    def test_deepcopy(self, float_frame):
        # [FUNCTION DOC]
        # A deep copy is independent: setting a value in the copy leaves the original not equal to the copy.
        cp = deepcopy(float_frame)
        cp.loc[0, "A"] = 10
        assert not float_frame.equals(cp)

    def test_inplace_return_self(self):
        # [FUNCTION DOC]
        # Regression test (GH 1893). Checks which in-place operations return nothing (set_index, reset_index, drop_duplicates, sort_values, sort_index, rename, and series reset_index with drop) and which return the object itself (fill missing values and replace, for frames and series).
        # GH 1893

        data = DataFrame(
            {"a": ["foo", "bar", "baz", "qux"], "b": [0, 0, 1, 1], "c": [1, 2, 3, 4]}
        )

        def _check_none(base, f):
            # [FUNCTION DOC]
            # HELPER: applies the given function to the base object and asserts the return value is none.
            result = f(base)
            assert result is None

        def _check_return(base, f):
            # [FUNCTION DOC]
            # HELPER: applies the given function to the base object and asserts the return value is the very same object.
            result = f(base)
            assert result is base

        # -----DataFrame-----

        # set_index
        f = lambda x: x.set_index("a", inplace=True)
        _check_none(data.copy(), f)

        # reset_index
        f = lambda x: x.reset_index(inplace=True)
        _check_none(data.set_index("a"), f)

        # drop_duplicates
        f = lambda x: x.drop_duplicates(inplace=True)
        _check_none(data.copy(), f)

        # sort
        f = lambda x: x.sort_values("b", inplace=True)
        _check_none(data.copy(), f)

        # sort_index
        f = lambda x: x.sort_index(inplace=True)
        _check_none(data.copy(), f)

        # fillna
        f = lambda x: x.fillna(0, inplace=True)
        _check_return(data.copy(), f)

        # replace
        f = lambda x: x.replace(1, 0, inplace=True)
        _check_return(data.copy(), f)

        # rename
        f = lambda x: x.rename({1: "foo"}, inplace=True)
        _check_none(data.copy(), f)

        # -----Series-----
        d = data.copy()["c"]

        # reset_index
        f = lambda x: x.reset_index(inplace=True, drop=True)
        _check_none(data.set_index("a")["c"], f)

        # fillna
        f = lambda x: x.fillna(0, inplace=True)
        _check_return(d.copy(), f)

        # replace
        f = lambda x: x.replace(1, 0, inplace=True)
        _check_return(d.copy(), f)

        # rename
        f = lambda x: x.rename({1: "foo"}, inplace=True)
        _check_none(d.copy(), f)

    def test_tab_complete_warning(self, ip, frame_or_series):
        # [FUNCTION DOC]
        # Regression test (GH 16409, GH 31324). Skipped without a sufficiently new IPython. Uses the interactive shell fixture to create an empty frame or series, then asks for completions in the provisional completer; no warning may be emitted (extra warnings tolerated).
        # GH 16409
        pytest.importorskip("IPython", minversion="6.0.0")
        from IPython.core.completer import provisionalcompleter

        if frame_or_series is DataFrame:
            code = "from pandas import DataFrame; obj = DataFrame()"
        else:
            code = "from pandas import Series; obj = Series(dtype=object)"

        ip.run_cell(code)
        # GH 31324 newer jedi version raises Deprecation warning;
        #  appears resolved 2021-02-02
        with tm.assert_produces_warning(None, raise_on_extra_warnings=False):
            with provisionalcompleter("ignore"):
                list(ip.Completer.completions("obj.", 1))

    def test_attrs(self):
        # [FUNCTION DOC]
        # A new frame has empty attrs; after setting an entry, renaming columns keeps the attrs.
        df = DataFrame({"A": [2, 3]})
        assert df.attrs == {}
        df.attrs["version"] = 1

        result = df.rename(columns=str)
        assert result.attrs == {"version": 1}

    def test_attrs_is_deepcopy(self):
        # [FUNCTION DOC]
        # Attrs propagated to a renamed copy are equal to but are not the same objects as the originals (deep copy).
        df = DataFrame({"A": [2, 3]})
        assert df.attrs == {}
        df.attrs["tags"] = {"spam", "ham"}

        result = df.rename(columns=str)
        assert result.attrs == df.attrs
        assert result.attrs["tags"] is not df.attrs["tags"]

    def test_attrs_concat(self):
        # [FUNCTION DOC]
        # Concatenation keeps the attrs when all inputs have equal attrs and drops them (empty) when they differ.
        # concat propagates attrs if all input attrs are equal
        df1 = DataFrame({"A": [2, 3]})
        df1.attrs = {"a": 1, "b": 2}
        df2 = DataFrame({"A": [4, 5]})
        df2.attrs = df1.attrs.copy()
        df3 = DataFrame({"A": [6, 7]})
        df3.attrs = df1.attrs.copy()
        assert pd.concat([df1, df2, df3]).attrs == df1.attrs
        # concat does not propagate attrs if input attrs are different
        df2.attrs = {"c": 3}
        assert pd.concat([df1, df2, df3]).attrs == {}

    def test_attrs_merge(self):
        # [FUNCTION DOC]
        # Merging keeps the attrs when both inputs have equal attrs and drops them when they differ.
        # merge propagates attrs if all input attrs are equal
        df1 = DataFrame({"key": ["a", "b"], "val1": [1, 2]})
        df1.attrs = {"a": 1, "b": 2}
        df2 = DataFrame({"key": ["a", "b"], "val2": [3, 4]})
        df2.attrs = df1.attrs.copy()
        assert pd.merge(df1, df2).attrs == df1.attrs
        # merge does not propagate attrs if input attrs are different
        df2.attrs = {"c": 3}
        assert pd.merge(df1, df2).attrs == {}

    @pytest.mark.parametrize("allows_duplicate_labels", [True, False, None])
    def test_set_flags(
        self,
        allows_duplicate_labels,
        frame_or_series,
    ):
        # [FUNCTION DOC]
        # Parametrized over true, false and none for allows-duplicate-labels and frame or series. Setting the flag returns a new object with the flag set (unchanged when none is passed); the original keeps its default true; data memory is shared; writing to the result leaves the original unchanged (copy-on-write).
        obj = DataFrame({"A": [1, 2]})
        key = (0, 0)
        if frame_or_series is Series:
            obj = obj["A"]
            key = 0

        result = obj.set_flags(allows_duplicate_labels=allows_duplicate_labels)

        if allows_duplicate_labels is None:
            # We don't update when it's not provided
            assert result.flags.allows_duplicate_labels is True
        else:
            assert result.flags.allows_duplicate_labels is allows_duplicate_labels

        # We made a copy
        assert obj is not result

        # We didn't mutate obj
        assert obj.flags.allows_duplicate_labels is True

        # But we didn't copy data
        if frame_or_series is Series:
            assert np.may_share_memory(obj.values, result.values)
        else:
            assert np.may_share_memory(obj["A"].values, result["A"].values)

        result.iloc[key] = 0
        assert obj.iloc[key] == 1

        # Now we do copy.
        result = obj.set_flags(allows_duplicate_labels=allows_duplicate_labels)
        result.iloc[key] = 10
        assert obj.iloc[key] == 1

    def test_constructor_expanddim(self):
        # [FUNCTION DOC]
        # Regression test (GH 33628, GH 38782). The expansion constructor attribute no longer exists: using it raises an attribute error naming it.
        # GH#33628 accessing _constructor_expanddim should not raise NotImplementedError
        # GH38782 pandas has no container higher than DataFrame (two-dim), so
        # DataFrame._constructor_expand_dim, doesn't make sense, so is removed.
        df = DataFrame()

        msg = "'DataFrame' object has no attribute '_constructor_expanddim'"
        with pytest.raises(AttributeError, match=msg):
            df._constructor_expanddim(np.arange(27).reshape(3, 3, 3))

    def test_inspect_getmembers(self):
        # [FUNCTION DOC]
        # Regression test (GH 38740). Introspecting all members of an empty frame does not raise.
        # GH38740
        df = DataFrame()
        inspect.getmembers(df)
