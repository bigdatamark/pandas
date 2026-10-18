# [MODULE DOC]
# Tests for iterating over the tabular container: key listing, column items,
# row iteration (as Series and as named tuples), and iteration over frames
# holding categorical data. Uses fixtures from the test-suite conftest and a
# platform check used to decide whether the textual form of integers is stable.
# No import-time side effects.
import datetime

import numpy as np
import pytest

from pandas.compat import (
    IS64,
    is_platform_windows,
)

from pandas import (
    Categorical,
    DataFrame,
    Series,
    date_range,
)
import pandas._testing as tm


class TestIteration:
    # [CLASS DOC]
    # Test group for the iteration protocols of the tabular container. Stateless.
    def test_keys(self, float_frame):
        # [FUNCTION DOC]
        # The keys accessor returns the very same object as the columns attribute.
        assert float_frame.keys() is float_frame.columns

    def test_iteritems(self):
        # [FUNCTION DOC]
        # Iterating items of a frame with duplicate column labels yields, for each
        # column, an object of the frame's single-column slice type.
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "a", "b"])
        for k, v in df.items():
            assert isinstance(v, DataFrame._constructor_sliced)

    def test_items(self):
        # [FUNCTION DOC]
        # GH#17213, GH#13918: items yields pairs of column label and Series in column
        # order; labels match the given ones and each Series equals the column selected
        # by that label.
        # GH#17213, GH#13918
        cols = ["a", "b", "c"]
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=cols)
        for c, (k, v) in zip(cols, df.items()):
            assert c == k
            assert isinstance(v, Series)
            assert (df[k] == v).all()

    def test_items_names(self, float_string_frame):
        # [FUNCTION DOC]
        # For a mixed float/string fixture frame, each Series yielded by items has its
        # name equal to the column label.
        for k, v in float_string_frame.items():
            assert v.name == k

    def test_iter(self, float_frame):
        # [FUNCTION DOC]
        # Iterating a frame directly yields the column labels, matching the list of the
        # columns.
        assert list(float_frame) == list(float_frame.columns)

    def test_iterrows(self, float_frame, float_string_frame):
        # [FUNCTION DOC]
        # Row iteration over the float fixture and the mixed float/string fixture:
        # each yielded Series equals the row selected by label from the frame.
        for k, v in float_frame.iterrows():
            exp = float_frame.loc[k]
            tm.assert_series_equal(v, exp)

        for k, v in float_string_frame.iterrows():
            exp = float_string_frame.loc[k]
            tm.assert_series_equal(v, exp)

    def test_iterrows_iso8601(self):
        # [FUNCTION DOC]
        # GH#19671: row iteration on a frame with a string column and a datetime column
        # yields Series equal to the label-selected rows.
        # GH#19671
        s = DataFrame(
            {
                "non_iso8601": ["M1701", "M1802", "M1903", "M2004"],
                "iso8601": date_range("2000-01-01", periods=4, freq="ME"),
            }
        )
        for k, v in s.iterrows():
            exp = s.loc[k]
            tm.assert_series_equal(v, exp)

    def test_iterrows_corner(self):
        # [FUNCTION DOC]
        # GH#12222: a single-row frame with a datetime, None values, an empty string,
        # an empty list, an empty set and an empty dict. The first row from iteration
        # must equal an object-dtype Series holding those values, indexed by column
        # names, with name zero.
        # GH#12222
        df = DataFrame(
            {
                "a": [datetime.datetime(2015, 1, 1)],
                "b": [None],
                "c": [None],
                "d": [""],
                "e": [[]],
                "f": [set()],
                "g": [{}],
            }
        )
        expected = Series(
            [datetime.datetime(2015, 1, 1), None, None, "", [], set(), {}],
            index=list("abcdefg"),
            name=0,
            dtype="object",
        )
        _, result = next(df.iterrows())
        tm.assert_series_equal(result, expected)

    def test_itertuples(self, float_frame):
        # [FUNCTION DOC]
        # Each named tuple from itertuples (index first, then values) rebuilt as a
        # Series equals the corresponding row of the float fixture with its index
        # reset.
        for i, tup in enumerate(float_frame.itertuples()):
            ser = DataFrame._constructor_sliced(tup[1:])
            ser.name = tup[0]
            expected = float_frame.iloc[i, :].reset_index(drop=True)
            tm.assert_series_equal(ser, expected)

    def test_itertuples_index_false(self):
        # [FUNCTION DOC]
        # Row tuples produced without the index have the first value (the integer
        # column's entry) as a built-in integer, not a NumPy scalar.
        df = DataFrame(
            {"floats": np.random.default_rng(2).standard_normal(5), "ints": range(5)},
            columns=["floats", "ints"],
        )

        for tup in df.itertuples(index=False):
            assert isinstance(tup[1], int)

    def test_itertuples_duplicate_cols(self):
        # [FUNCTION DOC]
        # Duplicate column labels selected from a frame still produce plain tuples of
        # index plus values from itertuples. On 64-bit non-Windows platforms the textual
        # form of the tuples with no name is also checked.
        df = DataFrame(data={"a": [1, 2, 3], "b": [4, 5, 6]})
        dfaa = df[["a", "a"]]

        assert list(dfaa.itertuples()) == [(0, 1, 1), (1, 2, 2), (2, 3, 3)]

        # repr with int on 32-bit/windows
        if not (is_platform_windows() or not IS64):
            assert (
                repr(list(df.itertuples(name=None)))
                == "[(0, 1, 4), (1, 2, 5), (2, 3, 6)]"
            )

    def test_itertuples_tuple_name(self):
        # [FUNCTION DOC]
        # With a name given, itertuples yields named tuples whose fields are the index
        # label then the column labels, whose values are accessible by attribute and
        # equal to the plain tuple, and whose type is named as requested.
        df = DataFrame(data={"a": [1, 2, 3], "b": [4, 5, 6]})
        tup = next(df.itertuples(name="TestName"))
        assert tup._fields == ("Index", "a", "b")
        assert (tup.Index, tup.a, tup.b) == tup
        assert type(tup).__name__ == "TestName"

    def test_itertuples_disallowed_col_labels(self):
        # [FUNCTION DOC]
        # Column labels that are not valid identifiers (reserved words) are renamed to
        # positional placeholders in the named tuple fields (index kept), while values
        # are unchanged.
        df = DataFrame(data={"def": [1, 2, 3], "return": [4, 5, 6]})
        tup2 = next(df.itertuples(name="TestName"))
        assert tup2 == (0, 1, 4)
        assert tup2._fields == ("Index", "_1", "_2")

    @pytest.mark.parametrize("limit", [254, 255, 1024])
    @pytest.mark.parametrize("index", [True, False])
    def test_itertuples_py2_3_field_limit_namedtuple(self, limit, index):
        # [FUNCTION DOC]
        # GH#28282: parametrized over field counts of 254, 255 and 1024 and over
        # including the index or not. Named-tuple production from a one-row frame with
        # that many columns must still return a tuple with a fields attribute (i.e. no
        # failure beyond old field limits).
        # GH#28282
        df = DataFrame([{f"foo_{i}": f"bar_{i}" for i in range(limit)}])
        result = next(df.itertuples(index=index))
        assert isinstance(result, tuple)
        assert hasattr(result, "_fields")

    def test_sequence_like_with_categorical(self):
        # [FUNCTION DOC]
        # GH#7839: frames with a categorical column can be iterated: the categorical
        # values listed equal the plain-array list, and itertuples (no index), row
        # iteration and items can be stringified without error.
        # GH#7839
        # make sure can iterate
        df = DataFrame(
            {"id": [1, 2, 3, 4, 5, 6], "raw_grade": ["a", "b", "b", "a", "a", "e"]}
        )
        df["grade"] = Categorical(df["raw_grade"])

        # basic sequencing testing
        result = list(df.grade.values)
        expected = np.array(df.grade.values).tolist()
        tm.assert_almost_equal(result, expected)

        # iteration
        for t in df.itertuples(index=False):
            str(t)

        for row, s in df.iterrows():
            str(s)

        for c, col in df.items():
            str(col)
