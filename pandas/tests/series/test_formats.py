# [MODULE DOC]
# Tests for the text representation (repr/str and LaTeX) of Series, in two classes: general Series representation (names, multi-level indexes, empty and long series, unicode, special index values, extension stubs) and the representation of categorical-valued Series (plain and ordered, of integers, datetimes, periods and timedeltas).
# Expected outputs are exact strings; some depend on options fixtures for the string inference mode. No import-time side effects.

from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Index,
    Series,
    date_range,
    option_context,
    period_range,
    timedelta_range,
)


class TestSeriesRepr:
    # [CLASS DOC]
    # Groups tests of the text representation of Series that are not specific to categoricals. Stateless.
    def test_multilevel_name_print_0(self):
        # [FUNCTION DOC]
        # Verifies (issue 55415) that in a two-level index with level names zero and None, the name zero is printed in the header line and None is not; the repr of a one-value float Series equals the expected multi-line text.
        # GH#55415 None does not get printed, but 0 does
        # (matching DataFrame and flat index behavior)
        mi = pd.MultiIndex.from_product([range(2, 3), range(3, 4)], names=[0, None])
        ser = Series(1.5, index=mi)

        res = repr(ser)
        expected = "0   \n2  3    1.5\ndtype: float64"
        assert res == expected

    def test_multilevel_name_print(self, lexsorted_two_level_string_multiindex):
        # [FUNCTION DOC]
        # Verifies that a Series with a named, lexsorted two-level string index (fixture) and name sth prints the level names header, repeated outer labels blanked out after the first of each group, values right-aligned, and a trailer with the name and dtype.
        index = lexsorted_two_level_string_multiindex
        ser = Series(range(len(index)), index=index, name="sth")
        expected = [
            "first  second",
            "foo    one       0",
            "       two       1",
            "       three     2",
            "bar    one       3",
            "       two       4",
            "baz    two       5",
            "       three     6",
            "qux    one       7",
            "       two       8",
            "       three     9",
            "Name: sth, dtype: int64",
        ]
        expected = "\n".join(expected)
        assert repr(ser) == expected

    def test_small_name_printing(self):
        # [FUNCTION DOC]
        # Verifies on a short Series that the repr includes the name line when a name is set and omits it when the name is cleared.
        # Test small Series.
        s = Series([0, 1, 2])

        s.name = "test"
        assert "Name: test" in repr(s)

        s.name = None
        assert "Name:" not in repr(s)

    def test_big_name_printing(self):
        # [FUNCTION DOC]
        # Verifies the same name display rule as the short-series test but for a 1000-element Series (the truncated-output code path).
        # Test big Series (diff code path).
        s = Series(range(1000))

        s.name = "test"
        assert "Name: test" in repr(s)

        s.name = None
        assert "Name:" not in repr(s)

    def test_empty_name_printing(self):
        # [FUNCTION DOC]
        # Verifies that an empty object-dtype Series with a daily datetime index and a name still shows the name in its repr.
        s = Series(index=date_range("20010101", "20020101"), name="test", dtype=object)
        assert "Name: test" in repr(s)

    @pytest.mark.parametrize("args", [(), (0, -1)])
    def test_float_range(self, args):
        # [FUNCTION DOC]
        # Smoke test: converting to text a 1000-element random float Series with a numeric index (ascending, or descending from 1000 to 1 when the extra arguments are given) does not raise. Parametrized over the range arguments.
        str(
            Series(
                np.random.default_rng(2).standard_normal(1000),
                index=np.arange(1000, *args),
            )
        )

    def test_empty_object(self):
        # [FUNCTION DOC]
        # Smoke test: converting an empty object-dtype Series to text does not raise.
        # empty
        str(Series(dtype=object))

    def test_string(self, string_series):
        # [FUNCTION DOC]
        # Smoke test: converting the string-indexed fixture to text works, also after cast to integer and after assigning NaN to two entries (mutating the fixture).
        str(string_series)
        str(string_series.astype(int))

        # with NaNs
        string_series[5:7] = np.nan
        str(string_series)

    def test_object(self, object_series):
        # [FUNCTION DOC]
        # Smoke test: converting the object-dtype fixture series to text does not raise.
        str(object_series)

    def test_datetime(self, datetime_series):
        # [FUNCTION DOC]
        # Smoke test: text conversion of the datetime fixture works, and the representation of its object-cast copy with every other element set to None works.
        str(datetime_series)
        # with Nones
        ots = datetime_series.astype("O")
        ots[::2] = None
        repr(ots)

    @pytest.mark.parametrize(
        "name",
        [
            "",
            1,
            1.2,
            "foo",
            "\u03b1\u03b2\u03b3",
            "loooooooooooooooooooooooooooooooooooooooooooooooooooong",
            ("foo", "bar", "baz"),
            (1, 2),
            ("foo", 1, 2.3),
            ("\u03b1", "\u03b2", "\u03b3"),
            ("\u03b1", "bar"),
        ],
    )
    def test_various_names(self, name, string_series):
        # [FUNCTION DOC]
        # Smoke test: sets each parametrized name (empty, numbers, ASCII and Greek strings, a very long string, tuples of mixed types) on the string fixture series and builds its repr without error.
        # various names
        string_series.name = name
        repr(string_series)

    def test_tuple_name(self):
        # [FUNCTION DOC]
        # Smoke test: the repr of a 1000-element Series with a tuple name does not raise.
        biggie = Series(
            np.random.default_rng(2).standard_normal(1000),
            index=np.arange(1000),
            name=("foo", "bar", "baz"),
        )
        repr(biggie)

    @pytest.mark.parametrize("arg", [100, 1001])
    def test_tidy_repr_name_0(self, arg):
        # [FUNCTION DOC]
        # Verifies, for a short (100) and a long (1001) random Series with name zero, that the repr contains the name line showing name zero (truncation does not drop it).
        # tidy repr
        ser = Series(np.random.default_rng(2).standard_normal(arg), name=0)
        rep_str = repr(ser)
        assert "Name: 0" in rep_str

    def test_newline(self, any_string_dtype):
        # [FUNCTION DOC]
        # Verifies that control characters (newline, carriage return, tab) in values, index labels and the name are escaped in the repr so none of the raw tab, carriage return or newline-following text appear. Parametrized over any string dtype.
        ser = Series(
            ["a\n\r\tb"],
            name="a\n\r\td",
            index=Index(["a\n\r\tf"], dtype=any_string_dtype),
            dtype=any_string_dtype,
        )
        assert "\t" not in repr(ser)
        assert "\r" not in repr(ser)
        assert "a\n" not in repr(ser)

    @pytest.mark.parametrize(
        "name, expected",
        [
            ["foo", "Series([], Name: foo, dtype: int64)"],
            [None, "Series([], dtype: int64)"],
        ],
    )
    def test_empty_int64(self, name, expected):
        # [FUNCTION DOC]
        # Verifies (issue 4651) the repr of empty int64 Series: with a name it reads as an empty list with the name and dtype; without a name only the dtype is shown. Parametrized over name and expected text.
        # with empty series (#4651)
        s = Series([], dtype=np.int64, name=name)
        assert repr(s) == expected

    def test_repr_bool_fails(self, capsys):
        # [FUNCTION DOC]
        # Verifies that building the repr of a Series whose elements are frames works and writes nothing to the captured error stream (no Cython exception output).
        s = Series(
            [
                DataFrame(np.random.default_rng(2).standard_normal((2, 2)))
                for i in range(5)
            ]
        )

        # It works (with no Cython exception barf)!
        repr(s)

        captured = capsys.readouterr()
        assert captured.err == ""

    def test_repr_name_iterable_indexable(self):
        # [FUNCTION DOC]
        # Smoke test: reprs work for a Series whose name is a NumPy integer and then for one whose name is a tuple of Hebrew characters.
        s = Series([1, 2, 3], name=np.int64(3))

        # it works!
        repr(s)

        s.name = ("\u05d0",) * 2
        repr(s)

    def test_repr_max_rows(self):
        # [FUNCTION DOC]
        # Verifies (issue 6863) that converting a 1001-element Series to text with the maximum-rows display option set to none does not raise.
        # GH 6863
        with option_context("display.max_rows", None):
            str(Series(range(1001)))  # should not raise exception

    def test_unicode_string_with_unicode(self):
        # [FUNCTION DOC]
        # Smoke test: text conversion works for Series with Hebrew values and name, Greek values (ten), and a long Hebrew series (1000) with an ASCII name.
        df = Series(["\u05d0"], name="\u05d1")
        str(df)

        ser = Series(["\u03c3"] * 10)
        repr(ser)

        ser2 = Series(["\u05d0"] * 1000)
        ser2.name = "title1"
        repr(ser2)

    def test_str_to_bytes_raises(self):
        # [FUNCTION DOC]
        # Verifies (issue 26447) that converting a Series to a bytes object raises a type error saying a string object cannot be interpreted as an integer (the bytes constructor treats the Series as a sequence of ints).
        # GH 26447
        df = Series(["abc"], name="abc")
        msg = "^'str' object cannot be interpreted as an integer$"
        with pytest.raises(TypeError, match=msg):
            bytes(df)

    def test_timeseries_repr_object_dtype(self):
        # [FUNCTION DOC]
        # Verifies repr of a 1000-element Series with an object-dtype datetime index works; that a Series with a daily datetime index of 20 entries ends its repr with a frequency line; and that the repr of a random 400-row selection (no regular frequency) can be built and split into lines.
        # KNOWN ISSUE: the final line of the test builds the last line of the representation of the random subset but never asserts anything about it.
        index = Index(
            [datetime(2000, 1, 1) + timedelta(i) for i in range(1000)], dtype=object
        )
        ts = Series(np.random.default_rng(2).standard_normal(len(index)), index)
        repr(ts)

        ts = Series(
            np.arange(20, dtype=np.float64), index=date_range("2020-01-01", periods=20)
        )
        assert repr(ts).splitlines()[-1].startswith("Freq:")

        ts2 = ts.iloc[np.random.default_rng(2).integers(0, len(ts) - 1, 400)]
        repr(ts2).splitlines()[-1]

    def test_latex_repr(self):
        # [FUNCTION DOC]
        # Verifies the LaTeX representation (needs the styling dependency): with escaping disabled and the styler render option set to latex, the representation of a three-element string Series equals the expected tabular markup; with the options restored the representation is none.
        pytest.importorskip("jinja2")  # uses Styler implementation
        result = r"""\begin{tabular}{ll}
\toprule
 & 0 \\
\midrule
0 & $\alpha$ \\
1 & b \\
2 & c \\
\bottomrule
\end{tabular}
"""
        with option_context(
            "styler.format.escape", None, "styler.render.repr", "latex"
        ):
            s = Series([r"$\alpha$", "b", "c"])
            assert result == s._repr_latex_()

        assert s._repr_latex_() is None

    def test_index_repr_in_frame_with_nan(self):
        # [FUNCTION DOC]
        # Verifies (issue 25061) that a Series whose index is a float index containing NaN prints the label as NaN and other labels as floats, with the exact expected text.
        # see gh-25061
        i = Index([1, np.nan])
        s = Series([1, 2], index=i)
        exp = """1.0    1\nNaN    2\ndtype: int64"""

        assert repr(s) == exp

    def test_series_repr_nat(self):
        # [FUNCTION DOC]
        # Verifies the repr of a nanosecond datetime Series including the missing-value sentinel: dates are shown with microsecond precision and the missing entry prints as NaT, aligned.
        series = Series([0, 1000, 2000, pd.NaT._value], dtype="M8[ns]")

        result = repr(series)
        expected = (
            "0   1970-01-01 00:00:00.000000\n"
            "1   1970-01-01 00:00:00.000001\n"
            "2   1970-01-01 00:00:00.000002\n"
            "3                          NaT\n"
            "dtype: datetime64[ns]"
        )
        assert result == expected

    def test_float_repr(self):
        # [FUNCTION DOC]
        # Verifies (issue 35603) that a float Series cast to object prints the value with its decimal point and an object dtype line.
        # GH#35603
        # check float format when cast to object
        ser = Series([1.0]).astype(object)
        expected = "0    1.0\ndtype: object"
        assert repr(ser) == expected

    def test_different_null_objects(self):
        # [FUNCTION DOC]
        # Verifies (issue 45263) that an index mixing True, None, NaN and NaT prints each distinct null-like label verbatim.
        # GH#45263
        ser = Series([1, 2, 3, 4], [True, None, np.nan, pd.NaT])
        result = repr(ser)
        expected = "True    1\nNone    2\nNaN     3\nNaT     4\ndtype: int64"
        assert result == expected

    def test_2d_extension_type(self):
        # [FUNCTION DOC]
        # Verifies (issue 33770) that a Series backed by a minimal extension array whose elements are lists (a stub with only a length, element access and dtype) can produce a repr with the expected text; the stub dtype has the name DtypeStub. Before the fix, this crashed.
        # GH#33770

        # Define a stub extension type with just enough code to run Series.__repr__()
        class DtypeStub(pd.api.extensions.ExtensionDtype):
            # [CLASS DOC]
            # HELPER: stub extension dtype exposing only the scalar type (array class) and the name DtypeStub, just enough for the repr.
            @property
            def type(self):
                # [FUNCTION DOC]
                # HELPER: property returning the NumPy array class as the scalar type.
                return np.ndarray

            @property
            def name(self):
                # [FUNCTION DOC]
                # HELPER: property returning the text name of the dtype.
                return "DtypeStub"

        class ExtTypeStub(pd.api.extensions.ExtensionArray):
            # [CLASS DOC]
            # HELPER: stub extension array of length two whose elements are two-boolean lists, with the stub dtype; provides only what the repr needs.
            def __len__(self) -> int:
                # [FUNCTION DOC]
                # HELPER: returns the constant length two.
                return 2

            def __getitem__(self, ix):
                # [FUNCTION DOC]
                # HELPER: returns, for any given position, a list of two booleans saying whether the position is one and whether it is zero.
                return [ix == 1, ix == 0]

            @property
            def dtype(self):
                # [FUNCTION DOC]
                # HELPER: property returning a new instance of the stub dtype.
                return DtypeStub()

        series = Series(ExtTypeStub(), copy=False)
        res = repr(series)  # This line crashed before GH#33770 was fixed.
        expected = "\n".join(
            ["0    [False True]", "1    [True False]", "dtype: DtypeStub"]
        )
        assert res == expected


class TestCategoricalRepr:
    # [CLASS DOC]
    # Groups tests of the text representation of Series holding categorical data (the trailing lines show the categories, wrapping and truncating long lists, with a less-than separator for ordered categories). Stateless.
    def test_categorical_repr_unicode(self):
        # [FUNCTION DOC]
        # Smoke test (issue 21002): repr and str work on a Series built from an index of sixty-one objects whose text includes a non-ASCII character.
        # see gh-21002

        class County:
            # [CLASS DOC]
            # HELPER: local class with fixed name and state attributes and a text form that joins them with a comma and space; used as categorical values.
            name = "San Sebastián"
            state = "PR"

            def __repr__(self) -> str:
                # [FUNCTION DOC]
                # HELPER: returns the name, a comma and space, then the state.
                return self.name + ", " + self.state

        cat = Categorical([County() for _ in range(61)])
        idx = Index(cat)
        ser = idx.to_series()

        repr(ser)
        str(ser)

    def test_categorical_repr(self, using_infer_string):
        # [FUNCTION DOC]
        # Verifies str of a short integer categorical Series; repr of a 50-element two-category Series with the display row limit of five (rows elided with dots and a length line) where the category dtype shows object, or the string dtype name when the infer-string fixture is on; and str of an ordered 26-category Series where the long categories list is abbreviated with ellipsis, again with the dtype name replaced when the infer-string option is on.
        a = Series(Categorical([1, 2, 3, 4]))
        exp = (
            "0    1\n1    2\n2    3\n3    4\n"
            "dtype: category\nCategories (4, int64): [1, 2, 3, 4]"
        )

        assert exp == a.__str__()

        a = Series(Categorical(["a", "b"] * 25))
        exp = (
            "0     a\n1     b\n"
            "     ..\n"
            "48    a\n49    b\n"
            "Length: 50, dtype: category\nCategories (2, object): ['a', 'b']"
        )
        if using_infer_string:
            exp = exp.replace("object", "str")
        with option_context("display.max_rows", 5):
            assert exp == repr(a)

        levs = list("abcdefghijklmnopqrstuvwxyz")
        a = Series(Categorical(["a", "b"], categories=levs, ordered=True))
        exp = (
            "0    a\n1    b\n"
            "dtype: category\n"
            "Categories (26, object): ['a' < 'b' < 'c' < 'd' ... "
            "'w' < 'x' < 'y' < 'z']"
        )
        if using_infer_string:
            exp = exp.replace("object", "str")
        assert exp == a.__str__()

    def test_categorical_series_repr(self):
        # [FUNCTION DOC]
        # Verifies the repr of unordered integer categorical Series: three categories shown in full, and ten categories abbreviated in the categories line with an ellipsis (the integer dtype name comes from the platform default).
        s = Series(Categorical([1, 2, 3]))
        exp = """0    1
1    2
2    3
dtype: category
Categories (3, int64): [1, 2, 3]"""

        assert repr(s) == exp

        s = Series(Categorical(np.arange(10)))
        exp = f"""0    0
1    1
2    2
3    3
4    4
5    5
6    6
7    7
8    8
9    9
dtype: category
Categories (10, {np.dtype(int)}): [0, 1, 2, 3, ..., 6, 7, 8, 9]"""

        assert repr(s) == exp

    def test_categorical_series_repr_ordered(self):
        # [FUNCTION DOC]
        # Verifies the repr of ordered integer categorical Series: categories joined with less-than signs, abbreviated for ten categories.
        s = Series(Categorical([1, 2, 3], ordered=True))
        exp = """0    1
1    2
2    3
dtype: category
Categories (3, int64): [1 < 2 < 3]"""

        assert repr(s) == exp

        s = Series(Categorical(np.arange(10), ordered=True))
        exp = f"""0    0
1    1
2    2
3    3
4    4
5    5
6    6
7    7
8    8
9    9
dtype: category
Categories (10, {np.dtype(int)}): [0 < 1 < 2 < 3 ... 6 < 7 < 8 < 9]"""

        assert repr(s) == exp

    def test_categorical_series_repr_datetime(self):
        # [FUNCTION DOC]
        # Verifies the repr of unordered categorical datetimes, naive and in the US/Eastern zone (hourly, nanosecond unit): values print as timestamps and the categories line wraps across lines with aligned continuation.
        idx = date_range("2011-01-01 09:00", freq="h", periods=5, unit="ns")
        s = Series(Categorical(idx))
        exp = """0   2011-01-01 09:00:00
1   2011-01-01 10:00:00
2   2011-01-01 11:00:00
3   2011-01-01 12:00:00
4   2011-01-01 13:00:00
dtype: category
Categories (5, datetime64[ns]): [2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00,
                                 2011-01-01 12:00:00, 2011-01-01 13:00:00]"""  # noqa: E501

        assert repr(s) == exp

        idx = date_range(
            "2011-01-01 09:00", freq="h", periods=5, tz="US/Eastern", unit="ns"
        )
        s = Series(Categorical(idx))
        exp = """0   2011-01-01 09:00:00-05:00
1   2011-01-01 10:00:00-05:00
2   2011-01-01 11:00:00-05:00
3   2011-01-01 12:00:00-05:00
4   2011-01-01 13:00:00-05:00
dtype: category
Categories (5, datetime64[ns, US/Eastern]): [2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00,
                                             2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00,
                                             2011-01-01 13:00:00-05:00]"""  # noqa: E501

        assert repr(s) == exp

    def test_categorical_series_repr_datetime_ordered(self):
        # [FUNCTION DOC]
        # Verifies the repr of ordered categorical datetimes (naive and US/Eastern): the categories line uses less-than separators and wraps with aligned continuation lines.
        idx = date_range("2011-01-01 09:00", freq="h", periods=5, unit="ns")
        s = Series(Categorical(idx, ordered=True))
        exp = """0   2011-01-01 09:00:00
1   2011-01-01 10:00:00
2   2011-01-01 11:00:00
3   2011-01-01 12:00:00
4   2011-01-01 13:00:00
dtype: category
Categories (5, datetime64[ns]): [2011-01-01 09:00:00 < 2011-01-01 10:00:00 < 2011-01-01 11:00:00 <
                                 2011-01-01 12:00:00 < 2011-01-01 13:00:00]"""  # noqa: E501

        assert repr(s) == exp

        idx = date_range(
            "2011-01-01 09:00", freq="h", periods=5, tz="US/Eastern", unit="ns"
        )
        s = Series(Categorical(idx, ordered=True))
        exp = """0   2011-01-01 09:00:00-05:00
1   2011-01-01 10:00:00-05:00
2   2011-01-01 11:00:00-05:00
3   2011-01-01 12:00:00-05:00
4   2011-01-01 13:00:00-05:00
dtype: category
Categories (5, datetime64[ns, US/Eastern]): [2011-01-01 09:00:00-05:00 < 2011-01-01 10:00:00-05:00 <
                                             2011-01-01 11:00:00-05:00 < 2011-01-01 12:00:00-05:00 <
                                             2011-01-01 13:00:00-05:00]"""  # noqa: E501

        assert repr(s) == exp

    def test_categorical_series_repr_period(self):
        # [FUNCTION DOC]
        # Verifies the repr of unordered categorical periods, hourly and monthly: values print as periods and the categories line (period dtype shown) wraps for the hourly case and fits on one line for the monthly case.
        idx = period_range("2011-01-01 09:00", freq="h", periods=5)
        s = Series(Categorical(idx))
        exp = """0    2011-01-01 09:00
1    2011-01-01 10:00
2    2011-01-01 11:00
3    2011-01-01 12:00
4    2011-01-01 13:00
dtype: category
Categories (5, period[h]): [2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00,
                            2011-01-01 13:00]"""  # noqa: E501

        assert repr(s) == exp

        idx = period_range("2011-01", freq="M", periods=5)
        s = Series(Categorical(idx))
        exp = """0    2011-01
1    2011-02
2    2011-03
3    2011-04
4    2011-05
dtype: category
Categories (5, period[M]): [2011-01, 2011-02, 2011-03, 2011-04, 2011-05]"""

        assert repr(s) == exp

    def test_categorical_series_repr_period_ordered(self):
        # [FUNCTION DOC]
        # Verifies the repr of ordered categorical periods, hourly and monthly, using less-than separators in the categories line.
        idx = period_range("2011-01-01 09:00", freq="h", periods=5)
        s = Series(Categorical(idx, ordered=True))
        exp = """0    2011-01-01 09:00
1    2011-01-01 10:00
2    2011-01-01 11:00
3    2011-01-01 12:00
4    2011-01-01 13:00
dtype: category
Categories (5, period[h]): [2011-01-01 09:00 < 2011-01-01 10:00 < 2011-01-01 11:00 < 2011-01-01 12:00 <
                            2011-01-01 13:00]"""  # noqa: E501

        assert repr(s) == exp

        idx = period_range("2011-01", freq="M", periods=5)
        s = Series(Categorical(idx, ordered=True))
        exp = """0    2011-01
1    2011-02
2    2011-03
3    2011-04
4    2011-05
dtype: category
Categories (5, period[M]): [2011-01 < 2011-02 < 2011-03 < 2011-04 < 2011-05]"""

        assert repr(s) == exp

    def test_categorical_series_repr_timedelta(self):
        # [FUNCTION DOC]
        # Verifies the repr of unordered categorical timedeltas, daily (five) and hourly (ten) ranges; the categories line shows microsecond timedelta dtype, and the ten-category case is abbreviated and wrapped.
        idx = timedelta_range("1 days", periods=5)
        s = Series(Categorical(idx))
        exp = """0   1 days
1   2 days
2   3 days
3   4 days
4   5 days
dtype: category
Categories (5, timedelta64[us]): [1 days, 2 days, 3 days, 4 days, 5 days]"""

        assert repr(s) == exp

        idx = timedelta_range("1 hours", periods=10)
        s = Series(Categorical(idx))
        exp = """0   0 days 01:00:00
1   1 days 01:00:00
2   2 days 01:00:00
3   3 days 01:00:00
4   4 days 01:00:00
5   5 days 01:00:00
6   6 days 01:00:00
7   7 days 01:00:00
8   8 days 01:00:00
9   9 days 01:00:00
dtype: category
Categories (10, timedelta64[us]): [0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00,
                                   3 days 01:00:00, ..., 6 days 01:00:00, 7 days 01:00:00,
                                   8 days 01:00:00, 9 days 01:00:00]"""  # noqa: E501

        assert repr(s) == exp

    def test_categorical_series_repr_timedelta_ordered(self):
        # [FUNCTION DOC]
        # Verifies the repr of ordered categorical timedeltas, daily (five) and hourly (ten): less-than separators; the ten-category case is abbreviated and wrapped.
        idx = timedelta_range("1 days", periods=5)
        s = Series(Categorical(idx, ordered=True))
        exp = """0   1 days
1   2 days
2   3 days
3   4 days
4   5 days
dtype: category
Categories (5, timedelta64[us]): [1 days < 2 days < 3 days < 4 days < 5 days]"""

        assert repr(s) == exp

        idx = timedelta_range("1 hours", periods=10)
        s = Series(Categorical(idx, ordered=True))
        exp = """0   0 days 01:00:00
1   1 days 01:00:00
2   2 days 01:00:00
3   3 days 01:00:00
4   4 days 01:00:00
5   5 days 01:00:00
6   6 days 01:00:00
7   7 days 01:00:00
8   8 days 01:00:00
9   9 days 01:00:00
dtype: category
Categories (10, timedelta64[us]): [0 days 01:00:00 < 1 days 01:00:00 < 2 days 01:00:00 <
                                   3 days 01:00:00 ... 6 days 01:00:00 < 7 days 01:00:00 <
                                   8 days 01:00:00 < 9 days 01:00:00]"""  # noqa: E501

        assert repr(s) == exp
