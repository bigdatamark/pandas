# [MODULE DOC]
# Tests for the text representation (repr and string rendering) of the tabular
# container: type of the result, unicode, multi-level and null-containing
# indexes, option-controlled output (dimensions, precision, truncation), special
# dtypes (datetime, period, interval, categorical, extension, complex), the
# LaTeX repr, and record-valued cells. Many are smoke tests that only require no
# exception. Uses fixtures from the test-suite conftest; no import-time side
# effects (some tests change display options within a scoped context).
from datetime import (
    datetime,
    timedelta,
)
from io import StringIO

import numpy as np
import pytest

from pandas import (
    NA,
    Categorical,
    CategoricalIndex,
    DataFrame,
    IntervalIndex,
    MultiIndex,
    NaT,
    PeriodIndex,
    Series,
    Timestamp,
    date_range,
    option_context,
    period_range,
)
import pandas._testing as tm


class TestDataFrameRepr:
    # [CLASS DOC]
    # Test group for the frame's repr and string rendering. Stateless.
    def test_repr_should_return_str(self):
        # [FUNCTION DOC]
        # The repr of a frame with unicode labels and of one of its columns is an
        # exact built-in string.
        # https://docs.python.org/3/reference/datamodel.html#object.__repr__
        # "...The return value must be a string object."

        # (str on py2.x, str (unicode) on py3)

        data = [8, 5, 3, 5]
        index1 = ["\u03c3", "\u03c4", "\u03c5", "\u03c6"]
        cols = ["\u03c8"]
        df = DataFrame(data, columns=cols, index=index1)
        assert type(df.__repr__()) is str

        ser = df[cols[0]]
        assert type(ser.__repr__()) is str

    def test_repr_bytes_61_lines(self):
        # [FUNCTION DOC]
        # GH#12857: smoke test that repr works for a 1000x50 frame of one-character
        # strings, including truncated 60-row and 61-row slices (once failed at 61).
        # GH#12857
        lets = list("ACDEFGHIJKLMNOP")
        words = np.random.default_rng(2).choice(lets, (1000, 50))
        df = DataFrame(words).astype("U1")
        assert (df.dtypes == object).all()

        # smoke tests; at one point this raised with 61 but not 60
        repr(df)
        repr(df.iloc[:60, :])
        repr(df.iloc[:61, :])

    def test_repr_unicode_level_names(self, frame_or_series):
        # [FUNCTION DOC]
        # Smoke test, parametrized over frame/Series: repr works with a two-level index
        # whose first level name is a non-ASCII character.
        index = MultiIndex.from_tuples([(0, 0), (1, 1)], names=["\u0394", "i1"])

        obj = DataFrame(np.random.default_rng(2).standard_normal((2, 4)), index=index)
        obj = tm.get_obj(obj, frame_or_series)
        repr(obj)

    def test_assign_index_sequences(self):
        # [FUNCTION DOC]
        # GH#2200: after setting a two-level index and then replacing the index by a
        # list of tuples (and then with one entry as a list rather than a tuple, noted
        # as an improper path), repr still works.
        # GH#2200
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}).set_index(
            ["a", "b"]
        )
        index = list(df.index)
        index[0] = ("faz", "boo")
        df.index = index
        repr(df)

        # this travels an improper code path
        index[0] = ["faz", "boo"]
        df.index = index
        repr(df)

    def test_repr_with_mi_nat(self):
        # [FUNCTION DOC]
        # A frame with a two-level index whose first level has a missing timestamp and a
        # date renders exactly the expected aligned text, showing NaT.
        df = DataFrame({"X": [1, 2]}, index=[[NaT, Timestamp("20130101")], ["a", "b"]])
        result = repr(df)
        expected = "              X\nNaT        a  1\n2013-01-01 b  2"
        assert result == expected

    def test_repr_with_different_nulls(self):
        # [FUNCTION DOC]
        # GH45263: a row index mixing boolean, None, NaN and NaT renders each label
        # verbatim as shown in the expected text.
        # GH45263
        df = DataFrame([1, 2, 3, 4], [True, None, np.nan, NaT])
        result = repr(df)
        expected = """      0
True  1
None  2
NaN   3
NaT   4"""
        assert result == expected

    def test_repr_with_different_nulls_cols(self):
        # [FUNCTION DOC]
        # GH45263: column labels NaN, None, NaT and true render as those labels in the
        # header in the expected text.
        # GH45263
        d = {np.nan: [1, 2], None: [3, 4], NaT: [6, 7], True: [8, 9]}
        df = DataFrame(data=d)
        result = repr(df)
        expected = """   NaN  None  NaT  True
0    1     3    6     8
1    2     4    7     9"""
        assert result == expected

    def test_multiindex_na_repr(self):
        # [FUNCTION DOC]
        # Only an issue with long column names: a frame with six 30-character column
        # names and a three-level row index, re-indexed by two of the columns that hold
        # missing values, can be repr'd without error.
        # only an issue with long columns
        df3 = DataFrame(
            {
                "A" * 30: {("A", "A0006000", "nuit"): "A0006000"},
                "B" * 30: {("A", "A0006000", "nuit"): np.nan},
                "C" * 30: {("A", "A0006000", "nuit"): np.nan},
                "D" * 30: {("A", "A0006000", "nuit"): np.nan},
                "E" * 30: {("A", "A0006000", "nuit"): "A"},
                "F" * 30: {("A", "A0006000", "nuit"): np.nan},
            }
        )

        idf = df3.set_index(["A" * 30, "C" * 30])
        repr(idf)

    def test_repr_name_coincide(self):
        # [FUNCTION DOC]
        # When index level names coincide with column-like text, the third output line
        # of a two-row, three-level index frame starts with the first row's level values.
        index = MultiIndex.from_tuples(
            [("a", 0, "foo"), ("b", 1, "bar")], names=["a", "b", "c"]
        )

        df = DataFrame({"value": [0, 1]}, index=index)

        lines = repr(df).split("\n")
        assert lines[2].startswith("a 0 foo")

    def test_repr_to_string(
        self,
        multiindex_year_month_day_dataframe_random_data,
        multiindex_dataframe_random_data,
    ):
        # [FUNCTION DOC]
        # Smoke test: repr of the multi-level fixture frames and their transposes, and
        # writing their string forms into one in-memory text buffer, do not raise.
        ymd = multiindex_year_month_day_dataframe_random_data
        frame = multiindex_dataframe_random_data

        repr(frame)
        repr(ymd)
        repr(frame.T)
        repr(ymd.T)

        buf = StringIO()
        frame.to_string(buf=buf)
        ymd.to_string(buf=buf)
        frame.T.to_string(buf=buf)
        ymd.T.to_string(buf=buf)

    def test_repr_empty(self):
        # [FUNCTION DOC]
        # Smoke test: repr works for a frame with no rows or columns and for one with
        # 1000 index entries and no columns.
        # empty
        repr(DataFrame())

        # empty with index
        frame = DataFrame(index=np.arange(1000))
        repr(frame)

    def test_repr_mixed(self, float_string_frame):
        # [FUNCTION DOC]
        # Smoke test: repr works for the mixed float/string fixture.
        # mixed
        repr(float_string_frame)

    @pytest.mark.slow
    def test_repr_mixed_big(self):
        # [FUNCTION DOC]
        # Marked slow. Smoke test: repr of a 200-row mixed numeric/string frame with the
        # first rows set to missing.
        # big mixed
        biggie = DataFrame(
            {
                "A": np.random.default_rng(2).standard_normal(200),
                "B": [str(i) for i in range(200)],
            },
            index=range(200),
        )
        biggie.loc[:20, "A"] = np.nan
        biggie.loc[:20, "B"] = np.nan

        repr(biggie)

    def test_repr(self):
        # [FUNCTION DOC]
        # A frame with columns but no rows can be repr'd; a frame whose labels and values
        # contain tab, carriage return and newline characters renders without any tab,
        # carriage return, or the original leading text plus newline.
        # columns but no index
        no_index = DataFrame(columns=[0, 1, 3])
        repr(no_index)

        df = DataFrame(["a\n\r\tb"], columns=["a\n\r\td"], index=["a\n\r\tf"])
        assert "\t" not in repr(df)
        assert "\r" not in repr(df)
        assert "a\n" not in repr(df)

    def test_repr_dimensions(self):
        # [FUNCTION DOC]
        # The dimension footer text (two rows by two columns) appears when the show
        # dimensions display option is true, and not when it is false or when set to
        # truncate on a small frame.
        df = DataFrame([[1, 2], [3, 4]])
        with option_context("display.show_dimensions", True):
            assert "2 rows x 2 columns" in repr(df)

        with option_context("display.show_dimensions", False):
            assert "2 rows x 2 columns" not in repr(df)

        with option_context("display.show_dimensions", "truncate"):
            assert "2 rows x 2 columns" not in repr(df)

    @pytest.mark.slow
    def test_repr_big(self):
        # [FUNCTION DOC]
        # Marked slow. Smoke test: repr of a 200x4 frame of zeros.
        # big one
        biggie = DataFrame(np.zeros((200, 4)), columns=range(4), index=range(200))
        repr(biggie)

    def test_repr_unsortable(self):
        # [FUNCTION DOC]
        # Smoke test: repr works for a frame whose column labels (strings and datetimes)
        # cannot be sorted together.
        # columns are not sortable

        unsortable = DataFrame(
            {
                "foo": [1] * 50,
                datetime.today(): [1] * 50,
                "bar": ["bar"] * 50,
                datetime.today() + timedelta(1): ["bar"] * 50,
            },
            index=np.arange(50),
        )
        repr(unsortable)

    def test_repr_float_frame_options(self, float_frame):
        # [FUNCTION DOC]
        # Smoke test: repr of the float fixture works under default settings and under
        # display options for precision, small maximum rows and columns, and large
        # maximum rows and columns.
        repr(float_frame)

        with option_context("display.precision", 3):
            repr(float_frame)

        with option_context("display.max_rows", 10, "display.max_columns", 2):
            repr(float_frame)

        with option_context("display.max_rows", 1000, "display.max_columns", 1000):
            repr(float_frame)

    def test_repr_unicode(self):
        # [FUNCTION DOC]
        # A frame with a unicode string column renders a header line whose trailing
        # whitespace-stripped text is the column name aligned over the wide strings.
        # KNOWN ISSUE: the second half repeats the same construction and check as the first half; it is redundant.
        uval = "\u03c3\u03c3\u03c3\u03c3"

        df = DataFrame({"A": [uval, uval]})

        result = repr(df)
        ex_top = "      A"
        assert result.split("\n")[0].rstrip() == ex_top

        df = DataFrame({"A": [uval, uval]})
        result = repr(df)
        assert result.split("\n")[0].rstrip() == ex_top

    def test_unicode_string_with_unicode(self):
        # [FUNCTION DOC]
        # Smoke test: string conversion of a frame with a Hebrew letter value works.
        df = DataFrame({"A": ["\u05d0"]})
        str(df)

    def test_repr_unicode_columns(self):
        # [FUNCTION DOC]
        # Smoke test: repr of columns containing Hebrew letters does not raise a
        # decoding error.
        df = DataFrame({"\u05d0": [1, 2, 3], "\u05d1": [4, 5, 6], "c": [7, 8, 9]})
        repr(df.columns)  # should not raise UnicodeDecodeError

    def test_str_to_bytes_raises(self):
        # [FUNCTION DOC]
        # GH 26447: converting a frame to bytes raises a type error because the frame is
        # treated as a sequence of integers; the message must match the expected text
        # exactly.
        # GH 26447
        df = DataFrame({"A": ["abc"]})
        msg = "^'str' object cannot be interpreted as an integer$"
        with pytest.raises(TypeError, match=msg):
            bytes(df)

    def test_very_wide_repr(self):
        # [FUNCTION DOC]
        # Smoke test: repr works for a 10x20 frame with twenty identical long column
        # names (wider than the display).
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 20)),
            columns=np.array(["a" * 10] * 20, dtype=object),
        )
        repr(df)

    def test_repr_column_name_unicode_truncation_bug(self):
        # [FUNCTION DOC]
        # Issue 1906: a column holding a long string with unusual characters is shown
        # (the column name appears in the repr) when the maximum columns option is 20.
        # #1906
        df = DataFrame(
            {
                "Id": [7117434],
                "StringCol": (
                    "Is it possible to modify drop plot code"
                    "so that the output graph is displayed "
                    "in iphone simulator, Is it possible to "
                    "modify drop plot code so that the "
                    "output graph is \xe2\x80\xa8displayed "
                    "in iphone simulator.Now we are adding "
                    "the CSV file externally. I want to Call "
                    "the File through the code.."
                ),
            }
        )

        with option_context("display.max_columns", 20):
            assert "StringCol" in repr(df)

    def test_latex_repr(self):
        # [FUNCTION DOC]
        # Skipped if the template library is unavailable. The LaTeX repr of a frame
        # with a math-like string, under options for no escaping and the LaTeX render
        # repr, equals the expected table text; GH 12182: outside of the option context
        # the LaTeX repr is None.
        pytest.importorskip("jinja2")
        expected = r"""\begin{tabular}{llll}
\toprule
 & 0 & 1 & 2 \\
\midrule
0 & $\alpha$ & b & c \\
1 & 1 & 2 & 3 \\
\bottomrule
\end{tabular}
"""
        with option_context(
            "styler.format.escape", None, "styler.render.repr", "latex"
        ):
            df = DataFrame([[r"$\alpha$", "b", "c"], [1, 2, 3]])
            result = df._repr_latex_()
            assert result == expected

        # GH 12182
        assert df._repr_latex_() is None

    def test_repr_with_datetimeindex(self):
        # [FUNCTION DOC]
        # Repr of a frame with a daily datetime index equals the expected aligned text.
        df = DataFrame({"A": [1, 2, 3]}, index=date_range("2000", periods=3))
        result = repr(df)
        expected = "            A\n2000-01-01  1\n2000-01-02  2\n2000-01-03  3"
        assert result == expected

    def test_repr_with_intervalindex(self):
        # [FUNCTION DOC]
        # Repr of a frame with an interval index built from breaks equals the expected
        # text, intervals shown closed on the right.
        # https://github.com/pandas-dev/pandas/pull/24134/files
        df = DataFrame(
            {"A": [1, 2, 3, 4]}, index=IntervalIndex.from_breaks([0, 1, 2, 3, 4])
        )
        result = repr(df)
        expected = "        A\n(0, 1]  1\n(1, 2]  2\n(2, 3]  3\n(3, 4]  4"
        assert result == expected

    def test_repr_with_categorical_index(self):
        # [FUNCTION DOC]
        # Repr of a frame with a categorical index equals the expected aligned text.
        df = DataFrame({"A": [1, 2, 3]}, index=CategoricalIndex(["a", "b", "c"]))
        result = repr(df)
        expected = "   A\na  1\nb  2\nc  3"
        assert result == expected

    def test_repr_categorical_dates_periods(self):
        # [FUNCTION DOC]
        # A frame with a timezone-aware hourly column and a monthly period column
        # renders to the expected text, and a frame with the same data as categoricals
        # renders identically.
        # normal DataFrame
        dt = date_range("2011-01-01 09:00", freq="h", periods=5, tz="US/Eastern")
        p = period_range("2011-01", freq="M", periods=5)
        df = DataFrame({"dt": dt, "p": p})
        exp = """                         dt        p
0 2011-01-01 09:00:00-05:00  2011-01
1 2011-01-01 10:00:00-05:00  2011-02
2 2011-01-01 11:00:00-05:00  2011-03
3 2011-01-01 12:00:00-05:00  2011-04
4 2011-01-01 13:00:00-05:00  2011-05"""

        assert repr(df) == exp

        df2 = DataFrame({"dt": Categorical(dt), "p": Categorical(p)})
        assert repr(df2) == exp

    @pytest.mark.parametrize(
        "nat",
        [np.datetime64("NaT", "ns"), np.timedelta64("NaT", "ns")],
    )
    @pytest.mark.parametrize(
        "box, expected",
        [[Series, "0    NaT\ndtype: object"], [DataFrame, "     0\n0  NaT"]],
    )
    def test_repr_np_nat_with_object(self, nat, box, expected):
        # [FUNCTION DOC]
        # GH 25445: parametrized over a datetime and a timedelta not-a-time value in
        # nanoseconds, and over Series/frame with expected texts. The repr of an
        # object-dtype container holding it shows the NaT label and (for Series) object
        # dtype.
        # GH 25445
        result = repr(box([nat], dtype=object))
        assert result == expected

    def test_frame_datetime64_pre1900_repr(self):
        # [FUNCTION DOC]
        # Smoke test: repr works for a frame with datetime values from year 1700 at
        # yearly frequency.
        df = DataFrame({"year": date_range("1/1/1700", periods=50, freq="YE-DEC")})
        # it works!
        repr(df)

    def test_frame_to_string_with_periodindex(self):
        # [FUNCTION DOC]
        # Smoke test: string rendering works for a frame with a monthly period index.
        index = PeriodIndex(["2011-1", "2011-2", "2011-3"], freq="M")
        frame = DataFrame(np.random.default_rng(2).standard_normal((3, 4)), index=index)

        # it works!
        frame.to_string()

    def test_to_string_ea_na_in_multiindex(self):
        # [FUNCTION DOC]
        # GH#47986: string rendering of a frame with a nullable integer index
        # containing a missing value shows the missing marker for NA.
        # GH#47986
        df = DataFrame(
            {"a": [1, 2]},
            index=MultiIndex.from_arrays([Series([NA, 1], dtype="Int64")]),
        )

        result = df.to_string()
        expected = """      a
<NA>  1
1     2"""
        assert result == expected

    def test_datetime64tz_slice_non_truncate(self):
        # [FUNCTION DOC]
        # GH 30263: the repr of a timezone-aware datetime frame equals the repr after
        # selecting the first five columns (which is all of them), i.e. no truncation
        # difference.
        # GH 30263
        df = DataFrame({"x": date_range("2019", periods=10, tz="UTC")})
        expected = repr(df)
        df = df.iloc[:, :5]
        result = repr(df)
        assert result == expected

    def test_to_records_no_typeerror_in_repr(self):
        # [FUNCTION DOC]
        # GH 48526: assigning the record-array conversion of two string columns as a new
        # column does not raise in repr, and the repr shows records as lists of index and
        # values.
        # GH 48526
        df = DataFrame([["a", "b"], ["c", "d"], ["e", "f"]], columns=["left", "right"])
        df["record"] = df[["left", "right"]].to_records()
        expected = """  left right     record
0    a     b  [0, a, b]
1    c     d  [1, c, d]
2    e     f  [2, e, f]"""
        result = repr(df)
        assert result == expected

    def test_to_records_with_na_record_value(self):
        # [FUNCTION DOC]
        # GH 48526: like the previous record repr test but with a missing value in a
        # cell, shown as nan inside the record text.
        # GH 48526
        df = DataFrame(
            [["a", np.nan], ["c", "d"], ["e", "f"]], columns=["left", "right"]
        )
        df["record"] = df[["left", "right"]].to_records()
        expected = """  left right       record
0    a   NaN  [0, a, nan]
1    c     d    [1, c, d]
2    e     f    [2, e, f]"""
        result = repr(df)
        assert result == expected

    def test_to_records_with_na_record(self):
        # [FUNCTION DOC]
        # GH 48526: record repr when a column label is NaN and a row is entirely missing;
        # the record text shows nan values.
        # GH 48526
        df = DataFrame(
            [["a", "b"], [np.nan, np.nan], ["e", "f"]], columns=[np.nan, "right"]
        )
        df["record"] = df[[np.nan, "right"]].to_records()
        expected = """   NaN right         record
0    a     b      [0, a, b]
1  NaN   NaN  [1, nan, nan]
2    e     f      [2, e, f]"""
        result = repr(df)
        assert result == expected

    def test_to_records_with_inf_record(self):
        # [FUNCTION DOC]
        # GH 48526: record repr when column labels are NaN and infinity; the expected
        # text shows infinity and nan in the records.
        # GH 48526
        expected = """   NaN  inf         record
0  inf    b    [0, inf, b]
1  NaN  NaN  [1, nan, nan]
2    e    f      [2, e, f]"""
        df = DataFrame(
            [[np.inf, "b"], [np.nan, np.nan], ["e", "f"]],
            columns=[np.nan, np.inf],
        )
        df["record"] = df[[np.nan, np.inf]].to_records()
        result = repr(df)
        assert result == expected

    def test_masked_ea_with_formatter(self):
        # [FUNCTION DOC]
        # GH#39336: string rendering of nullable float and integer columns with a
        # per-column two-decimal formatter gives the expected two-decimal text for both.
        # GH#39336
        df = DataFrame(
            {
                "a": Series([0.123456789, 1.123456789], dtype="Float64"),
                "b": Series([1, 2], dtype="Int64"),
            }
        )
        result = df.to_string(formatters=["{:.2f}".format, "{:.2f}".format])
        expected = """      a     b
0  0.12  1.00
1  1.12  2.00"""
        assert result == expected

    def test_repr_ea_columns(self, any_string_dtype):
        # [FUNCTION DOC]
        # GH#54797: skipped without the Arrow library. For each string dtype fixture,
        # casting the column labels to that dtype leaves the repr identical to the
        # expected aligned text.
        # GH#54797
        pytest.importorskip("pyarrow")
        df = DataFrame({"long_column_name": [1, 2, 3], "col2": [4, 5, 6]})
        df.columns = df.columns.astype(any_string_dtype)
        expected = """   long_column_name  col2
0                 1     4
1                 2     5
2                 3     6"""
        assert repr(df) == expected


@pytest.mark.parametrize(
    "data,output",
    [
        ([2, complex("nan"), 1], [" 2.0+0.0j", " NaN+0.0j", " 1.0+0.0j"]),
        ([2, complex("nan"), -1], [" 2.0+0.0j", " NaN+0.0j", "-1.0+0.0j"]),
        ([-2, complex("nan"), -1], ["-2.0+0.0j", " NaN+0.0j", "-1.0+0.0j"]),
        ([-1.23j, complex("nan"), -1], ["-0.00-1.23j", "  NaN+0.00j", "-1.00+0.00j"]),
        ([1.23j, complex("nan"), 1.23], [" 0.00+1.23j", "  NaN+0.00j", " 1.23+0.00j"]),
        (
            [-1.23j, complex(np.nan, np.nan), 1],
            ["-0.00-1.23j", "  NaN+ NaNj", " 1.00+0.00j"],
        ),
        (
            [-1.23j, complex(1.2, np.nan), 1],
            ["-0.00-1.23j", " 1.20+ NaNj", " 1.00+0.00j"],
        ),
        (
            [-1.23j, complex(np.nan, -1.2), 1],
            ["-0.00-1.23j", "  NaN-1.20j", " 1.00+0.00j"],
        ),
    ],
)
@pytest.mark.parametrize("as_frame", [True, False])
def test_repr_with_complex_nans(data, output, as_frame):
    # [FUNCTION DOC]
    # GH#53762, GH#53841: module-level test parametrized over complex data with
    # NaNs (and expected per-element text) and over frame/Series. Repr of a complex
    # Series is index, padding, formatted element, ending with the dtype line; as a
    # frame one column named val is right-aligned in a header line followed by
    # rows. The string form must equal the constructed expected text; the failure
    # message prints both.
    # GH#53762, GH#53841
    obj = Series(np.array(data))
    if as_frame:
        obj = obj.to_frame(name="val")
        reprs = [f"{i} {val}" for i, val in enumerate(output)]
        expected = f"{'val': >{len(reprs[0])}}\n" + "\n".join(reprs)
    else:
        reprs = [f"{i}   {val}" for i, val in enumerate(output)]
        expected = "\n".join(reprs) + "\ndtype: complex128"
    assert str(obj) == expected, f"\n{obj!s}\n\n{expected}"
