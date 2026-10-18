# [MODULE DOC]
# Tests for the frame's query and eval methods (expression evaluation against
# columns, index levels, local variables and resolvers) across engines (plain
# interpreter-style python and the optional numexpr library) and parsers (python
# and pandas). Defines module-level parametrizing fixtures for parser and engine
# and a skip helper, then test classes grouped by engine/parser combination and
# by topic: compatibility, basic eval, multi-level index queries, date queries,
# local variable and backtick-quoted names, error cases, and option-dependent
# behavior. Imports private computation helpers (numexpr availability flag).
# No import-time side effects besides fixture registration.
import operator
from tokenize import TokenError

import numpy as np
import pytest

from pandas.errors import (
    NumExprClobberingError,
    UndefinedVariableError,
)
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    date_range,
)
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED


@pytest.fixture(params=["python", "pandas"], ids=lambda x: x)
def parser(request):
    # [FUNCTION DOC]
    # HELPER: fixture for this module's tests. Parametrized over the parser names
    # python and pandas; returns the parser name for the current parameter.
    return request.param


@pytest.fixture(
    params=["python", pytest.param("numexpr", marks=td.skip_if_no("numexpr"))],
    ids=lambda x: x,
)
def engine(request):
    # [FUNCTION DOC]
    # HELPER: fixture for this module's tests. Parametrized over the engine names
    # python and numexpr (numexpr parametrization is skipped when that library is
    # missing); returns the engine name for the current parameter.
    return request.param


def skip_if_no_pandas_parser(parser):
    # [FUNCTION DOC]
    # HELPER: skips (marks the running test as skipped) when the given parser name
    # is anything other than pandas, with a message naming the parser. Used by tests
    # that rely on syntax only the pandas parser supports. Returns nothing.
    if parser != "pandas":
        pytest.skip(f"cannot evaluate with parser={parser}")


class TestCompat:
    # [CLASS DOC]
    # Test group checking that query and eval work with the default engine, an
    # explicit missing engine, the python engine, and numexpr (or give the proper
    # import error when missing). Uses the small fixtures below.
    @pytest.fixture
    def df(self):
        # [FUNCTION DOC]
        # HELPER: fixture for the compatibility tests. A single-column frame with
        # integer values one to three.
        return DataFrame({"A": [1, 2, 3]})

    @pytest.fixture
    def expected1(self, df):
        # [FUNCTION DOC]
        # HELPER: fixture for the compatibility tests. The expected result of the query
        # for positive values of the column: all rows of the frame.
        return df[df.A > 0]

    @pytest.fixture
    def expected2(self, df):
        # [FUNCTION DOC]
        # HELPER: fixture for the compatibility tests. The expected result of adding
        # one to the column: a Series with values two to four.
        return df.A + 1

    def test_query_default(self, df, expected1, expected2):
        # [FUNCTION DOC]
        # GH 12749: the default engine always works regardless of numexpr availability:
        # the query for positive values and the evaluation of the column plus one equal
        # the fixtures.
        # GH 12749
        # this should always work, whether NUMEXPR_INSTALLED or not
        result = df.query("A>0")
        tm.assert_frame_equal(result, expected1)
        result = df.eval("A+1")
        tm.assert_series_equal(result, expected2)

    def test_query_None(self, df, expected1, expected2):
        # [FUNCTION DOC]
        # Passing an engine of none behaves like the default for both query and eval.
        result = df.query("A>0", engine=None)
        tm.assert_frame_equal(result, expected1)
        result = df.eval("A+1", engine=None)
        tm.assert_series_equal(result, expected2)

    def test_query_python(self, df, expected1, expected2):
        # [FUNCTION DOC]
        # Explicitly selecting the python engine gives the expected results for both
        # query and eval.
        result = df.query("A>0", engine="python")
        tm.assert_frame_equal(result, expected1)
        result = df.eval("A+1", engine="python")
        tm.assert_series_equal(result, expected2)

    def test_query_numexpr(self, df, expected1, expected2):
        # [FUNCTION DOC]
        # If numexpr is installed, the numexpr engine gives the expected results;
        # otherwise both query and eval raise an import error whose message says numexpr
        # is not installed or unsupported.
        if NUMEXPR_INSTALLED:
            result = df.query("A>0", engine="numexpr")
            tm.assert_frame_equal(result, expected1)
            result = df.eval("A+1", engine="numexpr")
            tm.assert_series_equal(result, expected2)
        else:
            msg = (
                r"'numexpr' is not installed or an unsupported version. "
                r"Cannot use engine='numexpr' for query/eval if 'numexpr' is "
                r"not installed"
            )
            with pytest.raises(ImportError, match=msg):
                df.query("A>0", engine="numexpr")
            with pytest.raises(ImportError, match=msg):
                df.eval("A+1", engine="numexpr")


class TestDataFrameEval:
    # [CLASS DOC]
    # Test group for eval and query behavior independent of index type: operators,
    # argument validation, duplicate names, resolvers, and special dtypes.
    # smaller hits python, larger hits numexpr
    @pytest.mark.parametrize("n", [4, 4000])
    @pytest.mark.parametrize(
        "op_str,op,rop",
        [
            ("+", "__add__", "__radd__"),
            ("-", "__sub__", "__rsub__"),
            ("*", "__mul__", "__rmul__"),
            ("/", "__truediv__", "__rtruediv__"),
        ],
    )
    def test_ops(self, op_str, op, rop, n):
        # [FUNCTION DOC]
        # GH7198: parametrized over frame size four (python path) and 4000 (numexpr
        # path) and arithmetic operator (add, subtract, multiply, true divide) with names
        # of the normal and reflected methods. Builds an all-ones frame whose first row
        # is two, takes column means, expands them into a frame of repeated mean rows,
        # and checks that the operator with a mean Series on the left equals the
        # operator on the expanded frame.
        # KNOWN ISSUE: the later branches compare the dunder-method name to operator symbols, so neither branch can ever run; the commutative and non-commutative method-call checks are dead code (the operator symbol variable was probably intended).
        # tst ops and reversed ops in evaluation
        # GH7198

        df = DataFrame(1, index=range(n), columns=list("abcd"))
        df.iloc[0] = 2
        m = df.mean()

        base = DataFrame(  # noqa: F841
            np.tile(m.values, n).reshape(n, -1), columns=list("abcd")
        )

        expected = eval(f"base {op_str} df")

        # ops as strings
        result = eval(f"m {op_str} df")
        tm.assert_frame_equal(result, expected)

        # these are commutative
        if op in ["+", "*"]:
            result = getattr(df, op)(m)
            tm.assert_frame_equal(result, expected)

        # these are not
        elif op in ["-", "/"]:
            result = getattr(df, rop)(m)
            tm.assert_frame_equal(result, expected)

    def test_dataframe_sub_numexpr_path(self):
        # [FUNCTION DOC]
        # GH7192: with 25000 rows (enough to use the numexpr path) and the first five
        # rows missing, one minus the NaN indicator on the first 25 rows equals the same
        # slice of the full computation.
        # GH7192: Note we need a large number of rows to ensure this
        #  goes through the numexpr path
        df = DataFrame({"A": np.random.default_rng(2).standard_normal(25000)})
        df.iloc[0:5] = np.nan
        expected = 1 - np.isnan(df.iloc[0:25])
        result = (1 - np.isnan(df)).iloc[0:25]
        tm.assert_frame_equal(result, expected)

    def test_query_non_str(self):
        # [FUNCTION DOC]
        # GH 11485: query with a non-string expression (a function or an integer) raises
        # a value error saying the expression must be a string.
        # GH 11485
        df = DataFrame({"A": [1, 2, 3], "B": ["a", "b", "b"]})

        msg = "expr must be a string to be evaluated"
        with pytest.raises(ValueError, match=msg):
            df.query(lambda x: x.B == "b")

        with pytest.raises(ValueError, match=msg):
            df.query(111)

    def test_query_empty_string(self):
        # [FUNCTION DOC]
        # GH 13139: query with an empty string raises a value error saying the
        # expression cannot be empty.
        # GH 13139
        df = DataFrame({"A": [1, 2, 3]})

        msg = "expr cannot be an empty string"
        with pytest.raises(ValueError, match=msg):
            df.query("")

    def test_query_duplicate_column_name(self, engine, parser):
        # [FUNCTION DOC]
        # Query on a frame with duplicated column name A (after renaming) filtering by
        # another column returns the single matching row with both A columns, for every
        # engine and parser combination.
        df = DataFrame({"A": range(3), "B": range(3), "C": range(3)}).rename(
            columns={"B": "A"}
        )

        res = df.query("C == 1", engine=engine, parser=parser)

        expect = DataFrame([[1, 1, 1]], columns=["A", "A", "C"], index=[1])

        tm.assert_frame_equal(res, expect)

    def test_eval_resolvers_as_list(self):
        # [FUNCTION DOC]
        # GH 14095: both frame eval and the top-level eval accept resolvers as a list of
        # mappings, and combine them so the sum of a and b equals the sum of the values.
        # GH 14095
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 2)), columns=list("ab")
        )
        dict1 = {"a": 1}
        dict2 = {"b": 2}
        assert df.eval("a + b", resolvers=[dict1, dict2]) == dict1["a"] + dict2["b"]
        assert pd.eval("a + b", resolvers=[dict1, dict2]) == dict1["a"] + dict2["b"]

    def test_eval_resolvers_combined(self):
        # [FUNCTION DOC]
        # GH 34966: resolvers passed explicitly can be used together with the frame's
        # own column resolvers: a plus b times the external constant equals the Series
        # computed directly.
        # GH 34966
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 2)), columns=list("ab")
        )
        dict1 = {"c": 2}

        # Both input and default index/column resolvers should be usable
        result = df.eval("a + b * c", resolvers=[dict1])

        expected = df["a"] + df["b"] * dict1["c"]
        tm.assert_series_equal(result, expected)

    def test_eval_object_dtype_binop(self):
        # [FUNCTION DOC]
        # GH#24883: assignment-style eval creating a boolean column from comparing an
        # object column to a string and combining with a boolean constant returns a frame
        # with the new boolean column.
        # GH#24883
        df = DataFrame({"a1": ["Y", "N"]})
        res = df.eval("c = ((a1 == 'Y') & True)")
        expected = DataFrame({"a1": ["Y", "N"], "c": [True, False]})
        tm.assert_frame_equal(res, expected)

    def test_using_numpy(self, engine, parser):
        # [FUNCTION DOC]
        # GH 58041: skipped for non-pandas parser. A call to a NumPy function referenced
        # through the local-variable prefix works in eval and equals applying it to the
        # column directly.
        # GH 58041
        skip_if_no_pandas_parser(parser)
        df = Series([0.2, 1.5, 2.8], name="a").to_frame()
        res = df.eval("@np.floor(a)", engine=engine, parser=parser)
        expected = np.floor(df["a"])
        tm.assert_series_equal(expected, res)

    def test_eval_simple(self, engine, parser):
        # [FUNCTION DOC]
        # Evaluating just a column name returns the column, for every engine and parser.
        df = Series([0.2, 1.5, 2.8], name="a").to_frame()
        res = df.eval("a", engine=engine, parser=parser)
        expected = df["a"]
        tm.assert_series_equal(expected, res)

    def test_extension_array_eval(self, engine, parser, request):
        # [FUNCTION DOC]
        # GH#58748: dividing two nullable-integer columns gives the expected Series of
        # floats; for numexpr the test is marked expected-fail because that engine does
        # not support extension array dtypes.
        # GH#58748
        if engine == "numexpr":
            mark = pytest.mark.xfail(
                reason="numexpr does not support extension array dtypes"
            )
            request.applymarker(mark)
        df = DataFrame({"a": pd.array([1, 2, 3]), "b": pd.array([4, 5, 6])})
        result = df.eval("a / b", engine=engine, parser=parser)
        expected = Series(pd.array([0.25, 0.40, 0.50]))
        tm.assert_series_equal(result, expected)

    def test_complex_eval(self, engine, parser):
        # [FUNCTION DOC]
        # GH#21374: dividing complex columns gives the expected complex Series for every
        # engine and parser.
        # GH#21374
        df = DataFrame({"a": [1 + 2j], "b": [1 + 1j]})
        result = df.eval("a/b", engine=engine, parser=parser)
        expected = Series([1.5 + 0.5j])
        tm.assert_series_equal(result, expected)


class TestDataFrameQueryWithMultiIndex:
    # [CLASS DOC]
    # Test group for queries against named, unnamed and partly named multi-level
    # indexes using level names or positional placeholders.
    def test_query_with_named_multiindex(self, parser, engine):
        # [FUNCTION DOC]
        # Skipped for non-pandas parser. For a two-level index with names color and food,
        # compares query results with boolean selection for: equality and inequality (both
        # operand orders), comparison with a one-element list (membership semantic),
        # and in / not in, against the expected rows.
        skip_if_no_pandas_parser(parser)
        a = np.random.default_rng(2).choice(["red", "green"], size=10)
        b = np.random.default_rng(2).choice(["eggs", "ham"], size=10)
        index = MultiIndex.from_arrays([a, b], names=["color", "food"])
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), index=index)
        ind = Series(
            df.index.get_level_values("color").values, index=index, name="color"
        )

        # equality
        res1 = df.query('color == "red"', parser=parser, engine=engine)
        res2 = df.query('"red" == color', parser=parser, engine=engine)
        exp = df[ind == "red"]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # inequality
        res1 = df.query('color != "red"', parser=parser, engine=engine)
        res2 = df.query('"red" != color', parser=parser, engine=engine)
        exp = df[ind != "red"]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # list equality (really just set membership)
        res1 = df.query('color == ["red"]', parser=parser, engine=engine)
        res2 = df.query('["red"] == color', parser=parser, engine=engine)
        exp = df[ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        res1 = df.query('color != ["red"]', parser=parser, engine=engine)
        res2 = df.query('["red"] != color', parser=parser, engine=engine)
        exp = df[~ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # in/not in ops
        res1 = df.query('["red"] in color', parser=parser, engine=engine)
        res2 = df.query('"red" in color', parser=parser, engine=engine)
        exp = df[ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        res1 = df.query('["red"] not in color', parser=parser, engine=engine)
        res2 = df.query('"red" not in color', parser=parser, engine=engine)
        exp = df[~ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

    def test_query_with_unnamed_multiindex(self, parser, engine):
        # [FUNCTION DOC]
        # Skipped for non-pandas parser. Same checks as for named levels, on an unnamed
        # two-level index using positional level references, for both level zero (colors)
        # and level one (foods).
        skip_if_no_pandas_parser(parser)
        a = np.random.default_rng(2).choice(["red", "green"], size=10)
        b = np.random.default_rng(2).choice(["eggs", "ham"], size=10)
        index = MultiIndex.from_arrays([a, b])
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), index=index)
        ind = Series(df.index.get_level_values(0).values, index=index)

        res1 = df.query('ilevel_0 == "red"', parser=parser, engine=engine)
        res2 = df.query('"red" == ilevel_0', parser=parser, engine=engine)
        exp = df[ind == "red"]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # inequality
        res1 = df.query('ilevel_0 != "red"', parser=parser, engine=engine)
        res2 = df.query('"red" != ilevel_0', parser=parser, engine=engine)
        exp = df[ind != "red"]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # list equality (really just set membership)
        res1 = df.query('ilevel_0 == ["red"]', parser=parser, engine=engine)
        res2 = df.query('["red"] == ilevel_0', parser=parser, engine=engine)
        exp = df[ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        res1 = df.query('ilevel_0 != ["red"]', parser=parser, engine=engine)
        res2 = df.query('["red"] != ilevel_0', parser=parser, engine=engine)
        exp = df[~ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # in/not in ops
        res1 = df.query('["red"] in ilevel_0', parser=parser, engine=engine)
        res2 = df.query('"red" in ilevel_0', parser=parser, engine=engine)
        exp = df[ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        res1 = df.query('["red"] not in ilevel_0', parser=parser, engine=engine)
        res2 = df.query('"red" not in ilevel_0', parser=parser, engine=engine)
        exp = df[~ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # ## LEVEL 1
        ind = Series(df.index.get_level_values(1).values, index=index)
        res1 = df.query('ilevel_1 == "eggs"', parser=parser, engine=engine)
        res2 = df.query('"eggs" == ilevel_1', parser=parser, engine=engine)
        exp = df[ind == "eggs"]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # inequality
        res1 = df.query('ilevel_1 != "eggs"', parser=parser, engine=engine)
        res2 = df.query('"eggs" != ilevel_1', parser=parser, engine=engine)
        exp = df[ind != "eggs"]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # list equality (really just set membership)
        res1 = df.query('ilevel_1 == ["eggs"]', parser=parser, engine=engine)
        res2 = df.query('["eggs"] == ilevel_1', parser=parser, engine=engine)
        exp = df[ind.isin(["eggs"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        res1 = df.query('ilevel_1 != ["eggs"]', parser=parser, engine=engine)
        res2 = df.query('["eggs"] != ilevel_1', parser=parser, engine=engine)
        exp = df[~ind.isin(["eggs"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # in/not in ops
        res1 = df.query('["eggs"] in ilevel_1', parser=parser, engine=engine)
        res2 = df.query('"eggs" in ilevel_1', parser=parser, engine=engine)
        exp = df[ind.isin(["eggs"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        res1 = df.query('["eggs"] not in ilevel_1', parser=parser, engine=engine)
        res2 = df.query('"eggs" not in ilevel_1', parser=parser, engine=engine)
        exp = df[~ind.isin(["eggs"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

    def test_query_with_partially_named_multiindex(self, parser, engine):
        # [FUNCTION DOC]
        # Skipped for non-pandas parser. For an index with one unnamed level and one
        # named rating, queries by the name and by the positional placeholder return the
        # rows matching the equivalent boolean selection (equality and inequality).
        skip_if_no_pandas_parser(parser)
        a = np.random.default_rng(2).choice(["red", "green"], size=10)
        b = np.arange(10)
        index = MultiIndex.from_arrays([a, b])
        index.names = [None, "rating"]
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), index=index)
        res = df.query("rating == 1", parser=parser, engine=engine)
        ind = Series(
            df.index.get_level_values("rating").values, index=index, name="rating"
        )
        exp = df[ind == 1]
        tm.assert_frame_equal(res, exp)

        res = df.query("rating != 1", parser=parser, engine=engine)
        ind = Series(
            df.index.get_level_values("rating").values, index=index, name="rating"
        )
        exp = df[ind != 1]
        tm.assert_frame_equal(res, exp)

        res = df.query('ilevel_0 == "red"', parser=parser, engine=engine)
        ind = Series(df.index.get_level_values(0).values, index=index)
        exp = df[ind == "red"]
        tm.assert_frame_equal(res, exp)

        res = df.query('ilevel_0 != "red"', parser=parser, engine=engine)
        ind = Series(df.index.get_level_values(0).values, index=index)
        exp = df[ind != "red"]
        tm.assert_frame_equal(res, exp)

    def test_query_multiindex_get_index_resolvers(self):
        # [FUNCTION DOC]
        # The private resolver builder for a two-level named index maps: the index itself,
        # the columns Series, one Series per level name (indexed by the full multi-level
        # index) and the column level placeholder. Index values are compared by identity,
        # Series by equality, anything else fails the test.
        df = DataFrame(
            np.ones((10, 3)),
            index=MultiIndex.from_arrays(
                [range(10) for _ in range(2)], names=["spam", "eggs"]
            ),
        )
        resolvers = df._get_index_resolvers()

        def to_series(mi, level):
            # [FUNCTION DOC]
            # HELPER: for the resolver test. Converts the given level of a multi-level index
            # to a Series of that level's values indexed by the whole multi-level index.
            level_values = mi.get_level_values(level)
            s = level_values.to_series()
            s.index = mi
            return s

        col_series = df.columns.to_series()
        expected = {
            "index": df.index,
            "columns": col_series,
            "spam": to_series(df.index, "spam"),
            "eggs": to_series(df.index, "eggs"),
            "clevel_0": col_series,
        }
        for k, v in resolvers.items():
            if isinstance(v, Index):
                assert v.is_(expected[k])
            elif isinstance(v, Series):
                tm.assert_series_equal(v, expected[k])
            else:
                raise AssertionError("object must be a Series or Index")


@td.skip_if_no("numexpr")
class TestDataFrameQueryNumExprPandas:
    # [CLASS DOC]
    # Test group for date-related queries with the numexpr engine and pandas parser,
    # skipped without numexpr. Provides engine and parser fixtures fixing these
    # values; the other classes below are parallel variants.
    @pytest.fixture
    def engine(self):
        # [FUNCTION DOC]
        # HELPER: fixture overriding the module fixture so the group always uses the
        # numexpr engine.
        return "numexpr"

    @pytest.fixture
    def parser(self):
        # [FUNCTION DOC]
        # HELPER: fixture overriding the module fixture so the group always uses the
        # pandas parser.
        return "pandas"

    def test_date_query_with_attribute_access(self, engine, parser):
        # [FUNCTION DOC]
        # Skipped for non-pandas parser. Queries comparing a datetime column to an
        # integer date on both sides (using attribute-style references with the local
        # prefix) equal the boolean selection with date strings.
        skip_if_no_pandas_parser(parser)
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        df["dates1"] = date_range("1/1/2012", periods=5)
        df["dates2"] = date_range("1/1/2013", periods=5)
        df["dates3"] = date_range("1/1/2014", periods=5)
        res = df.query(
            "@df.dates1 < 20130101 < @df.dates3", engine=engine, parser=parser
        )
        expec = df[(df.dates1 < "20130101") & ("20130101" < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_query_no_attribute_access(self, engine, parser):
        # [FUNCTION DOC]
        # Chained comparison of a date column against a date integer and another date
        # column by plain column names equals the equivalent boolean selection.
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        df["dates1"] = date_range("1/1/2012", periods=5)
        df["dates2"] = date_range("1/1/2013", periods=5)
        df["dates3"] = date_range("1/1/2014", periods=5)
        res = df.query("dates1 < 20130101 < dates3", engine=engine, parser=parser)
        expec = df[(df.dates1 < "20130101") & ("20130101" < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_query_with_NaT(self, engine, parser):
        # [FUNCTION DOC]
        # As the plain date query, but with random rows of two date columns set to
        # missing; missing values are excluded from the results.
        n = 10
        df = DataFrame(np.random.default_rng(2).standard_normal((n, 3)))
        df["dates1"] = date_range("1/1/2012", periods=n)
        df["dates2"] = date_range("1/1/2013", periods=n)
        df["dates3"] = date_range("1/1/2014", periods=n)
        df.loc[np.random.default_rng(2).random(n) > 0.5, "dates1"] = pd.NaT
        df.loc[np.random.default_rng(2).random(n) > 0.5, "dates3"] = pd.NaT
        res = df.query("dates1 < 20130101 < dates3", engine=engine, parser=parser)
        expec = df[(df.dates1 < "20130101") & ("20130101" < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_index_query(self, engine, parser):
        # [FUNCTION DOC]
        # Using the index name placeholder and a column in a chained date comparison
        # after moving a date column to the index in place (the in-place operation must
        # return nothing) equals the boolean selection.
        n = 10
        df = DataFrame(np.random.default_rng(2).standard_normal((n, 3)))
        df["dates1"] = date_range("1/1/2012", periods=n)
        df["dates3"] = date_range("1/1/2014", periods=n)
        return_value = df.set_index("dates1", inplace=True, drop=True)
        assert return_value is None
        res = df.query("index < 20130101 < dates3", engine=engine, parser=parser)
        expec = df[(df.index < "20130101") & ("20130101" < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_index_query_with_NaT(self, engine, parser):
        # [FUNCTION DOC]
        # As the date index query, but the first column is cast to object holding a
        # missing value in the first row to avoid implicit casting, with an in-place
        # index move returning nothing.
        n = 10
        # Cast to object to avoid implicit cast when setting entry to pd.NaT below
        df = DataFrame(np.random.default_rng(2).standard_normal((n, 3))).astype(
            {0: object}
        )
        df["dates1"] = date_range("1/1/2012", periods=n)
        df["dates3"] = date_range("1/1/2014", periods=n)
        df.iloc[0, 0] = pd.NaT
        return_value = df.set_index("dates1", inplace=True, drop=True)
        assert return_value is None
        res = df.query("index < 20130101 < dates3", engine=engine, parser=parser)
        expec = df[(df.index < "20130101") & ("20130101" < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_index_query_with_NaT_duplicates(self, engine, parser):
        # [FUNCTION DOC]
        # Date index query where the index contains missing values from random
        # assignment (duplicated missing labels) and the index is referenced by its own
        # name; result equals the selection using the index as a Series.
        n = 10
        d = {}
        d["dates1"] = date_range("1/1/2012", periods=n)
        d["dates3"] = date_range("1/1/2014", periods=n)
        df = DataFrame(d)
        df.loc[np.random.default_rng(2).random(n) > 0.5, "dates1"] = pd.NaT
        return_value = df.set_index("dates1", inplace=True, drop=True)
        assert return_value is None
        res = df.query("dates1 < 20130101 < dates3", engine=engine, parser=parser)
        expec = df[(df.index.to_series() < "20130101") & ("20130101" < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_query_with_non_date(self, engine, parser):
        # [FUNCTION DOC]
        # Parametrized by engine and parser. A datetime column compared with an
        # integer column: equality selects nothing, inequality selects every row, and
        # ordering comparisons raise a type error about invalid comparison between
        # datetime and array.
        n = 10
        df = DataFrame(
            {
                "dates": date_range("1/1/2012", periods=n, unit="ns"),
                "nondate": np.arange(n),
            }
        )

        result = df.query("dates == nondate", parser=parser, engine=engine)
        assert len(result) == 0

        result = df.query("dates != nondate", parser=parser, engine=engine)
        tm.assert_frame_equal(result, df)

        msg = r"Invalid comparison between dtype=datetime64\[ns\] and ndarray"
        for op in ["<", ">", "<=", ">="]:
            with pytest.raises(TypeError, match=msg):
                df.query(f"dates {op} nondate", parser=parser, engine=engine)

    def test_query_syntax_error(self, engine, parser):
        # [FUNCTION DOC]
        # A query that is syntactically invalid (an operator with a missing operand)
        # raises a syntax error mentioning invalid syntax; the frame has columns named
        # like operators to ensure they are not interpreted.
        df = DataFrame({"i": range(10), "+": range(3, 13), "r": range(4, 14)})
        msg = "invalid syntax"
        with pytest.raises(SyntaxError, match=msg):
            df.query("i - +", engine=engine, parser=parser)

    def test_query_scope(self, engine, parser):
        # [FUNCTION DOC]
        # Skipped for non-pandas parser. Plain names resolve to columns (column a
        # greater than column b) while a prefixed local name resolves to the local
        # variable; an undefined prefixed name raises an undefined-variable error saying
        # local variable is not defined, and an undefined unprefixed name raises one saying
        # name is not defined.
        skip_if_no_pandas_parser(parser)

        df = DataFrame(
            np.random.default_rng(2).standard_normal((20, 2)), columns=list("ab")
        )

        a, b = 1, 2  # noqa: F841
        res = df.query("a > b", engine=engine, parser=parser)
        expected = df[df.a > df.b]
        tm.assert_frame_equal(res, expected)

        res = df.query("@a > b", engine=engine, parser=parser)
        expected = df[a > df.b]
        tm.assert_frame_equal(res, expected)

        # no local variable c
        with pytest.raises(
            UndefinedVariableError, match="local variable 'c' is not defined"
        ):
            df.query("@a > b > @c", engine=engine, parser=parser)

        # no column named 'c'
        with pytest.raises(UndefinedVariableError, match="name 'c' is not defined"):
            df.query("@a > b > c", engine=engine, parser=parser)

    def test_query_doesnt_pickup_local(self, engine, parser):
        # [FUNCTION DOC]
        # Names of local functions (like sine) are not picked up by query; using one as
        # a bare name raises an undefined-variable error.
        n = m = 10
        df = DataFrame(
            np.random.default_rng(2).integers(m, size=(n, 3)), columns=list("abc")
        )

        # we don't pick up the local 'sin'
        with pytest.raises(UndefinedVariableError, match="name 'sin' is not defined"):
            df.query("sin > 5", engine=engine, parser=parser)

    def test_query_builtin(self, engine, parser):
        # [FUNCTION DOC]
        # When the index is named with the same name as a builtin math function and the
        # numexpr engine is relevant, query raises a clobbering error saying variables in
        # the expression overlap. (Overridden for the python-engine classes, where it
        # works.)
        n = m = 10
        df = DataFrame(
            np.random.default_rng(2).integers(m, size=(n, 3)), columns=list("abc")
        )

        df.index.name = "sin"
        msg = "Variables in expression.+"
        with pytest.raises(NumExprClobberingError, match=msg):
            df.query("sin > 5", engine=engine, parser=parser)

    def test_query(self, engine, parser):
        # [FUNCTION DOC]
        # Basic queries with comparison and arithmetic between columns equal the same
        # boolean selections, for the engine and parser.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)), columns=["a", "b", "c"]
        )

        tm.assert_frame_equal(
            df.query("a < b", engine=engine, parser=parser), df[df.a < df.b]
        )
        tm.assert_frame_equal(
            df.query("a + b > b * c", engine=engine, parser=parser),
            df[df.a + df.b > df.b * df.c],
        )

    def test_query_index_with_name(self, engine, parser):
        # [FUNCTION DOC]
        # A named index can be referenced in a query by its name, alone or combined
        # with column conditions, and equals the selection on the index values.
        df = DataFrame(
            np.random.default_rng(2).integers(10, size=(10, 3)),
            index=Index(range(10), name="blob"),
            columns=["a", "b", "c"],
        )
        res = df.query("(blob < 5) & (a < b)", engine=engine, parser=parser)
        expec = df[(df.index < 5) & (df.a < df.b)]
        tm.assert_frame_equal(res, expec)

        res = df.query("blob < b", engine=engine, parser=parser)
        expec = df[df.index < df.b]

        tm.assert_frame_equal(res, expec)

    def test_query_index_without_name(self, engine, parser):
        # [FUNCTION DOC]
        # Unnamed integer index referenced by the special name index in a query, compared
        # with a column and with a scalar.
        df = DataFrame(
            np.random.default_rng(2).integers(10, size=(10, 3)),
            index=range(10),
            columns=["a", "b", "c"],
        )

        # "index" should refer to the index
        res = df.query("index < b", engine=engine, parser=parser)
        expec = df[df.index < df.b]
        tm.assert_frame_equal(res, expec)

        # test against a scalar
        res = df.query("index < 5", engine=engine, parser=parser)
        expec = df[df.index < 5]
        tm.assert_frame_equal(res, expec)

    def test_nested_scope(self, engine, parser):
        # [FUNCTION DOC]
        # Skipped for non-pandas parser. Frames referenced with the local prefix inside
        # query, and referenced by name in the top-level eval with logical-and chains
        # (including a nested selection), equal the combined boolean selections, and eval
        # equals the query result.
        skip_if_no_pandas_parser(parser)

        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        df2 = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        expected = df[(df > 0) & (df2 > 0)]

        result = df.query("(@df > 0) & (@df2 > 0)", engine=engine, parser=parser)
        tm.assert_frame_equal(result, expected)

        result = pd.eval("df[df > 0 and df2 > 0]", engine=engine, parser=parser)
        tm.assert_frame_equal(result, expected)

        result = pd.eval(
            "df[df > 0 and df2 > 0 and df[df > 0] > 0]", engine=engine, parser=parser
        )
        expected = df[(df > 0) & (df2 > 0) & (df[df > 0] > 0)]
        tm.assert_frame_equal(result, expected)

        result = pd.eval("df[(df>0) & (df2>0)]", engine=engine, parser=parser)
        expected = df.query("(@df>0) & (@df2>0)", engine=engine, parser=parser)
        tm.assert_frame_equal(result, expected)

    def test_nested_raises_on_local_self_reference(self, engine, parser):
        # [FUNCTION DOC]
        # A frame cannot refer to itself by name in query without the local prefix:
        # raises an undefined-variable error saying the name is not defined.
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))

        # can't reference ourself b/c we're a local so @ is necessary
        with pytest.raises(UndefinedVariableError, match="name 'df' is not defined"):
            df.query("df > 0", engine=engine, parser=parser)

    def test_local_syntax(self, engine, parser):
        # [FUNCTION DOC]
        # Skipped for non-pandas parser. A local variable referenced with the prefix is
        # used in the comparison, and names without the prefix mean columns.
        skip_if_no_pandas_parser(parser)

        df = DataFrame(
            np.random.default_rng(2).standard_normal((100, 10)),
            columns=list("abcdefghij"),
        )
        b = 1
        expect = df[df.a < b]
        result = df.query("a < @b", engine=engine, parser=parser)
        tm.assert_frame_equal(result, expect)

        expect = df[df.a < df.b]
        result = df.query("a < b", engine=engine, parser=parser)
        tm.assert_frame_equal(result, expect)

    def test_chained_cmp_and_in(self, engine, parser):
        # [FUNCTION DOC]
        # Skipped for non-pandas parser. A chained comparison combined with membership
        # operators in query equals the explicit boolean combination of comparisons
        # and membership tests.
        skip_if_no_pandas_parser(parser)
        cols = list("abc")
        df = DataFrame(
            np.random.default_rng(2).standard_normal((100, len(cols))), columns=cols
        )
        res = df.query(
            "a < b < c and a not in b not in c", engine=engine, parser=parser
        )
        ind = (df.a < df.b) & (df.b < df.c) & ~df.b.isin(df.a) & ~df.c.isin(df.b)
        expec = df[ind]
        tm.assert_frame_equal(res, expec)

    def test_local_variable_with_in(self, engine, parser):
        # [FUNCTION DOC]
        # Skipped for non-pandas parser. Arithmetic on a column inside a membership test
        # works, both with a column and with a prefixed local Series that shadows the
        # column name.
        skip_if_no_pandas_parser(parser)
        a = Series(np.random.default_rng(2).integers(3, size=15), name="a")
        b = Series(np.random.default_rng(2).integers(10, size=15), name="b")
        df = DataFrame({"a": a, "b": b})

        expected = df.loc[(df.b - 1).isin(a)]
        result = df.query("b - 1 in a", engine=engine, parser=parser)
        tm.assert_frame_equal(expected, result)

        b = Series(np.random.default_rng(2).integers(10, size=15), name="b")
        expected = df.loc[(b - 1).isin(a)]
        result = df.query("@b - 1 in a", engine=engine, parser=parser)
        tm.assert_frame_equal(expected, result)

    def test_at_inside_string(self, engine, parser):
        # [FUNCTION DOC]
        # Skipped for non-pandas parser. A prefix character appearing inside a quoted
        # string literal is not treated as a local variable reference.
        skip_if_no_pandas_parser(parser)
        c = 1  # noqa: F841
        df = DataFrame({"a": ["a", "a", "b", "b", "@c", "@c"]})
        result = df.query('a == "@c"', engine=engine, parser=parser)
        expected = df[df.a == "@c"]
        tm.assert_frame_equal(result, expected)

    def test_query_undefined_local(self):
        # [FUNCTION DOC]
        # Intended: with a prefix on an undefined name the query raises an
        # undefined-variable error about the local variable.
        # KNOWN ISSUE: the engine and parser are read from the test class as attributes, but in this file they are fixture methods, so the values are bound-method objects, not names; the parser check then always skips the test, so the assertions never run.
        engine, parser = self.engine, self.parser
        skip_if_no_pandas_parser(parser)

        df = DataFrame(np.random.default_rng(2).random((10, 2)), columns=list("ab"))
        with pytest.raises(
            UndefinedVariableError, match="local variable 'c' is not defined"
        ):
            df.query("a == @c", engine=engine, parser=parser)

    def test_index_resolvers_come_after_columns_with_the_same_name(
        self, engine, parser
    ):
        # [FUNCTION DOC]
        # Name resolution order: a column named index takes precedence over the index
        # resolver (even if the index has the same name); the positional placeholder
        # works for the index; and with an index named like a column, the column wins
        # and the special name index still refers to the index.
        n = 1  # noqa: F841
        a = np.r_[20:101:20]

        df = DataFrame(
            {"index": a, "b": np.random.default_rng(2).standard_normal(a.size)}
        )
        df.index.name = "index"
        result = df.query("index > 5", engine=engine, parser=parser)
        expected = df[df["index"] > 5]
        tm.assert_frame_equal(result, expected)

        df = DataFrame(
            {"index": a, "b": np.random.default_rng(2).standard_normal(a.size)}
        )
        result = df.query("ilevel_0 > 5", engine=engine, parser=parser)
        expected = df.loc[df.index[df.index > 5]]
        tm.assert_frame_equal(result, expected)

        df = DataFrame({"a": a, "b": np.random.default_rng(2).standard_normal(a.size)})
        df.index.name = "a"
        result = df.query("a > 5", engine=engine, parser=parser)
        expected = df[df.a > 5]
        tm.assert_frame_equal(result, expected)

        result = df.query("index > 5", engine=engine, parser=parser)
        expected = df.loc[df.index[df.index > 5]]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("op, f", [["==", operator.eq], ["!=", operator.ne]])
    def test_inf(self, op, f, engine, parser):
        # [FUNCTION DOC]
        # Parametrized over equality and inequality (operator text and function) and
        # engine and parser. Compares a query for equality with infinity against the
        # boolean selection.
        # KNOWN ISSUE: the assignment of infinity targets a row slice and new column labeled zero (not column a), so column a never contains infinity and the test does not exercise infinite values.
        n = 10
        df = DataFrame(
            {
                "a": np.random.default_rng(2).random(n),
                "b": np.random.default_rng(2).random(n),
            }
        )
        df.loc[::2, 0] = np.inf
        q = f"a {op} inf"
        expected = df[f(df.a, np.inf)]
        result = df.query(q, engine=engine, parser=parser)
        tm.assert_frame_equal(result, expected)

    def test_check_tz_aware_index_query(self, tz_aware_fixture):
        # [FUNCTION DOC]
        # GH 29463: query on a frame with a timezone-aware named datetime index (for
        # each timezone fixture) comparing against a UTC string literal returns all rows,
        # both while the datetime is the index and after resetting it into a column.
        # https://github.com/pandas-dev/pandas/issues/29463
        tz = tz_aware_fixture
        df_index = date_range(
            start="2019-01-01", freq="1D", periods=10, tz=tz, name="time"
        )
        expected = DataFrame(index=df_index)
        df = DataFrame(index=df_index)
        result = df.query('"2018-01-03 00:00:00+00" < time')
        tm.assert_frame_equal(result, expected)

        expected = DataFrame(df_index)
        result = df.reset_index().query('"2018-01-03 00:00:00+00" < time')
        tm.assert_frame_equal(result, expected)

    def test_method_calls_in_query(self, engine, parser):
        # [FUNCTION DOC]
        # GH 22435: method calls inside query strings work, e.g. converting a column
        # to integer or testing a column for non-missing values, for each engine and
        # parser.
        # https://github.com/pandas-dev/pandas/issues/22435
        n = 10
        df = DataFrame(
            {
                "a": 2 * np.random.default_rng(2).random(n),
                "b": np.random.default_rng(2).random(n),
            }
        )
        expected = df[df["a"].astype("int") == 0]
        result = df.query("a.astype('int') == 0", engine=engine, parser=parser)
        tm.assert_frame_equal(result, expected)

        df = DataFrame(
            {
                "a": np.where(
                    np.random.default_rng(2).random(n) < 0.5,
                    np.nan,
                    np.random.default_rng(2).standard_normal(n),
                ),
                "b": np.random.default_rng(2).standard_normal(n),
            }
        )
        expected = df[df["a"].notnull()]
        result = df.query("a.notnull()", engine=engine, parser=parser)
        tm.assert_frame_equal(result, expected)


@td.skip_if_no("numexpr")
class TestDataFrameQueryNumExprPython(TestDataFrameQueryNumExprPandas):
    # [CLASS DOC]
    # Variant of the date and syntax tests for the numexpr engine with the python
    # parser (skipped without numexpr). Inherits the tests from the pandas-parser
    # class and overrides those that behave differently: logical-and rather than
    # chained comparison, the unsupported nested scope prefix, and chained
    # comparisons with missing index values.
    @pytest.fixture
    def engine(self):
        # [FUNCTION DOC]
        # HELPER: fixture overriding the engine to numexpr.
        return "numexpr"

    @pytest.fixture
    def parser(self):
        # [FUNCTION DOC]
        # HELPER: fixture overriding the parser to python.
        return "python"

    def test_date_query_no_attribute_access(self, engine, parser):
        # [FUNCTION DOC]
        # Same as the inherited test but written with parenthesized logical-and since
        # the python parser lacks chained comparisons.
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        df["dates1"] = date_range("1/1/2012", periods=5)
        df["dates2"] = date_range("1/1/2013", periods=5)
        df["dates3"] = date_range("1/1/2014", periods=5)
        res = df.query(
            "(dates1 < 20130101) & (20130101 < dates3)", engine=engine, parser=parser
        )
        expec = df[(df.dates1 < "20130101") & ("20130101" < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_query_with_NaT(self, engine, parser):
        # [FUNCTION DOC]
        # Date columns with missing values, using parenthesized logical-and; equals the
        # boolean selection.
        n = 10
        df = DataFrame(np.random.default_rng(2).standard_normal((n, 3)))
        df["dates1"] = date_range("1/1/2012", periods=n)
        df["dates2"] = date_range("1/1/2013", periods=n)
        df["dates3"] = date_range("1/1/2014", periods=n)
        df.loc[np.random.default_rng(2).random(n) > 0.5, "dates1"] = pd.NaT
        df.loc[np.random.default_rng(2).random(n) > 0.5, "dates3"] = pd.NaT
        res = df.query(
            "(dates1 < 20130101) & (20130101 < dates3)", engine=engine, parser=parser
        )
        expec = df[(df.dates1 < "20130101") & ("20130101" < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_index_query(self, engine, parser):
        # [FUNCTION DOC]
        # Index-based date query using parenthesized logical-and, after moving a
        # column into the index in place (return value must be nothing).
        n = 10
        df = DataFrame(np.random.default_rng(2).standard_normal((n, 3)))
        df["dates1"] = date_range("1/1/2012", periods=n)
        df["dates3"] = date_range("1/1/2014", periods=n)
        return_value = df.set_index("dates1", inplace=True, drop=True)
        assert return_value is None
        res = df.query(
            "(index < 20130101) & (20130101 < dates3)", engine=engine, parser=parser
        )
        expec = df[(df.index < "20130101") & ("20130101" < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_index_query_with_NaT(self, engine, parser):
        # [FUNCTION DOC]
        # As the index date query, with an object-cast first column holding a missing
        # value, and parenthesized logical-and.
        n = 10
        # Cast to object to avoid implicit cast when setting entry to pd.NaT below
        df = DataFrame(np.random.default_rng(2).standard_normal((n, 3))).astype(
            {0: object}
        )
        df["dates1"] = date_range("1/1/2012", periods=n)
        df["dates3"] = date_range("1/1/2014", periods=n)
        df.iloc[0, 0] = pd.NaT
        return_value = df.set_index("dates1", inplace=True, drop=True)
        assert return_value is None
        res = df.query(
            "(index < 20130101) & (20130101 < dates3)", engine=engine, parser=parser
        )
        expec = df[(df.index < "20130101") & ("20130101" < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_index_query_with_NaT_duplicates(self, engine, parser):
        # [FUNCTION DOC]
        # Index with missing and duplicated values: the chained comparison form
        # (not supported by this parser) raises not-implemented mentioning the boolean
        # operation node.
        n = 10
        df = DataFrame(np.random.default_rng(2).standard_normal((n, 3)))
        df["dates1"] = date_range("1/1/2012", periods=n)
        df["dates3"] = date_range("1/1/2014", periods=n)
        df.loc[np.random.default_rng(2).random(n) > 0.5, "dates1"] = pd.NaT
        return_value = df.set_index("dates1", inplace=True, drop=True)
        assert return_value is None
        msg = r"'BoolOp' nodes are not implemented"
        with pytest.raises(NotImplementedError, match=msg):
            df.query("index < 20130101 < dates3", engine=engine, parser=parser)

    def test_nested_scope(self, engine, parser):
        # [FUNCTION DOC]
        # Smoke test that eval sees a plain local variable. The prefix syntax in query
        # raises a syntax error (only the pandas parser supports it); referring to frames
        # by bare name in query raises an undefined-variable error; the top-level eval
        # resolves frames by name and equals the combined boolean selections.
        # smoke test
        x = 1  # noqa: F841
        result = pd.eval("x + 1", engine=engine, parser=parser)
        assert result == 2

        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        df2 = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))

        # don't have the pandas parser
        msg = r"The '@' prefix is only supported by the pandas parser"
        with pytest.raises(SyntaxError, match=msg):
            df.query("(@df>0) & (@df2>0)", engine=engine, parser=parser)

        with pytest.raises(UndefinedVariableError, match="name 'df' is not defined"):
            df.query("(df>0) & (df2>0)", engine=engine, parser=parser)

        expected = df[(df > 0) & (df2 > 0)]
        result = pd.eval("df[(df > 0) & (df2 > 0)]", engine=engine, parser=parser)
        tm.assert_frame_equal(expected, result)

        expected = df[(df > 0) & (df2 > 0) & (df[df > 0] > 0)]
        result = pd.eval(
            "df[(df > 0) & (df2 > 0) & (df[df > 0] > 0)]", engine=engine, parser=parser
        )
        tm.assert_frame_equal(expected, result)

    def test_query_numexpr_with_min_and_max_columns(self):
        # [FUNCTION DOC]
        # Columns named min and max overlap builtin names: querying for equality to
        # one raises a clobbering error, whose message names the expression and
        # the overlapping builtin.
        df = DataFrame({"min": [1, 2, 3], "max": [4, 5, 6]})
        regex_to_match = (
            r"Variables in expression \"\(min\) == \(1\)\" "
            r"overlap with builtins: \('min'\)"
        )
        with pytest.raises(NumExprClobberingError, match=regex_to_match):
            df.query("min == 1")

        regex_to_match = (
            r"Variables in expression \"\(max\) == \(1\)\" "
            r"overlap with builtins: \('max'\)"
        )
        with pytest.raises(NumExprClobberingError, match=regex_to_match):
            df.query("max == 1")


class TestDataFrameQueryPythonPandas(TestDataFrameQueryNumExprPandas):
    # [CLASS DOC]
    # Variant for the python engine and pandas parser. Inherits the tests of the
    # numexpr/pandas class and overrides the builtin-name test, since the python
    # engine tolerates it.
    @pytest.fixture
    def engine(self):
        # [FUNCTION DOC]
        # HELPER: fixture overriding the engine to python.
        return "python"

    @pytest.fixture
    def parser(self):
        # [FUNCTION DOC]
        # HELPER: fixture overriding the parser to pandas.
        return "pandas"

    def test_query_builtin(self, engine, parser):
        # [FUNCTION DOC]
        # With an index named like a builtin function, the python engine does not raise
        # and the query equals the selection of rows whose index exceeds five.
        n = m = 10
        df = DataFrame(
            np.random.default_rng(2).integers(m, size=(n, 3)), columns=list("abc")
        )

        df.index.name = "sin"
        expected = df[df.index > 5]
        result = df.query("sin > 5", engine=engine, parser=parser)
        tm.assert_frame_equal(expected, result)


class TestDataFrameQueryPythonPython(TestDataFrameQueryNumExprPython):
    # [CLASS DOC]
    # Variant for the python engine and python parser. Inherits from the
    # numexpr/python class and overrides the builtin-name test, since the python
    # engine tolerates it.
    @pytest.fixture
    def engine(self):
        # [FUNCTION DOC]
        # HELPER: fixture overriding the engine to python.
        return "python"

    @pytest.fixture
    def parser(self):
        # [FUNCTION DOC]
        # HELPER: fixture overriding the parser to python.
        return "python"

    def test_query_builtin(self, engine, parser):
        # [FUNCTION DOC]
        # With an index named like a builtin function the python engine does not raise
        # and equals the selection of rows whose index exceeds five.
        n = m = 10
        df = DataFrame(
            np.random.default_rng(2).integers(m, size=(n, 3)), columns=list("abc")
        )

        df.index.name = "sin"
        expected = df[df.index > 5]
        result = df.query("sin > 5", engine=engine, parser=parser)
        tm.assert_frame_equal(expected, result)


class TestDataFrameQueryStrings:
    # [CLASS DOC]
    # Test group for string comparisons and membership in query, parametrized by
    # the module fixtures for parser and engine.
    def test_str_query_method(self, parser, engine):
        # [FUNCTION DOC]
        # String column equality and inequality in query: with the python parser each
        # combination of column and quoted string literal in either order raises a
        # not-implemented error about the membership node types; with the pandas parser
        # the results equal both the direct selection and the membership selection.
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 1)), columns=["b"])
        df["strings"] = Series(list("aabbccddee"))
        expect = df[df.strings == "a"]

        if parser != "pandas":
            col = "strings"
            lst = '"a"'

            lhs = [col] * 2 + [lst] * 2
            rhs = lhs[::-1]

            eq, ne = "==", "!="
            ops = 2 * ([eq, ne])
            msg = r"'(Not)?In' nodes are not implemented"

            for lh, op_, rh in zip(lhs, ops, rhs):
                ex = f"{lh} {op_} {rh}"
                with pytest.raises(NotImplementedError, match=msg):
                    df.query(
                        ex,
                        engine=engine,
                        parser=parser,
                        local_dict={"strings": df.strings},
                    )
        else:
            res = df.query('"a" == strings', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)

            res = df.query('strings == "a"', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)
            tm.assert_frame_equal(res, df[df.strings.isin(["a"])])

            expect = df[df.strings != "a"]
            res = df.query('strings != "a"', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)

            res = df.query('"a" != strings', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)
            tm.assert_frame_equal(res, df[~df.strings.isin(["a"])])

    def test_str_list_query_method(self, parser, engine):
        # [FUNCTION DOC]
        # List comparisons: with the python parser, comparing the string column to a
        # list literal in either order for equality and inequality raises not-implemented;
        # with the pandas parser the results equal membership selections.
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 1)), columns=["b"])
        df["strings"] = Series(list("aabbccddee"))
        expect = df[df.strings.isin(["a", "b"])]

        if parser != "pandas":
            col = "strings"
            lst = '["a", "b"]'

            lhs = [col] * 2 + [lst] * 2
            rhs = lhs[::-1]

            eq, ne = "==", "!="
            ops = 2 * ([eq, ne])
            msg = r"'(Not)?In' nodes are not implemented"

            for lh, ops_, rh in zip(lhs, ops, rhs):
                ex = f"{lh} {ops_} {rh}"
                with pytest.raises(NotImplementedError, match=msg):
                    df.query(ex, engine=engine, parser=parser)
        else:
            res = df.query('strings == ["a", "b"]', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)

            res = df.query('["a", "b"] == strings', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)

            expect = df[~df.strings.isin(["a", "b"])]

            res = df.query('strings != ["a", "b"]', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)

            res = df.query('["a", "b"] != strings', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)

    def test_query_with_string_columns(self, parser, engine):
        # [FUNCTION DOC]
        # With the pandas parser, membership of one string column in another and the
        # same combined with a numeric comparison equal the membership-based boolean
        # selections. With the python parser both raise not-implemented errors (membership
        # nodes, then boolean operation nodes).
        df = DataFrame(
            {
                "a": list("aaaabbbbcccc"),
                "b": list("aabbccddeeff"),
                "c": np.random.default_rng(2).integers(5, size=12),
                "d": np.random.default_rng(2).integers(9, size=12),
            }
        )
        if parser == "pandas":
            res = df.query("a in b", parser=parser, engine=engine)
            expec = df[df.a.isin(df.b)]
            tm.assert_frame_equal(res, expec)

            res = df.query("a in b and c < d", parser=parser, engine=engine)
            expec = df[df.a.isin(df.b) & (df.c < df.d)]
            tm.assert_frame_equal(res, expec)
        else:
            msg = r"'(Not)?In' nodes are not implemented"
            with pytest.raises(NotImplementedError, match=msg):
                df.query("a in b", parser=parser, engine=engine)

            msg = r"'BoolOp' nodes are not implemented"
            with pytest.raises(NotImplementedError, match=msg):
                df.query("a in b and c < d", parser=parser, engine=engine)

    def test_object_array_eq_ne(self, parser, engine):
        # [FUNCTION DOC]
        # Equality and inequality between two object (string) columns in query equal the
        # direct column comparisons.
        df = DataFrame(
            {
                "a": list("aaaabbbbcccc"),
                "b": list("aabbccddeeff"),
                "c": np.random.default_rng(2).integers(5, size=12),
                "d": np.random.default_rng(2).integers(9, size=12),
            }
        )
        res = df.query("a == b", parser=parser, engine=engine)
        exp = df[df.a == df.b]
        tm.assert_frame_equal(res, exp)

        res = df.query("a != b", parser=parser, engine=engine)
        exp = df[df.a != df.b]
        tm.assert_frame_equal(res, exp)

    def test_query_with_nested_strings(self, parser, engine):
        # [FUNCTION DOC]
        # Skipped for non-pandas parser. A query whose string literal itself contains
        # quotes, tested with membership in an event column, selects the rows whose
        # event equals the quoted text.
        skip_if_no_pandas_parser(parser)
        events = [
            f"page {n} {act}" for n in range(1, 4) for act in ["load", "exit"]
        ] * 2
        stamps1 = date_range("2014-01-01 0:00:01", freq="30s", periods=6)
        stamps2 = date_range("2014-02-01 1:00:01", freq="30s", periods=6)
        df = DataFrame(
            {
                "id": np.arange(1, 7).repeat(2),
                "event": events,
                "timestamp": stamps1.append(stamps2),
            }
        )

        expected = df[df.event == '"page 1 load"']
        res = df.query("""'"page 1 load"' in event""", parser=parser, engine=engine)
        tm.assert_frame_equal(expected, res)

    def test_query_with_nested_special_character(self, parser, engine):
        # [FUNCTION DOC]
        # Skipped for non-pandas parser. A string literal containing an ampersand inside
        # quotes is treated as a string, not an operator, and selects the matching row.
        skip_if_no_pandas_parser(parser)
        df = DataFrame({"a": ["a", "b", "test & test"], "b": [1, 2, 3]})
        res = df.query('a == "test & test"', parser=parser, engine=engine)
        expec = df[df.a == "test & test"]
        tm.assert_frame_equal(res, expec)

    @pytest.mark.parametrize(
        "op, func",
        [
            ["<", operator.lt],
            [">", operator.gt],
            ["<=", operator.le],
            [">=", operator.ge],
        ],
    )
    def test_query_lex_compare_strings(self, parser, engine, op, func):
        # [FUNCTION DOC]
        # Parametrized over the four ordering operators (text and function). Comparing a
        # string column lexicographically with a quoted string literal in query equals the
        # function applied to the column and the string.
        a = Series(np.random.default_rng(2).choice(list("abcde"), 20))
        b = Series(np.arange(a.size))
        df = DataFrame({"X": a, "Y": b})

        res = df.query(f'X {op} "d"', engine=engine, parser=parser)
        expected = df[func(df.X, "d")]
        tm.assert_frame_equal(res, expected)

    def test_query_single_element_booleans(self, parser, engine):
        # [FUNCTION DOC]
        # A query using bitwise-and of two boolean columns on a one-row frame equals the
        # boolean selection.
        columns = "bid", "bidsize", "ask", "asksize"
        data = np.random.default_rng(2).integers(2, size=(1, len(columns))).astype(bool)
        df = DataFrame(data, columns=columns)
        res = df.query("bid & ask", engine=engine, parser=parser)
        expected = df[df.bid & df.ask]
        tm.assert_frame_equal(res, expected)

    def test_query_string_scalar_variable(self, parser, engine):
        # [FUNCTION DOC]
        # Skipped for non-pandas parser. A local string variable with the prefix is
        # compared to a string column and selects the same rows as the literal.
        skip_if_no_pandas_parser(parser)
        df = DataFrame(
            {
                "Symbol": ["BUD US", "BUD US", "IBM US", "IBM US"],
                "Price": [109.70, 109.72, 183.30, 183.35],
            }
        )
        e = df[df.Symbol == "BUD US"]
        symb = "BUD US"  # noqa: F841
        r = df.query("Symbol == @symb", parser=parser, engine=engine)
        tm.assert_frame_equal(e, r)

    @pytest.mark.parametrize(
        "in_list",
        [
            [None, "asdf", "ghjk"],
            ["asdf", None, "ghjk"],
            ["asdf", "ghjk", None],
            [None, None, "asdf"],
            ["asdf", None, None],
            [None, None, None],
        ],
    )
    def test_query_string_null_elements(self, in_list):
        # [FUNCTION DOC]
        # GITHUB ISSUE #31516: parametrized over lists with missing values mixed with
        # strings. For a nullable string column and the pandas parser with python
        # engine, equality query, boolean selection and a less-or-equal query all give
        # the same rows as the expected frame (only rows equal to the string, with
        # the index kept as 64-bit integers).
        # GITHUB ISSUE #31516
        parser = "pandas"
        engine = "python"
        expected = {i: value for i, value in enumerate(in_list) if value == "asdf"}

        df_expected = DataFrame({"a": expected}, dtype="string")
        df_expected.index = df_expected.index.astype("int64")
        df = DataFrame({"a": in_list}, dtype="string")
        df.index = Index(list(df.index), dtype=df.index.dtype)
        res1 = df.query("a == 'asdf'", parser=parser, engine=engine)
        res2 = df[df["a"] == "asdf"]
        res3 = df.query("a <= 'asdf'", parser=parser, engine=engine)
        tm.assert_frame_equal(res1, df_expected)
        tm.assert_frame_equal(res1, res2)
        tm.assert_frame_equal(res1, res3)
        tm.assert_frame_equal(res2, res3)


class TestDataFrameEvalWithFrame:
    # [CLASS DOC]
    # Test group for frame eval with a random three-column frame, using the module
    # fixtures for engine and parser.
    @pytest.fixture
    def frame(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a 10x3 random normal frame with columns a, b, c.
        return DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)), columns=list("abc")
        )

    def test_simple_expr(self, frame, parser, engine):
        # [FUNCTION DOC]
        # Evaluating the sum of two columns equals the sum of those columns.
        res = frame.eval("a + b", engine=engine, parser=parser)
        expect = frame.a + frame.b
        tm.assert_series_equal(res, expect)

    def test_bool_arith_expr(self, frame, parser, engine):
        # [FUNCTION DOC]
        # Evaluating a column filtered by a boolean mask inside the expression, plus
        # another column, equals the same computation done directly (alignment fills
        # missing where filtered).
        res = frame.eval("a[a < 1] + b", engine=engine, parser=parser)
        expect = frame.a[frame.a < 1] + frame.b
        tm.assert_series_equal(res, expect)

    @pytest.mark.parametrize("op", ["+", "-", "*", "/"])
    def test_invalid_type_for_operator_raises(self, parser, engine, op):
        # [FUNCTION DOC]
        # Parametrized over the four arithmetic operator texts. Evaluating an
        # operation between an integer column and a string column raises a type error
        # (message about unsupported operand types, or a cannot-type message).
        df = DataFrame({"a": [1, 2], "b": ["c", "d"]})
        msg = r"unsupported operand type\(s\) for .+: '.+' and '.+'|Cannot"

        with pytest.raises(TypeError, match=msg):
            df.eval(f"a {op} b", engine=engine, parser=parser)


class TestDataFrameQueryBacktickQuoting:
    # [CLASS DOC]
    # Test group for backtick quoting of column names in query and eval: names with
    # spaces, punctuation, keywords, numbers, unicode, quotes, hash characters, nested
    # quotes and unmatched delimiters, and for calls of non-named expressions and
    # extension dtypes. Uses the shared wide-label frame fixture.
    @pytest.fixture
    def df(self):
        # [FUNCTION DOC]
        # HELPER: fixture for the backtick group. Returns a three-row frame with many
        # columns whose labels exercise quoting (spaces, repeated spaces, underscores,
        # dots, dashes, a number-like text, a keyword, parentheses, symbols,
        # the empty string, quotes, unicode, a hash, and an integer label one).
        # KNOWN ISSUE: the docstring says the last two columns cannot be escaped and should raise a value error, but the last columns are the hash-containing label and an integer label that tests quote successfully; the docstring is stale.
        """
        Yields a dataframe with strings that may or may not need escaping
        by backticks. The last two columns cannot be escaped by backticks
        and should raise a ValueError.
        """
        return DataFrame(
            {
                "A": [1, 2, 3],
                "B B": [3, 2, 1],
                "C C": [4, 5, 6],
                "C  C": [7, 4, 3],
                "C_C": [8, 9, 10],
                "D_D D": [11, 1, 101],
                "E.E": [6, 3, 5],
                "F-F": [8, 1, 10],
                "1e1": [2, 4, 8],
                "def": [10, 11, 2],
                "A (x)": [4, 1, 3],
                "B(x)": [1, 1, 5],
                "B (x)": [2, 7, 4],
                "  &^ :!€$?(} >    <++*''  ": [2, 5, 6],
                "": [10, 11, 1],
                " A": [4, 7, 9],
                "  ": [1, 2, 1],
                "it's": [6, 3, 1],
                "that's": [9, 1, 8],
                "☺": [8, 7, 6],
                "xy （z）": [1, 2, 3],  # noqa: RUF001
                "xy （z\\uff09": [4, 5, 6],  # noqa: RUF001
                "foo#bar": [2, 4, 5],
                1: [5, 7, 9],
            }
        )

    def test_single_backtick_variable_query(self, df):
        # [FUNCTION DOC]
        # Query with one backtick-quoted name containing a space, compared with a
        # literal, equals the boolean selection.
        res = df.query("1 < `B B`")
        expect = df[1 < df["B B"]]
        tm.assert_frame_equal(res, expect)

    def test_two_backtick_variables_query(self, df):
        # [FUNCTION DOC]
        # Query combining two backtick-quoted names with spaces equals the combined
        # boolean selection.
        res = df.query("1 < `B B` and 4 < `C C`")
        expect = df[(1 < df["B B"]) & (4 < df["C C"])]
        tm.assert_frame_equal(res, expect)

    def test_single_backtick_variable_expr(self, df):
        # [FUNCTION DOC]
        # Eval of a plain column plus a backtick-quoted spaced column equals the sum of
        # the columns.
        res = df.eval("A + `B B`")
        expect = df["A"] + df["B B"]
        tm.assert_series_equal(res, expect)

    def test_two_backtick_variables_expr(self, df):
        # [FUNCTION DOC]
        # Eval of the sum of two backtick-quoted spaced columns equals their sum.
        res = df.eval("`B B` + `C C`")
        expect = df["B B"] + df["C C"]
        tm.assert_series_equal(res, expect)

    def test_already_underscore_variable(self, df):
        # [FUNCTION DOC]
        # A column name already containing an underscore, quoted by backticks, can be
        # used with a plain column.
        res = df.eval("`C_C` + A")
        expect = df["C_C"] + df["A"]
        tm.assert_series_equal(res, expect)

    def test_same_name_but_underscores(self, df):
        # [FUNCTION DOC]
        # A plain name with underscores and a quoted spaced name with the same letters
        # refer to different columns and both evaluate correctly.
        res = df.eval("C_C + `C C`")
        expect = df["C_C"] + df["C C"]
        tm.assert_series_equal(res, expect)

    def test_mixed_underscores_and_spaces(self, df):
        # [FUNCTION DOC]
        # A column label mixing underscores and a space, quoted, evaluates correctly.
        res = df.eval("A + `D_D D`")
        expect = df["A"] + df["D_D D"]
        tm.assert_series_equal(res, expect)

    def test_backtick_quote_name_with_no_spaces(self, df):
        # [FUNCTION DOC]
        # Backtick quoting a name that needs no quoting works.
        res = df.eval("A + `C_C`")
        expect = df["A"] + df["C_C"]
        tm.assert_series_equal(res, expect)

    def test_special_characters(self, df):
        # [FUNCTION DOC]
        # Names with a dot and a dash, quoted, can be combined arithmetically with
        # another column.
        res = df.eval("`E.E` + `F-F` - A")
        expect = df["E.E"] + df["F-F"] - df["A"]
        tm.assert_series_equal(res, expect)

    def test_start_with_digit(self, df):
        # [FUNCTION DOC]
        # A column label that starts with a digit (looks like a number in scientific
        # notation), quoted, works.
        res = df.eval("A + `1e1`")
        expect = df["A"] + df["1e1"]
        tm.assert_series_equal(res, expect)

    def test_keyword(self, df):
        # [FUNCTION DOC]
        # A column labeled with a reserved word, quoted, works.
        res = df.eval("A + `def`")
        expect = df["A"] + df["def"]
        tm.assert_series_equal(res, expect)

    def test_unneeded_quoting(self, df):
        # [FUNCTION DOC]
        # Query quoting a plain identifier unnecessarily still works.
        res = df.query("`A` > 2")
        expect = df[df["A"] > 2]
        tm.assert_frame_equal(res, expect)

    def test_parenthesis(self, df):
        # [FUNCTION DOC]
        # A label with parentheses and a space, quoted, works in a query.
        res = df.query("`A (x)` > 2")
        expect = df[df["A (x)"] > 2]
        tm.assert_frame_equal(res, expect)

    def test_empty_string(self, df):
        # [FUNCTION DOC]
        # The empty string as a label, quoted by two adjacent backticks, works in a
        # query.
        res = df.query("`` > 5")
        expect = df[df[""] > 5]
        tm.assert_frame_equal(res, expect)

    def test_multiple_spaces(self, df):
        # [FUNCTION DOC]
        # A label with two consecutive spaces, quoted, selects the correct column
        # (distinct from the single-space label).
        res = df.query("`C  C` > 5")
        expect = df[df["C  C"] > 5]
        tm.assert_frame_equal(res, expect)

    def test_start_with_spaces(self, df):
        # [FUNCTION DOC]
        # Labels that start with a space and that consist only of spaces, quoted, work in
        # eval.
        res = df.eval("` A` + `  `")
        expect = df[" A"] + df["  "]
        tm.assert_series_equal(res, expect)

    def test_ints(self, df):
        # [FUNCTION DOC]
        # A backtick-quoted integer-looking name refers to the integer label, not a
        # string label.
        res = df.query("`1` == 7")
        expect = df[df[1] == 7]
        tm.assert_frame_equal(res, expect)

    def test_lots_of_operators_string(self, df):
        # [FUNCTION DOC]
        # A label filled with operator-like symbols and quotes, quoted, works in a query.
        res = df.query("`  &^ :!€$?(} >    <++*''  ` > 4")
        expect = df[df["  &^ :!€$?(} >    <++*''  "] > 4]
        tm.assert_frame_equal(res, expect)

    def test_missing_attribute(self, df):
        # [FUNCTION DOC]
        # Evaluating a reference to a nonexistent attribute of the pandas module via the
        # local prefix raises an attribute error saying the module has no such attribute.
        message = "module 'pandas' has no attribute 'thing'"
        with pytest.raises(AttributeError, match=message):
            df.eval("@pd.thing")

    def test_quote(self, df):
        # [FUNCTION DOC]
        # Labels containing apostrophes, quoted by backticks, can be compared.
        res = df.query("`it's` > `that's`")
        expect = df[df["it's"] > df["that's"]]
        tm.assert_frame_equal(res, expect)

    def test_character_outside_range_smiley(self, df):
        # [FUNCTION DOC]
        # A label that is a non-ASCII symbol (smiley), quoted, works.
        res = df.query("`☺` > 4")
        expect = df[df["☺"] > 4]
        tm.assert_frame_equal(res, expect)

    def test_character_outside_range_2_byte_parens(self, df):
        # [FUNCTION DOC]
        # GH 49633: a label containing wide (full-width) parentheses, quoted, works.
        # GH 49633
        res = df.query("`xy （z）` == 2")  # noqa: RUF001
        expect = df[df["xy （z）"] == 2]  # noqa: RUF001
        tm.assert_frame_equal(res, expect)

    def test_character_outside_range_and_actual_backslash(self, df):
        # [FUNCTION DOC]
        # GH 49633: a label with a wide parenthesis and a literal backslash-text, quoted,
        # matches the column whose label has the actual wide parenthesis character and
        # the literal text.
        # GH 49633
        res = df.query("`xy （z\\uff09` == 2")  # noqa: RUF001
        expect = df[df["xy \uff08z\\uff09"] == 2]
        tm.assert_frame_equal(res, expect)

    def test_hashtag(self, df):
        # [FUNCTION DOC]
        # A label containing a hash character, quoted, is not treated as a comment.
        res = df.query("`foo#bar` > 4")
        expect = df[df["foo#bar"] > 4]
        tm.assert_frame_equal(res, expect)

    def test_expr_with_column_name_with_hashtag_character(self):
        # [FUNCTION DOC]
        # GH 59285: a column named with a trailing hash, quoted, works in a query.
        # GH 59285
        df = DataFrame((1, 2, 3), columns=["a#"])
        result = df.query("`a#` < 2")
        expected = df[df["a#"] < 2]
        tm.assert_frame_equal(result, expected)

    def test_expr_with_comment(self):
        # [FUNCTION DOC]
        # GH 59285: a trailing comment after a query expression is ignored while a
        # quoted column name with a hash is respected.
        # GH 59285
        df = DataFrame((1, 2, 3), columns=["a#"])
        result = df.query("`a#` < 2  # This is a comment")
        expected = df[df["a#"] < 2]
        tm.assert_frame_equal(result, expected)

    def test_expr_with_column_name_with_backtick_and_hash(self):
        # [FUNCTION DOC]
        # GH 59285: a column name containing a backtick and a hash is quoted by doubling
        # the backtick, and works.
        # GH 59285
        df = DataFrame((1, 2, 3), columns=["a`#b"])
        result = df.query("`a``#b` < 2")
        expected = df[df["a`#b"] < 2]
        tm.assert_frame_equal(result, expected)

    def test_expr_with_column_name_with_backtick(self):
        # [FUNCTION DOC]
        # GH 59285: a column name with a backtick is referenced by doubling the
        # backtick, selecting the correct rows (a comment in the test notes a formatting
        # check may misread the inline code).
        # GH 59285
        df = DataFrame({"a`b": (1, 2, 3), "ab": (4, 5, 6)})
        result = df.query("`a``b` < 2")
        # Note: Formatting checks may wrongly consider the above ``inline code``.
        expected = df[df["a`b"] < 2]
        tm.assert_frame_equal(result, expected)

    def test_expr_with_string_with_backticks(self):
        # [FUNCTION DOC]
        # GH 59285: a string literal made of backticks, compared with a column named with
        # a hash, is not treated as a quoted name.
        # GH 59285
        df = DataFrame(("`", "`````", "``````````"), columns=["#backticks"])
        result = df.query("'```' < `#backticks`")
        expected = df["```" < df["#backticks"]]
        tm.assert_frame_equal(result, expected)

    def test_expr_with_string_with_backticked_substring_same_as_column_name(self):
        # [FUNCTION DOC]
        # GH 59285: a string literal that textually looks like a quoted column name is
        # treated as a string while the real quoted name is the column.
        # GH 59285
        df = DataFrame(("`", "`````", "``````````"), columns=["#backticks"])
        result = df.query("'`#backticks`' < `#backticks`")
        expected = df["`#backticks`" < df["#backticks"]]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "col1,col2,expr",
        [
            ("it's", "that's", "`it's` < `that's`"),
            ('it"s', 'that"s', '`it"s` < `that"s`'),
            ("it's", 'that\'s "nice"', "`it's` < `that's \"nice\"`"),
            ("it's", "that's #cool", "`it's` < `that's #cool` # This is a comment"),
        ],
    )
    def test_expr_with_column_names_with_special_characters(self, col1, col2, expr):
        # [FUNCTION DOC]
        # GH 59285: parametrized over pairs of column names with apostrophes, double
        # quotes, mixed quotes and a hash, plus the query text (the last has a trailing
        # comment). Query with both names backtick-quoted equals the boolean selection.
        # GH 59285
        df = DataFrame(
            [
                {col1: 1, col2: 2},
                {col1: 3, col2: 4},
                {col1: -1, col2: -2},
                {col1: -3, col2: -4},
            ]
        )
        result = df.query(expr)
        expected = df[df[col1] < df[col2]]
        tm.assert_frame_equal(result, expected)

    def test_expr_with_no_backticks(self):
        # [FUNCTION DOC]
        # GH 59285: with no backticks, a string literal on the left of a comparison with
        # a plain column works.
        # GH 59285
        df = DataFrame(("aaa", "vvv", "zzz"), columns=["column_name"])
        result = df.query("'value' < column_name")
        expected = df["value" < df["column_name"]]
        tm.assert_frame_equal(result, expected)

    def test_expr_with_no_quotes_and_backtick_is_unmatched(self):
        # [FUNCTION DOC]
        # GH 59285: an unmatched opening backtick (no closing) raises a syntax or token
        # error mentioning invalid syntax.
        # GH 59285
        df = DataFrame((1, 5, 10), columns=["column-name"])
        with pytest.raises((SyntaxError, TokenError), match="invalid syntax"):
            df.query("5 < `column-name")

    def test_expr_with_no_quotes_and_backtick_is_matched(self):
        # [FUNCTION DOC]
        # GH 59285: a correctly closed backtick-quoted name containing a dash works.
        # GH 59285
        df = DataFrame((1, 5, 10), columns=["column-name"])
        result = df.query("5 < `column-name`")
        expected = df[5 < df["column-name"]]
        tm.assert_frame_equal(result, expected)

    def test_expr_with_backtick_opened_before_quote_and_backtick_is_unmatched(self):
        # [FUNCTION DOC]
        # GH 59285: a backtick opened before an apostrophe and never closed raises a
        # syntax or token error saying unterminated string literal.
        # GH 59285
        df = DataFrame((1, 5, 10), columns=["It's"])
        with pytest.raises(
            (SyntaxError, TokenError), match="unterminated string literal"
        ):
            df.query("5 < `It's")

    def test_expr_with_backtick_opened_before_quote_and_backtick_is_matched(self):
        # [FUNCTION DOC]
        # GH 59285: a backtick-quoted name containing an apostrophe, correctly closed,
        # works.
        # GH 59285
        df = DataFrame((1, 5, 10), columns=["It's"])
        result = df.query("5 < `It's`")
        expected = df[5 < df["It's"]]
        tm.assert_frame_equal(result, expected)

    def test_expr_with_quote_opened_before_backtick_and_quote_is_unmatched(self):
        # [FUNCTION DOC]
        # GH 59285: a quote opened before a backtick and never closed raises a syntax or
        # token error saying unterminated string literal.
        # GH 59285
        df = DataFrame(("aaa", "vvv", "zzz"), columns=["column-name"])
        with pytest.raises(
            (SyntaxError, TokenError), match="unterminated string literal"
        ):
            df.query("`column-name` < 'It`s that\\'s \"quote\" #hash")

    def test_expr_with_quote_opened_before_backtick_and_quote_is_matched_at_end(self):
        # [FUNCTION DOC]
        # GH 59285: a quoted string literal containing a backtick, an escaped quote,
        # double quotes and a hash is correctly closed at the end of the expression.
        # GH 59285
        df = DataFrame(("aaa", "vvv", "zzz"), columns=["column-name"])
        result = df.query("`column-name` < 'It`s that\\'s \"quote\" #hash'")
        expected = df[df["column-name"] < 'It`s that\'s "quote" #hash']
        tm.assert_frame_equal(result, expected)

    def test_expr_with_quote_opened_before_backtick_and_quote_is_matched_in_mid(self):
        # [FUNCTION DOC]
        # GH 59285: the same complex string literal in the middle of an expression
        # (before a comparison with a quoted column) is parsed correctly.
        # GH 59285
        df = DataFrame(("aaa", "vvv", "zzz"), columns=["column-name"])
        result = df.query("'It`s that\\'s \"quote\" #hash' < `column-name`")
        expected = df['It`s that\'s "quote" #hash' < df["column-name"]]
        tm.assert_frame_equal(result, expected)

    def test_call_non_named_expression(self, df):
        # [FUNCTION DOC]
        # Only named functions (attributes and variables) can be called: calling a
        # local function by name works; calling an element of a list (or the call
        # method of it) raises a type error saying only named functions are supported
        # (https://github.com/pandas-dev/pandas/pull/32460).
        """
        Only attributes and variables ('named functions') can be called.
        .__call__() is not an allowed attribute because that would allow
        calling anything.
        https://github.com/pandas-dev/pandas/pull/32460
        """

        def func(*_):
            # [FUNCTION DOC]
            # HELPER: for the call test. Local function accepting any positional arguments
            # and returning the integer one.
            return 1

        funcs = [func]  # noqa: F841

        df.eval("@func()")

        with pytest.raises(TypeError, match="Only named functions are supported"):
            df.eval("@funcs[0]()")

        with pytest.raises(TypeError, match="Only named functions are supported"):
            df.eval("@funcs[0].__call__()")

    def test_ea_dtypes(self, any_numeric_ea_and_arrow_dtype):
        # [FUNCTION DOC]
        # GH#29618: for each numeric extension and Arrow dtype fixture, an assignment eval
        # subtracting two columns returns the frame with the new column, same dtype;
        # a runtime warning is expected exactly when numexpr is installed.
        # GH#29618
        df = DataFrame(
            [[1, 2], [3, 4]], columns=["a", "b"], dtype=any_numeric_ea_and_arrow_dtype
        )
        warning = RuntimeWarning if NUMEXPR_INSTALLED else None
        with tm.assert_produces_warning(warning):
            result = df.eval("c = b - a")
        expected = DataFrame(
            [[1, 2, 1], [3, 4, 1]],
            columns=["a", "b", "c"],
            dtype=any_numeric_ea_and_arrow_dtype,
        )
        tm.assert_frame_equal(result, expected)

    def test_ea_dtypes_and_scalar(self):
        # [FUNCTION DOC]
        # GH#29618: assignment eval with a scalar operand on nullable float columns gives
        # the expected frame; a runtime warning is expected exactly when numexpr is
        # installed.
        # GH#29618
        df = DataFrame([[1, 2], [3, 4]], columns=["a", "b"], dtype="Float64")
        warning = RuntimeWarning if NUMEXPR_INSTALLED else None
        with tm.assert_produces_warning(warning):
            result = df.eval("c = b - 1")
        expected = DataFrame(
            [[1, 2, 1], [3, 4, 3]], columns=["a", "b", "c"], dtype="Float64"
        )
        tm.assert_frame_equal(result, expected)

    def test_ea_dtypes_and_scalar_operation(self, any_numeric_ea_and_arrow_dtype):
        # [FUNCTION DOC]
        # GH#29618: an assignment eval of a constant arithmetic expression on a frame of
        # extension dtype adds a column of ones; the new column takes the dtype of the
        # result, and the others keep the original dtype.
        # GH#29618
        df = DataFrame(
            [[1, 2], [3, 4]], columns=["a", "b"], dtype=any_numeric_ea_and_arrow_dtype
        )
        result = df.eval("c = 2 - 1")
        expected = DataFrame(
            {
                "a": Series([1, 3], dtype=any_numeric_ea_and_arrow_dtype),
                "b": Series([2, 4], dtype=any_numeric_ea_and_arrow_dtype),
                "c": Series([1, 1], dtype=result["c"].dtype),
            }
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["int64", "Int64", "int64[pyarrow]"])
    def test_query_ea_dtypes(self, dtype):
        # [FUNCTION DOC]
        # GH#50261: parametrized over three integer dtypes (plain, nullable, Arrow-backed;
        # the Arrow one skipped without pyarrow). A membership query against a local set
        # selects the matching row, with a runtime warning expected only for the
        # nullable dtype when numexpr is installed.
        if dtype == "int64[pyarrow]":
            pytest.importorskip("pyarrow")
        # GH#50261
        df = DataFrame({"a": [1, 2]}, dtype=dtype)
        ref = {2}  # noqa: F841
        warning = RuntimeWarning if dtype == "Int64" and NUMEXPR_INSTALLED else None
        with tm.assert_produces_warning(warning):
            result = df.query("a in @ref")
        expected = DataFrame({"a": [2]}, index=range(1, 2), dtype=dtype)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("engine", ["python", "numexpr"])
    @pytest.mark.parametrize("dtype", ["int64", "Int64", "int64[pyarrow]"])
    def test_query_ea_equality_comparison(self, dtype, engine):
        # [FUNCTION DOC]
        # GH#50261: parametrized over engine and integer dtype. Comparing a nullable
        # column with a column of the given dtype for equality via query returns the rows
        # where they are equal, keeping the dtypes; numexpr runs expect a runtime
        # warning and are skipped if numexpr is missing; the Arrow dtype is skipped
        # without pyarrow.
        # GH#50261
        warning = RuntimeWarning if engine == "numexpr" else None
        if engine == "numexpr" and not NUMEXPR_INSTALLED:
            pytest.skip("numexpr not installed")
        if dtype == "int64[pyarrow]":
            pytest.importorskip("pyarrow")
        df = DataFrame(
            {"A": Series([1, 1, 2], dtype="Int64"), "B": Series([1, 2, 2], dtype=dtype)}
        )
        with tm.assert_produces_warning(warning):
            result = df.query("A == B", engine=engine)
        expected = DataFrame(
            {
                "A": Series([1, 2], dtype="Int64", index=range(0, 4, 2)),
                "B": Series([1, 2], dtype=dtype, index=range(0, 4, 2)),
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_all_nat_in_object(self):
        # [FUNCTION DOC]
        # GH#57068: a query comparing an object column of all-missing datetimes to a
        # local timestamp returns an empty object-dtype frame.
        # GH#57068
        now = pd.Timestamp.now("UTC")  # noqa: F841
        df = DataFrame({"a": pd.to_datetime([None, None], utc=True)}, dtype=object)
        result = df.query("a > @now")
        expected = DataFrame({"a": []}, dtype=object)
        tm.assert_frame_equal(result, expected)
