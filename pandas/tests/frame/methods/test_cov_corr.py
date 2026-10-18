# [MODULE DOC]
# Tests for the covariance and correlation methods of the frame class: pairwise
# covariance (cov), pairwise correlation (corr) with pearson, kendall, spearman and
# callable methods, and correlation against another object (corrwith). Covers
# min_periods, ddof, numeric_only, missing values, nullable and arrow dtypes, constant
# and tiny-valued columns, duplicate columns and numerical stability. Several tests
# require the scientific library for rank-based methods. Uses frame fixtures, optional
# dependency helpers and the testing helpers module. No import-time side effects.

import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    date_range,
    isna,
)
import pandas._testing as tm


class TestDataFrameCov:
    # [CLASS DOC]
    # Stateless pytest class of covariance tests.
    def test_cov(self, float_frame, float_string_frame):
        # [FUNCTION DOC]
        # Using the float frame: min_periods equal to the length equals the default;
        # larger than the length gives all missing; with missing values in two columns
        # the pair with too few overlapping observations is missing; an entry equals the
        # series covariance; the mixed string frame raises a value error unless
        # numeric_only is set (then equals covariance of the numeric columns);
        # single-column frames match numpy covariance, including with a missing row.
        # min_periods no NAs (corner case)
        expected = float_frame.cov()
        result = float_frame.cov(min_periods=len(float_frame))

        tm.assert_frame_equal(expected, result)

        result = float_frame.cov(min_periods=len(float_frame) + 1)
        assert isna(result.values).all()

        # with NAs
        frame = float_frame.copy()
        frame.iloc[:5, frame.columns.get_loc("A")] = np.nan
        frame.iloc[5:10, frame.columns.get_loc("B")] = np.nan
        result = frame.cov(min_periods=len(frame) - 8)
        expected = frame.cov()
        expected.loc["A", "B"] = np.nan
        expected.loc["B", "A"] = np.nan
        tm.assert_frame_equal(result, expected)

        # regular
        result = frame.cov()
        expected = frame["A"].cov(frame["C"])
        tm.assert_almost_equal(result["A"]["C"], expected)

        # fails on non-numeric types
        with pytest.raises(ValueError, match="could not convert string to float"):
            float_string_frame.cov()
        result = float_string_frame.cov(numeric_only=True)
        expected = float_string_frame.loc[:, ["A", "B", "C", "D"]].cov()
        tm.assert_frame_equal(result, expected)

        # Single column frame
        df = DataFrame(np.linspace(0.0, 1.0, 10))
        result = df.cov()
        expected = DataFrame(
            np.cov(df.values.T).reshape((1, 1)), index=df.columns, columns=df.columns
        )
        tm.assert_frame_equal(result, expected)
        df.loc[0] = np.nan
        result = df.cov()
        expected = DataFrame(
            np.cov(df.values[1:].T).reshape((1, 1)),
            index=df.columns,
            columns=df.columns,
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("test_ddof", [None, 0, 1, 2, 3])
    def test_cov_ddof(self, test_ddof):
        # [FUNCTION DOC]
        # Parametrized over ddof values (None, 0, 1, 2, 3). Result equals numpy
        # covariance with the same ddof for a two-column random frame.
        # GH#34611
        np_array1 = np.random.default_rng(2).random(10)
        np_array2 = np.random.default_rng(2).random(10)
        df = DataFrame({0: np_array1, 1: np_array2})
        result = df.cov(ddof=test_ddof)
        expected_np = np.cov(np_array1, np_array2, ddof=test_ddof)
        expected = DataFrame(expected_np)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "other_column", [pd.array([1, 2, 3]), np.array([1.0, 2.0, 3.0])]
    )
    def test_cov_nullable_integer(self, other_column):
        # [FUNCTION DOC]
        # Parametrized over a nullable integer array and a float array as second column.
        # A nullable integer column with a missing value yields the pairwise covariance
        # matrix computed on available pairs.
        # https://github.com/pandas-dev/pandas/issues/33803
        data = DataFrame({"a": pd.array([1, 2, None]), "b": other_column})
        result = data.cov()
        arr = np.array([[0.5, 0.5], [0.5, 1.0]])
        expected = DataFrame(arr, columns=["a", "b"], index=["a", "b"])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("numeric_only", [True, False])
    def test_cov_numeric_only(self, numeric_only):
        # [FUNCTION DOC]
        # Parametrized over numeric_only. With a string column present, numeric_only
        # true gives the covariance of the numeric column; false raises a value error
        # about converting string to float.
        # when dtypes of pandas series are different
        # then ndarray will have dtype=object,
        # so it need to be properly handled
        df = DataFrame({"a": [1, 0], "c": ["x", "y"]})
        expected = DataFrame(0.5, index=["a"], columns=["a"])
        if numeric_only:
            result = df.cov(numeric_only=numeric_only)
            tm.assert_frame_equal(result, expected)
        else:
            with pytest.raises(ValueError, match="could not convert string to float"):
                df.cov(numeric_only=numeric_only)


class TestDataFrameCorr:
    # [CLASS DOC]
    # Tests of the pairwise correlation method (as opposed to corrwith); stateless
    # pytest class.
    # DataFrame.corr(), as opposed to DataFrame.corrwith

    @pytest.mark.parametrize("method", ["pearson", "kendall", "spearman"])
    def test_corr_scipy_method(self, float_frame, method):
        # [FUNCTION DOC]
        # Parametrized over three methods; skipped without scipy. Sets some entries
        # missing in the float frame (mutating the fixture), then checks the matrix
        # entry for two columns equals the series-level correlation with the same
        # method.
        pytest.importorskip("scipy")
        float_frame.loc[float_frame.index[:5], "A"] = np.nan
        float_frame.loc[float_frame.index[5:10], "B"] = np.nan
        float_frame.loc[float_frame.index[:10], "A"] = float_frame["A"][10:20].copy()

        correls = float_frame.corr(method=method)
        expected = float_frame["A"].corr(float_frame["C"], method=method)
        tm.assert_almost_equal(correls["A"]["C"], expected)

    # ---------------------------------------------------------------------

    def test_corr_non_numeric(self, float_string_frame):
        # [FUNCTION DOC]
        # Mixed string frame raises a value error on correlation unless numeric_only is
        # set, in which case the numeric columns are used.
        with pytest.raises(ValueError, match="could not convert string to float"):
            float_string_frame.corr()
        result = float_string_frame.corr(numeric_only=True)
        expected = float_string_frame.loc[:, ["A", "B", "C", "D"]].corr()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("meth", ["pearson", "kendall", "spearman"])
    def test_corr_nooverlap(self, meth):
        # [FUNCTION DOC]
        # Parametrized over methods; needs scipy. Columns with no overlapping
        # observations give missing correlation; self-correlation is one unless the
        # column is all missing.
        # nothing in common
        pytest.importorskip("scipy")
        df = DataFrame(
            {
                "A": [1, 1.5, 1, np.nan, np.nan, np.nan],
                "B": [np.nan, np.nan, np.nan, 1, 1.5, 1],
                "C": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )
        rs = df.corr(meth)
        assert isna(rs.loc["A", "B"])
        assert isna(rs.loc["B", "A"])
        assert rs.loc["A", "A"] == 1
        assert rs.loc["B", "B"] == 1
        assert isna(rs.loc["C", "C"])

    @pytest.mark.parametrize("meth", ["pearson", "spearman"])
    def test_corr_constant(self, meth):
        # [FUNCTION DOC]
        # Parametrized over pearson and spearman. Columns that are constant (and do not
        # overlap) give an all-missing matrix.
        # constant --> all NA
        df = DataFrame(
            {
                "A": [1, 1, 1, np.nan, np.nan, np.nan],
                "B": [np.nan, np.nan, np.nan, 1, 1, 1],
            }
        )
        rs = df.corr(meth)
        assert isna(rs.values).all()

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.parametrize("meth", ["pearson", "kendall", "spearman"])
    def test_corr_int_and_boolean(self, meth):
        # [FUNCTION DOC]
        # Parametrized over methods; needs scipy; ignores runtime warnings. A boolean
        # column and an integer column with the same pattern correlate perfectly: the
        # result is a matrix of ones.
        # when dtypes of pandas series are different
        # then ndarray will have dtype=object,
        # so it need to be properly handled
        pytest.importorskip("scipy")
        df = DataFrame({"a": [True, False], "b": [1, 0]})

        expected = DataFrame(np.ones((2, 2)), index=["a", "b"], columns=["a", "b"])
        result = df.corr(meth)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("method", ["cov", "corr"])
    def test_corr_cov_independent_index_column(self, method):
        # [FUNCTION DOC]
        # Parametrized over cov and corr. The result's index and columns are equal but
        # must not be the same object.
        # GH#14617
        df = DataFrame(
            np.random.default_rng(2).standard_normal(4 * 10).reshape(10, 4),
            columns=list("abcd"),
        )
        result = getattr(df, method)()
        assert result.index is not result.columns
        assert result.index.equals(result.columns)

    def test_corr_invalid_method(self):
        # [FUNCTION DOC]
        # An unknown method name raises a value error listing the allowed methods
        # (pearson, spearman, kendall, or a callable).
        # GH#22298
        df = DataFrame(np.random.default_rng(2).normal(size=(10, 2)))
        msg = "method must be either 'pearson', 'spearman', 'kendall', or a callable, "
        with pytest.raises(ValueError, match=msg):
            df.corr(method="____")

    def test_corr_int(self):
        # [FUNCTION DOC]
        # Smoke test: covariance and correlation on integer-only columns do not fail. No
        # assertions.
        # dtypes other than float64 GH#1761
        df = DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]})

        df.cov()
        df.corr()

    @pytest.mark.parametrize(
        "nullable_column", [pd.array([1, 2, 3]), pd.array([1, 2, None])]
    )
    @pytest.mark.parametrize(
        "other_column",
        [pd.array([1, 2, 3]), np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, np.nan])],
    )
    @pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
    def test_corr_nullable_integer(self, nullable_column, other_column, method):
        # [FUNCTION DOC]
        # Parametrized over nullable integer arrays (with and without missing), other
        # columns and three methods; needs scipy. Result is a two by two matrix of ones.
        # https://github.com/pandas-dev/pandas/issues/33803
        pytest.importorskip("scipy")
        data = DataFrame({"a": nullable_column, "b": other_column})
        result = data.corr(method=method)
        expected = DataFrame(np.ones((2, 2)), columns=["a", "b"], index=["a", "b"])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("length", [2, 20, 200, 2000])
    def test_corr_for_constant_columns(self, length):
        # [FUNCTION DOC]
        # Parametrized over several lengths. A frame with constant columns gives an
        # all-missing correlation matrix regardless of length.
        # GH: 37448
        df = DataFrame(length * [[0.4, 0.1]], columns=["A", "B"])
        result = df.corr()
        expected = DataFrame(
            {"A": [np.nan, np.nan], "B": [np.nan, np.nan]}, index=["A", "B"]
        )
        tm.assert_frame_equal(result, expected)

    def test_calc_corr_small_numbers(self):
        # [FUNCTION DOC]
        # Very small values (order of 1e-20) still correlate perfectly (matrix of ones).
        # GH: 37452
        df = DataFrame(
            {"A": [1.0e-20, 2.0e-20, 3.0e-20], "B": [1.0e-20, 2.0e-20, 3.0e-20]}
        )
        result = df.corr()
        expected = DataFrame({"A": [1.0, 1.0], "B": [1.0, 1.0]}, index=["A", "B"])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
    def test_corr_min_periods_greater_than_length(self, method):
        # [FUNCTION DOC]
        # Parametrized over methods; needs scipy. A min_periods larger than the number
        # of rows gives all missing.
        pytest.importorskip("scipy")
        df = DataFrame({"A": [1, 2], "B": [1, 2]})
        result = df.corr(method=method, min_periods=3)
        expected = DataFrame(
            {"A": [np.nan, np.nan], "B": [np.nan, np.nan]}, index=["A", "B"]
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("meth", ["pearson", "kendall", "spearman"])
    @pytest.mark.parametrize("numeric_only", [True, False])
    def test_corr_numeric_only(self, meth, numeric_only):
        # [FUNCTION DOC]
        # Parametrized over methods and numeric_only; needs scipy. With a string column,
        # numeric_only true gives a ones matrix for the numeric columns; false raises a
        # value error.
        # when dtypes of pandas series are different
        # then ndarray will have dtype=object,
        # so it need to be properly handled
        pytest.importorskip("scipy")
        df = DataFrame({"a": [1, 0], "b": [1, 0], "c": ["x", "y"]})
        expected = DataFrame(np.ones((2, 2)), index=["a", "b"], columns=["a", "b"])
        if numeric_only:
            result = df.corr(meth, numeric_only=numeric_only)
            tm.assert_frame_equal(result, expected)
        else:
            with pytest.raises(ValueError, match="could not convert string to float"):
                df.corr(meth, numeric_only=numeric_only)


class TestDataFrameCorrWith:
    # [CLASS DOC]
    # Tests of correlation against another frame or series (corrwith); stateless pytest
    # class.
    @pytest.mark.parametrize(
        "dtype",
        [
            "float64",
            "Float64",
            pytest.param("float64[pyarrow]", marks=td.skip_if_no("pyarrow")),
        ],
    )
    def test_corrwith(self, datetime_frame, dtype):
        # [FUNCTION DOC]
        # Parametrized over numpy float, nullable float and arrow float dtypes. Compares
        # correlation with a noisy, reordered, partially overlapping frame along columns
        # and rows (rows equal the transposed computation), and checks that drop removes
        # non-shared labels.
        datetime_frame = datetime_frame.astype(dtype)

        a = datetime_frame
        noise = Series(np.random.default_rng(2).standard_normal(len(a)), index=a.index)

        b = datetime_frame.add(noise, axis=0)

        # make sure order does not matter
        b = b.reindex(columns=b.columns[::-1], index=b.index[::-1][len(a) // 2 :])
        del b["B"]

        colcorr = a.corrwith(b, axis=0)
        tm.assert_almost_equal(colcorr["A"], a["A"].corr(b["A"]))

        rowcorr = a.corrwith(b, axis=1)
        tm.assert_series_equal(rowcorr, a.T.corrwith(b.T, axis=0))

        dropped = a.corrwith(b, axis=0, drop=True)
        tm.assert_almost_equal(dropped["A"], a["A"].corr(b["A"]))
        assert "B" not in dropped

        dropped = a.corrwith(b, axis=1, drop=True)
        assert a.index[-1] not in dropped.index

    def test_corrwith_non_timeseries_data(self):
        # [FUNCTION DOC]
        # Row-wise correlation between two frames with string row labels, one missing a
        # row; each shared row equals the series-level correlation.
        index = ["a", "b", "c", "d", "e"]
        columns = ["one", "two", "three", "four"]
        df1 = DataFrame(
            np.random.default_rng(2).standard_normal((5, 4)),
            index=index,
            columns=columns,
        )
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=index[:4],
            columns=columns,
        )
        correls = df1.corrwith(df2, axis=1)
        for row in index[:4]:
            tm.assert_almost_equal(correls[row], df1.loc[row].corr(df2.loc[row]))

    def test_corrwith_with_objects(self, using_infer_string):
        # [FUNCTION DOC]
        # Takes the inferred-string fixture. With an object/string column present,
        # correlation raises a type error (message depends on string inference) unless
        # numeric_only is set; row-wise correlation raises a type error about
        # unsupported operand; numeric_only variants equal the computation on the
        # numeric columns only.
        df1 = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        df2 = df1.copy()
        cols = ["A", "B", "C", "D"]

        df1["obj"] = "foo"
        df2["obj"] = "bar"

        if using_infer_string:
            msg = "Cannot perform reduction 'mean' with string dtype"
            with pytest.raises(TypeError, match=msg):
                df1.corrwith(df2)
        else:
            with pytest.raises(TypeError, match="Could not convert"):
                df1.corrwith(df2)
        result = df1.corrwith(df2, numeric_only=True)
        expected = df1.loc[:, cols].corrwith(df2.loc[:, cols])
        tm.assert_series_equal(result, expected)

        with pytest.raises(TypeError, match="unsupported operand type"):
            df1.corrwith(df2, axis=1)
        result = df1.corrwith(df2, axis=1, numeric_only=True)
        expected = df1.loc[:, cols].corrwith(df2.loc[:, cols], axis=1)
        tm.assert_series_equal(result, expected)

    def test_corrwith_series(self, datetime_frame):
        # [FUNCTION DOC]
        # Correlating a frame with one of its columns equals applying the series
        # correlation per column.
        result = datetime_frame.corrwith(datetime_frame["A"])
        expected = datetime_frame.apply(datetime_frame["A"].corr)

        tm.assert_series_equal(result, expected)

    def test_corrwith_matches_corrcoef(self):
        # [FUNCTION DOC]
        # Result equals numpy correlation for a linear versus quadratic sequence and is
        # below one.
        df1 = DataFrame(np.arange(100), columns=["a"])
        df2 = DataFrame(np.arange(100) ** 2, columns=["a"])
        c1 = df1.corrwith(df2)["a"]
        c2 = np.corrcoef(df1["a"], df2["a"])[0][1]

        tm.assert_almost_equal(c1, c2)
        assert c1 < 1

    @pytest.mark.parametrize("numeric_only", [True, False])
    def test_corrwith_mixed_dtypes(self, numeric_only):
        # [FUNCTION DOC]
        # Parametrized over numeric_only. With a string column, true restricts to
        # numeric columns; false raises a value error.
        # GH#18570
        df = DataFrame(
            {"a": [1, 4, 3, 2], "b": [4, 6, 7, 3], "c": ["a", "b", "c", "d"]}
        )
        s = Series([0, 6, 7, 3])
        if numeric_only:
            result = df.corrwith(s, numeric_only=numeric_only)
            corrs = [df["a"].corr(s), df["b"].corr(s)]
            expected = Series(data=corrs, index=["a", "b"])
            tm.assert_series_equal(result, expected)
        else:
            with pytest.raises(
                ValueError,
                match="could not convert string to float",
            ):
                df.corrwith(s, numeric_only=numeric_only)

    def test_corrwith_index_intersection(self):
        # [FUNCTION DOC]
        # With drop true, the result index is the intersection of the column labels.
        df1 = DataFrame(
            np.random.default_rng(2).random(size=(10, 2)), columns=["a", "b"]
        )
        df2 = DataFrame(
            np.random.default_rng(2).random(size=(10, 3)), columns=["a", "b", "c"]
        )

        result = df1.corrwith(df2, drop=True).index.sort_values()
        expected = df1.columns.intersection(df2.columns).sort_values()
        tm.assert_index_equal(result, expected)

    def test_corrwith_index_union(self):
        # [FUNCTION DOC]
        # With drop false, the result index is the union of the column labels.
        df1 = DataFrame(
            np.random.default_rng(2).random(size=(10, 2)), columns=["a", "b"]
        )
        df2 = DataFrame(
            np.random.default_rng(2).random(size=(10, 3)), columns=["a", "b", "c"]
        )

        result = df1.corrwith(df2, drop=False).index.sort_values()
        expected = df1.columns.union(df2.columns).sort_values()
        tm.assert_index_equal(result, expected)

    def test_corrwith_dup_cols(self):
        # [FUNCTION DOC]
        # Duplicated labels in the other frame produce one result per column, ones for
        # identical columns, with the duplicated label repeated.
        # GH#21925
        df1 = DataFrame(np.vstack([np.arange(10)] * 3).T)
        df2 = df1.copy()
        df2 = pd.concat((df2, df2[0]), axis=1)

        result = df1.corrwith(df2)
        expected = Series(np.ones(4), index=[0, 0, 1, 2])
        tm.assert_series_equal(result, expected)

    def test_corr_numerical_instabilities(self):
        # [FUNCTION DOC]
        # Correlation of two anti-correlated columns is within a tiny tolerance of the
        # exact values (compared after subtracting one to allow absolute tolerance).
        # GH#45640
        df = DataFrame([[0.2, 0.4], [0.4, 0.2]])
        result = df.corr()
        expected = DataFrame({0: [1.0, -1.0], 1: [-1.0, 1.0]})
        tm.assert_frame_equal(result - 1, expected - 1, atol=1e-17)

    def test_corrwith_spearman(self):
        # [FUNCTION DOC]
        # Needs scipy. Spearman rank correlation of data with a monotone transform of
        # itself is one for every column.
        # GH#21925
        pytest.importorskip("scipy")
        df = DataFrame(np.random.default_rng(2).random(size=(100, 3)))
        result = df.corrwith(df**2, method="spearman")
        expected = Series(np.ones(len(result)))
        tm.assert_series_equal(result, expected)

    def test_corrwith_kendall(self):
        # [FUNCTION DOC]
        # Needs scipy. Same for Kendall.
        # GH#21925
        pytest.importorskip("scipy")
        df = DataFrame(np.random.default_rng(2).random(size=(100, 3)))
        result = df.corrwith(df**2, method="kendall")
        expected = Series(np.ones(len(result)))
        tm.assert_series_equal(result, expected)

    def test_corrwith_spearman_with_tied_data(self):
        # [FUNCTION DOC]
        # Needs scipy. Spearman correlation with ties against a boolean column gives the
        # expected values; boolean frame against boolean series gives about 0.577 for
        # each column.
        # GH#48826
        pytest.importorskip("scipy")
        df1 = DataFrame(
            {
                "A": [1, np.nan, 7, 8],
                "B": [False, True, True, False],
                "C": [10, 4, 9, 3],
            }
        )
        df2 = df1[["B", "C"]]
        result = (df1 + 1).corrwith(df2.B, method="spearman")
        expected = Series([0.0, 1.0, 0.0], index=["A", "B", "C"])
        tm.assert_series_equal(result, expected)

        df_bool = DataFrame(
            {"A": [True, True, False, False], "B": [True, False, False, True]}
        )
        ser_bool = Series([True, True, False, True])
        result = df_bool.corrwith(ser_bool)
        expected = Series([0.57735, 0.57735], index=["A", "B"])
        tm.assert_series_equal(result, expected)

    def test_corrwith_min_periods_method(self):
        # [FUNCTION DOC]
        # Needs scipy. Passing min_periods together with spearman gives the same
        # expected values as without it.
        # GH#9490
        pytest.importorskip("scipy")
        df1 = DataFrame(
            {
                "A": [1, np.nan, 7, 8],
                "B": [False, True, True, False],
                "C": [10, 4, 9, 3],
            }
        )
        df2 = df1[["B", "C"]]
        result = (df1 + 1).corrwith(df2.B, method="spearman", min_periods=2)
        expected = Series([0.0, 1.0, 0.0], index=["A", "B", "C"])
        tm.assert_series_equal(result, expected)

    def test_corrwith_min_periods_boolean(self):
        # [FUNCTION DOC]
        # Boolean frame against a boolean series with min_periods three gives the
        # expected correlation values.
        # GH#9490
        df_bool = DataFrame(
            {"A": [True, True, False, False], "B": [True, False, False, True]}
        )
        ser_bool = Series([True, True, False, True])
        result = df_bool.corrwith(ser_bool, min_periods=3)
        expected = Series([0.57735, 0.57735], index=["A", "B"])
        tm.assert_series_equal(result, expected)

    def test_corr_within_bounds(self):
        # [FUNCTION DOC]
        # Correlation values are clipped within minus one and one: a near-perfect pair
        # yields exactly one at the maximum, and a random matrix has all entries within
        # bounds.
        df1 = DataFrame({"x": [0, 1], "y": [1.35951, 1.3595100000000007]})
        result1 = df1.corr().max().max()
        expected1 = 1.0
        tm.assert_equal(result1, expected1)

        rng = np.random.default_rng(seed=42)
        df2 = DataFrame(rng.random((100, 4)))
        corr_matrix = df2.corr()
        assert corr_matrix.min().min() >= -1.0
        assert corr_matrix.max().max() <= 1.0

    def test_cov_with_missing_values(self):
        # [FUNCTION DOC]
        # Covariance of columns with a missing row equals the covariance of the frame
        # with that row dropped.
        df = DataFrame({"A": [1, 2, None, 4], "B": [2, 4, None, 9]})
        expected = DataFrame(
            {"A": [2.333333, 5.500000], "B": [5.5, 13.0]}, index=["A", "B"]
        )
        result1 = df.cov()
        result2 = df.dropna().cov()
        tm.assert_frame_equal(result1, expected)
        tm.assert_frame_equal(result2, expected)
