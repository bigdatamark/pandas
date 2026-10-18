# [MODULE DOC]
# Tests for expanding windows: constructor validation, empty frames, minimum
# periods behavior, count, iteration, standard error, numerical stability of
# skew and kurtosis, rank and unique count, agreement with equivalent fixed
# rolling windows (correlation, covariance, quantile, pairwise), apply with
# engines and arguments, zero-length inputs, differently indexed operands,
# first and last, and numeric-only handling. Uses frame_or_series, series,
# frame, engine_and_raw, ascending, arithmetic_win_operators and numeric_only
# fixtures.
import numpy as np
import pytest

from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    isna,
    notna,
)
import pandas._testing as tm


def test_doc_string():
    # [FUNCTION DOC]
    # Smoke test of the documentation example: builds a one-column frame with a
    # missing value and computes the expanding sum with minimum periods 2. Asserts
    # nothing; passes if no exception is raised.
    df = DataFrame({"B": [0, 1, 2, np.nan, 4]})
    df
    df.expanding(2).sum()


def test_constructor(frame_or_series):
    # [FUNCTION DOC]
    # Regression test for GH 12669, parametrized over series and frame: creating an
    # expanding window with minimum periods 1 does not raise.
    # GH 12669

    c = frame_or_series(range(5)).expanding

    # valid
    c(min_periods=1)


@pytest.mark.parametrize("w", [2.0, "foo", np.array([2])])
def test_constructor_invalid(frame_or_series, w):
    # [FUNCTION DOC]
    # Parametrized over series/frame and invalid minimum periods (a float, a
    # string, a numpy array). Verifies a value error saying minimum periods must be
    # an integer.
    # not valid

    c = frame_or_series(range(5)).expanding
    msg = "min_periods must be an integer"
    with pytest.raises(ValueError, match=msg):
        c(min_periods=w)


@pytest.mark.parametrize(
    "expander",
    [
        1,
        pytest.param(
            "ls",
            marks=pytest.mark.xfail(
                reason="GH#16425 expanding with offset not supported"
            ),
        ),
    ],
)
def test_empty_df_expanding(expander):
    # [FUNCTION DOC]
    # Regression test for GH 15819. Parametrized over an integer window argument
    # (works) and an offset string "ls" (expected to fail because expanding with an
    # offset is not supported, GH#16425). Verifies expanding sum of an empty frame,
    # with and without an empty datetime index, returns an equal empty frame.
    # GH 15819 Verifies that datetime and integer expanding windows can be
    # applied to empty DataFrames

    expected = DataFrame()
    result = DataFrame().expanding(expander).sum()
    tm.assert_frame_equal(result, expected)

    # Verifies that datetime and integer expanding windows can be applied
    # to empty DataFrames with datetime index
    expected = DataFrame(index=DatetimeIndex([]))
    result = DataFrame(index=DatetimeIndex([])).expanding(expander).sum()
    tm.assert_frame_equal(result, expected)


def test_missing_minp_zero():
    # [FUNCTION DOC]
    # Regression test for pull request 18921. A series with only a missing value:
    # expanding sum with minimum periods zero gives 0.0, with minimum periods one
    # gives missing.
    # https://github.com/pandas-dev/pandas/pull/18921
    # minp=0
    x = Series([np.nan])
    result = x.expanding(min_periods=0).sum()
    expected = Series([0.0])
    tm.assert_series_equal(result, expected)

    # minp=1
    result = x.expanding(min_periods=1).sum()
    expected = Series([np.nan])
    tm.assert_series_equal(result, expected)


def test_expanding():
    # [FUNCTION DOC]
    # Regression test for gh-23372. A frame of ones (10 rows by 20 columns) with
    # expanding sum and minimum periods 3 gives missing for the first two rows and
    # then the running counts 3 to 10 in every column.
    # see gh-23372.
    df = DataFrame(np.ones((10, 20)))

    expected = DataFrame(
        {i: [np.nan] * 2 + [float(j) for j in range(3, 11)] for i in range(20)}
    )
    result = df.expanding(3).sum()
    tm.assert_frame_equal(result, expected)


def test_expanding_count_with_min_periods(frame_or_series):
    # [FUNCTION DOC]
    # Regression test for GH 26996. Expanding count over five values with minimum
    # periods 3 gives missing, missing, 3, 4, 5 (series and frame).
    # GH 26996
    result = frame_or_series(range(5)).expanding(min_periods=3).count()
    expected = frame_or_series([np.nan, np.nan, 3.0, 4.0, 5.0])
    tm.assert_equal(result, expected)


def test_expanding_count_default_min_periods_with_null_values(frame_or_series):
    # [FUNCTION DOC]
    # Regression test for GH 26996. Expanding count with default minimum periods
    # over values containing one missing entry counts only non-missing values, so
    # the counts do not increase at the missing position.
    # GH 26996
    values = [1, 2, 3, np.nan, 4, 5, 6]
    expected_counts = [1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0]

    result = frame_or_series(values).expanding().count()
    expected = frame_or_series(expected_counts)
    tm.assert_equal(result, expected)


def test_expanding_count_with_min_periods_exceeding_series_length(frame_or_series):
    # [FUNCTION DOC]
    # Regression test for GH 25857. When minimum periods (6) exceeds the length (5),
    # every expanding count is missing.
    # GH 25857
    result = frame_or_series(range(5)).expanding(min_periods=6).count()
    expected = frame_or_series([np.nan, np.nan, np.nan, np.nan, np.nan])
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "df,expected,min_periods",
    [
        (
            {"A": [1, 2, 3], "B": [4, 5, 6]},
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [1, 2], "B": [4, 5]}, [0, 1]),
                ({"A": [1, 2, 3], "B": [4, 5, 6]}, [0, 1, 2]),
            ],
            3,
        ),
        (
            {"A": [1, 2, 3], "B": [4, 5, 6]},
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [1, 2], "B": [4, 5]}, [0, 1]),
                ({"A": [1, 2, 3], "B": [4, 5, 6]}, [0, 1, 2]),
            ],
            2,
        ),
        (
            {"A": [1, 2, 3], "B": [4, 5, 6]},
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [1, 2], "B": [4, 5]}, [0, 1]),
                ({"A": [1, 2, 3], "B": [4, 5, 6]}, [0, 1, 2]),
            ],
            1,
        ),
        ({"A": [1], "B": [4]}, [], 2),
        (None, [({}, [])], 1),
        (
            {"A": [1, np.nan, 3], "B": [np.nan, 5, 6]},
            [
                ({"A": [1.0], "B": [np.nan]}, [0]),
                ({"A": [1, np.nan], "B": [np.nan, 5]}, [0, 1]),
                ({"A": [1, np.nan, 3], "B": [np.nan, 5, 6]}, [0, 1, 2]),
            ],
            3,
        ),
        (
            {"A": [1, np.nan, 3], "B": [np.nan, 5, 6]},
            [
                ({"A": [1.0], "B": [np.nan]}, [0]),
                ({"A": [1, np.nan], "B": [np.nan, 5]}, [0, 1]),
                ({"A": [1, np.nan, 3], "B": [np.nan, 5, 6]}, [0, 1, 2]),
            ],
            2,
        ),
        (
            {"A": [1, np.nan, 3], "B": [np.nan, 5, 6]},
            [
                ({"A": [1.0], "B": [np.nan]}, [0]),
                ({"A": [1, np.nan], "B": [np.nan, 5]}, [0, 1]),
                ({"A": [1, np.nan, 3], "B": [np.nan, 5, 6]}, [0, 1, 2]),
            ],
            1,
        ),
    ],
)
def test_iter_expanding_dataframe(df, expected, min_periods):
    # [FUNCTION DOC]
    # Regression test for GH 11704. Parametrized over frame data (with and without
    # missing values, empty data, a frame shorter than minimum periods), the
    # expected sequence of growing prefix frames with their index labels, and
    # minimum periods. Verifies iterating over the expanding object yields prefix
    # frames equal to the expected ones.
    # GH 11704
    df = DataFrame(df)
    expecteds = [DataFrame(values, index=index) for (values, index) in expected]

    for expected, actual in zip(expecteds, df.expanding(min_periods), strict=False):
        tm.assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    "ser,expected,min_periods",
    [
        (Series([1, 2, 3]), [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])], 3),
        (Series([1, 2, 3]), [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])], 2),
        (Series([1, 2, 3]), [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])], 1),
        (Series([1, 2]), [([1], [0]), ([1, 2], [0, 1])], 2),
        (Series([np.nan, 2]), [([np.nan], [0]), ([np.nan, 2], [0, 1])], 2),
        (Series([], dtype="int64"), [], 2),
    ],
)
def test_iter_expanding_series(ser, expected, min_periods):
    # [FUNCTION DOC]
    # Regression test for GH 11704. Parametrized over series (including one with a
    # missing value, a short one and an empty one), expected growing prefixes and
    # minimum periods. Verifies iteration yields equal prefix series and the same
    # number of items as expected (strict pairing).
    # GH 11704
    expecteds = [Series(values, index=index) for (values, index) in expected]

    for expected, actual in zip(expecteds, ser.expanding(min_periods), strict=True):
        tm.assert_series_equal(actual, expected)


def test_center_invalid():
    # [FUNCTION DOC]
    # Regression test for GH 20647: passing the center option to expanding raises a
    # type error saying an unexpected keyword was received.
    # GH 20647
    df = DataFrame()
    with pytest.raises(TypeError, match=".* got an unexpected keyword"):
        df.expanding(center=True)


def test_expanding_sem(frame_or_series):
    # [FUNCTION DOC]
    # Regression test for GH 26476, parametrized over series and frame. Expanding
    # standard error of the mean of 0, 1, 2 gives missing, 0.5, and the square root
    # of one third (frame results are taken from column zero).
    # GH: 26476
    obj = frame_or_series([0, 1, 2])
    result = obj.expanding().sem()
    if isinstance(result, DataFrame):
        result = Series(result[0].values)
    expected = Series([np.nan, 0.5, (1 / 3) ** 0.5])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("method", ["skew", "kurt"])
def test_expanding_skew_kurt_numerical_stability(method):
    # [FUNCTION DOC]
    # Regression test for GH 6929. Parametrized over skew and kurtosis. Verifies
    # expanding results (minimum periods 3) on random data are unchanged when a
    # large constant (5000) is added to every value.
    # GH: 6929
    s = Series(np.random.default_rng(2).random(10))
    expected = getattr(s.expanding(3), method)()
    s = s + 5000
    result = getattr(s.expanding(3), method)()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("window", [1, 3, 10, 20])
@pytest.mark.parametrize("method", ["min", "max", "average"])
@pytest.mark.parametrize("pct", [True, False])
@pytest.mark.parametrize("test_data", ["default", "duplicates", "nans"])
def test_rank(window, method, pct, ascending, test_data):
    # [FUNCTION DOC]
    # Parametrized over window argument (as minimum periods) 1, 3, 10, 20, method
    # (min, max, average), percentile flag, the ascending fixture, and data kind
    # (random floats, many duplicates, floats with missing and infinite values) of
    # length 20. Verifies the expanding rank equals applying a function that ranks
    # the window with the same options and takes the last rank.
    length = 20
    if test_data == "default":
        ser = Series(data=np.random.default_rng(2).random(length))
    elif test_data == "duplicates":
        ser = Series(data=np.random.default_rng(2).choice(3, length))
    elif test_data == "nans":
        ser = Series(
            data=np.random.default_rng(2).choice(
                [1.0, 0.25, 0.75, np.nan, np.inf, -np.inf], length
            )
        )

    expected = ser.expanding(window).apply(
        lambda x: x.rank(method=method, pct=pct, ascending=ascending).iloc[-1]
    )
    result = ser.expanding(window).rank(method=method, pct=pct, ascending=ascending)

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("window", [1, 3, 10, 20])
@pytest.mark.parametrize("test_data", ["default", "duplicates", "nans", "precision"])
def test_nunique(window, test_data):
    # [FUNCTION DOC]
    # Parametrized over minimum periods 1, 3, 10, 20 and data kind (random floats,
    # duplicates, missing and infinite values, and values computed so that nearly
    # equal floats might differ in the last bit). Verifies the expanding unique
    # count equals applying a function that counts unique values of the window.
    length = 20
    if test_data == "default":
        ser = Series(data=np.random.default_rng(2).random(length))
    elif test_data == "duplicates":
        ser = Series(data=np.random.default_rng(2).choice(3, length))
    elif test_data == "nans":
        ser = Series(
            data=np.random.default_rng(2).choice(
                [1.0, 0.25, 0.75, np.nan, np.inf, -np.inf], length
            )
        )
    elif test_data == "precision":
        ser = Series(
            data=[
                0.3,
                0.1 * 3,  # Not necessarily exactly 0.3
                0.6,
                0.2 * 3,  # Not necessarily exactly 0.6
                0.9,
                0.3 * 3,  # Not necessarily exactly 0.9
                0.5,
                0.1 * 5,  # Not necessarily exactly 0.5
                0.8,
                0.2 * 4,  # Not necessarily exactly 0.8
            ],
            dtype=np.float64,
        )

    expected = ser.expanding(window).apply(lambda x: x.nunique())
    result = ser.expanding(window).nunique()

    tm.assert_series_equal(result, expected)


def test_expanding_corr(series):
    # [FUNCTION DOC]
    # Verifies expanding correlation of the series fixture (missing values dropped)
    # with a noisy shorter series equals rolling correlation with window equal to
    # the length and minimum periods one.
    A = series.dropna()
    B = (A + np.random.default_rng(2).standard_normal(len(A)))[:-5]

    result = A.expanding().corr(B)

    rolling_result = A.rolling(window=len(A), min_periods=1).corr(B)

    tm.assert_almost_equal(rolling_result, result)


def test_expanding_count(series):
    # [FUNCTION DOC]
    # Verifies expanding count with minimum periods zero on the series fixture
    # equals rolling count with window equal to the length and minimum periods zero.
    result = series.expanding(min_periods=0).count()
    tm.assert_almost_equal(
        result, series.rolling(window=len(series), min_periods=0).count()
    )


def test_expanding_quantile(series):
    # [FUNCTION DOC]
    # Verifies expanding median (quantile 0.5) equals the full-length rolling
    # quantile with minimum periods one.
    result = series.expanding().quantile(0.5)

    rolling_result = series.rolling(window=len(series), min_periods=1).quantile(0.5)

    tm.assert_almost_equal(result, rolling_result)


def test_expanding_cov(series):
    # [FUNCTION DOC]
    # Verifies expanding covariance of the series fixture with a noisy shorter
    # series equals full-length rolling covariance with minimum periods one.
    A = series
    B = (A + np.random.default_rng(2).standard_normal(len(A)))[:-5]

    result = A.expanding().cov(B)

    rolling_result = A.rolling(window=len(A), min_periods=1).cov(B)

    tm.assert_almost_equal(rolling_result, result)


def test_expanding_cov_pairwise(frame):
    # [FUNCTION DOC]
    # Verifies pairwise expanding covariance of the frame fixture equals full-length
    # rolling pairwise covariance with minimum periods one.
    result = frame.expanding().cov()

    rolling_result = frame.rolling(window=len(frame), min_periods=1).cov()

    tm.assert_frame_equal(result, rolling_result)


def test_expanding_corr_pairwise(frame):
    # [FUNCTION DOC]
    # Verifies pairwise expanding correlation of the frame fixture equals
    # full-length rolling pairwise correlation with minimum periods one.
    result = frame.expanding().corr()

    rolling_result = frame.rolling(window=len(frame), min_periods=1).corr()
    tm.assert_frame_equal(result, rolling_result)


@pytest.mark.parametrize(
    "func,static_comp",
    [
        ("sum", lambda x: np.sum(x, axis=0)),
        ("mean", lambda x: np.mean(x, axis=0)),
        ("max", lambda x: np.max(x, axis=0)),
        ("min", lambda x: np.min(x, axis=0)),
    ],
    ids=["sum", "mean", "max", "min"],
)
def test_expanding_func(func, static_comp, frame_or_series):
    # [FUNCTION DOC]
    # Parametrized over sum, mean, max, min with a matching static numpy reduction.
    # On ten numbers followed by ten missing values (series or frame), verifies the
    # expanding result has the input type and, at position 10, equals the static
    # reduction over the first 11 values.
    data = frame_or_series(np.array(list(range(10)) + [np.nan] * 10))

    obj = data.expanding(min_periods=1)
    result = getattr(obj, func)()
    assert isinstance(result, frame_or_series)

    expected = static_comp(data[:11])
    if frame_or_series is Series:
        tm.assert_almost_equal(result[10], expected)
    else:
        tm.assert_series_equal(result.iloc[10], expected, check_names=False)


@pytest.mark.parametrize(
    "func,static_comp",
    [("sum", np.sum), ("mean", np.mean), ("max", np.max), ("min", np.min)],
    ids=["sum", "mean", "max", "min"],
)
def test_expanding_min_periods(func, static_comp):
    # [FUNCTION DOC]
    # Parametrized over sum, mean, max, min with a static numpy reduction. On 50
    # random values verifies: with minimum periods 30 the first 29 results are
    # missing and the last equals the static result; results turn non-missing at
    # position 14 for minimum periods 15 and at position 4 for 5 on a shorter
    # series; minimum periods zero equals one; and the last value with one equals
    # the static result.
    ser = Series(np.random.default_rng(2).standard_normal(50))

    result = getattr(ser.expanding(min_periods=30), func)()
    assert result[:29].isna().all()
    tm.assert_almost_equal(result.iloc[-1], static_comp(ser[:50]))

    # min_periods is working correctly
    result = getattr(ser.expanding(min_periods=15), func)()
    assert isna(result.iloc[13])
    assert notna(result.iloc[14])

    ser2 = Series(np.random.default_rng(2).standard_normal(20))
    result = getattr(ser2.expanding(min_periods=5), func)()
    assert isna(result[3])
    assert notna(result[4])

    # min_periods=0
    result0 = getattr(ser.expanding(min_periods=0), func)()
    result1 = getattr(ser.expanding(min_periods=1), func)()
    tm.assert_almost_equal(result0, result1)

    result = getattr(ser.expanding(min_periods=1), func)()
    tm.assert_almost_equal(result.iloc[-1], static_comp(ser[:50]))


def test_expanding_apply(engine_and_raw, frame_or_series):
    # [FUNCTION DOC]
    # Verifies expanding apply with a mean function, minimum periods one, for each
    # engine and raw pair and for series and frame: the result has the same type
    # and the value at position 9 equals the mean of the first 11 values.
    engine, raw = engine_and_raw
    data = frame_or_series(np.array(list(range(10)) + [np.nan] * 10))
    result = data.expanding(min_periods=1).apply(
        lambda x: x.mean(), raw=raw, engine=engine
    )
    assert isinstance(result, frame_or_series)

    if frame_or_series is Series:
        tm.assert_almost_equal(result[9], np.mean(data[:11], axis=0))
    else:
        tm.assert_series_equal(
            result.iloc[9], np.mean(data[:11], axis=0), check_names=False
        )


def test_expanding_min_periods_apply(engine_and_raw):
    # [FUNCTION DOC]
    # Apply-based counterpart of the minimum periods test: with a mean function,
    # verifies the first 29 results are missing for minimum periods 30 and the last
    # equals the overall mean, the missing to non-missing switch positions for
    # minimum periods 15 and 5, and that minimum periods zero equals one.
    engine, raw = engine_and_raw
    ser = Series(np.random.default_rng(2).standard_normal(50))

    result = ser.expanding(min_periods=30).apply(
        lambda x: x.mean(), raw=raw, engine=engine
    )
    assert result[:29].isna().all()
    tm.assert_almost_equal(result.iloc[-1], np.mean(ser[:50]))

    # min_periods is working correctly
    result = ser.expanding(min_periods=15).apply(
        lambda x: x.mean(), raw=raw, engine=engine
    )
    assert isna(result.iloc[13])
    assert notna(result.iloc[14])

    ser2 = Series(np.random.default_rng(2).standard_normal(20))
    result = ser2.expanding(min_periods=5).apply(
        lambda x: x.mean(), raw=raw, engine=engine
    )
    assert isna(result[3])
    assert notna(result[4])

    # min_periods=0
    result0 = ser.expanding(min_periods=0).apply(
        lambda x: x.mean(), raw=raw, engine=engine
    )
    result1 = ser.expanding(min_periods=1).apply(
        lambda x: x.mean(), raw=raw, engine=engine
    )
    tm.assert_almost_equal(result0, result1)

    result = ser.expanding(min_periods=1).apply(
        lambda x: x.mean(), raw=raw, engine=engine
    )
    tm.assert_almost_equal(result.iloc[-1], np.mean(ser[:50]))


@pytest.mark.parametrize(
    "f",
    [
        lambda x: x.expanding(min_periods=5).cov(x, pairwise=True),
        lambda x: x.expanding(min_periods=5).corr(x, pairwise=True),
    ],
)
def test_moment_functions_zero_length_pairwise(f):
    # [FUNCTION DOC]
    # Parametrized over expanding pairwise covariance and correlation with minimum
    # periods 5 (lambda inputs). Verifies on a completely empty frame and on an
    # empty frame with a named float column and named empty index that the result
    # is an empty frame with a multi-level index built from the index and columns
    # (preserving names) and the expected column labels and dtype.
    df1 = DataFrame()
    df2 = DataFrame(columns=Index(["a"], name="foo"), index=Index([], name="bar"))
    df2["a"] = df2["a"].astype("float64")

    df1_expected = DataFrame(index=MultiIndex.from_product([df1.index, df1.columns]))
    df2_expected = DataFrame(
        index=MultiIndex.from_product([df2.index, df2.columns], names=["bar", "foo"]),
        columns=Index(["a"], name="foo"),
        dtype="float64",
    )

    df1_result = f(df1)
    tm.assert_frame_equal(df1_result, df1_expected)

    df2_result = f(df2)
    tm.assert_frame_equal(df2_result, df2_expected)


@pytest.mark.parametrize(
    "f",
    [
        lambda x: x.expanding().count(),
        lambda x: x.expanding(min_periods=5).cov(x, pairwise=False),
        lambda x: x.expanding(min_periods=5).corr(x, pairwise=False),
        lambda x: x.expanding(min_periods=5).max(),
        lambda x: x.expanding(min_periods=5).min(),
        lambda x: x.expanding(min_periods=5).first(),
        lambda x: x.expanding(min_periods=5).last(),
        lambda x: x.expanding(min_periods=5).sum(),
        lambda x: x.expanding(min_periods=5).mean(),
        lambda x: x.expanding(min_periods=5).std(),
        lambda x: x.expanding(min_periods=5).var(),
        lambda x: x.expanding(min_periods=5).skew(),
        lambda x: x.expanding(min_periods=5).kurt(),
        lambda x: x.expanding(min_periods=5).quantile(0.5),
        lambda x: x.expanding(min_periods=5).median(),
        lambda x: x.expanding(min_periods=5).apply(sum, raw=False),
        lambda x: x.expanding(min_periods=5).apply(sum, raw=True),
    ],
)
def test_moment_functions_zero_length(f):
    # [FUNCTION DOC]
    # Regression test for GH 8056. Parametrized over many expanding operations
    # (count, non-pairwise cov and corr, max, min, first, last, sum, mean, std, var,
    # skew, kurt, quantile, median, and apply with both raw settings). Verifies that
    # applying the operation to an empty float series, an empty frame and an empty
    # frame with one float column returns an object equal to the input.
    # GH 8056
    s = Series(dtype=np.float64)
    s_expected = s
    df1 = DataFrame()
    df1_expected = df1
    df2 = DataFrame(columns=["a"])
    df2["a"] = df2["a"].astype("float64")
    df2_expected = df2

    s_result = f(s)
    tm.assert_series_equal(s_result, s_expected)

    df1_result = f(df1)
    tm.assert_frame_equal(df1_result, df1_expected)

    df2_result = f(df2)
    tm.assert_frame_equal(df2_result, df2_expected)


def test_expanding_apply_empty_series(engine_and_raw):
    # [FUNCTION DOC]
    # Verifies expanding apply of a mean function on an empty float series returns
    # an equal empty series for each engine and raw pair.
    engine, raw = engine_and_raw
    ser = Series([], dtype=np.float64)
    tm.assert_series_equal(
        ser, ser.expanding().apply(lambda x: x.mean(), raw=raw, engine=engine)
    )


def test_expanding_apply_min_periods_0(engine_and_raw):
    # [FUNCTION DOC]
    # Regression test for GH 8080. With three missing objects and minimum periods
    # zero, applying a length function gives 1.0, 2.0, 3.0.
    # GH 8080
    engine, raw = engine_and_raw
    s = Series([None, None, None])
    result = s.expanding(min_periods=0).apply(lambda x: len(x), raw=raw, engine=engine)
    expected = Series([1.0, 2.0, 3.0])
    tm.assert_series_equal(result, expected)


def test_expanding_cov_diff_index():
    # [FUNCTION DOC]
    # Regression test for GH 7512. Verifies expanding covariance between series
    # with different indexes: two series of unequal length give the result at the
    # union of the labels with missing values until two common points exist; a
    # series with an explicit missing value gives the same result; and series with
    # interleaved but partly disjoint labels give a result over four labels with
    # only the last non-missing.
    # GH 7512
    s1 = Series([1, 2, 3], index=range(3))
    s2 = Series([1, 3], index=range(0, 4, 2))
    result = s1.expanding().cov(s2)
    expected = Series([None, None, 2.0])
    tm.assert_series_equal(result, expected)

    s2a = Series([1, None, 3], index=[0, 1, 2])
    result = s1.expanding().cov(s2a)
    tm.assert_series_equal(result, expected)

    s1 = Series([7, 8, 10], index=[0, 1, 3])
    s2 = Series([7, 9, 10], index=[0, 2, 3])
    result = s1.expanding().cov(s2)
    expected = Series([None, None, None, 4.5], index=list(range(4)))
    tm.assert_series_equal(result, expected)


def test_expanding_corr_diff_index():
    # [FUNCTION DOC]
    # Regression test for GH 7512. Correlation counterpart of the differing index
    # covariance test; the non-missing values are 1.0 (perfect correlation).
    # GH 7512
    s1 = Series([1, 2, 3], index=range(3))
    s2 = Series([1, 3], index=range(0, 4, 2))
    result = s1.expanding().corr(s2)
    expected = Series([None, None, 1.0])
    tm.assert_series_equal(result, expected)

    s2a = Series([1, None, 3], index=[0, 1, 2])
    result = s1.expanding().corr(s2a)
    tm.assert_series_equal(result, expected)

    s1 = Series([7, 8, 10], index=[0, 1, 3])
    s2 = Series([7, 9, 10], index=[0, 2, 3])
    result = s1.expanding().corr(s2)
    expected = Series([None, None, None, 1.0], index=list(range(4)))
    tm.assert_series_equal(result, expected)


def test_expanding_cov_pairwise_diff_length():
    # [FUNCTION DOC]
    # Regression test for GH 7512 (and gh-15826 for names). Verifies pairwise
    # expanding covariance between two column-labelled frames of different lengths
    # and indexes gives, at row label 2, the same 2 by 2 frame whichever of the
    # frames have the shortened index; the expected frame has named column and index
    # labels.
    # GH 7512
    df1 = DataFrame([[1, 5], [3, 2], [3, 9]], columns=Index(["A", "B"], name="foo"))
    df1a = DataFrame(
        [[1, 5], [3, 9]], index=[0, 2], columns=Index(["A", "B"], name="foo")
    )
    df2 = DataFrame(
        [[5, 6], [None, None], [2, 1]], columns=Index(["X", "Y"], name="foo")
    )
    df2a = DataFrame(
        [[5, 6], [2, 1]], index=[0, 2], columns=Index(["X", "Y"], name="foo")
    )
    # xref gh-15826
    # .loc is not preserving the names
    result1 = df1.expanding().cov(df2, pairwise=True).loc[2]
    result2 = df1.expanding().cov(df2a, pairwise=True).loc[2]
    result3 = df1a.expanding().cov(df2, pairwise=True).loc[2]
    result4 = df1a.expanding().cov(df2a, pairwise=True).loc[2]
    expected = DataFrame(
        [[-3.0, -6.0], [-5.0, -10.0]],
        columns=Index(["A", "B"], name="foo"),
        index=Index(["X", "Y"], name="foo"),
    )
    tm.assert_frame_equal(result1, expected)
    tm.assert_frame_equal(result2, expected)
    tm.assert_frame_equal(result3, expected)
    tm.assert_frame_equal(result4, expected)


def test_expanding_corr_pairwise_diff_length():
    # [FUNCTION DOC]
    # Regression test for GH 7512. Correlation counterpart: at row label 2 the
    # pairwise expanding correlation between differently sized frames gives all
    # -1.0 in a 2 by 2 frame for all four index combinations.
    # GH 7512
    df1 = DataFrame(
        [[1, 2], [3, 2], [3, 4]], columns=["A", "B"], index=Index(range(3), name="bar")
    )
    df1a = DataFrame(
        [[1, 2], [3, 4]], index=Index([0, 2], name="bar"), columns=["A", "B"]
    )
    df2 = DataFrame(
        [[5, 6], [None, None], [2, 1]],
        columns=["X", "Y"],
        index=Index(range(3), name="bar"),
    )
    df2a = DataFrame(
        [[5, 6], [2, 1]], index=Index([0, 2], name="bar"), columns=["X", "Y"]
    )
    result1 = df1.expanding().corr(df2, pairwise=True).loc[2]
    result2 = df1.expanding().corr(df2a, pairwise=True).loc[2]
    result3 = df1a.expanding().corr(df2, pairwise=True).loc[2]
    result4 = df1a.expanding().corr(df2a, pairwise=True).loc[2]
    expected = DataFrame(
        [[-1.0, -1.0], [-1.0, -1.0]], columns=["A", "B"], index=Index(["X", "Y"])
    )
    tm.assert_frame_equal(result1, expected)
    tm.assert_frame_equal(result2, expected)
    tm.assert_frame_equal(result3, expected)
    tm.assert_frame_equal(result4, expected)


@pytest.mark.parametrize(
    "values,method,expected",
    [
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "first",
            [float("nan"), float("nan"), 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        (
            [1.0, np.nan, 3.0, np.nan, 5.0, np.nan, 7.0, np.nan, 9.0, np.nan],
            "first",
            [
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
        ),
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "last",
            [float("nan"), float("nan"), 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ),
        (
            [1.0, np.nan, 3.0, np.nan, 5.0, np.nan, 7.0, np.nan, 9.0, np.nan],
            "last",
            [
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                5.0,
                5.0,
                7.0,
                7.0,
                9.0,
                9.0,
            ],
        ),
    ],
)
def test_expanding_first_last(values, method, expected):
    # [FUNCTION DOC]
    # Regression test for GH#33155. Parametrized over ten values with and without
    # missing entries, the method first or last, and expected results. With
    # minimum periods 3, verifies the first two results are missing, and afterwards
    # first returns the first valid value and last returns the latest valid value,
    # for series and frame.
    # GH#33155
    x = Series(values)
    result = getattr(x.expanding(3), method)()
    expected = Series(expected)
    tm.assert_almost_equal(result, expected)

    x = DataFrame({"A": values})
    result = getattr(x.expanding(3), method)()
    expected = DataFrame({"A": expected})
    tm.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "values,method,expected",
    [
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "first",
            [1.0] * 10,
        ),
        (
            [1.0, np.nan, 3.0, np.nan, 5.0, np.nan, 7.0, np.nan, 9.0, np.nan],
            "first",
            [1.0] * 10,
        ),
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "last",
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ),
        (
            [1.0, np.nan, 3.0, np.nan, 5.0, np.nan, 7.0, np.nan, 9.0, np.nan],
            "last",
            [1.0, 1.0, 3.0, 3.0, 5.0, 5.0, 7.0, 7.0, 9.0, 9.0],
        ),
    ],
)
def test_expanding_first_last_no_minp(values, method, expected):
    # [FUNCTION DOC]
    # Regression test for GH#33155. Same as the first/last test but with minimum
    # periods zero: results exist from the first position (first is always the first
    # value, last is the most recent valid value).
    # GH#33155
    x = Series(values)
    result = getattr(x.expanding(min_periods=0), method)()
    expected = Series(expected)
    tm.assert_almost_equal(result, expected)

    x = DataFrame({"A": values})
    result = getattr(x.expanding(min_periods=0), method)()
    expected = DataFrame({"A": expected})
    tm.assert_almost_equal(result, expected)


def test_expanding_apply_args_kwargs(engine_and_raw):
    # [FUNCTION DOC]
    # Verifies extra positional arguments (a tuple) and keyword arguments (a
    # mapping) are forwarded to the applied function for each engine and raw pair:
    # the result equals the expanding mean plus the constant 20.
    def mean_w_arg(x, const):
        # [FUNCTION DOC]
        # HELPER: nested user function. Takes an array x and a number const; returns the
        # mean of x plus const.
        return np.mean(x) + const

    engine, raw = engine_and_raw

    df = DataFrame(np.random.default_rng(2).random((20, 3)))

    expected = df.expanding().apply(np.mean, engine=engine, raw=raw) + 20.0

    result = df.expanding().apply(mean_w_arg, engine=engine, raw=raw, args=(20,))
    tm.assert_frame_equal(result, expected)

    result = df.expanding().apply(mean_w_arg, raw=raw, kwargs={"const": 20})
    tm.assert_frame_equal(result, expected)


def test_numeric_only_frame(arithmetic_win_operators, numeric_only):
    # [FUNCTION DOC]
    # Regression test for GH#46560. For each expanding operation present, verifies
    # the result on a frame with two numeric columns and one object column, for the
    # numeric_only fixture, equals the aggregation of the corresponding columns
    # (object column dropped when numeric_only is true) cast to float.
    # GH#46560
    kernel = arithmetic_win_operators
    df = DataFrame({"a": [1], "b": 2, "c": 3})
    df["c"] = df["c"].astype(object)
    expanding = df.expanding()
    op = getattr(expanding, kernel, None)
    if op is not None:
        result = op(numeric_only=numeric_only)

        columns = ["a", "b"] if numeric_only else ["a", "b", "c"]
        expected = df[columns].agg([kernel]).reset_index(drop=True).astype(float)
        assert list(expected.columns) == columns

        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("kernel", ["corr", "cov"])
@pytest.mark.parametrize("use_arg", [True, False])
def test_numeric_only_corr_cov_frame(kernel, numeric_only, use_arg):
    # [FUNCTION DOC]
    # Regression test for GH#46560. Parametrized over corr/cov, with or without the
    # other-frame argument, and numeric_only. Verifies the result equals the same
    # operation on a float-cast frame of the selected columns.
    # GH#46560
    df = DataFrame({"a": [1, 2, 3], "b": 2, "c": 3})
    df["c"] = df["c"].astype(object)
    arg = (df,) if use_arg else ()
    expanding = df.expanding()
    op = getattr(expanding, kernel)
    result = op(*arg, numeric_only=numeric_only)

    # Compare result to op using float dtypes, dropping c when numeric_only is True
    columns = ["a", "b"] if numeric_only else ["a", "b", "c"]
    df2 = df[columns].astype(float)
    arg2 = (df2,) if use_arg else ()
    expanding2 = df2.expanding()
    op2 = getattr(expanding2, kernel)
    expected = op2(*arg2, numeric_only=numeric_only)

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", [int, object])
def test_numeric_only_series(arithmetic_win_operators, numeric_only, dtype):
    # [FUNCTION DOC]
    # Regression test for GH#46560. Parametrized over integer and object dtype and
    # numeric_only. With object dtype and numeric_only true, verifies a not
    # implemented error saying the expanding operation does not implement
    # numeric_only; otherwise the result equals the series aggregate cast to float.
    # GH#46560
    kernel = arithmetic_win_operators
    ser = Series([1], dtype=dtype)
    expanding = ser.expanding()
    op = getattr(expanding, kernel)
    if numeric_only and dtype is object:
        msg = f"Expanding.{kernel} does not implement numeric_only"
        with pytest.raises(NotImplementedError, match=msg):
            op(numeric_only=numeric_only)
    else:
        result = op(numeric_only=numeric_only)
        expected = ser.agg([kernel]).reset_index(drop=True).astype(float)
        tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("kernel", ["corr", "cov"])
@pytest.mark.parametrize("use_arg", [True, False])
@pytest.mark.parametrize("dtype", [int, object])
def test_numeric_only_corr_cov_series(kernel, use_arg, numeric_only, dtype):
    # [FUNCTION DOC]
    # Regression test for GH#46560. Parametrized over corr/cov, with or without the
    # other argument, dtype integer or object, and numeric_only. Object dtype with
    # numeric_only true must raise a not implemented error; otherwise the result
    # equals the same operation on a float-cast copy.
    # GH#46560
    ser = Series([1, 2, 3], dtype=dtype)
    arg = (ser,) if use_arg else ()
    expanding = ser.expanding()
    op = getattr(expanding, kernel)
    if numeric_only and dtype is object:
        msg = f"Expanding.{kernel} does not implement numeric_only"
        with pytest.raises(NotImplementedError, match=msg):
            op(*arg, numeric_only=numeric_only)
    else:
        result = op(*arg, numeric_only=numeric_only)

        ser2 = ser.astype(float)
        arg2 = (ser2,) if use_arg else ()
        expanding2 = ser2.expanding()
        op2 = getattr(expanding2, kernel)
        expected = op2(*arg2, numeric_only=numeric_only)
        tm.assert_series_equal(result, expected)
