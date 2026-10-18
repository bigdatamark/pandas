# [MODULE DOC]
# Tests of the public resample API on datetime-indexed objects: textual representation,
# basic methods, group keys, pipe, column selection and attribute access, errors for missing
# columns, aggregation with functions, dictionaries, lists and named aggregations (across
# four equivalent forms of the same data: index-based resampler, column-based resampler,
# level-based resampler on a two-level index, and the time grouper), nested-renamer errors,
# selection validation, end and end-of-day origins, numeric-only behavior for frame and
# series downsampling, and several regressions. Provides many fixtures: a minute-frequency
# index over nine days, a random series and frame from it, and a set of frames, resamplers
# and aggregated pieces used by the aggregation tests, plus a combined fixture that
# iterates over the four forms.

from datetime import datetime
import re

import numpy as np
import pytest

from pandas._libs import lib
from pandas._libs.tslibs import Day

import pandas as pd
from pandas import (
    DataFrame,
    NamedAgg,
    Series,
)
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range


@pytest.fixture
def dti():
    # [FUNCTION DOC]
    # Fixture named dti: a datetime index at minute frequency from 2005-01-01 to 2005-01-10.
    return date_range(start=datetime(2005, 1, 1), end=datetime(2005, 1, 10), freq="Min")


@pytest.fixture
def _test_series(dti):
    # [FUNCTION DOC]
    # Fixture: a random-valued series on the minute index fixture. Name begins with an underscore
    # to avoid test collection.
    return Series(np.random.default_rng(2).random(len(dti)), dti)


@pytest.fixture
def test_frame(dti, _test_series):
    # [FUNCTION DOC]
    # Fixture named test_frame: a frame with columns A and B both equal to the random series
    # and column C counting positions, on the minute index.
    return DataFrame({"A": _test_series, "B": _test_series, "C": np.arange(len(dti))})


def test_str(_test_series):
    # [FUNCTION DOC]
    # Verifies the string representation of an hourly resampler on a series includes the class
    # name, frequency, closed side, label side, convention and origin start of day, and that
    # a custom origin string shows the origin as a full timestamp.
    r = _test_series.resample("h")
    assert (
        "DatetimeIndexResampler [freq=<Hour>, closed=left, "
        "label=left, convention=start, origin=start_day]" in str(r)
    )

    r = _test_series.resample("h", origin="2000-01-01")
    assert (
        "DatetimeIndexResampler [freq=<Hour>, closed=left, "
        "label=left, convention=start, origin=2000-01-01 00:00:00]" in str(r)
    )


def test_api(_test_series):
    # [FUNCTION DOC]
    # Verifies hourly mean of the series resampler yields a series of length 217, and of the
    # one-column frame resampler yields a frame of length 217.
    r = _test_series.resample("h")
    result = r.mean()
    assert isinstance(result, Series)
    assert len(result) == 217

    r = _test_series.to_frame().resample("h")
    result = r.mean()
    assert isinstance(result, DataFrame)
    assert len(result) == 217


def test_groupby_resample_api():
    # [FUNCTION DOC]
    # Regression test (GH 12448): group by group then resample daily and forward fill using
    # apply on a weekly frame gives the expected value pattern on a two-level index and
    # selecting the val column; ensures no warnings are triggered when appropriate.
    # GH 12448
    # .groupby(...).resample(...) hitting warnings
    # when appropriate
    df = DataFrame(
        {
            "date": date_range(start="2016-01-01", periods=4, freq="W"),
            "group": [1, 1, 2, 2],
            "val": [5, 6, 7, 8],
        }
    ).set_index("date")

    # replication step
    i = (
        date_range("2016-01-03", periods=8).tolist()
        + date_range("2016-01-17", periods=8).tolist()
    )
    index = pd.MultiIndex.from_arrays([[1] * 8 + [2] * 8, i], names=["group", "date"])
    expected = DataFrame({"val": [5] * 7 + [6] + [7] * 7 + [8]}, index=index)
    result = df.groupby("group").apply(lambda x: x.resample("1D").ffill())[["val"]]
    tm.assert_frame_equal(result, expected)


def test_groupby_resample_on_api():
    # [FUNCTION DOC]
    # Regression test (GH 15021): grouping by key and resampling daily with the on argument
    # equals setting the date column as index first (no unexpected keyword warning).
    # GH 15021
    # .groupby(...).resample(on=...) results in an unexpected
    # keyword warning.
    df = DataFrame(
        {
            "key": ["A", "B"] * 5,
            "dates": date_range("2016-01-01", periods=10),
            "values": np.random.default_rng(2).standard_normal(10),
        }
    )

    expected = df.set_index("dates").groupby("key").resample("D").mean()
    result = df.groupby("key").resample("D", on="dates").mean()
    tm.assert_frame_equal(result, expected)


def test_resample_group_keys():
    # [FUNCTION DOC]
    # Verifies the group keys argument of a resampler's apply: false and the default (also false)
    # return the identity-applied frame unchanged, while true adds the bin start as an extra
    # leading index level repeated per row (five rows each for 2000-01-01 and 2000-01-06).
    df = DataFrame({"A": 1, "B": 2}, index=date_range("2000", periods=10, unit="ns"))
    expected = df.copy()

    # group_keys=False
    g = df.resample("5D", group_keys=False)
    result = g.apply(lambda x: x)
    tm.assert_frame_equal(result, expected)

    # group_keys defaults to False
    g = df.resample("5D")
    result = g.apply(lambda x: x)
    tm.assert_frame_equal(result, expected)

    # group_keys=True
    expected.index = pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(["2000-01-01", "2000-01-06"]).as_unit("ns").repeat(5),
            expected.index,
        ]
    )
    g = df.resample("5D", group_keys=True)
    result = g.apply(lambda x: x)
    tm.assert_frame_equal(result, expected)


def test_pipe(test_frame, _test_series):
    # [FUNCTION DOC]
    # Regression test (GH17905): the pipe method of a resampler applies a function to the
    # resampler: maximum minus mean per hour, for a series and a frame.
    # GH17905

    # series
    r = _test_series.resample("h")
    expected = r.max() - r.mean()
    result = r.pipe(lambda x: x.max() - x.mean())
    tm.assert_series_equal(result, expected)

    # dataframe
    r = test_frame.resample("h")
    expected = r.max() - r.mean()
    result = r.pipe(lambda x: x.max() - x.mean())
    tm.assert_frame_equal(result, expected)


def test_getitem(test_frame):
    # [FUNCTION DOC]
    # Verifies selection through the resampler: the selected object has the frame's columns by
    # default, a single column selection gives the series named B, and a tuple selection of A and
    # B gives those two columns.
    # KNOWN ISSUE: the last tuple-selection check is an exact duplicate of the previous one.
    r = test_frame.resample("h")
    tm.assert_index_equal(r._selected_obj.columns, test_frame.columns)

    r = test_frame.resample("h")["B"]
    assert r._selected_obj.name == test_frame.columns[1]

    # technically this is allowed
    r = test_frame.resample("h")["A", "B"]
    tm.assert_index_equal(r._selected_obj.columns, test_frame.columns[[0, 1]])

    r = test_frame.resample("h")["A", "B"]
    tm.assert_index_equal(r._selected_obj.columns, test_frame.columns[[0, 1]])


@pytest.mark.parametrize("key", [["D"], ["A", "D"]])
def test_select_bad_cols(key, test_frame):
    # [FUNCTION DOC]
    # Parametrized by key (list with D, or A and D). Selecting a missing column from a resampler
    # raises a key error with the exact message that columns were not found and only D is named
    # (A is not reported). The comment warns that the message regex must be revisited if the message
    # changes.
    g = test_frame.resample("h")
    # 'A' should not be referenced as a bad column...
    # will have to rethink regex if you change message!
    msg = r"^\"Columns not found: 'D'\"$"
    with pytest.raises(KeyError, match=msg):
        g[key]


def test_attribute_access(test_frame):
    # [FUNCTION DOC]
    # Verifies attribute-style column access on a resampler equals item access for the sum of A.
    r = test_frame.resample("h")
    tm.assert_series_equal(r.A.sum(), r["A"].sum())


@pytest.mark.parametrize("attr", ["groups", "ngroups", "indices"])
def test_api_compat_before_use(attr):
    # [FUNCTION DOC]
    # Parametrized by groups, number of groups and indices. Verifies these attributes can be read
    # on a resampler before any aggregation (the binner is set up lazily) and again after
    # computing the mean.
    # make sure that we are setting the binner
    # on these attributes
    rng = date_range("1/1/2012", periods=100, freq="s")
    ts = Series(np.arange(len(rng)), index=rng)
    rs = ts.resample("30s")

    # before use
    getattr(rs, attr)

    # after grouper is initialized is ok
    rs.mean()
    getattr(rs, attr)


def tests_raises_on_nuisance(test_frame, using_infer_string):
    # [FUNCTION DOC]
    # Verifies nuisance (non-numeric) column behavior: with a string column added (mutating the
    # fixture frame), selecting A and B gives their means concatenated; the full resampler mean
    # raises a type error (message depends on the inferred-string setting); and with numeric-only
    # enabled it equals the mean of columns A, B and C.
    df = test_frame
    df["D"] = "foo"
    r = df.resample("h")
    result = r[["A", "B"]].mean()
    expected = pd.concat([r.A.mean(), r.B.mean()], axis=1)
    tm.assert_frame_equal(result, expected)

    expected = r[["A", "B", "C"]].mean()
    msg = re.escape("agg function failed [how->mean,dtype->")
    if using_infer_string:
        msg = "dtype 'str' does not support operation 'mean'"
    with pytest.raises(TypeError, match=msg):
        r.mean()
    result = r.mean(numeric_only=True)
    tm.assert_frame_equal(result, expected)


def test_downsample_but_actually_upsampling():
    # [FUNCTION DOC]
    # Verifies that converting the frequency of a second-frequency series to every twenty
    # seconds (nominally a downsample but really a reindex) gives values at positions zero, twenty,
    # forty, sixty and eighty.
    # this is reindex / asfreq
    rng = date_range("1/1/2012", periods=100, freq="s")
    ts = Series(np.arange(len(rng), dtype="int64"), index=rng)
    result = ts.resample("20s").asfreq()
    expected = Series(
        [0, 20, 40, 60, 80],
        index=date_range("2012-01-01 00:00:00", freq="20s", periods=5),
    )
    tm.assert_series_equal(result, expected)


def test_combined_up_downsampling_of_irregular():
    # [FUNCTION DOC]
    # Verifies a combined down- and upsampling of an irregular series: the mean in two-second
    # bins forward filled equals the specified sixteen values (0.5, 2.5, 5, 7, ... , 30) on a regular
    # two-second index, preserving the semantics of mean then forward fill.
    # since we are really doing an operation like this
    # ts2.resample('2s').mean().ffill()
    # preserve these semantics

    rng = date_range("1/1/2012", periods=100, freq="s", unit="ns")
    ts = Series(np.arange(len(rng)), index=rng)
    ts2 = ts.iloc[[0, 1, 2, 3, 5, 7, 11, 15, 16, 25, 30]]

    result = ts2.resample("2s").mean().ffill()
    expected = Series(
        [
            0.5,
            2.5,
            5.0,
            7.0,
            7.0,
            11.0,
            11.0,
            15.0,
            16.0,
            16.0,
            16.0,
            16.0,
            25.0,
            25.0,
            25.0,
            30.0,
        ],
        index=pd.DatetimeIndex(
            [
                "2012-01-01 00:00:00",
                "2012-01-01 00:00:02",
                "2012-01-01 00:00:04",
                "2012-01-01 00:00:06",
                "2012-01-01 00:00:08",
                "2012-01-01 00:00:10",
                "2012-01-01 00:00:12",
                "2012-01-01 00:00:14",
                "2012-01-01 00:00:16",
                "2012-01-01 00:00:18",
                "2012-01-01 00:00:20",
                "2012-01-01 00:00:22",
                "2012-01-01 00:00:24",
                "2012-01-01 00:00:26",
                "2012-01-01 00:00:28",
                "2012-01-01 00:00:30",
            ],
            dtype="datetime64[ns]",
            freq="2s",
        ),
    )
    tm.assert_series_equal(result, expected)


def test_transform_series(_test_series):
    # [FUNCTION DOC]
    # Verifies that the resampler's transform with the mean on twenty-minute bins equals the
    # grouper-based transform.
    r = _test_series.resample("20min")
    expected = _test_series.groupby(pd.Grouper(freq="20min")).transform("mean")
    result = r.transform("mean")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("on", [None, "date"])
def test_transform_frame(on):
    # [FUNCTION DOC]
    # Regression test (GH#47079), parametrized by on (none or the date column). Resampler
    # transform with the mean equals the grouper transform; when the date is a column the
    # expected result has a default integer index.
    # GH#47079
    index = date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D")
    index.name = "date"
    df = DataFrame(
        np.random.default_rng(2).random((10, 2)), columns=list("AB"), index=index
    )
    expected = df.groupby(pd.Grouper(freq="20min")).transform("mean")
    if on == "date":
        # Move date to being a column; result will then have a RangeIndex
        expected = expected.reset_index(drop=True)
        df = df.reset_index()

    r = df.resample("20min", on=on)
    result = r.transform("mean")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "func",
    [
        lambda x: x.resample("20min", group_keys=False),
        lambda x: x.groupby(pd.Grouper(freq="20min"), group_keys=False),
    ],
    ids=["resample", "groupby"],
)
def test_apply_without_aggregation(func, _test_series):
    # [FUNCTION DOC]
    # Parametrized by two constructors (resampler without group keys, grouper-based groupby
    # without group keys). Verifies applying an identity function with no aggregation returns
    # the original series for both.
    # both resample and groupby should work w/o aggregation
    t = func(_test_series)
    result = t.apply(lambda x: x)
    tm.assert_series_equal(result, _test_series)


def test_apply_without_aggregation2(_test_series):
    # [FUNCTION DOC]
    # Verifies applying an identity on a selected column of a resampler of a one-column frame
    # returns the original series renamed to the column name.
    grouped = _test_series.to_frame(name="foo").resample("20min", group_keys=False)
    result = grouped["foo"].apply(lambda x: x)
    tm.assert_series_equal(result, _test_series.rename("foo"))


def test_agg_consistency():
    # [FUNCTION DOC]
    # Verifies aggregation with a dictionary naming labels that are not columns raises a key
    # error listing the two missing labels.
    # make sure that we are consistent across
    # similar aggregations with and w/o selection list
    df = DataFrame(
        np.random.default_rng(2).standard_normal((1000, 3)),
        index=date_range("1/1/2012", freq="s", periods=1000),
        columns=["A", "B", "C"],
    )

    r = df.resample("3min")

    msg = r"Label\(s\) \['r1', 'r2'\] do not exist"
    with pytest.raises(KeyError, match=msg):
        r.agg({"r1": "mean", "r2": "sum"})


def test_agg_consistency_int_str_column_mix():
    # [FUNCTION DOC]
    # Regression test (GH#39025): same as the consistency test but with columns mixing an integer
    # and a string label; the error lists the missing integer and string labels.
    # GH#39025
    df = DataFrame(
        np.random.default_rng(2).standard_normal((1000, 2)),
        index=date_range("1/1/2012", freq="s", periods=1000),
        columns=[1, "a"],
    )

    r = df.resample("3min")

    msg = r"Label\(s\) \[2, 'b'\] do not exist"
    with pytest.raises(KeyError, match=msg):
        r.agg({2: "mean", "b": "sum"})


# TODO(GH#14008): once GH 14008 is fixed, move these tests into
# `Base` test class


@pytest.fixture
def index():
    # [FUNCTION DOC]
    # Fixture named index: a daily nanosecond datetime index of ten days from 2005-01-01 named
    # date. Comment above notes the following fixtures are meant to move into a base test class
    # once issue 14008 is fixed.
    index = date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D", unit="ns")
    index.name = "date"
    return index


@pytest.fixture
def df(index):
    # [FUNCTION DOC]
    # Fixture named df: a frame of ten random rows with columns A and B on the index fixture.
    frame = DataFrame(
        np.random.default_rng(2).random((10, 2)), columns=list("AB"), index=index
    )
    return frame


@pytest.fixture
def df_col(df):
    # [FUNCTION DOC]
    # Fixture named df_col: the frame with the index moved to a column.
    return df.reset_index()


@pytest.fixture
def df_mult(df_col, index):
    # [FUNCTION DOC]
    # Fixture named df_mult: a copy of the column-style frame whose index is a two-level index of
    # positions (named index) and dates (named date).
    df_mult = df_col.copy()
    df_mult.index = pd.MultiIndex.from_arrays(
        [range(10), index], names=["index", "date"]
    )
    return df_mult


@pytest.fixture
def a_mean(df):
    # [FUNCTION DOC]
    # Fixture: mean of column A resampled in two-day bins.
    return df.resample("2D")["A"].mean()


@pytest.fixture
def a_std(df):
    # [FUNCTION DOC]
    # Fixture: standard deviation of column A resampled in two-day bins.
    return df.resample("2D")["A"].std()


@pytest.fixture
def a_sum(df):
    # [FUNCTION DOC]
    # Fixture: sum of column A resampled in two-day bins.
    return df.resample("2D")["A"].sum()


@pytest.fixture
def b_mean(df):
    # [FUNCTION DOC]
    # Fixture: mean of column B resampled in two-day bins.
    return df.resample("2D")["B"].mean()


@pytest.fixture
def b_std(df):
    # [FUNCTION DOC]
    # Fixture: standard deviation of column B resampled in two-day bins.
    return df.resample("2D")["B"].std()


@pytest.fixture
def b_sum(df):
    # [FUNCTION DOC]
    # Fixture: sum of column B resampled in two-day bins.
    return df.resample("2D")["B"].sum()


@pytest.fixture
def df_resample(df):
    # [FUNCTION DOC]
    # Fixture: the two-day resampler of the date-indexed frame.
    return df.resample("2D")


@pytest.fixture
def df_col_resample(df_col):
    # [FUNCTION DOC]
    # Fixture: the two-day resampler of the column-style frame using the date column.
    return df_col.resample("2D", on="date")


@pytest.fixture
def df_mult_resample(df_mult):
    # [FUNCTION DOC]
    # Fixture: the two-day resampler of the two-level index frame using the date level.
    return df_mult.resample("2D", level="date")


@pytest.fixture
def df_grouper_resample(df):
    # [FUNCTION DOC]
    # Fixture: the frame grouped by a two-day time grouper (not a resampler).
    return df.groupby(pd.Grouper(freq="2D"))


@pytest.fixture(
    params=["df_resample", "df_col_resample", "df_mult_resample", "df_grouper_resample"]
)
def cases(request):
    # [FUNCTION DOC]
    # Fixture named cases, parametrized over the names of the four equivalent resampler fixtures
    # above; returns the object produced by requesting the named fixture.
    return request.getfixturevalue(request.param)


def test_agg_mixed_column_aggregation(cases, a_mean, a_std, b_mean, b_std, request):
    # [FUNCTION DOC]
    # Aggregating each case with a list of the mean function and a lambda for standard
    # deviation (one delta degree of freedom) gives the concatenated means and standard
    # deviations for A and B with a two-level column index (labels mean and the lambda's
    # auto-name). For the two-level index case the date column is also included in the aggregation
    # and appended before A and B.
    expected = pd.concat([a_mean, a_std, b_mean, b_std], axis=1)
    expected.columns = pd.MultiIndex.from_product([["A", "B"], ["mean", "<lambda_0>"]])
    # "date" is an index and a column, so get included in the agg
    if "df_mult" in request.node.callspec.id:
        date_mean = cases["date"].mean()
        date_std = cases["date"].std()
        expected = pd.concat([date_mean, date_std, expected], axis=1)
        expected.columns = pd.MultiIndex.from_product(
            [["date", "A", "B"], ["mean", "<lambda_0>"]]
        )
    result = cases.aggregate([np.mean, lambda x: np.std(x, ddof=1)])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "agg",
    [
        {"func": {"A": np.mean, "B": lambda x: np.std(x, ddof=1)}},
        {"A": ("A", np.mean), "B": ("B", lambda x: np.std(x, ddof=1))},
        {"A": NamedAgg("A", np.mean), "B": NamedAgg("B", lambda x: np.std(x, ddof=1))},
    ],
)
def test_agg_both_mean_std_named_result(cases, a_mean, b_std, agg):
    # [FUNCTION DOC]
    # Parametrized by three equivalent forms of named aggregation (func keyword with dictionary,
    # tuple, named aggregation) giving A as mean and B as standard deviation. Result equals the
    # concatenation of the mean of A and the standard deviation of B, ignoring column order.
    expected = pd.concat([a_mean, b_std], axis=1)
    result = cases.aggregate(**agg)
    tm.assert_frame_equal(result, expected, check_like=True)


def test_agg_both_mean_std_dict_of_list(cases, a_mean, a_std):
    # [FUNCTION DOC]
    # Aggregating with a dictionary of a list per column gives mean and standard deviation of A
    # under a two-level column index.
    expected = pd.concat([a_mean, a_std], axis=1)
    expected.columns = pd.MultiIndex.from_tuples([("A", "mean"), ("A", "std")])
    result = cases.aggregate({"A": ["mean", "std"]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "agg", [{"func": ["mean", "sum"]}, {"mean": "mean", "sum": "sum"}]
)
def test_agg_both_mean_sum(cases, a_mean, a_sum, agg):
    # [FUNCTION DOC]
    # Parametrized by a function list or named aggregation of mean and sum. Aggregating column A
    # gives a frame with those columns, equal to the concatenated mean and sum.
    expected = pd.concat([a_mean, a_sum], axis=1)
    expected.columns = ["mean", "sum"]
    result = cases["A"].aggregate(**agg)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "agg",
    [
        {"A": {"mean": "mean", "sum": "sum"}},
        {
            "A": {"mean": "mean", "sum": "sum"},
            "B": {"mean2": "mean", "sum2": "sum"},
        },
    ],
)
def test_agg_dict_of_dict_specificationerror(cases, agg):
    # [FUNCTION DOC]
    # Parametrized by two nested dictionaries. Aggregating with a dictionary of dictionaries
    # raises the specification error saying nested renamer is not supported.
    msg = "nested renamer is not supported"
    with pytest.raises(pd.errors.SpecificationError, match=msg):
        cases.aggregate(agg)


def test_agg_dict_of_lists(cases, a_mean, a_std, b_mean, b_std):
    # [FUNCTION DOC]
    # Aggregating with a dictionary of lists for A and B (mean and standard deviation) gives
    # four columns on a two-level column index.
    expected = pd.concat([a_mean, a_std, b_mean, b_std], axis=1)
    expected.columns = pd.MultiIndex.from_tuples(
        [("A", "mean"), ("A", "std"), ("B", "mean"), ("B", "std")]
    )
    result = cases.aggregate({"A": ["mean", "std"], "B": ["mean", "std"]})
    tm.assert_frame_equal(result, expected, check_like=True)


@pytest.mark.parametrize(
    "agg",
    [
        {"func": {"A": np.sum, "B": lambda x: np.std(x, ddof=1)}},
        {"A": ("A", np.sum), "B": ("B", lambda x: np.std(x, ddof=1))},
        {"A": NamedAgg("A", np.sum), "B": NamedAgg("B", lambda x: np.std(x, ddof=1))},
    ],
)
def test_agg_with_lambda(cases, agg):
    # [FUNCTION DOC]
    # Parametrized by three named-aggregation forms. Aggregating with the sum of A and the
    # standard deviation lambda on B equals concatenation of the sum of A and applying the same
    # lambda to B.
    # passed lambda
    rcustom = cases["B"].apply(lambda x: np.std(x, ddof=1))
    expected = pd.concat([cases["A"].sum(), rcustom], axis=1)
    result = cases.agg(**agg)
    tm.assert_frame_equal(result, expected, check_like=True)


@pytest.mark.parametrize(
    "agg",
    [
        {"func": {"result1": np.sum, "result2": np.mean}},
        {"A": ("result1", np.sum), "B": ("result2", np.mean)},
        {"A": NamedAgg("result1", np.sum), "B": NamedAgg("result2", np.mean)},
    ],
)
def test_agg_no_column(cases, agg):
    # [FUNCTION DOC]
    # Parametrized by three named-aggregation forms referring to non-existing columns result1 and
    # result2. Selecting A and B and aggregating raises a key error listing the missing labels.
    msg = r"Label\(s\) \['result1', 'result2'\] do not exist"
    with pytest.raises(KeyError, match=msg):
        cases[["A", "B"]].agg(**agg)


@pytest.mark.parametrize(
    "cols, agg",
    [
        [None, {"A": ["sum", "std"], "B": ["mean", "std"]}],
        [
            [
                "A",
                "B",
            ],
            {"A": ["sum", "std"], "B": ["mean", "std"]},
        ],
    ],
)
def test_agg_specificationerror_nested(cases, cols, agg, a_sum, a_std, b_mean, b_std):
    # [FUNCTION DOC]
    # Parametrized by column selection (none, or A and B) and a dictionary of function lists
    # (sum and std for A, mean and std for B). Aggregating gives the expected four columns on a
    # two-level index whether or not a selection list is used.
    # agg with different hows
    # equivalent of using a selection list / or not
    expected = pd.concat([a_sum, a_std, b_mean, b_std], axis=1)
    expected.columns = pd.MultiIndex.from_tuples(
        [("A", "sum"), ("A", "std"), ("B", "mean"), ("B", "std")]
    )
    if cols is not None:
        obj = cases[cols]
    else:
        obj = cases

    result = obj.agg(agg)
    tm.assert_frame_equal(result, expected, check_like=True)


@pytest.mark.parametrize(
    "agg", [{"A": ["sum", "std"]}, {"A": ["sum", "std"], "B": ["mean", "std"]}]
)
def test_agg_specificationerror_series(cases, agg):
    # [FUNCTION DOC]
    # Parametrized by dictionaries of lists. Aggregating a single column (series) with a
    # dictionary raises the specification error saying nested renamer is not supported.
    msg = "nested renamer is not supported"

    # series like aggs
    with pytest.raises(pd.errors.SpecificationError, match=msg):
        cases["A"].agg(agg)


def test_agg_specificationerror_invalid_names(cases):
    # [FUNCTION DOC]
    # Selecting column A and aggregating with a dictionary that also names column B raises a key
    # error saying label B does not exist.
    # errors
    # invalid names in the agg specification
    msg = r"Label\(s\) \['B'\] do not exist"
    with pytest.raises(KeyError, match=msg):
        cases[["A"]].agg({"A": ["sum", "std"], "B": ["mean", "std"]})


def test_agg_nested_dicts():
    # [FUNCTION DOC]
    # Builds the four equivalent resamplers inline (not using the fixtures) and verifies that nested
    # dictionary aggregation specifications raise the specification error, in whole-frame form,
    # selected-columns form and plain form.
    index = date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D")
    index.name = "date"
    df = DataFrame(
        np.random.default_rng(2).random((10, 2)), columns=list("AB"), index=index
    )
    df_col = df.reset_index()
    df_mult = df_col.copy()
    df_mult.index = pd.MultiIndex.from_arrays(
        [range(10), df.index], names=["index", "date"]
    )
    r = df.resample("2D")
    cases = [
        r,
        df_col.resample("2D", on="date"),
        df_mult.resample("2D", level="date"),
        df.groupby(pd.Grouper(freq="2D")),
    ]

    msg = "nested renamer is not supported"
    for t in cases:
        with pytest.raises(pd.errors.SpecificationError, match=msg):
            t.aggregate({"r1": {"A": ["mean", "sum"]}, "r2": {"B": ["mean", "sum"]}})

    for t in cases:
        with pytest.raises(pd.errors.SpecificationError, match=msg):
            t[["A", "B"]].agg(
                {"A": {"ra": ["mean", "std"]}, "B": {"rb": ["mean", "std"]}}
            )

        with pytest.raises(pd.errors.SpecificationError, match=msg):
            t.agg({"A": {"ra": ["mean", "std"]}, "B": {"rb": ["mean", "std"]}})


def test_try_aggregate_non_existing_column():
    # [FUNCTION DOC]
    # Regression test (GH 16766): aggregating a thirty-minute resample with a dictionary naming
    # a non-existing column z raises a key error saying label z does not exist.
    # GH 16766
    data = [
        {"dt": datetime(2017, 6, 1, 0), "x": 1.0, "y": 2.0},
        {"dt": datetime(2017, 6, 1, 1), "x": 2.0, "y": 2.0},
        {"dt": datetime(2017, 6, 1, 2), "x": 3.0, "y": 1.5},
    ]
    df = DataFrame(data).set_index("dt")

    # Error as we don't have 'z' column
    msg = r"Label\(s\) \['z'\] do not exist"
    with pytest.raises(KeyError, match=msg):
        df.resample("30min").agg({"x": ["mean"], "y": ["median"], "z": ["sum"]})


def test_agg_list_like_func_with_args():
    # [FUNCTION DOC]
    # Regression test (issue 50624): aggregation with a list of two functions and extra
    # positional and keyword arguments. Passing a keyword accepted by only one of the functions
    # raises a type error naming the unexpected keyword for the first function. Passing arguments
    # accepted by both gives a two-level column frame with each function's result added to the
    # constants.
    # 50624
    df = DataFrame(
        {"x": [1, 2, 3]}, index=date_range("2020-01-01", periods=3, freq="D")
    )

    def foo1(x, a=1, c=0):
        # [FUNCTION DOC]
        # HELPER: nested in the list-like aggregation test. Parameters: a value series x, a defaulting
        # to one, c defaulting to zero. Returns x plus a plus c.
        return x + a + c

    def foo2(x, b=2, c=0):
        # [FUNCTION DOC]
        # HELPER: nested in the list-like aggregation test. Parameters: a value series x, b defaulting
        # to two, c defaulting to zero. Returns x plus b plus c.
        return x + b + c

    msg = r"foo1\(\) got an unexpected keyword argument 'b'"
    with pytest.raises(TypeError, match=msg):
        df.resample("D").agg([foo1, foo2], 3, b=3, c=4)

    result = df.resample("D").agg([foo1, foo2], 3, c=4)
    expected = DataFrame(
        [[8, 8], [9, 9], [10, 10]],
        index=date_range("2020-01-01", periods=3, freq="D"),
        columns=pd.MultiIndex.from_tuples([("x", "foo1"), ("x", "foo2")]),
    )
    tm.assert_frame_equal(result, expected)


def test_selection_api_validation():
    # [FUNCTION DOC]
    # Regression test (GH 13500): validation of the on and level arguments. Level naming a
    # non-datetime level raises a type error (only valid with datetime-like indexes); passing both
    # on and level raises a value error; an unhashable on raises a type error; a list of
    # missing levels raises a key error; upsampling from level or on selection with frequency
    # conversion raises a value error recommending setting the index. Summing with on gives the
    # same as the plain date-indexed resample with the date name; summing with level raises a
    # type error for datetime sums unless numeric-only is enabled, after which it equals the
    # expected.
    # GH 13500
    index = date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D")

    rng = np.arange(len(index), dtype=np.int64)
    df = DataFrame(
        {"date": index, "a": rng},
        index=pd.MultiIndex.from_arrays([rng, index], names=["v", "d"]),
    )
    df_exp = DataFrame({"a": rng}, index=index)

    # non DatetimeIndex
    msg = (
        "Only valid with DatetimeIndex, TimedeltaIndex or PeriodIndex, "
        "but got an instance of 'Index'"
    )
    with pytest.raises(TypeError, match=msg):
        df.resample("2D", level="v")

    msg = "The Grouper cannot specify both a key and a level!"
    with pytest.raises(ValueError, match=msg):
        df.resample("2D", on="date", level="d")

    msg = "unhashable type: 'list'"
    with pytest.raises(TypeError, match=msg):
        df.resample("2D", on=["a", "date"])

    msg = r"\"Level \['a', 'date'\] not found\""
    with pytest.raises(KeyError, match=msg):
        df.resample("2D", level=["a", "date"])

    # upsampling not allowed
    msg = (
        "Upsampling from level= or on= selection is not supported, use "
        r"\.set_index\(\.\.\.\) to explicitly set index to datetime-like"
    )
    with pytest.raises(ValueError, match=msg):
        df.resample("2D", level="d").asfreq()
    with pytest.raises(ValueError, match=msg):
        df.resample("2D", on="date").asfreq()

    exp = df_exp.resample("2D").sum()
    exp.index.name = "date"
    result = df.resample("2D", on="date").sum()
    tm.assert_frame_equal(exp, result)

    exp.index.name = "d"
    with pytest.raises(
        TypeError, match="datetime64 type does not support operation 'sum'"
    ):
        df.resample("2D", level="d").sum()
    result = df.resample("2D", level="d").sum(numeric_only=True)
    tm.assert_frame_equal(exp, result)


@pytest.mark.parametrize(
    "col_name", ["t2", "t2x", "t2q", "T_2M", "t2p", "t2m", "t2m1", "T2M"]
)
def test_agg_with_datetime_index_list_agg_func(col_name):
    # [FUNCTION DOC]
    # Regression test (GH 22660), parametrized by column names that look like dates. Aggregating
    # a timezone-aware frame with a list containing the mean works for column names that a date
    # parser would mistake for dates (errors during parsing are caught internally), giving three
    # daily means on a two-level column index.
    # GH 22660
    # The parametrized column names would get converted to dates by our
    # date parser. Some would result in OutOfBoundsError (ValueError) while
    # others would result in OverflowError when passed into Timestamp.
    # We catch these errors and move on to the correct branch.
    df = DataFrame(
        list(range(200)),
        index=date_range(
            start="2017-01-01", freq="15min", periods=200, tz="Europe/Berlin"
        ),
        columns=[col_name],
    )
    result = df.resample("1D").aggregate(["mean"])
    expected = DataFrame(
        [47.5, 143.5, 195.5],
        index=date_range(start="2017-01-01", freq="D", periods=3, tz="Europe/Berlin"),
        columns=pd.MultiIndex(levels=[[col_name], ["mean"]], codes=[[0], [0]]),
    )
    tm.assert_frame_equal(result, expected)


def test_resample_agg_readonly():
    # [FUNCTION DOC]
    # Regression test (GH#31710): a series backed by a read-only array of zeros indexed by hours
    # can be resampled daily and aggregated with last, first, max and min, giving epoch timestamps
    # at two day starts; the index frequency is set to a calendar day (GH#41943).
    # GH#31710 cython needs to allow readonly data
    index = date_range("2020-01-01", "2020-01-02", freq="1h", unit="ns")
    arr = np.zeros_like(index)
    arr.setflags(write=False)

    ser = Series(arr, index=index)
    rs = ser.resample("1D")

    expected = Series([pd.Timestamp(0), pd.Timestamp(0)], index=index[::24])
    expected.index.freq = Day(1)  # GH#41943 no longer equivalent to 24h

    result = rs.agg("last")
    tm.assert_series_equal(result, expected)

    result = rs.agg("first")
    tm.assert_series_equal(result, expected)

    result = rs.agg("max")
    tm.assert_series_equal(result, expected)

    result = rs.agg("min")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "start,end,freq,data,resample_freq,origin,closed,exp_data,exp_end,exp_periods",
    [
        (
            "2000-10-01 23:30:00",
            "2000-10-02 00:26:00",
            "7min",
            [0, 3, 6, 9, 12, 15, 18, 21, 24],
            "17min",
            "end",
            None,
            [0, 18, 27, 63],
            "20001002 00:26:00",
            4,
        ),
        (
            "20200101 8:26:35",
            "20200101 9:31:58",
            "77s",
            [1] * 51,
            "7min",
            "end",
            "right",
            [1, 6, 5, 6, 5, 6, 5, 6, 5, 6],
            "2020-01-01 09:30:45",
            10,
        ),
        (
            "2000-10-01 23:30:00",
            "2000-10-02 00:26:00",
            "7min",
            [0, 3, 6, 9, 12, 15, 18, 21, 24],
            "17min",
            "end",
            "left",
            [0, 18, 27, 39, 24],
            "20001002 00:43:00",
            5,
        ),
        (
            "2000-10-01 23:30:00",
            "2000-10-02 00:26:00",
            "7min",
            [0, 3, 6, 9, 12, 15, 18, 21, 24],
            "17min",
            "end_day",
            None,
            [3, 15, 45, 45],
            "2000-10-02 00:29:00",
            4,
        ),
    ],
)
def test_end_and_end_day_origin(
    start,
    end,
    freq,
    data,
    resample_freq,
    origin,
    closed,
    exp_data,
    exp_end,
    exp_periods,
):
    # [FUNCTION DOC]
    # Parametrized by start, end, frequency, input data, resample frequency, origin (end or end of
    # day), closed side, expected data, expected end and number of expected periods. Resamples
    # the generated series with the given origin and closed side and sums; the result must equal
    # the expected values on a date range ending at the expected end.
    rng = date_range(start, end, freq=freq)
    ts = Series(data, index=rng)

    res = ts.resample(resample_freq, origin=origin, closed=closed).sum()
    expected = Series(
        exp_data,
        index=date_range(end=exp_end, freq=resample_freq, periods=exp_periods),
    )

    tm.assert_series_equal(res, expected)


@pytest.mark.parametrize(
    # expected_data is a string when op raises a ValueError
    "method, numeric_only, expected_data",
    [
        ("sum", True, {"num": [25]}),
        ("sum", False, {"cat": ["cat_1cat_2"], "num": [25]}),
        ("sum", lib.no_default, {"cat": ["cat_1cat_2"], "num": [25]}),
        ("prod", True, {"num": [100]}),
        ("prod", False, "can't multiply sequence"),
        ("prod", lib.no_default, "can't multiply sequence"),
        ("min", True, {"num": [5]}),
        ("min", False, {"cat": ["cat_1"], "num": [5]}),
        ("min", lib.no_default, {"cat": ["cat_1"], "num": [5]}),
        ("max", True, {"num": [20]}),
        ("max", False, {"cat": ["cat_2"], "num": [20]}),
        ("max", lib.no_default, {"cat": ["cat_2"], "num": [20]}),
        ("first", True, {"num": [5]}),
        ("first", False, {"cat": ["cat_1"], "num": [5]}),
        ("first", lib.no_default, {"cat": ["cat_1"], "num": [5]}),
        ("last", True, {"num": [20]}),
        ("last", False, {"cat": ["cat_2"], "num": [20]}),
        ("last", lib.no_default, {"cat": ["cat_2"], "num": [20]}),
        ("mean", True, {"num": [12.5]}),
        ("mean", False, "Could not convert"),
        ("mean", lib.no_default, "Could not convert"),
        ("median", True, {"num": [12.5]}),
        ("median", False, r"Cannot convert \['cat_1' 'cat_2'\] to numeric"),
        ("median", lib.no_default, r"Cannot convert \['cat_1' 'cat_2'\] to numeric"),
        ("std", True, {"num": [10.606601717798213]}),
        ("std", False, "could not convert string to float"),
        ("std", lib.no_default, "could not convert string to float"),
        ("var", True, {"num": [112.5]}),
        ("var", False, "could not convert string to float"),
        ("var", lib.no_default, "could not convert string to float"),
        ("sem", True, {"num": [7.5]}),
        ("sem", False, "could not convert string to float"),
        ("sem", lib.no_default, "could not convert string to float"),
    ],
)
def test_frame_downsample_method(
    method, numeric_only, expected_data, using_infer_string
):
    # [FUNCTION DOC]
    # Regression test (GH#46442), parametrized by reduction name, numeric-only (true, false or
    # default) and expected data. Checks that numeric-only behaves as expected for the frame
    # resampler at yearly frequency on a string and an integer column. Expected data is a mapping
    # for success (a string column result for sum, min, max, first, last when strings are allowed)
    # or a message string for failure.
    # KNOWN ISSUE: for variance, mean, median and product the message string in the parametrization is never used, because the test replaces it with the aggregation-failure message or the unsupported-operation message; those parameter strings are dead data.
    # GH#46442 test if `numeric_only` behave as expected for DataFrameGroupBy

    index = date_range("2018-01-01", periods=2, freq="D")
    expected_index = date_range("2018-12-31", periods=1, freq="YE")
    df = DataFrame({"cat": ["cat_1", "cat_2"], "num": [5, 20]}, index=index)
    resampled = df.resample("YE")
    if numeric_only is lib.no_default:
        kwargs = {}
    else:
        kwargs = {"numeric_only": numeric_only}

    func = getattr(resampled, method)
    if isinstance(expected_data, str):
        if method in ("var", "mean", "median", "prod"):
            klass = TypeError
            msg = re.escape(f"agg function failed [how->{method},dtype->")
            if using_infer_string:
                msg = f"dtype 'str' does not support operation '{method}'"
        elif method in ["sum", "std", "sem"] and using_infer_string:
            klass = TypeError
            msg = f"dtype 'str' does not support operation '{method}'"
        else:
            klass = ValueError
            msg = expected_data
        with pytest.raises(klass, match=msg):
            _ = func(**kwargs)
    else:
        result = func(**kwargs)
        expected = DataFrame(expected_data, index=expected_index)
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "method, numeric_only, expected_data",
    [
        ("sum", True, ()),
        ("sum", False, ["cat_1cat_2"]),
        ("sum", lib.no_default, ["cat_1cat_2"]),
        ("prod", True, ()),
        ("prod", False, ()),
        ("prod", lib.no_default, ()),
        ("min", True, ()),
        ("min", False, ["cat_1"]),
        ("min", lib.no_default, ["cat_1"]),
        ("max", True, ()),
        ("max", False, ["cat_2"]),
        ("max", lib.no_default, ["cat_2"]),
        ("first", True, ()),
        ("first", False, ["cat_1"]),
        ("first", lib.no_default, ["cat_1"]),
        ("last", True, ()),
        ("last", False, ["cat_2"]),
        ("last", lib.no_default, ["cat_2"]),
    ],
)
def test_series_downsample_method(
    method, numeric_only, expected_data, using_infer_string
):
    # [FUNCTION DOC]
    # Regression test (GH#46442), parametrized by reduction name, numeric-only flag and expected
    # values. Series resampler at yearly frequency on a series of strings: numeric-only true
    # raises a type error saying it cannot be used with the series group-by object; product raises a
    # type error (aggregation failure, or unsupported operation for inferred strings); other
    # reductions return a one-element series on a year-end index.
    # GH#46442 test if `numeric_only` behave as expected for SeriesGroupBy

    index = date_range("2018-01-01", periods=2, freq="D")
    expected_index = date_range("2018-12-31", periods=1, freq="YE")
    df = Series(["cat_1", "cat_2"], index=index)
    resampled = df.resample("YE")
    kwargs = {} if numeric_only is lib.no_default else {"numeric_only": numeric_only}

    func = getattr(resampled, method)
    if numeric_only and numeric_only is not lib.no_default:
        msg = rf"Cannot use numeric_only=True with SeriesGroupBy\.{method}"
        with pytest.raises(TypeError, match=msg):
            func(**kwargs)
    elif method == "prod":
        msg = re.escape("agg function failed [how->prod,dtype->")
        if using_infer_string:
            msg = "dtype 'str' does not support operation 'prod'"
        with pytest.raises(TypeError, match=msg):
            func(**kwargs)

    else:
        result = func(**kwargs)
        expected = Series(expected_data, index=expected_index)
        tm.assert_series_equal(result, expected)


def test_resample_empty():
    # [FUNCTION DOC]
    # Regression test (GH#52484): resampling a frame with no columns and three timestamps to
    # eight hours with mean gives an empty-column frame with the filled four-point index.
    # GH#52484
    df = DataFrame(
        index=pd.to_datetime(
            ["2018-01-01 00:00:00", "2018-01-01 12:00:00", "2018-01-02 00:00:00"]
        )
    )
    expected = DataFrame(
        index=pd.to_datetime(
            [
                "2018-01-01 00:00:00",
                "2018-01-01 08:00:00",
                "2018-01-01 16:00:00",
                "2018-01-02 00:00:00",
            ]
        )
    )
    result = df.resample("8h").mean()
    tm.assert_frame_equal(result, expected)


def test_asfreq_respects_origin_with_fixed_freq_all_seconds_equal():
    # [FUNCTION DOC]
    # Regression test (GH#62725): frequency conversion of a resampler with minute frequency and
    # origin start of day, where all timestamps have the same seconds value, respects the
    # origin: the index is floored to whole minutes with minute frequency and the values are all
    # missing (no original timestamp falls on the grid).
    # GH#62725: Ensure Resampler.asfreq respects origin="start_day"
    # when all datetimes share identical seconds values.
    idx = [
        datetime(2025, 10, 17, 17, 15, 10),
        datetime(2025, 10, 17, 17, 16, 10),
        datetime(2025, 10, 17, 17, 17, 10),
    ]
    df = DataFrame({"value": [0, 1, 2]}, index=idx)

    result = df.resample("1min", origin="start_day").asfreq()

    # Expected index: list of Timestamps, matching dtype
    exp_idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-10-17 17:15:00"),
            pd.Timestamp("2025-10-17 17:16:00"),
            pd.Timestamp("2025-10-17 17:17:00"),
        ],
        dtype=result.index.dtype,
        freq="min",
    )

    exp = DataFrame({"value": [np.nan, np.nan, np.nan]}, index=exp_idx)
    tm.assert_frame_equal(result, exp)
