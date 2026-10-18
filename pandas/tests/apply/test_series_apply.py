# [MODULE DOC]
# Tests for Series.apply and Series.agg (and the by_row option) covering boxing of datetime-like
# values, categoricals, list-like and dict-like arguments, reducers versus transformers, lambdas,
# argument passing and failures in transform. Many tests are parametrized by the by_row fixture, which
# is either false (whole-series evaluation) or the compat mode (element-wise first). Lambdas used inline
# as test functions are not documented separately.

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    concat,
    date_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels


@pytest.fixture(params=[False, "compat"])
def by_row(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over false and the compat string for the by_row option. Returns the parameter.
    return request.param


def test_series_map_box_timedelta(by_row):
    # [FUNCTION DOC]
    # GH#11349: apply on a series of timedeltas with a function converting to seconds; the function acts per
    # element when by_row is truthy and on the whole series via its accessor otherwise. Result must equal the
    # map result and the literal series of seconds.
    # GH#11349
    ser = Series(timedelta_range("1 day 1 s", periods=3, freq="h"))

    def f(x):
        # [FUNCTION DOC]
        # HELPER: returns total seconds of the element (compat) or of the whole series through the datetime
        # accessor (not by row).
        return x.total_seconds() if by_row else x.dt.total_seconds()

    result = ser.apply(f, by_row=by_row)

    expected = ser.map(lambda x: x.total_seconds())
    tm.assert_series_equal(result, expected)

    expected = Series([86401.0, 90001.0, 93601.0])
    tm.assert_series_equal(result, expected)


def test_apply(datetime_series, by_row):
    # [FUNCTION DOC]
    # Apply of numpy square root and exponential on the datetime series fixture matches the direct numpy
    # results. Also applies identity to an empty object series and a series with index but no data, requiring
    # equality; for the empty case also asserts the result is a new object, but shares the same index object,
    # dtype and name.
    result = datetime_series.apply(np.sqrt, by_row=by_row)
    with np.errstate(all="ignore"):
        expected = np.sqrt(datetime_series)
    tm.assert_series_equal(result, expected)

    # element-wise apply (ufunc)
    result = datetime_series.apply(np.exp, by_row=by_row)
    expected = np.exp(datetime_series)
    tm.assert_series_equal(result, expected)

    # empty series
    s = Series(dtype=object, name="foo", index=Index([], name="bar"))
    rs = s.apply(lambda x: x, by_row=by_row)
    tm.assert_series_equal(s, rs)

    # check all metadata (GH 9322)
    assert s is not rs
    assert s.index is rs.index
    assert s.dtype == rs.dtype
    assert s.name == rs.name

    # index but no data
    s = Series(index=[1, 2, 3], dtype=np.float64)
    rs = s.apply(lambda x: x, by_row=by_row)
    tm.assert_series_equal(s, rs)


def test_apply_map_same_length_inference_bug():
    # [FUNCTION DOC]
    # Series apply (compat) with a function returning a tuple must equal map with the same function:
    # checks the inference bug when result length equals the input length.
    s = Series([1, 2])

    def f(x):
        # [FUNCTION DOC]
        # HELPER: returns a tuple of the value and the value plus one.
        return (x, x + 1)

    result = s.apply(f, by_row="compat")
    expected = s.map(f)
    tm.assert_series_equal(result, expected)


def test_apply_args():
    # [FUNCTION DOC]
    # Apply with the string split method and a positional args tuple containing a comma splits the
    # string into a list; the result element is a list with the two parts.
    s = Series(["foo,bar"])

    result = s.apply(str.split, args=(",",))
    assert result[0] == ["foo", "bar"]
    assert isinstance(result[0], list)


@pytest.mark.parametrize(
    "args, kwargs, increment",
    [((), {}, 0), ((), {"a": 1}, 1), ((2, 3), {}, 32), ((1,), {"c": 2}, 201)],
)
def test_agg_args(args, kwargs, increment):
    # [FUNCTION DOC]
    # GH 43357: Series agg with a function taking extra positional and keyword arguments. Parametrized by
    # args, kwargs and increment; expected equals the series plus the increment.
    # GH 43357
    def f(x, a=0, b=0, c=0):
        # [FUNCTION DOC]
        # HELPER: adds the first argument, a, ten times b and a hundred times c (defaults zero).
        return x + a + 10 * b + 100 * c

    s = Series([1, 2])
    result = s.agg(f, 0, *args, **kwargs)
    expected = s + increment
    tm.assert_series_equal(result, expected)


def test_agg_mapping_func_deprecated():
    # [FUNCTION DOC]
    # GH 53325: calls agg with a function, a list of functions and a dict of functions, all with extra
    # positional and keyword arguments. Only verifies that none of these raise; no result or warning is checked.
    # KNOWN ISSUE: the name says deprecated but the test asserts nothing about a warning or about the result.
    # GH 53325
    s = Series([1, 2, 3])

    def foo1(x, a=1, c=0):
        # [FUNCTION DOC]
        # HELPER: adds a and c to the input.
        return x + a + c

    def foo2(x, b=2, c=0):
        # [FUNCTION DOC]
        # HELPER: adds b and c to the input.
        return x + b + c

    s.agg(foo1, 0, 3, c=4)
    s.agg([foo1, foo2], 0, 3, c=4)
    s.agg({"a": foo1, "b": foo2}, 0, 3, c=4)


def test_series_apply_map_box_timestamps(by_row):
    # [FUNCTION DOC]
    # GH#2689, GH#2627: apply on a datetime series with a function reading hour, day and month of each
    # element. With by_row false, an attribute error that the series has no hour attribute is expected; in compat
    # mode the result must equal map with the same function.
    # GH#2689, GH#2627
    ser = Series(date_range("1/1/2000", periods=10))

    def func(x):
        # [FUNCTION DOC]
        # HELPER: returns a tuple of the hour, day and month of its argument.
        return (x.hour, x.day, x.month)

    if not by_row:
        msg = "Series' object has no attribute 'hour'"
        with pytest.raises(AttributeError, match=msg):
            ser.apply(func, by_row=by_row)
        return

    result = ser.apply(func, by_row=by_row)
    expected = ser.map(func)
    tm.assert_series_equal(result, expected)


def test_apply_box_dt64():
    # [FUNCTION DOC]
    # Applying a formatting function to a datetime64 series in compat mode must receive boxed Timestamp values
    # (no timezone), giving strings of type name, day and tz.
    # ufunc will not be boxed. Same test cases as the test_map_box
    vals = [pd.Timestamp("2011-01-01"), pd.Timestamp("2011-01-02")]
    ser = Series(vals, dtype="M8[ns]")
    assert ser.dtype == "datetime64[ns]"
    # boxed value must be Timestamp instance
    res = ser.apply(lambda x: f"{type(x).__name__}_{x.day}_{x.tz}", by_row="compat")
    exp = Series(["Timestamp_1_None", "Timestamp_2_None"])
    tm.assert_series_equal(res, exp)


def test_apply_box_dt64tz():
    # [FUNCTION DOC]
    # Same as the datetime test but for a timezone-aware series (US/Eastern); the boxed element keeps its timezone.
    vals = [
        pd.Timestamp("2011-01-01", tz="US/Eastern"),
        pd.Timestamp("2011-01-02", tz="US/Eastern"),
    ]
    ser = Series(vals, dtype="M8[ns, US/Eastern]")
    assert ser.dtype == "datetime64[ns, US/Eastern]"
    res = ser.apply(lambda x: f"{type(x).__name__}_{x.day}_{x.tz}", by_row="compat")
    exp = Series(["Timestamp_1_US/Eastern", "Timestamp_2_US/Eastern"])
    tm.assert_series_equal(res, exp)


def test_apply_box_td64():
    # [FUNCTION DOC]
    # Applying to a timedelta series: elements are boxed as Timedelta (the dtype asserted is the
    # microsecond unit) and the function sees the type name and days.
    # timedelta
    vals = [pd.Timedelta("1 days"), pd.Timedelta("2 days")]
    ser = Series(vals)
    assert ser.dtype == "timedelta64[us]"
    res = ser.apply(lambda x: f"{type(x).__name__}_{x.days}", by_row="compat")
    exp = Series(["Timedelta_1", "Timedelta_2"])
    tm.assert_series_equal(res, exp)


def test_apply_box_period():
    # [FUNCTION DOC]
    # Applying to a period series (monthly): elements are boxed as Period; the function sees type name and
    # frequency string.
    # period
    vals = [pd.Period("2011-01-01", freq="M"), pd.Period("2011-01-02", freq="M")]
    ser = Series(vals)
    assert ser.dtype == "Period[M]"
    res = ser.apply(lambda x: f"{type(x).__name__}_{x.freqstr}", by_row="compat")
    exp = Series(["Period_M", "Period_M"])
    tm.assert_series_equal(res, exp)


def test_apply_datetimetz(by_row):
    # [FUNCTION DOC]
    # Apply on a Tokyo timezone-aware datetime series: adding a day offset per element gives the shifted
    # series; the hour yields int64 per-row (compat) or int32 via the accessor otherwise (with a final hour wrap
    # to zero); and a time zone string function returns per-element strings (compat) or a single string (not by row).
    values = date_range("2011-01-01", "2011-01-02", freq="h").tz_localize("Asia/Tokyo")
    s = Series(values, name="XX")

    result = s.apply(lambda x: x + pd.offsets.Day(), by_row=by_row)
    exp_values = date_range("2011-01-02", "2011-01-03", freq="h").tz_localize(
        "Asia/Tokyo"
    )
    exp = Series(exp_values, name="XX")
    tm.assert_series_equal(result, exp)

    result = s.apply(lambda x: x.hour if by_row else x.dt.hour, by_row=by_row)
    exp = Series([*list(range(24)), 0], name="XX", dtype="int64" if by_row else "int32")
    tm.assert_series_equal(result, exp)

    # not vectorized
    def f(x):
        # [FUNCTION DOC]
        # HELPER: returns the timezone string from the element (compat) or from the whole series accessor otherwise.
        return str(x.tz) if by_row else str(x.dt.tz)

    result = s.apply(f, by_row=by_row)
    if by_row:
        exp = Series(["Asia/Tokyo"] * 25, name="XX")
        tm.assert_series_equal(result, exp)
    else:
        assert result == "Asia/Tokyo"


def test_apply_categorical(by_row, using_infer_string):
    # [FUNCTION DOC]
    # Apply on an ordered categorical series. Not by row: lower-casing raises an attribute error for the series
    # and a constant function returns the scalar. Compat: lower-casing returns a series with the same categories
    # order mapped to lower case, preserving the categorical dtype and values; a constant function returns an object
    # (or string, depending on the infer-string option) series.
    # KNOWN ISSUE: the final assertion combines an equality with a conditional expression, so under the infer-string option it only asserts a non-empty string constant, which is always true, rather than comparing the dtype.
    values = pd.Categorical(list("ABBABCD"), categories=list("DCBA"), ordered=True)
    ser = Series(values, name="XX", index=list("abcdefg"))

    if not by_row:
        msg = "Series' object has no attribute 'lower"
        with pytest.raises(AttributeError, match=msg):
            ser.apply(lambda x: x.lower(), by_row=by_row)
        assert ser.apply(lambda x: "A", by_row=by_row) == "A"
        return

    result = ser.apply(lambda x: x.lower(), by_row=by_row)

    # should be categorical dtype when the number of categories are
    # the same
    values = pd.Categorical(list("abbabcd"), categories=list("dcba"), ordered=True)
    exp = Series(values, name="XX", index=list("abcdefg"))
    tm.assert_series_equal(result, exp)
    tm.assert_categorical_equal(result.values, exp.values)

    result = ser.apply(lambda x: "A")
    exp = Series(["A"] * 7, name="XX", index=list("abcdefg"))
    tm.assert_series_equal(result, exp)
    assert result.dtype == object if not using_infer_string else "str"


@pytest.mark.parametrize("series", [["1-1", "1-1", np.nan], ["1-1", "1-2", np.nan]])
def test_apply_categorical_with_nan_values(series, by_row):
    # [FUNCTION DOC]
    # GH 20714 / GH 24275 / GH 59966: apply on a categorical series with NaN, parametrized by two value
    # sets. Not by row: calling a string method raises an attribute error. Compat: a function splitting on a dash
    # or returning false for missing, cast to object, must equal the expected categorical (cast to object).
    # GH 20714 bug fixed in: GH 24275
    s = Series(series, dtype="category")
    if not by_row:
        msg = "'Series' object has no attribute 'split'"
        with pytest.raises(AttributeError, match=msg):
            s.apply(lambda x: x.split("-")[0], by_row=by_row)
        return
    # NaN for cat dtype fixed in (GH 59966)
    result = s.apply(lambda x: x.split("-")[0] if pd.notna(x) else False, by_row=by_row)
    result = result.astype(object)
    expected = Series(["1", "1", False], dtype="category")
    expected = expected.astype(object)
    tm.assert_series_equal(result, expected)


def test_apply_empty_integer_series_with_datetime_index(by_row):
    # [FUNCTION DOC]
    # GH 21245: applying identity to an empty integer series with an empty datetime index returns an equal series.
    # GH 21245
    s = Series([], index=date_range(start="2018-01-01", periods=0), dtype=int)
    result = s.apply(lambda x: x, by_row=by_row)
    tm.assert_series_equal(result, s)


def test_apply_dataframe_iloc():
    # [FUNCTION DOC]
    # Applies, per element of an index series, a function that picks the numbers value at the element's
    # row from a second frame passed through args. Expected is a uint64 series named Indices.
    uintDF = DataFrame(np.uint64([1, 2, 3, 4, 5]), columns=["Numbers"])
    indexDF = DataFrame([2, 3, 2, 1, 2], columns=["Indices"])

    def retrieve(targetRow, targetDF):
        # [FUNCTION DOC]
        # HELPER: returns the entry of the Numbers column of the given frame at the given position.
        val = targetDF["Numbers"].iloc[targetRow]
        return val

    result = indexDF["Indices"].apply(retrieve, args=(uintDF,))
    expected = Series([3, 4, 3, 2, 3], name="Indices", dtype="uint64")
    tm.assert_series_equal(result, expected)


def test_transform(string_series, by_row):
    # [FUNCTION DOC]
    # Transforming functions via apply: a ufunc gives the transformed series; a list of one (callable or
    # name) gives a one-column frame named sqrt; a list of two gives two columns (sqrt, absolute); a dict with
    # renaming gives a series with a MultiIndex of new name and original label. Compared after reindexing
    # like the expected. Floating point errors are ignored.
    # transforming functions

    with np.errstate(all="ignore"):
        f_sqrt = np.sqrt(string_series)
        f_abs = np.abs(string_series)

        # ufunc
        result = string_series.apply(np.sqrt, by_row=by_row)
        expected = f_sqrt.copy()
        tm.assert_series_equal(result, expected)

        # list-like
        result = string_series.apply([np.sqrt], by_row=by_row)
        expected = f_sqrt.to_frame().copy()
        expected.columns = ["sqrt"]
        tm.assert_frame_equal(result, expected)

        result = string_series.apply(["sqrt"], by_row=by_row)
        tm.assert_frame_equal(result, expected)

        # multiple items in list
        # these are in the order as if we are applying both functions per
        # series and then concatting
        expected = concat([f_sqrt, f_abs], axis=1)
        expected.columns = ["sqrt", "absolute"]
        result = string_series.apply([np.sqrt, np.abs], by_row=by_row)
        tm.assert_frame_equal(result, expected)

        # dict, provide renaming
        expected = concat([f_sqrt, f_abs], axis=1)
        expected.columns = ["foo", "bar"]
        expected = expected.unstack().rename("series")

        result = string_series.apply({"foo": np.sqrt, "bar": np.abs}, by_row=by_row)
        tm.assert_series_equal(result.reindex_like(expected), expected)


@pytest.mark.parametrize("op", series_transform_kernels)
def test_transform_partial_failure(op, request):
    # [FUNCTION DOC]
    # GH 35964: parametrized over the series transform kernels; the fill-forward, fill-backward and shift
    # kernels are expected failures because they work on any dtype. Series of objects: list/dict forms combining the
    # op with shift must raise a value error (transform failed) for fillna and ngroup, or a type error about
    # unsupported operands otherwise.
    # GH 35964
    if op in ("ffill", "bfill", "shift"):
        request.applymarker(
            pytest.mark.xfail(reason=f"{op} is successful on any dtype")
        )

    # Using object makes most transform kernels fail
    ser = Series(3 * [object])

    if op in ("fillna", "ngroup"):
        error = ValueError
        msg = "Transform function failed"
    else:
        error = TypeError
        msg = "|".join(
            [
                "not supported between instances of 'type' and 'type'",
                "unsupported operand type",
            ]
        )

    with pytest.raises(error, match=msg):
        ser.transform([op, "shift"])

    with pytest.raises(error, match=msg):
        ser.transform({"A": op, "B": "shift"})

    with pytest.raises(error, match=msg):
        ser.transform({"A": [op], "B": ["shift"]})

    with pytest.raises(error, match=msg):
        ser.transform({"A": [op, "shift"], "B": [op]})


def test_transform_partial_failure_valueerror():
    # [FUNCTION DOC]
    # GH 40211: when one of the functions raises a value error, transform with list, dict, dict of lists or mixed
    # must raise a value error that the transform function failed.
    # GH 40211
    def noop(x):
        # [FUNCTION DOC]
        # HELPER: returns its argument unchanged.
        return x

    def raising_op(_):
        # [FUNCTION DOC]
        # HELPER: ignores its argument and raises a value error.
        raise ValueError

    ser = Series(3 * [object])
    msg = "Transform function failed"

    with pytest.raises(ValueError, match=msg):
        ser.transform([noop, raising_op])

    with pytest.raises(ValueError, match=msg):
        ser.transform({"A": raising_op, "B": noop})

    with pytest.raises(ValueError, match=msg):
        ser.transform({"A": [raising_op], "B": [noop]})

    with pytest.raises(ValueError, match=msg):
        ser.transform({"A": [noop, raising_op], "B": [noop]})


def test_demo():
    # [FUNCTION DOC]
    # Demonstration test: agg of min and max gives a series indexed by those names named like the original; agg with a
    # dict gives a one-element series indexed by the dict key.
    # demonstration tests
    s = Series(range(6), dtype="int64", name="series")

    result = s.agg(["min", "max"])
    expected = Series([0, 5], index=["min", "max"], name="series")
    tm.assert_series_equal(result, expected)

    result = s.agg({"foo": "min"})
    expected = Series([0], index=["foo"], name="series")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", [str, lambda x: str(x)])
def test_apply_map_evaluate_lambdas_the_same(string_series, func, by_row, engine):
    # [FUNCTION DOC]
    # Parametrized over the str builtin and an equivalent lambda. With compat the result equals map with the same function and
    # engine fixture; otherwise the function evaluates on the whole series so the result equals the string form of the
    # series.
    # test that we are evaluating row-by-row first if by_row="compat"
    # else vectorized evaluation
    result = string_series.apply(func, by_row=by_row)

    if by_row:
        expected = string_series.map(func, engine=engine)
        tm.assert_series_equal(result, expected)
    else:
        assert result == str(string_series)


def test_agg_evaluate_lambdas(string_series):
    # [FUNCTION DOC]
    # GH53325: agg with a function (lambda or builtin) that returns the type of its argument must receive the
    # whole series, so the result is the Series class itself.
    # GH53325
    result = string_series.agg(lambda x: type(x))
    assert result is Series

    result = string_series.agg(type)
    assert result is Series


@pytest.mark.parametrize("op_name", ["agg", "apply"])
def test_with_nested_series(datetime_series, op_name):
    # [FUNCTION DOC]
    # GH 2316 and GH52123: agg or apply (parametrized) with a function returning a series of x and its
    # square. For apply the result is a frame with columns x and x^2 holding the series and its square; for agg
    # a series of two series indexed by those names.
    # GH 2316 & GH52123
    # .agg with a reducer and a transform, what to do
    result = getattr(datetime_series, op_name)(
        lambda x: Series([x, x**2], index=["x", "x^2"])
    )
    if op_name == "apply":
        expected = DataFrame({"x": datetime_series, "x^2": datetime_series**2})
        tm.assert_frame_equal(result, expected)
    else:
        expected = Series([datetime_series, datetime_series**2], index=["x", "x^2"])
        tm.assert_series_equal(result, expected)


def test_replicate_describe(string_series):
    # [FUNCTION DOC]
    # Applying a dict of statistic names (some as callables for quantiles) to the string series must reproduce
    # the describe output (a result consisting entirely of scalars).
    # this also tests a result set that is all scalars
    expected = string_series.describe()
    result = string_series.apply(
        {
            "count": "count",
            "mean": "mean",
            "std": "std",
            "min": "min",
            "25%": lambda x: x.quantile(0.25),
            "50%": "median",
            "75%": lambda x: x.quantile(0.75),
            "max": "max",
        },
    )
    tm.assert_series_equal(result, expected)


def test_reduce(string_series):
    # [FUNCTION DOC]
    # Reducing a series with a list of two names (sum, mean) gives a series with those labels, keeping the name.
    # reductions with named functions
    result = string_series.agg(["sum", "mean"])
    expected = Series(
        [string_series.sum(), string_series.mean()],
        ["sum", "mean"],
        name=string_series.name,
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "how, kwds",
    [("agg", {}), ("apply", {"by_row": "compat"}), ("apply", {"by_row": False})],
)
def test_non_callable_aggregates(how, kwds):
    # [FUNCTION DOC]
    # GH 39116: non-callable attributes as functions. Parametrized by agg, apply compat and apply not by row. A
    # string naming the size attribute returns the size; a list of size, count and mean gives a float series; a dict
    # of the same gives the same series.
    # test agg using non-callable series attributes
    # GH 39116 - expand to apply
    s = Series([1, 2, None])

    # Calling agg w/ just a string arg same as calling s.arg
    result = getattr(s, how)("size", **kwds)
    expected = s.size
    assert result == expected

    # test when mixed w/ callable reducers
    result = getattr(s, how)(["size", "count", "mean"], **kwds)
    expected = Series({"size": 3.0, "count": 2.0, "mean": 1.5})
    tm.assert_series_equal(result, expected)

    result = getattr(s, how)({"size": "size", "count": "count", "mean": "mean"}, **kwds)
    tm.assert_series_equal(result, expected)


def test_series_apply_no_suffix_index(by_row):
    # [FUNCTION DOC]
    # GH36189: a list with the sum name and two lambdas gives a series of three twelve values with the labels
    # sum and two repeated lambda labels (no suffix added).
    # GH36189
    s = Series([4] * 3)
    result = s.apply(["sum", lambda x: x.sum(), lambda x: x.sum()], by_row=by_row)
    expected = Series([12, 12, 12], index=["sum", "<lambda>", "<lambda>"])

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "dti,exp",
    [
        (
            Series([1, 2], index=pd.DatetimeIndex([0, 31536000000])),
            DataFrame(np.repeat([[1, 2]], 2, axis=0), dtype="int64"),
        ),
        (
            Series(
                np.arange(10, dtype=np.float64),
                index=date_range("2020-01-01", periods=10),
                name="ts",
            ),
            DataFrame(np.repeat([[1, 2]], 10, axis=0), dtype="int64"),
        ),
    ],
)
@pytest.mark.parametrize("aware", [True, False])
def test_apply_series_on_date_time_index_aware_series(dti, exp, aware):
    # [FUNCTION DOC]
    # GH 25959: applying a function returning a two-element series over a series built from an index
    # (optionally localized to UTC), for two datetime-indexed inputs. Expected is a frame of repeated one,
    # two values with int64 dtype; must not raise.
    # GH 25959
    # Calling apply on a localized time series should not cause an error
    if aware:
        index = dti.tz_localize("UTC").index
    else:
        index = dti.index
    result = Series(index).apply(lambda x: Series([1, 2]))
    tm.assert_frame_equal(result, exp)


@pytest.mark.parametrize(
    "by_row, expected", [("compat", Series(np.ones(10), dtype="int64")), (False, 1)]
)
def test_apply_scalar_on_date_time_index_aware_series(by_row, expected):
    # [FUNCTION DOC]
    # GH 25959: applying a scalar-returning function to a UTC-localized datetime index series: compat gives
    # a series of ones (int64), not by row gives the scalar one.
    # GH 25959
    # Calling apply on a localized time series should not cause an error
    series = Series(
        np.arange(10, dtype=np.float64),
        index=date_range("2020-01-01", periods=10, tz="UTC"),
    )
    result = Series(series.index).apply(lambda x: 1, by_row=by_row)
    tm.assert_equal(result, expected)


def test_apply_to_timedelta(by_row):
    # [FUNCTION DOC]
    # Applying the timedelta conversion function to a series of valid strings, and to strings with NaN and NaT, must
    # equal converting the list directly; run for both by_row settings.
    list_of_valid_strings = ["00:00:01", "00:00:02"]
    a = pd.to_timedelta(list_of_valid_strings)
    b = Series(list_of_valid_strings).apply(pd.to_timedelta, by_row=by_row)
    tm.assert_series_equal(Series(a), b)

    list_of_strings = ["00:00:01", np.nan, pd.NaT, pd.NaT]

    a = pd.to_timedelta(list_of_strings)
    ser = Series(list_of_strings)
    b = ser.apply(pd.to_timedelta, by_row=by_row)
    tm.assert_series_equal(Series(a), b)


@pytest.mark.parametrize(
    "ops, names",
    [
        ([np.sum], ["sum"]),
        ([np.sum, np.mean], ["sum", "mean"]),
        (np.array([np.sum]), ["sum"]),
        (np.array([np.sum, np.mean]), ["sum", "mean"]),
    ],
)
@pytest.mark.parametrize(
    "how, kwargs",
    [["agg", {}], ["apply", {"by_row": "compat"}], ["apply", {"by_row": False}]],
)
def test_apply_listlike_reducer(string_series, ops, names, how, kwargs):
    # [FUNCTION DOC]
    # GH 39140: list-like of reducers (sum and mean from numpy, given as lists or arrays) through agg and apply
    # modes. Expected is a series of the reducers applied to the string series, indexed by function names and named series.
    # GH 39140
    expected = Series(
        {name: op(string_series) for name, op in zip(names, ops, strict=True)}
    )
    expected.name = "series"
    result = getattr(string_series, how)(ops, **kwargs)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "ops",
    [
        {"A": np.sum},
        {"A": np.sum, "B": np.mean},
        Series({"A": np.sum}),
        Series({"A": np.sum, "B": np.mean}),
    ],
)
@pytest.mark.parametrize(
    "how, kwargs",
    [["agg", {}], ["apply", {"by_row": "compat"}], ["apply", {"by_row": False}]],
)
def test_apply_dictlike_reducer(string_series, ops, how, kwargs, by_row):
    # [FUNCTION DOC]
    # GH 39140: dict-like of reducers (dict or Series; one or two keys) through agg and apply modes.
    # Expected is a series with dict keys as index, named like the input. The by_row fixture is requested but unused.
    # GH 39140
    expected = Series({name: op(string_series) for name, op in ops.items()})
    expected.name = string_series.name
    result = getattr(string_series, how)(ops, **kwargs)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "ops, names",
    [
        ([np.sqrt], ["sqrt"]),
        ([np.abs, np.sqrt], ["absolute", "sqrt"]),
        (np.array([np.sqrt]), ["sqrt"]),
        (np.array([np.abs, np.sqrt]), ["absolute", "sqrt"]),
    ],
)
def test_apply_listlike_transformer(string_series, ops, names, by_row):
    # [FUNCTION DOC]
    # GH 39140: list-like of transformers (square root, absolute; lists or arrays) via apply for each by_row. Expected
    # is a frame of the concatenated outputs with function names as columns; floating point errors are ignored.
    # GH 39140
    with np.errstate(all="ignore"):
        expected = concat([op(string_series) for op in ops], axis=1)
        expected.columns = names
        result = string_series.apply(ops, by_row=by_row)
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "ops, expected",
    [
        ([lambda x: x], DataFrame({"<lambda>": [1, 2, 3]})),
        ([lambda x: x.sum()], Series([6], index=["<lambda>"])),
    ],
)
def test_apply_listlike_lambda(ops, expected, by_row):
    # [FUNCTION DOC]
    # GH53400: list-like containing a lambda via apply. A lambda returning the same values gives a frame with a
    # column labelled lambda; a lambda returning the sum gives a single-value series labelled lambda.
    # GH53400
    ser = Series([1, 2, 3])
    result = ser.apply(ops, by_row=by_row)
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "ops",
    [
        {"A": np.sqrt},
        {"A": np.sqrt, "B": np.exp},
        Series({"A": np.sqrt}),
        Series({"A": np.sqrt, "B": np.exp}),
    ],
)
def test_apply_dictlike_transformer(string_series, ops, by_row):
    # [FUNCTION DOC]
    # GH 39140: dict-like of transformers (dict or Series; one or two keys) via apply for each by_row. Expected
    # is a concatenation of per-key results forming a MultiIndex series, named like the input.
    # GH 39140
    with np.errstate(all="ignore"):
        expected = concat({name: op(string_series) for name, op in ops.items()})
        expected.name = string_series.name
        result = string_series.apply(ops, by_row=by_row)
        tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "ops, expected",
    [
        (
            {"a": lambda x: x},
            Series([1, 2, 3], index=MultiIndex.from_arrays([["a"] * 3, range(3)])),
        ),
        ({"a": lambda x: x.sum()}, Series([6], index=["a"])),
    ],
)
def test_apply_dictlike_lambda(ops, by_row, expected):
    # [FUNCTION DOC]
    # GH53400: dict-like containing a lambda via apply. An identity lambda gives a series with a MultiIndex of the key
    # and position; a sum lambda gives a one-element series labelled with the key.
    # GH53400
    ser = Series([1, 2, 3])
    result = ser.apply(ops, by_row=by_row)
    tm.assert_equal(result, expected)


def test_apply_retains_column_name(by_row):
    # [FUNCTION DOC]
    # GH 16380: apply returning a series per element retains the name of the index levels: the result is a
    # frame with float values (NaN padded) whose columns are named y and index named x.
    # GH 16380
    df = DataFrame({"x": range(3)}, Index(range(3), name="x"))
    result = df.x.apply(lambda x: Series(range(x + 1), Index(range(x + 1), name="y")))
    expected = DataFrame(
        [[0.0, np.nan, np.nan], [0.0, 1.0, np.nan], [0.0, 1.0, 2.0]],
        columns=Index(range(3), name="y"),
        index=Index(range(3), name="x"),
    )
    tm.assert_frame_equal(result, expected)


def test_apply_type():
    # [FUNCTION DOC]
    # GH 46719: apply with the type function on a series of mixed objects gives a series of the types of the elements.
    # GH 46719
    s = Series([3, "string", float], index=["a", "b", "c"])
    result = s.apply(type)
    expected = Series([int, str, type], index=["a", "b", "c"])
    tm.assert_series_equal(result, expected)


def test_series_apply_unpack_nested_data():
    # [FUNCTION DOC]
    # GH#55189: apply with a function converting each list element to a series unpacks nested data into a frame with
    # NaN padding for the shorter row.
    # GH#55189
    ser = Series([[1, 2, 3], [4, 5, 6, 7]])
    result = ser.apply(lambda x: Series(x))
    expected = DataFrame({0: [1.0, 4.0], 1: [2.0, 5.0], 2: [3.0, 6.0], 3: [np.nan, 7]})
    tm.assert_frame_equal(result, expected)
