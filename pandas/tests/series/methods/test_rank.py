# [MODULE DOC]
# Tests for Series.rank: tie-breaking methods (average, min, max, first, dense),
# percentile ranks, ascending/descending, na_option, categorical, nullable,
# Arrow and string dtypes, infinities, and large inputs. Defines fixtures for a
# sample Series with NaN, expected results per method, and dtypes under test,
# plus a helper that computes the expected result dtype. Uses rank_method,
# ascending, using_infer_string and using_nan_is_na fixtures. No import-time side
# effects.

from itertools import chain
import operator

import numpy as np
import pytest

from pandas._libs.algos import (
    Infinity,
    NegInfinity,
)
import pandas.util._test_decorators as td

from pandas import (
    NA,
    NaT,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm
from pandas.api.types import CategoricalDtype


@pytest.fixture
def ser():
    # [FUNCTION DOC]
    # HELPER: fixture returning the ten-element float Series with two NaN values
    # (1, 3, 4, 2, NaN, 2, 1, 5, NaN, 3) used by the tie method tests.
    return Series([1, 3, 4, 2, np.nan, 2, 1, 5, np.nan, 3])


@pytest.fixture(
    params=[
        ["average", np.array([1.5, 5.5, 7.0, 3.5, np.nan, 3.5, 1.5, 8.0, np.nan, 5.5])],
        ["min", np.array([1, 5, 7, 3, np.nan, 3, 1, 8, np.nan, 5])],
        ["max", np.array([2, 6, 7, 4, np.nan, 4, 2, 8, np.nan, 6])],
        ["first", np.array([1, 5, 7, 3, np.nan, 4, 2, 8, np.nan, 6])],
        ["dense", np.array([1, 3, 4, 2, np.nan, 2, 1, 5, np.nan, 3])],
    ],
    ids=lambda x: x[0],
)
def results(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over (method name, expected rank array) pairs for
    # average, min, max, first and dense on the sample Series; NaN stays NaN.
    return request.param


@pytest.fixture(
    params=[
        "object",
        "float64",
        "int64",
        "Float64",
        "Int64",
        pytest.param("float64[pyarrow]", marks=td.skip_if_no("pyarrow")),
        pytest.param("int64[pyarrow]", marks=td.skip_if_no("pyarrow")),
        pytest.param("string[pyarrow]", marks=td.skip_if_no("pyarrow")),
        "string[python]",
        "str",
    ]
)
def dtype(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over dtype names (object, float64, int64, masked
    # Float64/Int64, Arrow float/int/string when pyarrow is available, string
    # python, str); returns the name.
    return request.param


def expected_dtype(dtype, method, pct=False):
    # [FUNCTION DOC]
    # HELPER: computes the expected result dtype name of rank for the given input
    # dtype name, ranking method and percentile flag. Returns float64 by default;
    # Float64 for Arrow strings; for Arrow float/int the double type when the
    # method is average or the percentile flag is set, otherwise unsigned 64-bit
    # Arrow; likewise for masked Float64/Int64 (Float64 or UInt64).
    # KNOWN ISSUE: a stale commented-out branch (a line starting with "elif" in a
    # comment) remains in the body.
    exp_dtype = "float64"
    # elif dtype in ["Int64", "Float64", "string[pyarrow]", "string[python]"]:
    if dtype in ["string[pyarrow]"]:
        exp_dtype = "Float64"
    elif dtype in ["float64[pyarrow]", "int64[pyarrow]"]:
        if method == "average" or pct:
            exp_dtype = "double[pyarrow]"
        else:
            exp_dtype = "uint64[pyarrow]"
    elif dtype in ["Float64", "Int64"]:
        if method == "average" or pct:
            exp_dtype = "Float64"
        else:
            exp_dtype = "UInt64"

    return exp_dtype


class TestSeriesRank:
    # [CLASS DOC]
    # Groups the main Series.rank tests. Stateless.
    def test_rank(self, datetime_series):
        # [FUNCTION DOC]
        # Needs scipy. Rank of the datetime_series (with NaN and ties) equals the rank
        # of its object-dtype copy and scipy's ranking with NaN preserved. Then checks
        # ranks of integer ties, percentile ranks with and without NaN (including
        # all-NaN, with date index), tiny floats, timedeltas with NaT (GH 5968) and a
        # random permutation of ordered values.
        sp_stats = pytest.importorskip("scipy.stats")

        datetime_series[::2] = np.nan
        datetime_series[:10:3] = 4.0

        ranks = datetime_series.rank()
        oranks = datetime_series.astype("O").rank()

        tm.assert_series_equal(ranks, oranks)

        mask = np.isnan(datetime_series)
        filled = datetime_series.fillna(np.inf)

        # rankdata returns an ndarray
        exp = Series(sp_stats.rankdata(filled), index=filled.index, name="ts")
        exp[mask] = np.nan

        tm.assert_series_equal(ranks, exp)

        iseries = Series(np.arange(5).repeat(2))

        iranks = iseries.rank()
        exp = iseries.astype(float).rank()
        tm.assert_series_equal(iranks, exp)
        iseries = Series(np.arange(5)) + 1.0
        exp = iseries / 5.0
        iranks = iseries.rank(pct=True)

        tm.assert_series_equal(iranks, exp)

        iseries = Series(np.repeat(1, 100))
        exp = Series(np.repeat(0.505, 100))
        iranks = iseries.rank(pct=True)
        tm.assert_series_equal(iranks, exp)

        # Explicit cast to float to avoid implicit cast when setting nan
        iseries = iseries.astype("float")
        iseries[1] = np.nan
        exp = Series(np.repeat(50.0 / 99.0, 100))
        exp[1] = np.nan
        iranks = iseries.rank(pct=True)
        tm.assert_series_equal(iranks, exp)

        iseries = Series(np.arange(5)) + 1.0
        iseries[4] = np.nan
        exp = iseries / 4.0
        iranks = iseries.rank(pct=True)
        tm.assert_series_equal(iranks, exp)

        iseries = Series(np.repeat(np.nan, 100))
        exp = iseries
        iranks = iseries.rank(pct=True)
        tm.assert_series_equal(iranks, exp)

        # Explicit cast to float to avoid implicit cast when setting nan
        iseries = Series(np.arange(5), dtype="float") + 1
        iseries[4] = np.nan
        exp = iseries / 4.0
        iranks = iseries.rank(pct=True)
        tm.assert_series_equal(iranks, exp)

        rng = date_range("1/1/1990", periods=5)
        # Explicit cast to float to avoid implicit cast when setting nan
        iseries = Series(np.arange(5), rng, dtype="float") + 1
        iseries.iloc[4] = np.nan
        exp = iseries / 4.0
        iranks = iseries.rank(pct=True)
        tm.assert_series_equal(iranks, exp)

        iseries = Series([1e-50, 1e-100, 1e-20, 1e-2, 1e-20 + 1e-30, 1e-1])
        exp = Series([2, 1, 3, 5, 4, 6.0])
        iranks = iseries.rank()
        tm.assert_series_equal(iranks, exp)

        # GH 5968
        iseries = Series(["3 day", "1 day 10m", "-2 day", NaT], dtype="m8[ns]")
        exp = Series([3, 2, 1, np.nan])
        iranks = iseries.rank()
        tm.assert_series_equal(iranks, exp)

        values = np.array(
            [-50, -1, -1e-20, -1e-25, -1e-50, 0, 1e-40, 1e-20, 1e-10, 2, 40],
            dtype="float64",
        )
        random_order = np.random.default_rng(2).permutation(len(values))
        iseries = Series(values[random_order])
        exp = Series(random_order + 1.0, dtype="float64")
        iranks = iseries.rank()
        tm.assert_series_equal(iranks, exp)

    def test_rank_categorical(self):
        # [FUNCTION DOC]
        # GH issue #15420: ordered categoricals rank by category order, ascending and
        # descending; unordered categoricals rank as plain values; na_option top, bottom
        # and keep (also descending and with percentile) give the expected ranks; an
        # invalid na_option (bad string or boolean) raises a value error.
        # GH issue #15420 rank incorrectly orders ordered categories

        # Test ascending/descending ranking for ordered categoricals
        exp = Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        exp_desc = Series([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        ordered = Series(
            ["first", "second", "third", "fourth", "fifth", "sixth"]
        ).astype(
            CategoricalDtype(
                categories=["first", "second", "third", "fourth", "fifth", "sixth"],
                ordered=True,
            )
        )
        tm.assert_series_equal(ordered.rank(), exp)
        tm.assert_series_equal(ordered.rank(ascending=False), exp_desc)

        # Unordered categoricals should be ranked as objects
        unordered = Series(
            ["first", "second", "third", "fourth", "fifth", "sixth"]
        ).astype(
            CategoricalDtype(
                categories=["first", "second", "third", "fourth", "fifth", "sixth"],
                ordered=False,
            )
        )
        exp_unordered = Series([2.0, 4.0, 6.0, 3.0, 1.0, 5.0])
        res = unordered.rank()
        tm.assert_series_equal(res, exp_unordered)

        unordered1 = Series([1, 2, 3, 4, 5, 6]).astype(
            CategoricalDtype([1, 2, 3, 4, 5, 6], False)
        )
        exp_unordered1 = Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        res1 = unordered1.rank()
        tm.assert_series_equal(res1, exp_unordered1)

        # Test na_option for rank data
        na_ser = Series(
            ["first", "second", "third", "fourth", "fifth", "sixth", np.nan]
        ).astype(
            CategoricalDtype(
                ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"],
                True,
            )
        )

        exp_top = Series([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.0])
        exp_bot = Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        exp_keep = Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan])

        tm.assert_series_equal(na_ser.rank(na_option="top"), exp_top)
        tm.assert_series_equal(na_ser.rank(na_option="bottom"), exp_bot)
        tm.assert_series_equal(na_ser.rank(na_option="keep"), exp_keep)

        # Test na_option for rank data with ascending False
        exp_top = Series([7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        exp_bot = Series([6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 7.0])
        exp_keep = Series([6.0, 5.0, 4.0, 3.0, 2.0, 1.0, np.nan])

        tm.assert_series_equal(na_ser.rank(na_option="top", ascending=False), exp_top)
        tm.assert_series_equal(
            na_ser.rank(na_option="bottom", ascending=False), exp_bot
        )
        tm.assert_series_equal(na_ser.rank(na_option="keep", ascending=False), exp_keep)

        # Test invalid values for na_option
        msg = "na_option must be one of 'keep', 'top', or 'bottom'"

        with pytest.raises(ValueError, match=msg):
            na_ser.rank(na_option="bad", ascending=False)

        # invalid type
        with pytest.raises(ValueError, match=msg):
            na_ser.rank(na_option=True, ascending=False)

        # Test with pct=True
        na_ser = Series(["first", "second", "third", "fourth", np.nan]).astype(
            CategoricalDtype(["first", "second", "third", "fourth"], True)
        )
        exp_top = Series([0.4, 0.6, 0.8, 1.0, 0.2])
        exp_bot = Series([0.2, 0.4, 0.6, 0.8, 1.0])
        exp_keep = Series([0.25, 0.5, 0.75, 1.0, np.nan])

        tm.assert_series_equal(na_ser.rank(na_option="top", pct=True), exp_top)
        tm.assert_series_equal(na_ser.rank(na_option="bottom", pct=True), exp_bot)
        tm.assert_series_equal(na_ser.rank(na_option="keep", pct=True), exp_keep)

    def test_rank_nullable_integer(self):
        # [FUNCTION DOC]
        # GH 56976: nullable Int64 with missing values and ties and na_option keep gives
        # average ranks as Float64 with missing entries kept.
        # GH 56976
        exp = Series([None, 2, None, 3, 3, 2, 3, 1], dtype="Int64")
        result = exp.rank(na_option="keep")

        expected = Series([None, 2.5, None, 5.0, 5.0, 2.5, 5.0, 1.0], dtype="Float64")

        tm.assert_series_equal(result, expected)

    def test_rank_signature(self):
        # [FUNCTION DOC]
        # The first positional parameter of rank is the axis, so passing the method
        # name positionally raises a value error that no such axis exists; the keyword
        # form works.
        s = Series([0, 1])
        s.rank(method="average")
        msg = "No axis named average for object type Series"
        with pytest.raises(ValueError, match=msg):
            s.rank("average")

    def test_rank_tie_methods(
        self, ser, results, dtype, using_infer_string, using_nan_is_na
    ):
        # [FUNCTION DOC]
        # Parametrized over tie method (results fixture) and dtype. Skips the dtypes
        # that cannot hold NaN (int64, str without the inferred-string option, and some
        # Arrow/masked integer cases without the NaN-is-NA option). Casts the sample
        # Series to the dtype, ranks with the method and compares with the expected
        # ranks in the expected result dtype (using NA instead of NaN for masked and
        # Arrow dtypes when NaN is not NA).
        # KNOWN ISSUE: TODO comments in the body note that some skipped cases could
        # work and that replacing NaN by NA should use a replace call once that works.
        method, exp = results
        if (
            dtype == "int64"
            or (
                # TODO: these can work but need to update ser construction.
                dtype in ["int64[pyarrow]", "uint64[pyarrow]", "Int64"]
                and not using_nan_is_na
            )
            or (not using_infer_string and dtype == "str")
        ):
            pytest.skip("int64/str does not support NaN")

        ser = ser if dtype is None else ser.astype(dtype)
        if dtype in ["float64[pyarrow]", "Float64"] and not using_nan_is_na:
            # TODO: use ser.replace(np.nan, NA) once that works
            ser[np.isnan(ser.to_numpy(dtype=np.float64, na_value=np.nan))] = NA
            mask = np.isnan(exp)
            exp = exp.astype(object)
            exp[mask] = NA

        result = ser.rank(method=method)

        if dtype == "string[pyarrow]" and not using_nan_is_na:
            mask = np.isnan(exp)
            exp = exp.astype(object)
            exp[mask] = NA

        expected = Series(exp, dtype=expected_dtype(dtype, method))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("na_option", ["top", "bottom", "keep"])
    @pytest.mark.parametrize(
        "dtype, na_value, pos_inf, neg_inf",
        [
            ("object", None, Infinity(), NegInfinity()),
            ("float64", np.nan, np.inf, -np.inf),
            ("Float64", NA, np.inf, -np.inf),
            pytest.param(
                "float64[pyarrow]",
                NA,
                np.inf,
                -np.inf,
                marks=td.skip_if_no("pyarrow"),
            ),
        ],
    )
    def test_rank_tie_methods_on_infs_nans(
        self,
        rank_method,
        na_option,
        ascending,
        dtype,
        na_value,
        pos_inf,
        neg_inf,
        using_nan_is_na,
    ):
        # [FUNCTION DOC]
        # Parametrized over rank method, ascending, na_option and dtype/value tuples
        # (object with infinity helper objects, float64, Float64, Arrow float).
        # Needs scipy. A Series of three negative infinities, three missing values and
        # three positive infinities is ranked; expected ranks per method are computed by
        # group (missing values placed at top or bottom per na_option, or kept as
        # missing) and reversed when descending. GH#62043 applies to the Float64 result
        # dtype.
        pytest.importorskip("scipy")
        if dtype == "float64[pyarrow]":
            if rank_method == "average":
                exp_dtype = "float64[pyarrow]"
            else:
                exp_dtype = "uint64[pyarrow]"
        elif dtype == "Float64":
            # GH#62043
            if rank_method == "average":
                exp_dtype = "Float64"
            else:
                exp_dtype = "UInt64"
        else:
            exp_dtype = "float64"

        chunk = 3
        in_arr = [neg_inf] * chunk + [na_value] * chunk + [pos_inf] * chunk
        iseries = Series(in_arr, dtype=dtype)
        exp_ranks = {
            "average": ([2, 2, 2], [5, 5, 5], [8, 8, 8]),
            "min": ([1, 1, 1], [4, 4, 4], [7, 7, 7]),
            "max": ([3, 3, 3], [6, 6, 6], [9, 9, 9]),
            "first": ([1, 2, 3], [4, 5, 6], [7, 8, 9]),
            "dense": ([1, 1, 1], [2, 2, 2], [3, 3, 3]),
        }
        ranks = exp_ranks[rank_method]
        if na_option == "top":
            order = [ranks[1], ranks[0], ranks[2]]
        elif na_option == "bottom":
            order = [ranks[0], ranks[2], ranks[1]]
        elif dtype in ("float64[pyarrow]", "Float64") and not using_nan_is_na:
            order = [ranks[0], [NA] * chunk, ranks[1]]
        else:
            order = [ranks[0], [np.nan] * chunk, ranks[1]]
        expected = order if ascending else order[::-1]
        expected = list(chain.from_iterable(expected))
        result = iseries.rank(
            method=rank_method, na_option=na_option, ascending=ascending
        )
        exp_ser = Series(expected, dtype=exp_dtype)
        tm.assert_series_equal(result, exp_ser)

    def test_rank_desc_mix_nans_infs(self):
        # [FUNCTION DOC]
        # GH 19538: descending rank with NaN and both infinities gives the ranks 3, NaN,
        # 1, 4, 2.
        # GH 19538
        # check descending ranking when mix nans and infs
        iseries = Series([1, np.nan, np.inf, -np.inf, 25])
        result = iseries.rank(ascending=False)
        exp = Series([3, np.nan, 1, 4, 2], dtype="float64")
        tm.assert_series_equal(result, exp)

    @pytest.mark.parametrize(
        "op, value",
        [
            [operator.add, 0],
            [operator.add, 1e6],
            [operator.mul, 1e-6],
        ],
    )
    def test_rank_methods_series(self, rank_method, op, value):
        # [FUNCTION DOC]
        # Needs scipy. Parametrized over the rank method and three transformations
        # (shifting by 0, by 1e6, scaling by 1e-6). Random data with duplicates is ranked
        # and compared to scipy's rankdata (ordinal for first), as float64.
        sp_stats = pytest.importorskip("scipy.stats")

        xs = np.random.default_rng(2).standard_normal(9)
        xs = np.concatenate([xs[i:] for i in range(0, 9, 2)])  # add duplicates
        np.random.default_rng(2).shuffle(xs)

        index = [chr(ord("a") + i) for i in range(len(xs))]
        vals = op(xs, value)
        ts = Series(vals, index=index)
        result = ts.rank(method=rank_method)
        sprank = sp_stats.rankdata(
            vals, rank_method if rank_method != "first" else "ordinal"
        )
        expected = Series(sprank, index=index).astype("float64")
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "ser, exp",
        [
            ([1], [1]),
            ([2], [1]),
            ([0], [1]),
            ([2, 2], [1, 1]),
            ([1, 2, 3], [1, 2, 3]),
            ([4, 2, 1], [3, 2, 1]),
            ([1, 1, 5, 5, 3], [1, 1, 3, 3, 2]),
            ([-5, -4, -3, -2, -1], [1, 2, 3, 4, 5]),
        ],
    )
    def test_rank_dense_method(self, dtype, ser, exp):
        # [FUNCTION DOC]
        # Parametrized over dtype and (data, expected ranks) cases. Dense ranking of
        # the cast Series equals the expected ranks in the expected dtype. For string
        # dtypes with negative numbers the order is reversed, because strings sort
        # differently.
        if ser[0] < 0 and dtype.startswith("str"):
            exp = exp[::-1]
        s = Series(ser).astype(dtype)
        result = s.rank(method="dense")
        expected = Series(exp).astype(expected_dtype(dtype, "dense"))
        tm.assert_series_equal(result, expected)

    def test_rank_descending(
        self, ser, results, dtype, using_infer_string, using_nan_is_na
    ):
        # [FUNCTION DOC]
        # Descending rank equals ranking the (max minus value) Series ascending, for
        # the default method and for each method. Drops NaN for dtypes that cannot hold
        # them; string dtypes are converted to float for the arithmetic.
        method, _ = results
        if (
            dtype == "int64"
            or (dtype in ["int64[pyarrow]", "Int64", "Float64"] and not using_nan_is_na)
            or (not using_infer_string and dtype == "str")
        ):
            s = ser.dropna().astype(dtype)
        else:
            s = ser.astype(dtype)

        res = s.rank(ascending=False)
        if dtype.startswith("str"):
            expected = (s.astype("float64").max() - s.astype("float64")).rank()
        else:
            expected = (s.max() - s).rank()
        if dtype == "string[pyarrow]" and not using_nan_is_na:
            expected = expected.replace(np.nan, NA)
        tm.assert_series_equal(res, expected.astype(expected_dtype(dtype, "average")))

        if dtype.startswith("str"):
            expected = (s.astype("float64").max() - s.astype("float64")).rank(
                method=method
            )
        else:
            expected = (s.max() - s).rank(method=method)
        res2 = s.rank(method=method, ascending=False)
        if dtype == "string[pyarrow]" and not using_nan_is_na:
            expected = expected.replace(np.nan, NA)
        tm.assert_series_equal(res2, expected.astype(expected_dtype(dtype, method)))

    def test_rank_int(self, ser, results):
        # [FUNCTION DOC]
        # Ranking int64 data (NaN dropped) with each method matches the expected
        # ranks without the NaN entries.
        method, exp = results
        s = ser.dropna().astype("i8")

        result = s.rank(method=method)
        expected = Series(exp).dropna()
        expected.index = result.index
        tm.assert_series_equal(result, expected)

    def test_rank_object_bug(self):
        # [FUNCTION DOC]
        # GH 13445: smoke test that ranking an all-NaN object Series, ascending or
        # descending, does not raise.
        # GH 13445

        # smoke tests
        Series([np.nan] * 32).astype(object).rank(ascending=True)
        Series([np.nan] * 32).astype(object).rank(ascending=False)

    def test_rank_modify_inplace(self):
        # [FUNCTION DOC]
        # GH 18521: ranking a datetime Series does not modify it.
        # GH 18521
        # Check rank does not mutate series
        s = Series([Timestamp("2017-01-05 10:20:27.569000"), NaT])
        expected = s.copy()

        s.rank()
        result = s
        tm.assert_series_equal(result, expected)

    def test_rank_ea_small_values(self):
        # [FUNCTION DOC]
        # GH#52471: very small and very large nullable floats are ranked correctly with
        # the min method, with NA kept, as an unsigned 64-bit nullable result.
        # GH#52471
        ser = Series(
            [5.4954145e29, -9.791984e-21, 9.3715776e-26, NA, 1.8790257e-28],
            dtype="Float64",
        )
        result = ser.rank(method="min")
        expected = Series([4, 1, 3, NA, 2], dtype="UInt64")
        tm.assert_series_equal(result, expected)


# GH15630, pct should be on 100% basis when method='dense'


@pytest.mark.parametrize(
    "ser, exp",
    [
        ([1], [1.0]),
        ([1, 2], [1.0 / 2, 2.0 / 2]),
        ([2, 2], [1.0, 1.0]),
        ([1, 2, 3], [1.0 / 3, 2.0 / 3, 3.0 / 3]),
        ([1, 2, 2], [1.0 / 2, 2.0 / 2, 2.0 / 2]),
        ([4, 2, 1], [3.0 / 3, 2.0 / 3, 1.0 / 3]),
        ([1, 1, 5, 5, 3], [1.0 / 3, 1.0 / 3, 3.0 / 3, 3.0 / 3, 2.0 / 3]),
        ([1, 1, 3, 3, 5, 5], [1.0 / 3, 1.0 / 3, 2.0 / 3, 2.0 / 3, 3.0 / 3, 3.0 / 3]),
        ([-5, -4, -3, -2, -1], [1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5, 5.0 / 5]),
    ],
)
def test_rank_dense_pct(dtype, ser, exp):
    # [FUNCTION DOC]
    # GH15630: parametrized over dtype and cases; percentile dense ranks are on a
    # 100 percent basis of the number of distinct values. For string dtypes with
    # negative numbers expected values are reversed.
    if ser[0] < 0 and dtype.startswith("str"):
        exp = exp[::-1]
    s = Series(ser).astype(dtype)
    result = s.rank(method="dense", pct=True)
    expected = Series(exp).astype(expected_dtype(dtype, "dense", pct=True))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "ser, exp",
    [
        ([1], [1.0]),
        ([1, 2], [1.0 / 2, 2.0 / 2]),
        ([2, 2], [1.0 / 2, 1.0 / 2]),
        ([1, 2, 3], [1.0 / 3, 2.0 / 3, 3.0 / 3]),
        ([1, 2, 2], [1.0 / 3, 2.0 / 3, 2.0 / 3]),
        ([4, 2, 1], [3.0 / 3, 2.0 / 3, 1.0 / 3]),
        ([1, 1, 5, 5, 3], [1.0 / 5, 1.0 / 5, 4.0 / 5, 4.0 / 5, 3.0 / 5]),
        ([1, 1, 3, 3, 5, 5], [1.0 / 6, 1.0 / 6, 3.0 / 6, 3.0 / 6, 5.0 / 6, 5.0 / 6]),
        ([-5, -4, -3, -2, -1], [1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5, 5.0 / 5]),
    ],
)
def test_rank_min_pct(dtype, ser, exp):
    # [FUNCTION DOC]
    # Percentile rank with the min method: each tie gets the lowest rank divided by
    # the number of values.
    if ser[0] < 0 and dtype.startswith("str"):
        exp = exp[::-1]
    s = Series(ser).astype(dtype)
    result = s.rank(method="min", pct=True)
    expected = Series(exp).astype(expected_dtype(dtype, "min", pct=True))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "ser, exp",
    [
        ([1], [1.0]),
        ([1, 2], [1.0 / 2, 2.0 / 2]),
        ([2, 2], [1.0, 1.0]),
        ([1, 2, 3], [1.0 / 3, 2.0 / 3, 3.0 / 3]),
        ([1, 2, 2], [1.0 / 3, 3.0 / 3, 3.0 / 3]),
        ([4, 2, 1], [3.0 / 3, 2.0 / 3, 1.0 / 3]),
        ([1, 1, 5, 5, 3], [2.0 / 5, 2.0 / 5, 5.0 / 5, 5.0 / 5, 3.0 / 5]),
        ([1, 1, 3, 3, 5, 5], [2.0 / 6, 2.0 / 6, 4.0 / 6, 4.0 / 6, 6.0 / 6, 6.0 / 6]),
        ([-5, -4, -3, -2, -1], [1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5, 5.0 / 5]),
    ],
)
def test_rank_max_pct(dtype, ser, exp):
    # [FUNCTION DOC]
    # Percentile rank with the max method: each tie gets the highest rank divided by
    # the number of values.
    if ser[0] < 0 and dtype.startswith("str"):
        exp = exp[::-1]
    s = Series(ser).astype(dtype)
    result = s.rank(method="max", pct=True)
    expected = Series(exp).astype(expected_dtype(dtype, "max", pct=True))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "ser, exp",
    [
        ([1], [1.0]),
        ([1, 2], [1.0 / 2, 2.0 / 2]),
        ([2, 2], [1.5 / 2, 1.5 / 2]),
        ([1, 2, 3], [1.0 / 3, 2.0 / 3, 3.0 / 3]),
        ([1, 2, 2], [1.0 / 3, 2.5 / 3, 2.5 / 3]),
        ([4, 2, 1], [3.0 / 3, 2.0 / 3, 1.0 / 3]),
        ([1, 1, 5, 5, 3], [1.5 / 5, 1.5 / 5, 4.5 / 5, 4.5 / 5, 3.0 / 5]),
        ([1, 1, 3, 3, 5, 5], [1.5 / 6, 1.5 / 6, 3.5 / 6, 3.5 / 6, 5.5 / 6, 5.5 / 6]),
        ([-5, -4, -3, -2, -1], [1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5, 5.0 / 5]),
    ],
)
def test_rank_average_pct(dtype, ser, exp):
    # [FUNCTION DOC]
    # Percentile rank with the average method: ties get the mean rank divided by the
    # number of values.
    if ser[0] < 0 and dtype.startswith("str"):
        exp = exp[::-1]
    s = Series(ser).astype(dtype)
    result = s.rank(method="average", pct=True)
    expected = Series(exp).astype(expected_dtype(dtype, "average", pct=True))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "ser, exp",
    [
        ([1], [1.0]),
        ([1, 2], [1.0 / 2, 2.0 / 2]),
        ([2, 2], [1.0 / 2, 2.0 / 2.0]),
        ([1, 2, 3], [1.0 / 3, 2.0 / 3, 3.0 / 3]),
        ([1, 2, 2], [1.0 / 3, 2.0 / 3, 3.0 / 3]),
        ([4, 2, 1], [3.0 / 3, 2.0 / 3, 1.0 / 3]),
        ([1, 1, 5, 5, 3], [1.0 / 5, 2.0 / 5, 4.0 / 5, 5.0 / 5, 3.0 / 5]),
        ([1, 1, 3, 3, 5, 5], [1.0 / 6, 2.0 / 6, 3.0 / 6, 4.0 / 6, 5.0 / 6, 6.0 / 6]),
        ([-5, -4, -3, -2, -1], [1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5, 5.0 / 5]),
    ],
)
def test_rank_first_pct(dtype, ser, exp):
    # [FUNCTION DOC]
    # Percentile rank with the first method: ties are ranked in order of
    # appearance, divided by the number of values.
    if ser[0] < 0 and dtype.startswith("str"):
        exp = exp[::-1]
    s = Series(ser).astype(dtype)
    result = s.rank(method="first", pct=True)
    expected = Series(exp).astype(expected_dtype(dtype, "first", pct=True))
    tm.assert_series_equal(result, expected)


@pytest.mark.single_cpu
def test_pct_max_many_rows():
    # [FUNCTION DOC]
    # GH 18271 (single CPU only because of memory): the percentile rank maximum of
    # 2 to the 24th plus one elements is exactly 1.
    # GH 18271
    s = Series(np.arange(2**24 + 1))
    result = s.rank(pct=True).max()
    assert result == 1
