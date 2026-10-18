# [MODULE DOC]
# Tests that apply, agg and transform accept string function names and numpy/builtin function
# objects consistently with the equivalent named methods, for both Series and DataFrame.
# Includes the table of functions that map to cythonized reductions/transforms (with a FutureWarning
# for non-string callables) and the group-by transformation kernels.

from itertools import chain
import operator

import numpy as np
import pytest

from pandas.compat import (
    WASM,
)

from pandas.core.dtypes.common import is_number

from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm
from pandas.tests.apply.common import (
    frame_transform_kernels,
    series_transform_kernels,
)


@pytest.mark.parametrize("func", ["sum", "mean", "min", "max", "std"])
@pytest.mark.parametrize(
    "kwds",
    [
        pytest.param({}, id="no_kwds"),
        pytest.param({"axis": 1}, id="on_axis"),
        pytest.param({"numeric_only": True}, id="func_kwds"),
        pytest.param({"axis": 1, "numeric_only": True}, id="axis_and_func_kwds"),
    ],
)
@pytest.mark.parametrize("how", ["agg", "apply"])
def test_apply_with_string_funcs(float_frame, func, kwds, how):
    # [FUNCTION DOC]
    # Parametrized by func (sum, mean, min, max, std), kwds (none, axis, numeric_only, both) and how
    # (agg or apply). Calling the named string via agg or apply on the float_frame fixture must equal calling
    # the frame's method of that name with the same keywords; compared as series.
    result = getattr(float_frame, how)(func, **kwds)
    expected = getattr(float_frame, func)(**kwds)
    tm.assert_series_equal(result, expected)


def test_with_string_args(datetime_series, all_numeric_reductions):
    # [FUNCTION DOC]
    # Applying each numeric reduction name (all_numeric_reductions fixture) by string to the datetime
    # series fixture must give the same scalar as calling the named method directly.
    result = datetime_series.apply(all_numeric_reductions)
    expected = getattr(datetime_series, all_numeric_reductions)()
    assert result == expected


@pytest.mark.parametrize("op", ["mean", "median", "std", "var"])
@pytest.mark.parametrize("how", ["agg", "apply"])
def test_apply_np_reducer(op, how):
    # [FUNCTION DOC]
    # GH 39116: agg or apply with a reducer name (mean, median, std, var) on a 2x2 frame must equal the
    # numpy function applied along axis zero, using a sample standard deviation/variance degrees of freedom
    # of one for std and var (pandas defaults to 1, numpy to 0), as a series indexed by the columns.
    # GH 39116
    float_frame = DataFrame({"a": [1, 2], "b": [3, 4]})
    result = getattr(float_frame, how)(op)
    # pandas ddof defaults to 1, numpy to 0
    kwargs = {"ddof": 1} if op in ("std", "var") else {}
    expected = Series(
        getattr(np, op)(float_frame, axis=0, **kwargs), index=float_frame.columns
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.skipif(WASM, reason="No fp exception support in wasm")
@pytest.mark.parametrize(
    "op", ["abs", "ceil", "cos", "cumsum", "exp", "log", "sqrt", "square"]
)
@pytest.mark.parametrize("how", ["transform", "apply"])
def test_apply_np_transformer(float_frame, op, how):
    # [FUNCTION DOC]
    # GH 39116: transform or apply with a numpy-like function name must equal the numpy function applied to the
    # frame. Skipped under WASM (no floating point exception support). One element is set to -1 so log and sqrt
    # emit a RuntimeWarning, which is expected for those two ops only (stacklevel is not checked).
    # GH 39116

    # float_frame will _usually_ have negative values, which will
    #  trigger the warning here, but let's put one in just to be sure
    float_frame.iloc[0, 0] = -1.0
    warn = None
    if op in ["log", "sqrt"]:
        warn = RuntimeWarning

    with tm.assert_produces_warning(warn, check_stacklevel=False):
        # float_frame fixture is defined in conftest.py, so we don't check the
        # stacklevel as otherwise the test would fail.
        result = getattr(float_frame, how)(op)
        expected = getattr(np, op)(float_frame)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "series, func, expected",
    chain(
        tm.get_cython_table_params(
            Series(dtype=np.float64),
            [
                ("sum", 0),
                ("max", np.nan),
                ("min", np.nan),
                ("all", True),
                ("any", False),
                ("mean", np.nan),
                ("prod", 1),
                ("std", np.nan),
                ("var", np.nan),
                ("median", np.nan),
            ],
        ),
        tm.get_cython_table_params(
            Series([np.nan, 1, 2, 3]),
            [
                ("sum", 6),
                ("max", 3),
                ("min", 1),
                ("all", True),
                ("any", True),
                ("mean", 2),
                ("prod", 6),
                ("std", 1),
                ("var", 1),
                ("median", 2),
            ],
        ),
        tm.get_cython_table_params(
            Series("a b c".split()),
            [
                ("sum", "abc"),
                ("max", "c"),
                ("min", "a"),
                ("all", True),
                ("any", True),
            ],
        ),
    ),
)
def test_agg_cython_table_series(series, func, expected):
    # [FUNCTION DOC]
    # GH21224: reducing functions from the cython table. Parametrized by series, func and expected,
    # built from three series (empty float, floats with NaN, strings) and the table helper which yields both
    # the name and the corresponding callable. A callable produces a FutureWarning about the Series
    # implementation being used; a string produces none. Numeric expected values are compared with isclose
    # (NaN equal) and others with equality.
    # GH21224
    # test reducing functions in
    # pandas.core.base.SelectionMixin._cython_table
    warn = None if isinstance(func, str) else FutureWarning
    with tm.assert_produces_warning(warn, match="is currently using Series.*"):
        result = series.agg(func)
    if is_number(expected):
        assert np.isclose(result, expected, equal_nan=True)
    else:
        assert result == expected


@pytest.mark.parametrize(
    "series, func, expected",
    chain(
        tm.get_cython_table_params(
            Series(dtype=np.float64),
            [
                ("cumprod", Series([], dtype=np.float64)),
                ("cumsum", Series([], dtype=np.float64)),
            ],
        ),
        tm.get_cython_table_params(
            Series([np.nan, 1, 2, 3]),
            [
                ("cumprod", Series([np.nan, 1, 2, 6])),
                ("cumsum", Series([np.nan, 1, 3, 6])),
            ],
        ),
        tm.get_cython_table_params(
            Series("a b c".split()), [("cumsum", Series(["a", "ab", "abc"]))]
        ),
    ),
)
def test_agg_cython_table_transform_series(series, func, expected):
    # [FUNCTION DOC]
    # GH21224: transforming functions (cumulative product and sum) from the cython table applied by agg
    # to series (empty, with NaN, strings). Callables warn with FutureWarning, strings do not.
    # Result must equal the expected series.
    # GH21224
    # test transforming functions in
    # pandas.core.base.SelectionMixin._cython_table (cumprod, cumsum)
    warn = None if isinstance(func, str) else FutureWarning
    with tm.assert_produces_warning(warn, match="is currently using Series.*"):
        result = series.agg(func)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "df, func, expected",
    chain(
        tm.get_cython_table_params(
            DataFrame(),
            [
                ("sum", Series(dtype="float64")),
                ("max", Series(dtype="float64")),
                ("min", Series(dtype="float64")),
                ("all", Series(dtype=bool)),
                ("any", Series(dtype=bool)),
                ("mean", Series(dtype="float64")),
                ("prod", Series(dtype="float64")),
                ("std", Series(dtype="float64")),
                ("var", Series(dtype="float64")),
                ("median", Series(dtype="float64")),
            ],
        ),
        tm.get_cython_table_params(
            DataFrame([[np.nan, 1], [1, 2]]),
            [
                ("sum", Series([1.0, 3])),
                ("max", Series([1.0, 2])),
                ("min", Series([1.0, 1])),
                ("all", Series([True, True])),
                ("any", Series([True, True])),
                ("mean", Series([1, 1.5])),
                ("prod", Series([1.0, 2])),
                ("std", Series([np.nan, 0.707107])),
                ("var", Series([np.nan, 0.5])),
                ("median", Series([1, 1.5])),
            ],
        ),
    ),
)
def test_agg_cython_table_frame(df, func, expected, axis):
    # [FUNCTION DOC]
    # GH 21224: reducing functions from the cython table through frame agg along the axis fixture, for an
    # empty frame and a 2x2 frame with NaN. Callables warn with FutureWarning (GH#53425), strings do not.
    # Result must equal the expected series.
    # GH 21224
    # test reducing functions in
    # pandas.core.base.SelectionMixin._cython_table
    warn = None if isinstance(func, str) else FutureWarning
    with tm.assert_produces_warning(warn, match="is currently using DataFrame.*"):
        # GH#53425
        result = df.agg(func, axis=axis)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "df, func, expected",
    chain(
        tm.get_cython_table_params(
            DataFrame(), [("cumprod", DataFrame()), ("cumsum", DataFrame())]
        ),
        tm.get_cython_table_params(
            DataFrame([[np.nan, 1], [1, 2]]),
            [
                ("cumprod", DataFrame([[np.nan, 1], [1, 2]])),
                ("cumsum", DataFrame([[np.nan, 1], [1, 3]])),
            ],
        ),
    ),
)
def test_agg_cython_table_transform_frame(df, func, expected, axis):
    # [FUNCTION DOC]
    # GH 21224: transforming functions (cumulative product and sum) through frame agg along the axis
    # fixture. When the axis is columns the expected frame is first cast to float64 since blockwise
    # operation does not preserve dtypes. Callables warn with FutureWarning, strings do not.
    # GH 21224
    # test transforming functions in
    # pandas.core.base.SelectionMixin._cython_table (cumprod, cumsum)
    if axis in ("columns", 1):
        # operating blockwise doesn't let us preserve dtypes
        expected = expected.astype("float64")

    warn = None if isinstance(func, str) else FutureWarning
    with tm.assert_produces_warning(warn, match="is currently using DataFrame.*"):
        # GH#53425
        result = df.agg(func, axis=axis)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("op", series_transform_kernels)
def test_transform_groupby_kernel_series(request, string_series, op):
    # [FUNCTION DOC]
    # GH 35964: for each series group-by transformation kernel, transform on the string series with
    # axis zero (plus an argument of zero for fillna) must equal the same kernel on a group-by over a single
    # group. The ngroup kernel is marked an expected failure (ValueError) as it is not valid for non-grouped
    # objects. The fillna kernel emits a deprecation FutureWarning on the group-by side.
    # GH 35964
    if op == "ngroup":
        request.applymarker(
            pytest.mark.xfail(raises=ValueError, reason="ngroup not valid for NDFrame")
        )
    args = [0.0] if op == "fillna" else []
    ones = np.ones(string_series.shape[0])

    warn = FutureWarning if op == "fillna" else None
    msg = "SeriesGroupBy.fillna is deprecated"
    with tm.assert_produces_warning(warn, match=msg):
        expected = string_series.groupby(ones).transform(op, *args)
    result = string_series.transform(op, 0, *args)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("op", frame_transform_kernels)
def test_transform_groupby_kernel_frame(request, float_frame, op):
    # [FUNCTION DOC]
    # GH 35964: same as the series variant but for frames: transform of each kernel must equal the
    # group-by transform over one group, with the fillna deprecation warning expected. The ngroup kernel is an
    # expected failure. Then the test adds a copy of column A as a new column to guarantee multiple internal
    # blocks and repeats the comparison with a fresh grouping.
    if op == "ngroup":
        request.applymarker(
            pytest.mark.xfail(raises=ValueError, reason="ngroup not valid for NDFrame")
        )

    # GH 35964

    args = [0.0] if op == "fillna" else []
    ones = np.ones(float_frame.shape[0])
    gb = float_frame.groupby(ones)

    warn = FutureWarning if op == "fillna" else None
    op_msg = "DataFrameGroupBy.fillna is deprecated"
    with tm.assert_produces_warning(warn, match=op_msg):
        expected = gb.transform(op, *args)

    result = float_frame.transform(op, 0, *args)
    tm.assert_frame_equal(result, expected)

    # same thing, but ensuring we have multiple blocks
    assert "E" not in float_frame.columns
    float_frame["E"] = float_frame["A"].copy()
    assert len(float_frame._mgr.blocks) > 1

    ones = np.ones(float_frame.shape[0])
    gb2 = float_frame.groupby(ones)
    expected2 = gb2.transform(op, *args)
    result2 = float_frame.transform(op, 0, *args)
    tm.assert_frame_equal(result2, expected2)


@pytest.mark.parametrize("method", ["abs", "shift", "pct_change", "cumsum", "rank"])
def test_transform_method_name(method):
    # [FUNCTION DOC]
    # GH 19760: DataFrame transform by method name (abs, shift, pct_change, cumsum, rank) on a
    # two-row frame must equal calling that method on the frame via a method caller.
    # GH 19760
    df = DataFrame({"A": [-1, 2]})
    result = df.transform(method)
    expected = operator.methodcaller(method)(df)
    tm.assert_frame_equal(result, expected)
