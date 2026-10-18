# [MODULE DOC]
# Tests for DataFrameGroupBy.quantile and SeriesGroupBy.quantile: scalar and list-like q, all interpolation modes, missing keys and values, masked/nullable, datetime-like and timedelta dtypes, ordering of result index levels, and error handling for unsupported dtypes and out-of-range q.
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
)
import pandas._testing as tm


@pytest.mark.parametrize(
    "interpolation", ["linear", "lower", "higher", "nearest", "midpoint"]
)
@pytest.mark.parametrize(
    "a_vals,b_vals",
    [
        # Ints
        ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]),
        ([1, 2, 3, 4], [4, 3, 2, 1]),
        ([1, 2, 3, 4, 5], [4, 3, 2, 1]),
        # Floats
        ([1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]),
        # Missing data
        ([1.0, np.nan, 3.0, np.nan, 5.0], [5.0, np.nan, 3.0, np.nan, 1.0]),
        ([np.nan, 4.0, np.nan, 2.0, np.nan], [np.nan, 4.0, np.nan, 2.0, np.nan]),
        # Timestamps
        (
            pd.date_range("1/1/18", freq="D", periods=5),
            pd.date_range("1/1/18", freq="D", periods=5)[::-1],
        ),
        (
            pd.date_range("1/1/18", freq="D", periods=5).as_unit("s"),
            pd.date_range("1/1/18", freq="D", periods=5)[::-1].as_unit("s"),
        ),
        # All NA
        ([np.nan] * 5, [np.nan] * 5),
    ],
)
@pytest.mark.parametrize("q", [0, 0.25, 0.5, 0.75, 1])
def test_quantile(interpolation, a_vals, b_vals, q):
    # [FUNCTION DOC]
    # Parametrized over five interpolation modes, many pairs of value sequences (ints, floats, missing data, datetimes at two units, all-NA) and q in 0, 0.25, 0.5, 0.75, 1. Builds a frame with key a for the first sequence and key b for the second; the expected result is the plain Series quantile of each sequence with the same interpolation, arranged as a one-column frame indexed by key. For datetime inputs whose expected dtype is datetime, the expected is cast to the input dtype (workaround for unit inference). The groupby quantile must match.
    all_vals = pd.concat([pd.Series(a_vals), pd.Series(b_vals)])

    a_expected = pd.Series(a_vals).quantile(q, interpolation=interpolation)
    b_expected = pd.Series(b_vals).quantile(q, interpolation=interpolation)

    df = DataFrame({"key": ["a"] * len(a_vals) + ["b"] * len(b_vals), "val": all_vals})

    expected = DataFrame(
        [a_expected, b_expected], columns=["val"], index=Index(["a", "b"], name="key")
    )
    if all_vals.dtype.kind == "M" and expected.dtypes.values[0].kind == "M":
        # TODO(non-nano): this should be unnecessary once array_to_datetime
        #  correctly infers non-nano from Timestamp.unit
        expected = expected.astype(all_vals.dtype)
    result = df.groupby("key").quantile(q, interpolation=interpolation)

    tm.assert_frame_equal(result, expected)


def test_quantile_array():
    # [FUNCTION DOC]
    # Refs GitHub issue 27526: grouping by a numpy array and passing a one-element list q returns a result with a two-level index (group, q). A second case with two columns and two quantiles checks interpolated values per group.
    # https://github.com/pandas-dev/pandas/issues/27526
    df = DataFrame({"A": [0, 1, 2, 3, 4]})
    key = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    result = df.groupby(key).quantile([0.25])

    index = pd.MultiIndex.from_product([[0, 1], [0.25]])
    expected = DataFrame({"A": [0.25, 2.50]}, index=index)
    tm.assert_frame_equal(result, expected)

    df = DataFrame({"A": [0, 1, 2, 3], "B": [4, 5, 6, 7]})
    index = pd.MultiIndex.from_product([[0, 1], [0.25, 0.75]])

    key = np.array([0, 0, 1, 1], dtype=np.int64)
    result = df.groupby(key).quantile([0.25, 0.75])
    expected = DataFrame(
        {"A": [0.25, 0.75, 2.25, 2.75], "B": [4.25, 4.75, 6.25, 6.75]}, index=index
    )
    tm.assert_frame_equal(result, expected)


def test_quantile_array2():
    # [FUNCTION DOC]
    # Refs pandas PR 28085: random integer frame grouped by column A with q of 0.3 and 0.7. Expects hard-coded per-group quantiles for B and C with a two-level index (A and an unnamed q level). NOTE the expected values depend on the seeded random generator.
    # https://github.com/pandas-dev/pandas/pull/28085#issuecomment-524066959
    arr = np.random.default_rng(2).integers(0, 5, size=(10, 3), dtype=np.int64)
    df = DataFrame(arr, columns=list("ABC"))
    result = df.groupby("A").quantile([0.3, 0.7])
    expected = DataFrame(
        {
            "B": [2.0, 2.0, 2.3, 2.7, 0.3, 0.7, 3.2, 4.0, 0.3, 0.7],
            "C": [1.0, 1.0, 1.9, 3.0999999999999996, 0.3, 0.7, 2.6, 3.0, 1.2, 2.8],
        },
        index=pd.MultiIndex.from_product(
            [[0, 1, 2, 3, 4], [0.3, 0.7]], names=["A", None]
        ),
    )
    tm.assert_frame_equal(result, expected)


def test_quantile_array_no_sort():
    # [FUNCTION DOC]
    # With sort false, the group order of first appearance is retained (1 then 0) and, within each group, the quantile order follows the order of q as given, including a reversed list of q values.
    df = DataFrame({"A": [0, 1, 2], "B": [3, 4, 5]})
    key = np.array([1, 0, 1], dtype=np.int64)
    result = df.groupby(key, sort=False).quantile([0.25, 0.5, 0.75])
    expected = DataFrame(
        {"A": [0.5, 1.0, 1.5, 1.0, 1.0, 1.0], "B": [3.5, 4.0, 4.5, 4.0, 4.0, 4.0]},
        index=pd.MultiIndex.from_product([[1, 0], [0.25, 0.5, 0.75]]),
    )
    tm.assert_frame_equal(result, expected)

    result = df.groupby(key, sort=False).quantile([0.75, 0.25])
    expected = DataFrame(
        {"A": [1.5, 0.5, 1.0, 1.0], "B": [4.5, 3.5, 4.0, 4.0]},
        index=pd.MultiIndex.from_product([[1, 0], [0.75, 0.25]]),
    )
    tm.assert_frame_equal(result, expected)


def test_quantile_array_multiple_levels():
    # [FUNCTION DOC]
    # Grouping by two key columns with a two-element q list yields a three-level index (the two keys plus an unnamed q level), with per-group interpolated quantiles for A and B.
    df = DataFrame(
        {"A": [0, 1, 2], "B": [3, 4, 5], "c": ["a", "a", "a"], "d": ["a", "a", "b"]}
    )
    result = df.groupby(["c", "d"]).quantile([0.25, 0.75])
    index = pd.MultiIndex.from_tuples(
        [("a", "a", 0.25), ("a", "a", 0.75), ("a", "b", 0.25), ("a", "b", 0.75)],
        names=["c", "d", None],
    )
    expected = DataFrame(
        {"A": [0.25, 0.75, 2.0, 2.0], "B": [3.25, 3.75, 5.0, 5.0]}, index=index
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("frame_size", [(2, 3), (100, 10)])
@pytest.mark.parametrize("groupby", [[0], [0, 1]])
@pytest.mark.parametrize("q", [[0.5, 0.6]])
def test_groupby_quantile_with_arraylike_q_and_int_columns(frame_size, groupby, q):
    # [FUNCTION DOC]
    # GH30289: parametrized over frame size, one or two integer-labelled grouping columns, and list q. Builds a frame whose rows cycle through four values, computes the expected MultiIndex (group levels plus q level) and the expected columns excluding the grouping columns, and checks the groupby quantile result with integer column labels.
    # GH30289
    nrow, ncol = frame_size
    df = DataFrame(np.array([ncol * [_ % 4] for _ in range(nrow)]), columns=range(ncol))

    idx_levels = [np.arange(min(nrow, 4))] * len(groupby) + [q]
    idx_codes = [[x for x in range(min(nrow, 4)) for _ in q]] * len(groupby) + [
        list(range(len(q))) * min(nrow, 4)
    ]
    expected_index = pd.MultiIndex(
        levels=idx_levels, codes=idx_codes, names=[*groupby, None]
    )
    expected_values = [
        [float(x)] * (ncol - len(groupby)) for x in range(min(nrow, 4)) for _ in q
    ]
    expected_columns = [x for x in range(ncol) if x not in groupby]
    expected = DataFrame(
        expected_values, index=expected_index, columns=expected_columns
    )
    result = df.groupby(groupby).quantile(q)

    tm.assert_frame_equal(result, expected)


def test_quantile_raises():
    # [FUNCTION DOC]
    # Quantile on a frame with only string value columns must raise a type error stating that the object or str dtype does not support the quantile operation.
    df = DataFrame([["foo", "a"], ["foo", "b"], ["foo", "c"]], columns=["key", "val"])

    msg = "dtype '(object|str)' does not support operation 'quantile'"
    with pytest.raises(TypeError, match=msg):
        df.groupby("key").quantile()


def test_quantile_out_of_bounds_q_raises():
    # [FUNCTION DOC]
    # Refs GitHub issue 27470: q values outside the 0 to 1 range (50 and -1) must raise a value error whose message reports the offending value as a float.
    # https://github.com/pandas-dev/pandas/issues/27470
    df = DataFrame({"a": [0, 0, 0, 1, 1, 1], "b": range(6)})
    g = df.groupby([0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="Got '50.0' instead"):
        g.quantile(50)

    with pytest.raises(ValueError, match="Got '-1.0' instead"):
        g.quantile(-1)


def test_quantile_missing_group_values_no_segfaults():
    # [FUNCTION DOC]
    # GH 28662: guards against segfaults when group keys contain missing values. Repeats the quantile one hundred times on a grouping with a NaN key; passes if nothing crashes. No result is checked.
    # GH 28662
    data = np.array([1.0, np.nan, 1.0])
    df = DataFrame({"key": data, "val": range(3)})

    # Random segfaults; would have been guaranteed in loop
    grp = df.groupby("key")
    for _ in range(100):
        grp.quantile()


@pytest.mark.parametrize(
    "key, val, expected_key, expected_val",
    [
        ([1.0, np.nan, 3.0, np.nan], range(4), [1.0, 3.0], [0.0, 2.0]),
        ([1.0, np.nan, 2.0, 2.0], range(4), [1.0, 2.0], [0.0, 2.5]),
        (["a", "b", "b", np.nan], range(4), ["a", "b"], [0, 1.5]),
        ([0], [42], [0], [42.0]),
        ([], [], np.array([], dtype="float64"), np.array([], dtype="float64")),
    ],
)
def test_quantile_missing_group_values_correct_results(
    key, val, expected_key, expected_val
):
    # [FUNCTION DOC]
    # GH 28662, GH 33200, GH 33569: parametrized over key/value inputs including NaN keys (dropped), string keys, a single row and empty input. Grouped quantile with an explicit 0.5 and with the default q must both equal the expected one-column frame indexed by the non-missing keys, named key.
    # GH 28662, GH 33200, GH 33569
    df = DataFrame({"key": key, "val": val})

    expected = DataFrame(
        expected_val, index=Index(expected_key, name="key"), columns=["val"]
    )

    grp = df.groupby("key")

    result = grp.quantile(0.5)
    tm.assert_frame_equal(result, expected)

    result = grp.quantile()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "values",
    [
        pd.array([1, 0, None] * 2, dtype="Int64"),
        pd.array([True, False, None] * 2, dtype="boolean"),
    ],
)
@pytest.mark.parametrize("q", [0.5, [0.0, 0.5, 1.0]])
def test_groupby_quantile_nullable_array(values, q):
    # [FUNCTION DOC]
    # Refs GitHub issue 33136: Int64 and boolean nullable arrays with missing values grouped by a string column; selecting column b and taking quantile with scalar q or a list of q. Expects a Float64 Series named b; a list q gives a two-level index (a, q) and values 0.0, 0.5, 1.0, a scalar q gives 0.5 for each group.
    # https://github.com/pandas-dev/pandas/issues/33136
    df = DataFrame({"a": ["x"] * 3 + ["y"] * 3, "b": values})
    result = df.groupby("a")["b"].quantile(q)

    if isinstance(q, list):
        idx = pd.MultiIndex.from_product((["x", "y"], q), names=["a", None])
        true_quantiles = [0.0, 0.5, 1.0]
    else:
        idx = Index(["x", "y"], name="a")
        true_quantiles = [0.5]

    expected = pd.Series(true_quantiles * 2, index=idx, name="b", dtype="Float64")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("q", [0.5, [0.0, 0.5, 1.0]])
@pytest.mark.parametrize("numeric_only", [True, False])
def test_groupby_quantile_raises_on_invalid_dtype(q, numeric_only):
    # [FUNCTION DOC]
    # Parametrized over scalar or list q and numeric_only. With numeric_only true, the non-numeric column c is dropped and result equals the quantile on column b only. With numeric_only false, a type error saying the dtype does not support quantile must be raised.
    df = DataFrame({"a": [1], "b": [2.0], "c": ["x"]})
    if numeric_only:
        result = df.groupby("a").quantile(q, numeric_only=numeric_only)
        expected = df.groupby("a")[["b"]].quantile(q)
        tm.assert_frame_equal(result, expected)
    else:
        msg = "dtype '.*' does not support operation 'quantile'"
        with pytest.raises(TypeError, match=msg):
            df.groupby("a").quantile(q, numeric_only=numeric_only)


def test_groupby_quantile_NA_float(any_float_dtype):
    # [FUNCTION DOC]
    # GH#42849: float dtype fixture (numpy and nullable). A group with one valid value and one missing value gives a median equal to the valid value, with scalar q and with a list of q (two-level index). Expected dtype is the nullable float dtype if the input is nullable, else float64 default.
    # GH#42849
    dtype = pd.Series([], dtype=any_float_dtype).dtype
    item = np.nan if isinstance(dtype, np.dtype) else pd.NA
    df = DataFrame({"x": [1, 1], "y": [0.2, item]}, dtype=any_float_dtype)
    result = df.groupby("x")["y"].quantile(0.5)
    exp_index = Index([1.0], dtype=any_float_dtype, name="x")

    if any_float_dtype in ["Float32", "Float64"]:
        expected_dtype = any_float_dtype
    else:
        expected_dtype = None

    expected = pd.Series([0.2], dtype=expected_dtype, index=exp_index, name="y")
    tm.assert_series_equal(result, expected)

    result = df.groupby("x")["y"].quantile([0.5, 0.75])
    expected = pd.Series(
        [0.2] * 2,
        index=pd.MultiIndex.from_product((exp_index, [0.5, 0.75]), names=["x", None]),
        name="y",
        dtype=expected_dtype,
    )
    tm.assert_series_equal(result, expected)


def test_groupby_quantile_NA_int(any_int_ea_dtype):
    # [FUNCTION DOC]
    # GH#42849: nullable integer dtype fixture. Quantile 0.5 of values 2 and 5 is 3.5 returned as Float64, for both the selected Series and the full frame; the index keeps the nullable integer dtype.
    # GH#42849
    df = DataFrame({"x": [1, 1], "y": [2, 5]}, dtype=any_int_ea_dtype)
    result = df.groupby("x")["y"].quantile(0.5)
    expected = pd.Series(
        [3.5],
        dtype="Float64",
        index=Index([1], name="x", dtype=any_int_ea_dtype),
        name="y",
    )
    tm.assert_series_equal(expected, result)

    result = df.groupby("x").quantile(0.5)
    expected = DataFrame(
        {"y": 3.5}, dtype="Float64", index=Index([1], name="x", dtype=any_int_ea_dtype)
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "interpolation, val1, val2", [("lower", 2, 2), ("higher", 2, 3), ("nearest", 2, 2)]
)
def test_groupby_quantile_all_na_group_masked(
    interpolation, val1, val2, any_numeric_ea_dtype
):
    # [FUNCTION DOC]
    # GH#37493: masked numeric dtypes with a group that is entirely missing. For interpolation modes lower, higher and nearest, the quantile for q of 0.5 and 0.7 on the valid group equals the supplied integer picks, and the all-NA group yields missing values; result keeps the input dtype and the index has the group key (masked dtype) and q level.
    # GH#37493
    df = DataFrame(
        {"a": [1, 1, 1, 2], "b": [1, 2, 3, pd.NA]}, dtype=any_numeric_ea_dtype
    )
    result = df.groupby("a").quantile(q=[0.5, 0.7], interpolation=interpolation)
    expected = DataFrame(
        {"b": [val1, val2, pd.NA, pd.NA]},
        dtype=any_numeric_ea_dtype,
        index=pd.MultiIndex.from_arrays(
            [pd.Series([1, 1, 2, 2], dtype=any_numeric_ea_dtype), [0.5, 0.7, 0.5, 0.7]],
            names=["a", None],
        ),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("interpolation", ["midpoint", "linear"])
def test_groupby_quantile_all_na_group_masked_interp(
    interpolation, any_numeric_ea_dtype
):
    # [FUNCTION DOC]
    # GH#37493: same all-missing group scenario for midpoint and linear interpolation with q of 0.5 and 0.75. Expected values 2.0 and 2.5 for the valid group and missing for the other; result dtype is Float32 for Float32 input and Float64 otherwise.
    # GH#37493
    df = DataFrame(
        {"a": [1, 1, 1, 2], "b": [1, 2, 3, pd.NA]}, dtype=any_numeric_ea_dtype
    )
    result = df.groupby("a").quantile(q=[0.5, 0.75], interpolation=interpolation)

    if any_numeric_ea_dtype == "Float32":
        expected_dtype = any_numeric_ea_dtype
    else:
        expected_dtype = "Float64"

    expected = DataFrame(
        {"b": [2.0, 2.5, pd.NA, pd.NA]},
        dtype=expected_dtype,
        index=pd.MultiIndex.from_arrays(
            [
                pd.Series([1, 1, 2, 2], dtype=any_numeric_ea_dtype),
                [0.5, 0.75, 0.5, 0.75],
            ],
            names=["a", None],
        ),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", ["Float64", "Float32"])
def test_groupby_quantile_allNA_column(dtype):
    # [FUNCTION DOC]
    # GH#42849: a column that is entirely NA of Float64 or Float32 dtype. The quantile 0.5 per group must be a single NA of the same dtype, with the index of the same dtype named x.
    # GH#42849
    df = DataFrame({"x": [1, 1], "y": [pd.NA] * 2}, dtype=dtype)
    result = df.groupby("x")["y"].quantile(0.5)
    expected = pd.Series(
        [pd.NA], dtype=dtype, index=Index([1.0], dtype=dtype), name="y"
    )
    expected.index.name = "x"
    tm.assert_series_equal(expected, result)


def test_groupby_timedelta_quantile():
    # [FUNCTION DOC]
    # GH 29485: quantile 0.99 of timedelta values grouped by an integer column returns timedelta values (0.99 and 2.99 seconds) indexed by the group.
    # GH: 29485
    tdi = pd.to_timedelta(np.arange(4), unit="s").as_unit("us")
    df = DataFrame({"value": tdi, "group": [1, 1, 2, 2]})
    result = df.groupby("group").quantile(0.99)
    expected = DataFrame(
        {
            "value": [
                pd.Timedelta("0 days 00:00:00.990000"),
                pd.Timedelta("0 days 00:00:02.990000"),
            ]
        },
        index=Index([1, 2], name="group"),
    )
    tm.assert_frame_equal(result, expected)


def test_timestamp_groupby_quantile(unit):
    # [FUNCTION DOC]
    # GH 33168: parametrized over the unit fixture. A UTC-aware datetime column floored to the hour is used as key with category and value columns; quantile of 0.2 and 0.8 yields interpolated float values with a two-level index (timestamp, q) on the distinct hours.
    # GH 33168
    dti = pd.date_range(
        start="2020-04-19 00:00:00", freq="1min", periods=100, tz="UTC", unit=unit
    ).floor("1h")
    df = DataFrame(
        {
            "timestamp": dti,
            "category": list(range(1, 101)),
            "value": list(range(101, 201)),
        }
    )

    result = df.groupby("timestamp").quantile([0.2, 0.8])

    mi = pd.MultiIndex.from_product([dti[::99], [0.2, 0.8]], names=("timestamp", None))
    expected = DataFrame(
        [
            {"category": 12.8, "value": 112.8},
            {"category": 48.2, "value": 148.2},
            {"category": 68.8, "value": 168.8},
            {"category": 92.2, "value": 192.2},
        ],
        index=mi,
    )

    tm.assert_frame_equal(result, expected)


def test_groupby_quantile_dt64tz_period():
    # [FUNCTION DOC]
    # GH#51373: a frame with datetime64, tz-aware datetime, period and timedelta columns (last row set to NaT) grouped by a repeating five-way key. Median per group must equal the group-by-group Series quantile assembled into a frame, with the index cast to integer.
    # GH#51373
    dti = pd.date_range("2016-01-01", periods=1000, unit="ns")
    df = pd.Series(dti).to_frame().copy()
    df[1] = dti.tz_localize("US/Pacific")
    df[2] = dti.to_period("D")
    df[3] = dti - dti[0]
    df.iloc[-1] = pd.NaT

    by = np.tile(np.arange(5), 200)
    gb = df.groupby(by)

    result = gb.quantile(0.5)

    # Check that we match the group-by-group result
    exp = {i: df.iloc[i::5].quantile(0.5) for i in range(5)}
    expected = DataFrame(exp).T.infer_objects()
    expected.index = expected.index.astype(int)

    tm.assert_frame_equal(result, expected)


def test_groupby_quantile_nonmulti_levels_order():
    # [FUNCTION DOC]
    # Non-regression for GH 53009: grouping a Series on one level of a three-level MultiIndex, sort false, with q of 0.2 and 0.8. Expects the group order B then A with the q level, and additionally that the resulting index levels stay unsorted.
    # Non-regression test for GH #53009
    ind = pd.MultiIndex.from_tuples(
        [
            (0, "a", "B"),
            (0, "a", "A"),
            (0, "b", "B"),
            (0, "b", "A"),
            (1, "a", "B"),
            (1, "a", "A"),
            (1, "b", "B"),
            (1, "b", "A"),
        ],
        names=["sample", "cat0", "cat1"],
    )
    ser = pd.Series(range(8), index=ind)
    result = ser.groupby(level="cat1", sort=False).quantile([0.2, 0.8])

    qind = pd.MultiIndex.from_tuples(
        [("B", 0.2), ("B", 0.8), ("A", 0.2), ("A", 0.8)], names=["cat1", None]
    )
    expected = pd.Series([1.2, 4.8, 2.2, 5.8], index=qind)

    tm.assert_series_equal(result, expected)

    # We need to check that index levels are not sorted
    expected_levels = pd.core.indexes.frozen.FrozenList([["B", "A"], [0.2, 0.8]])
    tm.assert_equal(result.index.levels, expected_levels)
