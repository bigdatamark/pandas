# [MODULE DOC]
# Tests for conversion of Series and Index objects to Python and numpy
# objects: iteration boxing to Python or pandas scalars, the backing values and
# array attributes, to_numpy (dtype, copy, na_value handling and keyword
# validation, also for MultiIndex Series and DataFrame), and the numpy array
# protocol for datetime Series. Uses fixtures from the suite conftest
# (index_or_series, unit, any_numpy_dtype and others). No import-time side effects.

import numpy as np
import pytest

from pandas.compat import HAS_PYARROW
from pandas.compat.numpy import np_version_gt2

from pandas.core.dtypes.dtypes import DatetimeTZDtype

import pandas as pd
from pandas import (
    CategoricalIndex,
    Series,
    Timedelta,
    Timestamp,
    date_range,
)
import pandas._testing as tm
from pandas.core.arrays import (
    DatetimeArray,
    IntervalArray,
    NumpyExtensionArray,
    PeriodArray,
    SparseArray,
    StringArray,
    TimedeltaArray,
)
from pandas.core.arrays.string_arrow import ArrowStringArray


class TestToIterable:
    # [CLASS DOC]
    # Tests that iterating or converting to lists yields Python or pandas scalar
    # types rather than numpy scalars. Class attribute dtypes lists pairs of a dtype
    # name and the expected scalar type (integer, float, Timestamp, Timedelta).
    # test that we convert an iterable to python types

    dtypes = [
        ("int8", int),
        ("int16", int),
        ("int32", int),
        ("int64", int),
        ("uint8", int),
        ("uint16", int),
        ("uint32", int),
        ("uint64", int),
        ("float16", float),
        ("float32", float),
        ("float64", float),
        ("datetime64[ns]", Timestamp),
        ("datetime64[ns, US/Eastern]", Timestamp),
        ("timedelta64[ns]", Timedelta),
    ]

    @pytest.mark.parametrize("dtype, rdtype", dtypes)
    @pytest.mark.parametrize(
        "method",
        [
            lambda x: x.tolist(),
            lambda x: x.to_list(),
            lambda x: list(x),
            lambda x: list(x.__iter__()),
        ],
        ids=["tolist", "to_list", "list", "iter"],
    )
    def test_iterable(self, index_or_series, method, dtype, rdtype):
        # [FUNCTION DOC]
        # Regression test (gh-10904, gh-13258). Parametrized over (dtype, expected
        # scalar type) and four conversion methods (tolist, to_list, list, iteration) and
        # the container (Index or Series). Builds a one-element object; verifies the first
        # element has the expected scalar type. For float16 and Index verifies instead that
        # construction raises a not-implemented error ("float16 indexes are not ...").
        # gh-10904
        # gh-13258
        # coerce iteration to underlying python / pandas types
        typ = index_or_series
        if dtype == "float16" and issubclass(typ, pd.Index):
            with pytest.raises(NotImplementedError, match="float16 indexes are not "):
                typ([1], dtype=dtype)
            return
        s = typ([1], dtype=dtype)
        result = method(s)[0]
        assert isinstance(result, rdtype)

    @pytest.mark.parametrize(
        "dtype, rdtype, obj",
        [
            ("object", object, "a"),
            ("object", int, 1),
            ("category", object, "a"),
            ("category", int, 1),
        ],
    )
    @pytest.mark.parametrize(
        "method",
        [
            lambda x: x.tolist(),
            lambda x: x.to_list(),
            lambda x: list(x),
            lambda x: list(x.__iter__()),
        ],
        ids=["tolist", "to_list", "list", "iter"],
    )
    def test_iterable_object_and_category(
        self, index_or_series, method, dtype, rdtype, obj
    ):
        # [FUNCTION DOC]
        # Same checks as the previous test for the dtypes object and category with
        # an integer or a string element: the first element obtained via each conversion
        # method has the expected Python type.
        # gh-10904
        # gh-13258
        # coerce iteration to underlying python / pandas types
        typ = index_or_series
        s = typ([obj], dtype=dtype)
        result = method(s)[0]
        assert isinstance(result, rdtype)

    @pytest.mark.parametrize("dtype, rdtype", dtypes)
    def test_iterable_items(self, dtype, rdtype):
        # [FUNCTION DOC]
        # Regression test (gh-13258). Series only. Verifies the value yielded by items
        # iteration has the boxed scalar type.
        # KNOWN ISSUE: the items iteration check is performed twice with identical code.
        # gh-13258
        # test if items yields the correct boxed scalars
        # this only applies to series
        s = Series([1], dtype=dtype)
        _, result = next(iter(s.items()))
        assert isinstance(result, rdtype)

        _, result = next(iter(s.items()))
        assert isinstance(result, rdtype)

    @pytest.mark.parametrize(
        "dtype, rdtype", [*dtypes, ("object", int), ("category", int)]
    )
    def test_iterable_map(self, index_or_series, dtype, rdtype):
        # [FUNCTION DOC]
        # Regression test (gh-13236). Parametrized over dtypes plus object and category
        # with integer. Verifies mapping the type function over a one-element object gives
        # the expected scalar class (float16 with Index must raise the not-implemented
        # error at construction instead).
        # gh-13236
        # coerce iteration to underlying python / pandas types
        typ = index_or_series
        if dtype == "float16" and issubclass(typ, pd.Index):
            with pytest.raises(NotImplementedError, match="float16 indexes are not "):
                typ([1], dtype=dtype)
            return
        s = typ([1], dtype=dtype)
        result = s.map(type)[0]
        if not isinstance(rdtype, tuple):
            rdtype = (rdtype,)
        assert result in rdtype

    @pytest.mark.parametrize(
        "method",
        [
            lambda x: x.tolist(),
            lambda x: x.to_list(),
            lambda x: list(x),
            lambda x: list(x.__iter__()),
        ],
        ids=["tolist", "to_list", "list", "iter"],
    )
    def test_categorial_datetimelike(self, method):
        # [FUNCTION DOC]
        # Parametrized over the four conversion methods. Verifies that a categorical index
        # of timestamps converts to a list of Timestamp objects.
        i = CategoricalIndex([Timestamp("1999-12-31"), Timestamp("2000-12-31")])

        result = method(i)[0]
        assert isinstance(result, Timestamp)

    def test_iter_box_dt64(self, unit):
        # [FUNCTION DOC]
        # Uses the unit fixture. Verifies iterating a datetime Series cast to the unit
        # yields tz-naive Timestamps equal to the originals with that unit.
        vals = [Timestamp("2011-01-01"), Timestamp("2011-01-02")]
        ser = Series(vals).dt.as_unit(unit)
        assert ser.dtype == f"datetime64[{unit}]"
        for res, exp in zip(ser, vals, strict=True):
            assert isinstance(res, Timestamp)
            assert res.tz is None
            assert res == exp
            assert res.unit == unit

    def test_iter_box_dt64tz(self, unit):
        # [FUNCTION DOC]
        # Uses the unit fixture. Verifies iterating a tz-aware datetime Series cast to the
        # unit yields Timestamps with the same timezone, values and unit.
        vals = [
            Timestamp("2011-01-01", tz="US/Eastern"),
            Timestamp("2011-01-02", tz="US/Eastern"),
        ]
        ser = Series(vals).dt.as_unit(unit)

        assert ser.dtype == f"datetime64[{unit}, US/Eastern]"
        for res, exp in zip(ser, vals, strict=True):
            assert isinstance(res, Timestamp)
            assert res.tz == exp.tz
            assert res == exp
            assert res.unit == unit

    def test_iter_box_timedelta64(self, unit):
        # [FUNCTION DOC]
        # Uses the unit fixture. Verifies iterating a timedelta Series cast to the unit
        # yields Timedelta objects equal to the originals with that unit.
        # timedelta
        vals = [Timedelta("1 days"), Timedelta("2 days")]
        ser = Series(vals).dt.as_unit(unit)
        assert ser.dtype == f"timedelta64[{unit}]"
        for res, exp in zip(ser, vals, strict=True):
            assert isinstance(res, Timedelta)
            assert res == exp
            assert res.unit == unit

    def test_iter_box_period(self):
        # [FUNCTION DOC]
        # Verifies iterating a monthly period Series yields Period objects equal to
        # the originals whose frequency compares equal to ME.
        # period
        vals = [pd.Period("2011-01-01", freq="M"), pd.Period("2011-01-02", freq="M")]
        s = Series(vals)
        assert s.dtype == "Period[M]"
        for res, exp in zip(s, vals, strict=True):
            assert isinstance(res, pd.Period)
            assert res.freq == "ME"
            assert res == exp


@pytest.mark.parametrize(
    "arr, expected_type, dtype",
    [
        (np.array([0, 1], dtype=np.int64), np.ndarray, "int64"),
        (np.array(["a", "b"]), np.ndarray, "object"),
        (pd.Categorical(["a", "b"]), pd.Categorical, "category"),
        (
            pd.DatetimeIndex(["2017", "2018"], tz="US/Central"),
            DatetimeArray,
            "datetime64[ns, US/Central]",
        ),
        (
            pd.PeriodIndex([2018, 2019], freq="Y"),
            PeriodArray,
            pd.core.dtypes.dtypes.PeriodDtype("Y-DEC"),
        ),
        (pd.IntervalIndex.from_breaks([0, 1, 2]), IntervalArray, "interval"),
        (
            pd.DatetimeIndex(["2017", "2018"]),
            DatetimeArray,
            "datetime64[ns]",
        ),
        (
            pd.TimedeltaIndex([10**10]),
            TimedeltaArray,
            "m8[ns]",
        ),
    ],
)
def test_values_consistent(arr, expected_type, dtype, using_infer_string):
    # [FUNCTION DOC]
    # Parametrized over (array-like, expected values class, dtype) for numpy, categorical,
    # datetime (naive and aware), period, interval and timedelta inputs. Verifies that
    # the private values of a Series and of an Index built from the same input have the
    # expected class (string array types when the infer-string mode applies for object
    # data), are of the same class and are equal.
    if using_infer_string and dtype == "object":
        expected_type = ArrowStringArray if HAS_PYARROW else StringArray
    l_values = Series(arr)._values
    r_values = pd.Index(arr)._values
    assert type(l_values) is expected_type
    assert type(l_values) is type(r_values)

    tm.assert_equal(l_values, r_values)


@pytest.mark.parametrize("arr", [np.array([1, 2, 3])])
def test_numpy_array(arr):
    # [FUNCTION DOC]
    # Verifies the array attribute of an integer Series is a NumpyExtensionArray
    # equal to the one built from the numpy array.
    ser = Series(arr)
    result = ser.array
    expected = NumpyExtensionArray(arr)
    tm.assert_extension_array_equal(result, expected)


def test_numpy_array_all_dtypes(any_numpy_dtype):
    # [FUNCTION DOC]
    # Uses the any_numpy_dtype fixture. Verifies the array attribute of an empty
    # Series is a DatetimeArray for datetime64, a TimedeltaArray for timedelta64
    # and otherwise a NumpyExtensionArray.
    ser = Series(dtype=any_numpy_dtype)
    result = ser.array
    if np.dtype(any_numpy_dtype).kind == "M":
        assert isinstance(result, DatetimeArray)
    elif np.dtype(any_numpy_dtype).kind == "m":
        assert isinstance(result, TimedeltaArray)
    else:
        assert isinstance(result, NumpyExtensionArray)


@pytest.mark.parametrize(
    "arr, attr",
    [
        (pd.Categorical(["a", "b"]), "_codes"),
        (PeriodArray._from_sequence(["2000", "2001"], dtype="period[D]"), "_ndarray"),
        (pd.array([0, pd.NA], dtype="Int64"), "_data"),
        (IntervalArray.from_breaks([0, 1]), "_left"),
        (SparseArray([0, 1]), "_sparse_values"),
        (
            DatetimeArray._from_sequence(np.array([1, 2], dtype="datetime64[ns]")),
            "_ndarray",
        ),
        # tz-aware Datetime
        (
            DatetimeArray._from_sequence(
                np.array(
                    ["2000-01-01T12:00:00", "2000-01-02T12:00:00"], dtype="M8[ns]"
                ),
                dtype=DatetimeTZDtype(tz="US/Central"),
            ),
            "_ndarray",
        ),
    ],
)
def test_array(arr, attr, index_or_series):
    # [FUNCTION DOC]
    # Parametrized over arrays of different extension types and the name of an
    # inner storage attribute; uses the index_or_series fixture. Verifies that
    # wrapping the array without copying and reading its array attribute yields an
    # object whose inner storage shares memory with the original.
    box = index_or_series

    result = box(arr, copy=False).array

    if attr:
        arr = getattr(arr, attr)
        result = getattr(result, attr)

    assert np.shares_memory(result, arr)


def test_array_multiindex_raises():
    # [FUNCTION DOC]
    # Verifies the array attribute of a MultiIndex raises a value error ("MultiIndex has no
    # single backing array").
    idx = pd.MultiIndex.from_product([["A"], ["a", "b"]])
    msg = "MultiIndex has no single backing array"
    with pytest.raises(ValueError, match=msg):
        idx.array


@pytest.mark.parametrize(
    "arr, expected, zero_copy",
    [
        (np.array([1, 2], dtype=np.int64), np.array([1, 2], dtype=np.int64), True),
        (pd.Categorical(["a", "b"]), np.array(["a", "b"], dtype=object), False),
        (
            pd.core.arrays.period_array(["2000", "2001"], freq="D"),
            np.array([pd.Period("2000", freq="D"), pd.Period("2001", freq="D")]),
            False,
        ),
        (pd.array([0, pd.NA], dtype="Int64"), np.array([0, np.nan]), False),
        (
            IntervalArray.from_breaks([0, 1, 2]),
            np.array([pd.Interval(0, 1), pd.Interval(1, 2)], dtype=object),
            False,
        ),
        (SparseArray([0, 1]), np.array([0, 1], dtype=np.int64), False),
        # tz-naive datetime
        (
            DatetimeArray._from_sequence(np.array(["2000", "2001"], dtype="M8[ns]")),
            np.array(["2000", "2001"], dtype="M8[ns]"),
            True,
        ),
        # tz-aware stays tz`-aware
        (
            DatetimeArray._from_sequence(
                np.array(["2000-01-01T06:00:00", "2000-01-02T06:00:00"], dtype="M8[ns]")
            )
            .tz_localize("UTC")
            .tz_convert("US/Central"),
            np.array(
                [
                    Timestamp("2000-01-01", tz="US/Central"),
                    Timestamp("2000-01-02", tz="US/Central"),
                ]
            ),
            False,
        ),
        # Timedelta
        (
            TimedeltaArray._from_sequence(
                np.array([0, 3600000000000], dtype="i8").view("m8[ns]"),
                dtype=np.dtype("m8[ns]"),
            ),
            np.array([0, 3600000000000], dtype="m8[ns]"),
            True,
        ),
        # GH#26406 tz is preserved in Categorical[dt64tz]
        (
            pd.Categorical(date_range("2016-01-01", periods=2, tz="US/Pacific")),
            np.array(
                [
                    Timestamp("2016-01-01", tz="US/Pacific"),
                    Timestamp("2016-01-02", tz="US/Pacific"),
                ]
            ),
            False,
        ),
    ],
)
def test_to_numpy(arr, expected, zero_copy, index_or_series_or_array, using_nan_is_na):
    # [FUNCTION DOC]
    # Parametrized over (array, expected numpy result, whether zero-copy is
    # possible). Uses index_or_series_or_array and using_nan_is_na fixtures. Wrapping the
    # array (no warning allowed), verifies to_numpy and the numpy asarray function give
    # the expected values (missing values stay NA objects in object arrays when
    # NaN is not treated as NA), that copy true yields independent copies and, on numpy
    # 2 or newer, that copy false either raises a value error ("Unable to avoid copy
    # while creating") or, for zero-copy types, shares data between calls.
    if not using_nan_is_na and arr[-1] is pd.NA:
        expected = np.array([0, pd.NA], dtype=object)

    box = index_or_series_or_array

    with tm.assert_produces_warning(None):
        thing = box(arr)

    result = thing.to_numpy()
    tm.assert_numpy_array_equal(result, expected)

    result = np.asarray(thing)
    tm.assert_numpy_array_equal(result, expected)

    # Additionally, we check the `copy=` semantics for array/asarray
    # (these are implemented by us via `__array__`).
    result_cp1 = np.array(thing, copy=True)
    result_cp2 = np.array(thing, copy=True)
    # When called with `copy=True` NumPy/we should ensure a copy was made
    assert not np.may_share_memory(result_cp1, result_cp2)

    if not np_version_gt2:
        # copy=False semantics are only supported in NumPy>=2.
        return

    if not zero_copy:
        with pytest.raises(ValueError, match="Unable to avoid copy while creating"):
            # An error is always acceptable for `copy=False`
            np.array(thing, copy=False)

    else:
        result_nocopy1 = np.array(thing, copy=False)
        result_nocopy2 = np.array(thing, copy=False)
        # If copy=False was given, these must share the same data
        assert np.may_share_memory(result_nocopy1, result_nocopy2)


@pytest.mark.parametrize("as_series", [True, False])
@pytest.mark.parametrize(
    "arr", [np.array([1, 2, 3], dtype="int64"), np.array(["a", "b", "c"], dtype=object)]
)
def test_to_numpy_copy(arr, as_series, using_infer_string):
    # [FUNCTION DOC]
    # Parametrized over numpy int and object arrays and Series versus Index. Verifies
    # to_numpy by default and with copy false shares memory with the original data
    # (except pyarrow-backed strings under infer-string mode, which cannot), while copy true
    # does not.
    obj = pd.Index(arr, copy=False)
    if as_series:
        obj = Series(obj.values, copy=False)

    # no copy by default
    result = obj.to_numpy()
    if using_infer_string and arr.dtype == object and obj.dtype.storage == "pyarrow":
        assert np.shares_memory(arr, result) is False
    else:
        assert np.shares_memory(arr, result) is True

    result = obj.to_numpy(copy=False)
    if using_infer_string and arr.dtype == object and obj.dtype.storage == "pyarrow":
        assert np.shares_memory(arr, result) is False
    else:
        assert np.shares_memory(arr, result) is True

    # copy=True
    result = obj.to_numpy(copy=True)
    assert np.shares_memory(arr, result) is False


@pytest.mark.parametrize("as_series", [True, False])
def test_to_numpy_dtype(as_series):
    # [FUNCTION DOC]
    # Parametrized over Series versus Index. Verifies to_numpy of a tz-aware datetime
    # object gives object Timestamps by default and for object dtype, and with
    # datetime64 nanosecond dtype gives UTC wall values.
    tz = "US/Eastern"
    obj = pd.DatetimeIndex(["2000", "2001"], tz=tz)
    if as_series:
        obj = Series(obj)

    # preserve tz by default
    result = obj.to_numpy()
    expected = np.array(
        [Timestamp("2000", tz=tz), Timestamp("2001", tz=tz)], dtype=object
    )
    tm.assert_numpy_array_equal(result, expected)

    result = obj.to_numpy(dtype="object")
    tm.assert_numpy_array_equal(result, expected)

    result = obj.to_numpy(dtype="M8[ns]")
    expected = np.array(["2000-01-01T05", "2001-01-01T05"], dtype="M8[ns]")
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "values, dtype, na_value, expected",
    [
        ([1, 2, None], "float64", 0, [1.0, 2.0, 0.0]),
        (
            [Timestamp("2000").as_unit("s"), Timestamp("2000").as_unit("s"), pd.NaT],
            None,
            Timestamp("2000").as_unit("s"),
            [np.datetime64("2000-01-01T00:00:00", "s")] * 3,
        ),
    ],
)
def test_to_numpy_na_value_numpy_dtype(
    index_or_series, values, dtype, na_value, expected
):
    # [FUNCTION DOC]
    # Parametrized over (values, dtype, na_value, expected). Uses index_or_series.
    # Verifies to_numpy with a numpy dtype and an na_value replaces missing values
    # with it, for floats (zero fill) and second-unit datetimes.
    obj = index_or_series(values)
    result = obj.to_numpy(dtype=dtype, na_value=na_value)
    expected = np.array(expected)
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "data, multiindex, dtype, na_value, expected",
    [
        (
            [1, 2, None, 4],
            [(0, "a"), (0, "b"), (1, "b"), (1, "c")],
            float,
            None,
            [1.0, 2.0, np.nan, 4.0],
        ),
        (
            [1, 2, None, 4],
            [(0, "a"), (0, "b"), (1, "b"), (1, "c")],
            float,
            np.nan,
            [1.0, 2.0, np.nan, 4.0],
        ),
        (
            [1.0, 2.0, np.nan, 4.0],
            [("a", 0), ("a", 1), ("a", 2), ("b", 0)],
            int,
            0,
            [1, 2, 0, 4],
        ),
        (
            [Timestamp("2000").as_unit("s"), Timestamp("2000").as_unit("s"), pd.NaT],
            [
                (0, Timestamp("2021").as_unit("s")),
                (0, Timestamp("2022").as_unit("s")),
                (1, Timestamp("2000").as_unit("s")),
            ],
            None,
            Timestamp("2000").as_unit("s"),
            [np.datetime64("2000-01-01T00:00:00", "s")] * 3,
        ),
    ],
)
def test_to_numpy_multiindex_series_na_value(
    data, multiindex, dtype, na_value, expected
):
    # [FUNCTION DOC]
    # Parametrized over (data, MultiIndex tuples, dtype, na_value, expected). Verifies
    # a Series with a MultiIndex converts with to_numpy honoring dtype and na_value,
    # including NaN, a zero for integer dtype, and second-unit datetimes.
    index = pd.MultiIndex.from_tuples(multiindex)
    series = Series(data, index=index)
    result = series.to_numpy(dtype=dtype, na_value=na_value)
    expected = np.array(expected)
    tm.assert_numpy_array_equal(result, expected)


def test_to_numpy_kwargs_raises():
    # [FUNCTION DOC]
    # Verifies to_numpy with an unknown keyword raises a type error ("got an unexpected
    # keyword argument"), for numpy-backed and extension Series.
    # numpy
    s = Series([1, 2, 3])
    msg = r"to_numpy\(\) got an unexpected keyword argument 'foo'"
    with pytest.raises(TypeError, match=msg):
        s.to_numpy(foo=True)

    # extension
    s = Series([1, 2, 3], dtype="Int64")
    with pytest.raises(TypeError, match=msg):
        s.to_numpy(foo=True)


@pytest.mark.parametrize(
    "data",
    [
        {"a": [1, 2, 3], "b": [1, 2, None]},
        {"a": np.array([1, 2, 3]), "b": np.array([1, 2, np.nan])},
        {"a": pd.array([1, 2, 3]), "b": pd.array([1, 2, None])},
    ],
)
@pytest.mark.parametrize("dtype, na_value", [(float, np.nan), (object, None)])
def test_to_numpy_dataframe_na_value(data, dtype, na_value):
    # [FUNCTION DOC]
    # Regression test (GitHub 33820). Parametrized over three kinds of two-column data
    # with a missing value and over (dtype, na_value) of float with NaN and object with
    # None. Verifies DataFrame to_numpy with dtype and na_value fills the missing cell
    # with the given value.
    # https://github.com/pandas-dev/pandas/issues/33820
    df = pd.DataFrame(data)
    result = df.to_numpy(dtype=dtype, na_value=na_value)
    expected = np.array([[1, 1], [2, 2], [3, na_value]], dtype=dtype)
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "data, expected_data",
    [
        (
            {"a": pd.array([1, 2, None])},
            [[1.0], [2.0], [np.nan]],
        ),
        (
            {"a": [1, 2, 3], "b": [1, 2, 3]},
            [[1, 1], [2, 2], [3, 3]],
        ),
    ],
)
def test_to_numpy_dataframe_single_block(data, expected_data):
    # [FUNCTION DOC]
    # Regression test (GitHub 33820). Parametrized over single-block frames (one
    # masked integer column, or two integer columns). Verifies to_numpy to float with NaN
    # as the missing value gives the expected float values.
    # https://github.com/pandas-dev/pandas/issues/33820
    df = pd.DataFrame(data)
    result = df.to_numpy(dtype=float, na_value=np.nan)
    expected = np.array(expected_data, dtype=float)
    tm.assert_numpy_array_equal(result, expected)


def test_to_numpy_dataframe_single_block_no_mutate():
    # [FUNCTION DOC]
    # Regression test (GitHub 33820). Verifies calling DataFrame to_numpy with
    # na_value on a single-block frame does not modify the frame.
    # https://github.com/pandas-dev/pandas/issues/33820
    result = pd.DataFrame(np.array([1.0, 2.0, np.nan]))
    expected = pd.DataFrame(np.array([1.0, 2.0, np.nan]))
    result.to_numpy(na_value=0.0)
    tm.assert_frame_equal(result, expected)


class TestAsArray:
    # [CLASS DOC]
    # Tests of the numpy array protocol for datetime Series.
    @pytest.mark.parametrize("tz", [None, "US/Central"])
    def test_asarray_object_dt64(self, tz):
        # [FUNCTION DOC]
        # Parametrized over tz (none, US/Central). Verifies converting a datetime Series to
        # numpy object dtype gives Timestamp objects (with the timezone) and emits no
        # warning.
        ser = Series(date_range("2000", periods=2, tz=tz))

        with tm.assert_produces_warning(None):
            # Future behavior (for tzaware case) with no warning
            result = np.asarray(ser, dtype=object)

        expected = np.array(
            [Timestamp("2000-01-01", tz=tz), Timestamp("2000-01-02", tz=tz)]
        )
        tm.assert_numpy_array_equal(result, expected)

    def test_asarray_tz_naive(self):
        # [FUNCTION DOC]
        # Verifies converting a tz-naive nanosecond Series gives datetime64 nanosecond
        # values without a warning.
        # This shouldn't produce a warning.
        ser = Series(date_range("2000", periods=2, unit="ns"))
        expected = np.array(["2000-01-01", "2000-01-02"], dtype="M8[ns]")
        result = np.asarray(ser)

        tm.assert_numpy_array_equal(result, expected)

    def test_asarray_tz_aware(self):
        # [FUNCTION DOC]
        # Verifies converting a tz-aware Series with datetime64 nanosecond dtype gives the
        # UTC values, whether the dtype is spelled out or abbreviated.
        tz = "US/Central"
        ser = Series(date_range("2000", periods=2, tz=tz))
        expected = np.array(["2000-01-01T06", "2000-01-02T06"], dtype="M8[ns]")
        result = np.asarray(ser, dtype="datetime64[ns]")

        tm.assert_numpy_array_equal(result, expected)

        # Old behavior with no warning
        result = np.asarray(ser, dtype="M8[ns]")

        tm.assert_numpy_array_equal(result, expected)
