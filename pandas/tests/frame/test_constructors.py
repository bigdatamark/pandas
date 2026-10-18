# [MODULE DOC]
# Tests for the constructor of the tabular container: construction from
# dictionaries, lists, tuples, records, NumPy arrays (plain, structured, masked),
# Series, generators, dataclasses, named tuples, ranges, and extension arrays,
# with and without explicit index, columns and dtype; dtype inference and
# preservation (integers, floats, datetime-like, categorical, sparse, period,
# interval, strings, object), copy semantics, missing-value handling, overflow,
# and error cases. Defines lists of mixed float and integer dtype names as
# module constants, and a number of tests grouped by theme into several classes
# (general constructor, constructor on non-dtype-specific data, and dtype-specific
# construction). Uses fixtures from the test-suite conftest; no import-time side
# effects.
import array
from collections import (
    OrderedDict,
    abc,
    defaultdict,
    namedtuple,
)
from collections.abc import Iterator
from dataclasses import make_dataclass
from datetime import (
    date,
    datetime,
    timedelta,
)
import functools
import re
import zoneinfo

import numpy as np
from numpy import ma
from numpy.ma import mrecords
import pytest

from pandas._libs import lib
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IntCastingNaNError

from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    IntervalDtype,
    NumpyEADtype,
    PeriodDtype,
)

import pandas as pd
from pandas import (
    Categorical,
    CategoricalIndex,
    DataFrame,
    DatetimeIndex,
    Index,
    Interval,
    MultiIndex,
    Period,
    RangeIndex,
    Series,
    Timedelta,
    Timestamp,
    cut,
    date_range,
    isna,
)
import pandas._testing as tm
from pandas.arrays import (
    DatetimeArray,
    IntervalArray,
    PeriodArray,
    SparseArray,
    TimedeltaArray,
)

MIXED_FLOAT_DTYPES = ["float16", "float32", "float64"]
MIXED_INT_DTYPES = [
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
]


class TestDataFrameConstructors:
    # [CLASS DOC]
    # Main test group for the container's constructor: input kinds (dictionaries,
    # sequences, arrays, records), dtype handling, empty inputs, copying and errors.
    # Stateless.
    def test_constructor_from_ndarray_with_str_dtype(self):
        # [FUNCTION DOC]
        # Constructing from a 2-D integer array with a string dtype converts each element
        # to a string and keeps the 2-D shape (the array must be flattened and
        # reshaped around the string conversion), equal to building from the string-cast
        # array.
        # If we don't ravel/reshape around ensure_str_array, we end up
        #  with an array of strings each of which is e.g. "[0 1 2]"
        arr = np.arange(12).reshape(4, 3)
        df = DataFrame(arr, dtype=str)
        expected = DataFrame(arr.astype(str), dtype="str")
        tm.assert_frame_equal(df, expected)

    def test_constructor_from_2d_datetimearray(self):
        # [FUNCTION DOC]
        # Constructing from a two-dimensional timezone-aware datetime array gives a frame
        # equal to building from its columns, and keeps a single internal block
        # (GH#44724: avoid performance loss from splitting).
        dti = date_range("2016-01-01", periods=6, tz="US/Pacific")
        dta = dti._data.reshape(3, 2)

        df = DataFrame(dta)
        expected = DataFrame({0: dta[:, 0], 1: dta[:, 1]})
        tm.assert_frame_equal(df, expected)
        # GH#44724 big performance hit if we de-consolidate
        assert len(df._mgr.blocks) == 1

    def test_constructor_dict_with_tzaware_scalar(self):
        # [FUNCTION DOC]
        # GH#42505: a dictionary with a timezone-aware scalar timestamp (nanosecond
        # unit) with an index of length one, or alongside a list-valued column, builds a
        # datetime column equal to the list-of-one form.
        # GH#42505
        dt = Timestamp("2019-11-03 01:00:00-0700").tz_convert("America/Los_Angeles")
        dt = dt.as_unit("ns")

        df = DataFrame({"dt": dt}, index=[0])
        expected = DataFrame({"dt": [dt]})
        tm.assert_frame_equal(df, expected, check_index_type=False)

        # Non-homogeneous
        df = DataFrame({"dt": dt, "value": [1]})
        expected = DataFrame({"dt": [dt], "value": [1]})
        tm.assert_frame_equal(df, expected)

    def test_construct_ndarray_with_nas_and_int_dtype(self):
        # [FUNCTION DOC]
        # GH#26919: constructing from an array with NaN with an integer dtype raises an
        # integer-casting-NaN error mentioning non-finite values, for a frame and for a
        # Series of the first row.
        # GH#26919 match Series by not casting np.nan to meaningless int
        arr = np.array([[1, np.nan], [2, 3]])
        msg = r"Cannot convert non-finite values \(NA or inf\) to integer"
        with pytest.raises(IntCastingNaNError, match=msg):
            DataFrame(arr, dtype="i8")

        # check this matches Series behavior
        with pytest.raises(IntCastingNaNError, match=msg):
            Series(arr[0], dtype="i8", name=0)

    def test_construct_from_list_of_datetimes(self):
        # [FUNCTION DOC]
        # Constructing from a list of current datetimes gives a microsecond-resolution
        # datetime column.
        df = DataFrame([datetime.now(), datetime.now()])
        assert df[0].dtype == np.dtype("M8[us]")

    def test_constructor_from_tzaware_datetimeindex(self):
        # [FUNCTION DOC]
        # GH#6032: a timezone-aware datetime index converted to a Series keeps the
        # timezone dtype (it is not cast to object); the expected Series is built from
        # the object array of its timestamps.
        # KNOWN ISSUE: despite the name, only the Series constructor is exercised; the frame constructor is not used.
        # don't cast a DatetimeIndex WITH a tz, leave as object
        # GH#6032
        naive = DatetimeIndex(["2013-1-1 13:00", "2013-1-2 14:00"], name="B")
        idx = naive.tz_localize("US/Pacific")

        expected = Series(np.array(idx.tolist(), dtype="object"), name="B")
        assert expected.dtype == idx.dtype

        # convert index to series
        result = Series(idx)
        tm.assert_series_equal(result, expected)

    def test_columns_with_leading_underscore_work_with_to_dict(self):
        # [FUNCTION DOC]
        # A column name with a leading underscore survives conversion to records.
        col_underscore = "_b"
        df = DataFrame({"a": [1, 2], col_underscore: [3, 4]})
        d = df.to_dict(orient="records")

        ref_d = [{"a": 1, col_underscore: 3}, {"a": 2, col_underscore: 4}]

        assert ref_d == d

    def test_columns_with_leading_number_and_underscore_work_with_to_dict(self):
        # [FUNCTION DOC]
        # A column name starting with a digit and an underscore survives conversion to
        # records.
        col_with_num = "1_b"
        df = DataFrame({"a": [1, 2], col_with_num: [3, 4]})
        d = df.to_dict(orient="records")

        ref_d = [{"a": 1, col_with_num: 3}, {"a": 2, col_with_num: 4}]

        assert ref_d == d

    def test_array_of_dt64_nat_with_td64dtype_raises(self, frame_or_series):
        # [FUNCTION DOC]
        # GH#39462: parametrized over frame/Series. An object array holding a datetime
        # NaT with a timedelta dtype raises a type error saying it is an invalid type for a
        # timedelta scalar.
        # GH#39462
        nat = np.datetime64("NaT", "ns")
        arr = np.array([nat], dtype=object)
        if frame_or_series is DataFrame:
            arr = arr.reshape(1, 1)

        msg = "Invalid type for timedelta scalar: <class 'numpy.datetime64'>"
        with pytest.raises(TypeError, match=msg):
            frame_or_series(arr, dtype="m8[ns]")

    @pytest.mark.parametrize("kind", ["m", "M"])
    def test_datetimelike_values_with_object_dtype(self, kind, frame_or_series):
        # [FUNCTION DOC]
        # Parametrized over timedelta/datetime kinds and frame/Series. Building from
        # datetime-like NumPy values with object dtype yields object storage whose elements
        # are timestamp or timedelta objects (not native datetimes); checked through
        # direct construction, wrapping a container of the same, with the NumPy-extension
        # object dtype, and (frames) from a list of Series.
        # with dtype=object, we should cast dt64 values to Timestamps, not pydatetimes
        if kind == "M":
            dtype = "M8[ns]"
            scalar_type = Timestamp
        else:
            dtype = "m8[ns]"
            scalar_type = Timedelta

        arr = np.arange(6, dtype="i8").view(dtype).reshape(3, 2)
        if frame_or_series is Series:
            arr = arr[:, 0]

        obj = frame_or_series(arr, dtype=object)
        assert obj._mgr.blocks[0].values.dtype == object
        assert isinstance(obj._mgr.blocks[0].values.ravel()[0], scalar_type)

        # go through a different path in internals.construction
        obj = frame_or_series(frame_or_series(arr), dtype=object)
        assert obj._mgr.blocks[0].values.dtype == object
        assert isinstance(obj._mgr.blocks[0].values.ravel()[0], scalar_type)

        obj = frame_or_series(frame_or_series(arr), dtype=NumpyEADtype(object))
        assert obj._mgr.blocks[0].values.dtype == object
        assert isinstance(obj._mgr.blocks[0].values.ravel()[0], scalar_type)

        if frame_or_series is DataFrame:
            # other paths through internals.construction
            sers = [Series(x) for x in arr]
            obj = frame_or_series(sers, dtype=object)
            assert obj._mgr.blocks[0].values.dtype == object
            assert isinstance(obj._mgr.blocks[0].values.ravel()[0], scalar_type)

    def test_series_with_name_not_matching_column(self):
        # [FUNCTION DOC]
        # GH#9232: a Series named differently from the requested column labels yields an
        # empty frame with those columns (the Series is dropped from the selection).
        # GH#9232
        x = Series(range(5), name=1)
        y = Series(range(5), name=0)

        result = DataFrame(x, columns=[0])
        expected = DataFrame([], columns=[0])
        tm.assert_frame_equal(result, expected)

        result = DataFrame(y, columns=[1])
        expected = DataFrame([], columns=[1])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "constructor",
        [
            lambda: DataFrame(),
            lambda: DataFrame(None),
            lambda: DataFrame(()),
            lambda: DataFrame([]),
            lambda: DataFrame(_ for _ in []),
            lambda: DataFrame(range(0)),
            lambda: DataFrame(data=None),
            lambda: DataFrame(data=()),
            lambda: DataFrame(data=[]),
            lambda: DataFrame(data=(_ for _ in [])),
            lambda: DataFrame(data=range(0)),
        ],
    )
    def test_empty_constructor(self, constructor):
        # [FUNCTION DOC]
        # Parametrized over eleven ways of passing nothing (no argument, none, empty
        # tuple/list/generator/range, positional or by keyword). Every one gives an
        # empty frame with no rows and columns equal to a plain empty frame.
        expected = DataFrame()
        result = constructor()
        assert len(result.index) == 0
        assert len(result.columns) == 0
        tm.assert_frame_equal(result, expected)

    def test_empty_constructor_object_index(self):
        # [FUNCTION DOC]
        # An empty dictionary gives a frame with zero-length range index and range
        # columns (type of index checked strictly).
        expected = DataFrame(index=RangeIndex(0), columns=RangeIndex(0))
        result = DataFrame({})
        assert len(result.index) == 0
        assert len(result.columns) == 0
        tm.assert_frame_equal(result, expected, check_index_type=True)

    @pytest.mark.parametrize(
        "emptylike,expected_index,expected_columns",
        [
            ([[]], RangeIndex(1), RangeIndex(0)),
            ([[], []], RangeIndex(2), RangeIndex(0)),
            ([(_ for _ in [])], RangeIndex(1), RangeIndex(0)),
        ],
    )
    def test_emptylike_constructor(self, emptylike, expected_index, expected_columns):
        # [FUNCTION DOC]
        # Parametrized over empty-like inputs (a list with one empty list, two empty
        # lists, a list with an empty generator) and the expected row and column ranges:
        # the number of rows follows the number of inner items and there are no columns.
        expected = DataFrame(index=expected_index, columns=expected_columns)
        result = DataFrame(emptylike)
        tm.assert_frame_equal(result, expected)

    def test_constructor_mixed(self, float_string_frame, using_infer_string):
        # [FUNCTION DOC]
        # For the mixed float/string fixture, the string column has object dtype (or
        # the string dtype when inferred-string option is active).
        dtype = "str" if using_infer_string else np.object_
        assert float_string_frame["foo"].dtype == dtype

    def test_constructor_cast_failure(self):
        # [FUNCTION DOC]
        # Since 2.0, a dtype that cannot be respected raises: strings to float64 raise
        # a value error about conversion; a frame can take a nested list as a column but
        # assigning a 2-D array as a column raises a value error saying a 1-D array is
        # expected (GH 3010).
        # as of 2.0, we raise if we can't respect "dtype", previously we
        #  silently ignored
        msg = "could not convert string to float"
        with pytest.raises(ValueError, match=msg):
            DataFrame({"a": ["a", "b", "c"]}, dtype=np.float64)

        # GH 3010, constructing with odd arrays
        df = DataFrame(np.ones((4, 2)))

        # this is ok
        df["foo"] = np.ones((4, 2)).tolist()

        # this is not ok
        msg = "Expected a 1D array, got an array with shape \\(4, 2\\)"
        with pytest.raises(ValueError, match=msg):
            df["test"] = np.ones((4, 2))

        # this is ok
        df["foo2"] = np.ones((4, 2)).tolist()

    def test_constructor_dtype_copy(self):
        # [FUNCTION DOC]
        # Constructing from a frame with a dtype and an explicit copy produces a frame
        # whose modifications do not touch the original column.
        orig_df = DataFrame({"col1": [1.0], "col2": [2.0], "col3": [3.0]})

        new_df = DataFrame(orig_df, dtype=float, copy=True)

        new_df["col1"] = 200.0
        assert orig_df["col1"][0] == 1.0

    def test_constructor_dtype_nocast_view_dataframe(self):
        # [FUNCTION DOC]
        # Constructing from a frame with the same dtype and modifying the new frame in
        # place leaves the original value unchanged.
        # KNOWN ISSUE: the name and variable say the result should be a view, but the assertion checks that the original is unchanged (copy semantics).
        df = DataFrame([[1, 2]])
        should_be_view = DataFrame(df, dtype=df[0].dtype)
        should_be_view.iloc[0, 0] = 99
        assert df.values[0, 0] == 1

    def test_constructor_dtype_nocast_view_2d_array(self):
        # [FUNCTION DOC]
        # Constructing from the 2-D values of an integer frame with a matching dtype
        # gives a single block that is C-contiguous.
        df = DataFrame([[1, 2], [3, 4]], dtype="int64")
        df2 = DataFrame(df.values, dtype=df[0].dtype)
        assert df2._mgr.blocks[0].values.flags.c_contiguous

    def test_1d_object_array_does_not_copy(self, using_infer_string):
        # [FUNCTION DOC]
        # Issue 39272: a one-dimensional object array of strings passed with copy
        # disabled shares memory with the frame (for the inferred-string option with
        # Arrow storage nothing is compared); with an explicit object dtype memory is
        # always shared.
        # https://github.com/pandas-dev/pandas/issues/39272
        arr = np.array(["a", "b"], dtype="object")
        df = DataFrame(arr, copy=False)
        if using_infer_string:
            if df[0].dtype.storage == "pyarrow":
                # object dtype strings are converted to arrow memory,
                # no numpy arrays to compare
                pass
            else:
                assert np.shares_memory(df[0].to_numpy(), arr)
        else:
            assert np.shares_memory(df.values, arr)

        df = DataFrame(arr, dtype=object, copy=False)
        assert np.shares_memory(df.values, arr)

    def test_2d_object_array_does_not_copy(self, using_infer_string):
        # [FUNCTION DOC]
        # Issue 39272: same as the one-dimensional case but for a two-dimensional object
        # array.
        # https://github.com/pandas-dev/pandas/issues/39272
        arr = np.array([["a", "b"], ["c", "d"]], dtype="object")
        df = DataFrame(arr, copy=False)
        if using_infer_string:
            if df[0].dtype.storage == "pyarrow":
                # object dtype strings are converted to arrow memory,
                # no numpy arrays to compare
                pass
            else:
                assert np.shares_memory(df[0].to_numpy(), arr)
        else:
            assert np.shares_memory(df.values, arr)

        df = DataFrame(arr, dtype=object, copy=False)
        assert np.shares_memory(df.values, arr)

    def test_constructor_dtype_list_data(self):
        # [FUNCTION DOC]
        # A nested list with an object dtype keeps None and the string unchanged.
        df = DataFrame([[1, "2"], [None, "a"]], dtype=object)
        assert df.loc[1, 0] is None
        assert df.loc[0, 1] == "2"

    def test_constructor_list_of_2d_raises(self):
        # [FUNCTION DOC]
        # Issue 32289: a list holding an empty frame or empty array raises a value error
        # mentioning a three-dimensional shape; a list of two single-column frames raises
        # one mentioning shape two by two by one.
        # https://github.com/pandas-dev/pandas/issues/32289
        a = DataFrame()
        b = np.empty((0, 0))
        with pytest.raises(ValueError, match=r"shape=\(1, 0, 0\)"):
            DataFrame([a])

        with pytest.raises(ValueError, match=r"shape=\(1, 0, 0\)"):
            DataFrame([b])

        a = DataFrame({"A": [1, 2]})
        with pytest.raises(ValueError, match=r"shape=\(2, 2, 1\)"):
            DataFrame([a, a])

    @pytest.mark.parametrize(
        "typ, ad",
        [
            # mixed floating and integer coexist in the same frame
            ["float", {}],
            # add lots of types
            ["float", {"A": 1, "B": "foo", "C": "bar"}],
            # GH 622
            ["int", {}],
        ],
    )
    def test_constructor_mixed_dtypes(self, typ, ad):
        # [FUNCTION DOC]
        # Parametrized over (float or int, extra column dictionary). Builds a dict of
        # arrays cast to each of the mixed float (or int) dtypes, optionally plus
        # extra string/number columns (GH 622), and checks that each such column in the
        # frame keeps its requested dtype.
        if typ == "int":
            dtypes = MIXED_INT_DTYPES
            arrays = [
                np.array(np.random.default_rng(2).random(10), dtype=d) for d in dtypes
            ]
        elif typ == "float":
            dtypes = MIXED_FLOAT_DTYPES
            arrays = [
                np.array(np.random.default_rng(2).integers(10, size=10), dtype=d)
                for d in dtypes
            ]

        for d, a in zip(dtypes, arrays):
            assert a.dtype == d
        ad.update(dict(zip(dtypes, arrays)))
        df = DataFrame(ad)

        dtypes = MIXED_FLOAT_DTYPES + MIXED_INT_DTYPES
        for d in dtypes:
            if d in df:
                assert df.dtypes[d] == d

    def test_constructor_complex_dtypes(self):
        # [FUNCTION DOC]
        # GH10952: single- and double-precision complex columns keep their dtypes.
        # GH10952
        a = np.random.default_rng(2).random(10).astype(np.complex64)
        b = np.random.default_rng(2).random(10).astype(np.complex128)

        df = DataFrame({"a": a, "b": b})
        assert a.dtype == df.a.dtype
        assert b.dtype == df.b.dtype

    def test_constructor_dtype_str_na_values(self, string_dtype):
        # [FUNCTION DOC]
        # Issue 21083: with each string dtype fixture, None stays None for the object
        # variant (isna flags it) while NaN stays NaN in a second frame.
        # https://github.com/pandas-dev/pandas/issues/21083
        df = DataFrame({"A": ["x", None]}, dtype=string_dtype)
        result = df.isna()
        expected = DataFrame({"A": [False, True]})
        tm.assert_frame_equal(result, expected)
        assert df.iloc[1, 0] is None

        df = DataFrame({"A": ["x", np.nan]}, dtype=string_dtype)
        assert np.isnan(df.iloc[1, 0])

    def test_constructor_rec(self, float_frame):
        # [FUNCTION DOC]
        # Constructing from a record array (with reversed field names) from the float
        # fixture uses field names as columns, accepts an index (which becomes the frame
        # index), and honors an explicit columns subset with a different index equal to
        # reindexing.
        rec = float_frame.to_records(index=False)
        rec.dtype.names = list(rec.dtype.names)[::-1]

        index = float_frame.index

        df = DataFrame(rec)
        tm.assert_index_equal(df.columns, Index(rec.dtype.names))

        df2 = DataFrame(rec, index=index)
        tm.assert_index_equal(df2.columns, Index(rec.dtype.names))
        tm.assert_index_equal(df2.index, index)

        # case with columns != the ones we would infer from the data
        rng = np.arange(len(rec))[::-1]
        df3 = DataFrame(rec, index=rng, columns=["C", "B"])
        expected = DataFrame(rec, index=rng).reindex(columns=["C", "B"])
        tm.assert_frame_equal(df3, expected)

    def test_constructor_bool(self):
        # [FUNCTION DOC]
        # A dictionary of two boolean arrays gives boolean values dtype.
        df = DataFrame({0: np.ones(10, dtype=bool), 1: np.zeros(10, dtype=bool)})
        assert df.values.dtype == np.bool_

    def test_constructor_overflow_int64(self):
        # [FUNCTION DOC]
        # gh-14881 / gh-2355: unsigned 64-bit values above the signed range keep unsigned
        # 64-bit dtype for a plain column and in a structured array with unsigned fields.
        # see gh-14881
        values = np.array([2**64 - i for i in range(1, 10)], dtype=np.uint64)

        result = DataFrame({"a": values})
        assert result["a"].dtype == np.uint64

        # see gh-2355
        data_scores = [
            (6311132704823138710, 273),
            (2685045978526272070, 23),
            (8921811264899370420, 45),
            (17019687244989530680, 270),
            (9930107427299601010, 273),
        ]
        dtype = [("uid", "u8"), ("score", "u8")]
        data = np.zeros((len(data_scores),), dtype=dtype)
        data[:] = data_scores
        df_crawls = DataFrame(data)
        assert df_crawls["uid"].dtype == np.uint64

    @pytest.mark.parametrize(
        "values",
        [
            np.array([2**64], dtype=object),
            np.array([2**65]),
            [2**64 + 1],
            np.array([-(2**63) - 4], dtype=object),
            np.array([-(2**64) - 1]),
            [-(2**65) - 2],
        ],
    )
    def test_constructor_int_overflow(self, values):
        # [FUNCTION DOC]
        # gh-18584: parametrized over values outside the 64-bit integer ranges (arrays and
        # lists). The column has object dtype and holds the exact value.
        # see gh-18584
        value = values[0]
        result = DataFrame(values)

        assert result[0].dtype == object
        assert result[0][0] == value

    @pytest.mark.parametrize(
        "values",
        [
            np.array([1], dtype=np.uint16),
            np.array([1], dtype=np.uint32),
            np.array([1], dtype=np.uint64),
            [np.uint16(1)],
            [np.uint32(1)],
            [np.uint64(1)],
        ],
    )
    def test_constructor_numpy_uints(self, values):
        # [FUNCTION DOC]
        # GH#47294: parametrized over NumPy unsigned integer arrays and lists of scalars;
        # the column dtype equals the dtype of the values.
        # GH#47294
        value = values[0]
        result = DataFrame(values)

        assert result[0].dtype == value.dtype
        assert result[0][0] == value

    def test_constructor_ordereddict(self):
        # [FUNCTION DOC]
        # A shuffled list of labels passed via an ordered dictionary yields columns in
        # the same order.
        nitems = 100
        nums = list(range(nitems))
        np.random.default_rng(2).shuffle(nums)
        expected = [f"A{i:d}" for i in nums]
        df = DataFrame(OrderedDict(zip(expected, [[0]] * nitems)))
        assert expected == list(df.columns)

    def test_constructor_dict(self):
        # [FUNCTION DOC]
        # Dictionary of Series with different indexes: the result index is the union with
        # missing padding; an explicit columns list selects (and adds missing columns
        # as all missing); an empty construction has length zero; mixing a nested
        # dictionary with a list raises a value error about ambiguous ordering.
        datetime_series = Series(
            np.arange(30, dtype=np.float64), index=date_range("2020-01-01", periods=30)
        )
        # test expects index shifted by 5
        datetime_series_short = datetime_series[5:]

        frame = DataFrame({"col1": datetime_series, "col2": datetime_series_short})

        # col2 is padded with NaN
        assert len(datetime_series) == 30
        assert len(datetime_series_short) == 25

        tm.assert_series_equal(frame["col1"], datetime_series.rename("col1"))

        exp = Series(
            np.concatenate([[np.nan] * 5, datetime_series_short.values]),
            index=datetime_series.index,
            name="col2",
        )
        tm.assert_series_equal(exp, frame["col2"])

        frame = DataFrame(
            {"col1": datetime_series, "col2": datetime_series_short},
            columns=["col2", "col3", "col4"],
        )

        assert len(frame) == len(datetime_series_short)
        assert "col1" not in frame
        assert isna(frame["col3"]).all()

        # Corner cases
        assert len(DataFrame()) == 0

        # mix dict and array, wrong size - no spec for which error should raise
        # first
        msg = "Mixing dicts with non-Series may lead to ambiguous ordering."
        with pytest.raises(ValueError, match=msg):
            DataFrame({"A": {"a": "a", "b": "b"}, "B": ["a", "b", "c"]})

    def test_constructor_dict_length1(self):
        # [FUNCTION DOC]
        # A single-column dictionary of a one-level dictionary uses its keys as the index.
        # Length-one dict micro-optimization
        frame = DataFrame({"A": {"1": 1, "2": 2}})
        tm.assert_index_equal(frame.index, Index(["1", "2"]))

    def test_constructor_dict_with_index(self):
        # [FUNCTION DOC]
        # An empty dictionary with an index object keeps that very index object.
        # empty dict plus index
        idx = Index([0, 1, 2])
        frame = DataFrame({}, index=idx)
        assert frame.index is idx

    def test_constructor_dict_with_index_and_columns(self):
        # [FUNCTION DOC]
        # An empty dictionary with index and columns keeps both objects and creates
        # no internal series.
        # empty dict with index and columns
        idx = Index([0, 1, 2])
        frame = DataFrame({}, index=idx, columns=idx)
        assert frame.index is idx
        assert frame.columns is idx
        assert len(frame._series) == 3

    def test_constructor_dict_of_empty_lists(self):
        # [FUNCTION DOC]
        # Dictionary of empty lists with given columns gives an empty range index.
        # with dict of empty list and Series
        frame = DataFrame({"A": [], "B": []}, columns=["A", "B"])
        tm.assert_index_equal(frame.index, RangeIndex(0), exact=True)

    def test_constructor_dict_with_none(self):
        # [FUNCTION DOC]
        # GH 14381: a dictionary whose value is None, with an index, gives a cell that is
        # None, equal to the list-of-None form.
        # GH 14381
        # Dict with None value
        frame_none = DataFrame({"a": None}, index=[0])
        frame_none_list = DataFrame({"a": [None]}, index=[0])
        assert frame_none._get_value(0, "a") is None
        assert frame_none_list._get_value(0, "a") is None
        tm.assert_frame_equal(frame_none, frame_none_list)

    def test_constructor_dict_errors(self):
        # [FUNCTION DOC]
        # GH10856: a dictionary of scalars raises a value error saying an index must be
        # passed, even when columns are given.
        # GH10856
        # dict with scalar values should raise error, even if columns passed
        msg = "If using all scalar values, you must pass an index"
        with pytest.raises(ValueError, match=msg):
            DataFrame({"a": 0.7})

        with pytest.raises(ValueError, match=msg):
            DataFrame({"a": 0.7}, columns=["a"])

    @pytest.mark.parametrize("scalar", [2, np.nan, None, "D"])
    def test_constructor_invalid_items_unused(self, scalar):
        # [FUNCTION DOC]
        # Parametrized over scalar values. A scalar in the dictionary that is not selected
        # by the columns argument causes no error; the result is an empty frame with the
        # requested column.
        # No error if invalid (scalar) value is in fact not used:
        result = DataFrame({"a": scalar}, columns=["b"])
        expected = DataFrame(columns=["b"])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("value", [4, np.nan, None, float("nan")])
    def test_constructor_dict_nan_key(self, value):
        # [FUNCTION DOC]
        # GH 18455: parametrized over null-like key values. A dictionary with a null as
        # column label and index label is built, and sorting works: the result equals the
        # expected integer frame whether built from the dictionary alone or with index and
        # columns passed.
        # GH 18455
        cols = [1, value, 3]
        idx = ["a", value]
        values = [[0, 3], [1, 4], [2, 5]]
        data = {cols[c]: Series(values[c], index=idx) for c in range(3)}
        result = DataFrame(data).sort_values(1).sort_values("a", axis=1)
        expected = DataFrame(
            np.arange(6, dtype="int64").reshape(2, 3), index=idx, columns=cols
        )
        tm.assert_frame_equal(result, expected)

        result = DataFrame(data, index=idx).sort_values("a", axis=1)
        tm.assert_frame_equal(result, expected)

        result = DataFrame(data, index=idx, columns=cols)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("value", [np.nan, None, float("nan")])
    def test_constructor_dict_nan_tuple_key(self, value):
        # [FUNCTION DOC]
        # GH 18455: same as above but with tuple labels containing a null element.
        # GH 18455
        cols = Index([(11, 21), (value, 22), (13, value)])
        idx = Index([("a", value), (value, 2)])
        values = [[0, 3], [1, 4], [2, 5]]
        data = {cols[c]: Series(values[c], index=idx) for c in range(3)}
        result = DataFrame(data).sort_values((11, 21)).sort_values(("a", value), axis=1)
        expected = DataFrame(
            np.arange(6, dtype="int64").reshape(2, 3), index=idx, columns=cols
        )
        tm.assert_frame_equal(result, expected)

        result = DataFrame(data, index=idx).sort_values(("a", value), axis=1)
        tm.assert_frame_equal(result, expected)

        result = DataFrame(data, index=idx, columns=cols)
        tm.assert_frame_equal(result, expected)

    def test_constructor_dict_order_insertion(self):
        # [FUNCTION DOC]
        # GH19018: columns follow dictionary insertion order (not sorted) when the data is
        # a dictionary of Series.
        datetime_series = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        datetime_series_short = datetime_series[:5]

        # GH19018
        # initialization ordering: by insertion order if python>= 3.6
        d = {"b": datetime_series_short, "a": datetime_series}
        frame = DataFrame(data=d)
        expected = DataFrame(data=d, columns=list("ba"))
        tm.assert_frame_equal(frame, expected)

    def test_constructor_dict_nan_key_and_columns(self):
        # [FUNCTION DOC]
        # GH 16894: a dictionary with a NaN key and given columns includes the NaN column
        # in position.
        # GH 16894
        result = DataFrame({np.nan: [1, 2], 2: [2, 3]}, columns=[np.nan, 2])
        expected = DataFrame([[1, 2], [2, 3]], columns=[np.nan, 2])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("missing_value", [None, np.nan, pd.NA])
    def test_constructor_list_of_dict_with_str_na_key(
        self, missing_value, using_infer_string
    ):
        # [FUNCTION DOC]
        # Issue 63889: parametrized over None, NaN, NA as a dictionary key in a list of
        # dictionaries. Values are preserved when the key becomes a NaN column label (with
        # the inferred-string option the label is NaN, else it stays the given value).
        # https://github.com/pandas-dev/pandas/issues/63889
        # preserve values when None key is converted to NaN column name
        dict_data = [
            {"colA": 1, missing_value: 2},
            {"colA": 3, missing_value: 4},
        ]
        result = DataFrame(dict_data)
        expected = DataFrame(
            [[1, 2], [3, 4]],
            columns=["colA", np.nan if using_infer_string else missing_value],
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("missing_value", [None, np.nan, pd.NA])
    def test_constructor_dict_of_dict_with_str_na_key(
        self, missing_value, using_infer_string
    ):
        # [FUNCTION DOC]
        # Issue 63889: same as above, but with a null as the inner key of a dictionary of
        # dictionaries (becoming a row label).
        # https://github.com/pandas-dev/pandas/issues/63889
        dict_data = {"col": {"row1": 1, missing_value: 2, "row3": 3}}
        result = DataFrame(dict_data)
        expected = DataFrame(
            {"col": [1, 2, 3]},
            index=Index(
                ["row1", np.nan if using_infer_string else missing_value, "row3"]
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_constructor_multi_index(self):
        # [FUNCTION DOC]
        # GH 4078: an all-missing frame built with a multi-level index and columns (also
        # with a duplicate and reordered tuples) has only missing values.
        # GH 4078
        # construction error with mi and all-nan frame
        tuples = [(2, 3), (3, 3), (3, 3)]
        mi = MultiIndex.from_tuples(tuples)
        df = DataFrame(index=mi, columns=mi)
        assert isna(df).values.ravel().all()

        tuples = [(3, 3), (2, 3), (3, 3)]
        mi = MultiIndex.from_tuples(tuples)
        df = DataFrame(index=mi, columns=mi)
        assert isna(df).values.ravel().all()

    def test_constructor_2d_index(self):
        # [FUNCTION DOC]
        # GH 25416: a two-dimensional columns list builds a multi-level column index;
        # likewise a two-dimensional index builds a multi-level row index.
        # GH 25416
        # handling of 2d index in construction
        df = DataFrame([[1]], columns=[[1]], index=[1, 2])
        expected = DataFrame(
            [1, 1],
            index=Index([1, 2], dtype="int64"),
            columns=MultiIndex(levels=[[1]], codes=[[0]]),
        )
        tm.assert_frame_equal(df, expected)

        df = DataFrame([[1]], columns=[[1]], index=[[1, 2]])
        expected = DataFrame(
            [1, 1],
            index=MultiIndex(levels=[[1, 2]], codes=[[0, 1]]),
            columns=MultiIndex(levels=[[1]], codes=[[0]]),
        )
        tm.assert_frame_equal(df, expected)

    def test_constructor_error_msgs(self):
        # [FUNCTION DOC]
        # Many shape mismatch cases raise value errors with specific messages: empty data
        # with an index; mixing dicts with non-Series; arrays whose shape disagrees with
        # the given index/columns (GH 3105); 3-D input; column count mismatch for
        # tuples of ranges (gh-26429); dictionary of scalars needing an index.
        msg = "Empty data passed with indices specified."
        # passing an empty array with columns specified.
        with pytest.raises(ValueError, match=msg):
            DataFrame(np.empty(0), index=[1])

        msg = "Mixing dicts with non-Series may lead to ambiguous ordering."
        # mix dict and array, wrong size
        with pytest.raises(ValueError, match=msg):
            DataFrame({"A": {"a": "a", "b": "b"}, "B": ["a", "b", "c"]})

        # wrong size ndarray, GH 3105
        msg = r"Shape of passed values is \(4, 3\), indices imply \(3, 3\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(
                np.arange(12).reshape((4, 3)),
                columns=["foo", "bar", "baz"],
                index=date_range("2000-01-01", periods=3),
            )

        arr = np.array([[4, 5, 6]])
        msg = r"Shape of passed values is \(1, 3\), indices imply \(1, 4\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(index=[0], columns=range(4), data=arr)

        arr = np.array([4, 5, 6])
        msg = r"Shape of passed values is \(3, 1\), indices imply \(1, 4\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(index=[0], columns=range(4), data=arr)

        # higher dim raise exception
        with pytest.raises(ValueError, match="Must pass 2-d input"):
            DataFrame(np.zeros((3, 3, 3)), columns=["A", "B", "C"], index=[1])

        # wrong size axis labels
        msg = r"Shape of passed values is \(2, 3\), indices imply \(1, 3\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(
                np.random.default_rng(2).random((2, 3)),
                columns=["A", "B", "C"],
                index=[1],
            )

        msg = r"Shape of passed values is \(2, 3\), indices imply \(2, 2\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(
                np.random.default_rng(2).random((2, 3)),
                columns=["A", "B"],
                index=[1, 2],
            )

        # gh-26429
        msg = "2 columns passed, passed data had 10 columns"
        with pytest.raises(ValueError, match=msg):
            DataFrame((range(10), range(10, 20)), columns=("ones", "twos"))

        msg = "If using all scalar values, you must pass an index"
        with pytest.raises(ValueError, match=msg):
            DataFrame({"a": False, "b": True})

    def test_constructor_subclass_dict(self, dict_subclass):
        # [FUNCTION DOC]
        # Parametrized over dict subclass fixtures. A dictionary of subclassed
        # dictionaries, and a subclassed dictionary of subclassed dictionaries, give frames
        # equal to the plain dictionary form.
        # Test for passing dict subclass to constructor
        data = {
            "col1": dict_subclass((x, 10.0 * x) for x in range(10)),
            "col2": dict_subclass((x, 20.0 * x) for x in range(10)),
        }
        df = DataFrame(data)
        refdf = DataFrame({col: dict(val.items()) for col, val in data.items()})
        tm.assert_frame_equal(refdf, df)

        data = dict_subclass(data.items())
        df = DataFrame(data)
        tm.assert_frame_equal(refdf, df)

    def test_constructor_defaultdict(self, float_frame):
        # [FUNCTION DOC]
        # A dictionary of defaultdicts built from the float fixture (with some values set to
        # missing) reconstructs the frame (after reindexing to the fixture index).
        # try with defaultdict
        data = {}
        float_frame.loc[: float_frame.index[10], "B"] = np.nan

        for k, v in float_frame.items():
            dct = defaultdict(dict)
            dct.update(v.to_dict())
            data[k] = dct
        frame = DataFrame(data)
        expected = frame.reindex(index=float_frame.index)
        tm.assert_frame_equal(float_frame, expected)

    def test_constructor_dict_block(self):
        # [FUNCTION DOC]
        # A dictionary with columns in reverse alphabetical order and given columns yields
        # values in the given column order.
        expected = np.array([[4.0, 3.0, 2.0, 1.0]])
        df = DataFrame(
            {"d": [4.0], "c": [3.0], "b": [2.0], "a": [1.0]},
            columns=["d", "c", "b", "a"],
        )
        tm.assert_numpy_array_equal(df.values, expected)

    def test_constructor_dict_cast(self, using_infer_string):
        # [FUNCTION DOC]
        # Dictionary of dictionaries with a float dtype casts string numbers to float
        # (three rows, both columns float64); without dtype the integer column becomes float
        # (missing padding) and the string column is object (or string when inferred).
        # KNOWN ISSUE: the object-dtype assertion line is written as a conditional expression over the whole comparison, so with the inferred-string option the assertion is on a non-empty literal string (always true) and checks nothing.
        # cast float tests
        test_data = {"A": {"1": 1, "2": 2}, "B": {"1": "1", "2": "2", "3": "3"}}
        frame = DataFrame(test_data, dtype=float)
        assert len(frame) == 3
        assert frame["B"].dtype == np.float64
        assert frame["A"].dtype == np.float64

        frame = DataFrame(test_data)
        assert len(frame) == 3
        assert frame["B"].dtype == np.object_ if not using_infer_string else "str"
        assert frame["A"].dtype == np.float64

    def test_constructor_dict_cast2(self):
        # [FUNCTION DOC]
        # A dictionary containing words cannot be cast to float: a value error about
        # converting a string is raised.
        # can't cast to float
        test_data = {
            "A": dict(zip(range(20), [f"word_{i}" for i in range(20)])),
            "B": dict(zip(range(15), np.random.default_rng(2).standard_normal(15))),
        }
        with pytest.raises(ValueError, match="could not convert string"):
            DataFrame(test_data, dtype=float)

    def test_constructor_dict_dont_upcast(self):
        # [FUNCTION DOC]
        # A string and a missing value in one column stay unconverted: the missing entry
        # is a float NaN.
        d = {"Col1": {"Row1": "A String", "Row2": np.nan}}
        df = DataFrame(d)
        assert isinstance(df["Col1"]["Row2"], float)

    def test_constructor_dict_dont_upcast2(self):
        # [FUNCTION DOC]
        # A mixed list-of-lists frame keeps integer scalars as built-in integers.
        dm = DataFrame([[1, 2], ["a", "b"]], index=[1, 2], columns=[1, 2])
        assert isinstance(dm[1][1], int)

    def test_constructor_dict_of_tuples(self):
        # [FUNCTION DOC]
        # GH #1491: a dictionary of tuples equals a dictionary of lists (dtypes not
        # checked).
        # GH #1491
        data = {"a": (1, 2, 3), "b": (4, 5, 6)}

        result = DataFrame(data)
        expected = DataFrame({k: list(v) for k, v in data.items()})
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_constructor_dict_of_ranges(self):
        # [FUNCTION DOC]
        # GH 26356: a dictionary of ranges is expanded into integer columns.
        # GH 26356
        data = {"a": range(3), "b": range(3, 6)}

        result = DataFrame(data)
        expected = DataFrame({"a": [0, 1, 2], "b": [3, 4, 5]})
        tm.assert_frame_equal(result, expected)

    def test_constructor_dict_of_iterators(self):
        # [FUNCTION DOC]
        # GH 26349: a dictionary of an iterator and a reversed range is expanded.
        # GH 26349
        data = {"a": iter(range(3)), "b": reversed(range(3))}

        result = DataFrame(data)
        expected = DataFrame({"a": [0, 1, 2], "b": [2, 1, 0]})
        tm.assert_frame_equal(result, expected)

    def test_constructor_dict_of_generators(self):
        # [FUNCTION DOC]
        # GH 26349: a dictionary of generators is expanded.
        # GH 26349
        data = {"a": (i for i in (range(3))), "b": (i for i in reversed(range(3)))}
        result = DataFrame(data)
        expected = DataFrame({"a": [0, 1, 2], "b": [2, 1, 0]})
        tm.assert_frame_equal(result, expected)

    def test_constructor_dict_multiindex(self):
        # [FUNCTION DOC]
        # A dictionary keyed by tuples with dictionary values keyed by tuples builds
        # multi-level columns and rows (equal to sorted construction); adding a
        # non-tuple key makes a flat index of tuples and the result is compared after
        # reindexing.
        d = {
            ("a", "a"): {("i", "i"): 0, ("i", "j"): 1, ("j", "i"): 2},
            ("b", "a"): {("i", "i"): 6, ("i", "j"): 5, ("j", "i"): 4},
            ("b", "c"): {("i", "i"): 7, ("i", "j"): 8, ("j", "i"): 9},
        }
        _d = sorted(d.items())
        df = DataFrame(d)
        expected = DataFrame(
            [x[1] for x in _d], index=MultiIndex.from_tuples([x[0] for x in _d])
        ).T
        expected.index = MultiIndex.from_tuples(expected.index)
        tm.assert_frame_equal(
            df,
            expected,
        )

        d["z"] = {"y": 123.0, ("i", "i"): 111, ("i", "j"): 111, ("j", "i"): 111}
        _d.insert(0, ("z", d["z"]))
        expected = DataFrame(
            [x[1] for x in _d], index=Index([x[0] for x in _d], tupleize_cols=False)
        ).T
        expected.index = Index(expected.index, tupleize_cols=False)
        df = DataFrame(d)
        df = df.reindex(columns=expected.columns, index=expected.index)
        tm.assert_frame_equal(df, expected)

    def test_constructor_dict_datetime64_index(self):
        # [FUNCTION DOC]
        # GH 10160: a dictionary of dictionaries keyed by NumPy datetime values, native
        # datetimes or timestamps builds a datetime index; the NumPy-datetime case has
        # second resolution, native datetimes microsecond; all equal the expected
        # (missing filled) frame.
        # GH 10160
        dates_as_str = ["1984-02-19", "1988-11-06", "1989-12-03", "1990-03-15"]

        def create_data(constructor):
            # [FUNCTION DOC]
            # HELPER: for the datetime index test. Builds a dictionary keyed by position
            # whose values are single-entry dictionaries from a converted date string to
            # twice the position, using the given conversion callable.
            return {i: {constructor(s): 2 * i} for i, s in enumerate(dates_as_str)}

        data_datetime64 = create_data(np.datetime64)
        data_datetime = create_data(lambda x: datetime.strptime(x, "%Y-%m-%d"))
        data_Timestamp = create_data(Timestamp)

        expected = DataFrame(
            [
                [0, None, None, None],
                [None, 2, None, None],
                [None, None, 4, None],
                [None, None, None, 6],
            ],
            index=[Timestamp(dt) for dt in dates_as_str],
        )

        result_datetime64 = DataFrame(data_datetime64)
        assert result_datetime64.index.unit == "s"
        result_datetime64.index = result_datetime64.index.as_unit("us")
        result_datetime = DataFrame(data_datetime)
        assert result_datetime.index.unit == "us"
        result_Timestamp = DataFrame(data_Timestamp)
        tm.assert_frame_equal(result_datetime64, expected)
        tm.assert_frame_equal(result_datetime, expected)
        tm.assert_frame_equal(result_Timestamp, expected)

    @pytest.mark.parametrize(
        "klass,exp_dtype",
        [
            (lambda x: np.timedelta64(x, "D"), "m8[s]"),
            (lambda x: timedelta(days=x), "m8[us]"),
            (lambda x: Timedelta(x, "D"), "m8[s]"),
            (lambda x: Timedelta(x, "D").as_unit("ms"), "m8[ms]"),
        ],
    )
    def test_constructor_dict_timedelta64_index(self, klass, exp_dtype):
        # [FUNCTION DOC]
        # GH 10160: parametrized over four ways of building timedelta keys (NumPy,
        # native, Timedelta, millisecond Timedelta) and the expected index dtype. A
        # dictionary of dictionaries keyed by timedeltas builds a timedelta index with the
        # expected unit.
        # GH 10160
        td_as_int = [1, 2, 3, 4]

        data = {i: {klass(s): 2 * i} for i, s in enumerate(td_as_int)}

        expected = DataFrame(
            [
                {0: 0, 1: None, 2: None, 3: None},
                {0: None, 1: 2, 2: None, 3: None},
                {0: None, 1: None, 2: 4, 3: None},
                {0: None, 1: None, 2: None, 3: 6},
            ],
            index=[Timedelta(td, "D") for td in td_as_int],
        )
        expected.index = expected.index.astype(exp_dtype)

        result = DataFrame(data)

        tm.assert_frame_equal(result, expected)

    def test_constructor_period_dict(self):
        # [FUNCTION DOC]
        # Period indexes with different frequencies, and lists of Period objects, in a
        # dictionary keep their period dtypes.
        # PeriodIndex
        a = pd.PeriodIndex(["2012-01", "NaT", "2012-04"], freq="M")
        b = pd.PeriodIndex(["2012-02-01", "2012-03-01", "NaT"], freq="D")
        df = DataFrame({"a": a, "b": b})
        assert df["a"].dtype == a.dtype
        assert df["b"].dtype == b.dtype

        # list of periods
        df = DataFrame({"a": a.astype(object).tolist(), "b": b.astype(object).tolist()})
        assert df["a"].dtype == a.dtype
        assert df["b"].dtype == b.dtype

    def test_constructor_dict_extension_scalar(self, ea_scalar_and_dtype):
        # [FUNCTION DOC]
        # Parametrized over scalars of extension dtype fixture. A dictionary with one
        # extension scalar and an index of length one gives that dtype and equals the
        # scalar-filled frame.
        ea_scalar, ea_dtype = ea_scalar_and_dtype
        df = DataFrame({"a": ea_scalar}, index=[0])
        assert df["a"].dtype == ea_dtype

        expected = DataFrame(index=[0], columns=["a"], data=ea_scalar)

        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "data,dtype",
        [
            (Period("2020-01"), PeriodDtype("M")),
            (Interval(left=0, right=5), IntervalDtype("int64", "right")),
            (
                Timestamp("2011-01-01", tz="US/Eastern").as_unit("s"),
                DatetimeTZDtype(unit="s", tz="US/Eastern"),
            ),
        ],
    )
    def test_constructor_extension_scalar_data(self, data, dtype):
        # [FUNCTION DOC]
        # GH 34832: parametrized over a period, an interval and a timezone-aware
        # timestamp. Passing the scalar as data with index and columns broadcasts it with
        # the corresponding extension dtype to all columns.
        # GH 34832
        df = DataFrame(index=range(2), columns=["a", "b"], data=data)

        assert df["a"].dtype == dtype
        assert df["b"].dtype == dtype

        arr = pd.array([data] * 2, dtype=dtype)
        expected = DataFrame({"a": arr, "b": arr})

        tm.assert_frame_equal(df, expected)

    def test_nested_dict_frame_constructor(self):
        # [FUNCTION DOC]
        # GH nested dictionary constructor: a dictionary of dictionaries built by iterating
        # over a frame with a period-indexed columns (outer keys columns, inner keys
        # rows, and the transposed case with explicit index/columns) round trips to the
        # original frame.
        rng = pd.period_range("1/1/2000", periods=5)
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)), columns=rng)

        data = {}
        for col in df.columns:
            for row in df.index:
                data.setdefault(col, {})[row] = df._get_value(row, col)

        result = DataFrame(data, columns=rng)
        tm.assert_frame_equal(result, df)

        data = {}
        for col in df.columns:
            for row in df.index:
                data.setdefault(row, {})[col] = df._get_value(row, col)

        result = DataFrame(data, index=rng).T
        tm.assert_frame_equal(result, df)

    def _check_basic_constructor(self, empty):
        # [FUNCTION DOC]
        # HELPER: shared body for the ndarray and masked-array tests. The empty argument is
        # an array factory (ones or masked-all). Checks: 2-D and 1-D inputs give right
        # shapes; for non-ones input, casting to integer raises an integer-casting-NaN
        # error and the helper returns early; for ones input integer casting works;
        # mismatched labels raise value errors with shape messages; 3-D input raises "must
        # pass 2-d"; automatic labels are range indexes; zero-length axes are allowed.
        # KNOWN ISSUE: the comment about the matrix shape says three by two while the matrix actually built is two by three; and the early return means the checks after the dtype branch run only for the ones factory.
        # mat: 2d matrix with shape (3, 2) to input. empty - makes sized
        # objects
        mat = empty((2, 3), dtype=float)
        # 2-D input
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2])

        assert len(frame.index) == 2
        assert len(frame.columns) == 3

        # 1-D input
        frame = DataFrame(empty((3,)), columns=["A"], index=[1, 2, 3])
        assert len(frame.index) == 3
        assert len(frame.columns) == 1

        if empty is not np.ones:
            msg = r"Cannot convert non-finite values \(NA or inf\) to integer"
            with pytest.raises(IntCastingNaNError, match=msg):
                DataFrame(mat, columns=["A", "B", "C"], index=[1, 2], dtype=np.int64)
            return
        else:
            frame = DataFrame(
                mat, columns=["A", "B", "C"], index=[1, 2], dtype=np.int64
            )
            assert frame.values.dtype == np.int64

        # wrong size axis labels
        msg = r"Shape of passed values is \(2, 3\), indices imply \(1, 3\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(mat, columns=["A", "B", "C"], index=[1])
        msg = r"Shape of passed values is \(2, 3\), indices imply \(2, 2\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(mat, columns=["A", "B"], index=[1, 2])

        # higher dim raise exception
        with pytest.raises(ValueError, match="Must pass 2-d input"):
            DataFrame(empty((3, 3, 3)), columns=["A", "B", "C"], index=[1])

        # automatic labeling
        frame = DataFrame(mat)
        tm.assert_index_equal(frame.index, Index(range(2)), exact=True)
        tm.assert_index_equal(frame.columns, Index(range(3)), exact=True)

        frame = DataFrame(mat, index=[1, 2])
        tm.assert_index_equal(frame.columns, Index(range(3)), exact=True)

        frame = DataFrame(mat, columns=["A", "B", "C"])
        tm.assert_index_equal(frame.index, Index(range(2)), exact=True)

        # 0-length axis
        frame = DataFrame(empty((0, 3)))
        assert len(frame.index) == 0

        frame = DataFrame(empty((3, 0)))
        assert len(frame.columns) == 0

    def test_constructor_ndarray(self):
        # [FUNCTION DOC]
        # Runs the shared constructor check with ones, then a one-dimensional list of
        # strings with an index and a column has two rows.
        self._check_basic_constructor(np.ones)

        frame = DataFrame(["foo", "bar"], index=[0, 1], columns=["A"])
        assert len(frame) == 2

    def test_constructor_maskedarray(self):
        # [FUNCTION DOC]
        # Runs the shared check with a fully masked array, checks non-masked values show
        # through when set, and that a fully masked frame compares unequal to itself
        # everywhere (masked become NaN).
        # KNOWN ISSUE: a comment in the test itself questions what the last assertion on all-masked data is even verifying.
        self._check_basic_constructor(ma.masked_all)

        # Check non-masked values
        mat = ma.masked_all((2, 3), dtype=float)
        mat[0, 0] = 1.0
        mat[1, 2] = 2.0
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2])
        assert 1.0 == frame["A"][1]
        assert 2.0 == frame["C"][2]

        # what is this even checking??
        mat = ma.masked_all((2, 3), dtype=float)
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2])
        assert np.all(~np.asarray(frame == frame))

    @pytest.mark.filterwarnings(
        "ignore:elementwise comparison failed:DeprecationWarning"
    )
    def test_constructor_maskedarray_nonfloat(self):
        # [FUNCTION DOC]
        # Filters a comparison deprecation warning. Masked integer arrays promote to
        # float with missing values (all-masked frame compares unequal to itself), may be
        # cast to float64, and keep unmasked values. Masked datetime arrays stay datetime
        # with NaT, cannot be cast to int64 (type error about datetime values with that
        # dtype), and keep unmasked values. Masked boolean arrays promote to object with
        # missing, can be cast to object, and keep unmasked booleans as built-in booleans.
        # masked int promoted to float
        mat = ma.masked_all((2, 3), dtype=int)
        # 2-D input
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2])

        assert len(frame.index) == 2
        assert len(frame.columns) == 3
        assert np.all(~np.asarray(frame == frame))

        # cast type
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2], dtype=np.float64)
        assert frame.values.dtype == np.float64

        # Check non-masked values
        mat2 = ma.copy(mat)
        mat2[0, 0] = 1
        mat2[1, 2] = 2
        frame = DataFrame(mat2, columns=["A", "B", "C"], index=[1, 2])
        assert 1 == frame["A"][1]
        assert 2 == frame["C"][2]

        # masked np.datetime64 stays (use NaT as null)
        mat = ma.masked_all((2, 3), dtype="M8[ns]")
        # 2-D input
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2])

        assert len(frame.index) == 2
        assert len(frame.columns) == 3
        assert isna(frame).values.all()

        # cast type
        msg = r"datetime64\[ns\] values and dtype=int64 is not supported"
        with pytest.raises(TypeError, match=msg):
            DataFrame(mat, columns=["A", "B", "C"], index=[1, 2], dtype=np.int64)

        # Check non-masked values
        mat2 = ma.copy(mat)
        mat2[0, 0] = 1
        mat2[1, 2] = 2
        frame = DataFrame(mat2, columns=["A", "B", "C"], index=[1, 2])
        assert 1 == frame["A"].astype("i8")[1]
        assert 2 == frame["C"].astype("i8")[2]

        # masked bool promoted to object
        mat = ma.masked_all((2, 3), dtype=bool)
        # 2-D input
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2])

        assert len(frame.index) == 2
        assert len(frame.columns) == 3
        assert np.all(~np.asarray(frame == frame))

        # cast type
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2], dtype=object)
        assert frame.values.dtype == object

        # Check non-masked values
        mat2 = ma.copy(mat)
        mat2[0, 0] = True
        mat2[1, 2] = False
        frame = DataFrame(mat2, columns=["A", "B", "C"], index=[1, 2])
        assert frame["A"][1] is True
        assert frame["C"][2] is False

    def test_constructor_maskedarray_hardened(self):
        # [FUNCTION DOC]
        # GH24574: a hard-masked all-masked float array gives an all-NaN float frame; a
        # hard mask with nothing masked gives the data unchanged.
        # Check numpy masked arrays with hard masks -- from GH24574
        mat_hard = ma.masked_all((2, 2), dtype=float).harden_mask()
        result = DataFrame(mat_hard, columns=["A", "B"], index=[1, 2])
        expected = DataFrame(
            {"A": [np.nan, np.nan], "B": [np.nan, np.nan]},
            columns=["A", "B"],
            index=[1, 2],
            dtype=float,
        )
        tm.assert_frame_equal(result, expected)
        # Check case where mask is hard but no data are masked
        mat_hard = ma.ones((2, 2), dtype=float).harden_mask()
        result = DataFrame(mat_hard, columns=["A", "B"], index=[1, 2])
        expected = DataFrame(
            {"A": [1.0, 1.0], "B": [1.0, 1.0]},
            columns=["A", "B"],
            index=[1, 2],
            dtype=float,
        )
        tm.assert_frame_equal(result, expected)

    def test_constructor_maskedrecarray_dtype(self):
        # [FUNCTION DOC]
        # Constructing from a masked record array with an integer dtype raises a type
        # error telling the caller to pass a mapping of names to data (support for masked
        # records is deprecated, GH#40363).
        # Ensure constructor honors dtype
        data = np.ma.array(
            np.ma.zeros(5, dtype=[("date", "<f8"), ("price", "<f8")]), mask=[False] * 5
        )
        data = data.view(mrecords.mrecarray)
        with pytest.raises(TypeError, match=r"Pass \{name: data\[name\]"):
            # Support for MaskedRecords deprecated GH#40363
            DataFrame(data, dtype=int)

    def test_constructor_corner_shape(self):
        # [FUNCTION DOC]
        # A frame with an empty index list has a (0, 0) values shape.
        df = DataFrame(index=[])
        assert df.values.shape == (0, 0)

    @pytest.mark.parametrize(
        "data, index, columns, dtype, expected",
        [
            (None, list(range(10)), ["a", "b"], object, np.object_),
            (None, None, ["a", "b"], "int64", np.dtype("int64")),
            (None, list(range(10)), ["a", "b"], int, np.dtype("float64")),
            ({}, None, ["foo", "bar"], None, np.object_),
            ({"b": 1}, list(range(10)), list("abc"), int, np.dtype("float64")),
        ],
    )
    def test_constructor_dtype(self, data, index, columns, dtype, expected):
        # [FUNCTION DOC]
        # Parametrized over (data, index, columns, dtype, expected). The values dtype is
        # the expected one for: no data with an index and object dtype; no data with int64
        # and no index; no data with int dtype giving float; empty dictionary giving
        # object; and a dictionary with an int dtype requested for a missing column.
        df = DataFrame(data, index, columns, dtype)
        assert df.values.dtype == expected

    @pytest.mark.parametrize(
        "data,input_dtype,expected_dtype",
        (
            ([True, False, None], "boolean", pd.BooleanDtype),
            ([1.0, 2.0, None], "Float64", pd.Float64Dtype),
            ([1, 2, None], "Int64", pd.Int64Dtype),
            (["a", "b", "c"], "string", pd.StringDtype),
        ),
    )
    def test_constructor_dtype_nullable_extension_arrays(
        self, data, input_dtype, expected_dtype
    ):
        # [FUNCTION DOC]
        # Parametrized over data, nullable dtype name and the expected dtype class (boolean,
        # Float64, Int64, string). A dictionary column with the nullable dtype has the
        # expected dtype instance.
        df = DataFrame({"a": data}, dtype=input_dtype)
        assert df["a"].dtype == expected_dtype()

    def test_constructor_scalar_inference(self, using_infer_string):
        # [FUNCTION DOC]
        # Scalars in a dictionary with an index infer 64-bit integer, boolean, 64-bit
        # float, complex128, and object (string dtype when inference is enabled).
        # KNOWN ISSUE: the final assertion has a conditional expression around the entire comparison, so with the inferred-string option it asserts a non-empty literal (always true).
        data = {"int": 1, "bool": True, "float": 3.0, "complex": 4j, "object": "foo"}
        df = DataFrame(data, index=np.arange(10))

        assert df["int"].dtype == np.int64
        assert df["bool"].dtype == np.bool_
        assert df["float"].dtype == np.float64
        assert df["complex"].dtype == np.complex128
        assert df["object"].dtype == np.object_ if not using_infer_string else "str"

    def test_constructor_arrays_and_scalars(self):
        # [FUNCTION DOC]
        # A dictionary mixing an array and a scalar broadcasts the scalar; a dictionary of
        # only scalars raises a value error saying an index must be passed.
        df = DataFrame({"a": np.random.default_rng(2).standard_normal(10), "b": True})
        exp = DataFrame({"a": df["a"].values, "b": [True] * 10})

        tm.assert_frame_equal(df, exp)
        with pytest.raises(ValueError, match="must pass an index"):
            DataFrame({"a": False, "b": True})

    def test_constructor_DataFrame(self, float_frame):
        # [FUNCTION DOC]
        # Constructing from a frame gives an equal frame; an integer dtype cast gives
        # 64-bit integer values.
        df = DataFrame(float_frame)
        tm.assert_frame_equal(df, float_frame)

        df_casted = DataFrame(float_frame, dtype=np.int64)
        assert df_casted.values.dtype == np.int64

    def test_constructor_empty_dataframe(self):
        # [FUNCTION DOC]
        # GH 20624: constructing from an empty frame with object dtype equals an empty
        # frame of object dtype.
        # GH 20624
        actual = DataFrame(DataFrame(), dtype="object")
        expected = DataFrame([], dtype="object")
        tm.assert_frame_equal(actual, expected)

    def test_constructor_more(self, float_frame):
        # [FUNCTION DOC]
        # Used to be in another test module. Checks one-dimensional arrays become
        # one-column 2-D frames (also empty arrays), frames with only columns or only an
        # index have the right value shapes, strings cannot cast to float (value error),
        # a frame of a frame's series equals the fixture, and an int column plus float
        # column give float values.
        # KNOWN ISSUE: the assertion that the value array has two dimensions is duplicated for the empty-array case.
        # used to be in test_matrix.py
        arr = np.random.default_rng(2).standard_normal(10)
        dm = DataFrame(arr, columns=["A"], index=np.arange(10))
        assert dm.values.ndim == 2

        arr = np.random.default_rng(2).standard_normal(0)
        dm = DataFrame(arr)
        assert dm.values.ndim == 2
        assert dm.values.ndim == 2

        # no data specified
        dm = DataFrame(columns=["A", "B"], index=np.arange(10))
        assert dm.values.shape == (10, 2)

        dm = DataFrame(columns=["A", "B"])
        assert dm.values.shape == (0, 2)

        dm = DataFrame(index=np.arange(10))
        assert dm.values.shape == (10, 0)

        # can't cast
        mat = np.array(["foo", "bar"], dtype=object).reshape(2, 1)
        msg = "could not convert string to float: 'foo'"
        with pytest.raises(ValueError, match=msg):
            DataFrame(mat, index=[0, 1], columns=[0], dtype=float)

        dm = DataFrame(DataFrame(float_frame._series))
        tm.assert_frame_equal(dm, float_frame)

        # int cast
        dm = DataFrame(
            {"A": np.ones(10, dtype=int), "B": np.ones(10, dtype=np.float64)},
            index=np.arange(10),
        )

        assert len(dm.columns) == 2
        assert dm.values.dtype == np.float64

    def test_constructor_empty_list(self):
        # [FUNCTION DOC]
        # Empty lists with empty index, or with columns (GH 9939), and an empty
        # generator with columns, give frames equal to the equivalent empty dictionary
        # construction.
        df = DataFrame([], index=[])
        expected = DataFrame(index=[])
        tm.assert_frame_equal(df, expected)

        # GH 9939
        df = DataFrame([], columns=["A", "B"])
        expected = DataFrame({}, columns=["A", "B"])
        tm.assert_frame_equal(df, expected)

        # Empty generator: list(empty_gen()) == []
        def empty_gen():
            # [FUNCTION DOC]
            # HELPER: for the empty list test. A generator function that yields nothing.
            yield from ()

        df = DataFrame(empty_gen(), columns=["A", "B"])
        tm.assert_frame_equal(df, expected)

    def test_constructor_list_of_lists(self, using_infer_string):
        # [FUNCTION DOC]
        # GH #484: a list of lists gives an integer column and an object/string column;
        # GH 4851: a list of zero-dimensional arrays equals an integer range frame.
        # KNOWN ISSUE: the string-column assertion has a conditional expression around the entire comparison, so with the inferred-string option it asserts a non-empty literal (always true).
        # GH #484
        df = DataFrame(data=[[1, "a"], [2, "b"]], columns=["num", "str"])
        assert is_integer_dtype(df["num"])
        assert df["str"].dtype == np.object_ if not using_infer_string else "str"

        # GH 4851
        # list of 0-dim ndarrays
        expected = DataFrame(np.arange(10))
        data = [np.array(x) for x in range(10)]
        result = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_nested_pandasarray_matches_nested_ndarray(self):
        # [FUNCTION DOC]
        # GH#43986: a NumPy object array holding Series elements gives the same frame as
        # a pandas array of the same, with shape two by one and the same object values in
        # the column.
        # GH#43986
        ser = Series([1, 2])

        arr = np.array([None, None], dtype=object)
        arr[0] = ser
        arr[1] = ser * 2

        df = DataFrame(arr)
        expected = DataFrame(pd.array(arr))
        tm.assert_frame_equal(df, expected)
        assert df.shape == (2, 1)
        tm.assert_numpy_array_equal(df[0].values, arr)

    def test_constructor_list_like_data_nested_list_column(self):
        # [FUNCTION DOC]
        # GH 32173: a list of two lists as columns creates a multi-level column index.
        # GH 32173
        arrays = [list("abcd"), list("cdef")]
        result = DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=arrays)

        mi = MultiIndex.from_arrays(arrays)
        expected = DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=mi)

        tm.assert_frame_equal(result, expected)

    def test_constructor_wrong_length_nested_list_column(self):
        # [FUNCTION DOC]
        # GH 32173: nested column lists of length three with four data columns raise a
        # value error saying columns passed and data columns differ.
        # GH 32173
        arrays = [list("abc"), list("cde")]

        msg = "3 columns passed, passed data had 4"
        with pytest.raises(ValueError, match=msg):
            DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=arrays)

    def test_constructor_unequal_length_nested_list_column(self):
        # [FUNCTION DOC]
        # GH 32173: nested column lists of unequal lengths raise a value error from the
        # multi-level index constructor.
        # GH 32173
        arrays = [list("abcd"), list("cde")]

        # exception raised inside MultiIndex constructor
        msg = "all arrays must be same length"
        with pytest.raises(ValueError, match=msg):
            DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=arrays)

    @pytest.mark.parametrize(
        "data",
        [
            [[Timestamp("2021-01-01")]],
            [{"x": Timestamp("2021-01-01")}],
            {"x": [Timestamp("2021-01-01")]},
            {"x": Timestamp("2021-01-01")},
        ],
    )
    def test_constructor_one_element_data_list(self, data):
        # [FUNCTION DOC]
        # GH#42810: parametrized over a nested list, a list of dictionaries, a dictionary
        # of lists and a dictionary of scalars. With an index of three and one column, the
        # single timestamp is broadcast to all rows.
        # GH#42810
        result = DataFrame(data, index=range(3), columns=["x"])
        expected = DataFrame({"x": [Timestamp("2021-01-01")] * 3})
        tm.assert_frame_equal(result, expected)

    def test_constructor_sequence_like(self):
        # [FUNCTION DOC]
        # GH 3783: a list of sequence-like containers (custom class) is treated like a
        # list of lists (dtype not checked).
        # GH 3783
        # collections.Sequence like

        class DummyContainer(abc.Sequence):
            # [CLASS DOC]
            # HELPER: for the sequence-like test. A minimal immutable sequence wrapping a
            # list, based on the abstract sequence base class.
            def __init__(self, lst) -> None:
                # [FUNCTION DOC]
                # HELPER: stores the given list as the wrapped data.
                self._lst = lst

            def __getitem__(self, n):
                # [FUNCTION DOC]
                # HELPER: returns the wrapped list element or slice for the index.
                return self._lst.__getitem__(n)

            def __len__(self) -> int:
                # [FUNCTION DOC]
                # HELPER: returns the length of the wrapped list.
                return self._lst.__len__()

        lst_containers = [DummyContainer([1, "a"]), DummyContainer([2, "b"])]
        columns = ["num", "str"]
        result = DataFrame(lst_containers, columns=columns)
        expected = DataFrame([[1, "a"], [2, "b"]], columns=columns)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_constructor_stdlib_array(self):
        # [FUNCTION DOC]
        # GH 4297: standard-library typed arrays work as a column and as rows.
        # GH 4297
        # support Array
        result = DataFrame({"A": array.array("i", range(10))})
        expected = DataFrame({"A": list(range(10))})
        tm.assert_frame_equal(result, expected, check_dtype=False)

        expected = DataFrame([list(range(10)), list(range(10))])
        result = DataFrame([array.array("i", range(10)), array.array("i", range(10))])
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_constructor_range(self):
        # [FUNCTION DOC]
        # GH26342: a range builds a single-column frame equal to the list form.
        # GH26342
        result = DataFrame(range(10))
        expected = DataFrame(list(range(10)))
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_ranges(self):
        # [FUNCTION DOC]
        # A list of ranges builds rows equal to the list-of-lists form.
        result = DataFrame([range(10), range(10)])
        expected = DataFrame([list(range(10)), list(range(10))])
        tm.assert_frame_equal(result, expected)

    def test_constructor_iterable(self):
        # [FUNCTION DOC]
        # GH 21987: an iterable object whose iteration yields lists builds a frame of the
        # yielded rows.
        # GH 21987
        class Iter:
            # [CLASS DOC]
            # HELPER: for the iterable test. An object that is iterable and yields ten
            # lists of three integers; it has no length.
            def __iter__(self) -> Iterator:
                # [FUNCTION DOC]
                # HELPER: yields ten copies of the three-element list.
                for i in range(10):
                    yield [1, 2, 3]

        expected = DataFrame([[1, 2, 3]] * 10)
        result = DataFrame(Iter())
        tm.assert_frame_equal(result, expected)

    def test_constructor_iterator(self):
        # [FUNCTION DOC]
        # An iterator is consumed into a single column.
        result = DataFrame(iter(range(10)))
        expected = DataFrame(list(range(10)))
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_iterators(self):
        # [FUNCTION DOC]
        # A list of iterators builds rows.
        result = DataFrame([iter(range(10)), iter(range(10))])
        expected = DataFrame([list(range(10)), list(range(10))])
        tm.assert_frame_equal(result, expected)

    def test_constructor_generator(self):
        # [FUNCTION DOC]
        # Related #2305: a list of generators builds rows; a generator of two-element
        # lists builds columns of integers and a repeated string (dtype not checked).
        # related #2305

        gen1 = (i for i in range(10))
        gen2 = (i for i in range(10))

        expected = DataFrame([list(range(10)), list(range(10))])
        result = DataFrame([gen1, gen2])
        tm.assert_frame_equal(result, expected)

        gen = ([i, "a"] for i in range(10))
        result = DataFrame(gen)
        expected = DataFrame({0: range(10), 1: "a"})
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_constructor_list_of_dicts(self):
        # [FUNCTION DOC]
        # A list containing one empty dictionary builds a frame with one row and no
        # columns.
        result = DataFrame([{}])
        expected = DataFrame(index=RangeIndex(1), columns=[])
        tm.assert_frame_equal(result, expected)

    def test_constructor_ordered_dict_nested_preserve_order(self):
        # [FUNCTION DOC]
        # gh-18166: the order of inner keys of an ordered dictionary of ordered
        # dictionaries is preserved for the index.
        # see gh-18166
        nested1 = OrderedDict([("b", 1), ("a", 2)])
        nested2 = OrderedDict([("b", 2), ("a", 5)])
        data = OrderedDict([("col2", nested1), ("col1", nested2)])
        result = DataFrame(data)
        data = {"col2": [1, 2], "col1": [2, 5]}
        expected = DataFrame(data=data, index=["b", "a"])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dict_type", [dict, OrderedDict])
    def test_constructor_ordered_dict_preserve_order(self, dict_type):
        # [FUNCTION DOC]
        # gh-13304: parametrized over plain and ordered dictionary types. The key order
        # is kept for columns when building from a dictionary of lists and from a list
        # with one dictionary of scalars.
        # see gh-13304
        expected = DataFrame([[2, 1]], columns=["b", "a"])

        data = dict_type()
        data["b"] = [2]
        data["a"] = [1]

        result = DataFrame(data)
        tm.assert_frame_equal(result, expected)

        data = dict_type()
        data["b"] = 2
        data["a"] = 1

        result = DataFrame([data])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dict_type", [dict, OrderedDict])
    def test_constructor_ordered_dict_conflicting_orders(self, dict_type):
        # [FUNCTION DOC]
        # Parametrized over plain and ordered dictionary types. The first dictionary in a
        # list sets the column order even when later dictionaries have a different order.
        # the first dict element sets the ordering for the DataFrame,
        # even if there are conflicting orders from subsequent ones
        row_one = dict_type()
        row_one["b"] = 2
        row_one["a"] = 1

        row_two = dict_type()
        row_two["a"] = 1
        row_two["b"] = 2

        row_three = {"b": 2, "a": 1}

        expected = DataFrame([[2, 1], [2, 1]], columns=["b", "a"])
        result = DataFrame([row_one, row_two])
        tm.assert_frame_equal(result, expected)

        expected = DataFrame([[2, 1], [2, 1], [2, 1]], columns=["b", "a"])
        result = DataFrame([row_one, row_two, row_three])
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_series_aligned_index(self):
        # [FUNCTION DOC]
        # A list of Series with the same index (and names) builds rows labeled by their
        # names, with columns in the Series index order.
        series = [Series(i, index=["b", "a", "c"], name=str(i)) for i in range(3)]
        result = DataFrame(series)
        expected = DataFrame(
            {"b": [0, 1, 2], "a": [0, 1, 2], "c": [0, 1, 2]},
            columns=["b", "a", "c"],
            index=["0", "1", "2"],
        )
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_derived_dicts(self):
        # [FUNCTION DOC]
        # A list of a dictionary subclass produces the same frame as the plain dictionary.
        class CustomDict(dict):
            # [CLASS DOC]
            # HELPER: for the derived dictionaries test. A dictionary subclass with no
            # changes.
            pass

        d = {"a": 1.5, "b": 3}

        data_custom = [CustomDict(d)]
        data = [d]

        result_custom = DataFrame(data_custom)
        result = DataFrame(data)
        tm.assert_frame_equal(result, result_custom)

    def test_constructor_ragged(self):
        # [FUNCTION DOC]
        # Arrays of different lengths in a dictionary raise a value error saying all
        # arrays must have the same length.
        data = {
            "A": np.random.default_rng(2).standard_normal(10),
            "B": np.random.default_rng(2).standard_normal(8),
        }
        with pytest.raises(ValueError, match="All arrays must be of the same length"):
            DataFrame(data)

    def test_constructor_scalar(self):
        # [FUNCTION DOC]
        # A scalar value in a dictionary is broadcast to a given index (dtype not
        # checked).
        idx = Index(range(3))
        df = DataFrame({"a": 0}, index=idx)
        expected = DataFrame({"a": [0, 0, 0]}, index=idx)
        tm.assert_frame_equal(df, expected, check_dtype=False)

    def test_constructor_Series_copy_bug(self, float_frame):
        # [FUNCTION DOC]
        # Smoke test for a historical bug: building from a column Series with an explicit
        # index and columns, then copying it, does not raise.
        df = DataFrame(float_frame["A"], index=float_frame.index, columns=["A"])
        df.copy()

    def test_constructor_mixed_dict_and_Series(self):
        # [FUNCTION DOC]
        # A dictionary of a dictionary and a Series gives a sorted union index; a
        # dictionary with a plain list of an ambiguous mix raises a value error about
        # ambiguous ordering; a list with a Series indexed by the same labels is fine.
        data = {}
        data["A"] = {"foo": 1, "bar": 2, "baz": 3}
        data["B"] = Series([4, 3, 2, 1], index=["bar", "qux", "baz", "foo"])

        result = DataFrame(data)
        assert result.index.is_monotonic_increasing

        # ordering ambiguous, raise exception
        with pytest.raises(ValueError, match="ambiguous ordering"):
            DataFrame({"A": ["a", "b"], "B": {"a": "a", "b": "b"}})

        # this is OK though
        result = DataFrame({"A": ["a", "b"], "B": Series(["a", "b"], index=["a", "b"])})
        expected = DataFrame({"A": ["a", "b"], "B": ["a", "b"]}, index=["a", "b"])
        tm.assert_frame_equal(result, expected)

    def test_constructor_mixed_type_rows(self):
        # [FUNCTION DOC]
        # Issue 25075: a list mixing a list row and a tuple row builds a normal frame.
        # Issue 25075
        data = [[1, 2], (3, 4)]
        result = DataFrame(data)
        expected = DataFrame([[1, 2], [3, 4]])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "tuples,lists",
        [
            ((), []),
            (((),), [[]]),
            (((), ()), [(), ()]),
            (((), ()), [[], []]),
            (([], []), [[], []]),
            (([1], [2]), [[1], [2]]),  # GH 32776
            (([1, 2, 3], [4, 5, 6]), [[1, 2, 3], [4, 5, 6]]),
        ],
    )
    def test_constructor_tuple(self, tuples, lists):
        # [FUNCTION DOC]
        # GH 25691: parametrized over tuples and equivalent lists, including empty and
        # nested ones (GH 32776 for single-element rows). A tuple of rows builds the same
        # frame as the list of rows.
        # GH 25691
        result = DataFrame(tuples)
        expected = DataFrame(lists)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_tuples(self):
        # [FUNCTION DOC]
        # A dictionary column holding tuples keeps tuples as objects, equal to the Series
        # of tuples.
        result = DataFrame({"A": [(1, 2), (3, 4)]})
        expected = DataFrame({"A": Series([(1, 2), (3, 4)])})
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_namedtuples(self):
        # [FUNCTION DOC]
        # GH11181: a list of named tuples uses the field names as columns; an explicit
        # columns list overrides them.
        # GH11181
        named_tuple = namedtuple("Pandas", list("ab"))
        tuples = [named_tuple(1, 3), named_tuple(2, 4)]
        expected = DataFrame({"a": [1, 2], "b": [3, 4]})
        result = DataFrame(tuples)
        tm.assert_frame_equal(result, expected)

        # with columns
        expected = DataFrame({"y": [1, 2], "z": [3, 4]})
        result = DataFrame(tuples, columns=["y", "z"])
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses(self):
        # [FUNCTION DOC]
        # GH21910: a list of dataclass instances builds columns from their fields.
        # GH21910
        Point = make_dataclass("Point", [("x", int), ("y", int)])

        data = [Point(0, 3), Point(1, 3)]
        expected = DataFrame({"x": [0, 1], "y": [3, 3]})
        result = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses_with_varying_types(self):
        # [FUNCTION DOC]
        # GH21910: dataclasses with differing fields combine columns (union), filling
        # missing with NaN.
        # GH21910
        # varying types
        Point = make_dataclass("Point", [("x", int), ("y", int)])
        HLine = make_dataclass("HLine", [("x0", int), ("x1", int), ("y", int)])

        data = [Point(0, 3), HLine(1, 3, 3)]

        expected = DataFrame(
            {"x": [0, np.nan], "y": [3, 3], "x0": [np.nan, 1], "x1": [np.nan, 3]}
        )
        result = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses_error_thrown(self):
        # [FUNCTION DOC]
        # GH21910: a list mixing a dataclass and a dictionary raises a type error
        # saying the as-dictionary helper should be called on dataclass instances.
        # GH21910
        Point = make_dataclass("Point", [("x", int), ("y", int)])

        # expect TypeError
        msg = "asdict() should be called on dataclass instances"
        with pytest.raises(TypeError, match=re.escape(msg)):
            DataFrame([Point(0, 0), {"x": 1, "y": 0}])

    def test_constructor_list_of_dict_order(self):
        # [FUNCTION DOC]
        # GH10056: a list of dictionaries uses the first dictionary's key order, appending
        # new keys from later ones in their order of appearance, filling missing with None.
        # GH10056
        data = [
            {"First": 1, "Second": 4, "Third": 7, "Fourth": 10},
            {"Second": 5, "First": 2, "Fourth": 11, "Third": 8},
            {"Second": 6, "First": 3, "Fourth": 12, "Third": 9, "YYY": 14, "XXX": 13},
        ]
        expected = DataFrame(
            {
                "First": [1, 2, 3],
                "Second": [4, 5, 6],
                "Third": [7, 8, 9],
                "Fourth": [10, 11, 12],
                "YYY": [None, None, 14],
                "XXX": [None, None, 13],
            }
        )
        result = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_Series_named(self):
        # [FUNCTION DOC]
        # A Series gets its name as the single column label and keeps its index; an
        # unnamed Series gets the label zero and a shifted index is preserved; passing
        # too many columns for a Series raises a value error with a shape message; an
        # empty object Series keeps its name (#2234); a list of a named Series and an
        # array (or reverse) yields columns named by the Series name or position,
        # with unnamed ones labeled "Unnamed 0" or positional, and the arrays and Series
        # being treated differently.
        a = Series([1, 2, 3], index=["a", "b", "c"], name="x")
        df = DataFrame(a)
        assert df.columns[0] == "x"
        tm.assert_index_equal(df.index, a.index)

        # ndarray like
        arr = np.random.default_rng(2).standard_normal(10)
        s = Series(arr, name="x")
        df = DataFrame(s)
        expected = DataFrame({"x": s})
        tm.assert_frame_equal(df, expected)

        s = Series(arr, index=range(3, 13))
        df = DataFrame(s)
        expected = DataFrame({0: s})
        tm.assert_frame_equal(df, expected, check_column_type=False)

        msg = r"Shape of passed values is \(10, 1\), indices imply \(10, 2\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(s, columns=[1, 2])

        # #2234
        a = Series([], name="x", dtype=object)
        df = DataFrame(a)
        assert df.columns[0] == "x"

        # series with name and w/o
        s1 = Series(arr, name="x")
        df = DataFrame([s1, arr]).T
        expected = DataFrame({"x": s1, "Unnamed 0": arr}, columns=["x", "Unnamed 0"])
        tm.assert_frame_equal(df, expected)

        # this is a bit non-intuitive here; the series collapse down to arrays
        df = DataFrame([arr, s1]).T
        expected = DataFrame({1: s1, 0: arr}, columns=range(2))
        tm.assert_frame_equal(df, expected)

    def test_constructor_Series_named_and_columns(self):
        # [FUNCTION DOC]
        # GH 9232: a Series with a column name matching its own name gives its standard
        # frame; a non-matching name in the column list gives an empty frame.
        # GH 9232 validation

        s0 = Series(range(5), name=0)
        s1 = Series(range(5), name=1)

        # matching name and column gives standard frame
        tm.assert_frame_equal(DataFrame(s0, columns=[0]), s0.to_frame())
        tm.assert_frame_equal(DataFrame(s1, columns=[1]), s1.to_frame())

        # non-matching produces empty frame
        assert DataFrame(s0, columns=[1]).empty
        assert DataFrame(s1, columns=[0]).empty

    def test_constructor_Series_differently_indexed(self):
        # [FUNCTION DOC]
        # A Series with an index argument is reindexed to that index; the column is the
        # Series name or zero if unnamed.
        # name
        s1 = Series([1, 2, 3], index=["a", "b", "c"], name="x")

        # no name
        s2 = Series([1, 2, 3], index=["a", "b", "c"])

        other_index = Index(["a", "b"])

        df1 = DataFrame(s1, index=other_index)
        exp1 = DataFrame(s1.reindex(other_index))
        assert df1.columns[0] == "x"
        tm.assert_frame_equal(df1, exp1)

        df2 = DataFrame(s2, index=other_index)
        exp2 = DataFrame(s2.reindex(other_index))
        assert df2.columns[0] == 0
        tm.assert_index_equal(df2.index, other_index)
        tm.assert_frame_equal(df2, exp2)

    @pytest.mark.parametrize(
        "name_in1,name_in2,name_in3,name_out",
        [
            ("idx", "idx", "idx", "idx"),
            ("idx", "idx", None, None),
            ("idx", None, None, None),
            ("idx1", "idx2", None, None),
            ("idx1", "idx1", "idx2", None),
            ("idx1", "idx2", "idx3", None),
            (None, None, None, None),
        ],
    )
    def test_constructor_index_names(self, name_in1, name_in2, name_in3, name_out):
        # [FUNCTION DOC]
        # GH13475: parametrized over three index names and the expected name. A
        # dictionary of Series with differently named indexes takes the union index, whose
        # name is kept only if all three indexes agree (or the participating ones are
        # identical and none disagrees).
        # GH13475
        indices = [
            Index(["a", "b", "c"], name=name_in1),
            Index(["b", "c", "d"], name=name_in2),
            Index(["c", "d", "e"], name=name_in3),
        ]
        series = {
            c: Series([0, 1, 2], index=i) for i, c in zip(indices, ["x", "y", "z"])
        }
        result = DataFrame(series)

        exp_ind = Index(["a", "b", "c", "d", "e"], name=name_out)
        expected = DataFrame(
            {
                "x": [0, 1, 2, np.nan, np.nan],
                "y": [np.nan, 0, 1, 2, np.nan],
                "z": [np.nan, np.nan, 0, 1, 2],
            },
            index=exp_ind,
        )

        tm.assert_frame_equal(result, expected)

    def test_constructor_manager_resize(self, float_frame):
        # [FUNCTION DOC]
        # Constructing from a block manager with new index and columns emits a
        # deprecation warning about passing a manager and resizes (selects) accordingly.
        index = list(float_frame.index[:5])
        columns = list(float_frame.columns[:3])

        msg = "Passing a BlockManager to DataFrame"
        with tm.assert_produces_warning(
            DeprecationWarning, match=msg, check_stacklevel=False
        ):
            result = DataFrame(float_frame._mgr, index=index, columns=columns)
        tm.assert_index_equal(result.index, Index(index))
        tm.assert_index_equal(result.columns, Index(columns))

    def test_constructor_mix_series_nonseries(self, float_frame):
        # [FUNCTION DOC]
        # Mixing a Series and a list of the same length builds the expected frame; a
        # list shorter than the index raises a value error saying it does not match the
        # index length.
        df = DataFrame(
            {"A": float_frame["A"], "B": list(float_frame["B"])}, columns=["A", "B"]
        )
        tm.assert_frame_equal(df, float_frame.loc[:, ["A", "B"]])

        msg = "does not match index length"
        with pytest.raises(ValueError, match=msg):
            DataFrame({"A": float_frame["A"], "B": list(float_frame["B"])[:-2]})

    def test_constructor_miscast_na_int_dtype(self):
        # [FUNCTION DOC]
        # A list with NaN and an integer dtype raises the integer-casting-NaN error.
        msg = r"Cannot convert non-finite values \(NA or inf\) to integer"

        with pytest.raises(IntCastingNaNError, match=msg):
            DataFrame([[np.nan, 1], [1, 0]], dtype=np.int64)

    def test_constructor_column_duplicates(self):
        # [FUNCTION DOC]
        # Issue 2079: duplicate column labels pass through the constructor, by setting
        # the columns attribute, and via the record-based constructor, all equal.
        # it works! #2079
        df = DataFrame([[8, 5]], columns=["a", "a"])
        edf = DataFrame([[8, 5]])
        edf.columns = ["a", "a"]

        tm.assert_frame_equal(df, edf)

        idf = DataFrame.from_records([(8, 5)], columns=["a", "a"])

        tm.assert_frame_equal(idf, edf)

    def test_constructor_empty_with_string_dtype(self, using_infer_string):
        # [FUNCTION DOC]
        # GH 9428: an empty frame with a string, NumPy string or fixed-width unicode dtype
        # has object columns (or the string dtype with NaN for the plain str dtype when
        # the inferred-string option is on).
        # GH 9428
        expected = DataFrame(index=[0, 1], columns=[0, 1], dtype=object)
        expected_str = DataFrame(
            index=[0, 1], columns=[0, 1], dtype=pd.StringDtype(na_value=np.nan)
        )

        df = DataFrame(index=[0, 1], columns=[0, 1], dtype=str)
        if using_infer_string:
            tm.assert_frame_equal(df, expected_str)
        else:
            tm.assert_frame_equal(df, expected)
        df = DataFrame(index=[0, 1], columns=[0, 1], dtype=np.str_)
        tm.assert_frame_equal(df, expected)
        df = DataFrame(index=[0, 1], columns=[0, 1], dtype="U5")
        tm.assert_frame_equal(df, expected)

    def test_constructor_empty_with_string_extension(self, nullable_string_dtype):
        # [FUNCTION DOC]
        # GH 34915: an empty frame with a nullable string dtype and one column equals
        # another built the same way.
        # KNOWN ISSUE: the expected value is built with exactly the same call as the result, so the comparison cannot fail on any difference.
        # GH 34915
        expected = DataFrame(columns=["c1"], dtype=nullable_string_dtype)
        df = DataFrame(columns=["c1"], dtype=nullable_string_dtype)
        tm.assert_frame_equal(df, expected)

    def test_constructor_single_value(self):
        # [FUNCTION DOC]
        # A scalar fills the frame given index and columns (float, integer, string
        # cases with the right dtype). A scalar without both index and columns raises a
        # value error saying the constructor was not properly called; an incompatible
        # scalar and dtype raises a type error.
        # expecting single value upcasting here
        df = DataFrame(0.0, index=[1, 2, 3], columns=["a", "b", "c"])
        tm.assert_frame_equal(
            df, DataFrame(np.zeros(df.shape).astype("float64"), df.index, df.columns)
        )

        df = DataFrame(0, index=[1, 2, 3], columns=["a", "b", "c"])
        tm.assert_frame_equal(
            df, DataFrame(np.zeros(df.shape).astype("int64"), df.index, df.columns)
        )

        df = DataFrame("a", index=[1, 2], columns=["a", "c"])
        tm.assert_frame_equal(
            df,
            DataFrame(
                np.array([["a", "a"], ["a", "a"]], dtype=object),
                index=[1, 2],
                columns=["a", "c"],
            ),
        )

        msg = "DataFrame constructor not properly called!"
        with pytest.raises(ValueError, match=msg):
            DataFrame("a", [1, 2])
        with pytest.raises(ValueError, match=msg):
            DataFrame("a", columns=["a", "c"])

        msg = "incompatible data and dtype"
        with pytest.raises(TypeError, match=msg):
            DataFrame("a", [1, 2], ["a", "c"], float)

    def test_constructor_with_datetimes(self, using_infer_string):
        # [FUNCTION DOC]
        # Dtypes of columns built from scalars, a seconds-resolution timestamp and a
        # native datetime (microseconds); zero-dimensional NumPy arrays with a dtype
        # keep that dtype in the dictionary, as do full-length arrays; strings give object
        # (or the string dtype when inferred).
        intname = np.dtype(int).name
        floatname = np.dtype(np.float64).name
        objectname = np.dtype(np.object_).name

        # single item
        df = DataFrame(
            {
                "A": 1,
                "B": "foo",
                "C": "bar",
                "D": Timestamp("20010101").as_unit("s"),
                "E": datetime(2001, 1, 2, 0, 0),
            },
            index=np.arange(10),
        )
        result = df.dtypes
        expected = Series(
            [np.dtype("int64")]
            + [
                np.dtype(objectname)
                if not using_infer_string
                else pd.StringDtype(na_value=np.nan)
            ]
            * 2
            + [np.dtype("M8[s]"), np.dtype("M8[us]")],
            index=list("ABCDE"),
        )
        tm.assert_series_equal(result, expected)

        # check with ndarray construction ndim==0 (e.g. we are passing an ndim 0
        # ndarray with a dtype specified)
        df = DataFrame(
            {
                "a": 1.0,
                "b": 2,
                "c": "foo",
                floatname: np.array(1.0, dtype=floatname),
                intname: np.array(1, dtype=intname),
            },
            index=np.arange(10),
        )
        result = df.dtypes
        expected = Series(
            [
                np.dtype("float64"),
                np.dtype("int64"),
                np.dtype("object")
                if not using_infer_string
                else pd.StringDtype(na_value=np.nan),
                np.dtype("float64"),
                np.dtype(intname),
            ],
            index=["a", "b", "c", floatname, intname],
        )
        tm.assert_series_equal(result, expected)

        # check with ndarray construction ndim>0
        df = DataFrame(
            {
                "a": 1.0,
                "b": 2,
                "c": "foo",
                floatname: np.array([1.0] * 10, dtype=floatname),
                intname: np.array([1] * 10, dtype=intname),
            },
            index=np.arange(10),
        )
        result = df.dtypes
        expected = Series(
            [
                np.dtype("float64"),
                np.dtype("int64"),
                np.dtype("object")
                if not using_infer_string
                else pd.StringDtype(na_value=np.nan),
                np.dtype("float64"),
                np.dtype(intname),
            ],
            index=["a", "b", "c", floatname, intname],
        )
        tm.assert_series_equal(result, expected)

    def test_constructor_with_datetimes1(self):
        # [FUNCTION DOC]
        # GH 2809: a Series of native datetimes has microsecond datetime dtype.
        # KNOWN ISSUE: despite the name, this does not exercise the frame constructor, only the Series.
        # GH 2809
        ind = date_range(start="2000-01-01", freq="D", periods=10)
        datetimes = [ts.to_pydatetime() for ts in ind]
        datetime_s = Series(datetimes)
        assert datetime_s.dtype == "M8[us]"

    def test_constructor_with_datetimes2(self):
        # [FUNCTION DOC]
        # GH 2810: a frame from datetimes has microsecond datetime dtype, and a column
        # of plain dates added later has object dtype.
        # GH 2810
        ind = date_range(start="2000-01-01", freq="D", periods=10)
        datetimes = [ts.to_pydatetime() for ts in ind]
        dates = [ts.date() for ts in ind]
        df = DataFrame(datetimes, columns=["datetimes"])
        df["dates"] = dates
        result = df.dtypes
        expected = Series(
            [np.dtype("datetime64[us]"), np.dtype("object")],
            index=["datetimes", "dates"],
        )
        tm.assert_series_equal(result, expected)

    def test_constructor_with_datetimes3(self):
        # [FUNCTION DOC]
        # GH 7594: timezone-aware native datetimes in a dictionary or in a list of
        # dictionaries keep their zone-aware dtype with microsecond resolution.
        # GH 7594
        # don't coerce tz-aware
        dt = datetime(2012, 1, 1, tzinfo=zoneinfo.ZoneInfo("US/Eastern"))

        df = DataFrame({"End Date": dt}, index=[0])
        assert df.iat[0, 0] == dt
        tm.assert_series_equal(
            df.dtypes, Series({"End Date": "datetime64[us, US/Eastern]"}, dtype=object)
        )

        df = DataFrame([{"End Date": dt}])
        assert df.iat[0, 0] == dt
        tm.assert_series_equal(
            df.dtypes, Series({"End Date": "datetime64[us, US/Eastern]"}, dtype=object)
        )

    def test_constructor_with_datetimes4(self):
        # [FUNCTION DOC]
        # GH 8411: date ranges naive, UTC and Eastern give columns whose first element's
        # zone is none, UTC and Eastern respectively.
        # tz-aware (UTC and other tz's)
        # GH 8411
        dr = date_range("20130101", periods=3)
        df = DataFrame({"value": dr})
        assert df.iat[0, 0].tz is None
        dr = date_range("20130101", periods=3, tz="UTC")
        df = DataFrame({"value": dr})
        assert str(df.iat[0, 0].tz) == "UTC"
        dr = date_range("20130101", periods=3, tz="US/Eastern")
        df = DataFrame({"value": dr})
        assert str(df.iat[0, 0].tz) == "US/Eastern"

    def test_constructor_with_datetimes5(self):
        # [FUNCTION DOC]
        # GH 7822: setting a zone-aware range as a column of an empty frame or building
        # from a dictionary preserves the zone and equals the Series-based expectation.
        # GH 7822
        # preserver an index with a tz on dict construction
        i = date_range("1/1/2011", periods=5, freq="10s", tz="US/Eastern")

        expected = DataFrame({"a": i.to_series().reset_index(drop=True)})
        df = DataFrame()
        df["a"] = i
        tm.assert_frame_equal(df, expected)

        df = DataFrame({"a": i})
        tm.assert_frame_equal(df, expected)

    def test_constructor_with_datetimes6(self):
        # [FUNCTION DOC]
        # A dictionary of a zone-aware and a naive range keeps both dtypes.
        # multiples
        i = date_range("1/1/2011", periods=5, freq="10s", tz="US/Eastern")
        i_no_tz = date_range("1/1/2011", periods=5, freq="10s")
        df = DataFrame({"a": i, "b": i_no_tz})
        expected = DataFrame({"a": i.to_series().reset_index(drop=True), "b": i_no_tz})
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "arr",
        [
            np.array([None, None, None, None, datetime.now(), None]),
            np.array([None, None, datetime.now(), None]),
            [[np.datetime64("NaT", "ns")], [None]],
            [[np.datetime64("NaT", "ns")], [pd.NaT]],
            [[None], [np.datetime64("NaT", "ns")]],
            [[None], [pd.NaT]],
            [[pd.NaT], [np.datetime64("NaT", "ns")]],
            [[pd.NaT], [None]],
        ],
    )
    def test_constructor_datetimes_with_nulls(self, arr):
        # [FUNCTION DOC]
        # gh-15869, GH#11220: parametrized over arrays of None with a datetime, and lists
        # of NaT-like values. The resulting column dtype is datetime with microsecond
        # unit if from a native-datetime array, nanosecond if any NumPy datetime NaT is
        # present, else second.
        # KNOWN ISSUE: a comment in the test says the condition for choosing second resolution is unclear about why behavior differs; it is an unexplained special case.
        # gh-15869, GH#11220
        result = DataFrame(arr).dtypes
        unit = "ns"
        if isinstance(arr, np.ndarray):
            # inferred from a pydatetime object
            unit = "us"
        elif not any(isinstance(x, np.datetime64) for y in arr for x in y):
            # TODO: this condition is not clear about why we have different behavior
            unit = "s"
        expected = Series([np.dtype(f"datetime64[{unit}]")])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("order", ["K", "A", "C", "F"])
    @pytest.mark.parametrize(
        "unit",
        ["M", "D", "h", "m", "s", "ms", "us", "ns"],
    )
    def test_constructor_datetimes_non_ns(self, order, unit):
        # [FUNCTION DOC]
        # Parametrized over memory order and datetime unit (month through nanosecond).
        # A 2-D datetime array in that unit builds a frame equal to the nanosecond
        # conversion cast to the unit; units coarser than seconds are not castable (a type
        # error says cannot cast) and the constructor instead picks seconds.
        dtype = f"datetime64[{unit}]"
        na = np.array(
            [
                ["2015-01-01", "2015-01-02", "2015-01-03"],
                ["2017-01-01", "2017-01-02", "2017-02-03"],
            ],
            dtype=dtype,
            order=order,
        )
        df = DataFrame(na)
        expected = DataFrame(na.astype("M8[ns]"))
        if unit in ["M", "D", "h", "m"]:
            with pytest.raises(TypeError, match="Cannot cast"):
                expected.astype(dtype)

            # instead the constructor casts to the closest supported reso, i.e. "s"
            expected = expected.astype("datetime64[s]")
        else:
            expected = expected.astype(dtype=dtype)

        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("order", ["K", "A", "C", "F"])
    @pytest.mark.parametrize(
        "unit",
        [
            "D",
            "h",
            "m",
            "s",
            "ms",
            "us",
            "ns",
        ],
    )
    def test_constructor_timedelta_non_ns(self, order, unit):
        # [FUNCTION DOC]
        # Parametrized over memory order and timedelta unit (day through nanosecond). A
        # 2-D timedelta array builds a frame whose unit is the nearest supported one
        # (seconds for day, hour, minute) or the same unit.
        dtype = f"timedelta64[{unit}]"
        na = np.array(
            [
                [np.timedelta64(1, "D"), np.timedelta64(2, "D")],
                [np.timedelta64(4, "D"), np.timedelta64(5, "D")],
            ],
            dtype=dtype,
            order=order,
        )
        df = DataFrame(na)
        if unit in ["D", "h", "m"]:
            # we get the nearest supported unit, i.e. "s"
            exp_unit = "s"
        else:
            exp_unit = unit
        exp_dtype = np.dtype(f"m8[{exp_unit}]")
        expected = DataFrame(
            [
                [Timedelta(1, "D"), Timedelta(2, "D")],
                [Timedelta(4, "D"), Timedelta(5, "D")],
            ],
            dtype=exp_dtype,
        )
        # TODO(2.0): ideally we should get the same 'expected' without passing
        #  dtype=exp_dtype.
        tm.assert_frame_equal(df, expected)

    def test_constructor_for_list_with_dtypes(self, using_infer_string):
        # [FUNCTION DOC]
        # Dtype inference for lists of lists/arrays: default integer, 32-bit integers
        # preserved, large integers become 64-bit, default float and integer cases,
        # scalars with index, and a dictionary of object, float, string, datetime values
        # giving 64-bit integer, float, object/string, datetime, float (GH #2751).
        # test list of lists/ndarrays
        df = DataFrame([np.arange(5) for x in range(5)])
        result = df.dtypes
        expected = Series([np.dtype("int")] * 5)
        tm.assert_series_equal(result, expected)

        df = DataFrame([np.array(np.arange(5), dtype="int32") for x in range(5)])
        result = df.dtypes
        expected = Series([np.dtype("int32")] * 5)
        tm.assert_series_equal(result, expected)

        # overflow issue? (we always expected int64 upcasting here)
        df = DataFrame({"a": [2**31, 2**31 + 1]})
        assert df.dtypes.iloc[0] == np.dtype("int64")

        # GH #2751 (construction with no index specified), make sure we cast to
        # platform values
        df = DataFrame([1, 2])
        assert df.dtypes.iloc[0] == np.dtype("int64")

        df = DataFrame([1.0, 2.0])
        assert df.dtypes.iloc[0] == np.dtype("float64")

        df = DataFrame({"a": [1, 2]})
        assert df.dtypes.iloc[0] == np.dtype("int64")

        df = DataFrame({"a": [1.0, 2.0]})
        assert df.dtypes.iloc[0] == np.dtype("float64")

        df = DataFrame({"a": 1}, index=range(3))
        assert df.dtypes.iloc[0] == np.dtype("int64")

        df = DataFrame({"a": 1.0}, index=range(3))
        assert df.dtypes.iloc[0] == np.dtype("float64")

        # with object list
        df = DataFrame(
            {
                "a": [1, 2, 4, 7],
                "b": [1.2, 2.3, 5.1, 6.3],
                "c": list("abcd"),
                "d": [datetime(2000, 1, 1) for i in range(4)],
                "e": [1.0, 2, 4.0, 7],
            }
        )
        result = df.dtypes
        expected = Series(
            [
                np.dtype("int64"),
                np.dtype("float64"),
                np.dtype("object")
                if not using_infer_string
                else pd.StringDtype(na_value=np.nan),
                np.dtype("datetime64[us]"),
                np.dtype("float64"),
            ],
            index=list("abcde"),
        )
        tm.assert_series_equal(result, expected)

    def test_constructor_frame_copy(self, float_frame):
        # [FUNCTION DOC]
        # Constructing with copy true and assigning a column in the copy leaves the
        # original column unchanged.
        cop = DataFrame(float_frame, copy=True)
        cop["A"] = 5
        assert (cop["A"] == 5).all()
        assert not (float_frame["A"] == 5).all()

    def test_constructor_frame_shallow_copy(self, float_frame):
        # [FUNCTION DOC]
        # GH 49523: constructing a frame from a frame without copy gives a new manager
        # (shallow copy); replacing the new frame's index does not change the original.
        # constructing a DataFrame from DataFrame with copy=False should still
        # give a "shallow" copy (share data, not attributes)
        # https://github.com/pandas-dev/pandas/issues/49523
        orig = float_frame.copy()
        cop = DataFrame(float_frame)
        assert cop._mgr is not float_frame._mgr
        # Overwriting index of copy doesn't change original
        cop.index = np.arange(len(cop))
        tm.assert_frame_equal(float_frame, orig)

    def test_constructor_ndarray_copy(self, float_frame):
        # [FUNCTION DOC]
        # Constructing from an array without copy still keeps later array edits out of
        # the frame; with an explicit copy the array edits do not show either.
        arr = float_frame.values.copy()
        df = DataFrame(arr)

        arr[5] = 5
        assert not (df.values[5] == 5).all()
        df = DataFrame(arr, copy=True)
        arr[6] = 6
        assert not (df.values[6] == 6).all()

    def test_constructor_series_copy(self, float_frame):
        # [FUNCTION DOC]
        # Constructing from a Series with copy true and writing 5 over the whole column
        # leaves the original Series unchanged.
        series = float_frame._series

        df = DataFrame({"A": series["A"]}, copy=True)
        # TODO can be replaced with `df.loc[:, "A"] = 5` after deprecation about
        # inplace mutation is enforced
        df.loc[df.index[0] : df.index[-1], "A"] = 5

        assert not (series["A"] == 5).all()

    @pytest.mark.parametrize(
        "df",
        [
            DataFrame([[1, 2, 3], [4, 5, 6]], index=[1, np.nan]),
            DataFrame([[1, 2, 3], [4, 5, 6]], columns=[1.1, 2.2, np.nan]),
            DataFrame([[0, 1, 2, 3], [4, 5, 6, 7]], columns=[np.nan, 1.1, 2.2, np.nan]),
            DataFrame(
                [[0.0, 1, 2, 3.0], [4, 5, 6, 7]], columns=[np.nan, 1.1, 2.2, np.nan]
            ),
            DataFrame([[0.0, 1, 2, 3.0], [4, 5, 6, 7]], columns=[np.nan, 1, 2, 2]),
        ],
    )
    def test_constructor_with_nas(self, df):
        # [FUNCTION DOC]
        # GH 5016, GH 21428: parametrized over frames with missing values in the index or
        # columns (including non-unique). All columns are accessible by position; with no
        # NaN column label looking up NaN raises a key error; a single NaN label gives a
        # Series, multiple a frame.
        # GH 5016
        # na's in indices
        # GH 21428 (non-unique columns)

        for i in range(len(df.columns)):
            df.iloc[:, i]

        indexer = np.arange(len(df.columns))[isna(df.columns)]

        # No NaN found -> error
        if len(indexer) == 0:
            with pytest.raises(KeyError, match="^nan$"):
                df.loc[:, np.nan]
        # single nan should result in Series
        elif len(indexer) == 1:
            tm.assert_series_equal(df.iloc[:, indexer[0]], df.loc[:, np.nan])
        # multiple nans should result in DataFrame
        else:
            tm.assert_frame_equal(df.iloc[:, indexer], df.loc[:, np.nan])

    def test_constructor_lists_to_object_dtype(self):
        # [FUNCTION DOC]
        # From #1074: a column of NaN and False has object dtype and False is falsy.
        # from #1074
        d = DataFrame({"a": [np.nan, False]})
        assert d["a"].dtype == np.object_
        assert not d["a"][1]

    def test_constructor_ndarray_categorical_dtype(self):
        # [FUNCTION DOC]
        # Broadcasting a categorical column array to four columns and constructing with
        # the categorical dtype gives four categorical columns equal to the original.
        cat = Categorical(["A", "B", "C"])
        arr = np.array(cat).reshape(-1, 1)
        arr = np.broadcast_to(arr, (3, 4))

        result = DataFrame(arr, dtype=cat.dtype)

        expected = DataFrame({0: cat, 1: cat, 2: cat, 3: cat})
        tm.assert_frame_equal(result, expected)

    def test_constructor_categorical(self):
        # [FUNCTION DOC]
        # GH8626: categorical dtype applied to a dictionary column, to Series-to-frame
        # conversion (default column zero, or a custom name), and to a plain list keeps
        # the categorical dtype.
        # GH8626

        # dict creation
        df = DataFrame({"A": list("abc")}, dtype="category")
        expected = Series(list("abc"), dtype="category", name="A")
        tm.assert_series_equal(df["A"], expected)

        # to_frame
        s = Series(list("abc"), dtype="category")
        result = s.to_frame()
        expected = Series(list("abc"), dtype="category", name=0)
        tm.assert_series_equal(result[0], expected)
        result = s.to_frame(name="foo")
        expected = Series(list("abc"), dtype="category", name="foo")
        tm.assert_series_equal(result["foo"], expected)

        # list-like creation
        df = DataFrame(list("abc"), dtype="category")
        expected = Series(list("abc"), dtype="category", name=0)
        tm.assert_series_equal(df[0], expected)

    def test_construct_from_1item_list_of_categorical(self):
        # [FUNCTION DOC]
        # Since 2.0, a list with one categorical is treated as a row of objects (no
        # special case), equal to the object-cast form.
        # pre-2.0 this behaved as DataFrame({0: cat}), in 2.0 we remove
        #  Categorical special case
        # ndim != 1
        cat = Categorical(list("abc"))
        df = DataFrame([cat])
        expected = DataFrame([cat.astype(object)])
        tm.assert_frame_equal(df, expected)

    def test_construct_from_list_of_categoricals(self):
        # [FUNCTION DOC]
        # Since 2.0, a list of two categoricals builds two object rows.
        # pre-2.0 this behaved as DataFrame({0: cat}), in 2.0 we remove
        #  Categorical special case

        df = DataFrame([Categorical(list("abc")), Categorical(list("abd"))])
        expected = DataFrame([["a", "b", "c"], ["a", "b", "d"]])
        tm.assert_frame_equal(df, expected)

    def test_from_nested_listlike_mixed_types(self):
        # [FUNCTION DOC]
        # A list mixing a categorical and a plain list builds two object rows.
        # pre-2.0 this behaved as DataFrame({0: cat}), in 2.0 we remove
        #  Categorical special case
        # mixed
        df = DataFrame([Categorical(list("abc")), list("def")])
        expected = DataFrame([["a", "b", "c"], ["d", "e", "f"]])
        tm.assert_frame_equal(df, expected)

    def test_construct_from_listlikes_mismatched_lengths(self):
        # [FUNCTION DOC]
        # A list of categoricals with different lengths builds rows of unequal length,
        # padded with missing.
        df = DataFrame([Categorical(list("abc")), Categorical(list("abdefg"))])
        expected = DataFrame([list("abc"), list("abdefg")])
        tm.assert_frame_equal(df, expected)

    def test_constructor_categorical_series(self):
        # [FUNCTION DOC]
        # Constructing a Series with the category dtype equals casting; inserting a
        # categorical Series into a frame with a different index aligns it and gives
        # all-missing categorical values (GH 8076).
        items = [1, 2, 3, 1]
        exp = Series(items).astype("category")
        res = Series(items, dtype="category")
        tm.assert_series_equal(res, exp)

        items = ["a", "b", "c", "a"]
        exp = Series(items).astype("category")
        res = Series(items, dtype="category")
        tm.assert_series_equal(res, exp)

        # insert into frame with different index
        # GH 8076
        index = date_range("20000101", periods=3)
        expected = Series(
            Categorical(values=[np.nan, np.nan, np.nan], categories=["a", "b", "c"])
        )
        expected.index = index

        expected = DataFrame({"x": expected})
        df = DataFrame({"x": Series(["a", "b", "c"], dtype="category")}, index=index)
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "dtype",
        tm.ALL_NUMERIC_DTYPES
        + tm.DATETIME64_DTYPES
        + tm.TIMEDELTA64_DTYPES
        + tm.BOOL_DTYPES,
    )
    def test_check_dtype_empty_numeric_column(self, dtype):
        # [FUNCTION DOC]
        # GH24386: parametrized over numeric, datetime, timedelta and boolean dtypes. A
        # frame with data not matching the requested column gets the requested dtype for
        # the empty column.
        # GH24386: Ensure dtypes are set correctly for an empty DataFrame.
        # Empty DataFrame is generated via dictionary data with non-overlapping columns.
        data = DataFrame({"a": [1, 2]}, columns=["b"], dtype=dtype)

        assert data.b.dtype == dtype

    @pytest.mark.parametrize(
        "dtype", tm.STRING_DTYPES + tm.BYTES_DTYPES + tm.OBJECT_DTYPES
    )
    def test_check_dtype_empty_string_column(self, dtype):
        # [FUNCTION DOC]
        # GH24386: for string, bytes and object dtypes, the empty column has the object
        # dtype name.
        # GH24386: Ensure dtypes are set correctly for an empty DataFrame.
        # Empty DataFrame is generated via dictionary data with non-overlapping columns.
        data = DataFrame({"a": [1, 2]}, columns=["b"], dtype=dtype)
        assert data.b.dtype.name == "object"

    def test_to_frame_with_falsey_names(self):
        # [FUNCTION DOC]
        # GH 16114: converting an empty Series named zero (a falsy name) to a frame, and
        # constructing from it, give a zero-labeled object column.
        # GH 16114
        result = Series(name=0, dtype=object).to_frame().dtypes
        expected = Series({0: object})
        tm.assert_series_equal(result, expected)

        result = DataFrame(Series(name=0, dtype=object)).dtypes
        tm.assert_series_equal(result, expected)

    @pytest.mark.arm_slow
    @pytest.mark.parametrize("dtype", [None, "uint8", "category"])
    def test_constructor_range_dtype(self, dtype):
        # [FUNCTION DOC]
        # Marked slow on ARM. Parametrized over dtype (none, unsigned 8-bit, categorical).
        # A range builds the expected values with that dtype (or 64-bit integers) both
        # as list-like data with columns (GH 26342) and as a dictionary value (GH 16804).
        expected = DataFrame({"A": [0, 1, 2, 3, 4]}, dtype=dtype or "int64")

        # GH 26342
        result = DataFrame(range(5), columns=["A"], dtype=dtype)
        tm.assert_frame_equal(result, expected)

        # GH 16804
        result = DataFrame({"A": range(5)}, dtype=dtype)
        tm.assert_frame_equal(result, expected)

    def test_frame_from_list_subclass(self):
        # [FUNCTION DOC]
        # GH21226: nested lists of a list subclass behave as ordinary lists of lists.
        # GH21226
        class List(list):
            # [CLASS DOC]
            # HELPER: for the list subclass test. A list subclass with no changes.
            pass

        expected = DataFrame([[1, 2, 3], [4, 5, 6]])
        result = DataFrame(List([List([1, 2, 3]), List([4, 5, 6])]))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "extension_arr",
        [
            Categorical(list("aabbc")),
            SparseArray([1, np.nan, np.nan, np.nan]),
            IntervalArray([Interval(0, 1), Interval(1, 5)]),
            PeriodArray(pd.period_range(start="1/1/2017", end="1/1/2018", freq="M")),
        ],
    )
    def test_constructor_with_extension_array(self, extension_arr):
        # [FUNCTION DOC]
        # GH11363: parametrized over categorical, sparse, interval and period arrays.
        # Constructing a frame directly from an extension array equals building from a
        # Series wrapping it.
        # GH11363
        expected = DataFrame(Series(extension_arr))
        result = DataFrame(extension_arr)
        tm.assert_frame_equal(result, expected)

    def test_datetime_date_tuple_columns_from_dict(self):
        # [FUNCTION DOC]
        # GH 10863: a tuple of dates as a column label in a dictionary, with columns
        # specified, builds a frame with that tuple as the sole label.
        # GH 10863
        v = date.today()
        tup = v, v
        result = DataFrame({tup: Series(range(3), index=range(3))}, columns=[tup])
        expected = DataFrame([0, 1, 2], columns=Index(Series([tup])))
        tm.assert_frame_equal(result, expected)

    def test_construct_with_two_categoricalindex_series(self):
        # [FUNCTION DOC]
        # GH 14600: a list of two Series with categorical indexes aligns on the union of
        # categories, yielding float values with missing, in order of first appearance.
        # GH 14600
        s1 = Series([39, 6, 4], index=CategoricalIndex(["female", "male", "unknown"]))
        s2 = Series(
            [2, 152, 2, 242, 150],
            index=CategoricalIndex(["f", "female", "m", "male", "unknown"]),
        )
        result = DataFrame([s1, s2])
        expected = DataFrame(
            np.array([[39, 6, 4, np.nan, np.nan], [152.0, 242.0, 150.0, 2.0, 2.0]]),
            columns=["female", "male", "unknown", "f", "m"],
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in cast:RuntimeWarning"
    )
    def test_constructor_series_nonexact_categoricalindex(self):
        # [FUNCTION DOC]
        # Filters a cast warning. GH 42424: a dictionary of two Series, indexed by
        # interval categories from cutting data into ten bins and taking the head and
        # tail of the counts, aligns on the full categorical interval index with missing
        # where absent.
        # GH 42424
        ser = Series(range(100))
        ser1 = cut(ser, 10).value_counts().head(5)
        ser2 = cut(ser, 10).value_counts().tail(5)
        result = DataFrame({"1": ser1, "2": ser2})
        index = CategoricalIndex(
            [
                Interval(-0.099, 9.9, closed="right"),
                Interval(9.9, 19.8, closed="right"),
                Interval(19.8, 29.7, closed="right"),
                Interval(29.7, 39.6, closed="right"),
                Interval(39.6, 49.5, closed="right"),
                Interval(49.5, 59.4, closed="right"),
                Interval(59.4, 69.3, closed="right"),
                Interval(69.3, 79.2, closed="right"),
                Interval(79.2, 89.1, closed="right"),
                Interval(89.1, 99, closed="right"),
            ],
            ordered=True,
        )
        expected = DataFrame(
            {"1": [10] * 5 + [np.nan] * 5, "2": [np.nan] * 5 + [10] * 5}, index=index
        )
        tm.assert_frame_equal(expected, result)

    def test_from_M8_structured(self):
        # [FUNCTION DOC]
        # A structured array of datetime fields builds datetime columns equal to the
        # dates; a Series of one field gives Timestamp elements.
        dates = [(datetime(2012, 9, 9, 0, 0), datetime(2012, 9, 8, 15, 10))]
        arr = np.array(dates, dtype=[("Date", "M8[us]"), ("Forecasting", "M8[us]")])
        df = DataFrame(arr)

        assert df["Date"][0] == dates[0][0]
        assert df["Forecasting"][0] == dates[0][1]

        s = Series(arr["Date"])
        assert isinstance(s[0], Timestamp)
        assert s[0] == dates[0][0]

    def test_from_datetime_subclass(self):
        # [FUNCTION DOC]
        # GH21142: a subclass of the native datetime in a dictionary column gives
        # microsecond datetime dtype.
        # GH21142 Verify whether Datetime subclasses are also of dtype datetime
        class DatetimeSubclass(datetime):
            # [CLASS DOC]
            # HELPER: for the datetime subclass test. A native datetime subclass with no
            # changes.
            pass

        data = DataFrame({"datetime": [DatetimeSubclass(2020, 1, 1, 1, 1)]})
        assert data.datetime.dtype == "datetime64[us]"

    def test_with_mismatched_index_length_raises(self):
        # [FUNCTION DOC]
        # GH#33437: a timezone-aware datetime index with an index of different length
        # raises a value error about shape or array length.
        # GH#33437
        dti = date_range("2016-01-01", periods=3, tz="US/Pacific")
        msg = "Shape of passed values|Passed arrays should have the same length"
        with pytest.raises(ValueError, match=msg):
            DataFrame(dti, index=range(4))

    def test_frame_ctor_datetime64_column(self):
        # [FUNCTION DOC]
        # A dictionary of a float array and a NumPy datetime array gives a datetime64
        # nanosecond column.
        rng = date_range("1/1/2000 00:00:00", "1/1/2000 1:59:50", freq="10s")
        dates = np.asarray(rng)

        df = DataFrame(
            {"A": np.random.default_rng(2).standard_normal(len(rng)), "B": dates}
        )
        assert np.issubdtype(df["B"].dtype, np.dtype("M8[ns]"))

    def test_dataframe_constructor_infer_multiindex(self):
        # [FUNCTION DOC]
        # Passing a list of NumPy arrays as index builds a multi-level row index but not
        # multi-level columns; as columns it builds multi-level columns.
        index_lists = [["a", "a", "b", "b"], ["x", "y", "x", "y"]]

        multi = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=[np.array(x) for x in index_lists],
        )
        assert isinstance(multi.index, MultiIndex)
        assert not isinstance(multi.columns, MultiIndex)

        multi = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)), columns=index_lists
        )
        assert isinstance(multi.columns, MultiIndex)

    @pytest.mark.parametrize(
        "input_vals",
        [
            ([1, 2]),
            (["1", "2"]),
            (list(date_range("1/1/2011", periods=2, freq="h"))),
            (list(date_range("1/1/2011", periods=2, freq="h", tz="US/Eastern"))),
            ([Interval(left=0, right=5)]),
        ],
    )
    def test_constructor_list_str(self, input_vals, string_dtype):
        # [FUNCTION DOC]
        # GH#16605: parametrized over inputs (integers, strings, timestamps with and
        # without zone, an interval) and string dtype fixture. A dictionary column with a
        # string dtype equals building without a dtype and casting to the string dtype.
        # GH#16605
        # Ensure that data elements are converted to strings when
        # dtype is str, 'str', or 'U'

        result = DataFrame({"A": input_vals}, dtype=string_dtype)
        expected = DataFrame({"A": input_vals}).astype({"A": string_dtype})
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_str_na(self, string_dtype):
        # [FUNCTION DOC]
        # A float column with a missing value and a string dtype gives string values and a
        # missing as None in an object frame.
        result = DataFrame({"A": [1.0, 2.0, None]}, dtype=string_dtype)
        expected = DataFrame({"A": ["1.0", "2.0", None]}, dtype=object)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("copy", [False, True])
    def test_dict_nocopy(
        self,
        copy,
        any_numeric_ea_dtype,
        any_numpy_dtype,
    ):
        # [FUNCTION DOC]
        # Parametrized over copy flag, numeric extension dtype fixture and NumPy dtype
        # fixture. Checks view semantics of constructing from a dictionary of two NumPy
        # arrays and an extension array: with copy disabled the underlying buffers are
        # shared; writes via position are checked, extension values are replaced
        # without altering the original array when copying; string/bytes dtypes are
        # skipped; non-numeric NumPy kinds raise a type error on invalid values and the
        # test stops.
        # KNOWN ISSUE: comments in the test state that most of the rest of it belongs in indexing tests and carry FIXME notes (GH#35417) about assertions that are written in the reverse direction until that issue is resolved.
        a = np.array([1, 2], dtype=any_numpy_dtype)
        b = np.array([3, 4], dtype=any_numpy_dtype)
        if b.dtype.kind in ["S", "U"]:
            # These get cast, making the checks below more cumbersome
            pytest.skip(f"{b.dtype} get cast, making the checks below more cumbersome")

        c = pd.array([1, 2], dtype=any_numeric_ea_dtype)
        c_orig = c.copy()
        df = DataFrame({"a": a, "b": b, "c": c}, copy=copy)

        def get_base(obj):
            # [FUNCTION DOC]
            # HELPER: for the no-copy test. Returns the base array of a NumPy array or of a
            # datetime-like array's underlying NumPy array; raises a type error otherwise.
            if isinstance(obj, np.ndarray):
                return obj.base
            elif isinstance(obj.dtype, np.dtype):
                # i.e. DatetimeArray, TimedeltaArray
                return obj._ndarray.base
            else:
                raise TypeError

        def check_views(c_only: bool = False):
            # [FUNCTION DOC]
            # HELPER: for the no-copy test. Asserts that exactly one block holds the extension
            # array object itself and, unless only that is requested, that exactly one NumPy-backed
            # block is based on each of the two input arrays.
            # Check that the underlying data behind df["c"] is still `c`
            #  after setting with iloc.  Since we don't know which entry in
            #  df._mgr.blocks corresponds to df["c"], we just check that exactly
            #  one of these arrays is `c`.  GH#38939
            assert sum(x.values is c for x in df._mgr.blocks) == 1
            if c_only:
                # If we ever stop consolidating in setitem_with_indexer,
                #  this will become unnecessary.
                return

            assert (
                sum(
                    get_base(x.values) is a
                    for x in df._mgr.blocks
                    if isinstance(x.values.dtype, np.dtype)
                )
                == 1
            )
            assert (
                sum(
                    get_base(x.values) is b
                    for x in df._mgr.blocks
                    if isinstance(x.values.dtype, np.dtype)
                )
                == 1
            )

        if not copy:
            # constructor preserves views
            check_views()

        # TODO: most of the rest of this test belongs in indexing tests
        should_raise = not lib.is_np_dtype(df.dtypes.iloc[0], "fciuO")
        if should_raise:
            with pytest.raises(TypeError, match="Invalid value"):
                df.iloc[0, 0] = 0
                df.iloc[0, 1] = 0
            return
        else:
            df.iloc[0, 0] = 0
            df.iloc[0, 1] = 0
        if not copy:
            check_views(True)

        # FIXME(GH#35417): until GH#35417, iloc.setitem into EA values does not preserve
        #  view, so we have to check in the other direction
        df.iloc[:, 2] = pd.array([45, 46], dtype=c.dtype)
        assert df.dtypes.iloc[2] == c.dtype
        if copy:
            if a.dtype.kind == "M":
                assert a[0] == a.dtype.type(1, "ns")
                assert b[0] == b.dtype.type(3, "ns")
            else:
                assert a[0] == a.dtype.type(1)
                assert b[0] == b.dtype.type(3)
            # FIXME(GH#35417): enable after GH#35417
            assert c[0] == c_orig[0]  # i.e. df.iloc[0, 2]=45 did *not* update c

    def test_construct_from_dict_ea_series(self):
        # [FUNCTION DOC]
        # GH#53744: a dictionary with a nullable-integer Series copies by default, so the
        # frame column does not share memory with the Series data.
        # GH#53744 - default of copy=True should also apply for Series with
        # extension dtype
        ser = Series([1, 2, 3], dtype="Int64")
        df = DataFrame({"a": ser})
        assert not np.shares_memory(ser.values._data, df["a"].values._data)

    def test_from_series_with_name_with_columns(self):
        # [FUNCTION DOC]
        # GH 7893: a named Series with a different column name gives an empty frame with
        # only the requested column.
        # GH 7893
        result = DataFrame(Series(1, name="foo"), columns=["bar"])
        expected = DataFrame(columns=["bar"])
        tm.assert_frame_equal(result, expected)

    def test_nested_list_columns(self):
        # [FUNCTION DOC]
        # GH 14467: a nested list of labels as columns builds a multi-level column index.
        # GH 14467
        result = DataFrame(
            [[1, 2, 3], [4, 5, 6]], columns=[["A", "A", "A"], ["a", "b", "c"]]
        )
        expected = DataFrame(
            [[1, 2, 3], [4, 5, 6]],
            columns=MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("A", "c")]),
        )
        tm.assert_frame_equal(result, expected)

    def test_from_2d_object_array_of_periods_or_intervals(self):
        # [FUNCTION DOC]
        # Period analogue of GH#26825: 2-D object arrays of periods or intervals infer
        # period or interval dtypes per column; a mix infers each column separately.
        # Period analogue to GH#26825
        pi = pd.period_range("2016-04-05", periods=3)
        data = pi._data.astype(object).reshape(1, -1)
        df = DataFrame(data)
        assert df.shape == (1, 3)
        assert (df.dtypes == pi.dtype).all()
        assert (df == pi).all().all()

        ii = pd.IntervalIndex.from_breaks([3, 4, 5, 6])
        data2 = ii._data.astype(object).reshape(1, -1)
        df2 = DataFrame(data2)
        assert df2.shape == (1, 3)
        assert (df2.dtypes == ii.dtype).all()
        assert (df2 == ii).all().all()

        # mixed
        data3 = np.r_[data, data2, data, data2].T
        df3 = DataFrame(data3)
        expected = DataFrame({0: pi, 1: ii, 2: pi, 3: ii})
        tm.assert_frame_equal(df3, expected)

    @pytest.mark.parametrize(
        "col_a, col_b",
        [
            ([[1], [2]], np.array([[1], [2]])),
            (np.array([[1], [2]]), [[1], [2]]),
            (np.array([[1], [2]]), np.array([[1], [2]])),
        ],
    )
    def test_error_from_2darray(self, col_a, col_b):
        # [FUNCTION DOC]
        # Parametrized over list/array combinations of two-dimensional columns. A
        # dictionary whose values are two-dimensional raises a value error saying
        # per-column arrays must be one-dimensional.
        msg = "Per-column arrays must each be 1-dimensional"
        with pytest.raises(ValueError, match=msg):
            DataFrame({"a": col_a, "b": col_b})

    def test_from_dict_with_missing_copy_false(self):
        # [FUNCTION DOC]
        # GH#45369: constructing with index, columns and no data with copy disabled gives
        # columns that do not share memory; setting a value changes only the first column
        # (object dtype with NaN).
        # GH#45369 filled columns should not be views of one another
        df = DataFrame(index=[1, 2, 3], columns=["a", "b", "c"], copy=False)
        assert not np.shares_memory(df["a"]._values, df["b"]._values)

        df.iloc[0, 0] = 0
        expected = DataFrame(
            {
                "a": [0, np.nan, np.nan],
                "b": [np.nan, np.nan, np.nan],
                "c": [np.nan, np.nan, np.nan],
            },
            index=[1, 2, 3],
            dtype=object,
        )
        tm.assert_frame_equal(df, expected)

    def test_construction_empty_array_multi_column_raises(self):
        # [FUNCTION DOC]
        # GH#46822: an empty array with two column labels raises a value error with a
        # shape message.
        # GH#46822
        msg = r"Shape of passed values is \(0, 1\), indices imply \(0, 2\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(data=np.array([]), columns=["a", "b"])

    def test_construct_with_strings_and_none(self):
        # [FUNCTION DOC]
        # GH#32218: strings and None with the string dtype build the expected frame.
        # GH#32218
        df = DataFrame(["1", "2", None], columns=["a"], dtype="str")
        expected = DataFrame({"a": ["1", "2", None]}, dtype="str")
        tm.assert_frame_equal(df, expected)

    def test_frame_string_inference(self):
        # [FUNCTION DOC]
        # GH#54430: with the inferred-string option on, string data, string index and
        # string column labels use the NaN-based string dtype, mixed string and integer
        # stays object (column labels still strings), and an explicit object dtype is
        # respected.
        # GH#54430
        dtype = pd.StringDtype(na_value=np.nan)
        expected = DataFrame(
            {"a": ["a", "b"]}, dtype=dtype, columns=Index(["a"], dtype=dtype)
        )
        with pd.option_context("future.infer_string", True):
            df = DataFrame({"a": ["a", "b"]})
        tm.assert_frame_equal(df, expected)

        expected = DataFrame(
            {"a": ["a", "b"]},
            dtype=dtype,
            columns=Index(["a"], dtype=dtype),
            index=Index(["x", "y"], dtype=dtype),
        )
        with pd.option_context("future.infer_string", True):
            df = DataFrame({"a": ["a", "b"]}, index=["x", "y"])
        tm.assert_frame_equal(df, expected)

        expected = DataFrame(
            {"a": ["a", 1]}, dtype="object", columns=Index(["a"], dtype=dtype)
        )
        with pd.option_context("future.infer_string", True):
            df = DataFrame({"a": ["a", 1]})
        tm.assert_frame_equal(df, expected)

        expected = DataFrame(
            {"a": ["a", "b"]}, dtype="object", columns=Index(["a"], dtype=dtype)
        )
        with pd.option_context("future.infer_string", True):
            df = DataFrame({"a": ["a", "b"]}, dtype="object")
        tm.assert_frame_equal(df, expected)

    def test_frame_string_inference_array_string_dtype(self):
        # [FUNCTION DOC]
        # GH#54496: with the inferred-string option, NumPy string arrays (1-D in
        # dictionary, 2-D with and without columns) are converted to the string dtype.
        # GH#54496
        dtype = pd.StringDtype(na_value=np.nan)
        expected = DataFrame(
            {"a": ["a", "b"]}, dtype=dtype, columns=Index(["a"], dtype=dtype)
        )
        with pd.option_context("future.infer_string", True):
            df = DataFrame({"a": np.array(["a", "b"])})
        tm.assert_frame_equal(df, expected)

        expected = DataFrame({0: ["a", "b"], 1: ["c", "d"]}, dtype=dtype)
        with pd.option_context("future.infer_string", True):
            df = DataFrame(np.array([["a", "c"], ["b", "d"]]))
        tm.assert_frame_equal(df, expected)

        expected = DataFrame(
            {"a": ["a", "b"], "b": ["c", "d"]},
            dtype=dtype,
            columns=Index(["a", "b"], dtype=dtype),
        )
        with pd.option_context("future.infer_string", True):
            df = DataFrame(np.array([["a", "c"], ["b", "d"]]), columns=["a", "b"])
        tm.assert_frame_equal(df, expected)

    def test_frame_string_inference_block_dim(self):
        # [FUNCTION DOC]
        # GH#55363: with the inferred-string option, a 2-D NumPy string array makes the
        # first internal block two-dimensional.
        # GH#55363
        with pd.option_context("future.infer_string", True):
            df = DataFrame(np.array([["hello", "goodbye"], ["hello", "Hello"]]))
        assert df._mgr.blocks[0].ndim == 2

    @pytest.mark.parametrize("klass", [Series, Index])
    def test_inference_on_pandas_objects(self, klass):
        # [FUNCTION DOC]
        # GH#56012: parametrized over Series and Index. Wrapping a pandas object of object
        # dtype holding timestamps is not re-inferred: the frame column keeps object
        # dtype whether passed as data with columns or in a dictionary.
        # GH#56012
        obj = klass([Timestamp("2019-12-31")], dtype=object)
        result = DataFrame(obj, columns=["a"])
        assert result.dtypes.iloc[0] == np.object_

        result = DataFrame({"a": obj})
        assert result.dtypes.iloc[0] == np.object_

    def test_dict_keys_returns_rangeindex(self):
        # [FUNCTION DOC]
        # A dictionary with integer keys zero and one yields a range index as columns.
        result = DataFrame({0: [1], 1: [2]}).columns
        expected = RangeIndex(2)
        tm.assert_index_equal(result, expected, exact=True)

    @pytest.mark.parametrize(
        "cons", [Series, Index, DatetimeIndex, DataFrame, pd.array, pd.to_datetime]
    )
    def test_construction_datetime_resolution_inference(self, cons):
        # [FUNCTION DOC]
        # Parametrized over several constructors (Series, Index, datetime index, frame,
        # array factory, datetime conversion). A far-future naive timestamp list gives
        # microsecond resolution; a zone-aware one gives microsecond with zone.
        ts = Timestamp(2999, 1, 1)
        ts2 = ts.tz_localize("US/Pacific")

        obj = cons([ts])
        res_dtype = tm.get_dtype(obj)
        assert res_dtype == "M8[us]", res_dtype

        obj2 = cons([ts2])
        res_dtype2 = tm.get_dtype(obj2)
        assert res_dtype2 == "M8[us, US/Pacific]", res_dtype2

    def test_construction_nan_value_timedelta64_dtype(self):
        # [FUNCTION DOC]
        # GH#60064: a list with None and one with timedelta nanosecond dtype gives NaT and
        # a one-nanosecond timedelta.
        # GH#60064
        result = DataFrame([None, 1], dtype="timedelta64[ns]")
        expected = DataFrame(
            ["NaT", "0 days 00:00:00.000000001"], dtype="timedelta64[ns]"
        )
        tm.assert_frame_equal(result, expected)

    def test_dataframe_from_array_like_with_name_attribute(self):
        # [FUNCTION DOC]
        # GH#61443: an array subclass carrying a name attribute is treated as plain
        # array data: a frame from it equals one from the underlying array.
        # GH#61443
        class DummyArray(np.ndarray):
            # [CLASS DOC]
            # HELPER: for the named array test. A NumPy array subclass whose creator wraps
            # an input array as a view and sets a name attribute to a fixed string.
            def __new__(cls, input_array):
                # [FUNCTION DOC]
                # HELPER: constructor hook of the array subclass; views the input as the subclass,
                # sets the name attribute to a fixed string and returns the new instance.
                obj = np.asarray(input_array).view(cls)
                obj.name = "foo"
                return obj

        dummy = DummyArray(np.eye(3))
        df = DataFrame(dummy)
        expected = DataFrame(np.eye(3))
        tm.assert_frame_equal(df, expected)


class TestDataFrameConstructorIndexInference:
    # [CLASS DOC]
    # Test group for how the frame constructor infers the row index when combining
    # Series with different indexes (period, timezone-mixed, Arrow-categorical
    # column labels).
    def test_frame_from_dict_of_series_overlapping_monthly_period_indexes(self):
        # [FUNCTION DOC]
        # Two monthly period-indexed Series with overlapping ranges in a dictionary give
        # the full monthly period range as index.
        rng1 = pd.period_range("1/1/1999", "1/1/2012", freq="M")
        s1 = Series(np.random.default_rng(2).standard_normal(len(rng1)), rng1)

        rng2 = pd.period_range("1/1/1980", "12/1/2001", freq="M")
        s2 = Series(np.random.default_rng(2).standard_normal(len(rng2)), rng2)
        df = DataFrame({"s1": s1, "s2": s2})

        exp = pd.period_range("1/1/1980", "1/1/2012", freq="M")
        tm.assert_index_equal(df.index, exp)

    def test_frame_from_dict_with_mixed_tzaware_indexes(self):
        # [FUNCTION DOC]
        # GH#44091: dictionaries of Series with a mix of zone-aware (UTC, Central) indexes
        # and a plain integer index join into an object index concatenating the indexes
        # in dictionary order; adding a naive datetime Series (in any order) raises a type
        # error saying it cannot join naive and aware indexes.
        # KNOWN ISSUE: a comment in the test says it is unclear whether raising is the desired behavior; the test records de facto behavior as of December 2021.
        # GH#44091
        dti = date_range("2016-01-01", periods=3)

        ser1 = Series(range(3), index=dti)
        ser2 = Series(range(3), index=dti.tz_localize("UTC"))
        ser3 = Series(range(3), index=dti.tz_localize("US/Central"))
        ser4 = Series(range(3))

        # no tz-naive, but we do have mixed tzs and a non-DTI
        df1 = DataFrame({"A": ser2, "B": ser3, "C": ser4})
        exp_index = Index(
            list(ser2.index) + list(ser3.index) + list(ser4.index), dtype=object
        )
        tm.assert_index_equal(df1.index, exp_index)

        df2 = DataFrame({"A": ser2, "C": ser4, "B": ser3})
        exp_index3 = Index(
            list(ser2.index) + list(ser4.index) + list(ser3.index), dtype=object
        )
        tm.assert_index_equal(df2.index, exp_index3)

        df3 = DataFrame({"B": ser3, "A": ser2, "C": ser4})
        exp_index3 = Index(
            list(ser3.index) + list(ser2.index) + list(ser4.index), dtype=object
        )
        tm.assert_index_equal(df3.index, exp_index3)

        df4 = DataFrame({"C": ser4, "B": ser3, "A": ser2})
        exp_index4 = Index(
            list(ser4.index) + list(ser3.index) + list(ser2.index), dtype=object
        )
        tm.assert_index_equal(df4.index, exp_index4)

        # TODO: not clear if these raising is desired (no extant tests),
        #  but this is de facto behavior 2021-12-22
        msg = "Cannot join tz-naive with tz-aware DatetimeIndex"
        with pytest.raises(TypeError, match=msg):
            DataFrame({"A": ser2, "B": ser3, "C": ser4, "D": ser1})
        with pytest.raises(TypeError, match=msg):
            DataFrame({"A": ser2, "B": ser3, "D": ser1})
        with pytest.raises(TypeError, match=msg):
            DataFrame({"D": ser1, "A": ser2, "B": ser3})

    @pytest.mark.parametrize(
        "key_val, col_vals, col_type",
        [
            ["3", ["3", "4"], "utf8"],
            [3, [3, 4], "int8"],
        ],
    )
    def test_dict_data_arrow_column_expansion(self, key_val, col_vals, col_type):
        # [FUNCTION DOC]
        # GH 53617: skipped without pyarrow. Parametrized over key/value kinds (string
        # and 8-bit integer dictionary-encoded Arrow columns). A dictionary with one key
        # and an Arrow extension array as columns expands to two columns, the second
        # all-missing object.
        # GH 53617
        pa = pytest.importorskip("pyarrow")
        cols = pd.arrays.ArrowExtensionArray(
            pa.array(col_vals, type=pa.dictionary(pa.int8(), getattr(pa, col_type)()))
        )
        result = DataFrame({key_val: [1, 2]}, columns=cols)
        expected = DataFrame([[1, np.nan], [2, np.nan]], columns=cols)
        expected.isetitem(1, expected.iloc[:, 1].astype(object))
        tm.assert_frame_equal(result, expected)


class TestDataFrameConstructorWithDtypeCoercion:
    # [CLASS DOC]
    # Test group for coercion of float or datetime data to a requested dtype in the
    # constructor.
    def test_floating_values_integer_dtype(self):
        # [FUNCTION DOC]
        # GH#40110, GH#49599: float data with an integer dtype raises a value error
        # about coercing floats to integers; rounded floats are accepted; NaN data raises
        # the integer-casting-NaN error for frames and Series, same as with the astype
        # method.
        # GH#40110 make DataFrame behavior with arraylike floating data and
        #  inty dtype match Series behavior

        arr = np.random.default_rng(2).standard_normal((10, 5))

        # GH#49599 in 2.0 we raise instead of either
        #  a) silently ignoring dtype and returningfloat (the old Series behavior) or
        #  b) rounding (the old DataFrame behavior)
        msg = "Trying to coerce float values to integers"
        with pytest.raises(ValueError, match=msg):
            DataFrame(arr, dtype="i8")

        df = DataFrame(arr.round(), dtype="i8")
        assert (df.dtypes == "i8").all()

        # with NaNs, we go through a different path with a different warning
        arr[0, 0] = np.nan
        msg = r"Cannot convert non-finite values \(NA or inf\) to integer"
        with pytest.raises(IntCastingNaNError, match=msg):
            DataFrame(arr, dtype="i8")
        with pytest.raises(IntCastingNaNError, match=msg):
            Series(arr[0], dtype="i8")
        # The future (raising) behavior matches what we would get via astype:
        msg = r"Cannot convert non-finite values \(NA or inf\) to integer"
        with pytest.raises(IntCastingNaNError, match=msg):
            DataFrame(arr).astype("i8")
        with pytest.raises(IntCastingNaNError, match=msg):
            Series(arr[0]).astype("i8")


class TestDataFrameConstructorWithDatetimeTZ:
    # [CLASS DOC]
    # Test group for construction with timezone-aware datetimes.
    @pytest.mark.parametrize("tz", ["US/Eastern", "dateutil/US/Eastern"])
    def test_construction_preserves_tzaware_dtypes(self, tz):
        # [FUNCTION DOC]
        # Parametrized over two timezone spellings. A dictionary with a zone-aware range
        # keeps the zone dtype (nanosecond); assigned columns of range, zone range, native
        # datetimes (microsecond) and zone-aware native datetimes keep their expected
        # dtypes (GH#7822, GH#2810).
        # after GH#7822
        # these retain the timezones on dict construction
        dr = date_range("2011/1/1", "2012/1/1", freq="W-FRI", unit="ns")
        dr_tz = dr.tz_localize(tz)
        df = DataFrame({"A": "foo", "B": dr_tz}, index=dr)
        tz_expected = DatetimeTZDtype("ns", dr_tz.tzinfo)
        assert df["B"].dtype == tz_expected

        # GH#2810 (with timezones)
        datetimes_naive = [ts.to_pydatetime() for ts in dr]
        datetimes_with_tz = [ts.to_pydatetime() for ts in dr_tz]
        df = DataFrame({"dr": dr})
        df["dr_tz"] = dr_tz
        df["datetimes_naive"] = datetimes_naive
        df["datetimes_with_tz"] = datetimes_with_tz
        result = df.dtypes
        expected = Series(
            [
                np.dtype("datetime64[ns]"),
                DatetimeTZDtype(tz=tz),
                np.dtype("datetime64[us]"),
                DatetimeTZDtype(tz=tz, unit="us"),
            ],
            index=["dr", "dr_tz", "datetimes_naive", "datetimes_with_tz"],
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("pydt", [True, False])
    def test_constructor_data_aware_dtype_naive(self, tz_aware_fixture, pydt):
        # [FUNCTION DOC]
        # GH#25843, GH#41555, GH#33401: parametrized over zone-aware fixture and
        # native datetime or Timestamp. Requesting a naive datetime dtype with zone-aware
        # data raises a value error telling the caller to localize to none, via list,
        # array, nested list and Series inputs; scalars raise a type error saying a
        # zone-aware timestamp cannot be unboxed to a naive dtype.
        # GH#25843, GH#41555, GH#33401
        tz = tz_aware_fixture
        ts = Timestamp("2019", tz=tz)
        if pydt:
            ts = ts.to_pydatetime()

        msg = (
            "Cannot convert timezone-aware data to timezone-naive dtype. "
            r"Use pd.Series\(values\).dt.tz_localize\(None\) instead."
        )
        with pytest.raises(ValueError, match=msg):
            DataFrame({0: [ts]}, dtype="datetime64[ns]")

        msg2 = "Cannot unbox tzaware Timestamp to tznaive dtype"
        with pytest.raises(TypeError, match=msg2):
            DataFrame({0: ts}, index=[0], dtype="datetime64[ns]")

        with pytest.raises(ValueError, match=msg):
            DataFrame([ts], dtype="datetime64[ns]")

        with pytest.raises(ValueError, match=msg):
            DataFrame(np.array([ts], dtype=object), dtype="datetime64[ns]")

        with pytest.raises(TypeError, match=msg2):
            DataFrame(ts, index=[0], columns=[0], dtype="datetime64[ns]")

        with pytest.raises(ValueError, match=msg):
            DataFrame([Series([ts])], dtype="datetime64[ns]")

        with pytest.raises(ValueError, match=msg):
            DataFrame([[ts]], columns=[0], dtype="datetime64[ns]")

    def test_from_dict(self):
        # [FUNCTION DOC]
        # 8260: a dictionary of a zone-aware index and a naive range keeps names and
        # equals the corresponding Series.
        # KNOWN ISSUE: the first assertion has a message string as second operand and so only checks that the dtype object is truthy; it does not compare the dtype with the intended text (which is also missing its closing bracket).
        # 8260
        # support datetime64 with tz

        idx = Index(date_range("20130101", periods=3, tz="US/Eastern"), name="foo")
        dr = date_range("20130110", periods=3)

        # construction
        df = DataFrame({"A": idx, "B": dr})
        assert df["A"].dtype, "M8[ns, US/Eastern"
        assert df["A"].name == "A"
        tm.assert_series_equal(df["A"], Series(idx, name="A"))
        tm.assert_series_equal(df["B"], Series(dr, name="B"))

    def test_from_index(self):
        # [FUNCTION DOC]
        # A zone-aware named index, or a Series of it, builds a frame with the index
        # name (or zero) as the column label.
        # from index
        idx2 = date_range("20130101", periods=3, tz="US/Eastern", name="foo")
        df2 = DataFrame(idx2)
        tm.assert_series_equal(df2["foo"], Series(idx2, name="foo"))
        df2 = DataFrame(Series(idx2))
        tm.assert_series_equal(df2["foo"], Series(idx2, name="foo"))

        idx2 = date_range("20130101", periods=3, tz="US/Eastern")
        df2 = DataFrame(idx2)
        tm.assert_series_equal(df2[0], Series(idx2, name=0))
        df2 = DataFrame(Series(idx2))
        tm.assert_series_equal(df2[0], Series(idx2, name=0))

    def test_frame_dict_constructor_datetime64_1680(self):
        # [FUNCTION DOC]
        # Issue 1680 smoke test: a dictionary with a string scalar and a datetime Series
        # with a datetime index constructs without error (also with its values).
        dr = date_range("1/1/2012", periods=10)
        s = Series(dr, index=dr)

        # it works!
        DataFrame({"a": "foo", "b": s}, index=dr)
        DataFrame({"a": "foo", "b": s.values}, index=dr)

    def test_frame_datetime64_mixed_index_ctor_1681(self):
        # [FUNCTION DOC]
        # Issue 1681 smoke test: a datetime Series (default index) aligned onto a
        # datetime index gives all-missing values in that column.
        dr = date_range("2011/1/1", "2012/1/1", freq="W-FRI")
        ts = Series(dr)

        # it works!
        d = DataFrame({"A": "foo", "B": ts}, index=dr)
        assert d["B"].isna().all()

    def test_frame_timeseries_column(self):
        # [FUNCTION DOC]
        # GH19157: building from a zone-aware index with a column name gives a column of
        # zone-aware nanosecond timestamps.
        # GH19157
        dr = date_range(
            start="20130101T10:00:00", periods=3, freq="min", tz="US/Eastern", unit="ns"
        )
        result = DataFrame(dr, columns=["timestamps"])
        expected = DataFrame(
            {
                "timestamps": [
                    Timestamp("20130101T10:00:00", tz="US/Eastern"),
                    Timestamp("20130101T10:01:00", tz="US/Eastern"),
                    Timestamp("20130101T10:02:00", tz="US/Eastern"),
                ]
            },
            dtype="M8[ns, US/Eastern]",
        )
        tm.assert_frame_equal(result, expected)

    def test_nested_dict_construction(self):
        # [FUNCTION DOC]
        # GH22227: a nested dictionary with explicit index and columns reindexes,
        # filling missing with NaN.
        # GH22227
        columns = ["Nevada", "Ohio"]
        pop = {
            "Nevada": {2001: 2.4, 2002: 2.9},
            "Ohio": {2000: 1.5, 2001: 1.7, 2002: 3.6},
        }
        result = DataFrame(pop, index=[2001, 2002, 2003], columns=columns)
        expected = DataFrame(
            [(2.4, 1.7), (2.9, 3.6), (np.nan, np.nan)],
            columns=columns,
            index=Index([2001, 2002, 2003]),
        )
        tm.assert_frame_equal(result, expected)

    def test_from_tzaware_object_array(self):
        # [FUNCTION DOC]
        # GH#26825: a 2-D object array of zone-aware timestamps infers the zone dtype and
        # equals the index of the timestamps.
        # GH#26825 2D object array of tzaware timestamps should not raise
        dti = date_range("2016-04-05 04:30", periods=3, tz="UTC")
        data = dti._data.astype(object).reshape(1, -1)
        df = DataFrame(data)
        assert df.shape == (1, 3)
        assert (df.dtypes == dti.dtype).all()
        assert (df == dti).all().all()

    def test_from_tzaware_mixed_object_array(self):
        # [FUNCTION DOC]
        # GH#26825: a 2-D object array with naive, Eastern and CET columns (with NaT)
        # infers per-column datetime dtypes with microsecond resolution.
        # GH#26825
        arr = np.array(
            [
                [
                    Timestamp("2013-01-01 00:00:00"),
                    Timestamp("2013-01-02 00:00:00"),
                    Timestamp("2013-01-03 00:00:00"),
                ],
                [
                    Timestamp("2013-01-01 00:00:00-0500", tz="US/Eastern"),
                    pd.NaT,
                    Timestamp("2013-01-03 00:00:00-0500", tz="US/Eastern"),
                ],
                [
                    Timestamp("2013-01-01 00:00:00+0100", tz="CET"),
                    pd.NaT,
                    Timestamp("2013-01-03 00:00:00+0100", tz="CET"),
                ],
            ],
            dtype=object,
        ).T
        res = DataFrame(arr, columns=["A", "B", "C"])

        expected_dtypes = [
            "datetime64[us]",
            "datetime64[us, US/Eastern]",
            "datetime64[us, CET]",
        ]
        assert (res.dtypes == expected_dtypes).all()

    def test_from_2d_ndarray_with_dtype(self):
        # [FUNCTION DOC]
        # GH#12513: a 2-D integer array with a zone-aware datetime dtype equals the cast
        # of the integer frame.
        # GH#12513
        array_dim2 = np.arange(10).reshape((5, 2))
        df = DataFrame(array_dim2, dtype="datetime64[ns, UTC]")

        expected = DataFrame(array_dim2).astype("datetime64[ns, UTC]")
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("typ", [set, frozenset])
    def test_construction_from_set_raises(self, typ):
        # [FUNCTION DOC]
        # Parametrized over set and frozen set. A set (unordered) in a dictionary or Series
        # raises a type error saying the type is unordered.
        # https://github.com/pandas-dev/pandas/issues/32582
        values = typ({1, 2, 3})
        msg = f"'{typ.__name__}' type is unordered"
        with pytest.raises(TypeError, match=msg):
            DataFrame({"a": values})

        with pytest.raises(TypeError, match=msg):
            Series(values)

    def test_construction_from_ndarray_datetimelike(self):
        # [FUNCTION DOC]
        # A 2-D datetime64 NumPy array builds blocks backed by datetime arrays.
        # ensure the underlying arrays are properly wrapped as EA when
        # constructed from 2D ndarray
        arr = np.arange(0, 12, dtype="datetime64[ns]").reshape(4, 3)
        df = DataFrame(arr)
        assert all(isinstance(block.values, DatetimeArray) for block in df._mgr.blocks)

    def test_construction_from_ndarray_with_eadtype_mismatched_columns(self):
        # [FUNCTION DOC]
        # A float matrix with one column label and an extension dtype, or an extension
        # array with two column labels, raises a value error about arrays and columns
        # lengths.
        arr = np.random.default_rng(2).standard_normal((10, 2))
        dtype = pd.array([2.0]).dtype
        msg = r"len\(arrays\) must match len\(columns\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(arr, columns=["foo"], dtype=dtype)

        arr2 = pd.array([2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match=msg):
            DataFrame(arr2, columns=["foo", "bar"])

    def test_columns_indexes_raise_on_sets(self):
        # [FUNCTION DOC]
        # GH 47215: sets as index or columns raise value errors saying a set is not
        # allowed.
        # GH 47215
        data = [[1, 2, 3], [4, 5, 6]]
        with pytest.raises(ValueError, match="index cannot be a set"):
            DataFrame(data, index={"a", "b"})
        with pytest.raises(ValueError, match="columns cannot be a set"):
            DataFrame(data, columns={"a", "b", "c"})

    def test_from_dict_with_columns_na_scalar(self):
        # [FUNCTION DOC]
        # A dictionary with a NaT scalar, given columns and an index builds a datetime
        # column of NaT.
        result = DataFrame({"a": pd.NaT}, columns=["a"], index=range(2))
        expected = DataFrame({"a": Series([pd.NaT, pd.NaT])})
        tm.assert_frame_equal(result, expected)

    # TODO: make this not cast to object in pandas 3.0
    @pytest.mark.skipif(
        not np_version_gt2, reason="StringDType only available in numpy 2 and above"
    )
    @pytest.mark.parametrize(
        "data",
        [
            {"a": ["a", "b", "c"], "b": [1.0, 2.0, 3.0], "c": ["d", "e", "f"]},
        ],
    )
    def test_np_string_array_object_cast(self, data):
        # [FUNCTION DOC]
        # Skipped unless NumPy 2 or later. Parametrized over a dictionary data fixture. A
        # column of NumPy's variable-width string dtype is cast to object dtype by the
        # constructor.
        # KNOWN ISSUE: a comment above the test says the object cast is a TODO that is meant to change in a later release, so the asserted object dtype is acknowledged as temporary behavior.
        from numpy.dtypes import StringDType

        data["a"] = np.array(data["a"], dtype=StringDType())
        res = DataFrame(data)
        assert res["a"].dtype == np.object_
        assert (res["a"] == data["a"]).all()


def get1(obj):  # TODO: make a helper in tm?
    # [FUNCTION DOC]
    # HELPER: for the scalar construction tests. Returns the first element of a
    # Series, or the first row-first column element of a frame.
    # KNOWN ISSUE: a comment on the definition says this should become a helper in the shared testing module.
    if isinstance(obj, Series):
        return obj.iloc[0]
    else:
        return obj.iloc[0, 0]


class TestFromScalar:
    # [CLASS DOC]
    # Test group for constructing frames and Series from a scalar (optionally
    # wrapped in a list or dictionary) with a requested dtype, using constructor
    # factories from fixtures.
    @pytest.fixture(params=[list, dict, None])
    def box(self, request):
        # [FUNCTION DOC]
        # HELPER: fixture parametrized over a wrapping kind (list, dictionary, none);
        # returns the kind.
        return request.param

    @pytest.fixture
    def constructor(self, frame_or_series, box):
        # [FUNCTION DOC]
        # HELPER: fixture building a constructor factory from the frame-or-Series fixture
        # and the wrapping kind. All use index of length two (and a single column A for
        # frames). With no wrapping the scalar is passed directly; with a dictionary the
        # scalar is the dictionary value (two keys for Series); with a list the scalar is
        # repeated twice. Extra keyword arguments such as dtype pass through.
        extra = {"index": range(2)}
        if frame_or_series is DataFrame:
            extra["columns"] = ["A"]

        if box is None:
            return functools.partial(frame_or_series, **extra)

        elif box is dict:
            if frame_or_series is Series:
                return lambda x, **kwargs: frame_or_series(
                    {0: x, 1: x}, **extra, **kwargs
                )
            else:
                return lambda x, **kwargs: frame_or_series({"A": x}, **extra, **kwargs)
        elif frame_or_series is Series:
            return lambda x, **kwargs: frame_or_series([x, x], **extra, **kwargs)
        else:
            return lambda x, **kwargs: frame_or_series({"A": [x, x]}, **extra, **kwargs)

    @pytest.mark.parametrize("dtype", ["M8[ns]", "m8[ns]"])
    def test_from_nat_scalar(self, dtype, constructor):
        # [FUNCTION DOC]
        # Parametrized over datetime and timedelta nanosecond dtypes. A NaT scalar with
        # that dtype gives all-missing values of that dtype.
        obj = constructor(pd.NaT, dtype=dtype)
        assert np.all(obj.dtypes == dtype)
        assert np.all(obj.isna())

    def test_from_timedelta_scalar_preserves_nanos(self, constructor):
        # [FUNCTION DOC]
        # A one-nanosecond timedelta scalar with timedelta nanosecond dtype keeps the
        # nanosecond.
        td = Timedelta(1)

        obj = constructor(td, dtype="m8[ns]")
        assert get1(obj) == td

    def test_from_timestamp_scalar_preserves_nanos(self, constructor, fixed_now_ts):
        # [FUNCTION DOC]
        # A timestamp plus one nanosecond with datetime nanosecond dtype keeps the
        # nanosecond.
        ts = fixed_now_ts + Timedelta(1)

        obj = constructor(ts, dtype="M8[ns]")
        assert get1(obj) == ts

    def test_from_timedelta64_scalar_object(self, constructor):
        # [FUNCTION DOC]
        # A NumPy timedelta scalar with object dtype stays a NumPy timedelta.
        td = Timedelta(1)
        td64 = td.to_timedelta64()

        obj = constructor(td64, dtype=object)
        assert isinstance(get1(obj), np.timedelta64)

    @pytest.mark.parametrize("cls", [np.datetime64, np.timedelta64])
    def test_from_scalar_datetimelike_mismatched(self, constructor, cls):
        # [FUNCTION DOC]
        # Parametrized over NumPy datetime and timedelta classes. A NaT or concrete value
        # of one kind with the other kind's dtype raises a type error (cannot cast, or
        # invalid type for timedelta scalar, or not convertible to datetime).
        scalar = cls("NaT", "ns")
        dtype = {np.datetime64: "m8[ns]", np.timedelta64: "M8[ns]"}[cls]

        if cls is np.datetime64:
            msg1 = "Invalid type for timedelta scalar: <class 'numpy.datetime64'>"
        else:
            msg1 = "<class 'numpy.timedelta64'> is not convertible to datetime"
        msg = "|".join(["Cannot cast", msg1])

        with pytest.raises(TypeError, match=msg):
            constructor(scalar, dtype=dtype)

        scalar = cls(4, "ns")
        with pytest.raises(TypeError, match=msg):
            constructor(scalar, dtype=dtype)

    @pytest.mark.parametrize("cls", [datetime, np.datetime64])
    def test_from_out_of_bounds_ns_datetime(
        self, constructor, cls, request, box, frame_or_series
    ):
        # [FUNCTION DOC]
        # Parametrized over native datetime and NumPy day-resolution datetime. A year-9999
        # scalar, outside nanosecond bounds, infers microsecond (native) or second
        # (NumPy day input) resolution, element and column dtype matching the expected one.
        # scalar that won't fit in nanosecond dt64, but will fit in microsecond
        scalar = datetime(9999, 1, 1)
        exp_dtype = "M8[us]"  # pydatetime objects default to this reso

        if cls is np.datetime64:
            scalar = np.datetime64(scalar, "D")
            exp_dtype = "M8[s]"  # closest reso to input
        result = constructor(scalar)

        item = get1(result)
        dtype = tm.get_dtype(result)

        assert type(item) is Timestamp
        assert item.asm8.dtype == exp_dtype
        assert dtype == exp_dtype

    def test_out_of_s_bounds_datetime64(self, constructor):
        # [FUNCTION DOC]
        # A NumPy day-resolution datetime at the largest 64-bit value (out of second
        # bounds) stays a NumPy datetime in an object column.
        scalar = np.datetime64(np.iinfo(np.int64).max, "D")
        result = constructor(scalar)
        item = get1(result)
        assert type(item) is np.datetime64
        dtype = tm.get_dtype(result)
        assert dtype == object

    @pytest.mark.parametrize("cls", [timedelta, np.timedelta64])
    def test_from_out_of_bounds_ns_timedelta(
        self, constructor, cls, box, frame_or_series
    ):
        # [FUNCTION DOC]
        # Parametrized over native timedelta and NumPy day timedelta. A timedelta beyond
        # nanosecond bounds infers microsecond (native) or second (NumPy) resolution.
        scalar = datetime(9999, 1, 1) - datetime(1970, 1, 1)
        exp_dtype = "m8[us]"  # smallest reso that fits
        if cls is np.timedelta64:
            scalar = np.timedelta64(scalar, "D")
            exp_dtype = "m8[s]"  # closest reso to input
        result = constructor(scalar)

        item = get1(result)
        dtype = tm.get_dtype(result)

        assert type(item) is Timedelta
        assert item.asm8.dtype == exp_dtype
        assert dtype == exp_dtype

    @pytest.mark.parametrize("cls", [np.datetime64, np.timedelta64])
    def test_out_of_s_bounds_timedelta64(self, constructor, cls):
        # [FUNCTION DOC]
        # Parametrized over NumPy datetime and timedelta. A day-unit scalar at the largest
        # 64-bit value stays as the NumPy scalar in an object column.
        scalar = cls(np.iinfo(np.int64).max, "D")
        result = constructor(scalar)
        item = get1(result)
        assert type(item) is cls
        dtype = tm.get_dtype(result)
        assert dtype == object

    def test_tzaware_data_tznaive_dtype(self, constructor, box, frame_or_series):
        # [FUNCTION DOC]
        # A zone-aware timestamp with a naive datetime dtype raises a type error about
        # unboxing for scalar wrapping none or a frame dictionary, otherwise a value error
        # telling the caller to localize to none.
        tz = "US/Eastern"
        ts = Timestamp("2019", tz=tz)

        if box is None or (frame_or_series is DataFrame and box is dict):
            msg = "Cannot unbox tzaware Timestamp to tznaive dtype"
            err = TypeError
        else:
            msg = (
                "Cannot convert timezone-aware data to timezone-naive dtype. "
                r"Use pd.Series\(values\).dt.tz_localize\(None\) instead."
            )
            err = ValueError

        with pytest.raises(err, match=msg):
            constructor(ts, dtype="M8[ns]")


# TODO: better location for this test?
class TestAllowNonNano:
    # [CLASS DOC]
    # Test group, with an unresolved question about the best location, checking that
    # non-nanosecond datetime and timedelta arrays keep their unit when wrapped in
    # indexes, Series and frames. An original comment says that until 2.0 NumPy
    # arrays were not preserved, but datetime and timedelta arrays were.
    # Until 2.0, we do not preserve non-nano dt64/td64 when passed as ndarray,
    #  but do preserve it when passed as DTA/TDA

    @pytest.fixture(params=[True, False])
    def as_td(self, request):
        # [FUNCTION DOC]
        # HELPER: fixture parametrized over boolean; chooses timedelta (true) or datetime
        # (false) arrays.
        return request.param

    @pytest.fixture
    def arr(self, as_td):
        # [FUNCTION DOC]
        # HELPER: fixture building a five-element second-resolution datetime array, or
        # timedeltas relative to the first value, using the private array constructor.
        values = np.arange(5).astype(np.int64).view("M8[s]")
        if as_td:
            values = values - values[0]
            return TimedeltaArray._simple_new(values, dtype=values.dtype)
        else:
            return DatetimeArray._simple_new(values, dtype=values.dtype)

    def test_index_allow_non_nano(self, arr):
        # [FUNCTION DOC]
        # An Index built from the array keeps the array's dtype.
        idx = Index(arr)
        assert idx.dtype == arr.dtype

    def test_dti_tdi_allow_non_nano(self, arr, as_td):
        # [FUNCTION DOC]
        # A timedelta or datetime index built from the array keeps its dtype.
        if as_td:
            idx = pd.TimedeltaIndex(arr)
        else:
            idx = DatetimeIndex(arr)
        assert idx.dtype == arr.dtype

    def test_series_allow_non_nano(self, arr):
        # [FUNCTION DOC]
        # A Series built from the array keeps its dtype.
        ser = Series(arr)
        assert ser.dtype == arr.dtype

    def test_frame_allow_non_nano(self, arr):
        # [FUNCTION DOC]
        # A frame built from the array keeps its dtype in the single column.
        df = DataFrame(arr)
        assert df.dtypes[0] == arr.dtype

    def test_frame_from_dict_allow_non_nano(self, arr):
        # [FUNCTION DOC]
        # A frame built from a dictionary containing the array keeps its dtype.
        df = DataFrame({0: arr})
        assert df.dtypes[0] == arr.dtype
