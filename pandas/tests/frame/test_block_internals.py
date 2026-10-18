# [MODULE DOC]
# Tests for the tabular container that depend on the internal block-manager
# storage: frequency invalidation on in-place set, casting from a manager,
# consolidation of blocks, read-only values, constructor type inference,
# pickling, extension-array-wrapping blocks, and a multithreaded read stress
# test. Imports private pieces (the numpy-backed block class, test decorators).
# Defines a module-level data helper used by the threaded test. Uses fixtures from
# the test-suite conftest; no import-time side effects.
from datetime import (
    datetime,
    timedelta,
)
from io import StringIO
import itertools
from textwrap import dedent

import numpy as np
import pytest

from pandas.errors import Pandas4Warning
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Series,
    Timestamp,
    date_range,
    option_context,
)
import pandas._testing as tm
from pandas.core.internals.blocks import NumpyBlock

# Segregated collection of methods that require the BlockManager internal data
# structure


class TestDataFrameBlockInternals:
    # [CLASS DOC]
    # Test group for behaviors tied to the internal block manager. Stateless.
    def test_setitem_invalidates_datetime_index_freq(self):
        # [FUNCTION DOC]
        # GH#24096: setting an element in place on a timezone-aware datetime column
        # leaves the column's frequency unset, and the original index used to build the
        # frame keeps both its frequency and its element (not altered in place).
        # GH#24096 altering a datetime64tz column inplace invalidates the
        #  `freq` attribute on the underlying DatetimeIndex

        dti = date_range("20130101", periods=3, tz="US/Eastern")
        ts = dti[1]

        df = DataFrame({"B": dti})
        assert df["B"]._values.freq is None

        df.iloc[1, 0] = pd.NaT
        assert df["B"]._values.freq is None

        # check that the DatetimeIndex was not altered in place
        assert dti.freq == "D"
        assert dti[1] == ts

    def test_cast_internals(self, float_frame):
        # [FUNCTION DOC]
        # Constructing a frame from another frame's internal manager with an integer
        # dtype (default width and 32-bit) emits a deprecation warning (Pandas4Warning)
        # about passing a manager, and results equal casting the float fixture's column
        # mapping to that dtype.
        msg = "Passing a BlockManager to DataFrame"
        with tm.assert_produces_warning(
            Pandas4Warning, match=msg, check_stacklevel=False
        ):
            casted = DataFrame(float_frame._mgr, dtype=int)
        expected = DataFrame(float_frame._series, dtype=int)
        tm.assert_frame_equal(casted, expected)

        with tm.assert_produces_warning(
            Pandas4Warning, match=msg, check_stacklevel=False
        ):
            casted = DataFrame(float_frame._mgr, dtype=np.int32)
        expected = DataFrame(float_frame._series, dtype=np.int32)
        tm.assert_frame_equal(casted, expected)

    def test_consolidate(self, float_frame):
        # [FUNCTION DOC]
        # Consolidation: after adding a float column to the float fixture, a
        # consolidated copy has one block; consolidating again returns a new (not
        # identical) but equal frame; adding another column makes three blocks; the
        # in-place consolidation returns nothing and leaves one block.
        float_frame["E"] = 7.0
        consolidated = float_frame._consolidate()
        assert len(consolidated._mgr.blocks) == 1

        # Ensure copy, do I want this?
        recons = consolidated._consolidate()
        assert recons is not consolidated
        tm.assert_frame_equal(recons, consolidated)

        float_frame["F"] = 8.0
        assert len(float_frame._mgr.blocks) == 3

        return_value = float_frame._consolidate_inplace()
        assert return_value is None
        assert len(float_frame._mgr.blocks) == 1

    def test_consolidate_inplace(self, float_frame):
        # [FUNCTION DOC]
        # Adds a column per letter from A to Y (the range stops before Z) holding a
        # string to the float fixture, to trigger automatic in-place consolidation.
        # There are no assertions; passing means no error.
        # triggers in-place consolidation
        for letter in range(ord("A"), ord("Z")):
            float_frame[chr(letter)] = chr(letter)

    def test_modify_values(self, float_frame):
        # [FUNCTION DOC]
        # The values of a frame are read-only: assigning to a row of the values array
        # raises a value error mentioning read-only, and the row remains not equal to the
        # attempted value.
        with pytest.raises(ValueError, match="read-only"):
            float_frame.values[5] = 5
        assert (float_frame.values[5] != 5).all()

    def test_boolean_set_uncons(self, float_frame):
        # [FUNCTION DOC]
        # Boolean-mask assignment on a non-consolidated frame (after adding a column)
        # sets every element above one to two, matching the same operation on a copy of
        # the raw values.
        float_frame["E"] = 7.0

        expected = float_frame.values.copy()
        expected[expected > 1] = 2

        float_frame[float_frame > 1] = 2
        tm.assert_almost_equal(expected, float_frame.values)

    def test_constructor_with_convert(self):
        # [FUNCTION DOC]
        # Mostly exercises the object-conversion routine used by the constructor
        # (issue 2845): single-column frames from lists of mixed scalars yield: int64
        # for the largest signed 64-bit integer, unsigned 64-bit for one beyond it,
        # object for datetime with bool, float with missing as NaN, complex128 for
        # complex with int/float/None, object for complex with bool, float mixed with
        # bool/None, and datetime mixes. Each case compares the column to an expected
        # Series of the stated dtype.
        # this is actually mostly a test of lib.maybe_convert_objects
        # #2845
        df = DataFrame({"A": [2**63 - 1]})
        result = df["A"]
        expected = Series(np.asarray([2**63 - 1], np.int64), name="A")
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [2**63]})
        result = df["A"]
        expected = Series(np.asarray([2**63], np.uint64), name="A")
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [datetime(2005, 1, 1), True]})
        result = df["A"]
        expected = Series(
            np.asarray([datetime(2005, 1, 1), True], np.object_), name="A"
        )
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [None, 1]})
        result = df["A"]
        expected = Series(np.asarray([np.nan, 1], np.float64), name="A")
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [1.0, 2]})
        result = df["A"]
        expected = Series(np.asarray([1.0, 2], np.float64), name="A")
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [1.0 + 2.0j, 3]})
        result = df["A"]
        expected = Series(np.asarray([1.0 + 2.0j, 3], np.complex128), name="A")
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [1.0 + 2.0j, 3.0]})
        result = df["A"]
        expected = Series(np.asarray([1.0 + 2.0j, 3.0], np.complex128), name="A")
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [1.0 + 2.0j, True]})
        result = df["A"]
        expected = Series(np.asarray([1.0 + 2.0j, True], np.object_), name="A")
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [1.0, None]})
        result = df["A"]
        expected = Series(np.asarray([1.0, np.nan], np.float64), name="A")
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [1.0 + 2.0j, None]})
        result = df["A"]
        expected = Series(np.asarray([1.0 + 2.0j, np.nan], np.complex128), name="A")
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [2.0, 1, True, None]})
        result = df["A"]
        expected = Series(np.asarray([2.0, 1, True, None], np.object_), name="A")
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [2.0, 1, datetime(2006, 1, 1), None]})
        result = df["A"]
        expected = Series(
            np.asarray([2.0, 1, datetime(2006, 1, 1), None], np.object_), name="A"
        )
        tm.assert_series_equal(result, expected)

    def test_construction_with_mixed(self, float_string_frame, using_infer_string):
        # [FUNCTION DOC]
        # Mixed-type frame: adding datetime and timedelta columns to the float/string
        # fixture gives microsecond-resolution datetime and timedelta dtypes, and the
        # dtypes Series is four float64, then string (object, or the new string dtype
        # with NaN when the inferred-string option is active), then the two time dtypes.
        # mixed-type frames
        float_string_frame["datetime"] = datetime.now()
        float_string_frame["timedelta"] = timedelta(days=1, seconds=1)
        assert float_string_frame["datetime"].dtype == "M8[us]"
        assert float_string_frame["timedelta"].dtype == "m8[us]"
        result = float_string_frame.dtypes
        expected = Series(
            [np.dtype("float64")] * 4
            + [
                np.dtype("object")
                if not using_infer_string
                else pd.StringDtype(na_value=np.nan),
                np.dtype("datetime64[us]"),
                np.dtype("timedelta64[us]"),
            ],
            index=[*list("ABCD"), "foo", "datetime", "timedelta"],
        )
        tm.assert_series_equal(result, expected)

    def test_construction_with_conversions(self):
        # [FUNCTION DOC]
        # As of 2.0, constructing from a NumPy second-resolution timedelta array keeps
        # that resolution (values equal the original array). Also, a frame with a scalar
        # second-resolution timestamp and a second-resolution datetime column has
        # second-resolution dtypes, and a frame built from a NumPy day-resolution
        # datetime scalar and array equals that expected frame (dtypes converted to
        # seconds).
        # KNOWN ISSUE: the first expected frame (built from a timedelta range) is never compared to anything, since the variable is reassigned before use; dead assignment.
        # convert from a numpy array of non-ns timedelta64; as of 2.0 this does
        #  *not* convert
        arr = np.array([1, 2, 3], dtype="timedelta64[s]")
        df = DataFrame({"A": arr})
        expected = DataFrame(
            {"A": pd.timedelta_range("00:00:01", periods=3, freq="s")}, index=range(3)
        )
        tm.assert_numpy_array_equal(df["A"].to_numpy(), arr)

        expected = DataFrame(
            {
                "dt1": Timestamp("20130101").as_unit("s"),
                "dt2": date_range("20130101", periods=3).astype("M8[s]"),
                # 'dt3' : date_range('20130101 00:00:01',periods=3,freq='s'),
                # FIXME: don't leave commented-out
            },
            index=range(3),
        )
        assert expected.dtypes["dt1"] == "M8[s]"
        assert expected.dtypes["dt2"] == "M8[s]"

        dt1 = np.datetime64("2013-01-01")
        dt2 = np.array(
            ["2013-01-01", "2013-01-02", "2013-01-03"], dtype="datetime64[D]"
        )
        df = DataFrame({"dt1": dt1, "dt2": dt2})

        # df['dt3'] = np.array(['2013-01-01 00:00:01','2013-01-01
        # 00:00:02','2013-01-01 00:00:03'],dtype='datetime64[s]')
        # FIXME: don't leave commented-out

        tm.assert_frame_equal(df, expected)

    def test_constructor_compound_dtypes(self):
        # [FUNCTION DOC]
        # GH 5191: passing a compound (list of name-type pairs) dtype to the constructor
        # raises not-implemented; plain integer or float dtype with this mixed data
        # raises a type error; a nanosecond datetime dtype raises a value error about an
        # unparseable datetime string for the string column.
        # GH 5191
        # compound dtypes should raise not-implementederror

        def f(dtype):
            # [FUNCTION DOC]
            # HELPER: for the compound dtype test. Builds a frame of nine identical rows
            # (a datetime, a string, an integer) with columns A, B, C using the given dtype
            # argument and returns it; errors propagate to the caller.
            data = list(itertools.repeat((datetime(2001, 1, 1), "aa", 20), 9))
            return DataFrame(data=data, columns=["A", "B", "C"], dtype=dtype)

        msg = "compound dtypes are not implemented in the DataFrame constructor"
        with pytest.raises(NotImplementedError, match=msg):
            f([("A", "datetime64[h]"), ("B", "str"), ("C", "int32")])

        # pre-2.0 these used to work (though results may be unexpected)
        with pytest.raises(TypeError, match="argument must be"):
            f("int64")
        with pytest.raises(TypeError, match="argument must be"):
            f("float64")

        # 10822
        msg = "^Unknown datetime string format, unable to parse: aa$"
        with pytest.raises(ValueError, match=msg):
            f("M8[ns]")

    def test_pickle_float_string_frame(self, float_string_frame, temp_file):
        # [FUNCTION DOC]
        # Pickle round trip of the mixed float/string fixture (via a temporary file
        # fixture) gives an equal frame; then touches the dimension count of the internal
        # manager as a regression check (no assertion).
        unpickled = tm.round_trip_pickle(float_string_frame, temp_file)
        tm.assert_frame_equal(float_string_frame, unpickled)

        # buglet
        float_string_frame._mgr.ndim

    def test_pickle_empty(self, temp_file):
        # [FUNCTION DOC]
        # Pickle round trip of an empty frame can be repr'd without error.
        empty_frame = DataFrame()
        unpickled = tm.round_trip_pickle(empty_frame, temp_file)
        repr(unpickled)

    def test_pickle_empty_tz_frame(self, timezone_frame, temp_file):
        # [FUNCTION DOC]
        # Pickle round trip of the timezone-aware fixture frame gives an equal frame.
        unpickled = tm.round_trip_pickle(timezone_frame, temp_file)
        tm.assert_frame_equal(timezone_frame, unpickled)

    def test_consolidate_datetime64(self):
        # [FUNCTION DOC]
        # Regression for a NumPy stacking bug: two datetime columns are re-indexed by
        # their own values, localized to US/Eastern and converted to UTC, and are then
        # assigned back as columns; the columns as datetime indexes equal the converted
        # indexes.
        # numpy vstack bug

        df = DataFrame(
            {
                "starting": pd.to_datetime(
                    [
                        "2012-06-21 00:00",
                        "2012-06-23 07:00",
                        "2012-06-23 16:30",
                        "2012-06-25 08:00",
                        "2012-06-26 12:00",
                    ]
                ),
                "ending": pd.to_datetime(
                    [
                        "2012-06-23 07:00",
                        "2012-06-23 16:30",
                        "2012-06-25 08:00",
                        "2012-06-26 12:00",
                        "2012-06-27 08:00",
                    ]
                ),
                "measure": [77, 65, 77, 0, 77],
            }
        )

        ser_starting = df.starting
        ser_starting.index = ser_starting.values
        ser_starting = ser_starting.tz_localize("US/Eastern")
        ser_starting = ser_starting.tz_convert("UTC")
        ser_starting.index.name = "starting"

        ser_ending = df.ending
        ser_ending.index = ser_ending.values
        ser_ending = ser_ending.tz_localize("US/Eastern")
        ser_ending = ser_ending.tz_convert("UTC")
        ser_ending.index.name = "ending"

        df.starting = ser_starting.index
        df.ending = ser_ending.index

        tm.assert_index_equal(pd.DatetimeIndex(df.starting), ser_starting.index)
        tm.assert_index_equal(pd.DatetimeIndex(df.ending), ser_ending.index)

    def test_is_mixed_type(self, float_frame, float_string_frame):
        # [FUNCTION DOC]
        # The private mixed-type flag is false for the float fixture and true for the
        # float/string fixture.
        assert not float_frame._is_mixed_type
        assert float_string_frame._is_mixed_type

    def test_stale_cached_series_bug_473(self):
        # [FUNCTION DOC]
        # Bug 473 (stale cached series): with chained-assignment checking disabled,
        # convert a column to object, chained-assign a missing value into another column
        # (which must raise the chained assignment error), then repr and sum; the
        # assigned element must remain not missing (the chained assignment had no effect).
        # this is chained, but ok
        with option_context("chained_assignment", None):
            Y = DataFrame(
                np.random.default_rng(2).random((4, 4)),
                index=("a", "b", "c", "d"),
                columns=("e", "f", "g", "h"),
            )
            repr(Y)
            Y["e"] = Y["e"].astype("object")
            with tm.raises_chained_assignment_error():
                Y["g"]["c"] = np.nan
            repr(Y)
            Y.sum()
            Y["g"].sum()
            assert not pd.isna(Y["g"]["c"])

    def test_strange_column_corruption_issue(self, performance_warning):
        # [FUNCTION DOC]
        # Builds a two-row frame, adds a missing-valued column, then in a loop adds 100
        # columns and sets cell values by label while expecting a performance warning
        # (extra warnings allowed); afterwards the selection of missing values in column
        # 100 is empty. The author notes the relation to internals is unclear.
        # TODO(wesm): Unclear how exactly this is related to internal matters
        df = DataFrame(index=[0, 1])
        df[0] = np.nan
        wasCol = {}

        with tm.assert_produces_warning(
            performance_warning, raise_on_extra_warnings=False
        ):
            for i, dt in enumerate(df.index):
                for col in range(100, 200):
                    if col not in wasCol:
                        wasCol[col] = 1
                        df[col] = np.nan
                    df.loc[dt, col] = i

        myid = 100

        first = len(df.loc[pd.isna(df[myid]), [myid]])
        second = len(df.loc[pd.isna(df[myid]), [myid]])
        assert first == second == 0

    def test_constructor_no_pandas_array(self):
        # [FUNCTION DOC]
        # Constructing a frame from a numpy-extension array unwraps it: the result
        # equals a frame from a plain list, and its first block is of the numpy-backed
        # block type and numeric.
        # Ensure that NumpyExtensionArray isn't allowed inside Series
        # See https://github.com/pandas-dev/pandas/issues/23995 for more.
        arr = Series([1, 2, 3]).array
        result = DataFrame({"A": arr})
        expected = DataFrame({"A": [1, 2, 3]})
        tm.assert_frame_equal(result, expected)
        assert isinstance(result._mgr.blocks[0], NumpyBlock)
        assert result._mgr.blocks[0].is_numeric

    def test_add_column_with_pandas_array(self):
        # [FUNCTION DOC]
        # GH 26390: assigning a numpy-extension array of objects as a new column gives
        # the same block type (numpy-backed, object) as constructing a frame with that
        # array, and the two frames are equal.
        # GH 26390
        df = DataFrame({"a": [1, 2, 3, 4], "b": ["a", "b", "c", "d"]})
        df["c"] = pd.arrays.NumpyExtensionArray(np.array([1, 2, None, 3], dtype=object))
        df2 = DataFrame(
            {
                "a": [1, 2, 3, 4],
                "b": ["a", "b", "c", "d"],
                "c": pd.arrays.NumpyExtensionArray(
                    np.array([1, 2, None, 3], dtype=object)
                ),
            }
        )
        assert type(df["c"]._mgr.blocks[0]) == NumpyBlock
        assert df["c"]._mgr.blocks[0].is_object
        assert type(df2["c"]._mgr.blocks[0]) == NumpyBlock
        assert df2["c"]._mgr.blocks[0].is_object
        tm.assert_frame_equal(df, df2)


def test_update_inplace_sets_valid_block_values():
    # [FUNCTION DOC]
    # GH 33457: an in-place fill on a column selected from a categorical frame (a
    # chained assignment which must raise the chained assignment error) must not
    # leave a Series in the block; the first block's values stay a categorical array.
    # https://github.com/pandas-dev/pandas/issues/33457
    df = DataFrame({"a": Series([1, 2, None], dtype="category")})

    # inplace update of a single column
    with tm.raises_chained_assignment_error():
        df["a"].fillna(1, inplace=True)

    # check we haven't put a Series into any block.values
    assert isinstance(df._mgr.blocks[0].values, Categorical)


def get_longley_data():
    # [FUNCTION DOC]
    # HELPER: data builder for the multithreaded test. Parses an embedded CSV of the
    # 16-row Longley economic dataset (columns for deflator, GNP, unemployment, armed
    # forces, population, year), keeps all columns except the observation number,
    # and returns them as a float frame. No parameters; no side effects.
    # From statsmodels.datasets.longley
    # This specific dataset seems to trigger races in Pandas 3.0.0 more readily
    # than data frames used elsewhere in the tests
    longley_csv = StringIO(
        dedent(
            """"Obs","GNPDEFL","GNP","UNEMP","ARMED","POP","YEAR"
            1,83,234289,2356,1590,107608,1947
            2,88.5,259426,2325,1456,108632,1948
            3,88.2,258054,3682,1616,109773,1949
            4,89.5,284599,3351,1650,110929,1950
            5,96.2,328975,2099,3099,112075,1951
            6,98.1,346999,1932,3594,113270,1952
            7,99,365385,1870,3547,115094,1953
            8,100,363112,3578,3350,116219,1954
            9,101.2,397469,2904,3048,117388,1955
            10,104.6,419180,2822,2857,118734,1956
            11,108.4,442769,2936,2798,120445,1957
            12,110.8,444546,4681,2637,121950,1958
            13,112.6,482704,3813,2552,123366,1959
            14,114.2,502601,3931,2514,125368,1960
            15,115.7,518173,4806,2572,127852,1961
            16,116.9,554894,4007,2827,130081,1962
            """
        )
    )

    return pd.read_csv(longley_csv).iloc[:, [1, 2, 3, 4, 5, 6]].astype(float)


# See gh-63685, comparisons and copying led to races in statsmodels tests
#
# This test spawns a thread pool, so it shouldn't run under xdist.
# It generates warnings, so it needs warnings to be thread-safe as well
@td.skip_if_thread_unsafe_warnings
@pytest.mark.single_cpu
def test_multithreaded_reading():
    # [FUNCTION DOC]
    # See gh-63685: stress test for races in comparisons and copies. Skipped when
    # warnings are not thread-safe, and marked as single-CPU because it spawns a
    # thread pool. First runs eight workers concurrently, each waiting at a barrier
    # and comparing a copy of the shared data with a shifted-and-unshifted version;
    # then eight workers each build a constant-trend column, apply a constancy
    # check column-wise, concatenate and compare a frame to itself. Passing means no
    # exceptions in any worker.
    def numpy_assert(data, b):
        # [FUNCTION DOC]
        # HELPER: for the multithreaded test. Worker run in each thread: waits for the
        # barrier, then asserts approximate equality between the data plus one minus one
        # and a copy of the data.
        b.wait()
        tm.assert_almost_equal((data + 1) - 1, data.copy())

    tm.run_multithreaded(
        numpy_assert, max_workers=8, arguments=(get_longley_data(),), pass_barrier=True
    )

    def safe_is_const(s):
        # [FUNCTION DOC]
        # HELPER: for the multithreaded test. Returns true when the given column is
        # constant and not all zero (zero range and some nonzero element), false on any
        # exception.
        try:
            return np.ptp(s) == 0.0 and np.any(s != 0.0)
        except Exception:
            return False

    def concat(data, b):
        # [FUNCTION DOC]
        # HELPER: for the multithreaded test. Worker run in each thread: waits at the
        # barrier, copies the data, builds a flipped Vandermonde-style column of ones,
        # applies the constancy check along columns (result ignored), prepends the
        # column as a constant to the data and concatenates along columns, then asserts
        # the resulting frame equals itself.
        b.wait()
        x = data.copy()
        nobs = len(x)
        trendarr = np.fliplr(np.vander(np.arange(1, nobs + 1, dtype=np.float64), 1))
        x.apply(safe_is_const, 0)
        trendarr = DataFrame(trendarr, index=x.index, columns=["const"])
        x = [trendarr, x]
        x = pd.concat(x[::1], axis=1)
        tm.assert_frame_equal(x, x)

    tm.run_multithreaded(
        concat, max_workers=8, arguments=(get_longley_data(),), pass_barrier=True
    )
