# [MODULE DOC]
# Tests of the low-level lookup engines (compiled index engine classes imported from the internal index module) that back index label lookup: datetime and timedelta engines must refuse non-matching scalar types, and numeric and object engines must correctly report monotonicity, uniqueness and location results. Defines a parametrized fixture over the numeric engine classes and their numpy dtypes.

import re

import numpy as np
import pytest

from pandas._libs import index as libindex

import pandas as pd


@pytest.fixture(
    params=[
        (libindex.Int64Engine, np.int64),
        (libindex.Int32Engine, np.int32),
        (libindex.Int16Engine, np.int16),
        (libindex.Int8Engine, np.int8),
        (libindex.UInt64Engine, np.uint64),
        (libindex.UInt32Engine, np.uint32),
        (libindex.UInt16Engine, np.uint16),
        (libindex.UInt8Engine, np.uint8),
        (libindex.Float64Engine, np.float64),
        (libindex.Float32Engine, np.float32),
    ],
    ids=lambda x: x[0].__name__,
)
def numeric_indexing_engine_type_and_dtype(request):
    # [FUNCTION DOC]
    # HELPER: parametrized fixture yielding pairs of a numeric engine class (signed and unsigned integers of 8 to 64 bits, 32 and 64 bit floats) and its matching numpy dtype; test ids come from the engine class name.
    return request.param


class TestDatetimeEngine:
    # [CLASS DOC]
    # Tests for the datetime engine.
    @pytest.mark.parametrize(
        "scalar",
        [
            pd.Timedelta(pd.Timestamp("2016-01-01").asm8.view("m8[ns]")),
            pd.Timestamp("2016-01-01")._value,
            pd.Timestamp("2016-01-01").to_pydatetime(),
            pd.Timestamp("2016-01-01").to_datetime64(),
        ],
    )
    def test_not_contains_requires_timestamp(self, scalar):
        # [FUNCTION DOC]
        # Parametrized over scalars that are not timestamps (a timedelta, an integer, a standard-library datetime, a numpy datetime). For five datetime indexes (monotonic unique, non-monotonic, non-unique, a large monotonic one of two million, and a large non-unique one over the size threshold) membership in the engine raises a type error and location lookup raises a key error, both mentioning the scalar's text or repr.
        dti1 = pd.date_range("2016-01-01", periods=3)
        dti2 = dti1.insert(1, pd.NaT)  # non-monotonic
        dti3 = dti1.insert(3, dti1[0])  # non-unique
        dti4 = pd.date_range("2016-01-01", freq="ns", periods=2_000_000)
        dti5 = dti4.insert(0, dti4[0])  # over size threshold, not unique

        msg = "|".join([re.escape(str(scalar)), re.escape(repr(scalar))])
        for dti in [dti1, dti2, dti3, dti4, dti5]:
            with pytest.raises(TypeError, match=msg):
                scalar in dti._engine

            with pytest.raises(KeyError, match=msg):
                dti._engine.get_loc(scalar)


class TestTimedeltaEngine:
    # [CLASS DOC]
    # Tests for the timedelta engine.
    @pytest.mark.parametrize(
        "scalar",
        [
            pd.Timestamp(pd.Timedelta(days=42).asm8.view("datetime64[ns]")),
            pd.Timedelta(days=42)._value,
            pd.Timedelta(days=42).to_pytimedelta(),
            pd.Timedelta(days=42).to_timedelta64(),
        ],
    )
    def test_not_contains_requires_timedelta(self, scalar):
        # [FUNCTION DOC]
        # Mirror of the datetime test with scalars that are not timedeltas (a timestamp, an integer, a standard-library timedelta, a numpy timedelta); same five index shapes; expects a type error on membership and key error on location.
        tdi1 = pd.timedelta_range("42 days", freq="9h", periods=1234)
        tdi2 = tdi1.insert(1, pd.NaT)  # non-monotonic
        tdi3 = tdi1.insert(3, tdi1[0])  # non-unique
        tdi4 = pd.timedelta_range("42 days", freq="ns", periods=2_000_000)
        tdi5 = tdi4.insert(0, tdi4[0])  # over size threshold, not unique

        msg = "|".join([re.escape(str(scalar)), re.escape(repr(scalar))])
        for tdi in [tdi1, tdi2, tdi3, tdi4, tdi5]:
            with pytest.raises(TypeError, match=msg):
                scalar in tdi._engine

            with pytest.raises(KeyError, match=msg):
                tdi._engine.get_loc(scalar)


class TestNumericEngine:
    # [CLASS DOC]
    # Tests for the numeric engines, parametrized via the fixture over each engine class and dtype.
    def test_is_monotonic(self, numeric_indexing_engine_type_and_dtype):
        # [FUNCTION DOC]
        # With an array of 1000 each of ones, twos and threes: the engine reports increasing and not decreasing; for the reversed array the opposite; for an array with a repeat of the lowest value at the end (reversed) neither.
        engine_type, dtype = numeric_indexing_engine_type_and_dtype
        num = 1000
        arr = np.array([1] * num + [2] * num + [3] * num, dtype=dtype)

        # monotonic increasing
        engine = engine_type(arr)
        assert engine.is_monotonic_increasing is True
        assert engine.is_monotonic_decreasing is False

        # monotonic decreasing
        engine = engine_type(arr[::-1])
        assert engine.is_monotonic_increasing is False
        assert engine.is_monotonic_decreasing is True

        # neither monotonic increasing or decreasing
        arr = np.array([1] * num + [2] * num + [1] * num, dtype=dtype)
        engine = engine_type(arr[::-1])
        assert engine.is_monotonic_increasing is False
        assert engine.is_monotonic_decreasing is False

    def test_is_unique(self, numeric_indexing_engine_type_and_dtype):
        # [FUNCTION DOC]
        # An array of distinct values is unique; an array with a repeated value is not.
        engine_type, dtype = numeric_indexing_engine_type_and_dtype

        # unique
        arr = np.array([1, 3, 2], dtype=dtype)
        engine = engine_type(arr)
        assert engine.is_unique is True

        # not unique
        arr = np.array([1, 2, 1], dtype=dtype)
        engine = engine_type(arr)
        assert engine.is_unique is False

    def test_get_loc(self, numeric_indexing_engine_type_and_dtype):
        # [FUNCTION DOC]
        # Location lookup returns an integer for a unique array; a slice (1000 to 2000) for a monotonic array with repeated runs; a boolean mask for a non-monotonic array with repeated values.
        engine_type, dtype = numeric_indexing_engine_type_and_dtype

        # unique
        arr = np.array([1, 2, 3], dtype=dtype)
        engine = engine_type(arr)
        assert engine.get_loc(2) == 1

        # monotonic
        num = 1000
        arr = np.array([1] * num + [2] * num + [3] * num, dtype=dtype)
        engine = engine_type(arr)
        assert engine.get_loc(2) == slice(1000, 2000)

        # not monotonic
        arr = np.array([1, 2, 3] * num, dtype=dtype)
        engine = engine_type(arr)
        expected = np.array([False, True, False] * num, dtype=bool)
        result = engine.get_loc(2)
        assert (result == expected).all()


class TestObjectEngine:
    # [CLASS DOC]
    # Tests for the object engine. Class attributes define the engine class, object dtype and the values a, b, c.
    engine_type = libindex.ObjectEngine
    dtype = np.object_
    values = list("abc")

    def test_is_monotonic(self):
        # [FUNCTION DOC]
        # Same monotonic checks as for numeric engines but with strings.
        num = 1000
        arr = np.array(["a"] * num + ["a"] * num + ["c"] * num, dtype=self.dtype)

        # monotonic increasing
        engine = self.engine_type(arr)
        assert engine.is_monotonic_increasing is True
        assert engine.is_monotonic_decreasing is False

        # monotonic decreasing
        engine = self.engine_type(arr[::-1])
        assert engine.is_monotonic_increasing is False
        assert engine.is_monotonic_decreasing is True

        # neither monotonic increasing or decreasing
        arr = np.array(["a"] * num + ["b"] * num + ["a"] * num, dtype=self.dtype)
        engine = self.engine_type(arr[::-1])
        assert engine.is_monotonic_increasing is False
        assert engine.is_monotonic_decreasing is False

    def test_is_unique(self):
        # [FUNCTION DOC]
        # Distinct strings are unique; a repeated string makes it non-unique.
        # unique
        arr = np.array(self.values, dtype=self.dtype)
        engine = self.engine_type(arr)
        assert engine.is_unique is True

        # not unique
        arr = np.array(["a", "b", "a"], dtype=self.dtype)
        engine = self.engine_type(arr)
        assert engine.is_unique is False

    def test_get_loc(self):
        # [FUNCTION DOC]
        # Same three location cases as numeric engines (integer, slice, boolean mask) with strings.
        # unique
        arr = np.array(self.values, dtype=self.dtype)
        engine = self.engine_type(arr)
        assert engine.get_loc("b") == 1

        # monotonic
        num = 1000
        arr = np.array(["a"] * num + ["b"] * num + ["c"] * num, dtype=self.dtype)
        engine = self.engine_type(arr)
        assert engine.get_loc("b") == slice(1000, 2000)

        # not monotonic
        arr = np.array(self.values * num, dtype=self.dtype)
        engine = self.engine_type(arr)
        expected = np.array([False, True, False] * num, dtype=bool)
        result = engine.get_loc("b")
        assert (result == expected).all()
