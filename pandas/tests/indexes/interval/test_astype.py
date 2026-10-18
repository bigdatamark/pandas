# [MODULE DOC]
# Tests for type conversion (astype) of IntervalIndex. A shared base class holds
# checks that apply to any subtype; subclasses add subtype-specific checks and
# provide the index fixture over a list of indexes for integer, float and
# datetime-like subtypes.
# Pure tests module; the index lists are built at import time.

import re

import numpy as np
import pytest

from pandas.errors import Pandas4Warning

from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    IntervalDtype,
)

from pandas import (
    CategoricalIndex,
    Index,
    IntervalIndex,
    NaT,
    Timedelta,
    Timestamp,
    interval_range,
)
import pandas._testing as tm


class AstypeTests:
    # [CLASS DOC]
    # Mixin of conversion tests shared by every subtype class. Subclasses must
    # provide a fixture named index; this class itself is not collected as tests.
    """Tests common to IntervalIndex with any subtype"""

    def test_astype_idempotent(self, index):
        # [FUNCTION DOC]
        # Verifies converting to the generic interval dtype or to the index's own
        # dtype leaves the index unchanged.
        result = index.astype("interval")
        tm.assert_index_equal(result, index)

        result = index.astype(index.dtype)
        tm.assert_index_equal(result, index)

    def test_astype_object(self, index):
        # [FUNCTION DOC]
        # Verifies conversion to object gives an object Index of the interval values,
        # which does not compare equal to the original.
        result = index.astype(object)
        expected = Index(index.values, dtype="object")
        tm.assert_index_equal(result, expected)
        assert not result.equals(index)

    def test_astype_category(self, index):
        # [FUNCTION DOC]
        # Verifies conversion to category (by name or default categorical dtype) equals
        # a categorical index of the values; with custom categories and ordered flag a
        # future-change warning about values outside the categories is expected, and the
        # result equals the equivalently constructed categorical index.
        result = index.astype("category")
        expected = CategoricalIndex(index.values)
        tm.assert_index_equal(result, expected)

        result = index.astype(CategoricalDtype())
        tm.assert_index_equal(result, expected)

        # non-default params
        categories = index.dropna().unique().values[:-1]
        dtype = CategoricalDtype(categories=categories, ordered=True)
        msg = "Constructing a Categorical with a dtype and values containing"
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            result = index.astype(dtype)
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            expected = CategoricalIndex(
                index.values, categories=categories, ordered=True
            )
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "dtype",
        [
            "int64",
            "uint64",
            "float64",
            "complex128",
            "period[M]",
            "timedelta64",
            "timedelta64[ns]",
            "datetime64",
            "datetime64[ns]",
            "datetime64[ns, US/Eastern]",
        ],
    )
    def test_astype_cannot_cast(self, index, dtype):
        # [FUNCTION DOC]
        # Parametrized over numeric, period, timedelta and datetime dtypes. Verifies
        # that conversion raises a type error saying an interval index cannot be cast to the dtype.
        msg = "Cannot cast IntervalIndex to dtype"
        with pytest.raises(TypeError, match=msg):
            index.astype(dtype)

    def test_astype_invalid_dtype(self, index):
        # [FUNCTION DOC]
        # Verifies that conversion to an unknown dtype name raises a type error saying
        # the data type is not understood.
        msg = "data type [\"']fake_dtype[\"'] not understood"
        with pytest.raises(TypeError, match=msg):
            index.astype("fake_dtype")


class TestIntSubtype(AstypeTests):
    # [CLASS DOC]
    # Conversion tests for integer subtype indexes (signed from negative breaks,
    # and unsigned left-closed). Inherits the shared checks.
    """Tests specific to IntervalIndex with integer-like subtype"""

    indexes = [
        IntervalIndex.from_breaks(np.arange(-10, 11, dtype="int64")),
        IntervalIndex.from_breaks(np.arange(100, dtype="uint64"), closed="left"),
    ]

    @pytest.fixture(params=indexes)
    def index(self, request):
        # [FUNCTION DOC]
        # HELPER: parametrized fixture returning each integer-subtype index of the class list.
        return request.param

    @pytest.mark.parametrize(
        "subtype", ["float64", "datetime64[ns]", "timedelta64[ns]"]
    )
    def test_subtype_conversion(self, index, subtype):
        # [FUNCTION DOC]
        # Parametrized over target subtype (float, datetime, timedelta). Verifies
        # converting to an interval dtype with that subtype and the same closed side
        # equals an index built from the endpoints converted to the subtype.
        dtype = IntervalDtype(subtype, index.closed)
        result = index.astype(dtype)
        expected = IntervalIndex.from_arrays(
            index.left.astype(subtype), index.right.astype(subtype), closed=index.closed
        )
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "subtype_start, subtype_end", [("int64", "uint64"), ("uint64", "int64")]
    )
    def test_subtype_integer(self, subtype_start, subtype_end):
        # [FUNCTION DOC]
        # Parametrized over starting and ending integer kinds (signed to unsigned and
        # back). Verifies converting a non-negative integer index between signed and
        # unsigned subtypes equals an index with converted endpoints.
        index = IntervalIndex.from_breaks(np.arange(100, dtype=subtype_start))
        dtype = IntervalDtype(subtype_end, index.closed)
        result = index.astype(dtype)
        expected = IntervalIndex.from_arrays(
            index.left.astype(subtype_end),
            index.right.astype(subtype_end),
            closed=index.closed,
        )
        tm.assert_index_equal(result, expected)

    @pytest.mark.xfail(reason="GH#15832")
    def test_subtype_integer_errors(self):
        # [FUNCTION DOC]
        # Expected failure (GH#15832). Verifies converting an index with negative
        # endpoints to an unsigned subtype raises a value error whose message must not
        # be the misleading wrapped-around-value message.
        # int64 -> uint64 fails with negative values
        index = interval_range(-10, 10)
        dtype = IntervalDtype("uint64", "right")

        # Until we decide what the exception message _should_ be, we
        #  assert something that it should _not_ be.
        #  We should _not_ be getting a message suggesting that the -10
        #  has been wrapped around to a large-positive integer
        msg = "^(?!(left side of interval must be <= right side))"
        with pytest.raises(ValueError, match=msg):
            index.astype(dtype)


class TestFloatSubtype(AstypeTests):
    # [CLASS DOC]
    # Conversion tests for float subtype indexes (open-closed interval range, and an
    # index with missing values and both sides closed). Inherits the shared checks.
    """Tests specific to IntervalIndex with float subtype"""

    indexes = [
        interval_range(-10.0, 10.0, closed="neither"),
        IntervalIndex.from_arrays(
            [-1.5, np.nan, 0.0, 0.0, 1.5], [-0.5, np.nan, 1.0, 1.0, 3.0], closed="both"
        ),
    ]

    @pytest.fixture(params=indexes)
    def index(self, request):
        # [FUNCTION DOC]
        # HELPER: parametrized fixture returning each float-subtype index of the class list.
        return request.param

    @pytest.mark.parametrize("subtype", ["int64", "uint64"])
    def test_subtype_integer(self, subtype):
        # [FUNCTION DOC]
        # Parametrized over signed and unsigned integer subtype. Verifies a float index
        # with integral endpoints converts to the integer subtype with converted
        # endpoints, and that an index with a missing endpoint raises a value error about
        # non-finite values.
        index = interval_range(0.0, 10.0)
        dtype = IntervalDtype(subtype, "right")
        result = index.astype(dtype)
        expected = IntervalIndex.from_arrays(
            index.left.astype(subtype), index.right.astype(subtype), closed=index.closed
        )
        tm.assert_index_equal(result, expected)

        # raises with NA
        msg = r"Cannot convert non-finite values \(NA or inf\) to integer"
        with pytest.raises(ValueError, match=msg):
            index.insert(0, np.nan).astype(dtype)

    @pytest.mark.parametrize("subtype", ["int64", "uint64"])
    def test_subtype_integer_with_non_integer_borders(self, subtype):
        # [FUNCTION DOC]
        # Parametrized over integer subtypes. Verifies (truncation) conversion of a
        # quarter-step float interval range to integer subtypes equals an index of endpoints
        # converted to the subtype.
        index = interval_range(0.0, 3.0, freq=0.25)
        dtype = IntervalDtype(subtype, "right")
        result = index.astype(dtype)
        expected = IntervalIndex.from_arrays(
            index.left.astype(subtype), index.right.astype(subtype), closed=index.closed
        )
        tm.assert_index_equal(result, expected)

    def test_subtype_integer_errors(self):
        # [FUNCTION DOC]
        # Verifies converting a float index with negative endpoints to an unsigned
        # subtype raises a type error saying the subtypes are incompatible.
        # float64 -> uint64 fails with negative values
        index = interval_range(-10.0, 10.0)
        dtype = IntervalDtype("uint64", "right")
        msg = re.escape(
            "Cannot convert interval[float64, right] to interval[uint64, right]; "
            "subtypes are incompatible"
        )
        with pytest.raises(TypeError, match=msg):
            index.astype(dtype)

    @pytest.mark.parametrize("subtype", ["datetime64[ns]", "timedelta64[ns]"])
    def test_subtype_datetimelike(self, index, subtype):
        # [FUNCTION DOC]
        # Parametrized over datetime and timedelta subtypes plus the index fixture.
        # Verifies a float interval index cannot be converted to a datetime-like subtype
        # (type error: subtypes incompatible).
        dtype = IntervalDtype(subtype, "right")
        msg = "Cannot convert .* to .*; subtypes are incompatible"
        with pytest.raises(TypeError, match=msg):
            index.astype(dtype)

    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in cast:RuntimeWarning"
    )
    def test_astype_category(self, index):
        # [FUNCTION DOC]
        # Overrides the inherited category conversion test only to add a filter that
        # ignores the runtime warning about invalid values in a cast, then delegates to
        # the parent test.
        super().test_astype_category(index)


class TestDatetimelikeSubtype(AstypeTests):
    # [CLASS DOC]
    # Conversion tests for datetime-like subtype indexes (naive and aware
    # timestamps, timedeltas, with and without missing values, several closed
    # sides). Inherits the shared checks.
    """Tests specific to IntervalIndex with datetime-like subtype"""

    indexes = [
        interval_range(Timestamp("2018-01-01"), periods=10, closed="neither"),
        interval_range(Timestamp("2018-01-01"), periods=10).insert(2, NaT),
        interval_range(Timestamp("2018-01-01", tz="US/Eastern"), periods=10),
        interval_range(Timedelta("0 days"), periods=10, closed="both"),
        interval_range(Timedelta("0 days"), periods=10).insert(2, NaT),
    ]

    @pytest.fixture(params=indexes)
    def index(self, request):
        # [FUNCTION DOC]
        # HELPER: parametrized fixture returning each datetime-like index of the class list.
        return request.param

    @pytest.mark.parametrize("subtype", ["int64", "uint64"])
    def test_subtype_integer(self, index, subtype):
        # [FUNCTION DOC]
        # Parametrized over signed and unsigned integer subtype plus the index fixture.
        # For the unsigned subtype verifies a type error about incompatible conversion;
        # for the signed subtype verifies the result equals an index with endpoints
        # converted to the integer subtype.
        dtype = IntervalDtype(subtype, "right")

        if subtype != "int64":
            msg = (
                r"Cannot convert interval"
                r"\[(timedelta64\[us\]|datetime64\[us(, US/Eastern)?\]), .*\] "
                r"to interval\[uint64, .*\]"
            )
            with pytest.raises(TypeError, match=msg):
                index.astype(dtype)
            return

        result = index.astype(dtype)
        new_left = index.left.astype(subtype)
        new_right = index.right.astype(subtype)

        expected = IntervalIndex.from_arrays(new_left, new_right, closed=index.closed)
        tm.assert_index_equal(result, expected)

    def test_subtype_float(self, index):
        # [FUNCTION DOC]
        # Verifies a datetime-like interval index cannot be converted to a float
        # subtype (type error: subtypes incompatible).
        dtype = IntervalDtype("float64", "right")
        msg = "Cannot convert .* to .*; subtypes are incompatible"
        with pytest.raises(TypeError, match=msg):
            index.astype(dtype)

    def test_subtype_datetimelike(self):
        # [FUNCTION DOC]
        # Verifies converting between datetime-like subtypes (naive or aware timestamps to
        # timedelta, and timedelta to datetime) raises a type error about incompatible subtypes.
        # datetime -> timedelta raises
        dtype = IntervalDtype("timedelta64[ns]", "right")
        msg = "Cannot convert .* to .*; subtypes are incompatible"

        index = interval_range(Timestamp("2018-01-01"), periods=10)
        with pytest.raises(TypeError, match=msg):
            index.astype(dtype)

        index = interval_range(Timestamp("2018-01-01", tz="CET"), periods=10)
        with pytest.raises(TypeError, match=msg):
            index.astype(dtype)

        # timedelta -> datetime raises
        dtype = IntervalDtype("datetime64[ns]", "right")
        index = interval_range(Timedelta("0 days"), periods=10)
        with pytest.raises(TypeError, match=msg):
            index.astype(dtype)
