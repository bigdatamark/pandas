# [MODULE DOC]
# Tests of dtype coercion triggered by indexing-adjacent operations: setting with a new key
# (index coercion), index insertion, where, fillna and replace, for index and series objects of
# many dtypes (object, integer, float, complex, bool, datetime with and without timezone,
# timedelta, period, interval). Organized as a small base class plus one class per operation.
# Many placeholder tests are marked expected-fail with a not-implemented reason. Nothing is
# executed at import time except building the class-level replacement table.
from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

from pandas.compat import (
    IS64,
    is_platform_windows,
)
from pandas.compat.numpy import np_version_gt2

import pandas as pd
import pandas._testing as tm

###############################################################
# Index / Series common tests which may trigger dtype coercions
###############################################################


class CoercionBase:
    # [CLASS DOC]
    # Base class for the coercion test classes. Declares class attributes for the object kinds
    # (index and series) and dtype names under test, plus a property that subclasses override
    # with a string naming the operation.
    # KNOWN ISSUE: the kinds and dtypes lists are not referenced anywhere in this module's tests.
    klasses = ["index", "series"]
    dtypes = [
        "object",
        "int64",
        "float64",
        "complex128",
        "bool",
        "datetime64",
        "datetime64tz",
        "timedelta64",
        "period",
    ]

    @property
    def method(self):
        # [FUNCTION DOC]
        # Property that raises a not-implemented error when accessed; every concrete subclass shadows it
        # with a plain string class attribute (setitem, insert, where, fillna, replace), so the base
        # implementation is never called.
        raise NotImplementedError(self)


class TestSetitemCoercion(CoercionBase):
    # [CLASS DOC]
    # Tests index dtype coercion when a series is enlarged by assigning to a new key. Sets the
    # kinds list to empty because exhaustive cases moved elsewhere.
    method = "setitem"

    # disable comprehensiveness tests, as most of these have been moved to
    #  tests.series.indexing.test_setitem in SetitemCastingEquivalents subclasses.
    klasses: list[str] = []

    def test_setitem_series_no_coercion_from_values_list(self):
        # GH35865 - int casted to str when internally calling np.array(ser.values)
        # [FUNCTION DOC]
        # GH 35865. Assigning a full slice from a list of the values of a mixed string / integer series
        # leaves the series unchanged (the integer must not be cast to string).
        ser = pd.Series(["a", 1])
        ser[:] = list(ser.values)

        expected = pd.Series(["a", 1])

        tm.assert_series_equal(ser, expected)

    def _assert_setitem_index_conversion(
        self, original_series, loc_key, expected_index, expected_dtype
    ):
        # [FUNCTION DOC]
        # HELPER: for the setitem tests. Takes an original series (four elements), the new key, the
        # expected index and expected index dtype. On a copy, enlarges by plain setitem with value 5 and
        # then again on a fresh copy via loc; each time asserts the series equals the series of 1-5
        # with the expected index and that the index dtype equals the expected dtype. No return value.
        """test index's coercion triggered by assign key"""
        temp = original_series.copy()
        # GH#33469 pre-2.0 with int loc_key and temp.index.dtype == np.float64
        #  `temp[loc_key] = 5` treated loc_key as positional
        temp[loc_key] = 5
        exp = pd.Series([1, 2, 3, 4, 5], index=expected_index)
        tm.assert_series_equal(temp, exp)
        # check dtype explicitly for sure
        assert temp.index.dtype == expected_dtype

        temp = original_series.copy()
        temp.loc[loc_key] = 5
        exp = pd.Series([1, 2, 3, 4, 5], index=expected_index)
        tm.assert_series_equal(temp, exp)
        # check dtype explicitly for sure
        assert temp.index.dtype == expected_dtype

    @pytest.mark.parametrize(
        "val,exp_dtype", [("x", object), (5, IndexError), (1.1, object)]
    )
    def test_setitem_index_object(self, val, exp_dtype):
        # [FUNCTION DOC]
        # Parametrized over (key, expected dtype) for a string-labelled object index. New string, integer
        # and float labels all append to an object index.
        # KNOWN ISSUE: the integer case passes an exception class as the expected dtype; it only passes
        # because converting an arbitrary class to a numpy dtype yields the object dtype, so the
        # exception class has no meaning here.
        obj = pd.Series([1, 2, 3, 4], index=pd.Index(list("abcd"), dtype=object))
        assert obj.index.dtype == object

        exp_index = pd.Index([*list("abcd"), val], dtype=object)
        self._assert_setitem_index_conversion(obj, val, exp_index, exp_dtype)

    @pytest.mark.parametrize(
        "val,exp_dtype", [(5, np.int64), (1.1, np.float64), ("x", object)]
    )
    def test_setitem_index_int64(self, val, exp_dtype):
        # [FUNCTION DOC]
        # Parametrized over (new key, expected index dtype). On a default integer index, an integer key
        # keeps int64, a float key coerces the index to float64, and a string key to object.
        obj = pd.Series([1, 2, 3, 4])
        assert obj.index.dtype == np.int64

        exp_index = pd.Index([0, 1, 2, 3, val])
        self._assert_setitem_index_conversion(obj, val, exp_index, exp_dtype)

    @pytest.mark.parametrize(
        "val,exp_dtype", [(5, np.float64), (5.1, np.float64), ("x", object)]
    )
    def test_setitem_index_float64(self, val, exp_dtype, request):
        # [FUNCTION DOC]
        # Parametrized over key and expected dtype. On a float index, integer or float new keys keep
        # float64, and string coerces to object.
        # KNOWN ISSUE: the request fixture parameter is accepted but never used.
        obj = pd.Series([1, 2, 3, 4], index=[1.1, 2.1, 3.1, 4.1])
        assert obj.index.dtype == np.float64

        exp_index = pd.Index([1.1, 2.1, 3.1, 4.1, val])
        self._assert_setitem_index_conversion(obj, val, exp_index, exp_dtype)

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_series_period(self):
        # [FUNCTION DOC]
        # Placeholder, expected to fail; raises not-implemented.
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_index_complex128(self):
        # [FUNCTION DOC]
        # Placeholder, expected to fail; raises not-implemented.
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_index_bool(self):
        # [FUNCTION DOC]
        # Placeholder, expected to fail; raises not-implemented.
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_index_datetime64(self):
        # [FUNCTION DOC]
        # Placeholder, expected to fail; raises not-implemented.
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_index_datetime64tz(self):
        # [FUNCTION DOC]
        # Placeholder, expected to fail; raises not-implemented.
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_index_timedelta64(self):
        # [FUNCTION DOC]
        # Placeholder, expected to fail; raises not-implemented.
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_index_period(self):
        # [FUNCTION DOC]
        # Placeholder, expected to fail; raises not-implemented.
        raise NotImplementedError


class TestInsertIndexCoercion(CoercionBase):
    # [CLASS DOC]
    # Tests dtype coercion when inserting a value into an index. Only the index kind applies.
    klasses = ["index"]
    method = "insert"

    def _assert_insert_conversion(self, original, value, expected, expected_dtype):
        # [FUNCTION DOC]
        # HELPER: inserts value at position 1 of a copy of the original index and asserts the result
        # equals the expected index and has the expected dtype.
        """test coercion triggered by insert"""
        target = original.copy()
        res = target.insert(1, value)
        tm.assert_index_equal(res, expected)
        assert res.dtype == expected_dtype

    @pytest.mark.parametrize(
        "insert, coerced_val, coerced_dtype",
        [
            (1, 1, object),
            (1.1, 1.1, object),
            (False, False, object),
            ("x", "x", object),
        ],
    )
    def test_insert_index_object(self, insert, coerced_val, coerced_dtype):
        # [FUNCTION DOC]
        # Parametrized over insert values (integer, float, bool, string). Inserting into an object index
        # keeps object dtype and places the value second.
        obj = pd.Index(list("abcd"), dtype=object)
        assert obj.dtype == object

        exp = pd.Index(["a", coerced_val, "b", "c", "d"], dtype=object)
        self._assert_insert_conversion(obj, insert, exp, coerced_dtype)

    @pytest.mark.parametrize(
        "insert, coerced_val, coerced_dtype",
        [
            (1, 1, None),
            (1.1, 1.1, np.float64),
            (False, False, object),  # GH#36319
            ("x", "x", object),
        ],
    )
    def test_insert_int_index(
        self, any_int_numpy_dtype, insert, coerced_val, coerced_dtype
    ):
        # [FUNCTION DOC]
        # Parametrized over each integer numpy dtype fixture and inserts. Inserting an integer keeps
        # the dtype (None means keep), a float coerces to float64, bool and string coerce to object.
        dtype = any_int_numpy_dtype
        obj = pd.Index([1, 2, 3, 4], dtype=dtype)
        coerced_dtype = coerced_dtype if coerced_dtype is not None else dtype

        exp = pd.Index([1, coerced_val, 2, 3, 4], dtype=coerced_dtype)
        self._assert_insert_conversion(obj, insert, exp, coerced_dtype)

    @pytest.mark.parametrize(
        "insert, coerced_val, coerced_dtype",
        [
            (1, 1.0, None),
            # When float_numpy_dtype=float32, this is not the case
            # see the correction below
            (1.1, 1.1, np.float64),
            (False, False, object),  # GH#36319
            ("x", "x", object),
        ],
    )
    def test_insert_float_index(
        self, float_numpy_dtype, insert, coerced_val, coerced_dtype
    ):
        # [FUNCTION DOC]
        # Parametrized over float numpy dtypes. Inserting an integer keeps dtype, a float coerces to
        # float64 (except with newer numpy and float32 where 1.1 is exactly representable and the dtype
        # stays float32), bool and string give object.
        # KNOWN ISSUE: the 1.1 case is special-cased by a hack in the body for float32 under newer numpy;
        # the comment in the parametrization refers to a "correction below".
        dtype = float_numpy_dtype
        obj = pd.Index([1.0, 2.0, 3.0, 4.0], dtype=dtype)
        coerced_dtype = coerced_dtype if coerced_dtype is not None else dtype

        if np_version_gt2 and dtype == "float32" and coerced_val == 1.1:
            # Hack, in the 2nd test case, since 1.1 can be losslessly cast to float32
            # the expected dtype will be float32 if the original dtype was float32
            coerced_dtype = np.float32
        exp = pd.Index([1.0, coerced_val, 2.0, 3.0, 4.0], dtype=coerced_dtype)
        self._assert_insert_conversion(obj, insert, exp, coerced_dtype)

    @pytest.mark.parametrize(
        "fill_val,exp_dtype",
        [
            (pd.Timestamp("2012-01-01").as_unit("s"), "datetime64[ns]"),
            (
                pd.Timestamp("2012-01-01", tz="US/Eastern").as_unit("s"),
                "datetime64[ns, US/Eastern]",
            ),
        ],
        ids=["datetime64", "datetime64tz"],
    )
    @pytest.mark.parametrize(
        "insert_value",
        [
            pd.Timestamp("2012-01-01").as_unit("s"),
            pd.Timestamp("2012-01-01", tz="Asia/Tokyo").as_unit("s"),
            1,
        ],
    )
    def test_insert_index_datetimes(self, fill_val, exp_dtype, insert_value):
        # [FUNCTION DOC]
        # Parametrized over a naive or timezone-aware index and inserted values (matching timestamp,
        # timestamp with a different zone, integer). Inserting a matching timestamp keeps the dtype;
        # inserting a naive timestamp into aware or an aware into naive gives an object index; for the
        # aware-index, inserting a timestamp in a different zone converts it to the index zone; an
        # integer gives an object index with the integer second.
        obj = pd.DatetimeIndex(
            ["2011-01-01", "2011-01-02", "2011-01-03", "2011-01-04"], tz=fill_val.tz
        ).as_unit("ns")
        assert obj.dtype == exp_dtype

        exp = pd.DatetimeIndex(
            ["2011-01-01", fill_val.date(), "2011-01-02", "2011-01-03", "2011-01-04"],
            tz=fill_val.tz,
        ).as_unit("ns")
        self._assert_insert_conversion(obj, fill_val, exp, exp_dtype)

        if fill_val.tz:
            # mismatched tzawareness
            ts = pd.Timestamp("2012-01-01").as_unit("s")
            result = obj.insert(1, ts)
            expected = obj.astype(object).insert(1, ts)
            assert expected.dtype == object
            tm.assert_index_equal(result, expected)

            ts = pd.Timestamp("2012-01-01", tz="Asia/Tokyo").as_unit("s")
            result = obj.insert(1, ts)
            # once deprecation is enforced:
            expected = obj.insert(1, ts.tz_convert(obj.dtype.tz))
            assert expected.dtype == obj.dtype
            tm.assert_index_equal(result, expected)

        else:
            # mismatched tzawareness
            ts = pd.Timestamp("2012-01-01", tz="Asia/Tokyo").as_unit("s")
            result = obj.insert(1, ts)
            expected = obj.astype(object).insert(1, ts)
            assert expected.dtype == object
            tm.assert_index_equal(result, expected)

        item = 1
        result = obj.insert(1, item)
        expected = obj.astype(object).insert(1, item)
        assert expected[1] == item
        assert expected.dtype == object
        tm.assert_index_equal(result, expected)

    def test_insert_index_timedelta64(self):
        # [FUNCTION DOC]
        # Inserting a timedelta into a microsecond timedelta index keeps the dtype; inserting a timestamp
        # or integer gives an object index.
        obj = pd.TimedeltaIndex(["1 day", "2 day", "3 day", "4 day"])
        assert obj.dtype == "timedelta64[us]"

        # timedelta64 + timedelta64 => timedelta64
        exp = pd.TimedeltaIndex(["1 day", "10 day", "2 day", "3 day", "4 day"])
        self._assert_insert_conversion(
            obj, pd.Timedelta("10 day"), exp, "timedelta64[us]"
        )

        for item in [pd.Timestamp("2012-01-01").as_unit("s"), 1]:
            result = obj.insert(1, item)
            expected = obj.astype(object).insert(1, item)
            assert expected.dtype == object
            tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "insert, coerced_val, coerced_dtype",
        [
            (pd.Period("2012-01", freq="M"), "2012-01", "period[M]"),
            (
                pd.Timestamp("2012-01-01").as_unit("s"),
                pd.Timestamp("2012-01-01").as_unit("s"),
                object,
            ),
            (1, 1, object),
            ("x", "x", object),
        ],
    )
    def test_insert_index_period(self, insert, coerced_val, coerced_dtype):
        # [FUNCTION DOC]
        # Parametrized over insert value and expected value / dtype. A period of matching frequency (or
        # its string form) keeps the period dtype; any other value (timestamp, integer, string) results
        # in an object index, compared to inserting into the object-cast index; a string that cannot be
        # parsed as a period is also tested (skipped for timestamps).
        # KNOWN ISSUE: a TODO notes inserting a timestamp string at the monthly frequency is cast to a
        # period, which may not be desirable.
        obj = pd.PeriodIndex(["2011-01", "2011-02", "2011-03", "2011-04"], freq="M")
        assert obj.dtype == "period[M]"

        data = [
            pd.Period("2011-01", freq="M"),
            coerced_val,
            pd.Period("2011-02", freq="M"),
            pd.Period("2011-03", freq="M"),
            pd.Period("2011-04", freq="M"),
        ]
        if isinstance(insert, pd.Period):
            exp = pd.PeriodIndex(data, freq="M")
            self._assert_insert_conversion(obj, insert, exp, coerced_dtype)

            # string that can be parsed to appropriate PeriodDtype
            self._assert_insert_conversion(obj, str(insert), exp, coerced_dtype)

        else:
            result = obj.insert(0, insert)
            expected = obj.astype(object).insert(0, insert)
            tm.assert_index_equal(result, expected)

            # TODO: ATM inserting '2012-01-01 00:00:00' when we have obj.freq=="M"
            #  casts that string to Period[M], not clear that is desirable
            if not isinstance(insert, pd.Timestamp):
                # non-castable string
                result = obj.insert(0, str(insert))
                expected = obj.astype(object).insert(0, str(insert))
                tm.assert_index_equal(result, expected)

    @pytest.mark.xfail(reason="Test not implemented")
    def test_insert_index_complex128(self):
        # [FUNCTION DOC]
        # Placeholder, expected to fail; raises not-implemented.
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_insert_index_bool(self):
        # [FUNCTION DOC]
        # Placeholder, expected to fail; raises not-implemented.
        raise NotImplementedError


class TestWhereCoercion(CoercionBase):
    # [CLASS DOC]
    # Tests dtype coercion of the where method for index and series. A class attribute holds a fixed
    # four-element boolean condition array (true at positions 0 and 2).
    method = "where"
    _cond = np.array([True, False, True, False])

    def _assert_where_conversion(
        self, original, cond, values, expected, expected_dtype
    ):
        # [FUNCTION DOC]
        # HELPER: applies where to a copy of the original with the condition and replacement values,
        # asserts equality with expected (via the generic equality assertion) and that the dtype equals
        # the expected dtype.
        """test coercion triggered by where"""
        target = original.copy()
        res = target.where(cond, values)
        tm.assert_equal(res, expected)
        assert res.dtype == expected_dtype

    def _construct_exp(self, obj, klass, fill_val, exp_dtype):
        # [FUNCTION DOC]
        # HELPER: given the original object, container class, fill value and expected dtype, builds the
        # array-like of replacement values (for boolean true: a four-element boolean pattern; for
        # datetime-like: a daily date range starting at the fill value; otherwise the numbers 5-8
        # multiplied by the fill value) and the expected container whose elements at positions 1 and 3
        # come from the replacement values and others from the original. Returns the pair (values, expected).
        if fill_val is True:
            values = klass([True, False, True, True])
        elif isinstance(fill_val, (datetime, np.datetime64)):
            values = pd.date_range(fill_val, periods=4)
        else:
            values = klass(x * fill_val for x in [5, 6, 7, 8])

        exp = klass([obj[0], values[1], obj[2], values[3]], dtype=exp_dtype)
        return values, exp

    def _run_test(self, obj, fill_val, klass, exp_dtype):
        # [FUNCTION DOC]
        # HELPER: runs both variants: replacing with the scalar fill value (expected has the fill at
        # positions 1 and 3) and replacing with an array of values from the construction helper. The
        # condition is built by wrapping the class attribute with the container class. Asserts via
        # the assertion helper.
        cond = klass(self._cond)

        exp = klass([obj[0], fill_val, obj[2], fill_val], dtype=exp_dtype)
        self._assert_where_conversion(obj, cond, fill_val, exp, exp_dtype)

        values, exp = self._construct_exp(obj, klass, fill_val, exp_dtype)
        self._assert_where_conversion(obj, cond, values, exp, exp_dtype)

    @pytest.mark.parametrize(
        "fill_val,exp_dtype",
        [(1, object), (1.1, object), (1 + 1j, object), (True, object)],
    )
    def test_where_object(self, index_or_series, fill_val, exp_dtype):
        # [FUNCTION DOC]
        # Parametrized over fill values; for an object container, result remains object for all fills.
        klass = index_or_series
        obj = klass(list("abcd"), dtype=object)
        assert obj.dtype == object
        self._run_test(obj, fill_val, klass, exp_dtype)

    @pytest.mark.parametrize(
        "fill_val,exp_dtype",
        [(1, np.int64), (1.1, np.float64), (1 + 1j, np.complex128), (True, object)],
    )
    def test_where_int64(self, index_or_series, fill_val, exp_dtype, request):
        # [FUNCTION DOC]
        # Parametrized over fill value. Integer fill keeps int64, float coerces to float64, complex to
        # complex128, bool to object.
        # KNOWN ISSUE: the request fixture argument is unused.
        klass = index_or_series

        obj = klass([1, 2, 3, 4])
        assert obj.dtype == np.int64
        self._run_test(obj, fill_val, klass, exp_dtype)

    @pytest.mark.parametrize(
        "fill_val, exp_dtype",
        [(1, np.float64), (1.1, np.float64), (1 + 1j, np.complex128), (True, object)],
    )
    def test_where_float64(self, index_or_series, fill_val, exp_dtype, request):
        # [FUNCTION DOC]
        # Parametrized over fill value. Integer and float fills keep float64, complex gives complex128,
        # bool gives object.
        # KNOWN ISSUE: the request fixture argument is unused.
        klass = index_or_series

        obj = klass([1.1, 2.2, 3.3, 4.4])
        assert obj.dtype == np.float64
        self._run_test(obj, fill_val, klass, exp_dtype)

    @pytest.mark.parametrize(
        "fill_val,exp_dtype",
        [
            (1, np.complex128),
            (1.1, np.complex128),
            (1 + 1j, np.complex128),
            (True, object),
        ],
    )
    def test_where_complex128(self, index_or_series, fill_val, exp_dtype):
        # [FUNCTION DOC]
        # Parametrized over fill value. Numeric fills keep complex128; bool gives object.
        klass = index_or_series
        obj = klass([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], dtype=np.complex128)
        assert obj.dtype == np.complex128
        self._run_test(obj, fill_val, klass, exp_dtype)

    @pytest.mark.parametrize(
        "fill_val,exp_dtype",
        [(1, object), (1.1, object), (1 + 1j, object), (True, np.bool_)],
    )
    def test_where_series_bool(self, index_or_series, fill_val, exp_dtype):
        # [FUNCTION DOC]
        # Parametrized over fill value. Only a bool fill keeps boolean dtype; numeric fills give object.
        klass = index_or_series

        obj = klass([True, False, True, False])
        assert obj.dtype == np.bool_
        self._run_test(obj, fill_val, klass, exp_dtype)

    @pytest.mark.parametrize(
        "fill_val,exp_dtype",
        [
            (pd.Timestamp("2012-01-01").as_unit("s"), "datetime64[ns]"),
            (pd.Timestamp("2012-01-01", tz="US/Eastern").as_unit("s"), object),
        ],
        ids=["datetime64", "datetime64tz"],
    )
    def test_where_datetime64(self, index_or_series, fill_val, exp_dtype):
        # [FUNCTION DOC]
        # Parametrized over a naive timestamp (expected datetime64 nanosecond) and a timezone-aware one
        # (expected object). The container is built on a daily date range without frequency. For the naive
        # fill, scalars in three forms (timestamp, python datetime, numpy datetime64) are each tested.
        # KNOWN ISSUE: in the object branch the loop variable is never used; the original fill value is
        # passed on every iteration, so the python datetime form is not actually tested.
        klass = index_or_series

        obj = klass(
            pd.date_range("2011-01-01", periods=4, freq="D", unit="ns")._with_freq(None)
        )
        assert obj.dtype == "datetime64[ns]"

        fv = fill_val
        # do the check with each of the available datetime scalars
        if exp_dtype == "datetime64[ns]":
            for scalar in [fv, fv.to_pydatetime(), fv.to_datetime64()]:
                self._run_test(obj, scalar, klass, exp_dtype)
        else:
            for scalar in [fv, fv.to_pydatetime()]:
                self._run_test(obj, fill_val, klass, exp_dtype)

    @pytest.mark.xfail(reason="Test not implemented")
    def test_where_index_complex128(self):
        # [FUNCTION DOC]
        # Placeholder, expected to fail; raises not-implemented.
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_where_index_bool(self):
        # [FUNCTION DOC]
        # Placeholder, expected to fail; raises not-implemented.
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_where_series_timedelta64(self):
        # [FUNCTION DOC]
        # Placeholder, expected to fail; raises not-implemented.
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_where_series_period(self):
        # [FUNCTION DOC]
        # Placeholder, expected to fail; raises not-implemented.
        raise NotImplementedError

    @pytest.mark.parametrize(
        "value",
        [pd.Timedelta(days=9).as_unit("us"), timedelta(days=9), np.timedelta64(9, "D")],
    )
    def test_where_index_timedelta64(self, value):
        # [FUNCTION DOC]
        # Parametrized over three timedelta representations (pandas, python, numpy). Where on a
        # timedelta index replaces with the value, keeping timedelta dtype. A datetime NaT (wrong kind)
        # replaces by casting to object, keeping that exact object.
        tdi = pd.timedelta_range("1 Day", periods=4)
        cond = np.array([True, False, False, True])

        expected = pd.TimedeltaIndex(["1 Day", value, value, "4 Days"])
        result = tdi.where(cond, value)
        tm.assert_index_equal(result, expected)

        # wrong-dtyped NaT
        dtnat = np.datetime64("NaT", "ns")
        expected = pd.Index([tdi[0], dtnat, dtnat, tdi[3]], dtype=object)
        assert expected[1] is dtnat

        result = tdi.where(cond, dtnat)
        tm.assert_index_equal(result, expected)

    def test_where_index_period(self):
        # [FUNCTION DOC]
        # On a quarterly period index: a valid period scalar, an object array of periods, both keep period
        # dtype. A timedelta, or a period of different frequency, makes the result an object index.
        dti = pd.date_range("2016-01-01", periods=3, freq="QS")
        pi = dti.to_period("Q")

        cond = np.array([False, True, False])

        # Passing a valid scalar
        value = pi[-1] + pi.freq * 10
        expected = pd.PeriodIndex([value, pi[1], value])
        result = pi.where(cond, value)
        tm.assert_index_equal(result, expected)

        # Case passing ndarray[object] of Periods
        other = np.asarray(pi + pi.freq * 10, dtype=object)
        result = pi.where(cond, other)
        expected = pd.PeriodIndex([other[0], pi[1], other[2]])
        tm.assert_index_equal(result, expected)

        # Passing a mismatched scalar -> casts to object
        td = pd.Timedelta(days=4)
        expected = pd.Index([td, pi[1], td], dtype=object)
        result = pi.where(cond, td)
        tm.assert_index_equal(result, expected)

        per = pd.Period("2020-04-21", "D")
        expected = pd.Index([per, pi[1], per], dtype=object)
        result = pi.where(cond, per)
        tm.assert_index_equal(result, expected)


class TestFillnaSeriesCoercion(CoercionBase):
    # not indexing, but place here for consistency

    # [CLASS DOC]
    # Tests dtype coercion of fillna for index and series containing missing values. (Not an
    # indexing feature, located here for consistency.)
    method = "fillna"

    def _assert_fillna_conversion(self, original, value, expected, expected_dtype):
        # [FUNCTION DOC]
        # HELPER: calls fillna with value on a copy of the original, asserts equality to expected and that
        # the dtype equals the expected dtype.
        """test coercion triggered by fillna"""
        target = original.copy()
        res = target.fillna(value)
        tm.assert_equal(res, expected)
        assert res.dtype == expected_dtype

    @pytest.mark.parametrize(
        "fill_val, fill_dtype",
        [(1, object), (1.1, object), (1 + 1j, object), (True, object)],
    )
    def test_fillna_object(self, index_or_series, fill_val, fill_dtype):
        # [FUNCTION DOC]
        # Parametrized over fill value. For an object container with a missing element, fill with any
        # value keeps object dtype.
        klass = index_or_series
        obj = klass(["a", np.nan, "c", "d"], dtype=object)
        assert obj.dtype == object

        exp = klass(["a", fill_val, "c", "d"], dtype=object)
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.parametrize(
        "fill_val,fill_dtype",
        [(1, np.float64), (1.1, np.float64), (1 + 1j, np.complex128), (True, object)],
    )
    def test_fillna_float64(self, index_or_series, fill_val, fill_dtype):
        # [FUNCTION DOC]
        # Parametrized over fill. Integer or float keeps float64, complex gives complex128, bool gives
        # object.
        klass = index_or_series
        obj = klass([1.1, np.nan, 3.3, 4.4])
        assert obj.dtype == np.float64

        exp = klass([1.1, fill_val, 3.3, 4.4])
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.parametrize(
        "fill_val,fill_dtype",
        [
            (1, np.complex128),
            (1.1, np.complex128),
            (1 + 1j, np.complex128),
            (True, object),
        ],
    )
    def test_fillna_complex128(self, index_or_series, fill_val, fill_dtype):
        # [FUNCTION DOC]
        # Parametrized over fill. Numeric fills keep complex128; bool gives object.
        klass = index_or_series
        obj = klass([1 + 1j, np.nan, 3 + 3j, 4 + 4j], dtype=np.complex128)
        assert obj.dtype == np.complex128

        exp = klass([1 + 1j, fill_val, 3 + 3j, 4 + 4j])
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.parametrize(
        "fill_val,fill_dtype",
        [
            (pd.Timestamp("2012-01-01").as_unit("s"), "datetime64[s]"),
            (pd.Timestamp("2012-01-01", tz="US/Eastern").as_unit("s"), object),
            (1, object),
            ("x", object),
        ],
        ids=["datetime64", "datetime64tz", "object", "object"],
    )
    def test_fillna_datetime(self, index_or_series, fill_val, fill_dtype):
        # [FUNCTION DOC]
        # Parametrized over fill and expected dtype. On a second-unit datetime container with NaT, filling
        # with a naive timestamp keeps the dtype; a timezone-aware timestamp, integer, or string gives
        # object.
        klass = index_or_series
        obj = klass(
            [
                pd.Timestamp("2011-01-01").as_unit("s"),
                pd.NaT,
                pd.Timestamp("2011-01-03").as_unit("s"),
                pd.Timestamp("2011-01-04").as_unit("s"),
            ]
        )
        assert obj.dtype == "datetime64[s]"

        exp = klass(
            [
                pd.Timestamp("2011-01-01").as_unit("s"),
                fill_val,
                pd.Timestamp("2011-01-03").as_unit("s"),
                pd.Timestamp("2011-01-04").as_unit("s"),
            ]
        )
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.parametrize(
        "fill_val,fill_dtype",
        [
            (
                pd.Timestamp("2012-01-01", tz="US/Eastern").as_unit("s"),
                "datetime64[s, US/Eastern]",
            ),
            (pd.Timestamp("2012-01-01").as_unit("s"), object),
            # pre-2.0 with a mismatched tz we would get object result
            (
                pd.Timestamp("2012-01-01", tz="Asia/Tokyo").as_unit("s"),
                "datetime64[s, US/Eastern]",
            ),
            (1, object),
            ("x", object),
        ],
    )
    def test_fillna_datetime64tz(self, index_or_series, fill_val, fill_dtype):
        # [FUNCTION DOC]
        # Parametrized over fill. On a timezone-aware container, a timestamp in the same zone keeps the
        # dtype; a naive timestamp, integer, or string gives object; a timestamp in another zone is
        # converted to the container zone and keeps the dtype (the expected values use the converted fill).
        klass = index_or_series
        tz = "US/Eastern"

        obj = klass(
            [
                pd.Timestamp("2011-01-01", tz=tz).as_unit("s"),
                pd.NaT,
                pd.Timestamp("2011-01-03", tz=tz).as_unit("s"),
                pd.Timestamp("2011-01-04", tz=tz).as_unit("s"),
            ]
        )
        assert obj.dtype == "datetime64[s, US/Eastern]"

        if getattr(fill_val, "tz", None) is None:
            fv = fill_val
        else:
            fv = fill_val.tz_convert(tz)
        exp = klass(
            [
                pd.Timestamp("2011-01-01", tz=tz).as_unit("s"),
                fv,
                pd.Timestamp("2011-01-03", tz=tz).as_unit("s"),
                pd.Timestamp("2011-01-04", tz=tz).as_unit("s"),
            ]
        )
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.parametrize(
        "fill_val",
        [
            1,
            1.1,
            1 + 1j,
            True,
            pd.Interval(1, 2, closed="left"),
            pd.Timestamp("2012-01-01", tz="US/Eastern").as_unit("s"),
            pd.Timestamp("2012-01-01").as_unit("s"),
            pd.Timedelta(days=1),
            pd.Period("2016-01-01", "D"),
        ],
    )
    def test_fillna_interval(self, index_or_series, fill_val):
        # [FUNCTION DOC]
        # Parametrized over many fill values. An interval-dtype container with a missing value filled
        # with any listed value gives an object result with that value in place.
        ii = pd.interval_range(1.0, 5.0, closed="right").insert(1, np.nan)
        assert isinstance(ii.dtype, pd.IntervalDtype)
        obj = index_or_series(ii)

        exp = index_or_series([ii[0], fill_val, ii[2], ii[3], ii[4]], dtype=object)

        fill_dtype = object
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.xfail(reason="Test not implemented")
    def test_fillna_series_int64(self):
        # [FUNCTION DOC]
        # Placeholder, expected to fail; raises not-implemented.
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_fillna_index_int64(self):
        # [FUNCTION DOC]
        # Placeholder, expected to fail; raises not-implemented.
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_fillna_series_bool(self):
        # [FUNCTION DOC]
        # Placeholder, expected to fail; raises not-implemented.
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_fillna_index_bool(self):
        # [FUNCTION DOC]
        # Placeholder, expected to fail; raises not-implemented.
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_fillna_series_timedelta64(self):
        # [FUNCTION DOC]
        # Placeholder, expected to fail; raises not-implemented.
        raise NotImplementedError

    @pytest.mark.parametrize(
        "fill_val",
        [
            1,
            1.1,
            1 + 1j,
            True,
            pd.Interval(1, 2, closed="left"),
            pd.Timestamp("2012-01-01", tz="US/Eastern").as_unit("s"),
            pd.Timestamp("2012-01-01").as_unit("s"),
            pd.Timedelta(days=1),
            pd.Period("2016-01-01", "W"),
        ],
    )
    def test_fillna_series_period(self, index_or_series, fill_val):
        # [FUNCTION DOC]
        # Parametrized over many fill values. A period-dtype container with a missing value filled with
        # any listed value (including a period of different frequency) gives object dtype with the value
        # in place.
        pi = pd.period_range("2016-01-01", periods=4, freq="D").insert(1, pd.NaT)
        assert isinstance(pi.dtype, pd.PeriodDtype)
        obj = index_or_series(pi)

        exp = index_or_series([pi[0], fill_val, pi[2], pi[3], pi[4]], dtype=object)

        fill_dtype = object
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.xfail(reason="Test not implemented")
    def test_fillna_index_timedelta64(self):
        # [FUNCTION DOC]
        # Placeholder, expected to fail; raises not-implemented.
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_fillna_index_period(self):
        # [FUNCTION DOC]
        # Placeholder, expected to fail; raises not-implemented.
        raise NotImplementedError


class TestReplaceSeriesCoercion(CoercionBase):
    # [CLASS DOC]
    # Tests dtype coercion of series replace. Only the series kind applies. A class-level table maps
    # dtype names to two sample values (built when the class body executes, using a loop that also
    # leaves two stray class attributes behind). Fixtures supply how the replacement is passed
    # (dictionary or series), the source dtype key, the target dtype key, and the replacer object.
    klasses = ["series"]
    method = "replace"

    rep: dict[str, list] = {}
    rep["object"] = ["a", "b"]
    rep["int64"] = [4, 5]
    rep["float64"] = [1.1, 2.2]
    rep["complex128"] = [1 + 1j, 2 + 2j]
    rep["bool"] = [True, False]
    rep["datetime64[ns]"] = [
        pd.Timestamp("2011-01-01").as_unit("s"),
        pd.Timestamp("2011-01-03").as_unit("s"),
    ]

    for tz in ["UTC", "US/Eastern"]:
        # to test tz => different tz replacement
        key = f"datetime64[ns, {tz}]"
        rep[key] = [
            pd.Timestamp("2011-01-01", tz=tz).as_unit("s"),
            pd.Timestamp("2011-01-03", tz=tz).as_unit("s"),
        ]

    rep["timedelta64[us]"] = [pd.Timedelta("1 day"), pd.Timedelta("2 day")]

    @pytest.fixture(params=["dict", "series"])
    def how(self, request):
        # [FUNCTION DOC]
        # HELPER: fixture parametrized over the container type used to pass the replacements: a mapping
        # or a series.
        return request.param

    @pytest.fixture(
        params=[
            "object",
            "int64",
            "float64",
            "complex128",
            "bool",
            "datetime64[ns]",
            "datetime64[ns, UTC]",
            "datetime64[ns, US/Eastern]",
            "timedelta64[us]",
        ]
    )
    def from_key(self, request):
        # [FUNCTION DOC]
        # HELPER: fixture parametrized over nine source dtype names (object, int64, float64, complex128,
        # bool, datetime without and with two zones, timedelta in microseconds).
        return request.param

    @pytest.fixture(
        params=[
            "object",
            "int64",
            "float64",
            "complex128",
            "bool",
            "datetime64[ns]",
            "datetime64[ns, UTC]",
            "datetime64[ns, US/Eastern]",
            "timedelta64[us]",
        ],
        ids=[
            "object",
            "int64",
            "float64",
            "complex128",
            "bool",
            "datetime64",
            "datetime64tz",
            "datetime64tz",
            "timedelta64",
        ],
    )
    def to_key(self, request):
        # [FUNCTION DOC]
        # HELPER: fixture parametrized over the same nine target dtype names, with ids shortening the
        # datetime names.
        return request.param

    @pytest.fixture
    def replacer(self, how, from_key, to_key):
        # [FUNCTION DOC]
        # HELPER: fixture building the object passed to replace. Using the sample values for the source
        # as keys/index and the sample values for the target as values. Returns a mapping when how is
        # a mapping, or a series when it is a series.
        # KNOWN ISSUE: the final branch raising a value error is unreachable, because the how fixture
        # only has two values.
        """
        Object we will pass to `Series.replace`
        """
        if how == "dict":
            replacer = dict(zip(self.rep[from_key], self.rep[to_key]))
        elif how == "series":
            replacer = pd.Series(self.rep[to_key], index=self.rep[from_key])
        else:
            raise ValueError
        return replacer

    def test_replace_series(self, to_key, from_key, replacer):
        # [FUNCTION DOC]
        # Cross product of source and target dtypes (and replacer kind). Casts a two-element series to
        # the source dtype and replaces; the result must equal a series of the target values without
        # checking dtype. Combinations where both are datetime-like, or the source is timezone-aware,
        # return early (covered by later tests). For float to int and complex to int/float cases, the
        # expected dtype is the source dtype (no downcasting) and the case is skipped on 32-bit or Windows
        # platforms.
        # KNOWN ISSUE: the membership check on the target key uses a plain string in parentheses rather
        # than a tuple, so it is a substring test that also matches shorter strings (for example "int").
        index = pd.Index([3, 4], name="xxx")
        obj = pd.Series(self.rep[from_key], index=index, name="yyy")
        obj = obj.astype(from_key)
        assert obj.dtype == from_key

        if from_key.startswith("datetime") and to_key.startswith("datetime"):
            # tested below
            return
        elif from_key in ["datetime64[ns, US/Eastern]", "datetime64[ns, UTC]"]:
            # tested below
            return

        if (from_key == "float64" and to_key in ("int64")) or (
            from_key == "complex128" and to_key in ("int64", "float64")
        ):
            if not IS64 or is_platform_windows():
                pytest.skip(f"32-bit platform buggy: {from_key} -> {to_key}")

            # Expected: do not downcast by replacement
            exp = pd.Series(self.rep[to_key], index=index, name="yyy", dtype=from_key)

        else:
            exp = pd.Series(self.rep[to_key], index=index, name="yyy")

        result = obj.replace(replacer)
        tm.assert_series_equal(result, exp, check_dtype=False)

    @pytest.mark.parametrize(
        "to_key",
        ["timedelta64[us]", "bool", "object", "complex128", "float64", "int64"],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "from_key", ["datetime64[ns, UTC]", "datetime64[ns, US/Eastern]"], indirect=True
    )
    def test_replace_series_datetime_tz(
        self, to_key, from_key, replacer, using_infer_string
    ):
        # [FUNCTION DOC]
        # Parametrized with indirect fixtures: source is a timezone-aware datetime dtype in UTC or US
        # Eastern; target one of timedelta, bool, object, complex, float, integer. Replacement result
        # equals the series of target values (dtype not checked). When the infer-string option is on and
        # the target is object, the expected dtype is string.
        index = pd.Index([3, 4], name="xyz")
        obj = pd.Series(self.rep[from_key], index=index, name="yyy").dt.as_unit("ns")
        assert obj.dtype == from_key

        exp = pd.Series(self.rep[to_key], index=index, name="yyy")
        if using_infer_string and to_key == "object":
            assert exp.dtype == "string"
        else:
            assert exp.dtype == to_key

        result = obj.replace(replacer)
        tm.assert_series_equal(result, exp, check_dtype=False)

    @pytest.mark.parametrize(
        "to_key",
        ["datetime64[ns]", "datetime64[ns, UTC]", "datetime64[ns, US/Eastern]"],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "from_key",
        ["datetime64[ns]", "datetime64[ns, UTC]", "datetime64[ns, US/Eastern]"],
        indirect=True,
    )
    def test_replace_series_datetime_datetime(self, to_key, from_key, replacer):
        # [FUNCTION DOC]
        # Parametrized with indirect fixtures: all nine pairs of naive / UTC / US Eastern datetime
        # dtypes. Replacement result equals expected; when both are timezone-aware the original dtype is
        # retained (as of 2.0), and for identical dtypes the expected is converted to nanosecond unit.
        index = pd.Index([3, 4], name="xyz")
        obj = pd.Series(self.rep[from_key], index=index, name="yyy").dt.as_unit("ns")
        assert obj.dtype == from_key

        exp = pd.Series(self.rep[to_key], index=index, name="yyy")
        if isinstance(obj.dtype, pd.DatetimeTZDtype) and isinstance(
            exp.dtype, pd.DatetimeTZDtype
        ):
            # with mismatched tzs, we retain the original dtype as of 2.0
            exp = exp.astype(obj.dtype)
        elif to_key == from_key:
            exp = exp.dt.as_unit("ns")

        result = obj.replace(replacer)
        tm.assert_series_equal(result, exp, check_dtype=False)

    @pytest.mark.xfail(reason="Test not implemented")
    def test_replace_series_period(self):
        # [FUNCTION DOC]
        # Placeholder, expected to fail; raises not-implemented.
        raise NotImplementedError
