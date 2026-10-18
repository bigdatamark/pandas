# [MODULE DOC]
# Tests for Series.replace (and a few frame equivalents): None as the value,
# lists, dicts, scalar-to-scalar, regex, categorical and extension dtypes,
# dtype preservation or upcasting, in-place operation, cascading semantics and
# argument validation errors. Some tests need pyarrow. Uses the
# any_int_ea_dtype, nullable_string_dtype, fixed_now_ts, frame_or_series and
# any_int_numpy_dtype fixtures. No import-time side effects.

import re

import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray


class TestSeriesReplace:
    # [CLASS DOC]
    # Groups the tests of Series.replace. Stateless.
    def test_replace_explicit_none(self):
        # [FUNCTION DOC]
        # GH#36984, GH#19998: an explicit None value is honored as the replacement (not
        # treated as a missing argument) for object Series and for a frame with an object column.
        # GH#36984 if the user explicitly passes value=None, give it to them
        ser = pd.Series([0, 0, ""], dtype=object)
        result = ser.replace("", None)
        expected = pd.Series([0, 0, None], dtype=object)
        tm.assert_series_equal(result, expected)

        # Cast column 2 to object to avoid implicit cast when setting entry to ""
        df = pd.DataFrame(np.zeros((3, 3))).astype({2: object})
        df.iloc[2, 2] = ""
        result = df.replace("", None)
        expected = pd.DataFrame(
            {
                0: np.zeros(3),
                1: np.zeros(3),
                2: np.array([0.0, 0.0, None], dtype=object),
            }
        )
        assert expected.iloc[2, 2] is None
        tm.assert_frame_equal(result, expected)

        # GH#19998 same thing with object dtype
        ser = pd.Series([10, 20, 30, "a", "a", "b", "a"])
        result = ser.replace("a", None)
        expected = pd.Series([10, 20, 30, None, None, "b", None])
        assert expected.iloc[-1] is None
        tm.assert_series_equal(result, expected)

    def test_replace_noop_doesnt_downcast(self):
        # [FUNCTION DOC]
        # GH#44498: replacing NaN with None in an object Series that already holds None
        # and a timestamp is a no-op and keeps object dtype, with both the dict and the
        # two-argument forms.
        # GH#44498
        ser = pd.Series([None, None, pd.Timestamp("2021-12-16 17:31")], dtype=object)
        res = ser.replace({np.nan: None})  # should be a no-op
        tm.assert_series_equal(res, ser)
        assert res.dtype == object

        # same thing but different calling convention
        res = ser.replace(np.nan, None)
        tm.assert_series_equal(res, ser)
        assert res.dtype == object

    def test_replace(self):
        # [FUNCTION DOC]
        # Replacing a list of NaN with a value in place returns the Series itself and
        # equals fillna. Replacing zero by NaN equals assignment. For an object Series
        # with NaN and strings: lists to one value, a dict of values and two lists give
        # the same results and do not change the source unless inplace is used (which
        # returns the Series).
        N = 50
        ser = pd.Series(np.random.default_rng(2).standard_normal(N))
        ser[0:4] = np.nan
        ser[6:10] = 0

        # replace list with a single value
        result = ser.replace([np.nan], -1, inplace=True)
        assert result is ser

        exp = ser.fillna(-1)
        tm.assert_series_equal(ser, exp)

        rs = ser.replace(0.0, np.nan)
        ser[ser == 0.0] = np.nan
        tm.assert_series_equal(rs, ser)

        ser = pd.Series(
            np.fabs(np.random.default_rng(2).standard_normal(N)),
            pd.date_range("2020-01-01", periods=N),
            dtype=object,
        )
        ser[:5] = np.nan
        ser[6:10] = "foo"
        ser[20:30] = "bar"

        # replace list with a single value

        rs = ser.replace([np.nan, "foo", "bar"], -1)

        assert (rs[:5] == -1).all()
        assert (rs[6:10] == -1).all()
        assert (rs[20:30] == -1).all()
        assert (pd.isna(ser[:5])).all()

        # replace with different values
        rs = ser.replace({np.nan: -1, "foo": -2, "bar": -3})

        assert (rs[:5] == -1).all()
        assert (rs[6:10] == -2).all()
        assert (rs[20:30] == -3).all()
        assert (pd.isna(ser[:5])).all()

        # replace with different values with 2 lists
        rs2 = ser.replace([np.nan, "foo", "bar"], [-1, -2, -3])
        tm.assert_series_equal(rs, rs2)

        # replace inplace
        result = ser.replace([np.nan, "foo", "bar"], -1, inplace=True)
        assert result is ser

        assert (ser[:5] == -1).all()
        assert (ser[6:10] == -1).all()
        assert (ser[20:30] == -1).all()

    def test_replace_nan_with_inf(self):
        # [FUNCTION DOC]
        # Replacing NaN with 0 equals fillna for a float Series with an infinity and for
        # a mixed Series; replacing the infinity changes only that entry.
        ser = pd.Series([np.nan, 0, np.inf])
        tm.assert_series_equal(ser.replace(np.nan, 0), ser.fillna(0))

        ser = pd.Series([np.nan, 0, "foo", "bar", np.inf, None, pd.NaT])
        tm.assert_series_equal(ser.replace(np.nan, 0), ser.fillna(0))
        filled = ser.copy()
        filled[4] = 0
        tm.assert_series_equal(ser.replace(np.inf, 0), filled)

    def test_replace_listlike_value_listlike_target(self, datetime_series):
        # [FUNCTION DOC]
        # A datetime Series: replacing NaN is like fillna; lists of different length
        # raise a value error; replacing integers in a datetime Series is a no-op; a
        # permutation of integers swaps all values.
        ser = pd.Series(datetime_series.index)
        tm.assert_series_equal(ser.replace(np.nan, 0), ser.fillna(0))

        # malformed
        msg = r"Replacement lists must match in length\. Expecting 3 got 2"
        with pytest.raises(ValueError, match=msg):
            ser.replace([1, 2, 3], [np.nan, 0])

        # ser is dt64 so can't hold 1 or 2, so this replace is a no-op
        result = ser.replace([1, 2], [np.nan, 0])
        tm.assert_series_equal(result, ser)

        ser = pd.Series([0, 1, 2, 3, 4])
        result = ser.replace([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
        tm.assert_series_equal(result, pd.Series([4, 3, 2, 1, 0]))

    def test_replace_gh5319(self):
        # [FUNCTION DOC]
        # GH 5319: replace with only a list-like target and no value raises a value
        # error saying a value, dict-like target or dict-like regex must be given.
        # API change from 0.12?
        # GH 5319
        ser = pd.Series([0, np.nan, 2, 3, 4])
        msg = (
            "Series.replace must specify either 'value', "
            "a dict-like 'to_replace', or dict-like 'regex'"
        )
        with pytest.raises(ValueError, match=msg):
            ser.replace([np.nan])

        with pytest.raises(ValueError, match=msg):
            ser.replace(np.nan)

    def test_replace_datetime64(self):
        # [FUNCTION DOC]
        # GH 5797: replacing a Timestamp with a dict or two scalars works for a datetime
        # Series.
        # GH 5797
        ser = pd.Series(pd.date_range("20130101", periods=5))
        expected = ser.copy()
        expected.loc[2] = pd.Timestamp("20120101")
        result = ser.replace({pd.Timestamp("20130103"): pd.Timestamp("20120101")})
        tm.assert_series_equal(result, expected)
        result = ser.replace(pd.Timestamp("20130103"), pd.Timestamp("20120101"))
        tm.assert_series_equal(result, expected)

    def test_replace_nat_with_tz(self):
        # [FUNCTION DOC]
        # GH 11792: replacing NaN and NaT with the minimum Timestamp in a tz-aware
        # Series gives object dtype holding both values.
        # GH 11792: Test with replacing NaT in a list with tz data
        ts = pd.Timestamp("2015/01/01", tz="UTC")
        s = pd.Series([pd.NaT, pd.Timestamp("2015/01/01", tz="UTC")])
        result = s.replace([np.nan, pd.NaT], pd.Timestamp.min)
        expected = pd.Series([pd.Timestamp.min, ts], dtype=object)
        tm.assert_series_equal(expected, result)

    def test_replace_timedelta_td64(self):
        # [FUNCTION DOC]
        # Replacing one Timedelta with another via a single dict works (goes through the
        # list-replacement path).
        tdi = pd.timedelta_range(0, periods=5)
        ser = pd.Series(tdi)

        # Using a single dict argument means we go through replace_list
        result = ser.replace({ser[1]: ser[3]})

        expected = pd.Series([ser[0], ser[3], ser[2], ser[3], ser[4]])
        tm.assert_series_equal(result, expected)

    def test_replace_with_single_list(self):
        # [FUNCTION DOC]
        # Replacing with a single list and no value raises a value error with the same
        # message, also for the in-place form.
        ser = pd.Series([0, 1, 2, 3, 4])
        msg = (
            "Series.replace must specify either 'value', "
            "a dict-like 'to_replace', or dict-like 'regex'"
        )
        with pytest.raises(ValueError, match=msg):
            ser.replace([1, 2, 3])

        s = ser.copy()
        with pytest.raises(ValueError, match=msg):
            s.replace([1, 2, 3], inplace=True)

    def test_replace_mixed_types(self):
        # [FUNCTION DOC]
        # Integer Series with mixed-type replacements, via a local checking helper: a
        # float that fits the integer dtype does not upcast (GH#44940), a non-integer float
        # upcasts, strings, timestamps and booleans upcast to object. Also a datetime
        # object Series replaced with floats, an integer and a string.
        ser = pd.Series(np.arange(5), dtype="int64")

        def check_replace(to_rep, val, expected):
            # [FUNCTION DOC]
            # HELPER: given the targets, values and expected Series, runs replace both
            # non-in-place and in place on copies and asserts the in-place call returns the
            # copy itself and both equal the expected result.
            # KNOWN ISSUE: the first replace result is assigned to the result variable
            # and immediately overwritten by the in-place call, so the non-in-place
            # outcome is never checked.
            sc = ser.copy()
            result = ser.replace(to_rep, val)
            result = sc.replace(to_rep, val, inplace=True)
            assert result is sc
            tm.assert_series_equal(expected, result)
            tm.assert_series_equal(expected, sc)

        # 3.0 can still be held in our int64 series, so we do not upcast GH#44940
        tr, v = [3], [3.0]
        check_replace(tr, v, ser)
        # Note this matches what we get with the scalars 3 and 3.0
        check_replace(tr[0], v[0], ser)

        # MUST upcast to float
        e = pd.Series([0, 1, 2, 3.5, 4])
        tr, v = [3], [3.5]
        check_replace(tr, v, e)

        # casts to object
        e = pd.Series([0, 1, 2, 3.5, "a"])
        tr, v = [3, 4], [3.5, "a"]
        check_replace(tr, v, e)

        # again casts to object
        e = pd.Series([0, 1, 2, 3.5, pd.Timestamp("20130101")])
        tr, v = [3, 4], [3.5, pd.Timestamp("20130101")]
        check_replace(tr, v, e)

        # casts to object
        e = pd.Series([0, 1, 2, 3.5, True], dtype="object")
        tr, v = [3, 4], [3.5, True]
        check_replace(tr, v, e)

        # test an object with dates + floats + integers + strings
        dr = pd.Series(pd.date_range("1/1/2001", "1/10/2001", freq="D"))
        result = dr.astype(object).replace([dr[0], dr[1], dr[2]], [1.0, 2, "a"])
        expected = pd.Series([1.0, 2, "a", *dr[3:].tolist()], dtype=object)
        tm.assert_series_equal(result, expected)

    def test_replace_bool_with_string_no_op(self):
        # [FUNCTION DOC]
        # Replacing a missing string in a boolean Series is a no-op.
        s = pd.Series([True, False, True])
        result = s.replace("fun", "in-the-sun")
        tm.assert_series_equal(s, result)

    def test_replace_bool_with_string(self):
        # [FUNCTION DOC]
        # Replacing True with a string in a boolean Series gives object dtype with the
        # string and the untouched False.
        # KNOWN ISSUE: the comment says "nonexistent elements" though the replaced
        # element does exist in the Series.
        # nonexistent elements
        s = pd.Series([True, False, True])
        result = s.replace(True, "2u")
        expected = pd.Series(["2u", False, "2u"])
        tm.assert_series_equal(expected, result)

    def test_replace_bool_with_bool(self):
        # [FUNCTION DOC]
        # Replacing True with False in a boolean Series gives all False.
        s = pd.Series([True, False, True])
        result = s.replace(True, False)
        expected = pd.Series([False] * len(s))
        tm.assert_series_equal(expected, result)

    def test_replace_with_dict_with_bool_keys(self):
        # [FUNCTION DOC]
        # A dict with a boolean key and an unrelated string key replaces the boolean values
        # only.
        s = pd.Series([True, False, True])
        result = s.replace({"asdf": "asdb", True: "yes"})
        expected = pd.Series(["yes", False, "yes"])
        tm.assert_series_equal(result, expected)

    def test_replace_Int_with_na(self, any_int_ea_dtype):
        # [FUNCTION DOC]
        # GH 38267, parametrized over nullable integer dtypes: replacing a value with NA
        # keeps dtype, including in place (returns the Series).
        # GH 38267
        result = pd.Series([0, None], dtype=any_int_ea_dtype).replace(0, pd.NA)
        expected = pd.Series([pd.NA, pd.NA], dtype=any_int_ea_dtype)
        tm.assert_series_equal(result, expected)
        result = pd.Series([0, 1], dtype=any_int_ea_dtype).replace(0, pd.NA)
        result2 = result.replace(1, pd.NA, inplace=True)
        assert result2 is result
        tm.assert_series_equal(result, expected)

    def test_replace2(self):
        # [FUNCTION DOC]
        # Same scenarios as the second half of the replace test with an object Series on a
        # date index (lists, dict, two lists, in place).
        # KNOWN ISSUE: this repeats the object-Series part of the replace test
        # almost verbatim.
        N = 50
        ser = pd.Series(
            np.fabs(np.random.default_rng(2).standard_normal(N)),
            pd.date_range("2020-01-01", periods=N),
            dtype=object,
        )
        ser[:5] = np.nan
        ser[6:10] = "foo"
        ser[20:30] = "bar"

        # replace list with a single value
        rs = ser.replace([np.nan, "foo", "bar"], -1)

        assert (rs[:5] == -1).all()
        assert (rs[6:10] == -1).all()
        assert (rs[20:30] == -1).all()
        assert (pd.isna(ser[:5])).all()

        # replace with different values
        rs = ser.replace({np.nan: -1, "foo": -2, "bar": -3})

        assert (rs[:5] == -1).all()
        assert (rs[6:10] == -2).all()
        assert (rs[20:30] == -3).all()
        assert (pd.isna(ser[:5])).all()

        # replace with different values with 2 lists
        rs2 = ser.replace([np.nan, "foo", "bar"], [-1, -2, -3])
        tm.assert_series_equal(rs, rs2)

        # replace inplace
        result = ser.replace([np.nan, "foo", "bar"], -1, inplace=True)
        assert result is ser
        assert (ser[:5] == -1).all()
        assert (ser[6:10] == -1).all()
        assert (ser[20:30] == -1).all()

    @pytest.mark.parametrize("inplace", [True, False])
    def test_replace_cascade(self, inplace):
        # [FUNCTION DOC]
        # GH #50778, parametrized over inplace: replacing a list of values by a list
        # shifted by one is not cascaded (replaced values are not replaced again).
        # Test that replaced values are not replaced again
        # GH #50778
        ser = pd.Series([1, 2, 3])
        expected = pd.Series([2, 3, 4])

        res = ser.replace([1, 2, 3], [2, 3, 4], inplace=inplace)
        if inplace:
            tm.assert_series_equal(ser, expected)
        else:
            tm.assert_series_equal(res, expected)

    def test_replace_with_dictlike_and_string_dtype(self, nullable_string_dtype):
        # [FUNCTION DOC]
        # GH 32621, GH#44940: dict replacement on a nullable string Series keeps the
        # dtype and the missing value.
        # GH 32621, GH#44940
        ser = pd.Series(["one", "two", np.nan], dtype=nullable_string_dtype)
        expected = pd.Series(["1", "2", np.nan], dtype=nullable_string_dtype)
        result = ser.replace({"one": "1", "two": "2"})
        tm.assert_series_equal(expected, result)

    def test_replace_with_empty_dictlike(self):
        # [FUNCTION DOC]
        # GH 15289: replacing with an empty dict or empty Series leaves the Series
        # unchanged.
        # GH 15289
        s = pd.Series(list("abcd"))
        tm.assert_series_equal(s, s.replace({}))

        empty_series = pd.Series([])
        tm.assert_series_equal(s, s.replace(empty_series))

    def test_replace_string_with_number(self):
        # [FUNCTION DOC]
        # GH 15743: replacing the string "2" in an integer Series does nothing.
        # GH 15743
        s = pd.Series([1, 2, 3])
        result = s.replace("2", np.nan)
        expected = pd.Series([1, 2, 3])
        tm.assert_series_equal(expected, result)

    def test_replace_replacer_equals_replacement(self):
        # [FUNCTION DOC]
        # GH 20656: replacers are matched against the original values, so swapping two
        # values works.
        # GH 20656
        # make sure all replacers are matching against original values
        s = pd.Series(["a", "b"])
        expected = pd.Series(["b", "a"])
        result = s.replace({"a": "b", "b": "a"})
        tm.assert_series_equal(expected, result)

    def test_replace_unicode_with_number(self):
        # [FUNCTION DOC]
        # GH 15743: same as the string-number test.
        # KNOWN ISSUE: the body is identical to the test for replacing a string with a
        # number, so the "unicode" in the name does not reflect anything tested.
        # GH 15743
        s = pd.Series([1, 2, 3])
        result = s.replace("2", np.nan)
        expected = pd.Series([1, 2, 3])
        tm.assert_series_equal(expected, result)

    def test_replace_mixed_types_with_string(self):
        # [FUNCTION DOC]
        # Mixed Series with integers and a string; replacing a number and a string with NaN
        # gives object dtype with NaN at those positions.
        # Testing mixed
        s = pd.Series([1, 2, 3, "4", 4, 5])
        result = s.replace([2, "4"], np.nan)
        expected = pd.Series([1, np.nan, 3, np.nan, 4, 5], dtype=object)
        tm.assert_series_equal(expected, result)

    @pytest.mark.parametrize(
        "categorical, numeric",
        [
            (["A"], [1]),
            (["A", "B"], [1, 2]),
        ],
    )
    def test_replace_categorical(self, categorical, numeric):
        # [FUNCTION DOC]
        # GH 24971, GH#23305, parametrized over one or two categories: renaming the
        # categories to numbers gives the expected categorical with categories 1 and 2
        # (an unused category is added).
        # KNOWN ISSUE: the test does not call replace; it exercises the rename of
        # categories.
        # GH 24971, GH#23305
        ser = pd.Series(pd.Categorical(categorical, categories=["A", "B"]))
        result = ser.cat.rename_categories({"A": 1, "B": 2})
        expected = pd.Series(numeric).astype("category")
        if 2 not in expected.cat.categories:
            # i.e. categories should be [1, 2] even if there are no "B"s present
            # GH#44940
            expected = expected.cat.add_categories(2)
        tm.assert_series_equal(expected, result, check_categorical=False)

    def test_replace_categorical_inplace(self):
        # [FUNCTION DOC]
        # GH 53358: in-place replace on a categorical Series returns the Series and
        # changes the value while keeping categories.
        # GH 53358
        data = ["a", "b", "c"]
        data_exp = ["b", "b", "c"]
        ser = pd.Series(data, dtype="category")
        result = ser.replace(to_replace="a", value="b", inplace=True)
        assert result is ser
        expected = pd.Series(pd.Categorical(data_exp, categories=data))
        tm.assert_series_equal(ser, expected)

    def test_replace_categorical_single(self):
        # [FUNCTION DOC]
        # GH 26988: renaming the category that is a timestamp (tz-aware categories) to a
        # string gives the expected categorical and does not change the original.
        # KNOWN ISSUE: the test does not call replace; it exercises the rename of
        # categories.
        # GH 26988
        dti = pd.date_range("2016-01-01", periods=3, tz="US/Pacific")
        s = pd.Series(dti)
        c = s.astype("category")

        expected = c.copy()
        expected = expected.cat.add_categories("foo")
        expected[2] = "foo"
        expected = expected.cat.remove_unused_categories()
        assert c[2] != "foo"

        result = c.cat.rename_categories({c.values[2]: "foo"})
        tm.assert_series_equal(expected, result)
        assert c[2] != "foo"  # ensure non-inplace call does not alter original

    def test_replace_with_no_overflowerror(self):
        # [FUNCTION DOC]
        # GH 25616: replacing an integer with a very large number given as string avoids
        # an overflow error and gives object dtype; replacing a very large numeric string
        # with an integer also works.
        # GH 25616
        # casts to object without Exception from OverflowError
        s = pd.Series([0, 1, 2, 3, 4])
        result = s.replace([3], ["100000000000000000000"])
        expected = pd.Series([0, 1, 2, "100000000000000000000", 4])
        tm.assert_series_equal(result, expected)

        s = pd.Series([0, "100000000000000000000", "100000000000000000001"])
        result = s.replace(["100000000000000000000"], [1])
        expected = pd.Series([0, 1, "100000000000000000001"])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "ser, to_replace, exp",
        [
            ([1, 2, 3], {1: 2, 2: 3, 3: 4}, [2, 3, 4]),
            (["1", "2", "3"], {"1": "2", "2": "3", "3": "4"}, ["2", "3", "4"]),
        ],
    )
    def test_replace_commutative(self, ser, to_replace, exp):
        # [FUNCTION DOC]
        # GH 16051, parametrized over numeric and string data: replacement with a dict
        # whose values are keys of other entries is applied once, to original values.
        # GH 16051
        # DataFrame.replace() overwrites when values are non-numeric

        series = pd.Series(ser)

        expected = pd.Series(exp)
        result = series.replace(to_replace)

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "ser, exp", [([1, 2, 3], [1, True, 3]), (["x", 2, 3], ["x", True, 3])]
    )
    def test_replace_no_cast(self, ser, exp):
        # [FUNCTION DOC]
        # GH 9113, parametrized over integers and mixed data: replacing an integer with
        # True does not coerce the boolean to an integer.
        # GH 9113
        # BUG: replace int64 dtype with bool coerces to int64

        series = pd.Series(ser)
        result = series.replace(2, True)
        expected = pd.Series(exp)

        tm.assert_series_equal(result, expected)

    def test_replace_invalid_to_replace(self):
        # [FUNCTION DOC]
        # GH 18634: passing a callable as the target raises a type error listing the
        # accepted types.
        # GH 18634
        # API: replace() should raise an exception if invalid argument is given
        series = pd.Series(["a", "b", "c "])
        msg = (
            r"Expecting 'to_replace' to be either a scalar, array-like, "
            r"dict or None, got invalid type.*"
        )
        with pytest.raises(TypeError, match=msg):
            series.replace(lambda x: x.strip())

    @pytest.mark.parametrize("frame", [False, True])
    def test_replace_nonbool_regex(self, frame):
        # [FUNCTION DOC]
        # Parametrized over Series or frame: a non-boolean regex together with a target
        # list raises a value error that the target must be None.
        obj = pd.Series(["a", "b", "c "])
        if frame:
            obj = obj.to_frame()

        msg = "'to_replace' must be 'None' if 'regex' is not a bool"
        with pytest.raises(ValueError, match=msg):
            obj.replace(to_replace=["a"], regex="foo")

    @pytest.mark.parametrize("frame", [False, True])
    def test_replace_empty_copy(self, frame):
        # [FUNCTION DOC]
        # Parametrized over Series or frame: replacing in an empty float object in place
        # returns the object itself; without inplace returns an equal new object.
        obj = pd.Series([], dtype=np.float64)
        if frame:
            obj = obj.to_frame()

        res = obj.replace(4, 5, inplace=True)
        assert res is obj

        res = obj.replace(4, 5, inplace=False)
        tm.assert_equal(res, obj)
        assert res is not obj

    def test_replace_only_one_dictlike_arg(self, fixed_now_ts):
        # [FUNCTION DOC]
        # GH#33340: giving both a dict-like target and a value, or a scalar target with
        # a dict-like value, raises a value error with the corresponding message.
        # GH#33340

        ser = pd.Series([1, 2, "A", fixed_now_ts, True])
        to_replace = {0: 1, 2: "A"}
        value = "foo"
        msg = (
            "Series.replace cannot specify both a dict-like 'to_replace' and a 'value'"
        )
        with pytest.raises(ValueError, match=msg):
            ser.replace(to_replace, value)

        to_replace = 1
        value = {0: "foo", 2: "bar"}
        msg = "Series.replace cannot use dict-value and non-None to_replace"
        with pytest.raises(ValueError, match=msg):
            ser.replace(to_replace, value)

    def test_replace_dict_like_with_dict_like(self):
        # [FUNCTION DOC]
        # GH 59452: both target and value as Series raise a value error that they cannot
        # be dict-like.
        # GH 59452
        s = pd.Series([1, 2, 3, 4, 5])
        to_replace = pd.Series([1])
        value = pd.Series([75])
        msg = "to_replace and value cannot be dict-like for Series.replace"
        with pytest.raises(ValueError, match=msg):
            s.replace(to_replace, value)

    def test_replace_extension_other(self, frame_or_series):
        # [FUNCTION DOC]
        # GitHub issue 34530: replacing an empty string by an empty string in an Int64 frame or
        # Series raises nothing and keeps the object unchanged.
        # https://github.com/pandas-dev/pandas/issues/34530
        obj = frame_or_series(pd.array([1, 2, 3], dtype="Int64"))
        result = obj.replace("", "")  # no exception
        # should not have changed dtype
        tm.assert_equal(obj, result)

    def test_replace_with_compiled_regex(self):
        # [FUNCTION DOC]
        # GitHub issue 35680: a compiled regular expression as a dict key with the regex
        # option replaces matching strings.
        # https://github.com/pandas-dev/pandas/issues/35680
        s = pd.Series(["a", "b", "c"])
        regex = re.compile("^a$")
        result = s.replace({regex: "z"}, regex=True)
        expected = pd.Series(["z", "b", "c"])
        tm.assert_series_equal(result, expected)

    def test_pandas_replace_na(self):
        # [FUNCTION DOC]
        # GH#43344, GH#56599: regex mapping on a string dtype Series replaces patterns
        # simultaneously (CC becomes CC-REPL) and preserves NA and the empty string.
        # GH#43344
        # GH#56599
        ser = pd.Series(["AA", "BB", "CC", "DD", "EE", "", pd.NA, "AA"], dtype="string")
        regex_mapping = {
            "AA": "CC",
            "BB": "CC",
            "EE": "CC",
            "CC": "CC-REPL",
        }
        result = ser.replace(regex_mapping, regex=True)
        exp = pd.Series(
            ["CC", "CC", "CC-REPL", "DD", "CC", "", pd.NA, "CC"], dtype="string"
        )
        tm.assert_series_equal(result, exp)

    @pytest.mark.parametrize(
        "dtype, input_data, to_replace, expected_data",
        [
            ("bool", [True, False], {True: False}, [False, False]),
            ("int64", [1, 2], {1: 10, 2: 20}, [10, 20]),
            ("Int64", [1, 2], {1: 10, 2: 20}, [10, 20]),
            ("float64", [1.1, 2.2], {1.1: 10.1, 2.2: 20.5}, [10.1, 20.5]),
            ("Float64", [1.1, 2.2], {1.1: 10.1, 2.2: 20.5}, [10.1, 20.5]),
            ("string", ["one", "two"], {"one": "1", "two": "2"}, ["1", "2"]),
            (
                pd.IntervalDtype("int64"),
                IntervalArray([pd.Interval(1, 2), pd.Interval(2, 3)]),
                {pd.Interval(1, 2): pd.Interval(10, 20)},
                IntervalArray([pd.Interval(10, 20), pd.Interval(2, 3)]),
            ),
            (
                pd.IntervalDtype("float64"),
                IntervalArray([pd.Interval(1.0, 2.7), pd.Interval(2.8, 3.1)]),
                {pd.Interval(1.0, 2.7): pd.Interval(10.6, 20.8)},
                IntervalArray([pd.Interval(10.6, 20.8), pd.Interval(2.8, 3.1)]),
            ),
            (
                pd.PeriodDtype("M"),
                [pd.Period("2020-05", freq="M")],
                {pd.Period("2020-05", freq="M"): pd.Period("2020-06", freq="M")},
                [pd.Period("2020-06", freq="M")],
            ),
        ],
    )
    def test_replace_dtype(self, dtype, input_data, to_replace, expected_data):
        # [FUNCTION DOC]
        # GH#33484, parametrized over dtype, input data, replacement dict and expected
        # data (bool, ints, floats, string, interval, period): replacing by dict keeps
        # the dtype and gives the expected values.
        # GH#33484
        ser = pd.Series(input_data, dtype=dtype)
        result = ser.replace(to_replace)
        expected = pd.Series(expected_data, dtype=dtype)
        tm.assert_series_equal(result, expected)

    def test_replace_string_dtype(self):
        # [FUNCTION DOC]
        # GH#40732, GH#44940, GH#31644: replace on the string dtype with a dict, a scalar
        # (keeping NaN as missing) and with NA keeps the string dtype.
        # GH#40732, GH#44940
        ser = pd.Series(["one", "two", np.nan], dtype="string")
        res = ser.replace({"one": "1", "two": "2"})
        expected = pd.Series(["1", "2", np.nan], dtype="string")
        tm.assert_series_equal(res, expected)

        # GH#31644
        ser2 = pd.Series(["A", np.nan], dtype="string")
        res2 = ser2.replace("A", "B")
        expected2 = pd.Series(["B", np.nan], dtype="string")
        tm.assert_series_equal(res2, expected2)

        ser3 = pd.Series(["A", "B"], dtype="string")
        res3 = ser3.replace("A", pd.NA)
        expected3 = pd.Series([pd.NA, "B"], dtype="string")
        tm.assert_series_equal(res3, expected3)

    def test_replace_string_dtype_list_to_replace(self):
        # [FUNCTION DOC]
        # GH#41215, GH#44940: replacing a list of strings (one absent) with a string on
        # the string dtype.
        # GH#41215, GH#44940
        ser = pd.Series(["abc", "def"], dtype="string")
        res = ser.replace(["abc", "any other string"], "xyz")
        expected = pd.Series(["xyz", "def"], dtype="string")
        tm.assert_series_equal(res, expected)

    def test_replace_string_dtype_regex(self):
        # [FUNCTION DOC]
        # GH#31644: the regex option on the string dtype replaces every character
        # string with the replacement.
        # GH#31644
        ser = pd.Series(["A", "B"], dtype="string")
        res = ser.replace(r".", "C", regex=True)
        expected = pd.Series(["C", "C"], dtype="string")
        tm.assert_series_equal(res, expected)

    def test_replace_nullable_numeric(self):
        # [FUNCTION DOC]
        # GH#40732, GH#44940: for nullable Float64 and Int64, replace keeps the dtype for
        # numeric replacements (also float replacing integers when exactly
        # representable); a non-integer float for an Int64 raises a type error about
        # an invalid value.
        # GH#40732, GH#44940

        floats = pd.Series([1.0, 2.0, 3.999, 4.4], dtype=pd.Float64Dtype())
        assert floats.replace({1.0: 9}).dtype == floats.dtype
        assert floats.replace(1.0, 9).dtype == floats.dtype
        assert floats.replace({1.0: 9.0}).dtype == floats.dtype
        assert floats.replace(1.0, 9.0).dtype == floats.dtype

        res = floats.replace(to_replace=[1.0, 2.0], value=[9.0, 10.0])
        assert res.dtype == floats.dtype

        ints = pd.Series([1, 2, 3, 4], dtype=pd.Int64Dtype())
        assert ints.replace({1: 9}).dtype == ints.dtype
        assert ints.replace(1, 9).dtype == ints.dtype
        assert ints.replace({1: 9.0}).dtype == ints.dtype
        assert ints.replace(1, 9.0).dtype == ints.dtype

        # nullable (for now) raises instead of casting
        with pytest.raises(TypeError, match="Invalid value"):
            ints.replace({1: 9.5})
        with pytest.raises(TypeError, match="Invalid value"):
            ints.replace(1, 9.5)

    @pytest.mark.parametrize("regex", [False, True])
    def test_replace_regex_dtype_series(self, regex):
        # [FUNCTION DOC]
        # GH-48644, parametrized over the regex option: replacing a string with an integer
        # in an object Series gives the integer.
        # GH-48644
        series = pd.Series(["0"], dtype=object)
        expected = pd.Series([1], dtype=object)
        result = series.replace(to_replace="0", value=1, regex=regex)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("regex", [False, True])
    def test_replace_regex_dtype_series_string(self, regex):
        # [FUNCTION DOC]
        # Parametrized over the regex option: replacing a string with an integer in a
        # string dtype Series gives an object Series with the integer.
        series = pd.Series(["0"], dtype="str")
        expected = pd.Series([1], dtype=object)
        result = series.replace(to_replace="0", value=1, regex=regex)
        tm.assert_series_equal(result, expected)

    def test_replace_different_int_types(self, any_int_numpy_dtype):
        # [FUNCTION DOC]
        # GH#45311, parametrized over integer dtypes: replace with a dict built from the
        # values of a Series of one dtype works the same as with the equivalent plain
        # dict.
        # GH#45311
        labs = pd.Series([1, 1, 1, 0, 0, 2, 2, 2], dtype=any_int_numpy_dtype)

        maps = pd.Series([0, 2, 1], dtype=any_int_numpy_dtype)
        map_dict = dict(zip(maps.values, maps.index, strict=True))

        result = labs.replace(map_dict)
        expected = labs.replace({0: 0, 2: 1, 1: 2})
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("val", [2, np.nan, 2.0])
    def test_replace_value_none_dtype_numeric(self, val):
        # [FUNCTION DOC]
        # GH#48231, parametrized over 2, NaN and 2.0: replacing a value with None gives
        # an object Series with None.
        # GH#48231
        ser = pd.Series([1, val])
        result = ser.replace(val, None)
        expected = pd.Series([1, None], dtype=object)
        tm.assert_series_equal(result, expected)

    def test_replace_change_dtype_series(self):
        # [FUNCTION DOC]
        # GH#25797: replacing the value True in an object column by NaN and replacing None
        # with NaN both give the same frame as filling the missing value with NaN.
        # GH#25797
        df = pd.DataFrame({"Test": ["0.5", True, "0.6"]}, dtype=object)
        df["Test"] = df["Test"].replace([True], [np.nan])
        expected = pd.DataFrame({"Test": ["0.5", np.nan, "0.6"]}, dtype=object)
        tm.assert_frame_equal(df, expected)

        df = pd.DataFrame({"Test": ["0.5", None, "0.6"]}, dtype=object)
        df["Test"] = df["Test"].replace([None], [np.nan])
        tm.assert_frame_equal(df, expected)

        df = pd.DataFrame({"Test": ["0.5", None, "0.6"]}, dtype=object)
        df["Test"] = df["Test"].fillna(np.nan)
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("dtype", ["object", "Int64"])
    def test_replace_na_in_obj_column(self, dtype):
        # [FUNCTION DOC]
        # GH#47480, parametrized over object and Int64: replace keeps NA, including in
        # place which returns the Series itself.
        # GH#47480
        ser = pd.Series([0, 1, pd.NA], dtype=dtype)
        expected = pd.Series([0, 2, pd.NA], dtype=dtype)
        result = ser.replace(to_replace=1, value=2)
        tm.assert_series_equal(result, expected)

        result = ser.replace(to_replace=1, value=2, inplace=True)
        assert result is ser
        tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize("val", [0, 0.5])
    def test_replace_numeric_column_with_na(self, val):
        # [FUNCTION DOC]
        # GH#50758, parametrized over 0 and 0.5: replacing a value with NA in a numeric
        # Series gives NA in the position, also in place.
        # GH#50758
        ser = pd.Series([val, 1])
        expected = pd.Series([val, pd.NA])
        result = ser.replace(to_replace=1, value=pd.NA)
        tm.assert_series_equal(result, expected)

        result = ser.replace(to_replace=1, value=pd.NA, inplace=True)
        assert result is ser
        tm.assert_series_equal(ser, expected)

    def test_replace_ea_float_with_bool(self):
        # [FUNCTION DOC]
        # GH#55398: replacing a boolean in a masked float Series, or a float in a boolean
        # Series, with a nonmatching type is a no-op.
        # GH#55398
        ser = pd.Series([0.0], dtype="Float64")
        expected = ser.copy()
        result = ser.replace(False, 1.0)
        tm.assert_series_equal(result, expected)

        ser = pd.Series([False], dtype="boolean")
        expected = ser.copy()
        result = ser.replace(0.0, True)
        tm.assert_series_equal(result, expected)

    def test_replace_all_NA(self):
        # [FUNCTION DOC]
        # GH#60688: a regex dict replacement on a Series of all NA leaves it unchanged.
        # GH#60688
        df = pd.Series([pd.NA, pd.NA])
        result = df.replace({r"^#": "$"}, regex=True)
        expected = pd.Series([pd.NA, pd.NA])
        tm.assert_series_equal(result, expected)

    def test_replace_mixed_types_with_none(self):
        # [FUNCTION DOC]
        # GH#29813: replacing NaN with None in a mixed object Series puts None in place of
        # NaN.
        # GH#29813
        df = pd.Series([np.nan, 1, "foo"])
        result = df.replace({np.nan: None})
        expected = pd.Series([None, 1, "foo"])
        tm.assert_series_equal(result, expected)


@td.skip_if_no("pyarrow")
def test_replace_from_index():
    # [FUNCTION DOC]
    # GitHub issue 61622, requires pyarrow: building a Series from a pyarrow-string
    # Index and replacing with a dict keeps the dtype and replaces the matching entry.
    # https://github.com/pandas-dev/pandas/issues/61622
    idx = pd.Index(["a", "b", "c"], dtype="string[pyarrow]")
    expected = pd.Series(["d", "b", "c"], dtype="string[pyarrow]")
    result = pd.Series(idx).replace({"z": "b", "a": "d"})
    tm.assert_series_equal(result, expected)
