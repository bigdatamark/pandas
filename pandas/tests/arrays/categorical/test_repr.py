# [MODULE DOC]
# Tests for the string representation of Categorical (and categorical index): value truncation, the
# categories line with ordering symbols, length line, and handling of datetime, period, timedelta, unicode
# and string dtypes. Expected strings are literals; options changed are restored by the option context.
# Uses fixtures using_infer_string and string_dtype_no_object. No import-time side effects.

import numpy as np

from pandas import (
    Categorical,
    CategoricalDtype,
    CategoricalIndex,
    Index,
    Series,
    date_range,
    option_context,
    period_range,
    timedelta_range,
)


class TestCategoricalReprWithFactor:
    # [CLASS DOC]
    # Groups the basic print test, named for the historical Factor type. Stateless.
    def test_print(self, using_infer_string):
        # [FUNCTION DOC]
        # Ordered string Categorical prints values and a categories line with less-than separators; dtype shown as str when string inference is on, else object.
        factor = Categorical(["a", "b", "b", "a", "a", "c", "c", "c"], ordered=True)
        dtype = "str" if using_infer_string else "object"
        expected = [
            "['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c']",
            f"Categories (3, {dtype}): ['a' < 'b' < 'c']",
        ]
        expected = "\n".join(expected)
        actual = repr(factor)
        assert actual == expected


class TestCategoricalRepr:
    # [CLASS DOC]
    # Groups representation tests for Categorical and categorical index. Stateless.
    def test_big_print(self):
        # [FUNCTION DOC]
        # A 600-element Categorical shows the first and last five values, a length line and the categories.
        codes = np.array([0, 1, 2, 0, 1, 2] * 100)
        dtype = CategoricalDtype(categories=Index(["a", "b", "c"], dtype=object))
        factor = Categorical.from_codes(codes, dtype=dtype)
        expected = [
            "['a', 'b', 'c', 'a', 'b', ..., 'b', 'c', 'a', 'b', 'c']",
            "Length: 600",
            "Categories (3, object): ['a', 'b', 'c']",
        ]
        expected = "\n".join(expected)

        actual = repr(factor)

        assert actual == expected

    def test_empty_print(self):
        # [FUNCTION DOC]
        # Empty Categoricals (unordered, ordered, and with no categories) print as empty brackets followed by the
        # categories line.
        # KNOWN ISSUE: the unordered case is asserted twice (the second, reversed comparison is redundant).
        factor = Categorical([], Index(["a", "b", "c"], dtype=object))
        expected = "[], Categories (3, object): ['a', 'b', 'c']"
        actual = repr(factor)
        assert actual == expected

        assert expected == actual
        factor = Categorical([], Index(["a", "b", "c"], dtype=object), ordered=True)
        expected = "[], Categories (3, object): ['a' < 'b' < 'c']"
        actual = repr(factor)
        assert expected == actual

        factor = Categorical([], [])
        expected = "[], Categories (0, object): []"
        assert expected == repr(factor)

    def test_print_none_width(self):
        # [FUNCTION DOC]
        # With the display width option unset, a categorical Series prints in the normal compact form with its categories line.
        # GH10087
        a = Series(Categorical([1, 2, 3, 4]))
        exp = (
            "0    1\n1    2\n2    3\n3    4\n"
            "dtype: category\nCategories (4, int64): [1, 2, 3, 4]"
        )

        with option_context("display.width", None):
            assert exp == repr(a)

    def test_unicode_print(self, using_infer_string):
        # [FUNCTION DOC]
        # ASCII and Japanese string categories print truncated with length and category lines (object or str dtype by
        # fixture); the East Asian width option must not change the Categorical output.
        c = Categorical(["aaaaa", "bb", "cccc"] * 20)
        expected = """\
['aaaaa', 'bb', 'cccc', 'aaaaa', 'bb', ..., 'bb', 'cccc', 'aaaaa', 'bb', 'cccc']
Length: 60
Categories (3, object): ['aaaaa', 'bb', 'cccc']"""

        if using_infer_string:
            expected = expected.replace("object", "str")

        assert repr(c) == expected

        c = Categorical(["ああああ", "いいいいい", "ううううううう"] * 20)
        expected = """\
['ああああ', 'いいいいい', 'ううううううう', 'ああああ', 'いいいいい', ..., 'いいいいい', 'ううううううう', 'ああああ', 'いいいいい', 'ううううううう']
Length: 60
Categories (3, object): ['ああああ', 'いいいいい', 'ううううううう']"""  # noqa: E501

        if using_infer_string:
            expected = expected.replace("object", "str")

        assert repr(c) == expected

        # unicode option should not affect to Categorical, as it doesn't care
        # the repr width
        with option_context("display.unicode.east_asian_width", True):
            c = Categorical(["ああああ", "いいいいい", "ううううううう"] * 20)
            expected = """['ああああ', 'いいいいい', 'ううううううう', 'ああああ', 'いいいいい', ..., 'いいいいい', 'ううううううう', 'ああああ', 'いいいいい', 'ううううううう']
Length: 60
Categories (3, object): ['ああああ', 'いいいいい', 'ううううううう']"""  # noqa: E501

        if using_infer_string:
            expected = expected.replace("object", "str")

        assert repr(c) == expected

    def test_categorical_repr(self):
        # [FUNCTION DOC]
        # Integer Categoricals print short, truncated (fifty elements) and with truncated categories (twenty categories).
        c = Categorical([1, 2, 3])
        exp = """[1, 2, 3]
Categories (3, int64): [1, 2, 3]"""

        assert repr(c) == exp

        c = Categorical([1, 2, 3, 1, 2, 3], categories=[1, 2, 3])
        exp = """[1, 2, 3, 1, 2, 3]
Categories (3, int64): [1, 2, 3]"""

        assert repr(c) == exp

        c = Categorical([1, 2, 3, 4, 5] * 10)
        exp = """[1, 2, 3, 4, 5, ..., 1, 2, 3, 4, 5]
Length: 50
Categories (5, int64): [1, 2, 3, 4, 5]"""

        assert repr(c) == exp

        c = Categorical(np.arange(20, dtype=np.int64))
        exp = """[0, 1, 2, 3, 4, ..., 15, 16, 17, 18, 19]
Length: 20
Categories (20, int64): [0, 1, 2, 3, ..., 16, 17, 18, 19]"""

        assert repr(c) == exp

    def test_categorical_repr_ordered(self):
        # [FUNCTION DOC]
        # Same as the integer test but ordered, using the less-than separator and its truncation marker.
        c = Categorical([1, 2, 3], ordered=True)
        exp = """[1, 2, 3]
Categories (3, int64): [1 < 2 < 3]"""

        assert repr(c) == exp

        c = Categorical([1, 2, 3, 1, 2, 3], categories=[1, 2, 3], ordered=True)
        exp = """[1, 2, 3, 1, 2, 3]
Categories (3, int64): [1 < 2 < 3]"""

        assert repr(c) == exp

        c = Categorical([1, 2, 3, 4, 5] * 10, ordered=True)
        exp = """[1, 2, 3, 4, 5, ..., 1, 2, 3, 4, 5]
Length: 50
Categories (5, int64): [1 < 2 < 3 < 4 < 5]"""

        assert repr(c) == exp

        c = Categorical(np.arange(20, dtype=np.int64), ordered=True)
        exp = """[0, 1, 2, 3, 4, ..., 15, 16, 17, 18, 19]
Length: 20
Categories (20, int64): [0 < 1 < 2 < 3 ... 16 < 17 < 18 < 19]"""

        assert repr(c) == exp

    def test_categorical_repr_datetime(self):
        # [FUNCTION DOC]
        # Naive and US/Eastern datetime Categoricals (single and doubled) print values and wrapped categories lines.
        idx = date_range("2011-01-01 09:00", freq="h", periods=5, unit="ns")
        c = Categorical(idx)

        exp = (
            "[2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00, "
            "2011-01-01 12:00:00, 2011-01-01 13:00:00]\n"
            "Categories (5, datetime64[ns]): [2011-01-01 09:00:00, "
            "2011-01-01 10:00:00, 2011-01-01 11:00:00,\n"
            "                                 2011-01-01 12:00:00, "
            "2011-01-01 13:00:00]"
            ""
        )
        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx)
        exp = (
            "[2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00, "
            "2011-01-01 12:00:00, 2011-01-01 13:00:00, 2011-01-01 09:00:00, "
            "2011-01-01 10:00:00, 2011-01-01 11:00:00, 2011-01-01 12:00:00, "
            "2011-01-01 13:00:00]\n"
            "Categories (5, datetime64[ns]): [2011-01-01 09:00:00, "
            "2011-01-01 10:00:00, 2011-01-01 11:00:00,\n"
            "                                 2011-01-01 12:00:00, "
            "2011-01-01 13:00:00]"
        )

        assert repr(c) == exp

        idx = date_range(
            "2011-01-01 09:00", freq="h", periods=5, tz="US/Eastern", unit="ns"
        )
        c = Categorical(idx)
        exp = (
            "[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, "
            "2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, "
            "2011-01-01 13:00:00-05:00]\n"
            "Categories (5, datetime64[ns, US/Eastern]): "
            "[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00,\n"
            "                                             "
            "2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00,\n"
            "                                             "
            "2011-01-01 13:00:00-05:00]"
        )

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx)
        exp = (
            "[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, "
            "2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, "
            "2011-01-01 13:00:00-05:00, 2011-01-01 09:00:00-05:00, "
            "2011-01-01 10:00:00-05:00, 2011-01-01 11:00:00-05:00, "
            "2011-01-01 12:00:00-05:00, 2011-01-01 13:00:00-05:00]\n"
            "Categories (5, datetime64[ns, US/Eastern]): "
            "[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00,\n"
            "                                             "
            "2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00,\n"
            "                                             "
            "2011-01-01 13:00:00-05:00]"
        )

        assert repr(c) == exp

    def test_categorical_repr_datetime_ordered(self):
        # [FUNCTION DOC]
        # Ordered version of the datetime representation test, with less-than separators and line wrapping.
        idx = date_range("2011-01-01 09:00", freq="h", periods=5, unit="ns")
        c = Categorical(idx, ordered=True)
        exp = """[2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00, 2011-01-01 12:00:00, 2011-01-01 13:00:00]
Categories (5, datetime64[ns]): [2011-01-01 09:00:00 < 2011-01-01 10:00:00 < 2011-01-01 11:00:00 <
                                 2011-01-01 12:00:00 < 2011-01-01 13:00:00]"""  # noqa: E501

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx, ordered=True)
        exp = """[2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00, 2011-01-01 12:00:00, 2011-01-01 13:00:00, 2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00, 2011-01-01 12:00:00, 2011-01-01 13:00:00]
Categories (5, datetime64[ns]): [2011-01-01 09:00:00 < 2011-01-01 10:00:00 < 2011-01-01 11:00:00 <
                                 2011-01-01 12:00:00 < 2011-01-01 13:00:00]"""  # noqa: E501

        assert repr(c) == exp

        idx = date_range(
            "2011-01-01 09:00", freq="h", periods=5, tz="US/Eastern", unit="ns"
        )
        c = Categorical(idx, ordered=True)
        exp = """[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, 2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, 2011-01-01 13:00:00-05:00]
Categories (5, datetime64[ns, US/Eastern]): [2011-01-01 09:00:00-05:00 < 2011-01-01 10:00:00-05:00 <
                                             2011-01-01 11:00:00-05:00 < 2011-01-01 12:00:00-05:00 <
                                             2011-01-01 13:00:00-05:00]"""  # noqa: E501

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx, ordered=True)
        exp = """[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, 2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, 2011-01-01 13:00:00-05:00, 2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, 2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, 2011-01-01 13:00:00-05:00]
Categories (5, datetime64[ns, US/Eastern]): [2011-01-01 09:00:00-05:00 < 2011-01-01 10:00:00-05:00 <
                                             2011-01-01 11:00:00-05:00 < 2011-01-01 12:00:00-05:00 <
                                             2011-01-01 13:00:00-05:00]"""  # noqa: E501

        assert repr(c) == exp

    def test_categorical_repr_int_with_nan(self):
        # [FUNCTION DOC]
        # Integer Categorical and a categorical Series with NaN print the missing value as NaN and exclude it from the categories.
        c = Categorical([1, 2, np.nan])
        c_exp = """[1, 2, NaN]\nCategories (2, int64): [1, 2]"""
        assert repr(c) == c_exp

        s = Series([1, 2, np.nan], dtype="object").astype("category")
        s_exp = """0      1\n1      2\n2    NaN
dtype: category
Categories (2, int64): [1, 2]"""
        assert repr(s) == s_exp

    def test_categorical_repr_period(self):
        # [FUNCTION DOC]
        # Hourly and monthly period Categoricals print values and (wrapped) categories lines.
        idx = period_range("2011-01-01 09:00", freq="h", periods=5)
        c = Categorical(idx)
        exp = """[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00]
Categories (5, period[h]): [2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00,
                            2011-01-01 13:00]"""  # noqa: E501

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx)
        exp = """[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00, 2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00]
Categories (5, period[h]): [2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00,
                            2011-01-01 13:00]"""  # noqa: E501

        assert repr(c) == exp

        idx = period_range("2011-01", freq="M", periods=5)
        c = Categorical(idx)
        exp = """[2011-01, 2011-02, 2011-03, 2011-04, 2011-05]
Categories (5, period[M]): [2011-01, 2011-02, 2011-03, 2011-04, 2011-05]"""

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx)
        exp = """[2011-01, 2011-02, 2011-03, 2011-04, 2011-05, 2011-01, 2011-02, 2011-03, 2011-04, 2011-05]
Categories (5, period[M]): [2011-01, 2011-02, 2011-03, 2011-04, 2011-05]"""  # noqa: E501

        assert repr(c) == exp

    def test_categorical_repr_period_ordered(self):
        # [FUNCTION DOC]
        # Ordered version of the period representation test.
        idx = period_range("2011-01-01 09:00", freq="h", periods=5)
        c = Categorical(idx, ordered=True)
        exp = """[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00]
Categories (5, period[h]): [2011-01-01 09:00 < 2011-01-01 10:00 < 2011-01-01 11:00 < 2011-01-01 12:00 <
                            2011-01-01 13:00]"""  # noqa: E501

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx, ordered=True)
        exp = """[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00, 2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00]
Categories (5, period[h]): [2011-01-01 09:00 < 2011-01-01 10:00 < 2011-01-01 11:00 < 2011-01-01 12:00 <
                            2011-01-01 13:00]"""  # noqa: E501

        assert repr(c) == exp

        idx = period_range("2011-01", freq="M", periods=5)
        c = Categorical(idx, ordered=True)
        exp = """[2011-01, 2011-02, 2011-03, 2011-04, 2011-05]
Categories (5, period[M]): [2011-01 < 2011-02 < 2011-03 < 2011-04 < 2011-05]"""

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx, ordered=True)
        exp = """[2011-01, 2011-02, 2011-03, 2011-04, 2011-05, 2011-01, 2011-02, 2011-03, 2011-04, 2011-05]
Categories (5, period[M]): [2011-01 < 2011-02 < 2011-03 < 2011-04 < 2011-05]"""  # noqa: E501

        assert repr(c) == exp

    def test_categorical_repr_timedelta(self):
        # [FUNCTION DOC]
        # Timedelta Categoricals (day and hour steps, short and truncated) print values, length and categories with microsecond unit.
        idx = timedelta_range("1 days", periods=5)
        c = Categorical(idx)
        exp = """[1 days, 2 days, 3 days, 4 days, 5 days]
Categories (5, timedelta64[us]): [1 days, 2 days, 3 days, 4 days, 5 days]"""

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx)
        exp = """[1 days, 2 days, 3 days, 4 days, 5 days, 1 days, 2 days, 3 days, 4 days, 5 days]
Categories (5, timedelta64[us]): [1 days, 2 days, 3 days, 4 days, 5 days]"""  # noqa: E501

        assert repr(c) == exp

        idx = timedelta_range("1 hours", periods=20)
        c = Categorical(idx)
        exp = """[0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00, 3 days 01:00:00, 4 days 01:00:00, ..., 15 days 01:00:00, 16 days 01:00:00, 17 days 01:00:00, 18 days 01:00:00, 19 days 01:00:00]
Length: 20
Categories (20, timedelta64[us]): [0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00,
                                   3 days 01:00:00, ..., 16 days 01:00:00, 17 days 01:00:00,
                                   18 days 01:00:00, 19 days 01:00:00]"""  # noqa: E501

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx)
        exp = """[0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00, 3 days 01:00:00, 4 days 01:00:00, ..., 15 days 01:00:00, 16 days 01:00:00, 17 days 01:00:00, 18 days 01:00:00, 19 days 01:00:00]
Length: 40
Categories (20, timedelta64[us]): [0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00,
                                   3 days 01:00:00, ..., 16 days 01:00:00, 17 days 01:00:00,
                                   18 days 01:00:00, 19 days 01:00:00]"""  # noqa: E501

        assert repr(c) == exp

    def test_categorical_repr_timedelta_ordered(self):
        # [FUNCTION DOC]
        # Ordered version of the timedelta representation test.
        idx = timedelta_range("1 days", periods=5)
        c = Categorical(idx, ordered=True)
        exp = """[1 days, 2 days, 3 days, 4 days, 5 days]
Categories (5, timedelta64[us]): [1 days < 2 days < 3 days < 4 days < 5 days]"""

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx, ordered=True)
        exp = """[1 days, 2 days, 3 days, 4 days, 5 days, 1 days, 2 days, 3 days, 4 days, 5 days]
Categories (5, timedelta64[us]): [1 days < 2 days < 3 days < 4 days < 5 days]"""  # noqa: E501

        assert repr(c) == exp

        idx = timedelta_range("1 hours", periods=20)
        c = Categorical(idx, ordered=True)
        exp = """[0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00, 3 days 01:00:00, 4 days 01:00:00, ..., 15 days 01:00:00, 16 days 01:00:00, 17 days 01:00:00, 18 days 01:00:00, 19 days 01:00:00]
Length: 20
Categories (20, timedelta64[us]): [0 days 01:00:00 < 1 days 01:00:00 < 2 days 01:00:00 <
                                   3 days 01:00:00 ... 16 days 01:00:00 < 17 days 01:00:00 <
                                   18 days 01:00:00 < 19 days 01:00:00]"""  # noqa: E501

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx, ordered=True)
        exp = """[0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00, 3 days 01:00:00, 4 days 01:00:00, ..., 15 days 01:00:00, 16 days 01:00:00, 17 days 01:00:00, 18 days 01:00:00, 19 days 01:00:00]
Length: 40
Categories (20, timedelta64[us]): [0 days 01:00:00 < 1 days 01:00:00 < 2 days 01:00:00 <
                                   3 days 01:00:00 ... 16 days 01:00:00 < 17 days 01:00:00 <
                                   18 days 01:00:00 < 19 days 01:00:00]"""  # noqa: E501

        assert repr(c) == exp

    def test_categorical_index_repr(self):
        # [FUNCTION DOC]
        # Categorical index of integers prints with categories, ordered flag and dtype; ten categories truncated.
        idx = CategoricalIndex(Categorical([1, 2, 3]))
        exp = """CategoricalIndex([1, 2, 3], categories=[1, 2, 3], ordered=False, dtype='category')"""  # noqa: E501
        assert repr(idx) == exp

        i = CategoricalIndex(Categorical(np.arange(10, dtype=np.int64)))
        exp = """CategoricalIndex([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], categories=[0, 1, 2, 3, ..., 6, 7, 8, 9], ordered=False, dtype='category')"""  # noqa: E501
        assert repr(i) == exp

    def test_categorical_index_repr_ordered(self):
        # [FUNCTION DOC]
        # Ordered version of the integer categorical index representation test.
        i = CategoricalIndex(Categorical([1, 2, 3], ordered=True))
        exp = """CategoricalIndex([1, 2, 3], categories=[1, 2, 3], ordered=True, dtype='category')"""  # noqa: E501
        assert repr(i) == exp

        i = CategoricalIndex(Categorical(np.arange(10, dtype=np.int64), ordered=True))
        exp = """CategoricalIndex([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], categories=[0, 1, 2, 3, ..., 6, 7, 8, 9], ordered=True, dtype='category')"""  # noqa: E501
        assert repr(i) == exp

    def test_categorical_index_repr_datetime(self):
        # [FUNCTION DOC]
        # Categorical index of naive and tz-aware datetimes prints wrapped values, categories, ordered flag and dtype.
        idx = date_range("2011-01-01 09:00", freq="h", periods=5)
        i = CategoricalIndex(Categorical(idx))
        exp = """CategoricalIndex(['2011-01-01 09:00:00', '2011-01-01 10:00:00',
                  '2011-01-01 11:00:00', '2011-01-01 12:00:00',
                  '2011-01-01 13:00:00'],
                 categories=[2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00, 2011-01-01 12:00:00, 2011-01-01 13:00:00], ordered=False, dtype='category')"""  # noqa: E501

        assert repr(i) == exp

        idx = date_range("2011-01-01 09:00", freq="h", periods=5, tz="US/Eastern")
        i = CategoricalIndex(Categorical(idx))
        exp = """CategoricalIndex(['2011-01-01 09:00:00-05:00', '2011-01-01 10:00:00-05:00',
                  '2011-01-01 11:00:00-05:00', '2011-01-01 12:00:00-05:00',
                  '2011-01-01 13:00:00-05:00'],
                 categories=[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, 2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, 2011-01-01 13:00:00-05:00], ordered=False, dtype='category')"""  # noqa: E501

        assert repr(i) == exp

    def test_categorical_index_repr_datetime_ordered(self):
        # [FUNCTION DOC]
        # Ordered version of the datetime categorical index representation test, including a doubled index.
        idx = date_range("2011-01-01 09:00", freq="h", periods=5)
        i = CategoricalIndex(Categorical(idx, ordered=True))
        exp = """CategoricalIndex(['2011-01-01 09:00:00', '2011-01-01 10:00:00',
                  '2011-01-01 11:00:00', '2011-01-01 12:00:00',
                  '2011-01-01 13:00:00'],
                 categories=[2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00, 2011-01-01 12:00:00, 2011-01-01 13:00:00], ordered=True, dtype='category')"""  # noqa: E501

        assert repr(i) == exp

        idx = date_range("2011-01-01 09:00", freq="h", periods=5, tz="US/Eastern")
        i = CategoricalIndex(Categorical(idx, ordered=True))
        exp = """CategoricalIndex(['2011-01-01 09:00:00-05:00', '2011-01-01 10:00:00-05:00',
                  '2011-01-01 11:00:00-05:00', '2011-01-01 12:00:00-05:00',
                  '2011-01-01 13:00:00-05:00'],
                 categories=[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, 2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, 2011-01-01 13:00:00-05:00], ordered=True, dtype='category')"""  # noqa: E501

        assert repr(i) == exp

        i = CategoricalIndex(Categorical(idx.append(idx), ordered=True))
        exp = """CategoricalIndex(['2011-01-01 09:00:00-05:00', '2011-01-01 10:00:00-05:00',
                  '2011-01-01 11:00:00-05:00', '2011-01-01 12:00:00-05:00',
                  '2011-01-01 13:00:00-05:00', '2011-01-01 09:00:00-05:00',
                  '2011-01-01 10:00:00-05:00', '2011-01-01 11:00:00-05:00',
                  '2011-01-01 12:00:00-05:00', '2011-01-01 13:00:00-05:00'],
                 categories=[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, 2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, 2011-01-01 13:00:00-05:00], ordered=True, dtype='category')"""  # noqa: E501

        assert repr(i) == exp

    def test_categorical_index_repr_period(self):
        # [FUNCTION DOC]
        # Categorical index of periods for lengths one, two, three, five, doubled and monthly frequency.
        # test all length
        idx = period_range("2011-01-01 09:00", freq="h", periods=1)
        i = CategoricalIndex(Categorical(idx))
        exp = """CategoricalIndex(['2011-01-01 09:00'], categories=[2011-01-01 09:00], ordered=False, dtype='category')"""  # noqa: E501
        assert repr(i) == exp

        idx = period_range("2011-01-01 09:00", freq="h", periods=2)
        i = CategoricalIndex(Categorical(idx))
        exp = """CategoricalIndex(['2011-01-01 09:00', '2011-01-01 10:00'], categories=[2011-01-01 09:00, 2011-01-01 10:00], ordered=False, dtype='category')"""  # noqa: E501
        assert repr(i) == exp

        idx = period_range("2011-01-01 09:00", freq="h", periods=3)
        i = CategoricalIndex(Categorical(idx))
        exp = """CategoricalIndex(['2011-01-01 09:00', '2011-01-01 10:00', '2011-01-01 11:00'], categories=[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00], ordered=False, dtype='category')"""  # noqa: E501
        assert repr(i) == exp

        idx = period_range("2011-01-01 09:00", freq="h", periods=5)
        i = CategoricalIndex(Categorical(idx))
        exp = """CategoricalIndex(['2011-01-01 09:00', '2011-01-01 10:00', '2011-01-01 11:00',
                  '2011-01-01 12:00', '2011-01-01 13:00'],
                 categories=[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00], ordered=False, dtype='category')"""  # noqa: E501

        assert repr(i) == exp

        i = CategoricalIndex(Categorical(idx.append(idx)))
        exp = """CategoricalIndex(['2011-01-01 09:00', '2011-01-01 10:00', '2011-01-01 11:00',
                  '2011-01-01 12:00', '2011-01-01 13:00', '2011-01-01 09:00',
                  '2011-01-01 10:00', '2011-01-01 11:00', '2011-01-01 12:00',
                  '2011-01-01 13:00'],
                 categories=[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00], ordered=False, dtype='category')"""  # noqa: E501

        assert repr(i) == exp

        idx = period_range("2011-01", freq="M", periods=5)
        i = CategoricalIndex(Categorical(idx))
        exp = """CategoricalIndex(['2011-01', '2011-02', '2011-03', '2011-04', '2011-05'], categories=[2011-01, 2011-02, 2011-03, 2011-04, 2011-05], ordered=False, dtype='category')"""  # noqa: E501
        assert repr(i) == exp

    def test_categorical_index_repr_period_ordered(self):
        # [FUNCTION DOC]
        # Ordered version of the period categorical index representation test.
        idx = period_range("2011-01-01 09:00", freq="h", periods=5)
        i = CategoricalIndex(Categorical(idx, ordered=True))
        exp = """CategoricalIndex(['2011-01-01 09:00', '2011-01-01 10:00', '2011-01-01 11:00',
                  '2011-01-01 12:00', '2011-01-01 13:00'],
                 categories=[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00], ordered=True, dtype='category')"""  # noqa: E501

        assert repr(i) == exp

        idx = period_range("2011-01", freq="M", periods=5)
        i = CategoricalIndex(Categorical(idx, ordered=True))
        exp = """CategoricalIndex(['2011-01', '2011-02', '2011-03', '2011-04', '2011-05'], categories=[2011-01, 2011-02, 2011-03, 2011-04, 2011-05], ordered=True, dtype='category')"""  # noqa: E501
        assert repr(i) == exp

    def test_categorical_index_repr_timedelta(self):
        # [FUNCTION DOC]
        # Categorical index of timedeltas, five days and ten hours with truncated categories.
        idx = timedelta_range("1 days", periods=5)
        i = CategoricalIndex(Categorical(idx))
        exp = """CategoricalIndex(['1 days', '2 days', '3 days', '4 days', '5 days'], categories=[1 days, 2 days, 3 days, 4 days, 5 days], ordered=False, dtype='category')"""  # noqa: E501
        assert repr(i) == exp

        idx = timedelta_range("1 hours", periods=10)
        i = CategoricalIndex(Categorical(idx))
        exp = """CategoricalIndex(['0 days 01:00:00', '1 days 01:00:00', '2 days 01:00:00',
                  '3 days 01:00:00', '4 days 01:00:00', '5 days 01:00:00',
                  '6 days 01:00:00', '7 days 01:00:00', '8 days 01:00:00',
                  '9 days 01:00:00'],
                 categories=[0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00, 3 days 01:00:00, ..., 6 days 01:00:00, 7 days 01:00:00, 8 days 01:00:00, 9 days 01:00:00], ordered=False, dtype='category')"""  # noqa: E501

        assert repr(i) == exp

    def test_categorical_index_repr_timedelta_ordered(self):
        # [FUNCTION DOC]
        # Ordered version of the timedelta categorical index representation test.
        idx = timedelta_range("1 days", periods=5)
        i = CategoricalIndex(Categorical(idx, ordered=True))
        exp = """CategoricalIndex(['1 days', '2 days', '3 days', '4 days', '5 days'], categories=[1 days, 2 days, 3 days, 4 days, 5 days], ordered=True, dtype='category')"""  # noqa: E501
        assert repr(i) == exp

        idx = timedelta_range("1 hours", periods=10)
        i = CategoricalIndex(Categorical(idx, ordered=True))
        exp = """CategoricalIndex(['0 days 01:00:00', '1 days 01:00:00', '2 days 01:00:00',
                  '3 days 01:00:00', '4 days 01:00:00', '5 days 01:00:00',
                  '6 days 01:00:00', '7 days 01:00:00', '8 days 01:00:00',
                  '9 days 01:00:00'],
                 categories=[0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00, 3 days 01:00:00, ..., 6 days 01:00:00, 7 days 01:00:00, 8 days 01:00:00, 9 days 01:00:00], ordered=True, dtype='category')"""  # noqa: E501

        assert repr(i) == exp

    def test_categorical_str_repr(self):
        # [FUNCTION DOC]
        # Mixed integer and string values print the integer categories first, then the quoted string, with object dtype.
        # GH 33676
        result = repr(Categorical([1, "2", 3, 4]))
        expected = "[1, '2', 3, 4]\nCategories (4, object): [1, 3, 4, '2']"
        assert result == expected

    def test_categorical_with_string_dtype(self, string_dtype_no_object):
        # [FUNCTION DOC]
        # Uses the string_dtype_no_object fixture. Categories of string extension dtypes are printed with quotes and show
        # the dtype name in the categories line.
        # GH 63045 - ensure categories are quoted for string dtypes
        s = Series(
            ["apple", "banana", "cherry", "cherry"], dtype=string_dtype_no_object
        )
        result = repr(Categorical(s))
        expected = f"['apple', 'banana', 'cherry', 'cherry']\nCategories (3, {string_dtype_no_object!s}): ['apple', 'banana', 'cherry']"  # noqa: E501

        assert result == expected
