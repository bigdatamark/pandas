"""
Tests for CategoricalIndex.__repr__ and related methods.
"""
# [MODULE DOC]
# Tests for CategoricalIndex repr: exact repr text for string and Japanese categories (short,
# multi-line, truncated, many categories), with and without the East Asian width option.

import pandas._config.config as cf

from pandas import CategoricalIndex


class TestCategoricalIndexReprStringCategories:
    # [CLASS DOC]
    # Groups exact-repr tests for CategoricalIndex; no state.
    def test_string_categorical_index_repr(self):
        # [FUNCTION DOC]
        # Verifies the one-line repr of a short string CategoricalIndex including categories, ordered
        # flag and dtype.
        # short
        idx = CategoricalIndex(["a", "bb", "ccc"])
        expected = """CategoricalIndex(['a', 'bb', 'ccc'], categories=['a', 'bb', 'ccc'], ordered=False, dtype='category')"""  # noqa: E501
        assert repr(idx) == expected

    def test_categorical_index_repr_multiline(self):
        # [FUNCTION DOC]
        # Verifies the repr wraps over several lines for thirty string entries, with categories on a
        # final line.
        # multiple lines
        idx = CategoricalIndex(["a", "bb", "ccc"] * 10)
        expected = """CategoricalIndex(['a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a',
                  'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb',
                  'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc'],
                 categories=['a', 'bb', 'ccc'], ordered=False, dtype='category')"""  # noqa: E501
        assert repr(idx) == expected

    def test_categorical_index_repr_truncated(self):
        # [FUNCTION DOC]
        # Verifies the repr of 300 entries is truncated with an ellipsis line and shows the length.
        # truncated
        idx = CategoricalIndex(["a", "bb", "ccc"] * 100)
        expected = """CategoricalIndex(['a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a',
                  ...
                  'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc'],
                 categories=['a', 'bb', 'ccc'], ordered=False, dtype='category', length=300)"""  # noqa: E501
        assert repr(idx) == expected

    def test_categorical_index_repr_many_categories(self):
        # [FUNCTION DOC]
        # Verifies that with many categories the categories list is abbreviated with an ellipsis.
        # larger categories
        idx = CategoricalIndex(list("abcdefghijklmmo"))
        expected = """CategoricalIndex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                  'm', 'm', 'o'],
                 categories=['a', 'b', 'c', 'd', ..., 'k', 'l', 'm', 'o'], ordered=False, dtype='category')"""  # noqa: E501
        assert repr(idx) == expected

    def test_categorical_index_repr_unicode(self):
        # [FUNCTION DOC]
        # Verifies the short repr for Japanese string categories.
        # short
        idx = CategoricalIndex(["あ", "いい", "ううう"])
        expected = """CategoricalIndex(['あ', 'いい', 'ううう'], categories=['あ', 'いい', 'ううう'], ordered=False, dtype='category')"""  # noqa: E501
        assert repr(idx) == expected

    def test_categorical_index_repr_unicode_multiline(self):
        # [FUNCTION DOC]
        # Verifies the multi-line repr for Japanese string entries.
        # multiple lines
        idx = CategoricalIndex(["あ", "いい", "ううう"] * 10)
        expected = """CategoricalIndex(['あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ',
                  'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい',
                  'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう'],
                 categories=['あ', 'いい', 'ううう'], ordered=False, dtype='category')"""  # noqa: E501
        assert repr(idx) == expected

    def test_categorical_index_repr_unicode_truncated(self):
        # [FUNCTION DOC]
        # Verifies the truncated repr (with length) for Japanese string entries.
        # truncated
        idx = CategoricalIndex(["あ", "いい", "ううう"] * 100)
        expected = """CategoricalIndex(['あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ',
                  ...
                  'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう'],
                 categories=['あ', 'いい', 'ううう'], ordered=False, dtype='category', length=300)"""  # noqa: E501
        assert repr(idx) == expected

    def test_categorical_index_repr_unicode_many_categories(self):
        # [FUNCTION DOC]
        # Verifies the abbreviated categories repr for many Japanese categories.
        # larger categories
        idx = CategoricalIndex(list("あいうえおかきくけこさしすせそ"))
        expected = """CategoricalIndex(['あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く', 'け', 'こ', 'さ', 'し',
                  'す', 'せ', 'そ'],
                 categories=['あ', 'い', 'う', 'え', ..., 'し', 'す', 'せ', 'そ'], ordered=False, dtype='category')"""  # noqa: E501
        assert repr(idx) == expected

    def test_categorical_index_repr_east_asian_width(self):
        # [FUNCTION DOC]
        # With the East Asian width display option on, verifies the short repr for Japanese categories.
        with cf.option_context("display.unicode.east_asian_width", True):
            # short
            idx = CategoricalIndex(["あ", "いい", "ううう"])
            expected = """CategoricalIndex(['あ', 'いい', 'ううう'], categories=['あ', 'いい', 'ううう'], ordered=False, dtype='category')"""  # noqa: E501
            assert repr(idx) == expected

    def test_categorical_index_repr_east_asian_width_multiline(self):
        # [FUNCTION DOC]
        # With the East Asian width option on, verifies the multi-line repr wraps earlier because wide
        # characters count double.
        with cf.option_context("display.unicode.east_asian_width", True):
            # multiple lines
            idx = CategoricalIndex(["あ", "いい", "ううう"] * 10)
            expected = """CategoricalIndex(['あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい',
                  'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう',
                  'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい',
                  'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう'],
                 categories=['あ', 'いい', 'ううう'], ordered=False, dtype='category')"""  # noqa: E501

            assert repr(idx) == expected

    def test_categorical_index_repr_east_asian_width_truncated(self):
        # [FUNCTION DOC]
        # With the East Asian width option on, verifies the truncated repr for 300 entries.
        with cf.option_context("display.unicode.east_asian_width", True):
            # truncated
            idx = CategoricalIndex(["あ", "いい", "ううう"] * 100)
            expected = """CategoricalIndex(['あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい',
                  'ううう', 'あ',
                  ...
                  'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう',
                  'あ', 'いい', 'ううう'],
                 categories=['あ', 'いい', 'ううう'], ordered=False, dtype='category', length=300)"""  # noqa: E501

            assert repr(idx) == expected

    def test_categorical_index_repr_east_asian_width_many_categories(self):
        # [FUNCTION DOC]
        # With the East Asian width option on, verifies the abbreviated categories repr for many
        # Japanese categories.
        with cf.option_context("display.unicode.east_asian_width", True):
            idx = CategoricalIndex(list("あいうえおかきくけこさしすせそ"))
            expected = """CategoricalIndex(['あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く', 'け', 'こ',
                  'さ', 'し', 'す', 'せ', 'そ'],
                 categories=['あ', 'い', 'う', 'え', ..., 'し', 'す', 'せ', 'そ'], ordered=False, dtype='category')"""  # noqa: E501

            assert repr(idx) == expected
