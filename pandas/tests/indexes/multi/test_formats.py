# [MODULE DOC]
# Tests for text rendering of MultiIndex: repr and str with unicode, display option
# limits on number of items and tuple width, right-justification of mixed-width
# elements, and very long elements.
# Uses the shared idx fixture. Compares against literal expected output text.

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Index,
    MultiIndex,
)


def test_repr_with_unicode_data():
    # [FUNCTION DOC]
    # Verifies that the repr of an index built from data with a Hebrew character,
    # under UTF-8 encoding, does not contain escaped backslash sequences.
    with pd.option_context("display.encoding", "UTF-8"):
        d = {"a": ["\u05d0", 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
        index = pd.DataFrame(d).set_index(["a", "b"]).index
        assert "\\" not in repr(index)  # we don't want unicode-escaped


def test_repr_roundtrip_raises():
    # [FUNCTION DOC]
    # Verifies the repr text is not evaluable: evaluating it raises a type error saying
    # both levels and codes must be passed.
    mi = MultiIndex.from_product([list("ab"), range(3)], names=["first", "second"])
    msg = "Must pass both levels and codes"
    with pytest.raises(TypeError, match=msg):
        eval(repr(mi))


def test_unicode_string_with_unicode():
    # [FUNCTION DOC]
    # Smoke test: the string form of an index from data with non-ASCII text must not raise.
    d = {"a": ["\u05d0", 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
    idx = pd.DataFrame(d).set_index(["a", "b"]).index
    str(idx)


def test_repr_max_seq_item_setting(idx):
    # [FUNCTION DOC]
    # Verifies (GH10182) with no limit on displayed sequence items, a repeated
    # index prints in full (no ellipsis) in repr and str.
    # GH10182
    idx = idx.repeat(50)
    with pd.option_context("display.max_seq_items", None):
        repr(idx)
        assert "..." not in str(idx)


class TestRepr:
    # [CLASS DOC]
    # Groups repr tests on MultiIndex.
    def test_unicode_repr_issues(self):
        # [FUNCTION DOC]
        # Smoke test: repr of the levels and of a level's values for non-ASCII labels must not raise.
        levels = [Index(["a/\u03c3", "b/\u03c3", "c/\u03c3"]), Index([0, 1])]
        codes = [np.arange(3).repeat(2), np.tile(np.arange(2), 3)]
        index = MultiIndex(levels=levels, codes=codes)

        repr(index.levels)
        repr(index.get_level_values(1))

    def test_repr_max_seq_items_equal_to_n(self, idx):
        # [FUNCTION DOC]
        # Verifies that when the max item option equals the length, the repr shows all six
        # tuples with names.
        # display.max_seq_items == n
        with pd.option_context("display.max_seq_items", 6):
            result = idx.__repr__()
            expected = """\
MultiIndex([('foo', 'one'),
            ('foo', 'two'),
            ('bar', 'one'),
            ('baz', 'two'),
            ('qux', 'one'),
            ('qux', 'two')],
           names=['first', 'second'])"""
            assert result == expected

    def test_repr(self, idx):
        # [FUNCTION DOC]
        # Verifies repr for one element, all elements, a limit of five (shows the first two,
        # ellipsis, last two with the length) and a limit of one (shows ellipsis and the last
        # tuple with truncated names and the length).
        result = idx[:1].__repr__()
        expected = """\
MultiIndex([('foo', 'one')],
           names=['first', 'second'])"""
        assert result == expected

        result = idx.__repr__()
        expected = """\
MultiIndex([('foo', 'one'),
            ('foo', 'two'),
            ('bar', 'one'),
            ('baz', 'two'),
            ('qux', 'one'),
            ('qux', 'two')],
           names=['first', 'second'])"""
        assert result == expected

        with pd.option_context("display.max_seq_items", 5):
            result = idx.__repr__()
            expected = """\
MultiIndex([('foo', 'one'),
            ('foo', 'two'),
            ...
            ('qux', 'one'),
            ('qux', 'two')],
           names=['first', 'second'], length=6)"""
            assert result == expected

        # display.max_seq_items == 1
        with pd.option_context("display.max_seq_items", 1):
            result = idx.__repr__()
            expected = """\
MultiIndex([...
            ('qux', 'two')],
           names=['first', ...], length=6)"""
            assert result == expected

    def test_rjust(self):
        # [FUNCTION DOC]
        # Verifies tuples are right-justified so that mixed-width string and integer elements
        # align, for one element, a sampling, and the full 2000 element index with elision.
        n = 1000
        ci = pd.CategoricalIndex(list("a" * n) + (["abc"] * n))
        dti = pd.date_range("2000-01-01", freq="s", periods=n * 2)
        mi = MultiIndex.from_arrays([ci, ci.codes + 9, dti], names=["a", "b", "dti"])
        result = mi[:1].__repr__()
        expected = """\
MultiIndex([('a', 9, '2000-01-01 00:00:00')],
           names=['a', 'b', 'dti'])"""
        assert result == expected

        result = mi[::500].__repr__()
        expected = """\
MultiIndex([(  'a',  9, '2000-01-01 00:00:00'),
            (  'a',  9, '2000-01-01 00:08:20'),
            ('abc', 10, '2000-01-01 00:16:40'),
            ('abc', 10, '2000-01-01 00:25:00')],
           names=['a', 'b', 'dti'])"""
        assert result == expected

        result = mi.__repr__()
        expected = """\
MultiIndex([(  'a',  9, '2000-01-01 00:00:00'),
            (  'a',  9, '2000-01-01 00:00:01'),
            (  'a',  9, '2000-01-01 00:00:02'),
            (  'a',  9, '2000-01-01 00:00:03'),
            (  'a',  9, '2000-01-01 00:00:04'),
            (  'a',  9, '2000-01-01 00:00:05'),
            (  'a',  9, '2000-01-01 00:00:06'),
            (  'a',  9, '2000-01-01 00:00:07'),
            (  'a',  9, '2000-01-01 00:00:08'),
            (  'a',  9, '2000-01-01 00:00:09'),
            ...
            ('abc', 10, '2000-01-01 00:33:10'),
            ('abc', 10, '2000-01-01 00:33:11'),
            ('abc', 10, '2000-01-01 00:33:12'),
            ('abc', 10, '2000-01-01 00:33:13'),
            ('abc', 10, '2000-01-01 00:33:14'),
            ('abc', 10, '2000-01-01 00:33:15'),
            ('abc', 10, '2000-01-01 00:33:16'),
            ('abc', 10, '2000-01-01 00:33:17'),
            ('abc', 10, '2000-01-01 00:33:18'),
            ('abc', 10, '2000-01-01 00:33:19')],
           names=['a', 'b', 'dti'], length=2000)"""
        assert result == expected

    def test_tuple_width(self):
        # [FUNCTION DOC]
        # Verifies that an index with five levels truncates each tuple to the first four
        # elements followed by an ellipsis, for one element, ten elements and the full index.
        n = 1000
        ci = pd.CategoricalIndex(list("a" * n) + (["abc"] * n))
        dti = pd.date_range("2000-01-01", freq="s", periods=n * 2)
        levels = [ci, ci.codes + 9, dti, dti, dti]
        names = ["a", "b", "dti_1", "dti_2", "dti_3"]
        mi = MultiIndex.from_arrays(levels, names=names)
        result = mi[:1].__repr__()
        expected = """MultiIndex([('a', 9, '2000-01-01 00:00:00', '2000-01-01 00:00:00', ...)],
           names=['a', 'b', 'dti_1', 'dti_2', 'dti_3'])"""  # noqa: E501
        assert result == expected

        result = mi[:10].__repr__()
        expected = """\
MultiIndex([('a', 9, '2000-01-01 00:00:00', '2000-01-01 00:00:00', ...),
            ('a', 9, '2000-01-01 00:00:01', '2000-01-01 00:00:01', ...),
            ('a', 9, '2000-01-01 00:00:02', '2000-01-01 00:00:02', ...),
            ('a', 9, '2000-01-01 00:00:03', '2000-01-01 00:00:03', ...),
            ('a', 9, '2000-01-01 00:00:04', '2000-01-01 00:00:04', ...),
            ('a', 9, '2000-01-01 00:00:05', '2000-01-01 00:00:05', ...),
            ('a', 9, '2000-01-01 00:00:06', '2000-01-01 00:00:06', ...),
            ('a', 9, '2000-01-01 00:00:07', '2000-01-01 00:00:07', ...),
            ('a', 9, '2000-01-01 00:00:08', '2000-01-01 00:00:08', ...),
            ('a', 9, '2000-01-01 00:00:09', '2000-01-01 00:00:09', ...)],
           names=['a', 'b', 'dti_1', 'dti_2', 'dti_3'])"""
        assert result == expected

        result = mi.__repr__()
        expected = """\
MultiIndex([(  'a',  9, '2000-01-01 00:00:00', '2000-01-01 00:00:00', ...),
            (  'a',  9, '2000-01-01 00:00:01', '2000-01-01 00:00:01', ...),
            (  'a',  9, '2000-01-01 00:00:02', '2000-01-01 00:00:02', ...),
            (  'a',  9, '2000-01-01 00:00:03', '2000-01-01 00:00:03', ...),
            (  'a',  9, '2000-01-01 00:00:04', '2000-01-01 00:00:04', ...),
            (  'a',  9, '2000-01-01 00:00:05', '2000-01-01 00:00:05', ...),
            (  'a',  9, '2000-01-01 00:00:06', '2000-01-01 00:00:06', ...),
            (  'a',  9, '2000-01-01 00:00:07', '2000-01-01 00:00:07', ...),
            (  'a',  9, '2000-01-01 00:00:08', '2000-01-01 00:00:08', ...),
            (  'a',  9, '2000-01-01 00:00:09', '2000-01-01 00:00:09', ...),
            ...
            ('abc', 10, '2000-01-01 00:33:10', '2000-01-01 00:33:10', ...),
            ('abc', 10, '2000-01-01 00:33:11', '2000-01-01 00:33:11', ...),
            ('abc', 10, '2000-01-01 00:33:12', '2000-01-01 00:33:12', ...),
            ('abc', 10, '2000-01-01 00:33:13', '2000-01-01 00:33:13', ...),
            ('abc', 10, '2000-01-01 00:33:14', '2000-01-01 00:33:14', ...),
            ('abc', 10, '2000-01-01 00:33:15', '2000-01-01 00:33:15', ...),
            ('abc', 10, '2000-01-01 00:33:16', '2000-01-01 00:33:16', ...),
            ('abc', 10, '2000-01-01 00:33:17', '2000-01-01 00:33:17', ...),
            ('abc', 10, '2000-01-01 00:33:18', '2000-01-01 00:33:18', ...),
            ('abc', 10, '2000-01-01 00:33:19', '2000-01-01 00:33:19', ...)],
           names=['a', 'b', 'dti_1', 'dti_2', 'dti_3'], length=2000)"""
        assert result == expected

    def test_multiindex_long_element(self):
        # [FUNCTION DOC]
        # Verifies (GH#52960) a one-level multi-index with a very long element prints
        # the element in full with a trailing empty names line.
        # Non-regression test towards GH#52960
        data = MultiIndex.from_tuples([("c" * 62,)])

        expected = (
            "MultiIndex([('cccccccccccccccccccccccccccccccccccccccc"
            "cccccccccccccccccccccc',)],\n           )"
        )
        assert str(data) == expected
