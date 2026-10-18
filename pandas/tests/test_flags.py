# [MODULE DOC]
# Tests for the flags object attached to pandas objects (currently only the
# setting that controls whether duplicate labels are allowed): equality,
# mutation, text representation, weak reference behavior, and item access.
import pytest

import pandas as pd


class TestFlags:
    # [CLASS DOC]
    # Test container for the flags object obtained from a DataFrame; no state.
    def test_equality(self):
        # [FUNCTION DOC]
        # Verifies two flags objects compare equal to themselves, unequal when their
        # duplicate-label setting differs, and unequal to a non-flags value.
        a = pd.DataFrame().set_flags(allows_duplicate_labels=True).flags
        b = pd.DataFrame().set_flags(allows_duplicate_labels=False).flags

        assert a == a
        assert b == b
        assert a != b
        assert a != 2

    def test_set(self):
        # [FUNCTION DOC]
        # Verifies the duplicate-label setting can be changed both through attribute
        # assignment and through item assignment, and the new value is readable.
        df = pd.DataFrame().set_flags(allows_duplicate_labels=True)
        a = df.flags
        a.allows_duplicate_labels = False
        assert a.allows_duplicate_labels is False
        a["allows_duplicate_labels"] = True
        assert a.allows_duplicate_labels is True

    def test_repr(self):
        # [FUNCTION DOC]
        # Verifies the text representation has the form of the class name wrapping the
        # duplicate-label setting, for both true and false.
        a = repr(pd.DataFrame({"A"}).set_flags(allows_duplicate_labels=True).flags)
        assert a == "<Flags(allows_duplicate_labels=True)>"
        a = repr(pd.DataFrame({"A"}).set_flags(allows_duplicate_labels=False).flags)
        assert a == "<Flags(allows_duplicate_labels=False)>"

    def test_obj_ref(self):
        # [FUNCTION DOC]
        # Verifies that after the owning DataFrame is deleted, setting the duplicate-label
        # attribute on its flags raises a value error saying the object has been deleted
        # (flags hold only a weak reference to the owner).
        df = pd.DataFrame()
        flags = df.flags
        del df
        with pytest.raises(ValueError, match="object has been deleted"):
            flags.allows_duplicate_labels = True

    def test_getitem(self):
        # [FUNCTION DOC]
        # Verifies item access reads and writes the duplicate-label setting (default true),
        # that an unknown key raises a key error on read, and that setting an unknown key
        # raises a value error.
        df = pd.DataFrame()
        flags = df.flags
        assert flags["allows_duplicate_labels"] is True
        flags["allows_duplicate_labels"] = False
        assert flags["allows_duplicate_labels"] is False

        with pytest.raises(KeyError, match="a"):
            flags["a"]

        with pytest.raises(ValueError, match="a"):
            flags["a"] = 10
