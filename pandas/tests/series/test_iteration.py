# [MODULE DOC]
# Tests for iterating over a Series: keys, value iteration, and item iteration (label and value pairs).
# Relies on datetime and string series fixtures. No imports; no import-time side effects.

class TestIteration:
    # [CLASS DOC]
    # Groups tests of Series iteration. Stateless.
    def test_keys(self, datetime_series):
        # [FUNCTION DOC]
        # Verifies the keys method returns the very same index object (identity, not just equality) of the Series.
        assert datetime_series.keys() is datetime_series.index

    def test_iter_datetimes(self, datetime_series):
        # [FUNCTION DOC]
        # Verifies that iterating over the datetime-indexed fixture yields the same values as positional access.
        for i, val in enumerate(datetime_series):
            assert val == datetime_series.iloc[i]

    def test_iter_strings(self, string_series):
        # [FUNCTION DOC]
        # Verifies that iterating over the string-indexed fixture yields the same values as positional access.
        for i, val in enumerate(string_series):
            assert val == string_series.iloc[i]

    def test_iteritems_datetimes(self, datetime_series):
        # [FUNCTION DOC]
        # Verifies that item iteration over the datetime fixture yields label and value pairs where the value equals label-based access.
        # KNOWN ISSUE: the name refers to the removed iteritems method but the body calls the items method, making this identical to the items test of the same data.
        for idx, val in datetime_series.items():
            assert val == datetime_series[idx]  # noqa: PLR1733

    def test_iteritems_strings(self, string_series):
        # [FUNCTION DOC]
        # Verifies that item iteration over the string fixture yields pairs consistent with label-based access, and that the returned iterator is lazy (it has no list-style reverse method).
        # KNOWN ISSUE: the name refers to the removed iteritems method but the body calls the items method, making this identical to the items test of the same data.
        for idx, val in string_series.items():
            assert val == string_series[idx]  # noqa: PLR1733

        # assert is lazy (generators don't define reverse, lists do)
        assert not hasattr(string_series.items(), "reverse")

    def test_items_datetimes(self, datetime_series):
        # [FUNCTION DOC]
        # Verifies that item iteration over the datetime fixture yields label and value pairs where the value equals label-based access.
        for idx, val in datetime_series.items():
            assert val == datetime_series[idx]  # noqa: PLR1733

    def test_items_strings(self, string_series):
        # [FUNCTION DOC]
        # Verifies that item iteration over the string fixture yields consistent pairs and returns a lazy iterator (no list-style reverse method).
        for idx, val in string_series.items():
            assert val == string_series[idx]  # noqa: PLR1733

        # assert is lazy (generators don't define reverse, lists do)
        assert not hasattr(string_series.items(), "reverse")
