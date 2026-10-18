# [MODULE DOC]
# Tests for the private lexsort helpers of MultiIndex: the lexsorted check and
# the lexsort depth property.
from pandas import MultiIndex


class TestIsLexsorted:
    # [CLASS DOC]
    # Groups tests of the private lexsorted check.
    def test_is_lexsorted(self):
        # [FUNCTION DOC]
        # Checks the private lexsorted predicate: true for codes sorted
        # lexicographically,
        # false for out-of-order second-level codes and for unsorted first-level codes,
        # with depth zero for the last.
        levels = [[0, 1], [0, 1, 2]]

        index = MultiIndex(
            levels=levels, codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]]
        )
        assert index._is_lexsorted()

        index = MultiIndex(
            levels=levels, codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 2, 1]]
        )
        assert not index._is_lexsorted()

        index = MultiIndex(
            levels=levels, codes=[[0, 0, 1, 0, 1, 1], [0, 1, 0, 2, 2, 1]]
        )
        assert not index._is_lexsorted()
        assert index._lexsort_depth == 0


class TestLexsortDepth:
    # [CLASS DOC]
    # Groups tests of the lexsort depth property.
    def test_lexsort_depth(self):
        # [FUNCTION DOC]
        # When a sortorder is passed at construction, the lexsort depth property reports
        # that value (2, 1, 0 for three different code layouts) (GH#28518).
        # Test that lexsort_depth return the correct sortorder
        # when it was given to the MultiIndex const.
        # GH#28518

        levels = [[0, 1], [0, 1, 2]]

        index = MultiIndex(
            levels=levels, codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]], sortorder=2
        )
        assert index._lexsort_depth == 2

        index = MultiIndex(
            levels=levels, codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 2, 1]], sortorder=1
        )
        assert index._lexsort_depth == 1

        index = MultiIndex(
            levels=levels, codes=[[0, 0, 1, 0, 1, 1], [0, 1, 0, 2, 2, 1]], sortorder=0
        )
        assert index._lexsort_depth == 0
