# [MODULE DOC]
# Tests for Index.where on the base Index.
import numpy as np

from pandas import Index
import pandas._testing as tm


class TestWhere:
    # [CLASS DOC]
    # Groups where tests for the base Index; no state.
    def test_where_intlike_str_doesnt_cast_ints(self):
        # [FUNCTION DOC]
        # Verifies that replacing masked-out positions of an integer Index with an integer-like string
        # produces an object Index with the string while the other integers remain integers (no casting
        # of the integers).
        idx = Index(range(3))
        mask = np.array([True, False, True])
        res = idx.where(mask, "2")
        expected = Index([0, "2", 2])
        tm.assert_index_equal(res, expected)
