# [MODULE DOC]
# Tests for the sparse accessor of Series.
from pandas import Series


class TestSparseAccessor:
    # [CLASS DOC]
    # Groups the sparse accessor tests. Stateless.
    def test_sparse_accessor_updates_on_inplace(self):
        # [FUNCTION DOC]
        # Verifies the sparse accessor stays valid after an in-place drop: dropping two
        # of four sparse elements in place returns nothing and leaves a series with
        # density one.
        ser = Series([1, 1, 2, 3], dtype="Sparse[int]")
        return_value = ser.drop([0, 1], inplace=True)
        assert return_value is None
        assert ser.sparse.density == 1.0
