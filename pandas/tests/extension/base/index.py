"""
Tests for Indexes backed by arbitrary ExtensionArrays.
"""
# [MODULE DOC]
# Base tests for pandas Index objects backed by an arbitrary extension array.
# Public contents: BaseIndexTests mixin. Requires the data fixture from the conftest.

import pandas as pd


class BaseIndexTests:
    # [CLASS DOC]
    # Mixin with tests that an Index can be constructed from the extension array fixture.
    # Each test uses the data fixture (an extension array) and checks dtype preservation.
    """Tests for Index object backed by an ExtensionArray"""

    def test_index_from_array(self, data):
        # [FUNCTION DOC]
        # Builds an index directly from the data fixture and asserts the index dtype equals
        # the array's dtype.
        idx = pd.Index(data)
        assert data.dtype == idx.dtype

    def test_index_from_listlike_with_dtype(self, data):
        # [FUNCTION DOC]
        # Builds an index from the array with an explicit dtype and again from a plain list
        # made of the array's elements with the same explicit dtype. Asserts the resulting index dtype
        # equals the array's dtype in both cases.
        idx = pd.Index(data, dtype=data.dtype)
        assert idx.dtype == data.dtype

        idx = pd.Index(list(data), dtype=data.dtype)
        assert idx.dtype == data.dtype
