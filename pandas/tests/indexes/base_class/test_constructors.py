# [MODULE DOC]
# Tests for the base Index constructor, limited to cases that do not return a specialized
# subclass (error handling, mixed tuples, empty tuples, string inference, read-only behavior).
import numpy as np
import pytest

import pandas as pd
from pandas import (
    Index,
    MultiIndex,
    Series,
)
import pandas._testing as tm


class TestIndexConstructor:
    # [CLASS DOC]
    # Groups tests of the generic Index constructor; no state or fixtures of its own.
    # Tests for the Index constructor, specifically for cases that do
    #  not return a subclass

    @pytest.mark.parametrize("value", [1, np.int64(1)])
    def test_constructor_corner(self, value):
        # [FUNCTION DOC]
        # Parametrized over a Python int and a numpy int64. Verifies that constructing an Index from a
        # scalar raises a type error whose message says a collection is required and names the value.
        # corner case
        msg = (
            r"Index\(\.\.\.\) must be called with a collection of some "
            f"kind, {value} was passed"
        )
        with pytest.raises(TypeError, match=msg):
            Index(value)

    @pytest.mark.parametrize("index_vals", [[("A", 1), "B"], ["B", ("A", 1)]])
    def test_construction_list_mixed_tuples(self, index_vals):
        # [FUNCTION DOC]
        # Parametrized over two orderings of a list mixing a tuple and a string. Verifies the result is
        # a plain Index and not a MultiIndex, independent of element order.
        # see gh-10697: if we are constructing from a mixed list of tuples,
        # make sure that we are independent of the sorting order.
        index = Index(index_vals)
        assert isinstance(index, Index)
        assert not isinstance(index, MultiIndex)

    def test_constructor_cast(self):
        # [FUNCTION DOC]
        # Verifies that requesting float dtype for non-numeric strings raises a value error saying a
        # string could not be converted to float.
        msg = "could not convert string to float"
        with pytest.raises(ValueError, match=msg):
            Index(["a", "b", "c"], dtype=float)

    @pytest.mark.parametrize("tuple_list", [[()], [(), ()]])
    def test_construct_empty_tuples(self, tuple_list):
        # [FUNCTION DOC]
        # Parametrized over lists of one or two empty tuples. Verifies the Index constructor result
        # equals the MultiIndex built from the same tuples.
        # GH #45608
        result = Index(tuple_list)
        expected = MultiIndex.from_tuples(tuple_list)

        tm.assert_index_equal(result, expected)

    def test_index_string_inference(self):
        # [FUNCTION DOC]
        # Verifies that with the future string-inference option enabled, an all-string list yields the
        # NaN-backed string dtype, while a mixed string/int list stays object dtype.
        # GH#54430
        expected = Index(["a", "b"], dtype=pd.StringDtype(na_value=np.nan))
        with pd.option_context("future.infer_string", True):
            ser = Index(["a", "b"])
        tm.assert_index_equal(ser, expected)

        expected = Index(["a", 1], dtype="object")
        with pd.option_context("future.infer_string", True):
            ser = Index(["a", 1])
        tm.assert_index_equal(ser, expected)

    @pytest.mark.parametrize("klass", [Series, Index])
    def test_inference_on_pandas_objects(self, klass):
        # [FUNCTION DOC]
        # Parametrized over Series and Index containers. Verifies that an object-dtype container of a
        # single Timestamp stays object dtype when wrapped in an Index (no inference on pandas objects).
        # GH#56012
        obj = klass([pd.Timestamp("2019-12-31")], dtype=object)
        result = Index(obj)
        assert result.dtype == np.object_

    def test_constructor_not_read_only(self):
        # [FUNCTION DOC]
        # Verifies that an Index built from an object Series exposes a writeable underlying array
        # (the data is not marked read-only).
        # GH#57130
        ser = Series([1, 2], dtype=object)
        idx = Index(ser)
        assert idx._values.flags.writeable
