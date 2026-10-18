# [MODULE DOC]
# Tests for astype on CategoricalIndex: to object, to interval, to category with and without
# a dtype, and date round trip.
from datetime import date

import numpy as np
import pytest

from pandas.errors import Pandas4Warning

from pandas import (
    Categorical,
    CategoricalDtype,
    CategoricalIndex,
    Index,
    IntervalIndex,
)
import pandas._testing as tm


class TestAstype:
    # [CLASS DOC]
    # Groups astype tests for CategoricalIndex; no state.
    def test_astype(self):
        # [FUNCTION DOC]
        # Verifies casting to object gives a plain Index equal in value to the categorical, and that
        # casting a categorical index of intervals (with a missing code) to interval yields the
        # corresponding IntervalIndex with NaN for the missing entry, round-tripping through an
        # IntervalIndex built from its values.
        ci = CategoricalIndex(list("aabbca"), categories=list("cab"), ordered=False)

        result = ci.astype(object)
        tm.assert_index_equal(result, Index(np.array(ci), dtype=object))

        # this IS equal, but not the same class
        assert result.equals(ci)
        assert isinstance(result, Index)
        assert not isinstance(result, CategoricalIndex)

        # interval
        ii = IntervalIndex.from_arrays(left=[-0.001, 2.0], right=[2, 4], closed="right")

        ci = CategoricalIndex(
            Categorical.from_codes([0, 1, -1], categories=ii, ordered=True)
        )

        result = ci.astype("interval")
        expected = ii.take([0, 1, -1], allow_fill=True, fill_value=np.nan)
        tm.assert_index_equal(result, expected)

        result = IntervalIndex(result.values)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("name", [None, "foo"])
    @pytest.mark.parametrize("dtype_ordered", [True, False])
    @pytest.mark.parametrize("index_ordered", [True, False])
    def test_astype_category(self, name, dtype_ordered, index_ordered):
        # [FUNCTION DOC]
        # Parametrized over name (none or "foo"), dtype ordered flag and index ordered flag. Verifies
        # casting to a CategoricalDtype without categories keeps categories and takes the dtype's
        # orderedness and the name. Casting to a dtype with a subset of categories emits the
        # Pandas4Warning about constructing with values outside the dtype and matches direct
        # construction; casting to the string "category" (tested only when unordered) is a no-op.
        # GH#18630
        index = CategoricalIndex(
            list("aabbca"), categories=list("cab"), ordered=index_ordered
        )
        if name:
            index = index.rename(name)

        # standard categories
        dtype = CategoricalDtype(ordered=dtype_ordered)
        result = index.astype(dtype)
        expected = CategoricalIndex(
            index.tolist(),
            name=name,
            categories=index.categories,
            ordered=dtype_ordered,
        )
        tm.assert_index_equal(result, expected)

        # non-standard categories
        dtype = CategoricalDtype(index.unique().tolist()[:-1], dtype_ordered)
        msg = "Constructing a Categorical with a dtype and values containing"
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            result = index.astype(dtype)
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            expected = CategoricalIndex(index.tolist(), name=name, dtype=dtype)
        tm.assert_index_equal(result, expected)

        if dtype_ordered is False:
            # dtype='category' can't specify ordered, so only test once
            result = index.astype("category")
            expected = index
            tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("box", [True, False])
    def test_categorical_date_roundtrip(self, box):
        # [FUNCTION DOC]
        # Parametrized over whether the object Index is first boxed as an extension array. Verifies
        # that casting an object Index of date objects to category and back to object preserves the
        # object dtype and the date type of the elements.
        # astype to categorical and back should preserve date objects
        v = date.today()

        obj = Index([v, v])
        assert obj.dtype == object
        if box:
            obj = obj.array

        cat = obj.astype("category")

        rtrip = cat.astype(object)
        assert rtrip.dtype == object
        assert type(rtrip[0]) is date
