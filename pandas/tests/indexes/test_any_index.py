"""
Tests that can be parametrized over _any_ Index object.
"""

# [MODULE DOC]
# Tests that apply to every kind of Index. All tests use the shared index
# fixture, which is parametrized over many index types (numeric, string,
# datetime-like, interval, multi-level, and so on). Covers truthiness, hashing,
# mutability, mapping with an identity function, name handling, views, ravel,
# conversion to series, pickling, scalar getitem and lookup errors, string
# rendering and reduction-axis validation.

import re

import numpy as np
import pytest

from pandas.errors import InvalidIndexError

from pandas import StringDtype
import pandas._testing as tm


def test_boolean_context_compat(index):
    # [FUNCTION DOC]
    # Evaluating an index in a boolean context (conditional or explicit bool conversion) raises a value error saying the truth value is ambiguous.
    # GH#7897
    with pytest.raises(ValueError, match="The truth value of a"):
        if index:
            pass

    with pytest.raises(ValueError, match="The truth value of a"):
        bool(index)


def test_hash_error(index):
    # [FUNCTION DOC]
    # Hashing any index raises a type error stating that the index type is unhashable.
    with pytest.raises(TypeError, match=f"unhashable type: '{type(index).__name__}'"):
        hash(index)


def test_mutability(index):
    # [FUNCTION DOC]
    # Skips empty indexes. Assigning to an element of an index raises a type error saying that mutable operations are not supported.
    if not len(index):
        pytest.skip("Test doesn't make sense for empty index")
    msg = "Index does not support mutable operations"
    with pytest.raises(TypeError, match=msg):
        index[0] = index[0]


@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
def test_map_identity_mapping(index, request, using_infer_string):
    # [FUNCTION DOC]
    # Mapping each element to itself should return an index equal to the original (loose type comparison). Expects a failure (marked as expected-fail) for python-storage string dtype when string inference is off, since dtype is not preserved. Special case: for an object index whose result dtype is boolean or string, only element-wise equality is checked and the test returns early. Ignores the period-dtype business-frequency deprecation warning.
    # GH#12766
    if (
        not using_infer_string
        and isinstance(index.dtype, StringDtype)
        and index.dtype.storage == "python"
    ):
        mark = pytest.mark.xfail(reason="Does not preserve dtype")
        request.applymarker(mark)

    result = index.map(lambda x: x)
    if index.dtype == object and (result.dtype in (bool, "string")):
        assert (index == result).all()
        # TODO: could work that into the 'exact="equiv"'?
        return  # FIXME: doesn't belong in this file anymore!
    tm.assert_index_equal(result, index, exact="equiv")


def test_wrong_number_names(index):
    # [FUNCTION DOC]
    # Setting a names list whose length differs from the number of levels raises a value error whose message begins with "Length".
    names = index.nlevels * ["apple", "banana", "carrot"]
    with pytest.raises(ValueError, match="^Length"):
        index.names = names


def test_view_preserves_name(index):
    # [FUNCTION DOC]
    # A default view of the index has the same name as the original.
    assert index.view().name == index.name


def test_ravel(index):
    # [FUNCTION DOC]
    # Flattening an index returns an index equal to the original (view on self rather than an array).
    # GH#19956 ravel returning ndarray is deprecated, in 2.0 returns a view on self
    res = index.ravel()
    tm.assert_index_equal(res, index)


class TestConversion:
    # [CLASS DOC]
    # Tests for converting an index to other containers.
    def test_to_series(self, index):
        # [FUNCTION DOC]
        # Converting to a series produces new value storage and a new index object (copy semantics) and keeps the name.
        # assert that we are creating a copy of the index

        ser = index.to_series()
        assert ser.values is not index.values
        assert ser.index is not index
        assert ser.name == index.name

    def test_to_series_with_arguments(self, index):
        # [FUNCTION DOC]
        # With an explicit index argument, the series uses exactly that index object but still has new values and the original name. With a name argument, the series gets that name, a new index object and new values.
        # GH#18699

        # index kwarg
        ser = index.to_series(index=index)

        assert ser.values is not index.values
        assert ser.index is index
        assert ser.name == index.name

        # name kwarg
        ser = index.to_series(name="__test")

        assert ser.values is not index.values
        assert ser.index is not index
        assert ser.name != index.name

    def test_tolist_matches_list(self, index):
        # [FUNCTION DOC]
        # Converting to a Python list gives the same elements as iterating.
        assert index.tolist() == list(index)


class TestRoundTrips:
    # [CLASS DOC]
    # Serialization round-trip tests.
    def test_pickle_roundtrip(self, index, temp_file):
        # [FUNCTION DOC]
        # Round-trips the index through pickle using a temporary file fixture; the result must equal the original with exact type, and for multi-level indexes also have equal levels.
        result = tm.round_trip_pickle(index, temp_file)
        tm.assert_index_equal(result, index, exact=True)
        if result.nlevels > 1:
            # GH#8367 round-trip with timezone
            assert index.equal_levels(result)

    def test_pickle_preserves_name(self, index, temp_file):
        # [FUNCTION DOC]
        # Temporarily renames the index, pickles and unpickles it, and checks equality with the original; then restores the original name.
        original_name, index.name = index.name, "foo"
        unpickled = tm.round_trip_pickle(index, temp_file)
        assert index.equals(unpickled)
        index.name = original_name


class TestIndexing:
    # [CLASS DOC]
    # Scalar and slice access and location error tests.
    def test_getitem_0d_ndarray(self, index):
        # [FUNCTION DOC]
        # Skips empty indexes. Indexing with a zero-dimensional integer array behaves as indexing with the integer zero.
        # GH#55601
        if len(index) == 0:
            pytest.skip(reason="Test assumes non-empty index")
        key = np.array(0)
        result = index[key]

        assert result == index[0]

    def test_get_loc_listlike_raises_invalid_index_error(self, index):
        # [FUNCTION DOC]
        # Passing a list-like array (integer or boolean) to the location lookup raises the invalid-index error (never a type error), with the offending array's text in the message.
        # and never TypeError
        key = np.array([0, 1], dtype=np.intp)

        with pytest.raises(InvalidIndexError, match=r"\[0 1\]"):
            index.get_loc(key)

        with pytest.raises(InvalidIndexError, match=r"\[False  True\]"):
            index.get_loc(key.astype(bool))

    def test_getitem_ellipsis(self, index):
        # [FUNCTION DOC]
        # Indexing with an ellipsis returns an index equal to, but not the same object as, the original.
        # GH#21282
        result = index[...]
        assert result.equals(index)
        assert result is not index

    def test_slice_keeps_name(self, index):
        # [FUNCTION DOC]
        # Slicing keeps the index name.
        assert index.name == index[1:].name

    @pytest.mark.parametrize("item", [101, "no_int", 2.5])
    def test_getitem_error(self, index, item):
        # [FUNCTION DOC]
        # Parametrized over an out-of-bounds integer, a string and a float. Scalar indexing with these raises an index error; the message may be any of: out of bounds for the axis, only-integers-slices-... message from numpy, or the generic out-of-bounds message used by Arrow-backed strings.
        msg = "|".join(
            [
                r"index 101 is out of bounds for axis 0 with size [\d]+",
                re.escape(
                    "only integers, slices (`:`), ellipsis (`...`), "
                    "numpy.newaxis (`None`) and integer or boolean arrays "
                    "are valid indices"
                ),
                "index out of bounds",  # string[pyarrow]
            ]
        )
        with pytest.raises(IndexError, match=msg):
            index[item]


class TestRendering:
    # [CLASS DOC]
    # String rendering tests.
    def test_str(self, index):
        # [FUNCTION DOC]
        # After naming the index "foo", its string form includes the quoted name and the class name.
        # test the string repr
        index.name = "foo"
        assert "'foo'" in str(index)
        assert type(index).__name__ in str(index)


class TestReductions:
    # [CLASS DOC]
    # Reduction argument validation tests.
    def test_argmax_axis_invalid(self, index):
        # [FUNCTION DOC]
        # argmax, argmin, min and max with an axis beyond the single dimension (positive or negative out of range) raise a value error with the message that axis must be fewer than the number of dimensions.
        # GH#23081
        msg = r"`axis` must be fewer than the number of dimensions \(1\)"
        with pytest.raises(ValueError, match=msg):
            index.argmax(axis=1)
        with pytest.raises(ValueError, match=msg):
            index.argmin(axis=2)
        with pytest.raises(ValueError, match=msg):
            index.min(axis=-2)
        with pytest.raises(ValueError, match=msg):
            index.max(axis=-3)
