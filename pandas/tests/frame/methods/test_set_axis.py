# [MODULE DOC]
# Tests for set_axis on DataFrame and Series. A shared mixin class defines the tests
# against an abstract object fixture; two concrete subclasses (one for DataFrame, one for
# Series) provide the object, so every shared test runs for both container types.

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm


class SharedSetAxisTests:
    # [CLASS DOC]
    # Mixin holding set_axis tests common to Series and DataFrame. It is not collected on
    # its own; subclasses must override the obj fixture, otherwise it raises a
    # not-implemented error.
    @pytest.fixture
    def obj(self):
        # [FUNCTION DOC]
        # Abstract fixture. Subclasses override it to return the object under test.
        # Raises a not-implemented error if used directly.
        raise NotImplementedError("Implemented by subclasses")

    def test_set_axis(self, obj):
        # [FUNCTION DOC]
        # Verifies set_axis along the row axis with as many new labels as the object has rows
        # (GH 14636). The result must equal a copy whose index attribute was assigned directly.
        # GH14636; this tests setting index for both Series and DataFrame
        new_index = list("abcd")[: len(obj)]
        expected = obj.copy()
        expected.index = new_index
        result = obj.set_axis(new_index, axis=0)
        tm.assert_equal(expected, result)

    def test_set_axis_copy(self, obj):
        # [FUNCTION DOC]
        # Verifies set_axis returns a new object that shares memory with the original (no data copy)
        # for a series, or for every column of a frame in the first check, when called with an
        # explicit axis, and also when the axis argument is omitted (GH 47932).
        # KNOWN ISSUE: the second block repeats the first call verbatim but weakens the frame check to "any column shares memory" while its comment says no copy was made; it adds no distinct coverage.
        # Expected values equal the copy with the new index assigned directly.
        # Test copy keyword GH#47932
        new_index = list("abcd")[: len(obj)]

        orig = obj.iloc[:]
        expected = obj.copy()
        expected.index = new_index

        result = obj.set_axis(new_index, axis=0)
        tm.assert_equal(expected, result)
        assert result is not obj
        # check we did NOT make a copy
        if obj.ndim == 1:
            assert tm.shares_memory(result, obj)
        else:
            assert all(
                tm.shares_memory(result.iloc[:, i], obj.iloc[:, i])
                for i in range(obj.shape[1])
            )

        result = obj.set_axis(new_index, axis=0)
        tm.assert_equal(expected, result)
        assert result is not obj
        # check we DID NOT make a copy
        if obj.ndim == 1:
            assert tm.shares_memory(result, obj)
        else:
            assert any(
                tm.shares_memory(result.iloc[:, i], obj.iloc[:, i])
                for i in range(obj.shape[1])
            )

        res = obj.set_axis(new_index)
        tm.assert_equal(expected, res)
        # check we did NOT make a copy
        if res.ndim == 1:
            assert tm.shares_memory(res, orig)
        else:
            assert all(
                tm.shares_memory(res.iloc[:, i], orig.iloc[:, i])
                for i in range(res.shape[1])
            )

    def test_set_axis_unnamed_kwarg_warns(self, obj):
        # [FUNCTION DOC]
        # Calls set_axis with only the labels (axis omitted, default row axis) and expects a result equal to a copy with the index assigned directly.
        # KNOWN ISSUE: the name says it warns, but no warning is checked or expected.
        # omitting the "axis" parameter
        new_index = list("abcd")[: len(obj)]

        expected = obj.copy()
        expected.index = new_index

        result = obj.set_axis(new_index)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("axis", [3, "foo"])
    def test_set_axis_invalid_axis_name(self, axis, obj):
        # [FUNCTION DOC]
        # Parametrized over axis values 3 and a non-existent name. Expects a value error
        # containing the text "No axis named" when passing three labels.
        # wrong values for the "axis" parameter
        with pytest.raises(ValueError, match="No axis named"):
            obj.set_axis(list("abc"), axis=axis)

    def test_set_axis_setattr_index_not_collection(self, obj):
        # [FUNCTION DOC]
        # Assigning None directly to the index attribute must raise a type error saying the
        # index constructor must be called with a collection of some kind.
        # wrong type
        msg = (
            r"Index\(\.\.\.\) must be called with a collection of some "
            r"kind, None was passed"
        )
        with pytest.raises(TypeError, match=msg):
            obj.index = None

    def test_set_axis_setattr_index_wrong_length(self, obj):
        # [FUNCTION DOC]
        # Assigning an index one element too short must raise a value error reporting a length
        # mismatch with the expected and new element counts. For two-dimensional objects also
        # checks that assigning every other column label raises a length mismatch value error.
        # wrong length
        msg = (
            f"Length mismatch: Expected axis has {len(obj)} elements, "
            f"new values have {len(obj) - 1} elements"
        )
        with pytest.raises(ValueError, match=msg):
            obj.index = np.arange(len(obj) - 1)

        if obj.ndim == 2:
            with pytest.raises(ValueError, match="Length mismatch"):
                obj.columns = obj.columns[::2]


class TestDataFrameSetAxis(SharedSetAxisTests):
    # [CLASS DOC]
    # Runs the shared set_axis tests with a DataFrame, plus a DataFrame-specific test.
    @pytest.fixture
    def obj(self):
        # [FUNCTION DOC]
        # Fixture: returns a 3x3 float frame with columns A, B, C and integer-year index 2010 to 2012.
        df = DataFrame(
            {"A": [1.1, 2.2, 3.3], "B": [5.0, 6.1, 7.2], "C": [4.4, 5.5, 6.6]},
            index=[2010, 2011, 2012],
        )
        return df

    def test_set_axis_with_allows_duplicate_labels_false(self):
        # [FUNCTION DOC]
        # Regression test (GH 44958). A frame with the allows-duplicate-labels flag turned off
        # must still support set_axis with the keyword labels argument on the row axis;
        # the result equals the expected frame with the new index (flags are not compared).
        # GH#44958
        df = DataFrame([[1, 2], [3, 4]], columns=["a", "b"]).set_flags(
            allows_duplicate_labels=False
        )

        result = df.set_axis(labels=["x", "y"], axis=0)
        expected = DataFrame([[1, 2], [3, 4]], index=["x", "y"], columns=["a", "b"])
        tm.assert_frame_equal(result, expected, check_flags=False)


class TestSeriesSetAxis(SharedSetAxisTests):
    # [CLASS DOC]
    # Runs the shared set_axis tests with a Series.
    @pytest.fixture
    def obj(self):
        # [FUNCTION DOC]
        # Fixture: returns an int64 series of four values with index 1, 3, 5, 7.
        ser = Series(np.arange(4), index=[1, 3, 5, 7], dtype="int64")
        return ser
