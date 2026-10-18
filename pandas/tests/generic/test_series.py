# [MODULE DOC]
# Series-specific tests in the generic test package: axis naming with multi-level
# indexes, truth-value ambiguity, boolean data extraction, and metadata propagation
# through resample, transpose and concat.
from operator import methodcaller

import numpy as np
import pytest

import pandas as pd
from pandas import (
    MultiIndex,
    Series,
    date_range,
)
import pandas._testing as tm


class TestSeries:
    # [CLASS DOC]
    # Groups Series tests for axis naming, truthiness and metadata. Stateless.
    @pytest.mark.parametrize("func", ["rename_axis", "_set_axis_name"])
    def test_set_axis_name_mi(self, func):
        # [FUNCTION DOC]
        # Verifies setting level names on a multi-index series through the private
        # setter or the public rename method.
        # The source keeps its original names. Parametrized over the two method names.
        # KNOWN ISSUE: the last assertion uses a comma, so the second item is only an
        # assertion message and the names are only checked for being non-empty, not
        # equal to the new names.
        ser = Series(
            [11, 21, 31],
            index=MultiIndex.from_tuples(
                [("A", x) for x in ["a", "B", "c"]], names=["l1", "l2"]
            ),
        )

        result = methodcaller(func, ["L1", "L2"])(ser)
        assert ser.index.name is None
        assert ser.index.names == ["l1", "l2"]
        assert result.index.name is None
        assert result.index.names, ["L1", "L2"]

    def test_set_axis_name_raises(self):
        # [FUNCTION DOC]
        # Verifies the private axis-name setter raises a value error for axis one on a
        # Series ("No axis named 1").
        ser = Series([1])
        msg = "No axis named 1 for object type Series"
        with pytest.raises(ValueError, match=msg):
            ser._set_axis_name(name="a", axis=1)

    def test_get_bool_data_preserve_dtype(self):
        # [FUNCTION DOC]
        # Verifies that extracting boolean data from an all-boolean series returns an
        # equal series (dtype preserved).
        ser = Series([True, False, True])
        result = ser._get_bool_data()
        tm.assert_series_equal(result, ser)

    @pytest.mark.parametrize("data", [np.nan, pd.NaT, True, False])
    def test_nonzero_single_element_raise_1(self, data):
        # [FUNCTION DOC]
        # Verifies converting a one-element series to bool raises a value error about
        # ambiguous truth value
        # for missing values, NaT and booleans. Parametrized over the element.
        # single item nan to raise
        series = Series([data])

        msg = "The truth value of a Series is ambiguous"
        with pytest.raises(ValueError, match=msg):
            bool(series)

    @pytest.mark.parametrize("data", [(True, True), (False, False)])
    def test_nonzero_multiple_element_raise(self, data):
        # [FUNCTION DOC]
        # Verifies that a series whose single element is a two-element tuple of equal
        # booleans still raises the ambiguous truth value error when converted to bool.
        # Parametrized over the tuple.
        # multiple bool are still an error
        msg_err = "The truth value of a Series is ambiguous"
        series = Series([data])
        with pytest.raises(ValueError, match=msg_err):
            bool(series)

    @pytest.mark.parametrize("data", [1, 0, "a", 0.0])
    def test_nonbool_single_element_raise(self, data):
        # [FUNCTION DOC]
        # Verifies converting a single-element series to bool raises for non-bool
        # elements (integers, string, float). Parametrized over the element.
        # single non-bool are an error
        msg_err1 = "The truth value of a Series is ambiguous"
        series = Series([data])
        with pytest.raises(ValueError, match=msg_err1):
            bool(series)

    def test_metadata_propagation_indiv_resample(self):
        # [FUNCTION DOC]
        # Verifies metadata (such as name) is preserved by resampling a second-frequency
        # named series to one minute with mean, min and a custom apply of sum.
        # resample
        ts = Series(
            np.random.default_rng(2).random(1000),
            index=date_range("20130101", periods=1000, freq="s"),
            name="foo",
        )
        result = ts.resample("1min").mean()
        tm.assert_metadata_equivalent(ts, result)

        result = ts.resample("1min").min()
        tm.assert_metadata_equivalent(ts, result)

        result = ts.resample("1min").apply(lambda x: x.sum())
        tm.assert_metadata_equivalent(ts, result)

    def test_metadata_propagation_indiv(self, monkeypatch):
        # [FUNCTION DOC]
        # Verifies metadata of transpose, and custom metadata propagation in concat:
        # with the metadata list temporarily replaced by name and filename and
        # a custom finalize hook installed (restored by monkeypatch), concatenating two
        # series joins the filenames with a plus
        # and the result name is none because names differ.
        # check that the metadata matches up on the resulting ops

        ser = Series(range(3), range(3))
        ser.name = "foo"
        ser2 = Series(range(3), range(3))
        ser2.name = "bar"

        result = ser.T
        tm.assert_metadata_equivalent(ser, result)

        def finalize(self, other, method=None, **kwargs):
            # [FUNCTION DOC]
            # HELPER: replacement finalize hook for the enclosing test. For each
            # metadata name: on concat with the filename attribute sets
            # the plus-joined truthy filenames of the input objects; otherwise copies
            # the attribute from other (default none). Returns self.
            for name in self._metadata:
                if method == "concat" and name == "filename":
                    value = "+".join(
                        [
                            getattr(obj, name)
                            for obj in other.input_objs
                            if getattr(obj, name, None)
                        ]
                    )
                    object.__setattr__(self, name, value)
                else:
                    object.__setattr__(self, name, getattr(other, name, None))

            return self

        with monkeypatch.context() as m:
            m.setattr(Series, "_metadata", ["name", "filename"])
            m.setattr(Series, "__finalize__", finalize)

            ser.filename = "foo"
            ser2.filename = "bar"

            result = pd.concat([ser, ser2])
            assert result.filename == "foo+bar"
            assert result.name is None
