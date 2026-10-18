"""generic datetimelike tests"""

# [MODULE DOC]
# Generic tests shared by the three datetime-like index types (period, timedelta, datetime), all driven through a single parametrized fixture holding a five-element daily-frequency example of each type. Covers membership, argsort with missing, identifiers, shifting, string rendering, views, mapping, frequency preservation under slicing, conditional replacement with string casting, and differencing.

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


class TestDatetimeLike:
    # [CLASS DOC]
    # Test class for behavior common to period, timedelta and datetime indexes.
    @pytest.fixture(
        params=[
            pd.period_range("20130101", periods=5, freq="D"),
            pd.TimedeltaIndex(
                [
                    "0 days 01:00:00",
                    "1 days 01:00:00",
                    "2 days 01:00:00",
                    "3 days 01:00:00",
                    "4 days 01:00:00",
                ],
                dtype="timedelta64[ns]",
                freq="D",
            ),
            pd.DatetimeIndex(
                ["2013-01-01", "2013-01-02", "2013-01-03", "2013-01-04", "2013-01-05"],
                dtype="datetime64[ns]",
                freq="D",
            ),
        ]
    )
    def simple_index(self, request):
        # [FUNCTION DOC]
        # HELPER: parametrized fixture yielding, in turn, a daily period range, a daily-frequency timedelta index of one-hour offsets, and a daily datetime index, all with five elements starting at the epoch-like 2013 date or day zero.
        return request.param

    def test_isin(self, simple_index):
        # [FUNCTION DOC]
        # On the first four elements: membership against the index itself or a list of its elements is all true; membership against a list of one element and an unrelated integer marks only that element.
        index = simple_index[:4]
        result = index.isin(index)
        assert result.all()

        result = index.isin(list(index))
        assert result.all()

        result = index.isin([index[2], 5])
        expected = np.array([False, False, True, False])
        tm.assert_numpy_array_equal(result, expected)

    def test_argsort_matches_array(self, simple_index):
        # [FUNCTION DOC]
        # After inserting a missing value at position one, the index sort order equals the sort order of the underlying array.
        idx = simple_index
        idx = idx.insert(1, pd.NaT)

        result = idx.argsort()
        expected = idx._data.argsort()
        tm.assert_numpy_array_equal(result, expected)

    def test_can_hold_identifiers(self, simple_index):
        # [FUNCTION DOC]
        # The private identifier check is false for datetime-like indexes.
        idx = simple_index
        key = idx[0]
        assert idx._can_hold_identifiers_and_holds_name(key) is False

    def test_shift_identity(self, simple_index):
        # [FUNCTION DOC]
        # Shifting by zero returns an equal index.
        idx = simple_index
        tm.assert_index_equal(idx, idx.shift(0))

    def test_shift_empty(self, simple_index):
        # [FUNCTION DOC]
        # Shifting an empty index by one returns an equal empty index.
        # GH#14811
        idx = simple_index[:0]
        tm.assert_index_equal(idx, idx.shift(1))

    def test_str(self, simple_index):
        # [FUNCTION DOC]
        # After naming a copy "foo", the string form does not include a length marker, includes the quoted name and class name, includes the timezone when tz-aware, and contains the dtype with frequency for periods or the frequency string otherwise.
        # test the string repr
        idx = simple_index.copy()
        idx.name = "foo"
        assert f"length={len(idx)}" not in str(idx)
        assert "'foo'" in str(idx)
        assert type(idx).__name__ in str(idx)

        if hasattr(idx, "tz"):
            if idx.tz is not None:
                assert idx.tz in str(idx)
        if isinstance(idx, pd.PeriodIndex):
            assert f"dtype='period[{idx.freqstr}]'" in str(idx)
        else:
            assert f"freq='{idx.freqstr}'" in str(idx)

    def test_view(self, simple_index):
        # [FUNCTION DOC]
        # Constructing the same class from the index returns an equal index; viewing the index as the class type raises a type error about not being able to change the data-type.
        idx = simple_index

        result = type(simple_index)(idx)
        tm.assert_index_equal(result, idx)

        msg = (
            "Cannot change data-type for array of references.|"
            "Cannot change data-type for object array.|"
        )
        with pytest.raises(TypeError, match=msg):
            idx.view(type(simple_index))

    def test_map_callable(self, simple_index):
        # [FUNCTION DOC]
        # Mapping adding the frequency to each element equals adding the frequency to the index; mapping the first element to NaT gives a general index with NaT first and the rest unchanged.
        index = simple_index
        expected = index + index.freq
        result = index.map(lambda x: x + index.freq)
        tm.assert_index_equal(result, expected)

        # map to NaT
        result = index.map(lambda x: pd.NaT if x == index[0] else x)
        expected = pd.Index([pd.NaT, *index[1:].tolist()])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "mapper",
        [
            lambda values, index: {i: e for e, i in zip(values, index)},
            lambda values, index: pd.Series(values, index, dtype=object),
        ],
    )
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_map_dictlike(self, mapper, simple_index):
        # [FUNCTION DOC]
        # Parametrized over a dictionary mapper or an object series mapper. Ignores a period business-day deprecation warning. Mapping through a dictionary of element-to-shifted-value gives the shifted index (frequency ignored for datetime and timedelta); mapping to NaT for the first element works; mapping through an empty mapper gives an all-NaN index of the same length.
        index = simple_index
        expected = index + index.freq

        # don't compare the freqs
        if isinstance(expected, (pd.DatetimeIndex, pd.TimedeltaIndex)):
            expected = expected._with_freq(None)

        result = index.map(mapper(expected, index))
        tm.assert_index_equal(result, expected)

        expected = pd.Index([pd.NaT, *index[1:].tolist()])
        result = index.map(mapper(expected, index))
        tm.assert_index_equal(result, expected)

        # empty map; these map to np.nan because we cannot know
        # to re-infer things
        expected = pd.Index([np.nan] * len(index))
        result = index.map(mapper([], []))
        tm.assert_index_equal(result, expected)

    def test_getitem_preserves_freq(self, simple_index):
        # [FUNCTION DOC]
        # A full slice of an index with a frequency preserves it.
        index = simple_index
        assert index.freq is not None

        result = index[:]
        assert result.freq == index.freq

    def test_where_cast_str(self, simple_index):
        # [FUNCTION DOC]
        # With a mask false for the last element, where with the string form of the first element (scalar or one-element list) equals where with the element itself; where with an unparseable string ("foo") falls back to object dtype and equals where on the object-cast index (scalar or list).
        index = simple_index

        mask = np.ones(len(index), dtype=bool)
        mask[-1] = False

        result = index.where(mask, str(index[0]))
        expected = index.where(mask, index[0])
        tm.assert_index_equal(result, expected)

        result = index.where(mask, [str(index[0])])
        tm.assert_index_equal(result, expected)

        expected = index.astype(object).where(mask, "foo")
        result = index.where(mask, "foo")
        tm.assert_index_equal(result, expected)

        result = index.where(mask, ["foo"])
        tm.assert_index_equal(result, expected)

    def test_diff(self, unit):
        # [FUNCTION DOC]
        # Uses the unit fixture. Differencing a datetime index built from integer values in the given unit gives a timedelta index of NaT followed by tens in the same unit.
        # GH 55080
        dti = pd.to_datetime([10, 20, 30], unit=unit).as_unit(unit)
        result = dti.diff(1)
        expected = pd.to_timedelta([pd.NaT, 10, 10], unit=unit).as_unit(unit)
        tm.assert_index_equal(result, expected)
