# [MODULE DOC]
# Tests for iterating over a DatetimeIndex: nanosecond and timezone
# preservation and chunked iteration.
# Imports the internal datetimes array module to patch its chunk size.

import dateutil.tz
import numpy as np
import pytest

from pandas import (
    DatetimeIndex,
    date_range,
    to_datetime,
)
from pandas.core.arrays import datetimes


class TestDatetimeIndexIteration:
    # [CLASS DOC]
    # Groups tests of element-wise iteration over a DatetimeIndex.
    @pytest.mark.parametrize(
        "tz", [None, "UTC", "US/Central", dateutil.tz.tzoffset(None, -28800)]
    )
    def test_iteration_preserves_nanoseconds(self, tz):
        # [FUNCTION DOC]
        # Parametrized over timezone (none, UTC, a named zone, a fixed offset).
        # Verifies (GH#19603) that iterating an index with nanosecond precision yields
        # timestamps equal to the positionally indexed elements.
        # GH#19603
        index = DatetimeIndex(
            ["2018-02-08 15:00:00.168456358", "2018-02-08 15:00:00.168456359"], tz=tz
        )
        for i, ts in enumerate(index):
            assert ts == index[i]  # noqa: PLR1736 (unnecessary-list-index-lookup)

    def test_iter_readonly(self):
        # [FUNCTION DOC]
        # Verifies (GH#28055) that iterating an index built from a read-only numpy
        # datetime array does not raise.
        # GH#28055 ints_to_pydatetime with readonly array
        arr = np.array([np.datetime64("2012-02-15T12:00:00.000000000")])
        arr.setflags(write=False)
        dti = to_datetime(arr)
        list(dti)

    def test_iteration_preserves_tz(self):
        # [FUNCTION DOC]
        # Verifies (GH#8890) that iterating a timezone-aware hourly index yields
        # timestamps equal to the indexed elements, retaining timezone.
        # see GH#8890
        index = date_range("2012-01-01", periods=3, freq="h", tz="US/Eastern")

        for i, ts in enumerate(index):
            result = ts
            expected = index[i]  # noqa: PLR1736 (unnecessary-list-index-lookup)
            assert result == expected

    def test_iteration_preserves_tz2(self):
        # [FUNCTION DOC]
        # Same as the preceding timezone-preservation test but with a fixed-offset
        # dateutil timezone; also compares the internal base representation of each
        # yielded timestamp with the indexed one.
        index = date_range(
            "2012-01-01", periods=3, freq="h", tz=dateutil.tz.tzoffset(None, -28800)
        )

        for i, ts in enumerate(index):
            result = ts
            expected = index[i]  # noqa: PLR1736 (unnecessary-list-index-lookup)
            assert result._repr_base == expected._repr_base
            assert result == expected

    def test_iteration_preserves_tz3(self):
        # [FUNCTION DOC]
        # Verifies (GH#9100) that iterating an index parsed from offset-bearing strings
        # yields timestamps whose base representation and value equal the indexed ones.
        # GH#9100
        index = DatetimeIndex(
            ["2014-12-01 03:32:39.987000-08:00", "2014-12-01 04:12:34.987000-08:00"]
        )
        for i, ts in enumerate(index):
            result = ts
            expected = index[i]  # noqa: PLR1736 (unnecessary-list-index-lookup)
            assert result._repr_base == expected._repr_base
            assert result == expected

    @pytest.mark.parametrize("offset", [-5, -1, 0, 1])
    def test_iteration_over_chunksize(self, offset, monkeypatch):
        # [FUNCTION DOC]
        # Parametrized over offset (negative to positive) and uses the monkeypatch
        # fixture. With the module-level iteration chunk size patched to five, builds a
        # minute-frequency index with length smaller or larger than the chunk size and
        # verifies (GH#21012) iteration yields every element in order exactly once.
        # The patch is undone at context exit.
        # GH#21012
        chunksize = 5
        index = date_range(
            "2000-01-01 00:00:00", periods=chunksize - offset, freq="min"
        )
        num = 0
        with monkeypatch.context() as m:
            m.setattr(datetimes, "_ITER_CHUNKSIZE", chunksize)
            for stamp in index:
                assert index[num] == stamp
                num += 1
        assert num == len(index)
