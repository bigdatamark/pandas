# [MODULE DOC]
# Test that monotonicity flags of PeriodIndex, DatetimeIndex and timedelta Index, and of their
# engines, agree and are never increasing when NaT is present.
from pandas import (
    Index,
    NaT,
    date_range,
)


def test_is_monotonic_with_nat():
    # [FUNCTION DOC]
    # Regression test: for datetime, period and timedelta indexes (and their private engines) over
    # three days, checks monotonic increasing, not decreasing, unique; after inserting NaT at the
    # start or at the end, the objects are neither monotonic increasing nor decreasing but remain
    # unique (including hourly period conversion for the end case).
    # GH#31437
    # PeriodIndex.is_monotonic_increasing should behave analogously to DatetimeIndex,
    #  in particular never be monotonic when we have NaT
    dti = date_range("2016-01-01", periods=3)
    pi = dti.to_period("D")
    tdi = Index(dti.view("timedelta64[ns]"))

    for obj in [pi, pi._engine, dti, dti._engine, tdi, tdi._engine]:
        if isinstance(obj, Index):
            # i.e. not Engines
            assert obj.is_monotonic_increasing
        assert obj.is_monotonic_increasing
        assert not obj.is_monotonic_decreasing
        assert obj.is_unique

    dti1 = dti.insert(0, NaT)
    pi1 = dti1.to_period("D")
    tdi1 = Index(dti1.view("timedelta64[ns]"))

    for obj in [pi1, pi1._engine, dti1, dti1._engine, tdi1, tdi1._engine]:
        if isinstance(obj, Index):
            # i.e. not Engines
            assert not obj.is_monotonic_increasing
        assert not obj.is_monotonic_increasing
        assert not obj.is_monotonic_decreasing
        assert obj.is_unique

    dti2 = dti.insert(3, NaT)
    pi2 = dti2.to_period("h")
    tdi2 = Index(dti2.view("timedelta64[ns]"))

    for obj in [pi2, pi2._engine, dti2, dti2._engine, tdi2, tdi2._engine]:
        if isinstance(obj, Index):
            # i.e. not Engines
            assert not obj.is_monotonic_increasing
        assert not obj.is_monotonic_increasing
        assert not obj.is_monotonic_decreasing
        assert obj.is_unique
