# [MODULE DOC]
# Pickle compatibility check for MultiIndex construction.
import pytest

from pandas import MultiIndex


def test_pickle_compat_construction():
    # [FUNCTION DOC]
    # Constructing a MultiIndex with no levels and codes raises a type error saying
    # both must be passed (needed for unpickling).
    # this is testing for pickle compat
    # need an object to create with
    with pytest.raises(TypeError, match="Must pass both levels and codes"):
        MultiIndex()
