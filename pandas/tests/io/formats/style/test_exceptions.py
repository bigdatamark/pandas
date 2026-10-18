# [MODULE DOC]
# Tests for the argument validation errors raised when concatenating styler
# objects. Skipped at import time when jinja2 is missing (the skip call sits
# before the remaining imports on purpose). Defines a small two-row frame
# fixture and a styler fixture built from it with no unique id suffix.
import pytest

jinja2 = pytest.importorskip("jinja2")

from pandas import (
    DataFrame,
    MultiIndex,
)

from pandas.io.formats.style import Styler


@pytest.fixture
def df():
    # [FUNCTION DOC]
    # HELPER: fixture returning a frame with columns A and B, index labels x
    # and y, an integer first column and a float second column.
    return DataFrame(
        data=[[0, -0.609], [1, -1.228]],
        columns=["A", "B"],
        index=["x", "y"],
    )


@pytest.fixture
def styler(df):
    # [FUNCTION DOC]
    # HELPER: fixture returning a styler over the df fixture with the unique id
    # length set to zero, so rendered ids carry no random part.
    return Styler(df, uuid_len=0)


def test_concat_bad_columns(styler):
    # [FUNCTION DOC]
    # Concatenating a styler whose frame has different columns raises a value
    # error saying the other data must have the same columns.
    msg = "`other.data` must have same columns as `Styler.data"
    with pytest.raises(ValueError, match=msg):
        styler.concat(DataFrame([[1, 2]]).style)


def test_concat_bad_type(styler):
    # [FUNCTION DOC]
    # Concatenating something that is not a styler (a plain frame) raises a
    # type error saying the other object must be a styler.
    msg = "`other` must be of type `Styler`"
    with pytest.raises(TypeError, match=msg):
        styler.concat(DataFrame([[1, 2]]))


def test_concat_bad_index_levels(styler, df):
    # [FUNCTION DOC]
    # Concatenating a styler whose index has a different number of levels
    # (a two-level index on a copy of the fixture frame versus the flat index
    # of the base styler) raises a value error about the level count.
    df = df.copy()
    df.index = MultiIndex.from_tuples([(0, 0), (1, 1)])
    msg = "number of index levels must be same in `other`"
    with pytest.raises(ValueError, match=msg):
        styler.concat(df.style)
