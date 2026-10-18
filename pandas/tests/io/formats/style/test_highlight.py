# [MODULE DOC]
# Tests for the highlight methods of the table styler (null, minimum, maximum,
# between bounds, quantile). Each test compares the computed per-cell style
# context with the expected mapping from (row, column) to a background colour
# style. The shared frame fixture is parametrized over float data with nan and
# nullable integer data with the missing-value marker. The jinja2 skip call runs
# at import time before the styler import.
import numpy as np
import pytest

from pandas import (
    NA,
    DataFrame,
    IndexSlice,
)

pytest.importorskip("jinja2")

from pandas.io.formats.style import Styler


@pytest.fixture(params=[(None, "float64"), (NA, "Int64")])
def df(request):
    # [FUNCTION DOC]
    # HELPER: parametrized fixture (GH 45804) building a frame with columns A
    # (zero, missing, ten) and B (one, missing, two); either float64 with nan, or the
    # nullable Int64 type with the missing-value marker. Returns the frame.
    # GH 45804
    dtype = request.param[1]
    item = np.nan if dtype == "float64" else NA
    return DataFrame(
        {"A": [0, item, 10], "B": [1, request.param[0], 2]}, dtype=request.param[1]
    )


@pytest.fixture
def styler(df):
    # [FUNCTION DOC]
    # HELPER: fixture returning a styler over the df fixture with no unique id
    # suffix.
    return Styler(df, uuid_len=0)


def test_highlight_null(styler):
    # [FUNCTION DOC]
    # Verifies the default null highlight colours exactly the two missing cells red.
    result = styler.highlight_null()._compute().ctx
    expected = {
        (1, 0): [("background-color", "red")],
        (1, 1): [("background-color", "red")],
    }
    assert result == expected


def test_highlight_null_subset(styler):
    # [FUNCTION DOC]
    # Regression test (GH 31345): chained null highlights restricted by subset to
    # column A in red and column B in green colour the missing cells in the matching
    # colours.
    # GH 31345
    result = (
        styler.highlight_null(color="red", subset=["A"])
        .highlight_null(color="green", subset=["B"])
        ._compute()
        .ctx
    )
    expected = {
        (1, 0): [("background-color", "red")],
        (1, 1): [("background-color", "green")],
    }
    assert result == expected


@pytest.mark.parametrize("f", ["highlight_min", "highlight_max"])
def test_highlight_minmax_basic(df, f):
    # [FUNCTION DOC]
    # Verifies row-wise (axis one) minimum or maximum highlighting in red on the df
    # fixture, ignoring the missing row. Parametrized over method name; for the
    # minimum method the frame is negated first so that the same cells are expected.
    expected = {
        (0, 1): [("background-color", "red")],
        # ignores NaN row,
        (2, 0): [("background-color", "red")],
    }
    if f == "highlight_min":
        df = -df
    result = getattr(df.style, f)(axis=1, color="red")._compute().ctx
    assert result == expected


@pytest.mark.parametrize("f", ["highlight_min", "highlight_max"])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"axis": None, "color": "red"},  # test axis
        {"axis": 0, "subset": ["A"], "color": "red"},  # test subset and ignores NaN
        {"axis": None, "props": "background-color: red"},  # test props
    ],
)
def test_highlight_minmax_ext(df, f, kwargs):
    # [FUNCTION DOC]
    # Verifies extra options of minimum/maximum highlighting: whole-frame axis, a
    # column subset with column axis, and a props string instead of a colour. All
    # should highlight only the cell in row two, column A, ignoring missing values.
    # Parametrized over method and keyword set; the minimum case negates the frame.
    expected = {(2, 0): [("background-color", "red")]}
    if f == "highlight_min":
        df = -df
    result = getattr(df.style, f)(**kwargs)._compute().ctx
    assert result == expected


@pytest.mark.parametrize("f", ["highlight_min", "highlight_max"])
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_highlight_minmax_nulls(f, axis):
    # [FUNCTION DOC]
    # Regression test (GH 42750): with missing values in a frame built from the
    # missing-value marker, nan and none, minimum/maximum highlighting for axis none,
    # zero and one highlights tied extremes in yellow (and an extra cell for axis
    # one). Parametrized over method and axis; data differs by method (maximum or
    # minimum).
    # GH 42750
    expected = {
        (1, 0): [("background-color", "yellow")],
        (1, 1): [("background-color", "yellow")],
    }
    if axis == 1:
        expected.update({(2, 1): [("background-color", "yellow")]})

    if f == "highlight_max":
        df = DataFrame({"a": [NA, 1, None], "b": [np.nan, 1, -1]})
    else:
        df = DataFrame({"a": [NA, -1, None], "b": [np.nan, -1, 1]})

    result = getattr(df.style, f)(axis=axis)._compute().ctx
    assert result == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"left": 0, "right": 1},  # test basic range
        {"left": 0, "right": 1, "props": "background-color: yellow"},  # test props
        {"left": -100, "right": 100, "subset": IndexSlice[[0, 1], :]},  # test subset
        {"left": 0, "subset": IndexSlice[[0, 1], :]},  # test no right
        {"right": 1},  # test no left
        {"left": [0, 0, 11], "axis": 0},  # test left as sequence
        {"left": DataFrame({"A": [0, 0, 11], "B": [1, 1, 11]}), "axis": None},  # axis
        {"left": 0, "right": [0, 1], "axis": 1},  # test sequence right
    ],
)
def test_highlight_between(styler, kwargs):
    # [FUNCTION DOC]
    # Verifies the between highlight with various keyword sets (basic range, custom
    # props, subset with wide bounds, only left, only right, left as a sequence
    # along rows, left as a frame with no axis, right as a sequence along columns).
    # Each must highlight in yellow the first row of the styler fixture.
    expected = {
        (0, 0): [("background-color", "yellow")],
        (0, 1): [("background-color", "yellow")],
    }
    result = styler.highlight_between(**kwargs)._compute().ctx
    assert result == expected


@pytest.mark.parametrize(
    "arg, map, axis",
    [
        ("left", [1, 2], 0),  # 0 axis has 3 elements not 2
        ("left", [1, 2, 3], 1),  # 1 axis has 2 elements not 3
        ("left", np.array([[1, 2], [1, 2]]), None),  # df is (2,3) not (2,2)
        ("right", [1, 2], 0),  # same tests as above for 'right' not 'left'
        ("right", [1, 2, 3], 1),  # ..
        ("right", np.array([[1, 2], [1, 2]]), None),  # ..
    ],
)
def test_highlight_between_raises(arg, styler, map, axis):
    # [FUNCTION DOC]
    # Verifies that a left or right bound with the wrong shape for the given axis
    # (sequence too short or long, wrongly shaped two-dimensional array) raises a
    # value error saying the supplied argument is not the correct shape. Parametrized
    # over argument name, bound value (the parameter named map) and axis.
    msg = f"supplied '{arg}' is not correct shape"
    with pytest.raises(ValueError, match=msg):
        styler.highlight_between(**{arg: map, "axis": axis})._compute()


def test_highlight_between_raises2(styler):
    # [FUNCTION DOC]
    # Verifies that an invalid inclusive argument (an unknown string or an integer)
    # raises a value error listing the allowed values.
    msg = "values can be 'both', 'left', 'right', or 'neither'"
    with pytest.raises(ValueError, match=msg):
        styler.highlight_between(inclusive="badstring")._compute()

    with pytest.raises(ValueError, match=msg):
        styler.highlight_between(inclusive=1)._compute()


@pytest.mark.parametrize(
    "inclusive, expected",
    [
        (
            "both",
            {
                (0, 0): [("background-color", "yellow")],
                (0, 1): [("background-color", "yellow")],
            },
        ),
        ("neither", {}),
        ("left", {(0, 0): [("background-color", "yellow")]}),
        ("right", {(0, 1): [("background-color", "yellow")]}),
    ],
)
def test_highlight_between_inclusive(styler, inclusive, expected):
    # [FUNCTION DOC]
    # Verifies the inclusive option with bounds zero and one on a two-row subset:
    # both ends highlights both cells, neither none, left only the lower, right only
    # the upper.
    kwargs = {"left": 0, "right": 1, "subset": IndexSlice[[0, 1], :]}
    result = styler.highlight_between(**kwargs, inclusive=inclusive)._compute()
    assert result.ctx == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"q_left": 0.5, "q_right": 1, "axis": 0},  # base case
        {"q_left": 0.5, "q_right": 1, "axis": None},  # test axis
        {"q_left": 0, "q_right": 1, "subset": IndexSlice[2, :]},  # test subset
        {"q_left": 0.5, "axis": 0},  # test no high
        {"q_right": 1, "subset": IndexSlice[2, :], "axis": 1},  # test no low
        {"q_left": 0.5, "axis": 0, "props": "background-color: yellow"},  # tst prop
    ],
)
def test_highlight_quantile(styler, kwargs):
    # [FUNCTION DOC]
    # Verifies quantile highlighting with several keyword sets (base case, axis none,
    # subset, only lower quantile, only upper quantile with column axis, custom
    # props); each expects row two highlighted yellow in both columns.
    expected = {
        (2, 0): [("background-color", "yellow")],
        (2, 1): [("background-color", "yellow")],
    }
    result = styler.highlight_quantile(**kwargs)._compute().ctx
    assert result == expected


@pytest.mark.parametrize(
    "f,kwargs",
    [
        ("highlight_min", {"axis": 1, "subset": IndexSlice[1, :]}),
        ("highlight_max", {"axis": 0, "subset": [0]}),
        ("highlight_quantile", {"axis": None, "q_left": 0.6, "q_right": 0.8}),
        ("highlight_between", {"subset": [0]}),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        int,
        float,
        "datetime64[ns]",
        str,
        "timedelta64[ns]",
    ],
)
def test_all_highlight_dtypes(f, kwargs, dtype):
    # [FUNCTION DOC]
    # Verifies the highlight methods work for integer, float, datetime, string and
    # timedelta frames. Parametrized over method with keywords and dtype. Returns
    # early (passes) for the quantile method on strings; for the between method the
    # lower bound is set from the frame. Expects only cell (1, 0) in yellow.
    # KNOWN ISSUE: the between case assigns into the keyword dictionary that is shared
    # across parametrizations, so the lower bound persists between runs.
    df = DataFrame([[0, 10], [20, 30]], dtype=dtype)
    if f == "highlight_quantile" and isinstance(df.iloc[0, 0], (str)):
        return None  # quantile incompatible with str
    if f == "highlight_between":
        kwargs["left"] = df.iloc[1, 0]  # set the range low for testing

    expected = {(1, 0): [("background-color", "yellow")]}
    result = getattr(df.style, f)(**kwargs)._compute().ctx
    assert result == expected
