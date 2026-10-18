# [MODULE DOC]
# Tests for the styler features that rely on matplotlib colormaps: background and
# text gradients (colour maps, axis, thresholds for text colour, value limits,
# gradient map given as array, frame or series with alignment and shape errors,
# nullable dtypes) and bar colormaps, plus a plotting regression. Skipped at
# import time when matplotlib or jinja2 is missing. A module-wide marker applies
# a cleanup fixture that closes figures after each test.
import numpy as np
import pytest

from pandas import (
    DataFrame,
    IndexSlice,
    Series,
)

mpl = pytest.importorskip("matplotlib")
pytest.importorskip("jinja2")

from pandas.io.formats.style import Styler

pytestmark = pytest.mark.usefixtures("mpl_cleanup")


@pytest.fixture
def df():
    # [FUNCTION DOC]
    # HELPER: fixture returning a two-by-two integer frame with values one, two, two,
    # four and columns A and B.
    return DataFrame([[1, 2], [2, 4]], columns=["A", "B"])


@pytest.fixture
def styler(df):
    # [FUNCTION DOC]
    # HELPER: fixture returning a styler over the df fixture with no unique id suffix.
    return Styler(df, uuid_len=0)


@pytest.fixture
def df_blank():
    # [FUNCTION DOC]
    # HELPER: fixture returning a two-by-two frame of zeros with columns A, B and index
    # X, Y, used where colours must come from a gradient map instead of the data.
    return DataFrame([[0, 0], [0, 0]], columns=["A", "B"], index=["X", "Y"])


@pytest.fixture
def styler_blank(df_blank):
    # [FUNCTION DOC]
    # HELPER: fixture returning a styler over the df_blank fixture without unique id
    # suffix.
    return Styler(df_blank, uuid_len=0)


@pytest.mark.parametrize("f", ["background_gradient", "text_gradient"])
def test_function_gradient(styler, f):
    # [FUNCTION DOC]
    # Verifies that background and text gradients, with the default colormap or a
    # named one, give every cell a colour string (a hash colour in the first style
    # entry), and that cells in the same row get the same style because the default
    # scaling is per column. Parametrized over the two method names.
    for c_map in [None, "YlOrRd"]:
        result = getattr(styler, f)(cmap=c_map)._compute().ctx
        assert all("#" in x[0][1] for x in result.values())
        assert result[(0, 0)] == result[(0, 1)]
        assert result[(1, 0)] == result[(1, 1)]


@pytest.mark.parametrize("f", ["background_gradient", "text_gradient"])
def test_background_gradient_color(styler, f):
    # [FUNCTION DOC]
    # Verifies the exact style for one cell of column A selected with a subset: the
    # background gradient gives the lowest colour of the default map plus a black text
    # colour, while the text gradient gives only the colour entry.
    result = getattr(styler, f)(subset=IndexSlice[1, "A"])._compute().ctx
    if f == "background_gradient":
        assert result[(1, 0)] == [("background-color", "#fff7fb"), ("color", "#000000")]
    elif f == "text_gradient":
        assert result[(1, 0)] == [("color", "#fff7fb")]


@pytest.mark.parametrize(
    "axis, expected",
    [
        (0, ["low", "low", "high", "high"]),
        (1, ["low", "high", "low", "high"]),
        (None, ["low", "mid", "mid", "high"]),
    ],
)
@pytest.mark.parametrize("f", ["background_gradient", "text_gradient"])
def test_background_gradient_axis(styler, axis, expected, f):
    # [FUNCTION DOC]
    # Verifies colours for the Blues map on a two-by-two frame for axis zero, one and
    # none: the cells expect low, mid or high colours (with black or near-white text
    # for the background method; only the colour entry for the text method).
    # Parametrized over axis and method.
    if f == "background_gradient":
        colors = {
            "low": [("background-color", "#f7fbff"), ("color", "#000000")],
            "mid": [("background-color", "#abd0e6"), ("color", "#000000")],
            "high": [("background-color", "#08306b"), ("color", "#f1f1f1")],
        }
    elif f == "text_gradient":
        colors = {
            "low": [("color", "#f7fbff")],
            "mid": [("color", "#abd0e6")],
            "high": [("color", "#08306b")],
        }
    result = getattr(styler, f)(cmap="Blues", axis=axis)._compute().ctx
    for i, cell in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        assert result[cell] == colors[expected[i]]


@pytest.mark.parametrize(
    "cmap, expected",
    [
        (
            "PuBu",
            {
                (4, 5): [("background-color", "#86b0d3"), ("color", "#000000")],
                (4, 6): [("background-color", "#83afd3"), ("color", "#f1f1f1")],
            },
        ),
        (
            "YlOrRd",
            {
                (4, 8): [("background-color", "#fd913e"), ("color", "#000000")],
                (4, 9): [("background-color", "#fd8f3d"), ("color", "#f1f1f1")],
            },
        ),
        (
            None,
            {
                (7, 0): [("background-color", "#48c16e"), ("color", "#f1f1f1")],
                (7, 1): [("background-color", "#4cc26c"), ("color", "#000000")],
            },
        ),
    ],
)
def test_text_color_threshold(cmap, expected):
    # [FUNCTION DOC]
    # Regression test (GH 39888): on a ten-by-ten range frame scaled over the whole
    # frame, the automatically chosen text colour (black or near white) for selected
    # cells matches the expected background and text colours for the PuBu map, the
    # YlOrRd map and the default map.
    # GH 39888
    df = DataFrame(np.arange(100).reshape(10, 10))
    result = df.style.background_gradient(cmap=cmap, axis=None)._compute().ctx
    for k in expected.keys():
        assert result[k] == expected[k]


def test_background_gradient_vmin_vmax():
    # [FUNCTION DOC]
    # Regression test (GH 12145): with explicit minimum and maximum limits, values
    # beyond the limits get the same colour as the limit values (first and second, and
    # last two, rows equal).
    # GH 12145
    df = DataFrame(range(5))
    ctx = df.style.background_gradient(vmin=1, vmax=3)._compute().ctx
    assert ctx[(0, 0)] == ctx[(1, 0)]
    assert ctx[(4, 0)] == ctx[(3, 0)]


def test_background_gradient_int64():
    # [FUNCTION DOC]
    # Regression test (GH 28869): gradients computed on nullable Int64 data equal
    # those of plain integer data for the first three rows.
    # GH 28869
    df1 = Series(range(3)).to_frame()
    df2 = Series(range(3), dtype="Int64").to_frame()
    ctx1 = df1.style.background_gradient()._compute().ctx
    ctx2 = df2.style.background_gradient()._compute().ctx
    assert ctx2[(0, 0)] == ctx1[(0, 0)]
    assert ctx2[(1, 0)] == ctx1[(1, 0)]
    assert ctx2[(2, 0)] == ctx1[(2, 0)]


@pytest.mark.parametrize(
    "axis, gmap, expected",
    [
        (
            0,
            [1, 2],
            {
                (0, 0): [("background-color", "#fff7fb"), ("color", "#000000")],
                (1, 0): [("background-color", "#023858"), ("color", "#f1f1f1")],
                (0, 1): [("background-color", "#fff7fb"), ("color", "#000000")],
                (1, 1): [("background-color", "#023858"), ("color", "#f1f1f1")],
            },
        ),
        (
            1,
            [1, 2],
            {
                (0, 0): [("background-color", "#fff7fb"), ("color", "#000000")],
                (1, 0): [("background-color", "#fff7fb"), ("color", "#000000")],
                (0, 1): [("background-color", "#023858"), ("color", "#f1f1f1")],
                (1, 1): [("background-color", "#023858"), ("color", "#f1f1f1")],
            },
        ),
        (
            None,
            np.array([[2, 1], [1, 2]]),
            {
                (0, 0): [("background-color", "#023858"), ("color", "#f1f1f1")],
                (1, 0): [("background-color", "#fff7fb"), ("color", "#000000")],
                (0, 1): [("background-color", "#fff7fb"), ("color", "#000000")],
                (1, 1): [("background-color", "#023858"), ("color", "#f1f1f1")],
            },
        ),
    ],
)
def test_background_gradient_gmap_array(styler_blank, axis, gmap, expected):
    # [FUNCTION DOC]
    # Verifies a gradient map given as a sequence or array: with axis zero or one a
    # sequence aligned to rows or columns, with no axis a two-dimensional array. The
    # data are zeros, so the colours come from the map; compares the full computed
    # context to expected colours (low colour with black text, high colour with
    # near-white text).
    # tests when gmap is given as a sequence and converted to ndarray
    result = styler_blank.background_gradient(axis=axis, gmap=gmap)._compute().ctx
    assert result == expected


@pytest.mark.parametrize(
    "gmap, axis", [([1, 2, 3], 0), ([1, 2], 1), (np.array([[1, 2], [1, 2]]), None)]
)
def test_background_gradient_gmap_array_raises(gmap, axis):
    # [FUNCTION DOC]
    # Verifies that a gradient map of the wrong shape for the axis (too long for
    # rows, too short for columns, wrong two-dimensional shape) raises a value error
    # saying the gradient map is not the correct shape, when computing styles.
    # test when gmap as converted ndarray is bad shape
    df = DataFrame([[0, 0, 0], [0, 0, 0]])
    msg = "supplied 'gmap' is not correct shape"
    with pytest.raises(ValueError, match=msg):
        df.style.background_gradient(gmap=gmap, axis=axis)._compute()


@pytest.mark.parametrize(
    "gmap",
    [
        DataFrame(  # reverse the columns
            [[2, 1], [1, 2]], columns=["B", "A"], index=["X", "Y"]
        ),
        DataFrame(  # reverse the index
            [[2, 1], [1, 2]], columns=["A", "B"], index=["Y", "X"]
        ),
        DataFrame(  # reverse the index and columns
            [[1, 2], [2, 1]], columns=["B", "A"], index=["Y", "X"]
        ),
        DataFrame(  # add unnecessary columns
            [[1, 2, 3], [2, 1, 3]], columns=["A", "B", "C"], index=["X", "Y"]
        ),
        DataFrame(  # add unnecessary index
            [[1, 2], [2, 1], [3, 3]], columns=["A", "B"], index=["X", "Y", "Z"]
        ),
    ],
)
@pytest.mark.parametrize(
    "subset, exp_gmap",  # exp_gmap is underlying map DataFrame should conform to
    [
        (None, [[1, 2], [2, 1]]),
        (["A"], [[1], [2]]),  # slice only column "A" in data and gmap
        (["B", "A"], [[2, 1], [1, 2]]),  # reverse the columns in data
        (IndexSlice["X", :], [[1, 2]]),  # slice only index "X" in data and gmap
        (IndexSlice[["Y", "X"], :], [[2, 1], [1, 2]]),  # reverse the index in data
    ],
)
def test_background_gradient_gmap_dataframe_align(styler_blank, gmap, subset, exp_gmap):
    # [FUNCTION DOC]
    # Verifies that a gradient map given as a frame is aligned to the data labels
    # (reordered columns, reordered index, both, with extra columns, with extra rows),
    # also within a subset (all, a column list, a reversed column list, a row slice, a
    # reversed row list). The result must equal that of an equivalent plain
    # two-dimensional map (parameter exp_gmap) with the same subset.
    # test gmap given as DataFrame that it aligns to the data including subset
    expected = styler_blank.background_gradient(axis=None, gmap=exp_gmap, subset=subset)
    result = styler_blank.background_gradient(axis=None, gmap=gmap, subset=subset)
    assert expected._compute().ctx == result._compute().ctx


@pytest.mark.parametrize(
    "gmap, axis, exp_gmap",
    [
        (Series([2, 1], index=["Y", "X"]), 0, [[1, 1], [2, 2]]),  # reverse the index
        (Series([2, 1], index=["B", "A"]), 1, [[1, 2], [1, 2]]),  # reverse the cols
        (Series([1, 2, 3], index=["X", "Y", "Z"]), 0, [[1, 1], [2, 2]]),  # add idx
        (Series([1, 2, 3], index=["A", "B", "C"]), 1, [[1, 2], [1, 2]]),  # add col
    ],
)
def test_background_gradient_gmap_series_align(styler_blank, gmap, axis, exp_gmap):
    # [FUNCTION DOC]
    # Verifies that a gradient map given as a series is aligned to row or column
    # labels (reversed order, or with extra labels) and gives the same colours as the
    # equivalent two-dimensional map applied to the whole frame.
    # test gmap given as Series that it aligns to the data including subset
    expected = styler_blank.background_gradient(axis=None, gmap=exp_gmap)._compute()
    result = styler_blank.background_gradient(axis=axis, gmap=gmap)._compute()
    assert expected.ctx == result.ctx


@pytest.mark.parametrize("axis", [1, 0])
def test_background_gradient_gmap_wrong_dataframe(styler_blank, axis):
    # [FUNCTION DOC]
    # Verifies that a frame given as gradient map when the axis makes operations act
    # on a series (axis zero or one) raises a value error saying the map is a frame
    # but the underlying data is a series.
    # test giving a gmap in DataFrame but with wrong axis
    gmap = DataFrame([[1, 2], [2, 1]], columns=["A", "B"], index=["X", "Y"])
    msg = "'gmap' is a DataFrame but underlying data for operations is a Series"
    with pytest.raises(ValueError, match=msg):
        styler_blank.background_gradient(gmap=gmap, axis=axis)._compute()


def test_background_gradient_gmap_wrong_series(styler_blank):
    # [FUNCTION DOC]
    # Verifies that a series given as gradient map when the axis is none (operations
    # act on a frame) raises a value error saying the map is a series but the
    # underlying data is a frame.
    # test giving a gmap in Series form but with wrong axis
    msg = "'gmap' is a Series but underlying data for operations is a DataFrame"
    gmap = Series([1, 2], index=["X", "Y"])
    with pytest.raises(ValueError, match=msg):
        styler_blank.background_gradient(gmap=gmap, axis=None)._compute()


def test_background_gradient_nullable_dtypes():
    # [FUNCTION DOC]
    # Regression test (GH 50712): gradient styles for a float column with nan and a
    # nullable Int64 column with a missing value are equal.
    # GH 50712
    df1 = DataFrame([[1], [0], [np.nan]], dtype=float)
    df2 = DataFrame([[1], [0], [None]], dtype="Int64")

    ctx1 = df1.style.background_gradient()._compute().ctx
    ctx2 = df2.style.background_gradient()._compute().ctx
    assert ctx1 == ctx2


@pytest.mark.parametrize(
    "cmap",
    ["PuBu", mpl.colormaps["PuBu"]],
)
def test_bar_colormap(cmap):
    # [FUNCTION DOC]
    # Verifies bars coloured from a colormap given by name or as a colormap object:
    # the gradient string of each cell contains the expected colour (PuBu map) for a
    # two-by-two frame scaled over the whole frame.
    data = DataFrame([[1, 2], [3, 4]])
    ctx = data.style.bar(cmap=cmap, axis=None)._compute().ctx
    pubu_colors = {
        (0, 0): "#d0d1e6",
        (1, 0): "#056faf",
        (0, 1): "#73a9cf",
        (1, 1): "#023858",
    }
    for k, v in pubu_colors.items():
        assert v in ctx[k][1][1]


def test_bar_color_raises(df):
    # [FUNCTION DOC]
    # Verifies bar colour validation: a set or a three-item list as colour raises a
    # value error about the allowed colour types, and giving both a colour and a
    # colormap raises a value error saying they cannot both be given. Errors surface
    # when rendering html.
    msg = "`color` must be string or list or tuple of 2 strings"
    with pytest.raises(ValueError, match=msg):
        df.style.bar(color={"a", "b"}).to_html()
    with pytest.raises(ValueError, match=msg):
        df.style.bar(color=["a", "b", "c"]).to_html()

    msg = "`color` and `cmap` cannot both be given"
    with pytest.raises(ValueError, match=msg):
        df.style.bar(color="something", cmap="something else").to_html()


@pytest.mark.parametrize("plot_method", ["scatter", "hexbin"])
def test_pass_colormap_instance(df, plot_method):
    # [FUNCTION DOC]
    # Regression test (https://github.com/pandas-dev/pandas/issues/49374) for the
    # plotting accessor rather than the styler: a colormap object built from two
    # colours is accepted as the colormap argument of scatter and hexbin plots of the
    # fixture frame with a computed extra column (named c for scatter, passed as the
    # reduction column for hexbin). Mutates the fixture frame. Passes if no exception
    # is raised.
    # https://github.com/pandas-dev/pandas/issues/49374
    cmap = mpl.colors.ListedColormap([[1, 1, 1], [0, 0, 0]])
    df["c"] = df.A + df.B
    kwargs = {"x": "A", "y": "B", "c": "c", "colormap": cmap}
    if plot_method == "hexbin":
        kwargs["C"] = kwargs.pop("c")
    getattr(df.plot, plot_method)(**kwargs)
