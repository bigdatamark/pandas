# [MODULE DOC]
# Tests for the core of the table styler: sparsified multi-level headers, row and
# column trimming, copy / clear / export-and-use, hiding of index and column
# levels, labels and names, header style mappings, element-wise and axis-wise
# style application and their shape validation, subsets and slices (including the
# private non-reducing slice helper), table styles and attributes (from string
# or tuples), unique ids, cell classes, captions, pipe, and writing outputs to a
# buffer. Also tests the private render helpers imported from the style rendering
# module: level length calculation, trimming maximums, css string to tuple
# conversion and the slice helper. Defines fixtures: multi-level frames and
# stylers (one with every feature enabled), a blank-cell text value, and a simple
# frame and styler. Contains one test class and many module-level tests. The
# jinja2 skip call runs at import time before the styler imports.
import contextlib
import copy
import re
from textwrap import dedent

import numpy as np
import pytest

from pandas import (
    DataFrame,
    IndexSlice,
    MultiIndex,
    Series,
    option_context,
)
import pandas._testing as tm

jinja2 = pytest.importorskip("jinja2")
from pandas.io.formats.style import (  # isort:skip
    Styler,
)
from pandas.io.formats.style_render import (
    _get_level_lengths,
    _get_trimming_maximums,
    maybe_convert_css_to_tuples,
    non_reducing_slice,
)


@pytest.fixture
def mi_df():
    # [FUNCTION DOC]
    # HELPER: fixture returning a two-by-two integer frame with a two-level row index
    # (i0 with i1_a, i1_b) and a two-level column index (c0 with c1_a, c1_b).
    return DataFrame(
        [[1, 2], [3, 4]],
        index=MultiIndex.from_product([["i0"], ["i1_a", "i1_b"]]),
        columns=MultiIndex.from_product([["c0"], ["c1_a", "c1_b"]]),
        dtype=int,
    )


@pytest.fixture
def mi_styler(mi_df):
    # [FUNCTION DOC]
    # HELPER: fixture returning a styler over mi_df with no unique id suffix.
    return Styler(mi_df, uuid_len=0)


@pytest.fixture
def mi_styler_comp(mi_styler):
    # [FUNCTION DOC]
    # HELPER: fixture returning a deep copy of the mi_styler fixture with as many
    # features as possible switched on, so that copy, clear and export tests cover
    # everything: custom css class names, a fixed unique id, a caption, a table style
    # and attributes, hidden columns and rows (with names), a concatenated
    # aggregated styler, value formatting with missing text and precision, label
    # formatting with precisions on each axis, a maximum highlight, label style
    # mappings on both axes, cell classes, tooltips, and html-escaped index name
    # formatting.
    # comprehensively add features to mi_styler
    mi_styler = mi_styler._copy(deepcopy=True)
    mi_styler.css = {**mi_styler.css, "row": "ROW", "col": "COL"}
    mi_styler.uuid_len = 5
    mi_styler.uuid = "abcde"
    mi_styler.set_caption("capt")
    mi_styler.set_table_styles([{"selector": "a", "props": "a:v;"}])
    mi_styler.hide(axis="columns")
    mi_styler.hide([("c0", "c1_a")], axis="columns", names=True)
    mi_styler.hide(axis="index")
    mi_styler.hide([("i0", "i1_a")], axis="index", names=True)
    mi_styler.set_table_attributes('class="box"')
    other = mi_styler.data.agg(["mean"])
    other.index = MultiIndex.from_product([[""], other.index])
    mi_styler.concat(other.style)
    mi_styler.format(na_rep="MISSING", precision=3)
    mi_styler.format_index(precision=2, axis=0)
    mi_styler.format_index(precision=4, axis=1)
    mi_styler.highlight_max(axis=None)
    mi_styler.map_index(lambda x: "color: white;", axis=0)
    mi_styler.map_index(lambda x: "color: black;", axis=1)
    mi_styler.set_td_classes(
        DataFrame(
            [["a", "b"], ["a", "c"]], index=mi_styler.index, columns=mi_styler.columns
        )
    )
    mi_styler.set_tooltips(
        DataFrame(
            [["a2", "b2"], ["a2", "c2"]],
            index=mi_styler.index,
            columns=mi_styler.columns,
        )
    )
    mi_styler.format_index_names(escape="html", axis=0)
    mi_styler.format_index_names(escape="html", axis=1)
    return mi_styler


@pytest.fixture
def blank_value():
    # [FUNCTION DOC]
    # HELPER: fixture returning the html text for a blank cell (a non-breaking space
    # entity).
    return "&nbsp;"


@pytest.fixture
def df():
    # [FUNCTION DOC]
    # HELPER: fixture returning a two-row frame with an integer column A and a random
    # normal column B (seeded generator).
    df = DataFrame({"A": [0, 1], "B": np.random.default_rng(2).standard_normal(2)})
    return df


@pytest.fixture
def styler(df):
    # [FUNCTION DOC]
    # HELPER: fixture returning a styler (with default random unique id) over a newly
    # built frame like that of the df fixture.
    # KNOWN ISSUE: the fixture takes the df fixture as an argument but ignores it and
    # builds a second, separate random frame, so the styler data does not match the df
    # fixture values.
    df = DataFrame({"A": [0, 1], "B": np.random.default_rng(2).standard_normal(2)})
    return Styler(df)


@pytest.mark.parametrize(
    "sparse_columns, exp_cols",
    [
        (
            True,
            [
                {"is_visible": True, "attributes": 'colspan="2"', "value": "c0"},
                {"is_visible": False, "attributes": "", "value": "c0"},
            ],
        ),
        (
            False,
            [
                {"is_visible": True, "attributes": "", "value": "c0"},
                {"is_visible": True, "attributes": "", "value": "c0"},
            ],
        ),
    ],
)
def test_mi_styler_sparsify_columns(mi_styler, sparse_columns, exp_cols):
    # [FUNCTION DOC]
    # Verifies translated column header cells of a two-level column index: with
    # column sparsification the first top label spans two columns (visible, with a
    # span attribute) and the second is hidden, without it both are visible and plain.
    # The second-level labels are shown in both cases. Parametrized over the flag and
    # expected cell properties.
    exp_l1_c0 = {"is_visible": True, "attributes": "", "display_value": "c1_a"}
    exp_l1_c1 = {"is_visible": True, "attributes": "", "display_value": "c1_b"}

    ctx = mi_styler._translate(True, sparse_columns)

    assert exp_cols[0].items() <= ctx["head"][0][2].items()
    assert exp_cols[1].items() <= ctx["head"][0][3].items()
    assert exp_l1_c0.items() <= ctx["head"][1][2].items()
    assert exp_l1_c1.items() <= ctx["head"][1][3].items()


@pytest.mark.parametrize(
    "sparse_index, exp_rows",
    [
        (
            True,
            [
                {"is_visible": True, "attributes": 'rowspan="2"', "value": "i0"},
                {"is_visible": False, "attributes": "", "value": "i0"},
            ],
        ),
        (
            False,
            [
                {"is_visible": True, "attributes": "", "value": "i0"},
                {"is_visible": True, "attributes": "", "value": "i0"},
            ],
        ),
    ],
)
def test_mi_styler_sparsify_index(mi_styler, sparse_index, exp_rows):
    # [FUNCTION DOC]
    # Same as the column test for the row index: sparsified outer row label spans two
    # rows and the continuation cell is not visible; second-level labels are shown.
    # Parametrized over the flag and expected cell properties.
    exp_l1_r0 = {"is_visible": True, "attributes": "", "display_value": "i1_a"}
    exp_l1_r1 = {"is_visible": True, "attributes": "", "display_value": "i1_b"}

    ctx = mi_styler._translate(sparse_index, True)

    assert exp_rows[0].items() <= ctx["body"][0][0].items()
    assert exp_rows[1].items() <= ctx["body"][1][0].items()
    assert exp_l1_r0.items() <= ctx["body"][0][1].items()
    assert exp_l1_r1.items() <= ctx["body"][1][1].items()


def test_mi_styler_sparsify_options(mi_styler):
    # [FUNCTION DOC]
    # Verifies that the global sparsify options for index and for columns each change
    # the html output when switched between false and true.
    with option_context("styler.sparse.index", False):
        html1 = mi_styler.to_html()
    with option_context("styler.sparse.index", True):
        html2 = mi_styler.to_html()

    assert html1 != html2

    with option_context("styler.sparse.columns", False):
        html1 = mi_styler.to_html()
    with option_context("styler.sparse.columns", True):
        html2 = mi_styler.to_html()

    assert html1 != html2


@pytest.mark.parametrize(
    "rn, cn, max_els, max_rows, max_cols, exp_rn, exp_cn",
    [
        (100, 100, 100, None, None, 12, 6),  # reduce to (12, 6) < 100 elements
        (1000, 3, 750, None, None, 250, 3),  # dynamically reduce rows to 250, keep cols
        (4, 1000, 500, None, None, 4, 125),  # dynamically reduce cols to 125, keep rows
        (1000, 3, 750, 10, None, 10, 3),  # overwrite above dynamics with max_row
        (4, 1000, 500, None, 5, 4, 5),  # overwrite above dynamics with max_col
        (100, 100, 700, 50, 50, 25, 25),  # rows cols below given maxes so < 700 elmts
    ],
)
def test_trimming_maximum(rn, cn, max_els, max_rows, max_cols, exp_rn, exp_cn):
    # [FUNCTION DOC]
    # Verifies the private helper that computes the maximum rendered rows and columns
    # from the number of rows and columns, the maximum elements, optional explicit
    # maximums and a scaling factor of one half: automatic reduction keeps the total
    # below the element limit (reducing the larger dimension dynamically), and
    # explicit maxima override the automatic values. Parametrized over six cases with
    # expected row and column counts.
    rn, cn = _get_trimming_maximums(
        rn, cn, max_els, max_rows, max_cols, scaling_factor=0.5
    )
    assert (rn, cn) == (exp_rn, exp_cn)


@pytest.mark.parametrize(
    "option, val",
    [
        ("styler.render.max_elements", 6),
        ("styler.render.max_rows", 3),
    ],
)
def test_render_trimming_rows(option, val):
    # [FUNCTION DOC]
    # Verifies row trimming, both automatic (maximum elements of six) and explicit
    # (maximum rows of three), on a sixty-row, two-column frame: three data rows plus
    # a trimming row, and the header and body rows have the index plus two data cells.
    # test auto and specific trimming of rows
    df = DataFrame(np.arange(120).reshape(60, 2))
    with option_context(option, val):
        ctx = df.style._translate(True, True)
    assert len(ctx["head"][0]) == 3  # index + 2 data cols
    assert len(ctx["body"]) == 4  # 3 data rows + trimming row
    assert len(ctx["body"][0]) == 3  # index + 2 data cols


@pytest.mark.parametrize(
    "option, val",
    [
        ("styler.render.max_elements", 6),
        ("styler.render.max_columns", 2),
    ],
)
def test_render_trimming_cols(option, val):
    # [FUNCTION DOC]
    # Verifies column trimming, both automatic (maximum elements of six) and explicit
    # (maximum columns of two), on a three-row, ten-column frame: two data columns
    # plus a trimming column, in both the header and body rows.
    # test auto and specific trimming of cols
    df = DataFrame(np.arange(30).reshape(3, 10))
    with option_context(option, val):
        ctx = df.style._translate(True, True)
    assert len(ctx["head"][0]) == 4  # index + 2 data cols + trimming col
    assert len(ctx["body"]) == 3  # 3 data rows
    assert len(ctx["body"][0]) == 4  # index + 2 data cols + trimming col


def test_render_trimming_mi():
    # [FUNCTION DOC]
    # Verifies trimming with multi-level indexes and a small element limit: body width
    # includes both index levels, two data columns and a trimming column, the outer
    # index label spans two rows, the trimming column cells have the trimming classes,
    # and the body has two data rows plus the trimming row.
    midx = MultiIndex.from_product([[1, 2], [1, 2, 3]])
    df = DataFrame(np.arange(36).reshape(6, 6), columns=midx, index=midx)
    with option_context("styler.render.max_elements", 4):
        ctx = df.style._translate(True, True)

    assert len(ctx["body"][0]) == 5  # 2 indexes + 2 data cols + trimming row
    assert {"attributes": 'rowspan="2"'}.items() <= ctx["body"][0][0].items()
    assert {"class": "data row0 col_trim"}.items() <= ctx["body"][0][4].items()
    assert {"class": "data row_trim col_trim"}.items() <= ctx["body"][2][4].items()
    assert len(ctx["body"]) == 3  # 2 data rows + trimming row


def test_render_empty_mi():
    # [FUNCTION DOC]
    # Regression test (GH 43305): a frame with no columns but a two-level named index
    # (one unnamed level) renders a header row with only the index name cells, the
    # unnamed one shown as a blank.
    # GH 43305
    df = DataFrame(index=MultiIndex.from_product([["A"], [0, 1]], names=[None, "one"]))
    expected = dedent(
        """\
    >
      <thead>
        <tr>
          <th class="index_name level0" >&nbsp;</th>
          <th class="index_name level1" >one</th>
        </tr>
      </thead>
    """
    )
    assert expected in df.style.to_html()


@pytest.mark.parametrize("comprehensive", [True, False])
@pytest.mark.parametrize("render", [True, False])
@pytest.mark.parametrize("deepcopy", [True, False])
def test_copy(comprehensive, render, deepcopy, mi_styler, mi_styler_comp):
    # [FUNCTION DOC]
    # Verifies copying of a styler (shallow or deep), either the simple or the
    # fully-featured fixture, before or after rendering. A copy is a different object.
    # For a shallow copy all attributes (other than render-time caches and class level
    # templates) are the very same objects; for a deep copy the listed shallow
    # attributes (data, labels, id, caption, flags, attributes) remain the same
    # objects and every other attribute is a different object unless it is none.
    # Parametrized over comprehensive, render and deepcopy flags.
    # KNOWN ISSUE: the attribute filters apply the callable check to the attribute
    # name strings instead of the attribute values, so that part of the filter never
    # excludes anything.
    styler = mi_styler_comp if comprehensive else mi_styler
    styler.uuid_len = 5

    s2 = copy.deepcopy(styler) if deepcopy else copy.copy(styler)  # make copy and check
    assert s2 is not styler

    if render:
        styler.to_html()

    excl = [
        "cellstyle_map",  # render time vars..
        "cellstyle_map_columns",
        "cellstyle_map_index",
        "template_latex",  # render templates are class level
        "template_html",
        "template_html_style",
        "template_html_table",
    ]
    if not deepcopy:  # check memory locations are equal for all included attributes
        for attr in [a for a in styler.__dict__ if (not callable(a) and a not in excl)]:
            assert id(getattr(s2, attr)) == id(getattr(styler, attr))
    else:  # check memory locations are different for nested or mutable vars
        shallow = [
            "data",
            "columns",
            "index",
            "uuid_len",
            "uuid",
            "caption",
            "cell_ids",
            "hide_index_",
            "hide_columns_",
            "hide_index_names",
            "hide_column_names",
            "table_attributes",
        ]
        for attr in shallow:
            assert id(getattr(s2, attr)) == id(getattr(styler, attr))

        for attr in [
            a
            for a in styler.__dict__
            if (not callable(a) and a not in excl and a not in shallow)
        ]:
            if getattr(s2, attr) is None:
                assert id(getattr(s2, attr)) == id(getattr(styler, attr))
            else:
                assert id(getattr(s2, attr)) != id(getattr(styler, attr))


@pytest.mark.parametrize("deepcopy", [True, False])
def test_inherited_copy(mi_styler, deepcopy):
    # [FUNCTION DOC]
    # Regression test (GH 52728): shallow and deep copies of an instance of a styler
    # subclass are still instances of that subclass. Parametrized over the copy kind.
    # Ensure that the inherited class is preserved when a Styler object is copied.
    # GH 52728
    class CustomStyler(Styler):
        # [CLASS DOC]
        # HELPER: trivial local subclass of the styler used only to check that copying
        # preserves the subclass.
        pass

    custom_styler = CustomStyler(mi_styler.data)
    custom_styler_copy = (
        copy.deepcopy(custom_styler) if deepcopy else copy.copy(custom_styler)
    )
    assert isinstance(custom_styler_copy, CustomStyler)


def test_clear(mi_styler_comp):
    # [FUNCTION DOC]
    # Regression test (GH 40675): with the fully-featured fixture and styles
    # computed, a freshly built plain styler with the same data and id differs in
    # every attribute (excluding data, labels, id and render-time or class level
    # attributes); after the clear method all attributes equal those of the plain
    # styler. If this fails for new features the fixture should be extended.
    # NOTE: if this test fails for new features then 'mi_styler_comp' should be updated
    # to ensure proper testing of the 'copy', 'clear', 'export' methods with new feature
    # GH 40675
    styler = mi_styler_comp
    styler._compute()  # execute applied methods

    clean_copy = Styler(styler.data, uuid=styler.uuid)

    excl = [
        "data",
        "index",
        "columns",
        "uuid",
        "uuid_len",  # uuid is set to be the same on styler and clean_copy
        "cell_ids",
        "cellstyle_map",  # execution time only
        "cellstyle_map_columns",  # execution time only
        "cellstyle_map_index",  # execution time only
        "template_latex",  # render templates are class level
        "template_html",
        "template_html_style",
        "template_html_table",
    ]
    # tests vars are not same vals on obj and clean copy before clear (except for excl)
    for attr in [a for a in styler.__dict__ if not (callable(a) or a in excl)]:
        res = getattr(styler, attr) == getattr(clean_copy, attr)
        if hasattr(res, "__iter__") and len(res) > 0:
            assert not all(res)  # some element in iterable differs
        elif hasattr(res, "__iter__") and len(res) == 0:
            pass  # empty array
        else:
            assert not res  # explicit var differs

    # test vars have same vales on obj and clean copy after clearing
    styler.clear()
    for attr in [a for a in styler.__dict__ if not callable(a)]:
        res = getattr(styler, attr) == getattr(clean_copy, attr)
        assert all(res) if hasattr(res, "__iter__") else res


def test_export(mi_styler_comp, mi_styler):
    # [FUNCTION DOC]
    # Verifies the export and use pair: the exportable attributes (pending style
    # functions, hide flags, table attributes, table styles, css names) differ between
    # the plain and fully-featured fixtures; after exporting the latter and using the
    # export on the former, all those attributes are equal; the result still renders.
    exp_attrs = [
        "_todo",
        "hide_index_",
        "hide_index_names",
        "hide_columns_",
        "hide_column_names",
        "table_attributes",
        "table_styles",
        "css",
    ]
    for attr in exp_attrs:
        check = getattr(mi_styler, attr) == getattr(mi_styler_comp, attr)
        assert not (
            all(check) if (hasattr(check, "__iter__") and len(check) > 0) else check
        )

    export = mi_styler_comp.export()
    used = mi_styler.use(export)
    for attr in exp_attrs:
        check = getattr(used, attr) == getattr(mi_styler_comp, attr)
        assert all(check) if (hasattr(check, "__iter__") and len(check) > 0) else check

    used.to_html()


def test_hide_raises(mi_styler):
    # [FUNCTION DOC]
    # Verifies hide argument validation: passing both subset and level raises a value
    # error saying they cannot be passed simultaneously, and a level of an
    # unsupported type (a dict) raises a value error about the allowed types.
    msg = "`subset` and `level` cannot be passed simultaneously"
    with pytest.raises(ValueError, match=msg):
        mi_styler.hide(axis="index", subset="something", level="something else")

    msg = "`level` must be of type `int`, `str` or list of such"
    with pytest.raises(ValueError, match=msg):
        mi_styler.hide(axis="index", level={"bad": 1, "type": 2})


@pytest.mark.parametrize("level", [1, "one", [1], ["one"]])
def test_hide_index_level(mi_styler, level):
    # [FUNCTION DOC]
    # Verifies hiding one level of a two-level row index (by number or name, alone or
    # in a list), with names set on both axes: the header rows have the expected
    # lengths, the hidden level cells in the names row and in the body are not
    # visible while the other level stays visible. Parametrized over level selectors.
    mi_styler.index.names, mi_styler.columns.names = ["zero", "one"], ["zero", "one"]
    ctx = mi_styler.hide(axis="index", level=level)._translate(False, True)
    assert len(ctx["head"][0]) == 3
    assert len(ctx["head"][1]) == 3
    assert len(ctx["head"][2]) == 4
    assert ctx["head"][2][0]["is_visible"]
    assert not ctx["head"][2][1]["is_visible"]

    assert ctx["body"][0][0]["is_visible"]
    assert not ctx["body"][0][1]["is_visible"]
    assert ctx["body"][1][0]["is_visible"]
    assert not ctx["body"][1][1]["is_visible"]


@pytest.mark.parametrize("level", [1, "one", [1], ["one"]])
@pytest.mark.parametrize("names", [True, False])
def test_hide_columns_level(mi_styler, level, names):
    # [FUNCTION DOC]
    # Verifies hiding one column level (by number or name, alone or in a list) leaves
    # one header row, plus the index names row when the index has names. Parametrized
    # over level selector and whether index names are set.
    mi_styler.columns.names = ["zero", "one"]
    if names:
        mi_styler.index.names = ["zero", "one"]
    ctx = mi_styler.hide(axis="columns", level=level)._translate(True, False)
    assert len(ctx["head"]) == (2 if names else 1)


@pytest.mark.parametrize("method", ["map", "apply"])
@pytest.mark.parametrize("axis", ["index", "columns"])
def test_apply_map_header(method, axis):
    # [FUNCTION DOC]
    # Regression test (GH 41893): the label apply and map methods register a pending
    # function (counted in the to-do list) without populating the context for the
    # axis; after computing, the context for that axis holds the style for the label
    # that matched the function. Parametrized over method (map or apply) and axis.
    # GH 41893
    df = DataFrame({"A": [0, 0], "B": [1, 1]}, index=["C", "D"])
    func = {
        "apply": lambda s: ["attr: val" if ("A" in v or "C" in v) else "" for v in s],
        "map": lambda v: "attr: val" if ("A" in v or "C" in v) else "",
    }

    # test execution added to todo
    result = getattr(df.style, f"{method}_index")(func[method], axis=axis)
    assert len(result._todo) == 1
    assert len(getattr(result, f"ctx_{axis}")) == 0

    # test ctx object on compute
    result._compute()
    expected = {
        (0, 0): [("attr", "val")],
    }
    assert getattr(result, f"ctx_{axis}") == expected


@pytest.mark.parametrize("method", ["apply", "map"])
@pytest.mark.parametrize("axis", ["index", "columns"])
def test_apply_map_header_mi(mi_styler, method, axis):
    # [FUNCTION DOC]
    # Regression test (GH 41893): the label apply and map methods on a two-level index
    # produce the style for the single label matching the function, keyed by level and
    # position. Parametrized over method and axis.
    # GH 41893
    func = {
        "apply": lambda s: ["attr: val;" if "b" in v else "" for v in s],
        "map": lambda v: "attr: val" if "b" in v else "",
    }
    result = getattr(mi_styler, f"{method}_index")(func[method], axis=axis)._compute()
    expected = {(1, 1): [("attr", "val")]}
    assert getattr(result, f"ctx_{axis}") == expected


def test_apply_map_header_raises(mi_styler):
    # [FUNCTION DOC]
    # Regression test (GH 41893): an invalid axis name given to the label map method
    # raises a value error saying no such axis exists for the frame type when the
    # styles are computed.
    # GH 41893
    with pytest.raises(ValueError, match="No axis named bad for object type DataFrame"):
        mi_styler.map_index(lambda v: "attr: val;", axis="bad")._compute()


class TestStyler:
    # [CLASS DOC]
    # Groups tests of the styler's constructor, rendering, styling application and
    # subset selection behaviour, using the df and styler fixtures. No state.
    def test_init_non_pandas(self):
        # [FUNCTION DOC]
        # Verifies that constructing a styler from a list raises a type error saying the
        # data must be a series or frame.
        msg = "``data`` must be a Series or DataFrame"
        with pytest.raises(TypeError, match=msg):
            Styler([1, 2, 3])

    def test_init_series(self):
        # [FUNCTION DOC]
        # Verifies that constructing a styler from a series stores the data as a
        # two-dimensional frame.
        result = Styler(Series([1, 2]))
        assert result.data.ndim == 2

    def test_repr_html_ok(self, styler):
        # [FUNCTION DOC]
        # Smoke test: the notebook html representation of the styler can be produced.
        styler._repr_html_()

    def test_repr_html_mathjax(self, styler):
        # [FUNCTION DOC]
        # Regression test (gh-19824 and 41395): by default the html output has no maths
        # rendering ignore classes; with the mathjax option turned off both ignore
        # classes appear.
        # gh-19824 / 41395
        assert "tex2jax_ignore" not in styler._repr_html_()
        assert "mathjax_ignore" not in styler._repr_html_()

        with option_context("styler.html.mathjax", False):
            assert "tex2jax_ignore" in styler._repr_html_()
            assert "mathjax_ignore" in styler._repr_html_()

    def test_update_ctx(self, styler):
        # [FUNCTION DOC]
        # Verifies the private context update method parses a frame of css strings into
        # the context keyed by (row, column) with (property, value) tuples.
        styler._update_ctx(DataFrame({"A": ["color: red", "color: blue"]}))
        expected = {(0, 0): [("color", "red")], (1, 0): [("color", "blue")]}
        assert styler.ctx == expected

    def test_update_ctx_flatten_multi_and_trailing_semi(self, styler):
        # [FUNCTION DOC]
        # Verifies that the private context update method splits several declarations in
        # one string, trims whitespace around names and values, and tolerates a trailing
        # semicolon.
        attrs = DataFrame({"A": ["color: red; foo: bar", "color:blue ; foo: baz;"]})
        styler._update_ctx(attrs)
        expected = {
            (0, 0): [("color", "red"), ("foo", "bar")],
            (1, 0): [("color", "blue"), ("foo", "baz")],
        }
        assert styler.ctx == expected

    def test_render(self):
        # [FUNCTION DOC]
        # Smoke test: applying a column-wise style function returning a series of css
        # strings and rendering to html does not raise.
        df = DataFrame({"A": [0, 1]})
        style = lambda x: Series(["color: red", "color: blue"], name=x.name)
        s = Styler(df, uuid="AB").apply(style)
        s.to_html()
        # it worked?

    def test_multiple_render(self, df):
        # [FUNCTION DOC]
        # Regression test (GH 39396): rendering twice does not duplicate css rules; the
        # output contains exactly the style element for the two cells of column A.
        # GH 39396
        s = Styler(df, uuid_len=0).map(lambda x: "color: red;", subset=["A"])
        s.to_html()  # do 2 renders to ensure css styles not duplicated
        assert (
            '<style type="text/css">\n#T__row0_col0, #T__row1_col0 {\n'
            "  color: red;\n}\n</style>" in s.to_html()
        )

    def test_render_empty_dfs(self):
        # [FUNCTION DOC]
        # Smoke test: styling and rendering an empty frame, a frame with a column but no
        # rows, and one with an index but no columns do not raise.
        empty_df = DataFrame()
        es = Styler(empty_df)
        es.to_html()
        # An index but no columns
        DataFrame(columns=["a"]).style.to_html()
        # A column but no index
        DataFrame(index=["a"]).style.to_html()
        # No IndexError raised?

    def test_render_double(self):
        # [FUNCTION DOC]
        # Smoke test: a style function giving two declarations per cell renders without
        # error.
        df = DataFrame({"A": [0, 1]})
        style = lambda x: Series(
            ["color: red; border: 1px", "color: blue; border: 2px"], name=x.name
        )
        s = Styler(df, uuid="AB").apply(style)
        s.to_html()
        # it worked?

    def test_set_properties(self):
        # [FUNCTION DOC]
        # Verifies that setting properties applies the given name and value pairs to
        # every cell (compared order-independently) in the context.
        df = DataFrame({"A": [0, 1]})
        result = df.style.set_properties(color="white", size="10px")._compute().ctx
        # order is deterministic
        v = [("color", "white"), ("size", "10px")]
        expected = {(0, 0): v, (1, 0): v}
        assert result.keys() == expected.keys()
        for v1, v2 in zip(result.values(), expected.values()):
            assert sorted(v1) == sorted(v2)

    def test_set_properties_subset(self):
        # [FUNCTION DOC]
        # Verifies that setting properties with a single-cell subset affects only that
        # cell.
        df = DataFrame({"A": [0, 1]})
        result = (
            df.style.set_properties(subset=IndexSlice[0, "A"], color="white")
            ._compute()
            .ctx
        )
        expected = {(0, 0): [("color", "white")]}
        assert result == expected

    def test_empty_index_name_doesnt_display(self, blank_value):
        # [FUNCTION DOC]
        # Verifies that a frame with an unnamed index has a single header row and its
        # corner cell is a blank header cell (class, type, value and display value as the
        # blank text).
        # https://github.com/pandas-dev/pandas/pull/12090#issuecomment-180695902
        df = DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        result = df.style._translate(True, True)
        assert len(result["head"]) == 1
        expected = {
            "class": "blank level0",
            "type": "th",
            "value": blank_value,
            "is_visible": True,
            "display_value": blank_value,
        }
        assert expected.items() <= result["head"][0][0].items()

    def test_index_name(self):
        # [FUNCTION DOC]
        # Verifies that a named index yields a header row cell with the index name class
        # and the name as value and display value.
        # https://github.com/pandas-dev/pandas/issues/11655
        df = DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        result = df.set_index("A").style._translate(True, True)
        expected = {
            "class": "index_name level0",
            "type": "th",
            "value": "A",
            "is_visible": True,
            "display_value": "A",
        }
        assert expected.items() <= result["head"][1][0].items()

    def test_numeric_columns(self):
        # [FUNCTION DOC]
        # Smoke test for the translate step with numeric (non-string) column labels.
        # https://github.com/pandas-dev/pandas/issues/12125
        # smoke test for _translate
        df = DataFrame({0: [1, 2, 3]})
        df.style._translate(True, True)

    def test_apply_axis(self):
        # [FUNCTION DOC]
        # Verifies axis-wise application of a function returning one css string per
        # element (using the maximum of the slice): pending function registered but
        # context empty before computing; after computing, row-wise and column-wise
        # results match the expected context, and the default axis is columns.
        df = DataFrame({"A": [0, 0], "B": [1, 1]})
        f = lambda x: [f"val: {x.max()}" for v in x]
        result = df.style.apply(f, axis=1)
        assert len(result._todo) == 1
        assert len(result.ctx) == 0
        result._compute()
        expected = {
            (0, 0): [("val", "1")],
            (0, 1): [("val", "1")],
            (1, 0): [("val", "1")],
            (1, 1): [("val", "1")],
        }
        assert result.ctx == expected

        result = df.style.apply(f, axis=0)
        expected = {
            (0, 0): [("val", "0")],
            (0, 1): [("val", "1")],
            (1, 0): [("val", "0")],
            (1, 1): [("val", "1")],
        }
        result._compute()
        assert result.ctx == expected
        result = df.style.apply(f)  # default
        result._compute()
        assert result.ctx == expected

    @pytest.mark.parametrize("axis", [0, 1])
    def test_apply_series_return(self, axis):
        # [FUNCTION DOC]
        # Regression test (GH 42014): a function returning a series with only some labels
        # (short) or reordered labels is aligned to the data labels. Parametrized over
        # axis; checks the colour applies to the matching cells.
        # GH 42014
        df = DataFrame([[1, 2], [3, 4]], index=["X", "Y"], columns=["X", "Y"])

        # test Series return where len(Series) < df.index or df.columns but labels OK
        func = lambda s: Series(["color: red;"], index=["Y"])
        result = df.style.apply(func, axis=axis)._compute().ctx
        assert result[(1, 1)] == [("color", "red")]
        assert result[(1 - axis, axis)] == [("color", "red")]

        # test Series return where labels align but different order
        func = lambda s: Series(["color: red;", "color: blue;"], index=["Y", "X"])
        result = df.style.apply(func, axis=axis)._compute().ctx
        assert result[(0, 0)] == [("color", "blue")]
        assert result[(1, 1)] == [("color", "red")]
        assert result[(1 - axis, axis)] == [("color", "red")]
        assert result[(axis, 1 - axis)] == [("color", "blue")]

    @pytest.mark.parametrize("index", [False, True])
    @pytest.mark.parametrize("columns", [False, True])
    def test_apply_dataframe_return(self, index, columns):
        # [FUNCTION DOC]
        # Regression test (GH 42014): applying with no axis a function that returns a
        # frame of styles for a subset of labels styles only the cells whose row and
        # column are present. Parametrized over whether the row and column label sets are
        # complete.
        # GH 42014
        df = DataFrame([[1, 2], [3, 4]], index=["X", "Y"], columns=["X", "Y"])
        idxs = ["X", "Y"] if index else ["Y"]
        cols = ["X", "Y"] if columns else ["Y"]
        df_styles = DataFrame("color: red;", index=idxs, columns=cols)
        result = df.style.apply(lambda x: df_styles, axis=None)._compute().ctx

        assert result[(1, 1)] == [("color", "red")]  # (Y,Y) styles always present
        assert (result[(0, 1)] == [("color", "red")]) is index  # (X,Y) only if index
        assert (result[(1, 0)] == [("color", "red")]) is columns  # (Y,X) only if cols
        assert (result[(0, 0)] == [("color", "red")]) is (index and columns)  # (X,X)

    @pytest.mark.parametrize(
        "slice_",
        [
            IndexSlice[:],
            IndexSlice[:, ["A"]],
            IndexSlice[[1], :],
            IndexSlice[[1], ["A"]],
            IndexSlice[:2, ["A", "B"]],
        ],
    )
    @pytest.mark.parametrize("axis", [0, 1])
    def test_apply_subset(self, slice_, axis, df):
        # [FUNCTION DOC]
        # Verifies row-wise or column-wise application restricted to a subset slice with
        # an extra keyword argument forwarded: the context contains the style exactly for
        # the cells in the sliced frame. Parametrized over five slices and the axis. The
        # inner function builds a series of the css text for the colour keyword (default
        # bar) aligned to its input.
        def h(x, color="bar"):
            # [FUNCTION DOC]
            # HELPER: style function for the subset test. Takes a series x and a colour
            # (default bar) and returns a series with the same index and name whose every
            # element is a css colour declaration.
            return Series(f"color: {color}", index=x.index, name=x.name)

        result = df.style.apply(h, axis=axis, subset=slice_, color="baz")._compute().ctx
        expected = {
            (r, c): [("color", "baz")]
            for r, row in enumerate(df.index)
            for c, col in enumerate(df.columns)
            if row in df.loc[slice_].index and col in df.loc[slice_].columns
        }
        assert result == expected

    @pytest.mark.parametrize(
        "slice_",
        [
            IndexSlice[:],
            IndexSlice[:, ["A"]],
            IndexSlice[[1], :],
            IndexSlice[[1], ["A"]],
            IndexSlice[:2, ["A", "B"]],
        ],
    )
    def test_map_subset(self, slice_, df):
        # [FUNCTION DOC]
        # Verifies element-wise styling restricted to a slice: the context contains the
        # style exactly for the cells in the sliced frame. Parametrized over five slices.
        result = df.style.map(lambda x: "color:baz;", subset=slice_)._compute().ctx
        expected = {
            (r, c): [("color", "baz")]
            for r, row in enumerate(df.index)
            for c, col in enumerate(df.columns)
            if row in df.loc[slice_].index and col in df.loc[slice_].columns
        }
        assert result == expected

    @pytest.mark.parametrize(
        "slice_",
        [
            IndexSlice[:, IndexSlice["x", "A"]],
            IndexSlice[:, IndexSlice[:, "A"]],
            IndexSlice[:, IndexSlice[:, ["A", "C"]]],  # missing col element
            IndexSlice[IndexSlice["a", 1], :],
            IndexSlice[IndexSlice[:, 1], :],
            IndexSlice[IndexSlice[:, [1, 3]], :],  # missing row element
            IndexSlice[:, ("x", "A")],
            IndexSlice[("a", 1), :],
        ],
    )
    def test_map_subset_multiindex(self, slice_):
        # [FUNCTION DOC]
        # Regression test (GH 19861, edited for GH 33562): element-wise styling with
        # slices on multi-level row and column indexes. Slices that name a missing list
        # element (a missing column or row label) must raise a key error naming it; all
        # others must render without error. Parametrized over eight slices.
        # GH 19861
        # edited for GH 33562
        if (
            isinstance(slice_[-1], tuple)
            and isinstance(slice_[-1][-1], list)
            and "C" in slice_[-1][-1]
        ):
            ctx = pytest.raises(KeyError, match="C")
        elif (
            isinstance(slice_[0], tuple)
            and isinstance(slice_[0][1], list)
            and 3 in slice_[0][1]
        ):
            ctx = pytest.raises(KeyError, match="3")
        else:
            ctx = contextlib.nullcontext()

        idx = MultiIndex.from_product([["a", "b"], [1, 2]])
        col = MultiIndex.from_product([["x", "y"], ["A", "B"]])
        df = DataFrame(np.random.default_rng(2).random((4, 4)), columns=col, index=idx)

        with ctx:
            df.style.map(lambda x: "color: red;", subset=slice_).to_html()

    def test_map_subset_multiindex_code(self):
        # [FUNCTION DOC]
        # Regression test (issue 25858): element-wise styling works with a column index
        # built from codes and a label range slice. Smoke test: selects the cells and
        # registers a style function (never computed or rendered), with no assertions.
        # https://github.com/pandas-dev/pandas/issues/25858
        # Checks styler.map works with multindex when codes are provided
        codes = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
        columns = MultiIndex(
            levels=[["a", "b"], ["%", "#"]], codes=codes, names=["", ""]
        )
        df = DataFrame(
            [[1, -1, 1, 1], [-1, 1, 1, 1]], index=["hello", "world"], columns=columns
        )
        pct_subset = IndexSlice[:, IndexSlice[:, "%":"%"]]

        def color_negative_red(val):
            # [FUNCTION DOC]
            # HELPER: style function for the codes test. Returns a css colour declaration:
            # red for negative values, black otherwise.
            color = "red" if val < 0 else "black"
            return f"color: {color}"

        df.loc[pct_subset]
        df.style.map(color_negative_red, subset=pct_subset)

    @pytest.mark.parametrize(
        "stylefunc", ["background_gradient", "bar", "text_gradient"]
    )
    def test_subset_for_boolean_cols(self, stylefunc):
        # [FUNCTION DOC]
        # Regression test (GH47838): gradient, bar and text gradient styles on a frame
        # whose column labels are booleans compute styles for all four cells (no label
        # mix-up with a boolean subset). Parametrized over style method name.
        # GH47838
        df = DataFrame(
            [
                [1, 2],
                [3, 4],
            ],
            columns=[False, True],
        )
        styled = getattr(df.style, stylefunc)()
        styled._compute()
        assert set(styled.ctx) == {(0, 0), (0, 1), (1, 0), (1, 1)}

    def test_empty(self):
        # [FUNCTION DOC]
        # Verifies the translated cell style list for a context with an empty property
        # pair: each cell gets its own entry with its selector and props.
        df = DataFrame({"A": [1, 0]})
        s = df.style
        s.ctx = {(0, 0): [("color", "red")], (1, 0): [("", "")]}

        result = s._translate(True, True)["cellstyle"]
        expected = [
            {"props": [("color", "red")], "selectors": ["row0_col0"]},
            {"props": [("", "")], "selectors": ["row1_col0"]},
        ]
        assert result == expected

    def test_duplicate(self):
        # [FUNCTION DOC]
        # Verifies that cells with identical styles are merged into a single entry
        # listing both selectors in the translated cell style list.
        df = DataFrame({"A": [1, 0]})
        s = df.style
        s.ctx = {(0, 0): [("color", "red")], (1, 0): [("color", "red")]}

        result = s._translate(True, True)["cellstyle"]
        expected = [
            {"props": [("color", "red")], "selectors": ["row0_col0", "row1_col0"]}
        ]
        assert result == expected

    def test_init_with_na_rep(self):
        # [FUNCTION DOC]
        # Regression test (GH 21527 and 28358): the missing-value text given to the
        # constructor is used as display value of missing cells.
        # GH 21527 28358
        df = DataFrame([[None, None], [1.1, 1.2]], columns=["A", "B"])

        ctx = Styler(df, na_rep="NA")._translate(True, True)
        assert ctx["body"][0][1]["display_value"] == "NA"
        assert ctx["body"][0][2]["display_value"] == "NA"

    def test_caption(self, df):
        # [FUNCTION DOC]
        # Verifies a caption given to the constructor appears in the html output, and the
        # caption setter stores the value and returns the same styler (chaining).
        styler = Styler(df, caption="foo")
        result = styler.to_html()
        assert all(["caption" in result, "foo" in result])

        styler = df.style
        result = styler.set_caption("baz")
        assert styler is result
        assert styler.caption == "baz"

    def test_uuid(self, df):
        # [FUNCTION DOC]
        # Verifies a unique id given to the constructor appears in the html output, and
        # the id setter stores the value and returns the same styler.
        styler = Styler(df, uuid="abc123")
        result = styler.to_html()
        assert "abc123" in result

        styler = df.style
        result = styler.set_uuid("aaa")
        assert result is styler
        assert result.uuid == "aaa"

    def test_unique_id(self):
        # [FUNCTION DOC]
        # Regression test (issue 16780): all element ids in the rendered html are unique,
        # with an id prefix passed at render time.
        # See https://github.com/pandas-dev/pandas/issues/16780
        df = DataFrame({"a": [1, 3, 5, 6], "b": [2, 4, 12, 21]})
        result = df.style.to_html(uuid="test")
        assert "test" in result
        ids = re.findall('id="(.*?)"', result)
        assert np.unique(ids).size == len(ids)

    def test_table_styles(self, df):
        # [FUNCTION DOC]
        # Verifies table styles given to the constructor as a list of selector and
        # property-tuples render a css rule; the setter stores them and returns the same
        # styler; and (GH 39563) the css string format for the properties renders the same
        # rule.
        style = [{"selector": "th", "props": [("foo", "bar")]}]  # default format
        styler = Styler(df, table_styles=style)
        result = " ".join(styler.to_html().split())
        assert "th { foo: bar; }" in result

        styler = df.style
        result = styler.set_table_styles(style)
        assert styler is result
        assert styler.table_styles == style

        # GH 39563
        style = [{"selector": "th", "props": "foo:bar;"}]  # css string format
        styler = df.style.set_table_styles(style)
        result = " ".join(styler.to_html().split())
        assert "th { foo: bar; }" in result

    def test_table_styles_multiple(self, df):
        # [FUNCTION DOC]
        # Verifies that a style with a comma-separated selector is split into one entry
        # per selector in the translated table styles, with properties parsed to tuples.
        ctx = df.style.set_table_styles(
            [
                {"selector": "th,td", "props": "color:red;"},
                {"selector": "tr", "props": "color:green;"},
            ]
        )._translate(True, True)["table_styles"]
        assert ctx == [
            {"selector": "th", "props": [("color", "red")]},
            {"selector": "td", "props": [("color", "red")]},
            {"selector": "tr", "props": [("color", "green")]},
        ]

    def test_table_styles_dict_multiple_selectors(self, df):
        # [FUNCTION DOC]
        # Regression test (GH 44011): a dict of table styles keyed by a column label with
        # a comma-separated selector expands to one entry per selector with the column
        # class appended.
        # GH 44011
        result = df.style.set_table_styles(
            {
                "B": [
                    {"selector": "th,td", "props": [("border-left", "2px solid black")]}
                ]
            }
        )._translate(True, True)["table_styles"]

        expected = [
            {"selector": "th.col1", "props": [("border-left", "2px solid black")]},
            {"selector": "td.col1", "props": [("border-left", "2px solid black")]},
        ]

        assert result == expected

    def test_maybe_convert_css_to_tuples(self):
        # [FUNCTION DOC]
        # Verifies the private css string conversion: declarations separated by
        # semicolons become pairs with trimmed names and values; an empty string gives an
        # empty list; a value containing a colon and quotes is kept whole (issue 59623);
        # a declaration with no value gives an empty value.
        expected = [("a", "b"), ("c", "d e")]
        assert maybe_convert_css_to_tuples("a:b;c:d e;") == expected
        assert maybe_convert_css_to_tuples("a: b ;c:  d e  ") == expected
        expected = []
        assert maybe_convert_css_to_tuples("") == expected

        # issue #59623
        expected = [("a", "b"), ("c", "url('data:123')")]
        assert maybe_convert_css_to_tuples("a:b;c: url('data:123');") == expected

        # if no value, return attr and empty string
        expected = [("a", ""), ("c", "")]
        assert maybe_convert_css_to_tuples("a:;c: ") == expected

    def test_maybe_convert_css_to_tuples_err(self):
        # [FUNCTION DOC]
        # Verifies the private css conversion raises a value error with a message about
        # the required rule format when a declaration has no colon.
        msg = (
            "Styles supplied as string must follow CSS rule formats, "
            "for example 'attr: val;'. 'err' was given."
        )
        with pytest.raises(ValueError, match=msg):
            maybe_convert_css_to_tuples("err")

    def test_table_attributes(self, df):
        # [FUNCTION DOC]
        # Verifies that table attributes passed to the constructor, and set through the
        # setter, both appear in the html output.
        attributes = 'class="foo" data-bar'
        styler = Styler(df, table_attributes=attributes)
        result = styler.to_html()
        assert 'class="foo" data-bar' in result

        result = df.style.set_table_attributes(attributes).to_html()
        assert 'class="foo" data-bar' in result

    def test_apply_none(self):
        # [FUNCTION DOC]
        # Verifies application with no axis of a function returning a frame of css
        # strings (red for the maximum cell of the frame): only the maximum cell gets the
        # style.
        def f(x):
            # [FUNCTION DOC]
            # HELPER: style function for the no-axis test. Takes a frame x and returns a frame
            # with the same labels containing a red colour declaration where the value equals
            # the overall maximum and an empty string elsewhere.
            return DataFrame(
                np.where(x == x.max(), "color: red", ""),
                index=x.index,
                columns=x.columns,
            )

        result = DataFrame([[1, 2], [3, 4]]).style.apply(f, axis=None)._compute().ctx
        assert result[(1, 1)] == [("color", "red")]

    def test_trim(self, df):
        # [FUNCTION DOC]
        # Verifies that html without any colour styles contains no hash characters, and
        # after the maximum highlight there is exactly one per column (one colour code
        # each).
        result = df.style.to_html()  # trim=True
        assert result.count("#") == 0

        result = df.style.highlight_max().to_html()
        assert result.count("#") == len(df.columns)

    def test_export(self, df, styler):
        # [FUNCTION DOC]
        # Verifies exporting pending styles from one styler and using them on a second
        # styler over the same frame gives equal pending function lists, and the second
        # styler renders. The first styler is given element-wise styles (one with an
        # extra argument) and a maximum highlight, and is computed before export.
        f = lambda x: "color: red" if x > 0 else "color: blue"
        g = lambda x, z: f"color: {z}"
        style1 = styler
        style1.map(f).map(g, z="b").highlight_max()._compute()  # = render
        result = style1.export()
        style2 = df.style
        style2.use(result)
        assert style1._todo == style2._todo
        style2.to_html()

    def test_bad_apply_shape(self):
        # [FUNCTION DOC]
        # Verifies validation errors from the private apply method: a function returning a
        # scalar collapses to a series (error text about collapsing); wrong-length or
        # wrong-label returns on the row axis raise errors about invalid index labels;
        # wrong-length or wrong-label returns on the column axis raise errors about invalid
        # column labels; an array of the wrong shape with no axis raises an error about the
        # wrong shape.
        df = DataFrame([[1, 2], [3, 4]], index=["A", "B"], columns=["X", "Y"])

        msg = "resulted in the apply method collapsing to a Series."
        with pytest.raises(ValueError, match=msg):
            df.style._apply(lambda x: "x")

        msg = "created invalid {} labels"
        with pytest.raises(ValueError, match=msg.format("index")):
            df.style._apply(lambda x: [""])

        with pytest.raises(ValueError, match=msg.format("index")):
            df.style._apply(lambda x: ["", "", "", ""])

        with pytest.raises(ValueError, match=msg.format("index")):
            df.style._apply(lambda x: Series(["a:v;", ""], index=["A", "C"]), axis=0)

        with pytest.raises(ValueError, match=msg.format("columns")):
            df.style._apply(lambda x: ["", "", ""], axis=1)

        with pytest.raises(ValueError, match=msg.format("columns")):
            df.style._apply(lambda x: Series(["a:v;", ""], index=["X", "Z"]), axis=1)

        msg = "returned ndarray with wrong shape"
        with pytest.raises(ValueError, match=msg):
            df.style._apply(lambda x: np.array([[""], [""]]), axis=None)

    def test_apply_bad_return(self):
        # [FUNCTION DOC]
        # Verifies the private apply method raises a type error saying the function must
        # return a frame or array when used with no axis, if it returns a string.
        def f(x):
            # [FUNCTION DOC]
            # HELPER: style function returning an empty string for any input (invalid for a
            # no-axis apply).
            return ""

        df = DataFrame([[1, 2], [3, 4]])
        msg = (
            "must return a DataFrame or ndarray when passed to `Styler.apply` "
            "with axis=None"
        )
        with pytest.raises(TypeError, match=msg):
            df.style._apply(f, axis=None)

    @pytest.mark.parametrize("axis", ["index", "columns"])
    def test_apply_bad_labels(self, axis):
        # [FUNCTION DOC]
        # Verifies the private apply method raises a value error about invalid index (or
        # column) labels when a no-axis function returns a frame with unknown labels.
        # Parametrized over which axis carries the bad labels.
        def f(x):
            # [FUNCTION DOC]
            # HELPER: style function returning an empty frame whose index or columns (chosen by
            # the test's axis parameter) are two labels not in the data.
            return DataFrame(**{axis: ["bad", "labels"]})

        df = DataFrame([[1, 2], [3, 4]])
        msg = f"created invalid {axis} labels."
        with pytest.raises(ValueError, match=msg):
            df.style._apply(f, axis=None)

    def test_get_level_lengths(self):
        # [FUNCTION DOC]
        # Verifies the private level length helper on a two-level index with sparsify on
        # (outer labels get a length of three, inner labels one) and off (every entry
        # one). Compared as dictionaries keyed by (level, position).
        index = MultiIndex.from_product([["a", "b"], [0, 1, 2]])
        expected = {
            (0, 0): 3,
            (0, 3): 3,
            (1, 0): 1,
            (1, 1): 1,
            (1, 2): 1,
            (1, 3): 1,
            (1, 4): 1,
            (1, 5): 1,
        }
        result = _get_level_lengths(index, sparsify=True, max_index=100)
        tm.assert_dict_equal(result, expected)

        expected = {
            (0, 0): 1,
            (0, 1): 1,
            (0, 2): 1,
            (0, 3): 1,
            (0, 4): 1,
            (0, 5): 1,
            (1, 0): 1,
            (1, 1): 1,
            (1, 2): 1,
            (1, 3): 1,
            (1, 4): 1,
            (1, 5): 1,
        }
        result = _get_level_lengths(index, sparsify=False, max_index=100)
        tm.assert_dict_equal(result, expected)

    def test_get_level_lengths_un_sorted(self):
        # [FUNCTION DOC]
        # Verifies the private level length helper on an unsorted index: a run is counted
        # only for adjacent equal labels (the first two entries) when sparsify is on, and
        # every entry has length one when off.
        index = MultiIndex.from_arrays([[1, 1, 2, 1], ["a", "b", "b", "d"]])
        expected = {
            (0, 0): 2,
            (0, 2): 1,
            (0, 3): 1,
            (1, 0): 1,
            (1, 1): 1,
            (1, 2): 1,
            (1, 3): 1,
        }
        result = _get_level_lengths(index, sparsify=True, max_index=100)
        tm.assert_dict_equal(result, expected)

        expected = {
            (0, 0): 1,
            (0, 1): 1,
            (0, 2): 1,
            (0, 3): 1,
            (1, 0): 1,
            (1, 1): 1,
            (1, 2): 1,
            (1, 3): 1,
        }
        result = _get_level_lengths(index, sparsify=False, max_index=100)
        tm.assert_dict_equal(result, expected)

    def test_mi_sparse_index_names(self, blank_value):
        # [FUNCTION DOC]
        # Verifies the header row with index names for a two-level named row index: each
        # index name cell has the index name class for its level, the name as display
        # value and is visible, followed by a blank data-column cell.
        # Test the class names and displayed value are correct on rendering MI names
        df = DataFrame(
            {"A": [1, 2]},
            index=MultiIndex.from_arrays(
                [["a", "a"], [0, 1]], names=["idx_level_0", "idx_level_1"]
            ),
        )
        result = df.style._translate(True, True)
        head = result["head"][1]
        expected = [
            {
                "class": "index_name level0",
                "display_value": "idx_level_0",
                "is_visible": True,
            },
            {
                "class": "index_name level1",
                "display_value": "idx_level_1",
                "is_visible": True,
            },
            {
                "class": "blank col0",
                "display_value": blank_value,
                "is_visible": True,
            },
        ]
        for i, expected_dict in enumerate(expected):
            assert expected_dict.items() <= head[i].items()

    def test_mi_sparse_column_names(self, blank_value):
        # [FUNCTION DOC]
        # Verifies the header rows for a two-level named column index with cell ids off:
        # for each level the row has a blank cell and a cell with the index name class for
        # that level showing the column level name.
        df = DataFrame(
            np.arange(16).reshape(4, 4),
            index=MultiIndex.from_arrays(
                [["a", "a", "b", "a"], [0, 1, 1, 2]],
                names=["idx_level_0", "idx_level_1"],
            ),
            columns=MultiIndex.from_arrays(
                [["C1", "C1", "C2", "C2"], [1, 0, 1, 0]], names=["colnam_0", "colnam_1"]
            ),
        )
        result = Styler(df, cell_ids=False)._translate(True, True)

        for level in [0, 1]:
            head = result["head"][level]
            expected = [
                {
                    "class": "blank",
                    "display_value": blank_value,
                    "is_visible": True,
                },
                {
                    "class": f"index_name level{level}",
                    "display_value": f"colnam_{level}",
                    "is_visible": True,
                },
            ]
            for i, expected_dict in enumerate(expected):
                assert expected_dict.items() <= head[i].items()

    def test_hide_column_headers(self, df, styler):
        # [FUNCTION DOC]
        # Verifies that hiding columns with an unnamed index leaves no header rows, and
        # with a named index (name set on the fixture frame) one header row remains for
        # the index name (a behaviour changed in 42101 and reverted in 43404).
        ctx = styler.hide(axis="columns")._translate(True, True)
        assert len(ctx["head"]) == 0  # no header entries with an unnamed index

        df.index.name = "some_name"
        ctx = df.style.hide(axis="columns")._translate(True, True)
        assert len(ctx["head"]) == 1
        # index names still visible, changed in #42101, reverted in 43404

    def test_hide_single_index(self, df):
        # [FUNCTION DOC]
        # Regression test (GH 14194): with a single-level unnamed index, hiding the index
        # makes the corner and body index cells not visible; with a named index the header
        # has two rows and after hiding the index only one, with body and corner not
        # visible.
        # GH 14194
        # single unnamed index
        ctx = df.style._translate(True, True)
        assert ctx["body"][0][0]["is_visible"]
        assert ctx["head"][0][0]["is_visible"]
        ctx2 = df.style.hide(axis="index")._translate(True, True)
        assert not ctx2["body"][0][0]["is_visible"]
        assert not ctx2["head"][0][0]["is_visible"]

        # single named index
        ctx3 = df.set_index("A").style._translate(True, True)
        assert ctx3["body"][0][0]["is_visible"]
        assert len(ctx3["head"]) == 2  # 2 header levels
        assert ctx3["head"][0][0]["is_visible"]

        ctx4 = df.set_index("A").style.hide(axis="index")._translate(True, True)
        assert not ctx4["body"][0][0]["is_visible"]
        assert len(ctx4["head"]) == 1  # only 1 header levels
        assert not ctx4["head"][0][0]["is_visible"]

    def test_hide_multiindex(self):
        # [FUNCTION DOC]
        # Regression test (GH 14194): hiding a two-level named row index makes its body
        # cells and header corner invisible and reduces the first header row length from
        # four (two index cells and two data cells) to three.
        # GH 14194
        df = DataFrame(
            {"A": [1, 2], "B": [1, 2]},
            index=MultiIndex.from_arrays(
                [["a", "a"], [0, 1]], names=["idx_level_0", "idx_level_1"]
            ),
        )
        ctx1 = df.style._translate(True, True)
        # tests for 'a' and '0'
        assert ctx1["body"][0][0]["is_visible"]
        assert ctx1["body"][0][1]["is_visible"]
        # check for blank header rows
        assert len(ctx1["head"][0]) == 4  # two visible indexes and two data columns

        ctx2 = df.style.hide(axis="index")._translate(True, True)
        # tests for 'a' and '0'
        assert not ctx2["body"][0][0]["is_visible"]
        assert not ctx2["body"][0][1]["is_visible"]
        # check for blank header rows
        assert len(ctx2["head"][0]) == 3  # one hidden (col name) and two data columns
        assert not ctx2["head"][0][0]["is_visible"]

    def test_hide_columns_single_level(self, df):
        # [FUNCTION DOC]
        # Regression test (GH 14194): hiding one column by label, and then both, makes
        # those columns' header and body cells not visible while the others stay visible.
        # GH 14194
        # test hiding single column
        ctx = df.style._translate(True, True)
        assert ctx["head"][0][1]["is_visible"]
        assert ctx["head"][0][1]["display_value"] == "A"
        assert ctx["head"][0][2]["is_visible"]
        assert ctx["head"][0][2]["display_value"] == "B"
        assert ctx["body"][0][1]["is_visible"]  # col A, row 1
        assert ctx["body"][1][2]["is_visible"]  # col B, row 1

        ctx = df.style.hide("A", axis="columns")._translate(True, True)
        assert not ctx["head"][0][1]["is_visible"]
        assert not ctx["body"][0][1]["is_visible"]  # col A, row 1
        assert ctx["body"][1][2]["is_visible"]  # col B, row 1

        # test hiding multiple columns
        ctx = df.style.hide(["A", "B"], axis="columns")._translate(True, True)
        assert not ctx["head"][0][1]["is_visible"]
        assert not ctx["head"][0][2]["is_visible"]
        assert not ctx["body"][0][1]["is_visible"]  # col A, row 1
        assert not ctx["body"][1][2]["is_visible"]  # col B, row 1

    def test_hide_columns_index_mult_levels(self):
        # [FUNCTION DOC]
        # Regression test (GH 14194): with two-level row and column indexes, checks
        # visibility of header, index and data cells when hiding the top column level
        # (hides both columns), hiding one column by tuple, hiding one column and the
        # whole index, hiding the top row level (empty body), and hiding one row by tuple
        # (only the other row's cells remain, with the row class of the second row).
        # GH 14194
        # setup dataframe with multiple column levels and indices
        i1 = MultiIndex.from_arrays(
            [["a", "a"], [0, 1]], names=["idx_level_0", "idx_level_1"]
        )
        i2 = MultiIndex.from_arrays(
            [["b", "b"], [0, 1]], names=["col_level_0", "col_level_1"]
        )
        df = DataFrame([[1, 2], [3, 4]], index=i1, columns=i2)
        ctx = df.style._translate(True, True)
        # column headers
        assert ctx["head"][0][2]["is_visible"]
        assert ctx["head"][1][2]["is_visible"]
        assert ctx["head"][1][3]["display_value"] == "1"
        # indices
        assert ctx["body"][0][0]["is_visible"]
        # data
        assert ctx["body"][1][2]["is_visible"]
        assert ctx["body"][1][2]["display_value"] == "3"
        assert ctx["body"][1][3]["is_visible"]
        assert ctx["body"][1][3]["display_value"] == "4"

        # hide top column level, which hides both columns
        ctx = df.style.hide("b", axis="columns")._translate(True, True)
        assert not ctx["head"][0][2]["is_visible"]  # b
        assert not ctx["head"][1][2]["is_visible"]  # 0
        assert not ctx["body"][1][2]["is_visible"]  # 3
        assert ctx["body"][0][0]["is_visible"]  # index

        # hide first column only
        ctx = df.style.hide([("b", 0)], axis="columns")._translate(True, True)
        assert not ctx["head"][0][2]["is_visible"]  # b
        assert ctx["head"][0][3]["is_visible"]  # b
        assert not ctx["head"][1][2]["is_visible"]  # 0
        assert not ctx["body"][1][2]["is_visible"]  # 3
        assert ctx["body"][1][3]["is_visible"]
        assert ctx["body"][1][3]["display_value"] == "4"

        # hide second column and index
        ctx = df.style.hide([("b", 1)], axis=1).hide(axis=0)._translate(True, True)
        assert not ctx["body"][0][0]["is_visible"]  # index
        assert len(ctx["head"][0]) == 3
        assert ctx["head"][0][1]["is_visible"]  # b
        assert ctx["head"][1][1]["is_visible"]  # 0
        assert not ctx["head"][1][2]["is_visible"]  # 1
        assert not ctx["body"][1][3]["is_visible"]  # 4
        assert ctx["body"][1][2]["is_visible"]
        assert ctx["body"][1][2]["display_value"] == "3"

        # hide top row level, which hides both rows so body empty
        ctx = df.style.hide("a", axis="index")._translate(True, True)
        assert ctx["body"] == []

        # hide first row only
        ctx = df.style.hide(("a", 0), axis="index")._translate(True, True)
        for i in [0, 1, 2, 3]:
            assert "row1" in ctx["body"][0][i]["class"]  # row0 not included in body
            assert ctx["body"][0][i]["is_visible"]

    def test_pipe(self, df):
        # [FUNCTION DOC]
        # Verifies the pipe method: a function taking the styler plus positional and
        # keyword arguments can set the caption, and a (callable, keyword name) pair
        # passes the styler under that keyword so the result of the callable is returned.
        def set_caption_from_template(styler, a, b):
            # [FUNCTION DOC]
            # HELPER: function used with the pipe method: sets the styler caption to a text
            # built from the two arguments and returns the styler.
            return styler.set_caption(f"Dataframe with a = {a} and b = {b}")

        styler = df.style.pipe(set_caption_from_template, "A", b="B")
        assert "Dataframe with a = A and b = B" in styler.to_html()

        # Test with an argument that is a (callable, keyword_name) pair.
        def f(a, b, styler):
            # [FUNCTION DOC]
            # HELPER: function used with the pipe method's keyword form: returns the tuple of
            # its three arguments.
            return (a, b, styler)

        styler = df.style
        result = styler.pipe((f, "styler"), a=1, b=2)
        assert result == (1, 2, styler)

    def test_no_cell_ids(self):
        # [FUNCTION DOC]
        # Regression test (GH 35588 and 35663): with cell ids disabled and a fixed id,
        # rendering twice (to ensure the context is not updated by rendering) still gives
        # a data cell without an id.
        # GH 35588
        # GH 35663
        df = DataFrame(data=[[0]])
        styler = Styler(df, uuid="_", cell_ids=False)
        styler.to_html()
        s = styler.to_html()  # render twice to ensure ctx is not updated
        assert s.find('<td class="data row0 col0" >') != -1

    @pytest.mark.parametrize(
        "classes",
        [
            DataFrame(
                data=[["", "test-class"], [np.nan, None]],
                columns=["A", "B"],
                index=["a", "b"],
            ),
            DataFrame(data=[["test-class"]], columns=["B"], index=["a"]),
            DataFrame(data=[["test-class", "unused"]], columns=["B", "C"], index=["a"]),
        ],
    )
    def test_set_data_classes(self, classes):
        # [FUNCTION DOC]
        # Regression test (GH 36159 and 39317): the cell classes setter applies a class
        # only to the cell with a non-empty entry; frames with missing entries, a
        # subset of labels, or extra unused columns are accepted. Checks the html both
        # without and with cell ids. Parametrized over three class frames.
        # GH 36159
        df = DataFrame(data=[[0, 1], [2, 3]], columns=["A", "B"], index=["a", "b"])
        s = Styler(df, uuid_len=0, cell_ids=False).set_td_classes(classes).to_html()
        assert '<td class="data row0 col0" >0</td>' in s
        assert '<td class="data row0 col1 test-class" >1</td>' in s
        assert '<td class="data row1 col0" >2</td>' in s
        assert '<td class="data row1 col1" >3</td>' in s
        # GH 39317
        s = Styler(df, uuid_len=0, cell_ids=True).set_td_classes(classes).to_html()
        assert '<td id="T__row0_col0" class="data row0 col0" >0</td>' in s
        assert '<td id="T__row0_col1" class="data row0 col1 test-class" >1</td>' in s
        assert '<td id="T__row1_col0" class="data row1 col0" >2</td>' in s
        assert '<td id="T__row1_col1" class="data row1 col1" >3</td>' in s

    def test_set_data_classes_reindex(self):
        # [FUNCTION DOC]
        # Regression test (GH 39317): the cell classes frame is reindexed to the data, so
        # a smaller frame with some labels missing only affects the matching cells.
        # GH 39317
        df = DataFrame(
            data=[[0, 1, 2], [3, 4, 5], [6, 7, 8]], columns=[0, 1, 2], index=[0, 1, 2]
        )
        classes = DataFrame(
            data=[["mi", "ma"], ["mu", "mo"]],
            columns=[0, 2],
            index=[0, 2],
        )
        s = Styler(df, uuid_len=0).set_td_classes(classes).to_html()
        assert '<td id="T__row0_col0" class="data row0 col0 mi" >0</td>' in s
        assert '<td id="T__row0_col2" class="data row0 col2 ma" >2</td>' in s
        assert '<td id="T__row1_col1" class="data row1 col1" >4</td>' in s
        assert '<td id="T__row2_col0" class="data row2 col0 mu" >6</td>' in s
        assert '<td id="T__row2_col2" class="data row2 col2 mo" >8</td>' in s

    def test_chaining_table_styles(self):
        # [FUNCTION DOC]
        # Regression test (GH 35607): a second table style set call with overwrite off
        # appends to the first, giving two styles.
        # GH 35607
        df = DataFrame(data=[[0, 1], [1, 2]], columns=["A", "B"])
        styler = df.style.set_table_styles(
            [{"selector": "", "props": [("background-color", "yellow")]}]
        ).set_table_styles(
            [{"selector": ".col0", "props": [("background-color", "blue")]}],
            overwrite=False,
        )
        assert len(styler.table_styles) == 2

    def test_column_and_row_styling(self):
        # [FUNCTION DOC]
        # Regression test (GH 35607): a dict of table styles keyed by a column label gives
        # a rule with the column class; keyed by a row label with the row axis gives a rule
        # with the row class.
        # GH 35607
        df = DataFrame(data=[[0, 1], [1, 2]], columns=["A", "B"])
        s = Styler(df, uuid_len=0)
        s = s.set_table_styles({"A": [{"selector": "", "props": [("color", "blue")]}]})
        assert "#T_ .col0 {\n  color: blue;\n}" in s.to_html()
        s = s.set_table_styles(
            {0: [{"selector": "", "props": [("color", "blue")]}]}, axis=1
        )
        assert "#T_ .row0 {\n  color: blue;\n}" in s.to_html()

    @pytest.mark.parametrize("len_", [1, 5, 32, 33, 100])
    def test_uuid_len(self, len_):
        # [FUNCTION DOC]
        # Regression test (GH 36345): the generated unique id has the requested length,
        # capped at thirty-two. Parametrized over lengths one, five, thirty-two,
        # thirty-three and one hundred.
        # GH 36345
        df = DataFrame(data=[["A"]])
        s = Styler(df, uuid_len=len_, cell_ids=False).to_html()
        strt = s.find('id="T_')
        end = s[strt + 6 :].find('"')
        if len_ > 32:
            assert end == 32
        else:
            assert end == len_

    @pytest.mark.parametrize("len_", [-2, "bad", None])
    def test_uuid_len_raises(self, len_):
        # [FUNCTION DOC]
        # Regression test (GH 36345): an id length that is negative, not an integer or
        # none raises a type error saying it must be an integer in the range zero to
        # thirty-two, at render time. Parametrized over three invalid values.
        # GH 36345
        df = DataFrame(data=[["A"]])
        msg = "``uuid_len`` must be an integer in range \\[0, 32\\]."
        with pytest.raises(TypeError, match=msg):
            Styler(df, uuid_len=len_, cell_ids=False).to_html()

    @pytest.mark.parametrize(
        "slc",
        [
            IndexSlice[:, :],
            IndexSlice[:, 1],
            IndexSlice[1, :],
            IndexSlice[[1], [1]],
            IndexSlice[1, [1]],
            IndexSlice[[1], 1],
            IndexSlice[1],
            IndexSlice[1, 1],
            slice(None, None, None),
            [0, 1],
            np.array([0, 1]),
            Series([0, 1]),
        ],
    )
    def test_non_reducing_slice(self, slc):
        # [FUNCTION DOC]
        # Verifies the private slice helper turns a variety of subset forms (full and
        # partial slices, label lists, scalars, arrays, series) into a form that always
        # returns a frame when used to select from a frame. Parametrized over twelve
        # subset forms.
        df = DataFrame([[0, 1], [2, 3]])

        tslice_ = non_reducing_slice(slc)
        assert isinstance(df.loc[tslice_], DataFrame)

    @pytest.mark.parametrize("box", [list, Series, np.array])
    def test_list_slice(self, box):
        # [FUNCTION DOC]
        # Verifies the private slice helper treats a list, series or array of labels as
        # column selection (like frame item access): selecting with the result equals
        # selecting the column with a slice-all row. Parametrized over the container type.
        # like dataframe getitem
        subset = box(["A"])

        df = DataFrame({"A": [1, 2], "B": [3, 4]}, index=["A", "B"])
        expected = IndexSlice[:, ["A"]]

        result = non_reducing_slice(subset)
        tm.assert_frame_equal(df.loc[result], df.loc[expected])

    def test_non_reducing_slice_on_multiindex(self):
        # [FUNCTION DOC]
        # Regression test (GH 19861): the private slice helper keeps the frame shape when
        # the slice selects a single tuple column of a multi-level column index; the result
        # equals the one-column frame.
        # GH 19861
        dic = {
            ("a", "d"): [1, 4],
            ("a", "c"): [2, 3],
            ("b", "c"): [3, 2],
            ("b", "d"): [4, 1],
        }
        df = DataFrame(dic, index=[0, 1])
        idx = IndexSlice
        slice_ = idx[:, idx["b", "d"]]
        tslice_ = non_reducing_slice(slice_)

        result = df.loc[tslice_]
        expected = DataFrame({("b", "d"): [4, 1]})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "slice_",
        [
            IndexSlice[:, :],
            # check cols
            IndexSlice[:, IndexSlice[["a"]]],  # inferred deeper need list
            IndexSlice[:, IndexSlice[["a"], ["c"]]],  # inferred deeper need list
            IndexSlice[:, IndexSlice["a", "c", :]],
            IndexSlice[:, IndexSlice["a", :, "e"]],
            IndexSlice[:, IndexSlice[:, "c", "e"]],
            IndexSlice[:, IndexSlice["a", ["c", "d"], :]],  # check list
            IndexSlice[:, IndexSlice["a", ["c", "d", "-"], :]],  # don't allow missing
            IndexSlice[:, IndexSlice["a", ["c", "d", "-"], "e"]],  # no slice
            # check rows
            IndexSlice[IndexSlice[["U"]], :],  # inferred deeper need list
            IndexSlice[IndexSlice[["U"], ["W"]], :],  # inferred deeper need list
            IndexSlice[IndexSlice["U", "W", :], :],
            IndexSlice[IndexSlice["U", :, "Y"], :],
            IndexSlice[IndexSlice[:, "W", "Y"], :],
            IndexSlice[IndexSlice[:, "W", ["Y", "Z"]], :],  # check list
            IndexSlice[IndexSlice[:, "W", ["Y", "Z", "-"]], :],  # don't allow missing
            IndexSlice[IndexSlice["U", "W", ["Y", "Z", "-"]], :],  # no slice
            # check simultaneous
            IndexSlice[IndexSlice[:, "W", "Y"], IndexSlice["a", "c", :]],
        ],
    )
    def test_non_reducing_multi_slice_on_multiindex(self, slice_):
        # [FUNCTION DOC]
        # Regression test (GH 33562): selecting with the private slice helper result gives
        # the same frame as selecting with the original slice on three-level row and
        # column indexes. Parametrized over many slices. If a slice has a list containing a
        # label missing from its level, the original selection must raise a key error
        # naming it and the test then ends early.
        # GH 33562
        cols = MultiIndex.from_product([["a", "b"], ["c", "d"], ["e", "f"]])
        idxs = MultiIndex.from_product([["U", "V"], ["W", "X"], ["Y", "Z"]])
        df = DataFrame(np.arange(64).reshape(8, 8), columns=cols, index=idxs)

        for lvl in [0, 1]:
            key = slice_[lvl]
            if isinstance(key, tuple):
                for subkey in key:
                    if isinstance(subkey, list) and "-" in subkey:
                        # not present in the index level, raises KeyError since 2.0
                        with pytest.raises(KeyError, match="-"):
                            df.loc[slice_]
                        return

        expected = df.loc[slice_]
        result = df.loc[non_reducing_slice(slice_)]
        tm.assert_frame_equal(result, expected)


def test_hidden_index_names(mi_df):
    # [FUNCTION DOC]
    # Verifies that hiding the index names of a two-level named row index reduces the
    # header rows from three to two while all body cells remain visible, and that hiding
    # a level afterwards keeps the names row hidden and marks that level not visible.
    mi_df.index.names = ["Lev0", "Lev1"]
    mi_styler = mi_df.style
    ctx = mi_styler._translate(True, True)
    assert len(ctx["head"]) == 3  # 2 column index levels + 1 index names row

    mi_styler.hide(axis="index", names=True)
    ctx = mi_styler._translate(True, True)
    assert len(ctx["head"]) == 2  # index names row is unparsed
    for i in range(4):
        assert ctx["body"][0][i]["is_visible"]  # 2 index levels + 2 data values visible

    mi_styler.hide(axis="index", level=1)
    ctx = mi_styler._translate(True, True)
    assert len(ctx["head"]) == 2  # index names row is still hidden
    assert ctx["body"][0][0]["is_visible"] is True
    assert ctx["body"][0][1]["is_visible"] is False


def test_hidden_column_names(mi_df):
    # [FUNCTION DOC]
    # Verifies that column level names are displayed in the header cells by default,
    # become blank when column names are hidden, and that additionally hiding a column
    # level leaves one header row with a blank cell.
    mi_df.columns.names = ["Lev0", "Lev1"]
    mi_styler = mi_df.style
    ctx = mi_styler._translate(True, True)
    assert ctx["head"][0][1]["display_value"] == "Lev0"
    assert ctx["head"][1][1]["display_value"] == "Lev1"

    mi_styler.hide(names=True, axis="columns")
    ctx = mi_styler._translate(True, True)
    assert ctx["head"][0][1]["display_value"] == "&nbsp;"
    assert ctx["head"][1][1]["display_value"] == "&nbsp;"

    mi_styler.hide(level=0, axis="columns")
    ctx = mi_styler._translate(True, True)
    assert len(ctx["head"]) == 1  # no index names and only one visible column headers
    assert ctx["head"][0][1]["display_value"] == "&nbsp;"


@pytest.mark.parametrize("caption", [1, ("a", "b", "c"), (1, "s")])
def test_caption_raises(mi_styler, caption):
    # [FUNCTION DOC]
    # Verifies that a caption that is not a string or a pair of strings (an integer, a
    # triple, a mixed pair) raises a value error with the corresponding message.
    msg = "`caption` must be either a string or 2-tuple of strings."
    with pytest.raises(ValueError, match=msg):
        mi_styler.set_caption(caption)


def test_hiding_headers_over_index_no_sparsify():
    # [FUNCTION DOC]
    # Regression test (GH 43464): without sparsification on a row index with a
    # repeated label, hiding the repeated tuple removes two of six body rows and the
    # first remaining row has the class for the third row position.
    # GH 43464
    midx = MultiIndex.from_product([[1, 2], ["a", "a", "b"]])
    df = DataFrame(9, index=midx, columns=[0])
    ctx = df.style._translate(False, False)
    assert len(ctx["body"]) == 6
    ctx = df.style.hide((1, "a"), axis=0)._translate(False, False)
    assert len(ctx["body"]) == 4
    assert "row2" in ctx["body"][0][0]["class"]


def test_hiding_headers_over_columns_no_sparsify():
    # [FUNCTION DOC]
    # Regression test (GH 43464): without sparsification on a column index, header
    # cells are visible by default and all become not visible after hiding the repeated
    # tuple columns (checked for the first two header rows at positions one and two).
    # GH 43464
    midx = MultiIndex.from_product([[1, 2], ["a", "a", "b"]])
    df = DataFrame(9, columns=midx, index=[0])
    ctx = df.style._translate(False, False)
    for ix in [(0, 1), (0, 2), (1, 1), (1, 2)]:
        assert ctx["head"][ix[0]][ix[1]]["is_visible"] is True
    ctx = df.style.hide((1, "a"), axis="columns")._translate(False, False)
    for ix in [(0, 1), (0, 2), (1, 1), (1, 2)]:
        assert ctx["head"][ix[0]][ix[1]]["is_visible"] is False


def test_get_level_lengths_mi_hidden():
    # [FUNCTION DOC]
    # Regression test (GH 43464): the private level length helper with sparsify off and
    # a list of hidden element positions (which may repeat when the index has
    # duplicates) omits the hidden entries from the result.
    # GH 43464
    index = MultiIndex.from_arrays([[1, 1, 1, 2, 2, 2], ["a", "a", "b", "a", "a", "b"]])
    expected = {
        (0, 2): 1,
        (0, 3): 1,
        (0, 4): 1,
        (0, 5): 1,
        (1, 2): 1,
        (1, 3): 1,
        (1, 4): 1,
        (1, 5): 1,
    }
    result = _get_level_lengths(
        index,
        sparsify=False,
        max_index=100,
        hidden_elements=[0, 1, 0, 1],  # hidden element can repeat if duplicated index
    )
    tm.assert_dict_equal(result, expected)


def test_row_trimming_hide_index():
    # [FUNCTION DOC]
    # Regression test (gh 43703): with a limit of two rows and the first two rows
    # hidden, rendering shows the next two rows plus a trimming row, whose display
    # values are as expected.
    # gh 43703
    df = DataFrame([[1], [2], [3], [4], [5]])
    with option_context("styler.render.max_rows", 2):
        ctx = df.style.hide([0, 1], axis="index")._translate(True, True)
    assert len(ctx["body"]) == 3
    for r, val in enumerate(["3", "4", "..."]):
        assert ctx["body"][r][1]["display_value"] == val


def test_row_trimming_hide_index_mi():
    # [FUNCTION DOC]
    # Regression test (gh 44247): same as the row trimming test with a two-level row
    # index: the outer label is sparsified across the two visible rows with the hidden
    # continuation not visible, the trimming row shows a marker, and the second-level
    # labels and data values show the expected strings.
    # gh 44247
    df = DataFrame([[1], [2], [3], [4], [5]])
    df.index = MultiIndex.from_product([[0], [0, 1, 2, 3, 4]])
    with option_context("styler.render.max_rows", 2):
        ctx = df.style.hide([(0, 0), (0, 1)], axis="index")._translate(True, True)
    assert len(ctx["body"]) == 3

    # level 0 index headers (sparsified)
    assert {"value": 0, "attributes": 'rowspan="2"', "is_visible": True}.items() <= ctx[
        "body"
    ][0][0].items()
    assert {"value": 0, "attributes": "", "is_visible": False}.items() <= ctx["body"][
        1
    ][0].items()
    assert {"value": "...", "is_visible": True}.items() <= ctx["body"][2][0].items()

    for r, val in enumerate(["2", "3", "..."]):
        assert ctx["body"][r][1]["display_value"] == val  # level 1 index headers
    for r, val in enumerate(["3", "4", "..."]):
        assert ctx["body"][r][2]["display_value"] == val  # data values


def test_col_trimming_hide_columns():
    # [FUNCTION DOC]
    # Regression test (gh 44272): with a limit of two columns and the first two
    # columns hidden, the header has six cells (corner, two hidden, two visible and a
    # trimming cell) with the expected values and visibility, and the body rows have the
    # same width.
    # gh 44272
    df = DataFrame([[1, 2, 3, 4, 5]])
    with option_context("styler.render.max_columns", 2):
        ctx = df.style.hide([0, 1], axis="columns")._translate(True, True)

    assert len(ctx["head"][0]) == 6  # blank, [0, 1 (hidden)], [2 ,3 (visible)], + trim
    for c, vals in enumerate([(1, False), (2, True), (3, True), ("...", True)]):
        assert ctx["head"][0][c + 2]["value"] == vals[0]
        assert ctx["head"][0][c + 2]["is_visible"] == vals[1]

    assert len(ctx["body"][0]) == 6  # index + 2 hidden + 2 visible + trimming col


def test_no_empty_apply(mi_styler):
    # [FUNCTION DOC]
    # Regression test (45313): applying a style function with a subset of boolean
    # False values (selecting nothing) and computing does not raise.
    # 45313
    mi_styler.apply(lambda s: ["a:v;"] * 2, subset=[False, False])
    mi_styler._compute()


@pytest.mark.parametrize("format", ["html", "latex", "string"])
def test_output_buffer(mi_styler, format, temp_file):
    # [FUNCTION DOC]
    # Regression test (gh 47053): the html, latex and string outputs can each be
    # written to a file path from the temporary file fixture. Parametrized over format.
    # gh 47053
    getattr(mi_styler, f"to_{format}")(temp_file)
