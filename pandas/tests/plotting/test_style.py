# [MODULE DOC]
# Tests for the color-selection helper used by the matplotlib plotting backend.
# Skipped entirely when matplotlib is not installed (import-or-skip at import time).
# Imports the private standard-color helper from the matplotlib backend style module
# and checks how it expands the matplotlib property cycle or user-supplied colors
# to a requested count, and how it rejects invalid color arguments.

import pytest

from pandas import Series

mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")
from pandas.plotting._matplotlib.style import get_standard_colors


class TestGetStandardColors:
    # [CLASS DOC]
    # Groups tests of the standard-color helper: behavior with the default property cycle
    # (list or string form, or undefined) and with user-specified color sequences,
    # strings, float triples and named colors, plus error cases.
    # Stateless; each test uses a temporary matplotlib rc context where needed.
    @pytest.mark.parametrize(
        "num_colors, expected",
        [
            (3, ["red", "green", "blue"]),
            (5, ["red", "green", "blue", "red", "green"]),
            (7, ["red", "green", "blue", "red", "green", "blue", "red"]),
            (2, ["red", "green"]),
            (1, ["red"]),
        ],
    )
    def test_default_colors_named_from_prop_cycle(self, num_colors, expected):
        # [FUNCTION DOC]
        # Verifies that, with no user color, colors come from the rc property cycle (a list of
        # three named colors) and are repeated cyclically up to the requested count.
        # Parametrized by the requested number of colors and the expected list of names.
        # The result must be the same color as expected, compared via matplotlib.
        mpl_params = {
            "axes.prop_cycle": plt.cycler(color=["red", "green", "blue"]),
        }
        with mpl.rc_context(rc=mpl_params):
            result = get_standard_colors(num_colors=num_colors)
            assert mpl.colors.same_color(result, expected)

    @pytest.mark.parametrize(
        "num_colors, expected",
        [
            (1, ["b"]),
            (3, ["b", "g", "r"]),
            (4, ["b", "g", "r", "y"]),
            (5, ["b", "g", "r", "y", "b"]),
            (7, ["b", "g", "r", "y", "b", "g", "r"]),
        ],
    )
    def test_default_colors_named_from_prop_cycle_string(self, num_colors, expected):
        # [FUNCTION DOC]
        # Same as the list-cycle test but the property cycle is defined with a string of
        # single-letter color codes. The result is cycled to the requested count and must match
        # the expected letters as colors.
        mpl_params = {
            "axes.prop_cycle": plt.cycler(color="bgry"),
        }
        with mpl.rc_context(rc=mpl_params):
            result = get_standard_colors(num_colors=num_colors)
            assert mpl.colors.same_color(result, expected)

    @pytest.mark.parametrize(
        "num_colors, expected",
        [
            (1, ["C0"]),
            (3, ["C0", "C1", "C2"]),
            (
                12,
                [
                    "C0",
                    "C1",
                    "C2",
                    "C3",
                    "C4",
                    "C5",
                    "C6",
                    "C7",
                    "C8",
                    "C9",
                    "C0",
                    "C1",
                ],
            ),
        ],
    )
    def test_default_colors_named_undefined_prop_cycle(self, num_colors, expected):
        # [FUNCTION DOC]
        # Verifies that with an empty rc context (no property cycle override) the default
        # matplotlib cycle colors named by index are returned, wrapping around after ten.
        # Parametrized by color count and expected names; compared as colors.
        with mpl.rc_context(rc={}):
            result = get_standard_colors(num_colors=num_colors)
            assert mpl.colors.same_color(result, expected)

    @pytest.mark.parametrize(
        "num_colors, expected",
        [
            (1, ["red", "green", (0.1, 0.2, 0.3)]),
            (2, ["red", "green", (0.1, 0.2, 0.3)]),
            (3, ["red", "green", (0.1, 0.2, 0.3)]),
            (4, ["red", "green", (0.1, 0.2, 0.3), "red"]),
        ],
    )
    def test_user_input_color_sequence(self, num_colors, expected):
        # [FUNCTION DOC]
        # Verifies that a user-provided list of mixed color names and a float triple is
        # returned as a list. When the requested count is smaller than the list, the full list
        # is still returned (no truncation); when larger, it is repeated cyclically.
        # Parametrized by count and expected list; compared with plain equality.
        color = ["red", "green", (0.1, 0.2, 0.3)]
        result = get_standard_colors(color=color, num_colors=num_colors)
        assert result == expected

    @pytest.mark.parametrize(
        "num_colors, expected",
        [
            (1, ["r", "g", "b", "k"]),
            (2, ["r", "g", "b", "k"]),
            (3, ["r", "g", "b", "k"]),
            (4, ["r", "g", "b", "k"]),
            (5, ["r", "g", "b", "k", "r"]),
            (6, ["r", "g", "b", "k", "r", "g"]),
        ],
    )
    def test_user_input_color_string(self, num_colors, expected):
        # [FUNCTION DOC]
        # Verifies that a user color string of single-letter codes is treated as a sequence of
        # colors: kept whole when the count is smaller or equal, and cycled when larger.
        # Parametrized by count and expected list; compared with plain equality.
        color = "rgbk"
        result = get_standard_colors(color=color, num_colors=num_colors)
        assert result == expected

    @pytest.mark.parametrize(
        "num_colors, expected",
        [
            (1, [(0.1, 0.2, 0.3)]),
            (2, [(0.1, 0.2, 0.3), (0.1, 0.2, 0.3)]),
            (3, [(0.1, 0.2, 0.3), (0.1, 0.2, 0.3), (0.1, 0.2, 0.3)]),
        ],
    )
    def test_user_input_color_floats(self, num_colors, expected):
        # [FUNCTION DOC]
        # Verifies that a single float triple given as the color is wrapped into a list and
        # repeated once per requested color. Parametrized by count and expected list.
        color = (0.1, 0.2, 0.3)
        result = get_standard_colors(color=color, num_colors=num_colors)
        assert result == expected

    @pytest.mark.parametrize(
        "color, num_colors, expected",
        [
            ("Crimson", 1, ["Crimson"]),
            ("DodgerBlue", 2, ["DodgerBlue", "DodgerBlue"]),
            ("firebrick", 3, ["firebrick", "firebrick", "firebrick"]),
        ],
    )
    def test_user_input_named_color_string(self, color, num_colors, expected):
        # [FUNCTION DOC]
        # Verifies that a single named-color string (not a letter sequence) is repeated once
        # per requested color. Parametrized by the name, count and expected list.
        result = get_standard_colors(color=color, num_colors=num_colors)
        assert result == expected

    @pytest.mark.parametrize("color", ["", [], (), Series([], dtype="object")])
    def test_empty_color_raises(self, color):
        # [FUNCTION DOC]
        # Verifies that empty color arguments (empty string, empty list, empty tuple, empty
        # series) raise a value error whose message mentions an invalid color argument.
        with pytest.raises(ValueError, match="Invalid color argument"):
            get_standard_colors(color=color, num_colors=1)

    @pytest.mark.parametrize(
        "color",
        [
            "bad_color",
            ("red", "green", "bad_color"),
            (0.1,),
            (0.1, 0.2),
            (0.1, 0.2, 0.3, 0.4, 0.5),  # must be either 3 or 4 floats
        ],
    )
    def test_bad_color_raises(self, color):
        # [FUNCTION DOC]
        # Verifies that invalid colors (unknown name, a sequence containing an unknown name,
        # float tuples of wrong length) raise a value error with a message mentioning an
        # invalid color. Only 3 or 4 floats are valid in a tuple.
        with pytest.raises(ValueError, match="Invalid color"):
            get_standard_colors(color=color, num_colors=5)
