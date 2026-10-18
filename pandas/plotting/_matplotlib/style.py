# [MODULE DOC]
# HELPER: internal color-selection logic for the matplotlib plotting backend.
# Public entry point is get_standard_colors (with typing overloads), which turns a colormap, a color
# specification or a color type into a list (or dict) of colors. Private helpers derive colors from a
# colormap, a single color or a sequence of colors, or a color type (default property cycle or random),
# validate colors, and cycle the result. Used by the core, misc and hist plotting modules.
# Imports matplotlib and numpy at import time; no other side effects (a warning may be emitted at call time).

from __future__ import annotations

from collections.abc import (
    Collection,
    Iterator,
    Sequence,
)
import itertools
from typing import (
    TYPE_CHECKING,
    cast,
    overload,
)
import warnings

import matplotlib as mpl
import matplotlib.colors
import numpy as np

from pandas._typing import MatplotlibColor as Color
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import is_list_like

if TYPE_CHECKING:
    from matplotlib.colors import Colormap


@overload
def get_standard_colors(
    num_colors: int,
    colormap: Colormap | None = ...,
    color_type: str = ...,
    *,
    color: dict[str, Color],
) -> dict[str, Color]: ...


@overload
def get_standard_colors(
    num_colors: int,
    colormap: Colormap | None = ...,
    color_type: str = ...,
    *,
    color: Color | Sequence[Color] | None = ...,
) -> list[Color]: ...


@overload
def get_standard_colors(
    num_colors: int,
    colormap: Colormap | None = ...,
    color_type: str = ...,
    *,
    color: dict[str, Color] | Color | Sequence[Color] | None = ...,
) -> dict[str, Color] | list[Color]: ...


def get_standard_colors(
    num_colors: int,
    colormap: Colormap | None = None,
    color_type: str = "default",
    *,
    color: dict[str, Color] | Color | Sequence[Color] | None = None,
) -> dict[str, Color] | list[Color]:
    # [FUNCTION DOC]
    # Returns the colors to use for a plot. num_colors is the number of colors needed (ignored for dict colors);
    # colormap is an optional matplotlib colormap (or its name); color_type is the default or random mode used when
    # neither color nor colormap is given; color is a dict, a single color, or a sequence of colors (keyword-only).
    # If color is a dict it is returned as is. Otherwise colors are derived (color wins over colormap with a user warning
    # when both are given) and then cycled so that the result has at least num_colors entries (it is the maximum of
    # num_colors and the number of derived colors long). Returns a list. Raises value errors from the helpers for
    # invalid input.
    # [FUNCTION DOC]
    # Typing overload stub for get_standard_colors with the full color union: returns a dict or a list. See the
    # implementation below.
    # [FUNCTION DOC]
    # Typing overload stub for get_standard_colors when color is a single color, a sequence of colors or none: returns
    # a list. See the implementation below.
    # [FUNCTION DOC]
    # Typing overload stub for get_standard_colors when color is a dict: returns a dict. See the implementation below.
    """
    Get standard colors based on `colormap`, `color_type` or `color` inputs.

    Parameters
    ----------
    num_colors : int
        Minimum number of colors to be returned.
        Ignored if `color` is a dictionary.
    colormap : :py:class:`matplotlib.colors.Colormap`, optional
        Matplotlib colormap.
        When provided, the resulting colors will be derived from the colormap.
    color_type : {"default", "random"}, optional
        Type of colors to derive. Used if provided `color` and `colormap` are None.
        Ignored if either `color` or `colormap` are not None.
    color : dict or str or sequence, optional
        Color(s) to be used for deriving sequence of colors.
        Can be either be a dictionary, or a single color (single color string,
        or sequence of floats representing a single color),
        or a sequence of colors.

    Returns
    -------
    dict or list
        Standard colors. Can either be a mapping if `color` was a dictionary,
        or a list of colors with a length of `num_colors` or more.

    Warns
    -----
    UserWarning
        If both `colormap` and `color` are provided.
        Parameter `color` will override.
    """
    if isinstance(color, dict):
        return color

    colors = _derive_colors(
        color=color,
        colormap=colormap,
        color_type=color_type,
        num_colors=num_colors,
    )

    return list(_cycle_colors(colors, num_colors=num_colors))


def _derive_colors(
    *,
    color: Color | Collection[Color] | None,
    colormap: str | Colormap | None,
    color_type: str,
    num_colors: int,
) -> list[Color]:
    # [FUNCTION DOC]
    # HELPER: derives a list of colors from the first applicable input. If color is none and a colormap is given, colors come
    # from the colormap; if a color is given they come from the color (and a user warning is emitted when a colormap is also
    # given, since the color takes precedence); otherwise they come from the color type. Keyword-only parameters:
    # color, colormap (name or object), color_type, num_colors. Returns a list of colors.
    """
    Derive colors from either `colormap`, `color_type` or `color` inputs.

    Get a list of colors either from `colormap`, or from `color`,
    or from `color_type` (if both `colormap` and `color` are None).

    Parameters
    ----------
    color : str or sequence, optional
        Color(s) to be used for deriving sequence of colors.
        Can be either be a single color (single color string, or sequence of floats
        representing a single color), or a sequence of colors.
    colormap : :py:class:`matplotlib.colors.Colormap`, optional
        Matplotlib colormap.
        When provided, the resulting colors will be derived from the colormap.
    color_type : {"default", "random"}, optional
        Type of colors to derive. Used if provided `color` and `colormap` are None.
        Ignored if either `color` or `colormap`` are not None.
    num_colors : int
        Number of colors to be extracted.

    Returns
    -------
    list
        List of colors extracted.

    Warns
    -----
    UserWarning
        If both `colormap` and `color` are provided.
        Parameter `color` will override.
    """
    if color is None and colormap is not None:
        return _get_colors_from_colormap(colormap, num_colors=num_colors)
    elif color is not None:
        if colormap is not None:
            warnings.warn(
                "'color' and 'colormap' cannot be used simultaneously. Using 'color'",
                stacklevel=find_stack_level(),
            )
        return _get_colors_from_color(color)
    else:
        return _get_colors_from_color_type(color_type, num_colors=num_colors)


def _cycle_colors(colors: list[Color], num_colors: int) -> Iterator[Color]:
    # [FUNCTION DOC]
    # HELPER: generator yielding colors from the list, repeated cyclically, until the larger of num_colors and the list
    # length is reached. Extra colors are harmless because matplotlib ignores unused ones.
    """Cycle colors until achieving max of `num_colors` or length of `colors`.

    Extra colors will be ignored by matplotlib if there are more colors
    than needed and nothing needs to be done here.
    """
    max_colors = max(num_colors, len(colors))
    yield from itertools.islice(itertools.cycle(colors), max_colors)


def _get_colors_from_colormap(
    colormap: str | Colormap,
    num_colors: int,
) -> list[Color]:
    # [FUNCTION DOC]
    # HELPER: samples num_colors evenly spaced colors from the colormap (a name or object, resolved by the next helper)
    # across the range zero to one and returns them as a list.
    """Get colors from colormap."""
    cmap = _get_cmap_instance(colormap)
    return [cmap(num) for num in np.linspace(0, 1, num=num_colors)]


def _get_cmap_instance(colormap: str | Colormap) -> Colormap:
    # [FUNCTION DOC]
    # HELPER: resolves a colormap given as a name to the registered matplotlib colormap, or returns a colormap object
    # unchanged. Unknown names are intended to raise a value error that the colormap is not recognized.
    # KNOWN ISSUE: the lookup in the matplotlib colormap registry raises a key error for unknown names rather than returning none, so the check for none (and its value error) is effectively unreachable.
    """Get instance of matplotlib colormap."""
    if isinstance(colormap, str):
        cmap = colormap
        colormap = mpl.colormaps[colormap]
        if colormap is None:
            raise ValueError(f"Colormap {cmap} is not recognized")
    return colormap


def _get_colors_from_color(
    color: Color | Collection[Color],
) -> list[Color]:
    # [FUNCTION DOC]
    # HELPER: converts a user color argument to a list of colors. An empty argument raises a value error (invalid color
    # argument). A single color (see the single-color helper) becomes a one-element list; otherwise each element is validated
    # via the iterable helper and returned in a list.
    """Get colors from user input color."""
    if len(color) == 0:
        raise ValueError(f"Invalid color argument: {color}")

    if _is_single_color(color):
        color = cast(Color, color)
        return [color]

    color = cast(Collection[Color], color)
    return list(_gen_list_of_colors_from_iterable(color))


def _is_single_color(color: Color | Collection[Color]) -> bool:
    # [FUNCTION DOC]
    # HELPER: returns true when color is a single color: a named or alias color string accepted by matplotlib (GH #36972)
    # or a sequence of three or four numbers; false otherwise.
    """Check if `color` is a single color, not a sequence of colors.

    Single color is of these kinds:
        - Named color "red", "C0", "firebrick"
        - Alias "g"
        - Sequence of floats, such as (0.1, 0.2, 0.3) or (0.1, 0.2, 0.3, 0.4).

    See Also
    --------
    _is_single_string_color
    """
    if isinstance(color, str) and _is_single_string_color(color):
        # GH #36972
        return True

    if _is_floats_color(color):
        return True

    return False


def _gen_list_of_colors_from_iterable(color: Collection[Color]) -> Iterator[Color]:
    # [FUNCTION DOC]
    # HELPER: generator that yields each element of a collection of colors, raising a value error naming the first element
    # that is not a single color.
    """
    Yield colors from string of several letters or from collection of colors.
    """
    for x in color:
        if _is_single_color(x):
            yield x
        else:
            raise ValueError(f"Invalid color {x}")


def _is_floats_color(color: Color | Collection[Color]) -> bool:
    # [FUNCTION DOC]
    # HELPER: returns true if color is list-like of length three or four with every element an int or float (an RGB or
    # RGBA tuple).
    """Check if color comprises a sequence of floats representing color."""
    return bool(
        is_list_like(color)
        and (len(color) == 3 or len(color) == 4)
        and all(isinstance(x, (int, float)) for x in color)
    )


def _get_colors_from_color_type(color_type: str, num_colors: int) -> list[Color]:
    # [FUNCTION DOC]
    # HELPER: gets colors from a color type. For default, returns the colors of matplotlib's property cycle, at most
    # num_colors of them (and at most the cycle length). For random, returns num_colors random RGB triples as lists,
    # generated by a random generator seeded with num_colors, so the result is deterministic for a given count.
    # Any other value raises a value error saying color_type must be default or random.
    """Get colors from user input color type."""
    if color_type == "default":
        prop_cycle = mpl.rcParams["axes.prop_cycle"]
        return [
            c["color"]
            for c in itertools.islice(prop_cycle, min(num_colors, len(prop_cycle)))
        ]
    elif color_type == "random":
        return np.random.default_rng(num_colors).random((num_colors, 3)).tolist()
    else:
        raise ValueError("color_type must be either 'default' or 'random'")


def _is_single_string_color(color: Color) -> bool:
    # [FUNCTION DOC]
    # HELPER: returns true if the string or value can be converted to RGBA by matplotlib's color converter, false if the
    # conversion raises a value error.
    """Check if `color` is a single string color.

    Examples of single string colors:
        - 'r'
        - 'g'
        - 'red'
        - 'green'
        - 'C3'
        - 'firebrick'

    Parameters
    ----------
    color : Color
        Color string or sequence of floats.

    Returns
    -------
    bool
        True if `color` looks like a valid color.
        False otherwise.
    """
    conv = matplotlib.colors.ColorConverter()
    try:
        # error: Argument 1 to "to_rgba" of "ColorConverter" has incompatible type
        # "str | Sequence[float]"; expected "tuple[float, float, float] | ..."
        conv.to_rgba(color)  # type: ignore[arg-type]
    except ValueError:
        return False
    else:
        return True
