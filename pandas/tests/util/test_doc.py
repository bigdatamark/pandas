# [MODULE DOC]
# Tests for the docstring templating decorator: formatting a docstring template with keyword values, appending extra docstring fragments, taking the template from another function, and chaining templates. Defines decorated helper functions at import time; their docstrings are what the tests check.

from textwrap import dedent

from pandas.util._decorators import doc


@doc(method="cumsum", operation="sum")
def cumsum(whatever):
    # [FUNCTION DOC]
    # HELPER: function (body is only a docstring) decorated with template values method=cumsum and operation=sum. Its docstring is the base template with two placeholders; the parameter is unused.
    """
    This is the {method} method.

    It computes the cumulative {operation}.
    """


@doc(
    cumsum,
    dedent(
        """
        Examples
        --------

        >>> cumavg([1, 2, 3])
        2
        """
    ),
    method="cumavg",
    operation="average",
)
def cumavg(whatever):
    # [FUNCTION DOC]
    # HELPER: function decorated to reuse the first function's docstring template, append an examples section, and fill in cumavg and average. Does nothing.
    pass


@doc(cumsum, method="cummax", operation="maximum")
def cummax(whatever):
    # [FUNCTION DOC]
    # HELPER: function decorated to reuse the first function's template with maximum values. Does nothing.
    pass


@doc(cummax, method="cummin", operation="minimum")
def cummin(whatever):
    # [FUNCTION DOC]
    # HELPER: function decorated to reuse the previous function's (already formatted) template with minimum values. Does nothing.
    pass


def test_docstring_formatting():
    # [FUNCTION DOC]
    # The first helper's docstring equals the dedented text with the cumsum method and cumulative sum filled in.
    docstr = dedent(
        """
        This is the cumsum method.

        It computes the cumulative sum.
        """
    )
    assert cumsum.__doc__ == docstr


def test_docstring_appending():
    # [FUNCTION DOC]
    # The second helper's docstring equals the template filled with cumavg and average followed by the examples section.
    docstr = dedent(
        """
        This is the cumavg method.

        It computes the cumulative average.

        Examples
        --------

        >>> cumavg([1, 2, 3])
        2
        """
    )
    assert cumavg.__doc__ == docstr


def test_doc_template_from_func():
    # [FUNCTION DOC]
    # The third helper's docstring equals the template filled in for the maximum method.
    docstr = dedent(
        """
        This is the cummax method.

        It computes the cumulative maximum.
        """
    )
    assert cummax.__doc__ == docstr


def test_inherit_doc_template():
    # [FUNCTION DOC]
    # The fourth helper (whose template comes from a function that itself used a template) has a docstring filled in for the minimum method.
    docstr = dedent(
        """
        This is the cummin method.

        It computes the cumulative minimum.
        """
    )
    assert cummin.__doc__ == docstr
