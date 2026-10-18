"""Sphinx extension for listing code contributors to a release.

Usage::

   .. contributors:: v0.23.0..v0.23.1

This will be replaced with a message indicating the number of
code contributors and commits, and then list each contributor
individually. For development versions (before a tag is available)
use::

    .. contributors:: v0.23.0..v0.23.1|HEAD

While the v0.23.1 tag does not exist, that will use the HEAD of the
branch as the end of the revision range.
"""

# [MODULE DOC]
# Sphinx extension module defining the contributors directive for the release
# notes. Public contents: class ContributorsDirective and the extension entry
# function setup that Sphinx calls to register it. Relationships: imports
# build_components from the sibling announce module (which must be importable,
# i.e. this folder is on the module search path, and which opens the git
# repository at import time).
# KNOWN ISSUE: the module docstring says the directive output mentions the
# number of commits, but the code only reports the author message with the
# count of contributors.

from announce import build_components
from docutils import nodes
from docutils.parsers.rst import Directive
import git


class ContributorsDirective(Directive):
    # [CLASS DOC]
    # Docutils directive named contributors, subclass of the base directive.
    # Class attributes: required_arguments (one: the revision range) and name.
    # The single method run produces the document nodes.
    # KNOWN ISSUE: the name attribute is informational only; the name actually
    # used is the one passed when registering the directive in setup.
    required_arguments = 1
    name = "contributors"

    def run(self):
        # [FUNCTION DOC]
        # Produces the nodes for the directive. The first argument is the
        # revision range string. If it ends with the placeholder for an
        # unreleased version (an x then two dots then the word for head),
        # returns an empty paragraph and empty bullet list. Otherwise builds the
        # components via the announce helper; on a git command error returns a
        # single warning node reporting the range and error, with the directive's
        # line number. On success returns a paragraph holding the author
        # message followed by a bullet list with one item per author name.
        range_ = self.arguments[0]
        if range_.endswith("x..HEAD"):
            return [nodes.paragraph(), nodes.bullet_list()]
        try:
            components = build_components(range_)
        except git.GitCommandError as exc:
            return [
                self.state.document.reporter.warning(
                    f"Cannot find contributors for range {repr(range_)}: {exc}",
                    line=self.lineno,
                )
            ]
        else:
            message = nodes.paragraph()
            message += nodes.Text(components["author_message"])

            listnode = nodes.bullet_list()

            for author in components["authors"]:
                para = nodes.paragraph()
                para += nodes.Text(author)
                listnode += nodes.list_item("", para)

        return [message, listnode]


def setup(app):
    # [FUNCTION DOC]
    # Sphinx extension entry point. Registers the directive class under the
    # name contributors on the application object. Returns the extension
    # metadata mapping: version 0.1, safe for parallel reading and writing.
    app.add_directive("contributors", ContributorsDirective)

    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
