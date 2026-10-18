#!/usr/bin/env python3
"""
Script to generate contributor and pull request lists

This script generates contributor and pull request lists for release
announcements using GitHub v3 protocol. Use requires an authentication token in
order to have sufficient bandwidth, you can get one following the directions at
`<https://help.github.com/articles/creating-an-access-token-for-command-line-use/>_
Don't add any scope, as the default is read access to public information. The
token may be stored in an environment variable as you only get one chance to
see it.

Usage::

    $ ./scripts/announce.py <token> <revision range>

The output is utf8 rst.

Dependencies
------------

- gitpython
- pygithub

Some code was copied from scipy `tools/gh_lists.py` and `tools/authors.py`.

Examples
--------

From the bash command line with $GITHUB token.

    $ ./scripts/announce.py $GITHUB v1.11.0..v1.11.1 > announce.rst

"""

# [MODULE DOC]
# Library and command line script producing the contributor list for a release
# announcement, from the git history of the repository that contains this file
# (two directory levels above it).
# Public contents: constants CONTRIBUTOR_MAPPING (old to new contributor
# names, stored rot13-encoded), UTF8Writer, this_repo (the repository object),
# author_msg and pull_request_msg (message templates); functions get_authors,
# get_pull_requests, build_components, build_string and main.
# The sibling contributors extension imports build_components from here.
# Import-time side effects: opens the git repository (fails if not in one).
# Run as a script it takes one argument, a revision range of the form old
# revision, two dots, new revision, and prints the rst text to standard output.
# KNOWN ISSUE: the module docstring describes a script taking an access token
# and mentions the GitHub library as a dependency, but the code never uses a
# token or that library and the command line accepts only the revision range;
# the pull request helper is unused by the main path.

import codecs
import os
import re
import textwrap

from git import Repo

# Contributors to be renamed.
CONTRIBUTOR_MAPPING = {"znkjnffrezna": "znetbgcubravk"}

UTF8Writer = codecs.getwriter("utf8")
this_repo = Repo(os.path.join(os.path.dirname(__file__), "..", ".."))

author_msg = """\
A total of %d people contributed patches to this release.  People with a
"+" by their names contributed a patch for the first time.
"""

pull_request_msg = """\
A total of %d pull requests were merged for this release.
"""


def get_authors(revision_range):
    # [FUNCTION DOC]
    # Returns the sorted list of author names for a revision range, with a
    # plus marker appended (preceded by a space) to authors who are new in the
    # range, i.e. did not appear in history up to the range start.
    # Input: a string of the form start revision, two dots, end revision
    # (whitespace around parts is stripped). The end part may be written as a
    # tag name, a vertical bar and the word for head: in that case, if the tag
    # exists in the repository it is used as the end, otherwise the head is
    # used (an assertion error is raised if the part after the bar is not
    # exactly the word for head; a value error if there is more than one bar).
    # Sources of names, collected for both the range and for all history up to
    # the start: commit message trailers naming co-authors (from backport
    # commits), plus the names listed by the short log summary. Names from
    # the first pass are matched by a pattern on the trailer; names from the
    # second by a pattern matching a tab then the name at line end.
    # The automated merge user is removed from both sets, then contributors
    # in the rename mapping (decoded from rot13) are replaced by their new name
    # when present. Runs git subprocesses through the repository object and may
    # raise the git command error for invalid revisions.
    # KNOWN ISSUE: the range start is used alone for the history of previous
    # releases, so authors who contributed only on other branches may be
    # reported as new; this is inherent to the approach, not an error.
    pat = "^.*\\t(.*)$"
    lst_release, cur_release = (r.strip() for r in revision_range.split(".."))

    if "|" in cur_release:
        # e.g. v1.0.1|HEAD
        maybe_tag, head = cur_release.split("|")
        assert head == "HEAD"
        if maybe_tag in this_repo.tags:
            cur_release = maybe_tag
        else:
            cur_release = head
        revision_range = f"{lst_release}..{cur_release}"

    # authors, in current release and previous to current release.
    # We need two passes over the log for cur and prev, one to get the
    # "Co-authored by" commits, which come from backports by the bot,
    # and one for regular commits.
    xpr = re.compile(r"Co-authored-by: (?P<name>[^<\n]+)\s")
    cur = set(
        xpr.findall(
            this_repo.git.log("--grep=Co-authored", "--pretty=%b", revision_range)
        )
    )
    cur |= set(re.findall(pat, this_repo.git.shortlog("-s", revision_range), re.M))

    pre = set(
        xpr.findall(this_repo.git.log("--grep=Co-authored", "--pretty=%b", lst_release))
    )
    pre |= set(re.findall(pat, this_repo.git.shortlog("-s", lst_release), re.M))

    # Homu is the author of auto merges, clean him out.
    cur.discard("Homu")
    pre.discard("Homu")

    # Rename contributors according to mapping.
    for old_name, new_name in CONTRIBUTOR_MAPPING.items():
        old_name_decoded = codecs.decode(old_name, "rot13")
        new_name_decoded = codecs.decode(new_name, "rot13")
        if old_name_decoded in pre:
            pre.discard(old_name_decoded)
            pre.add(new_name_decoded)
        if old_name_decoded in cur:
            cur.discard(old_name_decoded)
            cur.add(new_name_decoded)

    # Append '+' to new authors.
    authors = [s + " +" for s in cur - pre] + list(cur & pre)
    authors.sort()
    return authors


def get_pull_requests(repo, revision_range):
    # [FUNCTION DOC]
    # Returns the list of GitHub pull request objects merged in a revision
    # range, sorted by pull request number. Parameters: repo (a GitHub
    # repository client object providing a method to fetch a pull request by
    # number) and revision_range (string passed to git).
    # Numbers are found from the local repository history (not the client) in
    # three ways: merge commit subjects of the regular form; subjects of the
    # form used by the old automated merge bot; and trailing parenthesized
    # numbers in one-line subjects of first-parent non-merge commits (squash
    # merges). One network request per number is made through repo.
    # KNOWN ISSUE: duplicates are not removed, so a pull request matched by
    # more than one pattern would be fetched twice. This function is not called
    # anywhere in this module.
    prnums = []

    # From regular merges
    merges = this_repo.git.log("--oneline", "--merges", revision_range)
    issues = re.findall("Merge pull request \\#(\\d*)", merges)
    prnums.extend(int(s) for s in issues)

    # From Homu merges (Auto merges)
    issues = re.findall("Auto merge of \\#(\\d*)", merges)
    prnums.extend(int(s) for s in issues)

    # From fast forward squash-merges
    commits = this_repo.git.log(
        "--oneline", "--no-merges", "--first-parent", revision_range
    )
    issues = re.findall("^.*\\(\\#(\\d+)\\)$", commits, re.M)
    prnums.extend(int(s) for s in issues)

    # get PR data from GitHub repo
    prnums.sort()
    prs = [repo.get_pull(n) for n in prnums]
    return prs


def build_components(revision_range, heading="Contributors"):
    # [FUNCTION DOC]
    # Builds the pieces of the contributors section for a revision range.
    # Parameters: revision_range (string, see the authors function) and heading
    # (section title, default Contributors). Returns a mapping with three
    # entries: heading (the given title), author_message (the template text
    # with the number of authors filled in) and authors (the sorted list of
    # names from the authors function). Propagates errors from git.
    # KNOWN ISSUE: the unpacked start and end revision values are unused.
    lst_release, cur_release = (r.strip() for r in revision_range.split(".."))
    authors = get_authors(revision_range)

    return {
        "heading": heading,
        "author_message": author_msg % len(authors),
        "authors": authors,
    }


def build_string(revision_range, heading="Contributors"):
    # [FUNCTION DOC]
    # Renders the contributors section as reStructuredText text. Parameters as
    # in the components builder. Returns a string made of the heading, an
    # underline of equals signs as long as the heading, a blank line, the
    # author message, and a bullet list with one asterisk item per author on
    # its own line. Note the components mapping is modified: an underline entry
    # is added and the authors entry becomes the joined text. The template
    # must remain a format-method template (per the source comment, converting
    # it to an f-string breaks the formatting).
    components = build_components(revision_range, heading=heading)
    components["uline"] = "=" * len(components["heading"])
    components["authors"] = "* " + "\n* ".join(components["authors"])

    # Don't change this to an fstring. It breaks the formatting.
    tpl = textwrap.dedent(
        """\
    {heading}
    {uline}

    {author_message}
    {authors}"""
    ).format(**components)
    return tpl


def main(revision_range):
    # [FUNCTION DOC]
    # Prints the contributors section text for the revision range to standard
    # output. Parameter revision_range is the string for the range. Returns
    # nothing.
    # document authors
    text = build_string(revision_range)
    print(text)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Generate author lists for release")
    parser.add_argument("revision_range", help="<revision>..<revision>")
    args = parser.parse_args()
    main(args.revision_range)
