#!/usr/bin/env python3
"""
Simple static site generator for the pandas web.

pandas_web.py takes a directory as parameter, and copies all the files into the
target directory after converting markdown files into html and rendering both
markdown and html files with a context. The context is obtained by parsing
the file ``config.yml`` in the root of the source directory.

The file should contain:
```
main:
  template_path: <path_to_the_jinja2_templates_directory>
  base_template: <template_file_all_other_files_will_extend>
  ignore:
  - <list_of_files_in_the_source_that_will_not_be_copied>
  github_repo_url: <organization/repo-name>
  context_preprocessors:
  - <list_of_functions_that_will_enrich_the_context_parsed_in_this_file>
  markdown_extensions:
  - <list_of_markdown_extensions_that_will_be_loaded>
```

The rest of the items in the file will be added directly to the context.
"""

# [MODULE DOC]
# Simple static site generator for the pandas website. Reads a source directory containing a configuration file
# (YAML) that supplies the template rendering context, runs configured context preprocessors that enrich the
# context, then copies every non-ignored file to a target directory; markdown and HTML files are rendered with
# Jinja2 using the context (markdown is converted to HTML first, with a table of contents and bootstrap table classes,
# and wrapped in the configured base template). Public contents: a Preprocessors class of built-in context
# enrichers (current year, navigation bar info, blog posts from files and feeds, maintainers info from the
# GitHub API, release list, enhancement proposal lists), helpers to resolve a callable from a dotted string, load
# the context, list source files and wrap documents in the base template, and the main function used by the
# command line (source path, optional target path defaulting to a build folder).
# Import-time side effects: imports third-party packages (feedparser, jinja2, markdown, packaging, requests,
# yaml) and reads the GitHub token environment variable to build a module-level authorization header dictionary
# (empty when no token).
import argparse
import collections
import datetime
import importlib
import itertools
import json
import operator
import os
import pathlib
import re
import shutil
import sys
import time
import typing

import feedparser
import jinja2
import markdown
from markdown.extensions.toc import TocExtension
from packaging import version
import requests
import yaml

api_token = os.environ.get("GITHUB_TOKEN")
if api_token is not None:
    GITHUB_API_HEADERS = {"Authorization": f"Bearer {api_token}"}
else:
    GITHUB_API_HEADERS = {}


class Preprocessors:
    # [CLASS DOC]
    # Namespace (never instantiated) of static methods that act as built-in context preprocessors: each takes the context
    # dictionary and returns it enriched. The names of these methods are referenced from the configuration file. Several
    # methods call the GitHub API with a short timeout and, on quota exhaustion (forbidden status), fall back to JSON
    # snapshots saved with the production website, and save fresh data into the target folder for future fallback.
    """
    Built-in context preprocessors.

    Context preprocessors are functions that receive the context used to
    render the templates, and enriches it with additional information.

    The original context is obtained by parsing ``config.yml``, and
    anything else needed just be added with context preprocessors.
    """

    @staticmethod
    def current_year(context):
        # [FUNCTION DOC]
        # Static. Adds the current year as an integer under the current-year key. Returns the context.
        """
        Add the current year to the context, so it can be used for the copyright
        note, or other places where it is needed.
        """
        context["current_year"] = datetime.datetime.now().year
        return context

    @staticmethod
    def navbar_add_info(context):
        # [FUNCTION DOC]
        # Static. For each navigation bar item, replaces it with a copy extended by a flag telling whether its target is a list
        # (a dropdown with subitems) and a slug (name with spaces replaced by hyphens, lowercased) to use as a style identifier.
        # Returns the context.
        """
        Items in the main navigation bar can be direct links, or dropdowns with
        subitems. This context preprocessor adds a boolean field
        ``has_subitems`` that tells which one of them every element is. It
        also adds a ``slug`` field to be used as a CSS id.
        """
        for i, item in enumerate(context["navbar"]):
            context["navbar"][i] = dict(
                item,
                has_subitems=isinstance(item["target"], list),
                slug=(item["name"].replace(" ", "-").lower()),
            )
        return context

    @staticmethod
    def blog_add_posts(context):
        # [FUNCTION DOC]
        # Static. Builds the list of blog posts. Reads markdown files from the configured posts folder (if set), skipping index files; each
        # is converted with the configured markdown extensions, its metadata title and date are read, and a summary of
        # about 30 words following the title is extracted from the tag-stripped text (a value error is raised if the title text is
        # not found in the body). Then reads every configured feed address, collecting entries. Posts are sorted newest first, truncated to
        # the configured number, and stored under the posts key of the blog section. Returns the context. Requires network access for feeds.
        """
        Given the blog feed defined in the configuration yaml, this context
        preprocessor fetches the posts in the feeds, and returns the relevant
        information for them (sorted from newest to oldest).
        """
        tag_expr = re.compile("<.*?>")
        posts = []
        # posts from the file system
        if context["blog"]["posts_path"]:
            posts_path = context["source_path"] / context["blog"]["posts_path"]
            for fname in posts_path.iterdir():
                if fname.name.startswith("index."):
                    continue
                link = f"/{context['blog']['posts_path']}/{fname.stem}.html"
                md = markdown.Markdown(
                    extensions=context["main"]["markdown_extensions"]
                )
                with fname.open(encoding="utf-8") as f:
                    html = md.convert(f.read())
                title = md.Meta["title"][0]
                summary = re.sub(tag_expr, "", html)
                try:
                    body_position = summary.index(title) + len(title)
                except ValueError as err:
                    raise ValueError(
                        f'Blog post "{fname}" should have a markdown header '
                        f'corresponding to its "Title" element "{title}"'
                    ) from err
                summary = " ".join(summary[body_position:].split(" ")[:30])
                posts.append(
                    {
                        "title": title,
                        "author": context["blog"]["author"],
                        "published": datetime.datetime.strptime(
                            md.Meta["date"][0], "%Y-%m-%d"
                        ),
                        "feed": context["blog"]["feed_name"],
                        "link": link,
                        "description": summary,
                        "summary": summary,
                    }
                )
        # posts from rss feeds
        for feed_url in context["blog"]["feed"]:
            feed_data = feedparser.parse(feed_url)
            for entry in feed_data.entries:
                published = datetime.datetime.fromtimestamp(
                    time.mktime(entry.published_parsed)
                )
                summary = re.sub(tag_expr, "", entry.summary)
                posts.append(
                    {
                        "title": entry.title,
                        "author": entry.author,
                        "published": published,
                        "feed": feed_data["feed"]["title"],
                        "link": entry.link,
                        "description": entry.description,
                        "summary": summary,
                    }
                )
        posts.sort(key=operator.itemgetter("published"), reverse=True)
        context["blog"]["posts"] = posts[: context["blog"]["num_posts"]]
        return context

    @staticmethod
    def maintainers_add_info(context):
        # [FUNCTION DOC]
        # Static. Fetches GitHub user information for all active, inactive and stub maintainers. Raises a value error if a
        # person is listed as both active and inactive. A forbidden response triggers use of the saved snapshot from the
        # production site (and stops fetching); other HTTP errors raise. Stores the result under the github info key of the
        # maintainers section and writes it to a JSON file in the target folder. Returns the context.
        """
        Given the active maintainers defined in the yaml file, it fetches
        the GitHub user information for them.
        """
        repeated = set(context["maintainers"]["active"]) & set(
            context["maintainers"]["inactive"]
        )
        if repeated:
            raise ValueError(f"Maintainers {repeated} are both active and inactive")

        maintainers_info = {}
        for user in (
            context["maintainers"]["active"]
            + context["maintainers"]["inactive"]
            + context["maintainers"]["pandasstubs"]
        ):
            resp = requests.get(
                f"https://api.github.com/users/{user}",
                headers=GITHUB_API_HEADERS,
                timeout=5,
            )
            if resp.status_code == 403:
                sys.stderr.write(
                    "WARN: GitHub API quota exceeded when fetching maintainers\n"
                )
                # if we exceed github api quota, we use the github info
                # of maintainers saved with the website
                resp_bkp = requests.get(
                    context["main"]["production_url"] + "maintainers.json", timeout=5
                )
                resp_bkp.raise_for_status()
                maintainers_info = resp_bkp.json()
                break

            resp.raise_for_status()
            maintainers_info[user] = resp.json()

        context["maintainers"]["github_info"] = maintainers_info

        # save the data fetched from github to use it in case we exceed
        # git github api quota in the future
        with open(
            pathlib.Path(context["target_path"]) / "maintainers.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(maintainers_info, f)

        return context

    @staticmethod
    def home_add_releases(context):
        # [FUNCTION DOC]
        # Static, no docstring. Builds the list of released versions. Fetches the repository releases from GitHub (with snapshot
        # fallback on a forbidden status), saves them as JSON in the target folder, skips prereleases, and records for each release a name
        # (tag without a leading v), parsed version, tag, publication time and first asset download address (empty if none).
        # Then groups consecutive releases by major and minor version, keeps one per group, and sorts the final list from newest
        # to oldest. Stores it under the releases key. Returns the context.
        # KNOWN ISSUE: the grouping assumes the releases arrive sorted by version (consecutive groups), and the selection inside each
        # group uses the minor number, which is identical within a group, so in effect the first entry of each group (the newest, if the
        # service lists newest first) is kept rather than the highest patch release.
        context["releases"] = []

        github_repo_url = context["main"]["github_repo_url"]
        resp = requests.get(
            f"https://api.github.com/repos/{github_repo_url}/releases",
            headers=GITHUB_API_HEADERS,
            timeout=5,
        )
        if resp.status_code == 403:
            sys.stderr.write("WARN: GitHub API quota exceeded when fetching releases\n")
            resp_bkp = requests.get(
                context["main"]["production_url"] + "releases.json", timeout=5
            )
            resp_bkp.raise_for_status()
            releases = resp_bkp.json()
        else:
            resp.raise_for_status()
            releases = resp.json()

        with open(
            pathlib.Path(context["target_path"]) / "releases.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(releases, f, default=datetime.datetime.isoformat)

        for release in releases:
            if release["prerelease"]:
                continue
            published = datetime.datetime.strptime(
                release["published_at"], "%Y-%m-%dT%H:%M:%SZ"
            )
            context["releases"].append(
                {
                    "name": release["tag_name"].lstrip("v"),
                    "parsed_version": version.parse(release["tag_name"].lstrip("v")),
                    "tag": release["tag_name"],
                    "published": published,
                    "url": (
                        release["assets"][0]["browser_download_url"]
                        if release["assets"]
                        else ""
                    ),
                }
            )
        # sorting out obsolete versions
        grouped_releases = itertools.groupby(
            context["releases"],
            key=lambda r: (r["parsed_version"].major, r["parsed_version"].minor),
        )
        context["releases"] = [
            max(release_group, key=lambda r: r["parsed_version"].minor)
            for _, release_group in grouped_releases
        ]
        # sorting releases by version number
        context["releases"].sort(key=lambda r: r["parsed_version"], reverse=True)
        return context

    @staticmethod
    def roadmap_pdeps(context):
        # [FUNCTION DOC]
        # Static. Builds the lists of enhancement proposals by status. Reads proposal markdown files in the configured folder
        # (sorted by name): the title is the first line without its heading marker, and the status is read from the line starting with
        # the status marker; an unknown or missing status raises a runtime error. Proposals are stored in a dictionary keyed by status with
        # title and page address. Then queries GitHub for open, non-draft pull requests labelled as proposals (with snapshot fallback
        # on a forbidden status), saves the answer to a JSON file in the target folder, and appends them, sorted by the number parsed from
        # their title, to the under-discussion list (a value error if a title lacks the proposal number prefix). Returns the context.
        """
        PDEP's (pandas enhancement proposals) are not part of the bar
        navigation. They are included as lists in the "Roadmap" page
        and linked from there. This preprocessor obtains the list of
        PDEP's in different status from the directory tree and GitHub.
        """
        KNOWN_STATUS = {
            "Draft",
            "Under discussion",
            "Accepted",
            "Implemented",
            "Rejected",
            "Withdrawn",
        }
        context["pdeps"] = collections.defaultdict(list)

        # accepted, rejected and implemented
        pdeps_path = (
            pathlib.Path(context["source_path"]) / context["roadmap"]["pdeps_path"]
        )
        for pdep in sorted(pdeps_path.iterdir()):
            if pdep.suffix != ".md":
                continue
            with pdep.open() as f:
                title = f.readline()[2:]  # removing markdown title "# "
                status = None
                for line in f:
                    if line.startswith("- Status: "):
                        status = line.strip().split(": ", 1)[1]
                        break
                if status not in KNOWN_STATUS:
                    raise RuntimeError(
                        f'PDEP "{pdep}" status "{status}" is unknown. '
                        f"Should be one of: {KNOWN_STATUS}"
                    )
            html_file = pdep.with_suffix(".html").name
            context["pdeps"][status].append(
                {
                    "title": title,
                    "url": f"pdeps/{html_file}",
                }
            )

        # under discussion
        github_repo_url = context["main"]["github_repo_url"]
        resp = requests.get(
            "https://api.github.com/search/issues?"
            f"q=is:pr is:open label:PDEP draft:false repo:{github_repo_url}",
            headers=GITHUB_API_HEADERS,
            timeout=5,
        )
        if resp.status_code == 403:
            sys.stderr.write("WARN: GitHub API quota exceeded when fetching pdeps\n")
            resp_bkp = requests.get(
                context["main"]["production_url"] + "pdeps.json", timeout=5
            )
            resp_bkp.raise_for_status()
            pdeps = resp_bkp.json()
        else:
            resp.raise_for_status()
            pdeps = resp.json()

        with open(
            pathlib.Path(context["target_path"]) / "pdeps.json", "w", encoding="utf-8"
        ) as f:
            json.dump(pdeps, f)

        compiled_pattern = re.compile(r"^PDEP-(\d+)")

        def sort_pdep(pdep: dict) -> int:
            # [FUNCTION DOC]
            # HELPER: sort key for pull request items. Input: pdep, an item with a title. Returns the integer after the leading
            # proposal prefix in the title; raises a value error with a hint about the expected title format if absent.
            title = pdep["title"]
            match = compiled_pattern.match(title)
            if not match:
                msg = f"""Could not find PDEP number in '{title}'. Please make sure to
                write the title as: 'PDEP-num: {title}'."""
                raise ValueError(msg)

            return int(match[1])

        context["pdeps"]["Under discussion"].extend(
            {"title": pdep["title"], "url": pdep["html_url"]}
            for pdep in sorted(pdeps["items"], key=sort_pdep)
        )

        return context


def get_callable(obj_as_str: str) -> object:
    # [FUNCTION DOC]
    # Resolves a dotted string to a Python object. Input: obj_as_str. Repeatedly tries to import the longest dotted prefix as a module,
    # peeling trailing components off as attribute names, then resolves those attributes in order. Returns the object.
    # KNOWN ISSUE: if no prefix can be imported as a module the loop ends without the module variable being assigned, so the
    # following emptiness check raises an unbound variable error instead of the intended import error.
    """
    Get a Python object from its string representation.

    For example, for ``sys.stdout.write`` would import the module ``sys``
    and return the ``write`` function.
    """
    components = obj_as_str.split(".")
    attrs = []
    while components:
        try:
            obj = importlib.import_module(".".join(components))
        except ImportError:
            attrs.insert(0, components.pop())
        else:
            break

    if not obj:
        raise ImportError(f'Could not import "{obj_as_str}"')

    for attr in attrs:
        obj = getattr(obj, attr)

    return obj


def get_context(config_fname: pathlib.Path, **kwargs):
    # [FUNCTION DOC]
    # Builds the template context. Inputs: config_fname, path of the YAML configuration; keyword arguments merged into the context
    # (for example the target path). Loads the configuration, adds the source directory path, applies the keyword arguments, then applies each
    # preprocessor named in the main section in order; each must return the context (assertion error naming the preprocessor if not).
    # Returns the context.
    """
    Load the config yaml as the base context, and enrich it with the
    information added by the context preprocessors defined in the file.
    """
    with config_fname.open(encoding="utf-8") as f:
        context = yaml.safe_load(f)

    context["source_path"] = config_fname.parent
    context.update(kwargs)

    preprocessors = (
        get_callable(context_prep)
        for context_prep in context["main"]["context_preprocessors"]
    )
    for preprocessor in preprocessors:
        context = preprocessor(context)
        msg = f"{preprocessor.__name__} is missing the return statement"
        assert context is not None, msg

    return context


def get_source_files(source_path: pathlib.Path) -> typing.Generator[str, None, None]:
    # [FUNCTION DOC]
    # Generator over all regular files below a directory, yielding each path relative to the directory.
    # KNOWN ISSUE: the declared yield type is text, but the yielded values are path objects.
    """
    Generate the list of files present in the source directory.
    """
    for path in source_path.rglob("*"):
        if path.is_file():
            yield path.relative_to(source_path)


def extend_base_template(content: str, base_template: str) -> str:
    # [FUNCTION DOC]
    # Wraps document content in a Jinja2 template inheritance statement. Inputs: content, document text; base_template, template
    # file name. Returns text that extends the base template and places the content in its body block.
    """
    Wrap document to extend the base template, before it is rendered with
    Jinja2.
    """
    result = '{% extends "' + base_template + '" %}'
    result += "{% block body %}"
    result += content
    result += "{% endblock %}"
    return result


def main(
    source_path: pathlib.Path,
    target_path: pathlib.Path,
) -> int:
    # [FUNCTION DOC]
    # Generates the site. Inputs: source_path and target_path, directories. First checks that the versions JSON file in the source is
    # valid JSON (runtime error otherwise). Deletes the target directory (ignoring errors) and recreates it. Builds the context, creates a
    # Jinja2 environment over the configured templates folder. For each source file not in the ignore list (compared as a posix path):
    # creates the target subfolder; for markdown or HTML files, reads the file, converts markdown to HTML with the configured
    # extensions plus a table of contents and adds bootstrap table classes and the base template, sets the base address as a relative
    # path with one parent step per directory level, renders with Jinja2 and writes a file with the HTML suffix; any other file is copied
    # unchanged. Progress is printed to standard error.
    # KNOWN ISSUE: the declared return type is integer, but nothing is returned, so the exit status given to the system exit call is always zero.
    """
    Copy every file in the source directory to the target directory.

    For ``.md`` and ``.html`` files, render them with the context
    before copying them. ``.md`` files are transformed to HTML.
    """

    # Sanity check: validate that versions.json is valid JSON
    versions_path = source_path / "versions.json"
    with versions_path.open(encoding="utf-8") as f:
        try:
            json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Invalid versions.json: {e}. Ensure it is valid JSON."
            ) from e

    config_fname = source_path / "config.yml"

    shutil.rmtree(target_path, ignore_errors=True)
    os.makedirs(target_path, exist_ok=True)

    sys.stderr.write("Generating context...\n")
    context = get_context(config_fname, target_path=target_path)
    sys.stderr.write("Context generated\n")

    templates_path = source_path / context["main"]["templates_path"]
    jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(templates_path))

    for fname in get_source_files(source_path):
        if fname.as_posix() in context["main"]["ignore"]:
            continue
        sys.stderr.write(f"Processing {fname}\n")
        dirname = fname.parent
        (target_path / dirname).mkdir(parents=True, exist_ok=True)

        extension = fname.suffix
        if extension in (".html", ".md"):
            with (source_path / fname).open(encoding="utf-8") as f:
                content = f.read()
            if extension == ".md":
                toc = TocExtension(
                    title="Table of Contents",
                    toc_depth="2-3",
                    permalink=" #",
                )
                body = markdown.markdown(
                    content, extensions=context["main"]["markdown_extensions"] + [toc]
                )
                # Apply Bootstrap's table formatting manually
                # Python-Markdown doesn't let us config table attributes by hand
                body = body.replace("<table>", '<table class="table table-bordered">')
                content = extend_base_template(body, context["main"]["base_template"])
            context["base_url"] = "../" * (len(fname.parents) - 1)
            content = jinja_env.from_string(content).render(**context)
            fname_html = fname.with_suffix(".html").name
            with (target_path / dirname / fname_html).open("w", encoding="utf-8") as f:
                f.write(content)
        else:
            shutil.copy(source_path / fname, target_path / fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Documentation builder.")
    parser.add_argument(
        "source_path", help="path to the source directory (must contain config.yml)"
    )
    parser.add_argument(
        "--target-path", default="build", help="directory where to write the output"
    )
    args = parser.parse_args()
    sys.exit(main(pathlib.Path(args.source_path), pathlib.Path(args.target_path)))
