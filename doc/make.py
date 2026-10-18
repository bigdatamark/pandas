#!/usr/bin/env python3
"""
Python script for building documentation.

To build the docs you must have all optional dependencies for pandas
installed. See the installation instructions for a list of these.

Usage
-----
    $ python make.py clean
    $ python make.py html
    $ python make.py latex
"""

# [MODULE DOC]
# Command line driver for building the pandas documentation with Sphinx.
# Public contents: the class DocBuilder, whose public methods (html, latex,
# latex_forced, clean, zip_html, linkcheck) are exposed as sub-commands, and the
# function main which parses arguments and runs the chosen command. Module
# constants: DOC_PATH (directory of this script), SOURCE_PATH (source folder),
# BUILD_PATH (build folder) and REDIRECTS_FILE (csv of page moves).
# Run as a script it exits with the numeric result of the chosen command.
# Import-time side effects: imports the docutils packages (requires them to be
# installed); no other effects. The name for the pandas package is NOT imported
# at module level: main injects it into the module globals at run time, and
# DocBuilder relies on that.
# Environment side effects (in main and the builder constructor): sets the
# module search path variable, the matplotlib backend variable (to the
# non-interactive Agg backend) and the pattern variable that tells the Sphinx
# configuration which pages to include.

import argparse
import csv
import importlib
import os
import shutil
import subprocess
import sys
import webbrowser

import docutils
import docutils.parsers.rst

DOC_PATH = os.path.dirname(os.path.abspath(__file__))
SOURCE_PATH = os.path.join(DOC_PATH, "source")
BUILD_PATH = os.path.join(DOC_PATH, "build")
REDIRECTS_FILE = os.path.join(DOC_PATH, "redirects.csv")


class DocBuilder:
    # [CLASS DOC]
    # Wraps the documentation build commands. Constructor state: num_jobs (the
    # parallel job count passed to Sphinx, a string, default "auto"),
    # include_api (whether API reference pages are built), whatsnew (build only
    # the release notes), verbosity (count of verbose flags), warnings_are_errors,
    # no_browser (do not open a browser) and single_doc_html (the output page of
    # a single-document build, or none). Public methods (those not starting with
    # an underscore) are the commands the script accepts; main discovers them by
    # listing attribute names, so adding a public method adds a command.
    """
    Class to wrap the different commands of this script.

    All public methods of this class can be called as parameters of the
    script.
    """

    def __init__(
        self,
        num_jobs="auto",
        include_api=True,
        whatsnew=False,
        single_doc=None,
        verbosity=0,
        warnings_are_errors=False,
        no_browser=False,
    ) -> None:
        # [FUNCTION DOC]
        # Stores the options on the instance and configures which pages Sphinx
        # builds by setting the pattern environment variable (global side effect):
        # if single_doc is given it is first validated/normalized by the
        # private single-doc processor (which may raise) and the pattern is set
        # to it; otherwise if include_api is false the pattern is set to the
        # marker that excludes the API; otherwise if whatsnew is true the
        # pattern is set to the release-notes name. Also computes the output
        # page of a single-document build: for a path ending in the rst
        # extension, the same path with an html extension; for an object name,
        # the generated API page for that object; otherwise none.
        # Parameters: num_jobs (string or number of parallel jobs), include_api
        # (bool), whatsnew (bool), single_doc (optional path or object name),
        # verbosity (int), warnings_are_errors (bool), no_browser (bool).
        self.num_jobs = num_jobs
        self.include_api = include_api
        self.whatsnew = whatsnew
        self.verbosity = verbosity
        self.warnings_are_errors = warnings_are_errors
        self.no_browser = no_browser

        if single_doc:
            single_doc = self._process_single_doc(single_doc)
            os.environ["SPHINX_PATTERN"] = single_doc
        elif not include_api:
            os.environ["SPHINX_PATTERN"] = "-api"
        elif whatsnew:
            os.environ["SPHINX_PATTERN"] = "whatsnew"

        self.single_doc_html = None
        if single_doc and single_doc.endswith(".rst"):
            self.single_doc_html = os.path.splitext(single_doc)[0] + ".html"
        elif single_doc:
            self.single_doc_html = f"reference/api/pandas.{single_doc}.html"

    def _process_single_doc(self, single_doc):
        # [FUNCTION DOC]
        # HELPER: validates the single-document option for the constructor.
        # Input: a string that is either a file name (rst or notebook extension)
        # relative to the source folder, or a dotted pandas object name starting
        # with the package prefix.
        # Returns: the file name unchanged for existing files; for object names,
        # the name with the package prefix removed, after confirming the object
        # can be reached by successive attribute lookup starting from the
        # package global injected by main.
        # Raises: file-not-found error if a named file does not exist; import
        # error if an attribute lookup fails; value error for any other input.
        # KNOWN ISSUE: relies on a global name for the package that is only
        # defined when main has run, so using the class from another module
        # raises a name error for object names. The first unpacked value
        # (the base name) is unused.
        """
        Make sure the provided value for --single is a path to an existing
        .rst/.ipynb file, or a pandas object that can be imported.

        For example, categorial.rst or pandas.DataFrame.head. For the latter,
        return the corresponding file path
        (e.g. reference/api/pandas.DataFrame.head.rst).
        """
        base_name, extension = os.path.splitext(single_doc)
        if extension in (".rst", ".ipynb"):
            if os.path.exists(os.path.join(SOURCE_PATH, single_doc)):
                return single_doc
            else:
                raise FileNotFoundError(f"File {single_doc} not found")

        elif single_doc.startswith("pandas."):
            try:
                obj = pandas  # noqa: F821
                for name in single_doc.split("."):
                    obj = getattr(obj, name)
            except AttributeError as err:
                raise ImportError(f"Could not import {single_doc}") from err
            else:
                return single_doc[len("pandas.") :]
        else:
            raise ValueError(
                f"--single={single_doc} not understood. "
                "Value should be a valid path to a .rst or .ipynb file, "
                "or a valid pandas object "
                "(e.g. categorical.rst or pandas.DataFrame.head)"
            )

    @staticmethod
    def _run_os(*args) -> None:
        # [FUNCTION DOC]
        # HELPER: runs an external command and waits for it. Takes the command
        # and its arguments as separate positional strings. Output and error
        # streams are forwarded to this process's streams. Raises the process
        # error from the subprocess module if the command exits nonzero.
        # Returns nothing.
        """
        Execute a command as an OS terminal.

        Parameters
        ----------
        *args : list of str
            Command and parameters to be executed

        Examples
        --------
        >>> DocBuilder()._run_os("python", "--version")
        """
        subprocess.check_call(args, stdout=sys.stdout, stderr=sys.stderr)

    def _sphinx_build(self, kind: str):
        # [FUNCTION DOC]
        # HELPER: runs the Sphinx build command for one builder kind.
        # Parameter kind must be html, latex or linkcheck, otherwise a value
        # error is raised. Command line assembled: builder flag; the job-count
        # flag if num_jobs is truthy; warnings-as-errors with keep-going if
        # enabled; a repeated-v verbosity flag with as many v letters as the
        # verbosity count; the doctree cache folder inside the build folder; the
        # source folder; and the output folder named after the kind inside the
        # build folder. Returns the process exit code (does not raise for build
        # failure). Side effect: runs a subprocess and writes the build tree.
        """
        Call sphinx to build documentation.

        Attribute `num_jobs` from the class is used.

        Parameters
        ----------
        kind : {'html', 'latex', 'linkcheck'}

        Examples
        --------
        >>> DocBuilder(num_jobs=4)._sphinx_build("html")
        """
        if kind not in ("html", "latex", "linkcheck"):
            raise ValueError(f"kind must be html, latex or linkcheck, not {kind}")

        cmd = ["sphinx-build", "-b", kind]
        if self.num_jobs:
            cmd += ["-j", self.num_jobs]
        if self.warnings_are_errors:
            cmd += ["-W", "--keep-going"]
        if self.verbosity:
            cmd.append(f"-{'v' * self.verbosity}")
        cmd += [
            "-d",
            os.path.join(BUILD_PATH, "doctrees"),
            SOURCE_PATH,
            os.path.join(BUILD_PATH, kind),
        ]
        return subprocess.call(cmd)

    def _open_browser(self, single_doc_html) -> None:
        # [FUNCTION DOC]
        # HELPER: opens the built page at the given path (relative to the html
        # build folder) in a new browser tab, via a local file address.
        # Side effect: launches the default browser. Returns nothing.
        # KNOWN ISSUE: the existing docstring is truncated ("showing single").
        """
        Open a browser tab showing single
        """
        url = os.path.join("file://", DOC_PATH, "build", "html", single_doc_html)
        webbrowser.open(url, new=2)

    def _get_page_title(self, page):
        # [FUNCTION DOC]
        # HELPER: returns the title text of a documentation page. Parameter page
        # is the page name relative to the source folder, without extension; the
        # corresponding rst file is read as UTF-8 and parsed with the docutils
        # reStructuredText parser, with the parser's diagnostic stream redirected
        # to the null device so no warnings are printed. The title of the first
        # top-level section is returned as plain text. Raises file errors if
        # the file is missing, and a stop-iteration error if the document has no
        # section or the section has no title.
        """
        Open the rst file `page` and extract its title.
        """
        fname = os.path.join(SOURCE_PATH, f"{page}.rst")
        doc = docutils.utils.new_document(
            "<doc>",
            docutils.frontend.get_default_settings(docutils.parsers.rst.Parser),
        )
        with open(fname, encoding="utf-8") as f:
            data = f.read()

        parser = docutils.parsers.rst.Parser()
        # do not generate any warning when parsing the rst
        with open(os.devnull, "a", encoding="utf-8") as f:
            doc.reporter.stream = f
            parser.parse(data, doc)

        section = next(
            node for node in doc.children if isinstance(node, docutils.nodes.section)
        )
        title = next(
            node for node in section.children if isinstance(node, docutils.nodes.title)
        )

        return title.astext()

    def _add_redirects(self) -> None:
        # [FUNCTION DOC]
        # HELPER: writes redirect pages into the html build folder. Reads the
        # redirects csv file (UTF-8); each row has the old page path (slash
        # separated, no extension) in the first field and the new page path in
        # the second. Blank rows and rows whose first field starts with a hash
        # are skipped, as are rows whose old path lies in the reference or
        # generated folders when the API is excluded. For each remaining row,
        # a small html file is written at the old path (with an html extension)
        # containing a zero-second meta refresh to the new page and a link
        # labelled with the new page's title (or a generic phrase if the title
        # cannot be obtained for any reason).
        # Raises file errors if the csv is missing or the old page's folder does
        # not exist. Side effect: creates/overwrites files in the build folder.
        # KNOWN ISSUE: the generated html ends with a second opening html tag
        # instead of a closing tag.
        """
        Create in the build directory an html file with a redirect,
        for every row in REDIRECTS_FILE.
        """
        with open(REDIRECTS_FILE, encoding="utf-8") as mapping_fd:
            reader = csv.reader(mapping_fd)
            for row in reader:
                if not row or row[0].strip().startswith("#"):
                    continue

                html_path = os.path.join(BUILD_PATH, "html")
                path = os.path.join(html_path, *row[0].split("/")) + ".html"

                if not self.include_api and (
                    os.path.join(html_path, "reference") in path
                    or os.path.join(html_path, "generated") in path
                ):
                    continue

                try:
                    title = self._get_page_title(row[1])
                except Exception:
                    # the file can be an ipynb and not an rst, or docutils
                    # may not be able to read the rst because it has some
                    # sphinx specific stuff
                    title = "this page"

                with open(path, "w", encoding="utf-8") as moved_page_fd:
                    html = f"""\
<html>
    <head>
        <meta http-equiv="refresh" content="0;URL={row[1]}.html"/>
    </head>
    <body>
        <p>
            The page has been moved to <a href="{row[1]}.html">{title}</a>
        </p>
    </body>
<html>"""

                    moved_page_fd.write(html)

    def html(self):
        # [FUNCTION DOC]
        # Command: builds the html documentation. Runs the Sphinx html build,
        # then deletes any previous zip archive of the html output. If the build
        # succeeded (exit code zero): for a single-document build, opens the
        # page in a browser unless browser opening is disabled; for a normal
        # build, writes the redirect pages and, in whatsnew mode and unless
        # browser opening is disabled, opens the release notes index.
        # Returns the Sphinx exit code (nonzero means failure).
        """
        Build HTML documentation.
        """
        ret_code = self._sphinx_build("html")
        zip_fname = os.path.join(BUILD_PATH, "html", "pandas.zip")
        if os.path.exists(zip_fname):
            os.remove(zip_fname)  # noqa: TID251

        if ret_code == 0:
            if self.single_doc_html is not None:
                if not self.no_browser:
                    self._open_browser(self.single_doc_html)
            else:
                self._add_redirects()
                if self.whatsnew and not self.no_browser:
                    self._open_browser(os.path.join("whatsnew", "index.html"))

        return ret_code

    def latex(self, force=False):
        # [FUNCTION DOC]
        # Command: builds the pdf documentation through latex. On Windows only
        # prints a warning to the error stream and returns nothing. Otherwise
        # runs the Sphinx latex build, then changes the process working
        # directory to the latex output folder (global side effect, never
        # restored). With force true: runs the pdf typesetter three times
        # non-interactively on the main tex file and then raises a system-exit
        # request carrying a message asking the user to check the resulting pdf
        # (so the normal return is not reached). With force false: runs make in
        # the latex folder and returns the Sphinx exit code. Raises the process
        # error if an external command fails.
        # KNOWN ISSUE: the Sphinx exit code is ignored before running the
        # typesetter, so a failed build is not reported unless make fails. The
        # loop counter in the forced branch is unused.
        """
        Build PDF documentation.
        """
        if sys.platform == "win32":
            sys.stderr.write("latex build has not been tested on windows\n")
        else:
            ret_code = self._sphinx_build("latex")
            os.chdir(os.path.join(BUILD_PATH, "latex"))
            if force:
                for i in range(3):
                    self._run_os("pdflatex", "-interaction=nonstopmode", "pandas.tex")
                raise SystemExit(
                    'You should check the file "build/latex/pandas.pdf" for problems.'
                )
            self._run_os("make")
            return ret_code

    def latex_forced(self):
        # [FUNCTION DOC]
        # Command: same as the latex command with force enabled; returns what
        # that call returns (in practice it exits via a system-exit request).
        """
        Build PDF documentation with retries to find missing references.
        """
        return self.latex(force=True)

    @staticmethod
    def clean() -> None:
        # [FUNCTION DOC]
        # Command: deletes the whole build folder and the generated API reference
        # folder inside the source folder. Missing folders are ignored (no
        # error). Side effect: removes directories recursively. Returns nothing.
        """
        Clean documentation generated files.
        """
        shutil.rmtree(BUILD_PATH, ignore_errors=True)
        shutil.rmtree(os.path.join(SOURCE_PATH, "reference", "api"), ignore_errors=True)

    def zip_html(self) -> None:
        # [FUNCTION DOC]
        # Command: packs the built html folder into a zip archive named
        # pandas.zip inside that same folder. Removes any existing archive
        # first, changes the process working directory to the html folder (global
        # side effect, never restored) and calls the external zip tool
        # recursively and quietly on all entries listed at that moment.
        # Raises file errors if the html folder does not exist, and the process
        # error if the zip tool fails. Returns nothing.
        """
        Compress HTML documentation into a zip file.
        """
        zip_fname = os.path.join(BUILD_PATH, "html", "pandas.zip")
        if os.path.exists(zip_fname):
            os.remove(zip_fname)  # noqa: TID251
        dirname = os.path.join(BUILD_PATH, "html")
        fnames = os.listdir(dirname)
        os.chdir(dirname)
        self._run_os("zip", zip_fname, "-r", "-q", *fnames)

    def linkcheck(self):
        # [FUNCTION DOC]
        # Command: runs the Sphinx link checker builder and returns its exit code.
        """
        Check for broken links in the documentation.
        """
        return self._sphinx_build("linkcheck")


def main():
    # [FUNCTION DOC]
    # Script entry point. Builds the list of available commands (the public
    # attribute names of the builder class), then parses command line options:
    # the command name (optional, default html); the parallel job count
    # (default auto); a flag to omit the API; a flag to build only the release
    # notes; a single document file or object name; a python path (default
    # the parent of the doc folder); a repeatable verbosity flag; a flag to
    # treat warnings as errors; and a flag to not open the browser.
    # Raises a value error for an unknown command. Side effects: sets the
    # module path environment variable and prepends the path to the interpreter's
    # search path, imports the pandas package and stores it in this module's
    # global namespace (needed by the builder), and sets the matplotlib backend
    # environment variable to the non-interactive Agg backend. Then constructs
    # the builder and returns the result of calling the chosen command on it
    # (an exit code or none), which the script passes to the interpreter exit.
    cmds = [method for method in dir(DocBuilder) if not method.startswith("_")]

    joined = ",".join(cmds)
    argparser = argparse.ArgumentParser(
        description="pandas documentation builder", epilog=f"Commands: {joined}"
    )

    joined = ", ".join(cmds)
    argparser.add_argument(
        "command", nargs="?", default="html", help=f"command to run: {joined}"
    )
    argparser.add_argument(
        "--num-jobs", default="auto", help="number of jobs used by sphinx-build"
    )
    argparser.add_argument(
        "--no-api", default=False, help="omit api and autosummary", action="store_true"
    )
    argparser.add_argument(
        "--whatsnew",
        default=False,
        help="only build whatsnew (and api for links)",
        action="store_true",
    )
    argparser.add_argument(
        "--single",
        metavar="FILENAME",
        type=str,
        default=None,
        help=(
            "filename (relative to the 'source' folder) of section or method name to "
            "compile, e.g. 'development/contributing.rst', "
            "'pandas.DataFrame.join'"
        ),
    )
    argparser.add_argument(
        "--python-path", type=str, default=os.path.dirname(DOC_PATH), help="path"
    )
    argparser.add_argument(
        "-v",
        action="count",
        dest="verbosity",
        default=0,
        help=(
            "increase verbosity (can be repeated), passed to the sphinx build command"
        ),
    )
    argparser.add_argument(
        "--warnings-are-errors",
        "-W",
        action="store_true",
        help="fail if warnings are raised",
    )
    argparser.add_argument(
        "--no-browser",
        help="Don't open browser",
        default=False,
        action="store_true",
    )
    args = argparser.parse_args()

    if args.command not in cmds:
        joined = ", ".join(cmds)
        raise ValueError(f"Unknown command {args.command}. Available options: {joined}")

    # Below we update both os.environ and sys.path. The former is used by
    # external libraries (namely Sphinx) to compile this module and resolve
    # the import of `python_path` correctly. The latter is used to resolve
    # the import within the module, injecting it into the global namespace
    os.environ["PYTHONPATH"] = args.python_path
    sys.path.insert(0, args.python_path)
    globals()["pandas"] = importlib.import_module("pandas")

    # Set the matplotlib backend to the non-interactive Agg backend for all
    # child processes.
    os.environ["MPLBACKEND"] = "module://matplotlib.backends.backend_agg"

    builder = DocBuilder(
        args.num_jobs,
        not args.no_api,
        args.whatsnew,
        args.single,
        args.verbosity,
        args.warnings_are_errors,
        args.no_browser,
    )
    return getattr(builder, args.command)()


if __name__ == "__main__":
    sys.exit(main())
