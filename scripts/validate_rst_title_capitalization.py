"""
Validate that the titles in the rst files follow the proper capitalization convention.

Print the titles that do not follow the convention.

Usage::

As pre-commit hook (recommended):
    pre-commit run title-capitalization --all-files

From the command-line:
    python scripts/validate_rst_title_capitalization.py <rst file>
"""

# [MODULE DOC]
# Pre-commit script checking that headings in reStructuredText files follow the pandas capitalization
# convention (sentence case with known exceptions kept in their canonical spelling), and printing the headings
# that do not. Public contents: the set of words with canonical capitalization and its lowercase lookup table,
# the message prefix, the set of heading underline symbols, a function computing the correct capitalization, a
# generator that finds headings in a file, and a main function returning the number of bad headings. As a script
# it exits with that count as the status. No import-time side effects other than building the lookup table.
from __future__ import annotations

import argparse
import re
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


CAPITALIZATION_EXCEPTIONS = {
    "pandas",
    "pd",
    "Python",
    "IPython",
    "PyTables",
    "Excel",
    "JSON",
    "HTML",
    "SAS",
    "SQL",
    "BigQuery",
    "STATA",
    "Interval",
    "IntervalArray",
    "PEP8",
    "Period",
    "Series",
    "Index",
    "DataFrame",
    "DataFrames",
    "C",
    "Git",
    "GitHub",
    "NumPy",
    "Apache",
    "Arrow",
    "Parquet",
    "MultiIndex",
    "NumFOCUS",
    "sklearn",
    "PeriodIndex",
    "NA",
    "NaN",
    "NaT",
    "ValueError",
    "Boolean",
    "BooleanArray",
    "KeyError",
    "API",
    "FAQ",
    "IO",
    "Timedelta",
    "TimedeltaIndex",
    "DatetimeIndex",
    "IntervalIndex",
    "Categorical",
    "CategoricalIndex",
    "GroupBy",
    "DataFrameGroupBy",
    "SeriesGroupBy",
    "SPSS",
    "ORC",
    "R",
    "HDF5",
    "HDFStore",
    "CDay",
    "CBMonthBegin",
    "CBMonthEnd",
    "BMonthBegin",
    "BMonthEnd",
    "BDay",
    "FY5253Quarter",
    "FY5253",
    "YearBegin",
    "YearEnd",
    "BYearBegin",
    "BYearEnd",
    "YearOffset",
    "HalfYearBegin",
    "HalfYearEnd",
    "BHalfYearBegin",
    "BHalfYearEnd",
    "HalfYearOffset",
    "QuarterBegin",
    "QuarterEnd",
    "BQuarterBegin",
    "BQuarterEnd",
    "QuarterOffset",
    "LastWeekOfMonth",
    "WeekOfMonth",
    "SemiMonthBegin",
    "SemiMonthEnd",
    "SemiMonthOffset",
    "CustomBusinessMonthBegin",
    "CustomBusinessMonthEnd",
    "BusinessMonthBegin",
    "BusinessMonthEnd",
    "MonthBegin",
    "MonthEnd",
    "MonthOffset",
    "CustomBusinessHour",
    "CustomBusinessDay",
    "BusinessHour",
    "BusinessDay",
    "DateOffset",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
    "Float64Index",
    "FloatIndex",
    "TZ",
    "GIL",
    "strftime",
    "XPORT",
    "Unicode",
    "East",
    "Asian",
    "None",
    "URLs",
    "UInt64",
    "SciPy",
    "Matplotlib",
    "PyPy",
    "SparseDataFrame",
    "Google",
    "CategoricalDtype",
    "UTC",
    "False",
    "Styler",
    "os",
    "str",
    "msgpack",
    "ExtensionArray",
    "LZMA",
    "Numba",
    "Timestamp",
    "PyArrow",
    "Liveserve",
    "I",
    "VSCode",
}

CAP_EXCEPTIONS_DICT = {word.lower(): word for word in CAPITALIZATION_EXCEPTIONS}

err_msg = "Heading capitalization formatted incorrectly. Please correctly capitalize"

symbols = ("*", "=", "-", "^", "~", "#", '"')


def correct_title_capitalization(title: str) -> str:
    # [FUNCTION DOC]
    # Computes the correctly capitalized form of a heading. Input: title, text (must not be empty). Titles starting with
    # a colon (link syntax) are returned unchanged. Otherwise, leading non-word characters are removed, the whole text is
    # capitalized (first letter upper, the rest lower), then any whole word whose lowercase form is in the exceptions
    # table (ignoring URLs in angle brackets when collecting words) is replaced by its canonical spelling. Returns the new title.
    # KNOWN ISSUE: the leading non-word characters are stripped from the corrected title, so a title starting with such
    # characters (apart from a colon) always differs from its corrected form and is reported.
    """
    Algorithm to create the correct capitalization for a given title.

    Parameters
    ----------
    title : str
        Heading string to correct.

    Returns
    -------
    str
        Correctly capitalized heading.
    """

    # Skip modification no matter what if title begins by ":" to exclude specific
    # syntax that is needed to build links.
    if title[0] == ":":
        return title

    # Strip all non-word characters from the beginning of the title to the
    # first word character.
    correct_title: str = re.sub(r"^\W*", "", title).capitalize()

    # Remove a URL from the title. We do this because words in a URL must
    # stay lowercase, even if they are a capitalization exception.
    removed_https_title = re.sub(r"<https?:\/\/.*[\r\n]*>", "", correct_title)

    # Split a title into a list using non-word character delimiters.
    word_list = re.split(r"\W", removed_https_title)

    for word in word_list:
        if word.lower() in CAP_EXCEPTIONS_DICT:
            correct_title = re.sub(
                rf"\b{word}\b", CAP_EXCEPTIONS_DICT[word.lower()], correct_title
            )

    return correct_title


def find_titles(rst_file: str) -> Iterable[tuple[str, int]]:
    # [FUNCTION DOC]
    # Generator finding headings in a reStructuredText file. Input: rst_file, path. Reads UTF-8 lines; a line made of one repeated
    # underline symbol and exactly as long as the previous line marks the previous line as a heading. Yields a pair (heading text
    # with backquotes, asterisks and underscores removed, zero-based index of the underline line, which equals the one-based line
    # number of the heading). Raises file errors if unreadable.
    """
    Algorithm to identify particular text that should be considered headings in an
    RST file.

    See <https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html> for details
    on what constitutes a string as a heading in RST.

    Parameters
    ----------
    rst_file : str
        RST file to scan through for headings.

    Yields
    -------
    title : str
        A heading found in the rst file.

    line_number : int
        The corresponding line number of the heading.
    """

    with open(rst_file, encoding="utf-8") as fd:
        previous_line = ""
        for i, line in enumerate(fd):
            line_no_last_elem = line[:-1]
            line_chars = set(line_no_last_elem)
            if (
                len(line_chars) == 1
                and line_chars.pop() in symbols
                and len(line_no_last_elem) == len(previous_line)
            ):
                yield re.sub(r"[`\*_]", "", previous_line), i
            previous_line = line_no_last_elem


def main(source_paths: list[str]) -> int:
    # [FUNCTION DOC]
    # Prints every heading with wrong capitalization and counts them. Input: source_paths, a list of file paths (not directories,
    # despite the original description). For each file and heading, if the heading differs from the corrected form, prints the file name,
    # line number, the message, the heading and the corrected heading. Returns the total number of bad headings.
    # KNOWN ISSUE: the docstring describes the parameter as a string naming directories, but it is a list of files that are opened directly.
    """
    The main method to print all headings with incorrect capitalization.

    Parameters
    ----------
    source_paths : str
        List of directories to validate, provided through command line arguments.

    Returns
    -------
    int
        Number of incorrect headings found overall.
    """

    number_of_errors: int = 0

    for filename in source_paths:
        for title, line_number in find_titles(filename):
            if title != correct_title_capitalization(title):
                print(
                    f"""{filename}:{line_number}:{err_msg} "{title}" to "{
                        correct_title_capitalization(title)
                    }" """
                )
                number_of_errors += 1

    return number_of_errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate heading capitalization")

    parser.add_argument(
        "paths", nargs="*", help="Source paths of file/directory to check."
    )

    args = parser.parse_args()

    sys.exit(main(args.paths))
