# [MODULE DOC]
# Tests for the function that reports installed versions and system information, in console and JSON forms (to a file path or to standard output).

import json
import os
import re

from pandas.util._print_versions import (
    _get_dependency_info,
    _get_sys_info,
)

import pandas as pd


def test_show_versions(tmpdir):
    # [FUNCTION DOC]
    # GH39701. Writes JSON output to a file in a temporary directory, loads it (which verifies valid JSON), and checks that it equals a dictionary of the system information and dependency information helpers' results.
    # GH39701
    as_json = os.path.join(tmpdir, "test_output.json")

    pd.show_versions(as_json=as_json)

    with open(as_json, encoding="utf-8") as fd:
        # check if file output is valid JSON, will raise an exception if not
        result = json.load(fd)

    # Basic check that each version element is found in output
    expected = {
        "system": _get_sys_info(),
        "dependencies": _get_dependency_info(),
    }

    assert result == expected


def test_show_versions_console_json(capsys):
    # [FUNCTION DOC]
    # GH39701. With JSON output requested to the console, captured standard output parses as JSON equal to the dictionary of system and dependency information.
    # GH39701
    pd.show_versions(as_json=True)
    stdout = capsys.readouterr().out

    # check valid json is printed to the console if as_json is True
    result = json.loads(stdout)

    # Basic check that each version element is found in output
    expected = {
        "system": _get_sys_info(),
        "dependencies": _get_dependency_info(),
    }

    assert result == expected


def test_show_versions_console(capsys):
    # [FUNCTION DOC]
    # gh-32041. Console output includes the installed versions header, a 40-hex-digit commit hash line, a required dependency line for numpy with a version beginning with digits and a dot (development versions vary), and an optional dependency line for pyarrow showing a version or none.
    # gh-32041
    # gh-32041
    pd.show_versions(as_json=False)
    result = capsys.readouterr().out

    # check header
    assert "INSTALLED VERSIONS" in result

    # check full commit hash
    assert re.search(r"commit\s*:\s[0-9a-f]{40}\n", result)

    # check required dependency
    # 2020-12-09 npdev has "dirty" in the tag
    # 2022-05-25 npdev released with RC wo/ "dirty".
    # Just ensure we match [0-9]+\..* since npdev version is variable
    assert re.search(r"numpy\s*:\s[0-9]+\..*\n", result)

    # check optional dependency
    assert re.search(r"pyarrow\s*:\s([0-9]+.*|None)\n", result)


def test_json_output_match(capsys, tmpdir):
    # [FUNCTION DOC]
    # GH39701. The JSON text printed to the console equals the text written to a file when a path is given.
    # GH39701
    pd.show_versions(as_json=True)
    result_console = capsys.readouterr().out

    out_path = os.path.join(tmpdir, "test_json.json")
    pd.show_versions(as_json=out_path)
    with open(out_path, encoding="utf-8") as out_fd:
        result_file = out_fd.read()

    assert result_console == result_file
