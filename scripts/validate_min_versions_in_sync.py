#!/usr/bin/env python3
"""
Check pandas required and optional dependencies are synced across:

ci/deps/actions-.*-minimum_versions.yaml
pandas/compat/_optional.py
setup.cfg

TODO: doc/source/getting_started/install.rst

This is meant to be run as a pre-commit hook - to run it manually, you can do:

    pre-commit run validate-min-versions-in-sync --all-files
"""

# [MODULE DOC]
# Pre-commit script keeping minimum dependency versions consistent across the conda environment
# files, the optional-dependency version table in the pandas compat package, and the project
# configuration file. It (1) rewrites the version constraints of every environment file in the CI
# dependencies folder and the root environment file so that the lower bound equals the configured
# minimum, and (2) compares the minimum versions from the CI minimum-versions file, the compat code
# and the project configuration, printing differences. Exit status is nonzero if any file was rewritten or
# any difference found. Public contents: path constants, an exclusion set, parsing and pinning helpers,
# three functions reading versions from the three sources, and main.
# Import-time side effects: because the pandas package may be unavailable in the hook environment, the module
# appends two pandas subfolders to the module search path, imports the exceptions utility, the version
# parser and the optional-dependency table as top-level modules, and registers the first two in the module
# registry under their pandas names. It also locates the CI minimum-versions file at import (a stop-iteration
# error occurs if none exists).
# KNOWN ISSUE: the original module docstring says the project configuration is a setup configuration file, but
# the code reads the pyproject file; and the install documentation page is only a to-do item and is not checked.
from __future__ import annotations

import pathlib
import sys
import tomllib
from typing import Any

import yaml

from scripts.generate_pip_deps_from_conda import CONDA_TO_PIP

BASE_PATH = pathlib.Path(__file__).parents[1]
DOC_PATH = (BASE_PATH / "doc/source/getting_started/install.rst").resolve()
CI_PATH = next(
    (BASE_PATH / "ci/deps").absolute().glob("actions-*-minimum_versions.yaml")
)
CODE_PATH = (BASE_PATH / "pandas/compat/_optional.py").resolve()
SETUP_PATH = (BASE_PATH / "pyproject.toml").resolve()
YAML_PATH = BASE_PATH / "ci/deps"
ENV_PATH = BASE_PATH / "environment.yml"
EXCLUDE_DEPS = {"tzdata", "pyqt", "pyqt5"}
# pandas package is not available
# in pre-commit environment
sys.path.append(str(BASE_PATH / "pandas/compat"))
sys.path.append(str(BASE_PATH / "pandas/util"))
import _exceptions
import version

sys.modules["pandas.util.version"] = version
sys.modules["pandas.util._exceptions"] = _exceptions
import _optional


def pin_min_versions_to_ci_deps() -> int:
    # [FUNCTION DOC]
    # Rewrites all environment files with pinned minimum versions. Takes no input. Reads the project configuration file
    # and, for each file in the CI dependencies folder plus the root environment file: parses the dependencies, maps
    # packages to their constraints, applies the pinning to the file text, and rewrites the file if the text changed.
    # Packages in the pip sub-list are not pinned. Returns one if any file was changed, else zero.
    """
    Pin minimum versions to CI dependencies.

    Pip dependencies are not pinned.
    """
    all_yaml_files = list(YAML_PATH.iterdir())
    all_yaml_files.append(ENV_PATH)
    toml_dependencies = {}
    with open(SETUP_PATH, "rb") as toml_f:
        toml_dependencies = tomllib.load(toml_f)
    ret = 0
    for curr_file in all_yaml_files:
        with open(curr_file, encoding="utf-8") as yaml_f:
            yaml_start_data = yaml_f.read()
        yaml_file = yaml.safe_load(yaml_start_data)
        yaml_dependencies = yaml_file["dependencies"]
        yaml_map = get_yaml_map_from(yaml_dependencies)
        toml_map = get_toml_map_from(toml_dependencies)
        yaml_result_data = pin_min_versions_to_yaml_file(
            yaml_map, toml_map, yaml_start_data
        )
        if yaml_result_data != yaml_start_data:
            with open(curr_file, "w", encoding="utf-8") as f:
                f.write(yaml_result_data)
            ret |= 1
    return ret


def get_toml_map_from(toml_dic: dict[str, Any]) -> dict[str, str]:
    # [FUNCTION DOC]
    # Builds a mapping from package name to minimum version string from the parsed project configuration. Input: toml_dic,
    # the configuration mapping. Takes the set of entries of the all-extras list, each assumed to have the form name, greater-or-equal
    # operator, version (a value error occurs otherwise). Returns the mapping.
    toml_deps = {}
    toml_dependencies = set(toml_dic["project"]["optional-dependencies"]["all"])
    for dependency in toml_dependencies:
        toml_package, toml_version = dependency.strip().split(">=")
        toml_deps[toml_package] = toml_version
    return toml_deps


def get_operator_from(dependency: str) -> str | None:
    # [FUNCTION DOC]
    # Finds the comparison operator used in a requirement string. Input: dependency, text. Checks in order for not-equal,
    # less-or-equal, greater-or-equal, single equal, greater, less; returns the first found (as text) or none.
    if "!=" in dependency:
        operator = "!="
    elif "<=" in dependency:
        operator = "<="
    elif ">=" in dependency:
        operator = ">="
    elif "=" in dependency:
        operator = "="
    elif ">" in dependency:
        operator = ">"
    elif "<" in dependency:
        operator = "<"
    else:
        operator = None
    return operator


def get_yaml_map_from(
    yaml_dic: list[str | dict[str, list[str]]],
) -> dict[str, list[str] | None]:
    # [FUNCTION DOC]
    # Builds a mapping from package to its constraints from an environment file's dependency list. Input: yaml_dic, list whose
    # entries are strings or dictionaries (dictionaries such as the pip section, and duplicates, are skipped). A
    # package with two comma-separated constraints maps to a list of two entries (the second constraint first, then the first
    # with its operator re-attached); one constraint maps to a one-entry list; a bare package maps to none.
    yaml_map: dict[str, list[str] | None] = {}
    for dependency in yaml_dic:
        if isinstance(dependency, dict) or dependency in yaml_map:
            continue
        search_text = str(dependency)
        operator = get_operator_from(search_text)
        if "," in dependency:
            yaml_dependency, yaml_version1 = search_text.split(",")
            operator = get_operator_from(yaml_dependency)
            assert operator is not None
            yaml_package, yaml_version2 = yaml_dependency.split(operator)
            yaml_version2 = operator + yaml_version2
            yaml_map[yaml_package] = [yaml_version1, yaml_version2]
        elif operator is not None:
            yaml_package, yaml_version = search_text.split(operator)
            yaml_version = operator + yaml_version
            yaml_map[yaml_package] = [yaml_version]
        else:
            yaml_package, yaml_version = search_text.strip(), None
            yaml_map[yaml_package] = yaml_version
    return yaml_map


def clean_version_list(
    yaml_versions: list[str], toml_version: version.Version
) -> list[str]:
    # [FUNCTION DOC]
    # HELPER: marks constraints below the new minimum for removal. Inputs: yaml_versions, list of constraint strings (modified
    # in place and returned); toml_version, parsed minimum version. Not-equal constraints are untouched. For each other constraint,
    # the version is parsed (operator stripped); if it is lower than the minimum it is replaced by a hyphen-prefixed
    # marker; if it is at least the minimum and the operator contains a greater-than sign it is also marked (the new
    # lower bound supersedes it). Equal constraints at or above the minimum and upper bounds are kept. Returns the list.
    # KNOWN ISSUE: removal is later detected by searching for a hyphen anywhere in the entry, so a kept constraint whose
    # version text itself contains a hyphen would also be dropped.
    for i in range(len(yaml_versions)):
        yaml_version = yaml_versions[i].strip()
        operator = get_operator_from(yaml_version)
        assert operator is not None
        if operator == "!=":
            continue
        if "<=" in operator or ">=" in operator:
            yaml_version = yaml_version[2:]
        else:
            yaml_version = yaml_version[1:]
        yaml_version = version.parse(yaml_version)
        if yaml_version < toml_version:
            yaml_versions[i] = "-" + str(yaml_version)
        elif yaml_version >= toml_version:
            if ">" in operator:
                yaml_versions[i] = "-" + str(yaml_version)
    return yaml_versions


def pin_min_versions_to_yaml_file(
    yaml_map: dict[str, list[str] | None], toml_map: dict[str, str], yaml_file_data: str
) -> str:
    # [FUNCTION DOC]
    # Applies minimum-version pinning to the text of one environment file. Inputs: yaml_map, package-to-constraints mapping
    # for the file; toml_map, package-to-minimum mapping (conda names are translated to pip names using the shared table);
    # yaml_file_data, the file text. Returns the new text. For each package that has a configured minimum: an unconstrained
    # package gets a greater-or-equal bound appended (first occurrence replaced); a constrained package has outdated
    # constraints removed and, unless the remaining constraint operator is a single equal, the greater-or-equal bound at the
    # minimum added; for an exact pin the trailing separator is trimmed. Packages without a configured minimum are untouched.
    data = yaml_file_data
    for yaml_package, yaml_versions in yaml_map.items():
        old_dep = yaml_package
        if yaml_versions is not None:
            old_dep = old_dep + ", ".join(yaml_versions)
        if CONDA_TO_PIP.get(yaml_package, yaml_package) in toml_map:
            min_dep = toml_map[CONDA_TO_PIP.get(yaml_package, yaml_package)]
        elif yaml_package in toml_map:
            min_dep = toml_map[yaml_package]
        else:
            continue
        if yaml_versions is None:
            new_dep = old_dep + ">=" + min_dep
            data = data.replace(old_dep, new_dep, 1)
            continue
        toml_version = version.parse(min_dep)
        yaml_versions_list = clean_version_list(yaml_versions, toml_version)
        cleaned_yaml_versions = [x for x in yaml_versions_list if "-" not in x]
        new_dep = yaml_package
        for clean_yaml_version in cleaned_yaml_versions:
            new_dep += clean_yaml_version + ", "
        operator = get_operator_from(new_dep)
        if operator != "=":
            new_dep += ">=" + min_dep
        else:
            new_dep = new_dep[:-2]
        data = data.replace(old_dep, new_dep)
    return data


def get_versions_from_code() -> dict[str, str]:
    # [FUNCTION DOC]
    # Returns the minimum versions known to the compat package as a mapping from lowercased install name to version text,
    # dropping the excluded packages first. Side effect: removes the excluded entries from the shared version table of the
    # compat module.
    """Min versions for checking within pandas code."""
    install_map = _optional.INSTALL_MAPPING
    inverse_install_map = {v: k for k, v in install_map.items()}
    versions = _optional.VERSIONS
    for item in EXCLUDE_DEPS:
        item = inverse_install_map.get(item, item)
        versions.pop(item, None)
    return {install_map.get(k, k).casefold(): v for k, v in versions.items()}


def get_versions_from_ci(content: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    # [FUNCTION DOC]
    # Reads minimum versions from the lines of the CI minimum-versions file. Input: content, list of lines. Parsed as text
    # (not a YAML parser) because comment markers are used. Returns a pair of mappings (required, optional) from lowercased
    # package to version. Only the pytest line is taken from the test section; lines after the required marker are
    # packages with equals-style pins (double or single equals) until the optional marker, after which they are optional;
    # the pip marker line and excluded packages are skipped.
    """Min versions in CI job for testing all optional dependencies."""
    # Don't parse with pyyaml because it ignores comments we're looking for
    seen_required = False
    seen_optional = False
    seen_test = False
    required_deps = {}
    optional_deps = {}
    for line in content:
        if "# test dependencies" in line:
            seen_test = True
        elif seen_test and "- pytest>=" in line:
            # Only grab pytest
            package, version = line.strip().split(">=")
            package = package[2:]
            optional_deps[package.casefold()] = version
        elif "# required dependencies" in line:
            seen_required = True
        elif "# optional dependencies" in line:
            seen_optional = True
        elif "- pip:" in line:
            continue
        elif seen_required and line.strip():
            if "==" in line:
                package, version = line.strip().split("==", maxsplit=1)
            else:
                package, version = line.strip().split("=", maxsplit=1)
            package = package.split()[-1]
            if package in EXCLUDE_DEPS:
                continue
            if not seen_optional:
                required_deps[package.casefold()] = version
            else:
                optional_deps[package.casefold()] = version
    return required_deps, optional_deps


def get_versions_from_toml() -> dict[str, str]:
    # [FUNCTION DOC]
    # Reads minimum versions of optional dependencies from the project configuration file. Takes no input. Uses the all-extras
    # list minus plugin entries of the test list (those starting with the pytest prefix), splits each on the
    # greater-or-equal operator, maps install names through the compat table, lowercases names, drops excluded packages. Returns a
    # mapping from name to version text.
    """Min versions in pyproject.toml for pip install pandas[extra]."""
    install_map = _optional.INSTALL_MAPPING
    optional_dependencies = {}
    with open(SETUP_PATH, "rb") as pyproject_f:
        pyproject_toml = tomllib.load(pyproject_f)
        opt_deps = pyproject_toml["project"]["optional-dependencies"]
        dependencies = set(opt_deps["all"])

        # remove pytest plugin dependencies
        pytest_plugins = {dep for dep in opt_deps["test"] if dep.startswith("pytest-")}
        dependencies = dependencies.difference(pytest_plugins)

    for dependency in dependencies:
        package, version = dependency.strip().split(">=")
        optional_dependencies[install_map.get(package, package).casefold()] = version

    for item in EXCLUDE_DEPS:
        optional_dependencies.pop(item, None)
    return optional_dependencies


def main() -> int:
    # [FUNCTION DOC]
    # Entry point. First pins versions in all environment files, then compares optional minimums from the CI file, the compat
    # code and the project configuration. Writes to standard output, for each package whose values differ between sources, the
    # package and each source's value ("Not specified" when absent). Returns zero if nothing changed and no differences,
    # otherwise one.
    ret = 0
    ret |= pin_min_versions_to_ci_deps()
    with open(CI_PATH, encoding="utf-8") as f:
        _, ci_optional = get_versions_from_ci(f.readlines())
    code_optional = get_versions_from_code()
    setup_optional = get_versions_from_toml()

    diff = (ci_optional.items() | code_optional.items() | setup_optional.items()) - (
        ci_optional.items() & code_optional.items() & setup_optional.items()
    )

    if diff:
        packages = {package for package, _ in diff}
        out = sys.stdout
        out.write(
            f"The follow minimum version differences were found between  "
            f"{CI_PATH}, {CODE_PATH} AND {SETUP_PATH}. "
            f"Please ensure these are aligned: \n\n"
        )

        for package in packages:
            out.write(
                f"{package}\n"
                f"{CI_PATH}: {ci_optional.get(package, 'Not specified')}\n"
                f"{CODE_PATH}: {code_optional.get(package, 'Not specified')}\n"
                f"{SETUP_PATH}: {setup_optional.get(package, 'Not specified')}\n\n"
            )
        ret |= 1
    return ret


if __name__ == "__main__":
    sys.exit(main())
