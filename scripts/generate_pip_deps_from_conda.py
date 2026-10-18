#!/usr/bin/env python3
"""
Convert the conda environment.yml to the pip requirements-dev.txt,
or check that they have the same packages (for the CI)

Usage:

    Generate `requirements-dev.txt`
    $ python scripts/generate_pip_deps_from_conda.py

    Compare and fail (exit status != 0) if `requirements-dev.txt` has not been
    generated with this script:
    $ python scripts/generate_pip_deps_from_conda.py --compare
"""

# [MODULE DOC]
# Script that converts the conda environment file of the repository into the pip
# requirements file for development, or verifies that the existing pip file is up to date
# (used in continuous integration). Public contents: constants listing conda packages to
# exclude, a version override table, and a conda-to-pip name table; a function converting one
# conda requirement to pip form; a function generating or comparing the pip file; and a
# command line block (run as script) using the files environment.yml and
# requirements-dev.txt in the repository root, one level above this scripts directory.
# Requires the yaml package and the standard TOML reader. The process exit status is one
# if the comparison found a difference, else zero; a message goes to standard error.
import argparse
import pathlib
import re
import sys
import tomllib

import yaml

EXCLUDE = {"python", "c-compiler", "cxx-compiler"}
REMAP_VERSION = {"tzdata": "2023.3"}
CONDA_TO_PIP = {
    "versioneer": "versioneer[toml]",
    "meson": "meson[ninja]",
    "pytables": "tables",
    "psycopg2": "psycopg2-binary",
    "dask-core": "dask",
    "seaborn-base": "seaborn",
    "sqlalchemy": "SQLAlchemy",
    "pyqt": "PyQt5",
}


def conda_package_to_pip(package: str):
    # [FUNCTION DOC]
    # Converts one conda requirement string to a pip requirement string. Input: package, the
    # conda spec. A single equals sign is turned into a double equals sign (comparison operators
    # are left alone) and whitespace is trimmed. For specs with a lower-or-equal, greater-or-equal
    # or equality comparison, the name is checked: excluded packages return nothing; packages with a
    # pinned override version are returned with the comparison operator and the override version
    # (ignoring the original version); renamed packages return the pip name, the operator and the
    # original version. Specs without these operators are checked against the exclusion and rename
    # tables by exact name; otherwise the cleaned spec is returned. Returns text, or none when excluded.
    # KNOWN ISSUE: only the first matching operator that triggers a branch returns; a spec such as a
    # name with a strict less-than or greater-than operator is never renamed or excluded and falls
    # through unchanged, so the rename tables apply only to unversioned names and to the three listed
    # operators.
    """
    Convert a conda package to its pip equivalent.

    In most cases they are the same, those are the exceptions:
    - Packages that should be excluded (in `EXCLUDE`)
    - Packages that should be renamed (in `CONDA_TO_PIP`)
    - A package requiring a specific version, in conda is defined with a single
      equal (e.g. ``pandas=1.0``) and in pip with two (e.g. ``pandas==1.0``)
    """
    package = re.sub("(?<=[^<>~!])=", "==", package).strip()

    for compare in ("<=", ">=", "=="):
        if compare in package:
            pkg, version = package.split(compare)
            if pkg in EXCLUDE:
                return
            if pkg in REMAP_VERSION:
                return "".join((pkg, compare, REMAP_VERSION[pkg]))
            if pkg in CONDA_TO_PIP:
                return "".join((CONDA_TO_PIP[pkg], compare, version))

    if package in EXCLUDE:
        return

    if package in CONDA_TO_PIP:
        return CONDA_TO_PIP[package]

    return package


def generate_pip_from_conda(
    conda_path: pathlib.Path, pip_path: pathlib.Path, compare: bool = False
) -> bool:
    # [FUNCTION DOC]
    # Generates or checks the pip requirements file. Inputs: conda_path, path to the conda file;
    # pip_path, path of the pip file; compare, default false. Reads the dependency list from the conda
    # file: each string entry is converted (excluded ones dropped), a single-key pip section dictionary is
    # appended as is, and any other entry raises a value error. Builds file content: a generated-file
    # header, one requirement per line, then every setuptools requirement from the build-system section
    # of the pyproject file located next to the conda file. In compare mode returns true when the
    # existing pip file differs from the generated content and writes nothing; otherwise overwrites the
    # pip file and returns false. Raises file-system and parsing errors if inputs are missing or invalid.
    """
    Generate the pip dependencies file from the conda file, or compare that
    they are synchronized (``compare=True``).

    Parameters
    ----------
    conda_path : pathlib.Path
        Path to the conda file with dependencies (e.g. `environment.yml`).
    pip_path : pathlib.Path
        Path to the pip file with dependencies (e.g. `requirements-dev.txt`).
    compare : bool, default False
        Whether to generate the pip file (``False``) or to compare if the
        pip file has been generated with this script and the last version
        of the conda file (``True``).

    Returns
    -------
    bool
        True if the comparison fails, False otherwise
    """
    with conda_path.open() as file:
        deps = yaml.safe_load(file)["dependencies"]

    pip_deps = []
    for dep in deps:
        if isinstance(dep, str):
            conda_dep = conda_package_to_pip(dep)
            if conda_dep:
                pip_deps.append(conda_dep)
        elif isinstance(dep, dict) and len(dep) == 1 and "pip" in dep:
            pip_deps.extend(dep["pip"])
        else:
            raise ValueError(f"Unexpected dependency {dep}")

    header = (
        f"# This file is auto-generated from {conda_path.name}, do not modify.\n"
        "# See that file for comments about the need/usage of each dependency.\n\n"
    )
    pip_content = header + "\n".join(pip_deps) + "\n"

    # add setuptools to requirements-dev.txt
    with open(pathlib.Path(conda_path.parent, "pyproject.toml"), "rb") as fd:
        meta = tomllib.load(fd)
    for requirement in meta["build-system"]["requires"]:
        if "setuptools" in requirement:
            pip_content += requirement
            pip_content += "\n"

    if compare:
        with pip_path.open() as file:
            return pip_content != file.read()

    with pip_path.open("w") as file:
        file.write(pip_content)
    return False


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="convert (or compare) conda file to pip"
    )
    argparser.add_argument(
        "--compare",
        action="store_true",
        help="compare whether the two files are equivalent",
    )
    args = argparser.parse_args()

    conda_fname = "environment.yml"
    pip_fname = "requirements-dev.txt"
    repo_path = pathlib.Path(__file__).parent.parent.absolute()
    res = generate_pip_from_conda(
        pathlib.Path(repo_path, conda_fname),
        pathlib.Path(repo_path, pip_fname),
        compare=args.compare,
    )
    if res:
        msg = (
            f"`{pip_fname}` has to be generated with `{__file__}` after "
            f"`{conda_fname}` is modified.\n"
        )
        sys.stderr.write(msg)
    sys.exit(res)
