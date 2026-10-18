#!/usr/bin/env python3

# Note: This file has to live next to setup.py or versioneer will not work

# [MODULE DOC]
# Build-time script that determines the pandas version and optionally writes
# it to a small python file and/or prints it. Public functions:
# write_version_info and main. The version comes from a generated meson
# version module when importable, otherwise from the versioneer helper (which
# needs the git history). Import-time side effects: imports versioneer,
# inserts the empty string (current directory) at the front of the module
# search path, and IMMEDIATELY RUNS main (unguarded call at module level), so
# the module is only meant to be run as a script.
# KNOWN ISSUE: main is called unconditionally at module level rather than behind
# a main-module guard.
import argparse
import os
import sys

import versioneer

sys.path.insert(0, "")


def write_version_info(path) -> None:
    # [FUNCTION DOC]
    # Writes a python file holding the version and git revision strings.
    # Parameter path is the output file path. Determines the version and full
    # git revision identifier from the generated meson version module if it can
    # be imported, otherwise from versioneer (version string and the full
    # revision id entry of its versions mapping). If the environment variable
    # naming the meson distribution root is set and non-empty, the path is
    # joined under that directory. The file (UTF-8, overwritten) contains two
    # assignment lines: the version name and the git version name, each
    # assigned a double-quoted string. Returns nothing.
    # KNOWN ISSUE: the initial assignment of none to both values is redundant
    # since both branches overwrite them.
    version = None
    git_version = None

    try:
        import _version_meson

        version = _version_meson.__version__
        git_version = _version_meson.__git_version__
    except ImportError:
        version = versioneer.get_version()
        git_version = versioneer.get_versions()["full-revisionid"]
    if os.environ.get("MESON_DIST_ROOT"):
        path = os.path.join(os.environ.get("MESON_DIST_ROOT"), path)
    with open(path, "w", encoding="utf-8") as file:
        file.write(f'__version__="{version}"\n')
        file.write(f'__git_version__="{git_version}"\n')


def main() -> None:
    # [FUNCTION DOC]
    # Script entry point. Options: an optional output file path, and a flag to
    # print the version. If an output path is given it must end with the py
    # suffix (otherwise raises a value error) and the version info file is
    # written there. If the print flag is set, prints the version string
    # (taken from the generated meson module if importable, otherwise from
    # versioneer) to standard output. Returns nothing.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        help="Path to write version info to",
        required=False,
    )
    parser.add_argument(
        "--print",
        default=False,
        action="store_true",
        help="Whether to print out the version",
        required=False,
    )
    args = parser.parse_args()

    if args.outfile:
        if not args.outfile.endswith(".py"):
            raise ValueError(
                f"Output file must be a Python file. "
                f"Got: {args.outfile} as filename instead"
            )

        write_version_info(args.outfile)

    if args.print:
        try:
            import _version_meson

            version = _version_meson.__version__
        except ImportError:
            version = versioneer.get_version()
        print(version)


main()
