"""Look for unreachable code."""

# [MODULE DOC]
# Command line script (runs only as main; no functions). Takes zero or more
# file paths as positional arguments, scans them with the vulture dead-code
# finder, and prints a report line for each finding of type unreachable code
# at the highest confidence level (other finding types are ignored). Exit
# status is 1 if any unreachable code was reported, else 0. Requires the
# vulture package. Intended for pre-commit use.
import argparse
import sys

from vulture import Vulture

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()

    v = Vulture()
    v.scavenge(args.files)
    ret = 0
    for item in v.get_unused_code(min_confidence=100):
        if item.typ == "unreachable_code":
            print(item.get_report())
            ret = 1

    sys.exit(ret)
