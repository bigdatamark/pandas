# [MODULE DOC]
# Build-time script that expands a Tempita template (a file with the suffix
# .in) into a generated source file. Invoked from the build system with the
# template path and an output directory option. Public functions:
# process_tempita and main.
# Import-time side effects: IMPORTING THIS MODULE RUNS main (the call is
# unguarded at module level) and therefore parses command line arguments and
# writes the output file; it is only intended to be run as a script.
# KNOWN ISSUE: main is called unconditionally at module level, with no
# main-module guard, so merely importing the module triggers processing.
import argparse
import os

from Cython import Tempita


def process_tempita(pxifile, outfile) -> None:
    # [FUNCTION DOC]
    # Reads the template file at path pxifile as UTF-8, expands it with the
    # Tempita substitution function (no extra variables supplied) and writes
    # the resulting text as UTF-8 to path outfile, overwriting it. Raises file
    # errors if the input cannot be read or the output cannot be written, and
    # propagates template errors. Returns nothing.
    with open(pxifile, encoding="utf-8") as f:
        tmpl = f.read()
    pyxcontent = Tempita.sub(tmpl)

    with open(outfile, "w", encoding="utf-8") as f:
        f.write(pyxcontent)


def main() -> None:
    # [FUNCTION DOC]
    # Script entry point. Parses a required positional input path and an
    # output directory option. Raises a value error if the input path does not
    # end with the in suffix. The output file is placed in the output directory
    # (interpreted relative to the current working directory) and named like the
    # input file's base name with its last extension removed. Then calls the
    # template processor. Returns nothing.
    # KNOWN ISSUE: the output directory option is not required by the parser,
    # but if omitted the path join receives none and fails with a type error.
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="Path to the input file")
    parser.add_argument("-o", "--outdir", type=str, help="Path to the output directory")
    args = parser.parse_args()

    if not args.infile.endswith(".in"):
        raise ValueError(f"Unexpected extension: {args.infile}")

    outdir_abs = os.path.join(os.getcwd(), args.outdir)
    outfile = os.path.join(
        outdir_abs, os.path.splitext(os.path.split(args.infile)[1])[0]
    )

    process_tempita(args.infile, outfile)


main()
