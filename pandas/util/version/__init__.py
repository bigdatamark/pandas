# Vendored from https://github.com/pypa/packaging/blob/main/src/packaging/_structures.py
# and https://github.com/pypa/packaging/blob/main/src/packaging/version.py
# changeset 24e5350b2ff3c5c7a36676c2af5f2cb39fd1baf8

# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. Licence at LICENSES/PACKAGING_LICENSE
# [MODULE DOC]
# Vendored subset of the packaging library's version handling (structures and
# version modules), implementing PEP 440 version parsing and ordering so pandas
# need not depend on the external package. Public contents (exported list):
# the version-matching regular expression text, the invalid-version exception,
# the Version class and a parse function. Internal contents: two singleton
# objects that compare as larger and smaller than anything (with their types),
# type aliases for comparison keys, a named tuple holding the parsed pieces,
# a base class providing ordering by a comparison key, and helper functions
# that normalize pre/post/dev/local segments and build the comparison key.
# Dual licensed (Apache 2.0 and BSD). No import-time side effects other than
# compiling regular expressions.
from __future__ import annotations

from collections.abc import Callable
import itertools
import re
from typing import (
    Any,
    NamedTuple,
    SupportsInt,
    TypeAlias,
)

__all__ = ["VERSION_PATTERN", "InvalidVersion", "Version", "parse"]


class InfinityType:
    # [CLASS DOC]
    # HELPER: internal type of the infinity singleton used inside version comparison
    # keys. Compares greater than every other object except another instance of
    # its own type (equal to those); never less. Negation yields the negative
    # infinity singleton. Stateless; hashable via its text form.
    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # HELPER: text form, the word Infinity.
        return "Infinity"

    def __hash__(self) -> int:
        # [FUNCTION DOC]
        # HELPER: hash derived from the text form, so all instances hash alike.
        return hash(repr(self))

    def __lt__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # HELPER: less-than; always false.
        return False

    def __le__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # HELPER: less-or-equal; always false.
        return False

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # HELPER: equality; true only if the other object is of the same type.
        return isinstance(other, type(self))

    def __gt__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # HELPER: greater-than; always true.
        return True

    def __ge__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # HELPER: greater-or-equal; always true.
        return True

    def __neg__(self: object) -> NegativeInfinityType:
        # [FUNCTION DOC]
        # HELPER: negation; returns the negative infinity singleton.
        return NegativeInfinity


Infinity = InfinityType()


class NegativeInfinityType:
    # [CLASS DOC]
    # HELPER: internal type of the negative infinity singleton used inside version
    # comparison keys. Compares less than every other object except another instance
    # of its own type (equal to those); never greater. Negation yields the infinity
    # singleton. Stateless; hashable via its text form.
    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # HELPER: text form, minus sign followed by the word Infinity.
        return "-Infinity"

    def __hash__(self) -> int:
        # [FUNCTION DOC]
        # HELPER: hash derived from the text form, so all instances hash alike.
        return hash(repr(self))

    def __lt__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # HELPER: less-than; always true.
        return True

    def __le__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # HELPER: less-or-equal; always true.
        return True

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # HELPER: equality; true only if the other object is of the same type.
        return isinstance(other, type(self))

    def __gt__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # HELPER: greater-than; always false.
        return False

    def __ge__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # HELPER: greater-or-equal; always false.
        return False

    def __neg__(self: object) -> InfinityType:
        # [FUNCTION DOC]
        # HELPER: negation; returns the infinity singleton.
        return Infinity


NegativeInfinity = NegativeInfinityType()


LocalType: TypeAlias = tuple[int | str, ...]

CmpPrePostDevType: TypeAlias = InfinityType | NegativeInfinityType | tuple[str, int]
CmpLocalType: TypeAlias = (
    NegativeInfinityType
    | tuple[tuple[int, str] | tuple[NegativeInfinityType, int | str], ...]
)
CmpKey: TypeAlias = tuple[
    int,
    tuple[int, ...],
    CmpPrePostDevType,
    CmpPrePostDevType,
    CmpPrePostDevType,
    CmpLocalType,
]
VersionComparisonMethod: TypeAlias = Callable[[CmpKey, CmpKey], bool]


class _Version(NamedTuple):
    # [CLASS DOC]
    # HELPER: immutable record of the parsed pieces of a version: integer epoch,
    # tuple of integer release numbers, and optional dev, pre and post segments (each a
    # letter and number pair, or none) and an optional local segment (tuple of
    # integers and lowercase strings). Held by the Version class.
    epoch: int
    release: tuple[int, ...]
    dev: tuple[str, int] | None
    pre: tuple[str, int] | None
    post: tuple[str, int] | None
    local: LocalType | None


def parse(version: str) -> Version:
    # [FUNCTION DOC]
    # Convenience constructor. Input: version, a version string. Returns a Version
    # built from it; raises the invalid-version exception for malformed text.
    return Version(version)


# The docstring is from an older version of the packaging library to avoid
# errors in the docstring validation.
class InvalidVersion(ValueError):
    # [CLASS DOC]
    # Error raised for version strings that do not follow PEP 440. Subclass of
    # the built-in value error; stateless. Its reported module is overridden to the
    # pandas errors module so it displays as a public pandas error.
    """
    An invalid version was found, users should refer to PEP 440.

    The ``InvalidVersion`` exception is raised when a version string is
    improperly formatted. Pandas uses this exception to ensure that all
    version strings are PEP 440 compliant.

    See Also
    --------
    util.version.Version : Class for handling and parsing version strings.

    Examples
    --------
    >>> pd.util.version.Version("1.")
    Traceback (most recent call last):
    InvalidVersion: Invalid version: '1.'
    """

    __module__ = "pandas.errors"


class _BaseVersion:
    # [CLASS DOC]
    # HELPER: base class giving versions hashing and all rich comparisons based on
    # a comparison key attribute (a tuple) set by subclasses. Comparing with an object
    # that is not a version of this base type defers to the other operand. Each
    # comparison repeats the type check deliberately to avoid extra call overhead.
    _key: tuple[Any, ...]

    def __hash__(self) -> int:
        # [FUNCTION DOC]
        # HELPER: hash of the comparison key.
        return hash(self._key)

    # Please keep the duplicated `isinstance` check
    # in the six comparisons hereunder
    # unless you find a way to avoid adding overhead function calls.
    def __lt__(self, other: _BaseVersion) -> bool:
        # [FUNCTION DOC]
        # HELPER: less-than on comparison keys; defers for non-version operands.
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key < other._key

    def __le__(self, other: _BaseVersion) -> bool:
        # [FUNCTION DOC]
        # HELPER: less-or-equal on comparison keys; defers for non-version operands.
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key <= other._key

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # HELPER: equality on comparison keys; defers for non-version operands.
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key == other._key

    def __ge__(self, other: _BaseVersion) -> bool:
        # [FUNCTION DOC]
        # HELPER: greater-or-equal on comparison keys; defers for non-version operands.
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key >= other._key

    def __gt__(self, other: _BaseVersion) -> bool:
        # [FUNCTION DOC]
        # HELPER: greater-than on comparison keys; defers for non-version operands.
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key > other._key

    def __ne__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # HELPER: inequality on comparison keys; defers for non-version operands.
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key != other._key


# Deliberately not anchored to the start and end of the string, to make it
# easier for 3rd party code to reuse
_VERSION_PATTERN = r"""
    v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?                           # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
        (?P<pre>                                          # pre-release
            [-_\.]?
            (?P<pre_l>alpha|a|beta|b|preview|pre|c|rc)
            [-_\.]?
            (?P<pre_n>[0-9]+)?
        )?
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [-_\.]?
                (?P<post_l>post|rev|r)
                [-_\.]?
                (?P<post_n2>[0-9]+)?
            )
        )?
        (?P<dev>                                          # dev release
            [-_\.]?
            (?P<dev_l>dev)
            [-_\.]?
            (?P<dev_n>[0-9]+)?
        )?
    )
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
"""

VERSION_PATTERN = _VERSION_PATTERN


class Version(_BaseVersion):
    # [CLASS DOC]
    # Parsed PEP 440 version, orderable and hashable. Constructed from a string.
    # State: the parsed record of pieces and the comparison key used for ordering
    # (epoch first, release with trailing zeros ignored, then pre/post/dev/local rules
    # so that development releases sort before pre-releases, which sort before final
    # releases, which sort before post-releases). A class-level compiled pattern,
    # case-insensitive and tolerant of surrounding whitespace, validates input. Exposes
    # read-only properties for the individual parts, the public and base text forms,
    # release-kind flags and major/minor/micro numbers.
    _regex = re.compile(r"^\s*" + VERSION_PATTERN + r"\s*$", re.VERBOSE | re.IGNORECASE)
    _key: CmpKey

    def __init__(self, version: str) -> None:
        # [FUNCTION DOC]
        # Parses and stores a version. Input: version, string; leading or trailing
        # whitespace and an optional leading v are accepted. Raises the invalid-version
        # exception if the text does not match the pattern. Builds the parsed record (epoch
        # default zero, release as integers, normalized pre/post/dev segments, local segment
        # split on separators) and the comparison key.
        # Validate the version and parse it into pieces
        match = self._regex.search(version)
        if not match:
            raise InvalidVersion(f"Invalid version: '{version}'")

        # Store the parsed out pieces of the version
        self._version = _Version(
            epoch=int(match.group("epoch")) if match.group("epoch") else 0,
            release=tuple(int(i) for i in match.group("release").split(".")),
            pre=_parse_letter_version(match.group("pre_l"), match.group("pre_n")),
            post=_parse_letter_version(
                match.group("post_l"), match.group("post_n1") or match.group("post_n2")
            ),
            dev=_parse_letter_version(match.group("dev_l"), match.group("dev_n")),
            local=_parse_local_version(match.group("local")),
        )

        # Generate a key which will be used for sorting
        self._key = _cmpkey(
            self._version.epoch,
            self._version.release,
            self._version.pre,
            self._version.post,
            self._version.dev,
            self._version.local,
        )

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Debug text form: the class name with the normalized version text in angle brackets.
        return f"<Version('{self}')>"

    def __str__(self) -> str:
        # [FUNCTION DOC]
        # Normalized version text: optional epoch plus exclamation mark, dotted release,
        # pre-release letter and number concatenated, a dot post suffix, a dot dev suffix and a
        # plus local suffix, each only if present.
        parts = []

        # Epoch
        if self.epoch != 0:
            parts.append(f"{self.epoch}!")

        # Release segment
        parts.append(".".join(str(x) for x in self.release))

        # Pre-release
        if self.pre is not None:
            parts.append("".join(str(x) for x in self.pre))

        # Post-release
        if self.post is not None:
            parts.append(f".post{self.post}")

        # Development release
        if self.dev is not None:
            parts.append(f".dev{self.dev}")

        # Local version segment
        if self.local is not None:
            parts.append(f"+{self.local}")

        return "".join(parts)

    @property
    def epoch(self) -> int:
        # [FUNCTION DOC]
        # Epoch number (integer, zero when not given).
        return self._version.epoch

    @property
    def release(self) -> tuple[int, ...]:
        # [FUNCTION DOC]
        # Release numbers as a tuple of integers.
        return self._version.release

    @property
    def pre(self) -> tuple[str, int] | None:
        # [FUNCTION DOC]
        # Pre-release as a pair of normalized letter (a, b or rc) and number, or none.
        return self._version.pre

    @property
    def post(self) -> int | None:
        # [FUNCTION DOC]
        # Post-release number, or none when absent.
        return self._version.post[1] if self._version.post else None

    @property
    def dev(self) -> int | None:
        # [FUNCTION DOC]
        # Development release number, or none when absent.
        return self._version.dev[1] if self._version.dev else None

    @property
    def local(self) -> str | None:
        # [FUNCTION DOC]
        # Local version segment joined with dots into text, or none when absent.
        if self._version.local:
            return ".".join(str(x) for x in self._version.local)
        else:
            return None

    @property
    def public(self) -> str:
        # [FUNCTION DOC]
        # Public version text: the normalized text with any local segment removed.
        return str(self).split("+", 1)[0]

    @property
    def base_version(self) -> str:
        # [FUNCTION DOC]
        # Base version text: optional epoch plus exclamation mark and the dotted release only.
        parts = []

        # Epoch
        if self.epoch != 0:
            parts.append(f"{self.epoch}!")

        # Release segment
        parts.append(".".join(str(x) for x in self.release))

        return "".join(parts)

    @property
    def is_prerelease(self) -> bool:
        # [FUNCTION DOC]
        # True if the version has a pre-release or development segment.
        return self.dev is not None or self.pre is not None

    @property
    def is_postrelease(self) -> bool:
        # [FUNCTION DOC]
        # True if the version has a post-release segment.
        return self.post is not None

    @property
    def is_devrelease(self) -> bool:
        # [FUNCTION DOC]
        # True if the version has a development segment.
        return self.dev is not None

    @property
    def major(self) -> int:
        # [FUNCTION DOC]
        # First release number, or zero when the release is empty.
        return self.release[0] if len(self.release) >= 1 else 0

    @property
    def minor(self) -> int:
        # [FUNCTION DOC]
        # Second release number, or zero when fewer than two numbers.
        return self.release[1] if len(self.release) >= 2 else 0

    @property
    def micro(self) -> int:
        # [FUNCTION DOC]
        # Third release number, or zero when fewer than three numbers.
        return self.release[2] if len(self.release) >= 3 else 0


def _parse_letter_version(
    letter: str | None, number: str | bytes | SupportsInt | None
) -> tuple[str, int] | None:
    # [FUNCTION DOC]
    # HELPER: normalizes a pre/post/dev segment. Inputs: letter, optional text label;
    # number, optional numeric text or number. With a label: the number defaults to zero,
    # the label is lowercased and spelling variants are mapped (alpha to a, beta to b,
    # c, pre and preview to rc, rev and r to post), and a pair (label, integer) is
    # returned. Without a label but with a non-empty, non-zero number: treated as the
    # implicit post-release form and returns the pair (post, integer). Otherwise returns none.
    # Raises a value error if the number cannot be converted to an integer.
    if letter:
        # We consider there to be an implicit 0 in a pre-release if there is
        # not a numeral associated with it.
        if number is None:
            number = 0

        # We normalize any letters to their lower case form
        letter = letter.lower()

        # We consider some words to be alternate spellings of other words and
        # in those cases we want to normalize the spellings to our preferred
        # spelling.
        if letter == "alpha":
            letter = "a"
        elif letter == "beta":
            letter = "b"
        elif letter in ["c", "pre", "preview"]:
            letter = "rc"
        elif letter in ["rev", "r"]:
            letter = "post"

        return letter, int(number)
    if not letter and number:
        # We assume if we are given a number, but we are not given a letter
        # then this is using the implicit post release syntax (e.g. 1.0-1)
        letter = "post"

        return letter, int(number)

    return None


_local_version_separators = re.compile(r"[\._-]")


def _parse_local_version(local: str | None) -> LocalType | None:
    # [FUNCTION DOC]
    # HELPER: parses the local version text. Input: local, text or none. Splits on
    # dot, underscore and hyphen; purely numeric parts become integers, others are
    # lowercased. Returns a tuple of those parts, or none when the input is none.
    if local is not None:
        return tuple(
            part.lower() if not part.isdigit() else int(part)
            for part in _local_version_separators.split(local)
        )
    return None


def _cmpkey(
    epoch: int,
    release: tuple[int, ...],
    pre: tuple[str, int] | None,
    post: tuple[str, int] | None,
    dev: tuple[str, int] | None,
    local: LocalType | None,
) -> CmpKey:
    # [FUNCTION DOC]
    # HELPER: builds the sortable comparison key. Inputs: epoch; release (tuple of
    # integers); pre, post, dev (optional letter-number pairs); local (optional tuple).
    # Returns a tuple of epoch, release without trailing zeros, and sentinel-or-value
    # entries for pre, post, dev and local. Sentinels: a development-only version gets
    # negative infinity as its pre entry so it sorts before pre-releases; a missing pre
    # gets infinity; a missing post gets negative infinity; a missing dev gets infinity;
    # a missing local gets negative infinity; local parts are tagged so that numeric
    # parts sort after text parts.
    # When we compare a release version, we want to compare it with all of the
    # trailing zeros removed. So we'll use a reverse the list, drop all the now
    # leading zeros until we come to something non zero, then take the rest
    # re-reverse it back into the correct order and make it a tuple and use
    # that for our sorting key.
    _release = tuple(
        reversed(list(itertools.dropwhile(lambda x: x == 0, reversed(release))))
    )

    # We need to "trick" the sorting algorithm to put 1.0.dev0 before 1.0a0.
    # We'll do this by abusing the pre segment, but we _only_ want to do this
    # if there is not a pre or a post segment. If we have one of those then
    # the normal sorting rules will handle this case correctly.
    if pre is None and post is None and dev is not None:
        _pre: CmpPrePostDevType = NegativeInfinity
    # Versions without a pre-release (except as noted above) should sort after
    # those with one.
    elif pre is None:
        _pre = Infinity
    else:
        _pre = pre

    # Versions without a post segment should sort before those with one.
    if post is None:
        _post: CmpPrePostDevType = NegativeInfinity

    else:
        _post = post

    # Versions without a development segment should sort after those with one.
    if dev is None:
        _dev: CmpPrePostDevType = Infinity

    else:
        _dev = dev

    if local is None:
        # Versions without a local segment should sort before those with one.
        _local: CmpLocalType = NegativeInfinity
    else:
        # Versions with a local segment need that segment parsed to implement
        # the sorting rules in PEP440.
        # - Alpha numeric segments sort before numeric segments
        # - Alpha numeric segments sort lexicographically
        # - Numeric segments sort numerically
        # - Shorter versions sort before longer versions when the prefixes
        #   match exactly
        _local = tuple(
            (i, "") if isinstance(i, int) else (NegativeInfinity, i) for i in local
        )

    return epoch, _release, _pre, _post, _dev, _local
