# [MODULE DOC]
# Internal module providing a mixin class that implements the string-accessor
# hook methods for arrays backed by a pyarrow chunked array, using pyarrow
# compute kernels.
# Public contents: one mixin class. It is combined with the arrow string array
# classes (and the generic arrow extension array) elsewhere in the arrays
# package, which supply the construction and result-conversion hooks.
# Imports pyarrow and its compute module only when pyarrow is available; the
# module can be imported without pyarrow. No other import-time side effects.
from __future__ import annotations

from functools import partial
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
)

import numpy as np

from pandas._libs import lib
from pandas.compat import (
    HAS_PYARROW,
    pa_version_under17p0,
    pa_version_under21p0,
)

if HAS_PYARROW:
    import pyarrow as pa
    import pyarrow.compute as pc

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandas._typing import Scalar


class ArrowStringArrayMixin:
    # [CLASS DOC]
    # Mixin supplying string-method implementations (the underscore-prefixed
    # string accessor hooks) on top of a chunked pyarrow array, using pyarrow compute
    # kernels. It is meant to be combined with a concrete array class that provides
    # the stored chunked array attribute and the hook methods declared here as
    # abstract placeholders. State: only the annotated chunked-array attribute, which
    # the host class sets. Every transformation method returns a new array of the
    # host type built through the host's construction hook; predicate and integer
    # methods hand the raw pyarrow result to the host's conversion hooks so the host
    # decides the final result type. Unsupported options raise a not-implemented error
    # so the caller can fall back to a slower path.
    _pa_array: pa.ChunkedArray

    def __init__(self, *args, **kwargs) -> None:
        # [FUNCTION DOC]
        # HELPER: abstract placeholder; the host class must provide its own constructor.
        # Accepts any arguments and always raises a not-implemented error.
        raise NotImplementedError

    def _from_pyarrow_array(self, pa_array) -> Self:
        # [FUNCTION DOC]
        # HELPER: abstract hook for the host class. Should wrap a pyarrow array in a new
        # instance of the host type. Parameter pa_array is the pyarrow data. This
        # placeholder always raises a not-implemented error.
        raise NotImplementedError

    def _convert_bool_result(self, result, na=lib.no_default, method_name=None):
        # [FUNCTION DOC]
        # HELPER: abstract hook for the host class. Should convert a boolean pyarrow
        # result into the host's boolean result type. Parameters: result is the pyarrow
        # boolean data; na is the value used for missing entries (default is the library
        # "no default" marker); method_name is an optional name of the calling string
        # method. This placeholder always raises a not-implemented error.
        # Convert a bool-dtype result to the appropriate result type
        raise NotImplementedError

    def _convert_int_result(self, result):
        # [FUNCTION DOC]
        # HELPER: abstract hook for the host class. Should convert an integer pyarrow
        # result into the host's integer result type. This placeholder always raises a
        # not-implemented error.
        # Convert an integer-dtype result to the appropriate result type
        raise NotImplementedError

    def _apply_elementwise(self, func: Callable) -> list[list[Any]]:
        # [FUNCTION DOC]
        # HELPER: abstract hook for the host class. Should apply a Python callable to
        # each element and return the results grouped as a list of lists, one inner list
        # per chunk of the underlying array. This placeholder always raises a
        # not-implemented error.
        raise NotImplementedError

    @staticmethod
    def _has_unsupported_regex(pat: str | re.Pattern) -> bool:
        # [FUNCTION DOC]
        # HELPER: static predicate used by the string methods to decide whether a
        # regular
        # expression can be handed to pyarrow's regex engine.
        # Parameter pat is a pattern string or compiled pattern. Returns true if the
        # parsed pattern contains a lookahead, lookbehind, negative assertion or
        # backreference, searching recursively through groups and alternation branches;
        # returns false otherwise, and also returns false when the pattern is not valid
        # for Python's own parser (such patterns, e.g. using the other engine's syntax,
        # are left to pyarrow to handle).
        # Raises: if the interpreter's private regex parser module cannot be imported,
        # an
        # exception of the same type is raised with a message asking to upgrade pandas
        # or
        # downgrade Python.
        """
        Determine if regex pattern contains features not supported by RE2 / pyarrow.

        This includes lookaround (lookahead or lookbehind) assertions and
        backreferences.

        Parameters
        ----------
        pat: str | re.Pattern
            Regex pattern.

        Returns
        -------
        bool
            Whether `pat` contains a lookahead or lookbehind.
        """
        try:
            # error: Module "re" has no attribute "_parser"
            from re import _parser  # type: ignore[attr-defined]

            regex_parser = _parser.parse
        except Exception as err:
            raise type(err)(
                "Incompatible version for regex; you will need to upgrade pandas "
                "or downgrade Python"
            ) from err

        def has_unsupported_code(tokens):
            # [FUNCTION DOC]
            # HELPER: nested recursive scanner for the regex support check. Takes a
            # parsed
            # token list of operation code and argument pairs and returns true when any
            # token
            # is an assertion, negative assertion or group reference, recursing into
            # subpatterns and alternation branches.
            # For certain op codes we need to recurse.
            for op_code, argument in tokens:
                if (
                    (
                        op_code == _parser.SUBPATTERN
                        and has_unsupported_code(argument[3])
                    )
                    or (
                        op_code == _parser.BRANCH
                        and any(has_unsupported_code(tokens) for tokens in argument[1])
                    )
                    or (
                        op_code
                        in [_parser.ASSERT_NOT, _parser.ASSERT, _parser.GROUPREF]
                    )
                ):
                    return True
            return False

        str_pat = pat.pattern if isinstance(pat, re.Pattern) else pat
        try:
            tokens = regex_parser(str_pat)
        except re.error:
            # Pattern not valid for Python's re (e.g. RE2 syntax like \x{...} or \p)
            # Let the pyarrow backend handle it.
            return False
        return has_unsupported_code(tokens)

    def _str_len(self):
        # [FUNCTION DOC]
        # Length of each string in characters (code points). Returns the integer result
        # produced through the host's integer conversion hook. Missing stays missing.
        result = pc.utf8_length(self._pa_array)
        return self._convert_int_result(result)

    def _str_lower(self) -> Self:
        # [FUNCTION DOC]
        # Lower-case every string. Returns a new host array.
        return self._from_pyarrow_array(pc.utf8_lower(self._pa_array))

    def _str_upper(self) -> Self:
        # [FUNCTION DOC]
        # Upper-case every string. Returns a new host array.
        return self._from_pyarrow_array(pc.utf8_upper(self._pa_array))

    def _str_strip(self, to_strip=None) -> Self:
        # [FUNCTION DOC]
        # Strip characters from both ends of each string. Parameter to_strip is a string
        # of characters to remove, or None (default) to remove whitespace. Returns a new
        # host array.
        if to_strip is None:
            result = pc.utf8_trim_whitespace(self._pa_array)
        else:
            result = pc.utf8_trim(self._pa_array, characters=to_strip)
        return self._from_pyarrow_array(result)

    def _str_lstrip(self, to_strip=None) -> Self:
        # [FUNCTION DOC]
        # Strip characters from the left end of each string. Parameter to_strip is a
        # string of characters to remove, or None (default) for whitespace. Returns a
        # new
        # host array.
        if to_strip is None:
            result = pc.utf8_ltrim_whitespace(self._pa_array)
        else:
            result = pc.utf8_ltrim(self._pa_array, characters=to_strip)
        return self._from_pyarrow_array(result)

    def _str_rstrip(self, to_strip=None) -> Self:
        # [FUNCTION DOC]
        # Strip characters from the right end of each string. Parameter to_strip is a
        # string of characters to remove, or None (default) for whitespace. Returns a
        # new
        # host array.
        if to_strip is None:
            result = pc.utf8_rtrim_whitespace(self._pa_array)
        else:
            result = pc.utf8_rtrim(self._pa_array, characters=to_strip)
        return self._from_pyarrow_array(result)

    def _str_pad(
        self,
        width: int,
        side: Literal["left", "right", "both"] = "left",
        fillchar: str = " ",
    ) -> Self:
        # [FUNCTION DOC]
        # Pad each string to a minimum width. Parameters: width is the target width;
        # side
        # is left (default, pads on the left), right, or both (centers); fillchar is the
        # padding character, default space. Returns a new host array.
        # For side both, when the installed pyarrow is older than version 17 the data is
        # converted to an object-dtype array, padded with the object implementation and
        # converted back to the host type with the same dtype. Otherwise the centering
        # kernel is used, leaning left on odd padding when the width is even, to match
        # Python's behavior.
        # Raises a value error for any other side value.
        if side == "left":
            pa_pad = pc.utf8_lpad
        elif side == "right":
            pa_pad = pc.utf8_rpad
        elif side == "both":
            if pa_version_under17p0:
                # GH#59624 fall back to object dtype
                from pandas import array

                obj_arr = self.astype(object, copy=False)  # type: ignore[attr-defined]
                obj = array(obj_arr, dtype=object)
                result = obj._str_pad(width, side, fillchar)  # type: ignore[attr-defined]
                return type(self)._from_sequence(result, dtype=self.dtype)  # type: ignore[attr-defined]
            else:
                # GH#54792
                # https://github.com/apache/arrow/issues/15053#issuecomment-2317032347
                lean_left = (width % 2) == 0
                pa_pad = partial(pc.utf8_center, lean_left_on_odd_padding=lean_left)
        else:
            raise ValueError(
                f"Invalid side: {side}. Side must be one of 'left', 'right', 'both'"
            )
        return self._from_pyarrow_array(
            pa_pad(self._pa_array, width=width, padding=fillchar)
        )

    def _str_get(self, i: int) -> Self:
        # [FUNCTION DOC]
        # Element at position i of each string, as a one-character string. Parameter i
        # is
        # an integer index; negative values count from the end. Strings too short for
        # the
        # index, and missing entries, give missing values. Implemented with a
        # one-element
        # codeunit slice (forward for non-negative, backward step for negative) and then
        # masking out-of-bounds positions. Returns a new host array.
        lengths = pc.utf8_length(self._pa_array)
        if i >= 0:
            out_of_bounds = pc.greater_equal(i, lengths)
            start = i
            stop = i + 1
            step = 1
        else:
            out_of_bounds = pc.greater(-i, lengths)
            start = i
            stop = i - 1
            step = -1
        not_out_of_bounds = pc.invert(out_of_bounds.fill_null(True))
        selected = pc.utf8_slice_codeunits(
            self._pa_array, start=start, stop=stop, step=step
        )
        null_value = pa.scalar(None, type=self._pa_array.type)
        result = pc.if_else(not_out_of_bounds, selected, null_value)
        return self._from_pyarrow_array(result)

    def _str_slice(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> Self:
        # [FUNCTION DOC]
        # Slice each string. Parameters start, stop, step default to None. A missing
        # start
        # becomes 0, or the last position when the step is negative; a missing step
        # becomes 1; a missing stop is passed through. Returns a new host array.
        if start is None:
            if step is not None and step < 0:
                # GH#59710
                start = -1
            else:
                start = 0
        if step is None:
            step = 1
        return self._from_pyarrow_array(
            pc.utf8_slice_codeunits(self._pa_array, start=start, stop=stop, step=step)
        )

    def _str_getitem(self, key: slice | int) -> Self:
        # [FUNCTION DOC]
        # Dispatcher for indexing the string accessor. Parameter key is a slice, which
        # is
        # forwarded as start, stop and step to the slice method, or an integer, which is
        # forwarded to the element-get method. Returns a new host array.
        if isinstance(key, slice):
            return self._str_slice(start=key.start, stop=key.stop, step=key.step)
        else:
            return self._str_get(key)

    def _str_slice_replace(
        self, start: int | None = None, stop: int | None = None, repl: str | None = None
    ) -> Self:
        # [FUNCTION DOC]
        # Replace a slice of each string with another string. Parameters: start (default
        # treated as 0), stop (default treated as the largest 64-bit integer, i.e. the
        # end) and repl (default treated as the empty string). Returns a new host array.
        if repl is None:
            repl = ""
        if start is None:
            start = 0
        if stop is None:
            stop = np.iinfo(np.int64).max
        return self._from_pyarrow_array(
            pc.utf8_replace_slice(self._pa_array, start, stop, repl)
        )

    def _str_replace(
        self,
        pat: str | re.Pattern,
        repl: str | Callable,
        n: int = -1,
        case: bool = True,
        flags: int = 0,
        regex: bool = True,
    ) -> Self:
        # [FUNCTION DOC]
        # Replace occurrences of a pattern in each string. Parameters: pat is the
        # pattern
        # or literal; repl is the replacement string; n is the maximum number of
        # replacements, negative meaning all (default); case is whether matching is case
        # sensitive (default true); flags are regex flags (default none); regex says
        # whether pat is a regular expression (default true).
        # Raises a not-implemented error when pat is a compiled pattern, repl is a
        # callable, case is false, flags are non-zero, or repl contains a named group
        # reference, so the caller can use another implementation.
        # An empty pattern is special-cased because pyarrow hangs on it: the replacement
        # is done element by element with Python semantics (regex substitution with a
        # count, or plain replace) and rebuilt into a chunked array of the original
        # type.
        # Otherwise uses the regex or literal substring replace kernel, with the maximum
        # replacements left unset when n is negative. Returns a new host array.
        if (
            isinstance(pat, re.Pattern)
            or callable(repl)
            or not case
            or flags
            or (isinstance(repl, str) and r"\g<" in repl)
        ):
            raise NotImplementedError(
                "replace is not supported with a re.Pattern, callable repl, "
                "case=False, flags!=0, or when the replacement string contains "
                "named group references (\\g<...>)"
            )

        if pat == "":
            # pyarrow hangs for empty patterns
            # (https://github.com/apache/arrow/issues/39149)
            # use same func definition as ObjectStringArrayMixin._str_replace
            if regex:
                count = n if n >= 0 else 0
                func = lambda val: re.sub(pat, repl, val, count=count)
            else:
                func = lambda val: val.replace(pat, repl, n)

            result = self._apply_elementwise(func)
            return self._from_pyarrow_array(
                pa.chunked_array(result, type=self._pa_array.type)
            )

        func = pc.replace_substring_regex if regex else pc.replace_substring
        # https://github.com/apache/arrow/issues/39149
        # GH 56404, unexpected behavior with negative max_replacements with pyarrow.
        pa_max_replacements = None if n < 0 else n
        result = func(
            self._pa_array,
            pattern=pat,
            replacement=repl,
            max_replacements=pa_max_replacements,
        )
        return self._from_pyarrow_array(result)

    def _str_capitalize(self) -> Self:
        # [FUNCTION DOC]
        # Capitalize each string (first character upper, rest lower). Returns a new host
        # array.
        return self._from_pyarrow_array(pc.utf8_capitalize(self._pa_array))

    def _str_title(self) -> Self:
        # [FUNCTION DOC]
        # Title-case each string. Returns a new host array.
        return self._from_pyarrow_array(pc.utf8_title(self._pa_array))

    def _str_swapcase(self) -> Self:
        # [FUNCTION DOC]
        # Swap the case of each string. Returns a new host array.
        return self._from_pyarrow_array(pc.utf8_swapcase(self._pa_array))

    def _str_removeprefix(self, prefix: str):
        # [FUNCTION DOC]
        # Remove a leading prefix from each string where present. Parameter prefix is a
        # string; an empty prefix returns the data unchanged. Otherwise strings starting
        # with the prefix get that many leading characters removed. Returns a new host
        # array.
        if prefix == "":
            return self._from_pyarrow_array(self._pa_array)
        starts_with = pc.starts_with(self._pa_array, pattern=prefix)
        removed = pc.utf8_slice_codeunits(self._pa_array, len(prefix))
        result = pc.if_else(starts_with, removed, self._pa_array)
        return self._from_pyarrow_array(result)

    def _str_removesuffix(self, suffix: str):
        # [FUNCTION DOC]
        # Remove a trailing suffix from each string where present. Parameter suffix is a
        # string; an empty suffix returns the data unchanged. Otherwise strings ending
        # with the suffix get that many trailing characters removed. Returns a new host
        # array.
        if suffix == "":
            return self._from_pyarrow_array(self._pa_array)
        ends_with = pc.ends_with(self._pa_array, pattern=suffix)
        removed = pc.utf8_slice_codeunits(self._pa_array, 0, stop=-len(suffix))
        result = pc.if_else(ends_with, removed, self._pa_array)
        return self._from_pyarrow_array(result)

    def _str_startswith(
        self, pat: str | tuple[str, ...], na: Scalar | lib.NoDefault = lib.no_default
    ):
        # [FUNCTION DOC]
        # Test whether each string starts with a prefix. Parameters: pat is a string or
        # a
        # tuple of strings (true if any matches); na is the value for missing entries
        # (default is the library "no default" marker), handled by the host's boolean
        # conversion hook. An empty tuple gives missing for missing entries and false
        # for
        # valid ones. Returns the result of the boolean conversion hook.
        if isinstance(pat, str):
            result = pc.starts_with(self._pa_array, pattern=pat)
        elif len(pat) == 0:
            # For empty tuple we return null for missing values and False
            #  for valid values.
            result = pc.if_else(pc.is_null(self._pa_array), None, False)
        else:
            result = pc.starts_with(self._pa_array, pattern=pat[0])

            for p in pat[1:]:
                result = pc.or_(result, pc.starts_with(self._pa_array, pattern=p))
        return self._convert_bool_result(result, na=na, method_name="startswith")

    def _str_endswith(
        self, pat: str | tuple[str, ...], na: Scalar | lib.NoDefault = lib.no_default
    ):
        # [FUNCTION DOC]
        # Test whether each string ends with a suffix. Parameters: pat is a string or a
        # tuple of strings (true if any matches); na is the value for missing entries
        # (default is the library "no default" marker), handled by the host's boolean
        # conversion hook. An empty tuple gives missing for missing entries and false
        # for
        # valid ones. Returns the result of the boolean conversion hook.
        if isinstance(pat, str):
            result = pc.ends_with(self._pa_array, pattern=pat)
        elif len(pat) == 0:
            # For empty tuple we return null for missing values and False
            #  for valid values.
            result = pc.if_else(pc.is_null(self._pa_array), None, False)
        else:
            result = pc.ends_with(self._pa_array, pattern=pat[0])

            for p in pat[1:]:
                result = pc.or_(result, pc.ends_with(self._pa_array, pattern=p))
        return self._convert_bool_result(result, na=na, method_name="endswith")

    def _str_isalnum(self):
        # [FUNCTION DOC]
        # Whether each string is non-empty and all alphanumeric. Returns the result of
        # the boolean conversion hook.
        result = pc.utf8_is_alnum(self._pa_array)
        return self._convert_bool_result(result)

    def _str_isalpha(self):
        # [FUNCTION DOC]
        # Whether each string is non-empty and all alphabetic. Returns the result of the
        # boolean conversion hook.
        result = pc.utf8_is_alpha(self._pa_array)
        return self._convert_bool_result(result)

    def _str_isascii(self):
        # [FUNCTION DOC]
        # Whether each string consists only of ASCII characters. Returns the result of
        # the
        # boolean conversion hook.
        result = pc.string_is_ascii(self._pa_array)
        return self._convert_bool_result(result)

    def _str_isdecimal(self):
        # [FUNCTION DOC]
        # Whether each string is non-empty and all decimal characters. Returns the
        # result
        # of the boolean conversion hook.
        result = pc.utf8_is_decimal(self._pa_array)
        return self._convert_bool_result(result)

    def _str_isdigit(self):
        # [FUNCTION DOC]
        # Whether each string is non-empty and all digit characters. When the installed
        # pyarrow is older than version 21 the native kernel is not trusted, so the
        # check
        # is done element by element with Python's own digit test and rebuilt as a
        # boolean
        # chunked array. Returns the result of the boolean conversion hook.
        if pa_version_under21p0:
            # https://github.com/pandas-dev/pandas/issues/61466
            res_list = self._apply_elementwise(str.isdigit)
            return self._convert_bool_result(
                pa.chunked_array(res_list, type=pa.bool_())
            )
        result = pc.utf8_is_digit(self._pa_array)
        return self._convert_bool_result(result)

    def _str_islower(self):
        # [FUNCTION DOC]
        # Whether each string has cased characters, all lower case. Returns the result
        # of
        # the boolean conversion hook.
        result = pc.utf8_is_lower(self._pa_array)
        return self._convert_bool_result(result)

    def _str_isnumeric(self):
        # [FUNCTION DOC]
        # Whether each string is non-empty and all numeric characters. Returns the
        # result
        # of the boolean conversion hook.
        result = pc.utf8_is_numeric(self._pa_array)
        return self._convert_bool_result(result)

    def _str_isspace(self):
        # [FUNCTION DOC]
        # Whether each string is non-empty and all whitespace. Returns the result of the
        # boolean conversion hook.
        result = pc.utf8_is_space(self._pa_array)
        return self._convert_bool_result(result)

    def _str_istitle(self):
        # [FUNCTION DOC]
        # Whether each string is title-cased. Returns the result of the boolean
        # conversion hook.
        result = pc.utf8_is_title(self._pa_array)
        return self._convert_bool_result(result)

    def _str_isupper(self):
        # [FUNCTION DOC]
        # Whether each string has cased characters, all upper case. Returns the result
        # of
        # the boolean conversion hook.
        result = pc.utf8_is_upper(self._pa_array)
        return self._convert_bool_result(result)

    def _str_contains(
        self,
        pat,
        case: bool = True,
        flags: int = 0,
        na: Scalar | lib.NoDefault = lib.no_default,
        regex: bool = True,
    ):
        # [FUNCTION DOC]
        # Test whether each string contains a pattern. Parameters: pat is the pattern;
        # case is case sensitivity (default true); flags are regex flags, only zero is
        # supported; na is the missing-value fill passed to the host conversion hook
        # (default is the "no default" marker); regex says whether pat is a regular
        # expression (default true) or a literal substring.
        # Raises a not-implemented error when flags are non-zero. Returns the result of
        # the boolean conversion hook.
        if flags:
            raise NotImplementedError(f"contains not implemented with {flags=}")

        if regex:
            pa_contains = pc.match_substring_regex
        else:
            pa_contains = pc.match_substring
        result = pa_contains(self._pa_array, pat, ignore_case=not case)
        return self._convert_bool_result(result, na=na, method_name="contains")

    def _str_match(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        na: Scalar | lib.NoDefault = lib.no_default,
    ):
        # [FUNCTION DOC]
        # Test whether each string matches a pattern at its start. Parameters: pat is a
        # regular expression string; case, flags and na as in the contains method. The
        # pattern is anchored by dropping any existing leading anchor and wrapping the
        # pattern in a group preceded by a start anchor, then the contains method is
        # called with regex on. Returns the result of the boolean conversion hook.
        if pat.startswith("^"):
            pat = pat[1:]
        pat = f"^({pat})"
        return ArrowStringArrayMixin._str_contains(
            self, pat, case, flags, na, regex=True
        )

    def _str_fullmatch(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        na: Scalar | lib.NoDefault = lib.no_default,
    ):
        # [FUNCTION DOC]
        # Test whether each whole string matches a pattern. Parameters as in the match
        # method. The pattern is rewritten to have both a leading and trailing anchor
        # (adding whichever is missing; an end anchor that is escaped counts as
        # missing),
        # then the match method is called. Returns the result of the boolean conversion
        # hook.
        if (not pat.endswith("$") or pat.endswith("\\$")) and not pat.startswith("^"):
            pat = f"^({pat})$"
        elif not pat.endswith("$") or pat.endswith("\\$"):
            pat = f"^({pat[1:]})$"
        elif not pat.startswith("^"):
            pat = f"^({pat[0:-1]})$"
        return ArrowStringArrayMixin._str_match(self, pat, case, flags, na)

    def _str_find(self, sub: str, start: int = 0, end: int | None = None):
        # [FUNCTION DOC]
        # Lowest index at which a substring is found in each string, or -1 when absent.
        # Parameters: sub is the substring; start (default 0, None treated as 0) is the
        # first position searched; end (default None) is the stop position.
        # If any string is non-ASCII, the native kernel would return byte offsets, so
        # the
        # search is done element by element with Python semantics. The same fallback is
        # used for an empty substring when start or end is given. Otherwise the strings
        # are sliced from start to end, searched, and the found positions are shifted by
        # the start offset (negative starts are converted to a non-negative offset
        # clamped
        # at zero). Results are cast to 64-bit integers and passed through the host's
        # integer conversion hook.
        if not pc.all(pc.string_is_ascii(self._pa_array)).as_py():
            # GH#64123 - pc.find_substring returns byte offsets instead of
            # character offsets for multi-byte UTF-8 characters, so we fall back
            # to Python str.find which correctly returns character offsets.
            res_list = self._apply_elementwise(lambda val: val.find(sub, start, end))
            return self._convert_int_result(pa.chunked_array(res_list))

        if (start == 0 or start is None) and end is None:
            result = pc.find_substring(self._pa_array, sub)
        else:
            if sub == "":
                # GH#56792
                res_list = self._apply_elementwise(
                    lambda val: val.find(sub, start, end)
                )
                return self._convert_int_result(pa.chunked_array(res_list))
            if start is None:
                start_offset = 0
                start = 0
            elif start < 0:
                start_offset = pc.add(start, pc.utf8_length(self._pa_array))
                start_offset = pc.if_else(pc.less(start_offset, 0), 0, start_offset)
            else:
                start_offset = start
            slices = pc.utf8_slice_codeunits(self._pa_array, start, stop=end)
            result = pc.find_substring(slices, sub)
            found = pc.not_equal(result, pa.scalar(-1, type=result.type))
            offset_result = pc.add(result, start_offset)
            result = pc.if_else(found, offset_result, -1)
        result = result.cast(pa.int64())
        return self._convert_int_result(result)
