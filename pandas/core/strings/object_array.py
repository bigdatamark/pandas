# [MODULE DOC]
# Object-dtype implementation of the private string methods behind the string accessor.
# Public contents: ObjectStringArrayMixin, a mixin class whose private methods (all with
# a "_str_" prefix) implement each string operation by mapping a plain Python function
# over the elements of an object-dtype array while skipping missing values.
# Relationships: the string accessor class in the accessor module calls these methods on
# the underlying array; array classes (the string extension array, numpy-extension
# array, categorical, and the arrow string array) inherit or override them. Depends on
# the compiled library helpers for mapping with a mask and for converting object
# results, and on the NA validation helper.
# No import-time side effects.
from __future__ import annotations

import functools
import re
import textwrap
from typing import (
    TYPE_CHECKING,
    Literal,
    cast,
)
import unicodedata

import numpy as np

from pandas._libs import lib
import pandas._libs.missing as libmissing
import pandas._libs.ops as libops
from pandas.util._validators import validate_na_arg

from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.missing import isna

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Sequence,
    )

    from pandas._typing import (
        NpDtype,
        Scalar,
    )


class ObjectStringArrayMixin:
    # [CLASS DOC]
    # Mixin providing the private string-method implementations for object-dtype data.
    # It has no state of its own and relies on the host class to be sized, convertible to
    # an object numpy array, and to expose a dtype with an NA value (used as the default
    # fill for missing results). Each private method takes the already-validated
    # arguments from the accessor and returns either an array (usually via the mapping
    # method), or for extract and get-dummies-style operations, plain Python lists /
    # tuples that the accessor wraps. Missing elements are skipped by the mapping step
    # and filled with the NA value (or a caller-provided value).
    """
    String Methods operating on object-dtype ndarrays.
    """

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Placeholder for typing only: the mapping method relies on the host object being
        # sized. Always raises a not-implemented error; host classes provide the real one.
        # For typing, _str_map relies on the object being sized.
        raise NotImplementedError

    def _str_getitem(self, key):
        # [FUNCTION DOC]
        # HELPER: string accessor indexing. Input: key, a slice or an element index.
        # A slice is turned into a slice operation on each string using its start, stop
        # and step; any other key is passed to the element-get operation. Returns that
        # operation's result.
        if isinstance(key, slice):
            return self._str_slice(start=key.start, stop=key.stop, step=key.step)
        else:
            return self._str_get(key)

    def _str_map(
        self,
        f,
        na_value=lib.no_default,
        dtype: NpDtype | None = None,
        convert: bool = True,
    ):
        # [FUNCTION DOC]
        # HELPER: core of every string method here. Apply a function to each non-missing
        # element and return the results as an array.
        # Inputs: f, a callable for one element; na_value, the value placed at missing
        # positions (default: the host dtype's NA value); dtype, the result dtype
        # (default object); convert, whether to let the library infer a better dtype
        # for object results (default true; inference is skipped if every element is
        # missing).
        # An empty host returns an empty array of the dtype. Otherwise the host is viewed
        # as an object array and the mask of missing elements computed. If calling f
        # raises a type or attribute error whose message looks like a wrong-number-of-
        # positional-arguments error, that error is re-raised; any other such error makes
        # the method retry with a wrapper that returns na_value for elements where f
        # raises a type or attribute error. If the library result is not a numpy array it
        # is returned as is. Unless na_value is the numpy NaN, missing positions are
        # overwritten with na_value and, when convert is true and the result is object,
        # types are re-inferred.
        # KNOWN ISSUE: the retry with the wrapper function does not forward the convert
        # argument, so conversion is always enabled on that path.
        # KNOWN ISSUE: the source carries a FIXME saying the argument-count re-raise check
        # (matching on the error message text) should be avoidable.
        """
        Map a callable over valid elements of the array.

        Parameters
        ----------
        f : Callable
            A function to call on each non-NA element.
        na_value : Scalar, optional
            The value to set for NA values. Might also be used for the
            fill value if the callable `f` raises an exception.
            This defaults to ``self.dtype.na_value`` which is ``np.nan``
            for object-dtype and Categorical and ``pd.NA`` for StringArray.
        dtype : Dtype, optional
            The dtype of the result array.
        convert : bool, default True
            Whether to call `maybe_convert_objects` on the resulting ndarray
        """
        if dtype is None:
            dtype = np.dtype("object")
        if na_value is lib.no_default:
            na_value = self.dtype.na_value  # type: ignore[attr-defined]

        if not len(self):
            return np.array([], dtype=dtype)

        arr = np.asarray(self, dtype=object)
        mask = isna(arr)
        map_convert = convert and not np.all(mask)
        try:
            result = lib.map_infer_mask(
                arr, f, mask.view(np.uint8), convert=map_convert
            )
        except (TypeError, AttributeError) as err:
            # Reraise the exception if callable `f` got wrong number of args.
            # The user may want to be warned by this, instead of getting NaN
            p_err = (
                r"((takes)|(missing)) (?(2)from \d+ to )?\d+ "
                r"(?(3)required )positional arguments?"
            )

            if len(err.args) >= 1 and re.search(p_err, err.args[0]):
                # FIXME: this should be totally avoidable
                raise err

            def g(x):
                # [FUNCTION DOC]
                # HELPER: _str_map fallback. Calls the user function on one element and
                # returns na_value if that raises a type or attribute error.
                # This type of fallback behavior can be removed once
                # we remove object-dtype .str accessor.
                try:
                    return f(x)
                except (TypeError, AttributeError):
                    return na_value

            return self._str_map(g, na_value=na_value, dtype=dtype)
        if not isinstance(result, np.ndarray):
            return result
        if na_value is not np.nan:
            np.putmask(result, mask, na_value)
            if convert and result.dtype == object:
                result = lib.maybe_convert_objects(result)
        return result

    def _str_count(self, pat, flags: int = 0):
        # [FUNCTION DOC]
        # Count regular-expression matches per element. Inputs: pat, a pattern string;
        # flags, regex flags (default 0). Compiles the pattern (invalid patterns raise
        # the regex error) and counts non-overlapping matches. Returns an integer array
        # (missing elements get the NA value).
        regex = re.compile(pat, flags=flags)
        f = lambda x: len(regex.findall(x))
        return self._str_map(f, dtype="int64")

    def _str_pad(
        self,
        width: int,
        side: Literal["left", "right", "both"] = "left",
        fillchar: str = " ",
    ):
        # [FUNCTION DOC]
        # Pad each string to a minimum width. Inputs: width; side, which side receives
        # padding ("left" default, "right" or "both", the last centering the text);
        # fillchar, the single padding character (default a space). Any other side value
        # raises a value error. Returns an object array of padded strings.
        if side == "left":
            f = lambda x: x.rjust(width, fillchar)
        elif side == "right":
            f = lambda x: x.ljust(width, fillchar)
        elif side == "both":
            f = lambda x: x.center(width, fillchar)
        else:  # pragma: no cover
            raise ValueError("Invalid side")
        return self._str_map(f)

    def _str_contains(
        self,
        pat,
        case: bool = True,
        flags: int = 0,
        na=lib.no_default,
        regex: bool = True,
    ):
        # [FUNCTION DOC]
        # Test whether each element contains a pattern. Inputs: pat; case (default true;
        # false makes the match case-insensitive); flags (regex flags); na (value for
        # missing elements, validated first; default the host NA value); regex (default
        # true; treat pat as a regular expression, otherwise as a literal substring).
        # With regex, the pattern is compiled with the ignore-case flag added when case is
        # false and searched for anywhere in the string. As a literal, case-sensitive
        # search uses substring containment; case-insensitive literal search compares
        # upper-cased pattern and text. Returns a boolean array.
        validate_na_arg(na, name="na")
        if regex:
            if not case:
                flags |= re.IGNORECASE

            pat = re.compile(pat, flags=flags)

            f = lambda x: pat.search(x) is not None
        elif case:
            f = lambda x: pat in x
        else:
            upper_pat = pat.upper()
            f = lambda x: upper_pat in x.upper()
        return self._str_map(f, na, dtype=np.dtype("bool"))

    def _str_startswith(self, pat, na=lib.no_default):
        # [FUNCTION DOC]
        # Test whether each element starts with a prefix. Inputs: pat, a string (or tuple
        # of strings); na, the value for missing elements (validated first). Returns a
        # boolean array.
        validate_na_arg(na, name="na")
        f = lambda x: x.startswith(pat)
        return self._str_map(f, na_value=na, dtype=np.dtype(bool))

    def _str_endswith(self, pat, na=lib.no_default):
        # [FUNCTION DOC]
        # Test whether each element ends with a suffix. Inputs: pat, a string (or tuple of
        # strings); na, the value for missing elements (validated first). Returns a
        # boolean array.
        validate_na_arg(na, name="na")
        f = lambda x: x.endswith(pat)
        return self._str_map(f, na_value=na, dtype=np.dtype(bool))

    def _str_replace(
        self,
        pat: str | re.Pattern,
        repl: str | Callable,
        n: int = -1,
        case: bool = True,
        flags: int = 0,
        regex: bool = True,
    ):
        # [FUNCTION DOC]
        # Replace occurrences of a pattern in each element. Inputs: pat, a string or
        # compiled pattern; repl, a replacement string or a callable receiving the match;
        # n, the maximum number of replacements (default -1 meaning all); case (false adds
        # the ignore-case flag); flags (regex flags); regex (default true).
        # If regex mode, any flags, or a callable replacement is in effect, the work is
        # done through a compiled pattern (a literal pattern is escaped first when regex
        # is false; a negative n becomes zero, meaning unlimited). Otherwise the plain
        # string replace with a count is used. Returns an array of strings.
        if case is False:
            # add case flag, if provided
            flags |= re.IGNORECASE

        if regex or flags or callable(repl):
            if not isinstance(pat, re.Pattern):
                if regex is False:
                    pat = re.escape(pat)
                pat = re.compile(pat, flags=flags)

            n = n if n >= 0 else 0
            f = lambda x: pat.sub(repl=repl, string=x, count=n)
        else:
            f = lambda x: x.replace(pat, repl, n)

        return self._str_map(f, dtype=str)

    def _str_repeat(self, repeats: int | Sequence[int]):
        # [FUNCTION DOC]
        # Repeat each element. Input: repeats, one integer applied to every element, or a
        # sequence of integers with one per element. Elements may be text or bytes (bytes
        # are tried first, then text).
        # Scalar repeats go through the mapping method. A sequence is combined element-wise
        # with the host values through the compiled binary-operation helper, passing
        # missing-value markers through unchanged; if the host is a string extension array
        # the result is rebuilt as that type and dtype, otherwise the raw array is
        # returned.
        if lib.is_integer(repeats):
            rint = cast(int, repeats)

            def scalar_rep(x):
                # [FUNCTION DOC]
                # HELPER: _str_repeat. Repeat one element a fixed number of times,
                # treating it as bytes if possible and otherwise as text.
                try:
                    return bytes.__mul__(x, rint)
                except TypeError:
                    return str.__mul__(x, rint)

            return self._str_map(scalar_rep, dtype=str)
        else:
            from pandas.core.arrays.string_ import BaseStringArray

            def rep(x, r):
                # [FUNCTION DOC]
                # HELPER: _str_repeat. Repeat element x r times (bytes first, then
                # text); the NA marker is returned unchanged.
                if x is libmissing.NA:
                    return x
                try:
                    return bytes.__mul__(x, r)
                except TypeError:
                    return str.__mul__(x, r)

            result = libops.vec_binop(
                np.asarray(self),
                np.asarray(repeats, dtype=object),
                rep,
            )
            if not isinstance(self, BaseStringArray):
                return result
            # Not going through map, so we have to do this here.
            return type(self)._from_sequence(result, dtype=self.dtype)

    def _str_match(
        self,
        pat: str | re.Pattern,
        case: bool = True,
        flags: int = 0,
        na: Scalar | lib.NoDefault = lib.no_default,
    ):
        # [FUNCTION DOC]
        # Test whether each element matches a pattern at its start. Inputs: pat, string or
        # compiled pattern; case (false adds ignore-case); flags; na, the value for
        # missing elements. If pat is already compiled, the combined flags (with the
        # unicode flag added) must equal the pattern's own flags, otherwise a value error
        # is raised. Returns a boolean array.
        if not case:
            flags |= re.IGNORECASE

        if isinstance(pat, re.Pattern):
            # We need to check that flags matches pat.flags.
            # pat.flags will have re.U regardless, so we need to add it here
            # before checking for a match
            flags = flags | re.U

            if flags != pat.flags:
                raise ValueError("Cannot pass flags that do not match pat.flags")
            regex = pat
        else:
            regex = re.compile(pat, flags=flags)

        f = lambda x: regex.match(x) is not None
        return self._str_map(f, na_value=na, dtype=np.dtype(bool))

    def _str_fullmatch(
        self,
        pat: str | re.Pattern,
        case: bool = True,
        flags: int = 0,
        na: Scalar | lib.NoDefault = lib.no_default,
    ):
        # [FUNCTION DOC]
        # Test whether each element matches a pattern in its entirety. Inputs: pat; case
        # (false adds ignore-case); flags; na, the value for missing elements. Returns a
        # boolean array.
        # KNOWN ISSUE: the pattern is typed as possibly pre-compiled, but it is always
        # passed through the compile call with the flags, which fails for a compiled
        # pattern combined with non-zero flags (unlike the match method, which checks
        # flags against the pattern).
        if not case:
            flags |= re.IGNORECASE

        regex = re.compile(pat, flags=flags)

        f = lambda x: regex.fullmatch(x) is not None
        return self._str_map(f, na_value=na, dtype=np.dtype(bool))

    def _str_encode(self, encoding, errors: str = "strict"):
        # [FUNCTION DOC]
        # Encode each string to bytes. Inputs: encoding name; errors, the error-handling
        # scheme (default "strict"). Encoding failures raise from the codec. Returns an
        # object array of bytes.
        f = lambda x: x.encode(encoding, errors=errors)
        return self._str_map(f, dtype=object)

    def _str_find(self, sub, start: int = 0, end=None):
        # [FUNCTION DOC]
        # Lowest index of a substring in each element, or -1. Inputs: sub; start (default
        # 0); end (default none, meaning to the end). Delegates to the shared find helper
        # with the left side. Returns an integer array.
        return self._str_find_(sub, start, end, side="left")

    def _str_rfind(self, sub, start: int = 0, end=None):
        # [FUNCTION DOC]
        # Highest index of a substring in each element, or -1. Inputs and result as for
        # the find method, but delegating with the right side.
        return self._str_find_(sub, start, end, side="right")

    def _str_find_(self, sub, start, end, side):
        # [FUNCTION DOC]
        # HELPER: _str_find and _str_rfind. Shared implementation. Inputs: sub; start; end
        # (none means unbounded); side, "left" to use the forward search or "right" for the
        # reverse search (anything else raises a value error). Returns an integer array of
        # positions, -1 where not found.
        if side == "left":
            method = "find"
        elif side == "right":
            method = "rfind"
        else:  # pragma: no cover
            raise ValueError("Invalid side")

        if end is None:
            f = lambda x: getattr(x, method)(sub, start)
        else:
            f = lambda x: getattr(x, method)(sub, start, end)
        return self._str_map(f, dtype="int64")

    def _str_findall(self, pat, flags: int = 0):
        # [FUNCTION DOC]
        # Find all regex matches in each element. Inputs: pat, pattern string; flags.
        # Returns an object array whose entries are lists of matches (or of group tuples
        # when the pattern has groups).
        regex = re.compile(pat, flags=flags)
        return self._str_map(regex.findall, dtype="object")

    def _str_get(self, i):
        # [FUNCTION DOC]
        # Get one component from each element. Input: i, an integer position or (for
        # dictionary elements) a key. For a dictionary element, looks up the key
        # (returning the default of the lookup if absent); for other elements, returns the
        # item at that position if the position is within range (negative positions
        # allowed), otherwise the host dtype's NA value. Returns an object array.
        def f(x):
            # [FUNCTION DOC]
            # HELPER: _str_get. Per-element lookup as described above.
            if isinstance(x, dict):
                return x.get(i)
            elif len(x) > i >= -len(x):
                return x[i]
            return self.dtype.na_value  # type: ignore[attr-defined]

        return self._str_map(f)

    def _str_index(self, sub, start: int = 0, end=None):
        # [FUNCTION DOC]
        # Lowest index of a substring in each element; raises a value error from the
        # underlying search for any element where it is absent. Inputs: sub; start
        # (default 0); end (default none). Returns an integer array.
        # KNOWN ISSUE: both branches of the conditional on end build the same function, so
        # the condition is dead code; the end value (including none) is always passed on.
        if end:
            f = lambda x: x.index(sub, start, end)
        else:
            f = lambda x: x.index(sub, start, end)
        return self._str_map(f, dtype="int64")

    def _str_rindex(self, sub, start: int = 0, end=None):
        # [FUNCTION DOC]
        # Highest index of a substring in each element; raises a value error from the
        # underlying search where it is absent. Inputs as for the index method. Returns an
        # integer array.
        # KNOWN ISSUE: as in the index method, both branches of the conditional on end are
        # identical.
        if end:
            f = lambda x: x.rindex(sub, start, end)
        else:
            f = lambda x: x.rindex(sub, start, end)
        return self._str_map(f, dtype="int64")

    def _str_join(self, sep: str):
        # [FUNCTION DOC]
        # Join the items of each list-like element with a separator. Input: sep, the
        # separator. Returns an object array of joined strings (elements that are not
        # iterable of strings become the NA value).
        return self._str_map(sep.join)

    def _str_partition(self, sep: str, expand):
        # [FUNCTION DOC]
        # Split each element at the first occurrence of a separator into a three-part
        # tuple (before, separator, after). Inputs: sep; expand (accepted for interface
        # compatibility, handled by the caller). Returns an object array of tuples.
        # KNOWN ISSUE: the expand argument is unused here, and the result is assigned to a
        # temporary variable before being returned.
        result = self._str_map(lambda x: x.partition(sep), dtype="object")
        return result

    def _str_rpartition(self, sep: str, expand):
        # [FUNCTION DOC]
        # Split each element at the last occurrence of a separator into a three-part
        # tuple. Inputs as for the partition method (expand is unused here). Returns an
        # object array of tuples.
        return self._str_map(lambda x: x.rpartition(sep), dtype="object")

    def _str_len(self):
        # [FUNCTION DOC]
        # Length of each element. Returns an integer array (NA value for missing).
        return self._str_map(len, dtype="int64")

    def _str_slice(self, start=None, stop=None, step=None):
        # [FUNCTION DOC]
        # Slice each element. Inputs: start, stop, step (all default none, as for a normal
        # slice). Returns an object array of the sliced strings.
        obj = slice(start, stop, step)
        return self._str_map(lambda x: x[obj])

    def _str_slice_replace(self, start=None, stop=None, repl=None):
        # [FUNCTION DOC]
        # Replace a slice of each element with another string. Inputs: start and stop
        # (positions, default none for unbounded); repl (default none, treated as the empty
        # string). For each element the text before start (if given), then repl, then the
        # text from the end of the slice (if stop is given) are concatenated. If the
        # selected slice is empty, the insertion happens at start instead, so nothing is
        # removed. Returns an object array.
        if repl is None:
            repl = ""

        def f(x):
            # [FUNCTION DOC]
            # HELPER: _str_slice_replace. Per-element replacement as described above.
            if x[start:stop] == "":
                local_stop = start
            else:
                local_stop = stop
            y = ""
            if start is not None:
                y += x[:start]
            y += repl
            if stop is not None:
                y += x[local_stop:]
            return y

        return self._str_map(f)

    def _str_split(
        self,
        pat: str | re.Pattern | None = None,
        n=-1,
        expand: bool = False,
        regex: bool | None = None,
    ):
        # [FUNCTION DOC]
        # Split each element into a list of parts. Inputs: pat (separator string or compiled
        # pattern; none means split on whitespace); n (maximum number of splits; none or
        # the sentinel for "unlimited" depending on mode; default -1); expand (accepted for
        # interface compatibility; handled by the caller); regex (true to treat pat as a
        # regular expression, false for literal, none for the legacy rule: a single
        # character is literal, longer patterns are regular expressions).
        # With no pattern, whitespace splitting is used and n of none or zero means
        # unlimited. With a regular-expression splitter, n of none or -1 is converted to
        # zero (unlimited). With a literal separator, n of none or zero is converted to -1
        # (unlimited). Returns an object array of lists.
        if pat is None:
            if n is None or n == 0:
                n = -1
            f = lambda x: x.split(pat, n)
        else:
            new_pat: str | re.Pattern
            if regex is True or isinstance(pat, re.Pattern):
                new_pat = re.compile(pat)
            elif regex is False:
                new_pat = pat
            # regex is None so link to old behavior #43563
            elif len(pat) == 1:
                new_pat = pat
            else:
                new_pat = re.compile(pat)

            if isinstance(new_pat, re.Pattern):
                if n is None or n == -1:
                    n = 0
                f = lambda x: new_pat.split(x, maxsplit=n)
            else:
                if n is None or n == 0:
                    n = -1
                f = lambda x: x.split(pat, n)
        return self._str_map(f, dtype=object)

    def _str_rsplit(self, pat=None, n=-1):
        # [FUNCTION DOC]
        # Split each element from the right. Inputs: pat (separator, none for whitespace);
        # n (maximum splits; none or zero is converted to -1, meaning unlimited; default
        # -1). Returns an object array of lists.
        if n is None or n == 0:
            n = -1
        f = lambda x: x.rsplit(pat, n)
        return self._str_map(f, dtype="object")

    def _str_translate(self, table):
        # [FUNCTION DOC]
        # Map characters of each element using a translation table (a mapping from
        # character ordinals to replacements or deletions). Returns an object array.
        return self._str_map(lambda x: x.translate(table))

    def _str_wrap(self, width: int, **kwargs):
        # [FUNCTION DOC]
        # Wrap each element into lines of at most a given width. Inputs: width; extra
        # keyword options accepted by the standard text wrapper (the width option is
        # overwritten with the width argument; the keyword dictionary is local). Each
        # element becomes its wrapped lines joined by newline characters. Returns an
        # object array.
        kwargs["width"] = width
        tw = textwrap.TextWrapper(**kwargs)
        return self._str_map(lambda s: "\n".join(tw.wrap(s)))

    def _str_get_dummies(self, sep: str = "|", dtype: NpDtype | None = None):
        # [FUNCTION DOC]
        # Build indicator (dummy) columns from separator-delimited strings.
        # Inputs: sep, the separator (default a vertical bar); dtype, the element type of
        # the indicator matrix (default 64-bit integer; a non-numpy extension dtype
        # results in boolean storage).
        # Missing values are treated as empty strings; non-text data is converted to text
        # if adding the separators fails. Each value is wrapped in separators, the
        # distinct tags are collected by splitting (empty tags excluded) and sorted. For
        # each tag, a column marks rows whose wrapped text contains the separator-wrapped
        # tag.
        # Returns a pair: a column-major two-dimensional array of shape (rows, tags), and
        # the sorted list of tag names.
        from pandas import Series

        if dtype is None:
            dtype = np.int64
        arr = Series(self).fillna("")
        try:
            arr = sep + arr + sep
        except (TypeError, NotImplementedError):
            arr = sep + arr.astype(str) + sep

        tags: set[str] = set()
        for ts in Series(arr, copy=False).str.split(sep):
            tags.update(ts)
        tags2 = sorted(tags - {""})

        _dtype = pandas_dtype(dtype)
        dummies_dtype: NpDtype
        if isinstance(_dtype, np.dtype):
            dummies_dtype = _dtype
        else:
            dummies_dtype = np.bool_
        dummies = np.empty((len(arr), len(tags2)), dtype=dummies_dtype, order="F")

        def _isin(test_elements: str, element: str) -> bool:
            # [FUNCTION DOC]
            # HELPER: _str_get_dummies. Returns whether the given element text occurs as a
            # substring of test_elements.
            return element in test_elements

        for i, t in enumerate(tags2):
            pat = sep + t + sep
            dummies[:, i] = lib.map_infer(
                arr.to_numpy(), functools.partial(_isin, element=pat)
            )
        return dummies, tags2

    def _str_upper(self):
        # [FUNCTION DOC]
        # Upper-case each element. Returns an object array.
        return self._str_map(lambda x: x.upper())

    def _str_isalnum(self):
        # [FUNCTION DOC]
        # Whether each element is non-empty and entirely alphanumeric. Returns a boolean
        # array.
        return self._str_map(str.isalnum, dtype="bool")

    def _str_isalpha(self):
        # [FUNCTION DOC]
        # Whether each element is non-empty and entirely alphabetic. Returns a boolean
        # array.
        return self._str_map(str.isalpha, dtype="bool")

    def _str_isascii(self):
        # [FUNCTION DOC]
        # Whether each element contains only ASCII characters (true for empty text).
        # Returns a boolean array.
        return self._str_map(str.isascii, dtype="bool")

    def _str_isdecimal(self):
        # [FUNCTION DOC]
        # Whether each element is non-empty and entirely decimal characters. Returns a
        # boolean array.
        return self._str_map(str.isdecimal, dtype="bool")

    def _str_isdigit(self):
        # [FUNCTION DOC]
        # Whether each element is non-empty and entirely digit characters. Returns a
        # boolean array.
        return self._str_map(str.isdigit, dtype="bool")

    def _str_islower(self):
        # [FUNCTION DOC]
        # Whether each element has at least one cased character and all are lower case.
        # Returns a boolean array.
        return self._str_map(str.islower, dtype="bool")

    def _str_isnumeric(self):
        # [FUNCTION DOC]
        # Whether each element is non-empty and entirely numeric characters. Returns a
        # boolean array.
        return self._str_map(str.isnumeric, dtype="bool")

    def _str_isspace(self):
        # [FUNCTION DOC]
        # Whether each element is non-empty and entirely whitespace. Returns a boolean
        # array.
        return self._str_map(str.isspace, dtype="bool")

    def _str_istitle(self):
        # [FUNCTION DOC]
        # Whether each element is title-cased. Returns a boolean array.
        return self._str_map(str.istitle, dtype="bool")

    def _str_isupper(self):
        # [FUNCTION DOC]
        # Whether each element has at least one cased character and all are upper case.
        # Returns a boolean array.
        return self._str_map(str.isupper, dtype="bool")

    def _str_capitalize(self):
        # [FUNCTION DOC]
        # Capitalize each element (first character upper, rest lower). Returns an object
        # array.
        return self._str_map(str.capitalize)

    def _str_casefold(self):
        # [FUNCTION DOC]
        # Case-fold each element for caseless comparison. Returns an object array.
        return self._str_map(str.casefold)

    def _str_title(self):
        # [FUNCTION DOC]
        # Title-case each element. Returns an object array.
        return self._str_map(str.title)

    def _str_swapcase(self):
        # [FUNCTION DOC]
        # Swap the case of each element. Returns an object array.
        return self._str_map(str.swapcase)

    def _str_lower(self):
        # [FUNCTION DOC]
        # Lower-case each element. Returns an object array.
        return self._str_map(str.lower)

    def _str_normalize(self, form):
        # [FUNCTION DOC]
        # Apply a Unicode normalization form to each element. Input: form, one of the
        # four normalization form names (an invalid name raises a value error from the
        # unicode library). Returns an object array.
        f = lambda x: unicodedata.normalize(form, x)
        return self._str_map(f)

    def _str_strip(self, to_strip=None):
        # [FUNCTION DOC]
        # Remove leading and trailing characters from each element. Input: to_strip, the
        # set of characters to remove (default none, meaning whitespace). Returns an
        # object array.
        return self._str_map(lambda x: x.strip(to_strip))

    def _str_lstrip(self, to_strip=None):
        # [FUNCTION DOC]
        # Remove leading characters from each element; to_strip as for strip. Returns an
        # object array.
        return self._str_map(lambda x: x.lstrip(to_strip))

    def _str_rstrip(self, to_strip=None):
        # [FUNCTION DOC]
        # Remove trailing characters from each element; to_strip as for strip. Returns an
        # object array.
        return self._str_map(lambda x: x.rstrip(to_strip))

    def _str_removeprefix(self, prefix: str):
        # [FUNCTION DOC]
        # Remove a leading prefix from each element if present. Input: prefix. Returns an
        # object array.
        return self._str_map(lambda x: x.removeprefix(prefix))

    def _str_removesuffix(self, suffix: str):
        # [FUNCTION DOC]
        # Remove a trailing suffix from each element if present. Input: suffix. Returns an
        # object array.
        return self._str_map(lambda x: x.removesuffix(suffix))

    def _str_extract(self, pat: str, flags: int = 0, expand: bool = True):
        # [FUNCTION DOC]
        # Extract capture groups of the first regex match in each element.
        # Inputs: pat, a pattern string; flags; expand (default true).
        # With expand false, returns an object array holding the first group of the first
        # match, or the host NA value when there is no match (the pattern is expected to
        # have exactly one group; the caller validates this). With expand true, returns a
        # plain list with one row per element, each row a list of the groups (unmatched
        # optional groups become the NA value); elements that are not text or do not match
        # give a row of NA values, one per group.
        regex = re.compile(pat, flags=flags)
        na_value = self.dtype.na_value  # type: ignore[attr-defined]

        if not expand:

            def g(x):
                # [FUNCTION DOC]
                # HELPER: _str_extract (single-group mode). Returns the first group of
                # the first match in x, or the NA value if no match.
                m = regex.search(x)
                return m.groups()[0] if m else na_value

            return self._str_map(g, convert=False)

        empty_row = [na_value] * regex.groups

        def f(x):
            # [FUNCTION DOC]
            # HELPER: _str_extract (expand mode). Returns the list of groups for the
            # first match in x (unmatched groups replaced by the NA value), or the shared
            # all-NA row for non-text or non-matching x.
            if not isinstance(x, str):
                return empty_row
            m = regex.search(x)
            if m:
                return [na_value if item is None else item for item in m.groups()]
            else:
                return empty_row

        return [f(val) for val in np.asarray(self)]

    def _str_zfill(self, width: int):
        # [FUNCTION DOC]
        # Left-pad each element with zeros to a minimum width, keeping any sign prefix at
        # the front. Input: width. Returns an object array.
        return self._str_map(lambda x: x.zfill(width))
