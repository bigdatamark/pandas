# [MODULE DOC]
# ASV benchmark module for the string accessor methods and string-typed data
# structures of pandas. Public contents: Dtypes (base class with the dtype
# parameter list and a setup creating a 10000-element string series), and
# benchmark classes Construction, Methods, Repeat, Cat, Contains, Split,
# Extract, Dummies, Encode, Slice, Iter and StringArrayConstruction. Several
# classes inherit from Dtypes to reuse its parameters and series.
# Import-time side effects: none beyond importing pandas and numpy names.
import warnings

import numpy as np

from pandas import (
    NA,
    Categorical,
    DataFrame,
    Index,
    Series,
    StringDtype,
)
from pandas.arrays import StringArray


class Dtypes:
    # [CLASS DOC]
    # HELPER: base class for string benchmarks that need a ready-made series.
    # Class attributes: params lists three dtype specifiers (the default string
    # dtype, python-backed string, and arrow-backed string); param_names has the
    # single name dtype. Subclasses call the setup method to get the series
    # stored on the instance.
    params = ["str", "string[python]", "string[pyarrow]"]
    param_names = ["dtype"]

    def setup(self, dtype):
        # [FUNCTION DOC]
        # HELPER: shared setup for the string benchmark subclasses.
        # Builds a series of 10000 distinct strings (prefix i- plus a counter)
        # with the requested dtype, stored on the instance. Raises a
        # not-implemented error (which asv treats as a skip) when creating the
        # series fails with an import error, e.g. when the arrow library is
        # not installed.
        try:
            self.s = Series(
                Index([f"i-{i}" for i in range(10000)], dtype=object)._values,
                dtype=dtype,
            )
        except ImportError as err:
            raise NotImplementedError from err


class Construction:
    # [CLASS DOC]
    # Benchmarks constructing a series, frame or categorical-backed series from
    # a numpy array of strings, with a string dtype. Parameterized over the
    # container kind and dtype. Class attributes pd_mapping and dtype_mapping
    # map the kind to the constructor and the dtype specifier to the numpy
    # dtype used to build the source array. State set by setup: the source array.
    params = (
        ["series", "frame", "categorical_series"],
        ["str", "string[python]", "string[pyarrow]"],
    )
    param_names = ["pd_type", "dtype"]
    pd_mapping = {"series": Series, "frame": DataFrame, "categorical_series": Series}
    dtype_mapping = {"str": "str", "string[python]": object, "string[pyarrow]": object}

    def setup(self, pd_type, dtype):
        # [FUNCTION DOC]
        # Builds a numpy array of 100000 strings (each counter repeated ten
        # times as text) with the numpy dtype mapped from dtype. For the series
        # kind the flat array is stored; for the frame kind it is reshaped to
        # 50000 rows by two columns and copied; for the categorical kind it is
        # wrapped in a categorical (regression check for issue 37371,
        # construction from extension arrays).
        series_arr = np.array(
            [str(i) * 10 for i in range(100_000)], dtype=self.dtype_mapping[dtype]
        )
        if pd_type == "series":
            self.arr = series_arr
        elif pd_type == "frame":
            self.arr = series_arr.reshape((50_000, 2)).copy()
        elif pd_type == "categorical_series":
            # GH37371. Testing construction of string series/frames from ExtensionArrays
            self.arr = Categorical(series_arr)

    def time_construction(self, pd_type, dtype):
        # [FUNCTION DOC]
        # Times calling the constructor mapped from pd_type on the prepared
        # array with the dtype specifier.
        self.pd_mapping[pd_type](self.arr, dtype=dtype)

    def peakmem_construction(self, pd_type, dtype):
        # [FUNCTION DOC]
        # Same operation as the timing method but measured by asv for peak
        # memory use instead of time.
        self.pd_mapping[pd_type](self.arr, dtype=dtype)


class Methods(Dtypes):
    # [CLASS DOC]
    # Benchmarks one string accessor method per timing method on the series
    # prepared by the inherited setup (parameterized over the dtype list).
    # The data are strings like i-123, so patterns using an uppercase letter
    # mostly do not match.
    def time_center(self, dtype):
        # [FUNCTION DOC]
        # Times centering each string in a field of width 100.
        self.s.str.center(100)

    def time_count(self, dtype):
        # [FUNCTION DOC]
        # Times counting occurrences of the pattern A in each string.
        self.s.str.count("A")

    def time_endswith(self, dtype):
        # [FUNCTION DOC]
        # Times testing whether each string ends with A.
        self.s.str.endswith("A")

    def time_extract(self, dtype):
        # [FUNCTION DOC]
        # Times extracting the two capture groups around a literal A from each
        # string, with warnings recorded and suppressed.
        with warnings.catch_warnings(record=True):
            self.s.str.extract("(\\w*)A(\\w*)")

    def time_findall(self, dtype):
        # [FUNCTION DOC]
        # Times finding all runs of uppercase letters in each string.
        self.s.str.findall("[A-Z]+")

    def time_find(self, dtype):
        # [FUNCTION DOC]
        # Times finding the first position of the given text in each string.
        # KNOWN ISSUE: the argument looks like a regular expression, but the
        # find method searches for a literal substring, so it never matches.
        self.s.str.find("[A-Z]+")

    def time_rfind(self, dtype):
        # [FUNCTION DOC]
        # Times finding the last position of the given text in each string.
        # KNOWN ISSUE: as with the find benchmark, the pattern-like argument is
        # treated as a literal substring.
        self.s.str.rfind("[A-Z]+")

    def time_fullmatch(self, dtype):
        # [FUNCTION DOC]
        # Times testing whether each string entirely matches the pattern A.
        self.s.str.fullmatch("A")

    def time_get(self, dtype):
        # [FUNCTION DOC]
        # Times taking the first character of each string.
        self.s.str.get(0)

    def time_len(self, dtype):
        # [FUNCTION DOC]
        # Times computing the length of each string.
        self.s.str.len()

    def time_join(self, dtype):
        # [FUNCTION DOC]
        # Times joining the characters of each string with a space.
        self.s.str.join(" ")

    def time_match(self, dtype):
        # [FUNCTION DOC]
        # Times testing whether each string starts with a match of A.
        self.s.str.match("A")

    def time_normalize(self, dtype):
        # [FUNCTION DOC]
        # Times Unicode normalization of each string to the composed form.
        self.s.str.normalize("NFC")

    def time_pad(self, dtype):
        # [FUNCTION DOC]
        # Times padding each string on both sides to width 100.
        self.s.str.pad(100, side="both")

    def time_partition(self, dtype):
        # [FUNCTION DOC]
        # Times splitting each string at the first A into three parts.
        self.s.str.partition("A")

    def time_rpartition(self, dtype):
        # [FUNCTION DOC]
        # Times splitting each string at the last A into three parts.
        self.s.str.rpartition("A")

    def time_replace(self, dtype):
        # [FUNCTION DOC]
        # Times replacing A with two control characters in each string.
        self.s.str.replace("A", "\x01\x01")

    def time_translate(self, dtype):
        # [FUNCTION DOC]
        # Times translating A to two control characters using a mapping.
        self.s.str.translate({"A": "\x01\x01"})

    def time_slice(self, dtype):
        # [FUNCTION DOC]
        # Times slicing characters 5 up to 15 with step 2 from each string.
        self.s.str.slice(5, 15, 2)

    def time_startswith(self, dtype):
        # [FUNCTION DOC]
        # Times testing whether each string starts with A.
        self.s.str.startswith("A")

    def time_strip(self, dtype):
        # [FUNCTION DOC]
        # Times stripping A characters from both ends of each string.
        self.s.str.strip("A")

    def time_rstrip(self, dtype):
        # [FUNCTION DOC]
        # Times stripping A characters from the right end of each string.
        self.s.str.rstrip("A")

    def time_lstrip(self, dtype):
        # [FUNCTION DOC]
        # Times stripping A characters from the left end of each string.
        self.s.str.lstrip("A")

    def time_title(self, dtype):
        # [FUNCTION DOC]
        # Times converting each string to title case.
        self.s.str.title()

    def time_upper(self, dtype):
        # [FUNCTION DOC]
        # Times converting each string to upper case.
        self.s.str.upper()

    def time_lower(self, dtype):
        # [FUNCTION DOC]
        # Times converting each string to lower case.
        self.s.str.lower()

    def time_wrap(self, dtype):
        # [FUNCTION DOC]
        # Times wrapping each string at a line width of 10.
        self.s.str.wrap(10)

    def time_zfill(self, dtype):
        # [FUNCTION DOC]
        # Times left-padding each string with zeros to width 10.
        self.s.str.zfill(10)

    def time_isalnum(self, dtype):
        # [FUNCTION DOC]
        # Times the alphanumeric test on each string.
        self.s.str.isalnum()

    def time_isalpha(self, dtype):
        # [FUNCTION DOC]
        # Times the alphabetic test on each string.
        self.s.str.isalpha()

    def time_isdecimal(self, dtype):
        # [FUNCTION DOC]
        # Times the decimal-character test on each string.
        self.s.str.isdecimal()

    def time_isdigit(self, dtype):
        # [FUNCTION DOC]
        # Times the digit test on each string.
        self.s.str.isdigit()

    def time_islower(self, dtype):
        # [FUNCTION DOC]
        # Times the lower-case test on each string.
        self.s.str.islower()

    def time_isnumeric(self, dtype):
        # [FUNCTION DOC]
        # Times the numeric test on each string.
        self.s.str.isnumeric()

    def time_isspace(self, dtype):
        # [FUNCTION DOC]
        # Times the whitespace test on each string.
        self.s.str.isspace()

    def time_istitle(self, dtype):
        # [FUNCTION DOC]
        # Times the title-case test on each string.
        self.s.str.istitle()

    def time_isupper(self, dtype):
        # [FUNCTION DOC]
        # Times the upper-case test on each string.
        self.s.str.isupper()


class Repeat:
    # [CLASS DOC]
    # Benchmarks repeating each string, with the repeat count given either as a
    # single integer or an array of counts. Parameterized over the kind (int or
    # array). State set by setup: an object-dtype series of 100000 strings and
    # the repeat values.
    params = ["int", "array"]
    param_names = ["repeats"]

    def setup(self, repeats):
        # [FUNCTION DOC]
        # Builds a series of 100000 distinct strings and the repeat argument:
        # the integer one for the int kind, or an array of random integers of
        # one or two for the array kind. Both candidates are computed eagerly
        # and the one named by repeats is stored.
        N = 10**5
        self.s = Series(Index([f"i-{i}" for i in range(N)], dtype=object))
        repeat = {"int": 1, "array": np.random.randint(1, 3, N)}
        self.values = repeat[repeats]

    def time_repeat(self, repeats):
        # [FUNCTION DOC]
        # Times repeating each string by the prepared count(s). Parameter
        # repeats is unused here.
        self.s.str.repeat(self.values)


class Cat:
    # [CLASS DOC]
    # Benchmarks concatenating strings across a series and optional other
    # columns. Parameterized over the number of other columns (0 or 3), the
    # separator, the missing-value replacement and the fraction of missing
    # values. State set by setup: a series with random missing entries and
    # either no other objects or a frame of other columns also with missing
    # entries.
    params = ([0, 3], [None, ","], [None, "-"], [0.0, 0.001, 0.15])
    param_names = ["other_cols", "sep", "na_rep", "na_frac"]

    def setup(self, other_cols, sep, na_rep, na_frac):
        # [FUNCTION DOC]
        # Builds a local helper (an assigned anonymous function) which, per
        # call, draws a random boolean mask of length 100000 that is true with
        # probability one minus na_frac. The main series of distinct strings is
        # masked with it (false positions become missing). When other_cols is
        # zero the others attribute is none (self-concatenation only occurs in
        # that case); otherwise it is a frame with other_cols columns of similarly
        # masked strings keyed by column number. Parameters sep and na_rep are
        # unused here.
        N = 10**5
        mask_gen = lambda: np.random.choice([True, False], N, p=[1 - na_frac, na_frac])
        self.s = Series(Index([f"i-{i}" for i in range(N)], dtype=object)).where(
            mask_gen()
        )
        if other_cols == 0:
            # str.cat self-concatenates only for others=None
            self.others = None
        else:
            self.others = DataFrame(
                {
                    i: Index([f"i-{i}" for i in range(N)], dtype=object).where(
                        mask_gen()
                    )
                    for i in range(other_cols)
                }
            )

    def time_cat(self, other_cols, sep, na_rep, na_frac):
        # [FUNCTION DOC]
        # Times concatenating the series with the others using the separator and
        # missing-value replacement. Per the source comment, with three other
        # columns and a 15 percent missing fraction about 48 percent of rows
        # contain a missing value before concatenation. Parameters other_cols and
        # na_frac are unused here.
        # before the concatenation (one caller + other_cols columns), the total
        # expected fraction of rows containing any NaN is:
        # reduce(lambda t, _: t + (1 - t) * na_frac, range(other_cols + 1), 0)
        # for other_cols=3 and na_frac=0.15, this works out to ~48%
        self.s.str.cat(others=self.others, sep=sep, na_rep=na_rep)


class Contains(Dtypes):
    # [CLASS DOC]
    # Benchmarks the substring/pattern containment test on the inherited series,
    # parameterized over dtype and whether the pattern is a regular expression.
    params = (Dtypes.params, [True, False])
    param_names = ["dtype", "regex"]

    def setup(self, dtype, regex):
        # [FUNCTION DOC]
        # Delegates to the base setup with only the dtype; regex is unused here.
        super().setup(dtype)

    def time_contains(self, dtype, regex):
        # [FUNCTION DOC]
        # Times testing whether each string contains A, as regex or literal
        # according to the flag.
        self.s.str.contains("A", regex=regex)


class Split(Dtypes):
    # [CLASS DOC]
    # Benchmarks splitting strings on a delimiter, parameterized over dtype and
    # whether to expand the result into columns. The inherited series is
    # modified in setup so that each string contains the delimiter.
    params = (Dtypes.params, [True, False])
    param_names = ["dtype", "expand"]

    def setup(self, dtype, expand):
        # [FUNCTION DOC]
        # Runs the base setup, then replaces the series with one where the
        # characters of every string are joined by a double dash, creating
        # many delimiters. Parameter expand is unused here.
        super().setup(dtype)
        self.s = self.s.str.join("--")

    def time_split(self, dtype, expand):
        # [FUNCTION DOC]
        # Times splitting from the left on the double dash, expanding to columns
        # or not according to the flag.
        self.s.str.split("--", expand=expand)

    def time_rsplit(self, dtype, expand):
        # [FUNCTION DOC]
        # Times splitting from the right on the double dash, expanding to
        # columns or not according to the flag.
        self.s.str.rsplit("--", expand=expand)


class Extract(Dtypes):
    # [CLASS DOC]
    # Benchmarks regular expression group extraction with a single group,
    # parameterized over dtype and the expand flag.
    params = (Dtypes.params, [True, False])
    param_names = ["dtype", "expand"]

    def setup(self, dtype, expand):
        # [FUNCTION DOC]
        # Delegates to the base setup with only the dtype; expand is unused here.
        super().setup(dtype)

    def time_extract_single_group(self, dtype, expand):
        # [FUNCTION DOC]
        # Times extracting the group of word characters before a literal A,
        # with warnings recorded and suppressed; the expand flag controls
        # whether a frame or series is returned.
        with warnings.catch_warnings(record=True):
            self.s.str.extract("(\\w*)A", expand=expand)


class Dummies(Dtypes):
    # [CLASS DOC]
    # Benchmarks creating indicator (dummy) columns from delimited strings,
    # parameterized over dtype only (inherited).
    def setup(self, dtype):
        # [FUNCTION DOC]
        # Runs the base setup, then keeps the first fifth of the series with the
        # characters of each string joined by a vertical bar.
        super().setup(dtype)
        N = len(self.s) // 5
        self.s = self.s[:N].str.join("|")

    def time_get_dummies(self, dtype):
        # [FUNCTION DOC]
        # Times building the indicator frame by splitting on the vertical bar.
        self.s.str.get_dummies("|")


class Encode:
    # [CLASS DOC]
    # Benchmarks encoding strings to bytes and decoding them back. No
    # parameters. State set by setup: a series of 10000 distinct strings.
    def setup(self):
        # [FUNCTION DOC]
        # Builds an object-dtype series of 10000 distinct strings.
        self.ser = Series(Index([f"i-{i}" for i in range(10_000)], dtype=object))

    def time_encode_decode(self):
        # [FUNCTION DOC]
        # Times a round trip encoding to UTF-8 bytes then decoding back.
        self.ser.str.encode("utf-8").str.decode("utf-8")


class Slice:
    # [CLASS DOC]
    # Benchmarks slicing strings using the indexing syntax on the accessor.
    # State set by setup: a series alternating a seven-letter string and a
    # missing value, one million entries long.
    def setup(self):
        # [FUNCTION DOC]
        # Builds the series by repeating a pair (a string and a missing value)
        # 500000 times.
        self.s = Series(["abcdefg", np.nan] * 500000)

    def time_vector_slice(self):
        # [FUNCTION DOC]
        # Times taking the first five characters of each element through the
        # accessor's indexing syntax (regression benchmark for issue 2602).
        # GH 2602
        self.s.str[:5]


class Iter(Dtypes):
    # [CLASS DOC]
    # Benchmarks plain iteration over the inherited string series,
    # parameterized over dtype.
    def time_iter(self, dtype):
        # [FUNCTION DOC]
        # Times a loop visiting every element of the series without using it.
        for i in self.s:
            pass


class StringArrayConstruction:
    # [CLASS DOC]
    # Benchmarks constructing the string extension array from object arrays,
    # with and without missing values. State set by setup: an object array of
    # 100000 strings and a copy extended with 1000 missing markers.
    def setup(self):
        # [FUNCTION DOC]
        # Builds the object array of 100000 strings (counter text repeated ten
        # times) and a second array equal to the first followed by 1000
        # missing-value markers.
        self.series_arr = np.array([str(i) * 10 for i in range(10**5)], dtype=object)
        self.series_arr_nan = np.concatenate([self.series_arr, np.array([NA] * 1000)])

    def time_string_array_construction(self):
        # [FUNCTION DOC]
        # Times constructing a string array from the array without missing values.
        StringArray(self.series_arr, dtype=StringDtype())

    def time_string_array_with_nan_construction(self):
        # [FUNCTION DOC]
        # Times constructing a string array from the array containing missing
        # markers.
        StringArray(self.series_arr_nan, dtype=StringDtype())

    def peakmem_stringarray_construction(self):
        # [FUNCTION DOC]
        # Measures peak memory (not time) of constructing a string array from the
        # array without missing values.
        StringArray(self.series_arr, dtype=StringDtype())
