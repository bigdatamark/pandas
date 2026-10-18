"""
Benchmarks for code in pandas/_libs, excluding pandas/_libs/tslibs,
which has its own directory.

If a PR does not edit anything in _libs/, then it is unlikely that the
benchmarks will be affected.
"""

# [MODULE DOC]
# Benchmarks for low-level helpers of the compiled library: scalar/list-like
# predicates, a fast zip routine, dtype inference, and the cached read-only
# property decorator. Public contents are the module-level sample collections
# scalars (eleven scalar values of many types, including missing-value markers),
# zero_dims (a zero-dimensional text array) and listlikes (arrays, mappings, sets,
# lists, tuples), and the benchmark classes ScalarListLike, FastZip, InferDtype
# and CacheReadonly. Takes the compiled library handle from the shared helper
# module, and imports the cached-property decorator from the util package,
# falling back to its old location. Does not re-export the common asv setup hook.
import numpy as np

from pandas._libs.lib import (
    infer_dtype,
    is_list_like,
    is_scalar,
)

from pandas import (
    NA,
    Index,
    NaT,
)

from .pandas_vb_common import lib

try:
    from pandas.util import cache_readonly
except ImportError:
    from pandas.util.decorators import cache_readonly


# TODO: share with something in pd._testing?
scalars = [
    0,
    1.0,
    1 + 2j,
    True,
    "foo",
    b"bar",
    None,
    np.datetime64(123, "ns"),
    np.timedelta64(123, "ns"),
    NaT,
    NA,
]
zero_dims = [np.array("123")]
listlikes = [np.array([1, 2, 3]), {0: 1}, {1, 2, 3}, [1, 2, 3], (1, 2, 3)]


class ScalarListLike:
    # [CLASS DOC]
    # Benchmark of the scalar and list-like predicates over many sample values.
    # Parametrized by one sample value (scalars, zero-dimensional arrays and
    # list-like containers combined); there is no parameter name list and no
    # setup.
    params = scalars + zero_dims + listlikes

    def time_is_list_like(self, param):
        # [FUNCTION DOC]
        # Times the list-like predicate on the sample value.
        is_list_like(param)

    def time_is_scalar(self, param):
        # [FUNCTION DOC]
        # Times the scalar predicate on the sample value.
        is_scalar(param)


class FastZip:
    # [CLASS DOC]
    # Benchmark of the low-level fast zip routine over a list of three arrays. No
    # parameters. State: col_array_list.
    def setup(self):
        # [FUNCTION DOC]
        # With N equal to 10000 and K equal to 10: builds two object arrays of
        # string keys (each key repeated K times) and a random float array of N times
        # K entries, stacks them as three rows, and stores the list of the three
        # rows as col_array_list.
        # KNOWN ISSUE: a copy of the stacked array is made and its first 10000
        # columns are set to missing, but that copy is a local variable that is
        # never used, so this work has no effect on the benchmark.
        N = 10000
        K = 10
        key1 = Index([f"i-{i}" for i in range(N)], dtype=object).values.repeat(K)
        key2 = Index([f"i-{i}" for i in range(N)], dtype=object).values.repeat(K)
        col_array = np.vstack([key1, key2, np.random.randn(N * K)])
        col_array2 = col_array.copy()
        col_array2[:, :10000] = np.nan
        self.col_array_list = list(col_array)

    def time_lib_fast_zip(self):
        # [FUNCTION DOC]
        # Times the fast zip routine on the list of three arrays.
        lib.fast_zip(self.col_array_list)


class InferDtype:
    # [CLASS DOC]
    # Benchmark of dtype inference on eight prepared inputs, with and without
    # skipping missing values. Parametrized by input name. The class attribute
    # data_dict maps names (np-object, py-object, np-null, py-null, np-int,
    # np-floating, empty, bytes) to prepared containers of up to 100000 elements,
    # and params is the list of those names. The inputs are built when the module
    # is imported.
    param_names = ["dtype"]
    data_dict = {
        "np-object": np.array([1] * 100000, dtype="O"),
        "py-object": [1] * 100000,
        "np-null": np.array([1] * 50000 + [np.nan] * 50000),
        "py-null": [1] * 50000 + [None] * 50000,
        "np-int": np.array([1] * 100000, dtype=int),
        "np-floating": np.array([1.0] * 100000, dtype=float),
        "empty": [],
        "bytes": [b"a"] * 100000,
    }
    params = list(data_dict.keys())

    def time_infer_dtype_skipna(self, dtype):
        # [FUNCTION DOC]
        # Times dtype inference of the chosen input, ignoring missing values.
        infer_dtype(self.data_dict[dtype], skipna=True)

    def time_infer_dtype(self, dtype):
        # [FUNCTION DOC]
        # Times dtype inference of the chosen input, treating missing values as
        # significant.
        infer_dtype(self.data_dict[dtype], skipna=False)


class CacheReadonly:
    # [CLASS DOC]
    # Benchmark of reading a cached read-only property after its value is cached.
    # No parameters. State: obj (an instance of a small local class).
    def setup(self):
        # [FUNCTION DOC]
        # Defines a small local class with one property decorated as cached
        # read-only that returns five, and stores an instance of it as obj.
        class Foo:
            # [CLASS DOC]
            # HELPER: local class defined in the setup method to host a cached
            # read-only property. No state.
            @cache_readonly
            def prop(self):
                # [FUNCTION DOC]
                # HELPER: cached read-only property used by the cache benchmark.
                # Returns the constant five; the decorator caches the result on
                # the instance after the first access.
                return 5

        self.obj = Foo()

    def time_cache_readonly(self):
        # [FUNCTION DOC]
        # Times reading the cached property of obj.
        self.obj.prop
