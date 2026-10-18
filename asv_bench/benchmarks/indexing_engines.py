"""
Benchmarks in this file depend mostly on code in _libs/

We have to created masked arrays to test the masked engine though. The
array is unpacked on the Cython level.

If a PR does not edit anything in _libs, it is very unlikely that benchmarks
in this file will be affected.
"""

# [MODULE DOC]
# Benchmarks for the low-level index lookup engines of the compiled library,
# including the masked-array engines and the object engine. Public contents are
# the benchmark classes NumericEngineIndexing, MaskedNumericEngineIndexing and
# ObjectEngineIndexing and the private helper functions _get_numeric_engines and
# _get_masked_engines. Imports the compiled index module and the masked array base
# class. Import-time side effects: the engine helper functions are called while
# the benchmark classes are defined, to build their parameter lists, by looking up
# engine classes by name on the compiled module (missing names are silently
# dropped).
import numpy as np

from pandas._libs import index as libindex

from pandas.core.arrays import BaseMaskedArray


def _get_numeric_engines():
    # [FUNCTION DOC]
    # HELPER: builds the engine parameter list for NumericEngineIndexing.
    # Takes no input. Returns a list of pairs of an engine class and the numpy
    # dtype it handles, for the signed and unsigned integer widths and the two
    # float widths, including only those engine names that exist on the compiled
    # index module.
    # KNOWN ISSUE: the unsigned 16-bit entry is spelled with a lowercase e in the
    # engine class name, but the real class name uses an uppercase E (per the
    # type stubs), so this dtype is silently dropped by the existence check and
    # never benchmarked.
    engine_names = [
        ("Int64Engine", np.int64),
        ("Int32Engine", np.int32),
        ("Int16Engine", np.int16),
        ("Int8Engine", np.int8),
        ("UInt64Engine", np.uint64),
        ("UInt32Engine", np.uint32),
        ("UInt16engine", np.uint16),
        ("UInt8Engine", np.uint8),
        ("Float64Engine", np.float64),
        ("Float32Engine", np.float32),
    ]
    return [
        (getattr(libindex, engine_name), dtype)
        for engine_name, dtype in engine_names
        if hasattr(libindex, engine_name)
    ]


def _get_masked_engines():
    # [FUNCTION DOC]
    # HELPER: builds the engine parameter list for MaskedNumericEngineIndexing.
    # Takes no input. Returns a list of pairs of a masked engine class and the
    # nullable dtype name it handles, including only engine names present on the
    # compiled index module.
    # KNOWN ISSUE: the unsigned 16-bit entry is spelled with a lowercase e in the
    # engine class name, but the real class name uses an uppercase E (per the
    # type stubs), so this dtype is silently dropped by the existence check and
    # never benchmarked.
    engine_names = [
        ("MaskedInt64Engine", "Int64"),
        ("MaskedInt32Engine", "Int32"),
        ("MaskedInt16Engine", "Int16"),
        ("MaskedInt8Engine", "Int8"),
        ("MaskedUInt64Engine", "UInt64"),
        ("MaskedUInt32Engine", "UInt32"),
        ("MaskedUInt16engine", "UInt16"),
        ("MaskedUInt8Engine", "UInt8"),
        ("MaskedFloat64Engine", "Float64"),
        ("MaskedFloat32Engine", "Float32"),
    ]
    return [
        (getattr(libindex, engine_name), dtype)
        for engine_name, dtype in engine_names
        if hasattr(libindex, engine_name)
    ]


class NumericEngineIndexing:
    # [CLASS DOC]
    # Benchmark of label lookup in numeric index engines for monotonic increasing,
    # monotonic decreasing and non-monotonic data, unique or repeated, at two
    # sizes (the larger is above the engine's size cutoff). Parametrized by
    # (engine, dtype) pair, index type, unique flag and N. State: data (the
    # engine), key_middle and key_early.
    params = [
        _get_numeric_engines(),
        ["monotonic_incr", "monotonic_decr", "non_monotonic"],
        [True, False],
        [10**5, 2 * 10**6],  # 2e6 is above SIZE_CUTOFF
    ]
    param_names = ["engine_and_dtype", "index_type", "unique", "N"]

    def setup(self, engine_and_dtype, index_type, unique, N):
        # [FUNCTION DOC]
        # Builds a key array of length three times N (or N repeated three times
        # for non-unique data) and wraps it in the engine. Raises a not-implemented
        # error for non-monotonic unique data in 8- and 16-bit dtypes (values
        # would overflow; the check lists signed 16-bit, signed 8-bit and unsigned
        # 8-bit). Layouts: monotonic increasing is a range or the values one, two,
        # three each repeated N times; monotonic decreasing is the reverse of
        # those; non-monotonic unique data places the top third of the range
        # first followed by the lower two thirds, and non-unique data is the
        # same as the monotonic increasing repeated pattern. Performs one lookup
        # to build the lazy mapping so it is not included in timings. Stores
        # key_middle (the middle element) and key_early (the element at position
        # two).
        engine, dtype = engine_and_dtype

        if (
            index_type == "non_monotonic"
            and dtype in [np.int16, np.int8, np.uint8]
            and unique
        ):
            # Values overflow
            raise NotImplementedError

        if index_type == "monotonic_incr":
            if unique:
                arr = np.arange(N * 3, dtype=dtype)
            else:
                arr = np.array([1, 2, 3], dtype=dtype).repeat(N)
        elif index_type == "monotonic_decr":
            if unique:
                arr = np.arange(N * 3, dtype=dtype)[::-1]
            else:
                arr = np.array([3, 2, 1], dtype=dtype).repeat(N)
        else:
            assert index_type == "non_monotonic"
            if unique:
                arr = np.empty(N * 3, dtype=dtype)
                arr[:N] = np.arange(N * 2, N * 3, dtype=dtype)
                arr[N:] = np.arange(N * 2, dtype=dtype)
            else:
                arr = np.array([1, 2, 3], dtype=dtype).repeat(N)

        self.data = engine(arr)
        # code below avoids populating the mapping etc. while timing.
        self.data.get_loc(2)

        self.key_middle = arr[len(arr) // 2]
        self.key_early = arr[2]

    def time_get_loc(self, engine_and_dtype, index_type, unique, N):
        # [FUNCTION DOC]
        # Times looking up the early key (the element at position two) in the
        # engine. Parameters are unused here.
        self.data.get_loc(self.key_early)

    def time_get_loc_near_middle(self, engine_and_dtype, index_type, unique, N):
        # [FUNCTION DOC]
        # Times looking up the middle key, since binary search performance may
        # differ near the middle of a range versus near an endpoint. Parameters are
        # unused here.
        # searchsorted performance may be different near the middle of a range
        #  vs near an endpoint
        self.data.get_loc(self.key_middle)


class MaskedNumericEngineIndexing:
    # [CLASS DOC]
    # Same as the numeric engine benchmark but for the masked (nullable) engines,
    # where each engine is built from a masked array of data and mask. Parametrized
    # by (engine, nullable dtype name) pair, index type, unique flag and N. State:
    # data (the engine), key_middle and key_early.
    params = [
        _get_masked_engines(),
        ["monotonic_incr", "monotonic_decr", "non_monotonic"],
        [True, False],
        [10**5, 2 * 10**6],  # 2e6 is above SIZE_CUTOFF
    ]
    param_names = ["engine_and_dtype", "index_type", "unique", "N"]

    def setup(self, engine_and_dtype, index_type, unique, N):
        # [FUNCTION DOC]
        # Converts the nullable dtype name to the lowercase numpy name. Raises a
        # not-implemented error for non-monotonic unique data in 16-bit signed,
        # 8-bit signed and 8-bit unsigned dtypes (overflow). Builds the data array
        # in the same layouts as the numeric benchmark (for the non-monotonic
        # unique case it starts from zeros and fills in the two ranges). Builds an
        # all-false mask, except that for non-monotonic data the last element is
        # masked as missing. Wraps data and mask in the masked array base class
        # and passes it to the engine. Performs one lookup to build the lazy
        # mapping, and stores key_middle and key_early taken from the data array.
        engine, dtype = engine_and_dtype
        dtype = dtype.lower()

        if (
            index_type == "non_monotonic"
            and dtype in ["int16", "int8", "uint8"]
            and unique
        ):
            # Values overflow
            raise NotImplementedError

        if index_type == "monotonic_incr":
            if unique:
                arr = np.arange(N * 3, dtype=dtype)
            else:
                arr = np.array([1, 2, 3], dtype=dtype).repeat(N)
            mask = np.zeros(N * 3, dtype=np.bool_)
        elif index_type == "monotonic_decr":
            if unique:
                arr = np.arange(N * 3, dtype=dtype)[::-1]
            else:
                arr = np.array([3, 2, 1], dtype=dtype).repeat(N)
            mask = np.zeros(N * 3, dtype=np.bool_)
        else:
            assert index_type == "non_monotonic"
            if unique:
                arr = np.zeros(N * 3, dtype=dtype)
                arr[:N] = np.arange(N * 2, N * 3, dtype=dtype)
                arr[N:] = np.arange(N * 2, dtype=dtype)

            else:
                arr = np.array([1, 2, 3], dtype=dtype).repeat(N)
            mask = np.zeros(N * 3, dtype=np.bool_)
            mask[-1] = True

        self.data = engine(BaseMaskedArray(arr, mask))
        # code below avoids populating the mapping etc. while timing.
        self.data.get_loc(2)

        self.key_middle = arr[len(arr) // 2]
        self.key_early = arr[2]

    def time_get_loc(self, engine_and_dtype, index_type, unique, N):
        # [FUNCTION DOC]
        # Times looking up the early key in the masked engine. Parameters are
        # unused here.
        self.data.get_loc(self.key_early)

    def time_get_loc_near_middle(self, engine_and_dtype, index_type, unique, N):
        # [FUNCTION DOC]
        # Times looking up the middle key in the masked engine, since search
        # performance may differ near the middle versus near an endpoint.
        # Parameters are unused here.
        # searchsorted performance may be different near the middle of a range
        #  vs near an endpoint
        self.data.get_loc(self.key_middle)


class ObjectEngineIndexing:
    # [CLASS DOC]
    # Benchmark of label lookup in the object engine for string data that is
    # monotonic increasing, monotonic decreasing or non-monotonic. Parametrized by
    # index type. State: data (the engine).
    params = [("monotonic_incr", "monotonic_decr", "non_monotonic")]
    param_names = ["index_type"]

    def setup(self, index_type):
        # [FUNCTION DOC]
        # Builds an object array of 100000 a values, 100000 b values and 100000 c
        # values; for the chosen index type uses it as given, reversed, or as the
        # pattern abc repeated 100000 times. Wraps it in the object engine and
        # performs one lookup of the label b so the lazy mapping is built outside
        # the timing. Stores the engine as data.
        N = 10**5
        values = list("a" * N + "b" * N + "c" * N)
        arr = {
            "monotonic_incr": np.array(values, dtype=object),
            "monotonic_decr": np.array(list(reversed(values)), dtype=object),
            "non_monotonic": np.array(list("abc") * N, dtype=object),
        }[index_type]

        self.data = libindex.ObjectEngine(arr)
        # code below avoids populating the mapping etc. while timing.
        self.data.get_loc("b")

    def time_get_loc(self, index_type):
        # [FUNCTION DOC]
        # Times looking up the label b in the object engine. The parameter is
        # unused here.
        self.data.get_loc("b")
