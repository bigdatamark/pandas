# [MODULE DOC]
# Tests for the shared utilities in the core common module and a few related
# checks: callable-name lookup, random-state normalization, none-checks, name
# matching for binary operations, mapping standardization for dictionary conversion,
# version metadata sanity, picklability of public callables, boolean-indexer
# detection, temporary attribute setting, and string memory size at import.
import collections
from functools import partial
import string
import subprocess
import sys

import numpy as np
import pytest

from pandas.compat import WASM

import pandas as pd
from pandas import Series
import pandas._testing as tm
from pandas.core import ops
import pandas.core.common as com
from pandas.util.version import Version


class TestGetCallableName:
    # [CLASS DOC]
    # Tests the helper that returns a display name for a callable. Class attributes hold
    # a plain method, partials wrapping it (one nested), an anonymous function and a
    # callable class, which are used as parametrization inputs.
    def fn(self, x):
        # [FUNCTION DOC]
        # HELPER: sample method returning its argument; used (and wrapped by partials) as an input for the callable-name test.
        return x

    partial1 = partial(fn)
    partial2 = partial(partial1)
    lambda_ = lambda x: x

    class SomeCall:
        # [CLASS DOC]
        # HELPER: sample class whose instances are callable; the instance name should be reported as the class name.
        def __call__(self):
            # [FUNCTION DOC]
            # HELPER: raises a not-implemented error; never expected to be invoked by the test (only its type name matters).
            # This shouldn't actually get called below; SomeCall.__init__
            #  should.
            raise NotImplementedError

    @pytest.mark.parametrize(
        "func, expected",
        [
            (fn, "fn"),
            (partial1, "fn"),
            (partial2, "fn"),
            (lambda_, "<lambda>"),
            (SomeCall(), "SomeCall"),
            (1, None),
        ],
    )
    def test_get_callable_name(self, func, expected):
        # [FUNCTION DOC]
        # Parametrized over (callable, expected name): a function, a partial, a partial of a
        # partial (both report the underlying function name), an anonymous function (reports
        # the angle-bracket lambda name), a callable instance (reports its class name), and a
        # non-callable integer (reports None).
        assert com.get_callable_name(func) == expected


class TestRandomState:
    # [CLASS DOC]
    # Tests normalization of the many accepted random-state specifications into a random
    # number generator object.
    def test_seed(self):
        # [FUNCTION DOC]
        # Verifies an integer seed yields a generator producing the same first uniform draw as a
        # legacy NumPy random state created with the same seed.
        seed = 5
        assert com.random_state(seed).uniform() == np.random.RandomState(seed).uniform()

    def test_object(self):
        # [FUNCTION DOC]
        # Verifies passing an existing legacy random state returns an equivalent state (first
        # uniform draw equals that of a fresh state with the same seed).
        seed = 10
        state_obj = np.random.RandomState(seed)
        assert (
            com.random_state(state_obj).uniform()
            == np.random.RandomState(seed).uniform()
        )

    def test_default(self):
        # [FUNCTION DOC]
        # Verifies that with no argument the NumPy global random module itself is returned.
        assert com.random_state() is np.random

    def test_array_like(self):
        # [FUNCTION DOC]
        # Verifies an array of 624 unsigned 32-bit integers is accepted as seed material and
        # gives the same first uniform draw as a legacy random state seeded with the same array.
        state = np.random.default_rng(None).integers(0, 2**31, size=624, dtype="uint32")
        assert (
            com.random_state(state).uniform() == np.random.RandomState(state).uniform()
        )

    def test_bit_generators(self):
        # [FUNCTION DOC]
        # Verifies bit generator objects (Mersenne Twister with one seed, PCG64 with another)
        # are wrapped so the first uniform draw matches a legacy random state built from an
        # equivalent bit generator.
        seed = 3
        assert (
            com.random_state(np.random.MT19937(seed)).uniform()
            == np.random.RandomState(np.random.MT19937(seed)).uniform()
        )

        seed = 11
        assert (
            com.random_state(np.random.PCG64(seed)).uniform()
            == np.random.RandomState(np.random.PCG64(seed)).uniform()
        )

    @pytest.mark.parametrize("state", ["test", 5.5])
    def test_error(self, state):
        # [FUNCTION DOC]
        # Parametrized over an invalid string and a float. Verifies a value error listing the
        # accepted kinds (integer, array-like, bit generator, generator, random state, or None).
        msg = (
            "random_state must be an integer, array-like, a BitGenerator, Generator, "
            "a numpy RandomState, or None"
        )
        with pytest.raises(ValueError, match=msg):
            com.random_state(state)


@pytest.mark.parametrize("args, expected", [((1, 2, None), True), ((1, 2, 3), False)])
def test_any_none(args, expected):
    # [FUNCTION DOC]
    # Parametrized over argument tuples. Verifies the any-none helper returns exactly true when
    # some argument is None and false otherwise.
    assert com.any_none(*args) is expected


@pytest.mark.parametrize(
    "args, expected",
    [((1, 2, 3), True), ((1, 2, None), False), ((None, None, None), False)],
)
def test_all_not_none(args, expected):
    # [FUNCTION DOC]
    # Parametrized over argument tuples. Verifies the all-not-none helper returns true only when
    # no argument is None.
    assert com.all_not_none(*args) is expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (Series([1], name="x"), Series([2], name="x"), "x"),
        (Series([1], name="x"), Series([2], name="y"), None),
        (Series([1]), Series([2], name="x"), None),
        (Series([1], name="x"), Series([2]), None),
        (Series([1], name="x"), [2], "x"),
        ([1], Series([2], name="y"), "y"),
        # matching NAs
        (Series([1], name=np.nan), pd.Index([], name=np.nan), np.nan),
        (Series([1], name=np.nan), pd.Index([], name=pd.NaT), None),
        (Series([1], name=pd.NA), pd.Index([], name=pd.NA), pd.NA),
        # tuple name GH#39757
        (
            Series([1], name=np.int64(1)),
            pd.Index([], name=(np.int64(1), np.int64(2))),
            None,
        ),
        (
            Series([1], name=(np.int64(1), np.int64(2))),
            pd.Index([], name=(np.int64(1), np.int64(2))),
            (np.int64(1), np.int64(2)),
        ),
        pytest.param(
            Series([1], name=(np.float64("nan"), np.int64(2))),
            pd.Index([], name=(np.float64("nan"), np.int64(2))),
            (np.float64("nan"), np.int64(2)),
            marks=pytest.mark.xfail(
                reason="Not checking for matching NAs inside tuples."
            ),
        ),
    ],
)
def test_maybe_match_name(left, right, expected):
    # [FUNCTION DOC]
    # Parametrized over (left, right, expected name) covering Series and Index and plain lists:
    # equal names are kept, differing or missing names give None, a list contributes no name,
    # missing-value names match when both are the same kind of missing value (NaN with NaN,
    # NA with NA) but not NaN with NaT, and tuple names (GitHub issue) match only when equal.
    # One case with NaN inside a tuple name is marked as an expected failure because missing
    # values inside tuples are not compared as matching. Result must be identical to or equal
    # the expected value.
    res = ops.common._maybe_match_name(left, right)
    assert res is expected or res == expected


@pytest.mark.parametrize(
    "into, msg",
    [
        (
            # uninitialized defaultdict
            collections.defaultdict,
            r"to_dict\(\) only accepts initialized defaultdicts",
        ),
        (
            # non-mapping subtypes,, instance
            [],
            "unsupported type: <class 'list'>",
        ),
        (
            # non-mapping subtypes, class
            list,
            "unsupported type: <class 'list'>",
        ),
    ],
)
def test_standardize_mapping_type_error(into, msg):
    # [FUNCTION DOC]
    # Parametrized over an uninitialized defaultdict class, an empty list instance, and the list
    # class. Verifies a type error: for the defaultdict class a message saying the dictionary
    # conversion only accepts initialized defaultdicts, otherwise an "unsupported type" message.
    with pytest.raises(TypeError, match=msg):
        com.standardize_mapping(into)


def test_standardize_mapping():
    # [FUNCTION DOC]
    # Verifies a plain mapping instance or class is standardized to the dict type, and an
    # initialized defaultdict becomes a partial object (a factory that builds that defaultdict).
    fill = {"bad": "data"}
    assert com.standardize_mapping(fill) == dict

    # Convert instance to type
    assert com.standardize_mapping({}) == dict

    dd = collections.defaultdict(list)
    assert isinstance(com.standardize_mapping(dd), partial)


def test_git_version():
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Verifies the package's git version string is exactly
    # forty characters, all hexadecimal digits.
    # GH 21295
    git_version = pd.__git_version__
    assert len(git_version) == 40
    assert all(c in string.hexdigits for c in git_version)


def test_version_tag():
    # [FUNCTION DOC]
    # Parses the package version and compares it against an old version; if comparison raises
    # a type error (no git tags in the checkout) re-raises a value error telling the developer
    # to sync tags between upstream and their repository.
    version = Version(pd.__version__)
    try:
        version > Version("0.0.1")
    except TypeError as err:
        raise ValueError(
            "No git tags exist, please sync tags between upstream and your repo"
        ) from err


@pytest.mark.parametrize("obj", [obj for obj in pd.__dict__.values() if callable(obj)])
def test_serializable(obj, temp_file):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over every callable found in the pandas
    # top-level namespace. Verifies a round trip through pickle (writing to a temporary file
    # fixture) gives an object of the same type.
    # GH 35611
    unpickled = tm.round_trip_pickle(obj, temp_file)
    assert type(obj) == type(unpickled)


class TestIsBoolIndexer:
    # [CLASS DOC]
    # Tests the helper that decides whether an object should be treated as a boolean indexer.
    def test_non_bool_array_with_na(self):
        # [FUNCTION DOC]
        # Verifies an object-dtype array of strings and NaN is not a boolean indexer and, in
        # particular, does not raise.
        # in particular, this should not raise
        arr = np.array(["A", "B", np.nan], dtype=object)
        assert not com.is_bool_indexer(arr)

    def test_list_subclass(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Verifies a list subclass holding a string is not a
        # boolean indexer, while one holding a boolean is.
        # GH#42433

        class MyList(list):
            # [CLASS DOC]
            # HELPER: trivial list subclass used to test indexer detection for subclasses.
            pass

        val = MyList(["a"])

        assert not com.is_bool_indexer(val)

        val = MyList([True])
        assert com.is_bool_indexer(val)

    def test_frozenlist(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Verifies the frozen list of index names is not
        # treated as a boolean indexer, so selecting columns with it returns an empty-column frame
        # identical to selecting with an empty list.
        # GH#42461
        data = {"col1": [1, 2], "col2": [3, 4]}
        df = pd.DataFrame(data=data)

        frozen = df.index.names[1:]
        assert not com.is_bool_indexer(frozen)

        result = df[frozen]
        expected = df[[]]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("scalar", [1, True])
    def test_numpyextensionarray(self, scalar):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Parametrized over an integer and a boolean scalar.
        # Verifies a NumPy-backed extension array is a boolean indexer exactly when the stored
        # scalar is boolean.
        # GH 63391
        arr = pd.arrays.NumpyExtensionArray(np.array([scalar]))
        assert com.is_bool_indexer(arr) is isinstance(scalar, bool)


@pytest.mark.parametrize("with_exception", [True, False])
def test_temp_setattr(with_exception):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized on whether an error is raised inside the
    # context. Verifies the temporary-attribute context manager sets the attribute inside the
    # block, restores the original value afterward whether or not the body raised, and lets
    # the exception propagate (a value error is raised in both variants only to satisfy the
    # raises check; the message identifies where).
    # GH#45954
    ser = Series(dtype=object)
    ser.name = "first"
    # Raise a ValueError in either case to satisfy pytest.raises
    match = "Inside exception raised" if with_exception else "Outside exception raised"
    with pytest.raises(ValueError, match=match):
        with com.temp_setattr(ser, "name", "second"):
            assert ser.name == "second"
            if with_exception:
                raise ValueError("Inside exception raised")
        raise ValueError("Outside exception raised")
    assert ser.name == "first"


@pytest.mark.skipif(WASM, reason="Can't start subprocesses in WASM")
@pytest.mark.single_cpu
def test_str_size():
    # [FUNCTION DOC]
    # Skipped on WebAssembly and marked single-CPU. Regression test for a GitHub issue. Runs a
    # subprocess interpreter that measures the memory size of a short string both before and
    # after importing pandas and prints the latter; verifies the printed size, taken from the
    # last few characters of output, equals the size measured in the test process, i.e. importing
    # pandas does not change string size.
    # GH#21758
    a = "a"
    expected = sys.getsizeof(a)
    pyexe = sys.executable.replace("\\", "/")
    call = [
        pyexe,
        "-c",
        "a='a';import sys;sys.getsizeof(a);import pandas;print(sys.getsizeof(a));",
    ]
    result = subprocess.check_output(call).decode()[-4:-1].strip("\n")
    assert int(result) == int(expected)
