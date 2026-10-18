# [MODULE DOC]
# Tests for the NaN-aware reduction functions in the core nanops module (any, all, sum, mean, median, variance,
# standard deviation, standard error, min, max, argmax, argmin, skew, kurtosis, product, correlation, covariance) and
# its helpers (infinity detection, bottleneck-eligibility by dtype, numeric coercion, minimum-count check). Provides a
# large set of array fixtures (float, complex, integer, boolean, string, datetime, timedelta, NaN-only, infinite, and
# stacked combinations) and a class that rebuilds the same arrays as instance attributes for comparison against NumPy
# and SciPy reference functions. Import-time side effect: stores the module's bottleneck-usage flag in a module-level
# variable so the class teardown can restore it.
from functools import partial

import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas.core.dtypes.common import is_integer_dtype

import pandas as pd
from pandas import (
    Series,
    isna,
)
import pandas._testing as tm
from pandas.core import nanops

use_bn = nanops._USE_BOTTLENECK


@pytest.fixture
def disable_bottleneck(monkeypatch):
    # [FUNCTION DOC]
    # HELPER: fixture. Within a monkeypatch context sets the nanops module's bottleneck-use flag to false for the duration of the test, then restores it.
    with monkeypatch.context() as m:
        m.setattr(nanops, "_USE_BOTTLENECK", False)
        yield


@pytest.fixture
def arr_shape():
    # [FUNCTION DOC]
    # HELPER: fixture returning the common array shape, 11 rows by 7 columns.
    return 11, 7


@pytest.fixture
def arr_float(arr_shape):
    # [FUNCTION DOC]
    # HELPER: fixture; random standard-normal float array of the common shape.
    return np.random.default_rng(2).standard_normal(arr_shape)


@pytest.fixture
def arr_complex(arr_float):
    # [FUNCTION DOC]
    # HELPER: fixture; complex array whose real and imaginary parts both equal the float array.
    return arr_float + arr_float * 1j


@pytest.fixture
def arr_int(arr_shape):
    # [FUNCTION DOC]
    # HELPER: fixture; random integers from minus ten to nine of the common shape.
    return np.random.default_rng(2).integers(-10, 10, arr_shape)


@pytest.fixture
def arr_bool(arr_shape):
    # [FUNCTION DOC]
    # HELPER: fixture; random boolean array of the common shape.
    return np.random.default_rng(2).integers(0, 2, arr_shape) == 0


@pytest.fixture
def arr_str(arr_float):
    # [FUNCTION DOC]
    # HELPER: fixture; absolute values of the float array converted to a byte-string dtype.
    return np.abs(arr_float).astype("S")


@pytest.fixture
def arr_utf(arr_float):
    # [FUNCTION DOC]
    # HELPER: fixture; absolute values of the float array converted to a unicode string dtype.
    return np.abs(arr_float).astype("U")


@pytest.fixture
def arr_date(arr_shape):
    # [FUNCTION DOC]
    # HELPER: fixture; random integers up to 20000 interpreted as nanosecond datetimes.
    return np.random.default_rng(2).integers(0, 20000, arr_shape).astype("M8[ns]")


@pytest.fixture
def arr_tdelta(arr_shape):
    # [FUNCTION DOC]
    # HELPER: fixture; random integers up to 20000 interpreted as nanosecond timedeltas.
    return np.random.default_rng(2).integers(0, 20000, arr_shape).astype("m8[ns]")


@pytest.fixture
def arr_nan(arr_shape):
    # [FUNCTION DOC]
    # HELPER: fixture; array of the common shape filled with NaN.
    return np.tile(np.nan, arr_shape)


@pytest.fixture
def arr_float_nan(arr_float, arr_nan):
    # [FUNCTION DOC]
    # HELPER: fixture; the float array stacked vertically on the all-NaN array.
    return np.vstack([arr_float, arr_nan])


@pytest.fixture
def arr_nan_float1(arr_nan, arr_float):
    # [FUNCTION DOC]
    # HELPER: fixture; the all-NaN array stacked on top of the float array (named for a second float array but built from the first one).
    return np.vstack([arr_nan, arr_float])


@pytest.fixture
def arr_nan_nan(arr_nan):
    # [FUNCTION DOC]
    # HELPER: fixture; two all-NaN arrays stacked.
    return np.vstack([arr_nan, arr_nan])


@pytest.fixture
def arr_inf(arr_float):
    # [FUNCTION DOC]
    # HELPER: fixture; the float array multiplied by infinity (values are plus or minus infinity, or NaN where a value is exactly zero).
    return arr_float * np.inf


@pytest.fixture
def arr_float_inf(arr_float, arr_inf):
    # [FUNCTION DOC]
    # HELPER: fixture; float array stacked on the infinity array.
    return np.vstack([arr_float, arr_inf])


@pytest.fixture
def arr_nan_inf(arr_nan, arr_inf):
    # [FUNCTION DOC]
    # HELPER: fixture; NaN array stacked on the infinity array.
    return np.vstack([arr_nan, arr_inf])


@pytest.fixture
def arr_float_nan_inf(arr_float, arr_nan, arr_inf):
    # [FUNCTION DOC]
    # HELPER: fixture; float, NaN and infinity arrays stacked.
    return np.vstack([arr_float, arr_nan, arr_inf])


@pytest.fixture
def arr_nan_nan_inf(arr_nan, arr_inf):
    # [FUNCTION DOC]
    # HELPER: fixture; two NaN arrays then the infinity array stacked.
    return np.vstack([arr_nan, arr_nan, arr_inf])


@pytest.fixture
def arr_obj(
    arr_float, arr_int, arr_bool, arr_complex, arr_str, arr_utf, arr_date, arr_tdelta
):
    # [FUNCTION DOC]
    # HELPER: fixture; object-dtype array stacking the object-cast versions of the float, integer, boolean, complex, byte-string, unicode-string, datetime and timedelta arrays.
    return np.vstack(
        [
            arr_float.astype("O"),
            arr_int.astype("O"),
            arr_bool.astype("O"),
            arr_complex.astype("O"),
            arr_str.astype("O"),
            arr_utf.astype("O"),
            arr_date.astype("O"),
            arr_tdelta.astype("O"),
        ]
    )


@pytest.fixture
def arr_nan_nanj(arr_nan):
    # [FUNCTION DOC]
    # HELPER: fixture; complex array whose real and imaginary parts are NaN, computed with invalid-value warnings suppressed.
    with np.errstate(invalid="ignore"):
        return arr_nan + arr_nan * 1j


@pytest.fixture
def arr_complex_nan(arr_complex, arr_nan_nanj):
    # [FUNCTION DOC]
    # HELPER: fixture; complex array stacked on the NaN-complex array (warnings suppressed).
    with np.errstate(invalid="ignore"):
        return np.vstack([arr_complex, arr_nan_nanj])


@pytest.fixture
def arr_nan_infj(arr_inf):
    # [FUNCTION DOC]
    # HELPER: fixture; the infinity array multiplied by the imaginary unit, giving complex values with infinite and NaN parts (warnings suppressed).
    with np.errstate(invalid="ignore"):
        return arr_inf * 1j


@pytest.fixture
def arr_complex_nan_infj(arr_complex, arr_nan_infj):
    # [FUNCTION DOC]
    # HELPER: fixture; complex array stacked on the infinite-imaginary array (warnings suppressed).
    with np.errstate(invalid="ignore"):
        return np.vstack([arr_complex, arr_nan_infj])


@pytest.fixture
def arr_float_1d(arr_float):
    # [FUNCTION DOC]
    # HELPER: fixture; first column of the float array.
    return arr_float[:, 0]


@pytest.fixture
def arr_nan_1d(arr_nan):
    # [FUNCTION DOC]
    # HELPER: fixture; first column of the NaN array.
    return arr_nan[:, 0]


@pytest.fixture
def arr_float_nan_1d(arr_float_nan):
    # [FUNCTION DOC]
    # HELPER: fixture; first column of the float-then-NaN array.
    return arr_float_nan[:, 0]


@pytest.fixture
def arr_float1_nan_1d(arr_float1_nan):
    # [FUNCTION DOC]
    # HELPER: fixture; first column of a fixture named for a second float array stacked on NaN.
    # KNOWN ISSUE: this fixture depends on a module-level fixture that does not exist (the second-float-array fixtures are defined only as instance attributes in the class), so requesting this fixture would fail; it appears to be unused.
    return arr_float1_nan[:, 0]


@pytest.fixture
def arr_nan_float1_1d(arr_nan_float1):
    # [FUNCTION DOC]
    # HELPER: fixture; first column of the NaN-then-float array.
    return arr_nan_float1[:, 0]


class TestnanopsDataFrame:
    # [CLASS DOC]
    # Compares each nanops reduction against a NumPy or SciPy reference over many array types, with and without skipping NaNs, with bottleneck disabled. State is set up per test as instance attributes: float arrays (two independent-named but identically seeded draws), complex, integer, boolean, string, datetime, timedelta, NaN, infinity and stacked combinations, plus two-dimensional and one-dimensional views. Contains generic comparison helpers (check_*) and the per-function tests.
    def setup_method(self):
        # [FUNCTION DOC]
        # Turns off the module's bottleneck flag and builds all the array attributes: random float, second float, complex, integer, boolean, byte-string, unicode, datetime and timedelta arrays of shape 11 by 7, NaN-filled and infinity arrays, various vertical stacks of them, an object array, complex NaN variants (warnings suppressed), two-dimensional aliases and first-column one-dimensional views.
        # KNOWN ISSUE: both float arrays are generated from generators created with the same seed, so the second float array is identical to the first rather than independent; correlation and covariance tests therefore compare an array with itself.
        nanops._USE_BOTTLENECK = False

        arr_shape = (11, 7)

        self.arr_float = np.random.default_rng(2).standard_normal(arr_shape)
        self.arr_float1 = np.random.default_rng(2).standard_normal(arr_shape)
        self.arr_complex = self.arr_float + self.arr_float1 * 1j
        self.arr_int = np.random.default_rng(2).integers(-10, 10, arr_shape)
        self.arr_bool = np.random.default_rng(2).integers(0, 2, arr_shape) == 0
        self.arr_str = np.abs(self.arr_float).astype("S")
        self.arr_utf = np.abs(self.arr_float).astype("U")
        self.arr_date = (
            np.random.default_rng(2).integers(0, 20000, arr_shape).astype("M8[ns]")
        )
        self.arr_tdelta = (
            np.random.default_rng(2).integers(0, 20000, arr_shape).astype("m8[ns]")
        )

        self.arr_nan = np.tile(np.nan, arr_shape)
        self.arr_float_nan = np.vstack([self.arr_float, self.arr_nan])
        self.arr_float1_nan = np.vstack([self.arr_float1, self.arr_nan])
        self.arr_nan_float1 = np.vstack([self.arr_nan, self.arr_float1])
        self.arr_nan_nan = np.vstack([self.arr_nan, self.arr_nan])

        self.arr_inf = self.arr_float * np.inf
        self.arr_float_inf = np.vstack([self.arr_float, self.arr_inf])

        self.arr_nan_inf = np.vstack([self.arr_nan, self.arr_inf])
        self.arr_float_nan_inf = np.vstack([self.arr_float, self.arr_nan, self.arr_inf])
        self.arr_nan_nan_inf = np.vstack([self.arr_nan, self.arr_nan, self.arr_inf])
        self.arr_obj = np.vstack(
            [
                self.arr_float.astype("O"),
                self.arr_int.astype("O"),
                self.arr_bool.astype("O"),
                self.arr_complex.astype("O"),
                self.arr_str.astype("O"),
                self.arr_utf.astype("O"),
                self.arr_date.astype("O"),
                self.arr_tdelta.astype("O"),
            ]
        )

        with np.errstate(invalid="ignore"):
            self.arr_nan_nanj = self.arr_nan + self.arr_nan * 1j
            self.arr_complex_nan = np.vstack([self.arr_complex, self.arr_nan_nanj])

            self.arr_nan_infj = self.arr_inf * 1j
            self.arr_complex_nan_infj = np.vstack([self.arr_complex, self.arr_nan_infj])

        self.arr_float_2d = self.arr_float
        self.arr_float1_2d = self.arr_float1

        self.arr_nan_2d = self.arr_nan
        self.arr_float_nan_2d = self.arr_float_nan
        self.arr_float1_nan_2d = self.arr_float1_nan
        self.arr_nan_float1_2d = self.arr_nan_float1

        self.arr_float_1d = self.arr_float[:, 0]
        self.arr_float1_1d = self.arr_float1[:, 0]

        self.arr_nan_1d = self.arr_nan[:, 0]
        self.arr_float_nan_1d = self.arr_float_nan[:, 0]
        self.arr_float1_nan_1d = self.arr_float1_nan[:, 0]
        self.arr_nan_float1_1d = self.arr_nan_float1[:, 0]

    def teardown_method(self):
        # [FUNCTION DOC]
        # Restores the module's bottleneck flag to the value recorded at import.
        nanops._USE_BOTTLENECK = use_bn

    def check_results(self, targ, res, axis, check_dtype=True):
        # [FUNCTION DOC]
        # HELPER: compares a reference result with the function result. Parameters: targ, the reference; res, the result; axis, the reduction axis; check_dtype, whether dtype must match (default true). Unwraps datetime-like results to their raw values; if the axis is not the first and shapes differ, trims the result to the reference's leading length. Asserts approximate equality. On failure it re-raises for timedelta references and for results that are neither complex nor object; otherwise (rounding differences) casts object results to the reference dtype or to complex, raising if the reference is object while the result is not, and compares real and imaginary parts separately.
        res = getattr(res, "asm8", res)

        if (
            axis != 0
            and hasattr(targ, "shape")
            and targ.ndim
            and targ.shape != res.shape
        ):
            res = np.split(res, [targ.shape[0]], axis=0)[0]

        try:
            tm.assert_almost_equal(targ, res, check_dtype=check_dtype)
        except AssertionError:
            # handle timedelta dtypes
            if hasattr(targ, "dtype") and targ.dtype == "m8[ns]":
                raise

            # There are sometimes rounding errors with
            # complex and object dtypes.
            # If it isn't one of those, re-raise the error.
            if not hasattr(res, "dtype") or res.dtype.kind not in ["c", "O"]:
                raise
            # convert object dtypes to something that can be split into
            # real and imaginary parts
            if res.dtype.kind == "O":
                if targ.dtype.kind != "O":
                    res = res.astype(targ.dtype)
                else:
                    cast_dtype = "c16" if hasattr(np, "complex128") else "f8"
                    res = res.astype(cast_dtype)
                    targ = targ.astype(cast_dtype)
            # there should never be a case where numpy returns an object
            # but nanops doesn't, so make that an exception
            elif targ.dtype.kind == "O":
                raise
            tm.assert_almost_equal(np.real(targ), np.real(res), check_dtype=check_dtype)
            tm.assert_almost_equal(np.imag(targ), np.imag(res), check_dtype=check_dtype)

    def check_fun_data(
        self,
        testfunc,
        targfunc,
        testar,
        testarval,
        targarval,
        skipna,
        check_dtype=True,
        empty_targfunc=None,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # HELPER: for one array, compares the nanops function against the reference function over each axis and over no axis. Parameters: testfunc, the nanops function; targfunc, the reference; testar, the fixture attribute name; testarval, the array to test; targarval, the array used for the reference (differs when the name ends with a NaN marker, so the reference ignores the NaN rows); skipna, whether to skip NaNs; check_dtype, empty_targfunc (reference used when all values are missing and skipping, so empty sums give zero), plus extra keyword arguments passed to both functions. For logical any/all on object arrays coerces the reference to boolean. For argmax/argmin on arrays containing NaN (names starting with the NaN marker, or ending with it when not skipping or on axis one) expects a value error about NA values and returns immediately. Handles a complex-NaN reference versus a float NaN result. Also calls the function with default skip argument and with no axis. Then recurses on the array with its last axis reduced to its first element until it is one-dimensional.
        # KNOWN ISSUE: the early return after the argmax/argmin value-error check exits the entire function (including the remaining axes and the recursion on lower dimensions) after the first matching axis, so those cases are checked less thoroughly than other functions.
        for axis in [*list(range(targarval.ndim)), None]:
            targartempval = targarval if skipna else testarval
            if skipna and empty_targfunc and isna(targartempval).all():
                targ = empty_targfunc(targartempval, axis=axis, **kwargs)
            else:
                targ = targfunc(targartempval, axis=axis, **kwargs)

            if targartempval.dtype == object and (
                targfunc is np.any or targfunc is np.all
            ):
                # GH#12863 the numpy functions will retain e.g. floatiness
                if isinstance(targ, np.ndarray):
                    targ = targ.astype(bool)
                else:
                    targ = bool(targ)

            if testfunc.__name__ in ["nanargmax", "nanargmin"] and (
                testar.startswith("arr_nan")
                or (testar.endswith("nan") and (not skipna or axis == 1))
            ):
                with pytest.raises(ValueError, match="Encountered .* NA value"):
                    testfunc(testarval, axis=axis, skipna=skipna, **kwargs)
                return
            res = testfunc(testarval, axis=axis, skipna=skipna, **kwargs)

            if (
                isinstance(targ, np.complex128)
                and isinstance(res, float)
                and np.isnan(targ)
                and np.isnan(res)
            ):
                # GH#18463
                targ = res

            self.check_results(targ, res, axis, check_dtype=check_dtype)
            if skipna:
                res = testfunc(testarval, axis=axis, **kwargs)
                self.check_results(targ, res, axis, check_dtype=check_dtype)
            if axis is None:
                res = testfunc(testarval, skipna=skipna, **kwargs)
                self.check_results(targ, res, axis, check_dtype=check_dtype)
            if skipna and axis is None:
                res = testfunc(testarval, **kwargs)
                self.check_results(targ, res, axis, check_dtype=check_dtype)

        if testarval.ndim <= 1:
            return

        # Recurse on lower-dimension
        testarval2 = np.take(testarval, 0, axis=-1)
        targarval2 = np.take(targarval, 0, axis=-1)
        self.check_fun_data(
            testfunc,
            targfunc,
            testar,
            testarval2,
            targarval2,
            skipna=skipna,
            check_dtype=check_dtype,
            empty_targfunc=empty_targfunc,
            **kwargs,
        )

    def check_fun(
        self, testfunc, targfunc, testar, skipna, empty_targfunc=None, **kwargs
    ):
        # [FUNCTION DOC]
        # HELPER: looks up the named array attribute and its reference array (the same name with a trailing NaN marker removed if that attribute exists) and delegates to the data comparison. Parameters: testfunc, targfunc, testar (attribute name), skipna, empty_targfunc (default none), extra keyword arguments.
        targar = testar
        if testar.endswith("_nan") and hasattr(self, testar[:-4]):
            targar = testar[:-4]

        testarval = getattr(self, testar)
        targarval = getattr(self, targar)
        self.check_fun_data(
            testfunc,
            targfunc,
            testar,
            testarval,
            targarval,
            skipna=skipna,
            empty_targfunc=empty_targfunc,
            **kwargs,
        )

    def check_funs(
        self,
        testfunc,
        targfunc,
        skipna,
        allow_complex=True,
        allow_all_nan=True,
        allow_date=True,
        allow_tdelta=True,
        allow_obj=True,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # HELPER: runs the comparison for a standard battery of arrays: float, float with NaN rows, integer, boolean; all-NaN (if allowed); complex, complex with NaN and all-NaN complex (if complex allowed); datetime (if allowed, first calling the reference on it once to ensure it works); timedelta (if allowed and the reference does not raise a type error); and an object array stacked from the object versions of the previous ones (if allowed). Parameters: testfunc, targfunc, skipna, flags allow_complex, allow_all_nan, allow_date, allow_tdelta (all default true), allow_obj (default true; the value "convert" wraps the reference to cast object arrays to complex or float first), plus keyword arguments forwarded. Side effect: replaces the object array attribute.
        self.check_fun(testfunc, targfunc, "arr_float", skipna, **kwargs)
        self.check_fun(testfunc, targfunc, "arr_float_nan", skipna, **kwargs)
        self.check_fun(testfunc, targfunc, "arr_int", skipna, **kwargs)
        self.check_fun(testfunc, targfunc, "arr_bool", skipna, **kwargs)
        objs = [
            self.arr_float.astype("O"),
            self.arr_int.astype("O"),
            self.arr_bool.astype("O"),
        ]

        if allow_all_nan:
            self.check_fun(testfunc, targfunc, "arr_nan", skipna, **kwargs)

        if allow_complex:
            self.check_fun(testfunc, targfunc, "arr_complex", skipna, **kwargs)
            self.check_fun(testfunc, targfunc, "arr_complex_nan", skipna, **kwargs)
            if allow_all_nan:
                self.check_fun(testfunc, targfunc, "arr_nan_nanj", skipna, **kwargs)
            objs += [self.arr_complex.astype("O")]

        if allow_date:
            targfunc(self.arr_date)
            self.check_fun(testfunc, targfunc, "arr_date", skipna, **kwargs)
            objs += [self.arr_date.astype("O")]

        if allow_tdelta:
            try:
                targfunc(self.arr_tdelta)
            except TypeError:
                pass
            else:
                self.check_fun(testfunc, targfunc, "arr_tdelta", skipna, **kwargs)
                objs += [self.arr_tdelta.astype("O")]

        if allow_obj:
            self.arr_obj = np.vstack(objs)
            # some nanops handle object dtypes better than their numpy
            # counterparts, so the numpy functions need to be given something
            # else
            if allow_obj == "convert":
                targfunc = partial(
                    self._badobj_wrap, func=targfunc, allow_complex=allow_complex
                )
            self.check_fun(testfunc, targfunc, "arr_obj", skipna, **kwargs)

    def _badobj_wrap(self, value, func, allow_complex=True, **kwargs):
        # [FUNCTION DOC]
        # HELPER: wrapper for reference functions that cannot handle object arrays. Parameters: value, the array; func, the reference function; allow_complex (default true), choosing complex128 versus float64 as the cast target for object input; extra keyword arguments. Returns the result of the reference function on the cast array.
        if value.dtype.kind == "O":
            if allow_complex:
                value = value.astype("c16")
            else:
                value = value.astype("f8")
        return func(value, **kwargs)

    @pytest.mark.parametrize(
        "nan_op,np_op", [(nanops.nanany, np.any), (nanops.nanall, np.all)]
    )
    def test_nan_funcs(self, nan_op, np_op, skipna):
        # [FUNCTION DOC]
        # Parametrized over any/all against their NumPy counterparts, and the skipna fixture. Compares via the battery without all-NaN or datetime arrays.
        self.check_funs(nan_op, np_op, skipna, allow_all_nan=False, allow_date=False)

    def test_nansum(self, skipna):
        # [FUNCTION DOC]
        # Uses the skipna fixture. Compares sum with NumPy sum over the battery (no datetimes), not checking dtypes, using the NaN-ignoring sum as the reference for all-NaN input.
        self.check_funs(
            nanops.nansum,
            np.sum,
            skipna,
            allow_date=False,
            check_dtype=False,
            empty_targfunc=np.nansum,
        )

    def test_nanmean(self, skipna):
        # [FUNCTION DOC]
        # Uses the skipna fixture. Compares mean with NumPy mean, without object or datetime arrays.
        self.check_funs(
            nanops.nanmean, np.mean, skipna, allow_obj=False, allow_date=False
        )

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_nanmedian(self, skipna):
        # [FUNCTION DOC]
        # Ignores runtime warnings. Uses the skipna fixture. Compares median with NumPy median without complex or datetime arrays; object arrays are converted for the reference.
        self.check_funs(
            nanops.nanmedian,
            np.median,
            skipna,
            allow_complex=False,
            allow_date=False,
            allow_obj="convert",
        )

    @pytest.mark.parametrize("ddof", range(3))
    def test_nanvar(self, ddof, skipna):
        # [FUNCTION DOC]
        # Parametrized over delta degrees of freedom zero to two and skipna. Compares variance with NumPy variance, excluding complex and datetime; objects converted.
        self.check_funs(
            nanops.nanvar,
            np.var,
            skipna,
            allow_complex=False,
            allow_date=False,
            allow_obj="convert",
            ddof=ddof,
        )

    @pytest.mark.parametrize("ddof", range(3))
    def test_nanstd(self, ddof, skipna):
        # [FUNCTION DOC]
        # Parametrized over delta degrees of freedom zero to two and skipna. Compares standard deviation with NumPy standard deviation, excluding complex and datetime; objects converted.
        self.check_funs(
            nanops.nanstd,
            np.std,
            skipna,
            allow_complex=False,
            allow_date=False,
            allow_obj="convert",
            ddof=ddof,
        )

    @pytest.mark.parametrize("ddof", range(3))
    def test_nansem(self, ddof, skipna):
        # [FUNCTION DOC]
        # Parametrized over delta degrees of freedom zero to two and skipna. Skips without scipy. Compares standard error with the SciPy standard-error function, excluding complex, datetime and timedelta; objects converted; invalid-value warnings suppressed.
        sp_stats = pytest.importorskip("scipy.stats")

        with np.errstate(invalid="ignore"):
            self.check_funs(
                nanops.nansem,
                sp_stats.sem,
                skipna,
                allow_complex=False,
                allow_date=False,
                allow_tdelta=False,
                allow_obj="convert",
                ddof=ddof,
            )

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.parametrize(
        "nan_op,np_op", [(nanops.nanmin, np.min), (nanops.nanmax, np.max)]
    )
    def test_nanops_with_warnings(self, nan_op, np_op, skipna):
        # [FUNCTION DOC]
        # Ignores runtime warnings. Parametrized over (min, NumPy min) and (max, NumPy max) and skipna. Compares via the battery without object arrays.
        self.check_funs(nan_op, np_op, skipna, allow_obj=False)

    def _argminmax_wrap(self, value, axis=None, func=None):
        # [FUNCTION DOC]
        # HELPER: reference wrapper imitating the NaN-aware argmin/argmax. Parameters: value, the array; axis (default none); func, the NumPy argument-position function. Computes the position, then sets entries (or the scalar) to minus one where the minimum along the axis is missing, i.e. where everything is missing.
        res = func(value, axis)
        nans = np.min(value, axis)
        nullnan = isna(nans)
        if res.ndim:
            res[nullnan] = -1
        elif (hasattr(nullnan, "all") and nullnan.all()) or (
            not hasattr(nullnan, "all") and nullnan
        ):
            res = -1
        return res

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_nanargmax(self, skipna):
        # [FUNCTION DOC]
        # Ignores runtime warnings. Uses skipna. Compares with the wrapped NumPy argmax, without object arrays.
        func = partial(self._argminmax_wrap, func=np.argmax)
        self.check_funs(nanops.nanargmax, func, skipna, allow_obj=False)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_nanargmin(self, skipna):
        # [FUNCTION DOC]
        # Ignores runtime warnings. Uses skipna. Compares with the wrapped NumPy argmin, without object arrays.
        func = partial(self._argminmax_wrap, func=np.argmin)
        self.check_funs(nanops.nanargmin, func, skipna, allow_obj=False)

    def _skew_kurt_wrap(self, values, axis=None, func=None):
        # [FUNCTION DOC]
        # HELPER: reference wrapper around the SciPy skew or kurtosis function. Parameters: values; axis (default none); func, the SciPy function. Casts non-float input to float64, calls the function with bias correction off, and sets the result to zero where all elements along the axis are equal (array result) or returns zero when the whole input is constant.
        # KNOWN ISSUE: the floating-point check applies the instance test to the dtype's scalar type, which is a class rather than an instance, so the check is always false and the input is always cast to float64.
        if not isinstance(values.dtype.type, np.floating):
            values = values.astype("f8")
        result = func(values, axis=axis, bias=False)
        # fix for handling cases where all elements in an axis are the same
        if isinstance(result, np.ndarray):
            result[np.max(values, axis=axis) == np.min(values, axis=axis)] = 0
            return result
        elif np.max(values) == np.min(values):
            return 0.0
        return result

    def test_nanskew(self, skipna):
        # [FUNCTION DOC]
        # Uses skipna. Skips without scipy. Compares skew against the wrapped SciPy skew, excluding complex, datetime and timedelta; invalid-value warnings suppressed.
        sp_stats = pytest.importorskip("scipy.stats")

        func = partial(self._skew_kurt_wrap, func=sp_stats.skew)
        with np.errstate(invalid="ignore"):
            self.check_funs(
                nanops.nanskew,
                func,
                skipna,
                allow_complex=False,
                allow_date=False,
                allow_tdelta=False,
            )

    def test_nankurt(self, skipna):
        # [FUNCTION DOC]
        # Uses skipna. Skips without scipy. Compares kurtosis against the wrapped SciPy kurtosis in Fisher (excess) definition, excluding complex, datetime and timedelta; warnings suppressed.
        sp_stats = pytest.importorskip("scipy.stats")

        func1 = partial(sp_stats.kurtosis, fisher=True)
        func = partial(self._skew_kurt_wrap, func=func1)
        with np.errstate(invalid="ignore"):
            self.check_funs(
                nanops.nankurt,
                func,
                skipna,
                allow_complex=False,
                allow_date=False,
                allow_tdelta=False,
            )

    def test_nanprod(self, skipna):
        # [FUNCTION DOC]
        # Uses skipna. Compares product against NumPy product, excluding datetime and timedelta, using the NaN-ignoring product as reference for all-NaN input.
        self.check_funs(
            nanops.nanprod,
            np.prod,
            skipna,
            allow_date=False,
            allow_tdelta=False,
            empty_targfunc=np.nanprod,
        )

    def check_nancorr_nancov_2d(self, checkfun, targ0, targ1, **kwargs):
        # [FUNCTION DOC]
        # HELPER: checks a correlation or covariance function on two-dimensional float arrays. Parameters: checkfun; targ0 (expected value for complete arrays); targ1 (expected for arrays that include a block of NaN rows); extra keyword arguments. Verifies the function matches targ0 for complete data (also with a minimum-period count one less than the length), targ1 for data with NaN rows, and gives NaN when one input is all NaN, both are, the NaN rows do not overlap, or the minimum periods exceeds the number of observations.
        res00 = checkfun(self.arr_float_2d, self.arr_float1_2d, **kwargs)
        res01 = checkfun(
            self.arr_float_2d,
            self.arr_float1_2d,
            min_periods=len(self.arr_float_2d) - 1,
            **kwargs,
        )
        tm.assert_almost_equal(targ0, res00)
        tm.assert_almost_equal(targ0, res01)

        res10 = checkfun(self.arr_float_nan_2d, self.arr_float1_nan_2d, **kwargs)
        res11 = checkfun(
            self.arr_float_nan_2d,
            self.arr_float1_nan_2d,
            min_periods=len(self.arr_float_2d) - 1,
            **kwargs,
        )
        tm.assert_almost_equal(targ1, res10)
        tm.assert_almost_equal(targ1, res11)

        targ2 = np.nan
        res20 = checkfun(self.arr_nan_2d, self.arr_float1_2d, **kwargs)
        res21 = checkfun(self.arr_float_2d, self.arr_nan_2d, **kwargs)
        res22 = checkfun(self.arr_nan_2d, self.arr_nan_2d, **kwargs)
        res23 = checkfun(self.arr_float_nan_2d, self.arr_nan_float1_2d, **kwargs)
        res24 = checkfun(
            self.arr_float_nan_2d,
            self.arr_nan_float1_2d,
            min_periods=len(self.arr_float_2d) - 1,
            **kwargs,
        )
        res25 = checkfun(
            self.arr_float_2d,
            self.arr_float1_2d,
            min_periods=len(self.arr_float_2d) + 1,
            **kwargs,
        )
        tm.assert_almost_equal(targ2, res20)
        tm.assert_almost_equal(targ2, res21)
        tm.assert_almost_equal(targ2, res22)
        tm.assert_almost_equal(targ2, res23)
        tm.assert_almost_equal(targ2, res24)
        tm.assert_almost_equal(targ2, res25)

    def check_nancorr_nancov_1d(self, checkfun, targ0, targ1, **kwargs):
        # [FUNCTION DOC]
        # HELPER: same checks as the two-dimensional helper but on the one-dimensional column views.
        res00 = checkfun(self.arr_float_1d, self.arr_float1_1d, **kwargs)
        res01 = checkfun(
            self.arr_float_1d,
            self.arr_float1_1d,
            min_periods=len(self.arr_float_1d) - 1,
            **kwargs,
        )
        tm.assert_almost_equal(targ0, res00)
        tm.assert_almost_equal(targ0, res01)

        res10 = checkfun(self.arr_float_nan_1d, self.arr_float1_nan_1d, **kwargs)
        res11 = checkfun(
            self.arr_float_nan_1d,
            self.arr_float1_nan_1d,
            min_periods=len(self.arr_float_1d) - 1,
            **kwargs,
        )
        tm.assert_almost_equal(targ1, res10)
        tm.assert_almost_equal(targ1, res11)

        targ2 = np.nan
        res20 = checkfun(self.arr_nan_1d, self.arr_float1_1d, **kwargs)
        res21 = checkfun(self.arr_float_1d, self.arr_nan_1d, **kwargs)
        res22 = checkfun(self.arr_nan_1d, self.arr_nan_1d, **kwargs)
        res23 = checkfun(self.arr_float_nan_1d, self.arr_nan_float1_1d, **kwargs)
        res24 = checkfun(
            self.arr_float_nan_1d,
            self.arr_nan_float1_1d,
            min_periods=len(self.arr_float_1d) - 1,
            **kwargs,
        )
        res25 = checkfun(
            self.arr_float_1d,
            self.arr_float1_1d,
            min_periods=len(self.arr_float_1d) + 1,
            **kwargs,
        )
        tm.assert_almost_equal(targ2, res20)
        tm.assert_almost_equal(targ2, res21)
        tm.assert_almost_equal(targ2, res22)
        tm.assert_almost_equal(targ2, res23)
        tm.assert_almost_equal(targ2, res24)
        tm.assert_almost_equal(targ2, res25)

    def test_nancorr(self):
        # [FUNCTION DOC]
        # Verifies the correlation function against NumPy correlation coefficients for two-dimensional data (default method) and one-dimensional data (Pearson).
        targ0 = np.corrcoef(self.arr_float_2d, self.arr_float1_2d)[0, 1]
        targ1 = np.corrcoef(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0, 1]
        self.check_nancorr_nancov_2d(nanops.nancorr, targ0, targ1)
        targ0 = np.corrcoef(self.arr_float_1d, self.arr_float1_1d)[0, 1]
        targ1 = np.corrcoef(self.arr_float_1d.flat, self.arr_float1_1d.flat)[0, 1]
        self.check_nancorr_nancov_1d(nanops.nancorr, targ0, targ1, method="pearson")

    def test_nancorr_pearson(self):
        # [FUNCTION DOC]
        # Same as the previous test but specifying the Pearson method for both dimensionalities.
        targ0 = np.corrcoef(self.arr_float_2d, self.arr_float1_2d)[0, 1]
        targ1 = np.corrcoef(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0, 1]
        self.check_nancorr_nancov_2d(nanops.nancorr, targ0, targ1, method="pearson")
        targ0 = np.corrcoef(self.arr_float_1d, self.arr_float1_1d)[0, 1]
        targ1 = np.corrcoef(self.arr_float_1d.flat, self.arr_float1_1d.flat)[0, 1]
        self.check_nancorr_nancov_1d(nanops.nancorr, targ0, targ1, method="pearson")

    def test_nancorr_kendall(self):
        # [FUNCTION DOC]
        # Skips without scipy. Verifies the Kendall method against the SciPy Kendall tau (first element), for two- and one-dimensional data.
        sp_stats = pytest.importorskip("scipy.stats")

        targ0 = sp_stats.kendalltau(self.arr_float_2d, self.arr_float1_2d)[0]
        targ1 = sp_stats.kendalltau(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0]
        self.check_nancorr_nancov_2d(nanops.nancorr, targ0, targ1, method="kendall")
        targ0 = sp_stats.kendalltau(self.arr_float_1d, self.arr_float1_1d)[0]
        targ1 = sp_stats.kendalltau(self.arr_float_1d.flat, self.arr_float1_1d.flat)[0]
        self.check_nancorr_nancov_1d(nanops.nancorr, targ0, targ1, method="kendall")

    def test_nancorr_spearman(self):
        # [FUNCTION DOC]
        # Skips without scipy. Verifies the Spearman method against the SciPy Spearman correlation (first element), for two- and one-dimensional data.
        sp_stats = pytest.importorskip("scipy.stats")

        targ0 = sp_stats.spearmanr(self.arr_float_2d, self.arr_float1_2d)[0]
        targ1 = sp_stats.spearmanr(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0]
        self.check_nancorr_nancov_2d(nanops.nancorr, targ0, targ1, method="spearman")
        targ0 = sp_stats.spearmanr(self.arr_float_1d, self.arr_float1_1d)[0]
        targ1 = sp_stats.spearmanr(self.arr_float_1d.flat, self.arr_float1_1d.flat)[0]
        self.check_nancorr_nancov_1d(nanops.nancorr, targ0, targ1, method="spearman")

    def test_invalid_method(self):
        # [FUNCTION DOC]
        # Skips without scipy. Verifies an unknown correlation method name raises a value error listing the known methods.
        pytest.importorskip("scipy")
        targ0 = np.corrcoef(self.arr_float_2d, self.arr_float1_2d)[0, 1]
        targ1 = np.corrcoef(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0, 1]
        msg = "Unknown method 'foo', expected one of 'kendall', 'spearman'"
        with pytest.raises(ValueError, match=msg):
            self.check_nancorr_nancov_1d(nanops.nancorr, targ0, targ1, method="foo")

    def test_nancov(self):
        # [FUNCTION DOC]
        # Verifies the covariance function against NumPy covariance entries for two- and one-dimensional data.
        targ0 = np.cov(self.arr_float_2d, self.arr_float1_2d)[0, 1]
        targ1 = np.cov(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0, 1]
        self.check_nancorr_nancov_2d(nanops.nancov, targ0, targ1)
        targ0 = np.cov(self.arr_float_1d, self.arr_float1_1d)[0, 1]
        targ1 = np.cov(self.arr_float_1d.flat, self.arr_float1_1d.flat)[0, 1]
        self.check_nancorr_nancov_1d(nanops.nancov, targ0, targ1)


@pytest.mark.parametrize(
    "arr, correct",
    [
        ("arr_complex", False),
        ("arr_int", False),
        ("arr_bool", False),
        ("arr_str", False),
        ("arr_utf", False),
        ("arr_complex_nan", False),
        ("arr_nan_nanj", False),
        ("arr_nan_infj", True),
        ("arr_complex_nan_infj", True),
    ],
)
def test_has_infs_non_float(request, arr, correct, disable_bottleneck):
    # [FUNCTION DOC]
    # Parametrized over (array fixture name, expected flag): complex, integer, boolean, string and NaN complex arrays have no infinities; the arrays with infinite imaginary parts do. Uses the disable-bottleneck fixture. Verifies the infinity detector, repeating on successively lower-dimensional slices down to a scalar.
    val = request.getfixturevalue(arr)
    while getattr(val, "ndim", True):
        res0 = nanops._has_infs(val)
        if correct:
            assert res0
        else:
            assert not res0

        if not hasattr(val, "ndim"):
            break

        # Reduce dimension for next step in the loop
        val = np.take(val, 0, axis=-1)


@pytest.mark.parametrize(
    "arr, correct",
    [
        ("arr_float", False),
        ("arr_nan", False),
        ("arr_float_nan", False),
        ("arr_nan_nan", False),
        ("arr_float_inf", True),
        ("arr_inf", True),
        ("arr_nan_inf", True),
        ("arr_float_nan_inf", True),
        ("arr_nan_nan_inf", True),
    ],
)
@pytest.mark.parametrize("astype", [None, "f4", "f2"])
def test_has_infs_floats(request, arr, correct, astype, disable_bottleneck):
    # [FUNCTION DOC]
    # Parametrized over float fixtures (false for plain, NaN-only; true when infinities present) and optional narrower float dtypes (none, 32-bit, 16-bit). Verifies the detector on the array and on each lower-dimensional slice.
    val = request.getfixturevalue(arr)
    if astype is not None:
        val = val.astype(astype)
    while getattr(val, "ndim", True):
        res0 = nanops._has_infs(val)
        if correct:
            assert res0
        else:
            assert not res0

        if not hasattr(val, "ndim"):
            break

        # Reduce dimension for next step in the loop
        val = np.take(val, 0, axis=-1)


@pytest.mark.parametrize(
    "fixture", ["arr_float", "arr_complex", "arr_int", "arr_bool", "arr_str", "arr_utf"]
)
def test_bn_ok_dtype(fixture, request, disable_bottleneck):
    # [FUNCTION DOC]
    # Parametrized over float, complex, integer, boolean and string fixtures. Verifies the bottleneck-eligibility check returns true for these dtypes.
    obj = request.getfixturevalue(fixture)
    assert nanops._bn_ok_dtype(obj.dtype, "test")


@pytest.mark.parametrize(
    "fixture",
    [
        "arr_date",
        "arr_tdelta",
        "arr_obj",
    ],
)
def test_bn_not_ok_dtype(fixture, request, disable_bottleneck):
    # [FUNCTION DOC]
    # Parametrized over datetime, timedelta and object fixtures. Verifies the bottleneck-eligibility check returns false.
    obj = request.getfixturevalue(fixture)
    assert not nanops._bn_ok_dtype(obj.dtype, "test")


class TestEnsureNumeric:
    # [CLASS DOC]
    # Tests for the helper that coerces input to a numeric type.
    def test_numeric_values(self):
        # [FUNCTION DOC]
        # Verifies integer, float and complex scalars pass through unchanged.
        # Test integer
        assert nanops._ensure_numeric(1) == 1

        # Test float
        assert nanops._ensure_numeric(1.1) == 1.1

        # Test complex
        assert nanops._ensure_numeric(1 + 2j) == 1 + 2j

    def test_ndarray(self):
        # [FUNCTION DOC]
        # Verifies numeric arrays and object arrays of numbers are returned numerically equal, while object arrays of strings (convertible-looking or not) raise a type error saying they could not be converted to numeric.
        # Test numeric ndarray
        values = np.array([1, 2, 3])
        assert np.allclose(nanops._ensure_numeric(values), values)

        # Test object ndarray
        o_values = values.astype(object)
        assert np.allclose(nanops._ensure_numeric(o_values), values)

        # Test convertible string ndarray
        s_values = np.array(["1", "2", "3"], dtype=object)
        msg = r"Could not convert \['1' '2' '3'\] to numeric"
        with pytest.raises(TypeError, match=msg):
            nanops._ensure_numeric(s_values)

        # Test non-convertible string ndarray
        s_values = np.array(["foo", "bar", "baz"], dtype=object)
        msg = r"Could not convert .* to numeric"
        with pytest.raises(TypeError, match=msg):
            nanops._ensure_numeric(s_values)

    def test_convertable_values(self):
        # [FUNCTION DOC]
        # Verifies strings that look like an integer, float or complex number still raise a type error naming the string.
        with pytest.raises(TypeError, match="Could not convert string '1' to numeric"):
            nanops._ensure_numeric("1")
        with pytest.raises(
            TypeError, match="Could not convert string '1.1' to numeric"
        ):
            nanops._ensure_numeric("1.1")
        with pytest.raises(
            TypeError, match=r"Could not convert string '1\+1j' to numeric"
        ):
            nanops._ensure_numeric("1+1j")

    def test_non_convertable_values(self):
        # [FUNCTION DOC]
        # Verifies a non-numeric string raises a type error naming it, and that an empty dictionary or list raise a type error from the underlying conversion saying the argument must be a string or a number.
        msg = "Could not convert string 'foo' to numeric"
        with pytest.raises(TypeError, match=msg):
            nanops._ensure_numeric("foo")

        # with the wrong type, python raises TypeError for us
        msg = "argument must be a string or a number"
        with pytest.raises(TypeError, match=msg):
            nanops._ensure_numeric({})
        with pytest.raises(TypeError, match=msg):
            nanops._ensure_numeric([])


class TestNanvarFixedValues:
    # [CLASS DOC]
    # Variance and standard deviation checks against known values, referencing a GitHub issue. Provides a fixture for the variance (three) and normal samples with that variance (one hundred thousand).
    # xref GH10242
    # Samples from a normal distribution.
    @pytest.fixture
    def variance(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning the true variance, 3.0.
        return 3.0

    @pytest.fixture
    def samples(self, variance):
        # [FUNCTION DOC]
        # HELPER: fixture returning 100000 normal samples with scale equal to the square root of the variance, drawn from the class's seeded generator.
        return self.prng.normal(scale=variance**0.5, size=100000)

    def test_nanvar_all_finite(self, samples, variance):
        # [FUNCTION DOC]
        # Verifies sample variance is within one percent of the true variance.
        actual_variance = nanops.nanvar(samples)
        tm.assert_almost_equal(actual_variance, variance, rtol=1e-2)

    def test_nanvar_nans(self, samples, variance):
        # [FUNCTION DOC]
        # Interleaves the samples with NaNs. Verifies variance is close to true when skipping NaNs and is NaN when not.
        samples_test = np.nan * np.ones(2 * samples.shape[0])
        samples_test[::2] = samples

        actual_variance = nanops.nanvar(samples_test, skipna=True)
        tm.assert_almost_equal(actual_variance, variance, rtol=1e-2)

        actual_variance = nanops.nanvar(samples_test, skipna=False)
        tm.assert_almost_equal(actual_variance, np.nan, rtol=1e-2)

    def test_nanstd_nans(self, samples, variance):
        # [FUNCTION DOC]
        # Interleaves the samples with NaNs. Verifies standard deviation is close to the square root of the variance when skipping NaNs.
        # KNOWN ISSUE: the second half is meant to check the standard deviation without skipping, but it calls the variance function again and names its result as a standard deviation (copy-paste of the previous test).
        samples_test = np.nan * np.ones(2 * samples.shape[0])
        samples_test[::2] = samples

        actual_std = nanops.nanstd(samples_test, skipna=True)
        tm.assert_almost_equal(actual_std, variance**0.5, rtol=1e-2)

        actual_std = nanops.nanvar(samples_test, skipna=False)
        tm.assert_almost_equal(actual_std, np.nan, rtol=1e-2)

    def test_nanvar_axis(self, samples, variance):
        # [FUNCTION DOC]
        # Stacks normal samples with uniform samples and verifies variances along axis one are the true variance and one twelfth.
        # Generate some sample data.
        samples_unif = self.prng.uniform(size=samples.shape[0])
        samples = np.vstack([samples, samples_unif])

        actual_variance = nanops.nanvar(samples, axis=1)
        tm.assert_almost_equal(
            actual_variance, np.array([variance, 1.0 / 12]), rtol=1e-2
        )

    def test_nanvar_ddof(self):
        # [FUNCTION DOC]
        # Uses uniform samples of 5 values plus a column of NaN per row (forcing the module's own algorithm). Verifies means of variances with delta degrees of freedom 0, 1, 2 equal the biased, unbiased and over-corrected expected values (relative tolerance one percent).
        n = 5
        samples = self.prng.uniform(size=(10000, n + 1))
        samples[:, -1] = np.nan  # Force use of our own algorithm.

        variance_0 = nanops.nanvar(samples, axis=1, skipna=True, ddof=0).mean()
        variance_1 = nanops.nanvar(samples, axis=1, skipna=True, ddof=1).mean()
        variance_2 = nanops.nanvar(samples, axis=1, skipna=True, ddof=2).mean()

        # The unbiased estimate.
        var = 1.0 / 12
        tm.assert_almost_equal(variance_1, var, rtol=1e-2)

        # The underestimated variance.
        tm.assert_almost_equal(variance_0, (n - 1.0) / n * var, rtol=1e-2)

        # The overestimated variance.
        tm.assert_almost_equal(variance_2, (n - 1.0) / (n - 2.0) * var, rtol=1e-2)

    @pytest.mark.parametrize("axis", range(2))
    @pytest.mark.parametrize("ddof", range(3))
    def test_ground_truth(self, axis, ddof):
        # [FUNCTION DOC]
        # Parametrized over axis and delta degrees of freedom. Against precomputed values from NumPy for a 3 by 3 block padded with a NaN row and column, verifies variance and standard deviation along the axis (skipping NaNs) match for the first three entries and the fourth is NaN.
        # Test against values that were precomputed with Numpy.
        samples = np.empty((4, 4))
        samples[:3, :3] = np.array(
            [
                [0.97303362, 0.21869576, 0.55560287],
                [0.72980153, 0.03109364, 0.99155171],
                [0.09317602, 0.60078248, 0.15871292],
            ]
        )
        samples[3] = samples[:, 3] = np.nan

        # Actual variances along axis=0, 1 for ddof=0, 1, 2
        variance = np.array(
            [
                [
                    [0.13762259, 0.05619224, 0.11568816],
                    [0.20643388, 0.08428837, 0.17353224],
                    [0.41286776, 0.16857673, 0.34706449],
                ],
                [
                    [0.09519783, 0.16435395, 0.05082054],
                    [0.14279674, 0.24653093, 0.07623082],
                    [0.28559348, 0.49306186, 0.15246163],
                ],
            ]
        )

        # Test nanvar.
        var = nanops.nanvar(samples, skipna=True, axis=axis, ddof=ddof)
        tm.assert_almost_equal(var[:3], variance[axis, ddof])
        assert np.isnan(var[3])

        # Test nanstd.
        std = nanops.nanstd(samples, skipna=True, axis=axis, ddof=ddof)
        tm.assert_almost_equal(std[:3], variance[axis, ddof] ** 0.5)
        assert np.isnan(std[3])

    @pytest.mark.parametrize("ddof", range(3))
    @pytest.mark.parametrize("axis", [0, 1, None])
    @pytest.mark.parametrize("method", ["var", "std", "sem"])
    def test_complex_nanvar(self, ddof, axis, method):
        # [FUNCTION DOC]
        # Parametrized over delta degrees of freedom, axis (0, 1, none) and variance/std/standard error. Verifies nanops results on a complex random array equal NumPy results (or SciPy for standard error, skipping if absent).
        arr = self.prng.standard_normal((5, 4)) + self.prng.standard_normal((5, 4)) * 1j
        nanops_method = getattr(nanops, f"nan{method}")
        if method in {"var", "std"}:
            comparator_method = getattr(np, method)
        elif method == "sem":
            comparator_method = pytest.importorskip("scipy.stats").sem
        result = nanops_method(arr, axis=axis, ddof=ddof)
        expected = comparator_method(arr, axis=axis, ddof=ddof)
        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize("ddof", range(3))
    def test_nanstd_roundoff(self, ddof):
        # [FUNCTION DOC]
        # Regression test for GitHub issues. Parametrized over delta degrees of freedom. Verifies the standard deviation of a constant Series of large values is exactly zero.
        # Regression test for GH 10242 (test data taken from GH 10489). Ensure
        # that variance is stable.
        data = Series(766897346 * np.ones(10))
        result = data.std(ddof=ddof)
        assert result == 0.0

    @property
    def prng(self):
        # [FUNCTION DOC]
        # HELPER: property returning a fresh random generator seeded with 2 (so each access restarts the same stream).
        return np.random.default_rng(2)


class TestNanskewFixedValues:
    # [CLASS DOC]
    # Skewness checks against known values (referencing a GitHub issue). Fixtures supply the sample (sine values) and its known skewness.
    # xref GH 11974
    # Test data + skewness value (computed with scipy.stats.skew)
    @pytest.fixture
    def samples(self):
        # [FUNCTION DOC]
        # HELPER: fixture; sine of 200 evenly spaced numbers from 0 to 1.
        return np.sin(np.linspace(0, 1, 200))

    @pytest.fixture
    def actual_skew(self):
        # [FUNCTION DOC]
        # HELPER: fixture; the known skewness of the samples, computed with SciPy.
        return -0.1875895205961754

    @pytest.mark.parametrize("val", [3075.2, 3075.3, 3075.5])
    def test_constant_series(self, val):
        # [FUNCTION DOC]
        # Parametrized over three large constant values. Verifies the skew of a constant array is exactly zero.
        # xref GH 11974
        data = val * np.ones(300)
        skew = nanops.nanskew(data)
        assert skew == 0.0

    def test_all_finite(self):
        # [FUNCTION DOC]
        # Verifies a left-tailed beta sample has negative skew and a right-tailed one positive skew.
        alpha, beta = 0.3, 0.1
        left_tailed = self.prng.beta(alpha, beta, size=100)
        assert nanops.nanskew(left_tailed) < 0

        alpha, beta = 0.1, 0.3
        right_tailed = self.prng.beta(alpha, beta, size=100)
        assert nanops.nanskew(right_tailed) > 0

    def test_ground_truth(self, samples, actual_skew):
        # [FUNCTION DOC]
        # Verifies skew of the samples equals the known value.
        skew = nanops.nanskew(samples)
        tm.assert_almost_equal(skew, actual_skew)

    def test_axis(self, samples, actual_skew):
        # [FUNCTION DOC]
        # Stacks the samples with an all-NaN row; verifies skew along axis one gives the known value and NaN.
        samples = np.vstack([samples, np.nan * np.ones(len(samples))])
        skew = nanops.nanskew(samples, axis=1)
        tm.assert_almost_equal(skew, np.array([actual_skew, np.nan]))

    def test_nans(self, samples):
        # [FUNCTION DOC]
        # Appends a NaN; verifies skew without skipping is NaN.
        samples = np.hstack([samples, np.nan])
        skew = nanops.nanskew(samples, skipna=False)
        assert np.isnan(skew)

    def test_nans_skipna(self, samples, actual_skew):
        # [FUNCTION DOC]
        # Appends a NaN; verifies skew when skipping equals the known value.
        samples = np.hstack([samples, np.nan])
        skew = nanops.nanskew(samples, skipna=True)
        tm.assert_almost_equal(skew, actual_skew)

    @pytest.mark.parametrize(
        "initial_data, nobs",
        [
            ([-2.05191341e-05, -4.10391103e-05], 27),
            ([-2.05191341e-10, -4.10391103e-10], 27),
            ([-2.05191341e-05, -4.10391103e-05], 10_000),
            ([-2.05191341e-10, -4.10391103e-10], 10_000),
        ],
    )
    def test_low_variance(self, initial_data, nobs):
        # [FUNCTION DOC]
        # Parametrized over tiny initial values at two scales and two lengths, with the rest zeros. Skips without scipy. Verifies skew equals SciPy's bias-corrected skew.
        st = pytest.importorskip("scipy.stats")
        data = np.zeros((nobs,), dtype=np.float64)
        data[: len(initial_data)] = initial_data
        skew = nanops.nanskew(data)
        expected = st.skew(data, bias=False)
        tm.assert_almost_equal(skew, expected)

    @property
    def prng(self):
        # [FUNCTION DOC]
        # HELPER: property returning a fresh generator seeded with 2.
        return np.random.default_rng(2)


class TestNankurtFixedValues:
    # [CLASS DOC]
    # Kurtosis checks against known values (referencing a GitHub issue). Fixtures supply the sample and its known excess kurtosis.
    # xref GH 11974
    # Test data + kurtosis value (computed with scipy.stats.kurtosis)
    @pytest.fixture
    def samples(self):
        # [FUNCTION DOC]
        # HELPER: fixture; sine of 200 evenly spaced numbers from 0 to 1.
        return np.sin(np.linspace(0, 1, 200))

    @pytest.fixture
    def actual_kurt(self):
        # [FUNCTION DOC]
        # HELPER: fixture; the known kurtosis of the samples, computed with SciPy.
        return -1.2058303433799713

    @pytest.mark.parametrize("val", [3075.2, 3075.3, 3075.5])
    def test_constant_series(self, val):
        # [FUNCTION DOC]
        # Parametrized over three large constants. Verifies the kurtosis of a constant array is exactly zero.
        # xref GH 11974
        data = val * np.ones(300)
        kurt = nanops.nankurt(data)
        tm.assert_equal(kurt, 0.0)

    def test_all_finite(self):
        # [FUNCTION DOC]
        # Verifies kurtosis of two beta samples is below the stated thresholds (below two, and below zero).
        alpha, beta = 0.3, 0.1
        left_tailed = self.prng.beta(alpha, beta, size=100)
        assert nanops.nankurt(left_tailed) < 2

        alpha, beta = 0.1, 0.3
        right_tailed = self.prng.beta(alpha, beta, size=100)
        assert nanops.nankurt(right_tailed) < 0

    def test_ground_truth(self, samples, actual_kurt):
        # [FUNCTION DOC]
        # Verifies kurtosis of the samples equals the known value.
        kurt = nanops.nankurt(samples)
        tm.assert_almost_equal(kurt, actual_kurt)

    def test_axis(self, samples, actual_kurt):
        # [FUNCTION DOC]
        # Stacks the samples with an all-NaN row; verifies kurtosis along axis one gives the known value and NaN.
        samples = np.vstack([samples, np.nan * np.ones(len(samples))])
        kurt = nanops.nankurt(samples, axis=1)
        tm.assert_almost_equal(kurt, np.array([actual_kurt, np.nan]))

    def test_nans(self, samples):
        # [FUNCTION DOC]
        # Appends a NaN; verifies kurtosis without skipping is NaN.
        samples = np.hstack([samples, np.nan])
        kurt = nanops.nankurt(samples, skipna=False)
        assert np.isnan(kurt)

    def test_nans_skipna(self, samples, actual_kurt):
        # [FUNCTION DOC]
        # Appends a NaN; verifies kurtosis when skipping equals the known value.
        samples = np.hstack([samples, np.nan])
        kurt = nanops.nankurt(samples, skipna=True)
        tm.assert_almost_equal(kurt, actual_kurt)

    @pytest.mark.parametrize(
        "initial_data, nobs",
        [
            ([-2.05191341e-05, -4.10391103e-05], 27),
            ([-2.05191341e-10, -4.10391103e-10], 27),
            ([-2.05191341e-05, -4.10391103e-05], 10_000),
            ([-2.05191341e-10, -4.10391103e-10], 10_000),
        ],
    )
    def test_low_variance(self, initial_data, nobs):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Parametrized as in the skew test. Skips without scipy. Verifies kurtosis equals SciPy's bias-corrected kurtosis.
        # GH#57972
        st = pytest.importorskip("scipy.stats")
        data = np.zeros((nobs,), dtype=np.float64)
        data[: len(initial_data)] = initial_data
        kurt = nanops.nankurt(data)
        expected = st.kurtosis(data, bias=False)
        tm.assert_almost_equal(kurt, expected)

    @property
    def prng(self):
        # [FUNCTION DOC]
        # HELPER: property returning a fresh generator seeded with 2.
        return np.random.default_rng(2)


class TestDatetime64NaNOps:
    # [CLASS DOC]
    # Tests mean of datetime-like data (enabling it changes DataFrame mean behavior; see GitHub issue).
    # Enabling mean changes the behavior of DataFrame.mean
    # See https://github.com/pandas-dev/pandas/issues/24752
    def test_nanmean(self, unit):
        # [FUNCTION DOC]
        # Uses the time-unit fixture. Verifies the mean of three consecutive dates (as an index and as its backing array) is the middle date, also after inserting a NaT in the middle.
        dti = pd.date_range("2016-01-01", periods=3).as_unit(unit)
        expected = dti[1]

        for obj in [dti, dti._data]:
            result = nanops.nanmean(obj)
            assert result == expected

        dti2 = dti.insert(1, pd.NaT)

        for obj in [dti2, dti2._data]:
            result = nanops.nanmean(obj)
            assert result == expected

    @pytest.mark.parametrize("constructor", ["M8", "m8"])
    def test_nanmean_skipna_false(self, constructor, unit):
        # [FUNCTION DOC]
        # Parametrized over datetime and timedelta kinds and the unit fixture. For a 4 by 3 array with the last element NaT, verifies the mean without skipping is NaT of the same dtype overall; along axis zero the first two column means are four and five and the third NaT; along axis one the row means are the middle elements and NaT for the last row.
        dtype = f"{constructor}[{unit}]"
        arr = np.arange(12).astype(np.int64).view(dtype).reshape(4, 3)

        arr[-1, -1] = "NaT"

        result = nanops.nanmean(arr, skipna=False)
        assert np.isnat(result)
        assert result.dtype == dtype

        result = nanops.nanmean(arr, axis=0, skipna=False)
        expected = np.array([4, 5, "NaT"], dtype=arr.dtype)
        tm.assert_numpy_array_equal(result, expected)

        result = nanops.nanmean(arr, axis=1, skipna=False)
        expected = np.array([arr[0, 1], arr[1, 1], arr[2, 1], arr[-1, -1]])
        tm.assert_numpy_array_equal(result, expected)


def test_use_bottleneck():
    # [FUNCTION DOC]
    # If bottleneck is installed, verifies the option context manager can set the use_bottleneck option true and false and reads back correctly; otherwise does nothing.
    if nanops._BOTTLENECK_INSTALLED:
        with pd.option_context("use_bottleneck", True):
            assert pd.get_option("use_bottleneck")

        with pd.option_context("use_bottleneck", False):
            assert not pd.get_option("use_bottleneck")


@pytest.mark.parametrize(
    "numpy_op, expected",
    [
        (np.sum, 10),
        (np.nansum, 10),
        (np.mean, 2.5),
        (np.nanmean, 2.5),
        (np.median, 2.5),
        (np.nanmedian, 2.5),
        (np.min, 1),
        (np.max, 4),
        (np.nanmin, 1),
        (np.nanmax, 4),
    ],
)
def test_numpy_ops(numpy_op, expected):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over NumPy reductions (sum, mean, median, min, max and NaN-ignoring variants) and their expected values for the Series of one to four. Verifies applying them to a Series gives the expected scalar.
    # GH8383
    result = numpy_op(Series([1, 2, 3, 4]))
    assert result == expected


@pytest.mark.parametrize(
    "operation",
    [
        nanops.nanany,
        nanops.nanall,
        nanops.nansum,
        nanops.nanmean,
        nanops.nanmedian,
        nanops.nanstd,
        nanops.nanvar,
        nanops.nansem,
        nanops.nanargmax,
        nanops.nanargmin,
        nanops.nanmax,
        nanops.nanmin,
        nanops.nanskew,
        nanops.nankurt,
        nanops.nanprod,
    ],
)
def test_nanops_independent_of_mask_param(operation):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over fifteen nanops functions. Verifies passing a precomputed missing-value mask gives the same result as not passing one, for a Series with NaNs.
    # GH22764
    ser = Series([1, 2, np.nan, 3, np.nan, 4])
    mask = ser.isna()
    median_expected = operation(ser._values)
    median_result = operation(ser._values, mask=mask)
    assert median_expected == median_result


@pytest.mark.parametrize("min_count", [-1, 0])
def test_check_below_min_count_negative_or_zero_min_count(min_count):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over minimum counts minus one and zero. Verifies the minimum-count check returns false.
    # GH35227
    result = nanops.check_below_min_count((21, 37), None, min_count)
    expected_result = False
    assert result == expected_result


@pytest.mark.parametrize(
    "mask", [None, np.array([False, False, True]), np.array([True] + 9 * [False])]
)
@pytest.mark.parametrize("min_count, expected_result", [(1, False), (101, True)])
def test_check_below_min_count_positive_min_count(mask, min_count, expected_result):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over masks (none, one masked, three masked of different shapes) and (minimum count, expected) pairs (1 gives false, 101 gives true). With a 10 by 10 shape verifies the check result.
    # GH35227
    shape = (10, 10)
    result = nanops.check_below_min_count(shape, mask, min_count)
    assert result == expected_result


@td.skip_if_windows
@td.skip_if_32bit
@pytest.mark.parametrize("min_count, expected_result", [(1, False), (2812191852, True)])
def test_check_below_min_count_large_shape(min_count, expected_result):
    # [FUNCTION DOC]
    # Skipped on Windows and 32-bit. Regression test for a GitHub issue using a shape whose element count exceeds 32-bit integer range. Verifies minimum count 1 gives false and a count exceeding the element count gives true, without overflow.
    # GH35227 large shape used to show that the issue is fixed
    shape = (2244367, 1253)
    result = nanops.check_below_min_count(shape, mask=None, min_count=min_count)
    assert result == expected_result


@pytest.mark.parametrize("func", ["nanmean", "nansum"])
def test_check_bottleneck_disallow(any_real_numpy_dtype, func):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over mean and sum and the real NumPy dtype fixture. Verifies the bottleneck-eligibility check is false for every real dtype for these two functions.
    # GH 42878 bottleneck sometimes produces unreliable results for mean and sum
    assert not nanops._bn_ok_dtype(np.dtype(any_real_numpy_dtype).type, func)


@pytest.mark.parametrize("val", [2**55, -(2**55), 20150515061816532])
def test_nanmean_overflow(disable_bottleneck, val, using_python_scalars):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over three large integers. Uses the disable-bottleneck and python-scalars fixtures. Verifies the mean of a Series of 500 identical 64-bit integers equals the value without overflow; the result is a Python float when python-scalars mode is on, otherwise equals the NumPy mean and is float64.
    # GH 10155
    # In the previous implementation mean can overflow for int dtypes, it
    # is now consistent with numpy

    ser = Series(val, index=range(500), dtype=np.int64)
    result = ser.mean()
    assert result == val
    if using_python_scalars:
        assert type(result) == float
    else:
        np_result = ser.values.mean()
        assert result == np_result
        assert result.dtype == np.float64


@pytest.mark.parametrize(
    "dtype",
    [
        np.int16,
        np.int32,
        np.int64,
        np.float32,
        np.float64,
        getattr(np, "float128", None),
    ],
)
@pytest.mark.parametrize("method", ["mean", "std", "var", "skew", "kurt", "min", "max"])
def test_returned_dtype(disable_bottleneck, dtype, method, using_python_scalars):
    # [FUNCTION DOC]
    # Parametrized over integer and float dtypes (including 128-bit float if the platform has it; otherwise the case skips) and the methods mean, std, var, skew, kurt, min, max. Uses disable-bottleneck and python-scalars. Verifies the result type: Python floats (or ints for integer min and max) in python-scalars mode; otherwise float64 for integer input except min and max, and the input dtype for the rest.
    if dtype is None:
        pytest.skip("np.float128 not available")

    ser = Series(range(10), dtype=dtype)
    result = getattr(ser, method)()
    if using_python_scalars:
        if is_integer_dtype(dtype) and method in ["min", "max"]:
            assert isinstance(result, int)
        else:
            assert type(result) == float
    elif is_integer_dtype(dtype) and method not in ["min", "max"]:
        assert result.dtype == np.float64
    else:
        assert result.dtype == dtype
