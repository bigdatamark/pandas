# [MODULE DOC]
# Shared pytest fixtures for the extension-array test suite. Third-party array authors are
# expected to override the fixtures that raise a not-implemented error (dtype, data, data_missing,
# data_for_sorting, data_missing_for_sorting, data_for_grouping, and when relevant
# data_for_twos) in their own conftest or test module; the rest are defaults or parametrization
# helpers (all_data, data_repeated, na_cmp, na_value, box_in_series, groupby_apply_op, as_frame,
# as_series, use_numpy, fillna_method, as_array, invalid_scalar). Importing this module has no
# side effects beyond defining fixtures.
import operator

import pytest

from pandas import Series


@pytest.fixture
def dtype():
    # [FUNCTION DOC]
    # HELPER: fixture that must be overridden; returns the extension dtype under test. The default
    # raises a not-implemented error.
    """A fixture providing the ExtensionDtype to validate."""
    raise NotImplementedError


@pytest.fixture
def data():
    # [FUNCTION DOC]
    # HELPER: fixture that must be overridden; returns a length-ten array of the type under test whose
    # first two elements are non-missing and unequal. The default raises a not-implemented error.
    """
    Length-10 array for this type.

    * data[0] and data[1] should both be non missing
    * data[0] and data[1] should not be equal
    """
    raise NotImplementedError


@pytest.fixture
def data_for_twos(dtype):
    # [FUNCTION DOC]
    # HELPER: fixture returning a length-ten array whose elements are all the value two, used for
    # divmod tests. The default skips the test unless the dtype is numeric or timedelta-kind, and
    # otherwise raises a not-implemented error; implementers override it, or skip if divmod is not
    # supported.
    """
    Length-10 array in which all the elements are two.

    Call pytest.skip in your fixture if the dtype does not support divmod.
    """
    if not (dtype._is_numeric or dtype.kind == "m"):
        # Object-dtypes may want to allow this, but for the most part
        #  only numeric and timedelta-like dtypes will need to implement this.
        pytest.skip(f"{dtype} is not a numeric dtype")

    raise NotImplementedError


@pytest.fixture
def data_missing():
    # [FUNCTION DOC]
    # HELPER: fixture that must be overridden; returns a length-two array of a missing value followed
    # by a valid value. The default raises a not-implemented error.
    """Length-2 array with [NA, Valid]"""
    raise NotImplementedError


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    # [FUNCTION DOC]
    # HELPER: parametrized fixture (ids data and data_missing) that returns either the data fixture or
    # the data_missing fixture depending on the parameter.
    # KNOWN ISSUE: if the parameter were anything other than the two declared values the function
    # would return nothing; it relies on only those two parameters existing.
    """Parametrized fixture giving 'data' and 'data_missing'"""
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing


@pytest.fixture
def data_repeated(data):
    # [FUNCTION DOC]
    # HELPER: fixture returning a generator factory. The returned callable takes a count and returns a
    # generator that yields the data fixture that many times (the same array object each time, not
    # copies).
    """
    Generate many datasets.

    Parameters
    ----------
    data : fixture implementing `data`

    Returns
    -------
    Callable[[int], Generator]:
        A callable that takes a `count` argument and
        returns a generator yielding `count` datasets.
    """

    def gen(count):
        # [FUNCTION DOC]
        # HELPER: the generator factory returned by the data_repeated fixture. Input count; yields the
        # captured data array count times.
        for _ in range(count):
            yield data

    return gen


@pytest.fixture
def data_for_sorting():
    # [FUNCTION DOC]
    # HELPER: fixture that must be overridden; returns a length-three array of items ordered B, C, A
    # with A less than B less than C (for boolean dtypes, with B and C both true). The default raises a
    # not-implemented error.
    """
    Length-3 array with a known sort order.

    This should be three items [B, C, A] with
    A < B < C

    For boolean dtypes (for which there are only 2 values available),
    set B=C=True
    """
    raise NotImplementedError


@pytest.fixture
def data_missing_for_sorting():
    # [FUNCTION DOC]
    # HELPER: fixture that must be overridden; returns a length-three array of items B, missing, A with
    # A less than B. The default raises a not-implemented error.
    """
    Length-3 array with a known sort order.

    This should be three items [B, NA, A] with
    A < B and NA missing.
    """
    raise NotImplementedError


@pytest.fixture
def na_cmp():
    # [FUNCTION DOC]
    # HELPER: fixture returning the binary comparison function used to check that two scalars are both
    # the missing value; the default is the identity comparison operator.
    """
    Binary operator for comparing NA values.

    Should return a function of two arguments that returns
    True if both arguments are (scalar) NA for your type.

    By default, uses ``operator.is_``
    """
    return operator.is_


@pytest.fixture
def na_value(dtype):
    # [FUNCTION DOC]
    # HELPER: fixture returning the scalar missing value of the dtype, taken from the dtype fixture's
    # missing-value attribute.
    """
    The scalar missing value for this type. Default dtype.na_value.

    TODO: can be removed in 3.x (see https://github.com/pandas-dev/pandas/pull/54930)
    """
    return dtype.na_value


@pytest.fixture
def data_for_grouping():
    # [FUNCTION DOC]
    # HELPER: fixture that must be overridden; returns eight values arranged B, B, missing, missing, A,
    # A, B, C with A less than B less than C (for boolean-kind dtypes with only two unique non-missing
    # values, C equals B). Used for factorize, groupby and unique tests. The default raises a
    # not-implemented error.
    """
    Data for factorization, grouping, and unique tests.

    Expected to be like [B, B, NA, NA, A, A, B, C]

    Where A < B < C and NA is missing.

    If a dtype has _is_boolean = True, i.e. only 2 unique non-NA entries,
    then set C=B.
    """
    raise NotImplementedError


@pytest.fixture(params=[True, False])
def box_in_series(request):
    # [FUNCTION DOC]
    # HELPER: boolean fixture (true and false): whether to wrap data in a series.
    """Whether to box the data in a Series"""
    return request.param


@pytest.fixture(
    params=[
        lambda x: 1,
        lambda x: [1] * len(x),
        lambda x: Series([1] * len(x)),
        lambda x: x,
    ],
    ids=["scalar", "list", "series", "object"],
)
def groupby_apply_op(request):
    # [FUNCTION DOC]
    # HELPER: parametrized fixture (ids scalar, list, series, object) returning a function for the
    # groupby apply tests that returns respectively the constant one, a list of ones of the input's
    # length, a series of ones of that length, and the input itself.
    """
    Functions to test groupby.apply().
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_frame(request):
    # [FUNCTION DOC]
    # HELPER: boolean fixture (true and false) to compare series behavior with single-column frame
    # behavior.
    """
    Boolean fixture to support Series and Series.to_frame() comparison testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_series(request):
    # [FUNCTION DOC]
    # HELPER: boolean fixture (true and false) to compare array behavior with series-wrapped
    # behavior.
    """
    Boolean fixture to support arr and Series(arr) comparison testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def use_numpy(request):
    # [FUNCTION DOC]
    # HELPER: boolean fixture (true and false) to compare calling the array method with calling the
    # equivalent numpy function.
    """
    Boolean fixture to support comparison testing of ExtensionDtype array
    and numpy array.
    """
    return request.param


@pytest.fixture(params=["ffill", "bfill"])
def fillna_method(request):
    # [FUNCTION DOC]
    # HELPER: fixture with two values, the strings ffill and bfill, naming the series fill method to
    # test.
    """
    Parametrized fixture giving method parameters 'ffill' and 'bfill' for
    Series.<method> testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_array(request):
    # [FUNCTION DOC]
    # HELPER: boolean fixture (true and false): whether a value should be passed as an extension array
    # (built with the sequence constructor) or as a plain list.
    """
    Boolean fixture to support ExtensionDtype _from_sequence method testing.
    """
    return request.param


@pytest.fixture
def invalid_scalar(data):
    # [FUNCTION DOC]
    # HELPER: fixture returning a scalar that the array cannot hold: a fresh bare object instance.
    # The default works for most arrays but is not guaranteed; arrays that can hold any object should
    # skip tests using it.
    """
    A scalar that *cannot* be held by this ExtensionArray.

    The default should work for most subclasses, but is not guaranteed.

    If the array can hold any item (i.e. object dtype), then use pytest.skip.
    """
    return object.__new__(object)
