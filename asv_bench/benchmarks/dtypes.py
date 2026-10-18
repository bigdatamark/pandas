# [MODULE DOC]
# Benchmarks for dtype handling: converting objects to dtype instances, the
# error path for invalid dtype inputs, selecting frame columns by dtype, and
# checking for extension dtypes. Public contents are the benchmark classes Dtypes,
# DtypesInvalid, SelectDtypes and CheckDtypes. Imports the dtype name lists from
# the shared helper module and the internal testing module for dtype lists. At
# import time it builds two module-level lists: _numpy_dtypes (numpy dtype
# objects for the numeric, datetime and string dtype names) and _dtypes (those
# plus the extension dtypes). The final line re-exports the common asv setup hook.
import string

import numpy as np

import pandas as pd
from pandas import (
    DataFrame,
    Index,
)
import pandas._testing as tm
from pandas.api.types import (
    is_extension_array_dtype,
    pandas_dtype,
)

from .pandas_vb_common import (
    datetime_dtypes,
    extension_dtypes,
    numeric_dtypes,
    string_dtypes,
)

_numpy_dtypes = [
    np.dtype(dtype) for dtype in (numeric_dtypes + datetime_dtypes + string_dtypes)
]
_dtypes = _numpy_dtypes + extension_dtypes


class Dtypes:
    # [CLASS DOC]
    # Benchmark of the public dtype-coercion function applied to dtype objects and
    # to their names. Parametrized by each entry of the combined dtype list, both
    # as objects and as name strings. Has no setup state.
    params = _dtypes + [dt.name for dt in _dtypes]
    param_names = ["dtype"]

    def time_pandas_dtype(self, dtype):
        # [FUNCTION DOC]
        # Times converting the given dtype object or name to a dtype instance.
        # Result is discarded.
        pandas_dtype(dtype)


class DtypesInvalid:
    # [CLASS DOC]
    # Benchmark of the dtype-coercion function when given invalid input of four
    # kinds (a string scalar, an integer scalar, a list of strings, an array of
    # strings), measuring the cost of the failure path. Parametrized by the
    # input kind key. Class attribute data_dict maps each key to the input.
    param_names = ["dtype"]
    params = ["scalar-string", "scalar-int", "list-string", "array-string"]
    data_dict = {
        "scalar-string": "foo",
        "scalar-int": 1,
        "list-string": ["foo"] * 1000,
        "array-string": np.array(["foo"] * 1000),
    }

    def time_pandas_dtype_invalid(self, dtype):
        # [FUNCTION DOC]
        # Times calling the dtype-coercion function on the invalid input chosen by
        # the key, swallowing the type error that is expected to be raised.
        # Any other exception would propagate.
        try:
            pandas_dtype(self.data_dict[dtype])
        except TypeError:
            pass


class SelectDtypes:
    # [CLASS DOC]
    # Benchmark of selecting frame columns by including or excluding a dtype,
    # for frames of integer, float, boolean and string columns. Parametrized by a
    # dtype taken from the testing-module dtype lists (the list names are chosen
    # at class creation, trying the newer names first and falling back to older
    # names if the attribute lookup fails). State: index, columns, df_int,
    # df_float, df_bool and df_string.
    try:
        params = [
            tm.ALL_INT_NUMPY_DTYPES
            + tm.ALL_INT_EA_DTYPES
            + tm.FLOAT_NUMPY_DTYPES
            + tm.COMPLEX_DTYPES
            + tm.DATETIME64_DTYPES
            + tm.TIMEDELTA64_DTYPES
            + tm.BOOL_DTYPES
        ]
    except AttributeError:
        params = [
            tm.ALL_INT_DTYPES
            + tm.ALL_EA_INT_DTYPES
            + tm.FLOAT_DTYPES
            + tm.COMPLEX_DTYPES
            + tm.DATETIME64_DTYPES
            + tm.TIMEDELTA64_DTYPES
            + tm.BOOL_DTYPES
        ]
    param_names = ["dtype"]

    def setup(self, dtype):
        # [FUNCTION DOC]
        # Builds four frames of 5000 rows by 50 columns sharing string row and
        # column labels: df_int (random integers below 100), df_float (random
        # normal), df_bool (random booleans) and df_string (random ASCII letters).
        # The dtype parameter is unused here. Stores labels and frames on the
        # instance.
        N, K = 5000, 50
        self.index = Index([f"i-{i}" for i in range(N)], dtype=object)
        self.columns = Index([f"i-{i}" for i in range(K)], dtype=object)

        def create_df(data):
            # [FUNCTION DOC]
            # HELPER: frame builder for the setup method of this benchmark class.
            # Wraps the given two-dimensional data in a frame using the shared
            # index and columns stored on the instance, and returns it.
            return DataFrame(data, index=self.index, columns=self.columns)

        self.df_int = create_df(np.random.randint(low=100, size=(N, K)))
        self.df_float = create_df(np.random.randn(N, K))
        self.df_bool = create_df(np.random.choice([True, False], size=(N, K)))
        self.df_string = create_df(
            np.random.choice(list(string.ascii_letters), size=(N, K))
        )

    def time_select_dtype_int_include(self, dtype):
        # [FUNCTION DOC]
        # Times selecting columns of the given dtype from the integer frame.
        self.df_int.select_dtypes(include=dtype)

    def time_select_dtype_int_exclude(self, dtype):
        # [FUNCTION DOC]
        # Times selecting columns not of the given dtype from the integer frame.
        self.df_int.select_dtypes(exclude=dtype)

    def time_select_dtype_float_include(self, dtype):
        # [FUNCTION DOC]
        # Times selecting columns of the given dtype from the float frame.
        self.df_float.select_dtypes(include=dtype)

    def time_select_dtype_float_exclude(self, dtype):
        # [FUNCTION DOC]
        # Times selecting columns not of the given dtype from the float frame.
        self.df_float.select_dtypes(exclude=dtype)

    def time_select_dtype_bool_include(self, dtype):
        # [FUNCTION DOC]
        # Times selecting columns of the given dtype from the boolean frame.
        self.df_bool.select_dtypes(include=dtype)

    def time_select_dtype_bool_exclude(self, dtype):
        # [FUNCTION DOC]
        # Times selecting columns not of the given dtype from the boolean frame.
        self.df_bool.select_dtypes(exclude=dtype)

    def time_select_dtype_string_include(self, dtype):
        # [FUNCTION DOC]
        # Times selecting columns of the given dtype from the string frame.
        self.df_string.select_dtypes(include=dtype)

    def time_select_dtype_string_exclude(self, dtype):
        # [FUNCTION DOC]
        # Times selecting columns not of the given dtype from the string frame.
        self.df_string.select_dtypes(exclude=dtype)


class CheckDtypes:
    # [CLASS DOC]
    # Benchmark of the extension-dtype predicate applied to an extension dtype
    # and to a plain numpy dtype. No parameters. State: ext_dtype and np_dtype.
    def setup(self):
        # [FUNCTION DOC]
        # Creates the nullable 64-bit integer dtype as ext_dtype and the numpy
        # 64-bit integer dtype as np_dtype.
        self.ext_dtype = pd.Int64Dtype()
        self.np_dtype = np.dtype("int64")

    def time_is_extension_array_dtype_true(self):
        # [FUNCTION DOC]
        # Times the predicate on the extension dtype (expected true).
        is_extension_array_dtype(self.ext_dtype)

    def time_is_extension_array_dtype_false(self):
        # [FUNCTION DOC]
        # Times the predicate on the numpy dtype (expected false).
        is_extension_array_dtype(self.np_dtype)


from .pandas_vb_common import setup  # noqa: F401 isort:skip
