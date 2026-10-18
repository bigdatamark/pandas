# [MODULE DOC]
# Tests that the public types namespace exposes exactly the expected names.
# Reuses the Base checker from the api test module. Contains one test class.

from __future__ import annotations

import pandas._testing as tm
from pandas.api import types
from pandas.tests.api.test_api import Base


class TestTypes(Base):
    # [CLASS DOC]
    # Test class verifying the contents of the public types namespace; inherits the name-checking
    # helper from the shared Base class. Class attributes hold the allowed function names, an empty list
    # of deprecated names, and the four dtype class names.
    allowed = [
        "is_any_real_numeric_dtype",
        "is_bool",
        "is_bool_dtype",
        "is_categorical_dtype",
        "is_complex",
        "is_complex_dtype",
        "is_datetime64_any_dtype",
        "is_datetime64_dtype",
        "is_datetime64_ns_dtype",
        "is_datetime64tz_dtype",
        "is_dtype_equal",
        "is_float",
        "is_float_dtype",
        "is_int64_dtype",
        "is_integer",
        "is_integer_dtype",
        "is_number",
        "is_numeric_dtype",
        "is_object_dtype",
        "is_scalar",
        "is_sparse",
        "is_string_dtype",
        "is_signed_integer_dtype",
        "is_timedelta64_dtype",
        "is_timedelta64_ns_dtype",
        "is_unsigned_integer_dtype",
        "is_period_dtype",
        "is_interval_dtype",
        "is_re",
        "is_re_compilable",
        "is_dict_like",
        "is_iterator",
        "is_file_like",
        "is_list_like",
        "is_hashable",
        "is_array_like",
        "is_named_tuple",
        "pandas_dtype",
        "union_categoricals",
        "infer_dtype",
        "is_extension_array_dtype",
    ]
    deprecated: list[str] = []
    dtypes = ["CategoricalDtype", "DatetimeTZDtype", "PeriodDtype", "IntervalDtype"]

    def test_types(self):
        # [FUNCTION DOC]
        # Checks that the types namespace contains exactly the allowed names plus the dtype names plus the
        # deprecated names (none), via the inherited check helper, ignoring private names.
        self.check(types, self.allowed + self.dtypes + self.deprecated)

    def test_deprecated_from_api_types(self):
        # [FUNCTION DOC]
        # Iterates over the deprecated names (currently empty, so nothing is exercised) and for each expects
        # a FutureWarning when it is called with a scalar argument.
        for t in self.deprecated:
            with tm.assert_produces_warning(FutureWarning):
                getattr(types, t)(1)
