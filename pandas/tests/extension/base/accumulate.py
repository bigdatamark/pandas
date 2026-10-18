# [MODULE DOC]
# Base tests for cumulative operations (cumsum, cummin and similar) on Series backed by an
# extension array. Public contents: BaseAccumulateTests mixin. Subclasses declare which
# accumulations they support by overriding the support hook. Uses the data and
# all_numeric_accumulations fixtures from the conftest.
import pytest

import pandas as pd
import pandas._testing as tm


class BaseAccumulateTests:
    # [CLASS DOC]
    # Mixin for accumulation tests. Meaningful mainly for numeric and boolean dtypes. Subclass
    # authors override the support hook to opt in and may override the check method to customize
    # the comparison.
    """
    Accumulation specific tests. Generally these only
    make sense for numeric/boolean operations.
    """

    def _supports_accumulation(self, ser: pd.Series, op_name: str) -> bool:
        # [FUNCTION DOC]
        # HELPER: hook deciding whether the named accumulation is expected to work for the given
        # series. Inputs: ser (series under test), op_name (accumulation name). Returns a boolean;
        # the default always answers false, so subclasses must override to enable the success path.
        # Do we expect this accumulation to be supported for this dtype?
        # We default to assuming "no"; subclass authors should override here.
        return False

    def check_accumulate(self, ser: pd.Series, op_name: str, skipna: bool):
        # [FUNCTION DOC]
        # HELPER: compares an accumulation on the extension series against the same accumulation on
        # a reference series. Inputs: ser, op_name, skipna (bool). The reference is the series cast
        # to float64, falling back to object dtype when the cast raises a type or value error (for
        # example period or string data). Calls the named accumulation with the skipna flag on both and
        # asserts the series are equal, ignoring dtype.
        try:
            alt = ser.astype("float64")
        except (TypeError, ValueError):
            # e.g. Period can't be cast to float64 (TypeError)
            #      String can't be cast to float64 (ValueError)
            alt = ser.astype(object)

        result = getattr(ser, op_name)(skipna=skipna)
        expected = getattr(alt, op_name)(skipna=skipna)
        tm.assert_series_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize("skipna", [True, False])
    def test_accumulate_series(self, data, all_numeric_accumulations, skipna):
        # [FUNCTION DOC]
        # Parametrized over skipna true and false and over all numeric accumulation names
        # (from the fixture). Builds a series from the data fixture. If the support hook says the
        # operation is supported, runs the comparison helper. Otherwise expects the accumulation to
        # raise either a not-implemented error or a type error.
        op_name = all_numeric_accumulations
        ser = pd.Series(data)

        if self._supports_accumulation(ser, op_name):
            self.check_accumulate(ser, op_name, skipna)
        else:
            with pytest.raises((NotImplementedError, TypeError)):
                # TODO: require TypeError for things that will _never_ work?
                getattr(ser, op_name)(skipna=skipna)
