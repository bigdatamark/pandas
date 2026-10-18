# [MODULE DOC]
# Tests for the unary operators (positive, negative, invert, and related ufuncs)
# on DataFrame and Series, covering numeric, boolean, timedelta, object,
# Decimal and nullable extension dtypes, plus the error cases.
from decimal import Decimal

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


class TestDataFrameUnaryOperators:
    # [CLASS DOC]
    # Groups the tests of unary operators on DataFrame. Stateless; relies on fixtures
    # such as float_frame and using_infer_string.
    # __pos__, __neg__, __invert__

    @pytest.mark.parametrize(
        "df_data,expected_data",
        [
            ([-1, 1], [1, -1]),
            ([False, True], [True, False]),
            (pd.to_timedelta([-1, 1]), pd.to_timedelta([1, -1])),
        ],
    )
    def test_neg_numeric(self, df_data, expected_data):
        # [FUNCTION DOC]
        # Verifies negation of numeric, boolean and timedelta columns.
        # Parametrized over input data and expected data; negating the frame and the
        # column must equal the expected frame and series (booleans are logically
        # inverted).
        df = pd.DataFrame({"a": df_data})
        expected = pd.DataFrame({"a": expected_data})
        tm.assert_frame_equal(-df, expected)
        tm.assert_series_equal(-df["a"], expected["a"])

    @pytest.mark.parametrize(
        "df, expected",
        [
            (np.array([1, 2], dtype=object), np.array([-1, -2], dtype=object)),
            ([Decimal("1.0"), Decimal("2.0")], [Decimal("-1.0"), Decimal("-2.0")]),
        ],
    )
    def test_neg_object(self, df, expected):
        # [FUNCTION DOC]
        # Verifies negation of object-dtype columns holding integers or Decimal values
        # (GH#21380). Parametrized over input and expected data; negated frame and
        # column must equal the expected ones.
        # GH#21380
        df = pd.DataFrame({"a": df})
        expected = pd.DataFrame({"a": expected})
        tm.assert_frame_equal(-df, expected)
        tm.assert_series_equal(-df["a"], expected["a"])

    @pytest.mark.parametrize(
        "df_data",
        [
            ["a", "b"],
            pd.to_datetime(["2017-01-22", "1970-01-01"]),
        ],
    )
    def test_neg_raises(self, df_data, using_infer_string):
        # [FUNCTION DOC]
        # Verifies that negating string or datetime columns raises a type error.
        # Parametrized over the column data. The message may be any of several
        # operand-type or unsupported-dtype messages depending on the string backend.
        # The using_infer_string fixture is accepted but not used.
        df = pd.DataFrame({"a": df_data})
        msg = (
            "bad operand type for unary -: 'str'|"
            r"bad operand type for unary -: 'DatetimeArray'|"
            "unary '-' not supported for dtype"
        )
        with pytest.raises(TypeError, match=msg):
            (-df)
        with pytest.raises(TypeError, match=msg):
            (-df["a"])

    def test_invert(self, float_frame):
        # [FUNCTION DOC]
        # Verifies that negating a boolean frame equals inverting it.
        df = float_frame

        tm.assert_frame_equal(-(df < 0), ~(df < 0))

    def test_invert_mixed(self):
        # [FUNCTION DOC]
        # Verifies that inverting a frame with boolean and integer columns inverts bools
        # logically and applies bitwise complement to the integers; column labels are
        # reset by concatenating with ignore_index. Compared to a frame of ones.
        shape = (10, 5)
        df = pd.concat(
            [
                pd.DataFrame(np.zeros(shape, dtype="bool")),
                pd.DataFrame(np.zeros(shape, dtype=int)),
            ],
            axis=1,
            ignore_index=True,
        )
        result = ~df
        expected = pd.concat(
            [
                pd.DataFrame(np.ones(shape, dtype="bool")),
                pd.DataFrame(-np.ones(shape, dtype=int)),
            ],
            axis=1,
            ignore_index=True,
        )
        tm.assert_frame_equal(result, expected)

    def test_invert_empty_not_input(self):
        # [FUNCTION DOC]
        # Verifies that inverting an empty frame returns an equal but distinct object
        # (GH#51032).
        # GH#51032
        df = pd.DataFrame()
        result = ~df
        tm.assert_frame_equal(df, result)
        assert df is not result

    @pytest.mark.parametrize(
        "df_data",
        [
            [-1, 1],
            [False, True],
            pd.to_timedelta([-1, 1]),
        ],
    )
    def test_pos_numeric(self, df_data):
        # [FUNCTION DOC]
        # Verifies the unary plus on numeric, boolean and timedelta columns returns
        # equal
        # data for both frame and column (GH#16073). Parametrized over input data.
        # GH#16073
        df = pd.DataFrame({"a": df_data})
        tm.assert_frame_equal(+df, df)
        tm.assert_series_equal(+df["a"], df["a"])

    @pytest.mark.parametrize(
        "df_data",
        [
            np.array([-1, 2], dtype=object),
            [Decimal("-1.0"), Decimal("2.0")],
        ],
    )
    def test_pos_object(self, df_data):
        # [FUNCTION DOC]
        # Verifies unary plus on object-dtype columns (integers or Decimals) returns
        # equal
        # data (GH#21380). Parametrized over input data.
        # GH#21380
        df = pd.DataFrame({"a": df_data})
        tm.assert_frame_equal(+df, df)
        tm.assert_series_equal(+df["a"], df["a"])

    @pytest.mark.filterwarnings("ignore:Applying:DeprecationWarning")
    def test_pos_object_raises(self):
        # [FUNCTION DOC]
        # Verifies unary plus on a string column raises a type error with the bad
        # operand message (GH#21380). A deprecation warning about applying is ignored.
        # GH#21380
        df = pd.DataFrame({"a": ["a", "b"]})
        with pytest.raises(
            TypeError, match=r"^bad operand type for unary \+: \'str\'$"
        ):
            tm.assert_frame_equal(+df, df)

    def test_pos_raises(self):
        # [FUNCTION DOC]
        # Verifies unary plus on a datetime column raises a type error, for frame and
        # for column.
        df = pd.DataFrame({"a": pd.to_datetime(["2017-01-22", "1970-01-01"])})
        msg = r"bad operand type for unary \+: 'DatetimeArray'"
        with pytest.raises(TypeError, match=msg):
            (+df)
        with pytest.raises(TypeError, match=msg):
            (+df["a"])

    def test_unary_nullable(self):
        # [FUNCTION DOC]
        # Verifies positive, negative and absolute value (operators and the matching
        # numpy ufuncs) on a frame mixing nullable integer, nullable float, nullable
        # boolean and numpy boolean columns. Missing values must stay missing; both
        # paths must equal the expected frame.
        df = pd.DataFrame(
            {
                "a": pd.array([1, -2, 3, pd.NA], dtype="Int64"),
                "b": pd.array([4.0, -5.0, 6.0, pd.NA], dtype="Float32"),
                "c": pd.array([True, False, False, pd.NA], dtype="boolean"),
                # include numpy bool to make sure bool-vs-boolean behavior
                #  is consistent in non-NA locations
                "d": np.array([True, False, False, True]),
            }
        )

        result = +df
        res_ufunc = np.positive(df)
        expected = df
        # TODO: assert that we have copies?
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(res_ufunc, expected)

        result = -df
        res_ufunc = np.negative(df)
        expected = pd.DataFrame(
            {
                "a": pd.array([-1, 2, -3, pd.NA], dtype="Int64"),
                "b": pd.array([-4.0, 5.0, -6.0, pd.NA], dtype="Float32"),
                "c": pd.array([False, True, True, pd.NA], dtype="boolean"),
                "d": np.array([False, True, True, False]),
            }
        )
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(res_ufunc, expected)

        result = abs(df)
        res_ufunc = np.abs(df)
        expected = pd.DataFrame(
            {
                "a": pd.array([1, 2, 3, pd.NA], dtype="Int64"),
                "b": pd.array([4.0, 5.0, 6.0, pd.NA], dtype="Float32"),
                "c": pd.array([True, False, False, pd.NA], dtype="boolean"),
                "d": np.array([True, False, False, True]),
            }
        )
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(res_ufunc, expected)
