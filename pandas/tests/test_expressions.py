# [MODULE DOC]
# Tests for the optional fast-evaluation layer (numexpr) in the core computation expressions module:
# that arithmetic, comparison and where operations give identical results with the layer enabled and
# disabled, the eligibility check, boolean operator handling, and Python-semantics preservation for
# modulo and floor division. Contains frame and array fixtures of various dtypes and sizes (some larger
# than the minimum element threshold, some smaller). All tests are in one class that is skipped when
# numexpr is not in use.
import operator
import re

import numpy as np
import pytest

from pandas.compat._optional import import_optional_dependency

from pandas import option_context
import pandas._testing as tm
from pandas.core.api import DataFrame
from pandas.core.computation import expressions as expr
from pandas.util.version import Version


@pytest.fixture
def _frame():
    # [FUNCTION DOC]
    # HELPER: fixture returning a 10001-row by 4-column float64 frame of random normal values, columns A to D.
    return DataFrame(
        np.random.default_rng(2).standard_normal((10001, 4)),
        columns=list("ABCD"),
        dtype="float64",
    )


@pytest.fixture
def _frame2():
    # [FUNCTION DOC]
    # HELPER: fixture returning a small (100-row by 4-column) float64 frame of random normal values, columns A to D.
    return DataFrame(
        np.random.default_rng(2).standard_normal((100, 4)),
        columns=list("ABCD"),
        dtype="float64",
    )


@pytest.fixture
def _mixed(_frame):
    # [FUNCTION DOC]
    # HELPER: fixture building a frame from the large float fixture with column B cast to float32, C to int64, D to int32.
    return DataFrame(
        {
            "A": _frame["A"],
            "B": _frame["B"].astype("float32"),
            "C": _frame["C"].astype("int64"),
            "D": _frame["D"].astype("int32"),
        }
    )


@pytest.fixture
def _mixed2(_frame2):
    # [FUNCTION DOC]
    # HELPER: fixture like the mixed frame but built from the small float frame.
    return DataFrame(
        {
            "A": _frame2["A"],
            "B": _frame2["B"].astype("float32"),
            "C": _frame2["C"].astype("int64"),
            "D": _frame2["D"].astype("int32"),
        }
    )


@pytest.fixture
def _integer():
    # [FUNCTION DOC]
    # HELPER: fixture returning a 10001-row by 4-column int64 frame of random integers from 1 to 99.
    return DataFrame(
        np.random.default_rng(2).integers(1, 100, size=(10001, 4)),
        columns=list("ABCD"),
        dtype="int64",
    )


@pytest.fixture
def _integer_integers(_integer):
    # [FUNCTION DOC]
    # HELPER: fixture multiplying the integer frame by random zeros and ones, giving integers with many zeros (for division edge cases).
    # integers to get a case with zeros
    return _integer * np.random.default_rng(2).integers(0, 2, size=np.shape(_integer))


@pytest.fixture
def _integer2():
    # [FUNCTION DOC]
    # HELPER: fixture returning a small (101-row) int64 frame of random integers from 1 to 99.
    return DataFrame(
        np.random.default_rng(2).integers(1, 100, size=(101, 4)),
        columns=list("ABCD"),
        dtype="int64",
    )


@pytest.fixture
def _array(_frame):
    # [FUNCTION DOC]
    # HELPER: fixture returning column A of the large float frame as a NumPy array.
    return _frame["A"].to_numpy()


@pytest.fixture
def _array2(_frame2):
    # [FUNCTION DOC]
    # HELPER: fixture returning column A of the small float frame as a NumPy array.
    return _frame2["A"].to_numpy()


@pytest.fixture
def _array_mixed(_mixed):
    # [FUNCTION DOC]
    # HELPER: fixture returning column D (int32) of the large mixed frame as a NumPy array.
    return _mixed["D"].to_numpy()


@pytest.fixture
def _array_mixed2(_mixed2):
    # [FUNCTION DOC]
    # HELPER: fixture returning column D (int32) of the small mixed frame as a NumPy array.
    return _mixed2["D"].to_numpy()


@pytest.mark.skipif(not expr.USE_NUMEXPR, reason="not using numexpr")
class TestExpressions:
    # [CLASS DOC]
    # Test class skipped when numexpr is not enabled. Compares results with the numexpr path against results
    # with the "compute.use_numexpr" option turned off.
    @staticmethod
    def call_op(df, other, flex: bool, opname: str):
        # [FUNCTION DOC]
        # HELPER: static method. Parameters: df, the left operand; other, the right operand; flex, whether to call the flexible
        # method named by opname on the left operand (true) or the operator-module function of that name (false); opname, the operation
        # name. Computes the expected value with numexpr disabled through the option context, then calls the module's test-result getter
        # (which resets the used-numexpr flag), computes the result again with the default setting, and returns the pair (result, expected).
        if flex:
            op = lambda x, y: getattr(x, opname)(y)
            op.__name__ = opname
        else:
            op = getattr(operator, opname)

        with option_context("compute.use_numexpr", False):
            expected = op(df, other)

        expr.get_test_result()

        result = op(df, other)
        return result, expected

    @pytest.mark.parametrize(
        "fixture",
        [
            "_integer",
            "_integer2",
            "_integer_integers",
            "_frame",
            "_frame2",
            "_mixed",
            "_mixed2",
        ],
    )
    @pytest.mark.parametrize("flex", [True, False])
    @pytest.mark.parametrize(
        "arith", ["add", "sub", "mul", "mod", "truediv", "floordiv"]
    )
    def test_run_arithmetic(self, request, fixture, flex, arith, monkeypatch):
        # [FUNCTION DOC]
        # Parametrized over seven frame fixtures, flex or operator form, and six arithmetic operations. With the minimum-element threshold
        # patched to zero so numexpr is always eligible, verifies result equals the numexpr-disabled expected result for whole frames and
        # for each column; for true division verifies every resulting dtype is a float kind.
        df = request.getfixturevalue(fixture)
        with monkeypatch.context() as m:
            m.setattr(expr, "_MIN_ELEMENTS", 0)
            result, expected = self.call_op(df, df, flex, arith)

            if arith == "truediv":
                assert all(x.kind == "f" for x in expected.dtypes.values)
            tm.assert_equal(expected, result)

            for i in range(len(df.columns)):
                result, expected = self.call_op(
                    df.iloc[:, i], df.iloc[:, i], flex, arith
                )
                if arith == "truediv":
                    assert expected.dtype.kind == "f"
                tm.assert_equal(expected, result)

    @pytest.mark.parametrize(
        "fixture",
        [
            "_integer",
            "_integer2",
            "_integer_integers",
            "_frame",
            "_frame2",
            "_mixed",
            "_mixed2",
        ],
    )
    @pytest.mark.parametrize("flex", [True, False])
    def test_run_binary(self, request, fixture, flex, comparison_op, monkeypatch):
        # [FUNCTION DOC]
        # Parametrized over frame fixtures, flex or operator form and the comparison operator fixture. The docstring says it tests solely that
        # the result is the same with and without numexpr. Builds the right operand as the frame plus one with numexpr off, patches the threshold
        # to zero and turns on test mode, then verifies numexpr was actually used (the test-result getter is true) and the result equals the expected
        # one; also runs per-column addition for side effects only.
        """
        tests solely that the result is the same whether or not numexpr is
        enabled.  Need to test whether the function does the correct thing
        elsewhere.
        """
        df = request.getfixturevalue(fixture)
        arith = comparison_op.__name__
        with option_context("compute.use_numexpr", False):
            other = df + 1

        with monkeypatch.context() as m:
            m.setattr(expr, "_MIN_ELEMENTS", 0)
            expr.set_test_mode(True)

            result, expected = self.call_op(df, other, flex, arith)

            used_numexpr = expr.get_test_result()
            assert used_numexpr, "Did not use numexpr as expected."
            tm.assert_equal(expected, result)

            for i in range(len(df.columns)):
                binary_comp = other.iloc[:, i] + 1
                self.call_op(df.iloc[:, i], binary_comp, flex, "add")

    def test_invalid(self):
        # [FUNCTION DOC]
        # Verifies the eligibility helper returns false when the operator string is None and when both arrays are small (below the minimum elements),
        # and true when only the first array is large (only the first operand is checked).
        array = np.random.default_rng(2).standard_normal(1_000_001)
        array2 = np.random.default_rng(2).standard_normal(100)

        # no op
        result = expr._can_use_numexpr(operator.add, None, array, array, "evaluate")
        assert not result

        # min elements
        result = expr._can_use_numexpr(operator.add, "+", array2, array2, "evaluate")
        assert not result

        # ok, we only check on first part of expression
        result = expr._can_use_numexpr(operator.add, "+", array, array2, "evaluate")
        assert result

    @pytest.mark.filterwarnings("ignore:invalid value encountered in:RuntimeWarning")
    @pytest.mark.parametrize(
        "opname,op_str",
        [("add", "+"), ("sub", "-"), ("mul", "*"), ("truediv", "/"), ("pow", "**")],
    )
    @pytest.mark.parametrize(
        "left_fix,right_fix", [("_array", "_array2"), ("_array_mixed", "_array_mixed2")]
    )
    def test_binary_ops(self, request, opname, op_str, left_fix, right_fix):
        # [FUNCTION DOC]
        # Ignores invalid-value runtime warnings. Parametrized over five operations and two (left, right) array fixture pairs. Verifies the evaluate
        # function gives equal arrays with numexpr enabled and disabled, and that the eligibility check is false for the small right-hand arrays; runs the
        # check with numexpr disabled, with one thread, and with the default number of threads.
        left = request.getfixturevalue(left_fix)
        right = request.getfixturevalue(right_fix)

        def testit(left, right, opname, op_str):
            # [FUNCTION DOC]
            # HELPER: local function for test_binary_ops. Takes left, right arrays, the operation name and its symbol. For power, first takes the absolute value of
            # the left array. Compares evaluate with numexpr on and off on the left array alone and asserts the eligibility check is false for the small right array.
            if opname == "pow":
                left = np.abs(left)

            op = getattr(operator, opname)

            # array has 0s
            result = expr.evaluate(op, left, left, use_numexpr=True)
            expected = expr.evaluate(op, left, left, use_numexpr=False)
            tm.assert_numpy_array_equal(result, expected)

            result = expr._can_use_numexpr(op, op_str, right, right, "evaluate")
            assert not result

        with option_context("compute.use_numexpr", False):
            testit(left, right, opname, op_str)

        expr.set_numexpr_threads(1)
        testit(left, right, opname, op_str)
        expr.set_numexpr_threads()
        testit(left, right, opname, op_str)

    @pytest.mark.parametrize(
        "left_fix,right_fix", [("_array", "_array2"), ("_array_mixed", "_array_mixed2")]
    )
    def test_comparison_ops(self, request, comparison_op, left_fix, right_fix):
        # [FUNCTION DOC]
        # Parametrized over comparison operators and array fixture pairs. Verifies comparison evaluation gives equal boolean arrays with numexpr enabled and disabled
        # and that the eligibility check is false for the small arrays; run with numexpr disabled, one thread and default threads.
        left = request.getfixturevalue(left_fix)
        right = request.getfixturevalue(right_fix)

        def testit():
            # [FUNCTION DOC]
            # HELPER: local function for test_comparison_ops. Compares evaluation on the left array and left-plus-one with numexpr on and off, then asserts the eligibility
            # check is false for the small right pair.
            f12 = left + 1
            f22 = right + 1

            op = comparison_op

            result = expr.evaluate(op, left, f12, use_numexpr=True)
            expected = expr.evaluate(op, left, f12, use_numexpr=False)
            tm.assert_numpy_array_equal(result, expected)

            result = expr._can_use_numexpr(op, op, right, f22, "evaluate")
            assert not result

        with option_context("compute.use_numexpr", False):
            testit()

        expr.set_numexpr_threads(1)
        testit()
        expr.set_numexpr_threads()
        testit()

    @pytest.mark.parametrize("cond", [True, False])
    @pytest.mark.parametrize("fixture", ["_frame", "_frame2", "_mixed", "_mixed2"])
    def test_where(self, request, cond, fixture):
        # [FUNCTION DOC]
        # Parametrized over a boolean fill and frame fixtures. Verifies the module's where function (with an all-true or all-false mask) equals NumPy's where, with numexpr
        # disabled, one thread, and default threads.
        df = request.getfixturevalue(fixture)

        def testit():
            # [FUNCTION DOC]
            # HELPER: local function for test_where. Builds an all-cond boolean mask and compares the module's where against NumPy's where on the frame values and the values plus one.
            c = np.empty(df.shape, dtype=np.bool_)
            c.fill(cond)
            result = expr.where(c, df.values, df.values + 1)
            expected = np.where(c, df.values, df.values + 1)
            tm.assert_numpy_array_equal(result, expected)

        with option_context("compute.use_numexpr", False):
            testit()

        expr.set_numexpr_threads(1)
        testit()
        expr.set_numexpr_threads()
        testit()

    @pytest.mark.parametrize(
        "op_str,opname", [("/", "truediv"), ("//", "floordiv"), ("**", "pow")]
    )
    def test_bool_ops_raise_on_arithmetic(self, op_str, opname):
        # [FUNCTION DOC]
        # Parametrized over true division, floor division and power. On a frame of random booleans verifies each operator raises a not-implemented error saying the operator is not
        # implemented for boolean dtypes, for frame-frame, Series-Series, Series-scalar, scalar-Series, scalar-frame and frame-scalar combinations.
        # KNOWN ISSUE: the parametrization includes an operator-symbol parameter that is never used; only the operation name matters.
        df = DataFrame(
            {
                "a": np.random.default_rng(2).random(10) > 0.5,
                "b": np.random.default_rng(2).random(10) > 0.5,
            }
        )

        msg = f"operator '{opname}' not implemented for bool dtypes"
        f = getattr(operator, opname)
        err_msg = re.escape(msg)

        with pytest.raises(NotImplementedError, match=err_msg):
            f(df, df)

        with pytest.raises(NotImplementedError, match=err_msg):
            f(df.a, df.b)

        with pytest.raises(NotImplementedError, match=err_msg):
            f(df.a, True)

        with pytest.raises(NotImplementedError, match=err_msg):
            f(False, df.a)

        with pytest.raises(NotImplementedError, match=err_msg):
            f(False, df)

        with pytest.raises(NotImplementedError, match=err_msg):
            f(df, True)

    @pytest.mark.parametrize(
        "op_str,opname", [("+", "add"), ("*", "mul"), ("-", "sub")]
    )
    def test_bool_ops_warn_on_arithmetic(self, op_str, opname, monkeypatch):
        # [FUNCTION DOC]
        # Parametrized over addition, multiplication and subtraction (symbol and name). With the threshold patched to five and numexpr on, verifies the arithmetic operator on boolean data equals
        # the corresponding logical operator (or, and) for all six operand combinations. If numexpr is installed with a version below 2.13.1 and the operation is addition or multiplication, a user
        # warning that the operator is not supported by numexpr is expected; otherwise no warning is expected.
        # KNOWN ISSUE: for subtraction the test returns early (comment says it raises a type error) before checking anything, so that parameter case verifies nothing; the xor lookup computed earlier is then unused.
        n = 10
        df = DataFrame(
            {
                "a": np.random.default_rng(2).random(n) > 0.5,
                "b": np.random.default_rng(2).random(n) > 0.5,
            }
        )

        subs = {"+": "|", "*": "&", "-": "^"}
        sub_funcs = {"|": "or_", "&": "and_", "^": "xor"}

        f = getattr(operator, opname)
        fe = getattr(operator, sub_funcs[subs[op_str]])

        if op_str == "-":
            # raises TypeError
            return

        msg = "operator is not supported by numexpr"
        ne = import_optional_dependency("numexpr", errors="ignore")
        warning = (
            UserWarning
            if ne
            and op_str in {"+", "*"}
            and Version(ne.__version__) < Version("2.13.1")
            else None
        )
        with monkeypatch.context() as m:
            m.setattr(expr, "_MIN_ELEMENTS", 5)
            with option_context("compute.use_numexpr", True):
                with tm.assert_produces_warning(warning, match=msg):
                    r = f(df, df)
                    e = fe(df, df)
                    tm.assert_frame_equal(r, e)

                with tm.assert_produces_warning(warning, match=msg):
                    r = f(df.a, df.b)
                    e = fe(df.a, df.b)
                    tm.assert_series_equal(r, e)

                with tm.assert_produces_warning(warning, match=msg):
                    r = f(df.a, True)
                    e = fe(df.a, True)
                    tm.assert_series_equal(r, e)

                with tm.assert_produces_warning(warning, match=msg):
                    r = f(False, df.a)
                    e = fe(False, df.a)
                    tm.assert_series_equal(r, e)

                with tm.assert_produces_warning(warning, match=msg):
                    r = f(False, df)
                    e = fe(False, df)
                    tm.assert_frame_equal(r, e)

                with tm.assert_produces_warning(warning, match=msg):
                    r = f(df, True)
                    e = fe(df, True)
                    tm.assert_frame_equal(r, e)

    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (
                DataFrame(
                    [[0, 1, 2, "aa"], [0, 1, 2, "aa"]], columns=["a", "b", "c", "dtype"]
                ),
                DataFrame([[False, False], [False, False]], columns=["a", "dtype"]),
            ),
            (
                DataFrame(
                    [[0, 3, 2, "aa"], [0, 4, 2, "aa"], [0, 1, 1, "bb"]],
                    columns=["a", "b", "c", "dtype"],
                ),
                DataFrame(
                    [[False, False], [False, False], [False, False]],
                    columns=["a", "dtype"],
                ),
            ),
        ],
    )
    def test_bool_ops_column_name_dtype(self, test_input, expected):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Parametrized over two frames, each having a column literally named dtype. Verifies the not-equal comparison of a frame column subset with itself works and
        # gives an all-false boolean frame.
        # GH 22383 - .ne fails if columns containing column name 'dtype'
        result = test_input.loc[:, ["a", "dtype"]].ne(test_input.loc[:, ["a", "dtype"]])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "arith", ("add", "sub", "mul", "mod", "truediv", "floordiv")
    )
    @pytest.mark.parametrize("axis", (0, 1))
    def test_frame_series_axis(self, axis, arith, _frame, monkeypatch):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Parametrized over axis and six arithmetic methods, using the large float frame fixture. With threshold zero, verifies a flexible arithmetic method of the frame
        # with a Series (first row for columns axis, first column for index axis) gives the same result with numexpr disabled and enabled.
        # GH#26736 Dataframe.floordiv(Series, axis=1) fails

        df = _frame
        if axis == 1:
            other = df.iloc[0, :]
        else:
            other = df.iloc[:, 0]

        with monkeypatch.context() as m:
            m.setattr(expr, "_MIN_ELEMENTS", 0)

            op_func = getattr(df, arith)

            with option_context("compute.use_numexpr", False):
                expected = op_func(other, axis=axis)

            result = op_func(other, axis=axis)
            tm.assert_frame_equal(expected, result)

    @pytest.mark.parametrize(
        "op",
        [
            "__mod__",
            "__rmod__",
            "__floordiv__",
            "__rfloordiv__",
        ],
    )
    @pytest.mark.parametrize("scalar", [-5, 5])
    def test_python_semantics_with_numexpr_installed(
        self, op, box_with_array, scalar, monkeypatch
    ):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Parametrized over modulo and floor division (and reflected versions), a negative and a positive scalar, and the box fixture. With threshold zero, applies the
        # method to boxed integers from minus fifty to forty-nine; verifies equal to the numexpr-disabled result and, element by element, to the Python integer method result (skipping division by zero).
        # https://github.com/pandas-dev/pandas/issues/36047
        with monkeypatch.context() as m:
            m.setattr(expr, "_MIN_ELEMENTS", 0)
            data = np.arange(-50, 50)
            obj = box_with_array(data)
            method = getattr(obj, op)
            result = method(scalar)

            # compare result with numpy
            with option_context("compute.use_numexpr", False):
                expected = method(scalar)

            tm.assert_equal(result, expected)

            # compare result element-wise with Python
            for i, elem in enumerate(data):
                if box_with_array == DataFrame:
                    scalar_result = result.iloc[i, 0]
                else:
                    scalar_result = result[i]
                try:
                    expected = getattr(int(elem), op)(scalar)
                except ZeroDivisionError:
                    pass
                else:
                    assert scalar_result == expected
