# [MODULE DOC]
# Benchmarks for logical operations (or, and, exclusive or) on nullable boolean
# arrays, with a scalar or another array as the other operand. Public contents
# are the benchmark class TimeLogicalOps. No import-time side effects.
import numpy as np

import pandas as pd


class TimeLogicalOps:
    # [CLASS DOC]
    # Benchmark of the three logical operators on nullable boolean arrays of
    # 10000 elements, each with a random mask of missing values. No parameters.
    # State: left and right.
    def setup(self):
        # [FUNCTION DOC]
        # Draws four random boolean arrays of 10000 elements (data and mask for
        # each side) and wraps them as two boolean extension arrays left and
        # right.
        N = 10_000
        left, right, lmask, rmask = np.random.randint(0, 2, size=(4, N)).astype("bool")
        self.left = pd.arrays.BooleanArray(left, lmask)
        self.right = pd.arrays.BooleanArray(right, rmask)

    def time_or_scalar(self):
        # [FUNCTION DOC]
        # Times the logical or of the left array with the scalar true and then with
        # the scalar false.
        self.left | True
        self.left | False

    def time_or_array(self):
        # [FUNCTION DOC]
        # Times the logical or of the left and right arrays.
        self.left | self.right

    def time_and_scalar(self):
        # [FUNCTION DOC]
        # Times the logical and of the left array with the scalar true and then
        # with the scalar false.
        self.left & True
        self.left & False

    def time_and_array(self):
        # [FUNCTION DOC]
        # Times the logical and of the left and right arrays.
        self.left & self.right

    def time_xor_scalar(self):
        # [FUNCTION DOC]
        # Times the exclusive or of the left array with the scalar true and then
        # with the scalar false.
        self.left ^ True
        self.left ^ False

    def time_xor_array(self):
        # [FUNCTION DOC]
        # Times the exclusive or of the left and right arrays.
        self.left ^ self.right
