# [MODULE DOC]
# Benchmark of the metadata propagation hook (finalize) for Series and DataFrame
# objects that carry many user attributes. Public contents are the benchmark
# class Finalize. No import-time side effects.
import pandas as pd


class Finalize:
    # [CLASS DOC]
    # Benchmark of calling the finalize hook on an empty float series or frame
    # holding 1000 attributes. Parametrized by the container class (Series or
    # DataFrame; the parameter names list shows series and frame). State: obj.
    param_names = ["series", "frame"]
    params = [pd.Series, pd.DataFrame]

    def setup(self, param):
        # [FUNCTION DOC]
        # Creates an empty object of the given class with float dtype and stores
        # 1000 attributes in its attrs mapping (keys and values are the integers
        # zero to 999). Stores the object as obj.
        # KNOWN ISSUE: param_names lists two names but params holds a single list
        # of classes, so there is only one parameter axis; the names are
        # inconsistent with the parameter structure.
        N = 1000
        obj = param(dtype=float)
        for i in range(N):
            obj.attrs[i] = i
        self.obj = obj

    def time_finalize_micro(self, param):
        # [FUNCTION DOC]
        # Times calling the finalize method of the object with itself as the
        # source and the method label set to the finalize name. Result is
        # discarded.
        self.obj.__finalize__(self.obj, method="__finalize__")
