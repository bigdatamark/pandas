# [MODULE DOC]
# Benchmark of an internal date-string detection helper from the compiled datetime
# parsing module. Public contents: the benchmark class DoesStringLookLikeDatetime.
# Import-time side effect: tries to import the private helper function from the
# compiled library; on failure (older versions) it silently continues, leaving the
# name undefined so the benchmark would fail at run time (the guard exists to keep
# asv from failing the whole suite import). Does not re-export the common asv
# setup hook.
try:
    from pandas._libs.tslibs.parsing import _does_string_look_like_datetime
except ImportError:
    # Avoid whole benchmark suite import failure on asv (currently 0.4)
    pass


class DoesStringLookLikeDatetime:
    # [CLASS DOC]
    # Benchmark of testing whether strings look like datetimes. Parametrized by the
    # sample string (a quarter notation, a decimal number, an integer-like
    # string); the parameter list is given as a one-element tuple. State: objects.
    params = (["2Q2005", "0.0", "10000"],)
    param_names = ["value"]

    def setup(self, value):
        # [FUNCTION DOC]
        # Builds a list of one million copies of the sample string as objects.
        self.objects = [value] * 1000000

    def time_check_datetimes(self, value):
        # [FUNCTION DOC]
        # Times calling the helper on every string in the list, discarding results.
        # The value parameter is unused here.
        for obj in self.objects:
            _does_string_look_like_datetime(obj)
