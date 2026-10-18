# [PACKAGE DOC]
# Test-support package for groupby tests, defining one shared helper for
# building method arguments.
def get_groupby_method_args(name, obj):
    # [FUNCTION DOC]
    # HELPER: shared by groupby tests parametrized over method names.
    # Returns the extra positional arguments a groupby method needs. Parameters:
    # name is the method name, obj is the grouped Series or DataFrame.
    # Returns a tuple: a single zero for the nth and take methods, a single 0.5
    # for quantile, a single-element tuple holding obj for corrwith, otherwise an empty
    # tuple.
    """
    Get required arguments for a groupby method.

    When parametrizing a test over groupby methods (e.g. "sum", "mean"),
    it is often the case that arguments are required for certain methods.

    Parameters
    ----------
    name: str
        Name of the method.
    obj: Series or DataFrame
        pandas object that is being grouped.

    Returns
    -------
    A tuple of required arguments for the method.
    """
    if name in ("nth", "take"):
        return (0,)
    if name == "quantile":
        return (0.5,)
    if name == "corrwith":
        return (obj,)
    return ()
