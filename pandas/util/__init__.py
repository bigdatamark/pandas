# [PACKAGE DOC]
# Public utility namespace of pandas. Submodules include decorators, doc tools,
# exception helpers, version printing, test decorators, the test entry point,
# argument validators and a vendored version parser. Nothing is imported
# eagerly: a few public names (array hashing, object hashing, and the
# decorator helpers for docstring appending, substitution and cached read-only
# properties) are resolved lazily through a module-level attribute hook, to
# avoid circular imports. No import-time side effects.
def __getattr__(key: str):
    # [FUNCTION DOC]
    # Module-level lazy attribute hook. Input: key, the requested attribute
    # name. For the five supported names (the array hash function, the object
    # hash function from the core hashing module, and the appender,
    # substitution and cached-read-only-property helpers from the decorators
    # module) it imports the target at call time and returns it. Any other
    # name raises an attribute error naming this module.
    # These imports need to be lazy to avoid circular import errors
    if key == "hash_array":
        from pandas.core.util.hashing import hash_array

        return hash_array
    if key == "hash_pandas_object":
        from pandas.core.util.hashing import hash_pandas_object

        return hash_pandas_object
    if key == "Appender":
        from pandas.util._decorators import Appender

        return Appender
    if key == "Substitution":
        from pandas.util._decorators import Substitution

        return Substitution

    if key == "cache_readonly":
        from pandas.util._decorators import cache_readonly

        return cache_readonly

    raise AttributeError(f"module 'pandas.util' has no attribute '{key}'")


def __dir__() -> list[str]:
    # [FUNCTION DOC]
    # Module-level directory hook. Returns a list of all current module global
    # names followed by the two hashing names. Takes no input.
    # KNOWN ISSUE: the three decorator helper names that the attribute hook
    # also resolves lazily are not included in the returned listing.
    return [*list(globals().keys()), "hash_array", "hash_pandas_object"]
