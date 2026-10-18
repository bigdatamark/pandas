# [MODULE DOC]
# Cython base class for the pandas indexer objects (.loc, .iloc, .at, .iat).
# Provides fast construction and cached access to the dimensionality of the
# indexed object; the Python subclasses in pandas/core/indexing.py add the logic.
cdef class NDFrameIndexerBase:
    # [CLASS DOC]
    # Base for the _NDFrameIndexer family. Stores the indexer name (e.g. "loc"),
    # the indexed Series/DataFrame (obj, public) and a lazily computed ndim cache
    # (private integer, -1 meaning "not yet computed").
    """
    A base class for _NDFrameIndexer for fast instantiation and attribute access.
    """
    cdef:
        Py_ssize_t _ndim

    cdef public:
        str name
        object obj

    def __init__(self, name: str, obj):
        # [FUNCTION DOC]
        # Initialize the indexer.
        # Inputs: name - indexer name string; obj - the Series/DataFrame being indexed.
        # Outputs: none.
        # Side effects: sets obj and name attributes; sets cached ndim to -1 (unknown).
        self.obj = obj
        self.name = name
        self._ndim = -1

    @property
    def ndim(self) -> int:
        # [FUNCTION DOC]
        # Property returning the number of dimensions of the indexed object.
        # Inputs: none (uses self.obj).
        # Outputs: integer ndim (1 or 2). Computed on first access by reading obj.ndim
        #   and
        # cached for later calls.
        # Exceptions: ValueError if obj.ndim is greater than 2.
        # Side effects: populates the ndim cache on first call (cache is set before the
        # dimension check, so a later call after an error returns the >2 value
        #   silently).
        # KNOWN ISSUE: the cached value is stored before the ndim > 2 check, so a second
        # access after the ValueError returns the invalid ndim instead of raising again.
        # Delay `ndim` instantiation until required as reading it
        # from `obj` isn't entirely cheap.
        ndim = self._ndim
        if ndim == -1:
            ndim = self._ndim = self.obj.ndim
            if ndim > 2:
                raise ValueError(  # pragma: no cover
                    "NDFrameIndexer does not support NDFrame objects with ndim > 2"
                )
        return ndim
