# [MODULE DOC]
# Pandas implementation of the interchange protocol's frame object.
# Public content: the private frame wrapper class, which subclasses the
# protocol frame class (imported under an alias) and is what the frame's
# interchange export method returns. It builds on the column wrapper
# (columns are exposed as column wrapper objects) and on the rechunking helper
# from the utils module. No import-time side effects.
from __future__ import annotations

from collections import abc
from typing import TYPE_CHECKING

from pandas.core.interchange.column import PandasColumn
from pandas.core.interchange.dataframe_protocol import DataFrame as DataFrameXchg
from pandas.core.interchange.utils import maybe_rechunk

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
        Sequence,
    )

    from pandas import (
        DataFrame,
        Index,
    )


class PandasDataFrameXchg(DataFrameXchg):
    # [CLASS DOC]
    # Interchange-protocol view of a pandas frame; subclass of the protocol frame.
    # State: a private frame whose column labels were converted to text (so the
    # original is never mutated) and a private flag saying whether copies are
    # allowed. Invariant: any Arrow-backed column with multiple chunks has been
    # combined into one chunk at construction (or construction failed). Always
    # reports a single chunk unless asked to split. Row labels are exposed only
    # through the metadata property.
    """
    A data frame class, with only the methods required by the interchange
    protocol defined.
    Instances of this (private) class are returned from
    ``pd.DataFrame.__dataframe__`` as objects with the methods and
    attributes defined on this class.
    """

    def __init__(self, df: DataFrame, allow_copy: bool = True) -> None:
        # [FUNCTION DOC]
        # Build the wrapper. Parameters: df (the frame to export); allow_copy (default
        # true) says whether data copies are permitted.
        # Stores a renamed frame with every column label converted to text and the
        # flag. Then for each column position, rechunks Arrow-backed multi-chunk
        # columns via the utils helper and replaces the column by position in the
        # internal frame. May raise a runtime error from the helper when a rechunk is
        # needed but copying is disallowed. Returns nothing.
        """
        Constructor - an instance of this (private) class is returned from
        `pd.DataFrame.__dataframe__`.
        """
        self._df = df.rename(columns=str)
        self._allow_copy = allow_copy
        for i, _col in enumerate(self._df.columns):
            rechunked = maybe_rechunk(self._df.iloc[:, i], allow_copy=allow_copy)
            if rechunked is not None:
                self._df.isetitem(i, rechunked)

    def __dataframe__(
        self, nan_as_null: bool = False, allow_copy: bool = True
    ) -> PandasDataFrameXchg:
        # [FUNCTION DOC]
        # Protocol method returning a new wrapper over the same internal frame.
        # Parameters: nan_as_null (default false) is accepted but ignored; allow_copy
        # (default true) is passed to the new wrapper.
        # `nan_as_null` can be removed here once it's removed from
        # Dataframe.__dataframe__
        return PandasDataFrameXchg(self._df, allow_copy)

    @property
    def metadata(self) -> dict[str, Index]:
        # [FUNCTION DOC]
        # Property returning a mapping with one entry, keyed by the pandas index key,
        # whose value is the frame's row index (row labels are not part of the
        # protocol so they are exported as library-specific metadata).
        # `index` isn't a regular column, and the protocol doesn't support row
        # labels - so we export it as Pandas-specific metadata here.
        return {"pandas.index": self._df.index}

    def num_columns(self) -> int:
        # [FUNCTION DOC]
        # Return the number of columns as an integer.
        return len(self._df.columns)

    def num_rows(self) -> int:
        # [FUNCTION DOC]
        # Return the number of rows as an integer.
        return len(self._df)

    def num_chunks(self) -> int:
        # [FUNCTION DOC]
        # Return the number of chunks of the frame; always 1.
        return 1

    def column_names(self) -> Index:
        # [FUNCTION DOC]
        # Return the column labels of the internal (text-converted) frame as an index.
        return self._df.columns

    def get_column(self, i: int) -> PandasColumn:
        # [FUNCTION DOC]
        # Return the column at integer position i, wrapped in a column wrapper that
        # inherits this object's copy flag.
        return PandasColumn(self._df.iloc[:, i], allow_copy=self._allow_copy)

    def get_column_by_name(self, name: str) -> PandasColumn:
        # [FUNCTION DOC]
        # Return the column with the given label (a text name) wrapped in a column
        # wrapper with the same copy flag. A missing label raises the usual lookup
        # error from the frame.
        return PandasColumn(self._df[name], allow_copy=self._allow_copy)

    def get_columns(self) -> list[PandasColumn]:
        # [FUNCTION DOC]
        # Return a list of column wrappers, one per column label in order, each with
        # this object's copy flag. Looks columns up by label.
        return [
            PandasColumn(self._df[name], allow_copy=self._allow_copy)
            for name in self._df.columns
        ]

    def select_columns(self, indices: Sequence[int]) -> PandasDataFrameXchg:
        # [FUNCTION DOC]
        # Return a new wrapper containing the columns at the given positions.
        # Parameter indices: a sequence of integers. Raises a value error if it is not
        # a sequence; non-list sequences are converted to a list. The new wrapper keeps
        # the copy flag.
        if not isinstance(indices, abc.Sequence):
            raise ValueError("`indices` is not a sequence")
        if not isinstance(indices, list):
            indices = list(indices)

        return PandasDataFrameXchg(
            self._df.iloc[:, indices], allow_copy=self._allow_copy
        )

    def select_columns_by_name(self, names: list[str]) -> PandasDataFrameXchg:  # type: ignore[override]
        # [FUNCTION DOC]
        # Return a new wrapper containing the columns with the given labels.
        # Parameter names: a sequence of text labels. Raises a value error if it is not
        # a sequence; non-list sequences are converted to a list. Selection is by
        # label, and the new wrapper keeps the copy flag. (The signature intentionally
        # differs from the protocol base signature, hence the type-ignore.)
        if not isinstance(names, abc.Sequence):
            raise ValueError("`names` is not a sequence")
        if not isinstance(names, list):
            names = list(names)

        return PandasDataFrameXchg(self._df.loc[:, names], allow_copy=self._allow_copy)

    def get_chunks(self, n_chunks: int | None = None) -> Iterable[PandasDataFrameXchg]:
        # [FUNCTION DOC]
        # Generator yielding wrapper objects for consecutive row slices.
        # Parameter n_chunks: optional integer. If it is greater than 1, the rows are
        # split into n_chunks pieces with the chunk length being the ceiling of the
        # row count divided by n_chunks; each slice is wrapped with the copy flag (the
        # last pieces may be short or empty). Otherwise yields this object once.
        # KNOWN ISSUE: for a frame with zero rows and n_chunks above 1 the computed chunk length is zero, and the range used to iterate has a zero step, which raises a value error.
        """
        Return an iterator yielding the chunks.
        """
        if n_chunks and n_chunks > 1:
            size = len(self._df)
            step = size // n_chunks
            if size % n_chunks != 0:
                step += 1
            for start in range(0, step * n_chunks, step):
                yield PandasDataFrameXchg(
                    self._df.iloc[start : start + step, :],
                    allow_copy=self._allow_copy,
                )
        else:
            yield self
