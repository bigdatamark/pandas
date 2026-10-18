# [MODULE DOC]
# Reader for SPSS .sav files, a thin wrapper around the optional pyreadstat
# dependency.
# Public contents: the read_spss function (exposed as part of the top-level
# pandas namespace).
# Relationships: uses path stringification from the io common module, dtype
# backend validation, and the data frame conversion of nullable dtypes.
# Import-time side effects: none; pyreadstat is imported only when reading.
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import set_module
from pandas.util._validators import check_dtype_backend

from pandas.core.dtypes.inference import is_list_like

from pandas.io.common import stringify_path

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from pandas._typing import DtypeBackend

    from pandas import DataFrame


@set_module("pandas")
def read_spss(
    path: str | Path,
    usecols: Sequence[str] | None = None,
    convert_categoricals: bool = True,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    **kwargs: Any,
) -> DataFrame:
    # [FUNCTION DOC]
    # Reads an SPSS file and returns a data frame.
    # Parameters: path is a string or path object (converted to a string);
    # usecols is an optional list-like of column names to load (default none
    # meaning all columns); convert_categoricals (default true) asks pyreadstat
    # to apply value labels so labelled columns become categorical-like;
    # dtype_backend selects nullable or pyarrow dtypes (default: no conversion);
    # extra keyword arguments are forwarded to the pyreadstat sav reader.
    # Returns: a data frame whose attrs mapping is replaced by the attribute
    # dictionary of the pyreadstat metadata object. When a dtype backend is
    # given, the frame is converted using that backend.
    # Raises: an import error if pyreadstat is missing; a type error if usecols
    # is not list-like; validation errors for an invalid dtype_backend.
    """
    Load an SPSS file from the file path, returning a DataFrame.

    Parameters
    ----------
    path : str or Path
        File path.
    usecols : list-like, optional
        Return a subset of the columns. If None, return all columns.
    convert_categoricals : bool, default is True
        Convert categorical columns into pd.Categorical.
    dtype_backend : {'numpy_nullable', 'pyarrow'}
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). If not specified, the default behavior
        is to not use nullable data types. If specified, the behavior
        is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
        * ``"pyarrow"``: returns pyarrow-backed
          nullable :class:`ArrowDtype` :class:`DataFrame`

        .. versionadded:: 2.0
    **kwargs
        Additional keyword arguments that can be passed to :func:`pyreadstat.read_sav`.

        .. versionadded:: 3.0

    Returns
    -------
    DataFrame
        DataFrame based on the SPSS file.

    See Also
    --------
    read_csv : Read a comma-separated values (csv) file into a pandas DataFrame.
    read_excel : Read an Excel file into a pandas DataFrame.
    read_sas : Read an SAS file into a pandas DataFrame.
    read_orc : Load an ORC object into a pandas DataFrame.
    read_feather : Load a feather-format object into a pandas DataFrame.

    Examples
    --------
    >>> df = pd.read_spss("spss_data.sav")  # doctest: +SKIP
    """
    pyreadstat = import_optional_dependency("pyreadstat")
    check_dtype_backend(dtype_backend)

    if usecols is not None:
        if not is_list_like(usecols):
            raise TypeError("usecols must be list-like.")
        usecols = list(usecols)  # pyreadstat requires a list

    df, metadata = pyreadstat.read_sav(
        stringify_path(path),
        usecols=usecols,
        apply_value_formats=convert_categoricals,
        **kwargs,
    )
    df.attrs = metadata.__dict__
    if dtype_backend is not lib.no_default:
        df = df.convert_dtypes(dtype_backend=dtype_backend)
    return df
