# [MODULE DOC]
# HELPER: internal module for groupby on categorical keys. Provides the
# function recode_for_groupby which re-codes a Categorical so group order
# follows the requested sort/observed semantics. Used by the grouper module
# when building groupings. Depends on the unique helper from the algorithms
# module and on the categorical array, dtype and recoding helper. No
# import-time side effects.

from __future__ import annotations

import numpy as np

from pandas.core.algorithms import unique1d
from pandas.core.arrays.categorical import (
    Categorical,
    CategoricalDtype,
    recode_for_categories,
)


def recode_for_groupby(c: Categorical, sort: bool, observed: bool) -> Categorical:
    """
    Code the categories to ensure we can groupby for categoricals.

    If observed=True, we return a new Categorical with the observed
    categories only.

    If sort=False, return a copy of self, coded with categories as
    returned by .unique(), followed by any categories not appearing in
    the data. If sort=True, return self.

    This method is needed solely to ensure the categorical index of the
    GroupBy result has categories in the order of appearance in the data
    (GH-8868).

    Parameters
    ----------
    c : Categorical
    sort : bool
        The value of the sort parameter groupby was called with.
    observed : bool
        Account only for the observed values

    Returns
    -------
    Categorical
        If sort=False, the new categories are set to the order of
        appearance in codes (unless ordered=True, in which case the
        original order is preserved), followed by any unrepresented
        categories in the original order.
    """
    # [FUNCTION DOC]
    # HELPER: re-codes a categorical key for groupby so that the resulting
    # group order is right. Inputs: c, the categorical; sort, whether groupby
    # sorts keys; observed, whether only categories present in the data count.
    # Returns a categorical (possibly the input itself, never mutated).
    # Behavior: if observed, builds a new categorical whose categories are only
    # those that occur in the data (missing-value codes excluded), in sorted
    # order of the old codes when sort is true, otherwise in order of first
    # appearance, preserving the ordered flag. If not observed and sort is
    # true, returns the input unchanged. If not observed and sort is false,
    # returns a new categorical with categories in order of appearance followed
    # by all unused categories in original order. No exceptions raised here.
    # KNOWN ISSUE: the second check of sort inside the unobserved, unsorted
    # path is dead code, because the function already returned earlier when
    # sort was true.
    # we only care about observed values
    if observed:
        # In cases with c.ordered, this is equivalent to
        #  return c.remove_unused_categories(), c

        take_codes = unique1d(c.codes[c.codes != -1])

        if sort:
            take_codes = np.sort(take_codes)

        # we recode according to the uniques
        categories = c.categories.take(take_codes)
        codes = recode_for_categories(c.codes, c.categories, categories, copy=False)

        # return a new categorical that maps our new codes
        # and categories
        dtype = CategoricalDtype(categories, ordered=c.ordered)
        return Categorical._simple_new(codes, dtype=dtype)

    # Already sorted according to c.categories; all is fine
    if sort:
        return c

    # sort=False should order groups in as-encountered order (GH-8868)

    # GH:46909: Re-ordering codes faster than using (set|add|reorder)_categories
    # GH 38140: exclude nan from indexer for categories
    unique_notnan_codes = unique1d(c.codes[c.codes != -1])
    if sort:
        unique_notnan_codes = np.sort(unique_notnan_codes)
    if (num_cat := len(c.categories)) > len(unique_notnan_codes):
        # GH 13179: All categories need to be present, even if missing from the data
        missing_codes = np.setdiff1d(
            np.arange(num_cat), unique_notnan_codes, assume_unique=True
        )
        take_codes = np.concatenate((unique_notnan_codes, missing_codes))
    else:
        take_codes = unique_notnan_codes

    return Categorical(c, c.categories.take(take_codes))
