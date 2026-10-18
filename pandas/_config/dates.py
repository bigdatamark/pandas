"""
config for datetime formatting
"""

# [MODULE DOC]
# Registers the two date-parsing display options, day-first and year-first. They
# live in the config package (not in the core config init module) because the
# low-level date parsing code needs them and cannot depend on the rest of
# pandas. Public contents: the two description strings (pc_date_dayfirst_doc and
# pc_date_yearfirst_doc), which are used as the option documentation.
# Import-time side effects: registers display.date_dayfirst and
# display.date_yearfirst, both with default false and a boolean-type validator,
# in the global option registry. Raises an option error if imported twice under
# different module objects (duplicate registration).
from __future__ import annotations

from pandas._config import config as cf

pc_date_dayfirst_doc = """
: boolean
    When True, prints and parses dates with the day first, eg 20/01/2005
"""

pc_date_yearfirst_doc = """
: boolean
    When True, prints and parses dates with the year first, eg 2005/01/20
"""

with cf.config_prefix("display"):
    # Needed upstream of `_libs` because these are used in tslibs.parsing
    cf.register_option(
        "date_dayfirst", False, pc_date_dayfirst_doc, validator=cf.is_bool
    )
    cf.register_option(
        "date_yearfirst", False, pc_date_yearfirst_doc, validator=cf.is_bool
    )
