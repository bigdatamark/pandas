# [MODULE DOC]
# Public re-export module for the computation package. It exposes only the
# top-level expression evaluation function, defined in the eval module, as
# the single public name. Import-time side effect: importing it loads the
# eval module (and transitively the parser, engines and scope modules).
__all__ = ["eval"]
from pandas.core.computation.eval import eval
