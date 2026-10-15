# [MODULE DOC]
# Frequency/resolution vocabulary for tslibs. Defines PeriodDtypeBase (an integer-coded
#   period dtype: a PeriodDtypeCode plus a multiple n), the Python enums FreqGroup,
#   Resolution and NpyDatetimeUnit mirroring the C enums in dtypes.pxd, and many module-
#   level lookup dicts mapping frequency strings (e.g. "Y-DEC", "QE", "h") to period
#   codes, offset aliases, deprecated aliases and resolution attribute names ("second",
#   "day", ...).
# Period codes are grouped in thousands: code // 1000 * 1000 is the frequency group
#   (FR_ANN=1000 ... FR_NS=12000); the remainder encodes a variant such as fiscal year-
#   end month or week-end day.
# Also provides conversions between numpy datetime units (NPY_DATETIMEUNIT) and their
#   string abbreviations, plus unit scale helpers (periods per day/second).
# Side effects at import: calls import_pandas_datetime() to load the datetime C-API
#   capsule.
# period frequency constants corresponding to scikits timeseries
# originals
from enum import Enum

from pandas._libs.tslibs.ccalendar cimport c_MONTH_NUMBERS
from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    get_conversion_factor,
    import_pandas_datetime,
)

import_pandas_datetime()

cdef class PeriodDtypeBase:
    # [CLASS DOC]
    # Lightweight immutable description of a period frequency. Fields (readonly):
    #   _dtype_code (PeriodDtypeCode int, e.g. 3000 for monthly, 2011 for Q-NOV) and _n
    #   (int64 multiple, e.g. 2 for "2M"). Equality and hashing are by (code, n).
    #   Subclassed by PeriodDtype.
    """
    Similar to an actual dtype, this contains all of the information
    describing a PeriodDtype in an integer code.
    """
    # cdef readonly:
    #    PeriodDtypeCode _dtype_code
    #    int64_t _n

    def __cinit__(self, PeriodDtypeCode code, int64_t n):
        # [FUNCTION DOC]
        # Construct: store code and n verbatim. Inputs: code - PeriodDtypeCode int; n -
        #   int64 multiple. No validation is performed. Outputs: none. Side effects:
        #   sets fields.
        self._dtype_code = code
        self._n = n

    def __eq__(self, other):
        # [FUNCTION DOC]
        # Equality: True only if other is a PeriodDtypeBase (or subclass) and both
        #   _dtype_code and _n match; otherwise False (never NotImplemented). The second
        #   isinstance check handles reversed-operand calls.
        if not isinstance(other, PeriodDtypeBase):
            return False
        if not isinstance(self, PeriodDtypeBase):
            # cython semantics, this is a reversed op
            return False
        return self._dtype_code == other._dtype_code and self._n == other._n

    def __hash__(self) -> int:
        # [FUNCTION DOC]
        # Hash of the tuple (_n, _dtype_code). Outputs: int.
        return hash((self._n, self._dtype_code))

    @property
    def _freq_group_code(self) -> int:
        # [FUNCTION DOC]
        # Property: the frequency group code, i.e. _dtype_code floored to a multiple of
        #   1000 (e.g. 2011 -> 2000). Outputs: int.
        # See also: libperiod.get_freq_group
        return (self._dtype_code // 1000) * 1000

    @property
    def _resolution_obj(self) -> "Resolution":
        # [FUNCTION DOC]
        # Property: map this dtype to a Resolution. Looks up the group code in
        #   FreqGroup, takes the abbreviation registered for that exact group code (e.g.
        #   "Y-DEC" for 1000, "W-SUN" for 4000), strips any "-suffix", and maps "B" to
        #   RESO_DAY; otherwise looks up the attribute name for the abbreviation and
        #   returns Resolution.from_attrname.
        # Exceptions: KeyError if the abbreviation has no resolution attribute name.
        # KNOWN ISSUE: for the weekly group the abbreviation is "W", which is not in the
        #   abbreviation-to-attrname map, so weekly dtypes raise KeyError here.
        fgc = self._freq_group_code
        freq_group = FreqGroup(fgc)
        abbrev = _period_code_to_abbrev[freq_group.value].split("-")[0]
        if abbrev == "B":
            return Resolution.RESO_DAY
        attrname = _abbrev_to_attrnames[abbrev]
        return Resolution.from_attrname(attrname)

    @property
    def _freqstr(self) -> str:
        # [FUNCTION DOC]
        # Property: frequency string. Looks up the abbreviation for the exact
        #   _dtype_code (e.g. "Q-NOV"); returns it as-is when n == 1, otherwise prefixes
        #   the decimal n (e.g. "2Q-NOV").
        # Exceptions: TypeError if the code is not in the map (lookup yields None and
        #   string concatenation fails when n != 1; returns None when n == 1).
        # Will be passed to to_offset in Period._maybe_convert_freq
        out = _period_code_to_abbrev.get(self._dtype_code)
        if self._n == 1:
            return out
        return str(self._n) + out

    cpdef int _get_to_timestamp_base(self):
        # [FUNCTION DOC]
        # Return the frequency group code used as the base unit when converting a period
        #   to a timestamp: FR_DAY for any code below FR_BUS (annual, quarterly,
        #   monthly, weekly), FR_SEC for codes from FR_HR through FR_SEC (hour, minute,
        #   second), and otherwise the raw code itself (business, day, ms, us, ns).
        #   Outputs: C int.
        """
        Return frequency code group used for base of to_timestamp against
        frequency code.

        Return day freq code against longer freq than day.
        Return second freq code against hour between second.

        Returns
        -------
        int
        """
        base = <c_FreqGroup>self._dtype_code
        if base < FR_BUS:
            return FR_DAY
        elif FR_HR <= base <= FR_SEC:
            return FR_SEC
        return base

    cpdef bint _is_tick_like(self):
        # [FUNCTION DOC]
        # True if the frequency is daily or finer (_dtype_code >= D == 6000), i.e. a
        #   fixed-length "tick" frequency; business-day (5000) and coarser return False.
        #   Outputs: bint.
        return self._dtype_code >= PeriodDtypeCode.D

    @property
    def _creso(self) -> int:
        # [FUNCTION DOC]
        # Property: numpy datetime unit matching this tick-like period code (D->day,
        #   H->hour, T->minute, S->second, L->ms, U->us, N->ns). Outputs: int
        #   NPY_DATETIMEUNIT value.
        # Exceptions: KeyError for non-tick codes (annual..business), and also for any
        #   code not exactly equal to one of those seven.
        return {
            PeriodDtypeCode.D: NPY_DATETIMEUNIT.NPY_FR_D,
            PeriodDtypeCode.H: NPY_DATETIMEUNIT.NPY_FR_h,
            PeriodDtypeCode.T: NPY_DATETIMEUNIT.NPY_FR_m,
            PeriodDtypeCode.S: NPY_DATETIMEUNIT.NPY_FR_s,
            PeriodDtypeCode.L: NPY_DATETIMEUNIT.NPY_FR_ms,
            PeriodDtypeCode.U: NPY_DATETIMEUNIT.NPY_FR_us,
            PeriodDtypeCode.N: NPY_DATETIMEUNIT.NPY_FR_ns,
        }[self._dtype_code]

    @property
    def _td64_unit(self) -> str:
        # [FUNCTION DOC]
        # Property: timedelta64 unit abbreviation for _creso (e.g. "ns"). Exceptions:
        #   KeyError as in _creso.
        return npy_unit_to_abbrev(self._creso)


_period_code_map = {
    # Annual freqs with various fiscal year ends.
    # eg, 2005 for Y-FEB runs Mar 1, 2004 to Feb 28, 2005
    "Y-DEC": PeriodDtypeCode.A_DEC,  # Annual - December year end
    "Y-JAN": PeriodDtypeCode.A_JAN,  # Annual - January year end
    "Y-FEB": PeriodDtypeCode.A_FEB,  # Annual - February year end
    "Y-MAR": PeriodDtypeCode.A_MAR,  # Annual - March year end
    "Y-APR": PeriodDtypeCode.A_APR,  # Annual - April year end
    "Y-MAY": PeriodDtypeCode.A_MAY,  # Annual - May year end
    "Y-JUN": PeriodDtypeCode.A_JUN,  # Annual - June year end
    "Y-JUL": PeriodDtypeCode.A_JUL,  # Annual - July year end
    "Y-AUG": PeriodDtypeCode.A_AUG,  # Annual - August year end
    "Y-SEP": PeriodDtypeCode.A_SEP,  # Annual - September year end
    "Y-OCT": PeriodDtypeCode.A_OCT,  # Annual - October year end
    "Y-NOV": PeriodDtypeCode.A_NOV,  # Annual - November year end

    # Quarterly frequencies with various fiscal year ends.
    # eg, Q42005 for Q-OCT runs Aug 1, 2005 to Oct 31, 2005
    "Q-DEC": PeriodDtypeCode.Q_DEC,    # Quarterly - December year end
    "Q-JAN": PeriodDtypeCode.Q_JAN,    # Quarterly - January year end
    "Q-FEB": PeriodDtypeCode.Q_FEB,    # Quarterly - February year end
    "Q-MAR": PeriodDtypeCode.Q_MAR,    # Quarterly - March year end
    "Q-APR": PeriodDtypeCode.Q_APR,    # Quarterly - April year end
    "Q-MAY": PeriodDtypeCode.Q_MAY,    # Quarterly - May year end
    "Q-JUN": PeriodDtypeCode.Q_JUN,    # Quarterly - June year end
    "Q-JUL": PeriodDtypeCode.Q_JUL,    # Quarterly - July year end
    "Q-AUG": PeriodDtypeCode.Q_AUG,    # Quarterly - August year end
    "Q-SEP": PeriodDtypeCode.Q_SEP,    # Quarterly - September year end
    "Q-OCT": PeriodDtypeCode.Q_OCT,    # Quarterly - October year end
    "Q-NOV": PeriodDtypeCode.Q_NOV,    # Quarterly - November year end

    "M": PeriodDtypeCode.M,        # Monthly

    "W-SUN": PeriodDtypeCode.W_SUN,    # Weekly - Sunday end of week
    "W-MON": PeriodDtypeCode.W_MON,    # Weekly - Monday end of week
    "W-TUE": PeriodDtypeCode.W_TUE,    # Weekly - Tuesday end of week
    "W-WED": PeriodDtypeCode.W_WED,    # Weekly - Wednesday end of week
    "W-THU": PeriodDtypeCode.W_THU,    # Weekly - Thursday end of week
    "W-FRI": PeriodDtypeCode.W_FRI,    # Weekly - Friday end of week
    "W-SAT": PeriodDtypeCode.W_SAT,    # Weekly - Saturday end of week

    "B": PeriodDtypeCode.B,        # Business days
    "D": PeriodDtypeCode.D,        # Daily
    "h": PeriodDtypeCode.H,        # Hourly
    "min": PeriodDtypeCode.T,      # Minutely
    "s": PeriodDtypeCode.S,        # Secondly
    "ms": PeriodDtypeCode.L,       # Millisecondly
    "us": PeriodDtypeCode.U,       # Microsecondly
    "ns": PeriodDtypeCode.N,       # Nanosecondly
}

cdef dict _period_code_to_abbrev = {
    _period_code_map[key]: key for key in _period_code_map}

cdef set _month_names = set(c_MONTH_NUMBERS.keys())

# Map attribute-name resolutions to resolution abbreviations
cdef dict attrname_to_abbrevs = {
    "year": "Y",
    "quarter": "Q",
    "month": "M",
    "day": "D",
    "hour": "h",
    "minute": "min",
    "second": "s",
    "millisecond": "ms",
    "microsecond": "us",
    "nanosecond": "ns",
}
cdef dict _abbrev_to_attrnames = {v: k for k, v in attrname_to_abbrevs.items()}

OFFSET_TO_PERIOD_FREQSTR: dict = {
    "WEEKDAY": "D",
    "EOM": "M",
    "BME": "M",
    "SME": "M",
    "BMS": "M",
    "CBME": "M",
    "CBMS": "M",
    "SMS": "M",
    "BQS": "Q",
    "QS": "Q",
    "BQE": "Q",
    "BQE-DEC": "Q-DEC",
    "BQE-JAN": "Q-JAN",
    "BQE-FEB": "Q-FEB",
    "BQE-MAR": "Q-MAR",
    "BQE-APR": "Q-APR",
    "BQE-MAY": "Q-MAY",
    "BQE-JUN": "Q-JUN",
    "BQE-JUL": "Q-JUL",
    "BQE-AUG": "Q-AUG",
    "BQE-SEP": "Q-SEP",
    "BQE-OCT": "Q-OCT",
    "BQE-NOV": "Q-NOV",
    "MS": "M",
    "D": "D",
    "B": "B",
    "min": "min",
    "s": "s",
    "ms": "ms",
    "us": "us",
    "ns": "ns",
    "h": "h",
    "QE": "Q",
    "QE-DEC": "Q-DEC",
    "QE-JAN": "Q-JAN",
    "QE-FEB": "Q-FEB",
    "QE-MAR": "Q-MAR",
    "QE-APR": "Q-APR",
    "QE-MAY": "Q-MAY",
    "QE-JUN": "Q-JUN",
    "QE-JUL": "Q-JUL",
    "QE-AUG": "Q-AUG",
    "QE-SEP": "Q-SEP",
    "QE-OCT": "Q-OCT",
    "QE-NOV": "Q-NOV",
    "YE": "Y",
    "YE-DEC": "Y-DEC",
    "YE-JAN": "Y-JAN",
    "YE-FEB": "Y-FEB",
    "YE-MAR": "Y-MAR",
    "YE-APR": "Y-APR",
    "YE-MAY": "Y-MAY",
    "YE-JUN": "Y-JUN",
    "YE-JUL": "Y-JUL",
    "YE-AUG": "Y-AUG",
    "YE-SEP": "Y-SEP",
    "YE-OCT": "Y-OCT",
    "YE-NOV": "Y-NOV",
    "W": "W",
    "ME": "M",
    "BYE": "Y",
    "BYE-DEC": "Y-DEC",
    "BYE-JAN": "Y-JAN",
    "BYE-FEB": "Y-FEB",
    "BYE-MAR": "Y-MAR",
    "BYE-APR": "Y-APR",
    "BYE-MAY": "Y-MAY",
    "BYE-JUN": "Y-JUN",
    "BYE-JUL": "Y-JUL",
    "BYE-AUG": "Y-AUG",
    "BYE-SEP": "Y-SEP",
    "BYE-OCT": "Y-OCT",
    "BYE-NOV": "Y-NOV",
    "YS": "Y",
    "BYS": "Y",
}
cdef dict c_OFFSET_RENAMED_FREQSTR = {
    "M": "ME",
    "Q": "QE",
    "Q-DEC": "QE-DEC",
    "Q-JAN": "QE-JAN",
    "Q-FEB": "QE-FEB",
    "Q-MAR": "QE-MAR",
    "Q-APR": "QE-APR",
    "Q-MAY": "QE-MAY",
    "Q-JUN": "QE-JUN",
    "Q-JUL": "QE-JUL",
    "Q-AUG": "QE-AUG",
    "Q-SEP": "QE-SEP",
    "Q-OCT": "QE-OCT",
    "Q-NOV": "QE-NOV",
    "Y": "YE",
    "Y-DEC": "YE-DEC",
    "Y-JAN": "YE-JAN",
    "Y-FEB": "YE-FEB",
    "Y-MAR": "YE-MAR",
    "Y-APR": "YE-APR",
    "Y-MAY": "YE-MAY",
    "Y-JUN": "YE-JUN",
    "Y-JUL": "YE-JUL",
    "Y-AUG": "YE-AUG",
    "Y-SEP": "YE-SEP",
    "Y-OCT": "YE-OCT",
    "Y-NOV": "YE-NOV",
    "BY": "BYE",
    "BY-DEC": "BYE-DEC",
    "BY-JAN": "BYE-JAN",
    "BY-FEB": "BYE-FEB",
    "BY-MAR": "BYE-MAR",
    "BY-APR": "BYE-APR",
    "BY-MAY": "BYE-MAY",
    "BY-JUN": "BYE-JUN",
    "BY-JUL": "BYE-JUL",
    "BY-AUG": "BYE-AUG",
    "BY-SEP": "BYE-SEP",
    "BY-OCT": "BYE-OCT",
    "BY-NOV": "BYE-NOV",
    "BM": "BME",
    "CBM": "CBME",
    "SM": "SME",
    "BQ": "BQE",
    "BQ-DEC": "BQE-DEC",
    "BQ-JAN": "BQE-JAN",
    "BQ-FEB": "BQE-FEB",
    "BQ-MAR": "BQE-MAR",
    "BQ-APR": "BQE-APR",
    "BQ-MAY": "BQE-MAY",
    "BQ-JUN": "BQE-JUN",
    "BQ-JUL": "BQE-JUL",
    "BQ-AUG": "BQE-AUG",
    "BQ-SEP": "BQE-SEP",
    "BQ-OCT": "BQE-OCT",
    "BQ-NOV": "BQE-NOV",
}
PERIOD_TO_OFFSET_FREQSTR = {
    "M": "ME",
    "Q": "QE",
    "Q-DEC": "QE-DEC",
    "Q-JAN": "QE-JAN",
    "Q-FEB": "QE-FEB",
    "Q-MAR": "QE-MAR",
    "Q-APR": "QE-APR",
    "Q-MAY": "QE-MAY",
    "Q-JUN": "QE-JUN",
    "Q-JUL": "QE-JUL",
    "Q-AUG": "QE-AUG",
    "Q-SEP": "QE-SEP",
    "Q-OCT": "QE-OCT",
    "Q-NOV": "QE-NOV",
    "Y": "YE",
    "Y-DEC": "YE-DEC",
    "Y-JAN": "YE-JAN",
    "Y-FEB": "YE-FEB",
    "Y-MAR": "YE-MAR",
    "Y-APR": "YE-APR",
    "Y-MAY": "YE-MAY",
    "Y-JUN": "YE-JUN",
    "Y-JUL": "YE-JUL",
    "Y-AUG": "YE-AUG",
    "Y-SEP": "YE-SEP",
    "Y-OCT": "YE-OCT",
    "Y-NOV": "YE-NOV",
}
cdef dict c_OFFSET_TO_PERIOD_FREQSTR = OFFSET_TO_PERIOD_FREQSTR
cdef dict c_PERIOD_TO_OFFSET_FREQSTR = PERIOD_TO_OFFSET_FREQSTR

cdef dict c_DEPR_UNITS = {
    "w": "W",
    "d": "D",
    "H": "h",
    "MIN": "min",
    "S": "s",
    "MS": "ms",
    "US": "us",
    "NS": "ns",
}

cdef dict c_PERIOD_AND_OFFSET_DEPR_FREQSTR = {
    "w": "W",
    "w-mon": "W-MON",
    "w-tue": "W-TUE",
    "w-wed": "W-WED",
    "w-thu": "W-THU",
    "w-fri": "W-FRI",
    "w-sat": "W-SAT",
    "w-sun": "W-SUN",
    "d": "D",
    "b": "B",
    "c": "C",
    "MIN": "min",
    "US": "us",
    "NS": "ns",
}

cdef str INVALID_FREQ_ERR_MSG = "Invalid frequency: {0}"


class FreqGroup(Enum):
    # [CLASS DOC]
    # Python Enum of frequency groups (multiples of 1000), mirroring c_FreqGroup in
    #   dtypes.pxd. FR_UND (-10000) means undefined.
    # Mirrors c_FreqGroup in the .pxd file
    FR_ANN = c_FreqGroup.FR_ANN
    FR_QTR = c_FreqGroup.FR_QTR
    FR_MTH = c_FreqGroup.FR_MTH
    FR_WK = c_FreqGroup.FR_WK
    FR_BUS = c_FreqGroup.FR_BUS
    FR_DAY = c_FreqGroup.FR_DAY
    FR_HR = c_FreqGroup.FR_HR
    FR_MIN = c_FreqGroup.FR_MIN
    FR_SEC = c_FreqGroup.FR_SEC
    FR_MS = c_FreqGroup.FR_MS
    FR_US = c_FreqGroup.FR_US
    FR_NS = c_FreqGroup.FR_NS
    FR_UND = c_FreqGroup.FR_UND  # undefined

    @staticmethod
    def from_period_dtype_code(code: int) -> "FreqGroup":
        # [FUNCTION DOC]
        # Static: floor an integer period code to its group (code // 1000 * 1000) and
        #   return the FreqGroup member. Inputs: code - int period dtype code.
        #   Exceptions: ValueError if the floored value is not a member.
        # See also: PeriodDtypeBase._freq_group_code
        code = (code // 1000) * 1000
        return FreqGroup(code)


class Resolution(Enum):
    # [CLASS DOC]
    # Python Enum of datetime resolutions, ordered finest (RESO_NS=0) to coarsest
    #   (RESO_YR=9), mirroring c_Resolution. Members compare by value; attrname gives
    #   the datetime attribute ("second"), attr_abbrev the offset alias ("s").
    RESO_NS = c_Resolution.RESO_NS
    RESO_US = c_Resolution.RESO_US
    RESO_MS = c_Resolution.RESO_MS
    RESO_SEC = c_Resolution.RESO_SEC
    RESO_MIN = c_Resolution.RESO_MIN
    RESO_HR = c_Resolution.RESO_HR
    RESO_DAY = c_Resolution.RESO_DAY
    RESO_MTH = c_Resolution.RESO_MTH
    RESO_QTR = c_Resolution.RESO_QTR
    RESO_YR = c_Resolution.RESO_YR

    def __lt__(self, other):
        # [FUNCTION DOC]
        # Less-than by integer value (finer resolution is "less"). Inputs: other -
        #   Resolution. Exceptions: AttributeError if other has no value.
        # KNOWN ISSUE: only lt and ge are defined; gt and le fall back to Enum defaults
        #   (reflected lt/ge works via Python's operator reflection, so a > b works, but
        #   comparison with non-Resolution raises AttributeError rather than returning
        #   NotImplemented).
        return self.value < other.value

    def __ge__(self, other):
        # [FUNCTION DOC]
        # Greater-or-equal by integer value. Inputs: other - Resolution.
        return self.value >= other.value

    @property
    def attr_abbrev(self) -> str:
        # [FUNCTION DOC]
        # Property: the frequency alias for this resolution (e.g. "h" for RESO_HR, "Y"
        #   for RESO_YR), via attrname then the attrname-to-abbreviation dict.
        # string that we can pass to to_offset
        return attrname_to_abbrevs[self.attrname]

    @property
    def attrname(self) -> str:
        # [FUNCTION DOC]
        # Property: datetime attribute name for this resolution, from the value-to-name
        #   map (e.g. 3 -> "second").
        """
        Return datetime attribute name corresponding to this Resolution.

        Examples
        --------
        >>> Resolution.RESO_SEC.attrname
        'second'
        """
        return _reso_str_map[self.value]

    @classmethod
    def from_attrname(cls, attrname: str) -> "Resolution":
        # [FUNCTION DOC]
        # Classmethod: inverse of attrname. Inputs: attrname - str like "second".
        #   Outputs: Resolution. Exceptions: KeyError if unknown.
        # Note: existing docstring says "Return resolution str against resolution code"
        #   which describes the opposite direction; the code returns a Resolution from a
        #   name.
        """
        Return resolution str against resolution code.

        Examples
        --------
        >>> Resolution.from_attrname('second')
        <Resolution.RESO_SEC: 3>

        >>> Resolution.from_attrname('second') == Resolution.RESO_SEC
        True
        """
        return cls(_str_reso_map[attrname])

    @classmethod
    def get_reso_from_freqstr(cls, freq: str) -> "Resolution":
        # [FUNCTION DOC]
        # Classmethod: resolution for a frequency string. Tries the abbreviation-to-
        #   attrname map directly ("h" -> hour). On miss, splits on "-"; if there are
        #   exactly two parts and the second is a month name (e.g. "Q-DEC"), looks up
        #   the first part; otherwise raises ValueError("Invalid frequency: <freq>").
        # Exceptions: ValueError as above; KeyError if the prefix before "-" is not a
        #   known abbreviation (e.g. "W-JAN" style with unknown prefix).
        """
        Return resolution code against frequency str.

        `freq` is given by the `offset.freqstr` for some DateOffset object.

        Examples
        --------
        >>> Resolution.get_reso_from_freqstr('h')
        <Resolution.RESO_HR: 5>

        >>> Resolution.get_reso_from_freqstr('h') == Resolution.RESO_HR
        True
        """
        try:
            attr_name = _abbrev_to_attrnames[freq]
        except KeyError as exc:
            msg = INVALID_FREQ_ERR_MSG.format(freq)
            # For quarterly and yearly resolutions, we need to chop off
            #  a month string.
            split_freq = freq.split("-")
            if len(split_freq) != 2:
                raise ValueError(msg) from exc
            if split_freq[1] not in _month_names:
                # i.e. we want e.g. "Q-DEC", not "Q-INVALID"
                raise ValueError(msg) from exc
            attr_name = _abbrev_to_attrnames[split_freq[0]]

        return cls.from_attrname(attr_name)


class NpyDatetimeUnit(Enum):
    # [CLASS DOC]
    # Python Enum mirroring the C NPY_DATETIMEUNIT values (Y, M, W, D, h, m, s, ms, us,
    #   ns, ps, fs, as, GENERIC).
    """
    Python-space analogue to NPY_DATETIMEUNIT.
    """
    NPY_FR_Y = NPY_DATETIMEUNIT.NPY_FR_Y
    NPY_FR_M = NPY_DATETIMEUNIT.NPY_FR_M
    NPY_FR_W = NPY_DATETIMEUNIT.NPY_FR_W
    NPY_FR_D = NPY_DATETIMEUNIT.NPY_FR_D
    NPY_FR_h = NPY_DATETIMEUNIT.NPY_FR_h
    NPY_FR_m = NPY_DATETIMEUNIT.NPY_FR_m
    NPY_FR_s = NPY_DATETIMEUNIT.NPY_FR_s
    NPY_FR_ms = NPY_DATETIMEUNIT.NPY_FR_ms
    NPY_FR_us = NPY_DATETIMEUNIT.NPY_FR_us
    NPY_FR_ns = NPY_DATETIMEUNIT.NPY_FR_ns
    NPY_FR_ps = NPY_DATETIMEUNIT.NPY_FR_ps
    NPY_FR_fs = NPY_DATETIMEUNIT.NPY_FR_fs
    NPY_FR_as = NPY_DATETIMEUNIT.NPY_FR_as
    NPY_FR_GENERIC = NPY_DATETIMEUNIT.NPY_FR_GENERIC


cdef NPY_DATETIMEUNIT get_supported_reso(NPY_DATETIMEUNIT reso):
    # [FUNCTION DOC]
    # Clamp a numpy datetime unit to the range pandas supports (seconds..nanoseconds).
    #   Inputs: reso - NPY_DATETIMEUNIT. Outputs: GENERIC -> ns; coarser than seconds ->
    #   s; finer than ns -> ns; else reso unchanged. Exceptions: none.
    # If we have an unsupported reso, return the nearest supported reso.
    if reso == NPY_DATETIMEUNIT.NPY_FR_GENERIC:
        # TODO: or raise ValueError? trying this gives unraisable errors, but
        #  "except? -1" breaks at compile-time for unknown reasons
        return NPY_DATETIMEUNIT.NPY_FR_ns
    if reso < NPY_DATETIMEUNIT.NPY_FR_s:
        return NPY_DATETIMEUNIT.NPY_FR_s
    elif reso > NPY_DATETIMEUNIT.NPY_FR_ns:
        return NPY_DATETIMEUNIT.NPY_FR_ns
    return reso


cdef bint is_supported_unit(NPY_DATETIMEUNIT reso):
    # [FUNCTION DOC]
    # True iff reso is one of the four supported units: s, ms, us, ns.
    return (
        reso == NPY_DATETIMEUNIT.NPY_FR_ns
        or reso == NPY_DATETIMEUNIT.NPY_FR_us
        or reso == NPY_DATETIMEUNIT.NPY_FR_ms
        or reso == NPY_DATETIMEUNIT.NPY_FR_s
    )


cdef str npy_unit_to_abbrev(NPY_DATETIMEUNIT unit):
    # [FUNCTION DOC]
    # Map a numpy datetime unit to its short string ("ns", "us", "ms", "s", "m", "h",
    #   "D", "W", "M", "Y", "ps", "fs", "as"); GENERIC maps to "ns". Exceptions:
    #   NotImplementedError(unit) for any other value.
    if unit == NPY_DATETIMEUNIT.NPY_FR_ns or unit == NPY_DATETIMEUNIT.NPY_FR_GENERIC:
        # generic -> default to nanoseconds
        return "ns"
    elif unit == NPY_DATETIMEUNIT.NPY_FR_us:
        return "us"
    elif unit == NPY_DATETIMEUNIT.NPY_FR_ms:
        return "ms"
    elif unit == NPY_DATETIMEUNIT.NPY_FR_s:
        return "s"
    elif unit == NPY_DATETIMEUNIT.NPY_FR_m:
        return "m"
    elif unit == NPY_DATETIMEUNIT.NPY_FR_h:
        return "h"
    elif unit == NPY_DATETIMEUNIT.NPY_FR_D:
        return "D"
    elif unit == NPY_DATETIMEUNIT.NPY_FR_W:
        return "W"
    elif unit == NPY_DATETIMEUNIT.NPY_FR_M:
        return "M"
    elif unit == NPY_DATETIMEUNIT.NPY_FR_Y:
        return "Y"

    # Checks for not-really-supported units go at the end, as we don't expect
    #  to see these often
    elif unit == NPY_DATETIMEUNIT.NPY_FR_ps:
        return "ps"
    elif unit == NPY_DATETIMEUNIT.NPY_FR_fs:
        return "fs"
    elif unit == NPY_DATETIMEUNIT.NPY_FR_as:
        return "as"

    else:
        raise NotImplementedError(unit)


cpdef NPY_DATETIMEUNIT abbrev_to_npy_unit(str abbrev):
    # [FUNCTION DOC]
    # Inverse of npy_unit_to_abbrev: map a unit string to NPY_DATETIMEUNIT. Accepts
    #   "Y","M","W","D"/"d","h","m","s","ms","us","ns","ps","fs","as"; None maps to
    #   GENERIC. Exceptions: ValueError("Unrecognized unit ...") otherwise.
    if abbrev == "Y":
        return NPY_DATETIMEUNIT.NPY_FR_Y
    elif abbrev == "M":
        return NPY_DATETIMEUNIT.NPY_FR_M
    elif abbrev == "W":
        return NPY_DATETIMEUNIT.NPY_FR_W
    elif abbrev == "D" or abbrev == "d":
        return NPY_DATETIMEUNIT.NPY_FR_D
    elif abbrev == "h":
        return NPY_DATETIMEUNIT.NPY_FR_h
    elif abbrev == "m":
        return NPY_DATETIMEUNIT.NPY_FR_m
    elif abbrev == "s":
        return NPY_DATETIMEUNIT.NPY_FR_s
    elif abbrev == "ms":
        return NPY_DATETIMEUNIT.NPY_FR_ms
    elif abbrev == "us":
        return NPY_DATETIMEUNIT.NPY_FR_us
    elif abbrev == "ns":
        return NPY_DATETIMEUNIT.NPY_FR_ns
    elif abbrev == "ps":
        return NPY_DATETIMEUNIT.NPY_FR_ps
    elif abbrev == "fs":
        return NPY_DATETIMEUNIT.NPY_FR_fs
    elif abbrev == "as":
        return NPY_DATETIMEUNIT.NPY_FR_as
    elif abbrev is None:
        return NPY_DATETIMEUNIT.NPY_FR_GENERIC
    else:
        raise ValueError(f"Unrecognized unit {abbrev}")


cdef NPY_DATETIMEUNIT freq_group_code_to_npy_unit(int freq) noexcept nogil:
    # [FUNCTION DOC]
    # Map a frequency group code (FR_MTH, FR_DAY, FR_HR ... FR_NS, FR_UND) to the numpy
    #   unit used when building datetimes from struct fields; FR_UND maps to day.
    #   Inputs: freq - group code int. Side effects: none; runs without the GIL.
    # KNOWN ISSUE: FR_ANN, FR_QTR, FR_WK, FR_BUS and any other value have no branch, so
    #   the function falls off the end and returns 0, which is NPY_FR_Y (silently
    #   "year"), with no error.
    """
    Convert the freq to the corresponding NPY_DATETIMEUNIT to pass
    to npy_datetimestruct_to_datetime.
    """
    if freq == FR_MTH:
        return NPY_DATETIMEUNIT.NPY_FR_M
    elif freq == FR_DAY:
        return NPY_DATETIMEUNIT.NPY_FR_D
    elif freq == FR_HR:
        return NPY_DATETIMEUNIT.NPY_FR_h
    elif freq == FR_MIN:
        return NPY_DATETIMEUNIT.NPY_FR_m
    elif freq == FR_SEC:
        return NPY_DATETIMEUNIT.NPY_FR_s
    elif freq == FR_MS:
        return NPY_DATETIMEUNIT.NPY_FR_ms
    elif freq == FR_US:
        return NPY_DATETIMEUNIT.NPY_FR_us
    elif freq == FR_NS:
        return NPY_DATETIMEUNIT.NPY_FR_ns
    elif freq == FR_UND:
        # Default to Day
        return NPY_DATETIMEUNIT.NPY_FR_D


cpdef int64_t periods_per_day(
    NPY_DATETIMEUNIT reso=NPY_DATETIMEUNIT.NPY_FR_ns
) except? -1:
    # [FUNCTION DOC]
    # Number of reso units in one day (e.g. 86400 * 10**9 for ns). Inputs: reso -
    #   NPY_DATETIMEUNIT, default ns. Outputs: int64. Exceptions: propagates errors from
    #   get_conversion_factor (e.g. for units coarser than a day or GENERIC).
    """
    How many of the given time units fit into a single day?
    """
    return get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_D, reso)


cpdef int64_t periods_per_second(NPY_DATETIMEUNIT reso) except? -1:
    # [FUNCTION DOC]
    # Number of reso units in one second (e.g. 1000 for ms). Inputs: reso -
    #   NPY_DATETIMEUNIT. Outputs: int64. Exceptions: as get_conversion_factor.
    return get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_s, reso)


cdef dict _reso_str_map = {
    Resolution.RESO_NS.value: "nanosecond",
    Resolution.RESO_US.value: "microsecond",
    Resolution.RESO_MS.value: "millisecond",
    Resolution.RESO_SEC.value: "second",
    Resolution.RESO_MIN.value: "minute",
    Resolution.RESO_HR.value: "hour",
    Resolution.RESO_DAY.value: "day",
    Resolution.RESO_MTH.value: "month",
    Resolution.RESO_QTR.value: "quarter",
    Resolution.RESO_YR.value: "year",
}

cdef dict _str_reso_map = {v: k for k, v in _reso_str_map.items()}

cdef dict npy_unit_to_attrname = {
    NPY_DATETIMEUNIT.NPY_FR_Y: "year",
    NPY_DATETIMEUNIT.NPY_FR_M: "month",
    NPY_DATETIMEUNIT.NPY_FR_D: "day",
    NPY_DATETIMEUNIT.NPY_FR_h: "hour",
    NPY_DATETIMEUNIT.NPY_FR_m: "minute",
    NPY_DATETIMEUNIT.NPY_FR_s: "second",
    NPY_DATETIMEUNIT.NPY_FR_ms: "millisecond",
    NPY_DATETIMEUNIT.NPY_FR_us: "microsecond",
    NPY_DATETIMEUNIT.NPY_FR_ns: "nanosecond",
}
cdef dict attrname_to_npy_unit = {v: k for k, v in npy_unit_to_attrname.items()}
