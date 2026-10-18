# [MODULE DOC]
# Timezone (tzinfo) classification, normalization and DST-transition extraction for
#   tslibs. Supports four tzinfo families: stdlib datetime.timezone (fixed offsets,
#   including timezone.utc), zoneinfo.ZoneInfo, dateutil (tzutc, tzlocal, tzfile), and
#   optional pytz (imported lazily; may be absent). Families are detected by isinstance
#   checks or duck-typing on private attributes (pytz:
#   _utc_transition_times/_transition_info; dateutil tzfile: _trans_list/_trans_idx).
# Key services: is_utc (recognizes every UTC singleton), get_timezone / maybe_get_tz
#   (round-trip a tzinfo to/from a serializable string key such as 'dateutil/<file>' or
#   a zone name), tz_compare (zone equality that ignores pytz's per-offset instances),
#   is_fixed_offset, and get_dst_info which returns (trans, deltas, typ, has_tz_rule): a
#   sorted int64 array of UTC transition instants in nanoseconds (first entry is the
#   sentinel NaT+1 meaning 'since the beginning of time'), a same-length int64 array of
#   UTC offsets in nanoseconds valid from each transition, a type string ('pytz',
#   'dateutil', 'zoneinfo', 'fixed', 'static', 'unknown'), and whether a recurring rule
#   applies after the last transition. Results are memoized in the module-level dict
#   dst_cache keyed by tz_cache_key. Consumed mainly by tzconversion, conversion and
#   vectorized.
# Module globals: NPY_NAT (int64 min, the NaT marker), utc_stdlib, utc_pytz (None if
#   pytz missing), utc_dateutil_str (dateutil gettz('UTC'), distinct from tzutc()),
#   utc_zoneinfo (lazily created ZoneInfo('UTC'), may stay None when tzdata is missing).

from datetime import (
    timedelta,
    timezone,
)
import zoneinfo
from zoneinfo._zoneinfo import ZoneInfo as _ZoneInfo

from pandas.compat._optional import import_optional_dependency

from cpython.datetime cimport (
    datetime,
    timedelta,
    tzinfo,
)

# dateutil compat

from dateutil.tz import (
    gettz as dateutil_gettz,
    tzfile as _dateutil_tzfile,
    tzlocal as _dateutil_tzlocal,
    tzutc as _dateutil_tzutc,
)
import numpy as np

pytz = import_optional_dependency("pytz", errors="ignore")

cimport numpy as cnp
from numpy cimport int64_t

cnp.import_array()

# ----------------------------------------------------------------------
from pandas._libs.tslibs.util cimport (
    get_nat,
    is_integer_object,
)


cdef int64_t NPY_NAT = get_nat()
cdef tzinfo utc_stdlib = timezone.utc
cdef tzinfo utc_pytz = pytz.UTC if pytz else None
cdef tzinfo utc_dateutil_str = dateutil_gettz("UTC")  # NB: *not* the same as tzutc()

cdef tzinfo utc_zoneinfo = None
cdef type ZoneInfo = zoneinfo.ZoneInfo


# ----------------------------------------------------------------------

cdef bint is_utc_zoneinfo(tzinfo tz):
    # [FUNCTION DOC]
    # HELPER: Return True iff tz is the cached ZoneInfo('UTC') singleton.
    # Inputs: tz - tzinfo or None.
    # Outputs: bool; False for None or when ZoneInfo('UTC') cannot be built (missing
    #   tzdata, ZoneInfoNotFoundError swallowed).
    # Exceptions: none.
    # Side effects: on first successful call creates and stores ZoneInfo('UTC') in the
    #   module global utc_zoneinfo. Identity comparison only.
    # Workaround for cases with missing tzdata
    #  https://github.com/pandas-dev/pandas/pull/46425#discussion_r830633025
    if tz is None:
        return False

    global utc_zoneinfo
    if utc_zoneinfo is None:
        try:
            utc_zoneinfo = zoneinfo.ZoneInfo("UTC")
        except zoneinfo.ZoneInfoNotFoundError:
            return False

    return tz is utc_zoneinfo


cpdef inline bint is_utc(tzinfo tz):
    # [FUNCTION DOC]
    # Return True iff tz represents UTC in any supported library: identical to stdlib
    #   timezone.utc, an instance of dateutil tzutc, identical to dateutil gettz('UTC'),
    #   the ZoneInfo('UTC') singleton, or identical to pytz.UTC (if pytz is installed).
    # Inputs: tz - tzinfo or None.
    # Outputs: bool (False for None and for non-UTC fixed offsets such as
    #   timezone(timedelta(0)) built separately).
    # Exceptions: none. Side effects: may initialize utc_zoneinfo.
    return (
        tz is utc_stdlib
        or isinstance(tz, _dateutil_tzutc)
        or tz is utc_dateutil_str
        or is_utc_zoneinfo(tz)
        or (utc_pytz is not None and tz is utc_pytz)
    )


cdef bint is_zoneinfo(tzinfo tz):
    # [FUNCTION DOC]
    # Return True iff tz is an instance of zoneinfo.ZoneInfo.
    # Inputs: tz - tzinfo or None. Outputs: bool. Exceptions/side effects: none.
    return isinstance(tz, ZoneInfo)


cdef bint is_tzlocal(tzinfo tz):
    # [FUNCTION DOC]
    # Return True iff tz is an instance of dateutil tzlocal (the system local zone).
    # Inputs: tz - tzinfo or None. Outputs: bool. Exceptions/side effects: none.
    return isinstance(tz, _dateutil_tzlocal)


cdef bint treat_tz_as_pytz(tzinfo tz):
    # [FUNCTION DOC]
    # Duck-typing check for a pytz DST-aware zone: True iff tz has both
    #   _utc_transition_times and _transition_info attributes. pytz static zones
    #   (StaticTzInfo, UTC) do not qualify.
    # Inputs: tz - tzinfo. Outputs: bool. Exceptions/side effects: none.
    return (hasattr(tz, "_utc_transition_times") and
            hasattr(tz, "_transition_info"))


cdef bint treat_tz_as_dateutil(tzinfo tz):
    # [FUNCTION DOC]
    # HELPER: Duck-typing check for a dateutil tzfile-like zone: True iff tz has both
    #   _trans_list and _trans_idx attributes (tzlocal/tzutc do not).
    # Inputs: tz - tzinfo. Outputs: bool. Exceptions/side effects: none.
    return hasattr(tz, "_trans_list") and hasattr(tz, "_trans_idx")


# Returns str or tzinfo object
cpdef inline object get_timezone(tzinfo tz):
    # [FUNCTION DOC]
    # Map a tzinfo to a serializable, zone-identifying value (inverse of maybe_get_tz).
    #   Order of checks: UTC (any flavor) -> returned unchanged as the tzinfo object;
    #   ZoneInfo -> its key string; pytz DST zone -> its .zone string, or the tz itself
    #   if .zone is None; dateutil tzfile -> 'dateutil/' + _filename; anything else ->
    #   tz unchanged.
    # Inputs: tz - tzinfo (must not be None).
    # Outputs: str or the original tzinfo object.
    # Exceptions: TypeError if tz is None; ValueError if a dateutil filename contains
    #   '.tar.gz' (Windows dateutil bug where all zones share a filename).
    # Side effects: none.
    """
    We need to do several things here:
    1) Distinguish between pytz and dateutil timezones
    2) Not be over-specific (e.g. US/Eastern with/without DST is same *zone*
       but a different tz object)
    3) Provide something to serialize when we're storing a datetime object
       in pytables.

    We return a string prefaced with dateutil if it's a dateutil tz, else just
    the tz name. It needs to be a string so that we can serialize it with
    UJSON/pytables. maybe_get_tz (below) is the inverse of this process.
    """
    if tz is None:
        raise TypeError("tz argument cannot be None")
    if is_utc(tz):
        return tz
    elif is_zoneinfo(tz):
        return tz.key
    elif treat_tz_as_pytz(tz):
        zone = tz.zone
        if zone is None:
            return tz
        return zone
    elif treat_tz_as_dateutil(tz):
        if ".tar.gz" in tz._filename:
            raise ValueError(
                "Bad tz filename. Dateutil on python 3 on windows has a "
                "bug which causes tzfile._filename to be the same for all "
                "timezone files. Please construct dateutil timezones "
                'implicitly by passing a string like "dateutil/Europe'
                '/London" when you construct your pandas objects instead '
                "of passing a timezone object. See "
                "https://github.com/pandas-dev/pandas/pull/7362")
        return "dateutil/" + tz._filename
    else:
        return tz


cpdef inline tzinfo maybe_get_tz(object tz):
    # [FUNCTION DOC]
    # Build a tzinfo from a user-supplied spec; tzinfo/None pass through. String rules,
    #   in order: 'tzlocal()' -> dateutil tzlocal(); 'dateutil/<name>' -> dateutil
    #   gettz(name), and if the result is a tzfile whose _filename contains '.tar.gz'
    #   its _filename is overwritten with name; leading '+' or '-' -> fixed offset
    #   parsed as sign+HH at chars 0-2 and MM at chars 4-5 (format '+HH:MM');
    #   'UTC+HH:MM'/'UTC-HH:MM' -> same parse shifted by 3 chars; 'UTC' or 'utc' ->
    #   stdlib timezone.utc; any other string -> zoneinfo.ZoneInfo(str). Integer ->
    #   fixed offset of that many seconds.
    # Inputs: tz - str, int, tzinfo or None.
    # Outputs: tzinfo or None.
    # Exceptions: TypeError(type(tz)) for unsupported types; ImportError if tz is a pytz
    #   zone object but pytz cannot be imported (re-raised via
    #   import_optional_dependency); ValueError from int() on malformed offsets;
    #   ZoneInfoNotFoundError for unknown names; ValueError from timezone() if the
    #   offset is out of range.
    # Side effects: may mutate the _filename of a dateutil tzfile object.
    """
    (Maybe) Construct a timezone object from a string. If tz is a string, use
    it to construct a timezone object. Otherwise, just return tz.
    """
    if isinstance(tz, str):
        if tz == "tzlocal()":
            tz = _dateutil_tzlocal()
        elif tz.startswith("dateutil/"):
            zone = tz[9:]
            tz = dateutil_gettz(zone)
            # On Python 3 on Windows, the filename is not always set correctly.
            if isinstance(tz, _dateutil_tzfile) and ".tar.gz" in tz._filename:
                tz._filename = zone
        elif tz[0] in {"-", "+"}:
            hours = int(tz[0:3])
            minutes = int(tz[0] + tz[4:6])
            tz = timezone(timedelta(hours=hours, minutes=minutes))
        elif tz[0:4] in {"UTC-", "UTC+"}:
            hours = int(tz[3:6])
            minutes = int(tz[3] + tz[7:9])
            tz = timezone(timedelta(hours=hours, minutes=minutes))
        elif tz == "UTC" or tz == "utc":
            tz = utc_stdlib
        else:
            tz = zoneinfo.ZoneInfo(tz)
    elif is_integer_object(tz):
        tz = timezone(timedelta(seconds=tz))
    elif isinstance(tz, tzinfo):
        if treat_tz_as_pytz(tz) and pytz is None:
            # call again for raising proper error
            import_optional_dependency("pytz")
        pass
    elif tz is None:
        pass
    else:
        raise TypeError(type(tz))
    return tz


def _p_tz_cache_key(tz: tzinfo):
    # [FUNCTION DOC]
    # HELPER: Python-visible wrapper around tz_cache_key, for tests.
    # Inputs: tz - tzinfo. Outputs/Exceptions: as tz_cache_key.
    """
    Python interface for cache function to facilitate testing.
    """
    return tz_cache_key(tz)


# Timezone data caches, key is the pytz string or dateutil file name.
dst_cache = {}


cdef object tz_cache_key(tzinfo tz):
    # [FUNCTION DOC]
    # HELPER: Compute the dst_cache key for tz. pytz BaseTzInfo (pytz installed) ->
    #   tz.zone; dateutil tzfile -> 'dateutil' + _filename (no separator; _filename is
    #   normally an absolute path); ZoneInfo -> 'zoneinfo/' + key; anything else (fixed
    #   offsets, tzlocal, stdlib timezone) -> None meaning 'not cacheable'.
    # Inputs: tz - tzinfo. Outputs: str or None.
    # Exceptions: ValueError for a dateutil tzfile whose _filename contains '.tar.gz'.
    #   Side effects: none.
    """
    Return the key in the cache for the timezone info object or None
    if unknown.

    The key is currently the tz string for pytz timezones, the filename for
    dateutil timezones.

    Notes
    -----
    This cannot just be the hash of a timezone object. Unfortunately, the
    hashes of two dateutil tz objects which represent the same timezone are
    not equal (even though the tz objects will compare equal and represent
    the same tz file). Also, pytz objects are not always hashable so we use
    str(tz) instead.
    """
    if pytz is not None and isinstance(tz, pytz.tzinfo.BaseTzInfo):
        return tz.zone
    elif isinstance(tz, _dateutil_tzfile):
        if ".tar.gz" in tz._filename:
            raise ValueError("Bad tz filename. Dateutil on python 3 on "
                             "windows has a bug which causes tzfile._filename "
                             "to be the same for all timezone files. Please "
                             "construct dateutil timezones implicitly by "
                             'passing a string like "dateutil/Europe/London" '
                             "when you construct your pandas objects instead "
                             "of passing a timezone object. See "
                             "https://github.com/pandas-dev/pandas/pull/7362")
        return "dateutil" + tz._filename
    elif is_zoneinfo(tz):
        return "zoneinfo/" + tz.key
    else:
        return None


# ----------------------------------------------------------------------
# UTC Offsets


cdef timedelta get_utcoffset(tzinfo tz, datetime obj):
    # [FUNCTION DOC]
    # Return tz's UTC offset for obj, using the cached _utcoffset attribute when present
    #   (pytz fast path), else tz.utcoffset(obj).
    # Inputs: tz - tzinfo; obj - datetime or None.
    # Outputs: timedelta, or None if the tzinfo returns None (e.g. ZoneInfo with
    #   obj=None).
    # Exceptions: whatever tz.utcoffset raises. Side effects: none.
    try:
        return tz._utcoffset
    except AttributeError:
        return tz.utcoffset(obj)


cpdef inline bint is_fixed_offset(tzinfo tz):
    # [FUNCTION DOC]
    # Return True if tz has no DST transitions. dateutil tzfile: True iff both
    #   _trans_idx and _trans_list are empty. pytz DST zone (pytz installed): True iff
    #   both _transition_info and _utc_transition_times are empty. ZoneInfo: builds the
    #   pure-Python zoneinfo implementation for tz.key and returns its _fixed_offset
    #   flag. Everything else (stdlib timezone, pytz static zones, tzutc, tzlocal,
    #   unknown tzinfo) returns True.
    # Inputs: tz - tzinfo. Outputs: bool (1/0).
    # Exceptions: errors from constructing the Python ZoneInfo (e.g. missing data). Side
    #   effects: allocates a Python ZoneInfo object per ZoneInfo call.
    # KNOWN ISSUE: the fallthrough returns True for dateutil tzlocal(), which can
    #   observe DST; callers must check is_tzlocal first.
    if treat_tz_as_dateutil(tz):
        if len(tz._trans_idx) == 0 and len(tz._trans_list) == 0:
            return 1
        else:
            return 0
    elif treat_tz_as_pytz(tz) and pytz is not None:
        if (len(tz._transition_info) == 0
                and len(tz._utc_transition_times) == 0):
            return 1
        else:
            return 0
    elif is_zoneinfo(tz):
        tz_py = _ZoneInfo(tz.key)
        if tz_py._fixed_offset:
            return 1
        else:
            return 0
    # This also implicitly accepts datetime.timezone objects which are
    # considered fixed
    return 1


cdef object _get_utc_trans_times_from_dateutil_tz(tzinfo tz):
    # [FUNCTION DOC]
    # HELPER: Convert a dateutil tzfile's transition list (stored in local standard
    #   time, seconds) to UTC seconds. Walks _trans_list and _trans_idx in parallel;
    #   tracks the most recent non-DST ttinfo offset (initially 0) and subtracts it from
    #   each transition.
    # Inputs: tz - dateutil tzfile. Outputs: new list of int seconds, same length as
    #   _trans_list.
    # Exceptions: ValueError if the two lists differ in length (strict zip). Side
    #   effects: none (input not mutated).
    """
    Transition times in dateutil timezones are stored in local non-dst
    time.  This code converts them to UTC. It's the reverse of the code
    in dateutil.tz.tzfile.__init__.
    """
    new_trans = list(tz._trans_list)
    last_std_offset = 0
    for i, (trans, tti) in enumerate(zip(tz._trans_list, tz._trans_idx, strict=True)):
        if not tti.isdst:
            last_std_offset = tti.offset
        new_trans[i] = trans - last_std_offset
    return new_trans


cdef int64_t[::1] unbox_utcoffsets(object transinfo):
    # [FUNCTION DOC]
    # HELPER: Convert pytz _transition_info (sequence of (utcoffset timedelta, dst,
    #   tzname) tuples) to a contiguous int64 array of the utcoffsets in nanoseconds
    #   (whole seconds * 1e9; sub-second parts truncated).
    # Inputs: transinfo - sequence of tuples whose element 0 is a timedelta.
    # Outputs: 1-D C-contiguous int64 memoryview of length len(transinfo).
    # Exceptions: none expected. Side effects: allocates a new numpy array.
    cdef:
        Py_ssize_t i
        cnp.npy_intp sz
        int64_t[::1] arr

    sz = len(transinfo)
    arr = cnp.PyArray_EMPTY(1, &sz, cnp.NPY_INT64, 0)

    for i in range(sz):
        arr[i] = int(transinfo[i][0].total_seconds()) * 1_000_000_000

    return arr


# ----------------------------------------------------------------------
# Daylight Savings

cdef datetime _UTC_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
# Smallest whole second representable as int64 nanoseconds.
cdef int64_t _MIN_TRANS_SECONDS = -(-(NPY_NAT + 1) // 1_000_000_000)


cdef tuple _get_zoneinfo_trans_and_deltas(tzinfo tz):
    # [FUNCTION DOC]
    # HELPER: Extract transitions and offsets for a ZoneInfo zone using the pure-Python
    #   zoneinfo implementation (rebuilt from tz.key).
    # Fixed-offset zones: return trans=[NaT+1], deltas=[offset ns], True, False.
    # Otherwise start with historical _trans_utc (UTC seconds) and per-transition
    #   utcoffsets from _ttinfos. If a POSIX rule (_tz_after with a transitions method)
    #   exists, generate future transitions: take std and dst offsets, find the calendar
    #   year of the last historical transition (computed via epoch + timedelta to avoid
    #   32-bit time_t overflow; 1970 on error or if no history), then for each year from
    #   that year to 2099 ask the rule for (dst start, dst end) in local
    #   standard/daylight seconds, convert to UTC by subtracting std_offset and
    #   dst_offset respectively, keep only those after the last historical transition,
    #   stop early on an empty result or any exception, sort them and append. Then drop
    #   a leading 'big bang' transition below the smallest int64-ns-representable
    #   second. Finally scale to ns and prepend NaT+1 to trans and the _tti_before
    #   offset to deltas.
    # Inputs: tz - ZoneInfo.
    # Outputs: tuple (trans int64 ndarray ns, deltas int64 ndarray ns, is_fixed bool,
    #   has_tz_rule bool); trans and deltas have equal length.
    # Exceptions: from building the Python ZoneInfo. Side effects: none.
    # KNOWN ISSUE: the docstring names the last output 'has_tz_tule' (typo). Transitions
    #   after 2099 are not generated, so the rule flag is what callers must rely on for
    #   later dates.
    """
    Get transition times and UTC offsets for a ZoneInfo timezone.

    Uses zoneinfo's Python fallback implementation to get transition data,
    including future transitions generated from POSIX TZ rules.

    Parameters
    ----------
    tz : ZoneInfo

    Returns
    -------
    trans : ndarray[int64_t]
        Nanosecond UTC times of DST transitions.
    deltas : ndarray[int64_t]
        Nanosecond UTC offsets corresponding to DST transitions.
    is_fixed : bint
        True if this is a fixed-offset timezone with no transitions.
    has_tz_tule : bint
        Bool indicating if the tz has a "rule" attached describing transitions
        after the array of trans/deltas.
    """
    cdef:
        int64_t fixed_offset_seconds, last_hist_ts, start_utc, end_utc
        list trans_utc, deltas_seconds, future_trans
        int year, last_year, std_offset, dst_offset

    tz_py = _ZoneInfo(tz.key)

    if tz_py._fixed_offset:
        fixed_offset_seconds = int(tz_py._tz_after.utcoff.total_seconds())
        trans = np.array([NPY_NAT + 1], dtype=np.int64)
        deltas = np.array([fixed_offset_seconds], dtype="i8") * 1_000_000_000
        return trans, deltas, True, False

    trans_utc = list(tz_py._trans_utc)
    deltas_seconds = [int(info.utcoff.total_seconds()) for info in tz_py._ttinfos]

    has_future_dst = (
        hasattr(tz_py, "_tz_after")
        and tz_py._tz_after is not None
        and hasattr(tz_py._tz_after, "transitions")
    )
    if has_future_dst:
        # ZoneInfo's _trans_utc only contains historical data (typically up to
        # the late 1990s for many timezones). POSIX TZ rules stored in _tz_after
        # are required to generate transitions for current and future dates.
        # Without this block, DST-observing timezones like Europe/Amsterdam
        # would return incorrect offsets for any date after ~1996.
        tz_after = tz_py._tz_after
        std_offset = int(tz_after.std.utcoff.total_seconds())
        dst_offset = int(tz_after.dst.utcoff.total_seconds())

        if trans_utc:
            last_hist_ts = trans_utc[-1]
            try:
                # NB: not fromtimestamp, which goes through the platform
                #  time_t and overflows past 2038 on 32-bit. GH#67826
                last_year = (_UTC_EPOCH + timedelta(seconds=last_hist_ts)).year
            except (OSError, OverflowError, ValueError):
                last_year = 1970
        else:
            last_hist_ts = 0
            last_year = 1970

        future_trans = []
        for year in range(last_year, 2100):
            try:
                year_trans = tz_after.transitions(year)
                if not year_trans:
                    break
                start_local, end_local = year_trans
                start_utc = start_local - std_offset
                end_utc = end_local - dst_offset
                if start_utc > last_hist_ts:
                    future_trans.append((start_utc, dst_offset))
                if end_utc > last_hist_ts:
                    future_trans.append((end_utc, std_offset))
            except Exception:
                break

        future_trans.sort()

        for t, d in future_trans:
            trans_utc.append(t)
            deltas_seconds.append(d)

    first_offset_seconds = int(tz_py._tti_before.utcoff.total_seconds())

    # Certain older systems (toolchain from between 2013 and 2018f) have TZif
    #  files that start with a "big bang" transition at -2**59 seconds, which
    #  wraps when scaled to nanoseconds and leaves trans unsorted; the
    #  NPY_NAT + 1 sentinel below stands in for it. GH#67066
    if trans_utc and trans_utc[0] < _MIN_TRANS_SECONDS:
        del trans_utc[0]
        del deltas_seconds[0]

    trans = np.array(trans_utc, dtype="i8") * 1_000_000_000
    trans = np.hstack([np.array([NPY_NAT + 1], dtype=np.int64), trans])

    deltas = np.array(deltas_seconds, dtype="i8") * 1_000_000_000
    deltas = np.hstack([[first_offset_seconds * 1_000_000_000], deltas])

    return trans, deltas, False, has_future_dst


cdef object get_dst_info(tzinfo tz):
    # [FUNCTION DOC]
    # Return (trans, deltas, typ, has_tz_rule) for tz (see module doc for the array
    #   semantics), memoized in dst_cache.
    # Uncacheable tz (tz_cache_key is None): not cached; returns ([NaT+1],
    #   [utcoffset(None) in ns], 'unknown', False).
    # pytz DST zone: trans = _utc_transition_times as ns (first entry replaced by NaT+1
    #   if it is year 1), deltas from unbox_utcoffsets, typ 'pytz'.
    # dateutil tzfile with transitions: trans = [NaT+1] + UTC transition times (via
    #   _get_utc_trans_times_from_dateutil_tz, seconds -> ns); deltas = offsets of
    #   _ttinfo_before followed by each _trans_idx entry, in ns; typ 'dateutil'. Without
    #   transitions but fixed: ([NaT+1], [_ttinfo_std offset ns], 'fixed').
    # ZoneInfo: from _get_zoneinfo_trans_and_deltas; typ 'fixed' or 'zoneinfo',
    #   has_tz_rule passed through. Other cacheable tz (e.g. pytz StaticTzInfo):
    #   ([NaT+1], [utcoffset(None) ns], 'static').
    # Inputs: tz - tzinfo (not None).
    # Outputs: 4-tuple as above; cached arrays are shared, callers must not mutate them.
    # Exceptions: AttributeError if utcoffset(None) returns None on the uncacheable
    #   path; AssertionError for a non-fixed dateutil zone with an empty _trans_list;
    #   ValueError from tz_cache_key.
    # Side effects: inserts into the global dst_cache.
    """
    Returns
    -------
    ndarray[int64_t]
        Nanosecond UTC times of DST transitions.
    ndarray[int64_t]
        Nanosecond UTC offsets corresponding to DST transitions.
    str
        Describing the type of tzinfo object.
    has_tz_rule
        Bool indicating if the tz has a "rule" attached describing transitions
        after the array of transitions/deltas (only used for ZoneInfo).
    """
    cache_key = tz_cache_key(tz)
    if cache_key is None:
        # e.g. pytz.FixedOffset, matplotlib.dates._UTC,
        # psycopg2.tz.FixedOffsetTimezone
        num = int(get_utcoffset(tz, None).total_seconds()) * 1_000_000_000
        # If we have e.g. ZoneInfo here, the get_utcoffset call will return None,
        #  so the total_seconds() call will raise AttributeError.
        return (np.array([NPY_NAT + 1], dtype=np.int64),
                np.array([num], dtype=np.int64),
                "unknown",
                False)

    has_tz_rule = False
    if cache_key not in dst_cache:
        if treat_tz_as_pytz(tz):
            trans = np.array(tz._utc_transition_times, dtype="M8[ns]")
            trans = trans.view("i8")
            if tz._utc_transition_times[0].year == 1:
                trans[0] = NPY_NAT + 1
            deltas = unbox_utcoffsets(tz._transition_info)
            typ = "pytz"

        elif treat_tz_as_dateutil(tz):
            if len(tz._trans_list):
                # get utc trans times
                trans_list = _get_utc_trans_times_from_dateutil_tz(tz)
                trans = np.hstack([
                    np.array([0], dtype="M8[s]"),  # place holder for 1st item
                    np.array(trans_list, dtype="M8[s]")]).astype(
                    "M8[ns]")  # all trans listed
                trans = trans.view("i8")
                trans[0] = NPY_NAT + 1

                # deltas
                deltas = np.array([v.offset for v in (
                    tz._ttinfo_before,) + tz._trans_idx], dtype="i8")
                deltas *= 1_000_000_000
                typ = "dateutil"

            elif is_fixed_offset(tz):
                trans = np.array([NPY_NAT + 1], dtype=np.int64)
                deltas = np.array([tz._ttinfo_std.offset],
                                  dtype="i8") * 1_000_000_000
                typ = "fixed"
            else:
                # 2018-07-12 this is not reached in the tests, and this case
                # is not handled in any of the functions that call
                # get_dst_info.  If this case _were_ hit the calling
                # functions would then hit an IndexError because they assume
                # `deltas` is non-empty.
                # (under the just-deleted code that returned empty arrays)
                raise AssertionError("dateutil tzinfo is not a FixedOffset "
                                     "and has an empty `_trans_list`.", tz)

        elif is_zoneinfo(tz):
            trans, deltas, is_fixed, has_tz_rule = _get_zoneinfo_trans_and_deltas(tz)
            typ = "fixed" if is_fixed else "zoneinfo"

        else:
            # static tzinfo, we can get here with pytz.StaticTZInfo
            #  which are not caught by treat_tz_as_pytz
            trans = np.array([NPY_NAT + 1], dtype=np.int64)
            num = int(get_utcoffset(tz, None).total_seconds()) * 1_000_000_000
            deltas = np.array([num], dtype=np.int64)
            typ = "static"

        dst_cache[cache_key] = (trans, deltas, typ, has_tz_rule)

    return dst_cache[cache_key]


def infer_tzinfo(datetime start, datetime end):
    # [FUNCTION DOC]
    # Return the common tzinfo of two optional datetimes: start.tzinfo if both given
    #   (after checking tz_compare equality), else the tzinfo of whichever is not None,
    #   else None.
    # Inputs: start, end - datetime or None.
    # Outputs: tzinfo or None.
    # Exceptions: AssertionError if both given with non-equal timezones.
    # Side effects: none.
    if start is not None and end is not None:
        tz = start.tzinfo
        if not tz_compare(tz, end.tzinfo):
            raise AssertionError(f"Inputs must both have the same timezone, "
                                 f"{tz} != {end.tzinfo}")
    elif start is not None:
        tz = start.tzinfo
    elif end is not None:
        tz = end.tzinfo
    else:
        tz = None
    return tz


cpdef bint tz_compare(tzinfo start, tzinfo end):
    # [FUNCTION DOC]
    # Zone-level equality of two tzinfos. If start is UTC (any flavor) return
    #   is_utc(end); else if end is UTC return False; else if either is None return
    #   whether both are None; else compare get_timezone(start) == get_timezone(end)
    #   (zone names/keys, so different pytz offset instances of one zone are equal).
    # Inputs: start, end - tzinfo or None.
    # Outputs: bool.
    # Exceptions: ValueError from get_timezone for broken dateutil filenames.
    # Side effects: none.
    """
    Compare string representations of timezones

    The same timezone can be represented as different instances of
    timezones. For example
    `<DstTzInfo 'Europe/Paris' LMT+0:09:00 STD>` and
    `<DstTzInfo 'Europe/Paris' CET+1:00:00 STD>` are essentially same
    timezones but aren't evaluated such, but the string representation
    for both of these is `'Europe/Paris'`.

    This exists only to add a notion of equality to pytz-style zones
    that is compatible with the notion of equality expected of tzinfo
    subclasses.

    Parameters
    ----------
    start : tzinfo
    end : tzinfo

    Returns:
    -------
    bool
    """
    # GH 18523
    if is_utc(start):
        # GH#38851 consider pytz/dateutil/stdlib UTCs as equivalent
        return is_utc(end)
    elif is_utc(end):
        # Ensure we don't treat tzlocal as equal to UTC when running in UTC
        return False
    elif start is None or end is None:
        return start is None and end is None
    return get_timezone(start) == get_timezone(end)


def tz_standardize(tz: tzinfo) -> tzinfo:
    # [FUNCTION DOC]
    # Normalize a pytz DST zone (possibly a localized per-offset instance) to the
    #   canonical pytz.timezone(str(tz)) object; any other tz is returned unchanged
    #   (also when pytz is not installed).
    # Inputs: tz - tzinfo. Outputs: tzinfo.
    # Exceptions: pytz UnknownTimeZoneError if str(tz) is not a zone name. Side effects:
    #   none.
    """
    If the passed tz is a pytz timezone object, "normalize" it to the a
    consistent version

    Parameters
    ----------
    tz : tzinfo

    Returns
    -------
    tzinfo

    Examples
    --------
    >>> from datetime import datetime
    >>> from pytz import timezone
    >>> tz = timezone('US/Pacific').normalize(
    ...     datetime(2014, 1, 1, tzinfo=pytz.utc)
    ... ).tzinfo
    >>> tz
    <DstTzInfo 'US/Pacific' PST-1 day, 16:00:00 STD>
    >>> tz_standardize(tz)
    <DstTzInfo 'US/Pacific' LMT-1 day, 16:07:00 STD>

    >>> tz = timezone('US/Pacific')
    >>> tz
    <DstTzInfo 'US/Pacific' LMT-1 day, 16:07:00 STD>
    >>> tz_standardize(tz)
    <DstTzInfo 'US/Pacific' LMT-1 day, 16:07:00 STD>
    """
    if treat_tz_as_pytz(tz) and pytz is not None:
        return pytz.timezone(str(tz))
    return tz
