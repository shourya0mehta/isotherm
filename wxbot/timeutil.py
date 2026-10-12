"""ISO-8601 parsing that works on Python 3.9+.

Kalshi timestamps carry a variable number of fractional digits ('08:43:49.31456Z'). Before Python 3.11,
datetime.fromisoformat accepts only 3 or 6 digits and no 'Z', so we normalize first.
"""
from __future__ import annotations

import datetime as dt
import re

_FRAC = re.compile(r"\.(\d+)")


def parse_dt(s) -> dt.datetime:
    s = str(s).strip().replace("Z", "+00:00")
    m = _FRAC.search(s)
    if m:
        s = s[:m.start()] + "." + (m.group(1) + "000000")[:6] + s[m.end():]
    d = dt.datetime.fromisoformat(s)
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)


def parse_ts(s) -> float:
    """Unix seconds, or 0.0 for empty input."""
    return parse_dt(s).timestamp() if s else 0.0


try:  # US Eastern for trading-day boundaries (daily loss stop, daily summary, dashboards)
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:  # no tz database: fall back to the US DST rule of thumb (EDT Mar-Oct, EST Nov-Feb)
    _ET = None


def to_et(ts: float) -> dt.datetime:
    """Unix seconds -> naive-free aware datetime in US Eastern time."""
    u = dt.datetime.fromtimestamp(ts, dt.timezone.utc)
    if _ET is not None:
        return u.astimezone(_ET)
    off = -4 if 3 <= u.month <= 10 else -5
    return u.astimezone(dt.timezone(dt.timedelta(hours=off)))


def et_day(ts: float | None = None) -> dt.date:
    import time as _t
    return to_et(_t.time() if ts is None else ts).date()


def _nth_sunday(year: int, month: int, n: int) -> dt.date:
    d = dt.date(year, month, 1)
    return d + dt.timedelta(days=(6 - d.weekday()) % 7 + 7 * (n - 1))


def et_standard_time(ts: float) -> bool:
    """True while US Eastern is on standard time (EST) at Unix time ts: from the first Sunday of November, 2 AM EDT
    (06:00 UTC), to the second Sunday of March, 2 AM EST (07:00 UTC); e.g. Nov 1, 2026 06:00 UTC to Mar 14, 2027
    07:00 UTC. The tz database decides when it is there; without it, the US rule above (to_et's fallback is only a
    month-level rule of thumb)."""
    if _ET is not None:
        return not dt.datetime.fromtimestamp(ts, dt.timezone.utc).astimezone(_ET).dst()
    u = dt.datetime.fromtimestamp(ts, dt.timezone.utc)
    start = dt.datetime.combine(_nth_sunday(u.year, 3, 2), dt.time(7), dt.timezone.utc)    # 2 AM EST -> EDT
    end = dt.datetime.combine(_nth_sunday(u.year, 11, 1), dt.time(6), dt.timezone.utc)     # 2 AM EDT -> EST
    return not (start <= u < end)
