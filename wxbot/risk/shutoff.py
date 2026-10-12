"""Per-sleeve automatic shut-off: a one-sided CUSUM on each live trading sleeve's daily realized P&L (holdout amendments
draft of Oct 9, 2026, section A12; calibration in docs/results/r9_power_notes.txt, research/ml/r9_shutoff.py).

On each settlement day t (UTC; the day a sleeve's P&L is booked; a day with nothing settled counts as $0):
    S_t = max(0, S_(t-1) + k - x_t),  S = 0 when the count starts,
where x_t is the sleeve's realized P&L that day in dollars. k is half the replay's mean daily P&L at the live size; h is
the smallest threshold that switches a sleeve performing as replayed off within 61 days at most 5% of the time. When
S reaches h the sleeve opens no new positions (its open ones are held to settlement); only a manual reset
(`python -m wxbot reset-shutoff <sleeve>`) turns it back on, and its count then starts afresh.

x_t comes from the bot's own database (its booked fills joined with the settlements it recorded), so the statistic is
recomputed the same way after any restart; the switch-off itself is latched in the database (kv shutoff:<sleeve>).
Only complete UTC days count (a day's end plus a grace period for late settlement records), as in the calibration.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

DAY = 86400.0


def parse_since(v) -> Optional[float]:
    """Config `since`: a date ("2026-10-09", 00:00 UTC that day), an ISO time, or Unix seconds; None if absent."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dt.datetime):
        return (v if v.tzinfo else v.replace(tzinfo=dt.timezone.utc)).timestamp()
    if isinstance(v, dt.date):
        return dt.datetime(v.year, v.month, v.day, tzinfo=dt.timezone.utc).timestamp()
    s = str(v).strip()
    if len(s) == 10:
        d = dt.date.fromisoformat(s)
        return dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp()
    from ..timeutil import parse_ts
    return parse_ts(s)


@dataclass
class SleeveRule:
    k: float
    h: float
    since: Optional[float] = None     # Unix time the count starts (None: when the bot first checked this sleeve)


@dataclass
class ShutoffConfig:
    enabled: bool = False
    sleeves: dict = field(default_factory=dict)    # sleeve letter -> SleeveRule; only listed sleeves are watched
    grace_h: float = 1.0                           # a UTC day counts this long after it ends (late settlement records)

    @classmethod
    def from_cfg(cls, d: Optional[dict]) -> "ShutoffConfig":
        d = d or {}
        if not d.get("enabled", False):
            return cls(False, {}, float(d.get("grace_h", 1.0)))
        rules = {}
        for s, r in (d.get("sleeves") or {}).items():
            r = r or {}
            if r.get("k") is None or r.get("h") is None:
                raise ValueError(f"shutoff.sleeves.{s}: needs k and h (dollars)")
            rules[str(s).upper()[:1]] = SleeveRule(float(r["k"]), float(r["h"]), parse_since(r.get("since")))
        return cls(bool(d.get("enabled", False)), rules, float(d.get("grace_h", 1.0)))


def cusum(xs, k: float) -> list:
    """The statistic after each day: S_t = max(0, S_(t-1) + k - x_t), S_0 = 0."""
    out, s = [], 0.0
    for x in xs:
        s = max(0.0, s + k - float(x))
        out.append(s)
    return out


def first_crossing(xs, k: float, h: float) -> Optional[int]:
    """Index of the first day on which S reaches h, or None."""
    for i, s in enumerate(cusum(xs, k)):
        if s >= h:
            return i
    return None


def utc_day(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).date().isoformat()


def complete_days(since_ts: float, now: float, grace_h: float = 1.0) -> list:
    """UTC days from the day of since_ts through the last day whose end lies grace_h hours or more before now."""
    first = dt.datetime.fromtimestamp(since_ts, dt.timezone.utc).date()
    last = dt.datetime.fromtimestamp(now - grace_h * 3600, dt.timezone.utc).date() - dt.timedelta(days=1)
    n = (last - first).days + 1
    return [(first + dt.timedelta(days=i)).isoformat() for i in range(max(n, 0))]


def daily_pnl(conn, mode: str, sleeve: str, since_ts: float, until_ts: float) -> dict:
    """{UTC day: realized P&L in dollars} of one sleeve: its own fills in markets settled yes/no at or after since_ts
    and before until_ts, booked on the day of the settlement (as Bot.sleeve_pnl_today, by day)."""
    out: dict = {}
    for side, pc, n, fee, res, sts in conn.execute(
            "SELECT f.side, f.price_cents, f.count, f.fee, s.result, s.settled_ts FROM fills f JOIN settlements s "
            "ON f.ticker = s.ticker WHERE f.mode = ? AND f.sleeve = ? AND s.settled_ts >= ? AND s.settled_ts < ? "
            "AND s.result IN ('yes', 'no')", (mode, sleeve, since_ts, until_ts)):
        won = (res == "yes") == (side == "bid")
        cost = pc / 100.0 if side == "bid" else (100 - pc) / 100.0
        d = utc_day(float(sts))
        out[d] = out.get(d, 0.0) + float(n) * ((1.0 if won else 0.0) - cost) - float(fee or 0.0)
    return out


def evaluate(conn, mode: str, sleeve: str, rule: SleeveRule, since_ts: float, now: float,
             grace_h: float = 1.0) -> dict:
    """The sleeve's count over its complete days since since_ts: days, the statistic today (S) and its largest value,
    the realized P&L, and the first day S reached h (crossed_day, None if never)."""
    days = complete_days(since_ts, now, grace_h)
    if not days:
        return {"days": 0, "S": 0.0, "S_max": 0.0, "pnl_usd": 0.0, "crossed_day": None, "S_cross": None,
                "first_day": None, "last_day": None}
    end = dt.datetime.fromisoformat(days[-1]).replace(tzinfo=dt.timezone.utc).timestamp() + DAY
    pnl = daily_pnl(conn, mode, sleeve, since_ts, end)
    xs = [pnl.get(d, 0.0) for d in days]
    path = cusum(xs, rule.k)
    i = next((j for j, s in enumerate(path) if s >= rule.h), None)
    return {"days": len(days), "S": round(path[-1], 2), "S_max": round(max(path), 2), "pnl_usd": round(sum(xs), 2),
            "crossed_day": days[i] if i is not None else None, "S_cross": round(path[i], 2) if i is not None else None,
            "first_day": days[0], "last_day": days[-1]}
