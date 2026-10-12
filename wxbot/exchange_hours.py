"""Kalshi's published trading hours (GET /exchange/schedule): is trading closed now, or about to close?

Every Thursday 3-5 AM ET Kalshi pauses trading for maintenance (docs: getting_started/maintenance_and_pauses): orders
cannot be placed or amended, and the bot's orders, sent with cancel_order_on_pause, are cancelled when it starts. The
bot learns of a pause from GET /exchange/status once a minute; until then every order it re-places would be refused
and counted as an API error (round-7 pre-mortem: an "API error burst" halt most Thursday nights). The schedule lets it
stop a minute before. Times in the schedule are ET "HH:MM"; a session closing at "00:00" runs to the end of the day.
Anything unreadable counts as open: the live status check still applies.
"""
from __future__ import annotations

from typing import Optional

from .timeutil import parse_ts, to_et

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _minutes(hhmm: str) -> Optional[int]:
    try:
        h, m = str(hhmm).split(":")
        return int(h) * 60 + int(m)
    except (ValueError, AttributeError):
        return None


def _open_in_week(week: dict, weekday: int, minute: float, lead_min: float) -> Optional[bool]:
    """Is trading open at this ET weekday/minute under one weekly schedule? None if the entry cannot be read. A
    session closing at "00:00" runs to the end of the day; one closing before it opens runs past midnight into the
    next day. The lead (stop that many minutes early) applies only before a real close, not at midnight."""
    today = week.get(DAYS[weekday])
    yesterday = week.get(DAYS[(weekday - 1) % 7])
    if today is None:
        return None
    for s in today:
        a, b = _minutes(s.get("open_time")), _minutes(s.get("close_time"))
        if a is None or b is None:
            return None
        if b == 0:
            b = 1440
        if b > a:
            close = b - (lead_min if b < 1440 else 0.0)
        else:
            close = 1440.0                  # runs past midnight: open to the end of today
        if a <= minute < close:
            return True
    for s in yesterday or []:
        a, b = _minutes(s.get("open_time")), _minutes(s.get("close_time"))
        if a is not None and b is not None and 0 < b < a and minute < b - lead_min:
            return True                     # yesterday's session running past midnight
    return False


def _effective_week(schedule: dict, now: float) -> Optional[dict]:
    weeks = [w for w in schedule.get("standard_hours") or []
             if (parse_ts(w.get("start_time")) or 0) <= now < (parse_ts(w.get("end_time")) or float("inf"))]
    return weeks[-1] if weeks else None


def closed_now(schedule: Optional[dict], now: float, lead_s: float = 60.0) -> bool:
    """True when the schedule has trading closed at `now` (epoch seconds) or closing within lead_s seconds."""
    if not schedule:
        return False
    try:
        for w in schedule.get("maintenance_windows") or []:
            a, b = parse_ts(w.get("start_datetime")), parse_ts(w.get("end_datetime"))
            if a and b and a - lead_s <= now < b:
                return True
        week = _effective_week(schedule, now)
        if week is None:
            return False
        et = to_et(now)
        is_open = _open_in_week(week, et.weekday(), et.hour * 60 + et.minute + et.second / 60.0, lead_s / 60.0)
        return is_open is False
    except Exception:
        return False


def closed_hours_per_week(schedule: Optional[dict], now: float) -> Optional[float]:
    """How many hours a week the schedule in effect now keeps trading closed (maintenance windows aside). Kalshi's
    is 2 (Thursday 3-5 AM ET); a much larger number means the schedule is not what the bot expects."""
    try:
        week = _effective_week(schedule or {}, now)
        if week is None:
            return None
        closed = 0
        for d in range(7):
            for m in range(0, 1440, 5):
                r = _open_in_week(week, d, m, 0.0)
                if r is None:
                    return None
                closed += 5 if r is False else 0
        return closed / 60.0
    except Exception:
        return None


def next_change(schedule: Optional[dict], now: float, horizon_h: float = 48.0) -> Optional[float]:
    """Epoch seconds of the next open/closed change within horizon_h hours (minute resolution), for the status."""
    if not schedule:
        return None
    state = closed_now(schedule, now, lead_s=0.0)
    t = (int(now) // 60 + 1) * 60
    end = now + horizon_h * 3600
    while t < end:
        if closed_now(schedule, t, lead_s=0.0) != state:
            return float(t)
        t += 60
    return None


def describe(schedule: Optional[dict], now: float) -> dict:
    nxt = next_change(schedule, now)
    return {"closed_now": closed_now(schedule, now, lead_s=0.0),
            "next_change_et": to_et(nxt).strftime("%a %H:%M") if nxt else None,
            "read": schedule is not None}

