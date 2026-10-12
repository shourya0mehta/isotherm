"""Live surface observations (free): AWC METAR API + Kalshi's Miami weather index + IEM CLI.

Used for intraday guards: the daily max can only ratchet up and the daily min only down, so brackets that the
observed running extreme has already ruled out are dead. METAR temperatures are 5-minute averages at the obs
time with 0.1 °C precision (T-group); the 6-hour max/min groups (1sTTT / 2sTTT) catch between-hour extremes.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
from typing import Optional

import requests

log = logging.getLogger(__name__)
AWC = "https://aviationweather.gov/api/data/metar"
IEM_CLI = "https://mesonet.agron.iastate.edu/json/cli.py"
UA = {"User-Agent": "wxbot/0.1 (personal research bot)"}

SIX_HR = re.compile(r"\s([12])([01])(\d{3})(?=\s|$)")


def c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def fetch_metars(stations: list[str], hours: int = 30, timeout: float = 30.0) -> dict[str, list[dict]]:
    """One batched call per refresh (AWC limit: 100 req/min). Returns {icao: [obs...]} newest first."""
    out: dict[str, list[dict]] = {s: [] for s in stations}
    for i in range(0, len(stations), 40):
        ids = ",".join(stations[i:i + 40])
        r = requests.get(AWC, params={"ids": ids, "format": "json", "hours": hours}, headers=UA, timeout=timeout)
        r.raise_for_status()
        for o in r.json() or []:
            out.setdefault(o.get("icaoId"), []).append(o)
    for s in out:
        out[s].sort(key=lambda o: o.get("obsTime") or 0, reverse=True)
    return out


def six_hour_groups(raw: str) -> dict:
    """Parse 1sTTT (6-h max) and 2sTTT (6-h min) from the RMK section. Returns {'max': °C, 'min': °C}."""
    res = {}
    if not raw or " RMK " not in raw:
        return res
    rmk = raw.split(" RMK ", 1)[1]
    for kind, sign, val in SIX_HR.findall(" " + rmk):
        v = int(val) / 10.0 * (-1 if sign == "1" else 1)
        res["max" if kind == "1" else "min"] = v
    return res


def lst_day_bounds(local_date: dt.date, lst_offset_h: int) -> tuple[float, float]:
    start = dt.datetime(local_date.year, local_date.month, local_date.day, tzinfo=dt.timezone.utc) - dt.timedelta(hours=lst_offset_h)
    return start.timestamp(), (start + dt.timedelta(days=1)).timestamp()


def _despike(points: list[tuple[float, float]], max_jump_f: float = 10.0) -> list[tuple[float, float]]:
    """Drop single-report sensor/typo spikes: a reading more than max_jump_f away from BOTH neighbours (within 3 h)
    while the neighbours agree with each other. The newest reading needs its predecessor's agreement."""
    pts = sorted(points)
    keep = []
    for i, (t, f) in enumerate(pts):
        prev = pts[i - 1] if i > 0 and t - pts[i - 1][0] <= 3 * 3600 else None
        nxt = pts[i + 1] if i + 1 < len(pts) and pts[i + 1][0] - t <= 3 * 3600 else None
        if prev and nxt:
            if abs(f - prev[1]) > max_jump_f and abs(f - nxt[1]) > max_jump_f and abs(prev[1] - nxt[1]) < max_jump_f:
                continue
        elif prev and not nxt:
            if abs(f - prev[1]) > max_jump_f:
                continue  # unconfirmed jump on the newest report: wait for the next one
        keep.append((t, f))
    return keep


def running_extremes(obs: list[dict], local_date: dt.date, lst_offset_h: int) -> dict:
    """Observed max/min (°F) so far inside the LST climate day, from hourly/special obs + 6-hour groups.
    Single-report spikes are ignored, and a 6-hour group more than 6 °F beyond the hourly readings in its own
    window is treated as a glitch."""
    t0, t1 = lst_day_bounds(local_date, lst_offset_h)
    pts, groups = [], []
    for o in obs:
        t = o.get("obsTime")
        if t is None or not (t0 <= t < t1):
            continue
        temp = o.get("temp")
        if temp is not None:
            pts.append((float(t), c_to_f(float(temp))))
        g = six_hour_groups(o.get("rawOb", ""))
        # a 6-hour group reported at time t covers (t-6h, t]; only trust it if the whole window is in the day
        if g and t - 6 * 3600 >= t0:
            groups.append((float(t), g))
    pts = _despike(pts)
    mx = max((f for _, f in pts), default=None)
    mn = min((f for _, f in pts), default=None)
    for t, g in groups:
        win = [f for tt, f in pts if t - 6 * 3600 < tt <= t]
        if "max" in g:
            v = c_to_f(g["max"])
            if not win or v - max(win) <= 6.0:
                mx = v if mx is None else max(mx, v)
        if "min" in g:
            v = c_to_f(g["min"])
            if not win or min(win) - v <= 6.0:
                mn = v if mn is None else min(mn, v)
    return {"max_f": mx, "min_f": mn, "n_obs": len(pts), "last_obs_ts": max((t for t, _ in pts), default=None)}


def fetch_cli(station: str, year: int, timeout: float = 30.0) -> dict:
    """IEM parsed NWS CLI history for settlement checks: {date_iso: {'high': int, 'low': int, 'product': id}}."""
    r = requests.get(IEM_CLI, params={"station": station, "year": year}, headers=UA, timeout=timeout)
    r.raise_for_status()
    out = {}
    for row in (r.json() or {}).get("results", []):
        out[row["valid"]] = {"high": row.get("high"), "low": row.get("low"), "product": row.get("product")}
    return out


def kalshi_index_latest(rest, city: str = "miami") -> Optional[dict]:
    """Latest published Kalshi Weather Index point {'t': ms, 'v': °F, 'status', 'contributors'} (5 min delayed)."""
    try:
        d = rest.weather_index(city)
    except Exception as e:  # pragma: no cover - network
        log.warning("weather index fetch failed: %s", e)
        return None
    ts = d.get("timeseries") or []
    good = [p for p in ts if p.get("status") in ("normal", "degraded") and p.get("v") is not None]
    return good[-1] if good else None
