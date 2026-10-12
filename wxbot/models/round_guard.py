"""Rounding-ambiguity guard for the daily high/low ladders (round 7, after the KXHIGHTOKC-26OCT08 loss).

ASOS measures temperature in whole degrees F (5-minute averages) and converts to C for its reports:
  - the free real-time 5-minute readings (NWS API / MADIS "HFMETAR") carry WHOLE degrees C;
  - the hourly and special METARs' T-groups and the 6-hour max/min groups (1sTTT / 2sTTT) carry tenths of C, which
    pin down the whole F exactly (each whole F has its own tenth);
  - Kalshi settles on the NWS climate report, in whole F.
A whole-C reading C is therefore consistent with every whole F whose (F - 32) * 5 / 9 rounds to C: one value when
C is a multiple of 5 (30 C = 86 F) and two values otherwise (31 C = 87 or 88 F). When the readings so far leave the
day's running max (or min) on two whole F values that sit in different brackets, those two brackets are ambiguous:
a near-certain price on either is a bet on the rounding, which the takers with the one-minute data know before us.
They stay ambiguous until an exact reading or a 6-hour group covering the readings settles it, or a later reading
makes the question moot.

Oct 8, KOKC: 31 C readings from 18:55Z left the high on 87 or 88 F (brackets 86-87 | 88-89); the market priced 88-89
at 93%, Sleeve A's resting near-certain bids there filled at 95c, and the 23:52Z 6-hour max (30.6 C = 87 F) settled it
the other way.

Pure functions, no I/O. Readings carry an optional `avail` time (when the bot had it) so replays can ask what the bot
knew at a given moment; live callers leave it at 0 (everything they hold is available).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Optional

SIX_H = 6 * 3600.0


def c_of_f(f: int) -> int:
    """Whole C an ASOS reports for a whole F. (F - 32) * 5 / 9 never ends in exactly .5 for a whole F, so there is
    no rounding tie."""
    return int(math.floor((f - 32) * 5.0 / 9.0 + 0.5))


F_OF_C: dict[int, tuple] = {}
for _f in range(-100, 161):
    F_OF_C.setdefault(c_of_f(_f), ())
    F_OF_C[c_of_f(_f)] += (_f,)


def f_set(c: int) -> tuple:
    """Every whole F a whole-C reading of c stands for (one or two values)."""
    if c in F_OF_C:
        return F_OF_C[c]
    lo = math.ceil((c - 0.5) * 9.0 / 5.0 + 32 - 1e-9)
    return tuple(f for f in range(lo - 1, lo + 3) if c_of_f(f) == c)


def f_values(temp_c: float) -> tuple:
    """The whole F values a reported temperature in C allows. Tenths (a T-group, a 6-hour group, the NWS API's hourly
    value) are an exact whole F; a whole C (a 5-minute reading, a METAR body without T-group) is its set. An exact
    conversion of a whole F is a whole C only at multiples of 5 C, whose set has one value, so 'within 0.05 of a whole
    C -> its set' is right for both kinds without knowing which feed the value came from."""
    r = round(temp_c)
    if abs(temp_c - r) < 0.05:
        return f_set(int(r))
    return (int(math.floor(temp_c * 9.0 / 5.0 + 32 + 0.5)),)


@dataclass(frozen=True)
class Reading:
    ts: float              # observation time, Unix seconds (UTC)
    vals: tuple            # whole F values it allows (1 = exact, 2 = whole-C ambiguity)
    kind: str = "inst"     # "inst" (a reading at ts) | "max6" | "min6" (6-hour extreme for (ts - 6 h, ts])
    avail: float = 0.0     # when the bot had it (replays); 0 = available


T_GROUP = re.compile(r"(?:^|\s)T([01])(\d{3})(?:([01])(\d{3}))?(?=\s|$)")
BODY_T = re.compile(r"(?:^|\s)(M?\d{1,2})/(M?\d{1,2})?(?=\s|$)")
SIX_G = re.compile(r"(?:^|\s)([12])([01])(\d{3})(?=\s|$)")


def _rmk(raw: str) -> tuple:
    if " RMK " in raw:
        body, rmk = raw.split(" RMK ", 1)
        return body, rmk
    return raw, ""


def t_group_c(raw: str) -> Optional[float]:
    """The T-group's temperature in C (tenths), or None."""
    if not raw:
        return None
    m = T_GROUP.search(_rmk(raw)[1])
    return int(m.group(2)) / 10.0 * (-1 if m.group(1) == "1" else 1) if m else None


def body_temp_c(raw: str) -> Optional[float]:
    """The METAR body's temperature (whole C), or None."""
    if not raw:
        return None
    for m in BODY_T.finditer(_rmk(raw)[0]):
        t = m.group(1)
        return -float(t[1:]) if t.startswith("M") else float(t)
    return None


def metar_temp_c(raw: str) -> Optional[float]:
    """Temperature in C from a METAR text: the T-group's tenths when present, else the body's whole C."""
    t = t_group_c(raw)
    return body_temp_c(raw) if t is None else t


def readings_from_metar(raw: str, ts: float, avail: float = 0.0, temp_c: Optional[float] = None) -> list:
    """Readings in one METAR/SPECI/5-minute report: the temperature (the T-group; else temp_c, the feed's decoded
    value, which keeps the tenths when the stored text was cut short; else the body's whole C) and any 6-hour max/min
    groups (tenths of C, exact)."""
    out = []
    t = t_group_c(raw)
    if t is None and temp_c is not None and isinstance(temp_c, (int, float)) and math.isfinite(temp_c):
        t = float(temp_c)
    if t is None:
        t = body_temp_c(raw)
    if t is not None:
        out.append(Reading(float(ts), f_values(float(t)), "inst", avail))
    if raw:
        _, rmk = _rmk(raw)
        for kind, sign, v in SIX_G.findall(rmk):
            c = int(v) / 10.0 * (-1 if sign == "1" else 1)
            out.append(Reading(float(ts), f_values(c), "max6" if kind == "1" else "min6", avail))
    return out


def extreme_candidates(readings: Iterable[Reading], t0: float, t1: float, kind: str = "high",
                       now: Optional[float] = None) -> tuple:
    """Whole F values the climate day's running max (kind "high") or min ("low") over the readings so far could be,
    from what is available at `now` (None = everything). The day is [t0, t1) in Unix seconds (midnight to midnight
    local standard time). Returns () when there is no reading yet.

    Each reading inside the day allows its values, cut by every 6-hour group whose window (ts - 6 h, ts] holds it (a
    6-hour max is the max of the window, so no reading in it exceeds it; if the two disagree the reading keeps its
    values). A same-minute exact reading settles an ambiguous one. Groups whose window lies inside the day are exact
    values of the running extreme. The answer is every whole F between the lowest and highest value the extreme of
    all that can take: (87, 88) after Oct 8's 31 C readings, (87,) once the 23:52Z 6-hour max (30.6 C) arrives."""
    hi_kind = kind != "low"
    gk = "max6" if hi_kind else "min6"
    inst, groups = [], []
    for r in readings:
        if now is not None and r.avail > now:
            continue
        if r.kind == "inst":
            if t0 <= r.ts < t1 and r.vals:
                inst.append(r)
        elif r.kind == gk and r.vals and r.ts > t0 and r.ts - SIX_H < t1:
            groups.append(r)
    if not inst and not any(g.ts - SIX_H >= t0 and g.ts < t1 for g in groups):
        return ()
    exact_at = {}
    for r in inst:
        if len(r.vals) == 1:
            exact_at.setdefault(r.ts, set()).add(r.vals[0])
    lows, highs = [], []
    for r in inst:
        vals = r.vals
        if len(vals) > 1:
            same = exact_at.get(r.ts)
            if same:
                v2 = tuple(v for v in vals if v in same)
                vals = v2 or vals
            for g in groups:
                if g.ts - SIX_H < r.ts <= g.ts:
                    v2 = tuple(v for v in vals if (v <= max(g.vals) if hi_kind else v >= min(g.vals)))
                    vals = v2 or vals
        lows.append(min(vals))
        highs.append(max(vals))
    for g in groups:
        if g.ts - SIX_H >= t0 and g.ts < t1:
            lows.append(min(g.vals))
            highs.append(max(g.vals))
    if hi_kind:
        lo, hi = max(lows), max(highs)
    else:
        lo, hi = min(lows), min(highs)
    return tuple(range(lo, hi + 1))


def ambiguous_brackets(brackets, cands: tuple) -> list:
    """Tickers of the brackets the candidate values fall in, when they fall in more than one (else [])."""
    if len(cands) < 2:
        return []
    hit = [b for b in brackets if any(b.lo <= v <= b.hi for v in cands)]
    return [b.ticker for b in hit] if len(hit) > 1 else []


def guard(readings, brackets, t0: float, t1: float, kind: str = "high", now: Optional[float] = None) -> tuple:
    """(ambiguous tickers as a frozenset, candidate values) for one city-day ladder."""
    cands = extreme_candidates(readings, t0, t1, kind, now)
    return frozenset(ambiguous_brackets(brackets, cands)), cands


def timeline(readings, brackets, t0: float, t1: float, kind: str = "high") -> list:
    """Replays: [(from_ts, frozenset of ambiguous tickers, candidates)] at every time the available set changes
    (ascending; the state holds until the next entry). Only changes are kept."""
    rs = [r for r in readings]
    times = sorted({r.avail for r in rs})
    out = []
    last = None
    for t in times:
        amb, cands = guard(rs, brackets, t0, t1, kind, now=t)
        if last is None or amb != last:
            out.append((t, amb, cands))
            last = amb
    return out
