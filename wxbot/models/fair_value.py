"""Fair value for daily high/low temperature bracket ladders (v0).

Philosophy (from the Oct 2 research): the market's own distribution is more accurate than NOAA's NBM in almost
every city, so v0 is MARKET-ANCHORED:
  1. market-implied bracket probabilities from mids (normalized across the ladder)
  2. favorite-longshot correction via an empirical calibration map (Sept 2026, shrunk 50% toward identity)
  3. light blend with NBM (weight `w_nbm`, default 0.2) using an error-calibrated spread
  4. hard observation constraints on the target day (running max can only go up, running min only down)
  5. a disagreement score (total-variation distance market vs NBM) used by the strategy as a guard
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Optional

from .brackets import Bracket

_CAL_PATH = os.path.join(os.path.dirname(__file__), "calib_daily_temp.json")
try:
    _CAL = {float(k): v for k, v in json.load(open(_CAL_PATH))["map"].items()}
except Exception:  # pragma: no cover
    _CAL = {}


def calibrate(p: float) -> float:
    if not _CAL or p < 0.04 or p > 0.96:
        return p
    lo = math.floor(p * 100) / 100
    hi = min(lo + 0.01, 0.99)
    a, b = _CAL.get(round(lo, 2), p), _CAL.get(round(hi, 2), p)
    w = (p - lo) / 0.01 if hi > lo else 0
    return a + (b - a) * w


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def nbm_bracket_probs(brackets: list[Bracket], mean: float, sd: float) -> dict:
    """Integer settlement V; P(V=k) = Phi((k+.5-mu)/sd) - Phi((k-.5-mu)/sd)."""
    out = {}
    for b in brackets:
        lo = -math.inf if b.lo <= -1e5 else b.lo - 0.5
        hi = math.inf if b.hi >= 1e5 else b.hi + 0.5
        pl = 0.0 if lo == -math.inf else norm_cdf((lo - mean) / sd)
        ph = 1.0 if hi == math.inf else norm_cdf((hi - mean) / sd)
        out[b.ticker] = max(ph - pl, 0.0)
    s = sum(out.values()) or 1.0
    return {k: v / s for k, v in out.items()}


@dataclass
class LadderQuote:
    """Top-of-book snapshot for one bracket in YES cents (None if that side is empty)."""
    yes_bid: Optional[int]
    yes_ask: Optional[int]
    yes_bid_size: float = 0.0
    yes_ask_size: float = 0.0

    def mid(self) -> Optional[float]:
        if self.yes_bid is not None and self.yes_ask is not None:
            return (self.yes_bid + self.yes_ask) / 200.0
        if self.yes_ask is not None:  # nobody bids YES: value is somewhere in [0, ask]
            return self.yes_ask / 200.0
        if self.yes_bid is not None:  # nobody offers YES: value is somewhere in [bid, 100]
            return (self.yes_bid + 100) / 200.0
        return None


@dataclass
class FairValueResult:
    probs: dict  # ticker -> fair P(YES)
    market: dict  # ticker -> normalized market prob
    nbm: dict  # ticker -> NBM prob (may be empty)
    dead: set = field(default_factory=set)  # brackets ruled out by observations
    disagreement: float = 0.0  # total-variation distance market vs NBM (0..1)
    notes: list = field(default_factory=list)
    conflict: bool = False  # an observation contradicts the market: do not quote this event


def ladder_fair_value(brackets: list[Bracket], quotes: dict, *, kind: str, nbm_mean: Optional[float] = None,
                      nbm_sd: Optional[float] = None, hours_to_close: float = 24.0, obs_max_f: Optional[float] = None,
                      obs_min_f: Optional[float] = None, w_nbm: float = 0.2, sd_floor: Optional[float] = None,
                      kill_max_market_p: float = 0.20) -> FairValueResult:
    """kill_max_market_p: an observation may zero a bracket only if the market already prices it at or below this.
    If the market disagrees (a bad METAR, or a move the market has not priced yet), the event is flagged
    `conflict` and Sleeve A skips it until the two agree."""
    notes = []
    # 1) market-implied
    raw = {}
    for b in brackets:
        q = quotes.get(b.ticker)
        m = q.mid() if q else None
        if m is not None:
            raw[b.ticker] = m
    if len(raw) < max(2, len(brackets) - 1):
        notes.append("thin ladder: market mids missing")
    tot = sum(raw.values())
    mkt = {k: v / tot for k, v in raw.items()} if tot > 0 else {}
    # 2) favorite-longshot calibration, renormalized
    cal = {k: calibrate(v) for k, v in mkt.items()}
    s = sum(cal.values()) or 1.0
    cal = {k: v / s for k, v in cal.items()}
    # 3) NBM blend
    nbm = {}
    if nbm_mean is not None:
        floor = sd_floor if sd_floor is not None else (2.2 if hours_to_close > 18 else 1.9)
        if kind == "low":
            floor += 0.4  # CLI calendar-day min can fall late in the evening; NBM min window is the morning
        sd = max(float(nbm_sd or 0), floor)
        nbm = nbm_bracket_probs(brackets, nbm_mean, sd)
    if cal and nbm:
        fair = {b.ticker: (1 - w_nbm) * cal.get(b.ticker, 0.0) + w_nbm * nbm.get(b.ticker, 0.0) for b in brackets}
    elif cal:
        fair = dict(cal)
    else:
        fair = dict(nbm)
    # 4) observation constraints
    dead = set()
    if kind == "high" and obs_max_f is not None:
        floor_f = math.floor(obs_max_f + 0.5) - 1  # 1 °F safety for METAR vs CLI rounding/averaging
        for b in brackets:
            if b.hi < floor_f:
                dead.add(b.ticker)
    if kind == "low" and obs_min_f is not None:
        ceil_f = math.floor(obs_min_f + 0.5) + 1
        for b in brackets:
            if b.lo > ceil_f:
                dead.add(b.ticker)
    conflict = sorted(t for t in dead if mkt.get(t, 0.0) > kill_max_market_p)
    if conflict:
        notes.append(f"obs/market conflict on {len(conflict)} bracket(s): {','.join(conflict)[:120]}")
        dead -= set(conflict)
    if dead:
        for t in dead:
            fair[t] = 0.0
        s = sum(fair.values())
        if s > 0:
            fair = {k: v / s for k, v in fair.items()}
    # 5) disagreement
    dis = 0.0
    if mkt and nbm:
        dis = 0.5 * sum(abs(mkt.get(b.ticker, 0.0) - nbm.get(b.ticker, 0.0)) for b in brackets)
    return FairValueResult(fair, mkt, nbm, dead, dis, notes, bool(conflict))


def book_to_quote(book: dict) -> LadderQuote:
    """Kalshi orderbook_fp {'yes_dollars': [[p, size]...], 'no_dollars': [...]}, ascending, NO side in NO-price.
    YES ask = 1 - best NO bid."""
    yb = book.get("yes_dollars") or book.get("y") or []
    nb = book.get("no_dollars") or book.get("n") or []
    best_y = max(yb, key=lambda x: float(x[0])) if yb else None
    best_n = max(nb, key=lambda x: float(x[0])) if nb else None
    yes_bid = int(round(float(best_y[0]) * 100)) if best_y else None
    yes_ask = 100 - int(round(float(best_n[0]) * 100)) if best_n else None
    return LadderQuote(yes_bid, yes_ask, float(best_y[1]) if best_y else 0.0, float(best_n[1]) if best_n else 0.0)
