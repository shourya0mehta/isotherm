"""Market strike parsing and probability helpers for bracket ladders.

Kalshi daily temperature markets settle on an integer °F value V:
  strike_type 'less'    cap=C      -> YES iff V <  C   (V <= C-1)
  strike_type 'between' floor=F,cap=C -> YES iff F <= V <= C (inclusive)
  strike_type 'greater' floor=F    -> YES iff V >  F   (V >= F+1)
Hourly index markets (KXTEMPMIAH) settle on a continuous value with 0.01 precision:
  'greater' floor=80.99 -> YES iff index > 80.99 (i.e. >= 81.00)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

INF = 10 ** 6


@dataclass(frozen=True)
class Bracket:
    ticker: str
    lo: float  # inclusive lower bound on settlement value (integer world) or exclusive strike (continuous)
    hi: float  # inclusive upper bound
    strike_type: str
    floor: Optional[float]
    cap: Optional[float]

    def contains_int(self, v: int) -> bool:
        return self.lo <= v <= self.hi

    def label(self) -> str:
        if self.lo <= -INF:
            return f"<={int(self.hi)}"
        if self.hi >= INF:
            return f">={int(self.lo)}"
        return f"{int(self.lo)}-{int(self.hi)}"


def bracket_from_market(m: dict) -> Bracket:
    st = m.get("strike_type")
    f = m.get("floor_strike")
    c = m.get("cap_strike")
    if st == "less":
        return Bracket(m["ticker"], -INF, float(c) - 1, st, f, c)
    if st == "between":
        return Bracket(m["ticker"], float(f), float(c), st, f, c)
    if st == "greater":
        return Bracket(m["ticker"], float(f) + 1, INF, st, f, c)
    raise ValueError(f"unsupported strike_type {st} for {m.get('ticker')}")


def int_probs_to_bracket_probs(pmf: dict, brackets: list[Bracket]) -> dict:
    """pmf: {int value: prob}. Returns {ticker: prob of YES}."""
    out = {}
    for b in brackets:
        out[b.ticker] = float(sum(p for v, p in pmf.items() if b.contains_int(v)))
    return out


def market_implied_pmf(brackets: list[Bracket], mids: dict) -> dict:
    """Spread each bracket's normalized mid probability evenly over its integer values.
    Tail brackets get a 2-value geometric spill so the pmf is defined. Used only for blending."""
    tot = sum(max(mids.get(b.ticker, 0.0), 0.0) for b in brackets)
    if tot <= 0:
        return {}
    pmf: dict = {}
    for b in brackets:
        p = max(mids.get(b.ticker, 0.0), 0.0) / tot
        if b.lo <= -INF:
            vals = [int(b.hi) - k for k in range(4)]
            w = [0.5, 0.25, 0.15, 0.10]
        elif b.hi >= INF:
            vals = [int(b.lo) + k for k in range(4)]
            w = [0.5, 0.25, 0.15, 0.10]
        else:
            vals = list(range(int(b.lo), int(b.hi) + 1))
            w = [1.0 / len(vals)] * len(vals)
        for v, ww in zip(vals, w):
            pmf[v] = pmf.get(v, 0.0) + p * ww
    return pmf
