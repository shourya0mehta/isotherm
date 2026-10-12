"""Sleeve B: liquidity-reward quoting on hourly Miami temperature (KXTEMPMIAH).

Each strike-hour carries a reward pool (verified $100 / 58-min period, target 1,000, discount 0.5) that only
pays for snapshots where BOTH sides hold >= target size. Overnight most strikes are one-sided. A bid at 1c on the
empty side makes the book two-sided; it only fills when someone buys the near-certain side at 99c from us, so
the cost per filled contract is ~1c and there is no adverse-selection tail (a move toward our side makes our bid
more valuable, so nobody sells to us at 1c).

DISABLED BY DEFAULT: Kalshi's terms let it revoke participants whose participation is "abusive or inconsistent
with the purpose of the Program". Turn on only after Kalshi support confirms this is acceptable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..models.fair_value import norm_cdf
from .lip import score_side
from .types import DesiredOrder, EventBook


@dataclass
class SleeveBConfig:
    enabled: bool = False
    max_win_prob: float = 0.01  # only quote 1c on a side whose fair win prob is <= 1%
    min_distance_f: float = 2.0
    sigma_60min_f: float = 1.45  # empirical 60-min sd of the Miami index (Oct 1-2, 2026)
    tail_mult: float = 1.6  # fat-tail inflation (storm outflows drop temps 5-8 F within an hour)
    max_contracts_per_strike: float = 1000
    target_default: float = 1000
    hours_utc: tuple = ()  # rollout control: only quote during these UTC hours (empty = all hours)
    max_strikes: int = 10  # rollout control: at most this many strikes per hourly event


def p_index_above(value_f: float, strike: float, minutes: float, cfg: SleeveBConfig) -> float:
    sd = max(cfg.sigma_60min_f * math.sqrt(max(minutes, 1.0) / 60.0) * cfg.tail_mult, 0.15)
    return 1 - norm_cdf((strike - value_f) / sd)


def propose(ev: EventBook, index_value_f: Optional[float], index_age_min: float, programs: dict,
            now_ts: float, cfg: SleeveBConfig) -> tuple[list[DesiredOrder], list[dict]]:
    """programs: ticker -> incentive program dict (active). Returns orders + per-strike diagnostics."""
    out, diag = [], []
    if index_value_f is None:
        return out, [{"event": ev.event, "note": "no index value"}]
    if cfg.hours_utc:
        import datetime as _dt
        if _dt.datetime.fromtimestamp(now_ts, _dt.timezone.utc).hour not in set(cfg.hours_utc):
            return out, [{"event": ev.event, "note": "outside hours_utc"}]
    minutes = max((ev.close_ts - now_ts) / 60.0 + index_age_min, 1.0)
    for m in ev.markets:
        t = m["ticker"]
        prog = programs.get(t)
        if not prog:
            continue
        strike = float(m.get("floor_strike") or 0.0)  # 'greater' strikes like 80.99 -> YES iff index > 80.99
        p_yes = p_index_above(index_value_f, strike, minutes, cfg)
        book = ev.books.get(t, {})
        yes_lv = [(int(round(float(p) * 100)), float(q)) for p, q in (book.get("yes_dollars") or [])]
        no_lv = [(int(round(float(p) * 100)), float(q)) for p, q in (book.get("no_dollars") or [])]
        target = float(prog.get("target_size_fp") or cfg.target_default)
        disc = (prog.get("discount_factor_bps") or 5000) / 10000.0
        sy, sn = score_side(yes_lv, target, disc), score_side(no_lv, target, disc)
        tot_y, tot_n = sum(q for _, q in yes_lv), sum(q for _, q in no_lv)
        # which side is cheap (near-certain loser)?
        if p_yes <= cfg.max_win_prob and abs(index_value_f - strike) >= cfg.min_distance_f:
            cheap, other_ok, have = "bid", sn.qualifies, tot_y  # YES is the cheap side; bid YES at 1c
        elif (1 - p_yes) <= cfg.max_win_prob and abs(index_value_f - strike) >= cfg.min_distance_f:
            cheap, other_ok, have = "ask", sy.qualifies, tot_n  # NO is cheap; buy NO at 1c == ask at YES 99c
        else:
            diag.append({"ticker": t, "p_yes": round(p_yes, 4), "note": "near the money: skip"})
            continue
        need = max(target - have, 0.0)
        if not other_ok:
            diag.append({"ticker": t, "p_yes": round(p_yes, 4), "note": "other side below target: skip"})
            continue
        size = min(cfg.max_contracts_per_strike, max(need, 0.0))
        if size < 1:
            diag.append({"ticker": t, "p_yes": round(p_yes, 4), "note": "cheap side already qualifies: skip"})
            continue
        price_yes = 1 if cheap == "bid" else 99
        out.append(DesiredOrder(ticker=t, side=cheap, price_cents=price_yes, count=float(math.ceil(size)),
                                sleeve="B", fair_yes=p_yes, event=ev.event,
                                reason=f"LIP quiet side ({'YES' if cheap == 'bid' else 'NO'} @1c) need={need:.0f}"))
        diag.append({"ticker": t, "p_yes": round(p_yes, 4), "note": f"quote {cheap} 1c x{math.ceil(size)}"})
    if len(out) > cfg.max_strikes:  # keep the strikes farthest from the index (safest 1c bids)
        def dist(o):
            m = next(mm for mm in ev.markets if mm["ticker"] == o.ticker)
            return abs(index_value_f - float(m.get("floor_strike") or 0.0))
        out = sorted(out, key=dist, reverse=True)[:cfg.max_strikes]
    return out, diag
