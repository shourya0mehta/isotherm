"""Sleeve X, shadow only: the model taker (pre-registered Oct 3, 2026; docs/ML_RESEARCH.md Section 6e).

At each new ML decision hour, on liquid books only, buy YES at the ask when the pooled probability exceeds the ask by
more than the taker fee plus a margin, or buy NO at 1 - bid when 1 - p exceeds that by the same amount. The first
signal per (event, bracket, side) counts; later hours never add to it. The bot only LOGS these trades (table
taker_shadow) and settles them against Kalshi's results; it never sends a taker order. Real orders need the holdout
replay (Oct-Nov 2026, scored Dec 1) and the shadow log to both come out positive.

Liquid = mean bid-ask spread over brackets quoted on both sides <= 10c and the bracket mids sum to 0.90-1.25, the
same definition the research used (research/ml/tier2_eval_v2.py prep).

Shadow sizing (outside review, Oct 8): each signal also carries the intended size, the depth seen (contracts offered
at the signal's price or better) and the booked size, min(intended, depth), 0 when the depth is zero or unknown. The
shadow reports sum the booked size. `size` keeps its old meaning (the intended size capped by the displayed size, or
the intended size when none is displayed): it is what the real taker path sends, which this does not change.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..execution.taker import booked_size, depth_at_or_better


@dataclass
class TakerConfig:
    enabled: bool = False
    margin: float = 0.06       # required edge beyond the fee, in probability (6c)
    size: float = 5.0          # contracts per logged trade (capped by the displayed size at that price)
    max_spread: float = 0.10
    msum_lo: float = 0.90
    msum_hi: float = 1.25


PRIORITIES = ("city", "hour", "edge")


def priority_order(items: list, mode: str) -> list:
    """Sleeve X's signals that compete for one day's budget, in the order the bot sends them (taker_shadow.priority,
    Oct 9). At 21Z the evening-before signals (h = -3, +7.0c a contract over Aug-Sep 2026) and the same-day last-hour
    signals (h = 21, +0.7c) share one clock time, and the $15 budget went to them in the bot's city order.
      city  as they came (each event's signals as it is priced, in the configured series order): the default
      hour  an earlier decision time first, then the evening-before signals before the same-day ones (smaller h
            first), then as they came: the order of the r9 replay's "evening-before first" figure
      edge  as hour, but the larger edge first in place of the city order
    items: dicts with dec_ts (decision time, Unix seconds), h, edge (probability) and seq (arrival order)."""
    if mode == "hour":
        return sorted(items, key=lambda x: (x["dec_ts"], x["h"], x["seq"]))
    if mode == "edge":
        return sorted(items, key=lambda x: (x["dec_ts"], x["h"], -x["edge"], x["seq"]))
    return sorted(items, key=lambda x: x["seq"])


def rule_fee(price: float) -> float:
    """Taker fee per contract rounded up to the cent, as in the research rule (Kalshi rounds per order, so this
    slightly overstates the fee)."""
    return math.ceil(round(0.07 * price * (1 - price) * 100, 9)) / 100


def book_state(quotes: dict) -> tuple[float, float]:
    """(mean spread over brackets quoted on both sides, sum of bracket mids). Quotes are LadderQuote in cents."""
    spreads, msum = [], 0.0
    for q in quotes.values():
        if q.yes_bid is not None and q.yes_ask is not None and q.yes_bid > 0 and q.yes_ask < 100:
            spreads.append((q.yes_ask - q.yes_bid) / 100.0)
        m = q.mid()
        if m is not None:
            msum += m
    return (sum(spreads) / len(spreads) if spreads else 0.2), msum


def signals(quotes: dict, probs: dict, cfg: TakerConfig,
            books: Optional[dict] = None) -> tuple[bool, float, float, list[dict]]:
    """Returns (liquid, spread, mid_sum, signals). Each signal: ticker, side ('yes'/'no'), price (cost per contract in
    dollars), p (probability the bought side wins), edge (p - price - fee), avail (displayed size at that price),
    size (contracts the real taker path sends), intended (cfg.size), depth (contracts offered at the price or better:
    from the raw book in `books` when given, else the displayed size, None when neither is known) and booked
    (min(intended, depth), 0 when the depth is zero or unknown)."""
    spr, msum = book_state(quotes)
    liquid = spr <= cfg.max_spread + 1e-12 and cfg.msum_lo <= msum <= cfg.msum_hi
    out: list[dict] = []
    if not liquid:
        return liquid, spr, msum, out
    for t, q in quotes.items():
        p: Optional[float] = probs.get(t)
        if p is None:
            continue
        legs = []
        if q.yes_ask is not None and 0 < q.yes_ask < 100:
            legs.append(("yes", q.yes_ask / 100.0, p, q.yes_ask_size))
        if q.yes_bid is not None and 0 < q.yes_bid < 100:
            legs.append(("no", 1.0 - q.yes_bid / 100.0, 1.0 - p, q.yes_bid_size))
        for side, price, pw, avail in legs:
            edge = pw - price - rule_fee(price)
            if edge > cfg.margin:
                avail = float(avail or 0.0)
                raw = (books or {}).get(t)
                depth = depth_at_or_better(raw, side, round(price, 4)) if raw is not None else (avail or None)
                out.append({"ticker": t, "side": side, "price": round(price, 4), "p": pw, "edge": edge,
                            "avail": avail, "size": min(cfg.size, avail) if avail > 0 else cfg.size,
                            "intended": float(cfg.size), "depth": depth, "booked": booked_size(cfg.size, depth)})
    return liquid, spr, msum, out
