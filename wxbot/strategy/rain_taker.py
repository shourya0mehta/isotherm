"""Sleeve R (shadow only): the model taker on Kalshi's daily rain markets (KXRAIN), rule frozen Oct 3, 2026
(docs/STRATEGY.md 4d, research/rain).

At each decision hour, for each city with a forecast: buy YES at the ask when the pooled probability beats it by the
taker fee plus MARGIN; buy NO at 1 - bid when the pooled chance of no rain beats that by the same; fee =
ceil(0.07 P (1 - P) * 100) / 100 per contract at price P; nothing at 1c or 99c and beyond; first signal per market and
side (the caller remembers what was logged)."""
from __future__ import annotations

import math


def fee(P: float) -> float:
    return math.ceil(round(0.07 * P * (1 - P) * 100, 9)) / 100


def signals(quotes: dict, p_pool: dict, margin: float = 0.06, max_spread: float = 0.25) -> list:
    """quotes: ticker -> (yes_bid, yes_ask) in dollars (None when absent); p_pool: ticker -> P(yes).
    Returns [{ticker, side, price, fee, edge}] for every side that clears the bar."""
    out = []
    for tk, p in p_pool.items():
        if tk not in quotes or p is None:
            continue
        b, a = quotes[tk]
        b = 0.0 if b is None else float(b)
        a = 1.0 if a is None else float(a)
        if a - b > max_spread + 1e-9 or a <= 0 or b >= 1:
            continue
        for side, P, edge in (("yes", a, p - a), ("no", 1 - b, (1 - p) - (1 - b))):
            if P <= 0.01 + 1e-9 or P >= 0.99 - 1e-9:
                continue
            f = fee(P)
            if edge > f + margin + 1e-12:
                out.append({"ticker": tk, "side": side, "price": round(P, 4), "fee": f, "edge": edge})
    return out
