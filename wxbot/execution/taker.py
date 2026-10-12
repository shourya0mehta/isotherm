"""Taker execution for the model-taker sleeves (Sleeve X on the daily highs, Sleeve R on the rain markets).

Both sleeves log only until their pre-registered tests pass (docs/HOLDOUT_DEC1.md). This module is the order path for
afterwards, kept off by default: a buy at most `limit` per contract, immediate-or-cancel, never resting on the book.

Sides are named by what we buy: "yes" lifts the YES offers (which are NO bids seen from the other side); "no" lifts
the NO offers (YES bids). Kalshi's order API names sides by the YES price: a YES buy is side "bid" at the YES price; a
NO buy at NO price q is side "ask" at YES price 100 - q (selling YES at no less than 100 - q).

Paper mode fills against an order-book snapshot (Kalshi orderbook_fp levels) at each level's price up to the limit,
so a thin book fills partly, as it would live. Fees: Kalshi's taker fee, ceil(0.07 x C x P x (1 - P)) to the cent per
order and price level.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("wxbot.taker")


def taker_fee(count: float, price: float) -> float:
    """Dollars for `count` contracts taken at `price` dollars in one fill."""
    return math.ceil(round(0.07 * count * price * (1 - price) * 100, 9)) / 100


def offers(book: dict, side: str) -> list:
    """Offers we can lift to buy `side`, cheapest first, as [(price_dollars, qty)]. YES offers are the NO bids at
    1 - price; NO offers are the YES bids at 1 - price."""
    levels = (book or {}).get("no_dollars" if side == "yes" else "yes_dollars") or []
    out = []
    for p, q in levels:
        p, q = float(p), float(q)
        if q > 0 and 0 < p < 1:
            out.append((round(1 - p, 4), q))
    return sorted(out)


def depth_at_or_better(book: Optional[dict], side: str, price: float) -> Optional[float]:
    """Contracts offered to a buyer of `side` at `price` (dollars) or better in an order-book snapshot; None when
    there is no snapshot (depth unknown)."""
    if book is None:
        return None
    return float(sum(q for p, q in offers(book, side) if p <= price + 1e-6))


def booked_size(intended: float, depth: Optional[float]) -> float:
    """What a shadow log books (outside review, Oct 8: the logs used to assume fills): the intended size capped by
    the contracts available at or better than the price, and nothing when that depth is unknown or zero."""
    if depth is None or not depth > 0:          # `not > 0` also catches NaN
        return 0.0
    return float(min(float(intended), float(depth)))


def simulate_take(book: dict, side: str, limit: float, count: float) -> tuple:
    """(filled, average price, fee) for buying up to `count` at no more than `limit` from a book snapshot."""
    filled = cost = fee = 0.0
    for p, q in offers(book, side):
        if p > limit + 1e-9 or filled >= count:
            break
        n = min(q, count - filled)
        filled += n
        cost += n * p
        fee += taker_fee(n, p)
    return filled, (cost / filled if filled else None), fee


@dataclass
class TakeResult:
    ticker: str
    side: str
    limit: float
    requested: float
    filled: float = 0.0
    avg_price: Optional[float] = None
    fee: float = 0.0
    order_id: Optional[str] = None
    status: str = "none"
    ts: float = field(default_factory=time.time)


class TakerExecutor:
    """take(ticker, side, limit, count, book) -> TakeResult. mode 'paper' simulates on the book; 'live' sends an
    immediate-or-cancel limit order. max_slip_c widens the limit by at most that many cents (default 0)."""

    def __init__(self, rest=None, mode: str = "paper", max_slip_c: int = 0):
        self.rest = rest
        self.mode = mode
        self.max_slip_c = int(max_slip_c)

    def take(self, ticker: str, side: str, limit: float, count: float, book: Optional[dict] = None,
             client_order_id: Optional[str] = None) -> TakeResult:
        if side not in ("yes", "no"):
            raise ValueError("side must be 'yes' or 'no'")
        lim = min(round(limit + self.max_slip_c / 100.0, 2), 0.99)
        res = TakeResult(ticker, side, lim, float(count))
        if count <= 0 or not 0 < lim < 1:
            res.status = "skipped"
            return res
        if self.mode != "live":
            if book is None:
                res.status = "no book"
                return res
            res.filled, res.avg_price, res.fee = simulate_take(book, side, lim, count)
            res.status = "filled" if res.filled >= count - 1e-9 else ("partial" if res.filled > 0 else "missed")
            return res
        price_c = int(round(lim * 100))
        api_side, api_price = ("bid", price_c) if side == "yes" else ("ask", 100 - price_c)
        try:
            r = self.rest.create_order(ticker, api_side, api_price, count, post_only=False,
                                       tif="immediate_or_cancel", client_order_id=client_order_id, retries=0)
        except Exception as e:  # network or exchange error: nothing filled as far as we know; the fills sync will tell
            log.warning("taker order %s %s failed: %s", ticker, side, e)
            res.status = "error"
            return res
        res.order_id = r.get("order_id")
        res.filled = float(r.get("fill_count") or 0)
        if res.filled > 0:
            avg = float(r.get("average_fill_price") or 0)
            # the v2 order API prices both sides on the YES scale (its docs do not say so for this field), so a NO
            # buy paid 1 - avg. An immediate-or-cancel take at the touch fills at or just inside its limit, never
            # worse: of the two readings, take the one at or nearest below the limit (round-7 review)
            if side == "yes":
                res.avg_price = avg
            else:
                as_yes = round(1 - avg, 4)
                ok = [p for p in (as_yes, avg) if p <= lim + 0.005]
                res.avg_price = min(ok, key=lambda p: lim - p) if ok else as_yes
            res.fee = float(r.get("average_fee_paid") or 0) * res.filled
        res.status = "filled" if res.filled >= count - 1e-9 else ("partial" if res.filled > 0 else "missed")
        return res
