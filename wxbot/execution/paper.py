"""Paper broker: simulates resting post-only orders against the live public trade tape and order books.

Fill model (conservative, queue-aware):
  - On placement we sit at the BACK of the queue at our price: queue_ahead = displayed size at that level.
  - Later book snapshots can only shrink queue_ahead (cancels ahead of us), never grow it.
  - A public trade where the taker SELLS into bids (for our YES bid: taker bought NO) at a price BELOW our bid
    means the taker swept through our level -> we fill (up to trade size).
  - A trade exactly AT our price first consumes queue_ahead, then fills us.
  - Trades print only after our placement time count.
Kalshi trade fields: yes_price_dollars, count_fp, taker_side ('yes' = taker bought YES, hitting YES asks /
NO bids; 'no' = taker bought NO, hitting YES bids).
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Optional

from ..strategy.types import DesiredOrder, Fill
from ..timeutil import parse_ts

_ids = itertools.count(1)


@dataclass
class PaperOrder:
    oid: str
    ticker: str
    event: str
    side: str
    price_cents: int
    remaining: float
    sleeve: str
    placed_ts: float
    queue_ahead: float
    fair_yes: float
    reason: str = ""
    filled: float = 0.0
    cancelled_ts: Optional[float] = None  # when the bot cancelled it; trades printed before then can still fill it
    queue_ts: float = 0.0   # time of the book snapshot that last shrank queue_ahead


def level_size(book: dict, side: str, price_cents: int) -> float:
    """Displayed size at our level. YES bids live in yes_dollars at YES price; our 'ask' (long NO at YES price
    p) rests among NO bids at NO price 100-p."""
    key, px = ("yes_dollars", price_cents) if side == "bid" else ("no_dollars", 100 - price_cents)
    for p, q in book.get(key) or []:
        if int(round(float(p) * 100)) == px:
            return float(q)
    return 0.0


class PaperBroker:
    def __init__(self, late_fills: bool = False):
        """late_fills: let a cancelled order still fill from trades printed before its cancel (live paper trading,
        where the tape arrives in batches after the quotes moved). Replays process trades in time order, so they
        keep the default (a cancel takes effect at once)."""
        self.late_fills = late_fills
        self.orders: dict[str, PaperOrder] = {}
        self.fills: list[Fill] = []
        self.by_ticker: dict[str, list[PaperOrder]] = {}

    # Trades arrive in batches (the bot polls the tape every ~20 s) while quotes are re-evaluated every tick, so an
    # order may be cancelled or replaced after a trade that should have filled it printed. A cancelled order stays
    # fillable by trades printed before its cancel time (as on the exchange, where the cancel arrives later).
    LATE_FILL_S = 120.0

    def resting(self) -> list[PaperOrder]:
        return [o for o in self.orders.values() if o.remaining > 0 and o.cancelled_ts is None]

    def resting_for(self, ticker: str) -> list[PaperOrder]:
        lst = [o for o in self.by_ticker.get(ticker, []) if o.remaining > 0 and o.cancelled_ts is None]
        return lst

    def fillable_for(self, ticker: str, ts: float) -> list[PaperOrder]:
        """Orders a trade printed at ts can fill: open ones, and cancelled ones whose cancel came after ts.
        Orders cancelled more than LATE_FILL_S before ts are dropped for good."""
        keep = [o for o in self.by_ticker.get(ticker, [])
                if o.remaining > 0 and (o.cancelled_ts is None or o.cancelled_ts > ts - self.LATE_FILL_S)]
        self.by_ticker[ticker] = keep
        return [o for o in keep if o.cancelled_ts is None or ts < o.cancelled_ts]

    def place(self, d: DesiredOrder, book: dict, now: Optional[float] = None) -> PaperOrder:
        now = now or time.time()
        oid = f"paper-{next(_ids)}"
        o = PaperOrder(oid, d.ticker, d.event, d.side, d.price_cents, d.count, d.sleeve, now,
                       level_size(book, d.side, d.price_cents), d.fair_yes, d.reason)
        self.orders[oid] = o
        self.by_ticker.setdefault(d.ticker, []).append(o)
        return o

    def cancel(self, oid: str, now: Optional[float] = None):
        o = self.orders.get(oid)
        if o and not self.late_fills:
            o.remaining = 0.0
            return
        if o and o.cancelled_ts is None:
            o.cancelled_ts = now or time.time()
        # forget orders that can no longer fill (keeps the dict small on long runs)
        if len(self.orders) > 5000:
            cut = (now or time.time()) - self.LATE_FILL_S
            self.orders = {k: v for k, v in self.orders.items()
                           if v.remaining > 0 and (v.cancelled_ts is None or v.cancelled_ts > cut)}

    def amend_size(self, oid: str, new_remaining: float):
        o = self.orders.get(oid)
        if o and new_remaining < o.remaining:
            o.remaining = new_remaining  # shrinking keeps queue priority on Kalshi

    def on_book(self, ticker: str, book: dict, ts: Optional[float] = None):
        """ts: when the book was read. A smaller displayed level shrinks our queue; the trades at our price that
        shrank it may reach us later (the live tape arrives in batches), and those must not fill us a second time
        (Oct 4 review, finding 2.11): on_trade skips at-price trades printed before queue_ts."""
        ts = ts or time.time()
        for o in self.resting_for(ticker):
            lvl = level_size(book, o.side, o.price_cents)
            # displayed level includes our own order in reality; in paper it does not -> others = lvl
            if lvl < o.queue_ahead:
                o.queue_ahead = lvl
                o.queue_ts = ts

    def on_trade(self, t: dict) -> list[Fill]:
        out = []
        tk = t.get("ticker")
        ts = _trade_ts(t)
        p = int(round(float(t["yes_price_dollars"]) * 100))
        cnt = float(t["count_fp"])
        taker = t.get("taker_side") or t.get("taker_outcome_side")
        for o in self.fillable_for(tk, ts):
            if ts <= o.placed_ts:
                continue
            if o.side == "bid":
                if taker != "no":  # only takers selling YES (buying NO) hit YES bids
                    continue
                through, at = p < o.price_cents, p == o.price_cents
            else:
                if taker != "yes":  # only takers buying YES hit our NO bid (YES ask at price_cents)
                    continue
                through, at = p > o.price_cents, p == o.price_cents
            if not (through or at):
                continue
            if at and ts <= o.queue_ts:
                continue   # this trade is already in the smaller queue a later book showed
            avail = cnt
            if at:
                eat = min(o.queue_ahead, avail)
                o.queue_ahead -= eat
                avail -= eat
            q = min(avail, o.remaining)
            if q <= 0:
                continue
            intended = o.remaining
            o.remaining -= q
            o.filled += q
            f = Fill(ts, tk, o.event, o.side, o.price_cents, q, o.sleeve, False, 0.0, o.oid,
                     intended=intended, depth=avail)
            self.fills.append(f)
            out.append(f)
            cnt -= q
            if cnt <= 0:
                break
        return out


def _trade_ts(t: dict) -> float:
    import datetime as dt
    s = t.get("created_time")
    if not s:
        return time.time()
    return parse_ts(s)
