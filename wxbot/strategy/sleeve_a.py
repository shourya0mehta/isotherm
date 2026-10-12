"""Sleeve A: favorite-skewed maker on daily high/low temperature ladders.

Evidence (Sept 2026, 35M contracts): maker fills buying the favorite side (>=80c) earned +1.14c/contract with
positive markouts at 1/5/60 min; mid-priced (20-80c) maker fills on the target day lost ~2.5c. So:
  - only bid for the side whose fair probability >= fav_min_prob (YES on favorites, NO on longshots)
  - never quote mid-priced brackets
  - hold to settlement (most of the edge is paid at resolution)
  - cap 'decided' (>= decided_prob) exposure: an erroneous first-final report can flip a 99c outcome
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..models.fair_value import FairValueResult, LadderQuote
from .types import DesiredOrder, EventBook


@dataclass
class SleeveAConfig:
    fav_min_prob: float = 0.78
    min_edge_c: float = 2.0  # required fair-minus-price edge in cents for the side we buy
    improve_if_edge_c: float = 4.0  # step 1c inside the spread only if edge stays >= this
    clip_contracts: float = 25
    stop_minutes_before_close: float = 30
    max_disagreement: float = 0.35
    decided_prob: float = 0.97
    decided_max_cost_usd: float = 25.0
    min_price_c: int = 60  # never pay less than this for the favorite side (filters thin/odd books)
    max_price_c: int = 98
    allow_yes_favorites: bool = True  # YES bids on favorites (flow that hits them is more informed)
    allow_no_longshots: bool = True  # NO bids on longshots (sells lottery tickets to retail YES buyers)
    skip_hours_to_close: tuple = ()  # e.g. ((12, 24),) skips the target-day morning window
    momentum_guard_c: float = 0.0  # skip a bracket if its YES mid moved against our side by >= this many cents
    momentum_lookback_s: float = 3600.0
    decided_only: bool = False  # only quote outcomes with p_win >= decided_prob
    fav_day_before_only: bool = False  # if not decided_only: non-decided favorites only when > 24h to close
    fav_day_before_min_prob: float = 0.85
    # Reachability (Oct 7, first live day): a resting order fills only after every contract ahead of it trades (all
    # resting at a better price for the taker, then the queue already at ours). Above this many contracts ahead the
    # quote is skipped, and when the caps bind the budget goes to the quotes with the fewest ahead first. None = off.
    # On the first live afternoon Stage 1 parked its whole $75 behind 1c walls of 11,000-44,000 contracts; in 8 days of
    # tape (Sep 29 - Oct 6) only 9 YES sweeps that reached 3c had bought 1,000 or more contracts below it.
    max_contracts_ahead: Optional[float] = None
    # Keep resting orders (Oct 9, from the Oct 8 live log: when the caps bound and the books moved, the ranking cut a
    # resting order to 1 contract and later grew it back, 159 times in the first 16 hours, up to 11 times on one quote;
    # each growth is an amend up, which Kalshi treats as a new place at the back of the queue, and costs two writes).
    # On: resting orders rank first unless a new quote has fewer than keep_resting_ratio times their contracts ahead
    # (within an event here, across events in the runner's ranking), the risk filter keeps a resting order at the size
    # it rests at and shrinks it only as far as the caps need, and a size change of at most
    # max(keep_resting_frac x the resting size, keep_resting_contracts) is not made (wxbot/risk/limits.filter_orders,
    # `keep`). Off (the default): exactly as before. The default ratio 0.5 is today's order (a resting order already
    # counts half its contracts ahead); a lower ratio holds resting orders harder, which cost money in the Aug-Sep
    # replay (docs/results/r10_bundle_checks.txt, Part 2).
    keep_resting: bool = False
    keep_resting_ratio: float = 0.5
    keep_resting_frac: float = 0.25
    keep_resting_contracts: float = 2.0


def keep_rank(ahead: float, resting: bool, ratio: float) -> float:
    """The ranking value under keep_resting: a resting order by its contracts ahead (its own contracts left out), a
    new quote by its contracts ahead divided by the ratio, so a new quote ranks before a resting one only with fewer
    than `ratio` times the resting one's contracts ahead."""
    return float(ahead) if resting else float(ahead) / max(float(ratio), 1e-9)


def keep_deadband(resting_size: float, cfg: "SleeveAConfig") -> float:
    """Contracts by which a resting order's size may differ from what the ranking would give it and still be left
    alone (keep_resting)."""
    return max(float(cfg.keep_resting_frac) * float(resting_size), float(cfg.keep_resting_contracts))


def contracts_ahead(book: dict, side: str, price_cents: int, own: float = 0.0) -> float:
    """Contracts that must trade before an order of ours at this price fills: everything resting at a better price
    for the taker plus the queue at ours (raw orderbook_fp). Our 'ask' at YES price p rests among the NO bids at
    100 - p, a 'bid' among the YES bids at p; a higher bid is better for the taker in both books. own: the size of
    our own resting order at that price, which is not ahead of itself."""
    key, px = ("yes_dollars", price_cents) if side == "bid" else ("no_dollars", 100 - price_cents)
    ahead = 0.0
    for p, q in book.get(key) or []:
        if int(round(float(p) * 100)) >= px:
            ahead += float(q)
    return max(ahead - own, 0.0)


def _side_view(q: LadderQuote, side: str):
    """Return (best_bid_cost_c, best_ask_cost_c) in the cost units of the side we would BUY."""
    if side == "bid":  # buying YES
        return q.yes_bid, q.yes_ask
    # buying NO: NO bid = 100 - yes_ask ; NO ask = 100 - yes_bid
    nb = 100 - q.yes_ask if q.yes_ask is not None else None
    na = 100 - q.yes_bid if q.yes_bid is not None else None
    return nb, na


def reach(book: dict, side: str, price_cents: int, ticker: str, resting: dict | None = None) -> float:
    """Contracts ahead of our order at this price, as the ranking uses them. resting: {(ticker, side, price): size of
    ours there that the displayed book includes} for orders already resting (0 for paper orders, which are not in the
    book): such an order counts half its contracts ahead, so two similar markets do not trade places back and forth
    (each swap loses the queue spot) and a resting order stays until twice the limit."""
    r = resting or {}
    k = (ticker, side, price_cents)
    return contracts_ahead(book, side, price_cents, r.get(k, 0.0)) * (0.5 if k in r else 1.0)


def propose(ev: EventBook, fv: FairValueResult, now_ts: float, cfg: SleeveAConfig,
            decided_cost_used: float = 0.0, skip_tickers: frozenset = frozenset(),
            mid_history: dict | None = None, resting: dict | None = None,
            out_of_reach: list | None = None) -> tuple[list[DesiredOrder], list[str]]:
    """resting, out_of_reach: with cfg.max_contracts_ahead set, quotes with more contracts ahead (see reach) are left
    out and appended to out_of_reach as (ticker, contracts ahead), and the event's decided budget goes to the
    quotes with the fewest ahead first instead of in ladder order."""
    orders: list[DesiredOrder] = []
    notes: list[str] = []
    mins_left = (ev.close_ts - now_ts) / 60.0
    if mins_left < cfg.stop_minutes_before_close:
        return [], [f"{ev.event}: inside close buffer"]
    hrs_left = mins_left / 60.0
    for lo, hi in cfg.skip_hours_to_close or ():
        if lo <= hrs_left < hi:
            return [], [f"{ev.event}: skip window {lo}-{hi}h to close"]
    if getattr(fv, "conflict", False):
        return [], [f"{ev.event}: observation/market conflict, waiting"]
    if fv.disagreement > cfg.max_disagreement:
        return [], [f"{ev.event}: market/NBM disagreement {fv.disagreement:.2f} > {cfg.max_disagreement}"]
    if fv.notes:
        notes.extend(f"{ev.event}: {n}" for n in fv.notes)
    decided_budget = max(cfg.decided_max_cost_usd - decided_cost_used, 0.0)
    cands = []
    keep_key = {}       # keep_resting: ticker -> its ranking value (keep_rank), used instead of the plain reach
    for b in ev.brackets:
        p_yes = fv.probs.get(b.ticker)
        q = ev.quotes.get(b.ticker)
        if p_yes is None or q is None or b.ticker in skip_tickers:
            continue
        if p_yes >= cfg.fav_min_prob and cfg.allow_yes_favorites:
            side, p_win = "bid", p_yes
        elif 1 - p_yes >= cfg.fav_min_prob and cfg.allow_no_longshots:
            side, p_win = "ask", 1 - p_yes
        else:
            continue  # mid-priced (or disabled side): no quote
        if cfg.decided_only and p_win < cfg.decided_prob:
            continue
        if cfg.fav_day_before_only and p_win < cfg.decided_prob:
            if hrs_left <= 24 or p_win < cfg.fav_day_before_min_prob:
                continue
        if cfg.momentum_guard_c > 0 and mid_history:
            hist = mid_history.get(b.ticker) or []
            past = [m for t, m in hist if now_ts - t >= cfg.momentum_lookback_s * 0.5 and now_ts - t <= cfg.momentum_lookback_s * 1.5]
            cur = q.mid()
            if past and cur is not None:
                move_c = (cur - past[0]) * 100.0  # + means YES got more likely
                against = move_c if side == "ask" else -move_c
                if against >= cfg.momentum_guard_c:
                    continue
        best_bid_c, best_ask_c = _side_view(q, side)
        fair_c = p_win * 100.0
        max_c = math.floor(fair_c - cfg.min_edge_c)
        if best_ask_c is not None:
            max_c = min(max_c, best_ask_c - 1)  # post-only: never cross
        max_c = min(max_c, cfg.max_price_c)
        if max_c < cfg.min_price_c:
            continue
        if best_bid_c is None:
            price_c = max_c
        elif best_bid_c + 1 <= max_c and fair_c - (best_bid_c + 1) >= cfg.improve_if_edge_c:
            price_c = best_bid_c + 1  # step inside the spread for queue priority
        else:
            price_c = min(best_bid_c, max_c)  # join the best bid, or sit below it
        decided = p_win >= cfg.decided_prob
        ahead = 0.0
        if cfg.max_contracts_ahead is not None:
            yes_px = int(price_c if side == "bid" else 100 - price_c)
            book = ev.books.get(b.ticker)
            ahead = (reach(book, side, yes_px, b.ticker, resting) if book is not None
                     else float(cfg.max_contracts_ahead))
            if ahead > cfg.max_contracts_ahead:
                if out_of_reach is not None:
                    out_of_reach.append((b.ticker, int(ahead)))
                continue
            if cfg.keep_resting:
                k = (b.ticker, side, yes_px)
                rest_ = resting or {}
                if book is not None:
                    raw = contracts_ahead(book, side, yes_px, rest_.get(k, 0.0))
                else:           # no book: last, as above (a resting order's limit is twice the cutoff)
                    raw = float(cfg.max_contracts_ahead) * (2.0 if k in rest_ else 1.0)
                keep_key[b.ticker] = keep_rank(raw, k in rest_, cfg.keep_resting_ratio)
        cands.append((ahead, b, side, p_win, p_yes, fair_c, price_c, decided))
    if cfg.max_contracts_ahead is not None:
        if cfg.keep_resting:
            cands.sort(key=lambda c: keep_key[c[1].ticker])     # stable: ladder order among equals
        else:
            cands.sort(key=lambda c: c[0])          # stable: ladder order among equals
    for ahead, b, side, p_win, p_yes, fair_c, price_c, decided in cands:
        cost_c = price_c
        count = cfg.clip_contracts
        if decided:
            if decided_budget <= 0:
                continue
            count = min(count, math.floor(decided_budget / (cost_c / 100.0)))
            if count < 1:
                continue
            decided_budget -= count * cost_c / 100.0
        yes_price = price_c if side == "bid" else 100 - price_c
        orders.append(DesiredOrder(
            ticker=b.ticker, side=side, price_cents=int(yes_price), count=float(count), sleeve="A",
            fair_yes=p_yes, event=ev.event,
            reason=f"fav {'YES' if side == 'bid' else 'NO'} p={p_win:.3f} pay={cost_c}c edge={fair_c - cost_c:.1f}c"
                   + (" decided" if decided else ""),
        ))
    return orders, notes
