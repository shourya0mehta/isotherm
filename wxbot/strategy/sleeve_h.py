"""Sleeve H: post-only maker on the tails of Kalshi's hourly Weather Index markets (KXTEMP*H*).

Research: research/hourly (README.md, maker_replay.py). The hourly ladders show the same favorite-longshot bias as the
daily ones: strikes priced 90-99c win more often than priced and strikes priced 2-10c win less often. This sleeve
quotes only those tails, one cent inside the touch when the spread allows, never crossing:
  best YES bid >= 1 - a_max  ->  bid YES at min(max_price, best bid + 1c, best ask - 1c)
  best YES ask <= a_max      ->  ask YES at max(min_price, best ask - 1c, best bid + 1c)
At most `size` contracts filled per strike and side per hour, a worst-case loss cap per hourly event over the ladder's outcome scenarios (filled
plus resting), and no quoting in the last stop_min minutes. It reads only the order book: no forecast, so it needs no weather data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .types import DesiredOrder, EventBook


@dataclass
class SleeveHConfig:
    enabled: bool = False          # send live orders (off until the shadow record and a holdout pass)
    shadow: bool = True            # simulate on the live tape with a separate paper broker (table hourly_shadow)
    series: tuple = ("KXTEMPMIAH", "KXTEMPNYCHS", "KXTEMPCHIHS", "KXTEMPLAXHS")
    a_max: float = 0.10            # quote a strike's tail side only when the touch is this close to 0 or 1
    size: int = 20                 # contracts per strike and side
    cap_usd: float = 25.0          # worst-case loss per hourly event, filled plus resting
    stop_min: float = 0.0          # stop quoting this many minutes before the settlement minute
    min_price_c: int = 2
    max_price_c: int = 98
    nowcast: bool = False          # also run variant H2: quote a tail side only if the nowcast agrees (shadow)
    p_lo: float = 0.02             # H2: nowcast P(yes) <= p_lo to offer YES, >= 1 - p_lo to bid YES
    edge: float = 0.02             # H2: quotes stay at least this far from the nowcast probability
    total_risk_usd: float = 50.0   # live/paper orders only: worst case across all open hourly events
    max_cash_usd: float = 150.0    # live/paper orders only: cash locked by hourly positions and resting orders
    # breaker (real orders only): once breaker_min_contracts real hourly contracts have settled, switch the real
    # orders off for good if more than breaker_max_wrong of them lost or the average P&L per contract is below
    # breaker_min_pnl_c cents. Cleared only by hand (python -m wxbot reset-h-breaker). The shadow keeps logging.
    # 3,000: losses come in bunches, so fewer contracts mostly measure luck (docs/results/hourly_breaker_sim.txt:
    # at 300 a September-quality strategy is switched off about half the time on day one; at 3,000 about 2% of the
    # time in four weeks, while a strategy losing about 1c a contract is caught within a week 98% of the time).
    breaker_min_contracts: int = 3000
    breaker_max_wrong: float = 0.06
    breaker_min_pnl_c: float = 0.0
    stale_book_s: float = 45.0     # real orders only: no hourly quotes when the order books are older than this
    # which tail sides to quote: "ask" offers YES at the low tail (a NO bid near 100c), "bid" bids YES at the high
    # tail. Both as frozen Oct 3; the ask-only variant is the Dec 1 secondary (HOLDOUT_DEC1.md amendment A7)
    sides: tuple = ("bid", "ask")

    @classmethod
    def from_cfg(cls, d: dict | None) -> "SleeveHConfig":
        d = dict(d or {})
        if "series" in d:
            d["series"] = tuple(d["series"])
        if "sides" in d:
            d["sides"] = (d["sides"],) if isinstance(d["sides"], str) else tuple(d["sides"])
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def breaker_stats(fills, results: dict, series, since: float = 0.0) -> dict:
    """Settled real contracts on the hourly series since `since`.

    fills: Store.fills rows (ts, mode, ticker, event, side, price_cents, count, sleeve, is_taker, fee, ...);
    results: {ticker: "yes" | "no"}. A YES bid (side "bid") wins on "yes"; an ask (long NO) wins on "no"."""
    hs = set(series)
    n = lost = pnl = 0.0
    for r in fills:
        ts, t, side, pc, cnt, fee = r[0], r[2], r[4], r[5], r[6], r[9]
        res = results.get(t)
        if ts < since or t.split("-")[0] not in hs or res not in ("yes", "no"):
            continue
        won = (res == "yes") == (side == "bid")
        cost = pc / 100.0 if side == "bid" else (100 - pc) / 100.0
        n += cnt
        lost += 0.0 if won else cnt
        pnl += cnt * ((1.0 if won else 0.0) - cost) - (fee or 0.0)
    return {"contracts": round(n, 2), "lost": round(lost, 2), "lost_share": round(lost / n, 4) if n else None,
            "pnl_usd": round(pnl, 2), "pnl_c_per_contract": round(100 * pnl / n, 2) if n else None}


def breaker_verdict(stats: dict, cfg: SleeveHConfig):
    """None while fewer than cfg.breaker_min_contracts real contracts have settled or both tests pass; otherwise
    the reason to switch the real hourly orders off."""
    n = stats.get("contracts") or 0
    if n < cfg.breaker_min_contracts:
        return None
    why = []
    if stats["lost_share"] > cfg.breaker_max_wrong:
        why.append(f"{100 * stats['lost_share']:.1f}% of {n:.0f} settled contracts lost "
                   f"(limit {100 * cfg.breaker_max_wrong:.0f}%)")
    if stats["pnl_c_per_contract"] < cfg.breaker_min_pnl_c:
        why.append(f"P&L {stats['pnl_c_per_contract']:+.2f}c per contract (limit {cfg.breaker_min_pnl_c:+.1f}c)")
    return "; ".join(why) or None


def unit_loss(side: str, price_cents: int) -> float:
    """Worst-case loss per contract in dollars: a YES bid at p loses p; a YES ask at p (long NO) loses 1 - p."""
    return price_cents / 100.0 if side == "bid" else (100 - price_cents) / 100.0


def _mid(bid, ask, px) -> float:
    if bid is not None and ask is not None:
        return (bid + ask) / 200.0
    return px / 100.0


def strike_of(m: dict):
    """The threshold of an 'above k' market (strike_type greater); None for any other kind."""
    if m.get("strike_type") != "greater" or m.get("floor_strike") is None:
        return None
    return float(m["floor_strike"])


def scenario_losses(strikes: list, side: str, price_cents: int, k: float) -> list:
    """Loss per contract (dollars, negative = gain) in each outcome scenario of a threshold ladder.
    Scenario j: the index ends above the j lowest strikes and at or below the rest (j = 0..n)."""
    p = price_cents / 100.0
    out = []
    for j in range(len(strikes) + 1):
        yes_wins = k in strikes[:j]
        if side == "bid":
            out.append(-(1 - p) if yes_wins else p)
        else:
            out.append((1 - p) if yes_wins else -p)
    return out


def propose(ev: EventBook, now: float, cfg: SleeveHConfig, held: list | None = None,
            p_of: dict | None = None, adverse: bool = True, filled: dict | None = None) -> list[DesiredOrder]:
    """Desired tail quotes for one hourly event.
    held: this event's filled shadow (or live) contracts as (ticker, side, price_cents, count). The worst case over
    the ladder's outcome scenarios stays within cap_usd: positions exactly, plus each resting quote where it loses.
    adverse=False is the pre-Oct-4 reading (every quote filling together, so a quote's gain in other scenarios
    widened the room for the next one); the pre-registered shadow keeps it so its record stays comparable.
    filled: gross contracts filled this hour per (ticker, side) for the `size` limit (live: from the fill feed, since
    net positions hide a YES and a NO fill on the same strike); default: from `held`.
    p_of: nowcast P(yes) per ticker (variant H2); a strike without one is not quoted when p_of is given."""
    if ev.close_ts is None or now >= ev.close_ts - cfg.stop_min * 60:
        return []
    k_of = {m["ticker"]: strike_of(m) for m in ev.markets}
    strikes = sorted(k for k in k_of.values() if k is not None)
    if not strikes:
        return []
    L = [0.0] * (len(strikes) + 1)
    gross = {}
    for t, side, px, n in held or []:
        if k_of.get(t) is None:
            continue
        gross[(t, side)] = gross.get((t, side), 0.0) + n
        for j, l in enumerate(scenario_losses(strikes, side, px, k_of[t])):
            L[j] += n * l
    if filled is None:
        filled = gross
    cands = []
    lo_c, hi_c = int(round((1 - cfg.a_max) * 100)), int(round(cfg.a_max * 100))
    for m in ev.markets:
        t = m["ticker"]
        q = ev.quotes.get(t)
        if q is None or k_of.get(t) is None:
            continue
        bid, ask = q.yes_bid, q.yes_ask   # cents or None
        p = None
        if p_of is not None:
            p = p_of.get(t)
            if p is None:
                continue
        if bid is not None and bid >= lo_c and (p is None or p >= 1 - cfg.p_lo):
            px = min(cfg.max_price_c, bid + 1, (ask - 1) if ask is not None else 99)
            if p is not None:
                px = min(px, int(math.floor((p - cfg.edge) * 100 + 1e-9)))
            if px >= lo_c - 1:
                cands.append((t, "bid", int(px), bid, ask))
        if ask is not None and ask <= hi_c and (p is None or p <= cfg.p_lo):
            px = max(cfg.min_price_c, ask - 1, (bid + 1) if bid is not None else 1)
            if p is not None:
                px = max(px, int(math.ceil((p + cfg.edge) * 100 - 1e-9)))
            if px <= hi_c + 1:
                cands.append((t, "ask", int(px), bid, ask))
    # most certain first: the deepest tails use the least worst-case room per cent of edge
    cands.sort(key=lambda c: (100 - c[2]) if c[1] == "bid" else c[2])
    out: list[DesiredOrder] = []
    for t, side, px, bid, ask in cands:
        if side not in cfg.sides:
            continue
        l = scenario_losses(strikes, side, px, k_of[t])
        n_max = int(cfg.size - filled.get((t, side), 0.0))   # at most `size` filled per strike and side per hour
        for j in range(len(L)):
            if l[j] > 1e-12:
                n_max = min(n_max, int(max(cfg.cap_usd - L[j], 0.0) // l[j]))
        if n_max <= 0:
            continue
        for j in range(len(L)):
            L[j] += n_max * (max(l[j], 0.0) if adverse else l[j])
        out.append(DesiredOrder(ticker=t, side=side, price_cents=px, count=n_max, sleeve="H",
                                fair_yes=_mid(bid, ask, px), reason=f"tail {side} (book {bid}/{ask})", event=ev.event))
    return out
