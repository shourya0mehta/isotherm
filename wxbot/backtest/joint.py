"""Joint, capital-constrained replay: all cities on one clock with the live risk caps.

Unlike sim.run (events replayed independently), this enforces the account-level limits the live bot uses:
max total open risk, per-event and per-bracket caps, and capital stays locked until settlement
(close + `settle_lag_h`). Output is a daily equity curve for a given bankroll.
"""
from __future__ import annotations

import bisect
import collections

import numpy as np
import datetime as dt
import json
import os

from ..execution.live import is_patient
from ..execution.paper import PaperBroker
from ..timeutil import parse_ts
from ..models import fair_value as fvmod
from ..models.brackets import bracket_from_market
from ..risk.limits import RiskConfig, cash_in_use, event_worst_case, filter_orders
from ..strategy import sleeve_a
from ..strategy.types import EventBook, Position


def pdt(s):
    return parse_ts(s)


def combine_daily(*dailies) -> dict:
    """Add several runs' daily P&L dicts ({UTC day: $}) into one, day by day (runs can share a day: a month's last
    events settle on the next month's first day)."""
    out = collections.defaultdict(float)
    for d in dailies:
        for k, v in d.items():
            out[k] += v
    return dict(out)


def daily_ci(daily: dict, n: int = 4000):
    """Mean daily P&L with a 95% bootstrap interval over days (n draws of whole days, seed 0). Moved here unchanged
    from research/ml/fill_stress.py on Oct 9 (which still exports it), so scorers can use it without that module's
    research-data imports."""
    v = np.array(list(daily.values()), float)
    if len(v) < 3:
        return float(v.mean()) if len(v) else 0.0, float("nan"), float("nan")
    rng = np.random.default_rng(0)
    bs = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def max_drawdown(daily: dict) -> float:
    """Largest fall of cumulative P&L below its running peak (the peak starts at $0), over the days in order. For
    several runs, combine their daily dicts first (combine_daily): the larger of two runs' own drawdowns is not the
    drawdown of the two together, since a loss at the end of one run and the start of the next add up (outside
    review, Oct 8)."""
    eq = peak = mdd = 0.0
    for d in sorted(daily):
        eq += daily[d]
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    return mdd


def _settle_due(events, positions, t, settle_lag_h, by_event, settled_pos, daily, force=False):
    """Settle every unsettled event whose close + settle_lag_h is at or before t (all of them with force=True): pay
    each held contract $1 on the winning side, book the event's P&L on day t (UTC), together with what the event's
    sales before settlement made (run_joint exit_fn; 0 without it). Returns the P&L settled."""
    total = 0.0
    for name, e in events.items():
        if e["settled"] or not (force or t >= e["close"] + settle_lag_h * 3600):
            continue
        e["settled"] = True
        pnl = 0.0
        held = []
        for tk, r in e["result"].items():
            p = positions.pop(tk, None)
            if p:
                pnl += (p.yes_qty if r == "yes" else 0) + (p.no_qty if r == "no" else 0) - p.yes_cost - p.no_cost
                held.append([tk, r, p.yes_qty, round(p.yes_cost, 4), p.no_qty, round(p.no_cost, 4)])
        pnl += e.get("exit_pnl", 0.0)
        total += pnl
        by_event[name] = pnl
        if held:
            settled_pos[name] = held
        day = dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%d")
        daily[day] += pnl
    return total


def _exits(exit_fn, name, t, quotes, fv, e, positions, held, exit_log):
    """run_joint's exit_fn: close what the hook sells, at its price, and keep the P&L for the event's settlement."""
    for tk, side, cnt, px_c, fee_usd in exit_fn(name, t, quotes, fv, held) or ():
        p = positions.get(tk)
        if p is None or cnt <= 0:
            continue
        qty, cost = (p.yes_qty, p.yes_cost) if side == "yes" else (p.no_qty, p.no_cost)
        q = min(float(cnt), qty)
        if q <= 1e-9:
            continue
        c = cost * q / qty
        if side == "yes":
            p.yes_qty, p.yes_cost = qty - q, cost - c
        else:
            p.no_qty, p.no_cost = qty - q, cost - c
        f = fee_usd * q / float(cnt)
        pnl = q * px_c / 100.0 - c - f
        e["exit_pnl"] = e.get("exit_pnl", 0.0) + pnl
        if exit_log is not None:
            exit_log.append({"ts": t, "event": name, "ticker": tk, "side": side, "count": q, "price_cents": px_c,
                             "cost": c, "fee": f, "pnl": pnl})


def run_joint(ev_files, cd_dir, *, cal_map=None, cfg_a=None, risk=None, step_s=300, queue_ahead=300.0,
              settle_lag_h=8.0, verbose=False, cash_cap=None, netting=False, fv_fn=None, fill_rule="queue",
              cfg_fn=None, patient_s=0.0, resize=True, topup=True, guard_fn=None, fill_log=None, exit_fn=None,
              exit_log=None, grow_amend=False, keep_first=True, keep_inherit=False):
    """cash_cap: max cash locked by positions + resting orders (None = only the worst-case risk caps).
    cfg_fn: optional hook cfg_fn(event_name, t, quotes) -> SleeveAConfig (per-event, per-moment quoting rules).
    fill_rule: "queue" = a join waits behind queue_ahead contracts, an improvement is first in line;
               "through" = fills only from trades at a strictly better price than ours (stress test: certain fills).
    netting: True = Kalshi MECNET collateral netting (cash locked = per-event worst case), False = full cost.
    patient_s: as the live bot's execution.patient_reprice_s: a resting order is not moved to a more aggressive price
               until it is that old (moves to a more cautious price are immediate); 0 = re-price at every step.
    resize: as live (Oct 4 review, finding 2.11): a resting order larger than the size the caps now allow is amended
            down (newest first), and one mostly filled is topped up behind the queue. False = the pre-Oct-4 replay,
            which placed only when nothing rested and never shrank an order, so it could keep filling past caps that
            had tightened.
    fv_fn: optional hook fv_fn(event_name, t, brackets, quotes, fv) -> FairValueResult (research fair values).
    guard_fn: optional hook guard_fn(event_name, t) -> set of tickers Sleeve A must leave alone at t (the rounding
              guard, wxbot/models/round_guard.py). They go to sleeve_a.propose as skip_tickers at every step, and a
              resting order on one counts as cancelled from the moment the hook names it: a trade printed then cannot
              fill it (the live bot cancels within its loop once the reading arrives). None = off (no change).
    fill_log: optional list; each fill is appended as a dict (ts, event, ticker, side, price_cents in YES cents,
              count, fair_yes of the order). None = off.
    exit_fn: optional hook exit_fn(event_name, t, quotes, fv, positions) -> [(ticker, side 'yes'|'no', count,
             price_cents of the side sold, fee in dollars for the count)]: sales before settlement (research on exit
             rules, docs/results/r8_exit_rules.txt). Called at every step for each event in its quoting window, after
             its fair value and before Sleeve A quotes; positions = that event's Position objects. A sale closes up to
             the count held at that price (the cost leaves at the position's average), frees its cash and caps from
             the next step, and its P&L (proceeds - cost - fee) is booked with the event's at settlement, on the same
             day. None = off (no change).
    exit_log: optional list; each sale is appended as a dict (ts, event, ticker, side, count, price_cents, cost, fee,
              pnl). None = off.
    cfg_a.keep_resting: the risk filter keeps resting orders as the live bot does with sleeve_a.keep_resting
              (filter_orders `keep`: resting first, at the size they rest at, small size changes not made). The
              replay has no order book depth, so the ranking by contracts ahead (and the ratio that lets a far more
              reachable quote go first) does not apply here: resting orders always go first. False = no change.
              Research variants (keep_resting on): keep_first=False keeps the replay's own order (by edge; the live
              analog is keep_resting_ratio 0.5, which is today's live order) and only keeps sizes; keep_inherit=True
              lets a quote that moved to a new price keep the place and size of the order it replaces (same market
              and side).
    grow_amend: growth as the live order manager does it (wxbot/execution/live.py, since Oct 6): when less than half
              the wanted size rests and what rests is not mostly filled, the newest order is amended up and the whole
              order goes to the back of its price's queue (Kalshi treats an amend up as a new place in line). False =
              the replay's top-up, a second order behind the queue while the first keeps its place (no change).
    The result also counts fills (fill events), shrinks (amends down), shrinks_to_1 (to 1 contract or less), grows
    (top-ups or amends up), regrows (grows of a market and side shrunk since its last fresh placement) and replaced
    (fresh placements at a price where an order of ours was cancelled as no longer wanted, same event)."""
    cfg_a = cfg_a or sleeve_a.SleeveAConfig()
    risk = risk or RiskConfig()
    topup_on = topup
    fvmod._CAL = dict(cal_map) if cal_map else {}
    events = {}
    trades = []
    for fn in ev_files:
        name = os.path.basename(fn)[:-5]
        ms = [m for m in json.load(open(fn)) if m.get("result") in ("yes", "no")]
        if len(ms) < 3:
            continue
        try:
            brs = [bracket_from_market(m) for m in ms]
        except ValueError:
            continue
        candles = {}
        for m in ms:
            cf = os.path.join(cd_dir, f"{m['ticker']}.json")
            c = sorted(json.load(open(cf))) if os.path.exists(cf) else []
            # compact storage: a month of 1-minute candles as Python lists took ~3.5 GB
            arr = np.array([x[:3] for x in c], dtype=float).reshape(-1, 3)
            candles[m["ticker"]] = (arr[:, 0].copy(), arr)
        o = min(pdt(m["open_time"]) for m in ms)
        cl = max(pdt(m["close_time"]) for m in ms)
        series = name.split("-")[0]
        ev = EventBook(name, series, "high", "", None, cl, ms, brs, {}, {})
        events[name] = {"ev": ev, "open": o, "close": cl, "candles": candles,
                        "result": {m["ticker"]: m["result"] for m in ms}, "settled": False}
        for m in ms:
            for t in m.pop("trades", None) or []:  # keep only what the paper broker reads (memory)
                trades.append((pdt(t["created_time"]), name,
                               {"ticker": t.get("ticker", m["ticker"]), "created_time": t["created_time"],
                                "yes_price_dollars": t["yes_price_dollars"], "count_fp": t["count_fp"],
                                "taker_side": t.get("taker_side") or t.get("taker_outcome_side")}))
    trades.sort(key=lambda x: x[0])
    t0 = min(e["open"] for e in events.values())
    t1 = max(e["close"] for e in events.values()) + settle_lag_h * 3600
    pb = PaperBroker()
    positions: dict[str, Position] = {}
    realized = 0.0
    daily = collections.defaultdict(float)
    by_event = {}
    settled_pos = {}  # event -> [[ticker, result, yes_qty, yes_cost, no_qty, no_cost], ...] at settlement
    ti = 0
    t = t0
    max_locked = 0.0
    max_cash = 0.0
    contracts = 0.0
    churn = {"fills": 0, "shrinks": 0, "shrinks_to_1": 0, "grows": 0, "regrows": 0, "replaced": 0}
    shrunk = set()        # (ticker, side) shrunk since its last fresh placement
    unwanted = set()      # (ticker, side, price) cancelled as no longer wanted
    peak_size = {}        # paper order id -> the largest size it had (judges "mostly filled" as the live manager)
    keep_on = bool(getattr(cfg_a, "keep_resting", False))
    while t < t1:
        # fills up to t
        while ti < len(trades) and trades[ti][0] <= t:
            if guard_fn is not None:
                tk_ = trades[ti][2]["ticker"]
                rest_ = pb.resting_for(tk_)
                if rest_ and tk_ in guard_fn(trades[ti][1], trades[ti][0]):
                    for po in rest_:
                        pb.cancel(po.oid)
            for f in pb.on_trade(trades[ti][2]):
                p = positions.setdefault(f.ticker, Position(f.ticker, trades[ti][1]))
                if f.side == "bid":
                    p.yes_qty += f.count
                    p.yes_cost += f.count * f.price_cents / 100
                else:
                    p.no_qty += f.count
                    p.no_cost += f.count * (100 - f.price_cents) / 100
                contracts += f.count
                churn["fills"] += 1
                if fill_log is not None:
                    o_ = pb.orders.get(f.order_ref)
                    fill_log.append({"ts": f.ts, "event": trades[ti][1], "ticker": f.ticker, "side": f.side,
                                     "price_cents": f.price_cents, "count": f.count,
                                     "fair_yes": o_.fair_yes if o_ is not None else None})
            ti += 1
        # settlements (capital release)
        realized += _settle_due(events, positions, t, settle_lag_h, by_event, settled_pos, daily)
        # quoting for open events
        pos_by_ev = collections.defaultdict(list)
        for p in positions.values():
            pos_by_ev[p.event].append(p)
        tick_by_ev = {n: [m["ticker"] for m in e["ev"].markets] for n, e in events.items()}
        locked = sum(event_worst_case(pos_by_ev[n], [], tick_by_ev[n]) for n in pos_by_ev)
        max_locked = max(max_locked, locked)
        cash_now = cash_in_use(list(positions.values()), tick_by_ev, netting)
        max_cash = max(max_cash, cash_now)
        desired = []
        for name, e in events.items():
            if not (e["open"] + 600 <= t < e["close"] - 60):
                continue
            quotes = {}
            for m in e["ev"].markets:
                ts_l, c = e["candles"][m["ticker"]]
                i = int(np.searchsorted(ts_l, t, side="right")) - 1
                if i < 0:
                    continue
                _, b, a = c[i]
                quotes[m["ticker"]] = fvmod.LadderQuote(int(round(b * 100)) if b > 0 else None,
                                                        int(round(a * 100)) if 0 < a < 1 else None)
            if len(quotes) < len(e["ev"].markets) - 1:
                continue
            e["ev"].quotes = quotes
            fv = fvmod.ladder_fair_value(e["ev"].brackets, quotes, kind="high", hours_to_close=(e["close"] - t) / 3600)
            if fv_fn is not None:
                fv = fv_fn(name, t, e["ev"].brackets, quotes, fv)
                if fv is None:
                    continue
            if exit_fn is not None:
                _exits(exit_fn, name, t, quotes, fv, e, positions, pos_by_ev.get(name, []), exit_log)
            cfg = cfg_fn(name, t, quotes) if cfg_fn is not None else cfg_a
            skip = frozenset(guard_fn(name, t)) if guard_fn is not None else frozenset()
            d, _ = sleeve_a.propose(e["ev"], fv, t, cfg, 0.0, skip_tickers=skip)
            desired += d
        if keep_on:
            sizes = collections.defaultdict(float)
            for po in pb.resting():
                sizes[(po.ticker, po.side, po.price_cents)] += po.remaining
            keep = {k: (r, sleeve_a.keep_deadband(r, cfg_a)) for k, r in sizes.items()}
            if keep_inherit:
                by_mkt = {(tk, sd): v for (tk, sd, _), v in keep.items()}
                for d in desired:
                    if (d.ticker, d.side, d.price_cents) not in keep and (d.ticker, d.side) in by_mkt:
                        keep[(d.ticker, d.side, d.price_cents)] = by_mkt[(d.ticker, d.side)]
            kept, _ = filter_orders(desired, pos_by_ev, tick_by_ev, risk, locked, cash_used=cash_now,
                                    cash_cap=cash_cap, netting=netting,
                                    incumbents=set(keep) if keep_first else None, keep=keep)
        else:
            kept, _ = filter_orders(desired, pos_by_ev, tick_by_ev, risk, locked, cash_used=cash_now,
                                    cash_cap=cash_cap, netting=netting)
        want = {d.key: d for d in kept}
        by_key = collections.defaultdict(list)
        for po in pb.resting():
            by_key[(po.ticker, po.side)].append(po)
        held = set()
        for k, lst in by_key.items():
            d = want.get(k)
            stale = [po for po in lst if d is not None and d.price_cents != po.price_cents]
            if stale and is_patient(d, stale, t, patient_s, ts_attr="placed_ts"):
                held.add(k)
                continue
            for po in lst:
                if d is None or d.price_cents != po.price_cents:
                    pb.cancel(po.oid)
                    if d is None:
                        unwanted.add((po.ticker, po.side, po.price_cents))
        for k, d in want.items():
            if k in held:
                continue
            lst = [po for po in pb.resting_for(d.ticker) if po.side == d.side]
            have = sum(po.remaining for po in lst)
            topup = resize and topup_on and 0.5 < have < 0.5 * d.count
            if topup:
                churn["grows"] += 1
                churn["regrows"] += int(k in shrunk)
            if topup and grow_amend and not all(po.filled >= 0.5 * max(peak_size.get(po.oid, 0.0),
                                                                       po.filled + po.remaining) for po in lst):
                # the live manager: one order per market and side, grown by an amend that sends the whole order to
                # the back of its price's queue
                newest = max(lst, key=lambda po: po.placed_ts)
                newest.remaining += d.count - have
                peak_size[newest.oid] = max(peak_size.get(newest.oid, 0.0), newest.filled + newest.remaining)
                newest.queue_ahead = float("inf") if fill_rule == "through" else queue_ahead
                newest.placed_ts = t
                continue
            if have <= 0.5 or topup:
                q = events[d.event]["ev"].quotes.get(d.ticker)
                best = None
                if q:
                    best = q.yes_bid if d.side == "bid" else (100 - q.yes_ask if q.yes_ask is not None else None)
                my_cost = d.price_cents if d.side == "bid" else 100 - d.price_cents
                dd = d if not topup else type(d)(**{**d.__dict__, "count": d.count - have})
                po = pb.place(dd, {}, now=t)
                peak_size[po.oid] = po.remaining
                if have <= 0.5:
                    shrunk.discard(k)
                    pk = (d.ticker, d.side, d.price_cents)
                    if pk in unwanted:
                        churn["replaced"] += 1
                        unwanted.discard(pk)
                # a top-up waits behind everything at our price, our own older order included
                po.queue_ahead = queue_ahead if topup else (0.0 if (best is None or my_cost > best) else queue_ahead)
                if fill_rule == "through":
                    po.queue_ahead = float("inf")
            elif resize and have > d.count + 0.5:
                newest = max(lst, key=lambda po: po.placed_ts)
                pb.amend_size(newest.oid, max(newest.remaining - (have - d.count), 0.0))
                churn["shrinks"] += 1
                churn["shrinks_to_1"] += int(sum(po.remaining for po in lst) <= 1.0)   # what rests after the amend
                shrunk.add(k)
        t += step_s
    # Oct 8 review fix: the loop stops before t1, so the events closing last (each run's final day) were never settled
    # and their fills' cost and payout were both left out. Settle whatever is still open at t1.
    realized += _settle_due(events, positions, t1, settle_lag_h, by_event, settled_pos, daily, force=True)
    mdd = max_drawdown(daily)
    return {"realized": realized, "daily": dict(daily), "by_event": by_event, "positions": settled_pos, "max_dd": mdd, "max_locked": max_locked, "max_cash": max_cash,
            "contracts": contracts, "days": len(daily),
            "up_days": sum(1 for v in daily.values() if v > 0), "down_days": sum(1 for v in daily.values() if v < 0),
            **churn}
