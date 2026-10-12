"""Walk-forward replay of Sleeve A on recorded/public Kalshi data (trade tape + 1-min bid/ask candles).

Honest-by-construction choices
- Calibration map is fit only on days BEFORE the test window (fit_calibration on the train window).
- No NBM / METAR in the replay (no historical depth of those feeds wired in yet) -> market-anchored fair value only.
- Queue position is unknown historically: every join sits behind `queue_ahead` contracts (parameter); a 1c
  improvement starts at the front of a new level. Fills come only from public trades after our placement.
- Positions are held to settlement and paid on the exchange's actual result. Maker fee = 0 (weather).
"""
from __future__ import annotations

import bisect
import collections
import datetime as dt
import glob
import json
import math
import os
from dataclasses import dataclass

from ..execution.paper import PaperBroker
from ..timeutil import parse_dt
from ..models import fair_value as fvmod
from ..models.brackets import bracket_from_market
from ..risk.limits import RiskConfig, event_worst_case, filter_orders
from ..strategy import sleeve_a
from ..strategy.types import DesiredOrder, EventBook, Position


def pdt(s):
    return parse_dt(s)


def fit_calibration(ev_files, cd_dir, lam=0.5):
    """Isotonic map mid -> realized YES freq from train events; returns {0.01..0.99: value}."""
    pairs = []
    for fn in ev_files:
        for m in json.load(open(fn)):
            r = m.get("result")
            cf = os.path.join(cd_dir, f"{m['ticker']}.json")
            if r not in ("yes", "no") or not os.path.exists(cf):
                continue
            c = sorted(json.load(open(cf)))
            if not c:
                continue
            ts = [x[0] for x in c]
            o, cl = pdt(m["open_time"]).timestamp(), pdt(m["close_time"]).timestamp()
            t = o + 1800
            while t < cl - 60:
                i = bisect.bisect_right(ts, t) - 1
                if i >= 0:
                    _, b, a = c[i]
                    a = 1.0 if a <= 0 else a
                    if a >= b and not (b == 0 and a >= 1.0) and a - b <= 0.10:
                        pairs.append(((a + b) / 2, 1.0 if r == "yes" else 0.0))
                t += 1800
    pairs.sort()
    blocks = []
    for x, y in pairs:
        blocks.append([y, 1, x, x])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, n, lo, hi = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += n
            blocks[-1][3] = hi

    def iso(x):
        for s, n, lo, hi in blocks:
            if x <= hi:
                return s / n
        return blocks[-1][0] / blocks[-1][1] if blocks else x

    grid = [round(0.01 * k, 2) for k in range(1, 100)]
    raw = [iso(g) for g in grid]
    sm = []
    for i in range(len(grid)):
        w = raw[max(0, i - 3):i + 4]
        sm.append(sum(w) / len(w))
    for i in range(1, len(sm)):
        sm[i] = max(sm[i], sm[i - 1])
    return {g: g + lam * (s - g) for g, s in zip(grid, sm)}, len(pairs)


@dataclass
class SimResult:
    fills: int
    contracts: float
    cost: float
    pnl: float
    by_event: dict
    daily: dict
    max_dd: float
    fill_log: list = None  # (ts, event, ticker, side, price_cents, count, result, hours_to_close)


def run(ev_files, cd_dir, *, cal_map=None, step_s=300, queue_ahead=300.0, cfg_a=None, risk=None,
        series_allow=None, rearm_cooldown_s=0.0) -> SimResult:
    cfg_a = cfg_a or sleeve_a.SleeveAConfig()
    risk = risk or RiskConfig()
    if cal_map is not None:
        fvmod._CAL = dict(cal_map)
    else:
        fvmod._CAL = {}
    total = collections.defaultdict(float)
    by_event = {}
    daily = collections.defaultdict(float)
    fills_n = 0
    contracts = 0.0
    cost_tot = 0.0
    fill_log = []
    for fn in ev_files:
        evname = os.path.basename(fn)[:-5]
        series = evname.split("-")[0]
        if series_allow and series not in series_allow:
            continue
        ms = [m for m in json.load(open(fn)) if m.get("result") in ("yes", "no")]
        if len(ms) < 3:
            continue
        candles = {}
        for m in ms:
            cf = os.path.join(cd_dir, f"{m['ticker']}.json")
            c = sorted(json.load(open(cf))) if os.path.exists(cf) else []
            candles[m["ticker"]] = ([x[0] for x in c], c)
        try:
            brs = [bracket_from_market(m) for m in ms]
        except ValueError:
            continue
        trades = []
        for m in ms:
            for t in m.get("trades") or []:
                trades.append(t)
        trades.sort(key=lambda t: t["created_time"])
        tr_ts = [pdt(t["created_time"]).timestamp() for t in trades]
        o = min(pdt(m["open_time"]).timestamp() for m in ms)
        cl = max(pdt(m["close_time"]).timestamp() for m in ms)
        ev = EventBook(evname, series, "high" if series.startswith("KXHIGH") else "low", "", None, cl, ms, brs, {}, {})
        pb = PaperBroker()
        positions = {}
        last_fill = {}
        mid_hist = collections.defaultdict(list)
        ti = 0
        t = o + 600
        while t < cl - 60:
            # quotes from candles (top of book at time t)
            quotes = {}
            for m in ms:
                ts_l, c = candles[m["ticker"]]
                i = bisect.bisect_right(ts_l, t) - 1
                if i < 0:
                    continue
                _, b, a = c[i]
                yb = int(round(b * 100)) if b > 0 else None
                ya = int(round(a * 100)) if 0 < a < 1 else None
                quotes[m["ticker"]] = fvmod.LadderQuote(yb, ya)
            ev.quotes = quotes
            for tk, qq in quotes.items():
                mm = qq.mid()
                if mm is not None:
                    mid_hist[tk].append((t, mm))
                    if len(mid_hist[tk]) > 40:
                        mid_hist[tk] = mid_hist[tk][-40:]
            # process trades up to t
            while ti < len(trades) and tr_ts[ti] <= t:
                for f in pb.on_trade(trades[ti]):
                    last_fill[f.ticker] = tr_ts[ti]
                    p = positions.setdefault(f.ticker, Position(f.ticker, evname))
                    if f.side == "bid":
                        p.yes_qty += f.count
                        p.yes_cost += f.count * f.price_cents / 100
                    else:
                        p.no_qty += f.count
                        p.no_cost += f.count * (100 - f.price_cents) / 100
                    fills_n += 1
                    fill_log.append([tr_ts[ti], evname, f.ticker, f.side, f.price_cents, f.count, None,
                                     (cl - tr_ts[ti]) / 3600])
                    contracts += f.count
                    cost_tot += f.count * (f.price_cents if f.side == "bid" else 100 - f.price_cents) / 100
                ti += 1
            if len(quotes) >= len(ms) - 1:
                fv = fvmod.ladder_fair_value(brs, quotes, kind=ev.kind, hours_to_close=(cl - t) / 3600)
                skip = frozenset(k for k, ft in last_fill.items() if rearm_cooldown_s and t - ft < rearm_cooldown_s)
                desired, _ = sleeve_a.propose(ev, fv, t, cfg_a, 0.0, skip, mid_hist)
                pos_list = list(positions.values())
                kept, _ = filter_orders(desired, {evname: pos_list}, {evname: [m["ticker"] for m in ms]}, risk, 0.0)
            else:
                kept = []
            want = {d.key: d for d in kept}
            for po in pb.resting():
                d = want.get((po.ticker, po.side))
                if d is None or d.price_cents != po.price_cents:
                    pb.cancel(po.oid)
            for k, d in want.items():
                have = sum(po.remaining for po in pb.resting() if (po.ticker, po.side) == k)
                if have <= 0.5:
                    q = quotes.get(d.ticker)
                    best = None
                    if q:
                        best = q.yes_bid if d.side == "bid" else (100 - q.yes_ask if q.yes_ask is not None else None)
                    my_cost = d.price_cents if d.side == "bid" else 100 - d.price_cents
                    qa = 0.0 if (best is None or my_cost > best) else queue_ahead
                    po = pb.place(d, {}, now=t)
                    po.queue_ahead = qa
            t += step_s
        # settle
        res = {m["ticker"]: m["result"] for m in ms}
        for row in fill_log:
            if row[1] == evname and row[6] is None:
                row[6] = res.get(row[2])
        ev_pnl = 0.0
        for tk, p in positions.items():
            r = res.get(tk)
            ev_pnl += (p.yes_qty if r == "yes" else 0) + (p.no_qty if r == "no" else 0) - p.yes_cost - p.no_cost
        by_event[evname] = ev_pnl
        tag = evname.split("-")[1]
        daily[tag] += ev_pnl
    # drawdown over days
    eq, peak, mdd = 0.0, 0.0, 0.0
    for d in sorted(daily, key=lambda s: dt.datetime.strptime(s, "%y%b%d")):
        eq += daily[d]
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    return SimResult(fills_n, contracts, cost_tot, sum(by_event.values()), by_event, dict(daily), mdd, fill_log)
