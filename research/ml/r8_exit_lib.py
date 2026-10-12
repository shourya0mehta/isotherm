"""Round 8 (exit rules), shared research helpers: should a sleeve sell a position before settlement?

Every sleeve holds to settlement today. This tests, against hold-to-settlement on the SAME fills, rules that sell a held
side at the bid (on Kalshi, selling a held side is buying the other side at 1 - price):
  model   sell when bid - fee > fair + m (m = 0, 2c, 5c): fair = the held side's probability from the fair-value hook
          the sleeve trades on, at that moment
  dead    sell when the observations already rule the held side out (the bot's own METAR kill: the running max, less
          the bot's 1 F safety, has passed a YES bracket; or it has reached the open top bracket we hold NO on) and
          bid > fee
  drop30  the owner's stop: sell when the held side's mid falls 30c or more below a lot's entry price within 2 hours
          of that lot's fill (the whole side is sold)
  below50 sell when the held side's mid falls below 50c (a lot bought at 50c or more, its side's mid now under 50c;
          positions bought under 50c, e.g. the thin sleeve's fills on empty books or a taker's long shots, never
          trigger it, since they did not fall there)
Fee: Kalshi's taker fee 0.07 P (1 - P) per contract at the execution price P (not rounded to the cent; Kalshi rounds
up per order, which would add under 0.04c a contract at 25 contracts). The original maker fills pay no fee.
Variants: one cent of slippage (the trigger reads the quote, the sale fills 1c below the bid, as research/ml/
taker_prod.py's slip1), and a size cap. The replay's quotes are 1-minute candles of the best bid and ask with no
size, so the cap uses a proxy instead of the displayed bid size: the contracts other takers sold at our price or
better in the 60 minutes after the signal (as research/ml/taker_depth.py's depth); the rest is held to settlement.

Accounting: selling q contracts at s changes the P&L against holding by q * (s - fee(s) - payout), payout 1 if the
held side wins, else 0, booked on the event's settlement day (as the hold P&L) so the paired daily difference
compares the same events. After an exit the side is flat until a later fill (same fills as the hold run) reopens it.
Holdout: no exit is evaluated at or after Oct 1, 2026 00:00Z (CUT); a position still open then is held.
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import math
import os
import sys
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:                     # wxbot (the research scripts that import this put research/ml first)
    sys.path.append(ROOT)

MLWX = os.environ.get("MLWX", "/home/claude/mlwx")
EV = os.environ.get("EVENTS", "/home/claude/research/ev")
CD = os.environ.get("CANDLES", "/home/claude/research/cd")
OUTDIR = os.environ.get("R8_DIR", f"{MLWX}/r8_exit")
CUT = dt.datetime(2026, 10, 1, tzinfo=dt.timezone.utc).timestamp()     # nothing at or after this is evaluated
INF = 10 ** 6
SETTLE_LAG_H = 8.0                                                      # run_joint's default
MON = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"],
                                  1)}


def fee(p: float) -> float:
    """Kalshi taker fee per contract at price p (dollars), unrounded."""
    return 0.07 * p * (1.0 - p)


def event_date(name: str) -> dt.date:
    tag = name.split("-")[1]
    return dt.date(2000 + int(tag[:2]), MON[tag[2:5]], int(tag[5:7]))


def settle_day(close_ts: float) -> str:
    """The UTC day run_joint books an event's P&L on (its first 5-minute step at or after close + 8 h; the daily-high
    ladders close at 05-08Z, so the step grid never moves it across midnight)."""
    return dt.datetime.fromtimestamp(close_ts + SETTLE_LAG_H * 3600, dt.timezone.utc).strftime("%Y-%m-%d")


def hm(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%b %d %H:%MZ")


def mid_of(yb, ya):
    """wxbot LadderQuote.mid() on YES cents (None = empty side): the value of YES in dollars, or None."""
    if yb is not None and ya is not None:
        return (yb + ya) / 200.0
    if ya is not None:
        return ya / 200.0
    if yb is not None:
        return (yb + 100) / 200.0
    return None


# ------------------------------------------------------------------------------------------- the maker replay hook
class Recorder:
    """fv_fn wrapper for wxbot.backtest.joint.run_joint: calls the sleeve's fair-value hook unchanged and, for every
    bracket with a fill so far, records (t, ticker, P(YES), YES bid, YES ask in cents, -1 = empty side). Fills are
    read off run_joint's fill_log, which holds every fill up to the step before fv_fn is called; so nothing in the
    replay changes (the hold run is the reference run) and every step at which the sleeve had a fair value for a held
    bracket is on record (the events inside their quoting window: open + 10 min to close - 1 min)."""

    def __init__(self, inner, fill_log: list):
        self.inner = inner
        self.fills = fill_log
        self.seen = 0
        self.held: set = set()
        self.t, self.tk, self.p, self.bid, self.ask = [], [], [], [], []
        self.calls = 0

    def __call__(self, name, t, brackets, quotes, fv):
        out = self.inner(name, t, brackets, quotes, fv) if self.inner is not None else fv
        self.calls += 1
        n = len(self.fills)
        while self.seen < n:
            self.held.add(self.fills[self.seen]["ticker"])
            self.seen += 1
        if out is None or not self.held:
            return out
        for b in brackets:
            tk = b.ticker
            if tk not in self.held:
                continue
            q = quotes.get(tk)
            self.t.append(t)
            self.tk.append(tk)
            self.p.append(float(out.probs.get(tk, np.nan)))
            self.bid.append(-1 if q is None or q.yes_bid is None else int(q.yes_bid))
            self.ask.append(-1 if q is None or q.yes_ask is None else int(q.yes_ask))
        return out

    def arrays(self) -> dict:
        """{ticker: (t, p_yes, yes_bid, yes_ask)} sorted by t."""
        out = {}
        tk = np.array(self.tk, dtype=object)
        t = np.array(self.t, float)
        p = np.array(self.p, float)
        b = np.array(self.bid, np.int16)
        a = np.array(self.ask, np.int16)
        for k in np.unique(tk) if len(tk) else []:
            s = np.flatnonzero(tk == k)
            o = s[np.argsort(t[s], kind="stable")]
            out[str(k)] = (t[o], p[o], b[o], a[o])
        return out


class ObsMax:
    """pool_replay2.obs_max, indexed: the running max of the METAR T-group temperatures (F) on local day D available
    at t (readings at or before t - 5 min), the input of the replay's METAR kills. Built from pool_replay2.OBS."""

    def __init__(self, obs: dict):
        self.idx = {}
        for icao, (ts, tp, ld) in obs.items():
            days = ld.astype("datetime64[D]")
            for D in np.unique(days):
                s = np.flatnonzero(days == D)
                o = s[np.argsort(ts[s], kind="stable")]
                self.idx[(icao, str(D))] = (ts[o].astype(float), np.maximum.accumulate(tp[o].astype(float)))

    def at(self, icao: str, D: dt.date, t: float):
        a = self.idx.get((icao, D.isoformat()))
        if a is None:
            return None
        i = int(np.searchsorted(a[0], t - 300, side="right"))
        return float(a[1][i - 1]) if i > 0 else None


def dead_side(lo: float, hi: float, side: str, om, margin: int = 1) -> bool:
    """The held side is ruled out by the observed running max om (F), with the bot's kill margin
    (wxbot/models/fair_value.ladder_fair_value: the CLI max is taken to be at least floor(om + 0.5) - 1; margin=0 takes
    it to be at least the rounded METAR max itself)."""
    if om is None:
        return False
    floor_f = math.floor(om + 0.5) - margin
    if side == "yes":
        return hi < floor_f
    return hi >= INF and lo <= floor_f


# ------------------------------------------------------------------------------------------------------- rules
@dataclass(frozen=True)
class Rule:
    name: str
    kind: str                 # model | dead | dead0 | drop30 | below50 | below50any
    m: float = 0.0            # model: margin over fair (dollars)
    slip: float = 0.0         # the sale fills this far below the bid (dollars)
    cap: bool = False         # size capped by the depth proxy
    tight: bool = False       # stops only: the mid must come from a two-sided book with a spread of 10c or less

    def label(self) -> str:
        base = {"model": f"model exit m={self.m * 100:.0f}c", "dead": "dead-side exit",
                "dead0": "dead-side exit without the 1 F margin", "drop30": "price-drop stop 30c/2h",
                "below50": "price-drop stop <50c", "below50any": "price-drop stop <50c, any entry price"}[self.kind]
        return (base + (", 1c slip" if self.slip else "") + (", depth-capped" if self.cap else "")
                + (", two-sided book <=10c" if self.tight else ""))


RULES = [Rule("model0", "model", 0.00), Rule("model2", "model", 0.02), Rule("model5", "model", 0.05),
         Rule("model0_s1", "model", 0.00, 0.01), Rule("model2_s1", "model", 0.02, 0.01),
         Rule("model5_s1", "model", 0.05, 0.01),
         Rule("model0_cap", "model", 0.00, 0.0, True), Rule("model2_cap", "model", 0.02, 0.0, True),
         Rule("model5_cap", "model", 0.05, 0.0, True),
         Rule("dead", "dead"), Rule("dead_s1", "dead", 0.0, 0.01), Rule("dead0", "dead0"),
         Rule("drop30", "drop30"), Rule("drop30_s1", "drop30", 0.0, 0.01),
         Rule("drop30_tight", "drop30", tight=True),
         Rule("below50", "below50"), Rule("below50_s1", "below50", 0.0, 0.01),
         Rule("below50_tight", "below50", tight=True), Rule("below50_any", "below50any")]
MAIN_RULES = ["model0", "model2", "model5", "dead", "drop30", "below50"]


def trigger(rule: Rule, fair, bid, mid, dead: bool, lots: list, t: float, tight_ok: bool = True,
            dead0: bool | None = None) -> bool:
    """Does the rule sell now? bid, mid: the held side's, in dollars (bid None = nobody to sell to); tight_ok: both
    sides quoted with a spread of 10c or less; dead / dead0: ruled out with / without the 1 F kill margin."""
    if bid is None or bid <= 0:
        return False
    if rule.tight and not tight_ok:
        return False
    if rule.kind == "model":
        return fair is not None and np.isfinite(fair) and bid - fee(bid) > fair + rule.m + 1e-12
    if rule.kind == "dead":
        return dead and bid - fee(bid) > 0
    if rule.kind == "dead0":
        return (dead if dead0 is None else dead0) and bid - fee(bid) > 0
    if rule.kind == "drop30":
        return mid is not None and any(t - l[0] <= 7200 and l[2] > 0 and mid <= l[1] - 0.30 + 1e-12 for l in lots)
    if rule.kind == "below50":     # "falls below 50c": a lot bought at 50c or more whose side's mid is now under 50c
        return mid is not None and mid < 0.50 - 1e-12 and any(l[2] > 0 and l[1] >= 0.50 - 1e-12 for l in lots)
    if rule.kind == "below50any":  # the literal reading: the mid is under 50c, whatever we paid
        return mid is not None and mid < 0.50 - 1e-12
    raise ValueError(rule.kind)


def side_quote(side: str, yb: int, ya: int):
    """(bid, mid, tight_ok) of the held side in dollars from the YES quote in cents (-1 = empty); tight_ok = both
    sides quoted, spread <= 10c."""
    yb_ = None if yb is None or yb < 0 else int(yb)
    ya_ = None if ya is None or ya < 0 else int(ya)
    m = mid_of(yb_, ya_)
    tight = yb_ is not None and ya_ is not None and ya_ - yb_ <= 10
    if side == "yes":
        return (yb_ / 100.0 if yb_ is not None and yb_ > 0 else None), m, tight
    return ((100 - ya_) / 100.0 if ya_ is not None and ya_ < 100 else None), (None if m is None else 1.0 - m), tight


def walk(steps, fills, payout: float, rule: Rule, info: dict, depth_fn=None, cut: float = CUT):
    """One position (ticker, side): steps = list of (t, fair, bid, mid, dead, tight_ok[, dead0]) for the held side; fills =
    [(ts, cost per contract, count)] sorted by ts. Returns the exits: dicts with t, qty, entry (mean cost of the
    contracts sold), exit (sale price), fair, bid, delta (P&L against holding them), won (the side would have won)."""
    exits = []
    lots = []                      # [ts, cost, remaining]
    j = 0
    for t, fair, bid, mid, dead, tight_ok, *more in steps:
        dead0 = more[0] if more else None
        if t >= cut:
            break
        while j < len(fills) and fills[j][0] <= t:
            lots.append([fills[j][0], fills[j][1], float(fills[j][2])])
            j += 1
        held = sum(l[2] for l in lots)
        if held <= 1e-9:
            continue
        if not trigger(rule, fair, bid, mid, dead, lots, t, tight_ok, dead0):
            continue
        s = bid - rule.slip
        if s <= 0:
            continue
        qty = held
        if rule.cap and depth_fn is not None:
            qty = min(held, depth_fn(t, s))
            if qty <= 1e-9:
                continue
        cost = 0.0
        left = qty
        for l in lots:                       # sell the oldest lots first (the P&L does not depend on the order)
            take = min(l[2], left)
            cost += take * l[1]
            l[2] -= take
            left -= take
            if left <= 1e-9:
                break
        lots = [l for l in lots if l[2] > 1e-9]
        exits.append({"t": t, "qty": qty, "entry": cost / qty, "exit": s, "fair": fair, "bid": bid, "mid": mid,
                      "delta": qty * (s - fee(s) - payout), "won": payout > 0.5, **info})
    return exits


# ------------------------------------------------------------------------------------------ event and tape helpers
def load_events(names, with_trades: bool = False, keep=None) -> dict:
    """{ticker: {event, lo, hi, label, result, close, D, series}} for the replay's event files (and the trades per
    ticker when asked, for the tickers in keep or all: [(ts, yes_price, count, taker_side)])."""
    from wxbot.models.brackets import bracket_from_market
    from wxbot.timeutil import parse_ts
    out, trades = {}, {}
    for name in names:
        fn = os.path.join(EV, f"{name}.json")
        ms = [m for m in json.load(open(fn)) if m.get("result") in ("yes", "no")]
        for m in ms:
            try:
                b = bracket_from_market(m)
            except ValueError:
                continue
            close = parse_ts(m["close_time"])
            out[m["ticker"]] = {"event": name, "lo": b.lo, "hi": b.hi, "label": b.label(), "result": m["result"],
                                "close": close, "D": event_date(name), "series": name.split("-")[0]}
            if with_trades and (keep is None or m["ticker"] in keep):
                tr = []
                for x in m.get("trades") or []:
                    tr.append((parse_ts(x["created_time"]), float(x["yes_price_dollars"]), float(x["count_fp"]),
                               x.get("taker_side") or x.get("taker_outcome_side")))
                tr.sort()
                trades[m["ticker"]] = tr
    return (out, trades) if with_trades else out


def depth_proxy(trades: list, side: str, window_s: float = 3600.0):
    """depth_fn(t, s) for a held side: the contracts other takers sold at our sale price s or better in (t, t +
    window]: selling YES at s rests on YES bids >= s, hit by takers selling YES (taker_side 'no') at yes_price >= s;
    selling NO at s is buying YES at 1 - s, which takes YES offers <= 1 - s, lifted by takers buying YES
    (taker_side 'yes') at yes_price <= 1 - s."""
    if isinstance(trades, tuple):                 # already as arrays (ts, price, count, taker side)
        ts, px, ct, tk = trades
    else:
        ts = np.array([x[0] for x in trades], float)
        px = np.array([x[1] for x in trades], float)
        ct = np.array([x[2] for x in trades], float)
        tk = np.array([x[3] or "" for x in trades], dtype=object)

    def fn(t, s):
        a = int(np.searchsorted(ts, t, side="right"))
        b = int(np.searchsorted(ts, t + window_s, side="right"))
        if b <= a:
            return 0.0
        if side == "yes":
            sel = (tk[a:b] == "no") & (px[a:b] >= s - 1e-9)
        else:
            sel = (tk[a:b] == "yes") & (px[a:b] <= 1.0 - s + 1e-9)
        return float(ct[a:b][sel].sum())
    return fn


# ----------------------------------------------------------------------------------------------- the overlay
def positions(fills: list) -> dict:
    """{(ticker, 'yes'|'no'): [(ts, cost, count)]} from run_joint fill_log entries."""
    out = collections.defaultdict(list)
    for f in fills:
        side = "yes" if f["side"] == "bid" else "no"
        cost = f["price_cents"] / 100.0 if side == "yes" else (100 - f["price_cents"]) / 100.0
        out[(f["ticker"], side)].append((float(f["ts"]), cost, float(f["count"])))
    for v in out.values():
        v.sort()
    return dict(out)


def steps_for(rec: dict, tk: str, side: str, t_from: float, I: dict, obs: ObsMax | None, icao: str | None):
    """The held side's (t, fair, bid, mid, dead, tight_ok, dead0) at each recorded step from t_from on."""
    r = rec.get(tk)
    if r is None:
        return []
    t, p, b, a = r
    k0 = int(np.searchsorted(t, t_from, side="left"))
    out = []
    for k in range(k0, len(t)):
        bid, mid, tight = side_quote(side, int(b[k]), int(a[k]))
        fair = float(p[k]) if side == "yes" else 1.0 - float(p[k])
        om = obs.at(icao, I["D"], float(t[k])) if (obs is not None and icao) else None
        out.append((float(t[k]), fair, bid, mid, dead_side(I["lo"], I["hi"], side, om), tight,
                    dead_side(I["lo"], I["hi"], side, om, margin=0)))
    return out


def overlay(fills: list, rec: dict, info: dict, rules: list, obs: ObsMax | None = None, ser2icao: dict | None = None,
            trades: dict | None = None) -> dict:
    """{rule name: [exit dicts]} for one replay run, plus the per-position disagreement record under key '_pos'."""
    pos = positions(fills)
    res = {r.name: [] for r in rules}
    prow = []
    for (tk, side), fl in sorted(pos.items()):
        I = info[tk]
        icao = (ser2icao or {}).get(I["series"])
        payout = 1.0 if I["result"] == side else 0.0
        st = steps_for(rec, tk, side, fl[0][0], I, obs, icao)
        meta = {"event": I["event"], "ticker": tk, "label": I["label"], "side": side, "day": settle_day(I["close"])}
        dfn = depth_proxy(trades[tk], side) if trades is not None and tk in trades else None
        for r in rules:
            res[r.name] += walk(st, fl, payout, r, meta, depth_fn=dfn)
        # disagreement after we bought: did the held side's fair ever sit 5c or more below its bid?
        gaps = [s_[2] - s_[1] for s_ in st if s_[0] < CUT and s_[2] is not None and np.isfinite(s_[1])]
        prow.append({**meta, "contracts": sum(x[2] for x in fl), "cost": sum(x[1] * x[2] for x in fl),
                     "won": payout > 0.5, "steps": len(st), "max_gap": max(gaps) if gaps else float("nan"),
                     "first_fill": fl[0][0], "dead_ever": any(s[4] for s in st if s[0] < CUT),
                     "dead0_ever": any(s[6] for s in st if s[0] < CUT)})
    res["_pos"] = prow
    return res
