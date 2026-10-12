"""Tier-2 dataset: one row per (event, decision time) with the market ladder, the Tier-1 model ladder, and the outcome.

Market snapshot at tau (EB = 03Z on D, MO = 15Z on D) from Kalshi candles:
  - historical events (settled before Aug 3 2026): hourly candles from raw/kcandles (+ raw/kalshi_lists for strikes)
  - Aug 1 - Sep 30 2026: 1-minute candles from research/cd (+ research/ev for strikes)
Bracket mids use the bot's LadderQuote rule (one-sided books valued at half the open interval).
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from wxbot.models.brackets import bracket_from_market  # noqa: E402
from wxbot.models.fair_value import LadderQuote  # noqa: E402

MLWX = os.environ.get("MLWX", "/home/claude/mlwx")
ST = json.load(open(os.path.join(os.path.dirname(__file__), "..", "..", "wxbot", "data", "stations.json")))
SER2ICAO = {k: v["icao"] for k, v in ST.items() if v["kind"] == "high"}
DEC = {"EB": 3, "MO": 15}
MON = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def event_date(ev: str):
    tag = ev.split("-")[-1]
    try:
        return dt.date(2000 + int(tag[:2]), MON[tag[2:5]], int(tag[5:7]))
    except Exception:
        return None


def series_of(ev: str) -> str:
    s = ev.split("-")[0]
    return s if s.startswith("KX") else "KX" + s  # legacy HIGHNY -> KXHIGHNY


def mid_from(b, a):
    q = LadderQuote(int(round(b * 100)) if b is not None and b > 0 else None,
                    int(round(a * 100)) if a is not None and 0 < a < 1 else None)
    return q.mid(), q


def load_hist():
    """{event: {"markets": [...], "candles": {ticker: [[ts, bid, ask, vol], ...]}}} from the historical pull."""
    cand = {}
    for fn in glob.glob(os.environ.get("KCANDLES", f"{MLWX}/raw/kcandles") + "/*.jsonl"):
        for line in open(fn):
            try:
                r = json.loads(line)
            except Exception:
                continue
            cand[r["t"]] = r["c"]
    evs = {}
    for fn in glob.glob(f"{MLWX}/raw/kalshi_lists/kalshi_hist_*.json"):
        for m in json.load(open(fn)):
            if m.get("result") not in ("yes", "no"):
                continue
            # keep every settled market of the event so a partly pulled ladder is detected as incomplete
            evs.setdefault(m["event_ticker"], {"markets": [], "candles": {}})
            evs[m["event_ticker"]]["markets"].append(m)
            if m["ticker"] in cand:
                evs[m["event_ticker"]]["candles"][m["ticker"]] = cand[m["ticker"]]
    return {k: v for k, v in evs.items() if v["candles"]}


class _LazyEvents:
    """Mapping-like view that loads one event's 1-minute candles at a time (holding all of Aug-Sep at once took
    several GB). Only .items() is needed by the builders."""

    def __init__(self, files):
        self.files = files

    def items(self):
        for fn in self.files:
            name = os.path.basename(fn)[:-5]
            ms = [m for m in json.load(open(fn)) if m.get("result") in ("yes", "no")]
            cs = {}
            for m in ms:
                cf = f"/home/claude/research/cd/{m['ticker']}.json"
                if os.path.exists(cf):
                    cs[m["ticker"]] = [[x[0], x[1], x[2], None] for x in json.load(open(cf))]
            yield name, {"markets": ms, "candles": cs}


def load_recent():
    files = [fn for fn in sorted(glob.glob("/home/claude/research/ev/KXHIGH*-26AUG*.json")
                                 + glob.glob("/home/claude/research/ev/KXHIGH*-26SEP*.json"))
             if os.path.getsize(fn) >= 10]
    return _LazyEvents(files)


def snapshot(candles, t: float, max_age_s: float = 3 * 3600):
    best = None
    for row in candles:
        if row[0] <= t and (best is None or row[0] > best[0]):
            best = row
    if best is None or t - best[0] > max_age_s:
        return None, None, None
    m, q = mid_from(best[1], best[2])
    return m, best[1], best[2]


def build():
    t1 = pd.read_parquet(f"{MLWX}/tier1_oos.parquet")
    t1["key"] = t1["icao"] + "|" + t1["D"].dt.strftime("%Y-%m-%d") + "|" + t1["dec"]
    t1 = t1.set_index("key")
    pcols = [c for c in t1.columns if c.startswith("p") and c[1:].isdigit()]
    K = (len(pcols) - 1) // 2
    rows = []
    hist = load_hist()
    recent = load_recent()
    src = [("hist", hist), ("recent", recent)]
    seen = set()
    for tag, evs in src:
        for ev, d in evs.items():
            D = event_date(ev)
            ser = series_of(ev)
            icao = SER2ICAO.get(ser)
            if D is None or icao is None or (ser, D) in seen:
                continue
            ms = d["markets"]
            if len(ms) < 3 or sum(m["result"] == "yes" for m in ms) != 1:
                continue
            if sum(m["ticker"] in d["candles"] for m in ms) < len(ms) - 1:
                continue  # candles for this event not fully pulled yet
            try:
                brs = [bracket_from_market(m) for m in ms]
            except (ValueError, TypeError):
                continue
            seen.add((ser, D))
            for dec, hh in DEC.items():
                tau = dt.datetime(D.year, D.month, D.day, hh, tzinfo=dt.timezone.utc).timestamp()
                mids, bids, asks = [], [], []
                for m in ms:
                    mm, b, a = snapshot(d["candles"].get(m["ticker"], []), tau)
                    mids.append(mm)
                    bids.append(b)
                    asks.append(a)
                if sum(x is not None for x in mids) < len(ms) - 1:
                    continue
                key = f"{icao}|{D.isoformat()}|{dec}"
                if key not in t1.index:
                    continue
                r1 = t1.loc[key]
                P = r1[pcols].values.astype(float)
                ar = float(r1["anchor_r"])
                fb = []
                for b in brs:
                    lo = max(b.lo, ar - K)
                    hi = min(b.hi, ar + K)
                    js = [int(v - ar + K) for v in range(int(np.ceil(lo)), int(np.floor(hi)) + 1)]
                    fb.append(float(P[js].sum()) if js else 0.0)
                win = [i for i, m in enumerate(ms) if m["result"] == "yes"][0]
                ev_val = ms[0].get("expiration_value")
                rows.append({
                    "event": ev, "series": ser, "icao": icao, "D": pd.Timestamp(D), "dec": dec, "src": tag,
                    "n": len(ms), "win": win,
                    "lo": [float(b.lo) for b in brs], "hi": [float(b.hi) for b in brs],
                    "m": [x if x is not None else np.nan for x in mids],
                    "bid": [x if x is not None else np.nan for x in bids],
                    "ask": [x if x is not None else np.nan for x in asks],
                    "f": fb, "y": float(ev_val) if ev_val not in (None, "") else np.nan,
                    "y_cli": float(r1["y"]) if pd.notna(r1["y"]) else np.nan,
                })
    df = pd.DataFrame(rows)
    df.to_pickle(f"{MLWX}/tier2.pkl")
    print("rows", len(df), "events", df["event"].nunique(), "by src", df["src"].value_counts().to_dict())
    return df


if __name__ == "__main__":
    build()
