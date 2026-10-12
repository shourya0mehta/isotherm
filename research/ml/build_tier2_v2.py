"""Tier-2 v2 dataset: one row per (event, decision hour h) with the market ladder at tau = D 00Z + h hours, the
walk-forward Tier-1 v2 PMF mapped to the same brackets, and the settled outcome (Kalshi expiration_value).

Market quotes: hourly candles (historical endpoint, settled before Aug 3 2026) or 1-minute candles (Aug-Sep 2026).
A snapshot uses the last candle that closed at or before tau (never after) and at most 3 hours old.
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from build_tier2 import SER2ICAO, event_date, load_hist, load_recent, series_of, snapshot  # noqa: E402
from wxbot.models.brackets import bracket_from_market  # noqa: E402

MLWX = os.environ.get("MLWX", "/home/claude/mlwx")
HOURS = [-3, 3, 9, 12, 15, 18, 21]


def build(t1_path=f"{MLWX}/tier1_v2_oos.parquet", out=f"{MLWX}/tier2_v2.pkl", hours=None, sources=None):
    """hours: decision hours to build (default HOURS). sources: list of (tag, events mapping) to use instead of the
    historical hourly candles + the Aug-Sep 2026 1-minute tape (e.g. the holdout tape, research/ml/holdout_eval.py)."""
    hours = list(hours or HOURS)
    t1 = pd.read_parquet(t1_path)
    t1["key"] = t1["icao"] + "|" + t1["D"].dt.strftime("%Y-%m-%d") + "|" + t1["h"].astype(int).astype(str)
    t1 = t1.drop_duplicates("key").set_index("key")
    pcols = [c for c in t1.columns if c.startswith("p") and c[1:].isdigit()]
    K = (len(pcols) - 1) // 2
    P_all = t1[pcols].values.astype(float)
    AR = t1["anchor_r"].values.astype(float)
    Y1 = t1["y"].values.astype(float)
    pos = {k: i for i, k in enumerate(t1.index)}
    rows = []
    seen = set()
    for tag, evs in (sources if sources is not None else (("hist", load_hist()), ("recent", load_recent()))):
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
                continue
            try:
                brs = [bracket_from_market(m) for m in ms]
            except (ValueError, TypeError):
                continue
            seen.add((ser, D))
            win = [i for i, m in enumerate(ms) if m["result"] == "yes"][0]
            ev_val = ms[0].get("expiration_value")
            y = float(ev_val) if ev_val not in (None, "") else np.nan
            lo = [float(b.lo) for b in brs]
            hi = [float(b.hi) for b in brs]
            for h in hours:
                tau = dt.datetime(D.year, D.month, D.day, tzinfo=dt.timezone.utc).timestamp() + h * 3600
                mids, bids, asks = [], [], []
                for m in ms:
                    mm, b, a = snapshot(d["candles"].get(m["ticker"], []), tau)
                    mids.append(mm); bids.append(b); asks.append(a)
                if sum(x is not None for x in mids) < len(ms) - 1:
                    continue
                i = pos.get(f"{icao}|{D.isoformat()}|{h}")
                if i is None:
                    continue
                lagged = {}
                for lag_h in (1, 3, 6):
                    lm = [snapshot(d["candles"].get(m["ticker"], []), tau - lag_h * 3600)[0] for m in ms]
                    lagged[f"m{lag_h}"] = ([x if x is not None else np.nan for x in lm]
                                           if sum(x is not None for x in lm) >= len(ms) - 1 else None)
                P = P_all[i]; ar = AR[i]
                fb = []
                for b in brs:
                    l_ = max(b.lo, ar - K)
                    h_ = min(b.hi, ar + K)
                    js = [int(v - ar + K) for v in range(int(np.ceil(l_)), int(np.floor(h_)) + 1)]
                    fb.append(float(P[js].sum()) if js else 0.0)
                rows.append({
                    "event": ev, "series": ser, "icao": icao, "D": pd.Timestamp(D), "h": h, "src": tag,
                    "n": len(ms), "win": win, "lo": lo, "hi": hi,
                    "m": [x if x is not None else np.nan for x in mids],
                    "bid": [x if x is not None else np.nan for x in bids],
                    "ask": [x if x is not None else np.nan for x in asks],
                    "f": fb, "y": y, "y_cli": Y1[i], **lagged,
                })
    df = pd.DataFrame(rows)
    df.to_pickle(out)
    print("rows", len(df), "events", df["event"].nunique(), "by src", df["src"].value_counts().to_dict(),
          "by h", df["h"].value_counts().sort_index().to_dict())
    return df


if __name__ == "__main__":
    build()
