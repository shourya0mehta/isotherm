"""Calibration tables for the walk-forward pools (review item: report reliability per hour).

For every bracket of every scored snapshot: predicted probability vs whether it won. Bins are narrow at both ends,
where the maker trades (near-certain favorites and dead longshots). Reports the raw market (M0), the per-hour
recalibrated market (M1h) and the production pool form (BTh), liquid books, for the full window and Jun-Sep 2026,
plus an expected calibration error (ECE, bracket-weighted mean |predicted - observed|).
Usage: python3 calibration.py [tag]   (default t2r1_base: reads {tag}_eval_meta.pkl and {tag}_P.npz)
"""
import os
import sys

import numpy as np
import pandas as pd

MLWX = os.environ.get("MLWX", "/home/claude/mlwx")
BINS = [0, 0.01, 0.03, 0.07, 0.15, 0.3, 0.5, 0.7, 0.85, 0.93, 0.97, 0.99, 1.0001]
HOURS = [-3, 3, 9, 12, 15, 18, 21]


def table(P, meta, sel):
    p, y = [], []
    for i in np.flatnonzero(sel):
        n = len(meta["lo"].iloc[i])
        p.append(P[i, :n])
        yy = np.zeros(n)
        yy[int(meta["win"].iloc[i])] = 1
        y.append(yy)
    p, y = np.concatenate(p), np.concatenate(y)
    b = np.digitize(p, BINS) - 1
    rows = []
    for k in range(len(BINS) - 1):
        s = b == k
        if s.sum():
            rows.append((BINS[k], BINS[k + 1], int(s.sum()), p[s].mean(), y[s].mean()))
    ece = sum(n * abs(pm - ym) for _, _, n, pm, ym in rows) / max(len(p), 1)
    return rows, ece


def main(tag="t2r1_base"):
    meta = pd.read_pickle(f"{MLWX}/{tag}_eval_meta.pkl").reset_index(drop=True)
    Pz = np.load(f"{MLWX}/{tag}_P.npz")
    D = pd.to_datetime(meta["D"])
    liq = meta["liquid"].values.astype(bool)
    for lab, s in (("full window", liq), ("Jun-Sep 2026", liq & (D >= "2026-06-01").values)):
        print(f"\n== liquid books, {lab}: {int(s.sum())} snapshots")
        tabs = {k: table(Pz[k], meta, s) for k in ("M0", "M1h", "BTh")}
        print(f"{'bin':>13s} " + "  ".join(f"{k + ' n / pred / won':>26s}" for k in tabs))
        for j in range(len(tabs["M0"][0])):
            cells = []
            for k in tabs:
                rows = tabs[k][0]
                if j < len(rows):
                    lo, hi, n, pm, ym = rows[j]
                    cells.append(f"{n:8d} / {pm:6.3f} / {ym:6.3f}")
            lo, hi = tabs["M0"][0][j][:2]
            print(f"{lo:5.2f}-{min(hi, 1):5.2f}  " + "  ".join(f"{c:>26s}" for c in cells))
        print("ECE: " + ", ".join(f"{k} {v[1]:.4f}" for k, v in tabs.items()))
        print("ECE by hour (BTh / M1h): " + ", ".join(
            f"h{h}: {table(Pz['BTh'], meta, s & (meta['h'] == h).values)[1]:.4f} / "
            f"{table(Pz['M1h'], meta, s & (meta['h'] == h).values)[1]:.4f}" for h in HOURS))


if __name__ == "__main__":
    main(*sys.argv[1:2])
