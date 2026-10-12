"""Is the shrinking edge seasonal or a maturing market? Reads the walk-forward eval outputs ({T2TAG}_eval_meta.pkl and
{T2TAG}_P.npz from tier2_eval_v2.py) and prints, all on liquid books unless noted:
  1. the original 7 cities by quarter: market (recalibrated) and Tier-1 log loss, Tier-1 minus market and pool minus
     market with day-block 95% CIs, median spread  -> also written to docs/results/ml_maturation_7cities.json
  2. same-city summer comparisons (Jun-Sep 2025 vs Jun-Sep 2026, Aug-Sep vs Aug-Sep)
  3. later listings: pool gain by months since a city's first market, and the newest listings on all books
  4. Aug-Sep 2026, all cities: liquid vs thin books
Pool = Benter per hour (Bh); market = M1h; Tier-1 = F."""
import json
import os

import numpy as np
import pandas as pd

MLWX = os.environ.get("MLWX", "/home/claude/mlwx")
TAG = os.environ.get("T2TAG", "tier2_v3")
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "results", "ml_maturation_7cities.json")
ORIG7 = ["KAUS", "KDEN", "KLAX", "KMDW", "KMIA", "KNYC", "KPHL"]


def day_ci(d, dates, n=2000):
    g = pd.DataFrame({"d": d, "D": dates}).groupby("D")["d"].agg(["sum", "size"])
    s, c = g["sum"].values, g["size"].values
    rng = np.random.default_rng(0)
    bs = [s[k].sum() / c[k].sum() for k in (rng.integers(0, len(s), len(s)) for _ in range(n))]
    return float(np.mean(d)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def fmt(t):
    return f"{t[0]:+.4f} [{t[1]:+.4f},{t[2]:+.4f}]"


def main():
    meta = pd.read_pickle(f"{MLWX}/{TAG}_eval_meta.pkl")
    Pz = np.load(f"{MLWX}/{TAG}_P.npz")
    win = meta["win"].astype(int).values
    idx = np.arange(len(win))
    ll = {k: -np.log(np.clip(Pz[k][idx, win], 1e-12, None)) for k in ("M1h", "F", "Bh")}
    m = meta.assign(lM=ll["M1h"], lF=ll["F"], lB=ll["Bh"])
    m["dP"], m["dF"] = m.lB - m.lM, m.lF - m.lM
    liq = m[m.liquid]

    print("1. original 7 cities, liquid, by quarter")
    q = liq[liq.icao.isin(ORIG7)].copy()
    q["Q"] = q.D.dt.to_period("Q")
    rows = []
    for Q, g in q.groupby("Q"):
        p, f = day_ci(g.dP.values, g.D.values), day_ci(g.dF.values, g.D.values)
        rows.append(dict(q=str(Q), n=len(g), events=int(g.event.nunique()), market=round(g.lM.mean(), 3),
                         tier1=round(g.lF.mean(), 3), pool=round(g.lB.mean(), 3), pool_d=round(p[0], 4), pool_lo=round(p[1], 4),
                         pool_hi=round(p[2], 4), t1_d=round(f[0], 4), t1_lo=round(f[1], 4), t1_hi=round(f[2], 4),
                         spread=round(float(g.spr.median()) * 100, 1)))
    print(pd.DataFrame(rows).to_string(index=False))
    json.dump(rows, open(OUT, "w"), indent=1)

    print("\n2. same 7 cities, summers")
    for lab, a, b in [("Jun-Sep 2025", "2025-06-01", "2025-09-30"), ("Jun-Sep 2026", "2026-06-01", "2026-09-30"),
                      ("Aug-Sep 2025", "2025-08-01", "2025-09-30"), ("Aug-Sep 2026", "2026-08-01", "2026-09-30")]:
        g = q[(q.D >= a) & (q.D <= b)]
        print(f"  {lab}: n={len(g)} events={g.event.nunique()} pool-market {fmt(day_ci(g.dP.values, g.D.values))} "
              f"Tier1-market {g.dF.mean():+.4f}")

    print("\n3. later listings")
    first = m.groupby("icao").D.min()
    later = liq[~liq.icao.isin(ORIG7)].copy()
    later["age_m"] = (later.D - later.icao.map(first)).dt.days // 30
    print(later.groupby("age_m").agg(n=("dP", "size"), pool=("dP", "mean"), tier1=("dF", "mean")).round(4).to_string())
    newest = first[first >= "2026-08-01"].index.tolist()
    g = m[m.icao.isin(newest)]
    print(f"  newest listings {newest}: all books n={len(g)} pool-market {fmt(day_ci(g.dP.values, g.D.values))} "
          f"Tier1-market {g.dF.mean():+.4f} median spread {g.spr.median():.3f}")

    print("\n4. Aug-Sep 2026, all cities")
    r = m[m.D >= "2026-08-01"]
    for lab, g in [("liquid", r[r.liquid]), ("thin", r[~r.liquid]), ("all books", r)]:
        print(f"  {lab}: n={len(g)} pool-market {fmt(day_ci(g.dP.values, g.D.values))}")


if __name__ == "__main__":
    main()
