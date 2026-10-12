"""HRRR station features for research: thin wrapper over wxbot.ml.hrrr (one implementation for research and live).
Running this file builds mlwx/hrrr_feats.parquet from mlwx/raw/hrrr/*.csv and prints a quick skill check.
`python3 hrrr_features.py tle` builds mlwx/hrrr_tle_feats.parquet (round-2 time-lagged ensemble columns) instead;
`python3 hrrr_features.py dpt` builds mlwx/hrrr_dpt_feats.parquet (round-2 dewpoint columns, needs raw/hrrr_dpt);
`python3 hrrr_features.py tcc` builds mlwx/hrrr_tcc_feats.parquet (round-2 cloud columns, needs raw/hrrr_tcc)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from wxbot.ml.hrrr import (DPT_FEATS, HRRR_FEATS, HRRR_LAG_FEATS, LAG_H, TCC_FEATS, TLE_FEATS,  # noqa: E402,F401
                           add_dpt_columns, add_lag_errors, add_model_columns, add_tcc_columns, add_tle_columns,
                           compute, compute_dpt, compute_tcc, compute_tle, load_runs)

if __name__ == "__main__":
    import sys
    import time
    root = os.environ.get("MLWX", "/home/claude/mlwx")
    t0 = time.time()
    sub = os.environ.get("HSUB", "hrrr")  # round 2: "hrrr_land" (KSFO at its nearest land point)
    W = load_runs(root, sub=sub)
    print("runs", W.index.get_level_values("init").nunique(), "rows", len(W), round(time.time() - t0, 1), "s", flush=True)
    df = pd.read_parquet(f"{root}/tier1_v3.parquet", columns=["icao", "D", "h", "lst_off", "obs_max", "obs_last",
                                                              "obs_age_h", "obs_dwp", "obs_sky", "y", "nbs_txn"])
    lo = W.index.get_level_values("init").min() - pd.Timedelta(days=1)
    df = df[df["D"] >= lo]
    if len(sys.argv) > 1 and sys.argv[1] == "tcc":
        Wc = load_runs(root, sub="hrrr_tcc", col="tcc")
        print("cloud runs", Wc.index.get_level_values("init").nunique(), flush=True)
        t = compute_tcc(df, W, Wc)
        res = pd.concat([df[["icao", "D", "h"]], t], axis=1)
        res.to_parquet(f"{root}/hrrr_tcc_feats.parquet")
        ok = res["hrrr_tcc_now"].notna() & df["y"].notna()
        print("rows", len(res), "with cloud", int(ok.sum()), round(time.time() - t0, 1), "s")
        for h in sorted(df["h"].unique()):
            s = ok & (df["h"] == h)
            if s.sum():
                print(f"h={h:+d} n={int(s.sum())} cover now {res.loc[s, 'hrrr_tcc_now'].mean():.0f}%, "
                      f"obs minus HRRR {res.loc[s, 'hrrr_tcc_err'].mean():+.0f}, rest {res.loc[s, 'hrrr_tcc_rest'].mean():.0f}%")
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "dpt":
        Wd = load_runs(root, sub="hrrr_dpt", col="d2m_f")
        print("dewpoint runs", Wd.index.get_level_values("init").nunique(), flush=True)
        t = compute_dpt(df, W, Wd)
        res = pd.concat([df[["icao", "D", "h"]], t], axis=1)
        res.to_parquet(f"{root}/hrrr_dpt_feats.parquet")
        ok = res["hrrr_tdd_peak"].notna() & df["y"].notna()
        print("rows", len(res), "with dewpoint", int(ok.sum()), round(time.time() - t0, 1), "s")
        for h in sorted(df["h"].unique()):
            s = ok & (df["h"] == h)
            if s.sum() == 0:
                continue
            print(f"h={h:+d} n={int(s.sum())} T-Td at peak median {res.loc[s, 'hrrr_tdd_peak'].median():.1f} F, "
                  f"dewpoint miss now mean {res.loc[s, 'hrrr_td_now_err'].mean():+.2f} F")
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "tle":
        t = compute_tle(df, W)
        res = pd.concat([df[["icao", "D", "h"]], t], axis=1)
        res.to_parquet(f"{root}/hrrr_tle_feats.parquet")
        ok = res["hrrr_tle_mean"].notna() & df["y"].notna()
        print("rows", len(res), "with TLE", int(ok.sum()), round(time.time() - t0, 1), "s")
        for h in sorted(df["h"].unique()):
            s = ok & (df["h"] == h)
            if s.sum() == 0:
                continue
            err = (np.fmax(res.loc[s, "hrrr_tle_mean"], df.loc[s, "obs_max"].fillna(-999)) - df.loc[s, "y"]).abs()
            sd = res.loc[s, "hrrr_tle_sd"]
            hi, lo_ = sd >= sd.quantile(0.8), sd <= sd.quantile(0.2)
            print(f"h={h:+d} n={int(s.sum())} TLE-mean MAE {err.mean():.2f}  SD median {sd.median():.2f}  "
                  f"MAE when SD high {err[hi].mean():.2f} / low {err[lo_].mean():.2f}")
        sys.exit(0)
    feats = compute(df, W)
    res = pd.concat([df[["icao", "D", "h"]], feats], axis=1)
    res.to_parquet(os.environ.get("HFEATS_OUT", f"{root}/hrrr_feats.parquet"))
    print("rows", len(res), "with HRRR", int(res["hrrr_rest"].notna().sum()), round(time.time() - t0, 1), "s")
    ok = res["hrrr_rest"].notna() & df["y"].notna()
    for h in sorted(df["h"].unique()):
        s = ok & (df["h"] == h)
        if s.sum() == 0:
            continue
        e_h = (np.fmax(res.loc[s, "hrrr_rest"], df.loc[s, "obs_max"].fillna(-999)) - df.loc[s, "y"])
        e_n = (np.fmax(df.loc[s, "nbs_txn"], df.loc[s, "obs_max"].fillna(-999)) - df.loc[s, "y"])
        print(f"h={h:+d} n={int(s.sum())}  HRRR max(rest,obs) bias {e_h.mean():+.2f} MAE {e_h.abs().mean():.2f}   "
              f"NBM bias {e_n.mean():+.2f} MAE {e_n.abs().mean():.2f}   now_err {res.loc[s, 'hrrr_now_err'].mean():+.2f}")
    sys.exit(0)
