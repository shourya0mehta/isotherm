"""Round 4, Track A (docs/results/r4_center_plan.md): ECMWF ensemble inputs for Tier-1.

From the member 2 m temperatures at each station (research extraction of the dynamical.org archives of ECMWF open
data: mlwx/r4_aifs_t2m.npz, AIFS ENS every 6 hours from Jul 2, 2025, 6-hourly steps; mlwx/r4_ifs_t2m.npz, IFS ENS
00Z from Apr 1, 2024, 3-hourly steps), for every Tier-1 row (icao, D, h): the latest run public at the decision time
(init + DELAY hours), each member's highest value over the steps in the rest of the local day, and across members:
mean, standard deviation, 10th and 90th percentiles (F), and the run's age. Nothing dated Oct 1, 2026 or later is
read (the extraction stops there). Writes mlwx/r4_{name}_feats.parquet; research/ml/tier1_r1_oos.py VARIANT=hrrr_aifs
/ hrrr_ifse / hrrr_ens adds them (relative to the anchor) to the shipped HRRR forecaster's inputs.
Usage: python3 research/ml/r4_ens_features.py aifs|ifs [early]   (early: the h = -9 / -6 rows -> r4_<name>_feats_early)
"""
import os
import sys

import numpy as np
import pandas as pd

MLWX = os.environ.get("MLWX", "/home/claude/mlwx")
DELAY_H = {"aifs": 7, "ifs": 9}            # conservative publication delays (ECMWF open data)
PREFIX = {"aifs": "aifs", "ifs": "ifse"}
HOUR = np.timedelta64(1, "h")


def features(name: str, rows: pd.DataFrame) -> pd.DataFrame:
    z = np.load(f"{MLWX}/r4_{name}_t2m.npz")
    T = z["t2m"] * 9 / 5 + 32                                  # station, init, lead, member (F)
    stations = list(z["stations"])
    inits = z["inits"].astype("datetime64[s]").astype("datetime64[ns]")
    leads = z["leads_h"].astype(float)
    p = PREFIX[name]
    out = pd.DataFrame(index=rows.index, columns=[f"{p}_mx", f"{p}_sd", f"{p}_q10", f"{p}_q90", f"{p}_age_h"],
                       dtype=float)
    for s, g in rows.groupby("icao"):
        if s not in stations:
            continue
        k = stations.index(s)
        D = pd.to_datetime(g["D"]).values.astype("datetime64[ns]")
        dec = D + g["h"].values.astype(int) * HOUR
        start = D + (-g["lst_off"].values.astype(int)) * HOUR
        end = start + 24 * HOUR
        n = np.searchsorted(inits, dec - DELAY_H[name] * HOUR, side="right") - 1
        ok = n >= 0
        n = np.clip(n, 0, len(inits) - 1)
        valid = inits[n][:, None] + (leads[None, :] * 3600).astype("timedelta64[s]").astype("timedelta64[ns]")
        lo = np.maximum(dec, start)[:, None]
        inday = (valid >= lo) & (valid < end[:, None])         # steps in the rest of the local day
        X = T[k][n]                                            # rows, lead, member
        M = np.where(inday[:, :, None] & np.isfinite(X), X, -np.inf).max(axis=1)   # rows, member
        good = ok & inday.any(axis=1) & np.isfinite(M).all(axis=1)
        Mg = np.where(good[:, None], M, np.nan)
        out.loc[g.index, f"{p}_mx"] = np.nanmean(Mg, axis=1)
        out.loc[g.index, f"{p}_sd"] = np.nanstd(Mg, axis=1)
        out.loc[g.index, f"{p}_q10"] = np.nanpercentile(Mg, 10, axis=1)
        out.loc[g.index, f"{p}_q90"] = np.nanpercentile(Mg, 90, axis=1)
        out.loc[g.index, f"{p}_age_h"] = np.where(good, (dec - inits[n]) / HOUR, np.nan)
    return out


FEATS = {"aifs": ["aifs_mx_d", "aifs_sd", "aifs_q10_d", "aifs_q90_d", "aifs_age_h"],
         "ifs": ["ifse_mx_d", "ifse_sd", "ifse_q10_d", "ifse_q90_d", "ifse_age_h"]}


def add_model_columns(x: pd.DataFrame, df: pd.DataFrame, name: str) -> pd.DataFrame:
    """The model's view: center and quantiles relative to the Tier-1 anchor, spread and age as they are."""
    p = PREFIX[name]
    for c in ("mx", "q10", "q90"):
        x[f"{p}_{c}_d"] = (df[f"{p}_{c}"] if f"{p}_{c}" in df else np.nan) - x["anchor"]
    x[f"{p}_sd"] = df[f"{p}_sd"] if f"{p}_sd" in df else np.nan
    x[f"{p}_age_h"] = df[f"{p}_age_h"] if f"{p}_age_h" in df else np.nan
    return x


def main(name: str, early: bool = False):
    src = "tier1_early_rows" if early else "tier1_v3"   # early: the h = -9 / -6 rows (research/ml/build_early.py)
    rows = pd.read_parquet(f"{MLWX}/{src}.parquet", columns=["icao", "D", "h", "lst_off", "y", "nbs_txn"])
    rows = rows[pd.to_datetime(rows["D"]) < pd.Timestamp("2026-10-01")].reset_index(drop=True)
    f = features(name, rows)
    out = pd.concat([rows[["icao", "D", "h"]], f], axis=1)
    out.to_parquet(f"{MLWX}/r4_{name}_feats{'_early' if early else ''}.parquet")
    p = PREFIX[name]
    have = out[f"{p}_mx"].notna()
    sub = rows[have & rows["y"].notna()]
    err = (out.loc[sub.index, f"{p}_mx"] - sub["y"])
    nb = (sub["nbs_txn"] - sub["y"])
    print(name, f"coverage {have.mean():.1%} of rows ({have.sum()}),", f"from {pd.to_datetime(rows.loc[have, 'D']).min().date()}")
    print(f"  ensemble mean of the members' rest-of-day max vs the settled high: bias {err.mean():+.2f} F, MAE "
          f"{err.abs().mean():.2f} F (NBM text max on the same rows: bias {nb.mean():+.2f}, MAE {nb.abs().mean():.2f}); "
          f"spread {out.loc[have, f'{p}_sd'].mean():.2f} F")
    for h in sorted(rows["h"].unique()):
        m = sub["h"] == h
        print(f"  h={h:+d}: MAE {err[m].abs().mean():.2f}, after removing its mean bias {(err[m] - err[m].mean()).abs().mean():.2f}"
              f" (NBM {nb[m].abs().mean():.2f}), corr(spread, |NBM miss|) "
              f"{np.corrcoef(out.loc[sub.index[m], f'{p}_sd'].astype(float), nb[m].abs())[0, 1]:.2f}")


if __name__ == "__main__":
    main(sys.argv[1], early="early" in sys.argv[2:])
