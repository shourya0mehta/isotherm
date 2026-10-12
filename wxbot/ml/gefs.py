"""Round-2 candidate (not in the shipped model): GEFS ensemble mean and spread of the 6-hour maximum 2 m temperature.

Source files: raw/gefs/YYYYMMDD_HHz.csv (research/pullers/pull_gefs.py): icao, fh, tmax_mean_f, tmax_sd_f for the
windows (fh - 6, fh], fh = 6..48. Timing rule: a run counts only if init <= decision - LAG_H (GEFS 0.5 degree files
for the first 48 hours appear about 4-5 hours after init).
Features (temperatures relative to the Tier-1 anchor):
  gefs_mx_d   largest ensemble-mean 6-hour max over the windows that overlap the rest of the local day, minus anchor
  gefs_sd     the ensemble spread (F) in that window: how much the 31 members disagree about the day's peak
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

LAG_H = 6
FHS = list(range(6, 49, 6))
GEFS_FEATS = ["gefs_mx_d", "gefs_sd"]
HOUR = np.timedelta64(1, "h")


def load_runs(root: str, since=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    d0 = f"{root}/raw/gefs" if os.path.isdir(f"{root}/raw/gefs") else f"{root}/gefs"
    parts = []
    for f in sorted(glob.glob(f"{d0}/*_*z.csv")):
        b = os.path.basename(f)[:-4]
        init = pd.Timestamp(b[:8]) + pd.Timedelta(hours=int(b[9:11]))
        if since is not None and init < pd.Timestamp(since):
            continue
        d = pd.read_csv(f)
        if d.empty:
            continue
        d["init"] = init
        parts.append(d)
    if not parts:
        e = pd.DataFrame(columns=FHS)
        return e, e
    G = pd.concat(parts, ignore_index=True)
    G["init"] = G["init"].astype("datetime64[ns]")
    M = G.set_index(["icao", "init", "fh"])["tmax_mean_f"].unstack("fh").reindex(columns=FHS).sort_index()
    S = G.set_index(["icao", "init", "fh"])["tmax_sd_f"].unstack("fh").reindex(columns=FHS).sort_index()
    return M, S


def compute(df: pd.DataFrame, M: pd.DataFrame, S: pd.DataFrame, lag_h: int = LAG_H) -> pd.DataFrame:
    """df: icao, D, h, lst_off. Returns gefs_mx (absolute F), gefs_sd, gefs_age_h aligned to df.index."""
    out = pd.DataFrame(index=df.index, columns=["gefs_mx", "gefs_sd", "gefs_age_h"], dtype=float)
    if M.empty:
        return out
    runs = M.index.to_frame(index=False).sort_values("init")
    icao = df["icao"].astype(str).values
    D = pd.to_datetime(df["D"]).values.astype("datetime64[ns]")
    dec = D + df["h"].values.astype(int) * HOUR
    start = D + (-df["lst_off"].values.astype(int)) * HOUR
    end = start + 24 * HOUR
    q = pd.DataFrame({"i": np.arange(len(df)), "icao": icao, "t": dec - lag_h * HOUR}).sort_values("t")
    m = pd.merge_asof(q, runs.assign(init=runs["init"].astype("datetime64[ns]")), left_on="t", right_on="init",
                      by="icao", direction="backward", tolerance=pd.Timedelta(hours=24))
    init = np.full(len(df), np.datetime64("NaT", "ns"))
    init[m["i"].values] = m["init"].values.astype("datetime64[ns]")
    idx = pd.MultiIndex.from_arrays([icao, pd.DatetimeIndex(init)])
    Mi = M.copy()
    Mi.index = Mi.index.set_levels(Mi.index.levels[1].astype("datetime64[ns]"), level=1)
    Si = S.copy()
    Si.index = Si.index.set_levels(Si.index.levels[1].astype("datetime64[ns]"), level=1)
    A = Mi.reindex(idx).to_numpy(float)
    Sd = Si.reindex(idx).to_numpy(float)
    fh = np.array(FHS)[None, :]
    w_end = init[:, None] + fh * HOUR
    w_start = w_end - 6 * HOUR
    lo = np.maximum(dec, start)[:, None]
    ok = (w_end > lo) & (w_start < end[:, None]) & np.isfinite(A)
    B = np.where(ok, A, -np.inf)
    j = B.argmax(axis=1)
    has = ok.any(axis=1)
    out["gefs_mx"] = np.where(has, B[np.arange(len(B)), j], np.nan)
    out["gefs_sd"] = np.where(has, Sd[np.arange(len(Sd)), j], np.nan)
    out["gefs_age_h"] = np.where(has, (dec - init) / HOUR, np.nan)
    return out


def add_model_columns(x: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    x["gefs_mx_d"] = (df["gefs_mx"] if "gefs_mx" in df else np.nan) - x["anchor"]
    x["gefs_sd"] = df["gefs_sd"] if "gefs_sd" in df else np.nan
    return x
