"""Tier-1 v2: settlement-aware integer PMF of the CLI daily high from free guidance + observations, any decision time.

Multiclass gradient boosting over integer offsets from an anchor. The anchor is the latest NBM max (else GFS MOS,
else LAMP), raised to the observed max so far once the local day has started, so late in the day the classes sit
just above what has already been observed. Rounding, the hourly-vs-5-minute peak gap and station quirks are
learned from the CLI labels rather than assumed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

K = 12
NCLS = 2 * K + 1
V5_DATE = pd.Timestamp("2026-05-01")
V5_WEIGHT = 3.0  # upweight post-upgrade days so the model adapts while v5 history is short

FEATS = ["h", "loc_h", "xnd", "nbs_age_h", "txn_d", "nbh_d", "nbh_n", "nbh_dpt_d", "nbh_sky", "nbh_wsp", "nbh_u",
         "nbh_v", "nbh_p06", "nbh_sol", "nbh_tsd", "nbh_gst", "nbe_d", "nbe_xnd", "gfs_d", "nam_d", "mex_d",
         "gfs_age_h", "nam_age_h", "lav_d", "lav_n", "lav_dpt_d", "lav_wsp", "ec_d", "ec18_d", "ec21_d", "ec00_d",
         "ec_age_h", "obs_max_d", "obs_last_d", "obs_age_h", "obs_dwp_d", "obs_sknt", "obs_u", "obs_v", "obs_sky",
         "obs_dt3", "obs_dp3", "norm_d", "e_nbs_l", "e_nbs_m7", "e_nbs_m30", "e_gfs_l", "e_gfs_m7", "e_ec_l",
         "e_ec_m7", "anom_l", "doy_s", "doy_c", "st", "spread", "anchor_frac", "base_d", "nbm_v5"]


def anchor(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    base = df["nbs_txn"].copy()
    base = base.fillna(df["gfs_n_x"]).fillna(df["lav_tmax"] - 1.7)
    a = np.fmax(base, df["obs_max"].fillna(-999))
    return a, base


def features(df: pd.DataFrame) -> pd.DataFrame:
    a, base = anchor(df)
    x = pd.DataFrame(index=df.index)
    x["anchor"] = a
    x["anchor_r"] = np.round(a)
    x["anchor_frac"] = a - np.round(a)
    x["base_d"] = base - a
    x["h"] = df["h"]
    x["loc_h"] = (df["h"] + df["lst_off"]) % 24
    x["xnd"] = df["nbs_xnd"]
    x["nbs_age_h"] = df["nbs_age_h"]
    x["txn_d"] = df["nbs_txn"] - a
    x["nbh_d"] = df["nbh_tmax"] - a
    x["nbh_dpt_d"] = df["nbh_dpt"] - a
    for c in ["nbh_n", "nbh_sky", "nbh_wsp", "nbh_u", "nbh_v", "nbh_p06", "nbh_sol", "nbh_tsd", "nbh_gst", "nbe_xnd",
              "gfs_age_h", "nam_age_h", "lav_n", "lav_wsp", "ec_age_h", "obs_age_h", "obs_sknt", "obs_u", "obs_v",
              "obs_sky", "obs_dt3", "obs_dp3", "e_nbs_l", "e_nbs_m7", "e_nbs_m30", "e_gfs_l", "e_gfs_m7", "e_ec_l",
              "e_ec_m7", "anom_l"]:
        x[c] = df[c]
    x["nbe_d"] = df["nbe_txn"] - a
    x["gfs_d"] = df["gfs_n_x"] - a
    x["nam_d"] = df["nam_n_x"] - a
    x["mex_d"] = df["mex_n_x"] - a
    x["lav_d"] = df["lav_tmax"] - a
    x["lav_dpt_d"] = df["lav_dpt"] - a
    x["ec_d"] = df["ec_max"] - a
    x["ec18_d"] = df["ec_t18"] - a
    x["ec21_d"] = df["ec_t21"] - a
    x["ec00_d"] = df["ec_t00"] - a
    x["obs_max_d"] = df["obs_max"] - a
    x["obs_last_d"] = df["obs_last"] - a
    x["obs_dwp_d"] = df["obs_dwp"] - a
    x["norm_d"] = df["normal"] - a
    x["doy_s"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    x["doy_c"] = np.cos(2 * np.pi * df["doy"] / 365.25)
    x["st"] = df["icao"].astype("category")
    g = df[["nbs_txn", "gfs_n_x", "nam_n_x", "ec_max", "lav_tmax"]]
    x["spread"] = g.max(axis=1) - g.min(axis=1)
    # NBM v5 (May 2026): cycles moved to 00/06/12/18Z and the 12Z run now carries TXN for the same day
    x["nbm_v5"] = (df["D"] >= V5_DATE).astype(int)
    return x


def labels(df: pd.DataFrame, x: pd.DataFrame) -> np.ndarray:
    k = (df["y"] - x["anchor_r"]).clip(-K, K)
    return (k + K).astype(int).values


def fit(df: pd.DataFrame, seed: int = 0, n_estimators: int = 300, lr: float = 0.05):
    import lightgbm as lgb
    x = features(df)
    y = labels(df, x)
    cats = sorted(df["icao"].unique())
    x["st"] = pd.Categorical(df["icao"], categories=cats)
    m = lgb.LGBMClassifier(objective="multiclass", num_class=NCLS, n_estimators=n_estimators, learning_rate=lr,
                           num_leaves=31, min_child_samples=100, subsample=0.7, subsample_freq=1,
                           colsample_bytree=0.7, reg_lambda=2.0, max_bin=63, random_state=seed, verbose=-1, n_jobs=2)
    w = np.where(df["D"] >= V5_DATE, V5_WEIGHT, 1.0)
    m.fit(x[FEATS], y, sample_weight=w)
    m._cats = cats
    return m


def predict_pmf(m, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = features(df)
    x["st"] = pd.Categorical(df["icao"], categories=m._cats)
    p = m.predict_proba(x[FEATS])
    full = np.zeros((len(df), NCLS))
    full[:, m.classes_] = p
    full = np.clip(full, 1e-6, None)
    full /= full.sum(axis=1, keepdims=True)
    return full, x["anchor_r"].values


def scores(P: np.ndarray, anchor_r: np.ndarray, y: np.ndarray) -> dict:
    j = np.clip(y - anchor_r + K, 0, NCLS - 1).astype(int)
    ll = -np.log(P[np.arange(len(y)), j])
    cdf = np.cumsum(P, axis=1)
    obs = (np.arange(NCLS)[None, :] >= j[:, None]).astype(float)
    rps = ((cdf - obs) ** 2).sum(axis=1)
    med = anchor_r + (np.argmax(cdf >= 0.5, axis=1) - K)
    mean = anchor_r + (P * (np.arange(NCLS) - K)[None, :]).sum(axis=1)
    return {"logloss": float(ll.mean()), "rps": float(rps.mean()), "mae_med": float(np.abs(med - y).mean()),
            "mae_mean": float(np.abs(mean - y).mean()), "n": int(len(y))}
