"""Round 1 Tier-1 variants (pre-registered Oct 3, 2026; docs: Perplexity review doc, Section 7).

Same walk-forward folds and model as Tier-1 v3, data through Sep 30, 2026 only (Oct 1 onward is the holdout).
VARIANT (env):
  v31        NBM spread inputs made relative to their trailing 60-day median at the station (version-aware: v5 days
             only use v5-era values). v5 reports ~40% wider spreads than v4, so raw spreads changed meaning in May
             2026. (A trailing bias correction of the NBM max was tried and dropped: it raised NBM's own error.)
  nonbm      no NBM inputs at all; the anchor falls back to GFS MOS (then LAMP). Measures how much NBM v5 costs.
  consensus  raw NBM inputs kept, but the anchor is 0.6 NBM + 0.2 GFS MOS + 0.2 ECMWF, the last two corrected by
             their trailing 45-day error (night-before MAE Jun-Sep 2026: 1.60 F vs 1.71 F for NBM alone; earlier
             summers within 0.01 F of NBM).
  hrrr       base + HRRR station features (research/ml/hrrr_features.py; Dec 2020 onward, 3 h availability lag).
  hrrr_cons  hrrr + the consensus anchor.            (round 2 candidates, judged on the Dec-Jan holdout)
  hrrr_nonbm hrrr without any NBM input.
  hrrr_early hrrr trained and scored with two more decision hours the day before (h = -9, -6; build_early.py)
  hrrr_combo hrrr + the round-2 inputs listed in COMBO (e.g. "dpt,tcc"); add HRRR_FEATS for the KSFO land point
  hrrr_land  hrrr with KSFO's HRRR taken at its nearest land point (the nearest point is bay water; run with
             HRRR_FEATS=mlwx/hrrr_feats_land.parquet built by HSUB=hrrr_land hrrr_features.py)
  hrrr_gefs  hrrr + GEFS ensemble mean and spread of the day's 6-hour max (wxbot/ml/gefs.py)
  hrrr_t600  hrrr with twice the trees (TIER1_TREES=600): is the forecaster under-fit?
  hrrr_r2    the round-2 candidates that passed, together: early decision hours (as hrrr_early), with TIER1_TREES
  hrrr_tcc   hrrr + HRRR total cloud cover (now, miss vs the observed sky, mean over the heating left)
  hrrr_dpt   hrrr + HRRR 2 m dewpoint (depression at the forecast peak, miss on the latest observation)
  hrrr_tle   hrrr + an HRRR time-lagged ensemble: mean and spread of the rest-of-day max over the last four synoptic
             runs allowed at decision time (research/ml/hrrr_features.py tle).
  hrrr_nonam  hrrr trained and scored without NAM MOS (nam_n_x, nam_age_h missing everywhere; the model spread
             over the other guidance). NAM MOS ends Nov 3, 2026, so a model frozen for Dec-Jan never sees NAM.
  hrrr_nonam6 the same with 600 trees (run with TIER1_TREES=600): the no-NAM version of round-2 candidate A.
  hrrr_t600fb hrrr with 600 trees (TIER1_TREES=600) trained WITH NAM and scored with the registered fallback (GFS
             MOS in NAM's place on every test row): candidate A as it would actually run in Dec-Jan.
  hrrr_fb    the shipped hrrr model (300 trees, with NAM) scored with the fallback: what runs live after Nov 3.
  hrrr_r2nonam  hrrr_r2 (600 trees, early decision hours) without NAM MOS: candidate B for a post-NAM freeze.
  hrrr_r2fb  hrrr_r2 trained with NAM and scored with the fallback: candidate B as it would run in Dec-Jan.
  hrrr_aifs  round 4, Track A: hrrr + ECMWF AIFS ENS member statistics (research/ml/r4_ens_features.py)
  hrrr_ifse  round 4, Track A: hrrr + ECMWF IFS ENS member statistics (3-hourly steps)
  hrrr_ens   round 4, Track A: both
  hrrr_aifs_early  round 4, exploratory: hrrr_aifs with the AIFS inputs only at h <= 3 (before the day starts)
  hrrr_d1    round 5, the NBM fix (candidate D): A (TIER1_TREES=600) plus NBM's departure from the guidance median,
             the median itself, an NBM v5.0.14 flag and recency weights (half-life D_HALF_LIFE days); run with
             FOLDS_FROM=2025-10-01 for the untouched Feb-May check
  hrrr_d2    hrrr_d1 with the classes centered on the guidance median instead of NBM
  hrrr_d3 / hrrr_d4  hrrr_d1 / hrrr_d2 plus the freshest hourly NBM run public at the decision (nbf_*)
  hrrr_d1x ... hrrr_d4x  the D variants without recency weights or the v5.0.14 flag (d1 was worse than the
             shipped forecaster in most months, Oct 8); hrrr_rw: hrrr + recency weights only; hrrr_nbf: hrrr + the
             freshest hourly NBM run only (ablations; OUT_SUFFIX keeps a short run's files apart)
  hrrr_d1p ... hrrr_d4p (and hrrr_d1xp ...)  round 5 step 5: the D variant plus NBM's probabilistic text (NBP: percentiles of the max from
             the freshest 01/07/13/19Z run public at the decision; research/ml/r5_nbp_feats.py)
  hrrr_r2_aifs  round 4, possible Dec-Jan candidate C: candidate B (hrrr_r2: early hours, run with TIER1_TREES=600)
             plus the AIFS ENS inputs at h <= 3 only (judged at h = -9 / -6 against B)
  hrrr_hf    round 4, Track B part 2: hrrr + the NWS 5-minute feed simulated from the 1-minute archive (r4_hf_features)
  hrrr_c2    round 5, early hours: candidate C (hrrr_r2_aifs, run with TIER1_TREES=600) plus round 5's inputs from
             the variant named by C2_D (hrrr_nbf: the freshest hourly NBM run only; hrrr_d3 / hrrr_d4: also the
             departure features); no recency weights unless C2_RW=1; C2_NBP=1 adds the NBP inputs.
             Judged against C at h = -9 / -6 (r4_c_vs_b.py hrrr_c2 hrrr_r2_aifs)
Output: {MLWX}/tier1_r1_{VARIANT}_oos.parquet (same columns as tier1_v3_oos.parquet).
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tier1_v3_oos import fit, predict  # noqa: E402
from wxbot.ml import features as F  # noqa: E402

MLWX = os.environ.get("MLWX", "/home/claude/mlwx")
VARIANT = os.environ.get("VARIANT", "v31")
HOLDOUT_START = pd.Timestamp("2026-10-01")
FOLDS = ["2024-10-01", "2025-04-01", "2025-10-01", "2026-01-01", "2026-04-01", "2026-05-01", "2026-06-01", "2026-07-01",
         "2026-08-01", "2026-09-01", "2026-10-01"]
FAST_R2 = {"hrrr_dpt", "hrrr_tcc", "hrrr_land", "hrrr_t600", "hrrr_gefs", "hrrr_combo", "hrrr_r2", "hrrr_nonam",
           "hrrr_nonam6", "hrrr_t600fb", "hrrr_fb", "hrrr_r2nonam", "hrrr_r2fb", "hrrr_aifs", "hrrr_ifse", "hrrr_ens", "hrrr_hf",
           "hrrr_aifs_early", "hrrr_r2_aifs", "hrrr_d1", "hrrr_d2", "hrrr_d3", "hrrr_d4", "hrrr_d1p", "hrrr_d2p",
           "hrrr_d3p", "hrrr_d4p", "hrrr_c2", "hrrr_d1x", "hrrr_d2x", "hrrr_d3x", "hrrr_d4x", "hrrr_d1xp",
           "hrrr_d2xp", "hrrr_d3xp", "hrrr_d4xp", "hrrr_rw", "hrrr_nbf", "hrrr_nbfp"}
D_VARIANTS = ("hrrr_d1", "hrrr_d2", "hrrr_d3", "hrrr_d4")
X_VARIANTS = tuple(v + "x" for v in D_VARIANTS)      # the same without recency weights or the v5.0.14 flag
NBP_VARIANTS = tuple(v + "p" for v in D_VARIANTS + X_VARIANTS + ("hrrr_nbf",))
_v = VARIANT
USE_NBP = _v in NBP_VARIANTS
if USE_NBP:
    _v = _v[:-1]
NO_RW = _v in X_VARIANTS
if NO_RW:
    _v = _v[:-1]
DBASE = _v                                             # the D design a variant builds on
if VARIANT == "hrrr_c2":
    DBASE = os.environ.get("C2_D", "hrrr_d3")
    USE_NBP = os.environ.get("C2_NBP") == "1"
    NO_RW = os.environ.get("C2_RW", "0") != "1"
RECENCY = (DBASE in D_VARIANTS and not NO_RW) or VARIANT == "hrrr_rw"
V5014_FLAG = DBASE in D_VARIANTS and not NO_RW
OUT_SUFFIX = os.environ.get("OUT_SUFFIX", "")          # ablation runs: keep their outputs apart
EARLY_VARIANTS = ("hrrr_early", "hrrr_r2", "hrrr_r2nonam", "hrrr_r2fb", "hrrr_r2_aifs", "hrrr_c2")
NBM_COLS = ["nbs_txn", "nbs_xnd", "nbs_age_h", "nbe_txn", "nbe_xnd", "e_nbs_l", "e_nbs_m7", "e_nbs_m30", "nbs_txn_prev",
            "e_prev_nbs"] + [c for c in ("nbh_tmax", "nbh_n", "nbh_dpt", "nbh_sky", "nbh_wsp", "nbh_u", "nbh_v", "nbh_p06",
                                          "nbh_sol", "nbh_tsd", "nbh_gst")]


def daily_table(df):
    """One row per station-day from the 03Z decision (the night-before guidance), with the label."""
    b = df[df["h"] == 3][["icao", "D", "y", "nbs_txn", "nbs_xnd", "nbe_xnd", "gfs_n_x", "ec_max"]]
    return b.drop_duplicates(["icao", "D"]).sort_values(["icao", "D"])


def trailing(g, col, win, minp, lag, how="mean", v5_split=False):
    """Trailing statistic of a daily series, shifted by `lag` days so only reports public at decision time count.
    v5_split: rows on/after V5_DATE only see v5-era values (the version change resets the window)."""
    s = g.set_index("D")[col]
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full)
    def roll(x):
        r = x.shift(lag).rolling(win, min_periods=minp)
        return r.mean() if how == "mean" else r.median()
    if not v5_split:
        return roll(s)
    v5 = pd.Timestamp(F.V5_DATE)
    old = roll(s.where(s.index < v5))
    new = roll(s.where(s.index >= v5))
    return old.where(s.index < v5, new)


def add_round1_columns(df):
    base = daily_table(df)
    base["e_nbs"] = base["y"] - base["nbs_txn"]
    base["e_gfs"] = base["y"] - base["gfs_n_x"]
    base["e_ec"] = base["y"] - base["ec_max"]
    parts = []
    for icao, g in base.groupby("icao"):
        for lag in (1, 2):
            f = pd.DataFrame({
                "b_nbs": trailing(g, "e_nbs", 45, 10, lag, v5_split=True),
                "b_gfs": trailing(g, "e_gfs", 45, 10, lag),
                "b_ec": trailing(g, "e_ec", 45, 10, lag),
                "xnd_med": trailing(g, "nbs_xnd", 60, 10, lag, how="median", v5_split=True),
                "nbe_xnd_med": trailing(g, "nbe_xnd", 60, 10, lag, how="median", v5_split=True),
            })
            f["icao"], f["lag0"] = icao, lag
            parts.append(f.rename_axis("D").reset_index())
    t = pd.concat(parts, ignore_index=True)
    df = df.copy()
    df["lag0"] = np.where(df["h"] >= 12, 1, 2)
    df = df.drop(columns=[c for c in ("b_nbs", "b_gfs", "b_ec", "xnd_med", "nbe_xnd_med") if c in df])
    return df.merge(t, on=["icao", "D", "lag0"], how="left")


def apply_variant(df, variant):
    df = df.copy()
    if variant == "v31":
        df["nbs_xnd"] = df["nbs_xnd"] / df["xnd_med"].where(df["xnd_med"] > 0)
        df["nbe_xnd"] = df["nbe_xnd"] / df["nbe_xnd_med"].where(df["nbe_xnd_med"] > 0)
    elif variant == "nonbm":
        for c in NBM_COLS:
            if c in df:
                df[c] = np.nan
    elif variant == "consensus":
        c_gfs = df["gfs_n_x"] + df["b_gfs"].clip(-4, 4).fillna(0.0)
        c_ec = (df["ec_max"] + df["b_ec"].clip(-6, 6)).where(df["b_ec"].notna())  # raw IFS max is biased low
        blend = 0.6 * df["nbs_txn"] + 0.2 * c_gfs + 0.2 * c_ec
        df["_consensus"] = blend.fillna(df["nbs_txn"])
    elif variant != "base":
        raise SystemExit(f"unknown VARIANT {variant}")
    return df


_orig_anchor = F.anchor


def consensus_anchor(df):
    if "_consensus" not in df:
        return _orig_anchor(df)
    base = df["_consensus"].copy()
    base = base.fillna(df["nbs_txn"]).fillna(df["gfs_n_x"]).fillna(df["lav_tmax"] - 1.7)
    return np.fmax(base, df["obs_max"].fillna(-999)), base


def add_hrrr(df):
    import hrrr_features as HF
    hf = pd.read_parquet(os.environ.get("HRRR_FEATS", f"{MLWX}/hrrr_feats.parquet"))
    if VARIANT in EARLY_VARIANTS:
        hf = pd.concat([hf, pd.read_parquet(f"{MLWX}/hrrr_feats_early.parquet")], ignore_index=True)
    df = df.drop(columns=[c for c in hf.columns if c not in ("icao", "D", "h") and c in df])
    df = df.merge(hf, on=["icao", "D", "h"], how="left")
    df = HF.add_lag_errors(df)
    orig = F.model_features

    def mf(d):
        return HF.add_model_columns(orig(d), d)

    F.model_features = mf
    print("HRRR coverage of rows:", round(float(df["hrrr_rest"].notna().mean()), 3), flush=True)
    return df, HF.HRRR_FEATS + HF.HRRR_LAG_FEATS


def add_tle(df):
    """Round 2: HRRR time-lagged ensemble (mean and spread of the last four synoptic runs), on top of add_hrrr."""
    import hrrr_features as HF
    t = pd.read_parquet(os.environ.get("HRRR_TLE_FEATS", f"{MLWX}/hrrr_tle_feats.parquet"))
    df = df.drop(columns=[c for c in t.columns if c not in ("icao", "D", "h") and c in df])
    df = df.merge(t, on=["icao", "D", "h"], how="left")
    prev = F.model_features

    def mf(d):
        return HF.add_tle_columns(prev(d), d)

    F.model_features = mf
    print("TLE coverage of rows:", round(float(df["hrrr_tle_mean"].notna().mean()), 3), flush=True)
    return df, list(HF.TLE_FEATS)


def add_dpt(df):
    """Round 2: HRRR 2 m dewpoint (depression at the forecast peak, miss on the latest observed dewpoint)."""
    import hrrr_features as HF
    t = pd.read_parquet(os.environ.get("HRRR_DPT_FEATS", f"{MLWX}/hrrr_dpt_feats.parquet"))
    df = df.drop(columns=[c for c in t.columns if c not in ("icao", "D", "h") and c in df])
    df = df.merge(t, on=["icao", "D", "h"], how="left")
    prev = F.model_features

    def mf(d):
        return HF.add_dpt_columns(prev(d), d)

    F.model_features = mf
    print("dewpoint coverage of rows:", round(float(df["hrrr_tdd_peak"].notna().mean()), 3), flush=True)
    return df, list(HF.DPT_FEATS)


def add_tcc(df):
    """Round 2: HRRR total cloud cover (now, its miss against the observed sky, mean over the heating left)."""
    import hrrr_features as HF
    t = pd.read_parquet(os.environ.get("HRRR_TCC_FEATS", f"{MLWX}/hrrr_tcc_feats.parquet"))
    df = df.drop(columns=[c for c in t.columns if c not in ("icao", "D", "h") and c in df])
    df = df.merge(t, on=["icao", "D", "h"], how="left")
    prev = F.model_features

    def mf(d):
        return HF.add_tcc_columns(prev(d), d)

    F.model_features = mf
    print("cloud coverage of rows:", round(float(df["hrrr_tcc_now"].notna().mean()), 3), flush=True)
    return df, list(HF.TCC_FEATS)


def add_gefs(df):
    """Round 2: GEFS ensemble mean and spread of the day's 6-hour maximum (wxbot/ml/gefs.py)."""
    from wxbot.ml import gefs as G
    t = pd.read_parquet(os.environ.get("GEFS_FEATS", f"{MLWX}/gefs_feats.parquet"))
    df = df.drop(columns=[c for c in t.columns if c not in ("icao", "D", "h") and c in df])
    df = df.merge(t, on=["icao", "D", "h"], how="left")
    prev = F.model_features

    def mf(d):
        return G.add_model_columns(prev(d), d)

    F.model_features = mf
    print("GEFS coverage of rows:", round(float(df["gefs_mx"].notna().mean()), 3), flush=True)
    return df, list(G.GEFS_FEATS)


def add_ens(df, name):
    """Round 4, Track A: ECMWF ensemble inputs (research/ml/r4_ens_features.py; AIFS ENS from Jul 2025, IFS ENS from
    Apr 2024), relative to the anchor."""
    import r4_ens_features as E
    t = pd.read_parquet(f"{MLWX}/r4_{name}_feats.parquet")
    if VARIANT in EARLY_VARIANTS:              # the h = -9 / -6 rows too (r4_ens_features.py <name> early)
        t = pd.concat([t, pd.read_parquet(f"{MLWX}/r4_{name}_feats_early.parquet")], ignore_index=True)
    df = df.drop(columns=[c for c in t.columns if c not in ("icao", "D", "h") and c in df])
    df = df.merge(t, on=["icao", "D", "h"], how="left")
    prev = F.model_features

    def mf(d):
        return E.add_model_columns(prev(d), d, name)

    F.model_features = mf
    print(name, "ensemble coverage of rows:", round(float(df[f"{E.PREFIX[name]}_mx"].notna().mean()), 3), flush=True)
    return df, list(E.FEATS[name])


def add_hf(df):
    """Round 4, Track B part 2: the NWS 5-minute feed simulated from the 1-minute archive (research/ml/r4_hf_features.py;
    from Jan 2025), relative to the anchor and to the METAR max so far."""
    import r4_hf_features as H
    t = pd.read_parquet(f"{MLWX}/r4_hf_feats.parquet")
    df = df.drop(columns=[c for c in t.columns if c not in ("icao", "D", "h") and c in df])
    df = df.merge(t, on=["icao", "D", "h"], how="left")
    prev = F.model_features

    def mf(d):
        return H.add_model_columns(prev(d), d)

    F.model_features = mf
    print("5-minute feed coverage of rows:", round(float(df["hf_lo_max"].notna().mean()), 3), flush=True)
    return df, list(H.FEATS)


V5014 = pd.Timestamp("2026-07-28")      # NBM v5.0.14 (SCN 26-70): new temperature post-processing from the 12Z run
D_HALF_LIFE = float(os.environ.get("D_HALF_LIFE", "45"))   # recency weights: 1 + 4 x 0.5^(age / half-life)


def guidance_median(df):
    """Median of the day's max from NBM, GFS MOS, HRRR (rest of day), ECMWF and LAMP (with F.anchor's -1.7 F)."""
    g = np.column_stack([df["nbs_txn"], df["gfs_n_x"], df["hrrr_rest"] if "hrrr_rest" in df else np.nan,
                         df["ec_max"], df["lav_tmax"] - 1.7]).astype(float)
    return np.nanmedian(np.where(np.isfinite(g), g, np.nan), axis=1)


def add_round5(df):
    """Round 5 (research report "Beating the market on daily highs", Oct 8): NBM's departure from the other
    guidance and a flag for NBM v5.0.14; hrrr_d2 also centers the classes on the guidance median."""
    df = df.copy()
    df["_gmed"] = guidance_median(df)
    prev = F.model_features

    def mf(d):
        x = prev(d)
        med = d["_gmed"] if "_gmed" in d else guidance_median(d)
        x["nbm_dep"] = d["nbs_txn"] - med
        x["gmed_d"] = med - x["anchor"]
        if V5014_FLAG:
            x["nbm_v5014"] = (pd.to_datetime(d["D"]) >= V5014).astype(int)
        return x

    F.model_features = mf
    if DBASE in ("hrrr_d2", "hrrr_d4"):
        def consensus_median_anchor(d):
            base = pd.Series(d["_gmed"] if "_gmed" in d else guidance_median(d), index=d.index)
            base = base.fillna(d["nbs_txn"]).fillna(d["gfs_n_x"]).fillna(d["lav_tmax"] - 1.7)
            return np.fmax(base, d["obs_max"].fillna(-999)), base
        F.anchor = consensus_median_anchor
    return df, ["nbm_dep", "gmed_d"] + (["nbm_v5014"] if V5014_FLAG else [])


def add_nbm_fresh(df):
    """Round 5: the freshest hourly NBM run public at the decision (research/ml/r5_nbm_fresh_feats.py; 2024 on)."""
    t = pd.read_parquet(f"{MLWX}/nbm_fresh_feats.parquet")
    if VARIANT in EARLY_VARIANTS:              # the h = -9 / -6 rows (r5_nbm_fresh_feats.py early)
        t = pd.concat([t, pd.read_parquet(f"{MLWX}/nbm_fresh_feats_early.parquet")], ignore_index=True)
    t["D"] = pd.to_datetime(t["D"]).astype("datetime64[ns]")
    df = df.drop(columns=[c for c in t.columns if c not in ("icao", "D", "h") and c in df])
    df = df.merge(t.drop_duplicates(["icao", "D", "h"]), on=["icao", "D", "h"], how="left")
    prev = F.model_features

    def mf(d):
        return F.add_nbm_fresh_columns(prev(d), d)

    F.model_features = mf
    print("fresh NBM coverage of rows:", round(float(df["nbf_txn"].notna().mean()), 3), flush=True)
    return df, list(F.NBM_FRESH_FEATS)


def add_nbp(df):
    """Round 5 step 5: NBM's probabilistic text from the freshest run public at the decision (2024 on)."""
    t = pd.read_parquet(f"{MLWX}/nbp_feats.parquet")
    if VARIANT in EARLY_VARIANTS:              # the h = -9 / -6 rows (r5_nbp_feats.py early)
        t = pd.concat([t, pd.read_parquet(f"{MLWX}/nbp_feats_early.parquet")], ignore_index=True)
    t["D"] = pd.to_datetime(t["D"]).astype("datetime64[ns]")
    df = df.drop(columns=[c for c in t.columns if c not in ("icao", "D", "h") and c in df])
    df = df.merge(t.drop_duplicates(["icao", "D", "h"]), on=["icao", "D", "h"], how="left")
    prev = F.model_features

    def mf(d):
        return F.add_nbp_columns(prev(d), d)

    F.model_features = mf
    print("NBP coverage of rows:", round(float(df["nbp_p50"].notna().mean()), 3), flush=True)
    return df, list(F.NBP_FEATS)


def main():
    df = pd.read_parquet(f"{MLWX}/tier1_v3.parquet")
    if VARIANT in EARLY_VARIANTS:  # two earlier decision hours the day before (build_early.py)
        early = pd.read_parquet(f"{MLWX}/tier1_early_rows.parquet")
        df = pd.concat([df, early[[c for c in df.columns if c in early.columns]]], ignore_index=True)
    df = df[df["D"] < HOLDOUT_START]  # the holdout is never touched in round 1
    df = add_round1_columns(df)
    feats = list(F.FEATS_V3)
    if VARIANT.startswith("hrrr"):
        df, extra = add_hrrr(df)
        feats += extra
    if VARIANT == "hrrr_tle":
        df, extra = add_tle(df)
        feats += extra
    if VARIANT == "hrrr_dpt":
        df, extra = add_dpt(df)
        feats += extra
    if VARIANT == "hrrr_tcc":
        df, extra = add_tcc(df)
        feats += extra
    if VARIANT == "hrrr_gefs":
        df, extra = add_gefs(df)
        feats += extra
    if VARIANT in ("hrrr_aifs", "hrrr_ens", "hrrr_aifs_early", "hrrr_r2_aifs", "hrrr_c2"):     # round 4, Track A
        df, extra = add_ens(df, "aifs")
        feats += extra
        if VARIANT in ("hrrr_aifs_early", "hrrr_r2_aifs", "hrrr_c2"):   # only before the day starts (h <= 3)
            late = df["h"].values >= 9
            for c in ("aifs_mx", "aifs_sd", "aifs_q10", "aifs_q90", "aifs_age_h"):
                df.loc[late, c] = np.nan
    if VARIANT in ("hrrr_ifse", "hrrr_ens"):
        df, extra = add_ens(df, "ifs")
        feats += extra
    if VARIANT == "hrrr_hf":                     # round 4, Track B part 2
        df, extra = add_hf(df)
        feats += extra
    if DBASE in D_VARIANTS:                     # round 5: the NBM fix (run with TIER1_TREES=600)
        df, extra = add_round5(df)
        feats += extra
    if DBASE in ("hrrr_d3", "hrrr_d4", "hrrr_nbf"):   # round 5: plus the freshest hourly NBM run
        df, extra = add_nbm_fresh(df)
        feats += extra
    if USE_NBP:                                 # round 5 step 5: plus NBM's probabilistic text
        df, extra = add_nbp(df)
        feats += extra
    if VARIANT == "hrrr_combo":  # the round-2 inputs that passed, together (COMBO="dpt,tcc,gefs,tle")
        adders = {"dpt": add_dpt, "tcc": add_tcc, "gefs": add_gefs, "tle": add_tle}
        for part in [x for x in os.environ.get("COMBO", "").split(",") if x]:
            df, extra = adders[part](df)
            feats += extra
    df = apply_variant(df, {"hrrr": "base", "hrrr_cons": "consensus", "hrrr_nonbm": "nonbm",
                            "hrrr_tle": "base", "hrrr_early": "base", "hrrr_dpt": "base",
                            "hrrr_tcc": "base", "hrrr_t600": "base", "hrrr_gefs": "base",
                            "hrrr_land": "base", "hrrr_combo": "base", "hrrr_r2": "base", "hrrr_nonam": "base",
                            "hrrr_nonam6": "base", "hrrr_t600fb": "base", "hrrr_fb": "base",
                            "hrrr_r2nonam": "base", "hrrr_r2fb": "base", "hrrr_aifs": "base", "hrrr_ifse": "base",
                            "hrrr_ens": "base", "hrrr_hf": "base", "hrrr_aifs_early": "base",
                            "hrrr_r2_aifs": "base", "hrrr_d1": "base", "hrrr_d2": "base", "hrrr_d3": "base",
                            "hrrr_d4": "base", "hrrr_rw": "base", "hrrr_nbf": "base"}.get(DBASE, VARIANT))
    if VARIANT in ("hrrr_nonam", "hrrr_nonam6", "hrrr_r2nonam"):
        df["nam_n_x"] = np.nan
        df["nam_age_h"] = np.nan
    if VARIANT in ("consensus", "hrrr_cons"):
        F.anchor = consensus_anchor
    keep = df["nbs_txn"].notna() | df["gfs_n_x"].notna() | df["lav_tmax"].notna()
    df = df[keep].reset_index(drop=True)
    if os.environ.get("DRYRUN"):                # build the inputs only: list them with their coverage by hour
        x = F.model_features(df)
        print(VARIANT, "rows", len(df), "inputs", len(feats), "anchor", F.anchor.__name__, flush=True)
        for c in feats[-12:]:
            if c in x:
                print(f"  {c:14s}", x[c].notna().groupby(df["h"]).mean().round(3).to_dict(), flush=True)
        return
    outs = []
    # Round-2 variants only need the folds the production-procedure bar uses (pool windows from Mar 2026 on): the
    # Jan 2026 fold onward. Skipping the earlier ones saves about a quarter of the compute.
    folds_from = os.environ.get("FOLDS_FROM") or ("2026-01-01" if VARIANT in FAST_R2 else "")
    for a, b in zip(FOLDS[:-1], FOLDS[1:]):
        if folds_from and a < folds_from:
            continue
        tr = df[(df["D"] < pd.Timestamp(a) - pd.Timedelta(days=1)) & df["y"].notna()]
        te = df[(df["D"] >= a) & (df["D"] < b)]
        ckpt = f"{MLWX}/_ckpt_{VARIANT}{OUT_SUFFIX}_{os.environ.get('TIER1_TREES', '300')}_{a}.parquet"
        if os.path.exists(ckpt):            # a fold finished before a restart of the machine
            outs.append(pd.read_parquet(ckpt))
            print(VARIANT, a, "from checkpoint", flush=True)
            continue
        t0 = time.time()
        w_extra = None
        if RECENCY:                             # recency weights relative to the fold's start
            age = (pd.Timestamp(a) - pd.to_datetime(tr["D"])).dt.days.clip(lower=0).to_numpy(float)
            w_extra = 1.0 + 4.0 * 0.5 ** (age / D_HALF_LIFE)
        m = fit(tr, feats, n_estimators=int(os.environ.get("TIER1_TREES", "300")), w_extra=w_extra)
        if VARIANT in ("hrrr_t600fb", "hrrr_fb", "hrrr_r2fb"):   # the registered NAM fallback on every test row (wxbot/ml/features.nam_fallback)
            te = te.copy()
            te["nam_n_x"] = te["gfs_n_x"]
            te["nam_age_h"] = te["gfs_age_h"] + F.NAM_GFS_AGE_GAP_H
        P, ar = predict(m, te)
        o = te[["icao", "D", "h", "y"]].copy()
        o["anchor_r"] = ar
        for j in range(P.shape[1]):
            o[f"p{j}"] = P[:, j]
        outs.append(o)
        o.to_parquet(ckpt)
        ok = te["y"].notna().values
        mean = ar + (P * (np.arange(F.NCLS) - F.K)).sum(axis=1)
        print(VARIANT, a, len(tr), len(te), round(time.time() - t0), "s MAE",
              round(float(np.abs(mean[ok] - te["y"].values[ok]).mean()), 3), flush=True)
    pd.concat(outs, ignore_index=True).to_parquet(f"{MLWX}/tier1_r1_{VARIANT}{OUT_SUFFIX}_oos.parquet")
    for a in FOLDS[:-1]:
        f = f"{MLWX}/_ckpt_{VARIANT}{OUT_SUFFIX}_{os.environ.get('TIER1_TREES', '300')}_{a}.parquet"
        if os.path.exists(f):
            os.remove(f)
    print("done", VARIANT, flush=True)


if __name__ == "__main__":
    main()
