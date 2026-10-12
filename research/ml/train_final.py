"""Train the production Tier-1 model on every labeled day and fit the production pool weights.
Artifacts -> wxbot/ml/artifacts/: tier1_v2.txt (LightGBM), tier1_v2_meta.json, pool_weights.json.
The pool is fit on the last POOL_WINDOW_DAYS days of settled events (default 90; 0 = all history): the market keeps
getting sharper, and a recent window tracks it (docs/ML_RESEARCH.md, section 6b).
Usage: python3 research/ml/train_final.py [--pool-only]"""
import json, os, sys, time, hashlib
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from wxbot.ml import features as F
MLWX = os.environ.get("MLWX", "/home/claude/mlwx")
ART = os.environ.get("ART_DIR") or os.path.join(os.path.dirname(__file__), "..", "..", "wxbot", "ml", "artifacts")  # ART_DIR: a staging folder (Dec 1 freeze)
V5_WEIGHT = 3.0

def kalshi_labels(df):
    """Train on the value Kalshi settled on wherever we have it (it differed from the CLI on 2 of ~3,800 events:
    KXHIGHDEN 2025-06-20 settled 98 vs CLI 99; KXHIGHMIA 2026-08-29 settled 90 vs a bad first CLI of 85)."""
    path = f"{MLWX}/tier2_v3.pkl"
    if not os.path.exists(path):
        return df
    k = pd.read_pickle(path)[["icao", "D", "y"]].dropna().drop_duplicates(["icao", "D"])
    k = k.rename(columns={"y": "y_kalshi"})
    k["D"] = pd.to_datetime(k["D"]).astype("datetime64[ns]")
    out = df.merge(k, on=["icao", "D"], how="left")
    n = int((out["y_kalshi"].notna() & (out["y_kalshi"] != out["y"])).sum())
    out["y"] = out["y_kalshi"].fillna(out["y"])
    print("labels: Kalshi settled value used where known;", n, "station-day-hour rows changed", flush=True)
    return out.drop(columns=["y_kalshi"])


TRAIN_END = os.environ.get("TRAIN_END")  # the Oct 11 freeze trains on data through 2026-09-30 only
T1_NAME = os.environ.get("TIER1_NAME", "tier1_v2")   # e.g. tier1_a600 for candidate A at the Dec 1 freeze
T1_TREES = int(os.environ.get("TIER1_TREES", "300"))  # candidate A: 600
USE_HRRR = os.environ.get("TIER1_HRRR", "0") == "1"  # add the HRRR inputs (needs mlwx/hrrr_feats.parquet)


def add_hrrr_columns(df):
    from wxbot.ml import hrrr as H
    hf = pd.read_parquet(f"{MLWX}/hrrr_feats.parquet")
    hf["D"] = pd.to_datetime(hf["D"]).astype("datetime64[ns]")
    df = df.drop(columns=[c for c in hf.columns if c not in ("icao", "D", "h") and c in df])
    df = df.merge(hf, on=["icao", "D", "h"], how="left")
    return H.add_lag_errors(df)


def train_tier1(n_estimators=300):
    import lightgbm as lgb
    src = F.RawSource(f"{MLWX}/raw")
    parts = [F.station_rows(src, icao) for icao in sorted(F.HIGH)]
    df = kalshi_labels(F.add_lag_errors(pd.concat(parts, ignore_index=True)))
    if TRAIN_END:
        df = df[df["D"] <= pd.Timestamp(TRAIN_END)]
    feats = list(F.FEATS_V3)
    if USE_HRRR:
        from wxbot.ml import hrrr as H
        df = add_hrrr_columns(df)
        feats += H.HRRR_FEATS + H.HRRR_LAG_FEATS
    df = df[(df["nbs_txn"].notna() | df["gfs_n_x"].notna() | df["lav_tmax"].notna()) & df["y"].notna()]
    x = F.model_features(df)
    if USE_HRRR:
        x = H.add_model_columns(x, df)
    y = ((df["y"] - x["anchor_r"]).clip(-F.K, F.K) + F.K).astype(int).values
    cats = sorted(df["icao"].unique())
    x["st"] = pd.Categorical(df["icao"], categories=cats)
    w = np.where(df["D"] >= F.V5_DATE, V5_WEIGHT, 1.0)
    t0 = time.time()
    m = lgb.LGBMClassifier(objective="multiclass", num_class=F.NCLS, n_estimators=n_estimators, learning_rate=0.05,
                           num_leaves=31, min_child_samples=100, subsample=0.7, subsample_freq=1, colsample_bytree=0.7,
                           reg_lambda=2.0, max_bin=63, random_state=0, verbose=-1, n_jobs=2)
    m.fit(x[feats], y, sample_weight=w)
    path = os.path.join(ART, f"{T1_NAME}.txt")
    m.booster_.save_model(path)
    import gzip
    open(path + ".gz", "wb").write(gzip.compress(open(path, "rb").read(), 9))
    meta = {"cats": cats, "feats": feats, "version": "tier1_v3" + ("+hrrr" if USE_HRRR else ""), "classes": [int(c) for c in m.classes_], "K": F.K, "rows": int(len(df)),
            "last_day": str(df["D"].max().date()), "trained_utc": pd.Timestamp.utcnow().isoformat(),
            "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    json.dump(meta, open(os.path.join(ART, f"{T1_NAME}_meta.json"), "w"), indent=1)
    print("tier1 trained", len(df), "rows", round(time.time() - t0), "s", meta["sha256"][:12], flush=True)

POOL_WINDOW_DAYS = int(os.environ.get("POOL_WINDOW_DAYS", "90"))


def fit_pool_weights(window_days=POOL_WINDOW_DAYS):
    from cgep import CGEP3, ladder_arrays
    from tier2_eval_v2 import HOURS, design, prep
    df = prep(os.environ.get("T2PKL", f"{MLWX}/tier2_v3.pkl"))  # the tier-2 rows built from the SHIPPED Tier-1's walk-forward
    if window_days:
        df = df[df["D"] > df["D"].max() - pd.Timedelta(days=window_days)].reset_index(drop=True)
    rows = df.to_dict("records")
    nmax = max(len(r["m"]) for r in rows)
    LM, LF, Z, MASK, WIN = ladder_arrays(rows, nmax)
    X, _, _ = design(df, [])
    mdl = CGEP3(2, tilt=True, curv=True, pen=np.ones(len(HOURS)), alpha0=np.ones(len(HOURS)))
    mdl.fit(X, [LM, LF], Z, MASK, WIN)
    W = mdl.w.reshape(-1, len(HOURS))
    out = {"hours": HOURS, "alpha": W[0].round(4).tolist(), "beta": W[1].round(4).tolist(), "theta": W[2].round(4).tolist(),
           "kappa": W[3].round(4).tolist(), "n_snapshots": int(len(df)), "first_day": str(df["D"].min().date()),
           "last_day": str(df["D"].max().date()), "window_days": window_days,
           "z_scale_f": 2.0, "note": "p_b ∝ m_b^alpha_h f_b^beta_h exp(theta_h z_b + kappa_h z_b^2), z=(center-mkt mean)/2"}
    json.dump(out, open(os.path.join(ART, "pool_weights.json"), "w"), indent=1)
    print("pool weights", json.dumps(out)[:400], flush=True)

if __name__ == "__main__":
    fit_pool_weights()
    if "--pool-only" not in sys.argv:
        train_tier1(T1_TREES)
