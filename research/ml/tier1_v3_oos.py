"""Tier-1 v3 = v2 + ECMWF convection-era fields (max 2 m temperature in 3 h, total cloud, CAPE, dewpoint, afternoon
precipitation). Same walk-forward folds and NBM v5 handling as v2; features built by wxbot.ml.features."""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from wxbot.ml import features as F
MLWX = os.environ.get("MLWX", "/home/claude/mlwx")
FOLDS = ["2024-10-01", "2025-04-01", "2025-10-01", "2026-01-01", "2026-04-01", "2026-05-01", "2026-06-01", "2026-07-01",
         "2026-08-01", "2026-09-01", "2026-10-01"]   # Oct 4 review (2.14): the last fold ends Sep 30; Oct 1 on is the holdout


def build():
    src = F.RawSource(f"{MLWX}/raw")
    parts = [F.station_rows(src, icao) for icao in sorted(F.HIGH)]
    df = F.add_lag_errors(pd.concat(parts, ignore_index=True))
    df.to_parquet(f"{MLWX}/tier1_v3.parquet")
    return df


def fit(df, feats, n_estimators=300, w_extra=None):
    """w_extra: optional per-row multipliers on the training weights (round 5 recency weights)."""
    import lightgbm as lgb
    x = F.model_features(df)
    y = ((df["y"] - x["anchor_r"]).clip(-F.K, F.K) + F.K).astype(int).values
    cats = sorted(df["icao"].unique())
    x["st"] = pd.Categorical(df["icao"], categories=cats)
    w = np.where(df["D"] >= F.V5_DATE, 3.0, 1.0)
    if w_extra is not None:
        w = w * np.asarray(w_extra, float)
    m = lgb.LGBMClassifier(objective="multiclass", num_class=F.NCLS, n_estimators=n_estimators, learning_rate=0.05,
                           num_leaves=31, min_child_samples=100, subsample=0.7, subsample_freq=1, colsample_bytree=0.7,
                           reg_lambda=2.0, max_bin=63, random_state=0, verbose=-1,
                           n_jobs=int(os.environ.get("LGB_THREADS", "2")))
    m.fit(x[feats], y, sample_weight=w)
    m._cats, m._feats = cats, feats
    return m


def predict(m, df):
    x = F.model_features(df)
    x["st"] = pd.Categorical(df["icao"], categories=m._cats)
    p = m.predict_proba(x[m._feats])
    full = np.zeros((len(df), F.NCLS)); full[:, m.classes_] = p
    full = np.clip(full, 1e-6, None); full /= full.sum(axis=1, keepdims=True)
    return full, x["anchor_r"].values


def main():
    df = build() if not os.path.exists(f"{MLWX}/tier1_v3.parquet") else pd.read_parquet(f"{MLWX}/tier1_v3.parquet")
    df = df[(df["nbs_txn"].notna() | df["gfs_n_x"].notna() | df["lav_tmax"].notna())].reset_index(drop=True)
    outs = []
    for a, b in zip(FOLDS[:-1], FOLDS[1:]):
        tr = df[(df["D"] < pd.Timestamp(a) - pd.Timedelta(days=1)) & df["y"].notna()]
        te = df[(df["D"] >= a) & (df["D"] < b)]
        t0 = time.time()
        m = fit(tr, F.FEATS_V3)
        P, ar = predict(m, te)
        o = te[["icao", "D", "h", "y"]].copy(); o["anchor_r"] = ar
        for j in range(P.shape[1]):
            o[f"p{j}"] = P[:, j]
        outs.append(o)
        ok = te["y"].notna().values
        mean = ar + (P * (np.arange(F.NCLS) - F.K)).sum(axis=1)
        print(a, len(tr), len(te), round(time.time() - t0), "s MAE", round(float(np.abs(mean[ok] - te["y"].values[ok]).mean()), 3), flush=True)
    pd.concat(outs, ignore_index=True).to_parquet(f"{MLWX}/tier1_v3_oos.parquet")
    print("done")


if __name__ == "__main__":
    main()
