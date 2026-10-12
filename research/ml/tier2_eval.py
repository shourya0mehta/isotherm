"""Walk-forward evaluation: does a settlement-aware ML forecaster, pooled with the market by CGEP, beat the market?

Bar to beat: the recalibrated market (M1), not raw NOAA. Every fit uses only events settled >= 2 days before the
test month starts. Scores: bracket log loss and Brier; paired differences with day-block bootstrap intervals.
Also a taker backtest at the decision time: buy YES at the ask or NO at 1-bid when edge > fee + margin.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from cgep import CGEP, ladder_arrays  # noqa: E402

MLWX = os.environ.get("MLWX", "/home/claude/mlwx")
EPS = 1e-4


def row_stats(r):
    m = np.array([x if np.isfinite(x) else 0.005 for x in r["m"]], float); m = np.clip(m, EPS, None); m /= m.sum()
    f = np.clip(np.array(r["f"], float), EPS, None); f /= f.sum()
    lo = np.array(r["lo"]); hi = np.array(r["hi"])
    c = np.where(lo < -1e5, hi - 1.0, np.where(hi > 1e5, lo + 1.0, (lo + hi) / 2.0))
    bid = np.array(r["bid"], float); ask = np.array(r["ask"], float)
    both = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask < 1)
    spr = float(np.nanmean((ask - bid)[both])) if both.any() else 0.2
    return {"mu_m": float((m * c).sum()), "mu_f": float((f * c).sum()),
            "H_m": float(-(m * np.log(m)).sum()), "H_f": float(-(f * np.log(f)).sum()),
            "spr": spr, "nq": int(both.sum())}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    st = pd.DataFrame([row_stats(r) for r in df.to_dict("records")], index=df.index)
    df = pd.concat([df, st], axis=1)
    df["dis"] = df["mu_f"] - df["mu_m"]
    df["adis"] = df["dis"].abs()
    df["dec_mo"] = (df["dec"] == "MO").astype(float)
    doy = df["D"].dt.dayofyear
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    df["err_m"] = df["y"] - df["mu_m"]
    df["err_f"] = df["y"] - df["mu_f"]
    # recent errors of each source at the station (same decision time), only from settled days
    out = []
    for (icao, dec), g in df.groupby(["icao", "dec"]):
        g = g.sort_values("D").set_index("D")
        lag = 1 if dec == "MO" else 2
        full = g[["err_m", "err_f"]].asfreq("D")
        sh = full.shift(lag)
        g["lag_err_m"] = sh["err_m"].reindex(g.index)
        g["lag_err_f"] = sh["err_f"].reindex(g.index)
        g["m7_err_m"] = sh["err_m"].rolling(7, min_periods=2).mean().reindex(g.index)
        g["m7_err_f"] = sh["err_f"].rolling(7, min_periods=2).mean().reindex(g.index)
        out.append(g.reset_index())
    df = pd.concat(out, ignore_index=True)
    for c in ["lag_err_m", "lag_err_f", "m7_err_m", "m7_err_f"]:
        df[c] = df[c].fillna(0.0).clip(-8, 8)
    return df


CTX = ["dec_mo", "cos_doy", "sin_doy", "dis", "adis", "H_m", "H_f", "spr"]
LAGS = ["lag_err_m", "lag_err_f", "m7_err_m", "m7_err_f"]


def design(df, cols, mu=None, sd=None):
    X = df[cols].values.astype(float)
    if mu is None:
        mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    X = (X - mu) / sd
    return np.hstack([np.ones((len(df), 1)), X]), mu, sd


def logloss_brier(P, WIN):
    n = len(WIN)
    pw = np.clip(P[np.arange(n), WIN], EPS, 1)
    Y = np.zeros_like(P); Y[np.arange(n), WIN] = 1
    return -np.log(pw), ((P - Y) ** 2).sum(axis=1)


def taker_pnl(P, rows, margin=0.03):
    """At the snapshot: buy YES at ask if p - ask > fee + margin; buy NO at (1 - bid) if (1-p) - (1-bid) > fee + margin."""
    pnl, n_tr = [], 0
    for i, r in enumerate(rows):
        tot = 0.0
        for b in range(len(r["m"])):
            p = P[i, b]
            a, bd = r["ask"][b], r["bid"][b]
            win = 1.0 if b == r["win"] else 0.0
            if np.isfinite(a) and 0 < a < 1:
                fee = math.ceil(0.07 * a * (1 - a) * 100) / 100
                if p - a > fee + margin:
                    tot += win - a - fee; n_tr += 1
            if np.isfinite(bd) and 0 < bd < 1:
                cost = 1 - bd
                fee = math.ceil(0.07 * cost * (1 - cost) * 100) / 100
                if (1 - p) - cost > fee + margin:
                    tot += (1 - win) - cost - fee; n_tr += 1
        pnl.append(tot)
    return np.array(pnl), n_tr


def block_ci(diff, dates, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    ud = np.unique(dates)
    idx = {d: np.where(dates == d)[0] for d in ud}
    sums = np.array([diff[idx[d]].sum() for d in ud]); cnts = np.array([len(idx[d]) for d in ud])
    bs = []
    for _ in range(n):
        s = rng.integers(0, len(ud), len(ud))
        bs.append(sums[s].sum() / cnts[s].sum())
    return np.percentile(bs, [2.5, 97.5])


def main(freq="W", min_train=500):
    df = pd.read_pickle(f"{MLWX}/tier2.pkl")
    df = df[df["y"].notna()].reset_index(drop=True)
    df = add_features(df).sort_values("D").reset_index(drop=True)
    df["msum"] = [float(np.nansum(r)) for r in df["m"]]
    df["liquid"] = (df["spr"] <= 0.10) & df["msum"].between(0.9, 1.25)
    df["per"] = df["D"].dt.to_period(freq)
    pers = sorted(df["per"].unique())
    variants = {
        "M1 market recalibrated": dict(cols=[], kw=dict(gate_alpha=False, gate_beta=False, tilt=False, curv=False), beta0=True),
        "M2 market recal + tilt/curv": dict(cols=[], kw=dict(gate_alpha=False, gate_beta=False, tilt=True, curv=True), beta0=True),
        "B  Benter pool (market + Tier-1)": dict(cols=[], kw=dict(gate_alpha=False, gate_beta=False, tilt=False, curv=False)),
        "BT Benter + tilt/curvature": dict(cols=[], kw=dict(gate_alpha=False, gate_beta=False, tilt=True, curv=True)),
        "G  CGEP (context-gated)": dict(cols=CTX, kw=dict(gate_alpha=True, gate_beta=True, tilt=True, curv=True)),
        "GL CGEP + recent-error memory": dict(cols=CTX + LAGS, kw=dict(gate_alpha=True, gate_beta=True, tilt=True, curv=True)),
    }
    res = {k: [] for k in ["M0 market (raw)", "F  Tier-1 model alone"] + list(variants)}
    meta, coefs = [], {}
    for k, per in enumerate(pers):
        start = per.start_time
        tr = df[df["D"] <= start - pd.Timedelta(days=2)]
        te = df[df["per"] == per]
        if len(tr) < min_train or len(te) == 0:
            continue
        rows_tr, rows_te = tr.to_dict("records"), te.to_dict("records")
        nmax = max(max(len(r["m"]) for r in rows_tr), max(len(r["m"]) for r in rows_te))
        A_tr = ladder_arrays(rows_tr, nmax)
        A_te = ladder_arrays(rows_te, nmax)
        LMt, LFt, Zt, Mt, Wt = A_te
        P0 = np.where(Mt, np.exp(LMt), 0); P0 /= P0.sum(axis=1, keepdims=True)
        PF = np.where(Mt, np.exp(LFt), 0); PF /= PF.sum(axis=1, keepdims=True)
        res["M0 market (raw)"].append(P0)
        res["F  Tier-1 model alone"].append(PF)
        for name, v in variants.items():
            X_tr, mu, sd = design(tr, v["cols"])
            X_te, _, _ = design(te, v["cols"], mu, sd)
            LM, LF, Z, MASK, WIN = A_tr
            if v.get("beta0"):
                LF = np.zeros_like(LF); LFt_ = np.zeros_like(LFt)
            else:
                LFt_ = LFt
            mdl = CGEP(**v["kw"]).fit(X_tr, LM, LF, Z, MASK, WIN)
            res[name].append(mdl.predict(X_te, LMt, LFt_, Zt, Mt))
            coefs[name] = mdl.coef_summary(["const"] + v["cols"])
        meta.append(te.assign(_rows=rows_te))
        print(per, "train", len(tr), "test", len(te), flush=True)
    meta = pd.concat(meta, ignore_index=True)
    WIN = meta["win"].values
    dates = meta["D"].values
    print(f"\nWALK-FORWARD TEST: {len(meta)} event-snapshots, {meta['event'].nunique()} events, "
          f"{meta['D'].min().date()}..{meta['D'].max().date()}, cities {meta['icao'].nunique()}, "
          f"liquid share {meta['liquid'].mean():.2f}")
    rows_all = list(meta["_rows"])
    Pall = {k: np.vstack(v) for k, v in res.items()}
    base_ll, _ = logloss_brier(Pall["M1 market recalibrated"], WIN)
    for label, sel in (("ALL", np.ones(len(meta), bool)), ("LIQUID (spread<=10c)", meta["liquid"].values),
                       ("EB (11pm ET eve)", (meta["dec"] == "EB").values), ("MO (11am ET)", (meta["dec"] == "MO").values)):
        print(f"\n== {label}: n={sel.sum()}")
        print(f"{'model':38s} {'logloss':>8s} {'Brier':>7s} {'vs recal mkt':>13s} {'95% CI (day blocks)':>22s} {'taker $/evt':>11s} {'trades':>7s}")
        for name, P in Pall.items():
            ll, br = logloss_brier(P[sel], WIN[sel])
            d = ll - base_ll[sel]
            ci = block_ci(d, dates[sel])
            pnl, ntr = taker_pnl(P[sel], [r for r, s_ in zip(rows_all, sel) if s_])
            print(f"{name:38s} {ll.mean():8.4f} {br.mean():7.4f} {d.mean():+13.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] {pnl.mean():+11.4f} {ntr:7d}")
    best = "GL CGEP + recent-error memory"
    ll, _ = logloss_brier(Pall[best], WIN)
    meta["d"] = ll - base_ll
    print("\nby city (GL minus recalibrated market; negative = better):")
    print(meta.groupby("icao")["d"].agg(["mean", "count"]).round(4).sort_values("mean").T.to_string())
    print("\ncoefficients (last fit):")
    for k, v in coefs.items():
        print(" ", k, v)
    keep = meta.drop(columns=["_rows"])
    keep.to_pickle(f"{MLWX}/tier2_eval_meta.pkl")
    np.savez(f"{MLWX}/tier2_P.npz", **{k.split()[0]: v for k, v in Pall.items()})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "W", int(sys.argv[2]) if len(sys.argv) > 2 else 500)
