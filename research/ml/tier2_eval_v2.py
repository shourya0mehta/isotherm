"""Walk-forward test across decision hours: does pooling the market with the Tier-1 v2 forecaster beat the market?

Bar to beat: the market recalibrated per decision hour (M1h). All pools are fit only on events settled at least two
days before each test period starts. Paired log-loss differences with day-block bootstrap intervals; taker backtest
at the snapshot with Kalshi taker fees.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from cgep import CGEP3, ladder_arrays, log_ladder  # noqa: E402
from tier2_eval import block_ci, logloss_brier, row_stats, taker_pnl  # noqa: E402

MLWX = os.environ.get("MLWX", "/home/claude/mlwx")
HOURS = [-3, 3, 9, 12, 15, 18, 21]
CTX = ["spr", "dH", "adis", "msum"]


HOLDOUT_START = pd.Timestamp(os.environ.get("HOLDOUT_START", "2026-10-01"))  # pre-registered holdout: never scored here


def prep(path):
    df = pd.read_pickle(path)
    df = df[df["win"].notna()].reset_index(drop=True)  # the winning bracket is all scoring needs
    df = df[pd.to_datetime(df["D"]) < HOLDOUT_START].reset_index(drop=True)
    st = pd.DataFrame([row_stats(r) for r in df.to_dict("records")], index=df.index)
    df = pd.concat([df, st], axis=1)
    df["dis"] = df["mu_f"] - df["mu_m"]
    df["adis"] = df["dis"].abs()
    df["dH"] = df["H_m"] - df["H_f"]
    df["msum"] = [float(np.nansum(r)) for r in df["m"]]
    df["liquid"] = (df["spr"] <= 0.10) & df["msum"].between(0.9, 1.25)
    df = add_relperf(df)
    for c in sorted(df["icao"].unique()):
        df[f"c_{c}"] = (df["icao"] == c).astype(float)
    return df.sort_values(["D", "h"]).reset_index(drop=True)


def _ll_at_win(v, win):
    x = np.array([q if q is not None and np.isfinite(q) else 0.005 for q in v], float)
    x = np.clip(x, 1e-4, None); x = x / x.sum()
    return -np.log(x[win])


def add_relperf(df, win_days=(14, 45)):
    """Adaptive-gate context: how the model has done against the raw market recently, using only events settled at
    least 2 days before the row's date (all cities and hours pooled; also per city for the long window)."""
    df = df.copy()
    df["d_ll"] = [_ll_at_win(r["f"], r["win"]) - _ll_at_win(r["m"], r["win"]) for r in df.to_dict("records")]
    daily = df[df["liquid"]].groupby("D")["d_ll"].agg(["sum", "count"]).asfreq("D").fillna(0.0)
    for w in win_days:
        s = daily["sum"].shift(2).rolling(w, min_periods=1).sum()
        c = daily["count"].shift(2).rolling(w, min_periods=1).sum()
        rel = (s / c.replace(0, np.nan)).reindex(df["D"]).values
        df[f"rel{w}"] = np.nan_to_num(rel, nan=0.0)
    cd = df[df["liquid"]].groupby(["icao", "D"])["d_ll"].agg(["sum", "count"])
    out = []
    for icao, g in cd.groupby(level=0):
        g = g.droplevel(0).asfreq("D").fillna(0.0)
        s = g["sum"].shift(2).rolling(60, min_periods=1).sum()
        c = g["count"].shift(2).rolling(60, min_periods=1).sum()
        out.append(pd.DataFrame({"icao": icao, "D": g.index, "relc60": (s / c.replace(0, np.nan)).values}))
    out = pd.concat(out, ignore_index=True)
    df = df.merge(out, on=["icao", "D"], how="left")
    df["relc60"] = df["relc60"].fillna(0.0)
    return df


def design(df, ctx, mu=None, sd=None):
    oh = np.stack([(df["h"].values == h).astype(float) for h in HOURS], axis=1)
    if not ctx:
        return oh, None, None
    X = df[ctx].values.astype(float)
    if mu is None:
        mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    return np.hstack([oh, (X - mu) / sd]), mu, sd


VARIANTS = {
    "M1h market recal per hour": dict(exp=["m"], tilt=False, curv=False, ctx=[]),
    "M2h + tilt/curvature": dict(exp=["m"], tilt=True, curv=True, ctx=[]),
    "ML  market + its own lags": dict(exp=["m", "m1", "m3"], tilt=True, curv=True, ctx=[]),
    "Bh  Benter per hour": dict(exp=["m", "f"], tilt=False, curv=False, ctx=[]),
    "BTh Benter + tilt/curv": dict(exp=["m", "f"], tilt=True, curv=True, ctx=[]),
    "BTL + market lags (temporal pool)": dict(exp=["m", "f", "m1", "m3"], tilt=True, curv=True, ctx=[]),
    "G   CGEP gated + lags": dict(exp=["m", "f", "m1", "m3"], tilt=True, curv=True, ctx=CTX),
    "A   adaptive Benter (recent skill gate)": dict(exp=["m", "f"], tilt=True, curv=True, ctx=["rel14", "rel45", "relc60"],
                                                  gate_pen=20.0),
    "AG  adaptive + context gates": dict(exp=["m", "f"], tilt=True, curv=True,
                                         ctx=["rel14", "rel45", "relc60"] + CTX, gate_pen=20.0),
}


def expert_logs(rows, keys, nmax, LM, LF):
    out = []
    for k in keys:
        out.append(LM if k == "m" else LF if k == "f" else log_ladder(rows, k, nmax))
    return out


def main(path=f"{MLWX}/tier2_v2.pkl", freq="M", min_train=3000, gate_pen=200.0, only=None):
    """only: optional list of variant short names (e.g. ["M1h", "Bh", "BTh"]) to fit; the others are skipped."""
    df = prep(path)
    df["per"] = df["D"].dt.to_period(freq)
    pers = sorted(df["per"].unique())
    variants = {k: v for k, v in VARIANTS.items() if only is None or k.split()[0] in only}
    res = {k: [] for k in ["M0  market raw", "F   Tier-1 v2 alone"] + list(variants)}
    meta, last = [], {}
    for per in pers:
        start = per.start_time
        tr = df[df["D"] <= start - pd.Timedelta(days=2)]
        te = df[df["per"] == per]
        if len(tr) < min_train or len(te) == 0:
            continue
        rows_tr, rows_te = tr.to_dict("records"), te.to_dict("records")
        nmax = max(max(len(r["m"]) for r in rows_tr), max(len(r["m"]) for r in rows_te))
        LM, LF, Z, MASK, WIN = ladder_arrays(rows_tr, nmax)
        LMt, LFt, Zt, Mt, Wt = ladder_arrays(rows_te, nmax)
        P0 = np.where(Mt, np.exp(LMt), 0); P0 /= P0.sum(axis=1, keepdims=True)
        PF = np.where(Mt, np.exp(LFt), 0); PF /= PF.sum(axis=1, keepdims=True)
        res["M0  market raw"].append(P0)
        res["F   Tier-1 v2 alone"].append(PF)
        for name, v in variants.items():
            ctx = [c for c in df.columns if c.startswith("c_K")] if v["ctx"] == "CITY" else v["ctx"]
            Xtr, mu, sd = design(tr, ctx)
            Xte, _, _ = design(te, ctx, mu, sd)
            d = Xtr.shape[1]
            pen = np.r_[np.ones(len(HOURS)), np.full(d - len(HOURS), v.get("gate_pen", gate_pen))]
            a0 = np.r_[np.ones(len(HOURS)), np.zeros(d - len(HOURS))]
            mdl = CGEP3(len(v["exp"]), tilt=v["tilt"], curv=v["curv"], pen=pen, alpha0=a0)
            mdl.fit(Xtr, expert_logs(rows_tr, v["exp"], nmax, LM, LF), Z, MASK, WIN)
            res[name].append(mdl.predict(Xte, expert_logs(rows_te, v["exp"], nmax, LMt, LFt), Zt, Mt))
            last[name] = mdl.coefs([f"h{h}" for h in HOURS] + ctx,
                                   v["exp"] + (["tilt"] if v["tilt"] else []) + (["curv"] if v["curv"] else []))
        meta.append(te.assign(_rows=rows_te))
        print(per, "train", len(tr), "test", len(te), flush=True)
    meta = pd.concat(meta, ignore_index=True)
    WIN = meta["win"].values
    dates = meta["D"].values
    rows_all = list(meta["_rows"])
    Pall = {k: np.vstack(v) for k, v in res.items()}
    base = "M1h market recal per hour"
    base_ll, _ = logloss_brier(Pall[base], WIN)
    print(f"\nWALK-FORWARD TEST: {len(meta)} snapshots, {meta['event'].nunique()} events, {meta['D'].dt.date.nunique()} days, "
          f"{meta['D'].min().date()}..{meta['D'].max().date()}, cities {meta['icao'].nunique()}, liquid {meta['liquid'].mean():.2f}")

    def table(label, sel):
        print(f"\n== {label}: n={int(sel.sum())}, events={meta.loc[sel, 'event'].nunique()}")
        print(f"{'model':28s} {'logloss':>8s} {'Brier':>7s} {'vs M1h':>8s} {'95% CI':>18s} {'taker$/snap':>11s} {'trades':>6s}")
        for name, P in Pall.items():
            ll, br = logloss_brier(P[sel], WIN[sel])
            dlt = ll - base_ll[sel]
            ci = block_ci(dlt, dates[sel], n=1000)
            pnl, ntr = taker_pnl(P[sel], [r for r, s_ in zip(rows_all, sel) if s_])
            print(f"{name:28s} {ll.mean():8.4f} {br.mean():7.4f} {dlt.mean():+8.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] "
                  f"{pnl.mean():+11.4f} {ntr:6d}")

    allsel = np.ones(len(meta), bool)
    table("ALL", allsel)
    table("LIQUID", meta["liquid"].values)
    for h in HOURS:
        table(f"LIQUID h={h:+d}", (meta["liquid"] & (meta["h"] == h)).values)
    for src in meta["src"].unique():
        table(f"LIQUID src={src}", (meta["liquid"] & (meta["src"] == src)).values)
    print("\nlast-fit coefficients:")
    for k, v in last.items():
        print(" ", k, v)
    tag = os.path.basename(path).replace(".pkl", "")  # tier2_v2 / tier2_v3 ...
    meta.drop(columns=["_rows"]).to_pickle(f"{MLWX}/{tag}_eval_meta.pkl")
    np.savez(f"{MLWX}/{tag}_P.npz", **{k.split()[0]: v for k, v in Pall.items()})


if __name__ == "__main__":
    a = sys.argv[1:]
    main(freq=a[0] if a else "M", min_train=int(a[1]) if len(a) > 1 else 3000)
