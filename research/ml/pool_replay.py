"""Economic test of the pooled fair value: replay the live maker (Sleeve A) on the Aug-Sep 2026 tape, all 24 cities,
with the joint capital-constrained simulator, under
  (a) the bot's market-anchored fair value (baseline), and
  (b) the pool of the live market ladder with the walk-forward Tier-1 v2 forecaster, weights fit per decision hour.
Out of sample: August is replayed with pool weights fit on events before Aug 1; September with weights fit on events
before Sep 1 (both after a 2-day settlement gap). The market term is re-read every replay step; Tier-1 updates at
the decision hours only (as it would live).
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from build_tier2 import SER2ICAO, event_date, series_of  # noqa: E402
from cgep import CGEP3, ladder_arrays  # noqa: E402
from tier2_eval_v2 import HOURS, design, prep  # noqa: E402
from wxbot.backtest.joint import run_joint  # noqa: E402
from wxbot.models.fair_value import FairValueResult  # noqa: E402
from wxbot.risk.limits import RiskConfig  # noqa: E402
from wxbot.strategy.sleeve_a import SleeveAConfig  # noqa: E402

MLWX = os.environ.get("MLWX", "/home/claude/mlwx")
EV, CD = "/home/claude/research/ev", "/home/claude/research/cd"
EPS = 1e-4


def fit_pool(df, cutoff, exp=("m", "f")):
    tr = df[df["D"] <= pd.Timestamp(cutoff) - pd.Timedelta(days=2)]
    rows = tr.to_dict("records")
    nmax = max(len(r["m"]) for r in rows)
    LM, LF, Z, MASK, WIN = ladder_arrays(rows, nmax)
    X, _, _ = design(tr, [])
    mdl = CGEP3(len(exp), tilt=True, curv=True, pen=np.ones(len(HOURS)), alpha0=np.ones(len(HOURS)))
    mdl.fit(X, [LM, LF][:len(exp)], Z, MASK, WIN)
    W = mdl.w.reshape(-1, len(HOURS))  # rows: alpha, beta, theta, kappa ; cols: hours
    print("pool fit on", len(tr), "snapshots before", cutoff, {f"h{h}": np.round(W[:, i], 3).tolist() for i, h in enumerate(HOURS)})
    return W


class PoolFV:
    def __init__(self, W, t1):
        self.W = W
        self.t1 = t1
        pc = [c for c in t1.columns if c.startswith("p") and c[1:].isdigit()]
        self.K = (len(pc) - 1) // 2
        self.P = t1[pc].values
        self.AR = t1["anchor_r"].values
        self.pos = {k: i for i, k in enumerate(t1["key"])}
        self.used = 0
        self.skipped = 0
        self.delay_s = 15 * 60

    def __call__(self, name, t, brackets, quotes, fv):
        D = event_date(name)
        icao = SER2ICAO.get(series_of(name))
        if D is None or icao is None:
            return fv
        d0 = dt.datetime(D.year, D.month, D.day, tzinfo=dt.timezone.utc).timestamp()
        hs = [h for h in HOURS if d0 + h * 3600 + self.delay_s <= t]  # live needs minutes to fetch and predict
        if not hs:
            self.skipped += 1
            return fv
        h = hs[-1]
        i = self.pos.get(f"{icao}|{D.isoformat()}|{h}")
        if i is None:
            self.skipped += 1
            return fv
        P, ar, K = self.P[i], self.AR[i], self.K
        m = np.array([fv.market.get(b.ticker, 0.005) for b in brackets], float)
        m = np.clip(m, EPS, None); m /= m.sum()
        f = []
        for b in brackets:
            lo, hi = max(b.lo, ar - K), min(b.hi, ar + K)
            js = [int(v - ar + K) for v in range(int(np.ceil(lo)), int(np.floor(hi)) + 1)]
            f.append(float(P[js].sum()) if js else 0.0)
        f = np.clip(np.array(f), EPS, None); f /= f.sum()
        lo = np.array([b.lo for b in brackets], float); hi = np.array([b.hi for b in brackets], float)
        c = np.where(lo < -1e5, hi - 1.0, np.where(hi > 1e5, lo + 1.0, (lo + hi) / 2.0))
        z = (c - (m * c).sum()) / 2.0
        k = HOURS.index(h)
        a, bta, th, ka = self.W[:, k]
        s = a * np.log(m) + bta * np.log(f) + th * z + ka * z * z
        p = np.exp(s - s.max()); p /= p.sum()
        self.used += 1
        probs = {b.ticker: float(v) for b, v in zip(brackets, p)}
        for tk in fv.dead:
            probs[tk] = 0.0
        return FairValueResult(probs, fv.market, fv.nbm, fv.dead, fv.disagreement, fv.notes, fv.conflict)


def main():
    df = prep(f"{MLWX}/tier2_v2.pkl")
    t1 = pd.read_parquet(f"{MLWX}/tier1_v2_oos.parquet")
    t1["key"] = t1["icao"] + "|" + t1["D"].dt.strftime("%Y-%m-%d") + "|" + t1["h"].astype(int).astype(str)
    aug = sorted(glob.glob(f"{EV}/KXHIGH*-26AUG*.json"))
    sep = sorted(glob.glob(f"{EV}/KXHIGH*-26SEP*.json"))
    aug = [f for f in aug if os.path.getsize(f) > 10]
    sep = [f for f in sep if os.path.getsize(f) > 10]
    W_aug = fit_pool(df, "2026-08-01")
    W_sep = fit_pool(df, "2026-09-01")
    risk = RiskConfig(max_loss_per_bracket_usd=30, max_loss_per_event_usd=40, max_total_open_risk_usd=350)
    cfgs = {
        "live config (decided-only favorites)": SleeveAConfig(decided_only=True, decided_prob=0.97, min_edge_c=2.0,
                                                               decided_max_cost_usd=40, improve_if_edge_c=99),
        "all favorites p>=0.78": SleeveAConfig(min_edge_c=2.0, decided_max_cost_usd=40),
        "all favorites, edge>=4c": SleeveAConfig(min_edge_c=4.0, decided_max_cost_usd=40),
    }
    out = {}
    for cname, cfg in cfgs.items():
        for label, files, W in (("Aug", aug, W_aug), ("Sep", sep, W_sep)):
            for fvname, hook in (("market-anchored", None), ("pooled (market + Tier-1 v2)", PoolFV(W, t1))):
                t0 = time.time()
                r = run_joint(files, CD, cfg_a=cfg, risk=risk, cash_cap=560, netting=False, fv_fn=hook)
                extra = f" pool used {hook.used} skipped {hook.skipped}" if hook else ""
                print(f"{cname:38s} {label} {fvname:28s}: ${r['realized']:8.2f} (${r['realized'] / max(r['days'], 1):.2f}/day) "
                      f"up {r['up_days']}/down {r['down_days']} maxDD ${r['max_dd']:.0f} contracts {r['contracts']:.0f} "
                      f"[{time.time() - t0:.0f}s]{extra}", flush=True)
                out[f"{cname}|{label}|{fvname}"] = {k: v for k, v in r.items() if k != "daily"}
    json.dump(out, open(f"{MLWX}/pool_replay.json", "w"), indent=1)


if __name__ == "__main__":
    main()
