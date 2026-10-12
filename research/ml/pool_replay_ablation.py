"""Ablations for the maker replay (Aug-Sep 2026 tape, same simulator, cutoffs and METAR kills as pool_replay2.py):
  D recal+obs  = the pool machinery with the model weight forced to zero (market sharpened per hour + tilt/curvature):
                 is the gain from the forecaster or from recalibrating the market?
  E pool90+obs = the production pool form fit only on the 90 days before each cutoff (tracks a sharpening market).
Compare with B (market + obs) and C (pool fit on all history + obs). Output: REPLAY_OUT (pool_replay3_ablation.json)."""
import glob, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from pool_replay import fit_pool
from pool_replay2 import CD, EV, MLWX, PoolObs
from tier2_eval_v2 import prep
from wxbot.backtest.joint import run_joint
from wxbot.risk.limits import RiskConfig
from wxbot.strategy.sleeve_a import SleeveAConfig


def market_only(df, cutoff):
    W = fit_pool(df, cutoff, exp=("m",))  # rows: alpha, theta, kappa
    return np.vstack([W[0], np.zeros(W.shape[1]), W[1], W[2]])  # beta = 0: the model has no say


def recent(df, cutoff, days=90):
    return df[df["D"] > pd.Timestamp(cutoff) - pd.Timedelta(days=days + 2)]


def main():
    df = prep(os.environ.get("T2PKL", f"{MLWX}/tier2_v3.pkl"))
    t1 = pd.read_parquet(os.environ.get("T1OOS", f"{MLWX}/tier1_v3_oos.parquet"))
    t1["key"] = t1["icao"] + "|" + t1["D"].dt.strftime("%Y-%m-%d") + "|" + t1["h"].astype(int).astype(str)
    aug = [f for f in sorted(glob.glob(f"{EV}/KXHIGH*-26AUG*.json")) if os.path.getsize(f) > 10]
    sep = [f for f in sorted(glob.glob(f"{EV}/KXHIGH*-26SEP*.json")) if os.path.getsize(f) > 10]
    WD = {"Aug": market_only(df, "2026-08-01"), "Sep": market_only(df, "2026-09-01")}
    WE = {"Aug": fit_pool(recent(df, "2026-08-01"), "2026-08-01"), "Sep": fit_pool(recent(df, "2026-09-01"), "2026-09-01")}
    risk = RiskConfig(max_loss_per_bracket_usd=30, max_loss_per_event_usd=40, max_total_open_risk_usd=350)
    cfgs = {"live config (decided-only)": SleeveAConfig(decided_only=True, decided_prob=0.97, min_edge_c=2.0,
                                                         decided_max_cost_usd=40, improve_if_edge_c=99),
            "all favorites p>=0.78, edge>=2c": SleeveAConfig(min_edge_c=2.0, decided_max_cost_usd=40)}
    out = {}
    runs = [(c, m, a) for c in cfgs for m in ("Aug", "Sep") for a in ("D recal+obs", "E pool90+obs")]
    for cname, label, arm in runs:
        files, W = {"Aug": aug, "Sep": sep}[label], {"D recal+obs": WD, "E pool90+obs": WE}[arm]
        t0 = time.time()
        r = run_joint(files, CD, cfg_a=cfgs[cname], risk=risk, cash_cap=560, netting=False, fv_fn=PoolObs(W[label], t1))
        print(f"{cname:32s} {label} {arm:14s}: ${r['realized']:8.2f} (${r['realized'] / max(r['days'], 1):5.2f}/day) "
              f"up {r['up_days']:2d}/down {r['down_days']:2d} maxDD ${r['max_dd']:4.0f} contracts {r['contracts']:6.0f} "
              f"[{time.time() - t0:.0f}s]", flush=True)
        out[f"{cname}|{label}|{arm}"] = {k: v for k, v in r.items() if k != "daily"}
        out[f"{cname}|{label}|{arm}"]["daily"] = r["daily"]
    json.dump(out, open(os.environ.get("REPLAY_OUT", f"{MLWX}/pool_replay3_ablation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
