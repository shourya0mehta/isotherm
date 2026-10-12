"""Maker replay, three arms on the Aug-Sep 2026 tape (all 24 cities, joint capital-constrained simulator):
  A market-anchored (the bot's v0 fair value, no observations: what earlier replays used)
  B market-anchored + live METAR kills (the live bot's actual logic: brackets below the observed max are dead)
  C pooled (market x Tier-1 v2, hour-specific weights fit out of sample) + the same METAR kills
Pool weights: August uses events before Aug 1, September uses events before Sep 1."""
import datetime as dt, glob, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from build_tier2 import SER2ICAO, event_date, series_of
from pool_replay import PoolFV, fit_pool
from tier2_eval_v2 import prep
from wxbot.backtest.joint import run_joint
from wxbot.ml import features as F
from wxbot.models import fair_value as fvmod
from wxbot.risk.limits import RiskConfig
from wxbot.strategy.sleeve_a import SleeveAConfig
MLWX = "/home/claude/mlwx"; EV, CD = "/home/claude/research/ev", "/home/claude/research/cd"

src = F.RawSource(f"{MLWX}/raw")
o = src.obs(); o = o[(o["valid"] >= "2026-07-25") & o["t_prec"].notna()].copy()
OBS = {}
for st, g in o.groupby("station"):
    off = int(F.HIGH["K" + st]["lst_offset_h"]) if ("K" + st) in F.HIGH else None
    if off is None:
        continue
    g = g.sort_values("valid")
    lday = F.localday(g["valid"], off)
    OBS["K" + st] = (g["valid"].values.astype("datetime64[s]").astype(np.int64), g["t_prec"].values, lday.values)


def obs_max(icao, D, t):
    a = OBS.get(icao)
    if a is None:
        return None
    ts, tp, ld = a
    i = np.searchsorted(ts, t - 300, side="right")
    sel = (ld[:i] == np.datetime64(D))
    return float(tp[:i][sel].max()) if sel.any() else None


def with_obs(name, t, brackets, quotes, fv):
    D = event_date(name); icao = SER2ICAO.get(series_of(name))
    om = obs_max(icao, D, t) if icao else None
    return fvmod.ladder_fair_value(brackets, quotes, kind="high", hours_to_close=24, obs_max_f=om)


class PoolObs(PoolFV):
    def __call__(self, name, t, brackets, quotes, fv):
        fv2 = with_obs(name, t, brackets, quotes, fv)
        out = super().__call__(name, t, brackets, quotes, fv2)
        if out is fv2:
            return fv2
        probs = dict(out.probs)
        for tk in fv2.dead:
            probs[tk] = 0.0
        s = sum(probs.values()) or 1.0
        out.probs = {k: v / s for k, v in probs.items()}
        out.conflict = fv2.conflict
        return out


def main():
    df = prep(os.environ.get("T2PKL", f"{MLWX}/tier2_v2.pkl"))
    t1 = pd.read_parquet(os.environ.get("T1OOS", f"{MLWX}/tier1_v2_oos.parquet"))
    t1["key"] = t1["icao"] + "|" + t1["D"].dt.strftime("%Y-%m-%d") + "|" + t1["h"].astype(int).astype(str)
    aug = [f for f in sorted(glob.glob(f"{EV}/KXHIGH*-26AUG*.json")) if os.path.getsize(f) > 10]
    sep = [f for f in sorted(glob.glob(f"{EV}/KXHIGH*-26SEP*.json")) if os.path.getsize(f) > 10]
    W = {"Aug": fit_pool(df, "2026-08-01"), "Sep": fit_pool(df, "2026-09-01")}
    risk = RiskConfig(max_loss_per_bracket_usd=30, max_loss_per_event_usd=40, max_total_open_risk_usd=350)
    cfgs = {"live config (decided-only)": SleeveAConfig(decided_only=True, decided_prob=0.97, min_edge_c=2.0,
                                                         decided_max_cost_usd=40, improve_if_edge_c=99),
            "all favorites p>=0.78, edge>=2c": SleeveAConfig(min_edge_c=2.0, decided_max_cost_usd=40)}
    out = {}
    for cname, cfg in cfgs.items():
        for label, files in (("Aug", aug), ("Sep", sep)):
            for arm, hook in (("A market", None), ("B market+obs", with_obs), ("C pooled+obs", PoolObs(W[label], t1))):
                t0 = time.time()
                r = run_joint(files, CD, cfg_a=cfg, risk=risk, cash_cap=560, netting=False, fv_fn=hook)
                print(f"{cname:32s} {label} {arm:14s}: ${r['realized']:8.2f} (${r['realized'] / max(r['days'], 1):5.2f}/day) "
                      f"up {r['up_days']:2d}/down {r['down_days']:2d} maxDD ${r['max_dd']:4.0f} contracts {r['contracts']:6.0f} "
                      f"[{time.time() - t0:.0f}s]", flush=True)
                out[f"{cname}|{label}|{arm}"] = {k: v for k, v in r.items() if k != "daily"}
                out[f"{cname}|{label}|{arm}"]["daily"] = r["daily"]
    json.dump(out, open(os.environ.get("REPLAY_OUT", f"{MLWX}/pool_replay2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
