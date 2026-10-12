"""Fill stress tests for the maker replay (Aug-Sep 2026 tape, joint capital-constrained simulator, METAR kills).

Pre-registered in the review doc (Section 7): trading-side results must hold under worse fills before money moves.
Fill rules:
  q300     baseline: a join waits behind 300 contracts at its price; fills only from later public trades
  q1500    five times the queue: a join waits behind 1,500 contracts
  through  only trades at a strictly better price than ours fill us (we would certainly have been hit first)
Arms: E = production pool (market x Tier-1 v3, weights fit on the 90 days before each month) + METAR kills,
      B = market-anchored + METAR kills (no model).
Thin sleeve: the four newest listings (Trenton, San Diego, Newark, Louisville; liquid share of snapshots 25-37% in
Aug-Sep vs 85-100% elsewhere), favorites with edge >= 3c, stepping inside the spread when edge >= 5c, $100 cap.
Output: STRESS_OUT (default mlwx/fill_stress.json), written after every run so partial results survive.
"""
import glob
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from pool_replay import fit_pool  # noqa: E402
from pool_replay2 import CD, EV, MLWX, PoolObs, with_obs  # noqa: E402
from pool_replay_ablation import recent  # noqa: E402
from tier2_eval_v2 import prep  # noqa: E402
from wxbot.backtest.joint import combine_daily, daily_ci, max_drawdown, run_joint  # noqa: E402,F401  (daily_ci: here until Oct 9)
from wxbot.risk.limits import RiskConfig  # noqa: E402
from wxbot.strategy.sleeve_a import SleeveAConfig  # noqa: E402

THIN = {"KXHIGHTTTN", "KXHIGHTSAN", "KXHIGHTEWR", "KXHIGHTSDF"}
OUT = os.environ.get("STRESS_OUT", f"{MLWX}/fill_stress.json")
RULES = {"q300": (300.0, "queue"), "q1500": (1500.0, "queue"), "through": (300.0, "through")}


def main():
    df = prep(os.environ.get("T2PKL", f"{MLWX}/tier2_v3.pkl"))
    t1 = pd.read_parquet(os.environ.get("T1OOS", f"{MLWX}/tier1_v3_oos.parquet"))
    t1["key"] = t1["icao"] + "|" + t1["D"].dt.strftime("%Y-%m-%d") + "|" + t1["h"].astype(int).astype(str)
    files = {m: [f for f in sorted(glob.glob(f"{EV}/KXHIGH*-26{m.upper()}*.json")) if os.path.getsize(f) > 10]
             for m in ("Aug", "Sep")}
    thin_files = {m: [f for f in fs if os.path.basename(f).split("-")[0] in THIN] for m, fs in files.items()}
    WE = {"Aug": fit_pool(recent(df, "2026-08-01"), "2026-08-01"), "Sep": fit_pool(recent(df, "2026-09-01"), "2026-09-01")}
    risk = RiskConfig(max_loss_per_bracket_usd=30, max_loss_per_event_usd=40, max_total_open_risk_usd=350)
    risk_thin = RiskConfig(max_loss_per_bracket_usd=15, max_loss_per_event_usd=25, max_total_open_risk_usd=100)
    live = SleeveAConfig(decided_only=True, decided_prob=0.97, min_edge_c=2.0, decided_max_cost_usd=40, improve_if_edge_c=99)
    favs = SleeveAConfig(min_edge_c=2.0, decided_max_cost_usd=40)
    thin = SleeveAConfig(min_edge_c=3.0, improve_if_edge_c=5.0, decided_max_cost_usd=25)
    plan = []
    rules = os.environ.get("STRESS_RULES", "q300,q1500,through").split(",")
    for rule in rules:
        plan += [("live", m, "E", rule) for m in ("Aug", "Sep")]
    for rule in ("q300", "q1500", "through"):
        plan += [("thin sleeve", m, "E", rule) for m in ("Aug", "Sep")]
    for rule in ("q300", "q1500", "through"):
        plan += [("live", m, "B", rule) for m in ("Aug", "Sep")]
    for rule in ("q300", "q1500", "through"):
        plan += [("all favorites", m, "E", rule) for m in ("Aug", "Sep")]
    only = os.environ.get("STRESS_ONLY")  # e.g. "live|E": run just those (config|arm) combinations
    if only:
        plan = [p for p in plan if f"{p[0]}|{p[2]}" in only.split(",")]
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for cname, month, arm, rule in plan:
        key = f"{cname}|{month}|{arm}|{rule}"
        if key in out:
            continue
        cfg = {"live": live, "all favorites": favs, "thin sleeve": thin}[cname]
        fs = thin_files[month] if cname == "thin sleeve" else files[month]
        hook = PoolObs(WE[month], t1) if arm == "E" else with_obs
        qa, fr = RULES[rule]
        t0 = time.time()
        r = run_joint(fs, CD, cfg_a=cfg, risk=risk_thin if cname == "thin sleeve" else risk,
                      cash_cap=100 if cname == "thin sleeve" else 560, netting=False, fv_fn=hook,
                      queue_ahead=qa, fill_rule=fr, topup=os.environ.get("TOPUP", "1") == "1",
                      resize=os.environ.get("RESIZE", "1") == "1")
        be = r["by_event"]
        thin_pnl = sum(v for k, v in be.items() if k.split("-")[0] in THIN)
        print(f"{cname:14s} {month} {arm} {rule:8s}: ${r['realized']:8.2f} (${r['realized'] / max(r['days'], 1):5.2f}/day) "
              f"up {r['up_days']:2d}/down {r['down_days']:2d} maxDD ${r['max_dd']:4.0f} contracts {r['contracts']:6.0f} "
              f"thin ${thin_pnl:7.2f} [{time.time() - t0:.0f}s]", flush=True)
        out[key] = {k: v for k, v in r.items()}
        json.dump(out, open(OUT + ".tmp", "w"), indent=1)
        os.replace(OUT + ".tmp", OUT)
    summarize(out)


def summarize(out):
    print("\nTwo-month totals (Aug + Sep 2026), mean daily P&L with day-bootstrap 95% CI")
    keys = sorted({k.rsplit("|", 3)[0] + "|" + k.split("|")[2] + "|" + k.split("|")[3] for k in out})
    for k in keys:
        cname, arm, rule = k.split("|")
        rs = [out.get(f"{cname}|{m}|{arm}|{rule}") for m in ("Aug", "Sep")]
        if any(r is None for r in rs):
            continue
        daily = combine_daily(*(r["daily"] for r in rs))
        tot = sum(r["realized"] for r in rs)
        mu, lo, hi = daily_ci(daily)
        be = {}
        for r in rs:
            be.update(r["by_event"])
        thin = sum(v for e, v in be.items() if e.split("-")[0] in THIN)
        # max drawdown of the two months' combined daily path (Oct 9: it was the larger month's own drawdown)
        print(f"  {cname:14s} {arm} {rule:8s}: ${tot:8.2f}  ${mu:6.2f}/day [{lo:6.2f},{hi:6.2f}]  "
              f"maxDD ${max_drawdown(daily):4.0f}  contracts {sum(r['contracts'] for r in rs):7.0f}  "
              f"thin-listing share ${thin:7.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarize(json.load(open(OUT)))
    else:
        main()
