"""Paper check of a forecaster artifact before it goes live (RUNBOOK "Stage 3", step 3): the live forecaster code
(wxbot.ml.live.Tier1Live) predicts every station's next settlement day at the latest decision hour with the shipped
model and with the candidate, each in its own process, on a copy of the live bot's raw folder (the predict call
refreshes it the way the bot does).

Reported (no outcome is read, so holdout days are fine): both parity checks; the candidate's inputs that the shipped
model lacks, with their coverage and range on the target rows; the implied means of the two models side by side;
wall time and peak memory of each predict call.
Usage: python3 scripts/model_check.py RAW_DIR CANDIDATE [BASE]      (BASE defaults to tier1_v2)
       python3 scripts/model_check.py --one RAW_DIR MODEL OUT.json  (internal: one model, one process)
"""
import datetime as dt
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def one(raw, model, out):
    import pandas as pd
    from wxbot.ml import features as F
    from wxbot.ml.live import Tier1Live
    t0 = time.time()
    t = Tier1Live(raw, sorted(F.HIGH), model=model)
    t_load = time.time() - t0
    now = dt.datetime.now(dt.timezone.utc)
    today = now.date()
    targets = [(s, d) for s in sorted(F.HIGH) for d in (today - dt.timedelta(days=1), today, today + dt.timedelta(days=1))]
    t1 = time.time()
    res = t.predict(targets, now)
    t_pred = time.time() - t1
    K = F.K
    rows = []
    for (icao, D), r in sorted(res.items()):
        P = np.asarray(r["P"])
        k = np.arange(len(P)) - K + r["anchor_r"]
        mu = float((P * k).sum())
        rows.append({"icao": icao, "D": str(D), "h": r["h"], "mean": round(mu, 3),
                     "sd": round(float(np.sqrt((P * (k - mu) ** 2).sum())), 3), "anchor_r": r["anchor_r"],
                     "nbs_txn": r["nbs_txn"], "obs_max": r["obs_max"]})
    x = getattr(t, "last_inputs", None)
    cov = {}
    if x is not None:
        for c in t.meta["feats"]:
            if c == "st":
                continue
            v = pd.to_numeric(x[c], errors="coerce")
            cov[c] = {"have": round(float(v.notna().mean()), 3),
                      "min": None if v.notna().sum() == 0 else round(float(v.min()), 2),
                      "max": None if v.notna().sum() == 0 else round(float(v.max()), 2)}
    json.dump({"model": model, "parity": t.parity_max_diff, "engine": t.engine, "load_s": round(t_load, 1),
               "predict_s": round(t_pred, 1), "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
               "feats": t.meta["feats"], "coverage": cov, "rows": rows, "now": now.isoformat()}, open(out, "w"))


def main(raw, cand, base="tier1_v2"):
    res = {}
    for m in (base, cand):
        out = f"/tmp/model_check_{m}.json"
        r = subprocess.run([sys.executable, __file__, "--one", raw, m, out], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"{m}: FAILED\n{r.stderr[-3000:]}")
            return 1
        res[m] = json.load(open(out))
    b, c = res[base], res[cand]
    print(f"decision time {c['now']}")
    for m in (b, c):
        print(f"  {m['model']:12s} parity {m['parity']:.1e} ({m['engine']}), load {m['load_s']}s, predict {m['predict_s']}s"
              f" (refresh included), peak {m['peak_mb']} MB")
    new = [f for f in c["feats"] if f not in b["feats"]]
    print(f"\ninputs {cand} adds ({len(new)}): coverage on the target rows, min, max")
    for f in new:
        v = c["coverage"].get(f, {})
        print(f"  {f:12s} {v.get('have')}  {v.get('min')}  {v.get('max')}")
    low = [f for f in c["feats"] if f != "st" and f not in new and c["coverage"].get(f, {}).get("have", 1) <
           b["coverage"].get(f, {}).get("have", 1)]
    if low:
        print("shared inputs with lower coverage than the base model:", low)
    bm = {(r["icao"], r["D"]): r for r in b["rows"]}
    print(f"\n{'station':8s} {'D':10s} {'h':>3s}  {'base mean':>9s} {'cand mean':>9s}  {'diff':>6s}  {'base sd':>7s} "
          f"{'cand sd':>7s}  NBM")
    diffs = []
    for r in c["rows"]:
        q = bm.get((r["icao"], r["D"]))
        if not q:
            continue
        d = r["mean"] - q["mean"]
        diffs.append(d)
        print(f"{r['icao']:8s} {r['D']:10s} {r['h']:+3d}  {q['mean']:9.2f} {r['mean']:9.2f}  {d:+6.2f}  {q['sd']:7.2f} "
              f"{r['sd']:7.2f}  {r['nbs_txn']}")
    if diffs:
        print(f"\nmean |candidate - base| {np.mean(np.abs(diffs)):.2f} F over {len(diffs)} rows; largest "
              f"{max(diffs, key=abs):+.2f} F")
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        one(*sys.argv[2:5])
    else:
        sys.exit(main(*sys.argv[1:]))
