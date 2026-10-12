"""Check the live bot's pooled ladders at one decision hour against the pool recomputed from its own logged inputs
(RUNBOOK "Stage 3"): the logged market mids, forecaster ladder and NBM value (ml_snapshots, ml_snap_ctx) through the
tilted weights and through the old untilted weights. After the Stage 3 switch every ladder should equal the tilted
pool (differences at rounding level) and differ from the old one. Read-only: public market list, no key, no orders.
Usage: python3 scripts/live_pool_check.py DB DAY H     e.g. data/live.sqlite 2026-10-09 3
"""
import datetime as dt
import os
import sqlite3
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from wxbot.app import early_shadow as ES  # noqa: E402
from wxbot.kalshi.rest import PROD, KalshiREST  # noqa: E402
from wxbot.ml.live import ART, Pool  # noqa: E402


def main(db, day, h):
    day, h = dt.date.fromisoformat(day), int(h)
    tag = "-" + day.strftime("%y%b%d").upper()
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = c.execute("SELECT event, ticker, market, tier1, pooled FROM ml_snapshots WHERE h = ? AND event LIKE ?",
                     (h, "%" + tag)).fetchall()
    ctx = {e: n for e, n in c.execute("SELECT event, nbs FROM ml_snap_ctx WHERE h = ? AND event LIKE ?",
                                      (h, "%" + tag))}
    live_w = os.path.join(os.path.dirname(db), "pool_weights_gate.json")
    gated_path = live_w if os.path.exists(live_w) else os.path.join(ART, "pool_weights_gate.json")
    plain, gated = Pool(os.path.join(ART, "pool_weights.json")), Pool(gated_path)
    evs = {e["event"]: e for e in ES.list_events(KalshiREST(base_url=PROD, auth=None, read_rate=5.0), day)}
    by = defaultdict(dict)
    for e, t, m, f, p in rows:
        by[e][t] = (m, f, p)
    dg = dp = 0.0
    n = with_nbm = 0
    for e, d in sorted(by.items()):
        ev = evs.get(e)
        br = [b for b in (ev or {}).get("brackets", []) if b.ticker in d]
        if not ev or len(br) != len(d):
            print(f"  {e}: brackets not matched ({len(br)} of {len(d)})")
            continue
        market = {b.ticker: d[b.ticker][0] for b in br if d[b.ticker][0] is not None}
        f = np.array([d[b.ticker][1] for b in br], float)
        logged = np.array([d[b.ticker][2] for b in br], float)
        pg, pp = gated.probs(br, market, f, h, nbs=ctx.get(e)), plain.probs(br, market, f, h)
        vg = np.array([pg[b.ticker] for b in br])
        vp = np.array([pp[b.ticker] for b in br])
        dg = max(dg, float(np.abs(vg / vg.sum() - logged).max()))
        dp = max(dp, float(np.abs(vp / vp.sum() - logged).max()))
        n += 1
        with_nbm += ctx.get(e) is not None
    print(f"{day} h={h}: {n} events; weights {os.path.relpath(gated_path)}; NBM logged for {with_nbm}; largest gap "
          f"to the tilted pool {dg:.1e}, to the old pool {dp:.3f}")
    ok = n > 0 and dg < 1e-6 and with_nbm == n
    print("LIVE POOL CHECK:", "PASS (the live ladders are the tilted pool)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
