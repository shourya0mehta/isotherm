"""Pre-registered ML scorecard: for every settled event the bot snapshotted, log loss of the market ladder, the
Tier-1 forecaster and the pool, scored against Kalshi's settled result. Rules fixed in advance (docs/ML_RESEARCH.md):
- one snapshot per (event, decision hour), written the first time the pool is applied (no edits afterwards)
- market = normalized mids at that moment; score = -log p(winning bracket), floored at 1e-4
- headline = pooled minus market, mean over snapshots, with a day-block bootstrap 95% interval
Usage: python3 scripts/ml_scorecard.py [db_path]
"""
import json, math, os, sqlite3, sys, time, urllib.request
from collections import defaultdict
import numpy as np

DB = next((a for a in sys.argv[1:] if not a.startswith("--")), "data/wxbot.sqlite")
API = "https://api.elections.kalshi.com/trade-api/v2"


def results(events):
    out = {}
    for ev in events:
        try:
            with urllib.request.urlopen(f"{API}/markets?event_ticker={ev}&limit=50", timeout=30) as r:
                ms = json.load(r).get("markets", [])
        except Exception:
            continue
        res = {m["ticker"]: m.get("result") for m in ms}
        if res and all(v in ("yes", "no") for v in res.values()) and sum(v == "yes" for v in res.values()) == 1:
            out[ev] = res
        time.sleep(0.3)
    return out


def main():
    c = sqlite3.connect(DB)
    sha = None
    if "--all-models" not in sys.argv:  # default: the model now deployed (the frozen one after Oct 11)
        meta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wxbot", "ml", "artifacts",
                            "tier1_v2_meta.json")
        sha = json.load(open(meta)).get("sha256", "")[:12] if os.path.exists(meta) else None
    rows = c.execute("SELECT event, h, ticker, market, tier1, pooled, model_sha FROM ml_snapshots").fetchall()
    snaps = defaultdict(list)
    for ev, h, t, m, f, p, ms in rows:
        if sha and ms != sha:
            continue
        snaps[(ev, h)].append((t, m or 0.0, f or 0.0, p or 0.0))
    print(f"model {sha or 'all'}")
    res = results(sorted({ev for ev, _ in snaps}))
    recs = []
    for (ev, h), br in snaps.items():
        if ev not in res:
            continue
        def ll(i):
            x = np.clip(np.array([b[i] for b in br], float), 1e-4, None)
            x /= x.sum()
            w = [k for k, b in enumerate(br) if res[ev].get(b[0]) == "yes"]
            return -math.log(max(x[w[0]], 1e-4)) if w else None
        lm, lf, lp = ll(1), ll(2), ll(3)
        if lm is None:
            continue
        recs.append((ev.split("-")[1], h, lm, lf, lp, ev))
    if not recs:
        print("no settled snapshots yet")
        return
    days = np.array([r[0] for r in recs]); lm = np.array([r[2] for r in recs]); lf = np.array([r[3] for r in recs])
    lp = np.array([r[4] for r in recs])
    d = lp - lm
    ud = np.unique(days)
    rng = np.random.default_rng(0)
    sums = np.array([d[days == u].sum() for u in ud]); cnt = np.array([(days == u).sum() for u in ud])
    bs = [sums[k].sum() / cnt[k].sum() for k in (rng.integers(0, len(ud), len(ud)) for _ in range(2000))]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"snapshots {len(recs)}, events {len({r[5] for r in recs})}, days {len(ud)}")
    print(f"log loss  market {lm.mean():.4f}  tier1 {lf.mean():.4f}  pooled {lp.mean():.4f}")
    print(f"pooled - market {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  (negative = we beat the market)")
    for h in sorted({r[1] for r in recs}):
        s = np.array([r[1] == h for r in recs])
        print(f"  h={h:+d}: n={s.sum():4d}  pooled-market {d[s].mean():+.4f}")
    ev_ids = np.array([r[5] for r in recs])
    pe = defaultdict(list)
    for e, x in zip(ev_ids, d):
        pe[e].append(x)
    pev = np.array([np.mean(v) for v in pe.values()])
    print(f"per event: {len(pev)} events, pooled - market {pev.mean():+.4f} (the Dec 1 holdout uses per-event scores)")


if __name__ == "__main__":
    main()
