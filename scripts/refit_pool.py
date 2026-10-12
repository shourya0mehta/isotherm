"""Weekly pool refit during paper/live trading (pre-registered: scheduled refits allowed, design frozen).

Window: the 90 days ending two days before today. Rows older than the bot's own record come from the shipped
research rows (wxbot/ml/artifacts/pool_rows_base.json.gz, exported at the freeze from the frozen Tier-1's walk-forward
ladders); newer rows come from the bot's pre-registered snapshots (ml_snapshots: market and Tier-1 ladders at each
decision hour) for events Kalshi has settled, with bracket bounds and results from Kalshi's public API.
Writes wxbot/ml/artifacts/pool_weights.json (previous copy kept as .prev). The bot runs the same refit by itself
every Monday after 06:00 UTC (wxbot/app/runner.py); this script is the manual version.
Usage: python3 scripts/refit_pool.py [db_path] [--dry-run]
"""
import datetime as dt
import gzip
import json
import os
import sys
import time

import requests

sys.path.insert(0, ".")
from wxbot.ml.pool_fit import refit, write_weights  # noqa: E402

API = "https://api.elections.kalshi.com/trade-api/v2"
ART = "wxbot/ml/artifacts"


def event_markets(ev, cache):
    if ev in cache:
        return cache[ev]
    for k in range(5):
        try:
            r = requests.get(f"{API}/markets", params={"event_ticker": ev, "limit": 100}, timeout=30)
            if r.status_code == 200:
                cache[ev] = r.json().get("markets", [])
                time.sleep(0.25)
                return cache[ev]
            if r.status_code == 429:
                time.sleep(3 * (k + 1))
                continue
        except requests.RequestException:
            time.sleep(2)
    cache[ev] = []
    return []


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    db = args[0] if args else "data/wxbot.sqlite"
    today = dt.datetime.now(dt.timezone.utc).date()
    base_path = os.path.join(ART, "pool_rows_base.json.gz")
    base = json.load(gzip.open(base_path)) if os.path.exists(base_path) else []
    cache = {}
    sha = json.load(open(os.path.join(ART, "tier1_v2_meta.json"))).get("sha256", "")[:12] or None
    out, msg = refit(db, base, today, lambda ev: event_markets(ev, cache), model_sha=sha)
    print(msg)
    if out is None:
        return
    path = os.path.join(ART, "pool_weights.json")
    old = json.load(open(path)) if os.path.exists(path) else None
    for k in ("alpha", "beta"):
        print(k, "new", out[k], "old", old.get(k) if old else None)
    if "--dry-run" in sys.argv:
        return
    write_weights(out, path)
    print("wrote", path, "(previous kept as pool_weights.json.prev); the bot reloads it at its next weekly check "
          "or restart")


if __name__ == "__main__":
    main()
