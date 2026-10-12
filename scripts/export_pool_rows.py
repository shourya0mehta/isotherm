"""Export the research ladders the weekly pool refit starts from (run at the freeze, from the frozen Tier-1's
walk-forward tier-2 rows). Writes wxbot/ml/artifacts/pool_rows_base.json.gz: market and model ladders, bracket
bounds, winner, decision hour and date for every scored snapshot from Jul 1 to Sep 30, 2026.
Usage: T2PKL=/home/claude/mlwx/tier2_v3.pkl python3 scripts/export_pool_rows.py
Candidate B's early hours (wxbot/app/early_shadow.py refit): T2PKL=/home/claude/mlwx/t2_early_new_hrrr_r2.pkl HOURS=-9,-6
  OUT=wxbot/ml/artifacts/pool_rows_base_r2b.json.gz python3 scripts/export_pool_rows.py"""
import gzip
import json
import math
import os
import sys

sys.path.insert(0, "research/ml")
import pandas as pd  # noqa: E402
from tier2_eval_v2 import prep  # noqa: E402

MLWX = os.environ.get("MLWX", "/home/claude/mlwx")
df = prep(os.environ.get("T2PKL", f"{MLWX}/tier2_v3.pkl"))
df = df[(df["D"] >= "2026-07-01") & (df["D"] <= "2026-09-30")]
if os.environ.get("HOURS"):        # e.g. HOURS=-9,-6 for candidate B's early hours
    df = df[df["h"].isin([int(x) for x in os.environ["HOURS"].split(",")])]
clean = lambda v: [None if (x is None or (isinstance(x, float) and not math.isfinite(x))) else round(float(x), 6) for x in v]
rows = [{"m": clean(r["m"]), "f": clean(r["f"]), "lo": [float(max(min(x, 1e9), -1e9)) for x in r["lo"]],
         "hi": [float(max(min(x, 1e9), -1e9)) for x in r["hi"]], "win": int(r["win"]), "h": int(r["h"]),
         "D": str(pd.Timestamp(r["D"]).date()), "src": "research"} for r in df.to_dict("records")]
path = os.environ.get("OUT", "wxbot/ml/artifacts/pool_rows_base.json.gz")
with gzip.open(path, "wt") as f:
    json.dump(rows, f)
print(len(rows), "rows ->", path, os.path.getsize(path) // 1024, "KB")
