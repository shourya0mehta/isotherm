"""Backfill the pool's disagreement context (ml_snap_ctx) by hand; the bot's weekly refit does the same for its own
database when its pool has the NBM-departure tilt (wxbot.ml.pool_fit.backfill_ctx).
Usage: python3 scripts/backfill_ml_ctx.py DB [RAW_DIR]   (RAW_DIR default data/ml_raw)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from wxbot.ml.pool_fit import backfill_ctx  # noqa: E402

if __name__ == "__main__":
    db, raw = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "data/ml_raw")
    print(backfill_ctx(db, raw), "snapshot hours filled")
