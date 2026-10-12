"""Copy the paper bot's forecaster snapshots (table ml_snapshots: the market's and the forecaster's ladders at each
decision hour) into a small database for the server, where the live bot's weekly pool refit reads it as
data/wxbot.sqlite beside data/live.sqlite (holdout amendment A8: the refit reads the paper and the live record, so its
window has no hole at the switch to real money). The paper bot runs on the Mac and the live bot on the server, so
without this the server's window would miss the paper bot's October snapshots.
Only that table is copied (no fills, positions or P&L). The source is opened read-only; the output is written under a
temporary name and moved into place.
Usage: python scripts/export_snapshots.py <paper database> <output database>
  On the Mac (Cowork VM), then copy the output to the server's data/wxbot.sqlite."""
import os
import sqlite3
import sys
from pathlib import Path

COLS = "ts, event, h, ticker, market, tier1, pooled, model_sha"


def export(src: str, dst: str) -> int:
    s = sqlite3.connect(Path(src).resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
    try:
        rows = s.execute(f"SELECT {COLS} FROM ml_snapshots ORDER BY ts").fetchall()
    finally:
        s.close()
    tmp = dst + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    d = sqlite3.connect(tmp)
    d.execute("CREATE TABLE ml_snapshots (ts REAL, event TEXT, h INTEGER, ticker TEXT, market REAL, tier1 REAL, "
              "pooled REAL, model_sha TEXT)")
    d.executemany(f"INSERT INTO ml_snapshots ({COLS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    d.commit()
    d.close()
    os.replace(tmp, dst)
    return len(rows)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    n = export(sys.argv[1], sys.argv[2])
    print(f"{n} snapshots -> {sys.argv[2]}")
