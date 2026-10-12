"""SQLite state: decisions, orders, fills, settlements, P&L, heartbeats. One file, no server."""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS fills (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, mode TEXT, ticker TEXT, event TEXT, side TEXT,
  price_cents INTEGER, count REAL, sleeve TEXT, is_taker INTEGER, fee REAL, order_ref TEXT, fair_yes REAL
);
CREATE TABLE IF NOT EXISTS quotes (
  ts REAL, mode TEXT, ticker TEXT, event TEXT, side TEXT, price_cents INTEGER, count REAL, sleeve TEXT,
  fair_yes REAL, reason TEXT
);
CREATE TABLE IF NOT EXISTS settlements (
  ticker TEXT PRIMARY KEY, event TEXT, result TEXT, settled_ts REAL, expiration_value TEXT
);
CREATE TABLE IF NOT EXISTS fair_values (
  ts REAL, event TEXT, ticker TEXT, fair REAL, market REAL, nbm REAL, disagreement REAL
);
CREATE TABLE IF NOT EXISTS lip_snaps (
  ts REAL, ticker TEXT, counts INTEGER, our_frac REAL, rate_per_hr REAL, est_usd REAL
);
CREATE TABLE IF NOT EXISTS live_settlements (
  ticker TEXT PRIMARY KEY, event TEXT, result TEXT, settled_ts REAL, revenue_usd REAL, cost_usd REAL,
  fee_usd REAL, pnl_usd REAL
);
CREATE TABLE IF NOT EXISTS ml_snapshots (
  ts REAL, event TEXT, h INTEGER, ticker TEXT, market REAL, tier1 REAL, pooled REAL, model_sha TEXT,
  PRIMARY KEY (event, h, ticker)
);
CREATE TABLE IF NOT EXISTS ml_snap_ctx (
  event TEXT, h INTEGER, nbs REAL, ts REAL,
  PRIMARY KEY (event, h)
);
CREATE TABLE IF NOT EXISTS taker_shadow (
  ts REAL, event TEXT, ticker TEXT, side TEXT, price REAL, size REAL, avail REAL, p REAL, edge REAL, h INTEGER,
  model_sha TEXT, spr REAL, msum REAL, result TEXT, settled_ts REAL,
  PRIMARY KEY (event, ticker, side)
);
CREATE TABLE IF NOT EXISTS hourly_shadow (
  ts REAL, event TEXT, ticker TEXT, side TEXT, price INTEGER, count REAL, result TEXT, settled_ts REAL,
  variant TEXT DEFAULT 'H'
);
CREATE TABLE IF NOT EXISTS index_points (city TEXT, t INTEGER, v REAL, PRIMARY KEY (city, t));
CREATE TABLE IF NOT EXISTS rain_shadow (
  ts REAL, event TEXT, ticker TEXT, city TEXT, day TEXT, h INTEGER, side TEXT, price REAL, size REAL, fee REAL,
  p_model REAL, p_pool REAL, mid REAL, edge REAL, model_sha TEXT, result TEXT, settled_ts REAL,
  PRIMARY KEY (ticker, side)
);
CREATE TABLE IF NOT EXISTS rain_shadow_v2 (
  ts REAL, event TEXT, ticker TEXT, city TEXT, day TEXT, h INTEGER, side TEXT, price REAL, size REAL, fee REAL,
  p_model REAL, p_pool REAL, mid REAL, edge REAL, model_sha TEXT, result TEXT, settled_ts REAL,
  PRIMARY KEY (ticker, side)
);
CREATE TABLE IF NOT EXISTS events_log (ts REAL, level TEXT, msg TEXT);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
CREATE INDEX IF NOT EXISTS ix_fills_ticker ON fills(ticker);
CREATE INDEX IF NOT EXISTS ix_quotes_ts ON quotes(ts);
"""

# Shadow sizing (outside review, Oct 8, 2026: the shadow logs assumed fills). Columns added to every shadow table;
# the old columns keep their meaning, so rows from before are read as they always were:
#   intended  the size the rule wanted (taker, rain: the configured size; hourly: the paper order's open size)
#   depth     contracts available at the price or better when it was logged (taker, rain: offered at the signal's
#             price or better; hourly: the trade print's contracts that reached our order). NULL: not known
#   booked    min(intended, depth); 0 when the depth is zero or unknown. Every report that sums a shadow log sums this
SHADOW_SIZE_COLS = ("intended", "depth", "booked")
SHADOW_TABLES = ("taker_shadow", "hourly_shadow", "rain_shadow", "rain_shadow_v2")
# the booked size of a row, for rows from before the columns existed too:
#   taker_shadow logged its displayed size (avail) at the touch, which is the depth at the signal's price, so its old
#   rows book min(size, avail), 0 when nothing was displayed (their `size` then fell back to the full 5);
#   hourly_shadow's count was already capped by the trade print (the paper broker's queue-aware fill);
#   rain rows from before carry no depth: they book 0 (depth unknown)
TAKER_BOOKED_SQL = "COALESCE(booked, CASE WHEN avail > 0 THEN MIN(size, avail) ELSE 0 END)"
HOURLY_BOOKED_SQL = "COALESCE(booked, count)"
RAIN_BOOKED_SQL = "COALESCE(booked, 0)"


class Store:
    def __init__(self, path: str):
        self.path = path
        self.c = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.c.executescript(SCHEMA)
        cols = [r[1] for r in self.c.execute("PRAGMA table_info(hourly_shadow)")]
        if "variant" not in cols:  # databases created before the H2 variant
            self.c.execute("ALTER TABLE hourly_shadow ADD COLUMN variant TEXT DEFAULT 'H'")
        if "model_tag" not in [r[1] for r in self.c.execute("PRAGMA table_info(fills)")]:
            self.c.execute("ALTER TABLE fills ADD COLUMN model_tag TEXT")   # forecaster and pool behind each fill
        for table in SHADOW_TABLES:        # shadow sizing columns (see SHADOW_SIZE_COLS), added to older databases
            have = {r[1] for r in self.c.execute(f"PRAGMA table_info({table})")}
            for col in SHADOW_SIZE_COLS:
                if col not in have:
                    self.c.execute(f"ALTER TABLE {table} ADD COLUMN {col} REAL")
        self.c.execute("CREATE INDEX IF NOT EXISTS ix_fills_event ON fills(event)")
        self.c.commit()

    def log(self, level: str, msg: str):
        self.c.execute("INSERT INTO events_log VALUES (?,?,?)", (time.time(), level, msg[:2000]))
        self.c.commit()

    def add_fill(self, mode: str, f, fair_yes: Optional[float] = None, tag: Optional[str] = None) -> bool:
        """Live fills carry Kalshi's fill_id in order_ref: a fill seen twice (e.g. across a restart) is skipped.
        tag: the forecaster and pool version in use (model_tag column)."""
        if mode == "live" and f.order_ref and self.c.execute(
                "SELECT 1 FROM fills WHERE mode='live' AND order_ref=? LIMIT 1", (f.order_ref,)).fetchone():
            return False
        self.c.execute("INSERT INTO fills (ts,mode,ticker,event,side,price_cents,count,sleeve,is_taker,fee,order_ref,"
                       "fair_yes,model_tag) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (f.ts, mode, f.ticker, f.event, f.side, f.price_cents, f.count, f.sleeve, int(f.is_taker),
                        f.fee, f.order_ref, fair_yes, tag))
        self.c.commit()
        return True

    def add_quotes(self, mode: str, orders, ts: Optional[float] = None):
        ts = ts or time.time()
        self.c.executemany("INSERT INTO quotes VALUES (?,?,?,?,?,?,?,?,?,?)",
                           [(ts, mode, o.ticker, o.event, o.side, o.price_cents, o.count, o.sleeve, o.fair_yes,
                             o.reason) for o in orders])
        self.c.commit()

    def add_ml_snapshots(self, rows):
        """Pre-registered scorecard rows: one per (event, decision hour, bracket), first write wins."""
        self.c.executemany("INSERT OR IGNORE INTO ml_snapshots VALUES (?,?,?,?,?,?,?,?)", rows)
        self.c.commit()

    def add_ml_ctx(self, rows):
        """Context the pool's disagreement tilt needs at refit time, one row per (event, decision hour): NBM's max
        for the event as the forecaster saw it (event, h, nbs, ts); first write wins, like the snapshots."""
        self.c.executemany("INSERT OR IGNORE INTO ml_snap_ctx VALUES (?,?,?,?)", rows)
        self.c.commit()

    def add_taker_shadow(self, rows):
        """Sleeve X shadow trades: (ts, event, ticker, side, price, size, avail, p, edge, h, model_sha, spr, msum
        [, intended, depth, booked]); rows without the last three leave them NULL (read as TAKER_BOOKED_SQL)."""
        rows = [tuple(r) + (None,) * (16 - len(r)) for r in rows]
        self.c.executemany("INSERT OR IGNORE INTO taker_shadow (ts,event,ticker,side,price,size,avail,p,edge,h,model_sha,"
                           "spr,msum,intended,depth,booked) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        self.c.commit()

    def taker_shadow_keys(self) -> set:
        return {(e, t, s) for e, t, s in self.c.execute("SELECT event, ticker, side FROM taker_shadow")}

    def taker_shadow_open(self, older_than: float) -> list:
        return [t for (t,) in self.c.execute("SELECT DISTINCT ticker FROM taker_shadow WHERE result IS NULL AND ts < ? "
                                             "ORDER BY ts", (older_than,))]

    def set_taker_result(self, ticker: str, result: str, ts: float):
        self.c.execute("UPDATE taker_shadow SET result=?, settled_ts=? WHERE ticker=? AND result IS NULL",
                       (result, ts, ticker))
        self.c.commit()

    # ---- Sleeve R shadow (daily rain markets, model taker; logged only, never sent)
    # table: "rain_shadow" (Sleeve R) or "rain_shadow_v2" (the R2 variant); never user input
    def add_rain_shadow(self, rows, table: str = "rain_shadow"):
        """(ts, event, ticker, city, day, h, side, price, size, fee, p_model, p_pool, mid, edge, model_sha
        [, intended, depth, booked]); rows without the last three leave them NULL (booked 0: depth unknown)."""
        rows = [tuple(r) + (None,) * (18 - len(r)) for r in rows]
        self.c.executemany(f"INSERT OR IGNORE INTO {table} (ts,event,ticker,city,day,h,side,price,size,fee,p_model,"
                           "p_pool,mid,edge,model_sha,intended,depth,booked) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           rows)
        self.c.commit()

    def rain_shadow_keys(self, table: str = "rain_shadow") -> set:
        return {(t, s) for t, s in self.c.execute(f"SELECT ticker, side FROM {table}")}

    def rain_shadow_open(self, older_than: float, table: str = "rain_shadow") -> list:
        return [t for (t,) in self.c.execute(f"SELECT DISTINCT ticker FROM {table} WHERE result IS NULL AND ts < ? "
                                             "ORDER BY ts", (older_than,))]

    def set_rain_result(self, ticker: str, result: str, ts: float, table: str = "rain_shadow"):
        self.c.execute(f"UPDATE {table} SET result=?, settled_ts=? WHERE ticker=? AND result IS NULL",
                       (result, ts, ticker))
        self.c.commit()

    def rain_pnl(self, table: str = "rain_shadow", sizing: str = "booked") -> tuple:
        """(settled P&L in dollars at the booked size, settled booked contracts, logged signals). The booked size is
        the logged size capped by the contracts offered at the price when it was logged, 0 when that was unknown
        (every row from before Oct 9) or zero: the log no longer assumes fills (outside review, Oct 8).
        sizing="logged": the logged size instead (5 contracts a signal), which is the Dec 1 holdout's registered
        test (HOLDOUT_DEC1.md item 7b); the Dec 1 scoring reports both, the registered one first."""
        n_all = self.c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        pnl = cnt = 0.0
        col = "size" if sizing == "logged" else RAIN_BOOKED_SQL
        for side, price, booked, fee, res in self.c.execute(
                f"SELECT side, price, {col}, fee, result FROM {table} WHERE result IN ('yes','no')"):
            win = (res == "yes") if side == "yes" else (res == "no")
            pnl += booked * ((1.0 if win else 0.0) - price - fee)
            cnt += booked
        return pnl, cnt, n_all

    # ---- Sleeve H shadow (hourly index markets; simulated fills on the live tape, never sent)
    def add_hourly_fill(self, row, variant: str = "H"):
        """(ts, event, ticker, side, price_cents, count [, intended, depth, booked])"""
        row = tuple(row) + (None,) * (9 - len(row))
        self.c.execute("INSERT INTO hourly_shadow (ts,event,ticker,side,price,count,intended,depth,booked,variant) "
                       "VALUES (?,?,?,?,?,?,?,?,?,?)", row + (variant,))
        self.c.commit()

    def hourly_open(self, variant: str = "H") -> list:
        return list(self.c.execute("SELECT ts, event, ticker, side, price, count FROM hourly_shadow "
                                   "WHERE result IS NULL AND variant = ?", (variant,)))

    def hourly_open_tickers(self, older_than: float) -> list:
        return [t for (t,) in self.c.execute("SELECT DISTINCT ticker FROM hourly_shadow WHERE result IS NULL AND ts < ? "
                                             "ORDER BY ts", (older_than,))]

    def set_hourly_result(self, ticker: str, result: str, ts: float):
        self.c.execute("UPDATE hourly_shadow SET result=?, settled_ts=? WHERE ticker=? AND result IS NULL",
                       (result, ts, ticker))
        self.c.commit()

    def hourly_pnl(self, since_ts: float = 0.0, variant: str = "H") -> tuple:
        """(settled P&L in dollars, settled contracts) of a Sleeve H shadow variant since since_ts (fill time), at the
        booked size (the paper broker's fill, capped by the trade print, as `count` always was)."""
        n = HOURLY_BOOKED_SQL
        r = self.c.execute(
            f"SELECT SUM(CASE WHEN side='bid' THEN {n}*((result='yes') - price/100.0) "
            f"ELSE {n}*(price/100.0 - (result='yes')) END), SUM({n}) FROM hourly_shadow "
            "WHERE result IN ('yes','no') AND ts >= ? AND variant = ?", (since_ts, variant)).fetchone()
        return float(r[0] or 0.0), float(r[1] or 0.0)

    # ---- Kalshi Weather Index points (1-minute, as published), kept for the hourly nowcast
    def add_index_points(self, city: str, pts):
        """pts: [(epoch_minute, value_F), ...]"""
        self.c.executemany("INSERT OR IGNORE INTO index_points VALUES (?,?,?)", [(city, int(t), float(v)) for t, v in pts])
        self.c.commit()

    def index_points(self, city: str, since_minute: int) -> list:
        return list(self.c.execute("SELECT t, v FROM index_points WHERE city = ? AND t >= ? ORDER BY t",
                                   (city, int(since_minute))))

    def prune_index_points(self, before_minute: int):
        self.c.execute("DELETE FROM index_points WHERE t < ?", (int(before_minute),))
        self.c.commit()

    def add_fair_values(self, rows):
        self.c.executemany("INSERT INTO fair_values VALUES (?,?,?,?,?,?,?)", rows)
        self.c.commit()

    def add_lip(self, rows):
        self.c.executemany("INSERT INTO lip_snaps VALUES (?,?,?,?,?,?)", rows)
        self.c.commit()

    def set_settlement(self, ticker, event, result, ts, value):
        self.c.execute("INSERT OR REPLACE INTO settlements VALUES (?,?,?,?,?)", (ticker, event, result, ts, value))
        self.c.commit()

    def add_live_settlement(self, ticker, event, result, ts, revenue_usd, cost_usd, fee_usd) -> bool:
        """Returns True if this settlement was new."""
        cur = self.c.execute("INSERT OR IGNORE INTO live_settlements VALUES (?,?,?,?,?,?,?,?)",
                             (ticker, event, result, ts, revenue_usd, cost_usd, fee_usd,
                              revenue_usd - cost_usd - fee_usd))
        self.c.commit()
        return cur.rowcount > 0

    def live_pnl(self, since_ts: float = 0.0) -> float:
        r = self.c.execute("SELECT SUM(pnl_usd) FROM live_settlements WHERE settled_ts >= ?", (since_ts,)).fetchone()
        return float(r[0] or 0.0)

    def settled(self) -> dict:
        return {r[0]: r[1] for r in self.c.execute("SELECT ticker, result FROM settlements")}

    def fills(self, mode: Optional[str] = None):
        q = "SELECT ts,mode,ticker,event,side,price_cents,count,sleeve,is_taker,fee,order_ref,fair_yes FROM fills"
        args = ()
        if mode:
            q += " WHERE mode=?"
            args = (mode,)
        return self.c.execute(q, args).fetchall()

    def kv_set(self, k, v):
        self.c.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (k, json.dumps(v)))
        self.c.commit()

    def kv_get(self, k, default=None):
        r = self.c.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return json.loads(r[0]) if r else default
