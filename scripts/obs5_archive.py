"""Forward archive of the NWS API's station observations (the 5-minute ASOS readings, whole degrees C, plus the
hourly METARs) for the 20 cities and Sleeve T's four, with the time each reading was first seen here. The bot's
rounding guard (wxbot/models/round_guard.py, config round_guard) reads it.

Why: the NWS API keeps under seven days, and what was public in real time, and when, cannot be rebuilt later. Track
B part 2 (docs/results/r4_obs_speed.txt) simulates this feed from the 1-minute archive with an assumed 25-minute
delay; this archive measures the real delay and coverage going forward. Read-only use of a public API; no key.

Runs on the server every 5 minutes (deploy/systemd/wxbot-obs5.timer), separate from the bots.
Usage: python3 scripts/obs5_archive.py [--db data/obs5.sqlite]
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time

import requests

API = "https://api.weather.gov/stations/{}/observations"
UA = {"User-Agent": "wxbot/0.2 (student research; contact via GitHub)", "Accept": "application/geo+json"}
STATIONS = ["KAUS", "KMDW", "KDEN", "KLAX", "KMIA", "KNYC", "KPHL", "KATL", "KBOS", "KDFW", "KDCA", "KHOU", "KLAS",
            "KMSY", "KOKC", "KPHX", "KSAT", "KSEA", "KSFO", "KMSP",
            # Sleeve T's four cities (Oct 9): the rounding guard (round_guard in the bot config) reads this archive
            "KTTN", "KSAN", "KEWR", "KSDF"]
SQL = """CREATE TABLE IF NOT EXISTS obs5 (
  station TEXT, ts TEXT, temp_c REAL, dewpoint_c REAL, max24_c REAL, raw TEXT, first_seen REAL,
  PRIMARY KEY (station, ts))"""


def rows_from(station: str, doc: dict, seen: float) -> list:
    out = []
    for f in doc.get("features") or []:
        p = f.get("properties") or {}
        ts = p.get("timestamp")
        if not ts:
            continue

        def val(k):
            v = (p.get(k) or {}).get("value")
            return float(v) if isinstance(v, (int, float)) else None
        out.append((station, ts, val("temperature"), val("dewpoint"), val("maxTemperatureLast24Hours"),
                    (p.get("rawMessage") or "")[:600], seen))   # whole METAR: the 6-hour groups sit at its end
    return out


def run(db: str, get=requests.get, limit: int = 60) -> dict:
    c = sqlite3.connect(db, timeout=30)
    c.execute(SQL)
    n_new, n_err = 0, 0
    for s in STATIONS:
        try:
            r = get(API.format(s), params={"limit": limit}, headers=UA, timeout=15)
            if r.status_code != 200:
                n_err += 1
                continue
            rows = rows_from(s, r.json(), time.time())
        except (requests.RequestException, ValueError):
            n_err += 1
            continue
        before = c.total_changes
        c.executemany("INSERT OR IGNORE INTO obs5 VALUES (?,?,?,?,?,?,?)", rows)
        n_new += c.total_changes - before
    c.commit()
    c.close()
    return {"new": n_new, "errors": n_err}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(os.path.dirname(__file__), "..", "data", "obs5.sqlite"))
    a = ap.parse_args(argv)
    print(json.dumps(run(a.db)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
