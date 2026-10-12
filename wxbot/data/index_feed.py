"""Kalshi Weather Index feed: the published 1-minute index per city (5 minutes delayed), kept for ~30 hours.

The endpoint (GET /live_data/weather/{city}) returns a rolling 24 hours; the hourly nowcast also needs the same
clock window yesterday, so the bot keeps its own copy in the store (table index_points) and survives restarts.
The store keeps every point for a year (about 5,800 rows a day for four cities): it is the only permanent record of
the exact published index, which the Dec 1 replays and a nowcast retrained on the real index need.
Only points with status normal or degraded count (they are the canonical observations settlement uses).
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd

log = logging.getLogger("wxbot.index")


class IndexFeed:
    def __init__(self, rest, store, cities, keep_h: float = 30.0, store_keep_days: float = 400.0):
        self.rest = rest
        self.store = store
        self.cities = list(cities)
        self.keep_min = int(keep_h * 60)
        self.store_keep_min = int(store_keep_days * 1440)
        self.series: dict[str, dict[int, float]] = {}
        now_min = int(time.time() // 60)
        for c in self.cities:
            self.series[c] = {int(t): float(v) for t, v in store.index_points(c, now_min - self.keep_min)}
        self.last_poll = 0.0

    def poll(self):
        now_min = int(time.time() // 60)
        for c in self.cities:
            try:
                d = self.rest.weather_index(c)
            except Exception as e:  # pragma: no cover - network
                log.warning("index %s fetch failed: %s", c, e)
                continue
            s = self.series.setdefault(c, {})
            new = []
            for p in d.get("timeseries") or []:
                if p.get("status") not in ("normal", "degraded") or p.get("v") is None or p.get("t") is None:
                    continue
                t = int(p["t"]) // 60000
                if t not in s:
                    s[t] = float(p["v"])
                    new.append((t, float(p["v"])))
            if new:
                self.store.add_index_points(c, new)
            for t in [t for t in s if t < now_min - self.keep_min]:
                del s[t]
        self.store.prune_index_points(now_min - self.store_keep_min)
        self.last_poll = time.time()

    def as_series(self, city: str) -> pd.Series:
        """Index in F on a naive-UTC minute index (published event minutes)."""
        s = self.series.get(city) or {}
        if not s:
            return pd.Series(dtype=float)
        ts = sorted(s)
        return pd.Series([s[t] for t in ts], index=pd.to_datetime([t * 60 for t in ts], unit="s"))

    def latest(self, city: str) -> Optional[tuple]:
        s = self.series.get(city) or {}
        if not s:
            return None
        t = max(s)
        return t, s[t]

    def hours_of_history(self, city: str) -> float:
        s = self.series.get(city) or {}
        return (max(s) - min(s)) / 60.0 if s else 0.0
