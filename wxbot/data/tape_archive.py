"""Keep our own copy of the public tape of settled events, while it is still on Kalshi's live API.

The Dec 1 holdout replays (docs/HOLDOUT_DEC1.md) need, per event: every market's result, 1-minute bid/ask candles and
every trade with its taker side. Kalshi moves settled markets, their candles and their trades to separate /historical
endpoints about two months after the fact (GET /historical/cutoff), and the hourly index series are large (tens of
thousands of trades a day). So the bot archives each event soon after it settles, in a background thread with its own
gentle pacing, public endpoints only (no auth, no orders):

  <root>/hourly/<series>/<event>.json.gz   hourly index ladders (KXTEMP*H*), candles from one event-candlesticks call
  <root>/daily/<series>/<event>.json.gz    daily high ladders (KXHIGH*), candles per market

Record: {event, series, kind, open_ts, close_ts, archived_ts,
         markets: [{ticker, strike, cap, type, result, value, volume}],
         candles: {ticker: [[end_ts, yes_bid, yes_ask, last, volume], ...]},     dollars, None when absent
         trades:  {ticker: [[ts, yes_price, count, taker_side], ...]}}           oldest first
The candle and trade lists match research/hourly/pull_hourly.py and pull_hourly_trades.py, so the replays read either.
Nothing here looks at prices to decide anything; it only writes files.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import logging
import os
import threading
import time
from typing import Optional

import requests

log = logging.getLogger("wxbot.tape")

PROD = "https://api.elections.kalshi.com/trade-api/v2"


def _ts(s: str) -> float:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _val(x):
    """A market's settlement value: a number on the index and temperature ladders, a word ("Yes", "No") on some
    others (the rain markets)."""
    try:
        return _f(x)
    except (TypeError, ValueError):
        return str(x)


def _f(x):
    return None if x is None or x == "" else float(x)


def compact(cs) -> list:
    out = []
    for c in cs or []:
        b = (c.get("yes_bid") or {}).get("close_dollars")
        a = (c.get("yes_ask") or {}).get("close_dollars")
        p = (c.get("price") or {}).get("close_dollars")
        out.append([int(c["end_period_ts"]), _f(b), _f(a), _f(p), float(c.get("volume_fp") or 0)])
    return out


def read_record(path: str) -> dict:
    with gzip.open(path, "rt") as f:
        return json.load(f)


class TapeArchiver:
    def __init__(self, root: str, hourly_series, daily_series, since_ts: float, base: str = PROD,
                 gap_s: float = 1.0, settle_lag_s: float = 900.0, session=None):
        self.root = root
        self.series = [("hourly", s) for s in hourly_series] + [("daily", s) for s in daily_series]
        self.since = float(since_ts)
        self.base = base.rstrip("/")
        self.gap = float(gap_s)
        self.lag = float(settle_lag_s)
        self.s = session or requests.Session()
        self.s.headers.update({"User-Agent": "wxbot-tape-archive/0.1"})
        self._last = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.stats = {"archived": 0, "calls": 0, "errors": 0, "pending": None, "last_run_ts": None,
                      "last_event": None, "files": None}
        self.failed: dict[str, int] = {}

    # ---- HTTP -------------------------------------------------------------------------------------------------
    def get(self, path: str, params=None) -> Optional[dict]:
        for k in range(6):
            wait = self.gap - (time.time() - self._last)
            if wait > 0 and self._stop.wait(wait):
                return None
            self._last = time.time()
            try:
                r = self.s.get(self.base + path, params=params, timeout=30)
            except Exception:
                if self._stop.wait(3 * (k + 1)):
                    return None
                continue
            self.stats["calls"] += 1
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                self.gap = min(5.0, self.gap * 1.5)
                if self._stop.wait(5 + 5 * k):
                    return None
                continue
            if self._stop.wait(3 * (k + 1)):
                return None
        self.stats["errors"] += 1
        return None

    # ---- what to archive --------------------------------------------------------------------------------------
    def path_for(self, kind: str, series: str, event: str) -> str:
        return os.path.join(self.root, kind, series, event + ".json.gz")

    def settled_events(self, series: str) -> list:
        """(close_ts, event) for settled events of a series closed in [since, now - lag], oldest first."""
        out, cur, n_old = [], None, 0
        for _ in range(100):
            p = {"series_ticker": series, "status": "settled", "with_nested_markets": "true", "limit": 200}
            if cur:
                p["cursor"] = cur
            d = self.get("/events", p)
            if not d:
                break
            for e in d.get("events", []):
                ms = e.get("markets") or []
                if not ms:
                    continue
                close = max(_ts(m["close_time"]) for m in ms)
                if close < self.since:
                    n_old += 1
                    continue
                if close <= time.time() - self.lag:
                    out.append((close, e))
            cur = d.get("cursor")
            if not cur or n_old > 50:   # listings run newest first; a run of old events means we are done
                break
        return sorted(out, key=lambda x: x[0])

    def trades(self, ticker: str) -> Optional[list]:
        out, cur = [], None
        for _ in range(500):
            p = {"ticker": ticker, "limit": 1000}
            if cur:
                p["cursor"] = cur
            d = self.get("/markets/trades", p)
            if d is None:
                return None   # a failed page: write nothing for this event, retry on a later pass
            for t in d.get("trades", []):
                out.append([round(_ts(t["created_time"]), 3), float(t.get("yes_price_dollars") or 0),
                            float(t.get("count_fp") or 0), t.get("taker_side") or t.get("taker_outcome_side")])
            cur = d.get("cursor")
            if not cur:
                break
        return sorted(out)

    def archive_event(self, kind: str, series: str, e: dict) -> bool:
        ms = e.get("markets") or []
        et = e["event_ticker"]
        close = int(max(_ts(m["close_time"]) for m in ms))
        opn = int(min(_ts(m["open_time"]) for m in ms))
        rec = {"event": et, "series": series, "kind": kind, "open_ts": opn, "close_ts": close,
               "archived_ts": int(time.time()),
               "markets": [{"ticker": m["ticker"], "strike": m.get("floor_strike"), "cap": m.get("cap_strike"),
                            "type": m.get("strike_type"), "result": m.get("result"),
                            "value": _val(m.get("expiration_value")), "volume": float(m.get("volume_fp") or 0)}
                           for m in ms]}
        candles = {}
        if kind == "hourly":
            d = self.get(f"/series/{series}/events/{et}/candlesticks",
                         {"start_ts": opn - 60, "end_ts": close, "period_interval": 1})
            if d is None:
                return False
            candles = {t: compact(c) for t, c in zip(d.get("market_tickers", []), d.get("market_candlesticks", []))}
        else:
            for m in ms:
                d = self.get(f"/series/{series}/markets/{m['ticker']}/candlesticks",
                             {"start_ts": int(_ts(m["open_time"])), "end_ts": int(_ts(m["close_time"])),
                              "period_interval": 1})
                if d is None:
                    return False
                candles[m["ticker"]] = compact(d.get("candlesticks"))
        rec["candles"] = candles
        tr = {}
        for m in ms:
            if float(m.get("volume_fp") or 0) <= 0:
                continue
            t = self.trades(m["ticker"])
            if t is None:
                return False
            tr[m["ticker"]] = t
        rec["trades"] = tr
        fn = self.path_for(kind, series, et)
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        with gzip.open(fn + ".tmp", "wt") as f:
            json.dump(rec, f, separators=(",", ":"))
        os.replace(fn + ".tmp", fn)
        return True

    def run_once(self, budget_s: float = 900.0) -> int:
        """Archive pending events, oldest first, until done or the time budget is spent. Returns events written."""
        t0 = time.time()
        pending = []
        for kind, s in self.series:
            if self._stop.is_set():
                return 0
            for close, e in self.settled_events(s):
                if not os.path.exists(self.path_for(kind, s, e["event_ticker"])):
                    pending.append((close, kind, s, e))
        pending.sort(key=lambda x: x[0])
        self.stats["pending"] = len(pending)
        n = 0
        for close, kind, s, e in pending:
            if self._stop.is_set() or time.time() - t0 > budget_s:
                break
            et = e["event_ticker"]
            if self.failed.get(et, 0) >= 5:
                continue
            try:
                ok = self.archive_event(kind, s, e)
            except Exception as ex:  # one bad event never stops the archive
                log.warning("tape archive %s failed: %s", et, ex)
                ok = False
            if ok:
                n += 1
                self.stats["archived"] += 1
                self.stats["last_event"] = et
                self.stats["pending"] -= 1
            else:
                self.failed[et] = self.failed.get(et, 0) + 1
        self.stats["last_run_ts"] = time.time()
        self.stats["files"] = sum(len(fs) for _, _, fs in os.walk(self.root)) if os.path.isdir(self.root) else 0
        if n:
            log.info("tape archive: %d events written, %d pending", n, self.stats["pending"])
        return n

    # ---- background thread ------------------------------------------------------------------------------------
    def start(self, every_s: float = 900.0):
        def loop():
            while not self._stop.is_set():
                try:
                    self.run_once()
                except Exception as ex:  # pragma: no cover - network
                    log.warning("tape archive pass failed: %s", ex)
                self._stop.wait(every_s)
        self._thread = threading.Thread(target=loop, name="tape-archive", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def status(self) -> dict:
        return dict(self.stats)
