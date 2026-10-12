"""Kalshi Trade API v2 REST client (public market data + authenticated V2 orders).

Design notes
- Prices on the wire are fixed-point dollar strings ("0.5600"); counts are fixed-point strings ("10.00").
  Internally we use integer cents for prices (weather markets tick in whole cents) and floats for counts.
- V2 orders use a single book: side="bid" == long YES at `price`; side="ask" == long NO, still quoted at the
  YES price (buying NO at 88c is an ask at 0.12).
- Token-bucket rate limiting mirrors Kalshi's tiers (Basic: 200 read / 100 write tokens per second, most calls
  cost 10 tokens, cancels cost 2). We stay at ~80% of budget.
"""
from __future__ import annotations

import collections
import json
import logging
import threading
import time
import uuid
from typing import Any, Iterable, Optional

import requests

from .auth import KalshiAuth

log = logging.getLogger(__name__)

PROD = "https://external-api.kalshi.com/trade-api/v2"
PROD_ALT = "https://api.elections.kalshi.com/trade-api/v2"
DEMO = "https://external-api.demo.kalshi.co/trade-api/v2"


def cents_to_str(c: int) -> str:
    if not 1 <= int(c) <= 99:
        raise ValueError(f"price out of range: {c}")
    return f"{int(c) / 100:.4f}"


def str_to_cents(s: Any) -> Optional[int]:
    if s is None or s == "":
        return None
    return int(round(float(s) * 100))


def count_str(n: float) -> str:
    return f"{round(float(n), 2):.2f}"


class TokenBucket:
    """Token bucket with adaptive slow-down: a 429 halves the refill rate (floor 10% of nominal) and empties the
    bucket; each later success restores 2% of the gap, so a burst of 429s settles near the server's real limit."""

    def __init__(self, rate: float, capacity: float):
        self.rate = rate
        self.nominal = rate
        self.capacity = capacity
        self.tokens = capacity
        self.t = time.monotonic()
        self.lock = threading.Lock()

    def take(self, cost: float):
        # a request that costs more than the bucket holds (a large batch) waits for a full bucket and then goes,
        # leaving it in debt, instead of waiting forever (round-7 review; Kalshi bills such a batch at once too)
        need = min(cost, self.capacity)
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.t) * self.rate)
                self.t = now
                if self.tokens >= need:
                    self.tokens -= cost
                    return
                wait = (need - self.tokens) / self.rate
            time.sleep(min(max(wait, 0.01), 1.0))

    def try_take(self, cost: float, reserve: float = 0.0) -> bool:
        """take() for optional requests: takes `cost` only if `cost + reserve` tokens are there now, otherwise takes
        nothing and says so. Never waits."""
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.t) * self.rate)
            self.t = now
            if self.tokens >= cost + reserve:
                self.tokens -= cost
                return True
            return False

    def throttled(self):
        with self.lock:
            self.rate = max(self.rate * 0.5, self.nominal * 0.1)
            self.tokens = 0.0

    def ok(self):
        if self.rate < self.nominal:
            with self.lock:
                self.rate = min(self.nominal, self.rate + 0.02 * (self.nominal - self.rate) + 0.01)


class KalshiError(Exception):
    """ambiguous: an earlier attempt of the same request got no definite answer (timeout, 5xx), so a write may have
    happened even though this answer is an error (a retried create can come back 409 'already exists')."""

    def __init__(self, status: int, body: str, method: str, path: str, ambiguous: bool = False):
        super().__init__(f"{method} {path} -> {status}: {body[:300]}")
        self.status = status
        self.body = body
        self.ambiguous = ambiguous


class KalshiREST:
    def __init__(
        self,
        base_url: str = PROD,
        auth: Optional[KalshiAuth] = None,
        read_rate: float = 160.0,  # tokens/s (80% of Basic 200)
        write_rate: float = 80.0,  # tokens/s (80% of Basic 100)
        timeout: float = 15.0,
        user_agent: str = "wxbot/0.1",
    ):
        self.base = base_url.rstrip("/")
        self.prefix = "/trade-api/v2"
        self.auth = auth
        self.read_bucket = TokenBucket(read_rate, read_rate * 3)
        self.write_bucket = TokenBucket(write_rate, write_rate)
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
        self.n_requests = 0
        self.n_errors = 0
        self.n_throttled = 0
        self.last_error: Optional[str] = None
        # round-trip times (seconds) of recent successful requests, for the speed stats in status.json
        self.rtt = {"read": collections.deque(maxlen=600), "write": collections.deque(maxlen=600)}

    # ------------------------------------------------------------------ core
    def _request(self, method: str, path: str, params=None, body=None, auth_required=False,
                 write=False, cost=10, retries=4, timeout: Optional[float] = None) -> Any:
        """cost 0: the tokens were already taken (try_take; an optional read that must not wait on the limiter).
        timeout: seconds for this request instead of the client's."""
        url = self.base + path
        sign_path = self.prefix + path
        if cost > 0:
            (self.write_bucket if write else self.read_bucket).take(cost)
        ambiguous = False   # did an earlier attempt possibly reach the exchange without a definite answer?
        for attempt in range(retries + 1):
            headers = {}
            if self.auth is not None:
                headers.update(self.auth.headers(method, sign_path))
            elif auth_required:
                raise RuntimeError(f"{method} {path} requires API credentials (set KALSHI_KEY_ID + key)")
            if body is not None:
                headers["Content-Type"] = "application/json"
            try:
                self.n_requests += 1
                t_req = time.time()
                r = self.s.request(method, url, params=params,
                                   data=json.dumps(body) if body is not None else None,
                                   headers=headers, timeout=timeout or self.timeout)
            except requests.RequestException as e:
                self.n_errors += 1
                self.last_error = str(e)
                if attempt >= retries:
                    raise
                ambiguous = ambiguous or not isinstance(e, requests.ConnectTimeout)
                time.sleep(0.5 * (2 ** attempt))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                self.n_errors += 1
                self.last_error = f"{r.status_code} {r.text[:200]}"
                if r.status_code == 429:
                    self.n_throttled += 1
                    (self.write_bucket if write else self.read_bucket).throttled()
                if attempt >= retries:
                    raise KalshiError(r.status_code, r.text, method, path,
                                      ambiguous=ambiguous or r.status_code >= 500)
                ambiguous = ambiguous or r.status_code >= 500
                time.sleep(0.4 * (2 ** attempt))
                continue
            (self.write_bucket if write else self.read_bucket).ok()
            self.rtt["write" if write else "read"].append(time.time() - t_req)
            if r.status_code >= 400:
                self.n_errors += 1
                self.last_error = f"{r.status_code} {r.text[:200]}"
                raise KalshiError(r.status_code, r.text, method, path, ambiguous=ambiguous)
            if not r.content:
                return {}
            try:
                return r.json()
            except ValueError as e:
                # a success answer whose body cannot be read: the request went through, so for a write the outcome
                # is unknown, never a refusal (round-2 verification, Oct 4)
                self.n_errors += 1
                self.last_error = f"{r.status_code} unreadable body {r.text[:120]!r}"
                raise KalshiError(r.status_code, r.text[:500], method, path, ambiguous=True) from e
        raise RuntimeError("unreachable")

    def rtt_ms(self) -> dict:
        """Median and 95th-percentile round trip (milliseconds) of recent reads and writes."""
        out = {}
        for k, v in self.rtt.items():
            if v:
                xs = sorted(v)
                out[k] = {"n": len(xs), "p50": round(1000 * xs[len(xs) // 2]),
                          "p95": round(1000 * xs[min(len(xs) - 1, int(0.95 * len(xs)))])}
        return out

    def get(self, path, params=None, auth_required=False):
        return self._request("GET", path, params=params, auth_required=auth_required)

    def paged(self, path, key, params=None, max_pages=200, auth_required=False):
        params = dict(params or {})
        out, cursor = [], None
        for _ in range(max_pages):
            p = dict(params)
            if cursor:
                p["cursor"] = cursor
            d = self.get(path, p, auth_required=auth_required)
            items = d.get(key) or []
            out.extend(items)
            cursor = d.get("cursor") or d.get("next_cursor")
            if not cursor or not items:
                break
        return out

    # ------------------------------------------------------------ public data
    def exchange_status(self):
        """Kalshi answers 500/503/504 during an exchange pause or outage with the status itself in the body (its
        docs): read it rather than failing, so a pause is seen as a pause."""
        try:
            return self.get("/exchange/status")
        except KalshiError as e:
            try:
                d = json.loads(e.body or "")
            except ValueError:
                d = None
            if e.status >= 500 and isinstance(d, dict) and "trading_active" in d:
                return d
            raise

    def exchange_schedule(self) -> dict:
        """Kalshi's published trading hours and maintenance windows (times in ET; wxbot/exchange_hours.py)."""
        return (self.get("/exchange/schedule") or {}).get("schedule") or {}

    def user_data_timestamp(self) -> Optional[float]:
        """Epoch seconds up to which GetBalance, GetOrder(s), GetFills and GetPositions reflect the exchange (Kalshi:
        'there is typically a short delay before exchange events are reflected in the API endpoints')."""
        from ..timeutil import parse_ts
        v = self.get("/exchange/user_data_timestamp").get("as_of_time")
        return parse_ts(v) if v else None

    def series(self, ticker: str):
        return self.get(f"/series/{ticker}").get("series", {})

    def events(self, series_ticker: str, status: str = "open", nested: bool = True, limit: int = 50):
        return self.paged("/events", "events", {"series_ticker": series_ticker, "status": status,
                                                "with_nested_markets": str(nested).lower(), "limit": limit},
                          max_pages=5)

    def markets(self, **params):
        return self.paged("/markets", "markets", params)

    def market(self, ticker: str):
        return self.get(f"/markets/{ticker}").get("market", {})

    def orderbooks(self, tickers: Iterable[str]) -> dict:
        tickers = list(tickers)
        out = {}
        for i in range(0, len(tickers), 100):
            chunk = tickers[i:i + 100]
            d = self.get("/markets/orderbooks", [("tickers", t) for t in chunk])
            for o in d.get("orderbooks", []) or []:
                out[o["ticker"]] = o.get("orderbook_fp") or {}
        return out

    def orderbooks_if_budget(self, tickers: Iterable[str], reserve_frac: float = 0.5,
                             timeout: Optional[float] = None) -> Optional[dict]:
        """orderbooks() for optional reads (the execution log's re-reads): None, with nothing sent, unless the read
        limiter holds the reads' tokens right now and keeps reserve_frac of its capacity besides. It never waits on
        the limiter and never retries, so it cannot hold up the trading loop's own reads or an order."""
        tickers = list(tickers)
        chunks = [tickers[i:i + 100] for i in range(0, len(tickers), 100)]
        if not chunks:
            return {}
        if not self.read_bucket.try_take(10 * len(chunks), reserve_frac * self.read_bucket.capacity):
            return None
        out = {}
        for chunk in chunks:
            d = self._request("GET", "/markets/orderbooks", params=[("tickers", t) for t in chunk], cost=0,
                              retries=0, timeout=timeout)
            for o in d.get("orderbooks", []) or []:
                out[o["ticker"]] = o.get("orderbook_fp") or {}
        return out

    def orderbook(self, ticker: str) -> dict:
        d = self.get(f"/markets/{ticker}/orderbook")
        return d.get("orderbook_fp") or {}

    def trades(self, ticker: Optional[str] = None, min_ts: Optional[int] = None,
               max_ts: Optional[int] = None, max_pages: int = 100):
        p: dict = {"limit": 1000}
        if ticker:
            p["ticker"] = ticker
        if min_ts is not None:
            p["min_ts"] = int(min_ts)
        if max_ts is not None:
            p["max_ts"] = int(max_ts)
        return self.paged("/markets/trades", "trades", p, max_pages=max_pages)

    def candlesticks(self, series: str, ticker: str, start_ts: int, end_ts: int, period: int = 1):
        d = self.get(f"/series/{series}/markets/{ticker}/candlesticks",
                     {"start_ts": int(start_ts), "end_ts": int(end_ts), "period_interval": period})
        return d.get("candlesticks", [])

    def incentive_programs(self, status: str = "active"):
        return self.paged("/incentive_programs", "incentive_programs", {"status": status, "limit": 10000},
                          max_pages=20)

    def weather_index(self, city: str = "miami"):
        return self.get(f"/live_data/weather/{city}")

    # ------------------------------------------------------- authenticated
    def balance(self):
        return self.get("/portfolio/balance", auth_required=True)

    def positions(self, **params):
        """(market_positions, event_positions), following the cursor across pages."""
        params = dict(params)
        params.setdefault("limit", 1000)
        mp, ep, cursor = [], [], None
        for _ in range(50):
            p = dict(params, **({"cursor": cursor} if cursor else {}))
            d = self.get("/portfolio/positions", p, auth_required=True)
            mp += d.get("market_positions", []) or []
            ep += d.get("event_positions", []) or []
            cursor = d.get("cursor")
            if not cursor:
                break
        return mp, ep

    def order(self, order_id: str) -> dict:
        d = self.get(f"/portfolio/orders/{order_id}", auth_required=True)
        return d.get("order") or {}

    def orders(self, status: str = "resting", **params):
        params = dict(params, status=status, limit=1000)
        return self.paged("/portfolio/orders", "orders", params, auth_required=True)

    def fills(self, min_ts: Optional[int] = None, **params):
        params = dict(params, limit=1000)
        if min_ts is not None:
            params["min_ts"] = int(min_ts)
        return self.paged("/portfolio/fills", "fills", params, auth_required=True, max_pages=20)

    def netting(self):
        return self.get("/portfolio/subaccounts/netting", auth_required=True)

    def set_netting(self, enabled: bool, subaccount: int = 0):
        return self._request("PUT", "/portfolio/subaccounts/netting",
                             body={"subaccount_number": int(subaccount), "enabled": bool(enabled)},
                             auth_required=True, write=True)

    def settlements(self, min_ts: Optional[int] = None, **params):
        params = dict(params, limit=1000)
        if min_ts is not None:
            params["min_ts"] = int(min_ts)
        return self.paged("/portfolio/settlements", "settlements", params, auth_required=True, max_pages=20)

    def create_order(self, ticker: str, side: str, price_cents: int, count: float, *,
                     post_only: bool = True, tif: str = "good_till_canceled",
                     expiration_ts: Optional[int] = None, client_order_id: Optional[str] = None,
                     exchange_index: int = 0, order_group_id: Optional[str] = None,
                     cancel_on_pause: bool = True, retries: int = 4) -> dict:
        """retries: attempts after a timeout or 5xx, re-sending the same client_order_id (Kalshi refuses a duplicate
        id). Immediate-or-cancel takers pass 0: a lost answer is not retried, so a take can never execute twice."""
        if side not in ("bid", "ask"):
            raise ValueError("side must be 'bid' (long YES) or 'ask' (long NO)")
        body = {
            "ticker": ticker,
            "client_order_id": client_order_id or str(uuid.uuid4()),
            "side": side,
            "count": count_str(count),
            "price": cents_to_str(price_cents),
            "time_in_force": tif,
            "self_trade_prevention_type": "taker_at_cross",   # a crossing order of ours dies; our resting one stays
            "post_only": bool(post_only),
            "cancel_order_on_pause": bool(cancel_on_pause),
            "exchange_index": int(exchange_index),
        }
        if expiration_ts:
            body["expiration_time"] = int(expiration_ts)
        if order_group_id:
            body["order_group_id"] = order_group_id
        return self._request("POST", "/portfolio/events/orders", body=body, auth_required=True, write=True,
                             retries=retries)

    def batch_create(self, orders: list[dict]) -> dict:
        return self._request("POST", "/portfolio/events/orders/batched", body={"orders": orders},
                             auth_required=True, write=True, cost=10 * max(1, len(orders)))

    def amend_order(self, order_id: str, ticker: str, side: str, price_cents: int, total_count: float,
                    exchange_index: int = 0, expiration_ts: Optional[int] = None) -> dict:
        """count = filled + desired remaining. Same price + count with a new expiration_ts = expiry-only amend,
        which keeps queue priority."""
        body = {"ticker": ticker, "side": side, "price": cents_to_str(price_cents),
                "count": count_str(total_count), "exchange_index": int(exchange_index)}
        if expiration_ts:
            body["expiration_time"] = int(expiration_ts)
        return self._request("POST", f"/portfolio/events/orders/{order_id}/amend", body=body,
                             auth_required=True, write=True)

    def cancel_order(self, order_id: str, exchange_index: int = 0) -> dict:
        return self._request("DELETE", f"/portfolio/events/orders/{order_id}",
                             params={"exchange_index": exchange_index}, auth_required=True, write=True, cost=2)

    def batch_cancel(self, order_ids: list[str], exchange_index: int = 0) -> dict:
        body = {"orders": [{"order_id": o, "exchange_index": exchange_index} for o in order_ids]}
        return self._request("DELETE", "/portfolio/events/orders/batched", body=body, auth_required=True,
                             write=True, cost=2 * max(1, len(order_ids)))

    def create_order_group(self, contracts_limit: int, exchange_index: int = 0) -> dict:
        body = {"contracts_limit": int(contracts_limit), "exchange_index": exchange_index}
        return self._request("POST", "/portfolio/order_groups/create", body=body, auth_required=True,
                             write=True)
