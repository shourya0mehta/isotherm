"""Execution log (outside review, Oct 8, 2026): the order book around each taker decision and order, so a shadow
trade or a real fill can be judged against the liquidity that was really there, and against how it moved after.

Rows go to table exec_log in the sqlite store the caller already writes (the bot's database for Sleeves X, E and R;
each early shadow candidate's own database for its signals). One `ref` links the rows of one decision or order:
  decision  the book behind a signal (Sleeve X and the early shadow: the snapshot the decision priced; Sleeve R has
            none at hand, so its book is read at the next pass, as an offset-0 re-read)
  send      a taker order: order id, intended price (`price`), the limit sent, intended count, filled count, average
            fill price, fee and status, with the book at send time (paper: the book it was filled against; live: the
            loop's last read of that market, or, when there is none, an offset-0 re-read at the next pass)
  reread    the same book fields again at +1, +5 and +15 minutes (offsets_s) after the ref's first row
Book fields: best YES bid and ask (dollars) with their sizes, the contracts offered to us at our price or better
(depth_at) and at no more than 2c above our price (depth_2c), and when that book was received (book_ts).

Nothing touches the disk or the network on the order path: record() keeps its row in memory and queues the re-reads
in a small due-queue, and the caller's loop runs run_due() after its trading work. That writes the kept rows, then
makes one batch order-book read for every ticker due (each ticker once), through the caller's REST client and its
read limiter, and only when the limiter has the tokens right now while keeping reserve_frac of its capacity for the
trading loop (KalshiREST.orderbooks_if_budget). Otherwise the due re-reads are skipped, silently (counted in
status()). The limiter is never waited on and a re-read is never retried. A (store, ref, offset) is queued once: a
decision and the order sent on it share one set of re-reads. Nothing runs unless the config's exec_log.enabled is
true.
"""
from __future__ import annotations

import collections
import heapq
import itertools
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..models.fair_value import book_to_quote
from .taker import offers

log = logging.getLogger("wxbot.exec_log")

BAND = 0.02     # depth_2c: contracts offered at no more than our price + 2c

EXEC_SQL = """CREATE TABLE IF NOT EXISTS exec_log (
  ts REAL, ref TEXT, kind TEXT, offset_s INTEGER, due_ts REAL, sleeve TEXT, mode TEXT, event TEXT, ticker TEXT,
  side TEXT, price REAL, book_ts REAL, yes_bid REAL, yes_bid_size REAL, yes_ask REAL, yes_ask_size REAL,
  depth_at REAL, depth_2c REAL, order_id TEXT, limit_price REAL, count REAL, filled REAL, avg_price REAL, fee REAL,
  status TEXT)"""
COLS = ("ts", "ref", "kind", "offset_s", "due_ts", "sleeve", "mode", "event", "ticker", "side", "price", "book_ts",
        "yes_bid", "yes_bid_size", "yes_ask", "yes_ask_size", "depth_at", "depth_2c", "order_id", "limit_price",
        "count", "filled", "avg_price", "fee", "status")
BOOK_COLS = ("yes_bid", "yes_bid_size", "yes_ask", "yes_ask_size", "depth_at", "depth_2c")


@dataclass
class ExecLogConfig:
    enabled: bool = False
    offsets_s: tuple = (60, 300, 900)   # re-reads, seconds after the decision or send
    reserve_frac: float = 0.5           # re-read only while the read limiter keeps this share of its capacity
    timeout_s: float = 3.0              # per re-read request; never retried
    max_late_s: float = 120.0           # a re-read due longer ago than this is skipped (its offset would mislead)

    @classmethod
    def from_cfg(cls, d: Optional[dict]) -> "ExecLogConfig":
        d = dict(d or {})
        if "offsets_s" in d:
            d["offsets_s"] = tuple(int(x) for x in d["offsets_s"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def book_fields(book: Optional[dict], side: str, price: float) -> dict:
    """The logged fields of a raw order book (Kalshi orderbook_fp) for a buyer of `side` ('yes'/'no') at `price`
    dollars; all None without a book."""
    if book is None:
        return dict.fromkeys(BOOK_COLS)
    q = book_to_quote(book)
    off = offers(book, side)
    return {"yes_bid": None if q.yes_bid is None else q.yes_bid / 100.0,
            "yes_bid_size": None if q.yes_bid is None else q.yes_bid_size,
            "yes_ask": None if q.yes_ask is None else q.yes_ask / 100.0,
            "yes_ask_size": None if q.yes_ask is None else q.yes_ask_size,
            "depth_at": float(sum(n for p, n in off if p <= price + 1e-6)),
            "depth_2c": float(sum(n for p, n in off if p <= price + BAND + 1e-6))}


def _key(store) -> Any:
    return getattr(store, "path", None) or id(store)


@dataclass(order=True)
class Due:
    due_ts: float
    seq: int
    ref: str = field(compare=False)
    offset_s: int = field(compare=False)
    store: Any = field(compare=False)        # where its row goes
    info: dict = field(compare=False)        # sleeve, mode, event, ticker, side, price


class DueQueue:
    """Re-reads waiting for their time, earliest first. add() ignores a (store, ref, offset) that it holds or handed
    out in the last keep_s seconds, and anything beyond max_items."""

    def __init__(self, max_items: int = 5000, keep_s: float = 86400.0):
        self.heap: list = []
        self.keys: dict = {}                 # (store, ref, offset) -> due time
        self.seq = itertools.count()
        self.max_items, self.keep_s = max_items, keep_s

    def __len__(self) -> int:
        return len(self.heap)

    def add(self, due_ts: float, ref: str, offset_s: int, store, info: dict) -> bool:
        k = (_key(store), ref, int(offset_s))
        if k in self.keys or len(self.heap) >= self.max_items:
            return False
        self.keys[k] = float(due_ts)
        heapq.heappush(self.heap, Due(float(due_ts), next(self.seq), ref, int(offset_s), store, dict(info)))
        return True

    def next_due(self) -> Optional[float]:
        return self.heap[0].due_ts if self.heap else None

    def pop_due(self, now: float) -> list:
        out = []
        while self.heap and self.heap[0].due_ts <= now:
            out.append(heapq.heappop(self.heap))
        if len(self.keys) > len(self.heap) + 2000:          # forget the keys of reads done long ago
            self.keys = {k: t for k, t in self.keys.items() if t > now - self.keep_s}
        return out

    def clear(self) -> int:
        n, self.heap = len(self.heap), []
        return n


class ExecLogger:
    """record() keeps a decision or send row in memory (no disk write on the order path) and queues its re-reads;
    run_due() writes the kept rows, then reads and logs the re-reads that are due (the caller's loop calls it after
    its trading work); flush() writes the kept rows alone; drain() runs everything for a one-shot process."""

    def __init__(self, rest, cfg: ExecLogConfig, mode: str = "paper", clock: Callable[[], float] = time.time):
        self.rest, self.cfg, self.mode, self.clock = rest, cfg, mode, clock
        self.queue = DueQueue()
        self.stats: collections.Counter = collections.Counter()
        self.pending: dict = {}             # store key -> (store, rows not yet written)
        self._tables: set = set()

    def _write(self, store, rows: list):
        if _key(store) not in self._tables:
            store.c.execute(EXEC_SQL)
            store.c.execute("CREATE INDEX IF NOT EXISTS ix_exec_ref ON exec_log(ref)")
            self._tables.add(_key(store))
        store.c.executemany(f"INSERT INTO exec_log ({','.join(COLS)}) VALUES ({','.join('?' * len(COLS))})", rows)
        store.c.commit()
        self.stats["rows"] += len(rows)

    def flush(self) -> int:
        """Write the rows record() kept. Returns how many."""
        pend, self.pending = self.pending, {}
        n = 0
        for store, rows in pend.values():
            self._write(store, rows)
            n += len(rows)
        return n

    @staticmethod
    def _row(ts, ref, kind, offset_s, due_ts, info, book, book_ts, order=None) -> tuple:
        o = order or {}
        d = {"ts": ts, "ref": ref, "kind": kind, "offset_s": int(offset_s), "due_ts": due_ts, **info,
             "book_ts": book_ts if book is not None else None, **book_fields(book, info["side"], info["price"]),
             "order_id": o.get("order_id"), "limit_price": o.get("limit"), "count": o.get("requested"),
             "filled": o.get("filled"), "avg_price": o.get("avg_price"), "fee": o.get("fee"), "status": o.get("status")}
        return tuple(d.get(c) for c in COLS)

    def record(self, store, kind: str, ref: str, sleeve: str, event: str, ticker: str, side: str, price: float,
               book: Optional[dict] = None, book_ts: Optional[float] = None, order: Optional[dict] = None,
               now: Optional[float] = None):
        """A decision or send row (kept in memory until the next run_due or flush), then its re-reads at now +
        offsets_s (once per ref). book None: none at hand, so it is read at the next pass (an offset-0 re-read).
        order: a TakeResult's fields, for send rows."""
        now = self.clock() if now is None else float(now)
        info = {"sleeve": sleeve, "mode": self.mode, "event": event, "ticker": ticker, "side": side,
                "price": float(price)}
        self.pending.setdefault(_key(store), (store, []))[1].append(
            self._row(now, ref, kind, 0, None, info, book, book_ts, order))
        self.stats[kind] += 1
        if book is None:
            self.queue.add(now, ref, 0, store, info)
        for off in self.cfg.offsets_s:
            self.queue.add(now + off, ref, off, store, info)

    def run_due(self, now: Optional[float] = None) -> int:
        """Write the kept rows, then read and log the re-reads that are due: one batch read for all their tickers when
        the read budget allows it, else they are skipped. Never raises for a failed read. Returns the re-read rows
        written."""
        self.flush()
        now = self.clock() if now is None else float(now)
        due = self.queue.pop_due(now)
        if not due:
            return 0
        fresh = [d for d in due if now - d.due_ts <= self.cfg.max_late_s]
        self.stats["skipped_late"] += len(due) - len(fresh)
        if not fresh:
            return 0
        read = getattr(self.rest, "orderbooks_if_budget", None)
        books = None
        if read is not None:
            try:
                books = read(sorted({d.info["ticker"] for d in fresh}), reserve_frac=self.cfg.reserve_frac,
                             timeout=self.cfg.timeout_s)
            except Exception as e:      # an optional read: counted, never raised, never retried
                self.stats["skipped_error"] += len(fresh)
                log.debug("exec log re-read failed: %r", e)
                return 0
        if books is None:               # the read budget is short (or the client cannot read optionally): skip
            self.stats["skipped_budget"] += len(fresh)
            return 0
        t = self.clock()
        groups: dict = {}
        for d in fresh:
            b = books.get(d.info["ticker"])
            groups.setdefault(_key(d.store), (d.store, []))[1].append(
                self._row(t, d.ref, "reread", d.offset_s, d.due_ts, d.info, b, t if b is not None else None))
        for store, rows in groups.values():
            self._write(store, rows)
        self.stats["reads"] += 1
        self.stats["rereads"] += len(fresh)
        return len(fresh)

    def drain(self, deadline: float, sleep: Callable[[float], None] = time.sleep) -> dict:
        """For a one-shot process (the early shadow): wait for each queued re-read and run it, until the queue is
        empty or the next one is due after `deadline` (those are skipped and counted). Returns status()."""
        self.flush()
        while self.queue.next_due() is not None and self.queue.next_due() <= deadline:
            nxt = self.queue.next_due()
            wait = nxt - self.clock()
            if wait > 0:
                sleep(wait)
            self.run_due(now=max(self.clock(), nxt))
        self.stats["skipped_deadline"] += self.queue.clear()
        return self.status()

    def status(self) -> dict:
        return {"on": True, "queued": len(self.queue), "unwritten": sum(len(r) for _, r in self.pending.values()),
                **{k: int(v) for k, v in sorted(self.stats.items())}}
