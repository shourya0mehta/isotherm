"""Read live books/trades from the market-data recorder's hourly JSONL files instead of polling Kalshi again.
Lets a paper bot share one data stream with the recorder (halves API usage on a shared, unauthenticated IP)."""
from __future__ import annotations

import glob
import json
import os
from typing import Optional


class RecorderFeed:
    def __init__(self, data_dir: str):
        self.dir = data_dir
        self.pos: dict[str, int] = {}
        self.books: dict[str, dict] = {}
        self.book_ts: dict[str, float] = {}

    def _new_lines(self, kind: str) -> list[dict]:
        out = []
        files = sorted(glob.glob(os.path.join(self.dir, f"{kind}_*.jsonl")))[-3:]
        for fn in files:
            start = self.pos.get(fn, 0)
            size = os.path.getsize(fn)
            if size <= start:
                continue
            with open(fn, "r") as f:
                f.seek(start)
                chunk = f.read()
            last_nl = chunk.rfind("\n")
            if last_nl < 0:
                continue
            self.pos[fn] = start + last_nl + 1
            for line in chunk[:last_nl].splitlines():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return out

    def skip_to_end(self):
        for kind in ("books", "trades"):
            for fn in sorted(glob.glob(os.path.join(self.dir, f"{kind}_*.jsonl")))[-3:]:
                self.pos[fn] = os.path.getsize(fn)

    def poll_books(self) -> dict:
        for r in self._new_lines("books"):
            self.books[r["t"]] = {"yes_dollars": r.get("y") or [], "no_dollars": r.get("n") or []}
            self.book_ts[r["t"]] = r["ts"]
        return self.books

    def poll_trades(self) -> list[dict]:
        return self._new_lines("trades")

    def latest_book_age(self, ticker: str, now: float) -> Optional[float]:
        t = self.book_ts.get(ticker)
        return None if t is None else now - t
