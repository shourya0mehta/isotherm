"""Kalshi WebSocket feed (shadow first): live order books, trades, our fills and our order updates.

Why: the bot polls order books and fills every 5 seconds. The stream delivers each change as it happens (Kalshi
queues a message within milliseconds of the change). In shadow mode nothing trades off it: the bot compares it with
the REST polls and records how much earlier the stream saw each change, and whether the two ever disagree.

Kalshi's protocol (docs/kalshi_api/websockets__*.md):
- one authenticated connection: the handshake carries the same signed headers as REST, signed over
  "<ts_ms>GET/trade-api/ws/v2"; Kalshi pings every 10 s;
- {"id", "cmd": "subscribe", "params": {"channels": [...], "market_tickers": [...]}} -> "subscribed" with a sid;
- orderbook_delta: an orderbook_snapshot per market, then orderbook_delta messages (price_dollars, delta_fp, side).
  The client asks for use_yes_price (Kalshi will make it the only behavior): NO-side levels then come on the YES price
  scale, and are turned back into NO prices (100 - p) so the books match the REST format;
  every message of a subscription carries seq, consecutive per sid: a gap means a lost message, so the books of that
  subscription are no longer trusted and the client reconnects (fresh snapshots);
- trade (public), fill and user_orders (ours) carry the same fields as the REST objects;
- update_subscription with add_markets / delete_markets changes the market list of a sid.

The client is standard library only (socket, ssl, threading), so the Mac needs no new package: RFC 6455 framing with
masked client frames, no extensions. It runs in one daemon thread that reconnects with backoff; readers take copies
under a lock.
"""
from __future__ import annotations

import base64
import collections
import hashlib
import json
import logging
import os
import socket
import ssl
import struct
import threading
import time
from typing import Callable, Optional
from urllib.parse import urlparse

log = logging.getLogger(__name__)

WS_PROD = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
WS_DEMO = "wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2"
_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class WSClosed(Exception):
    pass


class RawWebSocket:
    """A minimal blocking WebSocket client (RFC 6455): text frames, ping/pong, close; no extensions."""

    def __init__(self, url: str, headers: Optional[dict] = None, timeout: float = 10.0, cafile: Optional[str] = None):
        u = urlparse(url)
        self.secure = u.scheme == "wss"
        self.host = u.hostname
        self.port = u.port or (443 if self.secure else 80)
        self.path = (u.path or "/") + (("?" + u.query) if u.query else "")
        sock = socket.create_connection((self.host, self.port), timeout=timeout)
        if self.secure:
            ctx = ssl.create_default_context(cafile=cafile or _certifi())
            sock = ctx.wrap_socket(sock, server_hostname=self.host)
        self.sock = sock
        self.buf = b""
        self.frag, self.frag_op = [], None
        self.send_lock = threading.Lock()
        key = base64.b64encode(os.urandom(16)).decode()
        lines = [f"GET {self.path} HTTP/1.1", f"Host: {self.host}" + (f":{u.port}" if u.port else ""),
                 "Upgrade: websocket", "Connection: Upgrade", f"Sec-WebSocket-Key: {key}",
                 "Sec-WebSocket-Version: 13"]
        lines += [f"{k}: {v}" for k, v in (headers or {}).items()]
        self.sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
        head = self._read_until(b"\r\n\r\n", limit=65536).decode("latin-1")
        status = head.split("\r\n", 1)[0]
        if " 101 " not in status + " ":
            raise WSClosed(f"handshake refused: {status} {head[:300]!r}")
        hdrs = {ln.split(":", 1)[0].strip().lower(): ln.split(":", 1)[1].strip()
                for ln in head.split("\r\n")[1:] if ":" in ln}
        want = base64.b64encode(hashlib.sha1((key + _GUID).encode()).digest()).decode()
        if hdrs.get("sec-websocket-accept") != want:
            raise WSClosed("handshake: bad Sec-WebSocket-Accept")

    # ------------------------------------------------------------------ io
    def _read_until(self, marker: bytes, limit: int) -> bytes:
        while marker not in self.buf:
            if len(self.buf) > limit:
                raise WSClosed("header too long")
            chunk = self.sock.recv(65536)
            if not chunk:
                raise WSClosed("connection closed during handshake")
            self.buf += chunk
        i = self.buf.index(marker) + len(marker)
        out, self.buf = self.buf[:i], self.buf[i:]
        return out

    def send_frame(self, opcode: int, payload: bytes = b""):
        mask = os.urandom(4)
        n = len(payload)
        if n < 126:
            head = struct.pack("!BB", 0x80 | opcode, 0x80 | n)
        elif n < 65536:
            head = struct.pack("!BBH", 0x80 | opcode, 0x80 | 126, n)
        else:
            head = struct.pack("!BBQ", 0x80 | opcode, 0x80 | 127, n)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        with self.send_lock:
            self.sock.sendall(head + mask + masked)

    def send_text(self, text: str):
        self.send_frame(0x1, text.encode())

    def _frame(self):
        """One complete frame from the buffer as (fin, opcode, payload), or None while it is incomplete. Nothing is
        consumed until the whole frame is there, so a read timeout never desynchronizes the stream."""
        b = self.buf
        if len(b) < 2:
            return None
        fin, op, n, i = b[0] & 0x80, b[0] & 0x0F, b[1] & 0x7F, 2
        if n == 126:
            if len(b) < 4:
                return None
            n, i = struct.unpack("!H", b[2:4])[0], 4
        elif n == 127:
            if len(b) < 10:
                return None
            n, i = struct.unpack("!Q", b[2:10])[0], 10
        mask = None
        if b[1] & 0x80:
            if len(b) < i + 4:
                return None
            mask, i = b[i:i + 4], i + 4
        if len(b) < i + n:
            return None
        data = b[i:i + n]
        self.buf = b[i + n:]
        if mask:
            data = bytes(x ^ mask[k % 4] for k, x in enumerate(data))
        return fin, op, data

    def recv(self) -> Optional[str]:
        """The next text message (a fragmented one is reassembled). A ping is answered and returns None (the
        connection is alive), as do a pong and a binary message; raises WSClosed when the server closes, and
        socket.timeout when nothing arrives in time (any partial frame stays buffered)."""
        while True:
            fr = self._frame()
            if fr is None:
                chunk = self.sock.recv(65536)
                if not chunk:
                    raise WSClosed("connection closed")
                self.buf += chunk
                continue
            fin, op, data = fr
            if op == 0x9:
                self.send_frame(0xA, data)
                return None
            if op == 0xA:
                return None
            if op == 0x8:
                try:
                    self.send_frame(0x8, data[:2])
                except OSError:
                    pass
                code = struct.unpack("!H", data[:2])[0] if len(data) >= 2 else None
                raise WSClosed(f"closed by server (code {code} {data[2:].decode(errors='replace')})")
            if op in (0x1, 0x2):
                self.frag_op, self.frag = op, [data]
            elif op == 0x0:
                self.frag.append(data)
            if fin:
                msg, kind = b"".join(self.frag), self.frag_op
                self.frag, self.frag_op = [], None
                return msg.decode("utf-8") if kind == 0x1 else None

    def close(self):
        try:
            self.send_frame(0x8, struct.pack("!H", 1000))
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


def _certifi() -> Optional[str]:
    """The CA bundle requests would use: REQUESTS_CA_BUNDLE / SSL_CERT_FILE when set, else certifi's."""
    for k in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"):
        if os.environ.get(k):
            return os.environ[k]
    try:
        import certifi
        return certifi.where()
    except Exception:
        return None


class LocalBook:
    """One market's book from a snapshot plus deltas: price (cents) -> contracts, per side (YES bids, NO bids)."""
    __slots__ = ("yes", "no", "ts", "top_ts", "top")

    def __init__(self):
        self.yes, self.no = {}, {}
        self.ts = 0.0          # last change received
        self.top_ts = 0.0      # when the best YES bid / best NO bid last changed
        self.top = None

    def as_rest(self) -> dict:
        """The REST orderbook format (ascending [price_dollars, count] levels), so book_to_quote reads it."""
        f = lambda d: [[f"{p / 100:.4f}", f"{q:.2f}"] for p, q in sorted(d.items()) if q > 0]
        return {"yes_dollars": f(self.yes), "no_dollars": f(self.no)}

    def best(self) -> tuple:
        y = max((p for p, q in self.yes.items() if q > 0), default=None)
        n = max((p for p, q in self.no.items() if q > 0), default=None)
        return y, n


def _cents(x) -> Optional[int]:
    try:
        return int(round(float(x) * 100))
    except (TypeError, ValueError):
        return None


class KalshiWS:
    """Background stream: order books for the markets in `markets` (set_markets), the public trades of those
    markets, our fills and our order updates. Thread-safe readers: book(), events_since(), stats()."""

    def __init__(self, auth, url: str = WS_PROD, channels=("orderbook_delta", "trade", "fill", "user_orders"),
                 headers_fn: Optional[Callable[[], dict]] = None, max_events: int = 20000, stale_s: float = 30.0,
                 cafile: Optional[str] = None):
        self.auth = auth
        self.url = url
        self.path = urlparse(url).path or "/trade-api/ws/v2"
        self.channels = tuple(channels)
        self.headers_fn = headers_fn
        self.cafile = cafile
        self.lock = threading.Lock()
        self.books: dict[str, LocalBook] = {}
        self.events = collections.deque(maxlen=max_events)   # (recv_ts, kind, msg)
        self.want: set = set()
        self.sid_channel: dict[int, str] = {}
        self.seq: dict[int, int] = {}
        self.stale_s = stale_s
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.ws: Optional[RawWebSocket] = None
        self.cmd_id = 0
        self.pending_cmds: dict[int, tuple] = {}
        self.st = collections.Counter()
        self.connected_ts = 0.0
        self.last_msg_ts = 0.0
        self.last_error = ""
        self.lag_ms = collections.deque(maxlen=2000)          # receive time - Kalshi's sending_ts_ms
        self.markets_dirty = threading.Event()
        self.subscribed_markets: set = set()
        self.sid_markets: dict[int, set] = {}                 # sid -> its markets (market channels)
        self.cmd_markets: dict[int, set] = {}                 # subscribe command id -> the markets it asked for
        self.chunk = 100
        self.yes_price_scale = True          # NO levels arrive on the YES scale (use_yes_price)
        self.healthy_session_s = 60.0        # a session that lasted this long resets the reconnect backoff
        self.send_yes_flag = True            # Kalshi will remove the flag (yes-scale then the only behavior)

    # ------------------------------------------------------------- control
    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, name="kalshi-ws", daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        ws = self.ws
        if ws is not None:
            ws.close()
        if self.thread:
            self.thread.join(timeout=5)

    def set_markets(self, tickers):
        tickers = set(tickers)
        with self.lock:
            if tickers == self.want:
                return
            self.want = tickers
        self.markets_dirty.set()

    # ------------------------------------------------------------- readers
    def book(self, ticker: str, max_age_s: Optional[float] = None) -> Optional[dict]:
        """The market's book in REST format with its receive time, or None when unknown, or when the connection
        has been silent for longer than max_age_s (Kalshi pings every 10 s, so silence means trouble)."""
        with self.lock:
            b = self.books.get(ticker)
            if b is None:
                return None
            if max_age_s is not None and time.time() - self.last_msg_ts > max_age_s:
                return None
            out = b.as_rest()
            out["_ts"], out["_top_ts"] = b.ts, b.top_ts
            return out

    def tops(self) -> dict:
        """ticker -> ((best YES bid, best NO bid) in cents, when that top was first seen)."""
        with self.lock:
            return {t: (b.top, b.top_ts) for t, b in self.books.items()}

    def events_since(self, ts: float, kinds=None) -> list:
        with self.lock:
            return [e for e in self.events if e[0] > ts and (kinds is None or e[1] in kinds)]

    def healthy(self) -> bool:
        return bool(self.running and self.ws is not None and time.time() - self.last_msg_ts < self.stale_s)

    def stats(self) -> dict:
        lag = sorted(self.lag_ms)
        return {"connected": self.healthy(), "reconnects": self.st["connects"] - 1 if self.st["connects"] else 0,
                "msgs": self.st["msgs"], "deltas": self.st["deltas"], "snapshots": self.st["snapshots"],
                "trades": self.st["trades"], "fills": self.st["fills"], "order_updates": self.st["user_orders"],
                "seq_gaps": self.st["gaps"], "errors": self.st["errors"], "books": len(self.books),
                "subscriptions": len(self.sid_markets), "market_limit_errors": self.st["market_limit"],
                "silent_s": round(time.time() - self.last_msg_ts, 1) if self.last_msg_ts else None,
                "lag_ms_p50": lag[len(lag) // 2] if lag else None,
                "lag_ms_p95": lag[min(len(lag) - 1, int(0.95 * len(lag)))] if lag else None,
                "last_error": self.last_error[:200]}

    # ------------------------------------------------------------- thread
    def _headers(self) -> dict:
        if self.headers_fn is not None:
            return self.headers_fn()
        return self.auth.headers("GET", self.path) if self.auth is not None else {}

    def _send(self, cmd: str, params: dict) -> int:
        self.cmd_id += 1
        self.pending_cmds[self.cmd_id] = (cmd, params)
        self.ws.send_text(json.dumps({"id": self.cmd_id, "cmd": cmd, "params": params}))
        return self.cmd_id

    def _loop(self):
        backoff = 1.0
        while self.running:
            t_start = time.time()
            try:
                self._session()
            except Exception as e:   # noqa: BLE001 (any failure: reconnect)
                if not self.running:
                    break            # stop() closed the socket under the reader
                self.last_error = repr(e)
                self.st["errors"] += 1
                log.warning("websocket: %s; reconnecting in %.0f s", e, backoff)
            finally:
                ws, self.ws = self.ws, None
                if ws is not None:
                    ws.close()
                with self.lock:
                    self.books.clear()     # nothing is trusted across a reconnect: fresh snapshots follow
            if self.running:
                if time.time() - t_start > self.healthy_session_s:
                    backoff = 1.0          # the session was healthy for a while: this is a fresh problem
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    def _session(self):
        self.ws = RawWebSocket(self.url, self._headers(), cafile=self.cafile)
        self.ws.sock.settimeout(1.0)
        self.st["connects"] += 1
        self.connected_ts = self.last_msg_ts = time.time()
        self.sid_channel.clear()
        self.seq.clear()
        self.pending_cmds.clear()
        # clear the flag before reading the list: a set_markets() that lands after the read sets it again (cleared
        # after the read, a change made in between was lost and the stream kept no market subscriptions)
        self.markets_dirty.clear()
        with self.lock:
            want = set(self.want)
        self.sid_markets.clear()
        self.cmd_markets.clear()
        user_ch = [c for c in self.channels if c in ("fill", "user_orders")]
        if user_ch:
            self._send("subscribe", {"channels": user_ch})
        for c in self.market_channels():
            self._subscribe_chunks(c, sorted(want))
        self.subscribed_markets = want
        log.info("websocket connected (%s), %d markets", self.url, len(want))
        while self.running:
            if self.markets_dirty.is_set():
                self.markets_dirty.clear()
                self._update_markets()
            try:
                txt = self.ws.recv()
            except socket.timeout:
                if time.time() - self.last_msg_ts > self.stale_s:
                    raise WSClosed(f"no message or ping for {self.stale_s:.0f} s")
                continue
            self.last_msg_ts = time.time()
            if txt is None:
                continue
            self._handle(txt, self.last_msg_ts)

    def market_channels(self) -> list:
        return [c for c in self.channels if c in ("orderbook_delta", "trade")]

    def _subscribe_chunks(self, channel: str, tickers: list):
        """Subscribe in chunks of at most `chunk` markets (Kalshi has a per-subscription market limit, error 26)."""
        for i in range(0, len(tickers), self.chunk):
            part = tickers[i:i + self.chunk]
            params = {"channels": [channel], "market_tickers": part}
            if channel == "orderbook_delta" and self.send_yes_flag:
                params["use_yes_price"] = True
            cid = self._send("subscribe", params)
            self.cmd_markets[cid] = set(part)

    def _update_markets(self):
        with self.lock:
            want = set(self.want)
        have = self.subscribed_markets
        if want == have:
            return
        if any(c == "subscribe" and "market_tickers" in p for c, p in self.pending_cmds.values()):
            self.markets_dirty.set()      # a market subscription is not answered yet: try again shortly
            return
        add, drop = sorted(want - have), sorted(have - want)
        for ch in self.market_channels():
            sids = {sid: mk for sid, mk in self.sid_markets.items() if self.sid_channel.get(sid) == ch}
            for sid, mk in sids.items():
                gone = sorted(mk & set(drop))
                if gone:
                    self._send("update_subscription", {"sid": sid, "action": "delete_markets", "market_tickers": gone})
                    mk -= set(gone)
            rest = list(add)
            for sid, mk in sids.items():         # top up subscriptions that have room
                room = self.chunk - len(mk)
                if room > 0 and rest:
                    part, rest = rest[:room], rest[room:]
                    self._send("update_subscription", {"sid": sid, "action": "add_markets", "market_tickers": part})
                    mk |= set(part)
            if rest:
                self._subscribe_chunks(ch, rest)
        if drop:
            with self.lock:
                for t in drop:
                    self.books.pop(t, None)
        self.subscribed_markets = want

    def _handle(self, txt: str, now: float):
        try:
            m = json.loads(txt)
        except ValueError:
            self.st["bad_json"] += 1
            return
        self.st["msgs"] += 1
        typ = m.get("type")
        sid, seq = m.get("sid"), m.get("seq")
        if m.get("sending_ts_ms"):
            self.lag_ms.append(round(now * 1000 - float(m["sending_ts_ms"])))
        if typ == "subscribed":
            msg = m.get("msg") or {}
            sid = int(msg.get("sid"))
            self.sid_channel[sid] = msg.get("channel")
            if msg.get("channel") in ("orderbook_delta", "trade"):
                self.sid_markets[sid] = set(self.cmd_markets.get(m.get("id"), set()))
            self.pending_cmds.pop(m.get("id"), None)
            return
        if typ in ("ok", "unsubscribed", "error") and seq is not None and sid is not None:
            # an answer on a subscription carries its current seq: the same number as the last message, or the next
            # one if the answer takes a number of its own; anything further means messages were lost before it
            sid_i, seq_i = int(sid), int(seq)
            last = self.seq.get(sid_i)
            if last is not None and seq_i not in (last, last + 1):
                self.st["gaps"] += 1
                raise WSClosed(f"sequence gap before a {typ} answer on sid {sid_i}: {last} -> {seq_i}")
            self.seq[sid_i] = seq_i
        if typ in ("ok", "unsubscribed"):
            self.pending_cmds.pop(m.get("id"), None)
            if typ == "ok" and sid is not None and (m.get("msg") or {}).get("market_tickers") is not None:
                self.sid_markets[int(sid)] = set(m["msg"]["market_tickers"])
            if typ == "unsubscribed" and sid is not None:
                self.sid_channel.pop(int(sid), None)
                self.sid_markets.pop(int(sid), None)
            return
        if typ == "error":
            self.st["errors"] += 1
            code = (m.get("msg") or {}).get("code")
            if code == 26:
                self.st["market_limit"] += 1   # per-subscription market limit: smaller chunks next session
                self.chunk = max(10, self.chunk // 2)
            self.last_error = json.dumps(m.get("msg"))[:300]
            cmd = self.pending_cmds.pop(m.get("id"), None) or (None, {})
            log.warning("websocket error answer: %s (command %s)", m.get("msg"), cmd)
            if code == 26:
                raise WSClosed("subscription market limit: reconnecting with smaller subscriptions")
            if code == 11 and "use_yes_price" in cmd[1] and self.send_yes_flag:
                # an invalid parameter on a subscription that carried the flag: Kalshi has removed it (yes-scale
                # prices are then the only behavior), so resubscribe without it
                self.send_yes_flag = False
                raise WSClosed("use_yes_price refused: resubscribing without it")
            return
        if sid is not None and seq is not None:
            sid, seq = int(sid), int(seq)
            last = self.seq.get(sid)
            if last is not None and seq != last + 1:
                self.st["gaps"] += 1
                raise WSClosed(f"sequence gap on sid {sid}: {last} -> {seq}")
            self.seq[sid] = seq
        msg = m.get("msg") or {}
        if typ == "orderbook_snapshot":
            self.st["snapshots"] += 1
            b = LocalBook()
            for side, key in (("yes", "yes_dollars_fp"), ("no", "no_dollars_fp")):
                d = getattr(b, side)
                for p, q in msg.get(key) or msg.get(key.replace("_fp", "")) or []:
                    c = self._level(side, p)
                    if c is not None:
                        d[c] = d.get(c, 0.0) + float(q)    # sub-cent levels add up per cent
            self._touch(msg.get("market_ticker"), b, now)
        elif typ == "orderbook_delta":
            self.st["deltas"] += 1
            t = msg.get("market_ticker")
            c = self._level(msg.get("side"), msg.get("price_dollars"))
            with self.lock:
                b = self.books.get(t)
                if b is None or c is None:
                    self.st["orphan_deltas"] += 1   # before its snapshot: the next snapshot rebuilds the book
                    return
                d = b.yes if msg.get("side") == "yes" else b.no
                q = d.get(c, 0.0) + float(msg.get("delta_fp") or 0)
                if q <= 1e-9:
                    d.pop(c, None)
                else:
                    d[c] = q
            self._touch(t, b, now)
        elif typ in ("trade", "fill", "user_order"):
            self.st["trades" if typ == "trade" else "fills" if typ == "fill" else "user_orders"] += 1
            with self.lock:
                self.events.append((now, typ, msg))

    def _level(self, side, price) -> Optional[int]:
        """A book level's price in cents on the REST scale: YES levels at the YES price, NO levels at the NO price.
        With use_yes_price (asked for in every orderbook subscription) Kalshi sends NO levels on the YES scale."""
        c = _cents(price)
        if c is None:
            return None
        return 100 - c if (side == "no" and self.yes_price_scale) else c

    def _touch(self, ticker, b: LocalBook, now: float):
        if not ticker:
            return
        with self.lock:
            top = b.best()
            b.ts = now
            if top != b.top:
                b.top, b.top_ts = top, now
            self.books[ticker] = b
