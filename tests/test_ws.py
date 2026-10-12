"""The WebSocket feed against a local fake of Kalshi's stream (the real one needs an API key): handshake with the
signed headers, snapshots then deltas into local books, public trades and our fills, market list updates, Kalshi's
pings, a sequence gap (books dropped, reconnect, fresh snapshots), and a server that drops the connection."""
import asyncio
import json
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

websockets = pytest.importorskip("websockets")
from websockets.asyncio.server import serve  # noqa: E402

from wxbot.kalshi.ws import KalshiWS  # noqa: E402
from wxbot.models.fair_value import book_to_quote  # noqa: E402

T1, T2, T3 = "KXHIGHNY-26OCT05-T70", "KXHIGHNY-26OCT05-B70.5", "KXHIGHNY-26OCT05-T75"


class FakeKalshiStream:
    """Answers subscribe / update_subscription like Kalshi; tests push deltas, trades, fills, gaps and closes."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.conns = []
        self.headers = []
        self.cmds = []
        self.books = {T1: {"yes": [["0.0100", "500.00"]], "no": [["0.9700", "800.00"]]},
                      T2: {"yes": [["0.2000", "50.00"]], "no": [["0.7700", "60.00"]]},
                      T3: {"yes": [["0.0100", "300.00"]], "no": [["0.9800", "900.00"]]}}
        self.next_sid = 0
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.ready.wait(5)

    def _run(self):
        asyncio.set_event_loop(self.loop)

        async def main():
            self.done = asyncio.get_running_loop().create_future()
            async with serve(self._handler, "127.0.0.1", 0, ping_interval=None) as server:
                self.port = server.sockets[0].getsockname()[1]
                self.ready.set()
                await self.done
        self.loop.run_until_complete(main())
        self.loop.close()

    async def _handler(self, ws):
        self.headers.append({k.lower(): v for k, v in ws.request.headers.raw_items()})
        state = {"ws": ws, "subs": {}, "seq": {}}
        self.conns.append(state)
        try:
            async for raw in ws:
                m = json.loads(raw)
                self.cmds.append(m)
                p = m.get("params", {})
                if m["cmd"] == "subscribe":
                    for ch in p["channels"]:
                        self.next_sid += 1
                        sid = self.next_sid
                        state["subs"][sid] = (ch, set(p.get("market_tickers") or []))
                        state.setdefault("yes_scale", {})[sid] = bool(p.get("use_yes_price"))
                        state["seq"][sid] = 0
                        await ws.send(json.dumps({"id": m["id"], "type": "subscribed",
                                                  "msg": {"channel": ch, "sid": sid}}))
                        if ch == "orderbook_delta":
                            for t in p["market_tickers"]:
                                await self._snapshot(state, sid, t)
                elif m["cmd"] == "update_subscription":
                    sid = p["sid"]
                    ch, mk = state["subs"][sid]
                    if p["action"] == "add_markets":
                        mk |= set(p["market_tickers"])
                    else:
                        mk -= set(p["market_tickers"])
                    state["seq"][sid] += 1
                    await ws.send(json.dumps({"id": m["id"], "type": "ok", "sid": sid, "seq": state["seq"][sid],
                                              "msg": {"market_tickers": sorted(mk)}}))
                    if ch == "orderbook_delta" and p["action"] == "add_markets":
                        for t in p["market_tickers"]:
                            await self._snapshot(state, sid, t)
        except websockets.ConnectionClosed:
            pass

    async def _snapshot(self, state, sid, t):
        state["seq"][sid] += 1
        b = self.books[t]
        no = b["no"]
        if state.get("yes_scale", {}).get(sid):     # use_yes_price: NO levels on the YES scale
            no = [[f"{1 - float(p):.4f}", q] for p, q in no]
        await state["ws"].send(json.dumps({"type": "orderbook_snapshot", "sid": sid, "seq": state["seq"][sid],
                                           "sending_ts_ms": int(time.time() * 1000),
                                           "msg": {"market_ticker": t, "market_id": "x", "yes_dollars_fp": b["yes"],
                                                   "no_dollars_fp": no}}))

    # ---- test controls (run on the server loop)
    def _call(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(5)

    def sid(self, channel):
        st = self.conns[-1]
        return next(s for s, (ch, _) in st["subs"].items() if ch == channel)

    def push(self, channel, typ, msg, skip_seq=False, seq=True):
        """msg prices as on the REST scale (NO levels at the NO price); a NO delta goes out on the YES scale when
        the subscription asked for use_yes_price, as Kalshi does."""
        st = self.conns[-1]
        sid = self.sid(channel)
        if typ == "orderbook_delta" and msg.get("side") == "no" and st.get("yes_scale", {}).get(sid):
            msg = {**msg, "price_dollars": f"{1 - float(msg['price_dollars']):.4f}"}
        out = {"type": typ, "sid": sid, "msg": msg, "sending_ts_ms": int(time.time() * 1000)}
        if seq:
            st["seq"][sid] += 2 if skip_seq else 1
            out["seq"] = st["seq"][sid]
        self._call(st["ws"].send(json.dumps(out)))

    def ping(self):
        self._call(self.conns[-1]["ws"].ping(b"heartbeat"))

    def drop(self):
        self._call(self.conns[-1]["ws"].close())

    def stop(self):
        self.loop.call_soon_threadsafe(lambda: self.done.done() or self.done.set_result(None))
        self.thread.join(5)


def wait_for(cond, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


class Auth:
    def headers(self, method, path):
        return {"KALSHI-ACCESS-KEY": "kid", "KALSHI-ACCESS-SIGNATURE": f"sig:{method}{path}",
                "KALSHI-ACCESS-TIMESTAMP": "1"}


@pytest.fixture
def stream():
    s = FakeKalshiStream()
    yield s
    s.stop()


def client(stream, **kw):
    ws = KalshiWS(Auth(), url=f"ws://127.0.0.1:{stream.port}/trade-api/ws/v2", **kw)
    ws.set_markets([T1, T2])
    ws.start()
    return ws


def test_books_from_snapshot_and_deltas_match_the_rest_format(stream):
    ws = client(stream)
    try:
        assert wait_for(lambda: ws.book(T1) and ws.book(T2))
        h = stream.headers[0]
        assert h["kalshi-access-key"] == "kid" and h["kalshi-access-signature"] == "sig:GET/trade-api/ws/v2"
        q = book_to_quote(ws.book(T2))
        assert (q.yes_bid, q.yes_ask, q.yes_bid_size, q.yes_ask_size) == (20, 23, 50.0, 60.0)
        stream.push("orderbook_delta", "orderbook_delta",
                    {"market_ticker": T2, "price_dollars": "0.2100", "delta_fp": "15.00", "side": "yes"})
        stream.push("orderbook_delta", "orderbook_delta",
                    {"market_ticker": T2, "price_dollars": "0.7700", "delta_fp": "-60.00", "side": "no"})
        assert wait_for(lambda: book_to_quote(ws.book(T2)).yes_bid == 21 and not ws.book(T2)["no_dollars"])
        b = ws.book(T2)
        assert b["no_dollars"] == [] and b["yes_dollars"] == [["0.2000", "50.00"], ["0.2100", "15.00"]]
        assert ws.tops()[T2][0] == (21, None)
        st = ws.stats()
        assert st["connected"] and st["snapshots"] == 2 and st["deltas"] == 2 and st["seq_gaps"] == 0
    finally:
        ws.stop()


def test_trades_fills_and_order_updates_are_recorded_with_arrival_times(stream):
    ws = client(stream)
    try:
        assert wait_for(lambda: len(stream.conns) == 1 and len(stream.conns[0]["subs"]) == 4)
        t0 = time.time()
        stream.push("trade", "trade", {"trade_id": "t1", "market_ticker": T1, "yes_price_dollars": "0.0100",
                                       "count_fp": "5.00", "taker_side": "no"})
        stream.push("fill", "fill", {"trade_id": "f1", "order_id": "o1", "market_ticker": T1, "count_fp": "5.00",
                                     "yes_price_dollars": "0.0300", "book_side": "ask"}, seq=False)
        stream.push("user_orders", "user_order", {"order_id": "o1", "status": "executed", "ticker": T1}, seq=False)
        assert wait_for(lambda: len(ws.events_since(t0 - 1)) == 3)
        kinds = [e[1] for e in ws.events_since(t0 - 1)]
        assert kinds == ["trade", "fill", "user_order"]
        assert [e[2]["trade_id"] for e in ws.events_since(t0 - 1, kinds={"fill"})] == ["f1"]
    finally:
        ws.stop()


def test_market_list_changes_go_out_as_subscription_updates(stream):
    ws = client(stream)
    try:
        assert wait_for(lambda: ws.book(T1) and ws.book(T2))
        ws.set_markets([T2, T3])
        assert wait_for(lambda: ws.book(T3) is not None and ws.book(T1) is None)
        ups = [c for c in stream.cmds if c["cmd"] == "update_subscription"]
        assert {(c["params"]["action"], tuple(c["params"]["market_tickers"])) for c in ups} == \
            {("add_markets", (T3,)), ("delete_markets", (T1,))}
        # the ok answers carry the subscription's seq: later deltas are not taken for a gap
        stream.push("orderbook_delta", "orderbook_delta",
                    {"market_ticker": T3, "price_dollars": "0.0200", "delta_fp": "10.00", "side": "yes"})
        assert wait_for(lambda: book_to_quote(ws.book(T3)).yes_bid == 2)
        assert ws.stats()["seq_gaps"] == 0 and ws.stats()["reconnects"] == 0
    finally:
        ws.stop()


def test_a_sequence_gap_drops_the_books_and_reconnects_for_fresh_snapshots(stream):
    ws = client(stream)
    try:
        assert wait_for(lambda: ws.book(T1) and ws.book(T2))
        stream.push("orderbook_delta", "orderbook_delta",
                    {"market_ticker": T1, "price_dollars": "0.0200", "delta_fp": "10.00", "side": "yes"},
                    skip_seq=True)
        assert wait_for(lambda: ws.stats()["seq_gaps"] == 1)
        assert wait_for(lambda: len(stream.conns) == 2, timeout=8)       # reconnected (1 s backoff)
        assert wait_for(lambda: ws.book(T1) is not None)
        assert book_to_quote(ws.book(T1)).yes_bid == 1                   # the snapshot, not the delta after the gap
        assert ws.stats()["reconnects"] == 1
    finally:
        ws.stop()


def test_pings_keep_a_quiet_connection_alive_and_silence_or_a_drop_reconnects(stream):
    ws = client(stream, stale_s=2.0)
    try:
        assert wait_for(lambda: ws.book(T1) is not None)
        for _ in range(4):                     # 3 s with only pings: still the same connection
            time.sleep(0.75)
            stream.ping()
        assert len(stream.conns) == 1 and ws.healthy()
        stream.drop()
        assert wait_for(lambda: len(stream.conns) == 2, timeout=8)
        assert wait_for(lambda: ws.book(T1) is not None)
        time.sleep(3.5)                        # no message and no ping for longer than stale_s
        assert wait_for(lambda: len(stream.conns) >= 3, timeout=8)
        assert ws.book(T1, max_age_s=0.0) is None                       # a reader can ask for a fresh book only
    finally:
        ws.stop()


def test_large_messages_arrive_whole():
    """A frame bigger than one 64 KB read (a deep snapshot) is reassembled; sub-cent levels add up per cent."""
    s = FakeKalshiStream()
    try:
        s.books[T1] = {"yes": [["0.0100", "1.00"], ["0.0104", "2.00"], ["0.0200", "3.00"]] + [["0.0300", "1.00"]] * 6000,
                       "no": [["0.9700", "1.00"]]}
        ws = client(s)
        try:
            assert wait_for(lambda: ws.book(T1) is not None)
            assert ws.book(T1)["yes_dollars"] == [["0.0100", "3.00"], ["0.0200", "3.00"], ["0.0300", "6000.00"]]
        finally:
            ws.stop()
    finally:
        s.stop()


def test_the_bot_compares_the_stream_with_its_rest_polls(tmp_path, monkeypatch):
    """Live bot (simulated REST exchange) with the stream in shadow: the stream's books agree with the polls, a
    book change the stream delivered first shows up as lead time, and so does a fill."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from test_live_loop import FakeKalshi
    import wxbot.app.runner as runner
    k = Ed25519PrivateKey.generate()
    monkeypatch.setenv("KALSHI_KEY_ID", "kid")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY", k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                             serialization.NoEncryption()).decode())
    monkeypatch.delenv("KALSHI_PRIVATE_KEY_PATH", raising=False)
    monkeypatch.delenv("WXBOT_NTFY_TOPIC", raising=False)
    fake = FakeKalshi()
    stream = FakeKalshiStream()
    stream.books = {}
    for m in fake.markets:
        yb, ya = fake.quotes[m["ticker"].rsplit("-", 1)[1]]
        stream.books[m["ticker"]] = {"yes": [[f"{yb / 100:.4f}", "500.00"]], "no": [[f"{(100 - ya) / 100:.4f}", "800.00"]]}
    cfg = {"mode": "live", "env": "prod", "db_path": str(tmp_path / "live.sqlite"), "status_path": str(tmp_path / "st.json"),
           "loop": {"tick_seconds": 2, "universe_every_s": 600, "books_every_s": 1, "trades_every_s": 10,
                    "metar_every_s": 300, "nbm_every_s": 900, "index_every_s": 60, "settle_every_s": 900,
                    "positions_every_s": 1},
           "sleeve_a": {"enabled": True, "series": ["KXHIGHNY"], "decided_only": True, "decided_prob": 0.97,
                        "min_edge_c": 2.0, "improve_if_edge_c": 99, "clip_contracts": 12, "decided_max_cost_usd": 10,
                        "min_price_c": 60, "max_price_c": 98},
           "sleeve_b": {"enabled": False, "shadow": False}, "tape_archive": {"enabled": False},
           "risk": {"max_loss_per_bracket_usd": 8, "max_loss_per_event_usd": 10, "max_total_open_risk_usd": 75,
                    "max_cash_deployed_usd": 200, "account_stale_seconds": 30},
           "execution": {"order_ttl_s": 1200, "ttl_by_sleeve": {"A": 300}}, "dashboard_every_s": 10 ** 9,
           "ws": {"enabled": True, "url": f"ws://127.0.0.1:{stream.port}/trade-api/ws/v2"}}
    try:
        bot = runner.Bot(cfg, base_dir=str(tmp_path))
        bot.rest = fake
        bot.live.rest = fake
        bot.nbm.refresh = lambda *a, **kw: None
        monkeypatch.setattr(runner.obs_mod, "fetch_metars", lambda *a, **kw: {})
        bot.ws.start()
        bot.step()                                            # universe, first REST books: the stream subscribes
        assert wait_for(lambda: len(bot.ws.tops()) == 5)
        bot.next_due["books"] = 0
        bot.step()
        st = bot.ws_status()
        assert st["compared"] == 5 and st["agree_pct"] == 100.0
        # the book moves: the stream sees it first, the next poll finds it
        t75 = fake.markets[4]["ticker"]
        stream.push("orderbook_delta", "orderbook_delta",
                    {"market_ticker": t75, "price_dollars": "0.0200", "delta_fp": "40.00", "side": "yes"})
        assert wait_for(lambda: bot.ws.tops()[t75][0][0] == 2)
        time.sleep(0.3)
        fake.quotes["T75"] = (2, 3)
        bot.next_due["books"] = 0
        bot.step()
        st = bot.ws_status()
        assert st["agree_pct"] == 100.0 and st["lead_n"] == 1 and st["lead_s_p50"] >= 0.3
        # a fill: the stream first, then the poll
        stream.push("fill", "fill", {"trade_id": "tr-1", "order_id": "o9", "market_ticker": t75, "count_fp": "3.00",
                                     "yes_price_dollars": "0.0300", "book_side": "ask"}, seq=False)
        assert wait_for(lambda: bot.ws.events_since(0, kinds={"fill"}))
        time.sleep(0.2)
        fake.fills = lambda min_ts=None, **kw: [{"fill_id": "fl-1", "trade_id": "tr-1", "order_id": "o9",
                                                 "market_ticker": t75, "book_side": "ask", "yes_price_dollars": "0.0300",
                                                 "count_fp": "3.00", "created_time": "2026-10-04T12:00:00Z"}]
        bot.live.my_order_ids.add("o9")
        bot.refresh_live_state()
        st = bot.ws_status()
        assert st["fill_lead_n"] == 1 and st["fill_lead_s_p50"] >= 0.2
        assert bot.write_status()["ws_shadow"]["connected"]
    finally:
        try:
            bot.ws.stop()
        except Exception:
            pass
        stream.stop()


def test_a_market_list_set_while_connecting_is_never_lost(stream):
    """The race behind a flaky test: a set_markets() landing right after the session read the market list (and
    before it cleared the 'changed' flag) was lost, and the stream kept no market subscriptions."""
    ws = KalshiWS(Auth(), url=f"ws://127.0.0.1:{stream.port}/trade-api/ws/v2")

    class TriggerLock:
        """The stream's lock; on the first release by the stream thread, the bot changes the market list."""
        def __init__(self):
            self.lk, self.fired = threading.Lock(), False

        def __enter__(self):
            self.lk.acquire()

        def __exit__(self, *a):
            self.lk.release()
            if not self.fired and threading.current_thread().name == "kalshi-ws":
                self.fired = True
                ws.set_markets([T1, T2])
    ws.lock = TriggerLock()
    ws.start()
    try:
        assert wait_for(lambda: ws.book(T1) is not None and ws.book(T2) is not None)
    finally:
        ws.stop()


def test_orderbook_subscriptions_ask_for_yes_scale_prices_and_come_in_chunks(stream):
    """Kalshi will make use_yes_price the only behavior: the client asks for it and turns NO levels back into NO
    prices. Market lists are split into subscriptions of at most `chunk` markets (Kalshi's per-subscription limit),
    and later additions top up a subscription that has room before opening a new one."""
    ws = KalshiWS(Auth(), url=f"ws://127.0.0.1:{stream.port}/trade-api/ws/v2")
    ws.chunk = 2
    ws.set_markets([T1, T2, T3])
    ws.start()
    try:
        assert wait_for(lambda: all(ws.book(t) for t in (T1, T2, T3)))
        subs = [c for c in stream.cmds if c["cmd"] == "subscribe" and c["params"]["channels"] == ["orderbook_delta"]]
        assert [len(c["params"]["market_tickers"]) for c in subs] == [2, 1]
        assert all(c["params"]["use_yes_price"] is True for c in subs)
        assert book_to_quote(ws.book(T2)).yes_ask == 23                       # NO bid 77c, sent as 0.23 YES
        stream.push("orderbook_delta", "orderbook_delta",
                    {"market_ticker": T1, "price_dollars": "0.9800", "delta_fp": "5.00", "side": "no"})
        assert wait_for(lambda: book_to_quote(ws.book(T1)).yes_ask == 2)     # NO bid 98c -> YES ask 2c
        ws.set_markets([T2, T3])                                             # T1 leaves its subscription...
        assert wait_for(lambda: ws.book(T1) is None)
        stream.books["KXHIGHNY-26OCT05-B72.5"] = {"yes": [["0.4500", "5.00"]], "no": [["0.5200", "5.00"]]}
        ws.set_markets([T2, T3, "KXHIGHNY-26OCT05-B72.5"])                    # ...which then has room again
        assert wait_for(lambda: ws.book("KXHIGHNY-26OCT05-B72.5") is not None)
        adds = [c for c in stream.cmds if c["cmd"] == "update_subscription" and c["params"]["action"] == "add_markets"]
        assert len([c for c in stream.cmds if c["cmd"] == "subscribe"
                    and c["params"]["channels"] == ["orderbook_delta"]]) == 2 and adds
        assert ws.stats()["subscriptions"] == 4                              # 2 book + 2 trade subscriptions
    finally:
        ws.stop()


def test_an_answer_that_skips_messages_is_a_gap(stream):
    ws = client(stream)
    try:
        assert wait_for(lambda: ws.book(T1) and ws.book(T2))
        st = stream.conns[-1]
        sid = stream.sid("orderbook_delta")
        st["seq"][sid] += 3                                 # two messages lost, then an ok answer
        stream._call(st["ws"].send(json.dumps({"id": 99, "type": "ok", "sid": sid, "seq": st["seq"][sid],
                                               "msg": {"market_tickers": [T1, T2]}})))
        assert wait_for(lambda: ws.stats()["seq_gaps"] == 1)
        assert wait_for(lambda: len(stream.conns) == 2, timeout=8)
    finally:
        ws.stop()


def test_the_market_limit_error_reconnects_with_smaller_subscriptions(stream):
    ws = client(stream)
    try:
        assert wait_for(lambda: ws.book(T1) and ws.book(T2))
        st = stream.conns[-1]
        stream._call(st["ws"].send(json.dumps({"id": 1, "type": "error",
                                               "msg": {"code": 26, "msg": "Subscription market limit exceeded"}})))
        assert wait_for(lambda: len(stream.conns) == 2, timeout=8)
        assert ws.chunk == 50 and ws.stats()["market_limit_errors"] == 1
        assert wait_for(lambda: ws.book(T1) is not None)
    finally:
        ws.stop()


def test_the_reconnect_wait_starts_over_after_a_healthy_session(stream, monkeypatch):
    import wxbot.kalshi.ws as wsmod
    waits = []
    real_sleep = time.sleep
    monkeypatch.setattr(wsmod.time, "sleep", lambda s: (waits.append(s), real_sleep(0.01)))
    ws = client(stream)
    ws.healthy_session_s = 0.3
    try:
        assert wait_for(lambda: ws.book(T1) is not None)
        for k in range(3):                       # three quick drops: the wait grows 1, 2, 4
            stream.drop()
            assert wait_for(lambda: len(stream.conns) == k + 2, timeout=8)
        real_sleep(0.5)                          # then a healthy session...
        stream.drop()                            # ...and the next wait is back to 1 s
        assert wait_for(lambda: len(stream.conns) == 5, timeout=8)
        assert [w for w in waits if w >= 1][:4] == [1.0, 2.0, 4.0, 1.0]
    finally:
        ws.stop()


# ------------------------------------------------------------------ the shadow comparison (round 4 review)
class TopsOnly:
    def __init__(self):
        self.top = {}

    def healthy(self):
        return True

    def tops(self):
        return dict(self.top)


def bare_cmp():
    import collections
    import types
    import wxbot.app.runner as runner
    b = runner.Bot.__new__(runner.Bot)
    b.ws = TopsOnly()
    b.ws_cmp = {"n": 0, "agree": 0, "lead_s": collections.deque(maxlen=2000),
                "fill_lead_s": collections.deque(maxlen=500), "stuck": 0, "examples": [], "rest_top": {}, "mismatch": {}}
    b.next_due = collections.defaultdict(float)
    b.store = types.SimpleNamespace(log=lambda *a, **k: None)
    return b


def lvl(yb, nb):
    return {"yes_dollars": [[f"{yb / 100:.4f}", "10.00"]], "no_dollars": [[f"{nb / 100:.4f}", "10.00"]]}


def test_a_stream_that_lags_shows_negative_leads():
    b, t, t0 = bare_cmp(), T2, 1000.0
    b.ws.top[t] = ((20, 77), t0 - 50)
    b.ws_compare({t: lvl(20, 77)}, t0)
    for k in range(1, 6):                       # the poll sees each new top first; the stream gets it 2 s later
        t0 += 5
        b.ws_compare({t: lvl(20 + k, 77)}, t0)
        b.ws.top[t] = ((20 + k, 77), t0 + 2)
    b.ws_compare({t: lvl(25, 77)}, t0 + 5)
    assert list(b.ws_cmp["lead_s"]) == [-2.0] * 5


def test_a_stream_that_moves_to_another_wrong_top_is_still_flagged():
    b, t, t0 = bare_cmp(), T2, 1000.0
    b.ws.top[t] = ((20, 77), t0 - 50)
    b.ws_compare({t: lvl(20, 77)}, t0)
    b.ws.top[t] = ((19, 77), t0 + 1)                       # wrong once...
    for _ in range(10):
        t0 += 5
        b.ws_compare({t: lvl(20, 77)}, t0)
    first = b.ws_cmp["stuck"]
    assert first >= 1
    b.ws.top[t] = ((18, 77), t0 + 1)                       # ...moves, still wrong
    for _ in range(10):
        t0 += 5
        b.ws_compare({t: lvl(20, 77)}, t0)
    assert b.ws_cmp["stuck"] > first


def test_fills_the_poll_saw_first_give_negative_leads_and_missing_ones_are_counted(tmp_path, monkeypatch):
    import collections
    import types
    import wxbot.app.runner as runner
    from test_live_loop import FakeKalshi
    from wxbot.execution.live import LiveOrderManager

    class Events:
        def __init__(self):
            self.ev = []

        def events_since(self, ts, kinds=None):
            return [e for e in self.ev if e[0] > ts]
    clock = [1_000_000.0]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    b = bare_cmp()
    b.ws = Events()
    fake = FakeKalshi()
    b.live = LiveOrderManager(fake)
    b.live.my_order_ids.add("o9")
    fills = [{"fill_id": "f1", "trade_id": "tr1", "order_id": "o9", "market_ticker": T2, "book_side": "ask",
              "yes_price_dollars": "0.0300", "count_fp": "1.00", "created_time": "2026-10-04T12:00:00Z"}]
    fake.fills = lambda min_ts=None, **kw: list(fills)
    b.rest, b.store, b.counters, b.mode = fake, types.SimpleNamespace(add_fill=lambda *a, **k: False), \
        collections.defaultdict(int), "live"
    b.positions, b.fair_cache = {}, {}
    b.model_tag = lambda: ""
    b.refresh_live_state()                           # the poll has tr1; the stream does not (yet)
    clock[0] += 3
    b.ws.ev.append((clock[0], "fill", {"trade_id": "tr1"}))
    b.refresh_live_state()
    assert list(b.ws_cmp["fill_lead_s"]) == [-3.0]
    fills[0] = {**fills[0], "fill_id": "f2", "trade_id": "tr2"}
    b.refresh_live_state()                           # tr2: never on the stream
    clock[0] += 1000
    b.refresh_live_state()
    assert b.ws_cmp["fill_missing"] == 1


def test_when_kalshi_removes_the_yes_price_flag_the_client_resubscribes_without_it(stream):
    ws = client(stream)
    try:
        assert wait_for(lambda: ws.book(T1) is not None)
        sub = next(c for c in stream.cmds if c["cmd"] == "subscribe" and c["params"]["channels"] == ["orderbook_delta"])
        ws.pending_cmds[sub["id"]] = ("subscribe", sub["params"])     # as if its answer had not come yet
        st = stream.conns[-1]
        stream._call(st["ws"].send(json.dumps({"id": sub["id"], "type": "error",
                                               "msg": {"code": 11, "msg": "Invalid parameter"}})))
        assert wait_for(lambda: len(stream.conns) == 2, timeout=8)
        assert wait_for(lambda: ws.book(T1) is not None)
        last = [c for c in stream.cmds if c["cmd"] == "subscribe" and c["params"]["channels"] == ["orderbook_delta"]][-1]
        assert "use_yes_price" not in last["params"] and ws.yes_price_scale
    finally:
        ws.stop()
