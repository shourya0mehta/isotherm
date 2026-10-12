"""Execution log (outside review, Oct 8): the book at each taker decision and order and at +1, +5, +15 minutes, the
order's answer for real orders, re-reads through the read limiter that are skipped (never waited for) when the budget
is short, and order paths that do not change, with the option off or on."""
import datetime as dt
import time
from collections import defaultdict

import pytest

from wxbot.execution.exec_log import DueQueue, ExecLogConfig, ExecLogger, book_fields
from wxbot.kalshi.rest import KalshiREST, TokenBucket
from wxbot.models.fair_value import LadderQuote
from wxbot.store.db import Store
from wxbot.strategy import taker
from wxbot.strategy.types import EventBook

# YES bids 30c x10, 29c x5, 25c x8; NO bids 65c x7, 64c x4, 60c x20 -> YES offers 35c x7, 36c x4, 40c x20
BOOK = {"yes_dollars": [["0.25", "8"], ["0.29", "5"], ["0.30", "10"]],
        "no_dollars": [["0.60", "20"], ["0.64", "4"], ["0.65", "7"]]}
LATER = {"yes_dollars": [["0.31", "2"]], "no_dollars": [["0.62", "9"]]}     # the book a minute on: 31c / 38c


def rows(store, where=""):
    cur = store.c.execute(f"SELECT * FROM exec_log {where} ORDER BY ts, offset_s")
    names = [d[0] for d in cur.description]
    return [dict(zip(names, r)) for r in cur.fetchall()]


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class Reader:
    """The REST client's optional batch read: records each call, answers with `books` (None: budget short)."""

    def __init__(self, books=None):
        self.books, self.calls = books, []

    def orderbooks_if_budget(self, tickers, reserve_frac=0.5, timeout=None):
        self.calls.append((list(tickers), reserve_frac, timeout))
        return None if self.books is None else {t: self.books[t] for t in tickers if t in self.books}


# ----------------------------------------------------------------------------------------------------- book fields
def test_book_fields_give_the_touch_and_the_depth_at_and_within_2c_of_our_price():
    f = book_fields(BOOK, "yes", 0.35)
    assert f == {"yes_bid": 0.30, "yes_bid_size": 10.0, "yes_ask": 0.35, "yes_ask_size": 7.0, "depth_at": 7.0,
                 "depth_2c": 11.0}                                         # 35c x7 + 36c x4; 40c is beyond 2c
    g = book_fields(BOOK, "no", 0.70)                                      # NO offers: 70c x10, 71c x5, 75c x8
    assert (g["depth_at"], g["depth_2c"]) == (10.0, 15.0)
    assert set(book_fields(None, "yes", 0.35).values()) == {None}
    e = book_fields({}, "yes", 0.35)
    assert e["yes_bid"] is None and e["yes_ask"] is None and e["depth_at"] == 0.0 and e["depth_2c"] == 0.0


# ------------------------------------------------------------------------------------------------------- due-queue
def test_due_queue_hands_out_reads_at_their_times_once_each():
    q, s1, s2 = DueQueue(), object(), object()
    info = {"ticker": "T"}
    assert q.add(160.0, "r1", 60, s1, info) and q.add(400.0, "r1", 300, s1, info) and q.add(160.0, "r2", 60, s1, info)
    assert not q.add(170.0, "r1", 60, s1, info)                           # the same ref and offset: ignored
    assert q.add(160.0, "r1", 60, s2, info)                               # another store's ref is its own
    assert q.next_due() == 160.0 and len(q) == 4
    assert q.pop_due(159.9) == []
    got = q.pop_due(160.0)
    assert [(d.ref, d.offset_s) for d in got] == [("r1", 60), ("r2", 60), ("r1", 60)] and len(q) == 1
    assert not q.add(160.0, "r1", 60, s1, info)                           # handed out already: still ignored
    assert [(d.ref, d.offset_s, d.due_ts) for d in q.pop_due(1e9)] == [("r1", 300, 400.0)] and q.next_due() is None


def test_record_schedules_one_set_of_rereads_per_ref_and_they_log_the_new_book(tmp_path):
    clock = Clock()
    t0 = clock.t
    st = Store(str(tmp_path / "x.sqlite"))
    rd = Reader({"T1": LATER, "T2": BOOK})
    xl = ExecLogger(rd, ExecLogConfig(enabled=True), mode="paper", clock=clock)
    xl.record(st, "decision", "X:E:T1:yes", "X", "E", "T1", "yes", 0.35, BOOK, t0 - 4)
    clock.t += 0.5
    xl.record(st, "send", "X:E:T1:yes", "X", "E", "T1", "yes", 0.35, BOOK, t0 - 4,
              {"order_id": "o1", "limit": 0.35, "requested": 5.0, "filled": 5.0, "avg_price": 0.35, "fee": 0.08,
               "status": "filled"})
    assert len(xl.queue) == 3 and sorted(d.due_ts - t0 for d in xl.queue.heap) == [60, 300, 900]   # decision's times
    assert not st.c.execute("SELECT 1 FROM sqlite_master WHERE name = 'exec_log'").fetchone()   # kept in memory
    assert xl.flush() == 2
    d, s = rows(st)
    assert d["kind"] == "decision" and d["book_ts"] == t0 - 4 and (d["yes_ask"], d["depth_at"]) == (0.35, 7.0)
    assert s["kind"] == "send" and (s["order_id"], s["limit_price"], s["count"], s["filled"], s["avg_price"]) == (
        "o1", 0.35, 5.0, 5.0, 0.35) and s["status"] == "filled" and s["mode"] == "paper"
    xl.record(st, "decision", "X:E:T2:no", "X", "E", "T2", "no", 0.70, BOOK, t0)    # another ref, same due times
    assert xl.run_due(t0 + 59.9) == 0 and rd.calls == []
    clock.t = t0 + 61
    assert xl.run_due() == 2 and rd.calls == [(["T1", "T2"], 0.5, 3.0)]   # one batch read, each ticker once
    r1 = rows(st, "WHERE kind = 'reread' AND ticker = 'T1'")[0]
    assert (r1["offset_s"], r1["due_ts"], r1["ts"]) == (60, t0 + 60, t0 + 61)
    assert (r1["yes_bid"], r1["yes_ask"], r1["yes_ask_size"], r1["depth_at"], r1["depth_2c"]) == (0.31, 0.38, 9.0,
                                                                                                  0.0, 0.0)
    assert r1["order_id"] is None and r1["ref"] == "X:E:T1:yes"


def test_a_decision_without_a_book_reads_it_at_the_next_pass(tmp_path):
    clock = Clock()
    st = Store(str(tmp_path / "r.sqlite"))
    rd = Reader({"KXRAIN-26OCT09-NYC": LATER})
    xl = ExecLogger(rd, ExecLogConfig(enabled=True, offsets_s=(60,)), mode="live", clock=clock)
    xl.record(st, "decision", "R:K:KXRAIN-26OCT09-NYC:yes", "R", "K", "KXRAIN-26OCT09-NYC", "yes", 0.36)
    assert xl.run_due() == 1                                               # offset 0: right away
    d = rows(st, "WHERE kind = 'decision'")[0]
    assert d["yes_bid"] is None and d["book_ts"] is None and d["depth_at"] is None
    r = rows(st, "WHERE kind = 'reread'")[0]
    assert r["offset_s"] == 0 and r["yes_ask"] == 0.38 and r["depth_at"] == 0.0 and r["depth_2c"] == 9.0   # 38c


def test_rereads_are_skipped_silently_when_the_budget_is_short_or_they_are_late(tmp_path, caplog):
    clock = Clock()
    t0 = clock.t
    st = Store(str(tmp_path / "s.sqlite"))
    rd = Reader(None)                                                      # the limiter cannot spare the read
    xl = ExecLogger(rd, ExecLogConfig(enabled=True), clock=clock)
    xl.record(st, "decision", "X:E:T1:yes", "X", "E", "T1", "yes", 0.35, BOOK, t0)
    with caplog.at_level("DEBUG"):
        assert xl.run_due(t0 + 60) == 0
    assert xl.stats["skipped_budget"] == 1 and len(rd.calls) == 1 and len(rows(st)) == 1
    assert not [r for r in caplog.records if r.levelno >= 20]             # silent: nothing at INFO or above
    assert xl.run_due(t0 + 61) == 0 and len(rd.calls) == 1                # skipped, not retried
    rd.books = {"T1": BOOK}
    assert xl.run_due(t0 + 300 + 121) == 0 and xl.stats["skipped_late"] == 1 and len(rd.calls) == 1   # 2 min late
    assert xl.run_due(t0 + 900) == 1 and len(rows(st)) == 2
    st2 = Store(str(tmp_path / "s2.sqlite"))

    class Broken:
        def orderbooks_if_budget(self, *a, **k):
            raise TimeoutError("read timed out")
    xb = ExecLogger(Broken(), ExecLogConfig(enabled=True, offsets_s=(60,)), clock=clock)
    xb.record(st2, "decision", "r", "X", "E", "T1", "yes", 0.35, BOOK, t0)
    assert xb.run_due(clock.t + 60) == 0 and xb.stats["skipped_error"] == 1    # a failed read is never raised


def test_the_rest_client_takes_the_read_budget_without_waiting(monkeypatch):
    b = TokenBucket(100.0, 300.0)
    b.tokens, b.t = 30.0, time.monotonic()
    assert not b.try_take(10, reserve=25) and 29.0 < b.tokens < 31.0      # 10 + 25 not there: nothing taken
    assert b.try_take(10, reserve=15) and 19.0 < b.tokens < 21.5
    r = KalshiREST(base_url="https://example.invalid", read_rate=20.0)     # capacity 60
    calls = []

    def fake_request(method, path, params=None, cost=10, retries=4, timeout=None, **kw):
        calls.append((method, path, cost, retries, timeout, [t for _, t in params]))
        return {"orderbooks": [{"ticker": t, "orderbook_fp": BOOK} for _, t in params]}
    monkeypatch.setattr(r, "_request", fake_request)
    r.read_bucket.tokens, r.read_bucket.t = 35.0, time.monotonic()         # under 10 + half the capacity
    t0 = time.monotonic()
    assert r.orderbooks_if_budget(["A", "B"], reserve_frac=0.5, timeout=3.0) is None
    assert calls == [] and time.monotonic() - t0 < 0.2                     # nothing sent, no wait
    r.read_bucket.tokens, r.read_bucket.t = 60.0, time.monotonic()
    got = r.orderbooks_if_budget(["A", "B"], reserve_frac=0.5, timeout=3.0)
    assert set(got) == {"A", "B"} and calls == [("GET", "/markets/orderbooks", 0, 0, 3.0, ["A", "B"])]
    assert 49.0 < r.read_bucket.tokens < 51.5                              # one batch: 10 tokens, paid up front


def test_drain_runs_the_rereads_in_time_and_stops_at_the_deadline(tmp_path):
    clock = Clock()
    t0 = clock.t
    st = Store(str(tmp_path / "d.sqlite"))
    rd = Reader({"T1": LATER})
    xl = ExecLogger(rd, ExecLogConfig(enabled=True), clock=clock)
    xl.record(st, "decision", "B:E:T1:yes", "B", "E", "T1", "yes", 0.35, BOOK, t0)
    s = xl.drain(deadline=t0 + 400, sleep=clock.sleep)
    assert [r["offset_s"] for r in rows(st, "WHERE kind = 'reread'")] == [60, 300]
    assert s["skipped_deadline"] == 1 and s["queued"] == 0 and clock.t == t0 + 300


# --------------------------------------------------------------------------------------------------------- config
def test_the_option_is_off_by_default_and_on_only_where_the_configs_say(tmp_path):
    import glob
    import os
    import yaml
    from wxbot.app import runner
    root = os.path.join(os.path.dirname(__file__), "..", "config")
    on = {os.path.basename(p) for p in glob.glob(os.path.join(root, "*.yaml"))
          if ExecLogConfig.from_cfg((yaml.safe_load(open(p)) or {}).get("exec_log")).enabled}
    # live_stage4.yaml (ba1c9ab, the Oct 10 switch after the practice day) turns it on by design: Stage 3 plus the
    # rounding guard on the thin-city sleeve and the execution log; live_stage5_d4x.yaml (the d4x draft, Oct 9) is
    # live_stage4.yaml with only the forecaster and pool lines changed; live_stage4b.yaml (the post-Stage-3 bundle,
    # Oct 9) is live_stage4.yaml plus the bundle's options; live_stage4c.yaml and live_stage5c.yaml (Oct 11) are
    # live_stage4.yaml and live_stage5.yaml plus the stop guard's risk keys; live_stage4d.yaml (Oct 12) is
    # live_stage4c.yaml with the four newer cities off, live_stage4e.yaml (Oct 12) 4d with San Diego back on;
    # no other live config does
    assert on == {"paper_s3.yaml", "demo.yaml", "live_stage4.yaml", "live_stage4b.yaml", "live_stage5.yaml",
                  "live_stage5_d4x.yaml", "live_stage4c.yaml", "live_stage5c.yaml", "live_stage4d.yaml",
                  "live_stage4e.yaml"}   # live_stage5.yaml: option B of Oct 10
    assert ExecLogConfig.from_cfg(None) == ExecLogConfig() and not ExecLogConfig().enabled
    cfg = {"mode": "paper", "env": "prod", "db_path": str(tmp_path / "p.sqlite"), "status_path": str(tmp_path / "s.json"),
           "sleeve_a": {"enabled": True, "series": ["KXHIGHNY"]}, "sleeve_b": {"enabled": False, "shadow": False},
           "tape_archive": {"enabled": False}, "loop": {"tick_seconds": 2}}
    off = runner.Bot(dict(cfg), base_dir=str(tmp_path))
    assert off.exec_log is None and "exec_log" not in off._write_status()
    off.exec_log_pass()                                                # a no-op
    yes = runner.Bot({**cfg, "exec_log": {"enabled": True, "offsets_s": [30, 90]}}, base_dir=str(tmp_path))
    assert yes.exec_log.cfg.offsets_s == (30, 90) and yes.exec_log.mode == "paper" and yes.exec_log.rest is yes.rest
    assert yes._write_status()["exec_log"] == {"on": True, "queued": 0, "unwritten": 0}


# ----------------------------------------------------------------------------------------------- the bot's order paths
class LiveRest:
    """A live exchange that fills every immediate-or-cancel order, recording each call in order."""

    def __init__(self, seq):
        self.seq = seq

    def create_order(self, ticker, side, price_cents, count, **kw):
        self.seq.append(("create_order", ticker, side, price_cents, count,
                         {k: v for k, v in kw.items() if k != "client_order_id"}))
        return {"order_id": "ord-1", "fill_count": f"{count:.2f}", "remaining_count": "0.00",
                "average_fill_price": f"{price_cents / 100:.4f}", "average_fee_paid": "0.0200"}

    def orderbook(self, t):
        self.seq.append(("orderbook", t))
        return BOOK

    def orderbooks_if_budget(self, tickers, reserve_frac=0.5, timeout=None):
        self.seq.append(("reread", list(tickers)))
        return {t: LATER for t in tickers}


def live_bot(tmp_path, name, exec_on, seq, mode="live"):
    from wxbot.app.runner import Bot
    from wxbot.risk.limits import RiskConfig, RiskState
    now = time.time()
    bot = Bot.__new__(Bot)
    bot.store = Store(str(tmp_path / f"{name}.sqlite"))
    bot.rest, bot.mode, bot.cfg = LiveRest(seq), mode, {}
    bot.risk_cfg = RiskConfig()
    bot.risk = RiskState(bot.risk_cfg)
    bot.positions, bot.counters, bot.fair_cache = {}, defaultdict(int), {}
    bot.events, bot.hourly, bot.ev_tickers, bot.ev_tickers_miss = {}, {}, {}, {}
    bot.live, bot.paper, bot.netting = None, None, False
    bot.acct_ts, bot.account_stale_s, bot.eq_readings = {"sync": now, "balance": now, "positions": now}, 30.0, [1, 1, 1]
    bot.next_due = defaultdict(float)
    bot.ev_tickers["KXHIGHNY-26OCT10"] = ["KXHIGHNY-26OCT10-B70.5"]
    bot.exec_log = ExecLogger(bot.rest, ExecLogConfig(enabled=True), mode=mode) if exec_on else None
    return bot


CFG = {"trade": True, "trade_size": 5, "trade_max_usd_day": 15.0}


@pytest.mark.parametrize("mode", ["live", "paper"])
def test_with_the_option_off_the_order_path_is_unchanged_and_logs_nothing(tmp_path, mode):
    """The same take with the execution log off and on: the same order (side, price, count, flags), the same answer
    booked; off, nothing else is read or written; on, the order goes out before anything is logged or re-read."""
    off_seq, on_seq = [], []
    off = live_bot(tmp_path, "off", False, off_seq, mode)
    on = live_bot(tmp_path, "on", True, on_seq, mode)
    r_off = off.execute_take("E", "KXHIGHNY-26OCT10", "KXHIGHNY-26OCT10-B70.5", "yes", 0.35, 5, CFG)
    r_on = on.execute_take("E", "KXHIGHNY-26OCT10", "KXHIGHNY-26OCT10-B70.5", "yes", 0.35, 5, CFG)
    strip = lambda r: {k: v for k, v in r.items() if k != "ts"}
    assert strip(r_off) == strip(r_on) and r_off["filled"] == 5
    assert off_seq == on_seq                                               # the same calls, in the same order
    assert off_seq[0][0] == ("create_order" if mode == "live" else "orderbook")
    assert not off.store.c.execute("SELECT 1 FROM sqlite_master WHERE name = 'exec_log'").fetchone()
    assert off.store.kv_get(f"take_spent:E:{_budget_day()}") == on.store.kv_get(f"take_spent:E:{_budget_day()}")
    assert {t: (p.yes_qty, p.yes_cost) for t, p in off.positions.items()} == \
        {t: (p.yes_qty, p.yes_cost) for t, p in on.positions.items()}
    assert not on.store.c.execute("SELECT 1 FROM sqlite_master WHERE name = 'exec_log'").fetchone()  # no disk write yet
    on.exec_log_pass()                                                     # after the step: write, then re-read
    send = rows(on.store, "WHERE kind = 'send'")
    assert len(send) == 1 and send[0]["sleeve"] == "E" and send[0]["filled"] == 5.0
    if mode == "live":
        assert send[0]["order_id"] == "ord-1" and send[0]["limit_price"] == 0.35 and send[0]["avg_price"] == 0.35
        assert send[0]["yes_bid"] is None                                  # no book at hand: read at this pass
        assert on_seq[-1] == ("reread", ["KXHIGHNY-26OCT10-B70.5"]) and len(rows(on.store)) == 2
    else:
        assert send[0]["yes_ask"] == 0.35 and send[0]["depth_at"] == 7.0  # the book it was filled against
        assert len(rows(on.store)) == 1 and on_seq == off_seq


def _budget_day():
    return (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=4)).date().isoformat()


def test_x_logs_the_decision_book_after_its_orders_and_shares_the_rereads(tmp_path):
    seq = []
    bot = live_bot(tmp_path, "x", True, seq)
    bot.cfg_x, bot.x_seen, bot.ml_sha = taker.TakerConfig(enabled=True), set(), "sha"
    bot.cfg_x_raw = dict(CFG)
    quotes = {"KXHIGHNY-26OCT10-B70.5": LadderQuote(30, 35, 10.0, 7.0), "KXHIGHNY-26OCT10-B72.5": LadderQuote(55, 57),
              "KXHIGHNY-26OCT10-T73": LadderQuote(5, 7)}
    ev = EventBook("KXHIGHNY-26OCT10", "KXHIGHNY", "high", "KNYC", dt.date(2026, 10, 10), 0.0, [], [], quotes,
                   {"KXHIGHNY-26OCT10-B70.5": BOOK})
    bot.events, bot.ev_tickers[ev.event] = {ev.event: ev}, list(quotes)
    bot.book_ts = {"KXHIGHNY-26OCT10-B70.5": 123.0}
    real_write = bot.exec_log._write
    bot.exec_log._write = lambda store, rs: (seq.append(("write", sorted({r[2] for r in rs}))), real_write(store, rs))
    now = dt.datetime(2026, 10, 9, 21, 5, tzinfo=dt.timezone.utc).timestamp()
    bot.shadow_taker(ev, {"KXHIGHNY-26OCT10-B70.5": 0.6, "KXHIGHNY-26OCT10-B72.5": 0.3,
                          "KXHIGHNY-26OCT10-T73": 0.1}, -3, now)
    # two signals (B70.5 YES, B72.5 NO), two orders, and nothing logged on disk before the step is over
    assert [c[0] for c in seq] == ["create_order", "create_order"] and [c[2] for c in seq] == ["bid", "ask"]
    bot.net_down_since = time.time()
    bot.exec_log_pass()                                     # network down: the rows are written, nothing is read
    assert seq[2:] == [("write", ["decision", "send"])] and len(bot.exec_log.queue) == 7
    d = rows(bot.store, "WHERE kind = 'decision' AND side = 'yes'")[0]
    assert d["ref"] == "X:KXHIGHNY-26OCT10:KXHIGHNY-26OCT10-B70.5:yes" and d["book_ts"] == 123.0
    assert (d["yes_ask"], d["yes_ask_size"], d["depth_at"], d["depth_2c"]) == (0.35, 7.0, 7.0, 11.0)
    s = rows(bot.store, "WHERE kind = 'send' AND side = 'yes'")[0]
    assert s["ref"] == d["ref"] and s["order_id"] == "ord-1" and s["yes_ask"] == 0.35   # the loop's last book
    # one set of re-reads per ref, decision and order together: B70.5 at 60/300/900; B72.5 (no book at hand) also 0
    assert sorted((x.ref.split(":")[2], x.offset_s) for x in bot.exec_log.queue.heap) == [
        ("KXHIGHNY-26OCT10-B70.5", 60), ("KXHIGHNY-26OCT10-B70.5", 300), ("KXHIGHNY-26OCT10-B70.5", 900),
        ("KXHIGHNY-26OCT10-B72.5", 0), ("KXHIGHNY-26OCT10-B72.5", 60), ("KXHIGHNY-26OCT10-B72.5", 300),
        ("KXHIGHNY-26OCT10-B72.5", 900)]
    bot.net_down_since = None
    bot.exec_log_pass()
    assert seq[-2:] == [("reread", ["KXHIGHNY-26OCT10-B72.5"]), ("write", ["reread"])]
