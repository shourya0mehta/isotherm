"""Tests for the details of the Oct 4 review fixes (order states, equity, halts, hourly fills, settlement)."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests  # noqa: E402

from test_live_loop import FakeKalshi  # noqa: E402
from test_review_findings import close_of, risk_bot  # noqa: E402

from wxbot.execution.live import LiveOrderManager, sleeve_of  # noqa: E402
from wxbot.kalshi.rest import KalshiError, KalshiREST  # noqa: E402
from wxbot.strategy.types import DesiredOrder, Fill, Position  # noqa: E402


class Resp:
    def __init__(self, status, body):
        self.status_code, self._b = status, body
        self.content = json.dumps(body).encode()
        self.text = self.content.decode()

    def json(self):
        return self._b


def test_a_retried_create_that_comes_back_409_is_marked_ambiguous(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    r = KalshiREST(auth=None)
    answers = [requests.ReadTimeout("read timed out"),
               Resp(409, {"error": {"code": "conflict", "message": "order with this client_order_id exists"}})]

    def req(*a, **kw):
        x = answers.pop(0)
        if isinstance(x, Exception):
            raise x
        return x
    r.s.request = req
    try:
        r._request("POST", "/portfolio/events/orders", body={"x": 1}, write=True)
        assert False
    except KalshiError as e:
        assert e.status == 409 and e.ambiguous
    answers[:] = [Resp(400, {"error": {"code": "invalid_price"}})]
    try:
        r._request("POST", "/portfolio/events/orders", body={"x": 1}, write=True)
    except KalshiError as e:
        assert e.status == 400 and not e.ambiguous      # a clean refusal: no order exists


def test_an_ambiguous_refusal_keeps_the_order_uncertain_until_a_snapshot_shows_it():
    class Retried(FakeKalshi):
        def create_order(self, *a, **kw):
            super().create_order(*a, **kw)                # the first attempt reached the book
            raise KalshiError(409, '{"error":"exists"}', "POST", "/portfolio/events/orders", ambiguous=True)
    fake = Retried()
    m = LiveOrderManager(fake, ttl_s=1200)
    t, close = close_of(fake)
    d = DesiredOrder(t, "ask", 3, 10.0, "A", 0.0)
    s = m.reconcile([d], close)
    assert s["uncertain"] == 1 and [o.state for o in m.resting_list()] == ["uncertain"]
    m.sync()
    assert [(o.order_id, o.state) for o in m.resting_list()] == [("o1", "resting")]


def test_a_post_only_rejection_pauses_that_key_only():
    class Crossing(FakeKalshi):
        def create_order(self, ticker, *a, **kw):
            if ticker.endswith("T75"):
                raise KalshiError(400, '{"error":{"code":"post_only_cross","message":"post only order would cross"}}',
                                  "POST", "/portfolio/events/orders")
            return super().create_order(ticker, *a, **kw)
    fake = Crossing()
    m = LiveOrderManager(fake, ttl_s=1200, post_only_cooldown_s=10)
    t75, t70 = fake.markets[4]["ticker"], fake.markets[0]["ticker"]
    close = {t75: int(time.time()) + 3600, t70: int(time.time()) + 3600}
    ds = [DesiredOrder(t75, "ask", 3, 10.0, "A", 0.0), DesiredOrder(t70, "ask", 2, 10.0, "A", 0.0)]
    s = m.reconcile(ds, close)
    assert s["rejected"] == 1 and s["placed"] == 1 and m.errors_last_minute() == 0
    s = m.reconcile(ds, close)
    assert s["cooldown"] == 1 and s["placed"] == 0
    m.cooldown = {k: 0.0 for k in m.cooldown}
    assert m.reconcile(ds, close)["rejected"] == 1


def test_committed_orders_are_what_the_desired_set_does_not_stand_for():
    fake = FakeKalshi()
    m = LiveOrderManager(fake, ttl_s=1200)
    t, close = close_of(fake)
    m.reconcile([DesiredOrder(t, "ask", 3, 10.0, "A", 0.0, event=fake.ev)], close)
    assert m.committed([DesiredOrder(t, "ask", 3, 10.0, "A", 0.0)]) == []          # stood for by the desired order
    c = m.committed([DesiredOrder(t, "ask", 4, 10.0, "A", 0.0)])                   # being replaced: still exposure
    assert [(o.ticker, o.price_cents, o.count) for o in c] == [(t, 3, 10.0)] and c[0].event == fake.ev
    assert len(m.committed([])) == 1


def test_client_ids_carry_the_sleeve():
    fake = FakeKalshi()
    m = LiveOrderManager(fake, ttl_s=1200, ttl_by_sleeve={"H": 120, "A": 300})
    t, close = close_of(fake)
    m.reconcile([DesiredOrder(t, "ask", 3, 10.0, "H", 0.0)], close)
    cid = fake.created[0][4]["client_order_id"]
    assert cid.startswith("wxb-H-") and sleeve_of(cid) == "H" and sleeve_of("wxb-1234abcd") == ""
    exp = fake.created[0][4]["expiration_ts"]
    assert 100 <= exp - time.time() <= 121                                         # the hourly dead-man TTL


def test_cancel_all_keeps_what_it_could_not_cancel_as_exposure(monkeypatch):
    from test_live_safety import StubbornKalshi
    monkeypatch.setattr(time, "sleep", lambda s: None)
    fake = StubbornKalshi(stuck_rounds=0, never={"o1"})
    m = LiveOrderManager(fake, ttl_s=1200)
    t, close = close_of(fake)
    m.reconcile([DesiredOrder(t, "ask", 3, 10.0, "A", 0.0)], close)
    assert m.cancel_all("test") == 1
    assert [(o.order_id, o.state) for o in m.resting_list()] == [("o1", "cancel_pending")]
    assert len(m.committed([])) == 1


def live_bot(tmp_path, fake):
    bot = risk_bot(tmp_path, fake)
    bot.alerts = type("A", (), {"send": lambda *a, **kw: None})()
    bot.status_path = str(tmp_path / "data" / "live_status.json")
    return bot


def test_stops_use_settled_equity_and_ignore_one_odd_reading(tmp_path):
    class Acct(FakeKalshi):
        bal, pv, basis = 50000, 0, 100.0

        def balance(self):
            return {"balance": self.bal, "portfolio_value": self.pv}

        def positions(self, **kw):
            return [{"ticker": "KXHIGHNY-26OCT05-T75", "position_fp": "-100.00",
                     "market_exposure_dollars": f"{self.basis:.4f}"}], []
    fake = Acct()
    bot = live_bot(tmp_path, fake)
    bot.store.kv_set("balance_semantics", {"excludes_resting": True})
    for _ in range(3):
        bot.refresh_live_state()
    assert abs(bot.risk.equity - 600.0) < 1e-9 and abs(bot.risk.peak_equity - 600.0) < 1e-9
    fake.pv = -5000                         # the mark drops $50 on a thin book: settled equity does not move
    bot.refresh_live_state()
    assert abs(bot.risk.equity - 600.0) < 1e-9 and not bot.risk.check_stops("d")
    fake.bal = 52500                         # one inconsistent reading (+$25): neither the equity nor the peak move
    bot.refresh_live_state()
    fake.bal = 50000
    bot.refresh_live_state()
    assert abs(bot.risk.peak_equity - 600.0) < 1e-9
    fake.basis, fake.bal = 0.0, 57000        # it settles paying $70 for $100: -$30 lands after two readings
    bot.refresh_live_state()
    assert abs(bot.risk.equity - 600.0) < 1e-9
    bot.refresh_live_state()
    assert abs(bot.risk.equity - 570.0) < 1e-9


def test_a_daily_halt_lasts_the_day_across_restarts_and_reset_halt_clears_a_hard_one(tmp_path, monkeypatch):
    bot = live_bot(tmp_path, FakeKalshi())
    bot.halt("daily loss stop hit", kind="daily")
    b2 = live_bot(tmp_path, FakeKalshi())
    b2.restore_halt()
    assert b2.risk.halted and b2.risk.halt_kind == "daily"
    latch = b2.store.kv_get("halt_latch")
    latch["day"] = "2000-01-01"                                                  # a new ET day: it lapses
    b2.store.kv_set("halt_latch", latch)
    b3 = live_bot(tmp_path, FakeKalshi())
    b3.restore_halt()
    assert not b3.risk.halted
    b3.halt("drawdown stop hit", kind="hard")
    from wxbot.__main__ import main
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"db_path: {tmp_path / 'live.sqlite'}\nstatus_path: {tmp_path / 'data' / 's.json'}\n")
    main(["reset-halt", "--config", str(cfg)])
    b4 = live_bot(tmp_path, FakeKalshi())
    b4.restore_halt()
    assert not b4.risk.halted


def test_the_quiesce_flag_halts_without_stopping(tmp_path):
    bot = live_bot(tmp_path, FakeKalshi())
    os.makedirs(os.path.dirname(bot.status_path), exist_ok=True)
    open(os.path.join(os.path.dirname(bot.status_path), "QUIESCE"), "w").close()
    assert bot.quiesce_requested() and not bot.quiesce_requested()


def test_hourly_size_limit_counts_gross_fills(tmp_path):
    from test_sleeve_h import ladder
    bot = live_bot(tmp_path, FakeKalshi())
    ev = ladder()
    t = ev.markets[0]["ticker"]
    for side, px in (("bid", 95), ("ask", 95)):                                  # YES then NO on one strike: net 0
        bot.store.add_fill("live", Fill(time.time(), t, ev.event, side, px, 10.0, "H", order_ref=f"f-{side}"))
    g = bot.h_filled_gross(ev.event)
    assert g == {(t, "bid"): 10.0, (t, "ask"): 10.0}
    from wxbot.strategy import sleeve_h
    cfg = sleeve_h.SleeveHConfig(enabled=True, size=10, cap_usd=25.0)
    out = sleeve_h.propose(ev, time.time(), cfg, [], filled=g)
    assert not [o for o in out if o.ticker == t]                                 # its 10 a side are used up


def test_bot_pnl_comes_from_its_own_fills_not_the_account_record(tmp_path):
    bot = live_bot(tmp_path, FakeKalshi())
    t = "KXHIGHNY-26OCT01-T80"
    bot.store.add_fill("live", Fill(time.time(), t, "KXHIGHNY-26OCT01", "ask", 4, 25.0, "A", order_ref="f1"))
    rec = {"ticker": t, "event_ticker": "KXHIGHNY-26OCT01", "market_result": "no", "revenue": 3500,
           "yes_total_cost_dollars": "0", "no_total_cost_dollars": "33.6000", "fee_cost": "0",   # +10 manual NO
           "settled_time": "2026-10-02T10:00:00Z", "value": 0}
    bot.rest.settlements = lambda min_ts=None: [rec]
    bot.settle_live()
    assert abs(bot.risk.realized_total - 1.0) < 1e-9                             # 25 x (1 - 0.96), not the account's


def test_a_sleeve_that_loses_its_daily_stop_sits_out_the_rest_of_the_day(tmp_path):
    bot = live_bot(tmp_path, FakeKalshi())
    bot.cfg = {"risk": {"sleeve_daily_stop_usd": {"H": 15, "A": 15}}}
    t = "KXTEMPMIAH-26OCT05H14-T88"
    bot.store.add_fill("live", Fill(time.time(), t, "KXTEMPMIAH-26OCT05H14", "ask", 4, 17.0, "H", order_ref="f1"))
    bot.store.set_settlement(t, "KXTEMPMIAH-26OCT05H14", "yes", time.time(), None)   # the NOs lost: -$16.32
    assert bot.sleeve_pnl_today()["H"] < -16 and bot.stopped_sleeves() == {"H"}
    assert bot.store.kv_get(f"sleeve_stop:H:{__import__('wxbot.timeutil', fromlist=['et_day']).et_day().isoformat()}")
    assert bot.model_tag() == "off"


def test_a_trade_already_reflected_in_a_smaller_queue_does_not_fill_us_again():
    import datetime as dt
    from wxbot.execution.paper import PaperBroker
    pb = PaperBroker(late_fills=True)
    t0 = time.time() - 100
    book = {"yes_dollars": [["0.9700", "300.00"]], "no_dollars": []}
    o = pb.place(DesiredOrder("KXHIGHNY-26OCT05-B70.5", "bid", 97, 25.0, "A", 0.99), book, now=t0)
    assert o.queue_ahead == 300
    iso = lambda ts: dt.datetime.fromtimestamp(ts, dt.timezone.utc).isoformat().replace("+00:00", "Z")
    trade = {"ticker": o.ticker, "created_time": iso(t0 + 10), "yes_price_dollars": "0.9700", "count_fp": "300.00",
             "taker_side": "no"}
    pb.on_book(o.ticker, {"yes_dollars": [], "no_dollars": []}, ts=t0 + 20)   # the book read after that trade
    assert o.queue_ahead == 0
    assert pb.on_trade(trade) == []                                           # the trade arrives late: no fill
    later = dict(trade, created_time=iso(t0 + 30), count_fp="10.00")
    assert [f.count for f in pb.on_trade(later)] == [10.0]                    # a new trade at our price fills us


def test_one_snapshot_that_misses_an_order_does_not_drop_it():
    class Glitchy(FakeKalshi):
        hide = False

        def orders(self, status="resting", **kw):
            return [] if self.hide else super().orders(status, **kw)
    fake = Glitchy()
    m = LiveOrderManager(fake, ttl_s=1200)
    t, close = close_of(fake)
    d = DesiredOrder(t, "ask", 3, 10.0, "A", 0.0)
    m.reconcile([d], close)
    for o in m.resting_list():
        o.ack_ts -= 10
    fake.hide = True
    m.sync()                                     # a read that (wrongly) shows nothing
    assert len(m.resting_list()) == 1
    m.reconcile([d], close)
    assert len(fake.created) == 1                # no duplicate on one odd read
    fake.hide = False
    m.sync()
    assert m.resting_list()[0].misses == 0       # seen again: the count resets
