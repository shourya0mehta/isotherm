"""Round 7 (Oct 4): a pre-mortem of the first live week and a field-by-field check of the bot's Kalshi API use
against Kalshi's docs. Each item below was checked in the code first; the tests fail on the code before the fix.
C1  an amend without updated_client_order_id (every expiry refresh) may, for all Kalshi's docs say, give the order
    another client id; the bot knew its orders in snapshots only by the wxb- prefix, so it would have dropped the
    order (still resting) and placed a second one. Now it also knows them by order id, follows a changed client id,
    and the go-live order test checks what Kalshi does;
C2  Thursday 3-5 AM ET maintenance: Kalshi refuses new orders (and cancels the bot's, sent with cancel_order_on_pause)
    while the bot, checking the exchange status once a minute, kept re-placing them, each refusal an API error, so
    "HALT: API error burst" fired most Thursday nights (and failed the Stage 2 gate). Now the bot stops a minute before
    Kalshi's published closing time, reads the status out of a 5xx answer (Kalshi sends it there during a pause),
    stops placing after three unexplained refusals in a pass and re-reads the status at once, and cools a refused
    key down for 30 s;
C4  a batch-cancel answer counted as confirmed whenever it had no error object (reduced_by is always present):
    now it needs a cancel time or contracts actually taken off;
N1  a network outage of a few minutes ended steps in errors that counted toward the step-failure halt (10 in 10 min
    = 30-minute pause, a second burst within 2 hours = hard halt until reset-halt), and alerts sent while offline
    were lost: network and 5xx failures now never count (an alert after 15 minutes down), failed alerts are resent
    when the network is back, and a hard halt is re-announced every 4 hours in the daytime;
C3  the shadow sleeves (hourly H and H2, Sleeve B's reward estimate) ran inside the live decision step, so one bad
    record from Kalshi (an incentive program whose target size is null, which the schema allows) ended every live
    step before the order update: they now run as side tasks, and the nullable fields get sleeve_b's defaults;
M1  a forecast the forecaster failed to update (its inputs did not load) kept being pooled with the current market
    using its own, older decision hour's weights: now the market-anchored fair value stands when the forecast missed
    the latest decision hour by more than an hour;
C5  position cost: Kalshi's docs give no sign for market_exposure_dollars; a NO position reported with a negative
    cost would have lowered settled equity by twice its cost (tripping the daily stop) and read as negative risk."""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_live_loop import FakeKalshi  # noqa: E402
from test_review_round2 import Acct, make_bot  # noqa: E402
from test_smoke_alerts import smoke_env  # noqa: E402

import wxbot.app.smoke as smoke  # noqa: E402
from wxbot.execution.live import CANCEL_PENDING, RESTING, LiveOrderManager  # noqa: E402
from wxbot.strategy.types import DesiredOrder  # noqa: E402


@pytest.fixture
def clock(monkeypatch):
    """A hand-moved clock, kept away from ET midnight (the daily stop resets there by design)."""
    from wxbot.timeutil import to_et
    c = [time.time()]
    if to_et(c[0]).hour >= 22 or to_et(c[0]).hour < 2:
        c[0] += 4 * 3600
    monkeypatch.setattr(time, "time", lambda: c[0])
    return c


# ------------------------------------------------------------------------------------------------ C5
class SignedCost(Acct):
    """Reports a NO position's cost as a negative number (what the docs do not rule out)."""

    def positions(self, **kw):
        return [{"ticker": t, "position_fp": f"{q:.2f}",
                 "market_exposure_dollars": f"{(-c if q < 0 else c):.4f}"} for t, (q, c) in self.pos.items()], []


def test_a_no_position_reported_with_a_negative_cost_counts_its_cost(tmp_path, monkeypatch, clock):
    fake = SignedCost()
    bot = make_bot(tmp_path, monkeypatch, fake)
    bot.store.kv_set("balance_semantics", {"excludes_resting": True})
    t = fake.markets[4]["ticker"]
    fake.cash -= 9.70
    fake.pos[t] = (-10.0, 9.70)                      # 10 NO bought at 97c
    for _ in range(3):
        clock[0] += 5
        bot.refresh_live_state()
    p = bot.positions[t]
    assert p.no_qty == 10.0 and p.no_cost == pytest.approx(9.70)
    assert bot.live_equity == pytest.approx(625.0)   # cash 615.30 + cost 9.70, not 615.30 - 9.70
    assert bot.risk.realized_today == pytest.approx(0.0)


# ------------------------------------------------------------------------------------------------ C1
class CidChanger(FakeKalshi):
    """An exchange whose amend gives the order a new client id when none is named."""

    def amend_order(self, oid, ticker, side, price, total, expiration_ts=None, **kw):
        a = super().amend_order(oid, ticker, side, price, total, expiration_ts=expiration_ts, **kw)
        if oid in self.resting:
            self.resting[oid]["client_order_id"] = f"kx-{oid}-{len(self.amended)}"
        return dict(a, client_order_id=f"kx-{oid}-{len(self.amended)}")


def test_an_order_whose_client_id_changes_on_amend_stays_tracked(clock):
    fake = CidChanger()
    m = LiveOrderManager(fake, ttl_s=300)
    close = {x["ticker"]: int(clock[0]) + 86400 for x in fake.markets}
    want = [DesiredOrder(fake.markets[4]["ticker"], "ask", 3, 10.0, "A", 0.0)]
    m.reconcile(want, close)
    assert len(fake.created) == 1
    for _ in range(40):                       # 200 s: the expiry refresh amends it (and its client id changes)
        clock[0] += 5
        m.sync()
        m.reconcile(want, close)
    assert fake.amended and len(fake.created) == 1                        # never a second order beside it
    [o] = m.resting_list()
    assert o.state == RESTING and o.client_order_id.startswith("kx-")    # followed the new client id


def test_the_order_test_fails_if_an_amend_changes_the_client_id(monkeypatch, tmp_path, capsys):
    fake = smoke_env(monkeypatch)
    real_amend = fake.amend_order

    def amend(oid, ticker, side, price_cents, total, **kw):
        a = real_amend(oid, ticker, side, price_cents, total, **kw)
        fake.resting[oid]["client_order_id"] = "kalshi-generated"
        return a
    fake.amend_order = amend
    assert smoke.run_smoke({"env": "prod", "db_path": str(tmp_path / "live.sqlite")}) is False
    assert "[BAD] amended order keeps its client id" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------ C4
def test_a_cancel_answer_without_a_time_or_a_reduction_is_not_a_confirmation(clock):
    class Silent(FakeKalshi):
        def batch_cancel(self, ids, **kw):
            return {"orders": [{"order_id": i, "reduced_by": "0.00"} for i in ids]}   # nothing cancelled, no error
    fake = Silent()
    m = LiveOrderManager(fake, ttl_s=300)
    close = {x["ticker"]: int(clock[0]) + 86400 for x in fake.markets}
    m.reconcile([DesiredOrder(fake.markets[4]["ticker"], "ask", 3, 10.0, "A", 0.0)], close)
    clock[0] += 2
    st = m.reconcile([], close)
    assert st["cancelled"] == 0 and [o.state for o in m.resting_list()] == [CANCEL_PENDING]


def test_the_order_test_does_not_take_an_empty_cancel_answer_as_confirmed(monkeypatch, tmp_path, capsys):
    fake = smoke_env(monkeypatch)
    real_batch = fake.batch_cancel

    def batch(ids):
        if len([x for x in fake.log if x[0] == "batch_cancel"]) == 0:
            fake.log.append(("batch_cancel", tuple(ids)))
            return {"orders": [{"order_id": i, "reduced_by": "0.00"} for i in ids]}
        return real_batch(ids)
    fake.batch_cancel = batch
    assert smoke.run_smoke({"env": "prod", "db_path": str(tmp_path / "live.sqlite")}) is False
    assert "[BAD] batch cancel confirmed per order" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------ C3
def test_a_program_with_null_fields_is_priced_with_the_defaults(tmp_path, monkeypatch, clock):
    import types
    from test_live_loop import iso
    fake = Acct()
    bot = make_bot(tmp_path, monkeypatch, fake)
    t = "KXTEMPMIAH-26OCT05H14-T88"
    bot.lip_programs = {t: {"target_size_fp": None, "discount_factor_bps": None, "period_reward": 50000,
                            "start_date": iso(clock[0] - 3600), "end_date": iso(clock[0] + 3600)}}
    ev = types.SimpleNamespace(books={t: {"yes_dollars": [["0.0100", "100.00"]], "no_dollars": [["0.9800", "50.00"]]}})
    bot.shadow_lip(ev, [DesiredOrder(t, "ask", 99, 100.0, "B", 0.0)], clock[0])     # used to raise TypeError
    bot.shadow_lip(ev, [DesiredOrder("KXTEMPMIAH-26OCT05H14-T90", "ask", 99, 100.0, "B", 0.0)], clock[0])  # no program


def test_a_failing_shadow_never_stops_the_live_decision(tmp_path, monkeypatch, clock):
    fake = Acct()
    bot = make_bot(tmp_path, monkeypatch, fake)
    bot.h_on = bot.h2_on = True
    bot.hourly = {"KXTEMPMIAH-26OCT05H14": object()}
    bot.index_point = {"t": clock[0] * 1000, "v": 85.0}
    bot.b_shadow = True

    def boom(*a, **kw):
        raise TypeError("float() argument must be a string or a real number, not 'NoneType'")
    monkeypatch.setattr(bot, "shadow_h_orders", boom)
    monkeypatch.setattr(bot, "sleeve_b_pass", boom)
    monkeypatch.setattr(bot, "h_real_orders", lambda now: [])
    bot.decide()                                                    # no exception: the step goes on to execute
    errs = {k: v for k, v in bot.counters.items() if k.startswith("side_err:")}
    assert errs == {"side_err:h_shadow": 1, "side_err:h2_shadow": 1, "side_err:b_shadow": 1}


# ------------------------------------------------------------------------------------------------ C2
KALSHI_HOURS = {"maintenance_windows": [], "standard_hours": [{
    "start_time": "2024-12-01T00:00:00Z", "end_time": "2200-12-01T00:00:00Z",
    **{d: [{"open_time": "00:00", "close_time": "00:00"}]
       for d in ("monday", "tuesday", "wednesday", "friday", "saturday", "sunday")},
    "thursday": [{"open_time": "00:00", "close_time": "03:00"}, {"open_time": "05:00", "close_time": "00:00"}]}]}


def et(*a):
    import datetime as dt
    from zoneinfo import ZoneInfo
    return dt.datetime(*a, tzinfo=ZoneInfo("America/New_York")).timestamp()


@pytest.mark.parametrize("when, closed", [
    ((2026, 10, 8, 2, 58, 30), False), ((2026, 10, 8, 2, 59, 30), True),     # a minute's lead before 3 AM
    ((2026, 10, 8, 4, 30), True), ((2026, 10, 8, 5, 0), False),
    ((2026, 10, 7, 23, 59, 30), False), ((2026, 10, 8, 0, 0, 30), False),    # no lead at midnight (open both sides)
    ((2026, 11, 5, 3, 30), True), ((2026, 11, 5, 5, 1), False),              # after the DST change, still ET
    ((2026, 10, 5, 9, 0), False)])
def test_kalshi_published_hours(when, closed):
    from wxbot import exchange_hours
    assert exchange_hours.closed_now(KALSHI_HOURS, et(*when)) is closed


def test_maintenance_windows_and_unreadable_schedules():
    from wxbot import exchange_hours
    w = dict(KALSHI_HOURS, maintenance_windows=[{"start_datetime": "2026-10-06T14:00:00Z",
                                                 "end_datetime": "2026-10-06T15:00:00Z"}])
    assert exchange_hours.closed_now(w, et(2026, 10, 6, 9, 59, 30))         # 13:59:30 UTC: within the lead
    assert not exchange_hours.closed_now(w, et(2026, 10, 6, 11, 0))
    for bad in (None, {}, {"standard_hours": "x"}, {"standard_hours": [{"monday": [{"open_time": "x"}]}]}):
        assert exchange_hours.closed_now(bad, et(2026, 10, 8, 4, 0)) is False  # unreadable: open (status still applies)


def test_a_paused_exchange_status_comes_in_a_503(monkeypatch):
    import json
    from wxbot.kalshi.rest import KalshiError, KalshiREST
    r = KalshiREST(base_url="https://external-api.kalshi.com/trade-api/v2", auth=None)

    class Resp:
        def __init__(self, status, body):
            self.status_code, self.text = status, json.dumps(body)
            self.content = self.text.encode()

        def json(self):
            return json.loads(self.text)
    answers = {"body": {"exchange_active": False, "trading_active": False}, "status": 503}
    monkeypatch.setattr(r.s, "request", lambda *a, **kw: Resp(answers["status"], answers["body"]))
    monkeypatch.setattr(time, "sleep", lambda s: None)
    assert r.exchange_status()["trading_active"] is False
    answers.update(body={"error": "boom"})
    with pytest.raises(KalshiError):
        r.exchange_status()


def test_a_trading_pause_costs_three_refusals_not_an_error_burst(clock):
    from wxbot.kalshi.rest import KalshiError

    class Paused(FakeKalshi):
        def create_order(self, ticker, side, price_cents, count, **kw):
            self.created.append((ticker, side, price_cents, count, kw))
            raise KalshiError(400, '{"error":{"code":"trading_is_paused","message":"trading is paused"}}', "POST",
                              "/portfolio/events/orders")
    fake = Paused()
    m = LiveOrderManager(fake, ttl_s=300)
    close = {x["ticker"]: int(clock[0]) + 86400 for x in fake.markets}
    want = [DesiredOrder(x["ticker"], "ask", 3, 5.0, "A", 0.0) for x in fake.markets]
    st = m.reconcile(want, close)
    assert st["refused"] == 3 and st["refusal_stop"] == 1 and len(fake.created) == 3
    assert m.errors_last_minute() == 3 and not m.resting_list()
    clock[0] += 2
    st = m.reconcile(want, close)                        # the refused keys cool down; the other two are tried
    assert st["cooldown"] == 3 and len(fake.created) == 5


def test_the_bot_rereads_the_status_after_a_refusal_stop_and_skips_closed_hours(tmp_path, monkeypatch, clock):
    from wxbot.kalshi.rest import KalshiError

    class Paused(Acct):
        paused = True

        def create_order(self, ticker, side, price_cents, count, **kw):
            if self.paused:
                self.created.append((ticker, side, price_cents, count, kw))
                raise KalshiError(400, '{"error":{"code":"trading_is_paused"}}', "POST", "/portfolio/events/orders")
            return super().create_order(ticker, side, price_cents, count, **kw)

        def exchange_status(self):
            return {"trading_active": not self.paused}
    fake = Paused()
    bot = make_bot(tmp_path, monkeypatch, fake)
    bot.store.kv_set("balance_semantics", {"excludes_resting": True})
    for _ in range(3):
        clock[0] += 5
        bot.refresh_live_state()
    want = [DesiredOrder(x["ticker"], "ask", 97, 3.0, "A", 0.5, event=x["ticker"].rsplit("-", 1)[0])
            for x in fake.markets]
    bot.trading_active = True
    bot.execute(want)
    assert len(fake.created) <= 3 and bot.trading_active is False       # paused until a status read in 60 s
    assert bot.exchange_paused(clock[0]) == "Kalshi refused order after order"
    assert bot.next_due["exch"] == pytest.approx(clock[0] + 60)
    assert not bot.risk.halted
    # published hours: closed now -> the step skips execution before placing anything, even with the status open
    fake.paused = False
    closed = dict(KALSHI_HOURS, maintenance_windows=[{"start_datetime": "2000-01-01T00:00:00Z",
                                                      "end_datetime": "2200-01-01T00:00:00Z"}])
    monkeypatch.setattr(fake, "exchange_schedule", lambda: closed, raising=False)
    bot.next_due.clear()
    n = len(fake.created)
    for _ in range(3):
        clock[0] += 5
        bot.step()
    assert len(fake.created) == n and bot.schedule == closed
    assert any("published hours" in m for _, _, m in bot.store.c.execute("SELECT ts, level, msg FROM events_log"))
    monkeypatch.setattr(fake, "exchange_schedule", lambda: {}, raising=False)
    bot.schedule = None
    for _ in range(3):
        clock[0] += 5
        bot.step()
    assert len(fake.created) > n                                       # open again: it quotes


# ------------------------------------------------------------------------------------------------ N1
def test_network_failures_are_not_the_bots_errors():
    import requests
    from wxbot.app.runner import transient_failure
    from wxbot.kalshi.rest import KalshiError
    assert transient_failure(requests.ConnectionError("Max retries exceeded"))
    assert transient_failure(requests.Timeout("read timed out"))
    assert transient_failure(KalshiError(503, "unavailable", "GET", "/markets/orderbooks"))
    assert transient_failure(KalshiError(429, "too many requests", "GET", "/markets"))
    assert not transient_failure(KalshiError(400, "bad request", "POST", "/portfolio/events/orders"))
    assert not transient_failure(KeyError("yes_dollars"))
    assert not transient_failure(TypeError("NoneType"))


def test_a_long_wifi_outage_never_halts_and_is_announced_once(tmp_path, monkeypatch, clock):
    import requests
    from wxbot.app import runner
    fake = Acct()
    bot = make_bot(tmp_path, monkeypatch, fake)
    sent = []
    monkeypatch.setattr(bot.alerts, "topic", "t")
    monkeypatch.setattr(bot.alerts, "_post", lambda title, msg, pr, tags: sent.append((title, msg)) or True)
    down = {"on": True, "steps": 0}

    def step():
        down["steps"] += 1
        if down["on"]:
            raise requests.ConnectionError("Failed to establish a new connection: [Errno 8] nodename nor servname")
    monkeypatch.setattr(bot, "step", step)
    bot.running = True

    def fake_sleep(s):
        clock[0] += 10
        if down["steps"] >= 200:                 # 33 minutes of outage, then back
            down["on"] = False
        if down["steps"] >= 203:
            bot.running = False
    monkeypatch.setattr(runner.time, "sleep", fake_sleep)
    bot.run()
    assert not bot.risk.halted and bot.counters["step_errors"] == 0 and bot.counters["net_errors"] == 200
    titles = [t for t, _ in sent]
    assert sum("no connection" in t for t in titles) == 1                 # one alert after 15 minutes down
    logs = [m for _, _, m in bot.store.c.execute("SELECT ts, level, msg FROM events_log")]
    assert any("network back after" in m for m in logs) and not any("HALT" in m for m in logs)


def test_an_alert_the_network_swallowed_is_sent_when_it_is_back(monkeypatch):
    from wxbot.app.alerts import Alerter
    a = Alerter({"ntfy_topic": "t", "min_gap_s": 0})
    up = {"on": False}
    posted = []

    def post(url, data=None, headers=None, timeout=None):
        if not up["on"]:
            raise OSError("network is unreachable")
        posted.append((headers["Title"], data.decode()))

        class R:
            status_code = 200
        return R()
    import wxbot.app.alerts as al
    monkeypatch.setattr(al.requests, "post", post)
    assert a.send("halt:hard", "wxbot live HALT (hard)", "drawdown stop", priority="high") is False
    assert len(a.pending) == 1 and a.flush() == 0                          # still down: kept
    up["on"] = True
    assert a.flush() == 1 and not a.pending
    assert posted[0][0] == "wxbot live HALT (hard)" and "sent late" in posted[0][1]


def test_a_hard_halt_is_announced_again_in_the_daytime(tmp_path, monkeypatch, clock):
    fake = Acct()
    bot = make_bot(tmp_path, monkeypatch, fake)
    sent = []
    monkeypatch.setattr(bot.alerts, "topic", "t")
    monkeypatch.setattr(bot.alerts, "_post", lambda title, msg, pr, tags: sent.append(title) or True)
    from wxbot.timeutil import to_et
    while to_et(clock[0]).hour != 9:          # 9 AM ET
        clock[0] += 600
    bot.halt("drawdown stop: equity 40.00 below the peak", kind="hard")
    assert sent == ["wxbot live HALT (hard)"]
    bot.remind_halt()
    assert len(sent) == 1                      # not right after the halt's own alert
    clock[0] += 4 * 3600 + 1                   # 1 PM ET
    bot.remind_halt()
    assert sent[-1] == "wxbot live is still halted"
    clock[0] += 10 * 3600                      # 11 PM ET: quiet hours
    n = len(sent)
    bot.remind_halt()
    assert len(sent) == n


# ------------------------------------------------------------------------------------------------ M1
def test_a_forecast_that_missed_its_decision_hour_is_not_used(tmp_path):
    import datetime as dt
    import json
    import numpy as np
    from collections import defaultdict
    from wxbot.app.runner import Bot
    from wxbot.ml import features as F
    from wxbot.ml.live import Pool
    from wxbot.models.fair_value import FairValueResult
    from wxbot.store.db import Store
    from wxbot.strategy.types import EventBook

    class FakeML:
        h = 9

        def get(self, icao, D):
            P = np.zeros(F.NCLS); P[F.K + 2] = 1.0           # sure of 82
            return {"h": self.h, "P": P, "anchor_r": 80.0}
    pw = tmp_path / "pw.json"
    pw.write_text(json.dumps({"hours": F.DEC_H, "alpha": [0.9] * 7, "beta": [0.4] * 7, "theta": [0.0] * 7,
                              "kappa": [0.0] * 7, "z_scale_f": 2.0}))
    bot = Bot.__new__(Bot)
    bot.ml, bot.pool, bot.ml_sha, bot.ml_logged = FakeML(), Pool(str(pw)), "t", set()
    bot.store, bot.counters, bot.next_due = Store(str(tmp_path / "t.db")), defaultdict(int), defaultdict(float)
    from test_ml import ladder
    br = ladder()
    ev = EventBook("KXHIGHNY-26OCT02", "KXHIGHNY", "high", "KNYC", dt.date(2026, 10, 2), 0.0, [], br)
    mkt = {"T-LO": 0.1, "B78": 0.2, "B80": 0.4, "B82": 0.2, "T-HI": 0.1}
    at = lambda h, m=0: dt.datetime(2026, 10, 2, h, m, tzinfo=dt.timezone.utc).timestamp()
    fv = lambda: FairValueResult(dict(mkt), dict(mkt), {}, set(), 0.0, [], False)
    assert bot.apply_ml(ev, fv(), at(12, 30)).probs["B82"] > 0.2      # h = 9 at 12:30: the 12Z run is due by 13Z
    assert bot.apply_ml(ev, fv(), at(13, 30)).probs == mkt             # an hour late: the market stands
    assert bot.counters["ml_stale"] == 1
    bot.ml.h = 12
    assert bot.apply_ml(ev, fv(), at(13, 31)).probs["B82"] > 0.2      # caught up


# ------------------------------------------------------------------------------------------------ R15
def test_a_request_costing_more_than_the_bucket_holds_does_not_wait_forever():
    import threading
    from wxbot.kalshi.rest import TokenBucket
    b = TokenBucket(80.0, 80.0)                   # the write bucket: one second of the Basic tier, at 80%
    done = threading.Event()
    threading.Thread(target=lambda: (b.take(2 * 41), done.set()), daemon=True).start()   # a 41-order batch cancel
    assert done.wait(3) and b.tokens < 0          # went at once, in debt: the next request waits to repay it


# ------------------------------------------------------------------------------------------------ Q1
def test_two_hours_without_a_single_order_raises_one_alert(tmp_path, monkeypatch, clock):
    fake = Acct()
    bot = make_bot(tmp_path, monkeypatch, fake)
    sent = []
    monkeypatch.setattr(bot.alerts, "topic", "t")
    monkeypatch.setattr(bot.alerts, "_post", lambda title, msg, pr, tags: sent.append(title) or True)
    bot.watch_quoting(3, clock[0])
    for _ in range(14):                        # 2 h 20 min of nothing wanted
        clock[0] += 600
        bot.watch_quoting(0, clock[0])
    assert sent == ["wxbot live: no orders for 2 h"]
    bot.watch_quoting(2, clock[0])             # quoting again: the clock starts over
    clock[0] += 600
    bot.watch_quoting(0, clock[0])
    assert len(sent) == 1



# ------------------------------------------------------------------------------------------------ round 8
def test_odd_published_hours_are_set_aside_loudly(tmp_path, monkeypatch, clock):
    fake = Acct()
    bot = make_bot(tmp_path, monkeypatch, fake)
    sent = []
    monkeypatch.setattr(bot.alerts, "topic", "t")
    monkeypatch.setattr(bot.alerts, "_post", lambda title, msg, pr, tags: sent.append(title) or True)
    wrap = {"maintenance_windows": [], "standard_hours": [{"start_time": "2024-01-01T00:00:00Z",
            "end_time": "2200-01-01T00:00:00Z",
            **{d: [{"open_time": "08:00", "close_time": "03:00"}] for d in
               ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")}}]}
    from wxbot import exchange_hours
    assert exchange_hours.closed_hours_per_week(wrap, clock[0]) == 35.0      # read right: 3 AM - 8 AM closed
    assert not exchange_hours.closed_now(wrap, et(2026, 10, 6, 1, 0))        # 1 AM: Monday's session runs on
    monkeypatch.setattr(fake, "exchange_schedule", lambda: wrap, raising=False)
    bot.refresh_schedule()
    assert bot.schedule is None and sent == ["wxbot live: unusual exchange hours"]
    monkeypatch.setattr(fake, "exchange_schedule", lambda: KALSHI_HOURS, raising=False)
    bot.refresh_schedule()
    assert bot.schedule == KALSHI_HOURS


def test_published_hours_give_way_to_a_status_that_keeps_saying_open(tmp_path, monkeypatch, clock):
    fake = Acct()
    bot = make_bot(tmp_path, monkeypatch, fake)
    bot.schedule = dict(KALSHI_HOURS, maintenance_windows=[{"start_datetime": "2000-01-01T00:00:00Z",
                                                            "end_datetime": "2200-01-01T00:00:00Z"}])
    bot.trading_active, bot.trading_active_ts = True, clock[0]
    assert bot.exchange_paused(clock[0]) == "Kalshi's published hours"
    clock[0] += 601
    bot.trading_active_ts = clock[0]
    assert bot.exchange_paused(clock[0]) is None                              # ten minutes of "open": the status wins
    bot.trading_active = False
    assert bot.exchange_paused(clock[0]) == "Kalshi's status: trading paused"


def test_a_pause_the_status_can_no_longer_confirm_is_retried_and_a_long_one_alerts(tmp_path, monkeypatch, clock):
    fake = Acct()
    bot = make_bot(tmp_path, monkeypatch, fake)
    sent = []
    monkeypatch.setattr(bot.alerts, "topic", "t")
    monkeypatch.setattr(bot.alerts, "_post", lambda title, msg, pr, tags: sent.append(title) or True)
    state = {"ok": True}

    def status():
        if not state["ok"]:
            raise ConnectionError("status endpoint down")
        return {"trading_active": False}
    monkeypatch.setattr(fake, "exchange_status", status)
    bot.next_due.clear()
    for _ in range(3 * 60 + 2):                 # three hours of steps, a minute apart, paused by the status
        clock[0] += 60
        bot.step()
    assert any("not trading for 3 h" in t for t in sent)
    state["ok"] = False                         # the status read now fails: after 5 minutes the bot tries again
    for _ in range(6):
        clock[0] += 60
        bot.step()
    assert bot.trading_active is True and getattr(bot, "paused_since", None) is None


def test_a_sleeve_stopped_for_the_day_is_not_a_silent_failure(tmp_path, monkeypatch, clock):
    fake = Acct()
    bot = make_bot(tmp_path, monkeypatch, fake)
    sent = []
    monkeypatch.setattr(bot.alerts, "topic", "t")
    monkeypatch.setattr(bot.alerts, "_post", lambda title, msg, pr, tags: sent.append(title) or True)
    bot.watch_quoting(3, clock[0])
    bot.last_stopped = {"A"}
    for _ in range(14):
        clock[0] += 600
        bot.watch_quoting(0, clock[0])
    assert sent == []


def test_the_client_id_check_reads_the_amended_copy_and_accepts_an_empty_answer(monkeypatch, tmp_path, capsys):
    # an empty client id in the amend answer: none named, the order keeps its own
    fake = smoke_env(monkeypatch)
    real_amend = fake.amend_order
    fake.amend_order = lambda *a, **kw: dict(real_amend(*a, **kw), client_order_id="")
    assert smoke.run_smoke({"env": "prod", "db_path": str(tmp_path / "a.sqlite")}) is True
    assert "[OK ] amended order keeps its client id" in capsys.readouterr().out
    # a lagging read shows the old copy (old size, old id) first, then the amended copy with a new id: BAD
    fake = smoke_env(monkeypatch)
    real_amend, real_orders = fake.amend_order, fake.orders
    st = {"amended": None, "reads": 0}

    def amend(oid, ticker, side, price_cents, total, **kw):
        st["amended"] = dict(fake.resting[oid])
        a = real_amend(oid, ticker, side, price_cents, total, **kw)
        fake.resting[oid]["client_order_id"] = "kalshi-generated"
        return a

    def orders(status="resting", **p):
        out = real_orders(status, **p)
        if st["amended"] is not None and st["reads"] < 2:
            st["reads"] += 1
            return [st["amended"] if o["order_id"] == st["amended"]["order_id"] else o for o in out]
        return out
    fake.amend_order, fake.orders = amend, orders
    assert smoke.run_smoke({"env": "prod", "db_path": str(tmp_path / "b.sqlite")}) is False
    assert "[BAD] amended order keeps its client id: Kalshi changed it to 'kalshi-generated'" in capsys.readouterr().out


def test_refused_expiry_refreshes_stop_the_pass_too(clock):
    from wxbot.kalshi.rest import KalshiError
    fake = FakeKalshi()
    m = LiveOrderManager(fake, ttl_s=300)
    close = {x["ticker"]: int(clock[0]) + 86400 for x in fake.markets}
    want = [DesiredOrder(x["ticker"], "ask", fake.quotes[x["ticker"].rsplit("-", 1)[1]][1], 5.0, "A", 0.0)
            for x in fake.markets]                 # each at the YES ask: rests without crossing
    m.reconcile(want, close)
    n = len(m.resting_list())

    def refuse(*a, **kw):
        raise KalshiError(400, '{"error":{"code":"trading_is_paused"}}', "POST", "/portfolio/events/orders/x/amend")
    fake.amend_order = refuse
    clock[0] += 200                              # every order is due for its expiry refresh
    st = m.reconcile(want, close)
    assert n >= 4 and st["refusal_stop"] == 1 and m.errors_last_minute() == 3



# ------------------------------------------------------------------------------------------------ round 9
def test_a_pause_with_a_failing_status_costs_three_refusals_per_minute_not_a_halt(tmp_path, monkeypatch, clock):
    """Kalshi refuses every order (a pause) while its status endpoint fails and no schedule is in force: the bot
    used to go back to trading 5 minutes after the last good read and hit the error-burst halt every 15 minutes."""
    from wxbot.kalshi.rest import KalshiError

    class Paused(Acct):
        def create_order(self, ticker, side, price_cents, count, **kw):
            self.created.append((ticker, side, price_cents, count, kw))
            raise KalshiError(400, '{"error":{"code":"trading_is_paused"}}', "POST", "/portfolio/events/orders")

        def exchange_status(self):
            raise ConnectionError("status endpoint down")
    fake = Paused()
    bot = make_bot(tmp_path, monkeypatch, fake)
    bot.store.kv_set("balance_semantics", {"excludes_resting": True})
    monkeypatch.setattr(fake, "exchange_schedule", lambda: {}, raising=False)
    bot.next_due.clear()
    for _ in range(1800):                      # an hour of 2-second steps
        clock[0] += 2
        bot.step()
    assert not bot.risk.halted                 # was: "HALT: API error burst" every 15 minutes
    created = sorted(kw["expiration_ts"] for *_, kw in fake.created)
    per_min = max(sum(1 for t in created if a <= t < a + 60) for a in created)
    assert per_min < 15                        # never near the error-burst threshold (15 a minute)


def test_the_refusal_flag_does_not_outlive_its_pass(clock):
    from wxbot.kalshi.rest import KalshiError
    fake = FakeKalshi()
    m = LiveOrderManager(fake, ttl_s=300)
    close = {x["ticker"]: int(clock[0]) + 86400 for x in fake.markets}
    want = [DesiredOrder(x["ticker"], "ask", fake.quotes[x["ticker"].rsplit("-", 1)[1]][1], 5.0, "A", 0.0)
            for x in fake.markets]
    m.reconcile(want, close)
    real = fake.amend_order

    def refuse(*a, **kw):
        raise KalshiError(400, '{"error":{"code":"trading_is_paused"}}', "POST", "/portfolio/events/orders/x/amend")
    fake.amend_order = refuse
    clock[0] += 200
    assert m.reconcile(want, close)["refusal_stop"] == 1
    fake.amend_order = real
    clock[0] += 2
    assert m.reconcile(want, close, keep_alive=False)["refusal_stop"] == 0     # no refresh this pass: no stop


def test_the_no_orders_watch_stays_on_while_another_sleeve_is_stopped(tmp_path, monkeypatch, clock):
    fake = Acct()
    bot = make_bot(tmp_path, monkeypatch, fake)
    sent = []
    monkeypatch.setattr(bot.alerts, "topic", "t")
    monkeypatch.setattr(bot.alerts, "_post", lambda title, msg, pr, tags: sent.append(title) or True)
    bot.watch_quoting(3, clock[0])
    bot.last_stopped = {"X"}                   # the taker stopped for the day; Sleeve A still should be quoting
    for _ in range(14):
        clock[0] += 600
        bot.watch_quoting(0, clock[0])
    assert sent == ["wxbot live: no orders for 2 h"]


def test_the_heartbeat_goes_out_while_the_exchange_is_paused(tmp_path, monkeypatch, clock):
    fake = Acct()
    bot = make_bot(tmp_path, monkeypatch, fake)
    beats = []
    monkeypatch.setattr(bot.alerts, "heartbeat", lambda msg: beats.append(msg) or True)
    monkeypatch.setattr(fake, "exchange_status", lambda: {"trading_active": False}, raising=False)
    bot.next_due.clear()
    for _ in range(120):                       # two paused hours, a minute apart, through the run loop's upkeep
        clock[0] += 60
        bot.step()
        bot.alert_upkeep()
    assert len(beats) >= 11



class ThursdayExchange(Acct):
    """Kalshi's Thursday 3-5 AM ET trading pause: placements and amends refused, the bot's orders (sent with
    cancel_order_on_pause) cancelled when it starts. status_mode: what GET /exchange/status does meanwhile."""
    status_mode = "fails"
    P0 = et(2026, 10, 8, 3, 0)
    P1 = et(2026, 10, 8, 5, 0)

    def paused(self):
        return self.P0 <= time.time() < self.P1

    def _wipe(self):
        if self.paused() and not getattr(self, "wiped", False):
            self.resting.clear()
            self.wiped = True

    def create_order(self, ticker, side, price_cents, count, **kw):
        self._wipe()
        if self.paused():
            self.refused = getattr(self, "refused", 0) + 1
            from wxbot.kalshi.rest import KalshiError
            raise KalshiError(400, '{"error":{"code":"trading_is_paused"}}', "POST", "/portfolio/events/orders")
        return super().create_order(ticker, side, price_cents, count, **kw)

    def amend_order(self, oid, *a, **kw):
        self._wipe()
        if self.paused():
            from wxbot.kalshi.rest import KalshiError
            raise KalshiError(400, '{"error":{"code":"trading_is_paused"}}', "POST", "/amend")
        return super().amend_order(oid, *a, **kw)

    def orders(self, status="resting", **kw):
        self._wipe()
        return super().orders(status, **kw)

    def exchange_status(self):
        if not self.paused():
            return {"trading_active": True}
        if self.status_mode == "says_true":
            return {"trading_active": True}
        raise ConnectionError("GET /exchange/status: 503 Service Unavailable")


@pytest.mark.parametrize("status_mode, hours", [("fails", None), ("says_true", "kalshi")])
def test_thursday_maintenance_with_a_faulty_status_never_halts(tmp_path, monkeypatch, status_mode, hours):
    """Round 9: with the published hours unavailable and the status failing, or the status saying 'open' all through
    the pause, the bot used to trade into the pause and hit the error-burst halt about every 15 minutes."""
    c = [et(2026, 10, 8, 2, 50)]
    monkeypatch.setattr(time, "time", lambda: c[0])
    fake = ThursdayExchange()
    fake.status_mode = status_mode
    monkeypatch.setattr(fake, "exchange_schedule", lambda: KALSHI_HOURS if hours else {}, raising=False)
    bot = make_bot(tmp_path, monkeypatch, fake)
    bot.max_err = 15                                     # the live configs' threshold
    bot.store.kv_set("balance_semantics", {"excludes_resting": True})
    want = [DesiredOrder(f"KXHIGHNY-26OCT{9 + e:02d}-B{60 + 2 * b}.5", "ask", 97, 3.0, "A", 0.5,
                         event=f"KXHIGHNY-26OCT{9 + e:02d}") for e in range(12) for b in range(3)][:34]
    monkeypatch.setattr(bot, "decide", lambda: list(want))
    bot.next_due.clear()
    resumed = None
    while c[0] < et(2026, 10, 8, 5, 10):
        bot.step()
        if c[0] >= fake.P1 and resumed is None and bot.exchange_paused(c[0]) is None:
            resumed = c[0]
        c[0] += 2
    halts = [m for _, _, m in bot.store.c.execute("SELECT ts, level, msg FROM events_log") if m.startswith("HALT")]
    assert halts == [] and getattr(fake, "refused", 0) < 120
    assert resumed is not None and resumed - fake.P1 < 120


def test_a_broken_status_field_or_daily_summary_never_fails_a_step(tmp_path, monkeypatch, clock):
    fake = Acct()
    bot = make_bot(tmp_path, monkeypatch, fake)

    def boom():
        raise ZeroDivisionError("float division by zero")
    monkeypatch.setattr(bot, "mark_to_market", boom)
    bot.write_status()                                      # no exception
    assert bot.counters["side_err:status"] == 1
    monkeypatch.setattr(bot, "maybe_daily_summary", boom)
    bot.next_due.clear()
    bot.step()                                              # the summary fails as a side task, the step goes on
    assert bot.counters["side_err:daily_summary"] == 1 and bot.counters["step_errors"] == 0


def test_a_failing_stream_shadow_never_costs_a_fill(tmp_path, monkeypatch, clock):
    from test_live_loop import iso
    fake = Acct()
    bot = make_bot(tmp_path, monkeypatch, fake)
    bot.store.kv_set("balance_semantics", {"excludes_resting": True})

    class BrokenStream:
        def events_since(self, *a, **kw):
            raise KeyError("data")

        def set_markets(self, tickers):
            raise RuntimeError("socket closed")
    bot.ws = BrokenStream()
    oid = "o-mine"
    bot.live.my_order_ids.add(oid)
    bot.live.order_sleeve[oid] = "A"
    fake.fills = lambda min_ts=None, **kw: [{"fill_id": "f1", "trade_id": "t1", "order_id": oid,
                                             "market_ticker": fake.markets[4]["ticker"], "book_side": "ask",
                                             "yes_price_dollars": "0.0300", "count_fp": "5.00",
                                             "created_time": iso(clock[0]), "is_taker": False, "fee_cost": "0"}]
    bot.refresh_live_state()
    n = bot.store.c.execute("SELECT COUNT(*) FROM fills WHERE mode='live'").fetchone()[0]
    assert n == 1 and bot.counters["side_err:ws_fills"] == 1


def test_a_fill_that_cannot_be_recorded_is_retried_not_lost(tmp_path, monkeypatch, clock):
    import sqlite3
    from test_live_loop import iso
    fake = Acct()
    bot = make_bot(tmp_path, monkeypatch, fake)
    bot.store.kv_set("balance_semantics", {"excludes_resting": True})
    oid = "o-mine"
    bot.live.my_order_ids.add(oid)
    bot.live.order_sleeve[oid] = "A"
    fake.fills = lambda min_ts=None, **kw: [{"fill_id": "f1", "trade_id": "t1", "order_id": oid,
                                             "market_ticker": fake.markets[4]["ticker"], "book_side": "ask",
                                             "yes_price_dollars": "0.0300", "count_fp": "5.00",
                                             "created_time": iso(clock[0] - 30), "is_taker": False, "fee_cost": "0"}]
    real = bot.store.add_fill
    calls = {"n": 0}

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return real(*a, **kw)
    monkeypatch.setattr(bot.store, "add_fill", flaky)
    bot.refresh_live_state()
    assert bot.store.c.execute("SELECT COUNT(*) FROM fills WHERE mode='live'").fetchone()[0] == 0
    fake.fills = lambda min_ts=None, **kw: []          # the exchange's poll window has moved on
    clock[0] += 5
    bot.refresh_live_state()
    assert bot.store.c.execute("SELECT COUNT(*) FROM fills WHERE mode='live'").fetchone()[0] == 1


def test_a_fill_that_shows_up_late_is_still_polled(clock):
    from test_live_loop import iso
    fake = FakeKalshi()
    m = LiveOrderManager(fake, ttl_s=300)
    m.my_order_ids.add("o1")
    feed = []

    def fills(min_ts=None, **kw):
        from wxbot.timeutil import parse_ts
        return [f for f in feed if parse_ts(f["created_time"]) >= (min_ts or 0)]
    fake.fills = fills
    mk = lambda fid, ts: {"fill_id": fid, "order_id": "o1", "market_ticker": fake.markets[4]["ticker"],
                          "book_side": "ask", "yes_price_dollars": "0.0300", "count_fp": "1.00",
                          "created_time": iso(ts), "is_taker": False, "fee_cost": "0"}
    feed.append(mk("f2", clock[0] - 10))
    assert [f.order_ref for f in m.poll_fills()] == ["f2"]
    feed.append(mk("f1", clock[0] - 40))            # older, but visible only now (the account data lagged)
    clock[0] += 5
    assert [f.order_ref for f in m.poll_fills()] == ["f1"]


def test_a_fill_of_a_go_live_test_order_is_not_the_bots(clock):
    from test_live_loop import iso
    fake = FakeKalshi()
    m = LiveOrderManager(fake, ttl_s=300)
    fake.order = lambda oid: {"order_id": oid, "client_order_id": {"os": "wxb-smoke-1a2b3c", "ob": "wxb-A-1a2b"}[oid]}
    fake.fills = lambda min_ts=None, **kw: [
        {"fill_id": f"f-{oid}", "order_id": oid, "market_ticker": fake.markets[4]["ticker"], "book_side": "bid",
         "yes_price_dollars": "0.0100", "count_fp": "2.00", "created_time": iso(clock[0]), "is_taker": False,
         "fee_cost": "0"} for oid in ("os", "ob")]
    assert [(f.order_ref, f.sleeve) for f in m.poll_fills()] == [("f-ob", "A")]
