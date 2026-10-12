"""Second verification review of the Oct 4 fixes (commit 0148457): each problem reproduced first, then fixed.
N1  one cancel the exchange never confirms froze every new order on every market, pass after pass, and an amend
    answered 404 (the order filled meanwhile) became a cancel that failed the same way;
N2  a success answer whose body could not be read (requests' JSONDecodeError is a ValueError) was taken for a local
    refusal, so the order rested untracked and the next pass stacked a second one beside it;
N3  if Kalshi's account-data timestamp only moved with account activity, an idle account would never count as fresh
    (an unsigned read shows an exchange-wide clock that moves on its own; the smoke test now checks the signed one);
N4  while account reads failed, the stops could not see losses but the bot kept extending its orders' expiry;
L3  an oversized key was not shrunk while an unanswered placement sat beside it, and new orders went out anyway;
L5  takes did not wait for three equity readings; after an outage the median mixed readings from before it."""
import os
import sys
import time

import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_live_loop import FakeKalshi  # noqa: E402
from test_live_safety import StubbornKalshi, bare_bot  # noqa: E402
from test_review_round2 import Acct, make_bot, runner  # noqa: E402

from wxbot.execution.live import CANCEL_PENDING, RESTING, UNCERTAIN, LiveOrderManager, definitely_refused  # noqa: E402
from wxbot.kalshi.rest import KalshiError, KalshiREST  # noqa: E402
from wxbot.strategy.types import DesiredOrder  # noqa: E402


def want(fake, ks, px=3, n=10.0):
    return [DesiredOrder(fake.markets[k]["ticker"], "ask", px, n, "A", 0.0) for k in ks]


@pytest.fixture
def clock(monkeypatch):
    """A clock that tests move by hand. It starts at least two hours from an ET midnight, so a test that runs
    simulated minutes never crosses into a new trading day (which resets the daily stop by design)."""
    from wxbot.timeutil import to_et
    c = [time.time()]
    if to_et(c[0]).hour >= 22 or to_et(c[0]).hour < 2:
        c[0] += 4 * 3600
    monkeypatch.setattr(time, "time", lambda: c[0])
    return c


# ------------------------------------------------------------------------------------------------ N1
def test_a_cancel_that_keeps_failing_freezes_one_pass_only(clock):
    fake = StubbornKalshi(stuck_rounds=0, never={"o1"})
    m = LiveOrderManager(fake, ttl_s=300)
    close = {x["ticker"]: int(clock[0]) + 86400 for x in fake.markets}
    m.reconcile(want(fake, [4]), close)                       # o1 rests on key 4 and can never be cancelled
    others = want(fake, [0, 1], px=99, n=5.0)
    clock[0] += 2
    st = m.reconcile(others, close)                           # key 4 no longer wanted: its cancel fails
    assert st["frozen"] == 1 and st["placed"] == 0           # this pass's risk check may have counted it as gone
    assert [o.state for o in m.resting_list()] == [CANCEL_PENDING]
    placed = 0
    for _ in range(30):
        clock[0] += 2
        st = m.reconcile(others, close)                       # the retries keep failing...
        placed += st["placed"]
        assert st["cancel_unconfirmed"] == 1 and not st["frozen"]
    assert placed == 2                                        # ...but the other markets got their orders
    # the stuck order still counts in full and never has its expiry pushed: it lapses within its TTL
    assert [c.reason for c in m.committed(others)] == ["committed:cancel_pending"]
    assert not [a for a in fake.amended if a[0] == "o1"]


class FillRace(FakeKalshi):
    """Amending or cancelling an order that no longer rests (filled meanwhile) -> 404 / a per-order error."""

    def amend_order(self, oid, *a, **kw):
        if oid not in self.resting:
            raise KalshiError(404, '{"error":{"code":"not_found"}}', "POST", f"/portfolio/events/orders/{oid}/amend")
        return super().amend_order(oid, *a, **kw)

    def batch_cancel(self, ids, **kw):
        out = []
        for i in ids:
            if i in self.resting:
                rem = self.resting[i]["remaining_count_fp"]
                self.cancel_order(i)
                out.append({"order_id": i, "reduced_by": rem, "ts_ms": int(time.time() * 1000)})
            else:
                out.append({"order_id": i, "reduced_by": "0.00", "error": {"code": "not_found"}})
        return {"orders": out}


def test_an_amend_answered_404_is_settled_by_snapshots_not_by_a_cancel(clock):
    fake = FillRace()
    m = LiveOrderManager(fake, ttl_s=300)
    close = {x["ticker"]: int(clock[0]) + 86400 for x in fake.markets}
    m.reconcile(want(fake, [4]), close)                       # o1: 10 on key 4
    fake.resting.pop("o1")                                    # o1 fills in full; no snapshot has shown it yet
    clock[0] += 2
    w = want(fake, [4], n=4.0) + want(fake, [1], px=99, n=5.0)
    st = m.reconcile(w, close)                                # shrink o1 to 4: 404
    o1 = m.resting_list()[0]
    assert o1.state == UNCERTAIN and o1.order_id == "o1"     # still counted (its fill may not be in positions yet)
    assert not fake.cancelled and st["frozen"] == 1          # no cancel sent for it; nothing new this pass
    assert [c.reason for c in m.committed(w)] == ["committed:uncertain"]
    clock[0] += 2
    st = m.reconcile(w, close)
    assert st["placed"] == 1 and st["blocked"] == 1          # the other market goes ahead; key 4 waits
    for _ in range(2):                                        # two snapshots newer than the 404 settle it
        clock[0] += 5
        m.sync()
    assert all(o.order_id != "o1" for o in m.resting_list())
    st = m.reconcile(w, close)
    assert st["placed"] == 1 and fake.created[-1][0] == fake.markets[4]["ticker"]


def test_a_wrong_404_brings_the_order_back_from_the_next_snapshot(clock):
    """If a 404 were wrong (the order still rests), the next snapshot adopts it again: never stacked, never lost."""
    class Wrong404(FakeKalshi):
        def amend_order(self, oid, *a, **kw):
            raise KalshiError(404, "not found", "POST", "/amend")
    fake = Wrong404()
    m = LiveOrderManager(fake, ttl_s=300)
    close = {x["ticker"]: int(clock[0]) + 86400 for x in fake.markets}
    m.reconcile(want(fake, [4]), close)
    clock[0] += 2
    m.reconcile(want(fake, [4], n=4.0), close)                # 404 on the shrink: uncertain
    for _ in range(5):
        clock[0] += 2
        m.reconcile(want(fake, [4], n=4.0), close)            # the key stays blocked: nothing stacked
    assert len(fake.resting) == 1
    clock[0] += 5
    m.sync()
    assert [(o.order_id, o.state) for o in m.resting_list()] == [("o1", RESTING)]


# ------------------------------------------------------------------------------------------------ N2
class Garbled(FakeKalshi):
    garble = 1

    def create_order(self, *a, **kw):
        r = super().create_order(*a, **kw)
        if self.garble:
            self.garble -= 1
            raise requests.exceptions.JSONDecodeError("Expecting value", "<html>", 0)
        return r


def test_a_garbled_success_answer_leaves_an_uncertain_order_not_a_refusal(clock):
    fake = Garbled()
    m = LiveOrderManager(fake, ttl_s=300)
    close = {x["ticker"]: int(clock[0]) + 86400 for x in fake.markets}
    st = m.reconcile(want(fake, [4]), close)
    assert st["uncertain"] == 1 and [o.state for o in m.resting_list()] == [UNCERTAIN]
    clock[0] += 2
    m.reconcile(want(fake, [4]), close)
    assert len(fake.resting) == 1                             # never a second order beside it
    clock[0] += 5
    m.sync()
    assert [(o.order_id, o.state) for o in m.resting_list()] == [("o1", RESTING)]


def test_what_counts_as_a_definite_refusal():
    assert definitely_refused(KalshiError(400, "bad", "POST", "/x"))
    assert not definitely_refused(KalshiError(400, "dup", "POST", "/x", ambiguous=True))
    assert not definitely_refused(KalshiError(429, "slow down", "POST", "/x"))
    assert not definitely_refused(KalshiError(201, "<html", "POST", "/x", ambiguous=True))
    assert not definitely_refused(requests.exceptions.JSONDecodeError("Expecting value", "<html>", 0))
    assert not definitely_refused(requests.exceptions.ReadTimeout("read timed out"))
    assert not definitely_refused(ConnectionError("reset"))
    assert definitely_refused(ValueError("side must be 'bid' or 'ask'"))


def test_the_rest_client_turns_an_unreadable_success_answer_into_an_ambiguous_error():
    class Resp:
        status_code, content, text = 201, b"<html>gateway", "<html>gateway"

        def json(self):
            raise ValueError("Expecting value")

    class Auth:
        def headers(self, method, path):
            return {}
    r = KalshiREST(auth=Auth())
    r.s.request = lambda *a, **kw: Resp()
    with pytest.raises(KalshiError) as ei:
        r.create_order("KXHIGHNY-26OCT05-T75", "ask", 3, 10, client_order_id="wxb-A-x")
    assert ei.value.status == 201 and ei.value.ambiguous and not definitely_refused(ei.value)


# ------------------------------------------------------------------------------------------------ L3
def test_an_oversized_key_shrinks_beside_an_unanswered_placement(clock):
    class Lost(FakeKalshi):
        lose = 0

        def create_order(self, *a, **kw):
            r = super().create_order(*a, **kw)
            if self.lose:
                self.lose -= 1
                raise ConnectionError("read timed out")
            return r
    fake = Lost()
    m = LiveOrderManager(fake, ttl_s=300)
    t4, t1 = fake.markets[4]["ticker"], fake.markets[1]["ticker"]
    close = {x["ticker"]: int(clock[0]) + 86400 for x in fake.markets}
    m.reconcile([DesiredOrder(t4, "ask", 3, 10.0, "A", 0.0)], close)          # o1: 10
    m.resting_list()[0].remaining, m.resting_list()[0].filled = 4.0, 6.0      # 6 filled
    fake.resting["o1"]["remaining_count_fp"], fake.resting["o1"]["fill_count_fp"] = "4.00", "6.00"
    fake.lose = 1
    clock[0] += 2
    m.reconcile([DesiredOrder(t4, "ask", 3, 10.0, "A", 0.0)], close)          # top-up of 6: answer lost
    assert sorted(o.state for o in m.resting_list()) == [RESTING, UNCERTAIN]
    clock[0] += 2
    st = m.reconcile([DesiredOrder(t4, "ask", 3, 2.0, "A", 0.0), DesiredOrder(t1, "ask", 99, 5.0, "A", 0.0)], close)
    assert st["amended"] == 1 and fake.resting["o1"]["remaining_count_fp"] == "2.00"   # shrunk to what is wanted
    assert st["placed"] == 1                                                  # then the other market
    # the unanswered top-up (it did rest) is counted in full until a snapshot settles it, then shrunk away
    w = [DesiredOrder(t4, "ask", 3, 2.0, "A", 0.0), DesiredOrder(t1, "ask", 99, 5.0, "A", 0.0)]
    assert [(c.ticker, c.count, c.reason) for c in m.committed(w)] == [(t4, 6.0, "committed:uncertain")]
    clock[0] += 5
    m.sync()
    clock[0] += 2
    m.reconcile(w, close)
    assert sum(float(v["remaining_count_fp"]) for v in fake.resting.values() if v["ticker"] == t4) == 2.0


# ------------------------------------------------------------------------------------------------ L5
def test_takes_wait_for_three_equity_readings(tmp_path):
    from wxbot.risk.limits import RiskConfig
    fake = FakeKalshi()
    bot = bare_bot(tmp_path, fake)
    bot.risk_cfg = RiskConfig()
    bot.events, bot.hourly, bot.ev_tickers, bot.ev_tickers_miss, bot.netting = {}, {}, {}, {}, False
    bot.ev_tickers[fake.ev] = [m["ticker"] for m in fake.markets]
    bot.acct_ts = {k: time.time() for k in ("sync", "balance", "positions")}
    cfg = {"trade": True, "trade_size": 5, "trade_max_usd_day": 25}
    for n in (0, 2):
        bot.eq_readings = [625.0] * n
        assert bot.execute_take("X", fake.ev, fake.markets[2]["ticker"], "no", 0.40, 5, cfg) is None
    assert not fake.created
    bot.eq_readings = [625.0] * 3
    assert bot.execute_take("X", fake.ev, fake.markets[2]["ticker"], "no", 0.40, 5, cfg) is not None
    assert len(fake.created) == 1


# ------------------------------------------------------------------------------------------------ N4
class ExpiringAcct(Acct):
    """Orders lapse at their expiration_time, as on the exchange."""

    def expire(self, now):
        from wxbot.timeutil import parse_ts
        for oid, o in list(self.resting.items()):
            if o.get("expiration_time") and parse_ts(o["expiration_time"]) <= now:
                self.resting.pop(oid)


def test_a_bot_that_cannot_read_its_account_lets_its_orders_lapse_and_restarts_its_median(tmp_path, monkeypatch, clock):
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)
    fake = ExpiringAcct()
    fake.pos["KXHIGHCHI-26OCT05-B60.5"] = (20.0, 20.0)     # a $20 YES position elsewhere
    bot = make_bot(tmp_path, monkeypatch, fake)
    bot.store.kv_set("balance_semantics", {"excludes_resting": True})

    def tick(n):
        for _ in range(n):
            bot.step()
            clock[0] += 2
            fake.expire(clock[0])
    tick(12)
    assert fake.resting and not bot.risk.halted
    fake.orders_down = True                                   # GET /portfolio/orders fails; the rest works
    n_amend = len(fake.amended)
    fake.pos.pop("KXHIGHCHI-26OCT05-B60.5")                   # it settles NO: -$20, past the $15 daily stop
    tick(300)                                                 # ten minutes blind
    assert len(fake.amended) == n_amend                       # no expiry pushed while blind...
    assert not fake.resting                                   # ...so the orders lapsed (Sleeve A: 5 min)
    fake.orders_down = False
    created0 = len(fake.created)
    seen = []
    for _ in range(20):
        tick(1)
        seen.append(len(bot.eq_readings))
        if bot.risk.halted:
            break
    assert bot.risk.halted and bot.risk.halt_kind == "daily"
    assert seen[0] == 1                                       # the median started over after the gap
    assert len(fake.created) == created0                      # nothing placed before the stop had its say


# ------------------------------------------------------------------------------------------------ N3
class LastActivity(FakeKalshi):
    """A timestamp that only moves with account activity (NOT what Kalshi's unsigned read shows: there it is an
    exchange-wide clock that moves on its own; the smoke test checks the signed read)."""

    def __init__(self):
        super().__init__()
        self.last_event = time.time() - 3600

    def create_order(self, *a, **kw):
        self.last_event = time.time()
        return super().create_order(*a, **kw)

    def user_data_timestamp(self):
        return self.last_event


@pytest.mark.parametrize("moves_idle, trades", [(None, False), (False, True)])
def test_freshness_uses_read_times_when_the_smoke_test_found_a_per_account_timestamp(tmp_path, monkeypatch, clock,
                                                                                     moves_idle, trades):
    from wxbot.store.db import Store
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)
    if moves_idle is not None:
        Store(str(tmp_path / "live.sqlite")).kv_set("api_observations", {"user_data_ts_moves_idle": moves_idle})
    fake = LastActivity()
    fake.last_event = clock[0] - 3600
    bot = make_bot(tmp_path, monkeypatch, fake)
    for _ in range(30):
        bot.step()
        clock[0] += 2
    assert bool(fake.created) is trades


# ------------------------------------------------------------------------------------------------ round 3
def test_an_excess_spread_over_two_orders_comes_off_both_before_anything_new(clock):
    """Round 3: with 4 left on the first order and a top-up of 6 beside it, and 2 now wanted, the old code cancelled
    the top-up and placed elsewhere while 4 still rested against the 2 the risk check counted."""
    fake = FakeKalshi()
    m = LiveOrderManager(fake, ttl_s=300)
    t4, t1 = fake.markets[4]["ticker"], fake.markets[1]["ticker"]
    close = {x["ticker"]: int(clock[0]) + 86400 for x in fake.markets}
    want10 = [DesiredOrder(t4, "ask", 3, 10.0, "A", 0.0)]
    m.reconcile(want10, close)                                                # o1: 10
    m.resting_list()[0].remaining, m.resting_list()[0].filled = 4.0, 6.0
    fake.resting["o1"]["remaining_count_fp"], fake.resting["o1"]["fill_count_fp"] = "4.00", "6.00"   # 6 filled
    clock[0] += 2
    m.reconcile(want10, close)                                                # top-up o2: 6
    assert sorted(o.remaining for o in m.resting_list()) == [4.0, 6.0]
    clock[0] += 2
    st = m.reconcile([DesiredOrder(t4, "ask", 3, 2.0, "A", 0.0), DesiredOrder(t1, "ask", 99, 5.0, "A", 0.0)], close)
    on_t4 = sum(float(v["remaining_count_fp"]) for v in fake.resting.values() if v["ticker"] == t4)
    assert on_t4 == 2.0 and list(fake.resting) == ["o1", "o3"]               # o2 cancelled, o1 amended 4 -> 2
    assert st["cancelled"] == 1 and st["amended"] == 1 and st["placed"] == 1 and not st["frozen"]


def test_an_excess_that_cannot_come_off_this_pass_freezes_it(clock):
    class NoAmend(FakeKalshi):
        def amend_order(self, *a, **kw):
            raise KalshiError(503, "busy", "POST", "/amend", ambiguous=True)

        def batch_cancel(self, ids, **kw):
            return {"orders": [{"order_id": i, "error": {"code": "internal"}} for i in ids]}
    fake = NoAmend()
    m = LiveOrderManager(fake, ttl_s=300)
    t4, t1 = fake.markets[4]["ticker"], fake.markets[1]["ticker"]
    close = {x["ticker"]: int(clock[0]) + 86400 for x in fake.markets}
    m.reconcile([DesiredOrder(t4, "ask", 3, 10.0, "A", 0.0)], close)
    clock[0] += 2
    st = m.reconcile([DesiredOrder(t4, "ask", 3, 4.0, "A", 0.0), DesiredOrder(t1, "ask", 99, 5.0, "A", 0.0)], close)
    assert st["frozen"] == 1 and st["placed"] == 0 and len(fake.resting) == 1


def test_the_blind_alert_also_fires_when_kalshi_account_data_stalls(tmp_path, monkeypatch, clock):
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)
    fake = FakeKalshi()
    bot = make_bot(tmp_path, monkeypatch, fake)
    sent = []
    monkeypatch.setattr(bot.alerts, "send", lambda key, *a, **kw: sent.append(key))
    for _ in range(15):
        bot.step()
        clock[0] += 2
    fake.as_of = clock[0]                     # the account data stops moving; every read still succeeds
    for _ in range(90):
        bot.step()
        clock[0] += 2
    assert not bot.account_fresh(clock[0]) and sent.count("blind") == 1


# ------------------------------------------------------------------------------------------------ dress rehearsal
def test_a_failing_shadow_feature_never_stops_live_trading(tmp_path, monkeypatch, clock):
    """Oct 4 dress rehearsal of config/live_go.yaml: with the hourly shadow on, a trades read that fails (only the
    shadow needs it) ended every live step and, after ten, halted the bot. Side tasks now fail on their own."""
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)

    class NoTape(FakeKalshi):
        def trades(self, *a, **kw):
            raise KalshiError(503, "trades unavailable", "GET", "/markets/trades")
    fake = NoTape()
    bot = make_bot(tmp_path, monkeypatch, fake)
    bot.h_on = True                                   # the hourly shadow reads the tape in live mode
    bot.settle_h = lambda *a, **kw: 1 / 0             # and its settlement breaks too
    bot.next_due["h_settle"] = 0
    for _ in range(40):
        bot.step()
        clock[0] += 2
        bot.next_due["trades"] = 0
    assert bot.counters["step_errors"] == 0 and not bot.risk.halted and fake.created
    st = bot.write_status()
    assert st["side_task_errors"]["trades"] >= 30 and st["side_task_errors"]["h_settle"] == 1


def test_a_failing_breaker_check_switches_real_hourly_orders_off(tmp_path, monkeypatch, clock):
    """Round 4: wrapped as a side task, a failing breaker check left real hourly orders on. While they are real,
    a check that cannot run now trips the breaker (fail-safe); while they are shadow only, it is a side error."""
    import wxbot.strategy.sleeve_h as sleeve_h
    bot = make_bot(tmp_path, monkeypatch, FakeKalshi())
    monkeypatch.setattr(sleeve_h, "breaker_stats", lambda *a, **k: 1 / 0)
    bot.cfg_h.enabled = False
    bot.guarded_h_breaker()                         # shadow only: the check returns before computing anything
    bot.cfg_h.enabled = True
    bot.guarded_h_breaker()
    assert bot.cfg_h.enabled is False and bot.h_breaker["tripped"]
    assert "breaker check failed" in bot.h_breaker["reason"]


def test_events_that_break_the_bots_price_shard_or_payout_assumptions_are_skipped(tmp_path, monkeypatch, clock):
    """Kalshi moved products to sub-cent ticks, other shards and floor settlements in Aug-Oct 2026. Weather was
    standard on Oct 4; if that changes, the event is skipped with a warning instead of being mispriced."""
    fake = FakeKalshi()
    bot = make_bot(tmp_path, monkeypatch, fake)
    bot.refresh_universe()
    assert fake.ev in bot.events
    for field, value in (("price_level_structure", "center_deci_edge_centi_cent"), ("exchange_index", 2),
                         ("settlement_bounds_type", "floor")):
        for m in fake.markets:
            m.pop("price_level_structure", None), m.pop("exchange_index", None), m.pop("settlement_bounds_type", None)
        fake.markets[2][field] = value
        bot.next_due.clear()
        bot.refresh_universe()
        assert fake.ev not in bot.events, field
    warns = [r[0] for r in bot.store.c.execute("SELECT msg FROM events_log WHERE level='WARN'").fetchall()]
    assert any("skipped: price structure" in w for w in warns) and any("settlement bounds floor" in w for w in warns)


def test_a_whole_cent_grid_under_a_new_structure_name_is_not_skipped(tmp_path, monkeypatch, clock):
    """Round 5: Kalshi says price_ranges is the source of truth; whole cents are valid in every structure."""
    fake = FakeKalshi()
    bot = make_bot(tmp_path, monkeypatch, fake)
    for m in fake.markets:
        m["price_level_structure"] = "some_new_name"
        m["price_ranges"] = [{"start": "0.0000", "end": "1.0000", "step": "0.0100"}]
    bot.refresh_universe()
    assert fake.ev in bot.events
    fake.markets[1]["price_ranges"] = [{"start": "0.0000", "end": "0.0100", "step": "0.0010"},
                                       {"start": "0.0100", "end": "1.0000", "step": "0.0100"}]
    bot.next_due.clear()
    bot.refresh_universe()
    assert fake.ev not in bot.events


def test_a_single_side_written_as_a_string_is_one_side():
    from wxbot.strategy import sleeve_h
    assert sleeve_h.SleeveHConfig.from_cfg({"sides": "ask"}).sides == ("ask",)


def test_a_post_only_order_cancelled_on_arrival_is_treated_as_a_refusal(clock):
    """Kalshi's FIX docs report a post-only order that would cross as cancelled on arrival (POST_ONLY_CROSS). If REST
    answers that way (success, nothing left), the bot must not track an empty order and place again every pass."""
    class CancelOnArrival(FakeKalshi):
        def create_order(self, ticker, side, price_cents, count, **kw):
            yb, ya = self.quotes.get(ticker.rsplit("-", 1)[1], (None, None))
            if kw.get("post_only") and ((side == "bid" and ya is not None and price_cents >= ya) or
                                        (side == "ask" and yb is not None and price_cents <= yb)):
                self.created.append((ticker, side, price_cents, count, kw))
                return {"order_id": f"x{len(self.created)}", "remaining_count": "0.00", "fill_count": "0.00"}
            return super().create_order(ticker, side, price_cents, count, **kw)
    fake = CancelOnArrival()
    m = LiveOrderManager(fake, ttl_s=300)
    close = {x["ticker"]: int(clock[0]) + 86400 for x in fake.markets}
    t = fake.markets[4]["ticker"]                       # YES bid 1c: an ask at 1c would cross
    st = m.reconcile([DesiredOrder(t, "ask", 1, 10.0, "A", 0.0)], close)
    assert st["rejected"] == 1 and st["placed"] == 0 and not m.resting_list()
    clock[0] += 2
    st = m.reconcile([DesiredOrder(t, "ask", 1, 10.0, "A", 0.0)], close)
    assert st["cooldown"] == 1 and len(fake.created) == 1          # no second try inside the cooldown
