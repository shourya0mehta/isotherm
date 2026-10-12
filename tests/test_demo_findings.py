"""What the two-day demo test found on Oct 6 (scripts/demo_gate.py), each reproduced first, then fixed.
1. Item 4, duplicate orders: when the size wanted on a market grew past twice what rested there, the bot placed a
   second order beside the first even though nothing had filled (2 resting, 5 wanted: a second order of 3). The rules
   allow a second order only as a top-up beside a mostly filled one. Now a bigger size grows the resting order with
   an amend (Kalshi puts it at the back of its price's queue, where a second order would sit too), and a top-up
   still goes beside a mostly filled order. The order test (smoke) checks a growth on Kalshi, and that every amend
   keeps the expiry it names (the dead-man switch).
2. Item 3, slow cancel confirmations: Kalshi's demo answered "service unavailable" and then paused trading for about
   two minutes, so cancels the bot sent just before could not settle until it came back. The bot now logs where a
   pause starts and ends, and the gates leave the paused time out (tests/test_stage2_gate.py)."""
import logging
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_live_loop import FakeKalshi  # noqa: E402

from wxbot.execution.live import RESTING, UNCERTAIN, LiveOrderManager  # noqa: E402
from wxbot.kalshi.rest import KalshiError  # noqa: E402
from wxbot.strategy.types import DesiredOrder  # noqa: E402


@pytest.fixture
def clock(monkeypatch):
    c = [time.time()]
    monkeypatch.setattr(time, "time", lambda: c[0])
    return c


def setup(fake=None):
    fake = fake or FakeKalshi()
    m = LiveOrderManager(fake, ttl_s=300)
    t = fake.markets[4]["ticker"]
    close = {x["ticker"]: int(time.time()) + 86400 for x in fake.markets}
    return fake, m, t, close


def w(t, n, px=3):
    return [DesiredOrder(t, "ask", px, float(n), "A", 0.0)]


def fill(fake, m, oid, n, clock):
    """n contracts of a resting order fill on the exchange; the next snapshot tells the bot."""
    o = fake.resting[oid]
    o["remaining_count_fp"] = f"{float(o['remaining_count_fp']) - n:.2f}"
    o["fill_count_fp"] = f"{float(o['fill_count_fp']) + n:.2f}"
    clock[0] += 3
    m.sync()
    clock[0] += 3
    m.sync()


def on_market(fake, t):
    return [(k, float(v["remaining_count_fp"])) for k, v in fake.resting.items() if v["ticker"] == t]


def test_a_bigger_size_beside_an_unfilled_order_grows_it_instead_of_adding_one(clock):
    fake, m, t, close = setup()
    m.reconcile(w(t, 2), close)                                    # o1: 2 (the risk check had room for 2)
    clock[0] += 2
    st = m.reconcile(w(t, 5), close)                               # now 5 wanted
    assert len(fake.created) == 1, "a second order went out beside an unfilled one"
    assert on_market(fake, t) == [("o1", 5.0)]
    assert fake.amended[-1][:3] == ("o1", 3, 5.0)                  # the total it may fill: 0 filled + 5
    assert st["grown"] == 1 and st["placed"] == 0
    [o] = m.resting_list()
    assert o.state == RESTING and o.remaining == 5.0
    clock[0] += 2
    st = m.reconcile(w(t, 5), close)                               # nothing more to do
    assert st["kept"] == 1 and len(fake.created) == 1 and len(fake.amended) == 1


def test_a_top_up_still_goes_beside_a_mostly_filled_order(clock):
    fake, m, t, close = setup()
    m.reconcile(w(t, 10), close)                                   # o1: 10
    fill(fake, m, "o1", 7, clock)                                  # 7 filled, 3 left
    st = m.reconcile(w(t, 10), close)
    assert st["placed"] == 1 and st.get("grown", 0) == 0           # o1 keeps its place in line
    assert sorted(on_market(fake, t)) == [("o1", 3.0), ("o2", 7.0)]


def test_a_partly_filled_order_grows_unless_mostly_filled(clock):
    fake, m, t, close = setup()
    m.reconcile(w(t, 4), close)                                    # o1: 4
    fill(fake, m, "o1", 1, clock)                                  # 1 filled, 3 left: not mostly filled
    st = m.reconcile(w(t, 10), close)
    assert st["grown"] == 1 and len(fake.created) == 1
    assert fake.amended[-1][:3] == ("o1", 3, 11.0)                 # 1 filled + 10 to rest
    assert on_market(fake, t) == [("o1", 10.0)]


def test_mostly_filled_is_judged_on_the_largest_size_the_order_had(clock):
    """The gate judges a top-up on the earlier order's initial size (Kalshi's initial_count): an order shrunk from 10
    to 4 that then fills 2 is not mostly filled, so it grows rather than getting a second order beside it."""
    fake, m, t, close = setup()
    m.reconcile(w(t, 10), close)                                   # o1: 10
    clock[0] += 2
    m.reconcile(w(t, 4), close)                                    # amended down to 4
    assert on_market(fake, t) == [("o1", 4.0)]
    fill(fake, m, "o1", 2, clock)                                  # 2 of the 4 filled (2 of the 10 it started at)
    st = m.reconcile(w(t, 10), close)
    assert st["grown"] == 1 and len(fake.created) == 1
    assert on_market(fake, t) == [("o1", 10.0)]


def test_growth_waits_while_new_orders_are_not_allowed(clock):
    fake, m, t, close = setup()
    m.reconcile(w(t, 2), close)
    clock[0] += 2
    st = m.reconcile(w(t, 5), close, allow_new=False)              # stale account reads or cash below the reserve
    assert st["kept"] == 1 and not fake.amended and len(fake.created) == 1
    assert on_market(fake, t) == [("o1", 2.0)]


@pytest.mark.parametrize("body, key", [
    ('{"error":{"code":"insufficient_balance","message":"insufficient balance"}}', "no_funds"),
    ('{"error":{"code":"invalid_order","message":"invalid order"}}', "refused"),
])
def test_a_refused_growth_leaves_the_order_and_places_nothing_beside_it(clock, body, key):
    class NoGrow(FakeKalshi):
        def amend_order(self, oid, ticker, side, price, total, expiration_ts=None, **kw):
            if float(total) > float(self.resting[oid]["remaining_count_fp"]) + float(
                    self.resting[oid]["fill_count_fp"]):
                raise KalshiError(400, body, "POST", f"/portfolio/events/orders/{oid}/amend")
            return super().amend_order(oid, ticker, side, price, total, expiration_ts, **kw)
    fake, m, t, close = setup(NoGrow())
    m.reconcile(w(t, 2), close)
    for _ in range(5):
        clock[0] += 2
        st = m.reconcile(w(t, 5), close)
        assert len(fake.created) == 1 and on_market(fake, t) == [("o1", 2.0)]
    [o] = m.resting_list()
    assert o.state == RESTING and o.remaining == 2.0
    if key == "refused":                                           # a refusal cools the market down (no retry storm)
        assert len(m.errors) == 1 and st["cooldown"] == 1


def test_a_growth_whose_answer_is_lost_counts_the_larger_size_until_a_snapshot_settles_it(clock):
    class Lost(FakeKalshi):
        lose = 0

        def amend_order(self, *a, **kw):
            if self.lose:
                self.lose -= 1
                raise ConnectionError("read timed out")              # nothing changed on the exchange
            return super().amend_order(*a, **kw)
    fake, m, t, close = setup(Lost())
    m.reconcile(w(t, 2), close)
    fake.lose = 1
    clock[0] += 2
    st = m.reconcile(w(t, 5), close)
    assert st["uncertain"] == 1 and len(fake.created) == 1
    [o] = m.resting_list()
    assert o.remaining == 5.0                                      # counted at the larger size (the worst case)
    clock[0] += 2
    st = m.reconcile(w(t, 5), close)                               # an older snapshot cannot undo it: nothing new
    assert len(fake.created) == 1
    clock[0] += 3
    m.sync()                                                       # a newer snapshot: still 2 on the exchange
    assert m.resting_list()[0].remaining == 2.0
    clock[0] += 2
    st = m.reconcile(w(t, 5), close)
    assert st["grown"] == 1 and len(fake.created) == 1 and on_market(fake, t) == [("o1", 5.0)]


def test_a_growth_answered_404_places_nothing_beside_the_order(clock):
    class Gone(FakeKalshi):
        def amend_order(self, oid, *a, **kw):
            raise KalshiError(404, '{"error":{"code":"not_found"}}', "POST", f"/portfolio/events/orders/{oid}/amend")
    fake, m, t, close = setup(Gone())
    m.reconcile(w(t, 2), close)
    clock[0] += 2
    st = m.reconcile(w(t, 5), close)
    assert len(fake.created) == 1
    assert [o.state for o in m.resting_list()] == [UNCERTAIN]       # most likely filled: counted until settled
    clock[0] += 2
    st = m.reconcile(w(t, 5), close)
    assert st["blocked"] == 1 and len(fake.created) == 1


def test_dry_run_grows_too(clock):
    fake, m, t, close = setup()
    m.dry_run = True
    m.reconcile(w(t, 2), close)
    clock[0] += 2
    st = m.reconcile(w(t, 5), close)
    assert st["grown"] == 1 and st["placed"] == 0
    assert [o.remaining for o in m.resting_list()] == [5.0] and not fake.created


# ------------------------------------------------------------------- the order test (smoke) checks the same on Kalshi
def run_smoke(monkeypatch, tmp_path, patch):
    from test_smoke_alerts import smoke_env

    import wxbot.app.smoke as smoke
    fake = smoke_env(monkeypatch)
    patch(fake)
    return smoke.run_smoke({"env": "prod", "db_path": str(tmp_path / "live.sqlite")}), fake


def test_the_order_test_grows_an_order_and_reads_back_size_client_id_and_expiry(monkeypatch, tmp_path, capsys):
    ok, fake = run_smoke(monkeypatch, tmp_path, lambda f: None)
    out = capsys.readouterr().out
    assert ok is True
    for good in ("[OK ] amended order keeps the expiry it names", "[OK ] amend up to 2 contracts: remaining=2.00",
                 "[OK ] grown order keeps its client id and takes the new expiry"):
        assert good in out


def test_the_order_test_fails_if_kalshi_refuses_to_grow_an_order(monkeypatch, tmp_path, capsys):
    def patch(fake):
        real = fake.amend_order

        def amend(oid, ticker, side, price_cents, total, **kw):
            if float(total) > float(fake.resting[oid]["remaining_count_fp"]):
                raise KalshiError(400, '{"error":{"code":"invalid_order"}}', "POST", "/amend")
            return real(oid, ticker, side, price_cents, total, **kw)
        fake.amend_order = amend
    ok, fake = run_smoke(monkeypatch, tmp_path, patch)
    assert ok is False and not fake.resting                         # failed, and the order was still cancelled
    assert "[BAD] smoke test: KalshiError" in capsys.readouterr().out


@pytest.mark.parametrize("which", ["down", "up"])
def test_the_order_test_fails_if_an_amend_loses_the_expiry_it_names(monkeypatch, tmp_path, capsys, which):
    def patch(fake):
        real = fake.amend_order

        def amend(oid, ticker, side, price_cents, total, **kw):
            grow = float(total) > float(fake.resting[oid]["remaining_count_fp"])
            a = real(oid, ticker, side, price_cents, total, **kw)
            if grow == (which == "up"):
                fake.resting[oid]["expiration_time"] = None          # Kalshi dropped it: the dead-man switch is off
            return a
        fake.amend_order = amend
    ok, _ = run_smoke(monkeypatch, tmp_path, patch)
    out = capsys.readouterr().out
    assert ok is False
    assert ("[BAD] amended order keeps the expiry it names" if which == "down"
            else "[BAD] grown order keeps its client id and takes the new expiry") in out


# ------------------------------------------------------------------------------ 2. pauses in the log, for the gates
def test_a_pause_is_logged_where_it_starts_and_where_it_ends(tmp_path, monkeypatch, caplog):
    from test_review_round7 import KALSHI_HOURS, ThursdayExchange, et, make_bot
    c = [et(2026, 10, 8, 2, 50)]
    monkeypatch.setattr(time, "time", lambda: c[0])
    fake = ThursdayExchange()
    monkeypatch.setattr(fake, "exchange_schedule", lambda: KALSHI_HOURS, raising=False)
    bot = make_bot(tmp_path, monkeypatch, fake)
    bot.store.kv_set("balance_semantics", {"excludes_resting": True})
    bot.next_due.clear()
    caplog.set_level(logging.INFO, logger="wxbot")
    while c[0] < et(2026, 10, 8, 5, 10):
        bot.step()
        c[0] += 2
    lines = [(r.created, r.getMessage()) for r in caplog.records if r.getMessage().startswith("PAUSE")]
    assert [m.split(":")[0].split(" after")[0] for _, m in lines] == ["PAUSE start", "PAUSE end"]
    assert "Kalshi's published hours" in lines[0][1]
    took = float(lines[1][1].split("after=")[1].split("s")[0])
    assert 2 * 3600 <= took <= 2 * 3600 + 180                      # 2:59 to about 5:00 ET
