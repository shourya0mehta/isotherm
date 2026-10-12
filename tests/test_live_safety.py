"""Live-money safety details found in the Oct 3 API audit: cancels are confirmed, orders are never sent for a market
that is closing, manual trades in the same account are not counted as the bot's, equity counts the cash held by
resting orders exactly when Kalshi's available balance leaves it out (measured by the smoke test), and a taker order
whose answer was lost still counts against the daily spend cap."""
import datetime as dt
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_live_loop import FakeKalshi, iso  # noqa: E402

from wxbot.execution.live import LiveOrderManager  # noqa: E402
from wxbot.strategy.types import DesiredOrder  # noqa: E402


class StubbornKalshi(FakeKalshi):
    """Batch cancels answer 200 but leave some orders resting (per-order errors), like the real endpoint can."""

    def __init__(self, stuck_rounds=1, never=()):
        super().__init__()
        self.stuck_rounds, self.never, self.rounds = stuck_rounds, set(never), 0

    def batch_cancel(self, ids, **kw):
        self.rounds += 1
        out = []
        for i in ids:
            if i in self.never or (self.rounds <= self.stuck_rounds and i == ids[0]):
                out.append({"order_id": i, "error": {"code": "internal", "message": "try again"}})
            else:
                self.cancel_order(i)
                out.append({"order_id": i, "reduced_by": "1.00"})
        return {"orders": out}


def place(fake, n=3):
    m = LiveOrderManager(fake, ttl_s=1200)
    for k in range(n):
        d = DesiredOrder(fake.markets[k]["ticker"], "ask", 99, 5.0, "A", 0.0)   # NO at 1c: never crosses
        m._place(d, 5.0, int(time.time()) + 3600)
    return m


def test_cancel_all_rechecks_the_exchange_and_retries(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    fake = StubbornKalshi(stuck_rounds=1)
    m = place(fake)
    assert len(fake.resting) == 3
    assert m.cancel_all("test") == 0 and not fake.resting and fake.rounds == 2


def test_cancel_all_reports_orders_it_could_not_cancel(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    fake = StubbornKalshi(stuck_rounds=0, never={"o2"})
    m = place(fake)
    assert m.cancel_all("test") == 1 and list(fake.resting) == ["o2"]


def test_no_order_for_a_market_that_is_closing():
    fake = FakeKalshi()
    m = LiveOrderManager(fake, ttl_s=1200)
    d = DesiredOrder(fake.markets[0]["ticker"], "ask", 3, 5.0, "A", 0.0)
    assert m._place(d, 5.0, int(time.time()) - 10) is None and not fake.created
    assert m._place(d, 5.0, int(time.time()) + 600) is not None and len(fake.created) == 1


class FillsKalshi(FakeKalshi):
    def __init__(self, fills, owners):
        super().__init__()
        self._fills, self.owners, self.lookups = fills, owners, []

    def fills(self, min_ts=None, **kw):
        return self._fills

    def order(self, oid):
        self.lookups.append(oid)
        if self.owners[oid] is None:
            raise RuntimeError("timeout")
        return {"order_id": oid, "client_order_id": self.owners[oid]}


def fill(fid, oid, t0):
    return {"fill_id": fid, "order_id": oid, "market_ticker": "KXHIGHNY-26OCT05-T75", "book_side": "ask",
            "yes_price_dollars": "0.0300", "count_fp": "5.00", "is_taker": False, "fee_cost": "0.0000",
            "created_time": iso(t0), "ts": int(t0) + 999}


def test_manual_fills_in_the_same_account_are_skipped():
    t0 = time.time() - 30
    fake = FillsKalshi([fill("f1", "o-ours", t0), fill("f2", "o-manual", t0), fill("f3", "o-unknown", t0),
                        fill("f4", "o-bot-old", t0)],
                       {"o-manual": "my-own-click", "o-unknown": None, "o-bot-old": "wxb-1234"})
    m = LiveOrderManager(fake)
    m.my_order_ids.add("o-ours")
    got = m.poll_fills()
    assert [f.order_ref for f in got] == ["f1", "f4"]         # a failed lookup is not marked seen: it waits
    assert abs(got[0].ts - t0) < 1.0                           # created_time, not the legacy ts field
    assert sorted(fake.lookups) == ["o-bot-old", "o-manual", "o-unknown"]
    fake._fills = [fill("f5", "o-manual", t0)]                 # f3 has left the fill window meanwhile
    assert m.poll_fills() == [] and fake.lookups.count("o-manual") == 1   # the verdict is cached
    fake.owners["o-unknown"] = "my-own-click"                  # the lookup works again: f3 was a manual trade
    assert m.poll_fills() == [] and "f3" in m.seen_fill_ids
    m2 = LiveOrderManager(fake)
    fake.owners["o-unknown"], fake._fills = None, [fill("f6", "o-unknown", t0)]
    assert m2.poll_fills() == []
    m2.unresolved_since["f6"] -= 601                           # lookups failing for 10 minutes: counted as ours
    assert [f.order_ref for f in m2.poll_fills()] == ["f6"]


def bare_bot(tmp_path, fake):
    from wxbot.app.runner import Bot
    from wxbot.risk.limits import RiskConfig, RiskState
    from wxbot.store.db import Store
    bot = Bot.__new__(Bot)
    bot.store = Store(str(tmp_path / "live.sqlite"))
    bot.rest = fake
    bot.live = LiveOrderManager(fake)
    bot.risk = RiskState(RiskConfig())
    bot.fair_cache, bot.counters, bot.positions = {}, defaultdict(int), {}
    bot.mode = "live"
    return bot


def test_equity_adds_resting_cash_only_when_the_balance_leaves_it_out(tmp_path):
    fake = FakeKalshi()
    exp = int(time.time()) + 600
    fake.create_order(fake.markets[0]["ticker"], "bid", 90, 10, client_order_id="wxb-a", expiration_ts=exp)  # $9.00
    fake.create_order(fake.markets[4]["ticker"], "ask", 3, 10, client_order_id="wxb-b", expiration_ts=exp)   # $9.70
    fake.create_order(fake.markets[2]["ticker"], "bid", 50, 10, client_order_id="manual", expiration_ts=exp)  # $5.00
    bot = bare_bot(tmp_path, fake)
    bot.refresh_live_state()                       # not measured yet: assume the balance leaves it out (and warn)
    # the balance holds cash for every resting order, manual ones too, so equity adds all of it back (from the
    # orders snapshot just read, not from the bot's own order book)
    assert abs(bot.live_equity - (625.0 + 23.70)) < 1e-9
    assert bot.store.c.execute("SELECT COUNT(*) FROM events_log WHERE msg LIKE 'balance semantics%'").fetchone()[0] == 1
    bot.store.kv_set("balance_semantics", {"excludes_resting": False})
    bot.refresh_live_state()
    assert bot.live_equity == 625.0
    bot.store.kv_set("balance_semantics", {"excludes_resting": True})
    bot.refresh_live_state()
    assert abs(bot.live_equity - 648.70) < 1e-9 and abs(bot.resting_held_usd - 23.70) < 1e-9


class LedgerKalshi(FakeKalshi):
    """Balance as Kalshi might report it: either cash held by resting orders comes out of it, or not."""

    def __init__(self, excludes):
        super().__init__()
        self.excludes = excludes

    def balance(self):
        held = sum(float(o["remaining_count_fp"]) * float(o["yes_price_dollars"]) for o in self.resting.values()
                   if o["book_side"] == "bid")
        return {"balance": 62500 - (int(round(100 * held)) if self.excludes else 0), "portfolio_value": 0}

    def amend_order(self, oid, ticker, side, price, total, expiration_ts=None, **kw):
        self.resting[oid]["remaining_count_fp"] = f"{total:.2f}"
        return super().amend_order(oid, ticker, side, price, total, expiration_ts, **kw)


def run_smoke_with(tmp_path, monkeypatch, fake):
    import wxbot.app.smoke as smoke
    monkeypatch.setattr(smoke.KalshiAuth, "from_env", staticmethod(lambda: object()))
    monkeypatch.setattr(smoke, "KalshiREST", lambda **kw: fake)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    cfg = {"env": "prod", "db_path": str(tmp_path / "live.sqlite"), "sleeve_a": {"series": ["KXHIGHNY"]}}
    ok = smoke.run_smoke(cfg)
    from wxbot.store.db import Store
    return ok, Store(cfg["db_path"]).kv_get("balance_semantics")


def test_smoke_test_measures_and_stores_the_balance_semantics(tmp_path, monkeypatch):
    ok, sem = run_smoke_with(tmp_path, monkeypatch, LedgerKalshi(excludes=True))
    assert ok and sem["excludes_resting"] is True and sem["cents"] == [62500, 62498, 62500]
    ok, sem = run_smoke_with(tmp_path, monkeypatch, LedgerKalshi(excludes=False))
    assert ok and sem["excludes_resting"] is False


def test_a_taker_order_with_no_answer_still_counts_against_the_daily_cap(tmp_path):
    class Down(FakeKalshi):
        def create_order(self, *a, **kw):
            raise ConnectionError("reset by peer")
    from wxbot.risk.limits import RiskConfig
    bot = bare_bot(tmp_path, Down())
    bot.risk_cfg = RiskConfig()
    bot.events, bot.hourly, bot.ev_tickers, bot.ev_tickers_miss, bot.netting = {}, {}, {}, {}, False
    bot.ev_tickers["KXHIGHNY-26OCT05"] = ["KXHIGHNY-26OCT05-B72.5", "KXHIGHNY-26OCT05-T75"]
    bot.acct_ts = {k: time.time() for k in ("sync", "balance", "positions")}
    bot.eq_readings = [625.0] * 3                 # the stops have checked three readings
    out = bot.execute_take("X", "KXHIGHNY-26OCT05", "KXHIGHNY-26OCT05-B72.5", "no", 0.40, 5,
                           {"trade": True, "trade_size": 5, "trade_max_usd_day": 25})
    assert out["status"] == "error"
    day = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=4)).date().isoformat()   # budget day flips at 20Z
    assert abs(bot.store.kv_get(f"take_spent:X:{day}") - 2.09) < 1e-9   # 5 x 40c + the 9c taker fee
    bot.acct_ts["balance"] = time.time() - 120                          # account reads stale: nothing is sent
    assert bot.execute_take("X", "KXHIGHNY-26OCT05", "KXHIGHNY-26OCT05-B72.5", "no", 0.40, 5,
                            {"trade": True, "trade_size": 5, "trade_max_usd_day": 25}) is None


def test_patience_never_delays_a_cautious_move_and_limits_chasing(monkeypatch):
    fake = FakeKalshi()
    t = fake.markets[4]["ticker"]
    m = LiveOrderManager(fake, ttl_s=1200, patient_s=300)
    close = {t: int(time.time()) + 3600}
    first = DesiredOrder(t, "ask", 3, 10.0, "A", 0.0)                 # buy NO at 97c
    assert m.reconcile([first], close)["placed"] == 1
    chase = DesiredOrder(t, "ask", 2, 10.0, "A", 0.0)                 # NO at 98c: more aggressive
    s = m.reconcile([chase], close)
    assert s["placed"] == 0 and s["cancelled"] == 0 and s["kept"] == 1 and len(fake.resting) == 1
    careful = DesiredOrder(t, "ask", 4, 10.0, "A", 0.0)               # NO at 96c: more cautious, at once
    s = m.reconcile([careful], close)
    assert s["cancelled"] == 1 and s["placed"] == 1
    assert [o["yes_price_dollars"] for o in fake.resting.values()] == ["0.0400"]
    # once the order is older than the patience window it may chase
    for lst in m.orders.values():
        for o in lst:
            o.created_ts -= 301
    s = m.reconcile([chase], close)
    assert s["cancelled"] == 1 and s["placed"] == 1
    # the hourly sleeve and a zero window are never patient
    t0 = fake.markets[0]["ticker"]
    h = DesiredOrder(t0, "ask", 2, 10.0, "H", 0.0)
    assert m.reconcile([h], {t0: int(time.time()) + 3600})["placed"] == 1
    m0 = LiveOrderManager(FakeKalshi(), ttl_s=1200, patient_s=0)
    m0.reconcile([first], close)
    assert m0.reconcile([chase], close)["placed"] == 1


def test_paper_patience_matches_live(tmp_path):
    from wxbot.execution.paper import PaperBroker
    from wxbot.strategy.types import EventBook
    bot = bare_bot(tmp_path, FakeKalshi())
    bot.mode, bot.paper, bot.events, bot.hourly = "paper", PaperBroker(), {}, {}
    bot.patient_s, bot.patient_sleeves = 300.0, ("A", "T")
    t = "KXHIGHNY-26OCT05-T75"
    now = time.time()
    bot.paper_reconcile([DesiredOrder(t, "ask", 3, 10.0, "A", 0.0)], now)
    bot.paper_reconcile([DesiredOrder(t, "ask", 2, 10.0, "A", 0.0)], now + 60)     # chase: held
    assert [o.price_cents for o in bot.paper.resting()] == [3]
    bot.paper_reconcile([DesiredOrder(t, "ask", 4, 10.0, "A", 0.0)], now + 70)     # cautious: at once
    assert [o.price_cents for o in bot.paper.resting()] == [4]
    bot.paper_reconcile([DesiredOrder(t, "ask", 2, 10.0, "A", 0.0)], now + 400)    # old enough to chase
    assert [o.price_cents for o in bot.paper.resting()] == [2]
