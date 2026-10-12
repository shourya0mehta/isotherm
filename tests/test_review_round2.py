"""Regression tests for the second-round findings of the Oct 4 verification review (a fresh reviewer read the fixes
for test_review_findings.py and found these; each reproduction failed before its fix)."""
import datetime as dt
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

import wxbot.app.runner as runner  # noqa: E402
from test_live_loop import FakeKalshi  # noqa: E402
from wxbot.execution.live import LiveOrderManager  # noqa: E402
from wxbot.risk.limits import RiskConfig, event_worst_case, filter_orders  # noqa: E402
from wxbot.strategy.types import DesiredOrder, Position  # noqa: E402
from wxbot.timeutil import et_day  # noqa: E402


def make_bot(tmp_path, monkeypatch, fake):
    k = Ed25519PrivateKey.generate()
    pem = k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                          serialization.NoEncryption()).decode()
    monkeypatch.setenv("KALSHI_KEY_ID", "kid")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY", pem)
    monkeypatch.delenv("KALSHI_PRIVATE_KEY_PATH", raising=False)
    monkeypatch.delenv("WXBOT_NTFY_TOPIC", raising=False)
    cfg = {
        "mode": "live", "env": "prod", "db_path": str(tmp_path / "live.sqlite"),
        "status_path": str(tmp_path / "st.json"),
        "loop": {"tick_seconds": 2, "universe_every_s": 600, "books_every_s": 5, "trades_every_s": 10,
                 "metar_every_s": 300, "nbm_every_s": 900, "index_every_s": 60, "settle_every_s": 900,
                 "positions_every_s": 5},
        "sleeve_a": {"enabled": True, "series": ["KXHIGHNY"], "decided_only": True, "decided_prob": 0.97,
                     "min_edge_c": 2.0, "improve_if_edge_c": 99, "clip_contracts": 12, "decided_max_cost_usd": 10,
                     "min_price_c": 60, "max_price_c": 98},
        "sleeve_b": {"enabled": False, "shadow": False}, "tape_archive": {"enabled": False},
        "risk": {"max_loss_per_bracket_usd": 8, "max_loss_per_event_usd": 10, "max_total_open_risk_usd": 75,
                 "max_cash_deployed_usd": 200, "daily_loss_stop_usd": 15, "total_drawdown_stop_usd": 40,
                 "account_stale_seconds": 30, "max_api_errors_per_min": 1000},
        "execution": {"order_ttl_s": 1200, "ttl_by_sleeve": {"A": 300}}, "dashboard_every_s": 10 ** 9,
    }
    bot = runner.Bot(cfg, base_dir=str(tmp_path))
    bot.rest = fake
    bot.live.rest = fake
    bot.nbm.refresh = lambda *a, **kw: None
    monkeypatch.setattr(runner.obs_mod, "fetch_metars", lambda *a, **kw: {})
    return bot


class Acct(FakeKalshi):
    """The balance leaves out cash held by resting orders; a fill turns the hold into a position at cost."""

    def __init__(self):
        super().__init__()
        self.cash, self.pos, self.orders_down = 625.0, {}, False

    def _hold(self, o):
        px = float(o["yes_price_dollars"])
        return float(o["remaining_count_fp"]) * (px if o["book_side"] == "bid" else 1 - px)

    def balance(self):
        return {"balance": int(round(100 * (self.cash - sum(self._hold(o) for o in self.resting.values())))),
                "portfolio_value": 0}

    def positions(self, **kw):
        return [{"ticker": t, "position_fp": f"{q:.2f}", "market_exposure_dollars": f"{c:.4f}"}
                for t, (q, c) in self.pos.items()], []

    def orders(self, status="resting", **kw):
        if self.orders_down:
            raise ConnectionError("GET /portfolio/orders timed out")
        return super().orders(status, **kw)

    def fill(self, oid):
        o = self.resting.pop(oid)
        cost, n = self._hold(o), float(o["remaining_count_fp"])
        q, c = self.pos.get(o["ticker"], (0.0, 0.0))
        self.pos[o["ticker"]] = (q + (n if o["book_side"] == "bid" else -n), c + cost)
        self.cash -= cost


def equity_setup(tmp_path, monkeypatch, n):
    clock = [time.time()]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    fake = Acct()
    bot = make_bot(tmp_path, monkeypatch, fake)
    bot.store.kv_set("balance_semantics", {"excludes_resting": True})
    close = {m["ticker"]: int(clock[0]) + 86400 for m in fake.markets}
    for k in range(n):
        fake.quotes[fake.markets[k]["ticker"].rsplit("-", 1)[1]] = (1, 4)
    bot.live.reconcile([DesiredOrder(fake.markets[k]["ticker"], "ask", 3, 10.0, "A", 0.0) for k in range(n)], close)
    return clock, fake, bot


def reading(bot, clock):
    clock[0] += 5
    bot.refresh_live_state()


def test_fills_in_consecutive_refreshes_do_not_raise_the_peak(tmp_path, monkeypatch):
    clock, fake, bot = equity_setup(tmp_path, monkeypatch, 2)
    for _ in range(3):
        reading(bot, clock)
    for oid in list(fake.resting):
        fake.fill(oid)
        reading(bot, clock)
    for _ in range(3):
        reading(bot, clock)
    assert abs(bot.risk.peak_equity - 625.0) < 1e-6 and abs(bot.risk.equity - 625.0) < 1e-6


def test_an_orders_outage_does_not_inflate_equity(tmp_path, monkeypatch):
    clock, fake, bot = equity_setup(tmp_path, monkeypatch, 4)
    for _ in range(3):
        reading(bot, clock)
    fake.orders_down = True
    for oid in list(fake.resting):
        fake.fill(oid)
        reading(bot, clock)
    fake.orders_down = False
    for _ in range(4):
        reading(bot, clock)
    assert abs(bot.risk.peak_equity - 625.0) < 1e-6 and not bot.risk.check_stops("x")


def test_a_daily_stop_does_not_carry_into_the_next_day(tmp_path, monkeypatch):
    t0 = dt.datetime(2026, 10, 5, 3, 59, 50, tzinfo=dt.timezone.utc).timestamp()   # 23:59:50 EDT Oct 4
    clock = [t0]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    bot = make_bot(tmp_path, monkeypatch, FakeKalshi())
    day_d = et_day().isoformat()
    bot.store.kv_set("sod_equity", {"day": day_d, "equity": 641.0})
    for _ in range(3):
        bot.refresh_live_state()
    assert round(bot.risk.realized_today, 2) == -16.0 and bot.risk.check_stops(day_d)
    bot.risk_day = day_d
    bot.halt(bot.risk.halt_reason, kind=bot.risk.halt_kind)
    clock[0] = t0 + 12                                   # past midnight; the account refresh not due this step
    bot.next_due["positions"] = clock[0] + 3
    bot.step()
    assert not bot.risk.halted                           # the new day starts clean


def test_no_order_before_the_stops_have_seen_three_readings(tmp_path, monkeypatch):
    clock = [time.time()]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)
    fake = FakeKalshi()
    bot = make_bot(tmp_path, monkeypatch, fake)
    bot.store.kv_set("peak_equity", 700.0)               # $75 below the peak: the $40 stop must fire first
    for _ in range(12):
        bot.step()
        if bot.risk.halted:
            break
        clock[0] += 2.0
    assert bot.risk.halted and bot.risk.halt_kind == "hard" and not fake.created


def test_reads_that_succeed_on_stalled_account_data_are_not_fresh(tmp_path, monkeypatch):
    bot = make_bot(tmp_path, monkeypatch, FakeKalshi())
    now = time.time()
    bot.acct_ts = {k: now for k in ("sync", "balance", "positions")}
    bot.live.last_as_of = now - 5
    assert bot.account_fresh(now)
    bot.live.last_as_of = now - 120                      # Kalshi's account data stopped updating
    assert not bot.account_fresh(now)


def test_our_own_write_resets_the_snapshot_miss_count(monkeypatch):
    clock = [time.time()]
    monkeypatch.setattr(time, "time", lambda: clock[0])

    class Approx(FakeKalshi):
        hide = False

        def orders(self, status="resting", **kw):
            return [] if self.hide else super().orders(status, **kw)
    fake = Approx()
    m = LiveOrderManager(fake, ttl_s=300)
    t = fake.markets[4]["ticker"]
    close = {t: int(clock[0]) + 86400}
    want = [DesiredOrder(t, "ask", 3, 10.0, "A", 0.0)]
    m.reconcile(want, close)
    clock[0] += 200
    fake.hide = True
    m.sync()                                             # one stale read: misses 1
    fake.hide = False
    m.reconcile(want, close)                             # the expiry amend proves the order exists
    assert m.resting_list()[0].misses == 0
    clock[0] += 5
    fake.hide = True
    m.sync()                                             # one more stale read: still tracked
    fake.hide = False
    m.reconcile(want, close)
    assert len(fake.resting) == 1


def test_a_failed_shrink_never_leaves_uncounted_exposure():
    class NoAmend(FakeKalshi):
        def amend_order(self, *a, **kw):
            raise ConnectionError("read timed out")
    fake = NoAmend()
    m = LiveOrderManager(fake, ttl_s=1200)
    ev = fake.ev
    t0, t1 = fake.markets[0]["ticker"], fake.markets[4]["ticker"]
    close = {t1: int(time.time()) + 86400}
    d1 = DesiredOrder(t1, "ask", 3, 10.0, "A", 0.0, event=ev)
    m.reconcile([d1], close)                             # 10 NO at 97c rest ($9.70)
    cfg = RiskConfig(max_loss_per_bracket_usd=10, max_loss_per_event_usd=10, max_total_open_risk_usd=75)
    pos = {ev: [Position(t0, ev, yes_qty=10, yes_cost=5.0)]}   # a $5 YES fill elsewhere in the event
    tick = {ev: [x["ticker"] for x in fake.markets]}
    for _ in range(3):
        base = {}
        for o in m.committed([d1]):
            base.setdefault(o.event, []).append(o)
        kept, _ = filter_orders([d1], pos, tick, cfg, 0.0, base_pending=base, incumbents={(t1, "ask", 3)})
        m.reconcile(kept, close)                         # the amend down fails: the order is cancelled instead
        true = event_worst_case(pos[ev], [DesiredOrder(x["ticker"], x["book_side"],
                                                       int(round(100 * float(x["yes_price_dollars"]))),
                                                       float(x["remaining_count_fp"]), "A", 0.0)
                                          for x in fake.resting.values()], tick[ev])
        assert true <= 10.0 + 1e-9


def test_an_immediate_or_cancel_take_is_never_resent():
    from wxbot.execution.taker import TakerExecutor
    seen = {}

    class R:
        def create_order(self, *a, **kw):
            seen.update(kw)
            return {"order_id": "x", "fill_count": "0.00"}
    TakerExecutor(R(), "live").take("KXHIGHNY-26OCT05-B72.5", "yes", 0.40, 5, client_order_id="wxb-X-tk1")
    assert seen["retries"] == 0 and seen["tif"] == "immediate_or_cancel"


def test_a_locally_refused_order_leaves_no_phantom():
    class Bad(FakeKalshi):
        def create_order(self, *a, **kw):
            raise ValueError("price out of range: 0")
    fake = Bad()
    m = LiveOrderManager(fake, ttl_s=1200)
    t, close = fake.markets[4]["ticker"], {fake.markets[4]["ticker"]: int(time.time()) + 3600}
    m.reconcile([DesiredOrder(t, "ask", 3, 10.0, "A", 0.0)], close)
    assert m.resting_list() == [] and m.errors_last_minute() == 1
