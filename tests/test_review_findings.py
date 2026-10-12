"""Reproductions of the Oct 3 outside-review findings (docs/REVIEW_RESPONSE_2026-10-04.md). Each test was written
against commit 95df014 first and failed there for the reason named in its docstring; the fixes make them pass."""
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_live_loop import FakeKalshi  # noqa: E402
from test_live_safety import StubbornKalshi, bare_bot  # noqa: E402

from wxbot.execution.live import LiveOrderManager  # noqa: E402
from wxbot.risk.limits import RiskConfig, ThresholdLadder, event_worst_case  # noqa: E402
from wxbot.strategy import sleeve_h  # noqa: E402
from wxbot.models.fair_value import LadderQuote as Quote  # noqa: E402
from wxbot.strategy.types import DesiredOrder, EventBook, Position  # noqa: E402


def close_of(fake, k=4):
    t = fake.markets[k]["ticker"]
    return t, {t: int(time.time()) + 3600}


# ---------------------------------------------------------------- 2.1 failed cancels
def test_a_cancel_that_fails_keeps_the_order_and_blocks_its_replacement():
    """2.1: reconcile dropped an order from its books before the exchange confirmed the cancel, then placed the
    replacement: a failed cancel left two orders resting where one was wanted."""
    fake = StubbornKalshi(stuck_rounds=0, never={"o1"})
    m = LiveOrderManager(fake, ttl_s=1200)
    t, close = close_of(fake)
    m.reconcile([DesiredOrder(t, "ask", 3, 10.0, "A", 0.0)], close)              # o1: buy NO at 97c
    m.reconcile([DesiredOrder(t, "ask", 4, 10.0, "A", 0.0)], close)              # move to 96c: the cancel fails
    assert len(fake.resting) == 1                                                # no second order on top
    assert [o.order_id for o in m.resting_list()] == ["o1"]                      # still counted as exposure
    fake.never.clear()                                                           # the exchange recovers
    for o in m.resting_list():
        o.cancel_ts -= 5                                                         # next cycle: retry is due
    m.reconcile([DesiredOrder(t, "ask", 4, 10.0, "A", 0.0)], close)              # cancel retried, then replaced
    assert [o["yes_price_dollars"] for o in fake.resting.values()] == ["0.0400"]


def test_a_batch_cancel_that_raises_keeps_every_order_in_it():
    class Down(FakeKalshi):
        fail = True

        def batch_cancel(self, ids, **kw):
            if self.fail:
                raise ConnectionError("reset by peer")
            return super().batch_cancel(ids, **kw)
    fake = Down()
    m = LiveOrderManager(fake, ttl_s=1200)
    t, close = close_of(fake)
    m.reconcile([DesiredOrder(t, "ask", 3, 10.0, "A", 0.0)], close)
    m.reconcile([], close)                                                       # cancel everything: it raises
    assert len(m.resting_list()) == 1 and len(fake.resting) == 1
    m.reconcile([DesiredOrder(t, "ask", 3, 10.0, "A", 0.0)], close)              # wanted again: not stacked
    assert len(fake.created) == 1


# ---------------------------------------------------------------- 2.2 lost placements and lagged snapshots
class LostAnswerKalshi(FakeKalshi):
    """The order reaches the book but the answer is lost (read timeout)."""

    def __init__(self, lose=1, reach=True):
        super().__init__()
        self.lose, self.reach = lose, reach

    def create_order(self, *a, **kw):
        if self.lose and not self.reach:
            self.lose -= 1
            raise ConnectionError("connect timeout")
        r = super().create_order(*a, **kw)
        if self.lose:
            self.lose -= 1
            raise ConnectionError("read timed out")
        return r


def test_a_lost_order_answer_does_not_lead_to_a_duplicate():
    """2.2: a placement whose answer was lost was forgotten, so the next cycle placed the order again."""
    fake = LostAnswerKalshi()
    m = LiveOrderManager(fake, ttl_s=1200)
    t, close = close_of(fake)
    d = DesiredOrder(t, "ask", 3, 10.0, "A", 0.0)
    m.reconcile([d], close)
    m.reconcile([d], close)
    assert len(fake.created) == 1                                                # no duplicate before a sync
    m.sync()                                                                     # the snapshot shows it: adopted
    m.reconcile([d], close)
    assert len(fake.created) == 1 and len(fake.resting) == 1
    assert [o.order_id for o in m.resting_list()] == ["o1"]


def test_an_order_that_never_reached_the_exchange_is_retried_after_a_newer_snapshot():
    fake = LostAnswerKalshi(reach=False)
    m = LiveOrderManager(fake, ttl_s=1200)
    t, close = close_of(fake)
    d = DesiredOrder(t, "ask", 3, 10.0, "A", 0.0)
    m.reconcile([d], close)                                                      # nothing reached the book
    assert not fake.resting
    m.reconcile([d], close)
    assert len(fake.created) == 0                                                # blocked: the outcome is unknown
    for o in m.resting_list():
        o.ack_ts -= 10                                                           # later snapshots without it
    m.sync()
    m.reconcile([d], close)
    assert len(fake.created) == 0                                                # one snapshot is not enough
    m.sync()
    m.reconcile([d], close)
    assert len(fake.resting) == 1


class LaggedKalshi(FakeKalshi):
    """Account reads lag the matching engine: orders() returns a frozen snapshot and user_data_timestamp() says how
    old it is."""

    def __init__(self):
        super().__init__()
        self.frozen = None
        self.as_of = None

    def freeze(self):
        self.frozen = [dict(o) for o in self.resting.values()]
        self.as_of = time.time() - 1.0

    def orders(self, status="resting", **kw):
        return [dict(o) for o in self.frozen] if self.frozen is not None else super().orders(status, **kw)

    def user_data_timestamp(self):
        return self.as_of


def test_a_lagged_snapshot_does_not_erase_an_order_placed_after_it():
    """2.2: sync() replaced the local order book with the REST snapshot, which can predate an order just placed; the
    next cycle then placed a duplicate."""
    fake = LaggedKalshi()
    m = LiveOrderManager(fake, ttl_s=1200)
    t, close = close_of(fake)
    fake.freeze()                                                                # snapshot taken before the order
    d = DesiredOrder(t, "ask", 3, 10.0, "A", 0.0)
    m.reconcile([d], close)
    m.sync()
    m.reconcile([d], close)
    assert len(fake.created) == 1 and len(fake.resting) == 1


def test_a_lagged_snapshot_does_not_bring_back_a_confirmed_cancel():
    fake = LaggedKalshi()
    m = LiveOrderManager(fake, ttl_s=1200)
    t, close = close_of(fake)
    m.reconcile([DesiredOrder(t, "ask", 3, 10.0, "A", 0.0)], close)
    fake.freeze()                                                                # snapshot still shows o1
    fake.as_of = time.time() - 0.5
    m.reconcile([], close)                                                       # o1 cancelled and confirmed
    m.sync()                                                                     # stale snapshot: o1 'resting'
    assert m.resting_list() == []


def test_create_order_asks_kalshi_to_cancel_the_incoming_order_on_a_self_cross():
    """2.13: self-trade prevention 'maker' cancels our own resting order when a taker order of ours crosses it."""
    from wxbot.kalshi.rest import KalshiREST
    seen = {}

    class R(KalshiREST):
        def _request(self, method, path, params=None, body=None, **kw):
            seen.update(body or {})
            return {"order_id": "x"}
    R(auth=None).create_order("KXHIGHNY-26OCT05-T75", "bid", 3, 1)
    assert seen["self_trade_prevention_type"] == "taker_at_cross"


# ---------------------------------------------------------------- 2.3 pending orders offsetting each other
def test_pending_orders_do_not_offset_each_other():
    """2.3: resting orders were assumed to fill together, so the gain of one offset the loss of another. Adverse
    selection fills exactly the losing ones."""
    tick = ["E-A", "E-B", "E-C"]
    pend = [DesiredOrder("E-A", "ask", 40, 20, "A", 0.0, event="E"),            # NO at 60c on A and B
            DesiredOrder("E-B", "ask", 40, 20, "A", 0.0, event="E")]
    assert abs(event_worst_case([], pend, tick) - 12.0) < 1e-9                   # all-fill said 4.00
    held = [Position("E-B", "E", no_qty=20, no_cost=12.0)]                       # filled: offsets exactly
    assert abs(event_worst_case(held, pend[:1], tick) - 12.0) < 1e-9             # all-fill said 4.00
    assert abs(event_worst_case(held, [], tick) - 12.0) < 1e-9


def hourly_event(quotes: dict, close_in=1800.0):
    strikes = sorted(quotes)
    ms = [{"ticker": f"KXTEMPMIAH-26OCT05H14-T{k}", "strike_type": "greater", "floor_strike": k} for k in strikes]
    q = {f"KXTEMPMIAH-26OCT05H14-T{k}": Quote(*quotes[k]) for k in strikes}
    return EventBook("KXTEMPMIAH-26OCT05H14", "KXTEMPMIAH", "hourly", "IDX", None, time.time() + close_in, ms, [],
                     q, {})


def adverse_worst(ev, orders, held=()):
    strikes = sorted(float(m["floor_strike"]) for m in ev.markets)
    k_of = {m["ticker"]: float(m["floor_strike"]) for m in ev.markets}
    L = [0.0] * (len(strikes) + 1)
    for t, side, px, n in held:
        for j, l in enumerate(sleeve_h.scenario_losses(strikes, side, px, k_of[t])):
            L[j] += n * l
    for o in orders:
        for j, l in enumerate(sleeve_h.scenario_losses(strikes, o.side, o.price_cents, k_of[o.ticker])):
            L[j] += o.count * max(l, 0.0)
    return max(L)


def test_hourly_quotes_stay_inside_the_cap_when_only_the_losing_ones_fill():
    """2.3: Sleeve H credited each new quote's gains in the other scenarios to the room left for later quotes."""
    q = {k: (95, 97) for k in (80, 81, 82)}                                      # near-certain YES: bid 96
    q.update({k: (50, 52) for k in (83, 84, 85, 86)})
    q.update({k: (3, 5) for k in (87, 88, 89)})                                  # near-certain NO: offer YES at 4
    ev = hourly_event(q)
    cfg = sleeve_h.SleeveHConfig(enabled=True, size=20, cap_usd=25.0)
    out = sleeve_h.propose(ev, time.time(), cfg, [])
    assert out and adverse_worst(ev, out) <= cfg.cap_usd + 1e-9


# ---------------------------------------------------------------- 2.4 taker outside the shared caps
def risk_bot(tmp_path, fake=None):
    from wxbot.strategy.sleeve_h import SleeveHConfig
    bot = bare_bot(tmp_path, fake or FakeKalshi())
    bot.risk_cfg = RiskConfig(max_loss_per_bracket_usd=15, max_loss_per_event_usd=20, max_total_open_risk_usd=200,
                              max_cash_deployed_usd=400, min_cash_reserve_usd=25)
    bot.risk.cfg = bot.risk_cfg
    bot.events, bot.hourly, bot.ev_tickers, bot.ev_tickers_miss = {}, {}, {}, {}
    bot.netting, bot.series_t, bot.cfg_h = False, set(), SleeveHConfig()
    bot.live_equity, bot.avail_balance = 625.0, 600.0
    bot.max_err, bot.stale_book_s = 15, 90.0
    bot.next_due, bot.paper, bot.h_on = defaultdict(float), None, False
    return bot


def test_the_taker_respects_the_shared_event_cap(tmp_path):
    """2.4: Sleeve X checked only its own size and daily spend, not the per-event cap the makers share."""
    fake = FakeKalshi()
    bot = risk_bot(tmp_path, fake)
    ev = fake.ev
    tick = [m["ticker"] for m in fake.markets]
    bot.ev_tickers[ev] = tick
    bot.positions = {tick[2]: Position(tick[2], ev, yes_qty=40, yes_cost=19.0)}  # worst case $19 of the $20 cap
    bot.acct_ts = {k: time.time() for k in ("sync", "balance", "positions")}
    bot.eq_readings = [625.0] * 3                 # the stops have checked three readings
    bot.execute_take("X", ev, tick[3], "yes", 0.40, 5, {"trade": True, "trade_size": 5, "trade_max_usd_day": 25})
    sent = [c for c in fake.created if c[0] == tick[3]]
    assert len(sent) == 1 and sent[0][3] == 2                                    # $19 + 2 x ($0.40 + fee) <= $20
    bot.positions[tick[2]].yes_cost = 19.9                                       # no room left: nothing is sent
    bot.execute_take("X", ev, tick[3], "yes", 0.40, 5, {"trade": True, "trade_size": 5, "trade_max_usd_day": 25})
    assert len([c for c in fake.created if c[0] == tick[3]]) == 1


# ---------------------------------------------------------------- 2.5 hourly ladder structure after a restart
def test_a_closed_hourly_event_is_still_valued_as_a_nested_ladder(tmp_path):
    """2.5: after a restart the hourly ladder cache was empty and a closed hour's positions were valued as exclusive
    brackets, which halves the worst case of NO positions on nested strikes."""
    fake = FakeKalshi()
    ev = "KXTEMPMIAH-26OCT05H14"
    ms = [{"ticker": f"{ev}-T{k}", "strike_type": "greater", "floor_strike": k, "event_ticker": ev}
          for k in (88, 89)]
    fake.markets = lambda **kw: ms
    bot = risk_bot(tmp_path, fake)
    bot.positions = {f"{ev}-T88": Position(f"{ev}-T88", ev, no_qty=1, no_cost=0.95),
                     f"{ev}-T89": Position(f"{ev}-T89", ev, no_qty=1, no_cost=0.95)}
    bot.execute([])
    assert abs(bot.open_risk - 1.90) < 1e-9                                      # both lose above 89


def test_an_event_whose_structure_is_unknown_counts_its_full_cost(tmp_path):
    fake = FakeKalshi()

    def down(**kw):
        raise ConnectionError("timeout")
    fake.markets = down
    bot = risk_bot(tmp_path, fake)
    ev = "KXTEMPMIAH-26OCT05H14"
    bot.positions = {f"{ev}-T88": Position(f"{ev}-T88", ev, no_qty=1, no_cost=0.95),
                     f"{ev}-T89": Position(f"{ev}-T89", ev, no_qty=1, no_cost=0.95)}
    bot.execute([])
    assert abs(bot.open_risk - 1.90) < 1e-9                                      # an empty ladder said 0


# ---------------------------------------------------------------- 2.6 halts that do not stick
def test_a_pause_does_not_downgrade_a_hard_halt(tmp_path):
    """2.6: an API-error pause after a hard halt replaced the halt kind, so the bot resumed after 15 minutes."""
    bot = risk_bot(tmp_path)
    bot.alerts = type("A", (), {"send": lambda *a, **kw: None})()
    bot.halt("drawdown stop hit", kind="hard")
    bot.halt("API error burst: pausing 15 min", kind="pause", pause_s=900)
    assert bot.risk.halt_kind == "hard"
    assert not bot.risk.maybe_resume(time.time() + 3600, "2099-01-01") and bot.risk.halted


def test_a_hard_halt_survives_a_restart(tmp_path):
    """2.6: the halt lived only in memory: a restart (crash, launchd, deploy) resumed trading."""
    bot = risk_bot(tmp_path)
    bot.alerts = type("A", (), {"send": lambda *a, **kw: None})()
    bot.halt("drawdown stop hit", kind="hard")
    bot2 = risk_bot(tmp_path)                                                    # same database, new process
    bot2.restore_halt()
    assert bot2.risk.halted and bot2.risk.halt_kind == "hard"


# ---------------------------------------------------------------- 2.7 stale data treated as fresh
def test_a_market_missing_from_the_book_refresh_goes_stale(tmp_path):
    """2.7: a market missing from a book refresh kept its old quote while the global book clock moved on."""
    fake = FakeKalshi()
    bot = risk_bot(tmp_path, fake)
    bot.feed, bot.paper, bot.h_on = None, None, False
    ms = fake.markets
    ev = EventBook(fake.ev, "KXHIGHNY", "high", "KNYC", None, time.time() + 3600, ms, [], {}, {})
    bot.events = {fake.ev: ev}
    bot.refresh_books()
    assert bot.book_fresh(ev, time.time(), 90)
    t_missing = ms[0]["ticker"]
    old = fake.orderbooks
    fake.orderbooks = lambda tickers: {k: v for k, v in old(tickers).items() if k != t_missing}
    bot.book_ts[t_missing] -= 120                                                # its last real book: 2 min ago
    bot.refresh_books()                                                          # the others refresh, it does not
    assert all(time.time() - bot.book_ts[m["ticker"]] < 5 for m in ms[1:])
    assert not bot.book_fresh(ev, time.time(), 90)                               # so the event is not quoted


def test_no_new_orders_while_the_account_state_is_unknown(tmp_path):
    """2.7: a failed balance read left avail_balance None, and None allowed new orders."""
    class NoBalance(FakeKalshi):
        def balance(self):
            raise ConnectionError("timeout")
    fake = NoBalance()
    bot = risk_bot(tmp_path, fake)
    bot.avail_balance = None
    bot.refresh_live_state()
    assert not bot.account_fresh(time.time())
