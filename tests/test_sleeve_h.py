"""Sleeve H (hourly Weather Index tail maker): quoting rule, worst-case cap over outcome scenarios, and the runner's
shadow loop (separate paper broker, fills from the public tape, settlement from market results)."""
import datetime as dt
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wxbot.models.fair_value import LadderQuote
from wxbot.strategy import sleeve_h as H
from wxbot.strategy.types import EventBook


def ladder(close_in_s=1800):
    ms = [{"ticker": f"KXTEMPMIAH-26OCT0312-T{k}.99", "strike_type": "greater", "floor_strike": k + 0.99,
           "status": "active"} for k in (83, 84, 85, 86, 87)]
    ev = EventBook("KXTEMPMIAH-26OCT0312", "KXTEMPMIAH", "hourly", "KXTEMPMIAH-IDX", None, time.time() + close_in_s,
                   ms, [], {}, {})
    q = {83: (96, 99), 84: (91, 94), 85: (40, 48), 86: (4, 9), 87: (None, 3)}
    ev.quotes = {f"KXTEMPMIAH-26OCT0312-T{k}.99": LadderQuote(b, a, 10.0, 10.0) for k, (b, a) in q.items()}
    return ev


def worst(ev, orders, held=()):
    k_of = {m["ticker"]: H.strike_of(m) for m in ev.markets}
    strikes = sorted(k_of.values())
    L = [0.0] * (len(strikes) + 1)
    for t, side, px, n in list(held) + [(o.ticker, o.side, o.price_cents, o.count) for o in orders]:
        for j, l in enumerate(H.scenario_losses(strikes, side, px, k_of[t])):
            L[j] += n * l
    return max(L)


def test_quotes_only_tails_inside_the_touch_and_never_cross():
    ev = ladder()
    out = H.propose(ev, time.time(), H.SleeveHConfig())
    got = {(o.ticker.rsplit("-", 1)[1], o.side): o.price_cents for o in out}
    assert ("T83.99", "bid") in got and got[("T83.99", "bid")] == 97        # 96/99 -> bid 97
    assert ("T87.99", "ask") in got and got[("T87.99", "ask")] == 2         # -/3 -> ask 2
    assert ("T85.99", "bid") not in got and ("T85.99", "ask") not in got     # 40/48 is not a tail
    for o in out:
        q = ev.quotes[o.ticker]
        if o.side == "bid":
            assert q.yes_ask is None or o.price_cents < q.yes_ask
        else:
            assert q.yes_bid is None or o.price_cents > q.yes_bid
        assert o.sleeve == "H" and o.event == ev.event


def test_cap_is_the_worst_outcome_scenario_not_the_sum():
    ev = ladder()
    cfg = H.SleeveHConfig(cap_usd=25.0, size=20)
    out = H.propose(ev, time.time(), cfg)
    assert worst(ev, out) <= 25.0 + 1e-9
    # both sides quoted: a bid on a favorite and an ask on a longshot cannot both lose
    assert {o.side for o in out} == {"bid", "ask"}
    held = [("KXTEMPMIAH-26OCT0312-T83.99", "bid", 97, 20.0)]
    out2 = H.propose(ev, time.time(), cfg, held)
    assert worst(ev, out2, held) <= 25.0 + 1e-9
    assert sum(o.count for o in out2 if o.side == "bid") < sum(o.count for o in out if o.side == "bid")


def test_no_quotes_in_the_last_minutes_or_after_close():
    ev = ladder(close_in_s=50)
    assert H.propose(ev, time.time(), H.SleeveHConfig(stop_min=1.0)) == []
    assert H.propose(ladder(close_in_s=-5), time.time(), H.SleeveHConfig()) == []


def test_runner_shadow_fills_from_tape_and_settles(tmp_path):
    from wxbot.app.runner import Bot
    from wxbot.execution.paper import PaperBroker
    from wxbot.store.db import Store

    class FakeRest:
        def market(self, t):
            return {"result": "yes" if t.endswith("T83.99") else "no"}

    bot = Bot.__new__(Bot)
    bot.store = Store(str(tmp_path / "h.db"))
    bot.cfg_h = H.SleeveHConfig(size=10, cap_usd=50.0)
    bot.h_on = True
    bot.paper_h = PaperBroker()
    bot.paper_h2 = PaperBroker()
    bot.h2_held = defaultdict(list)
    bot.h2_on = False
    bot.paper = None
    bot.paper_b = PaperBroker()
    bot.positions_b = {}
    bot.h_held = defaultdict(list)
    bot.counters = defaultdict(int)
    bot.stale_book_s = 90
    bot.rest = FakeRest()
    ev = ladder()
    bot.hourly = {ev.event: ev}
    bot.book_ts = {m["ticker"]: time.time() for m in ev.markets}
    now = time.time()
    bot.shadow_h_orders(now)
    resting = {(o.ticker.rsplit("-", 1)[1], o.side): o for o in bot.paper_h.resting()}
    assert ("T83.99", "bid") in resting and resting[("T83.99", "bid")].price_cents == 97
    # a taker sells YES at 95c (through our 97c bid): we fill first, up to our size
    iso = dt.datetime.fromtimestamp(now + 5, dt.timezone.utc).isoformat().replace("+00:00", "Z")
    trade = {"ticker": "KXTEMPMIAH-26OCT0312-T83.99", "created_time": iso, "yes_price_dollars": "0.9500",
             "count_fp": "25.00", "taker_side": "no"}
    for f in bot.paper_h.on_trade(trade):
        bot.record_h_fill(f)
    rows = bot.store.hourly_open()
    assert len(rows) == 1 and rows[0][3] == "bid" and rows[0][4] == 97 and rows[0][5] == 10.0
    assert bot.h_held[ev.event] == [("KXTEMPMIAH-26OCT0312-T83.99", "bid", 97, 10.0)]
    # an hour later the market settles YES: +3c x 10
    bot.store.c.execute("UPDATE hourly_shadow SET ts = ts - 4000")
    bot.settle_h()
    pnl, n = bot.store.hourly_pnl()
    assert abs(pnl - 0.30) < 1e-9 and n == 10.0
    assert ev.event not in bot.h_held
    # the requote after close cancels everything
    bot.hourly = {}
    bot.shadow_h_orders(time.time())
    assert not bot.paper_h.resting()


def test_h2_quotes_only_where_the_nowcast_agrees():
    ev = ladder()
    tk = lambda k: f"KXTEMPMIAH-26OCT0312-T{k}.99"
    cfg = H.SleeveHConfig(nowcast=True)
    # nowcast: T83.99 is 99.5% yes (agrees with the 96/99 favourite); T84.99 only 90% (disagrees with 91/94);
    # T87.99 0.3% yes (agrees with the -/3 longshot); T86.99 6% (disagrees with 4/9)
    p_of = {tk(83): 0.995, tk(84): 0.90, tk(85): 0.45, tk(86): 0.06, tk(87): 0.003}
    out = H.propose(ev, time.time(), cfg, p_of=p_of)
    got = {(o.ticker.rsplit("-", 1)[1], o.side): o.price_cents for o in out}
    assert got == {("T83.99", "bid"): 97, ("T87.99", "ask"): 3}   # 0.3% + 2c edge rounds up to 3c
    # quotes keep the edge from the nowcast: p = 0.985 caps the bid at 96c even though the book allows 97c
    out = H.propose(ev, time.time(), cfg, p_of={**p_of, tk(83): 0.985})
    assert {(o.ticker.rsplit("-", 1)[1], o.side): o.price_cents for o in out}[("T83.99", "bid")] == 96
    # a strike without a nowcast is never quoted
    assert H.propose(ev, time.time(), cfg, p_of={}) == []


def test_index_feed_keeps_history_and_series(tmp_path):
    from wxbot.data.index_feed import IndexFeed
    from wxbot.store.db import Store
    now_ms = int(time.time() // 60) * 60000

    class R:
        def weather_index(self, city):
            pts = [{"t": now_ms - k * 60000, "v": 80.0 + k / 100, "status": "normal", "contributors": 5}
                   for k in range(5, 0, -1)]
            pts.append({"t": now_ms, "v": None, "status": "unavailable"})
            return {"timeseries": pts}
    st = Store(str(tmp_path / "i.db"))
    f = IndexFeed(R(), st, ["miami"])
    f.poll()
    s = f.as_series("miami")
    assert len(s) == 5 and s.index.is_monotonic_increasing and abs(s.iloc[-1] - 80.01) < 1e-9
    f2 = IndexFeed(R(), st, ["miami"])   # a restart reloads from the store
    assert len(f2.as_series("miami")) == 5


def test_runner_h2_uses_the_nowcast_probabilities(tmp_path):
    import numpy as np
    import pandas as pd
    from wxbot.app.runner import Bot
    from wxbot.execution.paper import PaperBroker
    from wxbot.ml import nowcast as NC
    from wxbot.store.db import Store

    class FakeNowcaster:
        def ok(self, city):
            return city == "miami"

        def predict(self, city, R, S, m):
            P = np.zeros(2 * NC.NB + 1)
            P[NC.NB] = 0.5       # unchanged from x0 = 85.5
            P[NC.NB + 1] = 0.5   # +0.36 F
            return P, 85.5

    class FakeFeed:
        cities = ["miami"]

        def as_series(self, city):
            return pd.Series([85.5], index=pd.to_datetime([int(time.time() // 60) * 60], unit="s"))

        def hours_of_history(self, city):
            return 1.0

        def latest(self, city):
            return int(time.time() // 60), 85.5

    bot = Bot.__new__(Bot)
    bot.store = Store(str(tmp_path / "h2.db"))
    bot.cfg_h = H.SleeveHConfig(nowcast=True)
    bot.h_on = bot.h2_on = True
    bot.paper_h, bot.paper_h2 = PaperBroker(), PaperBroker()
    bot.h_held, bot.h2_held = defaultdict(list), defaultdict(list)
    bot.counters = defaultdict(int)
    bot.stale_book_s = 90
    bot.nowcaster, bot.index_feed = FakeNowcaster(), FakeFeed()
    ev = ladder()
    bot.hourly = {ev.event: ev}
    bot.book_ts = {m["ticker"]: time.time() for m in ev.markets}
    bot.shadow_h_orders(time.time(), "H2")
    got = {(o.ticker.rsplit("-", 1)[1], o.side) for o in bot.paper_h2.resting()}
    # the value ends at 85.50 or 85.86: above 83.99 and 84.99 for sure, never above 86.99 or 87.99
    assert ("T83.99", "bid") in got and ("T84.99", "bid") in got and ("T87.99", "ask") in got
    assert ("T86.99", "ask") in got   # book 4/9: ask 9 <= 10c and the nowcast says 0%
    assert not bot.paper_h.resting()  # variant H untouched
    st = bot.h_status()
    assert st["H2"]["quotes"] == len(got)


def test_h2_lead_is_anchored_on_the_newest_index_point(tmp_path):
    """Live index points arrive ~5.5 minutes after their minute: the lead must come from the newest point held, so the
    nowcast sees a real x0 instead of a missing minute (which silenced H2 on its first live day)."""
    import pandas as pd
    from wxbot.app.runner import Bot
    from wxbot.ml import nowcast as NC

    now = time.time()
    close = (int(now // 60) + 30) * 60
    seen = []

    class Nc:
        def ok(self, city):
            return True

        def predict(self, city, R, S, m):
            X, x0 = NC.features(R, pd.DatetimeIndex([S]), m, city)
            seen.append((m, x0[0]))
            import numpy as np
            P = np.zeros(2 * NC.NB + 1)
            P[NC.NB] = 1.0
            return P, float(x0[0])

    class Feed:
        def __init__(self, lag_min):
            self.t = int(now // 60) - lag_min
            self.cities = ["miami"]

        def as_series(self, city):
            ts = range(self.t - 200, self.t + 1)
            return pd.Series([85.0] * len(ts), index=pd.to_datetime([t * 60 for t in ts], unit="s"))

        def latest(self, city):
            return self.t, 85.0

    ev = ladder()
    ev.close_ts = close
    bot = Bot.__new__(Bot)
    bot.nowcaster = Nc()
    for lag in (5, 6, 7):
        bot.index_feed = Feed(lag)
        bot._h2_cache = {}
        p = bot.h2_probs(ev, now)
        assert p is not None and seen[-1][0] == 30 + lag - NC.DELAY and seen[-1][1] == 85.0
        assert p["KXTEMPMIAH-26OCT0312-T83.99"] == 1.0 and p["KXTEMPMIAH-26OCT0312-T85.99"] == 0.0
    bot.index_feed = Feed(20)   # stale feed: no nowcast
    bot._h2_cache = {}
    assert bot.h2_probs(ev, now) is None


def test_index_store_keeps_points_beyond_the_memory_window(tmp_path):
    from wxbot.data.index_feed import IndexFeed
    from wxbot.store.db import Store
    now_min = int(time.time() // 60)
    st = Store(str(tmp_path / "k.db"))
    st.add_index_points("miami", [(now_min - 40 * 60, 81.0), (now_min - 3 * 1440, 80.0)])   # 40 h and 3 days old

    class R:
        def weather_index(self, city):
            return {"timeseries": [{"t": (now_min - 6) * 60000, "v": 85.0, "status": "normal"}]}
    f = IndexFeed(R(), st, ["miami"])
    f.poll()
    assert len(f.as_series("miami")) == 1                       # memory: the last 30 hours only
    assert len(st.index_points("miami", now_min - 5 * 1440)) == 3   # store: everything kept


def test_risk_worst_case_on_threshold_ladders_matches_sleeve_h_scenarios():
    """The shared risk check (used by any live order) must treat 'above k' strikes as nested, not exclusive: a YES
    bid at 97c on the lowest strike and a YES ask at 3c on the highest cannot both lose."""
    import random
    from wxbot.risk.limits import ThresholdLadder, event_worst_case
    from wxbot.strategy.types import DesiredOrder
    ev = ladder()
    lad = ThresholdLadder.from_markets(ev.markets)
    assert [t.rsplit("-", 1)[1] for t in lad] == ["T83.99", "T84.99", "T85.99", "T86.99", "T87.99"]
    k_of = {m["ticker"]: H.strike_of(m) for m in ev.markets}
    strikes = sorted(k_of.values())
    rng = random.Random(1)
    for _ in range(200):
        orders = []
        for t in k_of:
            for side in ("bid", "ask"):
                if rng.random() < 0.4:
                    px = rng.randint(1, 99)
                    orders.append(DesiredOrder(t, side, px, float(rng.randint(1, 20)), "H", 0.0, event=ev.event))
        L = [0.0] * (len(strikes) + 1)
        for o in orders:   # resting orders: adverse selection, each counts only where it loses
            for j, l in enumerate(H.scenario_losses(strikes, o.side, o.price_cents, k_of[o.ticker])):
                L[j] += o.count * max(l, 0.0)
        assert abs(event_worst_case([], orders, lad) - max(0.0, max(L))) < 1e-9
    # the exclusive-bracket reading would overstate this pair's risk
    pair = [DesiredOrder("KXTEMPMIAH-26OCT0312-T83.99", "bid", 97, 20.0, "H", 0.0, event=ev.event),
            DesiredOrder("KXTEMPMIAH-26OCT0312-T87.99", "ask", 3, 20.0, "H", 0.0, event=ev.event)]
    assert abs(event_worst_case([], pair, lad) - 19.40) < 1e-9   # value <= 83.99: the bid loses 19.40 (the ask's 0.60
    #                                                               gain is not counted: it may not fill)
    assert abs(event_worst_case([], pair, list(lad)) - 38.80) < 1e-9   # read as exclusive brackets: both lose
    from wxbot.strategy.types import Position   # filled, the same pair offsets exactly
    held = [Position(pair[0].ticker, ev.event, yes_qty=20, yes_cost=19.40),
            Position(pair[1].ticker, ev.event, no_qty=20, no_cost=19.40)]
    assert abs(event_worst_case(held, [], lad) - 18.80) < 1e-9


def test_real_order_path_uses_nested_ladder_risk_and_its_own_budget(tmp_path):
    """Sleeve H with enabled=True (off in every config until the Dec 1 tests pass): orders go through the shared risk
    filter with the hourly ladder read as nested strikes, and through Sleeve H's own total-risk budget."""
    from wxbot.app.runner import Bot
    from wxbot.execution.paper import PaperBroker
    from wxbot.risk.limits import RiskConfig, RiskState
    from wxbot.store.db import Store
    from wxbot.strategy.types import Position

    bot = Bot.__new__(Bot)
    bot.store = Store(str(tmp_path / "hl.db"))
    bot.cfg_h = H.SleeveHConfig(enabled=True, size=10, cap_usd=25.0, total_risk_usd=50.0, max_cash_usd=150.0)
    bot.positions, bot.events = {}, {}
    ev = ladder()
    bot.hourly = {ev.event: ev}
    bot.netting, bot.mode, bot.live = False, "paper", None
    bot.risk_cfg = RiskConfig()
    bot.risk = RiskState(bot.risk_cfg)
    bot.series_t = set()
    bot.paper = PaperBroker()
    bot.counters = defaultdict(int)
    bot.next_due = defaultdict(float)
    bot.ev_tickers, bot.ev_tickers_miss = {}, {}
    orders = H.propose(ev, time.time(), bot.cfg_h, bot.h_held_live(ev.event))
    assert {o.side for o in orders} == {"bid", "ask"}
    bot.execute(orders)
    rest = {(o.ticker.rsplit("-", 1)[1], o.side): o.remaining for o in bot.paper.resting()}
    # the favorite bid and the longshot offer both survive: read as exclusive brackets they would not fit $25
    assert rest.get(("T83.99", "bid")) == 10 and rest.get(("T87.99", "ask")) == 10
    assert bot.open_risk_h == 0.0
    # held contracts map back for the per-hour size limit, and the budget counts them
    bot.positions["KXTEMPMIAH-26OCT0312-T83.99"] = Position("KXTEMPMIAH-26OCT0312-T83.99", ev.event, yes_qty=10,
                                                           yes_cost=9.7)
    assert bot.h_held_live(ev.event) == [("KXTEMPMIAH-26OCT0312-T83.99", "bid", 97, 10.0)]
    again = H.propose(ev, time.time(), bot.cfg_h, bot.h_held_live(ev.event))
    assert not [o for o in again if o.ticker.endswith("T83.99") and o.side == "bid"]   # 10 already filled
    bot.cfg_h.total_risk_usd = 5.0     # a tiny budget leaves room for nothing new beyond the held risk
    bot.execute(again)
    assert not bot.paper.resting()


def test_the_ask_only_variant_quotes_only_the_low_tail():
    """Amendment A7: sides=(ask,) keeps the frozen rule but drops every YES bid at the high tail."""
    ev = ladder()
    both = H.propose(ev, time.time(), H.SleeveHConfig())
    asks = H.propose(ev, time.time(), H.SleeveHConfig.from_cfg({"sides": ["ask"]}))
    assert {o.side for o in both} == {"bid", "ask"}
    assert asks and {o.side for o in asks} == {"ask"}
    assert [(o.ticker, o.price_cents) for o in asks] == [(o.ticker, o.price_cents) for o in both if o.side == "ask"]
    assert H.SleeveHConfig().sides == ("bid", "ask")                              # the frozen default: both
