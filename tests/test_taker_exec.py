"""Taker order path (off until the taker sleeves pass their tests): book-side conversion, partial fills, fees, and
the immediate-or-cancel order the live path sends."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wxbot.execution.taker import TakerExecutor, offers, simulate_take, taker_fee

BOOK = {"yes_dollars": [["0.30", "10"], ["0.29", "5"]], "no_dollars": [["0.65", "7"], ["0.60", "20"]]}


def test_offers_come_from_the_other_side_of_the_book():
    assert offers(BOOK, "yes") == [(0.35, 7.0), (0.40, 20.0)]
    assert offers(BOOK, "no") == [(0.70, 10.0), (0.71, 5.0)]


def test_simulated_take_walks_levels_up_to_the_limit():
    f, avg, fee = simulate_take(BOOK, "yes", 0.35, 10)
    assert f == 7 and avg == 0.35 and fee == taker_fee(7, 0.35) == 0.12
    f, avg, fee = simulate_take(BOOK, "yes", 0.40, 10)
    assert f == 10 and abs(avg - (7 * 0.35 + 3 * 0.40) / 10) < 1e-12
    assert simulate_take(BOOK, "no", 0.69, 5)[0] == 0
    r = TakerExecutor(mode="paper").take("T", "no", 0.70, 12, BOOK)
    assert r.status == "partial" and r.filled == 10
    assert TakerExecutor(mode="paper").take("T", "yes", 0.35, 5, None).status == "no book"


def test_live_path_sends_ioc_on_the_yes_scale():
    calls = []

    class Rest:
        def create_order(self, ticker, side, price_cents, count, **kw):
            calls.append((ticker, side, price_cents, count, kw))
            return {"order_id": "o1", "fill_count": "4.00", "remaining_count": "0.00",
                    "average_fill_price": "0.3000", "average_fee_paid": "0.0150"}
    ex = TakerExecutor(Rest(), mode="live", max_slip_c=1)
    r = ex.take("KXRAIN-26OCT04-NYC", "no", 0.69, 5)
    t, side, price, count, kw = calls[-1]
    assert (side, price, count) == ("ask", 30, 5) and kw["tif"] == "immediate_or_cancel" and kw["post_only"] is False
    assert r.status == "partial" and r.filled == 4 and r.avg_price == 0.70 and abs(r.fee - 0.06) < 1e-12
    ex.take("KXRAIN-26OCT04-NYC", "yes", 0.42, 5)
    assert calls[-1][1:3] == ("bid", 43)


def test_a_no_fill_price_on_either_scale_is_read_as_the_no_price(monkeypatch):
    """Kalshi's docs do not say which scale average_fill_price uses for an ask: both readings give the NO price."""
    for answer in ("0.3100", "0.6900"):                # YES scale (the v2 convention) or NO scale
        class Rest:
            def create_order(self, ticker, side, price_cents, count, **kw):
                return {"order_id": "o1", "fill_count": "5.00", "remaining_count": "0.00",
                        "average_fill_price": answer, "average_fee_paid": "0.0150"}
        r = TakerExecutor(Rest(), mode="live", max_slip_c=1).take("KXRAIN-26OCT04-NYC", "no", 0.69, 5)
        assert r.avg_price == 0.69, (answer, r.avg_price)


def test_runner_take_path_is_off_by_default_and_capped(tmp_path):
    from collections import defaultdict
    from wxbot.app.runner import Bot
    from wxbot.risk.limits import RiskConfig, RiskState
    from wxbot.store.db import Store

    class Rest:
        def orderbook(self, t):
            return BOOK

    bot = Bot.__new__(Bot)
    bot.store = Store(str(tmp_path / "t.db"))
    bot.rest, bot.mode = Rest(), "paper"
    bot.risk_cfg = RiskConfig()
    bot.risk = RiskState(bot.risk_cfg)
    bot.positions, bot.counters, bot.fair_cache = {}, defaultdict(int), {}
    bot.events, bot.hourly, bot.ev_tickers, bot.ev_tickers_miss = {}, {}, {}, {}
    bot.live, bot.paper, bot.netting = None, None, False
    bot.ev_tickers["KXRAIN-26OCT04"] = ["KXRAIN-26OCT04-NYC"]
    assert bot.execute_take("R", "KXRAIN-26OCT04", "KXRAIN-26OCT04-NYC", "no", 0.70, 5, {}) is None   # trade off
    cfg = {"trade": True, "trade_size": 5, "trade_max_usd_day": 5.0}
    r = bot.execute_take("R", "KXRAIN-26OCT04", "KXRAIN-26OCT04-NYC", "no", 0.70, 5, cfg)
    assert r["filled"] == 5 and r["status"] == "filled"
    p = bot.positions["KXRAIN-26OCT04-NYC"]
    assert p.no_qty == 5 and abs(p.no_cost - 3.5) < 1e-9 and p.fees > 0          # NO at 70c, booked like a fill
    # the day's budget is now used up: the next signal is skipped
    assert bot.execute_take("R", "KXRAIN-26OCT04", "KXRAIN-26OCT04-NYC", "yes", 0.35, 5, cfg) is None
