"""Sleeve H breaker (real hourly orders switch off when too many settled contracts lose or the P&L per contract goes
negative), the tighter book-age guard for hourly quotes, the timed METAR polling and the speed stats."""
import json
import os
import sys
import time
from collections import defaultdict, deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wxbot.strategy import sleeve_h as H
from wxbot.strategy.types import Fill

T83 = "KXTEMPMIAH-26OCT0312-T83.99"
T87 = "KXTEMPMIAH-26OCT0312-T87.99"
NYC = "KXTEMPNYCHS-26OCT0312-T70.99"


def row(ts, ticker, side, pc, n, fee=0.0):
    return (ts, "live", ticker, ticker.rsplit("-", 1)[0], side, pc, n, "?", 0, fee, "", None)


def test_breaker_stats_counts_both_sides_and_ignores_other_series_open_markets_and_old_fills():
    fills = [row(100, T83, "bid", 97, 10),            # long YES at 97, settles yes: +0.30
             row(100, T87, "ask", 3, 10),             # long NO at 97 (YES ask at 3), settles no: +0.30
             row(100, NYC, "bid", 96, 10, fee=0.05),  # long YES at 96, settles no: -9.60, fee 0.05
             row(100, "KXHIGHMIA-26OCT03-B85.5", "bid", 50, 7),   # a daily market: not hourly
             row(100, "KXTEMPMIAH-26OCT0313-T80.99", "bid", 95, 5),   # not settled yet
             row(10, T83, "bid", 97, 10)]             # before the baseline
    res = {T83: "yes", T87: "no", NYC: "no", "KXHIGHMIA-26OCT03-B85.5": "no"}
    s = H.breaker_stats(fills, res, H.SleeveHConfig().series, since=50)
    assert s["contracts"] == 30 and s["lost"] == 10
    assert abs(s["lost_share"] - 1 / 3) < 1e-3
    assert abs(s["pnl_usd"] - (0.30 + 0.30 - 9.60 - 0.05)) < 1e-9
    assert abs(s["pnl_c_per_contract"] - 100 * (-9.05) / 30) < 0.01


def test_breaker_waits_for_enough_contracts_then_checks_both_rules():
    c = H.SleeveHConfig()
    assert c.breaker_min_contracts == 3000 and c.breaker_max_wrong == 0.06 and c.breaker_min_pnl_c == 0.0
    assert H.breaker_verdict({"contracts": 2999, "lost_share": 0.5, "pnl_c_per_contract": -40.0}, c) is None
    assert H.breaker_verdict({"contracts": 3000, "lost_share": 0.05, "pnl_c_per_contract": 0.4}, c) is None
    assert "lost" in H.breaker_verdict({"contracts": 3000, "lost_share": 0.061, "pnl_c_per_contract": 0.4}, c)
    assert "per contract" in H.breaker_verdict({"contracts": 4000, "lost_share": 0.03, "pnl_c_per_contract": -0.1}, c)


def make_bot(tmp_path, mode="live"):
    from wxbot.app.alerts import Alerter
    from wxbot.app.runner import Bot
    from wxbot.store.db import Store
    bot = Bot.__new__(Bot)
    bot.store = Store(str(tmp_path / "live.sqlite"))
    bot.mode = mode
    bot.cfg_h = H.SleeveHConfig(enabled=True, size=10, breaker_min_contracts=30)
    bot.alerts = Alerter(None)
    bot.load_h_breaker()
    return bot


def add(bot, ticker, side, pc, n, result, ts=None):
    bot.store.add_fill(bot.mode, Fill(ts or time.time(), ticker, ticker.rsplit("-", 1)[0], side, pc, n, "?",
                                      False, 0.0, f"f{time.time_ns()}"))
    bot.store.set_settlement(ticker, ticker.rsplit("-", 1)[0], result, time.time(), None)


def test_runner_breaker_trips_survives_a_restart_and_clears_by_hand(tmp_path, monkeypatch):
    bot = make_bot(tmp_path)
    add(bot, T83, "bid", 97, 20, "yes")
    bot.check_h_breaker()
    assert bot.cfg_h.enabled and bot.h_breaker_last["contracts"] == 20     # too few settled contracts to judge
    add(bot, T87, "ask", 3, 10, "yes")                                      # long NO, the index ended above 87.99
    bot.check_h_breaker()
    assert not bot.cfg_h.enabled
    kv = bot.store.kv_get("h_breaker")
    assert kv["tripped"] and "lost" in kv["reason"] and kv["stats"]["contracts"] == 30
    assert bot.store.c.execute("SELECT COUNT(*) FROM events_log WHERE level='ERROR' AND msg LIKE '%breaker%'"
                               ).fetchone()[0] == 1
    assert bot.h_real_orders(time.time()) == []          # nothing proposed, so reconcile cancels resting hourly orders
    # a restart keeps it off even though the config still says enabled
    again = make_bot(tmp_path)
    assert not again.cfg_h.enabled and again.h_breaker_status()["tripped"]
    # clearing it by hand: counting starts afresh and the bot is asked to restart
    cfg = {"db_path": str(tmp_path / "live.sqlite"), "status_path": str(tmp_path / "status.json"), "mode": "live"}
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(json.dumps(cfg))                 # JSON is valid YAML
    monkeypatch.chdir(tmp_path)
    from wxbot.__main__ import main
    main(["reset-h-breaker", "--config", str(cfg_path)])
    assert os.path.exists(tmp_path / "RESTART")
    fresh = make_bot(tmp_path)
    assert fresh.cfg_h.enabled
    fresh.check_h_breaker()
    assert fresh.cfg_h.enabled and fresh.h_breaker_last["contracts"] == 0   # the old losses are not counted again
    assert fresh.store.kv_get("h_breaker")["previous_reason"] == kv["reason"]


def test_breaker_off_when_the_sleeve_is_off(tmp_path):
    bot = make_bot(tmp_path)
    bot.cfg_h.enabled = False
    add(bot, T87, "ask", 3, 50, "yes")
    bot.check_h_breaker()
    assert not bot.store.kv_get("h_breaker")


def test_hourly_real_orders_need_fresher_books_than_the_daily_quotes(tmp_path):
    from test_sleeve_h import ladder
    bot = make_bot(tmp_path, mode="paper")
    ev = ladder()
    bot.hourly = {ev.event: ev}
    bot.positions = {}
    bot.stale_book_s = 90.0
    now = time.time()
    bot.book_ts = {m["ticker"]: now - 10 for m in ev.markets}
    assert bot.h_real_orders(now)
    bot.book_ts = {m["ticker"]: now - 60 for m in ev.markets}   # fine for the daily quotes (90 s), too old for hourly
    assert bot.h_real_orders(now) == []
    bot.book_ts = {m["ticker"]: now - 10 for m in ev.markets}
    bot.book_ts[ev.markets[0]["ticker"]] = now - 300              # one market missing from recent refreshes
    assert bot.h_real_orders(now) == []


def test_metar_polling_speeds_up_while_routine_reports_land():
    from wxbot.app.runner import Bot
    bot = Bot.__new__(Bot)
    bot.cfg = {"loop": {"metar_every_s": 60, "metar_fast_every_s": 20, "metar_fast_window": [53, 2]}}
    assert [bot.metar_interval(m) for m in (50, 52, 53, 57, 59, 0, 2, 3, 30)] == [60, 60, 20, 20, 20, 20, 20, 60, 60]
    bot.cfg = {"loop": {"metar_every_s": 300}}
    assert bot.metar_interval(55) == 300


def test_speed_stats_report_book_age_step_time_and_round_trips():
    from wxbot.app.runner import Bot
    from wxbot.kalshi.rest import KalshiREST
    bot = Bot.__new__(Bot)
    bot.rest = KalshiREST(base_url="https://example.invalid/trade-api/v2", auth=None)
    for i in range(1, 101):
        bot.rest.rtt["read"].append(i / 1000.0)
    bot.t_step = deque([0.2, 0.3, 0.4, 5.0])
    bot.t_book_age = deque(range(1, 21))
    t = bot.timing_status()
    assert t["api_round_trip_ms"]["read"] == {"n": 100, "p50": 51, "p95": 96}
    assert "write" not in t["api_round_trip_ms"]
    assert t["step_s"]["max"] == 5.0 and t["book_age_at_orders_s"]["p50"] == 11


def test_dashboard_shows_speed_and_the_hourly_breaker(tmp_path):
    from wxbot.app.dashboard import build_html
    from wxbot.store.db import Store
    db = str(tmp_path / "live.sqlite")
    Store(db)
    st = {"mode": "live", "timing": {"book_age_at_orders_s": {"n": 9, "p50": 3.0, "p95": 6.0, "max": 7.0},
                                     "step_s": {"n": 9, "p50": 0.4, "p95": 1.0, "max": 2.0},
                                     "api_round_trip_ms": {"read": {"n": 9, "p50": 80, "p95": 150}}},
          "sleeve_h_breaker": {"real_orders_on": False, "tripped": True, "reason": "7.0% of 300 settled contracts lost",
                               "stats": {"contracts": 300, "lost_share": 0.07, "pnl_c_per_contract": -1.2},
                               "limits": {"min_contracts": 300, "max_lost_share": 0.06, "min_pnl_c_per_contract": 0.0}}}
    sp = tmp_path / "status.json"
    sp.write_text(json.dumps(st))
    page = build_html({"db_path": db, "mode": "live", "taker_shadow": {"trade": True}}, db, str(sp))
    assert "Speed: order book age when orders update: median 3 s" in page
    assert "SWITCHED OFF" in page and "7.0%" in page and "-1.20c" in page
