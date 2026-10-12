"""Sleeve E in winter (Oct 9; holdout amendments draft K3, option (a)): from the end of daylight time (Nov 1, 2026) to
Mar 14, 2027 Kalshi lists the daily markets at 15Z, the minute of the h = -9 decision, and the research has no quotes
there. With early_taker.skip_h9_in_standard_time the bot sends no order from an h = -9 decision while US Eastern is on
standard time; the signal stays logged. Off by default."""
import datetime as dt
import sys
import os
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_early_taker import make_bot, shadow_db  # noqa: E402

from wxbot import timeutil  # noqa: E402

U = lambda *a: dt.datetime(*a, tzinfo=dt.timezone.utc).timestamp()   # noqa: E731

# (UTC time, US Eastern on standard time?) around both changes, and the h = -9 decision times either side of them
BOUNDARIES = [(U(2026, 11, 1, 5, 59, 59), False), (U(2026, 11, 1, 6, 0), True),       # 2 AM EDT -> 1 AM EST
              (U(2027, 3, 14, 6, 59, 59), True), (U(2027, 3, 14, 7, 0), False),       # 2 AM EST -> 3 AM EDT
              (U(2026, 10, 31, 15), False), (U(2026, 11, 1, 15), True), (U(2027, 1, 15, 15), True),
              (U(2027, 3, 13, 15), True), (U(2027, 3, 14, 15), False), (U(2026, 7, 4, 15), False),
              (U(2027, 11, 7, 6), True), (U(2027, 11, 7, 5, 59), False), (U(2028, 3, 12, 7), False),
              (U(2028, 3, 12, 6, 59), True)]


@pytest.mark.parametrize("tzdb", [True, False])
def test_standard_time_across_the_boundaries(monkeypatch, tzdb):
    if not tzdb:                       # without the tz database: the US rule, not to_et's month-level fallback
        monkeypatch.setattr(timeutil, "_ET", None)
    for ts, std in BOUNDARIES:
        assert timeutil.et_standard_time(ts) is std, dt.datetime.fromtimestamp(ts, dt.timezone.utc)


def row(ts, event, ticker, h, side="yes", price=0.27):
    return (ts, event, ticker, side, price, 5.0, 40.0, 0.6, 0.3, h, "c0ffee", 0.03, 1.0)


# event day -> (its h = -9 decision is in standard time?): the decision is 15Z the day before
EVENTS = {"KXHIGHNY-26NOV01": False,      # Oct 31 15Z, daylight time: sent
          "KXHIGHNY-26NOV02": True,       # Nov 1 15Z, standard time: not sent
          "KXHIGHNY-27JAN15": True,
          "KXHIGHNY-27MAR14": True,       # Mar 13 15Z: still standard time
          "KXHIGHNY-27MAR15": False}      # Mar 14 15Z: daylight time again


def winter_rows(now):
    rows = []
    for ev in EVENTS:
        rows.append(row(now - 30, ev, f"{ev}-B70.5", -9))
        rows.append(row(now - 20, ev, f"{ev}-T73", -6, "no", 0.77))      # h = -6 is never skipped
    return rows


def test_h9_orders_are_skipped_in_standard_time_and_the_signals_stay_logged(tmp_path):
    now = time.time()
    shadow_db(tmp_path, winter_rows(now))
    bot = make_bot(tmp_path, {"enabled": True, "trade": True, "max_age_s": 600, "skip_h9_in_standard_time": True})
    n = bot.early_take(now)
    sent = {s[2] for s in bot.sent}
    want = {f"{ev}-B70.5" for ev, std in EVENTS.items() if not std} | {f"{ev}-T73" for ev in EVENTS}
    assert sent == want and n == len(want) == 7
    for ev, std in EVENTS.items():
        rec = bot.store.kv_get(f"early_take:{ev}:{ev}-B70.5:yes")
        assert rec["h"] == -9 and (rec.get("skipped") == "h = -9 in standard time") is std
    assert bot.early_take(now + 30) == 0 and len(bot.sent) == 7                 # each signal handled once


def test_off_by_default_every_fresh_signal_is_sent(tmp_path):
    now = time.time()
    shadow_db(tmp_path, winter_rows(now))
    bot = make_bot(tmp_path, {"enabled": True, "trade": True, "max_age_s": 600})
    assert bot.early_take(now) == 10 and len(bot.sent) == 10


def test_an_unreadable_event_date_falls_back_to_the_time_the_signal_was_logged():
    from wxbot.app.runner import Bot
    assert Bot.h9_in_standard_time("ODD", U(2026, 11, 3, 15, 5)) is True
    assert Bot.h9_in_standard_time("ODD", U(2026, 10, 30, 15, 5)) is False
    assert Bot.h9_in_standard_time("KXHIGHNY-26NOV02", U(2026, 10, 1)) is True   # the date decides when readable
