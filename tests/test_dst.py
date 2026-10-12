"""The climate day is midnight to midnight local STANDARD time all year (NWS CLI), so the Nov 1, 2026 end of daylight
time must not move any day boundary: observations, features and the bot's target day all use fixed standard offsets
(Oct 4 review: DST and midnight highs)."""
import datetime as dt
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wxbot.data.obs import lst_day_bounds, running_extremes  # noqa: E402
from wxbot.ml.features import localday  # noqa: E402

STATIONS = json.load(open(os.path.join(os.path.dirname(__file__), "..", "wxbot", "data", "stations.json")))
U = lambda *a: dt.datetime(*a, tzinfo=dt.timezone.utc).timestamp()


def test_standard_offsets_are_fixed_for_every_traded_station():
    offs = {s: v["lst_offset_h"] for s, v in STATIONS.items() if v.get("kind") == "high"}
    assert offs["KXHIGHNY"] == -5 and offs["KXHIGHCHI"] == -6 and offs["KXHIGHDEN"] == -7 and offs["KXHIGHLAX"] == -8
    assert all(o in (-5, -6, -7, -8, -10) for o in offs.values())


def test_day_bounds_do_not_move_when_daylight_time_ends():
    for d in (dt.date(2026, 10, 31), dt.date(2026, 11, 1), dt.date(2026, 11, 2)):
        t0, t1 = lst_day_bounds(d, -5)
        assert t0 == U(d.year, d.month, d.day, 5) and t1 - t0 == 86400


def test_a_report_at_00_30_daylight_time_belongs_to_the_previous_climate_day():
    # Oct 31, 00:30 EDT = 04:30 UTC = Oct 30, 23:30 EST: still Oct 30 for the CLI
    ts = pd.Series(pd.to_datetime([U(2026, 10, 31, 4, 30), U(2026, 11, 2, 4, 30)], unit="s", utc=True))
    assert [str(x.date()) for x in localday(ts, -5)] == ["2026-10-30", "2026-11-01"]


def obs(ts, temp_c):
    return {"obsTime": ts, "temp": temp_c, "rawOb": ""}


def test_a_midnight_high_counts_for_the_day_it_starts():
    # Nov 2 (EST): 79 F right after LST midnight (05:10 UTC), a front, then 60s all day
    o = [obs(U(2026, 11, 2, 5, 10), 26.1), obs(U(2026, 11, 2, 6, 10), 25.6)] + \
        [obs(U(2026, 11, 2, h, 10), 16.0) for h in range(8, 23)] + [obs(U(2026, 11, 2, 4, 50), 27.0)]  # 4:50Z = Nov 1
    ext = running_extremes(o, dt.date(2026, 11, 2), -5)
    assert round(ext["max_f"]) == 79 and ext["n_obs"] == 17
    ext1 = running_extremes(o, dt.date(2026, 11, 1), -5)
    assert round(ext1["max_f"]) == 81 and ext1["n_obs"] == 1


def test_the_bots_target_day_uses_standard_time():
    from wxbot.app.runner import Bot
    from wxbot.strategy.types import EventBook
    b = Bot.__new__(Bot)
    ev = EventBook("KXHIGHNY-26NOV01", "KXHIGHNY", "high", "KNYC", dt.date(2026, 11, 1), 0)
    assert b.target_day(ev, U(2026, 11, 2, 4, 30))         # 23:30 EST Nov 1
    assert not b.target_day(ev, U(2026, 11, 2, 5, 30))     # 00:30 EST Nov 2
    ev_oct = EventBook("KXHIGHNY-26OCT30", "KXHIGHNY", "high", "KNYC", dt.date(2026, 10, 30), 0)
    assert b.target_day(ev_oct, U(2026, 10, 31, 4, 30))    # 00:30 EDT Oct 31 is still Oct 30 LST
