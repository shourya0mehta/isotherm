"""Rounding guard (round 7, Oct 9; wxbot/models/round_guard.py, docs/results/r7_round_guard.txt).

The Oct 8 Oklahoma City fixture below is the live loss on KXHIGHTOKC-26OCT08 (the one October reading set allowed in
tests): 5-minute readings of 31 C from 18:55Z (87 or 88 F, either side of the 86-87 | 88-89 line), hourly T-groups of
30.6 C (87 F) at 19:52-21:52Z, and the 23:52Z report's 6-hour max 10306 (30.6 C = 87 F). The final high was 87 F.
Values not given in the incident report (the 30 C readings' exact span, the 23:52Z report's body, the ladder's tails)
are placeholders consistent with it."""
import datetime as dt
import json
import sqlite3
import time

import pytest

from wxbot.models import round_guard as R
from wxbot.models.brackets import INF, Bracket, bracket_from_market


def T(h, m, day=8):
    return dt.datetime(2026, 10, day, h, m, tzinfo=dt.timezone.utc).timestamp()


HOT = ["18:55", "19:20", "19:25", "19:30", "19:35", "19:50", "19:55", "20:05", "20:15", "20:20", "20:25", "20:30",
       "20:35", "20:40", "20:45", "20:55", "21:00", "21:05", "21:10", "21:20", "21:25", "21:30", "21:35", "21:45"]
HOURLY = [((17, 52), "T02940161", 29.4), ((18, 52), "T03000156", 30.0), ((19, 52), "T03060150", 30.6),
          ((20, 52), "T03060150", 30.6), ((21, 52), "T03060144", 30.6), ((22, 52), "T02940150", 29.4)]
SIX_MAX_RAW = "KOKC 082352Z 17009KT 10SM CLR 28/14 A2990 RMK AO2 SLP120 10306 20283"
FIVE_MIN_DELAY = 15 * 60       # Oct 8: the feed saw the 5-minute readings 10-25 minutes after their time
METAR_DELAY = 4 * 60           # the AWC feed the bot polls: about 3.5 minutes after the report time
EV = "KXHIGHTOKC-26OCT08"


def five_minute():
    """(time, whole C): 31 at the reported times, 30 in between from 18:00Z to 22:30Z, 29 before and after."""
    out = []
    for h in range(16, 24):
        for m in range(0, 60, 5):
            s = f"{h:02d}:{m:02d}"
            out.append((T(h, m), 31 if s in HOT else (30 if "18:00" <= s <= "22:30" else 29)))
    return out


def okc_markets():
    return [{"ticker": f"{EV}-T86", "strike_type": "less", "cap_strike": 86},
            {"ticker": f"{EV}-B86.5", "strike_type": "between", "floor_strike": 86, "cap_strike": 87},
            {"ticker": f"{EV}-B88.5", "strike_type": "between", "floor_strike": 88, "cap_strike": 89},
            {"ticker": f"{EV}-B90.5", "strike_type": "between", "floor_strike": 90, "cap_strike": 91},
            {"ticker": f"{EV}-T91", "strike_type": "greater", "floor_strike": 91}]


def metar_raw(hm, tg, c):
    return f"KOKC 08{hm[0]:02d}{hm[1]:02d}Z 18010KT 10SM CLR {round(c):02d}/15 A2992 RMK AO2 SLP125 {tg}"


def okc_readings():
    rs = [R.Reading(t, R.f_set(c), "inst", t + FIVE_MIN_DELAY) for t, c in five_minute()]
    for hm, tg, c in HOURLY:
        rs += R.readings_from_metar(metar_raw(hm, tg, c), T(*hm), T(*hm) + METAR_DELAY)
    rs += R.readings_from_metar(SIX_MAX_RAW, T(23, 52), T(23, 52) + METAR_DELAY)
    return rs


# ------------------------------------------------------------------------------------------------ the table
def test_whole_c_table_matches_the_incident_and_covers_every_whole_f():
    assert R.f_set(30) == (86,)
    assert R.f_set(31) == (87, 88) and R.f_set(32) == (89, 90) and R.f_set(33) == (91, 92) and R.f_set(34) == (93, 94)
    seen = {}
    for f in range(-60, 131):
        assert abs(((f - 32) * 5 / 9) % 1 - 0.5) > 1e-6          # never a rounding tie
        c = R.c_of_f(f)
        assert f in R.f_set(c)
        seen.setdefault(c, []).append(f)
    for c, fs in seen.items():
        assert tuple(fs) == R.f_set(c) and len(fs) in (1, 2) and fs == list(range(fs[0], fs[0] + len(fs)))
        assert (len(fs) == 1) == (c % 5 == 0)                      # exact only at multiples of 5 C
    # the pairs' parity flips every 9 F: 31-34 C split even-start ladders (86-87 | 88-89), 36-39 C odd-start ones
    assert all(R.f_set(c)[0] % 2 == 1 for c in (31, 32, 33, 34))
    assert all(R.f_set(c)[0] % 2 == 0 for c in (36, 37, 38, 39))


def test_reported_temperatures_to_whole_f():
    assert R.f_values(30.6) == (87,) and R.f_values(31.1) == (88,) and R.f_values(30.0) == (86,)
    assert R.f_values(31.0) == (87, 88)                            # a whole C: both
    assert R.f_values(-0.6) == (31,) and R.f_values(-1.0) == (30, 31) and R.f_values(-17.8) == (0,)
    assert R.f_values(30.600000000000023) == (87,)                 # float noise from a decoded feed value


def test_metar_parsing():
    hf = R.readings_from_metar("KOKC 151800Z AUTO 18016G21KT 10SM CLR 37/19 A3002 RMK T03700190 MADISHF", 5.0)
    assert hf == [R.Reading(5.0, (98, 99), "inst", 0.0)]           # the 5-minute feed's T-group is whole C
    syn = R.readings_from_metar("KOKC 152352Z 15017G24KT 10SM FEW090 36/17 A2996 RMK AO2 PK WND 20026/1803 SLP125 "
                                "T03560172 10383 20350 55007", 7.0, avail=9.0)
    assert syn == [R.Reading(7.0, (96,), "inst", 9.0), R.Reading(7.0, (101,), "max6", 9.0),
                   R.Reading(7.0, (95,), "min6", 9.0)]
    assert R.readings_from_metar("KATL 111855Z AUTO 30008KT 10SM SCT046 A3007 RMK MADISHF", 1.0) == []
    assert R.readings_from_metar("KXXX 111855Z AUTO 30008KT 1/2SM M01/M03 A3007", 1.0)[0].vals == (30, 31)
    # the NWS API: 5-minute rows have no text and a whole-C value; a stored METAR cut before its T-group keeps the
    # feed's decoded tenths
    assert R.readings_from_metar("", 1.0, temp_c=31)[0].vals == (87, 88)
    assert R.readings_from_metar("KOKC 082052Z 18010KT 10SM CLR 31/15 A2992 RMK AO2", 1.0, temp_c=30.6)[0].vals == (87,)


# ------------------------------------------------------------------------------------------------ the signal
def okc_brackets():
    return [bracket_from_market(m) for m in okc_markets()]


def blocked_at(t, rs=None):
    t0, t1 = T(6, 0), T(6, 0, day=9)                               # Oct 8 in Central standard time
    amb, _ = R.guard(rs or okc_readings(), okc_brackets(), t0, t1, "high", now=t)
    return {x.rsplit("-", 1)[1] for x in amb}


def test_oct8_okc_blocks_86_87_and_88_89_from_the_first_31c_reading_until_the_six_hour_max():
    first = T(18, 55) + FIVE_MIN_DELAY                             # 19:10Z
    assert blocked_at(first - 1) == set()
    for t in (first, T(19, 30), T(21, 1), T(21, 7), T(22, 17), T(23, 0), T(23, 52) + METAR_DELAY - 1):
        assert blocked_at(t) == {"B86.5", "B88.5"}, dt.datetime.fromtimestamp(t, dt.timezone.utc)
    assert blocked_at(T(23, 52) + METAR_DELAY) == set()            # 10306: 30.6 C = 87 F settles it
    t0, t1 = T(6, 0), T(6, 0, day=9)
    tl = R.timeline(okc_readings(), okc_brackets(), t0, t1)
    on = [(t, {x.rsplit("-", 1)[1] for x in a}) for t, a, _ in tl if a]
    assert on == [(first, {"B86.5", "B88.5"})]                      # 90-91 never
    assert R.extreme_candidates(okc_readings(), t0, t1, "high", now=T(21, 5)) == (87, 88)
    assert R.extreme_candidates(okc_readings(), t0, t1, "high") == (87,)


def test_exact_and_later_readings_settle_or_move_the_question():
    t0, t1 = T(6, 0), T(6, 0, day=9)
    base = [R.Reading(T(19, 0), R.f_set(31), "inst")]
    assert blocked_at(T(20, 0), base) == {"B86.5", "B88.5"}
    # an exact 88 (T-group 31.1) settles it: the max so far is 88
    assert blocked_at(T(20, 0), base + [R.Reading(T(19, 52), (88,), "inst")]) == set()
    # an exact 87 elsewhere does not: the 31 C reading may still be 88
    assert blocked_at(T(20, 0), base + [R.Reading(T(19, 52), (87,), "inst")]) == {"B86.5", "B88.5"}
    # ... unless it is the same minute's reading
    assert blocked_at(T(20, 0), base + [R.Reading(T(19, 0), (87,), "inst")]) == set()
    # a later 32 C (89 or 90) moves the question to the 88-89 | 90-91 line; 86-87 is out of it
    assert blocked_at(T(20, 0), base + [R.Reading(T(19, 30), R.f_set(32), "inst")]) == {"B88.5", "B90.5"}
    # a 6-hour max whose window does not hold the reading settles nothing
    assert blocked_at(T(20, 0), base + [R.Reading(T(17, 52), (86,), "max6")]) == {"B86.5", "B88.5"}
    # on an odd-start ladder (87-88) the same readings are one bracket: nothing to block
    odd = [Bracket("le86", -INF, 86, "less", None, 87), Bracket("87-88", 87, 88, "between", 87, 88),
           Bracket("89-90", 89, 90, "between", 89, 90)]
    assert R.guard(base, odd, t0, t1, "high")[0] == frozenset()


def test_lows_use_the_running_min_and_the_six_hour_min_group():
    """KXLOWT-style ladder 56-57 | 58-59: 14 C readings (57 or 58 F) at dawn, then the 12Z report's 6-hour min
    (2snTTT 20139 = 13.9 C = 57 F) settles it. Eastern standard time: the climate day starts 05Z."""
    brs = [Bracket("le55", -INF, 55, "less", None, 56), Bracket("56-57", 56, 57, "between", 56, 57),
           Bracket("58-59", 58, 59, "between", 58, 59), Bracket("ge60", 60, INF, "greater", 59, None)]
    t0, t1 = T(5, 0), T(5, 0, day=9)
    rs = [R.Reading(T(6, 0), R.f_set(16), "inst", T(6, 15)), R.Reading(T(9, 30), R.f_set(14), "inst", T(9, 45)),
          R.Reading(T(10, 0), R.f_set(14), "inst", T(10, 15)), R.Reading(T(13, 0), R.f_set(20), "inst", T(13, 15))]
    rs += R.readings_from_metar("KXXX 081154Z 00000KT 10SM CLR 14/12 A3001 RMK AO2 T01440122 10200 20139", T(11, 54),
                                T(11, 58))
    amb = lambda t: set(R.guard(rs, brs, t0, t1, "low", now=t)[0])  # noqa: E731
    assert amb(T(9, 0)) == set()                                    # min so far 60 or 61: one bracket
    assert amb(T(9, 45)) == {"56-57", "58-59"}
    assert amb(T(11, 57)) == {"56-57", "58-59"}
    assert amb(T(11, 58)) == set()                                  # 20139: 57 F
    assert R.extreme_candidates(rs, t0, t1, "low") == (57,)


# ------------------------------------------------------------------------------------------------ the live bot
CFG = {
    "mode": "paper", "env": "prod",
    "loop": {"tick_seconds": 10, "universe_every_s": 180, "books_every_s": 15, "trades_every_s": 20,
             "metar_every_s": 300, "nbm_every_s": 900, "index_every_s": 60, "settle_every_s": 900,
             "positions_every_s": 20},
    "sleeve_a": {"enabled": True, "series": ["KXHIGHTOKC"], "fav_min_prob": 0.78, "min_edge_c": 1.0,
                 "improve_if_edge_c": 99, "max_disagreement": 1.0, "decided_max_cost_usd": 500},
    "sleeve_b": {"enabled": False, "shadow": False},
    "risk": {"max_loss_per_bracket_usd": 30, "max_loss_per_event_usd": 40, "max_total_open_risk_usd": 350,
             "daily_loss_stop_usd": 40, "total_drawdown_stop_usd": 90},
}


def obs5_db(path):
    """The observation archive as scripts/obs5_archive.py writes it from the NWS API: 5-minute rows with no text and
    a whole-C value, hourly METAR rows with their text and decoded tenths; first_seen = when the archive saw them."""
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE obs5 (station TEXT, ts TEXT, temp_c REAL, dewpoint_c REAL, max24_c REAL, raw TEXT, "
              "first_seen REAL, PRIMARY KEY (station, ts))")
    iso = lambda t: dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")  # noqa: E731
    for t, cc in five_minute():
        c.execute("INSERT INTO obs5 VALUES (?,?,?,?,?,?,?)", ("KOKC", iso(t), float(cc), None, None, "",
                                                              t + FIVE_MIN_DELAY))
    for hm, tg, cc in HOURLY:          # the NWS API's hourly rows: 20-40 minutes late on Oct 8
        c.execute("INSERT INTO obs5 VALUES (?,?,?,?,?,?,?)", ("KOKC", iso(T(*hm)), cc, None, None,
                                                              metar_raw(hm, tg, cc), T(*hm) + 30 * 60))
    c.commit()
    c.close()


def awc_metars(now):
    """The AWC METAR feed as the bot holds it (fetch_metars: newest first), with what had reached it by now."""
    obs = [{"icaoId": "KOKC", "obsTime": int(T(*hm)), "temp": cc, "rawOb": metar_raw(hm, tg, cc)}
           for hm, tg, cc in HOURLY]
    obs.append({"icaoId": "KOKC", "obsTime": int(T(23, 52)), "temp": 28.0, "rawOb": SIX_MAX_RAW})
    return {"KOKC": sorted([o for o in obs if o["obsTime"] + METAR_DELAY <= now], key=lambda o: -o["obsTime"])}


def okc_bot(tmp_path, guard=True, **extra):
    from wxbot.app.runner import Bot
    cfg = dict(CFG, db_path=str(tmp_path / "t.sqlite"), status_path=str(tmp_path / "st.json"))
    if guard:
        cfg["round_guard"] = dict({"enabled": True, "obs5_db": str(tmp_path / "obs5.sqlite")}, **extra)
    obs5_db(str(tmp_path / "obs5.sqlite"))
    return Bot(cfg, base_dir=str(tmp_path))


def okc_event():
    from wxbot.models.fair_value import LadderQuote
    from wxbot.strategy.types import EventBook
    ms = okc_markets()
    brs = [bracket_from_market(m) for m in ms]
    # 20:57Z: the market prices 88-89 at 93%
    quotes = {f"{EV}-T86": LadderQuote(None, 1), f"{EV}-B86.5": LadderQuote(3, 5), f"{EV}-B88.5": LadderQuote(92, 94),
              f"{EV}-B90.5": LadderQuote(1, 3), f"{EV}-T91": LadderQuote(None, 1)}
    return EventBook(EV, "KXHIGHTOKC", "high", "KOKC", dt.date(2026, 10, 8), T(6, 0, day=9), ms, brs, quotes, {})


def set_clock(bot, now):
    bot.read_obs5(now)
    bot.metars = awc_metars(now)
    bot.metar_version = getattr(bot, "metar_version", 0) + 1


def test_live_guard_blocks_from_the_first_31c_reading_the_bot_has_until_the_six_hour_max(tmp_path):
    bot = okc_bot(tmp_path)
    ev = okc_event()
    seen = []
    for h, m in [(19, 9), (19, 10), (20, 0), (21, 1), (21, 7), (22, 17), (23, 55), (23, 56), (23, 59)]:
        now = T(h, m)
        set_clock(bot, now)
        seen.append((f"{h:02d}:{m:02d}", sorted(x.rsplit("-", 1)[1] for x in bot.round_guard_skip(ev, now))))
    assert seen == [("19:09", []), ("19:10", ["B86.5", "B88.5"]), ("20:00", ["B86.5", "B88.5"]),
                    ("21:01", ["B86.5", "B88.5"]), ("21:07", ["B86.5", "B88.5"]), ("22:17", ["B86.5", "B88.5"]),
                    ("23:55", ["B86.5", "B88.5"]), ("23:56", []), ("23:59", [])]
    logs = [m for _, lvl, m in bot.store.c.execute("SELECT * FROM events_log").fetchall() if "round guard" in m]
    assert len(logs) == 2                                            # one note when it blocks, one when it lifts
    assert "skips 86-87 and 88-89" in logs[0] and "87 or 88 F" in logs[0]
    assert "lifted" in logs[1] and "87 F" in logs[1]
    st = bot.round_guard_status()
    assert st["blocked"] == {}


def test_live_guard_only_on_the_target_day_and_off_by_default(tmp_path):
    bot = okc_bot(tmp_path)
    ev = okc_event()
    now = T(21, 1)
    set_clock(bot, now)
    assert bot.round_guard_skip(ev, now)
    assert bot.round_guard_skip(ev, T(21, 1, day=7)) == frozenset()     # the day before: not the target day
    (tmp_path / "off").mkdir()
    off = okc_bot(tmp_path / "off", guard=False)
    assert off.round_guard_on is False
    set_clock(off, now)
    assert off.round_guard_skip(ev, now) == frozenset()


@pytest.mark.parametrize("guard,sleeves", [(True, None), (False, None), (True, ["T"])])
def test_decide_skips_the_ambiguous_brackets_and_keeps_the_rest(tmp_path, monkeypatch, guard, sleeves):
    """Sleeve A quotes KXHIGHTOKC here; round_guard.sleeves [T] leaves Sleeve A unguarded."""
    bot = okc_bot(tmp_path, guard=guard, **({"sleeves": sleeves} if sleeves else {}))
    guard = guard and sleeves is None
    ev = okc_event()
    now = T(21, 1)                                                   # Sleeve A's fills: 21:01-21:07Z
    monkeypatch.setattr(time, "time", lambda: now)
    set_clock(bot, now)
    bot.events = {EV: ev}
    bot.book_ts = {m["ticker"]: now for m in ev.markets}
    want = {o.ticker.rsplit("-", 1)[1]: o.side for o in bot.decide()}
    oct8 = {"B86.5": "ask", "B88.5": "bid", "B90.5": "ask"}           # what filled on Oct 8 (NO, YES, NO)
    if guard:
        assert "B86.5" not in want and "B88.5" not in want and want["B90.5"] == "ask"   # NO on 90-91 stays (it won)
    else:
        assert {k: want.get(k) for k in oct8} == oct8


def test_missing_archive_is_logged_and_the_guard_falls_back_to_metars(tmp_path):
    from wxbot.app.runner import Bot
    cfg = dict(CFG, db_path=str(tmp_path / "t.sqlite"), status_path=str(tmp_path / "st.json"),
               round_guard={"enabled": True, "obs5_db": str(tmp_path / "nowhere.sqlite")})
    bot = Bot(cfg, base_dir=str(tmp_path))
    bot.read_obs5(T(21, 1))
    assert bot.obs5 == {}
    logs = [m for _, lvl, m in bot.store.c.execute("SELECT * FROM events_log").fetchall() if lvl == "WARN"]
    assert any("round guard" in m and "no observation archive" in m for m in logs)
    bot.metars = awc_metars(T(21, 1))
    assert bot.round_guard_skip(okc_event(), T(21, 1)) == frozenset()  # T-groups alone: 87 F, nothing ambiguous


def test_archive_covers_every_daily_high_station():
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location("obs5", os.path.join(os.path.dirname(__file__), "..", "scripts",
                                                                        "obs5_archive.py"))
    arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arch)
    S = json.load(open(os.path.join(os.path.dirname(__file__), "..", "wxbot", "data", "stations.json")))
    assert {v["icao"] for v in S.values()} <= set(arch.STATIONS)


# ------------------------------------------------------------------------------------------------ the replay
def test_run_joint_guard_cancels_resting_orders_and_logs_fills(tmp_path):
    """A resting YES bid on 88-89 is filled by a trade through it at 3 h unless the guard names 88-89 from 2 h on;
    the NO on 90-91 fills either way. fill_log records each fill."""
    from wxbot.backtest.joint import run_joint
    from wxbot.strategy.sleeve_a import SleeveAConfig
    t0 = T(12, 0)
    iso = lambda t: dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
    quotes = {"B86.5": (0.03, 0.05), "B88.5": (0.92, 0.94), "B90.5": (0.01, 0.03)}
    trades = {"B88.5": [{"created_time": iso(t0 + 3 * 3600), "yes_price_dollars": "0.9000", "count_fp": "50",
                         "taker_side": "no"}],
              "B90.5": [{"created_time": iso(t0 + 3 * 3600 + 60), "yes_price_dollars": "0.0500", "count_fp": "50",
                         "taker_side": "yes"}]}
    result = {"B86.5": "yes", "B88.5": "no", "B90.5": "no"}
    ms = []
    (tmp_path / "cd").mkdir()
    for m in okc_markets()[1:4]:
        k = m["ticker"].rsplit("-", 1)[1]
        ms.append(dict(m, result=result[k], open_time=iso(t0), close_time=iso(t0 + 8 * 3600), trades=trades.get(k, [])))
        b, a = quotes[k]
        with open(tmp_path / "cd" / f"{m['ticker']}.json", "w") as fh:
            json.dump([[t0 + i * 60, b, a] for i in range(0, 8 * 60, 5)], fh)
    fn = tmp_path / f"{EV}.json"
    json.dump(ms, open(fn, "w"))
    cfg = SleeveAConfig(min_edge_c=1.0, improve_if_edge_c=99, max_disagreement=1.0)
    runs = {}
    for name, g in (("off", None), ("on", lambda e, t: {f"{EV}-B88.5"} if t >= t0 + 2 * 3600 else set())):
        log = []
        r = run_joint([str(fn)], str(tmp_path / "cd"), cfg_a=cfg, fill_rule="through", guard_fn=g, fill_log=log)
        runs[name] = (r, log)
    off, on = runs["off"], runs["on"]
    assert sorted((f["ticker"].rsplit("-", 1)[1], f["side"]) for f in off[1]) == [("B88.5", "bid"), ("B90.5", "ask")]
    assert [(f["ticker"].rsplit("-", 1)[1], f["side"]) for f in on[1]] == [("B90.5", "ask")]
    f = on[1][0]
    assert f["event"] == EV and f["side"] == "ask" and f["price_cents"] == 4 and f["count"] > 0
    assert abs(f["ts"] - (t0 + 3 * 3600 + 60)) < 1
    assert on[0]["realized"] > off[0]["realized"]                    # the YES on 88-89 lost
    # without the hooks the replay is unchanged
    r_plain = run_joint([str(fn)], str(tmp_path / "cd"), cfg_a=cfg, fill_rule="through")
    assert abs(r_plain["realized"] - off[0]["realized"]) < 1e-9 and r_plain["contracts"] == off[0]["contracts"]
