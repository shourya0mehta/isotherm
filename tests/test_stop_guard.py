"""The stop guard (Oct 11, 2026; wxbot/risk/stop_guard.py): the risk keys daily_resume_margin_usd, daily_resume_hold_s,
stop_confirm_s, drawdown_confirm_s and peak_from, and `python -m wxbot set-peak`. Each key is off unless set; without
them the stops behave exactly as before.

Oct 11: Kalshi paid out the previous day's markets by time zone (Eastern about 7:10 AM ET, Central 8:08, Mountain 9:07,
West 10:10). The two losing East Coast bets paid out first, the day hit -$20.99 at 7:11 and the daily stop halted the
whole day, although by 10:10 the day stood at -$9.87. And kv peak_equity sat about $15 above any real equity: a reading taken while a payout was in the balance but its position still in the list.

The bot runs its real step() on a fake account whose settled equity is its cash plus the cost of its positions (the
balance leaves out cash held by resting orders, as Kalshi's does), one account reading a step, a minute apart. The
stops use the median of the last three reads, so a new account value shows from the second read that has it."""
import datetime as dt
import os
import random
import sqlite3
import sys
import time
from zoneinfo import ZoneInfo

import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wxbot.app.runner as runner  # noqa: E402
from test_review_round2 import Acct  # noqa: E402
from wxbot.app.alerts import Alerter  # noqa: E402
from wxbot.risk.limits import RiskConfig, RiskState  # noqa: E402
from wxbot.risk.stop_guard import StopGuard, StopGuardConfig, usd  # noqa: E402
from wxbot.store.db import Store  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
NY = ZoneInfo("America/New_York")
SOD = 644.18                    # Oct 11's start-of-day equity
GUARD = {"daily_resume_margin_usd": 3, "daily_resume_hold_s": 300, "stop_confirm_s": 300, "drawdown_confirm_s": 600,
         "peak_from": "start_of_day"}                     # the risk keys of config/live_stage4c.yaml and live_stage5c.yaml
COUNT_ONLY = {**GUARD, "daily_resume_hold_s": 0}          # the 3 readings alone decide a resume
STEP = 60.0                     # seconds between steps, one account read each; a longer gap (90 s) restarts the reads
T = "KXHIGHTDAL-26OCT10-B84.5"  # a market of yesterday that pays out this morning


def et(h, m=0, day=11):
    return dt.datetime(2026, 10, day, h, m, tzinfo=NY).timestamp()


def cfg_for(tmp_path, guard=None):
    risk = {"max_loss_per_bracket_usd": 8, "max_loss_per_event_usd": 10, "max_total_open_risk_usd": 75,
            "max_cash_deployed_usd": 200, "daily_loss_stop_usd": 20, "total_drawdown_stop_usd": 50,
            "account_stale_seconds": 90, "max_api_errors_per_min": 1000, **(guard or {})}
    return {"mode": "live", "env": "prod", "db_path": str(tmp_path / "live.sqlite"),
            "status_path": str(tmp_path / "st.json"),
            "loop": {"tick_seconds": 2, "universe_every_s": 600, "books_every_s": 5, "trades_every_s": 10,
                     "metar_every_s": 300, "nbm_every_s": 900, "index_every_s": 60, "settle_every_s": 900,
                     "positions_every_s": 5},
            "sleeve_a": {"enabled": True, "series": ["KXHIGHNY"], "decided_only": True, "decided_prob": 0.97,
                         "min_edge_c": 2.0, "improve_if_edge_c": 99, "clip_contracts": 12, "decided_max_cost_usd": 10,
                         "min_price_c": 60, "max_price_c": 98},
            "sleeve_b": {"enabled": False, "shadow": False}, "tape_archive": {"enabled": False}, "risk": risk,
            "execution": {"order_ttl_s": 1200, "ttl_by_sleeve": {"A": 300}}, "dashboard_every_s": 10 ** 9}


def live_bot(tmp_path, monkeypatch, guard=None, sod=SOD, peak=SOD, clock=None, fake=None):
    """A live bot at 7:00 ET on Oct 11, its start-of-day equity and high point stored; phone alerts are recorded
    (bot.alerts.sent) instead of posted. Pass clock and fake to restart a bot on the same account and database."""
    if clock is None:
        clock = [et(7)]
        monkeypatch.setattr(time, "time", lambda: clock[0])
        key = Ed25519PrivateKey.generate().private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                         serialization.NoEncryption()).decode()
        monkeypatch.setenv("KALSHI_KEY_ID", "kid")
        monkeypatch.setenv("KALSHI_PRIVATE_KEY", key)
        monkeypatch.setattr(runner.obs_mod, "fetch_metars", lambda *a, **kw: {})
    if fake is None:
        fake = Acct()
        fake.cash = sod
    bot = runner.Bot(cfg_for(tmp_path, guard), base_dir=str(tmp_path))
    bot.store.c.execute("PRAGMA synchronous = OFF")   # a test database needs no flush to disk: each step commits twice
    bot.rest = fake
    bot.live.rest = fake
    bot.nbm.refresh = lambda *a, **kw: None
    if bot.store.kv_get("sod_equity") is None:
        bot.store.kv_set("balance_semantics", {"excludes_resting": True})
        bot.store.kv_set("sod_equity", {"day": "2026-10-11", "equity": sod})
        bot.store.kv_set("peak_equity", peak)
    bot.alerts = Alerter({"ntfy_topic": "wxbot-test"})
    bot.alerts._post = lambda *a, **kw: True
    return clock, fake, bot


def steps(clock, bot, n, step=STEP):
    """n steps, a minute apart (or `step` seconds): [(time, halted, today's result)] after each."""
    out = []
    for _ in range(n):
        clock[0] += step
        bot.step()
        out.append((clock[0], bot.risk.halted, round(bot.risk.realized_today, 2)))
    return out


def run(clock, bot, until):
    """Steps until the clock reaches `until`."""
    return steps(clock, bot, max(0, int(-(-(until - clock[0]) // STEP))))


def first(trace, cond):
    return next(t for t, h, rt in trace if cond(h, rt))


def logged(bot, like, level="WARN"):
    return [m for (m,) in bot.store.c.execute("SELECT msg FROM events_log WHERE level = ? AND msg LIKE ? ORDER BY ts",
                                              (level, like))]


def alerts(bot, title_end):
    return [m for t, m in bot.alerts.sent if t.endswith(title_end)]


def halted_at_seven(clock, fake, bot):
    """Oct 11 up to the daily halt: Eastern pays out the two losing bets at 7:10 and the day reads -$20.99."""
    run(clock, bot, et(7, 10))
    fake.cash = SOD - 20.99
    trace = run(clock, bot, et(7, 20))
    assert bot.risk.halted and bot.risk.halt_kind == "daily"
    return trace


# ------------------------------------------------------------------------------------- (1) the Oct 11 morning
def test_partial_payouts_trip_the_daily_stop_and_the_day_recovering_lifts_it(tmp_path, monkeypatch):
    clock, fake, bot = live_bot(tmp_path, monkeypatch, GUARD)
    trace = halted_at_seven(clock, fake, bot)
    below, halt = first(trace, lambda h, rt: rt <= -20), first(trace, lambda h, rt: h)
    assert halt - below == 300                                  # the condition held 5 minutes of readings first
    assert bot.risk.halt_reason == ("daily loss stop hit (-20.99) for 5 min; trading resumes today if the day's "
                                    "settled result recovers to -$17 for 5 min")
    assert bot.store.kv_get("halt_latch")["kind"] == "daily" and len(alerts(bot, "HALT (daily)")) == 1
    clock[0] = et(8, 8)                                         # Central pays out: still below the stop
    fake.cash = SOD - 19.40
    run(clock, bot, et(8, 20))
    clock[0] = et(9, 7)                                         # Mountain: above the stop, below the -$17 line
    fake.cash = SOD - 17.60
    run(clock, bot, et(9, 20))
    assert bot.risk.halted and bot.stop_guard.above == 0 and not alerts(bot, "resumed")
    clock[0] = et(10, 10)                                       # the West pays out: the day is -$9.87
    fake.cash = SOD - 9.87
    trace = steps(clock, bot, 10)
    # the reads start over after the gap: the first median at the third read (10:13), then five minutes of readings
    # above the line from there (the three in a row come first)
    assert [h for t, h, rt in trace[:8]] == [True] * 7 + [False] and not bot.risk.halted
    assert trace[7][0] == et(10, 18)
    assert bot.store.kv_get("halt_latch") == {} and bot.store.kv_get("daily_resumes")["n"] == 1
    msg = "Resumed: today's settled result recovered to -$9.87 (daily stop -$20; automatic resume 1 of 2 today)"
    assert alerts(bot, "resumed") == [msg]                      # one phone alert ...
    assert logged(bot, "Resumed:%") == [msg]                    # ... and one WARN line
    assert bot._write_status()["stop_guard"]["resumes_today"] == 1


# ------------------------------------------------------------------- the resume hold time (daily_resume_hold_s)
def test_a_20_second_recovery_does_not_lift_the_halt_and_a_5_minute_one_does(tmp_path, monkeypatch):
    """Reads every 5 seconds, as live: three readings in a row back above -$17 take about 20 seconds, as long as a
    phantom reading can last; daily_resume_hold_s 300 asks for 5 minutes from the first of them."""
    clock, fake, bot = live_bot(tmp_path, monkeypatch, GUARD)
    halted_at_seven(clock, fake, bot)
    seen = []
    for cash in [SOD - 9.87] * 4 + [SOD - 18.0] * 4:           # 20 seconds of reads at -$9.87, then -$18 again
        fake.cash = cash
        seen += [(h, bot.stop_guard.above) for t, h, rt in steps(clock, bot, 1, step=5)]
    assert max(a for h, a in seen) == 4 and all(h for h, a in seen)   # 3 in a row were reached: the hold kept it
    assert bot.stop_guard.above == 0 and not alerts(bot, "resumed")
    fake.cash = SOD - 9.87                                      # now the day stays at -$9.87
    trace = []
    for _ in range(70):
        t, h, rt = steps(clock, bot, 1, step=5)[0]
        trace.append((t, h, bot.stop_guard.above))
    began = next(t for t, h, a in trace if a == 1)              # the first reading back above the line
    resumed = next(t for t, h, a in trace if not h)
    assert resumed - began == 300 and len(alerts(bot, "resumed")) == 1
    assert bot.store.kv_get("halt_latch") == {} and bot.store.kv_get("daily_resumes")["n"] == 1


# --------------------------------------------------------------------------------- (2) two readings are not 3
def test_two_readings_back_above_the_line_leave_it_halted(tmp_path, monkeypatch):
    clock, fake, bot = live_bot(tmp_path, monkeypatch, COUNT_ONLY)
    halted_at_seven(clock, fake, bot)
    clock[0] = et(10, 10)
    seen = []
    for cash in [SOD - 18.0] * 4 + [SOD - 9.87] * 2 + [SOD - 18.0] * 2 + [SOD - 9.87] * 3:
        fake.cash = cash
        steps(clock, bot, 1)
        seen.append((bot.stop_guard.above, bot.risk.halted))
    # two reads at -$9.87 make two readings (medians) above the line; the low reads after them end the run
    assert [a for a, h in seen] == [0, 0, 0, 0, 0, 1, 2, 0, 0, 1, 2] and all(h for a, h in seen)
    assert not alerts(bot, "resumed") and bot.store.kv_get("halt_latch")["kind"] == "daily"
    steps(clock, bot, 1)                                        # a third reading in a row: resumed
    assert not bot.risk.halted and len(alerts(bot, "resumed")) == 1


# ------------------------------------------------------------------------ (3) the daily stop's confirmation
def test_a_short_dip_below_the_daily_stop_does_not_halt_and_a_persistent_one_does(tmp_path, monkeypatch):
    clock, fake, bot = live_bot(tmp_path, monkeypatch, GUARD)
    fake.pos, fake.cash = {T: (-26.0, 25.0)}, SOD - 25.0        # a $25 NO position from yesterday
    run(clock, bot, et(7, 10))
    fake.pos = {}                                               # it leaves the positions list before its $26 lands
    dip = steps(clock, bot, 3)
    assert [rt for t, h, rt in dip] == [0.0, -25.0, -25.0] and not any(h for t, h, rt in dip)
    fake.cash += 26.0                                           # the payout lands three minutes later
    back = steps(clock, bot, 5)
    assert not any(h for t, h, rt in back) and bot.risk.realized_today == pytest.approx(1.0)
    assert logged(bot, "daily loss stop%") == [
        "daily loss stop reached (-25.00 today): it halts if it still holds after 5 min of readings",
        "daily loss stop no longer reached (1.00 today) after 3 min of readings: not halted"]
    fake.cash -= 22.0                                           # real losses: -$21 for good
    trace = steps(clock, bot, 10)
    below, halt = first(trace, lambda h, rt: rt <= -20), first(trace, lambda h, rt: h)
    assert halt - below == 300 and bot.risk.halt_kind == "daily"
    assert bot.risk.halt_reason.startswith("daily loss stop hit (-21.00) for 5 min; ")


# --------------------------------------------------------------------- (4) the drawdown stop's confirmation
def test_a_short_drawdown_dip_does_not_hard_halt_and_a_persistent_one_does(tmp_path, monkeypatch):
    clock, fake, bot = live_bot(tmp_path, monkeypatch, GUARD, sod=655.0, peak=700.0)
    fake.pos, fake.cash = {T: (10.0, 5.0)}, 647.0               # equity 652: $48 below the high point, -$3 today
    run(clock, bot, et(7, 10))
    assert bot.risk.peak_equity == 700.0 and bot.risk.equity == pytest.approx(652.0)
    fake.pos = {}                                               # a winning position leaves the list before its $10
    dip = steps(clock, bot, 8)                                  # $53 below the high point for 7 readings (6 minutes)
    assert max(700.0 - (655.0 + rt) for t, h, rt in dip) == pytest.approx(53.0) and not any(h for t, h, rt in dip)
    fake.cash += 10.0                                           # the payout lands
    steps(clock, bot, 5)
    assert not bot.risk.halted and logged(bot, "drawdown stop%") == [
        "drawdown stop reached (53.00 below the high point): it halts if it still holds after 10 min of readings",
        "drawdown stop no longer reached (43.00 below the high point) after 8 min of readings: not halted"]
    fake.cash -= 12.0                                           # real losses: $55 below the high point for good
    trace = steps(clock, bot, 15)
    halt = first(trace, lambda h, rt: h)
    past = first(trace, lambda h, rt: 700.0 - (655.0 + rt) >= 50)
    assert halt - past == 600 and bot.risk.halt_kind == "hard"
    assert bot.risk.halt_reason == "drawdown stop hit (55.00) for 10 min - manual review needed"
    fake.cash += 30.0                                           # the day recovers: a hard halt is never lifted
    steps(clock, bot, 6)
    assert bot.risk.halted and bot.risk.halt_kind == "hard" and not alerts(bot, "resumed")


def test_a_gap_in_the_reads_keeps_a_stops_timer_and_restarts_the_count_toward_a_resume(tmp_path, monkeypatch):
    clock, fake, bot = live_bot(tmp_path, monkeypatch, COUNT_ONLY)
    run(clock, bot, et(7, 10))
    fake.cash = SOD - 21.0
    steps(clock, bot, 3)                                        # below the stop on the readings of 7:12 and 7:13
    clock[0] = et(7, 20)                                        # no account reads for 7 minutes (an outage)
    trace = steps(clock, bot, 4)
    # the reads start over: the first median at the third read (7:23), still below the stop, 11 minutes after 7:12
    assert [h for t, h, rt in trace] == [False, False, True, True]
    assert bot.risk.halt_reason.startswith("daily loss stop hit (-21.00) for 11 min; ")
    fake.cash = SOD - 9.87
    steps(clock, bot, 3)                                        # two readings back above the -$17 line ...
    assert bot.stop_guard.above == 2 and bot.risk.halted
    clock[0] += 300                                             # ... then a gap: the count starts over
    trace = steps(clock, bot, 5)
    assert [h for t, h, rt in trace] == [True] * 4 + [False]


# ------------------------------------------------------------------------------- (5) peak_from: start_of_day
def test_with_peak_from_start_of_day_only_a_new_days_start_raises_the_high_point(tmp_path, monkeypatch):
    clock, fake, bot = live_bot(tmp_path, monkeypatch, {"peak_from": "start_of_day"})
    fake.pos, fake.cash = {T: (15.58, 10.0)}, SOD - 10.0        # equity 644.18
    run(clock, bot, et(7, 10))
    fake.cash += 15.58                                          # the payout is in the balance, the position still listed
    steps(clock, bot, 3)
    assert bot.risk.equity == pytest.approx(659.76)             # the Oct 11 phantom reading ...
    assert bot.risk.peak_equity == SOD and bot.store.kv_get("peak_equity") == SOD   # ... does not raise the high point
    fake.pos = {}                                               # the position leaves the list: equity 649.76
    steps(clock, bot, 3)
    assert bot.risk.equity == pytest.approx(649.76) and bot.risk.peak_equity == SOD
    clock[0] = et(0, 0, day=12) + 30                            # the next ET day: its start-of-day equity raises it
    steps(clock, bot, 4)
    assert bot.store.kv_get("sod_equity") == {"day": "2026-10-12", "equity": pytest.approx(649.76)}
    assert bot.risk.peak_equity == pytest.approx(649.76) and bot.store.kv_get("peak_equity") == pytest.approx(649.76)
    fake.cash += 8.0                                            # an intraday gain on Oct 12 does not
    steps(clock, bot, 4)
    assert bot.risk.equity == pytest.approx(657.76) and bot.risk.peak_equity == pytest.approx(649.76)
    assert not bot.risk.halted and bot._write_status()["stop_guard"]["peak_from"] == "start_of_day"


# ---------------------------------------------------------------------------- (6) without the keys: as before
def test_without_the_keys_the_stops_and_the_peak_behave_as_before(tmp_path, monkeypatch):
    clock, fake, bot = live_bot(tmp_path, monkeypatch)
    assert bot.stop_guard.cfg == StopGuardConfig() and not bot.stop_guard.cfg.enabled
    fake.pos, fake.cash = {T: (15.58, 10.0)}, SOD - 10.0
    run(clock, bot, et(7, 5))
    fake.cash += 15.58                                          # the phantom reading raises the high point, as before
    steps(clock, bot, 3)
    assert bot.risk.peak_equity == pytest.approx(659.76) and bot.store.kv_get("peak_equity") == pytest.approx(659.76)
    fake.pos, fake.cash = {}, SOD - 20.99                       # the day at -$20.99: halted at the first median
    trace = steps(clock, bot, 3)
    assert [h for t, h, rt in trace] == [False, True, True]
    assert bot.risk.halt_reason == "daily loss stop hit (-20.99)" and bot.store.kv_get("halt_latch")["kind"] == "daily"
    clock[0] = et(10, 10)                                       # the day recovering lifts nothing
    fake.cash = SOD - 9.87
    steps(clock, bot, 6)
    assert bot.risk.halted and bot.risk.halt_kind == "daily" and not alerts(bot, "resumed")
    assert "stop_guard" not in bot._write_status() and bot.store.kv_get("daily_resumes") is None
    assert logged(bot, "%stop%reached%") == []
    clock[0] = et(0, 0, day=12) + 30                            # the next ET day resumes it, as before
    steps(clock, bot, 1)
    assert not bot.risk.halted and logged(bot, "resumed after halt", "INFO") == ["resumed after halt"]


def test_the_guard_with_nothing_set_halts_exactly_as_check_stops():
    """The guarded path with no confirmation and no margin: the same stop, the same reason, on any numbers."""
    rng = random.Random(11)
    g = StopGuard(StopGuardConfig())
    for _ in range(2000):
        cfg = RiskConfig(daily_loss_stop_usd=rng.choice([15, 20, 40]), total_drawdown_stop_usd=rng.choice([40, 50]))
        r = RiskState(cfg, realized_today=round(rng.uniform(-45, 10), 2), peak_equity=round(rng.uniform(600, 700), 2),
                      equity=round(rng.uniform(580, 700), 2))
        kind = g.due(r)
        ref = RiskState(cfg, realized_today=r.realized_today, peak_equity=r.peak_equity, equity=r.equity)
        assert (kind is not None) == ref.check_stops("d")
        if kind is not None:
            assert kind == ref.halt_kind and g.reason(kind, r, 2) == ref.halt_reason


# ------------------------------------------------------------------- the limits: two resumes, hard halts never
def test_at_most_two_automatic_resumes_a_day_even_across_a_restart(tmp_path, monkeypatch):
    clock, fake, bot = live_bot(tmp_path, monkeypatch, GUARD)
    halted_at_seven(clock, fake, bot)
    for n in (1, 2):
        clock[0] = et(7 + n)                                    # an hour later the day is back at -$9 (for 5 min) ...
        fake.cash = SOD - 9.0
        steps(clock, bot, 8)
        assert not bot.risk.halted and bot.store.kv_get("daily_resumes")["n"] == n
        if n == 2:                                              # (a restart between the second resume and the third
            clock, fake, bot = live_bot(tmp_path, monkeypatch, GUARD, clock=clock, fake=fake)     # halt)
        fake.cash = SOD - 22.0                                  # ... and then at -$22 for good
        steps(clock, bot, 9)
        assert bot.risk.halted and bot.risk.halt_kind == "daily"
    assert bot.risk.halt_reason == "daily loss stop hit (-22.00) for 5 min; no automatic resume left today (2 used)"
    fake.cash = SOD - 5.0
    steps(clock, bot, 8)
    assert bot.risk.halted and bot.store.kv_get("halt_latch")["kind"] == "daily" and not alerts(bot, "resumed")
    assert len(logged(bot, "Resumed:%")) == 2 and len(logged(bot, "HALT: daily loss stop hit%", "ERROR")) == 3
    clock[0] = et(0, 0, day=12) + 30                            # a new ET day: lifted, and two resumes again
    steps(clock, bot, 1)
    assert not bot.risk.halted and bot.daily_resumes("2026-10-12") == 0


def test_a_halt_after_a_resume_alerts_at_once(tmp_path, monkeypatch):
    clock, fake, bot = live_bot(tmp_path, monkeypatch, {"daily_resume_margin_usd": 3})
    run(clock, bot, et(7, 10))
    fake.cash = SOD - 20.99
    steps(clock, bot, 2)                                        # no confirmation set: halted at the first median
    fake.cash = SOD - 9.87
    steps(clock, bot, 4)                                        # three readings back above -$17: resumed
    fake.cash = SOD - 20.5
    steps(clock, bot, 2)                                        # halted again 6 minutes after the first halt
    assert bot.risk.halted and len(alerts(bot, "resumed")) == 1
    # the alerts keep 10 minutes between two of the same kind; the resume lets the next halt through at once
    assert [m.split(";")[0] for m in alerts(bot, "HALT (daily)")] == ["daily loss stop hit (-20.99)",
                                                                       "daily loss stop hit (-20.50)"]


def test_a_daily_halt_does_not_lift_while_the_drawdown_stop_holds(tmp_path, monkeypatch):
    """The day back above the line while equity stays $50 below the high point: still halted for the day (stops are
    not judged during a halt, as before; the drawdown stop then halts after midnight)."""
    clock, fake, bot = live_bot(tmp_path, monkeypatch, GUARD, sod=655.0, peak=700.0)
    run(clock, bot, et(7, 10))
    fake.cash = 634.0                                           # -$21 today, $66 below the high point
    steps(clock, bot, 8)
    assert bot.risk.halt_kind == "daily"                        # the daily stop confirms first (5 min against 10)
    fake.cash = 640.0                                           # -$15 today: above the -$17 line, but $60 below the peak
    steps(clock, bot, 10)
    assert bot.risk.halted and bot.risk.halt_kind == "daily" and bot.stop_guard.above == 0
    assert not alerts(bot, "resumed") and not logged(bot, "Resumed:%")
    clock[0] = et(0, 0, day=12) + 30                            # the next ET day: the daily halt lapses and the
    steps(clock, bot, 3)                                        # drawdown stop, held since yesterday, halts at once
    assert bot.risk.halt_kind == "hard" and bot.risk.halt_reason.startswith("drawdown stop hit (60.00) for ")


def test_a_manual_halt_is_never_lifted_automatically(tmp_path, monkeypatch):
    clock, fake, bot = live_bot(tmp_path, monkeypatch, GUARD)
    halted_at_seven(clock, fake, bot)
    open(tmp_path / "QUIESCE", "w").close()                     # stopped by hand (BACK_TO_PAPER) during the daily halt
    steps(clock, bot, 1)
    assert bot.risk.halt_kind == "hard"
    clock[0] = et(10, 10)
    fake.cash = SOD - 9.87
    steps(clock, bot, 8)
    assert bot.risk.halted and bot.risk.halt_kind == "hard" and not alerts(bot, "resumed")
    assert bot.stop_guard.above == 0 and bot.store.kv_get("halt_latch")["kind"] == "hard"


# ---------------------------------------------------------------------------------------------- the configs
def test_the_configs():
    assert StopGuardConfig.from_cfg(None) == StopGuardConfig() == StopGuardConfig.from_cfg({"daily_loss_stop_usd": 20})
    assert StopGuardConfig.from_cfg({"daily_resume_margin_usd": None, "stop_confirm_s": 0}) == StopGuardConfig()
    for bad in ({"peak_from": "readings"}, {"stop_confirm_s": -1}, {"daily_resume_margin_usd": -3},
                {"daily_resume_hold_s": -1}):
        with pytest.raises(ValueError):
            StopGuardConfig.from_cfg(bad)
    on = StopGuardConfig(daily_resume_margin_usd=3.0, stop_confirm_s=300.0, drawdown_confirm_s=600.0,
                         peak_from="start_of_day", daily_resume_hold_s=300.0)
    folder = os.path.join(ROOT, "config")
    for name in sorted(os.listdir(folder)):
        if name.endswith(".yaml"):
            c = StopGuardConfig.from_cfg((yaml.safe_load(open(os.path.join(folder, name))) or {}).get("risk"))
            # Stages 4c and 5c and the practice run; every other config exactly as before
            want = on if name in ("live_stage4c.yaml", "live_stage5c.yaml", "live_stage4d.yaml", "live_stage4e.yaml",
                                  "demo.yaml") \
                else StopGuardConfig()
            assert c == want, name
    for stage in ("4", "5"):            # the c configs: their stage plus the five keys and the same-weather budget only
        base = yaml.safe_load(open(os.path.join(folder, f"live_stage{stage}.yaml")))
        new = yaml.safe_load(open(os.path.join(folder, f"live_stage{stage}c.yaml")))
        sw = new.pop("same_weather")
        assert {**base, "risk": {**base["risk"], **GUARD}} == new
        assert sw == {"enabled": True, "budget_usd": 20, "sleeves": ["A", "T"],
                      "groups": [["KXHIGHPHIL", "KXHIGHTTTN", "KXHIGHTEWR", "KXHIGHNY"], ["KXHIGHAUS", "KXHIGHTSATX"]]}
    # Stage 4d (Oct 12) = Stage 4c with the four newer cities off, nothing else
    c4 = yaml.safe_load(open(os.path.join(folder, "live_stage4c.yaml")))
    d4 = yaml.safe_load(open(os.path.join(folder, "live_stage4d.yaml")))
    assert c4["sleeve_t"]["enabled"] is True and d4["sleeve_t"]["enabled"] is False
    assert {**c4, "sleeve_t": {**c4["sleeve_t"], "enabled": False}} == d4
    # Stage 4e (Oct 12) = Stage 4d with the newer cities' maker on for San Diego only, and its switch-off
    e4 = yaml.safe_load(open(os.path.join(folder, "live_stage4e.yaml")))
    assert e4.pop("shutoff") == {"enabled": True, "sleeves": {"T": {"k": 0, "h": 15, "since": __import__("datetime").date(2026, 10, 14)}}}
    assert {**d4, "sleeve_t": {**d4["sleeve_t"], "enabled": True, "series": ["KXHIGHTSAN"]}} == e4
    assert usd(-9.87) == "-$9.87" and usd(-20) == "-$20" and usd(1.5) == "+$1.50"


# --------------------------------------------------------------------------------------------- (7) set-peak
def table_rows(path):
    c = sqlite3.connect(path)
    out = {t: sorted(map(repr, c.execute(f"SELECT * FROM {t}")))
           for (t,) in c.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    c.close()
    return out


def test_set_peak_changes_only_the_high_point(tmp_path, capsys):
    from wxbot.__main__ import main
    db = tmp_path / "data" / "live.sqlite"
    os.makedirs(db.parent)
    st = Store(str(db))
    st.kv_set("peak_equity", 659.76)
    st.kv_set("sod_equity", {"day": "2026-10-11", "equity": SOD})
    st.kv_set("halt_latch", {"kind": "hard", "reason": "drawdown stop hit (50.10) - manual review needed"})
    st.c.close()
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"db_path: {db}\nstatus_path: {tmp_path / 'data' / 'st.json'}\n")
    before = table_rows(db)
    main(["set-peak", "--to-sod", "--config", str(cfg)])
    out = capsys.readouterr().out
    assert "peak_equity 659.76 -> 644.18 (the start-of-day equity of 2026-10-11)" in out and "reset-halt" in out
    after = table_rows(db)
    assert Store(str(db)).kv_get("peak_equity") == SOD
    assert [r for r in after["kv"] if "peak_equity" not in r] == [r for r in before["kv"] if "peak_equity" not in r]
    assert Store(str(db)).kv_get("halt_latch")["kind"] == "hard"                # no halt is cleared ...
    assert not os.path.exists(tmp_path / "data" / "RESTART")                   # ... and the bot not restarted
    assert {t: v for t, v in after.items() if t not in ("kv", "events_log")} == \
        {t: v for t, v in before.items() if t not in ("kv", "events_log")}
    new_log = [r for r in after["events_log"] if r not in before["events_log"]]
    assert len(new_log) == 1 and "set by hand: 659.76 -> 644.18" in new_log[0]
    main(["set-peak", "--value", "650.5", "--config", str(cfg)])
    assert Store(str(db)).kv_get("peak_equity") == 650.5 and "644.18 -> 650.50" in capsys.readouterr().out
    assert len(table_rows(db)["kv"]) == len(before["kv"])
    for argv in (["set-peak"], ["set-peak", "--to-sod", "--value", "640"], ["set-peak", "--value", "-5"],
                 ["set-peak", "--value", "nan"]):
        with pytest.raises(SystemExit):
            main(argv + ["--config", str(cfg)])
    assert Store(str(db)).kv_get("peak_equity") == 650.5                        # refused: nothing changed
    missing = tmp_path / "elsewhere" / "live.sqlite"                            # a wrong --config: refused, and no
    cfg2 = tmp_path / "c2.yaml"                                                 # database (or folder) is created
    cfg2.write_text(f"db_path: {missing}\n")
    with pytest.raises(SystemExit) as e:
        main(["set-peak", "--to-sod", "--config", str(cfg2)])
    assert "no bot database" in str(e.value) and not os.path.exists(missing.parent)
    st = Store(str(tmp_path / "fresh.sqlite"))                                  # no start-of-day equity yet
    st.kv_set("peak_equity", 640.0)
    (tmp_path / "c3.yaml").write_text(f"db_path: {tmp_path / 'fresh.sqlite'}\n")
    with pytest.raises(SystemExit):
        main(["set-peak", "--to-sod", "--config", str(tmp_path / "c3.yaml")])
    assert st.kv_get("peak_equity") == 640.0
