"""Per-sleeve automatic shut-off (Oct 9; holdout amendments draft A12): a one-sided CUSUM on each live sleeve's daily
realized P&L, S_t = max(0, S_(t-1) + k - x_t), from the bot's own fills and settlements. When S reaches h the sleeve
opens no new positions (open ones are held), the status file and an alert say so, and only reset-shutoff turns it back
on. Off by default."""
import datetime as dt
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "research", "ml"))

from wxbot.risk import shutoff as S  # noqa: E402
from wxbot.store.db import Store  # noqa: E402
from wxbot.strategy.types import DesiredOrder, Fill, Position  # noqa: E402

U = lambda *a: dt.datetime(*a, tzinfo=dt.timezone.utc).timestamp()   # noqa: E731
DRAFT = {"A": (1.68, 60.39), "T": (1.72, 21.09), "X": (0.94, 48.67), "E": (0.88, 55.29), "R": (0.74, 34.27)}


# ------------------------------------------------------------------------------------------------ the statistic
def test_the_statistic_on_synthetic_series():
    assert S.cusum([0, 0, 0], 1.0) == [1.0, 2.0, 3.0] and S.first_crossing([0, 0, 0], 1.0, 3.0) == 2
    assert S.cusum([5, -1, -1, 4, -2], 1.0) == [0.0, 2.0, 4.0, 1.0, 4.0]    # never below 0; a good day pays it down
    assert S.first_crossing([5, -1, -1, 4, -2], 1.0, 4.5) is None
    k, h = DRAFT["X"]
    assert S.first_crossing([0.0] * 60, k, h) == 51                   # nothing settled: day 52 (52 x 0.94 = 48.88)
    assert S.first_crossing([-2.0] * 60, k, h) == 16                  # losing $2 a day: day 17 (17 x 2.94 = 49.98)
    assert S.first_crossing([2 * k] * 400, k, h) is None              # earning twice k: never
    for s, (k, h) in DRAFT.items():                                   # an edge of exactly k a day never drifts up
        assert max(S.cusum([k] * 200, k)) == 0.0


def test_the_statistic_matches_the_calibration_code():
    """wxbot's statistic and research/ml/r9_shutoff.py (the code that set k and h) switch off on the same day."""
    import r9_shutoff as R9
    rng = np.random.default_rng(3)
    for s, (k, h) in DRAFT.items():
        X = rng.normal(0.5, 6.0, (300, 120))
        T = R9.first_cross(X, k, h)
        for i in range(len(X)):
            f = S.first_crossing(X[i], k, h)
            assert (f is None and not np.isfinite(T[i])) or (f is not None and f + 1 == T[i])
        assert np.allclose(R9.cusum_max(X, k), [max(S.cusum(x, k)) for x in X])


def test_config_and_days():
    assert S.ShutoffConfig.from_cfg(None) == S.ShutoffConfig() and not S.ShutoffConfig().enabled
    assert S.ShutoffConfig.from_cfg({"enabled": False, "sleeves": {"X": {"k": 1}}}).sleeves == {}   # off: not read
    c = S.ShutoffConfig.from_cfg({"enabled": True, "sleeves": {"x": {"k": 0.94, "h": 48.67, "since": "2026-10-09"},
                                                               "E": {"k": 0.88, "h": 55.29,
                                                                     "since": dt.date(2026, 10, 9)},
                                                               "R": {"k": 0.74, "h": 34.27,
                                                                     "since": "2026-10-09T02:25:00Z"}}})
    assert c.enabled and set(c.sleeves) == {"X", "E", "R"} and c.sleeves["X"].since == U(2026, 10, 9)
    assert c.sleeves["E"].since == U(2026, 10, 9) and c.sleeves["R"].since == U(2026, 10, 9, 2, 25)
    with pytest.raises(ValueError):
        S.ShutoffConfig.from_cfg({"enabled": True, "sleeves": {"A": {"k": 1.68}}})
    since = U(2026, 10, 9, 15)
    assert S.complete_days(since, U(2026, 10, 12, 0, 30)) == ["2026-10-09", "2026-10-10"]   # Oct 11 + 1 h grace
    assert S.complete_days(since, U(2026, 10, 12, 1, 0)) == ["2026-10-09", "2026-10-10", "2026-10-11"]
    assert S.complete_days(since, U(2026, 10, 9, 20)) == []


def fill(store, mode, sleeve, ticker, side, pc, n, fee, ts):
    store.add_fill(mode, Fill(ts, ticker, ticker.rsplit("-", 1)[0], side, pc, n, sleeve, False, fee,
                              f"{sleeve}{ticker}{side}{ts}"))


def test_daily_pnl_comes_from_the_bots_own_fills_and_settlements(tmp_path):
    st = Store(str(tmp_path / "d.sqlite"))
    fill(st, "live", "X", "KXHIGHNY-26OCT10-B70.5", "bid", 30, 5, 0.08, U(2026, 10, 9, 21, 5))   # YES at 30c: wins
    fill(st, "live", "X", "KXHIGHNY-26OCT10-B72.5", "ask", 40, 5, 0.09, U(2026, 10, 9, 21, 5))   # NO at 60c: loses
    fill(st, "live", "A", "KXHIGHNY-26OCT10-T75", "ask", 3, 10, 0.0, U(2026, 10, 9, 22))        # another sleeve
    fill(st, "paper", "X", "KXHIGHNY-26OCT10-B70.5", "bid", 30, 5, 0.08, U(2026, 10, 9, 21, 5))  # another mode
    fill(st, "live", "X", "KXHIGHNY-26OCT11-B70.5", "bid", 30, 5, 0.08, U(2026, 10, 10, 21, 5))  # not settled
    st.set_settlement("KXHIGHNY-26OCT10-B70.5", "KXHIGHNY-26OCT10", "yes", U(2026, 10, 11, 14), None)
    st.set_settlement("KXHIGHNY-26OCT10-B72.5", "KXHIGHNY-26OCT10", "yes", U(2026, 10, 11, 14), None)
    st.set_settlement("KXHIGHNY-26OCT10-T75", "KXHIGHNY-26OCT10", "no", U(2026, 10, 11, 14), None)
    got = S.daily_pnl(st.c, "live", "X", U(2026, 10, 9), U(2026, 10, 13))
    assert got == {"2026-10-11": pytest.approx(5 * 0.70 - 0.08 + 5 * -0.60 - 0.09)}
    assert S.daily_pnl(st.c, "live", "A", U(2026, 10, 9), U(2026, 10, 13)) == {"2026-10-11": pytest.approx(0.30)}
    assert S.daily_pnl(st.c, "live", "X", U(2026, 10, 11, 15), U(2026, 10, 13)) == {}   # settled before since


# ------------------------------------------------------------------------------------------------ in the bot
class Alerts:
    heartbeat_topic = ""

    def __init__(self):
        self.sent = []
        self.enabled = False

    def send(self, key, title, msg, **kw):
        self.sent.append((key, title, msg))
        return True


def cfg_for(tmp_path, shutoff=None, mode="paper"):
    c = {"mode": mode, "env": "prod", "db_path": str(tmp_path / "s.sqlite"), "status_path": str(tmp_path / "st.json"),
         "sleeve_a": {"enabled": True, "series": ["KXHIGHNY"]}, "sleeve_b": {"enabled": False, "shadow": False},
         "tape_archive": {"enabled": False}, "loop": {"tick_seconds": 2}}
    if shutoff is not None:
        c["shutoff"] = shutoff
    return c


SHUT = {"enabled": True, "sleeves": {s: {"k": k, "h": h, "since": "2026-10-09"} for s, (k, h) in DRAFT.items()}}


def lose(store, sleeve, day, amount, mode="paper"):
    """A settled loss of `amount` dollars for the sleeve (a gain when negative), booked on that UTC day: NO bought at
    50c on 2 x |amount| contracts, settled against it (YES) or for it (NO)."""
    t = f"KXHIGHNY-26OCT{day:02d}-{'L' if amount > 0 else 'W'}{sleeve}{abs(amount):g}"
    fill(store, mode, sleeve, t, "ask", 50, 2 * abs(amount), 0.0, U(2026, 10, day - 1, 21))
    store.set_settlement(t, t.rsplit("-", 1)[0], "yes" if amount > 0 else "no", U(2026, 10, day, 14), None)


def test_a_sleeve_is_switched_off_on_the_day_its_statistic_reaches_h_and_stays_off_across_a_restart(tmp_path):
    from wxbot.app.runner import Bot
    bot = Bot(cfg_for(tmp_path, SHUT), base_dir=str(tmp_path))
    bot.alerts = Alerts()
    for day in range(9, 14):                     # X loses $10 a day from Oct 9: S = 10.94, 21.88, 32.82, 43.76, 54.70
        lose(bot.store, "X", day, 10.0)
    lose(bot.store, "A", 10, 5.0)                # A loses once: S 6.68 at most, far from 60.39
    bot.check_shutoff(U(2026, 10, 13, 12))       # Oct 13 not complete yet: 4 days, S 43.76
    assert bot.shutoff_blocked() == set() and bot.shutoff_state["X"]["S"] == pytest.approx(43.76)
    assert bot.shutoff_state["X"]["days"] == 4 and bot.alerts.sent == []
    bot.check_shutoff(U(2026, 10, 14, 1, 5))     # Oct 13 complete (after the 1 h grace)
    assert bot.shutoff_blocked() == {"X"} and bot.shutoff_state["X"]["switched_off_day"] == "2026-10-13"
    assert bot.shutoff_state["X"]["S"] == pytest.approx(54.70) and not bot.shutoff_state["A"]["switched_off"]
    assert len(bot.alerts.sent) == 1 and bot.alerts.sent[0][0] == "shutoff:X" and "reset-shutoff" in \
        bot.alerts.sent[0][2]
    assert bot.store.c.execute("SELECT COUNT(*) FROM events_log WHERE level = 'ERROR' AND msg LIKE "
                               "'Sleeve X switched off%'").fetchone()[0] == 1
    st = bot._write_status()["shutoff"]
    assert st["switched_off"] == ["X"] and st["sleeves"]["X"]["h"] == 48.67 and "reset-shutoff" in st["reset"]
    # good days afterwards pull S back under h: the sleeve stays off until reset; no second alert
    for day in range(14, 18):
        lose(bot.store, "X", day, -20.0)
    bot.check_shutoff(U(2026, 10, 18, 2))
    assert bot.shutoff_state["X"]["S"] < 48.67 and bot.shutoff_blocked() == {"X"} and len(bot.alerts.sent) == 1
    # a restart: switched off from the first step, and the statistic recomputed from the database is the same
    again = Bot(cfg_for(tmp_path, SHUT), base_dir=str(tmp_path))
    assert again.shutoff_blocked() == {"X"}
    again.alerts = Alerts()
    again.check_shutoff(U(2026, 10, 18, 2))
    assert again.shutoff_state == bot.shutoff_state and again.alerts.sent == []


def test_only_new_positions_are_blocked(tmp_path):
    """A switched-off maker's resting orders are cancelled and its quotes not placed; a switched-off taker sends
    nothing; open positions stay as they are (held to settlement); the other sleeves keep trading."""
    from test_live_loop import FakeKalshi
    from test_review_findings import risk_bot
    from wxbot.risk.shutoff import ShutoffConfig
    fake = FakeKalshi()
    bot = risk_bot(tmp_path, fake)
    ev = fake.ev
    tick = [m["ticker"] for m in fake.markets]
    bot.ev_tickers[ev] = tick
    bot.acct_ts = {k: time.time() for k in ("sync", "balance", "positions")}
    bot.eq_readings, bot.account_stale_s = [625.0] * 3, 30
    bot.cfg_shutoff = ShutoffConfig.from_cfg(SHUT)
    bot.shutoff_off = set()
    held = Position(tick[2], ev, yes_qty=4, yes_cost=1.8)
    bot.positions = {tick[2]: held}
    want = [DesiredOrder(tick[0], "ask", 3, 5.0, "A", 0.004, event=ev)]
    bot.execute(want)
    assert [c[0] for c in fake.created] == [tick[0]]                    # A quotes while it is on
    bot.shutoff_off = {"A"}                                             # A switched off
    bot.execute(want)
    assert fake.cancelled and not fake.resting                          # its resting order is cancelled ...
    assert len(fake.created) == 1 and bot.positions == {tick[2]: held}  # ... nothing new, the position untouched
    assert held.yes_qty == 4 and held.yes_cost == 1.8 and not any(c[1] == "ask" and c[0] == tick[2]
                                                                   for c in fake.created)
    cfg = {"trade": True, "trade_size": 5, "trade_max_usd_day": 25}
    bot.shutoff_off = {"A", "X"}
    assert bot.execute_take("X", ev, tick[3], "yes", 0.40, 5, cfg) is None and len(fake.created) == 1
    bot.execute_take("R", ev, tick[3], "yes", 0.40, 5, cfg)            # R is on: it trades
    assert len(fake.created) == 2 and fake.created[-1][0] == tick[3]
    bot.cfg_shutoff = ShutoffConfig()                                   # the option off: nothing is blocked
    assert bot.shutoff_blocked() == set()


def test_reset_turns_it_back_on_and_the_count_starts_afresh(tmp_path):
    from wxbot.__main__ import reset_shutoff
    from wxbot.app.runner import Bot
    cfg = cfg_for(tmp_path, SHUT)
    bot = Bot(cfg, base_dir=str(tmp_path))
    bot.alerts = Alerts()
    for day in range(9, 14):
        lose(bot.store, "X", day, 10.0)
    bot.check_shutoff(U(2026, 10, 14, 2))
    assert bot.shutoff_blocked() == {"X"}
    out = reset_shutoff(cfg, "x", now=U(2026, 10, 14, 3))
    assert "X (was switched off on 2026-10-13)" in out and os.path.exists(tmp_path / "RESTART")
    again = Bot(cfg, base_dir=str(tmp_path))
    assert again.shutoff_blocked() == set()
    again.check_shutoff(U(2026, 10, 15, 2))       # counts from the reset: Oct 14 (after 03Z) only, nothing settled
    x = again.shutoff_state["X"]
    assert x["first_day"] == "2026-10-14" and x["days"] == 1 and x["S"] == pytest.approx(0.94)
    assert not x["switched_off"] and again.store.kv_get("shutoff:X")["previous"]["day"] == "2026-10-13"
    assert "A (was on)" in reset_shutoff(cfg, "all", now=U(2026, 10, 15, 3))
    with pytest.raises(SystemExit):
        reset_shutoff(cfg, "Z")
    with pytest.raises(SystemExit):
        reset_shutoff(cfg, None)


def test_the_count_starts_when_the_bot_first_checks_a_sleeve_without_a_since(tmp_path):
    from wxbot.app.runner import Bot
    cfg = cfg_for(tmp_path, {"enabled": True, "sleeves": {"E": {"k": 0.88, "h": 55.29}}})
    bot = Bot(cfg, base_dir=str(tmp_path))
    lose(bot.store, "E", 9, 30.0)                 # before the count starts: not counted
    bot.check_shutoff(U(2026, 10, 10, 12))
    assert bot.store.kv_get("shutoff:E")["first_ts"] == U(2026, 10, 10, 12)
    bot.check_shutoff(U(2026, 10, 12, 2))
    assert bot.shutoff_state["E"]["first_day"] == "2026-10-10" and bot.shutoff_state["E"]["pnl_usd"] == 0.0


def test_off_by_default_changes_nothing(tmp_path):
    from wxbot.app.runner import Bot
    bot = Bot(cfg_for(tmp_path), base_dir=str(tmp_path))
    for day in range(9, 14):
        lose(bot.store, "X", day, 100.0)
    bot.check_shutoff(U(2026, 10, 20))
    assert bot.shutoff_blocked() == set() and bot.shutoff_state == {} and "shutoff" not in bot._write_status()
    assert bot.store.kv_get("shutoff:X") is None
