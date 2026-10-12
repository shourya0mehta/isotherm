"""Sleeve E: the bot sends candidate C's logged early-hour trades once, fresh ones only, through the taker path."""
import time

from wxbot.store.db import Store


def make_bot(tmp_path, cfg):
    from wxbot.app.runner import Bot
    bot = Bot.__new__(Bot)
    bot.store = Store(str(tmp_path / "live.sqlite"))
    bot.base = str(tmp_path)
    bot.cfg = {"early_taker": cfg}
    bot.last, bot.next_due = {}, __import__('collections').defaultdict(float)
    bot.sent = []
    bot.execute_take = lambda sl, ev, tk, side, price, size, c: bot.sent.append((sl, ev, tk, side, price, size)) or {}
    return bot


def shadow_db(tmp_path, rows):
    (tmp_path / "data").mkdir(exist_ok=True)
    st = Store(str(tmp_path / "data" / "early_shadow_c.sqlite"))
    st.add_taker_shadow(rows)
    return st


def row(ts, ticker, side="yes", price=0.27, h=-9):
    return (ts, "KXHIGHNY-26OCT09", ticker, side, price, 5.0, 40.0, 0.6, 0.3, h, "c0ffee", 0.03, 1.0)


def test_fresh_signals_are_sent_once_as_sleeve_e_and_old_ones_never(tmp_path):
    now = time.time()
    shadow_db(tmp_path, [row(now - 30, "KXHIGHNY-26OCT09-B70.5"), row(now - 3600, "KXHIGHNY-26OCT09-T73", "no", 0.77),
                         row(now - 20, "KXHIGHNY-26OCT09-T70", "no", 0.79, h=-6)])
    bot = make_bot(tmp_path, {"enabled": True, "trade": True, "max_age_s": 600})
    assert bot.early_take(now) == 2
    assert [(s[0], s[2], s[3], s[4]) for s in bot.sent] == [("E", "KXHIGHNY-26OCT09-B70.5", "yes", 0.27),
                                                            ("E", "KXHIGHNY-26OCT09-T70", "no", 0.79)]
    assert bot.early_take(now + 30) == 0 and len(bot.sent) == 2                 # each signal once
    assert bot.store.kv_get("early_take:KXHIGHNY-26OCT09:KXHIGHNY-26OCT09-T70:no")["h"] == -6


def test_no_database_or_an_unreadable_one_sends_nothing(tmp_path):
    bot = make_bot(tmp_path, {"enabled": True, "trade": True})
    assert bot.early_take() == 0 and bot.sent == []
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "early_shadow_c.sqlite").write_text("not a database")
    assert bot.early_take() == 0 and bot.sent == []


def test_the_real_take_path_skips_while_trade_is_off(tmp_path):
    from collections import defaultdict
    from wxbot.app.runner import Bot
    from wxbot.risk.limits import RiskConfig, RiskState
    now = time.time()
    shadow_db(tmp_path, [row(now - 10, "KXHIGHNY-26OCT09-B70.5")])
    bot = Bot.__new__(Bot)
    bot.store, bot.base, bot.cfg = Store(str(tmp_path / "live.sqlite")), str(tmp_path), {
        "early_taker": {"enabled": True, "trade": False}}
    bot.mode, bot.risk_cfg = "paper", RiskConfig()
    bot.risk, bot.counters, bot.last, bot.next_due = RiskState(bot.risk_cfg), defaultdict(int), {}, defaultdict(float)
    assert bot.early_take(now) == 1                                               # read and recorded ...
    assert bot.store.c.execute("SELECT COUNT(*) FROM fills").fetchone()[0] == 0  # ... but nothing traded
