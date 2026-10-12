"""Sleeve X's budget order (taker_shadow.priority, Oct 9): at 21Z the evening-before signals (h = -3, +7.0c a contract
over Aug-Sep) and the same-day last-hour signals (h = 21, +0.7c) share one clock time, and the $15 budget went to them
in the bot's city order. With the option the step's signals are held and sent evening-before first (then by city
order, or by edge); without it nothing changes."""
import datetime as dt
import time
from collections import defaultdict

from wxbot.models.fair_value import LadderQuote
from wxbot.store.db import Store
from wxbot.strategy import taker
from wxbot.strategy.types import EventBook

NOW = dt.datetime(2026, 10, 9, 21, 5, tzinfo=dt.timezone.utc).timestamp()     # 21:05Z Oct 9


def item(seq, h, edge, dec=0.0):
    return {"seq": seq, "h": h, "edge": edge, "dec_ts": dec}


def test_priority_order_city_hour_and_edge():
    items = [item(0, 21, 0.10), item(1, -3, 0.07), item(2, 21, 0.20), item(3, -3, 0.12), item(4, -3, 0.07, -3600)]
    assert [x["seq"] for x in taker.priority_order(items, "city")] == [0, 1, 2, 3, 4]
    assert [x["seq"] for x in taker.priority_order(items, "hour")] == [4, 1, 3, 0, 2]     # older decision, then h
    assert [x["seq"] for x in taker.priority_order(items, "edge")] == [4, 3, 1, 2, 0]     # then the larger edge
    assert [x["seq"] for x in taker.priority_order(list(reversed(items)), "city")] == [0, 1, 2, 3, 4]


def event(day, probs):
    """A liquid three-bracket New York ladder (mids sum 0.93, spreads 2c); with these probabilities a bracket is a
    YES signal at its ask when p - ask - fee > 6c (A at 32c, B at 42c, C at 22c), and none buys NO."""
    ev = f"KXHIGHNY-26OCT{day:02d}"
    quotes = {f"{ev}-A": LadderQuote(30, 32, 50.0, 50.0), f"{ev}-B": LadderQuote(40, 42, 50.0, 50.0),
              f"{ev}-C": LadderQuote(20, 22, 50.0, 50.0)}
    return EventBook(ev, "KXHIGHNY", "high", "KNYC", dt.date(2026, 10, day), 0.0, [], [], quotes, {}), \
        {f"{ev}-{k}": p for k, p in zip("ABC", probs)}


def x_bot(path, raw):
    from wxbot.app.runner import Bot
    bot = Bot.__new__(Bot)
    bot.store = Store(str(path))
    bot.cfg_x, bot.x_seen, bot.ml_sha = taker.TakerConfig(enabled=True), set(), "sha"
    bot.cfg_x_raw = raw
    bot.sent = []
    bot.execute_take = lambda sl, ev, tk, side, price, size, c: bot.sent.append((sl, tk, side, price, size))
    return bot


def priced_in_city_order(bot, today=(0.45, 0.62, 0.21), tomorrow=(0.50, 0.70, 0.45)):
    """At 21:05Z the bot prices today's New York event (h = 21: A edge 0.11, B 0.18) before tomorrow's (h = -3: A
    0.16, B 0.26, C 0.21): the city order."""
    ev1, p1 = event(9, today)
    ev2, p2 = event(10, tomorrow)
    bot.shadow_taker(ev1, p1, 21, NOW)
    bot.shadow_taker(ev2, p2, -3, NOW)


def day_and_bracket(sent):
    return [(t.rsplit("-", 2)[1], t[-1]) for _, t, *_ in sent]


def test_without_the_option_signals_go_out_as_they_are_priced(tmp_path):
    for i, raw in enumerate(({"trade": True}, {"trade": True, "priority": "city"},
                             {"trade": True, "priority": "fastest"})):
        bot = x_bot(tmp_path / f"x{i}.sqlite", raw)
        priced_in_city_order(bot)
        assert day_and_bracket(bot.sent) == [("26OCT09", "A"), ("26OCT09", "B"), ("26OCT10", "A"), ("26OCT10", "B"),
                                             ("26OCT10", "C")]                  # inline, in the city order
        assert getattr(bot, "x_pending", []) == []
    assert bot.x_priority() == "city"                                  # an unknown value: city order, with a warning


def test_hour_priority_sends_the_evening_before_signals_first(tmp_path):
    bot = x_bot(tmp_path / "x.sqlite", {"trade": True, "priority": "hour"})
    priced_in_city_order(bot)
    assert bot.sent == [] and len(bot.x_pending) == 5                    # logged, held
    assert bot.store.c.execute("SELECT COUNT(*) FROM taker_shadow").fetchone()[0] == 5
    assert bot.flush_x_pending(NOW + 2) == 0 and bot.sent == []         # held 5 s: a split step can still join
    assert bot.flush_x_pending(NOW + 5) == 5 and bot.x_pending == []
    assert day_and_bracket(bot.sent) == [("26OCT10", "A"), ("26OCT10", "B"), ("26OCT10", "C"), ("26OCT09", "A"),
                                         ("26OCT09", "B")]                      # then city (ladder) order
    assert [s[4] for s in bot.sent] == [5.0] * 5 and {s[0] for s in bot.sent} == {"X"}


def test_edge_priority_sends_evening_before_first_then_the_larger_edge(tmp_path):
    bot = x_bot(tmp_path / "x.sqlite", {"trade": True, "priority": "edge", "priority_hold_s": 0})
    priced_in_city_order(bot)
    assert bot.flush_x_pending(NOW) == 5
    assert day_and_bracket(bot.sent) == [("26OCT10", "B"), ("26OCT10", "C"), ("26OCT10", "A"), ("26OCT09", "B"),
                                         ("26OCT09", "A")]


def test_a_held_signal_past_thirty_minutes_is_not_sent(tmp_path):
    bot = x_bot(tmp_path / "x.sqlite", {"trade": True, "priority": "hour"})
    priced_in_city_order(bot)
    assert bot.flush_x_pending(NOW + 1900) == 0 and bot.sent == [] and bot.x_pending == []


def test_the_budget_goes_to_the_evening_before_signal(tmp_path, monkeypatch):
    """Through the real execute_take (paper): a $2 budget day holds one 5-contract take at 32c ($1.68 with the fee),
    and with the option it is the evening-before one, not the same-day one priced first."""
    from wxbot.app import runner
    from wxbot.risk.limits import RiskConfig, RiskState

    class Clock(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return dt.datetime.fromtimestamp(NOW, tz)
    shim = type("dtmod", (), {k: getattr(dt, k) for k in ("date", "timedelta", "timezone")})
    shim.datetime = Clock
    monkeypatch.setattr(runner, "dt", shim)

    class Rest:
        def orderbook(self, t):
            return {"no_dollars": [["0.6800", "50"]]}         # YES offered at 32c

        def markets(self, **kw):
            return []
    sent = {}
    for mode in ("city", "hour", "edge"):
        bot = runner.Bot.__new__(runner.Bot)
        bot.store, bot.rest, bot.mode, bot.cfg = Store(str(tmp_path / f"{mode}.sqlite")), Rest(), "paper", {}
        bot.risk_cfg = RiskConfig()
        bot.risk = RiskState(bot.risk_cfg)
        bot.positions, bot.counters, bot.fair_cache = {}, defaultdict(int), {}
        bot.events, bot.hourly, bot.ev_tickers, bot.ev_tickers_miss = {}, {}, {}, {}
        bot.live, bot.paper, bot.netting, bot.exec_log = None, None, False, None
        bot.cfg_x, bot.x_seen, bot.ml_sha = taker.TakerConfig(enabled=True), set(), "sha"
        bot.cfg_x_raw = {"trade": True, "trade_size": 5, "trade_max_usd_day": 2.0, "priority": mode}
        priced_in_city_order(bot, today=(0.60, 0.41, 0.21), tomorrow=(0.50, 0.41, 0.21))   # one signal each: A
        bot.flush_x_pending(NOW, force=True)
        sent[mode] = [r[0] for r in bot.store.c.execute("SELECT ticker FROM fills WHERE sleeve = 'X'")]
    # the same-day signal has the larger edge (0.26 against 0.16): the edge order still puts the evening first
    assert sent == {"city": ["KXHIGHNY-26OCT09-A"], "hour": ["KXHIGHNY-26OCT10-A"], "edge": ["KXHIGHNY-26OCT10-A"]}


def test_decide_sends_the_held_signals(tmp_path):
    from wxbot.app import runner
    cfg = {"mode": "paper", "env": "prod", "db_path": str(tmp_path / "p.sqlite"), "status_path": str(tmp_path / "s.json"),
           "sleeve_a": {"enabled": True, "series": ["KXHIGHNY"]}, "sleeve_b": {"enabled": False, "shadow": False},
           "tape_archive": {"enabled": False}, "loop": {"tick_seconds": 2},
           "taker_shadow": {"enabled": True, "trade": True, "priority": "edge"}}
    bot = runner.Bot(cfg, base_dir=str(tmp_path))
    sent = []
    bot.execute_take = lambda *a: sent.append(a)
    t = time.time()
    bot.queue_x([(t, "KXHIGHNY-26OCT10", "KXHIGHNY-26OCT10-A", "yes", 0.32, 5.0, 50.0, 0.5, 0.16, -3, "s", 0.02, 0.93,
                  5.0, 50.0, 5.0)], None, t - 10)
    assert bot.decide() == [] and len(sent) == 1 and bot.x_pending == []
    assert sent[0][:6] == ("X", "KXHIGHNY-26OCT10", "KXHIGHNY-26OCT10-A", "yes", 0.32, 5.0)
