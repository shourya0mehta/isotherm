"""The same-weather budget (Oct 11, 2026; wxbot/risk/same_weather.py, docs/results/r14_same_weather.txt Q2): one $20
worst-case budget per group of cities whose forecast misses move together (Philadelphia/Trenton/Newark/New York;
Austin/San Antonio), per market date, shared by the two daily makers (A, the near-certain bets; T, the newer cities).
It runs after each maker's own caps and before the account's, as research/ml/r14_lib.py group_filter does in the
replay, and gives the same sizes as group_filter on the same inputs."""
import collections
import datetime as dt
import logging
import os
import random
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_live_loop import FakeKalshi  # noqa: E402
from test_review_findings import risk_bot  # noqa: E402
from wxbot.risk import same_weather as SW  # noqa: E402
from wxbot.risk.limits import RiskConfig, event_worst_case, filter_orders  # noqa: E402
from wxbot.strategy.types import DesiredOrder, EventBook, Position  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BLOCK = {"enabled": True, "budget_usd": 20, "sleeves": ["A", "T"],
         "groups": [["KXHIGHPHIL", "KXHIGHTTTN", "KXHIGHTEWR", "KXHIGHNY"], ["KXHIGHAUS", "KXHIGHTSATX"]]}
CFG = SW.SameWeatherConfig.from_cfg(BLOCK)
NE = "Philadelphia/Trenton/Newark/New York"
D12, D13 = "26OCT12", "26OCT13"
SUFFIX = ("T60", "B60.5", "B62.5", "B64.5", "T65")    # brackets the fake exchange quotes nothing on (never a cross)


def ev(series, day=D12):
    return f"{series}-{day}"


def ladder(event):
    return [f"{event}-{s}" for s in SUFFIX]


def no(event, k, pay_c, n, sleeve, fair_yes=0.01):
    """Buy NO on bracket k of the event at pay_c cents (edge (1 - fair_yes) - pay_c / 100 a contract)."""
    return DesiredOrder(ladder(event)[k], "ask", 100 - pay_c, float(n), sleeve, fair_yes, event=event)


def sizes(orders):
    out = collections.defaultdict(float)
    for o in orders:
        out[o.ticker] += o.count
    return dict(out)


def ticks(*events):
    return {e: ladder(e) for e in events}


# ------------------------------------------------------------------------------------------------ the budget
def test_the_budget_is_shared_by_both_makers_and_the_groups_events_of_one_date():
    ny, phl, ttn, aus, chi = ev("KXHIGHNY"), ev("KXHIGHPHIL"), ev("KXHIGHTTTN"), ev("KXHIGHAUS"), ev("KXHIGHCHI")
    want = [no(ny, 1, 95, 8, "A", 0.0),              # $7.60 worst case, edge 5c
            no(phl, 2, 95, 8, "A", 0.0),             # $7.60, edge 5c
            no(ttn, 1, 95, 10, "T", 0.02),           # $9.50, edge 3c: $4.80 of room left after the two A bets
            no(aus, 1, 96, 10, "A"),                 # the other group: its own $20
            no(chi, 1, 96, 30, "A")]                 # not in a group: untouched
    r = SW.apply(want, {}, {}, ticks(ny, phl, ttn, aus, chi), CFG)
    # filter_orders' own greedy shrink (10 -> 7 -> 4 contracts), as group_filter does
    assert sizes(r.kept) == {want[0].ticker: 8, want[1].ticker: 8, want[2].ticker: 4, want[3].ticker: 10,
                             want[4].ticker: 30}
    assert r.usage[(0, D12)]["used"] == pytest.approx(7.6 + 7.6 + 3.8) and r.usage[(1, D12)]["used"] == 9.6
    assert [(o.ticker, n) for o, n, g, used in r.refused] == [(want[2].ticker, 4.0)]
    # counted the way the live caps count it: two NO bets in one event can't both lose (adverse selection)
    two = [no(ny, 1, 95, 10, "A"), no(ny, 3, 95, 10, "A"), no(phl, 1, 95, 10, "A"), no(ttn, 2, 95, 10, "T")]
    r = SW.apply(two, {}, {}, ticks(ny, phl, ttn), CFG)
    assert sum(o.count for o in r.kept) < 40 and r.usage[(0, D12)]["used"] <= 20 + 1e-9
    assert sizes(r.kept)[two[0].ticker] == 10 and sizes(r.kept)[two[1].ticker] == 10   # $9.50 for both NY bets


def test_different_dates_are_independent():
    ny12, phl12, ny13, phl13 = ev("KXHIGHNY"), ev("KXHIGHPHIL"), ev("KXHIGHNY", D13), ev("KXHIGHPHIL", D13)
    held = {ny12: [Position(ladder(ny12)[1], ny12, no_qty=16, no_cost=15.2)]}       # $15.20 held on Oct 12
    want = [no(phl12, 1, 95, 10, "A"), no(ny13, 1, 95, 10, "A"), no(phl13, 1, 95, 10, "A")]
    r = SW.apply(want, held, {}, ticks(ny12, phl12, ny13, phl13), CFG)
    got = sizes(r.kept)
    assert got.get(want[0].ticker, 0) == 4                        # Oct 12: $4.80 left, 10 -> 7 -> 4 contracts
    assert got[want[1].ticker] == 10 and got[want[2].ticker] == 10     # Oct 13: its own $20 ($19 used)
    assert r.usage[(0, D12)]["positions"] == pytest.approx(15.2) and r.usage[(0, D13)]["positions"] == 0


def test_the_larger_edge_gets_the_room_first_and_positions_stay():
    ny, ewr = ev("KXHIGHNY"), ev("KXHIGHTEWR")
    small, large = no(ny, 1, 95, 12, "A", 0.03), no(ewr, 1, 95, 12, "T", 0.0)        # edges 2c and 5c, $11.40 each
    for order in ([small, large], [large, small]):
        got = sizes(SW.apply(order, {}, {}, ticks(ny, ewr), CFG).kept)
        assert got[large.ticker] == 12 and got.get(small.ticker, 0) == 8             # $8.60 left: 12 -> 8
    over = {ny: [Position(ladder(ny)[2], ny, no_qty=26, no_cost=25.0)]}              # $25 held: over the budget
    r = SW.apply([large], over, {}, ticks(ny, ewr), CFG)
    assert r.kept == [] and over[ny][0].no_qty == 26                                  # dropped; the position stays


def test_resting_orders_count_and_keep_their_place():
    ny, phl, ttn = ev("KXHIGHNY"), ev("KXHIGHPHIL"), ev("KXHIGHTTTN")
    # an order the exchange may still fill that the makers no longer stand for (a cancel not yet confirmed): it
    # counts first and is never dropped
    base = {ny: [no(ny, 2, 95, 16, "A")]}                                             # $15.20
    want = [no(phl, 1, 95, 10, "A")]
    r = SW.apply(want, {}, base, ticks(ny, phl), CFG)
    assert sizes(r.kept) == {want[0].ticker: 4} and r.usage[(0, D12)]["used"] == pytest.approx(15.2 + 3.8)
    # a resting order keeps its place ahead of a new one with a larger edge, as in the other caps (queue priority)
    resting, new = no(ttn, 1, 95, 12, "T", 0.03), no(phl, 1, 95, 12, "A", 0.0)
    inc = {(resting.ticker, resting.side, resting.price_cents)}
    got = sizes(SW.apply([new, resting], {}, {}, ticks(ttn, phl), CFG, incumbents=inc).kept)
    assert got[resting.ticker] == 12 and got[new.ticker] == 8


# ------------------------------------------------------------------------------------------------ in the bot
def bot_with(tmp_path, block=None):
    """A live bot (the round-1 review's bare one) with the four cities' events of Oct 12 and generous own caps, so
    only the same-weather budget can bind."""
    os.makedirs(tmp_path, exist_ok=True)
    fake = FakeKalshi()
    bot = risk_bot(tmp_path, fake)
    bot.cfg_same_weather, bot.same_weather_usage = SW.SameWeatherConfig.from_cfg(block), {}
    bot.series_t = {"KXHIGHTTTN", "KXHIGHTEWR"}
    bot.risk_t, bot.cash_cap_t = RiskConfig(max_loss_per_bracket_usd=15, max_loss_per_event_usd=25,
                                            max_total_open_risk_usd=100), 100.0
    close = time.time() + 18 * 3600
    for s in ("KXHIGHNY", "KXHIGHPHIL", "KXHIGHTTTN", "KXHIGHTEWR"):
        e = ev(s)
        bot.events[e] = EventBook(e, s, "high", "", dt.date(2026, 10, 12), close,
                                  markets=[{"ticker": t} for t in ladder(e)])
    bot.acct_ts = {k: time.time() for k in ("sync", "balance", "positions")}
    bot.eq_readings, bot.account_stale_s = [625.0] * 3, 30
    return fake, bot


def placed(fake):
    return {c[0]: c[3] for c in fake.created}


WANT = lambda: [no(ev("KXHIGHNY"), 1, 95, 8, "A", 0.0), no(ev("KXHIGHPHIL"), 2, 95, 9, "A", 0.0),   # noqa: E731
                no(ev("KXHIGHTTTN"), 1, 95, 10, "T", 0.02), no(ev("KXHIGHTEWR"), 3, 95, 6, "T", 0.03)]


def test_in_the_bot_it_cuts_before_the_account_caps_and_logs_each_refusal(tmp_path, caplog):
    fake, bot = bot_with(tmp_path, BLOCK)
    want = WANT()
    with caplog.at_level(logging.INFO, logger="wxbot"):
        bot.execute(want)
    # $7.60 + $8.55 for the two near-certain bets, Trenton 10 -> 7 -> 4 contracts ($3.80), Newark: 5 cents left
    assert placed(fake) == {want[0].ticker: 8, want[1].ticker: 9, want[2].ticker: 4}
    notes = [r.getMessage() for r in caplog.records if "same-weather budget" in r.getMessage()]
    assert notes == [f"QUOTE T {want[2].ticker} buy NO @95c x10 cut to 4: same-weather budget ({NE} $19.95 of $20)",
                     f"QUOTE T {want[3].ticker} buy NO @95c x6 skipped: same-weather budget ({NE} $19.95 of $20)"]
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="wxbot"):
        bot.execute(WANT())                                   # the next pass: the same refusals, not logged again
    assert not [r for r in caplog.records if "same-weather budget" in r.getMessage()]
    assert bot.same_weather_usage[(0, D12)] == {"used": pytest.approx(19.95), "positions": 0.0, "cut": 2}


def test_off_means_no_change(tmp_path):
    assert SW.SameWeatherConfig.from_cfg(None) == SW.SameWeatherConfig() == SW.SameWeatherConfig.from_cfg(
        {**BLOCK, "enabled": False}) and not SW.SameWeatherConfig().enabled
    runs = {}
    for name, block in (("absent", None), ("disabled", {**BLOCK, "enabled": False}), ("on", BLOCK)):
        fake, bot = bot_with(tmp_path / name, block)
        bot.execute(WANT())
        runs[name] = (placed(fake), bot.same_weather_usage)
    assert runs["absent"] == runs["disabled"] and runs["absent"][1] == {}
    assert runs["absent"][0] == {o.ticker: o.count for o in WANT()} != runs["on"][0]
    for bad in ({**BLOCK, "groups": []}, {**BLOCK, "groups": [["KXHIGHNY"], ["KXHIGHNY", "KXHIGHAUS"]]},
                {**BLOCK, "budget_usd": 0}, {**BLOCK, "sleeves": ["A", "X"]}):
        with pytest.raises(ValueError):
            SW.SameWeatherConfig.from_cfg(bad)


def test_the_status_file_shows_each_groups_worst_case_by_date(tmp_path):
    from wxbot.app import runner
    cfg = {"mode": "paper", "env": "prod", "db_path": str(tmp_path / "p.sqlite"), "status_path": str(tmp_path / "s.json"),
           "sleeve_a": {"enabled": True, "series": ["KXHIGHNY"]}, "sleeve_b": {"enabled": False, "shadow": False},
           "tape_archive": {"enabled": False}, "loop": {"tick_seconds": 2}}
    assert "same_weather" not in runner.Bot(dict(cfg), base_dir=str(tmp_path))._write_status()
    bot = runner.Bot({**cfg, "same_weather": BLOCK}, base_dir=str(tmp_path))
    aus13 = ev("KXHIGHAUS", D13)
    held = {aus13: [Position(ladder(aus13)[0], aus13, no_qty=10, no_cost=9.5)]}
    bot.same_weather_pass(WANT(), held, {}, {**ticks(*[o.event for o in WANT()]), **ticks(aus13)}, set(), None)
    st = bot._write_status()["same_weather"]
    assert st["budget_usd"] == 20 and st["sleeves"] == ["A", "T"] and list(st["groups"]) == [NE, "Austin/San Antonio"]
    assert st["groups"][NE] == {"series": BLOCK["groups"][0],
                                "days": {D12: {"used_usd": 19.95, "positions_usd": 0.0, "orders_cut": 2}}}
    assert st["groups"]["Austin/San Antonio"]["days"] == {D13: {"used_usd": 9.5, "positions_usd": 9.5, "orders_cut": 0}}


def test_the_configs():
    import yaml
    for name in ("live_stage4c.yaml", "live_stage5c.yaml", "live_stage4d.yaml", "live_stage4e.yaml"):
        c = SW.SameWeatherConfig.from_cfg(yaml.safe_load(open(os.path.join(ROOT, "config", name))).get("same_weather"))
        assert c == CFG and [c.name(i) for i in range(2)] == [NE, "Austin/San Antonio"], name
    demo = SW.SameWeatherConfig.from_cfg(yaml.safe_load(open(os.path.join(ROOT, "config", "demo.yaml")))["same_weather"])
    assert demo.enabled and demo.groups == (("KXHIGHTTTN", "KXHIGHTEWR"),) and demo.budget_usd == 20
    for name in sorted(os.listdir(os.path.join(ROOT, "config"))):    # on nowhere else
        if name.endswith(".yaml") and name not in ("live_stage4c.yaml", "live_stage5c.yaml", "live_stage4d.yaml",
                                                   "live_stage4e.yaml", "demo.yaml"):
            d = yaml.safe_load(open(os.path.join(ROOT, "config", name))) or {}
            assert not SW.SameWeatherConfig.from_cfg(d.get("same_weather")).enabled, name
    import json
    st = json.load(open(os.path.join(ROOT, "wxbot", "data", "stations.json")))
    assert [st[s]["city"] for g in CFG.groups for s in g] == ["Philadelphia", "Trenton", "Newark", "New York City",
                                                               "Austin", "San Antonio"]


# ------------------------------------------------------------------------------------- parity with the replay
def replay_group_filter():
    """research/ml/r14_lib.py's group_filter run from its own source, with the names r14_lib gives it (filter_orders
    through wxbot.backtest.joint, r14_base.series_of): importing r14_lib itself loads two years of observations
    (research/ml/pool_replay2), which only the research machine has."""
    import ast
    path = os.path.join(ROOT, "research", "ml", "r14_lib.py")
    if not (os.path.exists(path) and os.path.exists(os.path.join(ROOT, "research", "ml", "r14_base.py"))):
        pytest.skip("research/ml/r14_lib.py or r14_base.py is not here")
    sys.path.insert(0, os.path.join(ROOT, "research", "ml"))
    from r14_base import series_of
    import wxbot.backtest.joint as J
    src = open(path).read()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "group_filter")
    ns = {"collections": collections, "RiskConfig": RiskConfig, "event_worst_case": event_worst_case,
          "_filter0": J.filter_orders, "series_of": series_of, "ST": {"gstats": None}}
    exec(compile(ast.Module([fn], []), path, "exec"), ns)
    assert J.filter_orders is filter_orders
    return ns["group_filter"]


def test_the_same_sizes_as_the_replays_group_filter():
    group_filter = replay_group_filter()
    rng = random.Random(14)
    series = ["KXHIGHPHIL", "KXHIGHTTTN", "KXHIGHTEWR", "KXHIGHNY", "KXHIGHAUS", "KXHIGHTSATX", "KXHIGHCHI",
              "KXHIGHTSAN"]
    t_series = {"KXHIGHTTTN", "KXHIGHTEWR", "KXHIGHTSAN"}
    n_cut = 0
    for case in range(300):
        budget = rng.choice([10.0, 20.0, 30.0])
        cfg = SW.SameWeatherConfig.from_cfg({**BLOCK, "budget_usd": budget})
        groups = {s: f"g{i}" for i, g in enumerate(cfg.groups) for s in g}
        events = [ev(s, d) for s in series for d in (D12, D13) if rng.random() < 0.6]
        tick = ticks(*events)
        pos, orders = collections.defaultdict(list), []
        for e in events:
            for k in rng.sample(range(5), rng.randint(0, 2)):
                q = float(rng.randint(1, 15))
                if rng.random() < 0.7:
                    pos[e].append(Position(ladder(e)[k], e, no_qty=q, no_cost=round(q * rng.uniform(0.6, 0.97), 2)))
                else:
                    pos[e].append(Position(ladder(e)[k], e, yes_qty=q, yes_cost=round(q * rng.uniform(0.02, 0.4), 2)))
            for k in rng.sample(range(5), rng.randint(0, 3)):
                sl = "T" if series_of_(e) in t_series else "A"
                side = rng.choice(["ask", "ask", "bid"])
                px = rng.randint(1, 30) if side == "ask" else rng.randint(60, 97)
                orders.append(DesiredOrder(ladder(e)[k], side, px, float(rng.randint(1, 25)), sl,
                                           round(rng.uniform(0.0, 1.0), 3), event=e))
        rng.shuffle(orders)
        want = sorted((o.ticker, o.side, o.price_cents, o.sleeve, o.count) for o in group_filter(orders, pos, tick,
                                                                                                    groups, budget))
        got = sorted((o.ticker, o.side, o.price_cents, o.sleeve, o.count) for o in SW.apply(orders, pos, {}, tick,
                                                                                               cfg).kept)
        assert got == want, case
        n_cut += sum(o.count for o in orders) > sum(x[4] for x in got) + 1e-9
    assert n_cut > 50                                     # the budget binds in many of the cases


def series_of_(e):
    return e.split("-")[0]
