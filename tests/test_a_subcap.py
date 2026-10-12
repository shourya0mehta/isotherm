"""Sleeve A's own open-risk cap inside the shared one (sleeve_a.max_total_open_risk_usd, Oct 9;
docs/results/r12_x_priority_check.txt): A's orders are cut to the sub-cap, other sleeves' orders are untouched, and with
the option unset nothing changes."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from test_live_loop import FakeKalshi  # noqa: E402
from test_review_fixes import live_bot  # noqa: E402

from wxbot.risk.limits import event_worst_case  # noqa: E402
from wxbot.strategy.types import DesiredOrder  # noqa: E402

EV = "KXHIGHNY-26OCT05"
TICKS = {EV: [f"{EV}-B70.5", f"{EV}-B72.5", f"{EV}-B74.5"]}


def _orders():
    return [DesiredOrder(f"{EV}-B70.5", "bid", 90, 25.0, "A", 0.97, event=EV),
            DesiredOrder(f"{EV}-B74.5", "ask", 4, 25.0, "A", 0.02, event=EV),
            DesiredOrder(f"{EV}-B72.5", "bid", 60, 5.0, "X", 1.0, event=EV)]


def test_sleeve_a_orders_are_cut_to_its_own_cap_and_the_taker_is_left_alone(tmp_path):
    bot = live_bot(tmp_path, FakeKalshi())
    bot.series_a_own = {"KXHIGHNY"}
    out = bot.a_subcap(_orders(), {}, {}, TICKS, 0.0, 1000.0, set(), None, None, 10.0)
    a = [o for o in out if o.sleeve == "A"]
    x = [o for o in out if o.sleeve == "X"]
    assert len(x) == 1 and x[0].count == 5.0                       # the taker's order passes through untouched
    assert a and event_worst_case([], a, TICKS[EV]) <= 10.0 + 1e-9  # A's worst case fits its $10 sub-cap
    full = event_worst_case([], [o for o in _orders() if o.sleeve == "A"], TICKS[EV])
    assert full > 10.0                                              # ... and it had to be cut to get there


def test_no_sub_cap_by_default():
    import yaml
    import wxbot.app.runner as runner
    bot = runner.Bot.__new__(runner.Bot)
    load = lambda path: yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", path)))
    for path in ("config/settings.yaml", "config/live_stage3.yaml"):
        assert (load(path).get("sleeve_a") or {}).get("max_total_open_risk_usd") is None   # off there
    # Stage 4 (Shourya's yes, Oct 9): $110 of the shared $150
    assert load("config/live_stage4.yaml")["sleeve_a"]["max_total_open_risk_usd"] == 110
    assert getattr(bot, "a_open_risk_cap", None) is None


def test_sleeve_a_cash_cap_cuts_its_orders_when_open_risk_has_room(tmp_path):
    """sleeve_a.max_cash_usd (full-size stage): A's orders are also held to its own cash, inside the shared cash cap."""
    bot = live_bot(tmp_path, FakeKalshi())
    bot.series_a_own = {"KXHIGHNY"}
    bot.a_cash_cap = 10.0
    out = bot.a_subcap(_orders(), {}, {}, TICKS, 0.0, 1000.0, set(), None, None, None)   # no open-risk sub-cap
    a = [o for o in out if o.sleeve == "A"]
    x = [o for o in out if o.sleeve == "X"]
    assert len(x) == 1 and x[0].count == 5.0
    cost = sum(o.count * (o.price_cents if o.side == "bid" else 100 - o.price_cents) / 100.0 for o in a)
    assert a and cost <= 10.0 + 1e-9                                 # A's cash in its orders fits its $10
