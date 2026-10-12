"""Reachability (Oct 7, first live afternoon): Stage 1 parked its whole $75 open-risk budget on 11 orders, most of them
behind 1c walls of 11,000-44,000 contracts that only a giant sweep could clear (none did in 8 days of tape), while a
sweep went through a bracket it had no budget left for. Sleeve A now skips quotes with more than
max_contracts_ahead contracts ahead of them and, when the caps bind, gives the budget to the fewest ahead first."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_live_loop import FakeKalshi  # noqa: E402
from test_review_findings import risk_bot  # noqa: E402

from wxbot.app.runner import Bot  # noqa: E402
from wxbot.risk.limits import RiskConfig, filter_orders  # noqa: E402
from wxbot.strategy.sleeve_a import SleeveAConfig, contracts_ahead  # noqa: E402
from wxbot.strategy.types import DesiredOrder, EventBook  # noqa: E402

NY = {"no_dollars": [["0.9700", "2610.75"], ["0.9800", "5800.00"], ["0.9900", "23221.63"]],
      "yes_dollars": [["0.0100", "500.00"]]}                     # KXHIGHNY-26OCT07-B68.5 at 4:33 PM ET Oct 7
ATL = {"no_dollars": [["0.9500", "17.00"], ["0.9700", "25.00"], ["0.9800", "61.00"]], "yes_dollars": []}
HOU = {"no_dollars": [["0.9700", "7.00"], ["0.9800", "99.00"], ["0.9900", "25.20"]], "yes_dollars": []}


def test_contracts_ahead_counts_every_better_price_and_the_queue_at_ours():
    assert abs(contracts_ahead(NY, "ask", 3) - (2610.75 + 5800 + 23221.63)) < 1e-9     # NO bid at 97c
    assert abs(contracts_ahead(NY, "ask", 1) - 23221.63) < 1e-9                         # joining the 99c wall
    assert abs(contracts_ahead(NY, "ask", 3, own=7) - (2610.75 - 7 + 5800 + 23221.63)) < 1e-9
    assert contracts_ahead(ATL, "ask", 5) == 17 + 25 + 61                                # NO bid at 95c
    assert abs(contracts_ahead(HOU, "ask", 3, own=7) - (99 + 25.2)) < 1e-9               # our 7 are the whole level
    favs = {"yes_dollars": [["0.9400", "100"], ["0.9500", "50"], ["0.9600", "20"]]}
    assert contracts_ahead(favs, "bid", 95) == 70 and contracts_ahead(favs, "bid", 97) == 0
    assert contracts_ahead({}, "ask", 3) == 0


def order(ticker, ev, price=3, n=7.0):
    return DesiredOrder(ticker, "ask", price, n, "A", 0.004, event=ev)


def test_the_risk_filter_gives_the_budget_to_the_fewest_contracts_ahead_first():
    deep, thin, mid = order("E1-B1", "E1"), order("E2-B1", "E2"), order("E3-B1", "E3")
    tick = {e: [f"{e}-B1", f"{e}-B2"] for e in ("E1", "E2", "E3")}
    cfg = RiskConfig(max_loss_per_bracket_usd=8, max_loss_per_event_usd=10, max_total_open_risk_usd=14)
    rank = {("E1-B1", "ask", 3): 31000.0, ("E2-B1", "ask", 3): 96.0, ("E3-B1", "ask", 3): 300.0}
    kept, dropped = filter_orders([deep, thin, mid], {}, tick, cfg, 0.0, incumbents={("E1-B1", "ask", 3)}, rank=rank)
    assert [o.ticker for o in kept] == ["E2-B1", "E3-B1"] and dropped == ["E1-B1/ask: caps"]
    kept, _ = filter_orders([deep, thin, mid], {}, tick, cfg, 0.0, incumbents={("E1-B1", "ask", 3)})
    assert kept[0].ticker == "E1-B1"                       # without a rank: the resting order first, as before


class Stub:
    """Just what Bot.reach_rank reads."""
    resting_own = Bot.resting_own

    def __init__(self, books, mode="live", resting=(), cap=1000.0):
        self.cfg_a = SleeveAConfig(max_contracts_ahead=cap)
        self.events = {"E": EventBook(event="E", series="S", kind="high", station="X", local_date=None, close_ts=0,
                                      books=books)}
        self.mode = mode
        lo = [type("O", (), dict(ticker=t, side=s, price_cents=p, remaining=n, state="resting"))()
              for t, s, p, n in resting]
        if mode == "live":
            self.live, self.paper = type("L", (), {"resting_list": lambda self: lo})(), None
        else:
            self.live, self.paper = None, type("P", (), {"resting": lambda self: lo})()


def test_quotes_behind_a_wall_are_left_out_and_the_rest_ranked():
    s = Stub({"NY": NY, "ATL": ATL, "HOU": HOU}, resting=[("HOU", "ask", 3, 7.0)])
    rank = Bot.reach_rank(s, [order("NY", "E"), order("ATL", "E", 5), order("HOU", "E")])
    assert ("NY", "ask", 3) not in rank                                      # 31,600 ahead
    assert rank[("ATL", "ask", 5)] == 103.0
    assert abs(rank[("HOU", "ask", 3)] - (99 + 25.2) * 0.5) < 1e-9          # resting already: counts half
    st = s.reach_stats
    assert st["kept"] == 2 and st["skipped"] == 1 and st["skipped_examples"] == [("NY", 31632)]


def test_a_resting_order_keeps_its_place_until_twice_the_limit():
    wall = {"no_dollars": [["0.9700", "7.00"], ["0.9900", "1500.00"]]}
    new = Bot.reach_rank(Stub({"T": wall}), [order("T", "E")])
    held = Bot.reach_rank(Stub({"T": wall}, resting=[("T", "ask", 3, 7.0)]), [order("T", "E")])
    assert new == {} and held == {("T", "ask", 3): 750.0}
    paper = Bot.reach_rank(Stub({"T": wall}, mode="paper", resting=[("T", "ask", 3, 7.0)]), [order("T", "E")])
    assert paper == {("T", "ask", 3): (1507.0) * 0.5}                       # a paper order is not in the book


def test_the_bot_never_sends_a_quote_behind_a_wall(tmp_path):
    fake = FakeKalshi()
    bot = risk_bot(tmp_path, fake)
    ev = fake.ev
    tick = [m["ticker"] for m in fake.markets]
    bot.ev_tickers[ev] = tick
    bot.cfg_a = SleeveAConfig(max_contracts_ahead=1000)
    bot.events = {ev: EventBook(event=ev, series="KXHIGHNY", kind="high", station="KNYC", local_date=None,
                                close_ts=time.time() + 6 * 3600, markets=fake.markets,
                                books={tick[0]: NY, tick[4]: ATL})}         # the two near-certain NO tails
    bot.acct_ts = {k: time.time() for k in ("sync", "balance", "positions")}
    bot.eq_readings = [625.0] * 3
    bot.account_stale_s = 30
    bot.execute([order(tick[0], ev), order(tick[4], ev, 5)])
    assert [c[0] for c in fake.created] == [tick[4]], fake.created     # the one behind the 1c wall never goes out
    assert bot.reach_stats["skipped"] == 1 and bot.reach_stats["kept"] == 1
    bot.cfg_a = SleeveAConfig()                                         # off (paper, replays): unchanged behavior
    fake.created.clear()
    bot.execute([order(tick[0], ev), order(tick[4], ev, 5)])
    assert tick[0] in [c[0] for c in fake.created]


def test_each_event_spends_its_budget_on_its_most_reachable_bracket():
    """Sleeve A gives an event's decided budget ($10 at Stage 1) to brackets in ladder order: the first decided one
    took it all, so when that one sat behind a wall the event quoted nothing reachable."""
    from wxbot.models.fair_value import FairValueResult, LadderQuote
    from wxbot.strategy import sleeve_a
    tick = ["E-T60", "E-B61.5", "E-B63.5", "E-T64"]
    brs = [type("B", (), {"ticker": t})() for t in tick]
    quotes = {t: LadderQuote(yes_bid=None, yes_ask=1) for t in tick}
    quotes["E-B61.5"] = LadderQuote(yes_bid=None, yes_ask=3)
    books = {"E-T60": NY, "E-B61.5": HOU, "E-B63.5": ATL, "E-T64": NY}
    ev = EventBook(event="E", series="S", kind="high", station="X", local_date=None, close_ts=time.time() + 86400,
                   brackets=brs, quotes=quotes, books=books)
    fv = FairValueResult(probs={"E-T60": 0.002, "E-B61.5": 0.004, "E-B63.5": 0.006, "E-T64": 0.001}, market={}, nbm={},
                         disagreement=0.0)
    cfg = SleeveAConfig(decided_only=True, decided_prob=0.97, min_edge_c=2.0, improve_if_edge_c=99, clip_contracts=12,
                        decided_max_cost_usd=10, min_price_c=60, max_price_c=98)
    old, _ = sleeve_a.propose(ev, fv, time.time(), cfg)
    assert [o.ticker for o in old] == ["E-T60"]                         # ladder order: the wall took the budget
    far = []
    new, _ = sleeve_a.propose(ev, fv, time.time(), SleeveAConfig(**{**cfg.__dict__, "max_contracts_ahead": 1000}),
                              out_of_reach=far)
    assert [o.ticker for o in new] == ["E-B63.5"]                       # 86 ahead beats 131
    assert sorted(t for t, _ in far) == ["E-T60", "E-T64"]
    held, _ = sleeve_a.propose(ev, fv, time.time(), SleeveAConfig(**{**cfg.__dict__, "max_contracts_ahead": 1000}),
                               resting={("E-B61.5", "ask", 3): 7.0})
    assert [o.ticker for o in held] == ["E-B61.5"]                      # ours rests there: 62 counts, it stays
