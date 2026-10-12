"""Keep resting orders (sleeve_a.keep_resting, Oct 9). Found in the Oct 8 live log: when the caps bound and the books
moved, the risk filter ranked quotes by contracts ahead before resting orders, so a resting order was cut to 1 contract
and later grown back (159 times in the first 16 hours, up to 11 on one quote); each growth is an amend up, a new place
at the back of the queue, and two writes. With the option, resting orders keep their size unless the caps need it or a
new quote is far more reachable; without it the filter is exactly as before."""
import os
import random
import sys
import time
from collections import defaultdict

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_live_loop import FakeKalshi  # noqa: E402
from test_reach import Stub  # noqa: E402
from test_review_findings import risk_bot  # noqa: E402

from wxbot.app.runner import Bot  # noqa: E402
from wxbot.risk.limits import RiskConfig, bracket_cost, event_worst_case, filter_orders  # noqa: E402
from wxbot.strategy import sleeve_a  # noqa: E402
from wxbot.strategy.sleeve_a import SleeveAConfig, keep_deadband, keep_rank  # noqa: E402
from wxbot.strategy.types import DesiredOrder, EventBook, Position  # noqa: E402

CAPS = RiskConfig(max_loss_per_bracket_usd=15, max_loss_per_event_usd=20, max_total_open_risk_usd=20)
TICK = {e: [f"{e}-B1", f"{e}-B2"] for e in ("E1", "E2", "E3", "E4")}


def no_bid(ev, n=14.0):
    """A near-certain NO bid at 96c (YES price 4c) on the first bracket of event ev."""
    return DesiredOrder(f"{ev}-B1", "ask", 4, float(n), "A", 0.004, event=ev)


def k(o):
    return (o.ticker, o.side, o.price_cents)


def sizes(kept):
    return {o.event: o.count for o in kept}


def flat(res):
    kept, dropped = res
    return [(o.ticker, o.side, o.price_cents, o.count, o.sleeve) for o in kept], dropped


def random_book(rng, sleeves=("A", "A", "T", "B")):
    evs = [f"E{i}" for i in range(rng.randint(1, 5))]
    tick = {e: [f"{e}-B{j}" for j in range(3)] for e in evs}
    desired, seen = [], set()
    for e in evs:
        for _ in range(rng.randint(1, 3)):
            o = DesiredOrder(f"{e}-B{rng.randint(0, 2)}", rng.choice(["bid", "ask"]), rng.randint(2, 97),
                             float(rng.randint(1, 25)), rng.choice(sleeves), rng.random(), event=e)
            if o.key not in seen:
                seen.add(o.key)
                desired.append(o)
    pos = {e: [Position(f"{e}-B0", e, no_qty=2, no_cost=rng.random() * 2)] for e in evs if rng.random() < 0.4}
    cfg = RiskConfig(max_loss_per_bracket_usd=rng.uniform(3, 20), max_loss_per_event_usd=rng.uniform(5, 30),
                     max_total_open_risk_usd=rng.uniform(5, 60))
    return evs, tick, desired, pos, cfg


def test_off_or_empty_is_exactly_the_greedy_filter():
    """keep=None, keep={} and a keep that names no wanted order all give the old result (random books with
    positions, pending exposure, incumbents, ranks, cash caps and netting)."""
    rng = random.Random(7)
    for _ in range(300):
        evs, tick, desired, pos, cfg = random_book(rng)
        base = {e: [DesiredOrder(f"{e}-B1", "bid", 50, 2.0, "?", 0.0, event=e)] for e in evs if rng.random() < 0.3}
        args = dict(cash_used=rng.uniform(0, 10), cash_cap=rng.choice([None, rng.uniform(10, 60)]),
                    netting=rng.random() < 0.3, base_pending=base, incumbents={k(o) for o in desired if rng.random() < .3},
                    rank={k(o): rng.uniform(0, 500) for o in desired if rng.random() < 0.7})
        risk0 = rng.uniform(0, 10)
        ref = flat(filter_orders(desired, pos, tick, cfg, risk0, **args))
        for keep in (None, {}, {("nowhere", "bid", 50): (5.0, 2.0)}):
            assert flat(filter_orders(desired, pos, tick, cfg, risk0, **args, keep=keep)) == ref


def test_a_reshuffle_of_the_ranking_no_longer_shrinks_one_resting_order_and_grows_another():
    """$20 of open risk holds 14 + 6 contracts at 96c. E1 ranked first and rests at 14, E2 at 6. The books move and
    E2 now ranks first: the old filter gives E2 its 14 (an amend up: the back of the queue) and cuts E1 to 6."""
    want = [no_bid("E1"), no_bid("E2")]
    old, _ = filter_orders(want, {}, TICK, CAPS, 0.0, incumbents={k(o) for o in want},
                           rank={k(want[0]): 200.0, k(want[1]): 50.0})
    assert sizes(old) == {"E2": 14.0, "E1": 6.0}                       # the churn
    keep = {k(want[0]): (14.0, 3.5), k(want[1]): (6.0, 2.0)}
    new, _ = filter_orders(want, {}, TICK, CAPS, 0.0, incumbents=set(keep),
                           rank={k(want[0]): 400.0, k(want[1]): 100.0}, keep=keep)
    assert sizes(new) == {"E2": 6.0, "E1": 14.0}                       # both left as they rest


def test_a_far_more_reachable_new_quote_still_takes_the_budget_and_the_cut_is_the_smallest_that_fits():
    resting, new_q = no_bid("E1"), no_bid("E3")
    keep = {k(resting): (14.0, 3.5)}
    for ahead_new, first, want in ((80.0, "E3", {"E3": 14.0, "E1": 6.0}),      # 80 < 0.25 x 400: it goes first
                                   (120.0, "E1", {"E1": 14.0, "E3": 6.0})):    # 120 is not: the resting one first
        rank = {k(resting): keep_rank(400.0, True, 0.25), k(new_q): keep_rank(ahead_new, False, 0.25)}
        kept, _ = filter_orders([resting, new_q], {}, TICK, CAPS, 0.0, incumbents=set(keep), rank=rank, keep=keep)
        assert kept[0].event == first and sizes(kept) == want
    caps = RiskConfig(max_loss_per_bracket_usd=15, max_loss_per_event_usd=20, max_total_open_risk_usd=23)
    kept, _ = filter_orders([new_q, resting], {}, TICK, caps, 0.0, incumbents=set(keep),
                            rank={k(resting): 400.0, k(new_q): 10.0}, keep=keep)
    assert sizes(kept) == {"E3": 14.0, "E1": 9.0}     # the largest that fits (9 x 96c <= $9.56), not 30% steps (6)


def test_small_size_changes_are_not_made():
    o = no_bid("E1", n=9.0)
    assert keep_deadband(10.0, SleeveAConfig()) == 2.5 and keep_deadband(4.0, SleeveAConfig()) == 2.0
    k10 = {k(o): (10.0, keep_deadband(10.0, SleeveAConfig()))}
    run = lambda n, keep, caps=CAPS: filter_orders([no_bid("E1", n=n)], {}, TICK, caps, 0.0,  # noqa: E731
                                                   incumbents=set(keep), keep=keep)[0][0].count
    assert run(9.0, k10) == 10.0                                        # wants 9: left at 10
    assert run(6.0, k10) == 6.0                                         # wants 6: a real change
    assert run(14.0, k10) == 10.0                                       # 14 is not twice 10: no growth
    k4 = {k(o): (4.0, 2.0)}
    assert run(14.0, k4) == 14.0                                        # 4 -> 14: grown (more than twice)
    k1 = {k(o): (1.0, 2.0)}
    assert run(3.0, k1) == 1.0                                          # 1 -> 3: within the deadband
    kf = {k(o): (2.78, 2.0)}
    assert run(5.0, kf) == 2.78                                         # a fractional resting size stays exact
    assert run(14.0, kf) == pytest.approx(14.0)                         # ... and grows like any other
    tight = RiskConfig(max_loss_per_bracket_usd=2.5, max_loss_per_event_usd=20, max_total_open_risk_usd=20)
    assert run(5.0, kf, tight) == 2.0                                   # the caps need a cut: largest whole fit


def test_growth_comes_after_resting_orders_and_far_more_reachable_quotes_but_before_other_new_ones():
    r1, n1 = no_bid("E1"), no_bid("E2")
    keep = {k(r1): (4.0, 2.0)}
    caps23 = RiskConfig(max_loss_per_bracket_usd=15, max_loss_per_event_usd=20, max_total_open_risk_usd=23)
    kept, _ = filter_orders([n1, r1], {}, TICK, caps23, 0.0, incumbents=set(keep), keep=keep)
    assert sizes(kept) == {"E1": 14.0, "E2": 9.0}            # r1 grown 4 -> 14 first, the new quote gets the rest
    rank = {k(r1): 400.0, k(n1): keep_rank(50.0, False, 0.25)}          # n1 far more reachable: before r1
    kept, _ = filter_orders([n1, r1], {}, TICK, CAPS, 0.0, incumbents=set(keep), keep=keep, rank=rank)
    assert sizes(kept) == {"E2": 14.0, "E1": 4.0}            # n1 first; r1 keeps its 4; 2 more would not double it


def caps_hold(kept, pos, tick, cfg, cash_cap, risk0, cash0):
    by_ev = defaultdict(list)
    for o in kept:
        by_ev[o.event].append(o)
    risk = risk0
    for e, lst in by_ev.items():
        p = pos.get(e, [])
        wc = event_worst_case(p, lst, tick[e])
        risk += wc - event_worst_case(p, [], tick[e])
        assert wc <= cfg.max_loss_per_event_usd + 1e-9
        for o in lst:
            assert bracket_cost(p, lst, o.ticker) <= cfg.max_loss_per_bracket_usd + 1e-9
    assert risk <= cfg.max_total_open_risk_usd + 1e-9
    if cash_cap is not None:
        assert cash0 + sum(o.count * o.unit_cost for o in kept) <= cash_cap + 1e-9


def test_every_cap_still_holds_and_sizes_only_change_by_more_than_the_deadband():
    rng = random.Random(11)
    for _ in range(400):
        evs, tick, desired, pos, cfg = random_book(rng, sleeves=("A",))
        keep = {k(o): (rng.choice([rng.uniform(0.3, 30), float(rng.randint(1, 30))]), rng.uniform(0, 4))
                for o in desired if rng.random() < 0.6}
        cash_cap = rng.choice([None, rng.uniform(5, 60)])
        risk0 = sum(event_worst_case(v, [], tick[e]) for e, v in pos.items())
        cash0 = sum(p.no_cost for v in pos.values() for p in v)
        kept, dropped = filter_orders(desired, pos, tick, cfg, risk0, cash_used=cash0, cash_cap=cash_cap,
                                      incumbents=set(keep), rank={k(o): rng.uniform(0, 500) for o in desired},
                                      keep=keep)
        caps_hold(kept, pos, tick, cfg, cash_cap, risk0, cash0)
        assert len(kept) + len(dropped) == len(desired)
        for o in kept:
            want = next(d for d in desired if d.key == o.key).count
            if k(o) not in keep:
                assert o.count <= want
                continue
            r, band = keep[k(o)]
            assert o.count <= max(want, r) + 1e-9
            if o.count > r + 1e-9:                                      # grown: by more than the deadband, past 2x
                assert o.count - r > band and o.count > 2 * r


def test_the_runner_ranks_a_resting_order_by_its_contracts_ahead_and_a_new_quote_by_them_over_the_ratio():
    wall = {"no_dollars": [["0.9700", "7.00"], ["0.9900", "1500.00"]]}
    on = SleeveAConfig(max_contracts_ahead=1000, keep_resting=True, keep_resting_ratio=0.25)
    s = Stub({"T": wall}, resting=[("T", "ask", 3, 7.0)])
    s.cfg_a = on
    rank = Bot.reach_rank(s, [DesiredOrder("T", "ask", 3, 7.0, "A", 0.004, event="E")])
    assert rank == {("T", "ask", 3): 1500.0} and s.reach_stats["ahead_of_kept"] == [750]   # cutoff as before
    s2 = Stub({"T": {"no_dollars": [["0.9700", "100.00"]]}})
    s2.cfg_a = on
    assert Bot.reach_rank(s2, [DesiredOrder("T", "ask", 3, 7.0, "A", 0.004, event="E")]) == {("T", "ask", 3): 400.0}
    s3 = Stub({"T": wall}, resting=[("T", "ask", 3, 7.0)])               # off: exactly the old ranks
    assert Bot.reach_rank(s3, [DesiredOrder("T", "ask", 3, 7.0, "A", 0.004, event="E")]) == {("T", "ask", 3): 750.0}


def test_the_default_ratio_ranks_exactly_as_today():
    """keep_resting_ratio 0.5 (the default) gives every quote twice today's rank, so the budget goes in today's order
    and only the sizes are kept (books missing included)."""
    rng = random.Random(5)
    for _ in range(200):
        books, resting, desired = {}, [], []
        for i in range(rng.randint(1, 8)):
            t = f"T{i}"
            if rng.random() < 0.85:
                books[t] = {"no_dollars": [[f"{p / 100:.4f}", f"{rng.uniform(0, 900):.2f}"]
                                           for p in rng.sample(range(94, 100), rng.randint(1, 4))]}
            price = rng.randint(1, 6)
            desired.append(DesiredOrder(t, "ask", price, 5.0, "A", 0.004, event="E"))
            if rng.random() < 0.5:
                resting.append((t, "ask", price, float(rng.randint(1, 20))))
        today = Stub(dict(books), resting=list(resting))
        on = Stub(dict(books), resting=list(resting))
        on.cfg_a = SleeveAConfig(max_contracts_ahead=1000, keep_resting=True)
        r0, r1 = Bot.reach_rank(today, desired), Bot.reach_rank(on, desired)
        assert set(r0) == set(r1) and all(r1[x] == 2 * r0[x] for x in r0)
        assert today.reach_stats == on.reach_stats


def test_within_an_event_the_resting_bracket_keeps_the_budget_unless_far_less_reachable():
    from wxbot.models.fair_value import FairValueResult, LadderQuote
    tick = ["E-T60", "E-B61.5", "E-B63.5"]
    brs = [type("B", (), {"ticker": t})() for t in tick]
    quotes = {t: LadderQuote(yes_bid=None, yes_ask=3) for t in tick}
    books = {"E-T60": {"no_dollars": [["0.9700", "900"]]}, "E-B61.5": {"no_dollars": [["0.9700", "310"]]},
             "E-B63.5": {"no_dollars": [["0.9700", "90"]]}}
    ev = EventBook(event="E", series="S", kind="high", station="X", local_date=None, close_ts=time.time() + 86400,
                   brackets=brs, quotes=quotes, books=books)
    fv = FairValueResult(probs={"E-T60": 0.002, "E-B61.5": 0.004, "E-B63.5": 0.006}, market={}, nbm={},
                         disagreement=0.0)
    base = dict(decided_only=True, decided_prob=0.97, min_edge_c=2.0, improve_if_edge_c=99, clip_contracts=12,
                decided_max_cost_usd=10, min_price_c=60, max_price_c=98, max_contracts_ahead=1000)
    resting = {("E-T60", "ask", 3): 10.0}            # ours rests on T60: 890 others ahead of it
    pick = lambda **kw: [o.ticker for o in sleeve_a.propose(ev, fv, time.time(), SleeveAConfig(**base, **kw),  # noqa
                                                            resting=resting)[0]]
    assert pick() == ["E-B63.5"]                                        # 90 < 445 (half of 890): the budget moves
    assert pick(keep_resting=True, keep_resting_ratio=0.25) == ["E-B63.5"]   # 90 < 0.25 x 890: far more reachable
    books["E-B63.5"] = {"no_dollars": [["0.9700", "300"]]}
    assert pick(keep_resting=True, keep_resting_ratio=0.25) == ["E-T60"]     # 300 is not: the resting one keeps it
    assert pick() == ["E-B63.5"]                                        # the old rule moved it (300 < 445)
    assert pick(keep_resting=True) == ["E-B63.5"]                       # the default ratio 0.5: today's order
    books["E-B61.5"] = {"no_dollars": [["0.9700", "600"]]}
    books["E-B63.5"] = {"no_dollars": [["0.9700", "450"]]}
    assert pick() == pick(keep_resting=True) == ["E-T60"]               # 445 < 450 < 600: both keep the resting one


def replay_event(tmp_path):
    """One New York event on Aug 8 (8 h of 5-minute candles, 92/94 on 86-87), and YES sellers at our 92c bid: six
    prints of 50 that clear the 300 ahead of us, then 5 more at 72 minutes."""
    import datetime as dt
    import json
    t0 = dt.datetime(2026, 8, 8, 12, tzinfo=dt.timezone.utc).timestamp()
    iso = lambda t: dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
    ev = "KXHIGHNY-26AUG08"
    (tmp_path / "cd").mkdir()
    sell = lambda m, n: {"created_time": iso(t0 + 60 * m), "yes_price_dollars": "0.9200", "count_fp": str(n),  # noqa
                         "taker_side": "no"}
    ms = []
    for k, st, fl, cp, (b, a), res in (("T86", "less", None, 86, (0.03, 0.05), "no"),
                                       ("B86.5", "between", 86, 87, (0.92, 0.94), "yes"),
                                       ("B88.5", "between", 88, 89, (0.01, 0.03), "no")):
        m = {"ticker": f"{ev}-{k}", "strike_type": st, "result": res, "open_time": iso(t0),
             "close_time": iso(t0 + 8 * 3600), "trades": []}
        if fl is not None:
            m["floor_strike"] = fl
        if cp is not None:
            m["cap_strike"] = cp
        if k == "B86.5":
            m["trades"] = [sell(m_, 50) for m_ in (16, 21, 26, 31, 36, 41)] + [sell(72, 5)]
        ms.append(m)
        json.dump([[t0 + i * 60, b, a] for i in range(0, 8 * 60, 5)], open(tmp_path / "cd" / f"{ev}-{k}.json", "w"))
    json.dump(ms, open(tmp_path / f"{ev}.json", "w"))
    return ev, str(tmp_path / f"{ev}.json")


def test_the_replay_counts_shrink_and_regrow_and_grow_by_amend_loses_the_queue(tmp_path, monkeypatch):
    """The risk filter wants 10 contracts, then 2 (a shrink, keeping the place in line the six prints cleared), then 10
    again (a regrowth). The replay's top-up keeps the 2 at the front, so the next 5 at our price fill 2 of them; growing
    by amend, as the live order manager does, sends the whole order to the back: nothing fills."""
    from wxbot.backtest import joint
    ev, fn = replay_event(tmp_path)
    calls = []

    def schedule(desired, *a, **kw):
        if not desired:
            return [], []
        calls.append(1)
        n = 10.0 if len(calls) < 9 or len(calls) >= 13 else 2.0     # calls 9-12 (50-65 min) want 2
        return [DesiredOrder(f"{ev}-B86.5", "bid", 92, n, "A", 0.95, event=ev)], []
    monkeypatch.setattr(joint, "filter_orders", schedule)
    cfg = SleeveAConfig(min_edge_c=1.0, improve_if_edge_c=99, max_disagreement=1.0)
    got = {}
    for amend in (False, True):
        calls.clear()
        r = joint.run_joint([fn], str(tmp_path / "cd"), cfg_a=cfg, queue_ahead=300.0, grow_amend=amend)
        got[amend] = r
        assert (r["shrinks"], r["shrinks_to_1"], r["grows"], r["regrows"], r["replaced"]) == (1, 0, 1, 1, 0)
    assert got[False]["contracts"] == 2.0 and got[False]["fills"] == 1          # the front 2 filled
    assert got[True]["contracts"] == 0.0 and got[True]["fills"] == 0            # back of the queue: no fill
    assert abs(got[False]["realized"] - 2 * 0.08) < 1e-9 and got[True]["realized"] == 0.0


def live_pair(tmp_path, keep_on):
    """The live path end to end (risk_bot on the fake exchange): two near-certain NO bids on New York's tails, a $20
    cash cap that holds 14 + 6 of them, books where T70 is the more reachable; then the books move."""
    fake = FakeKalshi()
    bot = risk_bot(tmp_path, fake)
    ev = fake.ev
    tick = [m["ticker"] for m in fake.markets]
    bot.ev_tickers[ev] = tick
    bot.risk_cfg.max_cash_deployed_usd = 20.0
    bot.cfg_a = SleeveAConfig(max_contracts_ahead=1000, keep_resting=keep_on)
    bot.events = {ev: EventBook(event=ev, series="KXHIGHNY", kind="high", station="KNYC", local_date=None,
                                close_ts=time.time() + 6 * 3600, markets=fake.markets,
                                books={tick[0]: {"no_dollars": [["0.9700", "50.00"]]},
                                       tick[4]: {"no_dollars": [["0.9700", "200.00"]]}})}
    bot.acct_ts = {kk: time.time() for kk in ("sync", "balance", "positions")}
    bot.eq_readings = [625.0] * 3
    bot.account_stale_s = 30
    want = [DesiredOrder(tick[0], "ask", 3, 14.0, "A", 0.004, event=ev),
            DesiredOrder(tick[4], "ask", 3, 14.0, "A", 0.004, event=ev)]
    bot.execute(want)
    assert {c[0]: c[3] for c in fake.created} == {tick[0]: 14.0, tick[4]: 6.0}, fake.created
    # the books move (our own contracts are in them): T75 is now the more reachable
    bot.events[ev].books = {tick[0]: {"no_dollars": [["0.9700", "414.00"]]},
                            tick[4]: {"no_dollars": [["0.9700", "106.00"]]}}
    fake.amended.clear()
    bot.execute(want)
    return fake


def test_live_path_without_the_option_shrinks_one_and_grows_the_other(tmp_path):
    fake = live_pair(tmp_path, False)
    assert sorted(a[2] for a in fake.amended) == [6.0, 14.0]           # 14 -> 6 down, 6 -> 14 up (back of queue)


def test_live_path_with_the_option_leaves_both_orders_alone(tmp_path):
    fake = live_pair(tmp_path, True)
    assert fake.amended == [] and len(fake.created) == 2 and fake.cancelled == []
