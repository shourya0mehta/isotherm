"""Round 8 (exit rules, Oct 9; docs/results/r8_exit_rules.txt): the overlay that lays exit rules over a replay's fills
(research/ml/r8_exit_lib.py). Nothing in wxbot changes: the recorder wraps run_joint's fv_fn hook and reads its
fill_log, so a replay with it is the replay without it. These tests pin the accounting (a sale changes the P&L against
holding by q * (s - fee(s) - payout)), the triggers, the dead-side reading of the bot's METAR kill margin, and that
the recorder sees every step after the first fill without changing the run."""
import datetime as dt
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("r8_exit_lib", os.path.join(HERE, "..", "research", "ml",
                                                                            "r8_exit_lib.py"))
L = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("r8_exit_lib", L)      # dataclasses look their module up while the class is built
_spec.loader.exec_module(L)
INF = 10 ** 6


def test_fee_and_the_held_sides_quote():
    assert L.fee(0.5) == pytest.approx(0.0175) and L.fee(0.97) == pytest.approx(0.07 * 0.97 * 0.03)
    bid, mid, tight = L.side_quote("yes", 95, 97)
    assert bid == pytest.approx(0.95) and mid == pytest.approx(0.96) and tight
    bid, mid, tight = L.side_quote("no", 3, 5)           # NO bid = 1 - YES ask
    assert bid == pytest.approx(0.95) and mid == pytest.approx(0.96) and tight
    bid, mid, tight = L.side_quote("no", 2, -1)          # nobody offers YES: nobody bids NO
    assert bid is None and mid == pytest.approx(1 - 0.51) and not tight
    assert L.side_quote("yes", 2, 99)[2] is False         # a 97c spread is not a two-sided book


def test_dead_side_uses_the_bots_kill_margin():
    # YES on 86-87: dead once the METAR max rounds to 89 (the CLI max is taken to be at least that less 1 F)
    assert L.dead_side(86, 87, "yes", 88.6) and not L.dead_side(86, 87, "yes", 88.4)
    assert not L.dead_side(86, 87, "yes", None)
    # NO on the open top bracket (>= 91): dead once the max rounds to 92; NO on an inner bracket is never dead early
    assert L.dead_side(91, INF, "no", 91.6) and not L.dead_side(91, INF, "no", 91.4)
    assert not L.dead_side(86, 87, "no", 99.0)
    # without the margin the rounded METAR max itself counts
    assert L.dead_side(86, 87, "yes", 87.6, margin=0) and not L.dead_side(86, 87, "yes", 87.6)
    assert L.dead_side(91, INF, "no", 90.6, margin=0) and not L.dead_side(91, INF, "no", 90.4, margin=0)
    st = [(600, 0.0, 0.05, 0.06, False, True, True)]
    assert L.walk(st, [(0.0, 0.95, 4.0)], 0.0, L.Rule("dead", "dead"), {}) == []
    assert len(L.walk(st, [(0.0, 0.95, 4.0)], 0.0, L.Rule("dead0", "dead0"), {})) == 1


def _steps(rows):
    """(t, fair, bid, mid) -> walk's step tuples (no dead flag, two-sided book)."""
    return [(t, f, b, m, False, True) for t, f, b, m in rows]


def test_model_exit_sells_at_the_bid_and_a_later_fill_reopens_the_side():
    fills = [(0.0, 0.95, 10.0), (600.0, 0.96, 5.0), (3000.0, 0.94, 4.0)]
    steps = _steps([(300, 0.97, 0.94, 0.95),          # 10 held, bid - fee < fair: hold
                    (900, 0.90, 0.95, 0.955),         # 15 held, 0.95 - fee > 0.90: sell 15 at 95c
                    (1200, 0.80, 0.93, 0.94),         # flat
                    (3300, 0.80, 0.92, 0.93)])        # the 4 bought at 3000 s: sell at 92c
    ex = L.walk(steps, fills, payout=0.0, rule=L.Rule("m", "model"), info={})
    assert [round(e["qty"]) for e in ex] == [15, 4]
    assert ex[0]["entry"] == pytest.approx((10 * 0.95 + 5 * 0.96) / 15)
    assert ex[0]["delta"] == pytest.approx(15 * (0.95 - L.fee(0.95)))         # the side lost: all of it saved
    ex_w = L.walk(steps, fills, payout=1.0, rule=L.Rule("m", "model"), info={})
    assert ex_w[0]["delta"] == pytest.approx(15 * (0.95 - L.fee(0.95) - 1.0)) and ex_w[0]["won"]
    # a 5c margin waits for a bigger gap; one cent of slippage fills 1c under the bid
    assert [round(e["qty"]) for e in L.walk(steps, fills, 0.0, L.Rule("m5", "model", 0.05), {})] == [15, 4]
    assert L.walk(steps[:2], fills, 0.0, L.Rule("m10", "model", 0.10), {}) == []
    s1 = L.walk(steps, fills, 0.0, L.Rule("m_s1", "model", 0.0, 0.01), {})
    assert s1[0]["exit"] == pytest.approx(0.94) and s1[0]["delta"] == pytest.approx(15 * (0.94 - L.fee(0.94)))
    # nothing is evaluated from the holdout cut on
    assert L.walk(steps, fills, 0.0, L.Rule("m", "model"), {}, cut=900) == []


def test_price_drop_stops():
    fills = [(0.0, 0.95, 10.0)]
    rows = [(1800, 0.9, 0.62, 0.64), (9000, 0.9, 0.60, 0.62), (9300, 0.9, 0.45, 0.48)]
    # 30c under the entry within 2 hours: sells at 30 minutes
    ex = L.walk(_steps(rows), fills, 1.0, L.Rule("d", "drop30"), {})
    assert len(ex) == 1 and ex[0]["t"] == 1800 and ex[0]["exit"] == pytest.approx(0.62)
    # the same drop after 2 hours does not count; under 50c does
    ex = L.walk(_steps(rows[1:]), fills, 1.0, L.Rule("d", "drop30"), {})
    assert ex == []
    ex = L.walk(_steps(rows), fills, 1.0, L.Rule("b", "below50"), {})
    assert len(ex) == 1 and ex[0]["t"] == 9300
    # "falls below 50c" needs a lot bought at 50c or more; the literal variant does not
    cheap = [(0.0, 0.26, 5.0)]
    assert L.walk(_steps([(600, 0.9, 0.40, 0.45)]), cheap, 1.0, L.Rule("b", "below50"), {}) == []
    assert len(L.walk(_steps([(600, 0.9, 0.40, 0.45)]), cheap, 1.0, L.Rule("b", "below50any"), {})) == 1
    # the two-sided variant ignores a mid read off an empty book
    st = [(600, 0.9, 0.01, 0.495, False, False)]
    assert L.walk(st, fills, 1.0, L.Rule("bt", "below50", tight=True), {}) == []
    assert len(L.walk(st, fills, 1.0, L.Rule("b", "below50"), {})) == 1


def test_dead_side_exit_needs_a_bid_worth_more_than_the_fee():
    fills = [(0.0, 0.97, 20.0)]
    st = [(600, 0.0, None, 0.005, True, False), (900, 0.0, 0.03, 0.04, True, True)]
    ex = L.walk(st, fills, 0.0, L.Rule("dead", "dead"), {})
    assert len(ex) == 1 and ex[0]["t"] == 900 and ex[0]["delta"] == pytest.approx(20 * (0.03 - L.fee(0.03)))


def test_depth_proxy_caps_the_sale():
    # selling YES at 95c: other takers sold YES (taker_side 'no') at 95c or more within the hour: 3 + 4 contracts
    trades = [(100.0, 0.95, 3.0, "no"), (200.0, 0.94, 50.0, "no"), (300.0, 0.96, 4.0, "no"),
              (400.0, 0.97, 9.0, "yes"), (5000.0, 0.99, 100.0, "no")]
    fn = L.depth_proxy(trades, "yes")
    assert fn(0.0, 0.95) == pytest.approx(7.0)
    # selling NO at 3c = buying YES at 97c: takers who bought YES at 97c or less
    assert L.depth_proxy(trades, "no")(0.0, 0.03) == pytest.approx(9.0)
    fills = [(0.0, 0.95, 10.0)]
    ex = L.walk(_steps([(60, 0.80, 0.95, 0.96), (4000, 0.80, 0.95, 0.96)]), fills, 1.0,
                L.Rule("c", "model", cap=True), {}, depth_fn=fn)
    assert [e["qty"] for e in ex] == [7.0, 3.0]         # 3 left after the first sale; 3 more fill an hour later


def _event(tmp_path):
    """One ladder: 86-87 quoted 92/94 all day (it wins), a taker sells YES at 90c three hours in (fills a 92c bid)."""
    ev = "KXHIGHTOKC-26AUG08"
    t0 = dt.datetime(2026, 8, 8, 12, tzinfo=dt.timezone.utc).timestamp()
    iso = lambda t: dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
    mk = [("T86", "less", None, 86, (0.03, 0.05), "no"), ("B86.5", "between", 86, 87, (0.92, 0.94), "yes"),
          ("B88.5", "between", 88, 89, (0.01, 0.03), "no")]
    (tmp_path / "cd").mkdir()
    ms = []
    for k, st, fl, cp, (b, a), res in mk:
        tk = f"{ev}-{k}"
        m = {"ticker": tk, "strike_type": st, "result": res, "open_time": iso(t0), "close_time": iso(t0 + 8 * 3600),
             "trades": []}
        if fl is not None:
            m["floor_strike"] = fl
        if cp is not None:
            m["cap_strike"] = cp
        if k == "B86.5":
            m["trades"] = [{"created_time": iso(t0 + 3 * 3600), "yes_price_dollars": "0.9000", "count_fp": "50",
                            "taker_side": "no"}]
        ms.append(m)
        json.dump([[t0 + i * 60, b, a] for i in range(0, 8 * 60, 5)], open(tmp_path / "cd" / f"{tk}.json", "w"))
    fn = tmp_path / f"{ev}.json"
    json.dump(ms, open(fn, "w"))
    return ev, t0, fn


def test_recorder_sees_each_step_after_the_first_fill_and_changes_nothing(tmp_path):
    from wxbot.backtest.joint import run_joint
    from wxbot.strategy.sleeve_a import SleeveAConfig
    ev, t0, fn = _event(tmp_path)
    cfg = SleeveAConfig(min_edge_c=1.0, improve_if_edge_c=99, max_disagreement=1.0)
    plain = run_joint([str(fn)], str(tmp_path / "cd"), cfg_a=cfg, fill_rule="through")
    fills = []
    rec = L.Recorder(None, fills)
    r = run_joint([str(fn)], str(tmp_path / "cd"), cfg_a=cfg, fill_rule="through", fv_fn=rec, fill_log=fills)
    assert abs(r["realized"] - plain["realized"]) < 1e-9 and r["contracts"] == plain["contracts"] > 0
    arr = rec.arrays()
    assert set(arr) == {f"{ev}-B86.5"} and fills and all(f["ticker"] == f"{ev}-B86.5" for f in fills)
    t, p, b, a = arr[f"{ev}-B86.5"]
    assert t[0] >= fills[0]["ts"] and t[-1] < t0 + 8 * 3600 - 60
    assert (b == 92).all() and (a == 94).all() and ((p > 0) & (p < 1)).all()
    # every 5-minute step from the first fill to a minute before the close
    assert len(t) == len([x for x in range(int(min(t)), int(t0 + 8 * 3600 - 60), 300)])


def test_run_joint_exit_hook_is_off_by_default_and_books_a_sale_with_its_event(tmp_path):
    """wxbot.backtest.joint.run_joint exit_fn (round 8): selling the 86-87 YES at 95c an hour after the fill gives up
    the $1 payout for 95c less the fee, booked on the event's settlement day; without the hook nothing changes."""
    import inspect
    from wxbot.backtest.joint import run_joint
    from wxbot.strategy.sleeve_a import SleeveAConfig
    assert inspect.signature(run_joint).parameters["exit_fn"].default is None
    ev, t0, fn = _event(tmp_path)
    cfg = SleeveAConfig(min_edge_c=1.0, improve_if_edge_c=99, max_disagreement=1.0)
    fills0 = []
    plain = run_joint([str(fn)], str(tmp_path / "cd"), cfg_a=cfg, fill_rule="through", fill_log=fills0)
    calls = []

    def sell(name, t, quotes, fv, held):
        calls.append((t, len(held)))
        return [(p.ticker, "yes", p.yes_qty, 95, L.fee(0.95) * p.yes_qty) for p in held
                if p.yes_qty > 0 and t >= t0 + 4 * 3600]
    log, fills = [], []
    r = run_joint([str(fn)], str(tmp_path / "cd"), cfg_a=cfg, fill_rule="through", exit_fn=sell, exit_log=log,
                  fill_log=fills)
    q = sum(f["count"] for f in fills0)
    assert q > 0 and len(log) == 1 and log[0]["count"] == pytest.approx(q) and log[0]["ts"] >= t0 + 4 * 3600
    assert log[0]["pnl"] == pytest.approx(q * (0.95 - 0.92) - L.fee(0.95) * q)
    # against holding: q * (0.95 - fee - 1), on the same settlement day
    assert r["realized"] == pytest.approx(plain["realized"] + q * (0.95 - L.fee(0.95) - 1.0))
    assert r["by_event"][ev] == pytest.approx(r["realized"]) and list(r["daily"]) == list(plain["daily"])
    held = {row[0]: row for row in r["positions"][ev]}
    assert held[f"{ev}-B86.5"][2] == pytest.approx(0.0)          # nothing left to settle
    # called at every step of the quoting window; it holds something from the first fill on
    assert calls[0][1] == 0 and min(t for t, n in calls if n) >= fills0[0]["ts"]
