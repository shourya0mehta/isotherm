"""Runner state handling: restart restore, settlement bookkeeping, risk on closed-but-unsettled events."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wxbot.app.runner import Bot
from wxbot.strategy.types import DesiredOrder, Fill, Position

CFG = {
    "mode": "paper", "env": "prod",
    "loop": {"tick_seconds": 10, "universe_every_s": 180, "books_every_s": 15, "trades_every_s": 20,
             "metar_every_s": 300, "nbm_every_s": 900, "index_every_s": 60, "settle_every_s": 900,
             "positions_every_s": 20},
    "sleeve_a": {"enabled": True, "series": ["KXHIGHNY"], "decided_only": True},
    "sleeve_b": {"enabled": False, "shadow": False},
    "risk": {"max_loss_per_bracket_usd": 30, "max_loss_per_event_usd": 40, "max_total_open_risk_usd": 350,
             "daily_loss_stop_usd": 40, "total_drawdown_stop_usd": 90},
}


def make_bot(tmp_path, mode="paper"):
    cfg = dict(CFG, mode=mode, db_path=str(tmp_path / "t.sqlite"), status_path=str(tmp_path / "st.json"))
    if mode == "live":
        os.environ.pop("KALSHI_KEY_ID", None)
        b = Bot.__new__(Bot)  # live needs a key; build a paper bot and flip the mode for settlement tests
        Bot.__init__(b, dict(cfg, mode="paper"), base_dir=str(tmp_path))
        b.mode = "live"
        return b
    return Bot(cfg, base_dir=str(tmp_path))


def fill(t, side, pc, n, ts=None):
    return Fill(ts or time.time(), t, t.rsplit("-", 1)[0], side, pc, n, "A", False, 0.0, "x")


def test_restore_skips_settled_and_books_realized(tmp_path):
    b = make_bot(tmp_path)
    b.store.add_fill("paper", fill("KXHIGHNY-26OCT01-T80", "ask", 4, 25))  # NO @96c, cost 24.00
    b.store.add_fill("paper", fill("KXHIGHNY-26OCT02-T90", "ask", 3, 10))  # NO @97c, still open
    b.store.set_settlement("KXHIGHNY-26OCT01-T80", "KXHIGHNY-26OCT01", "no", time.time(), None)
    b2 = make_bot(tmp_path)
    b2.restore_paper_state()
    assert set(b2.positions) == {"KXHIGHNY-26OCT02-T90"}
    assert abs(b2.risk.realized_total - 1.00) < 1e-9  # 25 - 24.00
    assert abs(b2.positions["KXHIGHNY-26OCT02-T90"].no_cost - 9.70) < 1e-9


def test_paper_settle_pops_position(tmp_path):
    b = make_bot(tmp_path)
    b.apply_fill(fill("KXHIGHNY-26OCT02-T90", "bid", 97, 10))  # YES @97c
    b.rest.market = lambda t: {"result": "no"}
    b.settle()
    assert "KXHIGHNY-26OCT02-T90" not in b.positions
    assert abs(b.risk.realized_total + 9.70) < 1e-9
    assert b.store.settled()["KXHIGHNY-26OCT02-T90"] == "no"


def test_closed_event_no_position_counts_as_risk(tmp_path):
    b = make_bot(tmp_path)
    ev = "KXHIGHNY-26OCT01"
    b.positions["KXHIGHNY-26OCT01-T80"] = Position("KXHIGHNY-26OCT01-T80", ev, no_qty=25, no_cost=24.0)
    b.ev_tickers[ev] = [f"{ev}-T80", f"{ev}-B81.5", f"{ev}-T88"]
    b.events = {}
    b.paper_reconcile = lambda kept, now: None
    b.execute([])
    assert abs(b.open_risk - 24.0) < 1e-9  # worst case: T80 wins and our NO pays nothing


def test_tickers_for_event_fetch_and_backoff(tmp_path):
    b = make_bot(tmp_path)
    calls = []

    def markets(**kw):
        calls.append(kw)
        raise RuntimeError("429")
    b.rest.markets = markets
    assert b.tickers_for_event("KXHIGHNY-26SEP30") is None   # unknown (counts full cost), not an empty ladder
    assert b.tickers_for_event("KXHIGHNY-26SEP30") is None   # backoff: no second call within a minute
    assert len(calls) == 1
    b.rest.markets = lambda **kw: [{"ticker": "KXHIGHNY-26SEP30-T70"}, {"ticker": "KXHIGHNY-26SEP30-B71.5"}]
    b.ev_tickers_miss.clear()
    assert b.tickers_for_event("KXHIGHNY-26SEP30") == ["KXHIGHNY-26SEP30-T70", "KXHIGHNY-26SEP30-B71.5"]


def test_live_settlements_only_bot_tickers(tmp_path):
    b = make_bot(tmp_path, mode="live")
    b.store.add_fill("live", fill("KXHIGHNY-26OCT01-T80", "ask", 4, 25))
    recs = [
        {"ticker": "KXHIGHNY-26OCT01-T80", "event_ticker": "KXHIGHNY-26OCT01", "market_result": "no",
         "revenue": 2500, "yes_total_cost_dollars": "0", "no_total_cost_dollars": "24.0000", "fee_cost": "0",
         "settled_time": "2026-10-02T10:00:00Z", "value": 0},
        {"ticker": "KXBTC-MANUAL", "event_ticker": "KXBTC", "market_result": "yes", "revenue": 100,
         "yes_total_cost_dollars": "0.50", "no_total_cost_dollars": "0", "fee_cost": "0.01",
         "settled_time": "2026-10-02T10:00:00Z", "value": 100},
    ]
    b.rest.settlements = lambda min_ts=None: recs
    b.settle()
    b.settle()  # idempotent
    assert abs(b.risk.realized_total - 1.0) < 1e-9
    n = b.store.c.execute("SELECT COUNT(*) FROM live_settlements").fetchone()[0]
    assert n == 1


def test_decided_budget_counts_held_positions(tmp_path):
    from wxbot.models.brackets import bracket_from_market
    from wxbot.models.fair_value import LadderQuote
    from wxbot.strategy.types import EventBook
    b = make_bot(tmp_path)
    ev = "KXHIGHNY-26OCT02"
    ms = [{"ticker": f"{ev}-T70", "strike_type": "less", "cap_strike": 70},
          {"ticker": f"{ev}-B70.5", "strike_type": "between", "floor_strike": 70, "cap_strike": 71},
          {"ticker": f"{ev}-T71", "strike_type": "greater", "floor_strike": 71}]
    brs = [bracket_from_market(m) for m in ms]
    quotes = {f"{ev}-T70": LadderQuote(1, 2), f"{ev}-B70.5": LadderQuote(97, 99), f"{ev}-T71": LadderQuote(1, 2)}
    b.events = {ev: EventBook(ev, "KXHIGHNY", "high", "KNYC", None, time.time() + 20 * 3600, ms, brs, quotes, {})}
    b.book_ts = {m["ticker"]: time.time() for m in ms}
    free = b.decide()
    b.positions[f"{ev}-B70.5"] = Position(f"{ev}-B70.5", ev, yes_qty=45, yes_cost=44.0)
    capped = b.decide()
    assert sum(o.count for o in free) > 0
    assert sum(o.count for o in capped) < sum(o.count for o in free)


def test_cash_cap_without_netting_limits_multi_bracket_nos():
    from wxbot.risk.limits import RiskConfig, filter_orders
    ev = "KXHIGHNY-26OCT02"
    ticks = [f"{ev}-T70", f"{ev}-B70.5", f"{ev}-B72.5", f"{ev}-T74"]
    # NO at 98c on three dead brackets: worst case small (only one can win), cash = full cost
    desired = [DesiredOrder(t, "ask", 2, 25.0, "A", 0.0, "decided", ev) for t in ticks[:3]]
    cfg = RiskConfig(max_loss_per_bracket_usd=30, max_loss_per_event_usd=40, max_total_open_risk_usd=350)
    kept_free, _ = filter_orders(desired, {}, {ev: ticks}, cfg, 0.0)
    assert sum(o.count * o.unit_cost for o in kept_free) > 70  # all three fit the worst-case caps
    kept, _ = filter_orders(desired, {}, {ev: ticks}, cfg, 0.0, cash_used=0.0, cash_cap=40.0, netting=False)
    assert sum(o.count * o.unit_cost for o in kept) <= 40.0 + 1e-9
    kept_net, _ = filter_orders(desired, {}, {ev: ticks}, cfg, 0.0, cash_used=0.0, cash_cap=40.0, netting=True)
    assert sum(o.count for o in kept_net) == 75  # with netting, cash = worst case (~24.5) fits


def test_live_reconcile_allow_new_false_places_nothing():
    from wxbot.execution.live import LiveOrderManager

    class R:
        def __init__(self):
            self.created = 0

        def create_order(self, *a, **k):
            self.created += 1
            return {"order_id": "x", "remaining_count": "1.00", "fill_count": "0.00"}

        def batch_cancel(self, ids):
            return {}
    r = R()
    m = LiveOrderManager(r)
    d = DesiredOrder("KXHIGHNY-26OCT02-T70", "ask", 2, 25.0, "A", 0.0, "decided", "KXHIGHNY-26OCT02")
    st = m.reconcile([d], {}, allow_new=False)
    assert r.created == 0 and st["placed"] == 0
    st = m.reconcile([d], {}, allow_new=True)
    assert r.created == 1 and st["placed"] == 1


def test_dead_man_expiry_refresh():
    from wxbot.execution.live import LiveOrder, LiveOrderManager

    class R:
        def __init__(self):
            self.amends = []

        def amend_order(self, oid, ticker, side, price, total, expiration_ts=None):
            self.amends.append((oid, price, total, expiration_ts))
            return {}
    r = R()
    m = LiveOrderManager(r, ttl_s=1200)
    now = time.time()
    close = int(now + 3 * 3600)
    soon = LiveOrder("a", "wxb-1", "T1", "ask", 3, 20.0, 5.0, now - 900, now + 300)   # inside last half of TTL
    fresh = LiveOrder("b", "wxb-2", "T2", "ask", 4, 25.0, 0.0, now - 60, now + 1100)  # not due
    near_close = LiveOrder("c", "wxb-3", "T3", "bid", 97, 10.0, 0.0, now - 900, now + 200)
    m.orders = {("T1", "ask"): [soon], ("T2", "ask"): [fresh], ("T3", "bid"): [near_close]}
    used = m.refresh_expiries({"T1": close, "T2": close, "T3": int(now + 200)}, budget=10)
    assert used == 1 and len(r.amends) == 1
    oid, price, total, exp = r.amends[0]
    assert oid == "a" and price == 3 and total == 25.0  # same price, filled + remaining: expiry-only amend
    assert now + 1100 < exp <= now + 1201
    assert m._expiry(int(now + 100), now) == int(now + 100)  # never past market close


def test_live_status_fields(tmp_path):
    import json as _json
    from wxbot.execution.live import LiveOrder, LiveOrderManager
    b = make_bot(tmp_path, mode="live")
    b.live = LiveOrderManager(object(), ttl_s=1200)
    now = time.time()
    b.live.orders = {("T1", "ask"): [LiveOrder("a", "wxb-1", "T1", "ask", 4, 25.0, 0.0, now, now + 900)]}
    st = b.write_status()
    assert st["live_resting_orders"] == 1 and abs(st["live_resting_cost_usd"] - 24.0) < 1e-9
    assert 14.0 < st["live_min_minutes_to_expiry"] <= 15.0
    assert _json.load(open(b.status_path))["live_resting_orders"] == 1


def test_live_fill_dedupe_across_restart(tmp_path):
    b = make_bot(tmp_path, mode="live")
    f = Fill(time.time(), "KXHIGHNY-26OCT02-T70", "KXHIGHNY-26OCT02", "ask", 4, 10.0, "?", False, 0.0, "fill-abc")
    assert b.store.add_fill("live", f) is True
    assert b.store.add_fill("live", f) is False  # same Kalshi fill_id seen again after a restart
    assert b.store.c.execute("SELECT COUNT(*) FROM fills WHERE mode='live'").fetchone()[0] == 1


def test_gate_report_pass_and_fail(tmp_path):
    from wxbot.app.report import gate_report
    b = make_bot(tmp_path)
    b.store.c.execute("INSERT INTO events_log VALUES (?,?,?)", (time.time() - 4 * 86400, "INFO", "start mode=paper"))
    for i in range(16):  # 16 brackets x 25 NO at 96c, all settle NO: +4c/contract
        t = f"KXHIGHNY-26SEP{10 + i}-T90"
        b.store.add_fill("paper", fill(t, "ask", 4, 25))
        b.store.set_settlement(t, t.rsplit("-", 1)[0], "no", time.time(), None)
    b.store.c.commit()
    ok, txt = gate_report(dict(CFG, db_path=str(tmp_path / "t.sqlite")))
    assert ok, txt
    t = "KXHIGHNY-26SEP30-T90"  # one flip: 25 NO at 96c settle YES (-$24)
    b.store.add_fill("paper", fill(t, "ask", 4, 25))
    b.store.set_settlement(t, t.rsplit("-", 1)[0], "yes", time.time(), None)
    ok2, txt2 = gate_report(dict(CFG, db_path=str(tmp_path / "t.sqlite")))
    assert not ok2 and "-1.88c" in txt2, txt2  # 400 x +4c = +$16, 25 x -96c = -$24: -$8 over 425 contracts


def test_dashboard_renders_empty_and_with_data(tmp_path):
    from wxbot.app.dashboard import write_dashboard
    b = make_bot(tmp_path)
    cfg = dict(CFG, db_path=str(tmp_path / "t.sqlite"), status_path=str(tmp_path / "st.json"))
    out = write_dashboard(cfg, out_path=str(tmp_path / "d.html"))
    html0 = open(out).read()
    assert "No fills yet" in html0 and "<title>wxbot dashboard</title>" in html0
    b.write_status()
    for i in range(3):
        t = f"KXHIGHNY-26SEP{10 + i}-T90"
        b.store.add_fill("paper", fill(t, "ask", 4, 25))
        b.store.set_settlement(t, t.rsplit("-", 1)[0], "no" if i else "yes", time.time() - 86400 * (3 - i), None)
    b.store.add_fill("paper", fill("KXHIGHNY-26OCT02-T88", "ask", 3, 10))
    html1 = open(write_dashboard(cfg, out_path=str(tmp_path / "d.html"))).read()
    assert "KXHIGHNY-26OCT02-T88" in html1 and "Settled markets" in html1 and "<svg" in html1
