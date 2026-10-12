import base64
import json
import datetime as dt
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes

from wxbot.kalshi.auth import KalshiAuth, sign_text
from wxbot.kalshi.fees import maker_fee, taker_fee
from wxbot.kalshi.rest import cents_to_str, count_str, str_to_cents
from wxbot.models.brackets import bracket_from_market
from wxbot.models.fair_value import LadderQuote, book_to_quote, ladder_fair_value, nbm_bracket_probs
from wxbot.strategy.lip import score_side, snapshot_value
from wxbot.strategy.types import DesiredOrder, Position
from wxbot.risk.limits import RiskConfig, event_worst_case, filter_orders
from wxbot.execution.paper import PaperBroker
from wxbot.strategy import sleeve_a
from wxbot.strategy.types import EventBook


# ------------------------------------------------------------------ auth
def test_ed25519_signature_verifies():
    k = Ed25519PrivateKey.generate()
    auth = KalshiAuth("kid", k)
    h = auth.headers("GET", "/trade-api/v2/portfolio/balance?limit=5", ts_ms=1700000000000)
    assert h["KALSHI-ACCESS-KEY"] == "kid"
    msg = b"1700000000000GET/trade-api/v2/portfolio/balance"  # query stripped
    k.public_key().verify(base64.b64decode(h["KALSHI-ACCESS-SIGNATURE"]), msg)


def test_rsa_pss_signature_verifies():
    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    sig = sign_text(k, "123POST/trade-api/v2/portfolio/events/orders")
    k.public_key().verify(base64.b64decode(sig), b"123POST/trade-api/v2/portfolio/events/orders",
                          padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                          hashes.SHA256())


# ------------------------------------------------------------------ wire formats + fees
def test_price_and_count_strings():
    assert cents_to_str(56) == "0.5600"
    assert str_to_cents("0.9900") == 99
    assert count_str(10) == "10.00"
    with pytest.raises(ValueError):
        cents_to_str(0)


def test_fees():
    assert taker_fee(50, 1) == pytest.approx(0.0175)
    assert taker_fee(90, 100) == pytest.approx(0.63)
    assert maker_fee(50, 100, "quadratic") == 0.0
    assert maker_fee(50, 100, "quadratic_with_maker_fees") == pytest.approx(0.4375)


# ------------------------------------------------------------------ brackets + fair value
LADDER = [
    {"ticker": "E-T81", "strike_type": "less", "floor_strike": None, "cap_strike": 81},
    {"ticker": "E-B81.5", "strike_type": "between", "floor_strike": 81, "cap_strike": 82},
    {"ticker": "E-B83.5", "strike_type": "between", "floor_strike": 83, "cap_strike": 84},
    {"ticker": "E-T84", "strike_type": "greater", "floor_strike": 84, "cap_strike": None},
]


def test_bracket_semantics():
    b = [bracket_from_market(m) for m in LADDER]
    assert b[0].contains_int(80) and not b[0].contains_int(81)
    assert b[1].contains_int(81) and b[1].contains_int(82) and not b[1].contains_int(83)
    assert b[3].contains_int(85) and not b[3].contains_int(84)


def test_nbm_probs_sum_to_one_and_center():
    b = [bracket_from_market(m) for m in LADDER]
    p = nbm_bracket_probs(b, 82.0, 1.5)
    assert sum(p.values()) == pytest.approx(1.0)
    assert max(p, key=p.get) == "E-B81.5"


def test_book_to_quote_uses_no_side_for_yes_ask():
    q = book_to_quote({"yes_dollars": [["0.40", "10"], ["0.45", "5"]], "no_dollars": [["0.50", "7"]]})
    assert q.yes_bid == 45 and q.yes_ask == 50


def test_fair_value_obs_kills_brackets():
    b = [bracket_from_market(m) for m in LADDER]
    quotes = {"E-T81": LadderQuote(2, 5), "E-B81.5": LadderQuote(8, 12), "E-B83.5": LadderQuote(60, 66),
              "E-T84": LadderQuote(20, 26)}
    fv = ladder_fair_value(b, quotes, kind="high", obs_max_f=83.6)
    assert "E-T81" in fv.dead and fv.probs["E-T81"] == 0.0 and not fv.conflict
    assert sum(fv.probs.values()) == pytest.approx(1.0)
    # same observation while the market still prices E-T81 at ~22%: no kill, event flagged instead
    q2 = dict(quotes, **{"E-T81": LadderQuote(20, 24), "E-B81.5": LadderQuote(40, 44), "E-B83.5": LadderQuote(25, 29)})
    fv2 = ladder_fair_value(b, q2, kind="high", obs_max_f=83.6)
    assert fv2.conflict and "E-B81.5" not in fv2.dead  # market still at ~37%: not killed, event flagged


# ------------------------------------------------------------------ LIP scoring
def test_lip_empty_side_lone_1c_bid_scores_full():
    s = score_side([], 1000, 0.5, ours=[(1, 1000)])
    assert s.qualifies and s.ref_cents == 1 and s.our_share == pytest.approx(1.0)


def test_lip_1c_bid_discounted_under_real_depth():
    s = score_side([(50, 600), (49, 600)], 1000, 0.5, ours=[(1, 1000)])
    # cumulative reaches target at 49c before reaching our 1c level -> we do not qualify at all
    assert s.qualifies and s.our_share == 0.0


def test_lip_snapshot_needs_both_sides():
    v = snapshot_value([(98, 1500)], [], 1000, 0.5, our_yes=None, our_no=[(1, 1000)])
    assert v["counts"] and v["frac"] == pytest.approx(0.5)
    v2 = snapshot_value([(98, 500)], [], 1000, 0.5, our_no=[(1, 1000)])
    assert not v2["counts"]


# ------------------------------------------------------------------ risk
def test_event_worst_case_mutually_exclusive():
    pos = [Position("A", "E", yes_qty=10, yes_cost=8.5), Position("B", "E", no_qty=10, no_cost=9.0)]
    wc = event_worst_case(pos, [], ["A", "B", "C"])
    # outcome B: lose YES A (8.5) and NO B (9.0) -> loss 17.5
    assert wc == pytest.approx(17.5)


def test_filter_orders_respects_caps():
    cfg = RiskConfig(max_loss_per_bracket_usd=30, max_loss_per_event_usd=60, max_total_open_risk_usd=300)
    d = [DesiredOrder("A", "bid", 90, 100, "A", 0.95, event="E")]
    kept, dropped = filter_orders(d, {}, {"E": ["A", "B"]}, cfg, 0.0)
    assert kept and kept[0].count * 0.90 <= 30 + 1e-9


# ------------------------------------------------------------------ paper fills
def test_paper_queue_then_fill():
    pb = PaperBroker()
    d = DesiredOrder("A", "bid", 85, 10, "A", 0.9, event="E")
    book = {"yes_dollars": [["0.85", "20"]], "no_dollars": []}
    o = pb.place(d, book, now=1000.0)
    assert o.queue_ahead == 20
    t = {"ticker": "A", "yes_price_dollars": "0.85", "count_fp": "25", "taker_side": "no",
         "created_time": dt.datetime.fromtimestamp(1001, dt.timezone.utc).isoformat()}
    f = pb.on_trade(t)
    assert len(f) == 1 and f[0].count == 5  # 20 ahead of us, then 5 for us
    t2 = dict(t, yes_price_dollars="0.80", count_fp="100")
    f2 = pb.on_trade(t2)  # trade-through below our bid fills the rest
    assert f2[0].count == 5 and o.remaining == 0


def test_paper_ask_side_fill_direction():
    pb = PaperBroker()
    d = DesiredOrder("A", "ask", 12, 10, "A", 0.05, event="E")  # long NO paying 88c
    o = pb.place(d, {"yes_dollars": [], "no_dollars": []}, now=1000.0)
    wrong = {"ticker": "A", "yes_price_dollars": "0.12", "count_fp": "5", "taker_side": "no",
             "created_time": dt.datetime.fromtimestamp(1001, dt.timezone.utc).isoformat()}
    assert pb.on_trade(wrong) == []  # takers buying NO do not hit our NO bid
    right = dict(wrong, taker_side="yes")
    assert pb.on_trade(right)[0].count == 5


# ------------------------------------------------------------------ sleeve A
def test_sleeve_a_quotes_favorites_only():
    b = [bracket_from_market(m) for m in LADDER]
    quotes = {"E-T81": LadderQuote(3, 5), "E-B81.5": LadderQuote(84, 88), "E-B83.5": LadderQuote(8, 10),
              "E-T84": LadderQuote(1, 2)}
    ev = EventBook("E", "KXHIGHNY", "high", "KNYC", dt.date(2026, 10, 2), close_ts=10 ** 12,
                   markets=LADDER, brackets=b, quotes=quotes)
    fv = ladder_fair_value(b, quotes, kind="high")
    orders, _ = sleeve_a.propose(ev, fv, now_ts=0, cfg=sleeve_a.SleeveAConfig())
    sides = {(o.ticker, o.side) for o in orders}
    assert ("E-B81.5", "bid") in sides  # favorite YES
    assert all(o.edge_per_contract > 0 for o in orders)
    # never a YES bid on a longshot
    assert not any(o.side == "bid" and o.fair_yes < 0.5 for o in orders)


def test_daily_halt_resumes_next_day_but_drawdown_is_hard():
    from wxbot.risk.limits import RiskState, RiskConfig
    r = RiskState(RiskConfig(daily_loss_stop_usd=40, total_drawdown_stop_usd=90))
    r.realized_today, r.peak_equity, r.equity = -41, 0, -41
    assert r.check_stops("2026-10-02") and r.halt_kind == "daily"
    assert not r.maybe_resume(0, "2026-10-02")
    assert r.maybe_resume(0, "2026-10-03") and not r.halted
    r.realized_today, r.peak_equity, r.equity = 0, 100, 5
    assert r.check_stops("2026-10-03") and r.halt_kind == "hard"
    assert not r.maybe_resume(10 ** 10, "2026-10-09")


def test_nbm_header_single_digit_month():
    from wxbot.data import nbm
    txt = (" KNYC    NBM V5.0 NBP GUIDANCE    9/01/2026  0100 UTC\n"
           "        WED 02| THU 03\n"
           " UTC    00  12| 00  12\n"
           " FHR    23  35| 47  59\n"
           " TXNMN  81  61| 69  56\n"
           " TXNSD   1   2|  2   2\n"
           " TXNP1  80  58| 67  53\n"
           " TXNP2  81  60| 68  54\n"
           " TXNP5  81  61| 69  56\n"
           " TXNP7  82  62| 70  57\n"
           " TXNP9  83  63| 71  58\n\n")
    b = nbm.station_blocks(txt, {"KNYC"})
    p = nbm.parse_nbp(b["KNYC"])
    fc = nbm.forecast_from_nbp(p, "high", dt.date(2026, 9, 1))
    assert fc and fc.mean == 81 and fc.pct[90] == 83


def test_parse_ts_variable_fraction_digits():
    from wxbot.timeutil import parse_ts
    a = parse_ts("2026-10-02T08:43:49.31456Z")
    b = parse_ts("2026-10-02T08:43:49.314560+00:00")
    c = parse_ts("2026-10-02T08:43:49Z")
    assert abs(a - b) < 1e-6 and abs(a - c - 0.31456) < 1e-6
    assert parse_ts("2026-10-02T08:43:49.1Z") - c == 0.1 or abs(parse_ts("2026-10-02T08:43:49.1Z") - c - 0.1) < 1e-6
    assert parse_ts(None) == 0.0 and parse_ts("") == 0.0


def test_metar_spike_ignored_and_six_hour_glitch():
    import datetime as _dt
    from wxbot.data.obs import running_extremes
    day = _dt.date(2026, 10, 2)
    t0 = _dt.datetime(2026, 10, 2, 5, tzinfo=_dt.timezone.utc).timestamp()  # local midnight LST for -5
    # hourly obs around 20-22 C with one 35 C typo in the middle
    temps = [20, 21, 22, 35, 22, 21]
    obs = [{"obsTime": t0 + 3600 * (i + 8), "temp": c, "rawOb": "KXXX 021300Z 00000KT RMK AO2"} for i, c in enumerate(temps)]
    r = running_extremes(obs, day, -5)
    assert r["max_f"] < 75  # 22 C = 71.6 F; the 35 C (95 F) spike is ignored
    # a newest-report jump is held until confirmed
    obs2 = obs[:3] + [{"obsTime": t0 + 3600 * 11, "temp": 34, "rawOb": ""}]
    assert running_extremes(obs2, day, -5)["max_f"] < 75
    # 6-hour max group 10 F above every hourly reading in its window is ignored
    raw = "KXXX 021500Z 00000KT RMK AO2 10350 20200"
    obs3 = obs[:3] + [{"obsTime": t0 + 3600 * 11, "temp": 22, "rawOb": raw}]
    assert running_extremes(obs3, day, -5)["max_f"] < 75


def test_obs_market_conflict_blocks_quotes():
    from wxbot.models.brackets import bracket_from_market
    from wxbot.models.fair_value import LadderQuote, ladder_fair_value
    brs = [bracket_from_market(m) for m in (
        {"ticker": "E-T70", "strike_type": "less", "cap_strike": 70},
        {"ticker": "E-B70.5", "strike_type": "between", "floor_strike": 70, "cap_strike": 71},
        {"ticker": "E-T71", "strike_type": "greater", "floor_strike": 71})]
    q = {"E-T70": LadderQuote(30, 34), "E-B70.5": LadderQuote(40, 44), "E-T71": LadderQuote(22, 26)}
    fv = ladder_fair_value(brs, q, kind="high", obs_max_f=80.0)  # obs says T70 and B70.5 are dead
    assert fv.conflict and "E-T70" not in fv.dead  # market still prices them: no kill, flag conflict
    q2 = {"E-T70": LadderQuote(1, 3), "E-B70.5": LadderQuote(3, 6), "E-T71": LadderQuote(92, 95)}
    fv2 = ladder_fair_value(brs, q2, kind="high", obs_max_f=80.0)
    assert not fv2.conflict and {"E-T70", "E-B70.5"} <= fv2.dead and fv2.probs["E-T70"] == 0.0


def test_sleeve_b_rollout_controls():
    import datetime as _dt
    from wxbot.strategy import sleeve_b
    from wxbot.strategy.types import EventBook
    now = _dt.datetime(2026, 10, 2, 6, 10, tzinfo=_dt.timezone.utc).timestamp()
    ms = [{"ticker": f"H-T{k}.99", "floor_strike": k + 0.99} for k in (70, 72, 74, 88, 90, 92)]
    full_no = {"no_dollars": [["0.99", "5000"]], "yes_dollars": []}  # NO side deep, YES side empty
    full_yes = {"yes_dollars": [["0.99", "5000"]], "no_dollars": []}
    books = {m["ticker"]: (full_no if m["floor_strike"] > 85 else full_yes) for m in ms}
    ev = EventBook("H", "KXTEMPMIAH", "hourly", "MIAIDX", None, now + 3000, ms, [], {}, books)
    progs = {m["ticker"]: {"target_size_fp": "1000", "discount_factor_bps": 5000} for m in ms}
    cfg = sleeve_b.SleeveBConfig(enabled=True, max_strikes=2)
    out, _ = sleeve_b.propose(ev, 80.0, 5.0, progs, now, cfg)
    assert len(out) == 2 and {o.ticker for o in out} == {"H-T92.99", "H-T90.99"}  # farthest from 80 F
    cfg2 = sleeve_b.SleeveBConfig(enabled=True, hours_utc=(4, 5))
    assert sleeve_b.propose(ev, 80.0, 5.0, progs, now, cfg2)[0] == []  # 06Z not allowed


def test_token_bucket_adapts_to_429():
    from wxbot.kalshi.rest import TokenBucket
    b = TokenBucket(100.0, 300.0)
    b.throttled()
    assert b.rate == 50.0 and b.tokens == 0.0
    for _ in range(10):
        b.throttled()
    assert b.rate == 10.0  # floor at 10% of nominal
    for _ in range(400):
        b.ok()
    assert 99.0 < b.rate <= 100.0


def test_sleeve_t_orders_get_their_own_caps():
    """Sleeve T orders pass the thin-book caps first; Sleeve A orders only see the global caps."""
    from wxbot.risk.limits import RiskConfig, filter_orders
    from wxbot.strategy.types import DesiredOrder
    ticks = {"KXHIGHTSAN-26OCT03": ["KXHIGHTSAN-26OCT03-B70.5", "KXHIGHTSAN-26OCT03-B72.5"]}
    big = DesiredOrder("KXHIGHTSAN-26OCT03-B70.5", "bid", 70, 100.0, "T", 0.8, event="KXHIGHTSAN-26OCT03")
    kept, _ = filter_orders([big], {}, ticks, RiskConfig(max_loss_per_bracket_usd=15, max_loss_per_event_usd=25,
                                                           max_total_open_risk_usd=100), 0.0, cash_cap=100)
    assert kept and kept[0].count * 0.70 <= 15.0 + 1e-9  # shrunk to the $15 bracket cap


def test_shadow_taker_signals_liquid_only_and_fee_margin():
    from wxbot.strategy import taker
    cfg = taker.TakerConfig(enabled=True, margin=0.06, size=5)
    # liquid ladder: 2c spreads, mids sum to ~1.0
    q = {"A": LadderQuote(9, 11, 40, 30), "B": LadderQuote(59, 61, 50, 3), "C": LadderQuote(29, 31, 20, 20)}
    # pool: A is worth 25% (ask 11c: edge 0.25 - 0.11 - fee 0.01 = 0.13 > 0.06 -> buy YES, size capped at 5)
    #       B is worth 45% (NO costs 41c and wins 55%: edge 0.55 - 0.41 - 0.02 = 0.12 -> buy NO, only 50 shown)
    #       C is worth 30% (no edge either way)
    liquid, spr, msum, sig = taker.signals(q, {"A": 0.25, "B": 0.45, "C": 0.30}, cfg)
    assert liquid and abs(spr - 0.02) < 1e-9 and abs(msum - 1.0) < 1e-9
    got = {(s["ticker"], s["side"]): s for s in sig}
    assert set(got) == {("A", "yes"), ("B", "no")}
    assert got[("A", "yes")]["price"] == 0.11 and got[("A", "yes")]["size"] == 5
    assert abs(got[("B", "no")]["price"] - 0.41) < 1e-9 and got[("B", "no")]["avail"] == 50
    assert abs(got[("A", "yes")]["edge"] - (0.25 - 0.11 - taker.rule_fee(0.11))) < 1e-12
    # a 5c edge is below the 6c margin after the fee
    _, _, _, sig2 = taker.signals(q, {"A": 0.17, "B": 0.60, "C": 0.30}, cfg)
    assert not sig2
    # wide book: no signals at all
    wide = {"A": LadderQuote(5, 20), "B": LadderQuote(50, 70), "C": LadderQuote(20, 35)}
    liquid, spr, _, sig3 = taker.signals(wide, {"A": 0.6, "B": 0.1, "C": 0.3}, cfg)
    assert not liquid and spr > 0.10 and not sig3


def test_taker_shadow_store_roundtrip(tmp_path):
    from wxbot.store.db import Store
    st = Store(str(tmp_path / "t.sqlite"))
    row = (100.0, "KXHIGHNY-26OCT03", "KXHIGHNY-26OCT03-B70.5", "yes", 0.11, 5, 30, 0.25, 0.13, 21, "abc", 0.02, 1.0)
    st.add_taker_shadow([row, row])  # duplicates ignored
    assert st.taker_shadow_keys() == {("KXHIGHNY-26OCT03", "KXHIGHNY-26OCT03-B70.5", "yes")}
    assert st.taker_shadow_open(older_than=200.0) == ["KXHIGHNY-26OCT03-B70.5"]
    st.set_taker_result("KXHIGHNY-26OCT03-B70.5", "no", 300.0)
    assert st.taker_shadow_open(older_than=1e12) == []


def test_public_page_shows_own_forecast_and_no_market_data(tmp_path):
    import datetime as dt
    import numpy as np
    from wxbot.app import public_page
    from wxbot.store.db import Store
    db = str(tmp_path / "p.sqlite")
    Store(db)
    P = np.zeros(25)
    P[12], P[13], P[11] = 0.5, 0.3, 0.2
    cache = {("KNYC", dt.date(2026, 10, 4)): {"h": -3, "P": P, "anchor_r": 71.0, "t": 0.0}}
    path = public_page.write(str(tmp_path / "pub"), cache, {"KNYC": {"city": "New York"}}, db, {"mode": "paper"})
    h = open(path).read()
    assert "New York" in h and "71°F" in h and "70-72°F" in h
    assert "yes_bid" not in h and "ask" not in h.lower().replace("taker", "")
    j = json.load(open(str(tmp_path / "pub" / "forecasts.json")))
    assert j["forecasts"][0]["most_likely"] == 71.0 and abs(sum(j["forecasts"][0]["probs"]) - 1) < 1e-9


def test_paper_late_fills_only_for_trades_before_the_cancel():
    import datetime as dt
    from wxbot.execution.paper import PaperBroker
    from wxbot.strategy.types import DesiredOrder

    def trade(ts, px, n=5, side="no"):
        iso = dt.datetime.fromtimestamp(ts, dt.timezone.utc).isoformat().replace("+00:00", "Z")
        return {"ticker": "T", "created_time": iso, "yes_price_dollars": f"{px / 100:.4f}", "count_fp": f"{n:.2f}",
                "taker_side": side}
    d = DesiredOrder(ticker="T", side="bid", price_cents=95, count=10, sleeve="H", fair_yes=0.97, event="E")
    for late, expect in ((True, 5.0), (False, 0.0)):
        pb = PaperBroker(late_fills=late)
        o = pb.place(d, {}, now=1000.0)
        pb.cancel(o.oid, now=1100.0)
        assert not pb.resting()                       # the strategy sees it gone at once
        got = sum(f.count for f in pb.on_trade(trade(1050.0, 94)))   # printed while it was still resting
        assert got == expect
        assert not pb.on_trade(trade(1150.0, 94))     # after the cancel: never
