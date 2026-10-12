"""Shadow logs stop assuming fills (outside review, Oct 8): every shadow row records its intended size and the depth
seen, books min(intended, depth), books nothing when the depth is zero or unknown, and the reports sum the booked
size. Old rows stay readable and the real taker path keeps sending what it sent before."""
import datetime as dt
import importlib.util
import os
import sqlite3
import time
from collections import defaultdict

import pandas as pd

from wxbot.execution.paper import PaperBroker
from wxbot.execution.taker import booked_size, depth_at_or_better
from wxbot.models.fair_value import LadderQuote
from wxbot.store.db import SCHEMA, Store
from wxbot.strategy import taker
from wxbot.strategy.types import DesiredOrder, EventBook

ROOT = os.path.join(os.path.dirname(__file__), "..")
BOOK = {"yes_dollars": [["0.30", "10"], ["0.29", "5"]], "no_dollars": [["0.65", "7"], ["0.60", "20"]]}


def test_booked_size_is_the_intended_size_capped_by_depth_and_nothing_without_depth():
    assert booked_size(5, None) == 0.0                   # unknown
    assert booked_size(5, 0.0) == 0.0                    # zero
    assert booked_size(5, float("nan")) == 0.0
    assert booked_size(5, 3.0) == 3.0                    # partial
    assert booked_size(5, 40.0) == 5.0


def test_depth_counts_the_offers_at_the_price_or_better():
    assert depth_at_or_better(BOOK, "yes", 0.35) == 7.0              # YES offers: 35c x7, 40c x20
    assert depth_at_or_better(BOOK, "yes", 0.40) == 27.0
    assert depth_at_or_better(BOOK, "yes", 0.34) == 0.0
    assert depth_at_or_better(BOOK, "no", 0.71) == 15.0             # NO offers: 70c x10, 71c x5
    assert depth_at_or_better(None, "yes", 0.35) is None
    assert depth_at_or_better({}, "yes", 0.35) == 0.0


def ladder_quotes():
    """Three liquid brackets, each a YES signal: A shows 2 at the ask, B's ask level shows 0, C has no sizes."""
    quotes = {"A": LadderQuote(30, 32, 50.0, 2.0), "B": LadderQuote(40, 42, 50.0, 0.0), "C": LadderQuote(20, 22)}
    books = {"A": {"yes_dollars": [["0.30", "50"]], "no_dollars": [["0.68", "2"]]},
             "B": {"yes_dollars": [["0.40", "50"]], "no_dollars": [["0.58", "0"]]}}
    return quotes, books, {"A": 0.5, "B": 0.6, "C": 0.4}


def test_taker_signals_book_partial_zero_and_unknown_depth():
    quotes, books, probs = ladder_quotes()
    cfg = taker.TakerConfig(enabled=True)
    liquid, _, _, sig = taker.signals(quotes, probs, cfg, books=books)
    got = {x["ticker"]: (x["size"], x["intended"], x["depth"], x["booked"]) for x in sig if x["side"] == "yes"}
    assert liquid and got == {"A": (2.0, 5.0, 2.0, 2.0),          # partial: book what was offered
                              "B": (5.0, 5.0, 0.0, 0.0),          # zero: book nothing (size keeps its old 5)
                              "C": (5.0, 5.0, None, 0.0)}         # unknown: book nothing, still a signal
    # without raw books the displayed size stands in for the depth; none displayed is unknown
    _, _, _, sig2 = taker.signals(quotes, probs, cfg)
    assert {x["ticker"]: (x["depth"], x["booked"]) for x in sig2} == {"A": (2.0, 2.0), "B": (None, 0.0),
                                                                     "C": (None, 0.0)}


def test_runner_x_shadow_books_depth_but_the_real_take_keeps_the_rule_size(tmp_path):
    from wxbot.app.runner import Bot
    quotes, books, probs = ladder_quotes()
    bot = Bot.__new__(Bot)
    bot.store = Store(str(tmp_path / "x.sqlite"))
    bot.cfg_x, bot.x_seen, bot.ml_sha = taker.TakerConfig(enabled=True), set(), "sha"
    bot.cfg_x_raw = {"trade": True}
    sent = []
    bot.execute_take = lambda sl, ev, tk, side, price, size, c: sent.append((sl, tk, side, price, size))
    ev = EventBook("KXHIGHNY-26OCT10", "KXHIGHNY", "high", "KNYC", dt.date(2026, 10, 10), 0.0, [], [],
                   quotes, books)
    now = dt.datetime(2026, 10, 9, 21, 5, tzinfo=dt.timezone.utc).timestamp()     # h = -3, within 30 minutes
    bot.shadow_taker(ev, probs, -3, now)
    rows = {r[0]: r[1:] for r in bot.store.c.execute(
        "SELECT ticker, size, avail, intended, depth, booked FROM taker_shadow WHERE side = 'yes'")}
    assert rows == {"A": (2.0, 2.0, 5.0, 2.0, 2.0), "B": (5.0, 0.0, 5.0, 0.0, 0.0), "C": (5.0, 0.0, 5.0, None, 0.0)}
    # the trading decision is untouched: Sleeve X sends the rule's size (`size`), not the booked size
    assert sorted((tk, size) for sl, tk, side, price, size in sent if side == "yes") == [("A", 2.0), ("B", 5.0),
                                                                                       ("C", 5.0)]


class RainClient:
    """Three rain markets with signals: ATL buys NO (3 shown at the YES bid), MIA buys YES (0 shown at the ask), NYC
    buys YES with no sizes in the listing."""

    def markets(self, **kw):
        return [{"ticker": "KXRAIN-26OCT03-ATL", "status": "active", "yes_bid_dollars": "0.3000",
                 "yes_ask_dollars": "0.3200", "yes_bid_size_fp": "3.00", "yes_ask_size_fp": "9.00"},
                {"ticker": "KXRAIN-26OCT03-MIA", "status": "active", "yes_bid_dollars": "0.6000",
                 "yes_ask_dollars": "0.6200", "yes_bid_size_fp": "10.00", "yes_ask_size_fp": "0.00"},
                {"ticker": "KXRAIN-26OCT03-NYC", "status": "active", "yes_bid_dollars": "0.4000",
                 "yes_ask_dollars": "0.4200"}]

    def market(self, t):
        return {"result": "no" if t.endswith("ATL") else "yes"}


def test_rain_shadow_books_the_displayed_depth_and_reports_sum_it(tmp_path):
    from wxbot.app.runner import Bot
    from wxbot.ml import rain as R

    class Model:
        meta = {"sha256": "abc123def456", "cities": R.RAIN_CITY}

        def predict(self, raw, icao, D, h):
            return {"KATL": 0.05, "KMIA": 0.95, "KNYC": 0.95}.get(icao)

        def pool(self, mid, p):
            return p

    bot = Bot.__new__(Bot)
    bot.store = Store(str(tmp_path / "r.sqlite"))
    bot.rain, bot.rain_seen = Model(), set()
    rows = bot.rain_signals(RainClient(), "raw", pd.Timestamp("2026-10-03").date(), 12, 0.06, 5.0)
    got = {(r[2].rsplit("-", 1)[1], r[6]): (r[8], r[15], r[16], r[17]) for r in rows}
    assert got == {("ATL", "no"): (5.0, 5.0, 3.0, 3.0),        # partial: the 3 shown at the YES bid
                   ("MIA", "yes"): (5.0, 5.0, 0.0, 0.0),       # zero at the ask
                   ("NYC", "yes"): (5.0, 5.0, None, 0.0)}      # unknown: still logged, books nothing
    bot.store.add_rain_shadow(rows)
    bot.store.c.execute("UPDATE rain_shadow SET ts = ts - 9 * 3600")
    bot.rest = RainClient()
    bot.settle_rain_shadow()
    pnl, cnt, n = bot.store.rain_pnl()
    assert n == 3 and cnt == 3.0 and abs(pnl - 3 * (1.0 - 0.70 - 0.02)) < 1e-9     # only ATL's 3 contracts count


def test_hourly_shadow_records_the_print_and_books_what_reached_the_order(tmp_path):
    from wxbot.app.runner import Bot
    bot = Bot.__new__(Bot)
    bot.store = Store(str(tmp_path / "h.sqlite"))
    bot.h_held, bot.h2_held, bot.counters = defaultdict(list), defaultdict(list), defaultdict(int)
    bot.paper_h = bot.paper_h2 = None
    pb = PaperBroker()
    t = "KXTEMPMIAH-26OCT0312-T83.99"
    book = {"yes_dollars": [["0.97", "5"]], "no_dollars": [["0.01", "40"]]}       # 5 already bid at 97c
    pb.place(DesiredOrder(t, "bid", 97, 10, "H", 0.99, event="KXTEMPMIAH-26OCT0312"), book, 1000.0)

    def trade(ts, price, n):
        return {"ticker": t, "created_time": dt.datetime.fromtimestamp(ts, dt.timezone.utc).isoformat(),
                "yes_price_dollars": f"{price / 100:.4f}", "count_fp": f"{n:.2f}", "taker_side": "no"}
    fills = pb.on_trade(trade(1010.0, 97, 8)) + pb.on_trade(trade(1020.0, 95, 25))
    for f in fills:
        bot.record_h_fill(f)
    got = bot.store.c.execute("SELECT count, intended, depth, booked FROM hourly_shadow ORDER BY ts").fetchall()
    # at our price the queue ahead (5) eats first: 3 reach us; then a print through our price of 25 fills the other 7
    assert got == [(3.0, 10.0, 3.0, 3.0), (7.0, 7.0, 25.0, 7.0)]
    bot.store.add_hourly_fill((1030.0, "KXTEMPMIAH-26OCT0312", t, "bid", 97, 2.0))     # an old-style row: count
    bot.store.c.execute("UPDATE hourly_shadow SET result = 'yes'")
    pnl, n = bot.store.hourly_pnl(variant="H")
    assert n == 12.0 and abs(pnl - 12 * 0.03) < 1e-9


def old_database(path):
    """A database as the bot wrote it before Oct 9 (the schema without the sizing columns, which only Store adds),
    with one settled row per shadow log."""
    c = sqlite3.connect(path)
    c.executescript(SCHEMA)
    ts = time.time() - 86400
    c.executemany("INSERT INTO taker_shadow VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
        (ts, "E1", "E1-A", "yes", 0.30, 5.0, 0.0, 0.6, 0.2, 21, "s", 0.02, 1.0, "yes", ts),    # nothing displayed
        (ts, "E1", "E1-B", "yes", 0.40, 3.0, 3.0, 0.6, 0.1, 21, "s", 0.02, 1.0, "no", ts)])    # 3 displayed
    c.execute("INSERT INTO hourly_shadow VALUES (?,?,?,?,?,?,?,?,?)", (ts, "H1", "H1-T", "bid", 97, 10.0, "yes", ts,
                                                                       "H"))
    c.execute("INSERT INTO rain_shadow VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
              (ts, "R1", "R1-NYC", "NYC", "2026-10-03", 12, "yes", 0.42, 5.0, 0.02, 0.9, 0.9, 0.41, 0.4, "s", "yes",
               ts))
    c.commit()
    c.close()


def load_paper_report():
    spec = importlib.util.spec_from_file_location("paper_report_mod", os.path.join(ROOT, "scripts", "paper_report.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_reports_read_databases_from_before_the_columns(tmp_path):
    """The report script and the dashboard read an old database as it is (the bot never opened it with this code):
    taker rows book their displayed size, hourly rows their count, rain rows nothing (no depth was recorded)."""
    from wxbot.app.dashboard import collect
    db = str(tmp_path / "old.sqlite")
    old_database(db)
    pr = load_paper_report()
    out = []
    c = sqlite3.connect(db)
    pr.shadow_taker_section(c, out.append)
    pr.hourly_shadow_section(c, out.append)
    pr.rain_shadow_section(c, out.append)
    txt = "\n".join(out)
    assert "2 signals, 3 contracts booked" in txt and "1 signals booked none" in txt
    assert "at booked sizes $-1.2" in txt                     # B: 3 x (0 - 0.40) minus the fee; A booked nothing
    assert "1 simulated fills (10 contracts)" in txt
    assert "1 signals, 0 contracts booked" in txt and "depth zero or not recorded" in txt
    d = collect(db)
    assert sorted(r[-1] for r in d["taker"]) == [0, 3.0] and d["hourly"]["H"]["contracts"] == 10.0
    assert [r[-1] for r in d["rain"]["R"]] == [0]


def test_the_dashboard_shows_the_booked_size_and_averages_only_what_was_booked(tmp_path):
    from wxbot.app.dashboard import build_html
    db = str(tmp_path / "d.sqlite")
    st = Store(db)
    ts = time.time() - 3600
    st.add_taker_shadow([(ts, "E1", "E1-A", "yes", 0.30, 5.0, 0.0, 0.6, 0.2, 21, "s", 0.02, 1.0, 5.0, None, 0.0),
                         (ts, "E1", "E1-B", "yes", 0.40, 3.0, 3.0, 0.6, 0.1, 21, "s", 0.02, 1.0, 5.0, 3.0, 3.0)])
    st.set_taker_result("E1-A", "yes", ts)                    # won, but booked nothing: no trade
    st.set_taker_result("E1-B", "no", ts)
    st.add_rain_shadow([(ts, "R1", "R1-NYC", "NYC", "2026-10-03", 12, "yes", 0.42, 5.0, 0.02, 0.9, 0.9, 0.41, 0.4, "s",
                         5.0, 2.0, 2.0)])
    st.set_rain_result("R1-NYC", "yes", ts)
    page = build_html({"db_path": db, "mode": "paper"}, db, str(tmp_path / "none.json"))
    assert "<th>Booked</th>" in page and "1 booked nothing (depth zero or unknown)" in page
    assert "-42.00c per booked contract after fees" in page   # B alone: 0 - 40c - 2c
    assert "+56.00c per booked contract after fees" in page   # rain: 100c - 42c - 2c


def test_the_store_adds_the_columns_and_keeps_old_rows_as_they_were(tmp_path):
    db = str(tmp_path / "old.sqlite")
    old_database(db)
    st = Store(db)
    for table in ("taker_shadow", "hourly_shadow", "rain_shadow", "rain_shadow_v2"):
        cols = [r[1] for r in st.c.execute(f"PRAGMA table_info({table})")]
        assert cols[-3:] == ["intended", "depth", "booked"], (table, cols)
    assert st.c.execute("SELECT size, avail, booked FROM taker_shadow ORDER BY ticker").fetchall() == [
        (5.0, 0.0, None), (3.0, 3.0, None)]
    pnl, cnt, n = st.rain_pnl()
    assert (n, cnt, pnl) == (1, 0.0, 0.0)                     # an old rain row: depth unknown, nothing booked
    hp, hn = st.hourly_pnl(variant="H")
    assert hn == 10.0 and abs(hp - 0.30) < 1e-9               # an old hourly row: its count, as before
    Store(db)                                                 # opening it again changes nothing
    assert len([r for r in st.c.execute("PRAGMA table_info(taker_shadow)")]) == 18
