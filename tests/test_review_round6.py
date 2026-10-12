"""Round-6 review (Oct 4, after 779e190): each finding reproduced first, then fixed.
R1  a post-only order that Kalshi filled on arrival (post-only not honored) was dropped at once, so until the account
    reads showed the position (up to the 30 s freshness limit later) nothing counted it against the caps, and the
    same market was placed again after the 10 s cooldown;
R2  the go-live order test priced its crossing post-only order from the book read at the start of the test, several
    seconds earlier: if the ask was lifted meanwhile, an honest post-only order simply rested and the test said
    "post-only is not holding, do not go live";
R3  after a partial fill of that order the unfilled rest was left resting (until its 2-minute expiry)."""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_live_loop import FakeKalshi, iso  # noqa: E402
from test_review_round2 import Acct, make_bot  # noqa: E402
from test_smoke_alerts import cross_book, smoke_env  # noqa: E402

import wxbot.app.smoke as smoke  # noqa: E402
from wxbot.execution.live import UNCERTAIN, LiveOrderManager  # noqa: E402
from wxbot.strategy.types import DesiredOrder  # noqa: E402

CROSS = "KXHIGHNY-26OCT02-T80"


@pytest.fixture
def clock(monkeypatch):
    """A hand-moved clock, kept away from ET midnight (the daily stop resets there by design)."""
    from wxbot.timeutil import to_et
    c = [time.time()]
    if to_et(c[0]).hour >= 22 or to_et(c[0]).hour < 2:
        c[0] += 4 * 3600
    monkeypatch.setattr(time, "time", lambda: c[0])
    return c


# ------------------------------------------------------------------------------------------------ R1
class FillOnArrival(FakeKalshi):
    """Post-only NOT honored: a crossing order executes in full on arrival (what the go-live order test guards)."""

    def __init__(self):
        super().__init__()
        self.feed = []

    def create_order(self, ticker, side, price_cents, count, **kw):
        yb, ya = self.quotes.get(ticker.rsplit("-", 1)[1], (None, None))
        if kw.get("post_only") and ((side == "bid" and price_cents >= ya) or (side == "ask" and price_cents <= yb)):
            oid = f"x{len(self.created) + 1}"
            self.created.append((ticker, side, price_cents, count, kw))
            self.feed.append({"fill_id": f"f{oid}", "order_id": oid, "market_ticker": ticker, "book_side": side,
                              "yes_price_dollars": f"{price_cents / 100:.4f}", "count_fp": f"{count:.2f}",
                              "created_time": iso(time.time()), "is_taker": True, "fee_cost": "0.07"})
            return {"order_id": oid, "remaining_count": "0.00", "fill_count": f"{count:.2f}"}
        return super().create_order(ticker, side, price_cents, count, **kw)

    def fills(self, min_ts=None, **kw):
        return list(self.feed)


def test_an_order_filled_on_arrival_counts_until_the_account_reads_show_it(clock):
    fake = FillOnArrival()
    m = LiveOrderManager(fake, ttl_s=300)
    close = {x["ticker"]: int(clock[0]) + 86400 for x in fake.markets}
    t = fake.markets[4]["ticker"]                                   # T75: YES bid 1c, so an ask at 1c crosses
    want = [DesiredOrder(t, "ask", 1, 10.0, "A", 0.0)]
    st = m.reconcile(want, close)
    assert st["uncertain"] == 1 and st["placed"] == 0
    [o] = m.resting_list()
    assert o.state == UNCERTAIN and o.remaining == 10.0
    assert [(c.count, c.reason) for c in m.committed(want)] == [(10.0, "committed:uncertain")]
    assert [(f.sleeve, f.count) for f in m.poll_fills()] == [("A", 10.0)]   # P&L: the bot's sleeve A
    fake.as_of = clock[0] - 20                                      # the account reads lag 20 s
    for _ in range(3):
        clock[0] += 5
        m.sync()
        st = m.reconcile(want, close)
        assert st["blocked"] == 1 and len(fake.created) == 1       # past the 10 s cooldown, still not placed again
    assert m.committed(want)
    fake.as_of = None                                               # the reads catch up (they show the position)
    for _ in range(2):
        clock[0] += 5
        m.sync()
    assert not m.resting_list()


class Lagged(Acct):
    """Account reads (positions and orders) as of 20 s ago, inside the 30 s freshness limit; post-only not
    honored for a YES bid at the ask."""
    LAG = 20.0

    def __init__(self):
        super().__init__()
        self.fills_at = []

    def user_data_timestamp(self):
        return time.time() - self.LAG

    def create_order(self, ticker, side, price_cents, count, **kw):
        yb, ya = self.quotes.get(ticker.rsplit("-", 1)[1], (None, None))
        if kw.get("post_only") and side == "bid" and price_cents >= ya:
            oid = f"x{len(self.created) + 1}"
            self.created.append((ticker, side, price_cents, count, kw))
            self.fills_at.append((time.time(), ticker, count, count * price_cents / 100))
            self.cash -= count * price_cents / 100
            return {"order_id": oid, "remaining_count": "0.00", "fill_count": f"{count:.2f}"}
        return super().create_order(ticker, side, price_cents, count, **kw)

    def positions(self, **kw):
        as_of, agg = self.user_data_timestamp(), {}
        for ts, t, q, c in self.fills_at:
            if ts <= as_of:
                a = agg.setdefault(t, [0.0, 0.0])
                a[0] += q
                a[1] += c
        return [{"ticker": t, "position_fp": f"{q:.2f}", "market_exposure_dollars": f"{c:.4f}"}
                for t, (q, c) in agg.items()], []

    def fills(self, min_ts=None, **kw):
        return []


def test_fills_on_arrival_never_take_a_bracket_past_its_cap(tmp_path, monkeypatch, clock):
    fake = Lagged()
    bot = make_bot(tmp_path, monkeypatch, fake)
    bot.store.kv_set("balance_semantics", {"excludes_resting": True})
    for _ in range(3):
        clock[0] += 5
        bot.refresh_live_state()
    t = fake.markets[2]["ticker"]                                   # B72.5: YES 45/48, a 48c bid crosses
    want = [DesiredOrder(t, "bid", 48, 10.0, "A", 0.60, event=t.rsplit("-", 1)[0])]
    bot.execute(want)
    for _ in range(20):                                             # 40 s of 2-second passes
        clock[0] += 2
        bot.refresh_live_state()
        bot.execute(want)
    q = sum(x[2] for x in fake.fills_at)
    assert fake.created and q * 0.48 <= 8.0 + 1e-9                  # the $8 bracket cap holds (was $11.52)
    # and it is loud: an ERROR in the log, an alert, and a status field the health check flags
    assert bot.live.filled_on_arrival == len(fake.created)
    assert any("filled on arrival" in m for _, lvl, m in bot.store.c.execute(
        "SELECT ts, level, msg FROM events_log WHERE level='ERROR'"))
    bot.write_status()
    import json
    assert json.load(open(bot.status_path))["post_only_filled_on_arrival"] == len(fake.created)


# ------------------------------------------------------------------------------------------------ R2, R3
def moving_ask(fake, asks):
    """The crossing market's YES ask on successive book reads that include it (the last value repeats)."""
    real = fake.orderbooks
    n = {"k": 0}

    def books(tickers):
        out = real(tickers)
        if CROSS in out:
            a = asks[min(n["k"], len(asks) - 1)]
            n["k"] += 1
            out[CROSS] = {"no_dollars": [[f"{(100 - a) / 100:.4f}", "50.00"]], "yes_dollars": [["0.0200", "50.00"]]}
        return out
    fake.orderbooks = books


def run(tmp_path):
    return smoke.run_smoke({"env": "prod", "db_path": str(tmp_path / "live.sqlite")})


def test_an_ask_lifted_before_the_order_is_priced_is_not_a_failure(monkeypatch, tmp_path, capsys):
    fake = smoke_env(monkeypatch)
    cross_book(fake)
    moving_ask(fake, [5, 12])               # 5c at the start of the test, 12c by the time the order is priced
    assert run(tmp_path) is True
    out = capsys.readouterr().out
    assert "not measured" in out and not any(x[0] == "create" and x[1] == CROSS for x in fake.log)


def test_an_ask_lifted_while_the_order_travels_is_retried_then_not_measured(monkeypatch, tmp_path, capsys):
    fake = smoke_env(monkeypatch)
    cross_book(fake)
    moving_ask(fake, [5, 5, 12])            # priced at 5c, gone to 12c when it arrives: it rests without crossing
    assert run(tmp_path) is True
    out = capsys.readouterr().out
    assert "moved from 5c to 12c" in out and "not measured" in out
    assert not [o for o in fake.resting.values() if o["ticker"] == CROSS]          # cancelled


def test_a_retry_that_crosses_settles_it(monkeypatch, tmp_path, capsys):
    fake = smoke_env(monkeypatch)
    cross_book(fake)
    moving_ask(fake, [5, 5, 12, 6])         # the first try rests (the ask moved), the second meets a 6c ask
    real_create = fake.create_order
    sent = []

    def create(ticker, side, price_cents, count, **kw):
        if ticker == CROSS:
            sent.append(price_cents)
            if len(sent) == 2:              # this one crosses: cancelled on arrival
                return {"order_id": "ox", "remaining_count": "0.00", "fill_count": "0.00"}
        return real_create(ticker, side, price_cents, count, **kw)
    fake.create_order = create
    assert run(tmp_path) is True and sent == [5, 6]
    assert "[OK ] post-only order at the ask is refused" in capsys.readouterr().out


def test_an_order_resting_while_the_ask_still_crosses_fails(monkeypatch, tmp_path, capsys):
    fake = smoke_env(monkeypatch)
    cross_book(fake)                        # the ask stays at 5c and the fake rests every order
    assert run(tmp_path) is False
    assert "rests while the ask is still 5c" in capsys.readouterr().out
    assert not [o for o in fake.resting.values() if o["ticker"] == CROSS]


def test_a_partial_fill_fails_and_the_rest_is_cancelled(monkeypatch, tmp_path, capsys):
    fake = smoke_env(monkeypatch)
    cross_book(fake)
    real_create = fake.create_order

    def create(ticker, side, price_cents, count, **kw):
        r = real_create(ticker, side, price_cents, count, **kw)
        if ticker == CROSS:
            fake.resting[r["order_id"]]["remaining_count_fp"] = "0.50"
            return dict(r, remaining_count="0.50", fill_count="0.50")
        return r
    fake.create_order = create
    assert run(tmp_path) is False
    assert "FILLED" in capsys.readouterr().out
    assert not [o for o in fake.resting.values() if o["ticker"] == CROSS]
