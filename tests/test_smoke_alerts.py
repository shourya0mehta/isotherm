import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import wxbot.app.smoke as smoke
from wxbot.app.alerts import Alerter


def iso(ts):
    import datetime as dt
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).isoformat().replace("+00:00", "Z")


class FakeREST:
    """Kalshi as the smoke test sees it: the available balance leaves out cash held by resting orders (a YES bid
    holds its price, a YES ask holds 1 - price: the NO cost), orders read back with book_side and YES price."""

    def __init__(self, base_url=None, auth=None, clock=None):
        self.resting = {}
        self.log = []
        self.n = 0
        self.clock = clock

    def balance(self):
        held = sum(float(o["remaining_count_fp"]) * (o["px"] if o["book_side"] == "bid" else 100 - o["px"])
                   for o in self.resting.values())
        return {"balance": int(62500 - held)}

    def events(self, series, status="open", nested=True, limit=3):
        return [{"markets": [{"ticker": "KXHIGHNY-26OCT02-T60", "status": "active"},
                             {"ticker": "KXHIGHNY-26OCT02-B70.5", "status": "active"}]}]

    def orderbooks(self, tickers):
        # T60: best NO bid 99c -> YES ask 1c (a 1c bid would cross), YES bid 98c (too close for the 99c ask test);
        # B70.5: NO bid 60c -> YES ask 40c, YES bid 35c
        return {"KXHIGHNY-26OCT02-T60": {"no_dollars": [["0.9900", "500.00"]], "yes_dollars": [["0.9800", "5.00"]]},
                "KXHIGHNY-26OCT02-B70.5": {"no_dollars": [["0.6000", "10.00"]], "yes_dollars": [["0.3500", "9.00"]]}}

    def create_order(self, ticker, side, price_cents, count, **kw):
        self.n += 1
        oid = f"o{self.n}"
        self.log.append(("create", ticker, side, price_cents, count, kw.get("post_only")))
        self.resting[oid] = {"order_id": oid, "ticker": ticker, "remaining_count_fp": f"{count:.2f}", "px": price_cents,
                             "book_side": side, "outcome_side": "yes" if side == "bid" else "no",
                             "yes_price_dollars": f"{price_cents / 100:.4f}",
                             "client_order_id": kw.get("client_order_id"),
                             "expiration_time": iso(kw["expiration_ts"]) if kw.get("expiration_ts") else None}
        return {"order_id": oid, "remaining_count": f"{count:.2f}", "fill_count": "0.00"}

    def orders(self, status="resting", **params):
        return [o for o in self.resting.values() if o["ticker"] == params.get("ticker")]

    def amend_order(self, oid, ticker, side, price_cents, total, expiration_ts=None):
        self.log.append(("amend", oid, total))
        self.resting[oid]["remaining_count_fp"] = f"{total:.2f}"
        if expiration_ts:
            self.resting[oid]["expiration_time"] = iso(expiration_ts)
        return {"order_id": oid, "remaining_count": f"{total:.2f}"}

    def cancel_order(self, oid):
        self.log.append(("cancel", oid))
        self.resting.pop(oid, None)
        return {"order_id": oid, "reduced_by": "1.00"}

    def batch_cancel(self, ids):
        self.log.append(("batch_cancel", tuple(ids)))
        out = []
        for i in ids:
            if self.resting.pop(i, None) is not None:
                out.append({"order_id": i, "reduced_by": "1.00", "ts_ms": 1})
            else:
                out.append({"order_id": i, "reduced_by": "0.00", "error": {"code": "not_found"}})
        return {"orders": out}

    def user_data_timestamp(self):
        return self.clock[0] - 0.8


def smoke_env(monkeypatch, tmp_path=None):
    import time
    k = Ed25519PrivateKey.generate()
    pem = k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                          serialization.NoEncryption()).decode()
    monkeypatch.setenv("KALSHI_KEY_ID", "kid")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY", pem)
    monkeypatch.delenv("KALSHI_PRIVATE_KEY_PATH", raising=False)
    clock = [time.time()]
    monkeypatch.setattr(smoke.time, "time", lambda: clock[0])
    monkeypatch.setattr(smoke.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    fake = FakeREST(clock=clock)
    monkeypatch.setattr(smoke, "KalshiREST", lambda base_url=None, auth=None: fake)
    return fake


def test_smoke_flow(monkeypatch, tmp_path):
    fake = smoke_env(monkeypatch)
    db = str(tmp_path / "live.sqlite")
    assert smoke.run_smoke({"env": "prod", "db_path": db}) is True
    kinds = [x[0] for x in fake.log]
    assert kinds == ["create", "amend", "amend", "batch_cancel", "batch_cancel", "create", "cancel"]   # one cancel
    assert fake.log[0][1] == "KXHIGHNY-26OCT02-B70.5"  # skipped the market where 1c would cross
    assert fake.log[0][2:] == ("bid", 1, 2, True)
    assert [x[2] for x in fake.log[1:3]] == [1, 2]      # down to 1 (as the bot shrinks), back up to 2 (as it grows)
    assert fake.log[5][1:] == ("KXHIGHNY-26OCT02-B70.5", "ask", 99, 2, True)   # the ask side: a NO bid at 1c, on a
    # market whose best YES bid (35c) is far below 99c (T60, YES bid 98c, is skipped)
    from wxbot.store.db import Store
    obs = Store(db).kv_get("api_observations")
    assert obs["ask_side"]["held_cents"] == 2 and obs["ask_side"]["back_cents"] == 2
    assert obs["ask_side"]["book_side"] == "ask"
    assert obs["user_data_ts_moves_idle"] is True
    assert Store(db).kv_get("balance_semantics")["excludes_resting"] is True


def test_smoke_flags_an_ask_that_holds_the_yes_price(monkeypatch, tmp_path):
    """If Kalshi read an ask's price the other way (holding 99c a contract), the smoke test must fail."""
    fake = smoke_env(monkeypatch)
    real_balance = fake.balance

    def wrong_hold():
        b = real_balance()
        asks = sum(float(o["remaining_count_fp"]) for o in fake.resting.values() if o["book_side"] == "ask")
        return {"balance": b["balance"] - int(asks * 98)}      # 99c held instead of 1c
    fake.balance = wrong_hold
    assert smoke.run_smoke({"env": "prod", "db_path": str(tmp_path / "live.sqlite")}) is False


def test_alerter_rate_limit_and_disabled(monkeypatch):
    sent = []
    monkeypatch.setattr("wxbot.app.alerts.requests.post", lambda url, data=None, headers=None, timeout=None:
                        sent.append((url, data, headers)))
    off = Alerter({})
    assert off.send("x", "t", "m") is False
    a = Alerter({"ntfy_topic": "wxbot-test-abc", "min_gap_s": 600})
    assert a.send("halt", "HALT", "daily loss stop") is True
    assert a.send("halt", "HALT", "again") is False  # rate limited per key
    assert a.send("daily", "daily", "summary") is True
    assert sent[0][0] == "https://ntfy.sh/wxbot-test-abc" and sent[0][2]["Title"] == "HALT"


def test_alert_topic_can_come_from_the_environment(monkeypatch):
    from wxbot.app.alerts import Alerter
    monkeypatch.setenv("WXBOT_NTFY_TOPIC", "wxbot-from-env")
    assert Alerter({}).topic == "wxbot-from-env" and Alerter({"ntfy_topic": "x"}).topic == "wxbot-from-env"
    monkeypatch.delenv("WXBOT_NTFY_TOPIC")
    assert not Alerter({}).enabled


def test_an_unrelated_debit_during_the_ask_test_does_not_block_go_live(monkeypatch, tmp_path):
    """A 5c debit lands while the ask rests: the hold reads 7c but the cancel gives back 2c, so the verdict is
    'not measured cleanly', not a failure (the side and price read back fine)."""
    fake = smoke_env(monkeypatch)
    real_balance = fake.balance
    state = {"debit": 0}
    real_create = fake.create_order

    def create(ticker, side, price_cents, count, **kw):
        if side == "ask":
            state["debit"] = 5
        return real_create(ticker, side, price_cents, count, **kw)
    fake.create_order = create
    fake.balance = lambda: {"balance": real_balance()["balance"] - state["debit"]}
    assert smoke.run_smoke({"env": "prod", "db_path": str(tmp_path / "live.sqlite")}) is True


def test_a_slow_orders_read_does_not_block_go_live(monkeypatch, tmp_path):
    fake = smoke_env(monkeypatch)
    real_orders = fake.orders
    calls = {"n": 0}

    def slow(status="resting", **params):
        calls["n"] += 1
        out = real_orders(status, **params)
        return [o for o in out if o["book_side"] != "ask"] if calls["n"] <= 3 else out
    fake.orders = slow
    assert smoke.run_smoke({"env": "prod", "db_path": str(tmp_path / "live.sqlite")}) is True


def cross_book(fake):
    """A third market whose YES ask is 5c, so the smoke test tries its post-only crossing order there."""
    real_events, real_books = fake.events, fake.orderbooks
    fake.events = lambda *a, **kw: [{"markets": real_events("KXHIGHNY")[0]["markets"] + [
        {"ticker": "KXHIGHNY-26OCT02-T80", "status": "active"}]}]
    fake.orderbooks = lambda tickers: {**real_books(tickers), "KXHIGHNY-26OCT02-T80": {
        "no_dollars": [["0.9500", "50.00"]], "yes_dollars": [["0.0200", "50.00"]]}}


def test_a_post_only_cross_cancelled_on_arrival_passes(monkeypatch, tmp_path):
    fake = smoke_env(monkeypatch)
    cross_book(fake)
    real_create = fake.create_order

    def create(ticker, side, price_cents, count, **kw):
        if ticker.endswith("T80"):
            return {"order_id": "ox", "remaining_count": "0.00", "fill_count": "0.00"}
        return real_create(ticker, side, price_cents, count, **kw)
    fake.create_order = create
    assert smoke.run_smoke({"env": "prod", "db_path": str(tmp_path / "live.sqlite")}) is True


def test_a_post_only_cross_that_rests_fails(monkeypatch, tmp_path):
    fake = smoke_env(monkeypatch)
    cross_book(fake)
    assert smoke.run_smoke({"env": "prod", "db_path": str(tmp_path / "live.sqlite")}) is False


def test_slow_account_reads_do_not_fail_the_order_test(monkeypatch, tmp_path):
    """The orders read lags the write: the new order shows up only on the third read, and the cancelled one keeps
    showing for two more reads. Neither is a failure."""
    fake = smoke_env(monkeypatch)
    real_orders, real_cancel, real_batch = fake.orders, fake.cancel_order, fake.batch_cancel
    state = {"reads": 0, "ghost": None, "ghost_reads": 0}

    def orders(status="resting", **params):
        state["reads"] += 1
        out = real_orders(status, **params)
        if state["reads"] <= 2:
            out = [o for o in out if o["order_id"] != "o1"]
        if state["ghost"] is not None and state["ghost_reads"] < 2:
            state["ghost_reads"] += 1
            out = out + [state["ghost"]]
        return out

    def batch(ids):
        if "o1" in fake.resting and state["ghost"] is None:
            state["ghost"] = dict(fake.resting["o1"])
        return real_batch(ids)
    fake.orders, fake.batch_cancel = orders, batch
    assert smoke.run_smoke({"env": "prod", "db_path": str(tmp_path / "live.sqlite")}) is True
