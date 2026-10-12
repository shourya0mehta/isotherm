"""Chaos test of the live order manager (Oct 4 review): random lost answers, failed and partial cancels, 5xx errors and
lagged account snapshots over hundreds of cycles. Invariants after every cycle:
  1. every bot order resting on the exchange is known to the manager (by order id, or by client id while uncertain),
     so its exposure is counted and its market is blocked;
  2. never more than two bot orders rest on one market and side (one being replaced or topped up, at most);
  3. once the exchange behaves, everything converges: after a clean sync and reconcile the exchange holds exactly the
     desired orders."""
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_live_loop import FakeKalshi  # noqa: E402

from wxbot.execution.live import LiveOrderManager  # noqa: E402
from wxbot.kalshi.rest import KalshiError  # noqa: E402
from wxbot.strategy.types import DesiredOrder  # noqa: E402


class ChaosKalshi(FakeKalshi):
    def __init__(self, seed):
        super().__init__()
        self.rng = random.Random(seed)
        self.chaos = True
        self.prev_snap = None   # (as_of, orders)

    def create_order(self, ticker, side, price_cents, count, **kw):
        r = self.rng.random() if self.chaos else 1.0
        if r < 0.05:
            raise ConnectionError("connect timeout")                       # never reached the exchange
        out = super().create_order(ticker, side, price_cents, count, **kw)
        if r < 0.15:
            raise ConnectionError("read timed out")                        # placed, answer lost
        if r < 0.20:
            raise KalshiError(502, "bad gateway", "POST", "/portfolio/events/orders", ambiguous=True)
        return out

    def batch_cancel(self, ids, **kw):
        if self.chaos and self.rng.random() < 0.05:
            raise ConnectionError("reset by peer")
        out = []
        for i in ids:
            if self.chaos and self.rng.random() < 0.15:
                out.append({"order_id": i, "reduced_by": "0.00", "error": {"code": "internal"}})
            else:
                rem = self.resting.get(i, {}).get("remaining_count_fp", "0.00")
                self.cancel_order(i)
                out.append({"order_id": i, "reduced_by": rem, "ts_ms": int(time.time() * 1000)})
        return {"orders": out}

    def user_data_timestamp(self):
        return self._as_of

    def orders(self, status="resting", **kw):
        now = time.time()
        cur = (now, [dict(o) for o in self.resting.values()])
        if self.chaos and self.prev_snap is not None and self.rng.random() < 0.3:
            snap = self.prev_snap                                          # a lagged read
        else:
            snap = cur
        self.prev_snap = cur
        self._as_of = snap[0] - 0.05
        return [dict(o) for o in snap[1]]

    def fill_some(self, p=0.05):
        for oid, o in list(self.resting.items()):
            if self.rng.random() < p:
                self.resting.pop(oid)                                       # filled in full


def known(m, fake):
    ids = {o.order_id for o in m.resting_list() if o.order_id}
    cids = {o.client_order_id for o in m.resting_list()}
    return all(oid in ids or o["client_order_id"] in cids for oid, o in fake.resting.items())


def test_order_manager_invariants_under_chaos(monkeypatch):
    clock = [time.time()]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    for seed in range(6):
        fake = ChaosKalshi(seed)
        fake._as_of = clock[0]
        m = LiveOrderManager(fake, ttl_s=1200, max_writes_per_cycle=40)
        m.fallback_lag_s = 3.0
        tick = [x["ticker"] for x in fake.markets]
        close = {t: int(clock[0]) + 86400 for t in tick}
        rng = random.Random(100 + seed)
        tally = {"placed": 0, "uncertain": 0, "cancel_unconfirmed": 0, "cancelled": 0}
        for cycle in range(250):
            clock[0] += 2.0
            want = []
            for t in tick:
                if rng.random() < 0.6:
                    q = fake.quotes[t.rsplit("-", 1)[1]]
                    px = min(98, max(q[1] + rng.choice([0, 1, 2]), 2))      # an ask (NO bid) above the YES bid
                    want.append(DesiredOrder(t, "ask", px, float(rng.choice([5, 10, 12])), "A", 0.0))
            if cycle % 3 == 0:
                try:
                    m.sync()
                except Exception:
                    pass
            st = m.reconcile(want, close)
            for k in tally:
                tally[k] += st.get(k, 0)
            fake.fill_some()
            assert known(m, fake), f"seed {seed} cycle {cycle}: an exchange order is not tracked"
            per_key = {}
            for o in fake.resting.values():
                per_key[(o["ticker"], o["book_side"])] = per_key.get((o["ticker"], o["book_side"]), 0) + 1
            assert max(per_key.values(), default=0) <= 2, f"seed {seed} cycle {cycle}: stacked orders {per_key}"
        assert tally["placed"] >= 100 and tally["uncertain"] >= 5 and tally["cancel_unconfirmed"] >= 10, tally
        # the exchange behaves again: a few clean cycles converge to exactly the desired orders
        fake.chaos = False
        final = [DesiredOrder(tick[4], "ask", 5, 10.0, "A", 0.0)]
        for _ in range(6):
            clock[0] += 5.0
            m.sync()
            m.reconcile(final, close)
        assert [(o["ticker"], o["yes_price_dollars"], o["remaining_count_fp"]) for o in fake.resting.values()] == \
            [(tick[4], "0.0500", "10.00")]
        assert [o.state for o in m.resting_list()] == ["resting"]


def test_the_live_loop_survives_chaos_without_step_failures(tmp_path, monkeypatch):
    """The whole bot (universe, books, decide, risk, reconcile, account refresh, status) for 150 two-second ticks
    against the chaotic exchange: no step may fail, the invariants hold, and at shutdown every bot order is either
    cancelled or still tracked."""
    import logging
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    import wxbot.app.runner as runner
    k = Ed25519PrivateKey.generate()
    pem = k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                          serialization.NoEncryption()).decode()
    monkeypatch.setenv("KALSHI_KEY_ID", "kid")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY", pem)
    monkeypatch.delenv("KALSHI_PRIVATE_KEY_PATH", raising=False)
    monkeypatch.delenv("WXBOT_NTFY_TOPIC", raising=False)
    cfg = {
        "mode": "live", "env": "prod", "db_path": str(tmp_path / "live.sqlite"), "status_path": str(tmp_path / "st.json"),
        "loop": {"tick_seconds": 2, "universe_every_s": 600, "books_every_s": 5, "trades_every_s": 10,
                 "metar_every_s": 300, "nbm_every_s": 900, "index_every_s": 60, "settle_every_s": 900,
                 "positions_every_s": 5},
        "sleeve_a": {"enabled": True, "series": ["KXHIGHNY"], "decided_only": True, "decided_prob": 0.97,
                     "min_edge_c": 2.0, "improve_if_edge_c": 99, "clip_contracts": 12, "decided_max_cost_usd": 10,
                     "min_price_c": 60, "max_price_c": 98},
        "sleeve_b": {"enabled": False, "shadow": False}, "tape_archive": {"enabled": False},
        "risk": {"max_loss_per_bracket_usd": 8, "max_loss_per_event_usd": 10, "max_total_open_risk_usd": 75,
                 "max_cash_deployed_usd": 200, "account_stale_seconds": 30, "max_api_errors_per_min": 1000},
        "execution": {"order_ttl_s": 1200, "ttl_by_sleeve": {"A": 300}}, "dashboard_every_s": 10 ** 9,
    }
    clock = [time.time()]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    bot = runner.Bot(cfg, base_dir=str(tmp_path))
    fake = ChaosKalshi(7)
    fake._as_of = clock[0]
    bot.rest = fake
    bot.live.rest = fake
    bot.nbm.refresh = lambda *a, **kw: None
    monkeypatch.setattr(runner.obs_mod, "fetch_metars", lambda *a, **kw: {})
    ticks = {"n": 0}
    bad = []

    def fake_sleep(s):
        clock[0] += max(s, 0.5)
        ticks["n"] += 1
        if ticks["n"] % 4 == 0:   # the tails' books move, so the bot re-prices; some resting orders fill
            for t in ("T70", "T75"):
                fake.quotes[t] = (1, fake.rng.choice([2, 3, 4]))
            fake.fill_some()
        if not known(bot.live, fake):
            bad.append(("untracked", ticks["n"]))
        per_key = {}
        for o in fake.resting.values():
            per_key[(o["ticker"], o["book_side"])] = per_key.get((o["ticker"], o["book_side"]), 0) + 1
        if max(per_key.values(), default=0) > 2:
            bad.append(("stacked", ticks["n"], per_key))
        if ticks["n"] >= 150:
            bot.running = False
    monkeypatch.setattr(runner.time, "sleep", fake_sleep)
    fails = []

    class H(logging.Handler):
        def emit(self, rec):
            if "step failed" in rec.getMessage():
                fails.append(rec.getMessage())
    h = H()
    logging.getLogger().addHandler(h)
    try:
        bot.running = True
        bot.run()
    finally:
        logging.getLogger().removeHandler(h)
    assert not fails, fails[:3]
    assert bot.counters["step_errors"] == 0
    assert not bad, bad[:5]
    assert len(fake.created) >= 5
    assert known(bot.live, fake)          # whatever could not be cancelled at shutdown is still tracked


class RealisticChaos(ChaosKalshi):
    """Round-2 verification: also partial fills, and the answers the exchange gives for an order that is already
    gone (an amend answered 404, a cancel answered with a per-order not_found error). It also runs the Stage 2
    gate's duplicate check at the moment each new order reaches it (Oct 6 demo test): every bot order already resting
    on that market and side must be at least half filled, judged on its size when it was placed."""

    def __init__(self, seed):
        super().__init__(seed)
        self.dups = []

    def create_order(self, ticker, side, price_cents, count, **kw):
        before = [dict(o) for o in self.resting.values() if (o["ticker"], o["book_side"]) == (ticker, side)]
        n = len(self.created)
        try:
            return super().create_order(ticker, side, price_cents, count, **kw)
        finally:
            if len(self.created) > n:                                       # it reached the exchange
                self.resting[f"o{len(self.created)}"]["initial_count_fp"] = f"{count:.2f}"
                for o in before:
                    size = float(o.get("initial_count_fp") or 0)
                    if not (size and float(o["fill_count_fp"]) >= 0.5 * size):
                        self.dups.append((ticker, side, o["order_id"], o["fill_count_fp"], size, count))

    def amend_order(self, oid, *a, **kw):
        if oid not in self.resting:
            raise KalshiError(404, '{"error":{"code":"not_found"}}', "POST", f"/portfolio/events/orders/{oid}/amend")
        if self.chaos and self.rng.random() < 0.05:
            raise ConnectionError("read timed out")
        return super().amend_order(oid, *a, **kw)

    def batch_cancel(self, ids, **kw):
        if self.chaos and self.rng.random() < 0.05:
            raise ConnectionError("reset by peer")
        out = []
        for i in ids:
            if i not in self.resting:
                out.append({"order_id": i, "reduced_by": "0.00", "error": {"code": "not_found"}})
            elif self.chaos and self.rng.random() < 0.15:
                out.append({"order_id": i, "reduced_by": "0.00", "error": {"code": "internal"}})
            else:
                rem = self.resting[i]["remaining_count_fp"]
                self.cancel_order(i)
                out.append({"order_id": i, "reduced_by": rem, "ts_ms": int(time.time() * 1000)})
        return {"orders": out}

    def fill_some(self, p=0.05):
        for oid, o in list(self.resting.items()):
            r = self.rng.random()
            if r < p:
                self.resting.pop(oid)                                       # filled in full
            elif r < 2 * p:                                                 # partly filled
                rem, f = float(o["remaining_count_fp"]), float(o["fill_count_fp"])
                k = max(1.0, rem // 2)
                o["remaining_count_fp"], o["fill_count_fp"] = f"{rem - k:.2f}", f"{f + k:.2f}"
                if rem - k <= 0:
                    self.resting.pop(oid)


def test_order_manager_never_undercounts_under_realistic_chaos(monkeypatch):
    """Invariants: every exchange order is tracked; on every market the exchange never holds more contracts than the
    manager counts there; at most three orders per market (a top-up after each of two partial fills) and never more
    contracts than the largest size wanted; and after the chaos the book converges to what is wanted (the top-up rule
    leaves anything from half to all of it)."""
    clock = [time.time()]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    for seed in range(8):
        fake = RealisticChaos(seed)
        fake._as_of = clock[0]
        m = LiveOrderManager(fake, ttl_s=1200, max_writes_per_cycle=40)
        m.fallback_lag_s = 3.0
        tick = [x["ticker"] for x in fake.markets]
        close = {t: int(clock[0]) + 86400 for t in tick}
        rng = random.Random(100 + seed)
        for cycle in range(300):
            clock[0] += 2.0
            want = []
            for t in tick:
                if rng.random() < 0.6:
                    q = fake.quotes[t.rsplit("-", 1)[1]]
                    px = min(98, max(q[1] + rng.choice([0, 1, 2]), 2))
                    want.append(DesiredOrder(t, "ask", px, float(rng.choice([2, 5, 10, 12])), "A", 0.0))
            if cycle % 3 == 0:
                try:
                    m.sync()
                except Exception:
                    pass
            m.reconcile(want, close)
            fake.fill_some()
            assert known(m, fake), f"seed {seed} cycle {cycle}: an exchange order is not tracked"
            mine, ex, n = {}, {}, {}
            for o in m.resting_list():
                mine[o.key] = mine.get(o.key, 0.0) + o.remaining
            for o in fake.resting.values():
                k = (o["ticker"], o["book_side"])
                ex[k] = ex.get(k, 0.0) + float(o["remaining_count_fp"])
                n[k] = n.get(k, 0) + 1
            for k, v in ex.items():
                assert v <= mine.get(k, 0.0) + 1e-9, f"seed {seed} cycle {cycle}: {k} holds {v}, counted {mine.get(k)}"
                assert v <= 12 and n[k] <= 3, f"seed {seed} cycle {cycle}: {k} holds {v} in {n[k]} orders"
        fake.chaos = False
        final = [DesiredOrder(tick[4], "ask", 5, 10.0, "A", 0.0)]
        for _ in range(8):
            clock[0] += 5.0
            m.sync()
            m.reconcile(final, close)
        got = [(o["ticker"], o["yes_price_dollars"], float(o["remaining_count_fp"])) for o in fake.resting.values()]
        assert all(x[:2] == (tick[4], "0.0500") for x in got) and 5.0 <= sum(x[2] for x in got) <= 10.0, got
        assert all(o.state == "resting" for o in m.resting_list()) and len(m.resting_list()) == len(got)
        assert not fake.dups, f"seed {seed}: {len(fake.dups)} orders placed beside one not mostly filled: {fake.dups[:3]}"
