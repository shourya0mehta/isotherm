"""End-to-end live loop against a fake Kalshi API: universe -> books -> fair value -> risk -> real order calls
-> status -> shutdown cancel. Catches live-only integration bugs without a key or network."""
import datetime as dt
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import wxbot.app.runner as runner


def iso(ts):
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).isoformat().replace("+00:00", "Z")


class FakeKalshi:
    def __init__(self):
        self.n_requests = self.n_errors = 0
        self.last_error = None
        self.resting = {}
        self.created, self.amended, self.cancelled = [], [], []
        close = time.time() + 18 * 3600
        d = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=1)
        tag = f"{d.year % 100:02d}{d.strftime('%b').upper()}{d.day:02d}"
        self.ev = f"KXHIGHNY-{tag}"
        mk = lambda t, st, fl, cap: {"ticker": f"{self.ev}-{t}", "status": "active", "strike_type": st,
                                     "floor_strike": fl, "cap_strike": cap, "close_time": iso(close)}
        self.markets = [mk("T70", "less", None, 70), mk("B70.5", "between", 70, 71), mk("B72.5", "between", 72, 73),
                        mk("B74.5", "between", 74, 75), mk("T75", "greater", 75, None)]
        # YES bid/ask in cents: two tails near-certain NO, middle brackets carry the mass
        self.quotes = {"T70": (1, 2), "B70.5": (20, 23), "B72.5": (45, 48), "B74.5": (25, 28), "T75": (1, 3)}

    # public
    def exchange_status(self):
        return {"trading_active": True}

    def exchange_schedule(self):   # open around the clock (the Thursday pause is tested on its own, at fixed times)
        day = [{"open_time": "00:00", "close_time": "00:00"}]
        return {"maintenance_windows": [], "standard_hours": [dict(
            {d: day for d in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")},
            start_time="2024-12-01T00:00:00Z", end_time="2200-12-01T00:00:00Z")]}

    def events(self, series, status="open", nested=True, limit=10):
        return [{"event_ticker": self.ev, "markets": self.markets}] if series == "KXHIGHNY" else []

    def incentive_programs(self, status="active"):
        return []

    def orderbooks(self, tickers):
        out = {}
        for t in tickers:
            yb, ya = self.quotes[t.rsplit("-", 1)[1]]
            out[t] = {"yes_dollars": [[f"{yb / 100:.4f}", "500.00"]], "no_dollars": [[f"{(100 - ya) / 100:.4f}", "800.00"]]}
        return out

    def market(self, t):
        return {}

    def markets(self, **kw):
        return self.markets

    # portfolio
    def balance(self):
        return {"balance": 62500, "portfolio_value": 0}

    def positions(self, **kw):
        return [], []

    def orders(self, status="resting", **kw):
        return [dict(o) for o in self.resting.values()]

    def fills(self, min_ts=None, **kw):
        return []

    def settlements(self, min_ts=None, **kw):
        return []

    def netting(self):
        return {"netting_configs": [{"subaccount_number": 0, "enabled": False, "exchange_index": 0}]}

    def create_order(self, ticker, side, price_cents, count, **kw):
        if kw.get("post_only"):   # as the exchange: a post-only order that would cross is refused
            yb, ya = self.quotes.get(ticker.rsplit("-", 1)[1], (None, None))
            if (side == "bid" and ya is not None and price_cents >= ya) or \
                    (side == "ask" and yb is not None and price_cents <= yb):
                from wxbot.kalshi.rest import KalshiError
                raise KalshiError(400, '{"error":{"code":"invalid_order","message":"post only order would cross"}}',
                                  "POST", "/portfolio/events/orders")
        oid = f"o{len(self.created) + 1}"
        self.created.append((ticker, side, price_cents, count, kw))
        self.resting[oid] = {"order_id": oid, "client_order_id": kw.get("client_order_id"), "ticker": ticker,
                            "book_side": side, "yes_price_dollars": f"{price_cents / 100:.4f}",
                            "remaining_count_fp": f"{count:.2f}", "fill_count_fp": "0.00",
                            "created_time": iso(time.time()),
                            "expiration_time": iso(kw["expiration_ts"]) if kw.get("expiration_ts") else None}
        return {"order_id": oid, "client_order_id": kw.get("client_order_id"), "remaining_count": f"{count:.2f}",
                "fill_count": "0.00"}

    def amend_order(self, oid, ticker, side, price, total, expiration_ts=None, **kw):
        self.amended.append((oid, price, total, expiration_ts))
        o = self.resting.get(oid)
        if o is not None:   # as the exchange: the new total minus what already filled keeps resting
            o["remaining_count_fp"] = f"{max(float(total) - float(o.get('fill_count_fp') or 0), 0):.2f}"
            if expiration_ts:
                o["expiration_time"] = iso(expiration_ts)
        return {"order_id": oid, "remaining_count": f"{total:.2f}"}

    def cancel_order(self, oid, **kw):
        self.cancelled.append(oid)
        self.resting.pop(oid, None)
        return {"order_id": oid}

    def batch_cancel(self, ids, **kw):
        out = []
        for i in ids:
            rem = self.resting.get(i, {}).get("remaining_count_fp", "0.00")
            self.cancel_order(i)
            out.append({"order_id": i, "reduced_by": rem, "ts_ms": int(time.time() * 1000)})
        return {"orders": out}   # the V2 shape: one entry per order, with error set when that cancel failed

    def user_data_timestamp(self):
        return getattr(self, "as_of", None) or time.time()   # this fake's reads are never stale


def test_live_loop_places_post_only_decided_quotes_and_cleans_up(tmp_path, monkeypatch):
    k = Ed25519PrivateKey.generate()
    pem = k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                          serialization.NoEncryption()).decode()
    monkeypatch.setenv("KALSHI_KEY_ID", "kid")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY", pem)
    monkeypatch.delenv("KALSHI_PRIVATE_KEY_PATH", raising=False)
    cfg = {
        "mode": "live", "env": "prod", "db_path": str(tmp_path / "live.sqlite"), "status_path": str(tmp_path / "st.json"),
        "loop": {"tick_seconds": 10, "universe_every_s": 180, "books_every_s": 15, "trades_every_s": 20,
                 "metar_every_s": 300, "nbm_every_s": 900, "index_every_s": 60, "settle_every_s": 900,
                 "positions_every_s": 0},   # account reads every tick: new orders wait for three equity readings
        "sleeve_a": {"enabled": True, "series": ["KXHIGHNY"], "decided_only": True, "decided_prob": 0.97,
                     "min_edge_c": 2.0, "improve_if_edge_c": 99, "clip_contracts": 25, "decided_max_cost_usd": 20,
                     "min_price_c": 60, "max_price_c": 98},
        "sleeve_b": {"enabled": False, "shadow": False},
        "risk": {"max_loss_per_bracket_usd": 15, "max_loss_per_event_usd": 20, "max_total_open_risk_usd": 150,
                 "max_cash_deployed_usd": 300},
        "execution": {"order_ttl_s": 1200}, "dashboard_every_s": 10 ** 9,
    }
    bot = runner.Bot(cfg, base_dir=str(tmp_path))
    fake = FakeKalshi()
    bot.rest = fake
    bot.live.rest = fake
    bot.nbm.refresh = lambda *a, **kw: None
    monkeypatch.setattr(runner.obs_mod, "fetch_metars", lambda *a, **kw: {})
    # one startup + three ticks, then shutdown
    bot.running = True
    calls = {"n": 0}

    def fake_sleep(s):
        calls["n"] += 1
        if calls["n"] >= 5:
            bot.running = False
    monkeypatch.setattr(runner.time, "sleep", fake_sleep)
    import logging
    errors = []

    class H(logging.Handler):
        def emit(self, rec):
            if rec.levelno >= logging.WARNING and "cancel_all" not in rec.getMessage():
                errors.append(rec.getMessage())
    h = H()
    logging.getLogger().addHandler(h)
    try:
        bot.run()
    finally:
        logging.getLogger().removeHandler(h)
    assert not errors, errors
    assert fake.created, "no orders placed"
    now = time.time()
    for ticker, side, price, count, kw in fake.created:
        assert kw["post_only"] is True and kw["client_order_id"].startswith("wxb-")
        assert side == "ask" and price <= 4  # NO on the near-certain-NO tails, YES price 2-4c => pays 96-98c
        assert ticker.endswith(("-T70", "-T75"))
        assert kw["expiration_ts"] <= now + 1201  # dead-man expiry, never past close
    cost = sum(c * (100 - p) / 100 for _, s, p, c, _ in fake.created)
    assert cost <= 20.0 + 1e-6  # half-cap decided budget per event
    st = __import__("json").load(open(tmp_path / "st.json"))
    assert st["mode"] == "live" and st["netting"] is False and st["available_balance_usd"] == 625.0
    assert not fake.resting, "shutdown must cancel every bot order"
