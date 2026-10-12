import base64
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from wxbot.execution.live import LiveOrderManager, PREFIX
from wxbot.kalshi.auth import KalshiAuth
from wxbot.kalshi.rest import KalshiREST
from wxbot.strategy.types import DesiredOrder


class FakeResp:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = body or {}
        self.content = json.dumps(self._body).encode()
        self.text = self.content.decode()

    def json(self):
        return self._body


class Recorder:
    def __init__(self, key):
        self.calls = []
        self.key = key
        self.n = 0

    def __call__(self, method, url, params=None, data=None, headers=None, timeout=None):
        self.calls.append((method, url, params, json.loads(data) if data else None, headers))
        # verify signature on every authenticated call
        path = url.split("kalshi.com", 1)[1].split("?")[0]
        msg = (headers["KALSHI-ACCESS-TIMESTAMP"] + method + path).encode()
        self.key.public_key().verify(base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"]), msg)
        self.n += 1
        if method == "POST" and url.endswith("/portfolio/events/orders"):
            return FakeResp(201, {"order_id": f"o{self.n}", "client_order_id": data and json.loads(data)["client_order_id"],
                                  "fill_count": "0.00", "remaining_count": json.loads(data)["count"], "ts_ms": 1})
        if method == "GET" and url.endswith("/portfolio/orders"):
            return FakeResp(200, {"orders": [], "cursor": ""})
        if method == "DELETE" and url.endswith("/portfolio/events/orders/batched"):   # the V2 per-order answer
            return FakeResp(200, {"orders": [{"order_id": o["order_id"], "reduced_by": "1.00", "ts_ms": 1}
                                             for o in json.loads(data)["orders"]]})
        return FakeResp(200, {"ok": True})


def make():
    k = Ed25519PrivateKey.generate()
    rest = KalshiREST(base_url="https://external-api.kalshi.com/trade-api/v2", auth=KalshiAuth("kid", k))
    rec = Recorder(k)
    rest.s.request = rec
    return rest, rec


def test_create_order_body_and_signed_path():
    rest, rec = make()
    rest.create_order("KXHIGHNY-26OCT02-T88", "ask", 3, 25, expiration_ts=1890999999)
    m, url, params, body, headers = rec.calls[-1]
    assert m == "POST" and url.endswith("/trade-api/v2/portfolio/events/orders")
    assert body["side"] == "ask" and body["price"] == "0.0300" and body["count"] == "25.00"
    assert body["post_only"] is True and body["time_in_force"] == "good_till_canceled"
    assert body["exchange_index"] == 0 and body["expiration_time"] == 1890999999
    assert body["self_trade_prevention_type"] in ("maker", "taker_at_cross")


def test_reconcile_places_then_keeps_then_cancels():
    rest, rec = make()
    om = LiveOrderManager(rest)
    d = DesiredOrder("T1", "ask", 3, 25, "A", 0.01, event="E")
    s1 = om.reconcile([d], {"T1": int(time.time()) + 86400})
    assert s1["placed"] == 1
    s2 = om.reconcile([d], {"T1": int(time.time()) + 86400})
    assert s2["kept"] == 1 and s2["placed"] == 0
    s3 = om.reconcile([], {})
    assert s3["cancelled"] == 1
    assert any(c[0] == "DELETE" and c[1].endswith("/portfolio/events/orders/batched") for c in rec.calls)


def test_reconcile_price_change_cancels_and_replaces():
    rest, rec = make()
    om = LiveOrderManager(rest)
    om.reconcile([DesiredOrder("T1", "bid", 96, 10, "A", 0.99, event="E")], {})
    s = om.reconcile([DesiredOrder("T1", "bid", 95, 10, "A", 0.99, event="E")], {})
    assert s["cancelled"] == 1 and s["placed"] == 1


def test_client_ids_have_prefix():
    rest, rec = make()
    om = LiveOrderManager(rest)
    om.reconcile([DesiredOrder("T1", "bid", 96, 10, "A", 0.99, event="E")], {})
    body = [c[3] for c in rec.calls if c[0] == "POST"][0]
    assert body["client_order_id"].startswith(PREFIX)
