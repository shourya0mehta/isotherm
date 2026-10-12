"""Tape archiver: settled events are written once, oldest first, with candles and every trade page; a failed page
writes nothing and the event is retried on a later pass."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wxbot.data.tape_archive import TapeArchiver, read_record

T0 = 1790000000   # a fixed past epoch


def iso(t):
    import datetime as dt
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat().replace("+00:00", "Z")


def market(tk, close, vol=10, strike=84.99, result="no"):
    return {"ticker": tk, "open_time": iso(close - 3600), "close_time": iso(close), "floor_strike": strike,
            "strike_type": "greater", "result": result, "expiration_value": "85.1", "volume_fp": str(vol)}


class Resp:
    def __init__(self, code, body=None):
        self.status_code, self._b = code, body

    def json(self):
        return self._b


class FakeSession:
    def __init__(self):
        self.headers = {}
        self.calls = []
        self.fail_trades_once = set()

    def get(self, url, params=None, timeout=None):
        p = dict(params or {})
        self.calls.append((url, p))
        path = url.split("/trade-api/v2", 1)[-1] if "/trade-api/v2" in url else url[len("http://x"):]
        if path == "/events":
            s = p["series_ticker"]
            if s == "KXTEMPMIAH":
                evs = [{"event_ticker": f"KXTEMPMIAH-H{k}", "markets": [market(f"KXTEMPMIAH-H{k}-T84.99", T0 + 3600 * k),
                                                                       market(f"KXTEMPMIAH-H{k}-T85.99", T0 + 3600 * k, vol=0)]}
                       for k in (3, 2, 1)]           # newest first, like the API
                evs.append({"event_ticker": "KXTEMPMIAH-OLD", "markets": [market("KXTEMPMIAH-OLD-T1", T0 - 86400 * 9)]})
                return Resp(200, {"events": evs, "cursor": ""})
            if s == "KXHIGHMIA":
                return Resp(200, {"events": [{"event_ticker": "KXHIGHMIA-D1",
                                              "markets": [market("KXHIGHMIA-D1-B85.5", T0 + 7200),
                                                          market("KXHIGHMIA-D1-T88", T0 + 7200)]}], "cursor": ""})
            return Resp(200, {"events": [], "cursor": ""})
        if path.endswith("/candlesticks") and "/events/" in path:
            ev = path.split("/events/")[1].split("/")[0]
            c = [{"end_period_ts": T0, "yes_bid": {"close_dollars": "0.9500"}, "yes_ask": {"close_dollars": "0.9700"},
                  "price": {"close_dollars": None}, "volume_fp": "3.00"}]
            return Resp(200, {"market_tickers": [f"{ev}-T84.99", f"{ev}-T85.99"], "market_candlesticks": [c, []]})
        if path.endswith("/candlesticks"):
            return Resp(200, {"candlesticks": [{"end_period_ts": T0, "yes_bid": {"close_dollars": "0.1000"},
                                                "yes_ask": {"close_dollars": "0.1200"}, "volume_fp": "1.00"}]})
        if path == "/markets/trades":
            tk = p["ticker"]
            if tk in self.fail_trades_once:
                self.fail_trades_once.discard(tk)
                return Resp(500, {})
            if not p.get("cursor"):
                return Resp(200, {"trades": [{"created_time": iso(T0 + 50), "yes_price_dollars": "0.9600",
                                              "count_fp": "5.00", "taker_side": "no"}], "cursor": "page2"})
            return Resp(200, {"trades": [{"created_time": iso(T0 + 10), "yes_price_dollars": "0.9500",
                                          "count_fp": "2.00", "taker_side": "yes"}], "cursor": ""})
        return Resp(404, {})


def make(tmp_path, sess):
    a = TapeArchiver(str(tmp_path / "tape"), ["KXTEMPMIAH"], ["KXHIGHMIA"], since_ts=T0 - 86400, base="http://x",
                     gap_s=0.0, settle_lag_s=0.0, session=sess)
    return a


def test_archives_settled_events_oldest_first_with_all_trade_pages(tmp_path):
    sess = FakeSession()
    a = make(tmp_path, sess)
    n = a.run_once()
    assert n == 4   # three hourly events (the old one is before `since`) and one daily event
    order = [c[0].split("/")[-2] for c in sess.calls if c[0].endswith("/candlesticks") and "/events/" in c[0]]
    assert order == ["KXTEMPMIAH-H1", "KXTEMPMIAH-H2", "KXTEMPMIAH-H3"]
    r = read_record(a.path_for("hourly", "KXTEMPMIAH", "KXTEMPMIAH-H1"))
    assert r["kind"] == "hourly" and r["close_ts"] == T0 + 3600
    assert r["candles"]["KXTEMPMIAH-H1-T84.99"] == [[T0, 0.95, 0.97, None, 3.0]]
    # both trade pages, oldest first; the zero-volume market has no trade list
    assert [x[1:] for x in r["trades"]["KXTEMPMIAH-H1-T84.99"]] == [[0.95, 2.0, "yes"], [0.96, 5.0, "no"]]
    assert "KXTEMPMIAH-H1-T85.99" not in r["trades"]
    assert {m["ticker"]: m["value"] for m in r["markets"]}["KXTEMPMIAH-H1-T84.99"] == 85.1
    d = read_record(a.path_for("daily", "KXHIGHMIA", "KXHIGHMIA-D1"))
    assert set(d["candles"]) == {"KXHIGHMIA-D1-B85.5", "KXHIGHMIA-D1-T88"} and len(d["trades"]) == 2
    assert not os.path.exists(a.path_for("hourly", "KXTEMPMIAH", "KXTEMPMIAH-OLD"))
    # a second pass writes nothing new and fetches no candles
    sess.calls.clear()
    assert a.run_once() == 0
    assert not [c for c in sess.calls if c[0].endswith("/candlesticks")]
    assert a.status()["files"] == 4 and a.status()["pending"] == 0


def test_a_failed_trade_page_writes_nothing_and_retries_later(tmp_path):
    sess = FakeSession()
    a = make(tmp_path, sess)
    # a 500 is retried inside get(), so the market's trades fail on every try until "healed"
    orig = sess.get

    def flaky(url, params=None, timeout=None):
        if params and params.get("ticker") == "KXTEMPMIAH-H2-T84.99" and not getattr(flaky, "healed", False):
            return Resp(500, {})
        return orig(url, params, timeout)
    sess.get = flaky
    import wxbot.data.tape_archive as TA
    real_wait = a._stop.wait
    a._stop.wait = lambda s=None: False   # no sleeping in tests
    n = a.run_once()
    assert n == 3 and not os.path.exists(a.path_for("hourly", "KXTEMPMIAH", "KXTEMPMIAH-H2"))
    assert a.failed["KXTEMPMIAH-H2"] == 1 and a.stats["errors"] >= 1
    flaky.healed = True
    assert a.run_once() == 1 and os.path.exists(a.path_for("hourly", "KXTEMPMIAH", "KXTEMPMIAH-H2"))
    a._stop.wait = real_wait
    assert TA.compact([]) == []


def test_unsettled_hours_wait_for_the_lag(tmp_path):
    sess = FakeSession()
    a = TapeArchiver(str(tmp_path / "tape"), ["KXTEMPMIAH"], [], since_ts=T0 - 86400, base="http://x", gap_s=0.0,
                     settle_lag_s=time.time() - T0 - 2 * 3600 + 60, session=sess)   # only the first hour is old enough
    assert a.run_once() == 1 and os.path.exists(a.path_for("hourly", "KXTEMPMIAH", "KXTEMPMIAH-H1"))


def test_word_settlement_values_are_kept_not_fatal():
    """Rain markets settle with expiration_value "No"/"Yes": the archive keeps the word instead of failing."""
    from wxbot.data.tape_archive import _val
    assert _val("No") == "No" and _val("83.40") == 83.4 and _val(None) is None and _val("") is None
