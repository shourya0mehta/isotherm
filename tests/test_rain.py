"""Sleeve R (rain markets, shadow only): the feature windows, the frozen model, the taker rule and the runner's log."""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wxbot.ml import rain as R
from wxbot.strategy import rain_taker as RT


def tiny_raw(tmp_path):
    """One station (Atlanta, LST = UTC-5): an NBS run with two 6-hour PoP periods in the local day, hourly METAR
    precipitation, and last year's climate report for the normal."""
    raw = tmp_path / "raw"
    for d in ("mos", "obs", "cli"):
        (raw / d).mkdir(parents=True)
    # an older 06Z run (superseded at the decision times below) and the 12Z run; files under 200 bytes are skipped
    pd.DataFrame({"runtime": ["2026-10-03 06:00:00"] * 3 + ["2026-10-03 12:00:00"] * 3,
                  "ftime": ["2026-10-03 12:00:00", "2026-10-03 18:00:00", "2026-10-04 00:00:00"] +
                           ["2026-10-03 18:00:00", "2026-10-04 00:00:00", "2026-10-04 06:00:00"],
                  "model": ["NBS"] * 6, "station": ["KATL"] * 6,
                  "p06": [90, 90, 90, 30, 60, 10], "p12": [None, 90, None, None, 70, None],
                  "q06": [5, 5, 5, 1, 2, 0], "q12": [None, 6, None, None, 3, None]}
                 ).to_csv(raw / "mos" / "NBS_KATL_live.csv", index=False)
    pd.DataFrame({"station": ["ATL"] * 4,
                  "valid": ["2026-10-03 04:52", "2026-10-03 13:52", "2026-10-03 14:20", "2026-10-03 14:52"],
                  "p01i": ["0.05", "T", "0.01", "0.02"]}).to_csv(raw / "obs" / "asos_live.csv", index=False)
    json.dump([{"valid": "2025-10-03", "precip": "0.00", "precip_normal": 0.11,
                "product": "202510040600-KFFC-CDUS42-CLIATL"}], open(raw / "cli" / "KATL_2025.json", "w"))
    return str(raw)


def test_feature_windows_respect_the_local_day_and_the_decision_time(tmp_path):
    raw = tiny_raw(tmp_path)
    df = R.rows(raw, "KATL", pd.DataFrame({"D": ["2026-10-03"] * 2, "h": [15, 21]}))
    a, b = df.iloc[0], df.iloc[1]
    # local day starts 05Z: the 04:52Z report is the previous day; by 15Z (cut 14:55) the 13Z and 14Z hours count,
    # each hour once at its largest report (0.02), the trace only flags
    assert abs(a["obs_p"] - 0.02) < 1e-9 and a["obs_tr"] == 1 and a["obs_wet"] == 1
    assert a["hours_left"] == 14 and b["hours_left"] == 8
    # NBS 12Z run is public at 13:15Z; at 15Z all three periods remain (ends 18Z, 00Z, 06Z; mids in the local day)
    assert a["nbs_p06_max_rem"] == 60 and abs(a["nbs_p06_any_rem"] - (1 - 0.7 * 0.4 * 0.9)) < 1e-9
    # at 21Z the 12-18Z period has ended
    assert b["nbs_p06_max_rem"] == 60 and abs(b["nbs_p06_any_rem"] - (1 - 0.4 * 0.9)) < 1e-9
    assert b["nbs_p06_max_day"] == 60 and b["nbs_q06_max_rem"] == 2
    assert abs(a["pnorm"] - 0.11) < 1e-9          # the same calendar day's normal from last year's report
    assert np.isnan(a["gfs_p06_max_rem"])         # models without files stay missing (the trees route them)


def test_frozen_model_and_numpy_engine_agree():
    m = R.RainModel()
    assert m.features[-1] == "st" and len(m.w) == 3
    rows = pd.DataFrame([{"icao": "KATL", "obs_p": 0.0, "hours_left": 10.0, "nbs_p06_max_rem": 20.0,
                          "nam_p06_max_day": 30.0, "pnorm": 0.1},
                         {"icao": "KMIA", "obs_p": 0.3, "obs_wet": 1, "hours_left": 3.0, "pnorm": 0.2}])
    p = m.predict_rows(rows)
    assert ((p > 0) & (p < 1)).all() and p[1] > p[0]    # rain already measured -> far more likely
    from wxbot.ml.npgbm import NumpyBooster
    import gzip
    nb = NumpyBooster(gzip.open(os.path.join(R.ART, "rain_tier1.txt.gz"), "rt").read())
    assert np.allclose(np.asarray(nb.predict(m.matrix(rows)), float).reshape(-1), p, atol=1e-9)
    # pool: monotone in both inputs
    assert m.pool(0.5, 0.8) > m.pool(0.5, 0.2) and m.pool(0.7, 0.5) > m.pool(0.3, 0.5)


def test_taker_rule_fee_margin_and_extremes():
    assert RT.fee(0.5) == 0.02 and RT.fee(0.10) == 0.01
    q = {"A": (0.40, 0.42), "B": (0.10, 0.12), "C": (0.00, 0.01), "D": (0.30, 0.60), "E": (0.99, 1.00)}
    sig = RT.signals(q, {"A": 0.20, "B": 0.35, "C": 0.50, "D": 0.0, "E": 0.0})
    got = {(x["ticker"], x["side"]): x["price"] for x in sig}
    assert got == {("A", "no"): 0.60, ("B", "yes"): 0.12}   # C: 1c ask; D: spread 30c; E: NO at 1c is an extreme
    # just inside and just outside the bar: edge must beat fee + margin
    assert RT.signals({"X": (0.49, 0.50)}, {"X": 0.50 + 0.02 + 0.06 + 0.001})
    assert not RT.signals({"X": (0.49, 0.50)}, {"X": 0.50 + 0.02 + 0.06 - 0.001})


def test_runner_logs_first_signal_per_market_and_settles(tmp_path):
    from wxbot.app.runner import Bot
    from wxbot.store.db import Store

    class FakeModel:
        meta = {"sha256": "abc123def456", "cities": R.RAIN_CITY}
        engine = "test"

        def predict(self, raw, icao, D, h):
            return {"KATL": 0.05, "KMIA": 0.95}.get(icao)

        def pool(self, mid, p):
            return p

    class FakeClient:      # displayed sizes at the touch (the depth the shadow books against since Oct 9)
        def markets(self, **kw):
            return [{"ticker": "KXRAIN-26OCT03-ATL", "status": "active", "yes_bid_dollars": "0.3000",
                     "yes_ask_dollars": "0.3200", "yes_bid_size_fp": "40.00", "yes_ask_size_fp": "12.00"},
                    {"ticker": "KXRAIN-26OCT03-MIA", "status": "active", "yes_bid_dollars": "0.6000",
                     "yes_ask_dollars": "0.6200", "yes_bid_size_fp": "3.00", "yes_ask_size_fp": "25.00"},
                    {"ticker": "KXRAIN-26OCT03-CHI", "status": "active", "yes_bid_dollars": "0.1000",
                     "yes_ask_dollars": "0.1200"}]

        def market(self, t):
            return {"result": "no" if t.endswith("ATL") else "yes"}

    bot = Bot.__new__(Bot)
    bot.store = Store(str(tmp_path / "r.db"))
    bot.rain = FakeModel()
    bot.rain_seen = set()
    rows = bot.rain_signals(FakeClient(), "raw", pd.Timestamp("2026-10-03").date(), 12, 0.06, 5.0)
    assert {(r[2], r[6]) for r in rows} == {("KXRAIN-26OCT03-ATL", "no"), ("KXRAIN-26OCT03-MIA", "yes")}
    bot.store.add_rain_shadow(rows)
    assert bot.rain_signals(FakeClient(), "raw", pd.Timestamp("2026-10-03").date(), 15, 0.06, 5.0) == []
    bot.store.c.execute("UPDATE rain_shadow SET ts = ts - 9 * 3600")
    bot.rest = FakeClient()
    bot.settle_rain_shadow()
    pnl, cnt, n = bot.store.rain_pnl()
    # ATL: NO at 70c + 2c fee wins 28c; MIA: YES at 62c + 2c fee wins 36c; 5 contracts each
    assert n == 2 and cnt == 10 and abs(pnl - 5 * (0.28 + 0.36)) < 1e-9


def test_refresh_rain_runs_each_decision_hour_once_and_hands_rows_back(tmp_path):
    import datetime as dt
    from wxbot.app.runner import Bot
    from wxbot.store.db import Store

    seen = []

    class Raw:
        root = "raw"
        last_refresh_ts = time.time()

        def refresh(self, full=False):
            raise AssertionError("fresh data: no refresh expected")

    class ML:
        raw = Raw()

    class Rest:
        base = "https://example.invalid"

    bot = Bot.__new__(Bot)
    bot.store = Store(str(tmp_path / "rr.db"))
    bot.rain = object()
    bot.ml, bot.rest = ML(), Rest()
    bot.cfg_r = {"enabled": True, "size": 5, "margin": 0.06}
    bot.rain_done, bot.rain_pending, bot.rain_seen = set(), [], set()
    bot.rain_thread = bot.ml_thread = None

    def fake_signals(client, root, D, h, margin, size):
        seen.append((D, h))
        return [(time.time(), f"KXRAIN-{D:%y%b%d}".upper(), f"T{len(seen)}", "NYC", D.isoformat(), h, "no", 0.7, size,
                 0.02, 0.1, 0.2, 0.75, 0.1, "sha")]
    bot.rain_signals = fake_signals
    t21 = dt.datetime(2026, 10, 3, 21, 0, 40, tzinfo=dt.timezone.utc).timestamp()
    bot.refresh_rain(now=t21)
    bot.rain_thread.join(5)
    assert sorted(seen) == [(dt.date(2026, 10, 3), 21), (dt.date(2026, 10, 4), -3)]
    bot.refresh_rain(now=t21 + 60)       # drains the rows; nothing new is due
    bot.rain_thread.join(5)
    assert len(seen) == 2 and bot.store.rain_pnl()[2] == 2
    bot.refresh_rain(now=t21 + 2000)     # more than 30 minutes after: no late decisions
    assert len(seen) == 2


def test_r2_routes_extra_sites_to_their_own_folder(tmp_path):
    from wxbot.app.runner import Bot
    from wxbot.store.db import Store
    roots = []

    class M2:
        meta = {"sha256": "e2e22948826c00", "cities": {**R.RAIN_CITY, **R.RAIN_CITY_EXTRA}}

        def predict(self, raw, icao, D, h):
            roots.append((icao, raw))
            return 0.05

        def pool(self, mid, p):
            return p

    class Client:
        def markets(self, **kw):
            return [{"ticker": "KXRAIN-26OCT03-CHI", "status": "active", "yes_bid_dollars": "0.3000",
                     "yes_ask_dollars": "0.3200"},
                    {"ticker": "KXRAIN-26OCT03-ATL", "status": "active", "yes_bid_dollars": "0.3000",
                     "yes_ask_dollars": "0.3200"}]

    bot = Bot.__new__(Bot)
    bot.store = Store(str(tmp_path / "r2.db"))
    bot.rain, bot.rain_seen = object(), set()
    seen2 = set()
    rows = bot.rain_signals(Client(), "main", pd.Timestamp("2026-10-03").date(), 12, 0.06, 5.0, model=M2(),
                            seen=seen2, extra_root="extra")
    assert sorted(roots) == [("KATL", "main"), ("KORD", "extra")]
    assert {r[2] for r in rows} == {"KXRAIN-26OCT03-CHI", "KXRAIN-26OCT03-ATL"} and not bot.rain_seen
    bot.store.add_rain_shadow(rows, "rain_shadow_v2")
    assert bot.store.rain_pnl("rain_shadow_v2")[2] == 2 and bot.store.rain_pnl()[2] == 0
