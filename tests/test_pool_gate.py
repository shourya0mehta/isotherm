"""Round 5's NBM-departure tilt in the pool (research/ml/r5_pool_gate.py "nbmdep"): the live pool's math, the refit
that keeps the design and reads the stored NBM values, and the runner logging them."""
import datetime as dt
import json

import numpy as np

from wxbot.ml import features as F
from wxbot.ml import pool_fit
from wxbot.ml.live import Pool, nbm_dep
from test_ml import _fake_snapshots, ladder

MKT = {"T-LO": 0.1, "B78": 0.2, "B80": 0.4, "B82": 0.2, "T-HI": 0.1}


def _pool(tmp_path, gate=None):
    w = {"hours": [3], "alpha": [0.8], "beta": [0.4], "theta": [0.0], "kappa": [0.0], "z_scale_f": 2.0}
    if gate:
        w["gate"] = {"cols": ["nbmdep"], "alpha": [0.0], "beta": [0.0], "theta": [gate], "kappa": [0.0], "clip_f": 4.0}
    p = tmp_path / f"w{gate}.json"
    p.write_text(json.dumps(w))
    return Pool(str(p))


def test_covariate_and_the_tilt_fades_nbm():
    assert nbm_dep(None, 80.0) == 0.0 and nbm_dep(float("nan"), 80.0) == 0.0
    assert nbm_dep(82.0, 80.0) == 0.5 and nbm_dep(90.0, 80.0) == 1.0 and nbm_dep(70.0, 80.0) == -1.0
    import tempfile, pathlib
    tmp = pathlib.Path(tempfile.mkdtemp())
    f = np.full(5, 0.2)
    plain = _pool(tmp).probs(ladder(), MKT, f, 3, nbs=84.0)
    tilted = _pool(tmp, gate=-0.6).probs(ladder(), MKT, f, 3, nbs=84.0)
    assert abs(sum(tilted.values()) - 1) < 1e-9
    # NBM warmer than the market (market mean 80.0): mass moves to the cold side, away from NBM
    assert tilted["T-LO"] > plain["T-LO"] and tilted["T-HI"] < plain["T-HI"]
    # without NBM's value, or NBM on the market's mean, the tilt does nothing
    no_nbm = _pool(tmp, gate=-0.6).probs(ladder(), MKT, f, 3, nbs=None)
    assert all(abs(no_nbm[k] - plain[k]) < 1e-12 for k in plain)


def test_fit_with_the_gate_and_the_refit_keeps_it(tmp_path):
    db = str(tmp_path / "w.sqlite")
    markets = _fake_snapshots(db)
    import sqlite3
    from wxbot.store.db import Store
    Store(db)                                    # creates ml_snap_ctx
    c = sqlite3.connect(db)
    evs = sorted({r[0] for r in c.execute("SELECT event FROM ml_snapshots")})
    c.executemany("INSERT INTO ml_snap_ctx VALUES (?,?,?,?)", [(e, h, 71.0, 1.0) for e in evs for h in F.DEC_H])
    c.commit()
    get = lambda ev: markets.get(ev, [])
    rows = pool_fit.live_rows(db, dt.date(2026, 9, 1), dt.date(2026, 9, 28), get)
    assert rows and all(r["nbs"] == 71.0 for r in rows)
    out, msg = pool_fit.refit(db, [], dt.date(2026, 9, 30), get, min_rows=100, gate="nbmdep")
    assert out is not None and out["gate"]["cols"] == ["nbmdep"] and pool_fit.sane(out)
    assert out["gate"]["rows_with_nbm"] == len(rows) and "with NBM's max" in msg
    assert len(out["alpha"]) == len(F.DEC_H)
    bad = json.loads(json.dumps(out))
    bad["gate"]["theta"] = [2.5]
    assert not pool_fit.sane(bad)
    # a database from before the context table: rows come back without NBM (no tilt for them)
    old = str(tmp_path / "old.sqlite")
    m2 = _fake_snapshots(old)
    rows2 = pool_fit.live_rows(old, dt.date(2026, 9, 1), dt.date(2026, 9, 28), lambda e: m2.get(e, []))
    assert rows2 and all(r["nbs"] is None for r in rows2)
    # the gate column: 0 without NBM, clipped departure from the market's mean otherwise
    g = pool_fit.gate_column([dict(rows2[0]), dict(rows2[0], nbs=rows2[0]["lo"][0] - 20)])
    assert g[0] == 0.0 and g[1] == -1.0


def test_runner_passes_nbm_to_the_pool_and_logs_it(tmp_path):
    from wxbot.app.runner import Bot
    from wxbot.models.fair_value import FairValueResult
    from wxbot.store.db import Store
    from wxbot.strategy.types import EventBook

    class FakeML:
        def get(self, icao, D):
            P = np.zeros(F.NCLS); P[F.K] = 0.6; P[F.K + 2] = 0.4
            return {"h": 15, "P": P, "anchor_r": 80.0, "nbs_txn": 84.0}

    w = {"hours": F.DEC_H, "alpha": [0.9] * 7, "beta": [0.4] * 7, "theta": [0.0] * 7, "kappa": [0.0] * 7,
         "z_scale_f": 2.0, "gate": {"cols": ["nbmdep"], "alpha": [0.0], "beta": [0.0], "theta": [-0.6],
                                    "kappa": [0.0], "clip_f": 4.0}}
    pw = tmp_path / "pw.json"
    pw.write_text(json.dumps(w))
    bot = Bot.__new__(Bot)
    bot.ml, bot.pool, bot.ml_sha, bot.ml_logged = FakeML(), Pool(str(pw)), "test", set()
    bot.store = Store(str(tmp_path / "t.db"))
    ev = EventBook("KXHIGHNY-26OCT02", "KXHIGHNY", "high", "KNYC", dt.date(2026, 10, 2), 0.0, [], ladder())
    fv = FairValueResult(dict(MKT), dict(MKT), {}, set(), 0.0, [], False)
    out = bot.apply_ml(ev, fv, 1.0)
    w0 = dict(w)
    w0.pop("gate")
    pw0 = tmp_path / "pw0.json"
    pw0.write_text(json.dumps(w0))
    f = np.array([0.0, 0.0, 0.6, 0.4, 0.0])
    from wxbot.ml.live import bracket_pmf
    plain = Pool(str(pw0)).probs(ladder(), MKT, bracket_pmf(ladder(), FakeML().get(0, 0)["P"], 80.0), 15)
    assert out.probs["T-HI"] < plain["T-HI"]       # NBM 84 vs a market mean of 80: faded
    assert bot.store.c.execute("SELECT event, h, nbs FROM ml_snap_ctx").fetchall() == [("KXHIGHNY-26OCT02", 15, 84.0)]
