"""Candidate B's shadow log (wxbot/app/early_shadow.py): timing, the private folder, the frozen rule's log, counts only."""
import datetime as dt
import gzip
import json
import os

import numpy as np
import pandas as pd
import pytest

from wxbot.app import early_shadow as ES
from wxbot.ml.live import ART, Pool
from wxbot.store.db import Store

UTC = dt.timezone.utc


def test_decision_slots_are_15z_and_18z_for_thirty_minutes():
    t = dt.datetime(2026, 10, 8, 15, 5, tzinfo=UTC)
    assert ES.decision_slot(t) == (dt.datetime(2026, 10, 8, 15, tzinfo=UTC), -9)
    assert ES.decision_slot(t.replace(hour=18, minute=30)) == (dt.datetime(2026, 10, 8, 18, tzinfo=UTC), -6)
    assert ES.decision_slot(t.replace(hour=18, minute=31)) is None
    assert ES.decision_slot(t.replace(hour=14, minute=59)) is None
    assert ES.decision_slot(t.replace(hour=21, minute=5)) is None      # Sleeve X's hours are not this log's


def test_copy_stable_skips_unchanged_files_and_never_keeps_a_half_written_one(tmp_path, monkeypatch):
    src, dst = tmp_path / "a.csv", tmp_path / "b.csv"
    src.write_text("x,y\n1,2\n")
    assert ES.copy_stable(str(src), str(dst)) == "copied" and dst.read_text() == "x,y\n1,2\n"
    assert ES.copy_stable(str(src), str(dst)) == "same"
    real = ES.shutil.copy2

    def writer_mid_copy(a, b):              # the live bot rewrites the file while it is being copied
        real(a, b)
        with open(a, "a") as fh:
            fh.write("3,4\n")
        os.utime(a, ns=(os.stat(a).st_atime_ns, os.stat(a).st_mtime_ns + 10 ** 9))
    monkeypatch.setattr(ES.shutil, "copy2", writer_mid_copy)
    src.write_text("x,y\n5,6\n")
    os.utime(src, ns=(1, 2 * 10 ** 18))
    assert ES.copy_stable(str(src), str(dst), tries=2, wait_s=0) == "busy"
    assert dst.read_text() == "x,y\n1,2\n" and not os.path.exists(str(dst) + ".part")


def test_hrrr_files_are_merged_so_the_private_hours_survive_a_sync(tmp_path):
    live, priv = tmp_path / "live", tmp_path / "priv"
    (live / "hrrr").mkdir(parents=True)
    (live / "obs").mkdir()
    (live / "obs" / "asos_live.csv").write_text("station,valid\nNYC,2026-10-07 14:51\n")
    (live / "obs" / "asos_live.csv.tmp").write_text("half")
    run = live / "hrrr" / "20261007_12z.csv"
    pd.DataFrame({"icao": ["KNYC"] * 20, "fh": range(1, 21), "t2m_f": 60.0, "d2m_f": np.nan}).to_csv(run, index=False)
    assert ES.sync_raw(str(live), str(priv)) == {"same": 0, "copied": 2, "merged": 0, "busy": 0}
    assert not (priv / "obs" / "asos_live.csv.tmp").exists()
    full = pd.DataFrame({"icao": ["KNYC"] * 48, "fh": range(1, 49), "t2m_f": 61.0, "d2m_f": np.nan})
    full.to_csv(priv / "hrrr" / "20261007_12z.csv", index=False)       # the shadow fetched all 48 hours
    assert ES.sync_raw(str(live), str(priv))["same"] == 2               # the live bot's 20 hours add nothing
    assert len(pd.read_csv(priv / "hrrr" / "20261007_12z.csv")) == 48
    pd.DataFrame({"icao": ["KLAX"] * 20, "fh": range(1, 21), "t2m_f": 70.0, "d2m_f": np.nan}).to_csv(
        live / "hrrr" / "20261007_12z.csv", mode="a", header=False, index=False)
    assert ES.sync_raw(str(live), str(priv))["merged"] == 1
    m = pd.read_csv(priv / "hrrr" / "20261007_12z.csv")
    assert len(m) == 68 and m.groupby("icao")["fh"].max().to_dict() == {"KLAX": 20, "KNYC": 48}


def test_fetch_hrrr_asks_for_every_hour_training_saw_a_few_at_a_time_and_retries_an_incomplete_run(tmp_path):
    coords = json.load(open(ES.COORDS))
    calls = []
    published = {12: 30}                                  # the 12Z run has 30 hours out at first

    def fake_fetch(day, hh, fhs, root, cs):
        fhs = list(fhs)
        calls.append((hh, fhs[0], fhs[-1]))
        if hh == 12 and fhs[-1] == 48:
            published[12] = 48                               # ... and the rest by the next try
        path = os.path.join(root, "hrrr", f"{day:%Y%m%d}_{hh:02d}z.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        have = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["icao", "fh", "t2m_f", "d2m_f"])
        new = [(s, f, 60.0, np.nan) for s in cs for f in fhs if f <= published.get(hh, 99) and f not in set(have["fh"])]
        pd.concat([have, pd.DataFrame(new, columns=have.columns)]).to_csv(path, index=False)
        return len(new) // len(cs)
    dec = dt.datetime(2026, 10, 8, 15, tzinfo=UTC)
    assert ES.fetch_hrrr(str(tmp_path), dec, fetch=fake_fetch, wait_s=0) is True
    assert max(c[2] - c[1] + 1 for c in calls) <= 4                      # never more than four fields held at once
    twelve = [c for c in calls if c[0] == 12]
    assert len(twelve) == 24 and sorted({f for c in twelve for f in range(c[1], c[2] + 1)}) == list(range(1, 49))
    assert sorted({f for c in calls if c[0] == 6 for f in range(c[1], c[2] + 1)}) == list(range(1, 49))
    assert len([c for c in calls if c[0] == 6]) == 12                    # complete on the first try
    calls.clear()
    assert ES.fetch_hrrr(str(tmp_path), dec.replace(hour=18), fetch=fake_fetch, wait_s=0) is True
    assert sorted({f for c in calls if c[0] == 15 for f in range(c[1], c[2] + 1)}) == list(range(1, 19))
    assert {c[0] for c in calls} == {15, 12} and len(coords) == 24
    assert ES.fetch_hrrr(str(tmp_path / "none"), dec, fetch=lambda *a: 0, tries=2, wait_s=0) is False


def test_books_fall_back_to_one_market_at_a_time():
    class Rest:
        def orderbooks(self, tickers):
            raise RuntimeError("401 unauthorized")

        def orderbook(self, t):
            if t == "B":
                raise RuntimeError("404")
            return {"yes_dollars": [["0.40", "3"]]}
    assert ES.read_books(Rest(), ["A", "B"]) == {"A": {"yes_dollars": [["0.40", "3"]]}}
    assert ES.read_books(Rest(), []) == {}


def test_supported_mirrors_the_runner(tmp_path):
    ok = {"markets": [{"price_level_structure": "linear_cent"}]}
    assert ES.supported(ok)
    assert not ES.supported({"exchange_index": 1, "markets": []})
    assert not ES.supported({"markets": [{"price_level_structure": "deci_cent", "price_ranges": [{"step": "0.001"}]}]})
    assert ES.supported({"markets": [{"price_level_structure": "tapered", "price_ranges": [{"step": "0.01"}]}]})
    assert not ES.supported({"markets": [{"settlement_bounds_type": "floor"}]})


# ------------------------------------------------------------------------------------------------ the full run
def market(t, kind, floor=None, cap=None):
    return {"ticker": t, "status": "active", "strike_type": kind, "floor_strike": floor, "cap_strike": cap}


def book(bid, ask, size=50):
    return {"yes_dollars": [[f"{bid / 100:.2f}", str(size)]], "no_dollars": [[f"{(100 - ask) / 100:.2f}", str(size)]]}


class FakeRest:
    def __init__(self, day="26OCT08"):
        ev = f"KXHIGHNY-{day}"
        self.ms = [market(f"{ev}-T70", "less", cap=70), market(f"{ev}-B70.5", "between", 70, 71),
                   market(f"{ev}-B72.5", "between", 72, 73), market(f"{ev}-T73", "greater", floor=73)]
        self.events_calls = []
        self.results = {}

    def events(self, series, status="open", nested=True, limit=10):
        self.events_calls.append(series)
        if series == "KXHIGHNY":
            return [{"event_ticker": self.ms[0]["ticker"].rsplit("-", 1)[0], "markets": self.ms},
                    {"event_ticker": "KXHIGHNY-26OCT07", "markets": [market("KXHIGHNY-26OCT07-T70", "less", cap=70)]}]
        return []

    def orderbooks(self, tickers):
        return {t: book(23, 27) for t in tickers}

    def market(self, t):
        return {"result": self.results.get(t, "")}


class FakeModel:
    engine, sha, pool_sha = "numpy", "abc123", "def456"

    def __init__(self):
        self.pool = Pool(os.path.join(ART, "pool_weights_r2b.json"))

    def predict(self, df):
        P = np.full((len(df), 25), 0.06 / 24)
        P[:, 12] = 0.94                                   # 94% on 70 F: the 70-71 bracket
        return P, np.full(len(df), 70.0)


@pytest.fixture
def shadow(tmp_path, monkeypatch):
    (tmp_path / "data" / "ml_raw" / "obs").mkdir(parents=True)
    (tmp_path / "data" / "ml_raw" / "obs" / "asos_live.csv").write_text("station,valid\n")
    built = []

    def fake_rows(raw, targets, h, today):
        built.append((targets, h, today))
        return pd.DataFrame({"icao": [i for i, _ in targets], "hrrr_age_h": 3.0})
    monkeypatch.setattr(ES, "build_rows", fake_rows)
    monkeypatch.setattr(ES, "fetch_hrrr", lambda root, dec, fetch=None, wait_s=0: True)
    real_now = {"t": dt.datetime(2026, 10, 7, 15, 5, tzinfo=UTC)}
    monkeypatch.setattr(ES, "_utcnow", lambda: real_now["t"])
    return tmp_path, built, real_now


def test_a_run_logs_the_frozen_rules_signals_once_per_bracket_and_side_over_both_hours(shadow):
    base, built, clock = shadow
    rest, model = FakeRest(), FakeModel()
    s = ES.run(str(base), rest=rest, model=model)
    assert s["h"] == -9 and s["day"] == "2026-10-08" and s["events"] == 1 and s["liquid"] == 1
    assert built == [([("KNYC", dt.date(2026, 10, 8))], -9, dt.date(2026, 10, 7))]
    assert sorted(rest.events_calls) == sorted(ES.SERIES)
    st = Store(str(base / "data" / "early_shadow.sqlite"))
    rows = st.c.execute("SELECT ticker, side, price, size, h, model_sha FROM taker_shadow ORDER BY ticker, side").fetchall()
    assert ("KXHIGHNY-26OCT08-B70.5", "yes", 0.27, 5.0, -9, "abc123") in rows
    assert all(r[1] == "no" for r in rows if not r[0].endswith("B70.5")) and len(rows) == s["signals"] >= 2
    snaps = st.c.execute("SELECT event, h, ticker, market, tier1, pooled FROM ml_snapshots").fetchall()
    assert len(snaps) == 4 and abs(sum(r[5] for r in snaps) - 1) < 1e-9 and all(r[1] == -9 for r in snaps)
    assert abs(next(r[4] for r in snaps if r[2].endswith("B70.5")) - 0.94) < 0.02
    # 18Z: the same brackets and sides are not logged again; the hour's snapshots are
    clock["t"] = dt.datetime(2026, 10, 7, 18, 5, tzinfo=UTC)
    s2 = ES.run(str(base), rest=rest, model=model)
    assert s2["h"] == -6 and s2["signals"] == 0
    assert st.c.execute("SELECT COUNT(*) FROM taker_shadow").fetchone()[0] == len(rows)
    assert st.c.execute("SELECT COUNT(*) FROM ml_snapshots WHERE h = -6").fetchone()[0] == 4
    assert [r[0] for r in st.c.execute("SELECT h FROM early_runs ORDER BY ts")] == [-9, -6]


def test_outside_the_slots_it_only_settles_and_a_dry_run_writes_nothing(shadow):
    base, built, clock = shadow
    rest, model = FakeRest(), FakeModel()
    ES.run(str(base), rest=rest, model=model)
    st = Store(str(base / "data" / "early_shadow.sqlite"))
    n = st.c.execute("SELECT COUNT(*) FROM taker_shadow").fetchone()[0]
    clock["t"] = dt.datetime(2026, 10, 7, 16, 0, tzinfo=UTC)      # too late for 15Z
    built.clear()
    s = ES.run(str(base), rest=rest, model=model)
    assert "note" in s and built == [] and st.c.execute("SELECT COUNT(*) FROM early_runs").fetchone()[0] == 1
    clock["t"] = dt.datetime(2026, 10, 8, 16, 0, tzinfo=UTC)      # the next day: Oct 8 markets have settled
    rest.results = {t: "yes" if t.endswith("B70.5") else "no" for t in [m["ticker"] for m in rest.ms]}
    assert ES.run(str(base), rest=rest, model=model)["settled"] == n
    assert st.c.execute("SELECT COUNT(*) FROM taker_shadow WHERE result IS NULL").fetchone()[0] == 0
    # a dry run at a pretend 18Z computes but logs nothing; a real run cannot pretend
    d = ES.run(str(base), now=dt.datetime(2026, 10, 8, 18, 5, tzinfo=UTC), dry_run=True, rest=FakeRest("26OCT09"),
               model=model)
    assert d["signals"] >= 2 and st.c.execute("SELECT COUNT(*) FROM early_runs").fetchone()[0] == 1
    with pytest.raises(SystemExit):
        ES.run(str(base), now=dt.datetime(2026, 10, 8, 18, 5, tzinfo=UTC), rest=rest, model=model)


def test_status_gives_counts_only(shadow):
    base, _, _ = shadow
    assert ES.status(str(base)) == {"runs": 0}
    ES.run(str(base), rest=FakeRest(), model=FakeModel())
    s = ES.status(str(base))
    assert s["runs"] == 1 and s["open"] == sum(s["signals_by_h"].values()) and s["settled"] == 0
    txt = json.dumps(s).lower()
    assert "pnl" not in txt and "usd" not in txt and "profit" not in txt and "result" not in txt


def test_model_artifacts_pass_their_hash_and_parity_checks_on_both_engines(monkeypatch):
    if not os.path.exists(os.path.join(ART, "tier1_r2b.txt.gz")):
        pytest.skip("candidate B's artifacts are not built")
    import sys
    m = ES.EarlyModel()
    assert m.parity < 1e-9 and -9 in m.pool.w["hours"] and -6 in m.pool.w["hours"]
    assert m.meta["hours"] == [-9, -6, -3, 3, 9, 12, 15, 18, 21] and m.meta["last_day"] <= "2026-09-30"
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    n = ES.EarlyModel()
    assert n.engine == "numpy" and n.parity < 1e-9
    orig = type(n.booster).predict
    monkeypatch.setattr(type(n.booster), "predict", lambda self, X: orig(self, X) * 0.99 + 0.0004)
    with pytest.raises(RuntimeError, match="parity check failed"):
        ES.EarlyModel()


# ------------------------------------------------------------------------------------------- B's weekly refit
def base_rows():
    import gzip
    return json.load(gzip.open(os.path.join(ART, ES.BASE_ROWS), "rt"))


def test_refit_of_the_early_hours_reproduces_the_shipped_weights_on_the_same_rows():
    rows = [r for r in base_rows() if "2026-07-03" <= r["D"] <= "2026-09-30"]
    out = ES.fit_hours(rows, [-9, -6])
    w = json.load(open(os.path.join(ART, "pool_weights_r2b.json")))
    for name in ("alpha", "beta", "theta", "kappa"):
        assert np.allclose(out[name], w[name][:2], atol=0.01), (name, out[name], w[name][:2])
    assert out["converged"] and ES.sane_early(out, [-9, -6]) and not ES.sane_early(out, [-9, -6, -3])


class RefitRest:
    def __init__(self, results):
        self.results, self.calls = results, []

    def markets(self, event_ticker=None, limit=100):
        self.calls.append(event_ticker)
        return [{"ticker": t, "result": r, **m} for t, (r, m) in self.results.get(event_ticker, {}).items()]

    def paged(self, *a, **k):
        return []


def test_monday_refit_replaces_only_the_early_hours_once_and_uses_the_shadows_own_snapshots(tmp_path):
    data = str(tmp_path)
    st = Store(os.path.join(data, "early_shadow.sqlite"))
    sha = json.load(open(os.path.join(ART, "tier1_r2b_meta.json")))["sha256"][:12]
    ev = "KXHIGHNY-26OCT08"
    ladder = {f"{ev}-T70": ("no", {"strike_type": "less", "cap_strike": 70}),
              f"{ev}-B70.5": ("yes", {"strike_type": "between", "floor_strike": 70, "cap_strike": 71}),
              f"{ev}-T71": ("no", {"strike_type": "greater", "floor_strike": 71})}
    snaps = [(1.0, ev, -9, t, 1 / 3, f, 0.3, sha) for t, f in zip(ladder, (0.2, 0.6, 0.2))]
    snaps += [(1.0, ev, -6, t, 1 / 3, 0.3, 0.3, "othermodel") for t in ladder]      # another model's: ignored
    st.add_ml_snapshots(snaps)
    rest = RefitRest({ev: ladder})
    monday = dt.datetime(2026, 10, 12, 15, 5, tzinfo=UTC)
    assert ES.maybe_refit(st, data, rest, monday.replace(day=13)) is None             # a Tuesday
    assert ES.maybe_refit(st, data, rest, monday.replace(hour=5)) is None              # before 06Z
    msg = ES.maybe_refit(st, data, rest, monday)
    assert "1 shadow" in msg and "new early-hour weights" in msg and rest.calls == [ev]
    w = json.load(open(os.path.join(data, ES.LIVE_POOL)))
    shipped = json.load(open(os.path.join(ART, "pool_weights_r2b.json")))
    assert w["hours"] == shipped["hours"] and w["alpha"][2:] == shipped["alpha"][2:]   # current hours untouched
    assert w["early_refit"]["sources"]["shadow"] == 1 and w["early_refit"]["last_day"] == "2026-10-08"
    assert ES.maybe_refit(st, data, rest, monday.replace(hour=18)) is None            # once a Monday
    m = ES.EarlyModel(pool_path=os.path.join(data, ES.LIVE_POOL))
    assert m.pool.w["alpha"][:2] == w["alpha"][:2]
    assert os.path.exists(os.path.join(data, "early_refit_markets_cache.json.gz"))


def test_a_refit_outside_the_sanity_bounds_leaves_the_weights_alone(tmp_path, monkeypatch):
    data = str(tmp_path)
    st = Store(os.path.join(data, "early_shadow.sqlite"))
    monkeypatch.setattr(ES, "fit_hours", lambda rows, hours: {"hours": hours, "alpha": [3.0, 1.0], "beta": [0.4, 0.4],
                                                              "theta": [0, 0], "kappa": [0, 0], "converged": True})
    msg = ES.maybe_refit(st, data, RefitRest({}), dt.datetime(2026, 10, 12, 15, 5, tzinfo=UTC))
    assert "sanity" in msg and not os.path.exists(os.path.join(data, ES.LIVE_POOL))


# ------------------------------------------------------------------------------------------- candidate C beside B
class FakeAifsModel(FakeModel):
    uses_aifs = True
    sha, pool_sha = "c0ffee", "cafe00"

    def __init__(self):
        self.pool = Pool(os.path.join(ART, "pool_weights_r2c.json"))
        self.seen_cols = None

    def predict(self, df):
        self.seen_cols = list(df.columns)
        return super().predict(df)


def test_candidates_b_and_c_log_to_their_own_databases_and_c_gets_aifs_inputs(shadow, monkeypatch):
    base, built, clock = shadow
    calls = []

    def fake_aifs(df, root, dec):
        calls.append(dec)
        return df.assign(aifs_mx=80.0, aifs_sd=1.0, aifs_q10=78.0, aifs_q90=82.0, aifs_age_h=9.0), True
    monkeypatch.setattr(ES, "add_aifs", fake_aifs)
    b, c = FakeModel(), FakeAifsModel()
    s = ES.run(str(base), rest=FakeRest(), models={"B": b, "C": c})
    assert set(s["models"]) == {"B", "C"} and s["models"]["C"]["model"] == "c0ffee" and len(calls) == 1
    assert "aifs_mx" in c.seen_cols and s["signals"] == s["models"]["B"]["signals"]
    sb = Store(str(base / "data" / "early_shadow.sqlite"))
    sc = Store(str(base / "data" / "early_shadow_c.sqlite"))
    assert {r[0] for r in sb.c.execute("SELECT model_sha FROM taker_shadow")} == {"abc123"}
    assert {r[0] for r in sc.c.execute("SELECT model_sha FROM taker_shadow")} == {"c0ffee"}
    assert sc.c.execute("SELECT COUNT(*) FROM ml_snapshots").fetchone()[0] == 4
    assert ES.status(str(base), "C")["runs"] == 1 and ES.status(str(base), "B")["runs"] == 1


def test_c_without_its_aifs_inputs_still_logs_and_says_so(shadow, monkeypatch):
    base, _, _ = shadow

    def broken(df, root, dec):
        raise RuntimeError("ECMWF down")
    monkeypatch.setattr(ES, "add_aifs", broken)
    s = ES.run(str(base), rest=FakeRest(), models={"C": FakeAifsModel()})
    assert any("AIFS" in n for n in s["models"]["C"]["notes"]) and s["models"]["C"]["events"] == 1


def test_real_c_artifacts_load_with_aifs_inputs():
    if not os.path.exists(os.path.join(ART, "tier1_r2c.txt.gz")):
        pytest.skip("candidate C's artifacts are not built")
    m = ES.EarlyModel(name="tier1_r2c", pool_file="pool_weights_r2c.json")
    assert m.uses_aifs and m.parity < 1e-9 and "aifs_mx_d" in m.meta["feats"]


# ------------------------------------------------------------------------------- G: C with the NBM-departure tilt
def test_g_logs_with_the_tilted_pool_and_stores_nbm(shadow, monkeypatch):
    """G pools C's forecast with its own (tilted) weights, passes NBM's max to the pool and stores it for the refit."""
    base, built, clock = shadow
    monkeypatch.setattr(ES, "add_aifs", lambda df, root, dec: (df.assign(aifs_mx=80.0, aifs_sd=1.0, aifs_q10=78.0,
                                                                          aifs_q90=82.0, aifs_age_h=9.0), True))
    monkeypatch.setattr(ES, "build_rows", lambda raw, targets, h, today: pd.DataFrame(
        {"icao": [i for i, _ in targets], "hrrr_age_h": 3.0, "nbs_txn": 75.0}))
    c = FakeAifsModel()
    g = FakeAifsModel()
    g.pool = Pool(os.path.join(ART, "pool_weights_r2cg.json"))
    assert g.pool.gate and g.pool.gate["theta"][0] < 0
    s = ES.run(str(base), rest=FakeRest(), models={"C": c, "G": g})
    assert set(s["models"]) == {"C", "G"}
    sg = Store(str(base / "data" / "early_shadow_g.sqlite"))
    assert sg.c.execute("SELECT event, h, nbs FROM ml_snap_ctx").fetchall() == [("KXHIGHNY-26OCT08", -9, 75.0)]
    pc = dict(Store(str(base / "data" / "early_shadow_c.sqlite")).c.execute("SELECT ticker, pooled FROM ml_snapshots"))
    pg = dict(sg.c.execute("SELECT ticker, pooled FROM ml_snapshots"))
    # NBM at 75 F sits above the market's mean (ladder 70-73): G moves mass away from the top bracket
    assert pg["KXHIGHNY-26OCT08-T73"] < pc["KXHIGHNY-26OCT08-T73"]


def test_g_reuses_cs_loaded_forecaster(monkeypatch):
    """The real run loads C's model once and gives G its own pool."""
    if not os.path.exists(os.path.join(ART, "tier1_r2c.txt.gz")):
        pytest.skip("candidate C's artifacts are not built")
    specs = ES.available_specs()
    assert [s.tag for s in specs][-2:] == ["C", "G"] and specs[-1].model == specs[-2].model


def test_g_refit_keeps_the_tilt(tmp_path):
    rows = json.load(gzip.open(os.path.join(ART, "pool_rows_base_r2cg.json.gz"), "rt"))
    db = str(tmp_path / "g.sqlite")
    Store(db)
    out, msg = ES.refit_early(db, rows, dt.date(2026, 10, 2), lambda e: [], None, gate="nbmdep")
    assert out is not None and out["gate"]["theta"][0] < 0 and ES.sane_early(out, [-9, -6])
    w = ES.merge_weights(json.load(open(os.path.join(ART, "pool_weights_r2c.json"))), out)
    assert w["gate"] == out["gate"]


def test_a_round5_model_gets_the_median_anchor_and_the_hourly_nbm(shadow, monkeypatch):
    """A C2-style model (meta anchor_mode median, round5, nbf_* inputs) gets the hourly NBM columns once per run and
    its own anchor, while B keeps NBM's."""
    base, built, clock = shadow
    calls = []

    def fake_nbf(df, root, dec):
        calls.append(dec)
        return df.assign(nbf_txn=71.0, nbf_xnd=2.0, nbf_age_h=2.0)
    monkeypatch.setattr(ES, "add_nbf", fake_nbf)

    class FakeR5(FakeModel):
        uses_nbf, anchor_mode, round5 = True, "median", True
        sha, pool_sha = "d4d4d4", "e5e5e5"

        def predict(self, df):
            assert {"nbf_txn", "nbf_age_h"} <= set(df.columns)
            return super().predict(df)
    s = ES.run(str(base), rest=FakeRest(), models={"B": FakeModel(), "C": FakeR5()})
    assert len(calls) == 1 and s["models"]["C"]["model"] == "d4d4d4" and s["models"]["B"]["events"] == 1


# ------------------------------------------------------------------- shadow sizing and the execution log (Oct 9)
class ThinRest(FakeRest):
    """B70.5 offers only 2 at its 27c ask; T70's 23c bid level shows 0 contracts (so its NO buy has no depth)."""

    def orderbooks(self, tickers):
        out = {t: book(23, 27) for t in tickers}
        for t in tickers:
            if t.endswith("B70.5"):
                out[t] = {"yes_dollars": [["0.23", "50"]], "no_dollars": [["0.73", "2"]]}
            elif t.endswith("T70"):
                out[t] = {"yes_dollars": [["0.23", "0"]], "no_dollars": [["0.73", "50"]]}
        return out


def test_signals_book_the_depth_they_saw_and_size_keeps_what_sleeve_e_sends(shadow):
    base, _, _ = shadow
    ES.run(str(base), rest=ThinRest(), model=FakeModel())
    st = Store(str(base / "data" / "early_shadow.sqlite"))
    got = {(t.rsplit("-", 1)[1], side): rest for t, side, *rest in st.c.execute(
        "SELECT ticker, side, size, avail, intended, depth, booked FROM taker_shadow")}
    assert got[("B70.5", "yes")] == [2.0, 2.0, 5.0, 2.0, 2.0]       # partial: 2 offered, 2 booked
    assert got[("T70", "no")] == [5.0, 0.0, 5.0, 0.0, 0.0]          # zero shown: size stays 5, nothing booked
    assert all(v == [5.0, 50.0, 5.0, 50.0, 5.0] for k, v in got.items() if k[0] not in ("B70.5", "T70"))


class ExecRest(FakeRest):
    """FakeRest plus the optional batch read the execution log's re-reads use (a book that has moved: 25c / 29c)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.rereads = []

    def orderbooks_if_budget(self, tickers, reserve_frac=0.5, timeout=None):
        self.rereads.append(list(tickers))
        return {t: book(25, 29, size=7) for t in tickers}


class FakeClock:
    def __init__(self):
        self.t = 5000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_the_execution_log_follows_each_signal_for_fifteen_minutes(shadow):
    base, _, _ = shadow
    clock, rest = FakeClock(), ExecRest()
    s = ES.run(str(base), rest=rest, model=FakeModel(), exec_log={"enabled": True}, sleep=clock.sleep, clock=clock)
    n = s["signals"]
    st = Store(str(base / "data" / "early_shadow.sqlite"))
    dec = st.c.execute("SELECT ref, sleeve, mode, yes_bid, yes_ask, yes_ask_size, book_ts FROM exec_log "
                       "WHERE kind = 'decision'").fetchall()
    assert n >= 2 and len(dec) == n and {r[1:3] for r in dec} == {("B", "shadow")}
    assert all(r[3:] == (0.23, 0.27, 50.0, 5000.0) for r in dec)     # the book the decision priced
    rer = st.c.execute("SELECT offset_s, COUNT(*), MIN(yes_bid), MAX(yes_ask), MAX(ts - due_ts) FROM exec_log "
                       "WHERE kind = 'reread' GROUP BY offset_s ORDER BY offset_s").fetchall()
    assert rer == [(60, n, 0.25, 0.29, 0.0), (300, n, 0.25, 0.29, 0.0), (900, n, 0.25, 0.29, 0.0)]
    assert len(rest.rereads) == 3 and clock.t == 5900.0                 # one batch read per offset, on time
    assert s["exec_log"]["rereads"] == 3 * n and s["exec_log"]["queued"] == 0
    assert ES.status(str(base))["exec_log_rows"] == 4 * n


def test_the_execution_log_is_off_by_default_shared_by_candidates_and_stops_at_the_deadline(shadow, monkeypatch):
    base, _, _ = shadow

    def no_sleep(s):
        raise AssertionError("no execution log: nothing to wait for")
    off = base / "off"
    (off / "data").mkdir(parents=True)
    ES.run(str(off), rest=ExecRest(), model=FakeModel(), sleep=no_sleep)        # off: no table, no waiting
    st = Store(str(off / "data" / "early_shadow.sqlite"))
    assert st.c.execute("SELECT COUNT(*) FROM taker_shadow").fetchone()[0] >= 2
    assert not st.c.execute("SELECT 1 FROM sqlite_master WHERE name = 'exec_log'").fetchone()
    monkeypatch.setattr(ES, "add_aifs", lambda df, root, dec: (df.assign(aifs_mx=80.0, aifs_sd=1.0, aifs_q10=78.0,
                                                                          aifs_q90=82.0, aifs_age_h=9.0), True))
    monkeypatch.setattr(ES, "EXEC_DEADLINE_S", 400)
    clock, rest = FakeClock(), ExecRest()
    s = ES.run(str(base), rest=rest, models={"B": FakeModel(), "C": FakeAifsModel()},
               exec_log={"enabled": True}, sleep=clock.sleep, clock=clock)
    nb, nc = s["models"]["B"]["signals"], s["models"]["C"]["signals"]
    assert nb >= 2 and nc >= 2
    assert len(rest.rereads) == 2                                       # +1 and +5 min, each read once for both
    assert s["exec_log"]["skipped_deadline"] == nb + nc                 # +15 min falls after the deadline
    for db, n in (("early_shadow.sqlite", nb), ("early_shadow_c.sqlite", nc)):
        c = Store(str(base / "data" / db)).c
        assert c.execute("SELECT offset_s, COUNT(*) FROM exec_log WHERE kind = 'reread' GROUP BY offset_s "
                         "ORDER BY offset_s").fetchall() == [(60, n), (300, n)]
    d = ES.run(str(base), now=dt.datetime(2026, 10, 8, 18, 5, tzinfo=UTC), dry_run=True, rest=ExecRest("26OCT09"),
               model=FakeModel(), exec_log={"enabled": True}, sleep=no_sleep)
    assert "exec_log" not in d                                          # a dry run logs nothing
