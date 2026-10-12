import json
import os
import tempfile

import numpy as np
import pandas as pd

from wxbot.ml import features as F
from wxbot.ml.live import Pool, bracket_pmf
from wxbot.models.brackets import Bracket


def ladder():
    # tails + 2-degree brackets around 80: <=77, 78-79, 80-81, 82-83, >=84
    return [Bracket("T-LO", -1e6, 77, "less", None, 78), Bracket("B78", 78, 79, "between", 78, 79),
            Bracket("B80", 80, 81, "between", 80, 81), Bracket("B82", 82, 83, "between", 82, 83),
            Bracket("T-HI", 84, 1e6, "greater", 83, None)]


def test_bracket_pmf_maps_integers_and_tails():
    P = np.zeros(F.NCLS)
    P[F.K] = 0.5        # anchor 80
    P[F.K + 4] = 0.3    # 84 -> top tail
    P[F.K - 5] = 0.2    # 75 -> bottom tail
    f = bracket_pmf(ladder(), P, 80.0)
    assert abs(f.sum() - 1) < 1e-9
    assert f[2] > 0.49 and f[4] > 0.29 and f[0] > 0.19


def test_pool_reduces_to_market_when_model_weight_zero():
    w = {"hours": [3], "alpha": [1.0], "beta": [0.0], "theta": [0.0], "kappa": [0.0], "z_scale_f": 2.0}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(w, fh)
    pool = Pool(fh.name)
    os.unlink(fh.name)
    br = ladder()
    mkt = {"T-LO": 0.1, "B78": 0.2, "B80": 0.4, "B82": 0.2, "T-HI": 0.1}
    p = pool.probs(br, mkt, np.full(5, 0.2), 3)
    assert abs(sum(p.values()) - 1) < 1e-9
    for k, v in mkt.items():
        assert abs(p[k] - v) < 1e-9


def test_pool_moves_toward_model():
    w = {"hours": [3], "alpha": [0.8], "beta": [0.5], "theta": [0.0], "kappa": [0.0], "z_scale_f": 2.0}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(w, fh)
    pool = Pool(fh.name)
    os.unlink(fh.name)
    br = ladder()
    mkt = {"T-LO": 0.1, "B78": 0.2, "B80": 0.4, "B82": 0.2, "T-HI": 0.1}
    f = np.array([0.02, 0.08, 0.3, 0.4, 0.2])
    p = pool.probs(br, mkt, f, 3)
    assert p["B82"] > mkt["B82"] and p["T-LO"] < mkt["T-LO"]


def test_production_artifacts_present_and_consistent():
    art = os.path.join(os.path.dirname(F.__file__), "artifacts")
    meta = json.load(open(os.path.join(art, "tier1_v2_meta.json")))
    from wxbot.ml import hrrr as H
    assert meta["feats"] in (F.FEATS, F.FEATS_V3, F.FEATS_V3 + H.HRRR_FEATS + H.HRRR_LAG_FEATS)
    assert meta["last_day"] <= "2026-09-30" or "hrrr" not in meta["version"]  # the frozen model never sees the holdout
    pw = json.load(open(os.path.join(art, "pool_weights.json")))
    assert pw["hours"] == F.DEC_H and len(pw["alpha"]) == len(F.DEC_H)


def test_runner_apply_ml_keeps_obs_kills_and_logs_snapshot(tmp_path):
    import datetime as dt
    from wxbot.app.runner import Bot
    from wxbot.models.fair_value import FairValueResult
    from wxbot.store.db import Store
    from wxbot.strategy.types import EventBook

    class FakeML:
        def get(self, icao, D):
            P = np.zeros(F.NCLS); P[F.K] = 0.6; P[F.K + 2] = 0.4  # 80 or 82
            return {"h": 15, "P": P, "anchor_r": 80.0}

    w = {"hours": F.DEC_H, "alpha": [0.9] * 7, "beta": [0.4] * 7, "theta": [0.0] * 7, "kappa": [0.0] * 7,
         "z_scale_f": 2.0}
    pw = tmp_path / "pw.json"
    pw.write_text(json.dumps(w))
    bot = Bot.__new__(Bot)
    bot.ml, bot.pool, bot.ml_sha, bot.ml_logged = FakeML(), Pool(str(pw)), "test", set()
    bot.store = Store(str(tmp_path / "t.db"))
    br = ladder()
    ev = EventBook("KXHIGHNY-26OCT02", "KXHIGHNY", "high", "KNYC", dt.date(2026, 10, 2), 0.0, [], br)
    mkt = {"T-LO": 0.1, "B78": 0.2, "B80": 0.4, "B82": 0.2, "T-HI": 0.1}
    fv = FairValueResult(dict(mkt), dict(mkt), {}, {"T-LO"}, 0.0, [], False)
    out = bot.apply_ml(ev, fv, 1.0)
    assert out.probs["T-LO"] == 0.0 and abs(sum(out.probs.values()) - 1) < 1e-9
    assert out.probs["B82"] > mkt["B82"]
    n = bot.store.c.execute("SELECT COUNT(*) FROM ml_snapshots").fetchone()[0]
    assert n == len(br)
    bot.apply_ml(ev, fv, 2.0)  # same decision hour: no second snapshot
    assert bot.store.c.execute("SELECT COUNT(*) FROM ml_snapshots").fetchone()[0] == len(br)


def test_runner_shadow_taker_logs_first_signal_only(tmp_path):
    import datetime as dt
    from wxbot.app.runner import Bot
    from wxbot.models.fair_value import FairValueResult, LadderQuote
    from wxbot.store.db import Store
    from wxbot.strategy import taker
    from wxbot.strategy.types import EventBook

    class FakeML:
        def __init__(self):
            self.h = 15

        def get(self, icao, D):
            P = np.zeros(F.NCLS); P[F.K + 2] = 1.0  # the model is sure of 82
            return {"h": self.h, "P": P, "anchor_r": 80.0}

    w = {"hours": F.DEC_H, "alpha": [0.9] * 7, "beta": [0.4] * 7, "theta": [0.0] * 7, "kappa": [0.0] * 7,
         "z_scale_f": 2.0}
    pw = tmp_path / "pw.json"
    pw.write_text(json.dumps(w))
    bot = Bot.__new__(Bot)
    bot.ml, bot.pool, bot.ml_sha, bot.ml_logged = FakeML(), Pool(str(pw)), "test", set()
    bot.store = Store(str(tmp_path / "t.db"))
    bot.cfg_x, bot.x_seen, bot.series_t = taker.TakerConfig(enabled=True), set(), set()
    br = ladder()
    ev = EventBook("KXHIGHNY-26OCT02", "KXHIGHNY", "high", "KNYC", dt.date(2026, 10, 2), 0.0, [], br)
    mkt = {"T-LO": 0.1, "B78": 0.2, "B80": 0.4, "B82": 0.2, "T-HI": 0.1}
    ev.quotes = {t: LadderQuote(int(100 * v) - 1, int(100 * v) + 1, 50, 50) for t, v in mkt.items()}
    fv = FairValueResult(dict(mkt), dict(mkt), {}, set(), 0.0, [], False)
    bot.apply_ml(ev, fv, 1.0)
    rows = bot.store.c.execute("SELECT ticker, side, size, h FROM taker_shadow").fetchall()
    assert ("B82", "yes", 5.0, 15) in rows  # the pool lifts B82 well above its 21c ask
    n = len(rows)
    bot.ml.h = 18  # next decision hour: same signals are not logged twice
    bot.apply_ml(ev, fv, 2.0)
    assert bot.store.c.execute("SELECT COUNT(*) FROM taker_shadow").fetchone()[0] == n
    # more than 30 minutes after the decision time (e.g. a restart mid-hour): nothing is logged
    bot.x_seen, bot.ml.h = set(), 21
    late = dt.datetime(2026, 10, 2, 21, 45, tzinfo=dt.timezone.utc).timestamp()
    bot.apply_ml(ev, fv, late)
    assert bot.store.c.execute("SELECT COUNT(*) FROM taker_shadow").fetchone()[0] == n


def test_restart_flag_stops_the_loop_once(tmp_path):
    from wxbot.app.runner import Bot

    class Store:
        def __init__(self):
            self.lines = []

        def log(self, level, msg):
            self.lines.append((level, msg))

    bot = Bot.__new__(Bot)
    bot.status_path = str(tmp_path / "status.json")
    bot.store = Store()
    bot.running = True
    assert bot.restart_requested() is False and bot.running is True
    (tmp_path / "RESTART").write_text("")
    assert bot.restart_requested() is True
    assert bot.running is False and not (tmp_path / "RESTART").exists()


def test_numpy_booster_matches_lightgbm_and_loads_production_model():
    import gzip
    import pandas as pd
    from wxbot.ml.npgbm import NumpyBooster
    art = os.path.join(os.path.dirname(F.__file__), "artifacts")
    model_str = gzip.decompress(open(os.path.join(art, "tier1_v2.txt.gz"), "rb").read()).decode()
    meta = json.load(open(os.path.join(art, "tier1_v2_meta.json")))
    nb = NumpyBooster(model_str)
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(0, 3, (40, len(meta["feats"]))), columns=meta["feats"])
    X.iloc[::3, 4:20] = np.nan
    X["st"] = pd.Categorical(rng.choice(meta["cats"], 40), categories=meta["cats"])
    p = nb.predict(X)
    assert p.shape == (40, len(meta["classes"])) and np.allclose(p.sum(axis=1), 1.0)
    try:
        import lightgbm as lgb
        q = lgb.Booster(model_str=model_str).predict(X)
    except (ImportError, OSError):
        return  # no compiled LightGBM here: the shape/normalization checks above are what we can do
    assert np.abs(p - q).max() < 1e-9


def test_tier1live_falls_back_to_numpy_without_lightgbm(tmp_path, monkeypatch):
    import sys
    from wxbot.ml.live import Tier1Live
    monkeypatch.setitem(sys.modules, "lightgbm", None)  # import lightgbm -> ImportError, as on a Mac without libomp
    t = Tier1Live(str(tmp_path), ["KNYC"])
    assert t.engine == "numpy"


def test_mos_cycle_skip_rule():
    """A 6-hourly MOS model is re-fetched only when a new cycle has been published since the last fetch."""
    import datetime as dt
    from wxbot.ml.live import _new_cycle_since
    ts = lambda h, m=0: dt.datetime(2026, 10, 3, h, m, tzinfo=dt.timezone.utc).timestamp()
    # GFS MOS: 06Z cycle published 10:00Z (240 min lag)
    assert _new_cycle_since("GFS", ts(9, 5), ts(12, 5))       # 06Z cycle appeared at 10:00 -> fetch at 12Z
    assert not _new_cycle_since("GFS", ts(12, 5), ts(15, 5))  # next one (12Z) appears 16:00 -> skip at 15Z
    assert _new_cycle_since("GFS", ts(15, 5), ts(18, 5))
    assert not _new_cycle_since("LAV", ts(14, 45), ts(15, 2))  # 14Z LAMP (14:40) already fetched -> skip at 15Z
    assert _new_cycle_since("LAV", ts(14, 45), ts(15, 41))     # 15Z LAMP appears 15:40
    # NBM text: 12Z cycle published 13:15Z (75 min lag)
    assert not _new_cycle_since("NBS", ts(9, 5), ts(12, 5))   # 06Z (07:15) already fetched at 09Z -> skip at 12Z
    assert _new_cycle_since("NBS", ts(12, 5), ts(15, 5))      # 12Z appeared 13:15 -> fetch at 15Z
    assert _new_cycle_since("MEX", 0.0, ts(1))


def _fake_snapshots(db_path, n_events=40, days=("2026-09-20", "2026-09-28")):
    """ml_snapshots for fake 3-bracket events. Each event has a true probability vector; the market and Tier-1 are
    noisy views of it (the market less noisy), and the winner is drawn from it, like real ladders."""
    import sqlite3
    from wxbot.store.db import Store
    Store(db_path)  # schema
    c = sqlite3.connect(db_path)
    rows, markets = [], {}
    rng = np.random.default_rng(0)
    for d in pd.date_range(*days):
        tag = d.strftime("%y") + d.strftime("%b").upper() + d.strftime("%d")
        for k in range(n_events):
            ev = f"KXHIGHT{k:02d}-{tag}"
            pi = rng.dirichlet([2.0, 2.0, 2.0])
            win = int(rng.choice(3, p=pi))
            ms = []
            for b in range(3):
                ms.append({"ticker": f"{ev}-B{70 + 2 * b}.5", "result": "yes" if b == win else "no",
                           "strike_type": "between", "floor_strike": 70 + 2 * b, "cap_strike": 71 + 2 * b})
            for h in F.DEC_H:
                m = pi * np.exp(0.3 * rng.standard_normal(3)); m /= m.sum()
                f = pi * np.exp(0.6 * rng.standard_normal(3)); f /= f.sum()
                for b in range(3):
                    rows.append((1.0, ev, h, ms[b]["ticker"], float(m[b]), float(f[b]), 0.5, "x"))
            markets[ev] = ms
    c.executemany("INSERT OR IGNORE INTO ml_snapshots VALUES (?,?,?,?,?,?,?,?)", rows)
    c.commit()
    return markets


def test_pool_refit_core_windows_and_bounds(tmp_path):
    import datetime as dt
    from wxbot.ml import pool_fit
    db = str(tmp_path / "w.sqlite")
    markets = _fake_snapshots(db)
    calls = []

    def get_markets(ev):
        calls.append(ev)
        return markets.get(ev, [])
    # window ends two days before today: Sep 30 -> events through Sep 28 count
    out, msg = pool_fit.refit(db, [], dt.date(2026, 9, 30), get_markets, min_rows=100)
    assert out is not None and "from the bot" in msg and pool_fit.sane(out)
    assert out["sources"]["bot"] == 9 * 40 * 7 and out["last_day"] == "2026-09-28"
    assert all(0.5 <= a <= 1.2 for a in out["alpha"])  # the less noisy view (the fake market) carries the pool
    # too few rows: weights unchanged
    out2, msg2 = pool_fit.refit(db, [], dt.date(2026, 9, 30), get_markets, min_rows=10 ** 6)
    assert out2 is None and "unchanged" in msg2
    # an event date outside the window is never fetched
    calls.clear()
    pool_fit.refit(db, [], dt.date(2026, 9, 24), get_markets, min_rows=10 ** 6)
    assert calls and all(pool_fit.event_date_of(e) <= dt.date(2026, 9, 22) for e in calls)
    bad = dict(out, alpha=[3.0] * 7)
    assert not pool_fit.sane(bad)
    # only the deployed forecaster's snapshots count
    same, _ = pool_fit.refit(db, [], dt.date(2026, 9, 30), get_markets, min_rows=100, model_sha="x")
    other, msg3 = pool_fit.refit(db, [], dt.date(2026, 9, 30), get_markets, min_rows=100, model_sha="older")
    assert same is not None and other is None and "0 from the bot" in msg3


def test_pool_refit_reads_the_paper_and_the_live_record(tmp_path, caplog):
    """From the switch to real money on, the paper and the live snapshots are in two files: the refit reads both (no
    hole in its window at the switch), opens the other one read-only, and skips it if it is missing or unreadable."""
    import datetime as dt
    from wxbot.ml import pool_fit
    paper, live = str(tmp_path / "wxbot.sqlite"), str(tmp_path / "live.sqlite")
    markets = _fake_snapshots(paper, days=("2026-09-20", "2026-09-24"))
    markets.update(_fake_snapshots(live, days=("2026-09-25", "2026-09-28")))
    get = lambda ev: markets.get(ev, [])
    D0, D1, today = dt.date(2026, 9, 1), dt.date(2026, 9, 28), dt.date(2026, 9, 30)
    both = pool_fit.live_rows([live, paper], D0, D1, get)
    assert len(both) == 9 * 40 * 7 and min(r["D"] for r in both) == "2026-09-20"
    assert len(pool_fit.live_rows(live, D0, D1, get)) == 4 * 40 * 7
    _, msg = pool_fit.refit([live, paper], [], today, get, min_rows=10 ** 6)
    assert f"(0 research, {9 * 40 * 7} from the bot)" in msg
    # research rows only fill the days before the bot's first day, now the paper bot's
    base = [dict(r, src="research") for r in pool_fit.live_rows(paper, D0, dt.date(2026, 9, 22), get)]
    base += [dict(r, D="2026-09-10", src="research") for r in base[:70]]
    _, msg = pool_fit.refit([live, paper], base, today, get, min_rows=10 ** 6)
    assert f"(70 research, {9 * 40 * 7} from the bot)" in msg
    # a missing other file is skipped and not created; an unreadable one is skipped with a warning
    missing = str(tmp_path / "nope.sqlite")
    assert len(pool_fit.live_rows([live, missing], D0, D1, get)) == 4 * 40 * 7 and not os.path.exists(missing)
    junk = tmp_path / "junk.sqlite"
    junk.write_bytes(b"not a database at all" * 100)
    with caplog.at_level("WARNING"):
        assert len(pool_fit.live_rows([live, str(junk)], D0, D1, get)) == 4 * 40 * 7
    assert "skipped" in caplog.text


def test_pool_refit_keeps_the_ladder_written_first(tmp_path):
    """An event-hour logged by both bots (a switch back and forth within the hour) keeps the whole ladder written
    first, never a mix of the two; if that one misses a settled bracket, the other one."""
    import datetime as dt
    import sqlite3
    from wxbot.ml import pool_fit
    from wxbot.store.db import Store
    ev = "KXHIGHT00-26SEP25"
    ms = [{"ticker": f"{ev}-B{70 + 2 * b}.5", "result": "yes" if b == 1 else "no", "strike_type": "between",
           "floor_strike": 70 + 2 * b, "cap_strike": 71 + 2 * b} for b in range(3)]

    def db(name, ts, m, drop=None):
        path = str(tmp_path / name)
        Store(path)
        c = sqlite3.connect(path)
        c.executemany("INSERT INTO ml_snapshots VALUES (?,?,?,?,?,?,?,?)",
                      [(ts, ev, 9, x["ticker"], m[b], 0.3, 0.3, "x") for b, x in enumerate(ms) if b != drop])
        c.commit()
        c.close()
        return path
    a = db("a.sqlite", 200.0, [0.2, 0.5, 0.3])
    b = db("b.sqlite", 100.0, [0.1, 0.8, 0.1])
    D = dt.date(2026, 9, 25)
    rows = pool_fit.live_rows([a, b], D, D, lambda e: ms)
    assert len(rows) == 1 and rows[0]["m"] == [0.1, 0.8, 0.1]
    c = db("c.sqlite", 50.0, [0.3, 0.3, 0.4], drop=2)              # earliest, but one bracket missing
    rows = pool_fit.live_rows([a, c], D, D, lambda e: ms)
    assert len(rows) == 1 and rows[0]["m"] == [0.2, 0.5, 0.3]


def test_runner_weekly_refit_runs_once_on_monday(tmp_path, monkeypatch):
    import datetime as dt
    from wxbot.app.runner import Bot
    from wxbot.ml import pool_fit
    from wxbot.store.db import Store
    bot = Bot.__new__(Bot)
    bot.store = Store(str(tmp_path / "b.sqlite"))
    bot.ml = object()
    bot.refit_thread = None

    class R:
        base = "https://example.invalid/trade-api/v2"
    bot.rest = R()
    w = {"hours": F.DEC_H, "alpha": [0.9] * 7, "beta": [0.4] * 7, "theta": [0.0] * 7, "kappa": [0.0] * 7,
         "z_scale_f": 2.0, "converged": True}
    wp = tmp_path / "pool_weights.json"
    wp.write_text(json.dumps(w))
    bot.refit_cfg = {"enabled": True, "weekday": 0, "after_utc_hour": 6, "weights_path": str(wp),
                     "base_rows_path": str(tmp_path / "none.json.gz")}
    new_w = dict(w, alpha=[1.0] * 7)
    seen = []
    monkeypatch.setattr(pool_fit, "refit", lambda *a, **k: (seen.append(a[0]), (dict(new_w), "window ok"))[1])
    bot.maybe_refit_pool(dt.datetime(2026, 10, 4, 7, 0, tzinfo=dt.timezone.utc))  # Sunday: nothing
    assert bot.refit_thread is None
    bot.maybe_refit_pool(dt.datetime(2026, 10, 5, 5, 0, tzinfo=dt.timezone.utc))  # Monday before 06Z: nothing
    assert bot.refit_thread is None
    bot.maybe_refit_pool(dt.datetime(2026, 10, 5, 7, 0, tzinfo=dt.timezone.utc))
    bot.refit_thread.join(timeout=30)
    assert json.load(open(wp))["alpha"] == [1.0] * 7 and json.load(open(str(wp) + ".prev"))["alpha"] == [0.9] * 7
    assert bot.pool.w["alpha"] == [1.0] * 7
    assert seen == [[str(tmp_path / "b.sqlite"), str(tmp_path / "wxbot.sqlite"), str(tmp_path / "live.sqlite")]]
    t = bot.refit_thread
    bot.maybe_refit_pool(dt.datetime(2026, 10, 5, 9, 0, tzinfo=dt.timezone.utc))  # same day: not again
    assert bot.refit_thread is t
    # switching to live the same day (a new database in the same data folder) does not refit again
    live = Bot.__new__(Bot)
    live.store = Store(str(tmp_path / "live.sqlite"))
    live.ml, live.refit_thread, live.rest, live.refit_cfg = object(), None, R(), dict(bot.refit_cfg)
    live.maybe_refit_pool(dt.datetime(2026, 10, 5, 15, 0, tzinfo=dt.timezone.utc))
    assert live.refit_thread is None and (tmp_path / "pool_refit_date.txt").read_text().strip() == "2026-10-05"
    # a week later the live bot refits from its own file and the paper one
    live.maybe_refit_pool(dt.datetime(2026, 10, 12, 7, 0, tzinfo=dt.timezone.utc))
    live.refit_thread.join(timeout=30)
    assert seen[-1] == [str(tmp_path / "live.sqlite"), str(tmp_path / "wxbot.sqlite")]


def test_runner_prefetch_only_between_hours_and_not_during_a_refresh():
    import datetime as dt
    import threading
    from collections import defaultdict
    from wxbot.app.runner import Bot

    calls = []

    class Raw:
        last_hist_day = dt.date(2026, 10, 3)

        def refresh(self, full=False):
            calls.append(full)

    class ML:
        raw = Raw()
    bot = Bot.__new__(Bot)
    bot.ml, bot.cfg, bot.ml_thread, bot.next_due = ML(), {"ml": {}}, None, defaultdict(float)
    bot.maybe_prefetch_ml(dt.datetime(2026, 10, 3, 12, 5, tzinfo=dt.timezone.utc))   # minute 5: decision window
    assert bot.ml_thread is None
    busy = threading.Event()
    bot.ml_thread = threading.Thread(target=busy.wait, daemon=True)
    bot.ml_thread.start()
    bot.maybe_prefetch_ml(dt.datetime(2026, 10, 3, 12, 45, tzinfo=dt.timezone.utc))  # a refresh is still running
    busy.set()
    bot.ml_thread.join()
    assert calls == []
    bot.maybe_prefetch_ml(dt.datetime(2026, 10, 3, 12, 45, tzinfo=dt.timezone.utc))
    bot.ml_thread.join(timeout=5)
    assert len(calls) == 1


def test_liveraw_restart_same_day_skips_full_history(tmp_path):
    import datetime as dt
    from wxbot.ml.live import LiveRaw
    r = LiveRaw(str(tmp_path), ["KMIA"])
    assert r.last_hist_day is None
    today = dt.datetime.now(dt.timezone.utc).date()
    # a marker without the history files it vouches for is ignored
    open(tmp_path / "hist_day.txt", "w").write(today.isoformat())
    assert LiveRaw(str(tmp_path), ["KMIA"]).last_hist_day is None
    open(tmp_path / "obs" / "asos_hist.csv", "w").write("x")
    assert LiveRaw(str(tmp_path), ["KMIA"]).last_hist_day == today
    open(tmp_path / "hist_day.txt", "w").write("garbage")
    assert LiveRaw(str(tmp_path), ["KMIA"]).last_hist_day is None


def test_tier1_parity_check_passes_on_both_engines_and_catches_a_wrong_engine(tmp_path, monkeypatch):
    import sys
    from wxbot.ml.live import Tier1Live
    t = Tier1Live(str(tmp_path), ["KNYC"])
    assert t.parity_max_diff < 1e-9
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    t2 = Tier1Live(str(tmp_path), ["KNYC"])
    assert t2.engine == "numpy" and t2.parity_max_diff < 1e-9
    orig = t2.booster.predict
    monkeypatch.setattr(type(t2.booster), "predict", lambda self, X: orig(X) * 0.99 + 0.0004)
    try:
        t2.parity_check()
        assert False, "a drifting engine must fail the check"
    except RuntimeError as e:
        assert "parity check failed" in str(e)


def test_nam_fallback_switches_on_at_the_retirement_cycle():
    import pandas as pd
    df = pd.DataFrame({"D": pd.to_datetime(["2026-11-03", "2026-11-03", "2026-11-04"]), "h": [9, 15, -3],
                       "nam_n_x": [70.0, 71.0, 72.0], "nam_age_h": [9.0, 15.0, 30.0],
                       "gfs_n_x": [69.0, 68.0, 67.0], "gfs_age_h": [6.0, 9.0, 12.0]})
    g = F.nam_fallback(df, "gfs")
    assert list(g["nam_n_x"]) == [70.0, 68.0, 67.0] and list(g["nam_age_h"]) == [9.0, 13.4, 16.4]
    n = F.nam_fallback(df, "nan")
    assert n["nam_n_x"].isna().tolist() == [False, True, True]
    assert F.nam_fallback(df.iloc[:1], "gfs") is not None and df["nam_n_x"].tolist() == [70.0, 71.0, 72.0]


def test_refit_reads_events_past_the_historical_cutoff_from_the_archive(tmp_path, monkeypatch):
    """Kalshi lists a market settled more than about two months ago only under /historical/markets: the refit's
    lookup falls back to it (needed only if its cache file is lost), and caches what it finds."""
    import datetime as dt
    import gzip
    from wxbot.app import runner
    from wxbot.app.runner import Bot
    from wxbot.ml import pool_fit
    from wxbot.store.db import Store
    old_ms = [{"ticker": "KXHIGHNY-26JUL01-B80.5", "result": "yes"}, {"ticker": "KXHIGHNY-26JUL01-T82", "result": "no"}]
    calls = []

    class Client:
        def __init__(self, **kw):
            pass

        def markets(self, event_ticker=None, limit=100):
            calls.append(("markets", event_ticker))
            return [] if event_ticker.endswith("JUL01") else [{"ticker": event_ticker + "-T1", "result": "no"}]

        def paged(self, path, key, params, **kw):
            calls.append((path, params["event_ticker"]))
            return old_ms if path == "/historical/markets" else []
    monkeypatch.setattr(runner, "KalshiREST", Client)
    got = {}

    def fake_refit(db_path, base, today, get_markets, model_sha=None):
        got["old"] = get_markets("KXHIGHNY-26JUL01")
        got["new"] = get_markets("KXHIGHNY-26OCT01")
        return None, "window ok"
    monkeypatch.setattr(pool_fit, "refit", fake_refit)
    bot = Bot.__new__(Bot)
    bot.store = Store(str(tmp_path / "b.sqlite"))
    bot.ml, bot.refit_thread = object(), None

    class R:
        base = "https://example.invalid/trade-api/v2"
    bot.rest = R()
    bot.refit_cfg = {"enabled": True, "weekday": 0, "after_utc_hour": 6, "weights_path": str(tmp_path / "w.json"),
                     "base_rows_path": str(tmp_path / "none.json.gz")}
    bot.maybe_refit_pool(dt.datetime(2026, 10, 5, 7, 0, tzinfo=dt.timezone.utc))
    bot.refit_thread.join(timeout=30)
    assert got["old"] == old_ms and got["new"][0]["ticker"] == "KXHIGHNY-26OCT01-T1"
    assert ("/historical/markets", "KXHIGHNY-26JUL01") in calls and ("/historical/markets", "KXHIGHNY-26OCT01") not in calls
    cache = json.load(gzip.open(str(tmp_path / "refit_markets_cache.json.gz"), "rt"))
    assert cache["KXHIGHNY-26JUL01"] == old_ms
