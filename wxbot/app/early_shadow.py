"""Candidate B's shadow log (docs/HOLDOUT_DECJAN.md, "B's shadow log"): the frozen taker rule at two earlier decision
hours, h = -9 and -6 (15Z and 18Z the day before), priced by candidate B. B is the 600-tree forecaster trained with
the early hours (artifacts tier1_r2b*, research/ml/train_r2b.py) pooled with the market by B's nine-hour weights
(pool_weights_r2b.json). It logs only: it holds no key and never sends an order.

It runs on the server as its own short process (systemd timer wxbot-early-shadow, 15:05Z and 18:05Z), outside the
live bot, which it never touches. Each run:
  1. Syncs a private copy of the live bot's weather folder (data/ml_raw -> data/ml_raw_early). A file is taken only
     when its size and time stamp are the same before and after the copy, so a file the live bot is rewriting is
     never read half written. HRRR run files are merged instead: the private copy holds forecast hours the live bot
     does not fetch. Nothing comes from IEM, so the live bot's IEM allowance is untouched.
  2. Fetches every forecast hour of the HRRR runs the decision uses, as training saw them (NOAA's bucket): at 15Z the
     12Z run (48 hours) and the 06Z run before it (for the trend); at 18Z the 15Z run (18 hours) and the 12Z run.
  3. Lists tomorrow's events in Sleeve X's 20 cities and reads their books (Kalshi's public endpoints).
  4. Forecasts at the decision hour with the code training and the live bot use (wxbot.ml.features, wxbot.ml.hrrr;
     the NAM fallback included), pools with the market and logs to data/early_shadow.sqlite: one ml_snapshots row
     per bracket and the frozen rule's signals in taker_shadow (wxbot/strategy/taker.py: liquid books, beat the fee
     plus 6c, 5 contracts capped by the displayed size, first signal per event, bracket and side over both hours).
  5. Records Kalshi's result for logged trades whose markets have settled.
A run more than 30 minutes after its decision time logs nothing (as Sleeve X). No P&L is computed or shown: B's
shadow is scored once, on Feb 1, 2027 (item B3), and `status` gives counts only.

Shadow sizing (outside review, Oct 8): each logged signal also records its intended size, the depth seen (contracts
offered at its price or better in the decision's book) and the booked size, min(intended, depth), 0 when the depth
is zero or unknown; reports sum the booked size. `size` keeps its old meaning, since Sleeve E sends it.
Execution log (off unless --config names a config whose exec_log.enabled is true; wxbot/execution/exec_log.py): the
decision's book for each logged signal, then the same book re-read at +1, +5 and +15 minutes, in table exec_log of
the candidate's database. The run waits for the re-reads after logging (its models freed), ending them before the
systemd unit's time limit (EXEC_DEADLINE_S).

Usage: python -m wxbot.app.early_shadow run [--dry-run [--at ISO_UTC]] [--config CONFIG] | settle | status
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import gzip
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..execution.exec_log import ExecLogConfig, ExecLogger
from ..kalshi.rest import PROD, KalshiREST
from ..ml import aifs_ens as A
from ..ml import features as F
from ..ml import hrrr as H
from ..ml.live import ART, Pool, bracket_pmf
from ..models.brackets import bracket_from_market
from ..models.fair_value import book_to_quote, ladder_fair_value
from ..store.db import Store
from ..strategy import taker
from .runner import STATIONS, parse_event_date

log = logging.getLogger("wxbot.early")

# Sleeve X's 20 cities (config/live_go.yaml: sleeve_a.series + taker_shadow.extra_series)
SERIES = ["KXHIGHAUS", "KXHIGHCHI", "KXHIGHDEN", "KXHIGHLAX", "KXHIGHMIA", "KXHIGHNY", "KXHIGHPHIL", "KXHIGHTATL",
          "KXHIGHTBOS", "KXHIGHTDAL", "KXHIGHTDC", "KXHIGHTHOU", "KXHIGHTLV", "KXHIGHTNOLA", "KXHIGHTOKC",
          "KXHIGHTPHX", "KXHIGHTSATX", "KXHIGHTSEA", "KXHIGHTSFO", "KXHIGHTMIN"]
SLOTS = {15: -9, 18: -6}                         # decision hour (UTC, the day before) -> h for the next day's events
RUNS = {15: ((12, 48), (6, 48)), 18: ((15, 18), (12, 48))}   # HRRR runs the decision picks: (init hour, hours)
WINDOW_S = 1800                                  # log only within 30 minutes of the decision time (as Sleeve X)
MODEL, POOL_FILE = "tier1_r2b", "pool_weights_r2b.json"
RULE = taker.TakerConfig(enabled=True, margin=0.06, size=5.0, max_spread=0.10, msum_lo=0.90, msum_hi=1.25)
COORDS = os.path.join(os.path.dirname(__file__), "..", "ml", "station_coords.json")

EXEC_DEADLINE_S = 1380      # execution log: re-reads end this long after the run starts (the systemd unit stops a run
                            # after TimeoutStartSec = 1500 s); later ones are skipped and counted

REFIT_WEEKDAY, REFIT_AFTER_UTC = 0, 6             # B's two hours refit on Mondays (the live bot's schedule)
REFIT_WINDOW_DAYS, REFIT_MIN_ROWS = 90, 1000
LIVE_POOL = "pool_weights_r2b_live.json"          # in data/: the artifact's weights with the early hours refit
BASE_ROWS = "pool_rows_base_r2b.json.gz"          # research rows at h = -9 / -6, Jul 1 - Sep 30, 2026



@dataclass(frozen=True)
class Spec:
    """One candidate the job logs: its model, pool weights, research rows for the refit, database and refit file."""
    tag: str
    model: str
    pool_file: str
    base_rows: str
    db: str
    live_pool: str


SPECS = (Spec("B", MODEL, POOL_FILE, BASE_ROWS, "early_shadow.sqlite", LIVE_POOL),
         # candidate C (docs/HOLDOUT_DECJAN.md amendment C1): B plus ECMWF AIFS ENS inputs, read live
         Spec("C", "tier1_r2c", "pool_weights_r2c.json", "pool_rows_base_r2c.json.gz", "early_shadow_c.sqlite",
              "pool_weights_r2c_live.json"),
         # G (round 5, Oct 8): C's forecaster with the pool's NBM-departure tilt (research/ml/r5_pool_gate_early.py:
         # h = -9 / -6 per event -0.0101 [-0.0158, -0.0045] Jun-Sep, -0.0160 [-0.0283, -0.0034] Mar-May). C's own
         # registered shadow is unchanged; G is the version Sleeve E trades once its checks pass (Stage 3)
         Spec("G", "tier1_r2c", "pool_weights_r2cg.json", "pool_rows_base_r2cg.json.gz", "early_shadow_g.sqlite",
              "pool_weights_r2cg_live.json"))

RUNS_SQL = """CREATE TABLE IF NOT EXISTS early_runs (
  ts REAL, decision_ts REAL, h INTEGER, day TEXT, events INTEGER, liquid INTEGER, signals INTEGER, model_sha TEXT,
  pool_sha TEXT, engine TEXT, hrrr_ok INTEGER, notes TEXT)"""


# ------------------------------------------------------------------------------------------------------ the model
class EarlyModel:
    """B's forecaster and pool. The model text must match its recorded hash, and the engine in use (LightGBM, or the
    numpy tree evaluator) must reproduce LightGBM's probabilities on the stored parity rows, or nothing is logged."""

    def __init__(self, art: str = ART, name: str = MODEL, pool_file: str = POOL_FILE, pool_path: Optional[str] = None):
        self.meta = json.load(open(os.path.join(art, f"{name}_meta.json")))
        txt = gzip.decompress(open(os.path.join(art, f"{name}.txt.gz"), "rb").read()).decode()
        if hashlib.sha256(txt.encode()).hexdigest() != self.meta["sha256"]:
            raise RuntimeError(f"{name}.txt.gz does not match the hash in its meta file")
        try:
            import lightgbm as lgb
            self.booster = lgb.Booster(model_str=txt)
            self.engine = "lightgbm"
        except (ImportError, OSError):
            from ..ml.npgbm import NumpyBooster
            self.booster = NumpyBooster(txt)
            self.engine = "numpy"
        self.sha = self.meta["sha256"][:12]
        self.uses_aifs = any(f.startswith("aifs_") for f in self.meta["feats"])
        # round 5 (C with D's inputs): anchor on the guidance median, NBM's departure from it, the hourly NBM run
        self.anchor_mode = self.meta.get("anchor_mode", "nbm")
        self.round5 = bool(self.meta.get("round5", False))
        self.uses_nbf = any(f.startswith("nbf_") for f in self.meta["feats"])
        pool_path = pool_path or os.path.join(art, pool_file)    # data/pool_weights_r2b_live.json after a refit
        self.pool = Pool(pool_path)
        self.pool_sha = hashlib.sha256(open(pool_path, "rb").read()).hexdigest()[:12]
        if not all(h in self.pool.w["hours"] for h in SLOTS.values()):
            raise RuntimeError(f"{pool_file} has no weights for the early hours")
        self.parity = self.parity_check(os.path.join(art, f"{name}_parity.json.gz"))

    def parity_check(self, path: str, tol: float = 1e-6) -> float:
        fx = json.loads(gzip.decompress(open(path, "rb").read()).decode())
        if fx.get("model_sha256") != self.meta["sha256"]:
            raise RuntimeError("the parity rows are for another model")
        X = pd.DataFrame([[np.nan if v is None else v for v in r] for r in fx["X"]], columns=fx["num"])
        X["st"] = pd.Categorical(fx["st"], categories=self.meta["cats"])
        d = float(np.abs(np.asarray(self.booster.predict(X[fx["columns"]])) - np.asarray(fx["P"])).max())
        if not d <= tol:
            raise RuntimeError(f"parity check failed on the {self.engine} engine: max difference {d:.2e}")
        return d

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Integer PMFs (rows x NCLS) and the rounded anchors, exactly as Tier1Live.predict."""
        x = H.add_model_columns(F.model_features(df, anchor_mode=getattr(self, "anchor_mode", "nbm")), df)
        if self.uses_aifs:
            x = A.add_model_columns(x, df)
        if getattr(self, "round5", False):
            x = F.add_round5_columns(x, df)
        if getattr(self, "uses_nbf", False):
            x = F.add_nbm_fresh_columns(x, df)
        x["st"] = pd.Categorical(df["icao"], categories=self.meta["cats"])
        p = np.asarray(self.booster.predict(x[self.meta["feats"]]))
        full = np.zeros((len(df), F.NCLS))
        full[:, self.meta["classes"]] = p
        full = np.clip(full, 1e-6, None)
        full /= full.sum(axis=1, keepdims=True)
        return full, x["anchor_r"].to_numpy(float)


# --------------------------------------------------------------------------------------------- the private folder
def _sig(path: str) -> tuple[int, int]:
    s = os.stat(path)
    return s.st_size, s.st_mtime_ns


def copy_stable(src: str, dst: str, tries: int = 3, wait_s: float = 1.0) -> str:
    """Copy src to dst unless dst already has src's size and time stamp. The copy is kept only if src's size and time
    stamp are the same before and after it (otherwise src was being written: retry, then keep the old copy).
    Returns 'same', 'copied' or 'busy'."""
    for k in range(tries):
        before = _sig(src)
        if os.path.exists(dst) and _sig(dst) == before:
            return "same"
        tmp = dst + ".part"
        shutil.copy2(src, tmp)
        if _sig(src) == before and os.path.getsize(tmp) == before[0]:
            os.replace(tmp, dst)
            return "copied"
        os.remove(tmp)
        time.sleep(wait_s)
    return "busy"


def merge_hrrr(src: str, dst: str) -> str:
    """Union of the live bot's and the private copy's forecast hours of one HRRR run (the live bot writes these files
    whole, through a temporary file, so a read never sees half a file). Returns 'same', 'merged' or 'copied'."""
    a = pd.read_csv(src)
    if not os.path.exists(dst):
        a.to_csv(dst + ".part", index=False)
        os.replace(dst + ".part", dst)
        return "copied"
    b = pd.read_csv(dst)
    u = pd.concat([b, a], ignore_index=True).drop_duplicates(["icao", "fh"], keep="first")
    if len(u) == len(b):
        return "same"
    u.sort_values(["icao", "fh"]).to_csv(dst + ".part", index=False)
    os.replace(dst + ".part", dst)
    return "merged"


def sync_raw(src: str, dst: str) -> dict:
    """Bring the private folder up to the live bot's (see copy_stable / merge_hrrr). Counts by outcome."""
    out = {"same": 0, "copied": 0, "merged": 0, "busy": 0}
    for dirpath, _dirs, files in os.walk(src):
        rel = os.path.relpath(dirpath, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        top = rel.split(os.sep)[0]
        for fn in files:
            if fn.endswith((".tmp", ".part")):
                continue
            s, d = os.path.join(dirpath, fn), os.path.join(dst, rel, fn)
            try:
                r = merge_hrrr(s, d) if top == "hrrr" and fn.endswith(".csv") else copy_stable(s, d)
            except (OSError, ValueError, pd.errors.ParserError) as e:   # a file vanished or was replaced mid-read
                log.warning("sync %s: %s", os.path.join(rel, fn), e)
                r = "busy"
            out[r] += 1
    return out


def run_complete(root: str, day: dt.date, hh: int, nfh: int, stations: list[str]) -> bool:
    path = os.path.join(root, "hrrr", f"{day:%Y%m%d}_{hh:02d}z.csv")
    if not os.path.exists(path):
        return False
    d = pd.read_csv(path, usecols=["icao", "fh"])
    have = d[d["fh"].between(1, nfh)].groupby("icao")["fh"].nunique()
    return all(int(have.get(s, 0)) == nfh for s in stations)


def fetch_hrrr(root: str, dec: dt.datetime, fetch: Callable = H.fetch_run, tries: int = 3,
               wait_s: float = 60.0, chunk: int = 4) -> bool:
    """Every forecast hour of the runs this decision picks (compute() takes the latest run at least 3 hours old, and
    the synoptic run before it for the trend), so the inputs match what training saw. A few hours per call:
    fetch_run keeps each decoded CONUS field (about 15 MB) until its call ends, so 28 hours at once would add
    about 420 MB."""
    coords = json.load(open(COORDS))
    stations = sorted(coords)
    ok = True
    for hh, nfh in RUNS[dec.hour]:
        for k in range(tries):
            for i in range(1, nfh + 1, chunk):
                try:
                    fetch(dec.date(), hh, range(i, min(i + chunk, nfh + 1)), root, coords)
                except Exception as e:
                    log.warning("HRRR %02dZ fetch: %s", hh, e)
            if run_complete(root, dec.date(), hh, nfh, stations):
                break
            if k + 1 < tries:
                time.sleep(wait_s)
        else:
            ok = False
            log.warning("HRRR %s %02dZ run incomplete after %d tries", dec.date(), hh, tries)
    return ok


# ------------------------------------------------------------------------------------------------------- features
def build_rows(raw_root: str, targets: list[tuple[str, dt.date]], h: int, today: dt.date) -> pd.DataFrame:
    """Tier-1 rows for (icao, D) at decision hour h, built as Tier1Live.predict builds them (the past 44 days and
    today at the 03Z decision as history for the lagged errors), with the HRRR columns from the private folder."""
    src = F.RawSource(raw_root)
    parts = []
    for icao in sorted({i for i, _ in targets}):
        lst = [(pd.Timestamp(D), h) for i, D in targets if i == icao]
        hist_days = pd.date_range(today - dt.timedelta(days=44), today)
        frame = pd.DataFrame([(d, 3) for d in hist_days] + lst, columns=["D", "h"]).drop_duplicates()
        r = F.station_rows(src, icao, frame)
        r["_target"] = r.set_index(["D", "h"]).index.isin(pd.MultiIndex.from_tuples(lst))
        parts.append(r)
    df = pd.concat(parts, ignore_index=True)
    W = H.load_runs(raw_root, since=pd.Timestamp(today) - pd.Timedelta(days=20))
    hf = H.compute(df, W)
    df = pd.concat([df.drop(columns=[c for c in hf.columns if c in df]), hf], axis=1)
    hist = df[~df["_target"]]
    df = F.add_lag_errors(df[df["_target"]].drop(columns=["_target"]), history=hist)
    return H.add_lag_errors(df, history=hist)


# ------------------------------------------------------------------------------------------- B's weekly refit
def fit_hours(rows: list, hours: list, gate: Optional[str] = None) -> dict:
    """BTh weights per decision hour (wxbot/ml/pool_fit.fit_bth, the same model and penalty, for any set of hours).
    Each hour's weights depend only on its own rows (a nine-hour fit and a fit on h = -9 / -6 alone agree to 1e-4
    on Jul-Sep 2026), so refitting B's two hours alone is the registered nine-hour procedure for them.
    gate="nbmdep" (G): one more column shared by these hours, NBM's departure from the market (pool_fit.gate_column);
    fit on the early hours' rows alone, it serves only them."""
    from scipy.optimize import minimize
    from ..ml.pool_fit import gate_column, ladder_arrays
    rows = [r for r in rows if int(r["h"]) in hours]
    LM, LF, Z, MASK, WIN = ladder_arrays(rows)
    X = np.stack([np.array([int(r["h"]) == h for r in rows], float) for h in hours], axis=1)
    if gate:
        X = np.hstack([X, gate_column(rows)[:, None]])
    B = [np.where(MASK, b, 0.0) for b in (LM, LF, Z, Z * Z)]
    n, d, nb = X.shape[0], X.shape[1], 4
    w0 = np.zeros(nb * d)
    w0[:len(hours)] = 1.0
    Y = np.zeros(MASK.shape)
    Y[np.arange(n), WIN] = 1.0

    def f(w):
        sc = np.zeros(MASK.shape)
        for k in range(nb):
            sc += (X @ w[k * d:(k + 1) * d])[:, None] * B[k]
        sc = np.where(MASK, sc, -np.inf)
        smax = sc.max(axis=1, keepdims=True)
        e = np.exp(sc - smax)
        se = e.sum(axis=1, keepdims=True)
        ll = sc[np.arange(n), WIN] - (np.log(se[:, 0]) + smax[:, 0])
        R = e / se - Y
        g = np.concatenate([X.T @ (R * B[k]).sum(axis=1) / n for k in range(nb)])
        dw = w - w0
        return -float(ll.mean()) + 0.5 * float((dw * dw).sum()) / n, g + dw / n

    res = minimize(f, w0, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    W = res.x.reshape(nb, d)
    k = len(hours)
    out = {"hours": list(hours), "alpha": W[0, :k].round(4).tolist(), "beta": W[1, :k].round(4).tolist(),
           "theta": W[2, :k].round(4).tolist(), "kappa": W[3, :k].round(4).tolist(), "n_snapshots": int(n),
           "by_hour": {str(h): int(X[:, j].sum()) for j, h in enumerate(hours)}, "converged": bool(res.success)}
    if gate:
        out["gate"] = {"cols": [gate], **{name: [round(float(W[i, k]), 4)] for i, name in
                                          enumerate(("alpha", "beta", "theta", "kappa"))}, "clip_f": 4.0,
                       "rows_with_nbm": int(sum(1 for r in rows if r.get("nbs") is not None))}
    return out


def sane_early(out: dict, hours: list) -> bool:
    """wxbot/ml/pool_fit.sane's bounds (every research fit fell inside them) for the given hours."""
    try:
        ok = (bool(out.get("converged")) and out["hours"] == list(hours)
              and all(0.5 <= a <= 1.6 for a in out["alpha"]) and all(-0.1 <= b <= 1.0 for b in out["beta"])
              and all(abs(t) <= 1.0 for t in out["theta"]) and all(abs(k) <= 1.0 for k in out["kappa"]))
        if ok and out.get("gate"):
            ok = all(abs(float(out["gate"][n][0])) <= 1.5 for n in ("alpha", "beta", "theta", "kappa"))
        return ok
    except (KeyError, TypeError, IndexError):
        return False


def refit_early(db_path: str, base_rows: list, today: dt.date, get_markets: Callable, model_sha: Optional[str],
                window_days: int = REFIT_WINDOW_DAYS, min_rows: int = REFIT_MIN_ROWS, gate: Optional[str] = None):
    """The production procedure for B's two hours: the window_days days ending two days before today, the shadow's
    own snapshots (model_sha's) from its first day on, research rows before. Returns (weights or None, message)."""
    from ..ml.pool_fit import live_rows
    hours = sorted(SLOTS.values())
    end = today - dt.timedelta(days=2)
    start = end - dt.timedelta(days=window_days - 1)
    live = [r for r in live_rows(db_path, start, end, get_markets, model_sha) if int(r["h"]) in hours]
    first_live = min((r["D"] for r in live), default=None)
    rows = [r for r in base_rows if int(r["h"]) in hours and start.isoformat() <= r["D"] <= end.isoformat()
            and (first_live is None or r["D"] < first_live)] + live
    msg = f"window {start}..{end}: {len(rows)} snapshots ({len(rows) - len(live)} research, {len(live)} shadow)"
    if len(rows) < min_rows:
        return None, msg + "; too few, weights unchanged"
    out = fit_hours(rows, hours, gate=gate)
    out.update({"first_day": min(r["D"] for r in rows), "last_day": max(r["D"] for r in rows),
                "window_days": window_days, "sources": {"research": len(rows) - len(live), "shadow": len(live)},
                "refit_utc": _utcnow().isoformat(timespec="seconds")})
    if not sane_early(out, hours):
        return None, msg + f"; outside the sanity bounds (alpha {out['alpha']}, beta {out['beta']}), weights unchanged"
    return out, msg


def merge_weights(current: dict, early: dict) -> dict:
    """The current nine-hour weights with the early hours' entries replaced by the refit's."""
    w = json.loads(json.dumps(current))
    for j, h in enumerate(early["hours"]):
        k = w["hours"].index(h)
        for name in ("alpha", "beta", "theta", "kappa"):
            w[name][k] = early[name][j]
    if early.get("gate"):          # G: the early hours' own tilt (this pool file serves only those hours)
        w["gate"] = early["gate"]
    w["early_refit"] = {k: early[k] for k in ("first_day", "last_day", "n_snapshots", "by_hour", "sources",
                                                "refit_utc") if k in early}
    return w


def market_lookup(rest, cache_path: str):
    """get_markets(event) for the refit, caching settled events (never change again) in a gzip JSON file."""
    try:
        cache = json.load(gzip.open(cache_path, "rt"))
    except Exception:
        cache = {}

    def get(ev):
        if ev in cache:
            return cache[ev]
        ms = rest.markets(event_ticker=ev, limit=100)
        if not ms:      # settled before Kalshi's historical cutoff (about two months back)
            ms = rest.paged("/historical/markets", "markets", {"event_ticker": ev, "limit": 100})
        if ms and all(m.get("result") in ("yes", "no") for m in ms):
            cache[ev] = ms
        return ms

    def save():
        try:
            with gzip.open(cache_path + ".tmp", "wt") as fh:
                json.dump(cache, fh)
            os.replace(cache_path + ".tmp", cache_path)
        except OSError as e:
            log.warning("refit cache: %s", e)
    return get, save


def maybe_refit(store: Store, data: str, rest, now: dt.datetime, art: str = ART, spec: "Spec" = None) -> Optional[str]:
    """Mondays after 06Z, once: refit the candidate's two hours and write data/<its live pool file> (atomically, the
    old file kept as .prev). The attempt is recorded first, so a failure is not retried until the next Monday."""
    spec = spec or SPECS[0]
    if now.weekday() != REFIT_WEEKDAY or now.hour < REFIT_AFTER_UTC:
        return None
    if store.kv_get("early_refit_date") == now.date().isoformat():
        return None
    store.kv_set("early_refit_date", now.date().isoformat())
    live_path = os.path.join(data, spec.live_pool)
    try:
        meta = json.load(open(os.path.join(art, f"{spec.model}_meta.json")))
        bp = os.path.join(art, spec.base_rows)
        base = json.load(gzip.open(bp, "rt")) if os.path.exists(bp) else []
        get, save = market_lookup(rest, os.path.join(data, "early_refit_markets_cache.json.gz"))
        current = json.load(open(live_path if os.path.exists(live_path) else os.path.join(art, spec.pool_file)))
        gate = ((current.get("gate") or {}).get("cols") or [None])[0]
        out, msg = refit_early(store.path, base, now.date(), get, meta["sha256"][:12], gate=gate)
        save()
        if out is not None:
            from ..ml.pool_fit import write_weights
            write_weights(merge_weights(current, out), live_path)
            msg += f"; new early-hour weights (alpha {out['alpha']}, beta {out['beta']})"
    except Exception as e:
        msg = f"refit failed, weights unchanged: {e}"
    store.log("INFO", f"{spec.tag} early-hour refit: " + msg)
    log.info("%s early-hour refit: %s", spec.tag, msg)
    return msg


# ------------------------------------------------------------------------------------------------------- the run
def decision_slot(now: dt.datetime) -> Optional[tuple[dt.datetime, int]]:
    """(decision time, h) when `now` is within WINDOW_S after 15Z or 18Z, else None."""
    for hh, h in SLOTS.items():
        t = now.replace(hour=hh, minute=0, second=0, microsecond=0)
        if 0 <= (now - t).total_seconds() <= WINDOW_S:
            return t, h
    return None


def supported(e: dict) -> bool:
    """The runner's unsupported_event rules: whole cents, exchange shard 0, a 0/$1 payout."""
    if int(e.get("exchange_index") or 0) != 0:
        return False
    for m in e.get("markets", []):
        whole = bool(m.get("price_ranges")) and all(abs(float(r.get("step") or 0) - 0.01) < 1e-9
                                                    for r in m.get("price_ranges") or [])
        if (m.get("price_level_structure") or "linear_cent") != "linear_cent" and not whole:
            return False
        if int(m.get("exchange_index") or 0) != 0 or (m.get("settlement_bounds_type") or "default") != "default":
            return False
    return True


def list_events(rest, day: dt.date) -> list[dict]:
    """Open events for `day` in the 20 cities: {event, series, icao, brackets, tickers}."""
    out = []
    for s in SERIES:
        try:
            evs = rest.events(s, status="open", nested=True, limit=10)
        except Exception as e:
            log.warning("events %s: %s", s, e)
            continue
        for e in evs:
            ms = [m for m in e.get("markets", []) if m.get("status") in ("active", "open")]
            if not ms or parse_event_date(e["event_ticker"])[0] != day or not supported(e):
                continue
            try:
                brs = [bracket_from_market(m) for m in ms]
            except ValueError:
                continue
            out.append({"event": e["event_ticker"], "series": s, "icao": STATIONS[s]["icao"], "brackets": brs})
    return out


def read_books(rest, tickers: list[str]) -> dict:
    """Order books for the tickers: the batch endpoint, or one market at a time if the batch is refused."""
    if not tickers:
        return {}
    try:
        return rest.orderbooks(tickers)
    except Exception as e:
        log.warning("batch order books refused (%s): reading them one by one", str(e)[:120])
    out = {}
    for t in tickers:
        try:
            out[t] = rest.orderbook(t)
        except Exception as e:
            log.warning("order book %s: %s", t, str(e)[:120])
    return out


def settle(store: Store, rest, now_ts: float, limit: int = 60) -> int:
    """Kalshi's result for logged trades (markets at least 6 hours past their log time, oldest first)."""
    n = 0
    for t in store.taker_shadow_open(older_than=now_ts - 6 * 3600)[:limit]:
        try:
            m = rest.market(t)
        except Exception:
            continue
        if m.get("result") in ("yes", "no"):
            store.set_taker_result(t, m["result"], now_ts)
            n += 1
    return n


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def available_specs(art: str = ART) -> list:
    return [sp for sp in SPECS if os.path.exists(os.path.join(art, f"{sp.model}_meta.json"))]


def add_nbf(df: pd.DataFrame, root: str, dec: dt.datetime) -> pd.DataFrame:
    """The freshest hourly NBM run public at the decision for each row's day (wxbot/ml/nbm_hourly.py, as
    Tier1Live._add_nbm_fresh); a failed fetch leaves the columns empty, as for rows before the training archive."""
    from ..ml import nbm_hourly as NH
    dec_n = pd.Timestamp(dec).tz_convert("UTC").tz_localize(None) if pd.Timestamp(dec).tzinfo else pd.Timestamp(dec)
    try:
        need = {(i, pd.Timestamp(d)) for i, d in zip(df["icao"], df["D"])}
        fresh = NH.latest(dec_n, sorted(F.HIGH), os.path.join(root, "nbm_hourly"), need=need)   # cache keeps all
    except Exception as e:
        log.warning("hourly NBM unavailable: %s", e)
        fresh = None
    cols = NH.columns(df, fresh)
    return pd.concat([df.drop(columns=[c for c in cols.columns if c in df]), cols], axis=1)


def add_aifs(df: pd.DataFrame, root: str, dec: dt.datetime) -> tuple[pd.DataFrame, bool]:
    """Candidate C's AIFS ENS inputs for the rows: the run public at the decision (A.run_for), the steps the rows'
    local days need, fetched into the private folder (reused at 18Z: same run). Returns (rows, all inputs present)."""
    coords = json.load(open(COORDS))
    dec_n = pd.Timestamp(dec).tz_convert("UTC").tz_localize(None) if pd.Timestamp(dec).tzinfo else pd.Timestamp(dec)
    init = A.run_for(dec_n)
    targets = [(pd.Timestamp(D), pd.Timestamp(D) + pd.Timedelta(hours=int(h)), int(off))
               for D, h, off in zip(df["D"], df["h"], df["lst_off"])]
    run = A.fetch_run(init, A.steps_needed(init, targets), root, coords)
    f = A.features(df[["icao", "D", "h", "lst_off"]], run, init)
    out = pd.concat([df.drop(columns=[c for c in f.columns if c in df]), f], axis=1)
    return out, bool(f["aifs_mx"].notna().all())


def run(base: str, now: Optional[dt.datetime] = None, dry_run: bool = False, rest=None, model=None,
        fetch: Callable = H.fetch_run, hrrr_wait_s: float = 60.0, models: Optional[dict] = None,
        art: str = ART, exec_log: Optional[dict] = None, sleep: Callable = time.sleep,
        clock: Callable = time.time) -> dict:
    """One scheduled run (see the module docstring), for every candidate whose artifacts are present (B, and C from
    amendment C1). dry_run computes everything and logs nothing (it still refreshes the private weather folder).
    model / models: injected models ({tag: model}; model= is candidate B's) instead of loading the artifacts.
    exec_log: a config's exec_log section (the execution log, off by default; never in a dry run); sleep / clock:
    its waits and time."""
    t_start = clock()
    real_now = _utcnow()
    now = now or real_now
    if not dry_run and abs((now - real_now).total_seconds()) > 60:
        raise SystemExit("--at is only for --dry-run: a real run logs at the real time")
    data = os.path.join(base, "data")
    rest = rest or KalshiREST(base_url=PROD, auth=None, read_rate=20.0)
    xcfg = ExecLogConfig.from_cfg(exec_log)
    xl = ExecLogger(rest, xcfg, mode="shadow", clock=clock) if xcfg.enabled and not dry_run else None
    if model is not None:
        models = {"B": model}
    specs = [sp for sp in SPECS if sp.tag in models] if models else available_specs(art)
    stores = {}
    for sp in specs:
        if not dry_run:
            stores[sp.tag] = Store(os.path.join(data, sp.db))
            stores[sp.tag].c.execute(RUNS_SQL)
    settled = {sp.tag: settle(stores[sp.tag], rest, now.timestamp()) for sp in specs if sp.tag in stores}
    summary = {"ts": now.timestamp(), "settled": settled.get("B", 0), "settled_by_model": settled}
    slot = decision_slot(now)
    if slot is None:
        summary["note"] = "not a decision slot: settled only"
        return summary
    dec, h = slot
    day = (dec - dt.timedelta(hours=h)).date()           # dec = D 00Z + h hours
    priv = os.path.join(data, "ml_raw_early")
    summary.update(h=h, day=str(day), decision=dec.isoformat(), sync=sync_raw(os.path.join(data, "ml_raw"), priv))
    summary["hrrr_ok"] = fetch_hrrr(priv, dec, fetch=fetch, wait_s=hrrr_wait_s)
    evs = list_events(rest, day)
    books = read_books(rest, [b.ticker for e in evs for b in e["brackets"]])
    t_books = clock()
    df = build_rows(priv, sorted({(e["icao"], day) for e in evs}), h, dec.date()) if evs else None
    df_aifs, aifs_ok, df_nbf = None, None, None
    summary["models"] = {}
    loaded = {}                                              # model name -> loaded model, while a later spec needs it
    for sp in specs:
        store = stores.get(sp.tag)
        if store is not None and h == -9:
            summary.setdefault("refit", {})[sp.tag] = maybe_refit(store, data, rest, now, art=art, spec=sp)
        if models and sp.tag in models:
            m = models[sp.tag]
        else:
            live_pool = os.path.join(data, sp.live_pool)
            pool_path = live_pool if os.path.exists(live_pool) else None
            prev = loaded.get(sp.model)
            if prev is not None:            # G shares C's forecaster: same model, its own pool (no second load)
                import copy
                m = copy.copy(prev)
                pp = pool_path or os.path.join(art, sp.pool_file)
                m.pool = Pool(pp)
                m.pool_sha = hashlib.sha256(open(pp, "rb").read()).hexdigest()[:12]
            else:
                m = EarlyModel(art=art, name=sp.model, pool_file=sp.pool_file, pool_path=pool_path)
            if any(o.model == sp.model for o in specs[specs.index(sp) + 1:]):
                loaded[sp.model] = m
        notes = []
        rows = df
        if df is not None and getattr(m, "uses_nbf", False):
            if df_nbf is None:
                df_nbf = add_nbf(df, priv, dec)
            rows = df_nbf
        if rows is not None and getattr(m, "uses_aifs", False):
            if df_aifs is None:
                try:
                    df_aifs, aifs_ok = add_aifs(df, priv, dec)
                except Exception as e:                  # no AIFS this run: C sees missing inputs, as in training
                    log.warning("AIFS ENS inputs unavailable: %s", e)
                    df_aifs, aifs_ok = df, False
            rows = df_aifs if not getattr(m, "uses_nbf", False) else df_aifs.assign(
                **{c: df_nbf[c] for c in ("nbf_txn", "nbf_xnd", "nbf_age_h") if c in df_nbf})
            if not aifs_ok:
                notes.append("AIFS ENS inputs missing on some rows")
        res = price_and_log(sp, m, store, evs, books, rows, h, now, notes)
        res.update(engine=m.engine, model=m.sha, pool=m.pool_sha)
        summary["models"][sp.tag] = res
        if store is not None:
            store.add_ml_snapshots(res.pop("_snaps"))
            sigs = res.pop("_sigs")
            store.add_taker_shadow(sigs)
            store.add_ml_ctx(res.pop("_ctx"))
            store.c.execute("INSERT INTO early_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                            (now.timestamp(), dec.timestamp(), h, str(day), res["events"], res["liquid"],
                             res["signals"], m.sha, m.pool_sha, m.engine, int(summary["hrrr_ok"]), "; ".join(notes)))
            store.c.commit()
            if xl is not None:
                try:   # each new signal's decision book (the snapshot it was priced on), then its re-reads
                    for r in sigs:
                        xl.record(store, "decision", f"{sp.tag}:{r[1]}:{r[2]}:{r[3]}", sp.tag, r[1], r[2], r[3], r[4],
                                  books.get(r[2]), t_books)
                    xl.flush()                          # no orders here: written at once
                except Exception as e:                  # the shadow log above stands; only its book rows are lost
                    log.warning("%s execution log failed: %s", sp.tag, e)
        else:
            res.pop("_snaps"), res.pop("_sigs"), res.pop("_ctx")
        if not (models and sp.tag in models) and sp.model not in loaded:
            del m
            gc.collect()
        if not any(o.model == sp.model for o in specs[specs.index(sp) + 1:]):
            loaded.pop(sp.model, None)
    first = summary["models"].get("B") or next(iter(summary["models"].values()), {})
    summary.update({k: first.get(k) for k in ("events", "liquid", "signals", "snapshots", "notes", "engine", "model",
                                               "pool")})
    if xl is not None:      # the models are freed: wait for the re-reads (+1, +5, +15 min), within the unit's limit
        try:
            summary["exec_log"] = xl.drain(deadline=t_start + EXEC_DEADLINE_S, sleep=sleep)
        except Exception as e:
            log.warning("execution log re-reads failed: %s", e)
            summary["exec_log"] = {"error": str(e)[:200]}
    return summary


def price_and_log(sp: "Spec", m, store, evs: list, books: dict, df, h: int, now: dt.datetime, notes: list) -> dict:
    """Pool one candidate's forecasts with the market and apply the frozen rule. Returns counts plus the rows to log."""
    sig_rows, snap_rows, ctx_rows, n_liquid = [], [], [], 0
    if evs and df is not None:
        P, ar = m.predict(df)
        fc = {r.icao: i for i, r in enumerate(df.itertuples())}
        age = pd.to_numeric(df["hrrr_age_h"], errors="coerce")
        if not (age == 3).all():
            notes.append(f"HRRR age not 3 h on {int((age != 3).sum())} of {len(df)} rows")
        seen = store.taker_shadow_keys() if store is not None else set()
        ts = now.timestamp()
        for e in evs:
            i = fc.get(e["icao"])
            quotes = {b.ticker: book_to_quote(books.get(b.ticker) or {}) for b in e["brackets"]}
            fv = ladder_fair_value(e["brackets"], quotes, kind="high")
            if i is None or not fv.market:
                continue
            f = bracket_pmf(e["brackets"], P[i], ar[i])
            nbs = df["nbs_txn"].iloc[i] if "nbs_txn" in df else None
            nbs = float(nbs) if nbs is not None and pd.notna(nbs) else None
            probs = m.pool.probs(e["brackets"], fv.market, f, h, nbs=nbs)
            snap_rows += [(ts, e["event"], h, b.ticker, fv.market.get(b.ticker), float(fb), probs.get(b.ticker),
                           m.sha) for b, fb in zip(e["brackets"], f)]
            ctx_rows.append((e["event"], int(h), nbs, ts))
            liquid, spr, msum, sig = taker.signals(quotes, probs, RULE, books=books)
            n_liquid += int(liquid)
            for x in sig:
                k = (e["event"], x["ticker"], x["side"])
                if k in seen:
                    continue                                 # first signal per bracket and side, over both hours
                seen.add(k)
                sig_rows.append((ts, e["event"], x["ticker"], x["side"], x["price"], x["size"], x["avail"], x["p"],
                                 x["edge"], int(h), m.sha, spr, msum, x["intended"], x["depth"], x["booked"]))
    return {"events": len(evs), "liquid": n_liquid, "signals": len(sig_rows), "snapshots": len(snap_rows),
            "notes": notes, "_snaps": snap_rows, "_sigs": sig_rows, "_ctx": ctx_rows}


def status(base: str, tag: str = "B") -> dict:
    """Counts only (no P&L before Feb 1, 2027): runs, signals by hour, settled and open trades, the last run."""
    sp = next(s for s in SPECS if s.tag == tag)
    path = os.path.join(base, "data", sp.db)
    if not os.path.exists(path):
        return {"runs": 0}
    store = Store(path)
    store.c.execute(RUNS_SQL)
    q = store.c.execute
    last = q("SELECT ts, h, day, events, liquid, signals, model_sha, engine, hrrr_ok, notes FROM early_runs "
             "ORDER BY ts DESC LIMIT 1").fetchone()
    out = {"runs": q("SELECT COUNT(*) FROM early_runs").fetchone()[0],
           "signals_by_h": {str(h): n for h, n in q("SELECT h, COUNT(*) FROM taker_shadow GROUP BY h")},
           "settled": q("SELECT COUNT(*) FROM taker_shadow WHERE result IS NOT NULL").fetchone()[0],
           "open": q("SELECT COUNT(*) FROM taker_shadow WHERE result IS NULL").fetchone()[0],
           "snapshots": q("SELECT COUNT(*) FROM ml_snapshots").fetchone()[0]}
    if q("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'exec_log'").fetchone():
        out["exec_log_rows"] = q("SELECT COUNT(*) FROM exec_log").fetchone()[0]
    if last:
        out["last_run"] = dict(zip(["ts", "h", "day", "events", "liquid", "signals", "model", "engine", "hrrr_ok",
                                    "notes"], last))
        out["last_run"]["age_h"] = round((time.time() - last[0]) / 3600, 1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="early_shadow", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "settle", "status"])
    ap.add_argument("--base", default=os.getcwd(), help="the repo directory (data/ lives there)")
    ap.add_argument("--dry-run", action="store_true", help="compute everything, write nothing")
    ap.add_argument("--at", help="pretend it is this UTC time (dry runs only), e.g. 2026-10-08T15:05")
    ap.add_argument("--config", help="a bot config whose exec_log section turns the execution log on, e.g. "
                                     "config/paper_s3.yaml (without one it is off)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if a.cmd == "status":
        print(json.dumps({sp.tag: status(a.base, sp.tag) for sp in SPECS}, indent=1))
        return 0
    if a.cmd == "settle":
        rest = KalshiREST(base_url=PROD, auth=None, read_rate=20.0)
        out = {}
        for sp in available_specs():
            out[sp.tag] = settle(Store(os.path.join(a.base, "data", sp.db)), rest, time.time())
        print(json.dumps({"settled": out}))
        return 0
    now = None
    if a.at:
        now = dt.datetime.fromisoformat(a.at).replace(tzinfo=dt.timezone.utc)
    exec_cfg = None
    if a.config:
        import yaml
        with open(a.config) as fh:
            exec_cfg = (yaml.safe_load(fh) or {}).get("exec_log")
    t0 = time.time()
    s = run(a.base, now=now, dry_run=a.dry_run, exec_log=exec_cfg)
    s["seconds"] = round(time.time() - t0)
    print(json.dumps(s, default=str))
    if not a.dry_run:
        try:
            with open(os.path.join(a.base, "data", "early_shadow_status.json"), "w") as fh:
                json.dump({**{sp.tag: status(a.base, sp.tag) for sp in SPECS}, "last_summary": s}, fh, default=str,
                          indent=1)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
