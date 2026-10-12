"""Refit the production pool weights (Benter + tilt/curvature per decision hour) from ladders the bot can see.

Allowed under the pre-registered freeze: refits are, design changes are not. Same model and penalty as research
(research/ml/cgep.py CGEP3 with hour one-hots, research/ml/train_final.py fit_pool_weights), so the weights mean the
same thing as the ones shipped at the freeze.

rows: dicts with m (market probabilities per bracket), f (Tier-1 probabilities per bracket), lo, hi (bracket bounds
in F, open ends as -1e9 / 1e9), win (index of the bracket that settled YES), h (decision hour), D (event date).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Iterable, Optional

import numpy as np

log = logging.getLogger(__name__)

HOURS = [-3, 3, 9, 12, 15, 18, 21]
EPS = 1e-4


def ladder_arrays(rows, nmax=None):
    nmax = nmax or max(len(r["m"]) for r in rows)
    N = len(rows)
    LM, LF, Z = np.zeros((N, nmax)), np.zeros((N, nmax)), np.zeros((N, nmax))
    MASK, WIN = np.zeros((N, nmax), bool), np.zeros(N, int)
    for i, r in enumerate(rows):
        m = np.array([x if x is not None and np.isfinite(x) else 0.005 for x in r["m"]], float)
        m = np.clip(m, EPS, None)
        m = m / m.sum()
        f = np.clip(np.array(r["f"], float), EPS, None)
        f = f / f.sum()
        lo, hi = np.array(r["lo"], float), np.array(r["hi"], float)
        c = np.where(lo < -1e5, hi - 1.0, np.where(hi > 1e5, lo + 1.0, (lo + hi) / 2.0))
        mu = float((m * c).sum())
        n = len(m)
        LM[i, :n], LF[i, :n], Z[i, :n], MASK[i, :n] = np.log(m), np.log(f), (c - mu) / 2.0, True
        WIN[i] = int(r["win"])
    return LM, LF, Z, MASK, WIN


GATE_CLIP_F = 4.0


def market_mean(r: dict) -> float:
    """The market's implied mean of a pool row (normalized prices, bracket centers; open ends 1 F past the edge)."""
    m = np.clip(np.array([x if x is not None and np.isfinite(x) else 0.005 for x in r["m"]], float), EPS, None)
    m = m / m.sum()
    lo, hi = np.array(r["lo"], float), np.array(r["hi"], float)
    c = np.where(lo < -1e5, hi - 1.0, np.where(hi > 1e5, lo + 1.0, (lo + hi) / 2.0))
    return float((m * c).sum())


def gate_column(rows) -> np.ndarray:
    """Round 5 "nbmdep" (research/ml/r5_pool_gate.py): clip(NBM max - market mean, -4, 4) / 4; 0 without NBM."""
    out = np.zeros(len(rows))
    for i, r in enumerate(rows):
        nbs = r.get("nbs")
        if nbs is not None and np.isfinite(nbs):
            out[i] = np.clip(float(nbs) - market_mean(r), -GATE_CLIP_F, GATE_CLIP_F) / GATE_CLIP_F
    return out


def fit_bth(rows: Iterable[dict], gate: Optional[str] = None) -> dict:
    """Maximum likelihood weights per hour: alpha (market), beta (model), theta (tilt), kappa (curvature).
    gate="nbmdep": one more design column shared across hours (NBM's departure from the market), prior 0, whose
    coefficients shift each exponent with that departure (round 5; rows without "nbs" get 0 there)."""
    from scipy.optimize import minimize
    rows = [r for r in rows if int(r["h"]) in HOURS]
    LM, LF, Z, MASK, WIN = ladder_arrays(rows)
    X = np.stack([np.array([int(r["h"]) == h for r in rows], float) for h in HOURS], axis=1)
    if gate == "nbmdep":
        X = np.hstack([X, gate_column(rows)[:, None]])
    elif gate:
        raise ValueError(f"unknown pool gate {gate!r}")
    B = [np.where(MASK, b, 0.0) for b in (LM, LF, Z, Z * Z)]
    n, d, nb = X.shape[0], X.shape[1], 4
    w0 = np.zeros(nb * d)
    w0[:len(HOURS)] = 1.0
    Y = np.zeros(MASK.shape)
    Y[np.arange(n), WIN] = 1.0

    def f(w):
        s = np.zeros(MASK.shape)
        for k in range(nb):
            s += (X @ w[k * d:(k + 1) * d])[:, None] * B[k]
        s = np.where(MASK, s, -np.inf)
        smax = s.max(axis=1, keepdims=True)
        e = np.exp(s - smax)
        se = e.sum(axis=1, keepdims=True)
        ll = s[np.arange(n), WIN] - (np.log(se[:, 0]) + smax[:, 0])
        R = e / se - Y
        g = np.concatenate([X.T @ (R * B[k]).sum(axis=1) / n for k in range(nb)])
        dw = w - w0
        return -float(ll.mean()) + 0.5 * float((dw * dw).sum()) / n, g + dw / n

    res = minimize(f, w0, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    W = res.x.reshape(nb, d)
    k = len(HOURS)
    out = {"hours": HOURS, "alpha": W[0, :k].round(4).tolist(), "beta": W[1, :k].round(4).tolist(),
           "theta": W[2, :k].round(4).tolist(), "kappa": W[3, :k].round(4).tolist(), "n_snapshots": int(n),
           "z_scale_f": 2.0, "converged": bool(res.success),
           "note": "p_b ∝ m_b^alpha_h f_b^beta_h exp(theta_h z_b + kappa_h z_b^2), z=(center-mkt mean)/2"}
    if gate:
        out["gate"] = {"cols": [gate], "alpha": [round(float(W[0, k]), 4)], "beta": [round(float(W[1, k]), 4)],
                       "theta": [round(float(W[2, k]), 4)], "kappa": [round(float(W[3, k]), 4)],
                       "clip_f": GATE_CLIP_F, "rows_with_nbm": int(sum(1 for r in rows if r.get("nbs") is not None)),
                       "note": "each exponent += coef * clip(NBM max - market mean, -4, 4) / 4"}
    return out


def write_weights(out: dict, path: str, keep_backup: bool = True):
    """Atomic write; the previous weights are copied to path.prev first (the live file never disappears)."""
    import shutil
    if keep_backup and os.path.exists(path):
        shutil.copyfile(path, path + ".prev")
    json.dump(out, open(path + ".tmp", "w"), indent=1)
    os.replace(path + ".tmp", path)


# ------------------------------------------------------------------------------------------- weekly refit core
WINDOW_DAYS = 90
MIN_ROWS = 2000
_MON = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def event_date_of(event: str):
    """KXHIGHNY-26OCT03 -> date(2026, 10, 3); None if the ticker does not parse."""
    import datetime as dt
    try:
        tag = event.split("-")[1]
        return dt.date(2000 + int(tag[:2]), _MON[tag[2:5]], int(tag[5:7]))
    except Exception:
        return None


def live_rows(db_path, start, end, get_markets, model_sha: str | None = None) -> list:
    """Pool rows from the bot's pre-registered snapshots (ml_snapshots) for events dated start..end that Kalshi has
    settled. get_markets(event) -> list of market dicts (ticker, result, strike fields). model_sha: only snapshots
    made by that forecaster (an older model's ladders would fit the wrong weights).
    db_path: the bot's database, or a list of them with the bot's own first: from the switch to real money on, the
    paper and the live record are separate files, and the refit reads both so its window has no hole at the switch
    (the snapshots are the market and the forecaster at the decision hour, the same whichever bot logged them). The
    others are opened read-only and skipped if missing or unreadable. An event-hour logged in more than one keeps the
    whole ladder written first (a ladder is written at once), or the next one if that one misses a settled bracket."""
    import sqlite3
    from collections import defaultdict
    from pathlib import Path
    from ..models.brackets import bracket_from_market
    paths = [db_path] if isinstance(db_path, str) else [p for p in db_path if p]
    cands = defaultdict(list)                       # (event, h) -> [(first ts, {ticker: (market, tier1)})]
    ctx = {}                                        # (event, h) -> NBM's max at that snapshot (ml_snap_ctx)
    for i, path in enumerate(paths):
        if i and not os.path.exists(path):
            continue
        try:
            c = (sqlite3.connect(path, timeout=30) if i == 0 else
                 sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=30))
            try:
                got = c.execute("SELECT ts, event, h, ticker, market, tier1, model_sha FROM ml_snapshots").fetchall()
                try:
                    for ev, h, nbs in c.execute("SELECT event, h, nbs FROM ml_snap_ctx"):
                        ctx.setdefault((ev, int(h)), nbs)
                except sqlite3.Error:     # a database from before the table existed
                    pass
            finally:
                c.close()
        except sqlite3.Error as e:
            if i == 0:
                raise
            log.warning("pool refit: skipped %s (%s)", path, e)
            continue
        ladders = defaultdict(lambda: [float("inf"), {}])
        for ts, ev, h, t, m, f, sha in got:
            if model_sha and sha != model_sha:
                continue
            D = event_date_of(ev)
            if D is not None and start <= D <= end:
                lad = ladders[(ev, int(h))]
                lad[0] = min(lad[0], ts if ts is not None else float("inf"))
                lad[1][t] = (m, f)
        for k, (ts, br) in ladders.items():
            cands[k].append((ts, i, br))
    cache, rows = {}, []
    for (ev, h), options in sorted(cands.items()):
        if ev not in cache:
            try:
                cache[ev] = [m for m in (get_markets(ev) or []) if m.get("result") in ("yes", "no")]
            except Exception:
                cache[ev] = []
        ms = cache[ev]
        if len(ms) < 3 or sum(m["result"] == "yes" for m in ms) != 1:
            continue
        br = next((b for _, _, b in sorted(options, key=lambda o: (o[0], o[1]))
                   if all(m["ticker"] in b and b[m["ticker"]][0] is not None and b[m["ticker"]][1] is not None
                          for m in ms)), None)
        if br is None:
            continue
        try:
            bs = [bracket_from_market(m) for m in ms]
        except (ValueError, TypeError):
            continue
        rows.append({"m": [br[m["ticker"]][0] for m in ms], "f": [br[m["ticker"]][1] for m in ms],
                     "lo": [float(b.lo) for b in bs], "hi": [float(b.hi) for b in bs],
                     "win": [m["result"] for m in ms].index("yes"), "h": h, "D": str(event_date_of(ev)), "src": "bot",
                     "nbs": ctx.get((ev, h))})
    return rows


def backfill_ctx(db_path: str, raw_dir: str) -> int:
    """Fill ml_snap_ctx (NBM's max per event and decision hour) for snapshots logged without it, from the bot's raw
    folder with the forecaster's own feature code (wxbot.ml.features.station_rows: the NBM text run public at each
    decision time), so a refit sees the value the forecaster saw. The raw folder keeps 45 days of NBM text; older
    snapshots stay without (the refit gives them no tilt). Writes only to db_path (the bot's own database).
    Returns the number of rows filled."""
    import sqlite3
    import pandas as pd
    from . import features as F
    st_map = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "stations.json")))
    c = sqlite3.connect(db_path, timeout=60)
    try:
        c.execute("CREATE TABLE IF NOT EXISTS ml_snap_ctx (event TEXT, h INTEGER, nbs REAL, ts REAL, "
                  "PRIMARY KEY (event, h))")
        have = set(c.execute("SELECT event, h FROM ml_snap_ctx").fetchall())
        need = [(ev, int(h), ts) for ev, h, ts in
                c.execute("SELECT event, h, MIN(ts) FROM ml_snapshots GROUP BY event, h") if (ev, int(h)) not in have]
        by_st = {}
        for ev, h, ts in need:
            icao = (st_map.get(ev.split("-")[0]) or {}).get("icao")
            D = event_date_of(ev)
            if icao in F.HIGH and D is not None:
                by_st.setdefault(icao, []).append((ev, h, ts, pd.Timestamp(D)))
        src = F.RawSource(raw_dir)
        rows = []
        for icao, lst in sorted(by_st.items()):
            frame = pd.DataFrame([(D, h) for _, h, _, D in lst], columns=["D", "h"]).drop_duplicates()
            try:
                r = F.station_rows(src, icao, frame)
            except Exception as e:      # a station with no raw files yet
                log.debug("ctx backfill %s: %s", icao, e)
                continue
            val = {(pd.Timestamp(d), int(h)): v for d, h, v in zip(r["D"], r["h"], r["nbs_txn"])}
            for ev, h, ts, D in lst:
                v = val.get((D, h))
                if v is not None and pd.notna(v):
                    rows.append((ev, h, float(v), ts))
        c.executemany("INSERT OR IGNORE INTO ml_snap_ctx VALUES (?,?,?,?)", rows)
        c.commit()
        return len(rows)
    finally:
        c.close()


def sane(out: dict) -> bool:
    """Guard against a broken fit replacing working weights: converged, every hour present, weights in the range
    seen across all research fits (market 0.5-1.6, model -0.1-1.0)."""
    try:
        ok = (bool(out.get("converged")) and out["hours"] == HOURS
              and all(0.5 <= a <= 1.6 for a in out["alpha"]) and all(-0.1 <= b <= 1.0 for b in out["beta"])
              and all(abs(t) <= 1.0 for t in out["theta"]) and all(abs(k) <= 1.0 for k in out["kappa"]))
        g = out.get("gate")
        if ok and g:   # research fits, Feb-May and Jun-Sep 2026: tilt -0.59 / -0.60, the others within +-0.25
            ok = all(abs(float(g[n][0])) <= 1.5 for n in ("alpha", "beta", "theta", "kappa"))
        return ok
    except (KeyError, TypeError, IndexError):
        return False


def refit(db_path, base_rows: list, today, get_markets, window_days: int = WINDOW_DAYS,
          min_rows: int = MIN_ROWS, model_sha: str | None = None, gate: str | None = None):
    """The production procedure: weights on the window_days days ending two days before today. The bot's own
    record (db_path: one database or several, see live_rows) wins from its first day on; research rows (exported at
    the freeze) fill the earlier part.
    Returns (weights or None, message)."""
    import datetime as dt
    end = today - dt.timedelta(days=2)
    start = end - dt.timedelta(days=window_days - 1)
    live = live_rows(db_path, start, end, get_markets, model_sha)
    first_live = min((r["D"] for r in live), default=None)
    rows = [r for r in base_rows if start.isoformat() <= r["D"] <= end.isoformat()
            and (first_live is None or r["D"] < first_live)]
    rows += live
    msg = f"window {start}..{end}: {len(rows)} snapshots ({len(rows) - len(live)} research, {len(live)} from the bot)"
    if len(rows) < min_rows:
        return None, msg + "; too few for a stable fit, weights unchanged"
    out = fit_bth(rows, gate=gate)
    if gate:
        msg += f"; {sum(1 for r in rows if r.get('nbs') is not None)} with NBM's max for the {gate} gate"
    out.update({"first_day": min(r["D"] for r in rows), "last_day": max(r["D"] for r in rows),
                "window_days": window_days, "refit_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "sources": {"research": len(rows) - len(live), "bot": len(live)}})
    if not sane(out):
        return None, msg + f"; fit failed the sanity bounds (alpha {out['alpha']}, beta {out['beta']}), weights unchanged"
    return out, msg
