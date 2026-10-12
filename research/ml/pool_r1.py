"""Round 1, pool side (pre-registered Oct 3, 2026): does the pool do better if its weights
  (1) are refit every day instead of every week, and/or
  (2) get a separate offset on wide books (a "thin" gate: average bracket spread > 10c at the snapshot)?
Baseline = the production procedure: Benter + tilt/curvature per decision hour (BTh), fit on the 90 days ending two
days before, refit weekly (Mondays). All variants are scored on Jun 1 - Sep 30, 2026, per event, paired with the
baseline on the same snapshots, 95% day-block bootstrap intervals. Entry bar: per-event log loss interval below 0
and RPS no worse. Also reported: the recalibrated market (M1h, same window/refit), liquid vs thin books, and the
thin-book score against a bid/ask-bounded ladder (the market is given the benefit of the doubt anywhere inside its
quotes, so only disagreement outside the quotes counts).
Output: POOL_R1_OUT (default mlwx/pool_r1.pkl) with per-snapshot probabilities, and a text summary on stdout.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__))
from cgep import CGEP3, ladder_arrays  # noqa: E402
from tier2_eval_v2 import HOURS, prep  # noqa: E402

MLWX = os.environ.get("MLWX", "/home/claude/mlwx")
TEST_A = pd.Timestamp(os.environ.get("TEST_A", "2026-06-01"))   # round 5: other windows for untouched checks
TEST_B = pd.Timestamp(os.environ.get("TEST_B", "2026-09-30"))
WINDOW = 90


class WarmCGEP3(CGEP3):
    """CGEP3 with an optional warm start (daily refits start from yesterday's weights)."""

    def fit(self, X, LOGS, Z, MASK, WIN, w_init=None):
        n, d = X.shape
        B = self._bases(LOGS, Z, MASK)
        nb = len(B)
        w0 = self._w0(d, nb)
        pen = np.tile(np.ones(d) if self.pen is None else np.asarray(self.pen, float), nb)
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
            return -float(ll.mean()) + 0.5 * float((pen * dw * dw).sum()) / n, g + pen * dw / n

        x0 = w0 if w_init is None or len(w_init) != len(w0) else w_init
        res = minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
        self.w, self.d, self.nb = res.x, d, nb
        return self


def design(h, thin, gate):
    oh = np.stack([(h == hh).astype(float) for hh in HOURS], axis=1)
    return np.hstack([oh, thin[:, None].astype(float)]) if gate else oh


def rps_rows(P, lo, win):
    out = np.zeros(len(P))
    for i in range(len(P)):
        l_ = np.asarray(lo[i], float)
        n = len(l_)
        o = np.argsort(l_)
        p = P[i, :n][o]
        p = p / p.sum()
        y = np.zeros(n)
        y[int(np.flatnonzero(o == win[i])[0])] = 1.0
        out[i] = ((np.cumsum(p) - np.cumsum(y))[:-1] ** 2).sum() / max(n - 1, 1)
    return out


BOUND_FLOOR = 1e-4     # the bounded ladder's smallest probability (as the pool's ladders: cgep.EPS)


def quote_bounds(P, bid, ask, mask, floor=BOUND_FLOOR):
    """Per bracket, the market's [bid, ask] as probabilities: no bid -> 0, no ask (missing or <= 0) -> 1, a crossed
    pair taken in order, the lower end raised to `floor`. Padding (mask False) gets [0, 0]."""
    n, k = P.shape
    lo, hi = np.zeros((n, k)), np.zeros((n, k))
    for i in range(n):
        b = np.nan_to_num(np.asarray(bid[i], float), nan=0.0)
        a = np.nan_to_num(np.asarray(ask[i], float), nan=1.0)
        a = np.where(a <= 0, 1.0, a)
        m = len(b)
        lo[i, :m] = np.maximum(np.minimum(b, a), floor)
        hi[i, :m] = np.maximum(np.maximum(a, b), lo[i, :m])
    return np.where(mask, lo, 0.0), np.where(mask, hi, 0.0)


def _scale_into(p, lo, hi):
    """The c > 0 with sum(clip(c p, lo, hi)) = 1, or None when no c gets there. The sum is continuous, piecewise
    linear and nondecreasing in c, with corners at lo / p and hi / p: find the corners it crosses 1 between and
    interpolate (exact on a linear piece). Brackets with p = 0 sit at lo for every c."""
    pos = p > 0
    base = float(lo[~pos].sum())
    p, lo, hi = p[pos], lo[pos], hi[pos]
    if not len(p) or base + lo.sum() >= 1.0 or base + hi.sum() <= 1.0:
        return None
    cs = np.unique(np.concatenate([lo / p, hi / p]))
    g = np.array([base + np.clip(c * p, lo, hi).sum() for c in cs])   # g[0] = base + sum lo < 1 < g[-1]
    j = int(np.searchsorted(g, 1.0))
    if g[j] == 1.0:
        return float(cs[j])
    return float(cs[j - 1] + (1.0 - g[j - 1]) * (cs[j] - cs[j - 1]) / (g[j] - g[j - 1]))


def bounded_ladder(P, bid, ask, mask, floor=BOUND_FLOOR):
    """The bid/ask-bounded ladder: per row q = clip(c * P, lo, hi), with [lo, hi] the market's quotes (quote_bounds)
    and the one scale c > 0 that makes q sum to 1, so every bracket stays inside its quotes exactly. Among the ladders
    inside the quotes this is the closest to the pool in the Kullback-Leibler sense (it minimizes sum q log(q / P)),
    and it equals the old clip-then-renormalize ladder whenever renormalizing kept every bracket inside its quotes.
    Before Oct 9 (outside review of Oct 8) the ladder was clipped once and then renormalized, which pushes brackets
    back outside their quotes whenever the clipped ladder does not sum to 1. A row whose quotes admit no ladder (the
    lower ends sum past 1, or the upper ends fall short of it) gets the nearest one it can: lo / sum(lo), or
    hi / sum(hi). Padding (mask False) stays 0."""
    mask = np.asarray(mask, bool)
    P = np.where(mask, np.nan_to_num(np.asarray(P, float), nan=0.0), 0.0)
    lo, hi = quote_bounds(P, bid, ask, mask, floor)
    Q = np.zeros_like(P)
    for i in range(len(P)):
        m = mask[i]
        c = _scale_into(P[i, m], lo[i, m], hi[i, m])
        if c is not None:
            Q[i, m] = np.clip(c * P[i, m], lo[i, m], hi[i, m])
        elif lo[i, m].sum() >= 1.0:
            Q[i, m] = lo[i, m] / lo[i, m].sum()
        else:
            Q[i, m] = hi[i, m] / hi[i, m].sum()
    return Q


def bounded_ll(P, bid, ask, win, mask):
    """Log loss of the bid/ask-bounded ladder (bounded_ladder): -log q(win)."""
    Q = bounded_ladder(P, bid, ask, mask)
    return -np.log(Q[np.arange(len(Q)), win])


CI_BLOCK = int(os.environ.get("CI_BLOCK", "1"))   # 7: week blocks (Oct 8 review; amendment A3's promise)


def day_ci(d, events, dates, n=2000, block=None):
    """Per-event mean of d with a bootstrap 95% interval over days (all events of a day resampled together). block > 1
    (or CI_BLOCK in the environment) resamples runs of `block` consecutive days instead (a moving-block bootstrap), for
    errors that persist across days: weather regimes, weekly refits."""
    block = CI_BLOCK if block is None else int(block)
    e = pd.DataFrame({"ev": events, "D": dates, "d": d}).groupby(["ev", "D"], as_index=False)["d"].mean()
    g = e.groupby("D")["d"].agg(["sum", "size"])          # sorted by day
    s, c = g["sum"].values, g["size"].values
    rng = np.random.default_rng(0)
    k = len(s)
    if block <= 1 or k <= block:
        idx = rng.integers(0, k, (n, k))
    else:
        nb = -(-k // block)
        starts = rng.integers(0, k - block + 1, (n, nb))
        idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :k]
    bs = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    return float(e["d"].mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(e)


def fmt(t):
    return f"{t[0]:+.4f} [{t[1]:+.4f},{t[2]:+.4f}] n={t[3]}"


def main(path=None):
    path = path or os.environ.get("T2PKL", f"{MLWX}/t2r1_base.pkl")
    df = prep(path)
    df["D"] = pd.to_datetime(df["D"])
    rows = df.to_dict("records")
    nmax = max(len(r["m"]) for r in rows)
    LM, LF, Z, MASK, WIN = ladder_arrays(rows, nmax)
    h = df["h"].values.astype(int)
    thin = ~df["liquid"].values.astype(bool)
    D = df["D"].values
    test_days = pd.date_range(TEST_A, TEST_B, freq="D")
    specs = {  # name: (experts, tilt/curv, thin gate, refit rule)
        "M1h weekly": (("m",), False, False, "W"),
        "BTh weekly (production)": (("m", "f"), True, False, "W"),
        "BTh daily": (("m", "f"), True, False, "D"),
        "BTh weekly + thin gate": (("m", "f"), True, True, "W"),
        "BTh daily + thin gate": (("m", "f"), True, True, "D"),
    }
    test = (D >= np.datetime64(TEST_A)) & (D <= np.datetime64(TEST_B))
    P = {k: np.full(MASK.shape, np.nan) for k in specs}
    coefs = {k: [] for k in specs}
    t0 = time.time()
    for name, (exp, tc, gate, rule) in specs.items():
        mdl, w_prev, fit_day = None, None, None
        for day in test_days:
            refit = rule == "D" or fit_day is None or (rule == "W" and day.dayofweek == 0)
            if refit:
                end = day - pd.Timedelta(days=2)
                tr = (D <= np.datetime64(end)) & (D > np.datetime64(end - pd.Timedelta(days=WINDOW)))
                X = design(h[tr], thin[tr], gate)
                pen = np.ones(X.shape[1])
                a0 = np.r_[np.ones(len(HOURS)), np.zeros(X.shape[1] - len(HOURS))]
                mdl = WarmCGEP3(len(exp), tilt=tc, curv=tc, pen=pen, alpha0=a0)
                logs = [LM[tr] if e == "m" else LF[tr] for e in exp]
                mdl.fit(X, logs, Z[tr], MASK[tr], WIN[tr], w_init=w_prev)
                w_prev, fit_day = mdl.w.copy(), day
                coefs[name].append((day, mdl.w.copy()))
            te = D == np.datetime64(day)
            if not te.any():
                continue
            X = design(h[te], thin[te], gate)
            logs = [LM[te] if e == "m" else LF[te] for e in exp]
            P[name][te] = mdl.predict(X, logs, Z[te], MASK[te])
        print(f"{name}: {len(coefs[name])} fits, {time.time() - t0:.0f}s", flush=True)
    win = WIN
    sel = test
    ev, dd = df["event"].values, D
    LL = {k: -np.log(np.clip(v[np.arange(len(win)), win], 1e-12, None)) for k, v in P.items()}
    lo = df["lo"].values
    R = {k: np.full(len(win), np.nan) for k in P}
    for k in P:
        R[k][sel] = rps_rows(P[k][sel], lo[sel], win[sel])
    base = "BTh weekly (production)"
    print(f"\nTest Jun 1 - Sep 30, 2026: {int(sel.sum())} snapshots, {df.loc[sel, 'event'].nunique()} events, "
          f"thin share {thin[sel].mean():.2f}")
    for lab, s in (("all books", sel), ("liquid", sel & ~thin), ("thin", sel & thin)):
        print(f"\n== {lab}")
        for k in P:
            if k == base:
                continue
            a = day_ci(LL[k][s] - LL[base][s], ev[s], dd[s])
            b = day_ci(R[k][s] - R[base][s], ev[s], dd[s])
            ok = a[2] < 0 and (b[0] <= 0 or b[1] <= 0)
            print(f"  {k:26s} minus production: log loss {fmt(a)}  RPS {fmt(b)}  {'PASS' if ok else ''}")
        a = day_ci(LL[base][s] - LL["M1h weekly"][s], ev[s], dd[s])
        print(f"  production minus recalibrated market: log loss {fmt(a)}")
    s = sel & thin
    bnd = bounded_ll(P[base][s], df["bid"].values[s], df["ask"].values[s], win[s], MASK[s])
    for k in ("BTh weekly (production)", "BTh weekly + thin gate", "BTh daily + thin gate"):
        bk = bounded_ll(P[k][s], df["bid"].values[s], df["ask"].values[s], win[s], MASK[s])
        a = day_ci(LL[k][s] - bk, ev[s], dd[s])
        print(f"  thin books, {k} minus its bid/ask-bounded ladder: {fmt(a)}")
    del bnd
    last = {k: v[-1][1].round(3).tolist() for k, v in coefs.items() if v}
    print("\nlast weights:", last)
    out = df.loc[sel, ["event", "D", "h", "liquid", "win"]].copy()
    for k in P:
        out[f"ll_{k}"] = LL[k][sel]
        out[f"rps_{k}"] = R[k][sel]
    out.to_pickle(os.environ.get("POOL_R1_OUT", f"{MLWX}/pool_r1.pkl"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
