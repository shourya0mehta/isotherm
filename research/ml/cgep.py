"""Context-Gated Exponential Pooling (CGEP) of a prediction-market ladder with a model ladder.

For one event with brackets b = 1..n, market probabilities m_b, model probabilities f_b, and bracket centers z_b
(degrees F relative to the market mean, scaled), the pooled probability is

    p_b  proportional to  exp( alpha(x) log m_b + beta(x) log f_b + theta(x) z_b + kappa(x) z_b^2 )

with alpha, beta, theta, kappa linear in a context vector x (time of day, season, market quality, model-market
disagreement, both sides' recent errors, ...). With x = [1] and theta = kappa = 0 this is Benter's (1994)
conditional logit; beta = 0 gives a recalibrated market; the tilt terms shift and sharpen/flatten the ladder.
The log-likelihood is concave in the parameters (the score is linear in them), so the fit has one optimum.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

EPS = 1e-4


def ladder_arrays(rows, nmax=None):
    """rows: list of dicts with m, f, lo, hi, win. Returns padded arrays and a mask."""
    nmax = nmax or max(len(r["m"]) for r in rows)
    N = len(rows)
    LM = np.zeros((N, nmax)); LF = np.zeros((N, nmax)); Z = np.zeros((N, nmax)); MASK = np.zeros((N, nmax), bool)
    WIN = np.zeros(N, int)
    for i, r in enumerate(rows):
        m = np.array([x if np.isfinite(x) else 0.005 for x in r["m"]], float)
        m = np.clip(m, EPS, None); m = m / m.sum()
        f = np.clip(np.array(r["f"], float), EPS, None); f = f / f.sum()
        lo = np.array(r["lo"], float); hi = np.array(r["hi"], float)
        c = np.where(lo < -1e5, hi - 1.0, np.where(hi > 1e5, lo + 1.0, (lo + hi) / 2.0))
        mu = float((m * c).sum())
        n = len(m)
        LM[i, :n] = np.log(m); LF[i, :n] = np.log(f); Z[i, :n] = (c - mu) / 2.0; MASK[i, :n] = True
        WIN[i] = r["win"]
    return LM, LF, Z, MASK, WIN


class CGEP:
    def __init__(self, gate_alpha=True, gate_beta=True, tilt=True, curv=True, l2=1.0):
        self.gate_alpha, self.gate_beta, self.tilt, self.curv, self.l2 = gate_alpha, gate_beta, tilt, curv, l2
        self.w = None

    def _blocks(self, X):
        """Per-term design: each term's coefficient is x @ w_term (or a constant if not gated)."""
        n, d = X.shape
        one = np.ones((n, 1))
        terms = [("alpha", X if self.gate_alpha else one), ("beta", X if self.gate_beta else one)]
        if self.tilt:
            terms.append(("theta", X))
        if self.curv:
            terms.append(("kappa", one if not self.gate_alpha else X))
        return terms

    def _scores(self, w, terms, LM, LF, Z, MASK):
        s = np.zeros_like(LM)
        k = 0
        for name, D in terms:
            coef = D @ w[k:k + D.shape[1]]
            k += D.shape[1]
            if name == "alpha":
                s += coef[:, None] * LM
            elif name == "beta":
                s += coef[:, None] * LF
            elif name == "theta":
                s += coef[:, None] * Z
            else:
                s += coef[:, None] * Z ** 2
        return np.where(MASK, s, -np.inf)

    def _init(self, terms):
        w = []
        for name, D in terms:
            v = np.zeros(D.shape[1])
            if name == "alpha":
                v[0] = 1.0
            w.append(v)
        return np.concatenate(w)

    def fit(self, X, LM, LF, Z, MASK, WIN):
        terms = self._blocks(X)
        w0 = self._init(terms)
        n = len(WIN)

        def nll(w):
            s = self._scores(w, terms, LM, LF, Z, MASK)
            smax = s.max(axis=1, keepdims=True)
            lse = np.log(np.exp(s - smax).sum(axis=1)) + smax[:, 0]
            ll = s[np.arange(n), WIN] - lse
            reg = self.l2 * 0.5 * float(((w - self._init(terms)) ** 2).sum()) / n
            return -float(ll.mean()) + reg

        def grad(w):
            s = self._scores(w, terms, LM, LF, Z, MASK)
            smax = s.max(axis=1, keepdims=True)
            e = np.exp(s - smax)
            P = e / e.sum(axis=1, keepdims=True)
            Y = np.zeros_like(P); Y[np.arange(n), WIN] = 1.0
            R = (P - Y)  # d(-ll)/ds
            g = []
            for name, D in terms:
                base = LM if name == "alpha" else LF if name == "beta" else Z if name == "theta" else Z ** 2
                v = np.where(MASK, R * base, 0.0).sum(axis=1)  # per-row derivative wrt the coefficient
                g.append(D.T @ v / n)
            g = np.concatenate(g) + self.l2 * (w - self._init(terms)) / n
            return g

        res = minimize(nll, w0, jac=grad, method="L-BFGS-B", options={"maxiter": 500})
        self.w, self.terms_dims = res.x, [(nm, D.shape[1]) for nm, D in terms]
        return self

    def predict(self, X, LM, LF, Z, MASK):
        terms = self._blocks(X)
        s = self._scores(self.w, terms, LM, LF, Z, MASK)
        smax = s.max(axis=1, keepdims=True)
        e = np.where(MASK, np.exp(s - smax), 0.0)
        return e / e.sum(axis=1, keepdims=True)

    def coef_summary(self, names):
        out, k = {}, 0
        for nm, d in self.terms_dims:
            out[nm] = dict(zip(names if d == len(names) else ["const"], np.round(self.w[k:k + d], 3)))
            k += d
        return out


class CGEP2:
    """CGEP with one shared context design X (n x d), per-column L2 penalties and an alpha prior.

    score_b = (X w_a) log m_b + (X w_b) log f_b + (X w_t) z_b + (X w_k) z_b^2 for the terms in `use`.
    Penalty: 0.5 * sum_t sum_j pen_j (w_tj - w0_tj)^2 / n, with w0 = alpha0 for alpha and 0 otherwise.
    Concave log-likelihood, so L-BFGS finds the global optimum.
    """

    BASES = ("alpha", "beta", "theta", "kappa")

    def __init__(self, use=("alpha", "beta"), pen=None, alpha0=None):
        self.use, self.pen, self.alpha0 = tuple(use), pen, alpha0
        self.w = None

    def _base(self, name, LM, LF, Z):
        return {"alpha": LM, "beta": LF, "theta": Z, "kappa": Z * Z}[name]

    def _w0(self, d):
        out = []
        for t in self.use:
            v = np.zeros(d)
            if t == "alpha":
                v = np.array(self.alpha0, float) if self.alpha0 is not None else np.r_[1.0, np.zeros(d - 1)]
            out.append(v)
        return np.concatenate(out)

    def _scores(self, w, X, LM, LF, Z, MASK):
        d = X.shape[1]
        s = np.zeros_like(LM)
        for k, t in enumerate(self.use):
            coef = X @ w[k * d:(k + 1) * d]
            s += coef[:, None] * self._base(t, LM, LF, Z)
        return np.where(MASK, s, -np.inf)

    def fit(self, X, LM, LF, Z, MASK, WIN):
        n, d = X.shape
        w0 = self._w0(d)
        pen = np.tile(np.ones(d) if self.pen is None else np.asarray(self.pen, float), len(self.use))
        Y = np.zeros(LM.shape); Y[np.arange(n), WIN] = 1.0
        bases = [np.where(MASK, self._base(t, LM, LF, Z), 0.0) for t in self.use]

        def f(w):
            s = self._scores(w, X, LM, LF, Z, MASK)
            smax = s.max(axis=1, keepdims=True)
            e = np.exp(s - smax)
            se = e.sum(axis=1, keepdims=True)
            P = e / se
            ll = s[np.arange(n), WIN] - (np.log(se[:, 0]) + smax[:, 0])
            R = P - Y
            g = np.concatenate([X.T @ (R * B).sum(axis=1) / n for B in bases])
            dw = w - w0
            return -float(ll.mean()) + 0.5 * float((pen * dw * dw).sum()) / n, g + pen * dw / n

        res = minimize(f, w0, jac=True, method="L-BFGS-B", options={"maxiter": 1000})
        self.w, self.d = res.x, d
        return self

    def predict(self, X, LM, LF, Z, MASK):
        s = self._scores(self.w, X, LM, LF, Z, MASK)
        smax = s.max(axis=1, keepdims=True)
        e = np.where(MASK, np.exp(s - smax), 0.0)
        return e / e.sum(axis=1, keepdims=True)

    def coefs(self, names):
        return {t: dict(zip(names, np.round(self.w[k * self.d:(k + 1) * self.d], 3))) for k, t in enumerate(self.use)}


def log_ladder(rows, key, nmax, fallback="m"):
    """Log of a normalized bracket ladder stored under rows[i][key] (NaN mids -> 0.005), padded to nmax."""
    L = np.zeros((len(rows), nmax))
    for i, r in enumerate(rows):
        v = r.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = r[fallback]
        x = np.array([q if q is not None and np.isfinite(q) else 0.005 for q in v], float)
        x = np.clip(x, EPS, None); x = x / x.sum()
        L[i, :len(x)] = np.log(x)
    return L


class CGEP3:
    """Exponential pool of several expert ladders: score_b = sum_k (X w_k) base_k[b], where the bases are the experts'
    log-probabilities (market now, market h hours ago, model, ...) plus optional tilt z_b and curvature z_b^2.
    Per-column L2 toward w0 (market weight prior on expert 0). Concave; L-BFGS."""

    def __init__(self, n_exp, tilt=True, curv=True, pen=None, alpha0=None):
        self.n_exp, self.tilt, self.curv, self.pen, self.alpha0 = n_exp, tilt, curv, pen, alpha0

    def _bases(self, LOGS, Z, MASK):
        B = list(LOGS)
        if self.tilt:
            B.append(Z)
        if self.curv:
            B.append(Z * Z)
        return [np.where(MASK, b, 0.0) for b in B]

    def _w0(self, d, nb):
        w = np.zeros(nb * d)
        w[:d] = self.alpha0 if self.alpha0 is not None else np.r_[1.0, np.zeros(d - 1)]
        return w

    def fit(self, X, LOGS, Z, MASK, WIN):
        n, d = X.shape
        B = self._bases(LOGS, Z, MASK)
        nb = len(B)
        w0 = self._w0(d, nb)
        pen = np.tile(np.ones(d) if self.pen is None else np.asarray(self.pen, float), nb)
        Y = np.zeros(MASK.shape); Y[np.arange(n), WIN] = 1.0

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

        res = minimize(f, w0, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
        self.w, self.d, self.nb = res.x, d, nb
        return self

    def predict(self, X, LOGS, Z, MASK):
        B = self._bases(LOGS, Z, MASK)
        s = np.zeros(MASK.shape)
        for k in range(self.nb):
            s += (X @ self.w[k * self.d:(k + 1) * self.d])[:, None] * B[k]
        s = np.where(MASK, s, -np.inf)
        smax = s.max(axis=1, keepdims=True)
        e = np.where(MASK, np.exp(s - smax), 0.0)
        return e / e.sum(axis=1, keepdims=True)

    def coefs(self, names, base_names):
        return {bn: dict(zip(names, np.round(self.w[k * self.d:(k + 1) * self.d], 3))) for k, bn in enumerate(base_names)}
