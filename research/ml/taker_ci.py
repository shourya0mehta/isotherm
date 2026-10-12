"""Taker backtest with day-block bootstrap CIs, by model, margin and decision hour (liquid snapshots)."""
import os, sys, math
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from tier2_eval import block_ci
MLWX = os.environ.get("MLWX", "/home/claude/mlwx")
ORDER_N = int(os.environ.get("ORDER_N", "1"))  # contracts per order: Kalshi rounds the fee up to the cent per ORDER


def fee_per_contract(price, n=ORDER_N):
    """ceil(0.07 * n * P * (1 - P)) cents for an n-contract order, spread over its n contracts.
    n = 1 (default) rounds every contract up, which is conservative; n = 10 adds about 0.1c per contract."""
    return math.ceil(round(0.07 * n * price * (1 - price) * 100, 9)) / 100 / n

def trades(P, rows, margin):
    """Per snapshot: list of (pnl, side, price, p) for taker buys at the snapshot."""
    pnl = np.zeros(len(rows)); ntr = np.zeros(len(rows)); cost = np.zeros(len(rows))
    for i, r in enumerate(rows):
        for b in range(len(r["m"])):
            p = P[i, b]; a, bd = r["ask"][b], r["bid"][b]; win = 1.0 if b == r["win"] else 0.0
            if np.isfinite(a) and 0 < a < 1:
                fee = fee_per_contract(a)
                if p - a > fee + margin:
                    pnl[i] += win - a - fee; ntr[i] += 1; cost[i] += a + fee
            if np.isfinite(bd) and 0 < bd < 1:
                c = 1 - bd; fee = fee_per_contract(c)
                if (1 - p) - c > fee + margin:
                    pnl[i] += (1 - win) - c - fee; ntr[i] += 1; cost[i] += c + fee
    return pnl, ntr, cost

TAG = os.environ.get("T2TAG", "tier2_v2")
meta = pd.read_pickle(f"{MLWX}/{TAG}_eval_meta.pkl")
Pz = np.load(f"{MLWX}/{TAG}_P.npz")
df = pd.read_pickle(f"{MLWX}/{TAG}.pkl")
key = df.set_index(["event", "h"])
rows = [key.loc[(e, h)].to_dict() for e, h in zip(meta.event, meta.h)]
liq = meta.liquid.values
dates = meta.D.values
for name in ["M1h", "Bh", "BTh", "G", "A", "AG"]:
    for margin in (0.0, 0.03, 0.06):
        pnl, ntr, cost = trades(Pz[name], rows, margin)
        s = liq
        tot = pnl[s].sum(); n = ntr[s].sum()
        # CI on pnl per contract via day blocks: ratio of sums
        ud = np.unique(dates[s]); idx = {d: np.where((dates == d) & s)[0] for d in ud}
        P_ = np.array([pnl[idx[d]].sum() for d in ud]); N_ = np.array([ntr[idx[d]].sum() for d in ud]); C_ = np.array([cost[idx[d]].sum() for d in ud])
        rng = np.random.default_rng(0); bs = []
        for _ in range(2000):
            k = rng.integers(0, len(ud), len(ud)); bs.append(P_[k].sum() / max(N_[k].sum(), 1))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"{name:4s} margin {margin:.2f}: trades {int(n):6d}  pnl ${tot:8.2f}  per contract {100*tot/max(n,1):+.2f}c "
              f"[{100*lo:+.2f},{100*hi:+.2f}]  ROI {tot/max(C_.sum(),1e-9):+.3f}  days+ {(P_>0).mean():.2f}")
    print()
# by hour and source for Bh at margin 0.03
pnl, ntr, cost = trades(Pz[sys.argv[1] if len(sys.argv) > 1 else "A"], rows, 0.03)
m2 = meta.assign(pnl=pnl, ntr=ntr)
print(m2[liq].groupby("h")[["pnl", "ntr"]].sum().assign(c_per=lambda x: 100 * x.pnl / x.ntr).round(2))
print(m2[liq].groupby("src")[["pnl", "ntr"]].sum().assign(c_per=lambda x: 100 * x.pnl / x.ntr).round(2))
print(m2[liq].groupby(m2.D.dt.to_period("M"))[["pnl", "ntr"]].sum().assign(c_per=lambda x: 100 * x.pnl / x.ntr).round(2))
