"""Paper check of the pool's NBM-departure tilt on today's real books (RUNBOOK "Stage 3"): the live forecaster
(Tier1Live, shipped model) on a copy of the bot's raw folder, Kalshi's public order books for the day's KXHIGH events,
and the pool with the shipped weights and with the tilted ones, side by side. Read-only: no key, no orders.
Usage: python3 scripts/pool_check.py RAW_DIR [DAY]    (DAY default: today, UTC; RAW_DIR is refreshed by the predict)
"""
import datetime as dt
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from wxbot.app import early_shadow as ES  # noqa: E402
from wxbot.kalshi.rest import PROD, KalshiREST  # noqa: E402
from wxbot.ml import features as F  # noqa: E402
from wxbot.ml.live import ART, Pool, Tier1Live, bracket_pmf, nbm_dep  # noqa: E402
from wxbot.models.fair_value import book_to_quote, ladder_fair_value  # noqa: E402


def mean_of(brackets, p):
    c = np.array([(b.hi - 1.0) if b.lo < -1e5 else (b.lo + 1.0) if b.hi > 1e5 else (b.lo + b.hi) / 2.0
                  for b in brackets])
    return float((np.asarray(p) * c).sum())


def main(raw, day=None):
    now = dt.datetime.now(dt.timezone.utc)
    day = dt.date.fromisoformat(day) if day else now.date()
    rest = KalshiREST(base_url=PROD, auth=None, read_rate=10.0)
    evs = ES.list_events(rest, day)
    books = ES.read_books(rest, [b.ticker for e in evs for b in e["brackets"]])
    t = Tier1Live(raw, sorted(F.HIGH))
    fc = t.predict(sorted({(e["icao"], day) for e in evs}), now)
    plain, gated = Pool(os.path.join(ART, "pool_weights.json")), Pool(os.path.join(ART, "pool_weights_gate.json"))
    print(f"{day}, decision hour {next(iter(fc.values()))['h'] if fc else '?'}; gate {gated.gate['theta'][0]:+.3f} "
          f"tilt per unit departure")
    print(f"{'event':22s} {'mkt mean':>8s} {'NBM':>5s} {'dep':>5s} {'T1 mean':>7s} {'pool':>6s} {'tilted':>6s}  "
          f"largest bracket change")
    shifts = []
    for e in evs:
        f = fc.get((e["icao"], day))
        quotes = {b.ticker: book_to_quote(books.get(b.ticker) or {}) for b in e["brackets"]}
        fv = ladder_fair_value(e["brackets"], quotes, kind="high")
        if not f or not fv.market:
            continue
        pmf = bracket_pmf(e["brackets"], f["P"], f["anchor_r"])
        m = np.array([fv.market.get(b.ticker, 0.005) for b in e["brackets"]], float)
        m = np.clip(m, 1e-4, None) / np.clip(m, 1e-4, None).sum()
        mu_m = mean_of(e["brackets"], m)
        p0 = plain.probs(e["brackets"], fv.market, pmf, f["h"])
        p1 = gated.probs(e["brackets"], fv.market, pmf, f["h"], nbs=f.get("nbs_txn"))
        v0 = np.array([p0[b.ticker] for b in e["brackets"]])
        v1 = np.array([p1[b.ticker] for b in e["brackets"]])
        j = int(np.argmax(np.abs(v1 - v0)))
        shifts.append(mean_of(e["brackets"], v1) - mean_of(e["brackets"], v0))
        nbs = f.get("nbs_txn")
        print(f"{e['event']:22s} {mu_m:8.2f} {nbs if nbs is not None else float('nan'):5.0f} "
              f"{nbm_dep(nbs, mu_m):+5.2f} {mean_of(e['brackets'], pmf):7.2f} {mean_of(e['brackets'], v0):6.2f} "
              f"{mean_of(e['brackets'], v1):6.2f}  {e['brackets'][j].ticker.split('-')[-1]} {v0[j]:.2f} -> {v1[j]:.2f}")
    if shifts:
        print(f"\n{len(shifts)} events; the tilt moved the pooled mean by {np.mean(np.abs(shifts)):.2f} F on average "
              f"(largest {max(shifts, key=abs):+.2f} F)")


if __name__ == "__main__":
    main(*sys.argv[1:])
