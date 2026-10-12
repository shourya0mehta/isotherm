"""Paper-trading report for the running bot (standard library only, so it runs anywhere Python does).

Fills and settled P&L by sleeve (A = near-certain favorites, T = thin-book sleeve), by settlement day, open
positions, and the pre-registered ML scorecard so far (pool vs market log loss on settled events, per event).
Usage: python3 scripts/paper_report.py [db_path] [status_path]  ->  prints markdown and writes data/paper_report.md
"""
import datetime as dt
import json
import math
import os
import random
import sqlite3
import sys
from collections import defaultdict

DB = sys.argv[1] if len(sys.argv) > 1 else "data/wxbot.sqlite"
ST = sys.argv[2] if len(sys.argv) > 2 else "data/status.json"


def day(ts):
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d")


def main():
    c = sqlite3.connect(DB)
    st = json.load(open(ST)) if os.path.exists(ST) else {}
    out = []
    w = out.append
    now = dt.datetime.now(dt.timezone.utc)
    mode = st.get("mode") or "paper"
    w(f"# {'Live' if mode == 'live' else 'Paper'} report, {now:%Y-%m-%d %H:%M} UTC\n")
    w(f"Bot: mode {st.get('mode')}, up {st.get('uptime_min')} min, halted {st.get('halted')} {st.get('halt_reason') or ''}, "
      f"quotes {st.get('quotes')}, API errors {st.get('api_errors')} of {st.get('api_requests')} requests.\n")
    sleeve_section(c, w, mode)
    # pre-registered scorecard: one snapshot per (event, decision hour), scored once the event settles
    res = {t: r for t, r in c.execute("SELECT ticker, result FROM settlements")}
    sha = None
    meta_path = os.path.join("wxbot", "ml", "artifacts", "tier1_v2_meta.json")
    if os.path.exists(meta_path):
        sha = json.load(open(meta_path)).get("sha256", "")[:12]
    snaps = defaultdict(list)
    for ts, ev, h, t, m, f, p, ms in c.execute("SELECT ts, event, h, ticker, market, tier1, pooled, model_sha FROM ml_snapshots"):
        if sha and ms != sha:  # score the model now deployed (the frozen one after Oct 11)
            continue
        snaps[(ev, h)].append((t, m, f, p, ts))
    per_event = defaultdict(list)
    for (ev, h), rows in snaps.items():
        r = {t: res.get(t) for t, *_ in rows}
        if not rows or any(v not in ("yes", "no") for v in r.values()) or sum(v == "yes" for v in r.values()) != 1:
            continue
        msum = sum(max(m or 0.0, 1e-4) for _, m, _, _, _ in rows)
        psum = sum(max(p or 0.0, 1e-4) for _, _, _, p, _ in rows)
        for t, m, f, p, ts in rows:
            if r[t] == "yes":
                lm = -math.log(max(m or 0.0, 1e-4) / msum)
                lp = -math.log(max(p or 0.0, 1e-4) / psum)
                per_event[(ev, day(ts))].append(lp - lm)
    w(f"\n## Pre-registered ML scorecard, model {sha} (pool minus market log loss, per event; negative = pool better)\n")
    if per_event:
        evd = [(d, sum(v) / len(v)) for (ev, d), v in per_event.items()]
        days = defaultdict(list)
        for d, v in evd:
            days[d].append(v)
        keys = list(days)
        mean = sum(v for _, v in evd) / len(evd)
        rng = random.Random(0)
        bs = []
        for _ in range(2000):
            pick = [rng.choice(keys) for _ in keys]
            vals = [v for k in pick for v in days[k]]
            bs.append(sum(vals) / len(vals))
        bs.sort()
        w(f"{len(evd)} settled events over {len(days)} days: {mean:+.4f} (95% interval {bs[50]:+.4f} to {bs[1949]:+.4f}).")
    else:
        w("No settled snapshots yet.")
    latency_section(c, w, sha)
    shadow_taker_section(c, w)
    hourly_shadow_section(c, w)
    rain_shadow_section(c, w)
    rain_shadow_section(c, w, "rain_shadow_v2")
    text = "\n".join(out) + "\n"
    print(text)
    os.makedirs("data", exist_ok=True)
    name = "live_report.md" if mode == "live" else "paper_report.md"
    open(os.path.join(os.path.dirname(DB) or ".", name), "w").write(text)


def sleeve_pnl(c, mode):
    """Fills and settled P&L by sleeve, and settled P&L by day and sleeve, from the bot's own fills in this mode.
    Settled P&L is after fees: each fill's payout minus its cost minus the fee recorded with it (Kalshi's taker fee;
    makers pay none on the weather series), as the bot's own daily stops and shut-off count it (Bot.sleeve_pnl_today,
    wxbot/risk/shutoff.daily_pnl). Until Oct 9 this table left the fees out (outside review of Oct 8), so the taker
    sleeves read higher than they were. The fees of fills still open are in Fees (paid) and not yet in the P&L."""
    res = {t: r for t, r in c.execute("SELECT ticker, result FROM settlements")}
    settled_at = {t: s for t, s in c.execute("SELECT ticker, settled_ts FROM settlements")}
    fills = c.execute("SELECT ts, ticker, event, side, price_cents, count, sleeve, fee FROM fills WHERE mode=? "
                      "ORDER BY ts", (mode,)).fetchall()
    by_sleeve = defaultdict(lambda: {"fills": 0, "contracts": 0.0, "cost": 0.0, "fees": 0.0, "settled_pnl": 0.0,
                                     "open_cost": 0.0, "won": 0, "lost": 0})
    by_day = defaultdict(lambda: defaultdict(float))
    for ts, t, ev, side, pc, n, sl, fee in fills:
        cost = n * (pc if side == "bid" else 100 - pc) / 100.0
        fee = float(fee or 0.0)
        s = by_sleeve[sl or "A"]
        s["fills"] += 1
        s["contracts"] += n
        s["cost"] += cost
        s["fees"] += fee
        r = res.get(t)
        if r in ("yes", "no"):
            win = (side == "bid" and r == "yes") or (side == "ask" and r == "no")
            pnl = (n if win else 0.0) - cost - fee
            s["settled_pnl"] += pnl
            s["won" if win else "lost"] += 1
            by_day[day(settled_at.get(t) or ts)][sl or "A"] += pnl
        else:
            s["open_cost"] += cost
    return by_sleeve, by_day


def sleeve_section(c, w, mode):
    """The report's table of fills and P&L by sleeve and its settled P&L by day, both after fees (sleeve_pnl)."""
    by_sleeve, by_day = sleeve_pnl(c, mode)
    w("## Fills and P&L by sleeve\n")
    w("| Sleeve | Fills | Contracts | Cost | Fees | Settled P&L after fees | Fills won / lost | Open cost |")
    w("|---|---|---|---|---|---|---|---|")
    for sl, s in sorted(by_sleeve.items()):
        w(f"| {sl} | {s['fills']} | {s['contracts']:.0f} | ${s['cost']:.2f} | ${s['fees']:.2f} | "
          f"${s['settled_pnl']:+.2f} | {s['won']} / {s['lost']} | ${s['open_cost']:.2f} |")
    if not by_sleeve:
        w("| (no fills yet) | | | | | | | |")
    w("\n## Settled P&L by day (UTC, after fees)\n")
    for d in sorted(by_day):
        w(f"- {d}: " + ", ".join(f"{k} ${v:+.2f}" for k, v in sorted(by_day[d].items())))


MON = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV",
                                   "DEC"], 1)}


def latency_section(c, w, sha):
    """Minutes from each decision hour to the first pooled price the bot logged for it (deployed model only)."""
    lat = defaultdict(list)
    for ev, h, ts in c.execute("SELECT event, h, MIN(ts) FROM ml_snapshots" +
                               (" WHERE model_sha = ?" if sha else "") + " GROUP BY event, h", (sha,) if sha else ()):
        try:
            tag = ev.split("-")[1]
            d0 = dt.datetime(2000 + int(tag[:2]), MON[tag[2:5]], int(tag[5:7]), tzinfo=dt.timezone.utc)
        except (IndexError, KeyError, ValueError):
            continue
        m = (ts - (d0.timestamp() + int(h) * 3600)) / 60.0
        if 0 <= m < 90:  # later ones are restarts, not decision-hour refreshes
            lat[day(ts)].append(m)
    if lat:
        w("\n## Forecast latency (minutes from decision hour to first pooled price)\n")
        for d in sorted(lat):
            v = sorted(lat[d])
            w(f"- {d}: median {v[len(v) // 2]:.1f}, max {v[-1]:.1f} ({len(v)} event-hours)")


def booked_col(c, table):
    """The shadow table's booked-size column (added Oct 9), or NULL for a database from before it."""
    return "booked" if "booked" in {r[1] for r in c.execute(f"PRAGMA table_info({table})")} else "NULL"


def taker_booked(size, avail, booked):
    """A taker_shadow row's booked size (outside review, Oct 8: the shadow logs assumed fills): the logged size capped
    by the contracts offered at the price, 0 when that depth was zero or unknown. Rows from before the booked column
    had the displayed size at the touch (avail), which is that depth."""
    if booked is not None:
        return float(booked)
    return min(float(size), float(avail)) if avail is not None and avail > 0 else 0.0


def shadow_taker_section(c, w):
    """Sleeve X (logged only, never traded): settled results of the pre-registered model-taker rule, at the booked
    size; a signal that booked nothing (depth zero or unknown) counts as no trade."""
    try:
        rows = c.execute("SELECT ts, event, ticker, side, price, size, p, edge, h, result, avail, "
                         f"{booked_col(c, 'taker_shadow')} FROM taker_shadow").fetchall()
    except sqlite3.OperationalError:
        return
    w("\n## Shadow taker, Sleeve X (logged only, never traded)\n")
    if not rows:
        w("No signals yet.")
        return
    rows = [r[:10] + (taker_booked(r[5], r[10], r[11]),) for r in rows]
    settled_all = [r for r in rows if r[9] in ("yes", "no")]
    settled = [r for r in settled_all if r[10] > 0]
    w(f"{len(rows)} signals, {sum(r[10] for r in rows):.0f} contracts booked (the logged size capped by the contracts "
      f"offered at the price; {sum(1 for r in rows if r[10] <= 0)} signals booked none: depth zero or unknown), "
      f"{len(settled_all)} settled.")
    if not settled:
        return
    per_ct, by_day, dollars = [], defaultdict(list), 0.0
    for ts, ev, t, side, price, size, p, edge, h, res, booked in settled:
        win = 1.0 if res == side else 0.0
        fee1 = math.ceil(round(0.07 * price * (1 - price) * 100, 9)) / 100  # rule fee, per contract
        per_ct.append(win - price - fee1)
        by_day[day(ts)].append(win - price - fee1)
        fee_n = math.ceil(round(0.07 * booked * price * (1 - price) * 10000, 6)) / 10000  # Kalshi: per order
        dollars += booked * (win - price) - fee_n
    mean = sum(per_ct) / len(per_ct)
    keys = list(by_day)
    rng = random.Random(0)
    bs = []
    for _ in range(2000):
        vals = [v for k in (rng.choice(keys) for _ in keys) for v in by_day[k]]
        bs.append(sum(vals) / len(vals))
    bs.sort()
    w(f"Per contract: {100 * mean:+.2f}c (95% interval {100 * bs[50]:+.2f} to {100 * bs[1949]:+.2f}c over "
      f"{len(keys)} days, {len(settled)} settled signals that booked contracts); at booked sizes ${dollars:+.2f}. "
      "Expected from the replay: about +4 to +5c.")
    for side in ("yes", "no"):
        v = [x for x, r in zip(per_ct, settled) if r[3] == side]
        if v:
            w(f"- buy {side.upper()}: {len(v)} trades, {100 * sum(v) / len(v):+.2f}c per contract")


def hourly_shadow_section(c, w):
    """Sleeve H (simulated on the live tape, never traded): tail maker on the hourly Weather Index markets.
    Variant H is the pre-registered market-only rule; H2 adds the nowcast filter (secondary)."""
    try:
        cols = [r[1] for r in c.execute("PRAGMA table_info(hourly_shadow)")]
        vcol = "variant" if "variant" in cols else "'H'"
        ncol = "COALESCE(booked, count)" if "booked" in cols else "count"   # booked: each fill capped by its print
        rows = c.execute(f"SELECT ts, event, ticker, side, price, {ncol}, result, {vcol} FROM hourly_shadow").fetchall()
    except sqlite3.OperationalError:
        return
    w("\n## Hourly tail maker, Sleeve H (simulated fills on the live tape, never traded)\n")
    if not rows:
        w("No simulated fills yet.")
        return
    for variant in sorted({r[7] or "H" for r in rows}):
        vr = [r for r in rows if (r[7] or "H") == variant]
        settled = [r for r in vr if r[6] in ("yes", "no")]
        label = "market-only rule (pre-registered)" if variant == "H" else "with the nowcast filter (secondary)"
        w(f"**{variant}**, {label}: {len(vr)} simulated fills ({sum(r[5] for r in vr):.0f} contracts), "
          f"{len(settled)} settled.")
        if not settled:
            continue
        by_day, by_city, n_ct, wrong = defaultdict(float), defaultdict(float), 0.0, 0.0
        for ts, ev, t, side, price, n, res, _v in settled:
            y = 1.0 if res == "yes" else 0.0
            pnl = n * ((y - price / 100.0) if side == "bid" else (price / 100.0 - y))
            by_day[day(ts)] += pnl
            by_city[ev.split("-")[0]] += pnl
            n_ct += n
            wrong += n * ((y == 0.0) if side == "bid" else (y == 1.0))
        tot = sum(by_day.values())
        w(f"Settled P&L ${tot:+.2f} on {n_ct:.0f} contracts ({100 * tot / max(n_ct, 1):+.2f}c each), "
          f"wrong side on {100 * wrong / max(n_ct, 1):.1f}% of contracts. September replay: "
          f"{'+1.5c a contract, 3.5% wrong side' if variant == 'H' else '+3.1c a contract, 2.1% wrong side'}.")
        w("By day: " + ", ".join(f"{k} ${v:+.2f}" for k, v in sorted(by_day.items())))
        w("By city: " + ", ".join(f"{k} ${v:+.2f}" for k, v in sorted(by_city.items())))


def rain_shadow_section(c, w, table="rain_shadow"):
    """Sleeve R and its R2 variant: the pre-registered rain-market model taker (logged only)."""
    try:
        rows = c.execute("SELECT ts, ticker, side, price, size, fee, p_model, p_pool, mid, edge, h, result, "
                         f"{booked_col(c, table)} FROM {table}").fetchall()
    except sqlite3.OperationalError:
        return
    if table == "rain_shadow":
        w("\n## Rain markets, Sleeve R (model taker, logged only, never traded)\n")
    else:
        w("\n**R2** (secondary variant: version-2 model, all 30 cities):")
    if not rows:
        w("No signals yet.")
        return
    # booked: the logged size capped by the contracts displayed at the price; 0 when that depth was zero or unknown,
    # as for every row logged before Oct 9 (no depth was recorded then)
    rows = [r[:12] + (float(r[12] or 0.0),) for r in rows]
    settled_all = [r for r in rows if r[11] in ("yes", "no")]
    settled = [r for r in settled_all if r[12] > 0]
    w(f"{len(rows)} signals, {sum(r[12] for r in rows):.0f} contracts booked (the logged size capped by the contracts "
      f"offered at the price; {sum(1 for r in rows if r[12] <= 0)} signals booked none: depth zero or not recorded), "
      f"{len(settled_all)} settled.")
    if not settled:
        return
    per_ct, by_day, dollars = [], defaultdict(list), 0.0
    for ts, t, side, price, size, fee, pm, pp, mid, edge, h, res, booked in settled:
        v = (1.0 if res == side else 0.0) - price - fee
        per_ct.append(v)
        by_day[day(ts)].append(v)
        dollars += booked * v
    mean = sum(per_ct) / len(per_ct)
    keys = list(by_day)
    rng = random.Random(0)
    bs = sorted(sum(vals) / len(vals) for vals in
                ([v for k in (rng.choice(keys) for _ in keys) for v in by_day[k]] for _ in range(2000)))
    w(f"Per contract after fees: {100 * mean:+.2f}c (95% interval {100 * bs[50]:+.2f} to {100 * bs[1949]:+.2f}c over "
      f"{len(keys)} days, {len(settled)} settled signals that booked contracts); at booked sizes ${dollars:+.2f}. "
      f"Expected from the exploration replay: about {'+6 to +7c' if table == 'rain_shadow' else '+7 to +8c'}.")


if __name__ == "__main__":
    main()
