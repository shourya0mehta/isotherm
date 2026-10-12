"""Public status page: the model's own forecasts and the bot's track record, nothing else.

Writes data/public/index.html (self-contained, no scripts from anywhere) and data/public/forecasts.json. Safe to serve
as static files (docs/DEPLOY.md): it shows our Tier-1 weather forecast for each city (not the market-pooled
probabilities and no market prices, per Kalshi's developer terms), daily paper/live P&L by sleeve, the pre-registered
scorecard, and the Sleeve X shadow results. No order ids, balances or positions.
"""
from __future__ import annotations

import datetime as dt
import html
import json
import math
import os
import sqlite3
from collections import defaultdict

CSS = """
:root{--bg:#f6f6f3;--card:#fff;--fg:#1c1e21;--mut:#686d73;--line:#dcdedf;--acc:#2a62c9;--pos:#1c7f45;--neg:#bb3a2c}
@media (prefers-color-scheme: dark){:root{--bg:#131517;--card:#1b1e21;--fg:#e9e9e7;--mut:#9ba1a7;--line:#30353a;
--acc:#76a5ff;--pos:#48c27d;--neg:#ff7563}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,Segoe UI,
Roboto,Helvetica,Arial,sans-serif;padding:16px}main{max-width:980px;margin:0 auto}h1{font-size:22px;margin:4px 0}
h2{font-size:16px;margin:0 0 8px}.mut{color:var(--mut)}.card{background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:14px 16px;margin:12px 0;overflow-x:auto}.grid{display:grid;
grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px}.city{border:1px solid var(--line);border-radius:8px;
padding:10px}.city b{font-size:15px}.big{font-size:26px;font-weight:650;font-variant-numeric:tabular-nums}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:5px 8px;
border-bottom:1px solid var(--line);white-space:nowrap}th{font-size:12px;color:var(--mut);font-weight:500}
.pos{color:var(--pos)}.neg{color:var(--neg)}svg{display:block;width:100%;height:auto}.bar{fill:var(--acc)}
.lbl{font-size:9px;fill:var(--mut)}
"""


def forecast_rows(cache: dict, stations: dict, now: dt.datetime) -> list[dict]:
    """cache: Tier1Live cache {(icao, date): {"h", "P", "anchor_r", "t"}} -> plain rows (own model only)."""
    out = []
    for (icao, D), fc in sorted(cache.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        P = [float(x) for x in fc["P"]]
        K = (len(P) - 1) // 2
        temps = [float(fc["anchor_r"]) + j - K for j in range(len(P))]
        cdf, c = [], 0.0
        for p in P:
            c += p
            cdf.append(c)
        q = lambda a: temps[next(i for i, v in enumerate(cdf) if v >= a)]
        st = stations.get(icao, {})
        out.append({"icao": icao, "city": st.get("city", icao), "date": D.isoformat(), "h": int(fc["h"]),
                    "most_likely": temps[max(range(len(P)), key=lambda i: P[i])], "p10": q(0.10), "p90": q(0.90),
                    "temps": temps, "probs": [round(p, 4) for p in P],
                    "made_utc": dt.datetime.fromtimestamp(fc.get("t", now.timestamp()), dt.timezone.utc)
                    .isoformat(timespec="minutes")})
    return out


def _asof(row: dict) -> str:
    t = dt.datetime.fromisoformat(row["date"]) + dt.timedelta(hours=row["h"])
    return t.strftime("%b %d, %H:00 UTC").replace(" 0", " ")


def _spark(row: dict, w: int = 200, h: int = 46) -> str:
    pairs = [(t, p) for t, p in zip(row["temps"], row["probs"]) if p >= 0.004]
    if not pairs:
        return ""
    t0, t1 = pairs[0][0], pairs[-1][0]
    n = int(t1 - t0) + 1
    pm = max(p for _, p in pairs)
    bw = w / max(n, 1)
    parts = [f'<svg viewBox="0 0 {w} {h + 12}" role="img" aria-label="forecast distribution">']
    for t, p in pairs:
        x = (t - t0) * bw
        bh = (h - 2) * p / pm
        parts.append(f'<rect class="bar" x="{x + 1:.1f}" y="{h - bh:.1f}" width="{max(bw - 2, 1):.1f}" height="{bh:.1f}">'
                     f'<title>{t:.0f}°F: {p:.0%}</title></rect>')
    parts.append(f'<text class="lbl" x="0" y="{h + 10}">{t0:.0f}°</text>'
                 f'<text class="lbl" x="{w - 22}" y="{h + 10}">{t1:.0f}°</text></svg>')
    return "".join(parts)


def track_record(db_path: str) -> dict:
    c = sqlite3.connect(db_path, timeout=30)
    res = {t: (r, s) for t, r, s in c.execute("SELECT ticker, result, settled_ts FROM settlements")}
    day = lambda ts: dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d")
    by_day = defaultdict(lambda: defaultdict(float))
    n_fills = defaultdict(int)
    for ts, t, side, pc, n, sl, mode in c.execute("SELECT ts, ticker, side, price_cents, count, sleeve, mode FROM fills"):
        sl = sl or "A"
        if sl == "B":
            continue
        n_fills[(mode, sl)] += 1
        r = res.get(t)
        if r and r[0] in ("yes", "no"):
            cost = n * (pc if side == "bid" else 100 - pc) / 100.0
            win = (side == "bid" and r[0] == "yes") or (side == "ask" and r[0] == "no")
            by_day[(mode, day(r[1] or ts))][sl] += (n if win else 0.0) - cost
    x = []
    try:   # signals that booked nothing (depth zero or unknown: wxbot/store/db.py SHADOW_SIZE_COLS) are no trade
        cols = {r[1] for r in c.execute("PRAGMA table_info(taker_shadow)")}
        bk = "booked" if "booked" in cols else "NULL"
        for ts, side, price, result, size, avail, booked in c.execute(
                f"SELECT ts, side, price, result, size, avail, {bk} FROM taker_shadow"):
            if booked is None:          # a row from before the column: its displayed size was the depth
                booked = min(size, avail) if avail is not None and avail > 0 else 0.0
            if result in ("yes", "no") and booked > 0:
                fee = math.ceil(round(0.07 * price * (1 - price) * 100, 9)) / 100
                x.append((1.0 if result == side else 0.0) - price - fee)
    except sqlite3.OperationalError:
        pass
    n_taker = 0
    try:
        n_taker = c.execute("SELECT COUNT(*) FROM taker_shadow").fetchone()[0]
    except sqlite3.OperationalError:
        pass
    c.close()
    return {"by_day": {f"{m}|{d}": dict(v) for (m, d), v in sorted(by_day.items())},
            "fills": {f"{m}|{s}": n for (m, s), n in n_fills.items()},
            "taker": {"logged": n_taker, "settled": len(x), "cents": (100 * sum(x) / len(x)) if x else None}}


def build(rows: list[dict], rec: dict, status: dict, now: dt.datetime) -> str:
    e = html.escape
    mode = status.get("mode", "paper")
    out = [f"<title>wxbot live</title><style>{CSS}</style><main>",
           "<h1>wxbot: weather forecasts and trading, live</h1>",
           f"<div class='mut'>Updated {now:%Y-%m-%d %H:%M} UTC. Mode: {e(str(mode))}. The forecaster is a gradient-"
           "boosted model over free NOAA and ECMWF data (incl. the 3 km HRRR), predicting the official daily high "
           "at 24 US airports. Method and evidence: docs/ML_RESEARCH.md in the project repository.</div>"]
    days = sorted({r["date"] for r in rows})
    for D in days:
        rs = [r for r in rows if r["date"] == D]
        out.append(f"<div class='card'><h2>Forecast daily highs for {e(D)}</h2><div class='grid'>")
        for r in sorted(rs, key=lambda r: r["city"]):
            out.append(f"<div class='city'><b>{e(r['city'])}</b> <span class='mut'>{e(r['icao'])}</span>"
                       f"<div class='big'>{r['most_likely']:.0f}°F</div><div class='mut'>80% range "
                       f"{r['p10']:.0f}-{r['p90']:.0f}°F · as of {_asof(r)}</div>{_spark(r)}</div>")
        out.append("</div></div>")
    if not rows:
        out.append("<div class='card mut'>No forecasts yet (the bot refreshes at each decision hour).</div>")
    out.append("<div class='card'><h2>Track record (settled, by UTC day)</h2><table><tr><th>Day</th><th>Mode</th>"
               "<th>Sleeve A (near-certain maker)</th><th>Sleeve T (thin books)</th><th>Total</th></tr>")
    tot = 0.0
    for k, v in rec["by_day"].items():
        m, d = k.split("|")
        s = sum(v.values())
        tot += s
        cls = "pos" if s >= 0 else "neg"
        out.append(f"<tr><td>{e(d)}</td><td>{e(m)}</td><td>${v.get('A', 0.0):+.2f}</td><td>${v.get('T', 0.0):+.2f}</td>"
                   f"<td class='{cls}'>${s:+.2f}</td></tr>")
    if not rec["by_day"]:
        out.append("<tr><td colspan=5 class='mut'>Nothing settled yet (daily highs settle the next morning)</td></tr>")
    out.append(f"</table><div class='mut'>Cumulative: ${tot:+.2f}. Paper fills are simulated against the live "
               "public tape with a queue model; live fills are real.</div></div>")
    t = rec["taker"]
    if not status.get("show_shadow", False):
        t = None   # hypothetical trades priced off Kalshi's book: kept off the page until Kalshi approves (Oct 4)
    if t is not None:
        out.append("<div class='card'><h2>Model taker (logged only, never traded)</h2><div>"
                   f"{t['logged']} trades logged, {t['settled']} settled"
                   + (f", {t['cents']:+.2f}¢ per contract after fees" if t["cents"] is not None else "")
                   + ". The rule was frozen on Oct 3, 2026; it trades only if the Oct-Nov holdout and this log both "
                     "come out positive.</div></div>")
    out.append("<p class='mut'>Own forecasts and own results only; no exchange market data is shown.</p></main>")
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            "content='width=device-width,initial-scale=1'>" + out[0] + "</head><body>" + "".join(out[1:]) +
            "</body></html>")


def write(out_dir: str, cache: dict, stations: dict, db_path: str, status: dict) -> str:
    now = dt.datetime.now(dt.timezone.utc)
    os.makedirs(out_dir, exist_ok=True)
    rows = forecast_rows(cache, stations, now)
    rec = track_record(db_path)
    for name, body in (("forecasts.json", json.dumps({"updated_utc": now.isoformat(timespec="minutes"),
                                                      "forecasts": rows}, indent=0)),
                       ("index.html", build(rows, rec, status, now))):
        p = os.path.join(out_dir, name)
        with open(p + ".tmp", "w") as f:
            f.write(body)
        os.replace(p + ".tmp", p)
    return os.path.join(out_dir, "index.html")
