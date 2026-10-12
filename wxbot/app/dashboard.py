"""Self-contained HTML dashboard from the SQLite store (no server, no external scripts).

python -m wxbot dashboard [--config ...]   -> writes data/<db name>_dashboard.html
The running bot also refreshes it every `dashboard_every_s` (default 300 s). Open the file in a browser.
"""
from __future__ import annotations

import datetime as dt
import html
import json
import os
import sqlite3
from collections import defaultdict

from ..store.db import HOURLY_BOOKED_SQL, RAIN_BOOKED_SQL, TAKER_BOOKED_SQL
from ..timeutil import to_et


def _booked(db, table: str, expr: str, before: str) -> str:
    """SQL for a shadow row's booked size (wxbot/store/db.py SHADOW_SIZE_COLS); `before`: for a database the bot has
    not opened since the booked column was added."""
    return expr if "booked" in {r[1] for r in db.execute(f"PRAGMA table_info({table})")} else before


def _et_day(ts: float) -> str:
    return to_et(ts).date().isoformat()


def _fmt_ts(ts: float) -> str:
    return to_et(ts).strftime("%m-%d %H:%M ET")


def _opt(x, fmt: str) -> str:
    return "" if x is None else fmt.format(x)


def _rule_fee(price: float) -> float:
    import math
    return math.ceil(round(0.07 * price * (1 - price) * 100, 9)) / 100


def _money(x: float) -> str:
    return f"{'-' if x < 0 else ''}${abs(x):,.2f}"


def collect(db_path: str, mode: str | None = None) -> dict:
    db = sqlite3.connect(db_path)
    q = "SELECT ts,ticker,event,side,price_cents,count,sleeve,fair_yes,fee FROM fills"
    rows = db.execute(q + (" WHERE mode=?" if mode else "") + " ORDER BY ts", (mode,) if mode else ()).fetchall()
    settled = {t: (r, ts) for t, r, ts in db.execute("SELECT ticker,result,settled_ts FROM settlements")}
    pos = defaultdict(lambda: {"yq": 0.0, "yc": 0.0, "nq": 0.0, "nc": 0.0, "fee": 0.0, "event": "", "first": 0.0})
    fills = []
    for ts, t, ev, side, pc, n, sl, fair, fee in rows:
        if sl == "B":
            continue
        p = pos[t]
        p["event"], p["first"] = ev, p["first"] or ts
        if side == "bid":
            p["yq"] += n
            p["yc"] += n * pc / 100
            edge = (fair - pc / 100) * 100 if fair is not None else None
            pay = pc
        else:
            p["nq"] += n
            p["nc"] += n * (100 - pc) / 100
            edge = ((1 - fair) - (100 - pc) / 100) * 100 if fair is not None else None
            pay = 100 - pc
        p["fee"] += fee or 0.0
        fills.append({"ts": ts, "ticker": t, "buy": "YES" if side == "bid" else "NO", "pay": pay, "n": n,
                      "fair_win": None if fair is None else (fair if side == "bid" else 1 - fair), "edge": edge})
    daily = defaultdict(float)
    settled_rows, open_rows = [], []
    contracts_settled, pnl_settled = 0.0, 0.0
    calib = defaultdict(lambda: [0.0, 0.0])  # bucket -> [contracts, wins]
    for t, p in pos.items():
        cost = p["yc"] + p["nc"] + p["fee"]
        if t in settled:
            res, sts = settled[t]
            pnl = (p["yq"] if res == "yes" else 0) + (p["nq"] if res == "no" else 0) - cost
            daily[_et_day(sts or p["first"])] += pnl
            settled_rows.append({"ticker": t, "result": res, "yes": p["yq"], "no": p["nq"], "cost": cost, "pnl": pnl,
                                 "ts": sts or p["first"]})
            contracts_settled += p["yq"] + p["nq"]
            pnl_settled += pnl
        else:
            open_rows.append({"ticker": t, "yes": p["yq"], "no": p["nq"], "cost": cost})
    for f in fills:
        if f["ticker"] in settled and f["fair_win"] is not None:
            res = settled[f["ticker"]][0]
            won = (res == "yes") == (f["buy"] == "YES")
            b = min(int(f["fair_win"] * 20), 19) / 20  # 5% buckets
            calib[b][0] += f["n"]
            calib[b][1] += f["n"] if won else 0.0
    fv = {}
    for t, fair in db.execute("SELECT ticker, fair FROM fair_values WHERE ts > (SELECT MAX(ts) - 900 FROM fair_values)"):
        fv[t] = fair
    for r in open_rows:
        f = fv.get(r["ticker"])
        r["fair_yes"] = f
        r["mtm"] = None if f is None else r["yes"] * f + r["no"] * (1 - f) - r["cost"]
    # shadow logs at the booked size: the logged size capped by the contracts offered at the price, 0 when that depth
    # was zero or unknown (outside review, Oct 8); a signal that booked nothing is no trade in the averages
    taker = []
    try:
        bk = _booked(db, "taker_shadow", TAKER_BOOKED_SQL, "CASE WHEN avail > 0 THEN MIN(size, avail) ELSE 0 END")
        taker = db.execute(f"SELECT ts, ticker, side, price, size, p, edge, h, result, {bk} FROM taker_shadow "
                           "ORDER BY ts DESC").fetchall()
    except sqlite3.OperationalError:
        pass
    hourly = {}
    try:   # Sleeve H shadow: simulated fills on the live tape, settled hourly
        n = _booked(db, "hourly_shadow", HOURLY_BOOKED_SQL, "count")
        for var, side, price, count, res in db.execute(f"SELECT variant, side, price, {n}, result FROM hourly_shadow"):
            h = hourly.setdefault(var or "H", {"fills": 0, "contracts": 0.0, "settled": 0.0, "pnl": 0.0})
            h["fills"] += 1
            h["contracts"] += count
            if res in ("yes", "no"):
                y = 1.0 if res == "yes" else 0.0
                h["settled"] += count
                h["pnl"] += count * ((y - price / 100.0) if side == "bid" else (price / 100.0 - y))
    except sqlite3.OperationalError:
        pass
    rain = {}
    for table, label in (("rain_shadow", "R"), ("rain_shadow_v2", "R2")):
        try:
            bk = _booked(db, table, RAIN_BOOKED_SQL, "0")
            rain[label] = db.execute(f"SELECT ts, ticker, side, price, size, fee, p_model, p_pool, mid, edge, h, result, "
                                     f"{bk} FROM {table} ORDER BY ts DESC").fetchall()
        except sqlite3.OperationalError:
            pass
    lip = db.execute("SELECT COUNT(*), COALESCE(SUM(est_usd),0), MIN(ts), MAX(ts) FROM lip_snaps").fetchone()
    errs = db.execute("SELECT ts,msg FROM events_log WHERE level IN ('ERROR','WARN') ORDER BY ts DESC LIMIT 8").fetchall()
    n_contracts = sum(f["n"] for f in fills)
    edge_w = sum(f["edge"] * f["n"] for f in fills if f["edge"] is not None)
    return {
        "fills": fills, "daily": dict(sorted(daily.items())), "settled": sorted(settled_rows, key=lambda r: -r["ts"]),
        "open": sorted(open_rows, key=lambda r: r["ticker"]), "calib": dict(sorted(calib.items())),
        "n_contracts": n_contracts, "model_edge_c": (edge_w / n_contracts) if n_contracts else None,
        "realized": pnl_settled, "contracts_settled": contracts_settled,
        "realized_edge_c": (100 * pnl_settled / contracts_settled) if contracts_settled else None,
        "lip": {"n": lip[0], "usd": lip[1], "hours": ((lip[3] or 0) - (lip[2] or 0)) / 3600 if lip[0] else 0},
        "errors": errs, "taker": taker, "hourly": hourly, "rain": rain,
    }


def _bar_svg(daily: dict, w: int = 640, h: int = 180) -> str:
    if not daily:
        return '<p class="muted">No settled days yet.</p>'
    days = list(daily)
    vals = [daily[d] for d in days]
    cum, c = [], 0.0
    for v in vals:
        c += v
        cum.append(c)
    top = max([0.0] + vals + cum)
    bot = min([0.0] + vals + cum)
    span = (top - bot) or 1.0
    pad = 28
    y = lambda v: pad / 2 + (top - v) / span * (h - pad)
    bw = (w - 40) / max(len(days), 1)
    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="Daily P&L">',
             f'<line x1="36" x2="{w}" y1="{y(0):.1f}" y2="{y(0):.1f}" class="axis"/>',
             f'<text x="0" y="{y(top) + 4:.1f}" class="lbl">{_money(top)}</text>',
             f'<text x="0" y="{y(bot) + 4:.1f}" class="lbl">{_money(bot)}</text>']
    for i, (d, v) in enumerate(zip(days, vals)):
        x = 38 + i * bw
        y0, y1 = sorted((y(0), y(v)))
        cls = "pos" if v >= 0 else "neg"
        parts.append(f'<rect x="{x:.1f}" y="{y0:.1f}" width="{max(bw - 3, 1):.1f}" height="{max(y1 - y0, 1):.1f}" '
                     f'class="{cls}"><title>{d}: {_money(v)}</title></rect>')
    pts = " ".join(f"{38 + i * bw + bw / 2:.1f},{y(v):.1f}" for i, v in enumerate(cum))
    parts.append(f'<polyline points="{pts}" class="cum"/>')
    parts.append(f'<text x="38" y="{h - 2}" class="lbl">{days[0]}</text>')
    parts.append(f'<text x="{w - 70}" y="{h - 2}" class="lbl">{days[-1]}</text></svg>')
    return "".join(parts)


def _calib_svg(calib: dict, w: int = 300, h: int = 300) -> str:
    pts = [(b + 0.025, wins / n, n) for b, (n, wins) in calib.items() if n > 0]
    if not pts:
        return '<p class="muted">Needs settled fills.</p>'
    lo = min(0.75, min(p[0] for p in pts) - 0.05)
    sx = lambda v: 30 + (v - lo) / (1 - lo) * (w - 40)
    sy = lambda v: h - 30 - (v - lo) / (1 - lo) * (h - 40)
    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="Calibration">',
             f'<line x1="{sx(lo)}" y1="{sy(lo)}" x2="{sx(1)}" y2="{sy(1)}" class="diag"/>',
             f'<text x="{sx(lo)}" y="{h - 12}" class="lbl">{lo:.2f}</text><text x="{sx(1) - 20}" y="{h - 12}" '
             f'class="lbl">1.00</text><text x="{w / 2 - 40}" y="{h - 1}" class="lbl">fair win prob at fill</text>']
    nmax = max(p[2] for p in pts)
    for x, yv, n in pts:
        r = 3 + 7 * (n / nmax) ** 0.5
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(max(yv, lo)):.1f}" r="{r:.1f}" class="dot">'
                     f'<title>fair {x:.3f}: won {yv:.1%} of {n:.0f} contracts</title></circle>')
    parts.append("</svg>")
    return "".join(parts)


CSS = """
:root{--bg:#f7f7f5;--card:#fff;--fg:#1d1f21;--mut:#6b7075;--line:#d9dbdd;--pos:#1f8a4c;--neg:#c0392b;--acc:#2d6cdf}
@media (prefers-color-scheme: dark){:root{--bg:#141618;--card:#1d2023;--fg:#e8e8e6;--mut:#9aa0a6;--line:#33373b;
--pos:#3fbf73;--neg:#ff6b5b;--acc:#6ea0ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,Segoe UI,
Roboto,Helvetica,Arial,sans-serif;padding:16px}main{max-width:1060px;margin:0 auto}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:0 0 10px}.muted{color:var(--mut)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:12px;
overflow-x:auto}.k{font-size:12px;color:var(--mut)}.v{font-size:20px;font-weight:600;font-variant-numeric:tabular-nums}
.pos-t{color:var(--pos)}.neg-t{color:var(--neg)}table{border-collapse:collapse;width:100%;font-variant-numeric:
tabular-nums}th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th{font-size:12px;color:var(--mut);font-weight:500}.two{display:grid;grid-template-columns:2fr 1fr;gap:12px}
@media (max-width:760px){.two{grid-template-columns:1fr}}
.chart{width:100%;height:auto}.axis{stroke:var(--line)}.diag{stroke:var(--mut);stroke-dasharray:4 4}
.pos{fill:var(--pos)}.neg{fill:var(--neg)}.cum{fill:none;stroke:var(--acc);stroke-width:2}
.dot{fill:var(--acc);fill-opacity:.7}.lbl{font-size:10px;fill:var(--mut)}.badge{display:inline-block;padding:2px 8px;
border-radius:99px;font-size:12px;border:1px solid var(--line)}
"""


def build_html(cfg: dict, db_path: str | None = None, status_path: str | None = None) -> str:
    db_path = db_path or cfg["db_path"]
    d = collect(db_path)
    try:
        st = json.load(open(status_path or cfg.get("status_path", "data/status.json")))
    except Exception:
        st = {}
    e = html.escape
    now = dt.datetime.now(dt.timezone.utc)
    mode = st.get("mode", cfg.get("mode", "?"))
    halted = st.get("halted")
    sig = lambda x: "pos-t" if x >= 0 else "neg-t"
    tiles = [
        ("Realized P&L", _money(d["realized"]), sig(d["realized"])),
        ("Open MTM", _money(st.get("mtm_usd", 0.0)), sig(st.get("mtm_usd", 0.0))),
        ("Open risk", _money(st.get("open_risk_usd", 0.0)), ""),
        ("Contracts filled", f"{d['n_contracts']:,.0f}", ""),
        ("Model edge at fill", "n/a" if d["model_edge_c"] is None else f"{d['model_edge_c']:+.2f}c", ""),
        ("Realized edge", "n/a" if d["realized_edge_c"] is None else f"{d['realized_edge_c']:+.2f}c",
         "" if d["realized_edge_c"] is None else sig(d["realized_edge_c"])),
    ]
    out = [f"<title>wxbot dashboard</title><style>{CSS}</style><main>",
           f"<h1>wxbot <span class='badge'>{e(str(mode))}</span> "
           f"<span class='badge {'neg-t' if halted else 'pos-t'}'>{'HALTED' if halted else 'running'}</span></h1>",
           f"<div class='muted'>Updated {now:%Y-%m-%d %H:%M} UTC. Uptime {st.get('uptime_min', '?')} min. "
           f"{st.get('events', '?')} events, {st.get('quotes', '?')} live quotes. "
           f"{e(st.get('halt_reason') or '')}</div>"]
    tm = st.get("timing") or {}
    if tm.get("book_age_at_orders_s") or tm.get("api_round_trip_ms"):
        ba, rt = tm.get("book_age_at_orders_s") or {}, tm.get("api_round_trip_ms") or {}
        parts = []
        if ba:
            parts.append(f"order book age when orders update: median {ba['p50']:.0f} s, 95% under {ba['p95']:.0f} s")
        if tm.get("step_s"):
            parts.append(f"loop step median {tm['step_s']['p50']:.1f} s")
        for k in ("read", "write"):
            if rt.get(k):
                parts.append(f"API {k} round trip median {rt[k]['p50']} ms (95%: {rt[k]['p95']} ms)")
        out.append(f"<div class='muted'>Speed: {'; '.join(parts)}.</div>")
    out.append("<div class='grid'>")
    out += [f"<div class='card'><div class='k'>{k}</div><div class='v {c}'>{v}</div></div>" for k, v, c in tiles]
    out.append("</div><div class='two'><div class='card'><h2>Daily realized P&L (bars) and cumulative (line)</h2>"
               + _bar_svg(d["daily"]) + "</div><div class='card'><h2>Calibration of filled contracts</h2>"
               + _calib_svg(d["calib"]) + "</div></div>")
    out.append("<div class='card'><h2>Open positions</h2><table><tr><th>Market</th><th>YES</th><th>NO</th>"
               "<th>Cost</th><th>Fair YES</th><th>MTM</th></tr>")
    for r in d["open"]:
        out.append(f"<tr><td>{e(r['ticker'])}</td><td>{r['yes']:.0f}</td><td>{r['no']:.0f}</td>"
                   f"<td>{_money(r['cost'])}</td><td>{_opt(r['fair_yes'], '{:.3f}')}</td>"
                   f"<td class='{'' if r['mtm'] is None else sig(r['mtm'])}'>"
                   f"{'' if r['mtm'] is None else _money(r['mtm'])}</td></tr>")
    if not d["open"]:
        out.append("<tr><td colspan=6 class='muted'>None</td></tr>")
    out.append("</table></div><div class='card'><h2>Recent fills</h2><table><tr><th>Time</th><th>Market</th>"
               "<th>Bought</th><th>Paid</th><th>Qty</th><th>Fair win</th><th>Edge</th></tr>")
    for f in list(reversed(d["fills"]))[:60]:
        out.append(f"<tr><td>{_fmt_ts(f['ts'])}</td><td>{e(f['ticker'])}</td><td>{f['buy']}</td><td>{f['pay']}c</td>"
                   f"<td>{f['n']:.0f}</td><td>{_opt(f['fair_win'], '{:.3f}')}</td>"
                   f"<td>{_opt(f['edge'], '{:+.1f}c')}</td></tr>")
    if not d["fills"]:
        out.append("<tr><td colspan=7 class='muted'>No fills yet</td></tr>")
    out.append("</table></div><div class='card'><h2>Settled markets</h2><table><tr><th>Settled</th><th>Market</th>"
               "<th>Result</th><th>YES</th><th>NO</th><th>Cost</th><th>P&L</th></tr>")
    for r in d["settled"][:60]:
        out.append(f"<tr><td>{_fmt_ts(r['ts'])}</td><td>{e(r['ticker'])}</td><td>{r['result']}</td><td>{r['yes']:.0f}"
                   f"</td><td>{r['no']:.0f}</td><td>{_money(r['cost'])}</td><td class='{sig(r['pnl'])}'>"
                   f"{_money(r['pnl'])}</td></tr>")
    if not d["settled"]:
        out.append("<tr><td colspan=7 class='muted'>Nothing settled yet (daily highs settle the next morning)</td></tr>")
    out.append("</table></div>")
    lip = d["lip"]
    if lip["n"]:
        rate = lip["usd"] / lip["hours"] if lip["hours"] > 0 else 0
        out.append(f"<div class='card'><h2>Sleeve B (shadow only, not traded)</h2><div>{lip['n']:,} quote snapshots "
                   f"over {lip['hours']:.1f} h, estimated reward {_money(lip['usd'])} ({_money(rate)}/h) if the "
                   f"filed scoring formula pays as written. Unverified until Kalshi support confirms.</div></div>")
    if d["taker"]:
        done = [r for r in d["taker"] if r[8] in ("yes", "no")]
        pc = [(1.0 if r[8] == r[2] else 0.0) - r[3] - _rule_fee(r[3]) for r in done if (r[9] or 0) > 0]
        head = (f"{len(d['taker'])} logged, {len(done)} settled, "
                f"{sum(1 for r in d['taker'] if not (r[9] or 0) > 0)} booked nothing (depth zero or unknown)" +
                (f", {100 * sum(pc) / len(pc):+.2f}c per booked contract after fees" if pc else ""))
        x_real = mode == "live" and bool((cfg.get("taker_shadow") or {}).get("trade"))
        x_title = ("Sleeve X, model taker (signals; real orders on, see fills above)" if x_real else
                   "Sleeve X, model taker (logged only, never traded)")
        out.append(f"<div class='card'><h2>{x_title}</h2><div class='muted'>"
                   f"{head}. Replay expectation about +4 to +5c.</div><table><tr><th>Time</th><th>Market</th>"
                   "<th>Buy</th><th>Price</th><th>Booked</th><th>Model</th><th>Edge</th><th>h</th><th>Result</th></tr>")
        for ts, t, side, price, size, p, edge, h, res, booked in d["taker"][:40]:
            out.append(f"<tr><td>{_fmt_ts(ts)}</td><td>{e(t)}</td><td>{side.upper()}</td><td>{100 * price:.0f}c</td>"
                       f"<td>{booked or 0:g}</td><td>{p:.2f}</td><td>{100 * edge:+.1f}c</td><td>{h}</td>"
                       f"<td>{e(res or 'open')}</td></tr>")
        out.append("</table></div>")
    hb = st.get("sleeve_h_breaker") or {}
    h_real = bool(hb.get("real_orders_on") or hb.get("tripped"))
    if d.get("hourly") or h_real:
        out.append("<div class='card'><h2>Sleeve H, hourly index tail maker" +
                   (" (shadow below; real orders: see breaker line)" if h_real else
                    " (simulated fills, never traded)") + "</h2>")
        if h_real:
            hs, lim = hb.get("stats") or {}, hb.get("limits") or {}
            state = ("<span class='neg-t'>SWITCHED OFF: " + e(str(hb.get("reason"))) + "</span>") if hb.get("tripped") \
                else "<span class='pos-t'>on</span>"
            line = (f"Real hourly orders {state}. Settled real contracts {hs.get('contracts', 0):.0f}, lost "
                    f"{_opt(None if hs.get('lost_share') is None else 100 * hs['lost_share'], '{:.1f}%') or 'n/a'}, "
                    f"P&L {_opt(hs.get('pnl_c_per_contract'), '{:+.2f}c') or 'n/a'} per contract. Breaker: after "
                    f"{lim.get('min_contracts', 300)} contracts, off if more than "
                    f"{100 * lim.get('max_lost_share', 0.06):.0f}% lost or P&L per contract below "
                    f"{lim.get('min_pnl_c_per_contract', 0.0):+.1f}c.")
            out.append(f"<div>{line}</div>")
        out.append("<table><tr><th>Variant</th><th>Simulated fills</th><th>Contracts</th><th>Settled</th>"
                   "<th>Settled P&L</th><th>Per contract</th></tr>")
        for var in sorted(d["hourly"]):
            h = d["hourly"][var]
            pc = 100 * h["pnl"] / h["settled"] if h["settled"] else None
            out.append(f"<tr><td>{e(var)}{' (nowcast filter)' if var == 'H2' else ' (pre-registered)'}</td>"
                       f"<td>{h['fills']}</td><td>{h['contracts']:.0f}</td><td>{h['settled']:.0f}</td>"
                       f"<td class='{sig(h['pnl'])}'>{_money(h['pnl'])}</td><td>{_opt(pc, '{:+.2f}c')}</td></tr>")
        out.append("</table><div class='muted'>September replay: about +1.5c a contract (H), +3c (H2).</div></div>")
    for label, rows in (d.get("rain") or {}).items():
        if not rows:
            continue
        done = [r for r in rows if r[11] in ("yes", "no")]
        pc = [(1.0 if r[11] == r[2] else 0.0) - r[3] - r[5] for r in done if (r[12] or 0) > 0]
        head = (f"{len(rows)} logged, {len(done)} settled, "
                f"{sum(1 for r in rows if not (r[12] or 0) > 0)} booked nothing (depth zero or not recorded)" +
                (f", {100 * sum(pc) / len(pc):+.2f}c per booked contract after fees" if pc else ""))
        name = "Sleeve R, rain markets model taker" if label == "R" else "R2, rain variant (all 30 cities)"
        out.append(f"<div class='card'><h2>{name} (logged only, never traded)</h2><div class='muted'>{head}. "
                   "Replay expectation about +6 to +8c.</div><table><tr><th>Time</th><th>Market</th><th>Buy</th>"
                   "<th>Price</th><th>Booked</th><th>Model</th><th>Pool</th><th>Edge</th><th>h</th><th>Result</th></tr>")
        for ts, t, side, price, size, fee, pm, pp, mid, edge, h, res, booked in rows[:40]:
            out.append(f"<tr><td>{_fmt_ts(ts)}</td><td>{e(t)}</td><td>{side.upper()}</td><td>{100 * price:.0f}c</td>"
                       f"<td>{booked or 0:g}</td><td>{pm:.2f}</td><td>{pp:.2f}</td><td>{100 * edge:+.1f}c</td>"
                       f"<td>{h}</td><td>{e(res or 'open')}</td></tr>")
        out.append("</table></div>")
    if d["errors"]:
        out.append("<div class='card'><h2>Recent errors and warnings</h2><table>")
        out += [f"<tr><td>{_fmt_ts(t)}</td><td>{e(m[:180])}</td></tr>" for t, m in d["errors"]]
        out.append("</table></div>")
    out.append("<p class='muted'>Local file generated by wxbot. Contains your own trading data; do not publish "
               "Kalshi market data.</p></main>")
    return "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' " \
           "content='width=device-width,initial-scale=1'>" + "".join(out[:1]) + "</head><body>" + \
           "".join(out[1:]) + "</body></html>"


def write_dashboard(cfg: dict, out_path: str | None = None, base_dir: str = ".") -> str:
    db = os.path.join(base_dir, cfg["db_path"])
    st = os.path.join(base_dir, cfg.get("status_path", "data/status.json"))
    out_path = out_path or os.path.splitext(db)[0] + "_dashboard.html"
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        f.write(build_html(cfg, db, st))
    os.replace(tmp, out_path)
    return out_path
