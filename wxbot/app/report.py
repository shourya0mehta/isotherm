"""Text report from the SQLite store: fills, realized P&L on settled markets, mark-to-market, Sleeve B estimate."""
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
from collections import defaultdict


def build_report(cfg: dict, db_path: str = None) -> str:
    db = sqlite3.connect(db_path or cfg["db_path"])
    out = []
    now = dt.datetime.now(dt.timezone.utc)
    out.append(f"wxbot report  {now:%Y-%m-%d %H:%M} UTC   mode={cfg.get('mode')}")
    try:
        st = json.load(open(cfg.get("status_path", "data/status.json")))
        out.append(f"status: uptime {st['uptime_min']} min | events {st['events']} | quotes {st['quotes']} | "
                   f"fills {st['fills_total']} | open risk ${st['open_risk_usd']} | MTM ${st['mtm_usd']} | "
                   f"realized ${st['realized_usd']} | halted={st['halted']}")
    except Exception:
        pass
    fills = db.execute("SELECT ts,ticker,event,side,price_cents,count,sleeve,fair_yes,fee FROM fills "
                       "ORDER BY ts").fetchall()
    settled = {r[0]: r[1] for r in db.execute("SELECT ticker,result FROM settlements")}
    pos = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])  # yes_qty, yes_cost, no_qty, no_cost, fees
    edge_sum = 0.0
    for ts, t, ev, side, pc, n, sl, fair, fee in fills:
        pos[t][4] += fee or 0.0
        if side == "bid":
            pos[t][0] += n
            pos[t][1] += n * pc / 100
            if fair is not None:
                edge_sum += n * (fair - pc / 100)
        else:
            pos[t][2] += n
            pos[t][3] += n * (100 - pc) / 100
            if fair is not None:
                edge_sum += n * ((1 - fair) - (100 - pc) / 100)
    # after fees (outside review, Oct 8: the fees recorded with the fills were left out of the realized figure)
    realized, n_settled, fees = 0.0, 0, 0.0
    for t, (yq, yc, nq, nc, fe) in pos.items():
        if t in settled:
            r = settled[t]
            realized += (yq if r == "yes" else 0) + (nq if r == "no" else 0) - yc - nc - fe
            fees += fe
            n_settled += 1
    tot_contracts = sum(r[5] for r in fills)
    out.append(f"fills: {len(fills)} ({tot_contracts:.0f} contracts) across {len(pos)} markets; "
               f"settled markets {n_settled}; realized P&L ${realized:.2f} after ${fees:.2f} fees")
    if tot_contracts:
        out.append(f"model edge at fill (fair - price): {100 * edge_sum / tot_contracts:+.2f} c/contract (expected, not realized)")
    by_ev = defaultdict(float)
    for ts, t, ev, side, pc, n, sl, fair, fee in fills:
        by_ev[ev] += n
    if by_ev:
        out.append("contracts by event: " + ", ".join(f"{k}:{v:.0f}" for k, v in sorted(by_ev.items())))
    lip = db.execute("SELECT COUNT(*), SUM(est_usd), SUM(counts) FROM lip_snaps").fetchone()
    if lip and lip[0]:
        out.append(f"Sleeve B shadow: {lip[0]} quote-snapshots, est rewards ${lip[1] or 0:.2f} "
                   f"(would-qualify snapshots {lip[2]}) -- NOT traded")
    errs = db.execute("SELECT ts,msg FROM events_log WHERE level IN ('ERROR','WARN') ORDER BY ts DESC LIMIT 8").fetchall()
    if errs:
        out.append("recent errors and warnings:")
        out += [f"  {dt.datetime.fromtimestamp(t, dt.timezone.utc):%H:%M:%S} {m[:160]}" for t, m in errs]
    return "\n".join(out)


def gate_report(cfg: dict, db_path: str = None) -> tuple[bool, str]:
    """Mechanical go/no-go for the next step: paper -> live (half caps), or half caps -> full caps.
    Uses settled contracts only (realized, not marked)."""
    import time as _t
    mode = cfg.get("mode", "paper")
    db = sqlite3.connect(db_path or cfg["db_path"])
    first = db.execute("SELECT MIN(ts) FROM events_log WHERE msg LIKE 'start mode%'").fetchone()[0]
    days = (_t.time() - first) / 86400 if first else 0.0
    fills = db.execute("SELECT ticker,side,price_cents,count FROM fills WHERE mode=? AND sleeve!='B'", (mode,)).fetchall()
    settled = {t: r for t, r in db.execute("SELECT ticker,result FROM settlements")}
    n_set, pnl = 0.0, 0.0
    for t, side, pc, n in fills:
        if t not in settled:
            continue
        win = (settled[t] == "yes") == (side == "bid")
        pnl += n * ((100 if win else 0) - (pc if side == "bid" else 100 - pc)) / 100
        n_set += n
    edge = 100 * pnl / n_set if n_set else 0.0
    hard = db.execute("SELECT COUNT(*) FROM events_log WHERE msg LIKE 'HALT: drawdown%'").fetchone()[0]
    errs = db.execute("SELECT COUNT(*) FROM events_log WHERE level='ERROR' AND msg LIKE 'step failed%' AND ts>?",
                      (_t.time() - 86400,)).fetchone()[0]
    if mode == "paper":
        need_days, need_ct, need_edge, label = 3.0, 300, 0.5, "paper -> live at half caps (config/live_start.yaml)"
    else:
        need_days, need_ct, need_edge, label = 6.0, 500, 0.0, "half caps -> full caps (config/settings.yaml, mode: live)"
    checks = [
        (days >= need_days, f"running {days:.1f} days (need {need_days:g})"),
        (n_set >= need_ct, f"{n_set:.0f} settled contracts (need {need_ct})"),
        (n_set > 0 and edge >= need_edge, f"realized edge {edge:+.2f}c/contract (need >= {need_edge:+.1f}c; replay ~+2c)"),
        (pnl > -20, f"realized P&L ${pnl:+.2f} (need > -$20)"),
        (hard == 0, f"{hard} drawdown halts (need 0)"),
        (errs < 5, f"{errs} step failures in the last 24 h (need < 5)"),
    ]
    ok = all(c for c, _ in checks)
    lines = [f"GATE {label}: {'PASS' if ok else 'NOT YET'}"] + [f"  [{'OK ' if c else '-- '}] {m}" for c, m in checks]
    return ok, "\n".join(lines)
