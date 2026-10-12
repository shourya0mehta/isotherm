"""Stage 2 gate (docs/RUNBOOK.md): after two clean trading days at Stage 1, is the bot ready for half size and the
taker? Reads the live bot's log, database and status file; with the API key in the environment (GO_LIVE puts it in
.env, which this script loads), it also reads the account, read-only, to check what only Kalshi knows. Prints each
criterion as PASS / FAIL / CHECK and exits 0 only when nothing failed.

Usage (on the Mac, from the repo folder):  .venv/bin/python scripts/stage2_gate.py [--no-api] [--since YYYY-MM-DD]
       [--config config/x.yaml]  (another bot's record: its database, log and status, and the account on its exchange;
       scripts/demo_gate.py uses it for the two-day demo test)
(--since: judge only the record from that ET day on, e.g. after an aborted first attempt)

1. two ET days with at least six hours of live quoting each, and no halt other than the daily loss stop (the brake
   working) or a stop by hand; step failures make the verdict CHECK;
2. every placement without an answer resolved (adopted or gone) within 60 s;
3. every unconfirmed cancel resolved (confirmed on a retry, or gone from the snapshots) within 60 s;
   (2 and 3: exchange outages are left out, since nothing settles while the exchange is down: from the first
   "service unavailable" answer in the 3 minutes before the bot logged a pause to the end of that pause. The Oct 6
   demo test saw eight such outages, each with 503s for up to a minute before Kalshi's status said paused)
4. no duplicate orders: never two bot orders resting together on one market and side, unless the later one is a
   top-up (the earlier one was at least half filled when it went out);
5. the bot's fills are exactly Kalshi's fills for the bot's orders (same fill ids, prices and counts);
6. the bot's settled equity is within $1 of cash + held + positions at cost read fresh from Kalshi.
Items 4-6 need the API (--no-api skips them, which makes the verdict CHECK, not PASS)."""
import datetime as dt
import json
import os
import re
import sqlite3
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

LOG, DB, STATUS = "data/live.log", "data/live.sqlite", "data/live_status.json"
LINE_TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3}) ")


def parse_log_ts(line: str):
    m = LINE_TS.match(line)
    if not m:
        return None
    # the log is written in the Mac's local time
    return time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")) + int(m.group(2)) / 1000.0


def load_env(path=".env"):
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def scan_log(path: str, since: float):
    """Resolution times of uncertain placements and unconfirmed cancels, from the order manager's log lines."""
    out = {"uncertain_started": [], "uncertain_resolved": [], "pending_resolved": [], "pending_batches": 0,
           "lines": 0, "filled_on_arrival": 0, "pauses": []}
    if not os.path.exists(path):
        return out
    paused = None
    unavailable = []                       # when Kalshi answered "service unavailable" (503), for the outage starts
    for line in open(path, errors="replace"):
        ts = parse_log_ts(line)
        if ts is None or ts < since:
            continue
        out["lines"] += 1
        if "service_unavailable" in line or "-> 503" in line:
            unavailable.append(ts)
        # exchange outages as the bot saw them: PAUSE start / PAUSE end lines, each reaching back to the first
        # "service unavailable" answer in the 3 minutes before the bot noticed the pause (it reads Kalshi's status
        # once a minute, while its orders already get 503s: the Oct 6 demo outages). A restart in the middle of one
        # closes it (the new process says so again if Kalshi is still paused)
        if "PAUSE start:" in line:
            lead = [t for t in unavailable if ts - 180 <= t <= ts]
            paused = (min(lead) if lead else ts) if paused is None else paused
        elif "PAUSE end after=" in line:
            m = re.search(r"PAUSE end after=([\d.]+)s", line)
            out["pauses"].append((paused if paused is not None else ts - float(m.group(1) if m else 0), ts))
            paused = None
        elif " start mode=" in line and paused is not None:
            out["pauses"].append((paused, ts))
            paused = None
        if "filled" in line and "on arrival" in line and "post-only order" in line:
            out["filled_on_arrival"] += 1
        if "has no answer: kept as uncertain" in line:
            m = re.search(r"cid=([\w-]+)", line)
            out["uncertain_started"].append((ts, m.group(1) if m else None))
        m = re.search(r"RESOLVED (uncertain|cancel_pending) (\w+) after=([\d.]+)s cid=([\w-]*)", line)
        if m:
            key = "uncertain_resolved" if m.group(1) == "uncertain" else "pending_resolved"
            out[key].append((ts, m.group(2), float(m.group(3)), m.group(4)))
        if "PENDING cids=" in line:
            out["pending_batches"] += 1
            for cid in line.rsplit("PENDING cids=", 1)[1].strip().split(","):
                if cid:
                    out.setdefault("pending_started", []).append((ts, cid))
    out["paused_now"] = paused          # still paused at the end of the log: not left out (run the gate again after)
    return out


def paused_within(pauses, t0: float, t1: float) -> float:
    """Seconds of [t0, t1] the bot logged the exchange as paused."""
    return sum(max(0.0, min(b, t1) - max(a, t0)) for a, b in pauses)


def paths_from_config(path: str):
    """(log, database, status, env) of the bot a config describes: the log sits beside its database."""
    import yaml
    cfg = yaml.safe_load(open(path)) or {}
    db = cfg.get("db_path", DB)
    return os.path.splitext(db)[0] + ".log", db, cfg.get("status_path", STATUS), cfg.get("env", "prod")


def iso(s):
    from wxbot.timeutil import parse_ts
    return parse_ts(s) if s else None


def side_of(o: dict):
    return o.get("book_side") or {"no": "ask", "yes": "bid"}.get(o.get("outcome_side"))


def main(argv=None, more_items=None, label="STAGE 2 GATE", on_verdict=None, start_ts=None, facts=None):
    """more_items(c, start, res): more items (scripts/demo_gate.py); on_verdict(verdict, results, start): called
    before the exit with PASS, CHECK or FAIL, and may return the final verdict (the exit code follows it); start_ts: judge the record from this start on (the demo gate: the run
    of the code GO_LIVE will run); facts: a dict this fills with item 1's numbers (full_days, bad_halts, steps), for
    the demo gate's short-demo decision."""
    argv = sys.argv if argv is None else argv
    load_env()
    log_path, db_path, status_path, env = LOG, DB, STATUS, "prod"
    if "--config" in argv:
        log_path, db_path, status_path, env = paths_from_config(argv[argv.index("--config") + 1])
    use_api = "--no-api" not in argv
    results = []

    def res(name, verdict, detail):
        results.append((name, verdict, detail))
        print(f"[{verdict:5s}] {name}: {detail}")

    if not os.path.exists(db_path):
        print(f"no {db_path}: the bot has not run live yet")
        sys.exit(1)
    c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    from wxbot.timeutil import et_day
    since = None
    if "--since" in argv:   # midnight of that day on this computer's clock (the Mac runs on ET)
        since = time.mktime(time.strptime(argv[argv.index("--since") + 1], "%Y-%m-%d"))
    row = c.execute("SELECT MIN(ts) FROM events_log WHERE msg LIKE 'start mode=live%' AND ts >= ?",
                    (since or 0,)).fetchone()
    start = row[0] if row and row[0] else None
    if start_ts is not None:
        start = start_ts
    if start is None:
        res("live days", "FAIL", "no 'start mode=live' in the database" + (" since --since" if since else ""))
        sys.exit(1)
    # a trading day: an ET day (not today) with at least six hours of live quote snapshots (written every 5 min)
    per_day = defaultdict(set)
    for (t,) in c.execute("SELECT DISTINCT ts FROM quotes WHERE mode='live' AND ts >= ?", (start,)):
        per_day[et_day(t).isoformat()].add(int(t // 300))
    today = et_day(time.time()).isoformat()
    full_days = sorted(d for d, b in per_day.items() if d != today and len(b) >= 72)
    steps = c.execute("SELECT COUNT(*) FROM events_log WHERE ts >= ? AND msg LIKE 'step failed%'", (start,)).fetchone()[0]
    halts = [(t, m) for t, m in c.execute("SELECT ts, msg FROM events_log WHERE ts >= ? AND msg LIKE 'HALT:%'",
                                          (start,))]
    # the daily loss stop is the brake working, and a QUIESCE is you stopping it; anything else (the drawdown stop,
    # an API error burst, repeated step failures) needs a look before Stage 2
    bad_halts = [m for t, m in halts if not re.search(r"HALT: (daily loss stop hit|stopped by hand)", m)]
    verdict = "FAIL" if (len(full_days) < 2 or bad_halts) else ("CHECK" if steps else "PASS")
    if facts is not None:
        facts.update(full_days=list(full_days), bad_halts=list(bad_halts), steps=int(steps))
    res("1. two trading days, no unexplained halt", verdict,
        f"{len(full_days)} ET day(s) with 6 h+ of live quoting since {dt.datetime.fromtimestamp(start):%Y-%m-%d %H:%M}"
        f" ({', '.join(full_days)}); halts: {len(halts)} "
        f"({'; '.join(m[:80] for m in bad_halts) or 'only the daily loss stop or a stop by hand'}); step failures: "
        f"{steps}")

    lg = scan_log(log_path, start)
    pauses, now = lg["pauses"], time.time()
    still = (f"; the bot has logged Kalshi as paused since {dt.datetime.fromtimestamp(lg['paused_now']):%H:%M}: run "
             "the gate again after the pause") if lg.get("paused_now") else ""

    def net(ts, after):
        """How long it took with the exchange's paused time left out."""
        return after - paused_within(pauses, ts - after, ts)

    def waited(rows):
        """' ; N waited out an exchange pause (...)' when any of these overlapped one."""
        hit = [(after, net(ts, after)) for ts, after in rows if net(ts, after) < after - 0.5]
        return (f"; {len(hit)} waited out an exchange outage (Kalshi answering 'service unavailable', then paused), "
                f"counted without it (longest {max(a for a, _ in hit):.0f} s with it, "
                f"{max(n for _, n in hit):.0f} s without)" if hit else "")

    starts = lg["uncertain_started"]
    resolved = {cid: (how, after, ts) for ts, how, after, cid in lg["uncertain_resolved"]}
    slow = [(cid, after) for cid, (how, after, ts) in resolved.items() if net(ts, after) > 60]
    open_old = [cid for ts, cid in starts if cid and cid not in resolved and net(now, now - ts) > 60]
    res("2. answerless placements resolved within 60 s",
        "PASS" if not slow and not open_old else "FAIL",
        f"{len(starts)} without an answer; resolved {len(resolved)} (adopted "
        f"{sum(1 for h, _, _ in resolved.values() if h == 'adopted')}); slower than 60 s {len(slow)}; never resolved "
        f"{len(open_old)}{' (' + ', '.join(open_old[:3]) + '; a restart in between also leaves them open)' if open_old else ''}"
        + waited([(ts, after) for _, after, ts in resolved.values()]) + (still if open_old or slow else ""))
    pend = lg["pending_resolved"]
    done_p = {cid: (after, ts) for ts, how, after, cid in pend}
    named = lg.get("pending_started", [])
    open_p = sorted({cid for ts, cid in named if cid not in done_p and net(now, now - ts) > 60})
    slow_p = [cid for cid, (after, ts) in done_p.items() if net(ts, after) > 60]
    res("3. unconfirmed cancels resolved within 60 s", "PASS" if not slow_p and not open_p else "FAIL",
        f"{len({cid for _, cid in named})} orders whose cancel was not confirmed at first; resolved {len(done_p)} "
        f"(slowest {max((a for a, _ in done_p.values()), default=0):.0f} s); slower than 60 s {len(slow_p)}; never "
        f"resolved {len(open_p)}{' (' + ', '.join(open_p[:3]) + ')' if open_p else ''}"
        + waited([(ts, after) for after, ts in done_p.values()]) + (still if open_p or slow_p else ""))

    took_log = lg["filled_on_arrival"]
    if not use_api:
        res("4-7. account checks", "CHECK", "skipped (--no-api): run again with the API key in .env")
        if took_log:
            res("7. post-only held (no maker order took liquidity)", "FAIL",
                f"{took_log} 'filled on arrival' warnings in the log: Kalshi filled a post-only order as a taker")
    else:
        from wxbot.kalshi.auth import KalshiAuth
        from wxbot.kalshi.rest import DEMO, PROD, KalshiREST, str_to_cents
        auth = KalshiAuth.from_env()
        if auth is None:
            res("4-7. account checks", "CHECK", "no API key in the environment (.env): skipped")
        else:
            r = KalshiREST(base_url=DEMO if env == "demo" else (os.environ.get("KALSHI_BASE_URL") or PROD),
                           auth=auth)
            orders = r.paged("/portfolio/orders", "orders", {"min_ts": int(start), "limit": 1000},
                             auth_required=True)
            mine = [o for o in orders if str(o.get("client_order_id") or "").startswith("wxb-")
                    and "smoke" not in str(o.get("client_order_id"))]
            fills = list(r.fills(min_ts=int(start)))
            ids = {o["order_id"] for o in mine}
            fill_times = defaultdict(list)            # order id -> [(ts, count)]
            for f in fills:
                if f.get("order_id") in ids:
                    fill_times[f["order_id"]].append((iso(f.get("created_time")) or 0.0, float(f.get("count_fp") or 0)))
            by_key, estimated, est_ids = defaultdict(list), 0, set()
            for o in mine:
                if re.match(r"wxb-\w-tk", str(o.get("client_order_id"))):
                    continue                             # the taker's immediate-or-cancel orders never rest
                t0 = iso(o.get("created_time"))
                if o.get("status") == "resting":
                    end = time.time()
                else:
                    end = iso(o.get("last_update_time"))
                    if end is None:                      # optional in Kalshi's schema: its last fill, its expiry or
                        estimated += 1                   # its creation instead (never "still resting")
                        est_ids.add(o["order_id"])
                        fl = [t for t, n in fill_times.get(o["order_id"], [])]
                        end = max(fl) if fl else (iso(o.get("expiration_time")) or t0)
                by_key[(o["ticker"], side_of(o))].append((t0 or 0, end or t0 or 0, o))
            dups, topups, unsure = [], 0, []
            for k, lst in by_key.items():
                lst.sort(key=lambda x: x[0])
                for i, (b0, b1, ob) in enumerate(lst):
                    for a0, a1, oa in lst[:i]:
                        if a1 - 1.0 <= b0:
                            continue                     # the earlier one was gone before the later one went out
                        size = float(oa.get("initial_count_fp") or 0) or (
                            float(oa.get("fill_count_fp") or 0) + float(oa.get("remaining_count_fp") or 0))
                        filled = sum(n for t, n in fill_times.get(oa["order_id"], []) if t <= b0)
                        if size and filled >= 0.5 * size:
                            topups += 1                  # a top-up beside a mostly filled order: by design
                            continue
                        pair = f"{k[0]} {k[1]}: {oa.get('client_order_id')} and {ob.get('client_order_id')}"
                        # the earlier order's end is only an estimate (no update time): cannot tell, so CHECK
                        (unsure if oa["order_id"] in est_ids else dups).append(pair)
            res("4. no duplicate orders", "FAIL" if dups else ("CHECK" if unsure else "PASS"),
                f"{len(mine)} bot orders on {len(by_key)} markets; {topups} top-ups beside a mostly filled order "
                f"(by design); {len(dups)} resting beside another bot order{': ' + '; '.join(dups[:3]) if dups else ''}"
                + (f"; {estimated} orders without an update time (end estimated)" if estimated else "")
                + (f"; {len(unsure)} pairs that may overlap (check in the app): {'; '.join(unsure[:3])}"
                   if unsure else ""))
            kal = {str(f.get("fill_id") or f.get("trade_id")): f for f in fills if f.get("order_id") in ids}
            ours = {ref: (pc, cnt) for ref, pc, cnt in c.execute(
                "SELECT order_ref, price_cents, count FROM fills WHERE mode='live' AND ts >= ?", (start,))}
            missing = [k for k in kal if k not in ours]
            extra = [k for k in ours if k not in kal]
            wrong = [k for k, f in kal.items() if k in ours and (
                ours[k][0] != str_to_cents(f.get("yes_price_dollars")) or abs(ours[k][1] - float(f.get("count_fp")
                                                                                                    or 0)) > 1e-6)]
            res("5. the bot's fills are Kalshi's fills",
                "FAIL" if (missing or extra or wrong) else ("PASS" if kal else "CHECK"),
                f"Kalshi {len(kal)} fills on bot orders, the bot recorded {len(ours)}; missing {len(missing)}, extra "
                f"{len(extra)}, different price or count {len(wrong)}"
                + ("" if kal else "; no fills yet, so nothing was reconciled: let Stage 1 run until some have settled"))
            st = json.load(open(status_path)) if os.path.exists(status_path) else {}
            st_age = time.time() - float(st.get("ts") or 0)
            b = r.balance()
            cash = (b.get("balance") or 0) / 100.0
            held = sum(float(o.get("remaining_count_fp") or 0) * (
                str_to_cents(o.get("yes_price_dollars")) if side_of(o) == "bid"
                else 100 - str_to_cents(o.get("yes_price_dollars"))) / 100.0
                for o in r.orders(status="resting"))
            pos, _ = r.positions()
            basis = sum(abs(float(p.get("market_exposure_dollars") or 0)) for p in pos)   # a cost: never negative
            sem = st.get("balance_semantics") or {}
            fresh = cash + (held if sem.get("excludes_resting", True) else 0.0) + basis
            eq = st.get("live_equity_usd")
            ok = eq is not None and abs(float(eq) - fresh) <= 1.0
            res("6. settled equity matches the account",
                "CHECK" if st_age > 120 else ("PASS" if ok else "FAIL"),
                f"status ${eq} vs fresh ${fresh:.2f} (cash {cash:.2f}"
                + (f" + held {held:.2f}" if sem.get("excludes_resting", True)
                   else f", which already includes {held:.2f} held by resting orders,")
                + f" + positions at cost {basis:.2f}); " + (f"the status file is {st_age / 60:.0f} min old: is the bot running?"
                                     if st_age > 120 else "the status file is up to 30 s old, so rerun if a fill "
                                                          "just landed"))
            # the maker sleeves post only: a fill of theirs marked taker means post-only did not hold (the taker's
            # own orders, wxb-<sleeve>-tk..., take by design)
            maker_ids = {o["order_id"] for o in mine if not re.match(r"wxb-\w-tk", str(o.get("client_order_id")))}
            took = [f for f in fills if f.get("order_id") in maker_ids and f.get("is_taker") in (True, "true", 1)]
            res("7. post-only held (no maker order took liquidity)", "PASS" if not (took or took_log) else "FAIL",
                f"{len(took)} fills of maker orders marked taker on Kalshi; {took_log} 'filled on arrival' warnings in "
                "the log")
    if more_items is not None:
        more_items(c, start, res)
    fails = [n for n, v, _ in results if v == "FAIL"]
    checks = [n for n, v, _ in results if v == "CHECK"]
    verdict = "FAIL" if fails else "CHECK" if checks else "PASS"
    print(f"\n{label}:", "FAIL (" + "; ".join(fails) + ")" if fails else
          "CHECK (" + "; ".join(checks) + ")" if checks else "PASS")
    if on_verdict is not None:            # it may return the final verdict (the demo gate's short demo)
        verdict = on_verdict(verdict, results, start) or verdict
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
