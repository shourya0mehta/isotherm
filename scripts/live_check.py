"""Read-only health check of the running bot (stdlib only; never calls Kalshi, never trades).

Usage: python scripts/live_check.py [--wait SECONDS] [data/status.json] [data/live.sqlite]
Prints what to look at in the first live hours: mode and halts, equity and the balance semantics the stops use,
resting orders, fills and P&L so far, the hourly breaker, speed, API errors, recent warnings. Exit code 1 when
something needs a human (halted, stale status, breaker tripped, orders not confirmed cancelled).
--wait SECONDS (GO_LIVE uses it right after the switch): first wait, up to that long, until the live bot has written a
status newer than this command and has run for half a minute (a restart can take a couple of minutes)."""
import json
import os
import sqlite3
import sys
import time


def wait_for_live(path: str, seconds: float, settle_s: float = 30.0) -> bool:
    """Wait until `path` holds a live-mode status written after this call by a bot up for settle_s seconds."""
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            st = json.load(open(path))
            if (st.get("mode") == "live" and float(st.get("ts") or 0) > t0
                    and float(st.get("uptime_min") or 0) * 60 >= settle_s):
                return True
        except (OSError, ValueError):
            pass
        time.sleep(3)
    return False


def main():
    if "--wait" in sys.argv:
        i = sys.argv.index("--wait")
        secs = float(sys.argv[i + 1]) if i + 1 < len(sys.argv) else 240.0
        del sys.argv[i:i + 2]
        target = sys.argv[1] if len(sys.argv) > 1 else "data/live_status.json"
        print(f"waiting for the live bot to start (up to {secs / 60:.0f} min)...", flush=True)
        if not wait_for_live(target, secs):
            print(f"\nNEEDS A LOOK: no live status in {target} after {secs / 60:.0f} min. The bot may still be "
                  "restarting: run this again in a few minutes (.venv/bin/python scripts/live_check.py)")
            sys.exit(1)
        if len(sys.argv) == 1:
            sys.argv.append(target)
    if len(sys.argv) > 1:
        st_path = sys.argv[1]
    else:   # the live config writes data/live_status.json, paper data/status.json: take the fresher one
        cands = [p for p in ("data/live_status.json", "data/status.json") if os.path.exists(p)]
        st_path = max(cands, key=os.path.getmtime) if cands else "data/status.json"
    st = json.load(open(st_path))
    db_path = sys.argv[2] if len(sys.argv) > 2 else (
        "data/live.sqlite" if st.get("mode") == "live" else "data/wxbot.sqlite")
    bad = []
    age = time.time() - st.get("ts", 0)
    print(f"mode {st.get('mode')}  status {age:.0f} s old  uptime {st.get('uptime_min')} min")
    if age > 120:
        bad.append(f"status file is {age / 60:.0f} min old: is the bot running?")
    if st.get("halted"):
        bad.append(f"HALTED: {st.get('halt_reason')}")
    print(f"equity ${st.get('live_equity_usd')}  available ${st.get('available_balance_usd')}  "
          f"resting orders {st.get('live_resting_orders')} (${st.get('live_resting_cost_usd')}), added back to "
          f"equity ${st.get('live_equity_adds_resting_usd')}  balance semantics {st.get('balance_semantics')}")
    print(f"open risk ${st.get('open_risk_usd')}  cash in positions ${st.get('cash_in_positions_usd')}  "
          f"quotes {st.get('quotes')}  fills since start {st.get('fills_since_start')}  "
          f"realized ${st.get('realized_usd')}  mtm ${st.get('mtm_usd')}")
    hb = st.get("sleeve_h_breaker") or {}
    print(f"hourly real orders {'on' if hb.get('real_orders_on') else 'off'}"
          f"{'  BREAKER TRIPPED: ' + str(hb.get('reason')) if hb.get('tripped') else ''}  stats {hb.get('stats')}")
    if hb.get("tripped"):
        bad.append("hourly breaker tripped (review, then python -m wxbot reset-h-breaker)")
    if st.get("mode") == "live":
        print(f"account fresh {st.get('account_fresh')}  equity readings {st.get('equity_readings')} (orders wait "
              f"for 3)  read ages {st.get('account_read_age_s')}  "
              f"order states {st.get('live_order_states')}  minutes to the soonest expiry "
              f"{st.get('live_min_minutes_to_expiry')}")
        if st.get("account_fresh") is False:
            bad.append("account reads stale: no new orders, resting ones lapse within 5 min (check the network)")
    if st.get("mode") == "live" and not st.get("halted") and st.get("quotes") == 0 and \
            float(st.get("uptime_min") or 0) > 30:
        print("note: no orders wanted right now (caps full, account reads stale, every event skipped?); the bot alerts "
              "if this lasts two hours")
    tp = st.get("trading_paused")
    if tp:
        print(f"exchange: not trading for {tp.get('since_min')} min ({tp.get('why')}); "
              f"hours {st.get('exchange_hours')}")
        if float(tp.get("since_min") or 0) > 180:
            bad.append(f"not trading for {float(tp['since_min']) / 60:.1f} h ({tp.get('why')}): check Kalshi's site")
    if st.get("post_only_filled_on_arrival"):
        bad.append(f"{st['post_only_filled_on_arrival']} post-only order(s) filled on arrival: post-only is not "
                   "holding (check the Kalshi app; BACK_TO_PAPER.command stops trading)")
    if st.get("side_task_errors"):
        print(f"side task errors (trading unaffected): {st.get('side_task_errors')}")
    ws = st.get("ws_shadow")
    if ws:
        print(f"stream (shadow): connected {ws.get('connected')}, {ws.get('books')} books, agree "
              f"{ws.get('agree_pct')}% of {ws.get('compared')}, lead p50 {ws.get('lead_s_p50')} s, fill lead p50 "
              f"{ws.get('fill_lead_s_p50')} s, stuck {ws.get('stuck_mismatches')}, gaps {ws.get('seq_gaps')}, "
              f"reconnects {ws.get('reconnects')}")
    tm = st.get("timing") or {}
    print(f"speed: book age at order updates {tm.get('book_age_at_orders_s')}  step {tm.get('step_s')}  "
          f"API {tm.get('api_round_trip_ms')}")
    print(f"API requests {st.get('api_requests')}  errors {st.get('api_errors')}  last {st.get('last_api_error')}")
    if os.path.exists(db_path):
        c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        mode = st.get("mode")
        n, ct = c.execute("SELECT COUNT(*), COALESCE(SUM(count), 0) FROM fills WHERE mode=?", (mode,)).fetchone()
        print(f"{db_path}: {n} fills, {ct:.0f} contracts")
        for ts, t, side, pc, cnt, sl in c.execute("SELECT ts, ticker, side, price_cents, count, sleeve FROM fills "
                                                  "WHERE mode=? ORDER BY ts DESC LIMIT 8", (mode,)):
            pay = pc if side == "bid" else 100 - pc
            print(f"  {time.strftime('%m-%d %H:%M', time.gmtime(ts))}Z {t} buy {'YES' if side == 'bid' else 'NO'} "
                  f"at {pay}c x{cnt:g} ({sl})")
        try:
            pnl = c.execute("SELECT COALESCE(SUM(pnl_usd), 0), COUNT(*) FROM live_settlements").fetchone()
            print(f"settled live: ${pnl[0]:.2f} on {pnl[1]} markets")
        except sqlite3.OperationalError:
            pass
        rows = c.execute("SELECT ts, level, msg FROM events_log WHERE level IN ('ERROR', 'WARN') "
                         "ORDER BY ts DESC LIMIT 6").fetchall()
        for ts, lvl, msg in rows:
            print(f"  {time.strftime('%m-%d %H:%M', time.gmtime(ts))}Z {lvl} {msg[:150]}")
            if lvl == "ERROR" and "may still" in msg and time.time() - ts < 3600:
                bad.append("orders may still be resting after a halt: check the Kalshi app")
    print("\n" + ("NEEDS A LOOK: " + "; ".join(bad) if bad else "OK"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
