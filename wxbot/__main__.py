"""wxbot command line.

  python -m wxbot run     [--config config/settings.yaml] [--minutes N] [--mode paper|live]
  python -m wxbot check   # connectivity, auth, balance, exchange status (safe, read-only)
  python -m wxbot status  # print data/status.json
  python -m wxbot report  # P&L / fills / calibration report from the SQLite store
  python -m wxbot cancel-all  # emergency: cancel every wxbot order (live); exit code 1 if any may still rest
  python -m wxbot smoke   # live key test: place/amend/cancel one 1c order (max cost $0.02) [--series KXHIGHTATL]
  python -m wxbot dashboard  # write data/<db>_dashboard.html (the running bot refreshes it every 5 min)
  python -m wxbot netting [on|off]  # show / set Kalshi collateral netting for the primary account
  python -m wxbot reset-stops       # after a deposit/withdrawal: restart the drawdown/daily baselines
  python -m wxbot reset-h-breaker   # turn the hourly bot back on after its breaker tripped (counts afresh from now)
  python -m wxbot reset-shutoff X   # turn a sleeve (A, T, X, E, R, or all) back on after its shut-off rule switched it
                                    # off (its count starts afresh from now); restarts the bot
  python -m wxbot reset-halt        # clear a hard halt (drawdown stop, QUIESCE) after a human review; restarts the bot
  python -m wxbot set-peak --to-sod # the drawdown stop's high point (kv peak_equity) back to the start-of-day equity
                                    # (kv sod_equity), or --value X; changes nothing else and clears no halt (see
                                    # set_peak: after a drawdown halt caused by a wrong high point, set-peak, then
                                    # reset-halt)
  python -m wxbot gate              # go/no-go for the next step (paper -> live, half -> full caps)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys


def _load_env(path=".env"):
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


SHUTOFF_SLEEVES = ("A", "T", "X", "E", "R")


def reset_shutoff(cfg: dict, which, now: float | None = None) -> str:
    """Turn sleeves back on after their shut-off rule switched them off (wxbot/risk/shutoff.py): kv shutoff:<sleeve>
    becomes not switched off, with the count starting afresh now (reset_ts); the running bot restarts to pick it up."""
    import time as _time
    from .store.db import Store
    if not which:
        sys.exit("reset-shutoff needs a sleeve letter (A, T, X, E, R) or all")
    w = str(which).strip().upper()
    sleeves = list(SHUTOFF_SLEEVES) if w == "ALL" else [w[:1]]
    if any(s not in SHUTOFF_SLEEVES for s in sleeves):
        sys.exit(f"unknown sleeve {which!r}: one of A, T, X, E, R or all")
    now = _time.time() if now is None else now
    st = Store(cfg["db_path"])
    done = []
    for s in sleeves:
        old = st.kv_get(f"shutoff:{s}", {}) or {}
        prev = {k: old.get(k) for k in ("day", "ts", "S", "since_ts")} if old.get("tripped") else old.get("previous")
        st.kv_set(f"shutoff:{s}", {"tripped": False, "reset_ts": now, "first_ts": old.get("first_ts"),
                                   "previous": prev})
        was = f"switched off on {old.get('day')}" if old.get("tripped") else "on"
        st.log("INFO", f"Sleeve {s} shut-off cleared by hand (was {was}): its count starts afresh from now")
        done.append(f"{s} (was {was})")
    flag = os.path.join(os.path.dirname(cfg.get("status_path", "data/status.json")) or ".", "RESTART")
    os.makedirs(os.path.dirname(flag) or ".", exist_ok=True)
    open(flag, "w").close()
    return "shut-off cleared for " + ", ".join(done) + "; the bot restarts within a minute to pick it up"


def set_peak(cfg: dict, to_sod: bool = False, value=None) -> str:
    """Set the high point the drawdown stop measures from (kv peak_equity) by hand: to the start-of-day equity (kv
    sod_equity, --to-sod) or to a value (--value X). Oct 11, 2026: kv peak_equity sat about $15 above any real equity
    (a reading taken while a payout was in the balance but its position still in the list raised it
    by about $15, bringing the $50 hard halt $15 closer); config risk.peak_from: start_of_day keeps that from happening
    again, and this puts the old value right.
    It changes nothing else: no halt is cleared (that is reset-halt, which also restarts the bot) and sod_equity, the
    daily stop's baseline, stays (reset-stops clears both after a deposit or withdrawal). The running bot reads the
    stored high point at every equity reading, so no restart is needed; with peak_from: reading a reading above the new
    value raises it again at once. After a drawdown halt caused by a wrong high point: set-peak first, then reset-halt
    (reset-halt alone would restart the bot into the same halt). Refuses when the bot's database is not found (a wrong
    --config must not create a new one)."""
    import math
    from .store.db import Store
    path = cfg.get("db_path") or ""
    if not path or not os.path.isfile(path):
        sys.exit(f"set-peak: no bot database at {path or '(no db_path in the config)'}: check --config; nothing changed")
    if bool(to_sod) == (value is not None):
        sys.exit("set-peak needs exactly one of --to-sod or --value X; nothing changed")
    st = Store(path)
    old = st.kv_get("peak_equity")
    if to_sod:
        sod = st.kv_get("sod_equity") or {}
        new = sod.get("equity") if isinstance(sod, dict) else None
        if new is None:
            sys.exit(f"set-peak --to-sod: no start-of-day equity (kv sod_equity) in {path} yet; nothing changed")
        src = f"the start-of-day equity of {sod.get('day')}"
    else:
        new, src = value, "given by hand"
    try:
        new = float(new)
    except (TypeError, ValueError):
        new = float("nan")
    if not math.isfinite(new) or new <= 0:
        sys.exit(f"set-peak: {new} is not a positive dollar amount; nothing changed")
    was = "none" if old is None else f"{float(old):.2f}"
    st.kv_set("peak_equity", new)
    st.log("INFO", f"drawdown high point (kv peak_equity) set by hand: {was} -> {new:.2f} ({src})")
    return (f"peak_equity {was} -> {new:.2f} ({src}) in {path}. Nothing else changed: a halt stays until reset-halt; the "
            "running bot uses the new high point from its next equity reading")


def main(argv=None):
    _load_env()
    ap = argparse.ArgumentParser(prog="wxbot")
    ap.add_argument("cmd", choices=["run", "check", "status", "report", "cancel-all", "smoke", "dashboard", "netting", "reset-stops", "gate",
                                       "reset-h-breaker", "reset-halt", "reset-shutoff", "set-peak"])
    ap.add_argument("arg", nargs="?", default=None, help="netting: on|off; reset-shutoff: a sleeve letter or all")
    ap.add_argument("--to-sod", action="store_true", help="set-peak: to the start-of-day equity (kv sod_equity)")
    ap.add_argument("--value", type=float, default=None, help="set-peak: to this many dollars")
    ap.add_argument("--config", default=os.environ.get("WXBOT_CONFIG", "config/settings.yaml"))
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--mode", choices=["paper", "live"], default=None)
    ap.add_argument("--log", default="INFO")
    ap.add_argument("--series", default=None, help="smoke: the series to pick a market from (default: first Sleeve A series)")
    a = ap.parse_args(argv)
    os.makedirs("data", exist_ok=True)
    from .app.runner import Bot, load_config
    cfg = load_config(a.config)
    if a.cmd == "set-peak":
        # before the log file is set up below: a wrong --config leaves nothing behind (no folder, no log file)
        print(set_peak(cfg, to_sod=a.to_sod, value=a.value))
        return
    log_path = os.path.splitext(cfg.get("db_path", "data/wxbot.sqlite"))[0] + ".log"
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    logging.basicConfig(level=getattr(logging, a.log.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)])
    if a.mode:
        cfg["mode"] = a.mode
    if a.cmd == "run":
        Bot(cfg).run(max_minutes=a.minutes)
    elif a.cmd == "status":
        print(open(cfg.get("status_path", "data/status.json")).read())
    elif a.cmd == "check":
        from .app.check import run_check
        run_check(cfg)
    elif a.cmd == "report":
        from .app.report import build_report
        print(build_report(cfg))
    elif a.cmd == "gate":
        from .app.report import gate_report
        ok, txt = gate_report(cfg)
        print(txt)
        sys.exit(0 if ok else 1)
    elif a.cmd == "reset-stops":
        from .store.db import Store
        st = Store(cfg["db_path"])
        st.c.execute("DELETE FROM kv WHERE k IN ('peak_equity', 'sod_equity')")
        st.c.commit()
        st.log("INFO", "drawdown/daily baselines reset by hand (deposit or withdrawal)")
        print("baselines cleared: the running bot re-reads equity within a minute; restart it if it is halted")
    elif a.cmd == "reset-h-breaker":
        import time as _time
        from .store.db import Store
        st = Store(cfg["db_path"])
        old = st.kv_get("h_breaker", {}) or {}
        st.kv_set("h_breaker", {"tripped": False, "since": _time.time(), "reset_ts": _time.time(),
                                "previous_reason": old.get("reason"), "previous_ts": old.get("ts")})
        st.log("INFO", "Sleeve H breaker cleared by hand: it counts settled hourly contracts afresh from now")
        flag = os.path.join(os.path.dirname(cfg.get("status_path", "data/status.json")) or ".", "RESTART")
        open(flag, "w").close()
        print("hourly breaker cleared (was: %s); the bot restarts within a minute to pick it up"
              % (old.get("reason") or "not tripped"))
    elif a.cmd == "reset-shutoff":
        print(reset_shutoff(cfg, a.arg))
    elif a.cmd == "reset-halt":
        from .store.db import Store
        st = Store(cfg["db_path"])
        old = st.kv_get("halt_latch", {}) or {}
        st.kv_set("halt_latch", {})
        st.log("INFO", f"halt cleared by hand (was: {old.get('kind') or 'none'}: {old.get('reason') or ''})")
        flag = os.path.join(os.path.dirname(cfg.get("status_path", "data/status.json")) or ".", "RESTART")
        os.makedirs(os.path.dirname(flag) or ".", exist_ok=True)
        open(flag, "w").close()
        print("halt cleared (was: %s); the bot restarts within a minute" % (old.get("kind") or "not halted"))
    elif a.cmd == "netting":
        from .kalshi.auth import KalshiAuth
        from .kalshi.rest import DEMO, PROD, KalshiREST
        auth = KalshiAuth.from_env()
        if auth is None:
            sys.exit("needs KALSHI_KEY_ID + key in the environment")
        r = KalshiREST(base_url=DEMO if cfg.get("env") == "demo" else os.environ.get("KALSHI_BASE_URL", PROD),
                       auth=auth)
        if a.arg in ("on", "off"):
            r.set_netting(a.arg == "on", 0)
        print(json.dumps(r.netting(), indent=1))
    elif a.cmd == "dashboard":
        from .app.dashboard import write_dashboard
        print(write_dashboard(cfg))
    elif a.cmd == "smoke":
        from .app.smoke import run_smoke
        sys.exit(0 if run_smoke(cfg, series=a.series) else 1)
    elif a.cmd == "cancel-all":
        cfg["mode"] = "live"
        b = Bot(cfg)
        left = b.live.cancel_all("manual cancel-all")
        if left == 0:
            print("cancelled all wxbot orders (confirmed by the exchange)")
        else:
            print("WARNING: %s wxbot orders may still be resting; they expire on their own within their TTL. "
                  "Check the Kalshi app." % ("an unknown number of" if left < 0 else left))
            sys.exit(1)


if __name__ == "__main__":
    main()
