"""Read-only preflight: exchange status, market data reachability, NBM/METAR reachability, and (if keys are set)
authenticated balance, positions, and resting orders. Never places orders."""
from __future__ import annotations

import datetime as dt
import os

from ..data import nbm, obs
from ..kalshi.auth import KalshiAuth
from ..kalshi.rest import DEMO, PROD, KalshiREST


def run_check(cfg: dict):
    ok = True
    auth = KalshiAuth.from_env()
    base = DEMO if cfg.get("env") == "demo" else os.environ.get("KALSHI_BASE_URL", PROD)
    r = KalshiREST(base_url=base, auth=auth)

    def line(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print(f"[{'OK ' if good else 'BAD'}] {name}: {detail}")

    try:
        s = r.exchange_status()
        line("exchange status", s.get("trading_active"), f"trading_active={s.get('trading_active')}")
    except Exception as e:
        line("exchange status", False, repr(e))
    try:
        ser = ((cfg.get("sleeve_a") or {}).get("series") or ["KXHIGHNY"])[0]
        ev = r.events(ser, status="open", nested=True, limit=2)
        line("market data", bool(ev), f"{len(ev)} open {ser} events")
    except Exception as e:
        line("market data", False, repr(e))
    try:
        cyc, txt = nbm.latest("nbs", lookback_h=6)
        line("NBM NBS bulletin", bool(txt), f"cycle={cyc}")
    except Exception as e:
        line("NBM", False, repr(e))
    try:
        m = obs.fetch_metars(["KNYC", "KMIA"], hours=3)
        line("AWC METAR", any(m.values()), ", ".join(f"{k}:{len(v)}" for k, v in m.items()))
    except Exception as e:
        line("AWC METAR", False, repr(e))
    if auth is None:
        print("[--] no API key in environment: paper mode only (set KALSHI_KEY_ID + KALSHI_PRIVATE_KEY_PATH)")
    else:
        try:
            b = r.balance()
            line("auth + balance", True, f"balance=${b.get('balance', 0) / 100:.2f} portfolio=${(b.get('portfolio_value') or 0) / 100:.2f}")
            for row in b.get("balance_breakdown") or []:
                print("      shard", row)
        except Exception as e:
            line("auth + balance", False, repr(e))
        try:
            mp, _ = r.positions()
            line("positions", True, f"{len(mp)} market positions")
            od = r.orders(status="resting")
            line("resting orders", True, f"{len(od)} resting")
        except Exception as e:
            line("positions/orders", False, repr(e))
        try:
            cfgs = r.netting().get("netting_configs") or []
            on = any(c.get("enabled") and int(c.get("subaccount_number", -1)) == 0 for c in cfgs)
            print(f"[{'OK ' if on else '-- '}] collateral netting: {'ON' if on else 'OFF'}"
                  + ("" if on else "  (optional: `python -m wxbot netting on` frees cash locked by NO positions on "
                                   "several brackets of the same city-day; the bot works either way)"))
        except Exception as e:
            print(f"[-- ] collateral netting: unknown ({e!r})")
    print("PREFLIGHT", "PASSED" if ok else "FAILED", dt.datetime.now(dt.timezone.utc).isoformat())
    return ok
