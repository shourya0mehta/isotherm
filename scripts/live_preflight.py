"""Read-only check of a live Kalshi API key and the go-live config. It signs requests with the key and reads the
exchange status, the account balance, open positions and resting orders. It never places or cancels an order.
Usage: python scripts/live_preflight.py <key_id_file> <private_key_path> [config]   (exit code 0 = ready)"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import yaml  # noqa: E402

from wxbot.kalshi.auth import KalshiAuth, load_private_key  # noqa: E402
from wxbot.kalshi.rest import PROD, KalshiREST  # noqa: E402


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    key_id = open(sys.argv[1]).read().strip()
    key_path = os.path.expanduser(sys.argv[2])
    cfg_path = sys.argv[3] if len(sys.argv) > 3 else "config/live_go.yaml"
    ok = True
    mode = oct(os.stat(key_path).st_mode & 0o777)
    print(f"key id {key_id[:8]}..., key file {os.path.basename(key_path)} (permissions {mode})")
    auth = KalshiAuth(key_id=key_id, private_key=load_private_key(open(key_path, "rb").read()))
    r = KalshiREST(base_url=os.environ.get("KALSHI_BASE_URL", PROD), auth=auth)
    st = r.exchange_status()
    print(f"exchange active {st.get('exchange_active')}, trading active {st.get('trading_active')}")
    try:
        b = r.balance()
    except Exception as e:   # bad key id, wrong key file, or a key without read access
        print(f"FAILED to read the balance with this key: {e}")
        sys.exit(1)
    bal = (b.get("balance") or 0) / 100.0
    pv = (b.get("portfolio_value") or 0) / 100.0
    print(f"balance ${bal:.2f}, positions value ${pv:.2f}")
    pos, _ = r.positions()
    held = [p for p in pos if float(p.get("position_fp") or 0) != 0]
    orders = r.orders(status="resting")
    print(f"{len(held)} open positions, {len(orders)} resting orders in the account")
    if held or orders:
        print("WARNING: there are already positions or orders in this account. The bot counts every position toward "
              "its caps (Stage 1: $75 of open risk) and every change in cash toward its loss brakes, and it never "
              "touches orders it did not place. Close them first for a clean start, or expect it to quote less.")
    cfg = yaml.safe_load(open(cfg_path))
    rk = cfg.get("risk", {})
    print(f"config {cfg_path}: mode {cfg.get('mode')}, env {cfg.get('env')}; daily pause -${rk.get('daily_loss_stop_usd')}"
          f", total halt -${rk.get('total_drawdown_stop_usd')}, cash cap ${rk.get('max_cash_deployed_usd')}")
    print(f"  daily maker {'on' if cfg.get('sleeve_a', {}).get('enabled') else 'off'}, thin-book maker "
          f"{'on' if cfg.get('sleeve_t', {}).get('enabled') else 'off'}, daily taker "
          f"{'REAL ORDERS' if cfg.get('taker_shadow', {}).get('trade') else 'log only'}, hourly maker "
          f"{'REAL ORDERS' if cfg.get('sleeve_h', {}).get('enabled') else 'log only'}, rain "
          f"{'REAL ORDERS' if cfg.get('rain_shadow', {}).get('trade') else 'log only'}")
    if cfg.get("mode") != "live" or cfg.get("env") != "prod":
        print("FAILED: the config is not a live production config")
        ok = False
    every = float((cfg.get("loop") or {}).get("positions_every_s", 5))
    stale = float(rk.get("account_stale_seconds", 30))
    if 2 * every > stale:   # one missed read must not make the account stale (and restart the stops' median)
        print(f"FAILED: account reads every {every:.0f} s, but orders need them within {stale:.0f} s")
        ok = False
    need = float(rk.get("min_cash_reserve_usd", 25)) + 50
    if bal < need:
        print(f"FAILED: balance ${bal:.2f} is below ${need:.0f}; deposit first")
        ok = False
    if not st.get("trading_active", True):
        print("note: trading is paused on the exchange right now; the bot waits")
    print("READY" if ok else "NOT READY")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
