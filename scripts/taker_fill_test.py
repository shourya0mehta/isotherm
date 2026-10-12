#!/usr/bin/env python3
"""Practice-site test of the buy-right-away (immediate-or-cancel) order path, with real fills (Oct 8, 2026).

Kalshi's practice (demo) exchange has almost no other traders, so in two days the bot's immediate-or-cancel orders
never found anything to fill against. This test makes the fill happen on purpose, with demo money only:

1. demo account 2 (the troublemaker's account) posts a resting post-only offer inside the spread of a market where
   the bot's demo account has no orders;
2. the bot's demo account buys into it with the bot's own taker code (`TakerExecutor`, called the way
   `wxbot/app/runner.py` `execute_take` calls it, client order id `wxb-X-tk...`), once buying YES and once buying NO
   (on two different markets, so the second buy does not close the first);
3. it checks the order answer (fill count, average price, fee), the TakeResult the bot books its day budget and
   position from, and Kalshi's fill record, then cancels what is left of account 2's offers. The running demo bot
   books the fills through its own fill feed (they carry a wxb- client id); `--check-db` reads its database later.

Demo keys only: refuses if a key file matches the real key. Usage (server, next folder):
  .venv/bin/python scripts/taker_fill_test.py            # the two takes and the checks
  .venv/bin/python scripts/taker_fill_test.py --check-db # how the running demo bot booked them
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
import sys
import time
import uuid

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

from wxbot.execution.taker import TakerExecutor, taker_fee  # noqa: E402
from wxbot.kalshi.auth import KalshiAuth, load_private_key  # noqa: E402
from wxbot.kalshi.rest import DEMO, KalshiREST, str_to_cents  # noqa: E402


def kalshi_fee(count, price):
    """What Kalshi charged in the Oct 8 test: 0.07 x C x P x (1 - P), rounded up to $0.0001 (the bot's pre-trade budget
    check rounds up to the cent instead, which overstates it by under a cent per order)."""
    import math
    return math.ceil(round(0.07 * count * price * (1 - price) * 10000, 6)) / 10000

KEYS = os.path.join(os.path.dirname(ROOT), "deploy_keys")
STATE = os.path.join(ROOT, "data", "demo", "taker_fill_test.json")
SERIES = ["KXHIGHTATL", "KXHIGHTBOS", "KXHIGHTDAL", "KXHIGHTDC", "KXHIGHTHOU", "KXHIGHTNOLA", "KXHIGHTPHX",
          "KXHIGHTSEA"]
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def key_files(name):
    key = next((os.path.join(KEYS, name + e) for e in (".key", ".pem") if os.path.exists(os.path.join(KEYS, name + e))),
               None)
    idf = os.path.join(KEYS, name + "_key_id.txt")
    if key is None or not os.path.exists(idf):
        raise SystemExit(f"missing {name} key or key id in {KEYS}: nothing was sent")
    return open(idf).read().strip(), open(key, "rb").read()


def clients():
    adv, bot = key_files("kalshi_demo2"), key_files("kalshi_demo")
    real = [open(os.path.join(KEYS, f), "rb").read() for f in ("kalshi_live.key", "kalshi_live.pem")
            if os.path.exists(os.path.join(KEYS, f))]
    if adv[1] == bot[1] or adv[0] == bot[0] or adv[1] in real or bot[1] in real:
        raise SystemExit("a demo key matches the real key or the other demo key: stopping, nothing was sent")
    mk = lambda k: KalshiREST(base_url=DEMO, auth=KalshiAuth(k[0], load_private_key(k[1])))  # noqa: E731
    return mk(adv), mk(bot)


def book_edges(book):
    """(best YES bid, best YES ask) in cents from an orderbook_fp snapshot (None when a side is empty)."""
    yes = [float(p) for p, q in (book.get("yes_dollars") or []) if float(q) > 0]
    no = [float(p) for p, q in (book.get("no_dollars") or []) if float(q) > 0]
    bid = int(round(max(yes) * 100)) if yes else None
    ask = int(round((1 - max(no)) * 100)) if no else None
    return bid, ask


def resting_tickers(rest):
    return {o.get("ticker") or o.get("market_ticker") for o in rest.orders(status="resting")}


def pick_markets(adv, bot, n=2):
    """Bracket markets of the latest open events with no resting order from either account and room inside the
    spread (an empty side counts as 0c / 100c)."""
    busy = resting_tickers(bot) | resting_tickers(adv)
    picks = []
    for s in SERIES:
        try:
            evs = sorted(bot.events(s, status="open", nested=True), key=lambda e: e.get("event_ticker", ""))
        except Exception as e:
            print(f"  {s}: events failed ({e})")
            continue
        today = dt.datetime.now(dt.timezone(dt.timedelta(hours=-4))).date()   # US Eastern (daylight time)
        for ev in reversed(evs[-2:]):
            d = event_date(ev.get("event_ticker", ""))
            if d is None or d <= today:   # only days still ahead: the outcome is not known yet
                continue
            for m in ev.get("markets") or []:
                t = m.get("ticker")
                if not t or t in busy or "-B" not in t:
                    continue
                bid, ask = book_edges(bot.orderbook(t))
                lo, hi = (bid if bid is not None else 0), (ask if ask is not None else 100)
                if hi - lo >= 3:
                    picks.append((t, lo, hi))
                    break
            if picks and picks[-1][0].startswith(s):
                break                     # one market per city
        if len(picks) >= n:
            return picks
    return picks


def wait_resting(rest, order_id, tries=10):
    """The order's record once it shows as resting (account reads lag a moment after a create: retry 404s)."""
    o = {}
    for _ in range(tries):
        try:
            o = rest.order(order_id)
            if o.get("status") == "resting":
                return o
        except Exception as e:
            o = {"status": f"not visible yet ({str(e)[:60]})"}
        time.sleep(1)
    return o


def event_date(event_ticker):
    """2026-10-09 from KXHIGHTATL-26OCT09 (None if it does not parse)."""
    try:
        return dt.datetime.strptime(event_ticker.rsplit("-", 1)[1], "%y%b%d").date()
    except Exception:
        return None


def cancel_leftovers(adv):
    """Cancel account 2's resting orders from an earlier run of this test (client ids adv-test-...)."""
    for o in adv.orders(status="resting"):
        if str(o.get("client_order_id") or "").startswith("adv-test-"):
            try:
                adv.cancel_order(o["order_id"])
                print(f"  cancelled a leftover test offer on {o.get('ticker')}")
            except Exception as e:
                print(f"  leftover cancel failed: {e}")


def run():
    adv, bot = clients()
    start = int(time.time())
    cancel_leftovers(adv)
    picks = pick_markets(adv, bot)
    check("found two quiet markets on the practice site", len(picks) >= 2,
          "; ".join(f"{t} bid {b}c ask {a}c" for t, b, a in picks))
    if len(picks) < 2:
        return
    state = {"start": start, "takes": [], "adv_orders": []}
    plans = [("yes", picks[0]), ("no", picks[1])]
    for side, (ticker, bid, ask) in plans:
        if side == "yes":
            px = min(max(30, bid + 1), ask - 1)          # account 2 offers YES at px, inside the spread
            adv_side, our_limit = "ask", px / 100.0       # a resting YES ask at px; we buy YES at px
        else:
            px = max(min(70, ask - 1), bid + 1)          # account 2 bids YES at px, inside the spread
            adv_side, our_limit = "bid", (100 - px) / 100.0   # we buy NO at 100 - px
        if not 10 <= px <= 90:
            check(f"{side.upper()} test: a price between 10c and 90c fits inside {ticker}'s spread", False, f"{px}c")
            continue
        o = adv.create_order(ticker, adv_side, px, 2, post_only=True, tif="good_till_canceled",
                             client_order_id=f"adv-test-{uuid.uuid4().hex[:12]}", retries=0)
        oid = o.get("order_id")
        state["adv_orders"].append(oid)
        rested = wait_resting(adv, oid)
        check(f"{side.upper()} test: account 2's offer rests on {ticker} (YES {adv_side} at {px}c)",
              rested.get("status") == "resting", f"status {rested.get('status')}")
        if rested.get("status") != "resting":
            continue
        if ticker in resting_tickers(bot):
            check(f"{side.upper()} test: the bot's account still has no order on {ticker}", False)
            continue
        ex = TakerExecutor(bot, "live", 0)
        cid = f"wxb-X-tk{uuid.uuid4().hex[:16]}"
        t0 = time.time()
        r = ex.take(ticker, side, our_limit, 1, None, client_order_id=cid)   # the bot's own order path
        took = time.time() - t0
        exp_fee = kalshi_fee(1, our_limit)
        print(f"  take answer: status {r.status}, filled {r.filled}, avg price {r.avg_price}, fee {r.fee}, "
              f"order {r.order_id}, {took:.2f} s")
        state["takes"].append({"side": side, "ticker": ticker, "order_id": r.order_id, "cid": cid, "limit": our_limit,
                               "px_yes_scale": px})
        check(f"{side.upper()} buy filled 1 contract at once", r.status == "filled" and abs(r.filled - 1) < 1e-9,
              f"status {r.status}, filled {r.filled}")
        check(f"{side.upper()} buy price as the bot books it = {our_limit:.2f}",
              r.avg_price is not None and abs(r.avg_price - our_limit) < 0.005, f"booked {r.avg_price}")
        check(f"{side.upper()} buy fee as the bot books it = Kalshi's fee ({exp_fee:.4f})",
              abs(r.fee - exp_fee) < 0.00011, f"booked {r.fee:.4f}")
        # the raw order answer, for the record (which price scale Kalshi uses for a NO buy)
        try:
            raw = bot.order(r.order_id) if r.order_id else {}
            keep = {k: raw.get(k) for k in ("status", "side", "book_side", "outcome_side", "action", "fill_count_fp",
                                            "taker_fill_cost_dollars", "taker_fees_dollars", "yes_price_dollars",
                                            "no_price_dollars", "client_order_id") if k in raw}
            print(f"  order record: {json.dumps(keep)}")
        except Exception as e:
            print(f"  order record read failed: {e}")
    time.sleep(3)
    fills = [f for f in bot.fills(min_ts=start - 60) if f.get("order_id") in {t["order_id"] for t in state["takes"]}]
    for t in state["takes"]:
        fs = [f for f in fills if f.get("order_id") == t["order_id"]]
        cnt = sum(float(f.get("count_fp") or 0) for f in fs)
        check(f"{t['side'].upper()} buy: Kalshi's fill record shows 1 contract, as a taker", abs(cnt - 1) < 1e-9 and
              all(bool(f.get("is_taker")) for f in fs), f"{len(fs)} fill(s), {cnt} contracts")
        for f in fs:
            side = f.get("book_side") or ("bid" if f.get("outcome_side") == "yes" else "ask")
            want = "bid" if t["side"] == "yes" else "ask"
            yp = str_to_cents(f.get("yes_price_dollars"))
            check(f"{t['side'].upper()} buy: the fill feed books side {want} at YES {t['px_yes_scale']}c",
                  side == want and yp == t["px_yes_scale"], f"side {side}, yes price {yp}c")
            fee = float(f.get("fee_cost") or 0)
            check(f"{t['side'].upper()} buy: fee on the fill record matches", abs(fee - kalshi_fee(1, t["limit"])) <
                  0.00011, f"fee_cost {fee}")
            print(f"  fill record: {json.dumps({k: f.get(k) for k in ('fill_id', 'side', 'book_side', 'outcome_side', 'action', 'yes_price_dollars', 'no_price_dollars', 'count_fp', 'is_taker', 'fee_cost')})}")
    for oid in state["adv_orders"]:
        try:
            adv.cancel_order(oid)
        except Exception as e:
            print(f"  cancel of account 2's leftover {oid}: {e}")
    left = []
    for _ in range(6):            # account reads lag a few seconds behind a cancel
        time.sleep(3)
        left = [o for o in adv.orders(status="resting") if o.get("order_id") in set(state["adv_orders"])]
        if not left:
            break
    check("account 2's leftover offers cancelled", not left, f"{len(left)} still resting")
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    json.dump(state, open(STATE, "w"), indent=1)


def check_db():
    state = json.load(open(STATE))
    c = sqlite3.connect(f"file:{os.path.join(ROOT, 'data', 'demo', 'demo.sqlite')}?mode=ro", uri=True)
    for t in state["takes"]:
        rows = c.execute("SELECT sleeve, side, price_cents, count, is_taker, fee FROM fills WHERE ticker = ? AND "
                         "ts >= ?", (t["ticker"], state["start"] - 60)).fetchall()
        want = "bid" if t["side"] == "yes" else "ask"
        ok = len(rows) == 1 and rows[0][0] == "X" and rows[0][1] == want and rows[0][2] == t["px_yes_scale"] and \
            abs(rows[0][3] - 1) < 1e-9 and rows[0][4] == 1 and abs(rows[0][5] - kalshi_fee(1, t["limit"])) < 0.00011
        check(f"{t['side'].upper()} buy booked once by the running demo bot as a taker fill", ok, f"rows {rows}")


if __name__ == "__main__":
    check_db() if "--check-db" in sys.argv else run()
    bad = [n for n, ok in results if not ok]
    print(f"\nTAKER FILL TEST: {'PASS' if results and not bad else 'FAIL'}" + (f" ({len(bad)} failed)" if bad else ""))
    sys.exit(1 if bad or not results else 0)
