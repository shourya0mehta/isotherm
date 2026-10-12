"""Live smoke test for a new API key: place, see, amend and cancel tiny orders. Worst case it costs a few cents.

Steps: balance -> pick an open daily-high market whose best YES ask is >= 3c (so a 1c bid cannot cross)
-> post-only YES bid at 1c for 2 contracts (expires in 10 minutes even if everything else fails)
-> confirm it is resting -> amend it down to 1 contract, naming its expiry again (as the bot shrinks an order) -> amend
it back up to 2 with a later expiry (as the bot grows one, Oct 6) -> batch-cancel it and read the per-order
confirmation the bot relies on -> batch-cancel it again to see how a failed cancel is reported -> confirm it is gone.
After each amend the order must read back at its new size, with the bot's client id and the expiry the amend named
(the dead-man expiry must survive every amend).
Oct 4 additions (review findings 2.1/2.2/2.13): the account-data timestamp (how far account reads lag), the per-order
batch-cancel answer, and how a post-only order that would cross is refused (one contract bid at a 2-10c YES ask, priced
from a fresh read of that book; if the ask moves away before the order arrives, the order rests without ever crossing,
so it is cancelled and tried once more, then reported as not measured rather than failed; a fill fails; if post-only
were ignored it would cost at most 10c). The answers are printed and stored (kv api_observations).
Round-2 addition: whether the account-data timestamp keeps moving while the account is idle (20 s more).
Oct 4 addition: the ask side Sleeve A uses (a post-only YES ask at 99c, i.e. a NO bid at 1c) must read back as
book_side ask at YES price 0.99 and hold about 1c a contract; at most 2 more cents if it filled.

It also measures what Kalshi's "available balance" means for the bot's equity: the balance is read before the
order, while it rests and after the cancel. If resting the order lowers the balance (cash held for it), the bot adds
its resting orders' cash back when it computes equity for the stops; if not, it does not. The answer is stored in the
bot's database (kv balance_semantics), so a live start never has to guess.
"""
from __future__ import annotations

import os
import time
import uuid

from ..execution.live import PREFIX, _iso_ts
from ..kalshi.auth import KalshiAuth
from ..kalshi.rest import DEMO, PROD, KalshiREST, str_to_cents


def settled_balance(r, done, reads: int = 4, gap_s: float = 1.5) -> int:
    """The available balance in cents, re-read a few times (until `done(value)`) in case the ledger updates late."""
    v = int(r.balance().get("balance") or 0)
    for _ in range(reads - 1):
        if done(v):
            break
        time.sleep(gap_s)
        v = int(r.balance().get("balance") or 0)
    return v


def expiry_is(order: dict, ts: int) -> bool:
    """The order read back carries the expiry an amend named (Kalshi keeps whole seconds)."""
    got = _iso_ts(order.get("expiration_time"))
    return bool(got) and abs(got - ts) <= 1.0


def run_smoke(cfg: dict, series: str | None = None) -> bool:
    series = series or ((cfg.get("sleeve_a") or {}).get("series") or ["KXHIGHNY"])[0]
    auth = KalshiAuth.from_env()
    if auth is None:
        print("[BAD] no API key in the environment (KALSHI_KEY_ID + KALSHI_PRIVATE_KEY_PATH). See docs/SETUP_KEYS.md")
        return False
    base = DEMO if cfg.get("env") == "demo" else os.environ.get("KALSHI_BASE_URL", PROD)
    r = KalshiREST(base_url=base, auth=auth)
    ok = True
    oid = None

    def line(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print(f"[{'OK ' if good else 'BAD'}] {name}: {detail}")
        return good

    try:
        b = r.balance()
        if not line("auth + balance", True, f"${(b.get('balance') or 0) / 100:.2f} available"):
            return False
        evs = r.events(series, status="open", nested=True, limit=3)
        tickers = [m["ticker"] for e in evs for m in e.get("markets", []) if m.get("status") in ("active", "open")]
        books = r.orderbooks(tickers)
        pick = None
        for t in tickers:
            nb = books.get(t, {}).get("no_dollars") or []
            best_no_bid = max((str_to_cents(p) for p, q in nb if float(q) > 0), default=None)
            yes_ask = 100 - best_no_bid if best_no_bid is not None else None
            if yes_ask is None or yes_ask >= 3:
                pick = t
                break
        if not line("pick market", pick, pick or "no market with YES ask >= 3c"):
            return False
        cid = PREFIX + "smoke-" + uuid.uuid4().hex[:12]
        cents = [int(r.balance().get("balance") or 0)]
        exp0 = int(time.time()) + 600
        o = r.create_order(pick, "bid", 1, 2, post_only=True, expiration_ts=exp0, client_order_id=cid)
        oid = o.get("order_id")
        line("create post-only 1c YES bid x2", oid, f"order_id={oid} remaining={o.get('remaining_count')}")
        mine = []
        for _ in range(4):                 # account reads lag the matching engine a little: look up to four times
            time.sleep(1.5)
            mine = [x for x in r.orders(status="resting", ticker=pick) if x.get("order_id") == oid]
            if mine:
                break
        line("order visible as resting", mine, f"{len(mine)} found, client_order_id={cid}")
        cents.append(settled_balance(r, lambda v: v != cents[0]))
        a = r.amend_order(oid, pick, "bid", 1, 1, expiration_ts=exp0)    # as the bot shrinks: its expiry named again
        oid = a.get("order_id") or oid
        obs = {}
        # the bot amends every resting order every couple of minutes (expiry refresh) without naming a new client
        # order id, and knows its own orders by their wxb- client id after a restart: the amended order must keep it
        # (Kalshi's docs do not say what an amend without updated_client_order_id does; round-7 review)
        # judged only on a read-back that already shows the amended size (an older copy would still carry the old
        # client id); an empty client id in the answer means none was named (round-8 review)
        after = []
        for _ in range(6):
            time.sleep(1.5)
            after = [x for x in r.orders(status="resting", ticker=pick) if x.get("order_id") == oid
                     and abs(float(x.get("remaining_count_fp") or x.get("remaining_count") or 0) - 1.0) < 1e-6]
            if after:
                break
        # the amend took if its answer says 1 left or the account shows the order at 1 (the answer's remaining_count
        # is only "present when the resting size changed", per Kalshi's docs; round-9 review)
        answer_rem = a.get("remaining_count") if a.get("remaining_count") is not None else a.get("remaining_count_fp")
        line("amend to 1 contract", (answer_rem is not None and float(answer_rem) == 1.0) or bool(after),
             f"remaining={answer_rem}" + (" (read back at 1)" if after else ""))
        answer_cid = a.get("client_order_id") or None
        kept = after[0].get("client_order_id") if after else None
        obs["amend_client_order_id"] = {"answer": answer_cid, "read_back": kept}
        if answer_cid not in (None, cid) or (after and kept != cid):
            line("amended order keeps its client id", False,
                 f"Kalshi changed it to {kept if after and kept != cid else answer_cid!r}: the bot would not know "
                 "its amended orders after a restart; do not go live, send me these lines")
        elif after:
            line("amended order keeps its client id", True, f"{cid} read back after the amend")
        else:
            print("[-- ] amended order keeps its client id: not checked (the account reads did not show the amended "
                  "order in time); the bot also knows its orders by order id")
        if after:
            line("amended order keeps the expiry it names", expiry_is(after[0], exp0),
                 f"expiration_time {after[0].get('expiration_time')} (named {exp0})")
        # the bot grows an order instead of placing a second one beside it (Oct 6 demo test): one amend to a larger
        # size and a later expiry. Kalshi moves it to the back of its price's queue; it must rest at the new size with
        # the bot's client id and the new expiry
        exp1 = exp0 + 60
        a2 = r.amend_order(oid, pick, "bid", 1, 2, expiration_ts=exp1)
        oid = a2.get("order_id") or oid
        up = []
        for _ in range(6):
            time.sleep(1.5)
            up = [x for x in r.orders(status="resting", ticker=pick) if x.get("order_id") == oid
                  and abs(float(x.get("remaining_count_fp") or x.get("remaining_count") or 0) - 2.0) < 1e-6]
            if up:
                break
        up_rem = a2.get("remaining_count") if a2.get("remaining_count") is not None else a2.get("remaining_count_fp")
        obs["amend_up"] = {"answer": a2, "read_back": up[0] if up else None}
        line("amend up to 2 contracts", (up_rem is not None and float(up_rem) == 2.0) or bool(up),
             f"remaining={up_rem}" + (" (read back at 2)" if up else ""))
        if up:
            line("grown order keeps its client id and takes the new expiry",
                 up[0].get("client_order_id") == cid and expiry_is(up[0], exp1),
                 f"client id {up[0].get('client_order_id')}, expiration_time {up[0].get('expiration_time')} "
                 f"(named {exp1})")
        else:
            print("[-- ] grown order's client id and expiry: not checked (the account reads did not show it at 2 in "
                  "time)")
        try:
            as_of = r.user_data_timestamp()
            obs["user_data_lag_s"] = round(time.time() - as_of, 2) if as_of else None
            line("account-data timestamp", as_of, f"account reads lag about {obs['user_data_lag_s']} s")
        except Exception as e:
            line("account-data timestamp", False, repr(e))
        c = r.batch_cancel([oid]) or {}
        x = (c.get("orders") or [{}])[0]
        confirmed = bool(x) and not x.get("error") and (x.get("ts_ms") is not None
                                                        or float(x.get("reduced_by") or 0) > 0)
        obs["batch_cancel_ok"] = x
        line("batch cancel confirmed per order", confirmed, f"{x}")
        c2 = r.batch_cancel([oid]) or {}
        x2 = (c2.get("orders") or [{}])[0]
        obs["batch_cancel_again"] = x2
        print(f"[-- ] a second cancel of the same order is answered: {x2}")
        left = []
        for _ in range(4):
            time.sleep(1.5)
            left = [x for x in r.orders(status="resting", ticker=pick) if x.get("order_id") == oid]
            if not left:
                break
        line("order gone", not left, "no longer resting" if not left else "STILL RESTING: cancel it in the app")
        if not left:
            oid = None
            cents.append(settled_balance(r, lambda v: v == cents[0]))
            held, back = cents[0] - cents[1], cents[2] - cents[1]
            if held >= 2 and back >= 2:
                sem = True
            elif held == 0 and back == 0:
                sem = False
            else:
                sem = None
            detail = (f"balance {cents[0]}c before, {cents[1]}c while 2 contracts at 1c rested, {cents[2]}c after "
                      f"the cancel: " + {True: "the available balance leaves out cash held by resting orders (the "
                                               "bot adds it back for its equity)",
                                         False: "the available balance still counts resting orders",
                                         None: "unclear (something else moved the balance): run the smoke test "
                                               "again"}[sem])
            if line("balance semantics", sem is not None, detail) and cfg.get("db_path"):
                from ..store.db import Store
                Store(cfg["db_path"]).kv_set("balance_semantics", {"excludes_resting": sem, "cents": cents,
                                                                   "ts": time.time(), "env": cfg.get("env")})
                print(f"[OK ] stored in {cfg['db_path']} (kv balance_semantics)")
        # the ask side, which is how Sleeve A buys NO: a post-only YES ask at 99c (= a NO bid at 1c) where the best
        # YES bid is below 99c. It must read back as book_side ask at YES price 0.99 and hold about 1c a contract
        # (the NO cost), never 99c: that is the price convention every Sleeve A order relies on. At most 2 cents.
        ask_pick = None
        for t in tickers:
            yb = books.get(t, {}).get("yes_dollars") or []
            best_yes_bid = max((str_to_cents(p) for p, q in yb if float(q) > 0), default=0)
            if best_yes_bid < 97:
                ask_pick = t
                break
        if ask_pick:
            oid3 = None
            try:
                c0 = int(r.balance().get("balance") or 0)
                o3 = r.create_order(ask_pick, "ask", 99, 2, post_only=True, expiration_ts=int(time.time()) + 600,
                                    client_order_id=PREFIX + "smoke-a-" + uuid.uuid4().hex[:10])
                oid3 = o3.get("order_id")
                g = {}
                for _ in range(5):                    # the orders read can lag the write by a second or two
                    time.sleep(1.5)
                    got = [x for x in r.orders(status="resting", ticker=ask_pick) if x.get("order_id") == oid3]
                    if got:
                        g = got[0]
                        break
                c1 = settled_balance(r, lambda v: v != c0)
                side = g.get("book_side") or {"no": "ask", "yes": "bid"}.get(g.get("outcome_side"))
                convention = side == "ask" and str_to_cents(g.get("yes_price_dollars")) == 99
                # cancel, then see what the cancel gives back: the hold is measured twice, so an unrelated debit
                # in the window shows up as a mismatch instead of a wrong verdict
                r.cancel_order(oid3)
                oid3 = None
                c2 = settled_balance(r, lambda v: v != c1)
                held, back = c0 - c1, c2 - c1
                obs["ask_side"] = {"book_side": g.get("book_side"), "outcome_side": g.get("outcome_side"),
                                   "yes_price": g.get("yes_price_dollars"), "held_cents": held, "back_cents": back}
                wrong_hold = held >= 50 and back >= 50   # about 99c a contract: the YES price, not the NO cost
                if wrong_hold:
                    how = "Kalshi holds about the YES price for a YES ask: the bot's equity would be wrong"
                elif 1 <= held <= 4 and 1 <= back <= 4:
                    how = "about 1c a contract, as the bot counts it"
                elif held == 0 and back == 0:
                    how = "not measured (the available balance does not leave out resting orders)"
                else:
                    how = "not measured cleanly (something else moved the balance); the side and price read back fine"
                line("ask side = NO bid at 1 - price", convention and not wrong_hold,
                     f"book_side {g.get('book_side')}, outcome {g.get('outcome_side')}, YES price "
                     f"{g.get('yes_price_dollars')}; cash held {held}c, given back {back}c for 2 contracts: {how}")
            except Exception as e:
                line("ask side = NO bid at 1 - price", False, repr(e))
            finally:
                if oid3:
                    try:
                        r.cancel_order(oid3)
                    except Exception as e:
                        print(f"[--] cleanup cancel of the ask failed ({e!r}); it expires within 10 minutes")
        # how a post-only order that would cross is refused (the bot must not count it as an API error). The price
        # comes from a fresh read of that market's book just before the order (the read at the start is several
        # seconds old by now), and an order that rests is judged against the book after it: if the ask has moved
        # above the order's price, the order never crossed and says nothing about post-only, so it is tried once
        # more, then reported as not measured instead of blocking GO_LIVE (round-6 review). A fill always fails.
        from ..execution.live import reject_kind

        def yes_ask(book):
            nb = (book or {}).get("no_dollars") or []
            best = max((str_to_cents(p) for p, q in nb if float(q) > 0), default=None)
            return None if best is None else 100 - best

        def fresh_ask(t):
            try:
                return yes_ask(r.orderbooks([t]).get(t))
            except Exception:
                return None
        cands = [t for t in tickers if 2 <= (yes_ask(books.get(t)) or 0) <= 10]
        measured = False
        for attempt in range(2):
            cross = None
            for t in cands:
                a = fresh_ask(t)
                if a is not None and 2 <= a <= 10:
                    cross = (t, a)
                    break
            if cross is None:
                break
            try:
                o2 = r.create_order(cross[0], "bid", cross[1], 1, post_only=True, expiration_ts=int(time.time()) + 120,
                                    client_order_id=PREFIX + "smoke-x-" + uuid.uuid4().hex[:10])
            except Exception as e:
                kind = reject_kind(e)
                obs["post_only_cross"] = {"status": getattr(e, "status", None), "body": getattr(e, "body", str(e))[:300],
                                          "kind": kind}
                line("post-only order at the ask is refused", kind == "post_only",
                     f"{getattr(e, 'status', '?')} {getattr(e, 'body', e)!s:.200} -> the bot reads it as '{kind}'")
                measured = True
                break
            oid2 = o2.get("order_id")
            rem = o2.get("remaining_count")
            filled = float(o2.get("fill_count") or 0)
            obs["post_only_cross"] = {"accepted": o2}
            if rem is not None and float(rem) <= 0 and filled <= 0:
                # Kalshi's FIX docs report this case as cancelled on arrival (POST_ONLY_CROSS); the bot treats a
                # success answer with nothing resting like a refusal (cooldown, no order tracked)
                line("post-only order at the ask is refused", True,
                     f"accepted and cancelled at once ({o2}): the bot treats this as a post-only refusal")
                measured = True
                break
            if oid2 and (rem is None or float(rem) > 0):        # whatever rests goes, also after a partial fill
                try:
                    r.cancel_order(oid2)
                except Exception as e:
                    print(f"[--] cleanup cancel of the crossing order failed ({e!r}); it expires within 2 minutes")
            if filled > 0:
                line("post-only order at the ask is refused", False,
                     f"it was ACCEPTED and FILLED: {o2}; post-only is not holding, do not go live")
                measured = True
                break
            now_ask = fresh_ask(cross[0])
            if now_ask is None or now_ask > cross[1]:
                print(f"[-- ] post-only check: the ask moved from {cross[1]}c to "
                      f"{'nothing' if now_ask is None else str(now_ask) + 'c'} before the order arrived, so it never "
                      f"crossed (try {attempt + 1} of 2); cancelled")
                continue
            line("post-only order at the ask is refused", False,
                 f"it was ACCEPTED and rests while the ask is still {now_ask}c: {o2}; post-only is not holding, do not "
                 "go live; cancelled")
            measured = True
            break
        if cands and not measured:
            print("[-- ] post-only check: not measured (the ask moved away before the order arrived); the bot handles "
                  "a refusal and a cancel-on-arrival alike")
        # does the account-data timestamp move while the account is idle (an exchange-wide clock, as unsigned reads
        # show) or only with account activity? The bot's freshness check depends on the answer (round-2 review)
        try:
            time.sleep(5)
            t1 = r.user_data_timestamp()
            time.sleep(15)
            t2 = r.user_data_timestamp()
            moved = (t2 - t1) if (t1 and t2) else None
            obs["user_data_ts_moves_idle"] = None if moved is None else bool(moved >= 10)
            line("account-data timestamp moves while idle", moved is not None,
                 f"moved {moved:.1f} s in 15 s with no account activity" if moved is not None else "not readable")
            if moved is not None and moved < 10:
                print("[-- ] the bot will judge freshness by its read times alone (kv api_observations)")
        except Exception as e:
            line("account-data timestamp moves while idle", False, repr(e))
        if cfg.get("db_path"):
            from ..store.db import Store
            Store(cfg["db_path"]).kv_set("api_observations", {**obs, "ts": time.time()})
    except Exception as e:
        line("smoke test", False, repr(e))
    finally:
        if oid:
            try:
                r.cancel_order(oid)
                print(f"[--] cleanup: cancelled {oid}")
            except Exception as e:
                print(f"[--] cleanup cancel failed ({e!r}); the order expires on its own within 10 minutes")
    print("SMOKE TEST", "PASSED" if ok else "FAILED")
    return ok
