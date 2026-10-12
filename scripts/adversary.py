"""Troublemaker (Shourya's idea, Oct 6): a second DEMO account that trades against the demo bot on purpose, to put it
in the situations a quiet demo exchange never produces. Fake money only.

Each round makes one move against the bot's resting orders (read with the bot's demo key, read-only):
  nibble    take part of one order (a partial fill)
  swallow   take a whole order
  sweep     take the bot's orders across one event at once (several fills in one pass)
  rapid     three small takes from one order a third of a second apart
  chase     wait for the bot to place a new order, then take it at once
  wall      rest an order one cent inside the bot's quote on the other side for a few minutes: if the bot moves
            there, its post-only order must be refused, never filled
  undercut  rest an order one cent better than the bot's, on its side, for a few minutes
  dust      take a fraction of a contract
Then it checks, from Kalshi and from the bot's own record: the bot booked every fill Kalshi shows for its orders;
no event, market or total went over the bot's caps (config/demo.yaml); the bot is up, not halted, and logged no new
errors. The demo gate (scripts/demo_gate.py) checks everything else (fills, equity, duplicates, post-only).

On the server (scripts/vm_trouble.sh runs it as the service wxbot-adversary):
  .venv/bin/python scripts/adversary.py once [MOVE]   one round now
  .venv/bin/python scripts/adversary.py run           a round every 15 to 35 minutes
  .venv/bin/python scripts/adversary.py report        what it did, and every check that failed
Stop: touch data/demo/ADVERSARY_STOP (it cancels its orders and exits; the service stays down until the flag goes).
Phone alerts (the bot's ntfy topic, titled "wxbot DEMO"): a failed check, the demo bot silent for about ten minutes
while Kalshi trades (it watches the bot from outside), or three failed rounds in a row; each at most every 3 hours.
Safety: Kalshi's demo exchange only (the address is fixed here, whatever the environment says); it refuses to run
if account 2's key is the bot's or a real one, and never writes with the bot's key. At most 3 contracts and $3 a
round, 12 contracts and $15 a day, 4 contracts per event a day (fake money, and gentle enough that the bot keeps
quoting for the two-day test); never within 35 minutes of a market's close. Its resting orders expire within 10
minutes and are cancelled when a round ends and on exit.
"""
import datetime as dt
import json
import logging
import math
import os
import random
import signal
import sqlite3
import sys
import time
import uuid
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

from wxbot.kalshi.auth import KalshiAuth, load_private_key  # noqa: E402
from wxbot.kalshi.rest import DEMO, KalshiREST, str_to_cents  # noqa: E402
from wxbot.timeutil import et_day, parse_ts  # noqa: E402

KEYS = os.path.join(os.path.dirname(ROOT), "deploy_keys")
DATA = os.path.join(ROOT, "data", "demo")
PREFIX = "adv-"
MOVES = {"nibble": 3, "swallow": 1, "sweep": 1, "rapid": 2, "chase": 1, "wall": 2, "undercut": 2, "dust": 2}
TAKES = {"nibble", "swallow", "sweep", "rapid", "chase", "dust"}
LIMITS = {"round_contracts": 3.0, "round_usd": 3.0, "day_contracts": 12.0, "day_usd": 15.0,
          "event_day_contracts": 4.0, "close_margin_s": 35 * 60, "rest_ttl_s": 600, "settle_s": 30}
MONEY_WORDS = ("realized", "pnl", "p&l", "equity", "loss")
log = logging.getLogger("adversary")


def path(name):
    return os.path.join(DATA, name)


# ----------------------------------------------------------------------------------------------- keys and clients
class ReadOnly(KalshiREST):
    """The bot's demo key, for reads only: any write is refused before it is signed or sent."""

    def _request(self, method, p, *a, **kw):
        if method.upper() != "GET" or kw.get("write"):
            raise RuntimeError(f"read-only client: {method} {p} refused")
        return super()._request(method, p, *a, **kw)


def key_files(name, keys=None):
    keys = keys or KEYS
    key = next((os.path.join(keys, name + e) for e in (".key", ".pem") if os.path.exists(os.path.join(keys, name + e))),
               None)
    idf = os.path.join(keys, name + "_key_id.txt")
    if key is None or not os.path.exists(idf):
        return None
    return open(idf).read().strip(), open(key, "rb").read()


def clients(keys=None):
    """(account 2 for trading, the bot's account read-only), both on Kalshi's demo exchange."""
    keys = keys or KEYS
    adv, bot = key_files("kalshi_demo2", keys), key_files("kalshi_demo", keys)
    if adv is None:
        raise SystemExit(f"no key for demo account 2 in {keys} (kalshi_demo2.key + kalshi_demo2_key_id.txt)")
    if bot is None:
        raise SystemExit(f"no demo bot key in {keys} (kalshi_demo.key + kalshi_demo_key_id.txt)")
    real = [open(os.path.join(keys, f), "rb").read() for f in ("kalshi_live.key", "kalshi_live.pem")
            if os.path.exists(os.path.join(keys, f))]
    if adv[1] == bot[1] or adv[0] == bot[0] or adv[1] in real:
        raise SystemExit("demo account 2's key is the bot's key or a real key: stopping (nothing was sent)")
    return (KalshiREST(base_url=DEMO, auth=KalshiAuth(adv[0], load_private_key(adv[1]))),
            ReadOnly(base_url=DEMO, auth=KalshiAuth(bot[0], load_private_key(bot[1]))))


# ------------------------------------------------------------------------------------------------------ helpers
def event_of(ticker):
    return ticker.rsplit("-", 1)[0]


def unit_cost(side, px):
    """Dollars a contract costs the side that buys it: YES at px (bid) or NO at 100 - px (ask)."""
    return (px if side == "bid" else 100 - px) / 100.0


def bot_orders(bot):
    out = []
    for o in bot.orders(status="resting"):
        cid = str(o.get("client_order_id") or "")
        if not cid.startswith("wxb-") or "smoke" in cid:
            continue
        side = o.get("book_side") or ("bid" if o.get("outcome_side") == "yes" else "ask")
        out.append({"id": o["order_id"], "ticker": o["ticker"], "side": side,
                    "px": str_to_cents(o.get("yes_price_dollars")), "rem": float(o.get("remaining_count_fp") or 0)})
    return [o for o in out if o["px"] and o["rem"] > 0]


def annotate(adv, orders):
    """Each order with its market's best YES bid and ask (cents; None when that side is empty) and the size resting at
    its own price, from the public order book (a YES ask at px rests there as a NO bid at 100 - px)."""
    if not orders:
        return []
    books = adv.orderbooks(sorted({o["ticker"] for o in orders}))
    out = []
    for o in orders:
        b = books.get(o["ticker"]) or {}
        bids = {str_to_cents(p): float(q) for p, q in (b.get("yes_dollars") or []) if float(q) > 0}
        asks = {100 - str_to_cents(p): float(q) for p, q in (b.get("no_dollars") or []) if float(q) > 0}
        out.append(dict(o, best_bid=max(bids, default=None), best_ask=min(asks, default=None),
                        level=(asks if o["side"] == "ask" else bids).get(o["px"], 0.0)))
    return out


def at_top(adv, orders):
    """The bot orders that are the best price on their side, so a take at their price fills them rather than someone
    else's better order (the first live round, Oct 6: a take at the bot's 4c ask got 2 contracts from another ask at
    3c first). Others at the same price may still be ahead of the bot in the queue ('level' is the size there)."""
    return [o for o in annotate(adv, orders) if o[{"ask": "best_ask", "bid": "best_bid"}[o["side"]]] == o["px"]]


def rest_price(o, move):
    """(side, price) of a one-contract order one cent off a bot order: 'wall' on the other side, inside its price;
    'undercut' on its side, one cent better. None when that would cross the book (a post-only order is refused) or
    leave 1-99c. The second live round (Oct 6) put a wall where another trader's ask sat below the bot's."""
    if move == "wall":
        side, px = ("bid", o["px"] - 1) if o["side"] == "ask" else ("ask", o["px"] + 1)
    else:
        side, px = (o["side"], o["px"] - 1) if o["side"] == "ask" else (o["side"], o["px"] + 1)
    if not 1 <= px <= 99:
        return None
    if side == "bid" and o.get("best_ask") is not None and o["best_ask"] <= px:
        return None
    if side == "ask" and o.get("best_bid") is not None and o["best_bid"] >= px:
        return None
    return side, px


class Markets:
    """Close times and event tickers (public reads), cached."""

    def __init__(self, rest):
        self.rest, self.close, self.events = rest, {}, {}

    def closes_in(self, ticker):
        if ticker not in self.close:
            self.close[ticker] = parse_ts(self.rest.market(ticker).get("close_time")) or 0.0
        return self.close[ticker] - time.time()

    def tickers(self, event):
        if event not in self.events:
            self.events[event] = [m["ticker"] for m in self.rest.markets(event_ticker=event)]
        return self.events[event]


def journal_rows():
    try:
        return [json.loads(x) for x in open(path("adversary.jsonl")) if x.strip()]
    except OSError:
        return []


def used_today(rows=None, today=None):
    """(contracts, dollars, contracts per event) this account took from the bot today (ET)."""
    today = today or et_day().isoformat()
    n, usd, per_ev = 0.0, 0.0, defaultdict(float)
    for r in rows if rows is not None else journal_rows():
        if et_day(r.get("ts", 0)).isoformat() != today:
            continue
        for a in r.get("actions", []):
            if a.get("act") == "take" and a.get("got"):
                n += a["got"]
                usd += a["got"] * unit_cost(a["side"], a["px"])
                per_ev[event_of(a["ticker"])] += a["got"]
    return n, usd, per_ev


class Budget:
    """What a round may still take: its own limits and what is left of today's."""

    def __init__(self, used=None):
        n, usd, per_ev = used or used_today()
        self.contracts = min(LIMITS["round_contracts"], LIMITS["day_contracts"] - n)
        self.usd = min(LIMITS["round_usd"], LIMITS["day_usd"] - usd)
        self.ev_left = defaultdict(lambda: LIMITS["event_day_contracts"])
        for ev, k in per_ev.items():
            self.ev_left[ev] -= k

    def allow(self, o, n):
        """How many of n contracts against bot order o fit every limit (two decimals, rounded down)."""
        side = "bid" if o["side"] == "ask" else "ask"
        m = min(n, o["rem"], self.contracts, self.ev_left[event_of(o["ticker"])],
                self.usd / unit_cost(side, o["px"]))
        return max(0.0, math.floor(m * 100 + 1e-9) / 100)

    def spend(self, o, got):
        side = "bid" if o["side"] == "ask" else "ask"
        self.contracts -= got
        self.usd -= got * unit_cost(side, o["px"])
        self.ev_left[event_of(o["ticker"])] -= got

    @property
    def any(self):
        return self.contracts >= 0.01 and self.usd > 0.01


# ------------------------------------------------------------------------------------------------------ actions
def take(adv, o, n, move, budget):
    """Immediate-or-cancel against one bot order at its price: buy YES at its ask, or sell YES at its bid."""
    side = "bid" if o["side"] == "ask" else "ask"
    rec = {"act": "take", "move": move, "ticker": o["ticker"], "side": side, "px": o["px"], "asked": n,
           "bot_order": o["id"], "bot_rem": o["rem"], "level": o.get("level")}
    if n < 0.01:
        rec["skipped"] = "no budget left"
        return rec
    t0 = time.time()
    try:
        r = adv.create_order(o["ticker"], side, o["px"], n, post_only=False, tif="immediate_or_cancel",
                             client_order_id=f"{PREFIX}{move[:5]}-{uuid.uuid4().hex[:12]}", retries=0)
        rec.update(got=float(r.get("fill_count") or r.get("fill_count_fp") or 0), order=r.get("order_id"))
    except Exception as e:
        rec.update(got=0.0, error=f"{getattr(e, 'status', '')} {getattr(e, 'body', e)!s}"[:240])
    rec["ms"] = round(1000 * (time.time() - t0))
    budget.spend(o, rec["got"])
    log.info("%s: took %g of %g asked at %sc on %s (%s)", move, rec["got"], n, o["px"], o["ticker"],
             rec.get("error", "ok"))
    return rec


def rest(adv, ticker, side, px, move):
    """A resting post-only order of one contract that expires on its own within rest_ttl_s."""
    rec = {"act": "rest", "move": move, "ticker": ticker, "side": side, "px": px}
    try:
        r = adv.create_order(ticker, side, px, 1, post_only=True,
                             expiration_ts=int(time.time()) + LIMITS["rest_ttl_s"],
                             client_order_id=f"{PREFIX}{move[:5]}-{uuid.uuid4().hex[:12]}", retries=0)
        rec.update(order=r.get("order_id"), resting=float(r.get("remaining_count") or 0))
    except Exception as e:
        rec.update(error=f"{getattr(e, 'status', '')} {getattr(e, 'body', e)!s}"[:240])
    log.info("%s: rest %s 1 at %sc on %s (%s)", move, side, px, ticker, rec.get("error", "ok"))
    return rec


def cancel_mine(adv):
    mine = [o["order_id"] for o in adv.orders(status="resting")
            if str(o.get("client_order_id") or "").startswith(PREFIX)]
    if mine:
        adv.batch_cancel(mine)
    return len(mine)


def watch(bot, ticker, hold_s, sleep=time.sleep):
    """What the bot's orders on one market did while a resting order of ours sat beside them."""
    seen = []
    t_end = time.time() + hold_s
    while time.time() < t_end:
        sleep(min(15.0, max(0.0, t_end - time.time())))
        snap = sorted((o["side"], o["px"], o["rem"]) for o in bot_orders(bot) if o["ticker"] == ticker)
        if not seen or seen[-1][1] != snap:
            seen.append((round(time.time()), snap))
    return [{"ts": t, "bot": s} for t, s in seen]


def play_move(move, adv, bot, orders, budget, rng, sleep=time.sleep, far=lambda o: True):
    """One move's actions. far(order): the order's market is far enough from its close to trade (new orders only)."""
    if move == "nibble":
        o = rng.choice([x for x in orders if x["rem"] >= 1] or orders)
        return [take(adv, o, budget.allow(o, max(0.5, round(o["rem"] * rng.uniform(0.2, 0.6), 2))), move, budget)]
    if move == "swallow":
        small = [x for x in orders if x["rem"] <= budget.contracts] or orders
        o = rng.choice(small)
        return [take(adv, o, budget.allow(o, o["rem"]), move, budget)]
    if move == "sweep":
        by_ev = defaultdict(list)
        for x in orders:
            by_ev[event_of(x["ticker"])].append(x)
        ev = max(by_ev, key=lambda e: (len(by_ev[e]), rng.random()))
        return [take(adv, o, budget.allow(o, o["rem"]), move, budget) for o in by_ev[ev]]
    if move == "rapid":
        o = rng.choice([x for x in orders if x["rem"] >= 1.5] or orders)
        out = []
        for _ in range(3):
            out.append(take(adv, o, budget.allow(o, min(1.0, o["rem"] / 3)), move, budget))
            o = dict(o, rem=max(o["rem"] - out[-1]["got"], 0.0))
            sleep(0.33)
        return out
    if move == "chase":
        known = {x["id"] for x in orders}
        t_end = time.time() + 180
        while time.time() < t_end:
            sleep(0.7)
            new = at_top(adv, [x for x in bot_orders(bot) if x["id"] not in known and far(x)])
            if new:
                o = new[0]
                return [take(adv, o, budget.allow(o, min(o["rem"], 2.0)), move, budget)]
        return [{"act": "wait", "move": move, "note": "the bot placed no new order in 3 minutes"}]
    if move in ("wall", "undercut"):
        cands = [o for o in annotate(adv, orders) if rest_price(o, move)]
        if not cands:
            return [{"act": "skip", "move": move, "note": "no room beside the bot's orders without crossing the book"}]
        o = rng.choice(cands)
        side, px = rest_price(o, move)
        rec = rest(adv, o["ticker"], side, px, move)
        if rec.get("order"):
            rec["bot_seen"] = watch(bot, o["ticker"], rng.uniform(150, 300), sleep=sleep)
        return [rec]
    if move == "dust":
        o = rng.choice(orders)
        return [take(adv, o, budget.allow(o, rng.choice([0.01, 0.1, 0.37, 0.5])), move, budget)]
    raise ValueError(f"unknown move {move}")


# ------------------------------------------------------------------------------------------------------- checks
def caps():
    import yaml
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "demo.yaml"))) or {}
    r = cfg.get("risk") or {}
    return {"event": float(r.get("max_loss_per_event_usd", 10)), "bracket": float(r.get("max_loss_per_bracket_usd", 5)),
            "total": float(r.get("max_total_open_risk_usd", 40))}


def check_caps(bot, markets, limits=None, tol=0.10):
    """Worst case per event, cost per market and their total, from Kalshi's positions and the bot's resting orders,
    computed the bot's way (wxbot/risk/limits.py) but from the exchange's numbers."""
    from wxbot.risk.limits import bracket_cost, event_worst_case
    from wxbot.strategy.types import DesiredOrder, Position
    limits = limits or caps()
    mp, _ = bot.positions()
    pos = defaultdict(list)
    for m in mp:
        q, c = float(m.get("position_fp") or 0), abs(float(m.get("market_exposure_dollars") or 0))
        if q:
            p = Position(m["ticker"], event_of(m["ticker"]), yes_qty=max(q, 0), yes_cost=c if q > 0 else 0,
                         no_qty=max(-q, 0), no_cost=c if q < 0 else 0)
            pos[p.event].append(p)
    pend = defaultdict(list)
    for o in bot_orders(bot):
        pend[event_of(o["ticker"])].append(DesiredOrder(o["ticker"], o["side"], o["px"], o["rem"], "A", 0.0))
    over, worst = [], {}
    for ev in set(pos) | set(pend):
        tick = markets.tickers(ev)
        worst[ev] = event_worst_case(pos[ev], pend[ev], tick)
        if worst[ev] > limits["event"] + tol:
            over.append(f"event {ev}: worst case ${worst[ev]:.2f} > ${limits['event']:.0f}")
        for t in {p.ticker for p in pos[ev]} | {o.ticker for o in pend[ev]}:
            c = bracket_cost(pos[ev], pend[ev], t)
            if c > limits["bracket"] + tol:
                over.append(f"market {t}: ${c:.2f} > ${limits['bracket']:.0f}")
    total = sum(worst.values())
    if total > limits["total"] + tol:
        over.append(f"total worst case ${total:.2f} > ${limits['total']:.0f}")
    return {"ok": not over, "over": over, "total": round(total, 2), "events": len(worst)}


def check_booked(bot, since, db=None):
    """Every fill Kalshi shows for the bot's orders since `since` is in the bot's database."""
    kal = {str(f.get("fill_id") or f.get("trade_id")): float(f.get("count_fp") or 0)
           for f in bot.fills(min_ts=int(since) - 5)}
    try:
        c = sqlite3.connect(f"file:{db or path('demo.sqlite')}?mode=ro", uri=True)
        booked = {r[0] for r in c.execute("SELECT order_ref FROM fills WHERE mode='live' AND ts >= ?", (since - 300,))}
        c.close()
    except sqlite3.Error as e:
        return {"ok": False, "error": str(e)}
    missing = sorted(k for k in kal if k not in booked)
    return {"ok": not missing, "kalshi_fills": len(kal), "contracts": round(sum(kal.values()), 2), "missing": missing}


def check_health(since, status=None, log_path=None):
    try:
        st = json.load(open(status or path("demo_status.json")))
    except (OSError, ValueError):
        st = {}
    age = time.time() - float(st.get("ts") or 0)
    errs = []
    try:
        for line in open(log_path or path("demo.log"), errors="replace"):
            if " ERROR " not in line[:40]:
                continue
            try:
                ts = time.mktime(time.strptime(line[:19], "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                continue
            if ts >= since:
                msg = line[24:].strip()
                errs.append("[an error line about money]" if any(w in msg.lower() for w in MONEY_WORDS) else msg[:160])
    except OSError:
        pass
    ok = age < 120 and not st.get("halted")
    return {"ok": ok, "status_age_s": round(age), "halted": bool(st.get("halted")),
            "halt_reason": (st.get("halt_reason") or "")[:120], "new_errors": len(errs), "errors": errs[:3]}


# -------------------------------------------------------------------------------------------------------- rounds
def play(adv, bot, move=None, rng=None, sleep=time.sleep, markets=None, used=None):
    """One round: a move, our leftover orders cancelled, a pause for the bot to catch up, then the checks."""
    rng = rng or random.Random()
    markets = markets or Markets(adv)
    start = time.time()
    rec = {"ts": start, "move": move, "actions": []}
    try:
        open_ = bool(adv.exchange_status().get("trading_active"))
    except Exception as e:
        open_ = False
        rec["status_error"] = str(e)[:120]
    health = check_health(start - 600)
    far = lambda o: markets.closes_in(o["ticker"]) > LIMITS["close_margin_s"]     # noqa: E731
    orders = [o for o in bot_orders(bot) if far(o)] if open_ else []
    budget = Budget(used)
    if not open_:
        rec["skipped"] = "the exchange is not trading"
    elif health["status_age_s"] > 120:
        rec["skipped"] = "the demo bot is not running (no fresh status)"
    elif not orders:
        rec["skipped"] = "the bot has no resting orders far enough from a market's close"
    if rec.get("skipped"):
        log.info("round skipped: %s", rec["skipped"])
        return rec
    allowed = {m: w for m, w in MOVES.items() if budget.any or m not in TAKES}
    move = move or rng.choices(list(allowed), weights=list(allowed.values()))[0]
    if move in TAKES and move != "chase":
        tops = at_top(adv, orders)
        if tops:
            orders = tops
        else:                                     # every bot order has a better price beside it: rest instead
            rec["note"] = f"{move}: none of the bot's orders is the best price on its side"
            move = rng.choice(["wall", "undercut"])
    rec["move"] = move
    try:
        rec["actions"] = play_move(move, adv, bot, orders, budget, rng, sleep=sleep, far=far)
    finally:
        try:
            rec["cancelled_own"] = cancel_mine(adv)
        except Exception as e:
            rec["cancel_error"] = str(e)[:160]
    sleep(LIMITS["settle_s"])                     # the bot's fill poll and account reads catch up
    rec["checks"] = {"booked": check_booked(bot, start), "caps": check_caps(bot, markets),
                     "bot": check_health(start)}
    if not rec["checks"]["booked"]["ok"] and rec["checks"]["booked"].get("missing"):
        sleep(30)                                 # one more poll before calling a fill missing
        rec["checks"]["booked"] = check_booked(bot, start)
    rec["ok"] = all(c.get("ok") for c in rec["checks"].values())
    got = sum(a.get("got") or 0 for a in rec["actions"])
    log.info("round %s: took %g contracts; checks %s%s", move, got, "OK" if rec["ok"] else "FAILED",
             "" if rec["ok"] else ": " + json.dumps({k: v for k, v in rec["checks"].items() if not v.get("ok")})[:400])
    return rec


def topic():
    """The phone alert topic: WXBOT_NTFY_TOPIC from the environment, or that one line of the repo's .env (on the server
    it holds nothing else). Nothing else is read from it."""
    t = os.environ.get("WXBOT_NTFY_TOPIC", "").strip()
    if t:
        return t
    try:
        for line in open(os.path.join(ROOT, ".env")):
            k, _, v = line.strip().partition("=")
            if k.strip() == "WXBOT_NTFY_TOPIC":
                return v.strip().strip("'\"")
    except OSError:
        pass
    return ""


class Watch:
    """Phone alerts from the troublemaker, which also watches the demo bot from outside (the bot cannot report its own
    silence, and the demo bot sends no heartbeat): a failed check; the demo bot without a fresh status for two rounds
    in a row while Kalshi is trading (about ten minutes); three failed rounds in a row. Each at most every 3 hours."""

    def __init__(self, alerter=None):
        from wxbot.app.alerts import Alerter
        self.alerts = alerter or Alerter({"ntfy_topic": topic(), "min_gap_s": 3 * 3600}, demo=True)
        self.stale = self.failed = 0

    def after(self, rec=None, error=None):
        if error is not None:
            self.failed += 1
            if self.failed >= 3:
                self.alerts.send("rounds_failing", "wxbot DEMO troublemaker: its rounds keep failing",
                                 f"{self.failed} rounds in a row failed. Last error: {error!r}"[:300]
                                 + ". Ask Claude to look (data/demo/adversary.log).", tags="warning")
            return
        self.failed = 0
        if rec.get("skipped") == "the demo bot is not running (no fresh status)":
            self.stale += 1
            if self.stale >= 2:
                self.alerts.send("bot_down", "wxbot DEMO: the demo bot is not running",
                                 "No fresh status from the demo bot for about 10 minutes while Kalshi is trading. "
                                 "Ask Claude to look (systemctl status wxbot-demo).", priority="high",
                                 tags="warning")
            return
        self.stale = 0
        bad = [k for k, v in (rec.get("checks") or {}).items() if not v.get("ok")]
        if bad:
            c = rec["checks"]
            why = {"booked": lambda v: f"fills Kalshi shows that the bot did not record: {len(v.get('missing') or [])}",
                   "caps": lambda v: "; ".join(v.get("over") or [])[:160] or "over a cap",
                   "bot": lambda v: "halted: " + v.get("halt_reason", "") if v.get("halted")
                   else f"status {v.get('status_age_s')} s old"}
            msg = "; ".join(f"{k}: {why.get(k, lambda v: 'failed')(c[k])}" for k in bad)
            self.alerts.send("check:" + ",".join(bad), "wxbot DEMO troublemaker: a check failed",
                             f"After a {rec.get('move')} round: {msg}. Ask Claude to look "
                             "(scripts/adversary.py report).", priority="high", tags="warning")


def write_journal(rec):
    os.makedirs(DATA, exist_ok=True)
    with open(path("adversary.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def run(adv, bot, rng=None, sleep=time.sleep, watch=None):
    rng = rng or random.Random()
    watch = watch or Watch()
    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *a: stop.__setitem__("now", True))
    markets = Markets(adv)
    log.info("troublemaker started (demo exchange, fake money; phone alerts %s)",
             "on" if watch.alerts.enabled else "off: no WXBOT_NTFY_TOPIC")
    try:
        while not stop["now"] and not os.path.exists(path("ADVERSARY_STOP")):
            try:
                rec = play(adv, bot, rng=rng, sleep=sleep, markets=markets)
                write_journal(rec)
                watch.after(rec)
                wait = 300 if rec.get("skipped") else rng.uniform(15 * 60, 35 * 60)
            except Exception as e:                     # a failed round never ends the run
                log.warning("round failed: %r", e)
                watch.after(error=e)
                wait = 300
            t_end = time.time() + wait
            while time.time() < t_end and not stop["now"] and not os.path.exists(path("ADVERSARY_STOP")):
                sleep(5)
    finally:
        try:
            n = cancel_mine(adv)
            log.info("troublemaker stopped (%d of its orders cancelled)", n)
        except Exception as e:
            log.warning("troublemaker stopped; cancelling its orders failed (%r): they expire within 10 minutes", e)


def report(rows=None):
    rows = journal_rows() if rows is None else rows
    played = [r for r in rows if not r.get("skipped")]
    moves = Counter(r.get("move") for r in played)
    takes = [a for r in played for a in r.get("actions", []) if a.get("act") == "take"]
    got = sum(a.get("got") or 0 for a in takes)
    print(f"{len(rows)} rounds ({len(rows) - len(played)} skipped): " + ", ".join(f"{m} {k}" for m, k in moves.items()))
    print(f"took {got:g} contracts in {len([a for a in takes if a.get('got')])} fills; "
          f"{len([a for a in takes if a.get('error')])} takes refused or failed")
    n, usd, _ = used_today(rows)
    print(f"today (ET): {n:g} contracts, ${usd:.2f} of fake money (limits {LIMITS['day_contracts']:g} and "
          f"${LIMITS['day_usd']:g})")
    bad = [r for r in played if r.get("checks") and not r.get("ok")]
    for r in bad:
        when = dt.datetime.fromtimestamp(r["ts"]).strftime("%m-%d %H:%M")
        for k, v in r["checks"].items():
            if not v.get("ok"):
                detail = json.dumps({x: y for x, y in v.items() if x != "ok"})[:300]
                print(f"  FAILED {when} {r['move']} {k}: {detail}")
    print("every check passed" if not bad else f"{len(bad)} rounds with a failed check")
    return not bad


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    what = argv[0] if argv else ""
    os.makedirs(DATA, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(path("adversary.log"))])
    if what == "report":
        return 0 if report() else 1
    if what not in ("once", "run"):
        print(__doc__)
        return 2
    adv, bot = clients()
    if what == "run":
        run(adv, bot)
        return 0
    move = argv[1] if len(argv) > 1 else None
    if move and move not in MOVES:
        print(f"unknown move {move}: {', '.join(MOVES)}")
        return 2
    rec = play(adv, bot, move=move)
    write_journal(rec)
    Watch().after(rec)
    return 0 if rec.get("skipped") or rec.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
