"""Live order manager: reconciles desired quotes with resting orders on Kalshi (V2 endpoints).

Rules
- Desired state is one price and a target resting size per (ticker, side).
- Price change -> cancel the key's orders and place one fresh order once the cancel is confirmed.
- Size above target -> amend the newest order down at the same price (keeps queue priority).
- Size well below target (under half of it rests):
  - what rests is mostly filled (each order at least half filled, judged on the largest size it ever had) -> add a
    top-up order beside it (the older order keeps its priority);
  - otherwise -> grow the newest order with one amend. Kalshi moves a grown order to the back of its price's queue,
    where a second order would sit anyway, and a market and side never carries a second order beside one that is
    not mostly filled (the Stage 2 gate's duplicate check; found by the two-day demo test, Oct 6).
- Every amend names an expiry: a shrink keeps the order's own, a growth pushes it forward. No amend can leave an
  order without its dead-man expiry, whatever Kalshi does with an amend that names none.
- Anything resting with our client-id prefix that is not desired gets cancelled (batch cancel).
- Every order is post-only, GTC with an expiry, cancel_order_on_pause, exchange_index 0.
- Dead-man expiry: orders expire a sleeve's TTL after the last refresh, never later than market close. The bot pushes
  the expiry forward with expiry-only amends (same price and size, which keeps queue priority). If the bot, its host,
  or its network dies, every order is gone within the TTL.
- Manual orders (no prefix) are never touched.

Order states (Oct 4 review, findings 2.1 and 2.2). The exchange is the source of truth, but its account reads lag the
matching engine (GET /exchange/user_data_timestamp says by how much), and a write whose answer is lost may or may not
have happened. So every order the bot may have on the book is tracked in one of three states:
- resting: acknowledged by the exchange (our own write's answer or a snapshot).
- cancel_pending: we asked for a cancel and the exchange did not confirm it. It still counts as exposure, its key gets
  no new order, and the cancel is retried until confirmed or a newer snapshot shows the order gone.
- uncertain: a placement whose answer never came (timeout, 5xx, or a success answer that could not be read), or an
  order an amend was answered 404 for (gone, most likely filled, but its fill may not be in the positions read yet).
  The order may rest; it counts as exposure and blocks its key until a snapshot shows it (adopted) or two snapshots
  newer than the attempt do not.
A cancel that fails for the first time stops new placements for that pass only (the risk check that admitted them may
have counted the order as going away); afterwards the pending cancel counts in full and only blocks its own key.
sync() merges snapshots instead of replacing the local book: a local order missing from snapshots is dropped only when
two snapshots newer than what we know about the order agree it is gone, and a cancel we saw confirmed is not brought
back by an older snapshot.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import requests

from ..kalshi.rest import KalshiError, KalshiREST, str_to_cents
from ..strategy.types import DesiredOrder, Fill
from ..timeutil import parse_ts

log = logging.getLogger(__name__)
PREFIX = "wxb-"
RESTING, CANCEL_PENDING, UNCERTAIN = "resting", "cancel_pending", "uncertain"


@dataclass
class LiveOrder:
    order_id: str
    client_order_id: str
    ticker: str
    side: str
    price_cents: int
    remaining: float
    filled: float
    created_ts: float = 0.0
    expires_ts: float = 0.0
    state: str = RESTING
    sleeve: str = ""
    ack_ts: float = 0.0       # when our own write last told us this order's state (newer than a lagged snapshot)
    cancel_ts: float = 0.0    # last cancel attempt
    misses: int = 0           # consecutive newer snapshots that did not show it (dropped at two)
    pending_since: float = 0.0  # when it last became uncertain or cancel_pending (logged when that resolves)
    initial: float = 0.0      # the largest size (filled + resting) it is known to have had: Kalshi's initial count,
    #                           or our own placement or growth if larger (judges "mostly filled" for a top-up)

    @property
    def key(self) -> tuple:
        return (self.ticker, self.side)

    @property
    def unit_cost(self) -> float:
        return self.price_cents / 100.0 if self.side == "bid" else (100 - self.price_cents) / 100.0


def mostly_filled(o: LiveOrder) -> bool:
    """At least half of the largest size the order ever had has filled, so a top-up may go out beside it. The Stage 2
    gate judges a top-up on Kalshi's initial count of the earlier order and its fills up to the top-up's creation; the
    bot's sizes are never smaller than Kalshi's and its fill counts never ahead of them, so it is never more lenient."""
    size = max(o.initial, o.filled + o.remaining)
    return size > 0 and o.filled >= 0.5 * size


def sleeve_of(client_order_id: str) -> str:
    """The sleeve letter encoded in our client order ids (wxb-<S>-<hex>); '' for the older wxb-<hex> form."""
    c = client_order_id or ""
    if c.startswith(PREFIX) and len(c) > len(PREFIX) + 1 and c[len(PREFIX) + 1] == "-":
        return c[len(PREFIX)]
    return ""


def is_patient(d, resting, now: float, patient_s: float, sleeves=("A", "T"), ts_attr: str = "created_ts") -> bool:
    """True when the resting orders of a key all sit at a more cautious price than the one now wanted (a lower YES
    bid, a higher YES ask) and all are younger than patient_s: moving them now would only trade a place in line for a
    more aggressive price, so they stay (the daily replays re-priced every 5 minutes). Moves to a more cautious price
    are never delayed. Off when patient_s is 0 or the sleeve is not listed."""
    if not patient_s or not resting or getattr(d, "sleeve", None) not in sleeves:
        return False
    if d.side == "bid":
        cautious = all(o.price_cents < d.price_cents for o in resting)
    else:
        cautious = all(o.price_cents > d.price_cents for o in resting)
    if not cautious:
        return False
    ages = [now - getattr(o, ts_attr) if getattr(o, ts_attr, 0) else float("inf") for o in resting]
    return all(a < patient_s for a in ages)


def _iso_ts(s) -> float:
    if not s:
        return 0.0
    try:
        return parse_ts(s)
    except Exception:
        return 0.0


def _parse_order(o: dict) -> LiveOrder:
    side = o.get("book_side") or ("bid" if o.get("outcome_side") == "yes" else "ask")
    cid = o.get("client_order_id") or ""
    return LiveOrder(o["order_id"], cid, o["ticker"], side,
                     str_to_cents(o.get("yes_price_dollars")), float(o.get("remaining_count_fp") or 0),
                     float(o.get("fill_count_fp") or 0), _iso_ts(o.get("created_time")),
                     _iso_ts(o.get("expiration_time")), RESTING, sleeve_of(cid),
                     initial=float(o.get("initial_count_fp") or 0))


def reject_kind(e: Exception) -> str:
    """'post_only' (would have crossed: the book moved), 'no_funds', or 'error'."""
    body = (getattr(e, "body", "") or "").lower()
    st = getattr(e, "status", 0)
    if st in (400, 409) and ("post" in body or "cross" in body):
        return "post_only"
    if st in (400, 402, 403, 409) and ("insufficient" in body or "balance" in body):
        return "no_funds"
    return "error"


def definitely_refused(e: Exception) -> bool:
    """A 4xx answer to a request that had no earlier unanswered attempt: the exchange processed it and refused, so
    no order exists. Anything else (timeouts, 5xx, a 4xx after a lost first attempt, a success answer that could not
    be read) leaves the outcome unknown."""
    if isinstance(e, KalshiError):
        return 400 <= e.status < 500 and e.status != 429 and not getattr(e, "ambiguous", False)
    if isinstance(e, requests.RequestException):
        return False  # network trouble, or an unreadable body (requests' JSONDecodeError is also a ValueError)
    return isinstance(e, (ValueError, TypeError))   # rejected locally (price, side or count), before anything was sent


class LiveOrderManager:
    def __init__(self, rest: KalshiREST, max_writes_per_cycle: int = 40, dry_run: bool = False,
                 ttl_s: float = 1200.0, patient_s: float = 0.0, patient_sleeves=("A", "T"),
                 ttl_by_sleeve: Optional[dict] = None, post_only_cooldown_s: float = 10.0):
        self.rest = rest
        self.patient_s = float(patient_s or 0)
        self.patient_sleeves = tuple(patient_sleeves)
        self.ttl_s = float(ttl_s or 0)
        self.ttl_by_sleeve = {k: float(v) for k, v in (ttl_by_sleeve or {}).items()}
        self.max_writes = max_writes_per_cycle
        self.dry_run = dry_run
        self.orders: dict[tuple, list[LiveOrder]] = {}
        self.errors: list[float] = []
        self.last_fill_ts: int = int(time.time()) - 60
        self.seen_fill_ids: set = set()
        # order ids known to be ours (placed by this process or synced with our client_order_id prefix), and the
        # verdict for ids looked up once: fills of orders that are not ours (manual trades) are skipped
        self.my_order_ids: set = set()
        self.order_owner: dict = {}
        self.order_sleeve: dict = {}         # order id -> sleeve letter (from our client ids), for fill attribution
        self.unresolved_fills: dict = {}     # fill id -> fill whose owner lookup failed (retried each poll)
        self.unresolved_since: dict = {}
        self.owner_grace_s = 600.0
        self.gone: dict = {}                 # order id -> when its cancel was confirmed (or the exchange lost it)
        self.cooldown: dict = {}             # key -> no new order before this time (after a post-only rejection)
        self.filled_on_arrival = 0           # post-only orders Kalshi filled at once (post-only not holding)
        self.post_only_cooldown_s = float(post_only_cooldown_s)
        self.error_cooldown_s = 30.0         # a key whose order was refused for another reason
        self.refusal_stop = 3                # unexplained refusals in one pass before it stops placing (a trading
        #                                      pause refuses every order: the runner then re-reads the exchange status)
        self.cancel_retry_s = 1.0
        self.uncertain_margin_s = 3.0        # an uncertain placement is resolved only by a snapshot this much newer
        self.fallback_lag_s = 10.0           # assumed snapshot age when the user-data timestamp cannot be read
        self.last_sync_ts = 0.0              # local time of the last successful snapshot
        self.last_as_of = 0.0                # exchange time that snapshot reflects
        self.snapshot_cash = 0.0             # cash held by ALL resting orders in that snapshot (bot and manual): the
        #                                      equity term, consistent with the balance and positions read beside it

    # ---------------------------------------------------------------- helpers
    def ttl_for(self, sleeve: str) -> float:
        return self.ttl_by_sleeve.get(sleeve or "", self.ttl_s)

    def resting_list(self) -> list[LiveOrder]:
        """Every order that may be on the book: resting, cancel_pending and uncertain."""
        return [o for v in self.orders.values() for o in v]

    def unresolved(self) -> list[LiveOrder]:
        return [o for o in self.resting_list() if o.state != RESTING]

    def _drop(self, o: LiveOrder, gone: bool = False):
        lst = self.orders.get(o.key)
        if lst and o in lst:
            lst.remove(o)
            if not lst:
                self.orders.pop(o.key, None)
        if gone and o.order_id:
            self.gone[o.order_id] = time.time()

    def _err(self, e):
        self.errors.append(time.time())
        self.errors = [t for t in self.errors if t > time.time() - 60]
        self.last_error = str(e)[:300]
        log.error("order error: %s", e)

    def errors_last_minute(self) -> int:
        return len([t for t in self.errors if t > time.time() - 60])

    def _expiry(self, close_ts: Optional[int], now: float, sleeve: str = "") -> Optional[int]:
        ttl = self.ttl_for(sleeve)
        if not ttl:
            return close_ts
        e = int(now + ttl)
        return min(e, int(close_ts)) if close_ts else e

    # ---------------------------------------------------------------- sync
    def _as_of(self, now: float) -> float:
        """Exchange time the account reads reflect (fetched before the snapshot, so the snapshot is at least this
        fresh). Unknown -> assume it is fallback_lag_s old."""
        try:
            v = self.rest.user_data_timestamp()
            if isinstance(v, str):
                v = parse_ts(v)
            if v:
                return min(float(v), now)
        except Exception as e:
            log.debug("user data timestamp unavailable: %s", e)
        return now - self.fallback_lag_s

    def sync(self):
        """Merge the exchange's resting orders into the local book (see the module notes)."""
        now = time.time()
        as_of = self._as_of(now)
        snap = self.rest.orders(status="resting")
        by_oid, by_cid = {}, {}
        cash = 0.0
        for raw in snap:
            lo = _parse_order(raw)
            if lo.price_cents is not None:
                cash += lo.remaining * lo.unit_cost
            if not lo.client_order_id.startswith(PREFIX) and lo.order_id not in self.my_order_ids:
                continue  # never touch manual orders (ours are known by our client id prefix or by order id, in
                #           case an amend ever gave one of ours another client id)
            by_oid[lo.order_id] = lo
            by_cid[lo.client_order_id] = lo
        seen = set()
        for o in self.resting_list():
            s = by_oid.get(o.order_id) if o.order_id else None
            if s is None and o.client_order_id:
                s = by_cid.get(o.client_order_id)
            if s is not None:
                seen.add(s.order_id)
                o.misses = 0
                if o.order_id != s.order_id:   # unknown until now (uncertain), or changed by an amend
                    o.order_id = s.order_id
                if o.state == UNCERTAIN:
                    o.state = RESTING
                    log.warning("order %s (%s %s @%sc) is resting after all: adopted. RESOLVED uncertain adopted "
                                "after=%.1fs cid=%s", o.order_id, o.ticker, o.side, o.price_cents,
                                now - (o.pending_since or now), o.client_order_id)
                self.my_order_ids.add(o.order_id)
                self.order_sleeve.setdefault(o.order_id, o.sleeve or s.sleeve)
                o.initial = max(o.initial, s.initial)     # only ever larger: "mostly filled" never gets looser
                if as_of >= o.ack_ts:   # the snapshot is newer than our last write: its numbers win
                    o.remaining, o.filled = s.remaining, s.filled
                    o.expires_ts = s.expires_ts or o.expires_ts
                    o.created_ts = o.created_ts or s.created_ts
                continue
            margin = self.uncertain_margin_s if o.state == UNCERTAIN else 1.0
            if as_of >= o.ack_ts + margin:
                # filled, cancelled or expired (fills and positions are read separately). Two newer snapshots must
                # agree, because Kalshi calls its account-data timestamp approximate.
                o.misses += 1
                if o.misses >= 2:
                    if o.state in (UNCERTAIN, CANCEL_PENDING):
                        log.warning("%s order %s (%s %s @%sc) is in neither of two newer snapshots: dropped. RESOLVED "
                                    "%s dropped after=%.1fs cid=%s", o.state, o.order_id or "-", o.ticker, o.side,
                                    o.price_cents, o.state, now - (o.pending_since or now), o.client_order_id)
                    self._drop(o)
        for oid, s in by_oid.items():
            if oid in seen:
                continue
            g = self.gone.get(oid)
            if g is not None and as_of < g + 1.0:
                continue        # cancelled, and the snapshot predates the cancel
            self.gone.pop(oid, None)
            self.orders.setdefault(s.key, []).append(s)
            self.my_order_ids.add(oid)
            self.order_sleeve.setdefault(oid, s.sleeve)
        self.gone = {k: v for k, v in self.gone.items() if v > now - 900}
        self.last_sync_ts, self.last_as_of, self.snapshot_cash = now, as_of, cash

    # ---------------------------------------------------------------- cancels
    def _cancel(self, orders: list[LiveOrder]) -> tuple[int, int]:
        """Batch-cancel. Returns (confirmed, writes). A confirmed cancel leaves the books; anything else becomes
        cancel_pending (still exposure, retried) until confirmed or a newer snapshot shows the order gone."""
        now = time.time()
        todo = [o for o in orders if o.order_id and o.state != UNCERTAIN]
        retried = {id(o) for o in todo if o.state == CANCEL_PENDING}
        for o in todo:
            if o.state != CANCEL_PENDING:
                o.pending_since = now
            o.state, o.cancel_ts = CANCEL_PENDING, now
        if self.dry_run:
            for o in todo:
                o.state = "cancelled"
                self._drop(o, gone=True)
            return len(todo), 0
        done = writes = 0
        for i in range(0, len(todo), 20):
            chunk = todo[i:i + 20]
            writes += 1
            try:
                r = self.rest.batch_cancel([o.order_id for o in chunk]) or {}
            except Exception as e:
                self._err(e)
                log.error("batch cancel failed (%s); kept as pending exposure. PENDING cids=%s", e,
                          ",".join(o.client_order_id for o in chunk))
                continue
            res = {x.get("order_id"): x for x in (r.get("orders") or []) if isinstance(x, dict)}
            bad = 0
            for o in chunk:
                x = res.get(o.order_id)
                # confirmed: no error and a cancel time (Kalshi leaves ts_ms out when that cancel errored) or
                # contracts actually taken off (reduced_by is always present, zero when the cancel errored)
                if x is not None and not x.get("error") and (x.get("ts_ms") is not None
                                                           or float(x.get("reduced_by") or 0) > 0):
                    if id(o) in retried:
                        log.info("cancel of %s confirmed on a retry. RESOLVED cancel_pending confirmed after=%.1fs "
                                 "cid=%s", o.order_id, time.time() - (o.pending_since or now), o.client_order_id)
                    o.state = "cancelled"
                    self._drop(o, gone=True)
                    done += 1
                else:
                    bad += 1
            if bad:
                first = next((x.get("error") for x in res.values() if x.get("error")), "no per-order answer")
                log.error("batch cancel: %d of %d not confirmed (%s); kept as pending exposure. PENDING cids=%s", bad,
                          len(chunk), first, ",".join(o.client_order_id for o in chunk if o.state == CANCEL_PENDING))
        return done, writes

    def cancel_all(self, reason: str = "", rounds: int = 3) -> int:
        """Cancel every wxbot order and confirm it (a batch cancel answers 200 even when single cancels fail).
        Returns how many may still rest afterwards (-1: the exchange could not be read at all); anything left keeps
        counting as exposure, keeps its key blocked, and expires on its own within its TTL."""
        log.warning("cancel_all: %s", reason)
        if self.dry_run:
            self.orders = {}
            return 0
        left, synced = -1, False
        for attempt in range(rounds + 1):
            try:
                self.sync()
                synced = True
            except Exception as e:
                log.error("sync failed during cancel_all (attempt %d): %s", attempt + 1, e)
                if not synced:
                    time.sleep(min(1.0 + attempt, 3.0))
                    continue
            todo = [o for o in self.resting_list() if o.order_id and o.state != UNCERTAIN]
            if todo and attempt < rounds:
                self._cancel(todo)
            left = len(self.resting_list()) if synced else -1
            if left == 0:
                break
            time.sleep(min(0.5 * (attempt + 1), 2.0))
        if left != 0:
            log.error("cancel_all: %s wxbot orders may still be resting; they expire on their own within %.0f min",
                      "an unknown number of" if left < 0 else left, max([self.ttl_s] + list(self.ttl_by_sleeve.values())) / 60)
        return left

    # ---------------------------------------------------------------- placement
    def _place(self, d: DesiredOrder, count: float, close_ts: Optional[int]) -> Optional[LiveOrder]:
        """Place one post-only order. The intent is recorded before the request, so an answer that never comes
        leaves an 'uncertain' order (exposure that blocks the key) instead of nothing. Returns the order (resting or
        uncertain), or None when nothing was sent or the exchange refused it (self.last_reject says why)."""
        self.last_reject = None
        now = time.time()
        expiry = self._expiry(close_ts, now, d.sleeve)
        if expiry and expiry <= now + 5:
            return None  # the market closes now: the exchange would reject the expiry (and count it as an error)
        cid = f"{PREFIX}{(d.sleeve or 'x')[0]}-{uuid.uuid4().hex[:18]}"
        o = LiveOrder("", cid, d.ticker, d.side, d.price_cents, float(count), 0.0, now, float(expiry or 0),
                      UNCERTAIN, d.sleeve or "", now, pending_since=now, initial=float(count))
        self.orders.setdefault(d.key, []).append(o)
        if self.dry_run:
            o.order_id, o.state = "dry-" + uuid.uuid4().hex[:8], RESTING
            return o
        try:
            r = self.rest.create_order(d.ticker, d.side, d.price_cents, count, post_only=True, expiration_ts=expiry,
                                       client_order_id=cid)
        except Exception as e:
            if definitely_refused(e):
                self._drop(o)
                self.last_reject = reject_kind(e)
                if self.last_reject == "post_only":
                    self.cooldown[d.key] = time.time() + self.post_only_cooldown_s
                elif self.last_reject == "error":
                    self._err(e)
                    # an unexplained refusal: not this key again for a while (it would be refused every pass)
                    self.cooldown[d.key] = time.time() + self.error_cooldown_s
                return None
            o.ack_ts = time.time()
            self._err(e)
            log.warning("placement of %s %s @%sc x%g has no answer: kept as uncertain until a snapshot shows it "
                        "(cid=%s)", d.ticker, d.side, d.price_cents, count, cid)
            return o
        o.order_id = r.get("order_id", "") or ""
        o.state, o.ack_ts, o.misses = RESTING, time.time(), 0
        o.remaining = float(r.get("remaining_count") or count)
        o.filled = float(r.get("fill_count") or 0)
        if o.order_id:
            self.my_order_ids.add(o.order_id)
            self.order_sleeve[o.order_id] = o.sleeve
        else:
            o.state = UNCERTAIN    # an answer without an id: let the next snapshot settle it
            return o
        if r.get("remaining_count") is not None and float(r.get("remaining_count") or 0) <= 0:
            # accepted, but nothing rests. Kalshi's FIX docs report a post-only order that would cross as cancelled
            # on arrival (POST_ONLY_CROSS) rather than refused, so the REST answer may be a success with nothing
            # left: treat it like a post-only refusal (cool the market down) instead of tracking an empty order and
            # placing again every pass.
            self.last_reject = "post_only"
            self.cooldown[d.key] = time.time() + self.post_only_cooldown_s
            if o.filled > 0:
                # contracts filled at once (post-only not honored). The fill feed books them, but the risk caps see
                # them only once the account reads show the position (up to the freshness limit later): until two
                # snapshots newer than this answer settle it, the order stays counted at its filled size, like an
                # amend answered 404, and its market stays blocked (round-6 review, Oct 4)
                log.warning("post-only order %s %s @%sc filled %g on arrival: counted as exposure until the account "
                            "reads catch up", d.ticker, d.side, d.price_cents, o.filled)
                o.remaining = o.filled
                self.filled_on_arrival += 1
                self._vanished(o, "filled on arrival")
                return o
            o.remaining = 0.0
            self._drop(o, gone=True)
            return None
        return o

    def _grow(self, o: LiveOrder, d: DesiredOrder, more: float, close_ts: Optional[int]) -> str:
        """Grow a resting order by `more` contracts at its price with one amend, instead of placing a second order
        beside it (Oct 6 demo test). Kalshi moves a grown order to the back of its price's queue, where a second order
        would sit too; the amend also pushes the dead-man expiry forward. Returns the stats key for the attempt:
        'grown'; 'no_funds', 'rejected' or 'refused' (refused: the order stays as it was, and a refusal for another
        reason cools the market down like a refused placement); 'uncertain' (no answer: the larger size is counted
        until a snapshot newer than the attempt settles it, and the larger size is what the risk check admitted);
        'vanished' (answered 404: filled or gone meanwhile, counted until snapshots settle it); '' when nothing was
        sent (the market closes now)."""
        self.last_reject = None
        now = time.time()
        expiry = self._expiry(close_ts, now, o.sleeve or d.sleeve)
        if expiry and expiry <= now + 5:
            return ""
        old, new_rem = o.remaining, o.remaining + more
        total = o.filled + new_rem

        def grown():
            o.remaining, o.initial = new_rem, max(o.initial, total)
            o.expires_ts = float(expiry or o.expires_ts)
            o.ack_ts, o.misses = time.time(), 0

        if self.dry_run:
            grown()
            return "grown"
        try:
            a = self.rest.amend_order(o.order_id, d.ticker, d.side, d.price_cents, total, expiration_ts=expiry) or {}
        except Exception as e:
            if isinstance(e, KalshiError) and e.status == 404:
                self._vanished(o)
                return "vanished"
            if definitely_refused(e):
                self.last_reject = reject_kind(e)
                if self.last_reject == "no_funds":
                    return "no_funds"
                if self.last_reject == "post_only":
                    self.cooldown[d.key] = time.time() + self.post_only_cooldown_s
                    return "rejected"
                self._err(e)
                self.cooldown[d.key] = time.time() + self.error_cooldown_s
                return "refused"
            self._err(e)
            grown()     # it may have grown: count the larger size until a snapshot newer than this attempt says
            log.warning("growing %s %s @%sc from %g to %g has no answer: counted at %g until a snapshot settles it "
                        "(cid=%s)", d.ticker, d.side, d.price_cents, old, new_rem, new_rem, o.client_order_id)
            return "uncertain"
        self._amended(o, a)
        grown()
        log.info("grew %s %s @%sc from %g to %g resting: one order per market and side (cid=%s)", d.ticker, d.side,
                 d.price_cents, old, new_rem, o.client_order_id)
        return "grown"

    def _amended(self, o: LiveOrder, answer: dict):
        """An amend's answer may name a new order id: follow it (cancels go to the id that rests). It may also name
        a client id other than ours (Kalshi's docs do not say what an amend without updated_client_order_id does;
        the go-live order test checks it): follow that too, loudly, since a restart knows our orders by the prefix."""
        new_id = (answer or {}).get("order_id")
        if new_id and new_id != o.order_id:
            self.gone[o.order_id] = time.time()
            o.order_id = new_id
            self.my_order_ids.add(new_id)
            self.order_sleeve[new_id] = o.sleeve
        new_cid = (answer or {}).get("client_order_id")
        if new_cid and new_cid != o.client_order_id:
            log.warning("amend of %s changed its client id %s -> %s%s", o.order_id, o.client_order_id, new_cid,
                        "" if str(new_cid).startswith(PREFIX) else " (not ours by prefix: tracked by order id)")
            o.client_order_id = new_cid

    def refresh_expiries(self, expiry_by_ticker: dict, budget: int) -> int:
        """Expiry-only amends for resting orders inside the last half of their TTL. Returns writes used."""
        now = time.time()
        writes = refused = 0
        self.refresh_refusal_stop = False
        due = sorted((o for o in self.resting_list() if o.state == RESTING and o.expires_ts and o.order_id
                      and self.ttl_for(o.sleeve) and o.expires_ts - now < self.ttl_for(o.sleeve) / 2),
                     key=lambda o: o.expires_ts)
        for o in due:
            if writes >= budget:
                break
            close = expiry_by_ticker.get(o.ticker)
            new_exp = self._expiry(close, now, o.sleeve)
            if not new_exp or new_exp <= o.expires_ts + 30:
                continue  # already at market close
            try:
                if not self.dry_run:
                    a = self.rest.amend_order(o.order_id, o.ticker, o.side, o.price_cents, o.filled + o.remaining,
                                              expiration_ts=new_exp) or {}
                    self._amended(o, a)
                o.expires_ts = float(new_exp)
                o.ack_ts, o.misses = time.time(), 0
                writes += 1
            except KalshiError as e:
                writes += 1
                if e.status == 404:
                    self._vanished(o)  # filled or expired meanwhile: counted until a snapshot settles it
                else:
                    self._err(e)
                    if definitely_refused(e):
                        refused += 1
                        if refused >= self.refusal_stop:
                            # the exchange refuses every amend (a trading pause?): stop here; the runner re-reads
                            # the exchange status, and unrefreshed orders lapse on their own (round-8 review)
                            self.refresh_refusal_stop = True
                            break
            except Exception as e:
                self._err(e)
        return writes

    # ----------------------------------------------------------- reconcile
    def _vanished(self, o: LiveOrder, why: str = "answered 404 (filled or gone)"):
        """The exchange answered 404 for an order we track (filled, expired or cancelled meanwhile, most likely).
        It is not dropped yet: until a snapshot newer than this answer settles it, it counts as exposure (its fill
        may not be in the positions read yet) and blocks its key, like an unanswered placement; a snapshot that
        still shows it brings it back as resting."""
        o.state, o.ack_ts, o.misses = UNCERTAIN, time.time(), 0
        o.pending_since = o.ack_ts
        log.info("order %s (%s %s @%sc) %s: uncertain until snapshots settle it (cid=%s)",
                 o.order_id, o.ticker, o.side, o.price_cents, why, o.client_order_id)

    def reconcile(self, desired: list[DesiredOrder], expiry_by_ticker: dict, allow_new: bool = True,
                  keep_alive: bool = True) -> dict:
        """allow_new=False (e.g. account state stale, or available cash below the reserve): cancel and shrink only,
        never place. keep_alive=False (account reads failing): no expiry refreshes either, so every order lapses
        within its sleeve's TTL unless the reads come back (the dead-man rule applies to a blind bot too)."""
        writes = 0
        self.refresh_refusal_stop = False      # set by this pass's expiry refresh only (round-9 review)
        stats = {"placed": 0, "amended": 0, "cancelled": 0, "kept": 0, "rejected": 0, "no_funds": 0, "blocked": 0,
                 "cancel_unconfirmed": 0, "uncertain": 0, "cooldown": 0, "held_back": 0, "frozen": 0, "refused": 0,
                 "refusal_stop": 0, "grown": 0, "vanished": 0}
        want = {d.key: d for d in desired}
        now = time.time()
        freeze = False
        # 1) cancels: keys no longer wanted, price changes (unless patience keeps a cautious order), and retries of
        #    cancels the exchange did not confirm
        to_cancel, held = [], set()
        for k, lst in self.orders.items():
            d = want.get(k)
            for o in lst:
                if o.state == CANCEL_PENDING and now - o.cancel_ts >= self.cancel_retry_s:
                    to_cancel.append(o)
                elif o.state == RESTING and d is None:
                    to_cancel.append(o)
            if d is not None:
                stale = [o for o in lst if o.state == RESTING and o.price_cents != d.price_cents]
                if stale and is_patient(d, stale, now, self.patient_s, self.patient_sleeves):
                    held.add(k)
                elif stale:
                    to_cancel += stale
        if to_cancel:
            first_try = [o for o in to_cancel if o.state == RESTING]
            ok, w = self._cancel(to_cancel)
            writes += w
            stats["cancelled"] += ok
            stats["cancel_unconfirmed"] += len([o for o in to_cancel if o.state == CANCEL_PENDING])
            # an order that just failed to cancel is still on the book, and the risk filter that admitted this
            # pass's orders may have counted it as going away: nothing new goes out this pass. From the next pass
            # on, committed() counts it in full, so a cancel that keeps failing on retry no longer blocks other
            # markets (round-2 verification, Oct 4); its own key stays blocked and it lapses within its TTL
            # (expiry refreshes skip it)
            if any(o.state == CANCEL_PENDING for o in first_try):
                freeze = True
        # 2a) shrink first (resting orders larger than wanted), so new orders never go out beside an excess. Only
        #     acknowledged orders are amended; an unconfirmed cancel or unanswered placement beside them is counted
        #     in full by committed() already, and no longer stops the resting ones from shrinking. The excess comes
        #     off the newest orders first (older ones keep their place in line), across as many orders as it takes;
        #     whatever cannot be taken off this pass freezes new placements (round-3 verification)
        for k, d in want.items():
            if k in held:
                continue
            rest = [o for o in self.orders.get(k, []) if o.state == RESTING]
            excess = sum(o.remaining for o in rest) - d.count
            for o in sorted(rest, key=lambda x: x.created_ts, reverse=True):
                if excess <= 0.5:
                    break
                if writes >= self.max_writes:
                    break
                new_rem = max(o.remaining - excess, 0.0)
                if new_rem < 1:
                    ok, w = self._cancel([o])
                    writes += max(w, 1)
                    stats["cancelled"] += ok
                    if not ok:
                        stats["cancel_unconfirmed"] += 1
                        break
                    excess -= o.remaining
                    continue
                writes += 1
                try:
                    if not self.dry_run:
                        # named, unchanged: a size decrease keeps its place in line, and the order keeps its
                        # dead-man expiry whatever Kalshi does with an amend that names none (an order that expires
                        # within 30 s gets its refresh later in this pass)
                        keep_exp = int(o.expires_ts) if o.expires_ts and o.expires_ts > now + 30 else None
                        a = self.rest.amend_order(o.order_id, d.ticker, d.side, d.price_cents,
                                                  o.filled + new_rem, expiration_ts=keep_exp) or {}
                        self._amended(o, a)
                    excess -= o.remaining - new_rem
                    o.remaining = new_rem
                    o.ack_ts, o.misses = time.time(), 0
                    stats["amended"] += 1
                except KalshiError as e:
                    if e.status == 404:
                        self._vanished(o)   # gone already (most likely filled): no cancel to send
                        break
                    self._err(e)
                    ok, w = self._cancel([o])   # it may still be at its old size: cancel it instead
                    writes += w
                    stats["cancelled"] += ok
                    stats["cancel_unconfirmed"] += int(not ok)
                    break
                except Exception as e:
                    self._err(e)
                    ok, w = self._cancel([o])
                    writes += w
                    stats["cancelled"] += ok
                    stats["cancel_unconfirmed"] += int(not ok)
                    break
            if excess > 0.5:
                freeze = True     # an excess left in place this pass: place nothing beside it
        if freeze and allow_new:
            allow_new = False
            stats["frozen"] = 1
        # 2b) place / top up
        for k, d in want.items():
            if k in held:
                stats["kept"] += 1
                continue
            if writes >= self.max_writes:
                break
            lst = self.orders.get(k, [])
            if any(o.state != RESTING for o in lst):
                stats["blocked"] += 1   # an unconfirmed cancel or an unanswered placement: never stack on it
                continue
            have = sum(o.remaining for o in lst)
            if have > 0.5 and have >= 0.5 * d.count:
                stats["kept"] += 1
                continue
            if not allow_new:
                stats["kept" if have > 0.5 else "held_back"] += 1
                continue
            if now < self.cooldown.get(k, 0.0):
                stats["cooldown"] += 1
                continue
            left = [o for o in lst if o.remaining > 0]
            if left and not all(mostly_filled(o) for o in left):
                # more is wanted than twice what rests, and what rests is not mostly filled (the size wanted grew,
                # e.g. the risk check had room for 2 and now has room for 5): grow the newest order instead of
                # placing a second one beside it
                r = self._grow(max(left, key=lambda x: x.created_ts), d, d.count - have,
                               expiry_by_ticker.get(d.ticker))
                if r:
                    writes += 1
                    stats[r] += 1
                    if r == "refused" and stats["refused"] >= self.refusal_stop:
                        stats["refusal_stop"] = 1      # the exchange refuses everything: stop placing this pass
                        break
                continue
            o = self._place(d, d.count - (have if have > 0.5 else 0.0), expiry_by_ticker.get(d.ticker))
            if o is not None:
                writes += 1
                stats["placed" if o.state == RESTING else "uncertain"] += 1
            elif self.last_reject == "post_only":
                stats["rejected"] += 1
                writes += 1
            elif self.last_reject == "no_funds":
                stats["no_funds"] += 1
                writes += 1
            elif self.last_reject == "error":
                writes += 1
                stats["refused"] += 1
                if stats["refused"] >= self.refusal_stop:
                    stats["refusal_stop"] = 1      # the exchange refuses everything: stop placing this pass
                    break
        # 3) dead-man expiry refresh with whatever write budget is left (not while the account cannot be read)
        stats["refreshed"] = (self.refresh_expiries(expiry_by_ticker, max(self.max_writes - writes, 0))
                              if keep_alive else 0)
        if getattr(self, "refresh_refusal_stop", False):
            stats["refusal_stop"] = 1
        return stats

    def committed(self, desired: list[DesiredOrder]) -> list[DesiredOrder]:
        """Exposure the exchange may still fill that the desired orders do not stand for: orders at another price or
        on unwanted keys (being cancelled), unconfirmed cancels and unanswered placements. The risk filter counts
        these before admitting anything new."""
        want = {(d.ticker, d.side, d.price_cents): d.count for d in desired}
        out, matched = [], {}
        for o in self.resting_list():
            if o.remaining <= 0:
                continue
            k = (o.ticker, o.side, o.price_cents)
            if o.state == RESTING and k in want:
                matched.setdefault(k, []).append(o)
                continue
            out.append(DesiredOrder(o.ticker, o.side, o.price_cents, float(o.remaining), o.sleeve or "?", 0.0,
                                    reason=f"committed:{o.state}", event=o.ticker.rsplit("-", 1)[0]))
        # a resting order larger than what is now wanted at its price: the excess counts until the amend lands
        for k, lst in matched.items():
            excess = sum(o.remaining for o in lst) - want[k]
            if excess > 0.5:
                o = lst[0]
                out.append(DesiredOrder(o.ticker, o.side, o.price_cents, float(excess), o.sleeve or "?", 0.0,
                                        reason="committed:excess", event=o.ticker.rsplit("-", 1)[0]))
        return out

    # --------------------------------------------------------------- fills
    def owner_of(self, order_id) -> Optional[bool]:
        """True for orders this bot placed (known id, or a wxb- client_order_id when looked up once), False for
        others (manual trades in the same account), None when the lookup failed (retried later)."""
        if not order_id or order_id in self.my_order_ids:
            return True
        if order_id not in self.order_owner:
            try:
                o = self.rest.order(order_id)
                cid = str(o.get("client_order_id") or "")
                # the go-live order test's orders (wxb-smoke-...) are not the bot's trading: a fill of one stays out
                # of its P&L, as the Stage 2 gate leaves them out of Kalshi's side
                ours = cid.startswith(PREFIX) and not cid.startswith(PREFIX + "smoke")
                self.order_owner[order_id] = ours
                if ours:
                    self.order_sleeve.setdefault(order_id, sleeve_of(cid))
            except Exception as e:
                log.warning("order %s lookup failed (%s): its fill waits for the next poll", order_id, e)
                return None
        return self.order_owner[order_id]

    def is_mine(self, order_id) -> bool:
        return bool(self.owner_of(order_id))

    def poll_fills(self) -> list[Fill]:
        out = []
        self.last_new_raw = []      # the raw fill objects behind this poll's new fills (stream shadow matching)
        try:
            # two minutes of overlap: a fill that becomes visible a little after a later one is still caught (the
            # seen ids drop the repeats)
            fills = list(self.rest.fills(min_ts=self.last_fill_ts - 120))
        except Exception as e:
            self._err(e)
            return out
        now = time.time()
        fills += [f for fid, f in self.unresolved_fills.items()
                  if fid not in {x.get("fill_id") or x.get("trade_id") for x in fills}]
        for f in fills:
            fid = f.get("fill_id") or f.get("trade_id")
            if fid in self.seen_fill_ids:
                continue
            owner = self.owner_of(f.get("order_id"))
            if owner is None:
                first = self.unresolved_since.setdefault(fid, now)
                if now - first < self.owner_grace_s:
                    self.unresolved_fills[fid] = f
                    continue
                log.warning("fill %s: owner unknown after %.0f min of failed lookups; counted as the bot's", fid,
                            self.owner_grace_s / 60)
                owner = True
            self.seen_fill_ids.add(fid)
            self.unresolved_fills.pop(fid, None)
            self.unresolved_since.pop(fid, None)
            if not owner:
                continue  # a manual trade in the same account: not the bot's position or P&L
            self.last_new_raw.append(f)
            side = f.get("book_side") or ("bid" if f.get("outcome_side") == "yes" else "ask")
            ts = _iso_ts(f.get("created_time")) or float(f.get("ts") or time.time())
            out.append(Fill(ts, f.get("market_ticker") or f.get("ticker"), "", side,
                            str_to_cents(f.get("yes_price_dollars")), float(f.get("count_fp") or 0),
                            self.order_sleeve.get(f.get("order_id")) or "?",
                            bool(f.get("is_taker")), float(f.get("fee_cost") or 0), str(fid or f.get("order_id", ""))))
            self.last_fill_ts = max(self.last_fill_ts, int(ts))
        return out
