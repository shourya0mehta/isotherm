"""Exposure accounting and hard caps.

Brackets inside an event are mutually exclusive, so risk is evaluated per event as the worst case over which
bracket wins. Held positions count exactly. Resting orders count by adverse selection (Oct 4 review, finding 2.3):
in each outcome, every resting order that would lose there is assumed filled and none that would win, so a resting
order's possible gain never offsets anything. (Until Oct 4 resting orders were assumed to fill all together, which let
the gain of one offset the loss of another: two 60c NO bids on brackets of one event counted $4 of worst case
instead of $12.)
Threshold ladders ("above k" markets, the hourly index events) are not exclusive: several strikes can settle YES
together. Pass a ThresholdLadder instead of the ticker list and the worst case runs over the intervals between
strikes instead (the same scenarios as wxbot/strategy/sleeve_h.py).
An event whose structure is unknown (no ticker list) or a ticker missing from the list counts its full cost as lost:
until Oct 4 an empty list made NO positions look riskless.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..strategy.types import DesiredOrder, Position


@dataclass(frozen=True)
class ThresholdLadder:
    """The 'above k' markets of one event as ((strike, ticker), ...). Outcome j: the value ends above the j lowest
    strikes and at or below the rest, so exactly those j markets settle YES (j = 0..n)."""
    strikes: tuple

    @classmethod
    def from_markets(cls, markets) -> "ThresholdLadder":
        return cls(tuple(sorted((float(m["floor_strike"]), m["ticker"]) for m in markets
                                if m.get("strike_type") == "greater" and m.get("floor_strike") is not None)))

    def outcomes(self) -> list:
        ks = sorted(self.strikes)
        return [{t for _, t in ks[:j]} for j in range(len(ks) + 1)]

    def __iter__(self):   # behaves like the ticker list where one is expected
        return iter(t for _, t in sorted(self.strikes))

    def __len__(self):
        return len(self.strikes)


def outcome_sets(spec) -> list:
    """YES-ticker sets of every outcome: one per bracket plus 'none' for an exclusive ladder (a ticker list), one per
    interval for a ThresholdLadder."""
    if isinstance(spec, ThresholdLadder):
        return spec.outcomes()
    return [{t} for t in spec] + [set()]


@dataclass
class RiskConfig:
    max_loss_per_bracket_usd: float = 30.0
    max_loss_per_event_usd: float = 60.0
    max_total_open_risk_usd: float = 300.0
    decided_max_cost_per_event_usd: float = 25.0
    daily_loss_stop_usd: float = 40.0
    total_drawdown_stop_usd: float = 90.0
    # cash: without collateral netting Kalshi locks the full cost of every position until settlement, so NO
    # positions on several brackets of one event lock far more cash than their worst-case loss
    max_cash_deployed_usd: float = 560.0
    min_cash_reserve_usd: float = 25.0
    assume_netting: bool = False  # paper only; live reads the account's netting setting


def event_worst_case(positions: list[Position], pending: list[DesiredOrder], tickers) -> float:
    """Worst-case loss (positive number) across outcomes for one event: held positions exactly, pending orders by
    adverse selection (each counts only where it loses). Unknown/other outcome included.
    tickers: the event's bracket tickers (exclusive outcomes) or a ThresholdLadder; None or empty = unknown."""
    known = set(iter(tickers)) if tickers else set()
    yes = defaultdict(float)
    no = defaultdict(float)
    cost = 0.0
    for p in positions:
        cost += p.yes_cost + p.no_cost
        if p.ticker in known:
            yes[p.ticker] += p.yes_qty
            no[p.ticker] += p.no_qty
    unknown_pending = sum(o.count * o.unit_cost for o in pending if o.ticker not in known)
    pend = [o for o in pending if o.ticker in known]
    worst = 0.0
    for won in (outcome_sets(tickers) if known else [set()]):
        loss = cost - sum(q for t, q in yes.items() if t in won) - sum(q for t, q in no.items() if t not in won)
        for o in pend:
            wins = (o.ticker in won) == (o.side == "bid")
            if not wins:
                loss += o.count * o.unit_cost
        worst = max(worst, loss + unknown_pending)
    return worst


def bracket_cost(positions: list[Position], pending: list[DesiredOrder], ticker: str) -> float:
    c = sum(p.yes_cost + p.no_cost for p in positions if p.ticker == ticker)
    c += sum(o.count * o.unit_cost for o in pending if o.ticker == ticker)
    return c


@dataclass
class RiskState:
    cfg: RiskConfig
    realized_today: float = 0.0
    realized_total: float = 0.0
    peak_equity: float = 0.0
    equity: float = 0.0
    halted: bool = False
    halt_reason: str = ""
    halt_kind: str = ""  # 'daily' (auto-resumes next ET day) | 'pause' (auto-resumes at resume_ts) | 'hard'
    halt_day: str = ""
    resume_ts: float = 0.0
    notes: list = field(default_factory=list)

    def check_stops(self, today: str = ""):
        if self.peak_equity - self.equity >= self.cfg.total_drawdown_stop_usd:
            self.halted, self.halt_kind = True, "hard"
            self.halt_reason = f"drawdown stop hit ({self.peak_equity - self.equity:.2f}) - manual review needed"
        elif self.realized_today <= -self.cfg.daily_loss_stop_usd:
            self.halted, self.halt_kind, self.halt_day = True, "daily", today
            self.halt_reason = f"daily loss stop hit ({self.realized_today:.2f})"
        return self.halted

    def maybe_resume(self, now: float, today: str) -> bool:
        if not self.halted:
            return False
        if self.halt_kind == "daily" and today and today != self.halt_day:
            self.halted, self.halt_reason, self.halt_kind = False, "", ""
            return True
        if self.halt_kind == "pause" and now >= self.resume_ts:
            self.halted, self.halt_reason, self.halt_kind = False, "", ""
            return True
        return False


def cash_in_use(positions: list[Position], tickers_by_event: dict, netting: bool) -> float:
    """Cash locked by held positions: full cost without netting, per-event worst case with MECNET netting."""
    if not netting:
        return sum(p.yes_cost + p.no_cost + p.fees for p in positions)
    by_ev = defaultdict(list)
    for p in positions:
        by_ev[p.event].append(p)
    return sum(event_worst_case(ps, [], tickers_by_event.get(e, [])) for e, ps in by_ev.items())


def filter_orders(desired: list[DesiredOrder], positions_by_event: dict, tickers_by_event: dict,
                  cfg: RiskConfig, total_open_risk: float, cash_used: float = 0.0,
                  cash_cap: float | None = None, netting: bool = False, base_pending: dict | None = None,
                  incumbents: set | None = None, rank: dict | None = None,
                  keep: dict | None = None) -> tuple[list[DesiredOrder], list[str]]:
    """Greedy: keep orders while all caps hold, shrinking an order until it fits. Returns (kept, reasons_dropped).
    Resting orders count by adverse selection (see event_worst_case). cash_cap limits cash locked by positions and
    orders.
    base_pending: {event: [orders]} the exchange may still fill that `desired` does not stand for (live: unconfirmed
    cancels, unanswered placements, orders being replaced). They count in every cap first and are never dropped here;
    the caller includes them in total_open_risk and cash_used.
    incumbents: {(ticker, side, price_cents)} already resting: considered first (then by edge), so a resting order is
    not bumped by a new one and keeps its place in line.
    rank: {(ticker, side, price_cents): number} considered before everything else, lowest first (Sleeve A's contracts
    ahead, Oct 7); orders without one count as 0.
    keep: {(ticker, side, price_cents): (contracts resting, deadband)} for the resting orders of sleeves that keep them
    (sleeve_a.keep_resting, Oct 9): see _filter_keeping. None or empty: exactly the greedy pass below."""
    if keep:
        return _filter_keeping(desired, positions_by_event, tickers_by_event, cfg, total_open_risk, cash_used,
                               cash_cap, netting, base_pending, incumbents, rank, keep)
    kept: list[DesiredOrder] = []
    dropped: list[str] = []
    by_event_pending: dict = defaultdict(list)
    for e, v in (base_pending or {}).items():
        by_event_pending[e] = list(v)
    risk_used = total_open_risk
    cash = cash_used
    inc = incumbents or set()
    rk = rank or {}
    for o in sorted(desired, key=lambda x: (rk.get((x.ticker, x.side, x.price_cents), 0.0),
                                            0 if (x.ticker, x.side, x.price_cents) in inc else 1,
                                            -x.edge_per_contract)):
        if o.sleeve == "B":
            # Sleeve B (1c cheap-side reward quotes): worst case = its cost; it still counts against the cash cap
            c = o.count * o.unit_cost
            if cash_cap is not None and cash + c > cash_cap:
                dropped.append(f"{o.ticker}/{o.side}: cash")
                continue
            cash += c
            kept.append(o)
            continue
        ev = o.event
        pos = positions_by_event.get(ev, [])
        pend = by_event_pending[ev]
        tick = tickers_by_event.get(ev, [])
        wc_before = event_worst_case(pos, pend, tick)
        # shrink the order until it fits the caps
        count = o.count
        while count >= 1:
            trial = DesiredOrder(**{**o.__dict__, "count": count})
            bc = bracket_cost(pos, pend + [trial], o.ticker)
            wc = event_worst_case(pos, pend + [trial], tick)
            add_cash = (wc - wc_before) if netting else trial.count * trial.unit_cost
            if (bc <= cfg.max_loss_per_bracket_usd and wc <= cfg.max_loss_per_event_usd
                    and risk_used + (wc - wc_before) <= cfg.max_total_open_risk_usd
                    and (cash_cap is None or cash + add_cash <= cash_cap)):
                break
            count = int(count * 0.7) if count > 3 else count - 1
        if count < 1:
            dropped.append(f"{o.ticker}/{o.side}: caps")
            continue
        trial = DesiredOrder(**{**o.__dict__, "count": float(count)})
        pend.append(trial)
        d_wc = event_worst_case(pos, pend, tick) - wc_before
        risk_used += d_wc
        cash += d_wc if netting else trial.count * trial.unit_cost
        kept.append(trial)
    return kept, dropped


def _filter_keeping(desired, positions_by_event, tickers_by_event, cfg, total_open_risk, cash_used, cash_cap, netting,
                    base_pending, incumbents, rank, keep):
    """filter_orders when resting orders are kept (sleeve_a.keep_resting). The Oct 8 live log showed the greedy pass
    cutting a resting order to 1 contract when the caps bound and the ranking moved, and growing it back later (159
    times in 16 hours, up to 11 on one quote): each growth is a new place at the back of the queue and two writes.
    Same sort as filter_orders (the caller's rank puts a far more reachable new quote before a resting one). Then:
      1. every order up to the last resting one in that order: a resting order is tried at the size it rests at (or
         at the size wanted, when that is smaller by more than its deadband) and shrunk only as far as the caps need
         (the largest size that fits, not the greedy pass's 30% steps); a new order sorted among them as usual;
      2. resting orders that want more than they rest, in the same order: grown only when the new size is more than
         twice what rests and more than the deadband above it (the order manager grows an order only when less than
         half its wanted size rests, so a smaller growth would hold budget it never uses);
      3. the remaining new orders as usual.
    Every cap holds exactly as in filter_orders; only which orders get the room, and in what size, changes."""
    import math
    kept: list[DesiredOrder] = []
    dropped: list[str] = []
    by_event_pending: dict = defaultdict(list)
    for e, v in (base_pending or {}).items():
        by_event_pending[e] = list(v)
    state = {"risk": total_open_risk, "cash": cash_used}
    inc = incumbents or set()
    rk = rank or {}
    key_of = lambda x: (x.ticker, x.side, x.price_cents)
    order = sorted(desired, key=lambda x: (rk.get(key_of(x), 0.0), 0 if key_of(x) in inc else 1, -x.edge_per_contract))
    last = max((i for i, o in enumerate(order) if key_of(o) in keep), default=-1)

    def fits(o, count, pos, pend, tick, wc_before):
        trial = DesiredOrder(**{**o.__dict__, "count": count})
        bc = bracket_cost(pos, pend + [trial], o.ticker)
        wc = event_worst_case(pos, pend + [trial], tick)
        add_cash = (wc - wc_before) if netting else trial.count * trial.unit_cost
        return (bc <= cfg.max_loss_per_bracket_usd and wc <= cfg.max_loss_per_event_usd
                and state["risk"] + (wc - wc_before) <= cfg.max_total_open_risk_usd
                and (cash_cap is None or state["cash"] + add_cash <= cash_cap))

    def largest_fit(o, hi, pos, pend, tick, wc_before) -> float:
        """hi itself if it fits (a fractional resting size included), else the largest whole count below it that
        fits (every cap grows with the count, so a binary search finds it), else 0."""
        if hi > 0 and fits(o, float(hi), pos, pend, tick, wc_before):
            return float(hi)
        lo, top, best = 1, int(math.ceil(hi)) - 1, 0
        while lo <= top:
            mid = (lo + top) // 2
            if fits(o, float(mid), pos, pend, tick, wc_before):
                best, lo = mid, mid + 1
            else:
                top = mid - 1
        return float(best)

    def add(o, count, pend, pos, tick, wc_before):
        trial = DesiredOrder(**{**o.__dict__, "count": float(count)})
        pend.append(trial)
        d_wc = event_worst_case(pos, pend, tick) - wc_before
        state["risk"] += d_wc
        state["cash"] += d_wc if netting else trial.count * trial.unit_cost
        return trial

    def as_usual(o):
        """filter_orders' greedy step for one order (sleeve B's cash-only rule included)."""
        if o.sleeve == "B":
            c = o.count * o.unit_cost
            if cash_cap is not None and state["cash"] + c > cash_cap:
                dropped.append(f"{o.ticker}/{o.side}: cash")
                return
            state["cash"] += c
            kept.append(o)
            return
        pos, pend, tick = positions_by_event.get(o.event, []), by_event_pending[o.event], tickers_by_event.get(o.event, [])
        wc_before = event_worst_case(pos, pend, tick)
        count = o.count
        while count >= 1:
            if fits(o, count, pos, pend, tick, wc_before):
                break
            count = int(count * 0.7) if count > 3 else count - 1
        if count < 1:
            dropped.append(f"{o.ticker}/{o.side}: caps")
            return
        kept.append(add(o, count, pend, pos, tick, wc_before))

    grow = []                       # (index in kept, the order as wanted) for pass 2
    for o in order[:last + 1]:      # pass 1
        k = key_of(o)
        if k not in keep or o.sleeve == "B":
            as_usual(o)
            continue
        r, band = float(keep[k][0]), float(keep[k][1])
        want = float(o.count)
        target = want if (want < r and r - want > band) else r
        pos, pend, tick = positions_by_event.get(o.event, []), by_event_pending[o.event], tickers_by_event.get(o.event, [])
        wc_before = event_worst_case(pos, pend, tick)
        count = largest_fit(o, target, pos, pend, tick, wc_before)
        if count <= 0:
            dropped.append(f"{o.ticker}/{o.side}: caps")
            continue
        kept.append(add(o, count, pend, pos, tick, wc_before))
        if want > count:
            grow.append((len(kept) - 1, o, r, band))
    for i, o, r, band in grow:      # pass 2
        cur = kept[i]
        pos, pend, tick = positions_by_event.get(o.event, []), by_event_pending[o.event], tickers_by_event.get(o.event, [])
        wc_before = event_worst_case(pos, pend, tick)
        extra = largest_fit(o, float(o.count) - cur.count, pos, pend, tick, wc_before)
        new = cur.count + extra
        if extra > 0 and new > 2 * r and new - r > band:
            add(o, extra, pend, pos, tick, wc_before)
            kept[i] = DesiredOrder(**{**cur.__dict__, "count": float(new)})
    for o in order[last + 1:]:      # pass 3
        as_usual(o)
    return kept, dropped


def pending_cash(pending: list[DesiredOrder]) -> float:
    """Cash a list of resting orders holds (each order's full cost)."""
    return sum(o.count * o.unit_cost for o in pending)
