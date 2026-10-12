"""Same-weather budget (Oct 11, 2026; docs/results/r14_same_weather.txt, Q2, variant "chain, $20"; the replay's
research/ml/r14_lib.py group_filter).

Some cities' forecast misses move together. On two years of the forecaster's own errors, a same-direction miss of 2 F
or more in both cities of a pair happens 4-5 times as often as independence would give: Philadelphia/Trenton (error
correlation 0.65), Austin/San Antonio (0.63), Newark/New York (0.56), Newark/Trenton (0.53), New York/Trenton (0.50).
Linked into chains they make two groups: Philadelphia/Trenton/Newark/New York and Austin/San Antonio. One front can then
cost a bet in each of a group's cities on the same market day. The budget caps that: per group and market date, the
worst-case loss of the group's events, counted the way the live caps count it (positions plus resting orders by adverse
selection, wxbot/risk/limits.py event_worst_case), stays within budget_usd.
  - The near-certain maker (A) and the newer-city maker (T) share it (`sleeves`); their orders pass through it after
    each maker's own caps and before the account's caps (wxbot/app/runner.py Bot.execute), where group_filter sits in
    the replay.
  - Existing positions are never forced out: they and the orders the exchange may still fill that the makers no longer
    stand for (base_pending) count first; new or larger orders are shrunk or dropped (filter_orders' greedy pass with
    only this cap). Resting orders go first, as in the other caps (incumbents, and sleeve keep_resting when on); then
    the larger edge gets the room first.
  - Taker orders (X, E, R) do not pass through it, but their positions in a group's events count toward its budget, as
    every position in the account does.
In the Aug-Sep replay (both makers, Stage 4 caps): +$0.12 a day [-0.04, +0.39] against no budget, no gain proven;
adopted as a risk cap: the largest possible same-group loss on one market day fell from $29 to under $20.

Config, a top-level block (off when it is absent or enabled is false):
  same_weather:
    enabled: true
    budget_usd: 20
    sleeves: [A, T]
    groups:
      - [KXHIGHPHIL, KXHIGHTTTN, KXHIGHTEWR, KXHIGHNY]
      - [KXHIGHAUS, KXHIGHTSATX]
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .limits import RiskConfig, event_worst_case, filter_orders

STATIONS = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "stations.json")))
MAKERS = ("A", "T")


def series_of(event: str) -> str:
    return event.split("-")[0]


def date_of(event: str) -> str:
    """The market date of a daily event as group_filter keys it: KXHIGHNY-26OCT12 -> 26OCT12."""
    parts = event.split("-")
    return parts[1] if len(parts) > 1 else ""


def city(series: str) -> str:
    c = str((STATIONS.get(series) or {}).get("city") or series)
    return c[:-5] if c.endswith(" City") else c


@dataclass(frozen=True)
class SameWeatherConfig:
    enabled: bool = False
    budget_usd: float = 20.0
    sleeves: tuple = MAKERS
    groups: tuple = ()          # ((series, ...), ...): a series belongs to one group at most

    @classmethod
    def from_cfg(cls, d: Optional[dict]) -> "SameWeatherConfig":
        d = d or {}
        if not d.get("enabled", False):
            return cls()
        groups = tuple(tuple(str(s) for s in g) for g in (d.get("groups") or []) if g)
        sleeves = tuple(str(s).upper()[:1] for s in (d.get("sleeves") or MAKERS))
        budget = float(d.get("budget_usd", 20.0))
        every = [s for g in groups for s in g]
        if not groups:
            raise ValueError("same_weather.groups: at least one group of series")
        if len(every) != len(set(every)):
            raise ValueError("same_weather.groups: a series is in two groups")
        if budget <= 0:
            raise ValueError("same_weather.budget_usd: a positive dollar amount")
        if not sleeves or any(s not in MAKERS for s in sleeves):
            raise ValueError("same_weather.sleeves: the daily makers A and T (taker orders do not pass through the "
                             "budget; their positions count toward it)")
        return cls(True, budget, sleeves, groups)

    def group_of(self) -> dict:
        """series -> group number."""
        return {s: i for i, g in enumerate(self.groups) for s in g}

    def name(self, i: int) -> str:
        return "/".join(city(s) for s in self.groups[i])


@dataclass
class Pass:
    kept: list                                   # the orders, budgeted ones cut to fit, the others untouched
    usage: dict = field(default_factory=dict)    # (group, date) -> {"used": .., "positions": .., "cut": ..}
    refused: list = field(default_factory=list)  # (order as wanted, contracts kept, group, the group's worst case)


def apply(desired: list, pos_by_ev: dict, base: dict, tick_by_ev: dict, cfg: SameWeatherConfig,
          incumbents: Optional[set] = None, keep: Optional[dict] = None) -> Pass:
    """The budget over one set of orders. pos_by_ev, base and tick_by_ev as Bot.risk_context returns them (positions,
    committed orders the desired set does not stand for, and each event's outcome structure; base counts as fixed);
    incumbents and keep as the other caps get them. With no base, incumbents or keep this is group_filter."""
    gi = cfg.group_of()
    rest, cands = [], defaultdict(list)
    for o in desired:
        g = gi.get(series_of(o.event)) if o.sleeve in cfg.sleeves else None
        if g is None:
            rest.append(o)
        else:
            cands[(g, date_of(o.event))].append(o)
    out = Pass(list(rest))
    cap = RiskConfig(max_loss_per_bracket_usd=1e9, max_loss_per_event_usd=1e9, max_total_open_risk_usd=cfg.budget_usd)
    held_evs = [e for e in set(pos_by_ev) | set(base) if series_of(e) in gi]
    for key in sorted(set(cands) | {(gi[series_of(e)], date_of(e)) for e in held_evs}):
        g, day = key
        evs = {e for e in held_evs if gi[series_of(e)] == g and date_of(e) == day}
        pos = {e: pos_by_ev[e] for e in evs if pos_by_ev.get(e)}
        bas = {e: list(base[e]) for e in evs if base.get(e)}
        locked = sum(event_worst_case(pos.get(e, []), bas.get(e, []), tick_by_ev.get(e)) for e in evs)
        lst = cands.get(key, [])
        kept, _ = filter_orders(lst, pos, tick_by_ev, cap, locked, base_pending=bas, incumbents=incumbents,
                                keep=keep) if lst else ([], [])
        out.kept += kept
        mine = defaultdict(list)
        for o in kept:
            mine[o.event].append(o)
        used = sum(event_worst_case(pos.get(e, []), bas.get(e, []) + mine.get(e, []), tick_by_ev.get(e))
                   for e in evs | set(mine))
        got = defaultdict(float)
        for o in kept:
            got[(o.ticker, o.side, o.price_cents, o.sleeve)] += o.count
        cut = [(o, got.get((o.ticker, o.side, o.price_cents, o.sleeve), 0.0)) for o in lst]
        cut = [(o, n) for o, n in cut if n < o.count - 1e-9]
        out.refused += [(o, n, g, used) for o, n in cut]
        out.usage[key] = {"used": used, "positions": sum(event_worst_case(pos.get(e, []), [], tick_by_ev.get(e))
                                                         for e in evs), "cut": len(cut)}
    return out


def status(cfg: SameWeatherConfig, usage: dict) -> dict:
    """The status file's view: per group, its worst case by market date (dates with nothing held or wanted left out)."""
    groups = {}
    for i, g in enumerate(cfg.groups):
        days = {day: {"used_usd": round(u["used"], 2), "positions_usd": round(u["positions"], 2), "orders_cut": u["cut"]}
                for (gg, day), u in sorted(usage.items()) if gg == i}
        groups[cfg.name(i)] = {"series": list(g), "days": days}
    return {"budget_usd": cfg.budget_usd, "sleeves": list(cfg.sleeves), "groups": groups}
