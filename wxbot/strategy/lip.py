"""Kalshi Liquidity Incentive Program scoring (filed terms effective Jul 30, 2026).

Per side (YES bids, and NO bids priced in NO terms), each snapshot:
  - walk down from the best bid accumulating size; the Reference Price is the first level where cumulative size
    reaches Target/5; qualifying bids are all bids at levels visited until cumulative size reaches Target
    (whole levels included). If the side never reaches Target, nothing on that side qualifies.
  - score(bid) = size * DF ** max(ref_ticks - bid_ticks, 0); normalized within the side.
  - a snapshot is excluded unless BOTH sides have qualifying bids.
  - payout = (your snapshot scores summed) / (all scores summed) * period reward * (non-excluded / total).
Each side's normalized scores sum to 1, so a side is worth half the pool when both qualify.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SideScore:
    qualifies: bool
    ref_cents: Optional[int]
    total_weight: float
    our_weight: float

    @property
    def our_share(self) -> float:
        return self.our_weight / self.total_weight if self.total_weight > 0 else 0.0


def score_side(levels: list[tuple[int, float]], target: float, discount: float,
               ours: Optional[list[tuple[int, float]]] = None) -> SideScore:
    """levels: [(price_cents, size)] for ONE side (any order). ours: our orders on this side (already included
    in `levels` or not -- pass include_ours via merging before calling if they are resting)."""
    book: dict[int, float] = {}
    for p, q in levels:
        book[int(p)] = book.get(int(p), 0.0) + float(q)
    our_book: dict[int, float] = {}
    for p, q in ours or []:
        book[int(p)] = book.get(int(p), 0.0) + float(q)
        our_book[int(p)] = our_book.get(int(p), 0.0) + float(q)
    lv = sorted(book.items(), key=lambda x: -x[0])
    if sum(q for _, q in lv) < target:
        return SideScore(False, None, 0.0, 0.0)
    cum, ref = 0.0, None
    for p, q in lv:
        cum += q
        if cum >= target / 5:
            ref = p
            break
    cum, W, OW = 0.0, 0.0, 0.0
    for p, q in lv:
        if cum >= target:
            break
        mult = discount ** max(ref - p, 0)
        W += q * mult
        OW += our_book.get(p, 0.0) * mult
        cum += q
    return SideScore(True, ref, W, OW)


def snapshot_value(yes_levels, no_levels, target, discount, our_yes=None, our_no=None) -> dict:
    """Returns our expected reward fraction of the pool for this snapshot (0..1) and diagnostics."""
    sy = score_side(yes_levels, target, discount, our_yes)
    sn = score_side(no_levels, target, discount, our_no)
    counts = sy.qualifies and sn.qualifies
    frac = (sy.our_share + sn.our_share) / 2 if counts else 0.0
    return {"counts": counts, "frac": frac, "yes": sy, "no": sn}
