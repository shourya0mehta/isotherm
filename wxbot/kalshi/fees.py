"""Kalshi fee math (fee schedule effective Jul 7, 2026).

taker fee = roundup(M * 0.07   * C * P * (1-P))
maker fee = roundup(M * 0.0175 * C * P * (1-P))   only on series with fee_type 'quadratic_with_maker_fees'
Direct-member balances round to $0.0001 (fee + position cost rounded up to a centicent).
Weather series are all fee_type 'quadratic' with multiplier 1 -> makers pay nothing.
"""
from __future__ import annotations

import math


def _roundup_centicent(x: float) -> float:
    return math.ceil(round(x * 10000, 6)) / 10000.0


def taker_fee(price_cents: int, count: float, multiplier: float = 1.0) -> float:
    p = price_cents / 100.0
    return _roundup_centicent(multiplier * 0.07 * count * p * (1 - p))


def maker_fee(price_cents: int, count: float, fee_type: str = "quadratic", multiplier: float = 1.0) -> float:
    if "maker" not in (fee_type or ""):
        return 0.0
    p = price_cents / 100.0
    return _roundup_centicent(multiplier * 0.0175 * count * p * (1 - p))
